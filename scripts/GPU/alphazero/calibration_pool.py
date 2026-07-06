"""Post-opening sharp-drop calibration pool (design Mechanism B).

A fixed set of external replay positions where the checkpoint (as black)
overvalued a losing position. Each becomes a value-only training sample whose
target is a soft negative (black perspective). The pool is sampled each train
step; the value-only MSE term is added to total_loss in alphazero_loss_batch.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .goal_line_trigger_probe_cases import position_state
from .position_probe_cases import load_csv_manifest
from .self_play import PositionRecord


def legal_moves_sha1(legal) -> str:
    """SHA-1 over the canonical legal-move ordering. Order-sensitive: pins the
    alignment between teacher_policy_json and legal_moves between build time and
    train time (legal_moves() is sorted/deterministic, so the same reconstructed
    position yields the same hash)."""
    canonical = ";".join(f"{r},{c}" for r, c in legal)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


RETENTION_POLICY_LOSS_MODES = frozenset({"teacher_retention", "mcts_root_retention"})
CONTINUATION_LOSS_MODE = "searched_continuation_retention"
# Modes whose pools use the teacher-mode (masked 14-tuple) loss path. The
# continuation mode is NOT in RETENTION_POLICY_LOSS_MODES: its rows carry a
# policy target only per-row (has_policy_target), not by mode.
TEACHER_MODE_LOSS_MODES = RETENTION_POLICY_LOSS_MODES | {CONTINUATION_LOSS_MODE}
GUARDRAIL_LOSS_MODE = "asymmetric_guardrail_retention"  # v12: value-only one-sided hinge
VALID_LOSS_MODES = frozenset({"hard_value", GUARDRAIL_LOSS_MODE}) | TEACHER_MODE_LOSS_MODES
# A manifest may mix at most these retention-mode combinations (v6 keeps the
# inert v5 root rows alongside the new continuation rows):
_ALLOWED_RETENTION_MODE_SETS = (
    frozenset(), frozenset({"teacher_retention"}), frozenset({"mcts_root_retention"}),
    frozenset({CONTINUATION_LOSS_MODE}),
    frozenset({"mcts_root_retention", CONTINUATION_LOSS_MODE}),
    frozenset({GUARDRAIL_LOSS_MODE}),                    # v12: guardrail-only B/C/D
)


def target_in_to_move(side_to_move: str, calibration_target: float) -> float:
    """Express the black-perspective target in the side-to-move perspective.

    The value head outputs side-to-move perspective. For black-to-move the
    target is used as-is; for red-to-move it is negated.
    """
    if side_to_move == "black":
        return float(calibration_target)
    if side_to_move == "red":
        return float(-calibration_target)
    raise ValueError(f"unexpected side_to_move {side_to_move!r}")


@dataclass(frozen=True)
class CalibrationSample:
    """A calibration position plus its per-row weight/tag/target metadata.

    The loss reads the target from record.outcome (already in side-to-move
    perspective); target_black_value is retained as black-perspective metadata.
    """
    record: PositionRecord
    weight_scale: float = 1.0
    tag: str = ""
    target_black_value: float | None = None
    loss_mode: str = "hard_value"            # one of VALID_LOSS_MODES
    teacher_value: float | None = None        # side-to-move; telemetry/validation
    teacher_policy_len: int | None = None      # == len(legal_moves); validation
    has_policy_target: bool = False            # per-row policy-CE mask input

    def __post_init__(self):
        # RETENTION_POLICY_LOSS_MODES rows always carry a policy target, even
        # when a CalibrationSample is constructed directly (bypassing
        # build_calibration_sample) -- preserves pre-v6 mask semantics for
        # teacher_retention/mcts_root_retention regardless of construction path.
        if self.loss_mode in RETENTION_POLICY_LOSS_MODES and not self.has_policy_target:
            object.__setattr__(self, "has_policy_target", True)


def _resolve_target_black(case: dict, fallback: float) -> float:
    """Per-row black-perspective target, falling back to the global value.
    Validates finite and in [-1.0, +1.0]."""
    raw = case.get("target_black_value")
    target = float(fallback) if raw in (None, "") else float(raw)
    if not math.isfinite(target) or not (-1.0 <= target <= 1.0):
        raise ValueError(
            f"target_black_value {target!r} must be finite in [-1.0, 1.0] "
            f"(case {case.get('case_id')!r})")
    return target


def _parse_weight_scale(case: dict) -> tuple[float, bool]:
    """Return (weight_scale, was_explicit). Default 1.0; validate finite and >= 0."""
    raw = case.get("weight_scale")
    if raw in (None, ""):
        return 1.0, False
    w = float(raw)
    if not math.isfinite(w) or w < 0.0:
        raise ValueError(
            f"weight_scale {w!r} must be finite and >= 0 (case {case.get('case_id')!r})")
    return w, True


def _parse_policy_json(case: dict, legal, policy_col: str, sha1_col: str) -> list[float]:
    """Parse and validate a dense policy JSON column against the reconstructed
    legal_moves. Checks: non-empty, length == len(legal), all entries >= 0 and
    finite, sum in 1 ± 1e-3, and the stored sha1 matches the recomputed hash
    over legal (catches a same-length reorder / stale alignment)."""
    cid = case.get("case_id")
    raw = case.get(policy_col)
    if raw in (None, ""):
        raise ValueError(f"{cid}: retention row needs {policy_col}")
    policy = [float(x) for x in json.loads(raw)]
    if len(policy) != len(legal):
        raise ValueError(
            f"{cid}: {policy_col} length {len(policy)} != legal_moves {len(legal)}")
    if any(p < 0.0 or not math.isfinite(p) for p in policy):
        raise ValueError(f"{cid}: {policy_col} has negative/non-finite entries")
    if abs(sum(policy) - 1.0) > 1e-3:
        raise ValueError(f"{cid}: {policy_col} not normalized (sum={sum(policy)})")
    stored = case.get(sha1_col) or ""
    expected = legal_moves_sha1(legal)
    if stored != expected:
        raise ValueError(
            f"{cid}: {sha1_col} mismatch (alignment); "
            f"stored {stored!r} != recomputed {expected!r}")
    return policy


def _parse_teacher_policy(case: dict, legal) -> list[float]:
    return _parse_policy_json(case, legal, "teacher_policy_json", "teacher_legal_moves_sha1")


def _parse_teacher_value(case: dict) -> float:
    """Required raw eval-mode value anchor (side-to-move), finite in [-1, 1]."""
    raw = case.get("teacher_value")
    if raw in (None, ""):
        raise ValueError(
            f"{case.get('case_id')}: retention row needs teacher_value (raw stm anchor)")
    v = float(raw)
    if not math.isfinite(v) or not (-1.0 <= v <= 1.0):
        raise ValueError(
            f"{case.get('case_id')}: teacher_value {v!r} must be finite in [-1,1]")
    return v


def _parse_extra_moves(case: dict) -> list[tuple[int, int]]:
    """Required non-empty extra_moves_json for continuation rows: JSON list of
    {"row": int, "col": int} applied after the position_ply reconstruction.

    Depth-0 exception (v6c): rows with continuation_source == "root_value"
    anchor the ROOT state itself — they carry an explicit empty list (or
    blank) and MUST NOT list any moves. Side/sha1 verification still runs
    against the root state in _apply_extra_moves."""
    cid = case.get("case_id")
    raw = case.get("extra_moves_json")
    is_root_value = case.get("continuation_source") == "root_value"
    if raw in (None, ""):
        if is_root_value:
            return []
        raise ValueError(f"{cid}: continuation row needs extra_moves_json")
    try:
        moves = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{cid}: extra_moves_json invalid JSON: {e}") from e
    if not isinstance(moves, list):
        raise ValueError(f"{cid}: extra_moves_json must be a non-empty list")
    if not moves:
        if is_root_value:
            return []
        raise ValueError(f"{cid}: extra_moves_json must be a non-empty list")
    if is_root_value:
        raise ValueError(
            f"{cid}: root_value row must have empty extra_moves_json; got {raw!r}")
    out = []
    for m in moves:
        if not isinstance(m, dict) or "row" not in m or "col" not in m:
            raise ValueError(f"{cid}: extra_moves_json entries need row/col: {m!r}")
        out.append((int(m["row"]), int(m["col"])))
    return out


def _apply_extra_moves(state, case: dict):
    """Apply extra_moves_json, then verify continuation_side_to_move and
    continuation_legal_moves_sha1 against the reconstructed state. Fail loud."""
    cid = case.get("case_id")
    extra = _parse_extra_moves(case)
    for (r, c) in extra:
        if (r, c) not in set(state.legal_moves()):
            raise ValueError(
                f"{cid}: extra move ({r},{c}) illegal at reconstructed state")
        state = state.apply_move((r, c))
    expected_side = case.get("continuation_side_to_move")
    if expected_side in (None, ""):
        raise ValueError(f"{cid}: continuation row needs continuation_side_to_move")
    if state.to_move != expected_side:
        raise ValueError(
            f"{cid}: continuation_side_to_move {expected_side!r} != reconstructed "
            f"{state.to_move!r}")
    stored = case.get("continuation_legal_moves_sha1") or ""
    recomputed = legal_moves_sha1(state.legal_moves())
    if stored != recomputed:
        raise ValueError(
            f"{cid}: continuation_legal_moves_sha1 mismatch; stored {stored!r} "
            f"!= recomputed {recomputed!r}")
    return state, len(extra)


def build_calibration_position(case: dict, calibration_target: float) -> PositionRecord:
    """Reconstruct a case to a board and build a PositionRecord.

    For hard_value rows visit_counts is a zero vector (policy is not supervised);
    for teacher_retention rows visit_counts holds the dense teacher policy;
    for mcts_root_retention rows visit_counts holds the dense base MCTS root
    visit distribution. outcome carries the soft target (hard_value,
    teacher_retention) or the raw eval-mode value anchor (mcts_root_retention),
    always in side-to-move perspective.
    """
    replay_path = Path(case["replay_path"])
    if not replay_path.exists():
        raise FileNotFoundError(
            f"{case.get('case_id')}: replay not found: {replay_path}")
    replay = json.loads(replay_path.read_text())
    position_ply = int(case["position_ply"])
    side = case["side_to_move"]
    state = position_state(replay, position_ply, side)

    loss_mode = case.get("loss_mode") or "hard_value"
    if loss_mode not in VALID_LOSS_MODES:
        raise ValueError(
            f"{case.get('case_id')}: unknown loss_mode {loss_mode!r} "
            f"(valid: {sorted(VALID_LOSS_MODES)})")
    record_ply = position_ply
    is_guardrail_continuation = (
        loss_mode == GUARDRAIL_LOSS_MODE
        and case.get("extra_moves_json") not in (None, ""))
    if loss_mode == CONTINUATION_LOSS_MODE or is_guardrail_continuation:
        state, n_extra = _apply_extra_moves(state, case)
        record_ply = position_ply + n_extra

    board_chw = state.to_tensor()                       # (30, 24, 24) CHW
    board_hwc = np.transpose(board_chw, (1, 2, 0)).astype(np.float32)  # (24,24,30)
    legal = state.legal_moves()

    if loss_mode == "teacher_retention":
        teacher_value = _parse_teacher_value(case)
        teacher_policy = _parse_teacher_policy(case, legal)
        return PositionRecord(
            board_tensor=board_hwc,
            to_move=state.to_move,
            legal_moves=legal,
            visit_counts=teacher_policy,                 # float "counts"; make_padded_batch normalizes
            outcome=teacher_value,
            active_size=state.active_size,
            ply=record_ply,
            game_n_moves=None,
        )
    if loss_mode == "mcts_root_retention":
        teacher_value = _parse_teacher_value(case)
        root_policy = _parse_policy_json(
            case, legal, "root_visits_json", "root_legal_moves_sha1")
        return PositionRecord(
            board_tensor=board_hwc,
            to_move=state.to_move,
            legal_moves=legal,
            visit_counts=root_policy,        # BASE MCTS root visit distribution (normalized)
            outcome=teacher_value,           # raw eval-mode value anchor, stm, DIRECT
            active_size=state.active_size,
            ply=record_ply,
            game_n_moves=None,
        )
    if loss_mode == CONTINUATION_LOSS_MODE:
        teacher_value = _parse_teacher_value(case)
        if case.get("teacher_policy_json") not in (None, ""):
            visit_counts = _parse_teacher_policy(case, legal)
        else:
            visit_counts = [0] * len(legal)
        return PositionRecord(
            board_tensor=board_hwc,
            to_move=state.to_move,
            legal_moves=legal,
            visit_counts=visit_counts,       # dense teacher policy or zeros (mask 0)
            outcome=teacher_value,           # raw eval-mode value anchor, stm, DIRECT
            active_size=state.active_size,
            ply=record_ply,
            game_n_moves=None,
        )
    return PositionRecord(
        board_tensor=board_hwc,
        to_move=state.to_move,
        legal_moves=legal,
        visit_counts=[0] * len(legal),
        outcome=target_in_to_move(state.to_move, _resolve_target_black(case, calibration_target)),
        active_size=state.active_size,
        ply=record_ply,
        game_n_moves=None,
    )


def build_calibration_sample(case: dict, calibration_target: float) -> CalibrationSample:
    """Wrap a value-only PositionRecord with per-row weight/tag/target metadata."""
    loss_mode = case.get("loss_mode") or "hard_value"
    if loss_mode == "hard_value":
        populated = [k for k in ("teacher_value", "teacher_policy_json",
                                 "teacher_legal_moves_sha1",
                                 "root_visits_json", "root_legal_moves_sha1",
                                 "extra_moves_json", "continuation_side_to_move",
                                 "continuation_legal_moves_sha1")
                     if case.get(k) not in (None, "")]
        if populated:
            raise ValueError(
                f"{case.get('case_id')}: hard_value row must leave retention columns "
                f"blank; found {populated}")
    elif loss_mode == "mcts_root_retention":
        if case.get("teacher_policy_json") not in (None, ""):
            raise ValueError(
                f"{case.get('case_id')}: mcts_root_retention row must leave "
                f"teacher_policy_json blank (root_visits_json is the policy target)")
    elif loss_mode == CONTINUATION_LOSS_MODE:
        if case.get("root_visits_json") not in (None, ""):
            raise ValueError(
                f"{case.get('case_id')}: continuation row must leave "
                f"root_visits_json blank (it is not a root-policy target)")
    elif loss_mode == GUARDRAIL_LOSS_MODE:
        if case.get("target_black_value") in (None, ""):
            raise ValueError(
                f"{case.get('case_id')}: asymmetric_guardrail_retention row must "
                f"populate target_black_value (BASE black-perspective value)")
        if case.get("teacher_policy_json") not in (None, ""):
            raise ValueError(
                f"{case.get('case_id')}: asymmetric_guardrail_retention row must "
                f"leave teacher_policy_json blank (value-only, no policy CE)")
        if case.get("root_visits_json") not in (None, ""):
            raise ValueError(
                f"{case.get('case_id')}: asymmetric_guardrail_retention row must "
                f"leave root_visits_json blank (not a policy target)")
    record = build_calibration_position(case, calibration_target)
    weight_scale, _ = _parse_weight_scale(case)
    tag = case.get("tag") or ""
    target_black = _resolve_target_black(case, calibration_target)
    teacher_value = (float(case["teacher_value"])
                     if loss_mode in TEACHER_MODE_LOSS_MODES else None)
    teacher_policy_len = (len(record.visit_counts)
                          if loss_mode in TEACHER_MODE_LOSS_MODES else None)
    has_policy_target = (
        loss_mode in RETENTION_POLICY_LOSS_MODES
        or (loss_mode == CONTINUATION_LOSS_MODE
            and case.get("teacher_policy_json") not in (None, "")))
    return CalibrationSample(record=record, weight_scale=weight_scale,
                             tag=tag, target_black_value=target_black,
                             loss_mode=loss_mode, teacher_value=teacher_value,
                             teacher_policy_len=teacher_policy_len,
                             has_policy_target=has_policy_target)


class CalibrationPool:
    """Fixed pool of CalibrationSamples; sampled with replacement."""

    def __init__(self, samples, has_weight_scale: bool = False,
                 schema: str = "global_target"):
        if not samples:
            raise ValueError("CalibrationPool requires at least one sample")
        if any(not isinstance(s, CalibrationSample) for s in samples):
            raise TypeError(
                "CalibrationPool stores CalibrationSample objects; "
                "use build_calibration_sample / from_manifest")
        self._samples = list(samples)
        self.has_weight_scale = bool(has_weight_scale)
        self.schema = schema
        self._by_tag: dict[str, list] = {}
        for s in self._samples:
            self._by_tag.setdefault(s.tag, []).append(s)

    def __len__(self):
        return len(self._samples)

    def sample(self, k: int, rng):
        if k <= 0:
            return []
        return [rng.choice(self._samples) for _ in range(k)]

    def validate_tag_schedule(self, schedule: dict) -> None:
        """Raise ValueError if any positively-scheduled tag is absent from the pool.

        Call once at setup (before self-play) so a typo'd tag fails fast rather
        than after a wasted self-play iteration. Zero-count tags are ignored.
        """
        missing = sorted(tag for tag, n in schedule.items()
                         if n > 0 and tag not in self._by_tag)
        if missing:
            raise ValueError(
                f"calibration tag schedule requested missing tags {missing}; "
                f"pool has tags {sorted(self._by_tag)}")

    def sample_by_tag(self, schedule: dict, rng):
        """Draw per-tag counts (with replacement) per an explicit tag->count schedule.

        Validates the schedule first (raises on an absent positively-scheduled
        tag), then preserves schedule (dict-insertion) order and skips any tag
        whose count is <= 0.
        """
        self.validate_tag_schedule(schedule)
        out = []
        for tag, n in schedule.items():
            if n <= 0:
                continue
            out.extend(rng.choice(self._by_tag[tag]) for _ in range(n))
        return out

    def tag_counts(self) -> dict:
        counts: dict = {}
        for s in self._samples:
            counts[s.tag] = counts.get(s.tag, 0) + 1
        return counts

    @classmethod
    def from_manifest(cls, manifest_path, calibration_target: float):
        cases = load_csv_manifest(manifest_path)["cases"]
        modes = {(c.get("loss_mode") or "hard_value") for c in cases}
        retention_modes = frozenset(modes - {"hard_value"})
        if retention_modes not in _ALLOWED_RETENTION_MODE_SETS:
            raise ValueError(
                f"manifest mixes retention loss_modes {sorted(retention_modes)}; "
                f"allowed combinations: "
                f"{sorted(sorted(s) for s in _ALLOWED_RETENTION_MODE_SETS)}")
        samples = [build_calibration_sample(c, calibration_target) for c in cases]
        has_weight_scale = any(c.get("weight_scale") not in (None, "") for c in cases)
        if CONTINUATION_LOSS_MODE in modes:
            schema = CONTINUATION_LOSS_MODE
        elif "mcts_root_retention" in modes:
            schema = "mcts_root_retention"
        elif "teacher_retention" in modes:
            schema = "teacher_retention"
        elif GUARDRAIL_LOSS_MODE in modes:
            schema = GUARDRAIL_LOSS_MODE
        elif any(c.get("target_black_value") not in (None, "") for c in cases):
            schema = "per_row_target"
        else:
            schema = "global_target"
        return cls(samples, has_weight_scale=has_weight_scale, schema=schema)


def split_samples(samples, has_weight_scale: bool):
    """Split CalibrationSamples into (records, weights). weights is None when the
    manifest specified no explicit weight_scale (→ loss uses plain mx.mean,
    byte-identical to v1); otherwise a float32 array of per-sample weight_scale."""
    records = [s.record for s in samples]
    weights = (np.asarray([s.weight_scale for s in samples], dtype=np.float32)
               if has_weight_scale else None)
    return records, weights


def split_samples_with_modes(samples, has_weight_scale: bool):
    """Like split_samples, plus a teacher_policy_mask (float32 (N,), 1.0 for
    teacher_retention rows, 0.0 otherwise). Used by the v4 calibration loss to
    gate the policy-CE term to retention rows only."""
    records, weights = split_samples(samples, has_weight_scale)
    mask = np.asarray(
        [1.0 if s.has_policy_target else 0.0 for s in samples],
        dtype=np.float32)
    return records, weights, mask


def split_samples_with_guardrail(samples, has_weight_scale: bool):
    """Like split_samples, plus a guardrail_sign (float32 (N,)): +1.0 for a
    black-to-move guardrail row, -1.0 for a red-to-move guardrail row, 0.0 for
    non-guardrail rows. The v12 hinge uses this to convert the candidate value
    to black perspective per row (relu(sign*(v - target) - margin)**2)."""
    records, weights = split_samples(samples, has_weight_scale)
    sign = np.asarray(
        [(1.0 if s.record.to_move == "black" else -1.0)
         if s.loss_mode == GUARDRAIL_LOSS_MODE else 0.0
         for s in samples],
        dtype=np.float32)
    return records, weights, sign


def build_post_opening_calibration_block(config: dict, enabled: bool,
                                         loss_accumulator: dict) -> dict:
    """Per-iteration calibration telemetry for the training stats sidecar.

    calib_mean_value_pred is the headline signal: it should drift from ~+0.6
    toward the target (~-0.5) over the run.
    """
    steps = max(int(loss_accumulator.get("steps_done", 0)), 1)
    n_drawn = int(loss_accumulator.get("sum_calib_n_drawn", 0))
    return {
        "version": 1,
        "enabled": bool(enabled),
        "config": dict(config),
        "loss": {
            "calib_loss_avg_iter":
                float(loss_accumulator.get("sum_calib_loss", 0.0)) / steps,
            "calib_mean_value_pred":
                float(loss_accumulator.get("sum_calib_value_pred", 0.0)) / steps,
            "calib_n_drawn_total": n_drawn,
            "calib_n_drawn_per_step": n_drawn / steps,
            "calib_value_term_avg_iter":
                float(loss_accumulator.get("sum_calib_value_term", 0.0)) / steps,
            "calib_policy_ce_avg_iter":
                float(loss_accumulator.get("sum_calib_policy_ce", 0.0)) / steps,
            "calib_policy_kl_est_avg_iter":
                float(loss_accumulator.get("sum_calib_policy_kl_est", 0.0)) / steps,
            "n_teacher_retention_drawn":
                int(loss_accumulator.get("sum_n_teacher_retention", 0)),
            "guardrail_hinge_loss":
                float(loss_accumulator.get("sum_guardrail_hinge_loss", 0.0)) / steps,
            "guardrail_active_frac":
                float(loss_accumulator.get("sum_guardrail_active_frac", 0.0)) / steps,
            "guardrail_margin":
                float(loss_accumulator.get("guardrail_margin", 0.0)),
        },
        "draws_by_tag": dict(loss_accumulator.get("sum_calib_n_drawn_by_tag", {})),
    }
