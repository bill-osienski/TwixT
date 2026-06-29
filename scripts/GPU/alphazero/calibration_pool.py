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
    loss_mode: str = "hard_value"            # "hard_value" | "teacher_retention"
    teacher_value: float | None = None        # side-to-move; telemetry/validation
    teacher_policy_len: int | None = None      # == len(legal_moves); validation


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


def _parse_teacher_policy(case: dict, legal) -> list[float]:
    """Parse and validate teacher_policy_json against the reconstructed legal_moves.

    Checks: non-empty, length == len(legal), all entries >= 0 and finite,
    sum in 1 ± 1e-3, and teacher_legal_moves_sha1 matches the recomputed hash
    over legal (catches a same-length reorder).
    """
    cid = case.get("case_id")
    raw = case.get("teacher_policy_json")
    if raw in (None, ""):
        raise ValueError(f"{cid}: teacher_retention row needs teacher_policy_json")
    policy = [float(x) for x in json.loads(raw)]
    if len(policy) != len(legal):
        raise ValueError(
            f"{cid}: teacher_policy length {len(policy)} != legal_moves {len(legal)}")
    if any(p < 0.0 or not math.isfinite(p) for p in policy):
        raise ValueError(f"{cid}: teacher_policy has negative/non-finite entries")
    if abs(sum(policy) - 1.0) > 1e-3:
        raise ValueError(f"{cid}: teacher_policy not normalized (sum={sum(policy)})")
    stored = case.get("teacher_legal_moves_sha1") or ""
    expected = legal_moves_sha1(legal)
    if stored != expected:
        raise ValueError(
            f"{cid}: teacher_legal_moves_sha1 mismatch (alignment); "
            f"stored {stored!r} != recomputed {expected!r}")
    return policy


def build_calibration_position(case: dict, calibration_target: float) -> PositionRecord:
    """Reconstruct a case to a board and build a PositionRecord.

    For hard_value rows visit_counts is a zero vector (policy is not supervised);
    for teacher_retention rows visit_counts holds the dense teacher policy.
    outcome carries the soft target in side-to-move perspective.
    """
    replay_path = Path(case["replay_path"])
    if not replay_path.exists():
        raise FileNotFoundError(
            f"{case.get('case_id')}: replay not found: {replay_path}")
    replay = json.loads(replay_path.read_text())
    position_ply = int(case["position_ply"])
    side = case["side_to_move"]
    state = position_state(replay, position_ply, side)

    board_chw = state.to_tensor()                       # (30, 24, 24) CHW
    board_hwc = np.transpose(board_chw, (1, 2, 0)).astype(np.float32)  # (24,24,30)
    legal = state.legal_moves()

    loss_mode = case.get("loss_mode") or "hard_value"
    if loss_mode == "teacher_retention":
        teacher_value = float(case["teacher_value"])
        if not math.isfinite(teacher_value) or not (-1.0 <= teacher_value <= 1.0):
            raise ValueError(
                f"{case.get('case_id')}: teacher_value {teacher_value!r} must be finite in [-1,1]")
        teacher_policy = _parse_teacher_policy(case, legal)
        return PositionRecord(
            board_tensor=board_hwc,
            to_move=state.to_move,
            legal_moves=legal,
            visit_counts=teacher_policy,                 # float "counts"; make_padded_batch normalizes
            outcome=teacher_value,
            active_size=state.active_size,
            ply=position_ply,
            game_n_moves=None,
        )
    return PositionRecord(
        board_tensor=board_hwc,
        to_move=state.to_move,
        legal_moves=legal,
        visit_counts=[0] * len(legal),
        outcome=target_in_to_move(state.to_move, _resolve_target_black(case, calibration_target)),
        active_size=state.active_size,
        ply=position_ply,
        game_n_moves=None,
    )


def build_calibration_sample(case: dict, calibration_target: float) -> CalibrationSample:
    """Wrap a value-only PositionRecord with per-row weight/tag/target metadata."""
    if (case.get("loss_mode") or "hard_value") == "hard_value":
        populated = [k for k in ("teacher_value", "teacher_policy_json",
                                 "teacher_legal_moves_sha1")
                     if case.get(k) not in (None, "")]
        if populated:
            raise ValueError(
                f"{case.get('case_id')}: hard_value row must leave teacher columns "
                f"blank; found {populated}")
    record = build_calibration_position(case, calibration_target)
    weight_scale, _ = _parse_weight_scale(case)
    tag = case.get("tag") or ""
    target_black = _resolve_target_black(case, calibration_target)
    loss_mode = case.get("loss_mode") or "hard_value"
    teacher_value = (float(case["teacher_value"])
                     if loss_mode == "teacher_retention" else None)
    teacher_policy_len = (len(record.visit_counts)
                          if loss_mode == "teacher_retention" else None)
    return CalibrationSample(record=record, weight_scale=weight_scale,
                             tag=tag, target_black_value=target_black,
                             loss_mode=loss_mode, teacher_value=teacher_value,
                             teacher_policy_len=teacher_policy_len)


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
        samples = [build_calibration_sample(c, calibration_target) for c in cases]
        has_weight_scale = any(c.get("weight_scale") not in (None, "") for c in cases)
        if any((c.get("loss_mode") or "hard_value") == "teacher_retention" for c in cases):
            schema = "teacher_retention"
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
        },
        "draws_by_tag": dict(loss_accumulator.get("sum_calib_n_drawn_by_tag", {})),
    }
