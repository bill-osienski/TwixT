"""Context-relative FPU (policy-mass) staged discovery diagnostic.

Frozen design ref: docs/superpowers/specs/2026-07-10-context-relative-fpu-policy-mass-design.md
(§4 trace observer, §5 coefficient-selection protocol, §6 pre-registered numeric
gates). Plan Task 7.

=============================================================================
PURE SECTION -- typed configs, gate fns, stage/mode enforcement, controls-
artifact guards, and the trace observer. NO MCTS / evaluator / GPU / MLX /
checkpoint / corpus I/O here. Importing this module is GPU/MLX-free (verified:
`import scripts.GPU.alphazero.diagnose_fpu_policy_mass` leaves `mlx` out of
`sys.modules`); everything heavy is imported LAZILY inside `main()` / the stage
runners, mirroring build_fpu_dev_corpus's Task-6 discipline.
=============================================================================

The OPERATOR shell (`main(--mode,--stage)`) runs real 400-sim MCTS on the dev
corpus and is NEVER invoked by this task's tests. Two stages:

  controls   -- runs `{absolute_off, r0}`, writes joinable per-position rows
                (`controls_cases.csv`, keyed by canonical_sha1) + a summary +
                `controls_gate.json` carrying `r0_qualified` and a reproducible
                fingerprint. `r0_qualified` = NOT `dev_safety_verdict(r0_rows,
                ref=absolute_off, ...)` rejecting (§5 step 0 control-qualification).
  candidates -- LOADS the controls artifact, VALIDATES the fingerprint
                (`validate_controls_fingerprint`), REQUIRES `r0_qualified`
                (`require_r0_qualified`), joins control rows by canonical_sha1,
                runs the grid (tuning) / one r (frozen_check), applies the §6.2
                dev-safety gates vs BOTH references + the §6.3 selected-A gate,
                and reports the SMALLEST safe-passing r (§5 steps 1-3).

All §6 thresholds are FROZEN below. The only values learned from the control
run are `r0_target_lockin_count` / `absoff_target_lockin_count`, substituted
into the already-frozen `baseline + 2` lock-in caps.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Typed run-configs (fix 9) -- explicit labels so absolute_off / r0 / grid are
# all distinguishable, and `absolute_off` (reduction None) != `r0` (0.0).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FpuRunConfig:
    """One FPU run configuration. `reduction is None` => absolute-off
    (byte-identical legacy path); `reduction == 0.0` => policy-mass ENABLED at
    FPU = Q_parent; `reduction > 0` => the parent-relative reduction grid."""
    label: str
    reduction: Optional[float]


ABSOLUTE_OFF = FpuRunConfig("absolute_off", None)
R0 = FpuRunConfig("r0", 0.0)
GRID: Tuple[FpuRunConfig, ...] = (
    FpuRunConfig("r0.10", 0.10), FpuRunConfig("r0.20", 0.20),
    FpuRunConfig("r0.35", 0.35), FpuRunConfig("r0.50", 0.50),
    FpuRunConfig("r0.75", 0.75),
)

# §6.0: exact 6400-sim selected-A reference mean (frozen).
V_REF = -0.0451

# Frozen §6 gate thresholds (committed before any nonzero-r result).
DEV_NEW_COLLAPSE_TARGET = 0.05    # §6.2 target new-collapse-vs-X rate
DEV_NEW_COLLAPSE_BAND = 0.10      # §6.2 per-band (n >= 20) new-collapse rate
DEV_BAND_MIN_N = 20
DEV_LOCKIN_MARGIN = 2             # §6.2 lock-in > Xtarget_lockin_count + 2
DEV_P95_MOVER = 0.35             # §6.2 broad-distortion target p95 |mover Δ|
DEV_COMPOUND_EFF = 0.50          # §6.2 compound: mean eff-children reduction
DEV_COMPOUND_TOPSHARE = 0.15     # §6.2 compound: mean top-share increase
DEV_CONTROL_FLIP = 0.10          # §6.2 control top-move-flip-to-lower-prior rate
DEV_CONTROL_P95 = 0.35           # §6.2 control p95 |mover Δ|
A_REPLY_REDUCTION = 0.50         # §6.3 reply reduction
A_PROGRESS = 0.50                # §6.3 progress
A_NEW_COLLAPSE_MAX = 2           # §6.3 new-collapse count on A
A_TOPSHARE_MAX = 0.15            # §6.3 mean top-share increase on A
PERCENTILE_Q = 95                # the p95 in every §6.2 broad-distortion gate

MCTS_SIMS = 400
OBSERVER_SCHEMA_VERSION = 1

MODES = ("tuning", "frozen_check")
STAGES = ("controls", "candidates")


# ---------------------------------------------------------------------------
# §6.0 primitive gate formulas (pure)
# ---------------------------------------------------------------------------

def prior_rank(priors: Mapping[Any, float], move: Any) -> int:
    """Prior rank of `move`: `1 + |{b : prior(b) > prior(move)}|` (strictly
    greater; exact ties do NOT inflate the rank). Top move => 1."""
    p = priors[move]
    return 1 + sum(1 for other in priors.values() if other > p)


def progress(v_off: float, v_r: float) -> float:
    """§6.0 selected-A progress: `(V_off - V_r) / (V_off - V_REF)`."""
    return (v_off - v_r) / (v_off - V_REF)


def reply_reduction(replies_ref: float, replies_x: float) -> float:
    """§6.0 reply reduction of config X vs reference: `1 - replies_X/replies_ref`."""
    return 1.0 - replies_x / replies_ref


def top_share(root: Any) -> float:
    """§6.0 root top share: `top_child.visit_count / root.visit_count` (the v16a
    quantity; top child = max visit_count). 0.0 for an unvisited/childless root.
    Duck-typed on `root` (reads .visit_count / .children) so no MCTS import."""
    total = getattr(root, "visit_count", 0)
    children = getattr(root, "children", None) or {}
    if not children or total <= 0:
        return 0.0
    top = max(children.values(), key=lambda c: c.visit_count)
    return top.visit_count / total


def lock_in_event(row: Mapping[str, Any]) -> bool:
    """§6.1 early low-prior lock-in event (per target position). Qualifies iff
    ALL: prior-rank > 10, prior < 0.01, explored-mass-at-stabilization < 0.25,
    stabilization-sim <= 100, final-root-top-share >= 0.90."""
    return (
        row["selected_move_prior_rank"] > 10
        and row["selected_move_prior"] < 0.01
        and row["explored_mass_at_stabilization"] < 0.25
        and row["stabilization_sim"] <= 100
        and row["final_root_top_share"] >= 0.90
    )


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile -- the SAME convention as
    diagnose_fpu_sweep._percentile (shared so the frozen p95 boundary can never
    drift between the gate fn and any other summarizer)."""
    xs = sorted(values)
    n = len(xs)
    if n == 0:
        raise ValueError("percentile of empty sequence")
    if n == 1:
        return float(xs[0])
    rank = (q / 100.0) * (n - 1)
    lo = int(rank)
    if lo + 1 >= n:
        return float(xs[-1])
    return float(xs[lo] + (rank - lo) * (xs[lo + 1] - xs[lo]))


def _mean(values) -> float:
    xs = list(values)
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------------------
# §6.2 / §6.3 verdicts (pure aggregators over plain-dict rows)
# ---------------------------------------------------------------------------
#
# dev-safety target-row schema (per position, metrics already computed "vs the
# given reference X" by the operator):
#     role="target", band, new_collapse: bool, lock_in: bool,
#     mover_delta: float, eff_children_reduction: float, top_share_inc: float
# dev-safety control-row schema:
#     role="control", mover_delta: float, control_flip_to_lower_prior: bool
# `lock_in` is a PRE-COMPUTED boolean (the operator sets it via lock_in_event);
# this keeps dev_safety_verdict a pure aggregator and lets a test isolate the
# lock-in gate without also constructing the five lock_in_event fields.


@dataclass(frozen=True)
class SafetyVerdict:
    rejected: bool
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class AVerdict:
    passed: bool
    reply_reduction: float
    progress: float
    a_new_collapse: int
    a_top_share_inc: float


def dev_safety_verdict(rows: Sequence[Mapping[str, Any]], ref: FpuRunConfig,
                       r0_lockin: int, absoff_lockin: int) -> SafetyVerdict:
    """§6.2 development-safety verdict vs a single reference `X` (`ref`). REJECT
    (rejected=True) if ANY frozen gate trips. Target and control rows are split
    by `role`; each subset is evaluated only when non-empty (so a test can
    isolate one gate by supplying just the rows it needs). The lock-in baseline
    follows `ref`: `r0_lockin` when ref is r0 (reduction 0.0), else
    `absoff_lockin` (ref absolute_off / reduction None)."""
    target = [r for r in rows if r.get("role") == "target"]
    control = [r for r in rows if r.get("role") == "control"]
    reasons: List[str] = []

    if target:
        n = len(target)
        nc_rate = sum(1 for r in target if r["new_collapse"]) / n
        if nc_rate >= DEV_NEW_COLLAPSE_TARGET:
            reasons.append(f"target_new_collapse_rate={nc_rate:.4f}>={DEV_NEW_COLLAPSE_TARGET}")

        by_band: Dict[Any, List] = defaultdict(list)
        for r in target:
            by_band[r["band"]].append(r)
        for band, brows in by_band.items():
            if len(brows) >= DEV_BAND_MIN_N:
                brate = sum(1 for r in brows if r["new_collapse"]) / len(brows)
                if brate >= DEV_NEW_COLLAPSE_BAND:
                    reasons.append(f"band[{band}]_new_collapse={brate:.4f}>={DEV_NEW_COLLAPSE_BAND}")

        baseline = r0_lockin if getattr(ref, "reduction", None) == 0.0 else absoff_lockin
        lockins = sum(1 for r in target if r["lock_in"])
        if lockins > baseline + DEV_LOCKIN_MARGIN:
            reasons.append(f"lockin_count={lockins}>baseline+2={baseline + DEV_LOCKIN_MARGIN}")

        p95 = _percentile([abs(r["mover_delta"]) for r in target], PERCENTILE_Q)
        if p95 >= DEV_P95_MOVER:
            reasons.append(f"target_p95_mover_delta={p95:.4f}>={DEV_P95_MOVER}")

        eff = _mean(r["eff_children_reduction"] for r in target)
        tsi = _mean(r["top_share_inc"] for r in target)
        if eff >= DEV_COMPOUND_EFF and tsi >= DEV_COMPOUND_TOPSHARE:
            reasons.append(f"compound eff_reduction={eff:.4f}&top_share_inc={tsi:.4f}")

    if control:
        flip = sum(1 for r in control if r["control_flip_to_lower_prior"]) / len(control)
        if flip >= DEV_CONTROL_FLIP:
            reasons.append(f"control_flip_rate={flip:.4f}>={DEV_CONTROL_FLIP}")
        cp95 = _percentile([abs(r["mover_delta"]) for r in control], PERCENTILE_Q)
        if cp95 >= DEV_CONTROL_P95:
            reasons.append(f"control_p95_mover_delta={cp95:.4f}>={DEV_CONTROL_P95}")

    return SafetyVerdict(rejected=bool(reasons), reasons=tuple(reasons))


def selected_a_verdict(rows: Sequence[Mapping[str, Any]]) -> AVerdict:
    """§6.3 selected-A mechanism gate (vs absolute_off). PASS iff ALL:
    reply_reduction >= 0.50, progress >= 0.50, a_new_collapse <= 2,
    a_top_share_inc <= 0.15. Aggregates over A rows carrying `off_value`,
    `r_value`, `replies_ref`, `replies_x`, `new_collapse`, `top_share_inc`."""
    rows = list(rows)
    if not rows:
        raise ValueError("selected_a_verdict: no A rows")
    v_off = _mean(r["off_value"] for r in rows)
    v_r = _mean(r["r_value"] for r in rows)
    prog = progress(v_off, v_r)
    rr = reply_reduction(sum(r["replies_ref"] for r in rows),
                         sum(r["replies_x"] for r in rows))
    anc = sum(1 for r in rows if r["new_collapse"])
    atsi = _mean(r["top_share_inc"] for r in rows)
    passed = (rr >= A_REPLY_REDUCTION and prog >= A_PROGRESS
              and anc <= A_NEW_COLLAPSE_MAX and atsi <= A_TOPSHARE_MAX)
    return AVerdict(passed=passed, reply_reduction=rr, progress=prog,
                    a_new_collapse=anc, a_top_share_inc=atsi)


# ---------------------------------------------------------------------------
# Stage/mode enforcement -- EXACT config sets (fix 3)
# ---------------------------------------------------------------------------

def validate_stage_mode(cases: Sequence[Mapping[str, Any]], *, mode: str, stage: str,
                        run_configs) -> None:
    """Raise ValueError unless the (mode, stage) pairing carries EXACTLY the
    frozen run-config set (by value, not subset/superset) AND every row in
    `cases` has `split == mode`:

        (tuning, controls):         {absolute_off, r0}   exactly
        (tuning, candidates):       set(GRID)            exactly
        (frozen_check, controls):   {absolute_off, r0}   exactly
        (frozen_check, candidates): exactly one nonzero-r config
    """
    for row in cases:
        if row.get("split") != mode:
            raise ValueError(
                f"validate_stage_mode: row split {row.get('split')!r} != mode {mode!r}")

    configs = list(run_configs)
    rc = set(configs)
    if mode == "tuning" and stage == "controls":
        if rc != {ABSOLUTE_OFF, R0}:
            raise ValueError("(tuning, controls) requires exactly {absolute_off, r0}")
    elif mode == "tuning" and stage == "candidates":
        if rc != set(GRID):
            raise ValueError("(tuning, candidates) requires exactly set(GRID)")
    elif mode == "frozen_check" and stage == "controls":
        if rc != {ABSOLUTE_OFF, R0}:
            raise ValueError("(frozen_check, controls) requires exactly {absolute_off, r0}")
    elif mode == "frozen_check" and stage == "candidates":
        if len(configs) != 1:
            raise ValueError("(frozen_check, candidates) requires exactly one config")
        only = configs[0]
        if only.reduction is None or only.reduction == 0.0:
            raise ValueError(
                "(frozen_check, candidates) requires a single NONZERO-r config "
                "(not absolute_off / r0)")
    else:
        raise ValueError(f"validate_stage_mode: unknown (mode, stage) ({mode!r}, {stage!r})")


# ---------------------------------------------------------------------------
# Controls-artifact guards (pure; driven in tests by a fabricated gate dict)
# ---------------------------------------------------------------------------

FINGERPRINT_KEYS = (
    "dev_manifest_sha1", "selected_a_manifest_sha1", "checkpoint_identity",
    "mcts_sims", "search_cfg", "seeds", "git_commit", "observer_schema_version",
)


def validate_controls_fingerprint(gate_json: Mapping[str, Any],
                                  expected: Mapping[str, Any]) -> None:
    """Raise ValueError if the controls artifact's fingerprint is missing or
    differs from `expected` in ANY key (stale/mismatched controls => the
    candidate stage must refuse to reuse them)."""
    fp = gate_json.get("fingerprint")
    if fp is None:
        raise ValueError("controls_gate.json has no 'fingerprint' block")
    mismatched = sorted(k for k in set(fp) | set(expected) if fp.get(k) != expected.get(k))
    if mismatched:
        raise ValueError(
            f"stale/mismatched controls fingerprint (differing keys: {mismatched})")


def require_r0_qualified(gate_json: Mapping[str, Any]) -> None:
    """Raise ValueError unless `gate_json['r0_qualified']` is truthy. Per §5
    step 0, if r=0.0 failed the dev-safety table vs absolute_off the whole
    parent-relative family is rejected and the candidate stage must not run."""
    if not gate_json.get("r0_qualified"):
        raise ValueError(
            "controls gate: r0_qualified is not true -- r=0.0 failed the §6.2 "
            "control-qualification vs absolute_off; the candidate stage is refused")


def require_matching_mode(gate_json: Mapping[str, Any], expected_mode: str) -> None:
    """Refuse a controls artifact produced for a different --mode (its
    split-specific lock-in baselines and r0_qualified do not transfer)."""
    got = gate_json.get("mode")
    if got != expected_mode:
        raise ValueError(
            f"controls artifact mode {got!r} != requested mode {expected_mode!r}; "
            f"regenerate controls for --mode {expected_mode}")


# ---------------------------------------------------------------------------
# §4 trace observer -- INCREMENTAL per-completed-simulation collector.
# Consumes on_root_simulation(count, root, updated_root_move,
# current_root_leader_move); the leader is taken from the PASSED arg (never
# recomputed). Isolation lives here, in the diagnostic module -- mcts.py holds
# only the single guarded callback.
# ---------------------------------------------------------------------------


class FpuTraceObserver:
    """Records, per real 400-sim root search: first-visit sim per root move;
    explored-mass 25/50/75% crossing sims (mass = summed prior of first-visited
    children, excludes virtual/pending); the leader-change timeline (from the
    passed leader); the final leader's LAST-takeover sim (= stabilization); the
    explored mass at stabilization and when the final leader first led; the
    final top move's first-visit mass; and the end-state selected-move prior +
    rank + top share. All updates are O(1)/sim -- no 200-child rescan."""

    _MASS_THRESHOLDS = (0.25, 0.50, 0.75)
    _UNSET = object()

    def __init__(self) -> None:
        self.completed_simulation_count = 0
        self.first_visit_sim: Dict[Any, int] = {}
        self._first_visit_mass: Dict[Any, float] = {}
        self.explored_mass = 0.0
        self.mass_cross_sim: Dict[float, Optional[int]] = {t: None for t in self._MASS_THRESHOLDS}
        self._timeline: List[Tuple[int, Any, float]] = []   # (sim, leader, mass_at_change)
        self._prev_leader: Any = self._UNSET
        self._root: Any = None

    def on_root_simulation(self, completed_simulation_count: int, root: Any,
                           updated_root_move: Optional[int],
                           current_root_leader_move: Optional[int]) -> None:
        self.completed_simulation_count = completed_simulation_count
        self._root = root

        # First-visit + explored-mass bookkeeping. A None root move (the sim-1
        # edge case) is IGNORED here but the counter above still advanced.
        if updated_root_move is not None and updated_root_move not in self.first_visit_sim:
            self.first_visit_sim[updated_root_move] = completed_simulation_count
            priors = getattr(root, "priors", None) or {}
            self.explored_mass += priors.get(updated_root_move, 0.0)
            self._first_visit_mass[updated_root_move] = self.explored_mass
            for t in self._MASS_THRESHOLDS:
                if self.mass_cross_sim[t] is None and self.explored_mass >= t:
                    self.mass_cross_sim[t] = completed_simulation_count

        # Leader-change timeline -- leader is the PASSED arg, never recomputed.
        if current_root_leader_move != self._prev_leader:
            self._timeline.append(
                (completed_simulation_count, current_root_leader_move, self.explored_mass))
            self._prev_leader = current_root_leader_move

    # -- derived, read after the run ---------------------------------------
    @property
    def leader_timeline(self) -> List[Tuple[int, Any]]:
        return [(sim, leader) for sim, leader, _mass in self._timeline]

    @property
    def final_leader_move(self) -> Optional[int]:
        return self._timeline[-1][1] if self._timeline else None

    @property
    def stabilization_sim(self) -> Optional[int]:
        """Final leader's LAST takeover: the last timeline change-point, whose
        leader is (by construction) the final leader and which nothing supersedes."""
        return self._timeline[-1][0] if self._timeline else None

    @property
    def explored_mass_at_stabilization(self) -> float:
        return self._timeline[-1][2] if self._timeline else 0.0

    @property
    def mass_when_final_leader_first_led(self) -> float:
        final = self.final_leader_move
        for _sim, leader, mass in self._timeline:
            if leader == final:
                return mass
        return 0.0

    @property
    def final_top_first_visit_mass(self) -> Optional[float]:
        return self._first_visit_mass.get(self.final_leader_move)

    @property
    def selected_move_prior(self) -> Optional[float]:
        final = self.final_leader_move
        priors = getattr(self._root, "priors", None)
        if final is None or not priors:
            return None
        return priors.get(final)

    @property
    def selected_move_prior_rank(self) -> Optional[int]:
        final = self.final_leader_move
        priors = getattr(self._root, "priors", None)
        if final is None or not priors:
            return None
        return prior_rank(priors, final)

    @property
    def final_root_top_share(self) -> float:
        return top_share(self._root) if self._root is not None else 0.0

    def result(self) -> Dict[str, Any]:
        """End-state row: the five §6.1 lock_in_event fields + the trace
        dynamics. `lock_in_event(observer.result())` decides the lock-in flag."""
        return {
            "selected_move_prior_rank": self.selected_move_prior_rank,
            "selected_move_prior": self.selected_move_prior,
            "explored_mass_at_stabilization": self.explored_mass_at_stabilization,
            "stabilization_sim": self.stabilization_sim,
            "final_root_top_share": self.final_root_top_share,
            "final_leader_move": self.final_leader_move,
            "first_visit_sim": dict(self.first_visit_sim),
            "mass_cross_sim": dict(self.mass_cross_sim),
            "leader_timeline": self.leader_timeline,
            "explored_mass": self.explored_mass,
            "final_top_first_visit_mass": self.final_top_first_visit_mass,
            "mass_when_final_leader_first_led": self.mass_when_final_leader_first_led,
            "completed_simulation_count": self.completed_simulation_count,
        }


# =============================================================================
# OPERATOR SHELL -- real 400-sim MCTS on the dev corpus. NEVER run by tests.
# Everything below imports MCTS / eval_runner / build_fpu_dev_corpus LAZILY, so
# importing this module stays GPU/MLX-free. Mirrors diagnose_fpu_sweep /
# build_v16a_neutral_position_manifest for the checkpoint/search wiring.
# =============================================================================

DEFAULT_DEV_MANIFEST = "logs/eval/fpu_dev_corpus/dev_corpus_manifest.csv"
DEFAULT_OUT_DIR = "logs/eval/fpu_policy_mass"
DEFAULT_SEED_BASE = 20260711
COLLAPSE_TOP_SHARE = 0.95        # a root "collapsed" iff top share >= 0.95 (v16a)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _file_sha1(path: Optional[str]) -> str:
    """Streaming SHA1 of a file's bytes (identity for the fingerprint), or a
    sentinel when the path is absent/unreadable."""
    if not path:
        return "none"
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return "missing"


def _checkpoint_identity(checkpoint: str) -> str:
    return f"{Path(checkpoint).name}:{_file_sha1(checkpoint)}"


def _run_seed(seed_base: int, game_idx: int, ply: int) -> int:
    """Deterministic per-position seed (same row_seed idiom as build_fpu_dev_corpus)."""
    return int(seed_base) ^ int(game_idx) ^ int(ply)


def _search_cfg_dict(eval_batch_size: int, stall_flush_sims: int, c_puct: float) -> dict:
    return {"mcts_sims": MCTS_SIMS, "eval_batch_size": eval_batch_size,
            "stall_flush_sims": stall_flush_sims, "c_puct": c_puct}


def _build_fingerprint(*, dev_manifest: str, selected_a_manifest: Optional[str],
                       checkpoint: str, search_cfg: dict, seeds: dict) -> dict:
    """Reproducible controls fingerprint (both stages compute it identically
    from the same args, so a stale/mismatched controls artifact is detectable)."""
    return {
        "dev_manifest_sha1": _file_sha1(dev_manifest),
        "selected_a_manifest_sha1": _file_sha1(selected_a_manifest),
        "checkpoint_identity": _checkpoint_identity(checkpoint),
        "mcts_sims": MCTS_SIMS,
        "search_cfg": search_cfg,
        "seeds": seeds,
        "git_commit": _git_commit(),
        "observer_schema_version": OBSERVER_SCHEMA_VERSION,
    }


def _n_visited_children(node: Any) -> int:
    return sum(1 for c in node.children.values() if c.visit_count > 0)


def _visit_entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def _leader_child(root: Any):
    """The visit-leader child (max visit_count, ties by lowest encoded move id)
    -- the same comparator mcts.visit_leader_move uses."""
    visited = [c for c in root.children.values() if c.visit_count > 0]
    if not visited:
        return None
    return min(visited, key=lambda c: (-c.visit_count, c.move))


def _position_features(search_out, observer: FpuTraceObserver) -> dict:
    """Per-position raw features from one (visit_counts, root_value_stm, root)
    search plus its observer trace. `mover value` is root_value_stm (root
    perspective). `replies` = visited children of the visit-leader child."""
    _visit_counts, root_value_stm, root = search_out
    child_counts = [c.visit_count for c in root.children.values() if c.visit_count > 0]
    top = _leader_child(root)
    tshare = top_share(root)
    feats = {
        "root_value_stm": float(root_value_stm),
        "top_move": None if top is None else top.move,
        "top_move_prior": None if (top is None or not root.priors) else root.priors.get(top.move),
        "top_share": tshare,
        "collapsed": tshare >= COLLAPSE_TOP_SHARE,
        "effective_children": math.exp(_visit_entropy(child_counts)) if child_counts else 0.0,
        "replies": 0 if top is None else _n_visited_children(top),
        "trace": observer.result(),
    }
    return feats


def _eff_reduction(ref_eff: float, cand_eff: float) -> float:
    return (ref_eff - cand_eff) / ref_eff if ref_eff > 0 else 0.0


def _dev_target_row(band: str, cand: dict, ref: dict) -> dict:
    """One §6.2 target row: candidate metrics computed VS the reference `ref`."""
    return {
        "role": "target", "band": band,
        "new_collapse": bool(cand["collapsed"] and not ref["collapsed"]),
        "lock_in": lock_in_event(cand["trace"]),
        "mover_delta": cand["root_value_stm"] - ref["root_value_stm"],
        "eff_children_reduction": _eff_reduction(ref["effective_children"],
                                                 cand["effective_children"]),
        "top_share_inc": cand["top_share"] - ref["top_share"],
    }


def _dev_control_row(cand: dict, ref: dict) -> dict:
    """One §6.2 control row: mover Δ + top-move-flip-to-lower-prior flag."""
    flipped_lower = False
    if cand["top_move"] is not None and ref["top_move"] is not None:
        if cand["top_move"] != ref["top_move"]:
            cp, rp = cand["top_move_prior"], ref["top_move_prior"]
            flipped_lower = (cp is not None and rp is not None and cp < rp)
    return {
        "role": "control",
        "mover_delta": cand["root_value_stm"] - ref["root_value_stm"],
        "control_flip_to_lower_prior": flipped_lower,
    }


def _a_row(off: dict, cand: dict) -> dict:
    """One §6.3 selected-A row from the absolute_off run + a candidate run."""
    return {
        "off_value": off["root_value_stm"], "r_value": cand["root_value_stm"],
        "replies_ref": off["replies"], "replies_x": cand["replies"],
        "new_collapse": bool(cand["collapsed"] and not off["collapsed"]),
        "top_share_inc": cand["top_share"] - off["top_share"],
    }


def _write_csv(path: str, fieldnames: Sequence[str], rows: Sequence[Mapping]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Dev-corpus loading + per-config search (operator; lazy heavy imports)
# ---------------------------------------------------------------------------

def _load_dev_rows(dev_manifest: str, mode: str) -> List[dict]:
    """Read the Task-6 dev-corpus manifest and keep only `split == mode` rows."""
    with open(dev_manifest, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("split") == mode]
    if not rows:
        raise SystemExit(f"no rows with split=={mode!r} in {dev_manifest}")
    return rows


def _make_evaluator_and_base_cfg(checkpoint: str, eval_batch_size: int,
                                 stall_flush_sims: int):
    from .eval_runner import EvalConfig, cfg_from, _default_evaluator_factory
    evaluator = _default_evaluator_factory(checkpoint)
    base_cfg = cfg_from(EvalConfig(mcts_sims=MCTS_SIMS,
                                   mcts_eval_batch_size=eval_batch_size,
                                   mcts_stall_flush_sims=stall_flush_sims))
    return evaluator, base_cfg


def _config_for(base_cfg, run_config: FpuRunConfig):
    """Frozen MCTSConfig for one run config: absolute_off => reduction None
    (byte-identical path); r0/grid => the numeric reduction."""
    return dataclasses.replace(base_cfg, fpu_policy_mass_reduction=run_config.reduction)


def _search_position(evaluator, cfg, state, seed):
    """Run one 400-sim search with a fresh FpuTraceObserver. Returns
    (search_out, observer)."""
    import random
    from .mcts import MCTS
    obs = FpuTraceObserver()
    out = MCTS(evaluator, cfg, random.Random(seed), observer=obs).search_with_root(
        state, add_noise=False)
    if out[2].visit_count != MCTS_SIMS:
        raise RuntimeError(f"{out[2].visit_count} sims != {MCTS_SIMS}")
    return out, obs


def _reconstruct_state(row: dict, replay_by_game: Mapping[int, str]):
    """State at a dev-corpus row via the shared reconstruction path."""
    from .goal_line_trigger_probe_cases import position_state
    replay = json.loads(Path(replay_by_game[int(row["game_idx"])]).read_text())
    return position_state(replay, int(float(row["position_ply"])), row["side"])


def _run_configs_over_corpus(dev_rows, run_configs, *, evaluator, base_cfg,
                             replay_by_game, seed_base) -> Dict[str, Dict[str, dict]]:
    """{config_label: {canonical_sha1: features}} over every dev row -- the
    joinable per-position result feeding the gates."""
    out: Dict[str, Dict[str, dict]] = {c.label: {} for c in run_configs}
    for row in dev_rows:
        state = _reconstruct_state(row, replay_by_game)
        seed = _run_seed(seed_base, int(row["game_idx"]), int(float(row["position_ply"])))
        for c in run_configs:
            search_out, obs = _search_position(evaluator, _config_for(base_cfg, c), state, seed)
            out[c.label][row["canonical_position_sha1"]] = _position_features(search_out, obs)
    return out


# ---------------------------------------------------------------------------
# controls stage
# ---------------------------------------------------------------------------

CONTROLS_CASE_FIELDNAMES = [
    "canonical_sha1", "game_idx", "position_ply", "side", "role", "band",
    "config", "root_value_stm", "top_share", "effective_children", "replies",
    "collapsed", "selected_move_prior", "selected_move_prior_rank",
    "explored_mass_at_stabilization", "stabilization_sim", "final_root_top_share",
    "lock_in",
]


def _controls_case_row(row: dict, label: str, feats: dict) -> dict:
    tr = feats["trace"]
    return {
        "canonical_sha1": row["canonical_position_sha1"], "game_idx": row["game_idx"],
        "position_ply": row["position_ply"], "side": row["side"], "role": row["role"],
        "band": row["branching_band"], "config": label,
        "root_value_stm": feats["root_value_stm"], "top_share": feats["top_share"],
        "effective_children": feats["effective_children"], "replies": feats["replies"],
        "collapsed": feats["collapsed"], "selected_move_prior": tr["selected_move_prior"],
        "selected_move_prior_rank": tr["selected_move_prior_rank"],
        "explored_mass_at_stabilization": tr["explored_mass_at_stabilization"],
        "stabilization_sim": tr["stabilization_sim"],
        "final_root_top_share": tr["final_root_top_share"], "lock_in": lock_in_event(tr),
    }


def _dev_rows_vs(target_control_rows, cand_by_sha: Mapping[str, dict],
                 ref_by_sha: Mapping[str, dict]) -> List[dict]:
    """Build §6.2 rows for a candidate config vs a reference, joined by sha1."""
    rows: List[dict] = []
    for r in target_control_rows:
        sha = r["canonical_position_sha1"]
        cand, ref = cand_by_sha.get(sha), ref_by_sha.get(sha)
        if cand is None or ref is None:
            continue
        if r["role"] == "target":
            rows.append(_dev_target_row(r["branching_band"], cand, ref))
        else:
            rows.append(_dev_control_row(cand, ref))
    return rows


def _lockin_count(target_rows, feats_by_sha: Mapping[str, dict]) -> int:
    return sum(1 for r in target_rows if r["role"] == "target"
               and lock_in_event(feats_by_sha[r["canonical_position_sha1"]]["trace"]))


def run_controls_stage(args, dev_rows, run_configs) -> int:
    validate_stage_mode(dev_rows, mode=args.mode, stage="controls", run_configs=run_configs)
    from .build_fpu_dev_corpus import load_game_index
    replay_by_game = {r["game_idx"]: r["replay_path"]
                      for r in load_game_index(args.source_jsonl)}
    evaluator, base_cfg = _make_evaluator_and_base_cfg(
        args.checkpoint, args.eval_batch_size, args.stall_flush_sims)

    by_label = _run_configs_over_corpus(
        dev_rows, run_configs, evaluator=evaluator, base_cfg=base_cfg,
        replay_by_game=replay_by_game, seed_base=args.seed_base)
    off_by_sha, r0_by_sha = by_label[ABSOLUTE_OFF.label], by_label[R0.label]

    # joinable per-position rows for BOTH configs
    case_rows = []
    for row in dev_rows:
        for label in (ABSOLUTE_OFF.label, R0.label):
            case_rows.append(_controls_case_row(row, label, by_label[label][row["canonical_position_sha1"]]))
    out_dir = Path(args.out_dir)
    _write_csv(str(out_dir / "controls_cases.csv"), CONTROLS_CASE_FIELDNAMES, case_rows)

    # r0 must pass the FULL §6.2 table vs absolute_off (control-qualification)
    r0_dev_rows = _dev_rows_vs(dev_rows, r0_by_sha, off_by_sha)
    absoff_lockin = _lockin_count(dev_rows, off_by_sha)
    r0_lockin = _lockin_count(dev_rows, r0_by_sha)
    verdict = dev_safety_verdict(r0_dev_rows, ref=ABSOLUTE_OFF,
                                 r0_lockin=r0_lockin, absoff_lockin=absoff_lockin)
    r0_qualified = not verdict.rejected

    search_cfg = _search_cfg_dict(args.eval_batch_size, args.stall_flush_sims, base_cfg.c_puct)
    fingerprint = _build_fingerprint(
        dev_manifest=args.dev_manifest, selected_a_manifest=args.selected_a_manifest,
        checkpoint=args.checkpoint, search_cfg=search_cfg,
        seeds={"seed_base": args.seed_base, "eval_batch_size": args.eval_batch_size,
               "stall_flush_sims": args.stall_flush_sims})

    _write_csv(str(out_dir / "controls_summary.csv"),
               ["config", "n_positions", "target_lockin_count", "mean_top_share"],
               [{"config": ABSOLUTE_OFF.label, "n_positions": len(off_by_sha),
                 "target_lockin_count": absoff_lockin,
                 "mean_top_share": _mean(f["top_share"] for f in off_by_sha.values())},
                {"config": R0.label, "n_positions": len(r0_by_sha),
                 "target_lockin_count": r0_lockin,
                 "mean_top_share": _mean(f["top_share"] for f in r0_by_sha.values())}])

    gate = {
        "r0_qualified": r0_qualified, "r0_reject_reasons": list(verdict.reasons),
        "r0_target_lockin_count": r0_lockin, "absoff_target_lockin_count": absoff_lockin,
        "mode": args.mode, "n_positions": len(dev_rows), "fingerprint": fingerprint,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "controls_gate.json").write_text(json.dumps(gate, indent=2))
    print(f"[fpu-controls] r0_qualified={r0_qualified} "
          f"(r0_lockin={r0_lockin}, absoff_lockin={absoff_lockin}) "
          f"-> {out_dir/'controls_gate.json'}")
    if not r0_qualified:
        print(f"[fpu-controls] r0 REJECTED vs absolute_off: {list(verdict.reasons)}")
    return 0


# ---------------------------------------------------------------------------
# candidates stage
# ---------------------------------------------------------------------------

def _load_controls_gate(out_dir: str) -> dict:
    return json.loads((Path(out_dir) / "controls_gate.json").read_text())


def run_candidates_stage(args, dev_rows, run_configs) -> int:
    validate_stage_mode(dev_rows, mode=args.mode, stage="candidates", run_configs=run_configs)
    gate = _load_controls_gate(args.out_dir)

    # Recompute the fingerprint from the SAME args; refuse stale/mismatched
    # controls, then require r0 to have qualified (§5 step 0).
    search_cfg = _search_cfg_dict(args.eval_batch_size, args.stall_flush_sims,
                                  gate["fingerprint"].get("search_cfg", {}).get("c_puct"))
    expected = _build_fingerprint(
        dev_manifest=args.dev_manifest, selected_a_manifest=args.selected_a_manifest,
        checkpoint=args.checkpoint, search_cfg=search_cfg,
        seeds={"seed_base": args.seed_base, "eval_batch_size": args.eval_batch_size,
               "stall_flush_sims": args.stall_flush_sims})
    validate_controls_fingerprint(gate, expected)
    require_r0_qualified(gate)
    require_matching_mode(gate, args.mode)

    from .build_fpu_dev_corpus import load_game_index
    replay_by_game = {r["game_idx"]: r["replay_path"]
                      for r in load_game_index(args.source_jsonl)}
    evaluator, base_cfg = _make_evaluator_and_base_cfg(
        args.checkpoint, args.eval_batch_size, args.stall_flush_sims)

    # Rerun both references (joinable controls) + every candidate config.
    ref_configs = [ABSOLUTE_OFF, R0]
    all_configs = ref_configs + list(run_configs)
    by_label = _run_configs_over_corpus(
        dev_rows, all_configs, evaluator=evaluator, base_cfg=base_cfg,
        replay_by_game=replay_by_game, seed_base=args.seed_base)
    off_by_sha, r0_by_sha = by_label[ABSOLUTE_OFF.label], by_label[R0.label]
    r0_lockin = gate["r0_target_lockin_count"]
    absoff_lockin = gate["absoff_target_lockin_count"]

    a_rows_source = _load_selected_a(args, evaluator, base_cfg, run_configs)

    results = []
    smallest_safe = None
    for c in run_configs:
        cand_by_sha = by_label[c.label]
        v_off = dev_safety_verdict(
            _dev_rows_vs(dev_rows, cand_by_sha, off_by_sha), ref=ABSOLUTE_OFF,
            r0_lockin=r0_lockin, absoff_lockin=absoff_lockin)
        v_r0 = dev_safety_verdict(
            _dev_rows_vs(dev_rows, cand_by_sha, r0_by_sha), ref=R0,
            r0_lockin=r0_lockin, absoff_lockin=absoff_lockin)
        a_rows = a_rows_source.get(c.label, [])
        a_verdict = selected_a_verdict(a_rows) if a_rows else None
        safe = (not v_off.rejected and not v_r0.rejected
                and a_verdict is not None and a_verdict.passed)
        results.append({
            "config": c.label, "reduction": c.reduction, "safe": safe,
            "reject_vs_absolute_off": list(v_off.reasons),
            "reject_vs_r0": list(v_r0.reasons),
            "selected_a_passed": None if a_verdict is None else a_verdict.passed,
        })
        if safe and smallest_safe is None:
            smallest_safe = c.label

    out = {"mode": args.mode, "smallest_safe_r": smallest_safe, "candidates": results}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "candidates_result.json").write_text(json.dumps(out, indent=2))
    print(f"[fpu-candidates] smallest_safe_r={smallest_safe} -> "
          f"{Path(args.out_dir)/'candidates_result.json'}")
    return 0


def _load_selected_a(args, evaluator, base_cfg, run_configs):
    """{config_label: [A rows]} over the selected-A manifest (30 roots), each
    config vs absolute_off. Returns {} when no selected-A manifest is given."""
    if not args.selected_a_manifest:
        return {}
    from .position_probe_cases import load_csv_manifest
    from .goal_line_trigger_probe_cases import position_state
    cases = load_csv_manifest(args.selected_a_manifest)["cases"]
    per_config: Dict[str, List[dict]] = {c.label: [] for c in run_configs}
    for case in cases:
        replay = json.loads(Path(case["replay_path"]).read_text())
        state = position_state(replay, int(case["position_ply"]), case["side_to_move"])
        seed = _run_seed(args.seed_base, int(case["game_idx"]), int(case["position_ply"]))
        off_out, off_obs = _search_position(
            evaluator, _config_for(base_cfg, ABSOLUTE_OFF), state, seed)
        off_feat = _position_features(off_out, off_obs)
        for c in run_configs:
            cand_out, cand_obs = _search_position(
                evaluator, _config_for(base_cfg, c), state, seed)
            per_config[c.label].append(_a_row(off_feat, _position_features(cand_out, cand_obs)))
    return per_config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _configs_for_stage_mode(mode: str, stage: str, frozen_r: Optional[float]) -> List[FpuRunConfig]:
    """The canonical config set for a (mode, stage). frozen_check+candidates
    needs the single nonzero r via --frozen-r."""
    if stage == "controls":
        return [ABSOLUTE_OFF, R0]
    if mode == "tuning":
        return list(GRID)
    if frozen_r is None:
        raise SystemExit("frozen_check candidates requires --frozen-r <nonzero r>")
    return [FpuRunConfig(f"r{frozen_r}", frozen_r)]


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description="Context-relative FPU (policy-mass) staged discovery "
                    "diagnostic. OPERATOR phase: loads a real checkpoint and "
                    "runs 400-sim MCTS on the dev corpus. `controls` runs "
                    "{absolute_off, r0} and writes joinable fingerprinted "
                    "controls; `candidates` validates/joins them and reports "
                    "the smallest safe-passing r. design §4/§5/§6.")
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument("--dev-manifest", default=DEFAULT_DEV_MANIFEST)
    ap.add_argument("--source-jsonl", default=None,
                    help="replay index (game_idx -> replay_path) for state "
                         "reconstruction; defaults to build_fpu_dev_corpus's source.")
    ap.add_argument("--selected-a-manifest", default=None,
                    help="30-root selected-A probe manifest for the §6.3 gate.")
    ap.add_argument("--checkpoint", default=None,
                    help="defaults to diagnose_fpu_sweep.DEFAULT_CHECKPOINT.")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--frozen-r", type=float, default=None,
                    help="the single nonzero r for frozen_check candidates.")
    ap.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    ap.add_argument("--eval-batch-size", type=int, default=14)
    ap.add_argument("--stall-flush-sims", type=int, default=48)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    # Deferred so the pure-test import path stays MCTS/GPU/MLX-free.
    from .diagnose_fpu_sweep import DEFAULT_CHECKPOINT
    from .build_fpu_dev_corpus import DEFAULT_SOURCE_JSONL

    args = _parse_args(argv)
    args.checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
    args.source_jsonl = args.source_jsonl or DEFAULT_SOURCE_JSONL

    dev_rows = _load_dev_rows(args.dev_manifest, args.mode)
    run_configs = _configs_for_stage_mode(args.mode, args.stage, args.frozen_r)

    if args.stage == "controls":
        return run_controls_stage(args, dev_rows, run_configs)
    return run_candidates_stage(args, dev_rows, run_configs)


if __name__ == "__main__":
    raise SystemExit(main())
