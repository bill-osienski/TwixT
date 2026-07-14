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
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Stdlib-only provenance helpers (design §12.5). Importing it keeps this module
# GPU/MLX-free (fpu_provenance imports no MCTS/evaluator/mlx).
from . import fpu_provenance

# The effective result-determining source files whose BYTES the shared
# selection-context fingerprint pins (design §12.5). Referenced by PATH (from
# this package dir) so fingerprinting mcts.py never requires importing it --
# which would pull GPU/MLX -- keeping the pure import path clean. Includes the
# state-RECONSTRUCTION deps (goal_line_trigger_probe_cases.py +
# game/twixt_state.py): the diagnostic rebuilds every searched position via
# position_state -> TwixtState, so their bytes are as result-determining as the
# search core, and the builder already hashes goal_line_trigger_probe_cases.py
# (RF1; kept aligned with build_fpu_dev_corpus._CORPUS_SOURCES).
_MODULE_DIR = Path(__file__).resolve().parent
RESULT_DETERMINING_SOURCES: Tuple[Path, ...] = (
    _MODULE_DIR / "diagnose_fpu_policy_mass.py",
    _MODULE_DIR / "mcts.py",
    _MODULE_DIR / "build_fpu_dev_corpus.py",
    _MODULE_DIR / "goal_line_trigger_probe_cases.py",
    _MODULE_DIR / "game" / "twixt_state.py",
)

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
    # #4a: every number the §6.2 gate ALREADY computes, exposed for the persisted
    # candidate artifacts. `compare=False` keeps the frozen dataclass hashable and
    # leaves eq keyed on (rejected, reasons) -- so exposing metrics changes NO gate
    # semantics. The gate LOGIC (thresholds/comparators/AND-OR -> rejected/reasons)
    # is untouched; metrics is written where each value is already computed.
    metrics: Mapping[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class AVerdict:
    passed: bool
    reply_reduction: float
    progress: float
    a_new_collapse: int
    a_top_share_inc: float


def dev_safety_verdict(rows: Sequence[Mapping[str, Any]], ref: FpuRunConfig,
                       r0_lockin: int, absoff_lockin: int, *,
                       stratum_key: str = "band") -> SafetyVerdict:
    """§6.2 development-safety verdict vs a single reference `X` (`ref`). REJECT
    (rejected=True) if ANY frozen gate trips. Target and control rows are split
    by `role`; each subset is evaluated only when non-empty (so a test can
    isolate one gate by supplying just the rows it needs). The lock-in baseline
    follows `ref`: `r0_lockin` when ref is r0 (reduction 0.0), else
    `absoff_lockin` (ref absolute_off / reduction None).

    `stratum_key` picks the per-stratum new-collapse sub-gate's grouping key.
    Default `"band"` reproduces v1 byte-for-byte (reason string
    `band[{value}]_new_collapse=...`, metrics key `band_new_collapse_rates`) --
    no existing call site passes this kwarg. Passing `stratum_key="ply_bucket"`
    (v2's opt-in) renames both the reason prefix and the GATED metrics key to
    `ply_bucket`; band rates are THEN ALSO recorded in
    `metrics["band_new_collapse_rates"]` as a report-only, ungated entry (same
    n>=DEV_BAND_MIN_N rule) so band coverage stays visible when phase is the
    gated stratum. `DEV_NEW_COLLAPSE_BAND` (10%) and `DEV_BAND_MIN_N` (20) are
    unchanged either way. A `target` row missing a key needed for grouping
    raises `ValueError` (never a silent skip)."""
    target = [r for r in rows if r.get("role") == "target"]
    control = [r for r in rows if r.get("role") == "control"]
    reasons: List[str] = []
    # #4a: expose the SAME numbers the gate computes below (populated at each
    # computation point). Adding these lines changes NO threshold/comparator/
    # AND-OR and NO reason string, so `rejected`/`reasons` stay byte-identical.
    metrics: Dict[str, Any] = {}

    def _new_collapse_rates_by(key: str) -> Dict[Any, float]:
        """n>=DEV_BAND_MIN_N new-collapse rate per distinct `target` row[key]
        value, insertion-ordered by first occurrence -- for key=="band" this is
        byte-identical to the pre-stratum-parameterization band-only loop."""
        grouped: Dict[Any, List] = defaultdict(list)
        for r in target:
            if key not in r:
                raise ValueError(
                    f"dev_safety_verdict: target row missing required "
                    f"stratum key {key!r} (row keys: {sorted(r.keys())})")
            grouped[r[key]].append(r)
        rates: Dict[Any, float] = {}
        for value, srows in grouped.items():
            if len(srows) >= DEV_BAND_MIN_N:
                rates[value] = sum(1 for r in srows if r["new_collapse"]) / len(srows)
        return rates

    if target:
        n = len(target)
        nc_rate = sum(1 for r in target if r["new_collapse"]) / n
        metrics["target_new_collapse_rate"] = nc_rate
        if nc_rate >= DEV_NEW_COLLAPSE_TARGET:
            reasons.append(f"target_new_collapse_rate={nc_rate:.4f}>={DEV_NEW_COLLAPSE_TARGET}")

        stratum_rates = _new_collapse_rates_by(stratum_key)
        for value, srate in stratum_rates.items():
            if srate >= DEV_NEW_COLLAPSE_BAND:
                reasons.append(f"{stratum_key}[{value}]_new_collapse={srate:.4f}>={DEV_NEW_COLLAPSE_BAND}")
        metrics[f"{stratum_key}_new_collapse_rates"] = stratum_rates
        if stratum_key != "band":
            # Report-only, ungated -- keeps band coverage visible when phase
            # (ply_bucket) is the gated stratum. Never feeds `reasons`.
            metrics["band_new_collapse_rates"] = _new_collapse_rates_by("band")

        baseline = r0_lockin if getattr(ref, "reduction", None) == 0.0 else absoff_lockin
        lockins = sum(1 for r in target if r["lock_in"])
        metrics["target_lockin_count"] = lockins
        metrics["lockin_baseline"] = baseline
        if lockins > baseline + DEV_LOCKIN_MARGIN:
            reasons.append(f"lockin_count={lockins}>baseline+2={baseline + DEV_LOCKIN_MARGIN}")

        p95 = _percentile([abs(r["mover_delta"]) for r in target], PERCENTILE_Q)
        metrics["target_p95_mover_delta"] = p95
        if p95 >= DEV_P95_MOVER:
            reasons.append(f"target_p95_mover_delta={p95:.4f}>={DEV_P95_MOVER}")

        eff = _mean(r["eff_children_reduction"] for r in target)
        tsi = _mean(r["top_share_inc"] for r in target)
        metrics["mean_eff_children_reduction"] = eff
        metrics["mean_top_share_increase"] = tsi
        if eff >= DEV_COMPOUND_EFF and tsi >= DEV_COMPOUND_TOPSHARE:
            reasons.append(f"compound eff_reduction={eff:.4f}&top_share_inc={tsi:.4f}")

    if control:
        flip = sum(1 for r in control if r["control_flip_to_lower_prior"]) / len(control)
        metrics["control_flip_rate"] = flip
        if flip >= DEV_CONTROL_FLIP:
            reasons.append(f"control_flip_rate={flip:.4f}>={DEV_CONTROL_FLIP}")
        cp95 = _percentile([abs(r["mover_delta"]) for r in control], PERCENTILE_Q)
        metrics["control_p95_mover_delta"] = cp95
        if cp95 >= DEV_CONTROL_P95:
            reasons.append(f"control_p95_mover_delta={cp95:.4f}>={DEV_CONTROL_P95}")

    return SafetyVerdict(rejected=bool(reasons), reasons=tuple(reasons), metrics=metrics)


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

def _canonical(obj: Any) -> Any:
    """JSON-normalize (tuples -> lists, key order irrelevant) so an in-memory
    fingerprint block compares equal to the JSON-persisted one field-for-field.
    All fingerprint values are JSON-native today; this is cheap insurance
    against a future tuple field silently failing a legitimate match."""
    return json.loads(json.dumps(obj, sort_keys=True))


def validate_controls_fingerprint(gate_json: Mapping[str, Any],
                                  expected: Mapping[str, Any]) -> None:
    """Raise ValueError unless the controls artifact's SELECTION-CONTEXT
    fingerprint matches `expected` (the shared, result-determining identity).
    ONLY `selection_context` is compared: a differing `run_context` -- e.g.
    selected-A present in tuning vs absent in frozen_check, a different stage,
    or git/runtime provenance -- must NOT fail the join (design §12.2). A
    stale/mismatched selection_context => the candidate stage refuses to reuse
    the controls. `expected` may be a full fingerprint or a bare
    selection_context block. Pure/testable with fabricated dicts."""
    fp = gate_json.get("fingerprint")
    if fp is None:
        raise ValueError("controls_gate.json has no 'fingerprint' block")
    got = fp.get("selection_context")
    if got is None:
        raise ValueError("controls_gate.json fingerprint has no 'selection_context' block")
    exp = expected.get("selection_context", expected)
    got_n, exp_n = _canonical(got), _canonical(exp)
    if got_n != exp_n:
        mismatched = sorted(k for k in set(got_n) | set(exp_n) if got_n.get(k) != exp_n.get(k))
        raise ValueError(
            f"stale/mismatched controls selection-context (differing keys: {mismatched})")


def require_frozen_matches_tuning(tuning_result: Mapping[str, Any], *,
                                  frozen_reduction: Optional[float],
                                  expected_selection_context: Mapping[str, Any]) -> None:
    """§12.2 immutable frozen coefficient. The frozen_check candidate MUST equal
    the coefficient the tuning split selected. Raise ValueError unless ALL hold:
      (a) `tuning_result['mode'] == 'tuning'` (frozen must consume a TUNING
          selection artifact, not another frozen one);
      (b) `tuning_result['smallest_safe_r']` is non-null (a coefficient WAS
          selected -- otherwise there is nothing to lock to);
      (c) that result's `fingerprint.selection_context` equals
          `expected_selection_context` (the frozen stage's OWN shared identity)
          exactly -- same checkpoint / manifest / base config / source;
      (d) `frozen_reduction` equals the reduction the selected label maps to via
          GRID -- so an arbitrary `--frozen-r` cannot be smuggled past
          "one nonzero r". Pure -- testable with a fabricated tuning_result."""
    if tuning_result.get("mode") != "tuning":
        raise ValueError(
            f"tuning-result mode {tuning_result.get('mode')!r} != 'tuning' "
            "(frozen_check must consume a TUNING selection artifact)")
    selected = tuning_result.get("smallest_safe_r")
    if selected is None:
        raise ValueError(
            "tuning-result smallest_safe_r is null -- no coefficient was selected; "
            "frozen_check has nothing to lock to")
    fp = tuning_result.get("fingerprint") or {}
    tun_sel = fp.get("selection_context")
    if tun_sel is None:
        raise ValueError("tuning-result fingerprint has no 'selection_context' block")
    if _canonical(tun_sel) != _canonical(expected_selection_context):
        raise ValueError(
            "frozen_check selection-context != tuning selection-context "
            "(different checkpoint/manifest/config/source -- refusing to lock)")
    grid_reduction = {c.label: c.reduction for c in GRID}
    if selected not in grid_reduction:
        raise ValueError(f"tuning smallest_safe_r {selected!r} is not a GRID label")
    if frozen_reduction != grid_reduction[selected]:
        raise ValueError(
            f"--frozen-r reduction {frozen_reduction!r} != tuning-selected "
            f"{selected!r} reduction {grid_reduction[selected]!r} (immutable §12.2)")


def validate_selected_a_mode(mode: str, has_selected_a: bool) -> None:
    """§12.3 selected-A is tuning-only. Raise SystemExit if tuning candidates
    lack a selected-A manifest (the §6.3 mechanism gate is mandatory there) or
    if frozen_check is given one (frozen is a held-out dev screen; its `safe`
    verdict must not depend on selected-A). Pure -- extracted so the guard is
    testable without invoking the operator stage."""
    if mode == "tuning" and not has_selected_a:
        raise SystemExit(
            "tuning candidates require --selected-a-manifest (the §6.3 mechanism gate)")
    if mode == "frozen_check" and has_selected_a:
        raise SystemExit(
            "selected-A is tuning-only; do not pass --selected-a-manifest in "
            "frozen_check mode")


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


def _checkpoint_identity(checkpoint: str) -> str:
    return f"{Path(checkpoint).name}:{fpu_provenance.file_sha1(checkpoint)}"


def _run_seed(seed_base: int, game_idx: int, ply: int) -> int:
    """Deterministic per-position seed (same row_seed idiom as build_fpu_dev_corpus)."""
    return int(seed_base) ^ int(game_idx) ^ int(ply)


def build_run_fingerprint(*, dev_manifest: str, checkpoint: str, base_cfg: Any,
                          source_jsonl: Optional[str], replay_paths: Sequence[str],
                          seeds: Mapping[str, Any], selected_a_manifest: Optional[str],
                          mode: str, stage: str) -> dict:
    """Split run-identity fingerprint (design §12.2/§12.5), computed identically
    from the same args in every stage. TWO blocks:

    `selection_context` -- the SHARED, result-determining identity every stage of
    one protocol run (and tuning vs frozen) must match EXACTLY:
      - source-file BYTE hashes of the effective result-determining modules
        (§12.5: a git commit alone misses uncommitted edits);
      - checkpoint identity (name + sha1); dev-manifest sha1;
      - source-index sha1 + a deterministic replay-DATA hash (contents, not
        paths);
      - the FULL effective base MCTS config via `dataclasses.asdict` (this base
        has `fpu_policy_mass_reduction=None`; it captures c_puct / eval_batch_size
        / stall_flush_sims / n_simulations / ...);
      - mcts_sims; seeds; the frozen GRID as `[[label, reduction], ...]`.

    `run_context` -- RECORDED but NOT cross-matched (it legitimately differs):
    selected-A present (tuning) vs absent (frozen); explicit `add_noise=False`;
    git commit + clean-worktree flag; runtime provenance; mode; stage; observer
    schema version."""
    selection_context = {
        "source_file_sha1s": fpu_provenance.source_file_sha1s(RESULT_DETERMINING_SOURCES),
        "checkpoint_identity": _checkpoint_identity(checkpoint),
        "dev_manifest_sha1": fpu_provenance.file_sha1(dev_manifest),
        "source_index_sha1": fpu_provenance.file_sha1(source_jsonl),
        "replay_data_sha1": fpu_provenance.replay_data_sha1(replay_paths),
        "base_mcts_config": dataclasses.asdict(base_cfg),
        "mcts_sims": MCTS_SIMS,
        "seeds": dict(seeds),
        "grid": [[c.label, c.reduction] for c in GRID],
    }
    run_context = {
        "selected_a": {
            "present": bool(selected_a_manifest),
            "manifest_sha1": (fpu_provenance.file_sha1(selected_a_manifest)
                              if selected_a_manifest else None),
        },
        "add_noise": False,
        "git_commit": fpu_provenance.git_commit(),
        "worktree_clean": fpu_provenance.worktree_clean(),
        "runtime_provenance": fpu_provenance.runtime_provenance(),
        "mode": mode,
        "stage": stage,
        "observer_schema_version": OBSERVER_SCHEMA_VERSION,
    }
    return {"selection_context": selection_context, "run_context": run_context}


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


def _read_csv_rows(path: str) -> List[dict]:
    """Read a CSV written by `_write_csv` back as a list of dict rows (every cell
    is a string -- csv-native)."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _controls_cell(value: Any) -> str:
    """The EXACT text `csv.writer` emits for a `_controls_case_row` cell: ``None``
    -> ``""`` (empty), everything else -> ``str(value)``. Used both when a value
    is persisted and when the recompute is verified against the persisted CSV
    (§12.4), so a faithful round-trip compares byte-equal. Floats round-trip
    EXACTLY: Python 3's ``str(float)`` is the shortest round-tripping repr and is
    injective over finite floats, so equal cell text <=> bit-identical value (a
    1-ULP difference yields different text and is caught)."""
    return "" if value is None else str(value)


def verify_recomputed_controls(
        persisted_rows: Sequence[Mapping[str, Any]],
        recomputed_by_config: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> Tuple[int, str]:
    """§12.4 VERIFIED recompute. The candidate stage recomputes absolute_off/r0
    (the persisted controls_cases.csv omits top_move/top_move_prior the low-prior-
    flip gate needs); this proves the recompute is bit-identical to what was
    persisted rather than merely asserting it.

    `persisted_rows`: controls_cases.csv rows read back (csv-native strings).
    `recomputed_by_config`: ``{config_label: {canonical_sha1: recomputed_case_row}}``
    where each recomputed row is a freshly-built `_controls_case_row` (native
    types). For EVERY persisted row, look up the recompute by (config,
    canonical_sha1) and compare EVERY persisted field against the canonicalized
    recomputed value (`_controls_cell`); RAISE ValueError on a missing recompute,
    a missing field, or any mismatch. Returns ``(count, rows_sha1)`` -- the number
    of rows verified and a deterministic SHA1 over the compared rows (sorted by
    (canonical_sha1, config), fields in CONTROLS_CASE_FIELDNAMES order). Pure --
    unit-testable with fabricated dicts / temp files."""
    verified: List[Mapping[str, Any]] = []
    for prow in persisted_rows:
        config = prow.get("config")
        sha = prow.get("canonical_sha1")
        by_sha = recomputed_by_config.get(config)
        if by_sha is None or sha not in by_sha:
            raise ValueError(
                "verify_recomputed_controls: no recompute for "
                f"(config={config!r}, canonical_sha1={sha!r})")
        rec = by_sha[sha]
        for k, persisted_v in prow.items():
            if k not in rec:
                raise ValueError(
                    f"verify_recomputed_controls: recomputed row missing field {k!r} "
                    f"for (config={config!r}, canonical_sha1={sha!r})")
            rec_text = _controls_cell(rec[k])
            if str(persisted_v) != rec_text:
                raise ValueError(
                    f"verify_recomputed_controls: field {k!r} mismatch for "
                    f"(config={config!r}, canonical_sha1={sha!r}): persisted "
                    f"{persisted_v!r} != recomputed {rec_text!r}")
        verified.append(prow)

    verified.sort(key=lambda r: (str(r.get("canonical_sha1", "")), str(r.get("config", ""))))
    payload = json.dumps(
        [[str(r.get(fn, "")) for fn in CONTROLS_CASE_FIELDNAMES] for r in verified],
        separators=(",", ":"))
    return len(verified), hashlib.sha1(payload.encode("utf-8")).hexdigest()


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


def _make_base_cfg(eval_batch_size: int, stall_flush_sims: int):
    """Config-only half of `_make_evaluator_and_base_cfg` -- no checkpoint/
    evaluator load, so it's cheap enough to call before fingerprint
    validation (e.g. to derive the candidate stage's fingerprint c_puct
    independently, rather than echoing it from the loaded controls gate)."""
    from .eval_runner import EvalConfig, cfg_from
    return cfg_from(EvalConfig(mcts_sims=MCTS_SIMS,
                               mcts_eval_batch_size=eval_batch_size,
                               mcts_stall_flush_sims=stall_flush_sims))


def _make_evaluator_and_base_cfg(checkpoint: str, eval_batch_size: int,
                                 stall_flush_sims: int):
    from .eval_runner import _default_evaluator_factory
    evaluator = _default_evaluator_factory(checkpoint)
    base_cfg = _make_base_cfg(eval_batch_size, stall_flush_sims)
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


# #4c: per-position candidate-vs-reference rows persisted for audit -- the SAME
# _dev_target_row/_dev_control_row metrics `_dev_rows_vs` feeds the gate, tagged
# with canonical_sha1 + candidate config + reference so candidate_dev_rows.csv
# joins to controls_cases.csv (and across both references) by canonical_sha1.
# Target and control rows share a unioned schema; a column that doesn't apply to
# a role is written as "" (empty).
CANDIDATE_DEV_ROW_FIELDNAMES = [
    "canonical_sha1", "candidate_config", "reference", "role", "band",
    "new_collapse", "lock_in", "mover_delta", "eff_children_reduction",
    "top_share_inc", "control_flip_to_lower_prior",
]


def _candidate_dev_records(target_control_rows, cand_by_sha: Mapping[str, dict],
                           ref_by_sha: Mapping[str, dict], cand_label: str,
                           ref_label: str) -> List[dict]:
    """Joinable persistence rows mirroring `_dev_rows_vs` EXACTLY (same join, same
    skip condition, same _dev_target_row/_dev_control_row), tagged for audit. Does
    NOT feed any gate -- purely the persisted evidence trail."""
    records: List[dict] = []
    for r in target_control_rows:
        sha = r["canonical_position_sha1"]
        cand, ref = cand_by_sha.get(sha), ref_by_sha.get(sha)
        if cand is None or ref is None:
            continue
        base = {"canonical_sha1": sha, "candidate_config": cand_label,
                "reference": ref_label}
        if r["role"] == "target":
            tr = _dev_target_row(r["branching_band"], cand, ref)
            records.append({**base, "role": "target", "band": tr["band"],
                            "new_collapse": tr["new_collapse"], "lock_in": tr["lock_in"],
                            "mover_delta": tr["mover_delta"],
                            "eff_children_reduction": tr["eff_children_reduction"],
                            "top_share_inc": tr["top_share_inc"],
                            "control_flip_to_lower_prior": ""})
        else:
            cr = _dev_control_row(cand, ref)
            records.append({**base, "role": "control", "band": "",
                            "new_collapse": "", "lock_in": "",
                            "mover_delta": cr["mover_delta"],
                            "eff_children_reduction": "", "top_share_inc": "",
                            "control_flip_to_lower_prior": cr["control_flip_to_lower_prior"]})
    return records


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

    fingerprint = build_run_fingerprint(
        dev_manifest=args.dev_manifest, checkpoint=args.checkpoint, base_cfg=base_cfg,
        source_jsonl=args.source_jsonl, replay_paths=list(replay_by_game.values()),
        seeds={"seed_base": args.seed_base, "eval_batch_size": args.eval_batch_size,
               "stall_flush_sims": args.stall_flush_sims},
        selected_a_manifest=args.selected_a_manifest, mode=args.mode, stage="controls")

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


def _load_tuning_result(path: str) -> dict:
    """Load the tuning `candidates_result.json` whose selected `smallest_safe_r`
    a frozen_check run must lock to (§12.2). A frozen/non-tuning artifact is
    caught downstream by `require_frozen_matches_tuning`."""
    return json.loads(Path(path).read_text())


# #4c: the selected-A case rows (`_a_row` outputs) persisted per config (tuning
# only) -- tuning evidence for the §6.3 mechanism gate.
SELECTED_A_CASE_FIELDNAMES = [
    "config", "off_value", "r_value", "replies_ref", "replies_x",
    "new_collapse", "top_share_inc",
]


def _a_metrics(a_verdict: Optional[AVerdict]) -> Optional[dict]:
    """The §6.3 AVerdict numbers as a JSON-native dict (None when selected-A did
    not participate -- e.g. frozen_check)."""
    if a_verdict is None:
        return None
    return {"reply_reduction": a_verdict.reply_reduction, "progress": a_verdict.progress,
            "a_new_collapse": a_verdict.a_new_collapse,
            "a_top_share_inc": a_verdict.a_top_share_inc, "passed": a_verdict.passed}


def _candidate_result_record(config: FpuRunConfig, v_off: SafetyVerdict,
                             v_r0: SafetyVerdict, a_verdict: Optional[AVerdict],
                             safe: bool) -> dict:
    """#4c: one candidate's `candidates_result.json` record -- the pass/fail
    reasons AND the FULL numeric §6.2/§6.3 gate summaries (both references). Pure:
    reads only the verdicts' exposed numbers, so `safe` (computed by the caller
    with the UNCHANGED gate logic) is passed in, not recomputed here."""
    return {
        "config": config.label, "reduction": config.reduction, "safe": safe,
        "reject_vs_absolute_off": list(v_off.reasons),
        "reject_vs_r0": list(v_r0.reasons),
        "selected_a_passed": None if a_verdict is None else a_verdict.passed,
        "metrics_vs_absolute_off": dict(v_off.metrics),
        "metrics_vs_r0": dict(v_r0.metrics),
        "selected_a_metrics": _a_metrics(a_verdict),
    }


def run_candidates_stage(args, dev_rows, run_configs) -> int:
    validate_stage_mode(dev_rows, mode=args.mode, stage="candidates", run_configs=run_configs)
    # §12.3 selected-A is tuning-only -- checked BEFORE any load (tuning REQUIRES
    # it for the §6.3 mechanism gate; frozen_check FORBIDS it).
    validate_selected_a_mode(args.mode, bool(args.selected_a_manifest))
    gate = _load_controls_gate(args.out_dir)

    # Load the replay index up-front (a pure jsonl read -- no evaluator): the
    # shared selection-context fingerprint hashes the replay DATA, so the paths
    # are needed BEFORE fingerprint validation.
    from .build_fpu_dev_corpus import load_game_index
    replay_by_game = {r["game_idx"]: r["replay_path"]
                      for r in load_game_index(args.source_jsonl)}

    # Recompute OUR selection-context from the SAME args; refuse a
    # stale/mismatched controls artifact, then require r0 to have qualified
    # (§5 step 0) and a matching mode. base_cfg_for_fp is config-only (no
    # checkpoint/evaluator load), so this stays cheap to compute before
    # validation, and it is the SAME EvalConfig the search below uses -- so the
    # fingerprint's base_mcts_config exactly matches the search config.
    base_cfg_for_fp = _make_base_cfg(args.eval_batch_size, args.stall_flush_sims)
    expected_fp = build_run_fingerprint(
        dev_manifest=args.dev_manifest, checkpoint=args.checkpoint,
        base_cfg=base_cfg_for_fp, source_jsonl=args.source_jsonl,
        replay_paths=list(replay_by_game.values()),
        seeds={"seed_base": args.seed_base, "eval_batch_size": args.eval_batch_size,
               "stall_flush_sims": args.stall_flush_sims},
        selected_a_manifest=args.selected_a_manifest, mode=args.mode, stage="candidates")
    validate_controls_fingerprint(gate, expected_fp)
    require_r0_qualified(gate)
    require_matching_mode(gate, args.mode)

    # §12.2 immutable frozen coefficient: the single --frozen-r MUST equal the
    # coefficient the tuning split selected, under a matching selection-context.
    # ("One nonzero r" alone would permit an arbitrary value.)
    if args.mode == "frozen_check":
        tuning_result = _load_tuning_result(
            args.tuning_result or str(Path(args.out_dir) / "candidates_result.json"))
        require_frozen_matches_tuning(
            tuning_result, frozen_reduction=run_configs[0].reduction,
            expected_selection_context=expected_fp["selection_context"])

    evaluator, base_cfg = _make_evaluator_and_base_cfg(
        args.checkpoint, args.eval_batch_size, args.stall_flush_sims)

    # References are recomputed here rather than joined from the persisted
    # controls_cases.csv: CONTROLS_CASE_FIELDNAMES omits top_move/
    # top_move_prior, which _position_features produces and the control-flip
    # gate (_dev_control_row's control_flip_to_lower_prior, consumed by
    # dev_safety_verdict) needs -- so the persisted CSV can't feed that gate.
    # The fingerprint validated above pins determinism, so these recomputed
    # reference features are bit-identical to the persisted ones for the
    # fields the two do share.
    ref_configs = [ABSOLUTE_OFF, R0]
    all_configs = ref_configs + list(run_configs)
    by_label = _run_configs_over_corpus(
        dev_rows, all_configs, evaluator=evaluator, base_cfg=base_cfg,
        replay_by_game=replay_by_game, seed_base=args.seed_base)
    off_by_sha, r0_by_sha = by_label[ABSOLUTE_OFF.label], by_label[R0.label]
    r0_lockin = gate["r0_target_lockin_count"]
    absoff_lockin = gate["absoff_target_lockin_count"]

    # #4b VERIFIED recompute (§12.4): rebuild the controls_cases rows from THESE
    # recomputed reference features and compare every field to the persisted CSV,
    # aborting on any mismatch -- so "recompute == join" is proven, not asserted.
    persisted_controls = _read_csv_rows(str(Path(args.out_dir) / "controls_cases.csv"))
    recomputed_controls = {ABSOLUTE_OFF.label: {}, R0.label: {}}
    for row in dev_rows:
        sha = row["canonical_position_sha1"]
        for label in (ABSOLUTE_OFF.label, R0.label):
            recomputed_controls[label][sha] = _controls_case_row(row, label, by_label[label][sha])
    controls_verified_count, controls_verified_rows_sha1 = verify_recomputed_controls(
        persisted_controls, recomputed_controls)

    # §12.3: selected-A participates ONLY in tuning (in frozen_check it is
    # forbidden above, and the frozen `safe` verdict must not depend on it).
    a_rows_source = (_load_selected_a(args, evaluator, base_cfg, run_configs)
                     if args.mode == "tuning" else {})

    results = []
    smallest_safe = None
    smallest_safe_reduction = None
    candidate_dev_records: List[dict] = []          # #4c joinable per-position rows
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
        dev_safe = not v_off.rejected and not v_r0.rejected
        # tuning: dev-safety (both refs) AND the §6.3 selected-A mechanism gate.
        # frozen_check: a held-out dev screen -- dev-safety ONLY (§12.3).
        if args.mode == "tuning":
            safe = dev_safe and a_verdict is not None and a_verdict.passed
        else:
            safe = dev_safe
        results.append(_candidate_result_record(c, v_off, v_r0, a_verdict, safe))
        candidate_dev_records += _candidate_dev_records(
            dev_rows, cand_by_sha, off_by_sha, c.label, ABSOLUTE_OFF.label)
        candidate_dev_records += _candidate_dev_records(
            dev_rows, cand_by_sha, r0_by_sha, c.label, R0.label)
        if safe and smallest_safe is None:
            smallest_safe = c.label
            smallest_safe_reduction = c.reduction

    # #4c: persist the complete evidence trail (joinable per-position rows +
    # selected-A cases) alongside the numeric-summary candidates_result.json.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(str(out_dir / "candidate_dev_rows.csv"),
               CANDIDATE_DEV_ROW_FIELDNAMES, candidate_dev_records)
    if a_rows_source:                               # tuning only
        _write_csv(str(out_dir / "selected_a_cases.csv"), SELECTED_A_CASE_FIELDNAMES,
                   [{"config": label, **ar}
                    for label, arows in a_rows_source.items() for ar in arows])

    out = {"mode": args.mode, "smallest_safe_r": smallest_safe,
           "smallest_safe_reduction": smallest_safe_reduction,
           "candidates": results, "fingerprint": expected_fp,
           "controls_provenance": "recomputed_and_verified",
           "controls_verified_row_count": controls_verified_count,
           "controls_verified_rows_sha1": controls_verified_rows_sha1}
    (out_dir / "candidates_result.json").write_text(json.dumps(out, indent=2))
    print(f"[fpu-candidates] smallest_safe_r={smallest_safe} -> "
          f"{out_dir/'candidates_result.json'} "
          f"(controls verified: {controls_verified_count} rows)")
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
    ap.add_argument("--tuning-result", default=None,
                    help="frozen_check candidates only: path to the TUNING "
                         "candidates_result.json whose selected smallest_safe_r "
                         "this frozen run must match (§12.2 immutable "
                         "coefficient). Defaults to "
                         "<out_dir>/candidates_result.json.")
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
