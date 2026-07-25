"""v17 paired diagnostic -- modes, pure gates, and coefficient selection.

Task 5 of the v17 plan. The v16 diagnostic
(`diagnose_fpu_policy_mass.py`) already contains every metric definition and
threshold this stage needs, so they are IMPORTED here, never restated. The only
change made to that module is a backward-compatible keyword parameterization of
`dev_safety_verdict` (`stratum_gate`, `lockin_margin`), proved byte-identical
against 66 pre-change fixtures.

Frozen design ref:
`docs/superpowers/specs/2026-07-24-v17-baseline-preserving-policy-mass-fpu-design.md`
(SHA-1 `944f358c0e3ef66503d2cbb56e31dabd145bafc2`) §§7-9.

Layering, so every gate is testable from fabricated rows with no GPU:
  * pure row/gate/selection functions -- this file's bulk;
  * a thin execution shim that reuses the v16 search helpers;
  * a CLI that validates the protocol before anything is loaded.

Import-pure: no MLX at module import. The evaluator import is lazy, inside
`run_positions`.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import fpu_v17_protocol as protocol
from . import fpu_v17_provenance as prov
# Imported, never copied (plan Task 5): the frozen metric definitions and every
# gate threshold. A second definition here would be a second source of truth.
from .diagnose_fpu_policy_mass import (
    A_NEW_COLLAPSE_MAX,
    A_PROGRESS,
    A_REPLY_REDUCTION,
    A_TOPSHARE_MAX,
    COLLAPSE_TOP_SHARE,
    DEV_COMPOUND_EFF,
    DEV_COMPOUND_TOPSHARE,
    DEV_CONTROL_FLIP,
    DEV_CONTROL_P95,
    DEV_LOCKIN_MARGIN,
    DEV_NEW_COLLAPSE_TARGET,
    DEV_P95_MOVER,
    PERCENTILE_Q,
    V_REF,
    AVerdict,
    SafetyVerdict,
    _mean,
    _percentile,
    dev_safety_verdict,
    lock_in_event,
    prior_rank,
    progress,
    reply_reduction,
    selected_a_verdict,
    top_share,
)

__all__ = [
    "MODES", "SHIPPED", "ZERO", "configs_for_mode", "GateResult",
    "require_complete_pairing", "require_zero_identity",
    "dev_safety_v17", "dev_mechanism_verdict", "heldout_verdict",
    "abcd_verdict", "select_smallest_passing", "build_artifact", "main",
    # Deliberate RE-EXPORTS of the v16 definitions (plan Task 5: "import,
    # rather than copy"). Several are not referenced by name in this module --
    # the imported aggregator applies them internally -- but re-exporting them
    # here means a v17 consumer cannot accidentally acquire a second copy, and
    # a test can assert these ARE the v16 objects.
    "prior_rank", "reply_reduction", "top_share", "lock_in_event",
    "_percentile", "_mean", "progress", "selected_a_verdict",
    "dev_safety_verdict", "AVerdict", "SafetyVerdict",
    "COLLAPSE_TOP_SHARE", "DEV_NEW_COLLAPSE_TARGET", "DEV_P95_MOVER",
    "DEV_COMPOUND_EFF", "DEV_COMPOUND_TOPSHARE", "DEV_CONTROL_FLIP",
    "DEV_CONTROL_P95", "DEV_LOCKIN_MARGIN", "PERCENTILE_Q", "V_REF",
    "A_REPLY_REDUCTION", "A_PROGRESS", "A_NEW_COLLAPSE_MAX", "A_TOPSHARE_MAX",
]

# --- modes and their EXACT config sets (design §§5.3, 7, 8, 9) -------------
MODES = ("tooling_smoke", "development", "held_out", "abcd")

SHIPPED: Optional[float] = None      # the shipped absolute fpu_value path
ZERO = 0.0                           # structurally the same branch (§2.2)
SMOKE_POSITIVE = 0.35                # §5.3 plumbing only

# §7.2 tightens the lock-in allowance to 1 for development; §8.2 keeps 2.
DEV_LOCKIN_MARGIN_V17 = 1
HELDOUT_LOCKIN_MARGIN = DEV_LOCKIN_MARGIN

# §7.3 development mechanism gates
DEV_REPLY_REDUCTION = 0.50
DEV_MIN_TARGETS_WITH_FEWER_REPLIES = 8
DEV_MAX_RANK_WORSENED = 1
RANK_WORSENED_BY = 10
# §8.2 held-out mechanism confirmation -- a transfer floor, not a repeat of the
# §7.3 selection gate.
HELDOUT_REPLY_REDUCTION = 0.20

# §9 count-based A/B/C/D criteria
A_SEVERE_MAX = 5          # <= 5/30, i.e. v14b's 16.7%
B_SEVERE_MAX, B_OVER_MAX = 0, 2
C_SEVERE_MAX, C_OVER_MAX, C_MEAN_MAX = 4, 10, 0.099
D_SEVERE_MAX = 0

REQUIRED_ROW_FIELDS = (
    "canonical_sha1", "role", "coefficient", "selected_move", "selected_prior",
    "selected_prior_rank", "root_value_stm", "parent_value", "top_share",
    "eff_children", "replies", "collapse", "lock_in", "explored_mass",
    "stabilization_sim", "complete",
)


def configs_for_mode(mode: str, *, frozen_coefficient: Optional[float] = None
                     ) -> Tuple[Optional[float], ...]:
    """The EXACT config set a mode may run. Any other set is a protocol error.

    development runs the full grid; held_out and abcd run shipped plus exactly
    one already-frozen coefficient, which is how §8/§9 stay unable to select.
    """
    if mode not in MODES:
        raise prov.ProtocolViolation(f"unknown mode {mode!r}; frozen set is {MODES}")
    if mode == "tooling_smoke":
        if frozen_coefficient is not None:
            raise prov.ProtocolViolation(
                "tooling_smoke takes no frozen coefficient")
        return (SHIPPED, ZERO, SMOKE_POSITIVE)
    if mode == "development":
        if frozen_coefficient is not None:
            raise prov.ProtocolViolation(
                "development SELECTS the coefficient; it may not be given one")
        return (SHIPPED, ZERO) + tuple(prov.GRID)
    # held_out / abcd
    if frozen_coefficient is None:
        raise prov.ProtocolViolation(
            f"{mode} requires exactly one frozen coefficient, selected by "
            f"development; running a grid here would let a later stage select")
    prov.validate_coefficient(frozen_coefficient)
    if frozen_coefficient in (SHIPPED, ZERO):
        raise prov.ProtocolViolation(
            f"{mode} frozen coefficient must be positive, got "
            f"{frozen_coefficient!r}")
    return (SHIPPED, frozen_coefficient)


# --- pairing and the r=0 identity prerequisite -----------------------------

def require_complete_pairing(rows: Sequence[Mapping[str, Any]],
                             configs: Sequence[Optional[float]]) -> None:
    """Every position must have a complete, finite row for EVERY config.

    A missing or incomplete row is a refusal, never a silently smaller
    denominator (design §7.2's "any nonfinite metric, missing row, identity
    mismatch, or incomplete search").
    """
    for row in rows:
        missing = [f for f in REQUIRED_ROW_FIELDS if f not in row]
        if missing:
            raise prov.ProtocolViolation(
                f"row {row.get('canonical_sha1')!r} missing fields {missing}")
        if not row["complete"]:
            raise prov.ProtocolViolation(
                f"incomplete search for position {row['canonical_sha1']!r} at "
                f"coefficient {row['coefficient']!r}")
        for field_name in ("root_value_stm", "top_share", "eff_children",
                           "explored_mass"):
            value = row[field_name]
            if value is None or not math.isfinite(value):
                raise prov.ProtocolViolation(
                    f"nonfinite {field_name}={value!r} for position "
                    f"{row['canonical_sha1']!r}")
        # Domain checks. A negative reply count or an out-of-range share is not
        # a small metric error -- it silently corrupts every aggregate below.
        if not isinstance(row["replies"], int) or isinstance(row["replies"], bool) \
                or row["replies"] < 0:
            raise prov.ProtocolViolation(
                f"replies must be a non-negative int, got {row['replies']!r} for "
                f"position {row['canonical_sha1']!r}")
        if row["eff_children"] < 0:
            raise prov.ProtocolViolation(
                f"eff_children must be >= 0, got {row['eff_children']!r}")
        for field_name in ("top_share", "explored_mass"):
            if not 0.0 <= row[field_name] <= 1.0:
                raise prov.ProtocolViolation(
                    f"{field_name}={row[field_name]!r} outside [0, 1] for "
                    f"position {row['canonical_sha1']!r}")
    seen: Dict[Tuple[str, Any], int] = {}
    for row in rows:
        key = (row["canonical_sha1"], row["coefficient"])
        seen[key] = seen.get(key, 0) + 1
    duplicated = sorted(k for k, n in seen.items() if n > 1)
    if duplicated:
        raise prov.ProtocolViolation(
            f"duplicate rows for {len(duplicated)} (position, coefficient) "
            f"pair(s), e.g. {duplicated[:3]}; a duplicate double-counts in every "
            f"aggregate")
    by_position: Dict[str, set] = {}
    for row in rows:
        by_position.setdefault(row["canonical_sha1"], set()).add(row["coefficient"])
    expected = set(configs)
    incomplete = {sha: sorted(expected - got, key=lambda c: (c is not None, c))
                  for sha, got in by_position.items() if got != expected}
    if incomplete:
        raise prov.ProtocolViolation(
            f"incomplete pairing: {len(incomplete)} position(s) lack rows for "
            f"every config, e.g. {sorted(incomplete.items())[:3]}")
    if not by_position:
        raise prov.ProtocolViolation("no rows supplied")


def require_zero_identity(rows: Sequence[Mapping[str, Any]], *,
                          compare_fields: Sequence[str] = (
                              "selected_move", "root_value_stm", "top_share",
                              "eff_children", "replies", "collapse",
                              "explored_mass", "stabilization_sim")) -> None:
    """§7.1: `r=0` must be byte-identical to shipped for every persisted field.

    Runs BEFORE any positive coefficient is interpreted; a mismatch rejects the
    implementation rather than being reported as a candidate effect.
    """
    shipped = {r["canonical_sha1"]: r for r in rows if r["coefficient"] is SHIPPED}
    zero = {r["canonical_sha1"]: r for r in rows if r["coefficient"] == ZERO
            and r["coefficient"] is not SHIPPED}
    if not shipped or not zero:
        raise prov.ProtocolViolation(
            "the r=0 identity prerequisite needs both a shipped and an r=0 row "
            "for every position")
    if set(shipped) != set(zero):
        raise prov.ProtocolViolation(
            "shipped and r=0 cover different positions")
    for sha, s_row in shipped.items():
        z_row = zero[sha]
        diffs = [f for f in compare_fields if s_row[f] != z_row[f]]
        if diffs:
            raise prov.ProtocolViolation(
                f"r=0 identity FAILED at position {sha!r}: fields {diffs} differ "
                f"from shipped; this rejects the implementation before any "
                f"positive coefficient is interpreted (§7.1)")


# --- pure gates ------------------------------------------------------------

@dataclass(frozen=True)
class GateResult:
    coefficient: float
    passed: bool
    reasons: Tuple[str, ...]
    metrics: Mapping[str, Any]


def _paired(rows: Sequence[Mapping[str, Any]], coefficient: Optional[float]
            ) -> Tuple[List[dict], List[dict]]:
    """Per-position (shipped, candidate) pairs for one coefficient.

    Every gate routes through here, so the domain assertions live here too:
    `require_complete_pairing` is the front door, but a gate called directly
    must still refuse corrupt rows rather than fold them into an aggregate.
    """
    shipped = {r["canonical_sha1"]: r for r in rows if r["coefficient"] is SHIPPED}
    cand = [r for r in rows
            if r["coefficient"] == coefficient and r["coefficient"] is not SHIPPED]
    missing = [r["canonical_sha1"] for r in cand if r["canonical_sha1"] not in shipped]
    if missing:
        raise prov.ProtocolViolation(
            f"candidate rows without a shipped partner: {sorted(missing)[:3]}")
    for r in cand + [shipped[r["canonical_sha1"]] for r in cand]:
        if r["replies"] < 0 or r["eff_children"] < 0:
            raise prov.ProtocolViolation(
                f"negative replies/eff_children for position "
                f"{r['canonical_sha1']!r}; refusing to aggregate corrupt rows")
    return [shipped[r["canonical_sha1"]] for r in cand], cand


def _safety_rows(rows: Sequence[Mapping[str, Any]], coefficient: float
                 ) -> List[dict]:
    """Reshape paired rows into the schema `dev_safety_verdict` aggregates."""
    out = []
    for s_row, c_row in zip(*_paired(rows, coefficient)):
        eff_red = (0.0 if s_row["eff_children"] == 0 else
                   (s_row["eff_children"] - c_row["eff_children"]) / s_row["eff_children"])
        common = {"role": c_row["role"],
                  "mover_delta": c_row["root_value_stm"] - s_row["root_value_stm"]}
        if c_row["role"] == "target":
            out.append({**common,
                        "new_collapse": bool(c_row["collapse"]) and not bool(s_row["collapse"]),
                        "lock_in": bool(c_row["lock_in"]),
                        "eff_children_reduction": eff_red,
                        "top_share_inc": c_row["top_share"] - s_row["top_share"]})
        else:
            out.append({**common,
                        "control_flip_to_lower_prior": (
                            c_row["selected_move"] != s_row["selected_move"]
                            and c_row["selected_prior"] < s_row["selected_prior"])})
    return out


def dev_safety_v17(rows: Sequence[Mapping[str, Any]], coefficient: float, *,
                   shipped_lockin: int,
                   lockin_margin: int = DEV_LOCKIN_MARGIN_V17) -> SafetyVerdict:
    """§7.2 (development) / §8.2 (held-out) safety, via the v16 aggregator.

    The per-stratum sub-gate is disabled: v17's 16 late-only targets are far
    below `DEV_BAND_MIN_N`, so it could never activate.
    """
    from .diagnose_fpu_policy_mass import FpuRunConfig
    return dev_safety_verdict(
        _safety_rows(rows, coefficient),
        FpuRunConfig(f"v17_r{coefficient}", None),
        shipped_lockin, shipped_lockin,
        stratum_gate=False, lockin_margin=lockin_margin)


def _aggregate_reply_reduction(pairs: Sequence[Tuple[dict, dict]]) -> float:
    """§7.0's aggregate `1 - sum(candidate)/sum(shipped)`.

    A zero shipped denominator is INVALID rather than automatically passing
    (§7.0), so it is refused here instead of raising ZeroDivisionError deep
    inside the imported helper.
    """
    ref = sum(s["replies"] for s, _ in pairs)
    if ref <= 0:
        raise prov.ProtocolViolation(
            "aggregate reply reduction has a zero shipped denominator, which "
            "§7.0 defines as invalid rather than an automatic pass")
    return reply_reduction(ref, sum(c["replies"] for _, c in pairs))


def dev_mechanism_verdict(rows: Sequence[Mapping[str, Any]],
                          coefficient: float) -> GateResult:
    """§7.3 development mechanism gates. All four must hold."""
    shipped_rows, cand_rows = _paired(rows, coefficient)
    pairs = [(s, c) for s, c in zip(shipped_rows, cand_rows) if c["role"] == "target"]
    if not pairs:
        raise prov.ProtocolViolation("no target rows for the mechanism gate")
    reasons: List[str] = []
    rr = _aggregate_reply_reduction(pairs)
    if rr < DEV_REPLY_REDUCTION:
        reasons.append(f"reply_reduction={rr:.4f}<{DEV_REPLY_REDUCTION}")
    fewer = sum(1 for s, c in pairs if c["replies"] < s["replies"])
    if fewer < DEV_MIN_TARGETS_WITH_FEWER_REPLIES:
        reasons.append(
            f"targets_with_fewer_replies={fewer}<{DEV_MIN_TARGETS_WITH_FEWER_REPLIES}")
    eff = _mean((s["eff_children"] - c["eff_children"]) / s["eff_children"]
                if s["eff_children"] else 0.0 for s, c in pairs)
    if not (0.0 < eff < DEV_COMPOUND_EFF):
        reasons.append(f"mean_eff_children_reduction={eff:.4f} not in (0, "
                       f"{DEV_COMPOUND_EFF})")
    worsened = sum(1 for s, c in pairs
                   if c["selected_move"] != s["selected_move"]
                   and c["selected_prior_rank"] - s["selected_prior_rank"] >= RANK_WORSENED_BY)
    if worsened > DEV_MAX_RANK_WORSENED:
        reasons.append(f"targets_rank_worsened_by_{RANK_WORSENED_BY}={worsened}>"
                       f"{DEV_MAX_RANK_WORSENED}")
    return GateResult(coefficient, not reasons, tuple(reasons), {
        "reply_reduction": rr, "targets_with_fewer_replies": fewer,
        "mean_eff_children_reduction": eff, "targets_rank_worsened": worsened,
        "n_targets": len(pairs)})


def heldout_verdict(rows: Sequence[Mapping[str, Any]], coefficient: float, *,
                    shipped_lockin: int) -> GateResult:
    """§8.2: the same safety table at lock-in margin 2, plus a >=20% transfer
    floor. Failure rejects v17 even when collateral is otherwise safe."""
    # §8.2 runs ONE already-frozen coefficient; an off-grid value here would
    # mean development selected something the grid never contained.
    prov.validate_coefficient(coefficient)
    safety = dev_safety_v17(rows, coefficient, shipped_lockin=shipped_lockin,
                            lockin_margin=HELDOUT_LOCKIN_MARGIN)
    shipped_rows, cand_rows = _paired(rows, coefficient)
    pairs = [(s, c) for s, c in zip(shipped_rows, cand_rows) if c["role"] == "target"]
    if not pairs:
        raise prov.ProtocolViolation("no target rows for the held-out gate")
    rr = _aggregate_reply_reduction(pairs)
    reasons = list(safety.reasons)
    if rr < HELDOUT_REPLY_REDUCTION:
        reasons.append(f"heldout_reply_reduction={rr:.4f}<{HELDOUT_REPLY_REDUCTION}")
    attenuated = HELDOUT_REPLY_REDUCTION <= rr < DEV_REPLY_REDUCTION
    return GateResult(coefficient, not reasons, tuple(reasons),
                      {**safety.metrics, "heldout_reply_reduction": rr,
                       "attenuated_but_present": attenuated})


def abcd_verdict(gate: str, *, n: int, mean: float, over: int, severe: int,
                 a_rows: Optional[Sequence[Mapping[str, Any]]] = None
                 ) -> GateResult:
    """§9 count-based acceptance for one established probe.

    A additionally requires the four mechanism criteria, which are delegated to
    the imported `selected_a_verdict` rather than re-derived.
    """
    reasons: List[str] = []
    if gate == "A":
        if mean > 0.0:
            reasons.append(f"A_mean={mean:.6f}>0.0")
        if severe > A_SEVERE_MAX:
            reasons.append(f"A_severe={severe}/{n}>{A_SEVERE_MAX}")
        if a_rows is None:
            raise prov.ProtocolViolation("gate A requires per-case rows")
        av: AVerdict = selected_a_verdict(a_rows)
        if av.progress < A_PROGRESS:
            reasons.append(f"A_progress={av.progress:.4f}<{A_PROGRESS}")
        if av.reply_reduction < A_REPLY_REDUCTION:
            reasons.append(f"A_reply_reduction={av.reply_reduction:.4f}<{A_REPLY_REDUCTION}")
        if av.a_new_collapse > A_NEW_COLLAPSE_MAX:
            reasons.append(f"A_new_collapse={av.a_new_collapse}>{A_NEW_COLLAPSE_MAX}")
        if av.a_top_share_inc > A_TOPSHARE_MAX:
            reasons.append(f"A_top_share_inc={av.a_top_share_inc:.4f}>{A_TOPSHARE_MAX}")
        metrics = {"mean": mean, "over": over, "severe": severe, "n": n,
                   "progress": av.progress, "reply_reduction": av.reply_reduction,
                   "new_collapse": av.a_new_collapse,
                   "top_share_inc": av.a_top_share_inc}
    elif gate == "B":
        if severe > B_SEVERE_MAX:
            reasons.append(f"B_severe={severe}/{n}>{B_SEVERE_MAX}")
        if over > B_OVER_MAX:
            reasons.append(f"B_over={over}/{n}>{B_OVER_MAX}")
        metrics = {"mean": mean, "over": over, "severe": severe, "n": n}
    elif gate == "C":
        if severe > C_SEVERE_MAX:
            reasons.append(f"C_severe={severe}/{n}>{C_SEVERE_MAX}")
        if over > C_OVER_MAX:
            reasons.append(f"C_over={over}/{n}>{C_OVER_MAX}")
        if mean > C_MEAN_MAX:
            reasons.append(f"C_mean={mean:.6f}>{C_MEAN_MAX}")
        metrics = {"mean": mean, "over": over, "severe": severe, "n": n}
    elif gate == "D":
        if severe > D_SEVERE_MAX:
            reasons.append(f"D_severe={severe}/{n}>{D_SEVERE_MAX}")
        if mean > 0.0:
            reasons.append(f"D_mean={mean:.6f}>0.0")
        metrics = {"mean": mean, "over": over, "severe": severe, "n": n}
    else:
        raise prov.ProtocolViolation(f"unknown A/B/C/D gate {gate!r}")
    return GateResult(float("nan"), not reasons, tuple(reasons), metrics)


def select_smallest_passing(rows: Sequence[Mapping[str, Any]], *,
                            shipped_lockin: int,
                            grid: Sequence[float] = prov.GRID
                            ) -> Tuple[Optional[float], Dict[float, GateResult]]:
    """§4/§7.3: the SMALLEST grid coefficient passing §§7.2-7.3, or None.

    Evaluates every grid point so the full table is persisted, but selection is
    strictly by ascending coefficient -- never by best score.
    """
    # §13 forbids adding, interpolating, or extending coefficients. The `grid`
    # argument exists so a test can exercise a subset, never so a caller can
    # widen the frozen set.
    unknown = sorted(set(grid) - set(prov.GRID))
    if unknown:
        raise prov.ProtocolViolation(
            f"coefficients {unknown} are outside the frozen grid {prov.GRID}; "
            f"§13 forbids extending the grid after any scientific result")
    results: Dict[float, GateResult] = {}
    for r in sorted(grid):
        safety = dev_safety_v17(rows, r, shipped_lockin=shipped_lockin)
        mech = dev_mechanism_verdict(rows, r)
        reasons = tuple(safety.reasons) + tuple(mech.reasons)
        results[r] = GateResult(r, not reasons, reasons,
                                {**safety.metrics, **mech.metrics})
    for r in sorted(grid):
        if results[r].passed:
            return r, results
    return None, results


# --- artifacts -------------------------------------------------------------

def build_artifact(*, mode: str, coefficient: Optional[float],
                   rows: Sequence[Mapping[str, Any]],
                   gates: Mapping[str, Any],
                   checkpoints: Mapping[str, str],
                   manifest: Optional[str] = None,
                   source_index: Optional[str] = None,
                   replay_paths: Optional[Iterable[str]] = None,
                   source_files: Iterable[str] = ()) -> Dict[str, Any]:
    """A canonical, timestamp-free diagnostic artifact.

    Scientific modes must POPULATE every applicable identity; a null is a
    refusal, not an omission.
    """
    if mode not in MODES:
        raise prov.ProtocolViolation(f"unknown mode {mode!r}")
    run_kind = mode
    if prov.is_scientific(run_kind):
        for name, value in (("manifest", manifest), ("source_index", source_index),
                            ("replay_paths", replay_paths)):
            if not value:
                raise prov.ProtocolViolation(
                    f"scientific mode {mode!r} must populate the {name} identity; "
                    f"a null identity is a refusal, not an omission")
        if not checkpoints:
            raise prov.ProtocolViolation(
                f"scientific mode {mode!r} must populate checkpoint identities")
        if not list(source_files or ()):
            raise prov.ProtocolViolation(
                f"scientific mode {mode!r} must populate source-file identities")
    return {
        "schema_version": prov.SCHEMA_VERSION,
        "artifact_kind": "diagnostic",
        "mode": mode,
        "coefficient": coefficient,
        "n_rows": len(rows),
        "gates": dict(gates),
        "provenance": prov.build_provenance(
            run_kind=run_kind, coefficient=coefficient,
            checkpoints=checkpoints, source_files=source_files,
            manifest=manifest, source_index=source_index,
            replay_paths=replay_paths),
    }


# --- execution shim (reuses the v16 search helpers) ------------------------

def run_positions(dev_rows, configs, *, checkpoint: str,
                  eval_batch_size: int, stall_flush_sims: int,
                  seed_base: int,
                  searcher: Optional[Callable] = None) -> List[dict]:
    """Search every position under every config. The evaluator import is lazy,
    so this module stays import-pure. `searcher` is injectable for tests."""
    prov.validate_batching((eval_batch_size, stall_flush_sims,
                            prov.BATCHING[2]))
    if searcher is None:                                    # pragma: no cover
        from .diagnose_fpu_policy_mass import _make_evaluator_and_base_cfg
        evaluator, base_cfg = _make_evaluator_and_base_cfg(
            checkpoint, eval_batch_size, stall_flush_sims)
        searcher = _default_searcher(evaluator, base_cfg)
    return [searcher(row, cfg) for row in dev_rows for cfg in configs]


def _default_searcher(evaluator, base_cfg):                 # pragma: no cover
    import dataclasses

    def search(row, coefficient):
        cfg = dataclasses.replace(
            base_cfg, fpu_shipped_policy_mass_reduction=coefficient)
        prov.validate_batching(cfg)
        raise NotImplementedError(
            "v17 position execution lands with the operator stage; Task 5 is "
            "the protocol/gate layer only")
    return search


# --- CLI -------------------------------------------------------------------

def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--frozen-coefficient", type=float, default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-batch-size", type=int, default=prov.BATCHING[0])
    ap.add_argument("--stall-flush-sims", type=int, default=prov.BATCHING[1])
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        # Free CLI overrides of the §2.4 triple are refused, not honoured.
        prov.validate_batching((args.eval_batch_size, args.stall_flush_sims,
                                prov.BATCHING[2]))
        prov.validate_output_path(args.out_dir)
        prov.verify_frozen_design()
        prov.require_clean_worktree(args.mode)
        configs_for_mode(args.mode, frozen_coefficient=args.frozen_coefficient)
    except prov.ProtocolViolation as exc:
        print(f"PROTOCOL VIOLATION: {exc}")
        return protocol.EXIT_USAGE
    print("v17 diagnostic protocol validated; execution lands with the "
          "operator stage")
    return protocol.EXIT_OK


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
