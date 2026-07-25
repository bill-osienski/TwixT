"""v17 paired diagnostic -- row production, gates, selection, and artifacts.

Task 5 of the v17 plan. The v16 diagnostic (`diagnose_fpu_policy_mass.py`)
already contains every metric definition, threshold, search helper and observer
this stage needs, so they are IMPORTED here, never restated. The only change
made to that module is a backward-compatible keyword parameterization of
`dev_safety_verdict` (`stratum_gate`, `lockin_margin`), proved byte-identical
against 66 pre-change fixtures.

Frozen design ref:
`docs/superpowers/specs/2026-07-24-v17-baseline-preserving-policy-mass-fpu-design.md`
(SHA-1 `944f358c0e3ef66503d2cbb56e31dabd145bafc2`) §§7-9.

Gate arithmetic note: §7.0 freezes the aggregate reply reduction as
`1 - sum(candidate)/sum(shipped)`, and that float is what every artifact
reports. The gate DECISION uses the exact integer ratio, because
`1 - 80/100` is `0.19999999999999996` in IEEE754 and would reject a reduction
that is exactly 20% by count. Exact arithmetic is mathematical conformance with
the frozen formula, not a threshold change.

Import-pure: no MLX at module import. Heavy imports are lazy, inside
`run_diagnostic`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    "validate_row_set", "require_complete_pairing", "require_zero_identity",
    "dev_safety_v17", "dev_mechanism_verdict", "heldout_verdict",
    "abcd_verdict", "select_smallest_passing", "build_artifact",
    "build_row", "run_diagnostic", "main",
    # Deliberate RE-EXPORTS of the v16 definitions (plan Task 5: "import,
    # rather than copy"). Several are not referenced by name here -- the
    # imported aggregator applies them internally -- but re-exporting means a
    # v17 consumer cannot acquire a second copy, and a test can assert these
    # ARE the v16 objects.
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
DEV_REPLY_REDUCTION = Fraction(1, 2)
DEV_MIN_TARGETS_WITH_FEWER_REPLIES = 8
DEV_MAX_RANK_WORSENED = 1
RANK_WORSENED_BY = 10
# §8.2 held-out mechanism confirmation -- a transfer floor, not a repeat of the
# §7.3 selection gate.
HELDOUT_REPLY_REDUCTION = Fraction(1, 5)

# §9 count-based A/B/C/D criteria, with their frozen cardinalities
A_SEVERE_MAX = 5          # <= 5/30, i.e. v14b's 16.7%
B_SEVERE_MAX, B_OVER_MAX = 0, 2
C_SEVERE_MAX, C_OVER_MAX, C_MEAN_MAX = 4, 10, 0.099
D_SEVERE_MAX = 0
ABCD_CARDINALITY = {"A": 30, "B": 18, "C": 30, "D": 30}

# The complete persisted scientific-result payload for one (position, config).
SCIENTIFIC_FIELDS = (
    "selected_move", "selected_prior", "selected_prior_rank", "root_value_stm",
    "parent_value", "top_share", "eff_children", "replies", "collapse",
    "lock_in", "explored_mass", "stabilization_sim", "complete",
)
# `coefficient` is the configuration LABEL; §7.1 excludes exactly it from the
# identity comparison and nothing else.
IDENTITY_FIELDS = ("canonical_sha1", "role")
REQUIRED_ROW_FIELDS = IDENTITY_FIELDS + ("coefficient",) + SCIENTIFIC_FIELDS
FINITE_FIELDS = ("root_value_stm", "parent_value", "top_share", "eff_children",
                 "explored_mass", "selected_prior")
UNIT_INTERVAL_FIELDS = ("top_share", "explored_mass", "selected_prior")
ROLES = ("target", "control")


def configs_for_mode(mode: str, *, frozen_coefficient: Optional[float] = None
                     ) -> Tuple[Optional[float], ...]:
    """The EXACT config set a mode may run. Any other set is a protocol error.

    development runs the full grid; held_out and abcd run shipped plus exactly
    one already-frozen positive coefficient, which is how §8/§9 stay unable to
    select.
    """
    if mode not in MODES:
        raise prov.ProtocolViolation(f"unknown mode {mode!r}; frozen set is {MODES}")
    if mode == "tooling_smoke":
        if frozen_coefficient is not None:
            raise prov.ProtocolViolation("tooling_smoke takes no frozen coefficient")
        return (SHIPPED, ZERO, SMOKE_POSITIVE)
    if mode == "development":
        if frozen_coefficient is not None:
            raise prov.ProtocolViolation(
                "development SELECTS the coefficient; it may not be given one")
        return (SHIPPED, ZERO) + tuple(prov.GRID)
    if frozen_coefficient is None:
        raise prov.ProtocolViolation(
            f"{mode} requires exactly one frozen coefficient, selected by "
            f"development; running a grid here would let a later stage select")
    prov.validate_coefficient(frozen_coefficient)
    if frozen_coefficient in (SHIPPED, ZERO):
        raise prov.ProtocolViolation(
            f"{mode} frozen coefficient must be positive, got {frozen_coefficient!r}")
    return (SHIPPED, frozen_coefficient)


# --- one centralized validator; every gate and artifact routes through it ---

def validate_row_set(rows: Sequence[Mapping[str, Any]],
                     configs: Sequence[Optional[float]]) -> None:
    """Complete row/pair validation (design §7.2's "any nonfinite metric,
    missing row, identity mismatch, or incomplete search").

    Checks, in order: required fields; per-field types, finiteness and domains;
    duplicate (position, coefficient) pairs; role consistency across a
    position's rows; and complete pairing of every position with every config.
    """
    if not rows:
        raise prov.ProtocolViolation("no rows supplied")
    for row in rows:
        missing = [f for f in REQUIRED_ROW_FIELDS if f not in row]
        if missing:
            raise prov.ProtocolViolation(
                f"row {row.get('canonical_sha1')!r} missing fields {missing}")
        sha = row["canonical_sha1"]
        if row["role"] not in ROLES:
            raise prov.ProtocolViolation(
                f"row {sha!r} has role {row['role']!r}, not one of {ROLES}")
        if not row["complete"]:
            raise prov.ProtocolViolation(
                f"incomplete search for position {sha!r} at coefficient "
                f"{row['coefficient']!r}")
        for field_name in FINITE_FIELDS:
            value = row[field_name]
            if not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or not math.isfinite(value):
                raise prov.ProtocolViolation(
                    f"nonfinite or non-numeric {field_name}={value!r} for "
                    f"position {sha!r}")
        for field_name in UNIT_INTERVAL_FIELDS:
            if not 0.0 <= row[field_name] <= 1.0:
                raise prov.ProtocolViolation(
                    f"{field_name}={row[field_name]!r} outside [0, 1] for "
                    f"position {sha!r}")
        for field_name in ("replies", "selected_prior_rank", "stabilization_sim"):
            value = row[field_name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise prov.ProtocolViolation(
                    f"{field_name} must be a non-negative int, got {value!r} for "
                    f"position {sha!r}")
        if row["eff_children"] < 0:
            raise prov.ProtocolViolation(
                f"eff_children must be >= 0, got {row['eff_children']!r}")
    seen: Dict[Tuple[Any, Any], int] = {}
    roles: Dict[Any, set] = {}
    for row in rows:
        key = (row["canonical_sha1"], row["coefficient"])
        seen[key] = seen.get(key, 0) + 1
        roles.setdefault(row["canonical_sha1"], set()).add(row["role"])
    duplicated = sorted(k for k, n in seen.items() if n > 1)
    if duplicated:
        raise prov.ProtocolViolation(
            f"duplicate rows for {len(duplicated)} (position, coefficient) "
            f"pair(s), e.g. {duplicated[:3]}; a duplicate double-counts in every "
            f"aggregate")
    drifted = sorted(sha for sha, rs in roles.items() if len(rs) > 1)
    if drifted:
        raise prov.ProtocolViolation(
            f"role drift: position(s) {drifted[:3]} appear as more than one role; "
            f"a position is a target or a control, never both")
    expected = set(configs)
    by_position: Dict[Any, set] = {}
    for row in rows:
        by_position.setdefault(row["canonical_sha1"], set()).add(row["coefficient"])
    incomplete = sorted(sha for sha, got in by_position.items() if got != expected)
    if incomplete:
        raise prov.ProtocolViolation(
            f"incomplete pairing: {len(incomplete)} position(s) lack a row for "
            f"every config, e.g. {incomplete[:3]}")


def require_complete_pairing(rows, configs) -> None:
    """Backward-compatible alias for `validate_row_set`."""
    validate_row_set(rows, configs)


def require_zero_identity(rows: Sequence[Mapping[str, Any]]) -> None:
    """§7.1: `r=0` must be byte-identical to shipped for the COMPLETE persisted
    scientific-result payload -- every field except the configuration label.

    Runs BEFORE any positive coefficient is interpreted; a mismatch rejects the
    implementation rather than being reported as a candidate effect.
    """
    shipped = {r["canonical_sha1"]: r for r in rows if r["coefficient"] is SHIPPED}
    zero = {r["canonical_sha1"]: r for r in rows
            if r["coefficient"] is not SHIPPED and r["coefficient"] == ZERO}
    if not shipped or not zero:
        raise prov.ProtocolViolation(
            "the r=0 identity prerequisite needs both a shipped and an r=0 row "
            "for every position")
    if set(shipped) != set(zero):
        raise prov.ProtocolViolation("shipped and r=0 cover different positions")
    for sha, s_row in shipped.items():
        z_row = zero[sha]
        diffs = [f for f in IDENTITY_FIELDS + SCIENTIFIC_FIELDS
                 if s_row[f] != z_row[f]]
        if diffs:
            raise prov.ProtocolViolation(
                f"r=0 identity FAILED at position {sha!r}: fields {diffs} differ "
                f"from shipped; this rejects the implementation before any "
                f"positive coefficient is interpreted (§7.1)")


# --- pure gates ------------------------------------------------------------

@dataclass(frozen=True)
class GateResult:
    coefficient: Optional[float]
    passed: bool
    reasons: Tuple[str, ...]
    metrics: Mapping[str, Any]


def _paired(rows: Sequence[Mapping[str, Any]], coefficient: Optional[float]
            ) -> Tuple[List[dict], List[dict]]:
    """Per-position (shipped, candidate) pairs for one coefficient.

    Every gate routes through here, and every gate therefore gets the full
    `validate_row_set` treatment: a gate called directly must refuse corrupt
    rows rather than fold them into an aggregate.
    """
    validate_row_set(rows, sorted({r["coefficient"] for r in rows},
                                  key=lambda c: (c is not None, c)))
    shipped = {r["canonical_sha1"]: r for r in rows if r["coefficient"] is SHIPPED}
    cand = [r for r in rows
            if r["coefficient"] is not SHIPPED and r["coefficient"] == coefficient]
    if not cand:
        raise prov.ProtocolViolation(f"no rows for coefficient {coefficient!r}")
    missing = sorted(r["canonical_sha1"] for r in cand
                     if r["canonical_sha1"] not in shipped)
    if missing:
        raise prov.ProtocolViolation(
            f"candidate rows without a shipped partner: {missing[:3]}")
    return [shipped[r["canonical_sha1"]] for r in cand], cand


def _targets(rows, coefficient) -> List[Tuple[dict, dict]]:
    pairs = [(s, c) for s, c in zip(*_paired(rows, coefficient))
             if c["role"] == "target"]
    if not pairs:
        raise prov.ProtocolViolation(
            f"no target rows for coefficient {coefficient!r}")
    return pairs


def _reply_ratio(pairs: Sequence[Tuple[dict, dict]]) -> Tuple[Fraction, float]:
    """(exact reduction, reported float) for §7.0's aggregate.

    A zero shipped denominator is INVALID rather than an automatic pass (§7.0).
    """
    ref = sum(s["replies"] for s, _ in pairs)
    cand = sum(c["replies"] for _, c in pairs)
    if ref <= 0:
        raise prov.ProtocolViolation(
            "aggregate reply reduction has a zero shipped denominator, which "
            "§7.0 defines as invalid rather than an automatic pass")
    return Fraction(ref - cand, ref), reply_reduction(ref, cand)


def _safety_rows(rows: Sequence[Mapping[str, Any]], coefficient: float) -> List[dict]:
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


def dev_mechanism_verdict(rows: Sequence[Mapping[str, Any]],
                          coefficient: float) -> GateResult:
    """§7.3 development mechanism gates. All four must hold."""
    prov.validate_coefficient(coefficient)
    pairs = _targets(rows, coefficient)
    reasons: List[str] = []
    exact, reported = _reply_ratio(pairs)
    if exact < DEV_REPLY_REDUCTION:
        reasons.append(f"reply_reduction={reported:.4f}<{float(DEV_REPLY_REDUCTION)}")
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
        "reply_reduction": reported,
        "reply_reduction_exact": f"{exact.numerator}/{exact.denominator}",
        "targets_with_fewer_replies": fewer,
        "mean_eff_children_reduction": eff, "targets_rank_worsened": worsened,
        "n_targets": len(pairs)})


def heldout_verdict(rows: Sequence[Mapping[str, Any]], coefficient: float, *,
                    shipped_lockin: int) -> GateResult:
    """§8.2: the same safety table at lock-in margin 2, plus a >=20% transfer
    floor. Failure rejects v17 even when collateral is otherwise safe."""
    prov.validate_coefficient(coefficient)
    if coefficient in (SHIPPED, ZERO):
        raise prov.ProtocolViolation(
            "held-out runs one FROZEN POSITIVE coefficient")
    safety = dev_safety_v17(rows, coefficient, shipped_lockin=shipped_lockin,
                            lockin_margin=HELDOUT_LOCKIN_MARGIN)
    exact, reported = _reply_ratio(_targets(rows, coefficient))
    reasons = list(safety.reasons)
    if exact < HELDOUT_REPLY_REDUCTION:
        reasons.append(f"heldout_reply_reduction={reported:.4f}<"
                       f"{float(HELDOUT_REPLY_REDUCTION)}")
    attenuated = HELDOUT_REPLY_REDUCTION <= exact < DEV_REPLY_REDUCTION
    return GateResult(coefficient, not reasons, tuple(reasons),
                      {**safety.metrics, "heldout_reply_reduction": reported,
                       "heldout_reply_reduction_exact":
                           f"{exact.numerator}/{exact.denominator}",
                       "attenuated_but_present": attenuated})


def abcd_verdict(gate: str, *, coefficient: float, n: int, mean: float,
                 over: int, severe: int,
                 a_rows: Optional[Sequence[Mapping[str, Any]]] = None
                 ) -> GateResult:
    """§9 count-based acceptance for one established probe.

    Cardinality is frozen per gate, so a truncated or padded case set is a
    refusal rather than a pass on fewer cases.
    """
    if gate not in ABCD_CARDINALITY:
        raise prov.ProtocolViolation(f"unknown A/B/C/D gate {gate!r}")
    prov.validate_coefficient(coefficient)
    if coefficient in (SHIPPED, ZERO):
        raise prov.ProtocolViolation("A/B/C/D runs one FROZEN POSITIVE coefficient")
    expected_n = ABCD_CARDINALITY[gate]
    if n != expected_n:
        raise prov.ProtocolViolation(
            f"gate {gate} has frozen cardinality {expected_n}, got n={n!r}")
    for name, value in (("over", over), ("severe", severe)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= n:
            raise prov.ProtocolViolation(
                f"{gate}.{name} must be an int in [0, {n}], got {value!r}")
    if not isinstance(mean, (int, float)) or isinstance(mean, bool) \
            or not math.isfinite(mean):
        raise prov.ProtocolViolation(f"{gate}.mean must be finite, got {mean!r}")
    if severe > over:
        raise prov.ProtocolViolation(
            f"{gate}: severe={severe} exceeds over={over}; severe cases are a "
            f"subset of over cases")
    reasons: List[str] = []
    metrics: Dict[str, Any] = {"mean": mean, "over": over, "severe": severe, "n": n}
    if gate == "A":
        if a_rows is None:
            raise prov.ProtocolViolation("gate A requires per-case rows")
        if len(a_rows) != expected_n:
            raise prov.ProtocolViolation(
                f"gate A case rows {len(a_rows)} != frozen cardinality {expected_n}")
        if mean > 0.0:
            reasons.append(f"A_mean={mean:.6f}>0.0")
        if severe > A_SEVERE_MAX:
            reasons.append(f"A_severe={severe}/{n}>{A_SEVERE_MAX}")
        av: AVerdict = selected_a_verdict(a_rows)
        if av.progress < A_PROGRESS:
            reasons.append(f"A_progress={av.progress:.4f}<{A_PROGRESS}")
        if av.reply_reduction < A_REPLY_REDUCTION:
            reasons.append(f"A_reply_reduction={av.reply_reduction:.4f}<{A_REPLY_REDUCTION}")
        if av.a_new_collapse > A_NEW_COLLAPSE_MAX:
            reasons.append(f"A_new_collapse={av.a_new_collapse}>{A_NEW_COLLAPSE_MAX}")
        if av.a_top_share_inc > A_TOPSHARE_MAX:
            reasons.append(f"A_top_share_inc={av.a_top_share_inc:.4f}>{A_TOPSHARE_MAX}")
        metrics.update(progress=av.progress, reply_reduction=av.reply_reduction,
                       new_collapse=av.a_new_collapse,
                       top_share_inc=av.a_top_share_inc)
    elif gate == "B":
        if severe > B_SEVERE_MAX:
            reasons.append(f"B_severe={severe}/{n}>{B_SEVERE_MAX}")
        if over > B_OVER_MAX:
            reasons.append(f"B_over={over}/{n}>{B_OVER_MAX}")
    elif gate == "C":
        if severe > C_SEVERE_MAX:
            reasons.append(f"C_severe={severe}/{n}>{C_SEVERE_MAX}")
        if over > C_OVER_MAX:
            reasons.append(f"C_over={over}/{n}>{C_OVER_MAX}")
        if mean > C_MEAN_MAX:
            reasons.append(f"C_mean={mean:.6f}>{C_MEAN_MAX}")
    else:                                                   # D
        if severe > D_SEVERE_MAX:
            reasons.append(f"D_severe={severe}/{n}>{D_SEVERE_MAX}")
        if mean > 0.0:
            reasons.append(f"D_mean={mean:.6f}>0.0")
    return GateResult(coefficient, not reasons, tuple(reasons), metrics)


def select_smallest_passing(rows: Sequence[Mapping[str, Any]], *,
                            shipped_lockin: int
                            ) -> Tuple[Optional[float], Dict[float, GateResult]]:
    """§4/§7.3: the SMALLEST grid coefficient passing §§7.2-7.3, or None.

    Development evaluates EXACTLY the frozen grid -- not a subset, not a
    superset. §13 forbids extending it, and evaluating fewer points would let a
    caller hide a passing smaller coefficient and select a larger one.
    """
    results: Dict[float, GateResult] = {}
    for r in sorted(prov.GRID):
        safety = dev_safety_v17(rows, r, shipped_lockin=shipped_lockin)
        mech = dev_mechanism_verdict(rows, r)
        reasons = tuple(safety.reasons) + tuple(mech.reasons)
        results[r] = GateResult(r, not reasons, reasons,
                                {**safety.metrics, **mech.metrics})
    for r in sorted(prov.GRID):
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

    Enforces mode/coefficient legality, complete pairing and the r=0 identity
    itself, and persists the COMPLETE paired rows -- a count alone would make
    the gate numbers unauditable. Scientific modes must POPULATE every
    applicable identity; a null is a refusal, not an omission.
    """
    if mode not in MODES:
        raise prov.ProtocolViolation(f"unknown mode {mode!r}")
    expected_configs = configs_for_mode(
        mode, frozen_coefficient=None if mode in ("development", "tooling_smoke")
        else coefficient)
    if mode == "development" and coefficient is not None:
        prov.validate_coefficient(coefficient)      # the SELECTED one, if any
    if rows:
        validate_row_set(rows, expected_configs)
        if ZERO in expected_configs:
            require_zero_identity(rows)
    if prov.is_scientific(mode):
        if not rows:
            raise prov.ProtocolViolation(
                f"scientific mode {mode!r} must persist its paired rows")
        for name, value in (("manifest", manifest), ("source_index", source_index),
                            ("replay_paths", replay_paths),
                            ("checkpoints", checkpoints),
                            ("source_files", list(source_files or ()))):
            if not value:
                raise prov.ProtocolViolation(
                    f"scientific mode {mode!r} must populate the {name} identity; "
                    f"a null identity is a refusal, not an omission")
    return {
        "schema_version": prov.SCHEMA_VERSION,
        "artifact_kind": "diagnostic",
        "mode": mode,
        "coefficient": coefficient,
        "configs": [c for c in expected_configs],
        "n_rows": len(rows),
        "rows": [{k: row[k] for k in REQUIRED_ROW_FIELDS} for row in rows],
        "gates": dict(gates),
        "provenance": prov.build_provenance(
            run_kind=mode, coefficient=coefficient,
            checkpoints=checkpoints, source_files=source_files,
            manifest=manifest, source_index=source_index,
            replay_paths=replay_paths),
    }


# --- row production and the operator shell ---------------------------------

def build_row(*, canonical_sha1: str, role: str, coefficient: Optional[float],
              features: Mapping[str, Any], root: Any) -> Dict[str, Any]:
    """One complete scientific-result row from a v16 `_position_features` dict
    plus the searched root. Every metric definition is the imported one."""
    trace = features["trace"]
    leader_move = features["top_move"]
    leader = None if leader_move is None else root.children.get(leader_move)
    return {
        "canonical_sha1": canonical_sha1,
        "role": role,
        "coefficient": coefficient,
        "selected_move": leader_move,
        "selected_prior": float(trace["selected_move_prior"] or 0.0),
        "selected_prior_rank": int(trace["selected_move_prior_rank"] or 0),
        "root_value_stm": float(features["root_value_stm"]),
        # the visit leader's value in the MOVER's perspective
        "parent_value": 0.0 if leader is None else float(-leader.q_value),
        "top_share": float(features["top_share"]),
        "eff_children": float(features["effective_children"]),
        "replies": int(features["replies"]),
        "collapse": bool(features["collapsed"]),
        "lock_in": bool(lock_in_event(trace)),
        "explored_mass": float(trace["explored_mass_at_stabilization"] or 0.0),
        "stabilization_sim": int(trace["stabilization_sim"] or 0),
        "complete": int(trace["completed_simulation_count"]) == prov.MCTS_SIMS,
    }


def load_manifest(path: str) -> List[dict]:
    """v17 corpus manifest: one row per selected position."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise prov.ProtocolViolation(f"empty manifest {path}")
    needed = {"canonical_sha1", "game_idx", "position_ply", "side", "role",
              "replay_path"}
    missing = sorted(needed - set(rows[0]))
    if missing:
        raise prov.ProtocolViolation(f"manifest {path} missing columns {missing}")
    return rows


def run_diagnostic(*, mode: str, manifest_path: str, checkpoint: str,
                   out_path: str, frozen_coefficient: Optional[float] = None,
                   seed_base: int, source_index: Optional[str] = None,
                   searcher=None) -> Dict[str, Any]:
    """Search every manifest position under every config for `mode`, build the
    complete paired rows, evaluate the mode's gates, and emit the artifact.

    `searcher(manifest_row, coefficient) -> row` is injectable so the whole
    pipeline is testable without a GPU; the default performs real 400-sim MCTS.
    """
    prov.validate_batching(prov.BATCHING)
    prov.verify_frozen_design()
    prov.require_clean_worktree(mode)
    prov.validate_output_path(out_path)
    configs = configs_for_mode(mode, frozen_coefficient=frozen_coefficient)
    manifest_rows = load_manifest(manifest_path)

    if searcher is None:                                    # pragma: no cover
        searcher = _real_searcher(checkpoint, seed_base)

    rows = [searcher(m, c) for m in manifest_rows for c in configs]
    validate_row_set(rows, configs)
    if ZERO in configs:
        require_zero_identity(rows)

    shipped_lockin = sum(1 for r in rows
                         if r["coefficient"] is SHIPPED and r["role"] == "target"
                         and r["lock_in"])
    gates: Dict[str, Any] = {}
    selected = frozen_coefficient
    if mode == "development":
        selected, table = select_smallest_passing(rows, shipped_lockin=shipped_lockin)
        gates = {str(k): {"passed": v.passed, "reasons": list(v.reasons),
                          "metrics": v.metrics} for k, v in table.items()}
    elif mode == "held_out":
        v = heldout_verdict(rows, frozen_coefficient, shipped_lockin=shipped_lockin)
        gates = {"held_out": {"passed": v.passed, "reasons": list(v.reasons),
                              "metrics": v.metrics}}
    artifact = build_artifact(
        mode=mode, coefficient=selected, rows=rows, gates=gates,
        checkpoints={"anchor": checkpoint}, manifest=manifest_path,
        source_index=source_index,
        replay_paths=sorted({m["replay_path"] for m in manifest_rows}),
        source_files=[__file__.replace("\\", "/"),
                      str(Path(__file__).with_name("mcts.py"))])
    protocol.emit(out_path, artifact)
    return artifact


def _real_searcher(checkpoint: str, seed_base: int):        # pragma: no cover
    """Real 400-sim MCTS, reusing the v16 search/observer helpers verbatim."""
    import dataclasses
    from .diagnose_fpu_policy_mass import (
        _make_evaluator_and_base_cfg, _position_features, _reconstruct_state,
        _run_seed, _search_position,
    )
    evaluator, base_cfg = _make_evaluator_and_base_cfg(
        checkpoint, prov.BATCHING[0], prov.BATCHING[1])
    prov.validate_batching(base_cfg)

    def search(manifest_row, coefficient):
        cfg = dataclasses.replace(
            base_cfg, fpu_shipped_policy_mass_reduction=coefficient)
        prov.validate_batching(cfg)
        state = _reconstruct_state(
            {**manifest_row, "canonical_position_sha1": manifest_row["canonical_sha1"]},
            {int(manifest_row["game_idx"]): manifest_row["replay_path"]})
        seed = _run_seed(seed_base, int(manifest_row["game_idx"]),
                         int(float(manifest_row["position_ply"])))
        search_out, obs = _search_position(evaluator, cfg, state, seed)
        return build_row(canonical_sha1=manifest_row["canonical_sha1"],
                         role=manifest_row["role"], coefficient=coefficient,
                         features=_position_features(search_out, obs),
                         root=search_out[2])
    return search


# --- CLI -------------------------------------------------------------------

def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--frozen-coefficient", type=float, default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--source-index", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--eval-batch-size", type=int, default=prov.BATCHING[0])
    ap.add_argument("--stall-flush-sims", type=int, default=prov.BATCHING[1])
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        # Free CLI overrides of the §2.4 triple are refused, not honoured.
        prov.validate_batching((args.eval_batch_size, args.stall_flush_sims,
                                prov.BATCHING[2]))
        artifact = run_diagnostic(
            mode=args.mode, manifest_path=args.manifest,
            checkpoint=args.checkpoint, out_path=args.out,
            frozen_coefficient=args.frozen_coefficient,
            seed_base=args.seed_base, source_index=args.source_index)
    except prov.ProtocolViolation as exc:
        print(f"PROTOCOL VIOLATION: {exc}")
        return protocol.EXIT_USAGE
    print(json.dumps({"mode": artifact["mode"],
                      "coefficient": artifact["coefficient"],
                      "n_rows": artifact["n_rows"]}, sort_keys=True))
    return protocol.EXIT_OK


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
