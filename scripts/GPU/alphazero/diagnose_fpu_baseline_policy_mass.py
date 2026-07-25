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

Gate arithmetic: §7.0 freezes the aggregate reply reduction as
`1 - sum(candidate)/sum(shipped)`, and that float is what every artifact
reports. The gate DECISION uses the exact integer ratio, because
`1 - 80/100` is `0.19999999999999996` in IEEE754 and would reject a reduction
that is exactly 20% by count. Exact arithmetic computes the frozen ratio
correctly; it is conformance, not a threshold change.

Ordering: every scientific precondition -- checkpoint, manifest, source index,
replays, source files, configuration, protocol/config binding and corpus
geometry -- is validated in `preflight()` BEFORE any evaluator is loaded, so a
missing input costs zero searches.

Import-pure: no MLX at module import. Heavy imports are lazy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import fpu_v17_protocol as protocol
from . import fpu_v17_provenance as prov
from .fpu_dev_reservoir_protocol import canonical_json_bytes
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
    "require_corpus_geometry", "preflight",
    "dev_safety_v17", "dev_mechanism_verdict", "heldout_verdict",
    "abcd_verdict", "verify_abcd_baseline", "run_abcd",
    "select_smallest_passing", "build_artifact", "build_row", "search_result_sha1",
    "run_diagnostic", "main",
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
ABCD_GATES = ("A", "B", "C", "D")
ABCD_TOLERANCE = 1e-6
ABCD_BASELINE_PATH = ("logs/eval/fpu_v17_baseline_policy_mass/"
                      "prechange_baseline.json")
ABCD_MOVES_PATH = ("logs/eval/fpu_v17_baseline_policy_mass/"
                   "prechange_abcd_selected_moves.json")

# §6.2 / §8.1 corpus geometry, and §5.2's smoke selector profile.
CORPUS_GEOMETRY: Dict[str, Dict[str, int]] = {
    "development": {"target": 16, "control": 16},
    "held_out": {"target": 24, "control": 32},
    "tooling_smoke": {"control": 2},
}

# Every module whose bytes can change a result. §12 requires source identities
# to cover all of them, not just the module being run.
RESULT_DETERMINING_MODULES = (
    "diagnose_fpu_baseline_policy_mass.py", "diagnose_fpu_policy_mass.py",
    "fpu_v17_protocol.py", "fpu_v17_provenance.py", "fpu_provenance.py",
    "fpu_dev_reservoir_protocol.py", "mcts.py", "eval_runner.py",
    "evaluator.py", "local_evaluator.py", "network.py",
    "position_probe_cases.py", "goal_line_trigger_probe_cases.py",
)

# The complete persisted scientific-result payload for one (position, config).
# `coefficient` is the configuration LABEL; §7.1 excludes exactly it, and
# nothing else, from the identity comparison.
SCIENTIFIC_FIELDS = (
    "seed", "add_noise", "selected_move", "selected_prior",
    "selected_prior_rank", "root_value_stm", "parent_value", "selected_child_q",
    "top_share", "eff_children", "replies", "collapse", "lock_in",
    "explored_mass", "stabilization_sim", "complete", "tree_signature",
    "search_result_sha1",
)
IDENTITY_FIELDS = ("canonical_sha1", "role", "side")
REQUIRED_ROW_FIELDS = IDENTITY_FIELDS + ("coefficient",) + SCIENTIFIC_FIELDS
FLOAT_FIELDS = ("selected_prior", "root_value_stm", "parent_value",
                "selected_child_q", "top_share", "eff_children", "explored_mass")
UNIT_INTERVAL_FIELDS = ("top_share", "explored_mass", "selected_prior")
INT_FIELDS = ("seed", "selected_prior_rank", "replies", "stabilization_sim")
BOOL_FIELDS = ("add_noise", "collapse", "lock_in", "complete")
ROLES = ("target", "control")
SIDES = ("red", "black")


class MissingTelemetry(prov.ProtocolViolation):
    """An observer field the row needs was absent. Never coerced to zero:
    that would turn incomplete telemetry into apparently valid data."""


def configs_for_mode(mode: str, *, frozen_coefficient: Optional[float] = None
                     ) -> Tuple[Optional[float], ...]:
    """The EXACT config set a mode may run. Any other set is a protocol error."""
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
    _require_positive_grid_coefficient(frozen_coefficient, mode)
    return (SHIPPED, frozen_coefficient)


def _require_positive_grid_coefficient(r: Any, where: str) -> float:
    if isinstance(r, bool):
        raise prov.ProtocolViolation(
            f"{where}: coefficient must be a number, got bool {r!r}")
    prov.validate_coefficient(r)
    if r is None or r == 0.0:
        raise prov.ProtocolViolation(
            f"{where} runs one FROZEN POSITIVE coefficient, got {r!r}")
    return r


# --- row identity ----------------------------------------------------------

def search_result_sha1(row: Mapping[str, Any]) -> str:
    """Canonical hash of the scientific payload, excluding the config label.

    §2.2/§7.1: shipped and `r=0` must produce the same `search_result_sha1`;
    only the required config/provenance labels may differ.
    """
    payload = {k: row[k] for k in SCIENTIFIC_FIELDS if k != "search_result_sha1"}
    payload.update({k: row[k] for k in IDENTITY_FIELDS})
    try:
        encoded = canonical_json_bytes(payload)
    except ValueError as exc:
        # canonical JSON is allow_nan=False, so a nonfinite metric cannot even
        # be sealed into a row. Surface it as a protocol refusal like every
        # other rejection rather than as a bare encoder error.
        raise prov.ProtocolViolation(
            f"row {row.get('canonical_sha1')!r} cannot be canonically encoded "
            f"({exc}); a nonfinite metric can never enter the pipeline") from exc
    return hashlib.sha1(encoded).hexdigest()


# --- one centralized validator; every gate and artifact routes through it ---

def _require_bool(value: Any, name: str, sha: Any) -> bool:
    if not isinstance(value, bool):
        raise prov.ProtocolViolation(
            f"{name} must be a bool, got {value!r} ({type(value).__name__}) for "
            f"position {sha!r}")
    return value


def _require_int(value: Any, name: str, sha: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise prov.ProtocolViolation(
            f"{name} must be an int >= {minimum}, got {value!r} for position "
            f"{sha!r}")
    return value


def validate_row_set(rows: Sequence[Mapping[str, Any]],
                     configs: Sequence[Optional[float]]) -> None:
    """Complete row/pair validation (design §7.2's "any nonfinite metric,
    missing row, identity mismatch, or incomplete search")."""
    if not rows:
        raise prov.ProtocolViolation("no rows supplied")
    for row in rows:
        missing = [f for f in REQUIRED_ROW_FIELDS if f not in row]
        if missing:
            raise prov.ProtocolViolation(
                f"row {row.get('canonical_sha1')!r} missing fields {missing}")
        sha = row["canonical_sha1"]
        coefficient = row["coefficient"]
        if isinstance(coefficient, bool):
            raise prov.ProtocolViolation(
                f"coefficient must be None or a number, got bool {coefficient!r} "
                f"for position {sha!r}")
        if coefficient is not None and not isinstance(coefficient, (int, float)):
            raise prov.ProtocolViolation(
                f"coefficient {coefficient!r} is not numeric for position {sha!r}")
        if row["role"] not in ROLES:
            raise prov.ProtocolViolation(
                f"row {sha!r} has role {row['role']!r}, not one of {ROLES}")
        if row["side"] not in SIDES:
            raise prov.ProtocolViolation(
                f"row {sha!r} has side {row['side']!r}, not one of {SIDES}")
        for name in BOOL_FIELDS:
            _require_bool(row[name], name, sha)
        if not row["complete"]:
            raise prov.ProtocolViolation(
                f"incomplete search for position {sha!r} at coefficient "
                f"{coefficient!r}")
        if row["add_noise"]:
            raise prov.ProtocolViolation(
                f"add_noise must be false for every v17 diagnostic search "
                f"(position {sha!r})")
        for name in INT_FIELDS:
            _require_int(row[name], name, sha)
        for name in FLOAT_FIELDS:
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value):
                raise prov.ProtocolViolation(
                    f"nonfinite or non-numeric {name}={value!r} for position {sha!r}")
        for name in UNIT_INTERVAL_FIELDS:
            if not 0.0 <= row[name] <= 1.0:
                raise prov.ProtocolViolation(
                    f"{name}={row[name]!r} outside [0, 1] for position {sha!r}")
        if row["eff_children"] < 0:
            raise prov.ProtocolViolation(
                f"eff_children must be >= 0, got {row['eff_children']!r}")
        if not isinstance(row["tree_signature"], (list, tuple)) or not row["tree_signature"]:
            raise prov.ProtocolViolation(
                f"tree_signature must be a non-empty sequence for position {sha!r}")
        if row["search_result_sha1"] != search_result_sha1(row):
            raise prov.ProtocolViolation(
                f"search_result_sha1 does not match the row payload for position "
                f"{sha!r}; the row has been modified since it was produced")
    seen: Dict[Tuple[Any, Any], int] = {}
    roles: Dict[Any, set] = {}
    sides: Dict[Any, set] = {}
    for row in rows:
        seen[(row["canonical_sha1"], row["coefficient"])] = \
            seen.get((row["canonical_sha1"], row["coefficient"]), 0) + 1
        roles.setdefault(row["canonical_sha1"], set()).add(row["role"])
        sides.setdefault(row["canonical_sha1"], set()).add(row["side"])
    duplicated = sorted(k for k, n in seen.items() if n > 1)
    if duplicated:
        raise prov.ProtocolViolation(
            f"duplicate rows for {len(duplicated)} (position, coefficient) "
            f"pair(s), e.g. {duplicated[:3]}; a duplicate double-counts in every "
            f"aggregate")
    for label, mapping in (("role", roles), ("side", sides)):
        drifted = sorted(sha for sha, vs in mapping.items() if len(vs) > 1)
        if drifted:
            raise prov.ProtocolViolation(
                f"{label} drift: position(s) {drifted[:3]} report more than one "
                f"{label}")
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


def require_corpus_geometry(mode: str, roles: Mapping[str, int]) -> None:
    """§6.2/§8.1: the corpus shape is frozen per mode."""
    expected = CORPUS_GEOMETRY.get(mode)
    if expected is None:
        return
    actual = {r: roles.get(r, 0) for r in set(expected) | set(roles)}
    if actual != dict(expected):
        raise prov.ProtocolViolation(
            f"{mode} corpus geometry {actual} != the frozen {dict(expected)}")


def require_zero_identity(rows: Sequence[Mapping[str, Any]]) -> None:
    """§7.1: `r=0` must be identical to shipped for the COMPLETE persisted
    scientific-result payload and the full tree signature."""
    shipped = {r["canonical_sha1"]: r for r in rows if r["coefficient"] is SHIPPED}
    zero = {r["canonical_sha1"]: r for r in rows
            if r["coefficient"] is not SHIPPED and r["coefficient"] == ZERO
            and not isinstance(r["coefficient"], bool)}
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
    """Per-position (shipped, candidate) pairs. Every gate routes through here,
    so a directly-called gate gets the same validation as the front door."""
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
    """(exact reduction, reported float). A zero shipped denominator is INVALID
    rather than an automatic pass (§7.0)."""
    ref = sum(s["replies"] for s, _ in pairs)
    cand = sum(c["replies"] for _, c in pairs)
    if ref <= 0:
        raise prov.ProtocolViolation(
            "aggregate reply reduction has a zero shipped denominator, which "
            "§7.0 defines as invalid rather than an automatic pass")
    return Fraction(ref - cand, ref), reply_reduction(ref, cand)


def _safety_rows(rows: Sequence[Mapping[str, Any]], coefficient: float) -> List[dict]:
    out = []
    for s_row, c_row in zip(*_paired(rows, coefficient)):
        eff_red = (0.0 if s_row["eff_children"] == 0 else
                   (s_row["eff_children"] - c_row["eff_children"]) / s_row["eff_children"])
        common = {"role": c_row["role"],
                  "mover_delta": c_row["root_value_stm"] - s_row["root_value_stm"]}
        if c_row["role"] == "target":
            out.append({**common,
                        "new_collapse": c_row["collapse"] and not s_row["collapse"],
                        "lock_in": c_row["lock_in"],
                        "eff_children_reduction": eff_red,
                        "top_share_inc": c_row["top_share"] - s_row["top_share"]})
        else:
            out.append({**common,
                        "control_flip_to_lower_prior": (
                            c_row["selected_move"] != s_row["selected_move"]
                            and c_row["selected_prior"] < s_row["selected_prior"])})
    return out


def dev_safety_v17(rows, coefficient: float, *, shipped_lockin: int,
                   lockin_margin: int = DEV_LOCKIN_MARGIN_V17) -> SafetyVerdict:
    """§7.2 / §8.2 safety, via the v16 aggregator with the per-stratum sub-gate
    disabled (16 late-only targets can never reach `DEV_BAND_MIN_N`)."""
    from .diagnose_fpu_policy_mass import FpuRunConfig
    return dev_safety_verdict(
        _safety_rows(rows, coefficient),
        FpuRunConfig(f"v17_r{coefficient}", None),
        shipped_lockin, shipped_lockin,
        stratum_gate=False, lockin_margin=lockin_margin)


def dev_mechanism_verdict(rows, coefficient: float) -> GateResult:
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


def heldout_verdict(rows, coefficient: float, *, shipped_lockin: int) -> GateResult:
    """§8.2: the same safety table at lock-in margin 2, plus a >=20% transfer
    floor. Failure rejects v17 even when collateral is otherwise safe."""
    _require_positive_grid_coefficient(coefficient, "held-out")
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
    """§9 count-based acceptance for one established probe."""
    if gate not in ABCD_CARDINALITY:
        raise prov.ProtocolViolation(f"unknown A/B/C/D gate {gate!r}")
    _require_positive_grid_coefficient(coefficient, "A/B/C/D")
    expected_n = ABCD_CARDINALITY[gate]
    if n != expected_n:
        raise prov.ProtocolViolation(
            f"gate {gate} has frozen cardinality {expected_n}, got n={n!r}")
    for name, value in (("over", over), ("severe", severe)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= n:
            raise prov.ProtocolViolation(
                f"{gate}.{name} must be an int in [0, {n}], got {value!r}")
    if isinstance(mean, bool) or not isinstance(mean, (int, float)) \
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


def select_smallest_passing(rows, *, shipped_lockin: int
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


# --- Stage 4: A/B/C/D against the Task 1 frozen baseline -------------------

def verify_abcd_baseline(gate: str, shipped_cases: Sequence[Mapping[str, Any]],
                         *, baseline_path: str = ABCD_BASELINE_PATH,
                         moves_path: str = ABCD_MOVES_PATH,
                         tolerance: float = ABCD_TOLERANCE) -> Dict[str, Any]:
    """§9: the contemporaneous SHIPPED run must reproduce the Task 1 frozen
    baseline for every case within `tolerance`, with exact selected moves,
    over/severe counts, and case cardinality. A mismatch makes Stage 4 invalid;
    it does not rebase the gates.
    """
    with open(baseline_path) as f:
        frozen = json.load(f)["abcd_frozen_baseline"][gate]
    with open(moves_path) as f:
        moves = {c["case_id"]: c for c in json.load(f)["gates"][gate]["cases"]}
    if len(shipped_cases) != frozen["n"]:
        raise prov.ProtocolViolation(
            f"gate {gate}: shipped run has {len(shipped_cases)} cases != frozen "
            f"cardinality {frozen['n']}")
    by_id = {c["case_id"]: c for c in frozen["cases"]}
    if {c["case_id"] for c in shipped_cases} != set(by_id):
        raise prov.ProtocolViolation(f"gate {gate}: case ids differ from the frozen set")
    worst = 0.0
    for case in shipped_cases:
        want = float(by_id[case["case_id"]]["probe_black_root_value_repr"])
        delta = abs(float(case["black_value"]) - want)
        worst = max(worst, delta)
        if delta >= tolerance:
            raise prov.ProtocolViolation(
                f"gate {gate} case {case['case_id']}: shipped value differs from "
                f"the frozen baseline by {delta:.3e} >= {tolerance}")
        want_move = moves[case["case_id"]]["selected_move"]
        if list(case["selected_move"]) != list(want_move):
            raise prov.ProtocolViolation(
                f"gate {gate} case {case['case_id']}: shipped selected move "
                f"{case['selected_move']} != frozen {want_move}")
    over = sum(1 for c in shipped_cases if float(c["black_value"]) >= 0.25)
    severe = sum(1 for c in shipped_cases if float(c["black_value"]) >= 0.50)
    if (over, severe) != (frozen["over"], frozen["severe"]):
        raise prov.ProtocolViolation(
            f"gate {gate}: shipped over/severe ({over}, {severe}) != frozen "
            f"({frozen['over']}, {frozen['severe']})")
    return {"gate": gate, "n": frozen["n"], "max_abs_delta": worst,
            "over": over, "severe": severe}


def run_abcd(*, coefficient: float, shipped_by_gate: Mapping[str, Sequence[Mapping]],
             candidate_by_gate: Mapping[str, Sequence[Mapping]],
             a_rows: Sequence[Mapping[str, Any]],
             baseline_path: str = ABCD_BASELINE_PATH,
             moves_path: str = ABCD_MOVES_PATH) -> Dict[str, Any]:
    """All four §9 verdicts, after validating the shipped baseline.

    Every gate must pass; a result merely closer to v14b is not success.
    """
    _require_positive_grid_coefficient(coefficient, "A/B/C/D")
    if set(shipped_by_gate) != set(ABCD_GATES) or set(candidate_by_gate) != set(ABCD_GATES):
        raise prov.ProtocolViolation(
            f"A/B/C/D needs all four gates {ABCD_GATES}, got shipped="
            f"{sorted(shipped_by_gate)} candidate={sorted(candidate_by_gate)}")
    validation = {g: verify_abcd_baseline(g, shipped_by_gate[g],
                                          baseline_path=baseline_path,
                                          moves_path=moves_path)
                  for g in ABCD_GATES}
    gates: Dict[str, Any] = {}
    for g in ABCD_GATES:
        cases = candidate_by_gate[g]
        if len(cases) != ABCD_CARDINALITY[g]:
            raise prov.ProtocolViolation(
                f"gate {g}: candidate has {len(cases)} cases != frozen "
                f"cardinality {ABCD_CARDINALITY[g]}")
        values = [float(c["black_value"]) for c in cases]
        verdict = abcd_verdict(
            g, coefficient=coefficient, n=len(cases), mean=_mean(values),
            over=sum(1 for v in values if v >= 0.25),
            severe=sum(1 for v in values if v >= 0.50),
            a_rows=a_rows if g == "A" else None)
        gates[g] = {"passed": verdict.passed, "reasons": list(verdict.reasons),
                    "metrics": verdict.metrics}
    gates["baseline_validation"] = validation
    gates["all_passed"] = all(gates[g]["passed"] for g in ABCD_GATES)
    return gates


# --- preflight: everything validated BEFORE the evaluator loads ------------

def preflight(*, mode: str, manifest_path: str, checkpoint: str, out_path: str,
              source_index: Optional[str], frozen_coefficient: Optional[float],
              protocol_path: Optional[str] = None,
              config_path: Optional[str] = None) -> Dict[str, Any]:
    """Validate every scientific precondition. Costs zero searches.

    Returns the resolved {configs, manifest_rows, replay_paths, source_files}.
    """
    prov.validate_batching(prov.BATCHING)
    prov.verify_frozen_design()
    prov.require_clean_worktree(mode)
    prov.validate_output_path(out_path)
    configs = configs_for_mode(mode, frozen_coefficient=frozen_coefficient)
    if protocol_path or config_path:
        if not (protocol_path and config_path):
            raise prov.ProtocolViolation(
                "a protocol must be accompanied by its config")
        protocol.load_verified(protocol_path, config_path, consumer_run_kind=mode)
    elif prov.is_scientific(mode):
        raise prov.ProtocolViolation(
            f"scientific mode {mode!r} must run against a verified "
            f"protocol/config pair")
    if prov.is_scientific(mode) and not source_index:
        raise prov.ProtocolViolation(
            f"scientific mode {mode!r} must populate the source_index identity")

    def _readable(label: str, paths: Iterable[str]) -> None:
        for path in paths:
            if not Path(path).is_file():
                raise prov.ProtocolViolation(
                    f"{label} input is not readable at protocol time: {path}")

    # Readability BEFORE parsing, so a missing input is a protocol refusal
    # rather than an OSError from deep inside a loader.
    _readable("checkpoint", [checkpoint])
    _readable("manifest", [manifest_path])
    _readable("source_index", [source_index] if source_index else [])
    manifest_rows = load_manifest(manifest_path)
    replay_paths = sorted({m["replay_path"] for m in manifest_rows})
    module_dir = Path(__file__).resolve().parent
    source_files = [str(module_dir / name) for name in RESULT_DETERMINING_MODULES]
    _readable("replay", replay_paths)
    _readable("source", source_files)
    roles: Dict[str, int] = {}
    for m in manifest_rows:
        roles[m["role"]] = roles.get(m["role"], 0) + 1
    require_corpus_geometry(mode, roles)
    return {"configs": configs, "manifest_rows": manifest_rows,
            "replay_paths": replay_paths, "source_files": source_files}


# --- artifacts -------------------------------------------------------------

def build_artifact(*, mode: str, coefficient: Optional[float],
                   rows: Sequence[Mapping[str, Any]], gates: Mapping[str, Any],
                   checkpoints: Mapping[str, str],
                   effective_mcts_config: Optional[Mapping[str, Any]] = None,
                   manifest: Optional[str] = None,
                   source_index: Optional[str] = None,
                   replay_paths: Optional[Iterable[str]] = None,
                   source_files: Iterable[str] = (),
                   protocol_sha1: Optional[str] = None) -> Dict[str, Any]:
    """A canonical, timestamp-free diagnostic artifact.

    Persists the COMPLETE paired rows and the COMPLETE effective MCTS config
    per coefficient -- a bare coefficient label would not let anyone reproduce
    the search. Scientific modes must populate every identity and be bound to a
    verified protocol.
    """
    if mode not in MODES:
        raise prov.ProtocolViolation(f"unknown mode {mode!r}")
    expected_configs = configs_for_mode(
        mode, frozen_coefficient=None if mode in ("development", "tooling_smoke")
        else coefficient)
    if mode == "development" and coefficient is not None:
        prov.validate_coefficient(coefficient)
    if rows:
        validate_row_set(rows, expected_configs)
        if ZERO in expected_configs:
            require_zero_identity(rows)
        roles: Dict[str, int] = {}
        for sha in {r["canonical_sha1"] for r in rows}:
            role = next(r["role"] for r in rows if r["canonical_sha1"] == sha)
            roles[role] = roles.get(role, 0) + 1
        require_corpus_geometry(mode, roles)
    if prov.is_scientific(mode):
        if not rows and mode != "abcd":
            raise prov.ProtocolViolation(
                f"scientific mode {mode!r} must persist its paired rows")
        if not protocol_sha1:
            raise prov.ProtocolViolation(
                f"scientific mode {mode!r} must be bound to a verified protocol")
        if not effective_mcts_config:
            raise prov.ProtocolViolation(
                f"scientific mode {mode!r} must record the complete effective "
                f"MCTS configuration, not just a coefficient label")
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
        "configs": list(expected_configs),
        "effective_mcts_config": dict(effective_mcts_config or {}),
        "protocol_sha1": protocol_sha1,
        "n_rows": len(rows),
        "rows": [{k: row[k] for k in REQUIRED_ROW_FIELDS} for row in rows],
        "gates": dict(gates),
        "provenance": prov.build_provenance(
            run_kind=mode, coefficient=coefficient, checkpoints=checkpoints,
            source_files=source_files, manifest=manifest,
            source_index=source_index, replay_paths=replay_paths),
    }


# --- row production --------------------------------------------------------

def _need(trace: Mapping[str, Any], key: str, sha: Any) -> Any:
    value = trace.get(key)
    if value is None:
        raise MissingTelemetry(
            f"observer field {key!r} is missing for position {sha!r}; refusing "
            f"to substitute a default, which would turn incomplete telemetry "
            f"into apparently valid data")
    return value


def build_row(*, canonical_sha1: str, role: str, side: str,
              coefficient: Optional[float], seed: int,
              features: Mapping[str, Any], root: Any) -> Dict[str, Any]:
    """One complete scientific-result row. Every metric definition is the
    imported one; absent telemetry raises rather than defaulting to zero."""
    trace = features["trace"]
    leader_move = _need(features, "top_move", canonical_sha1)
    leader = root.children.get(leader_move)
    if leader is None:
        raise MissingTelemetry(
            f"visit leader {leader_move!r} has no child node for position "
            f"{canonical_sha1!r}")
    row = {
        "canonical_sha1": canonical_sha1,
        "role": role,
        "side": side,
        "coefficient": coefficient,
        "seed": int(seed),
        "add_noise": False,
        "selected_move": leader_move,
        "selected_prior": float(_need(trace, "selected_move_prior", canonical_sha1)),
        "selected_prior_rank": int(_need(trace, "selected_move_prior_rank",
                                         canonical_sha1)),
        "root_value_stm": float(features["root_value_stm"]),
        # the PARENT (root) node's own value -- the final parent value
        "parent_value": float(root.q_value),
        # the visit leader's Q, in the mover's perspective, labelled separately
        "selected_child_q": float(-leader.q_value),
        "top_share": float(features["top_share"]),
        "eff_children": float(features["effective_children"]),
        "replies": int(features["replies"]),
        "collapse": bool(features["collapsed"]),
        "lock_in": bool(lock_in_event(trace)),
        "explored_mass": float(_need(trace, "explored_mass_at_stabilization",
                                     canonical_sha1)),
        "stabilization_sim": int(_need(trace, "stabilization_sim", canonical_sha1)),
        "complete": int(_need(trace, "completed_simulation_count",
                              canonical_sha1)) == prov.MCTS_SIMS,
        "tree_signature": [[int(mid), int(ch.visit_count), float(ch.q_value).hex()]
                           for mid, ch in sorted(root.children.items())],
    }
    row["search_result_sha1"] = search_result_sha1({**row, "search_result_sha1": ""})
    return row


def load_manifest(path: str) -> List[dict]:
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


def _effective_config(coefficient: Optional[float]) -> Dict[str, Any]:
    """The COMPLETE effective MCTS configuration for one coefficient."""
    return {"n_simulations": prov.MCTS_SIMS, "add_noise": False,
            "fpu_value": 0.0, prov.CONFIG_FIELD: coefficient,
            "fpu_policy_mass_reduction": None,
            **dict(zip(prov.BATCHING_FIELDS, prov.BATCHING))}


def load_abcd_cases(*, baseline_path: str = ABCD_BASELINE_PATH,
                    moves_path: str = ABCD_MOVES_PATH) -> Dict[str, List[dict]]:
    """The four canonical A/B/C/D case lists, each carrying the frozen per-case
    seed and position identity captured at Task 1."""
    with open(baseline_path) as f:
        base = json.load(f)["abcd_frozen_baseline"]
    with open(moves_path) as f:
        moves = json.load(f)["gates"]
    out: Dict[str, List[dict]] = {}
    for gate in ABCD_GATES:
        by_id = {c["case_id"]: c for c in moves[gate]["cases"]}
        out[gate] = [{"gate": gate, "case_id": c["case_id"],
                      "seed": by_id[c["case_id"]]["seed"],
                      "game_idx": by_id[c["case_id"]]["game_idx"],
                      "position_ply": by_id[c["case_id"]]["position_ply"],
                      "side_to_move": by_id[c["case_id"]]["side_to_move"],
                      "cases_source": base[gate]["canonical_source"]}
                     for c in base[gate]["cases"]]
        if len(out[gate]) != ABCD_CARDINALITY[gate]:
            raise prov.ProtocolViolation(
                f"gate {gate}: frozen baseline has {len(out[gate])} cases != "
                f"cardinality {ABCD_CARDINALITY[gate]}")
    return out


def _a_rows_from(shipped: Sequence[Mapping], candidate: Sequence[Mapping]
                 ) -> List[dict]:
    """The per-case rows `selected_a_verdict` aggregates, built from the paired
    A results rather than restated."""
    by_id = {c["case_id"]: c for c in candidate}
    rows = []
    for s in shipped:
        c = by_id[s["case_id"]]
        rows.append({"off_value": float(s["black_value"]),
                     "r_value": float(c["black_value"]),
                     "replies_ref": int(s["replies"]),
                     "replies_x": int(c["replies"]),
                     "new_collapse": bool(c["collapse"]) and not bool(s["collapse"]),
                     "top_share_inc": float(c["top_share"]) - float(s["top_share"])})
    return rows


def run_abcd_stage(*, coefficient: float, checkpoint: str, out_path: str,
                   protocol_path: str, config_path: str,
                   baseline_path: str = ABCD_BASELINE_PATH,
                   moves_path: str = ABCD_MOVES_PATH,
                   searcher=None) -> Dict[str, Any]:
    """Stage 4. Consumes the four canonical manifests, re-runs shipped and the
    frozen candidate, validates the shipped run against the Task 1 freeze, and
    applies all four §9 verdicts."""
    _require_positive_grid_coefficient(coefficient, "A/B/C/D")
    prov.validate_batching(prov.BATCHING)
    prov.verify_frozen_design()
    prov.require_clean_worktree("abcd")
    prov.validate_output_path(out_path)
    protocol.load_verified(protocol_path, config_path, consumer_run_kind="abcd")
    for label, path in (("checkpoint", checkpoint), ("baseline", baseline_path),
                        ("moves", moves_path)):
        if not Path(path).is_file():
            raise prov.ProtocolViolation(
                f"{label} input is not readable at protocol time: {path}")
    cases = load_abcd_cases(baseline_path=baseline_path, moves_path=moves_path)
    for gate, rows in cases.items():
        for path in {r["cases_source"] for r in rows}:
            if not Path(path).is_file():
                raise prov.ProtocolViolation(
                    f"gate {gate} canonical source is not readable: {path}")
    if searcher is None:                                    # pragma: no cover
        searcher = _real_abcd_searcher(checkpoint)
    shipped = {g: [searcher(c, SHIPPED) for c in cases[g]] for g in ABCD_GATES}
    candidate = {g: [searcher(c, coefficient) for c in cases[g]] for g in ABCD_GATES}
    gates = run_abcd(coefficient=coefficient, shipped_by_gate=shipped,
                     candidate_by_gate=candidate,
                     a_rows=_a_rows_from(shipped["A"], candidate["A"]),
                     baseline_path=baseline_path, moves_path=moves_path)
    module_dir = Path(__file__).resolve().parent
    artifact = build_artifact(
        mode="abcd", coefficient=coefficient, rows=[], gates=gates,
        checkpoints={"anchor": checkpoint},
        effective_mcts_config={str(c): _effective_config(c)
                               for c in (SHIPPED, coefficient)},
        manifest=baseline_path, source_index=moves_path,
        replay_paths=sorted({r["cases_source"] for g in ABCD_GATES
                             for r in cases[g]}),
        source_files=[str(module_dir / n) for n in RESULT_DETERMINING_MODULES],
        protocol_sha1=protocol.protocol_sha1(protocol.load_json(protocol_path)))
    protocol.emit(out_path, artifact)
    return artifact


def _real_abcd_searcher(checkpoint: str):                   # pragma: no cover
    """Real 400-sim MCTS over the canonical probe cases, reusing the same
    harness path the Task 1 capture used."""
    import dataclasses
    import random
    from .capture_v17_abcd_selected_moves import load_gate_cases, mcts_config
    from .eval_runner import _default_evaluator_factory
    from .mcts import MCTS
    from .position_probe_cases import position_state
    evaluator = _default_evaluator_factory(checkpoint)
    base_cfg = mcts_config()
    prov.validate_batching(base_cfg)
    rows_by_gate = {g: {r["case_id"]: r for r in load_gate_cases(g)}
                    for g in ABCD_GATES}

    def search(case, coefficient):
        cfg = dataclasses.replace(
            base_cfg, fpu_shipped_policy_mass_reduction=coefficient)
        prov.validate_batching(cfg)
        src = rows_by_gate[case["gate"]][case["case_id"]]
        replay = json.loads(Path(src["replay_path"]).read_text())
        state = position_state(replay, int(src["position_ply"]),
                               src["side_to_move"])
        counts, root_value = MCTS(evaluator, cfg,
                                  random.Random(case["seed"])).search(
                                      state, add_noise=False)
        total = sum(counts.values())
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top_move, top_visits = ranked[0]
        return {"case_id": case["case_id"],
                "black_value": (root_value if state.to_move == "black"
                                else -root_value),
                "selected_move": [int(top_move[0]), int(top_move[1])],
                "replies": sum(1 for _m, v in ranked if v > 0) - 1,
                "top_share": top_visits / total,
                "collapse": (top_visits / total) >= COLLAPSE_TOP_SHARE}
    return search


def run_diagnostic(*, mode: str, manifest_path: str, checkpoint: str,
                   out_path: str, seed_base: int,
                   frozen_coefficient: Optional[float] = None,
                   source_index: Optional[str] = None,
                   protocol_path: Optional[str] = None,
                   config_path: Optional[str] = None,
                   searcher=None) -> Dict[str, Any]:
    """Search, gate and emit. Every precondition is checked in `preflight`
    before an evaluator is loaded."""
    if mode == "abcd":
        # Stage 4 consumes the four canonical probe manifests, not a dev
        # corpus, so it has its own entry point rather than a degenerate pass
        # through this one.
        if not (protocol_path and config_path):
            raise prov.ProtocolViolation(
                "abcd must run against a verified protocol/config pair")
        return run_abcd_stage(coefficient=frozen_coefficient,
                              checkpoint=checkpoint, out_path=out_path,
                              protocol_path=protocol_path,
                              config_path=config_path, searcher=searcher)
    pre = preflight(mode=mode, manifest_path=manifest_path, checkpoint=checkpoint,
                    out_path=out_path, source_index=source_index,
                    frozen_coefficient=frozen_coefficient,
                    protocol_path=protocol_path, config_path=config_path)
    configs, manifest_rows = pre["configs"], pre["manifest_rows"]
    if searcher is None:                                    # pragma: no cover
        searcher = _real_searcher(checkpoint, seed_base)
    rows = [searcher(m, c) for m in manifest_rows for c in configs]
    validate_row_set(rows, configs)
    if ZERO in configs:
        require_zero_identity(rows)
    shipped_lockin = sum(1 for r in rows if r["coefficient"] is SHIPPED
                         and r["role"] == "target" and r["lock_in"])
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
        checkpoints={"anchor": checkpoint},
        effective_mcts_config={str(c): _effective_config(c) for c in configs},
        manifest=manifest_path, source_index=source_index,
        replay_paths=pre["replay_paths"], source_files=pre["source_files"],
        protocol_sha1=(protocol.protocol_sha1(protocol.load_json(protocol_path))
                       if protocol_path else None))
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
                         role=manifest_row["role"], side=manifest_row["side"],
                         coefficient=coefficient, seed=seed,
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
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--eval-batch-size", type=int, default=prov.BATCHING[0])
    ap.add_argument("--stall-flush-sims", type=int, default=prov.BATCHING[1])
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        prov.validate_batching((args.eval_batch_size, args.stall_flush_sims,
                                prov.BATCHING[2]))
        artifact = run_diagnostic(
            mode=args.mode, manifest_path=args.manifest,
            checkpoint=args.checkpoint, out_path=args.out,
            frozen_coefficient=args.frozen_coefficient,
            seed_base=args.seed_base, source_index=args.source_index,
            protocol_path=args.protocol, config_path=args.config)
    except prov.ProtocolViolation as exc:
        print(f"PROTOCOL VIOLATION: {exc}")
        return protocol.EXIT_USAGE
    print(json.dumps({"mode": artifact["mode"],
                      "coefficient": artifact["coefficient"],
                      "n_rows": artifact["n_rows"]}, sort_keys=True))
    return protocol.EXIT_OK


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
