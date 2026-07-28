"""v17 paired diagnostic -- row production, gates, selection, and artifacts.

Task 5 of the v17 plan. The v16 diagnostic (`diagnose_fpu_policy_mass.py`)
already contains every metric definition, threshold, search helper and observer
this stage needs, so they are IMPORTED here, never restated. The only change
made to that module is a backward-compatible keyword parameterization of
`dev_safety_verdict` (`stratum_gate`, `lockin_margin`), proved byte-identical
against 66 pre-change fixtures.

Frozen design ref:
`docs/superpowers/specs/2026-07-24-v17-baseline-preserving-policy-mass-fpu-design.md`
(SHA-1 pinned once as `fpu_v17_provenance.FROZEN_DESIGN_SHA1`) §§7-9.

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

CONSUMED ARTIFACTS -- everything this stage takes from another tool, with its
producer and the evidence that authenticates it. Nothing here may be accepted
on a caller's word:

  selection manifest CSV   build_fpu_dev_corpus select   bound by membership in
                                                         the hash-pinned screen
  <manifest>.meta.json     build_fpu_dev_corpus select   names every hash below
  sidecar (same file)                                    config/screen/index SHA-1s
  selector config JSON     build_fpu_dev_corpus          screen_meta_provenance.config_sha1
  post-screen report       post-screen-qualify           status PASS + agreeing SHA-1s
  source index JSONL       eval match                    screen_meta_provenance.source_index_sha1
  replay JSON              eval match                    joined via load_game_index
  probe CSV (A/B/C/D)      probe harness                 source_sha1 in the Task 1 freeze
  goal-line manifest (B)   manifest generator            B_MANIFEST_SHA1
  Task 1 freeze            v17 Task 1                    ABCD_BASELINE_SHA1 / ABCD_MOVES_SHA1
  forbidden corpora        v16 / v16a / probes           canonical state hashes,
                                                         via load_forbidden_hashes

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

from . import fpu_provenance
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
    "require_corpus_geometry", "preflight", "authenticate_qualification",
    "dev_safety_v17", "dev_mechanism_verdict", "heldout_verdict",
    "abcd_verdict", "verify_abcd_baseline", "run_abcd",
    "select_smallest_passing", "build_artifact", "build_row", "search_result_sha1",
    "run_diagnostic", "build_selected_coefficient",
    "authenticate_selected_coefficient", "validate_selection_document",
    "validate_development_artifact", "require_development_consistency",
    "selection_context_from", "effective_configs", "gate_record", "main",
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
# §9's black-perspective over/severe convention, as frozen at Task 1.
OVER_THRESHOLD, SEVERE_THRESHOLD = 0.25, 0.50
# What Stage 4 persists per case so the 216 searches remain auditable.
ABCD_CASE_FIELDS = ("gate", "case_id", "seed", "replay_path", "shipped_value",
                    "candidate_value", "shipped_move", "candidate_move",
                    "shipped_replies", "candidate_replies", "shipped_top_share",
                    "candidate_top_share", "shipped_collapse",
                    "candidate_collapse", "value_delta", "move_changed",
                    "frozen_value", "abs_delta_vs_frozen")
ABCD_TOLERANCE = 1e-6
ABCD_BASELINE_PATH = ("logs/eval/fpu_v17_baseline_policy_mass/"
                      "prechange_baseline.json")
ABCD_MOVES_PATH = ("logs/eval/fpu_v17_baseline_policy_mass/"
                   "prechange_abcd_selected_moves.json")
# The Task 1 freeze. Verified at runtime so Stage 4 cannot rebase itself by
# being pointed at a rewritten file.
ABCD_BASELINE_SHA1 = "88cca942334ea7e9335086b0a3d3473f69a4f01e"
# B's cases CSV has no replay_path; the goal-line manifest supplies it.
B_MANIFEST_PATH = "logs/eval/loss_analysis_v2_1/goal_line_trigger_probe_manifest.json"
# B's manifest DETERMINES the case-to-replay mapping, so it is
# authenticated and recorded like any other result-determining input.
B_MANIFEST_SHA1 = "00a3a4220e593791eb4c9eec7973392e5906b0b9"
ABCD_MOVES_SHA1 = "162c9a5a1aac4d4012447d717943e35b405594d9"

# §6.2 / §8.1 corpus geometry, and §5.2's smoke selector profile.
CORPUS_GEOMETRY: Dict[str, Dict[str, int]] = {
    "development": {"target": 16, "control": 16},
    "held_out": {"target": 24, "control": 32},
    "tooling_smoke": {"control": 2},
}
# §6.2/§8.1: exact overall side balance, at most two positions per game, and at
# least twelve plies between two positions taken from the same game.
SIDE_BALANCE: Dict[str, int] = {"development": 16, "held_out": 28}
MAX_POSITIONS_PER_GAME = 2
MIN_PLY_GAP = 12
# §6.2/§8.1 control composition: four per phase for development, eight for
# held-out. The selector certifies these; the diagnostic authenticates them.
CONTROL_PHASE_QUOTA: Dict[str, Dict[str, int]] = {
    "development": {"opening": 4, "early_mid": 4, "midgame": 4, "late": 4},
    "held_out": {"opening": 8, "early_mid": 8, "midgame": 8, "late": 8},
}
# The scientific diagnostic always searches the ANCHOR checkpoint; a protocol
# also names the generation opponent, which must never be searched here.
ANCHOR_ROLE = "anchor"
# The ESTABLISHED selector manifest schema (build_fpu_dev_corpus). Replay paths
# are NOT inline: they are joined from the authenticated source-index JSONL.
MANIFEST_REQUIRED_COLUMNS = ("canonical_position_sha1", "game_idx",
                             "position_ply", "side", "role", "ply_bucket")
# The selector's screen holds EVERY proposal with its eligibility verdict, so
# it can authenticate role/side/phase -- not merely membership.
SCREEN_REQUIRED_COLUMNS = ("canonical_sha1", "game_idx", "ply", "side",
                           "ply_bucket", "raw_policy_role", "anchor_eligible",
                           "exclusion_status")
PHASES = ("opening", "early_mid", "midgame", "late")
# §6.2: zero overlap with all v16 production, v16a neutral, and A/B/C/D
# positions. Disjointness is COMPUTED against these, never self-attested.
FORBIDDEN_CORPORA = (
    "logs/eval/fpu_v16_policy_mass_v2/production_v2_b400amend_4000g_seed20300000/"
    "fpu_dev_corpus_v2_manifest.csv",
    "logs/eval/v16a_fpu_unbiased/neutral_position_manifest.csv",
    "logs/eval/calib020_0001_black_loss_post_opening_predrop_probe/"
    "position_probe_cases.csv",
    "logs/eval/black_predrop_calib010_goal_line/goal_line_trigger_probe_cases.csv",
    "logs/eval/calib020_post_opening_sweep/position_probe_cases.csv",
    "logs/eval/calib020_0001_red_loss_post_opening_predrop_probe/"
    "position_probe_cases.csv",
)
EXPECTED_CHECKED_POSITIONS = {"development": 32, "held_out": 56}
# Frozen SHA-1 for each forbidden corpus that is NOT covered by the Task 1
# freeze. Pinned HERE, in tracked source, rather than read from the selector
# sidecar: the artifacts are untracked, so taking their expected hash from a
# sidecar that travels with them is circular -- editing both would pass. The
# v16 production manifest hash is also recorded in the committed experiment
# ledger, and the v16a manifest is itself git-tracked.
FORBIDDEN_CORPUS_SHA1S = {
    "logs/eval/fpu_v16_policy_mass_v2/production_v2_b400amend_4000g_seed20300000/"
    "fpu_dev_corpus_v2_manifest.csv": "84cdd4b45e089a2ebb292491c146ba00bff17ea9",
    "logs/eval/v16a_fpu_unbiased/neutral_position_manifest.csv":
        "bf7a00ad7ca524ef3aa778b9e0decc11218a2f7d",
}

# Every module whose bytes can change a result. §12 requires source identities
# to cover all of them, not just the module being run.
RESULT_DETERMINING_MODULES = (
    "diagnose_fpu_baseline_policy_mass.py", "diagnose_fpu_policy_mass.py",
    "fpu_v17_protocol.py", "fpu_v17_provenance.py", "fpu_provenance.py",
    "fpu_dev_reservoir_protocol.py", "mcts.py", "eval_runner.py",
    "evaluator.py", "local_evaluator.py", "network.py",
    "position_probe_cases.py", "goal_line_trigger_probe_cases.py",
    "capture_v17_abcd_selected_moves.py", "probe_eval.py",
    "opening_diagnostics.py", "game/twixt_state.py",
    # determines the replay mapping (load_game_index) and the canonical
    # position identity used for disjointness (load_forbidden_hashes)
    "build_fpu_dev_corpus.py", "fpu_state_hash.py",
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
IDENTITY_FIELDS = ("canonical_sha1", "role", "side", "ply_bucket")
REQUIRED_ROW_FIELDS = IDENTITY_FIELDS + ("coefficient",) + SCIENTIFIC_FIELDS
FLOAT_FIELDS = ("selected_prior", "root_value_stm", "parent_value",
                "selected_child_q", "top_share", "eff_children", "explored_mass")
UNIT_INTERVAL_FIELDS = ("top_share", "explored_mass", "selected_prior")

# `explored_mass` alone gets a boundary tolerance. It is a SUM of hundreds of
# float32 priors; when every child is explored the exact total is 1.0, but
# accumulation lands a few ULPs above -- an observed 1.0000000229338184
# (2.3e-8) aborted a full development sweep. The frozen formula already clamps
# (`policy_mass_fpu`: "no clamp here -- policy_mass_fpu clamps"), so such a
# value is expected, not corrupt.
#
# A TOLERANCE, not validation-after-clamping: clamping first would also accept
# 1.5 and make the range check useless. `top_share` and `selected_prior` stay
# STRICT -- neither is an accumulated sum, so neither has this failure mode.
# The artifact keeps the RAW observed value; nothing is normalised or clamped
# on the way to disk, so the hash covers what was actually measured.
UNIT_INTERVAL_TOLERANCE = {"explored_mass": 1e-6}
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
            tol = UNIT_INTERVAL_TOLERANCE.get(name, 0.0)
            if not -tol <= row[name] <= 1.0 + tol:
                raise prov.ProtocolViolation(
                    f"{name}={row[name]!r} outside [0, 1]"
                    + (f" (tolerance {tol:g})" if tol else "")
                    + f" for position {sha!r}")
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
    # §7 runs every coefficient "using identical per-position seeds": a paired
    # positive row searched under a different seed is not comparable.
    seeds: Dict[Any, set] = {}
    for row in rows:
        seeds.setdefault(row["canonical_sha1"], set()).add(row["seed"])
    mixed = sorted(sha for sha, vs in seeds.items() if len(vs) > 1)
    if mixed:
        raise prov.ProtocolViolation(
            f"per-position seed drift: position(s) {mixed[:3]} were searched "
            f"under more than one seed, so their configs are not comparable")
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


def require_corpus_geometry(mode: str, manifest_rows: Sequence[Mapping[str, Any]]
                            ) -> None:
    """§6.2/§8.1: the COMPLETE frozen corpus shape.

    Role counts alone are not the geometry -- an all-red 24/32 held-out corpus
    satisfies them while violating the frozen 28/28 side balance. This checks
    roles, overall side balance, canonical-position uniqueness, the
    at-most-two-per-game cap, and the twelve-ply spacing.
    """
    expected = CORPUS_GEOMETRY.get(mode)
    if expected is None:
        return
    roles: Dict[str, int] = {}
    sides: Dict[str, int] = {}
    per_game: Dict[Any, List[int]] = {}
    seen_sha: Dict[Any, int] = {}
    for r in manifest_rows:
        roles[r["role"]] = roles.get(r["role"], 0) + 1
        sides[r["side"]] = sides.get(r["side"], 0) + 1
        per_game.setdefault(r["game_idx"], []).append(int(float(r["position_ply"])))
        seen_sha[r["canonical_sha1"]] = seen_sha.get(r["canonical_sha1"], 0) + 1
    actual_roles = {k: roles.get(k, 0) for k in set(expected) | set(roles)}
    if actual_roles != dict(expected):
        raise prov.ProtocolViolation(
            f"{mode} corpus roles {actual_roles} != the frozen {dict(expected)}")
    want_side = SIDE_BALANCE.get(mode)
    if want_side is not None and sides != {"red": want_side, "black": want_side}:
        raise prov.ProtocolViolation(
            f"{mode} corpus side balance {sides} != the frozen "
            f"{{'red': {want_side}, 'black': {want_side}}}")
    want_quota = CONTROL_PHASE_QUOTA.get(mode)
    if want_quota is not None:
        phases: Dict[str, int] = {p: 0 for p in PHASES}
        for r in manifest_rows:
            if r["role"] != "control":
                continue
            bucket = r.get("ply_bucket")
            if bucket not in phases:
                raise prov.ProtocolViolation(
                    f"{mode} control row has ply_bucket {bucket!r}, not one of "
                    f"{PHASES}")
            phases[bucket] += 1
        if phases != want_quota:
            raise prov.ProtocolViolation(
                f"{mode} control phase quotas {phases} != the frozen {want_quota}")
    dupes = sorted(s for s, n in seen_sha.items() if n > 1)
    if dupes:
        raise prov.ProtocolViolation(
            f"{mode} corpus repeats canonical position(s) {dupes[:3]}")
    for game, plies in per_game.items():
        if len(plies) > MAX_POSITIONS_PER_GAME:
            raise prov.ProtocolViolation(
                f"{mode} corpus takes {len(plies)} positions from game {game!r}; "
                f"the frozen cap is {MAX_POSITIONS_PER_GAME}")
        ordered = sorted(plies)
        for a, b in zip(ordered, ordered[1:]):
            if b - a < MIN_PLY_GAP:
                raise prov.ProtocolViolation(
                    f"{mode} corpus game {game!r} has positions {a} and {b}, "
                    f"closer than the frozen {MIN_PLY_GAP}-ply spacing")


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
        value = case["black_value"]
        # NaN must be rejected BEFORE any comparison: abs(NaN - x) is NaN, and
        # every comparison against NaN is false, so a corrupted value would
        # otherwise pass both the delta check and the over/severe counts.
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value):
            raise prov.ProtocolViolation(
                f"gate {gate} case {case['case_id']}: shipped value {value!r} is "
                f"not a finite number; it cannot authenticate against the "
                f"frozen baseline")
        want = float(by_id[case["case_id"]]["probe_black_root_value_repr"])
        delta = abs(float(value) - want)
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
    over = sum(1 for c in shipped_cases if float(c["black_value"]) >= OVER_THRESHOLD)
    severe = sum(1 for c in shipped_cases if float(c["black_value"]) >= SEVERE_THRESHOLD)
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
            over=sum(1 for v in values if v >= OVER_THRESHOLD),
            severe=sum(1 for v in values if v >= SEVERE_THRESHOLD),
            a_rows=a_rows if g == "A" else None)
        gates[g] = gate_record(verdict)
    gates["baseline_validation"] = validation
    gates["all_passed"] = all(gates[g]["passed"] for g in ABCD_GATES)
    return gates


# --- preflight: everything validated BEFORE the evaluator loads ------------

def preflight(*, mode: str, manifest_path: str, checkpoint: str, out_path: str,
              source_index: Optional[str], frozen_coefficient: Optional[float],
              protocol_path: Optional[str] = None,
              config_path: Optional[str] = None,
              qualification_report: Optional[str] = None,
              stage1_manifest: Optional[str] = None,
              stage1_source_index: Optional[str] = None,
              stage1_post_screen_report: Optional[str] = None,
              qualification_selected_coefficient: Optional[str] = None,
              selector=None) -> Dict[str, Any]:
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
        verified = protocol.load_verified(protocol_path, config_path,
                                          consumer_run_kind=mode)
        bind_protocol_to_runtime(verified, coefficient=frozen_coefficient,
                                 checkpoint=checkpoint)
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
    if prov.is_scientific(mode) and not source_index:
        raise prov.ProtocolViolation(
            f"scientific mode {mode!r} needs the source-index JSONL to resolve "
            f"replay paths")
    manifest_rows = load_manifest(manifest_path, source_index=source_index)
    replay_paths = sorted({m["replay_path"] for m in manifest_rows})
    module_dir = Path(__file__).resolve().parent
    source_files = [str(module_dir / name) for name in RESULT_DETERMINING_MODULES]
    _readable("replay", replay_paths)
    _readable("source", source_files)
    require_corpus_geometry(mode, manifest_rows)
    # Geometry is necessary but not sufficient: it accepts arbitrary rows
    # LABELLED target/control. The selector's report is what certifies the
    # predicates, the phase quotas, and the disjointness §12 requires.
    # ZERO-GPU authorization first: a later stage bound to the wrong selection
    # artifact must cost no searches at all.
    selection_binding = None
    if mode in ("held_out", "abcd"):
        selection_binding = authenticate_selected_coefficient(
            qualification_selected_coefficient, coefficient=frozen_coefficient,
            expected_sha1=_precommitted_selection_sha1(protocol_path))
    qualification = None
    disjointness = None
    if prov.is_scientific(mode):
        qualification = authenticate_qualification(
            manifest_path, mode=mode, source_index=source_index,
            config_path=config_path, checkpoint=checkpoint,
            post_screen_report=qualification_report, selector=selector)
        sidecar = json.loads(Path(manifest_path + ".meta.json").read_text())
        forbidden = list(FORBIDDEN_CORPORA)
        expected_sha1s = dict(expected_forbidden_sha1s(sidecar))
        forbidden_game_identities: Dict[str, Mapping[str, str]] = {}
        if mode == "held_out":
            # §8.1: "Complete game and position disjointness from Stage 1".
            if not (stage1_manifest and stage1_source_index):
                raise prov.ProtocolViolation(
                    "held-out requires the Stage-1 development manifest AND its "
                    "source index to establish complete game and position "
                    "disjointness (§8.1)")
            # Stage 1 is EVIDENCE here, so it is authenticated exactly as this
            # run's own corpus is -- a readable CSV is not proof.
            stage1_qualification = authenticate_qualification(
                stage1_manifest, mode="development",
                source_index=stage1_source_index, config_path=None,
                checkpoint=checkpoint,
                post_screen_report=stage1_post_screen_report,
                selector=selector)
            stage1_rows = load_manifest(stage1_manifest,
                                        source_index=stage1_source_index)
            forbidden.append(stage1_manifest)
            expected_sha1s[stage1_manifest] = fpu_provenance.file_sha1(
                stage1_manifest)
            forbidden_game_identities["stage1_development"] = game_identities(
                stage1_rows)
            # Stage 1 is evidence for this artifact, so its identities are
            # PERSISTED, not discarded once checked.
            qualification = {**(qualification or {}),
                             "stage1": {**stage1_qualification,
                                        "manifest": stage1_manifest,
                                        "manifest_sha1": fpu_provenance.file_sha1(
                                            stage1_manifest),
                                        "rows": len(stage1_rows)}}
        disjointness = compute_disjointness(
            manifest_rows, forbidden=forbidden, expected_sha1s=expected_sha1s,
            forbidden_game_identities=forbidden_game_identities)
        if disjointness["game_overlaps"]:
            raise prov.ProtocolViolation(
                f"held-out shares games with Stage 1: "
                f"{disjointness['game_overlaps'][:2]}")
        if disjointness["overlaps"]:
            raise prov.ProtocolViolation(
                f"corpus overlaps forbidden evidence: "
                f"{disjointness['overlaps'][:2]}")
        want = EXPECTED_CHECKED_POSITIONS.get(mode)
        if want is not None and disjointness["checked_positions"] != want:
            raise prov.ProtocolViolation(
                f"{mode} disjointness checked "
                f"{disjointness['checked_positions']} positions, expected {want}")
    return {"configs": configs, "manifest_rows": manifest_rows,
            "replay_paths": replay_paths, "source_files": source_files,
            "qualification": qualification, "disjointness": disjointness,
            "selection_binding": selection_binding}


# --- artifacts -------------------------------------------------------------

SELECTED_COEFFICIENT_KEYS = ("schema_version", "artifact_kind", "coefficient",
                             "grid", "gates", "selection_rule",
                             "development_artifact",
                             "development_artifact_sha1", "selection_context")
# The EXACT selection-rule identifier. A document is not free to relabel the
# rule it claims to have applied.
SELECTION_RULE = "smallest coefficient passing §§7.2-7.3"
DIAGNOSTIC_ARTIFACT_KEYS = ("schema_version", "artifact_kind", "mode",
                            "coefficient", "configs", "effective_mcts_config",
                            "protocol_sha1", "n_rows", "rows", "n_cases",
                            "cases", "identities", "gates", "provenance")
SELECTION_CONTEXT_KEYS = ("manifest_sha1", "protocol_sha1", "checkpoint_sha1",
                          "n_rows")


def build_selected_coefficient(*, coefficient: Optional[float],
                               table: Mapping[float, "GateResult"],
                               development_artifact: str,
                               development_artifact_sha1: str,
                               development: Mapping[str, Any]
                               ) -> Dict[str, Any]:
    """The immutable Stage-2 result held-out and A/B/C/D must be bound to.

    Records the whole grid table, not just the winner, so the smallest-passing
    rule can be re-checked rather than trusted.
    """
    return {
        "schema_version": prov.SCHEMA_VERSION,
        "artifact_kind": "selected_coefficient",
        "coefficient": coefficient,
        "grid": list(prov.GRID),
        "selection_rule": SELECTION_RULE,
        "gates": {str(r): {"passed": g.passed, "reasons": list(g.reasons)}
                  for r, g in sorted(table.items())},
        "development_artifact": development_artifact,
        "development_artifact_sha1": development_artifact_sha1,
        # derived, never passed in, so producer and verifier cannot diverge
        "selection_context": selection_context_from(development),
    }


def _strict_document(doc: Any, *, path: str, kind: str, keys: Sequence[str],
                     schema_version: int) -> Dict[str, Any]:
    """Exact schema-version, artifact-kind and key-set validation.

    Precommitting a document's BYTES fixes which document is used; it does not
    establish that the document is a supported artifact. A selection document
    with schema_version 999, and a development document declaring
    artifact_kind "not_a_diagnostic", both authenticated on their SHA-1 alone.
    """
    if not isinstance(doc, dict):
        raise prov.ProtocolViolation(f"{path} is not a JSON object")
    if doc.get("artifact_kind") != kind:
        raise prov.ProtocolViolation(
            f"{path} has artifact_kind {doc.get('artifact_kind')!r}, expected "
            f"{kind!r}")
    # `True == 1` and `1.0 == 1` in Python, so the TYPE is checked before the
    # value: equality alone is not exact schema typing.
    version = doc.get("schema_version")
    if type(version) is not int or version != schema_version:
        raise prov.ProtocolViolation(
            f"{path} has schema_version {version!r} ({type(version).__name__}), "
            f"but only int {schema_version} is supported")
    missing = sorted(set(keys) - set(doc))
    unknown = sorted(set(doc) - set(keys))
    if missing or unknown:
        raise prov.ProtocolViolation(
            f"{path} key set is wrong: missing {missing}, unknown {unknown}")
    return doc


def _require_type(value: Any, types: tuple, *, path: str, field: str) -> Any:
    if isinstance(value, bool) and bool not in types:
        raise prov.ProtocolViolation(
            f"{path} field {field!r} is a bool, expected "
            f"{'/'.join(t.__name__ for t in types)}")
    if not isinstance(value, types):
        raise prov.ProtocolViolation(
            f"{path} field {field!r} is {type(value).__name__}, expected "
            f"{'/'.join(t.__name__ for t in types)}")
    return value


def validate_development_artifact(doc: Any, *, path: str) -> Dict[str, Any]:
    """Strict shape + internal consistency for a development diagnostic."""
    _strict_document(doc, path=path, kind="diagnostic",
                     keys=DIAGNOSTIC_ARTIFACT_KEYS,
                     schema_version=prov.SCHEMA_VERSION)
    if doc["mode"] != "development":
        raise prov.ProtocolViolation(
            f"{path} is a {doc['mode']!r} artifact, not development")
    _require_type(doc["rows"], (list,), path=path, field="rows")
    _require_type(doc["n_rows"], (int,), path=path, field="n_rows")
    _require_type(doc["gates"], (dict,), path=path, field="gates")
    _require_type(doc["configs"], (list,), path=path, field="configs")
    _require_type(doc["effective_mcts_config"], (dict,), path=path,
                  field="effective_mcts_config")
    if not doc["rows"]:
        raise prov.ProtocolViolation(
            f"{path} persists no rows, so the selection cannot be recomputed")
    if doc["n_rows"] != len(doc["rows"]):
        raise prov.ProtocolViolation(
            f"{path} records n_rows={doc['n_rows']} but persists "
            f"{len(doc['rows'])} rows")
    expected_configs = list(configs_for_mode("development"))
    if doc["configs"] != expected_configs:
        raise prov.ProtocolViolation(
            f"{path} configs {doc['configs']} != the frozen development set "
            f"{expected_configs}")
    missing_cfg = sorted({str(c) for c in expected_configs}
                         - set(doc["effective_mcts_config"]))
    if missing_cfg:
        raise prov.ProtocolViolation(
            f"{path} records no effective MCTS config for {missing_cfg}")
    validate_row_set(doc["rows"], expected_configs)
    return doc


def gate_record(gate: "GateResult") -> Dict[str, Any]:
    """The persisted form of one gate verdict. Used by BOTH the producer and
    the verifier, so the two cannot drift apart in shape or content."""
    return {"passed": gate.passed, "reasons": list(gate.reasons),
            "metrics": dict(gate.metrics)}


def require_development_consistency(development: Mapping[str, Any], *,
                                    recomputed: Optional[float],
                                    table: Mapping[float, "GateResult"],
                                    path: str) -> None:
    """The development artifact's own result fields must equal recomputation.

    Recomputing from its rows and comparing only against the SEPARATE selection
    document left the artifact free to contradict itself: it could report
    coefficient 0.45, a forged gate table, and empty MCTS configs while its own
    rows select 0.25.
    """
    if development["coefficient"] != recomputed:
        raise prov.ProtocolViolation(
            f"{path} records coefficient {development['coefficient']!r}, but its "
            f"own rows recompute to {recomputed!r}")
    gates = development["gates"]
    if sorted(gates) != sorted(str(r) for r in table):
        raise prov.ProtocolViolation(
            f"{path} gate table covers {sorted(gates)}, but recomputation "
            f"produced {sorted(str(r) for r in table)}")
    for r, gate in table.items():
        # EXACT canonical comparison of the COMPLETE gate record. Comparing
        # only the verdict and reasons left `metrics` -- a required part of the
        # persisted scientific result -- free to be fabricated. Whole-record
        # equality also refuses missing or unknown gate fields.
        expected = canonical_json_bytes(gate_record(gate))
        claimed = canonical_json_bytes(gates[str(r)])
        if claimed != expected:
            raise prov.ProtocolViolation(
                f"{path} gate record at r={r} does not match recomputation; "
                f"recorded {json.dumps(gates[str(r)], sort_keys=True)[:200]}, "
                f"recomputed {json.dumps(gate_record(gate), sort_keys=True)[:200]}")
    # the effective configs must EQUAL the frozen ones, not merely be keyed
    expected = json.loads(json.dumps(
        effective_configs(configs_for_mode("development"))))
    if json.loads(json.dumps(development["effective_mcts_config"])) != expected:
        raise prov.ProtocolViolation(
            f"{path} effective MCTS configurations do not equal the frozen "
            f"configurations for the development config set")


def validate_selection_document(doc: Any, *, path: str) -> Dict[str, Any]:
    """Strict shape for a selected-coefficient artifact."""
    _strict_document(doc, path=path, kind="selected_coefficient",
                     keys=SELECTED_COEFFICIENT_KEYS,
                     schema_version=prov.SCHEMA_VERSION)
    if doc["selection_rule"] != SELECTION_RULE:
        raise prov.ProtocolViolation(
            f"{path} claims selection_rule {doc['selection_rule']!r}, not the "
            f"frozen {SELECTION_RULE!r}")
    _require_type(doc["grid"], (list,), path=path, field="grid")
    _require_type(doc["gates"], (dict,), path=path, field="gates")
    _require_type(doc["development_artifact"], (str,), path=path,
                  field="development_artifact")
    _require_type(doc["development_artifact_sha1"], (str,), path=path,
                  field="development_artifact_sha1")
    context = _require_type(doc["selection_context"], (dict,), path=path,
                            field="selection_context")
    if doc["coefficient"] is not None:
        _require_type(doc["coefficient"], (int, float), path=path,
                      field="coefficient")
    if set(context) != set(SELECTION_CONTEXT_KEYS):
        raise prov.ProtocolViolation(
            f"{path} selection_context keys {sorted(context)} != "
            f"{sorted(SELECTION_CONTEXT_KEYS)}")
    for gate in doc["gates"].values():
        if set(gate) != {"passed", "reasons"}:
            raise prov.ProtocolViolation(
                f"{path} gate entries must have exactly passed/reasons, got "
                f"{sorted(gate)}")
        _require_type(gate["passed"], (bool,), path=path, field="gate.passed")
        _require_type(gate["reasons"], (list,), path=path, field="gate.reasons")
    return doc


def selection_context_from(development: Mapping[str, Any]) -> Dict[str, Any]:
    """The complete §7.3 selection-context fingerprint, DERIVED from the
    development artifact rather than accepted from the selection document."""
    identities = (development.get("provenance") or {}).get("identities") or {}
    checkpoints = (development.get("provenance") or {}).get("checkpoints") or {}
    return {"manifest_sha1": identities.get("manifest_sha1"),
            "protocol_sha1": development.get("protocol_sha1"),
            "checkpoint_sha1": checkpoints.get(ANCHOR_ROLE),
            "n_rows": development.get("n_rows")}


def authenticate_selected_coefficient(path: Optional[str], *,
                                      coefficient: Optional[float],
                                      expected_sha1: Optional[str] = None
                                      ) -> Dict[str, Any]:
    """§7.3/§8: held-out and A/B/C/D run the coefficient development SELECTED.

    The artifact is required, its schema checked, its smallest-passing rule
    RE-CHECKED from the recorded table, and the runtime coefficient must equal
    it exactly. Without this, any grid point could be protocol-bound and run.
    """
    if not path or not Path(path).is_file():
        raise prov.ProtocolViolation(
            "a later stage must be bound to development's immutable "
            f"selected_coefficient artifact; got {path!r}")
    # The later stage's PROTOCOL precommits this artifact's SHA-1, so a
    # fabricated file cannot be substituted at run time.
    actual = fpu_provenance.file_sha1(path)
    if not expected_sha1:
        raise prov.ProtocolViolation(
            "the later-stage protocol must precommit the selected-coefficient "
            "artifact's SHA-1")
    if actual != expected_sha1:
        raise prov.ProtocolViolation(
            f"selected-coefficient artifact {path} has SHA-1 {actual}, but its "
            f"protocol precommitted {expected_sha1}")
    with open(path) as f:
        doc = validate_selection_document(json.load(f), path=path)
    if list(doc["grid"]) != list(prov.GRID):
        raise prov.ProtocolViolation(
            f"selected-coefficient artifact records grid {doc['grid']}, not the "
            f"frozen {list(prov.GRID)}")
    if sorted(float(r) for r in doc["gates"]) != sorted(prov.GRID):
        raise prov.ProtocolViolation(
            f"selected-coefficient gate table covers "
            f"{sorted(doc['gates'])}, not the whole frozen grid")
    # RECOMPUTE the table and the selection from the development artifact's own
    # persisted rows. Its internal claims are not evidence: a fabricated file
    # with one asserted pass authenticated before this.
    dev_path = doc["development_artifact"]
    if not Path(dev_path).is_file():
        raise prov.ProtocolViolation(
            f"development artifact {dev_path} is not readable, so the selection "
            f"cannot be recomputed")
    dev_actual = fpu_provenance.file_sha1(dev_path)
    if dev_actual != doc["development_artifact_sha1"]:
        raise prov.ProtocolViolation(
            f"development artifact {dev_path} has SHA-1 {dev_actual}, but the "
            f"selection artifact records {doc['development_artifact_sha1']}")
    with open(dev_path) as f:
        development = validate_development_artifact(json.load(f), path=dev_path)
    dev_rows = development["rows"]
    shipped_lockin = sum(1 for r in dev_rows if r["coefficient"] is None
                         and r["role"] == "target" and r["lock_in"])
    recomputed, table = select_smallest_passing(dev_rows,
                                                shipped_lockin=shipped_lockin)
    if recomputed != doc["coefficient"]:
        raise prov.ProtocolViolation(
            f"selected-coefficient artifact records {doc['coefficient']!r}, but "
            f"recomputing §§7.2-7.3 from the development artifact's own rows "
            f"selects {recomputed!r}")
    require_development_consistency(development, recomputed=recomputed,
                                    table=table, path=dev_path)
    for r, gate in table.items():
        claimed = doc["gates"].get(str(r))
        if claimed is None or claimed["passed"] != gate.passed:
            raise prov.ProtocolViolation(
                f"selected-coefficient gate table disagrees with the recomputed "
                f"verdict at r={r}")
        # the REASONS are evidence too, so they are compared rather than shown
        if list(claimed.get("reasons", [])) != list(gate.reasons):
            raise prov.ProtocolViolation(
                f"selected-coefficient gate reasons at r={r} are "
                f"{claimed.get('reasons')!r}, but recomputation produced "
                f"{list(gate.reasons)!r}")
    # The context is DERIVED from the authenticated development artifact and
    # exact-matched. Returned directly, it accepted forged manifest/protocol/
    # checkpoint hashes and n_rows=0 alongside a genuine artifact.
    derived_context = selection_context_from(development)
    if dict(doc["selection_context"]) != derived_context:
        raise prov.ProtocolViolation(
            f"selected-coefficient selection_context {doc['selection_context']!r} "
            f"!= the context derived from the development artifact "
            f"{derived_context!r}")
    if doc["coefficient"] is None:
        raise prov.ProtocolViolation(
            "development selected no coefficient, so no later stage may run")
    if float(doc["coefficient"]) != float(coefficient):
        raise prov.ProtocolViolation(
            f"runtime coefficient {coefficient!r} != development's selected "
            f"{doc['coefficient']!r}")
    return {"path": path, "sha1": actual,
            "coefficient": doc["coefficient"],
            "development_artifact": dev_path,
            "development_artifact_sha1": dev_actual,
            "recomputed_from_rows": len(dev_rows),
            "selection_context": derived_context}


def build_artifact(*, mode: str, coefficient: Optional[float],
                   rows: Sequence[Mapping[str, Any]], gates: Mapping[str, Any],
                   checkpoints: Mapping[str, str],
                   case_rows: Optional[Sequence[Mapping[str, Any]]] = None,
                   extra_identities: Optional[Mapping[str, Any]] = None,
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
        by_sha = {}
        for r in rows:
            by_sha.setdefault(r["canonical_sha1"], r)
        require_corpus_geometry(mode, [
            {"role": r["role"], "side": r["side"], "canonical_sha1": sha,
             "ply_bucket": r["ply_bucket"], "game_idx": sha, "position_ply": 0}
            for sha, r in by_sha.items()])
    if prov.is_scientific(mode):
        if mode == "abcd":
            if not case_rows:
                raise prov.ProtocolViolation(
                    "abcd must persist its paired per-case evidence")
            expected_cases = sum(ABCD_CARDINALITY.values())
            if len(case_rows) != expected_cases:
                raise prov.ProtocolViolation(
                    f"abcd persisted {len(case_rows)} cases != the frozen "
                    f"{expected_cases}")
            for case in case_rows:
                missing = [f for f in ABCD_CASE_FIELDS if f not in case]
                if missing:
                    raise prov.ProtocolViolation(
                        f"abcd case {case.get('case_id')!r} missing {missing}")
        elif not rows:
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
        "n_cases": len(case_rows or ()),
        "cases": [{k: c[k] for k in ABCD_CASE_FIELDS} for c in (case_rows or ())],
        "identities": dict(extra_identities or {}),
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


def build_row(*, canonical_sha1: str, role: str, side: str, ply_bucket: str,
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
        "ply_bucket": ply_bucket,
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


def load_manifest(path: str, *, source_index: str) -> List[dict]:
    """Read the ESTABLISHED selector manifest and resolve replay paths.

    The real selector emits `canonical_position_sha1` and no `replay_path`; the
    replay for a row is found by joining `game_idx` against the authenticated
    source-index JSONL, exactly as the v16 diagnostic does. An earlier draft
    required an inline `replay_path` and a `canonical_sha1` column, so a real
    Task-10 manifest could not reach Stage 2 at all.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise prov.ProtocolViolation(f"empty manifest {path}")
    missing = sorted(set(MANIFEST_REQUIRED_COLUMNS) - set(rows[0]))
    if missing:
        raise prov.ProtocolViolation(f"manifest {path} missing columns {missing}")
    from .build_fpu_dev_corpus import load_game_index
    replay_by_game = {int(r["game_idx"]): r["replay_path"]
                      for r in load_game_index(source_index)}
    out: List[dict] = []
    for r in rows:
        game_idx = int(r["game_idx"])
        if game_idx not in replay_by_game:
            raise prov.ProtocolViolation(
                f"manifest row for game {game_idx} has no record in the source "
                f"index {source_index}")
        out.append({**r,
                    "canonical_sha1": r["canonical_position_sha1"],
                    "game_idx": game_idx,
                    "position_ply": int(float(r["position_ply"])),
                    "replay_path": replay_by_game[game_idx]})
    return out


def base_mcts_config():
    """The v17 base `MCTSConfig`, built the same way every stage builds it.

    Imported lazily so this module stays MLX-free at import.
    """
    from .eval_runner import EvalConfig, cfg_from
    cfg = cfg_from(EvalConfig(mcts_sims=prov.MCTS_SIMS,
                              mcts_eval_batch_size=prov.BATCHING[0],
                              mcts_stall_flush_sims=prov.BATCHING[1]))
    prov.validate_batching(cfg)
    return cfg


def effective_configs(configs: Sequence[Optional[float]]) -> Dict[str, Any]:
    """The COMPLETE effective `MCTSConfig` per coefficient -- every field of the
    dataclass, from the exact object handed to MCTS.

    A hand-built subset cannot reveal drift in `c_puct`, the temperatures, the
    root penalties, the opening settings, or the closeout controls, so the full
    `dataclasses.asdict` is persisted instead.
    """
    import dataclasses
    base = base_mcts_config()
    out: Dict[str, Any] = {}
    for c in configs:
        cfg = dataclasses.replace(base, fpu_shipped_policy_mass_reduction=c)
        prov.validate_batching(cfg)
        out[str(c)] = {"add_noise": False, **dataclasses.asdict(cfg)}
    return out


def bind_protocol_to_runtime(config: Mapping[str, Any], *,
                             coefficient: Optional[float], checkpoint: str,
                             role: str = ANCHOR_ROLE) -> None:
    """The verified config must CONSTRAIN the run, not merely accompany it.

    The checkpoint is bound to its required ROLE, not to set membership: a
    development protocol names both `calib020_0001` and `0379`, but the
    diagnostic must search the anchor. Membership-testing the unordered set
    accepted the generation opponent.
    """
    want = config.get("coefficient")
    if (want is None) != (coefficient is None) or (
            want is not None and float(want) != float(coefficient)):
        raise prov.ProtocolViolation(
            f"runtime coefficient {coefficient!r} does not match the verified "
            f"protocol's {want!r}")
    protocol_ckpts = config.get("checkpoints") or {}
    if role not in protocol_ckpts:
        raise prov.ProtocolViolation(
            f"the verified protocol has no {role!r} checkpoint to bind against; "
            f"it names {sorted(protocol_ckpts)}")
    expected = fpu_provenance.file_sha1(protocol_ckpts[role])
    actual = fpu_provenance.file_sha1(checkpoint)
    if actual != expected or actual in prov.SENTINEL_HASHES:
        raise prov.ProtocolViolation(
            f"runtime checkpoint {checkpoint} (sha1 {actual}) is not the "
            f"protocol's {role!r} checkpoint {protocol_ckpts[role]} "
            f"(sha1 {expected})")


def expected_forbidden_sha1s(sidecar: Optional[Mapping[str, Any]] = None
                             ) -> Dict[str, str]:
    """Frozen SHA-1 per forbidden corpus, from AUTHENTICATED sources only:
    the A/B/C/D `source_sha1`s inside the Task 1 freeze, and the selector
    sidecar's `forbidden_manifest_sha1s` for the v16/v16a manifests.

    These files are untracked historical artifacts, so clean-worktree
    enforcement does not protect them; altered bytes would silently change the
    collision set.
    """
    _authenticate_task1_freeze(ABCD_BASELINE_PATH, ABCD_MOVES_PATH)
    with open(ABCD_BASELINE_PATH) as f:
        base = json.load(f)["abcd_frozen_baseline"]
    expected = {g["canonical_source"]: g["source_sha1"] for g in base.values()}
    by_name = ((sidecar or {}).get("screen_meta_provenance") or {}).get(
        "forbidden_manifest_sha1s") or {}
    for path in FORBIDDEN_CORPORA:
        if path in expected:
            continue
        want = FORBIDDEN_CORPUS_SHA1S.get(path)
        if not want:
            raise prov.ProtocolViolation(
                f"no authenticated SHA-1 is available for forbidden corpus "
                f"{path}; disjointness against unverified bytes proves nothing")
        # The sidecar is a CROSS-CHECK, never the root: if it disagrees with the
        # tracked pin, one of them has been edited.
        sidecar_says = by_name.get(Path(path).name)
        if sidecar_says and sidecar_says != want:
            raise prov.ProtocolViolation(
                f"selector sidecar records {path} as {sidecar_says}, but the "
                f"tracked frozen pin is {want}; one of them has been edited")
        expected[path] = want
    return expected


def authenticate_forbidden_corpora(expected: Mapping[str, str]) -> Dict[str, str]:
    """Verify every forbidden corpus still has its frozen bytes."""
    verified: Dict[str, str] = {}
    for path, want in sorted(expected.items()):
        actual = fpu_provenance.file_sha1(path)
        if actual != want:
            raise prov.ProtocolViolation(
                f"forbidden corpus {path} has SHA-1 {actual}, expected the "
                f"frozen {want}; altered bytes could silently remove a "
                f"collision")
        verified[path] = actual
    return verified


def forbidden_position_hashes(path: str) -> set:
    """Canonical position hashes for one forbidden corpus.

    Uses the ESTABLISHED `build_fpu_dev_corpus.load_forbidden_hashes`: read the
    `canonical_position_sha1` column when present, otherwise reconstruct the
    state and hash it. `(game_idx, position_ply)` is NOT an identity -- game
    indices are reservoir-local, so the same position reached from a different
    reservoir carries a different index and an exact canonical collision was
    missed.

    B's goal-line cases carry no `replay_path`, so its authenticated manifest is
    joined in first and the reconstruction runs over the joined rows.
    """
    from .build_fpu_dev_corpus import load_forbidden_hashes
    from .fpu_state_hash import canonical_state_sha1
    from .goal_line_trigger_probe_cases import position_state
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError as exc:
        raise prov.ProtocolViolation(
            f"forbidden corpus {path} is not readable, so disjointness cannot "
            f"be established: {exc}") from exc
    if not rows:
        return set()
    if "canonical_position_sha1" in rows[0] or "replay_path" in rows[0]:
        return load_forbidden_hashes([path])
    # B: join replay paths from its authenticated manifest, then reconstruct.
    b_actual = fpu_provenance.file_sha1(B_MANIFEST_PATH)
    if b_actual != B_MANIFEST_SHA1:
        raise prov.ProtocolViolation(
            f"gate B manifest {B_MANIFEST_PATH} has SHA-1 {b_actual}, expected "
            f"the frozen {B_MANIFEST_SHA1}; it determines the replay join")
    with open(B_MANIFEST_PATH) as f:
        by_key = {(int(c["game_idx"]), int(c["position_ply"])): c["replay_path"]
                  for c in json.load(f)["cases"]}
    out = set()
    for r in rows:
        key = (int(r["game_idx"]), int(float(r["position_ply"])))
        if key not in by_key:
            raise prov.ProtocolViolation(
                f"{path} row {key} has no replay in {B_MANIFEST_PATH}")
        replay = json.loads(Path(by_key[key]).read_text())
        state = position_state(replay, int(float(r["position_ply"])),
                               r["side_to_move"])
        out.add(canonical_state_sha1(state))
    return out


def game_identities(manifest_rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """Cross-reservoir game identity: the SHA-1 of the replay's own bytes.

    `game_idx` is reservoir-local -- every fresh match JSONL numbers games from
    zero -- so comparing raw indices across two independent reservoirs both
    invents overlaps and misses a copied game that was renumbered. Two games
    are the same game iff their replay content is the same.
    """
    out: Dict[str, str] = {}
    for r in manifest_rows:
        path = r["replay_path"]
        sha1 = fpu_provenance.file_sha1(path)
        if sha1 in prov.SENTINEL_HASHES:
            raise prov.ProtocolViolation(
                f"replay {path} is unreadable, so its game identity cannot be "
                f"established")
        out[sha1] = path
    return out


def compute_disjointness(manifest_rows: Sequence[Mapping[str, Any]], *,
                         forbidden: Sequence[str],
                         expected_sha1s: Optional[Mapping[str, str]] = None,
                         forbidden_game_identities:
                             Optional[Mapping[str, Mapping[str, str]]] = None
                         ) -> Dict[str, Any]:
    """The §12 disjointness result, COMPUTED from authenticated corpora.

    Position identity is the canonical state hash; GAME identity is the
    replay-content hash. `expected_sha1s` pins the forbidden bytes so an
    altered file cannot silently drop a collision.
    """
    if not forbidden:
        raise prov.ProtocolViolation(
            "disjointness requires a non-empty forbidden corpus set")
    if expected_sha1s is None:
        raise prov.ProtocolViolation(
            "disjointness requires the authenticated SHA-1 of every forbidden "
            "corpus; reading unverified bytes proves nothing")
    missing = sorted(set(forbidden) - set(expected_sha1s))
    if missing:
        raise prov.ProtocolViolation(
            f"no authenticated SHA-1 for forbidden corpora {missing}")
    verified = authenticate_forbidden_corpora(
        {p: expected_sha1s[p] for p in forbidden})
    mine_sha = {r["canonical_sha1"] for r in manifest_rows}
    mine_games = game_identities(manifest_rows)
    overlaps: List[Dict[str, Any]] = []
    for path in forbidden:
        hit = sorted(mine_sha & forbidden_position_hashes(path))
        if hit:
            overlaps.append({"corpus": path, "canonical_sha1": hit[:5]})
    game_overlaps: List[Dict[str, Any]] = []
    for label, identities in (forbidden_game_identities or {}).items():
        shared = sorted(set(mine_games) & set(identities))
        if shared:
            game_overlaps.append({"corpus": label, "replay_sha1": shared[:5]})
    return {"forbidden_corpora": list(forbidden),
            "forbidden_corpus_sha1s": verified, "overlaps": overlaps,
            "forbidden_game_sources": sorted(forbidden_game_identities or {}),
            "game_overlaps": game_overlaps,
            "game_identity": "replay_content_sha1",
            "checked_positions": len(manifest_rows),
            "checked_games": len(mine_games)}


def _precommitted_selection_sha1(protocol_path: Optional[str]) -> Optional[str]:
    """§8/§9: a later stage's protocol must name the selected-coefficient
    artifact it is bound to, so the binding is fixed before the run rather
    than chosen at run time."""
    if not protocol_path:
        raise prov.ProtocolViolation(
            "a later stage needs its protocol to precommit the "
            "selected-coefficient SHA-1")
    doc = protocol.load_json(protocol_path)
    want = (doc.get("extra") or {}).get("selected_coefficient_sha1")
    if not want:
        raise prov.ProtocolViolation(
            "the protocol does not precommit a selected_coefficient_sha1; a "
            "later stage may not choose its own authorization at run time")
    return want


def _canonical_cell(value: Any) -> str:
    """CSV round-trips every value as text, so compare canonical string forms.
    Numeric cells are compared by value, not by formatting."""
    if value is None:
        return ""
    text = str(value)
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    return repr(int(number)) if number.is_integer() else repr(number)


def rederive_selection(manifest_rows: Sequence[Mapping[str, Any]], *,
                       selector_config: str, screen_csv: str,
                       selector=None) -> Dict[str, Any]:
    """Re-run the ESTABLISHED deterministic selector and compare the COMPLETE
    manifest against its output.

    Eligibility checks alone accept any alternative eligible subset that
    satisfies the geometry -- swapping one genuine row for another kept row
    from the same cell passed them. `select_final_manifest` is the one frozen
    select stage: it hard-matches all eleven screen identities, re-derives the
    config from its pinned (protocol, reservoir) and byte-compares it (catching
    an edited `selection_seed` or floor that no identity hash can see), then
    performs the exact-or-raise `sample_v2_rows` selection at that seed.

    NOTE: the selector binds the source-file identities recorded at screen
    time, INCLUDING `mcts.py`. A corpus selected before the v17 source edits
    therefore cannot be re-derived under the v17 tree -- by design, not by
    defect. Stage 1's corpus is selected after Tasks 2-3, under this source.
    """
    from .fpu_dev_corpus_v2 import load_v2_config, select_final_manifest
    if selector is None:                                    # pragma: no cover
        selector = select_final_manifest
    cfg = load_v2_config(selector_config)
    screen_meta_path = screen_csv + ".meta.json"
    if not Path(screen_meta_path).is_file():
        raise prov.ProtocolViolation(
            f"screen sidecar {screen_meta_path} is missing; the selector cannot "
            f"be re-derived without the screen's recorded identities")
    with open(screen_meta_path) as f:
        screen_meta = json.load(f)
    try:
        # `forbidden_position_hashes`, not the raw loader: gate B's cases carry
        # no `replay_path`, so its authenticated manifest must be joined first.
        forbidden: set = set()
        for corpus in cfg.forbidden_manifests:
            forbidden |= forbidden_position_hashes(corpus)
        rows, info = selector(screen_meta, cfg, forbidden=forbidden,
                              screen_csv_path=screen_csv)
    except Exception as exc:
        raise prov.ProtocolViolation(
            f"the deterministic selector refused to reproduce this selection: "
            f"{type(exc).__name__}: {exc}") from exc

    # Project the selector's rows through the ESTABLISHED `_manifest_row_v2`
    # and compare EVERY column. A key-only comparison authenticated a manifest
    # whose n_legal, root_value_stm, normalized_entropy and split had all been
    # rewritten.
    from .fpu_dev_corpus_v2 import _manifest_row_v2
    derived = [_manifest_row_v2(r) if "canonical_sha1" in r else dict(r)
               for r in rows]
    if len(derived) != len(manifest_rows):
        raise prov.ProtocolViolation(
            f"manifest has {len(manifest_rows)} rows, the deterministic "
            f"selection produced {len(derived)}")
    fields = tuple(derived[0]) if derived else ()
    # EXACT schema: no missing columns and no unproduced extras.
    actual_fields = tuple(manifest_rows[0]) if manifest_rows else ()
    if set(actual_fields) != set(fields):
        raise prov.ProtocolViolation(
            f"manifest schema {sorted(actual_fields)} != the selector-produced "
            f"{sorted(fields)}; missing "
            f"{sorted(set(fields) - set(actual_fields))}, unproduced extras "
            f"{sorted(set(actual_fields) - set(fields))}")
    if actual_fields != fields:
        raise prov.ProtocolViolation(
            f"manifest header order {list(actual_fields)} != the established "
            f"schema order {list(fields)}; the column order is part of the "
            f"producer's output")

    def order(row):
        return (str(row["canonical_position_sha1"]), int(row["game_idx"]),
                int(float(row["position_ply"])))

    # Position identity first, so a differing row SET reports as such rather
    # than as a confusing field mismatch between misaligned rows.
    want_by_key = {order(r): r for r in derived}
    got_by_key = {order(r): r for r in manifest_rows}
    never_selected = sorted(set(got_by_key) - set(want_by_key))
    absent = sorted(set(want_by_key) - set(got_by_key))
    if never_selected or absent:
        raise prov.ProtocolViolation(
            f"manifest does not match the deterministic selection: "
            f"{len(never_selected)} row(s) were never selected "
            f"(e.g. {never_selected[:2]}), {len(absent)} selected row(s) are "
            f"absent; an eligible subset is not the selector's output")
    # EXACT ORDER: the producer writes rows in the order the deterministic
    # selector emits them, so a reordered manifest is not that producer's
    # output even when every row is present.
    for index, (want, got) in enumerate(zip(derived, manifest_rows)):
        if order(want) != order(got):
            raise prov.ProtocolViolation(
                f"manifest row {index} is {order(got)}, but the deterministic "
                f"selection produced {order(want)} at that position; the row "
                f"order is part of the producer's output")
        for field in fields:
            # the manifest is CSV, so compare canonical string forms
            if _canonical_cell(want[field]) != _canonical_cell(got[field]):
                raise prov.ProtocolViolation(
                    f"manifest row {order(got)} field {field!r} is "
                    f"{got[field]!r}, but the deterministic selection produced "
                    f"{want[field]!r}; an eligible subset with edited producer "
                    f"fields is not the selector's output")
    return {"rederived_rows": len(derived), "compared_fields": sorted(fields),
            "order_compared": True, "selector_info_keys": sorted(info)}


def authenticate_qualification(manifest_path: str, *, mode: str,
                               source_index: str, config_path: Optional[str],
                               checkpoint: str,
                               post_screen_report: Optional[str] = None,
                               rederive: bool = True, selector=None
                               ) -> Dict[str, Any]:
    """Authenticate the corpus against the REAL selector artifact chain.

    Every link is required and revalidated against recorded bytes:
      sidecar -> selector config (config_sha1) -> screen CSV (screen_csv_sha1)
      -> post-screen report (status PASS, agreeing config_sha1)
      -> source index (source_index_sha1) -> anchor checkpoint.

    The manifest itself is bound by SCREEN MEMBERSHIP: every selected canonical
    position must appear in the hash-pinned screen. The selector records no
    manifest hash, so a copied or edited manifest beside a genuine sidecar was
    previously indistinguishable; fabricated or altered rows now fail because
    they are not in the authenticated screen.
    """
    try:
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            header = list(reader.fieldnames or [])
            manifest_rows = list(reader)
    except OSError as exc:
        raise prov.ProtocolViolation(
            f"manifest {manifest_path} is not readable: {exc}") from exc
    missing = sorted(set(MANIFEST_REQUIRED_COLUMNS) - set(header))
    if missing:
        raise prov.ProtocolViolation(
            f"{manifest_path} is not a selection manifest (missing {missing}); "
            f"a screen or report cannot stand in for the qualified corpus")

    sidecar_path = manifest_path + ".meta.json"
    if not Path(sidecar_path).is_file():
        raise prov.ProtocolViolation(
            f"selector sidecar {sidecar_path} is missing; a bare manifest is "
            f"not evidence that its rows were qualified")
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    prov_block = sidecar.get("screen_meta_provenance") or {}
    for key in ("source_index_path", "checkpoint", "forbidden_manifests",
                "config_path", "screen_csv", "n_rows"):
        if key not in sidecar:
            raise prov.ProtocolViolation(
                f"selector sidecar {sidecar_path} is missing {key!r}")
    for key in ("config_sha1", "source_index_sha1", "screen_csv_sha1"):
        if not prov_block.get(key):
            raise prov.ProtocolViolation(
                f"selector sidecar {sidecar_path} records no {key}")

    # the sidecar must describe the inputs this run is actually using
    if Path(sidecar["source_index_path"]).name != Path(source_index).name:
        raise prov.ProtocolViolation(
            f"selector sidecar was built against source index "
            f"{sidecar['source_index_path']}, but the run uses {source_index}")
    actual_index_sha1 = fpu_provenance.file_sha1(source_index)
    if prov_block["source_index_sha1"] != actual_index_sha1:
        raise prov.ProtocolViolation(
            f"source index {source_index} has SHA-1 {actual_index_sha1}, but the "
            f"selector qualified {prov_block['source_index_sha1']}")
    ckpt_sha1 = fpu_provenance.file_sha1(checkpoint)
    if fpu_provenance.file_sha1(sidecar["checkpoint"]) != ckpt_sha1:
        raise prov.ProtocolViolation(
            f"selector sidecar qualified checkpoint {sidecar['checkpoint']}, but "
            f"the run searches {checkpoint}")

    # the selector CONFIG must still be the bytes the selector recorded
    selector_config = sidecar["config_path"]
    actual_cfg_sha1 = fpu_provenance.file_sha1(selector_config)
    if actual_cfg_sha1 != prov_block["config_sha1"]:
        raise prov.ProtocolViolation(
            f"selector config {selector_config} has SHA-1 {actual_cfg_sha1}, but "
            f"the selector recorded {prov_block['config_sha1']}")

    # the selector CONFIG must load through the ESTABLISHED loader, which
    # names every required key -- a plausible-looking JSON is not a config
    from .fpu_dev_corpus_v2 import load_v2_config
    try:
        loaded_cfg = load_v2_config(selector_config)
    except ValueError as exc:
        raise prov.ProtocolViolation(
            f"selector config {selector_config} is not a valid v2 config: "
            f"{exc}") from exc
    cfg_run_kind = getattr(loaded_cfg, "run_kind", None)
    if cfg_run_kind is not None and cfg_run_kind != mode:
        raise prov.ProtocolViolation(
            f"selector config run_kind is {cfg_run_kind!r}, not {mode!r}")

    # the SCREEN must still be the bytes the selector selected from, and every
    # selected row must be a KEPT, ANCHOR-ELIGIBLE screen row of the SAME role,
    # side and phase. Membership alone proves nothing: the screen holds every
    # proposal, including ineligible_anchor, ineligible_role and collision
    # rows, so a relabelled target/control pair passed while preserving counts.
    screen_csv = sidecar["screen_csv"]
    actual_screen_sha1 = fpu_provenance.file_sha1(screen_csv)
    if actual_screen_sha1 != prov_block["screen_csv_sha1"]:
        raise prov.ProtocolViolation(
            f"screen {screen_csv} has SHA-1 {actual_screen_sha1}, but the "
            f"selector recorded {prov_block['screen_csv_sha1']}")
    try:
        with open(screen_csv, newline="") as f:
            screen_rows = list(csv.DictReader(f))
    except OSError as exc:
        raise prov.ProtocolViolation(
            f"screen {screen_csv} is not readable: {exc}") from exc
    missing_cols = sorted(set(SCREEN_REQUIRED_COLUMNS) - set(screen_rows[0] if
                                                             screen_rows else {}))
    if missing_cols:
        raise prov.ProtocolViolation(
            f"screen {screen_csv} is missing columns {missing_cols}; it is not "
            f"the selector's screen")
    by_position = {(r["canonical_sha1"], int(r["game_idx"]),
                    int(float(r["ply"]))): r for r in screen_rows}
    for row in manifest_rows:
        key = (row["canonical_position_sha1"], int(row["game_idx"]),
               int(float(row["position_ply"])))
        screened = by_position.get(key)
        if screened is None:
            raise prov.ProtocolViolation(
                f"manifest position {key} is absent from the authenticated "
                f"screen; the manifest was copied, edited, or built from a "
                f"different screen")
        if screened["exclusion_status"] != "kept":
            raise prov.ProtocolViolation(
                f"manifest position {key} has screen exclusion_status "
                f"{screened['exclusion_status']!r}, not 'kept'; it was never "
                f"selectable")
        if screened["anchor_eligible"] != "True":
            raise prov.ProtocolViolation(
                f"manifest position {key} is not anchor-eligible in the screen")
        for manifest_key, screen_key in (("role", "raw_policy_role"),
                                         ("side", "side"),
                                         ("ply_bucket", "ply_bucket")):
            if row[manifest_key] != screened[screen_key]:
                raise prov.ProtocolViolation(
                    f"manifest position {key} claims {manifest_key}="
                    f"{row[manifest_key]!r}, but the authenticated screen "
                    f"records {screened[screen_key]!r}")
    if int(sidecar["n_rows"]) != len(manifest_rows):
        raise prov.ProtocolViolation(
            f"selector sidecar records {sidecar['n_rows']} rows, manifest has "
            f"{len(manifest_rows)}")

    # the artifact chain must be for THIS mode: a complete held-out chain
    # otherwise authenticated as development
    for label, doc in (("sidecar", sidecar),):
        recorded = doc.get("run_kind")
        if recorded is not None and recorded != mode:
            raise prov.ProtocolViolation(
                f"selector {label} run_kind is {recorded!r}, not {mode!r}")

    # the post-screen qualification report is REQUIRED, not optional
    if not post_screen_report or not Path(post_screen_report).is_file():
        raise prov.ProtocolViolation(
            f"scientific mode {mode!r} requires the selector's post-screen "
            f"qualification report; got {post_screen_report!r}")
    with open(post_screen_report) as f:
        report = json.load(f)
    for key in ("status", "config_sha1", "screen_csv_sha1"):
        if key not in report:
            raise prov.ProtocolViolation(
                f"post-screen report {post_screen_report} is missing {key!r}; a "
                f"hand-written PASS document is not the selector's report")
    if report.get("run_kind") is not None and report["run_kind"] != mode:
        raise prov.ProtocolViolation(
            f"post-screen report run_kind is {report['run_kind']!r}, not {mode!r}")
    if report["status"] != "PASS" or report.get("selector_error"):
        raise prov.ProtocolViolation(
            f"post-screen qualification did not PASS: status="
            f"{report['status']!r} error={report.get('selector_error')!r}")
    for key in ("config_sha1", "screen_csv_sha1"):
        if report[key] != prov_block[key]:
            raise prov.ProtocolViolation(
                f"post-screen report and manifest sidecar disagree on {key}: "
                f"{report[key]!r} vs {prov_block[key]!r}")
    # the rows must satisfy THIS mode's frozen geometry, not merely be eligible
    require_corpus_geometry(mode, [
        {**r, "canonical_sha1": r["canonical_position_sha1"]}
        for r in manifest_rows])
    rederivation = None
    if rederive:
        rederivation = rederive_selection(
            manifest_rows, selector_config=selector_config,
            screen_csv=screen_csv, selector=selector)
    return {"sidecar_path": sidecar_path,
            "sidecar_sha1": fpu_provenance.file_sha1(sidecar_path),
            "rederived_selection": rederivation,
            "selector_config": selector_config,
            "selector_config_sha1": actual_cfg_sha1,
            "screen_csv": screen_csv, "screen_csv_sha1": actual_screen_sha1,
            "source_index_sha1": actual_index_sha1,
            "checkpoint_sha1": ckpt_sha1,
            "post_screen_report": post_screen_report,
            "post_screen_report_sha1": fpu_provenance.file_sha1(post_screen_report),
            "selector_forbidden_manifests": sidecar["forbidden_manifests"],
            "screen_meta_provenance": prov_block}


# --- artifacts -------------------------------------------------------------





def _authenticate_task1_freeze(baseline_path: str, moves_path: str) -> None:
    """§9's frozen baseline is immutable evidence; verify its bytes so Stage 4
    cannot be pointed at a rewritten file and rebase itself."""
    for label, path, want in (("baseline", baseline_path, ABCD_BASELINE_SHA1),
                              ("selected-moves", moves_path, ABCD_MOVES_SHA1)):
        actual = fpu_provenance.file_sha1(path)
        if actual != want:
            raise prov.ProtocolViolation(
                f"Task 1 {label} artifact {path} has SHA-1 {actual}, expected "
                f"{want}; the frozen baseline may not be rewritten")


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
        # The replay JSON is what each search actually reads, so it -- not the
        # canonical probe CSV -- is what `replay_data_sha1` must fingerprint.
        source = base[gate]["canonical_source"]
        # Authenticate BEFORE reading: the immutable Task 1 record carries the
        # expected hash, so an altered probe input must never reach the
        # expensive contemporaneous search.
        actual = fpu_provenance.file_sha1(source)
        if actual != base[gate]["source_sha1"]:
            raise prov.ProtocolViolation(
                f"gate {gate} canonical source {source} has SHA-1 {actual}, "
                f"expected the frozen {base[gate]['source_sha1']}")
        with open(source, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("checkpoint") == "0001"]
        if rows and "replay_path" in rows[0]:
            replay_by_case = {r["case_id"]: r["replay_path"] for r in rows}
        else:
            # B's cases CSV carries no replay_path; the goal-line manifest does,
            # joined on (game_idx, position_ply) exactly as Task 1's capture did.
            b_actual = fpu_provenance.file_sha1(B_MANIFEST_PATH)
            if b_actual != B_MANIFEST_SHA1:
                raise prov.ProtocolViolation(
                    f"gate B manifest {B_MANIFEST_PATH} has SHA-1 {b_actual}, "
                    f"expected the frozen {B_MANIFEST_SHA1}; it determines the "
                    f"case-to-replay mapping")
            with open(B_MANIFEST_PATH) as f:
                manifest = json.load(f)["cases"]
            by_key = {(int(c["game_idx"]), int(c["position_ply"])): c["replay_path"]
                      for c in manifest}
            replay_by_case = {
                r["case_id"]: by_key[(int(r["game_idx"]), int(r["position_ply"]))]
                for r in rows}
        out[gate] = [{"gate": gate, "case_id": c["case_id"],
                      "seed": by_id[c["case_id"]]["seed"],
                      "game_idx": by_id[c["case_id"]]["game_idx"],
                      "position_ply": by_id[c["case_id"]]["position_ply"],
                      "side_to_move": by_id[c["case_id"]]["side_to_move"],
                      "cases_source": source,
                      "replay_path": replay_by_case[c["case_id"]]}
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


def run_abcd_stage(*, searcher=None, **kwargs):
    """Scientific Stage-4 entry point: builds and EMITS. Injection refused."""
    if searcher is not None:
        raise prov.ProtocolViolation(
            "abcd is scientific; an injected searcher is refused because an "
            "emitted artifact must come from the real one")
    artifact = _build_abcd_stage(**kwargs)
    protocol.emit(kwargs["out_path"], artifact)
    return artifact


def _build_abcd_stage(*, coefficient: float, checkpoint: str, out_path: str,
                   protocol_path: str, config_path: str,
                   baseline_path: str = ABCD_BASELINE_PATH,
                   moves_path: str = ABCD_MOVES_PATH,
                   selected_coefficient: Optional[str] = None,
                   searcher=None) -> Dict[str, Any]:
    """Stage 4. Consumes the four canonical manifests, re-runs shipped and the
    frozen candidate, validates the shipped run against the Task 1 freeze, and
    applies all four §9 verdicts."""
    _require_positive_grid_coefficient(coefficient, "A/B/C/D")
    # zero-GPU authorization first
    abcd_selection = authenticate_selected_coefficient(
        selected_coefficient, coefficient=coefficient,
        expected_sha1=_precommitted_selection_sha1(protocol_path))
    prov.validate_batching(prov.BATCHING)
    prov.verify_frozen_design()
    prov.require_clean_worktree("abcd")
    prov.validate_output_path(out_path)
    verified = protocol.load_verified(protocol_path, config_path,
                                      consumer_run_kind="abcd")
    bind_protocol_to_runtime(verified, coefficient=coefficient,
                             checkpoint=checkpoint)
    for label, path in (("checkpoint", checkpoint), ("baseline", baseline_path),
                        ("moves", moves_path)):
        if not Path(path).is_file():
            raise prov.ProtocolViolation(
                f"{label} input is not readable at protocol time: {path}")
    _authenticate_task1_freeze(baseline_path, moves_path)
    cases = load_abcd_cases(baseline_path=baseline_path, moves_path=moves_path)
    probe_sources = sorted({r["cases_source"] for g in ABCD_GATES for r in cases[g]})
    replay_paths = sorted({r["replay_path"] for g in ABCD_GATES for r in cases[g]})
    for label, paths in (("canonical probe source", probe_sources),
                         ("replay", replay_paths)):
        for path in paths:
            if not Path(path).is_file():
                raise prov.ProtocolViolation(
                    f"{label} is not readable at protocol time: {path}")
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
        case_rows=_abcd_case_rows(cases, shipped, candidate,
                                  baseline_path=baseline_path),
        checkpoints={"anchor": checkpoint},
        effective_mcts_config=effective_configs((SHIPPED, coefficient)),
        manifest=baseline_path, source_index=moves_path,
        replay_paths=replay_paths,
        source_files=[str(module_dir / n) for n in RESULT_DETERMINING_MODULES],
        extra_identities={"canonical_probe_sources":
                          {p: fpu_provenance.file_sha1(p) for p in probe_sources},
                          "b_manifest": {B_MANIFEST_PATH: B_MANIFEST_SHA1},
                          "selected_coefficient": abcd_selection,
                          "task1_baseline_sha1": ABCD_BASELINE_SHA1,
                          "task1_moves_sha1": ABCD_MOVES_SHA1,
                          "config_sha1": fpu_provenance.file_sha1(config_path)},
        protocol_sha1=protocol.protocol_sha1(protocol.load_json(protocol_path)))
    return artifact


def _abcd_case_rows(cases: Mapping[str, Sequence[Mapping]],
                    shipped: Mapping[str, Sequence[Mapping]],
                    candidate: Mapping[str, Sequence[Mapping]], *,
                    baseline_path: str) -> List[dict]:
    """The paired per-case evidence Task 13 requires. Without it the 216
    searches vanish behind four aggregate verdicts."""
    with open(baseline_path) as f:
        frozen = json.load(f)["abcd_frozen_baseline"]
    out: List[dict] = []
    for gate in ABCD_GATES:
        want = {c["case_id"]: float(c["probe_black_root_value_repr"])
                for c in frozen[gate]["cases"]}
        s_by = {c["case_id"]: c for c in shipped[gate]}
        c_by = {c["case_id"]: c for c in candidate[gate]}
        if set(s_by) != set(want) or set(c_by) != set(want):
            raise prov.ProtocolViolation(
                f"gate {gate}: shipped/candidate case ids differ from the frozen set")
        for case in cases[gate]:
            cid = case["case_id"]
            s, c = s_by[cid], c_by[cid]
            out.append({
                "gate": gate, "case_id": cid, "seed": case["seed"],
                "replay_path": case["replay_path"],
                "shipped_value": float(s["black_value"]),
                "candidate_value": float(c["black_value"]),
                "shipped_move": list(s["selected_move"]),
                "candidate_move": list(c["selected_move"]),
                "shipped_replies": int(s["replies"]),
                "candidate_replies": int(c["replies"]),
                "shipped_top_share": float(s["top_share"]),
                "candidate_top_share": float(c["top_share"]),
                "shipped_collapse": bool(s["collapse"]),
                "candidate_collapse": bool(c["collapse"]),
                "value_delta": float(c["black_value"]) - float(s["black_value"]),
                "move_changed": list(s["selected_move"]) != list(c["selected_move"]),
                "frozen_value": want[cid],
                "abs_delta_vs_frozen": abs(float(s["black_value"]) - want[cid]),
            })
    return out


def _real_abcd_searcher(checkpoint: str, *, evaluator=None):
    """Real 400-sim MCTS over the canonical probe cases, reusing the same
    harness path the Task 1 capture used.

    `evaluator` is injectable so this function is EXECUTABLE in a test with a
    CPU stub. Inspecting its source text does not establish that it runs.
    """
    import dataclasses
    from .diagnose_fpu_policy_mass import _position_features, _search_position
    from .mcts import decode_move
    from .position_probe_cases import position_state
    if evaluator is None:                                   # pragma: no cover
        from .eval_runner import _default_evaluator_factory
        evaluator = _default_evaluator_factory(checkpoint)
    base_cfg = base_mcts_config()
    prov.validate_batching(base_cfg)

    def search(case, coefficient):
        cfg = dataclasses.replace(
            base_cfg, fpu_shipped_policy_mass_reduction=coefficient)
        prov.validate_batching(cfg)
        replay = json.loads(Path(case["replay_path"]).read_text())
        state = position_state(replay, int(case["position_ply"]),
                               case["side_to_move"])
        # `_position_features` is the FROZEN §7.0 definition set: `replies` is
        # the visited-child count at the final root LEADER's reply node, not a
        # count of visited root alternatives. Reusing it is what keeps the A
        # mechanism gate measuring the quantity §9 names.
        search_out, obs = _search_position(evaluator, cfg, state, case["seed"])
        feats = _position_features(search_out, obs)
        top = feats["top_move"]
        if top is None:
            raise MissingTelemetry(
                f"case {case['case_id']!r} produced no visit leader")
        # `_position_features` returns `top.move`, an ENCODED int. The frozen
        # Task 1 moves are [row, col], so decode with the canonical helper.
        row_, col_ = decode_move(int(top))
        return {"case_id": case["case_id"],
                "black_value": (feats["root_value_stm"] if state.to_move == "black"
                                else -feats["root_value_stm"]),
                "selected_move": [int(row_), int(col_)],
                "replies": int(feats["replies"]),
                "top_share": float(feats["top_share"]),
                "collapse": bool(feats["collapsed"])}
    return search


def run_diagnostic(*, mode: str, selector=None, searcher=None, **kwargs):
    """The SCIENTIFIC entry point: builds and EMITS the artifact.

    Injected `selector`/`searcher` implementations are refused for scientific
    run kinds. They exist so the pipeline can be driven without a GPU, but a
    stub could otherwise fabricate rows or bypass exact selection and still
    emit a protocol-valid scientific artifact. Injection therefore lives
    strictly BELOW this boundary, in `_build_diagnostic`, which emits nothing.
    """
    if prov.is_scientific(mode) and (selector is not None or searcher is not None):
        raise prov.ProtocolViolation(
            f"run_kind {mode!r} is scientific; injected selector/searcher "
            f"implementations are refused because an emitted artifact must come "
            f"from the real ones. Use _build_diagnostic for non-emitting runs.")
    built = _build_diagnostic(mode=mode, selector=selector,
                              searcher=searcher, **kwargs)
    artifact = built["artifact"] if isinstance(built, dict) and \
        "artifact" in built else built
    out_path = kwargs["out_path"]
    protocol.emit(out_path, artifact)
    pending = (built or {}).get("pending_selection") if isinstance(built, dict) \
        else None
    if pending is not None:
        # Emitted only here, and only AFTER the development artifact it binds,
        # so its recorded SHA-1 is the artifact actually on disk.
        selection = build_selected_coefficient(
            coefficient=artifact["coefficient"], table=pending["table"],
            development_artifact=str(out_path),
            development_artifact_sha1=fpu_provenance.file_sha1(str(out_path)),
            development=artifact)
        protocol.emit(str(Path(out_path).with_name("selected_coefficient.json")),
                      selection)
    return artifact


def _build_diagnostic(*, mode: str, manifest_path: str, checkpoint: str,
                   out_path: str, seed_base: int,
                   frozen_coefficient: Optional[float] = None,
                   source_index: Optional[str] = None,
                   protocol_path: Optional[str] = None,
                   config_path: Optional[str] = None,
                   qualification_report: Optional[str] = None,
                   stage1_manifest: Optional[str] = None,
                   stage1_source_index: Optional[str] = None,
                   stage1_post_screen_report: Optional[str] = None,
                   selected_coefficient: Optional[str] = None,
                   selector=None, searcher=None) -> Dict[str, Any]:
    """Search, gate and emit. Every precondition is checked in `preflight`
    before an evaluator is loaded."""
    if mode == "abcd":
        # Stage 4 consumes the four canonical probe manifests, not a dev
        # corpus, so it has its own entry point rather than a degenerate pass
        # through this one.
        if not (protocol_path and config_path):
            raise prov.ProtocolViolation(
                "abcd must run against a verified protocol/config pair")
        # same shape as the other modes, so no caller special-cases abcd
        return {"artifact": _build_abcd_stage(
            coefficient=frozen_coefficient, checkpoint=checkpoint,
            out_path=out_path, protocol_path=protocol_path,
            config_path=config_path, searcher=searcher,
            selected_coefficient=selected_coefficient),
            "pending_selection": None}
    pre = preflight(mode=mode, manifest_path=manifest_path, checkpoint=checkpoint,
                    out_path=out_path, source_index=source_index,
                    frozen_coefficient=frozen_coefficient,
                    protocol_path=protocol_path, config_path=config_path,
                    qualification_report=qualification_report,
                    qualification_selected_coefficient=selected_coefficient,
                    stage1_manifest=stage1_manifest,
                    stage1_source_index=stage1_source_index,
                    stage1_post_screen_report=stage1_post_screen_report,
                    selector=selector)
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
    # already authorized in preflight, before any search
    selection_binding = pre["selection_binding"]
    pending_selection: Optional[Dict[str, Any]] = None
    if mode == "development":
        selected, table = select_smallest_passing(rows, shipped_lockin=shipped_lockin)
        gates = {str(k): gate_record(v) for k, v in table.items()}
        pending_selection = {"table": table}
    elif mode == "held_out":
        v = heldout_verdict(rows, frozen_coefficient, shipped_lockin=shipped_lockin)
        gates = {"held_out": gate_record(v)}
    artifact = build_artifact(
        mode=mode, coefficient=selected, rows=rows, gates=gates,
        checkpoints={ANCHOR_ROLE: checkpoint},
        extra_identities={
            "config_sha1": (fpu_provenance.file_sha1(config_path)
                            if config_path else None),
            "source_index_sha1": (fpu_provenance.file_sha1(source_index)
                                  if source_index else None),
            "qualification_report_sha1": (
                fpu_provenance.file_sha1(qualification_report)
                if qualification_report else None),
            "selector_qualification": pre["qualification"],
            "selected_coefficient": selection_binding,
            "disjointness": pre["disjointness"],
        },
        effective_mcts_config=effective_configs(configs),
        manifest=manifest_path, source_index=source_index,
        replay_paths=pre["replay_paths"], source_files=pre["source_files"],
        protocol_sha1=(protocol.protocol_sha1(protocol.load_json(protocol_path))
                       if protocol_path else None))
    # The builder EMITS NOTHING. `run_diagnostic` is the only writer, and it
    # refuses injected implementations -- so no stub can produce an artifact
    # that authorizes a later stage.
    return {"artifact": artifact, "pending_selection": pending_selection}


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
                         ply_bucket=manifest_row["ply_bucket"],
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
    ap.add_argument("--qualification-report", default=None)
    ap.add_argument("--stage1-manifest", default=None)
    ap.add_argument("--stage1-source-index", default=None)
    ap.add_argument("--stage1-post-screen-report", default=None)
    ap.add_argument("--selected-coefficient", default=None)
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
            protocol_path=args.protocol, config_path=args.config,
            qualification_report=args.qualification_report,
            stage1_manifest=args.stage1_manifest,
            stage1_source_index=args.stage1_source_index,
            stage1_post_screen_report=args.stage1_post_screen_report,
            selected_coefficient=args.selected_coefficient)
    except prov.ProtocolViolation as exc:
        print(f"PROTOCOL VIOLATION: {exc}")
        return protocol.EXIT_USAGE
    print(json.dumps({"mode": artifact["mode"],
                      "coefficient": artifact["coefficient"],
                      "n_rows": artifact["n_rows"]}, sort_keys=True))
    return protocol.EXIT_OK


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
