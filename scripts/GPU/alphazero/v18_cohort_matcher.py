"""v18 precommitted cohort matcher -- plan Task 4b.

Picks the final non-A control cohort AFTER Task 7's measurement, using only the
fields the frozen matcher is permitted to see. Every threshold and tolerance is
imported from the Task 5 preregistration UNCHANGED: this module holds logic, and
not one number of its own.

COLUMNS ARE GAMES, NOT POSITIONS. The cost matrix is
`n_A x distinct_control_games`; cell `(a, g)` carries game `g`'s BEST admissible
position for A row `a`. Hungarian assigns each column at most once, so
"at most one control position per game" is enforced STRUCTURALLY.

Reducing to the cheapest position per game per A row and then solving a
position-column matrix does NOT enforce it: two A rows can still take two
different positions from one game, because those are distinct columns. The
invariant has to live in the matrix shape, not in a pre-filter.

Greedy nearest-neighbour is forbidden -- it can fail on A-row ordering when a
complete matching exists, turning a solvable problem into a spurious
PREFLIGHT_FAIL. `greedy_match_for_comparison` exists ONLY so the test suite can
demonstrate that failure; it never selects a cohort.

This module runs no search, touches no evaluator and reads no residual field.

DETERMINISM CONTRACT -- frozen, and deliberately NOT a global lexicographic
minimum:

    A-row order                 (canonical_state_sha1, game_content_sha1,
                                 position_ply)
    game-column order           game_content_sha1 ascending
    within-game position choice (cost, canonical_state_sha1,
                                 game_content_sha1, position_ply)
    equal-cost assignments      resolved by deterministic Hungarian traversal
                                under those frozen row/column orders

The scientific requirement is a deterministic, order-independent, residual-blind
minimum-cost matching. It is NOT a globally lexicographically minimal optimum:
when two equal-cost assignments differ, the winner follows the column order
above rather than the within-game tuple, and no stronger claim is made. Ties are
between equally admissible controls, so the choice carries no scientific
content -- only its reproducibility does.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Sequence, Tuple

from .fpu_dev_reservoir_protocol import canonical_json_bytes
from .v18_preflight_criteria import MATCHING

ALGORITHM_VERSION = "v18.4b.1"

# Exactly the fields the matcher may consult. `_project` drops everything else
# BEFORE any cost is computed, so a residual-derived column cannot influence the
# cohort even by accident.
PERMITTED_ROW_FIELDS = frozenset({
    "canonical_state_sha1", "game_content_sha1", "game_idx", "position_ply",
    "phase", "side_to_move", "root_value_stm", "n_legal",
    "eligible_depth2_leaves",
})

# Anything the cohort will later CALIBRATE. Selecting on these would fit the
# threshold to its own sample (spec Sec 2.2.3).
FORBIDDEN_ROW_FIELDS = frozenset({
    "exposure_primary_0.50", "exposure_descriptive_count",
    "exposure_descriptive_clipped_mass", "positive_mass", "negative_mass",
    "sign_dominance", "would_clip_1.25", "would_clip_1.0", "would_clip_0.75",
    "would_clip_0.5", "clipped_amount_1.25", "clipped_amount_1.0",
    "clipped_amount_0.75", "clipped_amount_0.5",
})

# (matching variable, row field, whether magnitude is taken)
_NUMERIC_VARIABLES = (
    ("abs_root_value_stm", "root_value_stm", True),
    ("n_legal", "n_legal", False),
    ("eligible_depth2_leaves", "eligible_depth2_leaves", False),
)
_EXACT_VARIABLES = (("phase", "phase"), ("side_to_move", "side_to_move"))


def _project(row: Dict) -> Dict:
    """The only view of a row the matcher ever sees."""
    return {k: v for k, v in row.items() if k in PERMITTED_ROW_FIELDS}


def _value(row: Dict, field: str, magnitude: bool) -> float:
    return abs(row[field]) if magnitude else row[field]


def pair_cost(a_row: Dict, control_row: Dict, matching: Dict = MATCHING) -> float:
    """Sum of per-variable absolute differences, each divided by its OWN
    tolerance so every term lands in [0, 1]. A pair outside any tolerance, or
    disagreeing on an exact variable, costs `inf` and can never be matched."""
    a, c = _project(a_row), _project(control_row)
    tolerances = matching["tolerances"]
    for variable, field in _EXACT_VARIABLES:
        if tolerances[variable] != "exact":
            raise ValueError(f"{variable} is no longer an exact tolerance")
        if a[field] != c[field]:
            return math.inf
    total = 0.0
    for variable, field, magnitude in _NUMERIC_VARIABLES:
        tolerance = tolerances[variable]
        difference = abs(_value(a, field, magnitude) - _value(c, field, magnitude))
        if difference > tolerance:
            return math.inf
        total += difference / tolerance
    return total


def _tie_key(cost: float, control: Dict) -> Tuple:
    """The frozen WITHIN-GAME order, read from the imported contract rather than
    restated: `MATCHING["determinism"]["within_game_position_order"]`."""
    order = MATCHING["determinism"]["within_game_position_order"]
    if order != ("cost", "canonical_state_sha1", "game_content_sha1",
                 "position_ply"):
        raise ValueError(f"unrecognised within_game_position_order {order!r}")
    return (cost, control["canonical_state_sha1"], control["game_content_sha1"],
            control["position_ply"])


def _identity(row: Dict) -> Tuple:
    """The tuple that identifies one control position."""
    return (row["canonical_state_sha1"], row["game_content_sha1"],
            row["position_ply"])


def game_columns(census_rows: Sequence[Dict]) -> List[str]:
    """Distinct control GAMES, ascending by replay content SHA-1.

    Game identity is the replay's content hash -- never `game_idx`, which is
    reservoir-local and would both invent overlaps and miss a renumbered copy.
    The fixed ordering is what makes the assignment deterministic.
    """
    order = MATCHING["determinism"]["game_column_order"]
    if order != ("game_content_sha1",):
        raise ValueError(f"unrecognised game_column_order {order!r}")
    return sorted({row["game_content_sha1"] for row in census_rows})


def _cost_matrix(census_rows: Sequence[Dict], a_list: Sequence[Dict],
                 matching: Dict) -> Tuple[List[List[float]], List[str],
                                          List[List[Dict]]]:
    columns = game_columns(census_rows)
    index = {sha: i for i, sha in enumerate(columns)}
    by_game: List[List[Dict]] = [[] for _ in columns]
    for row in census_rows:
        by_game[index[row["game_content_sha1"]]].append(row)

    cost = [[math.inf] * len(columns) for _ in a_list]
    chosen: List[List[Dict]] = [[None] * len(columns) for _ in a_list]
    for i, a_row in enumerate(a_list):
        for j, candidates in enumerate(by_game):
            best = None
            for control in candidates:
                c = pair_cost(a_row, control, matching)
                if math.isinf(c):
                    continue
                key = _tie_key(c, control)
                if best is None or key < best[0]:
                    best = (key, c, control)
            if best is not None:
                cost[i][j] = best[1]
                chosen[i][j] = best[2]
    return cost, columns, chosen


def hungarian_rectangular(cost: Sequence[Sequence[float]]) -> List[Tuple[int, int]]:
    """Minimum-cost assignment of every row to a distinct column.

    Vendored: `scipy` is not installed in this venv, and adding it for one call
    is not justified. Shortest-augmenting-path with potentials, O(n^2 m), which
    is trivial on a 30 x 800 matrix. `inf` cells are genuinely unreachable --
    they are never relaxed, so an inadmissible pair can never be assigned; a row
    with no reachable column raises rather than returning a partial matching.
    """
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    if m < n:
        raise ValueError(f"cannot assign {n} rows to {m} columns")

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    parent = [0] * (m + 1)          # parent[j] = 1-indexed row assigned to col j
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        parent[0] = i
        j0 = 0
        minv = [math.inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = parent[j0]
            delta = math.inf
            j1 = -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1]
                if cur != math.inf:
                    cur = cur - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if delta == math.inf or j1 == -1:
                raise ValueError(
                    f"row {i - 1} has no admissible column: no complete matching "
                    f"exists")
            for j in range(m + 1):
                if used[j]:
                    u[parent[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if parent[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            parent[j0] = parent[j1]
            j0 = j1

    return sorted((parent[j] - 1, j - 1) for j in range(1, m + 1) if parent[j])


def greedy_match_for_comparison(census_rows: Sequence[Dict],
                                a_list: Sequence[Dict],
                                matching: Dict = MATCHING) -> List[Dict]:
    """FORBIDDEN as a selection rule; present only so the suite can show it
    failing where minimum-cost succeeds. Never call this to pick a cohort."""
    taken, out = set(), []
    for a_row in a_list:
        best = None
        for control in census_rows:
            if control["game_content_sha1"] in taken:
                continue
            c = pair_cost(a_row, control, matching)
            if math.isinf(c):
                continue
            key = _tie_key(c, control)
            if best is None or key < best[0]:
                best = (key, control)
        if best is not None:
            taken.add(best[1]["game_content_sha1"])
            out.append(best[1])
    return out


def _canonical_order(rows: Sequence[Dict]) -> List[Dict]:
    """The frozen A-row order, from `MATCHING["determinism"]["a_row_order"]`."""
    order = MATCHING["determinism"]["a_row_order"]
    return sorted(rows, key=lambda r: tuple(r[field] for field in order))


def _required(matching: Dict) -> int:
    """The frozen cardinality, and the ONLY source of it.

    There is deliberately no caller override. An override would let a caller
    accept a cohort of 2 or 29 while the report still stamped 30/30, and the
    approved AUC operating characteristics were computed at exactly
    n_A = n_C = 30 -- a smaller cohort silently invalidates them.
    """
    required = matching["cardinality"]["n_a"]
    if required != matching["cardinality"]["n_c"]:
        raise ValueError("the frozen cardinality is not 1:1")
    return required


def match_report_on_failure(census_rows, a_list, matching=MATCHING) -> Dict:
    """Which A rows could not be matched, and why -- reported rather than
    dropped, so a failure names its cause instead of shrinking the cohort."""
    a_sorted = _canonical_order(a_list)
    cost, columns, _chosen = _cost_matrix(census_rows, a_sorted, matching)
    unmatched = []
    for i, a_row in enumerate(a_sorted):
        if all(math.isinf(c) for c in cost[i]):
            unmatched.append({
                "canonical_state_sha1": a_row["canonical_state_sha1"],
                "phase": a_row["phase"],
                "side_to_move": a_row["side_to_move"],
                "reason": "no_admissible_control",
            })
    return {
        "requested": _required(matching),
        "a_rows": len(a_sorted),
        "distinct_control_games": len(columns),
        "unmatched_a_rows": unmatched,
    }


def match_cohort(census_rows: Sequence[Dict], a_list: Sequence[Dict],
                 matching: Dict = MATCHING) -> Tuple[List[Dict], Dict]:
    """The precommitted 1:1 cohort selection.

    Returns `(cohort, report)` or raises. A complete matching of every A row, or
    a refusal -- never a smaller cohort. The cardinality comes from the frozen
    contract and cannot be overridden.
    """
    required = _required(matching)
    a_sorted = _canonical_order(a_list)
    if len(a_sorted) != required:
        raise ValueError(
            f"expected exactly {required} A rows, got {len(a_sorted)}")

    cost, columns, chosen = _cost_matrix(census_rows, a_sorted, matching)
    if len(columns) < required:
        raise ValueError(
            f"only {len(columns)} distinct control games for {required} A rows: "
            f"at most one control position per game, so distinct games is the "
            f"binding supply. PREFLIGHT_FAIL, never a smaller cohort")

    failure = match_report_on_failure(census_rows, a_list, matching)
    if failure["unmatched_a_rows"]:
        raise ValueError(
            f"{len(failure['unmatched_a_rows'])} A rows have no admissible "
            f"control (unmatched): {failure['unmatched_a_rows']}. "
            f"PREFLIGHT_FAIL, never a cohort of {required - 1}")

    assignment = hungarian_rectangular(cost)
    if len(assignment) != required:
        raise ValueError(
            f"matched {len(assignment)} of {required} A rows; a complete "
            f"matching or a failure, never a partial cohort")

    pairs, cohort = [], []
    for i, j in assignment:
        control = chosen[i][j]
        if control is None or math.isinf(cost[i][j]):
            raise ValueError("an inadmissible pair reached the assignment")
        cohort.append(control)
        pairs.append({
            "a_canonical_state_sha1": a_sorted[i]["canonical_state_sha1"],
            "control_canonical_state_sha1": control["canonical_state_sha1"],
            "control_game_content_sha1": control["game_content_sha1"],
            "control_position_ply": control["position_ply"],
            "cost": cost[i][j],
        })

    games = {c["game_content_sha1"] for c in cohort}
    if len(games) != len(cohort):
        raise ValueError("two controls came from one game")

    report = {
        "matched": len(cohort),
        "cardinality": matching["cardinality"],
        "algorithm": matching["algorithm"],
        "algorithm_version": ALGORITHM_VERSION,
        "columns_are_games": True,
        "distinct_control_games": len(columns),
        "candidate_control_positions": len(census_rows),
        "greedy_nearest_neighbour_forbidden": True,
        "determinism": matching["determinism"],
        "pairs": pairs,
        "total_cost": sum(p["cost"] for p in pairs),
        "per_variable_balance": _balance(a_sorted, assignment, chosen, matching),
        "unmatched_a_rows": [],
    }
    return cohort, report


def _balance(a_sorted, assignment, chosen, matching) -> Dict:
    balance: Dict[str, Dict] = {}
    for variable, field, magnitude in _NUMERIC_VARIABLES:
        differences = [
            abs(_value(a_sorted[i], field, magnitude)
                - _value(chosen[i][j], field, magnitude))
            for i, j in assignment]
        balance[variable] = {
            "tolerance": matching["tolerances"][variable],
            "mean_abs_difference": (sum(differences) / len(differences)
                                    if differences else 0.0),
            "max_abs_difference": max(differences) if differences else 0.0,
        }
    for variable, field in _EXACT_VARIABLES:
        mismatches = sum(1 for i, j in assignment
                         if a_sorted[i][field] != chosen[i][j][field])
        balance[variable] = {"exact": True, "mismatches": mismatches}
    return balance


def emit_matched_cohort(out_path: str, *, cohort: Sequence[Dict], report: Dict,
                        universe_sha1: str, census_sha1: str,
                        criteria_sha1: str, a_source_sha1: str) -> str:
    """Write the authenticated cohort artifact canonically; return its SHA-1.

    Tasks 8 and 9 authenticate this file by SHA-1 before reading it, so the
    matched rows they consume are provably the ones this matcher produced.
    """
    bindings = {"universe_sha1": universe_sha1, "census_sha1": census_sha1,
                "criteria_sha1": criteria_sha1, "a_source_sha1": a_source_sha1}
    for name, value in bindings.items():
        if not isinstance(value, str) or len(value) != 40:
            raise ValueError(f"{name} must be a 40-character sha1, got {value!r}")

    # EVERY count must reconcile at the frozen cardinality before a byte is
    # written. The payload stamps MATCHING's 30/30 unconditionally, so without
    # this an under-filled cohort would be published as a complete one.
    required = _required(MATCHING)
    games = {r["game_content_sha1"] for r in cohort}
    checks = {
        "cohort rows": len(cohort),
        "distinct control games": len(games),
        "report pairs": len(report.get("pairs", [])),
        "report matched": report.get("matched"),
    }
    for label, value in checks.items():
        if value != required:
            raise ValueError(
                f"refusing to emit: {label} is {value}, frozen cardinality is "
                f"{required}. A cohort that does not reconcile at {required} "
                f"may not be published as one")
    if report.get("cardinality") != MATCHING["cardinality"]:
        raise ValueError(
            f"refusing to emit: report cardinality {report.get('cardinality')} "
            f"!= frozen {MATCHING['cardinality']}")
    if report.get("unmatched_a_rows"):
        raise ValueError(
            f"refusing to emit: {len(report['unmatched_a_rows'])} unmatched A "
            f"rows. A complete matching or a failure, never a partial cohort")

    # Counts reconciling is not the same as the pairs DESCRIBING these rows: a
    # 30-row cohort can be handed a different 30-entry report and still satisfy
    # every count. Compare identities positionally.
    for position, (control, pair) in enumerate(zip(cohort, report["pairs"])):
        recorded = (pair.get("control_canonical_state_sha1"),
                    pair.get("control_game_content_sha1"),
                    pair.get("control_position_ply"))
        if _identity(control) != recorded:
            raise ValueError(
                f"refusing to emit: report pair {position} describes "
                f"{recorded}, but cohort row {position} is "
                f"{_identity(control)}. The pairs must describe the rows they "
                f"are published with")

    payload = {
        "run_kind": "v18_matched_control_cohort",
        "scientific_interpretation_forbidden": True,
        "algorithm": MATCHING["algorithm"],
        "algorithm_version": ALGORITHM_VERSION,
        "cardinality": MATCHING["cardinality"],
        "greedy_nearest_neighbour_forbidden": True,
        "determinism": MATCHING["determinism"],
        "tolerances": MATCHING["tolerances"],
        "permitted_row_fields": sorted(PERMITTED_ROW_FIELDS),
        "forbidden_row_fields": sorted(FORBIDDEN_ROW_FIELDS),
        "matched_cohort": [_project(row) for row in cohort],
        "matching_report": report,
        **bindings,
    }
    raw = canonical_json_bytes(payload)
    with open(out_path, "wb") as handle:
        handle.write(raw)
    return hashlib.sha1(raw).hexdigest()
