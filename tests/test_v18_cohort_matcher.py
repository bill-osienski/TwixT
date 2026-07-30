"""v18 precommitted cohort matcher -- plan Task 4b.

Selects the non-A control cohort AFTER measurement, using only fields the frozen
matcher is permitted to see. No thresholds live here: they are imported from the
Task 5 preregistration unchanged.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path

import pytest

from scripts.GPU.alphazero import v18_cohort_matcher as M
from scripts.GPU.alphazero import v18_preflight_criteria as C


def row(idx, *, phase="early_mid", side="black", stm=0.05, n_legal=480,
        leaves=90, game=None, ply=None, **extra):
    game_sha = game if game is not None else f"{idx:040x}"
    return dict(
        canonical_state_sha1=f"{idx:040d}",
        game_content_sha1=game_sha,
        game_idx=idx,
        position_ply=ply if ply is not None else 40 + idx,
        phase=phase,
        side_to_move=side,
        root_value_stm=stm,
        n_legal=n_legal,
        eligible_depth2_leaves=leaves,
        **extra,
    )


N = C.MATCHING["cardinality"]["n_a"]           # 30, frozen; never overridden


def a_rows(n=N, **kw):
    return [row(i, **kw) for i in range(n)]


def census(n, offset=1000, **kw):
    return [row(offset + i, **kw) for i in range(n)]


# --- padding to the frozen 30/30 --------------------------------------------
# The matcher has no cardinality override, so every end-to-end fixture must be a
# real 30-row problem. Filler pairs are spread far outside every tolerance, so
# filler i is admissible ONLY to control i and cannot perturb the special rows
# under test. Genuinely small matrices stay in the low-level Hungarian tests.

def _filler_a(i):
    return row(500 + i, n_legal=1000 + 200 * i)


def _filler_c(i):
    return row(9000 + i, game=f"f{i:039x}", n_legal=1000 + 200 * i)


def padded(a_special=(), c_special=()):
    """`(a_list, census_rows)` at exactly N A rows, with fillers appended."""
    fill = N - len(a_special)
    assert fill >= 0, "too many special A rows"
    a_list = list(a_special) + [_filler_a(i) for i in range(fill)]
    controls = list(c_special) + [_filler_c(i) for i in range(fill)]
    return a_list, controls


def test_filler_pairs_cannot_perturb_the_rows_under_test():
    """The padding is only sound if each filler matches exactly one control."""
    a_list, controls = padded()
    for i, a in enumerate(a_list):
        admissible = [j for j, c in enumerate(controls)
                      if not math.isinf(M.pair_cost(a, c, C.MATCHING))]
        assert admissible == [i], (i, admissible)


# --- the permitted-field boundary -------------------------------------------

def test_matcher_reads_no_residual_field():
    """Residual-derived fields carry values that WOULD change the outcome if
    consulted. The cohort must be identical with and without them."""
    clean_a, clean_c = padded()
    cohort_clean, _ = M.match_cohort(clean_c, clean_a, C.MATCHING)

    poisoned_c = []
    for i, r in enumerate(clean_c):
        r = dict(r)
        r.update({
            "exposure_primary_0.50": 99.0 - i,
            "would_clip_1.25": 99 - i,
            "would_clip_0.5": 99 - i,
            "clipped_amount_1.25": 99.0 - i,
            "positive_mass": 99.0 - i,
            "sign_dominance": 0.99,
        })
        poisoned_c.append(r)
    cohort_poisoned, _ = M.match_cohort(poisoned_c, clean_a, C.MATCHING)
    assert ([c["canonical_state_sha1"] for c in cohort_clean]
            == [c["canonical_state_sha1"] for c in cohort_poisoned])

    # Structural: the projection the matcher works on cannot even carry them.
    for forbidden in ("exposure_primary_0.50", "would_clip_1.25", "would_clip_0.5",
                      "clipped_amount_1.25", "positive_mass", "sign_dominance"):
        assert forbidden not in M.PERMITTED_ROW_FIELDS
        assert forbidden not in M._project(poisoned_c[0])
    assert M.PERMITTED_ROW_FIELDS.isdisjoint(M.FORBIDDEN_ROW_FIELDS)


def test_matcher_respects_every_frozen_tolerance():
    tol = C.MATCHING["tolerances"]
    base = row(0)
    assert math.isinf(M.pair_cost(base, row(1, phase="late"), C.MATCHING))
    assert math.isinf(M.pair_cost(base, row(1, side="red"), C.MATCHING))
    assert math.isinf(M.pair_cost(
        base, row(1, stm=base["root_value_stm"] + tol["abs_root_value_stm"] + 1e-9),
        C.MATCHING))
    assert math.isinf(M.pair_cost(
        base, row(1, n_legal=base["n_legal"] + tol["n_legal"] + 1), C.MATCHING))
    assert math.isinf(M.pair_cost(
        base, row(1, leaves=base["eligible_depth2_leaves"] + tol["eligible_depth2_leaves"] + 1),
        C.MATCHING))
    # Exactly on a tolerance is admissible; each term is normalised to [0, 1].
    edge = M.pair_cost(base, row(1, n_legal=base["n_legal"] + tol["n_legal"]),
                       C.MATCHING)
    assert edge == pytest.approx(1.0)
    assert M.pair_cost(base, row(1), C.MATCHING) == 0.0
    # Magnitude, not sign: the matcher pairs on abs(root_value_stm).
    assert M.pair_cost(row(0, stm=0.2), row(1, stm=-0.2), C.MATCHING) == 0.0


def test_matcher_enforces_at_most_one_position_per_game():
    shared = "f" * 40
    a_list, controls = padded(
        a_special=[row(0), row(1)],
        c_special=[row(1000, game=shared, ply=40), row(1001, game=shared, ply=52)])
    # Two positions, but ONE game: distinct games is the binding supply, so the
    # 30 A rows face only 29 columns.
    with pytest.raises(ValueError, match="distinct control games"):
        M.match_cohort(controls, a_list, C.MATCHING)


def test_two_a_rows_preferring_different_positions_from_one_game():
    """The revision-4 bug. Two A rows whose best admissible controls are two
    DIFFERENT plies of the SAME game. A position-column matrix assigns both and
    silently breaks one-per-game; a game-column matrix structurally cannot."""
    shared = "a" * 40
    controls = [
        row(1000, game=shared, ply=40, n_legal=480),   # A0's best (cost 0)
        row(1001, game=shared, ply=52, n_legal=470),   # A1's best (cost 0)
        row(1002, game="b" * 40, ply=44, n_legal=470),  # admissible fallback
    ]
    # Both A rows prefer a position in `shared`; only one may take it.
    a_list, all_controls = padded(
        a_special=[row(0, n_legal=480), row(1, n_legal=470)], c_special=controls)
    cohort, report = M.match_cohort(all_controls, a_list, C.MATCHING)
    assert len(cohort) == N
    assert len({c["game_content_sha1"] for c in cohort}) == N, (
        "two controls came from the same game")
    assert report["columns_are_games"] is True
    assert report["distinct_control_games"] == N
    special = [c for c in cohort if c["game_content_sha1"] in (shared, "b" * 40)]
    assert len(special) == 2
    assert len({c["game_content_sha1"] for c in special}) == 2


def test_game_identity_is_the_replay_content_sha1_not_game_idx():
    # Same game content, different reservoir-local indices: still ONE game.
    shared = "c" * 40
    controls = [row(1000, game=shared, ply=40),
                dict(row(2000, game=shared, ply=52), game_idx=7)]
    columns = M.game_columns(controls)
    assert len(columns) == 1
    # Different content, same index: TWO games.
    controls = [row(1000, game="d" * 40, ply=40),
                dict(row(1001, game="e" * 40, ply=40), game_idx=1000)]
    assert len(M.game_columns(controls)) == 2


def test_matcher_is_deterministic_and_order_independent():
    a, c = padded()
    first, _ = M.match_cohort(c, a, C.MATCHING)
    again, _ = M.match_cohort(c, a, C.MATCHING)
    shuffled, _ = M.match_cohort(list(reversed(c)), list(reversed(a)), C.MATCHING)
    key = lambda rows: [r["canonical_state_sha1"] for r in rows]
    assert key(first) == key(again) == key(shuffled)


def test_matcher_finds_a_complete_matching_that_greedy_would_miss():
    """A0's nearest control is the ONLY admissible control for A1. Greedy takes
    it for A0, then fails on A1 and returns 29-of-30; min-cost returns both."""
    a = [row(0, n_legal=400), row(1, n_legal=380)]
    controls = [
        row(1000, game="1" * 40, n_legal=381),   # admissible to BOTH, nearest A0
        row(1001, game="2" * 40, n_legal=440),   # |440-380| = 60 > 50 -> A0 only
    ]
    assert math.isinf(M.pair_cost(a[1], controls[1], C.MATCHING))
    a_list, all_controls = padded(a_special=a, c_special=controls)
    greedy = M.greedy_match_for_comparison(all_controls, a_list, C.MATCHING)
    assert len(greedy) == N - 1, "the greedy baseline must actually fail here"
    cohort, _ = M.match_cohort(all_controls, a_list, C.MATCHING)
    assert len(cohort) == N


def test_matcher_refuses_unless_all_30_are_matched():
    assert C.MATCHING["cardinality"] == {"n_a": 30, "n_c": 30}
    a_list, controls = padded()
    with pytest.raises(ValueError, match="30"):
        M.match_cohort(controls[:-1], a_list, C.MATCHING)     # one short
    cohort, report = M.match_cohort(controls, a_list, C.MATCHING)
    assert len(cohort) == 30
    assert report["matched"] == 30
    assert report["cardinality"] == {"n_a": 30, "n_c": 30}
    # 29 A rows is equally a refusal: the count is frozen at BOTH ends.
    with pytest.raises(ValueError, match="exactly 30"):
        M.match_cohort(controls, a_list[:-1], C.MATCHING)


def test_matcher_reports_unmatched_a_rows_rather_than_dropping_them():
    late = row(0, phase="late")                 # no late control exists
    a_list, controls = padded(a_special=[late],
                              c_special=[row(1000, game="c" * 40, n_legal=999)])
    with pytest.raises(ValueError) as excinfo:
        M.match_cohort(controls, a_list, C.MATCHING)
    assert "unmatched" in str(excinfo.value).lower()
    unmatched = M.match_report_on_failure(
        controls, a_list, C.MATCHING)["unmatched_a_rows"]
    assert [u["canonical_state_sha1"] for u in unmatched] == [
        late["canonical_state_sha1"]]
    assert unmatched[0]["reason"] == "no_admissible_control"


def test_matching_report_records_per_variable_balance_and_per_pair_cost():
    a_special = [row(i, n_legal=480) for i in range(3)]
    c_special = [row(1000 + i, game=f"{i:040x}", n_legal=470) for i in range(3)]
    a_list, controls = padded(a_special=a_special, c_special=c_special)
    cohort, report = M.match_cohort(controls, a_list, C.MATCHING)
    assert len(report["pairs"]) == N
    for pair in report["pairs"]:
        assert set(pair) >= {"a_canonical_state_sha1", "control_canonical_state_sha1",
                             "control_game_content_sha1", "control_position_ply",
                             "cost"}
        assert pair["cost"] >= 0.0
    balance = report["per_variable_balance"]
    for var in ("abs_root_value_stm", "n_legal", "eligible_depth2_leaves"):
        assert set(balance[var]) >= {"mean_abs_difference", "max_abs_difference",
                                     "tolerance"}
        assert balance[var]["max_abs_difference"] <= balance[var]["tolerance"]
    assert balance["n_legal"]["max_abs_difference"] == 10
    for var in ("phase", "side_to_move"):
        assert balance[var]["exact"] is True
        assert balance[var]["mismatches"] == 0
    assert report["total_cost"] == pytest.approx(
        sum(p["cost"] for p in report["pairs"]))


def test_determinism_contract_is_structured_and_imported():
    """The single ambiguous tuple could not distinguish within-game selection
    from equal-cost assignment resolution, so the artifact over-claimed."""
    assert "tie_breaking" not in C.MATCHING
    d = C.MATCHING["determinism"]
    assert d["a_row_order"] == ("canonical_state_sha1", "game_content_sha1",
                                "position_ply")
    assert d["game_column_order"] == ("game_content_sha1",)
    assert d["within_game_position_order"] == (
        "cost", "canonical_state_sha1", "game_content_sha1", "position_ply")
    assert d["global_lexicographic_minimum"] is False
    # The matcher and its report carry the IMPORTED block, not a copy.
    a_list, controls = padded()
    _cohort, report = M.match_cohort(controls, a_list, C.MATCHING)
    assert report["determinism"] is d


def test_within_game_tie_break_follows_the_frozen_tuple():
    """WITHIN a game column, the cheapest position is chosen by
    `determinism["within_game_position_order"]`."""
    assert C.MATCHING["determinism"]["within_game_position_order"] == (
        "cost", "canonical_state_sha1", "game_content_sha1", "position_ply")
    shared = "7" * 40
    a_special = [row(0, n_legal=400)]
    # Same cost, same game; the lower canonical hash must win regardless of the
    # order the census supplies them in.
    early = dict(row(1000, game=shared, ply=44, n_legal=410),
                 canonical_state_sha1="1" * 40)
    late = dict(row(1001, game=shared, ply=88, n_legal=410),
                canonical_state_sha1="9" * 40)
    assert (M.pair_cost(a_special[0], early, C.MATCHING)
            == M.pair_cost(a_special[0], late, C.MATCHING))
    for order in ([early, late], [late, early]):
        a_list, controls = padded(a_special=a_special, c_special=order)
        cohort, _ = M.match_cohort(controls, a_list, C.MATCHING)
        chosen = [c for c in cohort if c["game_content_sha1"] == shared]
        assert [c["canonical_state_sha1"] for c in chosen] == ["1" * 40]


def test_assignment_is_deterministic_under_multiple_optima():
    """ACROSS game columns the result is deterministic -- identical for any
    input order -- because the columns are sorted by game content SHA-1.

    KNOWN LIMITATION, reported rather than papered over: that column order, not
    the frozen tuple, is what resolves an across-game tie, and the tuple ranks
    `canonical_state_sha1` FIRST. When the two orderings disagree the chosen
    control is the one with the lower GAME hash. This test asserts only what the
    implementation actually guarantees.
    """
    a_special = [row(0, n_legal=400)]
    lower_canonical = dict(row(1000, game="z" * 40, ply=40, n_legal=410),
                           canonical_state_sha1="1" * 40)
    lower_game = dict(row(1001, game="0" * 40, ply=40, n_legal=410),
                      canonical_state_sha1="5" * 40)
    assert (M.pair_cost(a_special[0], lower_canonical, C.MATCHING)
            == M.pair_cost(a_special[0], lower_game, C.MATCHING))
    results = set()
    for order in ([lower_canonical, lower_game], [lower_game, lower_canonical]):
        a_list, controls = padded(a_special=a_special, c_special=order)
        cohort, _ = M.match_cohort(controls, a_list, C.MATCHING)
        picked = [c for c in cohort
                  if c["canonical_state_sha1"] in {"1" * 40, "5" * 40}]
        assert len(picked) == 1
        results.add(picked[0]["canonical_state_sha1"])
    assert len(results) == 1, "the assignment must be order-independent"
    assert results == {"5" * 40}, "resolved by game column order, not the tuple"


# --- the vendored solver -----------------------------------------------------

def test_hungarian_rectangular_matches_a_known_optimum():
    cost = [[4.0, 1.0, 3.0],
            [2.0, 0.0, 5.0],
            [3.0, 2.0, 2.0]]
    assignment = M.hungarian_rectangular(cost)
    assert sorted(assignment) == [(0, 1), (1, 0), (2, 2)]
    assert sum(cost[r][c] for r, c in assignment) == 5.0
    # Rectangular: 2 rows, 4 columns.
    wide = [[1.0, 9.0, 9.0, 2.0],
            [9.0, 3.0, 9.0, 9.0]]
    assert sorted(M.hungarian_rectangular(wide)) == [(0, 0), (1, 1)]


def test_infinite_cost_pairs_can_never_be_matched():
    inf = math.inf
    cost = [[inf, 1.0],
            [2.0, inf]]
    assert sorted(M.hungarian_rectangular(cost)) == [(0, 1), (1, 0)]
    # A row with no finite cell cannot be assigned at all.
    with pytest.raises(ValueError, match="no admissible"):
        M.hungarian_rectangular([[inf, inf], [1.0, 2.0]])


def _brute_force(cost):
    n, m = len(cost), len(cost[0])
    best, best_assign = math.inf, None
    for cols in itertools.permutations(range(m), n):
        total = sum(cost[r][c] for r, c in enumerate(cols))
        if total < best:
            best, best_assign = total, list(enumerate(cols))
    return best, best_assign


def test_vendored_hungarian_agrees_with_brute_force_on_small_matrices():
    """Correctness evidence for hand-rolling the algorithm rather than importing
    one: exhaustive agreement for n <= 6, including inf cells and ties."""
    import random
    rng = random.Random(20260730)
    checked = 0
    for n in range(1, 5):
        for m in range(n, min(n + 2, 6) + 1):
            for trial in range(12):
                cost = [[float(rng.randint(0, 6)) for _ in range(m)]
                        for _ in range(n)]
                if trial % 3 == 0:              # sprinkle inf cells
                    for _ in range(rng.randint(1, n)):
                        cost[rng.randrange(n)][rng.randrange(m)] = math.inf
                if trial % 4 == 0:              # force ties
                    cost = [[1.0] * m for _ in range(n)]
                expected, _ = _brute_force(cost)
                if math.isinf(expected):
                    with pytest.raises(ValueError):
                        M.hungarian_rectangular(cost)
                    continue
                assignment = M.hungarian_rectangular(cost)
                assert len(assignment) == n
                assert len({c for _r, c in assignment}) == n
                got = sum(cost[r][c] for r, c in assignment)
                assert got == pytest.approx(expected), (n, m, cost)
                checked += 1
    assert checked > 40


# --- the emitted artifact ----------------------------------------------------

BINDINGS = dict(universe_sha1="1" * 40, census_sha1="2" * 40,
                criteria_sha1="3" * 40, a_source_sha1="4" * 40)


def _emit(tmp_path, name="cohort.json"):
    a_list, controls = padded()
    cohort, report = M.match_cohort(controls, a_list, C.MATCHING)
    path = tmp_path / name
    sha = M.emit_matched_cohort(str(path), cohort=cohort, report=report, **BINDINGS)
    return path, sha, json.loads(path.read_text())


def test_emitted_cohort_artifact_is_byte_reproducible(tmp_path):
    p1, s1, _ = _emit(tmp_path, "a.json")
    p2, s2, _ = _emit(tmp_path, "b.json")
    assert p1.read_bytes() == p2.read_bytes()
    assert s1 == s2 == hashlib.sha1(p1.read_bytes()).hexdigest()
    assert b"timestamp" not in p1.read_bytes()


def test_emitted_artifact_binds_universe_census_criteria_and_a_source_sha1s(tmp_path):
    _p, _s, payload = _emit(tmp_path)
    for key, value in BINDINGS.items():
        assert payload[key] == value, key
    with pytest.raises(ValueError, match="sha1"):
        M.emit_matched_cohort(str(tmp_path / "bad.json"), cohort=[], report={},
                              universe_sha1="short", census_sha1="2" * 40,
                              criteria_sha1="3" * 40, a_source_sha1="4" * 40)


def test_emitted_artifact_records_algorithm_version_cardinality_and_run_kind(tmp_path):
    _p, _s, payload = _emit(tmp_path)
    assert payload["algorithm"] == "rectangular_hungarian_minimum_cost"
    assert payload["algorithm"] == C.MATCHING["algorithm"]
    assert payload["algorithm_version"] == M.ALGORITHM_VERSION
    assert payload["cardinality"] == C.MATCHING["cardinality"]
    assert payload["run_kind"] == "v18_matched_control_cohort"
    assert payload["scientific_interpretation_forbidden"] is True
    assert payload["greedy_nearest_neighbour_forbidden"] is True


def test_cardinality_cannot_be_overridden_by_a_caller():
    """A cardinality override would let a caller accept a cohort of 2 or 29
    while the report still stamped 30/30, silently invalidating the approved
    AUC operating characteristics, which were computed at exactly 30."""
    import inspect
    for fn in (M.match_cohort, M.match_report_on_failure):
        assert "cardinality" not in inspect.signature(fn).parameters, fn.__name__
    a_list, controls = padded()
    with pytest.raises(TypeError):
        M.match_cohort(controls, a_list, C.MATCHING, cardinality=2)
    assert M._required(C.MATCHING) == 30


def test_emitter_refuses_a_short_cohort_and_writes_nothing(tmp_path):
    a_list, controls = padded()
    cohort, report = M.match_cohort(controls, a_list, C.MATCHING)

    def attempt(path, **overrides):
        c = overrides.pop("cohort", cohort)
        r = dict(report)
        r.update(overrides)
        out = tmp_path / path
        with pytest.raises(ValueError, match="refusing to emit"):
            M.emit_matched_cohort(str(out), cohort=c, report=r, **BINDINGS)
        assert not out.exists(), f"{path}: a refusal must leave no file"

    attempt("short_rows.json", cohort=cohort[:-1])          # 29 rows
    attempt("short_pairs.json", pairs=report["pairs"][:-1])  # 29 pairs
    attempt("bad_matched.json", matched=29)
    attempt("bad_cardinality.json", cardinality={"n_a": 29, "n_c": 29})
    attempt("unmatched.json",
            unmatched_a_rows=[{"canonical_state_sha1": "0" * 40}])
    # Duplicate game: 30 rows but only 29 distinct games.
    duped = list(cohort[:-1]) + [dict(cohort[-1],
                                      game_content_sha1=cohort[0]["game_content_sha1"])]
    attempt("duped_games.json", cohort=duped)

    # IDENTITY, not just counts: a 30-row cohort handed a different 30-entry
    # report satisfies every count while the pairs describe other rows.
    swapped = [dict(p) for p in report["pairs"]]
    swapped[3]["control_canonical_state_sha1"] = "d" * 40
    attempt("swapped_identity.json", pairs=swapped)

    fabricated = [dict(p) for p in report["pairs"]]
    fabricated[7]["control_position_ply"] = fabricated[7]["control_position_ply"] + 1
    attempt("fabricated_ply.json", pairs=fabricated)

    reordered = list(report["pairs"])
    reordered[0], reordered[1] = reordered[1], reordered[0]
    attempt("reordered_pairs.json", pairs=reordered)
    # The honest cohort still emits.
    good = tmp_path / "good.json"
    assert len(M.emit_matched_cohort(str(good), cohort=cohort, report=report,
                                     **BINDINGS)) == 40
    assert good.exists()


def test_matching_contract_is_imported_not_restated():
    assert M.MATCHING is C.MATCHING
    assert "side" not in M.MATCHING["tolerances"]
    assert set(M.MATCHING["variables"]) <= (
        set(C.CENSUS_SCHEMA) | set(M.MATCHING["derived_variables"]))
