"""Atlas Stage 2 -- corpus geometry, all on synthetic metadata."""
import itertools

import pytest

from scripts.GPU.alphazero.corpus_geometry import (
    MAX_SEED_RANGE_GAMES,
    PHASES,
    PILOT_GAMES,
    SIDES,
    GameMeta,
    assign_corpus,
    eligible_cells,
    eligible_plies,
    final_demands,
    match_games_to_cells,
    phase_for_ply,
    pilot_geometry_gate,
    select_ply,
    side_for_ply,
    size_continuation,
    stable_key,
)


# -- Amendment 5: phase is the quarter of the game's REALIZED trajectory -----

@pytest.mark.parametrize("ply,expected", [
    (0, "opening"), (9, "opening"),
    (10, "early_mid"), (19, "early_mid"),
    (20, "midgame"), (29, "midgame"),
    (30, "late"), (39, "late"),
])
def test_quarters_of_a_40_move_game(ply, expected):
    """(4p)//40 == p//10, so the quarters are exactly 10 plies each."""
    assert phase_for_ply(ply, 40) == expected


def test_the_SAME_ply_lands_in_different_phases_in_different_games():
    """The whole point of the amendment: phase is trajectory-relative, so an
    absolute ply carries no phase on its own."""
    assert phase_for_ply(30, 40) == "late"
    assert phase_for_ply(30, 200) == "opening"   # (4*30)//200 == 0


def test_the_final_ply_is_always_late_and_the_first_always_opening():
    for n in (8, 39, 57, 76, 280):
        assert phase_for_ply(0, n) == "opening"
        assert phase_for_ply(n - 1, n) == "late"


def test_a_ply_at_or_past_the_end_is_REJECTED_not_classified():
    """Amendment 5's domain is 0 <= ply < n_moves.

    Unguarded, `min(3, ...)` would label these `late` -- so a row whose ply and
    n_moves had come apart, which is corrupt metadata, would enter the corpus
    wearing a plausible phase. There is no valid position at ply == n_moves:
    the game is over.
    """
    for bad in (40, 41, 99):
        with pytest.raises(ValueError, match="ply"):
            phase_for_ply(bad, 40)


def test_the_domain_restriction_changes_no_valid_classification():
    """Over the whole domain the raw index never exceeds 3, so `min(3, ...)`
    never binds for a real position. Rejecting outside the domain therefore
    cannot alter any answer the formula gives inside it."""
    assert max((4 * p) // n for n in range(1, 300) for p in range(n)) == 3


def test_n_moves_is_REQUIRED_not_defaulted():
    """A default would let a stale call site silently keep the old behaviour,
    which is exactly what this amendment must not permit."""
    with pytest.raises(TypeError):
        phase_for_ply(5)


def test_guards_are_kept():
    with pytest.raises(ValueError, match="non-negative"):
        phase_for_ply(-1, 40)
    with pytest.raises(ValueError, match="n_moves"):
        phase_for_ply(0, 0)


def test_every_game_from_8_to_280_moves_serves_ALL_EIGHT_cells():
    """The geometry failure this amendment exists to fix, swept EXHAUSTIVELY
    over the whole producible range rather than sampled at four points.

    280 is the frozen `max_moves`, so 8..280 is every length generation can
    emit; below 8 a quarter can hold one ply and therefore one side. A sampled
    test would miss a parity or rounding hole at some specific length, which is
    precisely the class of defect that produced the no-go.
    """
    for n in range(8, 281):
        meta = GameMeta(game_id=0, seed=1, n_moves=n, start_player="red")
        assert len(eligible_cells(meta)) == 8, f"n_moves={n} serves fewer cells"


def test_below_eight_moves_a_quarter_can_hold_a_single_side():
    """Stated so the >= 8 bound is a measured boundary, not an assumption."""
    short = GameMeta(game_id=0, seed=1, n_moves=4, start_player="red")
    assert len(eligible_cells(short)) < 8


def test_the_retired_pilots_lengths_would_now_all_serve_late():
    """The 24 retired games ran 39-76 plies and produced ZERO late capacity
    under absolute bounds. Retained as a regression fixture ONLY -- these are
    lengths, not positions, and no retired position enters any corpus."""
    for n in (39, 46, 48, 55, 57, 62, 70, 76):
        meta = GameMeta(game_id=0, seed=1, n_moves=n, start_player="red")
        assert eligible_plies(meta, "late", "red")
        assert eligible_plies(meta, "late", "black")


def test_quarters_partition_the_whole_trajectory():
    """No ply is unassigned and none is double-assigned, over every producible
    length."""
    for n in range(8, 281):
        meta = GameMeta(game_id=0, seed=1, n_moves=n, start_player="red")
        seen = [p for ph in PHASES for s in SIDES
                for p in eligible_plies(meta, ph, s)]
        assert sorted(seen) == list(range(n))


def test_side_alternates_from_the_start_player():
    assert side_for_ply(0, "red") == "red"
    assert side_for_ply(1, "red") == "black"
    assert side_for_ply(0, "black") == "black"
    assert side_for_ply(91, "red") == "black"      # odd ply


def test_eligibility_derives_only_from_n_moves_and_start_player():
    """No value, residual, entropy, branching or outcome is consulted."""
    g = GameMeta(game_id=1, seed=100, n_moves=40, start_player="red")
    # Quarters of a 40-move game are 10 plies each; red is on even plies.
    assert eligible_plies(g, "opening", "red") == [0, 2, 4, 6, 8]
    # Amendment 5: a 40-move game DOES have late positions -- its own final
    # quarter. Under the old absolute bounds this was [] "never reaches 91".
    assert eligible_plies(g, "late", "red") == [30, 32, 34, 36, 38]
    cells = eligible_cells(g)
    assert ("opening", "red") in cells and ("early_mid", "black") in cells
    assert ("late", "red") in cells


def test_a_long_game_serves_every_cell():
    long_game = GameMeta(game_id=2, seed=101, n_moves=200, start_player="red")
    assert eligible_cells(long_game) == {(p, s) for p in PHASES for s in SIDES}


def test_stable_key_is_deterministic_across_processes():
    """Python hash() is process-randomized for str; a rerun would silently
    produce a different corpus. The key must be a digest."""
    a = stable_key(20260804, 7, "discovery", "late", "black", 95)
    b = stable_key(20260804, 7, "discovery", "late", "black", 95)
    assert a == b and len(a) == 40
    assert a != stable_key(20260804, 7, "discovery", "late", "black", 97)


# ---------------------------------------------------------------- matching --

def _games(specs):
    return [GameMeta(game_id=i, seed=1000 + i, n_moves=n, start_player=sp)
            for i, (n, sp) in enumerate(specs)]


def test_full_matching_succeeds_and_respects_one_position_per_game():
    games = _games([(200, "red")] * 4)
    demands = {("discovery", "late", "red"): 2, ("discovery", "late", "black"): 2}
    r = match_games_to_cells(games, demands, sampling_seed=1)
    assert r.achieved_flow == r.demanded_flow == 4
    assert len(r.assignment) == 4
    assert not r.unmet


def test_shortfall_reports_unmet_cells_and_a_min_cut():
    # Amendment 5 changed how a cell becomes unfillable. A short game no longer
    # lacks `late` -- it has its own final quarter -- so the negative is now
    # built from a quarter that cannot supply the SIDE: in a 4-move red-start
    # game each quarter is a single ply, and the last one is black to move.
    games = _games([(4, "red")] * 4)
    demands = {("discovery", "late", "red"): 2}
    r = match_games_to_cells(games, demands, sampling_seed=1)
    assert r.achieved_flow == 0 and r.demanded_flow == 2
    assert r.unmet == {("discovery", "late", "red"): 2}
    assert ("discovery", "late", "red") in r.min_cut_cells


def test_shared_capacity_is_caught_not_double_counted():
    """One long game can serve either late side but supplies only ONE row."""
    games = _games([(200, "red")])
    demands = {("discovery", "late", "red"): 1, ("discovery", "late", "black"): 1}
    r = match_games_to_cells(games, demands, sampling_seed=1)
    assert r.achieved_flow == 1 and r.demanded_flow == 2
    assert sum(r.unmet.values()) == 1


def test_matching_is_deterministic_under_the_sampling_seed():
    games = _games([(200, "red")] * 6)
    demands = {("discovery", "late", "red"): 3}
    a = match_games_to_cells(games, demands, sampling_seed=42).assignment
    b = match_games_to_cells(games, demands, sampling_seed=42).assignment
    assert a == b


# -------------------------------------------------------------- pilot gate --

def _pilot(n_moves=200):
    return _games([(n_moves, "red" if i % 2 == 0 else "black")
                   for i in range(PILOT_GAMES)])


def test_pilot_gate_passes_when_24_long_games_cover_every_cell():
    r = pilot_geometry_gate(_pilot(), sampling_seed=7)
    assert r["verdict"] == "PASS"
    assert len(r["assignment"]) == PILOT_GAMES


def test_pilot_gate_no_gos_when_a_quarter_cannot_supply_both_sides():
    """Amendment 5 dissolved the old no-go fixture: 60-move games now fill every
    cell. A genuine no-go needs quarters of a single ply, all on one side --
    24 four-move red-start games cover only 4 of the 8 cells."""
    r = pilot_geometry_gate(_games([(4, "red")] * PILOT_GAMES), sampling_seed=7)
    assert r["verdict"] == "PHASE_GEOMETRY_NO_GO"
    assert ("discovery", "late", "red") in r["min_cut_cells"]


def test_pilot_gate_rejects_a_wrong_sized_pilot():
    with pytest.raises(ValueError):
        pilot_geometry_gate(_games([(200, "red")] * 23), sampling_seed=7)


# ------------------------------------------------------------------ sizing --

def test_subset_sweep_covers_exactly_254_proper_nonempty_subsets():
    cells = [(p, s) for p in PHASES for s in SIDES]
    assert len(cells) == 8
    subsets = [c for r in range(1, 8) for c in itertools.combinations(cells, r)]
    assert len(subsets) == 2 ** 8 - 2 == 254


def test_sizing_never_exceeds_480_when_the_pilot_gate_passed():
    """q_S >= k/8 follows from the gate (3 distinct games per cell), so
    D_S/q_S <= N-24 and G_total <= 480 at the maximum N."""
    for n in (200, 240, 320, 400):
        r = size_continuation(_pilot(), n_target=n)
        assert r["verdict"] == "OK"
        assert r["G_total"] <= MAX_SEED_RANGE_GAMES
        assert r["G_total"] % 40 == 0
        assert r["g_cont"] >= n - PILOT_GAMES


def test_sizing_reports_the_binding_subset():
    r = size_continuation(_pilot(), n_target=240)
    assert r["binding_subset"] is not None
    assert 1 <= len(r["binding_subset"]) <= 7


def test_sizing_rejects_a_disallowed_n():
    with pytest.raises(ValueError):
        size_continuation(_pilot(), n_target=250)


def test_zero_capacity_subset_is_a_no_go():
    r = size_continuation(_games([(4, "red")] * PILOT_GAMES), n_target=200)
    assert r["verdict"] == "PHASE_GEOMETRY_NO_GO"


# -------------------------------------------------------------- assignment --

@pytest.mark.parametrize("n,disc,val", [
    (200, 12, 10), (240, 15, 12), (280, 18, 14),
    (320, 21, 16), (360, 24, 18), (400, 27, 20),
])
def test_final_demands_are_integral_and_sum_to_d_c(n, disc, val):
    d = final_demands(n)
    assert d[("discovery", "opening", "red")] == disc
    assert d[("validation", "opening", "red")] == val
    assert disc + val == n // 8 - 3
    assert sum(d.values()) == n - 24


def test_select_ply_is_stable_and_within_the_cell():
    meta = GameMeta(game_id=5, seed=9, n_moves=200, start_player="red")
    a = select_ply(meta, "discovery", "late", "black", sampling_seed=3)
    assert a == select_ply(meta, "discovery", "late", "black", sampling_seed=3)
    assert phase_for_ply(a, meta.n_moves) == "late"
    assert side_for_ply(a, "red") == "black"


def _continuation(n_games, n_moves=200):
    return [GameMeta(game_id=PILOT_GAMES + i, seed=2000 + i, n_moves=n_moves,
                     start_player="red" if i % 2 == 0 else "black")
            for i in range(n_games)]


def test_assign_corpus_emits_a_witness_on_success():
    pa = pilot_geometry_gate(_pilot(), sampling_seed=7)["assignment"]
    r = assign_corpus(pa, _continuation(216), n_target=200, sampling_seed=7)
    assert r["verdict"] == "OK"
    assert len(r["rows"]) == 200 - PILOT_GAMES
    assert len({row["game_id"] for row in r["rows"]}) == len(r["rows"])


def test_assign_corpus_stops_with_a_failure_artifact_on_shortfall():
    pa = pilot_geometry_gate(_pilot(), sampling_seed=7)["assignment"]
    # 50-move games now serve every cell, so the shortfall is built the same way
    # as the gate's: single-ply quarters, all red-start, covering 4 cells of 8.
    starved = [GameMeta(game_id=PILOT_GAMES + i, seed=2000 + i, n_moves=4,
                        start_player="red") for i in range(216)]
    r = assign_corpus(pa, starved, n_target=200, sampling_seed=7)
    assert r["verdict"] == "ASSIGNMENT_SHORTFALL"
    assert r["achieved_flow"] < r["demanded_flow"]
    assert r["unmet"] and r["min_cut_cells"]
    assert "rows" not in r          # no partial corpus is emitted
