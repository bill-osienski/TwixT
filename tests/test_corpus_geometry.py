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


@pytest.mark.parametrize("ply,expected", [
    (0, "opening"), (30, "opening"), (31, "early_mid"), (60, "early_mid"),
    (61, "midgame"), (90, "midgame"), (91, "late"), (279, "late"),
])
def test_phase_bounds_are_exact(ply, expected):
    assert phase_for_ply(ply) == expected


def test_side_alternates_from_the_start_player():
    assert side_for_ply(0, "red") == "red"
    assert side_for_ply(1, "red") == "black"
    assert side_for_ply(0, "black") == "black"
    assert side_for_ply(91, "red") == "black"      # odd ply


def test_eligibility_derives_only_from_n_moves_and_start_player():
    """No value, residual, entropy, branching or outcome is consulted."""
    short = GameMeta(game_id=1, seed=100, n_moves=40, start_player="red")
    assert eligible_plies(short, "opening", "red") == list(range(0, 31, 2))
    assert eligible_plies(short, "late", "red") == []        # never reaches ply 91
    cells = eligible_cells(short)
    assert ("opening", "red") in cells and ("early_mid", "black") in cells
    assert not any(p == "late" for p, _s in cells)


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
    games = _games([(40, "red")] * 4)
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


def test_pilot_gate_no_gos_when_late_cells_cannot_be_filled():
    r = pilot_geometry_gate(_pilot(n_moves=60), sampling_seed=7)
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
    r = size_continuation(_pilot(n_moves=60), n_target=200)
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
    assert phase_for_ply(a) == "late" and side_for_ply(a, "red") == "black"


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
    r = assign_corpus(pa, _continuation(216, n_moves=50), n_target=200,
                      sampling_seed=7)
    assert r["verdict"] == "ASSIGNMENT_SHORTFALL"
    assert r["achieved_flow"] < r["demanded_flow"]
    assert r["unmet"] and r["min_cut_cells"]
    assert "rows" not in r          # no partial corpus is emitted
