"""Telemetry contract tests: perspective, undefined values, schema version,
plus Task B4's search/readout separation and live capture."""
import random

import pytest

from scripts.GPU.alphazero import eval_readout as R
from scripts.GPU.alphazero.eval_replay import REPLAY_SCHEMA_VERSION, ply_record
from scripts.GPU.alphazero.eval_runner import (
    EvalConfig, play_eval_game, readout_from_eval_config, root_child_stats,
)
from scripts.GPU.alphazero.game.twixt_state import TwixtState
from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
from tests.eval_fakes import FakeEvaluator

SMALL = EvalConfig(board_size=8, mcts_sims=32, mcts_eval_batch_size=4,
                   mcts_stall_flush_sims=4, opening_temp_plies=2,
                   temp_high=1.0, temp_low=0.1, max_moves=16)


def _top2():
    return [
        R.ChildStat((2, 2), 190, 0.30, -0.30),
        R.ChildStat((1, 1), 40, -0.05, 0.05),
    ]


def test_schema_version_is_bumped_for_top2():
    assert REPLAY_SCHEMA_VERSION == 2


def test_ply_record_without_top2_keeps_the_field_null_not_empty():
    rec = ply_record(0, "red", (2, 2), {(2, 2): 5, (1, 1): 3}, 0.1)
    assert rec["top2"] is None
    assert rec["readout_overrode_leader"] is False


def test_ply_record_emits_both_perspectives():
    rec = ply_record(21, "red", (2, 2), {(2, 2): 190, (1, 1): 40}, 0.1,
                     top2=_top2())
    a, b = rec["top2"]
    assert a["completed_visit_count"] == 190
    assert a["q_value_child_perspective"] == pytest.approx(0.30)
    assert a["q_value_root_perspective"] == pytest.approx(-0.30)
    assert b["q_value_root_perspective"] == pytest.approx(0.05)


def test_ply_record_top2_carries_the_move_coordinates():
    rec = ply_record(21, "red", (2, 2), {(2, 2): 190, (1, 1): 40}, 0.1,
                     top2=_top2())
    assert (rec["top2"][0]["row"], rec["top2"][0]["col"]) == (2, 2)
    assert (rec["top2"][1]["row"], rec["top2"][1]["col"]) == (1, 1)


def test_ply_record_preserves_undefined_q_as_null():
    top2 = [R.ChildStat((2, 2), 190, 0.3, -0.3), R.ChildStat((1, 1), 0, None, None)]
    rec = ply_record(21, "red", (2, 2), {(2, 2): 190, (1, 1): 0}, 0.1, top2=top2)
    assert rec["top2"][1]["q_value_child_perspective"] is None
    assert rec["top2"][1]["q_value_root_perspective"] is None


def test_ply_record_records_the_override_flag():
    rec = ply_record(21, "red", (1, 1), {(2, 2): 190, (1, 1): 40}, 0.1,
                     top2=_top2(), overrode_leader=True)
    assert rec["readout_overrode_leader"] is True


def test_ply_record_still_fails_loud_on_a_move_outside_the_counts():
    with pytest.raises(ValueError):
        ply_record(0, "red", (9, 9), {(2, 2): 5}, 0.1)


def test_ply_record_still_fails_loud_on_empty_counts():
    with pytest.raises(ValueError):
        ply_record(0, "red", (2, 2), {}, 0.1)


def test_legacy_fields_are_unchanged():
    # B2 is additive: nothing that already existed may shift.
    rec = ply_record(7, "black", (1, 1), {(2, 2): 190, (1, 1): 40}, -0.25)
    assert rec["ply"] == 7
    assert rec["player"] == "black"
    assert (rec["row"], rec["col"]) == (1, 1)
    assert rec["root_value"] == pytest.approx(-0.25)
    assert rec["root_top1_share"] == pytest.approx(190 / 230)
    assert rec["selected_visit_rank"] == 2
    assert rec["selected_visit_count"] == 40
    assert rec["root_total_visits"] == 230
    assert rec["n_legal"] == 2


# --- Task B4: per-agent readout, split RNG, live capture -------------------


def test_readout_from_eval_config_maps_argmax_mode():
    assert readout_from_eval_config(
        EvalConfig(selection_mode="argmax")).mode == R.MODE_ARGMAX


def test_readout_from_eval_config_maps_opening_temperature():
    rd = readout_from_eval_config(EvalConfig(selection_mode="opening_temperature"))
    assert rd.mode == R.MODE_OPENING_TEMPERATURE
    assert rd.temp_high == 1.0 and rd.temp_low == 0.1
    assert rd.opening_temp_plies == 20


def test_readout_from_eval_config_rejects_unknown_modes():
    with pytest.raises(ValueError):
        readout_from_eval_config(EvalConfig(selection_mode="wishful"))


def _search_once(seed):
    """One independent search from the same fixed state and search seed."""
    state = TwixtState(active_size=8, to_move="red", max_plies_limit=16)
    mcts = MCTS(FakeEvaluator(0.0), MCTSConfig(n_simulations=32, eval_batch_size=4,
                                               stall_flush_sims=4),
                random.Random(seed))
    counts, root_value, root = mcts.search_with_root(state, add_noise=False)
    return counts, root_value, R.top_two(root_child_stats(counts, root))


def test_search_identity_across_two_independent_searches():
    """Two INDEPENDENT searches from the same state and search seed must
    produce identical visit counts, root value and top-two telemetry.

    Feeding one completed tree to two readouts would be tautological -- the
    statistics are the same object. The real claim is that the search is
    unaffected by which readout is configured, and only two separate runs can
    test it.

    This is a per-position property. It is FALSE across a game by
    construction, so it must never be asserted at game level.
    """
    counts_a, rv_a, top2_a = _search_once(7)
    counts_b, rv_b, top2_b = _search_once(7)

    assert counts_a == counts_b
    assert rv_a == rv_b
    assert [(s.move, s.visits, s.q_child, s.q_root) for s in top2_a] == \
           [(s.move, s.visits, s.q_child, s.q_root) for s in top2_b]

    a, _ = R.select(counts_a, 5, R.ReadoutConfig(mode=R.MODE_ARGMAX),
                    random.Random(1))
    b, _ = R.select(counts_b, 5, R.ReadoutConfig(mode=R.MODE_HOEFFDING_LCB,
                                                 opening_temp_plies=2),
                    random.Random(1), top2=top2_b)
    assert a in counts_a and b in counts_b


def test_search_identity_test_can_actually_fail():
    """CONSTRUCTED negative case: a DIFFERENT search seed must change the
    tree. If it does not, the identity test above is vacuous on this fixture.
    """
    counts_a, _rv, _t = _search_once(7)
    counts_c, _rv2, _t2 = _search_once(9999)
    assert counts_a != counts_c, (
        "search is seed-insensitive on this fixture -- the identity test "
        "proves nothing; enlarge the board or the simulation count")


def test_readout_cannot_advance_the_search_rng():
    """CONSTRUCTED negative case for the RNG split.

    mcts.MCTS draws prior-shuffle, PUCT tie-break and readout from ONE
    self.rng. The eval readout must never touch it -- otherwise a readout that
    consumes a different number of draws changes the generator state entering
    every later search, and the experiment is not readout-only.

    The negative case is mcts.select_move, which DOES advance the stream. If
    that assertion ever stops holding, this test has gone vacuous.
    """
    state = TwixtState(active_size=8, to_move="red", max_plies_limit=16)
    mcts = MCTS(FakeEvaluator(0.0), MCTSConfig(n_simulations=32, eval_batch_size=4,
                                               stall_flush_sims=4),
                random.Random(7))
    counts, _rv, root = mcts.search_with_root(state, add_noise=False)
    top2 = R.top_two(root_child_stats(counts, root))

    before = mcts.rng.getstate()
    R.select(counts, 0, R.ReadoutConfig(mode=R.MODE_OPENING_TEMPERATURE),
             random.Random(1))
    R.select(counts, 21, R.ReadoutConfig(mode=R.MODE_HOEFFDING_LCB),
             random.Random(1), top2=top2)
    assert mcts.rng.getstate() == before, "eval readout touched the search RNG"

    mcts.select_move(counts, ply=0)
    assert mcts.rng.getstate() != before, (
        "mcts.select_move no longer advances self.rng -- this test can no "
        "longer detect a leak and must be redesigned")


def test_root_visit_count_matches_the_nominal_budget():
    """Pins the MEASURED cold-root budget invariant (verified at 8/16/32/400)."""
    for sims in (8, 16, 32):
        state = TwixtState(active_size=8, to_move="red", max_plies_limit=16)
        mcts = MCTS(FakeEvaluator(0.0),
                    MCTSConfig(n_simulations=sims, eval_batch_size=4,
                               stall_flush_sims=4),
                    random.Random(3))
        _counts, _rv, root = mcts.search_with_root(state, add_noise=False)
        assert root.visit_count == sims


def test_root_child_stats_maps_unvisited_children_to_none():
    state = TwixtState(active_size=8, to_move="red", max_plies_limit=16)
    mcts = MCTS(FakeEvaluator(0.0), MCTSConfig(n_simulations=8, eval_batch_size=4,
                                               stall_flush_sims=4),
                random.Random(3))
    counts, _rv, root = mcts.search_with_root(state, add_noise=False)
    stats = root_child_stats(counts, root)
    assert set(stats) == set(counts)
    for move, (visits, q_child) in stats.items():
        if visits == 0:
            assert q_child is None, f"{move}: undefined mean must be None"
        else:
            assert q_child is not None


def test_play_eval_game_binds_each_readout_to_its_colour():
    red_rd = R.ReadoutConfig(mode=R.MODE_ARGMAX)
    black_rd = R.ReadoutConfig(mode=R.MODE_OPENING_TEMPERATURE,
                               opening_temp_plies=2, temp_high=1.0, temp_low=0.0)
    _w, reason, n, records = play_eval_game(
        FakeEvaluator(0.0), FakeEvaluator(0.0), SMALL, seed=11, capture=True,
        red_readout=red_rd, black_readout=black_rd)
    assert n > 0
    assert reason in {"win", "state_cap", "board_full"}
    # Red plays argmax at EVERY ply, so its selected visit rank is always 1.
    red_ranks = {r["selected_visit_rank"] for r in records if r["player"] == "red"}
    assert red_ranks == {1}


def test_play_eval_game_captures_top2_at_every_ply():
    rd = R.ReadoutConfig(mode=R.MODE_HOEFFDING_LCB, opening_temp_plies=2)
    _w, _r, _n, records = play_eval_game(
        FakeEvaluator(0.0), FakeEvaluator(0.0), SMALL, seed=13, capture=True,
        red_readout=rd, black_readout=rd)
    post = [r for r in records if r["ply"] >= 2]
    assert post, "test board must produce post-opening plies"
    assert all(r["top2"] is not None for r in records)
    assert all(r["readout_overrode_leader"] in (True, False) for r in post)
    for r in records:
        for stat in r["top2"]:
            assert stat["completed_visit_count"] >= 0
            if stat["completed_visit_count"] > 0:
                assert stat["q_value_root_perspective"] == pytest.approx(
                    -stat["q_value_child_perspective"])


def test_play_eval_game_defaults_to_the_eval_config_readout():
    """Both calls run the SAME implementation at the SAME seed, and
    readout_from_eval_config returns exactly what the default path builds, so
    the games must be identical. Disagreement is a bug in default resolution
    -- fix the code, not this assertion."""
    w1 = play_eval_game(FakeEvaluator(0.0), FakeEvaluator(0.0), SMALL, seed=5)
    w2 = play_eval_game(FakeEvaluator(0.0), FakeEvaluator(0.0), SMALL, seed=5,
                        red_readout=readout_from_eval_config(SMALL),
                        black_readout=readout_from_eval_config(SMALL))
    assert w1[:3] == w2[:3]


def test_play_eval_game_capture_does_not_change_the_outcome():
    off = play_eval_game(FakeEvaluator(0.0), FakeEvaluator(0.0), SMALL, seed=21)
    on = play_eval_game(FakeEvaluator(0.0), FakeEvaluator(0.0), SMALL, seed=21,
                        capture=True)
    assert off[:3] == on[:3]
