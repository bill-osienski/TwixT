# tests/test_warm_prefix_replay.py
import random

import pytest

from scripts.GPU.alphazero.corpus_geometry import GameMeta
from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
from scripts.GPU.alphazero.warm_prefix_replay import replay_prefix

from tests.eval_fakes import FakeEvaluator

BASE = 20400000
SIZE = 6


def shipped_cfg(n_simulations: int = 400) -> MCTSConfig:
    return MCTSConfig(n_simulations=n_simulations, eval_batch_size=14,
                      stall_flush_sims=48, pending_virtual_visits=8)


def _mcts(seed, n_simulations=400, **kw):
    return MCTS(FakeEvaluator(value=0.0), shipped_cfg(n_simulations),
                random.Random(seed), **kw)


def _legal_history(n_plies, size=SIZE, start_player="red"):
    """A genuinely legal move sequence, derived from the state itself.

    Row-major cells are NOT safe: (0, 0) is a corner and illegal, and each side
    has its own restricted border rows/columns. Walking legal_moves() keeps the
    fixture deterministic and correct on any board size.
    """
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    s = TwixtState(active_size=size, to_move=start_player)
    moves = []
    for _ in range(n_plies):
        lm = s.legal_moves()
        if not lm:
            break
        mv = lm[0]                 # deterministic: first legal move
        moves.append(mv)
        s = s.apply_move(mv)
    return moves


def _meta(game_idx=0, n_moves=12):
    return GameMeta(game_id=game_idx, seed=BASE + game_idx, n_moves=n_moves,
                    start_player="red")


def test_prefix_produces_a_NONZERO_inherited_root():
    """The gap Stage 1 could not close: every Stage 1 search started fresh, so
    I was always 0 and the subtraction was trivially correct."""
    m = _mcts(BASE)
    r = replay_prefix(m, _meta(n_moves=4), _legal_history(4), target_ply=3,
                      active_size=SIZE)
    assert r.inherited_I > 0
    assert r.root.visit_count == r.inherited_I


def test_prefix_stops_immediately_before_the_target_search():
    m = _mcts(BASE)
    r = replay_prefix(m, _meta(n_moves=4), _legal_history(4), target_ply=3,
                      active_size=SIZE)
    assert [s.ply for s in r.steps] == [0, 1, 2]      # searches at plies 0..2
    assert len(r.steps) == 3


def test_target_ply_zero_yields_a_cold_root():
    m = _mcts(BASE)
    r = replay_prefix(m, _meta(n_moves=4), _legal_history(4), target_ply=0,
                      active_size=SIZE)
    assert r.steps == [] and r.inherited_I == 0


def test_forced_move_present_records_exact_visits_not_a_threshold():
    m = _mcts(BASE)
    r = replay_prefix(m, _meta(n_moves=4), _legal_history(4), target_ply=3,
                      active_size=SIZE)
    for s in r.steps:
        if not s.inheritance_reset:
            assert isinstance(s.forced_child_visits, int)
            assert s.forced_child_visits >= 0
            # zero_effective_inheritance unions absent-or-zero WITHOUT losing
            # the underlying pair.
            assert s.zero_effective_inheritance == (s.forced_child_visits == 0)


def test_reset_statistics_are_recorded_and_no_row_is_dropped():
    m = _mcts(BASE)
    r = replay_prefix(m, _meta(n_moves=4), _legal_history(4), target_ply=3,
                      active_size=SIZE)
    assert r.reset_count == sum(1 for s in r.steps if s.inheritance_reset)
    assert r.reset_rate == pytest.approx(r.reset_count / len(r.steps))
    assert len(r.steps) == 3          # every step retained
    if r.reset_count == 0:
        assert r.last_reset_ply is None      # None, never -1 and never 0


def test_prefix_rejects_a_target_ply_outside_the_history():
    m = _mcts(BASE)
    with pytest.raises(ValueError):
        replay_prefix(m, _meta(n_moves=4), _legal_history(4), target_ply=9,
                      active_size=SIZE)


# append to tests/test_warm_prefix_replay.py
from scripts.GPU.alphazero.warm_prefix_replay import (
    BOUNDARY_THRESHOLD, BatchSafeBoundaryObserver,
)


class _FakeRoot:
    def __init__(self, visit_count):
        self.visit_count = visit_count


def test_boundary_fires_at_the_first_flush_at_or_after_320():
    obs = BatchSafeBoundaryObserver(inherited_I=21)
    for total in (21 + 300, 21 + 318, 21 + 326, 21 + 340):
        obs.on_flush_complete("full", _FakeRoot(total))
    rec = obs.record
    assert rec is not None
    assert rec.N_actual == 326                 # first at-or-after 320, not 340
    assert rec.overshoot == 6
    assert rec.remaining == 400 - 326
    assert rec.flush_type == "full"


def test_boundary_subtracts_inherited_I():
    """The Stage 1 gap: with I = 0 this subtraction is trivially correct and
    proves nothing."""
    obs = BatchSafeBoundaryObserver(inherited_I=137)
    obs.on_flush_complete("full", _FakeRoot(137 + 322))
    assert obs.record.N_actual == 322


def test_boundary_ignores_later_flushes_once_captured():
    obs = BatchSafeBoundaryObserver(inherited_I=0)
    obs.on_flush_complete("full", _FakeRoot(321))
    obs.on_flush_complete("tail", _FakeRoot(400))
    assert obs.record.N_actual == 321 and obs.record.flush_type == "full"


def test_boundary_is_none_until_the_threshold_is_reached():
    obs = BatchSafeBoundaryObserver(inherited_I=0)
    obs.on_flush_complete("full", _FakeRoot(300))
    assert obs.record is None          # None, never a zero-filled record


def test_tail_only_search_yields_zero_remaining():
    """The degenerate case the deployability rule exists to catch."""
    obs = BatchSafeBoundaryObserver(inherited_I=0)
    obs.on_flush_complete("tail", _FakeRoot(400))
    assert obs.record.N_actual == 400 and obs.record.remaining == 0


def test_boundary_asserts_the_frozen_range():
    obs = BatchSafeBoundaryObserver(inherited_I=0)
    with pytest.raises(AssertionError):
        obs.on_flush_complete("full", _FakeRoot(401))


def test_boundary_in_a_REAL_400_sim_leg_on_a_warm_root():
    """End-to-end: a real batched search on a genuinely inherited root."""
    m = _mcts(BASE)
    pre = replay_prefix(m, _meta(n_moves=4), _legal_history(4), target_ply=2,
                        active_size=SIZE)
    assert pre.inherited_I > 0
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I)
    # SAME MCTS, so the single frozen RNG stream continues from the prefix --
    # a fresh MCTS would reset it, which section 2b forbids. The observer is
    # attached only now, so it cannot capture a boundary from a PREFIX search.
    m._flush_observer = obs
    m.search_from_root(pre.root, add_noise=False, ply=2)
    rec = obs.record
    assert rec is not None
    assert BOUNDARY_THRESHOLD <= rec.N_actual <= 400
    assert rec.remaining == 400 - rec.N_actual
    assert rec.flush_type in {"full", "stall", "tail"}


# append to tests/test_warm_prefix_replay.py
from scripts.GPU.alphazero.mcts import encode_move, visit_leader_move
from scripts.GPU.alphazero.warm_prefix_replay import (
    LEG_INCREMENTS, NOMINAL_B, run_additive_ladder,
)


def test_leg_increments_sum_to_the_frozen_nominal_budgets():
    assert LEG_INCREMENTS == (400, 1200, 1600, 3200)
    assert NOMINAL_B == (400, 1600, 3200, 6400)
    running = 0
    for inc, b in zip(LEG_INCREMENTS, NOMINAL_B):
        running += inc
        assert running == b
    assert sum(LEG_INCREMENTS) == 6400


def test_ladder_records_B_I_and_effective_separately():
    m = _mcts(BASE, n_simulations=1)          # per-leg budget is set by the ladder
    pre = replay_prefix(m, _meta(n_moves=4), _legal_history(4), target_ply=2,
                        active_size=SIZE)
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I)
    legs, _snaps = run_additive_ladder(m, pre.root, pre.inherited_I, ply=2,
                                       boundary_observer=obs,
                                       increments=(4, 4, 4, 4))   # tiny, for speed
    assert [l.nominal_B for l in legs] == [4, 8, 12, 16]
    assert all(l.inherited_I == pre.inherited_I for l in legs)
    assert all(l.effective == l.inherited_I + l.nominal_B for l in legs)


def test_ladder_is_additive_on_one_tree_not_four_searches():
    """Root visits accumulate; each leg continues the previous tree."""
    m = _mcts(BASE, n_simulations=1)
    pre = replay_prefix(m, _meta(n_moves=4), _legal_history(4), target_ply=2,
                        active_size=SIZE)
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I)
    run_additive_ladder(m, pre.root, pre.inherited_I, ply=2,
                        boundary_observer=obs, increments=(4, 4, 4, 4))
    assert pre.root.visit_count == pre.inherited_I + 16


def test_every_rung_preserves_its_own_evidence():
    """After leg 4 the tree is at 6,400 and the earlier rungs are GONE. Each
    LegResult must already carry what sections 5 and 7 need."""
    m = _mcts(BASE, n_simulations=1)
    pre = replay_prefix(m, _meta(n_moves=4), _legal_history(4), target_ply=2,
                        active_size=SIZE)
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I)
    legs, _ = run_additive_ladder(m, pre.root, pre.inherited_I, ply=2,
                                  boundary_observer=obs,
                                  increments=(8, 8, 8, 8))
    for leg in legs:
        assert leg.visit_counts and all(v > 0 for v in leg.visit_counts.values())
        assert leg.n_visited_children == len(leg.visit_counts)
        assert leg.top_share is not None
        assert leg.effective_children is not None
        assert leg.selected_move_prior_rank is not None
    # The distributions genuinely differ between rungs -- otherwise preserving
    # them would be pointless.
    assert legs[0].visit_counts != legs[-1].visit_counts
    assert sum(legs[0].visit_counts.values()) < sum(legs[-1].visit_counts.values())


def test_tracer_snapshots_are_taken_at_DISTINCT_times_in_a_real_400_leg():
    """Section 8's two frozen snapshots, on the grounded real case.

    A tiny 8-simulation leg would make the tail the ONLY qualifying flush, so
    both snapshots would capture the same instant and `a <= b` would pass while
    proving nothing. At 400 simulations with batch 14 there are ~28 full flushes,
    so the boundary lands near 320 with real budget left after it.
    """
    m = _mcts(BASE, n_simulations=1)
    pre = replay_prefix(m, _meta(n_moves=4), _legal_history(4), target_ply=2,
                        active_size=SIZE)
    target_tracer = SelectionTracer()          # FRESH, attached after the prefix
    m._selection_observer = target_tracer
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I,
                                    tracer=target_tracer)
    _legs, snaps = run_additive_ladder(
        m, pre.root, pre.inherited_I, ply=2, boundary_observer=obs,
        target_tracer=target_tracer,
        increments=(400, 4, 4, 4))       # leg 1 REAL; later legs tiny for speed

    assert obs.record is not None
    assert obs.record.remaining > 0, (
        "the boundary landed on the tail flush, so both snapshots describe the "
        "same instant and this test would be vacuous")
    assert snaps["at_boundary"] is not None and snaps["at_400"] is not None
    a = snaps["at_boundary"]["by_shape"]["c4a05"]["overall"]["eligible_events"]
    b = snaps["at_400"]["by_shape"]["c4a05"]["overall"]["eligible_events"]
    assert 0 < a < b, "snapshots must be taken at DISTINCT times"


def test_ladder_rejects_a_wrong_length_increment_tuple():
    m = _mcts(BASE, n_simulations=1)
    pre = replay_prefix(m, _meta(n_moves=4), _legal_history(4), target_ply=1,
                        active_size=SIZE)
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I)
    with pytest.raises(ValueError):
        run_additive_ladder(m, pre.root, pre.inherited_I, ply=1,
                            boundary_observer=obs, increments=(4, 4))


# append to tests/test_warm_prefix_replay.py
from scripts.GPU.alphazero.selection_tracer import SelectionTracer


def test_tracer_cache_is_cleared_at_every_real_advance_root():
    """Stage 1 could only call clear_node_cache() directly. Here the clears
    happen at genuine advance_root boundaries, where detached subtrees free
    id() values for reuse -- the hazard the cache lifetime exists to prevent."""
    tracer = SelectionTracer()
    m = _mcts(BASE, selection_observer=tracer)
    replay_prefix(m, _meta(n_moves=5), _legal_history(5), target_ply=4, active_size=SIZE)
    # After the final advance_root the cache holds nothing stale.
    assert tracer._cache == {}


def test_target_tracer_is_attached_AFTER_the_prefix():
    """The prefix runs with NO target tracer, so its events never enter section
    8's counters."""
    m = _mcts(BASE, n_simulations=1)
    pre = replay_prefix(m, _meta(n_moves=4), _legal_history(4), target_ply=2,
                        active_size=SIZE)
    tracer = SelectionTracer()                    # FRESH, after the prefix
    m._selection_observer = tracer
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I, threshold=4,
                                    leg_B=8, tracer=tracer)
    run_additive_ladder(m, pre.root, pre.inherited_I, ply=2,
                        boundary_observer=obs, target_tracer=tracer,
                        increments=(8, 8, 8, 8))
    snap = tracer.snapshot()
    for shape in ("c4a05", "c13a03"):
        cell = snap["by_shape"][shape]["overall"]
        assert cell["eligible_events"] > 0
        assert cell["outside_rate"] is not None      # denominator is non-empty


def test_ladder_refuses_a_contaminated_tracer():
    """A tracer that ran through the prefix must be rejected, not silently used
    -- nothing downstream could tell the difference."""
    tracer = SelectionTracer()
    m = _mcts(BASE, n_simulations=1, selection_observer=tracer)
    pre = replay_prefix(m, _meta(n_moves=4), _legal_history(4), target_ply=2,
                        active_size=SIZE)
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I)
    with pytest.raises(ValueError, match="not empty"):
        run_additive_ladder(m, pre.root, pre.inherited_I, ply=2,
                            boundary_observer=obs, target_tracer=tracer,
                            increments=(4, 4, 4, 4))


def test_ladder_refuses_a_tracer_that_is_not_the_selection_observer():
    m = _mcts(BASE, n_simulations=1)
    pre = replay_prefix(m, _meta(n_moves=4), _legal_history(4), target_ply=2,
                        active_size=SIZE)
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I)
    with pytest.raises(ValueError, match="selection observer"):
        run_additive_ladder(m, pre.root, pre.inherited_I, ply=2,
                            boundary_observer=obs,
                            target_tracer=SelectionTracer(),   # never attached
                            increments=(4, 4, 4, 4))


def test_cache_is_cleared_ONCE_PER_ADVANCE_not_just_at_the_end():
    """Final emptiness alone would pass if only the LAST advance cleared it."""
    tracer = SelectionTracer()
    m = _mcts(BASE, selection_observer=tracer)
    r = replay_prefix(m, _meta(n_moves=5), _legal_history(5), target_ply=4,
                      active_size=SIZE)
    assert r.cache_clears == len(r.steps) == 4


def test_prefix_asserts_canonical_state_agreement_at_every_ply():
    m = _mcts(BASE)
    r = replay_prefix(m, _meta(n_moves=4), _legal_history(4), target_ply=3,
                      active_size=SIZE)
    assert all(s.state_agrees for s in r.steps)


def test_prefix_rejects_an_illegal_recorded_move():
    m = _mcts(BASE)
    history = _legal_history(4)
    history[2] = history[0]              # repeat an occupied cell
    with pytest.raises(ValueError, match="not legal"):
        replay_prefix(m, _meta(n_moves=4), history, target_ply=3,
                      active_size=SIZE)


def test_prefix_rejects_metadata_that_disagrees_with_the_history():
    m = _mcts(BASE)
    with pytest.raises(ValueError, match="disagree"):
        replay_prefix(m, _meta(n_moves=99), _legal_history(4), target_ply=2,
                      active_size=SIZE)


def test_within_forced_simulation_is_observed_during_a_warm_replay():
    """Stage 1 saw this covariate only on synchronous forced simulations."""
    tracer = SelectionTracer()
    m = _mcts(BASE, selection_observer=tracer)
    replay_prefix(m, _meta(n_moves=3), _legal_history(3), target_ply=2, active_size=SIZE)
    snap = tracer.snapshot()
    # A warm replay uses the BATCHED path, where no simulation is forced, so the
    # covariate must be present and zero -- not absent.
    assert "within_forced_events" in snap
    assert snap["within_forced_events"] == 0


# append to tests/test_warm_prefix_replay.py
def _row(seed, increments=(4, 4, 4, 4), reseed_between_legs=False):
    """Drive the legs directly rather than through run_additive_ladder.

    This task tests the RNG STREAM, not the ladder wrapper (Task 3 covers that),
    and driving searches directly avoids relying on zero-simulation legs to
    capture intermediate states.
    """
    m = MCTS(FakeEvaluator(value=0.0), shipped_cfg(1), random.Random(seed))
    pre = replay_prefix(m, _meta(n_moves=4), _legal_history(4), target_ply=2,
                        active_size=SIZE)
    states = [m.rng.getstate()]
    legs = []
    root = pre.root
    for i, inc in enumerate(increments):
        if reseed_between_legs and i > 0:
            m.rng = random.Random(seed)          # the FORBIDDEN behaviour
        m.config.n_simulations = inc
        vc, rv, root = m.search_from_root(root, add_noise=False, ply=2)
        # A local summary, NOT a LegResult: this task tests the RNG stream, and
        # coupling it to the ladder's dataclass shape would break it whenever
        # that shape changes -- as it just did.
        legs.append({
            "nominal_B": sum(increments[:i + 1]),
            "root_value": float(rv),
            "leader": visit_leader_move(root),          # canonical, not max()
            "visits": {encode_move(r, c): v for (r, c), v in vc.items() if v > 0},
        })
        states.append(m.rng.getstate())
    return legs, states


def test_the_rng_stream_advances_and_is_never_reseeded():
    _legs, states = _row(BASE)
    # Every checkpoint differs: the single stream is genuinely consumed.
    assert len({repr(s) for s in states}) == len(states)


def test_a_row_reproduces_exactly_under_the_same_replay_seed():
    a_legs, a_states = _row(BASE)
    b_legs, b_states = _row(BASE)
    assert [l["root_value"] for l in a_legs] == [l["root_value"] for l in b_legs]
    assert [l["leader"] for l in a_legs] == [l["leader"] for l in b_legs]
    assert [l["visits"] for l in a_legs] == [l["visits"] for l in b_legs]
    assert repr(a_states) == repr(b_states)


def test_reseeding_between_legs_CHANGES_the_result():
    """Non-vacuity: if continuity did not matter, this control would be
    indistinguishable and the continuity test would prove nothing."""
    ok_legs, _ = _row(BASE)
    bad_legs, _ = _row(BASE, reseed_between_legs=True)
    assert ([l["visits"] for l in ok_legs] != [l["visits"] for l in bad_legs]
            or [l["leader"] for l in ok_legs] != [l["leader"] for l in bad_legs]
            or [l["root_value"] for l in ok_legs] != [l["root_value"] for l in bad_legs])


def test_replay_seed_must_match_the_sidecar():
    """Section 2b: replay_seed = base_seed + game_idx, VERIFIED against the
    sidecar, not merely assumed."""
    from scripts.GPU.alphazero.warm_prefix_replay import replay_seed_for
    meta = _meta(game_idx=7)
    assert replay_seed_for(meta, base_seed=BASE) == BASE + 7
    bad = GameMeta(game_id=7, seed=BASE + 999, n_moves=12, start_player="red")
    with pytest.raises(ValueError, match="seed"):
        replay_seed_for(bad, base_seed=BASE)
