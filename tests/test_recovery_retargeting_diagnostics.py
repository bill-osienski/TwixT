"""Tests for Spec 4 recovery / re-targeting diagnostic."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from scripts.GPU.alphazero.recovery_retargeting_diagnostics import (
    RecoveryRetargetingConfig,
    validate_config,
)


def test_config_defaults_match_spec():
    c = RecoveryRetargetingConfig()
    assert c.enabled is True
    assert c.collapse_value_threshold == -0.75
    assert c.severe_collapse_value_threshold == -0.90
    assert c.diffuse_root_top1_threshold == 0.20
    assert c.very_diffuse_root_top1_threshold == 0.15
    assert c.delta_threshold == 0.50
    assert c.delta_max_current_score == -0.30
    assert c.alternate_component_min_size == 4
    assert c.classify_defense is True
    assert c.max_sampled_moves_per_side == 32
    assert c.sample_all_moves is False


def test_validate_collapse_lt_delta_max_current_score():
    cfg = RecoveryRetargetingConfig(collapse_value_threshold=-0.30, delta_max_current_score=-0.30)
    with pytest.raises(ValueError, match="collapse_value_threshold"):
        validate_config(cfg)


def test_validate_severe_le_collapse():
    cfg = RecoveryRetargetingConfig(collapse_value_threshold=-0.75, severe_collapse_value_threshold=-0.50)
    with pytest.raises(ValueError, match="severe_collapse_value_threshold"):
        validate_config(cfg)


def test_validate_very_diffuse_le_diffuse():
    cfg = RecoveryRetargetingConfig(diffuse_root_top1_threshold=0.20, very_diffuse_root_top1_threshold=0.30)
    with pytest.raises(ValueError, match="very_diffuse_root_top1_threshold"):
        validate_config(cfg)


def test_validate_top1_range():
    with pytest.raises(ValueError, match="diffuse_root_top1_threshold"):
        validate_config(RecoveryRetargetingConfig(diffuse_root_top1_threshold=1.5))


def test_validate_delta_positive():
    with pytest.raises(ValueError, match="delta_threshold"):
        validate_config(RecoveryRetargetingConfig(delta_threshold=0.0))


def test_validate_alternate_component_min_size_positive():
    with pytest.raises(ValueError, match="alternate_component_min_size"):
        validate_config(RecoveryRetargetingConfig(alternate_component_min_size=0))


def test_validate_max_sampled_non_negative():
    with pytest.raises(ValueError, match="max_sampled_moves_per_side"):
        validate_config(RecoveryRetargetingConfig(max_sampled_moves_per_side=-1))


def test_validate_default_config_passes():
    validate_config(RecoveryRetargetingConfig())   # must not raise


from scripts.GPU.alphazero.recovery_retargeting_diagnostics import (
    find_components,
    is_local_to_existing,
    knight_neighbors,
    selected_component_after,
)


class _StubState:
    """Minimal state shim: exposes .pegs dict, apply_move, _get_connected_component."""
    def __init__(self, pegs_dict, to_move="black"):
        # pegs_dict: {(r, c): "red" | "black"}
        self.pegs = dict(pegs_dict)
        self.to_move = to_move

    def apply_move(self, move):
        """Return a NEW _StubState with `move` placed for the current side.

        The real TwixtState.apply_move alternates to_move; the stub mirrors that.
        Tests that need a specific side-to-move should construct the stub with
        the desired to_move and call apply_move once.
        """
        new_pegs = dict(self.pegs)
        new_pegs[move] = self.to_move
        return _StubState(new_pegs, to_move="red" if self.to_move == "black" else "black")

    def _get_connected_component(self, peg, side):
        # BFS over knight-distance neighbors of the same color, no enemy blocking check
        # (sufficient for unit tests; real state has full enemy-block logic)
        if peg not in self.pegs or self.pegs[peg] != side:
            return frozenset()
        visited = {peg}
        frontier = [peg]
        while frontier:
            cur = frontier.pop()
            for n in knight_neighbors(*cur):
                if n in self.pegs and self.pegs[n] == side and n not in visited:
                    visited.add(n)
                    frontier.append(n)
        return frozenset(visited)


def _state_after(state_before, side, move):
    """Test helper: build a new _StubState representing state_before + move for side."""
    new_pegs = dict(state_before.pegs)
    new_pegs[move] = side
    return _StubState(new_pegs)


class _IsolateState(_StubState):
    """Stub variant where each peg is its own component (simulates enemy-blocked
    bridges between knight-distance neighbors). Lets tests verify locality
    flags and redundant-reinforcement classification without bridge formation."""
    def _get_connected_component(self, peg, side):
        if peg not in self.pegs or self.pegs[peg] != side:
            return frozenset()
        return frozenset({peg})


def test_knight_neighbors_returns_8_offsets():
    n = set(knight_neighbors(5, 5))
    assert n == {(3, 4), (3, 6), (4, 3), (4, 7), (6, 3), (6, 7), (7, 4), (7, 6)}


def test_find_components_groups_by_bridge_connectivity():
    # Two black pegs at knight distance form one component; a third isolated peg is its own component.
    state = _StubState({(0, 0): "black", (1, 2): "black", (10, 10): "black"})
    comps = find_components(state, "black")
    assert len(comps) == 2
    sizes = sorted(len(c) for c in comps)
    assert sizes == [1, 2]


def test_find_components_skips_other_color():
    state = _StubState({(0, 0): "black", (1, 2): "red"})
    comps = find_components(state, "black")
    assert len(comps) == 1
    assert next(iter(comps)) == frozenset({(0, 0)})


def test_is_local_to_existing_true_when_knight_neighbor_exists():
    state = _StubState({(0, 0): "black"})
    assert is_local_to_existing(state, "black", (1, 2)) is True
    assert is_local_to_existing(state, "black", (2, 1)) is True


def test_is_local_to_existing_false_when_no_same_color_knight_neighbor():
    state = _StubState({(0, 0): "black"})
    # (2, 2) is Chebyshev-2 from (0, 0) but NOT knight-distance.
    assert is_local_to_existing(state, "black", (2, 2)) is False


def test_is_local_to_existing_ignores_other_color():
    state = _StubState({(1, 2): "red"})
    assert is_local_to_existing(state, "black", (0, 0)) is False


def test_selected_component_after_includes_new_peg_and_merged_components():
    """Caller passes state_after (post-move). Helper does NOT mutate state."""
    # Two prior black pegs at (0, 0) and (4, 0). (2, 1) is knight-distance from both.
    state_before = _StubState({(0, 0): "black", (4, 0): "black"})
    state_after = _state_after(state_before, "black", (2, 1))
    comp_after = selected_component_after(state_after, "black", (2, 1))
    assert (0, 0) in comp_after
    assert (4, 0) in comp_after
    assert (2, 1) in comp_after
    assert len(comp_after) == 3


def test_selected_component_after_uses_post_move_state_without_mutation():
    """The helper must NOT mutate state_after.pegs (or any state)."""
    state_before = _StubState({(0, 0): "black", (4, 0): "black"})
    state_after = _state_after(state_before, "black", (2, 1))
    pegs_before_call = dict(state_after.pegs)
    selected_component_after(state_after, "black", (2, 1))
    assert state_after.pegs == pegs_before_call
    # state_before is untouched (it never received the move).
    assert (2, 1) not in state_before.pegs


from scripts.GPU.alphazero.recovery_retargeting_diagnostics import evaluate_trigger


def _cfg(**overrides):
    return RecoveryRetargetingConfig(**overrides)


def test_steady_state_trigger_fires_when_score_and_top1_both_low():
    r = evaluate_trigger(
        current_search_score=-0.80, root_top1_share=0.10,
        previous_own_search_score=-0.70, config=_cfg(),
    )
    assert r["triggered"] is True
    assert r["trigger_reason"] == "steady_state"


def test_steady_state_does_not_fire_when_score_bad_but_root_confident():
    r = evaluate_trigger(
        current_search_score=-0.80, root_top1_share=0.40,
        previous_own_search_score=None, config=_cfg(),
    )
    assert r["triggered"] is False
    assert r["trigger_reason"] is None


def test_steady_state_does_not_fire_when_root_diffuse_but_score_ok():
    r = evaluate_trigger(
        current_search_score=-0.20, root_top1_share=0.10,
        previous_own_search_score=None, config=_cfg(),
    )
    assert r["triggered"] is False


def test_delta_precursor_fires_on_sharp_drop():
    r = evaluate_trigger(
        current_search_score=-0.40, root_top1_share=0.12,
        previous_own_search_score=0.30, config=_cfg(),
    )
    assert r["triggered"] is True
    assert r["trigger_reason"] == "delta_precursor"


def test_delta_precursor_guard_blocks_when_current_score_still_positive():
    # Drop from +0.95 to +0.40 = delta 0.55 >= 0.50, top1 diffuse, but current > -0.30 guard.
    r = evaluate_trigger(
        current_search_score=0.40, root_top1_share=0.10,
        previous_own_search_score=0.95, config=_cfg(),
    )
    assert r["triggered"] is False


def test_trigger_reason_both_when_both_paths_fire():
    # current=-0.80 (steady fires) AND previous=-0.20 → delta=0.60 → delta also fires
    r = evaluate_trigger(
        current_search_score=-0.80, root_top1_share=0.10,
        previous_own_search_score=-0.20, config=_cfg(),
    )
    assert r["triggered"] is True
    assert r["trigger_reason"] == "both"


def test_missing_search_score_skips_trigger():
    r = evaluate_trigger(
        current_search_score=None, root_top1_share=0.10,
        previous_own_search_score=-0.30, config=_cfg(),
    )
    assert r["triggered"] is False
    assert r["missing_search_score"] is True


def test_missing_root_top1_share_skips_trigger():
    r = evaluate_trigger(
        current_search_score=-0.80, root_top1_share=None,
        previous_own_search_score=None, config=_cfg(),
    )
    assert r["triggered"] is False
    assert r["missing_root_top1_share"] is True


def test_severity_flags_reflect_current_score_and_share():
    r = evaluate_trigger(
        current_search_score=-0.95, root_top1_share=0.10,
        previous_own_search_score=None, config=_cfg(),
    )
    assert r["triggered"] is True
    assert r["is_severe_collapse"] is True
    assert r["is_very_diffuse"] is True


from scripts.GPU.alphazero.recovery_retargeting_diagnostics import classify_move


def _classify(
    state_before, side, move,
    own_td_before, own_td_after,
    opp_td_before=None, opp_td_after=None,
    classify_defense=True,
    alternate_component_min_size=4,
    state_after=None,
):
    """Test harness wrapper. Caller may pass state_after explicitly to override
    the default (state_before + move) — useful for testing isolated-bridge scenarios."""
    if state_after is None:
        state_after = _state_after(state_before, side, move)
    return classify_move(
        state_before=state_before,
        state_after=state_after,
        side=side,
        move=move,
        own_total_goal_distance_before=own_td_before,
        own_total_goal_distance_after=own_td_after,
        opponent_total_goal_distance_before=opp_td_before,
        opponent_total_goal_distance_after=opp_td_after,
        classify_defense=classify_defense,
        alternate_component_min_size=alternate_component_min_size,
    )


def test_classify_move_does_not_mutate_state_before():
    """classify_move must not mutate state_before.pegs in any code path."""
    state_before = _StubState({(0, 0): "black", (1, 2): "black"})
    state_after = _state_after(state_before, "black", (5, 5))
    pegs_snapshot = dict(state_before.pegs)
    classify_move(
        state_before=state_before, state_after=state_after,
        side="black", move=(5, 5),
        own_total_goal_distance_before=4, own_total_goal_distance_after=4,
        opponent_total_goal_distance_before=None,
        opponent_total_goal_distance_after=None,
        classify_defense=True, alternate_component_min_size=4,
    )
    assert state_before.pegs == pegs_snapshot


def test_classifies_blocks_opponent_closeout():
    state = _StubState({(0, 0): "black"})
    r = _classify(state, "black", (5, 5),
                  own_td_before=6, own_td_after=6,
                  opp_td_before=2, opp_td_after=3)
    assert r["primary_class"] == "blocks_opponent_closeout"
    assert r["flags"]["blocked_opponent_closeout"] is True


def test_classifies_reduces_own_goal_distance():
    state = _StubState({(0, 0): "black"})
    r = _classify(state, "black", (5, 5),
                  own_td_before=4, own_td_after=3)
    assert r["primary_class"] == "reduces_own_goal_distance"


def test_priority_defense_beats_reduces_goal_distance():
    state = _StubState({(0, 0): "black"})
    r = _classify(state, "black", (5, 5),
                  own_td_before=4, own_td_after=3,         # reduces own goal distance
                  opp_td_before=2, opp_td_after=3)         # also blocks opponent
    assert r["primary_class"] == "blocks_opponent_closeout"


def test_classifies_starts_or_extends_alternate_via_opens_new():
    # Dominant black component at (0,0)-(1,2) size 2; move at (10,10) opens new component size 1.
    # alternate_component_min_size=1 to make this test independent of default.
    state = _StubState({(0, 0): "black", (1, 2): "black"})
    r = _classify(state, "black", (10, 10),
                  own_td_before=5, own_td_after=5,
                  alternate_component_min_size=1)
    assert r["primary_class"] == "starts_or_extends_alternate_component"
    assert r["flags"]["opens_new_component"] is True


def test_classifies_connects_to_existing_component():
    # Move bridges to dominant component but does NOT reduce td.
    state = _StubState({(0, 0): "black", (1, 2): "black", (3, 1): "black"})
    # New move at (4, 3) is knight-from (3, 1) and joins dominant.
    r = _classify(state, "black", (4, 3),
                  own_td_before=5, own_td_after=5)
    assert r["primary_class"] == "connects_to_existing_component"
    assert r["flags"]["extends_dominant_component"] is True


def test_classifies_redundant_local_reinforcement():
    # Move is local (knight-distance) to a same-color peg, but the simulated
    # bridge is blocked, so the move does NOT actually join the component.
    state_before = _IsolateState({(0, 0): "black"})
    new_pegs = dict(state_before.pegs)
    new_pegs[(1, 2)] = "black"
    state_after = _IsolateState(new_pegs)
    r = _classify(state_before, "black", (1, 2),                # knight-local to (0, 0)
                  own_td_before=5, own_td_after=5,
                  state_after=state_after)
    assert r["primary_class"] == "redundant_local_reinforcement"
    assert r["flags"]["local_to_existing"] is True
    assert r["flags"]["extends_dominant_component"] is False


def test_classifies_off_plan_or_unclear_fallback():
    state = _StubState({(0, 0): "black"})
    # Move is far away (not local), no td change, no defense.
    r = _classify(state, "black", (15, 15),
                  own_td_before=5, own_td_after=5)
    assert r["primary_class"] == "off_plan_or_unclear"


def test_local_to_existing_uses_knight_not_chebyshev():
    # (2, 2) is Chebyshev-2 from (0, 0) but NOT knight-2.
    state_before = _IsolateState({(0, 0): "black"})
    new_pegs = dict(state_before.pegs)
    new_pegs[(2, 2)] = "black"
    state_after = _IsolateState(new_pegs)
    r = _classify(state_before, "black", (2, 2),
                  own_td_before=5, own_td_after=5,
                  state_after=state_after)
    assert r["flags"]["local_to_existing"] is False
    assert r["primary_class"] == "off_plan_or_unclear"


def test_classify_defense_disabled_never_returns_blocks_opponent_closeout():
    state = _StubState({(0, 0): "black"})
    r = _classify(state, "black", (5, 5),
                  own_td_before=4, own_td_after=3,
                  opp_td_before=2, opp_td_after=3,
                  classify_defense=False)
    assert r["primary_class"] == "reduces_own_goal_distance"
    assert r["flags"]["blocked_opponent_closeout"] is False


from scripts.GPU.alphazero.recovery_retargeting_diagnostics import RecoveryRetargetingTracker


def _gc_stub(td_before, td_after):
    """Helper to build a goal-completion-state provider that returns fixed tds."""
    calls = {"n": 0}
    def provider(state, side, enumerate_moves=False):
        calls["n"] += 1
        return {"total_goal_distance": td_before if calls["n"] % 2 == 1 else td_after}
    return provider


def test_observe_move_not_in_window_no_classify():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(
        state_before=state, selected_move=(5, 5), ply=10, side_to_move="black",
        search_score=+0.20, root_top1_share=0.30,
    )
    snap = tracker.side_snapshot("black")
    assert snap["triggered"] is False
    assert snap["in_window_own_moves"] == 0


def test_observe_move_opens_window_on_trigger():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(
        state_before=state, selected_move=(5, 5), ply=44, side_to_move="black",
        search_score=-0.85, root_top1_share=0.12,
    )
    snap = tracker.side_snapshot("black")
    assert snap["triggered"] is True
    assert snap["first_trigger_ply"] == 44
    assert snap["first_trigger_reason"] == "steady_state"
    assert snap["in_window_own_moves"] == 1
    assert snap["triggered_own_moves"] == 1


def test_observe_move_window_stays_open_across_non_triggered_plies():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(state_before=state, selected_move=(5, 5), ply=44, side_to_move="black", search_score=-0.85, root_top1_share=0.12)
    tracker.observe_move(state_before=state, selected_move=(6, 6), ply=46, side_to_move="black", search_score=-0.20, root_top1_share=0.30)
    snap = tracker.side_snapshot("black")
    assert snap["in_window_own_moves"] == 2
    assert snap["triggered_own_moves"] == 1
    assert snap["non_triggered_in_window_moves"] == 1


def test_observe_move_missing_signal_in_window_counts_separately():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(state_before=state, selected_move=(5, 5), ply=44, side_to_move="black", search_score=-0.85, root_top1_share=0.12)
    tracker.observe_move(state_before=state, selected_move=(6, 6), ply=46, side_to_move="black", search_score=None, root_top1_share=0.20)
    snap = tracker.side_snapshot("black")
    assert snap["missing_signal_moves"] == 1
    assert snap["missing_search_score_moves"] == 1
    assert sum(snap["selected_class_counts"].values()) == 1


def test_observe_move_other_side_does_not_affect_window():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(state_before=state, selected_move=(5, 5), ply=44, side_to_move="black", search_score=-0.85, root_top1_share=0.12)
    tracker.observe_move(state_before=state, selected_move=(6, 6), ply=45, side_to_move="red", search_score=-0.85, root_top1_share=0.12)
    snap = tracker.side_snapshot("black")
    assert snap["in_window_own_moves"] == 1
    red_snap = tracker.side_snapshot("red")
    assert red_snap["triggered"] is True
    assert red_snap["in_window_own_moves"] == 1


def test_observe_move_does_not_mutate_state_before():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black", (1, 2): "black"})
    pegs_snapshot = dict(state.pegs)
    tracker.observe_move(
        state_before=state, selected_move=(5, 5), ply=44, side_to_move="black",
        search_score=-0.85, root_top1_share=0.12,
    )
    assert state.pegs == pegs_snapshot


def test_observe_move_in_window_includes_missing_signal_in_count():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(state_before=state, selected_move=(5, 5), ply=44, side_to_move="black", search_score=-0.85, root_top1_share=0.12)
    tracker.observe_move(state_before=state, selected_move=(6, 6), ply=46, side_to_move="black", search_score=None, root_top1_share=0.20)
    tracker.observe_move(state_before=state, selected_move=(7, 7), ply=48, side_to_move="black", search_score=-0.80, root_top1_share=0.10)
    snap = tracker.side_snapshot("black")
    assert snap["in_window_own_moves"] == 3
    assert snap["missing_signal_moves"] == 1
    assert sum(snap["selected_class_counts"].values()) == 2


def test_observe_move_sampled_entry_previous_score_is_pre_current():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(state_before=state, selected_move=(5, 5), ply=44, side_to_move="black", search_score=-0.85, root_top1_share=0.12)
    tracker.observe_move(state_before=state, selected_move=(6, 6), ply=46, side_to_move="black", search_score=-0.99, root_top1_share=0.10)
    side_acc = tracker._sides["black"]
    entry_46 = next(e for e in side_acc.sampled_moves if e["ply"] == 46)
    assert entry_46["previous_own_search_score"] == -0.85
    assert entry_46["current_search_score"] == -0.99


def test_finalize_returns_none_when_no_side_triggered():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(state_before=state, selected_move=(5, 5), ply=10, side_to_move="black", search_score=+0.20, root_top1_share=0.30)
    rec = tracker.finalize_game(
        iteration=0, game_idx=0, game_id="game_000",
        winner="red", starting_player="red", n_moves=65, reason="win",
    )
    assert rec is None


def test_finalize_emits_record_when_one_side_triggered():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(state_before=state, selected_move=(5, 5), ply=44, side_to_move="black", search_score=-0.85, root_top1_share=0.12)
    rec = tracker.finalize_game(
        iteration=170, game_idx=22, game_id="game_022",
        winner="red", starting_player="red", n_moves=65, reason="win",
    )
    assert rec is not None
    assert rec["version"] == 1
    assert rec["iteration"] == 170
    assert rec["game_idx"] == 22
    assert rec["game_id"] == "game_022"
    assert rec["winner"] == "red"
    assert rec["loser"] == "black"
    assert rec["triggered_sides"] == ["black"]
    assert rec["first_trigger_ply"] == 44
    assert rec["first_trigger_side"] == "black"
    assert rec["first_trigger_reason"] == "steady_state"
    black_rec = rec["side_records"]["black"]
    assert black_rec["triggered"] is True
    assert black_rec["classified_in_window_moves"] == 1
    rollup_sum = (
        black_rec["constructive_recovery_moves"]
        + black_rec["defensive_moves"]
        + black_rec["structural_connection_moves"]
        + black_rec["local_drift_moves"]
    )
    assert rollup_sum == black_rec["classified_in_window_moves"]


def test_finalize_loser_is_none_on_draw():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(state_before=state, selected_move=(5, 5), ply=44, side_to_move="black", search_score=-0.85, root_top1_share=0.12)
    rec = tracker.finalize_game(
        iteration=170, game_idx=22, game_id="game_022",
        winner=None, starting_player="red", n_moves=65, reason="board_full",
    )
    assert rec["loser"] is None


def test_finalize_includes_config_block():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(state_before=state, selected_move=(5, 5), ply=44, side_to_move="black", search_score=-0.85, root_top1_share=0.12)
    rec = tracker.finalize_game(
        iteration=170, game_idx=22, game_id="game_022",
        winner="red", starting_player="red", n_moves=65, reason="win",
    )
    cfg = rec["config"]
    assert cfg["collapse_value_threshold"] == -0.75
    assert cfg["classify_defense"] is True


def test_finalize_sampled_moves_metadata():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(max_sampled_moves_per_side=2),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    for ply, mv in [(44, (5, 5)), (46, (6, 6)), (48, (7, 7)), (50, (8, 8))]:
        tracker.observe_move(state_before=state, selected_move=mv, ply=ply, side_to_move="black", search_score=-0.85, root_top1_share=0.12)
    rec = tracker.finalize_game(
        iteration=170, game_idx=22, game_id="game_022",
        winner="red", starting_player="red", n_moves=65, reason="win",
    )
    black_rec = rec["side_records"]["black"]
    assert black_rec["sampled_moves_count"] == 2
    assert black_rec["sampled_moves_cap"] == 2
    assert black_rec["sampled_moves_dropped"] == 2


def test_observe_move_disabled_via_config_is_no_op():
    """If config.enabled is False the tracker is not constructed by self_play
    in the first place. But ensure the tracker itself also no-ops if invoked
    despite enabled=False, so an integration bug doesn't silently corrupt state."""
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(enabled=False),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(state_before=state, selected_move=(5, 5), ply=44, side_to_move="black", search_score=-0.85, root_top1_share=0.12)
    snap = tracker.side_snapshot("black")
    assert snap["triggered"] is False
    rec = tracker.finalize_game(
        iteration=170, game_idx=0, game_id="game_000",
        winner="red", starting_player="red", n_moves=65, reason="win",
    )
    assert rec is None


from scripts.GPU.alphazero.recovery_retargeting_diagnostics import (
    aggregate_recovery_retargeting_records,
    aggregate_recovery_retargeting_with_side_split,
    _side_bucket_for_record,
)
from scripts.GPU.alphazero.recovery_retargeting_diagnostics import PRIMARY_CLASSES


def _record(side="black", classified=10, classes=None, in_window=10, triggered=8, severe=4, very_diffuse=6):
    classes = classes or {"redundant_local_reinforcement": classified}
    counts = {c: 0 for c in PRIMARY_CLASSES}
    counts.update(classes)
    other_side = "red" if side == "black" else "black"
    return {
        "version": 1,
        "iteration": 170, "game_idx": 0, "game_id": "game_000",
        "winner": "red" if side == "black" else "black",
        "loser": side,
        "triggered_sides": [side],
        "side_records": {
            other_side: {"triggered": False, "classifier_error_count": 0},
            side: {
                "triggered": True,
                "in_window_own_moves": in_window,
                "triggered_own_moves": triggered,
                "non_triggered_in_window_moves": in_window - triggered,
                "missing_signal_moves": 0,
                "severe_collapse_moves": severe,
                "very_diffuse_moves": very_diffuse,
                "trigger_reason_counts": {"delta_precursor": 1, "steady_state": triggered - 1, "both": 0},
                "classified_in_window_moves": classified,
                "selected_class_counts": counts,
                "constructive_recovery_moves": counts.get("reduces_own_goal_distance", 0) + counts.get("starts_or_extends_alternate_component", 0),
                "defensive_moves": counts.get("blocks_opponent_closeout", 0),
                "structural_connection_moves": counts.get("connects_to_existing_component", 0) + counts.get("improves_own_largest_component", 0),
                "local_drift_moves": counts.get("redundant_local_reinforcement", 0) + counts.get("off_plan_or_unclear", 0),
                "classifier_error_count": 0,
            },
        },
        "classifier_error_count": 0,
        "config": {
            "collapse_value_threshold": -0.75,
            "severe_collapse_value_threshold": -0.90,
            "diffuse_root_top1_threshold": 0.20,
            "very_diffuse_root_top1_threshold": 0.15,
            "delta_threshold": 0.50,
            "delta_max_current_score": -0.30,
            "alternate_component_min_size": 4,
            "classify_defense": True,
        },
    }


def test_aggregator_sums_counts_and_recomputes_rates():
    recs = [_record(), _record()]
    s = aggregate_recovery_retargeting_records(recs, games_total=100)
    assert s["version"] == 1
    assert s["games_total"] == 100
    assert s["games_triggered"] == 2
    assert s["triggered_own_moves_total"] == 16
    assert s["in_window_own_moves_total"] == 20
    assert s["selected_class_counts_total"]["redundant_local_reinforcement"] == 20
    assert s["local_drift_rate"] == 1.0


def test_aggregator_returns_empty_summary_when_no_records():
    s = aggregate_recovery_retargeting_records([], games_total=100)
    assert s["games_total"] == 100
    assert s["games_triggered"] == 0
    assert s["trigger_rate"] == 0.0


def test_aggregator_empty_records_emits_enabled_summary_with_zero_trigger_rate():
    s = aggregate_recovery_retargeting_records([], games_total=5)
    assert s["version"] == 1
    assert s["enabled"] is True
    assert s["games_total"] == 5
    assert s["games_triggered"] == 0
    assert s["trigger_rate"] == 0.0
    assert s["in_window_own_moves_total"] == 0
    assert s["classified_in_window_moves_total"] == 0
    assert s["schema_integrity"]["classifier_error_count_total"] == 0


def test_aggregator_skips_unknown_version():
    rec = _record()
    rec["version"] = 99
    s = aggregate_recovery_retargeting_records([_record(), rec], games_total=100)
    assert s["games_triggered"] == 1
    assert s["schema_integrity"]["skipped_unknown_version_count"] == 1


def test_aggregator_skips_config_mismatch():
    a = _record()
    b = _record()
    b["config"] = dict(b["config"])
    b["config"]["collapse_value_threshold"] = -0.50
    s = aggregate_recovery_retargeting_records([a, b], games_total=100)
    assert s["games_triggered"] == 1
    assert s["schema_integrity"]["skipped_config_mismatch_count"] == 1


def test_side_bucket_eventual_loser():
    rec = {"winner": "red", "loser": "black"}
    assert _side_bucket_for_record(rec, "black") == "eventual_loser"


def test_side_bucket_eventual_winner():
    rec = {"winner": "red", "loser": "black"}
    assert _side_bucket_for_record(rec, "red") == "eventual_winner"


def test_side_bucket_state_cap_or_draw():
    rec = {"winner": None, "loser": None}
    assert _side_bucket_for_record(rec, "red") == "state_cap_or_draw"
    assert _side_bucket_for_record(rec, "black") == "state_cap_or_draw"


def _record_two_sides(
    *,
    winner="red",
    loser_classes=None,
    winner_classes=None,
    loser_in_window=20, loser_triggered=10,
    winner_in_window=10, winner_triggered=4,
    loser_score_mean=-0.90, winner_score_mean=-0.80,
):
    """Record where both sides triggered. Used for split/filter tests."""
    loser = "black" if winner == "red" else "red"
    loser_classes = loser_classes or {"redundant_local_reinforcement": 10}
    winner_classes = winner_classes or {"redundant_local_reinforcement": 4}

    def _side(triggered, in_window, classes, score_mean, top1_mean=0.15, mins=None, maxs=None):
        counts = {c: 0 for c in PRIMARY_CLASSES}
        counts.update(classes)
        classified = sum(counts.values())
        return {
            "triggered": True,
            "in_window_own_moves": in_window,
            "triggered_own_moves": triggered,
            "non_triggered_in_window_moves": in_window - triggered,
            "missing_signal_moves": 0,
            "severe_collapse_moves": triggered // 2,
            "very_diffuse_moves": triggered,
            "trigger_reason_counts": {"delta_precursor": 1, "steady_state": triggered - 1, "both": 0},
            "classified_in_window_moves": classified,
            "selected_class_counts": counts,
            "mean_search_score_triggered_plies": score_mean,
            "min_search_score_triggered_plies": mins if mins is not None else score_mean - 0.05,
            "max_search_score_triggered_plies": maxs if maxs is not None else score_mean + 0.05,
            "mean_root_top1_share_triggered_plies": top1_mean,
            "classifier_error_count": 0,
        }

    return {
        "version": 1,
        "iteration": 170, "game_idx": 0, "game_id": "game_000",
        "winner": winner, "loser": loser,
        "triggered_sides": ["red", "black"],
        "side_records": {
            loser: _side(loser_triggered, loser_in_window, loser_classes, loser_score_mean),
            winner: _side(winner_triggered, winner_in_window, winner_classes, winner_score_mean),
        },
        "classifier_error_count": 0,
        "config": {
            "collapse_value_threshold": -0.75,
            "severe_collapse_value_threshold": -0.90,
            "diffuse_root_top1_threshold": 0.20,
            "very_diffuse_root_top1_threshold": 0.15,
            "delta_threshold": 0.50,
            "delta_max_current_score": -0.30,
            "alternate_component_min_size": 4,
            "classify_defense": True,
        },
    }


def _record_state_cap(triggered_sides=("red", "black"), classes=None, in_window=25, triggered=5, score_mean=-0.95):
    """Record where the game ended state-cap/draw (winner=None)."""
    classes = classes or {"redundant_local_reinforcement": 5}
    counts = {c: 0 for c in PRIMARY_CLASSES}
    counts.update(classes)
    classified = sum(counts.values())

    def _side():
        return {
            "triggered": True,
            "in_window_own_moves": in_window,
            "triggered_own_moves": triggered,
            "non_triggered_in_window_moves": in_window - triggered,
            "missing_signal_moves": 0,
            "severe_collapse_moves": triggered // 2,
            "very_diffuse_moves": triggered,
            "trigger_reason_counts": {"delta_precursor": 0, "steady_state": triggered, "both": 0},
            "classified_in_window_moves": classified,
            "selected_class_counts": counts,
            "mean_search_score_triggered_plies": score_mean,
            "min_search_score_triggered_plies": score_mean - 0.05,
            "max_search_score_triggered_plies": score_mean + 0.05,
            "mean_root_top1_share_triggered_plies": 0.15,
            "classifier_error_count": 0,
        }

    side_records = {}
    for s in ("red", "black"):
        if s in triggered_sides:
            side_records[s] = _side()
        else:
            side_records[s] = {"triggered": False, "classifier_error_count": 0}

    return {
        "version": 1,
        "iteration": 170, "game_idx": 0, "game_id": "game_000",
        "winner": None, "loser": None,
        "triggered_sides": list(triggered_sides),
        "side_records": side_records,
        "classifier_error_count": 0,
        "config": {
            "collapse_value_threshold": -0.75,
            "severe_collapse_value_threshold": -0.90,
            "diffuse_root_top1_threshold": 0.20,
            "very_diffuse_root_top1_threshold": 0.15,
            "delta_threshold": 0.50,
            "delta_max_current_score": -0.30,
            "alternate_component_min_size": 4,
            "classify_defense": True,
        },
    }


def test_side_split_rollup_recombines_to_pooled_counts():
    """Spec §6 Test 4. Sum of counts across the three side-split buckets
    must equal the corresponding totals from the pooled aggregator."""
    recs = [
        _record(side="black", in_window=10, triggered=8, classified=10),
        _record_two_sides(loser_in_window=20, loser_triggered=10, winner_in_window=8, winner_triggered=3),
        _record_state_cap(in_window=25, triggered=5),
    ]
    pooled = aggregate_recovery_retargeting_records(recs, games_total=100)
    split  = aggregate_recovery_retargeting_with_side_split(recs, games_total=100)

    split_in_window = (
        split["eventual_loser"]["in_window_own_moves_total"]
        + split["eventual_winner"]["in_window_own_moves_total"]
        + split["state_cap_or_draw"]["in_window_own_moves_total"]
    )
    split_classified = (
        split["eventual_loser"]["classified_in_window_moves_total"]
        + split["eventual_winner"]["classified_in_window_moves_total"]
        + split["state_cap_or_draw"]["classified_in_window_moves_total"]
    )
    assert split_in_window == pooled["in_window_own_moves_total"]
    assert split_classified == pooled["classified_in_window_moves_total"]


def test_bucket_score_aggregation_uses_pooled_weighted_mean():
    """Spec §6 Test 9. Two records, both with winner=red+loser=black, so both
    black sides land in eventual_loser. Exercises multi-side aggregation
    *within* the same bucket — the actual invariant the weighting rule enforces."""
    rec_long = _record(
        side="black", in_window=120, triggered=120, classified=120,
        classes={"redundant_local_reinforcement": 120},
    )
    rec_long["side_records"]["black"]["mean_search_score_triggered_plies"]    = -0.95
    rec_long["side_records"]["black"]["min_search_score_triggered_plies"]     = -0.97
    rec_long["side_records"]["black"]["max_search_score_triggered_plies"]     = -0.93
    rec_long["side_records"]["black"]["mean_root_top1_share_triggered_plies"] = 0.10

    rec_short = _record(
        side="black", in_window=3, triggered=3, classified=3,
        classes={"redundant_local_reinforcement": 3},
    )
    rec_short["side_records"]["black"]["mean_search_score_triggered_plies"]    = -0.40
    rec_short["side_records"]["black"]["min_search_score_triggered_plies"]     = -0.42
    rec_short["side_records"]["black"]["max_search_score_triggered_plies"]     = -0.38
    rec_short["side_records"]["black"]["mean_root_top1_share_triggered_plies"] = 0.30
    rec_short["game_idx"] = 1
    rec_short["game_id"]  = "game_001"

    split = aggregate_recovery_retargeting_with_side_split(
        [rec_long, rec_short], games_total=2,
    )
    bucket = split["eventual_loser"]
    assert bucket["sides"] == 2
    expected_score = (-0.95 * 120 + -0.40 * 3) / 123
    expected_top1  = (0.10 * 120 + 0.30 * 3) / 123
    assert abs(bucket["mean_search_score_triggered_plies"] - round(expected_score, 3)) < 1e-9
    assert abs(bucket["mean_root_top1_share_triggered_plies"] - round(expected_top1,  3)) < 1e-9
    assert bucket["min_search_score_triggered_plies"] == -0.97
    assert bucket["max_search_score_triggered_plies"] == -0.38


def test_side_view_for_record_side_matches_iter_triggered_side_views():
    """Helper invariant: _side_view_for_record_side and _iter_triggered_side_views
    must produce equivalent views for the same triggered sides."""
    from scripts.GPU.alphazero.recovery_retargeting_diagnostics import (
        _side_view_for_record_side, _iter_triggered_side_views,
    )
    rec = _record_two_sides(winner="red", loser_in_window=20, loser_triggered=10,
                            winner_in_window=8, winner_triggered=4)
    iterator_views = list(_iter_triggered_side_views([rec]))
    helper_views = [
        _side_view_for_record_side(rec, "red"),
        _side_view_for_record_side(rec, "black"),
    ]
    helper_views = [v for v in helper_views if v is not None]
    iterator_views.sort(key=lambda v: v["side_bucket"])
    helper_views.sort(key=lambda v: v["side_bucket"])
    assert iterator_views == helper_views


def test_side_view_derives_rates_when_missing():
    """_side_view_for_record_side derives the four bucket rates from
    selected_class_counts when the underlying side record omits them."""
    from scripts.GPU.alphazero.recovery_retargeting_diagnostics import (
        _side_view_for_record_side,
    )
    rec = _record(side="black", in_window=20, triggered=10, classified=10,
                  classes={"connects_to_existing_component": 4,
                           "redundant_local_reinforcement": 6})
    for k in ("constructive_recovery_rate", "defensive_rate",
              "structural_connection_rate", "local_drift_rate"):
        rec["side_records"]["black"].pop(k, None)
    view = _side_view_for_record_side(rec, "black")
    assert view is not None
    assert view["structural_connection_rate"] == 0.4
    assert view["local_drift_rate"]           == 0.6
    assert view["constructive_recovery_rate"] == 0.0
    assert view["defensive_rate"]             == 0.0


def test_side_split_schema_for_empty_records():
    """Empty records list returns a fully-shaped summary with all three
    buckets present (zero counts, None scores)."""
    out = aggregate_recovery_retargeting_with_side_split([], games_total=10)
    assert out["games_total"] == 10
    assert out["games_triggered"] == 0
    for bucket in ("eventual_loser", "eventual_winner", "state_cap_or_draw"):
        b = out[bucket]
        assert b["sides"] == 0
        assert b["in_window_own_moves_total"] == 0
        assert b["mean_search_score_triggered_plies"] is None
        assert b["constructive_recovery_rate"] == 0.0
    assert out["schema_integrity"]["classifier_error_count_total"] == 0


def test_bucket_score_aggregation_ignores_missing_mean_values():
    """A side view with triggered_own_moves > 0 but None mean_search_score
    must NOT contribute weight to the score denominator."""
    rec_with = _record(
        side="black", in_window=10, triggered=10, classified=10,
        classes={"redundant_local_reinforcement": 10},
    )
    rec_with["side_records"]["black"]["mean_search_score_triggered_plies"]    = -0.90
    rec_with["side_records"]["black"]["mean_root_top1_share_triggered_plies"] = 0.10

    rec_without = _record(
        side="black", in_window=10, triggered=10, classified=10,
        classes={"redundant_local_reinforcement": 10},
    )
    rec_without["game_idx"] = 1
    rec_without["game_id"]  = "game_001"
    rec_without["side_records"]["black"]["mean_search_score_triggered_plies"]    = None
    rec_without["side_records"]["black"]["mean_root_top1_share_triggered_plies"] = None
    rec_without["side_records"]["black"]["min_search_score_triggered_plies"]     = None
    rec_without["side_records"]["black"]["max_search_score_triggered_plies"]     = None

    split = aggregate_recovery_retargeting_with_side_split(
        [rec_with, rec_without], games_total=2,
    )
    bucket = split["eventual_loser"]
    assert bucket["sides"] == 2
    assert bucket["mean_search_score_triggered_plies"] == -0.9
    assert bucket["mean_root_top1_share_triggered_plies"] == 0.1


def test_split_schema_integrity_matches_existing_pooled_behavior():
    """Parity test: _filter_and_canonicalize must reproduce the existing
    aggregator's accepted-record / skipped-counts behavior on the same inputs."""
    a = _record(side="black")
    b = _record(side="black")
    b["version"] = 99
    c = _record(side="black")
    c["config"] = dict(c["config"])
    c["config"]["collapse_value_threshold"] = -0.50

    pooled = aggregate_recovery_retargeting_records([a, b, c], games_total=10)
    split  = aggregate_recovery_retargeting_with_side_split([a, b, c], games_total=10)

    assert pooled["games_triggered"] == split["games_triggered"]
    p_si = pooled["schema_integrity"]
    s_si = split["schema_integrity"]
    assert p_si["skipped_unknown_version_count"] == s_si["skipped_unknown_version_count"]
    assert p_si["skipped_config_mismatch_count"] == s_si["skipped_config_mismatch_count"]
    assert p_si["classifier_error_count_total"]  == s_si["classifier_error_count_total"]
