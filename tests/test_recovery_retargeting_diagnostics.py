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
