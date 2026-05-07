# tests/test_replay_buffer_conversion.py
"""ReplayBuffer eligibility tracking + stratified sampling (Spec 2 §7).
This file holds tests for Tasks 11 and 12. Task 11 covers eligibility
index pool only; sampling tests will be added in Task 12.
"""
import numpy as np
from scripts.GPU.alphazero.trainer import ReplayBuffer
from scripts.GPU.alphazero.self_play import PositionRecord


def _pos(eligible: bool, active_size: int = 24):
    conv = (
        {
            "version": 1,
            "endpoint_completion_moves": [[0, 0]],
            "distance_reducing_moves": [],
        }
        if eligible
        else None
    )
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move="red",
        legal_moves=[(0, 0), (1, 1)],
        visit_counts=[1, 1],
        outcome=1.0,
        active_size=active_size,
        ply=0,
        game_n_moves=10,
        conversion=conv,
    )


def test_replay_buffer_eligible_index_tracks_evictions():
    """Adding eligible positions, then enough non-eligible to evict them via
    ring-overwrite, should remove them from the eligible pool."""
    buf = ReplayBuffer(max_size=4)
    eligible = [_pos(True) for _ in range(4)]
    buf.add_positions(eligible)
    assert buf.count_eligible() == 4

    # Add 4 non-eligible — ring buffer overwrites all 4 eligible slots.
    buf.add_positions([_pos(False) for _ in range(4)])
    assert buf.count_eligible() == 0


def test_replay_buffer_eligible_index_swap_delete_correctness():
    """Add 5 eligibles; manually overwrite slot 2 with a non-eligible
    position to mimic ring-buffer overwrite at that index. Remaining
    indices must still resolve to eligible positions."""
    buf = ReplayBuffer(max_size=10)
    buf.add_positions([_pos(True) for _ in range(5)])
    assert buf.count_eligible() == 5

    # Mimic ring-buffer overwrite at idx=2: replace position and update
    # the eligibility index manually via the buffer's internal helper.
    buf.buffer[2] = _pos(False)
    buf._update_eligible_index(2, buf.buffer[2])
    assert buf.count_eligible() == 4
    # The remaining 4 indices in the eligible pool must point to eligible positions.
    for idx in buf._eligible_idxs:
        assert buf.buffer[idx].conversion is not None


def test_replay_buffer_count_eligible_with_active_size_filter():
    """count_eligible(active_size=N) filters by active_size."""
    buf = ReplayBuffer(max_size=20)
    buf.add_positions([_pos(True, active_size=12) for _ in range(3)])
    buf.add_positions([_pos(True, active_size=24) for _ in range(5)])
    buf.add_positions([_pos(False, active_size=24) for _ in range(2)])
    assert buf.count_eligible() == 8
    assert buf.count_eligible(active_size=12) == 3
    assert buf.count_eligible(active_size=24) == 5
    assert buf.count_eligible(active_size=8) == 0    # none at this size
