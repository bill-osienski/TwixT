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


import math


def test_sample_boost_1_is_pure_uniform():
    """ANCHOR (Spec 2 §11.3): boost=1.0 short-circuits — eligibility set
    is not consulted, cap is not consulted, behavior identical to pre-Spec-2."""
    buf = ReplayBuffer(max_size=100)
    buf.add_positions([_pos(True) for _ in range(50)])
    buf.add_positions([_pos(False) for _ in range(50)])
    import random
    rng = random.Random(42)
    eligible_drawn = 0
    total_drawn = 0
    for _ in range(20):
        batch = buf.sample(
            batch_size=10, rng=rng, active_size=24,
            conversion_sample_boost=1.0,
            conversion_max_batch_fraction=0.15,
        )
        eligible_drawn += sum(1 for p in batch if p.conversion is not None)
        total_drawn += len(batch)
    natural_rate = eligible_drawn / total_drawn
    # ~0.5 with some noise; boost=1.0 must short-circuit.
    assert 0.35 < natural_rate < 0.65, (
        f"natural_rate={natural_rate}; boost=1.0 should produce ~0.5"
    )


def test_sample_boost_2_produces_at_most_cap_fraction():
    buf = ReplayBuffer(max_size=200)
    buf.add_positions([_pos(True) for _ in range(50)])
    buf.add_positions([_pos(False) for _ in range(150)])
    import random
    rng = random.Random(42)
    for _ in range(20):
        batch = buf.sample(
            batch_size=20, rng=rng, active_size=24,
            conversion_sample_boost=10.0,
            conversion_max_batch_fraction=0.15,    # cap 15% = floor(20*0.15)=3
        )
        eligible = sum(1 for p in batch if p.conversion is not None)
        assert eligible <= 3, f"eligible={eligible} exceeds cap of 3"


def test_sample_boost_uses_ceil_rounding_for_target():
    """Spec 2 §7.3: ceil rounding so rare eligibles aren't rounded to zero.
    Fixture: batch=16, natural expectation < 1, cap allows 1+, boost > 1."""
    buf = ReplayBuffer(max_size=1000)
    buf.add_positions([_pos(True) for _ in range(10)])
    buf.add_positions([_pos(False) for _ in range(990)])
    # natural_expectation = 16 * (10 / 1000) = 0.16
    # ceil(0.16 * 2.0) = 1; cap_count = floor(16 * 0.15) = 2; min(1, 2, 10, 16) = 1
    import random
    rng = random.Random(42)
    eligible_counts = []
    for _ in range(50):
        batch = buf.sample(
            batch_size=16, rng=rng, active_size=24,
            conversion_sample_boost=2.0,
            conversion_max_batch_fraction=0.15,
        )
        eligible_counts.append(sum(1 for p in batch if p.conversion is not None))
    # All batches should have exactly 1 eligible (deterministic by formula).
    assert all(c == 1 for c in eligible_counts), (
        f"eligible counts: {set(eligible_counts)} — expected all 1 with ceil rounding"
    )


def test_sample_falls_back_to_uniform_when_eligible_pool_empty():
    buf = ReplayBuffer(max_size=100)
    buf.add_positions([_pos(False) for _ in range(50)])    # zero eligible
    import random
    rng = random.Random(42)
    batch = buf.sample(
        batch_size=10, rng=rng, active_size=24,
        conversion_sample_boost=2.0,
        conversion_max_batch_fraction=0.15,
    )
    assert len(batch) == 10
    assert all(p.conversion is None for p in batch)
    # Stats should record boost_was_inactive
    stats = buf.last_sample_stats
    assert stats.boost_was_inactive is True


def test_sample_active_size_intersects_eligibility():
    buf = ReplayBuffer(max_size=200)
    # 50 eligible at size 24, 50 eligible at size 12, 50 non-eligible at size 24
    buf.add_positions([_pos(True, active_size=24) for _ in range(50)])
    buf.add_positions([_pos(True, active_size=12) for _ in range(50)])
    buf.add_positions([_pos(False, active_size=24) for _ in range(50)])
    import random
    rng = random.Random(42)
    batch = buf.sample(
        batch_size=20, rng=rng, active_size=12,
        conversion_sample_boost=10.0,
        conversion_max_batch_fraction=0.5,
    )
    # Every position drawn must have active_size=12.
    assert all(p.active_size == 12 for p in batch)


def test_sample_no_duplicate_positions_with_two_strata():
    """ANCHOR (Spec 2 §11.3): no replacement across strata."""
    buf = ReplayBuffer(max_size=20)
    buf.add_positions([_pos(True) for _ in range(5)])      # 5 eligible
    buf.add_positions([_pos(False) for _ in range(15)])    # 15 non-eligible
    import random
    rng = random.Random(42)
    batch = buf.sample(
        batch_size=20, rng=rng, active_size=24,    # batch == buffer
        conversion_sample_boost=10.0,
        conversion_max_batch_fraction=0.5,         # cap 10
    )
    # batch should be all 20 buffer positions, no duplicates.
    ids = [id(p) for p in batch]
    assert len(set(ids)) == len(ids), "duplicate positions in batch"


def test_sample_stats_match_drawn_count():
    """Sampler-side count tracking: last_sample_stats.eligible_drawn matches
    actual eligible positions in returned batch."""
    buf = ReplayBuffer(max_size=100)
    buf.add_positions([_pos(True) for _ in range(20)])
    buf.add_positions([_pos(False) for _ in range(80)])
    import random
    rng = random.Random(42)
    batch = buf.sample(
        batch_size=20, rng=rng, active_size=24,
        conversion_sample_boost=2.0,
        conversion_max_batch_fraction=0.5,
    )
    drawn = sum(1 for p in batch if p.conversion is not None)
    assert buf.last_sample_stats.eligible_drawn == drawn
