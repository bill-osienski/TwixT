"""End-to-end self-play tracker integration (spec §8)."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.self_play import play_game, MCTSConfig
import random as _rng


def _make_evaluator(seed=42):
    """Construct a real LocalGPUEvaluator with a tiny seeded network.
    Mirrors the pattern in tests/test_self_play_closeout_diagnostics.py."""
    import numpy as np
    import mlx.core as mx
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    np.random.seed(seed)
    mx.random.seed(seed)
    net = create_network(hidden=32, n_blocks=2)
    return LocalGPUEvaluator(net)


def _short_cfg():
    # Tiny MCTS, small board, deterministic-ish: enough to produce a record.
    return MCTSConfig(n_simulations=8, c_puct=1.0)


def test_play_game_attaches_goal_completion_record_when_enabled():
    rec = play_game(
        evaluator=_make_evaluator(7),
        mcts_config=_short_cfg(),
        rng=_rng.Random(7),
        max_moves=30,
        active_size=8,
        goal_completion_record_enabled=True,
        goal_completion_emit_enabled=False,  # ensure tracker stands alone
    )
    assert rec.goal_completion_record is not None
    assert rec.goal_completion_record["version"] == 1
    assert rec.goal_completion_record["outcome_class"] in (1, 2, 3)
    # Class 1/2 records have primary_class_counts; Class 3 minimal record may not.
    if rec.goal_completion_record["outcome_class"] == 1:
        assert "primary_class_counts" in rec.goal_completion_record


def test_play_game_no_record_when_disabled():
    rec = play_game(
        evaluator=_make_evaluator(7),
        mcts_config=_short_cfg(),
        rng=_rng.Random(7),
        max_moves=30,
        active_size=8,
        goal_completion_record_enabled=False,
        goal_completion_emit_enabled=False,
    )
    assert rec.goal_completion_record is None


def test_play_game_invariant_violated_raises():
    """detection_threshold > emit_threshold must raise."""
    with pytest.raises(ValueError, match="detection_threshold"):
        play_game(
            evaluator=_make_evaluator(7),
            mcts_config=_short_cfg(),
            rng=_rng.Random(7),
            max_moves=10,
            active_size=8,
            goal_completion_detection_threshold=4,
            goal_completion_emit_threshold=3,
        )


def test_play_game_record_present_when_emit_disabled_record_enabled():
    """Compact record is independent of Phase 3 emit gating."""
    rec = play_game(
        evaluator=_make_evaluator(11),
        mcts_config=_short_cfg(),
        rng=_rng.Random(11),
        max_moves=30,
        active_size=8,
        goal_completion_record_enabled=True,
        goal_completion_emit_enabled=False,
    )
    assert rec.goal_completion_record is not None
    # Phase 3 fields should be empty / None.
    assert rec.goal_completion_diagnostics_meta is None
    assert rec.goal_completion_diagnostics == []


def test_play_game_record_iteration_metadata_default_zero():
    """Tracker is constructed inside play_game; iteration is not yet supplied
    here (trainer/saver path overwrites it). For now, the record carries
    iteration=0 / game_idx=game_id."""
    rec = play_game(
        evaluator=_make_evaluator(7),
        mcts_config=_short_cfg(),
        rng=_rng.Random(7),
        max_moves=20,
        active_size=8,
        game_id=5,
        goal_completion_record_enabled=True,
        goal_completion_emit_enabled=False,
    )
    assert rec.goal_completion_record is not None
    assert rec.goal_completion_record["game_idx"] == 5
    # iteration is not known inside play_game; default 0.
    assert rec.goal_completion_record["iteration"] == 0


def test_play_game_upgrades_to_gc_state_full_on_detection_ply(monkeypatch):
    """Caller-side BFS-upgrade integration guard.

    Independent of the tracker's defensive None-handling: this proves that
    on a ply where total_goal_distance <= detection_threshold pre-move,
    the self-play loop actually computes compute_goal_completion_state
    with enumerate_moves=True (so the tracker has gc_state_full available
    to classify, not just gc_state_cheap).
    """
    from scripts.GPU.alphazero import connectivity_diagnostics as _cd

    calls = []
    real_fn = _cd.compute_goal_completion_state

    def _spy(state, player, *args, **kwargs):
        em = kwargs.get("enumerate_moves", True)
        result = real_fn(state, player, *args, **kwargs)
        calls.append({
            "enumerate_moves": em,
            "total": (result or {}).get("total_goal_distance"),
        })
        return result

    monkeypatch.setattr(_cd, "compute_goal_completion_state", _spy)

    rec = play_game(
        evaluator=_make_evaluator(13),
        mcts_config=_short_cfg(),
        rng=_rng.Random(13),
        max_moves=40,
        active_size=8,
        goal_completion_record_enabled=True,
        goal_completion_emit_enabled=False,
        goal_completion_detection_threshold=2,
    )
    upgraded_calls = [c for c in calls if c["enumerate_moves"] is True]
    if rec.goal_completion_record and rec.goal_completion_record.get("detected"):
        assert len(upgraded_calls) >= 1, (
            "Expected at least one compute_goal_completion_state call with "
            "enumerate_moves=True on the detection ply"
        )
    # If the random game never reached dominant-unclosed, the test is vacuous.
