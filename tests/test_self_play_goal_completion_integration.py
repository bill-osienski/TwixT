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


# ---------------------------------------------------------------------------
# Spec 2 §5.5 — conversion attach point tests
# ---------------------------------------------------------------------------

def _stub_gc_state(target_ply: int):
    """Build a stub for compute_goal_completion_state that returns a
    synthetic eligible gc_state when ply >= target_ply.

    - On the cheap call (enumerate_moves=False): returns a dict with
      total_goal_distance=2 so needs_conversion_full fires.
    - On the full call (enumerate_moves=True): returns the full dict with
      endpoint_completion_moves / distance_reducing_moves populated from
      the actual legal_moves at that ply.

    Returns (stub_fn, real_fn).
    """
    from scripts.GPU.alphazero import connectivity_diagnostics as _cd
    real_fn = _cd.compute_goal_completion_state

    def _stub(state, player, *args, **kwargs):
        em = kwargs.get("enumerate_moves", False)
        if state.ply >= target_ply:
            legal = state.legal_moves()
            if len(legal) >= 2:
                if em:
                    return {
                        "total_goal_distance": 2,
                        "largest_component_size": 12,
                        "endpoint_completion_moves": [legal[0]],
                        "distance_reducing_moves":   [legal[1]],
                        "category": "two_endpoint_closeout_2ply",
                        "max_depth": 3,
                        "endpoint_distances": {"top": 0, "bottom": 1},
                        "component_pegs": [],
                    }
                else:
                    # Cheap path: just return distance so needs_conversion_full fires.
                    return {
                        "total_goal_distance": 2,
                        "largest_component_size": 12,
                        "endpoint_completion_moves": None,
                        "distance_reducing_moves":   None,
                        "category": "two_endpoint_closeout_2ply",
                        "max_depth": 3,
                        "endpoint_distances": {"top": 0, "bottom": 1},
                        "component_pegs": [],
                    }
        return real_fn(state, player, *args, **kwargs)

    return _stub, real_fn


def test_play_game_attaches_conversion_when_enabled_and_eligible(monkeypatch):
    """Spec 2 §5.5: PositionRecord.conversion populated on closeout plies.
    Deterministic via stubbed gc_state — no skip."""
    from scripts.GPU.alphazero.self_play import play_game
    from scripts.GPU.alphazero import connectivity_diagnostics as _cd
    stub, _ = _stub_gc_state(target_ply=4)
    monkeypatch.setattr(_cd, "compute_goal_completion_state", stub)

    record = play_game(
        evaluator=_make_evaluator(7),
        mcts_config=_short_cfg(),
        rng=_rng.Random(7),
        max_moves=20, active_size=8,
        conversion_policy_loss_enabled=True,
        conversion_max_total_goal_distance=2,
        goal_completion_record_enabled=False,
        goal_completion_emit_enabled=False,
    )
    closeout = [p for p in record.positions if p.conversion is not None]
    assert len(closeout) >= 1, (
        "Stubbed eligible gc_state did not produce conversion metadata — "
        "attach point in play_game is broken"
    )
    cp = closeout[0]
    assert cp.conversion["version"] == 1
    assert cp.conversion["total_goal_distance"] == 2
    assert cp.conversion["largest_component_size"] == 12
    assert (cp.conversion["endpoint_completion_moves"]
            or cp.conversion["distance_reducing_moves"])


def test_play_game_no_conversion_metadata_when_loss_disabled():
    """ANCHOR (Spec 2 §11.3): default config produces no conversion metadata.
    No skip — does not depend on a closeout state occurring."""
    from scripts.GPU.alphazero.self_play import play_game
    record = play_game(
        evaluator=_make_evaluator(7),
        mcts_config=_short_cfg(),
        rng=_rng.Random(7),
        max_moves=30, active_size=8,
        conversion_policy_loss_enabled=False,
        goal_completion_record_enabled=False,
        goal_completion_emit_enabled=False,
    )
    assert len(record.positions) > 0
    assert all(p.conversion is None for p in record.positions)


def test_position_record_conversion_pre_move_invariant(monkeypatch):
    """ANCHOR (Spec 2 §5.4 / §11.3): conversion describes the pre-move state.
    Strategy: capture the legal_moves the stub saw at the target ply, then
    assert those exact moves appear in the persisted conversion dict.
    If attach happened post-apply_move, one move would be consumed."""
    from scripts.GPU.alphazero.self_play import play_game
    from scripts.GPU.alphazero import connectivity_diagnostics as _cd

    target_ply = 4
    stub_seen = {}
    real_fn = _cd.compute_goal_completion_state

    def _stub(state, player, *args, **kwargs):
        em = kwargs.get("enumerate_moves", False)
        if state.ply == target_ply:
            legal = state.legal_moves()
            if len(legal) >= 2:
                if em and "captured" not in stub_seen:
                    stub_seen["captured"] = {
                        "ply": state.ply,
                        "legal_first_two": [tuple(legal[0]), tuple(legal[1])],
                    }
                    return {
                        "total_goal_distance": 2,
                        "largest_component_size": 12,
                        "endpoint_completion_moves": [legal[0]],
                        "distance_reducing_moves":   [legal[1]],
                        "category": "two_endpoint_closeout_2ply",
                        "max_depth": 3,
                        "endpoint_distances": {"top": 0, "bottom": 1},
                        "component_pegs": [],
                    }
                elif not em:
                    # Cheap path: return low distance so needs_conversion_full fires.
                    return {
                        "total_goal_distance": 2,
                        "largest_component_size": 12,
                        "endpoint_completion_moves": None,
                        "distance_reducing_moves":   None,
                        "category": "two_endpoint_closeout_2ply",
                        "max_depth": 3,
                        "endpoint_distances": {"top": 0, "bottom": 1},
                        "component_pegs": [],
                    }
        return real_fn(state, player, *args, **kwargs)

    monkeypatch.setattr(_cd, "compute_goal_completion_state", _stub)

    record = play_game(
        evaluator=_make_evaluator(7),
        mcts_config=_short_cfg(),
        rng=_rng.Random(7),
        max_moves=20, active_size=8,
        conversion_policy_loss_enabled=True,
        conversion_max_total_goal_distance=2,
        goal_completion_record_enabled=False,
        goal_completion_emit_enabled=False,
    )

    assert "captured" in stub_seen, (
        "Stub never called with enumerate_moves=True on target ply"
    )

    matching = [p for p in record.positions if p.conversion is not None
                and p.conversion["total_goal_distance"] == 2]
    assert len(matching) >= 1, (
        "Stubbed gc_state did not attach conversion metadata at target ply"
    )
    cp = matching[0]

    persisted_completion = {tuple(m) for m in cp.conversion["endpoint_completion_moves"]}
    persisted_reducing   = {tuple(m) for m in cp.conversion["distance_reducing_moves"]}
    seen_first  = stub_seen["captured"]["legal_first_two"][0]
    seen_second = stub_seen["captured"]["legal_first_two"][1]
    assert seen_first in persisted_completion, (
        f"completion_moves on disk={persisted_completion}, but stub saw "
        f"{seen_first} as first legal move at PRE-move ply {target_ply}. "
        "Conversion metadata was not captured PRE-apply_move."
    )
    assert seen_second in persisted_reducing, (
        f"distance_reducing_moves on disk={persisted_reducing}, but stub saw "
        f"{seen_second} as second legal move at PRE-move ply {target_ply}. "
        "Conversion metadata was not captured PRE-apply_move."
    )


def test_play_game_conversion_enabled_computes_full_state_when_emit_disabled(monkeypatch):
    """Spec 2 §3 cost-path: conversion forces full BFS even with Spec 1.5
    paths off. Deterministic — no skip."""
    from scripts.GPU.alphazero.self_play import play_game
    from scripts.GPU.alphazero import connectivity_diagnostics as _cd
    stub, _ = _stub_gc_state(target_ply=4)
    monkeypatch.setattr(_cd, "compute_goal_completion_state", stub)

    record = play_game(
        evaluator=_make_evaluator(7),
        mcts_config=_short_cfg(),
        rng=_rng.Random(7),
        max_moves=20, active_size=8,
        conversion_policy_loss_enabled=True,
        conversion_max_total_goal_distance=2,
        goal_completion_emit_enabled=False,
        goal_completion_record_enabled=False,
    )
    closeout = [p for p in record.positions if p.conversion is not None]
    assert len(closeout) >= 1, (
        "Conversion attach did not fire when Spec 1.5 emit/record paths were off"
    )
