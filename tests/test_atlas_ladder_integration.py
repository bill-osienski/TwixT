# tests/test_atlas_ladder_integration.py
"""Real Stage 2 producer -> real Stage 3 consumer. No surrogates."""
import json
import random
from pathlib import Path

import pytest

from scripts.GPU.alphazero.build_atlas_corpus import _jsonable
from scripts.GPU.alphazero.generate_atlas_reservoir import (
    game_meta_from_sidecar, generate_block,
)
from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
from scripts.GPU.alphazero.selection_tracer import SelectionTracer
from scripts.GPU.alphazero.warm_prefix_replay import (
    BatchSafeBoundaryObserver, replay_prefix, replay_seed_for,
    run_additive_ladder,
)

from tests.eval_fakes import FakeEvaluator

BASE = 20400000
SIZE = 6
FAKE_PROV = {"git_head": "d" * 40, "worktree_clean": True,
             "checkpoint_path": "fake://evaluator", "checkpoint_sha1": "0" * 40}


def _real_block(tmp_path):
    return generate_block(
        evaluator=FakeEvaluator(value=0.0), base_seed=BASE, start_index=0,
        n_games=2, out_dir=Path(tmp_path), provenance=dict(FAKE_PROV),
        n_simulations=8, max_moves=10, active_size=SIZE,
    )


def test_a_real_sidecar_drives_the_real_replay(tmp_path):
    rows = _real_block(tmp_path / "blk")
    side = json.loads(
        (tmp_path / "blk" / "game_000000.json").read_text())
    meta = game_meta_from_sidecar(side)

    seed = replay_seed_for(meta, base_seed=BASE)      # verifies against sidecar
    assert seed == side["seed"]

    history = [tuple(m) for m in side["move_history"]]
    target_ply = min(2, len(history))
    m = MCTS(FakeEvaluator(value=0.0),
             MCTSConfig(n_simulations=8, eval_batch_size=14,
                        stall_flush_sims=48, pending_virtual_visits=8),
             random.Random(seed))
    pre = replay_prefix(m, meta, history, target_ply=target_ply,
                        active_size=SIZE)
    assert pre.inherited_I >= 0
    assert len(pre.steps) == target_ply


def test_the_replayed_prefix_follows_the_recorded_moves_exactly(tmp_path):
    rows = _real_block(tmp_path / "blk2")
    side = json.loads((tmp_path / "blk2" / "game_000001.json").read_text())
    meta = game_meta_from_sidecar(side)
    history = [tuple(m) for m in side["move_history"]]
    m = MCTS(FakeEvaluator(value=0.0),
             MCTSConfig(n_simulations=8, eval_batch_size=14,
                        stall_flush_sims=48, pending_virtual_visits=8),
             random.Random(replay_seed_for(meta, BASE)))
    pre = replay_prefix(m, meta, history, target_ply=3, active_size=SIZE)
    from scripts.GPU.alphazero.mcts import encode_move
    assert [s.forced_move for s in pre.steps] == [
        encode_move(*history[i]) for i in range(3)]


def test_the_row_artifact_survives_jsonable(tmp_path):
    """Ladder and boundary types must serialize through the SAME converter
    Stage 2 uses. Re-keying the geometry module is forbidden."""
    rows = _real_block(tmp_path / "blk3")
    side = json.loads((tmp_path / "blk3" / "game_000000.json").read_text())
    meta = game_meta_from_sidecar(side)
    history = [tuple(m) for m in side["move_history"]]
    m = MCTS(FakeEvaluator(value=0.0),
             MCTSConfig(n_simulations=1, eval_batch_size=14,
                        stall_flush_sims=48, pending_virtual_visits=8),
             random.Random(replay_seed_for(meta, BASE)))
    pre = replay_prefix(m, meta, history, target_ply=2, active_size=SIZE)
    tracer = SelectionTracer()                 # FRESH, after the prefix
    m._selection_observer = tracer
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I, threshold=4,
                                    leg_B=8, tracer=tracer)
    legs, snaps = run_additive_ladder(m, pre.root, pre.inherited_I, ply=2,
                                      boundary_observer=obs,
                                      target_tracer=tracer,
                                      increments=(8, 8, 8, 8))
    artifact = {
        "game_idx": meta.game_id, "replay_seed": meta.seed,
        "inherited_I": pre.inherited_I,
        "reset_count": pre.reset_count, "reset_rate": pre.reset_rate,
        "last_reset_ply": pre.last_reset_ply,
        "legs": [vars(l) for l in legs],
        "boundary": (vars(obs.record) if obs.record else None),
        "tracer_snapshots": snaps,
    }
    text = json.dumps(_jsonable(artifact), sort_keys=True)
    back = json.loads(text)
    assert back["inherited_I"] == pre.inherited_I
    assert len(back["legs"]) == 4
    # Non-null snapshots must survive _jsonable -- serializing None would test
    # nothing about the tracer payload's serializability.
    assert back["tracer_snapshots"]["at_boundary"] is not None
    assert back["tracer_snapshots"]["at_400"] is not None
    assert back["legs"][0]["visit_counts"]        # per-rung evidence round-trips


# append to tests/test_atlas_ladder_integration.py
from scripts.GPU.alphazero.warm_prefix_replay import (
    LEG_INCREMENTS, project_runtime,
)


def test_runtime_projection_uses_stage1_measured_overhead():
    r = project_runtime(rows=240, mean_prefix_plies=69)
    # Ladder is 6,400 new sims per row; prefix is mean_plies * 400.
    assert r["ladder_sims_per_row"] == sum(LEG_INCREMENTS) == 6400
    assert r["prefix_sims_per_row"] == 69 * 400
    assert r["total_sims"] == 240 * (6400 + 69 * 400)
    assert r["tracer_overhead"] == 0.010
    assert r["tracer_overhead_is_upper_bound"] is True


def test_projection_states_the_prefix_dominates():
    r = project_runtime(rows=240, mean_prefix_plies=69)
    assert r["prefix_sims_per_row"] > r["ladder_sims_per_row"]
    assert r["dominant_term"] == "prefix_replay"


def test_projection_refuses_to_invent_a_ply_distribution():
    """mean_prefix_plies must be MEASURED from the corpus, never defaulted."""
    with pytest.raises(TypeError):
        project_runtime(rows=240)          # no default
