"""Trainer-level loss and sidecar smoke tests (Spec 2)."""
from __future__ import annotations
import json


def test_trainer_runs_with_conversion_enabled_smoke(tmp_path):
    """Phase 3 smoke: 1 iter with conversion enabled produces sidecar block
    with correct shape including consistency.available=True (sampler stats wired)."""
    from scripts.GPU.alphazero.trainer import train

    network = train(
        n_iterations=1,
        games_per_iteration=2,
        train_steps_per_iteration=2,
        batch_size=4,
        buffer_size=50,
        checkpoint_dir=str(tmp_path),
        games_dir_override=str(tmp_path),
        save_games=False,
        probes_inline_disable=True,
        conversion_policy_loss_enabled=True,
        conversion_policy_loss_weight=0.05,
        conversion_max_total_goal_distance=2,
        mcts_simulations=4,
        hidden=32,
        n_blocks=1,
        max_moves=10,
        seed=42,
    )
    assert network is not None
    sidecar_files = list(tmp_path.glob("**/iter_*_stats.json"))
    if not sidecar_files:
        # Sidecar may live in checkpoint subdir; search broader.
        sidecar_files = list(tmp_path.rglob("iter_*_stats.json"))
    assert len(sidecar_files) >= 1, f"No sidecar JSON in {tmp_path}"
    sidecar = json.loads(sidecar_files[0].read_text())
    cnv = sidecar["conversion_training"]
    assert cnv["version"] == 1
    assert cnv["enabled"] is True
    assert cnv["config"]["effective_loss_weight"] == 0.05
    # Phase 3: sampler stats wired. Consistency check IS available now.
    assert cnv["consistency"]["available"] is True
    assert cnv["consistency"]["drawn_vs_seen_match"] is True
    # Buffer stats from O(N) scan: with conversion enabled and games played,
    # buffer should contain real positions. eligible_positions_in_buffer can be
    # 0 legitimately (no closeouts in 2-game smoke), but the FIELD must be
    # populated with a real int and the rate must be a valid float.
    assert isinstance(cnv["buffer"]["eligible_positions_in_buffer"], int)
    assert 0.0 <= cnv["buffer"]["eligible_position_rate"] <= 1.0


def test_trainer_runs_with_sample_boost_smoke(tmp_path):
    """Phase 3 smoke: 1 iter, conversion enabled with boost=2.0.
    Asserts sample_stats populated and drawn-vs-seen invariant holds.
    """
    from scripts.GPU.alphazero.trainer import train

    network = train(
        n_iterations=1,
        games_per_iteration=2,
        train_steps_per_iteration=4,
        batch_size=8,
        buffer_size=100,
        checkpoint_dir=str(tmp_path),
        games_dir_override=str(tmp_path),
        save_games=False,
        probes_inline_disable=True,
        conversion_policy_loss_enabled=True,
        conversion_policy_loss_weight=0.05,
        conversion_sample_boost=2.0,
        conversion_max_batch_fraction=0.5,
    )
    assert network is not None

    sidecar_files = list(tmp_path.rglob("iter_*_stats.json"))
    assert len(sidecar_files) >= 1
    sidecar = json.loads(sidecar_files[0].read_text())
    cnv = sidecar["conversion_training"]
    # Phase 3 invariant: sampler stats wired; consistency check available.
    assert cnv["consistency"]["available"] is True
    assert cnv["consistency"]["drawn_vs_seen_match"] is True
    # eligible_drawn_total must equal aux_positions_seen_in_training.
    assert cnv["sample_stats"]["eligible_drawn_total"] == cnv["loss"]["aux_positions_seen_in_training"]


def test_trainer_writes_recovery_block_to_sidecar(tmp_path):
    """Spec 2 §8.6: trainer emits recovery_or_extreme_closeout_drift sidecar."""
    from scripts.GPU.alphazero.trainer import train
    train(
        n_iterations=1, games_per_iteration=2, train_steps_per_iteration=2,
        batch_size=4, buffer_size=50, checkpoint_dir=str(tmp_path),
        save_games=False, probes_inline_disable=True,
        recovery_bucket_enabled=True,
        games_dir_override=str(tmp_path),
    )
    import json
    sidecar = json.loads(list(tmp_path.rglob("iter_*_stats.json"))[0].read_text())
    rec = sidecar["recovery_or_extreme_closeout_drift"]
    assert rec["version"] == 1
    assert "config" in rec
    assert "trigger_breakdown" in rec
