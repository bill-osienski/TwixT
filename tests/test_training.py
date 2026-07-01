#!/usr/bin/env python3
"""Tests for AlphaZero training loop.

Run with: python3 tests/test_training.py
"""
import sys
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_loss_computes():
    """Test that loss computes without error.

    Uses the current public API `alphazero_loss_batch`, which returns a 7-tuple
    (total_loss, policy_loss, value_loss, l2_loss, aux_loss, aux_coverage,
    aux_n_eligible) — total_loss is first so that `nn.value_and_grad`
    differentiates it. Test validates the scalar total loss is finite and positive.
    """
    import numpy as np
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.trainer import alphazero_loss_batch
    from scripts.GPU.alphazero.self_play import PositionRecord
    from scripts.GPU.alphazero.game.twixt_state import NUM_CHANNELS

    network = create_network(hidden=64, n_blocks=2)

    # Create fake positions
    positions = []
    for i in range(3):
        pos = PositionRecord(
            board_tensor=np.random.randn(24, 24, NUM_CHANNELS).astype(np.float32),
            to_move="red" if i % 2 == 0 else "black",
            legal_moves=[(1, 1), (2, 2), (3, 3)],
            visit_counts=[50, 30, 20],
            outcome=1.0 if i == 0 else -1.0,
        )
        positions.append(pos)

    # Compute loss — batched API returns (total, policy, value, l2, aux, coverage, n_eligible)
    total_loss, policy_loss, value_loss, l2_loss, _, _, _ = alphazero_loss_batch(
        network, positions
    )
    total_val = float(total_loss)

    assert not np.isnan(total_val), "Loss is NaN"
    assert not np.isinf(total_val), "Loss is infinite"
    assert total_val > 0, "Loss should be positive"

    print(f"PASS: Loss computes (total={total_val:.4f}, "
          f"policy={float(policy_loss):.4f}, value={float(value_loss):.4f}, "
          f"l2={float(l2_loss):.4f})")


def test_loss_components():
    """Test that loss has policy, value, and L2 components.

    Uses `alphazero_loss_batch`. Verifies that the L2 component changes with
    `l2_weight` and that turning L2 off produces a strictly smaller total loss.
    """
    import numpy as np
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.trainer import alphazero_loss_batch
    from scripts.GPU.alphazero.self_play import PositionRecord
    from scripts.GPU.alphazero.game.twixt_state import NUM_CHANNELS

    network = create_network(hidden=64, n_blocks=2)

    # Create position
    pos = PositionRecord(
        board_tensor=np.zeros((24, 24, NUM_CHANNELS), dtype=np.float32),
        to_move="red",
        legal_moves=[(5, 5), (6, 6)],
        visit_counts=[80, 20],
        outcome=1.0,
    )

    # Loss with L2 — unpack total (7-tuple)
    total_with_l2, _, _, l2_with, _, _, _ = alphazero_loss_batch(network, [pos], l2_weight=1e-3)
    total_with_l2 = float(total_with_l2)
    l2_with = float(l2_with)

    # Loss without L2
    total_no_l2, _, _, l2_no, _, _, _ = alphazero_loss_batch(network, [pos], l2_weight=0.0)
    total_no_l2 = float(total_no_l2)
    l2_no = float(l2_no)

    # L2 should make total loss larger, and l2 component should match its weight scaling
    assert total_with_l2 > total_no_l2, (
        f"L2 should increase loss: {total_with_l2} vs {total_no_l2}"
    )
    assert l2_with > 0, "L2 component should be > 0 when l2_weight>0"
    assert l2_no == 0.0, f"L2 component should be 0 when l2_weight=0, got {l2_no}"

    print(f"PASS: Loss components (with L2={total_with_l2:.4f}, no L2={total_no_l2:.4f})")


def _make_train_step_fixtures(hidden: int = 64, n_blocks: int = 2,
                               lr: float = 1e-3, value_lr_scale: float = 0.1):
    """Build the (network, main_module, opt_main, opt_value) tuple that the
    current two-optimizer `train_step` expects. Mirrors `trainer.train()`.
    """
    import mlx.optimizers as optim
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.trainer import MainModule

    network = create_network(hidden=hidden, n_blocks=n_blocks)
    main_module = MainModule(network.encoder, network.policy_head)
    opt_main = optim.Adam(learning_rate=lr)
    opt_value = optim.Adam(learning_rate=lr * value_lr_scale)
    return network, main_module, opt_main, opt_value


def test_train_step():
    """Test single training step with the two-optimizer API.

    `train_step` now takes (network, main_module, opt_main, opt_value, batch, ...)
    and returns (total, policy, value, l2, aux, coverage, n_eligible). We assert
    the first four losses are finite and the total is positive.
    """
    import numpy as np
    from scripts.GPU.alphazero.trainer import train_step
    from scripts.GPU.alphazero.self_play import PositionRecord
    from scripts.GPU.alphazero.game.twixt_state import NUM_CHANNELS

    network, main_module, opt_main, opt_value = _make_train_step_fixtures()

    # Create batch
    batch = []
    for _ in range(4):
        pos = PositionRecord(
            board_tensor=np.random.randn(24, 24, NUM_CHANNELS).astype(np.float32),
            to_move="red",
            legal_moves=[(1, 1), (2, 2)],
            visit_counts=[60, 40],
            outcome=1.0,
        )
        batch.append(pos)

    # Run step — returns (total, policy, value, l2, aux, coverage, n_eligible) as floats/int
    total_loss, policy_loss, value_loss, l2_loss, _, _, _ = train_step(
        network, main_module, opt_main, opt_value, batch
    )

    for name, val in (("total", total_loss), ("policy", policy_loss),
                      ("value", value_loss), ("l2", l2_loss)):
        assert not np.isnan(val), f"{name} loss is NaN after train step"
        assert not np.isinf(val), f"{name} loss is inf after train step"
    assert total_loss > 0, "Total loss should be positive"

    print(f"PASS: Train step (total={total_loss:.4f}, "
          f"policy={policy_loss:.4f}, value={value_loss:.4f}, l2={l2_loss:.4f})")


def test_loss_decreases():
    """Test that loss decreases over multiple training steps.

    Same two-optimizer API. We overfit a tiny fixed batch and check the total
    loss at the end is lower than at the start.
    """
    import numpy as np
    from scripts.GPU.alphazero.trainer import train_step
    from scripts.GPU.alphazero.self_play import PositionRecord
    from scripts.GPU.alphazero.game.twixt_state import NUM_CHANNELS

    # Higher LR for faster convergence on this tiny fixed batch
    network, main_module, opt_main, opt_value = _make_train_step_fixtures(lr=1e-2)

    # Create fixed batch (same data every step) with consistent targets
    np.random.seed(42)
    batch = []
    for i in range(8):
        # Use simpler data: zeros board, uniform visits, consistent outcomes
        pos = PositionRecord(
            board_tensor=np.zeros((24, 24, NUM_CHANNELS), dtype=np.float32),
            to_move="red",
            legal_moves=[(1, 1), (2, 2)],
            visit_counts=[50, 50],  # Uniform distribution
            outcome=1.0,  # Consistent outcome
        )
        batch.append(pos)

    # Train for 50 steps and record total losses
    losses = []
    for step in range(50):
        total_loss, _, _, _, _, _, _ = train_step(
            network, main_module, opt_main, opt_value, batch,
            l2_weight=0.0,  # No L2 so we can see pure policy+value overfitting
        )
        losses.append(total_loss)

    initial_loss = float(np.mean(losses[:5]))
    final_loss = float(np.mean(losses[-5:]))

    assert final_loss < initial_loss, (
        f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )

    print(f"PASS: Loss decreases ({initial_loss:.4f} -> {final_loss:.4f})")


def test_replay_buffer():
    """Test ReplayBuffer add and sample."""
    import numpy as np
    from scripts.GPU.alphazero.trainer import ReplayBuffer
    from scripts.GPU.alphazero.self_play import PositionRecord, GameRecord
    from scripts.GPU.alphazero.game.twixt_state import NUM_CHANNELS

    buffer = ReplayBuffer(max_size=100)

    # Create a fake game
    positions = []
    for i in range(5):
        pos = PositionRecord(
            board_tensor=np.zeros((24, 24, NUM_CHANNELS), dtype=np.float32),
            to_move="red",
            legal_moves=[(i, i)],
            visit_counts=[100],
            outcome=1.0,
        )
        positions.append(pos)

    game = GameRecord(
        positions=positions,
        winner="red",
        n_moves=5,
        move_history=[(i, i) for i in range(5)],
    )

    # Add game
    buffer.add_game(game)
    assert len(buffer) == 5, f"Buffer should have 5 positions, got {len(buffer)}"

    # Sample
    sample = buffer.sample(3)
    assert len(sample) == 3, f"Sample should have 3 positions, got {len(sample)}"

    # Sample more than buffer size
    sample = buffer.sample(100)
    assert len(sample) == 5, f"Sample should be capped at buffer size"

    print("PASS: ReplayBuffer add and sample")


def test_replay_buffer_overflow():
    """Test ReplayBuffer ring buffer behavior."""
    import numpy as np
    from scripts.GPU.alphazero.trainer import ReplayBuffer
    from scripts.GPU.alphazero.self_play import PositionRecord
    from scripts.GPU.alphazero.game.twixt_state import NUM_CHANNELS

    buffer = ReplayBuffer(max_size=10)

    # Add 15 positions (should wrap around)
    for i in range(15):
        pos = PositionRecord(
            board_tensor=np.full((24, 24, NUM_CHANNELS), i, dtype=np.float32),
            to_move="red",
            legal_moves=[(0, 0)],
            visit_counts=[100],
            outcome=1.0,
        )
        buffer.add_positions([pos])

    # Buffer should have exactly max_size positions
    assert len(buffer) == 10, f"Buffer should have 10 positions, got {len(buffer)}"

    # Oldest positions (0-4) should be overwritten with (10-14)
    # Buffer should contain positions 5-14
    values = set()
    for pos in buffer.buffer:
        val = int(pos.board_tensor[0, 0, 0])
        values.add(val)

    expected = set(range(5, 15))
    assert values == expected, f"Buffer should contain {expected}, got {values}"

    print("PASS: ReplayBuffer overflow (ring buffer)")


def test_checkpoint_save_load():
    """Test that checkpoints save and load correctly."""
    import numpy as np
    import mlx.core as mx
    from scripts.GPU.alphazero.network import create_network

    network1 = create_network(hidden=64, n_blocks=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_model.safetensors"

        # Save
        network1.save_weights(str(ckpt_path))
        assert ckpt_path.exists(), "Checkpoint file should exist"

        # Load into new network
        network2 = create_network(hidden=64, n_blocks=2)
        network2.load_weights(str(ckpt_path))

        # Compare parameters
        params1 = network1.parameters()
        params2 = network2.parameters()

        def compare_params(p1, p2, path=""):
            if isinstance(p1, dict):
                for k in p1:
                    compare_params(p1[k], p2[k], f"{path}.{k}")
            elif isinstance(p1, list):
                for i, (a, b) in enumerate(zip(p1, p2)):
                    compare_params(a, b, f"{path}[{i}]")
            elif isinstance(p1, mx.array):
                diff = float(mx.max(mx.abs(p1 - p2)))
                assert diff < 1e-6, f"Param mismatch at {path}: max diff {diff}"

        compare_params(params1, params2)

    print("PASS: Checkpoint save/load")


def test_mini_training_run():
    """Test a minimal training run (1 iteration, 1 game)."""
    from scripts.GPU.alphazero.trainer import train

    with tempfile.TemporaryDirectory() as tmpdir:
        network = train(
            n_iterations=1,
            games_per_iteration=1,
            train_steps_per_iteration=5,
            batch_size=4,
            buffer_size=1000,
            checkpoint_dir=tmpdir,
            mcts_simulations=10,  # Very low for speed
            learning_rate=1e-3,
            hidden=64,
            n_blocks=2,
            max_moves=10,  # Short games
            seed=42,
            # Isolate self-play game output: train() defaults to writing games to
            # the shared scripts/GPU/logs/games/ dir, which is real fixture data
            # for the analyzer smoke tests. Keep test artifacts in tmpdir.
            games_dir_override=str(Path(tmpdir) / "games"),
        )

        # Check checkpoint was saved
        ckpt_files = list(Path(tmpdir).glob("*.safetensors"))
        assert len(ckpt_files) == 1, f"Should have 1 checkpoint, got {len(ckpt_files)}"

        # Check state file
        state_files = list(Path(tmpdir).glob("*.json"))
        assert len(state_files) == 1, f"Should have 1 state file, got {len(state_files)}"

    print("PASS: Mini training run")


def test_train_default_value_weight_is_half():
    """Default value_weight is 0.5 in train() signature."""
    import inspect
    from scripts.GPU.alphazero.trainer import train
    sig = inspect.signature(train)
    assert sig.parameters["value_weight"].default == 0.5


def test_train_cli_has_progress_weighted_flag():
    """CLI exposes --progress-weighted-value-loss and --progress-weight-floor and --value-weight."""
    import subprocess
    result = subprocess.run(
        [".venv/bin/python", "scripts/GPU/alphazero/train.py", "--help"],
        capture_output=True, text=True,
    )
    assert "--progress-weighted-value-loss" in result.stdout
    assert "--progress-weight-floor" in result.stdout
    assert "--value-weight" in result.stdout


def test_csv_fieldnames_includes_calibration_columns():
    """Post-opening calibration scalars must be CSV columns so they persist to
    metrics.csv and model_iter_*.json (state spreads iteration_metrics)."""
    from scripts.GPU.alphazero.trainer import CSV_FIELDNAMES
    for col in ("calib_loss_avg_iter", "calib_mean_value_pred",
                "calib_n_drawn_total", "calib_n_drawn_per_step"):
        assert col in CSV_FIELDNAMES, f"{col} missing from CSV_FIELDNAMES"
    print("PASS: CSV_FIELDNAMES includes calibration columns")


def test_calibration_telemetry_persisted_to_metrics_and_model_iter_json():
    """Regression: post_opening_calibration telemetry reached only the
    iter_<N>_stats.json sidecar, not metrics.csv / model_iter_*.json. With
    calibration enabled, a mini run must write finite, populated calib_* values
    into BOTH metrics.csv and the per-checkpoint model_iter JSON."""
    import csv as _csv
    import json as _json
    import math as _math
    from scripts.GPU.alphazero.trainer import train
    from tests.goal_line_probe_fixtures import legal_replay

    CALIB_COLS = ("calib_loss_avg_iter", "calib_mean_value_pred",
                  "calib_n_drawn_total", "calib_n_drawn_per_step")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Minimal 1-case global-target calibration manifest + its replay.
        # ply 5 is odd => black to move (legal_replay alternates from red).
        replay = legal_replay(8, game_idx=1)
        rpath = tmp / "game_000001.json"
        rpath.write_text(_json.dumps(replay))
        manifest = tmp / "calib.csv"
        with manifest.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["game_idx", "case_id", "replay_path",
                                               "position_ply", "side_to_move"])
            w.writeheader()
            w.writerow({"game_idx": 1, "case_id": "game_000001_ply_005",
                        "replay_path": str(rpath), "position_ply": 5,
                        "side_to_move": "black"})

        ckpt_dir = tmp / "ckpt"
        train(
            n_iterations=1, games_per_iteration=1, train_steps_per_iteration=5,
            batch_size=4, buffer_size=1000, checkpoint_dir=str(ckpt_dir),
            mcts_simulations=10, learning_rate=1e-3, hidden=64, n_blocks=2,
            max_moves=10, seed=42,
            post_opening_calibration_enabled=True,
            post_opening_calibration_manifest=str(manifest),
            post_opening_calibration_weight=0.02,
            post_opening_calibration_batch_fraction=0.10,
            # Isolate self-play game output (see test_mini_training_run): never
            # write into the shared scripts/GPU/logs/games/ analyzer fixtures.
            games_dir_override=str(tmp / "games"),
        )

        # metrics.csv carries the calib columns AND finite populated values.
        rows = list(_csv.DictReader((ckpt_dir / "metrics.csv").open()))
        assert rows, "no metrics.csv rows written"
        row = rows[-1]
        for col in CALIB_COLS:
            assert col in row, f"{col} missing from metrics.csv header"
            assert row[col] not in ("", None), f"{col} empty despite calibration enabled"
        assert _math.isfinite(float(row["calib_mean_value_pred"]))
        assert int(row["calib_n_drawn_total"]) > 0

        # model_iter_*.json (per-checkpoint state) carries the same keys.
        model_jsons = sorted(ckpt_dir.glob("model_iter_*.json"))
        assert model_jsons, "no model_iter_*.json written"
        state = _json.loads(model_jsons[-1].read_text())
        for col in CALIB_COLS:
            assert col in state, f"{col} missing from model_iter JSON"
        assert state["calib_n_drawn_total"] > 0

    print("PASS: calibration telemetry persisted to metrics.csv + model_iter JSON")


def test_teacher_calibration_scalars_and_freeze_flag_in_model_iter_json():
    """v4: a teacher-retention run must persist freeze_batchnorm_stats AND the
    teacher telemetry scalars (n_teacher_retention_drawn, calib_policy_ce/kl_est/
    value_term) into the per-checkpoint model_iter JSON state (flat scalars,
    alongside calib_n_drawn_by_tag; the full dict block stays in the sidecar)."""
    import csv as _csv
    import json as _json
    import math as _math
    from scripts.GPU.alphazero.trainer import train
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.build_teacher_calibration_manifest import build_rows
    from tests.goal_line_probe_fixtures import legal_replay

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        replay = legal_replay(8, game_idx=1)
        rpath = tmp / "game_000001.json"
        rpath.write_text(_json.dumps(replay))
        # Tiny teacher-retention manifest (1 retention row) built via the real
        # builder in eval mode, so the pool schema is teacher_retention and every
        # draw is a teacher row.
        net_t = create_network(hidden=64, n_blocks=2)
        net_t.eval()
        built = build_rows(
            [{"game_idx": "1", "case_id": "ret1", "replay_path": str(rpath),
              "position_ply": "5", "side_to_move": "black",
              "tag": "old_post_opening_retention", "weight_scale": "1.0"}],
            LocalGPUEvaluator(net_t))
        manifest = tmp / "teacher.csv"
        with manifest.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(built[0].keys()))
            w.writeheader(); w.writerows(built)

        ckpt_dir = tmp / "ckpt"
        train(
            n_iterations=1, games_per_iteration=1, train_steps_per_iteration=5,
            batch_size=4, buffer_size=1000, checkpoint_dir=str(ckpt_dir),
            mcts_simulations=10, learning_rate=1e-3, hidden=64, n_blocks=2,
            max_moves=10, seed=42,
            post_opening_calibration_enabled=True,
            post_opening_calibration_manifest=str(manifest),
            post_opening_calibration_weight=0.02,
            post_opening_calibration_batch_fraction=0.10,
            freeze_batchnorm_stats=True,
            games_dir_override=str(tmp / "games"),
        )

        state = _json.loads(sorted(ckpt_dir.glob("model_iter_*.json"))[-1].read_text())
        assert state.get("freeze_batchnorm_stats") is True, "freeze_batchnorm_stats not persisted"
        for k in ("n_teacher_retention_drawn", "calib_policy_ce_avg_iter",
                  "calib_policy_kl_est_avg_iter", "calib_value_term_avg_iter"):
            assert k in state, f"{k} missing from model_iter JSON state"
        assert state["n_teacher_retention_drawn"] > 0
        assert _math.isfinite(float(state["calib_policy_ce_avg_iter"]))
        assert _math.isfinite(float(state["calib_value_term_avg_iter"]))

    print("PASS: v4 teacher scalars + freeze flag persisted to model_iter JSON")


def test_calibration_tag_schedule_draw_counts_persisted():
    """v3: a tag schedule draws per-tag counts each step and persists
    calib_n_drawn_by_tag (a dict) into model_iter_*.json state -- in the
    scheduled ratio -- and never into metrics.csv (flat scalars only)."""
    import csv as _csv
    import json as _json
    from scripts.GPU.alphazero.trainer import train
    from tests.goal_line_probe_fixtures import legal_replay

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Two tagged black-to-move rows (ply 5 odd => black to move).
        rows = []
        for gi, tag in ((1, "correction"), (2, "retention")):
            replay = legal_replay(8, game_idx=gi)
            rpath = tmp / f"game_{gi:06d}.json"
            rpath.write_text(_json.dumps(replay))
            rows.append({"game_idx": gi, "case_id": f"game_{gi:06d}_ply_005",
                         "replay_path": str(rpath), "position_ply": 5,
                         "side_to_move": "black", "tag": tag})
        manifest = tmp / "calib_tagged.csv"
        with manifest.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["game_idx", "case_id", "replay_path",
                                               "position_ply", "side_to_move", "tag"])
            w.writeheader()
            w.writerows(rows)

        ckpt_dir = tmp / "ckpt"
        train(
            n_iterations=1, games_per_iteration=1, train_steps_per_iteration=5,
            batch_size=4, buffer_size=1000, checkpoint_dir=str(ckpt_dir),
            mcts_simulations=10, learning_rate=1e-3, hidden=64, n_blocks=2,
            max_moves=10, seed=42,
            post_opening_calibration_enabled=True,
            post_opening_calibration_manifest=str(manifest),
            post_opening_calibration_weight=0.02,
            post_opening_calibration_tag_schedule={"correction": 2, "retention": 1},
            # Isolate self-play output from the shared analyzer fixtures.
            games_dir_override=str(tmp / "games"),
        )

        # metrics.csv must NOT carry the dict (flat scalars only).
        header = next(_csv.reader((ckpt_dir / "metrics.csv").open()))
        assert "calib_n_drawn_by_tag" not in header

        # model_iter_*.json state carries the dict, in the scheduled 2:1 ratio.
        model_jsons = sorted(ckpt_dir.glob("model_iter_*.json"))
        assert model_jsons, "no model_iter_*.json written"
        state = _json.loads(model_jsons[-1].read_text())
        by_tag = state["calib_n_drawn_by_tag"]
        assert set(by_tag) == {"correction", "retention"}
        assert by_tag["correction"] > 0 and by_tag["retention"] > 0
        assert by_tag["correction"] == 2 * by_tag["retention"]  # each step: 2 corr : 1 ret

    print("PASS: tag-stratified calibration draw counts persisted to model_iter JSON")


def test_calibration_tag_schedule_unknown_tag_fails_before_selfplay():
    """A schedule naming a tag absent from the manifest must raise at trainer
    setup (before self-play), not after a wasted iteration. Validation runs
    right after the pool is built, so train() returns the error fast."""
    import csv as _csv
    import json as _json
    from scripts.GPU.alphazero.trainer import train
    from tests.goal_line_probe_fixtures import legal_replay

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        replay = legal_replay(8, game_idx=1)
        rpath = tmp / "game_000001.json"
        rpath.write_text(_json.dumps(replay))
        manifest = tmp / "calib_tagged.csv"
        with manifest.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["game_idx", "case_id", "replay_path",
                                               "position_ply", "side_to_move", "tag"])
            w.writeheader()
            w.writerow({"game_idx": 1, "case_id": "game_000001_ply_005",
                        "replay_path": str(rpath), "position_ply": 5,
                        "side_to_move": "black", "tag": "correction"})

        raised = False
        try:
            train(
                n_iterations=1, games_per_iteration=1, train_steps_per_iteration=5,
                batch_size=4, buffer_size=1000, checkpoint_dir=str(tmp / "ckpt"),
                mcts_simulations=10, learning_rate=1e-3, hidden=64, n_blocks=2,
                max_moves=10, seed=42,
                post_opening_calibration_enabled=True,
                post_opening_calibration_manifest=str(manifest),
                post_opening_calibration_weight=0.02,
                post_opening_calibration_tag_schedule={"correction": 1, "typo_tag": 1},
                games_dir_override=str(tmp / "games"),
            )
        except ValueError:
            raised = True
        assert raised, "expected ValueError for a schedule tag absent from the manifest"
    print("PASS: unknown scheduled tag fails before self-play")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TRAINING LOOP TESTS")
    print("=" * 60)
    print()

    tests = [
        test_loss_computes,
        test_loss_components,
        test_train_step,
        test_loss_decreases,
        test_replay_buffer,
        test_replay_buffer_overflow,
        test_checkpoint_save_load,
        test_mini_training_run,
        test_train_default_value_weight_is_half,
        test_train_cli_has_progress_weighted_flag,
        test_csv_fieldnames_includes_calibration_columns,
        test_calibration_telemetry_persisted_to_metrics_and_model_iter_json,
        test_calibration_tag_schedule_draw_counts_persisted,
        test_calibration_tag_schedule_unknown_tag_fails_before_selfplay,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()  # passes if no exception; assertions raise on failure
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} - {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("Gate PASSED: Training loop works, loss decreases, checkpoints save/load")
        return 0
    else:
        print("Gate FAILED: Training tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
