"""Live integration test for the trainer's forced-probe inline path.

Marked @pytest.mark.integration. Opt-in locally (pytest -m integration);
required in CI.

Runs 2 minimal training iterations against the committed bootstrap
probes and asserts:
  - forced_probe_summary lands in each iter's sidecar with n > 0
  - rolling-5 math populates correctly on iter 2 (None on iter 1)
  - sanity_by_connectivity is dict with winning/no_winning keys
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest


@pytest.mark.integration
def test_trainer_writes_forced_probe_summary_to_sidecar(tmp_path):
    """Full train() call with minimal config, 2 iterations, real probes file."""
    # Locate the most recent canonical checkpoint to resume from.
    ckpt_root = Path("checkpoints")
    if not ckpt_root.is_dir():
        pytest.skip("no checkpoints/ directory — cannot exercise live trainer path")
    # Pick the newest non-partial safetensors under any subdir.
    candidates = sorted(ckpt_root.glob("*/model_iter_*.safetensors"))
    candidates = [c for c in candidates if "_partial" not in c.name]
    if not candidates:
        pytest.skip("no canonical checkpoint available to resume from")
    resume = candidates[-1]
    # Extract the absolute iter number from the filename.
    iter_num = int(resume.stem.split("_")[-1])
    target_iter = iter_num + 2

    # Verify the committed probes file is present.
    probes_path = Path("tests/probes/twixt_probes.json")
    assert probes_path.exists(), (
        "Bootstrap probes file missing — Task 10 must land before this test. "
        "Run scripts/build_bootstrap_probe_suite.py --source-iter-range ..."
    )

    # Run 2 minimal iterations.
    import subprocess
    result = subprocess.run(
        [
            ".venv/bin/python", "-m", "scripts.GPU.alphazero.train",
            "--resume", str(resume),
            "--iterations", str(target_iter),
            "--games-per-iter", "2",
            "--simulations", "20",
            "--n-workers", "1",
            "--mcts-eval-batch-size", "2",
            "--checkpoint-dir", str(tmp_path / "ckpt"),
            "--probes-path", str(probes_path),
        ],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, f"train() failed:\n{result.stderr[-2000:]}"

    # Locate the written per-iter stats sidecars.
    games_dir = Path("scripts/GPU/logs/games")
    sidecar_1 = games_dir / f"iter_{iter_num:04d}_stats.json"
    sidecar_2 = games_dir / f"iter_{iter_num + 1:04d}_stats.json"
    assert sidecar_1.exists(), f"iter {iter_num} sidecar missing"
    assert sidecar_2.exists(), f"iter {iter_num + 1} sidecar missing"

    s1 = json.loads(sidecar_1.read_text())
    s2 = json.loads(sidecar_2.read_text())

    # Assertion 1: forced_probe_summary populated with n > 0 on both iters.
    fps1 = s1.get("forced_probe_summary")
    fps2 = s2.get("forced_probe_summary")
    assert fps1 is not None and fps1.get("n", 0) > 0, f"iter {iter_num}: {fps1}"
    assert fps2 is not None and fps2.get("n", 0) > 0, f"iter {iter_num+1}: {fps2}"

    # Assertion 2: rolling-5 / delta math — None on iter 1, float on iter 2.
    assert fps1.get("rolling5_sign_correct_pct") is None, \
        f"iter 1 rolling5 should be None, got {fps1.get('rolling5_sign_correct_pct')}"
    assert fps1.get("delta_sign_correct_pct") is None, \
        f"iter 1 delta should be None, got {fps1.get('delta_sign_correct_pct')}"
    assert isinstance(fps2.get("rolling5_sign_correct_pct"), float), \
        f"iter 2 rolling5 should be float, got {type(fps2.get('rolling5_sign_correct_pct'))}"
    assert isinstance(fps2.get("delta_sign_correct_pct"), float), \
        f"iter 2 delta should be float, got {type(fps2.get('delta_sign_correct_pct'))}"

    # Assertion 3: sanity_by_connectivity structure.
    sbc1 = s1.get("sanity_by_connectivity")
    assert isinstance(sbc1, dict), f"sbc missing or wrong type: {type(sbc1)}"
    assert "winning_structure" in sbc1 or "no_winning_structure" in sbc1, \
        f"sbc keys unexpected: {list(sbc1.keys())}"
