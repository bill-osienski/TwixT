"""Opt-in live smoke test for the strong_advantage labeling path.

Goal: confirm the labeling code path runs end-to-end without crashing —
checkpoint load, candidate replay, MCTS label call, admission filter,
draft output. NOT a label-correctness test.

Marker-gated; run with:
    .venv/bin/pytest -m slow_live tests/test_strong_advantage_smoke_live.py

Requires checkpoints/alphazero-v2-staged/model_iter_0059.safetensors on
disk and at least one decisive game in scripts/GPU/logs/games for the
hard-coded iter range below.
"""
from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT = PROJECT_ROOT / "checkpoints" / "alphazero-v2-staged" / "model_iter_0059.safetensors"


@pytest.mark.slow_live
def test_strong_advantage_smoke_live(tmp_path):
    if not CKPT.exists():
        pytest.skip(f"checkpoint not present: {CKPT}")

    import scripts.build_probe_suite as bps

    out_path = tmp_path / "smoke.json"
    rc = bps.main_with_args([
        "--tier", "strong_advantage",
        "--input", "scripts/GPU/logs/games",
        "--source-iter-range", "25", "26",
        "--label-checkpoint", str(CKPT),
        "--label-mcts-sims", "200",
        "--label-mcts-repeats", "1",
        "--max-probes", "3",
        "--out", str(out_path),
    ])
    # rc 0 (admitted some probes) OR rc 1 (zero admitted — drop reasons logged)
    # are both acceptable plumbing-wise. rc 2 is a usage/config error.
    assert rc in (0, 1), f"smoke run errored at config/usage stage: rc={rc}"

    if rc == 0:
        draft = out_path.with_suffix(".draft.json")
        assert draft.exists()
