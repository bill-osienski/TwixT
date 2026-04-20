"""Schema + basic-flow tests for probe suite tooling."""
import json
import subprocess
import tempfile
import os

import pytest


def test_sampler_cli_help():
    """Sampler CLI should respond to --help without error."""
    result = subprocess.run(
        [".venv/bin/python", "scripts/build_probe_candidates.py", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--out" in result.stdout
    assert "--min-source-iter" in result.stdout


@pytest.mark.slow
def test_sampler_produces_candidates_json(tmp_path):
    """Sampler against the current logs/games should produce non-empty candidates.json
    with required fields per candidate."""
    out = tmp_path / "candidates.json"
    result = subprocess.run(
        [".venv/bin/python", "scripts/build_probe_candidates.py",
         "--input", "scripts/GPU/logs/games",
         "--out", str(out),
         "--min-source-iter", "995",
         "--per-category-target", "10"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    data = json.loads(out.read_text())
    assert isinstance(data, dict)
    assert "candidates" in data
    assert len(data["candidates"]) > 0
    for cand in data["candidates"][:5]:
        assert "id" in cand
        assert "category" in cand
        assert "side_to_move" in cand
        assert "move_history" in cand
        assert "source_game" in cand
        assert "source_ply" in cand
        assert "active_size" in cand
        assert cand["active_size"] == 24  # default filter
