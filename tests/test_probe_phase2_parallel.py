"""Tests for Phase 2 parallel labeling, MCTSConfig wiring, borderline rerun.

See docs/superpowers/specs/2026-04-28-probe-phase2-parallel-labeling-design.md
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import replace as dc_replace
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.GPU.alphazero.mcts import MCTSConfig
from scripts.GPU.alphazero import probe_eval


def test_default_mcts_labeler_uses_global_config(monkeypatch):
    """When _DEFAULT_LABELER_MCTS_CONFIG is set, _default_mcts_labeler builds
    MCTSConfig with eval_batch_size and stall_flush_sims from the global,
    while n_simulations is overridden by the per-call sims argument.
    """
    captured_cfg = {}

    class FakeNet:
        pass

    class FakeMCTS:
        def __init__(self, evaluator, cfg, rng):
            captured_cfg["cfg"] = cfg

        def search(self, state, add_noise=False):
            return ({(0, 0): 7}, 0.5)

    monkeypatch.setattr(probe_eval, "_DEFAULT_LABELER_NETWORK", FakeNet())
    monkeypatch.setattr(
        probe_eval,
        "_DEFAULT_LABELER_MCTS_CONFIG",
        MCTSConfig(eval_batch_size=8, stall_flush_sims=4),
    )
    monkeypatch.setattr(probe_eval, "MCTS", FakeMCTS)
    monkeypatch.setattr(probe_eval, "LocalGPUEvaluator", lambda net: None)

    probe_eval._default_mcts_labeler(state=None, sims=2000, seed=42)

    cfg = captured_cfg["cfg"]
    assert cfg.eval_batch_size == 8
    assert cfg.stall_flush_sims == 4
    assert cfg.n_simulations == 2000


def test_default_mcts_labeler_back_compat_without_global(monkeypatch):
    """When _DEFAULT_LABELER_MCTS_CONFIG is None, _default_mcts_labeler builds
    MCTSConfig(n_simulations=sims) using dataclass defaults — preserves existing
    behavior for anything that hasn't been migrated to the new setter.
    """
    captured_cfg = {}

    class FakeNet:
        pass

    class FakeMCTS:
        def __init__(self, evaluator, cfg, rng):
            captured_cfg["cfg"] = cfg

        def search(self, state, add_noise=False):
            return ({(0, 0): 1}, 0.0)

    monkeypatch.setattr(probe_eval, "_DEFAULT_LABELER_NETWORK", FakeNet())
    monkeypatch.setattr(probe_eval, "_DEFAULT_LABELER_MCTS_CONFIG", None)
    monkeypatch.setattr(probe_eval, "MCTS", FakeMCTS)
    monkeypatch.setattr(probe_eval, "LocalGPUEvaluator", lambda net: None)

    probe_eval._default_mcts_labeler(state=None, sims=500, seed=0)

    cfg = captured_cfg["cfg"]
    assert cfg.n_simulations == 500
    assert cfg.eval_batch_size == 14   # MCTSConfig default
    assert cfg.stall_flush_sims == 16  # MCTSConfig default


def test_default_mcts_labeler_keeps_sims_authoritative(monkeypatch):
    """Even if _DEFAULT_LABELER_MCTS_CONFIG was constructed with a custom
    n_simulations, the per-call sims argument wins (via dataclasses.replace).
    """
    captured_cfg = {}

    class FakeNet:
        pass

    class FakeMCTS:
        def __init__(self, evaluator, cfg, rng):
            captured_cfg["cfg"] = cfg

        def search(self, state, add_noise=False):
            return ({(0, 0): 1}, 0.0)

    monkeypatch.setattr(probe_eval, "_DEFAULT_LABELER_NETWORK", FakeNet())
    monkeypatch.setattr(
        probe_eval,
        "_DEFAULT_LABELER_MCTS_CONFIG",
        MCTSConfig(n_simulations=99999, eval_batch_size=12),
    )
    monkeypatch.setattr(probe_eval, "MCTS", FakeMCTS)
    monkeypatch.setattr(probe_eval, "LocalGPUEvaluator", lambda net: None)

    probe_eval._default_mcts_labeler(state=None, sims=1234, seed=0)

    cfg = captured_cfg["cfg"]
    assert cfg.n_simulations == 1234
    assert cfg.eval_batch_size == 12


PROBE_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_probe_suite.py"
PYTHON = sys.executable


def _run_cli(*args):
    """Run build_probe_suite.py with given CLI args. Returns the full
    CompletedProcess (rc, stdout, stderr accessible)."""
    return subprocess.run(
        [PYTHON, str(PROBE_SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_argparse_flags_present():
    """Each new flag is registered with the documented default and choices."""
    spec = importlib.util.spec_from_file_location("build_probe_suite", PROBE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    parser = mod._build_arg_parser()  # introduced in this task; see step 4

    flags = {a.dest: a for a in parser._actions}
    assert flags["label_worker_mode"].choices == ["serial", "process"]
    assert flags["label_worker_mode"].default == "serial"
    assert flags["label_workers"].default == 1
    assert flags["mcts_eval_batch_size"].default == 14
    assert flags["mcts_stall_flush_sims"].default == 16
    assert flags["allow_unsafe_eval_batch"].default is False
    assert flags["admission_borderline_epsilon"].default == 0.01
    assert flags["no_borderline_rerun"].default is False


@pytest.mark.parametrize(
    "extra_args, fragment",
    [
        (["--label-workers", "0"], "label-workers"),
        (["--mcts-eval-batch-size", "0"], "mcts-eval-batch-size"),
        (["--mcts-stall-flush-sims", "-1"], "mcts-stall-flush-sims"),
        (["--admission-borderline-epsilon", "-0.1"], "admission-borderline-epsilon"),
    ],
)
def test_parallel_cli_numeric_validation(extra_args, fragment):
    """Reject negative / zero values where the spec disallows them."""
    proc = _run_cli(
        "--tier", "strong_advantage",
        "--source-iter-range", "0", "0",
        "--label-checkpoint", "/nonexistent.safetensors",
        *extra_args,
    )
    assert proc.returncode != 0
    assert fragment in proc.stderr


def test_unsafe_eval_batch_validation_rejects_high_value_without_flag():
    """--mcts-eval-batch-size > 14 without --allow-unsafe-eval-batch is an error."""
    proc = _run_cli(
        "--tier", "strong_advantage",
        "--source-iter-range", "0", "0",
        "--label-checkpoint", "/nonexistent.safetensors",
        "--mcts-eval-batch-size", "32",
    )
    assert proc.returncode != 0
    assert "--mcts-eval-batch-size > 14 is unsafe" in proc.stderr
    assert "--allow-unsafe-eval-batch" in proc.stderr


def test_unsafe_eval_batch_passes_with_flag():
    """--mcts-eval-batch-size > 14 with --allow-unsafe-eval-batch passes validation
    (run still fails because checkpoint doesn't exist, but past argparse).
    """
    proc = _run_cli(
        "--tier", "strong_advantage",
        "--source-iter-range", "0", "0",
        "--label-checkpoint", "/nonexistent.safetensors",
        "--mcts-eval-batch-size", "32",
        "--allow-unsafe-eval-batch",
    )
    # Argparse accepts the combination; run fails later because the checkpoint
    # doesn't exist. The argparse error message must NOT appear.
    assert "--mcts-eval-batch-size > 14 is unsafe" not in proc.stderr
