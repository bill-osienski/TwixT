"""Tests for Phase 2 parallel labeling, MCTSConfig wiring, borderline rerun.

See docs/superpowers/specs/2026-04-28-probe-phase2-parallel-labeling-design.md
"""
from __future__ import annotations

from dataclasses import replace as dc_replace
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
