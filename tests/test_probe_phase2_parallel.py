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


def test_helper_returns_admitted_status_for_passing_candidate(monkeypatch, tmp_path):
    """The extracted helper returns the same admission decision the old serial
    loop would have written. Uses a tiny mocked labeler.

    This test fails until the helper exists with the documented signature.
    """
    from scripts.build_probe_suite import _label_one_strong_advantage_candidate
    from scripts.GPU.alphazero import probe_eval

    def stub_labeler(state, sims, seed):
        # Strong red-side advantage: high mean root, high top1.
        return (0.9, 0.5)

    cand = {
        "source_game": "iter_0_game_0",
        "source_ply": 18,
        "ply": 18,
        "starting_player": "red",
        "winner": "red",
        "category": "central_red",
        "move_history": [(11, 11)],
        "phase1_features": {
            "cc_size": 12,
            "cc_axis_span": 0.7,
            "axis_span_margin": 0.2,
        },
    }

    monkeypatch.setattr(probe_eval, "_default_mcts_labeler", stub_labeler)

    result = _label_one_strong_advantage_candidate(
        cand,
        label_ckpt_name="ckpt.safetensors",
        sims=10,
        repeats=2,
        magnitude_threshold=0.45,
        top1_share_floor=0.15,
        stability_cap=0.15,
    )
    assert result["status"] == "admitted"
    assert result["candidate"] is not None
    assert result["candidate"]["phase2_label"]["mean_root_value"] == pytest.approx(0.9)
    assert result["audit_row"] is None
    assert result["error_message"] is None


# Module-level so spawn workers can pickle and import this function:
def _init_label_worker_stub_for_tests(label_checkpoint, mcts_cfg_payload):
    """Test-only worker initializer. Installs a deterministic stub labeler
    in the worker process instead of loading a real network. Imported by
    spawned children via its qualified name."""
    from scripts.GPU.alphazero import probe_eval
    from scripts.GPU.alphazero.mcts import MCTSConfig

    def _stub(state, sims, seed):
        return (0.6 + 0.001 * (seed % 7), 0.4 + 0.001 * (seed % 5))

    probe_eval._default_mcts_labeler = _stub
    probe_eval._set_default_labeler_network(object())
    probe_eval._set_default_labeler_mcts_config(MCTSConfig(**mcts_cfg_payload))


def _patch_phase1_extract(monkeypatch, candidates):
    """Bypass real Phase 1 mining; return the given candidate list.
    Mirrors the pattern in tests/test_strong_advantage_probe_suite.py."""
    monkeypatch.setattr(
        "scripts.GPU.alphazero.probe_eval.extract_strong_advantage_candidates",
        lambda games, **kw: (list(candidates), []),
    )


_SAMPLE_CENTRAL = {
    "move_history": [(0, 12), (1, 0), (2, 11), (1, 1)],
    "ply": 4, "winner": "red",
    "category": "chain_advantage_central_red",
    "phase1_features": {
        "cc_size": 12, "cc_axis_span": 0.65, "cc_touches_own_goal": True,
        "axis_span_margin": 0.20, "centroid_chebyshev_from_center": 4,
        "forced_within_2": False,
    },
    "source_game": "iter_0070_game_001", "source_ply": 4,
    "starting_player": "red",
}
_SAMPLE_EDGE = {
    "move_history": [(0, 1), (1, 22), (2, 0), (1, 21)],
    "ply": 4, "winner": "red",
    "category": "chain_advantage_edge_red",
    "phase1_features": {
        "cc_size": 11, "cc_axis_span": 0.60, "cc_touches_own_goal": True,
        "axis_span_margin": 0.15, "centroid_chebyshev_from_center": 10,
        "forced_within_2": False,
    },
    "source_game": "iter_0070_game_002", "source_ply": 4,
    "starting_player": "red",
}


def test_phase2_aggregate_sorts_by_probe_id():
    """Hand-crafted result list (as if returned in arbitrary completion order
    from a process pool) must be aggregated in probe_id-sorted order. This
    test validates the aggregation logic without spawning a real pool."""
    from scripts.build_probe_suite import _phase2_aggregate

    r_a = {
        "probe_id": "p_alpha", "status": "admitted",
        "candidate": {"source_game": "g0", "source_ply": 4, "marker": "alpha"},
        "audit_row": None, "rejection_reason": None,
        "phase2_label": {"mean_root_value": 0.7},
        "error_message": None,
    }
    r_b = {
        "probe_id": "p_beta", "status": "rejected",
        "candidate": {"source_game": "g1", "source_ply": 4, "marker": "beta"},
        "audit_row": {"source_game": "g1", "source_ply": 4,
                      "reason": "magnitude_below_threshold"},
        "rejection_reason": "magnitude_below_threshold",
        "phase2_label": {"mean_root_value": 0.1},
        "error_message": None,
    }
    r_c = {
        "probe_id": "p_gamma", "status": "replay_error",
        "candidate": None,
        "audit_row": {"source_game": "g2", "source_ply": 4,
                      "reason": "replay_error"},
        "rejection_reason": "replay_error",
        "phase2_label": None,
        "error_message": "ValueError: bad",
    }

    # Pass in deliberately scrambled order.
    admitted, audit_rows = _phase2_aggregate([r_c, r_a, r_b])

    assert [c["marker"] for c in admitted] == ["alpha"]
    # Audit rows in probe_id-sorted order: alpha (no row), beta, gamma.
    assert [a["source_game"] for a in audit_rows] == ["g1", "g2"]


def test_phase2_process_pool_smoke(tmp_path, monkeypatch):
    """Real ProcessPoolExecutor smoke test. Substitutes the production
    _init_label_worker with a top-level test function via late-bound
    module-attribute monkeypatch; that function installs a deterministic
    stub labeler in each worker. Verifies the pool launches, all candidates
    label, no exception escapes."""
    import json as _json
    import scripts.build_probe_suite as bps
    from tests.test_probe_phase2_parallel import _init_label_worker_stub_for_tests

    monkeypatch.setattr(bps, "_init_label_worker", _init_label_worker_stub_for_tests)
    _patch_phase1_extract(monkeypatch, [_SAMPLE_CENTRAL, _SAMPLE_EDGE])

    fake_ckpt = tmp_path / "fake_ckpt.safetensors"
    fake_ckpt.write_bytes(b"stub")

    out_path = tmp_path / "strong_advantage_probes.json"
    rc = bps.main_with_args([
        "--tier", "strong_advantage",
        "--input", "scripts/GPU/logs/games",
        "--source-iter-range", "70", "70",
        "--label-checkpoint", str(fake_ckpt),
        "--label-mcts-sims", "10",
        "--label-mcts-repeats", "1",
        "--magnitude-threshold", "0.45",
        "--out", str(out_path),
        "--label-worker-mode", "process",
        "--label-workers", "2",
        "--no-borderline-rerun",
        "--force",
    ])
    assert rc == 0

    draft = _json.loads(out_path.with_suffix(".draft.json").read_text())
    # NOTE: meta.phase2_run_stats is added in Task 6. For Task 4 we only verify
    # the pool ran end-to-end and produced a structurally valid draft.
    assert "meta" in draft
    assert "probes" in draft
    # Re-enable these in Task 6 when meta.phase2_run_stats lands:
    # stats = draft["meta"]["phase2_run_stats"]
    # assert stats["mode"] == "process"
    # assert stats["workers_effective"] == 2
    # assert stats["candidates_total"] == 2
    # assert stats["labeled"] == 2


def test_phase2_process_pool_init_failure(tmp_path, monkeypatch):
    """Worker init can't load the checkpoint -> first .result() raises ->
    main aborts with a clear error mentioning path/mode/workers; no draft
    is written. Uses the synthetic-candidate Phase-1 stub so workers
    actually start (vs. zero candidates -> pool never created)."""
    import scripts.build_probe_suite as bps
    _patch_phase1_extract(monkeypatch, [_SAMPLE_CENTRAL])

    bad_ckpt = tmp_path / "definitely_not_here.safetensors"
    bad_ckpt.write_bytes(b"")  # exists, but won't load as a network

    out_path = tmp_path / "strong_advantage_probes.json"
    rc = bps.main_with_args([
        "--tier", "strong_advantage",
        "--input", "scripts/GPU/logs/games",
        "--source-iter-range", "70", "70",
        "--label-checkpoint", str(bad_ckpt),
        "--label-mcts-sims", "10",
        "--label-mcts-repeats", "1",
        "--out", str(out_path),
        "--label-worker-mode", "process",
        "--label-workers", "2",
        "--force",
    ])
    assert rc != 0
    # Draft file MUST NOT exist on init failure.
    assert not out_path.with_suffix(".draft.json").exists()


def test_phase2_replay_error_isolated(monkeypatch):
    """A candidate whose move_history can't be replayed returns
    status='replay_error' with a populated audit_row and error_message —
    the helper does NOT raise, so the pool survives."""
    from scripts.build_probe_suite import _label_one_strong_advantage_candidate

    bad_cand = {
        "source_game": "g0",
        "source_ply": 5,
        "ply": 5,
        "starting_player": "red",
        "winner": "red",
        "category": "central_red",
        # Out-of-bounds move triggers TwixtState.apply_move to raise.
        "move_history": [(999, 999)],
        "phase1_features": {"cc_size": 1, "cc_axis_span": 0.1,
                            "axis_span_margin": 0.0},
    }
    result = _label_one_strong_advantage_candidate(
        bad_cand,
        label_ckpt_name="fake.safetensors", sims=10, repeats=1,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15,
    )
    assert result["status"] == "replay_error"
    assert result["audit_row"]["reason"] == "replay_error"
    assert result["audit_row"]["source_game"] == "g0"
    assert result["error_message"]
    assert result["candidate"] is None  # spec §7 invariant
    assert result["phase2_label"] is None


def test_phase2_mcts_error_isolated(monkeypatch):
    """A labeler that raises returns status='mcts_error' with a populated
    audit_row and error_message — the helper does NOT raise."""
    from scripts.build_probe_suite import _label_one_strong_advantage_candidate
    from scripts.GPU.alphazero import probe_eval

    def angry_labeler(state, sims, seed):
        raise RuntimeError("synthetic MCTS failure")
    monkeypatch.setattr(probe_eval, "_default_mcts_labeler", angry_labeler)

    cand = {
        "source_game": "g1",
        "source_ply": 18,
        "ply": 18,
        "starting_player": "red",
        "winner": "red",
        "category": "central_red",
        "move_history": [(11, 11)],
        "phase1_features": {"cc_size": 12, "cc_axis_span": 0.7,
                            "axis_span_margin": 0.2},
    }
    result = _label_one_strong_advantage_candidate(
        cand,
        label_ckpt_name="fake.safetensors", sims=10, repeats=1,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15,
    )
    assert result["status"] == "mcts_error"
    assert result["audit_row"]["reason"] == "mcts_error"
    assert "synthetic MCTS failure" in result["error_message"]
    assert result["candidate"] is not None  # spec §7 invariant
    assert result["phase2_label"] is None
