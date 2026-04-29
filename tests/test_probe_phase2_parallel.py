"""Tests for Phase 2 parallel labeling, MCTSConfig wiring, borderline rerun.

See docs/superpowers/specs/2026-04-28-probe-phase2-parallel-labeling-design.md
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

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
    assert flags["label_worker_mode"].default == "process"
    # None is a sentinel meaning "pick based on mode" — main() resolves it
    # post-parse to 10 (process) or 1 (serial). See build_probe_suite.main().
    assert flags["label_workers"].default is None
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


# ----------------------------------------------------------------------
# Task 5: borderline serial-rerun pass
# ----------------------------------------------------------------------


def _make_fake_result(*, status, mean_root_value, min_top1_share=0.5,
                     value_stability=0.0, source_game="g0", source_ply=18,
                     category="chain_advantage_central_red",
                     rejection_reason=None):
    """Build a result-dict shaped like _label_one_strong_advantage_candidate
    returns. probe_id is derived from (source_game, source_ply, category) the
    same way `_probe_id_for(cand)` derives it, so when these results are fed
    back through `_run_borderline_reruns` (which re-runs the helper and
    re-derives probe_id from the candidate), the spec §9 same-probe-id
    invariant holds."""
    from scripts.build_probe_suite import _probe_id_for

    label = {
        "mean_root_value": mean_root_value,
        "value_per_run": [mean_root_value, mean_root_value],
        "value_stability": value_stability,
        "min_top1_share": min_top1_share,
        "label_mcts_sims": 10,
        "label_mcts_repeats": 2,
        "rng_seed_base": 0,
        "label_checkpoint": "fake.safetensors",
    }
    cand = {
        "source_game": source_game, "source_ply": source_ply, "ply": source_ply,
        "starting_player": "red", "winner": "red",
        "category": category, "move_history": [(11, 11)],
        "phase1_features": {
            "cc_size": 12, "cc_axis_span": 0.7, "axis_span_margin": 0.2,
        },
        "phase2_label": label,
    }
    probe_id = _probe_id_for(cand)
    audit_row = None
    if status == "rejected":
        audit_row = {
            "source_game": cand["source_game"],
            "source_ply": cand["source_ply"],
            "phase1_features": cand["phase1_features"],
            "phase2_label": label,
            "reason": rejection_reason,
        }
    return {
        "probe_id": probe_id,
        "status": status,
        "candidate": cand,
        "audit_row": audit_row,
        "rejection_reason": rejection_reason,
        "phase2_label": label,
        "error_message": None,
    }


def test_borderline_rerun_admitted_to_rejected(monkeypatch):
    """Parallel result was admitted at just-above magnitude. Main-process
    rerun returns just-below. Status flips to rejected; audit metadata
    records the flip with old/new reasons."""
    from scripts.build_probe_suite import _run_borderline_reruns
    from scripts.GPU.alphazero import probe_eval

    def stub(state, sims, seed):
        # Main-process rerun: just-below threshold.
        return (0.449, 0.5)
    monkeypatch.setattr(probe_eval, "_default_mcts_labeler", stub)

    results = [_make_fake_result(status="admitted", mean_root_value=0.451)]
    counters = _run_borderline_reruns(
        results, epsilon=0.01,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15,
        label_ckpt_name="fake.safetensors", sims=10, repeats=2,
    )

    assert counters["candidates"] == 1
    assert counters["reruns"] == 1
    assert counters["flips"] == 1
    assert counters["seconds"] >= 0
    r = results[0]
    assert r["status"] == "rejected"
    assert r["audit_row"] is not None
    assert r["audit_row"]["borderline_rerun"] is True
    assert r["audit_row"]["borderline_rerun_flipped"] is True
    assert r["audit_row"]["parallel_admission_reason"] == "admitted"
    assert "magnitude" in r["audit_row"]["borderline_rerun_reason"]


def test_borderline_rerun_rejected_to_admitted(monkeypatch):
    """Symmetric flip: parallel rejected at just-below; main rerun just-above."""
    from scripts.build_probe_suite import _run_borderline_reruns
    from scripts.GPU.alphazero import probe_eval

    def stub(state, sims, seed):
        return (0.451, 0.5)
    monkeypatch.setattr(probe_eval, "_default_mcts_labeler", stub)

    results = [_make_fake_result(
        status="rejected", mean_root_value=0.449,
        rejection_reason="magnitude_below_threshold",
    )]
    counters = _run_borderline_reruns(
        results, epsilon=0.01,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15,
        label_ckpt_name="fake.safetensors", sims=10, repeats=2,
    )

    assert counters["flips"] == 1
    r = results[0]
    assert r["status"] == "admitted"
    assert r["audit_row"] is None
    # Rerun audit metadata lives on the candidate object so downstream
    # selector audit rows can merge it.
    rerun_meta = r["candidate"].get("_borderline_rerun_audit")
    assert rerun_meta is not None
    assert rerun_meta["borderline_rerun_flipped"] is True
    assert rerun_meta["parallel_admission_reason"] == "magnitude_below_threshold"
    assert rerun_meta["serial_rerun_admission_reason"] == "admitted"


def test_borderline_rerun_not_triggered_for_non_borderline(monkeypatch):
    """Candidates clearly admitted (mean=0.9) or clearly rejected (mean=0.1)
    are far from threshold and never trigger a rerun."""
    from scripts.build_probe_suite import _run_borderline_reruns
    from scripts.GPU.alphazero import probe_eval

    def stub(state, sims, seed):
        raise AssertionError("stub should not be called for non-borderline cands")
    monkeypatch.setattr(probe_eval, "_default_mcts_labeler", stub)

    results = [
        _make_fake_result(status="admitted", mean_root_value=0.9,
                          source_ply=18),
        _make_fake_result(
            status="rejected", mean_root_value=0.1, source_ply=22,
            rejection_reason="magnitude_below_threshold",
        ),
    ]
    counters = _run_borderline_reruns(
        results, epsilon=0.01,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15,
        label_ckpt_name="fake.safetensors", sims=10, repeats=2,
    )

    assert counters["candidates"] == 0
    assert counters["reruns"] == 0
    assert counters["flips"] == 0
    # Statuses unchanged.
    assert results[0]["status"] == "admitted"
    assert results[1]["status"] == "rejected"


def test_borderline_rerun_preserves_probe_id(monkeypatch):
    """Spec §9 invariant: rerun must produce a result with the same probe_id
    as the parallel result. The helper recomputes probe_id from
    source_game/source_ply, so any rerun-side mutation that changes those
    fields would surface here."""
    from scripts.build_probe_suite import _run_borderline_reruns
    from scripts.GPU.alphazero import probe_eval

    def stub(state, sims, seed):
        return (0.5, 0.5)
    monkeypatch.setattr(probe_eval, "_default_mcts_labeler", stub)

    results = [_make_fake_result(
        status="admitted", mean_root_value=0.451,
    )]
    pre_id = results[0]["probe_id"]
    _run_borderline_reruns(
        results, epsilon=0.01,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15,
        label_ckpt_name="fake.safetensors", sims=10, repeats=2,
    )
    assert results[0]["probe_id"] == pre_id


def test_borderline_rerun_skips_error_results(monkeypatch):
    """replay_error and mcts_error results have phase2_label=None and must
    be excluded from the borderline check (no IndexError, no rerun)."""
    from scripts.build_probe_suite import _run_borderline_reruns
    from scripts.GPU.alphazero import probe_eval

    def stub(state, sims, seed):
        raise AssertionError("stub should not be called for error rows")
    monkeypatch.setattr(probe_eval, "_default_mcts_labeler", stub)

    results = [
        {
            "probe_id": "p_replay_err",
            "status": "replay_error",
            "candidate": None,
            "audit_row": {"reason": "replay_error"},
            "rejection_reason": "replay_error",
            "phase2_label": None,
            "error_message": "ValueError: bad",
        },
        {
            "probe_id": "p_mcts_err",
            "status": "mcts_error",
            "candidate": {"source_game": "g0", "source_ply": 1},
            "audit_row": {"reason": "mcts_error"},
            "rejection_reason": "mcts_error",
            "phase2_label": None,
            "error_message": "RuntimeError: kaboom",
        },
    ]
    counters = _run_borderline_reruns(
        results, epsilon=0.01,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15,
        label_ckpt_name="fake.safetensors", sims=10, repeats=2,
    )
    assert counters["candidates"] == 0
    assert counters["reruns"] == 0
    assert counters["flips"] == 0


# ----------------------------------------------------------------------
# Task 5 Step 2: integration tests for the disable paths.
#
# Plan adaptation (Issue A): meta.phase2_run_stats is NOT added until
# Task 6, so these tests cannot read borderline_rerun_enabled / counters
# off the draft JSON yet. Instead, they install a spy on
# `_run_borderline_reruns` and assert on the spy's invocation log.
# Task 6 will replace the spy approach with real meta.phase2_run_stats
# assertions.
# ----------------------------------------------------------------------


def _run_run_strong_advantage_with_stub(
    tmp_path, monkeypatch, *, mode, epsilon, no_rerun,
    stub=None, candidates=None, init_worker_stub: bool = False,
):
    """Run main_with_args(...) with a Phase-1 candidate stub and a
    deterministic Phase-2 stub labeler. Returns the parsed draft JSON.

    Plan adaptation (Issue B): the original plan referenced a
    `_write_minimal_strong_advantage_game_fixture` helper that does not
    exist. Task 4 used `_patch_phase1_extract` + `_SAMPLE_*` fixtures
    instead — we follow that pattern.

    For mode="process", the worker process loads the labeler network via
    `_init_label_worker(checkpoint, ...)`. When the checkpoint is a stub
    file, that load fails — pass `init_worker_stub=True` to substitute
    the test-only `_init_label_worker_stub_for_tests`.
    """
    from scripts.GPU.alphazero import probe_eval
    import scripts.build_probe_suite as bps
    import json as _json

    if candidates is None:
        candidates = [_SAMPLE_CENTRAL]
    _patch_phase1_extract(monkeypatch, candidates)

    if stub is None:
        def stub(state, sims, seed):
            return (0.6, 0.4)
    monkeypatch.setattr(probe_eval, "_default_mcts_labeler", stub)

    # The main-process network is loaded when mode=serial OR when
    # rerun_enabled in process mode. Provide a fake with .eval() for
    # both code paths.
    class _FakeNet:
        def eval(self):
            return self

    monkeypatch.setattr(probe_eval, "load_network_for_scoring",
                        lambda p: (_FakeNet(), 30, 128, 6))
    monkeypatch.setattr(probe_eval, "_set_default_labeler_network",
                        lambda *a, **kw: None)
    monkeypatch.setattr(probe_eval, "_set_default_labeler_mcts_config",
                        lambda *a, **kw: None)

    if init_worker_stub:
        monkeypatch.setattr(bps, "_init_label_worker",
                            _init_label_worker_stub_for_tests)

    fake_ckpt = tmp_path / "fake_ckpt.safetensors"
    fake_ckpt.write_bytes(b"stub")

    target = tmp_path / "out.json"
    cli = [
        "--tier", "strong_advantage",
        "--input", "scripts/GPU/logs/games",
        "--source-iter-range", "70", "70",
        "--label-checkpoint", str(fake_ckpt),
        "--label-mcts-sims", "10",
        "--label-mcts-repeats", "2",
        "--out", str(target),
        "--label-worker-mode", mode,
        "--admission-borderline-epsilon", str(epsilon),
        "--force",
    ]
    if mode == "process":
        cli.extend(["--label-workers", "2"])
    if no_rerun:
        cli.append("--no-borderline-rerun")

    rc = bps.main_with_args(cli)
    assert rc == 0
    return _json.loads(target.with_suffix(".draft.json").read_text())


def test_borderline_rerun_disabled(tmp_path, monkeypatch):
    """--no-borderline-rerun disables the pass even when epsilon > 0 and
    mode=process. _run_borderline_reruns is never called.

    NOTE: meta.phase2_run_stats verification deferred to Task 6 — we use
    a spy on `_run_borderline_reruns` here.
    """
    import scripts.build_probe_suite as bps
    rerun_call_count = []

    def spy(results, **kwargs):
        rerun_call_count.append(1)
        return {"candidates": 0, "reruns": 0, "flips": 0, "seconds": 0.0}
    monkeypatch.setattr(bps, "_run_borderline_reruns", spy)

    def stub(state, sims, seed):
        return (0.451, 0.5)

    _run_run_strong_advantage_with_stub(
        tmp_path, monkeypatch, mode="process", epsilon=0.01,
        no_rerun=True, stub=stub, init_worker_stub=True,
    )
    assert rerun_call_count == []  # spy never called


def test_borderline_rerun_serial_mode_no_op(tmp_path, monkeypatch):
    """Serial mode never runs the borderline pass even with epsilon > 0.

    NOTE: meta.phase2_run_stats verification deferred to Task 6.
    """
    import scripts.build_probe_suite as bps
    rerun_call_count = []

    def spy(results, **kwargs):
        rerun_call_count.append(1)
        return {"candidates": 0, "reruns": 0, "flips": 0, "seconds": 0.0}
    monkeypatch.setattr(bps, "_run_borderline_reruns", spy)

    def stub(state, sims, seed):
        return (0.451, 0.5)

    _run_run_strong_advantage_with_stub(
        tmp_path, monkeypatch, mode="serial", epsilon=0.01,
        no_rerun=False, stub=stub,
    )
    assert rerun_call_count == []


def test_admission_borderline_epsilon_zero_disables(tmp_path, monkeypatch):
    """Epsilon=0 disables the rerun pass even in process mode.

    NOTE: meta.phase2_run_stats verification deferred to Task 6.
    """
    import scripts.build_probe_suite as bps
    rerun_call_count = []

    def spy(results, **kwargs):
        rerun_call_count.append(1)
        return {"candidates": 0, "reruns": 0, "flips": 0, "seconds": 0.0}
    monkeypatch.setattr(bps, "_run_borderline_reruns", spy)

    def stub(state, sims, seed):
        return (0.451, 0.5)

    _run_run_strong_advantage_with_stub(
        tmp_path, monkeypatch, mode="process", epsilon=0.0,
        no_rerun=False, stub=stub, init_worker_stub=True,
    )
    assert rerun_call_count == []


def test_committed_probes_have_no_private_rerun_keys(tmp_path, monkeypatch):
    """Spec §9: the committed probe JSON must contain no
    `_borderline_rerun_audit` or `parallel_phase2_label_before_rerun`
    keys, even when reruns happened.

    Synthesizes the post-rerun condition by monkeypatching
    `_run_borderline_reruns` to attach the private key to every
    admitted candidate, then asserts the serializer strips it.
    """
    import scripts.build_probe_suite as bps

    def fake_rerun(results, **kwargs):
        for r in results:
            if r["candidate"] is not None:
                r["candidate"]["_borderline_rerun_audit"] = {
                    "borderline_rerun": True,
                    "borderline_rerun_reason": ["magnitude"],
                    "parallel_phase2_label_before_rerun": {"sentinel": True},
                    "borderline_rerun_flipped": False,
                }
        return {"candidates": len(results), "reruns": len(results),
                "flips": 0, "seconds": 0.0}

    monkeypatch.setattr(bps, "_run_borderline_reruns", fake_rerun)

    _run_run_strong_advantage_with_stub(
        tmp_path, monkeypatch, mode="process", epsilon=0.01,
        no_rerun=False, init_worker_stub=True,
    )
    serialized = (tmp_path / "out.draft.json").read_text()
    assert "_borderline_rerun_audit" not in serialized
    assert "parallel_phase2_label_before_rerun" not in serialized


def test_phase2_run_stats_recorded_in_meta(tmp_path, monkeypatch):
    """meta.phase2_run_stats records all the documented fields, including
    workers_requested vs workers_effective and borderline_rerun_enabled.
    Use --label-worker-mode=serial --label-workers=4 to exercise the
    serial-mode clamp path."""
    import json as _json
    import scripts.build_probe_suite as bps

    _patch_phase1_extract(monkeypatch, [_SAMPLE_CENTRAL])

    def stub(state, sims, seed):
        return (0.7, 0.4)
    from scripts.GPU.alphazero import probe_eval
    monkeypatch.setattr(probe_eval, "_default_mcts_labeler", stub)

    # Serial mode loads a main-process labeler network; provide a fake
    # with an .eval() method (matches the helper pattern used by the
    # other tests in this file).
    class _FakeNet:
        def eval(self):
            return self

    monkeypatch.setattr(probe_eval, "load_network_for_scoring",
                        lambda p: (_FakeNet(), 30, 128, 6))
    monkeypatch.setattr(probe_eval, "_set_default_labeler_network",
                        lambda *a, **kw: None)
    monkeypatch.setattr(probe_eval, "_set_default_labeler_mcts_config",
                        lambda *a, **kw: None)

    fake_ckpt = tmp_path / "fake_ckpt.safetensors"
    fake_ckpt.write_bytes(b"stub")
    out_path = tmp_path / "out.json"

    rc = bps.main_with_args([
        "--tier", "strong_advantage",
        "--input", "scripts/GPU/logs/games",
        "--source-iter-range", "70", "70",
        "--label-checkpoint", str(fake_ckpt),
        "--label-mcts-sims", "10",
        "--label-mcts-repeats", "1",
        "--magnitude-threshold", "0.45",
        "--out", str(out_path),
        "--label-worker-mode", "serial",
        "--label-workers", "4",   # under serial, should clamp to 1 with warning
        "--force",
    ])
    assert rc == 0

    draft = _json.loads(out_path.with_suffix(".draft.json").read_text())
    stats = draft["meta"]["phase2_run_stats"]
    assert stats["mode"] == "serial"
    assert stats["workers_requested"] == 4
    assert stats["workers_effective"] == 1
    assert stats["eval_batch_size"] == 14
    assert stats["stall_flush_sims"] == 16
    assert stats["candidates_total"] == 1
    assert stats["labeled"] == 1
    assert stats["admitted_before_diversity"] == 1
    assert stats["rejected"] == 0
    assert stats["replay_errors"] == 0
    assert stats["mcts_errors"] == 0
    assert stats["borderline_rerun_enabled"] is False  # serial mode
    assert stats["admission_borderline_epsilon"] == 0.01
    assert stats["borderline_candidates"] == 0
    assert stats["borderline_reruns"] == 0
    assert stats["borderline_flips"] == 0
    assert "rejection_reasons" in stats
    assert isinstance(stats["seconds_total"], (int, float))
    assert stats["seconds_total"] >= 0


def test_phase2_serial_unchanged(tmp_path, monkeypatch):
    """Running with default (serial) flags on a frozen synthetic candidate set
    produces output byte-identical to the committed golden fixture. Guards
    against accidental drift in serial-mode Phase 2 + selector + JSON-write
    behavior."""
    from scripts.GPU.alphazero import probe_eval
    from scripts.build_probe_suite import _build_arg_parser, _run_strong_advantage
    import json as _json

    repo_root = Path(__file__).resolve().parent.parent
    golden_dir = repo_root / "tests" / "probes" / "golden"
    expected_draft_path = golden_dir / "phase2_serial_tiny_expected.draft.json"
    expected_audit_path = golden_dir / "phase2_serial_tiny_expected.audit.json"

    class _FakeNet:
        def eval(self):
            return self

    def stub(state, sims, seed):
        v = 0.6 + 0.001 * (seed % 7)
        t1 = 0.4 + 0.001 * (seed % 5)
        return (v, t1)

    _patch_phase1_extract(monkeypatch, [_SAMPLE_CENTRAL, _SAMPLE_EDGE])
    monkeypatch.setattr(probe_eval, "_default_mcts_labeler", stub)
    monkeypatch.setattr(probe_eval, "load_network_for_scoring",
                        lambda p: (_FakeNet(), 30, 128, 6))
    monkeypatch.setattr(probe_eval, "_set_default_labeler_network",
                        lambda *a, **kw: None)
    monkeypatch.setattr(probe_eval, "_set_default_labeler_mcts_config",
                        lambda *a, **kw: None)

    fake_ckpt = tmp_path / "fake_ckpt.safetensors"
    fake_ckpt.write_bytes(b"stub")
    target = tmp_path / "out.json"

    ap = _build_arg_parser()
    args = ap.parse_args([
        "--tier", "strong_advantage",
        "--input", "scripts/GPU/logs/games",
        "--source-iter-range", "70", "70",
        "--label-checkpoint", str(fake_ckpt),
        "--label-mcts-sims", "10",
        "--label-mcts-repeats", "2",
        "--magnitude-threshold", "0.45",
        "--out", str(target),
        "--label-worker-mode", "serial",
        "--force",
    ])
    # Mirror main()'s post-parse default resolution for --label-workers,
    # since this test bypasses main() and calls _run_strong_advantage directly.
    if args.label_workers is None:
        args.label_workers = 10 if args.label_worker_mode == "process" else 1
    args.label_workers_requested = args.label_workers
    rc = _run_strong_advantage(args)
    assert rc == 0

    def _normalize(blob):
        """Strip non-deterministic / environment-dependent fields before
        the byte-comparison:
        - seconds_total, borderline_rerun_seconds: wall-clock timings.
        - selection_rules.label_checkpoint: absolute path that varies
          between generator (golden_dir) and test (tmp_path).
        - selection_rules.label_checkpoint_sha256: SHA of the fake-blob
          checkpoint, which changes if anyone alters the test stub.
        """
        d = _json.loads(blob)
        stats = d.get("meta", {}).get("phase2_run_stats", {})
        stats.pop("seconds_total", None)
        stats.pop("borderline_rerun_seconds", None)
        sr = d.get("meta", {}).get("selection_rules", {})
        sr.pop("label_checkpoint", None)
        sr.pop("label_checkpoint_sha256", None)
        return _json.dumps(d, indent=2, sort_keys=True)

    actual_draft = (tmp_path / "out.draft.json").read_text()
    expected_draft = expected_draft_path.read_text()
    assert _normalize(actual_draft) == _normalize(expected_draft), (
        "Phase 2 serial draft drifted from golden fixture. "
        "If intentional, regenerate via "
        "scripts/probes/generate_golden_phase2_serial_fixture.py"
    )

    actual_audit = (tmp_path / "candidates_strong_advantage.json").read_text()
    expected_audit = expected_audit_path.read_text()
    assert actual_audit == expected_audit, (
        "Phase 2 serial audit drifted from golden fixture."
    )
