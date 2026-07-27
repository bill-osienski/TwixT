"""Tests for the committed v17 same-checkpoint match-smoke driver (Task 8C).

The driver is result-determining — it picks the agent configs, the seeds and
the colour schedule — so it is tested like any other production module, and its
SHA-1 is recorded in the protocol it emits.

GPU-free throughout: every run goes through the fake evaluator factory.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os

import pytest

from scripts.GPU.alphazero import fpu_v17_match_smoke as smoke
from scripts.GPU.alphazero import fpu_v17_provenance as prov
from scripts.GPU.alphazero.eval_runner import cfg_from
from tests.eval_fakes import FakeEvaluator


def _recording_factory(path):
    _recording_factory.calls.append(path)
    return FakeEvaluator()


_recording_factory.calls = []


@pytest.fixture
def tiny(monkeypatch):
    """Same driver, a board small enough to play in-process."""
    real = smoke.eval_config

    def small(**kw):
        kw.setdefault("board_size", 8)
        kw.setdefault("mcts_sims", 8)
        kw.setdefault("max_moves", 6)
        return real(**kw)

    monkeypatch.setattr(smoke, "eval_config", small)
    return small


# --------------------------------------------------------------------------
# Frozen §5.4 block table
# --------------------------------------------------------------------------

def test_blocks_match_the_frozen_design_seed_ranges():
    names = [b["name"] for b in smoke.BLOCKS]
    assert names == ["shipped_vs_shipped", "shipped_vs_r035"]
    spans = [(b["base_seed"], b["base_seed"] + b["games"]) for b in smoke.BLOCKS]
    assert spans == [(20309100, 20309104), (20309104, 20309108)]
    # Contiguous, and exactly the declared match-smoke range.
    assert (spans[0][0], spans[-1][1]) == (
        prov.MATCH_SMOKE_SEEDS[0],
        prov.MATCH_SMOKE_SEEDS[0] + prov.MATCH_SMOKE_SEEDS[1])


def test_candidate_is_a_frozen_grid_point():
    assert smoke.CANDIDATE in prov.GRID


def test_eval_config_carries_the_frozen_batching_triple():
    cfg = cfg_from(smoke.eval_config())
    assert prov.validate_batching(cfg) == prov.BATCHING
    assert cfg.n_simulations == prov.MCTS_SIMS


def test_agent_configs_differ_only_in_the_coefficient():
    base = cfg_from(smoke.eval_config())
    a, b = smoke.agent_config(base, 0.0), smoke.agent_config(base, 0.35)
    da, db = dataclasses.asdict(a), dataclasses.asdict(b)
    assert {k for k in da if da[k] != db[k]} == {smoke.V17_FIELD}
    for cfg in (a, b):
        assert prov.validate_batching(cfg) == prov.BATCHING


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

def test_labels_are_complete_and_forbid_interpretation():
    assert smoke.labels_for() == {"run_kind": "tooling_smoke",
                                  "scientific_interpretation_forbidden": True}


def test_labels_track_the_run_kind_rather_than_being_hardcoded():
    assert smoke.labels_for("development")["scientific_interpretation_forbidden"] is False
    with pytest.raises(prov.ProtocolViolation):
        smoke.labels_for("not_a_run_kind")


def test_driver_sha1_is_this_files_bytes():
    with open(smoke.__file__, "rb") as fh:
        assert smoke.driver_sha1() == hashlib.sha1(fh.read()).hexdigest()


# --------------------------------------------------------------------------
# Colour / config schedule
# --------------------------------------------------------------------------

@pytest.mark.parametrize("idx", [0, 1])
def test_each_block_has_an_exact_2_2_colour_split(idx):
    base = cfg_from(smoke.eval_config())
    blk = smoke.build_blocks(base)[idx]
    tasks = smoke.tasks_for(blk)
    assert [t.seed for t in tasks] == list(
        range(blk["base_seed"], blk["base_seed"] + blk["games"]))
    red_a = sum(1 for t in tasks if t.red_agent == blk["a_id"])
    assert red_a == blk["games"] // 2
    by_agent = {blk["a_id"]: blk["a_cfg"], blk["b_id"]: blk["b_cfg"]}
    assert all(t.red_mcts == by_agent[t.red_agent] for t in tasks)
    assert all(t.red_checkpoint == t.black_checkpoint == smoke.CKPT
               for t in tasks)


def test_asymmetric_block_swaps_the_coefficient_with_the_colour_schedule():
    base = cfg_from(smoke.eval_config())
    blk = smoke.build_blocks(base)[1]
    assert blk["a_r"] != blk["b_r"]
    for t in smoke.tasks_for(blk):
        expected = blk["a_r"] if t.game_idx % 2 == 0 else blk["b_r"]
        assert getattr(t.red_mcts, smoke.V17_FIELD) == expected


def test_symmetric_block_agents_are_distinct_identities_with_equal_configs():
    base = cfg_from(smoke.eval_config())
    blk = smoke.build_blocks(base)[0]
    assert blk["a_id"] != blk["b_id"]
    assert blk["a_cfg"] == blk["b_cfg"]      # identity is the only difference


# --------------------------------------------------------------------------
# The negative call-site proof
# --------------------------------------------------------------------------

def test_negative_call_site_proof_refuses_without_loading_or_writing(tmp_path,
                                                                     monkeypatch):
    """Finding 3: exercise the ACTUAL call site with invalid batching."""
    monkeypatch.setattr(smoke, "OUTPUT_DIR", str(tmp_path))
    _recording_factory.calls.clear()
    ev = smoke.negative_call_site_proof(_recording_factory)
    assert ev["refused"] is True
    assert ev["evaluator_loads"] == 0
    # Complete: no summary, no companion _games.jsonl, no replay sidecar.
    assert ev["files_written"] == []
    assert ev["output_written"] is False
    assert "batching triple" in ev["reason"]


def test_negative_proof_is_not_vacuous(tmp_path, monkeypatch, tiny):
    """The same call site with a VALID triple must load an evaluator and write
    output — otherwise the zero-loads assertion above proves nothing."""
    monkeypatch.setattr(smoke, "OUTPUT_DIR", str(tmp_path))
    _recording_factory.calls.clear()
    base = cfg_from(smoke.eval_config())
    blk = smoke.build_blocks(base)[1]
    smoke.run_block(blk, smoke.eval_config(), str(tmp_path),
                    evaluator_factory=_recording_factory)
    assert _recording_factory.calls == [smoke.CKPT]
    # The valid path writes BOTH files the negative proof asserts absent, so
    # `files_written == []` above is a real observation, not a vacuous one.
    assert os.path.exists(tmp_path / f"{blk['name']}.json")
    assert os.path.exists(tmp_path / f"{blk['name']}_games.jsonl")


def test_negative_proof_agents_agree_with_their_wrong_base(tmp_path, monkeypatch):
    """The invalid-batching case must be caught by the frozen-triple validator
    specifically, not by agent-consistency: both agents agree with the wrong
    base, so consistency alone would pass it."""
    from scripts.GPU.alphazero.eval_runner import (
        require_agent_config_consistency,
    )
    monkeypatch.setattr(smoke, "OUTPUT_DIR", str(tmp_path))
    wrong = smoke.eval_config(mcts_eval_batch_size=16, mcts_stall_flush_sims=16)
    base = cfg_from(wrong)
    require_agent_config_consistency(
        base, smoke.agent_config(base, 0.0),
        smoke.agent_config(base, smoke.CANDIDATE),
        allow_differ={smoke.V17_FIELD})          # passes -> validator is the catcher
    with pytest.raises(prov.ProtocolViolation):
        prov.validate_batching(base)


# --------------------------------------------------------------------------
# End-to-end, GPU-free
# --------------------------------------------------------------------------

def test_run_smoke_stamps_every_artifact(tmp_path, tiny, monkeypatch):
    monkeypatch.setattr(prov, "validate_output_path", lambda p: str(p))
    report = smoke.run_smoke(str(tmp_path), evaluator_factory=_recording_factory)

    assert report["run_kind"] == "tooling_smoke"
    assert report["scientific_interpretation_forbidden"] is True
    assert report["driver_sha1"] == smoke.driver_sha1()

    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["scientific_interpretation_forbidden"] is True
    assert cfg["run_kind"] == "tooling_smoke"

    for blk in smoke.BLOCKS:
        summary = json.loads((tmp_path / f"{blk['name']}.json").read_text())
        assert summary["run_kind"] == "tooling_smoke"
        assert summary["scientific_interpretation_forbidden"] is True
        assert summary["self_match"] is False
        rows = [json.loads(l) for l in
                (tmp_path / f"{blk['name']}_games.jsonl").read_text().splitlines()]
        assert len(rows) == blk["games"]
        for row in rows:
            assert row["run_kind"] == "tooling_smoke"
            assert row["scientific_interpretation_forbidden"] is True


def test_protocol_records_the_driver_among_its_source_files():
    """Finding 2: the driver is result-determining, so it must appear in the
    protocol's own provenance."""
    protocol = smoke.build_smoke_protocol()
    recorded = json.dumps(protocol["provenance"])
    assert "fpu_v17_match_smoke.py" in recorded
    assert protocol["extra"]["driver_sha1"] == smoke.driver_sha1()
    assert protocol["run_kind"] == "tooling_smoke"
    assert (protocol["base_seed"], protocol["games"]) == prov.MATCH_SMOKE_SEEDS


@pytest.mark.parametrize("flags,label", [
    (["--mcts-eval-batch-size", "16", "--mcts-stall-flush-sims", "48",
      "--mcts-pending-virtual-visits", "8"], "(16,48,8)"),
    (["--mcts-eval-batch-size", "14", "--mcts-stall-flush-sims", "16",
      "--mcts-pending-virtual-visits", "8"], "(14,16,8)"),
    (["--mcts-eval-batch-size", "14", "--mcts-stall-flush-sims", "48",
      "--mcts-pending-virtual-visits", "4"], "(14,48,4)"),
    (["--mcts-eval-batch-size", "14", "--mcts-stall-flush-sims", "48"],
     "pending visits omitted"),
])
def test_real_cli_refuses_wrong_batching_before_checkpoint_resolution(flags, label):
    """Exercises the REAL generation CLI. Both the effective value and (under
    the old design) the bar move together; the bar now comes from the frozen
    authority, so these cannot pass. A deliberately absent checkpoint proves
    the refusal precedes checkpoint resolution: a passing case reaches
    'checkpoint not found', a refused one never does."""
    import subprocess
    r = subprocess.run(
        [".venv/bin/python", "-m", "scripts.GPU.alphazero.eval_checkpoint_match",
         "--checkpoint-a", "no_such_ckpt", "--checkpoint-b", "no_such_ckpt",
         "--output", "/tmp/v17_must_not_exist.json", "--require-v17-batching",
         *flags],
        capture_output=True, text=True,
        cwd="/Users/bill/projects/TwixT_Game")
    out = r.stdout + r.stderr
    assert r.returncode != 0, label
    assert "checkpoint not found" not in out, f"{label}: reached checkpoint resolution"
    assert ("FROZEN v17 triple" in out
            or "requires --mcts-pending-virtual-visits" in out), out[-200:]


def test_real_cli_accepts_the_frozen_triple():
    """Counterpart: the valid triple passes the bar and goes on to resolve
    checkpoints — so the refusals above are not vacuous."""
    import subprocess
    r = subprocess.run(
        [".venv/bin/python", "-m", "scripts.GPU.alphazero.eval_checkpoint_match",
         "--checkpoint-a", "no_such_ckpt", "--checkpoint-b", "no_such_ckpt",
         "--output", "/tmp/v17_must_not_exist.json", "--require-v17-batching",
         "--mcts-eval-batch-size", "14", "--mcts-stall-flush-sims", "48",
         "--mcts-pending-virtual-visits", "8"],
        capture_output=True, text=True,
        cwd="/Users/bill/projects/TwixT_Game")
    assert "checkpoint not found" in (r.stdout + r.stderr)
