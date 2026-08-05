"""Stage 5, Task 3 -- the operator CLI, stop conditions and exit-status sidecars.

The launchable commands are driven end to end against a patched FakeEvaluator
factory over real production-schema blocks, artifacts and both sidecars.
No MLX, no checkpoint load, no reservoir.
"""
import hashlib
import json
import subprocess
import sys

import pytest

from scripts.GPU.alphazero.run_atlas import (
    EXIT_ABORTED, EXIT_OK, EXIT_PROVENANCE, EXIT_USAGE, STOP_CONDITIONS,
    launch_wrapper, main as run_atlas_main, measure_provenance,
    write_status_sidecar,
)

from tests.eval_fakes import FakeEvaluator

BASE = 20500000
SAMPLING_SEED = 20260805


def _cli(*args):
    return subprocess.run([sys.executable, "-m",
                           "scripts.GPU.alphazero.run_atlas", *args],
                          capture_output=True, text=True)


def _fake_ck(tmp_path):
    """A real FILE, so preflight has something to hash. Never loaded: the
    factory is patched, and only its digest is used."""
    ck = tmp_path / "net.safetensors"
    if not ck.exists():
        ck.write_bytes(b"atlas-stage5-fake-checkpoint")
    return ck


def _fixture_prov(ck):
    """ONE provenance object for the whole fixture chain.

    Symmetric validation compares the measured digest and HEAD against the
    manifests AND the pilot artifact, so a fixture that writes "0"*40 in the
    artifact while the manifests carry the real digest is rejected before
    run-final starts -- and the test would be asserting on a provenance
    failure it did not mean to create.
    """
    return {"git_head": "a" * 40, "worktree_clean": True,
            "checkpoint_path": str(ck),
            "checkpoint_sha1": hashlib.sha1(ck.read_bytes()).hexdigest()}


def _late_history():
    from tests.test_atlas_run import _late_history as h
    return h()


def _fake_block(tmp_path, name="pilot", n_games=24, start_index=0,
                checkpoint=None):
    """A block directory carrying the FULL PRODUCTION manifest.

    `load_block` verifies PRODUCTION_SETTINGS -- board 24, 400 simulations, 280
    max moves, batching (14, 48, 8), noise ON -- plus clean provenance, exact
    filenames, exact index coverage and `seed == base_seed + game_idx`. A
    manifest claiming active_size=6 could never pass it, and a fixture that
    cannot pass the real loader qualifies nothing.
    """
    from scripts.GPU.alphazero.generate_atlas_reservoir import (
        MANIFEST, seed_for_index,
    )
    ck = checkpoint or _fake_ck(tmp_path)
    prov = _fixture_prov(ck)          # the ONE fixture provenance object
    hist = _late_history()
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / MANIFEST).write_text(json.dumps({
        "base_seed": BASE, "start_index": start_index, "n_games": n_games,
        "seed_range": [seed_for_index(BASE, start_index),
                       seed_for_index(BASE, start_index + n_games)],
        "n_simulations": 400, "max_moves": 280, "active_size": 24,
        "batching": [14, 48, 8], "add_noise": True,
        **prov,
    }, indent=2, sort_keys=True))
    for i in range(start_index, start_index + n_games):
        (d / f"game_{i:06d}.json").write_text(json.dumps({
            "game_idx": i, "seed": seed_for_index(BASE, i),
            "start_player": "red", "n_moves": len(hist),
            "winner": None, "draw_reason": "state_cap",
            "move_history": [list(m) for m in hist]}))
    return d


def _authoritative_pilot_artifact(ck, n=200):
    """A complete, sized pilot artifact as `emit` would have written one, with
    SCHEMA-VALID rows: run-final recomposes all three read-outs over them."""
    from scripts.GPU.alphazero.atlas_artifact import emit
    from tests.test_atlas_run import _measured_pilot_rows
    return emit({"rows": _measured_pilot_rows(),
                 "provenance": _fixture_prov(ck),      # the SAME object
                 "mode": "pilot", "verdict": "OK", "authoritative": True,
                 "sampling_seed": SAMPLING_SEED,
                 "sizing": {"p_m": 8 / 24, "p_s": 9 / 24, "verdict": "OK",
                            "N": n, "required": n}})


def _assigned_176(tmp_path, cont_block):
    """`assign_corpus`'s real output over the real block -- not hand-written.
    run-final recomputes this, so a hand-written file would only ever test that
    the recomputation rejects it."""
    from scripts.GPU.alphazero.corpus_geometry import assign_corpus
    from scripts.GPU.alphazero.generate_atlas_reservoir import load_block
    from tests.test_atlas_run import _pilot_assignment
    games = load_block(cont_block, BASE, 24, 216)
    rows = assign_corpus(_pilot_assignment(), games, 200, SAMPLING_SEED)["rows"]
    p = tmp_path / "assign.json"
    p.write_text(json.dumps({"verdict": "OK", "rows": rows}, sort_keys=True))
    return p


def _patch_factory(monkeypatch):
    import scripts.GPU.alphazero.eval_runner as er
    monkeypatch.setattr(er, "_default_evaluator_factory",
                        lambda _p: FakeEvaluator(value=0.0), raising=True)


def _patch_measured_provenance(monkeypatch, ck):
    """Patch ONLY the measurement, so every comparison still runs for real.

    Without this, a TDD run necessarily observes its own uncommitted
    implementation as a DIRTY TREE, and symmetric HEAD validation compares the
    machine's real HEAD against the fixture's "a"*40.
    """
    import scripts.GPU.alphazero.run_atlas as ra
    monkeypatch.setattr(ra, "preflight_source_provenance",
                        lambda path: _fixture_prov(ck))


def _stub_measurement(monkeypatch, *, complete=True):
    """Patch the expensive half only, deriving its result from the REAL
    assigned rows so the substitution the seam must reject is impossible."""
    import scripts.GPU.alphazero.atlas_run as ar
    from tests.test_atlas_run import _complete_continuation_doc

    def _fake(evaluator, metas, assigned_rows, **kw):
        doc = _complete_continuation_doc(assigned_rows)
        if complete:
            return doc
        return dict(doc, verdict="ABORTED", authoritative=False,
                    rows=doc["rows"][:-1], measured=len(doc["rows"]) - 1,
                    failed_rows=[{"game_id": doc["rows"][-1]["game_idx"],
                                  "failure": "seed mismatch"}])

    monkeypatch.setattr(ar, "run_corpus", _fake)


def test_exit_codes_follow_the_established_convention():
    assert (EXIT_OK, EXIT_USAGE, EXIT_PROVENANCE, EXIT_ABORTED) == (0, 2, 3, 5)


def test_every_frozen_stop_condition_is_listed_with_its_owner():
    """The runbook is the operator's only document. A stop condition that is
    not in it does not exist as far as the run is concerned."""
    names = {s["verdict"] for s in STOP_CONDITIONS}
    assert names >= {"PHASE_GEOMETRY_NO_GO", "ASSIGNMENT_SHORTFALL",
                     "PROJECTED_CAPACITY_NO_GO", "CAPACITY_FAILURE",
                     "INSUFFICIENT_CLASSES", "NOT_DEPLOYABLE",
                     "NO_SHAPE_PASSES", "PROVENANCE_FAILURE", "ABORTED"}
    for s in STOP_CONDITIONS:
        assert s["owner"] and s["action"]      # who raises it, what to do


def test_a_read_out_verdict_is_a_RESULT_not_a_nonzero_exit():
    """CAPACITY_FAILURE and NO_SHAPE_PASSES are findings the run was asked to
    produce. Only process failures are nonzero."""
    for s in STOP_CONDITIONS:
        if s["verdict"] in ("CAPACITY_FAILURE", "NO_SHAPE_PASSES",
                            "NOT_DEPLOYABLE", "INSUFFICIENT_CLASSES",
                            "PROJECTED_CAPACITY_NO_GO"):
            assert s["exit_code"] == EXIT_OK


def test_a_row_failure_is_ABORTED_and_exits_5():
    """The corpus is exactly N assigned positions, so ONE unmeasured position
    disqualifies the run -- it is not a finding, it is an incomplete run."""
    aborted = [s for s in STOP_CONDITIONS if s["verdict"] == "ABORTED"][0]
    assert aborted["exit_code"] == EXIT_ABORTED


def test_emit_runbook_is_zero_gpu_and_prints_the_operator_stop():
    r = _cli("emit-runbook")
    assert r.returncode == EXIT_OK
    assert "OPERATOR STOP" in r.stdout
    for rule in ("nohup", "disown", "REAL_EXIT", "status.json",
                 "shell_status", "setsid"):
        assert rule in r.stdout


def test_the_runbook_launches_through_a_DETACHED_SHELL_WRAPPER():
    """Python cannot record its own exit code if it is killed, so the wrapper
    shell records it."""
    out = _cli("emit-runbook").stdout
    assert "nohup sh " in out
    assert 'REAL_EXIT=$rc' in out and "shell_status" in out


def test_the_runbook_NEVER_tells_the_operator_to_wait_on_a_disowned_pid():
    """The Phase 0 defect: after `disown` a later shell has neither the job
    table nor a usable $!, so `wait $!` recovers nothing."""
    out = _cli("emit-runbook").stdout
    assert "wait $!" not in out
    assert "cat" in out and "status.json" in out


def test_preflight_MEASURES_provenance_instead_of_accepting_claims():
    """No --git-head / --checkpoint-sha1 / --worktree-clean arguments exist: a
    typed claim is not evidence."""
    out = _cli("preflight", "--help").stdout
    for claim in ("--git-head", "--checkpoint-sha1", "--worktree-clean"):
        assert claim not in out
    assert "--checkpoint" in out          # the FILE, which preflight hashes


def test_the_production_parser_exposes_NO_frozen_parameter():
    """The board, the replay budget, the ladder and N are frozen. A flag that
    can change them is a protocol change with a command-line interface."""
    for sub in ("run-pilot", "run-final"):
        out = _cli(sub, "--help").stdout
        for leaked in ("--active-size", "--prefix-sims", "--tiny-legs",
                       "--increments", "--threshold", "--leg-b",
                       "--n-target", "--n-simulations"):
            assert leaked not in out, f"{sub} exposes {leaked}"


def test_the_dirty_tree_case_is_CONSTRUCTED_not_observed():
    """validate_source_provenance is pure and takes porcelain as a string, so
    the negative is built rather than made by dirtying the ambient worktree."""
    from scripts.GPU.alphazero.generate_atlas_reservoir import (
        validate_source_provenance,
    )
    with pytest.raises(RuntimeError, match="dirty"):
        validate_source_provenance(porcelain=" M scripts/GPU/alphazero/mcts.py",
                                   git_head="a" * 40, checkpoint_path="ck",
                                   checkpoint_sha1="0" * 40)
    ok = validate_source_provenance(porcelain="", git_head="a" * 40,
                                    checkpoint_path="ck",
                                    checkpoint_sha1="0" * 40)
    assert ok["worktree_clean"] is True


def test_preflight_rejects_a_checkpoint_that_disagrees_with_the_manifest(
        tmp_path, monkeypatch):
    import scripts.GPU.alphazero.run_atlas as ra
    ck = _fake_ck(tmp_path)
    _patch_measured_provenance(monkeypatch, ck)
    block = _fake_block(tmp_path, checkpoint=ck)
    manifest = json.loads((block / "block_manifest.json").read_text())
    (block / "block_manifest.json").write_text(
        json.dumps({**manifest, "checkpoint_sha1": "f" * 40}))
    with pytest.raises(ValueError, match="checkpoint_sha1"):
        ra.measure_provenance(str(ck), pilot_dir=str(block))


def test_a_HEAD_mismatch_is_refused_symmetrically(tmp_path, monkeypatch):
    """The chain is produced at ONE qualified commit, so a HEAD that differs
    from a manifest is regeneration or requalification -- not a note."""
    import scripts.GPU.alphazero.run_atlas as ra
    ck = _fake_ck(tmp_path)
    monkeypatch.setattr(ra, "preflight_source_provenance",
                        lambda path: {**_fixture_prov(ck),
                                      "git_head": "b" * 40})
    with pytest.raises(ValueError, match="git_head"):
        ra.measure_provenance(str(ck),
                              pilot_dir=str(_fake_block(tmp_path,
                                                        checkpoint=ck)))


def test_preflight_rejects_an_unreadable_corpus_artifact(tmp_path, monkeypatch):
    _patch_measured_provenance(monkeypatch, _fake_ck(tmp_path))
    r = run_atlas_main(["preflight", "--checkpoint", str(_fake_ck(tmp_path)),
                        "--corpus-artifact", str(tmp_path / "nope.json")])
    assert r == EXIT_USAGE


def test_the_status_sidecar_records_the_verdict_and_the_exit_code(tmp_path):
    p = tmp_path / "status.json"
    write_status_sidecar(p, verdict="OK", exit_code=0, rows=240)
    d = json.loads(p.read_text())
    assert d["verdict"] == "OK" and d["exit_code"] == 0 and d["rows"] == 240


def test_the_emitted_launch_wrapper_ACTUALLY_records_the_exit_code(tmp_path):
    """Executed, not asserted about.

    A substring check cannot catch a redirection-order defect -- a wrapper can
    read plausibly and leave the sidecar empty. So the wrapper is built for a
    harmless command that exits 3, run, and the file is read back.
    """
    out = tmp_path / "out"
    out.mkdir()
    script = launch_wrapper([sys.executable, "-c", "raise SystemExit(3)"],
                            out_dir=out)
    rc = subprocess.run(["sh", "-c", script]).returncode
    assert (out / "shell_status").read_text().strip() == "REAL_EXIT=3"
    assert rc == 3                      # the wrapper exits with the real code
    assert (out / "run.log").exists()   # ...and the log was still captured


def test_no_zero_gpu_subcommand_constructs_an_evaluator():
    """preflight and emit-runbook are zero-GPU. Only the two run modes build an
    evaluator, and they import the factory lazily inside those branches."""
    import scripts.GPU.alphazero.run_atlas as mod
    src = open(mod.__file__).read()
    assert "_default_evaluator_factory" in src
    assert src.index("def _cmd_run_pilot") < src.index(
        "_default_evaluator_factory")


# -- the launchable seam, driven once per mode --------------------------------

def _final_argv(tmp_path, ck, out, pilot):
    """The frozen production argument set -- no budget flags exist to pass."""
    cont_block = _fake_block(tmp_path, name="cont", n_games=216,
                             start_index=24, checkpoint=ck)   # G_total - 24
    return ["run-final", "--pilot-artifact", str(pilot),
            "--corpus-artifact", str(_assigned_176(tmp_path, cont_block)),
            "--pilot-dir", str(_fake_block(tmp_path, checkpoint=ck)),
            "--continuation-dir", str(cont_block),
            "--base-seed", str(BASE), "--checkpoint", str(ck),
            "--out-dir", str(out)]


def test_run_final_SUCCESS_path_end_to_end(tmp_path, monkeypatch):
    """The successful final run, qualified without a single ladder.

    `run_corpus` -- the expensive half -- is patched to return a schema-valid
    COMPLETE 176-row document at the pilot-produced N=200. Everything else is
    real: the pilot artifact is loaded and authenticated, the assignment is
    recomputed, the carry and recomposition run, the artifact is emitted, and
    the success sidecar is written.
    """
    ck = _fake_ck(tmp_path)
    _patch_factory(monkeypatch)
    _patch_measured_provenance(monkeypatch, ck)
    _stub_measurement(monkeypatch, complete=True)
    out = tmp_path / "out"
    pilot = tmp_path / "pilot_artifact.json"
    pilot.write_text(_authoritative_pilot_artifact(ck, n=200))

    rc = run_atlas_main(_final_argv(tmp_path, ck, out, pilot))
    assert rc == EXIT_OK
    doc = json.loads((out / "atlas_artifact.json").read_text())
    assert doc["n_target"] == 200 and doc["measured"] == 200
    assert doc["pilot_rows_carried"] == 24
    assert doc["authoritative"] is True
    status = json.loads((out / "status.json").read_text())
    assert status["verdict"] == "OK" and status["exit_code"] == EXIT_OK


def test_run_final_ABORTED_path_end_to_end(tmp_path, monkeypatch):
    """Same path, one unmeasured position: exit 5 and non-authoritative."""
    ck = _fake_ck(tmp_path)
    _patch_factory(monkeypatch)
    _patch_measured_provenance(monkeypatch, ck)
    _stub_measurement(monkeypatch, complete=False)
    out = tmp_path / "out"
    pilot = tmp_path / "pilot_artifact.json"
    pilot.write_text(_authoritative_pilot_artifact(ck, n=200))

    rc = run_atlas_main(_final_argv(tmp_path, ck, out, pilot))
    assert rc == EXIT_ABORTED
    doc = json.loads((out / "atlas_artifact.json").read_text())
    assert doc["authoritative"] is False and doc["failed_rows"]
    status = json.loads((out / "status.json").read_text())
    assert status["verdict"] == "ABORTED" and status["exit_code"] == EXIT_ABORTED
