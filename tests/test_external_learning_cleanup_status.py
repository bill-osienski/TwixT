"""Unverified process-group cleanup has its OWN exit status in all three external-learning stages.

The defect (stage-3 review, 2026-10-10): a PermissionError from `os.killpg(pgid, 0)` escaped `EL.end_group`,
so the parent crashed with Python's status 1 -- the code of a verdict (stage 1 STOP_F_ZERO, stage 3
STOP_HELDOUT). Now `run_group` keeps the error and reports the group UNVERIFIED (reaped None); every
stage answers EXIT_CLEANUP_UNVERIFIED, writes its cleanup-failed record with the error, and finalizes
nothing. A PermissionError is never read as "group gone". Synthetic only; every gate stays closed on disk.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import external_learning as EL
from scripts.GPU.alphazero import external_learning_authorization as G1
from scripts.GPU.alphazero import external_learning_stage2 as S2
from scripts.GPU.alphazero import external_learning_stage2_authorization as G2
from scripts.GPU.alphazero import external_learning_stage3 as S3
from scripts.GPU.alphazero import external_learning_stage3_authorization as G3

EPERM = "PermissionError: [Errno 1] Operation not permitted"


def test_the_status_is_DISTINCT_from_every_verdict_refusal_and_timeout():
    u = EL.EXIT_CLEANUP_UNVERIFIED
    assert u == S3.EXIT_CLEANUP_UNVERIFIED == 5
    for table in (EL.VERDICT_EXIT, S3.VERDICT_EXIT):                             # stage 2 answers through EL's
        assert u not in table.values()
    assert u not in (EL.EXIT_REFUSED, EL.EXIT_TIMEOUT, S3.EXIT_REFUSED, S3.EXIT_TIMEOUT)


@pytest.mark.parametrize("exit_code_for", [EL.exit_code_for, S3.exit_code_for], ids=["stages-1-2", "stage-3"])
@pytest.mark.parametrize("reaped", [False, None])
@pytest.mark.parametrize("status", ["EXITED", "TIMEOUT", "UNKNOWN"])
def test_an_UNVERIFIED_cleanup_answers_first_even_over_a_timeout(tmp_path, exit_code_for, reaped, status):
    rec = tmp_path / "record.json"
    rec.write_text(json.dumps({"verdict": "STOP"}))
    assert exit_code_for(rec, status, 2, reaped) == EL.EXIT_CLEANUP_UNVERIFIED
    assert exit_code_for(rec, "TIMEOUT", None, True) == EL.EXIT_TIMEOUT          # control: verified cleanup


def test_run_group_KEEPS_a_PermissionError_from_the_group_check_as_UNVERIFIED(monkeypatch):
    """The real path: the child exits and is reaped, then killpg(pgid, 0) raises EPERM. Not 'gone'."""
    real = os.killpg

    def killpg(pgid, sig):
        if sig == 0:
            raise PermissionError(1, "Operation not permitted")
        return real(pgid, sig)
    monkeypatch.setattr(EL.os, "killpg", killpg)
    monkeypatch.setattr(EL, "GROUP_GRACE_S", 0.3)
    status, code, reaped, error = EL.run_group([sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL,
                                               stderr=subprocess.DEVNULL, timeout=10)
    assert (status, code, reaped, error) == ("EXITED", 0, None, EPERM)


def test_control_a_VERIFIED_cleanup_reports_no_error():
    status, code, reaped, error = EL.run_group([sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL,
                                               stderr=subprocess.DEVNULL, timeout=10)
    assert (status, code, reaped, error) == ("EXITED", 0, True, None)


# ─────────────────────────────── the three parents ───────────────────────────────

def _stage(which, monkeypatch, tmp_path):
    """(run, module, verdict) with the gate opened IN MEMORY, ROOT a temp tree, git stubbed."""
    monkeypatch.setattr(EL, "_git", lambda *a: "" if a[0] == "status" else "f" * 40)
    if which == 1:
        monkeypatch.setattr(G1, "EXTERNAL_LEARNING_STAGE1_AUTHORIZED", True)
        monkeypatch.setattr(EL, "ROOT", tmp_path)
        return (lambda: EL.run(str(tmp_path / "clone"))), EL, "PASS"
    if which == 2:
        monkeypatch.setattr(G2, "EXTERNAL_LEARNING_STAGE2_AUTHORIZED", True)
        monkeypatch.setattr(S2, "ROOT", tmp_path)
        return S2.run, S2, "PASS"
    monkeypatch.setattr(G3, "EXTERNAL_LEARNING_STAGE3_AUTHORIZED", True)
    monkeypatch.setattr(S3, "ROOT", tmp_path)
    return S3.run, S3, "SCREEN_COMPLETED"


@pytest.mark.parametrize("which", [1, 2, 3])
def test_each_parent_RETURNS_the_status_KEEPS_the_error_and_FINALIZES_nothing(monkeypatch, tmp_path, which):
    run, M, verdict = _stage(which, monkeypatch, tmp_path)

    def fake_group(argv, *, stdout, stderr, timeout):
        run_dir = pathlib.Path(argv[argv.index("--run-dir") + 1])
        (run_dir / "record.json").write_text(json.dumps({"verdict": verdict}))
        return "EXITED", 0, None, EPERM                          # the child passed; its group is unverified
    monkeypatch.setattr(EL, "run_group", fake_group)
    assert run() == EL.EXIT_CLEANUP_UNVERIFIED                   # returned, not raised
    run_dir, evidence = tmp_path / M.RUN_DIR, tmp_path / M.EVIDENCE_DIR
    assert not evidence.exists() and not (run_dir / "finalized.json").exists()
    assert (run_dir / "record.json").exists()                    # the diagnostics stay where they were
    rec = json.loads((tmp_path / M.CLEANUP_FAILED).read_text())
    assert (rec["group_reaped"], rec["cleanup_error"], rec["exit_code"], rec["finalized"]) == \
        (None, EPERM, EL.EXIT_CLEANUP_UNVERIFIED, False)
    assert rec["child_exit"] == 0 and "files" not in rec and "verdict" not in rec


@pytest.mark.parametrize("which", [1, 2, 3])
def test_control_each_parent_with_a_VERIFIED_group_finalizes_with_no_error(monkeypatch, tmp_path, which):
    run, M, verdict = _stage(which, monkeypatch, tmp_path)

    def fake_group(argv, *, stdout, stderr, timeout):
        run_dir = pathlib.Path(argv[argv.index("--run-dir") + 1])
        (run_dir / "record.json").write_text(json.dumps({"verdict": verdict}))
        return "EXITED", 0, True, None
    monkeypatch.setattr(EL, "run_group", fake_group)
    assert run() == 0
    fin = json.loads((tmp_path / M.RUN_DIR / "finalized.json").read_text())
    assert (fin["group_reaped"], fin["cleanup_error"]) == (True, None)
    assert not (tmp_path / M.CLEANUP_FAILED).exists() and (tmp_path / M.EVIDENCE_DIR / "MANIFEST.sha256").exists()
