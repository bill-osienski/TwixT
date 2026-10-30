"""The combination policy pilot's CORPUS GENERATION (amendment
`docs/superpowers/2026-10-30-combination-policy-generation-amendment.md`), built only: the wrapper
(`docs/superpowers/evidence/combination-policy-generate-run/01_wrapper.sh`) and generation's admission without a
completed throughput check. CPU only; nothing here loads a model, touches a device, runs a seed procedure or opens a
real gate.

The wrapper is the throughput check's attempt-2 wrapper with only the stated differences, asserted line by line. THE
FAKE-LAUNCH CHECKS run the REAL generation wrapper's bytes in a throwaway git repository with the throughput tests'
stubs (tests/test_combination_policy_launch) moved to generation's paths: the exact command and the ceiling. THE
ADMISSION CHECKS use the real committed readers on a throwaway repository: the committed CPU timing alone admits
generation, and no throughput record is read."""
import json
import os
import pathlib
import re
import subprocess
import sys
import time

import pytest

from scripts.GPU.alphazero import combination_policy_pilot as P
from tests.test_combination_policy_launch import STUB_BG, STUB_IDLE, STUB_PILOT, _alive, _git

ROOT = pathlib.Path(__file__).resolve().parents[1]
EV = ROOT / "docs/superpowers/evidence"
WRAPPER2 = EV / "combination-policy-throughput2-run/01_wrapper.sh"
WRAPPER = EV / "combination-policy-generate-run/01_wrapper.sh"
GATE_NAME = "COMBINATION_POLICY_GENERATE_AUTHORIZED"
AMENDMENT = ("docs/superpowers/2026-10-30-combination-policy-generation-amendment.md",
             "11ced08185bd5e2727de07628a95e7cdc872ec31dae808e898640539742d7f55")


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """EVERY test (#74): the pilot's real seams, its record boundary and the real cache are unreachable."""
    def unreachable(*_a, **_k):
        raise AssertionError("a test reached a real seam or the real record boundary")
    monkeypatch.setattr(P, "_real_gen_seams", unreachable)
    monkeypatch.setattr(P, "_real_train_seams", unreachable)
    monkeypatch.setattr(P, "conclude", unreachable)
    monkeypatch.setattr(P, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv(P.SETTING, P.SETTING_VALUE)


def test_NOTHING_OF_GENERATION_HAS_RUN_and_both_throughput_runs_stay_VOID():
    """Built only: the gate is closed; the wrapper is the run directory's only file; no plan, run evidence, seed record
    or corpus exists. Both throughput runs stand as written -- VOID records under outer TIMEOUTs -- and their evidence
    is unchanged since their closings (attempt 1 e30e277c, attempt 2 ae361458). Read-only."""
    from scripts.GPU.alphazero import combination_policy_generate_authorization as G
    assert getattr(G, GATE_NAME) is False
    assert P.KINDS["generate"]["amendment"] == AMENDMENT
    assert P.M._sha((ROOT / AMENDMENT[0]).read_bytes()) == AMENDMENT[1]
    assert sorted(x.name for x in WRAPPER.parent.iterdir()) == ["01_wrapper.sh"]
    assert not (ROOT / P.KINDS["generate"]["plan"]).exists()
    assert not (ROOT / P.KINDS["generate"]["evidence"]).exists()
    assert not [x.name for x in EV.iterdir() if x.name.startswith("combination-policy-generate-seed")]
    assert not (pathlib.Path.home() / "twixt-cache/combination-policy-pilot/generate").exists()
    for kind in ("throughput", "throughput2"):
        rec = json.loads((ROOT / P.KINDS[kind]["evidence"] / "record.json").read_text(encoding="utf-8"))
        outer = json.loads((ROOT / pathlib.Path(P.KINDS[kind]["plan"]).parent / "outer.json").read_text(encoding="utf-8"))
        assert (rec["status"], rec["void"]["step"], outer["outcome"]) == ("VOID", "terminated", "TIMEOUT"), kind
    for since, kind in (("e30e277c", "throughput"), ("ae361458", "throughput2")):
        dirs = [f"docs/superpowers/evidence/combination-policy-{kind}{s}" for s in ("", "-run", "-seed-survey",
                                                                                    "-seed-closing")]
        assert subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", since, "HEAD", "--", *dirs]).returncode == 0


# ─────────────────────────────── the wrapper's bytes ───────────────────────────────

def _code(path):
    return path.read_text().split("set -u\n", 1)[1]


def test_THE_WRAPPER_IS_THROUGHPUT_ATTEMPT_2_S_WITH_ONLY_THE_STATED_DIFFERENCES():
    """The paths, the gate, `run generate` and TERM_S, line for line; everything else byte-identical. Its header is
    generation's paragraph followed by attempt 2's header unchanged."""
    two, gen = _code(WRAPPER2).splitlines(), _code(WRAPPER).splitlines()
    assert len(two) == len(gen)
    changed = [(a, b) for a, b in zip(two, gen) if a != b]
    assert [b for _a, b in changed] == [
        "OUT_REL=docs/superpowers/evidence/combination-policy-generate-run",
        "RUN_REL=docs/superpowers/evidence/combination-policy-generate",
        "GATE_REL=scripts/GPU/alphazero/combination_policy_generate_authorization.py",
        f"GATE={GATE_NAME}", "TERM_S=21590",
        '/usr/bin/perl -e "$SUPERVISOR" "$OUT_REL/pgid.txt" "$TERM_S" "$GRACE_S" env MLX_ENABLE_TF32=0 PYTHONPATH=. '
        '"$PY" -B -m "$MODULE" run generate --checkpoint "$CKPT" \\'], changed
    head2 = WRAPPER2.read_text().split("set -u\n", 1)[0]
    head = WRAPPER.read_text().split("set -u\n", 1)[0]
    assert head.startswith("#!/bin/bash\n# THE COMBINATION POLICY PILOT: CORPUS GENERATION (1,000 SLOTS, CARD STEP 3)")
    assert head.endswith(head2[len("#!/bin/bash\n"):]) and AMENDMENT[0] in head[:-len(head2)]
    assert os.access(WRAPPER, os.R_OK) and subprocess.run(["bash", "-n", str(WRAPPER)]).returncode == 0


def test_THE_CEILING_IS_THE_AMENDMENT_S_SIX_HOURS():
    """TERM at 21,590 s, KILL at 21,600 s (the card's maximum generation budget), the observation through 21,610 s."""
    w = _code(WRAPPER)
    term, grace = (int(re.search(rf"^{k}=(\d+)$", w, re.M).group(1)) for k in ("TERM_S", "GRACE_S"))
    assert (term, grace) == (21590, 10) and term + grace == 6 * 3600 == P.GENERATE_CEILING["max_seconds"]
    assert term + 2 * grace == 21610


def test_THE_WRAPPER_SERVES_THE_PILOT_S_GENERATE_PATHS():
    w = _code(WRAPPER)
    out, run = (re.search(rf"^{k}=(\S+)$", w, re.M).group(1) for k in ("OUT_REL", "RUN_REL"))
    kind = P.KINDS["generate"]
    assert run == kind["evidence"] and out == str(pathlib.Path(kind["plan"]).parent) == str(WRAPPER.parent.relative_to(ROOT))
    assert re.search(r"^GATE_REL=(\S+)$", w, re.M).group(1) == \
        str(pathlib.Path(kind["gate"][0].__file__).relative_to(ROOT))
    assert kind["gate"][1] == GATE_NAME and f"RECORD_KIND={P.RECORD_KIND}\n" in w
    others = {p for k in ("throughput", "throughput2")
              for p in (P.KINDS[k]["evidence"], str(pathlib.Path(P.KINDS[k]["plan"]).parent))}
    assert {run, out} & others == set()


# ─────────────────────────────── fake launches of the REAL generation wrapper ───────────────────────────────

#: the throughput tests' stub at generation's paths; its COMPLETED record carries the amendment's sha256, as the real
#: one does
STUB_PILOT_GEN = STUB_PILOT.replace("combination-policy-throughput", "combination-policy-generate").replace(
    '"head": head, "written_by": "script"}', '"head": head, "written_by": "script",\n'
    '                                                 "amendment_sha256": os.environ["FAKE_AMENDMENT_SHA"]}')
assert STUB_PILOT_GEN.count("combination-policy-generate") == 2 and "FAKE_AMENDMENT_SHA" in STUB_PILOT_GEN


def _fake_repo(tmp, term_s=None):
    """A throwaway repository: the real generation wrapper (TERM_S changed only when a test says so), its gate opened
    by a one-line commit, and three stubs."""
    repo = tmp / "repo"
    pkg = repo / "scripts/GPU/alphazero"
    pkg.mkdir(parents=True)
    for d in (repo / "scripts", repo / "scripts/GPU", pkg):
        (d / "__init__.py").write_text("")
    gate = pkg / "combination_policy_generate_authorization.py"
    gate.write_text(f'"""stub gate"""\n{GATE_NAME} = False\n')
    (pkg / "idle_gate.py").write_text(STUB_IDLE)
    (pkg / "background_compute.py").write_text(STUB_BG)
    (pkg / "combination_policy_pilot.py").write_text(STUB_PILOT_GEN)
    wdir = repo / "docs/superpowers/evidence/combination-policy-generate-run"
    wdir.mkdir(parents=True)
    raw = WRAPPER.read_bytes()
    if term_s is not None:
        assert raw.count(b"\nTERM_S=21590\n") == 1
        raw = raw.replace(b"\nTERM_S=21590\n", f"\nTERM_S={term_s}\n".encode())
    (wdir / "01_wrapper.sh").write_bytes(raw)
    (repo / ".venv/bin").mkdir(parents=True)
    (repo / ".venv/bin/python").symlink_to(sys.executable)
    ckpt = repo / "checkpoints/alphazero-v2-calib020-from0409"
    ckpt.mkdir(parents=True)
    (ckpt / "model_iter_0001.safetensors").write_bytes(b"not a model")
    (repo / ".gitignore").write_text(".venv/\ncheckpoints/\n__pycache__/\nscratch/\n")
    _git(tmp, "init", "-q", str(repo))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    gate.write_text(gate.read_text().replace(f"{GATE_NAME} = False", f"{GATE_NAME} = True"))
    _git(repo, "commit", "-qam", "open the gate")
    (repo / "scratch").mkdir()
    return repo, wdir, gate


def _launch(repo, wdir, idle="idle", bg="clear", run="ok", timeout=60):
    env = {k: v for k, v in os.environ.items() if k not in ("MLX_ENABLE_TF32", "PYTHONPATH")}
    env.update(FAKE_IDLE=idle, FAKE_BG=bg, FAKE_RUN=run, FAKE_OUT=str(repo / "scratch"), PYTHONDONTWRITEBYTECODE="1",
               FAKE_CARD_SHA=P.CARD_SHA256, FAKE_AMENDMENT_SHA=AMENDMENT[1])
    return subprocess.run(["bash", str(wdir / "01_wrapper.sh")], cwd=str(repo), env=env, capture_output=True,
                          text=True, timeout=timeout)


@pytest.mark.parametrize("idle, bg", [("busy", "clear"), ("idle", "boinc"), ("idle", "failed")],
                         ids=["busy", "boinc", "check-fails"])
def test_FAKE_LAUNCH_A_BUSY_MACHINE_BOINC_OR_A_FAILED_CHECK_PREVENTS_THE_LAUNCH(tmp_path, idle, bg):
    repo, wdir, gate = _fake_repo(tmp_path)
    p = _launch(repo, wdir, idle, bg)
    assert p.returncode == 1 and ": nothing launched" in p.stderr, p.stderr
    assert "FAKE_LAUNCHED" not in p.stdout + p.stderr
    assert sorted(x.name for x in wdir.iterdir()) == ["01_wrapper.sh"]
    assert not (repo / "docs/superpowers/evidence/combination-policy-generate").exists()
    assert f"{GATE_NAME} = False\n" in gate.read_text()


def test_FAKE_LAUNCH_AN_IDLE_MACHINE_REACHES_RUN_GENERATE_with_its_exact_arguments_and_ceiling(tmp_path):
    repo, wdir, gate = _fake_repo(tmp_path)
    p = _launch(repo, wdir, run="record")
    assert p.returncode == 0, p.stderr
    launched = json.loads((wdir / "run.stdout.txt").read_text().split("FAKE_LAUNCHED", 1)[1])
    ckpt = str(repo.resolve() / "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors")
    assert launched["argv"] == ["run", "generate", "--checkpoint", ckpt] and launched["tf32"] == "0"
    assert (wdir / "pgid.txt").read_text() == f"{launched['pgid']}\n"
    outer = json.loads((wdir / "outer.json").read_text())
    assert outer["outcome"] == "COMPLETED" and outer["survivors"] == "" and outer["gate_restored"] is True
    assert outer["term_s"] == 21590 and outer["grace_s"] == 10 and outer["record_by"] == "script"
    rec = json.loads((repo / "docs/superpowers/evidence/combination-policy-generate/record.json").read_text())
    assert rec["amendment_sha256"] == AMENDMENT[1] and not P.outer_faults(outer, rec)
    assert f"{GATE_NAME} = False\n" in gate.read_text()


def test_FAKE_LAUNCH_THE_CEILING_TERMINATES_THE_WHOLE_GROUP_at_generation_s_paths(tmp_path):
    """TERM_S changed to 2 in this copy only: the run and its 'worker' outlive it; the group TERM ends both, and the
    wrapper's VOID record lands in generation's evidence directory."""
    repo, wdir, gate = _fake_repo(tmp_path, term_s=2)
    t = time.monotonic()
    p = _launch(repo, wdir, run="outlive")
    child = int((repo / "scratch/child.pid").read_text())
    assert time.monotonic() - t < 30 and p.returncode == 124, p.stderr
    for _ in range(50):
        if not _alive(child):
            break
        time.sleep(0.1)
    assert not _alive(child)
    outer = json.loads((wdir / "outer.json").read_text())
    assert outer["outcome"] == "TIMEOUT" and outer["inner_exit"] == 124 and outer["survivors"] == ""
    rec = json.loads((repo / "docs/superpowers/evidence/combination-policy-generate/record.json").read_text())
    assert outer["record_by"] == "wrapper" and rec["status"] == "VOID" and rec["void"]["step"] == "timeout"
    assert f"{GATE_NAME} = False\n" in gate.read_text()


# ─────────────────────────────── admission through the real committed readers ───────────────────────────────

def _admission_repo(tmp, monkeypatch, timing, commit=True):
    """A throwaway repository holding both throughput runs' records as they really stand (VOID, outer TIMEOUT) and,
    unless `timing` is None, a timing record; the REAL committed readers pointed at it."""
    repo = tmp / "adm"
    repo.mkdir()

    def git(*a):
        return subprocess.check_output(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo,
                                       text=True).strip()
    git("init", "-q")
    for kind in ("throughput", "throughput2"):
        for rel, name in ((P.KINDS[kind]["evidence"], "record.json"),
                          (str(pathlib.Path(P.KINDS[kind]["plan"]).parent), "outer.json")):
            src = ROOT / rel / name
            (repo / rel).mkdir(parents=True, exist_ok=True)
            (repo / rel / name).write_bytes(src.read_bytes())
    git("add", "-A")
    git("commit", "-qm", "the throughput runs")
    if timing is not None:
        (repo / P.TIMING_EVIDENCE).mkdir(parents=True)
        (repo / P.TIMING_EVIDENCE / "record.json").write_text(json.dumps(timing))
        if commit:
            git("add", "-A")
            git("commit", "-qm", "the timing")
    monkeypatch.setattr(P, "ROOT", repo)
    monkeypatch.setattr(P.M, "_git", git)
    return repo


def _counting(reader, reads):
    def read(rel):
        reads.append(rel)
        return reader(rel)
    return read


def test_THE_COMMITTED_TIMING_ALONE_ADMITS_GENERATION_and_no_throughput_record_is_read(tmp_path, monkeypatch):
    _admission_repo(tmp_path, monkeypatch, {"status": "COMPLETED", "card_sha256": P.CARD_SHA256})
    reads, outers = [], []
    assert P.admitted("generate", _counting(P._read_record, reads), _counting(P._read_outer, outers)) == \
        {P.TIMING_EVIDENCE: "COMPLETED"}
    assert reads == [P.TIMING_EVIDENCE] and outers == []


@pytest.mark.parametrize("timing, commit, why", [
    (None, True, "combination-policy-timing/record.json does not exist"),
    ({"status": "COMPLETED", "card_sha256": P.CARD_SHA256}, False, "uncommitted or differs from HEAD"),
    ({"status": "VOID", "card_sha256": P.CARD_SHA256}, True, "not COMPLETED"),
    ({"status": "COMPLETED", "card_sha256": "0" * 64}, True, "not COMPLETED")],
    ids=["absent", "uncommitted", "void", "other-card"])
def test_WITHOUT_A_COMMITTED_COMPLETED_TIMING_GENERATION_IS_REFUSED(tmp_path, monkeypatch, timing, commit, why):
    """The throughput records are present (VOID, outer TIMEOUT) but are not what decides: the timing is."""
    _admission_repo(tmp_path, monkeypatch, timing, commit)
    with pytest.raises(P.Refused, match=why):
        P.admitted("generate", P._read_record, P._read_outer)
