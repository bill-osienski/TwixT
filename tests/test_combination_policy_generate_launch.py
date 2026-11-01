"""The combination policy pilot's CORPUS GENERATION (amendment
`docs/superpowers/2026-10-30-combination-policy-generation-amendment.md`), built only: the wrapper
(`docs/superpowers/evidence/combination-policy-generate-run/01_wrapper.sh`) and generation's admission without a
completed throughput check, its seed-survey copy in the 2027 band (`seed_band_2027`, the user's review 2026-10-30),
and training's admission through generation's outer.json. CPU only; nothing here loads a model, touches a device, runs a
seed procedure or opens a real gate.

The wrapper is the throughput check's attempt-2 wrapper with only the stated differences, asserted line by line. THE
FAKE-LAUNCH CHECKS run the REAL generation wrapper's bytes in a throwaway git repository with the throughput tests'
stubs (tests/test_combination_policy_launch) moved to generation's paths: the exact command and the ceiling. THE
ADMISSION CHECKS use the real committed readers on a throwaway repository: the committed CPU timing alone admits
generation, and no throughput record is read; training needs generation's committed record AND outer.json to show
success, so an outer TIMEOUT over a COMPLETED record refuses it."""
import difflib
import importlib.util
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
SURVEY2 = "combination-policy-throughput2-seed-survey/01_seed_survey.py"
SURVEY = "combination-policy-generate-seed-survey/01_seed_survey.py"
CLOSING2 = "combination-policy-throughput2-seed-closing/01_derive_usage.py"
CLOSING = "combination-policy-generate-seed-closing/01_derive_usage.py"
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


def test_GENERATION_RAN_ONCE_COMPLETED_AND_IS_CLOSED_and_both_throughput_runs_stay_VOID():
    """RE-ANCHORED 2026-10-30 (survey bb1549ea, pre-proof 4701b1d3, registration 326004d0, post-proof 5a17ea92): the
    survey is CLEAN and both proofs HOLD, each 55/55 on a clean tree, the proofs bound to the survey record's sha256;
    the plan is the copy's build_plan; the block (read from the plan and the records -- no seed literal here, since
    the proofs scan tests/) is the first in the 2027 band, ACCOUNTED exactly once and last.
    RE-ANCHORED 2026-10-30 (the run, acf07d24 -> 56e6f386, and its closing, K1 4cf54bac + K2 569ba2a4): the ONE run
    COMPLETED inside the ceiling -- outer exit 0, record COMPLETED, gate restored, no survivors, outer_faults empty --
    launched from the one-line gate opening on top of the post-proof's state, so the registry it launched with is the
    one the post-proof proved; every slot was claimed, started and ended once, slot g on seed BASE + g (games.jsonl is
    read for slot, seed and file fields only -- never a winner or reason); K1 RECONCILES with the one final `claims`
    event equal to the slot count, so the WHOLE block is EXPOSED and RETIRED, each exactly once and appended last,
    ACCOUNTED unchanged, and every seed is refused. The gate is closed. The run evidence is unchanged since it was
    committed. Both throughput runs stand as written -- VOID records under outer TIMEOUTs -- unchanged since their
    closings (attempt 1 e30e277c, attempt 2 ae361458). Read-only.
    RE-ANCHORED 2026-10-31 (the screen's registration 3aaff337): the block is no longer last in ACCOUNTED; every later
    entry lies above it.
    RE-ANCHORED 2026-10-31 (the screen's closing, K2 572d1dda): nor last in EXPOSED or RETIRED; every later entry in
    each lies above it."""
    import hashlib
    from scripts.GPU.alphazero import combination_policy_generate_authorization as G
    from scripts.GPU.alphazero import e4_screen_reference as REG
    from scripts.GPU.alphazero import seed_band_2027 as SB
    assert getattr(G, GATE_NAME) is False
    assert P.KINDS["generate"]["amendment"] == AMENDMENT
    assert P.M._sha((ROOT / AMENDMENT[0]).read_bytes()) == AMENDMENT[1]
    run_dir, evidence = WRAPPER.parent, ROOT / P.KINDS["generate"]["evidence"]
    assert sorted(x.name for x in run_dir.iterdir()) == ["01_wrapper.sh", "bg_check.txt", "idle_check.txt",
                                                         "outer.json", "pgid.txt", "plan.json", "run.stderr.txt",
                                                         "run.stdout.txt"]
    assert sorted(x.name for x in evidence.iterdir()) == ["games.jsonl", "progress.jsonl", "record.json"]
    here = (EV / SURVEY).parent
    assert sorted(x.name for x in here.iterdir() if x.name != "__pycache__") == \
        ["01_seed_survey.py", "02_seed_survey_run.txt", "03_collision_proof_pre_run.txt",
         "04_collision_proof_post_run.txt"]
    raw = (here / "02_seed_survey_run.txt").read_bytes()
    survey, sha = raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()
    assert survey.splitlines()[2].endswith("; working tree clean") and "controls passed: 55/55" in survey
    assert survey.endswith("SURVEY VERDICT   : THE COMBINATION-POLICY GENERATION BLOCK IS CLEAN\n")
    for rel, mode in (("03_collision_proof_pre_run.txt", "pre"), ("04_collision_proof_post_run.txt", "post")):
        text = (here / rel).read_text(encoding="utf-8")
        assert text.splitlines()[2].endswith("; working tree clean")
        assert f"02_seed_survey_run.txt sha256 {sha} (read at " in text
        assert f"EXPECTED STATE ({mode}): HOLDS" in text and "controls passed: 55/55" in text
        assert text.endswith(f"PROOF VERDICT ({mode}): THE COMBINATION-POLICY GENERATION BLOCK IS CLEAN AND THE STATE "
                             "IS AS EXPECTED\n")
    post = (here / "04_collision_proof_post_run.txt").read_text(encoding="utf-8")
    plan = json.loads((ROOT / P.KINDS["generate"]["plan"]).read_text(encoding="utf-8"))
    n = P.KINDS["generate"]["slots"]
    assert type(plan["base"]) is int and plan["n_slots"] == n == 1000
    assert plan == _load(SURVEY, "_gen_survey_registered").build_plan(plan["base"], sha)
    block = (plan["base"], plan["base"] + n)
    assert SB.BAND[0] <= block[0] < block[1] <= SB.BAND[1] and block[0] % SB.ALIGN == 0
    assert f"block = [{block[0]}, {block[1]}) -- {n} seeds" in survey
    assert f"THE BLOCK, read from the record: [{block[0]}, {block[1]}) -- BASE {block[0]}" in post
    # the run: launched from the one-line gate opening, COMPLETED, gate restored, no survivors
    outer = json.loads((run_dir / "outer.json").read_text(encoding="utf-8"))
    record = json.loads((evidence / "record.json").read_text(encoding="utf-8"))
    assert P.outer_faults(outer, record) == []
    assert (outer["outcome"], outer["inner_exit"], outer["outer_exit"]) == ("COMPLETED", 0, 0)
    assert outer["wall_seconds"] < outer["term_s"] == 21590 and outer["grace_s"] == 10
    assert (record["status"], record["run"], record["verdict"]) == ("COMPLETED", "generate", None)
    assert record["amendment_sha256"] == AMENDMENT[1] and record["seeds"] == {"base": block[0], "n_slots": n}
    assert record["setting"] == record["setting_at_end"] == {P.SETTING: P.SETTING_VALUE}
    assert record["mode_at_load"]["modules_in_training_mode"] == record["mode_at_end"]["modules_in_training_mode"] == 0
    tel = record["telemetry"]
    assert (tel["claims"], tel["started"], tel["ended"], tel["workers"], tel["max_batch_rows"]) == (n, n, n, 18, 14)
    gate_rel = "scripts/GPU/alphazero/combination_policy_generate_authorization.py"
    assert subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"{outer['head']}^"], capture_output=True, text=True,
                          check=True).stdout.strip() == outer["pre_open"]
    opened = subprocess.run(["git", "-C", str(ROOT), "diff", "--unified=0", outer["pre_open"], outer["head"]],
                            capture_output=True, text=True, check=True).stdout
    assert [ln for ln in opened.splitlines() if ln[:1] in "+-" and ln[:3] not in ("+++", "---")] == \
        [f"-{GATE_NAME} = False", f"+{GATE_NAME} = True"] and f"b/{gate_rel}" in opened
    reg_rel = "scripts/GPU/alphazero/e4_screen_reference.py"
    launched = subprocess.run(["git", "-C", str(ROOT), "show", f"{outer['head']}:{reg_rel}"],
                              capture_output=True, check=True).stdout
    assert f"\nregistry {reg_rel} sha256 {hashlib.sha256(launched).hexdigest()}\n" in post     # proved = launched
    games = [json.loads(line) for line in (evidence / "games.jsonl").read_text(encoding="utf-8").splitlines()]
    seen = [(g["slot"], g["seed"], g["rows_file"], g["rows_sha256"]) for g in games]           # no winner or reason
    assert sorted(s for s, *_ in seen) == list(range(n)) and all(seed == block[0] + s for s, seed, *_ in seen)
    assert len({f for _s, _seed, f, _h in seen}) == n and all(re.fullmatch(r"[0-9a-f]{64}", h) for *_, h in seen)
    # the closing: every slot started and the one claims event equals n, so the WHOLE block is EXPOSED and RETIRED
    closing = (EV / CLOSING).parent
    assert sorted(x.name for x in closing.iterdir() if x.name != "__pycache__") == \
        ["01_derive_usage.py", "02_derive_usage_run.txt"]
    k1 = (closing / "02_derive_usage_run.txt").read_text(encoding="utf-8")
    assert k1.splitlines()[2].endswith("; working tree clean")
    assert f"claims events: [{n}]\n" in k1 and "controls passed: 21/21" in k1
    assert k1.endswith(f"VERDICT: RECONCILES -- EXPOSE {list(block)}; RETIRE {list(block)}\n")
    # RE-ANCHORED 2026-10-31 (the screen's registration 3aaff337, then its closing 572d1dda): no longer last in
    # ACCOUNTED, EXPOSED or RETIRED; every later entry in each lies above it
    for intervals in (REG.ACCOUNTED_SEED_INTERVALS, REG.EXPOSED_SEED_INTERVALS, REG.RETIRED_SEED_INTERVALS):
        assert intervals.count(block) == 1
        assert all(lo >= block[1] for lo, _hi in intervals[intervals.index(block) + 1:])
    assert not any(lo < block[1] and block[0] < hi and (lo, hi) != block
                   for lo, hi in REG.EXPOSED_SEED_INTERVALS + REG.RETIRED_SEED_INTERVALS + REG.TEST_ONLY_SEED_INTERVALS)
    assert all(REG.seed_is_accounted(s) and REG.seed_is_exposed(s) and REG.seed_is_retired(s) and
               REG.seed_is_unavailable(s) and not REG.seed_is_test_only(s) for s in range(*block))
    assert subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "56e6f386", "HEAD", "--", str(run_dir),
                           str(evidence)]).returncode == 0
    for kind in ("throughput", "throughput2"):
        rec = json.loads((ROOT / P.KINDS[kind]["evidence"] / "record.json").read_text(encoding="utf-8"))
        tp_dir = ROOT / pathlib.Path(P.KINDS[kind]["plan"]).parent
        tp_outer = json.loads((tp_dir / "outer.json").read_text(encoding="utf-8"))
        assert (rec["status"], rec["void"]["step"], tp_outer["outcome"]) == ("VOID", "terminated", "TIMEOUT"), kind
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


# ─────────────────────────────── training's admission through the fake generation wrapper ─────────────────────────────

def _commit_with_timing(repo, monkeypatch):
    """Add a COMPLETED timing record, commit everything the fake run wrote, and point the REAL committed readers
    here."""
    def git(*a):
        return subprocess.check_output(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo,
                                       text=True).strip()
    (repo / P.TIMING_EVIDENCE).mkdir(parents=True)
    (repo / P.TIMING_EVIDENCE / "record.json").write_text(json.dumps({"status": "COMPLETED",
                                                                     "card_sha256": P.CARD_SHA256}))
    monkeypatch.setattr(P, "ROOT", repo)
    monkeypatch.setattr(P.M, "_git", git)
    git("add", "docs/superpowers/evidence")
    git("commit", "-qm", "the run's evidence and the timing")


def test_TRAINING_IS_ADMITTED_BY_A_COMPLETED_COMMITTED_GENERATION_RUN(tmp_path, monkeypatch):
    repo, wdir, _gate = _fake_repo(tmp_path)
    p = _launch(repo, wdir, run="record")
    assert p.returncode == 0, p.stderr
    _commit_with_timing(repo, monkeypatch)
    assert P.admitted("train", P._read_record, P._read_outer) == {
        "docs/superpowers/evidence/combination-policy-generate": "COMPLETED", P.TIMING_EVIDENCE: "COMPLETED",
        "docs/superpowers/evidence/combination-policy-generate-run/outer.json": "COMPLETED"}


def test_AN_OUTER_TIMEOUT_OVER_GENERATION_S_COMPLETED_RECORD_REFUSES_TRAINING(tmp_path, monkeypatch):
    """The user's ruling (2026-10-29), now at generation -> training (review, 2026-10-30): the script wrote COMPLETED,
    then the run outlived the ceiling (TERM_S 2 in this copy only). Both records stand as written; training is
    refused."""
    repo, wdir, _gate = _fake_repo(tmp_path, term_s=2)
    p = _launch(repo, wdir, run="record_outlive")
    outer = json.loads((wdir / "outer.json").read_text())
    rec = json.loads((repo / "docs/superpowers/evidence/combination-policy-generate/record.json").read_text())
    assert p.returncode == 124 and outer["outcome"] == "TIMEOUT" and outer["record_by"] == "script", outer
    assert rec["status"] == "COMPLETED"
    _commit_with_timing(repo, monkeypatch)
    with pytest.raises(P.Refused, match="inner_exit is 124, not 0"):
        P.admitted("train", P._read_record, P._read_outer)


def test_A_GENERATION_RUN_WITHOUT_ITS_OUTER_JSON_COMMITTED_DOES_NOT_ADMIT_TRAINING(tmp_path, monkeypatch):
    repo, wdir, _gate = _fake_repo(tmp_path)
    assert _launch(repo, wdir, run="record").returncode == 0
    (wdir / "outer.json").rename(repo / "scratch" / "outer.json")          # the wrapper's record never committed
    _commit_with_timing(repo, monkeypatch)
    with pytest.raises(P.Refused, match="outer.json does not exist"):
        P.admitted("train", P._read_record, P._read_outer)


# ─────────────────────────────── the generation seed-survey copy (built, never run here) ─────────────────────────────

def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, EV / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                     # module level only: no mode runs, nothing is written
    return mod


def _split(rel):
    text = (EV / rel).read_text()
    end = text.index('"""', 3)
    return text[3:end], text[end + 3:]


def test_THE_SURVEY_COPY_IS_THROUGHPUT_ATTEMPT_2_S_WITH_ONLY_THE_STATED_DIFFERENCES():
    """Below the docstring: the block size, paths, plan kind, module name, titles and verdict lines, the rule module
    and its install, the survey-mode rule moved after the scan (it reads the literals), and the two-band controls --
    nothing else. The docstring is generation's paragraph followed by attempt 2's description unchanged."""
    doc2, code2 = _split(SURVEY2)
    doc, code = _split(SURVEY)
    assert doc.endswith(doc2) and AMENDMENT[0] in doc[:-len(doc2)] and "seed_band_2027" in doc[:-len(doc2)]
    a, b = code2.splitlines(), code.splitlines()
    removed, added = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            removed += [x.strip() for x in a[i1:i2]]
            added += list(range(j1, j2))
    assert removed == [
        "N_SLOTS = 144", 'HERE = "docs/superpowers/evidence/combination-policy-throughput2-seed-survey"',
        'PLAN = "docs/superpowers/evidence/combination-policy-throughput2-run/plan.json"',
        'VERDICT_OK = "THE COMBINATION-POLICY THROUGHPUT ATTEMPT-2 BLOCK IS CLEAN"',
        '"""The two pinned modules, private copies; this line\'s values rebound on the SURVEY copy only."""',
        'spec = importlib.util.spec_from_file_location(f"_combination_policy_throughput2_{key}", p)',
        "from scripts.GPU.alphazero import seed_selection as SEL",
        'return {"kind": "combination-policy-throughput2-plan/1", "base": base, "n_slots": N_SLOTS,',
        'faults.append("the throughput attempt-2 plan is absent or not committed")',
        'faults.append("the throughput attempt-2 plan is not build_plan of the surveyed block")',
        "say(f\"COMBINATION POLICY PILOT -- THROUGHPUT ATTEMPT-2 GAME-SEED {'SURVEY' if mode == 'survey' else "
        "'COLLISION PROOF, mode ' + mode.upper()}\")",
        'sel_rel = "scripts/GPU/alphazero/seed_selection.py"',
        'say(f"rule {sel_rel} sha256 {_sha((ROOT / sel_rel).read_bytes())} (ALIGN {SEL.ALIGN})")',
        "reserved = [iv for v in registries.values() for iv in v] + [iv for v in named.values() for iv in v]",
        "highest, base, block = SEL.candidate(reserved, N_SLOTS)",
        'say(f"\\nCANDIDATE RULE: highest registered/named-spent SEED in the band = {highest}; BASE = {base}; "',
        'f"block = [{block[0]}, {block[1]}) -- {N_SLOTS} seeds")',
        "say(f\"PROOF VERDICT ({mode}): {'THE COMBINATION-POLICY THROUGHPUT ATTEMPT-2 BLOCK IS CLEAN AND THE STATE IS "
        "AS EXPECTED' if ok else 'NOT PROVEN -- STOP'}\""], removed
    fn = (next(i for i, x in enumerate(b) if x.startswith("def _two_band_controls(")),
          next(i for i, x in enumerate(b) if x.startswith("def _git(")))
    rule_start = ("if block is None:                         # survey: the rule's literals, the plan never exempt "
                  "(no block yet)")
    rule = (next(i for i, x in enumerate(b) if x.strip() == rule_start),
            next(i for i, x in enumerate(b) if x.strip() == "literals = S.repo_literals(files, block)"))
    stray = [b[j].strip() for j in added if not (fn[0] <= j < fn[1] or rule[0] <= j < rule[1])]
    assert stray == [
        "N_SLOTS = 1000", 'HERE = "docs/superpowers/evidence/combination-policy-generate-seed-survey"',
        'PLAN = "docs/superpowers/evidence/combination-policy-generate-run/plan.json"',
        'VERDICT_OK = "THE COMBINATION-POLICY GENERATION BLOCK IS CLEAN"',
        '"""The two pinned modules, private copies; this line\'s values rebound on the SURVEY copy only, its scanner',
        'extended to both bands; the rule is the two-band method\'s."""',
        'spec = importlib.util.spec_from_file_location(f"_combination_policy_generate_{key}", p)',
        "from scripts.GPU.alphazero import seed_band_2027 as SEL", "SEL.install(S)",
        'return {"kind": "combination-policy-generate-plan/1", "base": base, "n_slots": N_SLOTS,',
        'faults.append("the generation plan is absent or not committed")',
        'faults.append("the generation plan is not build_plan of the surveyed block")',
        "say(f\"COMBINATION POLICY PILOT -- GENERATION GAME-SEED {'SURVEY' if mode == 'survey' else "
        "'COLLISION PROOF, mode ' + mode.upper()}\")",
        'sel_rel = "scripts/GPU/alphazero/seed_band_2027.py"',
        'say(f"rule and scanner {sel_rel} sha256 {_sha((ROOT / sel_rel).read_bytes())} '
        '(ALIGN {SEL.ALIGN}; the rule\'s "',
        'f"band {list(SEL.BAND)}; literals scanned in {list(SEL.BAND_2026)} and {list(SEL.BAND)})")',
        "block = None                              # the rule reads the scan\'s literals: chosen below, after the scan",
        "results += _two_band_controls(S, SEL, block, prior, masks)",
        "say(f\"PROOF VERDICT ({mode}): {'THE COMBINATION-POLICY GENERATION BLOCK IS CLEAN AND THE STATE IS AS "
        "EXPECTED' if ok else 'NOT PROVEN -- STOP'}\""], stray


def test_THE_SURVEY_COPY_SERVES_THE_GENERATION_RUN_IN_THE_2027_BAND():
    survey = _load(SURVEY, "_gen_survey")
    kind = P.KINDS["generate"]
    assert survey.N_SLOTS == kind["slots"] == 1000 and survey.PLAN == kind["plan"]
    S, _Pf, SEL = survey.load_method()
    assert S.N == 1000 and S.scan is SEL.scan and SEL.__name__.endswith("seed_band_2027")
    assert survey.PINNED == _load(SURVEY2, "_tp2_survey_pins").PINNED
    base = SEL.candidate([], [], 1000)[1]
    assert survey.build_plan(base, "0" * 64) == {"kind": "combination-policy-generate-plan/1", "base": base,
                                                 "n_slots": 1000, "survey_record_sha256": "0" * 64}
    assert set(survey.OUT.values()) & set(_load(SURVEY2, "_tp2_survey_out").OUT.values()) == set()


def test_THE_SURVEY_COPY_S_CODE_HOLDS_NO_SEED_LITERAL_in_either_band():
    from scripts.GPU.alphazero import seed_band_2027 as SB
    code = _split(SURVEY)[1]
    assert not SB.scan(code) and not re.search(r"\b20[0-9]{7}\b", code)


def _synthetic(survey):
    S, Pf, SEL = survey.load_method()
    registries, named, gen, masks, consumed = S.prior_terms()
    block = SEL.candidate([], [], survey.N_SLOTS)[2]
    return S, Pf, SEL, block, (registries, named, gen, consumed, {}), masks


def test_THE_SURVEY_COPY_S_CONTROLS_ALL_PASS_on_a_synthetic_2027_block():
    """The copied controls (state 12, plants 8, exemptions 6, enumeration 6) and the two-band controls (10), with the
    real prior terms and a block the rule places in an empty band; the repository's literals are not scanned here."""
    survey = _load(SURVEY, "_gen_survey_controls")
    S, Pf, SEL, block, prior, masks = _synthetic(survey)
    rec = {"head": "0" * 40, "registry_sha256": "0" * 64, "base": block[0], "block": block}
    groups = {"state": survey._state_controls(S, Pf, rec, "a" * 64),
              "plants": S._plant_controls(block, prior, masks, {}, True),
              "exemptions": S._exemption_controls(block), "enumeration": Pf._enumeration_controls(S),
              "two-band": survey._two_band_controls(S, SEL, block, prior, masks)}
    assert {k: len(v) for k, v in groups.items()} == {"state": 12, "plants": 8, "exemptions": 6, "enumeration": 6,
                                                      "two-band": 10}
    for k, v in groups.items():
        assert all(ok for _l, ok, _d in v), (k, [r for r in v if not r[1]])


def test_WITHOUT_THE_INSTALL_A_2027_BLOCK_S_PLANT_CONTROLS_FAIL():
    """Necessity: the pinned scanner alone would not see a planted 2027-band seed, so the copied differential controls
    fail -- the extension is what makes them pass."""
    survey = _load(SURVEY, "_gen_survey_uninstalled")
    S, _Pf, SEL, block, prior, masks = _synthetic(survey)
    S.scan = _load("tactical-candidate-seed-survey/01_seed_survey.py", "_pinned_scan").scan
    results = S._plant_controls(block, prior, masks, {}, True)
    assert sum(not ok for _l, ok, _d in results) == 6        # every plant missed; the two fail-closed cases still stop


# ─────────────────────────────── the three modes, rehearsed on a SYNTHETIC repository ───────────────────────────────

PINNED_FILES = ("docs/superpowers/evidence/tactical-candidate-seed-survey/01_seed_survey.py",
                "docs/superpowers/evidence/tactical-candidate-seed-registration/01_collision_proof.py",
                "scripts/GPU/alphazero/injected_defect_controls.py", "scripts/GPU/alphazero/seed_band_2027.py")


def _mode(repo, mode, monkeypatch, capsys):
    """Run the copy's `main` for one mode, loaded from the synthetic repository (so its ROOT, the pinned method's and
    every output are there); the prior terms are the real ones, the scanned tree is the synthetic one."""
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spec = importlib.util.spec_from_file_location(f"_gen_survey_{mode}", repo / "docs/superpowers/evidence" / SURVEY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    code = mod.main(["x", "--mode", mode])
    capsys.readouterr()
    return mod, code, (repo / mod.OUT[mode]).read_text(encoding="utf-8")


def test_THE_THREE_MODES_REHEARSED_ON_A_SYNTHETIC_REPOSITORY(tmp_path, monkeypatch, capsys):
    """survey -> pre -> registration -> post, on a throwaway git repository holding only the copy, the pinned method,
    a synthetic registry and two planted literals (one per band): the rule reads the 2027-band literal and places the
    block above it; every mode's controls pass and each verdict holds. Nothing of the real repository is scanned or
    written; this is no survey of it."""
    from scripts.GPU.alphazero import seed_band_2027 as SB
    repo = tmp_path / "synthetic"

    def git(*a):
        return subprocess.check_output(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo,
                                       text=True).strip()
    for rel in (*PINNED_FILES, f"docs/superpowers/evidence/{SURVEY}"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes((ROOT / rel).read_bytes())
    from scripts.GPU.alphazero import e4_screen_reference as REG
    top = max([SB.BAND[0]] + [hi for lo, hi in REG.ACCOUNTED_SEED_INTERVALS if hi > SB.BAND[0]])
    old, new = SB.BAND_2026[0] + 61_000, top + 23_456          # the planted 2027 literal above every registered block
    reg = repo / "scripts/GPU/alphazero/e4_screen_reference.py"
    reg.write_text(f"ACCOUNTED_SEED_INTERVALS = (\n    ({old}, {old + 10}),\n)\nEXPOSED_SEED_INTERVALS = ()\n"
                   "RETIRED_SEED_INTERVALS = ()\nTEST_ONLY_SEED_INTERVALS = ()\n")
    (repo / "tests").mkdir()
    (repo / "tests/test_planted.py").write_text(f"A = {old + 500}\nB = {new:_}\n")
    (repo / "docs/superpowers/evidence/combination-policy-generate-run").mkdir(parents=True)
    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "synthetic")

    mod, code, survey = _mode(repo, "survey", monkeypatch, capsys)
    assert code == 0, survey[-2500:]
    anchor, base, block = SB.candidate(REG.ACCOUNTED_SEED_INTERVALS, [new], 1000)
    assert anchor == new                                     # the rule read the planted literal
    assert f"in the 2027 band = {new}; BASE = {base}; block = [{block[0]}, {block[1]}) -- 1000 seeds" in survey
    assert survey.endswith("SURVEY VERDICT   : THE COMBINATION-POLICY GENERATION BLOCK IS CLEAN\n")
    passed, total = re.search(r"^controls passed: (\d+)/(\d+)$", survey, re.M).groups()
    assert passed == total and int(total) == 55, total
    S, _Pf, SEL = mod.load_method()
    assert mod.read_record(survey, S, SEL)["block"] == block
    git("add", "-A")
    git("commit", "-qm", "survey record")

    _m, code, pre = _mode(repo, "pre", monkeypatch, capsys)
    assert code == 0 and "EXPECTED STATE (pre): HOLDS" in pre, pre[-2500:]
    git("add", "-A")
    git("commit", "-qm", "pre record")

    import hashlib
    sha = hashlib.sha256((repo / mod.OUT["survey"]).read_bytes()).hexdigest()
    reg.write_text(reg.read_text().replace(f"    ({old}, {old + 10}),\n)",
                                           f"    ({old}, {old + 10}),\n    ({block[0]}, {block[1]}),\n)"))
    (repo / mod.PLAN).write_text(json.dumps(mod.build_plan(block[0], sha), indent=1, sort_keys=True) + "\n")
    git("add", "-A")
    git("commit", "-qm", "registration")
    _m, code, post = _mode(repo, "post", monkeypatch, capsys)
    assert code == 0 and "EXPECTED STATE (post): HOLDS" in post, post[-2500:]
    assert post.endswith("PROOF VERDICT (post): THE COMBINATION-POLICY GENERATION BLOCK IS CLEAN AND THE STATE IS AS "
                         "EXPECTED\n")



# ─────────────────────────────── the generation closing copy (built, never run here) ───────────────────────────────

def test_THE_CLOSING_COPY_IS_THROUGHPUT_ATTEMPT_2_S_WITH_ONLY_THE_STATED_DIFFERENCES():
    """Below the docstring: the paths, the block size and the title, line for line -- the reviewed parallel-worker
    rule is untouched. The docstring is generation's paragraph followed by attempt 2's description unchanged."""
    doc2, code2 = _split(CLOSING2)
    doc, code = _split(CLOSING)
    assert doc.endswith(doc2) and AMENDMENT[0] in doc[:-len(doc2)]
    a, b = code2.splitlines(), code.splitlines()
    assert len(a) == len(b)
    assert {y.strip() for x, y in zip(a, b) if x != y} == {
        'HERE = "docs/superpowers/evidence/combination-policy-generate-seed-closing"',
        'RUN = "docs/superpowers/evidence/combination-policy-generate"',
        'RUN_DIR = "docs/superpowers/evidence/combination-policy-generate-run"', "N_SLOTS = 1000",
        'lines = ["COMBINATION POLICY PILOT -- GENERATION SEED-USAGE DERIVATION (closing)", "=" * 96,'}


def test_THE_CLOSING_COPY_SERVES_THE_GENERATION_RUN_and_holds_no_seed_literal():
    from scripts.GPU.alphazero import seed_band_2027 as SB
    closing = _load(CLOSING, "_gen_closing")
    kind = P.KINDS["generate"]
    assert closing.N_SLOTS == kind["slots"] == 1000 and closing.INPUTS[0] == kind["plan"]
    assert closing.RUN == kind["evidence"] and closing.RUN_DIR == str(pathlib.Path(kind["plan"]).parent)
    assert closing.OUT == "docs/superpowers/evidence/combination-policy-generate-seed-closing/02_derive_usage_run.txt"
    code = _split(CLOSING)[1]
    assert not SB.scan(code) and not re.search(r"\b20[0-9]{7}\b", code)


def test_THE_CLOSING_COPY_S_CONTROLS_ALL_PASS_at_the_registered_base():
    """Its 21 controls (the parallel-worker rule: out-of-order starts, claims, gaps, duplicates, refusals, survivors)
    at the plan's BASE, read from the committed plan."""
    base = json.loads((ROOT / P.KINDS["generate"]["plan"]).read_text(encoding="utf-8"))["base"]
    results = _load(CLOSING, "_gen_closing_c")._controls(base)
    assert len(results) == 21 and all(ok for _l, ok, _d in results), [r for r in results if not r[1]]
