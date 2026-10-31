"""The combination policy pilot's THROUGHPUT CHECK, ATTEMPT 2 (amendment
`docs/superpowers/2026-10-29-combination-policy-throughput-attempt2-amendment.md`), built only: the wrapper
(`docs/superpowers/evidence/combination-policy-throughput2-run/01_wrapper.sh`) and the seed-survey/proof and closing
copies. CPU only; nothing here loads a model, touches a device, runs a seed procedure or opens a real gate.

Each is a copy of attempt 1's with only the stated differences, asserted line by line. THE FAKE-LAUNCH CHECKS run the
REAL attempt-2 wrapper's bytes in a throwaway git repository with attempt 1's stubs (tests/test_combination_policy_launch)
moved to attempt 2's paths: the exact command, the ceiling, and the records a completed or timed-out run leaves.
Since the generation amendment (2026-10-30) generation reads no throughput record, so attempt 2 admits nothing."""
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
WRAPPER1 = EV / "combination-policy-throughput-run/01_wrapper.sh"
WRAPPER = EV / "combination-policy-throughput2-run/01_wrapper.sh"
SURVEY1 = "combination-policy-throughput-seed-survey/01_seed_survey.py"
CLOSING1 = "combination-policy-throughput-seed-closing/01_derive_usage.py"
SURVEY = "combination-policy-throughput2-seed-survey/01_seed_survey.py"
CLOSING = "combination-policy-throughput2-seed-closing/01_derive_usage.py"
GATE_NAME = "COMBINATION_POLICY_THROUGHPUT2_AUTHORIZED"
AMENDMENT_SHA = P.KINDS["throughput2"]["amendment"][1]


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


def test_ATTEMPT_2_IS_REGISTERED_PROVEN_RUN_ONCE_AND_CLOSED_with_the_survey_s_build_plan():
    """RE-ANCHORED 2026-10-30 (survey df0c0ec9, pre-proof fcd3969d): the survey record is CLEAN and the pre-proof HOLDS,
    bound to the survey record's sha256, on a clean tree. RE-ANCHORED 2026-10-30 (registration 22f81be1, post-proof
    1c2f4220): the plan is the survey copy's build_plan; the block (read from the plan and the records; no seed literal
    here, since the proofs scan tests/) is in ACCOUNTED exactly once; the post-proof HOLDS, bound to the same survey
    record, on a clean tree. RE-ANCHORED 2026-10-30 (the run, e95b0e51, and its closing, K1 f79174c4 + K2 ae361458):
    the ONE run hit the 1,790 s ceiling -- outer TIMEOUT (inner 124), record VOID, gate restored, no survivors -- so
    generation is never admitted from it (asked of the real committed readers); the registry it launched with is the
    one the post-proof proved; K1 RECONCILES with no final `claims` event, so the WHOLE block is EXPOSED and RETIRED,
    each exactly once and appended last, ACCOUNTED unchanged, and every seed is refused. The gate is closed. Attempt 1's
    evidence is untouched (its own closed-state test checks its content). RE-ANCHORED 2026-10-30 (the generation
    amendment): generation's admission no longer reads this run at all -- through the real committed readers it reads
    the CPU timing record only. The run's records stay as written: VOID, outer TIMEOUT. Read-only.
    RE-ANCHORED 2026-10-30 (generation's closing, K2 569ba2a4): the block is no longer the last EXPOSED/RETIRED entry;
    every later entry lies above it."""
    import hashlib
    from scripts.GPU.alphazero import combination_policy_throughput2_authorization as G
    from scripts.GPU.alphazero import e4_screen_reference as REG
    assert getattr(G, GATE_NAME) is False
    assert sorted(x.name for x in WRAPPER.parent.iterdir()) == ["01_wrapper.sh", "bg_check.txt", "idle_check.txt",
                                                                 "outer.json", "pgid.txt", "plan.json",
                                                                 "run.stderr.txt", "run.stdout.txt"]
    out = ROOT / P.KINDS["throughput2"]["evidence"]
    assert sorted(x.name for x in out.iterdir()) == ["games.jsonl", "progress.jsonl", "record.json"]
    here = (EV / SURVEY).parent
    assert sorted(x.name for x in here.iterdir() if x.name != "__pycache__") == \
        ["01_seed_survey.py", "02_seed_survey_run.txt", "03_collision_proof_pre_run.txt",
         "04_collision_proof_post_run.txt"]
    assert sorted(x.name for x in (EV / CLOSING).parent.iterdir() if x.name != "__pycache__") == \
        ["01_derive_usage.py", "02_derive_usage_run.txt"]
    raw = (here / "02_seed_survey_run.txt").read_bytes()
    survey, sha = raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()
    assert survey.splitlines()[2].endswith("; working tree clean")
    assert survey.endswith("SURVEY VERDICT   : THE COMBINATION-POLICY THROUGHPUT ATTEMPT-2 BLOCK IS CLEAN\n")
    assert "controls passed: 45/45" in survey
    for rel, mode in (("03_collision_proof_pre_run.txt", "pre"), ("04_collision_proof_post_run.txt", "post")):
        text = (here / rel).read_text(encoding="utf-8")
        assert text.splitlines()[2].endswith("; working tree clean")
        assert f"02_seed_survey_run.txt sha256 {sha} (read at " in text
        assert f"EXPECTED STATE ({mode}): HOLDS" in text and "controls passed: 45/45" in text
        assert text.endswith(f"PROOF VERDICT ({mode}): THE COMBINATION-POLICY THROUGHPUT ATTEMPT-2 BLOCK IS CLEAN AND "
                             "THE STATE IS AS EXPECTED\n")
    post = (here / "04_collision_proof_post_run.txt").read_text(encoding="utf-8")
    # the run: outer TIMEOUT over a VOID record, launched on the proved registry; generation never admitted from it
    outer = json.loads((WRAPPER.parent / "outer.json").read_text(encoding="utf-8"))
    record = json.loads((out / "record.json").read_text(encoding="utf-8"))
    assert (outer["inner_exit"], outer["outcome"], outer["gate_restored"], outer["survivors"], outer["record_by"],
            outer["term_s"], outer["grace_s"]) == (124, "TIMEOUT", True, "", "script", 1790, 10)
    assert record["status"] == "VOID" and record["void"]["step"] == "terminated" and record["head"] == outer["head"]
    assert record["run"] == "throughput2" and record["amendment_sha256"] == AMENDMENT_SHA
    assert P.outer_faults(outer, record)
    reads = []

    def read_record(rel):
        reads.append(rel)
        return P._read_record(rel)
    assert P.admitted("generate", read_record, P._read_outer) == {P.TIMING_EVIDENCE: "COMPLETED"}
    assert reads == [P.TIMING_EVIDENCE]
    reg_rel = "scripts/GPU/alphazero/e4_screen_reference.py"
    launched = subprocess.run(["git", "-C", str(ROOT), "show", f"{outer['head']}:{reg_rel}"],
                              capture_output=True, check=True).stdout
    assert f"\nregistry {reg_rel} sha256 {hashlib.sha256(launched).hexdigest()}\n" in post     # proved = launched
    plan = json.loads((ROOT / P.KINDS["throughput2"]["plan"]).read_text(encoding="utf-8"))
    n = P.KINDS["throughput2"]["slots"]
    assert type(plan["base"]) is int and plan["n_slots"] == n == 144
    assert plan == _load(SURVEY, "_tp2_survey_registered").build_plan(plan["base"], sha)
    block = (plan["base"], plan["base"] + n)
    assert f"block = [{block[0]}, {block[1]}) -- {n} seeds" in survey
    assert f"THE BLOCK, read from the record: [{block[0]}, {block[1]}) -- BASE {block[0]}" in post
    assert record["seeds"] == {"base": block[0], "n_slots": n}
    # the closing: no final `claims` event, so the WHOLE block is EXPOSED and RETIRED, each once and last
    closing = (EV / CLOSING).parent.joinpath("02_derive_usage_run.txt").read_text(encoding="utf-8")
    assert closing.splitlines()[2].endswith("; working tree clean")
    assert "claims events: []\n" in closing and "controls passed: 21/21" in closing
    assert closing.endswith(f"VERDICT: RECONCILES -- EXPOSE {list(block)}; RETIRE {list(block)}\n")
    assert REG.ACCOUNTED_SEED_INTERVALS.count(block) == 1
    later = REG.ACCOUNTED_SEED_INTERVALS[REG.ACCOUNTED_SEED_INTERVALS.index(block) + 1:]
    assert all(lo >= block[1] for lo, _hi in later)
    for name in ("EXPOSED_SEED_INTERVALS", "RETIRED_SEED_INTERVALS"):      # generation's closing lies above
        ivs = getattr(REG, name)
        assert ivs.count(block) == 1 and all(lo >= block[1] for lo, _hi in ivs[ivs.index(block) + 1:]), name
    assert not any(lo < block[1] and block[0] < hi and (lo, hi) != block
                   for lo, hi in REG.EXPOSED_SEED_INTERVALS + REG.RETIRED_SEED_INTERVALS + REG.TEST_ONLY_SEED_INTERVALS)
    assert all(REG.seed_is_accounted(s) and REG.seed_is_exposed(s) and REG.seed_is_retired(s) and
               REG.seed_is_unavailable(s) and not REG.seed_is_test_only(s) for s in range(*block))
    assert subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "e30e277c", "HEAD", "--",
                           "docs/superpowers/evidence/combination-policy-throughput",
                           "docs/superpowers/evidence/combination-policy-throughput-run",
                           "docs/superpowers/evidence/combination-policy-throughput-seed-survey",
                           "docs/superpowers/evidence/combination-policy-throughput-seed-closing"]).returncode == 0


# ─────────────────────────────── the wrapper's bytes ───────────────────────────────

def _code(path):
    return path.read_text().split("set -u\n", 1)[1]


def test_THE_WRAPPER_IS_ATTEMPT_1_S_WITH_ONLY_THE_STATED_DIFFERENCES():
    """The paths, the gate, `run throughput2` and TERM_S, line for line; everything else byte-identical. Its header is
    attempt 2's paragraph followed by attempt 1's header unchanged."""
    one, two = _code(WRAPPER1).splitlines(), _code(WRAPPER).splitlines()
    assert len(one) == len(two)
    changed = [(a, b) for a, b in zip(one, two) if a != b]
    assert [b for _a, b in changed] == [
        "OUT_REL=docs/superpowers/evidence/combination-policy-throughput2-run",
        "RUN_REL=docs/superpowers/evidence/combination-policy-throughput2",
        "GATE_REL=scripts/GPU/alphazero/combination_policy_throughput2_authorization.py",
        f"GATE={GATE_NAME}", "TERM_S=1790",
        '/usr/bin/perl -e "$SUPERVISOR" "$OUT_REL/pgid.txt" "$TERM_S" "$GRACE_S" env MLX_ENABLE_TF32=0 PYTHONPATH=. '
        '"$PY" -B -m "$MODULE" run throughput2 --checkpoint "$CKPT" \\'], changed
    head1 = WRAPPER1.read_text().split("set -u\n", 1)[0]
    head2 = WRAPPER.read_text().split("set -u\n", 1)[0]
    assert head2.startswith("#!/bin/bash\n# THE COMBINATION POLICY PILOT: THE THROUGHPUT CHECK, ATTEMPT 2")
    assert head2.endswith(head1[len("#!/bin/bash\n"):])
    assert os.access(WRAPPER, os.R_OK) and subprocess.run(["bash", "-n", str(WRAPPER)]).returncode == 0


def test_THE_CEILING_IS_THE_AMENDMENT_S_30_MINUTES():
    """TERM at 1,790 s, KILL at 1,800 s, the cleanup observation bounded through 1,810 s."""
    w = _code(WRAPPER)
    term, grace = (int(re.search(rf"^{k}=(\d+)$", w, re.M).group(1)) for k in ("TERM_S", "GRACE_S"))
    assert (term, grace) == (1790, 10) and term + grace == 30 * 60 and term + 2 * grace == 1810


def test_THE_WRAPPER_SERVES_THE_PILOT_S_THROUGHPUT2_PATHS():
    w = _code(WRAPPER)
    out, run = (re.search(rf"^{k}=(\S+)$", w, re.M).group(1) for k in ("OUT_REL", "RUN_REL"))
    kind = P.KINDS["throughput2"]
    assert run == kind["evidence"] and out == str(pathlib.Path(kind["plan"]).parent) == str(WRAPPER.parent.relative_to(ROOT))
    assert re.search(r"^GATE_REL=(\S+)$", w, re.M).group(1) == \
        str(pathlib.Path(kind["gate"][0].__file__).relative_to(ROOT))
    assert kind["gate"][1] == GATE_NAME and f"RECORD_KIND={P.RECORD_KIND}\n" in w
    one = P.KINDS["throughput"]
    assert {run, out} & {one["evidence"], str(pathlib.Path(one["plan"]).parent)} == set()


# ─────────────────────────────── fake launches of the REAL attempt-2 wrapper ───────────────────────────────

#: attempt 1's stub at attempt 2's paths; its COMPLETED record also carries the amendment's sha256, as the real one does
STUB_PILOT2 = STUB_PILOT.replace("combination-policy-throughput", "combination-policy-throughput2").replace(
    '"head": head, "written_by": "script"}', '"head": head, "written_by": "script",\n'
    '                                                 "amendment_sha256": os.environ["FAKE_AMENDMENT_SHA"]}')
assert STUB_PILOT2.count("combination-policy-throughput2") == 2 and "FAKE_AMENDMENT_SHA" in STUB_PILOT2


def _fake_repo(tmp, term_s=None):
    """A throwaway repository: the real attempt-2 wrapper (TERM_S changed only when a test says so), its gate opened by
    a one-line commit, and three stubs."""
    repo = tmp / "repo"
    pkg = repo / "scripts/GPU/alphazero"
    pkg.mkdir(parents=True)
    for d in (repo / "scripts", repo / "scripts/GPU", pkg):
        (d / "__init__.py").write_text("")
    gate = pkg / "combination_policy_throughput2_authorization.py"
    gate.write_text(f'"""stub gate"""\n{GATE_NAME} = False\n')
    (pkg / "idle_gate.py").write_text(STUB_IDLE)
    (pkg / "background_compute.py").write_text(STUB_BG)
    (pkg / "combination_policy_pilot.py").write_text(STUB_PILOT2)
    wdir = repo / "docs/superpowers/evidence/combination-policy-throughput2-run"
    wdir.mkdir(parents=True)
    raw = WRAPPER.read_bytes()
    if term_s is not None:
        assert raw.count(b"\nTERM_S=1790\n") == 1
        raw = raw.replace(b"\nTERM_S=1790\n", f"\nTERM_S={term_s}\n".encode())
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
               FAKE_CARD_SHA=P.CARD_SHA256, FAKE_AMENDMENT_SHA=AMENDMENT_SHA)
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
    assert not (repo / "docs/superpowers/evidence/combination-policy-throughput2").exists()
    assert f"{GATE_NAME} = False\n" in gate.read_text()


def test_FAKE_LAUNCH_AN_IDLE_MACHINE_REACHES_RUN_THROUGHPUT2_with_its_exact_arguments_and_ceiling(tmp_path):
    repo, wdir, gate = _fake_repo(tmp_path)
    p = _launch(repo, wdir)
    assert p.returncode == 0, p.stderr
    launched = json.loads((wdir / "run.stdout.txt").read_text().split("FAKE_LAUNCHED", 1)[1])
    ckpt = str(repo.resolve() / "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors")
    assert launched["argv"] == ["run", "throughput2", "--checkpoint", ckpt] and launched["tf32"] == "0"
    assert (wdir / "pgid.txt").read_text() == f"{launched['pgid']}\n"
    outer = json.loads((wdir / "outer.json").read_text())
    assert outer["outcome"] == "COMPLETED" and outer["survivors"] == "" and outer["gate_restored"] is True
    assert outer["term_s"] == 1790 and outer["grace_s"] == 10 and outer["record_by"] == "none"
    assert f"{GATE_NAME} = False\n" in gate.read_text()


def test_FAKE_LAUNCH_THE_CEILING_TERMINATES_THE_WHOLE_GROUP_at_attempt_2_s_paths(tmp_path):
    """TERM_S changed to 2 in this copy only: the run and its 'worker' outlive it; the group TERM ends both, and the
    wrapper's VOID record lands in attempt 2's evidence directory."""
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
    rec = json.loads((repo / "docs/superpowers/evidence/combination-policy-throughput2/record.json").read_text())
    assert outer["record_by"] == "wrapper" and rec["status"] == "VOID" and rec["void"]["step"] == "timeout"
    assert P.outer_faults(outer, {"head": outer["head"]})
    assert f"{GATE_NAME} = False\n" in gate.read_text()


def _commit_evidence(repo, monkeypatch):
    """Commit the fake run's outputs, then point the REAL committed readers at the throwaway repository."""
    def git(*a):
        return subprocess.check_output(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo,
                                       text=True).strip()
    monkeypatch.setattr(P, "ROOT", repo)
    monkeypatch.setattr(P.M, "_git", git)
    git("add", "docs/superpowers/evidence")
    git("commit", "-qm", "the run's evidence")


def test_ATTEMPT_2_S_PATHS_NO_LONGER_ADMIT_GENERATION_even_after_a_completed_committed_run(tmp_path, monkeypatch):
    """RE-ANCHORED 2026-10-30 (the generation amendment): generation reads only the committed CPU timing, which this
    throwaway repository lacks, so attempt 2's completed and committed fake run admits nothing; its records pass
    outer_faults and are simply not read."""
    repo, wdir, _gate = _fake_repo(tmp_path)
    p = _launch(repo, wdir, run="record")
    outer = json.loads((wdir / "outer.json").read_text())
    assert p.returncode == 0 and outer["outcome"] == "COMPLETED" and outer["record_by"] == "script", outer
    _commit_evidence(repo, monkeypatch)
    rec = P._read_record(P.KINDS["throughput2"]["evidence"])
    assert rec["amendment_sha256"] == AMENDMENT_SHA and not P.outer_faults(P._read_outer(str(wdir.relative_to(repo))),
                                                                           rec)
    with pytest.raises(P.Refused, match="combination-policy-timing/record.json does not exist"):
        P.admitted("generate", P._read_record, P._read_outer)


def test_AN_OUTER_TIMEOUT_OVER_ATTEMPT_2_S_COMPLETED_RECORD_STANDS_AS_WRITTEN(tmp_path, monkeypatch):
    """The user's ruling (2026-10-29), at attempt 2's paths: the script wrote COMPLETED, then the run outlived the
    ceiling (TERM_S 2 in this copy only). Both records stand as written, and outer_faults says the run did not
    succeed."""
    repo, wdir, _gate = _fake_repo(tmp_path, term_s=2)
    p = _launch(repo, wdir, run="record_outlive")
    outer = json.loads((wdir / "outer.json").read_text())
    rec = json.loads((repo / "docs/superpowers/evidence/combination-policy-throughput2/record.json").read_text())
    assert p.returncode == 124 and outer["outcome"] == "TIMEOUT" and outer["record_by"] == "script", outer
    assert rec["status"] == "COMPLETED" and rec["amendment_sha256"] == AMENDMENT_SHA
    _commit_evidence(repo, monkeypatch)
    assert "inner_exit is 124, not 0" in P.outer_faults(P._read_outer(str(wdir.relative_to(repo))),
                                                        P._read_record(P.KINDS["throughput2"]["evidence"]))


# ─────────────────────────────── the seed-procedure copies (built, never run here) ───────────────────────────────

def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, EV / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                     # module level only: no mode runs, nothing is written
    return mod


def _split(rel):
    text = (EV / rel).read_text()
    end = text.index('"""', 3)
    return text[3:end], text[end + 3:]


@pytest.mark.parametrize("one, two, changes", [
    (SURVEY1, SURVEY, {
        'HERE = "docs/superpowers/evidence/combination-policy-throughput2-seed-survey"',
        'PLAN = "docs/superpowers/evidence/combination-policy-throughput2-run/plan.json"',
        'VERDICT_OK = "THE COMBINATION-POLICY THROUGHPUT ATTEMPT-2 BLOCK IS CLEAN"',
        'spec = importlib.util.spec_from_file_location(f"_combination_policy_throughput2_{key}", p)',
        'return {"kind": "combination-policy-throughput2-plan/1", "base": base, "n_slots": N_SLOTS,',
        'faults.append("the throughput attempt-2 plan is absent or not committed")',
        'faults.append("the throughput attempt-2 plan is not build_plan of the surveyed block")',
        'say(f"COMBINATION POLICY PILOT -- THROUGHPUT ATTEMPT-2 GAME-SEED {\'SURVEY\' if mode == \'survey\' else '
        '\'COLLISION PROOF, mode \' + mode.upper()}")',
        "say(f\"PROOF VERDICT ({mode}): {'THE COMBINATION-POLICY THROUGHPUT ATTEMPT-2 BLOCK IS CLEAN AND THE STATE IS "
        "AS EXPECTED' if ok else 'NOT PROVEN -- STOP'}\""}),
    (CLOSING1, CLOSING, {
        'HERE = "docs/superpowers/evidence/combination-policy-throughput2-seed-closing"',
        'RUN = "docs/superpowers/evidence/combination-policy-throughput2"',
        'RUN_DIR = "docs/superpowers/evidence/combination-policy-throughput2-run"',
        'lines = ["COMBINATION POLICY PILOT -- THROUGHPUT ATTEMPT-2 SEED-USAGE DERIVATION (closing)", "=" * 96,'})],
    ids=["survey", "closing"])
def test_THE_SEED_COPY_IS_ATTEMPT_1_S_WITH_ONLY_THE_STATED_DIFFERENCES(one, two, changes):
    """Below the docstring: the paths, the plan's kind, the module names, the titles and verdict lines, line for line.
    The docstring is attempt 2's paragraph followed by attempt 1's description unchanged."""
    doc1, code1 = _split(one)
    doc2, code2 = _split(two)
    a, b = code1.splitlines(), code2.splitlines()
    assert len(a) == len(b)
    assert {y.strip() for x, y in zip(a, b) if x != y} == changes
    assert doc2.endswith(doc1) and "2026-10-29-combination-policy-throughput-attempt2-amendment.md" in doc2[:-len(doc1)]


def test_THE_SEED_COPIES_MATCH_THE_RUN_THEY_SERVE():
    survey, closing = _load(SURVEY, "_tp2_survey"), _load(CLOSING, "_tp2_closing")
    kind = P.KINDS["throughput2"]
    assert survey.N_SLOTS == closing.N_SLOTS == kind["slots"] == 144
    assert survey.PLAN == kind["plan"] == closing.INPUTS[0]
    assert closing.RUN == kind["evidence"] and closing.RUN_DIR == str(pathlib.Path(survey.PLAN).parent)
    assert survey.build_plan(202700000, "0" * 64) == {"kind": "combination-policy-throughput2-plan/1",
                                                      "base": 202700000, "n_slots": 144,
                                                      "survey_record_sha256": "0" * 64}
    assert survey.PINNED == _load(SURVEY1, "_tp1_survey").PINNED
    assert set(survey.OUT.values()) & set(_load(SURVEY1, "_tp1_survey_out").OUT.values()) == set()
    assert closing.OUT != _load(CLOSING1, "_tp1_closing").OUT


def test_THE_SEED_COPIES_CODE_HOLDS_NO_SEED_LITERAL():
    for rel in (SURVEY, CLOSING):
        assert not re.search(r"\b20[0-9]{7}\b", _split(rel)[1]), rel


def test_THE_SURVEY_COPY_S_STATE_CONTROLS_ALL_PASS():
    """The pinned method loaded read-only; the copy's own synthetic pre/post registry and plan cases."""
    survey = _load(SURVEY, "_tp2_survey_state")
    S, Pf, _SEL = survey.load_method()
    assert S.N == 144 and S.PLAN == survey.PLAN
    rec = {"head": "0" * 40, "registry_sha256": "0" * 64, "base": 202700000, "block": (202700000, 202700144)}
    results = survey._state_controls(S, Pf, rec, "a" * 64)
    assert len(results) == 12 and all(ok for _l, ok, _d in results), [r for r in results if not r[1]]


def test_THE_CLOSING_COPY_S_CONTROLS_ALL_PASS():
    results = _load(CLOSING, "_tp2_closing_c")._controls(202700000)
    assert len(results) == 21 and all(ok for _l, ok, _d in results), [r for r in results if not r[1]]
