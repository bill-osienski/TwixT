"""The combination policy pilot's THROUGHPUT-CHECK LAUNCH, built only (ledger, decision note after entry 40): the wrapper
(`docs/superpowers/evidence/combination-policy-throughput-run/01_wrapper.sh`) and the seed-survey/proof and closing
copies. CPU only; nothing here loads a model, touches a device, runs a seed procedure or opens a real gate.

THE FAKE-LAUNCH CHECKS run the REAL wrapper's bytes in a throwaway git repository holding only what it needs: its gate
(opened by a one-line commit), stub `idle_gate` and `background_compute` that report as the test says, and a stub
`combination_policy_pilot` that prints what it was launched with and, when told, leaves a process in its group and
kills its supervisor, outlives the ceiling, keeps a worker that ignores TERM, or deletes pgid.txt. THE CLOSING CHECKS feed the copy the progress.jsonl the REAL `_generate`
writes through the REAL record boundary (fake seams, temporary output)."""
import ast
import importlib.util
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time

import pytest

from scripts.GPU.alphazero import combination_policy_pilot as P
from tests.test_combination_policy_pilot import _REAL_CONCLUDE, GenHarness, _fake_rows

ROOT = pathlib.Path(__file__).resolve().parents[1]
EV = ROOT / "docs/superpowers/evidence"
WRAPPER = EV / "combination-policy-throughput-run/01_wrapper.sh"
SURVEY = "combination-policy-throughput-seed-survey/01_seed_survey.py"
CLOSING = "combination-policy-throughput-seed-closing/01_derive_usage.py"
GATE_NAME = "COMBINATION_POLICY_THROUGHPUT_AUTHORIZED"


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


def test_THE_THROUGHPUT_BLOCK_IS_REGISTERED_PROVEN_RUN_ONCE_AND_CLOSED_with_the_survey_s_build_plan():
    """RE-ANCHORED 2026-10-29 (registration, 88c5d162): the survey (cd4217a9) and pre-proof (086d6e37) records stand,
    the proof bound to the survey record's sha256; the plan is the survey copy's build_plan; the block is in ACCOUNTED
    exactly once. RE-ANCHORED 2026-10-29 (post-proof, 20486339): the post-registration proof HOLDS, bound to the same
    survey record, on a clean tree, and the registry it proved is the registry the run launched with.
    RE-ANCHORED 2026-10-29 (the run, dba747bf, and its closing, K1 b69515d5 + K2 e30e277c): the ONE run hit the 890 s
    ceiling -- outer TIMEOUT (inner 124), record VOID, gate restored, no survivors -- so generation is never admitted
    from it; K1 RECONCILES with no final `claims` event, so the WHOLE block is EXPOSED and RETIRED, each exactly once
    and appended last, ACCOUNTED unchanged, and every seed is refused. The gate is closed. Read-only.
    RE-ANCHORED 2026-10-30 (attempt 2's closing, ae361458): the block is no longer the last EXPOSED/RETIRED entry;
    every later entry lies above it."""
    import hashlib
    import subprocess
    from scripts.GPU.alphazero import combination_policy_throughput_authorization as G
    from scripts.GPU.alphazero import e4_screen_reference as REG
    assert getattr(G, GATE_NAME) is False
    run = EV / "combination-policy-throughput-run"
    assert sorted(x.name for x in run.iterdir()) == ["01_wrapper.sh", "bg_check.txt", "idle_check.txt", "outer.json",
                                                      "pgid.txt", "plan.json", "run.stderr.txt", "run.stdout.txt"]
    out = EV / "combination-policy-throughput"
    assert sorted(x.name for x in out.iterdir()) == ["games.jsonl", "progress.jsonl", "record.json"]
    here = (EV / SURVEY).parent
    assert sorted(x.name for x in here.iterdir() if x.name != "__pycache__") == \
        ["01_seed_survey.py", "02_seed_survey_run.txt", "03_collision_proof_pre_run.txt",
         "04_collision_proof_post_run.txt"]
    assert sorted(x.name for x in (EV / CLOSING).parent.iterdir() if x.name != "__pycache__") == \
        ["01_derive_usage.py", "02_derive_usage_run.txt"]
    raw = (here / "02_seed_survey_run.txt").read_bytes()
    survey, sha = raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()
    assert survey.endswith("SURVEY VERDICT   : THE COMBINATION-POLICY THROUGHPUT BLOCK IS CLEAN\n")
    assert "controls passed: 45/45" in survey
    for rel, mode in (("03_collision_proof_pre_run.txt", "pre"), ("04_collision_proof_post_run.txt", "post")):
        text = (here / rel).read_text(encoding="utf-8")
        assert text.splitlines()[2].endswith("; working tree clean")
        assert f"02_seed_survey_run.txt sha256 {sha} (read at " in text
        assert f"EXPECTED STATE ({mode}): HOLDS" in text and "controls passed: 45/45" in text
        assert text.endswith(f"PROOF VERDICT ({mode}): THE COMBINATION-POLICY THROUGHPUT BLOCK IS CLEAN AND THE STATE "
                             "IS AS EXPECTED\n")
    post = (here / "04_collision_proof_post_run.txt").read_text(encoding="utf-8")
    # the run: outer TIMEOUT over a VOID record, launched on the proved registry; generation never admitted from it
    outer = json.loads((run / "outer.json").read_text(encoding="utf-8"))
    record = json.loads((out / "record.json").read_text(encoding="utf-8"))
    assert (outer["inner_exit"], outer["outcome"], outer["gate_restored"], outer["survivors"], outer["record_by"]) == \
        (124, "TIMEOUT", True, "", "script")
    assert record["status"] == "VOID" and record["void"]["step"] == "terminated" and record["head"] == outer["head"]
    assert P.outer_faults(outer, record)
    reg_rel = "scripts/GPU/alphazero/e4_screen_reference.py"
    launched = subprocess.run(["git", "-C", str(ROOT), "show", f"{outer['head']}:{reg_rel}"],
                              capture_output=True, check=True).stdout
    assert f"e4_screen_reference.py sha256 {hashlib.sha256(launched).hexdigest()}\n" in post    # proved = launched
    survey_mod = _load(SURVEY, "_survey_throughput_registered")
    plan = json.loads((ROOT / P.KINDS["throughput"]["plan"]).read_text(encoding="utf-8"))
    n = P.KINDS["throughput"]["slots"]
    assert type(plan["base"]) is int and plan["n_slots"] == n and plan == survey_mod.build_plan(plan["base"], sha)
    block = (plan["base"], plan["base"] + n)
    assert f"block = [{block[0]}, {block[1]}) -- {n} seeds" in survey
    assert f"THE BLOCK, read from the record: [{block[0]}, {block[1]}) -- BASE {block[0]}\n" in post
    assert record["seeds"] == {"base": block[0], "n_slots": n}
    # the closing: no final `claims` event, so the WHOLE block is EXPOSED and RETIRED, each once and last
    closing = (EV / CLOSING).parent.joinpath("02_derive_usage_run.txt").read_text(encoding="utf-8")
    assert "claims events: []\n" in closing and "controls passed: 21/21" in closing
    assert closing.endswith(f"VERDICT: RECONCILES -- EXPOSE {list(block)}; RETIRE {list(block)}\n")
    assert REG.ACCOUNTED_SEED_INTERVALS.count(block) == 1
    later = REG.ACCOUNTED_SEED_INTERVALS[REG.ACCOUNTED_SEED_INTERVALS.index(block) + 1:]
    assert all(lo >= block[1] for lo, _hi in later)
    for name in ("EXPOSED_SEED_INTERVALS", "RETIRED_SEED_INTERVALS"):      # attempt 2's closing (2026-10-30) lies above
        ivs = getattr(REG, name)
        assert ivs.count(block) == 1 and all(lo >= block[1] for lo, _hi in ivs[ivs.index(block) + 1:]), name
    assert not any(lo < block[1] and block[0] < hi and (lo, hi) != block
                   for lo, hi in REG.EXPOSED_SEED_INTERVALS + REG.RETIRED_SEED_INTERVALS + REG.TEST_ONLY_SEED_INTERVALS)
    assert all(REG.seed_is_accounted(s) and REG.seed_is_exposed(s) and REG.seed_is_retired(s) and
               REG.seed_is_unavailable(s) and not REG.seed_is_test_only(s) for s in range(*block))


# ─────────────────────────────── the wrapper's bytes ───────────────────────────────

def _code(path):
    return path.read_text().split("set -u\n", 1)[1]


def test_THE_WRAPPER_IS_STRENGTH2_S_WITH_ONLY_THE_STATED_DIFFERENCES():
    """The paths, the gate, the module and `run throughput`, the record kind, the ceiling, the run's process group
    (pgid.txt, written by the supervisor; survivors include the group), the escalation outliving the main process, and
    that observation bounded at GRACE_S s after KILL (exit 125, cleanup unconfirmed) (2026-10-29). Everything else
    byte-identical."""
    two = _code(EV / "mirror-halving-strength2-run/01_wrapper.sh").splitlines()
    tp = _code(WRAPPER).splitlines()
    removed, added = sorted(set(two) - set(tp)), sorted(set(tp) - set(two))
    assert set(added) == {
        "OUT_REL=docs/superpowers/evidence/combination-policy-throughput-run",
        "RUN_REL=docs/superpowers/evidence/combination-policy-throughput",
        "GATE_REL=scripts/GPU/alphazero/combination_policy_throughput_authorization.py",
        f"GATE={GATE_NAME}", "MODULE=scripts.GPU.alphazero.combination_policy_pilot", "TERM_S=890",
        "RECORD_KIND=combination-policy-pilot-record/1",
        '    if PGID=$(cat "$OUT_REL/pgid.txt" 2>/dev/null) && [[ "$PGID" =~ ^[0-9]+$ ]]; then',
        "      survivors=$( { printf '%s\\n' $survivors; pgrep -g \"$PGID\"; } | grep -E '^[0-9]+$' | sort -un | "
        "tr '\\n' ' ')",
        '    else', '      survivors="unknown: no readable pgid.txt ${survivors}"',
        "for f in idle_check.txt bg_check.txt pgid.txt run.stdout.txt run.stderr.txt outer.json; do",
        "use strict; use warnings; use Fcntl; use POSIX \":sys_wait_h\";",
        "my ($pgid_file, $term, $grace, @cmd) = @ARGV;",
        "my $pf;", 'sysopen($pf, $pgid_file, O_WRONLY | O_CREAT | O_EXCL) && print($pf "$pid\\n") && close($pf)',
        '  or do { kill "KILL", -$pid; waitpid($pid, 0); exit 126; };',
        '      if [ "$inner" = 125 ] || kill -0 -- "-$PGID" 2>/dev/null; then    # the supervisor gave up, or the '
        'group remains',
        '        survivors="unconfirmed cleanup: process group $PGID ${survivors}"',      # its `fi` pre-exists
        "  124|125|137) outcome=TIMEOUT ;;",
        "my ($termed, $killed, $gave_up, $timed_out) = (0, 0, 0, 0);",
        'my $stop = sub { if (!$termed) { $termed = 1; kill "TERM", -$pid; alarm $grace; }',
        '                 elsif (!$killed) { $killed = 1; kill "KILL", -$pid; alarm $grace; } else { $gave_up = 1; } };',
        "my $st;",
        "until ($gave_up) {                     # the ceiling stays armed while the run or any of its group survives, "
        "and at",
        "  $st = $? if !defined $st && waitpid($pid, WNOHANG) == $pid;    # most GRACE_S s after KILL",
        "  last if defined $st && !kill(0, -$pid);",
        "  select(undef, undef, undef, 0.1);",
        "exit 125 if $gave_up;",
        '/usr/bin/perl -e "$SUPERVISOR" "$OUT_REL/pgid.txt" "$TERM_S" "$GRACE_S" env MLX_ENABLE_TF32=0 PYTHONPATH=. '
        '"$PY" -B -m "$MODULE" run throughput --checkpoint "$CKPT" \\'}, added
    assert len(removed) == 17, removed      # seven settings, the one-shot list, the launch line, TIMEOUT's codes, and
    assert len(tp) == len(two) + 18         # seven perl lines; plus the group check, the pgid write, the bounded wait


def test_THE_CEILING_IS_THE_CARD_S_15_MINUTES():
    w = _code(WRAPPER)
    term, grace = (int(re.search(rf"^{k}=(\d+)$", w, re.M).group(1)) for k in ("TERM_S", "GRACE_S"))
    assert term + grace == 15 * 60 and grace == 10


def test_THE_IDLE_GATE_THEN_THE_BACKGROUND_CHECK_PRECEDE_THE_ONE_LAUNCH():
    w = _code(WRAPPER)
    idle, bg, started, perl = (w.index(x) for x in ("idle_gate wait", "background_compute check", "\nstarted=true",
                                                    '/usr/bin/perl -e "$SUPERVISOR"'))
    assert idle < bg < started < perl and w.count("/usr/bin/perl -e") == 1
    assert w.count("idle_gate wait") == w.count("background_compute check") == 1


def test_THE_WRAPPER_SERVES_THE_PILOT_S_THROUGHPUT_PATHS():
    w = _code(WRAPPER)
    out, run = (re.search(rf"^{k}=(\S+)$", w, re.M).group(1) for k in ("OUT_REL", "RUN_REL"))
    assert run == P.KINDS["throughput"]["evidence"] and out == str(pathlib.Path(P.KINDS["throughput"]["plan"]).parent)
    assert re.search(r"^GATE_REL=(\S+)$", w, re.M).group(1) == \
        str(pathlib.Path(P.KINDS["throughput"]["gate"][0].__file__).relative_to(ROOT))
    assert P.KINDS["throughput"]["gate"][1] == GATE_NAME and f"RECORD_KIND={P.RECORD_KIND}\n" in w


# ─────────────────────────────── fake launches of the REAL wrapper ───────────────────────────────

STUB_IDLE = """import os, sys
print("STUB_IDLE_GATE", sys.argv[1:], os.environ.get("FAKE_IDLE"))
sys.exit(0 if os.environ.get("FAKE_IDLE") == "idle" else 1)
"""
STUB_BG = """import os, sys
state = os.environ.get("FAKE_BG")
print("STUB_BACKGROUND_COMPUTE", sys.argv[1:], state)
sys.exit({"clear": 0, "boinc": 1}.get(state, 3))
"""
STUB_PILOT = """import json, os, pathlib, subprocess, sys, time
print("FAKE_LAUNCHED", json.dumps({"argv": sys.argv[1:], "tf32": os.environ.get("MLX_ENABLE_TF32"),
                                   "pgid": os.getpgid(0)}), flush=True)
mode, here = os.environ.get("FAKE_RUN", "ok"), pathlib.Path(os.environ["FAKE_OUT"])
if mode in ("linger", "outlive"):        # a 'worker': same process group, a command line without the module's name
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (here / "child.pid").write_text(str(child.pid))
if mode == "linger":                    # the supervisor lost (the only way a group member now outlives it)
    os.kill(os.getppid(), 9)
if mode == "deaf":                      # a worker that ignores TERM; the main process dies of it
    child = subprocess.Popen([sys.executable, "-c",
                              "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
    (here / "child.pid").write_text(str(child.pid))
    time.sleep(60)
if mode == "outlive":
    time.sleep(60)
if mode in ("record", "record_outlive"):   # the script's own COMPLETED record (the wrapper runs from the repo root)
    rec = pathlib.Path("docs/superpowers/evidence/combination-policy-throughput")
    rec.mkdir()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (rec / "record.json").write_text(json.dumps({"status": "COMPLETED", "card_sha256": os.environ["FAKE_CARD_SHA"],
                                                 "head": head, "written_by": "script"}))
if mode == "record_outlive":            # ... and then the run outlives the ceiling
    time.sleep(60)
if mode == "lose_pgid":                 # the wrapper runs from the repository root
    pathlib.Path("docs/superpowers/evidence/combination-policy-throughput-run/pgid.txt").unlink()
"""


def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True,
                          capture_output=True, text=True).stdout


def _fake_repo(tmp, term_s=None, sentinel=None):
    """A throwaway repository: the real wrapper (with TERM_S changed, and the group probe also counting a sentinel pid,
    only when a test says so), its gate opened by a one-line commit, and three stubs."""
    repo = tmp / "repo"
    pkg = repo / "scripts/GPU/alphazero"
    pkg.mkdir(parents=True)
    for d in (repo / "scripts", repo / "scripts/GPU", pkg):
        (d / "__init__.py").write_text("")
    gate = pkg / "combination_policy_throughput_authorization.py"
    gate.write_text(f'"""stub gate"""\n{GATE_NAME} = False\n')
    (pkg / "idle_gate.py").write_text(STUB_IDLE)
    (pkg / "background_compute.py").write_text(STUB_BG)
    (pkg / "combination_policy_pilot.py").write_text(STUB_PILOT)
    wdir = repo / "docs/superpowers/evidence/combination-policy-throughput-run"
    wdir.mkdir(parents=True)
    raw = WRAPPER.read_bytes()
    if term_s is not None:
        assert raw.count(b"\nTERM_S=890\n") == 1
        raw = raw.replace(b"\nTERM_S=890\n", f"\nTERM_S={term_s}\n".encode())
    if sentinel is not None:
        assert raw.count(b"kill(0, -$pid)") == 1
        raw = raw.replace(b"kill(0, -$pid)", f"(kill(0, -$pid) || kill(0, {sentinel}))".encode())
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
    (repo / "scratch").mkdir()                    # the stub's own notes: ignored, so the tree stays clean
    return repo, wdir, gate


def _launch(repo, wdir, idle="idle", bg="clear", run="ok", timeout=60):
    env = {k: v for k, v in os.environ.items() if k not in ("MLX_ENABLE_TF32", "PYTHONPATH")}
    env.update(FAKE_IDLE=idle, FAKE_BG=bg, FAKE_RUN=run, FAKE_OUT=str(repo / "scratch"), PYTHONDONTWRITEBYTECODE="1",
               FAKE_CARD_SHA=P.CARD_SHA256)
    return subprocess.run(["bash", str(wdir / "01_wrapper.sh")], cwd=str(repo), env=env, capture_output=True,
                          text=True, timeout=timeout)


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.parametrize("idle, bg, why", [("busy", "clear", "did not become idle (idle_gate exit 1)"),
                                           ("idle", "boinc", "BOINC detected or the check failed (background_compute "
                                                             "exit 1)"),
                                           ("idle", "failed", "BOINC detected or the check failed (background_compute "
                                                              "exit 3)")], ids=["busy", "boinc", "check-fails"])
def test_FAKE_LAUNCH_A_BUSY_MACHINE_BOINC_OR_A_FAILED_CHECK_PREVENTS_THE_LAUNCH(tmp_path, idle, bg, why):
    repo, wdir, gate = _fake_repo(tmp_path)
    p = _launch(repo, wdir, idle, bg)
    assert p.returncode == 1 and f"{why}: nothing launched" in p.stderr, p.stderr
    assert "FAKE_LAUNCHED" not in p.stdout + p.stderr
    assert ("STUB_BACKGROUND_COMPUTE" in p.stderr) == (idle == "idle")      # busy is refused before the BOINC check
    assert sorted(x.name for x in wdir.iterdir()) == ["01_wrapper.sh"]      # nothing written, nothing started
    assert not (repo / "docs/superpowers/evidence/combination-policy-throughput").exists()
    assert f"{GATE_NAME} = False\n" in gate.read_text()                       # the gate restored anyway


def test_FAKE_LAUNCH_AN_IDLE_MACHINE_REACHES_THE_COMMAND_with_its_exact_arguments_and_records_the_group(tmp_path):
    repo, wdir, gate = _fake_repo(tmp_path)
    p = _launch(repo, wdir)
    assert p.returncode == 0, p.stderr
    launched = json.loads((wdir / "run.stdout.txt").read_text().split("FAKE_LAUNCHED", 1)[1])
    ckpt = str(repo.resolve() / "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors")
    assert launched["argv"] == ["run", "throughput", "--checkpoint", ckpt] and launched["tf32"] == "0"
    assert (wdir / "pgid.txt").read_text() == f"{launched['pgid']}\n"       # the supervisor wrote the run's group
    assert (wdir / "idle_check.txt").read_text() == "STUB_IDLE_GATE ['wait'] idle\n"
    assert (wdir / "bg_check.txt").read_text() == "STUB_BACKGROUND_COMPUTE ['check'] clear\n"
    outer = json.loads((wdir / "outer.json").read_text())
    assert outer["outcome"] == "COMPLETED" and outer["survivors"] == "" and outer["gate_restored"] is True
    assert outer["term_s"] == 890 and outer["grace_s"] == 10 and outer["record_by"] == "none"
    assert f"{GATE_NAME} = False\n" in gate.read_text()


def test_FAKE_LAUNCH_A_WORKER_LEFT_IN_THE_GROUP_IS_A_SURVIVOR_although_its_command_does_not_name_the_module(tmp_path):
    """The supervisor now waits for the whole group, so the worker can outlive it only if the supervisor is lost:
    the stub KILLs it (the wrapper reads 137 as TIMEOUT), and the wrapper's group search must still find the worker."""
    repo, wdir, _gate = _fake_repo(tmp_path)
    p = _launch(repo, wdir, run="linger")
    child = int((repo / "scratch/child.pid").read_text())
    try:
        outer = json.loads((wdir / "outer.json").read_text())
        assert p.returncode == 137 and _alive(child) and outer["record_by"] == "none"
        pgid = (wdir / "pgid.txt").read_text().strip()
        assert outer["survivors"].startswith(f"unconfirmed cleanup: process group {pgid} "), outer
        assert P.outer_faults(outer, {"head": outer["head"]})          # never admits generation
        assert str(child) in outer["survivors"].split(), outer      # strength2's name-only search would miss it
        assert "combination_policy_pilot" not in subprocess.run(["ps", "-o", "command=", "-p", str(child)],
                                                               capture_output=True, text=True).stdout
    finally:
        os.kill(child, signal.SIGKILL)


def test_FAKE_LAUNCH_THE_CEILING_TERMINATES_THE_WHOLE_GROUP_workers_included(tmp_path):
    """TERM_S changed to 2 in this copy only: the run and its 'worker' outlive it; the group TERM ends both."""
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
    rec = json.loads((repo / "docs/superpowers/evidence/combination-policy-throughput/record.json").read_text())
    assert outer["record_by"] == "wrapper" and rec["written_by"] == "wrapper" and rec["status"] == "VOID"
    assert rec["void"]["step"] == "timeout" and rec["kind"] == P.RECORD_KIND
    assert f"{GATE_NAME} = False\n" in gate.read_text()


def test_FAKE_LAUNCH_A_WORKER_THAT_IGNORES_TERM_IS_KILLED_within_the_ceiling_after_the_main_process_exits(tmp_path):
    """TERM_S changed to 2 in this copy only (ceiling 12 s): the main process dies of the group TERM, its worker
    ignores it, and the KILL 10 s later must still come (the user's review, 2026-10-29)."""
    repo, wdir, gate = _fake_repo(tmp_path, term_s=2)
    t = time.monotonic()
    p = _launch(repo, wdir, run="deaf")
    elapsed = time.monotonic() - t
    child = int((repo / "scratch/child.pid").read_text())
    try:
        assert not _alive(child), "the worker survived the ceiling"
        assert p.returncode == 137 and 12 <= elapsed < 30, (p.returncode, elapsed, p.stderr)
        outer = json.loads((wdir / "outer.json").read_text())
        assert outer["outcome"] == "TIMEOUT" and outer["inner_exit"] == 137 and outer["survivors"] == ""
        rec = json.loads((repo / "docs/superpowers/evidence/combination-policy-throughput/record.json").read_text())
        assert outer["record_by"] == "wrapper" and rec["status"] == "VOID" and rec["void"]["step"] == "timeout"
        assert f"{GATE_NAME} = False\n" in gate.read_text()
        assert P.outer_faults(outer, {"head": outer["head"]})          # never admits generation
    finally:
        if _alive(child):
            os.kill(child, signal.SIGKILL)


def test_FAKE_LAUNCH_A_GROUP_THAT_OUTLIVES_KILL_IS_REPORTED_NOT_WAITED_FOR_FOREVER(tmp_path):
    """No real process survives KILL here, so in this copy only (besides TERM_S 2) the supervisor's group probe also
    counts a sentinel process outside the group, which KILL cannot reach: to the supervisor, a group that never
    empties. It must stop watching GRACE_S s after KILL, exit 125, and the wrapper must record TIMEOUT with cleanup
    unconfirmed (the user's review, 2026-10-29)."""
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        repo, wdir, gate = _fake_repo(tmp_path, term_s=2, sentinel=sentinel.pid)
        t = time.monotonic()
        p = _launch(repo, wdir, run="outlive")
        elapsed = time.monotonic() - t
        assert p.returncode == 125 and 22 <= elapsed < 40, (p.returncode, elapsed, p.stderr)   # TERM 2 + 10 + 10 s
        outer = json.loads((wdir / "outer.json").read_text())
        assert outer["outcome"] == "TIMEOUT" and outer["inner_exit"] == 125
        pgid = (wdir / "pgid.txt").read_text().strip()
        assert outer["survivors"].startswith(f"unconfirmed cleanup: process group {pgid}"), outer
        assert outer["record_by"] == "none"                           # no wrapper record while cleanup is unconfirmed
        assert not (repo / "docs/superpowers/evidence/combination-policy-throughput/record.json").exists()
        assert not _alive(int((repo / "scratch/child.pid").read_text()))      # the real group itself was KILLed
        assert f"{GATE_NAME} = False\n" in gate.read_text()
        assert P.outer_faults(outer, {"head": outer["head"]})          # never admits generation
    finally:
        sentinel.kill()
        sentinel.wait()


def test_FAKE_LAUNCH_A_MISSING_PGID_RECORDS_SURVIVORS_AS_UNKNOWN(tmp_path):
    repo, wdir, _gate = _fake_repo(tmp_path)
    p = _launch(repo, wdir, run="lose_pgid")
    outer = json.loads((wdir / "outer.json").read_text())
    assert p.returncode == 0 and outer["survivors"].startswith("unknown: no readable pgid.txt")
    assert P.outer_faults(outer, {"head": outer["head"]})              # never admits generation


def _commit_evidence(repo, monkeypatch):
    """Commit the fake run's outputs and point the REAL admission (the real committed readers) at the fake repo."""
    def git(*a):
        return subprocess.check_output(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo,
                                       text=True).strip()
    monkeypatch.setattr(P, "ROOT", repo)
    monkeypatch.setattr(P.M, "_git", git)
    git("add", "docs/superpowers/evidence")
    git("commit", "-qm", "the run's evidence")


def test_ATTEMPT_1_S_PATHS_NEVER_ADMIT_GENERATION_even_after_a_completed_committed_run(tmp_path, monkeypatch):
    """RE-ANCHORED 2026-10-29 (the attempt-2 amendment): generation is admitted from attempt 2's records only, so a
    COMPLETED fake run on attempt 1's wrapper, its records committed, admits nothing. Attempt 2's accept and
    TIMEOUT-over-COMPLETED checks live in tests/test_combination_policy_throughput2_launch.py."""
    repo, wdir, _gate = _fake_repo(tmp_path)
    p = _launch(repo, wdir, run="record")
    outer = json.loads((wdir / "outer.json").read_text())
    assert p.returncode == 0 and outer["outcome"] == "COMPLETED" and outer["record_by"] == "script", outer
    _commit_evidence(repo, monkeypatch)
    assert not P.outer_faults(outer, json.loads((repo / P.KINDS["throughput"]["evidence"] / "record.json").read_text()))
    with pytest.raises(P.Refused, match="combination-policy-throughput2/record.json does not exist"):
        P.admitted("generate", P._read_record, P._read_outer)


# ─────────────────────────────── the seed-procedure copies (built, never run here) ───────────────────────────────

def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, EV / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                     # module level only: no mode runs, nothing is written
    return mod


def _below_docstring(rel):
    text = (EV / rel).read_text()
    return text[text.index('"""', 3) + 3:]


def test_THE_SURVEY_COPY_IS_STRENGTH2_S_WITH_ONLY_THE_STATED_DIFFERENCES():
    """The block size, the paths, the plan's kind, the module names, the titles and verdict lines, the slot lines."""
    two = _below_docstring("mirror-halving-strength2-seed-survey/01_seed_survey.py").splitlines()
    tp = _below_docstring(SURVEY).splitlines()
    assert len(tp) == len(two)
    changed = [(a, b) for a, b in zip(two, tp) if a != b]
    assert {b.strip() for _a, b in changed} == {
        "N_SLOTS = 144", 'HERE = "docs/superpowers/evidence/combination-policy-throughput-seed-survey"',
        'PLAN = "docs/superpowers/evidence/combination-policy-throughput-run/plan.json"',
        'VERDICT_OK = "THE COMBINATION-POLICY THROUGHPUT BLOCK IS CLEAN"',
        'spec = importlib.util.spec_from_file_location(f"_combination_policy_throughput_{key}", p)',
        'return {"kind": "combination-policy-throughput-plan/1", "base": base, "n_slots": N_SLOTS,',
        'return [f"    SLOT {g} seed {block[0] + g}"',
        'faults.append("the throughput plan is absent or not committed")',
        'faults.append("the throughput plan is not build_plan of the surveyed block")',
        'say(f"COMBINATION POLICY PILOT -- THROUGHPUT GAME-SEED {\'SURVEY\' if mode == \'survey\' else '
        '\'COLLISION PROOF, mode \' + mode.upper()}")',
        'say("SLOTS (combination_selfplay.generate: slot g plays seed BASE + g, whichever worker claims it):")',
        "say(f\"PROOF VERDICT ({mode}): {'THE COMBINATION-POLICY THROUGHPUT BLOCK IS CLEAN AND THE STATE IS AS "
        "EXPECTED' if ok else 'NOT PROVEN -- STOP'}\""}, changed


def _functions(rel):
    src = (EV / rel).read_text()
    return {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}


def test_THE_CLOSING_COPY_CHANGES_ONLY_THE_RULE_AND_ITS_CONTROLS():
    """Card §5's one diff lives in `started_k`, `reconcile` and `_controls`; `_git` is identical and `main` differs
    only in the title and the logged slot field (plus the claims line)."""
    two, tp = _functions("mirror-halving-strength2-seed-closing/01_derive_usage.py"), _functions(CLOSING)
    assert set(tp) == set(two) | {"started_k"} and tp["_git"] == two["_git"]
    claims = "f\"claims events: {[e.get('claims') for e in events if e.get('event') == 'claims']}\"]"
    a, b = two["main"].splitlines(), tp["main"].splitlines()
    assert [y.strip() for y in b].count(claims) == 1
    b = [y for y in b if y.strip() != claims]                  # the one added line
    assert len(b) == len(a)
    changed = [(x, y) for x, y in zip(a, b) if x != y]
    assert [y.strip() for _x, y in changed] == [
        'lines = ["COMBINATION POLICY PILOT -- THROUGHPUT SEED-USAGE DERIVATION (closing)", "=" * 96,',
        "f\"game starts (slots, as logged): {[e.get('slot') for e in events if e.get('event') == 'game_start']}\","]


def test_THE_SEED_COPIES_MATCH_THE_RUN_THEY_SERVE():
    survey, closing = _load(SURVEY, "_tp_survey"), _load(CLOSING, "_tp_closing")
    kind = P.KINDS["throughput"]
    assert survey.N_SLOTS == closing.N_SLOTS == kind["slots"] == 144
    assert survey.PLAN == kind["plan"] == closing.INPUTS[0]
    assert closing.RUN == kind["evidence"] and closing.RUN_DIR == str(pathlib.Path(survey.PLAN).parent)
    assert survey.build_plan(202700000, "0" * 64) == {"kind": "combination-policy-throughput-plan/1",
                                                      "base": 202700000, "n_slots": 144,
                                                      "survey_record_sha256": "0" * 64}
    assert survey.slot_lines((202700000, 202700144))[-1] == "    SLOT 143 seed 202700143"
    assert survey.PINNED == _load("mirror-halving-strength2-seed-survey/01_seed_survey.py", "_s2_survey").PINNED


def test_THE_SEED_COPIES_CODE_HOLDS_NO_SEED_LITERAL():
    """Below the docstring. The survey's copied description quotes attempt 1's retired block by value, as
    strength2's does; the scanner exempts the file as its own record."""
    for rel in (SURVEY, CLOSING):
        assert not re.search(r"\b20[0-9]{7}\b", _below_docstring(rel)), rel


def test_THE_SURVEY_COPY_S_STATE_CONTROLS_ALL_PASS():
    """The pinned method loaded read-only; the copy's own synthetic pre/post registry and plan cases."""
    survey = _load(SURVEY, "_tp_survey_state")
    S, Pf, _SEL = survey.load_method()
    assert S.N == 144 and S.PLAN == survey.PLAN
    rec = {"head": "0" * 40, "registry_sha256": "0" * 64, "base": 202700000, "block": (202700000, 202700144)}
    results = survey._state_controls(S, Pf, rec, "a" * 64)
    assert len(results) == 12 and all(ok for _l, ok, _d in results), [r for r in results if not r[1]]


def test_THE_CLOSING_COPY_S_CONTROLS_ALL_PASS():
    results = _load(CLOSING, "_tp_closing_c")._controls(202700000)
    assert len(results) == 21 and all(ok for _l, ok, _d in results), [r for r in results if not r[1]]


# ─────────────────────────────── the closing reads what the real entry writes ───────────────────────────────

class ShuffledHarness(GenHarness):
    """Starts reported out of order, as 18 workers report them."""
    def generate(self, *, evaluator, n_slots, slot_base, n_workers, max_batch_rows, on_start, on_game):
        order = sorted(range(n_slots), key=lambda s: (s * 5) % n_slots)
        for g in order:
            on_start(g, g % 3)
        for g in order:
            game = {"seed": slot_base + g + self.seed_shift, "moves": [[1, g % 20]] * 22, "winner": "red",
                    "reason": "win", "n_moves": 22, "n_rows": 2, "row_plies": [20, 21]}
            on_game(g, g % 3, game, _fake_rows([20, 21]), 1.5)
        return {"claims": n_slots, "started": list(range(n_slots)), "ended": list(range(n_slots)),
                "hellos": {0: {"setting": "0", "mlx_loaded": False}}, "workers": 18,
                "server": {"requests": 10, "rows": 100, "batches": 10}, "max_batch_rows": 14}


def _real_events(tmp_path, monkeypatch, harness):
    monkeypatch.setitem(P.KINDS["throughput"], "slots", 6)
    out = tmp_path / "ev"
    code, rec = _REAL_CONCLUDE("throughput", lambda o, r, lg: P._generate("throughput", harness.seams, o, r, lg),
                               out, {})
    return code, rec, [json.loads(x) for x in (out / "progress.jsonl").read_text().splitlines()]


def _closing_for(n):
    closing = _load(CLOSING, f"_tp_closing_{n}")
    closing.N_SLOTS = n                              # the rule at the harness's size; the copy's own 144 is checked above
    return closing


def test_THE_CLOSING_RECONCILES_A_REAL_COMPLETED_RUN_S_EVENTS_out_of_order(tmp_path, monkeypatch):
    code, rec, events = _real_events(tmp_path, monkeypatch, ShuffledHarness(tmp_path, kind="throughput", n=6))
    assert code == P.EXIT_COMPLETED, rec.get("void")
    starts = [e["slot"] for e in events if e["event"] == "game_start"]
    assert starts != sorted(starts) and sorted(starts) == list(range(6))
    closing = _closing_for(6)
    regs = {"ACCOUNTED": [(5000, 5006)], "EXPOSED": [], "RETIRED": []}
    faults, expose, retire = closing.reconcile({"base": 5000, "n_slots": 6}, events, {"outer_exit": 0, "survivors": ""},
                                               rec["status"], regs)
    assert faults == [] and expose == retire == (5000, 5006)
    assert closing.started_k(events) == 6


def test_A_REAL_RUN_VOID_MID_GENERATION_HAS_NO_CLAIMS_so_the_closing_exposes_the_whole_block(tmp_path, monkeypatch):
    bg = (lambda i: {"ok": i < 4, "n_procs": 5, "hits": ["1 boinc"] if i >= 4 else [], "exempt": [], "error": None})
    code, rec, events = _real_events(tmp_path, monkeypatch, GenHarness(tmp_path, kind="throughput", n=6, bg=bg))
    assert code == P.EXIT_VOID and rec["void"]["step"] == "background_compute"
    assert "claims" not in [e["event"] for e in events] and any(e["event"] == "game_start" for e in events)
    closing = _closing_for(6)
    regs = {"ACCOUNTED": [(5000, 5006)], "EXPOSED": [], "RETIRED": []}
    faults, expose, retire = closing.reconcile({"base": 5000, "n_slots": 6}, events, {"outer_exit": 3, "survivors": ""},
                                               rec["status"], regs)
    assert faults == [] and closing.started_k(events) is None and expose == retire == (5000, 5006)
