"""The mirror-averaging + root-halving screen and strength LAUNCH: the idle gate
(`scripts/GPU/alphazero/idle_gate.py`), the two wrappers (`docs/superpowers/evidence/mirror-halving-{screen,strength}-
run/01_wrapper.sh`) and the seed-procedure copies. CPU only; nothing here loads a model, touches a device, runs a seed
procedure or opens a real gate.

THE FAKE-LAUNCH CHECKS run each REAL wrapper's bytes in a throwaway git repository that holds only what the wrapper
needs: its gate file (opened by a one-line commit), a stub `idle_gate` that reports busy or idle as the test says, and
a stub `mirror_halving_match` that only prints what it was launched with. A busy machine must prevent the launch; an
idle one may reach the stub (ledger #93: a printed check is not a check)."""
import importlib.util
import json
import math
import os
import pathlib
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import idle_gate as IG
from scripts.GPU.alphazero import mirror_halving_match as MATCH

ROOT = pathlib.Path(__file__).resolve().parents[1]
EV = ROOT / "docs/superpowers/evidence"
KINDS = {"screen": {"term": 3486, "slots": 64}, "strength": {"term": 55591, "slots": 1018},
         "strength2": {"term": 55591, "slots": 1018}}          # attempt 2: the strength comparison's ceiling


# ─────────────────────────────── the idle gate ───────────────────────────────

class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _wait(samples):
    """wait() over a scripted sequence of samples (the last one repeats), with a fake clock."""
    clock, said, it = Clock(), [], iter(samples)
    last = [samples[-1]]

    def sample():
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]

    return IG.wait(sample, clock, clock.sleep, said.append), said, clock.t


IDLE, LOADED, HOG = (1.0, [(17.0, "claude")]), (4.0, []), (1.0, [(313.5, "mds_stores")])


def test_AN_IDLE_MACHINE_PASSES_ONLY_AFTER_THE_FULL_HOLD():
    code, said, t = _wait([IDLE])
    assert code == IG.EXIT_IDLE and t == IG.HOLD_S and said[-1].startswith("IDLE")


@pytest.mark.parametrize("busy", [LOADED, HOG])
def test_A_BUSY_MACHINE_NEVER_PASSES_and_the_gate_gives_up_at_the_maximum_wait(busy):
    code, said, t = _wait([busy])
    assert code == IG.EXIT_BUSY and t == IG.MAX_WAIT_S and said[-1].startswith("BUSY")
    assert all("BUSY" in line for line in said)


def test_C1_S_OWN_LAUNCH_CONDITIONS_WOULD_HAVE_BEEN_REFUSED():
    """Entry 39: load 4.00 with Spotlight's mds_stores at ~313% CPU."""
    assert len(IG.busy_reasons((4.0, [(313.5, "mds_stores")]))) == 2


def test_A_BUSY_SAMPLE_RESTARTS_THE_HOLD():
    n = int(IG.HOLD_S / IG.INTERVAL_S)
    code, _said, t = _wait([IDLE] * n + [HOG] + [IDLE])
    assert code == IG.EXIT_IDLE and t == (n + 1) * IG.INTERVAL_S + IG.HOLD_S


def test_THE_LIMITS_load_3_is_busy_and_50_percent_is_not():
    assert IG.busy_reasons((3.0, [])) and not IG.busy_reasons((2.99, [(50.0, "x")]))
    assert IG.busy_reasons((0.5, [(50.1, "x")]))
    assert (IG.LOAD_LIMIT, IG.CPU_LIMIT, IG.HOLD_S, IG.INTERVAL_S, IG.MAX_WAIT_S) == (3.0, 50.0, 60.0, 5.0, 1800.0)


def test_THE_REAL_SAMPLE_READS_THE_LOAD_AND_EXCLUDES_ITSELF():
    load, procs = IG.sample()
    assert load >= 0 and all(isinstance(c, float) for c, _ in procs)


def test_ANYTHING_BUT_wait_IS_USAGE():
    assert IG.main([]) == IG.main(["go"]) == IG.EXIT_USAGE


# ─────────────────────────────── the wrappers' bytes ───────────────────────────────

def _wrapper(kind):
    return (EV / f"mirror-halving-{kind}-run/01_wrapper.sh").read_text()


@pytest.mark.parametrize("kind", KINDS)
def test_THE_CEILING_IS_C1_S_PROJECTION_TIMES_1_5_ROUNDED_UP(kind):
    rec = json.loads((EV / "mirror-halving-c1/record.json").read_text())
    total = math.ceil(rec["derived"]["search_hours"][{"strength2": "strength"}.get(kind, kind)] * 3600 * 1.5)
    w = _wrapper(kind)
    assert f"\nTERM_S={KINDS[kind]['term']}\n" in w and "\nGRACE_S=10\n" in w
    assert KINDS[kind]["term"] + 10 == total == {"screen": 3496, "strength": 55601, "strength2": 55601}[kind]


@pytest.mark.parametrize("kind", ["screen", "strength"])
def test_THE_WRAPPER_IS_C1_S_WITH_ONLY_THE_STATED_DIFFERENCES(kind):
    c1 = (EV / "mirror-halving-c1-run/01_wrapper.sh").read_text().split("set -u\n", 1)[1].splitlines()
    w = _wrapper(kind).split("set -u\n", 1)[1].splitlines()
    removed, added = sorted(set(c1) - set(w)), sorted(set(w) - set(c1))
    k = kind.upper()
    assert set(added) == {
        f"OUT_REL=docs/superpowers/evidence/mirror-halving-{kind}-run",
        f"RUN_REL=docs/superpowers/evidence/mirror-halving-{kind}",
        f"GATE_REL=scripts/GPU/alphazero/mirror_halving_{kind}_authorization.py",
        f"GATE=MIRROR_HALVING_{k}_AUTHORIZED", "MODULE=scripts.GPU.alphazero.mirror_halving_match",
        f"TERM_S={KINDS[kind]['term']}", "RECORD_KIND=mirror-halving-match-record/1",
        "for f in idle_check.txt run.stdout.txt run.stderr.txt outer.json; do",
        "# THE IDLE GATE (last precondition): the launch below is reached ONLY if it exits 0.",
        'IDLE_LOG=$(env PYTHONPATH=. "$PY" -B -m scripts.GPU.alphazero.idle_gate wait 2>&1) || {',
        "  idle_rc=$?", "  printf '%s\\n' \"$IDLE_LOG\" >&2",
        '  refuse "the machine did not become idle (idle_gate exit $idle_rc): nothing launched"',
        "printf '%s\\n' \"$IDLE_LOG\" > \"$OUT_REL/idle_check.txt\"",
        f'/usr/bin/perl -e "$SUPERVISOR" "$TERM_S" "$GRACE_S" env MLX_ENABLE_TF32=0 PYTHONPATH=. "$PY" -B -m "$MODULE" '
        f'run {kind} --checkpoint "$CKPT" \\'}, added
    assert len(removed) == 9, removed                         # "}" closes the gate block; C1 has it already


@pytest.mark.parametrize("kind", KINDS)
def test_THE_IDLE_GATE_PRECEDES_EVERY_LAUNCH_LINE(kind):
    w = _wrapper(kind).split("set -u\n", 1)[1]                 # the code, not the header's prose
    gate, started, perl = (w.index(x) for x in ("idle_gate wait", "\nstarted=true", '/usr/bin/perl -e "$SUPERVISOR"'))
    assert gate < started < perl and w.count("/usr/bin/perl -e") == 1 and w.count("idle_gate wait") == 1


def test_ATTEMPT_2_S_WRAPPER_IS_ATTEMPT_1_S_WITH_ONLY_THE_STATED_DIFFERENCES():
    """Amendment 2026-10-27: the paths, the gate, `run strength2`, and the background-compute check after the idle
    gate, recorded in bg_check.txt. Everything else -- the ceiling, the supervisor, the restore -- byte-identical."""
    one = _wrapper("strength").split("set -u\n", 1)[1].splitlines()
    two = _wrapper("strength2").split("set -u\n", 1)[1].splitlines()
    removed, added = sorted(set(one) - set(two)), sorted(set(two) - set(one))
    assert set(added) == {
        "OUT_REL=docs/superpowers/evidence/mirror-halving-strength2-run",
        "RUN_REL=docs/superpowers/evidence/mirror-halving-strength2",
        "GATE_REL=scripts/GPU/alphazero/mirror_halving_strength2_authorization.py",
        "GATE=MIRROR_HALVING_STRENGTH2_AUTHORIZED",
        "for f in idle_check.txt bg_check.txt run.stdout.txt run.stderr.txt outer.json; do",
        "# THE BACKGROUND-COMPUTE CHECK (after the idle gate): the launch below is reached ONLY if it exits 0.",
        'BG_LOG=$(env PYTHONPATH=. "$PY" -B -m scripts.GPU.alphazero.background_compute check 2>&1) || {',
        "  bg_rc=$?", "  printf '%s\\n' \"$BG_LOG\" >&2",
        '  refuse "BOINC detected or the check failed (background_compute exit $bg_rc): nothing launched"',
        "printf '%s\\n' \"$BG_LOG\" > \"$OUT_REL/bg_check.txt\"",
        '/usr/bin/perl -e "$SUPERVISOR" "$TERM_S" "$GRACE_S" env MLX_ENABLE_TF32=0 PYTHONPATH=. "$PY" -B -m "$MODULE" '
        'run strength2 --checkpoint "$CKPT" \\'}, added
    assert len(removed) == 6, removed                       # the four settings, the one-shot list, the launch line
    assert len(two) == len(one) + 7                         # the six-line check and its record line


def test_ATTEMPT_2_S_BACKGROUND_CHECK_FOLLOWS_THE_IDLE_GATE_AND_PRECEDES_THE_LAUNCH():
    w = _wrapper("strength2").split("set -u\n", 1)[1]
    idle, bg, started, perl = (w.index(x) for x in ("idle_gate wait", "background_compute check", "\nstarted=true",
                                                    '/usr/bin/perl -e "$SUPERVISOR"'))
    assert idle < bg < started < perl and w.count("background_compute check") == 1


# ─────────────────────────────── fake launches of the REAL wrapper ───────────────────────────────

STUB_IDLE = """import os, sys
print("STUB_IDLE_GATE", sys.argv[1:], os.environ.get("FAKE_IDLE"))
sys.exit(0 if os.environ.get("FAKE_IDLE") == "idle" else 1)
"""
STUB_MATCH = """import json, os, sys
print("FAKE_LAUNCHED", json.dumps({"argv": sys.argv[1:], "tf32": os.environ.get("MLX_ENABLE_TF32")}))
"""
STUB_BG = """import os, sys
state = os.environ.get("FAKE_BG")
print("STUB_BACKGROUND_COMPUTE", sys.argv[1:], state)
sys.exit({"clear": 0, "boinc": 1}.get(state, 3))
"""


def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True,
                          capture_output=True, text=True).stdout


def _fake_repo(tmp, kind):
    """A throwaway repository holding the real wrapper, its gate (opened by a one-line commit) and two stubs."""
    repo = tmp / "repo"
    pkg = repo / "scripts/GPU/alphazero"
    pkg.mkdir(parents=True)
    for d in (repo / "scripts", repo / "scripts/GPU", pkg):
        (d / "__init__.py").write_text("")
    gate_name = f"MIRROR_HALVING_{kind.upper()}_AUTHORIZED"
    gate = pkg / f"mirror_halving_{kind}_authorization.py"
    gate.write_text(f'"""stub gate"""\n{gate_name} = False\n')
    (pkg / "idle_gate.py").write_text(STUB_IDLE)
    (pkg / "background_compute.py").write_text(STUB_BG)
    (pkg / "mirror_halving_match.py").write_text(STUB_MATCH)
    wdir = repo / f"docs/superpowers/evidence/mirror-halving-{kind}-run"
    wdir.mkdir(parents=True)
    (wdir / "01_wrapper.sh").write_bytes((EV / f"mirror-halving-{kind}-run/01_wrapper.sh").read_bytes())
    (repo / ".venv/bin").mkdir(parents=True)
    (repo / ".venv/bin/python").symlink_to(sys.executable)
    ckpt = repo / "checkpoints/alphazero-v2-calib020-from0409"
    ckpt.mkdir(parents=True)
    (ckpt / "model_iter_0001.safetensors").write_bytes(b"not a model")
    (repo / ".gitignore").write_text(".venv/\ncheckpoints/\n__pycache__/\n")
    _git(tmp, "init", "-q", str(repo))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    gate.write_text(gate.read_text().replace(f"{gate_name} = False", f"{gate_name} = True"))
    _git(repo, "commit", "-qam", "open the gate")
    return repo, wdir, gate, gate_name


def _launch(repo, wdir, idle, bg="clear"):
    env = {k: v for k, v in os.environ.items() if k not in ("MLX_ENABLE_TF32", "PYTHONPATH")}
    env.update(FAKE_IDLE=idle, FAKE_BG=bg, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(["bash", str(wdir / "01_wrapper.sh")], cwd=str(repo), env=env, capture_output=True,
                          text=True, timeout=60)


@pytest.mark.parametrize("kind", KINDS)
def test_FAKE_LAUNCH_A_BUSY_MACHINE_PREVENTS_THE_LAUNCH(tmp_path, kind):
    repo, wdir, gate, gate_name = _fake_repo(tmp_path, kind)
    p = _launch(repo, wdir, "busy")
    assert p.returncode == 1 and "did not become idle (idle_gate exit 1): nothing launched" in p.stderr, p.stderr
    assert "STUB_IDLE_GATE ['wait'] busy" in p.stderr and "FAKE_LAUNCHED" not in p.stdout + p.stderr
    assert sorted(x.name for x in wdir.iterdir()) == ["01_wrapper.sh"]          # nothing written, nothing started
    assert not (repo / f"docs/superpowers/evidence/mirror-halving-{kind}").exists()
    assert f"{gate_name} = False\n" in gate.read_text()                           # the gate restored anyway


@pytest.mark.parametrize("kind", KINDS)
def test_FAKE_LAUNCH_AN_IDLE_MACHINE_REACHES_THE_COMMAND_with_its_exact_arguments(tmp_path, kind):
    repo, wdir, gate, gate_name = _fake_repo(tmp_path, kind)
    p = _launch(repo, wdir, "idle")
    assert p.returncode == 0, p.stderr
    out = (wdir / "run.stdout.txt").read_text()
    launched = json.loads(out.split("FAKE_LAUNCHED", 1)[1])
    ckpt = str(repo.resolve() / "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors")
    assert launched == {"argv": ["run", kind, "--checkpoint", ckpt], "tf32": "0"}
    assert (wdir / "idle_check.txt").read_text() == "STUB_IDLE_GATE ['wait'] idle\n"
    outer = json.loads((wdir / "outer.json").read_text())
    assert outer["outcome"] == "COMPLETED" and outer["gate_restored"] is True and outer["term_s"] == KINDS[kind]["term"]
    assert f"{gate_name} = False\n" in gate.read_text()


@pytest.mark.parametrize("bg, why", [("boinc", "exit 1"), ("failed", "exit 3")], ids=["boinc-present", "check-fails"])
def test_FAKE_LAUNCH_ATTEMPT_2_BOINC_PRESENT_OR_A_FAILED_CHECK_PREVENTS_THE_LAUNCH(tmp_path, bg, why):
    repo, wdir, gate, gate_name = _fake_repo(tmp_path, "strength2")
    p = _launch(repo, wdir, "idle", bg)
    assert p.returncode == 1 and f"BOINC detected or the check failed (background_compute {why}): nothing launched" \
        in p.stderr, p.stderr
    assert f"STUB_BACKGROUND_COMPUTE ['check'] {bg}" in p.stderr and "FAKE_LAUNCHED" not in p.stdout + p.stderr
    assert sorted(x.name for x in wdir.iterdir()) == ["01_wrapper.sh"]          # nothing written, nothing started
    assert not (repo / "docs/superpowers/evidence/mirror-halving-strength2").exists()
    assert f"{gate_name} = False\n" in gate.read_text()                           # the gate restored anyway


def test_FAKE_LAUNCH_ATTEMPT_2_A_BUSY_MACHINE_IS_REFUSED_BEFORE_THE_BACKGROUND_CHECK(tmp_path):
    repo, wdir, _gate, _name = _fake_repo(tmp_path, "strength2")
    p = _launch(repo, wdir, "busy", "boinc")
    assert p.returncode == 1 and "did not become idle" in p.stderr and "STUB_BACKGROUND_COMPUTE" not in p.stderr


def test_FAKE_LAUNCH_ATTEMPT_2_BOINC_ABSENT_REACHES_THE_COMMAND_and_keeps_the_observation(tmp_path):
    repo, wdir, gate, gate_name = _fake_repo(tmp_path, "strength2")
    p = _launch(repo, wdir, "idle", "clear")
    assert p.returncode == 0, p.stderr
    launched = json.loads((wdir / "run.stdout.txt").read_text().split("FAKE_LAUNCHED", 1)[1])
    assert launched["argv"][:2] == ["run", "strength2"] and launched["tf32"] == "0"
    assert (wdir / "bg_check.txt").read_text() == "STUB_BACKGROUND_COMPUTE ['check'] clear\n"
    assert (wdir / "idle_check.txt").read_text() == "STUB_IDLE_GATE ['wait'] idle\n"
    assert json.loads((wdir / "outer.json").read_text())["gate_restored"] is True
    assert f"{gate_name} = False\n" in gate.read_text()


# ─────────────────────────────── the seed-procedure copies (built, never run here) ───────────────────────────────

def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, EV / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                     # module level only: no mode runs, nothing is written
    return mod


@pytest.mark.parametrize("kind", KINDS)
def test_THE_SEED_COPIES_MATCH_THE_RUN_THEY_SERVE(kind):
    n = KINDS[kind]["slots"]
    survey = _load(f"mirror-halving-{kind}-seed-survey/01_seed_survey.py", f"_survey_{kind}")
    closing = _load(f"mirror-halving-{kind}-seed-closing/01_derive_usage.py", f"_closing_{kind}")
    assert survey.N_SLOTS == closing.N_SLOTS == n == 2 * MATCH.KINDS[kind]["pairs"]
    assert survey.PLAN == MATCH.KINDS[kind]["plan"] and closing.INPUTS[0] == MATCH.KINDS[kind]["plan"]
    assert closing.RUN == MATCH.KINDS[kind]["evidence"] and closing.RUN_DIR == str(pathlib.Path(survey.PLAN).parent)
    plan = survey.build_plan(202700000, "0" * 64)
    assert plan == {"kind": f"mirror-halving-{kind}-plan/1", "base": 202700000, "n_slots": n,
                    "survey_record_sha256": "0" * 64}
    assert survey.slot_lines((202700000, 202700000 + n))[-1] == \
        f"    SLOT {n - 1} game {n - 1} pair {(n - 1) // 2} seed {202700000 + n - 1}"
    assert all(ok for _label, ok, _detail in closing._controls(202700000))


def test_THE_STRENGTH_PLAN_is_the_survey_s_build_plan_and_its_block_is_REGISTERED_and_CLOSED():
    """The strength comparison's plan (the one the public entry reads) is its survey's build_plan, bound to the
    committed survey record; the survey (2ccc9037), pre-proof (9631ec72) and post-proof (5c4b60a7) records all stand;
    the block is in ACCOUNTED exactly once (registered 2026-10-26, 0c1c18b5) and, before any run, NOT exposed, NOT
    retired and every seed schedulable by the entry's own seed predicates. Since the closing edit after the one
    (VOID, timed-out) run, the started prefix is EXPOSED, the whole block RETIRED, and every seed refused.
    Read-only; nothing is drawn."""
    import hashlib
    from scripts.GPU.alphazero import e4_screen_reference as REG
    kind = "strength"
    here = EV / f"mirror-halving-{kind}-seed-survey"
    n = KINDS[kind]["slots"]
    assert sorted(p.name for p in here.glob("0[234]_*")) == \
        ["02_seed_survey_run.txt", "03_collision_proof_pre_run.txt", "04_collision_proof_post_run.txt"]
    survey_raw = (here / "02_seed_survey_run.txt").read_bytes()
    survey, sha = survey_raw.decode("utf-8"), hashlib.sha256(survey_raw).hexdigest()
    assert "SURVEY VERDICT   : THE MIRROR-HALVING STRENGTH BLOCK IS CLEAN\n" in survey and "controls passed: 45/45" in survey
    for rel, mode in (("03_collision_proof_pre_run.txt", "pre"), ("04_collision_proof_post_run.txt", "post")):
        text = (here / rel).read_text(encoding="utf-8")
        assert f"02_seed_survey_run.txt sha256 {sha} (read at " in text
        assert f"EXPECTED STATE ({mode}): HOLDS" in text and "controls passed: 45/45" in text
    survey_mod = _load(f"mirror-halving-{kind}-seed-survey/01_seed_survey.py", "_survey_strength_registered")
    plan = json.loads((ROOT / MATCH.KINDS[kind]["plan"]).read_text(encoding="utf-8"))
    assert type(plan["base"]) is int and plan["n_slots"] == n
    assert plan == survey_mod.build_plan(plan["base"], sha)
    block = (plan["base"], plan["base"] + n)
    assert f"block = [{block[0]}, {block[1]}) -- {n} seeds" in survey
    assert REG.ACCOUNTED_SEED_INTERVALS.count(block) == 1
    # RE-ANCHORED 2026-10-27 (the strength closing, after K1 RECONCILES: TIMEOUT / VOID, 800 games started): the
    # started prefix is EXPOSED and the whole block RETIRED, each exactly once, and every seed is refused.
    started = (block[0], block[0] + 800)
    closing = (EV / f"mirror-halving-{kind}-seed-closing/02_derive_usage_run.txt").read_text(encoding="utf-8")
    assert f"VERDICT: RECONCILES -- EXPOSE {list(started)}; RETIRE {list(block)}\n" in closing
    assert REG.EXPOSED_SEED_INTERVALS.count(started) == 1 and REG.RETIRED_SEED_INTERVALS.count(block) == 1
    assert not any(lo < block[1] and block[0] < hi and (lo, hi) not in (started, block)
                   for lo, hi in REG.EXPOSED_SEED_INTERVALS + REG.RETIRED_SEED_INTERVALS + REG.TEST_ONLY_SEED_INTERVALS)
    assert all(REG.seed_is_accounted(s) and REG.seed_is_exposed(s) == (s < started[1]) and REG.seed_is_retired(s)
               and REG.seed_is_unavailable(s) and not REG.seed_is_test_only(s) for s in range(*block))


def test_THE_SCREEN_PLAN_is_the_survey_s_build_plan_and_its_block_is_REGISTERED():
    """The screen's plan (the one the public entry reads) is its survey's build_plan, bound to the committed survey
    record, and the block is in ACCOUNTED exactly once (registered 2026-10-26, aa977403; post-registration proof
    4dc418db). Since the closing edit after the one screen run, every seed is EXPOSED, RETIRED and refused.
    Read-only; nothing is drawn."""
    import hashlib
    from scripts.GPU.alphazero import e4_screen_reference as REG
    here = EV / "mirror-halving-screen-seed-survey"
    survey = _load("mirror-halving-screen-seed-survey/01_seed_survey.py", "_survey_screen_registered")
    record = (here / "02_seed_survey_run.txt").read_bytes()
    plan = json.loads((ROOT / MATCH.KINDS["screen"]["plan"]).read_text(encoding="utf-8"))
    assert type(plan["base"]) is int and plan["n_slots"] == KINDS["screen"]["slots"]
    assert plan == survey.build_plan(plan["base"], hashlib.sha256(record).hexdigest())
    block = (plan["base"], plan["base"] + plan["n_slots"])
    assert f"block = [{block[0]}, {block[1]})" in record.decode()
    for rel, mode in (("03_collision_proof_pre_run.txt", "pre"), ("04_collision_proof_post_run.txt", "post")):
        text = (here / rel).read_text(encoding="utf-8")
        assert f"EXPECTED STATE ({mode}): HOLDS" in text and "controls passed: 45/45" in text
    assert REG.ACCOUNTED_SEED_INTERVALS.count(block) == 1
    # RE-ANCHORED 2026-10-26 (the screen's closing, after K1 RECONCILES: all 64 games started): the whole block is
    # EXPOSED and RETIRED, each exactly once, and every seed is refused.
    assert REG.EXPOSED_SEED_INTERVALS.count(block) == 1 and REG.RETIRED_SEED_INTERVALS.count(block) == 1
    assert all(REG.seed_is_accounted(s) and REG.seed_is_exposed(s) and REG.seed_is_retired(s) and
               REG.seed_is_unavailable(s) and not REG.seed_is_test_only(s) for s in range(*block))
    assert not any(lo < block[1] and block[0] < hi and (lo, hi) != block
                   for lo, hi in REG.EXPOSED_SEED_INTERVALS + REG.RETIRED_SEED_INTERVALS)


def _code(rel):
    """The file below its module docstring."""
    return (EV / rel).read_text().split('"""', 2)[2]


@pytest.mark.parametrize("rel", ["seed-survey/01_seed_survey.py", "seed-closing/01_derive_usage.py"])
def test_ATTEMPT_2_S_SEED_COPIES_ARE_ATTEMPT_1_S_WITH_ONLY_THE_NAMES_CHANGED(rel):
    """Amendment 2026-10-27: the same method and rule. Below the docstring, attempt 2's code with its one name put back
    is attempt 1's, byte for byte: the paths, the plan's kind, the module names and the titles are all that differ."""
    one, two = _code(f"mirror-halving-strength-{rel}"), _code(f"mirror-halving-strength2-{rel}")
    assert two != one and two.replace("strength2", "strength").replace("STRENGTH2", "STRENGTH") == one
    assert "strength2" not in one and "STRENGTH2" not in one


def test_ATTEMPT_2_HAS_NO_SEEDS_YET():
    """Built only: no survey or proof record, no plan, no closing record, no run output, and its gate closed. Fresh seeds
    are chosen by the survey's rule only when that survey is separately approved and run."""
    from scripts.GPU.alphazero import mirror_halving_strength2_authorization as GATE2
    assert GATE2.MIRROR_HALVING_STRENGTH2_AUTHORIZED is False
    def names(d):
        return sorted(p.name for p in (EV / d).iterdir() if p.name != "__pycache__")
    assert names("mirror-halving-strength2-seed-survey") == ["01_seed_survey.py"]
    assert names("mirror-halving-strength2-seed-closing") == ["01_derive_usage.py"]
    assert names("mirror-halving-strength2-run") == ["01_wrapper.sh"]
    assert not (EV / "mirror-halving-strength2").exists() and not (ROOT / MATCH.KINDS["strength2"]["plan"]).exists()
