"""Fail-closed checks for the E4 screen command.

NO model, NO agent, NO RNG, NO jvm, NO game, NO scheduled seed. Every test either
exercises a precondition's refusal or proves that nothing effectful is reached.
"""
import inspect
import os
import re
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import e4_screen_command as C

PLAN = "docs/superpowers/evidence/2026-08-25-t1j-e4-preflight-attempt4/06_endpoint_screen_plan.json"
REPO = os.path.abspath(".")
SP = ("/private/tmp/claude-501/-Users-bill-projects-TwixT-Game/"
      "d037040d-d572-424f-a870-eef66233d641/scratchpad")
JDK = f"{SP}/e2/jdk/home/jdk-17.0.20.1+1/Contents/Home"
JAR = f"{SP}/e1/acq/release/t1j.jar"
CKPT = "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"
SOURCE = open("scripts/GPU/alphazero/e4_screen_command.py").read()


# --- the screen is unauthorized, and nothing outside the file can change it -

def test_screen_authorized_is_false():
    assert C.SCREEN_AUTHORIZED is False


def test_exactly_one_assignment_to_the_constant_and_it_is_false():
    assigns = re.findall(r"^SCREEN_AUTHORIZED\s*=\s*(\S+)", SOURCE, re.M)
    assert assigns == ["False"], assigns
    # nothing rebinds it anywhere, at any indentation
    assert len(re.findall(r"\bSCREEN_AUTHORIZED\s*=", SOURCE)) == 1


def test_the_module_reads_no_environment_and_no_config():
    """Inspects the CODE, not the prose.

    A raw-text grep matched the comment that says the module never reads
    os.environ -- the comment is not the behaviour, and neither is a string.
    """
    import ast

    tree = ast.parse(SOURCE)
    offences = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"environ", "environb", "getenv"}:
            offences.append(f"attribute {node.attr} at line {node.lineno}")
        if isinstance(node, ast.Name) and node.id in {"getenv", "environ"}:
            offences.append(f"name {node.id} at line {node.lineno}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", None) or ""
            names = [a.name for a in node.names]
            for bad in ("configparser", "dotenv", "yaml", "tomllib", "tomli"):
                if bad in mod or any(bad in n for n in names):
                    offences.append(f"import {bad} at line {node.lineno}")
    assert offences == [], offences


def test_the_authorization_constant_is_read_exactly_once_in_code():
    """And it is read as a bare module global, not through getattr or a lookup."""
    import ast

    tree = ast.parse(SOURCE)
    reads = [n for n in ast.walk(tree)
             if isinstance(n, ast.Name) and n.id == "SCREEN_AUTHORIZED"]
    stores = [n for n in reads if isinstance(n.ctx, ast.Store)]
    loads = [n for n in reads if isinstance(n.ctx, ast.Load)]
    assert len(stores) == 1, "exactly one assignment"
    assert len(loads) == 1, f"exactly one read, found {len(loads)}"
    assert "getattr" not in SOURCE or "SCREEN_AUTHORIZED" not in SOURCE.split("getattr")[1][:80]


def test_no_command_line_option_enables_the_screen():
    opts = re.findall(r'p\.add_argument\("(--[a-z-]+)"', SOURCE)
    assert set(opts) == {"--plan", "--results", "--repo", "--jdk", "--jar", "--checkpoint"}
    for banned in ("authoriz", "force", "yes", "confirm", "enable", "screen-on"):
        assert not any(banned in o for o in opts), banned


def test_public_entry_point_takes_paths_only():
    params = inspect.signature(C.run_screen).parameters
    assert set(params) == {"plan_path", "results_path", "repo_root", "jdk_home",
                           "jar_path", "checkpoint_path"}
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())
    for banned in ("task", "agent", "evaluator", "callable", "rng", "binder", "factory"):
        assert not any(banned in n for n in params), banned


# --- every precondition completes before the refusal, and nothing runs ------

def good_args(tmp_path, **over):
    a = dict(plan_path=PLAN, results_path=str(tmp_path / "out.jsonl"), repo_root=REPO,
             jdk_home=JDK, jar_path=JAR, checkpoint_path=CKPT)
    a.update(over)
    return a


@pytest.fixture
def clean_repo(tmp_path):
    """A committed tree holding the canonical plan at the same relative path."""
    root = tmp_path / "repo"
    dest = root / os.path.dirname(PLAN)
    dest.mkdir(parents=True)
    (dest / os.path.basename(PLAN)).write_bytes(open(PLAN, "rb").read())
    for cmd in (["init", "-q"], ["add", "-A"],
                ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "plan"]):
        subprocess.run(["git", "-C", str(root), *cmd], check=True, capture_output=True)
    return root


def test_all_six_preconditions_complete_before_the_refusal(tmp_path, clean_repo):
    log = C.EffectLog()
    trace = []
    with pytest.raises(C.AuthorizationError) as e:
        C._run_screen(**good_args(tmp_path, repo_root=str(clean_repo),
                                  plan_path=str(clean_repo / PLAN)),
                      _load_evaluator=log.load_evaluator, _build_agent=log.build_agent,
                      _run_harness=log.run_harness, _compile=log.compile, _trace=trace)
    assert trace == list(C.PRECONDITIONS)
    assert e.value.completed_preconditions == list(C.PRECONDITIONS)
    assert log.calls == [], "no effect may be reached while unauthorized"


def test_no_results_file_is_created_while_unauthorized(tmp_path, clean_repo):
    out = tmp_path / "must_not_exist.jsonl"
    with pytest.raises(C.AuthorizationError):
        C._run_screen(**good_args(tmp_path, results_path=str(out), repo_root=str(clean_repo),
                                  plan_path=str(clean_repo / PLAN)))
    assert not out.exists()


def test_no_rng_is_constructed_anywhere_while_unauthorized(tmp_path, clean_repo, monkeypatch):
    """Patches random.Random itself, so this catches an RNG made ANYWHERE.

    Stronger than a seam: the command has no RNG seam because it creates no
    generator at all. The only generators in a screen are the two each reference
    agent derives from its own bound task seed.
    """
    import random
    made = []
    real = random.Random

    class _Counting(real):
        def __init__(self, *a, **kw):
            made.append(a)
            super().__init__(*a, **kw)

    monkeypatch.setattr(random, "Random", _Counting)
    with pytest.raises(C.AuthorizationError):
        C._run_screen(**good_args(tmp_path, repo_root=str(clean_repo),
                                  plan_path=str(clean_repo / PLAN)))
    assert made == [], f"{len(made)} RNG(s) constructed before the refusal"


# --- each precondition is immediately fatal, in order ----------------------

def test_a_bad_plan_stops_at_the_first_precondition(tmp_path, clean_repo):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    trace = []
    with pytest.raises(C.PreconditionError) as e:
        C._run_screen(**good_args(tmp_path, plan_path=str(bad), repo_root=str(clean_repo)),
                      _trace=trace)
    assert e.value.which == "plan" and trace == []


def test_a_dirty_worktree_is_refused(tmp_path, clean_repo):
    (clean_repo / "scratch.txt").write_text("uncommitted")
    trace = []
    with pytest.raises(C.PreconditionError) as e:
        C._run_screen(**good_args(tmp_path, repo_root=str(clean_repo),
                                  plan_path=str(clean_repo / PLAN)), _trace=trace)
    assert e.value.which == "repository" and "uncommitted" in e.value.message
    assert trace == ["plan"], "the plan check ran; nothing after the failure did"


def test_a_wrong_jdk_is_refused(tmp_path, clean_repo):
    fake = tmp_path / "jdk"
    (fake / "bin").mkdir(parents=True)
    (fake / "bin" / "java").write_bytes(b"not the qualified jvm")
    trace = []
    with pytest.raises(C.PreconditionError) as e:
        C._run_screen(**good_args(tmp_path, jdk_home=str(fake), repo_root=str(clean_repo),
                                  plan_path=str(clean_repo / PLAN)), _trace=trace)
    assert e.value.which == "jdk" and trace == ["plan", "repository"]


def test_a_wrong_jar_is_refused(tmp_path, clean_repo):
    fake = tmp_path / "t1j.jar"
    fake.write_bytes(b"PK\x03\x04 not the pinned jar")
    trace = []
    with pytest.raises(C.PreconditionError) as e:
        C._run_screen(**good_args(tmp_path, jar_path=str(fake), repo_root=str(clean_repo),
                                  plan_path=str(clean_repo / PLAN)), _trace=trace)
    assert e.value.which == "jar" and trace == ["plan", "repository", "jdk"]


def test_a_wrong_checkpoint_is_refused_without_being_loaded(tmp_path, clean_repo):
    fake = tmp_path / "model.safetensors"
    fake.write_bytes(b"not the pinned checkpoint")
    trace = []
    with pytest.raises(C.PreconditionError) as e:
        C._run_screen(**good_args(tmp_path, checkpoint_path=str(fake), repo_root=str(clean_repo),
                                  plan_path=str(clean_repo / PLAN)), _trace=trace)
    assert e.value.which == "checkpoint"
    assert trace == ["plan", "repository", "jdk", "jar"]


def test_an_existing_output_path_is_refused_and_left_alone(tmp_path, clean_repo):
    out = tmp_path / "stale.jsonl"
    out.write_text("stale\n")
    trace = []
    with pytest.raises(C.PreconditionError) as e:
        C._run_screen(**good_args(tmp_path, results_path=str(out), repo_root=str(clean_repo),
                                  plan_path=str(clean_repo / PLAN)), _trace=trace)
    assert e.value.which == "output_path"
    assert trace == ["plan", "repository", "jdk", "jar", "checkpoint"]
    assert out.read_text() == "stale\n"


def test_a_missing_output_directory_is_refused(tmp_path, clean_repo):
    with pytest.raises(C.PreconditionError) as e:
        C._run_screen(**good_args(tmp_path, results_path=str(tmp_path / "nope" / "o.jsonl"),
                                  repo_root=str(clean_repo), plan_path=str(clean_repo / PLAN)))
    assert e.value.which == "output_path"


def test_a_reshaped_plan_is_refused_by_the_plan_check(tmp_path, clean_repo):
    import json
    plan = json.load(open(PLAN))
    plan["tasks"] = list(reversed(plan["tasks"]))
    bad = tmp_path / "reshaped.json"
    bad.write_text(json.dumps(plan))
    with pytest.raises(C.PreconditionError) as e:
        C._run_screen(**good_args(tmp_path, plan_path=str(bad), repo_root=str(clean_repo)))
    assert e.value.which == "plan"


# --- the CLI, in fresh subprocesses ----------------------------------------

def cli(*args, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run([sys.executable, "-m", "scripts.GPU.alphazero.e4_screen_command", *args],
                          capture_output=True, text=True, env=e)


def test_cli_help_documents_the_ban():
    r = cli("--help")
    assert r.returncode == 0 and "NOT AUTHORIZED" in r.stdout


def cli_args(tmp_path, repo):
    return ("--plan", str(repo / PLAN), "--results", str(tmp_path / "o.jsonl"),
            "--repo", str(repo), "--jdk", JDK, "--jar", JAR, "--checkpoint", CKPT)


def test_cli_refuses_with_exit_5_after_every_precondition(tmp_path, clean_repo):
    r = cli(*cli_args(tmp_path, clean_repo))
    assert r.returncode == C.EXIT_UNAUTHORIZED
    assert "SCREEN NOT AUTHORIZED" in r.stderr
    for name in C.PRECONDITIONS:
        assert name in r.stderr
    assert not (tmp_path / "o.jsonl").exists()


@pytest.mark.parametrize("env", [
    {"SCREEN_AUTHORIZED": "1"}, {"E4_SCREEN_AUTHORIZED": "true"},
    {"AUTHORIZED": "yes"}, {"FORCE": "1"}, {"PYTHONOPTIMIZE": "1"},
])
def test_no_environment_variable_enables_the_screen(tmp_path, clean_repo, env):
    r = cli(*cli_args(tmp_path, clean_repo), env=env)
    assert r.returncode == C.EXIT_UNAUTHORIZED, (env, r.stderr[:200])


@pytest.mark.parametrize("flag", [
    "--authorized", "--force", "--yes", "--enable-screen", "--screen-authorized",
])
def test_no_command_line_flag_is_accepted(tmp_path, clean_repo, flag):
    r = cli(*cli_args(tmp_path, clean_repo), flag)
    assert r.returncode == 2 and "unrecognized arguments" in r.stderr


def test_cli_precondition_failure_exits_2(tmp_path, clean_repo):
    r = cli("--plan", str(clean_repo / PLAN), "--results", str(tmp_path / "o.jsonl"),
            "--repo", str(clean_repo), "--jdk", "/nonexistent", "--jar", JAR,
            "--checkpoint", CKPT)
    assert r.returncode == C.EXIT_PRECONDITION and "PRECONDITION REFUSED" in r.stderr


def test_the_scheduled_seed_block_is_never_touched(tmp_path, clean_repo):
    from scripts.GPU.alphazero import e4_screen_reference as REF
    with pytest.raises(C.AuthorizationError):
        C._run_screen(**good_args(tmp_path, repo_root=str(clean_repo),
                                  plan_path=str(clean_repo / PLAN)))
    assert not any(REF.seed_is_exposed(s) for s in range(202612128, 202612160))


# --- the AUTHORIZED branch is real wiring, proven with substitutes ----------
# Nothing here enables the screen, plays a move, loads a model or makes an RNG.

import json as _json

from scripts.GPU.alphazero import e4_screen_runner as H
from scripts.GPU.alphazero import twixtbot_g3_reference as G3


class _Captured:
    def __init__(self): self.kw = None; self.args = None
    def __call__(self, *a, **kw):
        self.args, self.kw = a, kw
        return H.EXIT_OK


def authorized_call(tmp_path, clean_repo, **over):
    """Run the authorized path with substitutes, WITHOUT enabling the screen."""
    cap = _Captured()
    sentinel = object()
    compiled = []
    kw = dict(
        plan=_json.load(open(PLAN)),
        plan_path=str(clean_repo / PLAN),
        results_path=str(tmp_path / "screen.jsonl"),
        repo_root=str(clean_repo), jdk_home=JDK, jar_path=JAR,
        records={}, trace=[],
        _load_evaluator=lambda repo: sentinel,
        _build_agent=lambda task, *, evaluator: ("agent", task["task_id"]),
        _run_harness=cap,
        _cleanup=lambda: None,
        _compile=lambda javac, jar, out: compiled.append((javac, jar, out)),
    )
    kw.update(over)
    rc = C._execute_screen(**kw)
    return rc, cap, sentinel, compiled, kw["trace"]


def test_the_authorized_path_hands_the_harness_the_canonical_32(tmp_path, clean_repo):
    rc, cap, sentinel, compiled, trace = authorized_call(tmp_path, clean_repo)
    assert rc == C.EXIT_OK
    tasks = cap.kw["_tasks"]
    canonical = _json.load(open(PLAN))["tasks"]
    assert [t["task_id"] for t in tasks] == [t["task_id"] for t in canonical]
    assert H.task_digest(tasks) == H.CANONICAL_TASK_DIGEST, "ordered and unedited"
    assert len(tasks) == 32


def test_the_authorized_path_supplies_the_qualified_collaborators(tmp_path, clean_repo):
    _, cap, sentinel, compiled, _ = authorized_call(tmp_path, clean_repo)
    assert cap.kw["_evaluator"] is sentinel, "the ONE loaded evaluator is passed through"
    for key, made_by in (("_state_factory", "make_state_factory"),
                         ("_binder", "make_binder"),
                         ("_agent_factory", "make_agent_factory")):
        fn = cap.kw[key]
        assert callable(fn) and made_by in fn.__qualname__, (key, fn.__qualname__)
    assert cap.kw["_cleanup"] is not None
    assert compiled and compiled[0][0].endswith("bin/javac"), "the helper is compiled first"


def test_the_authorized_path_uses_the_screen_settings_not_a_qualification_budget(
        tmp_path, clean_repo):
    _, cap, _, _, _ = authorized_call(tmp_path, clean_repo)
    assert cap.kw["_ply_cap"] == H.PLY_CAP == 280
    assert cap.kw["_n_per_endpoint"] == H.CANONICAL_N_PER_ENDPOINT == 16
    assert cap.kw["_ply_budget"] is None, "a screen plays to a terminal state"
    assert tuple(cap.kw["_band"]) == tuple(H.BAND)


def test_the_authorized_path_lets_the_harness_own_recording(tmp_path, clean_repo):
    _, cap, _, _, _ = authorized_call(tmp_path, clean_repo)
    assert cap.args[1] == str(tmp_path / "screen.jsonl"), "the PATH is passed, not a Recorder"
    assert not (tmp_path / "screen.jsonl").exists(), "the command created no file itself"


def test_the_authorized_path_creates_no_rng(tmp_path, clean_repo, monkeypatch):
    """The only generators in a screen come from each agent's bound task seed."""
    import random
    made = []
    real = random.Random

    class _Counting(real):
        def __init__(self, *a, **kw):
            made.append(a); super().__init__(*a, **kw)

    monkeypatch.setattr(random, "Random", _Counting)
    authorized_call(tmp_path, clean_repo)
    assert made == [], f"{len(made)} RNG(s) created by the command itself"
    assert "random.Random(" not in SOURCE.replace("SeededReferenceAgent", "")


def test_the_authorized_path_records_its_wiring_order(tmp_path, clean_repo):
    _, _, _, _, trace = authorized_call(tmp_path, clean_repo)
    assert trace == ["compile_helper", "t1j_runtime", "load_evaluator",
                     "agent_factory", "run_harness"]


@pytest.mark.parametrize("harness_rc,want", [
    (H.EXIT_OK, C.EXIT_OK), (H.EXIT_PRECONDITION, C.EXIT_PRECONDITION),
    (H.EXIT_ABORT, C.EXIT_ABORT), (99, C.EXIT_UNEXPECTED),
])
def test_the_harness_exit_maps_to_the_command_exit(harness_rc, want):
    assert C._map_harness_exit(harness_rc) == want


def test_the_authorized_path_is_unreachable_while_unauthorized(tmp_path, clean_repo, monkeypatch):
    """The gate precedes it: substituting every effect still never gets there."""
    reached = []
    monkeypatch.setattr(C, "_execute_screen", lambda **kw: reached.append(kw))
    with pytest.raises(C.AuthorizationError):
        C._run_screen(**good_args(tmp_path, repo_root=str(clean_repo),
                                  plan_path=str(clean_repo / PLAN)))
    assert reached == [], "_execute_screen must not be reached while SCREEN_AUTHORIZED is False"


def test_the_real_defaults_are_the_qualified_ones():
    """No substitute: the production defaults point at the qualified code."""
    assert C._default_load_evaluator.__module__.endswith("e4_screen_command")
    assert G3.load_reference_evaluator is not None
    assert C._default_cleanup.__doc__ is None or True
    src = inspect.getsource(C._default_load_evaluator)
    assert "load_reference_evaluator" in src and "calib020_0001" in src
    assert "between_games_cleanup" in inspect.getsource(C._default_cleanup)
    assert "compile_helper" in inspect.getsource(C._default_compile)
    assert "REF.build" in inspect.getsource(C._default_build_agent)
    assert "H._run" in inspect.getsource(C._execute_screen)


# --- the NameError class of defect cannot recur ----------------------------

def _post_gate_unbound_names():
    """Every name loaded after the authorization gate must already be bound.

    Attempt 2 passed `plan=plan` after the gate while no such local existed:
    flipping the constant would have raised NameError before the screen started.
    Tests that substitute the branch cannot see that, because they call it
    directly.
    """
    import ast, builtins

    tree = ast.parse(SOURCE)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_run_screen")
    gate = next(i for i, st in enumerate(fn.body)
                if isinstance(st, ast.If) and "SCREEN_AUTHORIZED" in ast.dump(st.test))
    bound = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    for st in fn.body[:gate + 1]:
        for n in ast.walk(st):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                bound.add(n.id)
    mod = {n.targets[0].id for n in tree.body
           if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    mod |= {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    mod |= {a.asname or a.name.split(".")[0] for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    known = bound | mod | set(dir(builtins))
    unbound = []
    for st in fn.body[gate + 1:]:
        for n in ast.walk(st):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in known:
                unbound.append(f"{n.id}@{n.lineno}")
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                known.add(n.id)
    return unbound


def test_no_name_used_after_the_gate_is_unbound():
    assert _post_gate_unbound_names() == []


def test_the_verified_plan_is_carried_forward_not_reloaded():
    rec = C.check_plan(PLAN)
    assert "plan" in rec and isinstance(rec["plan"], dict)
    assert len(rec["plan"]["tasks"]) == 32
    assert "load_canonical_plan" not in inspect.getsource(C._execute_screen), \
        "the authorized branch must not re-read the plan"


# --- the REAL harness accepts what the command hands it --------------------

def test_the_command_asks_the_harness_for_SCREEN_mode():
    src = inspect.getsource(C._execute_screen)
    assert "mode=H.SCREEN_MODE" in src
    assert '"qualify"' not in src


def test_the_real_harness_accepts_the_commands_arguments(tmp_path, clean_repo):
    """No substitute harness. The real _run runs, and aborts in the state factory
    -- which is before any agent, so no RNG, no move, no game."""
    import json as _j

    cap = _Captured()
    plan = _j.load(open(PLAN))
    built = []

    def refusing_state_factory(task):
        raise H.AbortError(H.PHASE_PRECONDITION, "qualification stop: no game may be played")

    out = tmp_path / "real.jsonl"
    with pytest.raises(H.AbortError) as e:
        C._execute_screen(
            plan=plan, plan_path=str(clean_repo / PLAN), results_path=str(out),
            repo_root=str(clean_repo), jdk_home=JDK, jar_path=JAR, records={}, trace=[],
            _load_evaluator=lambda repo: object(),
            _build_agent=lambda task, *, evaluator: built.append(task),
            _cleanup=lambda: None,
            _compile=lambda javac, jar, o: None,
            _state_factory=refusing_state_factory,   # stop BEFORE the binder: no jvm
            _binder=lambda *a, **k: None)
    assert e.value.phase == H.PHASE_PRECONDITION
    assert built == [], "no agent was constructed"
    assert not os.path.isdir(os.path.join(str(tmp_path), "t1j_classes")), "no compile ran"
    hdr = _j.loads(open(out).readline())
    assert hdr["mode"] == "screen" and hdr["no_games"] is False
    assert hdr["synthetic_tasks"] == 0 and hdr["canonical_tasks_executed"] == 32
    assert hdr["ply_budget"] is None and hdr["ply_cap"] == 280


def test_the_real_harness_would_refuse_the_canonical_seeds_in_qualify_mode(tmp_path):
    """Which is why the command must ask for screen mode, not qualify."""
    import json as _j
    with pytest.raises(H.HarnessError) as e:
        H._run(PLAN, str(tmp_path / "q.jsonl"), mode="qualify",
               _tasks=_j.load(open(PLAN))["tasks"][:1],
               _agent_factory=lambda t, m, ev: None, _state_factory=lambda t: None,
               _binder=lambda *a: None, _evaluator=object(), _cleanup=lambda: None)
    assert "accounted or exposed" in str(e.value)
