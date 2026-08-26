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
                      _make_rng=log.make_rng, _trace=trace)
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
    """Patches random.Random itself, so this catches an RNG made ANYWHERE."""
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
