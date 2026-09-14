"""H3 PILOT — the runner: gated, seedless, contained.

Nothing here opens a gate, reserves a seed, loads a model, starts a JVM or plays
a game — and the point of the containment tests is that nothing here CAN.
"""
import ast
import inspect
import os
import sys

import pytest

from scripts.GPU.alphazero import e4_screen_reference as REF
from scripts.GPU.alphazero import h3_pilot_rules as R
from scripts.GPU.alphazero import h3_pilot_runner as RUN


# ═══════════════════════ closed, and seedless ═══════════════════════════════

def test_THE_GATE_IS_SHUT_IN_THE_REAL_REPOSITORY():
    assert RUN.H3_PILOT_EXECUTION_AUTHORIZED is False
    src = open(RUN.__file__, encoding="utf-8").read()
    assert src.count("H3_PILOT_EXECUTION_AUTHORIZED = False\n") == 1
    assert "H3_PILOT_EXECUTION_AUTHORIZED = True" not in src


def test_NO_SEED_BLOCK_IS_RESERVED_and_that_is_a_SECOND_barrier():
    """🔑 A gate opens with one reviewed edit. A seed block cannot be conjured by
    one, so the pilot is closed twice over."""
    assert RUN.PILOT_SEED_BLOCK is None
    with pytest.raises(RUN.H3PilotRunError, match="NO SEED BLOCK IS RESERVED"):
        RUN.check_seed_registration()


def test_the_public_entry_READS_THE_GATE_before_anything_else(tmp_path):
    with pytest.raises(RUN.H3PilotRunError, match="NOT AUTHORIZED"):
        RUN.run_pilot(results_path=str(tmp_path / "r.jsonl"),
                      trace_path=str(tmp_path / "t.jsonl"),
                      report_path=str(tmp_path / "rep.json"))


def test_run_pilot_CALLS_check_gate_FIRST_by_AST():
    """The behavioural test above asserts the refusal; this asserts the CALL IS
    THERE and is first, which is the property a control would delete."""
    fn = ast.parse(inspect.getsource(RUN.run_pilot).lstrip()).body[0]
    stmts = [n for n in fn.body
             if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    first = stmts[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
    assert getattr(first.value.func, "id", "") == "check_gate", ast.dump(first)


def test_the_entry_TAKES_ONLY_THE_OUTPUT_PATHS():
    """Nothing on the signature can reach the gate, the schedule or the seeds."""
    sig = inspect.signature(RUN.run_pilot)
    assert set(sig.parameters) == {"results_path", "trace_path", "report_path"}
    assert all(p.kind is p.KEYWORD_ONLY for p in sig.parameters.values())


def test_THERE_IS_NO_CLI_OVERRIDE_FOR_THE_GATE_OR_THE_SEEDS():
    src = open(RUN.__file__, encoding="utf-8").read()
    for forbidden in ("argparse", "--authorize", "--seed-block", "os.environ",
                      "getenv"):
        assert forbidden not in src, forbidden


# ═══════════════════════ create-only outputs ════════════════════════════════

def test_the_output_paths_are_CREATE_ONLY_and_must_be_THREE_files(tmp_path):
    r, t, rep = (tmp_path / "r.jsonl", tmp_path / "t.jsonl", tmp_path / "rep.json")
    RUN.check_output_paths(str(r), str(t), str(rep))          # all absent: fine
    with pytest.raises(RUN.H3PilotRunError, match="requires a trace"):
        RUN.check_output_paths(str(r), None, str(rep))
    with pytest.raises(RUN.H3PilotRunError, match="THREE files"):
        RUN.check_output_paths(str(r), str(r), str(rep))
    r.write_text("x")
    with pytest.raises(RUN.H3PilotRunError, match="already exists"):
        RUN.check_output_paths(str(r), str(t), str(rep))


def test_a_DANGLING_SYMLINK_at_an_output_path_is_REFUSED(tmp_path):
    """`exists` follows the link and reports False; `lexists` does not. A dangling
    link is a path that would be followed on write."""
    link = tmp_path / "r.jsonl"
    link.symlink_to(tmp_path / "nowhere")
    assert not os.path.exists(link) and os.path.lexists(link)
    with pytest.raises(RUN.H3PilotRunError, match="already exists"):
        RUN.check_output_paths(str(link), str(tmp_path / "t.jsonl"),
                               str(tmp_path / "rep.json"))


# ═══════════════════════ the schedule is pinned ═════════════════════════════

def test_the_schedule_must_BE_the_frozen_40_by_digest():
    tasks = R.build_tasks(R.generate_openings())
    got = RUN.check_schedule(tasks)
    assert got["n_tasks"] == 40 and got["pairs"] == 20
    assert got["task_digest"] == R.TASK_DIGEST


def test_a_DIFFERENT_schedule_is_REFUSED():
    tasks = R.build_tasks(R.generate_openings())
    tasks[0] = dict(tasks[0], anchor_colour="red")
    with pytest.raises(RUN.H3PilotRunError, match="different schedule"):
        RUN.check_schedule(tasks)


def test_a_SHORT_schedule_is_refused_because_a_budget_bounds_nothing_below():
    tasks = R.build_tasks(R.generate_openings())
    with pytest.raises(RUN.H3PilotRunError, match="expected"):
        RUN.check_schedule(tasks[:30])


# ═══════════════ the production seam is INERT under a test runner ═══════════

def test_THE_SEAM_CHECKS_THE_GATE_ITSELF_not_only_the_entry():
    play = RUN._production_play("/tmp/h3-never-written")
    with pytest.raises(RUN.H3PilotRunError, match="NOT AUTHORIZED"):
        play(task={}, identity={}, timeout_s=1)


def test_THE_BOUNDARY_REFUSES_INSIDE_A_TEST_PROCESS_even_with_the_gate_OPEN(monkeypatch):
    """🔑 THE CONTAINMENT ITSELF. The gate is forced open -- the exact state the
    2026-09-12 incident created -- and the boundary must STILL refuse."""
    monkeypatch.setattr(RUN, "H3_PILOT_EXECUTION_AUTHORIZED", True)
    RUN.check_gate()                                   # the gate no longer refuses
    play = RUN._production_play("/tmp/h3-never-written")
    with pytest.raises(RUN.H3PilotContainmentError, match="loaded"):
        play(task={}, identity={}, timeout_s=1)


def test_the_boundary_refuses_with_EVERY_authorization_check_removed(monkeypatch):
    """The incident's exact shape: no gate anywhere."""
    monkeypatch.setattr(RUN, "H3_PILOT_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(RUN, "check_gate", lambda: None)
    play = RUN._production_play("/tmp/h3-never-written")
    with pytest.raises(RUN.H3PilotContainmentError):
        play(task={}, identity={}, timeout_s=1)


def test_the_seam_checks_BEFORE_anything_effectful():
    """Order, by AST: the gate and the boundary precede the toolchain, the
    compile, the evaluator and the game loop."""
    src = inspect.getsource(RUN._production_play)
    import textwrap
    outer = ast.parse(textwrap.dedent(src)).body[0]
    fn = next(n for n in ast.walk(outer)
              if isinstance(n, ast.FunctionDef) and n.name == "play")
    lines = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
            if name:
                lines[name] = min(lines.get(name, node.lineno), node.lineno)
    for guard in ("check_gate", "_assert_the_production_acts_are_inert"):
        assert guard in lines, guard
    for effect in ("verified_paths", "_default_compile", "_default_load_evaluator"):
        assert effect in lines, effect
        assert lines["check_gate"] < lines[effect], f"gate after {effect}"
        assert lines["_assert_the_production_acts_are_inert"] < lines[effect], effect


def test_the_gate_is_checked_for_EVERY_GAME_not_only_the_first():
    src = inspect.getsource(RUN._production_play)
    assert src.count("check_gate()") >= 2, src.count("check_gate()")


def test_the_containment_error_is_a_FAILURE_not_a_verdict():
    assert issubclass(RUN.H3PilotContainmentError, RUN.H3PilotRunError)
    assert not issubclass(RUN.H3PilotContainmentError, RUN.H3PilotVoidError)


def test_NOTHING_EFFECTFUL_IS_IMPORTED_AT_MODULE_LEVEL():
    """Importing the runner must start no JVM and load no model."""
    mod = ast.parse(open(RUN.__file__, encoding="utf-8").read())
    top = {n.module for n in mod.body if isinstance(n, ast.ImportFrom)}
    for effectful in ("e4_screen_runner", "e4_screen_integration", "t1j_toolchain",
                      "d1_probe", "twixtbot_g3_reference", "e4_screen_command"):
        assert effectful not in {str(t) for t in top}, effectful


# ═══════════════ the REAL builder accepts the pilot's own tasks ═════════════

def _stub_evaluator():
    class _Eval:
        _g3_reference = "calib020_0001"
        _g3_sha1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"
    return _Eval()


def test_EVERY_PAIRS_BOTH_TASKS_construct_through_the_REAL_builder():
    """🔴 THE CHECK ATTEMPT 1's VOID MADE NECESSARY: the builder refused H2's own
    readout and nothing caught it, because every seam test had mocked the builder.
    All 40 tasks, both colour assignments, through the real one."""
    from scripts.GPU.alphazero import eval_readout as RO
    from scripts.GPU.alphazero import h2_match_rules as H2R
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3
    cfg = G3.eval_config()
    argmax = cfg.__class__(**{**cfg.__dict__, "selection_mode": H2R.SELECTION_MODE})
    tasks = R.build_tasks(R.generate_openings())
    seen_colours = set()
    for t in tasks:
        task = dict(t, seed=777000000 + t["index"],
                    reference="calib020_0001",
                    reference_sha1="209cf2d4fd24a48553d259dd71b4954867b9473e")
        agent = G3.build_reference_agent(
            task=task, evaluator=_stub_evaluator(),
            colour=REF.reference_colour(task), config=argmax, capture=True)
        assert agent.readout.mode == RO.MODE_ARGMAX
        assert agent.config.mcts_sims == H2R.MCTS_SIMS
        assert agent.seed == task["seed"], "the SCHEDULED seed, not another"
        seen_colours.add(REF.reference_colour(task))
    assert seen_colours == {"red", "black"}, "one arm would prove only one arm"


def test_the_builder_REFUSES_the_WRONG_COLOUR_for_the_arm():
    """The negative half: if it accepted any colour the test above proves nothing."""
    from scripts.GPU.alphazero import h2_match_rules as H2R
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3
    cfg = G3.eval_config()
    argmax = cfg.__class__(**{**cfg.__dict__, "selection_mode": H2R.SELECTION_MODE})
    t = R.build_tasks(R.generate_openings())[0]
    task = dict(t, seed=777000000, reference="calib020_0001",
                reference_sha1="209cf2d4fd24a48553d259dd71b4954867b9473e")
    wrong = "red" if REF.reference_colour(task) == "black" else "black"
    with pytest.raises(Exception):
        G3.build_reference_agent(task=task, evaluator=_stub_evaluator(),
                                 colour=wrong, config=argmax, capture=True)


def test_the_seam_WIRES_THE_HARNESS_GAME_LOOP():
    """By AST, not by reading source: a refusal inserted above the loop went
    unseen when the test only grepped."""
    import textwrap
    outer = ast.parse(textwrap.dedent(inspect.getsource(RUN._production_play))).body[0]
    fn = next(n for n in ast.walk(outer)
              if isinstance(n, ast.FunctionDef) and n.name == "play")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    assert any(getattr(c.func, "attr", "") == "play_task" for c in calls), \
        "the seam must call e4_screen_runner.play_task"


def test_THE_SEAM_TIMES_EVERY_GAME_ON_A_MONOTONIC_CLOCK():
    """🔑 A wall clock can step backwards over an NTP correction and emit a
    negative duration, which the analysis refuses -- a game voided for the
    weather."""
    src = inspect.getsource(RUN._production_play)
    assert "time.monotonic()" in src
    assert "time.time()" not in src, "a wall clock must not time a game"
    assert 'result["elapsed_s"]' in src


def test_the_cleanup_runs_after_EVERY_game_including_a_failed_one():
    """H1 clears state after every game; H2 played 736 and called it never."""
    import textwrap
    outer = ast.parse(textwrap.dedent(inspect.getsource(RUN._production_play))).body[0]
    fn = next(n for n in ast.walk(outer)
              if isinstance(n, ast.FunctionDef) and n.name == "play")
    tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try) and n.finalbody]
    assert tries, "the game loop must clean up in a finally"
    body = "\n".join(ast.dump(n) for t in tries for n in t.finalbody)
    assert "cleanup" in body
