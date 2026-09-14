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
from scripts.GPU.alphazero import h2_match_rules as H2R
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
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call), (
        f"THE FIRST STATEMENT IN run_pilot IS A {type(first).__name__}, NOT A CALL")
    assert getattr(first.value.func, "id", "") == "check_gate", ast.dump(first)


def test_run_pilot_REFUSES_a_set_that_does_not_match_the_pin(monkeypatch, tmp_path):
    """The positions are the experiment. If the generator drifts from the pinned
    set the pilot answers a different question, so it refuses rather than run.

    The gate is neutralised here ONLY so the check after it can be reached; the
    refusal happens long before any seam, seed or file."""
    monkeypatch.setattr(RUN, "check_gate", lambda: None)
    monkeypatch.setattr(RUN.RULES, "OPENING_SET_DIGEST", "0" * 64)
    with pytest.raises(RUN.H3PilotRunError, match="does not match the frozen pin"):
        RUN.run_pilot(results_path=str(tmp_path / "r.jsonl"),
                      trace_path=str(tmp_path / "t.jsonl"),
                      report_path=str(tmp_path / "rep.json"))
    assert not list(tmp_path.iterdir()), "it refused before writing anything"


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


def test_check_schedule_REFUSES_a_seeded_schedule_while_the_pin_is_unset():
    """🔴 Assigning 40 seeds changes the full-field digest, so the runner must
    know WHICH pin applies. Compared against the unseeded one, the real schedule
    would be refused by its own check at execution time."""
    tasks = R.build_tasks(R.generate_openings(),
                          seed_interval=(777000000, 777000040))
    with pytest.raises(RUN.H3PilotRunError, match="SEEDED_TASK_DIGEST is None"):
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
    for guard in ("check_gate", "assert_production_acts_are_inert"):
        assert guard in lines, guard
    for effect in ("verified_paths", "_default_compile", "_default_load_evaluator"):
        assert effect in lines, effect
        assert lines["check_gate"] < lines[effect], f"gate after {effect}"
        assert lines["assert_production_acts_are_inert"] < lines[effect], effect


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
        # 🔑 ASSERTED INDEPENDENTLY, not derived. `reference_colour` reads
        # `anchor_colour`, so passing its own output back in is self-consistent
        # however wrong the anchor is -- a control that moved the anchor went NOT
        # CAUGHT against the first version of this line.
        assert REF.reference_colour(task) == task["incumbent_colour"], (
            f"{task['task_id']}: the anchor and the incumbent's colour disagree")
        agent = G3.build_reference_agent(
            task=task, evaluator=_stub_evaluator(),
            colour=task["incumbent_colour"], config=argmax, capture=True)
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
    fn = ast.parse(textwrap.dedent(inspect.getsource(RUN._play_one))).body[0]
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    assert any(getattr(c.func, "attr", "") == "play_task" for c in calls), \
        "the seam must call e4_screen_runner.play_task"


def test_THE_SEAM_TIMES_EVERY_GAME_ON_A_MONOTONIC_CLOCK():
    """🔑 A wall clock can step backwards over an NTP correction and emit a
    negative duration, which the analysis refuses -- a game voided for the
    weather."""
    src = inspect.getsource(RUN._play_one)
    assert "time.monotonic()" in src
    assert "time.time()" not in src, "a wall clock must not time a game"
    assert "elapsed = time.monotonic() - t0" in src


def test_the_cleanup_runs_after_EVERY_game_including_a_failed_one():
    """H1 clears state after every game; H2 played 736 and called it never."""
    import textwrap
    fn = ast.parse(textwrap.dedent(inspect.getsource(RUN._play_one))).body[0]
    tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try) and n.finalbody]
    assert tries, "the game loop must clean up in a finally"
    body = "\n".join(ast.dump(n) for t in tries for n in t.finalbody)
    assert "cleanup" in body


# ═══════════ THE RUN BODY, driven with INERT COLLABORATORS ══════════════════
# Nothing below touches a toolchain, a model, a JVM or a seed: the play seam is
# supplied, and the private entry exists precisely so the machinery can be
# exercised WITHOUT lifting the gate.

import contextlib
import json


def _inert_play(*, cap_every=None, fail_at=None, slow_after=None, elapsed=1.0):
    """A play seam that plays nothing. Returns the shape the harness returns."""
    calls = {"n": 0}

    def play(*, task, identity, timeout_s):
        i = calls["n"]
        calls["n"] += 1
        if fail_at is not None and i == fail_at:
            raise RuntimeError("the inert seam was told to fail")
        capped = cap_every is not None and i % cap_every == 0
        bound = task["opening_plies"]
        n_plies = 4
        # 🔑 DISTINCT PER TASK. The first version used `i % 20`, which made pairs
        # 0 and 10, 1 and 11 ... byte-identical -- ten duplicate pairs, and S1
        # fired on what was supposed to be a CLEAN run. The fixture, not the code.
        plies = [{"ply": bound + k + 1, "mover": H2R.colour_at_ply(bound + k + 1),
                  "move": [i, k], "record_type": "ply"} for k in range(n_plies)]
        winner = None if capped else task["incumbent_colour"]
        row = {"terminal_reason": "cap" if capped else "win", "winner": winner,
               "plies": bound + n_plies, "seed": task["seed"],
               "t1j_points": 0.5 if capped else 0.0}
        return {"result": row, "plies": plies, "opening_bound": bound,
                "elapsed_s": elapsed}

    play.calls = calls
    return play


class _FakeDeadline:
    """A deadline whose clock the test drives. `elapsed` is read, never wall time."""
    def __init__(self, budget, expire_after=None):
        self.budget, self.expire_after, self.n = budget, expire_after, 0
        self.started = False

    def start(self):
        self.started = True
        return self

    def elapsed(self):
        self.n += 1
        if self.expire_after is not None and self.n > self.expire_after:
            return self.budget + 1
        return 0.0


@contextlib.contextmanager
def _no_supervisor(deadline):
    yield


def _run(tmp_path, play, deadline=None):
    from scripts.GPU.alphazero import h3_pilot_rules as RULES
    ops = RULES.generate_openings()
    tasks = RULES.build_tasks(ops, seed_interval=(777000000, 777000040))
    # the seed barrier is the runner's, and these tasks are not from a real block
    import unittest.mock as _m
    with _m.patch.object(RUN, "PILOT_SEED_BLOCK", (777000000, 777000040)), \
         _m.patch.object(RUN, "check_seed_registration", lambda: None), \
         _m.patch.object(RUN, "check_schedule", lambda t: {
             "n_tasks": len(t), "task_digest": "x" * 64,
             "pairs": len({x["pair_id"] for x in t})}):
        return RUN._run_pilot_unguarded(
            tasks=tasks, openings=ops,
            results_path=str(tmp_path / "r.jsonl"),
            trace_path=str(tmp_path / "t.jsonl"),
            report_path=str(tmp_path / "rep.json"),
            play=play, deadline_s=7200,
            _deadline=deadline or _FakeDeadline(7200),
            _supervisor=_no_supervisor)


def _lines(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def test_A_COMPLETE_RUN_plays_all_40_writes_all_three_and_reports(tmp_path):
    play = _inert_play()
    rep = _run(tmp_path, play)
    assert play.calls["n"] == 40
    assert rep["complete"] is True and rep["timed_out"] is False
    assert rep["games_completed"] == 40 and rep["pairs_scored"] == 20
    for name in ("r.jsonl", "t.jsonl", "rep.json"):
        assert (tmp_path / name).exists(), name
    trace = _lines(tmp_path / "t.jsonl")
    assert trace[0]["event"] == "run_start"
    assert trace[-1] == {"event": "run_end", "verdict": "OK",
                         "games_completed": 40, "any_stop_rule_fired": False}


def test_the_results_file_carries_the_transcripts_and_the_durations(tmp_path):
    """A screen whose input is unrecorded is a number taken on trust."""
    _run(tmp_path, _inert_play())
    rows = _lines(tmp_path / "r.jsonl")
    kinds = {r["record_type"] for r in rows}
    assert {"header", "ply", "transcript", "task_result"} <= kinds
    results = [r for r in rows if r["record_type"] == "task_result"]
    assert len(results) == 40
    assert all("elapsed_s" in r and r["elapsed_s"] >= 0 for r in results)
    assert all(len(r["transcript_digest"]) == 64 for r in results)


def test_the_report_on_disk_IS_the_report_returned(tmp_path):
    rep = _run(tmp_path, _inert_play())
    on_disk = json.load(open(tmp_path / "rep.json"))
    assert on_disk["pairs_distinct"] == rep["pairs_distinct"]
    assert on_disk["is_strength_verdict"] is False


def test_A_TIMEOUT_STOPS_CLEANLY_and_still_reports_PARTIAL(tmp_path):
    """🔑 THE CARD'S POINT, and the one place this differs from H2: a partial run
    is informative, so the deadline is not a VOID."""
    play = _inert_play()
    rep = _run(tmp_path, play, deadline=_FakeDeadline(7200, expire_after=12))
    assert play.calls["n"] == 12, "it stopped at the deadline, not after 40"
    assert rep["timed_out"] is True and rep["complete"] is False
    trace = _lines(tmp_path / "t.jsonl")
    assert any(e["event"] == "deadline_reached" for e in trace)
    assert trace[-1]["verdict"] == "PARTIAL", "a deadline is not a VOID"
    assert (tmp_path / "rep.json").exists(), "a partial run still reports"


def test_a_partial_report_declares_NOTHING_clear(tmp_path):
    rep = _run(tmp_path, _inert_play(), deadline=_FakeDeadline(7200, expire_after=12))
    assert rep["stop_rules"]["S4b"]["status"] == "UNRESOLVED -- SCHEDULE INCOMPLETE"
    for rule in ("S1", "S2", "S3", "S4a"):
        assert rep["stop_rules"][rule]["status"] != "CLEAR"


def test_AN_EXCEPTION_IS_A_VOID_and_the_trace_says_so(tmp_path):
    """A deadline stops cleanly; a fault does not."""
    with pytest.raises(RUN.H3PilotVoidError, match="RuntimeError"):
        _run(tmp_path, _inert_play(fail_at=7))
    trace = _lines(tmp_path / "t.jsonl")
    assert trace[-1]["verdict"] == "VOID"
    assert trace[-1]["games_completed"] == 7
    assert not (tmp_path / "rep.json").exists(), "a VOID writes no report"


def test_the_terminal_record_is_written_AFTER_the_report_is_durable(tmp_path):
    """H2's lesson: run_end/OK before the report meant a write failure left the
    trace and the exit code disagreeing about the same run."""
    _run(tmp_path, _inert_play())
    src = inspect.getsource(RUN._run_pilot_unguarded)
    assert src.index("json.dump(report") < src.index('"event": "run_end",\n'
                                                     '                     "verdict"')


def test_the_run_REFUSES_when_an_output_already_exists(tmp_path):
    (tmp_path / "r.jsonl").write_text("x")
    with pytest.raises(RUN.H3PilotRunError, match="already exists"):
        _run(tmp_path, _inert_play())


def test_O_EXCL_refuses_even_with_the_precheck_disabled(tmp_path, monkeypatch):
    """The create-only OPEN, REACHED ALONE.

    🔑 `check_output_paths` refuses an existing file first, so removing `O_EXCL`
    from the open changes nothing any ordinary test can see -- a control aimed at
    the precheck's test went NOT CAUGHT. The open is defence in depth against the
    gap BETWEEN the precheck and the write, and the only way to exercise it is to
    disable the check in front of it.
    """
    monkeypatch.setattr(RUN, "check_output_paths", lambda *a, **k: None)
    (tmp_path / "r.jsonl").write_text("an earlier run's file")
    with pytest.raises(RUN.H3PilotVoidError, match="FileExistsError"):
        _run(tmp_path, _inert_play())
    assert (tmp_path / "r.jsonl").read_text() == "an earlier run's file", \
        "the existing file must not be truncated"


def test_a_capped_game_is_recorded_and_counted(tmp_path):
    rep = _run(tmp_path, _inert_play(cap_every=4))
    assert rep["capped_games"] == 10
    assert rep["stop_rules"]["S2"]["status"] == "FIRED", "10 > 8"
    assert rep["any_fired"] is True


# ═════════════ THE WRAPPER: gate restoration and process cleanup ════════════

from scripts.GPU.alphazero import h3_pilot_command as CMD


def test_the_wrapper_refuses_with_the_shut_gate_and_verifies_it_closed(capsys):
    assert CMD.main([]) == CMD.EXIT_UNAUTHORIZED
    assert "NOT AUTHORIZED" in capsys.readouterr().err
    assert RUN.H3_PILOT_EXECUTION_AUTHORIZED is False


def test_the_wrapper_has_NO_runner_source_flag():
    assert "--runner-source" not in CMD._parser().format_help()
    assert "os.environ" not in open(CMD.__file__, encoding="utf-8").read()


def test_restore_gate_is_TRUE_when_the_gate_is_already_closed():
    assert CMD.restore_gate() is True


def test_restore_gate_REWRITES_an_open_gate_and_VERIFIES_it(tmp_path):
    f = tmp_path / "runner.py"
    f.write_text("x = 1\nH3_PILOT_EXECUTION_AUTHORIZED = True\ny = 2\n")
    assert CMD.restore_gate(str(f)) is True
    assert "H3_PILOT_EXECUTION_AUTHORIZED = False" in f.read_text()
    assert "= True" not in f.read_text()


def test_restore_gate_is_FALSE_when_it_cannot_verify(tmp_path):
    missing = tmp_path / "nope.py"
    assert CMD.restore_gate(str(missing)) is False
    two = tmp_path / "two.py"
    two.write_text("H3_PILOT_EXECUTION_AUTHORIZED = True\n" * 2)
    assert CMD.restore_gate(str(two)) is False, "two open lines: which is the gate?"


def test_a_failed_restoration_becomes_the_wrappers_OWN_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(CMD, "restore_gate", lambda *a, **k: False)
    assert CMD.main([]) == CMD.EXIT_GATE_NOT_RESTORED
    assert "BY HAND" in capsys.readouterr().err


def test_THE_FINALLY_PATH_also_restores_and_reports_its_own_failure(
        monkeypatch, capsys, tmp_path):
    """The OTHER path -- gate open, restoration failing -- which is the one a real
    run takes. Hermetic: tmp outputs and a stubbed supervisor, so its behaviour
    cannot change the day a real run occupies the defaults."""
    monkeypatch.setattr(CMD, "gate_is_open", lambda: True)
    monkeypatch.setattr(CMD, "restore_gate", lambda *a, **k: False)
    monkeypatch.setattr(CMD, "supervise", lambda *a, **k: {
        "exit_code": 0, "timed_out": False, "interrupted": False,
        "group_cleared": True})
    argv = ["--results", str(tmp_path / "r.jsonl"), "--trace", str(tmp_path / "t.jsonl"),
            "--report", str(tmp_path / "rep.json")]
    assert CMD.main(argv) == CMD.EXIT_GATE_NOT_RESTORED
    assert "BY HAND" in capsys.readouterr().err
    assert not list(tmp_path.iterdir()), "the stubbed supervisor wrote nothing"


def test_a_SURVIVING_DESCENDANT_is_a_CLEANUP_FAILURE_not_a_success(
        monkeypatch, capsys, tmp_path):
    """🔑 `group_cleared` False outranks the worker's own exit code: an orphaned
    JVM is not a completed run."""
    monkeypatch.setattr(CMD, "gate_is_open", lambda: True)
    monkeypatch.setattr(CMD, "supervise", lambda *a, **k: {
        "exit_code": 0, "timed_out": False, "interrupted": False,
        "group_cleared": False})
    argv = ["--results", str(tmp_path / "r.jsonl"), "--trace", str(tmp_path / "t.jsonl"),
            "--report", str(tmp_path / "rep.json")]
    assert CMD.main(argv) == CMD.EXIT_CLEANUP_FAILED
    assert "CLEANUP FAILED" in capsys.readouterr().err


@pytest.mark.parametrize("r,want", [
    ({"timed_out": False, "interrupted": False, "group_cleared": True,
      "exit_code": 0}, CMD.EXIT_COMPLETED),
    ({"timed_out": True, "interrupted": False, "group_cleared": True,
      "exit_code": 0}, CMD.EXIT_TIMEOUT),
    ({"timed_out": False, "interrupted": True, "group_cleared": True,
      "exit_code": 0}, CMD.EXIT_INTERRUPTED),
    ({"timed_out": False, "interrupted": False, "group_cleared": False,
      "exit_code": 0}, CMD.EXIT_CLEANUP_FAILED),
])
def test_every_supervisor_outcome_gets_ITS_OWN_exit_code(monkeypatch, tmp_path, r, want):
    monkeypatch.setattr(CMD, "gate_is_open", lambda: True)
    monkeypatch.setattr(CMD, "supervise", lambda *a, **k: dict(r))
    argv = ["--results", str(tmp_path / "r.jsonl"), "--trace", str(tmp_path / "t.jsonl"),
            "--report", str(tmp_path / "rep.json")]
    assert CMD.main(argv) == want


def test_the_wrapper_REFUSES_BEFORE_SPAWNING_when_an_output_exists(
        monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(CMD, "gate_is_open", lambda: True)
    spawned = []
    monkeypatch.setattr(CMD, "supervise", lambda *a, **k: spawned.append(1) or {
        "exit_code": 0, "timed_out": False, "interrupted": False,
        "group_cleared": True})
    (tmp_path / "r.jsonl").write_text("x")
    argv = ["--results", str(tmp_path / "r.jsonl"), "--trace", str(tmp_path / "t.jsonl"),
            "--report", str(tmp_path / "rep.json")]
    assert CMD.main(argv) == CMD.EXIT_REFUSED
    assert not spawned, "it must refuse BEFORE spawning a worker"


def test_WORKER_REFUSES_without_the_supervisors_CAPABILITY(capsys):
    assert CMD.worker_main(["--worker"]) == CMD.EXIT_REFUSED
    assert "not a way to run the pilot by hand" in capsys.readouterr().err


def test_a_FORGED_capability_is_refused_INCLUDING_one_of_the_RIGHT_LENGTH(tmp_path):
    r_fd, w_fd = os.pipe()
    with os.fdopen(w_fd, "w") as fh:
        fh.write("z" * (CMD.CAPABILITY_BYTES * 2))     # right length, not hex
    assert CMD._consume_capability(r_fd) is False


def test_the_capability_is_a_PIPE_and_is_SINGLE_USE():
    fd = CMD._make_capability()
    assert CMD._consume_capability(fd) is True
    assert CMD._consume_capability(fd) is False, "the descriptor is closed after use"


def test_NOTHING_IS_EVER_DELETED_by_the_capability_check(tmp_path):
    """🔴 A previous design UNLINKED whatever path it was handed, valid or not."""
    victim = tmp_path / "precious.txt"
    victim.write_text("keep me")
    assert CMD._consume_capability(12345) is False       # not a real descriptor
    assert victim.read_text() == "keep me"


# ══════════ the outcome classifier: results, not failures ═══════════════════

@pytest.mark.parametrize("report,want", [
    ({"complete": True, "timed_out": False, "any_fired": False}, CMD.EXIT_COMPLETED),
    ({"complete": False, "timed_out": True, "any_fired": False}, CMD.EXIT_PARTIAL),
    ({"complete": True, "timed_out": False, "any_fired": True},
     CMD.EXIT_STOP_RULE_FIRED),
    # 🔑 A FIRED RULE OUTRANKS PARTIAL: a monotone count that has crossed its
    # threshold is CONCLUSIVE; a partial run is not.
    ({"complete": False, "timed_out": True, "any_fired": True},
     CMD.EXIT_STOP_RULE_FIRED),
])
def test_the_classifier_separates_RESULTS_from_failures(report, want):
    assert CMD._classify(report) == want


def test_THE_OUTPUT_DESTINATION_IS_NOT_A_SPENT_RUNS_DIRECTORY():
    defaults = (CMD.DEFAULT_RESULTS, CMD.DEFAULT_TRACE, CMD.DEFAULT_REPORT)
    assert len(set(defaults)) == 3
    assert CMD.SPENT_OUT_DIRS, "vacuous: no spent directory is named"
    for spent in CMD.SPENT_OUT_DIRS:
        assert os.path.isdir(spent), f"{spent} is named as spent but does not exist"
        for d in defaults:
            assert not d.startswith(spent.rstrip("/") + "/"), (d, spent)
    for d in defaults:
        assert d.startswith(CMD.OUT_DIR.rstrip("/") + "/")


# ════════ THE ACTUAL SEAM, DRIVEN INTO THE REAL HARNESS ═════════════════════
# 🔴 THE INTERFACE THE FIXTURES REPLACED. Every test above supplies its own
# `play`, so neither of these could be seen:
#   * `rec=None` was passed, and `play_task` calls `rec.emit(...)` for the
#     `opening_bound` BEFORE A SINGLE MOVE -- it would have failed on game one;
#   * `play_task` returns a FLAT dict, while the run body reads `result`, `plies`
#     and `opening_bound`, so `out["result"]` would have raised on game one.
# These drive `RUN._play_one` into the REAL `e4_screen_runner.play_task` with
# inert agents, an inert binder and no toolchain, model or JVM anywhere.

def _inert_state(openings, *, cleanups):
    """A prepared seam state whose HARNESS IS REAL and whose players are not."""
    from scripts.GPU.alphazero import e4_screen_runner as HARNESS
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    mapping = R.openings_mapping(openings)

    def state_factory(task):
        st = TwixtState()
        for mv in mapping[task["opening"]]:
            st = st.apply_move(tuple(mv))
        return st

    def agent_for(task, mover):
        # the first legal move, deterministically: a player, not a strategy
        return lambda state: sorted(state.legal_moves())[0]

    return {"harness": HARNESS, "state_factory": state_factory,
            "agent_factory": agent_for, "binder": lambda *a, **k: None,
            "cleanup": lambda: cleanups.append(1)}


def test_THE_SEAM_DRIVES_THE_REAL_HARNESS_AND_RETURNS_THE_RUN_BODYS_CONTRACT():
    """The real `play_task`, a real recorder, real records -- and the exact shape
    `_run_pilot_unguarded` consumes."""
    openings = R.generate_openings()
    task = dict(R.build_tasks(openings)[0], seed=777000000)
    cleanups = []
    out = RUN._play_one(task=task, state=_inert_state(openings, cleanups=cleanups),
                        timeout_s=1.0)

    assert set(out) >= {"result", "plies", "opening_bound", "records", "elapsed_s"}
    assert out["opening_bound"] == R.OPENING_PLIES == 6
    assert out["result"]["task_id"] == task["task_id"]
    assert out["result"]["seed"] == 777000000
    assert out["result"]["terminal_reason"] in ("win", "cap")
    assert isinstance(out["result"]["plies"], int)
    assert out["plies"], "the harness emitted no ply records"
    assert out["elapsed_s"] >= 0
    assert cleanups == [1], "cleanup runs after the game"


def test_THE_RECORDER_IS_REAL_and_receives_the_openings_bound_first():
    """`rec=None` would have raised here, before a single move."""
    openings = R.generate_openings()
    task = dict(R.build_tasks(openings)[0], seed=777000000)
    out = RUN._play_one(task=task, state=_inert_state(openings, cleanups=[]),
                        timeout_s=1.0)
    kinds = [r["record_type"] for r in out["records"]]
    assert kinds[0] == "opening_bound", kinds[:3]
    assert kinds.count("opening_bound") == 1
    assert set(kinds) == {"opening_bound", "ply"}


def test_THE_RECORDS_THE_SEAM_RETURNS_BUILD_A_TRANSCRIPT():
    """The end of the chain: what the harness emits must satisfy H2's transcript
    contract, which the run body applies to every game."""
    openings = R.generate_openings()
    task = dict(R.build_tasks(openings)[0], seed=777000000)
    out = RUN._play_one(task=task, state=_inert_state(openings, cleanups=[]),
                        timeout_s=1.0)
    from scripts.GPU.alphazero import h3_pilot_analysis as A
    t = A.transcript(out["plies"], out["result"],
                     opening_bound=out["opening_bound"])
    assert t[-1][0] == "terminal"
    assert len(A.transcript_digest(t)) == 64


def test_MORE_THAN_ONE_opening_bound_is_a_VOID():
    """The anchor the ply span is measured from must be unique."""
    openings = R.generate_openings()
    task = dict(R.build_tasks(openings)[0], seed=777000000)
    state = dict(_inert_state(openings, cleanups=[]))

    real = RUN._capturing_recorder          # captured BEFORE the patch, or the
                                            # doubling recorder builds itself

    class _Doubling:
        def __init__(self):
            self.inner = real()

        def emit(self, obj):
            self.inner.emit(obj)
            if obj.get("record_type") == "opening_bound":
                self.inner.emit(dict(obj))

        @property
        def records(self):
            return self.inner.records

    import unittest.mock as _m
    with _m.patch.object(RUN, "_capturing_recorder", _Doubling):
        with pytest.raises(RUN.H3PilotVoidError, match="opening_bound"):
            RUN._play_one(task=task, state=state, timeout_s=1.0)


def test_THE_RUN_BODY_CONSUMES_THE_REAL_SEAMS_OUTPUT(tmp_path):
    """End to end over the interface: the real harness's records, through the real
    run body, into a real report -- with only the players inert."""
    openings = R.generate_openings()
    cleanups = []
    state = _inert_state(openings, cleanups=cleanups)
    rep = _run(tmp_path, lambda *, task, identity, timeout_s: RUN._play_one(
        task=task, state=state, timeout_s=timeout_s))
    assert rep["games_completed"] == 40 and rep["complete"] is True
    assert len(cleanups) == 40, "cleanup after every game"
    results = [r for r in _lines(tmp_path / "r.jsonl")
               if r["record_type"] == "task_result"]
    assert len(results) == 40
    assert all(len(r["transcript_digest"]) == 64 for r in results)
