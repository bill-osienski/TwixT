"""D1 fail-closed machinery. NO EXECUTION.

No model is loaded, no JVM started, no seed registered or drawn, no position
queried, no game played. Where a test exercises the real query path it patches
`subprocess.run` -- the process boundary -- so the whole of our own code runs
while nothing is ever spawned.

The timeout tests deliberately observe at `subprocess.run` and NOT at the call
site. There are three default-None hops between a caller and that boundary, and
each one silently restores unbounded waiting; proving the value was passed in at
the top proves nothing about whether it arrived.
"""
import signal
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import d1_probe as D1
from scripts.GPU.alphazero import t1j_adapter as A

MOVES = [(11, 11), (12, 13), (13, 12), (10, 13), (12, 10), (14, 14)]


def _moves_digest():
    """12.7's recorded digest for MOVES, computed the way selection computes it."""
    from scripts.GPU.alphazero import d1_selection as _SEL
    from scripts.GPU.alphazero.game.twixt_state import TwixtState as _TS
    st = _TS(active_size=24, to_move="red")
    for m in MOVES:
        st = st.apply_move(m)
    return _SEL.canonical_digest(st)


MOVES_DIGEST = _moves_digest()
RUNTIME = D1.T1jPaths(java="/nonexistent/java", jar="/nonexistent/t1j.jar",
                      classes="/nonexistent/classes", ply_cap=280)


@pytest.fixture
def spy(monkeypatch):
    """Intercept the PROCESS BOUNDARY. Nothing is ever spawned."""
    calls = []

    def fake_run(args, **kw):
        calls.append({"args": args, "kw": kw})
        depth = int(args[args.index("query") + 1]) if "query" in args else 6
        # A REALISTIC reply: completed, legal, real move, and a state dump. The
        # first version of this fixture emitted a QUERY line and no dump, which
        # is what let two empty dumps compare equal and pass.
        return subprocess.CompletedProcess(args, 0, _stdout(depth=depth,
                                                            moveNr=len(MOVES)), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


# ------------------------------------------- timeout arrival AT the boundary

def test_every_t1j_call_reaches_subprocess_run_with_the_frozen_timeout(spy):
    D1._probe_position(moves=MOVES, depth=6, paths=RUNTIME,
                      budget=D1.QueryBudget(D1.QUERY_CAP), deadline=D1.Deadline())
    assert spy, "no subprocess call was observed -- the assertion below would be vacuous"
    for c in spy:
        assert c["kw"].get("timeout") == D1.PER_QUERY_TIMEOUT_S == 120, c["kw"]


def test_the_boundary_check_catches_a_dropped_timeout_hop(spy, monkeypatch):
    """NEGATIVE CONTROL. Drop the value at the LAST hop and the check must fail.

    Without this, a green timeout test proves only that the code happens to work,
    not that the test could ever notice if it stopped.
    """
    real = A.query
    monkeypatch.setattr(A, "query", lambda *a, **k: real(*a, **{**k, "timeout_s": None}))
    D1._probe_position(moves=MOVES, depth=6, paths=RUNTIME,
                      budget=D1.QueryBudget(D1.QUERY_CAP), deadline=D1.Deadline())
    assert spy
    assert any(c["kw"].get("timeout") is None for c in spy), \
        "the injected defect did not reach the boundary; the control proves nothing"


# ------------------------------- duplicate queries are separate repeats=1 JVMs

def test_each_depth_issues_two_separate_query_mode_invocations(spy):
    D1._probe_position(moves=MOVES, depth=6, paths=RUNTIME,
                      budget=D1.QueryBudget(D1.QUERY_CAP), deadline=D1.Deadline())
    assert len(spy) == D1.INVOCATIONS_PER_DEPTH == 2, [c["args"] for c in spy]
    for c in spy:
        assert "query" in c["args"], c["args"]


def test_the_same_jvm_determinism_mode_is_never_used(spy):
    """`repeats>1` reuses ONE process's Zobrist salt, so it cannot test the
    cross-process variable at all. The adapter puts the mode in argv, so the
    prohibition is observable at the boundary rather than asserted about a kwarg."""
    D1._probe_position(moves=MOVES, depth=6, paths=RUNTIME,
                      budget=D1.QueryBudget(D1.QUERY_CAP), deadline=D1.Deadline())
    assert spy
    for c in spy:
        assert "determinism" not in c["args"], c["args"]


def test_two_invocations_are_distinct_processes_not_one_repeated(spy):
    D1._probe_position(moves=MOVES, depth=6, paths=RUNTIME,
                      budget=D1.QueryBudget(D1.QUERY_CAP), deadline=D1.Deadline())
    assert len(spy) == 2 and spy[0]["args"] == spy[1]["args"], \
        "two identical invocations expected -- same argv, separate processes"


# ------------------------------------------------------------- the deadline

def test_deadline_uses_a_monotonic_clock_by_default():
    import time as _t
    assert D1.Deadline()._clock is _t.monotonic


def test_deadline_limit_is_ninety_minutes():
    assert D1.Deadline().limit_s == D1.RUN_DEADLINE_S == 90 * 60


def test_an_unstarted_deadline_is_void_not_silently_ignored():
    with pytest.raises(D1.D1VoidError, match="never started"):
        D1.Deadline().check("anywhere")


def test_deadline_breach_yields_void():
    ticks = iter([0.0, 5401.0])
    d = D1.Deadline(clock=lambda: next(ticks)).start()
    with pytest.raises(D1.D1VoidError, match="deadline exceeded"):
        d.check("mid-run")


def test_the_deadline_starts_before_helper_compilation(tmp_path, registered):
    """A window opened after compilation cannot bound compilation."""
    seen = {}

    def spy_compile(deadline):
        seen["started"] = deadline.started
        seen["elapsed"] = deadline.elapsed()

    D1._run_d1_unguarded(positions=[], paths=RUNTIME, out_path=str(tmp_path / "r.json"),
              _compile=spy_compile)
    assert seen["started"] is True, "compilation ran before the deadline started"
    assert seen["elapsed"] >= 0.0


# ------------------------------------------- VOID produces NO partial analysis

def test_a_forced_query_timeout_is_void_and_writes_no_report(tmp_path, monkeypatch, registered):
    def boom(args, **kw):
        raise subprocess.TimeoutExpired(args, kw.get("timeout"))
    monkeypatch.setattr(subprocess, "run", boom)
    out = tmp_path / "r.json"
    with pytest.raises(D1.D1VoidError, match="timed out"):
        D1._run_d1_unguarded(positions=[{"task_id": "t", "ply": len(MOVES), "prefix": MOVES,
                              "seed": D1.SEED_INTERVAL[0], "digest": MOVES_DIGEST}],
                  paths=RUNTIME, out_path=str(out), _compile=lambda d: None)
    assert not out.exists(), "a VOID run wrote a report -- that is partial analysis"


def test_a_forced_deadline_breach_is_void_and_writes_no_report(tmp_path, spy, registered):
    ticks = iter([0.0, 0.0, 99999.0, 99999.0, 99999.0, 99999.0])
    out = tmp_path / "r.json"
    with pytest.raises(D1.D1VoidError, match="deadline exceeded"):
        D1._run_d1_unguarded(positions=[{"task_id": "t", "ply": len(MOVES), "prefix": MOVES,
                              "seed": D1.SEED_INTERVAL[0], "digest": MOVES_DIGEST}],
                  paths=RUNTIME, out_path=str(out), _compile=lambda d: None,
                  deadline=D1.Deadline(clock=lambda: next(ticks)))
    assert not out.exists(), "a VOID run wrote a report -- that is partial analysis"


def test_void_is_raised_not_returned_so_a_caller_cannot_ignore_it(tmp_path, monkeypatch, registered):
    monkeypatch.setattr(subprocess, "run",
                        lambda a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired(a, 1)))
    with pytest.raises(D1.D1VoidError):
        D1._run_d1_unguarded(positions=[{"task_id": "t", "ply": len(MOVES), "prefix": MOVES,
                              "seed": D1.SEED_INTERVAL[0], "digest": MOVES_DIGEST}],
                  paths=RUNTIME, out_path=str(tmp_path / "r.json"), _compile=lambda d: None)


# ---------------------------------------------- budget, prefix, seed interval

def test_the_query_cap_is_the_frozen_value_and_its_arithmetic_holds():
    assert D1.QUERY_CAP == 1135
    assert D1.QUERY_CAP == D1.N_POSITIONS * (1 + len(D1.T1J_DEPTHS) * D1.INVOCATIONS_PER_DEPTH)


def test_the_budget_refuses_the_query_that_would_exceed_the_cap():
    b = D1.QueryBudget(cap=2)
    b.spend(); b.spend()
    with pytest.raises(D1.D1BudgetError, match="exhausted"):
        b.spend()
    assert b.spent == 2, "a refused spend must not be counted"


def test_probing_stops_at_the_cap_rather_than_overrunning_it(spy):
    b = D1.QueryBudget(cap=1)
    with pytest.raises(D1.D1BudgetError):
        D1._probe_position(moves=MOVES, depth=6, paths=RUNTIME,
                          budget=b, deadline=D1.Deadline())
    assert len(spy) == 1, "the budget did not stop the second invocation"


def test_a_position_without_a_retained_prefix_is_void(tmp_path, registered):
    with pytest.raises(D1.D1VoidError, match="no retained move prefix"):
        D1._run_d1_unguarded(positions=[{"task_id": "t", "ply": 6, "seed": D1.SEED_INTERVAL[0]}],
                  paths=RUNTIME, out_path=str(tmp_path / "r.json"), _compile=lambda d: None)


def test_a_prefix_inconsistent_with_its_ply_is_void(tmp_path, registered):
    """A digest cannot be replayed; a prefix of the wrong length replays the
    WRONG POSITION, which is worse than refusing."""
    with pytest.raises(D1.D1VoidError, match="different position"):
        D1._run_d1_unguarded(positions=[{"task_id": "t", "ply": 99, "prefix": MOVES,
                              "seed": D1.SEED_INTERVAL[0]}],
                  paths=RUNTIME, out_path=str(tmp_path / "r.json"), _compile=lambda d: None)


@pytest.mark.parametrize("seed", [202613999, 202614227, 0, -1, True, "202614000", None])
def test_a_seed_outside_the_reserved_interval_is_void(tmp_path, seed, registered):
    with pytest.raises(D1.D1VoidError, match="outside the reserved"):
        D1._run_d1_unguarded(positions=[{"task_id": "t", "ply": len(MOVES), "prefix": MOVES,
                              "seed": seed}],
                  paths=RUNTIME, out_path=str(tmp_path / "r.json"), _compile=lambda d: None)


def test_the_seed_interval_matches_the_plan_and_is_registered_nowhere():
    from scripts.GPU.alphazero import e4_screen_reference as REF
    assert D1.SEED_INTERVAL == (202614000, 202614227)
    assert D1.SEED_INTERVAL[1] - D1.SEED_INTERVAL[0] == D1.N_POSITIONS
    every = (REF.ACCOUNTED_SEED_INTERVALS + REF.EXPOSED_SEED_INTERVALS
             + REF.RETIRED_SEED_INTERVALS + REF.TEST_ONLY_SEED_INTERVALS)
    assert every, "no registry loaded -- the disjointness assertion would be vacuous"
    for s in range(*D1.SEED_INTERVAL):
        assert not any(lo <= s < hi for lo, hi in every), f"{s} is registered"


# ------------------------------------------------------------------ the gate

def test_the_d1_gate_is_false_as_published():
    assert D1.D1_EXECUTION_AUTHORIZED is False


def test_the_d1_gate_never_reads_another_experiments_gate():
    """One gate must never be openable by opening another. AST, not grep: the
    module docstring and comments NAME the other two gates."""
    import ast
    import pathlib
    tree = ast.parse(pathlib.Path(D1.__file__).read_text(encoding="utf-8"))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert names, "no identifiers parsed -- the absence checks below would be vacuous"
    assert "D1_EXECUTION_AUTHORIZED" in names, \
        "D1 does not read its OWN gate; the absence of the others proves nothing"
    for other in ("L0_EXECUTION_AUTHORIZED", "SCREEN_AUTHORIZED"):
        assert other not in names, f"D1 reads {other}"


def test_the_default_compile_step_refuses_while_the_gate_is_shut(tmp_path, registered):
    with pytest.raises(D1.D1Error, match="unauthorized"):
        D1._run_d1_unguarded(positions=[], paths=RUNTIME, out_path=str(tmp_path / "r.json"))


def test_a_valid_cli_invocation_refuses_in_a_fresh_subprocess(tmp_path):
    """The CLI is qualified as a FRESH SUBPROCESS with a valid invocation."""
    r = subprocess.run(
        [sys.executable, "-m", "scripts.GPU.alphazero.d1_probe",
         "--out", str(tmp_path / "r.json")],
        capture_output=True, text=True, cwd=".", timeout=60)
    assert r.returncode == D1.EXIT_UNAUTHORIZED == 5, (r.returncode, r.stdout, r.stderr)
    assert not (tmp_path / "r.json").exists()


@pytest.mark.parametrize("env", ["D1_EXECUTION_AUTHORIZED", "D1_AUTHORIZED", "AUTHORIZED"])
def test_no_environment_variable_opens_the_gate(tmp_path, env):
    import os as _os
    r = subprocess.run(
        [sys.executable, "-m", "scripts.GPU.alphazero.d1_probe",
         "--out", str(tmp_path / f"{env}.json")],
        capture_output=True, text=True, cwd=".", timeout=60,
        env={**_os.environ, env: "1", "PYTHONDONTWRITEBYTECODE": "1"})
    assert r.returncode == 5, (env, r.returncode, r.stderr)


def test_a_deadline_that_expires_only_at_the_write_step_still_voids(tmp_path, monkeypatch):
    """Reaches the FINAL pre-write deadline check specifically.

    The other breach test trips a check inside the position loop, so it passes
    even with the pre-write check deleted -- an injected-defect control proved
    exactly that. With no positions, the loop cannot fire, so only the last check
    can catch a deadline that expires between compilation and writing.
    """
    from scripts.GPU.alphazero import e4_screen_reference as _REF
    monkeypatch.setattr(_REF, "ACCOUNTED_SEED_INTERVALS",
                        _REF.ACCOUNTED_SEED_INTERVALS + (D1.SEED_INTERVAL,))
    ticks = iter([0.0, 1.0, 99999.0, 99999.0])
    out = tmp_path / "r.json"
    with pytest.raises(D1.D1VoidError, match="deadline exceeded"):
        D1._run_d1_unguarded(positions=[], paths=RUNTIME, out_path=str(out),
                  _compile=lambda d: None, _incumbent=lambda **kw: {},
                  deadline=D1.Deadline(clock=lambda: next(ticks)))
    assert not out.exists()


# ═══════════════════ review round 2: four guards that did not bind ═══════════

LEGAL_BITS_OK = "1" * 576
def _dump(moveNr, hist_pts):
    hist = " ".join(f"{x},{y}" for x, y in hist_pts)
    return (f"PLY {moveNr} moveNr={moveNr} next=Y termY=false termX=false\n"
            f"  PEGS 12,12,Y\n  BRIDGES \n  HIST {hist}\n  LEGAL {LEGAL_BITS_OK}\n")

def _stdout(depth=6, moveNr=6, completed=True, legal=True, sentinel=False,
            completed_depth=None, dump=True, hist=None):
    cd = depth if completed_depth is None else completed_depth
    line = (f"QUERY q=1 requested_depth={depth} move_x=11 move_y=12 to_move=Y "
            f"usealphabeta=true currentMaxPly={depth} completed_depth={cd} "
            f"completed={'true' if completed else 'false'} legal={'true' if legal else 'false'} "
            f"null_sentinel={'true' if sentinel else 'false'} moveNr={moveNr} "
            f"eval_regime=fixed elapsed_us=1000\n")
    h = hist if hist is not None else [(i + 1, i + 1) for i in range(moveNr)]
    return line + (_dump(moveNr, h) if dump else "")


@pytest.fixture
def reply(monkeypatch):
    """Drive BOTH invocations from one stdout template."""
    box = {"out": _stdout(), "calls": []}
    def fake_run(args, **kw):
        box["calls"].append({"args": args, "kw": kw})
        return subprocess.CompletedProcess(args, 0, box["out"], "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    return box


def _probe(**kw):
    return D1._probe_position(moves=MOVES, depth=6, paths=RUNTIME,
                             budget=D1.QueryBudget(D1.QUERY_CAP),
                             deadline=D1.Deadline(), **kw)


# ---- defect 1: run_d1 was ungated -------------------------------------------

def test_run_d1_refuses_directly_while_the_gate_is_shut_and_makes_no_calls(tmp_path, spy):
    """The CLI was gated; the PUBLIC RUNNER was not. A direct Python caller
    bypassed the gate entirely."""
    compiled = []
    out = tmp_path / "r.json"
    with pytest.raises(D1.D1Error, match="UNAUTHORIZED|unauthorized"):
        D1.run_d1(positions=[{"task_id": "t", "ply": len(MOVES), "prefix": MOVES,
                              "seed": D1.SEED_INTERVAL[0], "digest": MOVES_DIGEST}],
                  paths=RUNTIME, out_path=str(out),
                  _compile=lambda d: compiled.append(1))
    assert compiled == [], "compilation ran despite the shut gate"
    assert spy == [], "a T1j call was made despite the shut gate"
    assert not out.exists()


def test_the_gate_is_read_as_a_guard_not_merely_assigned():
    """The old AST test counted the ASSIGNMENT as a name, so it passed even with
    every guard read removed. Only a Load-context reference is a read."""
    import ast, pathlib
    tree = ast.parse(pathlib.Path(D1.__file__).read_text(encoding="utf-8"))
    loads = [n for n in ast.walk(tree) if isinstance(n, ast.Name)
             and n.id == "D1_EXECUTION_AUTHORIZED" and isinstance(n.ctx, ast.Load)]
    assert len(loads) >= 2, f"only {len(loads)} guard read(s); runner and CLI must each check"


def test_the_guard_read_check_fails_when_the_reads_are_stripped():
    """NEGATIVE CONTROL for the check above."""
    import ast, pathlib
    src = pathlib.Path(D1.__file__).read_text(encoding="utf-8")
    stripped = src.replace("if not D1_EXECUTION_AUTHORIZED:", "if False:")
    assert stripped != src
    loads = [n for n in ast.walk(ast.parse(stripped)) if isinstance(n, ast.Name)
             and n.id == "D1_EXECUTION_AUTHORIZED" and isinstance(n.ctx, ast.Load)]
    assert loads == [], "stripping the guards left a Load; the check cannot bind"


# ---- defect 2: identical INVALID replies were accepted -----------------------

@pytest.mark.parametrize("kw,why", [
    ({"completed": False}, "did not complete"),
    ({"legal": False}, "illegal"),
    ({"sentinel": True}, "null sentinel"),
    ({"completed_depth": 4}, "completed depth"),
])
def test_two_identical_invalid_replies_are_void(reply, kw, why):
    """Agreement is not validity. Two equally invalid answers agreed perfectly
    and were accepted -- 12.7 requires each reply to be valid on its own."""
    reply["out"] = _stdout(**kw)
    with pytest.raises(D1.D1VoidError, match=why):
        _probe()


def test_a_valid_pair_still_passes(reply):
    r = _probe()
    assert r["depth"] == 6 and r["invocations"] == 2


# ---- defect 3: two EMPTY dumps compared equal --------------------------------

def test_two_empty_dumps_are_void_not_equal(reply):
    """Both dumps empty compared equal and passed. Absence is not agreement."""
    reply["out"] = _stdout(dump=False)
    with pytest.raises(D1.D1VoidError, match="dump"):
        _probe()


def test_a_dump_whose_final_ply_disagrees_with_the_prefix_is_void(reply):
    reply["out"] = _stdout(moveNr=3)
    with pytest.raises(D1.D1VoidError, match="dump"):
        _probe()


# ---- defect 4: the deadline could not interrupt a hung stage -----------------

def test_a_blocking_stage_is_terminated_by_an_outer_supervisor(tmp_path, registered):
    """Cooperative checks run BETWEEN stages and cannot interrupt one that hangs.
    A stage that blocks past the deadline must still be terminated."""
    import time as _t
    out = tmp_path / "r.json"
    t0 = _t.monotonic()
    with pytest.raises(D1.D1VoidError, match="deadline"):
        D1._run_d1_unguarded(positions=[], paths=RUNTIME, out_path=str(out),
                  deadline=D1.Deadline(limit_s=0.3),
                  _compile=lambda d: _t.sleep(30))
    assert _t.monotonic() - t0 < 10, "the supervisor did not interrupt the blocking stage"
    assert not out.exists(), "a terminated run wrote a report"


#: What "executes" means for the structural gate test. EVERY effectful surface
#: D1 has, not just the T1j one: loading the incumbent checkpoint and running a
#: 400-simulation search are executions too, and the first version of this list
#: named only `A.query`, `compile_fn` and `_default_compile` -- so a public
#: entry point that loaded a model and searched would have passed it.
_EXECUTING_MARKERS = ("A.query(", "A.replay(", "compile_fn(", "_default_compile",
                      "self._load(", "self._build(", "load_reference_evaluator",
                      "_default_load_evaluator", "REF.build(")


def test_no_public_callable_can_execute_without_reading_the_gate():
    """The CLI gate protected nothing because run_d1 was public and ungated.
    The same hole exists one level down for any public function that queries
    T1j, compiles, or loads and searches with the incumbent. Enumerate them
    structurally rather than trusting a review.

    CLASSES ARE WALKED TOO. Scanning only module-level `FunctionDef` left a
    public class whose methods execute completely invisible to this check --
    a structural test with a hole in exactly the shape of the code about to be
    written is worse than no test, because it reads as coverage.
    """
    import ast, pathlib
    src = pathlib.Path(D1.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    offenders = []
    scanned = 0
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            candidates = [(node.name, node)]
        elif isinstance(node, ast.ClassDef):
            candidates = [(f"{node.name}.{m.name}", m) for m in node.body
                          if isinstance(m, ast.FunctionDef)]
        else:
            continue
        if node.name.startswith("_"):
            continue
        for label, fn in candidates:
            scanned += 1
            body = ast.get_source_segment(src, fn) or ""
            executes = any(m in body for m in _EXECUTING_MARKERS)
            reads_gate = any(isinstance(n, ast.Name) and n.id == "D1_EXECUTION_AUTHORIZED"
                             and isinstance(n.ctx, ast.Load) for n in ast.walk(fn))
            if executes and not reads_gate:
                offenders.append(label)
    assert scanned, "no public callable was scanned; this check would pass vacuously"
    assert not offenders, f"public and executing but ungated: {offenders}"


def test_the_structural_gate_check_notices_an_ungated_executing_entry_point(tmp_path):
    """NEGATIVE CONTROL for the check above, over a SYNTHETIC module.

    The real module is not edited. Without this the check could enumerate
    nothing, or use a marker list that matches nothing, and still pass.
    """
    import ast
    src = ("def run_it():\n"
           "    return REF.build(task, evaluator=ev)\n"
           "class Runner:\n"
           "    def go(self):\n"
           "        return A.query(m, depth=3)\n")
    tree = ast.parse(src)
    found = []
    for node in tree.body:
        if node.name.startswith("_"):
            continue
        fns = ([(node.name, node)] if isinstance(node, ast.FunctionDef)
               else [(f"{node.name}.{m.name}", m) for m in node.body
                     if isinstance(m, ast.FunctionDef)])
        for label, fn in fns:
            body = ast.get_source_segment(src, fn) or ""
            if any(m in body for m in _EXECUTING_MARKERS):
                found.append(label)
    assert found == ["run_it", "Runner.go"], found


# ═══════════ D1 INTEGRATION: registration, E3b binding, prefix identity ══════
#
# Still NO EXECUTION. `subprocess.run` is intercepted at the process boundary,
# the seed registries are read but NEVER written, and every test that needs the
# reserved block to look registered supplies a TEMPORARY FIXTURE registry --
# a monkeypatched tuple, never an edit to the real one.

from scripts.GPU.alphazero import d1_selection as SEL          # noqa: E402
from scripts.GPU.alphazero import e4_screen_reference as REF   # noqa: E402
from scripts.GPU.alphazero.e4_screen_runner import AbortError  # noqa: E402
from scripts.GPU.alphazero.game.twixt_state import TwixtState  # noqa: E402

CLEAN_POST = ("POSTCOND no_throw=true windows=0 frames=0 headless=true prefs_ok=true "
              "refl_ok=true refl_n={n} failures=0")
PREFIX = [(11, 11), (12, 13), (13, 12), (10, 13), (12, 10), (14, 14)]


def _state_after(moves):
    st = TwixtState(active_size=24, to_move="red")
    for mv in moves:
        st = st.apply_move(tuple(mv))
    return st


def _ply_block(state, moves):
    pegs, bridges = A.our_snapshot(state)
    legal = {A.to_t1j(r, c) for (r, c) in state.legal_moves()}
    bits = "".join("1" if (i // A.BOARD_N, i % A.BOARD_N) in legal else "0"
                   for i in range(A.LEGAL_BITS))
    hist = " ".join(f"{x},{y}" for x, y in (A.to_t1j(*m) for m in moves))
    return (f"PLY {state.ply} moveNr={state.ply} "
            f"next={A.PLAYER_TO_T1J[state.to_move]} "
            f"termY={'true' if state.winner() == 'red' else 'false'} "
            f"termX={'true' if state.winner() == 'black' else 'false'}\n"
            f"  PEGS {' '.join(sorted(pegs))}\n"
            f"  BRIDGES {' '.join(sorted(bridges))}\n"
            f"  HIST {hist}\n  LEGAL {bits}\n")


def _replay_stdout(prefix):
    """A faithful E3bDump replay transcript: one PLY block per ply, 0..len."""
    st, moves, out = _state_after([]), [], []
    for mv in list(prefix) + [None]:
        out.append(_ply_block(st, moves))
        if mv is None:
            break
        moves.append(tuple(mv))
        st = st.apply_move(tuple(mv))
    return "".join(out) + CLEAN_POST.format(n=D1.INT.REPLAY_REFL_N) + "\n"


def _position(prefix=PREFIX, seed=None, digest=None):
    st = _state_after(prefix)
    return {"task_id": "t", "ply": len(prefix), "prefix": list(prefix),
            "seed": D1.SEED_INTERVAL[0] if seed is None else seed,
            "digest": SEL.canonical_digest(st) if digest is None else digest}


@pytest.fixture
def registered(monkeypatch):
    """A TEMPORARY FIXTURE registry. The real tuple is never edited."""
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS",
                        REF.ACCOUNTED_SEED_INTERVALS + (D1.SEED_INTERVAL,))


@pytest.fixture
def wire(monkeypatch):
    """Serve replay transcripts and query replies from the process boundary."""
    box = {"calls": [], "prefix": PREFIX}

    def fake_run(args, **kw):
        box["calls"].append({"args": args, "kw": kw})
        if "replay" in args:
            return subprocess.CompletedProcess(args, 0, _replay_stdout(box["prefix"]), "")
        depth = int(args[args.index("query") + 1])
        return subprocess.CompletedProcess(
            args, 0, _stdout(depth=depth, moveNr=len(box["prefix"]),
                             hist=[A.to_t1j(*m) for m in box["prefix"]]), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return box


# ───────────────── 12.5: the block must be REGISTERED before a draw ──────────

def test_the_reserved_block_is_unregistered_so_d1_refuses():
    """The state of the repository RIGHT NOW: reserved on paper, in no registry."""
    with pytest.raises(D1.D1Error, match="not registered"):
        D1._check_seed_registration()


def test_a_registered_block_satisfies_the_check(registered):
    D1._check_seed_registration()


def test_a_PARTLY_registered_block_is_still_refused(monkeypatch):
    """NEGATIVE CONTROL. Registering all but the last seed must not pass: the
    check is over the whole block, not over its first element."""
    lo, hi = D1.SEED_INTERVAL
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS",
                        REF.ACCOUNTED_SEED_INTERVALS + ((lo, hi - 1),))
    with pytest.raises(D1.D1Error, match="not registered"):
        D1._check_seed_registration()


def test_the_check_reads_the_registry_and_never_writes_it(registered):
    before = REF.ACCOUNTED_SEED_INTERVALS
    D1._check_seed_registration()
    assert REF.ACCOUNTED_SEED_INTERVALS is before


def test_an_unregistered_block_stops_the_run_before_anything_is_compiled(tmp_path):
    compiled = []
    with pytest.raises(D1.D1Error, match="not registered"):
        D1._run_d1_unguarded(positions=[], paths=RUNTIME,
                             out_path=str(tmp_path / "r.json"),
                             _compile=lambda d: compiled.append(1),
                             _incumbent=lambda **kw: {})
    assert compiled == [], "the helper was compiled before the registration check"


# ─────────────────────── 5.5: the E3b binder, reused as-is ───────────────────

def test_every_retained_prefix_is_replayed_through_the_e3b_binder(wire, registered, tmp_path):
    D1._run_d1_unguarded(positions=[_position()], paths=RUNTIME,
                         out_path=str(tmp_path / "r.json"),
                         _compile=lambda d: None, _incumbent=lambda **kw: {"ok": True})
    replays = [c for c in wire["calls"] if "replay" in c["args"]]
    assert len(replays) == 1, [c["args"][-3:] for c in wire["calls"]]


def test_the_binder_call_carries_the_frozen_timeout_and_the_explicit_ply_cap(
        wire, registered, tmp_path):
    D1._run_d1_unguarded(positions=[_position()], paths=RUNTIME,
                         out_path=str(tmp_path / "r.json"),
                         _compile=lambda d: None, _incumbent=lambda **kw: {"ok": True})
    replays = [c for c in wire["calls"] if "replay" in c["args"]]
    assert replays, "no replay reached the boundary; the assertions below are vacuous"
    for c in replays:
        assert c["kw"].get("timeout") == D1.PER_QUERY_TIMEOUT_S == 120
        assert c["args"][c["args"].index("replay") + 1] == str(RUNTIME.ply_cap)


def test_a_binder_divergence_becomes_a_VOID_not_an_unexpected_error(
        monkeypatch, registered, tmp_path):
    """`make_binder` raises e4_screen_runner.AbortError, which is NOT a D1Error.
    Untranslated it escapes `main`'s handlers and exits 4 UNEXPECTED instead of
    3 VOID -- a fully understood refusal reported as a crash."""
    def diverging(args, **kw):
        st = _state_after(PREFIX[:-1] + [(20, 20)])          # same ply, other position
        blocks = _replay_stdout(PREFIX)
        return subprocess.CompletedProcess(
            args, 0, blocks.replace(_ply_block(_state_after(PREFIX), PREFIX),
                                    _ply_block(st, PREFIX)), "")

    monkeypatch.setattr(subprocess, "run", diverging)
    with pytest.raises(D1.D1VoidError, match="E3b"):
        D1._run_d1_unguarded(positions=[_position()], paths=RUNTIME,
                             out_path=str(tmp_path / "r.json"),
                             _compile=lambda d: None, _incumbent=lambda **kw: {})


def test_the_abort_translation_is_reachable_only_through_a_real_abort(registered):
    """The translated error must still name the phase the binder died in."""
    assert issubclass(D1.D1VoidError, D1.D1Error)
    assert not issubclass(AbortError, D1.D1Error)


# ────────────────── 12.7: the prefix must replay to its digest ───────────────

def test_a_prefix_that_does_not_replay_to_its_recorded_digest_voids(
        wire, registered, tmp_path):
    pos = _position(digest="0" * 64)
    with pytest.raises(D1.D1VoidError, match="digest"):
        D1._run_d1_unguarded(positions=[pos], paths=RUNTIME,
                             out_path=str(tmp_path / "r.json"),
                             _compile=lambda d: None, _incumbent=lambda **kw: {})


def test_an_illegal_move_in_a_retained_prefix_voids(wire, registered, tmp_path):
    pos = dict(_position(), prefix=PREFIX[:-1] + [PREFIX[0]])  # replays onto its own peg
    with pytest.raises(D1.D1VoidError, match="illegal"):
        D1._run_d1_unguarded(positions=[pos], paths=RUNTIME,
                             out_path=str(tmp_path / "r.json"),
                             _compile=lambda d: None, _incumbent=lambda **kw: {})


# ═══════════ ONE deadline, ONE origin: the enforced clock is the reported one ═
#
# The supervisor's SIGALRM was armed BEFORE `_check_seed_registration` and
# BEFORE `Deadline.start()`, so the enforced clock and the reported clock had
# different start points. Conservative, but not one coherent auditable deadline:
# the report's `elapsed_s` and the timer that can actually terminate the run were
# measuring from different instants, and a refused registration had already armed
# a 90-minute timer.

@pytest.fixture
def timer(monkeypatch):
    """Observe SIGALRM arming at the point it happens."""
    calls = []
    real = signal.setitimer
    monkeypatch.setattr(signal, "setitimer",
                        lambda which, value, *a: calls.append(value) or real(which, 0))
    return calls


def test_an_unregistered_block_arms_no_timer_at_all(tmp_path, timer):
    """The ordering, asserted at the effect. A block that is not registered must
    cost nothing -- not a compile, not a checkpoint read, and not an armed
    90-minute timer either."""
    d = D1.Deadline()
    with pytest.raises(D1.D1Error, match="not registered"):
        D1._run_d1_unguarded(positions=[], paths=RUNTIME, deadline=d,
                             out_path=str(tmp_path / "r.json"),
                             _compile=lambda x: None, _incumbent=lambda **kw: {})
    assert timer == [], "a timer was armed before the registration check refused"
    assert d.started is False, "the reported clock started before registration passed"


def test_the_supervisor_arms_from_the_started_deadlines_REMAINING_time(
        tmp_path, registered, timer):
    """Same clock, same origin. Arming from `limit_s` would restart the window,
    so the timer would fire later than the deadline the report describes."""
    ticks = iter([1000.0, 1005.0] + [1005.0] * 50)
    d = D1.Deadline(limit_s=100, clock=lambda: next(ticks))
    D1._run_d1_unguarded(positions=[], paths=RUNTIME, deadline=d,
                         out_path=str(tmp_path / "r.json"),
                         _compile=lambda x: None, _incumbent=lambda **kw: {})
    assert timer, "no timer was armed; the assertion below would be vacuous"
    assert timer[0] == 95.0, (
        f"armed with {timer[0]}, expected the started deadline's remaining 95.0 "
        f"(limit 100 minus 5 elapsed). 100.0 means it armed from limit_s and "
        f"restarted the window.")


def test_the_supervisor_refuses_a_deadline_that_was_never_started(timer):
    """Structural enforcement of the order: it cannot arm from a clock that has
    no origin, so 'start, then arm' cannot be silently reversed.

    MATCHED ON THE SUPERVISOR'S OWN WORDING. `Deadline.remaining()` also refuses
    an unstarted clock, with its own "deadline was never started", so a test
    matching that phrase passed with this guard deleted -- an injected-defect
    control caught exactly that. Two guards, one condition: the second needs a
    message only it can produce.
    """
    with pytest.raises(D1.D1Error, match="supervisor cannot arm.*different instant"):
        with D1._supervisor(D1.Deadline()):
            pass
    assert timer == []


def test_the_supervisor_refuses_when_no_time_remains(timer):
    """`setitimer(ITIMER_REAL, 0)` DISABLES the timer. Arming with a non-positive
    remaining would therefore switch the supervisor OFF while looking armed --
    the exact shape of a guard that does not bind."""
    ticks = iter([0.0, 20.0] + [20.0] * 10)
    d = D1.Deadline(limit_s=10, clock=lambda: next(ticks)).start()
    with pytest.raises(D1.D1VoidError, match="no time remain"):
        with D1._supervisor(d):
            pass
    assert timer == [], "a disabled timer was armed instead of refusing"


def test_the_reported_elapsed_and_the_enforced_timer_share_one_origin(
        tmp_path, registered, timer):
    ticks = iter([500.0, 500.0, 500.0] + [560.0] * 50)
    d = D1.Deadline(limit_s=90 * 60, clock=lambda: next(ticks))
    report = D1._run_d1_unguarded(positions=[], paths=RUNTIME, deadline=d,
                                  out_path=str(tmp_path / "r.json"),
                                  _compile=lambda x: None, _incumbent=lambda **kw: {})
    assert timer[0] == D1.RUN_DEADLINE_S, "the timer did not start at the deadline's origin"
    assert report["elapsed_s"] == 60.0
    assert report["run_deadline_s"] == D1.RUN_DEADLINE_S
