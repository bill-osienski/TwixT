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
import json
import pathlib
import signal
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import d1_probe as D1
from scripts.GPU.alphazero import e4_screen_integration as INT
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
        return subprocess.CompletedProcess(args, 0, _stdout(depth=depth), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


# ------------------------------------------- timeout arrival AT the boundary

def test_every_t1j_call_reaches_subprocess_run_with_the_frozen_timeout(spy):
    D1._probe_position(moves=MOVES, depth=6, paths=RUNTIME,
                      state=_state_after(MOVES), label=LABEL,
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
                      state=_state_after(MOVES), label=LABEL,
                      budget=D1.QueryBudget(D1.QUERY_CAP), deadline=D1.Deadline())
    assert spy
    assert any(c["kw"].get("timeout") is None for c in spy), \
        "the injected defect did not reach the boundary; the control proves nothing"


# ------------------------------- duplicate queries are separate repeats=1 JVMs

def test_each_depth_issues_two_separate_query_mode_invocations(spy):
    D1._probe_position(moves=MOVES, depth=6, paths=RUNTIME,
                      state=_state_after(MOVES), label=LABEL,
                      budget=D1.QueryBudget(D1.QUERY_CAP), deadline=D1.Deadline())
    assert len(spy) == D1.INVOCATIONS_PER_DEPTH == 2, [c["args"] for c in spy]
    for c in spy:
        assert "query" in c["args"], c["args"]


def test_the_same_jvm_determinism_mode_is_never_used(spy):
    """`repeats>1` reuses ONE process's Zobrist salt, so it cannot test the
    cross-process variable at all. The adapter puts the mode in argv, so the
    prohibition is observable at the boundary rather than asserted about a kwarg."""
    D1._probe_position(moves=MOVES, depth=6, paths=RUNTIME,
                      state=_state_after(MOVES), label=LABEL,
                      budget=D1.QueryBudget(D1.QUERY_CAP), deadline=D1.Deadline())
    assert spy
    for c in spy:
        assert "determinism" not in c["args"], c["args"]


def test_two_invocations_are_distinct_processes_not_one_repeated(spy):
    D1._probe_position(moves=MOVES, depth=6, paths=RUNTIME,
                      state=_state_after(MOVES), label=LABEL,
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

def test_the_query_cap_is_the_PROSPECTIVE_value_and_its_arithmetic_holds():
    """§13.3 lowered 12.4's ceiling by excluding six observed-failing rows."""
    assert D1.N_POSITIONS == 221 and D1.QUERY_CAP == 1105
    assert D1.QUERY_CAP == D1.N_POSITIONS * (1 + len(D1.T1J_DEPTHS) * D1.INVOCATIONS_PER_DEPTH)


def test_the_frozen_12_4_figures_are_kept_as_the_historical_record():
    """12.4's numbers are what the frozen rule yielded; §13 supersedes them
    prospectively without erasing them."""
    assert D1.N_POSITIONS_FROZEN_12 == 227 and D1.QUERY_CAP_FROZEN_12 == 1135
    assert D1.QUERY_CAP_FROZEN_12 == D1.N_POSITIONS_FROZEN_12 * 5


def test_the_ceiling_only_ever_moved_DOWN():
    """§13: 'lowered, never raised, and the difference may not be spent
    elsewhere.'"""
    assert D1.N_POSITIONS < D1.N_POSITIONS_FROZEN_12
    assert D1.QUERY_CAP < D1.QUERY_CAP_FROZEN_12
    assert D1.N_POSITIONS_FROZEN_12 - D1.N_POSITIONS == 6


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
                      state=_state_after(MOVES), label=LABEL,
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


def test_the_seed_interval_is_ACCOUNTED_and_now_RETIRED_after_the_VOID():
    """Registered 2026-08-28 for the execution authorization, then RETIRED the
    same day when the single authorized run VOIDED at 3m18s.

    Retired WHOLE, drawn and undrawn alike. At least one seed was drawn -- the
    VOID came from `_probe_position`, which runs after that position's incumbent
    readout -- and HOW MANY is undetermined, because a VOID writes no record and
    the message names the depth, not the position.

    NOT exposed: that list records seeds that WERE drawn, and marking all 227
    would claim 226 draws that may never have happened. Retiring is the claim
    the evidence supports -- these may not be used again -- and it is what makes
    `validate_task_executable` refuse them.
    """
    from scripts.GPU.alphazero import e4_screen_reference as REF
    assert D1.SEED_INTERVAL == (202614000, 202614227)
    # It sized the FROZEN §12 cohort, not §13's. §13 reserves no replacement, so
    # there is no interval matching 221 and a run cannot proceed without one.
    assert D1.SEED_INTERVAL[1] - D1.SEED_INTERVAL[0] == D1.N_POSITIONS_FROZEN_12
    assert D1.SEED_INTERVAL[1] - D1.SEED_INTERVAL[0] != D1.N_POSITIONS
    for name in ("ACCOUNTED_SEED_INTERVALS", "EXPOSED_SEED_INTERVALS",
                 "RETIRED_SEED_INTERVALS", "TEST_ONLY_SEED_INTERVALS"):
        assert getattr(REF, name), f"vacuous: {name} is empty"
    for seed in range(*D1.SEED_INTERVAL):
        st = REF.seed_status(seed)
        assert st["accounted"] and st["retired"], (seed, st)
        assert not st["exposed"] and not st["test_only"], (seed, st)
        assert REF.seed_is_unavailable(seed), seed


def test_the_retired_block_can_no_longer_be_SCHEDULED():
    """The consequence that matters: a spent one-shot block must be refused by
    the executable question, not merely annotated."""
    from scripts.GPU.alphazero import e4_screen_reference as REF
    task = {"seed": D1.SEED_INTERVAL[0], "reference": "calib020_0001",
            "reference_sha1": "209cf2d4fd24a48553d259dd71b4954867b9473e",
            "anchor_colour": "black"}
    REF.validate_task_structure(task)                 # still WELL FORMED, forever
    with pytest.raises(REF.E4ReferenceError, match="RETIRED"):
        REF.validate_task_executable(task)            # but no longer RUNNABLE


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


def test_the_default_compile_step_now_refuses_a_TOOLCHAIN_it_cannot_verify(
        tmp_path, registered):
    """`_default_compile` used to raise "unauthorized" unconditionally, which was
    the truthful thing while it was unwritten. It now compiles, so what it must
    refuse is a toolchain it cannot verify -- RUNTIME is a fabricated path, so
    resolution fails before anything is built. The GATE is a separate refusal and
    is asserted on `run_d1` and the CLI, where it belongs."""
    with pytest.raises((TC.ToolchainError, D1.D1Error)):
        D1._run_d1_unguarded(positions=[], paths=RUNTIME, out_path=str(tmp_path / "r.json"))
    assert not (tmp_path / "r.json").exists()


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

CLEAN_POST = ("POSTCOND no_throw=true windows=0 frames=0 headless=true prefs_ok=true "
              "refl_ok=true refl_n={n} failures=0")


def _state_after(moves):
    from scripts.GPU.alphazero.game.twixt_state import TwixtState as _TS
    st = _TS(active_size=24, to_move="red")
    for mv in moves:
        st = st.apply_move(tuple(mv))
    return st


def _ply_block(state, moves):
    """A dump that AGREES with `state`, in the helper's own vocabulary."""
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


def _dump(prefix):
    return _ply_block(_state_after(prefix), list(prefix))


def _stdout(depth=6, prefix=None, completed=True, legal=True, sentinel=False,
            completed_depth=None, dump=True, post=True, refl_n=None,
            dump_prefix=None, move=None, clean=True):
    """One faithful E4Preflight query reply, with knobs for each injected defect."""
    prefix = MOVES if prefix is None else list(prefix)
    cd = depth if completed_depth is None else completed_depth
    st = _state_after(prefix)
    mv = move if move is not None else sorted(st.legal_moves())[0]
    x, y = A.to_t1j(*mv)
    line = (f"QUERY q=1 requested_depth={depth} move_x={x} move_y={y} "
            f"to_move={A.PLAYER_TO_T1J[st.to_move]} "
            f"usealphabeta=true currentMaxPly={depth} completed_depth={cd} "
            f"completed={'true' if completed else 'false'} legal={'true' if legal else 'false'} "
            f"null_sentinel={'true' if sentinel else 'false'} moveNr={len(prefix)} "
            f"eval_regime=fixed elapsed_us=1000\n")
    body = _dump(dump_prefix if dump_prefix is not None else prefix) if dump else ""
    tail = ""
    if post:
        tail = CLEAN_POST.format(n=INT.QUERY_REFL_N if refl_n is None else refl_n)
        if not clean:
            tail = tail.replace("no_throw=true", "no_throw=false")
        tail += "\n"
    return line + body + tail


@pytest.fixture
def reply(monkeypatch):
    """Drive BOTH invocations from one stdout template."""
    box = {"out": _stdout(), "calls": [], "rc": 0}
    def fake_run(args, **kw):
        box["calls"].append({"args": args, "kw": kw})
        return subprocess.CompletedProcess(args, box["rc"], box["out"], "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    return box


def _probe(**kw):
    return D1._probe_position(moves=MOVES, depth=6, paths=RUNTIME,
                      state=_state_after(MOVES), label=LABEL,
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
    reply["out"] = _stdout(dump_prefix=MOVES[:3])
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

PREFIX = list(MOVES)


def _replay_stdout(prefix):
    """A faithful E3bDump replay transcript: one PLY block per ply, 0..len."""
    st, moves, out = _state_after([]), [], []
    for mv in list(prefix) + [None]:
        out.append(_ply_block(st, moves))
        if mv is None:
            break
        moves.append(tuple(mv))
        st = st.apply_move(tuple(mv))
    return "".join(out) + CLEAN_POST.format(n=INT.REPLAY_REFL_N) + "\n"


def _position(prefix=PREFIX, seed=None, digest=None):
    st = _state_after(prefix)
    return {"task_id": "t", "ply": len(prefix), "prefix": list(prefix),
            "seed": D1.SEED_INTERVAL[0] if seed is None else seed,
            "digest": SEL.canonical_digest(st) if digest is None else digest}


@pytest.fixture
def registered():
    """The block IS registered in the real tuple now (12.5, execution
    authorization), so this fixture no longer fakes anything. It is kept as the
    name the tests below read by, and asserts the fact rather than arranging it:
    a fixture that silently stopped arranging what its name claims would be the
    quietest way for these tests to go vacuous."""
    assert tuple(D1.SEED_INTERVAL) in {tuple(i) for i in REF.ACCOUNTED_SEED_INTERVALS}


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
            args, 0, _stdout(depth=depth, prefix=box["prefix"]), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return box


# ───────────────── 12.5: the block must be REGISTERED before a draw ──────────

def _registry_without_d1():
    """The ACCOUNTED tuple as it stood BEFORE the D1 execution authorization.

    The negative controls below need a registry that does not contain the block;
    now that it is really registered, they must remove it rather than pretend.
    """
    out = tuple(i for i in REF.ACCOUNTED_SEED_INTERVALS
                if tuple(i) != tuple(D1.SEED_INTERVAL))
    assert len(out) == len(REF.ACCOUNTED_SEED_INTERVALS) - 1, \
        "the D1 block is not in ACCOUNTED as a single interval; these controls are stale"
    return out


def test_the_registered_block_satisfies_the_check():
    """The state of the repository RIGHT NOW: registered, and D1 may proceed."""
    D1._check_seed_registration()


def test_an_unregistered_block_is_refused(monkeypatch):
    """NEGATIVE CONTROL. Strip the block and the check must refuse -- otherwise
    it would pass for any registry at all."""
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS", _registry_without_d1())
    with pytest.raises(D1.D1Error, match="not registered"):
        D1._check_seed_registration()


def test_a_PARTLY_registered_block_is_still_refused(monkeypatch):
    """NEGATIVE CONTROL. Registering all but the last seed must not pass: the
    check is over the whole block, not over its first element."""
    lo, hi = D1.SEED_INTERVAL
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS",
                        _registry_without_d1() + ((lo, hi - 1),))
    with pytest.raises(D1.D1Error, match="not registered"):
        D1._check_seed_registration()


def test_the_check_reads_the_registry_and_never_writes_it(registered):
    before = REF.ACCOUNTED_SEED_INTERVALS
    D1._check_seed_registration()
    assert REF.ACCOUNTED_SEED_INTERVALS is before


def test_an_unregistered_block_stops_the_run_before_anything_is_compiled(tmp_path,
                                                                          monkeypatch):
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS", _registry_without_d1())
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


def test_an_unregistered_block_arms_no_timer_at_all(tmp_path, timer, monkeypatch):
    """The ordering, asserted at the effect. A block that is not registered must
    cost nothing -- not a compile, not a checkpoint read, and not an armed
    90-minute timer either."""
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS", _registry_without_d1())
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


# ═══════════════ the real compile step: verified toolchain, create-only ══════
#
# `_default_compile` used to raise unconditionally. It now compiles, so every
# test below mocks BOTH the toolchain resolution and `compile_helper`: no javac
# is started here. The real compile is exercised by the run itself, once.

from scripts.GPU.alphazero import e4_screen_command as SCREEN_CMD  # noqa: E402
from scripts.GPU.alphazero import t1j_toolchain as TC              # noqa: E402


@pytest.fixture
def toolchain(monkeypatch, tmp_path):
    """A fake VERIFIED toolchain. `verified_paths` is the seam because it is the
    only thing that hashes before returning a path."""
    root = tmp_path / "tc"
    (root / "jdk" / "bin").mkdir(parents=True)
    (root / "t1j.jar").write_bytes(b"jar")
    for exe in ("java", "javac"):
        (root / "jdk" / "bin" / exe).write_text("#!/bin/sh\nexit 0\n")
    info = {"root": str(root), "source": "explicit", "jar": str(root / "t1j.jar"),
            "jdk_home": str(root / "jdk"), "verified": 5}
    monkeypatch.setattr(TC, "verified_paths", lambda *a, **k: dict(info))
    monkeypatch.setattr(SCREEN_CMD, "JAR_SHA256", D1.INT._sha256(info["jar"]))
    monkeypatch.setattr(D1.INT, "verify_jdk_identity",
                        lambda home, pinned=None: {"ok": home})
    return info


@pytest.fixture
def javac(monkeypatch):
    """Record compile_helper calls; never run javac."""
    calls = []

    def fake(javac_path, jar, out_dir, sources=None):
        calls.append({"javac": javac_path, "jar": jar, "out": out_dir,
                      "sources": tuple(sources or ())})
        pathlib_ = __import__("pathlib")
        pathlib_.Path(out_dir, "E4Preflight.class").write_bytes(b"\xca\xfe\xba\xbe")
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(A, "compile_helper", fake)
    return calls


def _paths_for(info, classes):
    return D1.T1jPaths(java=str(pathlib.Path(info["jdk_home"], "bin", "java")),
                       jar=info["jar"], classes=str(classes), ply_cap=280)


def test_the_compile_step_verifies_the_toolchain_before_compiling(toolchain, javac,
                                                                  tmp_path):
    out = D1._default_compile(D1.Deadline().start(),
                              paths=_paths_for(toolchain, tmp_path / "classes"))
    assert javac, "compile_helper was never called"
    assert out["toolchain"]["root"] == toolchain["root"]
    assert out["toolchain"]["source"] == "explicit"
    assert out["jar_sha256"] == SCREEN_CMD.JAR_SHA256
    assert out["classes"], "no compiled class was recorded"
    assert set(out["sources"]) == {p.name for p in A.PREFLIGHT_SOURCES}


def test_the_compile_step_builds_the_PREFLIGHT_sources_d1_actually_queries(
        toolchain, javac, tmp_path):
    """D1 queries through `E4Preflight`, not the bare E3b dump helper. Compiling
    the E3b set alone would leave the query main class absent and every query
    would fail at the process boundary."""
    D1._default_compile(D1.Deadline().start(),
                        paths=_paths_for(toolchain, tmp_path / "classes"))
    assert javac[0]["sources"] == tuple(A.PREFLIGHT_SOURCES)
    assert any(p.name == "E4Preflight.java" for p in javac[0]["sources"])


def test_a_jar_that_is_not_the_verified_one_is_refused(toolchain, javac, tmp_path):
    other = tmp_path / "other.jar"
    other.write_bytes(b"jar")
    paths = D1.T1jPaths(java=str(pathlib.Path(toolchain["jdk_home"], "bin", "java")),
                        jar=str(other), classes=str(tmp_path / "c"), ply_cap=280)
    with pytest.raises(D1.D1Error, match="jar"):
        D1._default_compile(D1.Deadline().start(), paths=paths)
    assert javac == [], "it compiled against an unverified jar"


def test_a_java_binary_outside_the_verified_jdk_is_refused(toolchain, javac, tmp_path):
    paths = D1.T1jPaths(java="/usr/bin/java", jar=toolchain["jar"],
                        classes=str(tmp_path / "c"), ply_cap=280)
    with pytest.raises(D1.D1Error, match="java"):
        D1._default_compile(D1.Deadline().start(), paths=paths)
    assert javac == []


def test_an_existing_class_directory_is_refused(toolchain, javac, tmp_path):
    classes = tmp_path / "classes"
    classes.mkdir()
    with pytest.raises(D1.D1Error, match="already exists"):
        D1._default_compile(D1.Deadline().start(), paths=_paths_for(toolchain, classes))
    assert javac == [], "it compiled into a directory it did not create"


def test_a_failing_javac_is_a_VOID(toolchain, monkeypatch, tmp_path):
    """The exit-status guard, REACHED ALONE.

    A javac that fails AND writes nothing is also caught by the no-.class guard
    further down, so a fixture that writes nothing proves only whichever fires
    first -- an injected-defect control showed the exit check could be deleted
    with no test noticing. This one writes a class file and still fails.
    """
    def failing(javac_path, jar, out_dir, sources=None):
        pathlib.Path(out_dir, "E4Preflight.class").write_bytes(b"\xca\xfe\xba\xbe")
        return subprocess.CompletedProcess([], 1, "", "boom")

    monkeypatch.setattr(A, "compile_helper", failing)
    with pytest.raises(D1.D1VoidError, match="javac exit 1"):
        D1._default_compile(D1.Deadline().start(),
                            paths=_paths_for(toolchain, tmp_path / "classes"))


def test_a_jar_that_disagrees_with_E4s_PIN_is_refused(toolchain, javac, monkeypatch,
                                                      tmp_path):
    """The acquisition lock says what was fetched; `JAR_SHA256` says what was
    QUALIFIED. Comparing them is the only thing that shows they agree -- and the
    earlier test cannot see this guard, because its fixture sets the pin FROM the
    fake jar, so the two match however the code behaves."""
    monkeypatch.setattr(SCREEN_CMD, "JAR_SHA256", "0" * 64)
    with pytest.raises(D1.D1Error, match="qualified"):
        D1._default_compile(D1.Deadline().start(),
                            paths=_paths_for(toolchain, tmp_path / "classes"))
    assert javac == [], "it compiled against a jar E4 never qualified"


def test_an_unverifiable_toolchain_stops_the_compile(monkeypatch, javac, tmp_path):
    monkeypatch.setattr(TC, "verified_paths",
                        lambda *a, **k: (_ for _ in ()).throw(TC.ToolchainError("nope")))
    paths = D1.T1jPaths(java="j", jar="j", classes=str(tmp_path / "c"), ply_cap=280)
    with pytest.raises(TC.ToolchainError):
        D1._default_compile(D1.Deadline().start(), paths=paths)
    assert javac == []


# ═══════ 5.3: the FULL T1j-side capture, and the checks that make it real ════
#
# Persisting a searched-position dump without comparing it to our state records
# an answer to a question nobody asked. The qualified `T1jAgent` re-binds it for
# exactly this reason: the per-prefix replay proves *a* jvm can rebuild the
# history and says nothing about the jvm that actually searched.

LABEL = "t@ply6 [sig/role] digest=abc"


def _probe(prefix=None, state=None, label=LABEL, **kw):
    prefix = MOVES if prefix is None else prefix
    return D1._probe_position(moves=prefix, depth=6, paths=RUNTIME, label=label,
                              state=state if state is not None else _state_after(prefix),
                              budget=D1.QueryBudget(), deadline=D1.Deadline(), **kw)


def test_the_reply_record_carries_every_5_3_observable(reply):
    r = _probe()
    assert r["requested_depth"] == 6 and r["completed_depth"] == 6
    assert r["completed"] is True and r["legal"] is True and r["null_sentinel"] is False
    assert r["move"] and len(r["move"]) == 2
    assert r["invocations"] == 2 and len(r["elapsed_us"]) == 2
    assert r["searched_state"]["ply"] == len(MOVES)
    assert r["searched_state"]["n_legal"] == len(_state_after(MOVES).legal_moves())
    assert r["postcond"] and all(p["refl_n"] == INT.QUERY_REFL_N for p in r["postcond"])


def test_the_postcondition_surface_is_read_on_every_invocation(reply):
    assert len(_probe()["postcond"]) == D1.INVOCATIONS_PER_DEPTH == 2


def test_a_reply_with_no_postcondition_line_is_VOID(reply):
    reply["out"] = _stdout(post=False)
    with pytest.raises(D1.D1VoidError, match="POSTCOND"):
        _probe()


def test_an_unclean_postcondition_is_VOID(reply):
    reply["out"] = _stdout(clean=False)
    with pytest.raises(D1.D1VoidError, match="postcondition"):
        _probe()


def test_a_wrong_reflection_COUNT_is_VOID(reply):
    """Authorized NAMES are not an authorized COUNT: a repeated or missing
    authorized access passes `PostCond.clean` and must still abort."""
    reply["out"] = _stdout(refl_n=INT.QUERY_REFL_N + 1)
    with pytest.raises(D1.D1VoidError, match="reflective"):
        _probe()


def test_a_SEARCH_jvm_that_rebuilt_a_different_position_is_VOID(reply):
    """The prefix replay binds one jvm; this binds the one that searched."""
    other = MOVES[:-1] + [(20, 20)]                 # same ply, different position
    reply["out"] = _stdout(dump_prefix=other)
    with pytest.raises(D1.D1VoidError, match="SEARCH jvm"):
        _probe()


def test_more_than_one_searched_dump_is_VOID(reply):
    reply["out"] = _stdout() + _dump(MOVES)
    with pytest.raises(D1.D1VoidError, match="dump"):
        _probe()


def test_a_move_illegal_in_OUR_engine_is_VOID(reply):
    """T1j may report `legal=true` about its own board; the move still has to be
    legal in ours, or the two engines are not describing one position."""
    reply["out"] = _stdout(move=MOVES[0])           # already occupied
    with pytest.raises(D1.D1VoidError, match="illegal in OUR"):
        _probe()


def _run_one(tmp_path, monkeypatch, *, moves_by_depth=None):
    """One position end to end, with each depth's reply controllable."""
    st = _state_after(MOVES)
    legal = sorted(st.legal_moves())

    def fake_run(args, **kw):
        if "replay" in args:
            return subprocess.CompletedProcess(args, 0, _replay_stdout(MOVES), "")
        depth = int(args[args.index("query") + 1])
        mv = (moves_by_depth or {}).get(depth, legal[0])
        return subprocess.CompletedProcess(args, 0, _stdout(depth=depth, move=mv), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return D1._run_d1_unguarded(
        positions=[_position()], paths=RUNTIME, out_path=str(tmp_path / "r.json"),
        _compile=lambda d: {"stub": True}, _incumbent=lambda **kw: {"ok": True})


def test_the_position_record_says_the_depths_AGREE_when_they_do(
        registered, tmp_path, monkeypatch):
    report = _run_one(tmp_path, monkeypatch)
    pos = report["positions"][0]
    assert pos["depths_agree"] is True, pos["depths"]
    assert [d["depth"] for d in pos["depths"]] == list(D1.T1J_DEPTHS)
    assert report["toolchain_identity"] == {"stub": True}


def test_the_position_record_says_the_depths_DISAGREE_when_they_do(
        registered, tmp_path, monkeypatch):
    """The discriminating half. Asserting only the agreeing case passes for a
    field hard-coded to True -- an injected-defect control proved exactly that.
    T1j's two qualified depths choosing different moves is a RESULT, not an
    abort: 12.7's determinism check compares the two JVMs at ONE depth."""
    st = _state_after(MOVES)
    a, b = sorted(st.legal_moves())[:2]
    report = _run_one(tmp_path, monkeypatch, moves_by_depth={3: a, 6: b})
    pos = report["positions"][0]
    assert pos["depths_agree"] is False, pos["depths"]
    assert {tuple(d["move"]) for d in pos["depths"]} == {a, b}


# ═════ the repair the VOID demanded: say WHAT failed, and WHERE ══════════════
#
# The single authorized D1 run aborted with
#   "VOID: depth 3 invocation 0: exit 3 with 1 query records"
# and that sentence is the whole reason the failure is still unexplained. Exit 3
# is E4Preflight's `System.exit(failures == 0 ? 0 : 3)`; it had PRINTED a `FAIL`
# line naming the check that failed, and `_probe_position` threw the stdout
# away. It also named the depth and invocation but never the position, so the
# abort could not be located among 227 -- which is why the seed accounting could
# not be closed either.
#
# MOCKED ONLY. No JVM, no model, no seed, no retry.

FAILING_OUT = (
    "PROC pid=1 java_version=17 vm=x headless=true prefs_factory=e2probe\n"
    + _stdout()
    + "FAIL q1: requested depth 3 completed\n"
)


def _pos_ref(**kw):
    base = {"task_id": "l0match-000-strong6-o1_center-t1j_red-r0", "ply": len(MOVES),
            "signature": "mover_fragmentation", "role": "position",
            "digest": "a" * 64, "seed": D1.SEED_INTERVAL[0], "prefix": MOVES}
    base.update(kw)
    return base


def test_a_nonzero_exit_carries_the_helpers_OWN_failure_lines(reply):
    """The instrument said what was wrong; the probe must not drop it."""
    reply["out"] = FAILING_OUT
    reply["rc"] = 3
    with pytest.raises(D1.D1VoidError) as e:
        _probe(label=D1.position_label(_pos_ref()))
    assert "FAIL q1: requested depth 3 completed" in str(e.value), str(e.value)


def test_the_failure_excerpt_never_carries_the_legal_cell_map(reply):
    """A dump carries a 576-character legal-cell map per ply; it must not reach
    the refusal at all."""
    reply["out"] = FAILING_OUT
    reply["rc"] = 3
    with pytest.raises(D1.D1VoidError) as e:
        _probe(label=D1.position_label(_pos_ref()))
    assert "1" * 100 not in str(e.value), "the legal-cell map leaked into the refusal"


def test_the_CHARACTER_cap_truncates_one_enormous_failure_line():
    """The character cap, REACHED ALONE. A previous version of this test used
    many SHORT lines, so the LINE cap bounded the message first and the
    character cap could be raised to 100,000 with nothing noticing."""
    out = "FAIL " + "x" * 5000 + "\n"
    excerpt = A.helper_failure_excerpt(out)
    assert len(excerpt) <= A.FAILURE_EXCERPT_CHARS + 3, len(excerpt)
    assert excerpt.endswith("...")


def test_the_LINE_cap_drops_the_tail_of_a_long_verdict_list():
    """The line cap, REACHED ALONE: short lines, so the character cap is never
    the thing doing the bounding."""
    out = "".join(f"FAIL check {i}\n" for i in range(40))
    excerpt = A.helper_failure_excerpt(out)
    assert "FAIL check 0" in excerpt
    assert f"FAIL check {A.FAILURE_EXCERPT_LINES}" not in excerpt
    assert excerpt.count("FAIL check") == A.FAILURE_EXCERPT_LINES == 12


def test_a_THREW_line_is_carried_because_it_is_a_verdict(reply):
    reply["out"] = _stdout() + "THREW: java.lang.IllegalStateException: boom\n"
    reply["rc"] = 3
    with pytest.raises(D1.D1VoidError, match="IllegalStateException"):
        _probe(label=D1.position_label(_pos_ref()))


def test_output_with_NO_verdict_line_falls_back_to_the_tail_not_silence():
    """The fallback, REACHED ALONE. `THREW` is itself a verdict prefix, so a
    transcript containing one never exercises this branch -- which is how the
    fallback could be deleted with every test still green."""
    out = ("PROC pid=1 java_version=17 vm=x headless=true prefs_factory=e2probe\n"
           "PLY 6 moveNr=6 next=Y termY=false termX=false\n"
           "  PEGS 12,12,Y\n  LEGAL " + "1" * 576 + "\n")
    excerpt = A.helper_failure_excerpt(out)
    assert "PROC pid=1" in excerpt, excerpt
    assert "1" * 100 not in excerpt, "the dump body leaked through the fallback"


def test_completely_empty_output_says_so_rather_than_nothing():
    assert A.helper_failure_excerpt("") == "(the helper produced no readable output)"


@pytest.mark.parametrize("kw,pattern", [
    ({}, "exit 3"),
    ({"completed": False}, "did not complete"),
    ({"legal": False}, "illegal move"),
    ({"post": False}, "POSTCOND"),
])
def test_every_probe_refusal_identifies_the_position(reply, kw, pattern):
    """task, ply, cohort and prefix digest, on EVERY refusal path -- not just the
    one the real VOID happened to take."""
    reply["out"] = _stdout(**kw)
    reply["rc"] = 3 if not kw else 0
    with pytest.raises(D1.D1VoidError) as e:
        _probe(label=D1.position_label(_pos_ref()))
    msg = str(e.value)
    assert pattern in msg, msg
    assert "l0match-000-strong6-o1_center-t1j_red-r0" in msg
    assert "ply6" in msg and "mover_fragmentation/position" in msg
    assert "a" * 16 in msg, "the prefix digest is absent"


def test_the_label_names_all_four_fields():
    label = D1.position_label(_pos_ref())
    assert "l0match-000-strong6-o1_center-t1j_red-r0" in label
    assert "ply6" in label and "mover_fragmentation/position" in label
    assert "digest=" + "a" * 16 in label


def test_a_position_missing_its_labels_still_produces_a_usable_label():
    """A manifest row without cohort labels must not make the label crash --
    a refusal that raises while reporting a refusal reports nothing."""
    label = D1.position_label({"task_id": "t", "ply": 3})
    assert "t" in label and "ply3" in label and "?" in label


def test_run_stage_refusals_identify_the_position_too(wire, registered, tmp_path):
    """The same label on the digest check, so every VOID is locatable."""
    pos = dict(_position(), digest="0" * 64, signature="created_threat", role="control")
    with pytest.raises(D1.D1VoidError) as e:
        D1._run_d1_unguarded(positions=[pos], paths=RUNTIME,
                             out_path=str(tmp_path / "r.json"),
                             _compile=lambda d: None, _incumbent=lambda **kw: {})
    assert "created_threat/control" in str(e.value), str(e.value)


# ══════ the parse path the low-ply fix closed, still open in D1 ═════════════
#
# `A.query`/`A.replay` raise `HelperOutputError` (a ValueError) when the helper's
# output cannot be parsed, and `parse_postconds` does too. D1 translated none of
# them: they escaped `main`, which has no catch-all, as an uncaught traceback
# with no verdict. Recorded as unfixed when the low-ply runner was built; closed
# here. Mocked throughout -- no JVM.

BAD_QUERY_LINE = "QUERY q=1 requested_depth=3 move_x=11\n"
BAD_POSTCOND = "POSTCOND no_throw=true windows=0\n"


def test_a_malformed_QUERY_line_is_a_VOID_naming_the_position(reply):
    reply["out"] = BAD_QUERY_LINE
    with pytest.raises(D1.D1VoidError, match="could not be parsed") as e:
        _probe(label=LABEL)
    assert LABEL in str(e.value), str(e.value)


def test_a_malformed_POSTCOND_in_a_QUERY_reply_is_a_VOID(reply):
    reply["out"] = _stdout(post=False) + BAD_POSTCOND
    with pytest.raises(D1.D1VoidError, match="could not be parsed"):
        _probe(label=LABEL)


def test_a_malformed_REPLAY_dump_is_a_VOID_and_writes_nothing(
        registered, monkeypatch, tmp_path):
    """The binder's own parse path, reached alone."""
    def fake_run(args, **kw):
        if "replay" in args:
            return subprocess.CompletedProcess(
                args, 0, "PLY 3 next=Y termY=false termX=false\n  LEGAL 0101\n", "")
        return subprocess.CompletedProcess(args, 0, _stdout(), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = tmp_path / "r.json"
    with pytest.raises(D1.D1VoidError, match="could not be parsed"):
        D1._run_d1_unguarded(positions=[_position()], paths=RUNTIME,
                             out_path=str(out), _compile=lambda d: None,
                             _incumbent=lambda **kw: {})
    assert not out.exists()


def test_the_parse_VOID_carries_the_bounded_transcript(reply):
    reply["out"] = BAD_QUERY_LINE + "  LEGAL " + "1" * 576 + "\n"
    with pytest.raises(D1.D1VoidError) as e:
        _probe(label=LABEL)
    assert "1" * 100 not in str(e.value), "the legal-cell map leaked into the refusal"
    assert len(str(e.value)) < 2000


def test_d1_main_reports_rather_than_escaping_on_an_unexpected_error(
        monkeypatch, tmp_path):
    """`main` had no catch-all, so anything it did not name escaped as a
    traceback with no verdict at all."""
    monkeypatch.setattr(D1, "D1_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(D1, "run_d1",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    rc = D1.main(["--out", str(tmp_path / "r.json"), "--positions", "/nonexistent",
                  "--ply-cap", "280"])
    assert rc == D1.EXIT_UNEXPECTED == 4
    assert not (tmp_path / "r.json").exists()


# ═══════════ the VOID trace: predeclared, and NON-ANALYTIC by construction ═══
#
# D1's VOID wrote nothing, so nobody could say how far it got -- which is why its
# seed accounting could not be closed and had to retire 227 seeds of which an
# unknown number were drawn. A trace fixes that. It must NOT become a way to
# publish a partial cohort, so it carries ONLY counters and identity, enforced by
# a predeclared allowlist rather than by intention.

def test_the_trace_field_allowlist_is_predeclared_and_small():
    assert D1.TRACE_FIELDS == frozenset({
        "event", "ts", "schema", "index", "n_positions", "task_id", "ply",
        "digest", "stage", "positions_completed", "queries_spent",
        "seeds_drawn", "verdict"})


@pytest.mark.parametrize("analytic", [
    "move", "moves", "policy", "raw_policy", "root_value", "value", "visits",
    "counts", "root_visits", "q_value", "top2", "depths", "record", "incumbent",
])
def test_the_trace_refuses_every_analytic_field(tmp_path, analytic):
    """The non-analytic guarantee is STRUCTURAL. A trace that could carry a move
    or a value would be a partial-cohort analysis wearing a different filename."""
    with open(tmp_path / "t.jsonl", "w", encoding="utf-8") as fh:
        with pytest.raises(D1.D1Error, match="not a permitted trace field"):
            D1._trace(fh, event="position_done", **{analytic: "anything"})


def test_the_trace_writes_only_allowlisted_keys(wire, registered, tmp_path):
    out = tmp_path / "r.json"
    D1._run_d1_unguarded(positions=[_position()], paths=RUNTIME, out_path=str(out),
                         _compile=lambda d: None, _incumbent=lambda **kw: {"ok": 1})
    lines = [json.loads(l) for l in open(str(out) + ".trace.jsonl", encoding="utf-8")]
    assert lines, "no trace was written"
    for row in lines:
        assert set(row) <= D1.TRACE_FIELDS, set(row) - D1.TRACE_FIELDS


def test_the_trace_SURVIVES_a_VOID_and_says_how_far_it_got(
        registered, monkeypatch, tmp_path):
    """THE POINT. D1's VOID left no trace and its seed accounting could not be
    closed. Here the record is still absent -- no partial cohort -- but the trace
    names the position it stopped on and the stage it reached."""
    def fake_run(args, **kw):
        if "replay" in args:
            return subprocess.CompletedProcess(args, 0, _replay_stdout(PREFIX), "")
        raise subprocess.TimeoutExpired(args, kw.get("timeout"))

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = tmp_path / "r.json"
    with pytest.raises(D1.D1VoidError):
        D1._run_d1_unguarded(positions=[_position()], paths=RUNTIME,
                             out_path=str(out), _compile=lambda d: None,
                             _incumbent=lambda **kw: {"ok": 1})
    assert not out.exists(), "a VOID wrote a record"
    lines = [json.loads(l) for l in open(str(out) + ".trace.jsonl", encoding="utf-8")]
    events = [r["event"] for r in lines]
    assert "position_start" in events and "run_end" in events
    assert "position_done" not in events, "it did not finish that position"
    end = [r for r in lines if r["event"] == "run_end"][0]
    assert end["verdict"] == "VOID"
    assert end["positions_completed"] == 0
    assert end["seeds_drawn"] == 1, "one seed was drawn; the accounting must say so"
    stages = [r["stage"] for r in lines if r["event"] == "position_stage"]
    # The stub incumbent succeeded, so the run reached the FIRST T1j query before
    # the timeout. The trace pins the exact stage -- which is the whole point,
    # and is more than "somewhere in position 0".
    assert stages == ["bound", "incumbent", "depth3"], stages


def test_the_trace_counts_seeds_drawn_so_the_accounting_can_close(
        wire, registered, tmp_path):
    out = tmp_path / "r.json"
    D1._run_d1_unguarded(positions=[_position(), dict(_position(), task_id="t2")],
                         paths=RUNTIME, out_path=str(out),
                         _compile=lambda d: None, _incumbent=lambda **kw: {"ok": 1})
    lines = [json.loads(l) for l in open(str(out) + ".trace.jsonl", encoding="utf-8")]
    end = [r for r in lines if r["event"] == "run_end"][0]
    assert end["verdict"] == "OK"
    assert end["positions_completed"] == 2 and end["seeds_drawn"] == 2


def test_the_trace_is_create_only(wire, registered, tmp_path):
    out = tmp_path / "r.json"
    trace = tmp_path / "r.json.trace.jsonl"
    trace.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        D1._run_d1_unguarded(positions=[_position()], paths=RUNTIME,
                             out_path=str(out), _compile=lambda d: None,
                             _incumbent=lambda **kw: {})
    assert trace.read_text(encoding="utf-8") == "{}\n"


def test_the_trace_is_not_the_record_and_says_so(wire, registered, tmp_path):
    out = tmp_path / "r.json"
    D1._run_d1_unguarded(positions=[_position()], paths=RUNTIME, out_path=str(out),
                         _compile=lambda d: None, _incumbent=lambda **kw: {"ok": 1})
    first = json.loads(open(str(out) + ".trace.jsonl", encoding="utf-8").readline())
    assert first["event"] == "run_start" and first["schema"] == D1.TRACE_SCHEMA
    assert "void-trace" in D1.TRACE_SCHEMA


def test_every_trace_line_is_fsynced(tmp_path, monkeypatch):
    """Durability is why the trace exists: one that is lost when the run is
    terminated mid-stage answers nothing. It is not observable after the fact --
    a closed file looks identical either way -- so the call itself is observed.
    """
    import os as _os
    calls = []
    real = _os.fsync
    monkeypatch.setattr(_os, "fsync", lambda fd: calls.append(fd) or real(fd))
    with open(tmp_path / "t.jsonl", "w", encoding="utf-8") as fh:
        D1._trace(fh, event="run_start", schema=D1.TRACE_SCHEMA, n_positions=3)
        D1._trace(fh, event="position_start", index=0)
        assert len(calls) == 2, calls
