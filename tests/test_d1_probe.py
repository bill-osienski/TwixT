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
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import d1_probe as D1
from scripts.GPU.alphazero import t1j_adapter as A

QUERY_LINE = (
    "QUERY q=1 requested_depth={d} move_x=11 move_y=12 to_move=Y usealphabeta=true "
    "currentMaxPly={d} completed_depth={d} completed=true legal=true null_sentinel=false "
    "moveNr=7 eval_regime=fixed elapsed_us=1000"
)
MOVES = [(11, 11), (12, 13), (13, 12), (10, 13), (12, 10), (14, 14)]
RUNTIME = D1.T1jPaths(java="/nonexistent/java", jar="/nonexistent/t1j.jar",
                      classes="/nonexistent/classes")


@pytest.fixture
def spy(monkeypatch):
    """Intercept the PROCESS BOUNDARY. Nothing is ever spawned."""
    calls = []

    def fake_run(args, **kw):
        calls.append({"args": args, "kw": kw})
        depth = args[args.index("query") + 1] if "query" in args else "6"
        return subprocess.CompletedProcess(args, 0, QUERY_LINE.format(d=depth), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


# ------------------------------------------- timeout arrival AT the boundary

def test_every_t1j_call_reaches_subprocess_run_with_the_frozen_timeout(spy):
    D1.probe_position(moves=MOVES, depth=6, paths=RUNTIME,
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
    D1.probe_position(moves=MOVES, depth=6, paths=RUNTIME,
                      budget=D1.QueryBudget(D1.QUERY_CAP), deadline=D1.Deadline())
    assert spy
    assert any(c["kw"].get("timeout") is None for c in spy), \
        "the injected defect did not reach the boundary; the control proves nothing"


# ------------------------------- duplicate queries are separate repeats=1 JVMs

def test_each_depth_issues_two_separate_query_mode_invocations(spy):
    D1.probe_position(moves=MOVES, depth=6, paths=RUNTIME,
                      budget=D1.QueryBudget(D1.QUERY_CAP), deadline=D1.Deadline())
    assert len(spy) == D1.INVOCATIONS_PER_DEPTH == 2, [c["args"] for c in spy]
    for c in spy:
        assert "query" in c["args"], c["args"]


def test_the_same_jvm_determinism_mode_is_never_used(spy):
    """`repeats>1` reuses ONE process's Zobrist salt, so it cannot test the
    cross-process variable at all. The adapter puts the mode in argv, so the
    prohibition is observable at the boundary rather than asserted about a kwarg."""
    D1.probe_position(moves=MOVES, depth=6, paths=RUNTIME,
                      budget=D1.QueryBudget(D1.QUERY_CAP), deadline=D1.Deadline())
    assert spy
    for c in spy:
        assert "determinism" not in c["args"], c["args"]


def test_two_invocations_are_distinct_processes_not_one_repeated(spy):
    D1.probe_position(moves=MOVES, depth=6, paths=RUNTIME,
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


def test_the_deadline_starts_before_helper_compilation(tmp_path):
    """A window opened after compilation cannot bound compilation."""
    seen = {}

    def spy_compile(deadline):
        seen["started"] = deadline.started
        seen["elapsed"] = deadline.elapsed()

    D1.run_d1(positions=[], paths=RUNTIME, out_path=str(tmp_path / "r.json"),
              _compile=spy_compile)
    assert seen["started"] is True, "compilation ran before the deadline started"
    assert seen["elapsed"] >= 0.0


# ------------------------------------------- VOID produces NO partial analysis

def test_a_forced_query_timeout_is_void_and_writes_no_report(tmp_path, monkeypatch):
    def boom(args, **kw):
        raise subprocess.TimeoutExpired(args, kw.get("timeout"))
    monkeypatch.setattr(subprocess, "run", boom)
    out = tmp_path / "r.json"
    with pytest.raises(D1.D1VoidError, match="timed out"):
        D1.run_d1(positions=[{"task_id": "t", "ply": len(MOVES), "prefix": MOVES,
                              "seed": D1.SEED_INTERVAL[0]}],
                  paths=RUNTIME, out_path=str(out), _compile=lambda d: None)
    assert not out.exists(), "a VOID run wrote a report -- that is partial analysis"


def test_a_forced_deadline_breach_is_void_and_writes_no_report(tmp_path, spy):
    ticks = iter([0.0, 0.0, 99999.0, 99999.0, 99999.0, 99999.0])
    out = tmp_path / "r.json"
    with pytest.raises(D1.D1VoidError, match="deadline exceeded"):
        D1.run_d1(positions=[{"task_id": "t", "ply": len(MOVES), "prefix": MOVES,
                              "seed": D1.SEED_INTERVAL[0]}],
                  paths=RUNTIME, out_path=str(out), _compile=lambda d: None,
                  deadline=D1.Deadline(clock=lambda: next(ticks)))
    assert not out.exists(), "a VOID run wrote a report -- that is partial analysis"


def test_void_is_raised_not_returned_so_a_caller_cannot_ignore_it(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired(a, 1)))
    with pytest.raises(D1.D1VoidError):
        D1.run_d1(positions=[{"task_id": "t", "ply": len(MOVES), "prefix": MOVES,
                              "seed": D1.SEED_INTERVAL[0]}],
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
        D1.probe_position(moves=MOVES, depth=6, paths=RUNTIME,
                          budget=b, deadline=D1.Deadline())
    assert len(spy) == 1, "the budget did not stop the second invocation"


def test_a_position_without_a_retained_prefix_is_void(tmp_path):
    with pytest.raises(D1.D1VoidError, match="no retained move prefix"):
        D1.run_d1(positions=[{"task_id": "t", "ply": 6, "seed": D1.SEED_INTERVAL[0]}],
                  paths=RUNTIME, out_path=str(tmp_path / "r.json"), _compile=lambda d: None)


def test_a_prefix_inconsistent_with_its_ply_is_void(tmp_path):
    """A digest cannot be replayed; a prefix of the wrong length replays the
    WRONG POSITION, which is worse than refusing."""
    with pytest.raises(D1.D1VoidError, match="different position"):
        D1.run_d1(positions=[{"task_id": "t", "ply": 99, "prefix": MOVES,
                              "seed": D1.SEED_INTERVAL[0]}],
                  paths=RUNTIME, out_path=str(tmp_path / "r.json"), _compile=lambda d: None)


@pytest.mark.parametrize("seed", [202613999, 202614227, 0, -1, True, "202614000", None])
def test_a_seed_outside_the_reserved_interval_is_void(tmp_path, seed):
    with pytest.raises(D1.D1VoidError, match="outside the reserved"):
        D1.run_d1(positions=[{"task_id": "t", "ply": len(MOVES), "prefix": MOVES,
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


def test_the_default_compile_step_refuses_while_the_gate_is_shut(tmp_path):
    with pytest.raises(D1.D1Error, match="unauthorized"):
        D1.run_d1(positions=[], paths=RUNTIME, out_path=str(tmp_path / "r.json"))


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


def test_a_deadline_that_expires_only_at_the_write_step_still_voids(tmp_path):
    """Reaches the FINAL pre-write deadline check specifically.

    The other breach test trips a check inside the position loop, so it passes
    even with the pre-write check deleted -- an injected-defect control proved
    exactly that. With no positions, the loop cannot fire, so only the last check
    can catch a deadline that expires between compilation and writing.
    """
    ticks = iter([0.0, 1.0, 99999.0, 99999.0])
    out = tmp_path / "r.json"
    with pytest.raises(D1.D1VoidError, match="deadline exceeded"):
        D1.run_d1(positions=[], paths=RUNTIME, out_path=str(out),
                  _compile=lambda d: None,
                  deadline=D1.Deadline(clock=lambda: next(ticks)))
    assert not out.exists()
