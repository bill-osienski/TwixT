"""H4 §4B -- the production adapter's H4 acceptance mode, and its gated runner.

NOTHING EXECUTES. No JVM is started and no gate opened in the source. Every
control goes through the REAL entry point -- `T1jAgent.__call__` on an agent
built by `make_agent_factory`, and the binder from `make_binder`, on one H4
runtime -- and is stubbed at the PROCESS BOUNDARY (`subprocess.run`), never at
the `_query` seam, so the adapter's own argv building and parsing run. A
permissive double hides the seam (card §6).

Frozen by `docs/superpowers/2026-09-22-t1j-h4-4b-acceptance-qualification-card.md`.
"""
import ast
import json
import pathlib
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import e4_screen_integration as INT
from scripts.GPU.alphazero import h4_4b_acceptance_qualification as B
from scripts.GPU.alphazero import h4_repair_qualification as Q
from scripts.GPU.alphazero import t1j_adapter as A
from scripts.GPU.alphazero.e4_screen_runner import AbortError, PHASE_MOVE
from tests.test_h4_repair_qualification import _state, query_out, replay_out

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATHS = Q.T1jPaths(java="/nonexistent/java", jar="/nonexistent/t1j.jar",
                   classes="/nonexistent/classes", ply_cap=280)

#: the o1_center positions of the frozen matrix, plies 0..5, plus a ply-6 one
P = {0: []}
P.update({r["ply"]: [tuple(m) for m in r["prefix"]] for r in Q.derive_matrix()
          if r["family"] == "o1_center"})
P[6] = P[5] + [sorted(_state(P[5]).legal_moves())[0]]
OTHER = {r["ply"]: [tuple(m) for m in r["prefix"]] for r in Q.derive_matrix()
         if r["family"] == "o3_low"}


def _post_line(out, old, new):
    """Edit the POSTCOND line only -- `headless=true` also appears on PROC."""
    return "\n".join(l.replace(old, new) if l.startswith("POSTCOND ") else l
                     for l in out.split("\n"))


@pytest.fixture
def wire(monkeypatch):
    """Serve the helper at the process boundary. Spawns nothing."""
    box = {"calls": [], "query": None, "replay": None, "rc_query": None,
           "rc_replay": 0, "timeout": False, "pid": 20000}

    def fake_run(args, **kw):
        box["calls"].append(list(args))
        if box["timeout"]:
            raise subprocess.TimeoutExpired(args, kw.get("timeout"))
        box["pid"] += 1
        pairs = [tuple(int(v) for v in a.split(",")) for a in args
                 if "," in a and a.replace(",", "").isdigit()]
        prefix = [A.to_ours(x, y) for (x, y) in pairs]
        if "replay" in args:
            out = (box["replay"](prefix) if box["replay"]
                   else replay_out(prefix, pid=box["pid"]))
            return subprocess.CompletedProcess(args, box["rc_replay"], out, "")
        native = len(prefix) <= 3
        out = (box["query"](prefix) if box["query"]
               else query_out(prefix, native=native, pid=box["pid"]))
        rc = box["rc_query"]
        if rc is None:
            rc = Q.NATIVE_EXIT if native else Q.SEARCHED_EXIT
        return subprocess.CompletedProcess(args, rc, out, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return box


def h4_adapter(ctx=None, runtime=None):
    ctx = ctx or INT.IntegrationContext()
    rt = runtime or INT.T1jRuntime(java="/j", jar="/x.jar", classes="/c", ply_cap=280,
                                   timeout_s=120, h4_acceptance=True)
    binder = INT.make_binder(rt, ctx)
    factory = INT.make_agent_factory(runtime=rt, ctx=ctx, evaluator=None,
                                     reference_build=B._no_reference, t1j_timeout_s=120)
    return ctx, rt, binder, factory


_TASKS = iter(range(10 ** 6))


def call_at(ply, *, prefix=None, adapter=None):
    """One H4 agent call at a position, through the real factory. Returns the move."""
    ctx, _rt, _b, factory = adapter or h4_adapter()
    moves = P[ply] if prefix is None else prefix
    st = _state(moves)
    task = f"t{next(_TASKS)}"
    ctx.reset(task, moves)
    return factory(B._task(task, st.to_move), st.to_move)(st), ctx


def records(ctx):
    return B.all_records(ctx)


SEARCHED = dict(native=False)


# ───────────────────────── 1. the gate ─────────────────────────

def test_the_gate_is_false_as_published():
    assert B.H4_4B_ACCEPTANCE_QUALIFICATION_AUTHORIZED is False


def test_the_public_runner_refuses_while_the_gate_is_shut(tmp_path, wire):
    with pytest.raises(B.H4BError, match="UNAUTHORIZED"):
        B.run_qualification(paths=PATHS, out_path=str(tmp_path / "r.json"))
    assert wire["calls"] == [] and not (tmp_path / "r.json").exists()


def test_the_cli_refuses_in_a_fresh_subprocess(tmp_path):
    out, classes = tmp_path / "r.json", tmp_path / "classes"
    r = subprocess.run(
        [sys.executable, "-m", "scripts.GPU.alphazero.h4_4b_acceptance_qualification",
         "--out", str(out), "--classes", str(classes)],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == B.EXIT_UNAUTHORIZED, r.stderr
    assert not out.exists() and not classes.exists()


def test_the_gate_is_read_at_both_public_entries_and_has_no_override():
    tree = ast.parse(pathlib.Path(B.__file__).read_text(encoding="utf-8"))
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for entry in ("run_qualification", "main"):
        names = {n.id for n in ast.walk(fns[entry]) if isinstance(n, ast.Name)}
        assert "H4_4B_ACCEPTANCE_QUALIFICATION_AUTHORIZED" in names, entry
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not {"environ", "getenv"} & attrs
    gates = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
             and n.id.endswith("_AUTHORIZED")}
    assert gates == {"H4_4B_ACCEPTANCE_QUALIFICATION_AUTHORIZED"}


# ─────────── 2. CLEAN BASELINES through the real entry point (§6.2) ───────────

@pytest.mark.parametrize("ply,source", [
    (0, Q.NATIVE_FIRST), (1, Q.NATIVE_SECOND_TO_FOURTH), (2, Q.NATIVE_SECOND_TO_FOURTH),
    (3, Q.NATIVE_SECOND_TO_FOURTH),
    # 🔴 CONSTRUCTED: never observed in this programme. It proves the production
    # path CAN accept that routine's reply, not that the routine ever answers.
    (4, Q.NATIVE_FIFTH_OR_MORE), (5, Q.NATIVE_FIFTH_OR_MORE)])
def test_baseline_a_NATIVE_reply_is_ACCEPTED_by_routine(wire, ply, source):
    """exit 3 + failures 1 + completed false -- the reply all THREE default
    refusal sites reject (card §2). Accepted here only if none of them survives."""
    wire["query"] = lambda p: query_out(p, native=True)
    wire["rc_query"] = Q.NATIVE_EXIT
    move, ctx = call_at(ply)
    assert move in set(_state(P[ply]).legal_moves())
    [r] = records(ctx)
    assert r["outcome"] == "accepted" and r["source"] == source
    assert "h4query" in wire["calls"][-1]


@pytest.mark.parametrize("ply", [3, 4, 5])
def test_baseline_a_SEARCHED_reply_is_ACCEPTED_and_the_rule_does_not_spread(wire, ply):
    wire["query"] = lambda p: query_out(p, **SEARCHED)
    wire["rc_query"] = Q.SEARCHED_EXIT
    move, ctx = call_at(ply)
    assert records(ctx)[0]["source"] == Q.SEARCHED


def test_baseline_an_H4_replay_with_one_PROC_binds_and_is_recorded(wire):
    ctx, _rt, binder, _f = h4_adapter()
    ctx.reset("r", P[3])
    binder({"task_id": "r"}, _state(P[3]), 3)
    [r] = records(ctx)
    assert r["role"] == "replay" and r["outcome"] == "accepted" and r["proc"]["pid"]


# ─────────────── 3. NEGATIVE CONTROLS -- each must REJECT (§6.1) ───────────────

def _refused(wire, ply, match, *, rc=None, prefix=None, **kw):
    wire["query"] = kw.pop("serve", None) or (lambda p: query_out(p, **kw))
    wire["rc_query"] = rc
    with pytest.raises(AbortError, match=match):
        call_at(ply, prefix=prefix)


def test_SEARCH_AT_PLY_ZERO_is_refused(wire):
    _refused(wire, 0, "ply 0: a completed search is impossible", rc=0, **SEARCHED)


def test_SEARCH_AT_PLY_ONE_is_refused(wire):
    _refused(wire, 1, "ply 1: a completed search is impossible", rc=0, **SEARCHED)


def test_SEARCH_AT_PLY_TWO_is_refused(wire):
    _refused(wire, 2, "ply 2: a completed search is impossible", rc=0, **SEARCHED)


def test_a_NATIVE_claim_at_PLY_6_is_refused(wire):
    """SYNTHETIC -- the frozen matrix ends at ply 5."""
    _refused(wire, 6, "impossible here", rc=3, native=True)


@pytest.mark.parametrize("rc,failures", [(0, 1), (3, 0), (0, 0), (3, 2)])
def test_a_native_signature_with_the_wrong_exit_or_failures_is_refused(wire, rc, failures):
    _refused(wire, 1, "permits ONLY", rc=rc, native=True, failures=failures)


@pytest.mark.parametrize("rc,failures", [(3, 0), (0, 1), (3, 1)])
def test_a_searched_signature_with_any_failure_is_refused(wire, rc, failures):
    _refused(wire, 5, "required to be", rc=rc, failures=failures, **SEARCHED)


def test_telemetry_outside_the_table_is_refused(wire):
    _refused(wire, 5, "outside the frozen table", rc=0, native=False,
             current_max_ply=3, completed_depth=2)


@pytest.mark.parametrize("kw,match", [
    (dict(null_sentinel=True), "unusable move"),
    (dict(legal=False), "unusable move")])
def test_an_unusable_move_is_refused(wire, kw, match):
    _refused(wire, 1, match, rc=3, native=True, **kw)


def test_a_move_illegal_in_OUR_engine_is_refused(wire):
    _refused(wire, 1, "illegal in OUR engine", rc=3, native=True, move=P[1][0])


def test_a_dump_of_a_DIFFERENT_board_is_refused(wire):
    _refused(wire, 3, "reconstructed a different position", rc=3,
             serve=lambda p: query_out(OTHER[3], native=True))


@pytest.mark.parametrize("dumps", [0, 2])
def test_zero_or_two_dumps_are_refused(wire, dumps):
    def serve(p):
        out = query_out(p, native=True, dump=dumps > 0)
        if dumps == 2:
            i = out.index("PLY ")
            j = out.index("POSTCOND ")
            out = out[:j] + out[i:j] + out[j:]
        return out
    _refused(wire, 1, "dumps, expected 1", rc=3, serve=serve)


@pytest.mark.parametrize("old,new,match", [
    ("QUERY q=1 ", "QUERY q=2 ", "query index"),
    ("moveNr=1 ", "moveNr=8 ", "moveNr=8"),
    ("requested_depth=6 ", "requested_depth=5 ", "requested depth 5")])
def test_an_incoherent_QUERY_line_is_refused(wire, old, new, match):
    _refused(wire, 1, match, rc=3,
             serve=lambda p: query_out(p, native=True).replace(old, new, 1))


def test_a_wrong_to_move_is_refused(wire):
    def serve(p):
        out = query_out(p, native=True)
        ours = A.PLAYER_TO_T1J[_state(p).to_move]
        return out.replace(f"to_move={ours}", f"to_move={'X' if ours == 'Y' else 'Y'}", 1)
    _refused(wire, 1, "to_move", rc=3, serve=serve)


@pytest.mark.parametrize("kw,match", [
    (dict(n_md=0), "0 MATCHDATA lines"), (dict(n_md=2), "2 MATCHDATA lines"),
    (dict(pie=True), "did not read back"), (dict(xs=23), "did not read back"),
    (dict(yst=False), "did not read back")])
def test_MATCHDATA_mismatches_are_refused(wire, kw, match):
    _refused(wire, 1, match, rc=3, native=True, **kw)


def test_MATCHDATA_identity_false_is_refused(wire):
    _refused(wire, 1, "identity=false", rc=3,
             serve=lambda p: query_out(p, native=True).replace("identity=true",
                                                               "identity=false"))


@pytest.mark.parametrize("refl", [3, 5])
def test_opt_in_reflection_drift_is_refused(wire, refl):
    _refused(wire, 1, f"refl_n={refl}, expected exactly 4", rc=3, native=True,
             refl_n=refl)


@pytest.mark.parametrize("n", [0, 2])
def test_PROC_counts_other_than_one_are_refused_on_the_QUERY_path(wire, n):
    _refused(wire, 1, f"{n} PROC lines", rc=3, native=True, n_proc=n)


@pytest.mark.parametrize("n", [0, 2])
def test_PROC_counts_other_than_one_are_refused_on_the_REPLAY_path(wire, n):
    wire["replay"] = lambda p: replay_out(p, n_proc=n)
    ctx, _rt, binder, _f = h4_adapter()
    ctx.reset("r", P[2])
    with pytest.raises(AbortError, match=f"{n} PROC lines"):
        binder({"task_id": "r"}, _state(P[2]), 2)


@pytest.mark.parametrize("serve,match", [
    (lambda p: query_out(p, native=True, no_throw=False), "the helper threw"),
    (lambda p: query_out(p, native=True, windows=1), "1 windows opened"),
    (lambda p: query_out(p, native=True, prefs_ok=False), "preferences"),
    (lambda p: _post_line(query_out(p, native=True), "headless=true", "headless=false"),
     "not headless"),
    (lambda p: _post_line(query_out(p, native=True), "refl_ok=true", "refl_ok=false"),
     "reflective-access")])
def test_a_DIRTY_safety_surface_is_refused(wire, serve, match):
    _refused(wire, 1, match, rc=3, serve=serve)


def test_H4_construction_refuses_a_depth_other_than_6():
    ctx, rt, _b, _f = h4_adapter()
    with pytest.raises(AbortError, match="depth 6 only"):
        INT.T1jAgent(runtime=rt, ctx=ctx, depth=5, colour="red", timeout_s=120)


def test_H4_construction_refuses_an_unbounded_QUERY_timeout():
    """🔴 The QUERY timeout (`t1j_timeout_s`), not `T1jRuntime.timeout_s` -- which
    already refuses None and is the REPLAY timeout."""
    ctx, rt, _b, _f = h4_adapter()
    factory = INT.make_agent_factory(runtime=rt, ctx=ctx, evaluator=None,
                                     reference_build=B._no_reference, t1j_timeout_s=None)
    with pytest.raises(AbortError, match="unbounded QUERY timeout"):
        factory(B._task("t", "red"), "red")


# ── pairing (§3.1.1): refused whenever EITHER side is H4, both directions ──

def _rt(h4):
    return INT.T1jRuntime(java="/j", jar="/x.jar", classes="/c", ply_cap=280,
                          timeout_s=120, h4_acceptance=h4)


def _factory(rt, ctx):
    return INT.make_agent_factory(runtime=rt, ctx=ctx, evaluator=None,
                                  reference_build=B._no_reference, t1j_timeout_s=120)


def test_PAIRING_an_H4_binder_with_a_DEFAULT_agent_is_refused():
    ctx = INT.IntegrationContext()
    INT.make_binder(_rt(True), ctx)
    with pytest.raises(AbortError, match="runtime mismatch"):
        _factory(_rt(False), ctx)


def test_PAIRING_a_DEFAULT_binder_with_an_H4_agent_is_refused():
    ctx = INT.IntegrationContext()
    INT.make_binder(_rt(False), ctx)
    with pytest.raises(AbortError, match="runtime mismatch"):
        _factory(_rt(True), ctx)


def test_PAIRING_two_distinct_H4_runtimes_are_refused():
    ctx = INT.IntegrationContext()
    INT.make_binder(_rt(True), ctx)
    with pytest.raises(AbortError, match="runtime mismatch"):
        _factory(_rt(True), ctx)


def test_PAIRING_a_directly_constructed_agent_is_checked_too():
    ctx = INT.IntegrationContext()
    INT.make_binder(_rt(True), ctx)
    with pytest.raises(AbortError, match="runtime mismatch"):
        INT.T1jAgent(runtime=_rt(False), ctx=ctx, depth=6, colour="red", timeout_s=120)


def test_PAIRING_BASELINE_one_H4_runtime_for_both_is_accepted():
    ctx, rt, _b, factory = h4_adapter()
    assert factory(B._task("t", "red"), "red").runtime is rt


def test_PAIRING_two_DEFAULT_runtimes_are_still_accepted():
    """No new refusal on the default path: `_binder` overrides exist."""
    ctx = INT.IntegrationContext()
    INT.make_binder(_rt(False), ctx)
    _factory(_rt(False), ctx)(B._task("t", "red"), "red")


def test_an_already_used_task_id_is_refused_on_an_H4_context():
    ctx, *_ = h4_adapter()
    ctx.reset("same", [])
    with pytest.raises(AbortError, match="already used"):
        ctx.reset("same", [])


# ──────────────── 4. THE SHARED CLASSIFICATION (§6.3) ────────────────

def test_the_classifier_IS_the_qualified_one():
    assert INT._h4_classifier() is Q.classify_reply


def test_the_classifier_is_called_EXACTLY_ONCE_per_call(wire, monkeypatch):
    seen = []
    real = Q.classify_reply
    monkeypatch.setattr(Q, "classify_reply", lambda **kw: seen.append(kw) or real(**kw))
    wire["query"] = lambda p: query_out(p, native=True)
    call_at(1)
    assert len(seen) == 1 and seen[0]["exit_status"] == 3 and seen[0]["failures"] == 1


def test_a_spy_that_ACCEPTS_makes_the_adapter_accept(wire, monkeypatch):
    """An exit-3 SEARCHED reply, which the real classifier refuses, is accepted
    when the classifier says so: no other site re-judges rc/failures/completed."""
    monkeypatch.setattr(Q, "classify_reply", lambda **kw: Q.SEARCHED)
    wire["query"] = lambda p: query_out(p, **SEARCHED)
    wire["rc_query"] = 3
    move, ctx = call_at(5)
    assert records(ctx)[0]["outcome"] == "accepted"


def test_a_spy_that_REFUSES_makes_the_adapter_refuse(wire, monkeypatch):
    def refuse(**kw):
        raise Q.H4RQStop("spy refuses")
    monkeypatch.setattr(Q, "classify_reply", refuse)
    wire["query"] = lambda p: query_out(p, **SEARCHED)
    wire["rc_query"] = 0
    with pytest.raises(AbortError, match="spy refuses"):
        call_at(5)


# ─────────── 5. RECORDS AND FAILURE KINDS (§4, §5, §6.3.1) ───────────

@pytest.mark.parametrize("step,kw,rc,ply", [
    ("proc", dict(native=True, n_proc=0), 3, 1),
    ("postcond", dict(native=True, refl_n=3), 3, 1),
    ("matchdata", dict(native=True, pie=True), 3, 1),
    ("query_record", dict(native=True, n_records=0), 3, 1),
    ("dump", dict(native=True, dump=False), 3, 1),
    ("classify", dict(native=False), 0, 1),
    ("move", dict(native=True, null_sentinel=True), 3, 1)])
def test_a_REFUSED_call_keeps_exactly_one_record(wire, step, kw, rc, ply):
    wire["query"] = lambda p: query_out(p, **kw)
    wire["rc_query"] = rc
    ctx, *_ = adapter = h4_adapter()
    with pytest.raises(AbortError):
        call_at(ply, adapter=adapter)
    [r] = records(ctx)
    assert r["outcome"] == "refused" and r["refused_at"] == step
    later = {"proc": ["postcond", "matchdata", "telemetry", "source", "move"],
             "postcond": ["matchdata", "telemetry", "source", "move"],
             "matchdata": ["telemetry", "source", "move"],
             "query_record": ["telemetry", "source", "move"],
             "dump": ["source", "move"], "classify": ["source", "move"],
             "move": []}[step]
    assert all(r[k] is None for k in later), {k: r[k] for k in later}


def test_a_REFUSED_replay_keeps_its_record(wire):
    wire["replay"] = lambda p: replay_out(p, n_proc=2)
    ctx, _rt, binder, _f = h4_adapter()
    ctx.reset("r", P[1])
    with pytest.raises(AbortError):
        binder({"task_id": "r"}, _state(P[1]), 1)
    [r] = records(ctx)
    assert (r["role"], r["outcome"], r["refused_at"]) == ("replay", "refused", "proc")


def test_a_SEMANTIC_refusal_carries_the_FULL_stdout_on_the_AGENT(wire):
    served = {}

    def serve(p):
        served["out"] = query_out(p, native=True, n_proc=2)
        return served["out"]
    wire["query"] = serve
    with pytest.raises(AbortError) as ei:
        call_at(1)
    assert isinstance(ei.value.__cause__, A.HelperOutputError)
    assert ei.value.__cause__.stdout == served["out"]


def test_a_SEMANTIC_refusal_carries_the_FULL_stdout_on_the_H4_BINDER(wire):
    served = {}

    def serve(p):
        served["out"] = replay_out(p, n_proc=0)
        return served["out"]
    wire["replay"] = serve
    ctx, _rt, binder, _f = h4_adapter()
    ctx.reset("r", P[2])
    with pytest.raises(AbortError) as ei:
        binder({"task_id": "r"}, _state(P[2]), 2)
    assert ei.value.__cause__.stdout == served["out"]


def test_an_UNREADABLE_line_is_NOT_converted_to_AbortError(wire):
    """A malformed PROC line tells us nothing readable: VOID, not STOP."""
    wire["query"] = lambda p: query_out(p, native=True).replace(
        "PROC pid=", "PROC xid=", 1)
    ctx, *_ = adapter = h4_adapter()
    with pytest.raises(ValueError) as ei:
        call_at(1, adapter=adapter)
    assert not isinstance(ei.value, AbortError)
    [r] = records(ctx)
    assert (r["outcome"], r["refused_at"]) == ("unreadable", "proc")


def test_an_UNREADABLE_query_record_is_NOT_converted(wire):
    wire["query"] = lambda p: query_out(p, native=True).replace(
        "usealphabeta=false", "usealphabeta=garbage", 1)
    ctx, *_ = adapter = h4_adapter()
    with pytest.raises(A.HelperOutputError):
        call_at(1, adapter=adapter)
    [r] = records(ctx)
    assert (r["outcome"], r["refused_at"], r["return_code"]) == ("unreadable", "query",
                                                                  None)


def test_a_TIMEOUT_is_NOT_converted_and_leaves_NO_record(wire):
    wire["timeout"] = True
    ctx, *_ = adapter = h4_adapter()
    with pytest.raises(subprocess.TimeoutExpired):
        call_at(1, adapter=adapter)
    assert records(ctx) == []


def test_task_buckets_SURVIVE_reset(wire):
    ctx, *_ = adapter = h4_adapter()
    call_at(1, adapter=adapter)
    call_at(2, adapter=adapter)
    assert len(ctx.processes) == 2 and len(records(ctx)) == 2
    assert all(len(b) == 1 for b in ctx.processes.values())


# ──────────────── 6. DEFAULT CALLERS UNCHANGED (§6.4) ────────────────

def _default_agent(ply):
    ctx = INT.IntegrationContext()
    rt = _rt(False)
    st = _state(P[ply])
    ctx.reset("d", P[ply])
    return INT.T1jAgent(runtime=rt, ctx=ctx, depth=6, colour=st.to_move), st


def test_the_DEFAULT_argv_is_byte_identical_and_says_query(wire):
    wire["query"] = lambda p: query_out(p, native=False, n_md=0, refl_n=3)
    wire["rc_query"] = 0
    agent, st = _default_agent(5)
    agent(st)
    args = wire["calls"][-1]
    i = args.index(A.PREFLIGHT_MAIN)
    assert args[i + 1:i + 3] == ["query", "6"] and "h4query" not in args
    assert args[:i] == ["/j", f"-Djava.util.prefs.PreferencesFactory={A.PREFS_FACTORY}",
                        "-Djava.awt.headless=true", "-cp", "/x.jar:/c"]


def test_a_DEFAULT_agent_still_REFUSES_a_native_reply_at_site_1(wire):
    wire["query"] = lambda p: query_out(p, native=True, n_md=0, refl_n=3)
    wire["rc_query"] = 3
    agent, st = _default_agent(1)
    with pytest.raises(AbortError, match=r"exit 3 with 1 record\(s\)") as ei:
        agent(st)
    assert ei.value.phase == PHASE_MOVE


def test_a_DEFAULT_agent_still_expects_refl_3(wire):
    wire["query"] = lambda p: query_out(p, native=False, n_md=0, refl_n=4)
    wire["rc_query"] = 0
    agent, st = _default_agent(5)
    with pytest.raises(AbortError, match="4 reflective accesses, expected exactly 3"):
        agent(st)


def test_a_DEFAULT_binder_still_binds_WITHOUT_a_PROC_line(wire):
    wire["replay"] = lambda p: replay_out(p, n_proc=0)
    ctx = INT.IntegrationContext()
    ctx.reset("d", P[2])
    INT.make_binder(_rt(False), ctx)({"task_id": "d"}, _state(P[2]), 2)
    assert ctx.stats["d"]["binds"] == 1 and ctx.processes == {}


def _opt_ins(source):
    """Calls passing `h4_acceptance=<anything but a literal False>`."""
    hits = []
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Call):
            for k in n.keywords:
                if k.arg == "h4_acceptance" and not (
                        isinstance(k.value, ast.Constant) and k.value.value is False):
                    hits.append(ast.unparse(n)[:80])
    return hits


def test_NO_existing_caller_opts_in():
    allowed = {"e4_screen_integration.py", "h4_4b_acceptance_qualification.py"}
    hits = {p.name: _opt_ins(p.read_text(encoding="utf-8"))
            for p in sorted((ROOT / "scripts").rglob("*.py")) if p.name not in allowed}
    assert {k: v for k, v in hits.items() if v} == {}


def test_the_opt_in_walker_is_NOT_vacuous():
    """CLEAN-BASELINE CONTROL: the walker must see an opt-in when one is planted."""
    assert _opt_ins("INT.T1jRuntime(java=j, h4_acceptance=True)")
    assert _opt_ins("f(h4_acceptance=flag)")
    assert not _opt_ins("INT.T1jRuntime(java=j, h4_acceptance=False)")


# ─────────────────────── 7. THE RUNNER (§6.6, §7, §8) ───────────────────────

def _open(monkeypatch):
    monkeypatch.setattr(B, "H4_4B_ACCEPTANCE_QUALIFICATION_AUTHORIZED", True)


def _run(tmp_path, name="r.json", **kw):
    return B.run_qualification(paths=PATHS, out_path=str(tmp_path / name),
                               _compile=lambda d: {"stub": True}, **kw)


def _natives_at_ply4_once(box):
    """The default stub, except ONE ply-4 position answers NATIVELY -- so
    native_initial_fifth_or_more is reached end to end. CONSTRUCTED."""
    done = {"n": 0}

    def serve(p):
        box_pid = box["pid"]
        if len(p) == 4 and done["n"] == 0:
            done["n"] += 1
            box["rc_query"] = Q.NATIVE_EXIT
            return query_out(p, native=True, pid=box_pid)
        box["rc_query"] = None
        return query_out(p, native=len(p) <= 3, pid=box_pid)
    return serve


def test_the_PUBLIC_ENTRY_with_the_gate_OPEN_reaches_CLEAN(tmp_path, wire, monkeypatch):
    """🔴 Tests must run in the REAL state: the public entry, gate OPEN."""
    _open(monkeypatch)
    wire["query"] = _natives_at_ply4_once(wire)
    r = _run(tmp_path)
    assert r["verdict"] == "CLEAN"
    assert r["n_records"] == r["subprocesses_spent"] == r["distinct_pids"] == 72
    assert r["derived_caps"] == {"queries": 56, "replays": 16, "total": 72}
    assert r["matrix_sha256"] == Q.MATRIX_SHA256
    assert r["source_by_ply"]["4"].get(Q.NATIVE_FIFTH_OR_MORE) == 1
    assert r["source_by_ply"]["0"] == {Q.NATIVE_FIRST: 5}
    assert all("h4query" in c for c in wire["calls"] if "replay" not in c)
    assert json.loads((tmp_path / "r.json").read_text())["verdict"] == "CLEAN"


def test_the_run_uses_16_distinct_buckets(tmp_path, wire, monkeypatch):
    _open(monkeypatch)
    seen = {}
    real = B.all_records

    def spy(ctx):
        seen["buckets"] = len(ctx.processes)
        return real(ctx)
    monkeypatch.setattr(B, "all_records", spy)
    _run(tmp_path)
    assert seen["buckets"] == 16


def test_a_STOP_leaves_a_DURABLE_record_with_the_FULL_stdout(tmp_path, wire, monkeypatch):
    _open(monkeypatch)
    served = {}

    def serve(p):
        served["out"] = query_out(p, native=True, null_sentinel=True)
        return served["out"]
    wire["query"] = serve
    with pytest.raises(B.H4BStop):
        _run(tmp_path, "stop.json")
    rec = json.loads((tmp_path / "stop.json").read_text())
    assert rec["verdict"] == "STOP" and rec["exception"] == "AbortError"
    assert rec["stdout"] == served["out"]
    assert rec["failing_position"]["ply"] == 0
    assert [r["outcome"] for r in rec["records"]] == ["accepted", "refused"]
    assert "NOT a partial qualification" in rec["scope"]


def test_a_TIMEOUT_is_a_VOID_with_a_durable_record(tmp_path, wire, monkeypatch):
    _open(monkeypatch)
    wire["timeout"] = True
    with pytest.raises(B.H4BVoidError):
        _run(tmp_path, "void.json")
    rec = json.loads((tmp_path / "void.json").read_text())
    assert rec["verdict"] == "VOID" and rec["exception"] == "TimeoutExpired"


def test_an_UNREADABLE_reply_is_a_VOID_not_a_STOP(tmp_path, wire, monkeypatch):
    _open(monkeypatch)
    wire["query"] = lambda p: query_out(p, native=True).replace("PROC pid=", "PROC xid=")
    with pytest.raises(B.H4BVoidError):
        _run(tmp_path, "void.json")
    assert json.loads((tmp_path / "void.json").read_text())["verdict"] == "VOID"


def test_a_CONSTRUCTION_refusal_writes_nothing_and_spawns_nothing(tmp_path, wire,
                                                                   monkeypatch):
    _open(monkeypatch)
    monkeypatch.setattr(B, "DEPTH", 5)
    with pytest.raises(B.H4BError, match="refused construction"):
        _run(tmp_path, "c.json")
    assert wire["calls"] == [] and not (tmp_path / "c.json").exists()


def test_a_matrix_that_does_not_reproduce_is_refused_before_anything(tmp_path, wire,
                                                                      monkeypatch):
    _open(monkeypatch)
    from scripts.GPU.alphazero import h4_4a_characterization as H4A
    rows = [dict(r) for r in H4A.load_frozen_prefixes()]
    next(r for r in rows if r["opening"] == "o1_center" and r["ply"] == 1)["prefix"] = \
        [[19, 19]]
    with pytest.raises(B.H4BError, match="did not reproduce"):
        _run(tmp_path, "m.json", prefixes=rows)
    assert wire["calls"] == [] and not (tmp_path / "m.json").exists()


def test_an_OCCUPIED_destination_is_refused_before_any_subprocess(tmp_path, wire,
                                                                  monkeypatch):
    _open(monkeypatch)
    (tmp_path / "taken.json").write_text("{}")
    with pytest.raises(FileExistsError):
        _run(tmp_path, "taken.json")
    assert wire["calls"] == []


def test_the_switch_DEFAULTS_OFF(wire):
    """A runtime built WITHOUT the keyword is the default path, and still refuses
    a native reply. Every existing caller builds it exactly this way."""
    rt = INT.T1jRuntime(java="/j", jar="/x.jar", classes="/c", ply_cap=280, timeout_s=120)
    assert rt.h4_acceptance is False
    wire["query"] = lambda p: query_out(p, native=True, n_md=0, refl_n=3)
    wire["rc_query"] = 3
    ctx = INT.IntegrationContext()
    st = _state(P[1])
    ctx.reset("d", P[1])
    with pytest.raises(AbortError, match=r"exit 3 with 1 record\(s\)"):
        INT.T1jAgent(runtime=rt, ctx=ctx, depth=6, colour=st.to_move)(st)


# ═══════════ 8. CONTROLS ADDED AFTER REVIEW OF b86420d ═══════════
# Each enumerated check must fail a test of its own when deleted: ysize and
# frames had none, and the H4 binder -- a SEPARATE implementation of the replay
# checks -- was controlled only for its PROC count.

def test_MATCHDATA_ysize_other_than_24_is_refused(wire):
    _refused(wire, 1, "did not read back", rc=3, native=True, ys=23)


def test_POSTCOND_frames_nonzero_is_refused_on_the_QUERY_path(wire):
    _refused(wire, 1, "1 frames opened", rc=3,
             serve=lambda p: _post_line(query_out(p, native=True), "frames=0", "frames=1"))


def _h4_bind(wire, serve=None, ply=2):
    """The PRODUCTION H4 binder on an H4 runtime. Returns (ctx, call)."""
    if serve is not None:
        wire["replay"] = serve
    ctx, _rt, binder, _f = h4_adapter()
    ctx.reset("r", P[ply])
    return ctx, lambda: binder({"task_id": "r"}, _state(P[ply]), ply)


def _bind_refused(wire, step, match, serve=None, rc=0):
    served = {}

    def keep(p):
        served["out"] = (serve or (lambda q: replay_out(q)))(p)
        return served["out"]
    wire["rc_replay"] = rc
    ctx, call = _h4_bind(wire, keep)
    with pytest.raises(AbortError, match=match) as ei:
        call()
    [r] = records(ctx)
    assert (r["role"], r["outcome"], r["refused_at"]) == ("replay", "refused", step)
    assert ei.value.__cause__.stdout == served["out"], "the FULL stdout must ride along"


def test_H4_BINDER_a_nonzero_replay_exit_is_refused(wire):
    _bind_refused(wire, "exit", "T1j replay exit 1", rc=1)


def test_H4_BINDER_replay_failures_nonzero_is_refused(wire):
    _bind_refused(wire, "postcond", "failures=1 on a replay",
                  serve=lambda p: replay_out(p, failures=1))


def test_H4_BINDER_frames_nonzero_is_refused(wire):
    _bind_refused(wire, "postcond", "1 frames opened",
                  serve=lambda p: _post_line(replay_out(p), "frames=0", "frames=1"))


def test_H4_BINDER_a_wrong_block_count_is_refused(wire):
    _bind_refused(wire, "plies", "reported 1 plies, expected 3",
                  serve=lambda p: replay_out(p, blocks=1))


def test_H4_BINDER_a_divergent_replayed_state_is_refused(wire):
    _bind_refused(wire, "state", "pegs|history",
                  serve=lambda p: replay_out(OTHER[2]))


def test_H4_BINDER_unreadable_replay_output_is_NOT_converted(wire):
    """`A.replay`'s own parser refuses: VOID, with a record and no return code."""
    ctx, call = _h4_bind(wire, lambda p: replay_out(p).replace("moveNr=0", "moveNr=zz", 1))
    with pytest.raises(A.HelperOutputError) as ei:
        call()
    assert not isinstance(ei.value, AbortError)
    [r] = records(ctx)
    assert (r["outcome"], r["refused_at"], r["return_code"]) == ("unreadable", "replay",
                                                                  None)


def test_H4_BINDER_an_unreadable_PROC_line_is_NOT_converted(wire):
    ctx, call = _h4_bind(wire, lambda p: replay_out(p).replace("PROC pid=", "PROC xid=", 1))
    with pytest.raises(ValueError) as ei:
        call()
    assert not isinstance(ei.value, AbortError)
    [r] = records(ctx)
    assert (r["outcome"], r["refused_at"]) == ("unreadable", "proc")


def test_H4_BINDER_a_TIMEOUT_is_NOT_converted_and_leaves_NO_record(wire):
    wire["timeout"] = True
    ctx, call = _h4_bind(wire)
    with pytest.raises(subprocess.TimeoutExpired):
        call()
    assert records(ctx) == []


# ── the create-only class directory: the DEFAULT compile route (card §6.6) ──
# The runner tests above inject `_compile`. These drive the runner's REAL default
# route into the real `d1_probe._default_compile`, whose occupied-directory
# refusal is `tests/test_d1_probe.py::test_an_existing_class_directory_is_refused`.
# The toolchain and javac are faked by THAT file's fixtures; nothing executes.
from tests.test_d1_probe import _paths_for, javac, toolchain  # noqa: E402,F401
from scripts.GPU.alphazero import d1_probe as D1  # noqa: E402


def test_the_default_compile_route_IS_the_verified_create_only_compiler():
    assert B._compile_helper_verified is D1._default_compile


def test_the_DEFAULT_route_refuses_an_OCCUPIED_class_directory(tmp_path, wire, toolchain,
                                                               javac, monkeypatch):
    _open(monkeypatch)
    classes = tmp_path / "classes"
    classes.mkdir()
    with pytest.raises(B.H4BVoidError, match="already exists"):
        B.run_qualification(paths=_paths_for(toolchain, classes),
                            out_path=str(tmp_path / "r.json"))
    assert javac == [] and wire["calls"] == [], "it compiled or spawned into a used dir"
    rec = json.loads((tmp_path / "r.json").read_text())
    assert rec["verdict"] == "VOID" and "already exists" in rec["reason"]


def test_the_DEFAULT_route_CREATES_the_class_directory_and_compiles_into_it(
        tmp_path, wire, toolchain, javac, monkeypatch):
    _open(monkeypatch)
    classes = tmp_path / "classes"
    r = B.run_qualification(paths=_paths_for(toolchain, classes),
                            out_path=str(tmp_path / "r.json"))
    assert r["verdict"] == "CLEAN" and classes.is_dir()
    assert [c["out"] for c in javac] == [str(classes)]
    assert r["toolchain_identity"]["classes_dir"] == str(classes)
