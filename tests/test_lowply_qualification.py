"""The gated low-ply qualification runner. NOTHING EXECUTES.

No JVM is started, no model loaded, no seed drawn or registered, no D1 touched.
`subprocess.run` is intercepted at the process boundary, so the whole of our own
code runs while nothing is ever spawned.

THE ONE THING THIS FILE EXISTS TO PIN. D1 conflated FAIL with VOID: a T1j reply
that did not complete its requested depth ABORTED THE WHOLE RUN, so the
observation could not be recorded -- and that is exactly why the low-ply
question is still open. Here T1j failing to complete IS THE MEASUREMENT. VOID is
reserved for the instrument.
"""
import json
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import e4_screen_integration as INT
from scripts.GPU.alphazero import lowply_qualification as LP
from scripts.GPU.alphazero import t1j_adapter as A
from scripts.GPU.alphazero.game.twixt_state import TwixtState

PREFIX = [(11, 11), (12, 13), (13, 12)]
PATHS = LP.T1jPaths(java="/nonexistent/java", jar="/nonexistent/t1j.jar",
                    classes="/nonexistent/classes", ply_cap=280)

CLEAN_POST = ("POSTCOND no_throw=true windows=0 frames=0 headless=true prefs_ok=true "
              "refl_ok=true refl_n={n} failures={f}")


def _state(moves):
    st = TwixtState(active_size=24, to_move="red")
    for mv in moves:
        st = st.apply_move(tuple(mv))
    return st


def _block(state, moves):
    pegs, bridges = A.our_snapshot(state)
    legal = {A.to_t1j(r, c) for (r, c) in state.legal_moves()}
    bits = "".join("1" if (i // A.BOARD_N, i % A.BOARD_N) in legal else "0"
                   for i in range(A.LEGAL_BITS))
    hist = " ".join(f"{x},{y}" for x, y in (A.to_t1j(*m) for m in moves))
    return (f"PLY {state.ply} moveNr={state.ply} "
            f"next={A.PLAYER_TO_T1J[state.to_move]} termY=false termX=false\n"
            f"  PEGS {' '.join(sorted(pegs))}\n"
            f"  BRIDGES {' '.join(sorted(bridges))}\n"
            f"  HIST {hist}\n  LEGAL {bits}\n")


def _replay_out(prefix):
    st, moves, out = _state([]), [], []
    for mv in list(prefix) + [None]:
        out.append(_block(st, moves))
        if mv is None:
            break
        moves.append(tuple(mv))
        st = st.apply_move(tuple(mv))
    return "".join(out) + CLEAN_POST.format(n=INT.REPLAY_REFL_N, f=0) + "\n"


def _query_out(prefix, depth, *, completed=True, legal=True, move=None,
               refl_n=None, failures=None, post=True, dump=True):
    st = _state(prefix)
    mv = move if move is not None else sorted(st.legal_moves())[0]
    x, y = A.to_t1j(*mv)
    fails = (0 if (completed and legal) else 1) if failures is None else failures
    line = (f"QUERY q=1 requested_depth={depth} move_x={x} move_y={y} "
            f"to_move={A.PLAYER_TO_T1J[st.to_move]} usealphabeta=true "
            f"currentMaxPly={depth} completed_depth={depth if completed else -1} "
            f"completed={'true' if completed else 'false'} "
            f"legal={'true' if legal else 'false'} null_sentinel=false "
            f"moveNr={len(prefix)} eval_regime=normal elapsed_us=1000\n")
    body = _block(st, list(prefix)) if dump else ""
    tail = ""
    if not completed:
        tail += f"FAIL q1: requested depth {depth} completed\n"
    if post:
        tail += CLEAN_POST.format(n=A.parse_postconds("") and 3 or
                                  (INT.QUERY_REFL_N if refl_n is None else refl_n),
                                  f=fails) + "\n"
    return line + body + tail


@pytest.fixture
def wire(monkeypatch):
    """Serve replays and query replies at the process boundary. Spawns nothing."""
    box = {"calls": [], "prefix": PREFIX, "rc": 0, "query": None}

    def fake_run(args, **kw):
        box["calls"].append({"args": args, "kw": kw})
        if "replay" in args:
            return subprocess.CompletedProcess(args, 0, _replay_out(box["prefix"]), "")
        depth = int(args[args.index("query") + 1])
        out = box["query"](depth) if box["query"] else _query_out(box["prefix"], depth)
        return subprocess.CompletedProcess(args, box["rc"], out, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return box


def _prefixes(n=1):
    return [{"task_id": f"t{i}", "ply": len(PREFIX), "prefix": [list(m) for m in PREFIX],
             "digest": LP.digest_of(_state(PREFIX)), "opening": "o1_center",
             "colour_arm": "t1j_red", "signature": "mover_fragmentation",
             "role": "position", "phase": "opening"} for i in range(n)]


def _run(tmp_path, prefixes=None, **kw):
    return LP._run_unguarded(prefixes=prefixes or _prefixes(),
                             paths=PATHS, out_path=str(tmp_path / "r.json"),
                             _compile=lambda d: {"stub": True}, **kw)


# ────────────────────────────────── the gate ────────────────────────────────

def test_the_gate_is_false_as_published():
    assert LP.LOWPLY_QUALIFICATION_AUTHORIZED is False


def test_the_public_runner_refuses_while_the_gate_is_shut(tmp_path, wire):
    with pytest.raises(LP.LowPlyError, match="UNAUTHORIZED"):
        LP.run_qualification(prefixes=_prefixes(), paths=PATHS,
                             out_path=str(tmp_path / "r.json"))
    assert wire["calls"] == [], "it queried T1j while unauthorized"
    assert not (tmp_path / "r.json").exists()


def test_the_cli_refuses_in_a_fresh_subprocess(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "scripts.GPU.alphazero.lowply_qualification",
         "--out", str(tmp_path / "r.json")],
        capture_output=True, text=True, cwd=".", timeout=60)
    assert r.returncode == LP.EXIT_UNAUTHORIZED == 5, (r.returncode, r.stderr)
    assert not (tmp_path / "r.json").exists()


def test_the_gate_is_read_at_both_public_entries():
    """Load-context reads only: an assignment is not a read."""
    import ast, pathlib
    tree = ast.parse(pathlib.Path(LP.__file__).read_text(encoding="utf-8"))
    reads = [n for n in ast.walk(tree)
             if isinstance(n, ast.Name) and n.id == "LOWPLY_QUALIFICATION_AUTHORIZED"
             and isinstance(n.ctx, ast.Load)]
    assert len(reads) >= 2, len(reads)


def test_this_module_reads_NO_OTHER_experiments_gate():
    """One gate must never be openable by opening another. D1's timing machinery
    is imported; D1's GATE is not."""
    import pathlib
    src = pathlib.Path(LP.__file__).read_text(encoding="utf-8")
    for other in ("D1_EXECUTION_AUTHORIZED", "L0_EXECUTION_AUTHORIZED",
                  "SCREEN_AUTHORIZED"):
        assert other not in src, other


# ────────────────────── the card's frozen numbers, in code ──────────────────

def test_the_frozen_constants_match_the_card():
    assert LP.DEPTHS == (3, 6)
    assert LP.INVOCATIONS_PER_DEPTH == 2
    assert LP.N_PREFIXES == 9
    assert LP.QUERY_CAP == 36 == LP.N_PREFIXES * len(LP.DEPTHS) * LP.INVOCATIONS_PER_DEPTH
    assert LP.PER_CALL_TIMEOUT_S == 120
    assert LP.RUN_DEADLINE_S == 900


def test_the_frozen_prefix_file_is_pinned_by_hash():
    got = LP.load_frozen_prefixes()
    assert len(got) == LP.N_PREFIXES == 9
    assert sorted({len(p["prefix"]) for p in got}) == [1, 3, 5]


def test_a_tampered_prefix_file_is_refused(tmp_path):
    bad = tmp_path / "p.json"
    bad.write_text(json.dumps({"prefixes": []}), encoding="utf-8")
    with pytest.raises(LP.LowPlyError, match="sha256"):
        LP.load_frozen_prefixes(str(bad))


# ══════════ THE POINT OF THIS RUNNER: FAIL IS A RESULT, NOT AN ABORT ════════

def test_a_reply_that_does_not_complete_its_depth_is_a_recorded_FAIL(wire, tmp_path):
    """THE measurement. In D1 this aborted the whole run and the observation was
    lost; here the run completes and records it."""
    wire["query"] = lambda d: _query_out(PREFIX, d, completed=False)
    wire["rc"] = 3                       # E4Preflight exits 3 when failures > 0
    report = _run(tmp_path)
    assert report["verdict"] == "FAIL"
    obs = report["prefixes"][0]["depths"][0]["invocations"][0]
    assert obs["completed"] is False
    assert "did not complete" in " ".join(obs["failures"])
    assert (tmp_path / "r.json").exists(), "a FAIL must still produce its record"


def test_a_nonzero_exit_with_a_usable_reply_is_a_FAIL_carrying_the_transcript(
        wire, tmp_path):
    wire["query"] = lambda d: _query_out(PREFIX, d, completed=False)
    wire["rc"] = 3
    report = _run(tmp_path)
    obs = report["prefixes"][0]["depths"][0]["invocations"][0]
    assert obs["exit_status"] == 3
    assert "FAIL q1: requested depth" in obs["helper_report"]


def test_an_illegal_move_is_a_FAIL_not_an_abort(wire, tmp_path):
    wire["query"] = lambda d: _query_out(PREFIX, d, move=PREFIX[0])   # occupied
    report = _run(tmp_path)
    assert report["verdict"] == "FAIL"
    obs = report["prefixes"][0]["depths"][0]["invocations"][0]
    assert any("illegal" in f for f in obs["failures"]), obs["failures"]


def test_two_invocations_that_disagree_are_a_FAIL(wire, tmp_path):
    st = _state(PREFIX)
    a, b = sorted(st.legal_moves())[:2]
    seen = {"n": 0}

    def alternating(depth):
        seen["n"] += 1
        return _query_out(PREFIX, depth, move=a if seen["n"] % 2 else b)

    wire["query"] = alternating
    report = _run(tmp_path)
    assert report["verdict"] == "FAIL"
    assert any("disagree" in f for f in report["prefixes"][0]["depths"][0]["failures"])


def test_an_all_good_run_is_a_PASS_and_spends_the_whole_budget(wire, tmp_path):
    report = _run(tmp_path, prefixes=_prefixes(2))
    assert report["verdict"] == "PASS"
    assert report["queries_spent"] == 2 * len(LP.DEPTHS) * LP.INVOCATIONS_PER_DEPTH
    assert all(d["agree"] for p in report["prefixes"] for d in p["depths"])


# ───────────────────────── VOID is for the instrument ───────────────────────

def test_a_REPLAY_timeout_is_a_VOID_and_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run",
                        lambda a, **k: (_ for _ in ()).throw(
                            subprocess.TimeoutExpired(a, k.get("timeout"))))
    out = tmp_path / "r.json"
    with pytest.raises(LP.LowPlyVoidError, match="prefix replay did not answer"):
        LP._run_unguarded(prefixes=_prefixes(), paths=PATHS, out_path=str(out),
                          _compile=lambda d: None)
    assert not out.exists()


def test_a_QUERY_timeout_is_a_VOID_and_writes_nothing(monkeypatch, tmp_path):
    """REACHED ALONE. The replay succeeds and only the query times out.

    A previous version patched every subprocess call, so the BINDER's replay
    timed out first -- and `TimeoutExpired.__str__` itself contains "timed out",
    so a match on that phrase passed without `_query_once`'s handler ever
    running. An injected-defect control caught it: deleting that handler changed
    nothing the test could see.
    """
    def fake_run(args, **kw):
        if "replay" in args:
            return subprocess.CompletedProcess(args, 0, _replay_out(PREFIX), "")
        raise subprocess.TimeoutExpired(args, kw.get("timeout"))

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = tmp_path / "r.json"
    with pytest.raises(LP.LowPlyVoidError, match="T1j did not answer within"):
        LP._run_unguarded(prefixes=_prefixes(), paths=PATHS, out_path=str(out),
                          _compile=lambda d: {"stub": True})
    assert not out.exists()


def test_a_deadline_breach_is_a_VOID_and_writes_nothing(wire, tmp_path):
    ticks = iter([0.0, 0.0, 0.0, 99999.0] + [99999.0] * 40)
    out = tmp_path / "r.json"
    with pytest.raises(LP.LowPlyVoidError, match="deadline"):
        LP._run_unguarded(prefixes=_prefixes(), paths=PATHS, out_path=str(out),
                          _compile=lambda d: None,
                          deadline=LP.Deadline(limit_s=LP.RUN_DEADLINE_S,
                                               clock=lambda: next(ticks)))
    assert not out.exists()


def test_a_reply_with_no_query_record_at_all_is_a_VOID(wire, tmp_path):
    wire["query"] = lambda d: "PROC pid=1 java_version=17 vm=x headless=true prefs_factory=e\n"
    wire["rc"] = 1
    with pytest.raises(LP.LowPlyVoidError, match="no usable query record"):
        _run(tmp_path)


def test_the_budget_refuses_the_query_that_would_exceed_the_cap(wire, tmp_path):
    with pytest.raises(LP.LowPlyError, match="budget"):
        _run(tmp_path, prefixes=_prefixes(2), budget=LP.QueryBudget(cap=3))


def test_the_record_is_create_only(wire, tmp_path):
    out = tmp_path / "r.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        LP._run_unguarded(prefixes=_prefixes(), paths=PATHS, out_path=str(out),
                          _compile=lambda d: {"stub": True})
    assert out.read_text(encoding="utf-8") == "{}"


# ───────────────────── shape at the process boundary ────────────────────────

def test_every_call_carries_the_frozen_timeout_at_the_boundary(wire, tmp_path):
    _run(tmp_path)
    assert wire["calls"]
    for c in wire["calls"]:
        assert c["kw"].get("timeout") == LP.PER_CALL_TIMEOUT_S == 120, c["kw"]


def test_each_depth_issues_two_separate_repeats_1_invocations(wire, tmp_path):
    _run(tmp_path)
    q = [c for c in wire["calls"] if "query" in c["args"]]
    assert len(q) == len(LP.DEPTHS) * LP.INVOCATIONS_PER_DEPTH == 4
    for c in q:
        assert "determinism" not in c["args"], c["args"]


def test_one_replay_per_prefix(wire, tmp_path):
    _run(tmp_path, prefixes=_prefixes(2))
    assert len([c for c in wire["calls"] if "replay" in c["args"]]) == 2


# ─────────────── the boundary this runner may never cross ───────────────────

@pytest.mark.parametrize("forbidden", [
    "load_reference_evaluator", "_default_load_evaluator", "SeededReferenceAgent",
    "rng_stream_seeds", "ACCOUNTED_SEED_INTERVALS", "seed_is_accounted",
    "random.Random", "SEED_INTERVAL",
])
def test_the_runner_touches_no_model_and_no_seed_machinery(forbidden):
    """It queries T1j alone. It needs no seed interval because it never invokes
    the incumbent, and a reference to either would be the first step to doing so."""
    import pathlib
    src = pathlib.Path(LP.__file__).read_text(encoding="utf-8")
    assert forbidden not in src, forbidden


# ══════ malformed helper output is INSTRUMENT failure, so it must VOID ═══════
#
# The card defines unparseable output as instrument failure. But the adapter's
# parsers raise bare ValueError/KeyError, and none of it was translated: a
# malformed QUERY line, a malformed dump header or a malformed POSTCOND escaped
# `main` entirely -- not even as EXIT_UNEXPECTED, since `main` has no catch-all,
# so the process died on an uncaught traceback with no verdict at all.
#
# Each case below is ISOLATED to one parser, and each asserts NO RECORD IS
# WRITTEN: a VOID that left a file behind would be the partial-cohort artifact
# the card forbids.

BAD_QUERY = "QUERY q=1 requested_depth=3 move_x=11\n"          # missing fields
BAD_DUMP = "PLY 3 next=Y termY=false termX=false\n  PEGS \n  LEGAL 0101\n"
BAD_POST = "POSTCOND no_throw=true windows=0\n"                # missing fields


def _void_on(monkeypatch, tmp_path, *, query_out=None, replay_out=None, match=""):
    def fake_run(args, **kw):
        if "replay" in args:
            return subprocess.CompletedProcess(
                args, 0, replay_out if replay_out is not None else _replay_out(PREFIX), "")
        depth = int(args[args.index("query") + 1])
        return subprocess.CompletedProcess(
            args, 0, query_out if query_out is not None else _query_out(PREFIX, depth), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = tmp_path / "r.json"
    with pytest.raises(LP.LowPlyVoidError, match=match) as e:
        LP._run_unguarded(prefixes=_prefixes(), paths=PATHS, out_path=str(out),
                          _compile=lambda d: {"stub": True})
    assert not out.exists(), "a VOID wrote a record"
    return str(e.value)


def test_a_malformed_QUERY_line_is_a_VOID_not_a_traceback(monkeypatch, tmp_path):
    msg = _void_on(monkeypatch, tmp_path, query_out=BAD_QUERY,
                   match="could not be parsed")
    assert "QUERY" in msg


def test_a_malformed_DUMP_header_is_a_VOID_not_a_traceback(monkeypatch, tmp_path):
    """Reached alone: the QUERY line is well formed, only the dump is not."""
    st = _state(PREFIX)
    x, y = A.to_t1j(*sorted(st.legal_moves())[0])
    good_query = (f"QUERY q=1 requested_depth=3 move_x={x} move_y={y} to_move=X "
                  "usealphabeta=true currentMaxPly=3 completed_depth=3 completed=true "
                  "legal=true null_sentinel=false moveNr=3 eval_regime=normal "
                  "elapsed_us=1000\n")
    _void_on(monkeypatch, tmp_path, query_out=good_query + BAD_DUMP,
             match="could not be parsed")


def test_a_malformed_POSTCOND_is_a_VOID_not_a_traceback(monkeypatch, tmp_path):
    """Reached alone: QUERY and dump both parse; only POSTCOND is malformed, so
    only `_observe_reply`'s parse can be the thing that fails."""
    st = _state(PREFIX)
    depth = 3
    x, y = A.to_t1j(*sorted(st.legal_moves())[0])
    good = (f"QUERY q=1 requested_depth={depth} move_x={x} move_y={y} to_move="
            f"{A.PLAYER_TO_T1J[st.to_move]} usealphabeta=true currentMaxPly={depth} "
            f"completed_depth={depth} completed=true legal=true null_sentinel=false "
            f"moveNr={len(PREFIX)} eval_regime=normal elapsed_us=1000\n")
    msg = _void_on(monkeypatch, tmp_path,
                   query_out=good + _block(st, list(PREFIX)) + BAD_POST,
                   match="could not be parsed")
    assert "POSTCOND" in msg


def test_a_malformed_REPLAY_dump_is_a_VOID_not_a_traceback(monkeypatch, tmp_path):
    """The binder's own parse path, reached alone."""
    _void_on(monkeypatch, tmp_path, replay_out=BAD_DUMP, match="could not be parsed")


def test_every_malformed_case_carries_the_bounded_transcript(monkeypatch, tmp_path):
    """The helper's own output is the diagnosis, bounded, and the dump body never
    travels -- the same rule the discarded-transcript repair established."""
    msg = _void_on(monkeypatch, tmp_path,
                   query_out=BAD_QUERY + "  LEGAL " + "1" * 576 + "\n",
                   match="could not be parsed")
    assert "1" * 100 not in msg, "the legal-cell map leaked into the refusal"
    assert len(msg) < 2000, len(msg)


def test_main_reports_rather_than_escaping_on_an_unexpected_error(monkeypatch, tmp_path):
    """`main` had no catch-all, so anything it did not name escaped as a
    traceback with no verdict. A reported exit is the minimum."""
    monkeypatch.setattr(LP, "LOWPLY_QUALIFICATION_AUTHORIZED", True)
    monkeypatch.setattr(LP, "load_frozen_prefixes",
                        lambda p=None: (_ for _ in ()).throw(RuntimeError("boom")))
    rc = LP.main(["--out", str(tmp_path / "r.json")])
    assert rc == LP.EXIT_UNEXPECTED == 4
    assert not (tmp_path / "r.json").exists()
