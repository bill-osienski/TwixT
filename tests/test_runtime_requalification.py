"""The gated Java-only runtime requalification runner. NOTHING EXECUTES.

No JVM is started, no model loaded, no seed drawn, no gate opened. `subprocess.run`
is intercepted at the process boundary for the T1j paths; the process-tree
supervisor is exercised with a throwaway PYTHON child (never Java).

WHAT THIS FILE PINS. (1) The gate binds at every public entry and the CLI refuses
as a fresh subprocess. (2) The frozen input is the eight sha-pinned screen
openings, o3_low flagged as the 2026-09-05 H1 failure. (3) The new POSTCOND
observation fields survive the REAL adapter/diagnostic path -- T1jAgent, the
AbortError-from-HelperOutputError chain, `_bounded_excerpt`/`_helper_text` -- and
their ABSENCE is a VOID, because the compiled class is known to emit them.
(4) FAIL is a RESULT; VOID is the instrument. (5) The outer supervisor kills the
whole PROCESS GROUP on timeout, not just its direct child, and exit codes mean
one thing each.
"""
import json
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from scripts.GPU.alphazero import e4_screen_integration as INT
from scripts.GPU.alphazero import h1_viability_plan as PLAN
from scripts.GPU.alphazero import runtime_requalification as RQ
from scripts.GPU.alphazero import t1j_adapter as A
from scripts.GPU.alphazero.game.twixt_state import TwixtState

PATHS = RQ.T1jPaths(java="/nonexistent/java", jar="/nonexistent/t1j.jar",
                    classes="/nonexistent/classes", ply_cap=280)
SHA = "6cb3a052650f90de53f34a8eb25455c470c6254c5f0fcac3f80c3ca9e8d0128d"
OBS = " prefs_before={b} prefs_after={a} count_before={cb} count_after={ca}"
CLEAN_POST = ("POSTCOND no_throw=true windows=0 frames=0 headless=true prefs_ok={ok} "
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


def _post(*, ok=True, n=INT.QUERY_REFL_N, f=0, obs=True, before=SHA, after=SHA,
          cb=527, ca=527):
    line = CLEAN_POST.format(ok="true" if ok else "false", n=n, f=f)
    if obs:
        line += OBS.format(b=before, a=after, cb=cb, ca=ca)
    return line + "\n"


def _replay_out(prefix):
    st, moves, out = _state([]), [], []
    for mv in list(prefix) + [None]:
        out.append(_block(st, moves))
        if mv is None:
            break
        moves.append(tuple(mv))
        st = st.apply_move(tuple(mv))
    # E3bDump emits the EARLIER shape: no observation fields, by design
    return "".join(out) + _post(n=INT.REPLAY_REFL_N, obs=False)


def _query_out(prefix, depth, *, completed=True, legal=True, move=None, post=None,
               fail_lines=(), dump=True, dump_prefix=None):
    st = _state(prefix)
    mv = move if move is not None else sorted(st.legal_moves())[0]
    x, y = A.to_t1j(*mv)
    line = (f"QUERY q=1 requested_depth={depth} move_x={x} move_y={y} "
            f"to_move={A.PLAYER_TO_T1J[st.to_move]} usealphabeta=true "
            f"currentMaxPly={depth + 1} completed_depth={depth if completed else -1} "
            f"completed={'true' if completed else 'false'} "
            f"legal={'true' if legal else 'false'} null_sentinel=false "
            f"moveNr={len(prefix)} eval_regime=early_moveNr_lt_8 elapsed_us=1000\n")
    dp = prefix if dump_prefix is None else dump_prefix
    body = _block(_state(dp), list(dp)) if dump else ""
    tail = "".join(fail_lines)
    if not completed:
        tail += f"FAIL q1: requested depth {depth} completed\n"
    if post is None:
        post = _post(f=0 if (completed and legal and not fail_lines) else 1)
    return line + body + tail + post


@pytest.fixture
def wire(monkeypatch):
    """Serve replays and query replies at the process boundary. Spawns nothing."""
    box = {"calls": [], "rc": 0, "query": None}

    def fake_run(args, **kw):
        box["calls"].append({"args": args, "kw": kw})
        if "replay" in args:
            moves = [tuple(int(v) for v in s.split(",")) for s in args[args.index("replay") + 2:]]
            ours = [A.to_ours(x, y) for (x, y) in moves]
            return subprocess.CompletedProcess(args, 0, _replay_out(ours), "")
        depth = int(args[args.index("query") + 1])
        moves = [tuple(int(v) for v in s.split(",")) for s in args[args.index("query") + 2:]]
        ours = [A.to_ours(x, y) for (x, y) in moves]
        out = box["query"](ours, depth) if box["query"] else _query_out(ours, depth)
        rc = box["rc"]
        if isinstance(out, tuple):                 # a per-reply exit status
            rc, out = out
        return subprocess.CompletedProcess(args, rc, out, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return box


def _prefixes(n=None):
    got = RQ.load_frozen_prefixes()
    return got if n is None else got[:n]


def _run(tmp_path, prefixes=None, **kw):
    return RQ._run_unguarded(prefixes=prefixes or _prefixes(), paths=PATHS,
                             out_path=str(tmp_path / "r.json"),
                             _compile=lambda d: {"stub": True}, **kw)


# ────────────────────────────────── the gate ────────────────────────────────

def test_the_gate_is_false_as_published():
    assert RQ.RUNTIME_REQUAL_AUTHORIZED is False


def test_the_public_runner_refuses_while_the_gate_is_shut(tmp_path, wire):
    with pytest.raises(RQ.RequalError, match="UNAUTHORIZED"):
        RQ.run_requalification(prefixes=_prefixes(), paths=PATHS,
                               out_path=str(tmp_path / "r.json"))
    assert wire["calls"] == [], "it queried T1j while unauthorized"
    assert not (tmp_path / "r.json").exists()


def test_the_cli_refuses_in_a_fresh_subprocess(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "scripts.GPU.alphazero.runtime_requalification",
         "--out", str(tmp_path / "r.json")],
        capture_output=True, text=True, cwd=".", timeout=60)
    assert r.returncode == RQ.EXIT_UNAUTHORIZED == 5, (r.returncode, r.stderr)
    assert "UNAUTHORIZED" in r.stderr
    assert not (tmp_path / "r.json").exists()


def test_the_WORKER_entry_refuses_in_a_fresh_subprocess_too(tmp_path):
    """The supervisor spawns the worker as its own process; a worker that trusted
    its parent to have checked would be a gate with a way around it."""
    r = subprocess.run(
        [sys.executable, "-m", "scripts.GPU.alphazero.runtime_requalification",
         "--worker", "--out", str(tmp_path / "r.json")],
        capture_output=True, text=True, cwd=".", timeout=60)
    assert r.returncode == RQ.EXIT_UNAUTHORIZED == 5, (r.returncode, r.stderr)
    assert not (tmp_path / "r.json").exists()


def test_the_gate_is_read_at_all_three_public_entries():
    import ast, pathlib
    tree = ast.parse(pathlib.Path(RQ.__file__).read_text(encoding="utf-8"))
    reads = [n for n in ast.walk(tree)
             if isinstance(n, ast.Name) and n.id == "RUNTIME_REQUAL_AUTHORIZED"
             and isinstance(n.ctx, ast.Load)]
    assert len(reads) >= 3, len(reads)


def test_this_module_reads_NO_OTHER_experiments_gate():
    import pathlib
    src = pathlib.Path(RQ.__file__).read_text(encoding="utf-8")
    for other in ("D1_EXECUTION_AUTHORIZED", "L0_EXECUTION_AUTHORIZED",
                  "SCREEN_AUTHORIZED", "LOWPLY_QUALIFICATION_AUTHORIZED",
                  "H1_EXECUTION_AUTHORIZED"):
        assert other not in src, other


@pytest.mark.parametrize("forbidden", [
    "load_reference_evaluator", "_default_load_evaluator", "SeededReferenceAgent",
    "rng_stream_seeds", "ACCOUNTED_SEED_INTERVALS", "seed_is_accounted",
    "random.Random", "SEED_INTERVAL", "play_task", "make_agent_factory",
])
def test_the_runner_touches_no_model_no_seed_and_no_game_machinery(forbidden):
    import pathlib
    src = pathlib.Path(RQ.__file__).read_text(encoding="utf-8")
    assert forbidden not in src, forbidden


# ────────────────────── the card's frozen numbers, in code ──────────────────

def test_the_frozen_constants_match_the_card():
    assert RQ.DEPTH == 6
    assert RQ.INVOCATIONS_PER_PREFIX == 1
    assert RQ.N_PREFIXES == 8
    assert RQ.QUERY_CAP == 8 == RQ.N_PREFIXES * RQ.INVOCATIONS_PER_PREFIX
    assert RQ.REPLAY_LAUNCHES == 8
    assert RQ.HELPER_LAUNCHES == 16 == RQ.QUERY_CAP + RQ.REPLAY_LAUNCHES
    assert RQ.JAVA_PROCESSES == 17 == RQ.HELPER_LAUNCHES + 1
    assert RQ.PER_CALL_TIMEOUT_S == 120
    assert RQ.RUN_DEADLINE_S == 900
    assert RQ.SUPERVISOR_GRACE_S == 60
    assert RQ.H1_FAILED_OPENING == "o3_low"


def test_the_frozen_prefix_file_is_pinned_by_hash_and_is_the_screen_plans_openings():
    got = RQ.load_frozen_prefixes()
    assert len(got) == RQ.N_PREFIXES == 8
    openings = PLAN.load_source_plan()["openings"]
    assert [p["opening"] for p in got] == list(openings)
    assert [p["prefix"] for p in got] == [[list(m) for m in mv] for mv in openings.values()]
    assert all(p["ply"] == 6 and len(p["prefix"]) == 6 for p in got)
    failed = [p for p in got if p["h1_failed"]]
    assert [p["opening"] for p in failed] == ["o3_low"]


def test_a_tampered_prefix_file_is_refused(tmp_path):
    doc = json.loads(open(RQ.FROZEN_PREFIXES_REL).read())
    doc["prefixes"][2]["prefix"][0] = [15, 12]
    p = tmp_path / "tampered.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(RQ.RequalError, match="sha256"):
        RQ.load_frozen_prefixes(str(p))


def test_a_prefix_file_that_disagrees_with_the_PLAN_is_refused(tmp_path, monkeypatch):
    """Two frozen sources; both must say the same thing. A pinned file whose
    content drifted from the pinned plan would pass a hash check alone."""
    plan = PLAN.load_source_plan()
    plan["openings"] = dict(plan["openings"])
    plan["openings"]["o3_low"] = [[15, 12]] + plan["openings"]["o3_low"][1:]
    monkeypatch.setattr(RQ.PLAN, "load_source_plan", lambda path=None: plan)
    with pytest.raises(RQ.RequalError, match="disagree"):
        RQ.load_frozen_prefixes()


# ─────────────── a clean run: PASS, and the observation is RECORDED ─────────

def test_an_all_good_run_is_a_PASS_and_records_the_helpers_observation(wire, tmp_path):
    rep = _run(tmp_path)
    assert rep["verdict"] == "PASS" and rep["n_failures"] == 0
    assert rep["queries_spent"] == RQ.QUERY_CAP == 8
    assert len(rep["prefixes"]) == 8
    for p in rep["prefixes"]:
        q = p["query"]
        assert q["postcond"]["clean"] and q["postcond"]["prefs_ok"] is True
        assert q["postcond"]["prefs_before"] == SHA == q["postcond"]["prefs_after"]
        assert q["postcond"]["count_before"] == 527 == q["postcond"]["count_after"]
        # the diagnostic reader agrees with the parser
        assert q["prefs_observation"] == {
            "prefs_ok": True, "prefs_before": SHA, "prefs_after": SHA,
            "count_before": 527, "count_after": 527}
        assert q["abort"] is None
    assert rep["h1_failed_prefix"]["opening"] == "o3_low"
    assert rep["h1_failed_prefix"]["passed"] is True
    assert json.loads(open(tmp_path / "r.json").read())["verdict"] == "PASS"


def test_one_replay_and_one_query_per_prefix_each_with_the_frozen_timeout(wire, tmp_path):
    _run(tmp_path)
    replays = [c for c in wire["calls"] if "replay" in c["args"]]
    queries = [c for c in wire["calls"] if "query" in c["args"]]
    assert len(replays) == RQ.REPLAY_LAUNCHES == 8
    assert len(queries) == RQ.QUERY_CAP == 8
    assert all(c["kw"].get("timeout") == RQ.PER_CALL_TIMEOUT_S for c in wire["calls"])
    assert all(int(c["args"][c["args"].index("query") + 1]) == RQ.DEPTH for c in queries)
    assert not any("determinism" in c["args"] for c in wire["calls"]), "repeats>1"


def test_the_record_is_create_only_and_refuses_BEFORE_any_launch(wire, tmp_path):
    (tmp_path / "r.json").write_text("{}")
    with pytest.raises(FileExistsError):
        _run(tmp_path)
    assert wire["calls"] == [], "it ran the whole sample and then failed to write"


def test_the_parser_and_the_diagnostic_reader_must_AGREE(wire, tmp_path, monkeypatch):
    monkeypatch.setattr(RQ.A, "postcond_prefs_observation", lambda text: None)
    _void(tmp_path, "disagree")


def test_the_public_runner_requires_EXACTLY_the_frozen_eight(wire, tmp_path, monkeypatch):
    monkeypatch.setattr(RQ, "RUNTIME_REQUAL_AUTHORIZED", True)
    with pytest.raises(RQ.RequalError, match="EXACTLY"):
        RQ.run_requalification(prefixes=_prefixes(7), paths=PATHS,
                               out_path=str(tmp_path / "r.json"),
                               _compile=lambda d: {"stub": True})
    assert wire["calls"] == [] and not (tmp_path / "r.json").exists()


def _forged(mutate):
    rows = [dict(p) for p in _prefixes()]
    mutate(rows)
    return rows


def _relabel_opening(rows):
    rows[2]["opening"], rows[3]["opening"] = rows[3]["opening"], rows[2]["opening"]


def _drop_h1_flag(rows):
    rows[2]["h1_failed"] = False


def _move_h1_flag(rows):
    rows[2]["h1_failed"], rows[0]["h1_failed"] = False, True


def _reply_ply(rows):
    rows[2]["ply"] = 7


def _same_position_other_move_order(rows):
    m = rows[2]["prefix"]
    rows[2]["prefix"] = [m[2], m[1], m[0]] + m[3:]     # same pegs, same digest


def _bool_as_int(rows):
    rows[2]["h1_failed"] = 1                            # == True, but is not True


def _ply_as_str(rows):
    rows[2]["ply"] = "6"


def _extra_key(rows):
    rows[2]["note"] = "the failed one is really o1_center"


@pytest.mark.parametrize("mutate", [
    _relabel_opening, _drop_h1_flag, _move_h1_flag, _reply_ply,
    _same_position_other_move_order, _bool_as_int, _ply_as_str, _extra_key,
], ids=lambda f: f.__name__)
def test_a_DIGEST_PRESERVING_metadata_change_is_refused_by_the_public_entry(
        wire, tmp_path, monkeypatch, mutate):
    """🔴 REVIEW REPRO: every digest unchanged while `opening`, `ply` or
    `h1_failed` changed reached the stages -- the output could be mislabelled or
    lose the failed-H1 comparison. Every field of the frozen row is bound,
    TYPE-STRICTLY (`False == 0`, `6 == 6.0`); only serialization shape is
    normalised."""
    rows = _forged(mutate)
    assert [r["digest"] for r in rows] == [p["digest"] for p in _prefixes()]
    monkeypatch.setattr(RQ, "RUNTIME_REQUAL_AUTHORIZED", True)
    with pytest.raises(RQ.RequalError, match="frozen") as e:
        RQ.run_requalification(prefixes=rows, paths=PATHS,
                               out_path=str(tmp_path / "r.json"),
                               _compile=lambda d: {"stub": True})
    assert wire["calls"] == [] and not (tmp_path / "r.json").exists()


def test_serialization_SHAPE_is_the_only_normalisation_the_public_entry_allows(
        wire, tmp_path, monkeypatch):
    """Tuples for moves (what a caller builds in Python) are the frozen rows'
    lists; nothing else is coerced."""
    rows = [dict(p, prefix=[tuple(m) for m in p["prefix"]]) for p in _prefixes()]
    monkeypatch.setattr(RQ, "RUNTIME_REQUAL_AUTHORIZED", True)
    rep = RQ.run_requalification(prefixes=rows, paths=PATHS,
                                 out_path=str(tmp_path / "r.json"),
                                 _compile=lambda d: {"stub": True})
    assert rep["verdict"] == "PASS" and rep["h1_failed_prefix"]["opening"] == "o3_low"


# ──────── the observation's ABSENCE is a VOID: the wrong class ran ──────────

def test_a_reply_WITHOUT_the_observation_is_a_VOID_not_a_legacy_reading(wire, tmp_path):
    """The compiled E4Preflight (sha fa662339..., evidence 2026-09-06 compile-only)
    emits the four fields on every POSTCOND line. A reply without them did not
    come from that class: instrument, not T1j."""
    wire["query"] = lambda ours, depth: _query_out(ours, depth, post=_post(obs=False))
    out = tmp_path / "r.json"
    with pytest.raises(RQ.RequalVoidError, match="observation"):
        _run(tmp_path)
    assert not out.exists()


def test_a_PARTIAL_observation_is_a_VOID(wire, tmp_path):
    line = CLEAN_POST.format(ok="true", n=INT.QUERY_REFL_N, f=0) + \
        f" prefs_before={SHA} prefs_after={SHA} count_before=527\n"
    wire["query"] = lambda ours, depth: _query_out(ours, depth, post=line)
    with pytest.raises(RQ.RequalVoidError, match="partial"):
        _run(tmp_path)
    assert not (tmp_path / "r.json").exists()


# ───── prefs_ok=false is a FAIL RESULT, recorded through the REAL path ───────

def test_a_prefs_failure_is_a_recorded_FAIL_whose_observation_SURVIVES_truncation(
        wire, tmp_path):
    """The 2026-09-05 shape, worse: eleven FAIL lines push POSTCOND past the
    800-char excerpt. The run continues, the verdict is FAIL, and the record
    carries the helper's own before/after values -- read through T1jAgent's
    AbortError-from-HelperOutputError chain, exactly as the H1 diagnostic does."""
    fails = [f"FAIL q1: check number {i} did not hold for reason {'x' * 50}\n"
             for i in range(11)]

    def q(ours, depth):
        if [list(m) for m in ours] == _prefixes()[2]["prefix"]:
            # E4Preflight exits 3 whenever its own failures counter is set
            return 3, _query_out(ours, depth, fail_lines=fails,
                                 post=_post(ok=False, f=12, after="ERROR", ca=-1))
        return _query_out(ours, depth)

    wire["query"] = q
    rep = _run(tmp_path)
    assert rep["verdict"] == "FAIL"
    bad = rep["prefixes"][2]
    assert bad["opening"] == "o3_low" and not bad["passed"]
    q = bad["query"]
    assert any("preference" in f or "postcondition" in f for f in q["failures"]), q["failures"]
    assert q["postcond"]["prefs_ok"] is False
    assert q["postcond"]["prefs_after"] == "ERROR" and q["postcond"]["count_after"] == -1
    assert q["helper_excerpt"].endswith("...")
    assert "count_after=-1" not in q["helper_excerpt"]
    assert q["helper_prefs_observed"] == {
        "prefs_ok": False, "prefs_before": SHA, "prefs_after": "ERROR",
        "count_before": 527, "count_after": -1}
    assert q["exit_status"] == 3 and q["abort"]["phase"] == "move"
    assert rep["h1_failed_prefix"]["passed"] is False
    good = [p for p in rep["prefixes"] if p["opening"] != "o3_low"]
    assert all(p["passed"] for p in good), "the run must continue past a FAIL"


def test_a_nonzero_exit_with_a_usable_reply_is_a_FAIL_carrying_the_transcript(wire, tmp_path):
    wire["rc"] = 3
    wire["query"] = lambda ours, depth: _query_out(ours, depth, completed=False)
    rep = _run(tmp_path)
    assert rep["verdict"] == "FAIL"
    q = rep["prefixes"][0]["query"]
    assert q["exit_status"] == 3
    assert any("complete" in f for f in q["failures"])
    assert "FAIL q1: requested depth 6 completed" in q["helper_excerpt"]
    assert q["abort"]["phase"] == "move"


def test_an_illegal_move_is_a_FAIL_not_an_abort(wire, tmp_path):
    wire["query"] = lambda ours, depth: _query_out(ours, depth, move=(0, 0))
    rep = _run(tmp_path)
    assert rep["verdict"] == "FAIL"
    assert any("illegal" in f for f in rep["prefixes"][0]["query"]["failures"])


def test_a_DIFFERENT_searched_position_is_a_FAIL(wire, tmp_path):
    wire["query"] = lambda ours, depth: _query_out(
        ours, depth, dump_prefix=list(ours[:-1]) + [(20, 20)])
    rep = _run(tmp_path)
    assert rep["verdict"] == "FAIL"
    assert any("different position" in f for f in rep["prefixes"][0]["query"]["failures"])


def test_a_replay_that_does_not_bind_is_a_FAIL(monkeypatch, tmp_path, wire):
    real = subprocess.run

    def fake_run(args, **kw):
        if "replay" in args:
            return subprocess.CompletedProcess(args, 0, _replay_out([(1, 1)]), "")
        return real(args, **kw)

    monkeypatch.setattr(subprocess, "run", fake_run)
    rep = _run(tmp_path, prefixes=_prefixes(1))
    assert rep["verdict"] == "FAIL"
    assert rep["prefixes"][0]["bind"]["bound"] is False


# ───────────────────────────── VOID: the instrument ─────────────────────────

def _void(tmp_path, match, **kw):
    out = tmp_path / "r.json"
    with pytest.raises(RQ.RequalVoidError, match=match) as e:
        RQ._run_unguarded(prefixes=_prefixes(1), paths=PATHS, out_path=str(out),
                          _compile=lambda d: {"stub": True}, **kw)
    assert not out.exists(), "a VOID wrote a record"
    return str(e.value)


def test_a_REPLAY_timeout_is_a_VOID_and_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run",
                        lambda a, **k: (_ for _ in ()).throw(
                            subprocess.TimeoutExpired(a, k.get("timeout"))))
    _void(tmp_path, "replay did not answer")


def test_a_QUERY_timeout_is_a_VOID_and_writes_nothing(monkeypatch, tmp_path, wire):
    real = subprocess.run

    def fake_run(args, **kw):
        if "query" in args:
            raise subprocess.TimeoutExpired(args, kw.get("timeout"))
        return real(args, **kw)

    monkeypatch.setattr(subprocess, "run", fake_run)
    _void(tmp_path, "T1j did not answer within")


def test_a_deadline_breach_is_a_VOID_and_writes_nothing(wire, tmp_path):
    ticks = iter([0.0, 0.0, 0.0, 99999.0] + [99999.0] * 40)
    _void(tmp_path, "deadline",
          deadline=RQ.Deadline(limit_s=RQ.RUN_DEADLINE_S, clock=lambda: next(ticks)))


def test_a_reply_with_no_query_record_is_a_VOID(wire, tmp_path):
    wire["rc"] = 3
    wire["query"] = lambda ours, depth: "FAIL q1: something\n" + _post(f=1)
    _void(tmp_path, "no usable query record")


def test_a_malformed_POSTCOND_is_a_VOID_not_a_traceback(wire, tmp_path):
    wire["query"] = lambda ours, depth: _query_out(
        ours, depth, post="POSTCOND no_throw=true windows=0\n")
    msg = _void(tmp_path, "POSTCOND")
    assert "VOID" in msg


def test_a_frozen_prefix_that_replays_to_another_digest_is_a_VOID(wire, tmp_path):
    p = dict(_prefixes(1)[0]); p["digest"] = "0" * 64
    out = tmp_path / "r.json"
    with pytest.raises(RQ.RequalVoidError, match="digest"):
        RQ._run_unguarded(prefixes=[p], paths=PATHS, out_path=str(out),
                          _compile=lambda d: {"stub": True})
    assert not out.exists()


# ───────────── the OUTER supervisor kills the PROCESS GROUP ─────────────────

FAKE_WORKER = textwrap.dedent("""
    import signal, subprocess, sys, time
    signal.signal(signal.SIGTERM, signal.SIG_IGN)         # a worker that will not go quietly
    child = subprocess.Popen([sys.executable, "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)"])
    open(sys.argv[1], "w").write(str(child.pid))          # a GRANDCHILD of the supervisor
    time.sleep(300)
""")


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        # a zombie still answers signal 0; reap-free check via waitpid on a non-child fails
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    return True


def test_the_supervisor_kills_the_WHOLE_process_group_on_timeout(tmp_path):
    """🔴 THE COMPILE-ONLY DRIVER'S LIMITATION (review 2026-09-06): killing the
    Python worker does not necessarily kill its javac/java child. The runtime
    supervisor starts the worker in its OWN SESSION and kills the group."""
    pidfile = tmp_path / "grandchild.pid"
    t0 = time.monotonic()
    r = RQ.supervise([sys.executable, "-c", FAKE_WORKER, str(pidfile)],
                     timeout_s=2, kill_grace_s=1)
    # 🔴 BOUNDED IN TIME, NOT ONLY IN OUTCOME. Without this an injected defect
    # that dropped the SIGKILL step was NOT CAUGHT: the SIGTERM-ignoring worker
    # ran out its own 300 s sleep and every assertion below still held.
    assert time.monotonic() - t0 < 2 + 1 + 10, "the supervisor did not KILL"
    assert r["timed_out"] is True and r["exit_code"] == RQ.EXIT_TIMEOUT == 6
    gpid = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while _alive(gpid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(gpid), "the grandchild survived the supervisor's timeout"
    assert r["group_cleared"] is True


def test_the_supervisor_passes_a_finished_workers_exit_code_through(tmp_path):
    r = RQ.supervise([sys.executable, "-c", "import sys; sys.exit(2)"],
                     timeout_s=30, kill_grace_s=1)
    assert r == {"exit_code": 2, "timed_out": False, "group_cleared": True}


def test_the_supervisor_starts_the_worker_in_a_NEW_SESSION(tmp_path):
    """A worker in the supervisor's own group would make killpg suicide."""
    r = RQ.supervise([sys.executable, "-c",
                      "import os, sys; sys.exit(0 if os.getpgid(0) == os.getpid() else 9)"],
                     timeout_s=30, kill_grace_s=1)
    assert r["exit_code"] == 0, "the worker is not its own process-group leader"


EARLY_EXIT_WORKER = textwrap.dedent("""
    import signal, subprocess, sys
    child = subprocess.Popen([sys.executable, "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)"])
    open(sys.argv[1], "w").write(str(child.pid))
    sys.exit(0)                                          # the WORKER is done; its child is not
""")


def test_a_child_that_OUTLIVES_a_finished_worker_is_killed_and_the_group_cleared(tmp_path):
    """🔴 REVIEW REPRO: the worker exits 0 BEFORE the outer timeout, leaving a
    descendant; `supervise` saw group_cleared=False and sent NO signal. Cleanup
    must run after a normal exit exactly as after a timeout."""
    pidfile = tmp_path / "grandchild.pid"
    t0 = time.monotonic()
    r = RQ.supervise([sys.executable, "-c", EARLY_EXIT_WORKER, str(pidfile)],
                     timeout_s=30, kill_grace_s=1)
    assert time.monotonic() - t0 < 15, "cleanup must not wait for the child's own sleep"
    assert r["exit_code"] == 0 and r["timed_out"] is False
    assert r["group_cleared"] is True
    gpid = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while _alive(gpid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(gpid), "the grandchild survived a finished worker"


def test_a_group_that_CANNOT_be_cleared_is_reported_as_such(tmp_path, monkeypatch):
    """The signals are neutralised (as an unkillable descendant would neutralise
    them); the result must say the group did NOT clear, never that it did."""
    pidfile = tmp_path / "grandchild.pid"
    monkeypatch.setattr(RQ, "_killpg", lambda pgid, sig: None)
    try:
        r = RQ.supervise([sys.executable, "-c", EARLY_EXIT_WORKER, str(pidfile)],
                         timeout_s=30, kill_grace_s=0.5)
        assert r["exit_code"] == 0 and r["group_cleared"] is False
    finally:
        try:
            os.kill(int(pidfile.read_text()), signal.SIGKILL)
        except (ProcessLookupError, FileNotFoundError, ValueError):
            pass


def test_a_group_member_we_CANNOT_SIGNAL_counts_as_occupied_not_cleared(monkeypatch):
    """🔴 EPERM from `killpg(pgid, 0)` means a member exists that this process
    may not signal -- on macOS a SIGKILLed descendant re-parented to launchd,
    until it is reaped. It was raised straight through once; and reading it as
    "cleared" would report success beside a survivor. It is "occupied".
    The previous test could not see this: its survivor was our own descendant,
    which answers the probe without EPERM -- so the branch is exercised here
    with a probe that refuses."""
    calls = {"n": 0}

    def refusing_killpg(pgid, sig):
        assert sig == 0
        calls["n"] += 1
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr(RQ.os, "killpg", refusing_killpg)
    t0 = time.monotonic()
    assert RQ._group_cleared(4242, wait_s=0.3) is False
    assert calls["n"] >= 2, "it must keep polling, not decide on the first EPERM"
    assert time.monotonic() - t0 < 5


@pytest.mark.parametrize("worker_code", [0, 2, 3])
def test_main_NEVER_reports_the_workers_code_when_cleanup_failed(monkeypatch, tmp_path,
                                                                worker_code):
    """🔴 A surviving JVM beside exit 0 is a success report for a run that is
    still running. Whatever the worker said, a failed cleanup is its own code."""
    monkeypatch.setattr(RQ, "RUNTIME_REQUAL_AUTHORIZED", True)
    monkeypatch.setattr(RQ, "supervise", lambda cmd, **k: {
        "exit_code": worker_code, "timed_out": False, "group_cleared": False})
    assert RQ.main(["--out", str(tmp_path / "r.json")]) == RQ.EXIT_CLEANUP_FAILED == 8


def test_main_passes_the_workers_code_through_ONLY_when_the_group_cleared(monkeypatch,
                                                                         tmp_path):
    monkeypatch.setattr(RQ, "RUNTIME_REQUAL_AUTHORIZED", True)
    monkeypatch.setattr(RQ, "supervise", lambda cmd, **k: {
        "exit_code": 0, "timed_out": False, "group_cleared": True})
    assert RQ.main(["--out", str(tmp_path / "r.json")]) == RQ.EXIT_PASS == 0


# ────────────────────────────── exit codes mean one thing ───────────────────

def test_exit_codes_are_distinct_and_named():
    codes = {RQ.EXIT_PASS: "PASS", RQ.EXIT_FAIL: "FAIL", RQ.EXIT_VOID: "VOID",
             RQ.EXIT_UNEXPECTED: "UNEXPECTED", RQ.EXIT_UNAUTHORIZED: "UNAUTHORIZED",
             RQ.EXIT_TIMEOUT: "TIMEOUT", RQ.EXIT_REFUSED: "REFUSED",
             RQ.EXIT_CLEANUP_FAILED: "CLEANUP_FAILED"}
    assert len(codes) == 8
    assert (RQ.EXIT_PASS, RQ.EXIT_FAIL, RQ.EXIT_VOID, RQ.EXIT_UNEXPECTED,
            RQ.EXIT_UNAUTHORIZED, RQ.EXIT_TIMEOUT, RQ.EXIT_REFUSED,
            RQ.EXIT_CLEANUP_FAILED) == (0, 2, 3, 4, 5, 6, 7, 8)


@pytest.mark.parametrize("outcome,code", [
    ("PASS", 0), ("FAIL", 2), ("void", 3), ("refused", 7), ("boom", 4)])
def test_the_worker_maps_each_outcome_to_its_own_exit_code(monkeypatch, tmp_path,
                                                          outcome, code):
    """🔴 A FAIL VERDICT EXITS 2, NOT 0. The 2026-09-05 wrapper exited 0
    unconditionally and its VOID was invisible to anything that read only the
    status. Here a result and a pass are different numbers."""
    monkeypatch.setattr(RQ, "RUNTIME_REQUAL_AUTHORIZED", True)

    def fake_run(**kw):
        if outcome == "void":
            raise RQ.RequalVoidError("instrument")
        if outcome == "refused":
            raise RQ.RequalError("precondition")
        if outcome == "boom":
            raise RuntimeError("boom")
        return {"verdict": outcome, "n_failures": 0 if outcome == "PASS" else 3,
                "n_prefixes": 8, "queries_spent": 8}

    monkeypatch.setattr(RQ, "run_requalification", fake_run)
    monkeypatch.setattr(RQ, "load_frozen_prefixes", lambda p=None: [])
    rc = RQ.worker_main(["--out", str(tmp_path / "r.json"),
                         "--java", "/j", "--jar", "/x.jar", "--classes", "/c"])
    assert rc == code


def test_the_worker_refuses_before_loading_anything_while_the_gate_is_shut(
        monkeypatch, tmp_path):
    monkeypatch.setattr(RQ, "load_frozen_prefixes",
                        lambda p=None: (_ for _ in ()).throw(AssertionError("loaded")))
    assert RQ.worker_main(["--out", str(tmp_path / "r.json")]) == RQ.EXIT_UNAUTHORIZED


def test_main_refuses_without_spawning_while_the_gate_is_shut(monkeypatch, tmp_path):
    monkeypatch.setattr(RQ, "supervise",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("spawned")))
    assert RQ.main(["--out", str(tmp_path / "r.json")]) == RQ.EXIT_UNAUTHORIZED


def test_main_supervises_a_WORKER_subprocess_under_the_outer_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(RQ, "RUNTIME_REQUAL_AUTHORIZED", True)
    seen = {}

    def fake_supervise(cmd, *, timeout_s, kill_grace_s):
        seen.update(cmd=cmd, timeout_s=timeout_s, kill_grace_s=kill_grace_s)
        return {"exit_code": 2, "timed_out": False, "group_cleared": True}

    monkeypatch.setattr(RQ, "supervise", fake_supervise)
    rc = RQ.main(["--out", str(tmp_path / "r.json")])
    assert rc == 2
    assert seen["cmd"][:3] == [sys.executable, "-m", "scripts.GPU.alphazero.runtime_requalification"]
    assert "--worker" in seen["cmd"] and str(tmp_path / "r.json") in seen["cmd"]
    assert seen["timeout_s"] == RQ.RUN_DEADLINE_S + RQ.SUPERVISOR_GRACE_S == 960


def test_main_reports_TIMEOUT_when_the_supervisor_killed_the_tree(monkeypatch, tmp_path):
    monkeypatch.setattr(RQ, "RUNTIME_REQUAL_AUTHORIZED", True)
    monkeypatch.setattr(RQ, "supervise", lambda cmd, **k: {
        "exit_code": RQ.EXIT_TIMEOUT, "timed_out": True, "group_cleared": True})
    assert RQ.main(["--out", str(tmp_path / "r.json")]) == RQ.EXIT_TIMEOUT == 6
