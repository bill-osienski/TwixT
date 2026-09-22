"""The gated H4 repair qualification. NOTHING EXECUTES.

No JVM is started, no gate opened, no game played, no seed drawn, no score
computed. `subprocess.run` is intercepted at the PROCESS BOUNDARY so all of our
own code runs while nothing is spawned; the one real subprocess runs OUR CLI,
which refuses at its gate before touching anything.

🔑 WHAT THIS FILE PINS, from the frozen card:

1. The gate is shipped CLOSED and read at both public entries.
2. The DERIVED matrix is pinned in its own right -- content, order, row count,
   truncation length -- because the source file's hash authenticates the source
   SEQUENCES and says nothing about the 16 rows truncated from them.
3. 🔴 EXIT SEMANTICS IN BOTH DIRECTIONS. A legitimate native-initial reply is
   `exit 3` + `failures 1`; a completed search is `exit 0` + `failures 0`. Read
   "clean" as "exit 0 everywhere" and every native reply is rejected; ignore the
   exit code and a broken one is accepted. This is the D1 shape.
4. FOUR POSITIVE BASELINES, one per accepted classification -- not one
   aggregate valid record. `native_initial_fifth_or_more` has never been
   observed anywhere in this programme, so without its own baseline it could
   stay unreachable while every refusal control passes.
"""
import ast
import json
import pathlib
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import e4_screen_integration as INT
from scripts.GPU.alphazero import h4_repair_qualification as Q
from scripts.GPU.alphazero import t1j_adapter as A
from scripts.GPU.alphazero.game.twixt_state import TwixtState

PATHS = Q.T1jPaths(java="/nonexistent/java", jar="/nonexistent/t1j.jar",
                   classes="/nonexistent/classes", ply_cap=280)

PROC = ("PROC pid={pid} java_version=17.0.20.1 vm=OpenJDK_64-Bit_Server_VM "
        "headless=true prefs_factory=e2probe.ScratchPrefs")
MD = ("MATCHDATA pieRule={pie} xsize={xs} ysize={ys} ystarts={yst} "
      "identity=true")
POST = ("POSTCOND no_throw={nt} windows={w} frames=0 headless=true "
        "prefs_ok={ok} refl_ok=true refl_n={n} failures={f}")


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


def query_out(prefix, *, native=True, move=None, pid=4321, n_proc=1, n_md=1,
              pie=False, xs=24, ys=24, yst=True, n_records=1, dump=True,
              refl_n=None, failures=None, no_throw=True, windows=0,
              prefs_ok=True, legal=True, null_sentinel=False,
              current_max_ply=None, completed_depth=None):
    """One h4query stdout. `native` picks the native-initial telemetry shape."""
    st = _state(prefix)
    mv = move if move is not None else sorted(st.legal_moves())[0]
    x, y = A.to_t1j(*mv)
    cmp_ = current_max_ply if current_max_ply is not None else (
        0 if native else Q.SEARCHED_CURRENT_MAX_PLY)
    cd = completed_depth if completed_depth is not None else (
        -1 if native else Q.DEPTH)
    f = failures if failures is not None else (
        Q.NATIVE_FAILURES if native else Q.SEARCHED_FAILURES)
    head = "".join(PROC.format(pid=pid + i) + "\n" for i in range(n_proc))
    head += "".join(MD.format(pie=str(pie).lower(), xs=xs, ys=ys,
                              yst=str(yst).lower()) + "\n" for _ in range(n_md))
    lines = "".join(
        f"QUERY q={i + 1} requested_depth={Q.DEPTH} move_x={x} move_y={y} "
        f"to_move={A.PLAYER_TO_T1J[st.to_move]} "
        f"usealphabeta={'false' if native else 'true'} "
        f"currentMaxPly={cmp_} completed_depth={cd} "
        f"completed={'false' if native else 'true'} "
        f"legal={'true' if legal else 'false'} "
        f"null_sentinel={'true' if null_sentinel else 'false'} "
        f"moveNr={len(prefix)} eval_regime=normal elapsed_us=1000\n"
        for i in range(n_records))
    body = _block(st, list(prefix)) if dump else ""
    tail = POST.format(nt=str(no_throw).lower(), w=windows,
                       ok=str(prefs_ok).lower(),
                       n=Q.QUERY_REFL_N_OPTIN if refl_n is None else refl_n,
                       f=f) + "\n"
    return head + lines + body + tail


def replay_out(prefix, *, pid=1234, n_proc=1, blocks=None, refl_n=None,
               failures=0):
    st, moves, out = _state([]), [], []
    for mv in list(prefix) + [None]:
        out.append(_block(st, moves))
        if mv is None:
            break
        moves.append(tuple(mv))
        st = st.apply_move(tuple(mv))
    if blocks is not None:
        out = out[:blocks]
    head = "".join(PROC.format(pid=pid + i) + "\n" for i in range(n_proc))
    return (head + "".join(out)
            + POST.format(nt="true", w=0, ok="true",
                          n=Q.REPLAY_REFL_N if refl_n is None else refl_n,
                          f=failures) + "\n")


def _native_for(prefix):
    """Plies 0-3 are answered natively here; 4-5 search. Matches §4A's shape."""
    return len(prefix) <= 3


@pytest.fixture
def wire(monkeypatch):
    """Serve the repaired helper at the process boundary. Spawns nothing."""
    box = {"calls": [], "query": None, "replay": None, "rc_query": None,
           "rc_replay": 0}

    def fake_run(args, **kw):
        box["calls"].append({"args": list(args), "kw": kw})
        pairs = [tuple(int(v) for v in a.split(",")) for a in args
                 if "," in a and a.replace(",", "").isdigit()]
        prefix = [A.to_ours(x, y) for (x, y) in pairs]
        if "replay" in args:
            out = box["replay"](prefix) if box["replay"] else replay_out(prefix)
            return subprocess.CompletedProcess(args, box["rc_replay"], out, "")
        native = _native_for(prefix)
        out = (box["query"](prefix) if box["query"]
               else query_out(prefix, native=native))
        rc = box["rc_query"]
        if rc is None:
            rc = Q.NATIVE_EXIT if native else Q.SEARCHED_EXIT
        return subprocess.CompletedProcess(args, rc, out, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return box


def run(tmp_path, name="r.json", **kw):
    return Q._run_unguarded(paths=PATHS, out_path=str(tmp_path / name),
                            _compile=lambda d: {"stub": True}, **kw)


# ───────────────────────────── 1. the closed gate ────────────────────────────

def test_the_gate_is_false_as_published():
    assert Q.H4_REPAIR_QUALIFICATION_AUTHORIZED is False


def test_the_public_runner_refuses_while_the_gate_is_shut(tmp_path, wire):
    with pytest.raises(Q.H4RQError, match="UNAUTHORIZED"):
        Q.run_qualification(paths=PATHS, out_path=str(tmp_path / "r.json"))
    assert wire["calls"] == [], "it spawned a process while unauthorized"
    assert not (tmp_path / "r.json").exists()


def test_the_cli_refuses_in_a_fresh_subprocess(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "scripts.GPU.alphazero.h4_repair_qualification",
         "--out", str(tmp_path / "r.json"), "--classes", str(tmp_path / "cls")],
        capture_output=True, text=True, cwd=".", timeout=60)
    assert r.returncode == Q.EXIT_UNAUTHORIZED == 5, (r.returncode, r.stderr)
    assert "UNAUTHORIZED" in r.stderr
    assert not (tmp_path / "r.json").exists()
    assert not (tmp_path / "cls").exists(), "made a class dir while unauthorized"


def test_the_gate_is_read_at_both_public_entries():
    tree = ast.parse(pathlib.Path(Q.__file__).read_text(encoding="utf-8"))
    reads = [n for n in ast.walk(tree)
             if isinstance(n, ast.Name)
             and n.id == "H4_REPAIR_QUALIFICATION_AUTHORIZED"
             and isinstance(n.ctx, ast.Load)]
    assert len(reads) >= 2, len(reads)


def test_it_reads_no_other_experiments_gate_and_offers_no_override():
    src = pathlib.Path(Q.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for other in ("LOWPLY_QUALIFICATION_AUTHORIZED", "D1_EXECUTION_AUTHORIZED",
                  "SCREEN_AUTHORIZED", "H1_EXECUTION_AUTHORIZED",
                  "H2_EXECUTION_AUTHORIZED", "H3_STUDY_EXECUTION_AUTHORIZED",
                  "H4_4A_CHARACTERIZATION_AUTHORIZED"):
        assert other not in names, f"it names another experiment's gate: {other}"
    for token in ("environ", "getenv", "setattr", "putenv", "exec", "eval"):
        assert token not in names, f"override path: {token}"


# ────────────────────── 2. the DERIVED matrix is pinned ──────────────────────

def test_the_matrix_is_the_pinned_16_rows(tmp_path, wire):
    m = Q.derive_matrix()
    assert len(m) == Q.N_POSITIONS == 16
    assert m[0] == {"family": "empty_board", "ply": 0, "prefix": [],
                    "source": "constructed"}
    assert [r["family"] for r in m[1:6]] == ["o1_center"] * 5
    assert [r["ply"] for r in m[1:6]] == [1, 2, 3, 4, 5]
    assert Q.verify_matrix(m) == Q.MATRIX_SHA256


def test_the_pin_matches_the_CARD_and_is_recomputed_not_read_back():
    """The canonical form is the card's; the runner reproduces it."""
    import hashlib
    canon = Q._canonical(Q.derive_matrix())
    assert len(canon.encode()) == 1002, len(canon.encode())
    assert not canon.endswith("\n")
    assert canon.startswith('[{"family":"empty_board","ply":0,"prefix":[]}')
    assert hashlib.sha256(canon.encode("utf-8")).hexdigest() == Q.MATRIX_SHA256


@pytest.mark.parametrize("mutate,label", [
    (lambda m: m[:-1], "row dropped"),
    (lambda m: m + [dict(m[1])], "row duplicated"),
    (lambda m: [m[0]] + list(reversed(m[1:])), "reordered"),
])
def test_matrix_drop_duplication_and_reordering_are_REFUSED(mutate, label):
    m = mutate(Q.derive_matrix())
    with pytest.raises(Q.H4RQError, match="rows|pinned"):
        Q.verify_matrix(m)


def test_a_WRONG_TRUNCATION_LENGTH_is_refused():
    """Same 16 rows, same order, one prefix the wrong length."""
    m = Q.derive_matrix()
    victim = next(r for r in m if r["ply"] == 2)
    victim["prefix"] = victim["prefix"] + [[19, 19]]
    with pytest.raises(Q.H4RQError, match="pinned"):
        Q.verify_matrix(m)


def test_UNNESTED_source_prefixes_are_refused(tmp_path, wire):
    """Truncation is only valid because the families nest. Verified, not assumed."""
    from scripts.GPU.alphazero import h4_4a_characterization as H4A
    rows = [dict(r) for r in H4A.load_frozen_prefixes()]
    v = next(r for r in rows if r["opening"] == "o1_center" and r["ply"] == 1)
    v["prefix"] = [[19, 19]]
    with pytest.raises(Q.H4RQError, match="NOT NESTED"):
        Q.derive_matrix(rows)


def test_the_caps_are_DERIVED_from_the_matrix(tmp_path, wire):
    caps = Q.derived_caps(Q.derive_matrix())
    assert caps == {"queries": 56, "replays": 16, "total": 72}
    r = run(tmp_path)
    assert r["derived_caps"] == caps
    assert r["subprocesses_spent"] == 72 == r["n_observations"]


# ───────────── 3. EXIT SEMANTICS, both directions (the D1 shape) ─────────────

def _rec(**kw):
    base = dict(q=1, requested_depth=Q.DEPTH, move=(9, 9), to_move="Y",
                usealphabeta=False, current_max_ply=0, completed_depth=-1,
                completed=False, legal=True, null_sentinel=False, move_nr=0,
                eval_regime="normal", elapsed_us=1)
    base.update(kw)
    return A.QueryRecord(**base)


def test_a_LEGITIMATE_native_reply_is_exit_3_with_failures_1():
    """🔴 THE D1 SHAPE. The completion requirement is unchanged, so this is what
    a correct native-initial reply looks like at the process boundary."""
    assert Q.classify_reply(ply=0, rec=_rec(), exit_status=3,
                            failures=1) == Q.NATIVE_FIRST


@pytest.mark.parametrize("exit_status,failures", [(0, 1), (3, 0), (0, 0), (3, 2)])
def test_a_native_signature_with_ANY_OTHER_exit_or_failure_count_STOPS(
        exit_status, failures):
    with pytest.raises(Q.H4RQStop, match="permits ONLY"):
        Q.classify_reply(ply=0, rec=_rec(), exit_status=exit_status,
                         failures=failures)


def test_a_completed_search_requires_exit_0_and_failures_0():
    rec = _rec(usealphabeta=True, current_max_ply=Q.SEARCHED_CURRENT_MAX_PLY,
               completed=True, completed_depth=Q.DEPTH)
    assert Q.classify_reply(ply=5, rec=rec, exit_status=0,
                            failures=0) == Q.SEARCHED


@pytest.mark.parametrize("exit_status,failures", [(3, 0), (0, 1), (3, 1)])
def test_a_SEARCHED_reply_with_ANY_failure_STOPS(exit_status, failures):
    rec = _rec(usealphabeta=True, current_max_ply=Q.SEARCHED_CURRENT_MAX_PLY,
               completed=True, completed_depth=Q.DEPTH)
    with pytest.raises(Q.H4RQStop, match="required to be"):
        Q.classify_reply(ply=5, rec=rec, exit_status=exit_status,
                         failures=failures)


def test_INVALID_source_telemetry_STOPS():
    rec = _rec(usealphabeta=True, current_max_ply=3, completed=False)
    with pytest.raises(Q.H4RQStop, match="outside the frozen table"):
        Q.classify_reply(ply=1, rec=rec, exit_status=3, failures=1)


def test_a_NATIVE_reply_at_PLY_6_is_IMPOSSIBLE_and_STOPS():
    """SYNTHETIC BY NECESSITY -- the frozen matrix ends at ply 5, so this is a
    classifier control driven with constructed telemetry, not a position."""
    with pytest.raises(Q.H4RQStop, match="impossible here"):
        Q.classify_reply(ply=6, rec=_rec(), exit_status=3, failures=1)


# ───────── 4. FOUR POSITIVE BASELINES, one per accepted classification ───────
# 🔴 NOT ONE AGGREGATE VALID RECORD. `native_initial_fifth_or_more` has never
# been observed anywhere in this programme -- §4A queried three ply-5 positions
# and all three came back `searched` -- so without its own baseline it could
# stay unreachable while every refusal control above passes.

def test_baseline_native_initial_first():
    assert Q.classify_reply(ply=0, rec=_rec(), exit_status=3,
                            failures=1) == Q.NATIVE_FIRST


@pytest.mark.parametrize("ply", [1, 2, 3])
def test_baseline_native_initial_second_to_fourth(ply):
    assert Q.classify_reply(ply=ply, rec=_rec(), exit_status=3,
                            failures=1) == Q.NATIVE_SECOND_TO_FOURTH


@pytest.mark.parametrize("ply", [4, 5])
def test_baseline_native_initial_FIFTH_OR_MORE(ply):
    """🔴 THE BRANCH WITH NO PRIOR EVIDENCE. It must be reachable."""
    assert Q.classify_reply(ply=ply, rec=_rec(), exit_status=3,
                            failures=1) == Q.NATIVE_FIFTH_OR_MORE


@pytest.mark.parametrize("ply", [3, 4, 5])
def test_baseline_searched(ply):
    """🔴 PLIES 0, 1 AND 2 ARE EXCLUDED -- under the frozen 24x24 no-pie
    configuration InitialMoves answers there natively, so search is impossible
    (card §4 amendment). An earlier version of this test BLESSED ply 0, and the
    version after it still blessed ply 2."""
    rec = _rec(usealphabeta=True, current_max_ply=Q.SEARCHED_CURRENT_MAX_PLY,
               completed=True, completed_depth=Q.DEPTH)
    assert Q.classify_reply(ply=ply, rec=rec, exit_status=0,
                            failures=0) == Q.SEARCHED


def test_all_four_classifications_are_reachable_and_distinct():
    """CLEAN BASELINE over the whole classifier: four accepted outcomes, and no
    fifth. A classifier tightened until nothing passes satisfies every refusal
    control above while making the qualification unusable."""
    native = _rec()
    searched = _rec(usealphabeta=True, current_max_ply=Q.SEARCHED_CURRENT_MAX_PLY,
                    completed=True, completed_depth=Q.DEPTH)
    got = {
        Q.classify_reply(ply=0, rec=native, exit_status=3, failures=1),
        Q.classify_reply(ply=2, rec=native, exit_status=3, failures=1),
        Q.classify_reply(ply=4, rec=native, exit_status=3, failures=1),
        Q.classify_reply(ply=5, rec=searched, exit_status=0, failures=0),
    }
    assert got == {Q.NATIVE_FIRST, Q.NATIVE_SECOND_TO_FOURTH,
                   Q.NATIVE_FIFTH_OR_MORE, Q.SEARCHED}
    assert len(got) == 4


def test_a_FULL_CLEAN_RUN_reaches_the_clean_verdict(tmp_path, wire):
    """The whole path, end to end, on a valid observation set."""
    r = run(tmp_path)
    assert r["verdict"] == "CLEAN"
    assert r["matrix_sha256"] == Q.MATRIX_SHA256
    assert r["n_positions"] == 16 and r["n_observations"] == 72
    assert r["refl_n_optin"] == 4 and r["refl_n_default"] == 3
    assert r["refl_n_replay"] == 1
    # plies 0-3 native, 4-5 searched, per the fixture's shape
    assert r["source_counts"] == {Q.NATIVE_FIRST: 5,
                                  Q.NATIVE_SECOND_TO_FOURTH: 45,
                                  Q.SEARCHED: 6}


# ──────────────────── 5. the remaining refusal controls ─────────────────────

def test_the_OPT_IN_mode_is_used_and_the_DEFAULT_argv_is_UNCHANGED(tmp_path, wire):
    run(tmp_path)
    qcalls = [c for c in wire["calls"] if "h4query" in c["args"]]
    assert len(qcalls) == 56, len(qcalls)
    assert not [c for c in wire["calls"] if "query" in c["args"]], \
        "a default `query` argv leaked into the opt-in run"
    # and a default adapter call still says `query`
    seen = {}
    import subprocess as sp
    def cap(args, **kw):
        seen["args"] = list(args)
        return sp.CompletedProcess(args, 0, query_out([], native=True), "")
    orig = sp.run
    sp.run = cap
    try:
        A.query([], depth=6, java="j", jar="ja", classes="c", timeout_s=1)
    except Exception:
        pass
    finally:
        sp.run = orig
    assert "query" in seen["args"] and "h4query" not in seen["args"]


def test_MISSING_PROC_is_refused(tmp_path, wire):
    wire["query"] = lambda p: query_out(p, native=_native_for(p), n_proc=0)
    with pytest.raises(Q.H4RQStop, match="PROC lines"):
        run(tmp_path)


def test_MULTIPLE_PROC_is_refused(tmp_path, wire):
    wire["replay"] = lambda p: replay_out(p, n_proc=2)
    with pytest.raises(Q.H4RQStop, match="PROC lines"):
        run(tmp_path)


@pytest.mark.parametrize("kw,label", [
    ({"pie": True}, "mdPieRule true"),
    ({"xs": 19}, "wrong xsize"),
    ({"ys": 19}, "wrong ysize"),
    ({"yst": False}, "mdYstarts false"),
])
def test_MATCHDATA_readback_mismatch_is_refused(tmp_path, wire, kw, label):
    wire["query"] = lambda p: query_out(p, native=_native_for(p), **kw)
    with pytest.raises(Q.H4RQStop, match="did not read back as frozen"):
        run(tmp_path)


def test_a_MISSING_matchdata_line_is_refused(tmp_path, wire):
    wire["query"] = lambda p: query_out(p, native=_native_for(p), n_md=0)
    with pytest.raises(Q.H4RQStop, match="MATCHDATA lines"):
        run(tmp_path)


@pytest.mark.parametrize("kw,label", [
    ({"no_throw": False}, "threw"),
    ({"windows": 1}, "window opened"),
    ({"prefs_ok": False}, "preferences disturbed"),
])
def test_DIRTY_SAFETY_STATE_is_refused(tmp_path, wire, kw, label):
    wire["query"] = lambda p: query_out(p, native=_native_for(p), **kw)
    with pytest.raises(Q.H4RQStop, match="safety surface not clean"):
        run(tmp_path)


def test_OPT_IN_REFLECTION_DRIFT_is_refused(tmp_path, wire):
    """The opt-in path must report exactly four."""
    wire["query"] = lambda p: query_out(p, native=_native_for(p),
                                        refl_n=Q.QUERY_REFL_N_DEFAULT)
    with pytest.raises(Q.H4RQStop, match="refl_n=3, expected exactly 4"):
        run(tmp_path)


def test_REPLAY_REFLECTION_DRIFT_is_refused(tmp_path, wire):
    """The replay path must stay at one.

    ⚠ RENAMED. This was called "DEFAULT PATH reflection drift", but mutating
    the REPLAY fixture proves only that replay stays at 1 -- it says nothing
    about whether the DEFAULT QUERY path still demands 3. That claim now has
    its own fixture-driven test below.
    """
    wire["replay"] = lambda p: replay_out(p, refl_n=Q.QUERY_REFL_N_OPTIN)
    with pytest.raises(Q.H4RQStop, match="refl_n=4, expected exactly 1"):
        run(tmp_path)


def test_the_DEFAULT_QUERY_PATH_REJECTS_refl_n_other_than_three():
    """🔴 THE REAL DEFAULT-PATH CONTROL, driven through the REAL checker.

    The qualification only ever calls the opt-in path, so nothing in this
    runner exercises the default query contract. It is exercised HERE, against
    `INT.check_postcond` with `INT.QUERY_REFL_N` -- the exact call every default
    caller (E4, L0, H1, H2, H3, D1) makes.
    """
    from scripts.GPU.alphazero.e4_screen_runner import AbortError
    good = POST.format(nt="true", w=0, ok="true", n=INT.QUERY_REFL_N, f=0) + "\n"
    assert INT.check_postcond(good, expected_refl=INT.QUERY_REFL_N,
                              where="default", phase="move")
    for drifted in (Q.QUERY_REFL_N_OPTIN, 2, 0):
        bad = POST.format(nt="true", w=0, ok="true", n=drifted, f=0) + "\n"
        with pytest.raises(AbortError):
            INT.check_postcond(bad, expected_refl=INT.QUERY_REFL_N,
                               where="default", phase="move")


def test_the_DEFAULT_QUERY_CONTRACT_IS_STILL_THREE():
    """The whole point of opt-in: a default caller's expectation is unmoved."""
    assert Q.QUERY_REFL_N_DEFAULT == INT.QUERY_REFL_N == 3
    assert Q.QUERY_REFL_N_OPTIN == 4
    assert Q.REPLAY_REFL_N == INT.REPLAY_REFL_N == 1


def test_an_ILLEGAL_or_NULL_move_is_refused(tmp_path, wire):
    wire["query"] = lambda p: query_out(p, native=_native_for(p),
                                        null_sentinel=True)
    with pytest.raises(Q.H4RQStop, match="null sentinel"):
        run(tmp_path)


def test_a_move_ILLEGAL_IN_OUR_ENGINE_is_refused(tmp_path, wire):
    wire["query"] = lambda p: query_out(p, native=_native_for(p), move=(0, 0))
    with pytest.raises(Q.H4RQStop, match="illegal in OUR engine"):
        run(tmp_path)


def test_a_DIVERGENT_dump_is_refused(tmp_path, wire):
    def wrong(prefix):
        good = query_out(prefix, native=_native_for(prefix))
        other = query_out(list(prefix) + [(20, 20)], native=True)
        return good.split("  PEGS")[0] + "  PEGS" + other.split("  PEGS", 1)[1]
    wire["query"] = wrong
    with pytest.raises(Q.H4RQStop, match="diverges"):
        run(tmp_path)


def test_a_TRUNCATED_replay_is_refused(tmp_path, wire):
    wire["replay"] = lambda p: (replay_out(p, blocks=1) if p else replay_out(p))
    with pytest.raises(Q.H4RQStop, match="ply blocks"):
        run(tmp_path)


def test_a_TIMEOUT_is_VOID_not_a_stop(tmp_path, wire, monkeypatch):
    def boom(args, **kw):
        raise subprocess.TimeoutExpired(args, Q.PER_CALL_TIMEOUT_S)
    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(Q.H4RQVoidError, match="VOID"):
        run(tmp_path)


def test_the_record_is_create_only_and_carries_its_scope(tmp_path, wire):
    r = run(tmp_path, name="once.json")
    assert "No game was played" in r["scope"]
    assert "NOT a probability estimate" in r["scope"]
    with pytest.raises(FileExistsError):
        run(tmp_path, name="once.json")


def test_it_plays_no_game_draws_no_seed_and_computes_no_score():
    tree = ast.parse(pathlib.Path(Q.__file__).read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {n.name for n in ast.walk(tree)
              if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for token in ("pair_score", "winner", "t1j_points", "half_width", "Random",
                  "seed", "seeds", "SEED_INTERVAL", "evaluator", "mcts",
                  "T1jAgent", "make_binder"):
        assert token not in names, f"it strays into: {token}"


# ═══════════ 6. CONTROLS ADDED AFTER REVIEW OF f4499c5 ══════════════════════

# ── search is IMPOSSIBLE at plies 0, 1 AND 2 (card §4 amendment) ──
# THREE SEPARATE controls, one per ply, so dropping any one ply from the rule
# fails a control of its own. Each drives an otherwise CLEAN search -- exit 0,
# failures 0 -- because that is exactly the reply that must be refused.

def test_SEARCH_AT_PLY_ZERO_is_refused():
    """🔴 DERIVED, not assumed: `firstMove()` has no `aconst_null` and its
    no-pie branch always builds `new Move(x,y)` with x = 12 + nextInt(6) - 3
    on a 24-wide board, i.e. 9..14, so `initialMove()`'s `getX() >= 0` gate
    always passes it and no search can run. ⚠ This used to add "a search at
    ply 0 means the injection did not take" -- NOT derived: the MATCHDATA
    readback is verified before classification, and a missing injection makes
    `firstMove()` throw, not search."""
    rec = _rec(usealphabeta=True, current_max_ply=Q.SEARCHED_CURRENT_MAX_PLY,
               completed=True, completed_depth=Q.DEPTH)
    with pytest.raises(Q.H4RQStop,
                       match="ply 0: a completed search is impossible"):
        Q.classify_reply(ply=0, rec=rec, exit_status=0, failures=0)


def test_SEARCH_AT_PLY_ONE_is_refused():
    """🔴 DERIVED from the pinned jar: at moveNr 1 `secondToFourthMove()` takes
    `292: if_icmpne 499`, which jumps OVER its only -1,-1 stores (495-498), so
    the move keeps coordinates computed from a placed move -- never negative --
    and `initialMove()` returns it. The first amendment left this ply
    permissive, having read that jump as passing THROUGH 495-498."""
    rec = _rec(usealphabeta=True, current_max_ply=Q.SEARCHED_CURRENT_MAX_PLY,
               completed=True, completed_depth=Q.DEPTH)
    with pytest.raises(Q.H4RQStop,
                       match="ply 1: a completed search is impossible"):
        Q.classify_reply(ply=1, rec=rec, exit_status=0, failures=0)


def test_SEARCH_AT_PLY_TWO_is_refused():
    """🔴 DERIVED from the pinned jar: at moveNr 2 `secondToFourthMove()` takes
    `442: if_icmpne 499`, jumping OVER the -1,-1 stores exactly as ply 1 does
    at 292. The first amendment ACCEPTED a search here, and the `searched`
    positive baseline blessed it."""
    rec = _rec(usealphabeta=True, current_max_ply=Q.SEARCHED_CURRENT_MAX_PLY,
               completed=True, completed_depth=Q.DEPTH)
    with pytest.raises(Q.H4RQStop,
                       match="ply 2: a completed search is impossible"):
        Q.classify_reply(ply=2, rec=rec, exit_status=0, failures=0)


@pytest.mark.parametrize("ply", [3, 4, 5])
def test_search_at_plies_3_to_5_REMAINS_LEGITIMATE(ply):
    """CONTROL THAT THE RULE DOES NOT SPREAD past ply 2. At ply 3 -- the only
    moveNr that reaches the -1,-1 stores -- `secondToFourthMove()` may return
    that sentinel, and at plies 4-5 `fifthOrMoreMove()` may return null.
    `initialMove()` turns either into null and the search runs, so BOTH
    outcomes are legitimate and neither may be refused."""
    rec = _rec(usealphabeta=True, current_max_ply=Q.SEARCHED_CURRENT_MAX_PLY,
               completed=True, completed_depth=Q.DEPTH)
    assert Q.classify_reply(ply=ply, rec=rec, exit_status=0,
                            failures=0) == Q.SEARCHED


# ── query-record coherence: the QUERY line must describe the position sent ──

def test_a_WRONG_moveNr_in_the_query_record_is_refused(tmp_path, wire):
    """The dump is checked separately; the QUERY line is a DIFFERENT line and
    was unchecked, so a record about another move number would be classified as
    though it were about this position."""
    def wrong(prefix):
        out = query_out(prefix, native=_native_for(prefix))
        return out.replace(f"moveNr={len(prefix)}", f"moveNr={len(prefix) + 7}")
    wire["query"] = wrong
    with pytest.raises(Q.H4RQStop, match="moveNr"):
        run(tmp_path)


def test_a_WRONG_to_move_in_the_query_record_is_refused(tmp_path, wire):
    def wrong(prefix):
        st = _state(prefix)
        ours = A.PLAYER_TO_T1J[st.to_move]
        other = "X" if ours == "Y" else "Y"
        return query_out(prefix, native=_native_for(prefix)).replace(
            f"to_move={ours}", f"to_move={other}", 1)
    wire["query"] = wrong
    with pytest.raises(Q.H4RQStop, match="to_move"):
        run(tmp_path)


def test_a_WRONG_query_index_is_refused(tmp_path, wire):
    wire["query"] = lambda p: query_out(p, native=_native_for(p)).replace(
        "QUERY q=1 ", "QUERY q=2 ", 1)
    with pytest.raises(Q.H4RQStop, match="query index"):
        run(tmp_path)


def test_MATCHDATA_identity_false_is_refused(tmp_path, wire):
    """The helper emits `identity` specifically to prove `getMatchData()`
    returned the injected object. Values agreeing is not the same thing."""
    wire["query"] = lambda p: query_out(p, native=_native_for(p)).replace(
        "identity=true", "identity=false")
    with pytest.raises(Q.H4RQStop, match="identity=false"):
        run(tmp_path)


# ── the destination is claimed BEFORE the run, and a STOP leaves a record ──

def test_an_OCCUPIED_destination_is_refused_BEFORE_any_subprocess(tmp_path, wire):
    """🔴 It used to be claimed after all 72 subprocesses, so an occupied
    destination was discovered only once the run had been spent."""
    (tmp_path / "taken.json").write_text("{}")
    with pytest.raises(FileExistsError):
        run(tmp_path, name="taken.json")
    assert wire["calls"] == [], "it spawned processes before claiming the path"


def test_a_STOP_LEAVES_A_DURABLE_RECORD(tmp_path, wire):
    """🔴 RETRIES ARE FORBIDDEN, so the one observation of a failure is the only
    one there will ever be. It used to be lost entirely."""
    wire["query"] = lambda p: query_out(p, native=_native_for(p),
                                        null_sentinel=True)
    with pytest.raises(Q.H4RQStop):
        run(tmp_path, name="stopped.json")
    rec = json.loads((tmp_path / "stopped.json").read_text())
    assert rec["verdict"] == "STOP"
    assert "null sentinel" in rec["stop_reason"]
    assert rec["stop_where"] and rec["stop_position"]["ply"] == 0
    assert rec["stop_stdout"].startswith("PROC ")
    assert rec["matrix_sha256"] == Q.MATRIX_SHA256
    assert "no retry" in rec["scope"]


def test_the_stop_record_keeps_the_observations_completed_before_it(tmp_path, wire):
    """A stop partway through must not discard what was already observed --
    and the record must say they are not a partial qualification."""
    def late(prefix):
        # ply 0 is fine; the first o1_center position stops
        if len(prefix) == 1:
            return query_out(prefix, native=True, null_sentinel=True)
        return query_out(prefix, native=_native_for(prefix))
    wire["query"] = late
    with pytest.raises(Q.H4RQStop):
        run(tmp_path, name="partial.json")
    rec = json.loads((tmp_path / "partial.json").read_text())
    assert rec["observations_completed"] == len(rec["observations"]) > 0
    assert rec["subprocesses_spent"] < rec["derived_caps"]["total"]
    assert "not a partial qualification" in rec["scope"]


# ── boolean parsing is not fail-open ──

@pytest.mark.parametrize("field", ["pieRule", "ystarts", "identity"])
def test_MATCHDATA_booleans_require_exact_tokens(field):
    """`kv[k] == "true"` is FAIL-OPEN: `pieRule=garbage` reads as False and
    passes a check meaning "the swap rule is off"."""
    line = ("MATCHDATA pieRule=false xsize=24 ysize=24 ystarts=true "
            "identity=true")
    bad = line.replace(f"{field}=false", f"{field}=garbage").replace(
        f"{field}=true", f"{field}=garbage")
    with pytest.raises(ValueError, match="not exactly"):
        A.parse_matchdata(bad)
    assert A.parse_matchdata(line), "the clean line must still parse"


@pytest.mark.parametrize("field", ["usealphabeta", "completed", "legal",
                                   "null_sentinel"])
def test_QUERY_booleans_the_classifier_uses_require_exact_tokens(field):
    good = query_out([], native=True)
    line = [l for l in good.splitlines() if l.startswith("QUERY ")][0]
    import re
    bad = re.sub(rf"\b{field}=(true|false)\b", f"{field}=garbage", line)
    with pytest.raises(ValueError, match="not exactly"):
        A.parse_queries(bad)
    assert A.parse_queries(line), "the clean line must still parse"
