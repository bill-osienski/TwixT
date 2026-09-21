"""The gated H4 §4A characterization runner. NOTHING EXECUTES.

No JVM is started, no model loaded, no seed drawn or registered, no game played,
no score computed. `subprocess.run` is intercepted at the PROCESS BOUNDARY, so
the whole of our own code runs while nothing is ever spawned -- and the one test
that runs a real subprocess runs OUR CLI, which refuses at its gate before
touching anything.

🔑 WHAT THESE TESTS EXIST TO PIN.

1. The gate is shipped CLOSED and is READ at both public entries. A fixture that
   flips an execution gate IS the gate failing, so the machinery is exercised
   through `_run_unguarded` and the gate is proven separately.
2. A SHORTFALL BY T1J IS A RESULT. A non-completing query that exits 3 must be
   RECORDED and classified, never raised. That is the defect D1 paid for.
3. The frozen matrix cannot be widened, and §4A cannot reach §4B: it never
   relaxes `completed`, never imports `T1jAgent`, and computes no score.
4. Every one of the four frozen branches fires on constructed output, AND each
   has a negative control proving it does NOT fire on clean output.
"""
import ast
import json
import pathlib
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import e4_screen_integration as INT
from scripts.GPU.alphazero import h4_4a_characterization as H4A
from scripts.GPU.alphazero import t1j_adapter as A
from scripts.GPU.alphazero.game.twixt_state import TwixtState

PATHS = H4A.T1jPaths(java="/nonexistent/java", jar="/nonexistent/t1j.jar",
                     classes="/nonexistent/classes", ply_cap=280)

PROC = ("PROC pid={pid} java_version=17.0.20.1 vm=OpenJDK_64-Bit_Server_VM "
        "headless=true prefs_factory=e2probe.ScratchPrefs")
POST = ("POSTCOND no_throw=true windows=0 frames=0 headless=true prefs_ok={ok} "
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


def replay_out(prefix, *, pid=1234, n_proc=1, blocks=None):
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
    return head + "".join(out) + POST.format(ok="true", n=INT.REPLAY_REFL_N, f=0) + "\n"


def query_out(prefix, *, completed=True, dump=True, pid=4321, n_proc=1,
              post_ok=True, refl_n=None, n_records=1, move=None):
    st = _state(prefix)
    mv = move if move is not None else sorted(st.legal_moves())[0]
    x, y = A.to_t1j(*mv)
    head = "".join(PROC.format(pid=pid + i) + "\n" for i in range(n_proc))
    lines = "".join(
        f"QUERY q={i + 1} requested_depth={H4A.DEPTH} move_x={x} move_y={y} "
        f"to_move={A.PLAYER_TO_T1J[st.to_move]} "
        f"usealphabeta={'true' if completed else 'false'} "
        f"currentMaxPly={H4A.DEPTH if completed else 0} "
        f"completed_depth={H4A.DEPTH if completed else -1} "
        f"completed={'true' if completed else 'false'} "
        f"legal=true null_sentinel=false moveNr={len(prefix)} "
        f"eval_regime={'normal' if completed else 'early_moveNr_lt_8'} "
        f"elapsed_us=1000\n"
        for i in range(n_records))
    body = _block(st, list(prefix)) if dump else ""
    tail = "" if completed else f"FAIL q1: requested depth {H4A.DEPTH} completed\n"
    tail += POST.format(ok="true" if post_ok else "false",
                        n=INT.QUERY_REFL_N if refl_n is None else refl_n,
                        f=0 if completed and post_ok else 1) + "\n"
    return head + lines + body + tail


@pytest.fixture
def wire(monkeypatch):
    """Serve replays and query replies at the process boundary. Spawns nothing."""
    box = {"calls": [], "query": None, "replay": None, "rc_query": 0, "rc_replay": 0}

    def fake_run(args, **kw):
        box["calls"].append({"args": list(args), "kw": kw})
        moves = [tuple(int(v) for v in a.split(",")) for a in args
                 if "," in a and a.replace(",", "").isdigit()]
        prefix = [A.to_ours(x, y) for (x, y) in moves]
        if "replay" in args:
            out = (box["replay"](prefix) if box["replay"] else replay_out(prefix))
            return subprocess.CompletedProcess(args, box["rc_replay"], out, "")
        out = (box["query"](prefix) if box["query"] else query_out(prefix))
        return subprocess.CompletedProcess(args, box["rc_query"], out, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return box


def frozen():
    return H4A.load_frozen_prefixes()


def run(tmp_path, prefixes=None, name="r.json", **kw):
    return H4A._run_unguarded(prefixes=prefixes if prefixes is not None else frozen(),
                              paths=PATHS, out_path=str(tmp_path / name),
                              _compile=lambda d: {"stub": True}, **kw)


# ───────────────────────────────── the gate ─────────────────────────────────

def test_the_gate_is_false_as_published():
    assert H4A.H4_4A_CHARACTERIZATION_AUTHORIZED is False


def test_the_public_runner_refuses_while_the_gate_is_shut(tmp_path, wire):
    with pytest.raises(H4A.H4A4Error, match="UNAUTHORIZED"):
        H4A.run_characterization(prefixes=frozen(), paths=PATHS,
                                 out_path=str(tmp_path / "r.json"))
    assert wire["calls"] == [], "it spawned a process while unauthorized"
    assert not (tmp_path / "r.json").exists()


def test_the_cli_refuses_in_a_fresh_subprocess(tmp_path):
    """The CLI is qualified as a REAL subprocess, not by calling main() in-process."""
    r = subprocess.run(
        [sys.executable, "-m", "scripts.GPU.alphazero.h4_4a_characterization",
         "--out", str(tmp_path / "r.json")],
        capture_output=True, text=True, cwd=".", timeout=60)
    assert r.returncode == H4A.EXIT_UNAUTHORIZED == 5, (r.returncode, r.stderr)
    assert "UNAUTHORIZED" in r.stderr
    assert not (tmp_path / "r.json").exists()


def test_the_gate_is_read_at_both_public_entries():
    tree = ast.parse(pathlib.Path(H4A.__file__).read_text(encoding="utf-8"))
    reads = [n for n in ast.walk(tree)
             if isinstance(n, ast.Name)
             and n.id == "H4_4A_CHARACTERIZATION_AUTHORIZED"
             and isinstance(n.ctx, ast.Load)]
    assert len(reads) >= 2, len(reads)


# A SOURCE GREP IS NOT A TEST. The first version of the four tests below matched
# this module's own DOCSTRING, which discusses `T1jAgent` in prose precisely to
# say it must never be touched. Text search cannot tell an explanation from a
# call. These walk the AST and look only at EXECUTABLE code.

def _code_names(module) -> set:
    """Every identifier reachable in executable code. Docstrings excluded."""
    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, ast.alias):
            names.update(filter(None, (n.name, n.asname)))
            names.update(n.name.split("."))
        elif isinstance(n, ast.ImportFrom) and n.module:
            names.update(n.module.split("."))
        elif isinstance(n, ast.keyword) and n.arg:
            names.add(n.arg)
        elif isinstance(n, ast.arg):
            names.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # A DEFINITION counts too: §4A growing its own `T1jAgent` would be
            # the same violation as importing one, and a Name-only walk is blind
            # to it.
            names.add(n.name)
    return names


def _string_constants(module) -> set:
    """Every string literal EXCEPT docstrings and bare string expressions."""
    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    bare = {id(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)}
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in bare}


def test_the_ast_helpers_can_actually_see_a_violation():
    """CLEAN-BASELINE CONTROL. A checker that finds nothing because it looks at
    nothing passes every one of the four tests below for free."""
    names = _code_names(INT)
    assert "T1jAgent" in names and "make_binder" in names, (
        "the AST walker cannot see identifiers it is supposed to catch")
    assert "argparse" in _code_names(H4A)
    assert "--out" in _string_constants(H4A)
    assert "H4 §4A is UNAUTHORIZED" not in _string_constants(INT)


def test_this_module_reads_NO_OTHER_experiments_gate():
    """One gate must never be openable by opening another."""
    names = _code_names(H4A)
    for other in ("LOWPLY_QUALIFICATION_AUTHORIZED", "D1_AUTHORIZED",
                  "SCREEN_AUTHORIZED", "H1_MATCH_AUTHORIZED", "H2_MATCH_AUTHORIZED",
                  "H3_STUDY_AUTHORIZED", "H3_GENERATION_AUTHORIZED",
                  "MATCH_AUTHORIZED", "REQUALIFICATION_AUTHORIZED"):
        assert other not in names, f"§4A names another experiment's gate: {other}"


def test_no_override_reaches_the_gate():
    """Not argv, not the environment, not a config file, not monkeypatching."""
    names = _code_names(H4A)
    for token in ("environ", "getenv", "setattr", "putenv", "exec", "eval",
                  "importlib", "__dict__"):
        assert token not in names, f"§4A exposes an override path: {token}"
    for flag in _string_constants(H4A):
        assert "authoriz" not in flag.lower() or not flag.startswith("-"), (
            f"§4A exposes an authorization flag: {flag}")


# ────────────────────── §4A cannot become §4B or a match ─────────────────────

def test_it_never_touches_the_adapters_acceptance_path():
    """§4A characterizes; it must not reach the refusal sites at all."""
    names = _code_names(H4A)
    for token in ("T1jAgent", "make_agent_factory", "make_binder", "agent_factory"):
        assert token not in names, f"§4A reaches into the acceptance path: {token}"


def test_it_computes_no_score_and_plays_no_game():
    names = _code_names(H4A)
    for token in ("pair_score", "winner", "t1j_points", "hoeffding", "half_width",
                  "SeededReferenceAgent", "load_reference_evaluator", "mcts",
                  "evaluator", "apply_move_and_score", "summarise"):
        assert token not in names, f"§4A strays toward a game or a score: {token}"


def test_it_reserves_no_seed():
    names = _code_names(H4A)
    for token in ("SEED_INTERVAL", "seed_is_accounted", "rng_stream_seeds",
                  "Random", "seed", "seeds"):
        assert token not in names, f"§4A touches seed machinery: {token}"


# ─────────────────────────── the frozen matrix ───────────────────────────────

def test_the_matrix_is_the_frozen_one(tmp_path, wire):
    m = H4A.build_matrix(frozen())
    assert [p["ply"] for p in m][0] == 0
    assert sorted({p["ply"] for p in m}) == sorted(H4A.MATRIX_PLIES) == [0, 1, 3, 5]
    assert len(m) == H4A.N_POSITIONS_EXPECTED == 10
    assert m[0]["task_id"] == "empty_board" and m[0]["prefix"] == []


def test_the_matrix_refuses_a_prefix_set_that_is_not_the_frozen_plies():
    """NEGATIVE CONTROL: a short input cannot silently shrink the question."""
    only_ply_1 = [p for p in frozen() if p["ply"] == 1]
    with pytest.raises(H4A.H4A4Error, match="frozen"):
        H4A.build_matrix(only_ply_1)


def test_the_frozen_prefixes_are_pinned_by_content(tmp_path):
    """NEGATIVE CONTROL: relocating the file is allowed, changing it is not."""
    good = pathlib.Path(H4A.FROZEN_PREFIXES_REL).read_bytes()
    moved = tmp_path / "moved.json"
    moved.write_bytes(good)
    assert H4A.load_frozen_prefixes(str(moved))          # relocation is fine

    tampered = tmp_path / "tampered.json"
    d = json.loads(good)
    d["prefixes"][0]["ply"] = 2
    tampered.write_text(json.dumps(d))
    with pytest.raises(H4A.H4A4Error, match="sha256"):
        H4A.load_frozen_prefixes(str(tampered))


def test_the_budget_is_derived_from_the_matrix_not_the_constant(tmp_path, wire):
    r = run(tmp_path)
    assert r["subprocess_cap"] == len(r["positions"]) * 2 == 20
    assert r["subprocesses_spent"] == r["n_observations"] == 20


# ──────────────────────── the empty-board argument tail ──────────────────────

def test_ply_zero_invokes_the_jvm_WITH_AN_EMPTY_POSITION_TAIL(tmp_path, wire):
    """The mechanism §4A exists to measure, asserted at the process boundary."""
    run(tmp_path)
    q = [c for c in wire["calls"] if "query" in c["args"]][0]
    r = [c for c in wire["calls"] if "replay" in c["args"]][0]
    assert q["args"][-1] == str(H4A.DEPTH), q["args"][-3:]
    assert q["args"][q["args"].index("query") + 1] == str(H4A.DEPTH)
    assert not [a for a in q["args"] if "," in a and a.replace(",", "").isdigit()]
    assert r["args"][-1] == str(PATHS.ply_cap), r["args"][-3:]
    assert not [a for a in r["args"] if "," in a and a.replace(",", "").isdigit()]


def test_both_helper_mains_are_exercised(tmp_path, wire):
    """The query and replay paths are DIFFERENT Java mains; neither answers for
    the other, so §4A must reach both."""
    mains = {a for c in wire["calls"] for a in c["args"]
             if a.startswith("net.schwagereit.t1j.")}
    run(tmp_path)
    mains = {a for c in wire["calls"] for a in c["args"]
             if a.startswith("net.schwagereit.t1j.")}
    assert mains == {A.PREFLIGHT_MAIN, A.HELPER_MAIN}


# ───────────────────── what the record must actually carry ───────────────────

def test_it_records_complete_stdout_verbatim_and_the_PROC_records(tmp_path, wire):
    r = run(tmp_path)
    for o in r["observations"]:
        assert o["stdout"].startswith("PROC "), "stdout is not verbatim"
        assert o["n_procs"] == 1 and o["procs"][0]["java_version"] == "17.0.20.1"
        assert o["procs"][0]["pid"] > 0
        assert o["postcond"]["rows"] and o["postcond"]["safety_clean"]
    # query and replay are distinct processes, and the record says which is which
    roles = [o["role"] for o in r["observations"]]
    assert roles[:2] == ["query", "replay"] and len(set(roles)) == 2


def test_the_record_is_create_only(tmp_path, wire):
    run(tmp_path, name="once.json")
    with pytest.raises(FileExistsError):
        run(tmp_path, name="once.json")


def test_the_record_carries_its_own_scope(tmp_path, wire):
    r = run(tmp_path)
    assert "plies 2 or 4" in r["scope"] and "No game was played" in r["scope"]
    assert r["stage"] == "h4_4A_raw_capability_characterization"


def test_the_colour_role_is_derived_from_the_position_not_a_label(tmp_path, wire):
    """Ply 0 is the ONLY red-to-move position in the frozen matrix."""
    r = run(tmp_path)
    by = {(o["ply"], o["role"]): o for o in r["observations"]}
    assert by[(0, "query")]["t1j_colour_role"] == "red"
    for ply in (1, 3, 5):
        assert by[(ply, "query")]["t1j_colour_role"] == "black"


# ─────────── a shortfall by T1j is a RESULT, not an abort (the D1 lesson) ─────

def test_a_non_completing_query_at_exit_3_is_RECORDED_not_raised(tmp_path, wire):
    wire["query"] = lambda p: query_out(p, completed=False)
    wire["rc_query"] = 3
    r = run(tmp_path)
    q = [o for o in r["observations"] if o["role"] == "query"]
    assert all(o["return_code"] == 3 for o in q)
    assert all(o["path"] == "native_low_ply_fallback" for o in q)
    assert all(o["record"]["completed"] is False for o in q)
    assert r["verdict"] == H4A.BRANCH_PROCEED, r["branch_reasons"]


# ───────────────────────── the four frozen branches ──────────────────────────

def test_branch_PROCEED_on_a_clean_fallback_with_one_coherent_dump(tmp_path, wire):
    wire["query"] = lambda p: query_out(p, completed=False, dump=True)
    wire["rc_query"] = 3
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_PROCEED
    assert r["branches_fired"] == []


def test_branch_NO_SAME_PROCESS_DUMP_when_a_fallback_emits_none(tmp_path, wire):
    wire["query"] = lambda p: query_out(p, completed=False, dump=False)
    wire["rc_query"] = 3
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_NO_SAME_PROCESS_DUMP
    assert any("0 searched-position dumps" in w
               for w in r["branch_reasons"][H4A.BRANCH_NO_SAME_PROCESS_DUMP])


def test_a_SEARCHED_query_without_a_dump_does_NOT_fire_the_dump_branch(tmp_path, wire):
    """NEGATIVE CONTROL: the dump branch is about FALLBACK queries only."""
    wire["query"] = lambda p: query_out(p, completed=True, dump=False)
    r = run(tmp_path)
    assert H4A.BRANCH_NO_SAME_PROCESS_DUMP not in r["branches_fired"]


def test_branch_ZERO_LENGTH_REFUSED_when_the_empty_query_yields_no_record(tmp_path, wire):
    wire["query"] = lambda p: (query_out(p, n_records=0) if not p else query_out(p))
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_ZERO_LENGTH_REFUSED
    assert all("@ply0" in w
               for w in r["branch_reasons"][H4A.BRANCH_ZERO_LENGTH_REFUSED])


def test_branch_ZERO_LENGTH_REFUSED_when_the_empty_REPLAY_yields_no_block(tmp_path, wire):
    wire["replay"] = lambda p: (replay_out(p, blocks=0) if not p else replay_out(p))
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_ZERO_LENGTH_REFUSED
    assert any("ply block" in w
               for w in r["branch_reasons"][H4A.BRANCH_ZERO_LENGTH_REFUSED])


def test_a_NONZERO_PLY_yielding_no_record_is_NOT_the_zero_length_branch(tmp_path, wire):
    """NEGATIVE CONTROL: the zero-length branch is about ply 0, exclusively."""
    wire["query"] = lambda p: (query_out(p, n_records=0) if p else query_out(p))
    r = run(tmp_path)
    assert H4A.BRANCH_ZERO_LENGTH_REFUSED not in r["branches_fired"]


def test_branch_INSTRUMENT_on_a_dirty_postcondition(tmp_path, wire):
    wire["query"] = lambda p: query_out(p, post_ok=False)
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_INSTRUMENT
    assert any("not clean" in w for w in r["branch_reasons"][H4A.BRANCH_INSTRUMENT])


def test_branch_INSTRUMENT_on_an_unexpected_reflection_count(tmp_path, wire):
    wire["query"] = lambda p: query_out(p, refl_n=INT.QUERY_REFL_N + 1)
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_INSTRUMENT
    assert any("reflective accesses" in w
               for w in r["branch_reasons"][H4A.BRANCH_INSTRUMENT])


def test_branch_INSTRUMENT_on_more_than_one_PROC_line(tmp_path, wire):
    """One jvm per call is the frozen lifecycle; two PROC lines means it broke."""
    wire["query"] = lambda p: query_out(p, n_proc=2)
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_INSTRUMENT
    assert any("PROC lines" in w for w in r["branch_reasons"][H4A.BRANCH_INSTRUMENT])


def test_branch_INSTRUMENT_on_multiple_query_records(tmp_path, wire):
    wire["query"] = lambda p: query_out(p, n_records=2)
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_INSTRUMENT
    assert any("query records" in w for w in r["branch_reasons"][H4A.BRANCH_INSTRUMENT])


def test_zero_length_refusal_OUTRANKS_the_instrument_branch(tmp_path, wire):
    """A refusal at ply 0 often produces unclean output; relabelling it
    'instrument failure' would bury the exact finding §4A exists to make."""
    wire["query"] = lambda p: (query_out(p, n_records=0, post_ok=False)
                               if not p else query_out(p))
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_ZERO_LENGTH_REFUSED
    assert H4A.BRANCH_INSTRUMENT in r["branches_fired"], "the instrument note was lost"


def test_every_branch_that_fired_is_recorded_even_when_not_the_headline(tmp_path, wire):
    wire["query"] = lambda p: (query_out(p, n_records=0, post_ok=False) if not p
                               else query_out(p, completed=False, dump=False))
    wire["rc_query"] = 3
    r = run(tmp_path)
    assert set(r["branches_fired"]) == {
        H4A.BRANCH_ZERO_LENGTH_REFUSED, H4A.BRANCH_INSTRUMENT,
        H4A.BRANCH_NO_SAME_PROCESS_DUMP}
    assert r["verdict"] == H4A.BRANCH_ZERO_LENGTH_REFUSED
    assert r["branch_precedence"] == list(H4A.BRANCH_PRECEDENCE)


def test_classify_adds_no_fifth_branch():
    """The four are frozen in the card. A fifth would be a design change."""
    assert {H4A.BRANCH_PROCEED, H4A.BRANCH_ZERO_LENGTH_REFUSED,
            H4A.BRANCH_NO_SAME_PROCESS_DUMP, H4A.BRANCH_INSTRUMENT} == {
        H4A.BRANCH_PROCEED, *H4A.BRANCH_PRECEDENCE}
    assert len(H4A.BRANCH_PRECEDENCE) == 3


# ───────────────────────────── VOID is the instrument ────────────────────────

def test_a_timeout_is_VOID_not_a_finding(tmp_path, wire, monkeypatch):
    def boom(args, **kw):
        raise subprocess.TimeoutExpired(args, H4A.PER_CALL_TIMEOUT_S)
    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(H4A.H4A4VoidError, match="VOID"):
        run(tmp_path)
    assert not (tmp_path / "r.json").exists()


def test_unparseable_output_is_VOID(tmp_path, wire):
    """A TRUNCATED POSTCOND line. `parse_postconds` raises HelperOutputError
    where `parse_procs` raises ValueError -- catching only one leaks the other
    past the runner and reports UNEXPECTED (4) instead of VOID (3)."""
    wire["query"] = lambda p: "POSTCOND no_throw=true windows=0\n"
    with pytest.raises(H4A.H4A4VoidError, match="VOID"):
        run(tmp_path)


def test_output_that_simply_parses_EMPTY_is_not_VOID(tmp_path, wire):
    """NEGATIVE CONTROL: unreadable is VOID; merely absent is an OBSERVATION."""
    wire["query"] = lambda p: (PROC.format(pid=7) + "\n"
                               + "PLY not-a-number junk\n"
                               + POST.format(ok="true", n=INT.QUERY_REFL_N, f=0) + "\n")
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_ZERO_LENGTH_REFUSED
    assert all(o["n_query_records"] == 0 for o in r["observations"]
               if o["role"] == "query")


def test_a_fallbacks_FAILURES_COUNTER_does_not_fire_the_instrument_branch(tmp_path, wire):
    """🔴 THE D1 DEFECT, GUARDED. E4Preflight sets `failures` precisely when the
    requested depth did not complete, so `PostCond.clean` is False for EVERY
    fallback. Judging `clean` would classify §4A's own subject matter as a
    broken instrument and make PROCEED unreachable."""
    wire["query"] = lambda p: query_out(p, completed=False)
    wire["rc_query"] = 3
    r = run(tmp_path)
    q = [o for o in r["observations"] if o["role"] == "query"]
    assert all(o["postcond"]["failures_counter"] == 1 for o in q)
    assert all(o["postcond"]["clean_single"] is False for o in q)
    assert all(o["postcond"]["safety_clean"] is True for o in q)
    assert H4A.BRANCH_INSTRUMENT not in r["branches_fired"]
    assert r["verdict"] == H4A.BRANCH_PROCEED


def test_a_malformed_PROC_line_is_VOID(tmp_path, wire):
    wire["query"] = lambda p: query_out(p).replace(" headless=true", "", 1)
    with pytest.raises(H4A.H4A4VoidError, match="PROC"):
        run(tmp_path)


def test_a_prefix_that_replays_to_a_different_digest_is_VOID(tmp_path, wire):
    bad = [dict(p) for p in frozen()]
    bad[0]["digest"] = "0" * 64
    with pytest.raises(H4A.H4A4VoidError, match="digest"):
        run(tmp_path, prefixes=bad)
