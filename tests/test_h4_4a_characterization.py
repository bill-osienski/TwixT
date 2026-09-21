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
              post_ok=True, refl_n=None, n_records=1, move=None,
              legal=True, null_sentinel=False):
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
        f"legal={'true' if legal else 'false'} "
        f"null_sentinel={'true' if null_sentinel else 'false'} "
        f"moveNr={len(prefix)} "
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
         "--out", str(tmp_path / "r.json"), "--classes", str(tmp_path / "cls")],
        capture_output=True, text=True, cwd=".", timeout=60)
    assert r.returncode == H4A.EXIT_UNAUTHORIZED == 5, (r.returncode, r.stderr)
    assert not (tmp_path / "cls").exists(), "it made a class dir while unauthorized"
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
    assert all(o["path"] == "incomplete" for o in q)
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


def test_SECTION_4A_NEVER_NAMES_A_FALLBACK(tmp_path, wire):
    """🔴 §4B's job, not §4A's. `completed == false` does not establish the
    qualified native-fallback signature, and labelling it here would hand §4B
    a conclusion it is supposed to reach."""
    wire["query"] = lambda p: query_out(p, completed=False)
    wire["rc_query"] = 3
    r = run(tmp_path)
    paths = {o.get("path") for o in r["observations"] if o["role"] == "query"}
    assert paths == {"incomplete"}
    assert "native_low_ply_fallback" not in json.dumps(r)


def test_the_parsed_record_carries_move_nr(tmp_path, wire):
    """Omitted from the first version. A field absent from a create-only record
    cannot be recovered later, and §4B's coherence work needs T1j's OWN count."""
    r = run(tmp_path)
    for o in r["observations"]:
        if o["role"] == "query" and o["record"]:
            assert o["record"]["move_nr"] == o["ply"]


def test_a_SEARCHED_query_without_a_dump_is_INSTRUMENT_not_the_dump_branch(tmp_path, wire):
    """The dump branch is about INCOMPLETE queries only -- but a completed
    search missing its dump is still a defect, and the first version fired
    NOTHING and reported PROCEED."""
    wire["query"] = lambda p: query_out(p, completed=True, dump=False)
    r = run(tmp_path)
    assert H4A.BRANCH_NO_SAME_PROCESS_DUMP not in r["branches_fired"]
    assert r["verdict"] == H4A.BRANCH_INSTRUMENT
    assert any("completed search emitted 0" in w
               for w in r["branch_reasons"][H4A.BRANCH_INSTRUMENT])


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


def test_a_PIN_WHOSE_OWN_DIGEST_LIES_is_VOID(tmp_path, wire, monkeypatch):
    """Defence in depth, BEHIND the matrix guard. Since the input must now equal
    the pinned file, a tampered digest can only arrive if the PIN ITSELF is
    internally inconsistent -- hash intact, contents lying. The digest is
    RECOMPUTED from the moves rather than read back, so that still fails."""
    import hashlib as _h
    good = json.loads(pathlib.Path(H4A.FROZEN_PREFIXES_REL).read_bytes())
    good["prefixes"][0]["digest"] = "0" * 64
    raw = json.dumps(good).encode()
    lying = tmp_path / "lying_pin.json"
    lying.write_bytes(raw)
    monkeypatch.setattr(H4A, "FROZEN_PREFIXES_REL", str(lying))
    monkeypatch.setattr(H4A, "FROZEN_PREFIXES_SHA256", _h.sha256(raw).hexdigest())
    with pytest.raises(H4A.H4A4VoidError, match="digest"):
        run(tmp_path, prefixes=H4A.load_frozen_prefixes())


# ═══════════════ NEGATIVE CONTROLS FOR THE REVIEWED FAIL-OPEN PATHS ═══════════
# Each of these returned PROCEED_TO_4B before the 2026-09-21 correction. They
# are grouped so the set cannot be quietly thinned: a §4A that reports PROCEED
# on malformed output would send §4B off to implement against evidence that was
# never actually established.

def test_SAME_PLY_SUBSTITUTION_is_refused(tmp_path, wire):
    """🔴 THE MATRIX FAIL-OPEN. Three prefixes -- one per ply -- still present
    {1,3,5}, and the old check accepted the resulting four-position matrix."""
    one_each, seen = [], set()
    for p in frozen():
        if p["ply"] not in seen:
            seen.add(p["ply"])
            one_each.append(p)
    assert len(one_each) == 3 and {p["ply"] for p in one_each} == {1, 3, 5}
    with pytest.raises(H4A.H4A4Error, match="not the pinned one"):
        H4A.build_matrix(one_each)


def test_a_DUPLICATED_pinned_row_is_refused():
    rows = frozen()
    with pytest.raises(H4A.H4A4Error, match="not the pinned one"):
        H4A.build_matrix(rows + [rows[0]])


def test_a_SWAPPED_position_is_refused():
    """Same count, same plies, different board. The set of ply numbers cannot
    see this, which is exactly why the whole matrix is compared."""
    rows = [dict(p) for p in frozen()]
    victim = next(p for p in rows if p["ply"] == 1)
    victim["prefix"] = [[12, 12]]
    with pytest.raises(H4A.H4A4Error, match="not the pinned one"):
        H4A.build_matrix(rows)


def test_a_REORDERED_pinned_set_is_ALSO_refused():
    """ORDER IS PART OF THE PIN, and an earlier draft of this test asserted the
    opposite on brittleness grounds. It was wrong: every observation carries a
    monotonic `ordinal`, so a reordered input yields a differently-numbered
    record of the same positions. The only supported input is
    `load_frozen_prefixes()`, whose order is deterministic -- nothing
    legitimate reorders, so accepting it would buy laxity for no caller."""
    rows = list(reversed(frozen()))
    with pytest.raises(H4A.H4A4Error, match="DIFFERENT ORDER"):
        H4A.build_matrix(rows)


def test_the_matrix_guard_accepts_the_pin_ITSELF():
    """CLEAN-BASELINE CONTROL: a guard that refused everything would pass every
    refusal test above while making the runner unusable."""
    m = H4A.build_matrix(frozen())
    assert len(m) == H4A.N_POSITIONS_EXPECTED == 10


def test_the_PUBLIC_runner_is_bound_to_the_pin_too(tmp_path, wire, monkeypatch):
    """The gate and the input binding are different protections. Opening one
    must not leave the other unguarded, so this drives the PUBLIC entry with the
    gate lifted ONLY on a patched copy of the module constant."""
    monkeypatch.setattr(H4A, "H4_4A_CHARACTERIZATION_AUTHORIZED", True)
    with pytest.raises(H4A.H4A4Error, match="not the pinned one"):
        H4A.run_characterization(prefixes=frozen()[:3], paths=PATHS,
                                 out_path=str(tmp_path / "r.json"),
                                 _compile=lambda d: {"stub": True})
    assert wire["calls"] == [], "it compiled or queried on an unpinned matrix"
    assert not (tmp_path / "r.json").exists()


def test_an_ABSENT_record_at_a_NONZERO_ply_is_INSTRUMENT(tmp_path, wire):
    """Only ply 0 may be a refusal. Anywhere else a vanished reply is the
    instrument failing -- and it fired nothing at all before."""
    wire["query"] = lambda p: (query_out(p, n_records=0) if p else query_out(p))
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_INSTRUMENT
    assert any("no query record at ply" in w
               for w in r["branch_reasons"][H4A.BRANCH_INSTRUMENT])


def test_a_TRUNCATED_replay_is_INSTRUMENT(tmp_path, wire):
    wire["replay"] = lambda p: (replay_out(p, blocks=1) if p else replay_out(p))
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_INSTRUMENT
    assert any("ply blocks, expected" in w
               for w in r["branch_reasons"][H4A.BRANCH_INSTRUMENT])


def test_an_OVERLONG_replay_is_INSTRUMENT(tmp_path, wire):
    """Extra blocks are as wrong as missing ones: the arity is exact."""
    wire["replay"] = lambda p: replay_out(list(p) + [(20, 20)])
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_INSTRUMENT
    assert any("ply blocks, expected" in w
               for w in r["branch_reasons"][H4A.BRANCH_INSTRUMENT])


def test_an_INCOHERENT_replay_final_state_is_INSTRUMENT(tmp_path, wire):
    """Right number of blocks, wrong board. Arity alone cannot see this."""
    def wrong(prefix):
        if not prefix:
            return replay_out(prefix)
        return replay_out([(20, 20)] + list(prefix)[1:])
    wire["replay"] = wrong
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_INSTRUMENT


def test_an_INCOHERENT_SEARCHED_dump_is_INSTRUMENT(tmp_path, wire):
    """A completed search whose dump describes a different position."""
    def wrong(prefix):
        out = query_out(prefix)
        other = query_out(list(prefix) + [(20, 20)])
        return out.split("  PEGS")[0] + "  PEGS" + other.split("  PEGS", 1)[1]
    wire["query"] = wrong
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_INSTRUMENT


def test_a_COMPLETED_query_with_failures_is_INSTRUMENT(tmp_path, wire):
    """🔴 THE EXEMPTION IS CONTEXT-SENSITIVE. `failures > 0` is excused only for
    an INCOMPLETE query, because that is the condition which sets it. A
    completed query must not ride on an exemption earned elsewhere."""
    def completed_but_failing(prefix):
        return query_out(prefix).replace("failures=0", "failures=1")
    wire["query"] = completed_but_failing
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_INSTRUMENT
    assert any("exemption does not apply" in w
               for w in r["branch_reasons"][H4A.BRANCH_INSTRUMENT])


def test_a_REPLAY_with_failures_is_INSTRUMENT(tmp_path, wire):
    """The exemption is for queries. A replay never searches, so it has no
    incomplete-search condition to excuse."""
    wire["replay"] = lambda p: replay_out(p).replace("failures=0", "failures=1")
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_INSTRUMENT
    assert any("on a replay" in w
               for w in r["branch_reasons"][H4A.BRANCH_INSTRUMENT])


def test_an_INCOMPLETE_query_with_failures_is_STILL_exempt(tmp_path, wire):
    """NEGATIVE CONTROL ON THE EXEMPTION: narrowing it until it excuses nothing
    would resurrect the bug it was added to fix -- every fallback classified as
    a broken instrument, and PROCEED unreachable."""
    wire["query"] = lambda p: query_out(p, completed=False)
    wire["rc_query"] = 3
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_PROCEED
    assert H4A.BRANCH_INSTRUMENT not in r["branches_fired"]


# ────────────────── the CLI resolves rather than defaults ────────────────────

def test_the_cli_offers_NO_path_or_cap_DEFAULTS():
    """🔴 `T1jPaths` has no default for `ply_cap` on purpose. An argparse
    default of None put that silence straight back, and an omitted argument
    reached execution as None."""
    names = _string_constants(H4A)
    for gone in ("--java", "--jar", "--ply-cap"):
        assert gone not in names, f"{gone} lets an unverified value reach a jvm"
    assert "--classes" in names and "--out" in names


def test_the_cli_requires_classes_in_a_fresh_subprocess(tmp_path):
    """Qualified as a REAL subprocess. argparse exits 2 before the gate is even
    consulted, so a missing output location cannot become a None."""
    r = subprocess.run(
        [sys.executable, "-m", "scripts.GPU.alphazero.h4_4a_characterization",
         "--out", str(tmp_path / "r.json")],
        capture_output=True, text=True, cwd=".", timeout=60)
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "--classes" in r.stderr
    assert not (tmp_path / "r.json").exists()


def test_the_frozen_cap_is_READ_from_the_rules_not_retyped():
    from scripts.GPU.alphazero import l0_match_rules as L0R
    assert H4A.PLY_CAP is L0R.PLY_CAP == 280


def test_resolve_paths_goes_through_the_VERIFIED_toolchain(monkeypatch):
    """It must resolve, never accept. The jar and java come from
    `verified_paths`, which hashes before it returns."""
    from scripts.GPU.alphazero import t1j_toolchain as TC
    seen = {}

    def fake_verified(*a, **kw):
        seen["called"] = True
        return {"jar": "/verified/t1j.jar", "jdk_home": "/verified/jdk"}

    monkeypatch.setattr(TC, "verified_paths", fake_verified)
    p = H4A.resolve_paths("/tmp/does-not-exist-classes")
    assert seen.get("called") is True
    assert p.jar == "/verified/t1j.jar"
    assert p.java == "/verified/jdk/bin/java"
    assert p.ply_cap == 280
    assert p.classes == "/tmp/does-not-exist-classes"


# ═════════ MOVE VALIDATION APPLIES TO EVERY QUERY RECORD, NOT JUST SEARCHED ══
# 🔴 THE LAST FAIL-OPEN. Validation was nested under `path == "searched"`, so an
# INCOMPLETE query returning a null sentinel, a move T1j itself called illegal,
# or a move illegal in our engine returned PROCEED_TO_4B. These are parametrized
# over BOTH paths precisely to pin that the check is path-independent: a
# regression that re-nests it would pass the searched half and fail here.

@pytest.mark.parametrize("completed", [True, False], ids=["searched", "incomplete"])
def test_a_NULL_SENTINEL_reply_is_INSTRUMENT_on_either_path(tmp_path, wire, completed):
    wire["query"] = lambda p: query_out(p, completed=completed, null_sentinel=True)
    wire["rc_query"] = 0 if completed else 3
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_INSTRUMENT, r["branch_reasons"]
    assert any("null sentinel" in w
               for w in r["branch_reasons"][H4A.BRANCH_INSTRUMENT])


@pytest.mark.parametrize("completed", [True, False], ids=["searched", "incomplete"])
def test_T1J_CALLING_ITS_OWN_MOVE_ILLEGAL_is_INSTRUMENT_on_either_path(
        tmp_path, wire, completed):
    wire["query"] = lambda p: query_out(p, completed=completed, legal=False)
    wire["rc_query"] = 0 if completed else 3
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_INSTRUMENT, r["branch_reasons"]
    assert any("reports its own move" in w
               for w in r["branch_reasons"][H4A.BRANCH_INSTRUMENT])


@pytest.mark.parametrize("completed", [True, False], ids=["searched", "incomplete"])
def test_A_MOVE_ILLEGAL_IN_OUR_ENGINE_is_INSTRUMENT_on_either_path(
        tmp_path, wire, completed):
    """(0,0) is a corner and is legal on no TwixT board, at any ply. T1j may
    report `legal=true` about it; our engine is the second opinion that matters."""
    wire["query"] = lambda p: query_out(p, completed=completed, move=(0, 0))
    wire["rc_query"] = 0 if completed else 3
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_INSTRUMENT, r["branch_reasons"]
    assert any("not legal in our engine" in w
               for w in r["branch_reasons"][H4A.BRANCH_INSTRUMENT])


def test_A_CLEAN_INCOMPLETE_QUERY_IS_STILL_THE_PROCEED_BASELINE(tmp_path, wire):
    """🔴 CLEAN-BASELINE CONTROL FOR THE WHOLE VALIDATION BLOCK. Tightening move
    validation until nothing passes would satisfy every test above while making
    PROCEED unreachable -- which is the same defect, from the other direction,
    as the `failures` exemption that once condemned every fallback."""
    wire["query"] = lambda p: query_out(p, completed=False, legal=True,
                                        null_sentinel=False)
    wire["rc_query"] = 3
    r = run(tmp_path)
    assert r["verdict"] == H4A.BRANCH_PROCEED, r["branch_reasons"]
    assert r["branches_fired"] == []
    q = [o for o in r["observations"] if o["role"] == "query"]
    assert all(o["path"] == "incomplete" for o in q)
    assert all(o["move_legal_in_our_engine"] for o in q)
