"""H4 runner (card step 2). SYNTHETIC FIXTURES ONLY -- nothing here is evidence.

No JVM, no research seed, no research game. Games run through the REAL harness
`play_task`, the REAL §4B adapter and the REAL persistence, stubbed only at the
process boundary (`subprocess.run`) and with a STUB incumbent. Seeds are negative
(card §10.1); every run writes into pytest's temporary directory.

Stub pids start at 10,000,000 -- above the platform's pid ceiling -- so the real
`os.kill(pid, 0)` liveness check sees them as dead; the leaked-pid control uses
this very test process's pid, which is alive.
"""
import ast
import dataclasses
import json
import os
import pathlib
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import e4_screen_integration as INT
from scripts.GPU.alphazero import h4_repair_qualification as Q
from scripts.GPU.alphazero import h4_runner as R
from scripts.GPU.alphazero import t1j_adapter as A
from scripts.GPU.alphazero.e4_screen_runner import AbortError
from scripts.GPU.alphazero.game.twixt_state import TwixtState
from tests.test_h4_repair_qualification import _block, _state, query_out, replay_out

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: A 25-ply game Red wins, checked against our engine. Whoever holds a colour
#: plays that colour's moves, so Red wins in both arms: Arm A is an incumbent
#: win and Arm B a T1j win.
SEQ = [(0, 4), (2, 20), (2, 5), (4, 20), (4, 6), (6, 20), (6, 7), (8, 20), (8, 8),
       (10, 20), (10, 9), (12, 20), (12, 10), (14, 20), (14, 11), (16, 20), (16, 12),
       (18, 20), (18, 13), (20, 20), (20, 14), (22, 20), (22, 15), (5, 22), (23, 17)]


def replay(prefix, *, pid, **kw):
    """`replay_out`, but REPORTING THE WIN on the terminal block, as the real
    helper does -- the shared builder always prints termY=false termX=false."""
    out = replay_out(prefix, pid=pid, **kw)
    winner = _state(prefix).winner()
    if winner is None:
        return out
    head = f"PLY {len(prefix)} moveNr={len(prefix)} "
    i = out.index(head)
    j = out.index("\n", i)
    flags = "termY=true termX=false" if winner == "red" else "termY=false termX=true"
    return out[:i] + out[i:j].replace("termY=false termX=false", flags) + out[j:]


def stub_incumbent(task, evaluator=None):
    """THE STUB incumbent: the scripted line. Never the real construction path."""
    return lambda state: SEQ[state.ply]


@pytest.fixture
def wire(monkeypatch):
    box = {"calls": [], "query": None, "replay": None, "timeout_at": None,
           "pid": 10_000_000, "rc_replay": 0}

    def fake_run(args, **kw):
        box["calls"].append(list(args))
        pairs = [tuple(int(v) for v in a.split(",")) for a in args
                 if "," in a and a.replace(",", "").isdigit()]
        prefix = [A.to_ours(x, y) for (x, y) in pairs]
        if box["timeout_at"] is not None and "replay" not in args \
                and len(prefix) == box["timeout_at"]:
            raise subprocess.TimeoutExpired(args, kw.get("timeout"))
        box["pid"] += 1
        if "replay" in args:
            out = (box["replay"](prefix, box["pid"]) if box["replay"]
                   else replay(prefix, pid=box["pid"]))
            return subprocess.CompletedProcess(args, box["rc_replay"], out, "")
        native = len(prefix) <= 2
        out = (box["query"](prefix, box["pid"]) if box["query"]
               else query_out(prefix, native=native, move=SEQ[len(prefix)], pid=box["pid"]))
        return subprocess.CompletedProcess(args, Q.NATIVE_EXIT if native else Q.SEARCHED_EXIT,
                                           out, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return box


@pytest.fixture
def gate_open(monkeypatch):
    """Tests must run in the REAL state: the public entry, gate OPEN."""
    monkeypatch.setattr(R, "H4_PILOT_EXECUTION_AUTHORIZED", True)


def _paths(tmp_path):
    return R.T1jPaths(java="/j", jar="/x.jar", classes=str(tmp_path / "classes"),
                      ply_cap=280)


def run(tmp_path, pairs=(("p0", -1, -2),), name="run", **kw):
    sched = R.make_schedule(pairs, mode="fixture")
    kw.setdefault("incumbent_build", stub_incumbent)
    kw.setdefault("incumbent_identity", {"stub": True})
    kw.setdefault("_compile", lambda d: {"stub_compile": True})
    return R.run_games(mode="fixture", schedule=sched, out_dir=str(tmp_path / name),
                       paths=_paths(tmp_path), deadline_s=600, **kw)


def results_of(tmp_path, name="run"):
    return str(tmp_path / name / "results.jsonl")


def voided(tmp_path, cls, name="run", **kw):
    with pytest.raises(R.H4RunVoid) as ei:
        run(tmp_path, name=name, **kw)
    assert ei.value.classification == cls, ei.value
    recs = R.read_records(results_of(tmp_path, name))
    assert recs[-1]["record_type"] == "run_void"
    assert recs[-1]["classification"] == cls
    assert not any(r["record_type"] == "segment_end" for r in recs)
    return recs[-1]


# ─────────────────────────────── 1. the gate ───────────────────────────────

def test_the_gate_is_false_as_published():
    assert R.H4_PILOT_EXECUTION_AUTHORIZED is False


def test_the_public_entry_refuses_while_the_gate_is_shut(tmp_path, wire):
    with pytest.raises(R.H4RunError, match="UNAUTHORIZED"):
        run(tmp_path)
    assert wire["calls"] == [] and not (tmp_path / "run").exists()


def test_the_cli_refuses_in_a_fresh_subprocess(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "scripts.GPU.alphazero.h4_runner", "--mode", "pilot",
         "--schedule", "s.json", "--out-dir", str(tmp_path / "o"),
         "--classes", str(tmp_path / "c")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == R.EXIT_UNAUTHORIZED, r.stderr
    assert not (tmp_path / "o").exists() and not (tmp_path / "c").exists()


def test_the_gate_is_read_at_both_entries_and_has_no_override():
    tree = ast.parse(pathlib.Path(R.__file__).read_text(encoding="utf-8"))
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for entry in ("run_games", "main"):
        names = {n.id for n in ast.walk(fns[entry]) if isinstance(n, ast.Name)}
        assert "H4_PILOT_EXECUTION_AUTHORIZED" in names, entry
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not {"environ", "getenv"} & attrs
    gates = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
             and n.id.endswith("_AUTHORIZED")}
    assert gates == {"H4_PILOT_EXECUTION_AUTHORIZED"}


def test_PILOT_and_STUDY_modes_refuse_at_step_2(tmp_path, wire, gate_open):
    for mode in ("pilot", "study"):
        sched = R.make_schedule([("p", 202699000, 202699001)], mode=mode)
        with pytest.raises(R.H4RunError, match="not runnable at step 2"):
            R.run_games(mode=mode, schedule=sched, out_dir=str(tmp_path / mode),
                        paths=_paths(tmp_path), deadline_s=60,
                        incumbent_build=stub_incumbent, incumbent_identity={"stub": 1})
    assert wire["calls"] == []


# ─────────────── 2. the schedule, seeds and destinations (§1, §9, §10.1) ───────────────

def test_a_pair_is_Arm_A_then_Arm_B_from_the_empty_board():
    a, b = R.make_schedule([("p", -1, -2)], mode="fixture")
    assert (a["arm"], a["incumbent_colour"], a["t1j_colour"]) == ("A", "red", "black")
    assert (b["arm"], b["incumbent_colour"], b["t1j_colour"]) == ("B", "black", "red")
    assert a["anchor_colour"] == "black" and b["anchor_colour"] == "red"
    assert a["opening"] == b["opening"] == R.EMPTY_BOARD and a["t1j_mdPly"] == 6


def test_FIXTURE_mode_refuses_a_non_negative_seed():
    with pytest.raises(R.H4RunError, match="non-negative seed"):
        R.make_schedule([("p", -1, 202699000)], mode="fixture")


def test_PILOT_mode_refuses_a_negative_seed():
    with pytest.raises(R.H4RunError, match="negative seed"):
        R.make_schedule([("p", 202699000, -2)], mode="pilot")


@pytest.mark.parametrize("sched,match", [
    (lambda: R.make_schedule([("p", -1, -1)], mode="fixture"), "seed repeats"),
    (lambda: R.check_schedule([{"task_id": "x", "arm": "B", "pair_id": "p", "seed": -1},
                               {"task_id": "y", "arm": "A", "pair_id": "p", "seed": -2}],
                              mode="fixture"), "adjacent A-then-B"),
    (lambda: R.check_schedule([], mode="fixture"), "whole pairs"),
    (lambda: R.check_schedule([], mode="nonsense"), "unknown mode")])
def test_schedule_integrity_faults_are_refused(sched, match):
    with pytest.raises(R.H4RunError, match=match):
        sched()


def test_FIXTURE_mode_refuses_a_destination_under_the_EVIDENCE_tree(tmp_path):
    with pytest.raises(R.H4RunError, match="evidence tree"):
        R.check_destinations(str(R.EVIDENCE_ROOT / "x"), str(tmp_path / "c"), mode="fixture")


def test_the_class_directory_must_be_OUTSIDE_the_repository(tmp_path):
    with pytest.raises(R.H4RunError, match="OUTSIDE the repository"):
        R.check_destinations(str(tmp_path / "o"), str(ROOT / "classes"), mode="fixture")


def test_an_OCCUPIED_results_file_is_refused_before_any_subprocess(tmp_path, wire,
                                                                   gate_open):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "results.jsonl").write_text("")
    with pytest.raises(FileExistsError):
        run(tmp_path)
    assert wire["calls"] == []


# ──────────────── 3. a pair end to end, then RE-READ (§3, §4, §8) ────────────────

def test_a_FIXTURE_PAIR_runs_end_to_end_and_every_record_re_derives(tmp_path, wire,
                                                                    gate_open):
    out = run(tmp_path)
    assert out["complete"] and out["games_completed"] == 2
    recs = R.read_records(results_of(tmp_path))
    assert recs[0]["record_type"] == "header"
    assert recs[0]["design"] == "H4_SYNTHETIC_FIXTURE" and recs[0]["evidence"] is False
    assert recs[0]["t1j_runtime"]["h4_acceptance"] is True
    assert recs[-1]["record_type"] == "segment_end"
    games = R.load_games(results_of(tmp_path), allow_fixture=True)
    assert [g["task_id"] for g in games] == ["p0-A", "p0-B"]
    for g in games:
        res = g["result"]
        assert res["terminal_reason"] == "win" and res["winner"] == "red"
        assert res["plies"] == len(SEQ) == len(g["plies"])
        t1j = [p for p in g["plies"] if p["actor"] == "t1j"]
        assert res["queries"] == len(t1j)
        assert res["replays"] == res["plies"] + 1
        assert all(p["source"] in (Q.NATIVE_FIRST, Q.NATIVE_SECOND_TO_FOURTH, Q.SEARCHED)
                   for p in t1j)
        assert all(o["t1j_position_digest"] == o["expected_position_digest"]
                   for o in g["processes"])
    a, b = games
    assert (a["result"]["incumbent_points"], b["result"]["t1j_points"]) == (1.0, 1.0)
    # the moves replay in OUR engine to the recorded result
    st = TwixtState(active_size=24, to_move="red")
    for p in a["plies"]:
        st = st.apply_move(tuple(p["move"]))
    assert st.winner() == a["result"]["winner"]


def test_the_viewer_EXPORT_validates_and_is_marked_a_FIXTURE(tmp_path, wire, gate_open):
    run(tmp_path)
    paths = R.write_exports(results_of(tmp_path), str(tmp_path / "exports"),
                            allow_fixture=True)
    assert len(paths) == 2
    v = json.loads(pathlib.Path(paths[0]).read_text())
    assert [m["turn"] for m in v["moves"]] == list(range(1, len(SEQ) + 1))
    assert v["winner"] == "red" and v["starting_player"] == "red"
    assert v["meta"]["evidence_note"].startswith("SYNTHETIC FIXTURE — NOT A GAME")
    import hashlib
    src = pathlib.Path(results_of(tmp_path)).read_bytes()
    assert v["meta"]["source_sha256"] == hashlib.sha256(src).hexdigest()


def test_FIXTURE_data_is_REFUSED_as_a_result(tmp_path, wire, gate_open):
    run(tmp_path)
    with pytest.raises(R.H4RunError, match="never a result"):
        R.load_games(results_of(tmp_path))


def test_MULTIPLICITY_identical_games_are_all_persisted_and_exported(tmp_path, wire,
                                                                    gate_open):
    run(tmp_path, pairs=(("p0", -1, -2), ("p1", -3, -4)))
    games = R.load_games(results_of(tmp_path), allow_fixture=True)
    assert len(games) == 4
    arm_a = [g["result"]["transcript_digest"] for g in games if g["result"]["arm"] == "A"]
    assert len(arm_a) == 2 and arm_a[0] == arm_a[1], "identical games, both kept"
    assert len(R.write_exports(results_of(tmp_path), str(tmp_path / "x"),
                               allow_fixture=True)) == 4


# ───────────────── 4. OPERATIONAL FAILURES -> run_void (§3.4, §6) ─────────────────

def test_a_TIMEOUT_voids_the_run_as_timeout(tmp_path, wire, gate_open):
    wire["timeout_at"] = 3
    v = voided(tmp_path, "timeout")
    assert v["exception"] == "TimeoutExpired" and v["stage"] == "game"
    assert (v["task_id"], v["pair_id"], v["arm"]) == ("p0-A", "p0", "A")


def test_an_ADAPTER_REFUSAL_voids_with_the_FULL_stdout(tmp_path, wire, gate_open):
    served = {}

    def serve(prefix, pid):
        out = query_out(prefix, native=len(prefix) <= 2, move=SEQ[len(prefix)], pid=pid,
                        null_sentinel=len(prefix) == 3)
        served[len(prefix)] = out
        return out
    wire["query"] = serve
    v = voided(tmp_path, "adapter_refusal")
    assert v["exception"] == "AbortError" and v["stdout"] == served[3]
    procs = [r for r in R.read_records(results_of(tmp_path)) if r["record_type"] == "process"]
    assert procs[-1]["outcome"] == "refused", "the refused call's record must survive"


def test_UNREADABLE_output_voids_as_unreadable_not_as_a_refusal(tmp_path, wire, gate_open):
    wire["query"] = lambda p, pid: query_out(p, native=len(p) <= 2, move=SEQ[len(p)],
                                             pid=pid).replace("PROC pid=", "PROC xid=", 1)
    v = voided(tmp_path, "unreadable_output")
    assert v["exception"] == "ValueError"


def test_a_REPLAY_MISMATCH_voids_as_replay_mismatch(tmp_path, wire, gate_open):
    wire["replay"] = lambda p, pid: replay(SEQ[1:len(p) + 1] if len(p) == 2 else p,
                                           pid=pid)
    voided(tmp_path, "replay_mismatch")


def test_a_COHERENCE_MISMATCH_voids_as_coherence_mismatch(tmp_path, wire, gate_open,
                                                         monkeypatch):
    """The digests are observational, so a mismatch that `compare_state` would not
    see can only be staged by corrupting the recorded field itself."""
    real = INT._h4_coherence

    def corrupt(obs, state, tp, moves):
        real(obs, state, tp, moves)
        obs["t1j_position_digest"] = "0" * 64
    monkeypatch.setattr(INT, "_h4_coherence", corrupt)
    voided(tmp_path, "coherence_mismatch")


def test_an_ILLEGAL_incumbent_move_voids_as_illegal_move(tmp_path, wire, gate_open):
    v = voided(tmp_path, "illegal_move",
               incumbent_build=lambda t, evaluator=None: lambda s: (0, 4))
    assert v["exception"] == "H4IllegalMove"


def test_a_LEAKED_pid_voids_as_leaked_process(tmp_path, wire, gate_open):
    live = os.getpid()
    wire["replay"] = lambda p, pid: replay(p, pid=live if len(p) == 1 else pid)
    voided(tmp_path, "leaked_process")


def test_a_REUSED_pid_voids_as_process_count(tmp_path, wire, gate_open):
    wire["replay"] = lambda p, pid: replay(p, pid=10_000_001 if len(p) == 2 else pid)
    voided(tmp_path, "process_count")


def test_an_incumbent_that_RAISES_voids_as_unexpected(tmp_path, wire, gate_open):
    def boom(t, evaluator=None):
        def f(state):
            raise RuntimeError("incumbent broke")
        return f
    v = voided(tmp_path, "unexpected", incumbent_build=boom)
    assert v["exception"] == "RuntimeError"


@pytest.mark.parametrize("cls", R.CLASSIFICATIONS)
def test_EVERY_classification_writes_a_run_void_record(tmp_path, wire, gate_open,
                                                       monkeypatch, cls):
    """The failure record itself, qualified for every §6 category."""
    def fail(*a, **k):
        raise R.H4RunVoid(cls, "staged")
    monkeypatch.setattr(R, "check_game", fail)
    v = voided(tmp_path, cls)
    assert v["stage"] == "game" and v["completed"]["games"] == 0
    assert v["completed"]["ply"] == len(SEQ) and v["elapsed_s"] >= 0


def test_run_void_COUNTS_are_recomputed_from_the_file(tmp_path, wire, gate_open):
    wire["timeout_at"] = 3
    v = voided(tmp_path, "timeout")
    recs = R.read_records(results_of(tmp_path))[:-1]
    kinds = [r["record_type"] for r in recs]
    assert v["completed"] == {"games": 0, "ply": kinds.count("ply"),
                              "process": kinds.count("process"), "game_result": 0}


# ──────────────────── 5. the record checks, and a tampered file ────────────────────

def _one_game(tmp_path, wire):
    run(tmp_path)
    return R.load_games(results_of(tmp_path), allow_fixture=True)[0]


def test_check_game_refuses_a_non_contiguous_ply_sequence(tmp_path, wire, gate_open):
    g = _one_game(tmp_path, wire)
    with pytest.raises(R.H4RunVoid, match="missing_record"):
        R.check_game(g["start"], g["plies"][:-1], g["processes"], g["result"])


def test_check_game_refuses_a_missing_replay_record(tmp_path, wire, gate_open):
    g = _one_game(tmp_path, wire)
    procs = [o for o in g["processes"] if o["role"] == "query"] + \
            [o for o in g["processes"] if o["role"] == "replay"][:-1]
    with pytest.raises(R.H4RunVoid, match="process_count"):
        R.check_game(g["start"], g["plies"], procs, g["result"], check_pids=False)


def test_check_game_refuses_a_record_from_ANOTHER_PAIR(tmp_path, wire, gate_open):
    g = _one_game(tmp_path, wire)
    plies = [dict(g["plies"][0], pair_id="elsewhere")] + g["plies"][1:]
    with pytest.raises(R.H4RunVoid, match="another game"):
        R.check_game(g["start"], plies, g["processes"], g["result"], check_pids=False)


def _tamper(path, pred, edit=None):
    recs = R.read_records(path)
    out = []
    dropped = False
    for r in recs:
        if not dropped and pred(r):
            dropped = True
            if edit is None:
                continue
            r = edit(r)
        out.append(r)
    pathlib.Path(path).write_text("".join(json.dumps(r) + "\n" for r in out))


@pytest.mark.parametrize("pred,edit,match", [
    (lambda r: r["record_type"] == "ply" and r["ply"] == 7, None, "the ply sequence is"),
    (lambda r: r["record_type"] == "process", None, "process_count"),
    (lambda r: r["record_type"] == "game_result", None, "missing"),
    (lambda r: r["record_type"] == "game_result",
     lambda r: dict(r, transcript_digest="0" * 64), "does not recompute"),
    # BOTH stored digests changed to the SAME value: they still agree with each
    # other, so only recomputing from the MOVES can see it (a one-sided edit is
    # caught earlier, by the coherence comparison, and proves nothing here).
    (lambda r: r["record_type"] == "process",
     lambda r: dict(r, expected_position_digest="0" * 64, t1j_position_digest="0" * 64),
     "does not recompute from the moves")])
def test_a_TAMPERED_results_file_is_refused_by_the_reader(tmp_path, wire, gate_open,
                                                          pred, edit, match):
    run(tmp_path)
    _tamper(results_of(tmp_path), pred, edit)
    with pytest.raises(R.H4RunError, match=match):
        R.load_games(results_of(tmp_path), allow_fixture=True)


def test_a_VOID_file_is_never_read_as_results(tmp_path, wire, gate_open):
    wire["timeout_at"] = 3
    with pytest.raises(R.H4RunVoid):
        run(tmp_path)
    with pytest.raises(R.H4RunError, match="VOID"):
        R.load_games(results_of(tmp_path), allow_fixture=True)


def test_the_exporter_refuses_a_digest_that_does_not_recompute(tmp_path, wire, gate_open):
    g = _one_game(tmp_path, wire)
    bad = dict(g, result=dict(g["result"], transcript_digest="0" * 64))
    with pytest.raises(R.H4RunError, match="does not recompute"):
        R.export_game(bad, results_of(tmp_path))


# ───────────────────────────── 6. OUTCOME BLINDING (§7) ─────────────────────────────

def test_the_TRACE_and_CONSOLE_never_carry_an_outcome(tmp_path, wire, gate_open, capsys):
    run(tmp_path)
    cap = capsys.readouterr()
    trace = (tmp_path / "run" / "trace.jsonl").read_text()
    assert R.blinding_violations(trace + cap.out + cap.err) == []
    assert "game_done" in trace


def test_the_blinding_check_is_NOT_vacuous():
    """CLEAN-BASELINE CONTROL: the check must see an outcome when one is there."""
    assert R.blinding_violations('{"winner": "red"}') == ["winner"]
    assert R.blinding_violations("t1j_points=1") == ["points", "t1j_points"]
    assert R.blinding_violations('{"terminal_reason": "win", "plies": 25}') == []


# ─────────────────── 7. the coherence payload ⇔ compare_state (§4) ───────────────────

FIELDS = {"ply": "ply", "next_player": "side to move", "term_y": "terminal",
          "term_x": "terminal", "pegs": "pegs", "bridges": "bridges",
          "legal": "legal set", "history": "history"}


def _tp(moves):
    st = _state(moves)
    return st, A.parse_dump(_block(st, moves))[0]


def test_equal_payloads_WHEN_compare_state_finds_no_divergence():
    st, tp = _tp(SEQ[:5])
    assert INT.compare_state(st, tp, SEQ[:5]) == []
    assert INT.position_payload(tp) == INT.expected_payload(st, SEQ[:5])


@pytest.mark.parametrize("field", list(FIELDS))
def test_each_compared_field_changes_the_payload_AND_compare_state(field):
    st, tp = _tp(SEQ[:5])
    change = {"ply": 9, "next_player": "X" if tp.next_player == "Y" else "Y",
              "term_y": not tp.term_y, "term_x": not tp.term_x,
              "pegs": set(tp.pegs) | {"1,1,Y"}, "bridges": set(tp.bridges) | {"1,1|2,3|Y"},
              "legal": set(tp.legal) - {sorted(tp.legal)[0]},
              "history": tuple(tp.history[:-1])}[field]
    bad = dataclasses.replace(tp, **{field: change})
    assert INT.position_payload(bad) != INT.expected_payload(st, SEQ[:5])
    assert any(FIELDS[field] in d for d in INT.compare_state(st, bad, SEQ[:5]))


def test_the_DIGEST_is_the_payloads_fingerprint():
    st, tp = _tp(SEQ[:5])
    assert INT.position_digest(INT.position_payload(tp)) == \
        INT.position_digest(INT.expected_payload(st, SEQ[:5]))


def test_the_coherence_fields_change_NO_accept_refuse_decision(wire, monkeypatch):
    """Card §4: additive. The same replies are accepted, and refused, without them."""
    def call(prefix, serve):
        wire["query"] = serve
        ctx = INT.IntegrationContext()
        rt = INT.T1jRuntime(java="/j", jar="/x.jar", classes="/c", ply_cap=280,
                            timeout_s=120, h4_acceptance=True)
        INT.make_binder(rt, ctx)
        st = _state(prefix)
        ctx.reset("t", prefix)
        agent = INT.T1jAgent(runtime=rt, ctx=ctx, depth=6, colour=st.to_move,
                             timeout_s=120)
        try:
            return ("accepted", agent(st))
        except AbortError as e:
            return ("refused", e.message.split(": ", 1)[-1])
    good = lambda p, pid: query_out(p, native=True, pid=pid)
    bad = lambda p, pid: query_out(p, native=True, pid=pid, null_sentinel=True)
    with_fields = [call(SEQ[:1], good), call(SEQ[:1], bad)]
    monkeypatch.setattr(INT, "_h4_coherence", lambda *a: None)
    assert [call(SEQ[:1], good), call(SEQ[:1], bad)] == with_fields
    assert [w[0] for w in with_fields] == ["accepted", "refused"]


def test_the_two_coherence_digests_are_computed_INDEPENDENTLY():
    """Each side from its OWN source: T1j's from its dump, ours from our state. A
    divergent dump must give different digests, the expected one equal to OUR
    payload's -- or a copied field would make every mismatch invisible."""
    st, tp = _tp(SEQ[:5])
    bad = dataclasses.replace(tp, pegs=set(tp.pegs) | {"1,1,Y"})
    obs = {}
    INT._h4_coherence(obs, st, bad, SEQ[:5])
    assert obs["expected_position_digest"] == INT.position_digest(
        INT.expected_payload(st, SEQ[:5]))
    assert obs["t1j_position_digest"] == INT.position_digest(INT.position_payload(bad))
    assert obs["t1j_position_digest"] != obs["expected_position_digest"]
