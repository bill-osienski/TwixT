"""H5 cap-completion diagnostic (EXPLORATORY) -- card §1-§9 (§6 A1, §7 A2, §8 A3, §9 A4).

SYNTHETIC FIXTURES for everything on the T1j and incumbent side: no JVM, no compile, no
model, no query. T1j's replay is faked FROM OUR OWN ENGINE (a perfectly agreeing T1j),
then broken one field at a time. Stage 2's T1j path runs the REAL §4B agent with only
`subprocess.run` served synthetically (the process boundary, never the `_query` seam).
One test re-derives the REAL frozen set from the committed H5 records, read-only, and
checks it against the card's pins. Messages fixed.
"""
import collections
import copy
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import types

import pytest

from scripts.GPU.alphazero import h5_cap_completion as C
from scripts.GPU.alphazero import h5_cap_completion_authorization as AUTH
from scripts.GPU.alphazero import h4_repair_qualification as Q
from scripts.GPU.alphazero import h4_runner as R4
from scripts.GPU.alphazero import t1j_adapter as A
from scripts.GPU.alphazero.game.twixt_state import TwixtState
from tests.test_h4_pilot_feasibility import commit, new_repo
from tests.test_h4_repair_qualification import query_out

ROOT = pathlib.Path(__file__).resolve().parents[1]
RED_TOP = [(0, 2), (2, 3), (4, 2), (6, 3), (8, 2), (10, 3)]
#: one linking hole (12,4); black's early (11,5) makes (12,3) and (13,4) bridge-blocks too
RED_BOT_ONE = [(14, 5), (16, 4), (18, 5), (20, 4), (22, 5), (23, 7)]
BLACK_ONE = [(11, 5)] + [(2 * i + 1, 20) for i in range(10)] + [(2, 21)]
#: two linking holes (12,2), (12,4): two winning moves, no single defence
RED_BOT_TWO = [(14, 3), (16, 2), (18, 3), (20, 2), (22, 3), (23, 5)]
BLACK_TWO = [(2 * i + 1, 20) for i in range(11)] + [(2, 21)]
ONE = [m for pair in zip(RED_TOP + RED_BOT_ONE, BLACK_ONE) for m in pair]   # 24 moves
TWO = [m for pair in zip(RED_TOP + RED_BOT_TWO, BLACK_TWO) for m in pair]


def actors(n):            # Arm B: T1j red (odd plies), the incumbent black
    return ["t1j" if i % 2 == 0 else "incumbent" for i in range(n)]


def game(tid, moves):
    return {"task_id": tid, "moves": [tuple(m) for m in moves], "actors": actors(len(moves))}


GAMES = [game("s1-B", ONE + [(2, 10), (4, 21)]), game("s2-B", ONE + [(2, 10), (4, 21)]),
         game("c-ct", ONE + [(12, 4)]), game("c-cd", ONE[:23] + [(13, 4)])]
EPISODES = {"S1": {"members": ("s1-B", "s2-B"), "t1j_ply": 25,
                   "t1j_digest": C.prefix_digest(ONE), "inc_digest": C.prefix_digest(ONE[:23]),
                   "W": ((12, 4),), "t1j_recorded": (2, 10), "inc_recorded": (2, 21),
                   "saving": ((12, 3), (12, 4), (13, 4))}}
CONTROLS = {"XT": {"game": "c-ct", "n": 24, "agent": "t1j", "role": "complete", "threat": None,
                   "recorded": (12, 4), "digest": C.prefix_digest(ONE)},
            "XD": {"game": "c-cd", "n": 23, "agent": "incumbent", "role": "defend",
                   "threat": ((12, 4),), "recorded": (13, 4), "digest": C.prefix_digest(ONE[:23])}}
FSHA, TOOL = "f" * 64, {"jar_sha256": "j" * 64}
BINDING = {"frozen_set_sha256": FSHA, "toolchain": TOOL}


def frozen():
    return C.build_frozen_set(GAMES, EPISODES, CONTROLS)


def binding(fz):
    return {**BINDING, "position_digests": C.position_digests(fz)}


def ply_state(state, moves):
    pegs, bridges = A.our_snapshot(state)
    w = state.winner()
    return A.PlyState(ply=state.ply, next_player=A.PLAYER_TO_T1J[state.to_move],
                      term_y=w == "red", term_x=w == "black", pegs=pegs, bridges=bridges,
                      history=tuple(A.to_t1j(*m) for m in moves),
                      legal={A.to_t1j(r, c) for r, c in state.legal_moves()})


def faithful(moves):
    """A T1j that agrees with our engine on every ply."""
    s, plies = TwixtState(active_size=A.BOARD_N, to_move="red"), []
    plies.append(ply_state(s, []))
    for i, m in enumerate(moves):
        s = s.apply_move(tuple(m))
        plies.append(ply_state(s, moves[:i + 1]))
    return {"plies": plies, "rc": 0, "out": f"SYNTHETIC {len(moves)} moves\n", "error": None}


def mutated(fn):
    """Break the LAST ply of a faithful replay with `fn(PlyState) -> PlyState`."""
    def replay(moves):
        r = faithful(moves)
        r["plies"][-1] = fn(r["plies"][-1])
        return r
    return replay


def item(fz, ident):
    return next(i for i in C.stage1_items(fz) if i["id"] == ident)


def run_item(fz, ident, replay):
    it = item(fz, ident)
    states, moves, problems = C.ours_check(it)
    assert problems == [], "our own engine does not meet the fixture's expectation"
    return C.classify(it, states, moves, replay(moves))


# ─────────────────────────────── the gate ───────────────────────────────

def test_the_gate_is_false_as_published():
    assert AUTH.H5_CAP_COMPLETION_AUTHORIZED is False


@pytest.mark.parametrize("entry", ["freeze", "run_stage1", "run_stage2", "run_stage2_claimed"])
def test_every_public_entry_refuses_with_the_gate_shut_BEFORE_anything(entry, monkeypatch,
                                                                        tmp_path):
    for owner, name in ((C, "_frozen_from_records"), (C, "_verified_paths"), (C, "_compile"),
                        (C, "_incumbent_choose"), (C, "_claim"), (C, "_watch"),
                        (C.F, "check_durable"), (os, "makedirs"), (os, "mkdir")):
        monkeypatch.setattr(owner, name, lambda *a, **k: pytest.fail("touched before the gate"))
    args = (str(tmp_path / "x-EXPLORATORY"),) + (() if entry == "freeze" else (str(tmp_path),))
    with pytest.raises(C.CapUnauthorized):
        getattr(C, entry)(*args)


def test_the_CLI_exits_5_as_a_fresh_subprocess_and_creates_nothing(tmp_path):
    out = tmp_path / "x-EXPLORATORY"
    r = subprocess.run([sys.executable, "-m", "scripts.GPU.alphazero.h5_cap_completion",
                        "freeze", "--out-dir", str(out)], cwd=ROOT, capture_output=True,
                       text=True)
    assert r.returncode == C.EXIT_UNAUTHORIZED == 5, r.stderr
    assert not out.exists(), "the closed CLI created the evidence directory"


# ─────────────────────────────── the frozen set (§1, §6 A1.1) ───────────────────────────────

def test_the_frozen_set_keeps_every_PROVENANCE_LINK_and_its_frequency():
    fz = frozen()
    ep = fz["episodes"]["S1"]
    assert (ep["members"], ep["frequency"]) == (["s1-B", "s2-B"], 2), \
        "the provenance links or the frequency are lost"
    assert ep["saving"] == [[12, 3], [12, 4], [13, 4]], "the saving set is not exact"
    assert fz["controls"]["XD"]["threat"] == [[12, 4]], "the pinned threat is not recorded"


def test_a_WRONG_THREAT_COORDINATE_is_refused():
    bad = copy.deepcopy(CONTROLS)
    bad["XD"]["threat"] = ((12, 5),)
    with pytest.raises(C.CapRefused, match="not the pinned threat"):
        C.build_frozen_set(GAMES, EPISODES, bad)


def test_a_wrong_DIGEST_and_a_member_with_another_prefix_are_refused():
    bad = copy.deepcopy(CONTROLS)
    bad["XT"]["digest"] = "0" * 64
    with pytest.raises(C.CapRefused, match="does not hash to the pin"):
        C.build_frozen_set(GAMES, EPISODES, bad)
    other = list(GAMES)
    moves = list(ONE)
    moves[0] = (0, 4)
    other[1] = game("s2-B", moves + [(2, 10), (4, 21)])
    with pytest.raises(C.CapRefused, match="prefix does not hash to the pin"):
        C.build_frozen_set(other, EPISODES, CONTROLS)


def test_a_wrong_WITNESS_and_a_wrong_SAVING_set_are_refused():
    bad = copy.deepcopy(EPISODES)
    bad["S1"]["W"] = ((12, 2),)
    with pytest.raises(C.CapRefused, match="not the pinned witness"):
        C.build_frozen_set(GAMES, bad, CONTROLS)
    bad = copy.deepcopy(EPISODES)
    bad["S1"]["saving"] = ((12, 4),)
    with pytest.raises(C.CapRefused, match="saving moves are"):
        C.build_frozen_set(GAMES, bad, CONTROLS)


# ─────────────────────────────── acceptable replies (§6 A1.1) ───────────────────────────────

def test_ALTERNATIVE_valid_WINNING_moves_are_accepted_and_history_is_not_required():
    s = C.state_of(TWO)
    assert C.immediate_wins(s) == [(12, 2), (12, 4)], "the fixture lost its two wins"
    for m in ((12, 2), (12, 4)):
        assert C.acceptable(s, m, "complete") and C.acceptable(s, m, "defend"), \
            "a valid winning move was not accepted"
    assert not C.acceptable(s, (2, 10), "complete"), "a non-winning move was accepted"


def test_TERMINAL_PRECEDENCE_nothing_after_a_winning_move_is_examined(monkeypatch):
    """§7 C1 / §6 A1.1: a win counts as saving, and no opponent reply after it is ever
    scanned -- observable only by watching which positions get scanned."""
    s = C.state_of(TWO)
    real, scanned = C.D0.one_ply_threats, []

    def spy(state):
        scanned.append(state.ply)
        return real(state)
    monkeypatch.setattr(C.D0, "one_ply_threats", spy)
    assert C.acceptable(s, (12, 2), "defend"), "a winning move was not accepted as saving"
    assert s.ply + 1 not in scanned, "an opponent reply after the win was examined"


def test_ALTERNATIVE_valid_BLOCKS_are_accepted_and_a_miss_or_illegal_move_is_not():
    s = C.state_of(ONE[:23])
    for m in ((12, 3), (12, 4), (13, 4)):
        assert C.acceptable(s, m, "defend"), "a valid block was not accepted"
    assert not C.acceptable(s, (2, 21), "defend"), "the recorded miss was accepted"
    assert not C.acceptable(s, (12, 3), "complete"), "a block counted as a completion"
    assert not C.acceptable(s, (0, 0), "defend"), "an illegal corner move was accepted"


# ─────────────────────────────── Stage 1 (§2, §6 A1.3) ───────────────────────────────

def test_the_STAGE1_item_set_is_exact_for_the_fixture():
    ids = [i["id"] for i in C.stage1_items(frozen())]
    assert ids == ["S1/t1j+W12,4", "S1/t1j+recorded", "S1/incumbent+12,3+W12,4",
                   "S1/incumbent+12,4=W", "S1/incumbent+13,4+W12,4", "XT/witness",
                   "XD/block+threat12,4"], "the Stage-1 item set is not the frozen one"


def stage1(tmp_path, replay=None, compile_fn=None):
    return C._stage1(frozen(), FSHA, compile_fn or (lambda: (None, TOOL)),
                     lambda paths: replay or faithful, tmp_path)


def test_a_FAITHFUL_T1j_agrees_on_every_item(tmp_path):
    fz = frozen()
    rec = stage1(tmp_path)
    assert (rec["status"], rec["all_agree"], len(rec["items"])) == ("COMPLETE", True, 7), \
        "a faithful T1j did not agree"
    assert rec["binding"] == binding(fz), "the binding was not built from the run"
    out = rec["items"][0]["streams"]["stdout"]
    t = tmp_path / C.STAGE1_DIR / out["file"]
    assert hashlib.sha256(t.read_bytes()).hexdigest() == out["sha256"]
    assert rec["provenance"]["S1"] == {"members": ["s1-B", "s2-B"], "frequency": 2}


def test_a_TERMINAL_flag_disagreement_is_classified_as_such():
    flip = mutated(lambda p: A.PlyState(**{**p.__dict__, "term_y": not p.term_y}))
    assert run_item(frozen(), "S1/t1j+W12,4", flip)["outcome"] == "TERMINAL_DISAGREEMENT"
    assert run_item(frozen(), "S1/t1j+recorded", flip)["outcome"] == "TERMINAL_DISAGREEMENT"


def test_a_RECONSTRUCTION_disagreement_bridges_or_an_earlier_ply():
    nob = mutated(lambda p: A.PlyState(**{**p.__dict__, "bridges": set(list(p.bridges)[1:])}))
    assert run_item(frozen(), "S1/t1j+W12,4", nob)["outcome"] == "RECONSTRUCTION_DISAGREEMENT"

    def early(moves):
        r = faithful(moves)
        p = r["plies"][3]
        r["plies"][3] = A.PlyState(**{**p.__dict__, "pegs": set(list(p.pegs)[1:])})
        return r
    assert run_item(frozen(), "S1/t1j+recorded", early)["outcome"] == \
        "RECONSTRUCTION_DISAGREEMENT", "a divergence at an earlier ply was not caught"


@pytest.mark.parametrize("broken", [{"rc": 1}, {"error": "TimeoutExpired: 60 s"},
                                    {"plies": []}])
def test_an_UNAVAILABLE_replay_is_never_agreement(broken):
    def replay(moves):
        return {**faithful(moves), **broken}
    assert run_item(frozen(), "S1/t1j+W12,4", replay)["outcome"] == "UNAVAILABLE"


def _chain(col):
    """A red top-to-bottom chain of knight bridges down columns col / col+1."""
    pegs = [(2 * i, col + (i % 2)) for i in range(12)] + [(23, col + 3)]
    return pegs, list(zip(pegs, pegs[1:]))


def test_PATHS_are_validated_per_graph_and_NOT_compared_so_two_valid_paths_agree():
    (p1, b1), (p2, b2) = _chain(2), _chain(14)
    both = {p: "red" for p in p1 + p2}
    assert C.has_winning_path(both, b1 + b2, "red"), "two valid paths were not accepted"
    assert C.has_winning_path({p: "red" for p in p1}, b1, "red") and \
        C.has_winning_path({p: "red" for p in p2}, b2, "red"), "either path alone must do"
    assert not C.has_winning_path(both, [], "red"), "pegs without bridges made a path"
    tp = faithful(ONE + [(12, 4)])["plies"][-1]
    assert C.has_winning_path(*C.t1j_graph(tp), "red"), \
        "our independent check of T1j's reported board found no path"


# ─────────────────────────────── the recording contract (§7 A2) ───────────────────────────────

def read_stream(tmp_path, row, s):
    stored = row["streams"][s]
    return None if stored is None else (tmp_path / C.STAGE1_DIR / stored["file"]).read_text()


def test_the_CLAIM_comes_BEFORE_the_compile(tmp_path):
    seen = []

    def compile_fn():
        seen.append((tmp_path / C.STAGE1_DIR).is_dir())
        return None, TOOL
    stage1(tmp_path, compile_fn=compile_fn)
    assert seen == [True], "the compile ran before Stage 1 was claimed"


def test_a_NORMAL_return_records_its_rc_and_says_stderr_is_UNAVAILABLE(tmp_path):
    row = stage1(tmp_path)["items"][0]
    assert (row["rc"], row["rc_note"], row["exception_type"]) == (0, None, None)
    assert row["streams"]["stderr"] is None and row["stderr_note"] == C.NO_STDERR, \
        "a missing stderr was not recorded as explicitly unavailable"
    assert read_stream(tmp_path, row, "stdout").startswith("SYNTHETIC")


def test_MALFORMED_replay_output_keeps_the_stdout_the_exception_carries(tmp_path):
    def replay(moves):
        raise A.HelperOutputError("the helper's replay output could not be parsed", "GARBLED\n")
    row = stage1(tmp_path, replay=replay)["items"][0]
    assert (row["outcome"], row["exception_type"]) == ("UNAVAILABLE", "HelperOutputError")
    assert read_stream(tmp_path, row, "stdout") == "GARBLED\n", "the carried stdout was lost"
    assert row["rc"] is None and row["rc_note"].startswith("unavailable: raised before")
    assert row["streams"]["stderr"] is None and row["stderr_note"] == C.NO_STDERR


@pytest.mark.parametrize("out,err", [("PARTIAL OUT\n", "PARTIAL ERR\n"),
                                     (b"PARTIAL OUT\n", b"PARTIAL ERR\n")])
def test_a_TIMEOUT_keeps_its_partial_stdout_and_stderr(tmp_path, out, err):
    def replay(moves):
        raise subprocess.TimeoutExpired(["java"], 60, output=out, stderr=err)
    row = stage1(tmp_path, replay=replay)["items"][0]
    assert (row["outcome"], row["exception_type"]) == ("UNAVAILABLE", "TimeoutExpired")
    assert (read_stream(tmp_path, row, "stdout"), read_stream(tmp_path, row, "stderr")) == \
        ("PARTIAL OUT\n", "PARTIAL ERR\n"), "the partial output was lost"
    assert row["rc"] is None and "killed at the 60 s timeout" in row["rc_note"]


def test_ANY_OTHER_exception_records_every_field_as_explicitly_unavailable(tmp_path):
    def replay(moves):
        raise OSError("java: not found")
    row = stage1(tmp_path, replay=replay)["items"][0]
    assert (row["outcome"], row["exception_type"]) == ("UNAVAILABLE", "OSError")
    assert row["streams"] == {"stdout": None, "stderr": None}
    assert all(row[k].startswith("unavailable") for k in ("stdout_note", "stderr_note", "rc_note"))


def test_a_COMPILE_FAILURE_leaves_a_claimed_directory_and_a_STOP_record(tmp_path):
    def compile_fn():
        e = RuntimeError("javac exited 1")
        e.stdout, e.stderr, e.returncode = "javac says\n", "error: cannot find symbol\n", 1
        raise e
    rec = stage1(tmp_path, compile_fn=compile_fn)
    assert (rec["status"], rec["failure"]["failing_stage"], rec["failure"]["exception_type"]) \
        == ("STOP", "compile", "RuntimeError")
    assert rec["items"] == [] and rec["binding"] is None and rec["binding_note"], \
        "a STOP before the binding claimed to have one"
    d = tmp_path / C.STAGE1_DIR
    assert (d / rec["failure"]["streams"]["stderr"]["file"]).read_text() == \
        "error: cannot find symbol\n" and rec["failure"]["rc"] == 1, "the compiler's output was lost"
    with pytest.raises(C.CapRefused, match="occupied"):
        stage1(tmp_path)


def test_an_INTERRUPTION_after_a_completed_item_is_recorded_and_re_raised(tmp_path):
    calls = []

    def replay(moves):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return faithful(moves)
    with pytest.raises(KeyboardInterrupt):
        stage1(tmp_path, replay=replay)
    d = tmp_path / C.STAGE1_DIR
    rec = json.loads((d / "record.json").read_text())
    assert (rec["status"], rec["failure"]["failing_stage"], len(rec["items"])) == \
        ("INTERRUPTED", "replay:S1/t1j+recorded", 1), "the interruption was not recorded"
    log = [json.loads(line)["step"] for line in (d / "log.jsonl").read_text().splitlines()]
    assert log == ["compile", "binding", "item"], "the completed observation was not persisted"
    assert not rec["all_agree"]


# ─────────────────────────────── Stage-2 permission (§6 A1.2) ───────────────────────────────

def committed_stage1(tmp_path, mutate=None, commit_after=True, name="repo"):
    """A Stage-1 record in its own temporary git repository, optionally edited."""
    repo = new_repo(tmp_path / name)
    fz = frozen()
    stage1(repo)
    d = repo / C.STAGE1_DIR
    if mutate:
        rec = json.loads((d / "record.json").read_text())
        mutate(rec, d)
        (d / "record.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    if commit_after:
        commit(repo, ".")
    return fz, d


def test_PERMISSION_baseline_an_exact_committed_bound_record_is_accepted(tmp_path):
    fz, d = committed_stage1(tmp_path)
    assert C.stage2_permission(d, fz, binding(fz), _fixture=True)["all_agree"]


@pytest.mark.parametrize("case,mutate,match", [
    ("missing", lambda r, d: r["items"].pop(), "not exact"),
    ("duplicated", lambda r, d: r["items"].append(dict(r["items"][0])), "not exact"),
    ("extra", lambda r, d: r["items"].append({**r["items"][0], "id": "X9/forged"}), "not exact"),
    ("not AGREE", lambda r, d: r["items"][2].update(outcome="TERMINAL_DISAGREEMENT"),
     "not AGREE"),
    ("frozen-set sha", lambda r, d: r["binding"].update(frozen_set_sha256="e" * 64),
     "binding differs"),
    ("position digest", lambda r, d: r["binding"]["position_digests"].update(XT="0" * 64),
     "binding differs"),
    ("toolchain", lambda r, d: r["binding"]["toolchain"].update(jar_sha256="k" * 64),
     "binding differs"),
    ("transcript hash", lambda r, d: r["items"][1]["streams"]["stdout"].update(sha256="0" * 64),
     "does not hash to its record"),
    ("not COMPLETE", lambda r, d: r.update(status="STOP"), "not COMPLETE"),
    ("no stored stdout", lambda r, d: r["items"][0]["streams"].update(stdout=None),
     "no stored stdout"),
])
def test_PERMISSION_refuses_an_incomplete_or_rebound_record(tmp_path, case, mutate, match):
    fz, d = committed_stage1(tmp_path, mutate)
    with pytest.raises(C.CapRefused, match=match):
        C.stage2_permission(d, fz, binding(fz), _fixture=True)


def test_PERMISSION_refuses_an_uncommitted_record_and_a_modified_transcript(tmp_path):
    fz, d = committed_stage1(tmp_path / "a", commit_after=False)
    with pytest.raises(C.CapRefused, match="the Stage-1 record is not durable"):
        C.stage2_permission(d, fz, binding(fz), _fixture=True)
    fz, d = committed_stage1(tmp_path / "c")
    rec = json.loads((d / "record.json").read_text())
    rec["note"] = "edited after the commit; every check below would still pass"
    (d / "record.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    with pytest.raises(C.CapRefused, match="the Stage-1 record is not durable"):
        C.stage2_permission(d, fz, binding(fz), _fixture=True)
    fz, d = committed_stage1(tmp_path / "b")
    t = sorted((d / "streams").iterdir())[0]
    t.write_text("edited after the commit\n")
    with pytest.raises(C.CapRefused, match="transcript not durable"):
        C.stage2_permission(d, fz, binding(fz), _fixture=True)


# ─────────────────────────────── Stage 2 (§3, §6 A1.1, §8 A3) ───────────────────────────────

PERMIT = {"status": "COMPLETE", "all_agree": True}
STAGE1 = {"binding": "synthetic", "record_sha256": "0" * 64}
PATHS = types.SimpleNamespace(java="/nonexistent/java", jar="/nonexistent/t1j.jar",
                              classes="/nonexistent/classes")
IDENTITY = {"incumbent": "synthetic"}


def chooser(replies):
    calls = collections.Counter()

    def choose(moves, tag, ev):
        pos = tag.split("#")[0]
        calls[pos] += 1
        r = replies[pos]
        if isinstance(r, BaseException):
            raise r
        if callable(r):
            return r(moves, tag, ev)
        return {"move": None if r is None else list(r), "stub": True}
    choose.calls = calls
    return choose


def t1j_stub():
    return chooser({"S1/t1j": (2, 10), "XT": (12, 4)})


def inc_stub():
    return chooser({"S1/incumbent": (13, 4), "XD": (12, 3)})


def stage2(tmp_path, t1j=None, inc=None, compile_fn=None, load=None, **kw):
    t1j, inc = t1j or t1j_stub(), inc or inc_stub()
    C._claim(tmp_path / C.STAGE2_DIR)            # §9 A4.1: the parent's claim
    return C._stage2(frozen(), PERMIT, STAGE1, compile_fn or (lambda dl: (PATHS, TOOL)),
                     load or (lambda: (inc, IDENTITY)), lambda paths: t1j, tmp_path, **kw)


def log_steps(tmp_path):
    return [json.loads(line)["step"] for line in
            (tmp_path / C.STAGE2_DIR / "log.jsonl").read_text().splitlines()]


def test_STAGE2_judges_by_the_rules_and_reports_history_apart(tmp_path):
    t1j, inc = t1j_stub(), inc_stub()
    rec = stage2(tmp_path, t1j, inc)
    assert (rec["status"], rec["calls"]) == ("COMPLETE", {"t1j": "6/6", "incumbent": "4/4"})
    assert rec["controls_ok"], "valid alternative replies failed their controls"
    ep = rec["episodes"]["S1"]
    assert (ep["t1j_completes"], ep["reading"]) == ("0/3", "AGENT_CHOICE (reproduced)")
    assert ep["incumbent_blocks"] == "2/2" and ep["incumbent_equals_recorded"] == 0, \
        "an alternative block was judged by history"
    assert (ep["members"], ep["frequency"]) == (["s1-B", "s2-B"], 2)
    assert t1j.calls == {"S1/t1j": 3, "XT": 3} and inc.calls == {"S1/incumbent": 2, "XD": 2}
    assert log_steps(tmp_path) == ["compile", "model_load"] + ["call"] * 10


def test_a_FAILED_CONTROL_makes_every_episode_uninterpretable(tmp_path):
    rec = stage2(tmp_path, chooser({"S1/t1j": (2, 10), "XT": (2, 10)}),
                 chooser({"S1/incumbent": (2, 21), "XD": (12, 3)}))
    assert rec["status"] == "COMPLETE" and not rec["controls_ok"]
    assert rec["episodes"]["S1"]["reading"] == "uninterpretable"


def test_STAGE2_refuses_without_a_COMPLETE_all_AGREE_permission(tmp_path):
    for permit in ({"status": "COMPLETE", "all_agree": False},
                   {"status": "STOP", "all_agree": True}):
        with pytest.raises(C.CapRefused, match="all-AGREE"):
            C._stage2(frozen(), permit, STAGE1, lambda dl: pytest.fail("compiled"),
                      lambda: pytest.fail("loaded"), lambda p: pytest.fail("t1j"), tmp_path)
    assert not (tmp_path / C.STAGE2_DIR).exists()


# ── §8 A3.2: the claim, then ONE deadline

def test_STAGE2_runs_ONLY_on_an_UNCONSUMED_claim_so_compile_and_load_follow_it(tmp_path):
    d = tmp_path / C.STAGE2_DIR
    args = (frozen(), PERMIT, STAGE1, lambda dl: pytest.fail("compiled"),
            lambda: pytest.fail("loaded"), lambda p: pytest.fail("t1j"), tmp_path)
    with pytest.raises(C.CapRefused, match="unconsumed claim"):
        C._stage2(*args)                         # never claimed
    C._claim(d)
    (d / "log.jsonl").write_text("")             # a claim already consumed
    with pytest.raises(C.CapRefused, match="unconsumed claim"):
        C._stage2(*args)


def test_an_OCCUPIED_stage2_directory_is_refused_by_the_CLAIM_and_left_untouched(tmp_path):
    (tmp_path / C.STAGE2_DIR).mkdir()
    (tmp_path / C.STAGE2_DIR / "earlier.txt").write_text("an earlier attempt\n")
    with pytest.raises(C.CapRefused, match="occupied"):
        C._claim(tmp_path / C.STAGE2_DIR)
    assert [p.name for p in (tmp_path / C.STAGE2_DIR).iterdir()] == ["earlier.txt"]


def test_the_DEADLINE_covers_the_COMPILE_a_slow_compile_stops_before_the_load(tmp_path):
    t = [0.0]

    def compile_fn(dl):
        t[0] += 3601.0                           # a compile that used the whole hour
        return PATHS, TOOL
    rec = stage2(tmp_path, compile_fn=compile_fn, load=lambda: pytest.fail("loaded"),
                 clock=lambda: t[0])
    assert (rec["status"], rec["failure"]["failing_step"]) == ("STOP", "model_load")
    assert "deadline passed before model_load" in rec["failure"]["reason"]
    assert rec["compile"] == TOOL and rec["incumbent_identity"] is None
    assert rec["episodes"] is None


def test_the_DEADLINE_covers_the_MODEL_LOAD_too(tmp_path):
    t, t1j = [0.0], t1j_stub()

    def load():
        t[0] += 3601.0
        return inc_stub(), IDENTITY
    rec = stage2(tmp_path, t1j=t1j, load=load, clock=lambda: t[0])
    assert (rec["status"], rec["failure"]["failing_step"]) == ("STOP", "query:S1/t1j#0")
    assert sum(t1j.calls.values()) == 0 and rec["rows"] == []


def test_the_DEADLINE_is_enforced_MID_STEP_a_hung_compile_is_terminated(tmp_path):
    t0 = time.monotonic()
    rec = stage2(tmp_path, compile_fn=lambda dl: time.sleep(30), wall_s=0.3)
    assert time.monotonic() - t0 < 10, "the hung compile was not interrupted"
    assert (rec["status"], rec["failure"]["failing_step"], rec["failure"]["exception_type"]) \
        == ("STOP", "compile", "D1VoidError")


def test_the_STAGE2_COMPILE_runs_under_THE_stage_deadline_and_must_equal_the_bound_toolchain(
        monkeypatch):
    from scripts.GPU.alphazero import d1_probe as D1
    seen, content = [], {"value": dict(TOOL)}
    monkeypatch.setattr(C.R4, "_verified_t1j_paths", lambda classes: PATHS)
    monkeypatch.setattr(D1, "_default_compile",
                        lambda deadline, paths: seen.append((deadline, paths)) or {"id": 1})
    monkeypatch.setattr(C.R4, "split_toolchain", lambda ident: (content["value"], {}))
    monkeypatch.setattr(C, "_bound_toolchain", lambda: TOOL)
    stage_deadline = object()
    assert C._compile("/elsewhere/cls", stage_deadline) == (PATHS, TOOL)
    assert seen[0][0] is stage_deadline and seen[0][1] is PATHS, \
        "the Stage-2 compile did not run under THE stage deadline"
    content["value"] = {"jar_sha256": "k" * 64}
    with pytest.raises(C.CapRefused, match="not H5's bound toolchain"):
        C._compile("/elsewhere/cls", stage_deadline)


# ── §8 A3.4: persistence, and a record on EVERY exit

def test_a_STAGE2_COMPILE_FAILURE_leaves_a_claimed_directory_and_a_STOP_record(tmp_path):
    def compile_fn(dl):
        e = RuntimeError("javac exited 1")
        e.stdout, e.stderr, e.returncode = "javac says\n", "error: cannot find symbol\n", 1
        raise e
    rec = stage2(tmp_path, compile_fn=compile_fn, load=lambda: pytest.fail("loaded"))
    f = rec["failure"]
    assert (rec["status"], f["failing_step"], f["exception_type"]) == \
        ("STOP", "compile", "RuntimeError")
    assert (tmp_path / C.STAGE2_DIR / f["streams"]["stderr"]["file"]).read_text() == \
        "error: cannot find symbol\n" and f["rc"] == 1, "the compiler's output was lost"
    assert rec["compile"] is None and rec["compile_note"] and rec["rows"] == []
    assert rec["calls"] == {"t1j": "0/6", "incumbent": "0/4"} and rec["episodes"] is None
    with pytest.raises(C.CapRefused, match="occupied"):
        stage2(tmp_path)


def test_a_MODEL_LOAD_FAILURE_is_a_STOP_naming_the_load_and_no_query_runs(tmp_path):
    t1j = t1j_stub()

    def load():
        raise FileNotFoundError("checkpoint not found")
    rec = stage2(tmp_path, t1j=t1j, load=load)
    assert (rec["status"], rec["failure"]["failing_step"], rec["failure"]["exception_type"]) \
        == ("STOP", "model_load", "FileNotFoundError")
    assert rec["compile"] == TOOL and rec["incumbent_identity"] is None
    assert sum(t1j.calls.values()) == 0 and log_steps(tmp_path) == ["compile"]


def test_each_ROW_is_PERSISTED_before_the_next_call(tmp_path):
    seen = []

    def reply(moves, tag, ev):
        seen.append(log_steps(tmp_path).count("call"))
        return {"move": [2, 10]}
    stage2(tmp_path, t1j=chooser({"S1/t1j": reply, "XT": (12, 4)}))
    assert seen == [0, 1, 2], "a completed row was not on disk when the next call started"


def test_an_INTERRUPTION_after_a_completed_call_is_recorded_and_re_raised(tmp_path):
    n = []

    def reply(moves, tag, ev):
        n.append(1)
        if len(n) == 2:
            raise KeyboardInterrupt
        return {"move": [2, 10]}
    with pytest.raises(KeyboardInterrupt):
        stage2(tmp_path, t1j=chooser({"S1/t1j": reply, "XT": (12, 4)}))
    rec = json.loads((tmp_path / C.STAGE2_DIR / "record.json").read_text())
    assert (rec["status"], rec["failure"]["failing_step"], len(rec["rows"])) == \
        ("INTERRUPTED", "query:S1/t1j#1", 1), "the interruption was not recorded"
    assert rec["failure"]["partial"]["step"] == "query:S1/t1j#1"
    assert log_steps(tmp_path) == ["compile", "model_load", "call"]
    assert (rec["episodes"], rec["controls_ok"]) == (None, None)


def test_a_call_that_returns_NO_MOVE_is_a_STOP(tmp_path):
    rec = stage2(tmp_path, inc=chooser({"S1/incumbent": None, "XD": (12, 3)}))
    assert (rec["status"], rec["failure"]["failing_step"]) == ("STOP", "decision:S1/incumbent#0")
    assert "no move was returned" in rec["failure"]["reason"]
    assert rec["failure"]["partial"]["processes_note"] == C.IN_PROCESS


def _drop_xd(ps):
    return [p for p in ps if p["id"] != "XD"]


def _twice_s1(ps):
    return ps + [p for p in ps if p["id"] == "S1/t1j"]


def _plus_xe(ps):
    return ps + [{**p, "id": "XE"} for p in ps if p["id"] == "XD"]


@pytest.mark.parametrize("change,missing,duplicated,extra,calls", [
    (_drop_xd, [["XD", "incumbent", 0], ["XD", "incumbent", 1]], [], [], ("6/6", "2/4")),
    (_twice_s1, [], [["S1/t1j", "t1j", k] for k in range(3)], [], ("9/6", "4/4")),
    (_plus_xe, [], [], [["XE", "incumbent", 0], ["XE", "incumbent", 1]], ("6/6", "6/4")),
])
def test_an_INEXACT_call_set_is_INCOMPLETE_names_what_is_wrong_and_invents_no_exception(
        tmp_path, monkeypatch, change, missing, duplicated, extra, calls):
    real = C._positions
    monkeypatch.setattr(C, "_positions", lambda fz: change(real(fz)))
    inc = chooser({"S1/incumbent": (13, 4), "XD": (12, 3), "XE": (12, 3)})
    rec = stage2(tmp_path, inc=inc)
    f = rec["failure"]
    assert (rec["status"], f["failing_step"], f["exception_type"]) == \
        ("INCOMPLETE", "completion_check", None)
    assert (f["missing"], f["duplicated"], f["extra"]) == (missing, duplicated, extra)
    assert (rec["calls"]["t1j"], rec["calls"]["incumbent"]) == calls
    assert (rec["episodes"], rec["controls_ok"]) == (None, None)
    assert "INCOMPLETE" in rec["episodes_note"]


# ── §8 A3.3: the diagnostic boundary, and the REAL §4B path behind it

def test_the_BOUNDARY_returns_THE_object_and_re_raises_THE_exception(monkeypatch):
    sentinel, boom, seen = ([], [], 0, "OUT\n"), OSError("java: not found"), []

    def fake(*a, **k):
        seen.append((a, k))
        if k.get("fail"):
            raise boom
        return sentinel
    monkeypatch.setattr(C.A, "query", fake)
    ev = {}
    keep = C.recording_query(ev)
    assert keep([(1, 2)], depth=6, inject_matchdata=True) is sentinel, "not THE returned object"
    assert seen[0] == (([(1, 2)],), {"depth": 6, "inject_matchdata": True}), "arguments changed"
    with pytest.raises(OSError) as e:
        keep([(1, 2)], fail=True)
    assert e.value is boom, "not THE raised exception"
    assert [q["exception_type"] for q in ev["queries"]] == [None, "OSError"]
    assert (ev["queries"][0]["stdout"], ev["queries"][0]["rc"]) == ("OUT\n", 0)


@pytest.fixture
def served(monkeypatch):
    """T1j's helper served at the PROCESS BOUNDARY: each java call pops the next reply,
    `(rc, stdout)` or an exception. Spawns nothing; every other process (git) is real."""
    real_run, box = subprocess.run, {"replies": [], "argv": []}

    def fake_run(args, **kw):
        if args[0] != PATHS.java:
            return real_run(args, **kw)
        box["argv"].append(list(args))
        r = box["replies"].pop(0)
        if isinstance(r, BaseException):
            raise r
        return subprocess.CompletedProcess(args, r[0], r[1], "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    return box


def searched(move):
    return Q.SEARCHED_EXIT, query_out(ONE, native=False, move=move)


def stage2_real_t1j(tmp_path, served, replies):
    served["replies"] = list(replies)
    C._claim(tmp_path / C.STAGE2_DIR)
    return C._stage2(frozen(), PERMIT, STAGE1, lambda dl: (PATHS, TOOL),
                     lambda: (inc_stub(), IDENTITY), C._t1j_choose, tmp_path)


def stored(tmp_path, entry, s):
    st = entry["streams"][s]
    return None if st is None else (tmp_path / C.STAGE2_DIR / st["file"]).read_text()


def test_4B_PATH_an_accepted_query_keeps_its_FULL_stdout_rc_and_observations(tmp_path,
                                                                                served):
    replies = [searched((2, 10))] * 3 + [searched((12, 4))] * 3
    rec = stage2_real_t1j(tmp_path, served, replies)
    assert (rec["status"], rec["controls_ok"]) == ("COMPLETE", True), rec["failure"]
    row = rec["rows"][0]
    proc = row["processes"][0]
    assert stored(tmp_path, proc, "stdout") == replies[0][1], "the full stdout was not kept"
    assert (proc["rc"], proc["streams"]["stderr"], proc["stderr_note"]) == \
        (Q.SEARCHED_EXIT, None, C.NO_QUERY_STDERR)
    assert [(o["outcome"], o["source"]) for o in row["observations"]] == \
        [("accepted", Q.SEARCHED)]
    assert len(served["argv"]) == 6 and all("h4query" in a for a in served["argv"]), \
        "not the §4B opt-in query"


def test_4B_PATH_a_4B_REFUSAL_is_a_STOP_that_keeps_the_transcript(tmp_path, served):
    bad = (Q.SEARCHED_EXIT, query_out(ONE, native=False, move=(2, 10), failures=1))
    rec = stage2_real_t1j(tmp_path, served, [searched((2, 10)), bad])
    f = rec["failure"]
    assert (rec["status"], f["failing_step"], f["exception_type"]) == \
        ("STOP", "query:S1/t1j#1", "AbortError")
    proc = f["partial"]["processes"][0]
    assert stored(tmp_path, proc, "stdout") == bad[1] and proc["rc"] == Q.SEARCHED_EXIT
    obs = f["partial"]["observations"]
    assert [(o["outcome"], o["refused_at"]) for o in obs] == [("refused", "classify")]
    assert len(rec["rows"]) == 1 and len(served["argv"]) == 2, "a call ran after the STOP"


def test_4B_PATH_a_TIMEOUT_keeps_its_partial_output(tmp_path, served):
    to = subprocess.TimeoutExpired(["java"], 120, output=b"PROC partial\n",
                                   stderr=b"still searching\n")
    rec = stage2_real_t1j(tmp_path, served, [to])
    f = rec["failure"]
    assert (rec["status"], f["failing_step"], f["exception_type"]) == \
        ("STOP", "query:S1/t1j#0", "TimeoutExpired")
    proc = f["partial"]["processes"][0]
    assert (stored(tmp_path, proc, "stdout"), stored(tmp_path, proc, "stderr")) == \
        ("PROC partial\n", "still searching\n"), "the partial output was lost"
    assert proc["rc"] is None and "killed at the 120 s timeout" in proc["rc_note"]


def test_4B_PATH_MALFORMED_output_keeps_the_stdout_it_carried(tmp_path, served):
    garbled = "QUERY q=1 move_x=notanumber\n"
    rec = stage2_real_t1j(tmp_path, served, [(0, garbled)])
    f = rec["failure"]
    assert (rec["status"], f["failing_step"], f["exception_type"]) == \
        ("STOP", "query:S1/t1j#0", "HelperOutputError")
    proc = f["partial"]["processes"][0]
    assert stored(tmp_path, proc, "stdout") == garbled, "the carried stdout was lost"
    assert proc["rc"] is None and proc["rc_note"].startswith("unavailable: raised before")


def test_4B_PATH_an_UNEXPECTED_exception_is_a_STOP_with_every_field_unavailable(tmp_path,
                                                                                   served):
    rec = stage2_real_t1j(tmp_path, served, [OSError("java: not found")])
    f = rec["failure"]
    assert (rec["status"], f["failing_step"], f["exception_type"]) == \
        ("STOP", "query:S1/t1j#0", "OSError")
    proc = f["partial"]["processes"][0]
    assert proc["streams"] == {"stdout": None, "stderr": None}
    assert all(proc[k].startswith("unavailable") for k in ("stdout_note", "stderr_note",
                                                            "rc_note"))


# ── §8 A3.1: the PUBLIC entry in the real state

def public_stage2(open_gate, monkeypatch, *, commit_after=True):
    """Gate open IN THIS PROCESS; a committed Stage-1 record in its own repository, judged
    by the REAL permission check against THAT repository. Synthetic: the frozen set's
    re-derivation, the bound toolchain, compile, model load. `_watch` stays REAL unless a
    test runs the child in-process (`inproc`)."""
    fz, d = committed_stage1(open_gate, commit_after=commit_after, name="x-EXPLORATORY")
    out = d.parent
    real_permission, spy = C.stage2_permission, {"compile": [], "load": []}
    monkeypatch.setattr(C, "_committed_frozen", lambda out: (fz, FSHA))
    monkeypatch.setattr(C, "_bound_toolchain", lambda: TOOL)
    monkeypatch.setattr(C, "stage2_permission",
                        lambda *a, _fixture: real_permission(*a, _fixture=True))

    def compile_(classes, dl):
        spy["compile"].append((out / C.STAGE2_DIR / C.CLAIM_FILE).exists())
        return PATHS, TOOL

    def load():
        spy["load"].append((out / C.STAGE2_DIR / C.CLAIM_FILE).exists())
        return inc_stub(), IDENTITY
    monkeypatch.setattr(C, "_compile", compile_)
    monkeypatch.setattr(C, "_incumbent_choose", load)
    return out, spy


def inproc(monkeypatch):
    """The child run IN THIS PROCESS through its real CLI, with the nonce the parent hands
    it -- so the fakes above reach it. The argv must be the real child invocation."""
    def watch(argv, *, limit_s, env, cwd):
        assert list(argv[1:4]) == ["-m", C.MODULE, "stage2-claimed"], argv
        monkeypatch.setenv(C.CLAIM_ENV, env[C.CLAIM_ENV])
        t0 = time.monotonic()
        rc = C.main(list(argv[3:]))
        return {"child_pid": os.getpid(), "limit_s": limit_s, "terminated": False,
                "cause": None, "signal": None, "rc": rc, "elapsed_s": time.monotonic() - t0,
                "survivors_killed": False}
    monkeypatch.setattr(C, "_watch", watch)


def test_PUBLIC_ENTRY_runs_Stage_2_in_the_real_state(open_gate, monkeypatch, served):
    out, spy = public_stage2(open_gate, monkeypatch)
    inproc(monkeypatch)
    served["replies"] = [searched((2, 10))] * 3 + [searched((12, 4))] * 3
    res = C.run_stage2(str(out), str(open_gate / "cls"))
    d = out / C.STAGE2_DIR
    rec = json.loads((d / "record.json").read_text())
    assert (res["status"], res["record_status"], res["child_rc"]) == \
        ("COMPLETE", "COMPLETE", 0), rec["failure"]
    assert res == json.loads((d / C.WATCHDOG_FILE).read_text()), \
        "the parent returned something other than what it recorded"
    assert (rec["calls"], rec["controls_ok"]) == ({"t1j": "6/6", "incumbent": "4/4"}, True)
    assert rec["episodes"]["S1"]["reading"] == "AGENT_CHOICE (reproduced)"
    assert rec["stage1"]["record_sha256"] == hashlib.sha256(
        (out / C.STAGE1_DIR / "record.json").read_bytes()).hexdigest()
    assert rec["stage1"]["claim"] == json.loads((d / C.CLAIM_FILE).read_text())
    assert 3500 < rec["deadline_s"] <= C.STAGE2_WALL_S - C.RECORD_MARGIN_S, \
        "the child's deadline is not the claim's, less the record margin"
    assert spy == {"compile": [True], "load": [True]}, "a compile or load ran unclaimed or twice"
    assert res["journal"]["lines"] == 2 + 10


def test_PUBLIC_ENTRY_refuses_on_PERMISSION_before_any_claim_compile_or_load(open_gate,
                                                                            monkeypatch):
    out, spy = public_stage2(open_gate, monkeypatch, commit_after=False)
    monkeypatch.setattr(C, "_watch", lambda *a, **k: pytest.fail("a child was started"))
    with pytest.raises(C.CapRefused, match="the Stage-1 record is not durable"):
        C.run_stage2(str(out), str(open_gate / "cls"))
    assert not (out / C.STAGE2_DIR).exists() and spy == {"compile": [], "load": []}


@pytest.mark.parametrize("occupied", ["stage2", "classes"])
def test_PUBLIC_ENTRY_refuses_an_OCCUPIED_destination_before_the_permission(open_gate,
                                                                          monkeypatch,
                                                                          occupied):
    out, spy = public_stage2(open_gate, monkeypatch)
    monkeypatch.setattr(C, "stage2_permission", lambda *a, **k: pytest.fail("permission"))
    cls = open_gate / "cls"
    (out / C.STAGE2_DIR if occupied == "stage2" else cls).mkdir()
    with pytest.raises(C.CapRefused, match="occupied"):
        C.run_stage2(str(out), str(cls))
    assert spy == {"compile": [], "load": []}
    if occupied == "stage2":
        assert list((out / C.STAGE2_DIR).iterdir()) == [], "the occupied directory was written"
    else:
        assert not (out / C.STAGE2_DIR).exists(), "Stage 2 was claimed past an occupied refusal"


def test_the_CHILD_takes_only_an_UNCONSUMED_claim_with_ITS_nonce(tmp_path, monkeypatch):
    d = tmp_path / C.STAGE2_DIR
    with pytest.raises(C.CapRefused, match="unconsumed claim"):
        C._accept_claim(d)                       # nothing claimed
    claim, nonce = C._claim(d)
    assert nonce not in (d / C.CLAIM_FILE).read_text(), "the nonce itself was written down"
    monkeypatch.delenv(C.CLAIM_ENV, raising=False)
    with pytest.raises(C.CapRefused, match="nonce does not match"):
        C._accept_claim(d)                       # no nonce handed over
    monkeypatch.setenv(C.CLAIM_ENV, "0" * 32)
    with pytest.raises(C.CapRefused, match="nonce does not match"):
        C._accept_claim(d)                       # another parent's nonce
    monkeypatch.setenv(C.CLAIM_ENV, nonce)
    assert C._accept_claim(d) == claim
    (d / "log.jsonl").write_text("")             # consumed by its child
    with pytest.raises(C.CapRefused, match="unconsumed claim"):
        C._accept_claim(d)


# ── §9 A4.1: the hard deadline -- an external parent watchdog, harmless dummy processes only

#: blocks EVERY signal, so no Python handler of the child's can run, then hangs in C
MASKED_CHILD = r'''
import os, signal, subprocess, sys, time
signal.pthread_sigmask(signal.SIG_BLOCK, signal.valid_signals())
g = subprocess.Popen(["/bin/sleep", "60"])
open(sys.argv[1] + ".tmp", "w").write(f"{g.pid}\n")
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
time.sleep(60)
'''
#: leaves its own subprocess running and exits normally
SURVIVOR_CHILD = r'''
import os, subprocess, sys
g = subprocess.Popen(["/bin/sleep", "60"])
open(sys.argv[1] + ".tmp", "w").write(f"{g.pid}\n")
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
'''


def gone(pid, within=5.0):
    end = time.monotonic() + within
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def watch(tmp_path, code, *args, limit_s):
    return C._watch([sys.executable, "-c", code, *args], limit_s=limit_s, env=os.environ,
                    cwd=str(tmp_path))


def test_WATCHDOG_kills_a_child_with_EVERY_SIGNAL_BLOCKED_and_its_own_subprocess(tmp_path):
    """The control: detection needs nothing from the child. Every catchable signal is
    blocked there and it hangs in a C call; the parent measures, the kernel kills."""
    pidfile, t0 = tmp_path / "grandchild.pid", time.monotonic()
    w = watch(tmp_path, MASKED_CHILD, str(tmp_path / "grandchild.pid"), limit_s=3.0)
    assert time.monotonic() - t0 < 15, "the hung child was not killed near its limit"
    assert (w["terminated"], w["signal"], w["rc"]) == (True, "SIGKILL", -9)
    assert "hard limit" in w["cause"]
    assert pidfile.exists(), "the dummy never started its subprocess: the control is vacuous"
    assert gone(int(pidfile.read_text())), "the child's own subprocess survived the kill"


def test_WATCHDOG_returns_the_status_of_a_child_that_finishes(tmp_path):
    w = watch(tmp_path, "import sys; sys.exit(3)", limit_s=30)
    assert (w["terminated"], w["rc"], w["signal"], w["survivors_killed"], w["cause"]) == \
        (False, 3, None, False, None)


def test_WATCHDOG_kills_a_SURVIVING_subprocess_after_a_normal_exit_and_records_it(tmp_path):
    pidfile = tmp_path / "grandchild.pid"
    w = watch(tmp_path, SURVIVOR_CHILD, str(pidfile), limit_s=30)
    assert (w["terminated"], w["rc"], w["survivors_killed"]) == (False, 0, True)
    assert gone(int(pidfile.read_text())), "a subprocess outlived the run"


def test_WATCHDOG_an_INTERRUPTED_parent_kills_the_group_and_hands_back_the_interrupt(
        tmp_path, monkeypatch):
    pidfile, real_wait = tmp_path / "grandchild.pid", subprocess.Popen.wait

    def wait(self, timeout=None):
        if timeout is None:
            return real_wait(self)
        end = time.monotonic() + 10
        while not pidfile.exists() and time.monotonic() < end:
            time.sleep(0.05)
        raise KeyboardInterrupt
    monkeypatch.setattr(subprocess.Popen, "wait", wait)
    w = watch(tmp_path, MASKED_CHILD, str(pidfile), limit_s=30)
    assert (w["terminated"], w["signal"], w["rc"]) == (True, "SIGKILL", -9)
    assert isinstance(w["parent_exception"], KeyboardInterrupt) and "interrupted" in w["cause"]
    assert gone(int(pidfile.read_text())), "the child's subprocess survived the interrupt"


#: a dummy child that writes a journal line, then (by mode) a COMPLETE record, an exit,
#: or a hang with every signal blocked
DUMMY_CHILD = r'''
import json, os, signal, sys, time
d, mode = sys.argv[1], sys.argv[2]
with open(os.path.join(d, "log.jsonl"), "x") as fh:
    fh.write(json.dumps({"step": "compile", "observation": {}}) + "\n")
if mode == "complete":
    with open(os.path.join(d, "record.json"), "x") as fh:
        json.dump({"status": "COMPLETE"}, fh)
if mode == "exit":
    sys.exit(0)
signal.pthread_sigmask(signal.SIG_BLOCK, signal.valid_signals())
time.sleep(60)
'''


@pytest.mark.parametrize("mode,status,signal_rc", [
    ("hang", "TERMINATED", ("SIGKILL", -9)),
    ("complete", "TERMINATED", ("SIGKILL", -9)),
    ("exit", "EXITED_WITHOUT_RECORD", (None, 0)),
])
def test_PUBLIC_ENTRY_the_PARENT_settles_a_dummy_child_and_never_reports_COMPLETE(
        open_gate, monkeypatch, mode, status, signal_rc):
    out, spy = public_stage2(open_gate, monkeypatch)
    d = out / C.STAGE2_DIR
    monkeypatch.setattr(C, "_child_argv",
                        lambda o, cls: [sys.executable, "-c", DUMMY_CHILD, str(d), mode])
    monkeypatch.setattr(C, "STAGE2_WALL_S", 3.0)
    t0 = time.monotonic()
    res = C.run_stage2(str(out), str(open_gate / "cls"))
    assert time.monotonic() - t0 < 20, "the parent did not enforce its limit"
    assert res["status"] == status and res == json.loads((d / C.WATCHDOG_FILE).read_text())
    assert (res["signal"], res["child_rc"]) == signal_rc
    journal = (d / "log.jsonl").read_bytes()
    assert journal == json.dumps({"step": "compile", "observation": {}}).encode() + b"\n", \
        "the journal was altered"
    assert res["journal"] == {"file": "log.jsonl", "lines": 1,
                              "sha256": hashlib.sha256(journal).hexdigest()}
    if mode == "complete":
        assert res["record_status"] == "COMPLETE" and \
            json.loads((d / "record.json").read_text()) == {"status": "COMPLETE"}, \
            "the child's record was touched"
    else:
        assert not (d / "record.json").exists() and res["record_status"] is None


def test_PUBLIC_ENTRY_a_child_that_CANNOT_START_still_gets_the_parents_record(open_gate,
                                                                              monkeypatch):
    out, spy = public_stage2(open_gate, monkeypatch)
    monkeypatch.setattr(C, "_child_argv", lambda o, cls: ["/nonexistent/python"])
    res = C.run_stage2(str(out), str(open_gate / "cls"))
    assert (res["status"], res["child_pid"], res["child_rc"]) == \
        ("EXITED_WITHOUT_RECORD", None, None)
    assert "could not be started" in res["cause"]
    assert res == json.loads((out / C.STAGE2_DIR / C.WATCHDOG_FILE).read_text())


def test_the_MONOTONIC_clock_the_child_reads_is_the_parents(tmp_path):
    """The child's deadline is computed from the parent's `claimed_monotonic`: that
    needs ONE system-wide monotonic clock across processes. Checked, not assumed."""
    before = time.monotonic()
    r = subprocess.run([sys.executable, "-c", "import time; print(repr(time.monotonic()))"],
                       capture_output=True, text=True, check=True)
    after = time.monotonic()
    assert before <= float(r.stdout) <= after, "a child process reads a different clock"


def test_PUBLIC_ENTRY_an_INTERRUPTED_parent_records_TERMINATED_and_re_raises(open_gate,
                                                                            monkeypatch):
    out, spy = public_stage2(open_gate, monkeypatch)
    stop = KeyboardInterrupt()
    monkeypatch.setattr(C, "_watch", lambda argv, **k: {
        "child_pid": 1, "limit_s": k["limit_s"], "terminated": True, "signal": "SIGKILL",
        "cause": "the parent was interrupted (KeyboardInterrupt)", "rc": -9, "elapsed_s": 0.1,
        "survivors_killed": False, "parent_exception": stop})
    with pytest.raises(KeyboardInterrupt) as e:
        C.run_stage2(str(out), str(open_gate / "cls"))
    assert e.value is stop
    w = json.loads((out / C.STAGE2_DIR / C.WATCHDOG_FILE).read_text())
    assert (w["status"], w["journal"]) == ("TERMINATED", None) and w["journal_note"]


def test_PUBLIC_ENTRY_the_REAL_child_invocation_reads_the_CLOSED_on_disk_gate(open_gate,
                                                                             monkeypatch):
    """The parent's gate is open IN THIS PROCESS only. The child is a fresh interpreter,
    started by the real argv, reading the published gate (False): it exits 5, doing
    nothing, and the parent says so."""
    out, spy = public_stage2(open_gate, monkeypatch)
    res = C.run_stage2(str(out), str(open_gate / "cls"))
    d = out / C.STAGE2_DIR
    assert (res["status"], res["child_rc"], res["signal"]) == \
        ("EXITED_WITHOUT_RECORD", C.EXIT_UNAUTHORIZED, None)
    assert sorted(p.name for p in d.iterdir()) == [C.CLAIM_FILE, C.WATCHDOG_FILE]
    assert spy == {"compile": [], "load": []}


# ── §9 A4.3: the loader's wiring, with fakes (no model is loaded)

def test_the_REAL_LOADER_loads_ONCE_and_gives_each_colour_its_task_at_seed_minus_1(
        monkeypatch):
    bound = json.loads((ROOT / C.MANIFEST).read_text())["segments"][0]["incumbent_identity"]
    config, evaluator, loads, built = object(), object(), [], []
    monkeypatch.setattr(C.R4, "frozen_argmax_config", lambda: config)

    def identity_of(cfg, *, design):
        assert cfg is config and design == C.H5.STUDY_DESIGN
        return bound
    monkeypatch.setattr(C.R4, "frozen_incumbent_identity", identity_of)
    monkeypatch.setattr(C.R4, "check_incumbent_identity", lambda ident, cfg, *, design: None)
    monkeypatch.setattr(C.R4, "load_production_evaluator",
                        lambda ident: loads.append(ident) or evaluator)

    def build_for(cfg):
        assert cfg is config, "the builder was not given THE config"

        def build(task, ev):
            def agent(state):
                built.append((task, ev, state.to_move))
                return sorted(state.legal_moves())[0]
            return agent
        return build
    monkeypatch.setattr(C.R4, "production_incumbent_build", build_for)
    choose, identity = C._incumbent_choose()
    assert callable(choose) and identity is bound and loads == [bound]
    red, black = choose(ONE[:22], "R#0", {}), choose(ONE[:23], "B#0", {})
    assert len(loads) == 1, "the model was loaded more than once"
    assert [(t["incumbent_colour"], t["seed"], ev is evaluator, mover)
            for t, ev, mover in built] == [("red", -1, True, "red"), ("black", -1, True, "black")]
    assert (red["seed"], black["seed"]) == (-1, -1) and red["move"] and black["move"]


def test_the_LOADER_refuses_an_UNBOUND_identity_BEFORE_any_load(monkeypatch):
    monkeypatch.setattr(C.R4, "frozen_argmax_config", lambda: object())
    monkeypatch.setattr(C.R4, "frozen_incumbent_identity",
                        lambda cfg, *, design: {"reference": "another"})
    monkeypatch.setattr(C.R4, "check_incumbent_identity", lambda *a, **k: None)
    monkeypatch.setattr(C.R4, "load_production_evaluator", lambda ident: pytest.fail("loaded"))
    with pytest.raises(C.CapRefused, match="not H5's bound identity"):
        C._incumbent_choose()


# ─────────────────────────────── the public path, real state ───────────────────────────────

@pytest.fixture
def open_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(AUTH, "H5_CAP_COMPLETION_AUTHORIZED", True)
    monkeypatch.setattr(R4, "EVIDENCE_ROOT", str(tmp_path))
    return tmp_path


def test_a_stage_REFUSES_an_uncommitted_frozen_set_before_any_compile(open_gate, monkeypatch):
    out = open_gate / "x-EXPLORATORY"
    out.mkdir()
    (out / C.FROZEN_FILE).write_text("{}\n")
    monkeypatch.setattr(C, "_verified_paths", lambda *a: pytest.fail("compiled"))
    monkeypatch.setattr(C, "_compile", lambda *a: pytest.fail("compiled"))
    for entry in (C.run_stage1, C.run_stage2):
        with pytest.raises(C.CapRefused, match="must be committed before any JVM"):
            entry(str(out), str(open_gate / "cls"))
    assert sorted(p.name for p in out.iterdir()) == [C.FROZEN_FILE], "a refusal wrote"


def test_the_evidence_directory_must_say_EXPLORATORY(open_gate):
    with pytest.raises(C.CapRefused, match="EXPLORATORY"):
        C.freeze(str(open_gate / "plain"))


# ─────────────────────────────── the REAL frozen set (read-only) ───────────────────────────────

@pytest.fixture(scope="module")
def real():
    from scripts.GPU.alphazero import h5_weakness_review as WR
    games, _ = WR.load_and_verify()
    return C.build_frozen_set(games)


def test_the_REAL_frozen_set_is_the_cards(real):
    eps = real["episodes"]
    assert {e: (v["frequency"], len(v["members"])) for e, v in eps.items()} == \
        {"E1": (7, 7), "E2": (5, 5), "E3": (1, 1)}, "the 13 provenance links moved"
    assert sorted(real["controls"]) == ["CD1", "CD2", "CI1", "CI2", "CT1", "CT2", "CT3"]
    assert real["controls"]["CD2"]["threat"] == [[13, 5]], "CD2's pinned threat moved"
    assert len(C.stage1_items(real)) == 18, "the Stage-1 bound is not 18 items"
    calls = collections.Counter(a for _p, a, _k in C.stage2_expected(real))
    assert calls == {"t1j": 18, "incumbent": 14}, "the Stage-2 call set is not 18 + 14"


def test_FREEZE_through_the_public_entry_writes_exactly_the_real_set_create_only(open_gate,
                                                                                   real):
    out = open_gate / "x-EXPLORATORY"
    path = C.freeze(str(out))
    assert pathlib.Path(path).read_bytes() == C.frozen_bytes(real), "the frozen file drifted"
    with pytest.raises(C.CapRefused, match="occupied"):
        C.freeze(str(out))
