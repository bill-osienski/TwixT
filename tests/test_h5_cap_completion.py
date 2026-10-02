"""H5 cap-completion diagnostic (EXPLORATORY) -- card §1-§6 with §6 A1.

SYNTHETIC FIXTURES for everything on the T1j and incumbent side: no JVM, no compile, no
model, no query. T1j's replay is faked FROM OUR OWN ENGINE (a perfectly agreeing T1j),
then broken one field at a time. One test re-derives the REAL frozen set from the
committed H5 records, read-only, and checks it against the card's pins. Messages fixed.
"""
import collections
import copy
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import h5_cap_completion as C
from scripts.GPU.alphazero import h5_cap_completion_authorization as AUTH
from scripts.GPU.alphazero import h4_runner as R4
from scripts.GPU.alphazero import t1j_adapter as A
from scripts.GPU.alphazero.game.twixt_state import TwixtState
from tests.test_h4_pilot_feasibility import commit, new_repo

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
BINDING = {"frozen_set_sha256": "f" * 64, "toolchain": {"jar_sha256": "j" * 64}}


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


@pytest.mark.parametrize("entry", ["freeze", "run_stage1", "run_stage2"])
def test_every_public_entry_refuses_with_the_gate_shut_BEFORE_anything(entry, monkeypatch,
                                                                        tmp_path):
    for owner, name in ((C, "_frozen_from_records"), (C, "_verified_paths"),
                        (C.F, "check_durable"), (os, "makedirs")):
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


def test_a_FAITHFUL_T1j_agrees_on_every_item(tmp_path):
    fz = frozen()
    rec = C._stage1(fz, binding(fz), faithful, tmp_path)
    assert rec["all_agree"] and len(rec["items"]) == 7, "a faithful T1j did not agree"
    t = tmp_path / C.STAGE1_DIR / rec["items"][0]["transcript"]
    assert hashlib.sha256(t.read_bytes()).hexdigest() == rec["items"][0]["transcript_sha256"]
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


# ─────────────────────────────── Stage-2 permission (§6 A1.2) ───────────────────────────────

def committed_stage1(tmp_path, mutate=None, commit_after=True):
    """A Stage-1 record in its own temporary git repository, optionally edited."""
    repo = new_repo(tmp_path / "repo")
    fz = frozen()
    C._stage1(fz, binding(fz), faithful, repo)
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
    ("transcript hash", lambda r, d: r["items"][1].update(transcript_sha256="0" * 64),
     "does not hash to its record"),
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
    t = sorted((d / "transcripts").iterdir())[0]
    t.write_text("edited after the commit\n")
    with pytest.raises(C.CapRefused, match="transcript not durable"):
        C.stage2_permission(d, fz, binding(fz), _fixture=True)


# ─────────────────────────────── Stage 2 (§3, §6 A1.1) ───────────────────────────────

def chooser(replies):
    calls = collections.Counter()

    def choose(moves, tag):
        pos = tag.split("#")[0]
        calls[pos] += 1
        r = replies[pos]
        if isinstance(r, Exception):
            raise r
        return {"move": list(r), "stub": True}
    choose.calls = calls
    return choose


def test_STAGE2_judges_by_the_rules_and_reports_history_apart(tmp_path):
    fz = frozen()
    t1j = chooser({"S1/t1j": (2, 10), "XT": (12, 4)})
    inc = chooser({"S1/incumbent": (13, 4), "XD": (12, 3)})
    rec = C._stage2(fz, {"all_agree": True}, t1j, inc, tmp_path)
    assert rec["controls_ok"], "valid alternative replies failed their controls"
    ep = rec["episodes"]["S1"]
    assert (ep["t1j_completes"], ep["reading"]) == ("0/3", "AGENT_CHOICE (reproduced)")
    assert ep["incumbent_blocks"] == "2/2" and ep["incumbent_equals_recorded"] == 0, \
        "an alternative block was judged by history"
    assert (ep["members"], ep["frequency"]) == (["s1-B", "s2-B"], 2)
    assert t1j.calls == {"S1/t1j": 3, "XT": 3} and inc.calls == {"S1/incumbent": 2, "XD": 2}


def test_a_FAILED_CONTROL_makes_every_episode_uninterpretable(tmp_path):
    t1j = chooser({"S1/t1j": (2, 10), "XT": (2, 10)})
    inc = chooser({"S1/incumbent": (2, 21), "XD": (12, 3)})
    rec = C._stage2(frozen(), {"all_agree": True}, t1j, inc, tmp_path)
    assert not rec["controls_ok"] and rec["episodes"]["S1"]["reading"] == "uninterpretable"


def test_a_STOP_is_recorded_and_ends_the_stage(tmp_path):
    t1j = chooser({"S1/t1j": C.CapStop("the §4B path refused"), "XT": (12, 4)})
    inc = chooser({"S1/incumbent": (13, 4), "XD": (12, 3)})
    rec = C._stage2(frozen(), {"all_agree": True}, t1j, inc, tmp_path)
    assert rec["stop"].startswith("S1/t1j#0") and not rec["controls_ok"]
    assert sum(t1j.calls.values()) == 1, "queries continued after a STOP"


def test_STAGE2_refuses_without_an_all_AGREE_permission(tmp_path):
    with pytest.raises(C.CapRefused, match="all-AGREE"):
        C._stage2(frozen(), {"all_agree": False}, chooser({}), chooser({}), tmp_path)
    assert not (tmp_path / C.STAGE2_DIR).exists()


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


def test_FREEZE_through_the_public_entry_writes_exactly_the_real_set_create_only(open_gate,
                                                                                   real):
    out = open_gate / "x-EXPLORATORY"
    path = C.freeze(str(out))
    assert pathlib.Path(path).read_bytes() == C.frozen_bytes(real), "the frozen file drifted"
    with pytest.raises(C.CapRefused, match="occupied"):
        C.freeze(str(out))
