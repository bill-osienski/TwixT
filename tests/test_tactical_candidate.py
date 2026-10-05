"""Tactical candidate -- the default-off wrapper and the independent checker (card §1-§2).

SYNTHETIC ONLY: no model is loaded and no real decision is made. Hand-built trees on pinned
24x24 positions, and the REAL frozen search driven by the steered stub evaluator. Every position
here already exists in the H5 tests or fixtures: these are REGRESSION controls, not fresh
qualification evidence (card §2, §4).
"""
import ast
import inspect

import pytest

from scripts.GPU.alphazero import h5_incumbent_defence as D
from scripts.GPU.alphazero import h5_incumbent_defence_factors as X
from scripts.GPU.alphazero import h5_incumbent_tactics as T
from scripts.GPU.alphazero import h5_weakness_review as WR
from scripts.GPU.alphazero import h5_cap_completion as CC
from scripts.GPU.alphazero import tactical_candidate as TC
from scripts.GPU.alphazero import tactical_candidate_authorization as AUTH
from scripts.GPU.alphazero.game.twixt_state import TwixtState
from tests.test_h5_cap_completion import TWO
from tests.test_h5_incumbent_defence import BY_ID, CONFIG, FIX, IDENT, Steered
from tests.test_h5_incumbent_tactics import (
    DRAW24, MULTI_269, OWN_WIN_279, OWN_WIN_AND_THREAT_264, QUIET_40, SINGLE_260, THREAT_279,
    UNDEFENDABLE_256, child, counts_of, other, threat279_tree, tree_root)

WITNESS_KEYS = {"ply", "colour", "class", "rule", "reason", "override", "unsearched", "baseline",
                "final", "value_note", "scan_s"}


def test_the_gate_is_created_CLOSED():
    assert AUTH.TACTICAL_CANDIDATE_QUALIFICATION_AUTHORIZED is False


def test_the_constants_are_the_tactics_cards():
    assert (TC.BOARD, TC.PLY_CAP) == (T.BOARD, T.PLY_CAP) == (24, 280)


# ─────────────────────────────── the frozen M1 readout, on hand-built trees ───────────────────

def tree(moves, visits, values=None):
    """A 24x24 root with one child per visited move; `values` = {move: value_sum}."""
    s = T.replay(moves)
    root = tree_root(s)
    for m, v in visits.items():
        n = child(root, m, visits=v)
        n.value_sum = (values or {}).get(m, 0.0)
    return s, root, counts_of(s, visits)


def test_an_OWN_WIN_takes_precedence_and_a_zero_visit_win_is_UNSEARCHED():
    s = T.replay(OWN_WIN_AND_THREAT_264)
    block = other(s, CC.immediate_wins(s))
    s, root, counts = tree(OWN_WIN_AND_THREAT_264, {block: 50, (4, 7): 3})
    w = TC.witness(s, counts, root, block)
    assert set(w) == WITNESS_KEYS
    assert (w["rule"], w["final"]["move"], w["override"], w["unsearched"]) == ("override", [4, 7], True, False)
    assert w["reason"] == TC.REASON["OWN_WIN"] and w["class"]["class"] == "OWN_WIN"
    s, root, counts = tree(OWN_WIN_AND_THREAT_264, {block: 50})
    w = TC.witness(s, counts, root, block)
    assert (w["final"]["move"], w["unsearched"], w["final"]["visits"], w["final"]["q_root_mover"]) == \
        ([4, 6], True, 0, None), "an unvisited win: the (row, col) minimum, recorded unsearched"


def test_MULTIPLE_threats_take_the_COMMON_defence_even_unsearched():
    s = T.replay(MULTI_269)
    out = other(s, T.classify(s, semantics="protocol")["D"])
    s, root, counts = tree(MULTI_269, {out: 30})
    w = TC.witness(s, counts, root, out)
    assert (w["rule"], w["final"]["move"], w["unsearched"]) == ("override", [11, 17], True)
    assert w["reason"] == TC.REASON["DEFEND_MULTI"]


def test_ALTERNATIVE_defences_take_the_most_visited():
    s = T.replay(SINGLE_260)
    out = other(s, T.classify(s, semantics="protocol")["D"])
    s, root, counts = tree(SINGLE_260, {(9, 10): 7, (10, 10): 9, out: 100})
    w = TC.witness(s, counts, root, out)
    assert (w["rule"], w["final"]["move"], w["final"]["visits"], w["unsearched"]) == \
        ("override", [10, 10], 9, False)


def test_an_ADMISSIBLE_baseline_is_PRESERVED():
    s, root, counts = tree(SINGLE_260, {(9, 10): 50, (10, 10): 9})
    w = TC.witness(s, counts, root, (9, 10))
    assert (w["rule"], w["final"]["move"], w["override"], w["reason"]) == ("preserved", [9, 10], False, None)


@pytest.mark.parametrize("moves", [UNDEFENDABLE_256, QUIET_40], ids=["UNDEFENDABLE", "QUIET"])
def test_UNDEFENDABLE_and_QUIET_keep_the_baseline(moves):
    s = T.replay(moves)
    some = sorted(s.legal_moves())[3]
    s, root, counts = tree(moves, {some: 5})
    w = TC.witness(s, counts, root, some)
    assert (w["rule"], w["final"]["move"], w["override"]) == ("inactive", list(some), False)


def test_the_REAL_279_280_BOUNDARY_under_the_protocol_only():
    s = T.replay(OWN_WIN_279)                    # black at ply 279: the last allowed move
    block = other(s, [(6, 19)])
    s, root, counts = tree(OWN_WIN_279, {block: 40})
    w = TC.witness(s, counts, root, block)
    assert (w["rule"], w["final"]["move"]) == ("override", [6, 19]), "a winning 280th move wins"
    s, root, bad, dfn, counts = threat279_tree()  # red's reply would be the move at ply 280
    w = TC.witness(s, counts, root, bad)
    assert (w["class"]["class"], w["class"]["T_void"], w["rule"], w["final"]["move"]) == \
        ("QUIET", [[7, 3]], "inactive", list(bad)), "the board-only defence is never applied"
    at_cap = T.replay(OWN_WIN_279 + [block])
    s2, root2, counts2 = tree(OWN_WIN_279 + [block], {other(at_cap): 1})
    w = TC.witness(s2, counts2, root2, other(at_cap))
    assert (w["class"], w["rule"]) == (None, "inactive"), "no decision position at ply 280"


def test_VALUES_are_the_ROOT_MOVERs_and_explicitly_labelled():
    s = T.replay(SINGLE_260)
    bad = other(s, T.classify(s, semantics="protocol")["D"])
    s, root, counts = tree(SINGLE_260, {bad: 9, (9, 10): 3}, {bad: 4.5, (9, 10): -0.6})
    w = TC.witness(s, counts, root, bad)
    assert (w["baseline"]["q_node"], w["baseline"]["q_root_mover"]) == (0.5, -0.5)
    assert w["final"]["move"] == [9, 10] and w["final"]["q_root_mover"] == pytest.approx(0.2)
    assert "ROOT's mover" in w["value_note"] and isinstance(w["scan_s"], float) and w["scan_s"] >= 0


def test_a_baseline_that_is_NOT_the_visit_argmax_is_refused():
    s, root, counts = tree(SINGLE_260, {(9, 10): 50, (10, 10): 9})
    with pytest.raises(TC.CandidateRefused, match="not the visit argmax"):
        TC.witness(s, counts, root, (10, 10))


# ─────────────────────────────── the wrapper: default off, refusals ───────────────────────────

class Dummy:
    """An agent that must never be asked to search."""
    colour, capture, last_capture = "red", True, None

    def __call__(self, state):
        pytest.fail("searched before the refusal")


def test_the_wrapper_is_DEFAULT_OFF_and_needs_the_captured_search():
    assert inspect.signature(TC.TacticalCandidate).parameters["enabled"].default is False
    with pytest.raises(TC.CandidateRefused, match="capture=True"):
        TC.TacticalCandidate(type("NoCapture", (), {"capture": False})())


@pytest.mark.parametrize("dims", [dict(active_size=8), dict(board_size=26, active_size=24)],
                         ids=["playable-8", "physical-26"])
def test_a_non_official_board_is_refused_BEFORE_any_search(dims):
    bad = TwixtState(to_move="red", **dims)
    for enabled in (False, True):
        with pytest.raises(T.TacticsRefused, match="not the official 24x24"):
            TC.TacticalCandidate(Dummy(), enabled=enabled)(bad)
    with pytest.raises(TC.CandidateRefused, match="not the official 24x24"):
        TC.exhaustive_scan(bad)


# ─────────────────────────────── the wrapper on the REAL search (stub evaluator) ──────────────

def builder():
    return D.study_builder(CONFIG, Steered(), D.tasks_for(IDENT))


def play(agent, positions, box):
    """Successive decisions by ONE agent instance: both streams advance across them."""
    out = []
    for p in positions:
        state = T.replay(p["moves"])
        move = agent(state)
        counts, root_value, root = box["results"][-1]
        inner = agent.agent if isinstance(agent, TC.TacticalCandidate) else agent
        out.append({"move": move, "fp": D.fingerprint(root, counts, root_value), "counts": counts,
                    "search": inner.mcts.rng.getstate(), "readout": inner.readout_rng.getstate(),
                    "diff": X.config_diff(inner, CONFIG), "state": state})
    return out


#: successive decisions per colour: the H5 fixtures (all black to move) and pinned positions
BY_COLOUR = {"red": [{"moves": m} for m in (SINGLE_260, OWN_WIN_AND_THREAT_264, TWO)],
             "black": list(FIX) + [{"moves": MULTI_269}]}


@pytest.mark.parametrize("colour", ["red", "black"])
def test_TREE_CONFIGURATION_and_BOTH_STREAMS_are_unchanged_on_the_real_search(colour):
    positions = BY_COLOUR[colour]
    assert all(T.replay(p["moves"]).to_move == colour for p in positions)
    plain = builder()(colour, False)
    cand = TC.TacticalCandidate(builder()(colour, True), enabled=True)
    a = play(plain, positions, D.retain_search(plain.mcts))
    b = play(cand, positions, D.retain_search(cand.agent.mcts))
    assert len(cand.witnesses) == len(positions)
    assert {"override", "preserved"} <= {w["rule"] for w in cand.witnesses}, \
        "both paths must be exercised, or the stream check is vacuous"
    for x, y, w in zip(a, b, cand.witnesses):
        assert (x["fp"], x["counts"], x["search"], x["readout"]) == (y["fp"], y["counts"], y["search"], y["readout"])
        assert x["diff"] == y["diff"] == set()
        assert w["baseline"]["move"] == list(x["move"]), "the baseline IS the incumbent's move"
        c = T.classify(y["state"], semantics="protocol")
        if c is None or c["class"] in TC.INACTIVE:
            expect = x["move"]
        else:
            A = T.admissible(c, list(x["counts"]))
            expect = x["move"] if x["move"] in A else tuple(T.m1_choice(c, x["counts"])["choice"])
            assert expect in A, "an active rule always ends on a win or a full defence"
        assert y["move"] == expect
        assert w["search_s"] > 0 and w["scan_s"] >= 0


def test_DISABLED_the_wrapper_IS_the_incumbent():
    positions = [p for p in FIX if T.replay(p["moves"]).to_move == "black"]
    plain = builder()("black", False)
    off = TC.TacticalCandidate(builder()("black", True))
    a = play(plain, positions, D.retain_search(plain.mcts))
    b = play(off, positions, D.retain_search(off.agent.mcts))
    assert [x["move"] for x in a] == [y["move"] for y in b] and off.witnesses == []
    assert [(x["search"], x["readout"]) for x in a] == [(y["search"], y["readout"]) for y in b]


# ─────────────────────────────── the independent exhaustive checker (card §2.1) ───────────────

def test_the_CHECKER_shares_NO_helper_with_the_optimized_detector():
    tree_ = ast.parse(inspect.getsource(TC.exhaustive_scan))
    names = {n.id for n in ast.walk(tree_) if isinstance(n, ast.Name)} | \
        {n.attr for n in ast.walk(tree_) if isinstance(n, ast.Attribute)}
    assert not names & {"CC", "WR", "D0", "T", "classify", "immediate_wins", "threats_against",
                        "saving_moves_of", "one_ply_threats", "_flip", "copy", "m1_choice"}, names


REGRESSION = {"OWN_WIN_AND_THREAT_264": OWN_WIN_AND_THREAT_264, "SINGLE_260": SINGLE_260,
              "MULTI_269": MULTI_269, "UNDEFENDABLE_256": UNDEFENDABLE_256, "QUIET_40": QUIET_40,
              "OWN_WIN_279": OWN_WIN_279, "THREAT_279": THREAT_279, "DRAW24": DRAW24, "TWO": TWO,
              "E2": BY_ID["E2"]["moves"]}


@pytest.mark.parametrize("name", REGRESSION)
def test_the_OPTIMIZED_detector_agrees_with_the_EXHAUSTIVE_scan(name):
    r = TC.cross_check(T.replay(REGRESSION[name]))
    assert r["agree"] and r["classify"] == r["exhaustive"] and r["new_opponent_wins"] == [], r


@pytest.mark.parametrize("name,examined", [("OWN_WIN_AND_THREAT_264", 0), ("OWN_WIN_279", 0),
                                           ("THREAT_279", 0), ("DRAW24", 0)])
def test_NO_reply_is_examined_after_a_win_or_protocol_termination(name, examined):
    assert TC.exhaustive_scan(T.replay(REGRESSION[name]))[1]["replies_examined"] == examined


def test_EVERY_reply_after_EVERY_move_is_examined_otherwise():
    s = T.replay(MULTI_269)
    assert TC.exhaustive_scan(s)[1]["replies_examined"] == \
        sum(len(s.apply_move(m).legal_moves()) for m in s.legal_moves())


@pytest.mark.parametrize("defect", ["drop a saving move", "drop a threat"])
def test_the_CROSS_CHECK_catches_a_wrong_detector(defect, monkeypatch):
    real_D, real_T = WR.saving_moves_of, CC.threats_against
    if defect == "drop a saving move":
        monkeypatch.setattr(WR, "saving_moves_of", lambda s, t: real_D(s, t)[1:])
    else:
        monkeypatch.setattr(CC, "threats_against", lambda s: real_T(s)[1:])
    assert not TC.cross_check(T.replay(MULTI_269))["agree"]


def test_a_STALE_capture_is_refused():
    class Stale:                                  # refreshes, but with another ply's search
        colour, capture, last_capture = "red", True, None

        def __call__(self, state):
            self.last_capture = {"ply": 3, "move": (5, 5), "counts": {}, "root": None}
            return (5, 5)
    with pytest.raises(TC.CandidateRefused, match="captured search is not the one"):
        TC.TacticalCandidate(Stale(), enabled=True)(T.replay(QUIET_40))


def test_the_CHECKER_flags_a_reply_outside_the_threat_set_and_then_never_agrees(monkeypatch):
    real = TwixtState
    monkeypatch.setattr(TC, "TwixtState", lambda **k: real(
        **{**k, "to_move": "black" if k["to_move"] == "red" else "red"}))   # a WRONG flip
    s = T.replay(SINGLE_260)
    assert TC.exhaustive_scan(s)[1]["new_opponent_wins"], "replies won outside the threat set"
    monkeypatch.setattr(T, "classify", lambda st, semantics: TC.exhaustive_scan(st)[0])
    r = TC.cross_check(s)
    assert r["classify"] == r["exhaustive"] and not r["agree"], \
        "equal class dicts must not agree while the reduction premise fails"


# ─────────────────────────────── review repairs: capture freshness, protocol termination ────

def test_an_UNCHANGED_capture_is_refused():
    s = T.replay(QUIET_40)
    m = other(s)

    class Lazy:                                   # never refreshes its capture
        colour, capture = s.to_move, True

        def __init__(self):
            self.last_capture = {"ply": s.ply, "move": m, "counts": {m: 1}, "root": tree_root(s)}

        def __call__(self, state):
            return m
    with pytest.raises(TC.CandidateRefused, match="did not refresh its capture"):
        TC.TacticalCandidate(Lazy(), enabled=True)(s)


def test_a_capture_of_ANOTHER_BOARD_with_matching_ply_and_move_is_refused():
    s, elsewhere = T.replay(QUIET_40), T.replay(SINGLE_260[:40])
    assert (elsewhere.ply, elsewhere.to_move) == (s.ply, s.to_move)
    assert T.state_digest(elsewhere) != T.state_digest(s)
    m = next(x for x in sorted(s.legal_moves()) if elsewhere.is_valid_placement(*x))

    class Fresh:                                  # refreshes -- but searched another board
        colour, capture, last_capture = s.to_move, True, None

        def __call__(self, state):
            self.last_capture = {"ply": state.ply, "move": m, "counts": {m: 1},
                                 "root": tree_root(elsewhere)}
            return m
    with pytest.raises(TC.CandidateRefused, match="captured search is not the one"):
        TC.TacticalCandidate(Fresh(), enabled=True)(s)


def test_a_PROTOCOL_TERMINAL_position_is_refused_BEFORE_any_search():
    s = T.replay(OWN_WIN_279)
    at_cap = T.replay(OWN_WIN_279 + [other(s, [(6, 19)])])     # ply 280, no winner
    won = T.replay(list(TWO) + [(12, 2)])                       # red has won at ply 25
    assert (at_cap.ply, at_cap.winner(), won.winner()) == (280, None, "red")
    for st in (at_cap, won):
        for enabled in (False, True):
            agent = Dummy()
            agent.colour = st.to_move
            with pytest.raises(TC.CandidateRefused, match="protocol-terminal"):
                TC.TacticalCandidate(agent, enabled=enabled)(st)


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "enabled"])
def test_the_LAST_allowed_decision_at_ply_279_is_made_and_a_win_there_counts(enabled):
    move = TC.TacticalCandidate(builder()("black", True), enabled=enabled)(T.replay(OWN_WIN_279))
    if enabled:
        assert T.replay(OWN_WIN_279 + [move]).winner() == "black", "the 280th move wins"
    else:
        assert T.replay(OWN_WIN_279 + [move]).ply == 280
