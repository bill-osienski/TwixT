"""H5 weakness review (EXPLORATORY) -- synthetic controls, card §4 and §7.

Positions are built on small boards (active_size 6) from explicit pegs or alternating
legal moves; no H5 game is read except by the one integrity baseline, which loads the
committed records read-only. Messages are fixed.
"""
import json

import pytest

from scripts.GPU.alphazero import d0_postmortem as D0
from scripts.GPU.alphazero import h5_weakness_review as W
from scripts.GPU.alphazero.game.twixt_state import TwixtState

N = 6
#: Red to move: red (0,2)-(2,3)-(4,2) wins at (5,4).
WIN_PEGS = [((0, 2), "red"), ((2, 3), "red"), ((4, 2), "red")]
#: Red to move: black (2,0)-(3,2) . (2,5) wins at (4,4); red's (1,3) holds the other hole.
ONE_THREAT = [((2, 0), "black"), ((3, 2), "black"), ((2, 5), "black"), ((1, 3), "red")]
#: Red to move: black wins at (1,3) OR (4,4); red has no peg to bridge with.
TWO_THREATS = [((2, 0), "black"), ((3, 2), "black"), ((2, 5), "black")]
#: Alternating legal moves reaching ONE_THREAT with red to move, then red (0,1).
TO_ONE_THREAT = [(0, 4), (2, 0), (1, 3), (3, 2), (5, 1), (2, 5)]
#: Found by random playout on 6x6: black to move with W = [(2,0)] and red's T = [(1,1)].
RACE = [(2, 3), (4, 5), (3, 1), (4, 3), (4, 2), (4, 4), (4, 1), (1, 2), (0, 3), (2, 4),
        (2, 2), (3, 4), (5, 2)]


def build(pegs, to_move):
    s = TwixtState(active_size=N, to_move="red")
    for pos, colour in pegs:
        s = s.copy()
        s.to_move = colour
        s = s.apply_move(pos)
    s = s.copy()
    s.to_move = to_move
    return s


def wt(state):
    return D0.one_ply_threats(state), D0.one_ply_threats(D0._flip(state))


def actors(n):
    return ["incumbent" if i % 2 == 0 else "t1j" for i in range(n)]


def brute_saving(state):
    """The FULL scan the reduction must equal: every move after which the opponent
    has no immediate win, found by scanning ALL opponent moves."""
    return sorted(tuple(m) for m in state.legal_moves()
                  if not D0.one_ply_threats(state.apply_move(m)))


# ─────────────────────────────── fixtures hold ───────────────────────────────

def test_the_synthetic_positions_are_what_they_claim():
    assert wt(build(WIN_PEGS, "red")) == ([(5, 4)], []), "WIN_PEGS is not a single red win"
    assert wt(build(ONE_THREAT, "red")) == ([], [(4, 4)]), "ONE_THREAT is not one threat"
    assert wt(build(TWO_THREATS, "red")) == ([], [(1, 3), (4, 4)]), "TWO_THREATS is not two"
    s = TwixtState(active_size=N, to_move="red")
    for m in RACE:
        s = s.apply_move(m)
    assert (s.to_move, wt(s)) == ("black", ([(2, 0)], [(1, 1)])), "RACE is not a race"


# ─────────────────────────────── the three labels ───────────────────────────────

def test_MISSED_WIN_with_the_winning_move_as_witness():
    s = build(WIN_PEGS, "red")
    Wm, T = wt(s)
    label, _ = W.label_ply(Wm, T, (1, 1), [], lambda: pytest.fail("no defence after a win"))
    assert label == "MISSED_WIN", "a non-winning move with a win available is not MISSED_WIN"
    assert W.label_ply(Wm, T, (5, 4), None, lambda: pytest.fail("x")) == (None, None), \
        "playing the win is labelled"


def test_MISSED_DEFENCE_needs_a_saving_move_and_lists_it():
    s = build(ONE_THREAT, "red")
    Wm, T = wt(s)
    left = D0.one_ply_threats(s.apply_move((0, 1)))
    label, saving = W.label_ply(Wm, T, (0, 1), left, lambda: W.saving_moves_of(s, T))
    assert label == "MISSED_DEFENCE", "an unanswered single threat is not MISSED_DEFENCE"
    assert (4, 4) in saving and (0, 1) not in saving, "the saving witness is wrong"
    assert W.label_ply(Wm, T, (4, 4), D0.one_ply_threats(s.apply_move((4, 4))),
                       lambda: pytest.fail("x"))[0] == "DEFENDED", "the block is not DEFENDED"


def test_UNDEFENDABLE_when_no_move_removes_every_threat():
    s = build(TWO_THREATS, "red")
    Wm, T = wt(s)
    left = D0.one_ply_threats(s.apply_move((1, 3)))
    assert W.label_ply(Wm, T, (1, 3), left, lambda: W.saving_moves_of(s, T)) == \
        ("UNDEFENDABLE", []), "a double threat is not UNDEFENDABLE"
    assert brute_saving(s) == [], "the full scan finds a defence the label denies"


# ─────────────────────────────── terminal precedence ───────────────────────────────

def test_TERMINAL_PRECEDENCE_a_win_is_saving_and_nothing_after_it_is_examined():
    calls = []
    assert W.label_ply([(9, 9)], [(1, 1), (2, 2)], (9, 9), None,
                       lambda: calls.append(1)) == (None, None), "playing the win is labelled"
    assert W.label_ply([(9, 9)], [(1, 1)], (3, 3), [(1, 1)], lambda: calls.append(1)) == \
        ("MISSED_WIN", None), "a missed win under threat is not MISSED_WIN"
    assert W.label_ply([(9, 9)], [(1, 1)], (3, 3), [(1, 1)], lambda: []) == \
        ("MISSED_WIN", None), "no pure defence plus an available win became UNDEFENDABLE"
    assert calls == [], "a defence search ran although a win was available"


def test_the_REAL_RACE_trajectory_labels_the_win_and_examines_no_reply_after_it():
    rows = W.scan_trajectory(RACE + [(2, 0)], actors(len(RACE) + 1), "win", active_size=N)
    last = rows[-1]
    assert (last["W"], last["T"]) == ([[2, 0]], [[1, 1]]), "the race position is not reached"
    assert last["label"] is None, "playing the win under threat is labelled"
    assert last["threats_left"] is None and last["saving_moves"] is None, \
        "a reply after the win was examined"
    missed = W.scan_trajectory(RACE + [(1, 0)], actors(len(RACE) + 1), "cap", active_size=N)
    assert missed[-1]["label"] == "MISSED_WIN", "the missed win in the race is not MISSED_WIN"


# ─────────────────────────────── replies: available vs capped ───────────────────────────────

def test_a_CAPPED_final_miss_is_marked_reply_UNAVAILABLE():
    seq = TO_ONE_THREAT + [(0, 1)]
    rows = W.scan_trajectory(seq, actors(len(seq)), "cap", active_size=N)
    last = rows[-1]
    assert last["label"] == "MISSED_DEFENCE" and last["reply_available"] is False, \
        "a miss at a capped final ply is not marked reply-unavailable"
    assert last["threats_left"] == [[4, 4]] and not last["opponent_won_next_ply"], \
        "the capped miss records the wrong reply"


def test_an_ANSWERED_reply_is_available_and_the_opponent_won_next_ply():
    seq = TO_ONE_THREAT + [(0, 1), (4, 4)]
    rows = W.scan_trajectory(seq, actors(len(seq)), "win", active_size=N)
    miss, win = rows[-2], rows[-1]
    assert (miss["label"], miss["reply_available"], miss["opponent_won_next_ply"]) == \
        ("MISSED_DEFENCE", True, True), "the punished miss is recorded wrongly"
    assert win["label"] is None and win["threats_left"] is None, "the winning ply is examined"


# ─────────────────────────────── the exact reduction ───────────────────────────────

def test_the_REDUCTION_equals_the_full_scan_on_every_threatened_position_of_random_games():
    import random
    rng, checked = random.Random(11), 0
    for _ in range(60):
        s = TwixtState(active_size=N, to_move="red")
        while s.winner() is None and s.legal_moves():
            Wm, T = wt(s)
            if not Wm and T:
                assert W.saving_moves_of(s, T) == brute_saving(s), \
                    "the reduced defence check differs from the full scan"
                checked += 1
            s = s.apply_move(rng.choice(s.legal_moves()))
    assert checked >= 20, "too few threatened positions to compare"


def test_a_LEFT_threat_outside_T_stops_the_scan(monkeypatch):
    real = D0.one_ply_threats

    def lying(state):
        out = real(state)
        return out + [(5, 3)] if state.ply == 3 and state.to_move == "black" else out
    monkeypatch.setattr(D0, "one_ply_threats", lying)
    with pytest.raises(W.ReviewRefused, match="exact reduction does not hold"):
        W.scan_trajectory(TO_ONE_THREAT, actors(len(TO_ONE_THREAT)), "cap", active_size=N)


# ─────────────────────────────── multiplicity and rates ───────────────────────────────

def _game(tid, arm, moves, outcome):
    return {"task_id": tid, "arm": arm, "moves": moves, "actors": actors(len(moves)),
            "movers": ["red" if i % 2 == 0 else "black" for i in range(len(moves))],
            "outcome": outcome, "winner": None, "terminal_reason": "win",
            "replay": f"r/{tid}.json"}


def test_MULTIPLICITY_identical_games_are_one_trajectory_with_their_frequency():
    a, b = [(0, 1), (1, 1)], [(0, 2), (1, 1)]
    games = [_game(f"g{i}", "A", a, "loss") for i in range(3)] + [_game("g9", "A", b, "win")]
    trajs = W.distinct_trajectories(games)
    assert sorted(t["frequency"] for t in trajs.values()) == [1, 3], "frequencies are wrong"
    tab = W.prefix_tables(trajs, depths=(1,))["A"]["1"]
    row = next(r for r in tab if r["prefix"] == "0,1")
    assert (row["occurrence"]["loss"], row["distinct"]["loss"]) == (3, 1), \
        "occurrence and distinct prefix counts are confused"
    scans = {d: [{"engine": "incumbent", "label": "MISSED_WIN" if t["outcome"] == "loss" else None},
                 {"engine": "t1j", "label": None}] for d, t in trajs.items()}
    cell = W.label_rates(trajs, scans)["A/incumbent"]
    assert cell["occurrence"]["MISSED_WIN"] == {"numerator": 3, "denominator": 4, "rate": 0.75}
    assert cell["distinct"]["MISSED_WIN"] == {"numerator": 1, "denominator": 2, "rate": 0.5}


def test_the_same_moves_with_another_outcome_are_REFUSED():
    games = [_game("g1", "A", [(0, 1)], "loss"), _game("g2", "A", [(0, 1)], "win")]
    with pytest.raises(W.ReviewRefused, match="different outcome"):
        W.distinct_trajectories(games)


def test_a_label_at_ONE_fixed_terminal_distance_is_flagged():
    scans = {"x": [{"label": "MISSED_WIN", "distance_to_terminal": 1}],
             "y": [{"label": "MISSED_WIN", "distance_to_terminal": 1},
                   {"label": "UNDEFENDABLE", "distance_to_terminal": 1},
                   {"label": "UNDEFENDABLE", "distance_to_terminal": 7}]}
    f = W.fixed_distance_flags(scans)
    assert f["MISSED_WIN"]["restates_the_ending"] and not f["UNDEFENDABLE"]["restates_the_ending"]


def test_a_cap_is_never_a_loss():
    assert W.outcome_of(None, "cap", "red") == "cap"
    assert W.outcome_of("black", "win", "red") == "loss" and W.outcome_of("red", "win", "red") == "win"


# ─────────────────────────────── integrity refusal ───────────────────────────────

@pytest.fixture(scope="module")
def loaded():
    """The committed records, loaded ONCE (read-only; ~20 s)."""
    return W.load_and_verify()


def test_INTEGRITY_baseline_the_committed_records_load_and_cross_check(loaded):
    games, inputs = loaded
    assert len(games) == 592 and set(inputs) == set(W.PINS), "the clean baseline does not load"


def test_INTEGRITY_a_pin_mismatch_refuses_before_anything_is_parsed(monkeypatch):
    rel = next(iter(W.PINS))
    monkeypatch.setitem(W.PINS, rel, "0" * 64)
    monkeypatch.setattr(W.R4, "load_games", lambda *a, **k: pytest.fail("parsed after a bad pin"))
    with pytest.raises(W.ReviewRefused, match="does not hash to its pin"):
        W.load_and_verify()


def test_INTEGRITY_games_that_do_not_reproduce_the_estimate_are_refused(loaded):
    est = json.loads((W.ROOT / W.ESTIMATE).read_text())
    games, _ = loaded
    flipped = [dict(g) for g in games]
    k = next(i for i, g in enumerate(flipped) if g["terminal_reason"] == "win")
    flipped[k]["winner"] = "red" if flipped[k]["winner"] == "black" else "black"
    with pytest.raises(W.ReviewRefused, match="do not reproduce the committed estimate"):
        W.cross_check(flipped, est)


def test_the_CLI_refuses_an_unlabelled_directory_and_records_an_integrity_refusal(tmp_path,
                                                                                  monkeypatch):
    assert W.main(["--out", str(tmp_path / "plain")]) == 4 and not (tmp_path / "plain").exists()

    def refuse():
        raise W.ReviewRefused("synthetic")
    monkeypatch.setattr(W, "load_and_verify", refuse)
    out = tmp_path / "x-EXPLORATORY"
    assert W.main(["--out", str(out)]) == 4
    assert sorted(p.name for p in out.iterdir()) == ["00_REFUSED.txt"], "a refusal analysed"
