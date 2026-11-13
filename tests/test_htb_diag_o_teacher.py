"""Diagnostic O, stage B: model-free tests with a fake teacher (no TensorFlow, no clone, gate closed)."""
import json
import pathlib

import numpy as np
import pytest

from scripts.GPU.alphazero import htb_diag_o_teacher as T
from scripts.GPU.alphazero.game.twixt_state import TwixtState


class FakeTB:
    def __init__(self):
        self.moves = []


class FakeTeacher:
    """Value = a fixed function of the position that is NOT reflection-symmetric, so mirror-averaging is visible.
    Policy logits likewise. Values are from the side to move, as the real teacher's."""

    def __init__(self, diverge_at=None):
        self.calls = 0
        self.diverge_at = diverge_at

    def new_game(self):
        return FakeTB()

    def play(self, tb, rc):
        tb.moves.append(tuple(rc))

    def divergences(self, our, tb):
        if self.diverge_at is not None and len(tb.moves) == self.diverge_at:
            return ["fake divergence"]
        return []

    def evaluate(self, tb, colour, legal):
        self.calls += 1
        s = sum((r + 1) * (c + 3) for r, c in tb.moves)
        v = float(np.tanh(0.001 * s - 0.2 * len(tb.moves) % 1)) * (1 if colour == "black" else -0.5)
        z = np.array([0.01 * c + 0.002 * r for r, c in legal])
        p = np.exp(z - z.max())
        return p / p.sum(), v


PREFIX = [[18, 19], [9, 10], [12, 5]]  # red, black, red: Black to move


def test_gate_is_closed_in_the_committed_state():
    assert T.gate_open() is False


def test_main_refuses_while_the_gate_is_closed(tmp_path):
    inp = tmp_path / "in.json"
    inp.write_text("[]")
    assert T.main(["run", "--clone-root", "x", "--in", str(inp), "--out", str(tmp_path / "out.json")]) == 4
    assert not (tmp_path / "out.json").exists()


def test_mirror_evaluate_averages_the_reflection_and_maps_policy_back():
    t = FakeTeacher()
    to_move, legal, prob, value = T.mirror_evaluate(t, PREFIX)
    _, legal1, p1, v1 = T.evaluate_final(t, PREFIX)
    _, rlegal, rp, rv = T.evaluate_final(t, [list(T.P.reflect_rc(m)) for m in PREFIX])
    assert to_move == "black"
    assert value == pytest.approx(0.5 * (v1 + rv))
    at = {tuple(m): i for i, m in enumerate(rlegal)}
    for i, m in enumerate(legal):
        assert prob[i] == pytest.approx(0.5 * (p1[i] + rp[at[T.P.reflect_rc(m)]]))
    assert prob.sum() == pytest.approx(1.0)


def test_evaluate_position_reports_black_side_values_and_candidate_ranks():
    t = FakeTeacher()
    cands = ["11,13", "20,19"]
    rec = T.evaluate_position(t, PREFIX, cands)
    _, _, _, v_root = T.mirror_evaluate(t, PREFIX)
    assert rec["value_black"] == pytest.approx(v_root)  # Black to move: unchanged
    for c in cands:
        r, col = map(int, c.split(","))
        tm, _, _, v_after = T.mirror_evaluate(t, PREFIX + [[r, col]])
        assert tm == "red"
        assert rec["after_value_black"][c] == pytest.approx(-v_after)  # Red to move there: negated
        assert rec["rank"][c] >= 1 and 0 < rec["prob"][c] < 1
    assert rec["top_move"] not in ("",)


def test_an_illegal_candidate_or_a_divergence_stops():
    with pytest.raises(T.Stop):
        T.evaluate_position(FakeTeacher(), PREFIX, ["18,19"])  # occupied
    with pytest.raises(T.Stop):
        T.evaluate_position(FakeTeacher(diverge_at=3), PREFIX, ["11,13"])
    with pytest.raises(T.Stop):
        T.evaluate_position(FakeTeacher(), PREFIX[:2], ["11,13"])  # Red to move, not Black
