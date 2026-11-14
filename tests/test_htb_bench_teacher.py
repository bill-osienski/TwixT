"""Student-teacher benchmark, the teacher process: model-free tests (card 2026-11-14, revision 1, §7).

No TensorFlow and no network: a fake evaluator stands in for the raw network. The lockstep tests use twixtbot's real
`Game` (pure Python and NumPy) from the pinned clone; they are skipped only if the clone is absent.
"""
import io
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from scripts.GPU.alphazero import htb_bench_teacher as T
from scripts.GPU.alphazero import htb_diag_o_teacher as DO
from scripts.GPU.alphazero import twixtbot_adapter as A
from scripts.GPU.alphazero.game.twixt_state import TwixtState

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLONE = pathlib.Path(os.path.expanduser("~/twixt-cache/htb-teacher/twixtbot-ui"))
G3_DRAW = ROOT / "docs/superpowers/evidence/htb-g3-match/games/pair_0057_game_1.json"

needs_clone = pytest.mark.skipif(not (CLONE / "src/backend/twixt.py").exists(), reason="pinned twixtbot clone absent")


def _twixt():
    if str(CLONE) not in sys.path:
        sys.path.insert(0, str(CLONE))
    from src.backend import twixt, point  # noqa: E402  (the clone's pure-Python game)
    return twixt, point.Point


class FakeTeacher:
    """new_game/play/evaluate like htb_pilot.TwixtbotTeacher; evaluate is a fixed, NOT reflection-symmetric function
    of the game's history, so mirror-averaging is visible. Values from the side to move."""

    def __init__(self, allow_scl=False):
        self.twixt, self.Point = _twixt()
        self.allow_scl = allow_scl
        self.calls = 0

    def new_game(self):
        return self.twixt.Game(allow_scl=self.allow_scl)

    def play(self, tb, rc):
        tb.play(self.Point(*A.rc_to_xy(*rc)))

    def divergences(self, our, tb):  # for htb_diag_o_teacher's replay (the comparison test)
        return T.bench_divergences(our, tb, self.twixt)

    def evaluate(self, tb, colour, legal):
        self.calls += 1
        s = sum((p.y + 1) * (p.x + 3) for p in tb.history)
        z = np.array([0.013 * c + 0.002 * r + 0.0001 * ((s + r * 7 + c) % 11) for r, c in legal])
        p = np.exp(z - z.max())
        return p / p.sum(), float(np.tanh(0.001 * s)) * (1 if colour == "black" else -0.7)


def test_gates_are_closed_in_the_committed_state():
    assert T.gate_open("smoke") is False
    assert T.gate_open("match") is False
    assert T.gate_open("other") is False


def test_serve_refuses_while_the_gate_is_closed(capsys):
    assert T.main(["serve", "--stage", "smoke", "--clone-root", "x"]) == 4
    assert "BENCH_SMOKE_AUTHORIZED is closed" in capsys.readouterr().err


def test_canonical_is_sorted_and_bridges_carry_their_owner():
    s = TwixtState()
    for m in [(10, 10), (5, 5), (12, 11), (5, 8)]:
        s = s.apply_move(m)
    c = T.canonical(s)
    assert c["pegs"] == {"red": [[10, 10], [12, 11]], "black": [[5, 5], [5, 8]]}
    assert c["bridges"] == {"red": [[[10, 10], [12, 11]]], "black": []}
    assert c["legal"] == sorted(c["legal"]) and c["ply"] == 4 and c["to_move"] == "red"
    assert c["winner"] is None and c["terminal"] is False
    assert T.canon_digest(s) == T.canon_digest(TwixtState().apply_move((10, 10)).apply_move((5, 5))
                                                .apply_move((12, 11)).apply_move((5, 8)))


@needs_clone
def test_regression_g3_draw_product_rule_holds_and_the_tb_flag_is_recorded():
    """G3 pair_0057_game_1: twixtbot flags a draw after ply 73; the product plays to ply 572 (card §6, §7)."""
    moves = json.loads(G3_DRAW.read_text())["moves"]
    assert len(moves) == 572
    bench = T.BenchTeacher(FakeTeacher(), _twixt()[0])
    rep = bench.start(moves[:4])
    assert rep["ok"] and rep["tb_draw"] is False
    flagged_at = None
    for k, m in enumerate(moves[4:], start=4):
        rep = bench.apply(m)
        assert rep["ply"] == k + 1
        if rep["tb_draw"] and flagged_at is None:
            flagged_at = rep["ply"]
            # the adapter's unfiltered check WOULD report it: the one line the benchmark leaves out
            unfiltered = A.state_divergences(bench.our, bench.tb, bench.twixt, ply_cap=None)
            assert unfiltered and all(d.startswith("terminal") for d in unfiltered)
    assert flagged_at == 73 and rep["tb_draw_ply"] == 73
    assert T.natural_terminal(bench.our) and bench.our.winner() is None and bench.our.ply == 572
    with pytest.raises(T.Stop):
        bench.apply((0, 1))  # nothing after the product's end


@needs_clone
def test_a_real_bridge_mismatch_stops():
    """A twixtbot lane that allows own-link crossings forms a bridge ours forbids: the run must stop."""
    bench = T.BenchTeacher(FakeTeacher(allow_scl=True), _twixt()[0])
    bench.start([(10, 10), (5, 5), (12, 11), (5, 8)])
    bench.apply((10, 11))
    bench.apply((5, 12))
    with pytest.raises(T.Stop) as e:
        bench.apply((12, 10))  # (10,11)-(12,10) crosses red's own (10,10)-(12,11)
    assert e.value.step == "lockstep" and "bridges[red]" in e.value.detail


@needs_clone
def test_a_peg_mismatch_and_an_illegal_move_stop():
    bench = T.BenchTeacher(FakeTeacher(), _twixt()[0])
    bench.start([(10, 10), (5, 5), (12, 11), (5, 8)])
    bench.teacher.play(bench.tb, (15, 15))  # tamper with one lane only
    with pytest.raises(T.Stop):
        bench.apply((14, 14))
    bench2 = T.BenchTeacher(FakeTeacher(), _twixt()[0])
    bench2.start([(10, 10), (5, 5), (12, 11), (5, 8)])
    with pytest.raises(T.Stop) as e:
        bench2.apply((10, 10))
    assert e.value.step == "illegal"


@needs_clone
def test_mirror_policy_is_the_label_paths_averaging_and_move_is_its_argmax():
    prefix = [(10, 10), (5, 5), (12, 11), (5, 8), (14, 6)]
    t = FakeTeacher()
    bench = T.BenchTeacher(t, _twixt()[0])
    bench.start(prefix)
    legal, prob, value = bench.mirror_policy()
    to_move, legal_do, prob_do, value_do = DO.mirror_evaluate(FakeTeacher(), [list(m) for m in prefix])
    assert to_move == "black" and legal == [tuple(m) for m in legal_do]
    np.testing.assert_allclose(prob, prob_do, rtol=0, atol=1e-15)
    assert value == pytest.approx(value_do, abs=1e-15)
    rep = bench.move()
    best = max(range(len(legal)), key=lambda i: (prob[i], -legal[i][0], -legal[i][1]))
    assert rep["move"] == list(legal[best]) and rep["n_legal"] == len(legal)
    assert bench.our.ply == 5  # move() does not apply


@needs_clone
def test_move_ties_go_to_the_lower_row_col():
    class Flat(FakeTeacher):
        def evaluate(self, tb, colour, legal):
            return np.full(len(legal), 1.0 / len(legal)), 0.0
    bench = T.BenchTeacher(Flat(), _twixt()[0])
    bench.start([(10, 10), (5, 5), (12, 11), (5, 8)])
    assert bench.move()["move"] == [min(bench.our.legal_moves())[0], min(bench.our.legal_moves())[1]]


@needs_clone
def test_serve_loop_replies_per_line_and_exits_by_the_table():
    bench = T.BenchTeacher(FakeTeacher(), _twixt()[0])
    req = [{"cmd": "start", "opening": [[10, 10], [5, 5], [12, 11], [5, 8]]}, {"cmd": "move"},
           {"cmd": "apply", "move": [14, 14]}, {"cmd": "close"}]
    out = io.StringIO()
    assert T.serve(bench, io.StringIO("\n".join(json.dumps(r) for r in req) + "\n"), out) == 0
    reps = [json.loads(x) for x in out.getvalue().splitlines()]
    assert [r["ok"] for r in reps] == [True] * 4 and reps[0]["ply"] == 4 and reps[2]["ply"] == 5
    out = io.StringIO()
    bad = [{"cmd": "start", "opening": [[10, 10]]}, {"cmd": "apply", "move": [10, 10]}]
    assert T.serve(T.BenchTeacher(FakeTeacher(), _twixt()[0]), io.StringIO("\n".join(json.dumps(r) for r in bad)), out) == 3
    assert json.loads(out.getvalue().splitlines()[-1])["stop"] == "illegal"
    out = io.StringIO()
    assert T.serve(bench, io.StringIO('{"cmd": "apply"}\n'), out) == 2
    assert T.serve(bench, io.StringIO(""), io.StringIO()) == 3  # stdin closed without close


def test_canon_cli_prints_one_digest_per_ply(tmp_path):
    f = tmp_path / "m.json"
    f.write_text(json.dumps([[10, 10], [5, 5]]))
    r = subprocess.run([sys.executable, "-B", "-m", "scripts.GPU.alphazero.htb_bench_teacher", "canon", str(f)],
                       cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."})
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert len(d) == 3 and d[0] == T.canon_digest(TwixtState())


@needs_clone
def test_the_early_draw_flag_is_observational_in_both_lanes():
    """A flag set in one lane only never stops play; each lane's first flagged ply is recorded separately (§6)."""
    bench = T.BenchTeacher(FakeTeacher(), _twixt()[0])
    bench.start([(10, 10), (5, 5), (12, 11), (5, 8)])
    bench.rtb.result = "DRAW"  # the reflected lane alone
    rep = bench.apply((14, 14))
    assert rep["ok"] and rep["tb_draw"] is False and rep["tb_draw_ply"] is None
    assert rep["tb_draw_ply_reflected"] == 5
    bench.tb.result = "DRAW"
    rep = bench.apply((7, 7))
    assert rep["tb_draw_ply"] == 6 and rep["tb_draw_ply_reflected"] == 5
    assert bench.apply((16, 14))["ok"]  # play continues


def test_the_teacher_runtime_is_checked_before_any_model_loads(monkeypatch, capsys):
    assert T.env_mismatches(dict(T.TEACHER_ENV)) == []
    assert T.env_mismatches({**T.TEACHER_ENV, "tensorflow": "2.20.0"}) == ["tensorflow: 2.20.0 (card: 2.21.0)"]
    monkeypatch.setattr(T, "gate_open", lambda stage, root=T.ROOT: True)
    monkeypatch.setattr(T, "runtime_versions", lambda: {**T.TEACHER_ENV, "numpy": "2.4.0"})

    def must_not_load(*a, **k):
        raise AssertionError("the teacher model was loaded before the runtime check")
    monkeypatch.setattr(T.P, "_load_teacher_real", must_not_load)
    assert T.main(["serve", "--stage", "smoke", "--clone-root", "x"]) == 4
    out = capsys.readouterr()
    rep = json.loads(out.out.splitlines()[0])
    assert rep == {"ok": False, "refused": "runtime", "detail": ["numpy: 2.4.0 (card: 2.5.2)"],
                   "env": {**T.TEACHER_ENV, "numpy": "2.4.0"}}
