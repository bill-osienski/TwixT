"""External-engine learning, stage 1 prototype -- focused tests, synthetic inputs only.

No twixtbot runtime, no checkpoint, no registered seed. The twixtbot side is a FAKE Game built on our own
geometry (`_proper_intersect_knight`); its broken variants are the negative controls the fixtures must
reject. Tests that need MLX skip where it is absent (the homebrew regression interpreter); run this file
under `.venv` for the full set.
"""
import json
import math
import os
import pathlib
import subprocess
import sys
import time
from collections import deque, namedtuple
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.GPU.alphazero import external_learning as EL
from scripts.GPU.alphazero import external_learning_authorization as GATE
from scripts.GPU.alphazero import twixtbot_adapter as A
from scripts.GPU.alphazero.game.twixt_state import TwixtState, _proper_intersect_knight as X

Point = namedtuple("Point", "x y")
KN = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]      # (dx, dy)
SYNTHETIC = 1_000_000                                    # never a 2026-band seed


class FakeGame:
    """twixtbot's Game surface as the adapter reads it: WHITE (1) = red moves first, pegs[colour][x, y],
    link planes owner*8 + direction, describe_link, is_winning, history, result."""
    own_cross_ok = False
    any_cross_ok = False
    drop_play_at = None

    def __init__(self, allow_scl=False):
        self.allow_scl, self.turn, self.history, self.result = allow_scl, 1, [], None
        self.pegs = {0: np.zeros((24, 24), bool), 1: np.zeros((24, 24), bool)}
        self.links = [np.zeros((24, 24), bool) for _ in range(16)]
        self._links = []

    def play(self, p):
        if self.drop_play_at is not None and len(self.history) == self.drop_play_at:
            self.history.append(p)
            self.turn = 1 - self.turn
            return
        x, y, o = p.x, p.y, self.turn
        self.pegs[o][x, y] = True
        for k, (dx, dy) in enumerate(KN):
            q = (x + dx, y + dy)
            if 0 <= q[0] < 24 and 0 <= q[1] < 24 and self.pegs[o][q] and not self._blocked(o, (x, y), q):
                self._links.append((o, (x, y), q))
                self.links[o * 8 + k][x, y] = True
        self.history.append(p)
        self.turn = 1 - o

    def _blocked(self, o, a, b):
        for o2, c, d in self._links:
            if self.any_cross_ok or (o2 == o and (self.allow_scl or self.own_cross_ok)):
                continue
            if {a, b} & {c, d}:
                continue
            if X(a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]):
                return True
        return False

    @staticmethod
    def describe_link(plane, x, y):
        o, k = divmod(plane, 8)
        dx, dy = KN[k]
        return SimpleNamespace(p1=Point(x, y), p2=Point(x + dx, y + dy), owner=o)

    def is_winning(self, colour):
        adj = {}
        for o, a, b in self._links:
            if o == colour:
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
        axis = 1 if colour == 1 else 0                   # red joins y 0..23, black joins x 0..23
        start = [n for n in adj if n[axis] == 0]
        seen, todo = set(start), deque(start)
        while todo:
            n = todo.popleft()
            if n[axis] == 23:
                return True
            for m in adj[n]:
                if m not in seen:
                    seen.add(m)
                    todo.append(m)
        return False


def fake_twixt(**flags):
    cls = type("G", (FakeGame,), flags)
    return SimpleNamespace(Game=cls)


def rt(**flags):
    return {"twixt": fake_twixt(**flags), "Point": Point, "TwixtState": TwixtState}


# ─────────────────────────────── fixtures, corruption, rule flag ───────────────────────────────

def test_the_twelve_fixtures_pass_against_an_engine_with_our_rules():
    assert len(EL.FIXTURES) == 12 and len({f["name"] for f in EL.FIXTURES}) == 12
    for fx in EL.FIXTURES:
        EL.check_fixture(fx, fake_twixt(), Point, TwixtState)


@pytest.mark.parametrize("name", ["F07_own_cross_red", "F08_own_cross_black"])
def test_own_crossing_fixtures_reject_an_engine_that_allows_own_crossings(name):
    fx = next(f for f in EL.FIXTURES if f["name"] == name)
    with pytest.raises(EL.Stage1Stop, match="fixtures"):
        EL.check_fixture(fx, fake_twixt(own_cross_ok=True), Point, TwixtState)


def test_the_opponent_crossing_fixture_rejects_an_engine_that_ignores_crossings():
    fx = next(f for f in EL.FIXTURES if f["name"] == "F06_opponent_cross")
    EL.check_fixture(fx, fake_twixt(own_cross_ok=True), Point, TwixtState)      # own rule irrelevant here
    with pytest.raises(EL.Stage1Stop, match="fixtures"):
        EL.check_fixture(fx, fake_twixt(any_cross_ok=True), Point, TwixtState)


def test_a_wrong_expectation_is_caught_even_when_both_engines_agree():
    fx = dict(EL.FIXTURES[1], red=[((10, 10), (12, 11))])
    with pytest.raises(EL.Stage1Stop, match="red bridges"):
        EL.check_fixture(fx, fake_twixt(), Point, TwixtState)


def test_cap_fixtures_sit_on_both_sides_of_the_boundary():
    reached, below = (next(f for f in EL.FIXTURES if f["name"] == n) for n in ("F11_cap_reached", "F12_cap_one_below"))
    assert reached["moves"] == below["moves"] and below["cap"] == reached["cap"] + 1 == len(reached["moves"]) + 1
    with pytest.raises(EL.Stage1Stop, match="capped terminality"):
        EL.check_fixture(dict(reached, terminal=False), fake_twixt(), Point, TwixtState)


def test_every_corruption_is_detected():
    assert EL.corruption_controls(fake_twixt(), Point, TwixtState) == ["peg", "bridge", "side", "ply"]


def test_the_rule_flag_must_survive_every_copy():
    assert EL.rule_flag_copies(fake_twixt(), Point) == ["copy.copy", "copy.deepcopy"]

    class Drops(FakeGame):
        def clone(self):
            g = FakeGame(allow_scl=True)
            return g
    with pytest.raises(EL.Stage1Stop, match="Game.clone does not preserve"):
        EL.rule_flag_copies(SimpleNamespace(Game=Drops), Point)

    class RaisingProperty(FakeGame):
        @property
        def rating(self):
            raise RuntimeError("never read: not a copy method")
    assert EL.rule_flag_copies(SimpleNamespace(Game=RaisingProperty), Point) == ["copy.copy", "copy.deepcopy"]

    class NeedsArgs(FakeGame):
        def copy_from(self, other):
            return other
    with pytest.raises(EL.Stage1Stop, match="needs arguments"):
        EL.rule_flag_copies(SimpleNamespace(Game=NeedsArgs), Point)


def test_query_fixtures_are_nonterminal_and_distinct():
    by = {f["name"]: f for f in EL.FIXTURES}
    for name in EL.QUERY_FIXTURES:
        st = TwixtState.from_moves(by[name]["moves"])
        assert st.winner() is None and st.legal_moves()
    q = [{"fixture": n, "input_sha256": "x", "move": [1, 1]} for n in EL.QUERY_FIXTURES]
    assert EL.compare_queries(q, [dict(x) for x in q]) == q
    other = [dict(x) for x in q]
    other[2]["move"] = [1, 2]
    with pytest.raises(EL.Stage1Stop, match="disagree"):
        EL.compare_queries(q, other)
    with pytest.raises(EL.Stage1Stop):
        EL.compare_queries(q[:2], q[:2])


# ─────────────────────────────── plan and openings ───────────────────────────────

def test_the_slot_layout_is_exactly_the_block():
    block = (SYNTHETIC, SYNTHETIC + 18)
    lay = EL.slot_layout(block)
    seeds = [o["opening_seed"] for o in lay["openings"]] + [g["seed"] for g in lay["games"]]
    assert sorted(seeds) == list(range(*block)) and len(lay["games"]) == 12
    assert [o["split"] for o in lay["openings"]] == ["train"] * 4 + ["heldout"] * 2
    for j in range(6):
        arms = [g for g in lay["games"] if g["family"] == j]
        assert [g["teacher_colour"] for g in arms] == ["red", "black"]
        assert {g["split"] for g in arms} == {EL.split_of(j)}
    with pytest.raises(EL.Stage1Stop, match="18 slots"):
        EL.slot_layout((SYNTHETIC, SYNTHETIC + 24))


def test_check_plan_refuses_an_unregistered_block_and_any_tampering(monkeypatch):
    from scripts.GPU.alphazero import e4_screen_reference as REG

    plan = EL.build_plan((SYNTHETIC, SYNTHETIC + 18), "a" * 64)
    with pytest.raises(EL.Stage1Stop, match="not ACCOUNTED"):
        EL.check_plan(plan)                                          # the real registry
    monkeypatch.setattr(REG, "seed_is_accounted", lambda s: True)
    monkeypatch.setattr(REG, "seed_is_unavailable", lambda s: False)
    assert EL.check_plan(json.loads(json.dumps(plan))) == plan
    bad = json.loads(json.dumps(plan))
    bad["games"][3]["seed"] += 1
    with pytest.raises(EL.Stage1Stop, match="frozen slot layout"):
        EL.check_plan(bad)
    with pytest.raises(EL.Stage1Stop, match="survey record"):
        EL.check_plan({**plan, "survey_record_sha256": None})
    assert "block" not in plan and EL.plan_block(plan) == (SYNTHETIC, SYNTHETIC + 18)
    for wrong in ({**plan, "n_slots": 19}, {**plan, "n_slots": 18.0}, {**plan, "base": float(SYNTHETIC)},
                  {**plan, "block": [SYNTHETIC, SYNTHETIC + 18]}):
        with pytest.raises(EL.Stage1Stop):
            EL.check_plan(wrong)
    monkeypatch.setattr(REG, "seed_is_unavailable", lambda s: s == SYNTHETIC + 7)
    with pytest.raises(EL.Stage1Stop, match="spent"):
        EL.check_plan(plan)


def test_openings_are_seeded_four_legal_plies():
    a, b = EL.generate_opening(SYNTHETIC + 1), EL.generate_opening(SYNTHETIC + 2)
    assert a == EL.generate_opening(SYNTHETIC + 1) and a != b and len(a) == 4
    TwixtState.from_moves(a)


# ─────────────────────────────── games, rows, export ───────────────────────────────

class FakePlayer:
    def pick_move(self, game, window=None):
        r, c = sorted(A.tb_legal_moves(game))[-1]
        return {"moves": [Point(*A.rc_to_xy(r, c))], "Y": [1], "proven": False}


class FakeAgent:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, state):
        return state.legal_moves()[0]


GAME = {"game_id": "f0-teacher_red", "family": 0, "split": "train", "arm": "teacher_red",
        "teacher_colour": "red", "seed": SYNTHETIC + 6}


def _play(game=GAME, opening=((12, 12), (11, 10), (13, 9), (10, 13)), seed=None, **flags):
    return EL.play_teacher_game(game=game, opening=[list(m) for m in opening], rt=rt(**flags), player=FakePlayer(),
                                reference_agent=FakeAgent(game["seed"] if seed is None else seed))


@pytest.fixture(scope="module")
def played():
    games = [_play(), _play(game=dict(GAME, game_id="f4-teacher_black", family=4, split="heldout",
                                      arm="teacher_black", teacher_colour="black", seed=SYNTHETIC + 15))]
    return games, EL.rows_from_games(games, config_sha256="c" * 64, runtime_sha256="r" * 64)


def test_a_game_records_one_row_per_teacher_decision_and_nothing_else(played):
    games, rows = played
    g = games[0]
    assert not g["aborted"] and g["result"] in ("teacher", "incumbent", "draw", "draw_ply_cap")
    teacher_plies = [m["ply"] for m in g["moves"] if m["mover"] == "teacher"]
    assert [r["ply"] for r in g["rows"]] == teacher_plies and teacher_plies[0] == 4
    assert all(m["mover"] in ("opening", "teacher", "incumbent") for m in g["moves"])
    for r in rows:
        EL.verify_row(r)


def test_state_is_compared_after_every_OPENING_ply():
    g = _play(drop_play_at=2)
    assert g["aborted"] and g["abort_reason"] == "state_divergence"
    assert "after opening ply 2" in g["abort_detail"]


def test_an_incumbent_not_bound_to_the_game_seed_aborts_the_game():
    g = _play(seed=SYNTHETIC + 99)
    assert g["aborted"] and "seed" in g["abort_detail"] and g["rows"] == []


def test_an_aborted_game_compiles_no_labels(played):
    games, _ = played
    with pytest.raises(EL.Stage1Stop, match="aborted"):
        EL.rows_from_games([games[0], _play(drop_play_at=2)], config_sha256="c", runtime_sha256="r")


def test_export_reloads_identically_in_a_fresh_read(played, tmp_path):
    _games, rows = played
    path = tmp_path / "rows.jsonl"
    sha = EL._write_new(path, EL.rows_bytes(rows))
    s = EL.reload_summary(path)
    assert s["n"] == len(rows) and s["file_sha256"] == sha == s["rows_sha256"]
    assert EL.load_rows(path) == rows
    with pytest.raises(FileExistsError):
        EL._write_new(path, b"")


@pytest.mark.parametrize("tamper,match", [
    (lambda r: r["legal_moves"].reverse(), "legal moves"),
    (lambda r: r.update(chosen_index=r["chosen_index"] + 1), "recorded index"),
    (lambda r: r.update(split="heldout"), "split"),
    (lambda r: r["prefix"].append(r["prefix"][-1]), "illegal"),
    (lambda r: r.update(target_type="visit_distribution"), "teacher_chosen_move"),
])
def test_a_tampered_row_is_rejected(played, tamper, match):
    row = json.loads(json.dumps(played[1][0]))
    tamper(row)
    with pytest.raises(EL.Stage1Stop, match=match):
        EL.verify_row(row)


def test_a_crossing_bridge_claimed_in_a_row_is_rejected():
    """F06 in a row: the recorded digest claims black's CROSSING bridge exists; the rebuild refuses it."""
    moves = [(5, 5), (5, 6), (7, 6), (7, 5)]
    st = TwixtState.from_moves(moves)
    forged = st.copy()
    forged.bridges.add(((5, 6), (7, 5)))
    legal = [list(m) for m in st.legal_moves()]
    row = {"schema": EL.ROW_SCHEMA, "target_type": EL.TARGET_TYPE, "row_id": "x", "family": 0, "split": "train",
           "teacher_colour": "red", "prefix": [list(m) for m in moves], "ply": 4, "side": "red",
           "legal_moves": legal, "legal_digest": EL.legal_digest(legal), "chosen": legal[0], "chosen_index": 0,
           "state_digest": EL.state_digest(forged)}
    with pytest.raises(EL.Stage1Stop, match="rebuilt state"):
        EL.verify_row(row)
    EL.verify_row(dict(row, state_digest=EL.state_digest(st)))


def _rows(n, split, family, game_id="g", digest=None):
    return [{"split": split, "family": family, "game_id": game_id, "state_digest": digest or f"{split}{i}"}
            for i in range(n)]


def test_split_floors_leakage_and_family_binding():
    assert EL.split_checks(_rows(96, "train", 0) + _rows(32, "heldout", 4))["train"] == 96
    with pytest.raises(EL.Stage1Stop, match="insufficient"):
        EL.split_checks(_rows(95, "train", 0) + _rows(32, "heldout", 4))
    with pytest.raises(EL.Stage1Stop, match="insufficient"):
        EL.split_checks(_rows(96, "train", 0) + _rows(31, "heldout", 4))
    with pytest.raises(EL.Stage1Stop, match="both splits"):
        EL.split_checks(_rows(96, "train", 0) + _rows(31, "heldout", 4) + _rows(1, "heldout", 5, digest="train0"))
    with pytest.raises(EL.Stage1Stop, match="frozen split"):
        EL.split_checks(_rows(96, "train", 0) + _rows(32, "heldout", 3))


# ─────────────────────────────── the logit artifact and f ───────────────────────────────

def _train_rows(played):
    return [r for r in played[1] if r["split"] == "train"]


def test_the_logit_artifact_binds_bytes_rows_and_incumbent(played, tmp_path):
    rows = _train_rows(played)
    logits = [[float(i % 7) for i in range(len(r["legal_moves"]))] for r in rows]
    path = tmp_path / "logits.json"
    sha = EL.write_logit_artifact(path, rows, logits, EL.REFERENCE_SHA1)
    assert EL.load_logit_artifact(path, sha256=sha, rows=rows, incumbent_sha1=EL.REFERENCE_SHA1) == logits
    with pytest.raises(EL.Stage1Stop, match="not bound"):
        EL.load_logit_artifact(path, sha256=sha, rows=rows, incumbent_sha1="0" * 40)
    reordered = json.loads(json.dumps(rows))
    reordered[0]["legal_moves"].reverse()
    reordered[0]["legal_digest"] = EL.legal_digest(reordered[0]["legal_moves"])
    with pytest.raises(EL.Stage1Stop, match="not bound"):
        EL.load_logit_artifact(path, sha256=sha, rows=reordered, incumbent_sha1=EL.REFERENCE_SHA1)
    changed = tmp_path / "changed.json"
    changed.write_bytes(path.read_bytes().replace(b"1.0", b"1.5", 1))
    with pytest.raises(EL.Stage1Stop, match="pinned bytes"):
        EL.load_logit_artifact(changed, sha256=sha, rows=rows, incumbent_sha1=EL.REFERENCE_SHA1)
    held = [dict(rows[0], split="heldout")]
    with pytest.raises(EL.Stage1Stop, match="held-out"):
        EL.write_logit_artifact(tmp_path / "h.json", held, logits[:1], EL.REFERENCE_SHA1)


def test_f_counts_ties_by_the_first_maximum_and_f_zero_refuses_training():
    rows = [{"game_id": "a", "teacher_colour": "red", "chosen_index": i} for i in (0, 1, 2)]
    logits = [[3.0, 3.0, 1.0], [3.0, 3.0, 1.0], [0.0, 1.0, 5.0]]
    rep = EL.disagreement(rows, logits)
    assert (rep["disagree"], rep["match"], rep["f"]) == (1, 2, 1 / 3)       # the tied second index disagrees
    EL.require_training_opportunity(rep)
    zero = EL.disagreement([rows[0], rows[2]], [logits[0], logits[2]])
    assert zero["f"] == 0.0
    with pytest.raises(EL.Stage1Stop, match="f = 0"):
        EL.require_training_opportunity(zero)


# ─────────────────────────────── MLX: batching, canonicalization, the scalar control ───────────────────────────────

def _black_row(played):
    return next(r for r in played[1] if r["side"] == "black")


def test_a_chosen_move_beyond_slot_512_keeps_its_target(played):
    pytest.importorskip("mlx.core")
    from scripts.GPU.alphazero.trainer import make_padded_batch

    row = json.loads(json.dumps(_train_rows(played)[0]))
    assert len(row["legal_moves"]) > 520
    row["chosen_index"], row["chosen"] = 520, row["legal_moves"][520]
    target = np.asarray(EL.target_batch([row])[4])
    assert target[0, 520] == 1.0 and target.sum() == 1.0
    legacy = np.asarray(make_padded_batch([EL.position_record(row)])[4])       # the default cap of 512
    assert legacy.sum() == 0.0, "the hazard the 576 cap exists for"


def test_the_target_follows_the_move_through_black_canonicalization(played):
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    from scripts.GPU.alphazero.network import AlphaZeroNetwork
    from scripts.GPU.alphazero.self_play import PositionRecord
    from scripts.GPU.alphazero.trainer import compute_masked_log_probs, make_padded_batch

    mx.random.seed(0)
    net = AlphaZeroNetwork()
    net.eval()
    row = _black_row(played)
    rec = EL.position_record(row)
    assert rec.to_move == "black" and rec.target_type == EL.TARGET_TYPE

    def target_logprob(r):
        b, mr, mc, mask, t, _ = make_padded_batch([r], max_moves_cap=EL.MAX_MOVES_CAP)
        logits, _v, _ = net.forward_padded(b, mr, mc, mask, active_size=24)
        lp = compute_masked_log_probs(logits, mask)
        return float(mx.sum(lp * t)), np.asarray(lp)[0, :len(r.legal_moves)]

    base, lp = target_logprob(rec)
    perm = list(reversed(range(len(rec.legal_moves))))
    flipped = PositionRecord(board_tensor=rec.board_tensor, to_move="black",
                             legal_moves=[rec.legal_moves[i] for i in perm],
                             visit_counts=[rec.visit_counts[i] for i in perm], outcome=0.0, active_size=24, ply=rec.ply)
    again, _ = target_logprob(flipped)
    assert math.isclose(base, again, rel_tol=1e-5, abs_tol=1e-5)
    assert np.ptp(lp) > 0, "a constant policy would make the alignment check vacuous"


def test_scaled_probabilities_remask_padding_after_scaling():
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    Z, mask = EL._padded([[1.0, 0.0], [0.0, 0.0, 0.0]])
    p0 = EL.scaled_probs(Z, mask, 0.0)
    assert np.allclose(p0[0, :2], 0.5) and p0[0, 2] == 0.0 and np.allclose(p0[1], 1 / 3)
    small = EL.scaled_probs(Z, mask, 1e-6)
    assert small[0, 2] == 0.0
    naive = np.asarray(mx.softmax(mx.array(np.float32(0.0) * Z), axis=1))
    assert naive[0, 2] > 0.3, "without re-masking, beta = 0 ranks padding with the legal moves"


def test_fit_beta_finds_known_optima():
    pytest.importorskip("mlx.core")
    # interior: p(teacher) = sigmoid(beta) on two rows of three; the third row is beta-free -> ln 2
    fit = EL.fit_beta([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0, 0.0]], [0, 0, 1, 2])
    assert abs(fit["beta"] - math.log(2)) < 1e-5 and fit["iterations"] <= 64
    assert abs(fit["derivative"]) < 1e-4 and fit["bracket"][1] - fit["bracket"][0] <= 1e-6
    # endpoint 0: the teacher always plays the LOWEST logit -> the one-sided derivative at 0 is >= 0
    zero = EL.fit_beta([[2.0, 0.0], [0.0, 2.0]], [1, 0])
    assert zero["beta"] == 0.0 and zero["derivative"] >= 0 and zero["iterations"] == 0
    # unbounded: the teacher is always top-1 -> CE still decreasing at 100
    with pytest.raises(EL.Stage1Stop, match="still decreasing"):
        EL.fit_beta([[2.0, 0.0], [0.0, 2.0]], [0, 1])          # derivative underflows to 0 at 100
    with pytest.raises(EL.Stage1Stop, match="still decreasing"):
        EL.fit_beta([[0.01, 0.0], [0.0, 0.01]], [0, 1])        # visibly negative at 100


def test_pure_temperature_rescaling_cannot_pass_the_heldout_comparison():
    rows = [{"game_id": g, "chosen_index": i} for g, i in (("a", 1), ("a", 0), ("b", 2))]
    inc = [[2.0, 1.0, 0.0], [2.0, 1.0, 0.0], [0.0, 3.0, 1.0]]
    m_inc = EL.game_metrics(rows, inc)
    for beta in (0.3, 0.5, 2.0):
        scaled = EL.game_metrics(rows, [[beta * x for x in z] for z in inc])
        assert not EL.heldout_passes(scaled, m_inc, scaled)
        assert not EL.heldout_passes(scaled, m_inc, m_inc)
    better = [[1.0, 2.5, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 3.5]]
    assert EL.heldout_passes(EL.game_metrics(rows, better), m_inc, EL.game_metrics(rows, [[0.5 * x for x in z] for z in inc]))
    with pytest.raises(EL.Stage1Stop):
        EL.heldout_passes({"a": {"ce": 1, "top1": 1}}, m_inc, m_inc)


# ─────────────────────────────── the ceiling and the record of progress ───────────────────────────────

PARENT = """
import pathlib, signal, subprocess, sys, time
mode, pidfile = sys.argv[1], pathlib.Path(sys.argv[2])
body = "import signal, time\\n"
if mode == "stubborn":
    body += "signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"
g = subprocess.Popen([sys.executable, "-c", body + "time.sleep(60)"])
pidfile.write_text(str(g.pid))
if mode != "orphan":
    time.sleep(60)
"""


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _grandchild(tmp_path, mode, monkeypatch):
    """parent -> child -> grandchild under run_group; returns (status, exit, reaped, grandchild pid)."""
    monkeypatch.setattr(EL, "GROUP_GRACE_S", 0.5)
    script, pidfile = tmp_path / "parent.py", tmp_path / "grandchild.pid"
    script.write_text(PARENT)
    t0 = time.monotonic()
    status, code, reaped = EL.run_group([sys.executable, str(script), mode, str(pidfile)],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
    assert time.monotonic() - t0 < 10
    return status, code, reaped, int(pidfile.read_text())


@pytest.mark.parametrize("mode,status", [("hang", "TIMEOUT"), ("stubborn", "TIMEOUT"), ("orphan", "EXITED")])
def test_the_ceiling_ends_the_whole_process_group(tmp_path, monkeypatch, mode, status):
    """A timed-out child's query subprocess (hang), one that ignores SIGTERM (stubborn), and one left
    behind by a child that exited (orphan): none outlives the run."""
    got, _code, reaped, pid = _grandchild(tmp_path, mode, monkeypatch)
    assert (got, reaped) == (status, True) and not _alive(pid)


def test_control_a_plain_timeout_leaves_the_grandchild_running(tmp_path):
    """The defect the group fixes: subprocess.run(timeout=) kills only the direct child."""
    script, pidfile = tmp_path / "parent.py", tmp_path / "grandchild.pid"
    script.write_text(PARENT)
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run([sys.executable, str(script), "hang", str(pidfile)], timeout=2)
    pid = int(pidfile.read_text())
    try:
        assert _alive(pid)
    finally:
        os.kill(pid, 9)


def test_a_query_timeout_keeps_its_partial_output(tmp_path, monkeypatch):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\necho partial-out\necho partial-err >&2\nexec sleep 30\n")
    fake.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(fake))
    with pytest.raises(EL.Stage1Stop, match="timed out after 1 s"):
        EL._subprocess_json(["_query"], tmp_path / "queries-a.json", 1, "queries")
    assert (tmp_path / "queries-a.stdout.txt").read_text() == "partial-out\n"
    assert (tmp_path / "queries-a.stderr.txt").read_text() == "partial-err\n"


def test_every_move_is_logged_as_it_is_played_and_an_aborted_game_keeps_its_partial_moves():
    full, part = [], []
    g = EL.play_teacher_game(game=GAME, opening=[[12, 12], [11, 10], [13, 9], [10, 13]], rt=rt(),
                             player=FakePlayer(), reference_agent=FakeAgent(GAME["seed"]), log=full.append)
    assert [{k: v for k, v in e.items() if k != "event"} for e in full] == g["moves"]
    a = EL.play_teacher_game(game=GAME, opening=[[12, 12], [11, 10], [13, 9], [10, 13]], rt=rt(drop_play_at=2),
                             player=FakePlayer(), reference_agent=FakeAgent(GAME["seed"]), log=part.append)
    assert a["aborted"] and [e["ply"] for e in part] == [0, 1, 2]


def _staged(tmp_path, monkeypatch, *, logits_agree):
    """The REAL stage1 and _child against the fake engine: gate opened in memory only, the plan a
    synthetic block in tmp_path, every engine/model entry a stand-in."""
    from scripts.GPU.alphazero import e4_screen_reference as REG
    from scripts.GPU.alphazero import twixtbot_g3_reference as RF

    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(EL.build_plan((SYNTHETIC, SYNTHETIC + 18), "a" * 64)))
    clone, run_dir = tmp_path / "clone", tmp_path / "run"
    clone.mkdir(), run_dir.mkdir()
    (clone / "LICENSE").write_text("synthetic\n")
    monkeypatch.setattr(GATE, "EXTERNAL_LEARNING_STAGE1_AUTHORIZED", True)
    monkeypatch.setattr(EL, "PLAN", str(plan))
    monkeypatch.setattr(EL, "_committed", lambda rel: True)
    monkeypatch.setattr(REG, "seed_is_accounted", lambda s: True)
    monkeypatch.setattr(REG, "seed_is_unavailable", lambda s: False)
    fake_rt = {**rt(), "nneval": SimpleNamespace(NNEvaluater=lambda path: None),
               "nnmplayer": SimpleNamespace(Player=lambda **kw: FakePlayer()), "ct": None}
    monkeypatch.setattr(EL, "load_runtime", lambda c: (fake_rt, {"runtime_sha256": "r" * 64}))
    monkeypatch.setattr(EL, "teacher_kwargs", lambda ct: {})
    monkeypatch.setattr(EL, "PLY_CAP", 24)                         # short games; the floors scaled to match
    monkeypatch.setattr(EL, "MIN_TRAIN_ROWS", 8)
    monkeypatch.setattr(EL, "MIN_HELDOUT_ROWS", 4)

    def subproc(args, out, timeout, step):
        if step == "queries":
            return [{"fixture": f} for f in EL.QUERY_FIXTURES]
        return EL.reload_summary(pathlib.Path(args[args.index("--rows") + 1]))
    monkeypatch.setattr(EL, "_subprocess_json", subproc)
    monkeypatch.setattr(RF, "load_reference_evaluator",
                        lambda ref, root: SimpleNamespace(_g3_sha1=EL.REFERENCE_SHA1, network=None))
    monkeypatch.setattr(RF, "build_reference_agent", lambda task, evaluator, colour: FakeAgent(task["seed"]))
    monkeypatch.setattr(RF, "between_games_cleanup", lambda: None)
    monkeypatch.setattr(EL, "raw_logits", lambda net, rows: [
        [1.0 if (i == r["chosen_index"]) == logits_agree else 0.0 for i in range(len(r["legal_moves"]))]
        for r in rows])
    return run_dir, clone


def _progress(run_dir):
    return [json.loads(ln) for ln in (run_dir / "progress.jsonl").read_text().splitlines()]


def test_an_f_zero_stop_keeps_every_completed_step_and_the_f_report(tmp_path, monkeypatch):
    run_dir, clone = _staged(tmp_path, monkeypatch, logits_agree=True)
    assert EL._child(run_dir, str(clone)) == EL.EXIT_STOP_F_ZERO
    rec = json.loads((run_dir / "record.json").read_text())
    assert rec["verdict"] == "STOP_F_ZERO" and rec["stop"]["step"] == "f"
    assert rec["steps"]["disagreement"]["disagree"] == 0 and rec["steps"]["rows"]["n"] > 0
    events = _progress(run_dir)
    assert sorted(e["step"] for e in events if e["event"] == "step") == sorted(rec["steps"])
    assert [e["game_id"] for e in events if e["event"] == "game_start"] == [f"f{j}-{a}" for j in range(6)
                                                                         for a in ("teacher_red", "teacher_black")]
    moves = [e for e in events if e["event"] == "move"]
    assert len(moves) == sum(g["plies"] for g in rec["steps"]["games"])


def test_the_staged_stage_passes_when_rows_disagree(tmp_path, monkeypatch):
    run_dir, clone = _staged(tmp_path, monkeypatch, logits_agree=False)
    assert EL._child(run_dir, str(clone)) == EL.EXIT_PASS
    rec = json.loads((run_dir / "record.json").read_text())
    assert rec["verdict"] == "PASS" and rec["steps"]["disagreement"]["disagree"] > 0


def test_the_ceilings_sigterm_is_a_recorded_stop_and_the_handler_is_restored(tmp_path, monkeypatch):
    import signal

    run_dir, clone = _staged(tmp_path, monkeypatch, logits_agree=False)

    def not_the_childs(signum, frame):                  # so a missing handler fails cleanly, never kills pytest
        raise RuntimeError("the test's own SIGTERM handler ran")
    outer = signal.signal(signal.SIGTERM, not_the_childs)

    class Killed(FakePlayer):
        def pick_move(self, game, window=None):
            os.kill(os.getpid(), signal.SIGTERM)
            return super().pick_move(game, window)
    fake_rt = {**rt(), "nneval": SimpleNamespace(NNEvaluater=lambda p: None),
               "nnmplayer": SimpleNamespace(Player=lambda **kw: Killed()), "ct": None}
    monkeypatch.setattr(EL, "load_runtime", lambda c: (fake_rt, {"runtime_sha256": "r" * 64}))
    try:
        assert EL._child(run_dir, str(clone)) == EL.EXIT_STOP
        assert signal.getsignal(signal.SIGTERM) is not_the_childs
    finally:
        signal.signal(signal.SIGTERM, outer)
    rec = json.loads((run_dir / "record.json").read_text())
    assert rec["stop"]["step"] == "games" and "ceiling" in rec["stop"]["detail"]


def test_a_stop_mid_game_keeps_the_attempted_game_and_its_partial_moves(tmp_path, monkeypatch):
    run_dir, clone = _staged(tmp_path, monkeypatch, logits_agree=False)

    class Dies(FakePlayer):
        calls = 0

        def pick_move(self, game, window=None):
            Dies.calls += 1
            if Dies.calls == 3:
                raise RuntimeError("synthetic teacher failure")
            return super().pick_move(game, window)
    monkeypatch.setattr(EL, "load_runtime", lambda c: ({**rt(), "nneval": SimpleNamespace(NNEvaluater=lambda p: None),
                                                        "nnmplayer": SimpleNamespace(Player=lambda **kw: Dies()),
                                                        "ct": None}, {"runtime_sha256": "r" * 64}))
    assert EL._child(run_dir, str(clone)) == EL.EXIT_STOP
    rec = json.loads((run_dir / "record.json").read_text())
    assert rec["stop"]["step"] == "games" and "fixtures" in rec["steps"] and "games" not in rec["steps"]
    events = _progress(run_dir)
    assert [e["game_id"] for e in events if e["event"] == "game_start"] == ["f0-teacher_red"]
    assert [e["mover"] for e in events if e["event"] == "move"] == ["opening"] * 4 + ["teacher", "incumbent"] * 2


# ─────────────────────────────── the gate and the exit code ───────────────────────────────

def test_the_gate_is_closed_and_every_engine_entry_refuses(tmp_path):
    assert GATE.EXTERNAL_LEARNING_STAGE1_AUTHORIZED is False
    assert EL.main(["run", "--clone-root", str(tmp_path)]) == EL.EXIT_REFUSED
    assert EL.main(["_query", "--clone-root", str(tmp_path), "--out", str(tmp_path / "q.json")]) == EL.EXIT_REFUSED
    assert EL.main(["_child", "--run-dir", str(tmp_path), "--clone-root", str(tmp_path)]) == EL.EXIT_REFUSED
    assert not (EL.ROOT / EL.RUN_DIR).exists() and not (EL.ROOT / EL.EVIDENCE_DIR).exists()
    assert list(tmp_path.iterdir()) == []


def test_the_gate_module_holds_only_its_declaration():
    import ast

    tree = ast.parse(pathlib.Path(GATE.__file__).read_text())
    assert [type(n).__name__ for n in tree.body] == ["Expr", "Assign"]


def _run_with(tmp_path, monkeypatch, *, reaped):
    """The REAL run() with the gate opened in memory, ROOT a temp tree and the group runner replaced by
    a fake child that writes its record; returns (exit code, run dir, evidence dir, cleanup record)."""
    monkeypatch.setattr(GATE, "EXTERNAL_LEARNING_STAGE1_AUTHORIZED", True)
    monkeypatch.setattr(EL, "ROOT", tmp_path)
    monkeypatch.setattr(EL, "_git", lambda *a: "" if a[0] == "status" else "f" * 40)

    def fake_group(argv, *, stdout, stderr, timeout):
        run_dir = pathlib.Path(argv[argv.index("--run-dir") + 1])
        (run_dir / "record.json").write_text(json.dumps({"verdict": "PASS"}))
        return "EXITED", EL.EXIT_PASS, reaped
    monkeypatch.setattr(EL, "run_group", fake_group)
    code = EL.run(str(tmp_path / "clone"))
    return code, tmp_path / EL.RUN_DIR, tmp_path / EL.EVIDENCE_DIR, tmp_path / EL.CLEANUP_FAILED


def test_an_unreaped_group_finalizes_nothing_and_leaves_the_run_directory(tmp_path, monkeypatch):
    code, run_dir, evidence, cleanup = _run_with(tmp_path, monkeypatch, reaped=False)
    assert code == EL.EXIT_STOP and not evidence.exists()
    assert not (run_dir / "finalized.json").exists() and (run_dir / "record.json").exists()
    rec = json.loads(cleanup.read_text())
    assert rec["group_reaped"] is False and rec["finalized"] is False and "files" not in rec
    assert cleanup.parent == run_dir.parent and "verdict" not in rec


def test_control_a_reaped_group_finalizes_and_copies(tmp_path, monkeypatch):
    code, run_dir, evidence, cleanup = _run_with(tmp_path, monkeypatch, reaped=True)
    assert code == EL.EXIT_PASS and not cleanup.exists()
    assert (run_dir / "finalized.json").exists() and (evidence / "MANIFEST.sha256").exists()


@pytest.mark.parametrize("verdict,status,child,code", [
    ("PASS", "EXITED", 0, 0), ("STOP_F_ZERO", "EXITED", 1, 1), ("STOP", "EXITED", 2, 2),
    ("PASS", "EXITED", 1, 2), ("STOP", "EXITED", 0, 2), (None, "EXITED", 0, 2), ("PASS", "TIMEOUT", None, 4),
])
def test_the_exit_code_is_the_recorded_verdict_and_must_agree(tmp_path, verdict, status, child, code):
    rec = tmp_path / "record.json"
    if verdict is not None:
        rec.write_text(json.dumps({"verdict": verdict}))
    assert EL.exit_code_for(rec, status, child) == code
    assert EL.exit_code_for(rec, status, child, False) == (4 if status == "TIMEOUT" else 2)   # a live descendant
    assert EL.exit_code_for(rec, status, child, None) == (4 if status == "TIMEOUT" else 2)    # never ran
