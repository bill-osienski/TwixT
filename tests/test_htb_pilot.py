"""H-TB pilot: focused CPU checks with fake teachers, synthetic games and tiny networks.

No checkpoint, corpus, TensorFlow or GPU: every test pins MLX to the CPU, and the two runtime entries are only
shown to refuse behind their closed gates.
"""
import hashlib
import json
import pathlib

import mlx.core as mx
import numpy as np
import pytest

from scripts.GPU.alphazero import htb_fit_authorization as FIT_GATE
from scripts.GPU.alphazero import htb_label_authorization as LABEL_GATE
from scripts.GPU.alphazero import htb_pilot as H
from scripts.GPU.alphazero import twixtbot_adapter as A
from scripts.GPU.alphazero.capacity_c0_inventory import split_of
from scripts.GPU.alphazero.game.twixt_state import TwixtState
from scripts.GPU.alphazero.mirror_averaging import MirrorAveragingEvaluator
from scripts.GPU.alphazero.network import AlphaZeroNetwork
from scripts.GPU.alphazero.self_play import _mirror_position_lr
from scripts.GPU.alphazero.trainer import compute_masked_log_probs

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def random_game(seed, n):
    rng = np.random.default_rng(seed)
    s, moves = TwixtState(), []
    for _ in range(n):
        legal = sorted(s.legal_moves())
        if not legal or s.winner() is not None:
            break
        m = legal[int(rng.integers(len(legal)))]
        moves.append(m)
        s = s.apply_move(m)
    return moves


def ids_in(arm, n, start=0):
    out, i = [], start
    while len(out) < n:
        gid = f"iter_9999_game_{i:05d}"
        if split_of(gid) == arm:
            out.append(gid)
        i += 1
    return out


class _Box:
    def __init__(self):
        self.s = TwixtState()


class FakeTeacher:
    """Our own engine as the 'teacher engine'; outputs a fixed, NON-symmetric function of the state."""

    def __init__(self, corrupt_at=None, smooth=False):
        self.calls, self.corrupt_at, self.smooth = 0, corrupt_at, smooth

    def new_game(self):
        return _Box()

    def play(self, tb, rc):
        tb.s = tb.s.apply_move(rc)

    def divergences(self, our, tb):
        if self.corrupt_at is not None and our.ply == self.corrupt_at:
            return ["injected"]
        return [] if (our.pegs == tb.s.pegs and our.bridges == tb.s.bridges and our.to_move == tb.s.to_move) else ["x"]

    def evaluate(self, tb, colour, legal):
        self.calls += 1
        if self.smooth:                              # learnable from our input: prefer the centre; value by peg count
            logits = np.zeros(A.POLICY_N)
            for r, c in legal:
                logits[A.policy_index(colour, r, c)] = -0.4 * (abs(r - 11.5) + abs(c - 11.5))
            return H.legal_softmax(logits, colour, legal), float(np.clip((tb.s.ply - 15) / 20, -0.9, 0.9))
        base = hashlib.sha256(H.state_serial(tb.s)).digest()
        logits = np.zeros(A.POLICY_N)
        for r, c in legal:
            h = hashlib.sha256(base + bytes([r, c])).digest()
            logits[A.policy_index(colour, r, c)] = h[0] / 32.0
        v = (base[1] / 255.0) * 2 - 1
        return H.legal_softmax(logits, colour, legal), v


def tiny_net(seed=0):
    mx.random.seed(seed)                             # MLX initializes randomly: pin it so every test is repeatable
    net = AlphaZeroNetwork(hidden=16, n_blocks=1)    # 8 channels can leave the 2-channel policy bottleneck dead
    mx.eval(net.parameters())
    net.eval()
    return net


def twin(net):
    other = AlphaZeroNetwork(hidden=16, n_blocks=1)
    other.update(net.parameters())
    mx.eval(other.parameters())
    other.eval()
    return other


# ─────────────────────────────── card, gates, frozen numbers ───────────────────────────────

def test_THE_CARD_IS_THE_PINNED_ONE():
    assert hashlib.sha256((ROOT / H.CARD).read_bytes()).hexdigest() == H.CARD_SHA256


def test_BOTH_GATES_ARE_CLOSED_AND_BOTH_ENTRIES_REFUSE_BEFORE_ANY_WORK(capsys):
    assert LABEL_GATE.HTB_LABEL_AUTHORIZED is False and FIT_GATE.HTB_FIT_AUTHORIZED is False
    assert H.main(["run", "label", "--clone-root", "/nonexistent"]) == 2
    assert H.main(["run", "fit"]) == 2
    err = capsys.readouterr().err
    assert err.count("REFUSED") == 2 and "HTB_LABEL_AUTHORIZED is closed" in err and "HTB_FIT_AUTHORIZED is closed" in err
    assert not (ROOT / H.EVIDENCE_LABEL).exists() and not (ROOT / H.EVIDENCE_FIT).exists()


def test_THE_FROZEN_COUNTS_AND_RECIPE_ARE_THE_CARDS():
    assert (H.N_TRAIN, H.N_VAL) == (240_000, 24_000)
    assert H.FROZEN_UPDATES == 15_000
    assert H.FROZEN_RECIPE == H.Recipe(epochs=2, rows_per_batch=32, learning_rate=1e-4, value_weight=1.0,
                                       l2_weight=1e-4, clip_norm=1.0)
    assert (H.F1_FACTOR, H.F3_FACTOR, H.START_KL_FLOOR) == (0.5, 0.5, 0.01)
    from scripts.GPU.alphazero.twixtbot_g3_schedule import ANCHOR
    assert {k: ANCHOR[k] for k in H.TEACHER} == H.TEACHER


# ─────────────────────────────── reflection and keys ───────────────────────────────

@pytest.mark.parametrize("seed", range(6))
def test_THE_REFLECTED_REPLAY_IS_R_OF_THE_STATE_AND_OF_ITS_TENSOR(seed):
    moves = random_game(seed, 70)
    s, rs = TwixtState(), TwixtState()
    for m in moves:
        assert H.reflect_state_matches(s, rs)
        assert H.state_key(s) == H.state_key(rs)
        assert H.board_hwc(s).shape == (H.BOARD, H.BOARD, 30)
        mb = _mirror_position_lr(H.board_hwc(s), [], [], H.BOARD)[0]
        np.testing.assert_array_equal(mb, H.board_hwc(rs))
        s, rs = s.apply_move(m), rs.apply_move(H.reflect_rc(m))
    assert len(s.bridges) > 0


def test_THE_KEY_IDENTIFIES_STATES_NOT_MOVE_ORDERS():
    a = [(5, 5), (12, 3), (18, 18), (3, 12)]
    b = [(18, 18), (3, 12), (5, 5), (12, 3)]                 # same pegs, no bridges, same side to move
    sa, sb = TwixtState.from_moves(a), TwixtState.from_moves(b)
    assert sa == sb and H.state_key(sa) == H.state_key(sb)
    sc = TwixtState.from_moves([(5, 5), (12, 3), (18, 18), (3, 13)])
    assert H.state_key(sc) != H.state_key(sa)
    assert H.state_key(TwixtState.from_moves(a[:3])) != H.state_key(sa)


def test_A_TRUE_REFLECTION_MISMATCH_IS_DETECTED():
    s = TwixtState.from_moves([(5, 5), (12, 3)])
    assert not H.reflect_state_matches(s, s)                 # s is not its own reflection here


# ─────────────────────────────── selection ───────────────────────────────

def _red(gid, moves):
    """A checked red-first game, as `eligible_pool` returns it (card §14)."""
    return {"id": gid, "starting_player": "red", "moves": list(moves),
            "players": [("red", "black")[k % 2] for k in range(len(moves))]}


def _games(opening, specs):
    out = []
    for gid, seed in specs:
        tail = random_game(seed, 40)
        s = TwixtState.from_moves(opening)
        moves = list(opening)
        for m in tail:
            if m in s.legal_moves() and s.winner() is None:
                moves.append(m)
                s = s.apply_move(m)
        out.append(_red(gid, moves))
    return out


def test_SELECTION_DEDUPS_EXCLUDES_EVERY_TRAIN_STATE_FROM_VAL_AND_IS_DETERMINISTIC():
    opening = [(10, 10), (5, 8), (12, 11)]
    tr, va = ids_in("train", 6), ids_in("val", 4)
    games = _games(opening, [(g, 100 + i) for i, g in enumerate(tr)] + [(g, 200 + i) for i, g in enumerate(va)])
    # a val game that is the exact reflection of a train game: every one of its states must leak out
    games.append(_red(ids_in("val", 1, start=10_000)[0], [H.reflect_rc(m) for m in games[0]["moves"]]))
    train_keys = {H.state_key(s) for g in games if split_of(g["id"]) == "train" for _, s in H.positions(g["moves"])}
    sel = H.select(games, n_train=40, n_val=20)
    assert sel["val_keys_excluded_as_in_train"] >= len(games[0]["moves"])        # opening + the reflected game
    for arm, n in (("train", 40), ("val", 20)):
        keys = [r["key"] for r in sel["rows"][arm]]
        assert len(keys) == n == len(set(keys))
    assert not {r["key"] for r in sel["rows"]["val"]} & train_keys
    assert H.select(list(reversed(games)), n_train=40, n_val=20)["rows"] == sel["rows"]
    for arm in ("train", "val"):                             # each row's key is its recorded position's
        by_id = {g["id"]: g for g in games}
        for r in sel["rows"][arm][:10]:
            s = dict(H.positions(by_id[r["game_id"]]["moves"]))[r["ply"]]
            assert H.state_key(s) == r["key"]


def test_SELECTION_REFUSES_A_SMALL_POOL_AND_ANY_TEST_ARM_GAME():
    games = _games([(10, 10)], [(g, 300 + i) for i, g in enumerate(ids_in("train", 2) + ids_in("val", 1))])
    with pytest.raises(H.Refused, match="fewer than"):
        H.select(games, n_train=10_000, n_val=1)
    with pytest.raises(H.Refused, match="test arm"):
        H.select(games + [_red(ids_in("test", 1)[0], [(10, 10)])], n_train=1, n_val=1)


def test_A_RECORDED_PLAYER_THAT_DISAGREES_STOPS():
    with pytest.raises(H.Stop, match="recorded player"):
        list(H.positions([(10, 10), (5, 8)], ["red", "red"]))


# ─────────────────────────────── card §14: red-first games only ───────────────────────────────

def _raw(gid, moves, start, players=None, **extra):
    """A corpus file's JSON, as self-play writes it."""
    other = {"red": "black", "black": "red"}.get(start, "red")
    pl = players or [(start, other)[k % 2] for k in range(len(moves))]
    g = {"id": gid, "timestamp": "2026-01-01T00:00:00", "winner": "red",
         "meta": {"iteration": 1, "reason": "win", "board_size": 24, "n_moves": len(moves)},
         "moves": [{"turn": k + 1, "player": pl[k], "row": r, "col": c} for k, (r, c) in enumerate(moves)], **extra}
    if start is not None:
        g["starting_player"] = start
    return g


def test_THE_POOL_ACCEPTS_RED_FIRST_GAMES_AND_EXCLUDES_BLACK_FIRST_ONES_COUNTING_BOTH():
    tr, va, te = ids_in("train", 3), ids_in("val", 2), ids_in("test", 1)
    red = random_game(900, 20)
    black = [(5, 10), (10, 5), (6, 12)]                       # legal for black (rows 1-22) then red (cols 1-22) ...
    raw = [_raw(tr[0], red, "red"), _raw(tr[1], black, "black"), _raw(tr[2], red[:9], "red"),
           _raw(va[0], random_game(905, 7), "red"), _raw(va[1], black, "black"), _raw(te[0], red, "red")]
    pool = H.pool_from(raw)
    assert [g["id"] for g in pool["games"]] == [tr[0], tr[2], va[0]]
    assert all(g["starting_player"] == "red" and g["players"][0] == "red" for g in pool["games"])
    assert pool["counts"] == {
        "train": {"games": 2, "positions": 29, "black_first_games_excluded": 1, "black_first_positions_excluded": 3},
        "val": {"games": 1, "positions": 7, "black_first_games_excluded": 1, "black_first_positions_excluded": 3}}
    assert [g["id"] for g in H.pool_from(raw, arms=("train",))["games"]] == [tr[0], tr[2]]
    assert H.pool_from(raw, arms=("train",))["counts"] == pool["counts"]      # val counted, never returned
    sel = H.select(pool["games"], n_train=5, n_val=2)                         # the filtered pool selects as before
    assert sel["games"] == {"train": 2, "val": 1}


@pytest.mark.parametrize("case", ["missing", "invalid", "disagrees", "not_alternating"])
def test_INCONSISTENT_START_METADATA_STOPS_AND_IS_NEVER_DEFAULTED(case):
    gid, mv = ids_in("train", 1)[0], random_game(901, 6)
    raw = {"missing": _raw(gid, mv, None),
           "invalid": _raw(gid, mv, "white"),
           "disagrees": _raw(gid, mv, "black", players=[("red", "black")[k % 2] for k in range(6)]),
           "not_alternating": _raw(gid, mv, "red", players=["red", "black", "black", "red", "black", "red"])}[case]
    with pytest.raises(H.Stop, match="starting_player") as e:
        H.pool_from([raw])
    assert e.value.step == "corpus"


def test_A_TEST_ARM_GAME_IS_NEITHER_CHECKED_NOR_COUNTED():
    raw = [_raw(ids_in("test", 1)[0], random_game(902, 6), None)]             # would STOP if it were checked
    assert H.pool_from(raw) == {"games": [], "counts": {a: {"games": 0, "positions": 0, "black_first_games_excluded": 0,
                                                            "black_first_positions_excluded": 0} for a in H.ARMS}}


def test_THE_POOL_AUTHENTICATES_THE_COMPLETE_CORPUS_BEFORE_FILTERING(tmp_path, monkeypatch):
    tr = ids_in("train", 2)
    files = {"a.json": _raw(tr[0], random_game(903, 8), "red"), "b.json": _raw(tr[1], [(5, 10)], "black")}
    for name, g in files.items():
        (tmp_path / name).write_text(json.dumps(g))
    seen = []

    def digest(pairs):
        seen.extend(name for name, _ in pairs)
        return H.ELIGIBLE_DIGEST
    monkeypatch.setattr(H.C0, "_digest_of", digest)
    monkeypatch.setattr(H.C0, "intact_suffix", lambda w: (1, 1))
    pool = H.eligible_pool(tmp_path)
    assert seen == ["a.json", "b.json"]                        # the black-first game is inside the digest
    assert [g["id"] for g in pool["games"]] == [tr[0]] and pool["counts"]["train"]["black_first_games_excluded"] == 1
    assert H.eligible_games(tmp_path) == pool["games"]


def test_EVERY_CONSUMER_REFUSES_A_GAME_THAT_IS_NOT_A_CHECKED_RED_FIRST_GAME():
    g = _red(ids_in("train", 1)[0], random_game(904, 6))
    for bad in ({**g, "starting_player": "black"}, {k: v for k, v in g.items() if k != "starting_player"},
                {k: v for k, v in g.items() if k != "players"}, {**g, "players": ["black"] + g["players"][1:]}):
        with pytest.raises(H.Stop, match="red-first") as e:
            H.select([bad], n_train=1, n_val=0)
        assert e.value.step == "pool"
    H.require_red_first([g])


def test_THE_LIVE_CORPUS_CHECK_REFUSES_A_CORPUS_THAT_IS_NOT_C0S(tmp_path):
    g = {"id": "iter_0001_game_000", "timestamp": "2026-01-01T00:00:00", "winner": "red",
         "meta": {"iteration": 1, "reason": "win", "board_size": 24, "n_moves": 1},
         "moves": [{"turn": 1, "player": "red", "row": 10, "col": 10}]}
    (tmp_path / "iter_0001_game_000.json").write_text(json.dumps(g))
    with pytest.raises(H.Refused, match="eligible digest"):
        H.eligible_games(tmp_path)


# ─────────────────────────────── the teacher's mapping and the labels ───────────────────────────────

@pytest.mark.parametrize("colour", ["red", "black"])
def test_THE_TEACHERS_SLOTS_LAND_ON_OUR_CELLS_AND_ONLY_LEGAL_MOVES_SHARE_THE_MASS(colour):
    s = TwixtState() if colour == "red" else TwixtState.from_moves([(10, 10)])
    legal = list(s.legal_moves())
    target = legal[len(legal) // 3]
    z = np.full(A.POLICY_N, -3.0)
    z[A.policy_index(colour, *target)] = 5.0
    p = H.legal_softmax(z, colour, legal)
    assert abs(p.sum() - 1) < 1e-12 and legal[int(np.argmax(p))] == target
    if colour == "black":                                    # an occupied cell's slot carries no mass
        assert (10, 10) not in legal
        z2 = z.copy()
        z2[A.policy_index(colour, 10, 10)] = 50.0
        assert legal[int(np.argmax(H.legal_softmax(z2, colour, legal)))] == target
    bad = z.copy()
    bad[0] = np.nan
    with pytest.raises(H.Stop, match="non-finite"):
        H.legal_softmax(bad, colour, legal)
    with pytest.raises(H.Stop, match="move logits"):
        H.legal_softmax(z[:-1], colour, legal)


def _label(moves, plies, teacher=None):
    rows = []
    for k, s in H.positions(moves):
        if k in plies:
            rows.append({"ply": k, "key": H.state_key(s)})
    return H.label_game("g", moves, rows, teacher or FakeTeacher())


def test_LABELS_ARE_MIRROR_AVERAGED_AND_THE_REFLECTED_GAME_GETS_THE_REFLECTED_LABELS():
    moves = random_game(7, 50)
    plies = {0, 3, 17, len(moves) - 1}
    t = FakeTeacher()
    rows = _label(moves, plies, t)
    assert t.calls == 2 * len(plies)
    rrows = _label([H.reflect_rc(m) for m in moves], plies)
    for a, b in zip(rows, rrows):
        assert a["key"] == b["key"]
        at = {tuple(m): i for i, m in enumerate(b["legal"])}
        np.testing.assert_allclose([b["target"][at[H.reflect_rc(m)]] for m in a["legal"]], a["target"], atol=1e-15)
        assert abs(a["value"] - b["value"]) < 1e-15 and abs(a["target"].sum() - 1) < 1e-12
        assert a["value"] == pytest.approx(0.5 * (a["value_id"] + a["value_ref"]))
    assert any(abs(r["value_id"] - r["value_ref"]) > 1e-6 for r in rows)    # the fake is not symmetric


def test_LABELLING_STOPS_ON_A_LOCKSTEP_DIVERGENCE_AND_ON_A_WRONG_KEY():
    moves = random_game(8, 30)
    with pytest.raises(H.Stop, match="lockstep"):
        _label(moves, {5}, FakeTeacher(corrupt_at=5))
    with pytest.raises(H.Stop, match="key differs"):
        H.label_game("g", moves, [{"ply": 5, "key": "0" * 64}], FakeTeacher())


def test_LABEL_SHARDS_ROUND_TRIP_AND_ANY_TAMPERING_IS_REFUSED(tmp_path):
    moves = random_game(9, 40)
    rows = _label(moves, set(range(0, 40, 3)))
    man = H.write_label_shards(rows, tmp_path / "labels", shard_rows=5)
    msha = hashlib.sha256((tmp_path / "labels" / "manifest.json").read_bytes()).hexdigest()
    back = H.load_label_shards(tmp_path / "labels", msha)
    assert man["n"] == len(rows) == len(back) and len(man["shards"]) == -(-len(rows) // 5)
    for a, b in zip(rows, back):
        t = np.array([b["target576"][A.padded_index(r, c)] for r, c in a["legal"]])
        np.testing.assert_allclose(t, a["target"], rtol=1e-6)
        assert b["value"] == a["value"] and b["key"] == a["key"]
    with pytest.raises(FileExistsError):
        H.write_label_shards(rows, tmp_path / "labels")
    with pytest.raises(H.Refused, match="manifest"):
        H.load_label_shards(tmp_path / "labels", "0" * 64)
    shard = tmp_path / "labels" / man["shards"][0]["file"]
    raw = bytearray(shard.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    shard.write_bytes(bytes(raw))
    with pytest.raises(H.Refused, match="changed since labelling"):
        H.load_label_shards(tmp_path / "labels", msha)


# ─────────────────────────────── examples, mirror averaging, scoring ───────────────────────────────

def _examples(n_games=3, every=4, seed0=20, smooth=False):
    labelled, moves_by_game = [], {}
    for i in range(n_games):
        moves = random_game(seed0 + i, 36)
        gid = f"g{i}"
        moves_by_game[gid] = moves
        rows = [{"ply": k, "key": H.state_key(s)} for k, s in H.positions(moves) if k % every == 1]
        for r in H.label_game(gid, moves, rows, FakeTeacher(smooth=smooth)):
            labelled.append({"game_id": gid, "ply": r["ply"], "key": r["key"], "value": r["value"],
                             "target576": _to576(r)})
    return H.build_examples(labelled, moves_by_game)


def _to576(r):
    t = np.zeros(H.PADDED)
    for (a, b), x in zip(r["legal"], r["target"]):
        t[A.padded_index(a, b)] = x
    return t


def test_EACH_EXAMPLES_BOARDS_ARE_THE_POSITION_AND_ITS_REFLECTION():
    moves = random_game(20, 36)                      # _examples' first game
    states = dict(H.positions(moves))
    rstates = dict(H.positions([H.reflect_rc(m) for m in moves]))
    for e in _examples(n_games=1)[:4]:
        rec, mir = e["record"], e["mirror"]
        assert H.state_key(states[rec.ply]) == e["key"]
        np.testing.assert_array_equal(rec.board_tensor, H.board_hwc(states[rec.ply]))
        np.testing.assert_array_equal(mir.board_tensor, H.board_hwc(rstates[rec.ply]))
        assert rec.legal_moves == list(states[rec.ply].legal_moves())
        assert mir.legal_moves == [H.reflect_rc(m) for m in rec.legal_moves]
        assert mir.visit_counts == rec.visit_counts and abs(sum(rec.visit_counts) - 1) < 1e-12


class _Base:
    def __init__(self, net):
        self.network = net

    def infer(self, boards, rows, cols, mask, S):
        logits, v, _ = self.network.forward_padded(mx.array(boards), mx.array(rows), mx.array(cols), mx.array(mask),
                                                   active_size=S)
        lp = compute_masked_log_probs(logits, mx.array(mask))
        return np.exp(np.array(lp)) * mask, np.array(v)


def test_THE_SCORED_MIRROR_AVERAGE_IS_THE_AGENTS_MIRROR_AVERAGE():
    net, ex = tiny_net(), _examples()[:9]
    q, vq = H.mirror_averaged(net, ex, chunk=4)
    boards, rows, cols, mask, _ = H._batch([e["record"] for e in ex])
    p, v = MirrorAveragingEvaluator(_Base(net)).infer(np.array(boards), np.array(rows), np.array(cols),
                                                      np.array(mask), H.BOARD)
    for j, e in enumerate(ex):
        np.testing.assert_allclose(q[j], p[j, :len(e["record"].legal_moves)], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(vq, v, rtol=1e-5, atol=1e-6)


def test_SCORE_AGAINST_ITS_OWN_MIRROR_AVERAGE_IS_ZERO_KL_FULL_AGREEMENT():
    net, ex = tiny_net(), _examples()[:8]
    q, vq = H.mirror_averaged(net, ex)
    own = [{**e, "target": qi / qi.sum(), "value": float(v)} for e, qi, v in zip(ex, q, vq)]
    s = H.score(net, own)
    assert abs(s["kl"]) < 1e-6 and s["top1"] == 1.0 and s["value_mse"] < 1e-12
    with pytest.raises(H.Stop, match="start"):
        H.start_check(net, own, n=8)


def test_TOP1_TIES_GO_TO_THE_FIRST_LEGAL_MOVE():
    assert H.top1(np.array([0.2, 0.4, 0.4])) == 1


@pytest.mark.parametrize("s,r,want", [
    ({"kl": 0.5, "top1": 0.31, "value_mse": 0.05}, {"kl": 1.0, "top1": 0.30, "value_mse": 0.10}, "ELIGIBLE"),
    ({"kl": 0.51, "top1": 0.31, "value_mse": 0.05}, {"kl": 1.0, "top1": 0.30, "value_mse": 0.10}, "STOP"),
    ({"kl": 0.5, "top1": 0.30, "value_mse": 0.05}, {"kl": 1.0, "top1": 0.30, "value_mse": 0.10}, "STOP"),
    ({"kl": 0.5, "top1": 0.31, "value_mse": 0.0501}, {"kl": 1.0, "top1": 0.30, "value_mse": 0.10}, "STOP"),
])
def test_THE_VERDICT_IS_F1_AND_F2_STRICT_AND_F3(s, r, want):
    assert H.verdict(s, r)["verdict"] == want


# ─────────────────────────────── the fit ───────────────────────────────

def test_EPOCH_ORDER_IS_A_DETERMINISTIC_PERMUTATION_THAT_DIFFERS_BY_EPOCH():
    keys = [f"{i:064x}" for i in range(50)]
    o0, o1 = H.epoch_order(keys, 0), H.epoch_order(keys, 1)
    assert sorted(o0) == list(range(50)) and o0 == H.epoch_order(keys, 0) and o0 != o1


def test_THE_FIT_MOVES_EVERY_PART_KEEPS_BATCHNORM_STATISTICS_AND_REPEATS_BITWISE():
    ex = _examples()[:16]
    recipe = H.Recipe(epochs=2, rows_per_batch=4, learning_rate=1e-2)
    ref = tiny_net()
    out = []
    for _ in range(2):
        st = twin(ref)
        rec = H.fit(st, ref, ex, recipe)
        out.append(rec)
        assert rec["updates"] == 8 and rec["optimizer"]["betas"] == [0.9, 0.999]
        assert rec["running_statistics_unchanged"] > 0
        assert any(n.startswith("value_head.") for n in H.surface_names(st))
    assert out[0]["student_digest"] == out[1]["student_digest"] != out[0]["reference_digest"]
    assert [x["loss"] for x in out[0]["log"]] == [x["loss"] for x in out[1]["log"]]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_THE_FIT_LEARNS_A_LEARNABLE_TEACHER_IN_BOTH_HEADS(seed):
    ex = _examples(smooth=True)[:16]
    ref = tiny_net(seed)
    st = twin(ref)
    before = H.score(st, ex)
    H.fit(st, ref, ex, H.Recipe(epochs=20, rows_per_batch=4, learning_rate=1e-3))
    after = H.score(st, ex)
    assert after["kl"] < 0.5 * before["kl"] and after["value_mse"] < 0.5 * before["value_mse"], (before, after)


def test_THE_FIT_REFUSES_A_SHARED_OBJECT_A_NON_COPY_TRAINING_MODE_AND_A_RAGGED_BATCH():
    ex = _examples()[:8]
    ref = tiny_net()
    with pytest.raises(H.Stop, match="same object"):
        H.fit(ref, ref, ex)
    with pytest.raises(H.Stop, match="exact copy"):
        H.fit(tiny_net(seed=1), ref, ex, H.Recipe(rows_per_batch=4))
    st = twin(ref)
    st.train()
    with pytest.raises(H.Stop, match="training mode"):
        H.fit(st, ref, ex, H.Recipe(rows_per_batch=4))
    with pytest.raises(H.Stop, match="batches"):
        H.fit(twin(ref), ref, ex[:7], H.Recipe(rows_per_batch=4))


def test_A_REFLECTED_REPLAY_THAT_IS_NOT_R_OF_THE_STATE_STOPS(monkeypatch):
    moves = random_game(10, 30)
    rows = [{"ply": 12, "key": H.state_key(dict(H.positions(moves))[12])}]
    real, calls = H._replay_and_evaluate, []

    def second_replay_is_not_reflected(ms, plies, teacher):         # the reflected replay replays the ORIGINAL moves
        calls.append(1)
        return real(moves if len(calls) == 2 else ms, plies, teacher)

    monkeypatch.setattr(H, "_replay_and_evaluate", second_replay_is_not_reflected)
    with pytest.raises(H.Stop, match="reflection"):
        H.label_game("g", moves, rows, FakeTeacher())


# ─────────────────────────────── the entries, end to end on fakes ───────────────────────────────

import os
import shutil
import signal


class _Spy:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, *a, **k):
        self.calls += 1
        return self.fn(*a, **k)


def _fake_games():
    games = []
    for i, gid in enumerate(ids_in("train", 4) + ids_in("val", 3)):
        moves = random_game(400 + i, 30)
        games.append(_red(gid, moves))
    return games


def _cfg(tmp_path, committed=None):
    root = tmp_path / "repo"
    (root / H.CARD).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / H.CARD, root / H.CARD)
    games = _fake_games()
    allowed = committed if committed is not None else {H.CARD}
    return H.Config(
        root=root, data_root=root, cache=tmp_path / "cache", n_train=16, n_val=8,
        memory_required_bytes=1, available_memory=lambda: {"physical_bytes": 2, "available_bytes": 2, "page_size": 1},
        recipe=H.Recipe(epochs=1, rows_per_batch=4, learning_rate=1e-3), start_rows=16, shard_rows=5,
        committed=lambda r, rel: rel in allowed,
        games=_Spy(lambda r: games),
        load_teacher=_Spy(lambda clone: (FakeTeacher(smooth=True), {"fake": True})),
        load_reference=_Spy(lambda r: tiny_net(0))), allowed


@pytest.fixture
def opened(monkeypatch):
    """Both gates open IN MEMORY ONLY (the files stay closed), and the TF32 setting the entries require."""
    monkeypatch.setattr(LABEL_GATE, "HTB_LABEL_AUTHORIZED", True)
    monkeypatch.setattr(FIT_GATE, "HTB_FIT_AUTHORIZED", True)
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")


def _labelled(tmp_path):
    cfg, allowed = _cfg(tmp_path)
    assert H.run_label("/fake-clone", cfg) == 0
    allowed |= {f"{H.EVIDENCE_LABEL}/record.json", f"{H.EVIDENCE_LABEL}/selection.json"}
    return cfg, allowed


def _record(path):
    return json.loads((path / "record.json").read_text())


def test_LABEL_THEN_FIT_COMPLETE_ON_FAKES_AND_LEAVE_TERMINAL_RECORDS(tmp_path, opened):
    cfg, _ = _labelled(tmp_path)
    lab = _record(cfg.evidence_label)
    assert lab["status"] == "COMPLETED" and lab["counts"] == {"train": 16, "val": 8} and lab["teacher_calls"] == 48
    assert (cfg.evidence_label / "started.json").is_file()
    assert H.run_fit(cfg) == 0
    rec = _record(cfg.evidence_fit)
    assert rec["status"] == "COMPLETED" and rec["fit"]["updates"] == 4 and rec["verdict"]["verdict"] in ("ELIGIBLE", "STOP")
    assert rec["selection_sha256"] == lab["selection_sha256"] and cfg.student.is_file()
    assert cfg.load_reference.calls == 2


def test_LABELLING_REFUSES_BEFORE_ANY_LOAD_WHEN_A_DESTINATION_EXISTS(tmp_path, opened):
    for make in (lambda c: c.evidence_label.mkdir(parents=True), lambda c: c.labels("val").mkdir(parents=True)):
        cfg, _ = _cfg(tmp_path / str(id(make)))
        make(cfg)
        assert H.main(["run", "label", "--clone-root", "/fake"], cfg) == 2
        assert cfg.load_teacher.calls == 0 and cfg.games.calls == 0
        assert not (cfg.evidence_label / "record.json").exists()


def _break(cfg, allowed, how):
    ev = cfg.evidence_label
    rec = _record(ev)
    if how == "uncommitted":
        allowed.discard(f"{H.EVIDENCE_LABEL}/record.json")
    elif how in ("card", "teacher", "status"):
        rec[{"card": "card_sha256", "teacher": "teacher", "status": "status"}[how]] = \
            {"card": "0" * 64, "teacher": {**H.TEACHER, "commit": "x"}, "status": "STOP"}[how]
        os.unlink(ev / "record.json")
        (ev / "record.json").write_text(json.dumps(rec))
    elif how == "selection":
        sel = json.loads((ev / "selection.json").read_text())
        sel["rows"]["val"][0], sel["rows"]["val"][1] = sel["rows"]["val"][1], sel["rows"]["val"][0]
        (ev / "selection.json").write_text(json.dumps(sel))
    elif how == "counts":
        cfg.n_val = 7
    elif how == "misassigned":                      # the arms' shards swapped, with the record made to agree
        a, b = cfg.labels("train"), cfg.labels("val")
        a.rename(cfg.cache / "tmp")
        b.rename(a)
        (cfg.cache / "tmp").rename(b)
        rec["labels"][0]["manifest_sha256"], rec["labels"][1]["manifest_sha256"] = \
            rec["labels"][1]["manifest_sha256"], rec["labels"][0]["manifest_sha256"]
        rec["labels"][0]["n"], rec["labels"][1]["n"] = 16, 8
        os.unlink(ev / "record.json")
        (ev / "record.json").write_text(json.dumps(rec))
    elif how == "evidence_exists":
        cfg.evidence_fit.mkdir(parents=True)
    elif how == "student_exists":
        cfg.student.write_bytes(b"x")


@pytest.mark.parametrize("how,match", [
    ("evidence_exists", "already exists"), ("student_exists", "already exists"),
    ("uncommitted", "uncommitted"), ("card", "does not bind"), ("teacher", "does not bind"),
    ("status", "does not bind"), ("selection", "does not bind"), ("counts", "does not bind"),
    ("misassigned", "not the selection's rows"),
])
def test_FIT_REFUSES_UNBOUND_LABELS_OR_A_SPENT_DESTINATION_BEFORE_ANY_REPLAY_LOAD_OR_UPDATE(tmp_path, opened, capsys,
                                                                                           how, match):
    cfg, allowed = _labelled(tmp_path)
    cfg.committed = lambda r, rel: rel in allowed
    games_before = cfg.games.calls
    _break(cfg, allowed, how)
    assert H.main(["run", "fit"], cfg) == 2
    assert match in capsys.readouterr().err
    assert cfg.load_reference.calls == 0 and cfg.games.calls == games_before
    if how != "evidence_exists":
        assert not cfg.evidence_fit.exists()


@pytest.mark.parametrize("raise_,status,code", [
    (lambda: (_ for _ in ()).throw(H.Stop("reference", "fake")), "STOP", 3),
    (lambda: (_ for _ in ()).throw(RuntimeError("boom")), "FAILED", 4),
    (lambda: os.kill(os.getpid(), signal.SIGTERM), "INTERRUPTED", 5),
])
def test_A_STOP_A_FAILURE_OR_A_SIGTERM_INSIDE_THE_FIT_LEAVES_A_TERMINAL_RECORD(tmp_path, opened, raise_, status, code):
    cfg, allowed = _labelled(tmp_path)
    cfg.committed = lambda r, rel: rel in allowed
    before = signal.getsignal(signal.SIGTERM)

    def load(_r):
        raise_()
        return tiny_net(0)

    cfg.load_reference = load
    assert H.run_fit(cfg) == code
    rec = _record(cfg.evidence_fit)
    assert rec["status"] == status and (cfg.evidence_fit / "started.json").is_file()
    if status == "FAILED":
        assert "boom" in rec["error"]["traceback"]
    assert signal.getsignal(signal.SIGTERM) is before and not cfg.student.exists()


def test_A_STOP_INSIDE_LABELLING_LEAVES_A_TERMINAL_RECORD(tmp_path, opened):
    cfg, _ = _cfg(tmp_path)
    cfg.load_teacher = lambda clone: (FakeTeacher(corrupt_at=3), {})
    assert H.run_label("/fake", cfg) == 3
    rec = _record(cfg.evidence_label)
    assert rec["status"] == "STOP" and rec["stop"]["step"] == "lockstep"



# ─────────────────────────────── the 2026-11-04 review: shard reload and memory ───────────────────────────────

def test_A_SHARDS_ARRAYS_ARE_READ_ONCE_NOT_ONCE_PER_ROW(tmp_path, monkeypatch):
    moves = random_game(11, 40)
    rows = _label(moves, set(range(0, 40, 2)))
    H.write_label_shards(rows, tmp_path / "labels", shard_rows=6)
    msha = hashlib.sha256((tmp_path / "labels" / "manifest.json").read_bytes()).hexdigest()
    expected = H.load_label_shards(tmp_path / "labels", msha)
    real, reads = np.load, []

    class Counting:
        def __init__(self, z):
            self.z = z

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.z.close()

        def __getitem__(self, k):
            reads.append(k)
            return self.z[k]

    monkeypatch.setattr(np, "load", lambda *a, **k: Counting(real(*a, **k)))
    got = H.load_label_shards(tmp_path / "labels", msha)
    n_shards = -(-len(rows) // 6)
    assert sorted(reads) == sorted(["target", "value"] * n_shards)       # once per shard, whatever the row count
    assert len(got) == len(expected) == len(rows)
    for a, b in zip(got, expected):
        assert (a["game_id"], a["ply"], a["key"], a["value"]) == (b["game_id"], b["ply"], b["key"], b["value"])
        np.testing.assert_array_equal(a["target576"], b["target576"])


@pytest.mark.parametrize("required,available,match", [(None, 10**12, "not set"), (10**9, 10**8, "below")])
def test_THE_FIT_REFUSES_ON_MEMORY_BEFORE_LOADING_SHARDS_OR_THE_CHECKPOINT(tmp_path, opened, capsys, monkeypatch,
                                                                         required, available, match):
    cfg, allowed = _labelled(tmp_path)
    cfg.committed = lambda r, rel: rel in allowed
    cfg.memory_required_bytes = required
    cfg.available_memory = lambda: {"physical_bytes": 2 * available, "available_bytes": available, "page_size": 1}
    loads = []
    monkeypatch.setattr(H, "load_label_shards", lambda *a: loads.append(a))
    assert H.main(["run", "fit"], cfg) == 2
    assert match in capsys.readouterr().err
    assert not loads and cfg.load_reference.calls == 0 and not cfg.evidence_fit.exists()


def test_THE_REAL_DEFAULT_MEMORY_REQUIREMENT_IS_UNSET_SO_THE_FIT_REFUSES():
    assert H.Config().memory_required_bytes is None
    mem = H.available_memory()
    assert mem["physical_bytes"] > 0 and 0 < mem["available_bytes"] <= mem["physical_bytes"]


@pytest.mark.parametrize("entry", ["label", "fit"])
def test_LABEL_AND_FIT_STOP_ON_A_GAME_THAT_IS_NOT_RED_FIRST(tmp_path, opened, entry):
    """Card §14.4: selection/labelling and example construction both take only checked red-first games."""
    if entry == "label":
        cfg, _ = _cfg(tmp_path)
    else:
        cfg, allowed = _labelled(tmp_path)
        cfg.committed = lambda r, rel: rel in allowed
    games = _fake_games()
    games[0] = {**games[0], "starting_player": "black"}
    cfg.games = _Spy(lambda r: games)
    if entry == "label":
        assert H.run_label("/fake-clone", cfg) == 3
        rec = _record(cfg.evidence_label)
    else:
        assert H.run_fit(cfg) == 3
        rec = _record(cfg.evidence_fit)
    assert rec["status"] == "STOP" and rec["stop"]["step"] == "pool"
