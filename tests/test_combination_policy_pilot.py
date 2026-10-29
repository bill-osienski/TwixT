"""The combination policy pilot's focused CPU checks (card `docs/superpowers/2026-10-28-combination-policy-pilot-card.md`
§6 and §10): fake evaluators, tiny random networks on the CPU, temporary outputs. No checkpoint, no GPU, no real seed
or registry write, no real game against the incumbent."""
import json
import math
import multiprocessing as mp
import pathlib
import queue
import threading

import numpy as np
import pytest

from scripts.GPU.alphazero import combination_policy_pilot as P
from scripts.GPU.alphazero import combination_selfplay as CS
from scripts.GPU.alphazero import halving_policy_target as T
from scripts.GPU.alphazero import mirror_halving_combination as K
from scripts.GPU.alphazero import root_sequential_halving as RSH
from tests import _combination_fakes as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
TINY_ARCH = {"in_channels": 30, "hidden": 128, "n_blocks": 1}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """EVERY test, whatever the gate: the real seams, the real record boundary and the real cache are unreachable,
    and MLX runs on the CPU. A gate-removal mutant reaches only these stand-ins."""
    def unreachable(*_a, **_k):
        raise AssertionError("a test reached a real seam or the real record boundary")
    monkeypatch.setattr(P, "_real_gen_seams", unreachable)
    monkeypatch.setattr(P, "_real_train_seams", unreachable)
    monkeypatch.setattr(P, "conclude", unreachable)
    monkeypatch.setattr(P, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv(P.SETTING, P.SETTING_VALUE)
    import mlx.core as mx
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_THE_ISOLATION_IS_IN_FORCE():
    import mlx.core as mx
    with pytest.raises(AssertionError, match="real seam"):
        P._real_gen_seams(pathlib.Path("/nonexistent"))
    with pytest.raises(AssertionError, match="real seam"):
        P.conclude("time", None, pathlib.Path("/nonexistent"), {})
    assert mx.default_device() == mx.cpu and "twixt-cache" not in str(P.CACHE_DIR)


# ─────────────────────────────── the target (card §4) ───────────────────────────────

def _explicit(prior, cand, q, n_max):
    w = [prior[i] * math.exp((RSH.C_VISIT + n_max) * RSH.C_SCALE * (v + 1) / 2) for i, v in zip(cand, q)]
    pc = sum(prior[i] for i in cand)
    out = list(prior)
    for i, wi in zip(cand, w):
        out[i] = pc * wi / sum(w)
    return np.array(out)


def test_THE_TARGET_SUMS_TO_ONE_KEEPS_UNSEARCHED_MASS_AND_FOLLOWS_HALVING_S_SCORE():
    rng = np.random.default_rng(3)
    prior = rng.random(40)
    prior /= prior.sum()
    cand = list(np.argsort(-prior)[:16])
    q = list(rng.uniform(-0.3, 0.3, 16))
    pi = T.halving_target(prior, cand, q, 49)
    assert abs(math.fsum(pi) - 1.0) < 1e-12
    rest = [i for i in range(40) if i not in cand]
    assert np.array_equal(pi[rest], prior[rest])
    assert np.allclose(pi, _explicit(prior, cand, q, 49), rtol=1e-10, atol=0)


def test_THE_TARGET_IS_EQUIVARIANT_UNDER_A_RELABELLING_OF_MOVES():
    """The mirror maps moves one to one; relabelling the moves (and the candidates with them) relabels the target."""
    rng = np.random.default_rng(4)
    prior = rng.random(30)
    prior /= prior.sum()
    cand, q = [2, 7, 11, 19], [0.1, -0.2, 0.05, 0.3]
    perm = rng.permutation(30)                      # new index of old move i is perm[i]
    p2 = np.empty(30)
    p2[perm] = prior
    pi, pi2 = T.halving_target(prior, cand, q, 49), T.halving_target(p2, [int(perm[i]) for i in cand], q, 49)
    assert np.array_equal(pi2[perm], pi)


@pytest.mark.parametrize("cand, q", [([], []), ([1, 1], [0.0, 0.0]), ([1, 2], [0.0]), ([99], [0.0])],
                         ids=["empty", "repeated", "q-count", "out-of-range"])
def test_THE_TARGET_REFUSES_A_MALFORMED_CANDIDATE_SET(cand, q):
    with pytest.raises(ValueError):
        T.halving_target(np.full(10, 0.1), cand, q, 49)


def _state_at(ply, seed=0):
    import random
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    rng, s = random.Random(seed), TwixtState(active_size=24, to_move="red", max_plies_limit=280)
    for _ in range(ply):
        s = s.apply_move(rng.choice(s.legal_moves()))
    return s


def test_A_ROW_IS_READ_FROM_THE_ROOT_THAT_CHOSE_THE_MOVE():
    import random
    from scripts.GPU.alphazero.eval_runner import cfg_from
    from scripts.GPU.alphazero.mirror_averaging import MirrorAveragingEvaluator
    from scripts.GPU.alphazero.mcts import encode_move
    fake = F.FakeEvaluator()
    avg = MirrorAveragingEvaluator(fake)
    state = _state_at(22)
    m = RSH.RootSequentialHalvingMCTS(avg, cfg_from(CS.protocol_config(), 200), random.Random(5))
    _counts, _v, root = m.search_with_root(state, add_noise=False)
    row = T.search_row(m, root)
    legal = state.legal_moves()
    board = np.transpose(fake.build_input_tensor(state), (1, 2, 0))[None].astype(np.float32)
    r = np.array([[a for a, _ in legal]], np.int32)
    c = np.array([[b for _, b in legal]], np.int32)
    p, _ = avg.infer(board, r, c, np.ones_like(r, np.float32), 24)
    assert np.array_equal(row["prior"], p[0])                      # the averaged prior, not the raw one
    top16 = sorted(range(len(legal)), key=lambda i: (-p[0][i], legal[i]))[:16]
    assert row["cand"] == top16 and row["n_legal"] == len(legal) and row["ply"] == 22
    assert row["q"] == [-root.children[encode_move(*legal[i])].q_value for i in row["cand"]]
    assert sorted(set(row["visits"])) == [3, 9, 21, 49] and row["n_max"] == 49 and row["n_rounds"] == 4
    assert legal[row["choice"]] == m.last_choice and row["rounds"].count(4) == 2
    assert np.array_equal(row["target"], T.halving_target(row["prior"].astype(np.float64), row["cand"], row["q"], 49))


def test_NO_ROW_BEFORE_PLY_20():
    import random
    from scripts.GPU.alphazero.eval_runner import cfg_from
    from scripts.GPU.alphazero.mirror_averaging import MirrorAveragingEvaluator
    seen = []
    cls = CS.capturing(lambda s, root: seen.append(root))
    m = cls(MirrorAveragingEvaluator(F.FakeEvaluator()), cfg_from(CS.protocol_config(), 200), random.Random(1))
    m.search_with_root(_state_at(19), add_noise=False)
    assert seen == [] and m.last_choice is None
    with pytest.raises(ValueError, match="not a halving search"):
        T.search_row(m, None)


# ─────────────────────────────── the agent is the proven agent (card §6) ───────────────────────────────

@pytest.fixture
def short(monkeypatch):
    monkeypatch.setattr(CS, "protocol_config", F.short_config)


def test_A_SELF_PLAY_GAME_IS_THE_MATCH_LOOP_S_GAME_MOVE_FOR_MOVE(short):
    """play_slot == mirror_halving_combination.play_game with the plain halving class on both sides, same seed: the
    capture changes nothing, and the rows are exactly the halving plies, each choosing the move played."""
    from scripts.GPU.alphazero import eval_runner as ER
    from scripts.GPU.alphazero.mirror_averaging import MirrorAveragingEvaluator
    game, rows = CS.play_slot(F.FakeEvaluator(), 4242)
    config = F.short_config()
    rd = ER.readout_from_eval_config(config)
    avg = MirrorAveragingEvaluator(F.FakeEvaluator())
    _w, reason, n, records = K.play_game(avg, avg, config, 4242, capture=True, red_readout=rd, black_readout=rd,
                                         red_sims=200, black_sims=200, red_cls=RSH.RootSequentialHalvingMCTS,
                                         black_cls=RSH.RootSequentialHalvingMCTS)
    assert game["moves"] == [(r["row"], r["col"]) for r in records] and game["n_moves"] == n == F.SHORT_MAX_MOVES
    assert game["row_plies"] == list(range(20, F.SHORT_MAX_MOVES)) and list(rows["ply"]) == game["row_plies"]
    for i, ply in enumerate(game["row_plies"]):
        legal = _replay(game["moves"][:ply]).legal_moves()
        assert legal[rows["choice"][i]] == tuple(game["moves"][ply])


def _replay(moves):
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    s = TwixtState(active_size=24, to_move="red", max_plies_limit=280)
    for m in moves:
        s = s.apply_move(tuple(m))
    return s


class Recording(F.FakeEvaluator):
    def __init__(self):
        super().__init__()
        self.log = []

    def infer(self, boards, rows, cols, mask, active_size):
        self.log.append((boards.copy(), cols.copy(), mask.copy()))
        return super().infer(boards, rows, cols, mask, active_size)


def test_EVERY_LEAF_MAKES_TWO_UNDERLYING_CALLS_THE_SECOND_MIRRORED(short):
    from scripts.GPU.alphazero.self_play import _mirror_position_lr
    ev = Recording()
    CS.play_slot(ev, 77)
    assert len(ev.log) % 2 == 0 and len(ev.log) > 0
    for (b, c, m), (mb, mc, mm) in zip(ev.log[0::2], ev.log[1::2]):
        assert np.array_equal(mm, m) and len(mb) == len(b)
        assert all(np.array_equal(mb[k], _mirror_position_lr(b[k], [], [], 24)[0]) for k in range(len(b)))
        assert np.array_equal(mc, np.where(m > 0, 23 - c, c))


def test_THE_SEED_FIXES_THE_GAME_AND_ANOTHER_SEED_CHANGES_IT(short):
    g1, r1 = CS.play_slot(F.FakeEvaluator(), 9001)
    g2, r2 = CS.play_slot(F.FakeEvaluator(), 9001)
    g3, _ = CS.play_slot(F.FakeEvaluator(), 9002)
    assert g1 == g2 and all(np.array_equal(r1[k], r2[k], equal_nan=True) for k in r1)
    assert g1["moves"] != g3["moves"]


def test_PLAY_SLOT_REFUSES_A_NON_INT_SEED_THE_WRONG_PROTOCOL_AND_OTHER_HALVING_CONSTANTS(monkeypatch):
    import random
    with pytest.raises(ValueError, match="int"):
        CS.play_slot(F.FakeEvaluator(), random.Random(1))
    from scripts.GPU.alphazero import external_learning_stage3 as S3
    monkeypatch.setitem(S3.PROTOCOL, "max_moves", 200)
    with pytest.raises(ValueError, match="280"):
        CS.protocol_config()
    monkeypatch.setitem(S3.PROTOCOL, "max_moves", 280)
    monkeypatch.setattr(RSH, "C_VISIT", 40.0)
    with pytest.raises(ValueError, match="constants"):
        CS.protocol_config()


def test_PACKED_ROWS_REFUSE_MORE_THAN_THE_WIDTH_CAP():
    row = {"ply": 20, "n_legal": 577, "n_max": 49, "choice": 0, "n_rounds": 1, "prior": np.zeros(577),
           "target": np.zeros(577), "cand": [0], "q": [0.0], "visits": [3], "rounds": [1]}
    with pytest.raises(ValueError, match="576"):
        CS.pack_rows([row])


# ─────────────────────────────── the remote path (card §3, §6) ───────────────────────────────

def _local_server(evaluator):
    from scripts.GPU.alphazero.inference_server import InferenceServer
    from scripts.GPU.alphazero.ipc_messages import DEFAULT_MODEL_ID, StopSignal
    req, resp = queue.Queue(), queue.Queue()
    server = InferenceServer({DEFAULT_MODEL_ID: evaluator}, req, {(0, DEFAULT_MODEL_ID): resp}, max_batch_rows=14)
    th = threading.Thread(target=server.run_forever, daemon=True)
    th.start()
    return req, resp, lambda: (req.put(StopSignal()), th.join(5))


def test_MIRROR_AVERAGING_OVER_THE_REMOTE_EVALUATOR_EQUALS_IT_OVER_THE_LOCAL_ONE_AND_NEEDS_THE_SHIM():
    from scripts.GPU.alphazero.mirror_averaging import MirrorAveragingEvaluator
    from scripts.GPU.alphazero.remote_evaluator import RemoteEvaluator
    fake = F.FakeEvaluator()
    req, resp, stop = _local_server(fake)
    try:
        remote = RemoteEvaluator(0, req, resp)
        with pytest.raises(AttributeError):            # the audit's finding: no `network` on the remote evaluator
            MirrorAveragingEvaluator(remote)
        remote.network = None                          # the shim: an instance attribute
        s = _state_at(25, seed=3)
        legal = s.legal_moves()
        board = np.transpose(s.to_tensor(), (1, 2, 0))[None].astype(np.float32)
        r = np.array([[a for a, _ in legal]], np.int32)
        c = np.array([[b for _, b in legal]], np.int32)
        m = np.ones_like(r, np.float32)
        p1, v1 = MirrorAveragingEvaluator(remote).infer(board, r, c, m, 24)
        p2, v2 = MirrorAveragingEvaluator(fake).infer(board, r, c, m, 24)
        assert np.array_equal(p1, p2) and np.array_equal(v1, v2)
    finally:
        stop()


# ─────────────────────────────── seeds independent of scheduling (card §3.4, §6) ───────────────────────────────

def _run_generate(n_workers, n_slots, base, play, evaluator=None):
    starts, games = [], {}
    tele = CS.generate(evaluator or F.FakeEvaluator(), n_slots, base, n_workers, 14,
                       on_start=lambda g, w: starts.append(g),
                       on_game=lambda g, w, game, rows, s: games.__setitem__(g, (game, rows)), play=play, poll_s=0.2)
    return starts, games, tele


def test_THE_SAME_SLOTS_GIVE_THE_SAME_GAMES_WITH_ONE_TWO_OR_THREE_WORKERS():
    runs = {n: _run_generate(n, 7, 1000, F.probe_play) for n in (1, 2, 3)}
    first = {g: v[0] for g, v in runs[1][1].items()}
    for n, (starts, games, tele) in runs.items():
        assert sorted(starts) == list(range(7)) and tele["claims"] == 7 and tele["workers"] == n
        assert tele["started"] == tele["ended"] == list(range(7))
        assert {g: v[0] for g, v in games.items()} == first
        assert all(games[g][0]["seed"] == 1000 + g for g in games)
        assert all(h == {"setting": "0", "mlx_loaded": False} for h in tele["hellos"].values())
        assert tele["server"]["rows"] > 0
    assert not mp.active_children()


@pytest.mark.integration
def test_A_REAL_GAME_THROUGH_THE_ARBITER_EQUALS_THE_SAME_GAME_PLAYED_LOCALLY():
    """OPT-IN (-m integration): 503 s when first run (2026-10-28), because with one request in flight per worker each
    call waits out the unchanged arbiter's 50 ms poll. Spawned workers, the shim and play_slot end to end."""
    _s, games, tele = _run_generate(2, 2, 5000, F.short_play)
    for g in (0, 1):
        local_game, local_rows = F.short_play(F.FakeEvaluator(), 5000 + g)
        game, rows = games[g]
        assert game == local_game and all(np.array_equal(rows[k], local_rows[k], equal_nan=True) for k in rows)
    assert tele["server"]["rows"] >= tele["server"]["batches"] > 0


def test_A_FAILING_WORKER_FAILS_THE_RUN_AND_NO_WORKER_SURVIVES():
    with pytest.raises(CS.GenerationFailed, match="injected failure"):
        _run_generate(2, 4, 0, F.failing_play)
    assert not mp.active_children()


def test_A_WORKER_WITHOUT_THE_SETTING_FAILS_THE_RUN(monkeypatch):
    monkeypatch.setenv(P.SETTING, "1")
    with pytest.raises(CS.GenerationFailed, match="MLX_ENABLE_TF32='1'"):
        _run_generate(1, 1, 0, F.probe_play)
    assert not mp.active_children()


def test_A_FAILURE_IN_THE_CALLER_STOPS_EVERY_WORKER():
    def boom(*_a):
        raise RuntimeError("caller failed")
    with pytest.raises(RuntimeError, match="caller failed"):
        CS.generate(F.FakeEvaluator(), 6, 0, 2, 14, on_start=lambda g, w: None, on_game=boom, play=F.probe_play,
                    poll_s=0.2)
    assert not mp.active_children()


# ─────────────────────────────── the generation entry (card §5 steps 2-3) ───────────────────────────────

def _fake_rows(plies):
    rows = []
    for p in plies:
        prior = np.full(30, 1 / 30)
        cand, q = list(range(16)), [0.01 * i for i in range(16)]
        rows.append({"ply": p, "n_legal": 30, "n_max": 49, "choice": 0, "n_rounds": 4, "prior": prior.astype(np.float32),
                     "target": T.halving_target(prior.astype(np.float32).astype(np.float64), cand, q, 49), "cand": cand,
                     "q": q, "visits": [3] * 16, "rounds": [1] * 8 + [2] * 4 + [3] * 2 + [4] * 2})
    return CS.pack_rows(rows)


class GenHarness:
    def __init__(self, tmp_path, kind="generate", base=5000, n=None, bg=None, seed_shift=0):
        self.kind, self.base = kind, base
        self.n = n if n is not None else P.KINDS[kind]["slots"]
        self.bg_calls = 0
        self.bg = bg or (lambda i: {"ok": True, "n_procs": 5, "hits": [], "exempt": [], "error": None})
        self.seed_shift = seed_shift
        self.seams = P.GenSeams(
            versions=lambda: dict(P.VERSIONS), read_plan=lambda rel: {"base": base, "n_slots": self.n},
            seed_ok=lambda s: True, checkpoint_sha1=lambda: P.M.CHECKPOINT_SHA1,
            load_evaluator=lambda: F.FakeEvaluator(), require_eval=lambda ev, when: {"when": when},
            loaded_modules=lambda: {"x.py": "0"}, background=self.background, generate=self.generate,
            rows_dir=tmp_path / "cache")

    def background(self):
        self.bg_calls += 1
        return self.bg(self.bg_calls)

    def generate(self, *, evaluator, n_slots, slot_base, n_workers, max_batch_rows, on_start, on_game):
        assert (n_workers, max_batch_rows) == (18, 14)
        for g in range(n_slots):
            on_start(g, g % 3)
            game = {"seed": slot_base + g + self.seed_shift, "moves": [[1, g % 20]] * 22, "winner": "red",
                    "reason": "win", "n_moves": 22, "n_rows": 2, "row_plies": [20, 21]}
            on_game(g, g % 3, game, _fake_rows([20, 21]), 1.5)
        return {"claims": n_slots, "started": list(range(n_slots)), "ended": list(range(n_slots)),
                "hellos": {0: {"setting": "0", "mlx_loaded": False}}, "workers": 18,
                "server": {"requests": 10, "rows": 100, "batches": 10}, "max_batch_rows": 14}


_REAL_CONCLUDE = P.conclude            # bound at import, before the isolation replaces the module's name


def _conclude(kind, execute, out):
    """The real record boundary, with test seams and a temporary output directory."""
    return _REAL_CONCLUDE(kind, execute, out, {})


def test_GENERATION_WRITES_EACH_GAME_ITS_ROWS_AND_THE_CLOSING_EVENTS(tmp_path, monkeypatch):
    monkeypatch.setitem(P.KINDS["generate"], "slots", 4)
    h = GenHarness(tmp_path, n=4)
    code, rec = _conclude("generate", lambda o, r, lg: P._generate("generate", h.seams, o, r, lg), tmp_path / "ev")
    assert code == P.EXIT_COMPLETED and rec["status"] == "COMPLETED", rec.get("void")
    games = [json.loads(x) for x in (tmp_path / "ev" / "games.jsonl").read_text().splitlines()]
    assert [g["slot"] for g in games] == [0, 1, 2, 3] and all(g["seed"] == 5000 + g["slot"] for g in games)
    for g in games:
        path = tmp_path / "cache" / "generate" / g["rows_file"]
        assert P._sha_file(path) == g["rows_sha256"]
    events = [json.loads(x) for x in (tmp_path / "ev" / "progress.jsonl").read_text().splitlines()]
    names = [e["event"] for e in events]
    for s in range(4):
        start = next(i for i, e in enumerate(events) if e["event"] == "game_start" and e["slot"] == s)
        end = next(i for i, e in enumerate(events) if e["event"] == "game_end" and e["slot"] == s)
        assert start < end
    assert names.count("claims") == 1 and h.bg_calls == rec["background_checks"] == 4 + 2
    assert rec["summary"]["rows"] == 8 and rec["summary"]["duplicate_rows"] == 0
    assert rec["throughput"]["mean_rows_per_call"] == 10.0 and "generate_ceiling" not in rec
    assert 0.0 < rec["summary"]["mean_mass_round1_out"] < 1.0


def test_THE_THROUGHPUT_CHECK_KEEPS_NO_ROWS_AND_PRICES_GENERATION(tmp_path, monkeypatch):
    monkeypatch.setitem(P.KINDS["throughput"], "slots", 3)
    h = GenHarness(tmp_path, kind="throughput", n=3)
    code, rec = _conclude("throughput", lambda o, r, lg: P._generate("throughput", h.seams, o, r, lg), tmp_path / "e")
    assert code == P.EXIT_COMPLETED, rec.get("void")
    assert not (tmp_path / "cache").exists()
    games = [json.loads(x) for x in (tmp_path / "e" / "games.jsonl").read_text().splitlines()]
    assert all("rows_file" not in g for g in games)
    c = rec["generate_ceiling"]
    assert c["ceiling_seconds"] == min(math.ceil(1.5 * c["projected_seconds"]), 6 * 3600)


@pytest.mark.parametrize("fault, step", [
    ("boinc", "background_compute"), ("seed", "seeds"), ("sha1", "checkpoint"), ("plan", "seeds"),
    ("versions", "versions")])
def test_GENERATION_IS_VOID_ON_EACH_INTEGRITY_FAULT(tmp_path, monkeypatch, fault, step):
    monkeypatch.setitem(P.KINDS["generate"], "slots", 3)
    bg = (lambda i: {"ok": i < 3, "n_procs": 5, "hits": ["1 boinc"] if i >= 3 else [], "exempt": [], "error": None})
    h = GenHarness(tmp_path, n=3, bg=bg if fault == "boinc" else None, seed_shift=1 if fault == "seed" else 0)
    if fault == "sha1":
        h.seams.checkpoint_sha1 = lambda: "0" * 40
    if fault == "plan":
        h.seams.read_plan = lambda rel: {"base": 5000, "n_slots": 2}
    if fault == "versions":
        h.seams.versions = lambda: {**P.VERSIONS, "mlx": "0.0"}
    code, rec = _conclude("generate", lambda o, r, lg: P._generate("generate", h.seams, o, r, lg), tmp_path / "ev")
    assert code == P.EXIT_VOID and rec["status"] == "VOID" and rec["void"]["step"] == step
    assert (tmp_path / "ev" / "record.json").is_file()


def test_AN_INCOMPLETE_GENERATION_IS_VOID(tmp_path, monkeypatch):
    monkeypatch.setitem(P.KINDS["generate"], "slots", 3)
    h = GenHarness(tmp_path, n=3)
    real = h.generate

    def short(**kw):
        out = real(**kw)
        return {**out, "claims": 2}
    h.seams.generate = short
    code, rec = _conclude("generate", lambda o, r, lg: P._generate("generate", h.seams, o, r, lg), tmp_path / "ev")
    assert rec["status"] == "VOID" and rec["void"]["step"] == "games"


# ─────────────────────────────── the objective and training (card §4, §5 step 4) ───────────────────────────────

def _tiny_network(seed=1):
    import mlx.core as mx
    from scripts.GPU.alphazero.network import create_network
    mx.random.seed(seed)
    net = create_network(hidden=128, n_blocks=1, in_channels=30)
    net.eval()
    return net


@pytest.fixture(scope="module")
def corpus_game():
    game, rows = F.short_play(F.FakeEvaluator(), 31337)
    return json.loads(json.dumps(game)), rows


def _train_seams(corpus_game, arch=TINY_ARCH, mutate=None):
    game, rows = corpus_game
    games = [{"slot": 0, **game}, {"slot": 9, **game}]      # slot 9 is held out: the same positions, unseen in fitting
    stored = {0: dict(rows), 9: dict(rows)}
    if mutate:
        mutate(stored)
    return P.TrainSeams(versions=lambda: dict(P.VERSIONS), read_record=lambda rel: {"arch": dict(arch)},
                        read_games=lambda: games, read_rows=lambda g: stored[g["slot"]],
                        checkpoint_sha1=lambda: P.M.CHECKPOINT_SHA1,
                        load_network=lambda: (_tiny_network(), dict(TINY_ARCH)), loaded_modules=lambda: {})


def _recipe(**kw):
    return {**P.RECIPE, "arch": dict(TINY_ARCH), "updates": 12, "batch": 3, "lr": 1e-2, "min_kl": 0.0, **kw}


def _train(tmp_path, seams, recipe):
    return _conclude("train", lambda o, r, lg: P._train(recipe, seams, o, r, lg), tmp_path / "train")


def test_TRAINING_LEARNS_THE_POLICY_AND_LEAVES_THE_VALUE_BITWISE_UNCHANGED(tmp_path, corpus_game):
    code, rec = _train(tmp_path, _train_seams(corpus_game), _recipe())
    assert code == P.EXIT_COMPLETED, rec.get("void")
    assert rec["verdict"] == "ELIGIBLE", rec.get("stop")
    assert rec["device"].startswith("Device(cpu") and rec["corpus"]["train_rows"] == rec["corpus"]["heldout_rows"] == 6
    assert rec["value_check"]["bitwise_equal"] and rec["value_check"]["max_abs_diff"] == 0.0
    assert rec["digests_after"]["encoder."] == rec["digests_before"]["encoder."]
    assert rec["digests_after"]["value_head."] == rec["digests_before"]["value_head."]
    assert rec["digests_after"]["policy_head."] != rec["digests_before"]["policy_head."]
    assert rec["heldout_after"]["ce"] < rec["prefit_heldout"]["ce"] and len(rec["losses"]) == 12
    assert rec["surface"]["parameters"] == 8840
    with np.load(tmp_path / "train" / "head.npz") as z:
        assert "policy_head.extra_out.weight" in z.files
    assert P._sha_file(tmp_path / "train" / "head.npz") == rec["head_sha256"]


def test_THE_PREFIT_KL_STOP_FIRES_BEFORE_ANY_FITTING(tmp_path, corpus_game):
    code, rec = _train(tmp_path, _train_seams(corpus_game), _recipe(min_kl=1e9))
    assert code == P.EXIT_COMPLETED and rec["verdict"] == "STOP" and rec["stop"]["rule"] == "prefit_kl"
    assert "losses" not in rec and not (tmp_path / "train" / "head.npz").exists()


def test_THE_HELDOUT_STOP_FIRES_WHEN_NOTHING_IS_LEARNED(tmp_path, corpus_game):
    code, rec = _train(tmp_path, _train_seams(corpus_game), _recipe(lr=0.0, l2=0.0))
    assert rec["verdict"] == "STOP" and rec["stop"]["rule"] == "heldout_not_improved"
    assert rec["value_check"]["bitwise_equal"]


def test_A_NON_FINITE_LOSS_STOPS(tmp_path, corpus_game):
    code, rec = _train(tmp_path, _train_seams(corpus_game), _recipe(l2=float("inf")))
    assert rec["verdict"] == "STOP" and rec["stop"]["rule"] == "nonfinite"


def test_A_CHANGED_ARRAY_OUTSIDE_THE_SURFACE_IS_CAUGHT():
    before = {"encoder.w": np.zeros(2), "policy_head.bn.running_mean": np.zeros(2), "policy_head.fc.weight": np.zeros(2)}
    surface = ["policy_head.fc.weight", "policy_head.bn.running_mean"]
    assert P.changed_outside(before, {**before, "policy_head.fc.weight": np.ones(2)}, surface) == []
    assert P.changed_outside(before, {**before, "encoder.w": np.ones(2)}, surface) == ["encoder.w"]
    assert P.changed_outside(before, {**before, "policy_head.bn.running_mean": np.ones(2)}, surface) == \
        ["policy_head.bn.running_mean"]


@pytest.mark.parametrize("what", ["target", "prior", "n_legal"])
def test_A_CORRUPT_CORPUS_IS_VOID(tmp_path, corpus_game, what):
    def mutate(stored):
        rows = {k: v.copy() for k, v in stored[9].items()}
        if what == "target":
            rows["target"][0, 0] += 1e-9
        elif what == "prior":
            rows["prior"][1, 2] *= 1.5
        else:
            rows["n_legal"][0] -= 1
        stored[9] = rows
    code, rec = _train(tmp_path, _train_seams(corpus_game, mutate=mutate), _recipe())
    assert code == P.EXIT_VOID and rec["void"]["step"] == "corpus"


def test_TRAINING_REFUSES_AN_ARCHITECTURE_OTHER_THAN_THE_TIMED_ONE(tmp_path, corpus_game):
    code, rec = _train(tmp_path, _train_seams(corpus_game, arch={**TINY_ARCH, "n_blocks": 6}), _recipe())
    assert rec["status"] == "VOID" and rec["void"]["step"] == "timing"


def test_THE_OBJECTIVE_IS_THE_AGENT_S_AVERAGED_PRIOR_AND_IS_MIRROR_INVARIANT(corpus_game):
    """exp(log p_bar) equals MirrorAveragingEvaluator's prior over the same network on the CPU; a row and its mirror
    with the mirrored target give the same loss."""
    import mlx.core as mx
    from scripts.GPU.alphazero import policy_head_capacity as PHC
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.mirror_averaging import MirrorAveragingEvaluator
    game, rows = corpus_game
    data = P.build_corpus([{"slot": 0, **game}], lambda g: rows, 20)
    net = _tiny_network()
    PHC.install(net, "R", init_seed=5)
    net.policy_head.extra_out.weight = mx.array(np.random.default_rng(2).normal(0, 0.1, (1, 64)).astype(np.float32))
    idx = np.arange(len(data["slot"]))
    lp = np.asarray(P.log_averaged(net.policy_head, *P.batch_inputs(net, data, idx)))
    avg = MirrorAveragingEvaluator(LocalGPUEvaluator(net))
    for i in idx:
        n = int(data["n_legal"][i])
        p, _ = avg.infer(data["boards"][i:i + 1], data["rows"][i:i + 1, :n], data["cols"][i:i + 1, :n],
                         data["mask"][i:i + 1, :n], 24)
        assert np.allclose(np.exp(lp[i, :n]), p[0], rtol=0, atol=2e-6)
    mb, mc = P.mirrored(data["boards"], data["cols"], data["mask"])
    flip = {**data, "boards": mb, "cols": mc}
    lpm = np.asarray(P.log_averaged(net.policy_head, *P.batch_inputs(net, flip, idx)))
    for i in idx:
        n = int(data["n_legal"][i])
        # the mirrored row's move j is the mirror of the original move j, so the targets align index for index
        ce = -np.sum(data["target64"][i, :n] * lp[i, :n])
        cem = -np.sum(data["target64"][i, :n] * lpm[i, :n])
        assert abs(ce - cem) < 1e-4


def test_THE_DRAW_ORDER_IS_FRESH_PERMUTATIONS_FROM_ITS_SEED():
    a, b = P.draw_order(10, 7, 4, 20261029), P.draw_order(10, 7, 4, 20261029)
    assert np.array_equal(a, b) and a.shape == (7, 4)
    flat = a.ravel()
    assert sorted(flat[:10]) == list(range(10)) and sorted(flat[10:20]) == list(range(10))
    assert not np.array_equal(a, P.draw_order(10, 7, 4, 20261028))


def test_THE_R_INSTALL_STARTS_BIT_EQUAL_TO_THE_PARENT(corpus_game):
    from scripts.GPU.alphazero import policy_head_capacity as PHC
    game, rows = corpus_game
    data = P.build_corpus([{"slot": 0, **game}], lambda g: rows, 20)
    net = _tiny_network()
    idx = np.arange(len(data["slot"]))
    before = np.array(net.policy_head.forward_padded(*P.batch_inputs(net, data, idx)[0]))
    PHC.install(net, "R", init_seed=P.RECIPE["init_seed"])
    assert np.array_equal(np.array(net.policy_head.forward_padded(*P.batch_inputs(net, data, idx)[0])), before)
    assert PHC.residual_sha256(PHC.residual_arrays(net.policy_head)) == \
        PHC.residual_sha256(PHC.residual_init(20261028))


# ─────────────────────────────── the build's timing (card §5 step 1) ───────────────────────────────

def test_THE_TIMING_SYNCHRONIZES_EACH_UPDATE_AND_PRICES_THE_CEILING(tmp_path):
    spec = {**P.TIMING, "updates": 2, "warmup": 1, "forward_batches": 1, "tensor_calls": 3}
    recipe = {**P.RECIPE, "arch": dict(TINY_ARCH), "batch": 4}
    code, rec = _conclude("time", lambda o, r, lg: P._timing(spec, recipe, o, r, lg), tmp_path / "t")
    assert code == P.EXIT_COMPLETED, rec.get("void")
    t, pr = rec["timing"], rec["projection"]
    assert len(t["update_seconds"]) == 2 and all(s > 0 for s in t["update_seconds"])
    assert pr["updates_seconds"] == pytest.approx(t["mean_update_seconds"] * recipe["updates"])
    assert pr["ceiling_seconds"] == math.ceil(1.5 * pr["total_seconds"] / 60) * 60 and pr["ceiling_seconds"] % 60 == 0
    assert rec["device"].startswith("Device(cpu") and "no checkpoint" in rec["weights"]


# ─────────────────────────────── the entry, the gates and the default path ───────────────────────────────

def test_ALL_THREE_GATES_ARE_CLOSED_SINGLE_DECLARATIONS():
    import ast
    for kind in ("throughput", "generate", "train"):
        gate, name = P.KINDS[kind]["gate"]
        assert getattr(gate, name) is False
        tree = ast.parse(pathlib.Path(gate.__file__).read_text())
        body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        assert len(body) == 1 and body[0].targets[0].id == name


def test_THE_ENTRY_REFUSES_WITHOUT_THE_SETTING_AND_ON_A_CLOSED_GATE(monkeypatch, capsys):
    monkeypatch.setenv(P.SETTING, "1")
    assert P.main(["time"]) == P.EXIT_REFUSED and "MLX_ENABLE_TF32" in capsys.readouterr().err
    monkeypatch.setenv(P.SETTING, "0")
    monkeypatch.setattr(P, "_mlx_loaded", lambda: False)
    monkeypatch.setattr(P, "CARD_SHA256", P.M._sha((ROOT / P.CARD).read_bytes()))
    for kind in ("throughput", "generate", "train"):
        assert P.main(["run", kind, "--checkpoint", "/x"]) == P.EXIT_REFUSED
        assert "is closed" in capsys.readouterr().err


def test_THE_ENTRY_REFUSES_WITH_MLX_ALREADY_LOADED_AND_A_CHANGED_CARD_OR_MODULE(monkeypatch, capsys):
    assert P.main(["time"]) == P.EXIT_REFUSED and "MLX was imported" in capsys.readouterr().err  # MLX is loaded here
    monkeypatch.setattr(P, "_mlx_loaded", lambda: False)
    monkeypatch.setattr(P, "CARD_SHA256", "0" * 64)
    assert P.main(["time"]) == P.EXIT_REFUSED and "pinned card" in capsys.readouterr().err
    monkeypatch.setattr(P, "CARD_SHA256", P.M._sha((ROOT / P.CARD).read_bytes()))
    monkeypatch.setitem(P.PINNED, "scripts/GPU/alphazero/mirror_averaging.py", "0" * 64)
    assert P.main(["time"]) == P.EXIT_REFUSED and "pinned bytes" in capsys.readouterr().err


def test_THE_CARD_AND_THE_FROZEN_MODULES_ARE_THE_PINNED_BYTES():
    assert P.M._sha((ROOT / P.CARD).read_bytes()) == P.CARD_SHA256
    for rel, sha in P.PINNED.items():
        assert P.M._sha((ROOT / rel).read_bytes()) == sha, rel


def test_ADMISSION_NEEDS_EACH_EARLIER_COMMITTED_RECORD():
    ok = {"status": "COMPLETED", "card_sha256": P.CARD_SHA256}

    def reader(have):
        def read(rel):
            if rel not in have:
                raise FileNotFoundError(rel)
            return have[rel]
        return read
    with pytest.raises(P.Refused, match="does not exist"):
        P.admitted("throughput", reader({}))
    assert P.admitted("throughput", reader({P.TIMING_EVIDENCE: ok})) == {P.TIMING_EVIDENCE: "COMPLETED"}
    with pytest.raises(P.Refused, match="not COMPLETED"):
        P.admitted("generate", reader({P.KINDS["throughput"]["evidence"]: {**ok, "status": "VOID"}}))
    with pytest.raises(P.Refused):
        P.admitted("train", reader({P.KINDS["generate"]["evidence"]: ok}))


THROUGHPUT_RUN = str(pathlib.Path(P.KINDS["throughput"]["plan"]).parent)
GOOD_OUTER = {"inner_exit": 0, "outcome": "COMPLETED", "outer_exit": 0, "gate_restored": True, "survivors": "",
              "record_by": "script", "head": "a" * 40}


def _gen_readers(inner, outer):
    """Readers for generation's admission: the throughput record, and its outer.json (None: absent)."""
    def read_record(rel):
        assert rel == P.KINDS["throughput"]["evidence"], rel
        return inner

    def read_outer(rel):
        assert rel == THROUGHPUT_RUN, rel
        if outer is None:
            raise FileNotFoundError(rel)
        return outer
    return read_record, read_outer


def test_GENERATION_IS_ADMITTED_ONLY_WITH_A_SUCCESSFUL_COMMITTED_OUTER_JSON():
    """The user's ruling (2026-10-29): the inner record AND the wrapper's outer.json must both show success."""
    inner = {"status": "COMPLETED", "card_sha256": P.CARD_SHA256, "head": "a" * 40}
    assert P.admitted("generate", *_gen_readers(inner, GOOD_OUTER)) == \
        {P.KINDS["throughput"]["evidence"]: "COMPLETED", f"{THROUGHPUT_RUN}/outer.json": "COMPLETED"}
    with pytest.raises(P.Refused, match="outer.json does not exist"):           # missing outer evidence
        P.admitted("generate", *_gen_readers(inner, None))
    with pytest.raises(P.Refused, match="outer.json does not exist"):           # no outer reader at all
        P.admitted("generate", _gen_readers(inner, GOOD_OUTER)[0])


@pytest.mark.parametrize("change", [
    {"inner_exit": 124, "outcome": "TIMEOUT", "outer_exit": 124},                  # TERM sufficed
    {"inner_exit": 137, "outcome": "TIMEOUT", "outer_exit": 137, "record_by": "script"},   # KILL needed
    {"inner_exit": 125, "outcome": "TIMEOUT", "outer_exit": 125,                   # the supervisor gave up
     "survivors": "unconfirmed cleanup: process group 4242 "},
    {"survivors": "unconfirmed cleanup: process group 4242 4243 "},                # the group remained
    {"survivors": "unknown: no readable pgid.txt "},
    {"survivors": "4243 "},
    {"survivors": None},
    {"gate_restored": False, "outer_exit": 5},
    {"outer_exit": 5},
    {"record_by": "wrapper"}, {"record_by": "none"},
    {"inner_exit": False}, {"inner_exit": "0"}, {"inner_exit": None},
    {"outcome": "VOID", "inner_exit": 3, "outer_exit": 3},
    {"head": "b" * 40}, {"head": None}],
    ids=["timeout-124", "timeout-137", "unconfirmed-125", "group-remains", "unknown", "survivor", "no-survivors-field",
         "gate-not-restored", "outer-exit-5", "wrapper-record", "no-record", "exit-bool", "exit-str", "exit-null",
         "void", "other-head", "no-head"])
def test_AN_OUTER_FAILURE_REFUSES_GENERATION_EVEN_OVER_A_COMPLETED_INNER_RECORD(change):
    inner = {"status": "COMPLETED", "card_sha256": P.CARD_SHA256, "head": "a" * 40}
    with pytest.raises(P.Refused, match="does not show a successful run"):
        P.admitted("generate", *_gen_readers(inner, {**GOOD_OUTER, **change}))


def test_ONLY_GENERATION_READS_AN_OUTER_JSON():
    ok = {"status": "COMPLETED", "card_sha256": P.CARD_SHA256}
    assert P.admitted("throughput", lambda rel: ok) == {P.TIMING_EVIDENCE: "COMPLETED"}
    assert set(P.OUTER_ADMISSION) == {"generate"}


@pytest.mark.parametrize("state", ["committed", "absent", "untracked", "ignored", "modified"])
def test_THE_COMMITTED_READER_needs_the_file_tracked_and_unchanged(tmp_path, monkeypatch, state):
    import subprocess

    def git(*a):
        return subprocess.check_output(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=tmp_path,
                                       text=True).strip()
    git("init", "-q")
    (tmp_path / ".gitignore").write_text("ignored/\n")
    git("add", "-A")
    git("commit", "-qm", "base")
    rel = {"ignored": "ignored/outer.json"}.get(state, "run/outer.json")
    path = tmp_path / rel
    if state != "absent":
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(GOOD_OUTER))
    if state in ("committed", "modified"):
        git("add", rel)
        git("commit", "-qm", "outer")
    if state == "modified":
        path.write_text(json.dumps({**GOOD_OUTER, "outcome": "TIMEOUT"}))
    monkeypatch.setattr(P, "ROOT", tmp_path)
    monkeypatch.setattr(P.M, "_git", git)
    if state == "committed":
        assert P._read_outer(str(pathlib.Path(rel).parent)) == GOOD_OUTER
    elif state == "absent":
        with pytest.raises(FileNotFoundError):
            P._read_outer("run")
    else:
        with pytest.raises(P.Refused, match="uncommitted or differs from HEAD"):
            P._read_outer(str(pathlib.Path(rel).parent))


def test_USAGE():
    assert P.main([]) == P.main(["run", "screen", "--checkpoint", "/x"]) == P.main(["time", "now"]) == P.EXIT_USAGE


def test_THE_FROZEN_RECIPE_IS_THE_CARD_S():
    r = P.RECIPE
    assert (r["updates"], r["batch"], r["lr"], r["l2"], r["init_seed"], r["order_seed"], r["min_kl"]) == \
        (3000, 64, 1e-3, 1e-4, 20261028, 20261029, 0.01)
    assert (r["heldout_mod"], r["heldout_rem"], r["from_ply"]) == (10, 9, 20)
    assert (P.N_WORKERS, P.MAX_BATCH_ROWS) == (18, 14)
    assert {k: v["slots"] for k, v in P.KINDS.items() if "slots" in v} == {"throughput": 144, "generate": 1000}
    assert P.GENERATE_CEILING == {"factor": 1.5, "max_seconds": 21600}


def test_DEFAULT_SELF_PLAY_IS_UNTOUCHED():
    """§10.1: the combination path sits beside self-play; the default modules neither import nor mention it."""
    for rel in ("self_play.py", "self_play_worker.py", "trainer.py", "mcts.py"):
        text = (ROOT / "scripts/GPU/alphazero" / rel).read_text()
        assert "combination" not in text, rel


def test_IMPORTING_THE_PILOT_LOADS_NO_MLX():
    import subprocess
    import sys
    code = ("import sys; import scripts.GPU.alphazero.combination_policy_pilot, "
            "scripts.GPU.alphazero.combination_selfplay, scripts.GPU.alphazero.halving_policy_target; "
            "sys.exit(any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules))")
    assert subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT).returncode == 0


def _bg(ok):
    return {"ok": ok, "n_procs": 9, "hits": [] if ok else ["2090 boinc_project hsgamma"], "n_hits": 0 if ok else 1,
            "exempt": [], "error": None}


def test_THE_TIMING_REFUSES_WHILE_BOINC_RUNS_AND_CONSUMES_NOTHING(monkeypatch, tmp_path, capsys):
    from scripts.GPU.alphazero import background_compute as BG
    monkeypatch.setattr(P, "_preflight", lambda: None)
    monkeypatch.setattr(P, "ROOT", tmp_path)
    monkeypatch.setattr(BG, "observe", lambda: _bg(False))
    assert P.main(["time"]) == P.EXIT_REFUSED and "quit BOINC" in capsys.readouterr().err
    assert not (tmp_path / P.TIMING_EVIDENCE).exists()


def test_THE_TIMING_IS_VOID_IF_BOINC_APPEARS_BY_THE_END(monkeypatch, tmp_path):
    from scripts.GPU.alphazero import background_compute as BG
    seen = iter([_bg(True), _bg(False)])
    monkeypatch.setattr(P, "_preflight", lambda: None)
    monkeypatch.setattr(P, "ROOT", tmp_path)
    monkeypatch.setattr(P, "_timing", lambda *a: None)
    monkeypatch.setattr(P, "conclude", _REAL_CONCLUDE)
    monkeypatch.setattr(BG, "observe", lambda: next(seen))
    assert P.main(["time"]) == P.EXIT_VOID
    rec = json.loads((tmp_path / P.TIMING_EVIDENCE / "record.json").read_text())
    assert rec["status"] == "VOID" and rec["void"]["step"] == "background_compute"
    assert rec["background_before"]["ok"] and not rec["background_after"]["ok"]
