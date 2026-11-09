"""Fixtures for the H-TB integration's G1 checks (integration proposal §7, V1 and V4): the research code driven by a
deterministic FAKE evaluator -- no model, no MLX, no checkpoint.

    .venv/bin/python tests/fixtures/htb_integration/generate_fixtures.py

Writes rng.json, mirror.json and search.json next to this file. Each records the sha256 of every research source
file it exercises, so the JS tests can refuse fixtures that no longer describe the research code.

The fake evaluator uses only integer arithmetic, one true division and float32 rounding, so its JS twin
(fake_evaluator.mjs) reproduces it bit for bit. Its priors are built to tie: equal weights give exact ties, and the
(j % 3) * 4e-9 offsets give scores within the 1e-8 tie tolerance, including order-dependent running-best sets.
Game states carry no 280-ply limit (max_plies_limit=None): the product's game-end rule, which the port serves.
"""
import hashlib
import json
import pathlib
import platform
import random
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from scripts.GPU.alphazero import eval_readout  # noqa: E402
from scripts.GPU.alphazero.eval_runner import EvalConfig, cfg_from, readout_from_eval_config  # noqa: E402
from scripts.GPU.alphazero.external_learning_stage3 import PROTOCOL  # noqa: E402
from scripts.GPU.alphazero.game.twixt_state import TwixtState  # noqa: E402
from scripts.GPU.alphazero.mcts import MCTS, encode_move  # noqa: E402
from scripts.GPU.alphazero.mirror_averaging import MirrorAveragingEvaluator  # noqa: E402
from scripts.GPU.alphazero.root_sequential_halving import RootSequentialHalvingMCTS  # noqa: E402
from scripts.GPU.alphazero.self_play import _mirror_position_lr  # noqa: E402

RESEARCH_SOURCES = [
    "scripts/GPU/alphazero/mcts.py",
    "scripts/GPU/alphazero/root_sequential_halving.py",
    "scripts/GPU/alphazero/mirror_averaging.py",
    "scripts/GPU/alphazero/self_play.py",
    "scripts/GPU/alphazero/eval_readout.py",
    "scripts/GPU/alphazero/eval_runner.py",
    "scripts/GPU/alphazero/external_learning_stage3.py",
    "scripts/GPU/alphazero/game/twixt_state.py",
    "tests/fixtures/htb_integration/generate_fixtures.py",
]

TIE_EPS = 1e-8
IDX = (np.arange(576, dtype=np.int64) + 1).reshape(24, 24)


def provenance():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sources_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in RESEARCH_SOURCES},
    }


# --- the fake evaluator (twin: fake_evaluator.mjs) --------------------------------------------------------------

def board_hash(b):
    """b: (24, 24, 30) float32. Integer sums only."""
    h = int((IDX * 3 * (b[:, :, 0] > 0.5)).sum()) + int((IDX * 5 * (b[:, :, 1] > 0.5)).sum())
    for ch in range(2, 18):
        h += int((IDX * (ch + 7) * (b[:, :, ch] > 0.5)).sum())
    return h % 100003


def fake_row(b, moves):
    h = board_hash(b)
    w = [((int(r) * 31 + int(c) * 17 + h) % 97) + 1 for r, c in moves]
    s = sum(w)
    priors = [np.float32(w[j] / s + (j % 3) * 4e-9) for j in range(len(moves))]
    value = np.float32(((((h * 37 + len(moves)) % 201) - 100)) / 128)
    return priors, value


class FakeEvaluator:
    network = "fake-network"

    def build_input_tensor(self, state):
        return state.to_tensor()

    def infer(self, boards, rows, cols, mask, active_size):
        B, Mw = rows.shape
        priors = np.zeros((B, Mw), np.float32)
        values = np.zeros(B, np.float32)
        for i in range(B):
            n = int(mask[i].sum())
            p, v = fake_row(boards[i], list(zip(rows[i, :n], cols[i, :n])))
            priors[i, :n] = p
            values[i] = v
        return priors, values


# --- tie statistics: evidence that the fixtures exercise the tie rule (does not touch the stream) -----------------

class TieStatsMixin:
    def _select_child(self, node, pending_node_ids=None):
        c = self.config.c_puct
        sqrt_parent = np.sqrt(node.visit_count + 1)
        scores = []
        for move_id, prior in node.priors.items():
            child = node.children.get(move_id)
            if child is not None and child.visit_count > 0:
                q, n = -child.q_value, child.visit_count
            else:
                q, n = self.config.fpu_value, 0
            scores.append(q + c * prior * float(sqrt_parent) / (1 + n))
        best, running = float("-inf"), []
        for s in scores:
            if s > best + TIE_EPS:
                best, running = s, [s]
            elif abs(s - best) <= TIE_EPS:
                running.append(s)
        top = max(scores)
        st = self.tie_stats
        st["selections"] += 1
        if len(running) > 1:
            st["tie_sets"] += 1
            if len({x for x in running}) < len(running):
                st["exact_ties"] += 1
            if max(running) != min(running):
                st["near_ties"] += 1
        if len(running) != sum(1 for s in scores if abs(s - top) <= TIE_EPS) or (running and max(running) != top):
            st["order_dependent"] += 1
        return super()._select_child(node, pending_node_ids)


def new_stats():
    return {"selections": 0, "tie_sets": 0, "exact_ties": 0, "near_ties": 0, "order_dependent": 0}


class TMCTS(TieStatsMixin, MCTS):
    pass


class THalving(TieStatsMixin, RootSequentialHalvingMCTS):
    pass


CONFIG = EvalConfig(**PROTOCOL)
READOUT = readout_from_eval_config(CONFIG)


def key(m):
    return f"{int(m[0])},{int(m[1])}"


def state_from(moves):
    s = TwixtState(active_size=24, to_move="red")
    for m in moves:
        s = s.apply_move(tuple(m))
    return s


# --- RNG ----------------------------------------------------------------------------------------------------------

def rng_fixture():
    cases = []
    for seed in [0, 1, 7, 202759800 ^ 0xA5A5A5, 2**32 - 1, 2**32 + 7, 2**48 - 2]:
        r = random.Random(seed)
        rand = [r.random() for _ in range(5)]
        bits = [r.getrandbits(k) for k in (0, 1, 2, 5, 10, 31, 32)]
        choices = [r.choice(list(range(n))) for n in (1, 2, 3, 7, 576, 577)]
        x = list(range(100))
        r.shuffle(x)
        tail = [r.random() for _ in range(700)]          # crosses a 624-word regeneration
        cases.append({"seed": seed, "random": rand, "getrandbits": bits, "choice": choices, "shuffle": x,
                      "tail_sum_hex": float(sum(tail)).hex(), "tail_last": tail[-1]})
    return {"provenance": provenance(), "cases": cases}


# --- V1: mirror ---------------------------------------------------------------------------------------------------

def corpus_positions():
    d = json.loads((ROOT / "tests/parity/corpus.json").read_text())
    return [p for p in d["primary"] + d["edge"] if not p.get("terminal")]


def hwc(state):
    return np.ascontiguousarray(np.transpose(state.to_tensor(), (1, 2, 0)).astype(np.float32))


def mirror_fixture():
    fake = FakeEvaluator()
    mirror = MirrorAveragingEvaluator(fake)
    out = []
    for i, p in enumerate(corpus_positions()):
        s = state_from(p["moves"])
        legal = s.legal_moves()
        b = hwc(s)
        mb, mmoves, _ = _mirror_position_lr(b, legal, [0] * len(legal), 24)
        mb = np.ascontiguousarray(mb.astype(np.float32))
        rows = np.array([[m[0] for m in legal]], np.int32)
        cols = np.array([[m[1] for m in legal]], np.int32)
        pri, val = mirror.infer(b[None], rows, cols, np.ones_like(rows, np.float32), 24)
        avg = np.concatenate([pri[0].astype(np.float32), val.astype(np.float32)])
        row = {"id": p["id"], "moves": p["moves"], "board_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
               "mirrored_sha256": hashlib.sha256(mb.tobytes()).hexdigest(),
               "mirrored_moves_sha256": hashlib.sha256(json.dumps([key(m) for m in mmoves]).encode()).hexdigest(),
               "averaged_sha256": hashlib.sha256(avg.tobytes()).hexdigest()}
        if i < 3:
            row["mirrored_moves"] = [key(m) for m in mmoves]
            row["averaged_priors"] = [float(x) for x in pri[0]]
            row["averaged_value"] = float(val[0])
        out.append(row)
    return {"provenance": provenance(), "positions": out}


# --- V4: search ---------------------------------------------------------------------------------------------------

def near_full(target_m, seed):
    """Random play that avoids wins, until the side to move has exactly target_m legal moves (ply >= 20)."""
    rng = random.Random(seed)
    for _ in range(200):
        s, moves = TwixtState(active_size=24, to_move="red"), []
        while True:
            legal = s.legal_moves()
            if s.winner() is not None or not legal:
                break
            if len(legal) == target_m and s.ply >= 20:
                return moves
            if len(legal) < target_m:
                break
            rng.shuffle(legal)
            for mv in legal:
                t = s.apply_move(mv)
                if t.winner() is None:
                    break
            else:
                break
            s, moves = t, moves + [list(mv)]
    raise RuntimeError(f"no position with exactly {target_m} legal moves")


def chain_positions():
    """Red's knight chain down column 5/6: at ply 24 red wins with (23,4) or (23,8); at ply 23 black is to move and
    red threatens that win (a terminal child two plies down). The extra black peg is chosen so that a winning move is
    among the fake's 16 highest-prior candidates."""
    red = [(r, 5 if (r // 2) % 2 == 0 else 6) for r in range(0, 23, 2)]
    base_black = [(r, 19) for r in range(1, 23, 2)]
    for extra in [(r, c) for r in range(1, 23) for c in range(9, 17)]:
        black = base_black + [extra]
        moves = [list(m) for pair in zip(red, black) for m in pair]
        s = state_from(moves)
        if s.winner() is not None or len(set(map(tuple, moves))) != len(moves):
            continue
        legal = s.legal_moves()
        pri, _ = fake_row(hwc(s), legal)
        order = sorted(range(len(legal)), key=lambda j: (-pri[j], legal[j]))[:16]
        top = {tuple(legal[j]) for j in order}
        wins = [w for w in [(23, 4), (23, 8)] if s.apply_move(w).winner() == "red"]
        if wins and top & set(wins):
            return {"winning_child": moves, "threat_two_plies": moves[:-1], "wins": [key(w) for w in wins]}
    raise RuntimeError("no chain position puts a winning move among the candidates")


def run_plain(moves, evaluator, n, seed, rseed):
    s = state_from(moves)
    m = TMCTS(evaluator, cfg_from(CONFIG, n), random.Random(seed))
    m.tie_stats = new_stats()
    counts, root_value, root = m.search_with_root(s, add_noise=False)
    move, _ = eval_readout.select(counts, s.ply, READOUT, random.Random(rseed))
    return {"n_legal": len(counts), "counts": {key(k): v for k, v in counts.items() if v},
            "root_value": root_value, "readout_move": key(move), "tie_stats": m.tie_stats}


def run_halving(moves, evaluator, n, seed):
    s = state_from(moves)
    m = THalving(evaluator, cfg_from(CONFIG, n), random.Random(seed))
    m.tie_stats = new_stats()
    counts, root_value, root = m.search_with_root(s, add_noise=False)
    child = root.children[encode_move(*m.last_choice)]
    return {"n_legal": len(counts), "counts": {key(k): v for k, v in counts.items() if v},
            "root_value": root_value, "rounds": [[key(x) for x in r] for r in m.last_rounds],
            "choice": key(m.last_choice), "choice_visits": child.visit_count, "choice_value": -child.q_value,
            "tie_stats": m.tie_stats}


def search_fixture():
    fake = FakeEvaluator()
    mirror = MirrorAveragingEvaluator(fake)
    corpus = {p["id"]: p for p in corpus_positions()}
    opening = [p for p in corpus.values() if p["stratum"] == "opening"]
    later = [p for p in corpus.values() if p["stratum"] in ("early_mid", "midgame", "late")]
    cases = []

    def add(kind, cid, moves, ev_name, n, seed, rseed=None):
        ev = {"fake": fake, "mirror_fake": mirror}[ev_name]
        res = run_plain(moves, ev, n, seed, rseed) if kind == "plain" else run_halving(moves, ev, n, seed)
        cases.append({"id": cid, "kind": kind, "evaluator": ev_name, "moves": moves, "ply": len(moves),
                      "n_sims": n, "search_seed": seed, "readout_seed": rseed, "result": res})

    for i, seed in enumerate([11, 12, 13]):
        add("plain", f"plain_empty_{i}", [], "fake", 200, seed, 1000 + seed)
    for i, p in enumerate(opening[::5][:6]):
        add("plain", f"plain_{p['id']}", p["moves"], "fake", 200, 21 + i, 2000 + i)
    add("plain", "plain_n50", opening[1]["moves"], "fake", 50, 31, 3001)
    add("plain", "plain_mirror_empty", [], "mirror_fake", 200, 41, 4001)
    add("plain", "plain_mirror_opening", opening[7]["moves"], "mirror_fake", 200, 42, 4002)

    for i, p in enumerate(later[::10][:6]):
        add("halving", f"halving_{p['id']}", p["moves"], "fake", 200, 51 + i)
    for i, p in enumerate(later[3::25][:3]):
        add("halving", f"halving_mirror_{p['id']}", p["moves"], "mirror_fake", 200, 61 + i)
    for m, seed in [(9, 71), (5, 72), (2, 73), (1, 74)]:
        add("halving", f"halving_m{m}", near_full(m, 900 + m), "fake", 200, seed)
    chain = chain_positions()
    add("halving", "halving_winning_child", chain["winning_child"], "fake", 200, 81)
    add("halving", "halving_threat_two_plies", chain["threat_two_plies"], "fake", 200, 82)
    return {"provenance": provenance(), "tie_eps": TIE_EPS, "chain_wins": chain["wins"], "cases": cases}


def main():
    for name, build in [("rng.json", rng_fixture), ("mirror.json", mirror_fixture), ("search.json", search_fixture)]:
        data = build()
        (HERE / name).write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
