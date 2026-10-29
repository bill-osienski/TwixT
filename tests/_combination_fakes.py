"""Test doubles for the combination policy pilot's CPU checks (no MLX, no checkpoint, no GPU). Module-level so
spawned workers can import them by name."""
import math
import os
import random
import time

import numpy as np

SHORT_MAX_MOVES = 26          # halving from ply 20: six searched rows per game, in seconds rather than a minute


class FakeEvaluator:
    """Deterministic, row-wise and padding-invariant: a row's output depends only on that row's board and legal moves,
    never on the batch it arrives in or on padding columns (sums are math.fsum over the legal moves only)."""

    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        self.bias = rng.normal(0, 1.5, (24, 24))
        self.w = rng.normal(0, 0.3, 30)
        self.v = rng.normal(0, 0.02, (24, 24, 30))
        self.network = "fake-network"
        self.calls = 0

    def build_input_tensor(self, state):
        return state.to_tensor()

    def infer(self, boards, rows, cols, mask, active_size):
        self.calls += 1
        B, Mw = rows.shape
        priors = np.zeros((B, Mw), np.float32)
        values = np.zeros(B, np.float32)
        for i in range(B):                       # per row, over the legal moves only: shape-stable sums
            n = int(mask[i].sum())
            r, c = rows[i, :n], cols[i, :n]
            z = self.bias[r, c] + (boards[i, r, c, :].astype(np.float64) * self.w).sum(-1)
            e = np.exp(z - z.max())
            priors[i, :n] = e / math.fsum(e)
            values[i] = math.tanh(float((boards[i].astype(np.float64) * self.v).sum()))
        return priors, values


def short_config():
    from scripts.GPU.alphazero import eval_runner as ER
    from scripts.GPU.alphazero.external_learning_stage3 import PROTOCOL
    return ER.EvalConfig(**{**PROTOCOL, "max_moves": SHORT_MAX_MOVES})


def short_play(evaluator, seed):
    """play_slot under a 26-ply cap, patched for this one call in this process only."""
    from scripts.GPU.alphazero import combination_selfplay as CS
    keep, CS.protocol_config = CS.protocol_config, short_config
    try:
        return CS.play_slot(evaluator, seed)
    finally:
        CS.protocol_config = keep


def probe_play(evaluator, seed):
    """A cheap game stand-in: a few remote inferences whose inputs depend on the seed and whose outputs decide the
    record; each process sleeps a different amount so workers finish out of order."""
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    time.sleep(0.02 * (os.getpid() % 4))
    rng = random.Random(seed)
    state = TwixtState(active_size=24, to_move="red", max_plies_limit=280)
    moves = []
    for _ in range(3 + seed % 3):
        legal = state.legal_moves()
        board = np.transpose(state.to_tensor(), (1, 2, 0))[None].astype(np.float32)
        r = np.array([[m[0] for m in legal]], np.int32)
        c = np.array([[m[1] for m in legal]], np.int32)
        p, _v = evaluator.infer(board, r, c, np.ones_like(r, np.float32), 24)
        noise = [rng.random() * 1e-3 for _ in legal]
        move = legal[int(np.argmax(p[0] + np.array(noise, np.float32)))]
        moves.append(move)
        state = state.apply_move(move)
    game = {"seed": seed, "moves": moves, "winner": None, "reason": "probe", "n_moves": len(moves), "n_rows": 0,
            "row_plies": []}
    return game, {"ply": np.zeros(0, np.int32)}


def failing_play(evaluator, seed):
    raise RuntimeError(f"injected failure at seed {seed}")
