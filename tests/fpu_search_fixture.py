"""CPU fake evaluator + deterministic search harness (no GPU/MLX). Uniform priors +
value 0.0; with a fixed rng seed the real search is fully deterministic, so its output
is a bit-exact fingerprint."""
import dataclasses, random
import numpy as np
from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
from scripts.GPU.alphazero.game.twixt_state import TwixtState


class FakeEvaluator:
    network = None
    def build_input_tensor(self, state):
        a = state.active_size
        return np.zeros((1, a, a), dtype=np.float32)          # infer ignores boards
    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        B, M = move_rows.shape
        s = move_mask.sum(axis=1, keepdims=True)
        priors = np.divide(move_mask, s, out=np.zeros_like(move_mask), where=s > 0)
        return priors.astype(np.float32), np.zeros(B, dtype=np.float32)


def run_search(config=None, *, seed=1234, active_size=6, moves=((2, 2), (3, 3)),
               n_sims=200, observer=None):
    st = TwixtState(active_size=active_size, to_move="red", max_plies_limit=None)
    for m in moves:
        st = st.apply_move(m)
    cfg = dataclasses.replace(config or MCTSConfig(), n_simulations=n_sims)   # FIX: apply n_sims
    mcts = MCTS(FakeEvaluator(), cfg, random.Random(seed),
                **({"observer": observer} if observer is not None else {}))
    visit_counts, root_value, root = mcts.search_with_root(st, add_noise=False)
    assert root.visit_count == n_sims, (root.visit_count, n_sims)
    assert sum(visit_counts.values()) == n_sims               # each sim descends one root child
    fp = {
        "n_sims": n_sims,
        "root_visit_count": int(root.visit_count),
        "root_value_hex": float(root_value).hex(),            # bit-exact, not rounded (fix 4)
        "visits": [[f"{r},{c}", int(v)] for (r, c), v in sorted(visit_counts.items())],
    }
    return fp, root, mcts
