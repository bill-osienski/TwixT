"""search_with_root must be the SAME search as search() (the gate-faithful
synchronous path), returning the root node as a third element. NOT
search_from_root (different, batched leaf-eval path — forbidden for target
generation; see the v5 path diagnostic)."""
import random

import numpy as np


def _mcts(sims=50, seed=42):
    import mlx.core as mx
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    np.random.seed(7)
    mx.random.seed(7)
    net = create_network(hidden=64, n_blocks=2)
    evaluator = LocalGPUEvaluator(net)
    return (MCTS(evaluator, MCTSConfig(n_simulations=sims), rng=random.Random(seed)),
            MCTS(evaluator, MCTSConfig(n_simulations=sims), rng=random.Random(seed)))


def test_search_with_root_matches_search_and_exposes_tree():
    from scripts.GPU.alphazero.game import TwixtState
    from scripts.GPU.alphazero.mcts import MCTSNode, decode_move
    m1, m2 = _mcts()
    state = TwixtState()
    counts_a, value_a = m1.search(state, add_noise=False)
    counts_b, value_b, root = m2.search_with_root(state, add_noise=False)
    assert counts_a == counts_b
    assert value_a == value_b
    assert isinstance(root, MCTSNode)
    # the returned tree IS the searched tree: child visits match the counts dict
    for move_id, child in root.children.items():
        assert counts_b[decode_move(move_id)] == child.visit_count
    # walkable: some expanded child carries state + nn_value
    visited = [c for c in root.children.values() if c.visit_count > 0]
    assert visited, "no visited children after 50 sims"
    top = max(visited, key=lambda c: c.visit_count)
    assert top.is_expanded and top.nn_value is not None
    assert top.state.to_move != state.to_move
