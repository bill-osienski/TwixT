"""Atlas Stage 1 identity qualification.

The golden is captured from UNMODIFIED mcts.py in Task 0; every later task
re-proves it. Capturing it after a hook edit would prove nothing.
"""
import json
import random
from pathlib import Path

from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig, MCTSNode
from scripts.GPU.alphazero.game.twixt_state import TwixtState

from tests.eval_fakes import FakeEvaluator

GOLDEN = Path("tests/golden/atlas_stage1_prehook_search.json")


def shipped_config(n_simulations: int = 200) -> MCTSConfig:
    # Shipped batching is (14, 48, 8). stall_flush_sims defaults to 16, so 48
    # MUST be set explicitly or this is not the shipped batched path.
    return MCTSConfig(
        n_simulations=n_simulations,
        eval_batch_size=14,
        stall_flush_sims=48,
        pending_virtual_visits=8,
    )


def run_fixed_search(n_simulations: int = 200, **observer_kwargs):
    """One deterministic batched search. observer_kwargs are passed to MCTS.

    The 200 default is the GOLDEN's budget and must not change -- every identity
    comparison depends on it. The timing smoke passes 400 explicitly, because
    design section 8 requires a real 400-simulation measurement.
    """
    state = TwixtState(active_size=24, to_move="red")
    mcts = MCTS(
        FakeEvaluator(value=0.0),
        shipped_config(n_simulations),
        random.Random(20260803),
        **observer_kwargs,
    )
    visit_counts, root_value, root = mcts.search_from_root(
        MCTSNode(state=state), add_noise=False, ply=0
    )
    return {
        "root_value": round(float(root_value), 12),
        "root_visit_count": root.visit_count,
        "visit_counts": sorted([f"{r}:{c}", v] for (r, c), v in visit_counts.items()),
        "flush_full": mcts._flush_full,
        "flush_stall": mcts._flush_stall,
        "flush_tail": mcts._flush_tail,
    }


def test_golden_exists_and_reproduces():
    assert GOLDEN.exists(), (
        "Task 0 golden missing. It MUST be captured from UNMODIFIED mcts.py "
        "before any hook edit; capturing it later proves nothing."
    )
    assert run_fixed_search() == json.loads(GOLDEN.read_text())


def test_golden_is_not_vacuous():
    """A different seed must differ, or the comparison proves nothing."""
    state = TwixtState(active_size=24, to_move="red")
    mcts = MCTS(FakeEvaluator(value=0.0), shipped_config(), random.Random(999999))
    vc, _rv, _root = mcts.search_from_root(
        MCTSNode(state=state), add_noise=False, ply=0
    )
    other = sorted([f"{r}:{c}", v] for (r, c), v in vc.items())
    assert other != json.loads(GOLDEN.read_text())["visit_counts"]


def test_batched_path_was_exercised():
    """No full-batch flush means the golden says nothing about the batched path."""
    assert run_fixed_search()["flush_full"] > 0
