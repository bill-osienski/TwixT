"""Shared fixtures for Atlas Stage 1 tests.

Lives in its own module so `test_atlas_observer_hooks` and
`test_atlas_observer_identity` never import from each other -- importing across
test modules in both directions is a circular import that fails during pytest
collection while passing when either file is run alone.
"""
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
    # NOTE: lists, not tuples -- json.dumps writes arrays, so a tuple-valued
    # fresh run would never compare equal to the JSON-decoded golden.
    return {
        "root_value": round(float(root_value), 12),
        "root_visit_count": root.visit_count,
        "visit_counts": sorted([f"{r}:{c}", v] for (r, c), v in visit_counts.items()),
        "flush_full": mcts._flush_full,
        "flush_stall": mcts._flush_stall,
        "flush_tail": mcts._flush_tail,
    }


class RecordingFlush:
    def __init__(self):
        self.calls = []

    def on_flush_complete(self, flush_type, root):
        self.calls.append((flush_type, root.visit_count))


class RecordingSelection:
    def __init__(self):
        self.calls = []

    def on_select_child(self, parent, selected_move, existing_child, depth,
                        parent_completed_visits, root_override,
                        within_forced_simulation):
        self.calls.append({
            "move": selected_move,
            "first_touch": existing_child is None,
            "depth": depth,
            "parent_visits": parent_completed_visits,
            "root_override": root_override,
            "forced_sim": within_forced_simulation,
        })
