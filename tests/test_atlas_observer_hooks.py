"""Atlas Stage 1 -- observer hook firing, placement and reachability."""
import random

from scripts.GPU.alphazero.mcts import (
    MCTS,
    MCTSFlushObserver,
    MCTSNode,
    MCTSSelectionObserver,
)
from scripts.GPU.alphazero.game.twixt_state import TwixtState

from tests.eval_fakes import FakeEvaluator
from tests.test_atlas_observer_identity import shipped_config


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


def _run(n_simulations=200, **kw):
    state = TwixtState(active_size=24, to_move="red")
    mcts = MCTS(FakeEvaluator(value=0.0), shipped_config(n_simulations),
                random.Random(20260803), **kw)
    root = MCTSNode(state=state)
    mcts.search_from_root(root, add_noise=False, ply=0)
    return mcts, root


def test_mcts_accepts_both_new_observers_independently():
    ev, cfg = FakeEvaluator(value=0.0), shipped_config()
    MCTS(ev, cfg, random.Random(1), flush_observer=RecordingFlush())
    MCTS(ev, cfg, random.Random(1), selection_observer=RecordingSelection())
    MCTS(ev, cfg, random.Random(1),
         flush_observer=RecordingFlush(), selection_observer=RecordingSelection())


def test_existing_observer_protocol_is_untouched():
    """FpuTraceObserver implements ONLY on_root_simulation. Adding the new hooks
    to MCTSObserver would break it."""
    assert not hasattr(MCTSFlushObserver, "on_root_simulation")
    assert not hasattr(MCTSSelectionObserver, "on_root_simulation")
