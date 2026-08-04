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


def _run(n_simulations=200, active_size=24, **kw):
    state = TwixtState(active_size=active_size, to_move="red")
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


def test_flush_hook_fires_and_types_match_counters():
    fo = RecordingFlush()
    mcts, _root = _run(flush_observer=fo)
    assert fo.calls, "no flush events emitted"
    seen = [t for t, _n in fo.calls]
    assert seen.count("full") == mcts._flush_full
    assert seen.count("stall") == mcts._flush_stall
    assert seen.count("tail") == mcts._flush_tail


def test_flush_events_are_ordered_and_tail_is_last():
    """What this CAN establish: events arrive in non-decreasing root-visit order
    and the tail flush is last.

    What it CANNOT: that pending_nodes / pending_waiters / pending_node_ids were
    empty at the callback. Those are locals inside search_from_root and are not
    observable from a test. After-the-clears placement is enforced as a CALL-SITE
    REQUIREMENT and by review -- not by this test, and not by exposing internal
    queues.
    """
    fo = RecordingFlush()
    _run(flush_observer=fo)
    counts = [n for _t, n in fo.calls]
    assert counts == sorted(counts)
    assert fo.calls[-1][0] == "tail"


def test_selection_hook_fires_with_first_touch_and_depth():
    """Small board on purpose. At active_size=24 the FakeEvaluator's uniform
    priors over ~528 moves mean 200 sims never revisit a child, so EVERY event
    is a first touch and the present-child path goes unexercised."""
    so = RecordingSelection()
    _run(selection_observer=so, active_size=6)
    assert so.calls
    assert any(c["first_touch"] for c in so.calls)
    assert any(not c["first_touch"] for c in so.calls)
    assert min(c["depth"] for c in so.calls) == 0
    assert all(c["parent_visits"] >= 0 for c in so.calls)


def test_root_override_is_reachable():
    """_run_single_simulation(root_move_override=...) must emit an event with
    root_override=True. A hook only in the batched loop makes this unreachable."""
    so = RecordingSelection()
    state = TwixtState(active_size=24, to_move="red")
    mcts = MCTS(FakeEvaluator(value=0.0), shipped_config(8),
                random.Random(20260803), selection_observer=so)
    root = MCTSNode(state=state)
    mcts._expand(root)
    forced_move = sorted(root.priors)[0]
    mcts._run_single_simulation(root, root_move_override=forced_move)

    overrides = [c for c in so.calls if c["root_override"]]
    assert len(overrides) == 1, f"expected exactly one override event, got {len(overrides)}"
    assert overrides[0]["move"] == forced_move
    assert overrides[0]["depth"] == 0
    assert all(not c["root_override"] for c in so.calls if c["depth"] > 0)


def test_within_forced_simulation_is_a_covariate():
    so = RecordingSelection()
    state = TwixtState(active_size=24, to_move="red")
    mcts = MCTS(FakeEvaluator(value=0.0), shipped_config(8),
                random.Random(20260803), selection_observer=so)
    root = MCTSNode(state=state)
    mcts._expand(root)
    mcts._run_single_simulation(root, root_move_override=sorted(root.priors)[0])
    assert all(c["forced_sim"] for c in so.calls)
