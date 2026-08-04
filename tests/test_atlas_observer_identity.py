"""Atlas Stage 1 identity qualification.

The golden is captured from UNMODIFIED mcts.py in Task 0; every later task
re-proves it. Capturing it after a hook edit would prove nothing.
"""
import json
import random

from scripts.GPU.alphazero.mcts import MCTS, MCTSNode
from scripts.GPU.alphazero.game.twixt_state import TwixtState
from scripts.GPU.alphazero.selection_tracer import SelectionTracer

from tests.eval_fakes import FakeEvaluator
from tests.atlas_stage1_fixtures import (
    GOLDEN,
    RecordingFlush,
    RecordingSelection,
    run_fixed_search,
    shipped_config,
)


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


def test_each_hook_individually_preserves_identity():
    baseline = json.loads(GOLDEN.read_text())
    assert run_fixed_search(flush_observer=RecordingFlush()) == baseline
    assert run_fixed_search(selection_observer=RecordingSelection()) == baseline
    assert run_fixed_search(selection_observer=SelectionTracer()) == baseline


def test_all_hooks_together_preserve_identity():
    assert run_fixed_search(
        flush_observer=RecordingFlush(),
        selection_observer=SelectionTracer(),
    ) == json.loads(GOLDEN.read_text())


def test_hooks_actually_fired_during_the_identity_run():
    """Identity is vacuous if the hooks never ran."""
    fo, so = RecordingFlush(), RecordingSelection()
    run_fixed_search(flush_observer=fo, selection_observer=so)
    assert fo.calls and so.calls
