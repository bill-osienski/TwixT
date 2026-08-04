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


def test_tracer_overhead_is_measured_and_reported():
    """Not a pass/fail bar -- a MEASUREMENT. The atlas-budget decision is the
    operator's, made against this number.

    FakeEvaluator isolates tracer cost from NN cost, so this OVERSTATES relative
    overhead versus a real evaluator where inference dominates. Report it as an
    upper bound, never as the production figure.
    """
    import time

    def timed(**kw):
        # 400, not the golden's 200: design section 8 requires a real
        # 400-simulation smoke.
        t0 = time.perf_counter()
        run_fixed_search(n_simulations=400, **kw)
        return time.perf_counter() - t0

    off = min(timed() for _ in range(5))
    on = min(timed(flush_observer=RecordingFlush(),
                   selection_observer=SelectionTracer()) for _ in range(5))
    overhead = (on - off) / off
    verdict = (
        "BELOW MEASUREMENT NOISE (not a speedup -- treat as ~0)"
        if overhead <= 0 else f"{overhead:+.1%} upper bound"
    )
    print(f"\nATLAS STAGE 1 TRACER OVERHEAD: {overhead:+.1%} -- {verdict}"
          f"\n  off {off:.4f}s, all-on {on:.4f}s, 400 sims, min-of-5, FakeEvaluator")
    assert on > 0 and off > 0
