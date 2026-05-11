"""Unit tests for Spec 3 Fix 1 — td=1 root visit forcing (mcts side)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.mcts import MCTSConfig


def test_config_defaults_disable_visit_forcing():
    c = MCTSConfig()
    assert c.closeout_td1_visit_forcing_enabled is False
    assert c.closeout_td1_min_visits == 8
    assert c.closeout_td1_max_forced_moves == 4
    assert c.closeout_td1_require_high_value is False
    assert c.closeout_td1_high_value_threshold == 0.95


def test_config_accepts_overrides():
    c = MCTSConfig(closeout_td1_visit_forcing_enabled=True,
                   closeout_td1_min_visits=16,
                   closeout_td1_max_forced_moves=2,
                   closeout_td1_require_high_value=True,
                   closeout_td1_high_value_threshold=0.9)
    assert c.closeout_td1_visit_forcing_enabled is True
    assert c.closeout_td1_min_visits == 16
    assert c.closeout_td1_high_value_threshold == 0.9


# ---------------------------------------------------------------------------
# Task 11 — force_root_visits
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock


def _make_mcts_with_stub_eval(value_fn, prior_uniform=True, n_sims=64):
    """Build an MCTS whose NN eval is a deterministic stub.

    value_fn(state) -> (priors_dict, value) for any state.

    Stubs BOTH _expand (used by the synchronous search() / force_root_visits
    paths) and _expand_batch (used by search_from_root's batched leaf eval)
    so the evaluator is never called.
    """
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
    cfg = MCTSConfig(n_simulations=n_sims)
    m = MCTS(evaluator=None, config=cfg)

    def _apply(node):
        priors, value = value_fn(node.state)
        node.priors_raw = dict(priors)
        node.priors = dict(priors)
        node.nn_value = value
        return value

    def stub_expand(node):
        return _apply(node)

    def stub_expand_batch(nodes):
        return [_apply(n) for n in nodes]

    m._expand = stub_expand
    m._expand_batch = stub_expand_batch
    m.rng = MagicMock()
    m.rng.choice = lambda xs: xs[0]
    m.rng.random = lambda: 0.5
    return m


def test_force_root_visits_runs_exactly_min_visits_per_candidate():
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    from scripts.GPU.alphazero.mcts import MCTSConfig, MCTSNode, encode_move

    cfg = MCTSConfig(n_simulations=400,
                     closeout_td1_visit_forcing_enabled=True,
                     closeout_td1_min_visits=3,
                     closeout_td1_max_forced_moves=2)
    state = TwixtState()

    def stub(state):
        legal = state.legal_moves()
        if not legal:
            return {}, 0.0
        p = 1.0 / len(legal)
        return {encode_move(r, c): p for (r, c) in legal}, 0.5
    m = _make_mcts_with_stub_eval(stub, n_sims=400)
    m.config = cfg
    root = MCTSNode(state=state)
    m._expand(root)
    # Pick the first two legal moves as candidates
    legal = list(state.legal_moves())[:2]
    forced = m.force_root_visits(
        root=root,
        candidate_moves=legal,
        min_visits=cfg.closeout_td1_min_visits,
        max_candidates=cfg.closeout_td1_max_forced_moves,
    )
    assert forced == 6  # 3 visits each * 2 candidates
    for mv in legal:
        child = root.children[encode_move(*mv)]
        assert child.visit_count == 3


# ---------------------------------------------------------------------------
# Task 13 — telemetry + search_from_root wiring
# ---------------------------------------------------------------------------
def test_search_from_root_invokes_force_when_td1_triggers():
    """When closeout_td1_visit_forcing_enabled and gc_state has td=1 and
    endpoint_completion_moves non-empty, the MCTS telemetry counters update."""
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    from scripts.GPU.alphazero.mcts import MCTSConfig, MCTSNode, encode_move

    cfg = MCTSConfig(n_simulations=20,
                     closeout_td1_visit_forcing_enabled=True,
                     closeout_td1_min_visits=2,
                     closeout_td1_max_forced_moves=2)
    state = TwixtState()

    def stub(state):
        legal = state.legal_moves()
        p = 1.0 / len(legal)
        return {encode_move(r, c): p for (r, c) in legal}, 0.5
    m = _make_mcts_with_stub_eval(stub, n_sims=20)
    m.config = cfg
    # Reset telemetry to a known state
    m.reset_closeout_td1_telemetry()
    root = MCTSNode(state=state)
    legal = list(state.legal_moves())[:2]
    gc_state = {
        "total_goal_distance": 1,
        "endpoint_completion_moves": legal,
    }
    m.search_from_root(root, add_noise=False, ply=42, gc_state_full=gc_state)
    tel = m.get_closeout_td1_telemetry()
    assert tel["positions_triggered"] == 1
    assert tel["forced_sims_total"] == 4   # min_visits=2 * 2 candidates
