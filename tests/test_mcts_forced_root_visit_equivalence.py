"""Equivalence test for Spec 3 Fix 1 (§9.1).

A single forced sim with ``root_move_override=move_id`` MUST produce the
same child.visit_count increment, the same ``value`` backed up through
the search path, and the same ``root.value_sum`` delta as a normal sim
that selects ``move_id`` by PUCT.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.game.twixt_state import TwixtState
from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig, MCTSNode, encode_move


def _stub_value_fn():
    """Deterministic stub: uniform priors over legal moves, value=0.5."""
    def f(state):
        legal = state.legal_moves()
        if not legal:
            return {}, 0.0
        p = 1.0 / len(legal)
        return {encode_move(r, c): p for (r, c) in legal}, 0.5
    return f


def _build_mcts(stub):
    cfg = MCTSConfig(n_simulations=1)
    m = MCTS(evaluator=None, config=cfg)

    def stub_expand(node):
        priors, value = stub(node.state)
        node.priors_raw = dict(priors)
        node.priors = dict(priors)
        return value
    m._expand = stub_expand
    return m


def test_forced_override_matches_normal_puct_path():
    state = TwixtState()
    stub = _stub_value_fn()

    # Branch A: normal sim — expand root, then run one normal PUCT sim.
    m_a = _build_mcts(stub)
    root_a = MCTSNode(state=state)
    m_a._expand(root_a)
    # Determine which child PUCT picks (with uniform priors, ties broken by rng).
    # Force the rng to pick a specific child deterministically.
    legal = list(state.legal_moves())
    chosen_move = legal[0]
    chosen_id = encode_move(*chosen_move)

    # Stub _select_child to return our chosen move (uniform priors -> arbitrary).
    def stub_select_a(node, pending_ids=None):
        # Return chosen move at root level; deeper descent won't recurse here
        # because the freshly-expanded child has no children yet and will be
        # treated as a (newly expanded) leaf on the next iteration.
        return chosen_id, node.children.get(chosen_id)
    m_a._select_child = stub_select_a
    m_a._run_single_simulation(root_a, root_move_override=None)

    # Branch B: forced sim — override root to chosen_id.
    m_b = _build_mcts(stub)
    root_b = MCTSNode(state=state)
    m_b._expand(root_b)
    m_b._run_single_simulation(root_b, root_move_override=chosen_id)

    # Compare child visit counts at root.
    child_a = root_a.children[chosen_id]
    child_b = root_b.children[chosen_id]
    assert child_a.visit_count == child_b.visit_count
    assert child_a.value_sum == child_b.value_sum
    # And the parent root accumulators must match.
    assert root_a.visit_count == root_b.visit_count
    assert root_a.value_sum == root_b.value_sum


def test_multiple_forced_overrides_match_multiple_normal_sims():
    """Force-visit one move 5 times — child.visit_count must equal exactly 5.

    The plan describes a Branch A / Branch B comparison here, but explicitly
    notes that mirroring the multi-sim PUCT path in Branch A is hard (the
    stub _select_child returns root-legal move ids which are illegal at
    deeper plies). The asserted invariant is the Branch B one — forcing N
    visits on a single move increments that child's visit_count by exactly N.
    """
    state = TwixtState()
    stub = _stub_value_fn()
    legal = list(state.legal_moves())[:3]

    cfg = MCTSConfig(n_simulations=5, closeout_td1_visit_forcing_enabled=True,
                     closeout_td1_min_visits=5, closeout_td1_max_forced_moves=1)
    m_b = _build_mcts(stub)
    m_b.config = cfg
    root_b = MCTSNode(state=state)
    m_b._expand(root_b)
    forced = m_b.force_root_visits(root_b, [legal[0]], min_visits=5, max_candidates=1)
    assert forced == 5
    assert root_b.children[encode_move(*legal[0])].visit_count == 5
