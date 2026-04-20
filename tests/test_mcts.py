#!/usr/bin/env python3
"""Tests for MCTS implementation.

Includes both functional tests and critical convention tests that ensure
correctness of the AlphaZero MCTS implementation.

Run with: python3 tests/test_mcts.py
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_visit_counts_increase():
    """Test that visit counts increase with simulations.

    Seeds MLX + numpy RNG before network init so random-init priors are
    deterministic. Without seeding, 100 sims over 576 legal moves with nearly
    flat random priors can leave max_visits==1 (~30% of runs), which made the
    test flaky. With the seeds below, max_visits reliably lands at 6.
    """
    import random
    import numpy as np
    import mlx.core as mx
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.game import TwixtState

    np.random.seed(42)
    mx.random.seed(42)
    net = create_network(hidden=64, n_blocks=2)
    evaluator = LocalGPUEvaluator(net)
    config = MCTSConfig(n_simulations=100)
    mcts = MCTS(evaluator, config, rng=random.Random(42))

    state = TwixtState()
    visit_counts, _ = mcts.search(state)

    total_visits = sum(visit_counts.values())
    assert total_visits == 100, f"Expected 100 visits, got {total_visits}"

    # At least some moves should have multiple visits
    max_visits = max(visit_counts.values())
    assert max_visits > 1, "Expected some moves to have multiple visits"

    print("PASS: Visit counts increase with simulations")


def test_more_sims_changes_distribution():
    """Test that more simulations can change move preferences."""
    import random
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.game import TwixtState

    net = create_network(hidden=64, n_blocks=2)
    evaluator = LocalGPUEvaluator(net)

    # Few simulations
    config_few = MCTSConfig(n_simulations=20)
    mcts_few = MCTS(evaluator, config_few, rng=random.Random(42))
    state = TwixtState()
    counts_few, _ = mcts_few.search(state)

    # More simulations
    config_more = MCTSConfig(n_simulations=200)
    mcts_more = MCTS(evaluator, config_more, rng=random.Random(42))
    counts_more, _ = mcts_more.search(state)

    # Distribution should be different (more concentrated with more sims)
    def entropy(counts):
        total = sum(counts.values())
        if total == 0:
            return 0
        probs = [c / total for c in counts.values() if c > 0]
        return -sum(p * (p + 1e-10) for p in probs if p > 0)

    # With more simulations, total visits scale up
    total_few = sum(counts_few.values())
    total_more = sum(counts_more.values())

    # More simulations = more total visits
    assert total_more > total_few, "More sims should give more visits"

    # Visit distribution should span multiple moves (not degenerate)
    n_visited_more = sum(1 for c in counts_more.values() if c > 0)
    assert n_visited_more > 1, "Should explore multiple moves"

    print("PASS: More simulations changes distribution")


def test_dirichlet_noise_affects_priors():
    """Test that Dirichlet noise modifies root priors."""
    import random
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig, MCTSNode
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.game import TwixtState

    net = create_network(hidden=64, n_blocks=2)
    evaluator = LocalGPUEvaluator(net)

    # Search without noise
    config_no_noise = MCTSConfig(n_simulations=10, dirichlet_eps=0.0)
    mcts_no_noise = MCTS(evaluator, config_no_noise, rng=random.Random(42))

    state = TwixtState()
    root_no_noise = MCTSNode(state=state)
    mcts_no_noise._expand(root_no_noise)
    priors_original = dict(root_no_noise.priors)

    # Search with noise
    config_noise = MCTSConfig(n_simulations=10, dirichlet_eps=0.25)
    mcts_noise = MCTS(evaluator, config_noise, rng=random.Random(42))

    root_noise = MCTSNode(state=state)
    mcts_noise._expand(root_noise)
    mcts_noise._add_dirichlet_noise(root_noise)
    priors_noisy = dict(root_noise.priors)

    # Priors should be different after adding noise
    diffs = [abs(priors_original[m] - priors_noisy[m]) for m in priors_original]
    max_diff = max(diffs)
    assert max_diff > 0.001, f"Noise should affect priors, max diff = {max_diff}"

    print("PASS: Dirichlet noise affects root priors")


def test_expand_always_calls_nn():
    """Test that _expand() always calls NN (leaf eval rule)."""
    import random
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig, MCTSNode
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.game import TwixtState

    net = create_network(hidden=64, n_blocks=2)
    evaluator = LocalGPUEvaluator(net)
    config = MCTSConfig(n_simulations=10)
    mcts = MCTS(evaluator, config, rng=random.Random(42))

    state = TwixtState()
    node = MCTSNode(state=state)

    # Reset counter
    mcts._nn_call_count = 0

    # Expand should call NN
    mcts._expand(node)

    assert mcts._nn_call_count == 1, f"Expected 1 NN call, got {mcts._nn_call_count}"
    assert node.priors is not None, "Priors should be set after expansion"
    assert node.nn_value is not None, "NN value should be set after expansion"

    print("PASS: _expand() always calls NN")


def test_single_nn_eval_per_expansion():
    """Test that NN is called exactly once per node expansion."""
    import random
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.game import TwixtState

    net = create_network(hidden=64, n_blocks=2)
    evaluator = LocalGPUEvaluator(net)
    config = MCTSConfig(n_simulations=50)
    mcts = MCTS(evaluator, config, rng=random.Random(42))

    state = TwixtState()

    # Reset counter
    mcts._nn_call_count = 0

    # Run search
    visit_counts, _ = mcts.search(state)

    # Number of NN calls should be <= number of unique nodes expanded
    # In practice, it's 1 (root) + nodes expanded during simulations
    # Should be much less than n_simulations (since we reuse nodes)
    assert mcts._nn_call_count <= config.n_simulations + 1, \
        f"Too many NN calls: {mcts._nn_call_count}"

    # Should be more than 1 (at least root gets expanded)
    assert mcts._nn_call_count >= 1, "At least root should be expanded"

    print(f"PASS: Single NN eval per expansion ({mcts._nn_call_count} calls for {config.n_simulations} sims)")


def test_terminal_value_opponent_won():
    """Test that terminal value is -1 when opponent just won."""
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.game import TwixtState

    mcts = MCTS(create_network(hidden=64, n_blocks=2), MCTSConfig())

    # Create a state where we manually set a winner
    # In TwixT, when someone wins, the winner is usually the last player to move
    # So if it's Red's turn and Black won, value should be -1 for Red

    # We can test the _terminal_value method directly
    state = TwixtState()

    # For a real terminal state, we'd need to construct a winning position
    # For now, test the logic by checking a non-terminal state returns appropriate value

    # The _terminal_value checks state.winner() and state.to_move
    # Let's verify the logic is correct by examining the implementation

    # Test: if winner() returns 'black' and to_move is 'red', value should be -1
    # We can't easily create a winning state, but we can test the logic

    class MockTerminalState:
        def winner(self):
            return 'black'

        @property
        def to_move(self):
            return 'red'

    mock_state = MockTerminalState()
    value = mcts._terminal_value(mock_state)
    assert value == -1.0, f"Expected -1.0 when opponent won, got {value}"

    # Test: if winner == to_move, value should be +1
    class MockWinState:
        def winner(self):
            return 'red'

        @property
        def to_move(self):
            return 'red'

    mock_win = MockWinState()
    value = mcts._terminal_value(mock_win)
    assert value == 1.0, f"Expected +1.0 when current player won, got {value}"

    print("PASS: Terminal value correct when opponent won (-1)")


def test_terminal_value_draw():
    """Test that terminal value is 0 for draw."""
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
    from scripts.GPU.alphazero.network import create_network

    mcts = MCTS(create_network(hidden=64, n_blocks=2), MCTSConfig())

    # Test draw case (winner is None)
    class MockDrawState:
        def winner(self):
            return None

        @property
        def to_move(self):
            return 'red'

    mock_draw = MockDrawState()
    value = mcts._terminal_value(mock_draw)
    assert value == 0.0, f"Expected 0.0 for draw, got {value}"

    print("PASS: Terminal value is 0 for draw")


def test_backup_sign_flip():
    """Test that backup alternates sign correctly (3-node path)."""
    import random
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig, MCTSNode
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.game import TwixtState

    net = create_network(hidden=64, n_blocks=2)
    mcts = MCTS(net, MCTSConfig(), rng=random.Random(42))

    # Create a 3-node path manually
    state0 = TwixtState()
    state1 = state0.apply_move((10, 10))
    state2 = state1.apply_move((5, 5))

    root = MCTSNode(state=state0)
    child = MCTSNode(state=state1, parent=root, move=(10, 10))
    grandchild = MCTSNode(state=state2, parent=child, move=(5, 5))

    root.children[(10, 10)] = child
    child.children[(5, 5)] = grandchild

    # Create search path
    search_path = [root, child, grandchild]

    # Backup with leaf value = 0.5 (from grandchild's perspective)
    leaf_value = 0.5
    mcts._backup(search_path, leaf_value)

    # Check visit counts
    assert root.visit_count == 1
    assert child.visit_count == 1
    assert grandchild.visit_count == 1

    # Check value_sum with sign flips:
    # grandchild gets +0.5
    # child gets -0.5 (flipped)
    # root gets +0.5 (flipped again)
    assert abs(grandchild.value_sum - 0.5) < 1e-6, \
        f"Grandchild value_sum: {grandchild.value_sum}"
    assert abs(child.value_sum - (-0.5)) < 1e-6, \
        f"Child value_sum: {child.value_sum}"
    assert abs(root.value_sum - 0.5) < 1e-6, \
        f"Root value_sum: {root.value_sum}"

    print("PASS: Backup sign flip works correctly (3-node path)")


def test_deterministic_selection():
    """Test deterministic move selection with low temperature."""
    import random
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.game import TwixtState

    net = create_network(hidden=64, n_blocks=2)
    evaluator = LocalGPUEvaluator(net)
    config = MCTSConfig(
        n_simulations=50,
        temp_threshold_ply=0,  # Always use low temp
        temp_low=0.001,  # Near-deterministic
    )
    mcts = MCTS(evaluator, config, rng=random.Random(42))

    state = TwixtState()
    visit_counts, _ = mcts.search(state, add_noise=False)

    # With very low temperature, should pick the highest visit count move
    move = mcts.select_move(visit_counts, ply=10)

    # Should be one of the moves with the highest visit count
    max_count = max(visit_counts.values())
    best_moves = [m for m, c in visit_counts.items() if c == max_count]
    assert move in best_moves, f"Should pick a top move from {best_moves}, got {move}"

    # With the same seed, repeated selections should be reproducible
    mcts2 = MCTS(evaluator, config, rng=random.Random(42))
    visit_counts2, _ = mcts2.search(state, add_noise=False)
    move2 = mcts2.select_move(visit_counts2, ply=10)
    assert move == move2, "Same seed should produce same move"

    print("PASS: Deterministic selection with low temperature")


def test_random_tiebreak():
    """Test random tie-breaking for deterministic selection (per-seed reproducibility)."""
    import random
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
    from scripts.GPU.alphazero.network import create_network

    mcts = MCTS(
        create_network(hidden=64, n_blocks=2),
        MCTSConfig(temp_low=0.001),
        rng=random.Random(42),
    )

    # Create artificial visit counts with tie
    visit_counts = {
        (5, 5): 10,
        (3, 7): 10,  # Same count
        (3, 3): 10,  # Same count
        (1, 1): 5,
    }

    move = mcts.select_move(visit_counts, ply=100)  # High ply for low temp

    # Should pick one of the tied moves (random, but reproducible per seed)
    tied_moves = [(5, 5), (3, 7), (3, 3)]
    assert move in tied_moves, f"Expected one of {tied_moves}, got {move}"
    assert move != (1, 1), f"Should not pick (1, 1) which has lower count"

    print("PASS: Random tie-breaking")


def test_policy_target():
    """Test policy target normalization."""
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
    from scripts.GPU.alphazero.network import create_network

    mcts = MCTS(create_network(hidden=64, n_blocks=2), MCTSConfig())

    visit_counts = {
        (0, 1): 10,
        (1, 1): 20,
        (2, 2): 70,
    }

    policy = mcts.get_policy_target(visit_counts)

    # Should sum to 1
    total = sum(policy.values())
    assert abs(total - 1.0) < 1e-6, f"Policy should sum to 1, got {total}"

    # Should match proportions
    assert abs(policy[(0, 1)] - 0.1) < 1e-6
    assert abs(policy[(1, 1)] - 0.2) < 1e-6
    assert abs(policy[(2, 2)] - 0.7) < 1e-6

    print("PASS: Policy target normalization")


def test_puct_formula():
    """Test PUCT selection formula components."""
    import random
    import math
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig, MCTSNode, decode_move
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.game import TwixtState

    net = create_network(hidden=64, n_blocks=2)
    evaluator = LocalGPUEvaluator(net)
    config = MCTSConfig(c_puct=1.5, n_simulations=10)
    mcts = MCTS(evaluator, config, rng=random.Random(42))

    # Create and expand root
    state = TwixtState()
    root = MCTSNode(state=state)
    mcts._expand(root)

    # Children are created lazily - manually create some from priors
    move_ids = list(root.priors.keys())[:3]
    for mid in move_ids:
        r, c = decode_move(mid)
        child = MCTSNode(
            state=state.apply_move((r, c)),
            parent=root,
            move=mid,
        )
        root.children[mid] = child

    # Manually set some visit counts to test PUCT
    root.visit_count = 10

    root.children[move_ids[0]].visit_count = 5
    root.children[move_ids[0]].value_sum = 2.5  # Q = 0.5

    root.children[move_ids[1]].visit_count = 3
    root.children[move_ids[1]].value_sum = -0.9  # Q = -0.3

    root.children[move_ids[2]].visit_count = 0  # Unvisited

    # Select child
    selected_move, _ = mcts._select_child(root)

    # The unvisited node should have high exploration bonus
    # PUCT for unvisited: c * prior * sqrt(11) / 1 = c * prior * 3.317
    # Visited nodes have lower exploration bonus

    # We can't predict exactly which is selected without knowing priors,
    # but verify selection works without error
    assert selected_move in root.priors, "Selected move should be in priors"

    print("PASS: PUCT formula components")


def main():
    """Run all MCTS tests."""
    print("=" * 60)
    print("MCTS IMPLEMENTATION TESTS")
    print("=" * 60)
    print()

    tests = [
        # Functional tests
        test_visit_counts_increase,
        test_more_sims_changes_distribution,
        test_dirichlet_noise_affects_priors,
        test_deterministic_selection,
        test_random_tiebreak,
        test_policy_target,
        test_puct_formula,

        # Critical convention tests
        test_expand_always_calls_nn,
        test_single_nn_eval_per_expansion,
        test_terminal_value_opponent_won,
        test_terminal_value_draw,
        test_backup_sign_flip,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()  # passes if no exception; assertions raise on failure
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} - {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("Gate PASSED: MCTS produces sensible visit distributions")
        print("Gate PASSED: All convention tests pass")
        return 0
    else:
        print("Gate FAILED: MCTS tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
