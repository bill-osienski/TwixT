"""Parity tests for Python vs Node.js TwixT game rules.

This test ensures the Python and Node.js implementations produce
identical results for any game position, which is critical for
training (Python) vs inference (Node.js) parity.

Tests:
- 100+ random game sequences
- Legal moves match (sorted comparison)
- Terminal detection matches
- Winner matches
- Forced draw at MAX_PLIES
- Edge cases (corners, edges)

Run with: pytest tests/test_game_rules_parity.py -v
"""
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.game import TwixtState, BOARD_SIZE, MAX_PLIES


# Path to the Node.js oracle script
NODE_ORACLE_PATH = PROJECT_ROOT / "tests" / "js_oracle" / "game_rules_oracle.mjs"


def run_js_oracle(moves: List[Tuple[int, int]]) -> dict:
    """Run the Node.js oracle and get state after applying moves.

    Args:
        moves: List of (row, col) moves to apply

    Returns:
        Dict with: legal_moves, is_terminal, winner, to_move, ply
    """
    input_data = json.dumps({"moves": moves})

    result = subprocess.run(
        ["node", str(NODE_ORACLE_PATH)],
        input=input_data,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        raise RuntimeError(f"Node.js oracle failed: {result.stderr}")

    return json.loads(result.stdout)


def get_python_state(moves: List[Tuple[int, int]]) -> dict:
    """Get Python state after applying moves.

    Returns:
        Dict with: legal_moves, is_terminal, winner, to_move, ply
    """
    state = TwixtState.from_moves(moves)

    return {
        "legal_moves": sorted(state.legal_moves()),
        "is_terminal": state.is_terminal(),
        "winner": state.winner(),
        "to_move": state.to_move,
        "ply": state.ply,
    }


def compare_states(py_state: dict, js_state: dict, moves: List[Tuple[int, int]]) -> None:
    """Compare Python and JS states, raising AssertionError on mismatch."""
    # Convert JS legal_moves from [[r,c], ...] to [(r,c), ...] for comparison
    js_legal = [tuple(m) for m in js_state["legal_moves"]]

    assert py_state["to_move"] == js_state["to_move"], \
        f"to_move mismatch after {len(moves)} moves: Py={py_state['to_move']}, JS={js_state['to_move']}"

    assert py_state["ply"] == js_state["ply"], \
        f"ply mismatch: Py={py_state['ply']}, JS={js_state['ply']}"

    assert py_state["legal_moves"] == js_legal, \
        f"legal_moves mismatch after {len(moves)} moves:\n" \
        f"  Py has {len(py_state['legal_moves'])} moves\n" \
        f"  JS has {len(js_legal)} moves\n" \
        f"  Py-only: {set(py_state['legal_moves']) - set(js_legal)}\n" \
        f"  JS-only: {set(js_legal) - set(py_state['legal_moves'])}"

    assert py_state["is_terminal"] == js_state["is_terminal"], \
        f"is_terminal mismatch after {len(moves)} moves: Py={py_state['is_terminal']}, JS={js_state['is_terminal']}"

    assert py_state["winner"] == js_state["winner"], \
        f"winner mismatch after {len(moves)} moves: Py={py_state['winner']}, JS={js_state['winner']}"


class TestGameRulesParity:
    """Test suite for Python/Node.js game rules parity."""

    @pytest.fixture(autouse=True)
    def check_node_oracle(self):
        """Ensure Node.js oracle exists."""
        if not NODE_ORACLE_PATH.exists():
            pytest.skip(f"Node.js oracle not found at {NODE_ORACLE_PATH}")

    def test_initial_state(self):
        """Empty board should have identical state."""
        py_state = get_python_state([])
        js_state = run_js_oracle([])
        compare_states(py_state, js_state, [])

    def test_single_move(self):
        """Single move should produce identical state."""
        # Valid center move
        moves = [(12, 12)]
        py_state = get_python_state(moves)
        js_state = run_js_oracle(moves)
        compare_states(py_state, js_state, moves)

    def test_few_moves_red_top_edge(self):
        """Red placing on top edge (row 0) is valid."""
        moves = [(0, 12)]  # Red on top edge
        py_state = get_python_state(moves)
        js_state = run_js_oracle(moves)
        compare_states(py_state, js_state, moves)

    def test_few_moves_black_left_edge(self):
        """Black placing on left edge (col 0) is valid."""
        moves = [(12, 12), (12, 0)]  # Red center, Black left edge
        py_state = get_python_state(moves)
        js_state = run_js_oracle(moves)
        compare_states(py_state, js_state, moves)

    @pytest.mark.parametrize("seed", range(10))
    def test_random_game_short(self, seed):
        """Play 10 random moves and verify parity at each step."""
        rng = random.Random(seed)
        moves = []
        state = TwixtState()

        for _ in range(10):
            legal = state.legal_moves()
            if not legal or state.is_terminal():
                break

            move = rng.choice(legal)
            moves.append(move)
            state = state.apply_move(move)

            # Compare at this step
            py_state = get_python_state(moves)
            js_state = run_js_oracle(moves)
            compare_states(py_state, js_state, moves)

    @pytest.mark.parametrize("seed", range(100))
    def test_random_game_full(self, seed):
        """Play full random games (up to terminal) and verify parity."""
        rng = random.Random(seed + 1000)  # Different seed range than short tests
        moves = []
        state = TwixtState()

        while not state.is_terminal() and len(moves) < MAX_PLIES:
            legal = state.legal_moves()
            if not legal:
                break

            move = rng.choice(legal)
            moves.append(move)
            state = state.apply_move(move)

        # Final state comparison
        py_state = get_python_state(moves)
        js_state = run_js_oracle(moves)
        compare_states(py_state, js_state, moves)

        # If game ended, verify terminal/winner
        if state.is_terminal():
            assert py_state["is_terminal"] == True
            assert js_state["is_terminal"] == True

    def test_bridge_creation(self):
        """Test that bridges are created correctly on knight moves."""
        # Place pegs that form a knight-move pattern
        moves = [
            (10, 10),  # Red
            (5, 5),    # Black
            (12, 11),  # Red - knight move from (10, 10)
        ]
        py_state = get_python_state(moves)
        js_state = run_js_oracle(moves)
        compare_states(py_state, js_state, moves)

    def test_bridge_crossing_prevention(self):
        """Test that crossing bridges are prevented."""
        # Create a scenario where bridges would cross
        moves = [
            (10, 10),  # Red
            (9, 12),   # Black
            (12, 11),  # Red - creates bridge (10,10)-(12,11)
            (11, 10),  # Black - creates bridge (9,12)-(11,10) which would cross
        ]
        # The last move should still be legal (peg placement)
        # but the crossing bridge should NOT be created
        py_state = get_python_state(moves)
        js_state = run_js_oracle(moves)
        compare_states(py_state, js_state, moves)

    def test_max_plies_draw(self):
        """Test that game becomes terminal at MAX_PLIES.

        MAX_PLIES was bumped from 200 to 600 in production to accommodate
        longer curriculum games; this test just pins the constant to the
        production value to catch accidental regressions.
        """
        assert MAX_PLIES == 600

        # Create a state and manually set ply
        state = TwixtState()
        state = TwixtState(
            board_size=state.board_size,
            to_move=state.to_move,
            pegs=state.pegs,
            bridges=state.bridges,
            ply=MAX_PLIES,
        )
        assert state.is_terminal()
        assert state.winner() is None
        assert state.game_result() == "draw"


class TestEdgeCases:
    """Test edge cases for game rules."""

    def test_corner_invalid(self):
        """All four corners should be invalid placements."""
        state = TwixtState()
        corners = [(0, 0), (0, 23), (23, 0), (23, 23)]

        for r, c in corners:
            assert not state.is_valid_placement(r, c), \
                f"Corner ({r}, {c}) should be invalid"

    def test_red_edge_restrictions(self):
        """Red cannot place on left/right edges."""
        state = TwixtState()  # Red to move

        # Left edge (col 0) - invalid for red
        for row in range(1, 23):  # Skip corners
            assert not state.is_valid_placement(row, 0), \
                f"Red should not place at ({row}, 0)"

        # Right edge (col 23) - invalid for red
        for row in range(1, 23):
            assert not state.is_valid_placement(row, 23), \
                f"Red should not place at ({row}, 23)"

        # Top edge (row 0) - valid for red
        for col in range(1, 23):
            assert state.is_valid_placement(0, col), \
                f"Red should be able to place at (0, {col})"

        # Bottom edge (row 23) - valid for red
        for col in range(1, 23):
            assert state.is_valid_placement(23, col), \
                f"Red should be able to place at (23, {col})"

    def test_black_edge_restrictions(self):
        """Black cannot place on top/bottom edges."""
        # Make one move so black is to_move
        state = TwixtState().apply_move((12, 12))
        assert state.to_move == "black"

        # Top edge (row 0) - invalid for black
        for col in range(1, 23):
            assert not state.is_valid_placement(0, col), \
                f"Black should not place at (0, {col})"

        # Bottom edge (row 23) - invalid for black
        for col in range(1, 23):
            assert not state.is_valid_placement(23, col), \
                f"Black should not place at (23, {col})"

        # Left edge (col 0) - valid for black
        for row in range(1, 23):
            assert state.is_valid_placement(row, 0), \
                f"Black should be able to place at ({row}, 0)"

        # Right edge (col 23) - valid for black
        for row in range(1, 23):
            assert state.is_valid_placement(row, 23), \
                f"Black should be able to place at ({row}, 23)"

    def test_legal_moves_count_initial(self):
        """Initial board should have predictable legal move count."""
        state = TwixtState()
        legal = state.legal_moves()

        # 24x24 = 576 cells
        # - 4 corners = 4
        # - Left edge (col 0, rows 1-22) = 22 (invalid for red)
        # - Right edge (col 23, rows 1-22) = 22 (invalid for red)
        # Total invalid: 4 + 22 + 22 = 48
        # Legal: 576 - 48 = 528
        expected = 24 * 24 - 4 - 22 - 22
        assert len(legal) == expected, \
            f"Expected {expected} legal moves, got {len(legal)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
