#!/usr/bin/env python3
"""Encoding parity check script.

Verifies that Python and Node.js produce identical 24-channel tensor encodings.

Run with: python3 tests/run_encoding_parity.py
"""
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.game import TwixtState, NUM_CHANNELS, BOARD_SIZE, MAX_PLIES

NODE_ORACLE_PATH = PROJECT_ROOT / "tests" / "js_oracle" / "game_rules_oracle.mjs"


def run_js_oracle_with_tensor(moves):
    """Run the Node.js oracle and get tensor encoding."""
    input_data = json.dumps({"moves": moves, "include_tensor": True})

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


def get_python_tensor(moves):
    """Get Python tensor encoding after applying moves."""
    state = TwixtState.from_moves(moves)
    return state.to_tensor()


def compare_tensors(py_tensor, js_tensor, moves, tolerance=1e-9):
    """Compare Python and JS tensors element by element.

    Args:
        py_tensor: numpy array (24, 24, 24)
        js_tensor: nested list [24][24][24]
        moves: move sequence (for error reporting)
        tolerance: max allowed difference

    Returns:
        List of error messages (empty if match)
    """
    errors = []
    max_diff = 0.0
    diff_count = 0

    for c in range(NUM_CHANNELS):
        for r in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                py_val = py_tensor[c, r, col]
                js_val = js_tensor[c][r][col]
                diff = abs(py_val - js_val)

                if diff > max_diff:
                    max_diff = diff

                if diff > tolerance:
                    diff_count += 1
                    if diff_count <= 5:  # Only report first 5 differences
                        errors.append(
                            f"Mismatch at channel={c}, row={r}, col={col}: "
                            f"Py={py_val:.6f}, JS={js_val:.6f}, diff={diff:.2e}"
                        )

    if diff_count > 5:
        errors.append(f"... and {diff_count - 5} more differences")

    if max_diff > tolerance:
        errors.append(f"Max difference: {max_diff:.2e}")

    return errors


def run_test(name, moves, verbose=True):
    """Run a single encoding parity test."""
    try:
        py_tensor = get_python_tensor(moves)
        js_result = run_js_oracle_with_tensor(moves)
        js_tensor = js_result["tensor"]

        errors = compare_tensors(py_tensor, js_tensor, moves)

        if errors:
            print(f"FAIL: {name}")
            for e in errors:
                print(f"  {e}")
            return False
        else:
            if verbose:
                print(f"PASS: {name}")
            return True
    except Exception as e:
        print(f"ERROR: {name} - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("ENCODING PARITY CHECK")
    print("=" * 60)
    print(f"NUM_CHANNELS: {NUM_CHANNELS}")
    print(f"BOARD_SIZE: {BOARD_SIZE}")
    print()

    passed = 0
    failed = 0

    # Test 1: Empty state
    if run_test("Empty state", []):
        passed += 1
    else:
        failed += 1

    # Test 2: Single red move
    if run_test("Single red move", [(10, 10)]):
        passed += 1
    else:
        failed += 1

    # Test 3: Red + Black moves
    if run_test("Red + Black moves", [(10, 10), (5, 5)]):
        passed += 1
    else:
        failed += 1

    # Test 4: With bridge
    if run_test("With bridge (knight move)", [(10, 10), (5, 5), (12, 11)]):
        passed += 1
    else:
        failed += 1

    # Test 5: Multiple bridges
    if run_test("Multiple bridges", [
        (10, 10), (5, 5), (12, 11), (7, 6), (14, 12)
    ]):
        passed += 1
    else:
        failed += 1

    # Test 6: Edge positions
    if run_test("Edge positions", [
        (0, 12),   # Red on top edge
        (12, 0),   # Black on left edge
        (23, 12),  # Red on bottom edge
        (12, 23),  # Black on right edge
    ]):
        passed += 1
    else:
        failed += 1

    # Test 7-16: Short random games (10 moves each)
    print("\nRunning 10 short random games...")
    for seed in range(10):
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

        if run_test(f"Short random game (seed={seed})", moves, verbose=False):
            passed += 1
        else:
            failed += 1

    # Test 17-66: Medium random games (50 moves each)
    print("\nRunning 50 medium random games...")
    medium_passed = 0
    for seed in range(50):
        rng = random.Random(seed + 100)
        moves = []
        state = TwixtState()

        for _ in range(50):
            legal = state.legal_moves()
            if not legal or state.is_terminal():
                break
            move = rng.choice(legal)
            moves.append(move)
            state = state.apply_move(move)

        if run_test(f"Medium random game (seed={seed+100})", moves, verbose=False):
            medium_passed += 1
            passed += 1
        else:
            failed += 1

    print(f"Medium random games: {medium_passed}/50 passed")

    # Test: Early game
    print("\nTesting game phase encoding...")
    rng = random.Random(999)
    state = TwixtState()
    moves = []
    for _ in range(5):
        legal = state.legal_moves()
        if not legal:
            break
        move = rng.choice(legal)
        moves.append(move)
        state = state.apply_move(move)

    if run_test("Early game (5 moves)", moves):
        passed += 1
    else:
        failed += 1

    # Test: Mid game
    for _ in range(45):  # Get to 50 moves
        legal = state.legal_moves()
        if not legal or state.is_terminal():
            break
        move = rng.choice(legal)
        moves.append(move)
        state = state.apply_move(move)

    if run_test("Mid game (50 moves)", moves):
        passed += 1
    else:
        failed += 1

    # Test: Late game
    for _ in range(50):  # Get to 100 moves
        legal = state.legal_moves()
        if not legal or state.is_terminal():
            break
        move = rng.choice(legal)
        moves.append(move)
        state = state.apply_move(move)

    if run_test("Late game (100+ moves)", moves):
        passed += 1
    else:
        failed += 1

    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("Gate PASSED: 100% encoding parity on all tests")
        return 0
    else:
        print("Gate FAILED: Encoding parity issues detected")
        return 1


if __name__ == "__main__":
    sys.exit(main())
