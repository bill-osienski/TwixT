#!/usr/bin/env python3
"""Manual parity check script (no pytest required).

Run with: python3 tests/run_parity_check.py
"""
import json
import random
import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.game import TwixtState, BOARD_SIZE, MAX_PLIES

NODE_ORACLE_PATH = PROJECT_ROOT / "tests" / "js_oracle" / "game_rules_oracle.mjs"


def run_js_oracle(moves):
    """Run the Node.js oracle and get state after applying moves."""
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


def get_python_state(moves):
    """Get Python state after applying moves."""
    state = TwixtState.from_moves(moves)

    return {
        "legal_moves": sorted(state.legal_moves()),
        "is_terminal": state.is_terminal(),
        "winner": state.winner(),
        "to_move": state.to_move,
        "ply": state.ply,
    }


def compare_states(py_state, js_state, moves):
    """Compare Python and JS states."""
    js_legal = [tuple(m) for m in js_state["legal_moves"]]

    errors = []

    if py_state["to_move"] != js_state["to_move"]:
        errors.append(f"to_move mismatch: Py={py_state['to_move']}, JS={js_state['to_move']}")

    if py_state["ply"] != js_state["ply"]:
        errors.append(f"ply mismatch: Py={py_state['ply']}, JS={js_state['ply']}")

    if py_state["legal_moves"] != js_legal:
        py_set = set(py_state["legal_moves"])
        js_set = set(js_legal)
        errors.append(
            f"legal_moves mismatch: Py has {len(py_state['legal_moves'])}, JS has {len(js_legal)}\n"
            f"  Py-only: {py_set - js_set}\n"
            f"  JS-only: {js_set - py_set}"
        )

    if py_state["is_terminal"] != js_state["is_terminal"]:
        errors.append(f"is_terminal mismatch: Py={py_state['is_terminal']}, JS={js_state['is_terminal']}")

    if py_state["winner"] != js_state["winner"]:
        errors.append(f"winner mismatch: Py={py_state['winner']}, JS={js_state['winner']}")

    return errors


def run_test(name, moves, verbose=True):
    """Run a single parity test."""
    try:
        py_state = get_python_state(moves)
        js_state = run_js_oracle(moves)
        errors = compare_states(py_state, js_state, moves)

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
        return False


def main():
    print("=" * 60)
    print("GAME RULES PARITY CHECK")
    print("=" * 60)
    print(f"BOARD_SIZE: {BOARD_SIZE}")
    print(f"MAX_PLIES: {MAX_PLIES}")
    print()

    passed = 0
    failed = 0

    # Test 1: Initial state
    if run_test("Initial state", []):
        passed += 1
    else:
        failed += 1

    # Test 2: Single move
    if run_test("Single move", [(12, 12)]):
        passed += 1
    else:
        failed += 1

    # Test 3: Red on top edge
    if run_test("Red on top edge", [(0, 12)]):
        passed += 1
    else:
        failed += 1

    # Test 4: Black on left edge
    if run_test("Black on left edge", [(12, 12), (12, 0)]):
        passed += 1
    else:
        failed += 1

    # Test 5: Bridge creation
    if run_test("Bridge creation", [(10, 10), (5, 5), (12, 11)]):
        passed += 1
    else:
        failed += 1

    # Test 6-15: Short random games (10 moves each)
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

    # Test 16-115: Full random games (100 games)
    print("\nRunning 100 full random games...")
    full_passed = 0
    for seed in range(100):
        rng = random.Random(seed + 1000)
        moves = []
        state = TwixtState()

        while not state.is_terminal() and len(moves) < MAX_PLIES:
            legal = state.legal_moves()
            if not legal:
                break
            move = rng.choice(legal)
            moves.append(move)
            state = state.apply_move(move)

        if run_test(f"Full random game (seed={seed+1000})", moves, verbose=False):
            full_passed += 1
            passed += 1
        else:
            failed += 1
            # Print first failure details
            if full_passed == 0:
                print(f"  First failure: {len(moves)} moves, terminal={state.is_terminal()}")

    print(f"Full random games: {full_passed}/100 passed")

    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("Gate PASSED: 100% parity on all tests")
        return 0
    else:
        print("Gate FAILED: Parity issues detected")
        return 1


if __name__ == "__main__":
    sys.exit(main())
