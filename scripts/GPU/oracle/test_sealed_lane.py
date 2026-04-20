#!/usr/bin/env python3
"""
Cross-validation test for Python vs JS sealed lane detection.

This test ensures that Python's sealed lane detection in sealed_lane.py
produces identical results to the JS implementation in search.js.

RUN:
    python -m scripts.GPU.oracle.test_sealed_lane
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.game.state import GameState
from scripts.GPU.game.rules import apply_move, generate_moves
from scripts.GPU.ai.heuristics import component_metrics
from scripts.GPU.ai.sealed_lane import check_sealed_lane
from scripts.GPU.oracle.base import JSOracle

# Path to JS oracle
ORACLE_PATH = Path(__file__).parent / "sealed_lane_oracle.js"


def state_to_sealed_lane_format(state: GameState, player: str, metrics: Dict) -> Dict:
    """Convert Python state to JS sealed lane oracle format."""
    pegs = []
    for (row, col), p in state.pegs.items():
        pegs.append({"row": row, "col": col, "player": p})

    bridges = []
    for (r1, c1), (r2, c2) in state.bridges:
        bridges.append({"r1": r1, "c1": c1, "r2": r2, "c2": c2})

    component = []
    for row, col in metrics.get("largest_component", []):
        component.append({"row": row, "col": col})

    return {
        "boardSize": state.board_size,
        "pegs": pegs,
        "bridges": bridges,
        "player": player,
        "component": component,
        "touchesTop": metrics.get("touches_top", False),
        "touchesBottom": metrics.get("touches_bottom", False),
        "touchesLeft": metrics.get("touches_left", False),
        "touchesRight": metrics.get("touches_right", False),
    }


class SealedLaneOracleTest:
    """Test cases for Python vs JS sealed lane detection."""

    def __init__(self, verbose: bool = False):
        self.oracle = JSOracle(ORACLE_PATH)
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.errors = 0

    def _call_js_oracle(self, input_data: Dict) -> bool:
        """Call the sealed lane JS oracle."""
        result = self.oracle.call_raw(input_data)
        if result.get("error"):
            return None
        return result.get("reachable")

    def _report(self, name: str, match: bool, py: bool, js: bool):
        if js is None:
            print(f"  {name}: ERROR (JS oracle failed)")
            self.errors += 1
        elif match:
            print(f"  {name}: PASS (py={py}, js={js})")
            self.passed += 1
        else:
            print(f"  {name}: FAIL (py={py}, js={js})")
            self.failed += 1

    def _compare(self, state: GameState, player: str, metrics: Dict) -> Tuple[bool, bool, bool]:
        """Compare Python and JS results."""
        player_int = 0 if player == "red" else 1
        component = metrics.get("largest_component", [])

        if player == "red":
            touches_tl = metrics.get("touches_top", False)
            touches_br = metrics.get("touches_bottom", False)
        else:
            touches_tl = metrics.get("touches_left", False)
            touches_br = metrics.get("touches_right", False)

        py_result = check_sealed_lane(state, player_int, component, touches_tl, touches_br, None)

        oracle_input = state_to_sealed_lane_format(state, player, metrics)
        js_result = self._call_js_oracle(oracle_input)

        if js_result is None:
            return py_result, None, False

        return py_result, js_result, py_result == js_result

    def test_empty_component(self):
        """Empty component should return False for both."""
        state = GameState(board_size=24)
        py_result = check_sealed_lane(state, 0, [], False, False, None)
        js_result = self._call_js_oracle({
            "boardSize": 24, "pegs": [], "bridges": [], "player": "red",
            "component": [], "touchesTop": False, "touchesBottom": False,
            "touchesLeft": False, "touchesRight": False
        })
        match = py_result == js_result == False
        self._report("empty_component", match, py_result, js_result)

    def test_single_peg_center(self):
        """Single peg in center - lane should be open."""
        state = GameState(board_size=24)
        state = apply_move(state, 12, 12)
        metrics = component_metrics(state, "red")
        py, js, match = self._compare(state, "red", metrics)
        self._report("single_peg_center", match, py, js)

    def test_touching_top_edge(self):
        """Peg on top edge - should be able to reach bottom."""
        state = GameState(board_size=24)
        state = apply_move(state, 0, 12)
        metrics = component_metrics(state, "red")
        py, js, match = self._compare(state, "red", metrics)
        self._report("touching_top_edge", match, py, js)

    def test_touching_bottom_edge(self):
        """Peg on bottom edge - should be able to reach top."""
        state = GameState(board_size=24)
        state = apply_move(state, 23, 12)
        metrics = component_metrics(state, "red")
        py, js, match = self._compare(state, "red", metrics)
        self._report("touching_bottom_edge", match, py, js)

    def test_black_touching_left(self):
        """Black peg on left edge - should be able to reach right."""
        state = GameState(board_size=24)
        state = apply_move(state, 12, 12)  # Red
        state = apply_move(state, 12, 0)   # Black on left edge
        metrics = component_metrics(state, "black")
        py, js, match = self._compare(state, "black", metrics)
        self._report("black_touching_left", match, py, js)

    def test_random_positions(self, n_positions: int = 50):
        """Test random positions for statistical agreement."""
        print(f"\n  Running {n_positions} random position tests...")

        agreements = 0
        disagreements = 0
        errors = 0

        random.seed(42)

        for i in range(n_positions):
            state = GameState(board_size=24)

            n_moves = random.randint(5, 30)
            for _ in range(n_moves):
                legal = generate_moves(state)
                if not legal:
                    break
                row, col = random.choice(legal)
                try:
                    state = apply_move(state, row, col)
                except:
                    break

            for player in ["red", "black"]:
                metrics = component_metrics(state, player)
                py, js, match = self._compare(state, player, metrics)

                if js is None:
                    errors += 1
                elif match:
                    agreements += 1
                else:
                    disagreements += 1
                    if disagreements <= 3:
                        print(f"\n  Disagreement {disagreements}:")
                        print(f"    Position {i}, player={player}")
                        print(f"    Python={py}, JS={js}")

        total = agreements + disagreements
        if total > 0:
            rate = agreements / total * 100
            print(f"\n  Random test results:")
            print(f"    Agreements: {agreements}/{total} ({rate:.1f}%)")
            print(f"    Disagreements: {disagreements}")
            print(f"    Errors: {errors}")

            if disagreements == 0:
                print(f"  random_positions: PASS")
                self.passed += 1
            else:
                print(f"  random_positions: FAIL ({disagreements} disagreements)")
                self.failed += 1
        else:
            print(f"  random_positions: ERROR (no valid tests)")
            self.errors += 1

    def run_all(self) -> bool:
        """Run all test cases."""
        print("\n" + "=" * 60)
        print("Sealed Lane JS Oracle Tests")
        print("=" * 60)
        print("\nTesting Python vs JS sealed lane detection alignment...\n")

        print("Basic tests:")
        self.test_empty_component()
        self.test_single_peg_center()
        self.test_touching_top_edge()
        self.test_touching_bottom_edge()
        self.test_black_touching_left()

        print("\nStatistical tests:")
        self.test_random_positions(50)

        print("\n" + "=" * 60)
        print(f"SUMMARY: {self.passed} passed, {self.failed} failed, {self.errors} errors")

        if self.failed == 0 and self.errors == 0:
            print("\nAll tests passed! Python and JS sealed lane detection are aligned.")
            return True
        else:
            print("\nSome tests failed! Python and JS have semantic differences.")
            return False


def main():
    """Run all Oracle tests."""
    import argparse
    parser = argparse.ArgumentParser(description="Run sealed lane oracle tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    try:
        tester = SealedLaneOracleTest(verbose=args.verbose)
        success = tester.run_all()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
