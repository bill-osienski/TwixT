#!/usr/bin/env python3
"""
Cross-validation tests for Python vs JS heuristics implementations.

Tests verify that Python's heuristics in heuristics.py produce identical
results to the JS implementations in heuristics.js and search.js.

WHY THIS MATTERS:
- GPU training uses Python's heuristics during self-play
- Deployed model runs in JS with JS's heuristics
- If semantics differ, model learns wrong game dynamics

RUN:
    python -m scripts.GPU.oracle.test_heuristics

OUTPUTS:
    - Pass/fail for each test case
    - Summary of agreements/disagreements
    - Detailed diff for any failures
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.game.state import GameState
from scripts.GPU.game.rules import apply_move, generate_moves
from scripts.GPU.ai.heuristics import (
    component_metrics,
    compute_frontier,
    find_connected_components,
    evaluate_connected_paths,
)
from scripts.GPU.oracle.base import JSOracle, state_to_json

# Path to JS oracle
ORACLE_PATH = Path(__file__).parent / "heuristics_oracle.js"


class HeuristicsOracleTest:
    """Test suite for Python vs JS heuristics alignment."""

    def __init__(self, verbose: bool = False):
        self.oracle = JSOracle(ORACLE_PATH)
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.errors = 0

    def _report(self, name: str, match: bool, py: Any = None, js: Any = None, details: str = ""):
        """Report test result."""
        if match:
            print(f"  {name}: PASS")
            self.passed += 1
        else:
            print(f"  {name}: FAIL")
            if self.verbose or py != js:
                print(f"    Python: {py}")
                print(f"    JS:     {js}")
                if details:
                    print(f"    Details: {details}")
            self.failed += 1

    def _report_error(self, name: str, error: str):
        """Report test error."""
        print(f"  {name}: ERROR - {error}")
        self.errors += 1

    # =========================================================================
    # Component Metrics Tests
    # =========================================================================

    def test_component_metrics_empty(self):
        """Empty board should return empty metrics."""
        state = GameState(board_size=24)

        # Python
        py_result = component_metrics(state, "red")

        # JS
        js_input = state_to_json(state, "red")
        js_result = self.oracle.call("componentMetrics", js_input)

        # Compare key fields
        match = (
            py_result.get("touches_top") == js_result.get("touchesTop") and
            py_result.get("touches_bottom") == js_result.get("touchesBottom") and
            len(py_result.get("largest_component", [])) == len(js_result.get("largestComponent", []))
        )
        self._report("component_metrics_empty", match, py_result, js_result)

    def test_component_metrics_single_peg(self):
        """Single peg should have correct metrics."""
        state = GameState(board_size=24)
        state = apply_move(state, 12, 12)  # Red at center

        py_result = component_metrics(state, "red")
        js_input = state_to_json(state, "red")
        js_result = self.oracle.call("componentMetrics", js_input)

        match = (
            len(py_result.get("largest_component", [])) == len(js_result.get("largestComponent", [])) and
            py_result.get("touches_top") == js_result.get("touchesTop") and
            py_result.get("touches_bottom") == js_result.get("touchesBottom")
        )
        self._report("component_metrics_single_peg", match,
                     {"size": len(py_result.get("largest_component", []))},
                     {"size": len(js_result.get("largestComponent", []))})

    def test_component_metrics_edge_touch(self):
        """Peg on edge should touch that edge."""
        state = GameState(board_size=24)
        state = apply_move(state, 0, 12)  # Red on top edge

        py_result = component_metrics(state, "red")
        js_input = state_to_json(state, "red")
        js_result = self.oracle.call("componentMetrics", js_input)

        match = (
            py_result.get("touches_top") == js_result.get("touchesTop") == True and
            py_result.get("touches_bottom") == js_result.get("touchesBottom") == False
        )
        self._report("component_metrics_edge_touch", match,
                     {"touches_top": py_result.get("touches_top")},
                     {"touchesTop": js_result.get("touchesTop")})

    def test_component_metrics_with_bridges(self):
        """Component with bridges should connect pegs."""
        state = GameState(board_size=24)
        # Place red pegs that can form bridges
        state = apply_move(state, 12, 12)  # Red
        state = apply_move(state, 6, 6)    # Black
        state = apply_move(state, 14, 11)  # Red - bridges to 12,12
        state = apply_move(state, 8, 8)    # Black

        py_result = component_metrics(state, "red")
        js_input = state_to_json(state, "red")
        js_result = self.oracle.call("componentMetrics", js_input)

        py_size = len(py_result.get("largest_component", []))
        js_size = len(js_result.get("largestComponent", []))

        match = py_size == js_size
        self._report("component_metrics_with_bridges", match,
                     {"largest_size": py_size, "span": py_result.get("max_row_span")},
                     {"largest_size": js_size, "span": js_result.get("maxRowSpan")})

    # =========================================================================
    # Frontier Tests
    # =========================================================================

    def test_compute_frontier_empty(self):
        """Empty board should have no frontier."""
        state = GameState(board_size=24)

        py_result = compute_frontier(state, "red")
        js_input = state_to_json(state, "red")
        js_result = self.oracle.call("computeFrontier", js_input)

        match = len(py_result.get("frontier", [])) == len(js_result.get("frontier", [])) == 0
        self._report("compute_frontier_empty", match)

    def test_compute_frontier_single_peg(self):
        """Single peg should have frontier cells."""
        state = GameState(board_size=24)
        state = apply_move(state, 12, 12)

        py_result = compute_frontier(state, "red")
        js_input = state_to_json(state, "red")
        js_result = self.oracle.call("computeFrontier", js_input)

        py_frontier = set((c["row"], c["col"]) if isinstance(c, dict) else c
                          for c in py_result.get("frontier", []))
        js_frontier = set((c["row"], c["col"]) for c in js_result.get("frontier", []))

        match = py_frontier == js_frontier
        self._report("compute_frontier_single_peg", match,
                     {"count": len(py_frontier)},
                     {"count": len(js_frontier)})

    # =========================================================================
    # Evaluate Connected Paths Tests
    # =========================================================================

    def test_evaluate_connected_paths_empty(self):
        """Empty board should return -100 (no pegs penalty)."""
        state = GameState(board_size=24)

        py_result = evaluate_connected_paths(state, "red", {})
        js_input = state_to_json(state, "red")
        js_result = self.oracle.call("evaluateConnectedPaths", js_input)

        # Both should return -100 for empty board
        match = py_result == js_result == -100
        self._report("evaluate_connected_paths_empty", match, py_result, js_result)

    def test_evaluate_connected_paths_single(self):
        """Single peg path evaluation."""
        state = GameState(board_size=24)
        state = apply_move(state, 12, 12)

        py_result = evaluate_connected_paths(state, "red", {})
        js_input = state_to_json(state, "red")
        js_result = self.oracle.call("evaluateConnectedPaths", js_input)

        match = py_result == js_result
        self._report("evaluate_connected_paths_single", match, py_result, js_result)

    # =========================================================================
    # Random Position Tests
    # =========================================================================

    def test_random_positions(self, n_positions: int = 30):
        """Test random positions for statistical agreement."""
        print(f"\n  Running {n_positions} random position tests...")

        agreements = 0
        disagreements = []

        random.seed(42)

        for i in range(n_positions):
            state = GameState(board_size=24)

            # Make random moves
            n_moves = random.randint(5, 25)
            for _ in range(n_moves):
                legal = generate_moves(state)
                if not legal:
                    break
                row, col = random.choice(legal)
                try:
                    state = apply_move(state, row, col)
                except:
                    break

            # Test both players
            for player in ["red", "black"]:
                js_input = state_to_json(state, player)

                # Test component_metrics
                py_metrics = component_metrics(state, player)
                js_metrics = self.oracle.call("componentMetrics", js_input)

                py_size = len(py_metrics.get("largest_component", []))
                js_size = len(js_metrics.get("largestComponent", []))

                if py_size != js_size:
                    disagreements.append({
                        "position": i,
                        "player": player,
                        "function": "componentMetrics",
                        "py": py_size,
                        "js": js_size
                    })
                else:
                    agreements += 1

        total = agreements + len(disagreements)
        rate = (agreements / total * 100) if total > 0 else 0

        print(f"    Agreements: {agreements}/{total} ({rate:.1f}%)")
        print(f"    Disagreements: {len(disagreements)}")

        # Show first few disagreements
        for d in disagreements[:3]:
            print(f"      Position {d['position']}, {d['player']}, {d['function']}: py={d['py']}, js={d['js']}")

        if len(disagreements) == 0:
            self.passed += 1
            print("  random_positions: PASS")
        else:
            self.failed += 1
            print(f"  random_positions: FAIL ({len(disagreements)} disagreements)")

    # =========================================================================
    # Run All Tests
    # =========================================================================

    def run_all(self) -> bool:
        """Run all test cases."""
        print("\n" + "=" * 60)
        print("Heuristics JS Oracle Tests")
        print("=" * 60)
        print("\nTesting Python vs JS heuristics alignment...\n")

        print("Component Metrics Tests:")
        self.test_component_metrics_empty()
        self.test_component_metrics_single_peg()
        self.test_component_metrics_edge_touch()
        self.test_component_metrics_with_bridges()

        print("\nFrontier Tests:")
        self.test_compute_frontier_empty()
        self.test_compute_frontier_single_peg()

        print("\nEvaluate Connected Paths Tests:")
        self.test_evaluate_connected_paths_empty()
        self.test_evaluate_connected_paths_single()

        print("\nRandom Position Tests:")
        self.test_random_positions(30)

        print("\n" + "=" * 60)
        print(f"SUMMARY: {self.passed} passed, {self.failed} failed, {self.errors} errors")

        if self.failed == 0 and self.errors == 0:
            print("\nAll tests passed! Python and JS heuristics are aligned.")
            return True
        else:
            print("\nSome tests failed! Python and JS have semantic differences.")
            return False


def main():
    """Run all oracle tests."""
    import argparse
    parser = argparse.ArgumentParser(description="Run heuristics oracle tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    try:
        tester = HeuristicsOracleTest(verbose=args.verbose)
        success = tester.run_all()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
