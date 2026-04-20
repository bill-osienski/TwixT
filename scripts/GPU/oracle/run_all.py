#!/usr/bin/env python3
"""
Run all GPU Oracle tests.

This script runs all cross-validation tests between Python and JS implementations
to ensure semantic alignment before training or deployment.

Usage:
    python -m scripts.GPU.oracle.run_all
    python -m scripts.GPU.oracle.run_all --verbose
    python -m scripts.GPU.oracle.run_all --quick  # Skip random tests
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_sealed_lane_tests(verbose: bool = False) -> bool:
    """Run sealed lane oracle tests."""
    from scripts.GPU.oracle.test_sealed_lane import SealedLaneOracleTest
    tester = SealedLaneOracleTest(verbose=verbose)
    return tester.run_all()


def run_heuristics_tests(verbose: bool = False) -> bool:
    """Run heuristics oracle tests."""
    from scripts.GPU.oracle.test_heuristics import HeuristicsOracleTest
    tester = HeuristicsOracleTest(verbose=verbose)
    return tester.run_all()


def main():
    parser = argparse.ArgumentParser(
        description="Run all GPU Oracle cross-validation tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m scripts.GPU.oracle.run_all           # Run all tests
    python -m scripts.GPU.oracle.run_all -v        # Verbose output
    python -m scripts.GPU.oracle.run_all --quick   # Skip random tests

Exit codes:
    0 - All tests passed
    1 - Some tests failed
    2 - Error occurred
"""
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output with detailed diffs")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Skip random position tests (faster)")
    parser.add_argument("--sealed-lane-only", action="store_true",
                        help="Run only sealed lane tests")
    parser.add_argument("--heuristics-only", action="store_true",
                        help="Run only heuristics tests")
    args = parser.parse_args()

    print("=" * 70)
    print("GPU ORACLE CROSS-VALIDATION TEST SUITE")
    print("=" * 70)
    print()
    print("Verifying Python AI implementations match JS exactly...")
    print("This is CRITICAL for model training/deployment alignment.")
    print()

    start_time = time.time()
    all_passed = True
    results = {}

    try:
        if not args.heuristics_only:
            print("\n" + "-" * 70)
            print("RUNNING: Sealed Lane Oracle Tests")
            print("-" * 70)
            sealed_lane_passed = run_sealed_lane_tests(args.verbose)
            results["Sealed Lane"] = sealed_lane_passed
            if not sealed_lane_passed:
                all_passed = False

        if not args.sealed_lane_only:
            print("\n" + "-" * 70)
            print("RUNNING: Heuristics Oracle Tests")
            print("-" * 70)
            heuristics_passed = run_heuristics_tests(args.verbose)
            results["Heuristics"] = heuristics_passed
            if not heuristics_passed:
                all_passed = False

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)

    elapsed = time.time() - start_time

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print()
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    print()
    print(f"Time elapsed: {elapsed:.2f}s")
    print()

    if all_passed:
        print("ALL ORACLE TESTS PASSED!")
        print("Python and JS implementations are semantically aligned.")
        print("Safe to proceed with GPU training.")
        sys.exit(0)
    else:
        print("SOME ORACLE TESTS FAILED!")
        print("Python and JS have semantic differences.")
        print("DO NOT proceed with GPU training until fixed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
