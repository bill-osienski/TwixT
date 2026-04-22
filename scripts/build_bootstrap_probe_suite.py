"""Bootstrap rule-selected forced-probe suite generator.

Produces tests/probes/twixt_probes.json from historical game replays
using strict rule-based selection (no human review). See the spec at
docs/superpowers/specs/2026-04-21-probes-and-calibration-closure-design.md
§5 for selection rules.

The output is a rule-selected bootstrap suite, NOT the spec §7 review-
curated gate suite. See tests/probes/README.md for the distinction.

Reruns with identical --source-iter-range produce byte-identical output
(deterministic probe IDs, deterministic dedup canonicalization, stable
sort keys, no wall-clock fields).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--input", default="scripts/GPU/logs/games",
                    help="Directory containing iter_NNNN_game_MMM.json files.")
    ap.add_argument("--source-iter-range", nargs=2, type=int, required=True,
                    metavar=("MIN", "MAX"),
                    help="Inclusive iteration range to sample from (e.g., 25 30).")
    ap.add_argument("--out", default="tests/probes/twixt_probes.json",
                    help="Output path.")
    ap.add_argument("--samples-per-bucket", type=int, default=12,
                    help="Per winner class, before dedup.")
    ap.add_argument("--max-probes", type=int, default=30,
                    help="Final cap on probe count.")
    args = ap.parse_args()

    # Real implementation in Task 9.
    print(f"[bootstrap] input={args.input}")
    print(f"[bootstrap] source-iter-range={args.source_iter_range}")
    print(f"[bootstrap] out={args.out}")
    raise NotImplementedError("Generation logic lands in Task 9.")


if __name__ == "__main__":
    sys.exit(main() or 0)
