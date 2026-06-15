"""Mode A generator: goal-line trigger candidates CSV -> probe manifest JSON.

Reads the checked-in candidates CSV, applies the selection filter, and writes the
fixed probe manifest. Reproduces the canonical 18-case manifest from the
canonical candidates CSV.

Mode B (DEFERRED): a future generator may RE-DERIVE the candidates CSV by scanning
V2.1 collapse/replay outputs (collapse_timing.csv, drop_windows.csv, replays) and
classifying trigger zones, so candidates can be regenerated from any new capture.
Mode A intentionally consumes the checked/known candidates CSV so the probe target
stays fixed and reproducible.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .goal_line_trigger_probe_cases import DEFAULT_SELECTION, select_cases

MANIFEST_NAME = "goal_line_trigger_black_defense_probe"
MANIFEST_DESCRIPTION = (
    "Positions where a checkpoint as black confidently overvalued the position "
    "immediately before a red goal-line or near-goal-line trigger move.")


def build_manifest(candidate_rows, selection, source):
    cases = select_cases(candidate_rows, selection)
    return {
        "schema_version": 1,
        "name": MANIFEST_NAME,
        "source": source,
        "description": MANIFEST_DESCRIPTION,
        "selection": selection,
        "num_cases": len(cases),
        "cases": cases,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Mode A: goal-line trigger candidates CSV -> probe manifest.")
    p.add_argument("--from-candidates-csv", required=True, metavar="PATH")
    p.add_argument("--output", required=True, metavar="PATH")
    p.add_argument("--min-prev-black-value", type=float,
                   default=DEFAULT_SELECTION["min_prev_black_value"])
    p.add_argument("--min-prev-black-top1", type=float,
                   default=DEFAULT_SELECTION["min_prev_black_top1"])
    p.add_argument("--post-opening-only", action="store_true", default=True)
    p.add_argument("--no-post-opening-only", action="store_false",
                   dest="post_opening_only")
    p.add_argument("--trigger-zone-prefix",
                   default=DEFAULT_SELECTION["trigger_zone_prefix"])
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.from_candidates_csv) as f:
        rows = list(csv.DictReader(f))
    selection = {
        "min_prev_black_value": args.min_prev_black_value,
        "min_prev_black_top1": args.min_prev_black_top1,
        "post_opening_only": args.post_opening_only,
        "trigger_zone_prefix": args.trigger_zone_prefix,
    }
    manifest = build_manifest(rows, selection, args.from_candidates_csv)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest['num_cases']} cases -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
