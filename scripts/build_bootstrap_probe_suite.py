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
    ap.add_argument("--input", default="scripts/GPU/logs/games")
    ap.add_argument("--source-iter-range", nargs=2, type=int, required=True,
                    metavar=("MIN", "MAX"))
    ap.add_argument("--out", default="tests/probes/twixt_probes.json")
    ap.add_argument("--samples-per-bucket", type=int, default=12)
    ap.add_argument("--max-probes", type=int, default=30)
    args = ap.parse_args()

    # Add project root to sys.path so this script can import scripts.GPU.alphazero.*
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games

    min_iter, max_iter = args.source_iter_range
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"[bootstrap] ERROR: --input path is not a directory: {input_dir}",
              file=sys.stderr)
        return 2

    # 1. Scan for iter_NNNN_game_MMM.json in range.
    games: list[dict] = []
    for fp in sorted(input_dir.glob("iter_*_game_*.json")):
        with open(fp) as f:
            try:
                g = json.load(f)
            except json.JSONDecodeError:
                continue
        iteration = (g.get("meta") or {}).get("iteration")
        if iteration is None or not (min_iter <= iteration <= max_iter):
            continue
        games.append(g)

    # 2. Extract probes via shared helper (filters: size=24, natural wins,
    #    K=2, dedup exact+mirror).
    probes = extract_forced_probes_from_games(
        games,
        active_size=24,
        k_plies=2,
        winner_reasons=frozenset({"win"}),
        dedupe_exact=True,
        dedupe_mirror=True,
        max_probes=None,  # we balance and truncate below
    )

    # 3. Split by category and enforce ≤ 2:1 balance.
    red = [p for p in probes if p["category"] == "near_win_red"]
    black = [p for p in probes if p["category"] == "near_win_black"]
    if len(red) > 2 * max(len(black), 1):
        red = red[: 2 * max(len(black), 1)]
    if len(black) > 2 * max(len(red), 1):
        black = black[: 2 * max(len(red), 1)]
    balanced = red + black

    # 4. Re-sort balanced set (per spec §4.1 order) and truncate to max_probes.
    # extract_forced_probes_from_games already returned in sort order, but
    # concatenation of red+black may disrupt it — re-sort by the same keys.
    def _sort_key(p: dict) -> tuple:
        # Extract iteration from source_game basename 'iter_NNNN_game_MMM'.
        basename = p["source_game"]
        try:
            iter_num = int(basename.split("_")[1])
        except (IndexError, ValueError):
            iter_num = 0
        return (-iter_num, -p["source_ply"], basename)

    balanced.sort(key=_sort_key)
    if len(balanced) > args.max_probes:
        balanced = balanced[: args.max_probes]

    # 5. Serialize (no wall-clock fields).
    payload = {
        "meta": {
            "type": "bootstrap_rule_selected",
            "not_gate_suite": True,
            "note": ("Rule-selected bootstrap suite for trainer-side inline "
                     "telemetry and practical regression monitoring. NOT the "
                     "spec §7 review-curated gate suite — see "
                     "tests/probes/README.md for the distinction."),
            "generator": "scripts/build_bootstrap_probe_suite.py",
            "generator_version": 1,
            "selection_rules": {
                "board_size": 24,
                "winner_reasons": ["win"],
                "k_plies_from_terminal": 2,
                "dedup": "exact + 4-form-mirror-canonical",
                "source_iter_range": [min_iter, max_iter],
            },
        },
        "probes": balanced,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")

    print(f"[bootstrap] wrote {len(balanced)} probes to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
