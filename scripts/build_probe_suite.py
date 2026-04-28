"""Tier-parameterized probe suite generator.

Replaces scripts/build_bootstrap_probe_suite.py as the real implementation
(that script is kept as a thin --tier forced shim for muscle memory and
existing CI/cron commands).

Tiers:
  --tier forced            Bootstrap forced suite (existing behavior,
                           writes tests/probes/twixt_probes.json by default).
  --tier strong_advantage  Bootstrap strong-advantage suite (deep-MCTS
                           labeled, light-reviewed). Phases 1/2/3 per
                           docs/superpowers/specs/2026-04-28-...

Both tiers produce byte-identical output for identical inputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# --- Tier dispatch ---

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--tier", choices=["forced", "strong_advantage"], required=True)
    ap.add_argument("--input", default="scripts/GPU/logs/games")
    ap.add_argument("--source-iter-range", nargs=2, type=int,
                    metavar=("MIN", "MAX"))
    ap.add_argument("--out", default=None,
                    help="Output path. Defaults: forced -> tests/probes/twixt_probes.json, "
                         "strong_advantage -> tests/probes/strong_advantage_probes.json")
    ap.add_argument("--samples-per-bucket", type=int, default=12)
    ap.add_argument("--max-probes", type=int, default=30)

    # strong_advantage-specific flags (ignored for forced)
    ap.add_argument("--label-checkpoint", default=None)
    ap.add_argument("--label-mcts-sims", type=int, default=10000)
    ap.add_argument("--label-mcts-repeats", type=int, default=3)
    ap.add_argument("--magnitude-threshold", type=float, default=0.45)
    ap.add_argument("--top1-share-floor", type=float, default=0.15)
    ap.add_argument("--stability-cap", type=float, default=0.15)
    ap.add_argument("--promote", action="store_true",
                    help="Promote *.draft.json to committed file")
    ap.add_argument("--reviewer", default=None,
                    help="Reviewer name, required with --promote")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing draft or committed file")

    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if args.tier == "forced":
        return _run_forced(args)
    elif args.tier == "strong_advantage":
        return _run_strong_advantage(args)
    else:
        print(f"[probe_suite] ERROR: unknown tier {args.tier}", file=sys.stderr)
        return 2


# --- Forced tier (lifted from build_bootstrap_probe_suite.py) ---

def _run_forced(args) -> int:
    if args.out is None:
        args.out = "tests/probes/twixt_probes.json"
    if args.source_iter_range is None:
        print("[probe_suite] ERROR: --source-iter-range required for --tier forced",
              file=sys.stderr)
        return 2

    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games

    min_iter, max_iter = args.source_iter_range
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"[probe_suite] ERROR: --input path is not a directory: {input_dir}",
              file=sys.stderr)
        return 2

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

    probes = extract_forced_probes_from_games(
        games,
        active_size=24,
        k_plies=2,
        winner_reasons=frozenset({"win"}),
        dedupe_exact=True,
        dedupe_mirror=True,
        max_probes=None,
    )

    # Interleave-then-truncate: balance must survive truncation.
    # extract_forced_probes_from_games already returned each color's probes
    # in canonical sort order. We merge red/black greedily into `balanced`,
    # at each step taking the color with the better sort key AS LONG AS
    # the ≤ 2:1 balance rule would still hold. Stop at max_probes.
    #
    # An earlier version applied a pre-truncation cap and then truncated,
    # but the final truncation could skew the output (e.g., all top-N
    # probes came from the same color when the most recent iters favored
    # that color). Interleaving closes that gap.

    def _sort_key(p: dict) -> tuple:
        basename = p["source_game"]
        try:
            iter_num = int(basename.split("_")[1])
        except (IndexError, ValueError):
            iter_num = 0
        return (-iter_num, -p["source_ply"], basename)

    red = [p for p in probes if p["category"] == "near_win_red"]
    black = [p for p in probes if p["category"] == "near_win_black"]

    balanced: list[dict] = []
    ri = bi = 0
    red_count = black_count = 0
    while len(balanced) < args.max_probes:
        can_red = ri < len(red) and red_count + 1 <= 2 * max(black_count, 1)
        can_black = bi < len(black) and black_count + 1 <= 2 * max(red_count, 1)
        if not can_red and not can_black:
            break
        if can_red and can_black:
            if _sort_key(red[ri]) <= _sort_key(black[bi]):
                balanced.append(red[ri]); ri += 1; red_count += 1
            else:
                balanced.append(black[bi]); bi += 1; black_count += 1
        elif can_red:
            balanced.append(red[ri]); ri += 1; red_count += 1
        else:
            balanced.append(black[bi]); bi += 1; black_count += 1

    balanced.sort(key=_sort_key)

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

    print(f"[probe_suite] wrote {len(balanced)} forced probes to {out_path}")
    return 0


# --- Strong-advantage tier (filled in during Step 2 of the plan) ---

def _run_strong_advantage(args) -> int:
    raise NotImplementedError(
        "Strong-advantage tier added in Step 2 of the implementation plan. "
        "See docs/superpowers/plans/2026-04-28-strong-advantage-probe-tier.md."
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
