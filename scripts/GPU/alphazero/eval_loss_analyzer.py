"""CLI for the eval loss analyzer.

Reads one or more *_games.jsonl files, writes a per-match loss summary
(JSON) + by-color / by-length / worst-loss CSVs, a cross-branch comparison
CSV, and prints a console summary. All analysis lives in eval_loss_analysis;
this module is only IO + formatting.
"""
from __future__ import annotations

import argparse
import csv
import glob as globmod
import json
import sys
from pathlib import Path

from .eval_loss_analysis import (
    LENGTH_BUCKETS_DEFAULT, analyze_match, combine_branch_summaries,
    resolve_checkpoints, sample_worst_losses, validate_rows,
)
from .eval_runner import short_id


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Explain how checkpoint A loses to B from *_games.jsonl.")
    p.add_argument("--games-jsonl", action="append", default=[], metavar="PATH",
                   help="input games jsonl (repeatable)")
    p.add_argument("--glob", default=None, metavar="PATTERN",
                   help="glob for input games jsonl files")
    p.add_argument("--output-dir", default=Path("logs/eval/loss_analysis"), type=Path)
    p.add_argument("--a-checkpoint", default=None)
    p.add_argument("--b-checkpoint", default=None)
    p.add_argument("--length-buckets", default=None,
                   help="comma-separated upper-inclusive edges, e.g. 40,60,80,120,279,280")
    p.add_argument("--worst-losses", type=int, default=50)
    return p.parse_args(argv)


def load_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_sibling_summary(games_path):
    sib = Path(str(games_path).replace("_games.jsonl", ".json"))
    if sib != Path(games_path) and sib.exists():
        try:
            return json.loads(sib.read_text())
        except (OSError, json.JSONDecodeError):
            return None
    return None


def stem_of(games_path):
    name = Path(games_path).name
    if name.endswith("_games.jsonl"):
        return name[:-len("_games.jsonl")]
    return Path(games_path).stem


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def resolve_inputs(args):
    paths = list(args.games_jsonl)
    if args.glob:
        paths += sorted(globmod.glob(args.glob))
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _weighted_rate(by_len, labels):
    subs = [b for b in by_len if b["length_bucket"] in labels]
    g = sum(b["games"] for b in subs)
    if not g:
        return None
    return sum(b["a_score_rate"] * b["games"] for b in subs) / g


def loss_shape(s):
    overall = s["a_score_rate"]
    by_len = s["by_length"]
    signals = []
    short = _weighted_rate(by_len, {"<=40", "41-60"})
    if short is not None and short < overall - 0.03:
        signals.append("short/opening games")
    long = _weighted_rate(by_len, {"81-120", "121-279"})
    if long is not None and long < overall - 0.03:
        signals.append("long/endgame games")
    if s["color_gap"] is not None and abs(s["color_gap"]) >= 0.05:
        weaker = "red" if s["color_gap"] < 0 else "black"
        signals.append(f"as {weaker}")
    if s["termination"]["state_cap_rate"] >= 0.05:
        signals.append("state-cap tail")
    if not signals:
        return "A is losing broadly, no strong length/color/termination concentration."
    return "A is losing primarily in " + ", ".join(signals) + "."


def print_console_summary(s):
    print("=" * 60)
    print(f"LOSS ANALYSIS: {s['match']}")
    print("=" * 60)
    print(f"Games: {s['games']}")
    print(f"A score: {s['a_score_rate']:.4f}")
    print(f"Elo: {s['elo']:.1f} [{s['elo_ci95'][0]:.1f}, {s['elo_ci95'][1]:.1f}]")
    print(f"Verdict: {s['verdict']}")
    print("By A color:")
    for c in s["by_color"]:
        rate = c["a_score_rate"]
        rate_s = "n/a" if rate is None else f"{rate:.4f}"
        print(f"  A as {c['a_color']:<5}: {rate_s} over {c['games']} games")
    if s["color_gap"] is not None:
        print(f"  Gap: {s['color_gap']:+.4f}")
    print("By length:")
    for b in s["by_length"]:
        print(f"  {b['length_bucket']:<9}: {b['a_score_rate']:.4f} over {b['games']} games")
    t = s["termination"]
    print(f"State caps: {t['state_cap']} / {s['games']} ({t['state_cap_rate']:.1%})")
    print(f"Board full: {t['board_full']} / {s['games']} ({t['board_full_rate']:.1%})")
    print("Likely loss shape:")
    print(f"  {loss_shape(s)}")


def main(argv=None):
    args = parse_args(argv)
    inputs = resolve_inputs(args)
    if not inputs:
        print("error: no input files (use --games-jsonl and/or --glob)", file=sys.stderr)
        return 2
    buckets = (tuple(int(x) for x in args.length_buckets.split(","))
               if args.length_buckets else LENGTH_BUCKETS_DEFAULT)
    out_dir = args.output_dir
    summaries = []
    for path in inputs:
        rows = load_jsonl(path)
        stem = stem_of(path)
        if not rows:
            print(f"skip {stem}: empty file")
            continue
        sidecar = load_sibling_summary(path)
        a, b = resolve_checkpoints(rows, rows[0]["pairing_id"],
                                   args.a_checkpoint, args.b_checkpoint, sidecar)
        if a == b:
            print(f"skip {stem}: self-match ({short_id(a)})")
            continue
        validate_rows(rows, a, b)
        summary = analyze_match(rows, a, b, match=stem,
                                pairing_id=rows[0]["pairing_id"],
                                length_buckets=buckets)
        worst = [{"match": stem, **w}
                 for w in sample_worst_losses(rows, a, b, args.worst_losses)]
        write_json(out_dir / f"{stem}_loss_summary.json", summary)
        write_csv(out_dir / f"{stem}_by_color.csv",
                  [{"match": stem, **c} for c in summary["by_color"]])
        write_csv(out_dir / f"{stem}_by_length.csv",
                  [{"match": stem, **c} for c in summary["by_length"]])
        write_csv(out_dir / f"{stem}_worst_losses.csv", worst)
        print_console_summary(summary)
        summaries.append(summary)
    if summaries:
        write_csv(out_dir / "combined_branch_comparison.csv",
                  combine_branch_summaries(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
