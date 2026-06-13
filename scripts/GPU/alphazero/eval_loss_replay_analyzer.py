"""CLI for the V2 Phase B replay-aware loss analyzer.

Reads Phase A capture data (*_games.jsonl rows carrying replay_path + per-game
replay sidecars), explains WHY checkpoint A loses in the focus window, and
writes seven artifacts per match to --output-dir. All analysis lives in
eval_loss_replay_analysis; this module is IO + composition + formatting.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .eval_loss_analysis import (
    a_color, resolve_checkpoints, score_for_checkpoint, validate_rows,
)
from .eval_loss_analyzer import (
    load_jsonl, load_sibling_summary, resolve_inputs, stem_of, write_csv,
    write_json,
)
from .eval_loss_replay_analysis import (
    MIN_WIN_COHORT, Thresholds, b_side_features, build_replay_summary,
    classify_collapse, cohort_comparison_row, game_features, make_verdict,
    drop_window_rows, opening_cluster_rows, phase_bucket_rows,
    review_queue_rows,
    secondary_contrast_summary, side_plies, validate_replay,
)
from .eval_runner import short_id


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Explain WHY checkpoint A loses, from Phase A replay data.")
    p.add_argument("--games-jsonl", action="append", default=[], metavar="PATH",
                   help="input games jsonl with replay_path rows (repeatable)")
    p.add_argument("--glob", default=None, metavar="PATTERN",
                   help="glob for input games jsonl files")
    p.add_argument("--output-dir", default=Path("logs/eval/loss_analysis_v2"),
                   type=Path)
    p.add_argument("--a-checkpoint", default=None)
    p.add_argument("--b-checkpoint", default=None)
    p.add_argument("--a-color", choices=("red", "black"), default="black")
    p.add_argument("--min-moves", type=int, default=41)
    p.add_argument("--max-moves", type=int, default=80)
    p.add_argument("--opening-plies", type=int, default=20,
                   help="temperature-sampled opening window; confidence/"
                        "diffusion features use plies >= this only")
    p.add_argument("--opening-key-plies", type=int, default=4)
    p.add_argument("--bad-value", type=float, default=-0.25)
    p.add_argument("--lost-value", type=float, default=-0.50)
    p.add_argument("--sharp-drop", type=float, default=0.40)
    p.add_argument("--low-top1-share", type=float, default=0.10)
    p.add_argument("--low-visit-rank", type=int, default=5)
    p.add_argument("--review-queue", type=int, default=50)
    args = p.parse_args(argv)
    if args.bad_value <= args.lost_value:
        p.error("--bad-value must be greater than --lost-value")
    if args.sharp_drop <= 0:
        p.error("--sharp-drop must be > 0")
    return args


def thresholds_from_args(args):
    return Thresholds(
        bad_value=args.bad_value, lost_value=args.lost_value,
        sharp_drop=args.sharp_drop, low_top1_share=args.low_top1_share,
        low_visit_rank=args.low_visit_rank, opening_plies=args.opening_plies)


def load_replay(row):
    path = row.get("replay_path")
    if path is None:
        raise ValueError(
            f"game {row['game_idx']}: focus-window row has no replay_path "
            "(partially captured file)")
    with open(path) as fh:
        replay = json.load(fh)
    validate_replay(row, replay)
    return replay


def analyze_input(path, args, th):
    """Full analysis for one games.jsonl; returns the artifact bundle or
    None when the file is skippable (empty / no capture / self-match)."""
    rows = load_jsonl(path)
    stem = stem_of(path)
    if not rows:
        print(f"skip {stem}: empty file")
        return None
    if not any(r.get("replay_path") for r in rows):
        print(f"skip {stem}: no replay capture (no replay_path in rows)")
        return None
    sidecar = load_sibling_summary(path)
    a, b = resolve_checkpoints(rows, rows[0]["pairing_id"],
                               args.a_checkpoint, args.b_checkpoint, sidecar)
    if a == b:
        print(f"skip {stem}: self-match ({short_id(a)})")
        return None
    validate_rows(rows, a, b)
    a_clr = args.a_color
    b_clr = "red" if a_clr == "black" else "black"
    window = [r for r in rows if a_color(r, a) == a_clr
              and args.min_moves <= r["n_moves"] <= args.max_moves]
    decisive = [r for r in window if r["reason"] == "win"]
    loss_rows = [r for r in decisive if score_for_checkpoint(r, a) == 0.0]
    win_rows = [r for r in decisive if score_for_checkpoint(r, a) == 1.0]
    if not loss_rows:
        raise ValueError(
            f"{stem}: no decisive A losses in the focus window (a_color="
            f"{a_clr}, moves {args.min_moves}-{args.max_moves}) — "
            "nothing to explain")

    feats = {"loss": [], "win": []}
    plies_games = {"loss": [], "win": []}
    cluster_games = []
    loss_pairs = []
    for cohort, cohort_rows_in in (("loss", loss_rows), ("win", win_rows)):
        for r in cohort_rows_in:
            replay = load_replay(r)
            f = game_features(r, replay, a_clr, th, args.opening_key_plies)
            f["cohort"] = cohort
            label, flags = classify_collapse(f, th)
            f["collapse_type"] = label
            f.update(flags)
            if cohort == "loss":
                f.update(b_side_features(
                    replay, b_clr, th, f["first_a_value_below_lost_fraction"]))
                loss_pairs.append((replay, f))
            feats[cohort].append(f)
            plies_games[cohort].append((side_plies(replay, a_clr), r["n_moves"]))
            cluster_games.append((replay, a_clr, cohort == "win"))

    cohort_rows = [
        cohort_comparison_row(c, [g for g, _n in plies_games[c]],
                              th.opening_plies)
        for c in ("loss", "win") if plies_games[c]]
    phase_rows = [row for c in ("loss", "win") if plies_games[c]
                  for row in phase_bucket_rows(c, plies_games[c],
                                               th.opening_plies)]
    cohort_desc = f"A-as-{a_clr} {args.min_moves}-{args.max_moves}"
    verdict = make_verdict([f["collapse_type"] for f in feats["loss"]],
                           cohort_desc)
    summary = build_replay_summary(
        match=stem, pairing_id=rows[0]["pairing_id"], a_ckpt=a, b_ckpt=b,
        filters={"a_color": a_clr, "min_moves": args.min_moves,
                 "max_moves": args.max_moves,
                 "opening_key_plies": args.opening_key_plies, **asdict(th)},
        counts={"focus_window_games": len(window),
                "excluded_draws": len(window) - len(decisive),
                "loss": len(loss_rows), "win": len(win_rows)},
        loss_feats=feats["loss"], win_feats=feats["win"], verdict=verdict,
        cohort_rows=cohort_rows,
        secondary=secondary_contrast_summary(feats["loss"]))
    return {
        "stem": stem, "summary": summary, "feats": feats,
        "cohort_rows": cohort_rows, "phase_rows": phase_rows,
        "queue": review_queue_rows(feats["loss"], args.review_queue),
        "clusters": opening_cluster_rows(
            cluster_games, args.opening_key_plies,
            f"A_{a_clr}_{args.min_moves}_{args.max_moves}_decisive",
            th.opening_plies),
        "drop_windows": drop_window_rows(loss_pairs),
    }


def timing_csv_rows(feats):
    """One row per focus game, loss rows first so the CSV header carries the
    B-side columns; win rows get blanks for those."""
    rows = feats["loss"] + feats["win"]
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    return [{k: r.get(k) for k in keys} for r in rows]


def write_outputs(out_dir, res):
    stem = res["stem"]
    write_json(out_dir / f"{stem}_replay_summary.json", res["summary"])
    write_csv(out_dir / f"{stem}_cohort_comparison.csv", res["cohort_rows"])
    write_csv(out_dir / f"{stem}_phase_buckets.csv", res["phase_rows"])
    write_csv(out_dir / f"{stem}_collapse_timing.csv",
              timing_csv_rows(res["feats"]))
    write_csv(out_dir / f"{stem}_manual_review_queue.csv", res["queue"])
    write_csv(out_dir / f"{stem}_opening_clusters.csv", res["clusters"])
    write_csv(out_dir / f"{stem}_drop_windows.csv", res["drop_windows"])


def print_console_summary(res, out_dir):
    s = res["summary"]
    print("=" * 60)
    print(f"REPLAY LOSS ANALYSIS (V2): {s['match']}")
    print("=" * 60)
    f, c = s["filters"], s["cohorts"]
    print(f"Focus window: A as {f['a_color']}, {f['min_moves']}-"
          f"{f['max_moves']} moves -> {c['loss']} losses, {c['win']} wins "
          f"({c['excluded_draws']} draws excluded)")
    dist = s["collapse_type_distribution"]
    print("Collapse types (losses):")
    for lab, cnt in sorted(dist["counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {lab:<20} {cnt:>4}  ({cnt / dist['n']:.0%})")
    phase, nloss = dist["largest_drop_phase"], c["loss"]
    print("Sharp value drops:")
    for label, key in (("post-opening", "post_opening"), ("opening", "opening")):
        n = phase[key]
        print(f"  {label + ':':<13}{n:>4} / {nloss} = {n / nloss:.1%}")
    pc = s["primary_contrast"]
    if pc["effect_sizes"] is None:
        print(f"Effect sizes: {pc['note']} (win cohort < {MIN_WIN_COHORT})")
    else:
        print("Effect sizes (loss vs win, Cohen's d):")
        for name, e in pc["effect_sizes"]["metrics"].items():
            d = e["d"]
            d_s = "n/a" if d is None else f"{d:+.2f}"
            print(f"  {name:<34} d={d_s}")
    sec = s["secondary_contrast"]
    if sec["b_saw_it_first_share"] is not None:
        print(f"B saw the win first in {sec['b_saw_it_first_share']:.0%} of "
              f"losses (onset-gap games: {sec['onset_gap_games']})")
    print(f"Phase B verdict: {s['verdict']['narrative']}")
    print(f"Manual review queue: "
          f"{out_dir / (s['match'] + '_manual_review_queue.csv')}")


def main(argv=None):
    args = parse_args(argv)
    inputs = resolve_inputs(args)
    if not inputs:
        print("error: no input files (use --games-jsonl and/or --glob)",
              file=sys.stderr)
        return 2
    th = thresholds_from_args(args)
    for path in inputs:
        res = analyze_input(path, args, th)
        if res is None:
            continue
        write_outputs(args.output_dir, res)
        print_console_summary(res, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
