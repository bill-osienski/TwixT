"""CLI for the V2 Phase B replay-aware loss analyzer.

Reads Phase A capture data (*_games.jsonl rows carrying replay_path + per-game
replay sidecars), explains WHY checkpoint A loses in the focus window, and
writes six artifacts per match to --output-dir. All analysis lives in
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
    opening_cluster_rows, phase_bucket_rows, review_queue_rows,
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
