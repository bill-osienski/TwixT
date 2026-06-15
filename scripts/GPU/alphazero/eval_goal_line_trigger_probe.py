"""Goal-line trigger probe: re-evaluate fixed trigger positions across
checkpoints, measuring whether each overvalues black before red's goal-line
trigger move. Lower black root_value = better calibrated.

Run:
  .venv/bin/python -m scripts.GPU.alphazero.eval_goal_line_trigger_probe \
    --manifest logs/eval/loss_analysis_v2_1/goal_line_trigger_probe_manifest.json \
    --checkpoint checkpoints/.../model_iter_0379.safetensors \
    --checkpoint checkpoints/.../model_iter_0399.safetensors \
    --output-dir logs/eval/goal_line_trigger_probe --mcts-sims 400
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .eval_runner import (
    EvalConfig, cfg_from, short_id, _default_evaluator_factory,
)
from .mcts import MCTS
from .goal_line_trigger_probe_cases import (
    OVERVALUE_THRESHOLD, SEVERE_OVERVALUE_THRESHOLD, case_id, position_state,
    summarize,
)

REQUIRED_CASE_KEYS = ("game_idx", "replay_path", "position_ply", "side_to_move",
                      "trigger_zone", "baseline_black_prev_value",
                      "baseline_black_prev_top1")
CASE_CSV_COLUMNS = (
    "checkpoint", "game_idx", "case_id", "rank", "position_ply", "trigger_zone",
    "side_to_move", "baseline_black_prev_value", "baseline_black_prev_top1",
    "probe_black_root_value", "probe_top1_share", "black_overvalue",
    "severe_black_overvalue")


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"manifest schema_version != 1: {manifest.get('schema_version')}")
    cases = manifest.get("cases") or []
    if not cases:
        raise ValueError(f"manifest has no cases: {path}")
    for i, c in enumerate(cases):
        missing = [k for k in REQUIRED_CASE_KEYS if k not in c]
        if missing:
            raise ValueError(f"case {i}: missing keys {missing}")
    return manifest


def evaluate_case(evaluator, case, mcts_cfg, base_seed):
    """Reconstruct the case position and search it -> (black_value, top1_share)."""
    replay = json.loads(Path(case["replay_path"]).read_text())
    state = position_state(replay, case["position_ply"], case["side_to_move"])
    rng = random.Random(base_seed ^ case["game_idx"])
    counts, root_value = MCTS(evaluator, mcts_cfg, rng).search(state, add_noise=False)
    total = sum(counts.values())
    if total <= 0:
        raise ValueError(f"{case_id(case)}: empty search counts")
    return root_value, max(counts.values()) / total


def run_probe(manifest, checkpoints, config, base_seed, evaluator_factory):
    """Evaluate every case with every checkpoint -> (summary, case_rows)."""
    cases = manifest["cases"]
    mcts_cfg = cfg_from(config)                     # built once, reused for every search
    per_ckpt, case_rows = {}, []
    for ckpt in checkpoints:
        evaluator = evaluator_factory(ckpt)        # one load, reused across cases
        sid = short_id(ckpt)
        values, shares = [], []
        for case in cases:
            v, t1 = evaluate_case(evaluator, case, mcts_cfg, base_seed)
            values.append(v)
            shares.append(t1)
            case_rows.append({
                "checkpoint": sid, "game_idx": case["game_idx"],
                "case_id": case_id(case), "rank": case.get("rank"),
                "position_ply": case["position_ply"],
                "trigger_zone": case["trigger_zone"],
                "side_to_move": case["side_to_move"],
                "baseline_black_prev_value": case["baseline_black_prev_value"],
                "baseline_black_prev_top1": case["baseline_black_prev_top1"],
                "probe_black_root_value": v, "probe_top1_share": t1,
                "black_overvalue": v >= OVERVALUE_THRESHOLD,
                "severe_black_overvalue": v >= SEVERE_OVERVALUE_THRESHOLD,
            })
        per_ckpt[sid] = summarize(values, shares)
    summary = {
        "manifest": manifest.get("name"),
        "num_cases": len(cases),
        "mcts_sims": config.mcts_sims,
        "base_seed": base_seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "checkpoints": per_ckpt,
    }
    return summary, case_rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Goal-line trigger calibration probe.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", action="append", default=[], required=True,
                   dest="checkpoints", metavar="PATH")
    p.add_argument("--output-dir", type=Path,
                   default=Path("logs/eval/goal_line_trigger_probe"))
    p.add_argument("--mcts-sims", type=int, default=400)
    p.add_argument("--mcts-eval-batch-size", type=int, default=14)
    p.add_argument("--mcts-stall-flush-sims", type=int, default=48)
    p.add_argument("--base-seed", type=int, default=20260614)
    return p.parse_args(argv)


def main(argv=None, evaluator_factory=None):
    args = parse_args(argv)
    factory = evaluator_factory or _default_evaluator_factory
    for ckpt in args.checkpoints:               # fail before the long MLX load
        if not Path(ckpt).exists():
            print(f"error: checkpoint not found: {ckpt}", file=sys.stderr)
            return 2
    manifest = load_manifest(args.manifest)
    config = EvalConfig(mcts_sims=args.mcts_sims,
                        mcts_eval_batch_size=args.mcts_eval_batch_size,
                        mcts_stall_flush_sims=args.mcts_stall_flush_sims)
    summary, case_rows = run_probe(manifest, args.checkpoints, config,
                                   args.base_seed, factory)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "goal_line_trigger_probe_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    with (out / "goal_line_trigger_probe_cases.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CASE_CSV_COLUMNS))
        w.writeheader()
        w.writerows(case_rows)
    for sid, m in summary["checkpoints"].items():
        print(f"{sid}: overvalue_rate={m['black_overvalue_rate']:.1%} "
              f"mean_black_value={m['mean_black_root_value']:+.3f} "
              f"(severe {m['severe_black_overvalue_rate']:.1%})")
    print(f"summary -> {out / 'goal_line_trigger_probe_summary.json'}")
    print(f"cases   -> {out / 'goal_line_trigger_probe_cases.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
