"""Generic fixed-position probe.

Re-evaluates fixed replay positions across checkpoints.

For black-defense probes:
  lower probe_black_root_value = better danger recognition
  higher positive probe_black_root_value = black overconfidence
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
from .position_probe_cases import (
    OVERVALUE_THRESHOLD,
    SEVERE_OVERVALUE_THRESHOLD,
    case_id,
    position_state,
    summarize,
    load_csv_manifest,
)

CASE_CSV_COLUMNS = (
    "checkpoint",
    "game_idx",
    "case_id",
    "case_rank",
    "position_ply",
    "side_to_move",
    "probe_black_root_value",
    "probe_top1_share",
    "black_overvalue",
    "severe_black_overvalue",
    "replay_path",
    "drop_ply",
    "initial_a_value",
    "final_a_value",
    "largest_a_value_drop",
    "largest_drop_phase",
    "collapse_type",
)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def load_manifest(path):
    return load_csv_manifest(path)


def evaluate_case(evaluator, case, mcts_cfg, base_seed):
    """Reconstruct the case position and search it -> black_value, top1_share."""
    replay_path = Path(case["replay_path"])
    if not replay_path.exists():
        raise FileNotFoundError(f"{case_id(case)}: replay not found: {replay_path}")

    replay = json.loads(replay_path.read_text())
    state = position_state(replay, case["position_ply"], case["side_to_move"])

    rng = random.Random(base_seed ^ int(case["game_idx"]) ^ int(case["position_ply"]))
    counts, root_value = MCTS(evaluator, mcts_cfg, rng).search(state, add_noise=False)

    total = sum(counts.values())
    if total <= 0:
        raise ValueError(f"{case_id(case)}: empty search counts")

    top1_share = max(counts.values()) / total

    # MCTS root_value is from side-to-move perspective. Normalize to black.
    if state.to_move == "black":
        black_value = root_value
    elif state.to_move == "red":
        black_value = -root_value
    else:
        raise ValueError(f"{case_id(case)}: unexpected side to move {state.to_move!r}")

    return black_value, top1_share


def _checkpoint_labels(checkpoints):
    """One label per checkpoint; disambiguate duplicate iteration numbers."""
    sids = [short_id(c) for c in checkpoints]
    if len(set(sids)) == len(sids):
        return sids
    return [f"{Path(c).parent.name}:{s}" for c, s in zip(checkpoints, sids)]


def run_probe(manifest, checkpoints, config, base_seed, evaluator_factory):
    cases = manifest["cases"]
    mcts_cfg = cfg_from(config)
    labels = _checkpoint_labels(checkpoints)

    per_ckpt = {}
    case_rows = []

    for ckpt, sid in zip(checkpoints, labels):
        evaluator = evaluator_factory(ckpt)
        values = []
        shares = []

        for case in cases:
            v, t1 = evaluate_case(evaluator, case, mcts_cfg, base_seed)
            values.append(v)
            shares.append(t1)

            case_rows.append({
                "checkpoint": sid,
                "game_idx": case["game_idx"],
                "case_id": case_id(case),
                "case_rank": case.get("case_rank", case.get("rank")),
                "position_ply": case["position_ply"],
                "side_to_move": case["side_to_move"],
                "probe_black_root_value": v,
                "probe_top1_share": t1,
                "black_overvalue": v >= OVERVALUE_THRESHOLD,
                "severe_black_overvalue": v >= SEVERE_OVERVALUE_THRESHOLD,
                "replay_path": case.get("replay_path"),
                "drop_ply": case.get("drop_ply"),
                "initial_a_value": case.get("initial_a_value"),
                "final_a_value": case.get("final_a_value"),
                "largest_a_value_drop": case.get("largest_a_value_drop"),
                "largest_drop_phase": case.get("largest_drop_phase"),
                "collapse_type": case.get("collapse_type"),
            })

        per_ckpt[sid] = summarize(values, shares)

    summary = {
        "manifest": manifest.get("name"),
        "source": manifest.get("source"),
        "num_cases": len(cases),
        "mcts_sims": config.mcts_sims,
        "base_seed": base_seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "checkpoints": per_ckpt,
    }
    return summary, case_rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generic fixed-position calibration probe.")
    p.add_argument("--manifest", required=True)
    p.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        required=True,
        dest="checkpoints",
        metavar="PATH",
    )
    p.add_argument("--output-dir", type=Path, default=Path("logs/eval/position_probe"))
    p.add_argument("--mcts-sims", type=int, default=400)
    p.add_argument("--mcts-eval-batch-size", type=int, default=14)
    p.add_argument("--mcts-stall-flush-sims", type=int, default=48)
    p.add_argument("--base-seed", type=int, default=20260616)
    return p.parse_args(argv)


def main(argv=None, evaluator_factory=None):
    args = parse_args(argv)
    factory = evaluator_factory or _default_evaluator_factory

    for ckpt in args.checkpoints:
        if not Path(ckpt).exists():
            print(f"error: checkpoint not found: {ckpt}", file=sys.stderr)
            return 2

    if not Path(args.manifest).exists():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    manifest = load_manifest(args.manifest)
    config = EvalConfig(
        mcts_sims=args.mcts_sims,
        mcts_eval_batch_size=args.mcts_eval_batch_size,
        mcts_stall_flush_sims=args.mcts_stall_flush_sims,
    )

    summary, case_rows = run_probe(
        manifest, args.checkpoints, config, args.base_seed, factory
    )

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    summary_path = out / "position_probe_summary.json"
    cases_path = out / "position_probe_cases.csv"

    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    with cases_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CASE_CSV_COLUMNS))
        w.writeheader()
        w.writerows(case_rows)

    for sid, m in summary["checkpoints"].items():
        print(
            f"{sid}: overvalue_rate={m['black_overvalue_rate']:.1%} "
            f"mean_black_value={m['mean_black_root_value']:+.3f} "
            f"(severe {m['severe_black_overvalue_rate']:.1%})"
        )

    print(f"summary -> {summary_path}")
    print(f"cases   -> {cases_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
