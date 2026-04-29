#!/usr/bin/env python
"""Manual verifier for serial-vs-process Phase 2 equivalence on real MLX.

NOT in CI. Run by hand on a real machine with a real checkpoint to validate
that --label-worker-mode=process produces the same admitted probe IDs and
final committed probe IDs as serial mode, with phase2_label numeric fields
within tolerance.

Usage:
    .venv/bin/python scripts/probes/verify_parallel_equivalence.py \\
        --input scripts/GPU/logs/games \\
        --source-iter-range 57 58 \\
        --label-checkpoint checkpoints/.../model_iter_0059.safetensors \\
        --sample-candidates 20 \\
        --label-workers 4
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> int:
    print("[verify] $", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--source-iter-range", nargs=2, type=int, required=True,
                    metavar=("MIN", "MAX"))
    ap.add_argument("--label-checkpoint", required=True)
    ap.add_argument("--label-mcts-sims", type=int, default=2000)
    ap.add_argument("--label-mcts-repeats", type=int, default=2)
    ap.add_argument("--sample-candidates", type=int, default=20,
                    help="--max-probes for the equivalence run")
    ap.add_argument("--label-workers", type=int, default=4)
    ap.add_argument("--workdir", default="/tmp/probe_parallel_verify")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    py = sys.executable
    script = str(Path(__file__).resolve().parents[1].parent
                 / "scripts" / "build_probe_suite.py")

    common = [
        py, script,
        "--tier", "strong_advantage",
        "--input", args.input,
        "--source-iter-range", str(args.source_iter_range[0]),
                                str(args.source_iter_range[1]),
        "--label-checkpoint", args.label_checkpoint,
        "--label-mcts-sims", str(args.label_mcts_sims),
        "--label-mcts-repeats", str(args.label_mcts_repeats),
        "--max-probes", str(args.sample_candidates),
        "--force",
    ]

    serial_out = workdir / "serial.json"
    process_out = workdir / "process.json"
    rc1 = _run(common + ["--out", str(serial_out),
                          "--label-worker-mode", "serial"])
    rc2 = _run(common + ["--out", str(process_out),
                          "--label-worker-mode", "process",
                          "--label-workers", str(args.label_workers)])
    if rc1 != 0 or rc2 != 0:
        print("[verify] one of the runs failed", file=sys.stderr)
        return 1

    serial = json.loads((workdir / "serial.draft.json").read_text())
    process = json.loads((workdir / "process.draft.json").read_text())

    serial_ids = {p["id"] for p in serial["probes"]}
    process_ids = {p["id"] for p in process["probes"]}

    serial_by_id = {p["id"]: p for p in serial["probes"]}
    process_by_id = {p["id"]: p for p in process["probes"]}

    def _max_diff(field):
        diffs = []
        for pid in serial_ids & process_ids:
            sl = serial_by_id[pid]["phase2_label"]
            pl = process_by_id[pid]["phase2_label"]
            if isinstance(sl[field], list):
                diffs.extend(abs(a - b) for a, b in zip(sl[field], pl[field]))
            else:
                diffs.append(abs(sl[field] - pl[field]))
        return max(diffs) if diffs else 0.0

    print()
    print(f"serial_admitted_ids == process_admitted_ids: "
          f"{serial_ids == process_ids}")
    print(f"serial_final_ids    == process_final_ids:    "
          f"{serial_ids == process_ids}")
    print(f"max_abs_mean_root_value_diff:  {_max_diff('mean_root_value'):.6f}")
    print(f"max_abs_value_per_run_diff:    {_max_diff('value_per_run'):.6f}")
    print(f"max_abs_min_top1_share_diff:   {_max_diff('min_top1_share'):.6f}")
    process_stats = process["meta"]["phase2_run_stats"]
    print(f"borderline_reruns: {process_stats['borderline_reruns']}")
    print(f"borderline_flips:  {process_stats['borderline_flips']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
