#!/usr/bin/env python3
"""
Self-play throughput sweep on 24x24: find the best --n-workers, then the best
--mcts-eval-batch-size at that worker count.

Replaces the original bench_knee.py (workers only, 16x16 by default, random
weights; see git history before 2026-10-21). This version:

  * runs `scripts.GPU.alphazero.train --iterations 1 --train-steps 0` on 24x24
    with real weights (--load-weights), so games and batch patterns are real;
  * sends EVERYTHING the trainer writes (checkpoint, game replays, the
    per-iteration stats sidecar) into --out: nothing lands in the repository,
    and scripts/GPU/logs/games/ is never touched;
  * stage 1 sweeps workers at --start-eval-batch; stage 2 sweeps eval batch at
    stage 1's best worker count;
  * kills a hung run's whole process group (TERM, then KILL) and, in stage 2,
    skips every eval batch larger than the first one that hung.

Throughput only. A larger eval batch changes the search (more pending leaves,
different visit counts), so a stage-2 winner is not a behaviour-preserving
setting, and one clean sweep does not prove the Metal hang is gone. Games are
unseeded and their outcomes are not a strength measurement.

Usage:
  .venv/bin/python scripts/bench_knee.py \\
      --load-weights checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \\
      --out /path/outside/the/repo/selfplay-sweep

Exit: 0 completed, 2 usage, 4 refused before any run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOARD = 24                      # research is 24x24 only
FIELDS = {                      # trainer stdout -> metric (last match wins)
    "games_positions": re.compile(r"Generated\s+(\d+)\s+games,\s+(\d+)\s+positions"),
    "selfplay_s": re.compile(r"\bselfplay=(\d+(?:\.\d+)?)s\b"),
    "nn": re.compile(r"Leaf evals:\s*(\d+),\s*NN batches:\s*(\d+)"),
    "batch": re.compile(r"Avg batch:\s*([\d.]+),\s*Avg waiters:\s*([\d.]+),\s*Max waiters:\s*(\d+)"),
    "flushes": re.compile(r"Flushes:\s*full=(\d+),\s*stall=(\d+),\s*tail=(\d+)"),
    "avg_plies": re.compile(r"Avg plies:\s*([\d.]+)"),
    "results": re.compile(r"Results:\s*Red=(\d+),\s*Black=(\d+),\s*Draws=(\d+)"),
    "workers": re.compile(r"Workers:\s*(\d+)"),
}
CSV_COLUMNS = ["stage", "workers", "eval_batch", "repeat", "status", "wall_s", "selfplay_s",
               "games", "positions", "plies_per_s", "pos_per_s", "games_per_h", "s_per_game", "avg_plies",
               "leaf_evals", "nn_batches", "avg_batch", "avg_waiters", "max_waiters",
               "flush_full", "flush_stall", "flush_tail", "children_cpu_s", "red", "black",
               "draws", "workers_reported"]


def parse_list(text: str) -> list[int]:
    values = sorted({int(x) for x in text.split(",") if x.strip()})
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError(f"need positive integers: {text!r}")
    return values


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def contention() -> list[str]:
    """Other heavy jobs that would skew the timings."""
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    pats = ("-m pytest", "guarded_suite", "alphazero.train", "build_probe_suite", "benchmark_phase2")
    return [line.strip() for line in out.splitlines()
            if int(line.split(None, 1)[0]) != os.getpid() and any(p in line for p in pats)]


def hardware() -> dict:
    hw = {"platform": platform.platform(), "python": platform.python_version()}
    for key in ("machdep.cpu.brand_string", "hw.ncpu", "hw.memsize"):
        r = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
        hw[key] = r.stdout.strip()
    return hw


def train_cmd(args, workers: int, eval_batch: int, run_dir: Path) -> list[str]:
    return [
        sys.executable, "-m", "scripts.GPU.alphazero.train",
        "--iterations", "1", "--train-steps", "0",
        "--games-per-iter", str(args.games),
        "--curriculum-sizes", str(BOARD),
        "--simulations", str(args.sims),
        "--mcts-eval-batch-size", str(eval_batch),
        "--mcts-pending-virtual-visits", str(args.virtual_visits),
        "--mcts-stall-flush-sims", str(args.stall_flush),
        "--n-workers", str(workers),
        "--load-weights", str(args.load_weights),
        "--checkpoint-dir", str(run_dir / "checkpoint"),
        "--games-dir", str(run_dir / "games"),
        "--no-save-games",
        "--probes-inline-disable",
    ]


def parse(text: str) -> dict:
    def last(name):
        found = list(FIELDS[name].finditer(text))
        return found[-1].groups() if found else None

    m = {}
    if g := last("games_positions"):
        m["games"], m["positions"] = int(g[0]), int(g[1])
    if g := last("selfplay_s"):
        m["selfplay_s"] = float(g[0])
    if g := last("nn"):
        m["leaf_evals"], m["nn_batches"] = int(g[0]), int(g[1])
    if g := last("batch"):
        m["avg_batch"], m["avg_waiters"], m["max_waiters"] = float(g[0]), float(g[1]), int(g[2])
    if g := last("flushes"):
        m["flush_full"], m["flush_stall"], m["flush_tail"] = map(int, g)
    if g := last("avg_plies"):
        m["avg_plies"] = float(g[0])
    if g := last("results"):
        m["red"], m["black"], m["draws"] = map(int, g)
    if g := last("workers"):
        m["workers_reported"] = int(g[0])
    if m.get("selfplay_s") and m.get("positions") and m.get("avg_plies"):
        m["plies_per_s"] = m["avg_plies"] * m["games"] / m["selfplay_s"]   # 400-sim searches per second
        m["pos_per_s"] = m["positions"] / m["selfplay_s"]
        m["games_per_h"] = m["games"] / m["selfplay_s"] * 3600
        m["s_per_game"] = m["selfplay_s"] / m["games"]
    return m


def run_one(args, stage: str, workers: int, eval_batch: int, repeat: int, timeout: float) -> dict:
    run_dir = args.out / "runs" / f"{stage}_w{workers}_eb{eval_batch}_r{repeat}"
    run_dir.mkdir(parents=True)
    cmd = train_cmd(args, workers, eval_batch, run_dir)
    (run_dir / "command.txt").write_text(" ".join(cmd) + "\n")
    env = {**os.environ, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    row = {"stage": stage, "workers": workers, "eval_batch": eval_batch, "repeat": repeat}
    print(f"[bench] {stage}: workers={workers} eval_batch={eval_batch} repeat={repeat} "
          f"(timeout {timeout:.0f}s)", flush=True)

    cpu0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    with open(run_dir / "stdout.txt", "w") as out, open(run_dir / "stderr.txt", "w") as err:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out, stderr=err, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
            status = "OK" if rc == 0 else f"EXIT_{rc}"
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
            for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 10)):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=grace)
                    break
                except subprocess.TimeoutExpired:
                    continue
    cpu1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    row["wall_s"] = round(time.monotonic() - start, 2)
    row["children_cpu_s"] = round((cpu1.ru_utime - cpu0.ru_utime) + (cpu1.ru_stime - cpu0.ru_stime), 1)
    row.update(parse((run_dir / "stdout.txt").read_text() + "\n" + (run_dir / "stderr.txt").read_text()))
    if status == "OK" and "plies_per_s" not in row:
        status = "UNPARSED"
    row["status"] = status
    print(f"[bench]   -> {status}, wall {row['wall_s']}s"
          + (f", {row['plies_per_s']:.2f} plies/s, {row['games_per_h']:.0f} games/h, "
             f"avg batch {row.get('avg_batch', 0):.1f}" if "plies_per_s" in row else ""), flush=True)
    for name in ("checkpoint", "games"):          # weights copies are ~7.5 MB each; keep logs only
        shutil.rmtree(run_dir / name, ignore_errors=True)
    return row


def summarize(rows: list[dict], stage: str, key: str) -> dict[int, dict]:
    """{knob value: median metrics over OK repeats} for one stage."""
    out = {}
    for value in sorted({r[key] for r in rows if r["stage"] == stage}):
        ok = [r for r in rows if r["stage"] == stage and r[key] == value and r["status"] == "OK"]
        all_runs = [r for r in rows if r["stage"] == stage and r[key] == value]
        s = {"ok_runs": len(ok), "runs": len(all_runs),
             "statuses": sorted({r["status"] for r in all_runs})}
        for metric in ("plies_per_s", "pos_per_s", "games_per_h", "s_per_game", "avg_batch", "avg_plies"):
            vals = [r[metric] for r in ok if metric in r]
            if vals:
                s[metric] = statistics.median(vals)
                s[metric + "_range"] = (min(vals), max(vals))
        out[value] = s
    return out


def best(summary: dict[int, dict], within: float) -> tuple[int | None, int | None]:
    """(best value by median plies/s, smallest value within `within` of it)."""
    scored = {v: s["plies_per_s"] for v, s in summary.items() if "plies_per_s" in s}
    if not scored:
        return None, None
    top = max(scored, key=scored.get)
    knee = min(v for v, p in scored.items() if p >= scored[top] * (1 - within))
    return top, knee


def report(args, rows, s1, s2, picks, meta) -> str:
    lines = [f"# Self-play throughput sweep ({BOARD}x{BOARD}, {args.sims} sims, {args.games} games/run)", "",
             f"Machine: {meta['hardware'].get('machdep.cpu.brand_string')}, {meta['hardware'].get('hw.ncpu')} CPUs; "
             f"git {meta['git_head'][:8]}; weights {args.load_weights.name} sha1 {meta['weights_sha1'][:12]}", "",
             "Ranking metric: median self-play plies/s (400-sim searches per second, all workers) over OK "
             "repeats; positions/s counts the trainer's stored positions, which are not one per ply. "
             "Throughput only: a larger eval batch changes the search, and one clean sweep does not prove "
             "the Metal hang is gone.", ""]
    for title, summary, key in (("Stage 1: workers", s1, "workers"), ("Stage 2: eval batch", s2, "eval_batch")):
        lines += [f"## {title}", "", f"| {key} | OK/runs | plies/s (median, range) | pos/s | games/h | s/game | "
                  "avg batch | avg plies | statuses |", "|---|---|---|---|---|---|---|---|---|"]
        for v, s in summary.items():
            if "plies_per_s" in s:
                lo, hi = s["plies_per_s_range"]
                lines.append(f"| {v} | {s['ok_runs']}/{s['runs']} | {s['plies_per_s']:.2f} ({lo:.2f}-{hi:.2f}) | "
                             f"{s['pos_per_s']:.1f} | "
                             f"{s['games_per_h']:.0f} | {s['s_per_game']:.1f} | {s.get('avg_batch', 0):.1f} | "
                             f"{s.get('avg_plies', 0):.1f} | {', '.join(s['statuses'])} |")
            else:
                lines.append(f"| {v} | 0/{s['runs']} | - | - | - | - | - | - | {', '.join(s['statuses'])} |")
        lines.append("")
    lines += ["## Picks", "", json.dumps(picks, indent=2), ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load-weights", type=Path, required=True, help="checkpoint to self-play with")
    ap.add_argument("--out", type=Path, required=True, help="output directory (created; must not exist)")
    ap.add_argument("--workers", type=parse_list, default=parse_list("10,12,14,16,18"))
    ap.add_argument("--eval-batches", type=parse_list, default=parse_list("14,16,24,32,48"))
    ap.add_argument("--start-eval-batch", type=int, default=14, help="eval batch held during stage 1")
    ap.add_argument("--games", type=int, default=36, help="self-play games per run")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--sims", type=int, default=400, help="simulations per move (SIMS_TABLE[24] = 400)")
    ap.add_argument("--stall-flush", type=int, default=48, help="last training run's mcts_stall_flush_sims")
    ap.add_argument("--virtual-visits", type=int, default=8, help="last training run's pending virtual visits")
    ap.add_argument("--within", type=float, default=0.03, help="knee: smallest value within this of the best")
    ap.add_argument("--first-timeout", type=float, default=3600, help="seconds, the first (reference) run")
    ap.add_argument("--timeout-mult", type=float, default=3.0, help="later runs: x the reference run's wall")
    ap.add_argument("--min-timeout", type=float, default=300)
    ap.add_argument("--skip-stage2", action="store_true")
    args = ap.parse_args(argv)

    args.load_weights = args.load_weights.resolve()
    args.out = args.out.resolve()
    refusals = []
    if args.out.exists():
        refusals.append(f"{args.out} exists (create-only)")
    if args.out == ROOT or ROOT in args.out.parents:
        refusals.append(f"{args.out} is inside the repository")
    if not args.load_weights.is_file():
        refusals.append(f"{args.load_weights} is not a file")
    if busy := contention():
        refusals.append("other heavy jobs are running: " + " | ".join(busy))
    if refusals:
        for r in refusals:
            print(f"REFUSED: {r}", file=sys.stderr)
        return 4

    args.out.mkdir(parents=True)
    meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "hardware": hardware(),
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                       text=True).stdout.strip(),
            "weights": str(args.load_weights), "weights_sha1": sha1_of(args.load_weights),
            "args": {k: str(v) for k, v in vars(args).items()}}
    rows: list[dict] = []
    timeout = args.first_timeout

    def next_timeout(row):
        return max(args.min_timeout, row["wall_s"] * args.timeout_mult) if row["status"] == "OK" else timeout

    # Stage 1: workers at the start eval batch. The first OK run sets the timeout for every later run.
    for w in args.workers:
        for rep in range(1, args.repeats + 1):
            row = run_one(args, "workers", w, args.start_eval_batch, rep, timeout)
            rows.append(row)
            if timeout == args.first_timeout:
                timeout = next_timeout(row)
    s1 = summarize(rows, "workers", "workers")
    w_best, w_knee = best(s1, args.within)

    # Stage 2: eval batch at stage 1's best worker count; stop climbing after the first hang.
    s2, hung_at = {}, None
    if w_best is not None and not args.skip_stage2:
        for eb in args.eval_batches:
            if hung_at is not None and eb > hung_at:
                print(f"[bench] skip eval_batch={eb}: {hung_at} hung", flush=True)
                rows.append({"stage": "eval_batch", "workers": w_best, "eval_batch": eb, "repeat": 0,
                             "status": f"SKIPPED_AFTER_HANG_AT_{hung_at}"})
                continue
            for rep in range(1, args.repeats + 1):
                if eb == args.start_eval_batch:       # reuse stage 1's runs for the reference value
                    rows += [dict(r, stage="eval_batch") for r in rows
                             if r["stage"] == "workers" and r["workers"] == w_best and r["repeat"] == rep]
                    continue
                row = run_one(args, "eval_batch", w_best, eb, rep, timeout)
                rows.append(row)
                if row["status"] == "TIMEOUT":
                    hung_at = eb
                    break
        s2 = summarize(rows, "eval_batch", "eval_batch")
    eb_best, eb_knee = best(s2, args.within) if s2 else (None, None)

    picks = {"workers_best": w_best, "workers_knee": w_knee, "eval_batch_best": eb_best,
             "eval_batch_knee": eb_knee, "first_hang_eval_batch": hung_at, "within": args.within}
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(args.out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "record.json").write_text(json.dumps(
        {"meta": meta, "picks": picks, "stage1": s1, "stage2": s2, "rows": rows}, indent=2, default=str))
    text = report(args, rows, s1, s2, picks, meta)
    (args.out / "report.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
