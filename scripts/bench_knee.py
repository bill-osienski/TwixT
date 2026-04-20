#!/usr/bin/env python3
"""
Benchmark runner to find the 'knee' for parallel self-play workers.

Runs training with different worker counts, measures selfplay positions/sec,
and finds the optimal --n-workers value where diminishing returns start.

Usage:
  python scripts/bench_knee.py --board 16 --sims 200 --games 8 --repeats 2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# =============================================================================
# Regex Patterns (use LAST match found in stdout)
# =============================================================================

RE_POSITIONS = re.compile(r"Generated\s+(\d+)\s+games,\s+(\d+)\s+positions")
RE_SELFPLAY = re.compile(r"\bselfplay=(\d+(?:\.\d+)?)s\b")
RE_AVG_BATCH = re.compile(r"Avg batch:\s*(\d+(?:\.\d+)?)")
RE_WAITERS = re.compile(r"Avg waiters:\s*(\d+(?:\.\d+)?),\s*Max waiters:\s*(\d+)")
RE_FLUSHES = re.compile(r"Flushes:\s*full=(\d+),\s*stall=(\d+),\s*tail=(\d+)")
RE_AVG_PLIES = re.compile(r"Avg plies:\s*(\d+(?:\.\d+)?)")
RE_NN_BATCHES = re.compile(r"NN batches:\s*(\d+)")
RE_RESULTS = re.compile(r"Results:\s*Red=(\d+),\s*Black=(\d+),\s*Draws=(\d+)")
RE_DRAW_BREAKDOWN = re.compile(
    r"Draw breakdown:\s*timeout=(\d+),\s*board_full=(\d+),\s*state_cap=(\d+),\s*unknown=(\d+)"
)
RE_WORKERS_MODE = re.compile(r"Workers:\s*(\d+)\s*\((parallel|sequential)\)")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RunResult:
    """Result from a single benchmark run."""
    workers: int
    positions: int = 0
    games: int = 0
    selfplay_s: float = 0.0
    pos_per_s: float = 0.0
    pos_per_game: float = 0.0
    nn_batches: int = 0
    nn_batches_per_pos: float = 0.0
    avg_batch: float = 0.0
    avg_waiters: float = 0.0
    max_waiters: int = 0
    flush_full: int = 0
    flush_stall: int = 0
    flush_tail: int = 0
    stall_ratio: float = 0.0
    red_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    timeout_draws: int = 0
    timeout_rate: float = 0.0
    draw_rate: float = 0.0
    avg_plies: float = 0.0
    ok: bool = False
    parallel_mode: bool = True
    shape_changed: bool = False
    error_msg: str = ""


@dataclass
class WorkerSummary:
    """Summary statistics for a worker count across repeats."""
    workers: int
    median_pos_per_s: float = 0.0
    min_pos_per_s: float = 0.0
    max_pos_per_s: float = 0.0
    cv: float = 0.0  # Coefficient of variation
    median_pos_per_game: float = 0.0
    median_avg_batch: float = 0.0
    median_avg_waiters: float = 0.0
    median_max_waiters: int = 0
    median_stall_ratio: float = 0.0
    median_timeout_rate: float = 0.0
    valid_repeats: int = 0
    excluded: bool = False
    exclude_reason: str = ""


# =============================================================================
# Helper Functions
# =============================================================================

def get_git_hash() -> str:
    """Get short git hash, or 'unknown' if not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def default_worker_sweep() -> List[int]:
    """Generate smart worker sweep based on CPU count."""
    cpu = os.cpu_count() or 4
    candidates = [
        1, 2, 3, 4, 5,        # small integers
        6, 8, 9,               # round numbers
        max(1, cpu - 4),       # near saturation
        max(1, cpu - 2),
        max(1, cpu - 1),
        cpu,
    ]
    # Add 7 for Apple Silicon (P-core sweet spot)
    if cpu >= 10:
        candidates.append(7)
    return sorted({w for w in candidates if 1 <= w <= cpu})


def parse_workers(value: str) -> List[int]:
    """Parse --workers argument: 'auto' or comma-separated list."""
    if value.lower() == "auto":
        return default_worker_sweep()
    return sorted(set(int(x.strip()) for x in value.split(",")))


def build_train_cmd(
    board: int,
    sims: int,
    games: int,
    eval_batch: int,
    workers: int,
    tmp_dir: str,
    load_weights: Optional[str] = None,
) -> List[str]:
    """Build benchmark-pure training command."""
    cmd = [
        sys.executable, "-m", "scripts.GPU.alphazero.train",
        "--iterations", "1",
        "--train-steps", "0",
        "--games-per-iter", str(games),
        "--curriculum-sizes", str(board),
        "--simulations", str(sims),
        "--mcts-eval-batch-size", str(eval_batch),
        "--n-workers", str(workers),
        "--checkpoint-dir", tmp_dir,
    ]
    if load_weights:
        cmd.extend(["--load-weights", load_weights])
    return cmd


def parse_output(text: str, workers: int, games_expected: int) -> RunResult:
    """Parse combined stdout+stderr to extract metrics."""
    result = RunResult(workers=workers)

    # Use LAST match for each pattern (in case of multiple prints)
    def last_match(pattern, text):
        matches = list(pattern.finditer(text))
        return matches[-1] if matches else None

    # Positions and games
    m = last_match(RE_POSITIONS, text)
    if m:
        result.games = int(m.group(1))
        result.positions = int(m.group(2))

    # Selfplay time
    m = last_match(RE_SELFPLAY, text)
    if m:
        result.selfplay_s = float(m.group(1))

    # NN batches
    m = last_match(RE_NN_BATCHES, text)
    if m:
        result.nn_batches = int(m.group(1))

    # Avg batch
    m = last_match(RE_AVG_BATCH, text)
    if m:
        result.avg_batch = float(m.group(1))

    # Waiters
    m = last_match(RE_WAITERS, text)
    if m:
        result.avg_waiters = float(m.group(1))
        result.max_waiters = int(m.group(2))

    # Flushes
    m = last_match(RE_FLUSHES, text)
    if m:
        result.flush_full = int(m.group(1))
        result.flush_stall = int(m.group(2))
        result.flush_tail = int(m.group(3))
        total_flushes = result.flush_full + result.flush_stall + result.flush_tail
        if total_flushes > 0:
            result.stall_ratio = result.flush_stall / total_flushes

    # Avg plies
    m = last_match(RE_AVG_PLIES, text)
    if m:
        result.avg_plies = float(m.group(1))

    # Results (wins/draws)
    m = last_match(RE_RESULTS, text)
    if m:
        result.red_wins = int(m.group(1))
        result.black_wins = int(m.group(2))
        result.draws = int(m.group(3))

    # Draw breakdown
    m = last_match(RE_DRAW_BREAKDOWN, text)
    if m:
        result.timeout_draws = int(m.group(1))

    # Workers mode validation
    m = last_match(RE_WORKERS_MODE, text)
    if m:
        mode = m.group(2)
        result.parallel_mode = (mode == "parallel") if workers > 1 else True
    else:
        # If we can't find the mode line, assume it's valid
        result.parallel_mode = True

    # Compute derived metrics
    if result.selfplay_s > 0:
        result.pos_per_s = result.positions / result.selfplay_s
    if result.games > 0:
        result.pos_per_game = result.positions / result.games
        result.timeout_rate = result.timeout_draws / result.games
        result.draw_rate = result.draws / result.games
    if result.positions > 0 and result.nn_batches > 0:
        result.nn_batches_per_pos = result.nn_batches / result.positions

    # Validity check
    result.ok = (
        result.positions > 0 and
        result.selfplay_s > 0 and
        result.parallel_mode
    )

    return result


def run_one(
    workers: int,
    board: int,
    sims: int,
    games: int,
    eval_batch: int,
    load_weights: Optional[str],
    timeout_s: int,
    keep_logs: bool,
    log_dir: Path,
    timestamp: str,
    rep: int,
) -> RunResult:
    """Run a single benchmark iteration."""
    # Create temp checkpoint dir
    tmp_dir = tempfile.mkdtemp(prefix=f"bench_knee_w{workers}_")
    cmd = build_train_cmd(board, sims, games, eval_batch, workers, tmp_dir, load_weights)
    cmd_str = " ".join(cmd)
    cmd_hash = hashlib.sha1(cmd_str.encode()).hexdigest()[:8]

    # Environment for repeatability
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    start_time = time.time()
    stdout = ""
    stderr = ""
    return_code = -1

    try:
        # Launch in own process group for clean timeout handling
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )

        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
            return_code = proc.returncode
        except subprocess.TimeoutExpired:
            # Kill process group
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
                    proc.wait(timeout=5)
            except (ProcessLookupError, OSError):
                pass  # Process already exited
            time.sleep(0.3)  # Cleanup delay
            stdout = proc.stdout.read() if proc.stdout else ""
            stderr = proc.stderr.read() if proc.stderr else ""
            return RunResult(workers=workers, ok=False, error_msg="TIMEOUT")

    except Exception as e:
        return RunResult(workers=workers, ok=False, error_msg=str(e))
    finally:
        # Cleanup checkpoint dir
        if not keep_logs:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    end_time = time.time()

    # Parse combined output
    combined = stdout + "\n" + stderr
    result = parse_output(combined, workers, games)

    # Save log if parse failed or if keeping logs
    if not result.ok or keep_logs:
        log_file = log_dir / f"bench_knee_{timestamp}_w{workers}_r{rep}.log"
        with open(log_file, "w") as f:
            f.write(f"Command: {cmd_str}\n")
            f.write(f"cmd_hash: {cmd_hash}\n")
            f.write(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
            f.write(f"End: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
            f.write(f"Return code: {return_code}\n")
            f.write(f"ok: {result.ok}\n")
            f.write("\n=== STDOUT ===\n")
            f.write(stdout)
            f.write("\n=== STDERR ===\n")
            f.write(stderr)

    return result


def median(values: List[float]) -> float:
    """Compute median of a list."""
    if not values:
        return 0.0
    return statistics.median(values)


def cv(values: List[float]) -> float:
    """Compute coefficient of variation (stdev/mean)."""
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0
    return statistics.stdev(values) / mean


def find_knee(
    worker_points: List[Tuple[int, float]],
    min_gain: float = 0.07,
    consecutive: int = 2,
    slack: float = 0.05,
) -> int:
    """
    Find the knee in the throughput curve.

    Definition: Knee = first index where a run of `consecutive` gains falls
    below `min_gain`, returning the point immediately before that run began.
    """
    if len(worker_points) < 2:
        return worker_points[0][0] if worker_points else 1

    # Monotonic smoothing (prevents noisy dip from faking knee)
    smoothed = []
    for w, t in worker_points:
        if smoothed:
            t = max(t, smoothed[-1][1])
        smoothed.append((w, t))

    # Find best
    best_w, best_t = max(smoothed, key=lambda x: x[1])
    below = 0
    knee_w = best_w  # Default to best if no knee found

    for i in range(1, len(smoothed)):
        prev_t = smoothed[i - 1][1]
        curr_t = smoothed[i][1]
        gain = (curr_t - prev_t) / max(prev_t, 1e-9)

        if gain < min_gain:
            below += 1
        else:
            below = 0

        if below >= consecutive:
            # Knee is at i-consecutive (last point BEFORE the diminishing run)
            knee_w = smoothed[i - consecutive][0]
            break

    # Check knee is within slack% of best
    knee_t = dict(smoothed).get(knee_w, 0)
    if knee_t < (1 - slack) * best_t:
        return best_w

    return knee_w


def print_summary_table(summaries: List[WorkerSummary], knee_w: int) -> None:
    """Print ASCII summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    header = f"{'workers':>7}  {'pos/s [min..max]':>22}  {'pos/game':>8}  {'avg_batch':>9}  {'stall':>6}  {'timeout':>7}"
    print(header)
    print("-" * 80)

    for s in summaries:
        knee_marker = " <- knee" if s.workers == knee_w else ""
        if s.excluded:
            knee_marker = f" ({s.exclude_reason})"
        spread = f"{s.median_pos_per_s:.1f} [{s.min_pos_per_s:.1f}..{s.max_pos_per_s:.1f}]"
        print(
            f"{s.workers:>7}  {spread:>22}  {s.median_pos_per_game:>8.1f}  "
            f"{s.median_avg_batch:>9.1f}  {s.median_stall_ratio:>6.2f}  "
            f"{s.median_timeout_rate:>7.2f}{knee_marker}"
        )


def write_csv(
    csv_path: Path,
    results: List[RunResult],
    args: argparse.Namespace,
    git_hash: str,
    timestamp: str,
) -> None:
    """Append results to CSV file."""
    # Ensure directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if we need to write header
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp", "git_hash", "cpu_count", "board", "sims", "games",
                "max_moves", "eval_batch", "workers", "positions", "selfplay_s",
                "pos_per_s", "pos_per_game", "nn_batches", "nn_batches_per_pos",
                "avg_batch", "avg_waiters", "max_waiters", "flush_full",
                "flush_stall", "flush_tail", "stall_ratio", "red_wins",
                "black_wins", "draws", "timeout_draws", "timeout_rate",
                "draw_rate", "avg_plies", "resume_model", "cmd_hash", "ok"
            ])

        resume_model = Path(args.load_weights).name if args.load_weights else ""
        cmd_base = f"{args.board}_{args.sims}_{args.games}_{args.eval_batch}"

        for r in results:
            cmd_hash = hashlib.sha1(f"{cmd_base}_{r.workers}".encode()).hexdigest()[:8]
            writer.writerow([
                timestamp, git_hash, os.cpu_count(), args.board, args.sims,
                args.games, args.max_moves, args.eval_batch, r.workers,
                r.positions, f"{r.selfplay_s:.3f}", f"{r.pos_per_s:.3f}",
                f"{r.pos_per_game:.1f}", r.nn_batches, f"{r.nn_batches_per_pos:.4f}",
                f"{r.avg_batch:.2f}", f"{r.avg_waiters:.2f}", r.max_waiters,
                r.flush_full, r.flush_stall, r.flush_tail, f"{r.stall_ratio:.4f}",
                r.red_wins, r.black_wins, r.draws, r.timeout_draws,
                f"{r.timeout_rate:.4f}", f"{r.draw_rate:.4f}", f"{r.avg_plies:.1f}",
                resume_model, cmd_hash, int(r.ok)
            ])


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark to find optimal --n-workers (knee detection)"
    )
    parser.add_argument("--board", type=int, default=16, help="Board/curriculum size")
    parser.add_argument("--sims", type=int, default=200, help="MCTS simulations")
    parser.add_argument("--games", type=int, default=8, help="Games per iteration")
    parser.add_argument("--max-moves", type=int, default=200, help="Max moves (logged only)")
    parser.add_argument("--eval-batch", type=int, default=14, help="MCTS eval batch size")
    parser.add_argument("--workers", type=str, default="auto", help="'auto' or comma-separated list")
    parser.add_argument("--repeats", type=int, default=2, help="Runs per worker count")
    parser.add_argument("--min-gain", type=float, default=0.07, help="Knee threshold (0.07=7%%)")
    parser.add_argument("--consecutive", type=int, default=2, help="Below-threshold steps for knee")
    parser.add_argument("--timeout-s", type=int, default=1800, help="Per-run timeout (seconds)")
    parser.add_argument("--max-timeout-rate", type=float, default=0.3, help="Max timeout rate before exclusion")
    parser.add_argument("--load-weights", type=str, default=None, help="Weights-only load (no training state restore)")
    parser.add_argument("--keep-logs", action="store_true", help="Keep temp checkpoint dirs")
    parser.add_argument("--csv", type=str, default="Benchmarks/bench_knee_results.csv", help="CSV output path")
    args = parser.parse_args()

    # Setup
    worker_sweep = parse_workers(args.workers)
    git_hash = get_git_hash()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path("Benchmarks/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv)

    print("=" * 60)
    print("BENCH KNEE: Worker Count Optimization")
    print("=" * 60)
    print(f"Board: {args.board}, Sims: {args.sims}, Games: {args.games}")
    print(f"Eval batch: {args.eval_batch}, Repeats: {args.repeats}")
    print(f"Worker sweep: {worker_sweep}")
    print(f"Timeout: {args.timeout_s}s, Max timeout rate: {args.max_timeout_rate}")
    print(f"Git hash: {git_hash}")
    print(f"Load weights: {args.load_weights or 'None'}")
    print()

    # Warmup run (compile kernels, non-fatal parse failures)
    print("Running warmup (compiling kernels)...")
    warmup_workers = max(worker_sweep)
    warmup_timeout = min(args.timeout_s // 2, 300)
    warmup_result = run_one(
        workers=warmup_workers,
        board=args.board,
        sims=args.sims,
        games=2,  # Fewer games for warmup
        eval_batch=args.eval_batch,
        load_weights=args.load_weights,
        timeout_s=warmup_timeout,
        keep_logs=args.keep_logs,
        log_dir=log_dir,
        timestamp=timestamp,
        rep=0,
    )
    if warmup_result.ok:
        print(f"  Warmup OK: {warmup_result.pos_per_s:.1f} pos/s")
    else:
        print(f"  Warmup completed (parse issues are non-fatal)")
    print()

    # Main benchmark loop
    all_results: List[RunResult] = []
    summaries: List[WorkerSummary] = []

    for w in worker_sweep:
        print(f"[w={w}] Running {args.repeats} repeats...")
        repeats: List[RunResult] = []
        retry_count = 0
        max_retries = 1

        for rep in range(args.repeats):
            result = run_one(
                workers=w,
                board=args.board,
                sims=args.sims,
                games=args.games,
                eval_batch=args.eval_batch,
                load_weights=args.load_weights,
                timeout_s=args.timeout_s,
                keep_logs=args.keep_logs,
                log_dir=log_dir,
                timestamp=timestamp,
                rep=rep + 1,
            )

            # Retry once on failure
            if not result.ok and retry_count < max_retries:
                print(f"  Repeat {rep + 1} failed, retrying...")
                retry_count += 1
                result = run_one(
                    workers=w,
                    board=args.board,
                    sims=args.sims,
                    games=args.games,
                    eval_batch=args.eval_batch,
                    load_weights=args.load_weights,
                    timeout_s=args.timeout_s,
                    keep_logs=args.keep_logs,
                    log_dir=log_dir,
                    timestamp=timestamp,
                    rep=rep + 1,
                )

            repeats.append(result)
            all_results.append(result)

            if result.ok:
                print(f"  Repeat {rep + 1}: {result.pos_per_s:.1f} pos/s, {result.pos_per_game:.1f} pos/game")
            else:
                print(f"  Repeat {rep + 1}: FAILED ({result.error_msg})")

        # Compute summary for this worker count
        valid_results = [r for r in repeats if r.ok]
        summary = WorkerSummary(workers=w, valid_repeats=len(valid_results))

        if valid_results:
            pos_per_s_values = [r.pos_per_s for r in valid_results]
            summary.median_pos_per_s = median(pos_per_s_values)
            summary.min_pos_per_s = min(pos_per_s_values)
            summary.max_pos_per_s = max(pos_per_s_values)
            summary.cv = cv(pos_per_s_values)
            summary.median_pos_per_game = median([r.pos_per_game for r in valid_results])
            summary.median_avg_batch = median([r.avg_batch for r in valid_results])
            summary.median_avg_waiters = median([r.avg_waiters for r in valid_results])
            summary.median_max_waiters = int(median([r.max_waiters for r in valid_results]))
            summary.median_stall_ratio = median([r.stall_ratio for r in valid_results])
            summary.median_timeout_rate = median([r.timeout_rate for r in valid_results])

            # Variance warning
            if summary.cv > 0.08:
                print(f"  WARNING: High variance (CV={summary.cv:.1%}), consider more repeats/games")

            # Timeout rate warning/exclusion
            if summary.median_timeout_rate > args.max_timeout_rate:
                summary.excluded = True
                summary.exclude_reason = "high timeout"
                print(f"  EXCLUDED: timeout rate {summary.median_timeout_rate:.1%} > {args.max_timeout_rate:.0%}")
        else:
            summary.excluded = True
            summary.exclude_reason = "all failed"
            print(f"  EXCLUDED: all repeats failed")

        summaries.append(summary)
        print()

    # Shape-change detection (baseline from smallest valid worker)
    valid_summaries = [s for s in summaries if not s.excluded]
    if valid_summaries:
        baseline_summary = min(valid_summaries, key=lambda s: s.workers)
        baseline_pos_per_game = baseline_summary.median_pos_per_game

        for s in valid_summaries:
            if baseline_pos_per_game > 0:
                deviation = abs(s.median_pos_per_game - baseline_pos_per_game) / baseline_pos_per_game
                if deviation > 0.30:
                    s.excluded = True
                    s.exclude_reason = "shape changed"
                    print(f"[w={s.workers}] EXCLUDED: pos_per_game deviation {deviation:.0%} > 30%")

    # Find knee from non-excluded points
    knee_points = [(s.workers, s.median_pos_per_s) for s in summaries if not s.excluded]
    if knee_points:
        knee_w = find_knee(knee_points, args.min_gain, args.consecutive)
        best_w, best_t = max(knee_points, key=lambda x: x[1])
        knee_t = dict(knee_points).get(knee_w, 0)
    else:
        knee_w = worker_sweep[0]
        best_w, best_t = knee_w, 0
        knee_t = 0

    # Print summary table
    print_summary_table(summaries, knee_w)

    # Print recommendation
    print("\n" + "=" * 60)
    print("KNEE RECOMMENDATION")
    print("=" * 60)
    if best_t > 0:
        pct_of_best = (knee_t / best_t) * 100 if best_t > 0 else 0
        print(f"  recommended --n-workers {knee_w}")
        print(f"  (throughput: {knee_t:.1f} pos/s, {pct_of_best:.0f}% of best)")
        print(f"  best throughput at workers={best_w} ({best_t:.1f} pos/s)")
    else:
        print(f"  No valid results - cannot recommend")
    print()

    # Write CSV
    write_csv(csv_path, all_results, args, git_hash, timestamp)
    print(f"CSV appended: {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
