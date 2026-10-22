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
  * runs the trainer unbuffered and keeps, per run, progress.txt (each stdout
    line with seconds since start) and samples.txt (GPU utilization and each
    process's CPU % and state every --sample-every seconds);
  * a run is TIMEOUT past its timeout, or STALLED after --stall-timeout seconds
    with no trainer output; either way its whole process group gets SIGABRT
    first (PYTHONFAULTHANDLER=1, so every process prints all its threads'
    stacks to stderr.txt), then TERM, then KILL. In stage 2 it skips every eval
    batch larger than the first one that timed out or stalled.

Ranking: steady-state games/h, the game rate while every worker is busy.
Total wall time for a few dozen games is dominated by whether one ~270-ply
game hits the state cap and then runs alone at the end (2026-10-21: the same
setting took 140-484 s, steady rate within +-4%), so by default each run is
stopped at the first "Worker done" line (status STEADY); --full runs to the end.

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
import threading
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
GAMES_LINE = re.compile(r"^\s*Games:\s*(\d+)/\d+")
WORKER_DONE = re.compile(r"^\s*Worker\s+\d+\s+done\b")
CSV_COLUMNS = ["stage", "workers", "eval_batch", "repeat", "status", "wall_s", "steady_games_per_h",
               "steady_games", "steady_from_s", "steady_to_s", "first_done_s", "selfplay_s",
               "games", "positions", "plies_per_s", "pos_per_s", "games_per_h", "s_per_game", "avg_plies",
               "leaf_evals", "nn_batches", "avg_batch", "avg_waiters", "max_waiters",
               "flush_full", "flush_stall", "flush_tail", "children_cpu_s", "red", "black",
               "draws", "workers_reported", "last_output_s", "last_games_line", "max_outside_cpu"]


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
    """Other heavy jobs that would skew the timings (this process and its ancestors excluded, so a
    shell that launched the sweep is not mistaken for one)."""
    out = subprocess.run(["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True).stdout
    procs = {int(pid): (int(ppid), cmd) for pid, ppid, cmd in
             (line.split(None, 2) for line in out.splitlines() if len(line.split(None, 2)) == 3)}
    mine, pid = set(), os.getpid()
    while pid in procs and pid not in mine:
        mine.add(pid)
        pid = procs[pid][0]
    pats = ("-m pytest", "guarded_suite", "alphazero.train", "build_probe_suite", "benchmark_phase2")
    return [f"{pid} {cmd}" for pid, (_, cmd) in procs.items()
            if pid not in mine and any(p in cmd for p in pats)]


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


def steady(events: list[tuple[float, int]], first_done: float | None, min_games: int) -> dict:
    """Game rate while every worker is busy: from the first "Games: n" line with n >= 5 to the last one
    at or before the first "Worker done" line. Leaves out start-up and the straggler tail."""
    if first_done is None:
        return {}
    window = [(t, n) for t, n in events if n >= 5 and t <= first_done]
    if len(window) < 2 or window[-1][1] - window[0][1] < min_games:
        return {}
    (t0, n0), (t1, n1) = window[0], window[-1]
    return {"steady_games": n1 - n0, "steady_from_s": round(t0, 1), "steady_to_s": round(t1, 1),
            "first_done_s": round(first_done, 1), "steady_games_per_h": (n1 - n0) / (t1 - t0) * 3600}


def pump(proc, out, prog, start: float, last: dict) -> None:
    """Copy the trainer's stdout to stdout.txt, and to progress.txt with seconds since start."""
    for line in proc.stdout:
        now = time.monotonic()
        last["t"] = now
        if m := GAMES_LINE.search(line):
            last["games"] = line.strip()
            last["events"].append((now - start, int(m.group(1))))
        elif "first_done" not in last and WORKER_DONE.search(line):
            last["first_done"] = now - start
        out.write(line)
        prog.write(f"{now - start:8.1f}  {line}")
        prog.flush()


def gpu_utilization() -> str:
    text = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True,
                          text=True).stdout
    m = re.search(r'"Device Utilization %"=(\d+)', text)
    return m.group(1) if m else "?"


def sample(pgid: int, samp, start: float, every: float, done, last: dict) -> None:
    """Every `every` seconds: GPU utilization, each process in the run's group (CPU %, state), and the
    CPU used by everything outside the group (total and top three), to catch other jobs competing."""
    me = os.getpid()
    while not done.wait(every):
        ps = subprocess.run(["ps", "-axo", "pid=,pgid=,pcpu=,state=,rss=,comm="], capture_output=True,
                            text=True).stdout.splitlines()
        rows = [p.split(None, 5) for p in ps]
        rows = [p for p in rows if len(p) == 6]
        procs = [p for p in rows if p[1] == str(pgid)]
        others = sorted((p for p in rows if p[1] != str(pgid) and int(p[0]) != me),
                        key=lambda p: -float(p[2]))
        total = sum(float(p[2]) for p in procs)
        outside = sum(float(p[2]) for p in others)
        last["max_outside_cpu"] = max(last.get("max_outside_cpu", 0.0), outside)
        top = " ".join(f"{p[0]}:{Path(p[5]).name}:{p[2]}%" for p in others[:3])
        detail = " ".join(f"{p[0]}:{p[2]}%:{p[3]}" for p in procs)
        samp.write(f"{time.monotonic() - start:8.1f}  gpu={gpu_utilization()}%  procs={len(procs)}  "
                   f"cpu_total={total:.0f}%  outside={outside:.0f}% [{top}]  {detail}\n")
        samp.flush()


def group_pids(pgid: int) -> list[tuple[int, str]]:
    ps = subprocess.run(["ps", "-axo", "pid=,pgid=,comm="], capture_output=True, text=True).stdout
    rows = [line.split(None, 2) for line in ps.splitlines()]
    return [(int(r[0]), r[2] if len(r) > 2 else "") for r in rows if len(r) >= 2 and r[1] == str(pgid)]


def abort_one_by_one(pgid: int, err) -> None:
    """SIGABRT each process in the group in turn (the trainer first), so faulthandler's stack dumps
    land in stderr.txt one after another, each under a marker, instead of interleaved."""
    pids = sorted(group_pids(pgid), key=lambda p: p[0] != pgid)
    for pid, comm in pids:
        err.write(f"\n===== bench_knee: SIGABRT pid {pid} ({'trainer' if pid == pgid else comm}) =====\n")
        err.flush()
        try:
            os.kill(pid, signal.SIGABRT)
        except ProcessLookupError:
            continue
        time.sleep(1.0)


def run_one(args, stage: str, workers: int, eval_batch: int, repeat: int, timeout: float) -> dict:
    run_dir = args.out / "runs" / f"{stage}_w{workers}_eb{eval_batch}_r{repeat}"
    run_dir.mkdir(parents=True)
    cmd = train_cmd(args, workers, eval_batch, run_dir)
    (run_dir / "command.txt").write_text(" ".join(cmd) + "\n")
    # Unbuffered so a killed run still shows how far it got; faulthandler so SIGABRT makes every
    # process in the group (spawned workers inherit the env) print all its threads' stacks to stderr.
    env = {**os.environ, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "PYTHONUNBUFFERED": "1", "PYTHONFAULTHANDLER": "1"}
    row = {"stage": stage, "workers": workers, "eval_batch": eval_batch, "repeat": repeat}
    print(f"[bench] {stage}: workers={workers} eval_batch={eval_batch} repeat={repeat} "
          f"(timeout {timeout:.0f}s, stall {args.stall_timeout:.0f}s)", flush=True)

    cpu0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    last = {"t": start, "games": "", "events": []}   # last stdout line time, last "Games: n/N", events
    with open(run_dir / "stdout.txt", "w") as out, open(run_dir / "progress.txt", "w") as prog, \
            open(run_dir / "samples.txt", "w") as samp, open(run_dir / "stderr.txt", "w") as err:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=err,
                                text=True, bufsize=1, start_new_session=True)
        done = threading.Event()
        reader = threading.Thread(target=pump, args=(proc, out, prog, start, last), daemon=True)
        sampler = threading.Thread(target=sample, daemon=True,
                                   args=(proc.pid, samp, start, args.sample_every, done, last))
        reader.start()
        sampler.start()
        status = None
        while status is None:
            rc = proc.poll()
            now = time.monotonic()
            if rc is not None:
                status = "OK" if rc == 0 else f"EXIT_{rc}"
            elif not args.full and "first_done" in last:
                status = "STEADY"                      # window measured; skip the straggler tail
            elif now - start > timeout:
                status = "TIMEOUT"
            elif args.stall_timeout and now - last["t"] > args.stall_timeout:
                status = "STALLED"
            else:
                time.sleep(1)
        if status in ("TIMEOUT", "STALLED"):
            abort_one_by_one(proc.pid, err)
        if status in ("TIMEOUT", "STALLED", "STEADY"):
            for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 10)):
                try:
                    os.killpg(proc.pid, sig)
                except (ProcessLookupError, PermissionError):   # macOS: EPERM once only a zombie is left
                    proc.wait(timeout=grace)
                    break
                try:
                    proc.wait(timeout=grace)
                    break
                except subprocess.TimeoutExpired:
                    continue
        done.set()
        reader.join(timeout=10)
        sampler.join(timeout=10)
    cpu1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    row["last_output_s"] = round(last["t"] - start, 1)
    row["max_outside_cpu"] = round(last.get("max_outside_cpu", 0.0))
    row["last_games_line"] = last["games"]
    row["wall_s"] = round(time.monotonic() - start, 2)
    row["children_cpu_s"] = round((cpu1.ru_utime - cpu0.ru_utime) + (cpu1.ru_stime - cpu0.ru_stime), 1)
    row.update(parse((run_dir / "stdout.txt").read_text() + "\n" + (run_dir / "stderr.txt").read_text()))
    row.update(steady(last["events"], last.get("first_done"), args.min_steady_games))
    if status in ("OK", "STEADY") and "steady_games_per_h" not in row:
        status = "UNPARSED"
    row["status"] = status
    print(f"[bench]   -> {status}, wall {row['wall_s']}s, last output at {row['last_output_s']}s"
          f" ({row['last_games_line'] or 'no Games line'})"
          + (f", steady {row['steady_games_per_h']:.0f} games/h over {row['steady_games']} games"
             if "steady_games_per_h" in row else "")
          + (f", {row['plies_per_s']:.2f} plies/s, {row['games_per_h']:.0f} games/h, "
             f"avg batch {row.get('avg_batch', 0):.1f}" if "plies_per_s" in row else ""), flush=True)
    for name in ("checkpoint", "games"):          # weights copies are ~7.5 MB each; keep logs only
        shutil.rmtree(run_dir / name, ignore_errors=True)
    return row


GOOD = ("OK", "STEADY")


def summarize(rows: list[dict], stage: str, key: str) -> dict[int, dict]:
    """{knob value: median metrics over OK repeats} for one stage."""
    out = {}
    for value in sorted({r[key] for r in rows if r["stage"] == stage}):
        ok = [r for r in rows if r["stage"] == stage and r[key] == value and r["status"] in GOOD]
        all_runs = [r for r in rows if r["stage"] == stage and r[key] == value]
        s = {"ok_runs": len(ok), "runs": len(all_runs),
             "statuses": sorted({r["status"] for r in all_runs})}
        for metric in ("steady_games_per_h", "plies_per_s", "games_per_h", "avg_batch", "avg_plies"):
            vals = [r[metric] for r in ok if metric in r]
            if vals:
                s[metric] = statistics.median(vals)
                s[metric + "_range"] = (min(vals), max(vals))
        out[value] = s
    return out


def best(summary: dict[int, dict], within: float) -> tuple[int | None, int | None]:
    """(best value by median steady games/h, smallest value within `within` of it)."""
    scored = {v: s["steady_games_per_h"] for v, s in summary.items() if "steady_games_per_h" in s}
    if not scored:
        return None, None
    top = max(scored, key=scored.get)
    knee = min(v for v, p in scored.items() if p >= scored[top] * (1 - within))
    return top, knee


def report(args, rows, s1, s2, picks, meta) -> str:
    lines = [f"# Self-play throughput sweep ({BOARD}x{BOARD}, {args.sims} sims, {args.games} games/run)", "",
             f"Machine: {meta['hardware'].get('machdep.cpu.brand_string')}, {meta['hardware'].get('hw.ncpu')} CPUs; "
             f"git {meta['git_head'][:8]}; weights {args.load_weights.name} sha1 {meta['weights_sha1'][:12]}", "",
             "Ranking metric: median steady-state games/h over good repeats, i.e. the game rate while every "
             "worker is busy (first `Games: n` line with n >= 5 up to the first `Worker done`), which leaves "
             "out start-up and the straggler tail (one long game finishing alone). "
             + ("Runs went to the end (--full). " if args.full else
                "Runs were stopped at the first `Worker done` (status STEADY), so the trainer's batch and "
                "plies stats are absent. ")
             + "Games/h assumes the same game-length mix across settings; a larger eval batch changes the "
             "search and may change it. Throughput only, and one clean sweep does not prove the Metal hang "
             "is gone.", ""]
    for title, summary, key in (("Stage 1: workers", s1, "workers"), ("Stage 2: eval batch", s2, "eval_batch")):
        lines += [f"## {title}", "", f"| {key} | good/runs | steady games/h (median, range) | full-run games/h | "
                  "avg batch | avg plies | statuses |", "|---|---|---|---|---|---|---|"]
        for v, s in summary.items():
            if "steady_games_per_h" in s:
                lo, hi = s["steady_games_per_h_range"]
                full = lambda m, f: format(s[m], f) if m in s else "-"
                lines.append(f"| {v} | {s['ok_runs']}/{s['runs']} | {s['steady_games_per_h']:.0f} "
                             f"({lo:.0f}-{hi:.0f}) | {full('games_per_h', '.0f')} | {full('avg_batch', '.1f')} | "
                             f"{full('avg_plies', '.1f')} | {', '.join(s['statuses'])} |")
            else:
                lines.append(f"| {v} | 0/{s['runs']} | - | - | - | - | {', '.join(s['statuses'])} |")
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
    ap.add_argument("--games", type=int, default=48, help="self-play games per run")
    ap.add_argument("--min-steady-games", type=int, default=15,
                    help="fewest games in the steady window for a run to count")
    ap.add_argument("--full", action="store_true",
                    help="run every game to the end (adds the trainer's batch/plies stats and the tail)")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--sims", type=int, default=400, help="simulations per move (SIMS_TABLE[24] = 400)")
    ap.add_argument("--stall-flush", type=int, default=48, help="last training run's mcts_stall_flush_sims")
    ap.add_argument("--virtual-visits", type=int, default=8, help="last training run's pending virtual visits")
    ap.add_argument("--within", type=float, default=0.03, help="knee: smallest value within this of the best")
    ap.add_argument("--first-timeout", type=float, default=3600, help="seconds, the first (reference) run")
    ap.add_argument("--timeout-mult", type=float, default=3.0, help="later runs: x the reference run's wall")
    ap.add_argument("--min-timeout", type=float, default=300)
    ap.add_argument("--stall-timeout", type=float, default=600,
                    help="seconds with no trainer output before a run is STALLED and killed; 0 = off")
    ap.add_argument("--sample-every", type=float, default=15, help="seconds between CPU/GPU samples")
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
        return max(args.min_timeout, row["wall_s"] * args.timeout_mult) if row["status"] in GOOD else timeout

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
                if row["status"] in ("TIMEOUT", "STALLED"):
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
