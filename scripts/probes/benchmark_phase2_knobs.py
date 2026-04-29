#!/usr/bin/env python
"""Manual benchmark sweep for Phase 2 parallel-labeling knobs.

NOT in CI. Sweeps several (label-workers, mcts-eval-batch-size,
mcts-stall-flush-sims) configurations against a real checkpoint and
reports wallclock + admitted-ID equivalence vs. a serial baseline.

CRITICAL safety property: each subprocess is launched in its own
process group with a timeout watchdog. On timeout, SIGTERM is sent to
the whole process group (catches spawned MLX worker children); after
5s, SIGKILL. Configs with --mcts-eval-batch-size above the documented
Metal-safe cap of 14 require --include-unsafe and are subject to the
same kill-on-hang protection. If a config froze, larger eval_batch
values are skipped automatically (poison-pill heuristic).

Usage:
    .venv/bin/python scripts/probes/benchmark_phase2_knobs.py \\
        --input scripts/GPU/logs/games \\
        --source-iter-range 57 58 \\
        --label-checkpoint checkpoints/.../model_iter_0059.safetensors \\
        --candidates 10 \\
        --mcts-sims 500

Add --include-unsafe to additionally probe eval_batch in {16, 24}
(the documented Metal-hang region). Use only on a machine you can
afford to have momentarily frozen during the test.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Config:
    mode: str            # "serial" or "process"
    workers: int
    eval_batch: int
    stall_flush: int
    unsafe: bool = False  # requires --allow-unsafe-eval-batch

    @property
    def label(self) -> str:
        return (f"{self.mode} workers={self.workers} eval_batch={self.eval_batch} "
                f"stall_flush={self.stall_flush}"
                + (" *unsafe*" if self.unsafe else ""))

    @property
    def slug(self) -> str:
        return (f"{self.mode}_w{self.workers}_e{self.eval_batch}_s{self.stall_flush}"
                + ("_unsafe" if self.unsafe else ""))


@dataclass
class Result:
    config: Config
    status: str          # "OK", "FROZE", "FAILED", "SKIPPED", "INTERRUPTED"
    wallclock_s: float = 0.0
    admitted_ids: set = field(default_factory=set)
    phase2_run_stats: dict = field(default_factory=dict)
    error_tail: str = ""


# ---------------------------------------------------------------------------
# Subprocess watchdog
# ---------------------------------------------------------------------------


def _kill_group(pgid: int, sig: int) -> None:
    """Best-effort: send `sig` to the process group `pgid`."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _run_with_watchdog(
    cmd: list[str],
    timeout: float,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
) -> tuple[int, str, str, float, bool]:
    """Run `cmd` with a timeout. Returns (returncode, stdout, stderr, wallclock_s, froze).

    On timeout: SIGTERM the process group, wait 5s, SIGKILL if still alive.
    Returns froze=True in that case. The process is launched in its own
    process group via start_new_session=True so the kill propagates to
    spawned worker children (critical for MLX/Metal hangs).
    """
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    pgid = proc.pid  # because start_new_session=True, pid == pgid
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout, stderr, time.time() - t0, False
    except subprocess.TimeoutExpired:
        # Kill the whole process group (catches spawned MLX workers).
        _kill_group(pgid, signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            _kill_group(pgid, signal.SIGKILL)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
        return -1, stdout or "", stderr or "", time.time() - t0, True
    except KeyboardInterrupt:
        # User Ctrl-C while we were communicate()-ing. Make sure the child
        # and its workers don't survive.
        _kill_group(pgid, signal.SIGTERM)
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            _kill_group(pgid, signal.SIGKILL)
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        raise


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


def _build_grid(args) -> list[Config]:
    # Baseline first.
    grid = [Config(mode="serial", workers=1, eval_batch=14, stall_flush=16)]
    safe_eval_batches = [8, 14]
    unsafe_eval_batches = [16, 24] if args.include_unsafe else []
    for w in args.workers_grid:
        for eb in safe_eval_batches:
            grid.append(Config(mode="process", workers=w,
                               eval_batch=eb, stall_flush=16, unsafe=False))
        for eb in unsafe_eval_batches:
            grid.append(Config(mode="process", workers=w,
                               eval_batch=eb, stall_flush=16, unsafe=True))
    return grid


def _build_cli(args, cfg: Config, out_path: Path) -> list[str]:
    """Build the build_probe_suite.py argv for one config."""
    py = sys.executable
    script = str(Path(__file__).resolve().parents[1].parent
                 / "scripts" / "build_probe_suite.py")
    cmd = [
        py, script,
        "--tier", "strong_advantage",
        "--input", args.input,
        "--source-iter-range", str(args.source_iter_range[0]),
                                str(args.source_iter_range[1]),
        "--label-checkpoint", args.label_checkpoint,
        "--label-mcts-sims", str(args.mcts_sims),
        "--label-mcts-repeats", str(args.mcts_repeats),
        "--max-probes", str(args.candidates),
        "--out", str(out_path),
        "--label-worker-mode", cfg.mode,
        "--mcts-eval-batch-size", str(cfg.eval_batch),
        "--mcts-stall-flush-sims", str(cfg.stall_flush),
        "--force",
    ]
    if cfg.mode == "process":
        cmd.extend(["--label-workers", str(cfg.workers)])
    if cfg.unsafe:
        cmd.append("--allow-unsafe-eval-batch")
    return cmd


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _parse_results(out_path: Path) -> tuple[set, dict]:
    """Read draft.json; return (admitted_id_set, phase2_run_stats_dict)."""
    draft_path = out_path.with_suffix(".draft.json")
    draft = json.loads(draft_path.read_text())
    admitted_ids = {p["id"] for p in draft.get("probes", [])}
    stats = draft.get("meta", {}).get("phase2_run_stats", {})
    return admitted_ids, stats


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _format_report(results: list[Result], baseline_wallclock: float) -> str:
    """Build the markdown report. Sort OK rows by wallclock asc; FROZE/SKIPPED
    rows go at the end."""
    lines: list[str] = []
    lines.append("# Phase 2 Knob Sweep")
    lines.append("")

    baseline_results = [r for r in results if r.config.mode == "serial"]
    baseline = baseline_results[0] if baseline_results else None
    if baseline is not None:
        lines.append(f"Baseline: {baseline.config.label}")
        lines.append(f"Baseline wallclock: {baseline_wallclock:.2f}s")
        lines.append(f"Baseline admitted: {len(baseline.admitted_ids)}")
        lines.append("")

    header = ("| Mode | Workers | Eval batch | Stall flush | Wallclock | "
              "Speedup | IDs match | Reruns | Flips | Status |")
    sep = ("|------|---------|-----------|-------------|-----------|---------"
           "|-----------|--------|-------|--------|")
    lines.append(header)
    lines.append(sep)

    ok_results = [r for r in results if r.status == "OK"]
    ok_results.sort(key=lambda r: r.wallclock_s)
    other_results = [r for r in results if r.status != "OK"]

    baseline_ids = baseline.admitted_ids if baseline is not None else set()

    for r in ok_results + other_results:
        cfg = r.config
        if r.status == "OK":
            speedup = (baseline_wallclock / r.wallclock_s
                       if r.wallclock_s > 0 else float("inf"))
            if cfg.mode == "serial":
                ids_match = "-"
            else:
                ids_match = "yes" if r.admitted_ids == baseline_ids else "NO"
            stats = r.phase2_run_stats
            reruns = stats.get("borderline_reruns", 0)
            flips = stats.get("borderline_flips", 0)
            lines.append(
                f"| {cfg.mode} | {cfg.workers} | {cfg.eval_batch} | "
                f"{cfg.stall_flush} | {r.wallclock_s:.2f}s | "
                f"{speedup:.2f}x | {ids_match} | {reruns} | {flips} | OK |"
            )
        elif r.status == "FROZE":
            lines.append(
                f"| {cfg.mode} | {cfg.workers} | {cfg.eval_batch} | "
                f"{cfg.stall_flush} | ~{r.wallclock_s:.0f}s (killed) | "
                f"- | - | - | - | FROZE |"
            )
        else:  # SKIPPED, FAILED, INTERRUPTED
            lines.append(
                f"| {cfg.mode} | {cfg.workers} | {cfg.eval_batch} | "
                f"{cfg.stall_flush} | - | - | - | - | - | {r.status} |"
            )

    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    process_oks = [r for r in ok_results
                   if r.config.mode == "process" and not r.config.unsafe]
    if process_oks:
        best = process_oks[0]  # already sorted by wallclock asc
        speedup = (baseline_wallclock / best.wallclock_s
                   if best.wallclock_s > 0 else float("inf"))
        lines.append(
            f"Best safe config: `--label-worker-mode process --label-workers "
            f"{best.config.workers} --mcts-eval-batch-size "
            f"{best.config.eval_batch}` "
            f"({speedup:.2f}x speedup vs serial baseline)"
        )
    else:
        lines.append("No successful process-mode safe configs to recommend.")

    frozen = [r for r in results if r.status == "FROZE"]
    if frozen:
        lines.append("")
        lines.append(
            f"Frozen configs ({len(frozen)}): "
            + ", ".join(
                f"eval_batch={r.config.eval_batch}/workers={r.config.workers}"
                for r in frozen
            )
        )
        lines.append(
            "Keep `--mcts-eval-batch-size` at the largest non-frozen value."
        )

    drift = [r for r in ok_results
             if r.config.mode == "process" and r.admitted_ids != baseline_ids]
    if drift:
        lines.append("")
        lines.append(
            f"WARNING: {len(drift)} process-mode config(s) produced different "
            "admitted IDs than serial. Borderline-rerun should have caught "
            "these — investigate."
        )

    failed = [r for r in results if r.status == "FAILED"]
    if failed:
        lines.append("")
        lines.append(f"FAILED configs ({len(failed)}):")
        for r in failed:
            tail = (r.error_tail or "").strip().splitlines()
            tail_one = tail[-1] if tail else "(no stderr)"
            lines.append(f"- {r.config.label}: {tail_one}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


_UNSAFE_BANNER = """
================================================================================
  WARNING: about to run an UNSAFE configuration (eval_batch > {cap}).
  This is the documented Metal-hang regime. The watchdog will SIGTERM the
  process group on timeout and SIGKILL after 5s, but your machine may
  briefly become unresponsive while the GPU is wedged.
  Press Ctrl-C now if you want to abort.
================================================================================
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="games directory")
    ap.add_argument("--source-iter-range", nargs=2, type=int, required=True,
                    metavar=("MIN", "MAX"),
                    help="source-iter-range for build_probe_suite.py")
    ap.add_argument("--label-checkpoint", required=True,
                    help="path to a real .safetensors checkpoint")
    ap.add_argument("--candidates", type=int, default=10,
                    help="passed as --max-probes; small for fast turnaround")
    ap.add_argument("--mcts-sims", type=int, default=500,
                    help="passed as --label-mcts-sims; small for speed")
    ap.add_argument("--mcts-repeats", type=int, default=1,
                    help="passed as --label-mcts-repeats")
    ap.add_argument("--include-unsafe", action="store_true",
                    help="Include eval_batch in {16, 24}; requires "
                         "--allow-unsafe-eval-batch (kill-on-hang protected).")
    ap.add_argument("--timeout-multiplier", type=float, default=5.0,
                    help="kill if wallclock > T x baseline_wallclock")
    ap.add_argument("--min-timeout", type=float, default=60.0,
                    help="floor for the kill timeout (seconds)")
    ap.add_argument("--workdir", default="/tmp/probe_bench",
                    help="working directory for per-config outputs")
    ap.add_argument("--report-path", default=None,
                    help="if set, also write the markdown report to this path")
    ap.add_argument("--workers-grid", default="1,2,4",
                    help="Comma-separated worker counts (process mode).")
    args = ap.parse_args()
    args.workers_grid = [int(x) for x in args.workers_grid.split(",")
                         if x.strip()]
    if not args.workers_grid:
        ap.error("--workers-grid must contain at least one integer")
    for w in args.workers_grid:
        if w < 1:
            ap.error(f"--workers-grid entries must be >= 1 (got {w})")

    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    grid = _build_grid(args)
    print(f"[bench] {len(grid)} configurations to sweep")
    print(f"[bench] workdir: {workdir}")
    if args.include_unsafe:
        print("[bench] --include-unsafe enabled: eval_batch in {16, 24} "
              "will be probed under kill-on-hang watchdog")

    results: list[Result] = []
    baseline_wallclock: Optional[float] = None
    frozen_eval_batch_min: Optional[int] = None
    interrupted = False

    try:
        for cfg in grid:
            # Poison-pill: skip if any earlier eval_batch <= ours froze.
            # Once eval_batch=N froze, any eval_batch >= N is suspect
            # regardless of worker count.
            if (frozen_eval_batch_min is not None
                    and cfg.mode == "process"
                    and cfg.eval_batch >= frozen_eval_batch_min):
                print(f"[bench] SKIPPED {cfg.label} — earlier eval_batch="
                      f"{frozen_eval_batch_min} froze; larger values likely "
                      "also unsafe")
                results.append(Result(config=cfg, status="SKIPPED"))
                continue

            cfg_dir = workdir / cfg.slug
            cfg_dir.mkdir(parents=True, exist_ok=True)
            out_path = cfg_dir / "out.json"
            cmd = _build_cli(args, cfg, out_path)

            if cfg.mode == "serial" and baseline_wallclock is None:
                # Generous floor for the baseline since we have no reference yet.
                timeout = args.min_timeout * 10
                print(f"[bench] running BASELINE: {cfg.label} "
                      f"(timeout={timeout:.0f}s)")
            else:
                base = baseline_wallclock if baseline_wallclock is not None else 0.0
                timeout = max(args.min_timeout,
                              base * args.timeout_multiplier)
                print(f"[bench] running: {cfg.label} (timeout={timeout:.0f}s)")

            if cfg.unsafe:
                # Loud banner before unsafe configs.
                print(_UNSAFE_BANNER.format(cap=14), flush=True)

            try:
                rc, stdout, stderr, wallclock, froze = _run_with_watchdog(
                    cmd, timeout
                )
            except KeyboardInterrupt:
                print("\n[bench] interrupted by user (Ctrl-C); current config "
                      "killed", file=sys.stderr)
                results.append(Result(
                    config=cfg, status="INTERRUPTED",
                    wallclock_s=0.0,
                    error_tail="KeyboardInterrupt during run",
                ))
                interrupted = True
                break

            if froze:
                print(f"[bench] FROZE after {wallclock:.0f}s — killed process "
                      "group")
                results.append(Result(config=cfg, status="FROZE",
                                      wallclock_s=wallclock))
                if (frozen_eval_batch_min is None
                        or cfg.eval_batch < frozen_eval_batch_min):
                    frozen_eval_batch_min = cfg.eval_batch
                continue
            if rc != 0:
                tail = (stderr or "")[-500:]
                print(f"[bench] FAILED rc={rc}: {tail}")
                results.append(Result(
                    config=cfg, status="FAILED", wallclock_s=wallclock,
                    error_tail=tail,
                ))
                continue

            try:
                admitted_ids, stats = _parse_results(out_path)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"[bench] FAILED to parse output: {e}")
                results.append(Result(
                    config=cfg, status="FAILED", wallclock_s=wallclock,
                    error_tail=f"parse error: {e}",
                ))
                continue

            results.append(Result(
                config=cfg, status="OK", wallclock_s=wallclock,
                admitted_ids=admitted_ids, phase2_run_stats=stats,
            ))
            print(f"[bench] OK {wallclock:.2f}s — {len(admitted_ids)} admitted")

            if cfg.mode == "serial":
                baseline_wallclock = wallclock
    except KeyboardInterrupt:
        # Defensive: any KeyboardInterrupt outside the inner try goes here.
        print("\n[bench] interrupted by user (Ctrl-C)", file=sys.stderr)
        interrupted = True

    if baseline_wallclock is None:
        print("[bench] ERROR: baseline serial run didn't complete; cannot "
              "produce a comparative report.", file=sys.stderr)
        # Still emit a partial table so the user sees what happened.
        if results:
            partial = _format_partial_report(results)
            print()
            print(partial)
            if args.report_path:
                Path(args.report_path).write_text(partial)
                print(f"\n[bench] partial report saved to {args.report_path}")
        return 1

    report = _format_report(results, baseline_wallclock)
    print()
    print(report)
    if args.report_path:
        Path(args.report_path).write_text(report)
        print(f"\n[bench] report saved to {args.report_path}")

    return 130 if interrupted else 0


def _format_partial_report(results: list[Result]) -> str:
    """Used when the baseline failed: print whatever we have, no speedups."""
    lines = ["# Phase 2 Knob Sweep (partial — baseline failed)", ""]
    lines.append("| Mode | Workers | Eval batch | Stall flush | Wallclock | "
                 "Status |")
    lines.append("|------|---------|-----------|-------------|-----------|"
                 "--------|")
    for r in results:
        cfg = r.config
        wc = (f"{r.wallclock_s:.2f}s" if r.status == "OK"
              else f"~{r.wallclock_s:.0f}s (killed)" if r.status == "FROZE"
              else "-")
        lines.append(
            f"| {cfg.mode} | {cfg.workers} | {cfg.eval_batch} | "
            f"{cfg.stall_flush} | {wc} | {r.status} |"
        )
    failed = [r for r in results if r.status == "FAILED"]
    if failed:
        lines.append("")
        lines.append("## Failures")
        for r in failed:
            tail = (r.error_tail or "").strip().splitlines()
            tail_one = tail[-1] if tail else "(no stderr)"
            lines.append(f"- {r.config.label}: {tail_one}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
