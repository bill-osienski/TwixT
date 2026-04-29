#!/usr/bin/env python
"""Manual coordinate-descent benchmark for Phase 2 parallel-labeling knobs.

NOT in CI. Adaptively searches the (label-workers, mcts-eval-batch-size,
mcts-stall-flush-sims) space against a real checkpoint and reports
wallclock + admitted-ID equivalence vs. a serial baseline.

Algorithm: a serial baseline runs first to provide a timeout reference.
Then for each pass, we sweep ONE knob at a time while holding the others
at the current best ("center"). The fastest OK config in each per-knob
sweep becomes the new center for that knob. If no knob moves during a
pass, we converged and stop early. Default: 2 passes. Configs already
seen are served from a cache (mode, workers, eval_batch, stall_flush)
so we never re-run identical work across passes.

CRITICAL safety property: each subprocess is launched in its own
process group with a timeout watchdog. On timeout, SIGTERM is sent to
the whole process group (catches spawned MLX worker children); after
5s, SIGKILL. Configs with --mcts-eval-batch-size above the documented
Metal-safe cap of 14 require --include-unsafe and are subject to the
same kill-on-hang protection. Once an eval_batch value froze, it (and
any larger value) is skipped for the rest of the run (poison-pill).

Usage:
    .venv/bin/python scripts/probes/benchmark_phase2_knobs.py \\
        --input scripts/GPU/logs/games \\
        --source-iter-range 57 58 \\
        --label-checkpoint checkpoints/.../model_iter_0059.safetensors \\
        --candidates 10 \\
        --mcts-sims 500 \\
        --workers-values 1,2,4,6,8 \\
        --eval-batch-values 8,12,14 \\
        --stall-flush-values 4,8,16,32 \\
        --passes 2

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

    @property
    def cache_key(self) -> tuple:
        return (self.mode, self.workers, self.eval_batch, self.stall_flush)


@dataclass
class Result:
    config: Config
    status: str          # "OK", "FROZE", "FAILED", "SKIPPED", "INTERRUPTED"
    wallclock_s: float = 0.0
    admitted_ids: set = field(default_factory=set)
    phase2_run_stats: dict = field(default_factory=dict)
    error_tail: str = ""


@dataclass
class TraceEntry:
    """One row in the chronological trace report."""
    pass_idx: int                # 0 = baseline, 1.. = coordinate-descent passes
    knob: str                    # "baseline", "workers", "eval_batch", "stall_flush"
    result: Result
    cached: bool = False
    cached_wallclock: float = 0.0  # if cached, the original wallclock
    is_winner: bool = False


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
# Coordinate-descent driver
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


class Bench:
    """Coordinate-descent driver. Holds state shared across per-knob sweeps."""

    def __init__(self, args, workdir: Path):
        self.args = args
        self.workdir = workdir
        # Cache: (mode, workers, eval_batch, stall_flush) -> Result
        self.cache: dict[tuple, Result] = {}
        # Chronological trace of every config we considered.
        self.trace: list[TraceEntry] = []
        # eval_batch values that froze; future runs at >= min(frozen) are skipped.
        self.frozen_eval_batches: set[int] = set()
        self.baseline_wallclock: Optional[float] = None
        self.interrupted = False

    # ------------------------------------------------------------------
    # Single-config execution (with cache + watchdog)
    # ------------------------------------------------------------------

    def run_or_cached(self, cfg: Config) -> tuple[Result, bool]:
        """Run `cfg` if not cached; return (result, was_cached).

        was_cached=True means we returned the previous Result for the
        same cache_key.
        """
        cached = self.cache.get(cfg.cache_key)
        if cached is not None:
            return cached, True
        result = self._execute(cfg)
        self.cache[cfg.cache_key] = result
        return result, False

    def _execute(self, cfg: Config) -> Result:
        """Run a single config. Returns Result; updates frozen_eval_batches."""
        cfg_dir = self.workdir / cfg.slug
        cfg_dir.mkdir(parents=True, exist_ok=True)
        out_path = cfg_dir / "out.json"
        cmd = _build_cli(self.args, cfg, out_path)

        if cfg.mode == "serial" and self.baseline_wallclock is None:
            timeout = self.args.min_timeout * 10
            print(f"[bench] running BASELINE: {cfg.label} "
                  f"(timeout={timeout:.0f}s)")
        else:
            base = self.baseline_wallclock if self.baseline_wallclock is not None else 0.0
            timeout = max(self.args.min_timeout,
                          base * self.args.timeout_multiplier)
            print(f"[bench] running: {cfg.label} (timeout={timeout:.0f}s)")

        if cfg.unsafe:
            print(_UNSAFE_BANNER.format(cap=14), flush=True)

        rc, stdout, stderr, wallclock, froze = _run_with_watchdog(cmd, timeout)

        if froze:
            print(f"[bench] FROZE after {wallclock:.0f}s — killed process group")
            if cfg.eval_batch is not None:
                self.frozen_eval_batches.add(cfg.eval_batch)
            return Result(config=cfg, status="FROZE", wallclock_s=wallclock)
        if rc != 0:
            tail = (stderr or "")[-500:]
            print(f"[bench] FAILED rc={rc}: {tail}")
            return Result(config=cfg, status="FAILED",
                          wallclock_s=wallclock, error_tail=tail)
        try:
            admitted_ids, stats = _parse_results(out_path)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[bench] FAILED to parse output: {e}")
            return Result(config=cfg, status="FAILED",
                          wallclock_s=wallclock,
                          error_tail=f"parse error: {e}")

        if cfg.mode == "serial" and self.baseline_wallclock is None:
            self.baseline_wallclock = wallclock

        print(f"[bench] OK {wallclock:.2f}s — {len(admitted_ids)} admitted")
        return Result(config=cfg, status="OK", wallclock_s=wallclock,
                      admitted_ids=admitted_ids, phase2_run_stats=stats)

    # ------------------------------------------------------------------
    # Per-knob sweeps
    # ------------------------------------------------------------------

    def _eval_batch_candidates(self) -> list[tuple[int, bool]]:
        """Return [(value, unsafe)] for the current eval_batch sweep."""
        cands: list[tuple[int, bool]] = [(v, False)
                                         for v in self.args.eval_batch_values]
        if self.args.include_unsafe:
            cands.extend((v, True) for v in self.args.unsafe_eval_batch_values)
        return cands

    def sweep_knob(
        self,
        pass_idx: int,
        knob_name: str,
        center: dict,
    ) -> list[Result]:
        """Sweep `knob_name` while holding other knobs at `center`.

        Returns the OK results from this sweep (sorted by wallclock asc),
        but also appends every attempt (cached or not) to self.trace.
        """
        sweep_results: list[Result] = []

        if knob_name == "workers":
            value_specs: list[tuple[int, bool]] = [
                (v, False) for v in self.args.workers_values
            ]
        elif knob_name == "eval_batch":
            value_specs = self._eval_batch_candidates()
        elif knob_name == "stall_flush":
            value_specs = [(v, False) for v in self.args.stall_flush_values]
        else:
            raise ValueError(f"unknown knob: {knob_name}")

        for v, unsafe_flag in value_specs:
            cfg = Config(
                mode="process",
                workers=center["workers"] if knob_name != "workers" else v,
                eval_batch=center["eval_batch"] if knob_name != "eval_batch" else v,
                stall_flush=center["stall_flush"] if knob_name != "stall_flush" else v,
                unsafe=unsafe_flag if knob_name == "eval_batch" else False,
            )

            # Poison-pill: skip eval_batch values >= any frozen value.
            if (knob_name == "eval_batch" and self.frozen_eval_batches
                    and cfg.eval_batch >= min(self.frozen_eval_batches)):
                print(f"[bench] SKIPPED {cfg.label} — earlier eval_batch="
                      f"{min(self.frozen_eval_batches)} froze")
                skipped = Result(config=cfg, status="SKIPPED")
                sweep_results.append(skipped)
                self.trace.append(TraceEntry(
                    pass_idx=pass_idx, knob=knob_name, result=skipped,
                ))
                # Larger values will also be >= min(frozen); they will be
                # individually checked above and skipped — but per spec we
                # also break here once we hit a frozen value to avoid noise.
                break

            try:
                result, cached = self.run_or_cached(cfg)
            except KeyboardInterrupt:
                print("\n[bench] interrupted by user (Ctrl-C); current "
                      "config killed", file=sys.stderr)
                interrupted = Result(
                    config=cfg, status="INTERRUPTED", wallclock_s=0.0,
                    error_tail="KeyboardInterrupt during run",
                )
                sweep_results.append(interrupted)
                self.trace.append(TraceEntry(
                    pass_idx=pass_idx, knob=knob_name, result=interrupted,
                ))
                self.interrupted = True
                raise

            entry = TraceEntry(
                pass_idx=pass_idx, knob=knob_name, result=result,
                cached=cached,
                cached_wallclock=result.wallclock_s if cached else 0.0,
            )
            self.trace.append(entry)
            sweep_results.append(result)

            if result.status == "FROZE" and knob_name == "eval_batch":
                # Larger eval_batch values will also freeze; stop sweep.
                break

        return sweep_results

    # ------------------------------------------------------------------
    # Top-level driver
    # ------------------------------------------------------------------

    def run(self) -> tuple[list[Result], dict]:
        """Run baseline + coordinate-descent passes. Returns (all_results, center)."""
        # Step 0: baseline.
        baseline_cfg = Config(
            mode="serial",
            workers=1,
            eval_batch=self.args.start_eval_batch,
            stall_flush=self.args.start_stall_flush,
        )
        try:
            baseline_result, _ = self.run_or_cached(baseline_cfg)
        except KeyboardInterrupt:
            print("\n[bench] interrupted by user (Ctrl-C) during baseline",
                  file=sys.stderr)
            self.interrupted = True
            return list(self.cache.values()), {
                "workers": self.args.start_workers,
                "eval_batch": self.args.start_eval_batch,
                "stall_flush": self.args.start_stall_flush,
            }

        self.trace.append(TraceEntry(
            pass_idx=0, knob="baseline", result=baseline_result,
        ))
        if baseline_result.status != "OK":
            return list(self.cache.values()), {
                "workers": self.args.start_workers,
                "eval_batch": self.args.start_eval_batch,
                "stall_flush": self.args.start_stall_flush,
            }

        center = {
            "workers": self.args.start_workers,
            "eval_batch": self.args.start_eval_batch,
            "stall_flush": self.args.start_stall_flush,
        }

        try:
            for pass_idx in range(1, self.args.passes + 1):
                moved = False
                for knob_name in ("workers", "eval_batch", "stall_flush"):
                    sweep = self.sweep_knob(pass_idx, knob_name, center)
                    ok = [r for r in sweep if r.status == "OK"]
                    if not ok:
                        print(f"[bench] no successful runs sweeping "
                              f"{knob_name}; keeping center value "
                              f"{center[knob_name]}")
                        continue
                    ok.sort(key=lambda r: r.wallclock_s)
                    winner = ok[0]
                    winner_v = getattr(winner.config, knob_name)
                    # Mark the winner in the trace (the most recent entry
                    # whose result IS winner).
                    for entry in reversed(self.trace):
                        if entry.pass_idx != pass_idx or entry.knob != knob_name:
                            continue
                        if entry.result.config.cache_key == winner.config.cache_key:
                            entry.is_winner = True
                            break
                    if winner_v != center[knob_name]:
                        moved = True
                        print(f"[bench] {knob_name}: {center[knob_name]} -> "
                              f"{winner_v} ({winner.wallclock_s:.2f}s)")
                        center[knob_name] = winner_v
                    else:
                        print(f"[bench] {knob_name}: {center[knob_name]} "
                              f"(unchanged, best at {winner.wallclock_s:.2f}s)")
                if not moved:
                    print(f"[bench] converged after pass {pass_idx}")
                    break
        except KeyboardInterrupt:
            print("\n[bench] interrupted by user (Ctrl-C)", file=sys.stderr)
            self.interrupted = True

        return list(self.cache.values()), center


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _format_config_short(cfg: Config) -> str:
    return (f"{cfg.mode} w={cfg.workers} eb={cfg.eval_batch} "
            f"sf={cfg.stall_flush}"
            + (" *unsafe*" if cfg.unsafe else ""))


def _format_report(
    trace: list[TraceEntry],
    center: dict,
    baseline_wallclock: Optional[float],
    all_results: list[Result],
) -> str:
    lines: list[str] = []
    lines.append("# Phase 2 Knob Sweep — Coordinate Descent")
    lines.append("")

    if baseline_wallclock is not None:
        lines.append(f"Baseline wallclock: {baseline_wallclock:.2f}s")
        lines.append("")

    lines.append("## Coordinate descent trace")
    lines.append("")
    lines.append("| Pass | Knob       | Config                       | "
                 "Wallclock          | Speedup | Status |")
    lines.append("|------|------------|------------------------------|"
                 "--------------------|---------|--------|")

    converged_after: Optional[int] = None
    last_pass_seen = 0
    for entry in trace:
        cfg = entry.result.config
        cfg_str = _format_config_short(cfg)
        pass_label = "-" if entry.pass_idx == 0 else str(entry.pass_idx)
        knob_label = entry.knob
        winner_marker = " *winner*" if entry.is_winner else ""
        status = entry.result.status
        if entry.cached:
            wc_str = f"(cached: {entry.cached_wallclock:.2f}s)"
            if (status == "OK" and baseline_wallclock
                    and entry.cached_wallclock > 0):
                speedup_str = f"{baseline_wallclock / entry.cached_wallclock:.2f}x"
            else:
                speedup_str = "-"
        elif status == "OK":
            wc_str = f"{entry.result.wallclock_s:.2f}s"
            if baseline_wallclock and entry.result.wallclock_s > 0:
                speedup_str = f"{baseline_wallclock / entry.result.wallclock_s:.2f}x"
            else:
                speedup_str = "-"
        elif status == "FROZE":
            wc_str = f"~{entry.result.wallclock_s:.0f}s (killed)"
            speedup_str = "-"
        else:
            wc_str = "-"
            speedup_str = "-"
        lines.append(
            f"| {pass_label:>4} | {knob_label:<10} | {cfg_str:<28} | "
            f"{wc_str:<18} | {speedup_str:<7} | "
            f"{status}{winner_marker} |"
        )
        last_pass_seen = max(last_pass_seen, entry.pass_idx)

    # Detect convergence: if the last pass had no movers we'd have broken
    # the loop. Caller doesn't tell us directly; infer from center vs trace.
    # We just print the highest pass index seen.
    lines.append("")
    if last_pass_seen > 0:
        lines.append(f"Last pass executed: {last_pass_seen}.")

    # ----- Recommendation section -----
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")

    # Find the best OK process-mode result (in cache).
    process_oks = [r for r in all_results
                   if r.status == "OK" and r.config.mode == "process"
                   and not r.config.unsafe]
    process_oks.sort(key=lambda r: r.wallclock_s)
    # Prefer reporting the FINAL center if it's an OK config; fall back to
    # global best if the center somehow isn't in the cache.
    final_key = ("process", center["workers"], center["eval_batch"],
                 center["stall_flush"])
    final_result: Optional[Result] = None
    for r in all_results:
        if (r.config.mode == "process" and r.status == "OK"
                and r.config.cache_key == final_key):
            final_result = r
            break

    baseline_results = [r for r in all_results
                        if r.config.mode == "serial" and r.status == "OK"]
    baseline_ids = (baseline_results[0].admitted_ids
                    if baseline_results else set())

    if final_result is not None:
        cfg = final_result.config
        if baseline_wallclock and final_result.wallclock_s > 0:
            speedup = baseline_wallclock / final_result.wallclock_s
            speedup_str = f"{speedup:.2f}x speedup vs. {baseline_wallclock:.2f}s serial baseline"
        else:
            speedup_str = "(no baseline for speedup)"
        lines.append(
            f"Best config: `--label-worker-mode process --label-workers "
            f"{cfg.workers} --mcts-eval-batch-size {cfg.eval_batch} "
            f"--mcts-stall-flush-sims {cfg.stall_flush}`"
        )
        lines.append(f"Wallclock: {final_result.wallclock_s:.2f}s ({speedup_str})")
        if baseline_results:
            ids_match = "yes" if final_result.admitted_ids == baseline_ids else "NO"
            lines.append(f"Admitted IDs: same as baseline ({ids_match})")
        stats = final_result.phase2_run_stats
        reruns = stats.get("borderline_reruns", 0)
        flips = stats.get("borderline_flips", 0)
        lines.append(f"Borderline reruns: {reruns}")
        lines.append(f"Borderline flips: {flips}")
    elif process_oks:
        best = process_oks[0]
        cfg = best.config
        if baseline_wallclock and best.wallclock_s > 0:
            speedup = baseline_wallclock / best.wallclock_s
            speedup_str = f"{speedup:.2f}x speedup vs. {baseline_wallclock:.2f}s serial baseline"
        else:
            speedup_str = "(no baseline for speedup)"
        lines.append(
            "Final center didn't produce an OK result; falling back to "
            "global best OK process config."
        )
        lines.append(
            f"Best config: `--label-worker-mode process --label-workers "
            f"{cfg.workers} --mcts-eval-batch-size {cfg.eval_batch} "
            f"--mcts-stall-flush-sims {cfg.stall_flush}`"
        )
        lines.append(f"Wallclock: {best.wallclock_s:.2f}s ({speedup_str})")
    else:
        lines.append("No successful process-mode safe configs to recommend.")

    # Notes about frozen values.
    frozen = [r for r in all_results if r.status == "FROZE"]
    if frozen:
        lines.append("")
        lines.append("Notes:")
        frozen_ebs = sorted({r.config.eval_batch for r in frozen})
        lines.append(
            f"- eval_batch values that froze on this machine: {frozen_ebs}. "
            "Keep eval_batch at the largest non-frozen value."
        )

    # Drift / failure warnings.
    drift = [r for r in all_results
             if r.status == "OK" and r.config.mode == "process"
             and r.admitted_ids != baseline_ids and baseline_results]
    if drift:
        lines.append("")
        lines.append(
            f"WARNING: {len(drift)} process-mode config(s) produced different "
            "admitted IDs than serial. Borderline-rerun should have caught "
            "these — investigate."
        )

    failed = [r for r in all_results if r.status == "FAILED"]
    if failed:
        lines.append("")
        lines.append(f"FAILED configs ({len(failed)}):")
        for r in failed:
            tail = (r.error_tail or "").strip().splitlines()
            tail_one = tail[-1] if tail else "(no stderr)"
            lines.append(f"- {r.config.label}: {tail_one}")

    return "\n".join(lines)


def _format_partial_report(trace: list[TraceEntry]) -> str:
    """Used when the baseline failed: print whatever we have, no speedups."""
    lines = ["# Phase 2 Knob Sweep — Coordinate Descent (partial — baseline failed)",
             ""]
    lines.append("| Pass | Knob       | Config                       | "
                 "Wallclock          | Status |")
    lines.append("|------|------------|------------------------------|"
                 "--------------------|--------|")
    for entry in trace:
        cfg = entry.result.config
        cfg_str = _format_config_short(cfg)
        pass_label = "-" if entry.pass_idx == 0 else str(entry.pass_idx)
        if entry.cached:
            wc = f"(cached: {entry.cached_wallclock:.2f}s)"
        elif entry.result.status == "OK":
            wc = f"{entry.result.wallclock_s:.2f}s"
        elif entry.result.status == "FROZE":
            wc = f"~{entry.result.wallclock_s:.0f}s (killed)"
        else:
            wc = "-"
        lines.append(
            f"| {pass_label:>4} | {entry.knob:<10} | {cfg_str:<28} | "
            f"{wc:<18} | {entry.result.status} |"
        )
    failed = [e.result for e in trace if e.result.status == "FAILED"]
    if failed:
        lines.append("")
        lines.append("## Failures")
        for r in failed:
            tail = (r.error_tail or "").strip().splitlines()
            tail_one = tail[-1] if tail else "(no stderr)"
            lines.append(f"- {r.config.label}: {tail_one}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def _parse_int_list(name: str, raw: str, ap: argparse.ArgumentParser) -> list[int]:
    out: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            ap.error(f"--{name} expects comma-separated integers (got {tok!r})")
    if not out:
        ap.error(f"--{name} must contain at least one integer")
    return out


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
                    help="Include unsafe eval_batch values; requires "
                         "--allow-unsafe-eval-batch (kill-on-hang protected).")
    ap.add_argument("--timeout-multiplier", type=float, default=5.0,
                    help="kill if wallclock > T x baseline_wallclock")
    ap.add_argument("--min-timeout", type=float, default=60.0,
                    help="floor for the kill timeout (seconds)")
    ap.add_argument("--workdir", default="/tmp/probe_bench",
                    help="working directory for per-config outputs")
    ap.add_argument("--report-path", default=None,
                    help="if set, also write the markdown report to this path")
    # ---- Coordinate-descent knobs ----
    ap.add_argument("--workers-values", default="1,2,4,6,8",
                    help="Comma-separated worker counts to sweep "
                         "(process mode).")
    ap.add_argument("--eval-batch-values", default="8,12,14",
                    help="Comma-separated safe eval_batch values to sweep.")
    ap.add_argument("--unsafe-eval-batch-values", default="16,24",
                    help="Comma-separated unsafe eval_batch values to add "
                         "when --include-unsafe is set.")
    ap.add_argument("--stall-flush-values", default="4,8,16,32",
                    help="Comma-separated stall_flush values to sweep.")
    ap.add_argument("--passes", type=int, default=2,
                    help="Coordinate-descent passes (early-exit on convergence).")
    ap.add_argument("--start-workers", type=int, default=2,
                    help="Initial center value for workers.")
    ap.add_argument("--start-eval-batch", type=int, default=14,
                    help="Initial center value for eval_batch.")
    ap.add_argument("--start-stall-flush", type=int, default=16,
                    help="Initial center value for stall_flush.")
    args = ap.parse_args()

    args.workers_values = _parse_int_list("workers-values",
                                          args.workers_values, ap)
    args.eval_batch_values = _parse_int_list("eval-batch-values",
                                             args.eval_batch_values, ap)
    args.unsafe_eval_batch_values = _parse_int_list(
        "unsafe-eval-batch-values", args.unsafe_eval_batch_values, ap)
    args.stall_flush_values = _parse_int_list("stall-flush-values",
                                              args.stall_flush_values, ap)
    for w in args.workers_values:
        if w < 1:
            ap.error(f"--workers-values entries must be >= 1 (got {w})")
    if args.passes < 1:
        ap.error(f"--passes must be >= 1 (got {args.passes})")

    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    print(f"[bench] coordinate descent: passes={args.passes}, "
          f"workers={args.workers_values}, eval_batch={args.eval_batch_values}, "
          f"stall_flush={args.stall_flush_values}")
    print(f"[bench] start center: workers={args.start_workers}, "
          f"eval_batch={args.start_eval_batch}, "
          f"stall_flush={args.start_stall_flush}")
    print(f"[bench] workdir: {workdir}")
    if args.include_unsafe:
        print(f"[bench] --include-unsafe enabled: eval_batch in "
              f"{args.unsafe_eval_batch_values} will be probed under "
              "kill-on-hang watchdog")

    bench = Bench(args, workdir)
    all_results, center = bench.run()

    if bench.baseline_wallclock is None:
        print("[bench] ERROR: baseline serial run didn't complete; cannot "
              "produce a comparative report.", file=sys.stderr)
        if bench.trace:
            partial = _format_partial_report(bench.trace)
            print()
            print(partial)
            if args.report_path:
                Path(args.report_path).write_text(partial)
                print(f"\n[bench] partial report saved to {args.report_path}")
        return 1

    report = _format_report(bench.trace, center, bench.baseline_wallclock,
                            all_results)
    print()
    print(report)
    if args.report_path:
        Path(args.report_path).write_text(report)
        print(f"\n[bench] report saved to {args.report_path}")

    return 130 if bench.interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
