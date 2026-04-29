# Probe Suite Phase 2 — Parallel Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in process-pool execution mode for Phase 2 of `scripts/build_probe_suite.py --tier strong_advantage`, plus user-tunable `MCTSConfig` knobs and a borderline serial-rerun pass that preserves admission decisions.

**Architecture:** A new `--label-worker-mode {serial,process}` toggles between the existing serial loop (default, unchanged) and a `ProcessPoolExecutor` that fans candidate labeling across N worker processes. Each worker holds its own loaded MLX network and `MCTSConfig`. After the pool returns, results are sorted by probe ID; any candidate within ε of an admission threshold is re-labeled in the main process so threshold-sensitive admission decisions match the serial reference path. New flags expose `MCTSConfig.eval_batch_size` and `stall_flush_sims`, capped at 14 unless `--allow-unsafe-eval-batch` is passed.

**Tech Stack:** Python 3.14, MLX (Apple Metal), `concurrent.futures.ProcessPoolExecutor` with `multiprocessing.get_context("spawn")`, `dataclasses.replace` for per-call sim overrides, pytest.

**Spec:** [`docs/superpowers/specs/2026-04-28-probe-phase2-parallel-labeling-design.md`](../specs/2026-04-28-probe-phase2-parallel-labeling-design.md). Read it before starting Task 1; the plan refers to spec sections (e.g. "spec §9") rather than re-deriving every decision.

---

## File Structure

| File | Type | Responsibility |
|---|---|---|
| `scripts/GPU/alphazero/probe_eval.py` | modify | Add `_DEFAULT_LABELER_MCTS_CONFIG` global + `_set_default_labeler_mcts_config(cfg)` setter; modify `_default_mcts_labeler` to read both globals via `dataclasses.replace`. |
| `scripts/build_probe_suite.py` | modify | Add new CLI flags + validation; refactor the Phase 2 loop body into `_label_one_strong_advantage_candidate(cand, ...)`; add `_init_label_worker(...)`; add `_is_borderline(...)` helper; add the process-pool branch, the borderline-rerun pass, the Phase 2 summary block, the `meta.phase2_run_stats` block, and the `_`-prefixed-key strip before serialization. |
| `tests/test_probe_phase2_parallel.py` | create | All new tests for the parallel path, MCTSConfig wiring, borderline rerun, CLI validation, and argparse introspection. |
| `tests/probes/golden/phase2_serial_tiny.json` | create | Tiny golden fixture used by `test_phase2_serial_unchanged`. Generated once, committed alongside the code. |
| `tests/probes/golden/phase2_serial_tiny_input.json` | create | Frozen input candidates the fixture is generated from. |
| `docs/probe-suite-generation.md` | modify | Knob table additions, new "Parallel labeling" subsection under Performance, recommended-starting-values block, byte-reference clarification, example-command update. |
| `tests/probes/README.md` | modify | One-line cross-reference back to `docs/probe-suite-generation.md` for the new flags (no full duplication). |
| `scripts/probes/verify_parallel_equivalence.py` | create | Manual-only script (NOT in CI) that runs serial vs process modes on a small sample of real candidates and reports admitted-ID equivalence + label tolerance. |

---

## Task 1: Add MCTSConfig wiring globals to probe_eval.py

**Files:**
- Modify: `scripts/GPU/alphazero/probe_eval.py:953-990`
- Test: `tests/test_probe_phase2_parallel.py` (new file)

Spec §8.

- [ ] **Step 1: Create the new test file with the two MCTSConfig wiring unit tests**

Create `tests/test_probe_phase2_parallel.py`:

```python
"""Tests for Phase 2 parallel labeling, MCTSConfig wiring, borderline rerun.

See docs/superpowers/specs/2026-04-28-probe-phase2-parallel-labeling-design.md
"""
from __future__ import annotations

from dataclasses import replace as dc_replace
from unittest.mock import patch

import pytest

from scripts.GPU.alphazero.mcts import MCTSConfig
from scripts.GPU.alphazero import probe_eval


def test_default_mcts_labeler_uses_global_config(monkeypatch):
    """When _DEFAULT_LABELER_MCTS_CONFIG is set, _default_mcts_labeler builds
    MCTSConfig with eval_batch_size and stall_flush_sims from the global,
    while n_simulations is overridden by the per-call sims argument.
    """
    captured_cfg = {}

    class FakeNet:
        pass

    class FakeMCTS:
        def __init__(self, evaluator, cfg, rng):
            captured_cfg["cfg"] = cfg

        def search(self, state, add_noise=False):
            return ({(0, 0): 7}, 0.5)

    monkeypatch.setattr(probe_eval, "_DEFAULT_LABELER_NETWORK", FakeNet())
    monkeypatch.setattr(
        probe_eval,
        "_DEFAULT_LABELER_MCTS_CONFIG",
        MCTSConfig(eval_batch_size=8, stall_flush_sims=4),
    )
    monkeypatch.setattr(probe_eval, "MCTS", FakeMCTS)
    monkeypatch.setattr(probe_eval, "LocalGPUEvaluator", lambda net: None)

    probe_eval._default_mcts_labeler(state=None, sims=2000, seed=42)

    cfg = captured_cfg["cfg"]
    assert cfg.eval_batch_size == 8
    assert cfg.stall_flush_sims == 4
    assert cfg.n_simulations == 2000


def test_default_mcts_labeler_back_compat_without_global(monkeypatch):
    """When _DEFAULT_LABELER_MCTS_CONFIG is None, _default_mcts_labeler builds
    MCTSConfig(n_simulations=sims) using dataclass defaults — preserves existing
    behavior for anything that hasn't been migrated to the new setter.
    """
    captured_cfg = {}

    class FakeNet:
        pass

    class FakeMCTS:
        def __init__(self, evaluator, cfg, rng):
            captured_cfg["cfg"] = cfg

        def search(self, state, add_noise=False):
            return ({(0, 0): 1}, 0.0)

    monkeypatch.setattr(probe_eval, "_DEFAULT_LABELER_NETWORK", FakeNet())
    monkeypatch.setattr(probe_eval, "_DEFAULT_LABELER_MCTS_CONFIG", None)
    monkeypatch.setattr(probe_eval, "MCTS", FakeMCTS)
    monkeypatch.setattr(probe_eval, "LocalGPUEvaluator", lambda net: None)

    probe_eval._default_mcts_labeler(state=None, sims=500, seed=0)

    cfg = captured_cfg["cfg"]
    assert cfg.n_simulations == 500
    assert cfg.eval_batch_size == 14   # MCTSConfig default
    assert cfg.stall_flush_sims == 16  # MCTSConfig default


def test_default_mcts_labeler_keeps_sims_authoritative(monkeypatch):
    """Even if _DEFAULT_LABELER_MCTS_CONFIG was constructed with a custom
    n_simulations, the per-call sims argument wins (via dataclasses.replace).
    """
    captured_cfg = {}

    class FakeNet:
        pass

    class FakeMCTS:
        def __init__(self, evaluator, cfg, rng):
            captured_cfg["cfg"] = cfg

        def search(self, state, add_noise=False):
            return ({(0, 0): 1}, 0.0)

    monkeypatch.setattr(probe_eval, "_DEFAULT_LABELER_NETWORK", FakeNet())
    monkeypatch.setattr(
        probe_eval,
        "_DEFAULT_LABELER_MCTS_CONFIG",
        MCTSConfig(n_simulations=99999, eval_batch_size=12),
    )
    monkeypatch.setattr(probe_eval, "MCTS", FakeMCTS)
    monkeypatch.setattr(probe_eval, "LocalGPUEvaluator", lambda net: None)

    probe_eval._default_mcts_labeler(state=None, sims=1234, seed=0)

    cfg = captured_cfg["cfg"]
    assert cfg.n_simulations == 1234
    assert cfg.eval_batch_size == 12
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_probe_phase2_parallel.py -v`
Expected: FAIL with `AttributeError: module 'scripts.GPU.alphazero.probe_eval' has no attribute '_DEFAULT_LABELER_MCTS_CONFIG'`.

- [ ] **Step 3: Add the global, setter, and `dataclasses.replace` import to probe_eval.py**

Find the existing module-level `_DEFAULT_LABELER_NETWORK = None` block at `scripts/GPU/alphazero/probe_eval.py:984` and add the companion global, setter, and import. Also modify `_default_mcts_labeler` to read it via `replace(...)`.

In the imports block at the top of `probe_eval.py`, add:

```python
from dataclasses import replace
```

Replace the existing `_default_mcts_labeler` (lines ~953–981) with:

```python
def _default_mcts_labeler(state, sims, seed):
    """Production deep-MCTS labeler. Uses the network registered via
    _set_default_labeler_network() and the MCTSConfig registered via
    _set_default_labeler_mcts_config() (or MCTSConfig(n_simulations=sims)
    if no config global is registered, for back-compat).

    The per-call `sims` argument is always authoritative — it overrides
    n_simulations on the registered config via dataclasses.replace.

    Returns (root_value_from_stm_perspective, top1_visit_share).

    Tests should pass an explicit `labeler=` rather than rely on this.
    """
    if _DEFAULT_LABELER_NETWORK is None:
        raise RuntimeError(
            "Default MCTS labeler called without a registered network. "
            "Either pass labeler= explicitly or call "
            "_set_default_labeler_network() first."
        )
    evaluator = LocalGPUEvaluator(_DEFAULT_LABELER_NETWORK)
    if _DEFAULT_LABELER_MCTS_CONFIG is None:
        cfg = MCTSConfig(n_simulations=sims)
    else:
        cfg = replace(_DEFAULT_LABELER_MCTS_CONFIG, n_simulations=sims)
    mcts = MCTS(evaluator, cfg, rng=random.Random(seed))
    visit_counts, root_value = mcts.search(state, add_noise=False)
    if not visit_counts:
        return float(root_value), 0.0
    total = sum(visit_counts.values()) or 1
    top1 = max(visit_counts.values())
    return float(root_value), top1 / total


_DEFAULT_LABELER_NETWORK = None
_DEFAULT_LABELER_MCTS_CONFIG = None


def _set_default_labeler_network(network) -> None:
    """Register the production network for `_default_mcts_labeler`."""
    global _DEFAULT_LABELER_NETWORK
    _DEFAULT_LABELER_NETWORK = network


def _set_default_labeler_mcts_config(config) -> None:
    """Register MCTSConfig (eval_batch_size, stall_flush_sims) for the
    production labeler. n_simulations on this config is ignored — per-call
    sims is authoritative via dataclasses.replace inside _default_mcts_labeler.
    """
    global _DEFAULT_LABELER_MCTS_CONFIG
    _DEFAULT_LABELER_MCTS_CONFIG = config
```

- [ ] **Step 4: Run the new tests and verify they pass**

Run: `.venv/bin/python -m pytest tests/test_probe_phase2_parallel.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the full existing probe-eval test suite to ensure no regression**

Run: `.venv/bin/python -m pytest tests/test_strong_advantage_probe_suite.py tests/test_strong_advantage_diversity_selector.py -v`
Expected: all pre-existing tests still pass (the back-compat path is exercised because none of them call the new setter).

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/probe_eval.py tests/test_probe_phase2_parallel.py
git commit -m "feat(probes): add MCTSConfig wiring global to default labeler

Adds _DEFAULT_LABELER_MCTS_CONFIG and _set_default_labeler_mcts_config()
mirroring the existing _DEFAULT_LABELER_NETWORK pattern. _default_mcts_labeler
now reads eval_batch_size and stall_flush_sims from the registered config
(when set), with per-call sims kept authoritative via dataclasses.replace.

Back-compat: when no config is registered, behavior is identical to before
(MCTSConfig(n_simulations=sims) with dataclass defaults).

Spec: docs/superpowers/specs/2026-04-28-probe-phase2-parallel-labeling-design.md §8"
```

---

## Task 2: Add new CLI flags and validation to build_probe_suite.py

**Files:**
- Modify: `scripts/build_probe_suite.py:243-271` (argparse block)
- Modify: `scripts/build_probe_suite.py` top of file (add `SAFE_METAL_EVAL_BATCH_SIZE_MAX` constant)
- Test: `tests/test_probe_phase2_parallel.py` (extend)

Spec §5.

- [ ] **Step 1: Append CLI validation + introspection tests to `tests/test_probe_phase2_parallel.py`**

Append to `tests/test_probe_phase2_parallel.py`:

```python
import importlib
import subprocess
import sys
from pathlib import Path


PROBE_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_probe_suite.py"
PYTHON = sys.executable


def _run_cli(*args):
    """Run build_probe_suite.py with given CLI args. Returns the full
    CompletedProcess (rc, stdout, stderr accessible)."""
    return subprocess.run(
        [PYTHON, str(PROBE_SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_argparse_flags_present():
    """Each new flag is registered with the documented default and choices."""
    spec = importlib.util.spec_from_file_location("build_probe_suite", PROBE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    parser = mod._build_arg_parser()  # introduced in this task; see step 4

    flags = {a.dest: a for a in parser._actions}
    assert flags["label_worker_mode"].choices == ["serial", "process"]
    assert flags["label_worker_mode"].default == "serial"
    assert flags["label_workers"].default == 1
    assert flags["mcts_eval_batch_size"].default == 14
    assert flags["mcts_stall_flush_sims"].default == 16
    assert flags["allow_unsafe_eval_batch"].default is False
    assert flags["admission_borderline_epsilon"].default == 0.01
    assert flags["no_borderline_rerun"].default is False


@pytest.mark.parametrize(
    "extra_args, fragment",
    [
        (["--label-workers", "0"], "label-workers"),
        (["--mcts-eval-batch-size", "0"], "mcts-eval-batch-size"),
        (["--mcts-stall-flush-sims", "-1"], "mcts-stall-flush-sims"),
        (["--admission-borderline-epsilon", "-0.1"], "admission-borderline-epsilon"),
    ],
)
def test_parallel_cli_numeric_validation(extra_args, fragment):
    """Reject negative / zero values where the spec disallows them."""
    proc = _run_cli(
        "--tier", "strong_advantage",
        "--source-iter-range", "0", "0",
        "--label-checkpoint", "/nonexistent.safetensors",
        *extra_args,
    )
    assert proc.returncode != 0
    assert fragment in proc.stderr


def test_unsafe_eval_batch_validation_rejects_high_value_without_flag():
    """--mcts-eval-batch-size > 14 without --allow-unsafe-eval-batch is an error."""
    proc = _run_cli(
        "--tier", "strong_advantage",
        "--source-iter-range", "0", "0",
        "--label-checkpoint", "/nonexistent.safetensors",
        "--mcts-eval-batch-size", "32",
    )
    assert proc.returncode != 0
    assert "--mcts-eval-batch-size > 14 is unsafe" in proc.stderr
    assert "--allow-unsafe-eval-batch" in proc.stderr


def test_unsafe_eval_batch_passes_with_flag():
    """--mcts-eval-batch-size > 14 with --allow-unsafe-eval-batch passes validation
    (run still fails because checkpoint doesn't exist, but past argparse).
    """
    proc = _run_cli(
        "--tier", "strong_advantage",
        "--source-iter-range", "0", "0",
        "--label-checkpoint", "/nonexistent.safetensors",
        "--mcts-eval-batch-size", "32",
        "--allow-unsafe-eval-batch",
    )
    # Argparse accepts the combination; run fails later because the checkpoint
    # doesn't exist. The argparse error message must NOT appear.
    assert "--mcts-eval-batch-size > 14 is unsafe" not in proc.stderr
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run: `.venv/bin/python -m pytest tests/test_probe_phase2_parallel.py::test_argparse_flags_present -v`
Expected: FAIL with `AttributeError: module 'build_probe_suite' has no attribute '_build_arg_parser'`.

- [ ] **Step 3: Add `SAFE_METAL_EVAL_BATCH_SIZE_MAX` constant near top of `scripts/build_probe_suite.py`**

After the `import` block (around line 23, before `# --- Diversity selector constants and helpers ---`), add:

```python
# Maximum MCTS evaluator batch size known stable on Apple Metal/MLX.
# scripts/GPU/alphazero/mcts.py:106 documents that batches > this value have
# previously caused Metal GPU hangs. The probe builder caps --mcts-eval-batch-size
# at this value unless --allow-unsafe-eval-batch is passed.
SAFE_METAL_EVAL_BATCH_SIZE_MAX = 14
```

- [ ] **Step 4: Refactor `main()` to extract a `_build_arg_parser()` factory**

In `scripts/build_probe_suite.py`, factor the argparse setup out of `main()` so tests can introspect it without invoking the run. Replace the existing `def main() -> int:` opening through the `args = ap.parse_args()` line (~243–272) with:

```python
def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--tier", choices=["forced", "strong_advantage"], required=True)
    ap.add_argument("--input", default="scripts/GPU/logs/games")
    ap.add_argument("--source-iter-range", nargs=2, type=int,
                    metavar=("MIN", "MAX"))
    ap.add_argument("--out", default=None,
                    help="Output path. Defaults: forced -> tests/probes/twixt_probes.json, "
                         "strong_advantage -> tests/probes/strong_advantage_probes.json")
    ap.add_argument("--samples-per-bucket", type=int, default=12)
    ap.add_argument("--max-probes", type=int, default=30)
    ap.add_argument("--max-probes-per-game", type=int, default=2,
                    help="Maximum number of admitted probes from any single "
                         "source game. Counts total across all 4 categories. "
                         "Default 2. Strong-advantage tier only.")

    # strong_advantage-specific flags (ignored for forced)
    ap.add_argument("--label-checkpoint", default=None)
    ap.add_argument("--label-mcts-sims", type=int, default=10000)
    ap.add_argument("--label-mcts-repeats", type=int, default=3)
    ap.add_argument("--magnitude-threshold", type=float, default=0.45)
    ap.add_argument("--top1-share-floor", type=float, default=0.15)
    ap.add_argument("--stability-cap", type=float, default=0.15)
    ap.add_argument("--promote", action="store_true",
                    help="Promote *.draft.json to committed file")
    ap.add_argument("--reviewer", default=None,
                    help="Reviewer name, required with --promote")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing draft or committed file")

    # Phase 2 parallel-labeling flags (strong_advantage tier only)
    ap.add_argument("--label-worker-mode", choices=["serial", "process"],
                    default="serial",
                    help="Phase 2 execution mode. Default 'serial' is the "
                         "byte-reference path. 'process' enables a process pool.")
    ap.add_argument("--label-workers", type=int, default=1,
                    help="Worker count under --label-worker-mode=process. "
                         "Ignored under serial. Apple Silicon: start with 2-4.")
    ap.add_argument("--mcts-eval-batch-size", type=int, default=14,
                    help=(f"NN batch size for the labeler's MCTS. Capped at "
                          f"{SAFE_METAL_EVAL_BATCH_SIZE_MAX} because larger "
                          "batches have caused Metal hangs; pass "
                          "--allow-unsafe-eval-batch to exceed."))
    ap.add_argument("--mcts-stall-flush-sims", type=int, default=16,
                    help="MCTS stall-flush threshold (see MCTSConfig). 0 disables.")
    ap.add_argument("--allow-unsafe-eval-batch", action="store_true",
                    help="Required to set --mcts-eval-batch-size > "
                         f"{SAFE_METAL_EVAL_BATCH_SIZE_MAX}. Benchmark only.")
    ap.add_argument("--admission-borderline-epsilon", type=float, default=0.01,
                    help="In process mode, candidates whose phase-2 label is "
                         "within epsilon of any admission threshold are "
                         "re-labeled in the main process to use the serial "
                         "reference label. 0 disables.")
    ap.add_argument("--no-borderline-rerun", action="store_true",
                    help="Disable borderline rerun even when epsilon > 0.")
    return ap


def _validate_parallel_args(ap: argparse.ArgumentParser, args) -> None:
    """Validate the new Phase 2 parallel flags. Calls ap.error() on failure."""
    if args.label_workers < 1:
        ap.error("--label-workers must be >= 1")
    if args.mcts_eval_batch_size < 1:
        ap.error("--mcts-eval-batch-size must be >= 1")
    if (args.mcts_eval_batch_size > SAFE_METAL_EVAL_BATCH_SIZE_MAX
            and not args.allow_unsafe_eval_batch):
        ap.error(
            f"--mcts-eval-batch-size > {SAFE_METAL_EVAL_BATCH_SIZE_MAX} "
            "is unsafe on Metal/MLX and may hang. "
            "Pass --allow-unsafe-eval-batch to benchmark higher values intentionally."
        )
    if args.mcts_stall_flush_sims < 0:
        ap.error("--mcts-stall-flush-sims must be >= 0")
    if args.admission_borderline_epsilon < 0:
        ap.error("--admission-borderline-epsilon must be >= 0")


def main() -> int:
    ap = _build_arg_parser()
    args = ap.parse_args()
    _validate_parallel_args(ap, args)

    # Workers under serial mode: warn if explicitly set to anything other than 1.
    if args.label_worker_mode == "serial" and args.label_workers != 1:
        print("[probe_suite] warning: --label-workers is ignored when "
              "--label-worker-mode=serial", file=sys.stderr)
        args.label_workers = 1

    project_root = Path(__file__).resolve().parent.parent
```

(Keep the rest of `main()` body — `if args.tier == "forced": ...` etc. — exactly as it was.)

- [ ] **Step 5: Run all tests added in this task and verify they pass**

Run: `.venv/bin/python -m pytest tests/test_probe_phase2_parallel.py -v`
Expected: all pass (Task 1 tests + the 4 new flag tests in this task).

- [ ] **Step 6: Run existing strong-advantage suite to confirm no regression**

Run: `.venv/bin/python -m pytest tests/test_strong_advantage_probe_suite.py tests/test_strong_advantage_diversity_selector.py -v`
Expected: all pass.

- [ ] **Step 7: Sanity-check `--help` output manually**

Run: `.venv/bin/python scripts/build_probe_suite.py --help | grep -E "label-worker|mcts-eval|mcts-stall|allow-unsafe|borderline"`
Expected: 7 lines, one per new flag, each with the help text from step 4.

- [ ] **Step 8: Commit**

```bash
git add scripts/build_probe_suite.py tests/test_probe_phase2_parallel.py
git commit -m "feat(probes): add Phase 2 parallel CLI flags and validation

Adds --label-worker-mode, --label-workers, --mcts-eval-batch-size,
--mcts-stall-flush-sims, --allow-unsafe-eval-batch,
--admission-borderline-epsilon, --no-borderline-rerun. All defaults
preserve current behavior. eval_batch_size > 14 requires explicit
--allow-unsafe-eval-batch (Metal hang protection). Refactors main() to
extract _build_arg_parser() so tests can introspect it.

Spec: docs/superpowers/specs/2026-04-28-probe-phase2-parallel-labeling-design.md §5"
```

---

## Task 3: Extract Phase 2 loop body into `_label_one_strong_advantage_candidate` helper

**Files:**
- Modify: `scripts/build_probe_suite.py:467-577` (current Phase 2 loop body)
- Test: `tests/test_strong_advantage_probe_suite.py` — existing tests must continue to pass byte-identically.

Spec §7. **No CLI behavior change in this task** — the helper is called from the existing serial loop; output must remain byte-identical to before.

- [ ] **Step 1: Add a regression-anchor test that asserts existing serial-mode output is unchanged**

This test is a structural sanity check — it doesn't generate a new fixture (Task 7 does that). Append to `tests/test_probe_phase2_parallel.py`:

```python
def test_helper_returns_admitted_status_for_passing_candidate(monkeypatch, tmp_path):
    """The extracted helper returns the same admission decision the old serial
    loop would have written. Uses a tiny mocked labeler.

    This test fails until the helper exists with the documented signature.
    """
    from scripts.build_probe_suite import _label_one_strong_advantage_candidate
    from scripts.GPU.alphazero import probe_eval

    def stub_labeler(state, sims, seed):
        # Strong red-side advantage: high mean root, high top1.
        return (0.9, 0.5)

    cand = {
        "source_game": "iter_0_game_0",
        "source_ply": 18,
        "ply": 18,
        "starting_player": "red",
        "winner": "red",
        "category": "central_red",
        "move_history": [(11, 11)],
        "phase1_features": {
            "cc_size": 12,
            "cc_axis_span": 0.7,
            "axis_span_margin": 0.2,
        },
    }

    monkeypatch.setattr(probe_eval, "_default_mcts_labeler", stub_labeler)

    result = _label_one_strong_advantage_candidate(
        cand,
        label_ckpt_name="ckpt.safetensors",
        sims=10,
        repeats=2,
        magnitude_threshold=0.45,
        top1_share_floor=0.15,
        stability_cap=0.15,
    )
    assert result["status"] == "admitted"
    assert result["candidate"] is not None
    assert result["candidate"]["phase2_label"]["mean_root_value"] == pytest.approx(0.9)
    assert result["audit_row"] is None
    assert result["error_message"] is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_probe_phase2_parallel.py::test_helper_returns_admitted_status_for_passing_candidate -v`
Expected: FAIL with `ImportError: cannot import name '_label_one_strong_advantage_candidate'`.

- [ ] **Step 3: Add the helper function and refactor the serial loop in `_run_strong_advantage`**

In `scripts/build_probe_suite.py`, add the following helper near the top of the strong-advantage section (just before `def _run_strong_advantage(args) -> int:` at line ~403). Add `import copy` and `import hashlib` if not already present at module level.

```python
def _probe_id_and_seed_base(cand: dict) -> tuple[str, int]:
    """Compute the deterministic (probe_id, rng_seed_base) pair for a candidate.

    Stable across processes because hashlib.sha256 is not subject to
    Python's randomized hash().
    """
    probe_id = _probe_id_for(cand)
    seed_base = int.from_bytes(
        hashlib.sha256(probe_id.encode("utf-8")).digest()[:4],
        "big",
    )
    return probe_id, seed_base


def _label_one_strong_advantage_candidate(
    cand: dict,
    *,
    label_ckpt_name: str,
    sims: int,
    repeats: int,
    magnitude_threshold: float,
    top1_share_floor: float,
    stability_cap: float,
) -> dict:
    """Phase 2 per-candidate labeling helper. Used by both the serial loop
    and the process-pool path.

    Returns a structured result dict. See spec §7.
    """
    from scripts.GPU.alphazero.probe_eval import (
        label_candidate_with_mcts,
        apply_admission_filter,
    )
    from scripts.GPU.alphazero.game.twixt_state import TwixtState

    cand = copy.deepcopy(cand)
    probe_id, seed_base = _probe_id_and_seed_base(cand)

    # Replay candidate moves into a TwixtState.
    try:
        state = TwixtState(active_size=24, to_move=cand["starting_player"])
        for r, c in cand["move_history"]:
            state = state.apply_move((r, c))
    except Exception as exc:
        return {
            "probe_id": probe_id,
            "status": "replay_error",
            "candidate": None,
            "audit_row": {
                "source_game": cand["source_game"],
                "source_ply": cand["source_ply"],
                "phase1_features": cand["phase1_features"],
                "reason": "replay_error",
            },
            "rejection_reason": "replay_error",
            "phase2_label": None,
            "error_message": f"{type(exc).__name__}: {exc}",
        }

    # Run MCTS labeling.
    try:
        label = label_candidate_with_mcts(
            state,
            sims=sims,
            repeats=repeats,
            rng_seed_base=seed_base,
        )
    except Exception as exc:
        return {
            "probe_id": probe_id,
            "status": "mcts_error",
            "candidate": cand,
            "audit_row": {
                "source_game": cand["source_game"],
                "source_ply": cand["source_ply"],
                "phase1_features": cand["phase1_features"],
                "reason": "mcts_error",
            },
            "rejection_reason": "mcts_error",
            "phase2_label": None,
            "error_message": f"{type(exc).__name__}: {exc}",
        }

    # Normalize STM perspective (red-perspective for downstream consumers).
    stm = _stm_at_ply(cand)
    if stm == "black":
        label["mean_root_value"] = -label["mean_root_value"]
        label["value_per_run"] = [-v for v in label["value_per_run"]]

    cand["phase2_label"] = label
    ok, reason = apply_admission_filter(
        cand,
        magnitude_threshold=magnitude_threshold,
        top1_share_floor=top1_share_floor,
        stability_cap=stability_cap,
    )
    cand["phase2_label"]["label_checkpoint"] = label_ckpt_name

    if ok:
        return {
            "probe_id": probe_id,
            "status": "admitted",
            "candidate": cand,
            "audit_row": None,
            "rejection_reason": None,
            "phase2_label": cand["phase2_label"],
            "error_message": None,
        }
    return {
        "probe_id": probe_id,
        "status": "rejected",
        "candidate": cand,
        "audit_row": {
            "source_game": cand["source_game"],
            "source_ply": cand["source_ply"],
            "phase1_features": cand["phase1_features"],
            "phase2_label": cand["phase2_label"],
            "reason": reason,
        },
        "rejection_reason": reason,
        "phase2_label": cand["phase2_label"],
        "error_message": None,
    }
```

Now replace the existing Phase 2 loop body in `_run_strong_advantage` (the block starting around line 486 with `for idx, cand in enumerate(candidates):` through line ~577 ending the per-candidate `else: audit.append(...)`) with a call into the helper. Keep the progress-logging / ETA block exactly as it was; only the per-candidate labeling guts change:

```python
    admitted = []
    import time as _time
    n_total = len(candidates)
    progress_every = max(1, n_total // 20)
    t_phase2_start = _time.time()
    for idx, cand in enumerate(candidates):
        if idx % progress_every == 0:
            elapsed = _time.time() - t_phase2_start
            n_admitted = len(admitted)
            if idx > 0:
                rate = idx / elapsed
                eta_s = (n_total - idx) / rate if rate > 0 else 0.0
                eta_str = f"ETA {eta_s/60:.1f}m" if eta_s < 3600 else f"ETA {eta_s/3600:.1f}h"
            else:
                eta_str = "ETA --"
            print(
                f"[probe_suite] Phase 2: {idx}/{n_total} labeled "
                f"({n_admitted} admitted, {elapsed:.0f}s elapsed, {eta_str})",
                flush=True,
            )

        result = _label_one_strong_advantage_candidate(
            cand,
            label_ckpt_name=label_ckpt.name,
            sims=args.label_mcts_sims,
            repeats=args.label_mcts_repeats,
            magnitude_threshold=args.magnitude_threshold,
            top1_share_floor=args.top1_share_floor,
            stability_cap=args.stability_cap,
        )

        if result["status"] == "replay_error":
            print(f"[probe_suite] WARN: state replay error on "
                  f"{cand['source_game']} ply {cand['source_ply']}: "
                  f"{result['error_message']}", file=sys.stderr)
            audit.append(result["audit_row"])
        elif result["status"] == "mcts_error":
            print(f"[probe_suite] WARN: MCTS error on "
                  f"{cand['source_game']} ply {cand['source_ply']}: "
                  f"{result['error_message']}", file=sys.stderr)
            audit.append(result["audit_row"])
        elif result["status"] == "admitted":
            admitted.append(result["candidate"])
        else:  # rejected
            audit.append(result["audit_row"])
```

- [ ] **Step 4: Run the new helper test and verify it passes**

Run: `.venv/bin/python -m pytest tests/test_probe_phase2_parallel.py::test_helper_returns_admitted_status_for_passing_candidate -v`
Expected: PASS.

- [ ] **Step 5: Run the full existing strong-advantage suite and verify byte-identical behavior**

Run: `.venv/bin/python -m pytest tests/test_strong_advantage_probe_suite.py tests/test_strong_advantage_diversity_selector.py tests/test_probe_suite_forced_parity.py -v`
Expected: all pre-existing tests still pass. The refactor is byte-identical.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_probe_suite.py tests/test_probe_phase2_parallel.py
git commit -m "refactor(probes): extract Phase 2 body into _label_one_strong_advantage_candidate

Pure refactor of the strong-advantage Phase 2 loop body into a helper
that returns a structured result dict. Serial loop continues to call it
exactly as before; behavior is byte-identical. Sets the stage for the
process-pool path in the next commit.

Spec: docs/superpowers/specs/2026-04-28-probe-phase2-parallel-labeling-design.md §7"
```

---

## Task 4: Add process-pool labeling path with worker initializer

**Files:**
- Modify: `scripts/build_probe_suite.py` (add `_init_label_worker`, branch on `args.label_worker_mode`, register MCTSConfig in main and worker)
- Test: `tests/test_probe_phase2_parallel.py` (extend)

Spec §6, §8, §11.

- [ ] **Step 1: Add the aggregation unit test, the process-pool smoke test, and the fixed init-failure test**

**Note on `spawn` workers and monkeypatch.** `multiprocessing.get_context("spawn")` workers import all modules fresh in the child process. `monkeypatch.setattr(probe_eval, "_default_mcts_labeler", stub)` only affects the parent's module instance — workers will see the *original* `_default_mcts_labeler`, not the stub. Tests that need to control labeling in workers must therefore either (a) test the helper as a unit in the parent (what `test_phase2_replay_error_isolated` and `test_phase2_mcts_error_isolated` do, see Step 4), (b) test the aggregation logic on hand-crafted result lists (the `_phase2_aggregate` unit test below), or (c) replace the *worker initializer itself* with a top-level test function via late-bound module-attribute monkeypatch (the smoke test below). The implementation must reference `_init_label_worker` via the module globals (not via a local `from … import`) so the monkeypatched binding is what `ProcessPoolExecutor(initializer=...)` actually pickles.

Append to `tests/test_probe_phase2_parallel.py`. Two of these reference helpers introduced in the implementation (Step 3): `_phase2_aggregate(results)` (returns the `(admitted, audit_rows)` tuple from a sorted result list) and the module-attribute lookup of `_init_label_worker`.

```python
# Module-level so spawn workers can pickle and import this function:
def _init_label_worker_stub_for_tests(label_checkpoint, mcts_cfg_payload):
    """Test-only worker initializer. Installs a deterministic stub labeler
    in the worker process instead of loading a real network. Imported by
    spawned children via its qualified name."""
    from scripts.GPU.alphazero import probe_eval
    from scripts.GPU.alphazero.mcts import MCTSConfig

    def _stub(state, sims, seed):
        return (0.6 + 0.001 * (seed % 7), 0.4 + 0.001 * (seed % 5))

    probe_eval._default_mcts_labeler = _stub
    probe_eval._set_default_labeler_network(object())
    probe_eval._set_default_labeler_mcts_config(MCTSConfig(**mcts_cfg_payload))


def _patch_phase1_extract(monkeypatch, candidates):
    """Bypass real Phase 1 mining; return the given candidate list.
    Mirrors the pattern in tests/test_strong_advantage_probe_suite.py."""
    monkeypatch.setattr(
        "scripts.GPU.alphazero.probe_eval.extract_strong_advantage_candidates",
        lambda games, **kw: (list(candidates), []),
    )


_SAMPLE_CENTRAL = {
    "move_history": [(0, 12), (1, 0), (2, 11), (1, 1)],
    "ply": 4, "winner": "red",
    "category": "chain_advantage_central_red",
    "phase1_features": {
        "cc_size": 12, "cc_axis_span": 0.65, "cc_touches_own_goal": True,
        "axis_span_margin": 0.20, "centroid_chebyshev_from_center": 4,
        "forced_within_2": False,
    },
    "source_game": "iter_0070_game_001", "source_ply": 4,
    "starting_player": "red",
}
_SAMPLE_EDGE = {
    "move_history": [(0, 1), (1, 22), (2, 0), (1, 21)],
    "ply": 4, "winner": "red",
    "category": "chain_advantage_edge_red",
    "phase1_features": {
        "cc_size": 11, "cc_axis_span": 0.60, "cc_touches_own_goal": True,
        "axis_span_margin": 0.15, "centroid_chebyshev_from_center": 10,
        "forced_within_2": False,
    },
    "source_game": "iter_0070_game_002", "source_ply": 4,
    "starting_player": "red",
}


def test_phase2_aggregate_sorts_by_probe_id():
    """Hand-crafted result list (as if returned in arbitrary completion order
    from a process pool) must be aggregated in probe_id-sorted order. This
    test validates the aggregation logic without spawning a real pool."""
    from scripts.build_probe_suite import _phase2_aggregate

    r_a = {
        "probe_id": "p_alpha", "status": "admitted",
        "candidate": {"source_game": "g0", "source_ply": 4, "marker": "alpha"},
        "audit_row": None, "rejection_reason": None,
        "phase2_label": {"mean_root_value": 0.7},
        "error_message": None,
    }
    r_b = {
        "probe_id": "p_beta", "status": "rejected",
        "candidate": {"source_game": "g1", "source_ply": 4, "marker": "beta"},
        "audit_row": {"source_game": "g1", "source_ply": 4,
                      "reason": "magnitude_below_threshold"},
        "rejection_reason": "magnitude_below_threshold",
        "phase2_label": {"mean_root_value": 0.1},
        "error_message": None,
    }
    r_c = {
        "probe_id": "p_gamma", "status": "replay_error",
        "candidate": None,
        "audit_row": {"source_game": "g2", "source_ply": 4,
                      "reason": "replay_error"},
        "rejection_reason": "replay_error",
        "phase2_label": None,
        "error_message": "ValueError: bad",
    }

    # Pass in deliberately scrambled order.
    admitted, audit_rows = _phase2_aggregate([r_c, r_a, r_b])

    assert [c["marker"] for c in admitted] == ["alpha"]
    # Audit rows in probe_id-sorted order: alpha (no row), beta, gamma.
    assert [a["source_game"] for a in audit_rows] == ["g1", "g2"]


def test_phase2_process_pool_smoke(tmp_path, monkeypatch):
    """Real ProcessPoolExecutor smoke test. Substitutes the production
    _init_label_worker with a top-level test function via late-bound
    module-attribute monkeypatch; that function installs a deterministic
    stub labeler in each worker. Verifies the pool launches, all candidates
    label, no exception escapes."""
    import json as _json
    import scripts.build_probe_suite as bps
    from tests.test_probe_phase2_parallel import _init_label_worker_stub_for_tests

    monkeypatch.setattr(bps, "_init_label_worker", _init_label_worker_stub_for_tests)
    _patch_phase1_extract(monkeypatch, [_SAMPLE_CENTRAL, _SAMPLE_EDGE])

    fake_ckpt = tmp_path / "fake_ckpt.safetensors"
    fake_ckpt.write_bytes(b"stub")

    out_path = tmp_path / "strong_advantage_probes.json"
    rc = bps.main_with_args([
        "--tier", "strong_advantage",
        "--input", "scripts/GPU/logs/games",
        "--source-iter-range", "70", "70",
        "--label-checkpoint", str(fake_ckpt),
        "--label-mcts-sims", "10",
        "--label-mcts-repeats", "1",
        "--magnitude-threshold", "0.45",
        "--out", str(out_path),
        "--label-worker-mode", "process",
        "--label-workers", "2",
        "--no-borderline-rerun",
        "--force",
    ])
    assert rc == 0

    draft = _json.loads(out_path.with_suffix(".draft.json").read_text())
    stats = draft["meta"]["phase2_run_stats"]
    assert stats["mode"] == "process"
    assert stats["workers_effective"] == 2
    assert stats["candidates_total"] == 2
    assert stats["labeled"] == 2


def test_phase2_process_pool_init_failure(tmp_path, monkeypatch):
    """Worker init can't load the checkpoint -> first .result() raises ->
    main aborts with a clear error mentioning path/mode/workers; no draft
    is written. Uses the synthetic-candidate Phase-1 stub so workers
    actually start (vs. zero candidates -> pool never created)."""
    import scripts.build_probe_suite as bps
    _patch_phase1_extract(monkeypatch, [_SAMPLE_CENTRAL])

    bad_ckpt = tmp_path / "definitely_not_here.safetensors"
    bad_ckpt.write_bytes(b"")  # exists, but won't load as a network

    out_path = tmp_path / "strong_advantage_probes.json"
    rc = bps.main_with_args([
        "--tier", "strong_advantage",
        "--input", "scripts/GPU/logs/games",
        "--source-iter-range", "70", "70",
        "--label-checkpoint", str(bad_ckpt),
        "--label-mcts-sims", "10",
        "--label-mcts-repeats", "1",
        "--out", str(out_path),
        "--label-worker-mode", "process",
        "--label-workers", "2",
        "--force",
    ])
    assert rc != 0
    # Draft file MUST NOT exist on init failure.
    assert not out_path.with_suffix(".draft.json").exists()
```

- [ ] **Step 2: Run the new tests; verify init-failure test fails for the right reason**

Run: `.venv/bin/python -m pytest tests/test_probe_phase2_parallel.py::test_phase2_process_pool_init_failure -v`
Expected: FAIL — argparse currently accepts `--label-worker-mode process` but the run never enters the process branch, so the failure mode comes out wrong (or it might inadvertently pass if Phase 1 errors first; that's acceptable — we'll re-run it after Step 3).

- [ ] **Step 3: Add `_init_label_worker` and the process-pool branch in `_run_strong_advantage`**

In `scripts/build_probe_suite.py`, near the top of the file after the imports, add:

```python
def _init_label_worker(label_checkpoint: str, mcts_cfg_payload: dict) -> None:
    """ProcessPoolExecutor initializer: load network and register MCTSConfig.

    Each worker process holds its own MLX network (own MLX context) and its
    own copy of the registered MCTSConfig. n_simulations is per-call and
    NOT in the payload — see spec §8.
    """
    from scripts.GPU.alphazero.probe_eval import (
        load_network_for_scoring,
        _set_default_labeler_network,
        _set_default_labeler_mcts_config,
    )
    from scripts.GPU.alphazero.mcts import MCTSConfig
    network, _ic, _h, _nb = load_network_for_scoring(label_checkpoint)
    network.eval()
    _set_default_labeler_network(network)
    _set_default_labeler_mcts_config(MCTSConfig(**mcts_cfg_payload))
```

Then, inside `_run_strong_advantage`, replace the section that begins `network, _ic, _h, _nb = load_network_for_scoring(...)` (line ~475) and continues through the end of the per-candidate Phase 2 loop with a mode-aware version:

```python
    network, _ic, _h, _nb = load_network_for_scoring(str(label_ckpt))
    network.eval()
    from scripts.GPU.alphazero.probe_eval import (
        _set_default_labeler_network,
        _set_default_labeler_mcts_config,
    )
    from scripts.GPU.alphazero.mcts import MCTSConfig
    _set_default_labeler_network(network)
    mcts_cfg_payload = {
        "eval_batch_size": args.mcts_eval_batch_size,
        "stall_flush_sims": args.mcts_stall_flush_sims,
    }
    _set_default_labeler_mcts_config(MCTSConfig(**mcts_cfg_payload))

    admitted = []
    import time as _time
    n_total = len(candidates)
    progress_every = max(1, n_total // 20)
    t_phase2_start = _time.time()

    helper_kwargs = dict(
        label_ckpt_name=label_ckpt.name,
        sims=args.label_mcts_sims,
        repeats=args.label_mcts_repeats,
        magnitude_threshold=args.magnitude_threshold,
        top1_share_floor=args.top1_share_floor,
        stability_cap=args.stability_cap,
    )

    if args.label_worker_mode == "serial":
        results = []
        admitted_so_far = 0
        for idx, cand in enumerate(candidates):
            if idx % progress_every == 0:
                _print_phase2_progress(idx, n_total, admitted_so_far,
                                       t_phase2_start)
            r = _label_one_strong_advantage_candidate(cand, **helper_kwargs)
            results.append(r)
            if r["status"] == "admitted":
                admitted_so_far += 1
    else:  # "process"
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = _mp.get_context("spawn")
        # NOTE: the bare name `_init_label_worker` is resolved through the
        # module globals of `scripts.build_probe_suite` at this exact moment,
        # so tests can monkeypatch `bps._init_label_worker` to a top-level
        # test helper and have ProcessPoolExecutor pickle the patched
        # function (by qualified name) for the worker. Do NOT replace the
        # bare name with `from scripts.build_probe_suite import
        # _init_label_worker`; that would freeze the binding and break the
        # smoke test in tests/test_probe_phase2_parallel.py.
        try:
            with ProcessPoolExecutor(
                max_workers=args.label_workers,
                mp_context=ctx,
                initializer=_init_label_worker,
                initargs=(str(label_ckpt), mcts_cfg_payload),
            ) as pool:
                futures = [
                    pool.submit(_label_one_strong_advantage_candidate, cand,
                                **helper_kwargs)
                    for cand in candidates
                ]
                results = []
                completed = 0
                for fut in as_completed(futures):
                    results.append(fut.result())
                    completed += 1
                    if completed % progress_every == 0:
                        _print_phase2_progress(completed, n_total,
                                               sum(1 for r in results
                                                   if r["status"] == "admitted"),
                                               t_phase2_start)
        except Exception as exc:
            print(f"[probe_suite] ERROR: failed to initialize process label "
                  f"worker from checkpoint {label_ckpt}\n"
                  f"  mode={args.label_worker_mode} workers={args.label_workers}\n"
                  f"  {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    # NOTE: aggregation (sort by probe_id + partition) is deferred to
    # _phase2_aggregate(), called AFTER the borderline-rerun pass added in
    # Task 5. Until Task 5 lands, this Task 4 implementation calls it
    # immediately. After Task 5 lands, the rerun pass runs first, then
    # _phase2_aggregate() is called once on the post-rerun results. The
    # function is idempotent w.r.t. already-sorted inputs.
    admitted, new_audit_rows = _phase2_aggregate(results)
    audit.extend(new_audit_rows)
    for r in results:
        if r["status"] == "replay_error":
            print(f"[probe_suite] WARN: state replay error: "
                  f"{r['error_message']}", file=sys.stderr)
        elif r["status"] == "mcts_error":
            print(f"[probe_suite] WARN: MCTS error: "
                  f"{r['error_message']}", file=sys.stderr)
```

Add the aggregation helper at module level in `scripts/build_probe_suite.py`:

```python
def _phase2_aggregate(results: list[dict]) -> tuple[list, list]:
    """Sort Phase 2 result dicts by probe_id and partition into
    (admitted candidates, audit rows). Idempotent on already-sorted input.
    Used by both serial and process modes, and called once AFTER the
    borderline-rerun pass (Task 5) so post-rerun statuses drive the
    partition. Spec §6."""
    results.sort(key=lambda r: r["probe_id"])
    admitted = [r["candidate"] for r in results if r["status"] == "admitted"]
    audit_rows = [r["audit_row"] for r in results if r["audit_row"] is not None]
    return admitted, audit_rows
```

Also add the progress helper just above `_run_strong_advantage`:

```python
def _print_phase2_progress(idx: int, n_total: int, n_admitted: int,
                           t_start: float) -> None:
    import time as _time
    elapsed = _time.time() - t_start
    if idx > 0:
        rate = idx / elapsed
        eta_s = (n_total - idx) / rate if rate > 0 else 0.0
        eta_str = f"ETA {eta_s/60:.1f}m" if eta_s < 3600 else f"ETA {eta_s/3600:.1f}h"
    else:
        eta_str = "ETA --"
    print(
        f"[probe_suite] Phase 2: {idx}/{n_total} labeled "
        f"({n_admitted} admitted, {elapsed:.0f}s elapsed, {eta_str})",
        flush=True,
    )
```

- [ ] **Step 4: Add error-isolation tests for the helper**

`_label_one_strong_advantage_candidate` catches replay/MCTS errors and returns a structured result; the pool never sees an exception across the process boundary. So error-isolation is tested as a unit on the helper itself — no real pool needed. (Cross-process equivalence is covered by `test_phase2_aggregate_sorts_by_probe_id` plus `test_phase2_process_pool_smoke` in Step 1.)

```python
def test_phase2_replay_error_isolated(monkeypatch):
    """A candidate whose move_history can't be replayed returns
    status='replay_error' with a populated audit_row and error_message —
    the helper does NOT raise, so the pool survives."""
    from scripts.build_probe_suite import _label_one_strong_advantage_candidate

    bad_cand = {
        "source_game": "g0",
        "source_ply": 5,
        "ply": 5,
        "starting_player": "red",
        "winner": "red",
        "category": "central_red",
        # Out-of-bounds move triggers TwixtState.apply_move to raise.
        "move_history": [(999, 999)],
        "phase1_features": {"cc_size": 1, "cc_axis_span": 0.1,
                            "axis_span_margin": 0.0},
    }
    result = _label_one_strong_advantage_candidate(
        bad_cand,
        label_ckpt_name="fake.safetensors", sims=10, repeats=1,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15,
    )
    assert result["status"] == "replay_error"
    assert result["audit_row"]["reason"] == "replay_error"
    assert result["audit_row"]["source_game"] == "g0"
    assert result["error_message"]
    assert result["candidate"] is None  # spec §7 invariant
    assert result["phase2_label"] is None


def test_phase2_mcts_error_isolated(monkeypatch):
    """A labeler that raises returns status='mcts_error' with a populated
    audit_row and error_message — the helper does NOT raise."""
    from scripts.build_probe_suite import _label_one_strong_advantage_candidate
    from scripts.GPU.alphazero import probe_eval

    def angry_labeler(state, sims, seed):
        raise RuntimeError("synthetic MCTS failure")
    monkeypatch.setattr(probe_eval, "_default_mcts_labeler", angry_labeler)

    cand = {
        "source_game": "g1",
        "source_ply": 18,
        "ply": 18,
        "starting_player": "red",
        "winner": "red",
        "category": "central_red",
        "move_history": [(11, 11)],
        "phase1_features": {"cc_size": 12, "cc_axis_span": 0.7,
                            "axis_span_margin": 0.2},
    }
    result = _label_one_strong_advantage_candidate(
        cand,
        label_ckpt_name="fake.safetensors", sims=10, repeats=1,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15,
    )
    assert result["status"] == "mcts_error"
    assert result["audit_row"]["reason"] == "mcts_error"
    assert "synthetic MCTS failure" in result["error_message"]
    assert result["candidate"] is not None  # spec §7 invariant
    assert result["phase2_label"] is None
```

- [ ] **Step 5: Run all tests in the file**

Run: `.venv/bin/python -m pytest tests/test_probe_phase2_parallel.py -v`
Expected: all pass (including the equivalence and isolated-error tests).

- [ ] **Step 6: Run the full strong-advantage regression suite**

Run: `.venv/bin/python -m pytest tests/test_strong_advantage_probe_suite.py tests/test_strong_advantage_diversity_selector.py -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/build_probe_suite.py tests/test_probe_phase2_parallel.py tests/test_strong_advantage_probe_suite.py
git commit -m "feat(probes): add Phase 2 process-pool path with worker initializer

Adds --label-worker-mode=process branch in _run_strong_advantage with
ProcessPoolExecutor (spawn context), per-worker MLX network + MCTSConfig
via _init_label_worker. Results sorted by probe_id in _phase2_aggregate
for deterministic audit ordering. Isolates per-candidate replay/MCTS
errors at the helper level so the pool survives.

Aggregation and process-pool smoke tests verify deterministic result
ordering and worker initialization under spawn; helper-level tests
cover mocked label behavior. Cross-process equivalence under real MLX
is intentionally out of CI (covered by the manual
verify_parallel_equivalence.py script in Task 9).

Spec: docs/superpowers/specs/2026-04-28-probe-phase2-parallel-labeling-design.md §6, §8, §11"
```

---

## Task 5: Add borderline serial-rerun pass in main process

**Files:**
- Modify: `scripts/build_probe_suite.py` (add `_is_borderline`, rerun loop, audit-merge logic, private-key strip on serialization)
- Test: `tests/test_probe_phase2_parallel.py` (extend)

Spec §9.

- [ ] **Step 1: Add the borderline-rerun unit tests (flip + non-borderline)**

Test `_run_borderline_reruns` as a pure function on hand-crafted result lists. This avoids the cross-process labeler dispatch problem (monkeypatch affects only the parent) and tests the rerun logic in isolation. The end-to-end integration tests come in Step 2.

Append to `tests/test_probe_phase2_parallel.py`:

```python
def _make_fake_result(*, status, mean_root_value, min_top1_share=0.5,
                     value_stability=0.0, source_game="g0", source_ply=18,
                     category="chain_advantage_central_red",
                     rejection_reason=None):
    """Build a result-dict shaped like _label_one_strong_advantage_candidate
    returns. probe_id is derived from (source_game, source_ply, category) the
    same way `_probe_id_for(cand)` derives it, so when these results are fed
    back through `_run_borderline_reruns` (which re-runs the helper and
    re-derives probe_id from the candidate), the spec §9 same-probe-id
    invariant holds."""
    from scripts.build_probe_suite import _probe_id_for

    label = {
        "mean_root_value": mean_root_value,
        "value_per_run": [mean_root_value, mean_root_value],
        "value_stability": value_stability,
        "min_top1_share": min_top1_share,
        "label_mcts_sims": 10,
        "label_mcts_repeats": 2,
        "rng_seed_base": 0,
        "label_checkpoint": "fake.safetensors",
    }
    cand = {
        "source_game": source_game, "source_ply": source_ply, "ply": source_ply,
        "starting_player": "red", "winner": "red",
        "category": category, "move_history": [(11, 11)],
        "phase1_features": {
            "cc_size": 12, "cc_axis_span": 0.7, "axis_span_margin": 0.2,
        },
        "phase2_label": label,
    }
    probe_id = _probe_id_for(cand)
    audit_row = None
    if status == "rejected":
        audit_row = {
            "source_game": cand["source_game"],
            "source_ply": cand["source_ply"],
            "phase1_features": cand["phase1_features"],
            "phase2_label": label,
            "reason": rejection_reason,
        }
    return {
        "probe_id": probe_id,
        "status": status,
        "candidate": cand,
        "audit_row": audit_row,
        "rejection_reason": rejection_reason,
        "phase2_label": label,
        "error_message": None,
    }


def test_borderline_rerun_admitted_to_rejected(monkeypatch):
    """Parallel result was admitted at just-above magnitude. Main-process
    rerun returns just-below. Status flips to rejected; audit metadata
    records the flip with old/new reasons."""
    from scripts.build_probe_suite import _run_borderline_reruns
    from scripts.GPU.alphazero import probe_eval

    def stub(state, sims, seed):
        # Main-process rerun: just-below threshold.
        return (0.449, 0.5)
    monkeypatch.setattr(probe_eval, "_default_mcts_labeler", stub)

    results = [_make_fake_result(status="admitted", mean_root_value=0.451)]
    counters = _run_borderline_reruns(
        results, epsilon=0.01,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15,
        label_ckpt_name="fake.safetensors", sims=10, repeats=2,
    )

    assert counters["candidates"] == 1
    assert counters["reruns"] == 1
    assert counters["flips"] == 1
    assert counters["seconds"] >= 0
    r = results[0]
    assert r["status"] == "rejected"
    assert r["audit_row"] is not None
    assert r["audit_row"]["borderline_rerun"] is True
    assert r["audit_row"]["borderline_rerun_flipped"] is True
    assert r["audit_row"]["parallel_admission_reason"] == "admitted"
    assert "magnitude" in r["audit_row"]["borderline_rerun_reason"]


def test_borderline_rerun_rejected_to_admitted(monkeypatch):
    """Symmetric flip: parallel rejected at just-below; main rerun just-above."""
    from scripts.build_probe_suite import _run_borderline_reruns
    from scripts.GPU.alphazero import probe_eval

    def stub(state, sims, seed):
        return (0.451, 0.5)
    monkeypatch.setattr(probe_eval, "_default_mcts_labeler", stub)

    results = [_make_fake_result(
        status="rejected", mean_root_value=0.449,
        rejection_reason="magnitude_below_threshold",
    )]
    counters = _run_borderline_reruns(
        results, epsilon=0.01,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15,
        label_ckpt_name="fake.safetensors", sims=10, repeats=2,
    )

    assert counters["flips"] == 1
    r = results[0]
    assert r["status"] == "admitted"
    assert r["audit_row"] is None
    # Rerun audit metadata lives on the candidate object so downstream
    # selector audit rows can merge it.
    rerun_meta = r["candidate"].get("_borderline_rerun_audit")
    assert rerun_meta is not None
    assert rerun_meta["borderline_rerun_flipped"] is True
    assert rerun_meta["parallel_admission_reason"] == "magnitude_below_threshold"
    assert rerun_meta["serial_rerun_admission_reason"] == "admitted"


def test_borderline_rerun_not_triggered_for_non_borderline(monkeypatch):
    """Candidates clearly admitted (mean=0.9) or clearly rejected (mean=0.1)
    are far from threshold and never trigger a rerun."""
    from scripts.build_probe_suite import _run_borderline_reruns
    from scripts.GPU.alphazero import probe_eval

    def stub(state, sims, seed):
        raise AssertionError("stub should not be called for non-borderline cands")
    monkeypatch.setattr(probe_eval, "_default_mcts_labeler", stub)

    results = [
        _make_fake_result(status="admitted", mean_root_value=0.9,
                          source_ply=18),
        _make_fake_result(
            status="rejected", mean_root_value=0.1, source_ply=22,
            rejection_reason="magnitude_below_threshold",
        ),
    ]
    counters = _run_borderline_reruns(
        results, epsilon=0.01,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15,
        label_ckpt_name="fake.safetensors", sims=10, repeats=2,
    )

    assert counters["candidates"] == 0
    assert counters["reruns"] == 0
    assert counters["flips"] == 0
    # Statuses unchanged.
    assert results[0]["status"] == "admitted"
    assert results[1]["status"] == "rejected"


def test_borderline_rerun_preserves_probe_id(monkeypatch):
    """Spec §9 invariant: rerun must produce a result with the same probe_id
    as the parallel result. The helper recomputes probe_id from
    source_game/source_ply, so any rerun-side mutation that changes those
    fields would surface here."""
    from scripts.build_probe_suite import _run_borderline_reruns
    from scripts.GPU.alphazero import probe_eval

    def stub(state, sims, seed):
        return (0.5, 0.5)
    monkeypatch.setattr(probe_eval, "_default_mcts_labeler", stub)

    results = [_make_fake_result(
        status="admitted", mean_root_value=0.451,
    )]
    pre_id = results[0]["probe_id"]
    _run_borderline_reruns(
        results, epsilon=0.01,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15,
        label_ckpt_name="fake.safetensors", sims=10, repeats=2,
    )
    assert results[0]["probe_id"] == pre_id


def test_borderline_rerun_skips_error_results(monkeypatch):
    """replay_error and mcts_error results have phase2_label=None and must
    be excluded from the borderline check (no IndexError, no rerun)."""
    from scripts.build_probe_suite import _run_borderline_reruns
    from scripts.GPU.alphazero import probe_eval

    def stub(state, sims, seed):
        raise AssertionError("stub should not be called for error rows")
    monkeypatch.setattr(probe_eval, "_default_mcts_labeler", stub)

    results = [
        {
            "probe_id": "p_replay_err",
            "status": "replay_error",
            "candidate": None,
            "audit_row": {"reason": "replay_error"},
            "rejection_reason": "replay_error",
            "phase2_label": None,
            "error_message": "ValueError: bad",
        },
        {
            "probe_id": "p_mcts_err",
            "status": "mcts_error",
            "candidate": {"source_game": "g0", "source_ply": 1},
            "audit_row": {"reason": "mcts_error"},
            "rejection_reason": "mcts_error",
            "phase2_label": None,
            "error_message": "RuntimeError: kaboom",
        },
    ]
    counters = _run_borderline_reruns(
        results, epsilon=0.01,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15,
        label_ckpt_name="fake.safetensors", sims=10, repeats=2,
    )
    assert counters["candidates"] == 0
    assert counters["reruns"] == 0
    assert counters["flips"] == 0
```

- [ ] **Step 2: Add the disable-path integration tests**

These tests run the full `_run_strong_advantage` and assert the rerun pass was a no-op for serial mode, `--no-borderline-rerun`, and `epsilon=0`. They reuse the fixture helper from Task 4.

Append to `tests/test_probe_phase2_parallel.py`:

```python
def _run_run_strong_advantage_with_stub(
    tmp_path, monkeypatch, *, mode, epsilon, no_rerun,
    stub=None,
):
    """Helper: run _run_strong_advantage with the tiny game fixture and a
    deterministic stub labeler. Returns the parsed draft JSON."""
    from scripts.GPU.alphazero import probe_eval
    from scripts.build_probe_suite import _build_arg_parser, _run_strong_advantage
    from tests.test_strong_advantage_probe_suite import (
        _write_minimal_strong_advantage_game_fixture,
    )
    import json as _json

    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_minimal_strong_advantage_game_fixture(games_dir)

    if stub is None:
        def stub(state, sims, seed):
            return (0.6, 0.4)
    monkeypatch.setattr(probe_eval, "_default_mcts_labeler", stub)
    monkeypatch.setattr(probe_eval, "load_network_for_scoring",
                        lambda p: (object(), 30, 128, 6))
    monkeypatch.setattr(probe_eval, "_set_default_labeler_network",
                        lambda *a, **kw: None)
    monkeypatch.setattr(probe_eval, "_set_default_labeler_mcts_config",
                        lambda *a, **kw: None)

    target = tmp_path / "out.json"
    cli = [
        "--tier", "strong_advantage",
        "--input", str(games_dir),
        "--source-iter-range", "0", "0",
        "--label-checkpoint", "fake.safetensors",
        "--label-mcts-sims", "10",
        "--label-mcts-repeats", "2",
        "--out", str(target),
        "--label-worker-mode", mode,
        "--admission-borderline-epsilon", str(epsilon),
        "--force",
    ]
    if mode == "process":
        cli.extend(["--label-workers", "2"])
    if no_rerun:
        cli.append("--no-borderline-rerun")
    args = _build_arg_parser().parse_args(cli)
    rc = _run_strong_advantage(args)
    assert rc == 0
    return _json.loads((tmp_path / "out.draft.json").read_text())


def test_borderline_rerun_disabled(tmp_path, monkeypatch):
    """--no-borderline-rerun disables the pass even when epsilon > 0 and
    mode=process. borderline_reruns counter is 0."""
    # Stub returns just-above threshold (would be borderline if rerun ran).
    def stub(state, sims, seed):
        return (0.451, 0.5)

    draft = _run_run_strong_advantage_with_stub(
        tmp_path, monkeypatch, mode="process", epsilon=0.01,
        no_rerun=True, stub=stub,
    )
    stats = draft["meta"]["phase2_run_stats"]
    assert stats["borderline_rerun_enabled"] is False
    assert stats["borderline_reruns"] == 0
    assert stats["borderline_flips"] == 0


def test_borderline_rerun_serial_mode_no_op(tmp_path, monkeypatch):
    """Serial mode never runs the borderline pass even with epsilon > 0."""
    def stub(state, sims, seed):
        return (0.451, 0.5)

    draft = _run_run_strong_advantage_with_stub(
        tmp_path, monkeypatch, mode="serial", epsilon=0.01,
        no_rerun=False, stub=stub,
    )
    stats = draft["meta"]["phase2_run_stats"]
    assert stats["borderline_rerun_enabled"] is False
    assert stats["borderline_reruns"] == 0


def test_admission_borderline_epsilon_zero_disables(tmp_path, monkeypatch):
    """Epsilon=0 disables the rerun pass even in process mode."""
    def stub(state, sims, seed):
        return (0.451, 0.5)

    draft = _run_run_strong_advantage_with_stub(
        tmp_path, monkeypatch, mode="process", epsilon=0.0,
        no_rerun=False, stub=stub,
    )
    stats = draft["meta"]["phase2_run_stats"]
    assert stats["borderline_rerun_enabled"] is False
    assert stats["borderline_reruns"] == 0


def test_committed_probes_have_no_private_rerun_keys(tmp_path, monkeypatch):
    """Spec §9: the committed probe JSON must contain no `_borderline_rerun_audit`
    or `parallel_phase2_label_before_rerun` keys, even when reruns happened."""
    # Hard to force a real flip end-to-end with cross-process control, so
    # synthesize the post-rerun condition: monkeypatch _run_borderline_reruns
    # to attach the private key to every admitted candidate, then assert
    # the serializer strips it.
    from scripts.build_probe_suite import _run_borderline_reruns as real_rerun
    import scripts.build_probe_suite as bps

    def fake_rerun(results, **kwargs):
        for r in results:
            if r["candidate"] is not None:
                r["candidate"]["_borderline_rerun_audit"] = {
                    "borderline_rerun": True,
                    "borderline_rerun_reason": ["magnitude"],
                    "parallel_phase2_label_before_rerun": {"sentinel": True},
                    "borderline_rerun_flipped": False,
                }
        return {"candidates": len(results), "reruns": len(results),
                "flips": 0, "seconds": 0.0}

    monkeypatch.setattr(bps, "_run_borderline_reruns", fake_rerun)

    draft = _run_run_strong_advantage_with_stub(
        tmp_path, monkeypatch, mode="process", epsilon=0.01,
        no_rerun=False,
    )
    serialized = (tmp_path / "out.draft.json").read_text()
    assert "_borderline_rerun_audit" not in serialized
    assert "parallel_phase2_label_before_rerun" not in serialized
    # Sanity: the test actually exercised reruns.
    assert draft["meta"]["phase2_run_stats"]["borderline_reruns"] > 0
```

- [ ] **Step 3: Run the tests; verify they fail**

Run: `.venv/bin/python -m pytest tests/test_probe_phase2_parallel.py -k "borderline or private_rerun_keys" -v`
Expected: FAIL with `ImportError: cannot import name '_run_borderline_reruns'`.

- [ ] **Step 4: Add `_is_borderline`, `_run_borderline_reruns`, and the private-key strip**

In `scripts/build_probe_suite.py`, add near `_label_one_strong_advantage_candidate`:

```python
_BORDERLINE_TRIGGERS = ("magnitude", "top1_share", "stability")


def _is_borderline(
    label: dict,
    *,
    epsilon: float,
    magnitude_threshold: float,
    top1_share_floor: float,
    stability_cap: float,
) -> list[str]:
    """Return the list of triggers (subset of _BORDERLINE_TRIGGERS) for which
    the label is within epsilon of the corresponding admission threshold.
    Empty list means not borderline.
    """
    triggers = []
    if abs(abs(label["mean_root_value"]) - magnitude_threshold) <= epsilon:
        triggers.append("magnitude")
    if abs(label["min_top1_share"] - top1_share_floor) <= epsilon:
        triggers.append("top1_share")
    if abs(label["value_stability"] - stability_cap) <= epsilon:
        triggers.append("stability")
    return triggers


def _run_borderline_reruns(
    results: list[dict],
    *,
    epsilon: float,
    magnitude_threshold: float,
    top1_share_floor: float,
    stability_cap: float,
    label_ckpt_name: str,
    sims: int,
    repeats: int,
) -> dict:
    """Re-label borderline candidates synchronously in the main process.
    Mutates `results` in place. Returns counters for instrumentation:
        {"candidates": N, "reruns": N, "flips": N, "seconds": float}
    """
    import time as _time
    counters = {"candidates": 0, "reruns": 0, "flips": 0, "seconds": 0.0}
    t0 = _time.time()
    for r in results:
        if r["phase2_label"] is None:
            continue  # replay/mcts errors carry no label
        triggers = _is_borderline(
            r["phase2_label"],
            epsilon=epsilon,
            magnitude_threshold=magnitude_threshold,
            top1_share_floor=top1_share_floor,
            stability_cap=stability_cap,
        )
        if not triggers:
            continue
        counters["candidates"] += 1
        # Re-execute the helper in the main process. Same seed/sims/cfg.
        # Use the candidate object that came back from the worker (already
        # deepcopy'd inside the helper, but we deepcopy again to keep the
        # pre-rerun label intact for audit metadata).
        cand = copy.deepcopy(r["candidate"])
        # Strip the prior phase2_label so the helper re-labels cleanly.
        cand.pop("phase2_label", None)
        rerun = _label_one_strong_advantage_candidate(
            cand,
            label_ckpt_name=label_ckpt_name,
            sims=sims,
            repeats=repeats,
            magnitude_threshold=magnitude_threshold,
            top1_share_floor=top1_share_floor,
            stability_cap=stability_cap,
        )
        # Spec §9: rerun must preserve probe identity. The helper
        # recomputes seed_base from sha256(probe_id) and builds the same
        # probe_id internally, so this should always hold; assert
        # explicitly to catch future regressions (e.g. accidental copy
        # mutation that changes source_game/source_ply/move_history).
        assert rerun["probe_id"] == r["probe_id"], (
            f"borderline rerun changed probe_id: {r['probe_id']} -> "
            f"{rerun['probe_id']}"
        )
        counters["reruns"] += 1
        flipped = rerun["status"] != r["status"]
        if flipped:
            counters["flips"] += 1
            print(
                f"[probe_suite] borderline rerun flipped {r['probe_id']}: "
                f"{r['rejection_reason'] or 'admitted'} -> "
                f"{rerun['rejection_reason'] or 'admitted'}",
                file=sys.stderr,
            )

        rerun_audit_meta = {
            "borderline_rerun": True,
            "borderline_rerun_reason": triggers,
            "parallel_phase2_label_before_rerun": r["phase2_label"],
            "borderline_rerun_flipped": flipped,
        }
        if flipped:
            rerun_audit_meta["parallel_admission_reason"] = (
                r["rejection_reason"] or "admitted"
            )
            rerun_audit_meta["serial_rerun_admission_reason"] = (
                rerun["rejection_reason"] or "admitted"
            )

        # Attach to the candidate so any audit row built downstream
        # (Phase 2 rejection or selector audit) merges these fields.
        if rerun["candidate"] is not None:
            rerun["candidate"]["_borderline_rerun_audit"] = rerun_audit_meta

        # Replace the parallel result with the rerun result. Audit row, if
        # any, also takes the rerun's content with the rerun metadata merged.
        if rerun["audit_row"] is not None:
            rerun["audit_row"].update(rerun_audit_meta)
        # Update the result entry in-place.
        r.update(rerun)

    counters["seconds"] = _time.time() - t0
    return counters
```

Now, in `_run_strong_advantage`, insert the rerun pass BETWEEN building `results` (end of either serial or process branch) and the existing call to `_phase2_aggregate(results)` from Task 4. Replace the prior `admitted, new_audit_rows = _phase2_aggregate(results)` with:

```python
    rerun_enabled = (
        args.label_worker_mode == "process"
        and args.admission_borderline_epsilon > 0
        and not args.no_borderline_rerun
    )
    if rerun_enabled:
        rerun_counters = _run_borderline_reruns(
            results,
            epsilon=args.admission_borderline_epsilon,
            magnitude_threshold=args.magnitude_threshold,
            top1_share_floor=args.top1_share_floor,
            stability_cap=args.stability_cap,
            label_ckpt_name=label_ckpt.name,
            sims=args.label_mcts_sims,
            repeats=args.label_mcts_repeats,
        )
    else:
        rerun_counters = {"candidates": 0, "reruns": 0, "flips": 0, "seconds": 0.0}

    # _phase2_aggregate sorts AFTER reruns so post-rerun statuses drive
    # admitted/audit partition. Spec §6.
    admitted, new_audit_rows = _phase2_aggregate(results)
    audit.extend(new_audit_rows)
```

`rerun_counters` is also consumed by Task 6 (instrumentation). For serial mode the pass is a no-op by design (zero-filled counters).

Finally, ensure private rerun-audit keys never reach the committed suite. In the `probes_out` construction loop (around line ~620 in `_run_strong_advantage`), add a strip step:

```python
    probes_out = []
    for cand in admitted:
        probes_out.append({
            "id": _probe_id_for(cand),
            "category": cand["category"],
            "confidence": "strong_advantage",
            "side_to_move": _stm_at_ply(cand),
            "expected_value_sign": 1 if cand["winner"] == "red" else -1,
            "active_size": 24,
            "ply": cand["ply"],
            "move_history": cand["move_history"],
            "source_game": cand["source_game"],
            "source_ply": cand["source_ply"],
            "starting_player": cand["starting_player"],
            "phase1_features": cand["phase1_features"],
            "phase2_label": cand["phase2_label"],
            # No private _ keys — _borderline_rerun_audit is intentionally
            # omitted; it lives in the audit file only.
        })
```

The dict literal explicitly enumerates which fields land in the suite; private `_`-prefixed keys are never copied. Then, when the diversity selector emits its admitted-candidate audit rows, ensure those rows merge in the `_borderline_rerun_audit` dict from the candidate object. Look at where the selector currently writes its `reason="admitted"` rows (search for `_select_diverse_admitted_candidates`) and add a `_merge_borderline_audit(audit_row, cand)` call.

`_merge_borderline_audit` is a tiny helper:

```python
def _merge_borderline_audit(audit_row: dict, cand: dict) -> dict:
    """If cand has _borderline_rerun_audit, merge those keys into audit_row.
    Returns the (potentially mutated) audit_row."""
    rerun_meta = cand.get("_borderline_rerun_audit")
    if rerun_meta:
        audit_row.update(rerun_meta)
    return audit_row
```

Call it everywhere an audit row references a candidate that may have been reran (selector admitted rows; selector diversity-drop rows). The Phase-2-rejection audit rows already received the merge via `_run_borderline_reruns`'s in-place update, so no further wiring needed there.

- [ ] **Step 5: Run all borderline-rerun tests**

Run: `.venv/bin/python -m pytest tests/test_probe_phase2_parallel.py -k "borderline or private_rerun_keys" -v`
Expected: all pass.

- [ ] **Step 6: Run full file + regression suite**

Run: `.venv/bin/python -m pytest tests/test_probe_phase2_parallel.py tests/test_strong_advantage_probe_suite.py tests/test_strong_advantage_diversity_selector.py -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/build_probe_suite.py tests/test_probe_phase2_parallel.py
git commit -m "feat(probes): borderline serial-rerun for threshold-sensitive candidates

After the process-pool Phase 2 returns, candidates whose label is within
--admission-borderline-epsilon of any admission threshold are re-labeled
synchronously in the main process (single MLX context). Rerun result is
authoritative; admission filter is re-applied once. Audit metadata
records flips with old/new reasons. Committed probe JSON has no private
_borderline_rerun_audit keys (audit file only).

Disabled in serial mode (no contention to correct) and via
--no-borderline-rerun.

Spec: docs/superpowers/specs/2026-04-28-probe-phase2-parallel-labeling-design.md §9"
```

---

## Task 6: Add Phase 2 instrumentation (summary line + meta.phase2_run_stats)

**Files:**
- Modify: `scripts/build_probe_suite.py` (track counters, emit summary line, write `meta.phase2_run_stats`)
- Test: `tests/test_probe_phase2_parallel.py` (add `test_phase2_run_stats_in_meta`)

Spec §10.

- [ ] **Step 1: Add the instrumentation test**

Reuses `_run_run_strong_advantage_with_stub` (from Task 5, Step 2) and `_patch_phase1_extract` + `_SAMPLE_CENTRAL` (from Task 4, Step 1). Append to `tests/test_probe_phase2_parallel.py`:

```python
def test_phase2_run_stats_recorded_in_meta(tmp_path, monkeypatch):
    """meta.phase2_run_stats records all the documented fields, including
    workers_requested vs workers_effective and borderline_rerun_enabled.
    Use --label-worker-mode=serial --label-workers=4 to exercise the
    serial-mode clamp path."""
    import json as _json
    import scripts.build_probe_suite as bps

    _patch_phase1_extract(monkeypatch, [_SAMPLE_CENTRAL])

    def stub(state, sims, seed):
        return (0.7, 0.4)
    from scripts.GPU.alphazero import probe_eval
    monkeypatch.setattr(probe_eval, "_default_mcts_labeler", stub)
    monkeypatch.setattr(probe_eval, "load_network_for_scoring",
                        lambda p: (object(), 30, 128, 6))
    monkeypatch.setattr(probe_eval, "_set_default_labeler_network",
                        lambda *a, **kw: None)
    monkeypatch.setattr(probe_eval, "_set_default_labeler_mcts_config",
                        lambda *a, **kw: None)

    fake_ckpt = tmp_path / "fake_ckpt.safetensors"
    fake_ckpt.write_bytes(b"stub")
    out_path = tmp_path / "out.json"

    rc = bps.main_with_args([
        "--tier", "strong_advantage",
        "--input", "scripts/GPU/logs/games",
        "--source-iter-range", "70", "70",
        "--label-checkpoint", str(fake_ckpt),
        "--label-mcts-sims", "10",
        "--label-mcts-repeats", "1",
        "--magnitude-threshold", "0.45",
        "--out", str(out_path),
        "--label-worker-mode", "serial",
        "--label-workers", "4",   # under serial, should clamp to 1 with warning
        "--force",
    ])
    assert rc == 0

    draft = _json.loads(out_path.with_suffix(".draft.json").read_text())
    stats = draft["meta"]["phase2_run_stats"]
    assert stats["mode"] == "serial"
    assert stats["workers_requested"] == 4
    assert stats["workers_effective"] == 1
    assert stats["eval_batch_size"] == 14
    assert stats["stall_flush_sims"] == 16
    assert stats["candidates_total"] == 1
    assert stats["labeled"] == 1
    assert stats["admitted_before_diversity"] == 1
    assert stats["rejected"] == 0
    assert stats["replay_errors"] == 0
    assert stats["mcts_errors"] == 0
    assert stats["borderline_rerun_enabled"] is False  # serial mode
    assert stats["admission_borderline_epsilon"] == 0.01
    assert stats["borderline_candidates"] == 0
    assert stats["borderline_reruns"] == 0
    assert stats["borderline_flips"] == 0
    assert "rejection_reasons" in stats
    assert isinstance(stats["seconds_total"], (int, float))
    assert stats["seconds_total"] >= 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_probe_phase2_parallel.py::test_phase2_run_stats_recorded_in_meta -v`
Expected: FAIL.

- [ ] **Step 3: Track counters + emit summary line + write `meta.phase2_run_stats`**

Capture the user-requested workers value BEFORE the serial-mode clamp in `main()`. In `main()` (added in Task 2), modify the warning-and-clamp branch:

```python
    args.label_workers_requested = args.label_workers
    if args.label_worker_mode == "serial" and args.label_workers != 1:
        print(...)
        args.label_workers = 1
```

Then in `_run_strong_advantage`, after Phase 2 completes (post-rerun, pre-selector), build the summary block:

```python
    phase2_elapsed = _time.time() - t_phase2_start
    n_replay_errors = sum(1 for r in results if r["status"] == "replay_error")
    n_mcts_errors = sum(1 for r in results if r["status"] == "mcts_error")
    n_admitted_pre_diversity = sum(1 for r in results if r["status"] == "admitted")
    n_rejected = sum(1 for r in results if r["status"] == "rejected")
    n_labeled = n_admitted_pre_diversity + n_rejected
    rejection_reasons = Counter(
        r["rejection_reason"] for r in results
        if r["status"] == "rejected" and r["rejection_reason"]
    )

    args._phase2_run_stats = {
        "mode": args.label_worker_mode,
        "workers_requested": args.label_workers_requested,
        "workers_effective": args.label_workers,
        "eval_batch_size": args.mcts_eval_batch_size,
        "stall_flush_sims": args.mcts_stall_flush_sims,
        "candidates_total": n_total,
        "labeled": n_labeled,
        "replay_errors": n_replay_errors,
        "mcts_errors": n_mcts_errors,
        "admitted_before_diversity": n_admitted_pre_diversity,
        "rejected": n_rejected,
        "borderline_rerun_enabled": rerun_enabled,
        "admission_borderline_epsilon": args.admission_borderline_epsilon,
        "borderline_candidates": rerun_counters["candidates"],
        "borderline_reruns": rerun_counters["reruns"],
        "borderline_flips": rerun_counters["flips"],
        "borderline_rerun_seconds": round(rerun_counters["seconds"], 2),
        "seconds_total": round(phase2_elapsed, 2),
        "rejection_reasons": dict(rejection_reasons),
    }
```

Print the summary line:

```python
    breakdown_str = ", ".join(f"{r}={n}" for r, n in rejection_reasons.most_common())
    print(
        f"[probe_suite] Phase 2 complete: {n_labeled}/{n_total} labeled "
        f"({n_admitted_pre_diversity} admitted, {phase2_elapsed:.0f}s total)\n"
        f"  mode={args.label_worker_mode} "
        f"workers_requested={args.label_workers_requested} "
        f"workers_effective={args.label_workers} "
        f"eval_batch={args.mcts_eval_batch_size} "
        f"stall_flush={args.mcts_stall_flush_sims}\n"
        f"  candidates_total={n_total} labeled={n_labeled} "
        f"replay_errors={n_replay_errors} mcts_errors={n_mcts_errors}\n"
        f"  admitted_before_diversity={n_admitted_pre_diversity} "
        f"rejected={n_rejected}\n"
        f"  borderline_candidates={rerun_counters['candidates']} "
        f"borderline_reruns={rerun_counters['reruns']} "
        f"borderline_flips={rerun_counters['flips']}\n"
        f"  borderline_rerun_seconds={rerun_counters['seconds']:.2f}\n"
        f"  Per-reason: {breakdown_str}",
        flush=True,
    )
```

Replace the existing "Phase 2 complete" print at line ~593 with the block above.

When constructing the draft payload (`payload = {...}` around line ~638), add the `phase2_run_stats` field as a **sibling** of `selection_rules` (not nested):

```python
    payload = {
        "meta": {
            "type": "bootstrap_rule_selected",
            "tier": "strong_advantage",
            "not_gate_suite": True,
            "review_mode": "draft",
            "reviewer": None,
            "reviewed_at_utc": None,
            "generator": "scripts/build_probe_suite.py",
            "generator_version": 1,
            "selection_rules": { ... },          # unchanged
            "phase2_run_stats": args._phase2_run_stats,
        },
        "probes": probes_out,
    }
```

- [ ] **Step 4: Run the new test and verify it passes**

Run: `.venv/bin/python -m pytest tests/test_probe_phase2_parallel.py::test_phase2_run_stats_recorded_in_meta -v`
Expected: PASS.

- [ ] **Step 5: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/test_probe_phase2_parallel.py tests/test_strong_advantage_probe_suite.py tests/test_strong_advantage_diversity_selector.py -v`
Expected: all pass.

> **Compatibility note:** the existing test `test_committed_meta_block_well_formed` in `tests/test_strong_advantage_probe_suite.py` reads the committed `tests/probes/strong_advantage_probes.json`. That file was generated before this PR and has no `phase2_run_stats` field. The test must continue to allow this — `phase2_run_stats` is REQUIRED in newly-generated drafts but OPTIONAL in already-committed suites. If the existing test enforces a closed schema, relax it to allow optional `phase2_run_stats` in this step.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_probe_suite.py tests/test_probe_phase2_parallel.py tests/test_strong_advantage_probe_suite.py
git commit -m "feat(probes): record Phase 2 run stats in meta.phase2_run_stats

New meta.phase2_run_stats block records mode, workers_requested vs
workers_effective, MCTSConfig values, candidate counters
(replay/mcts errors, admitted/rejected pre-diversity), borderline
rerun counters (candidates/reruns/flips/seconds), epsilon, and
per-reason rejection breakdown. Phase 2 summary print line mirrors the
JSON. Sibling of selection_rules so deterministic config and runtime
stats stay separate.

Spec: docs/superpowers/specs/2026-04-28-probe-phase2-parallel-labeling-design.md §10"
```

---

## Task 7: Generate the tiny golden fixture and add the byte-identity regression test

**Files:**
- Create: `tests/probes/golden/phase2_serial_tiny_input.json` (frozen input)
- Create: `tests/probes/golden/phase2_serial_tiny.json` (expected output)
- Test: `tests/test_probe_phase2_parallel.py` (add `test_phase2_serial_unchanged`)

Spec §12 (`test_phase2_serial_unchanged`).

- [ ] **Step 1: Decide the fixture shape**

The fixture is a tiny input directory (1 game, 3-5 candidates) plus the corresponding `out.draft.json` and `candidates_strong_advantage.json` outputs generated by the current code with a deterministic mocked labeler. The mocked labeler is the same `stub_labeler` pattern used in earlier tests: deterministic value-as-a-function-of-seed.

- [ ] **Step 2: Add a fixture-generator helper script (one-off)**

Create `scripts/probes/generate_golden_phase2_serial_fixture.py` (small, deliberately not in CI):

```python
"""One-off generator for tests/probes/golden/phase2_serial_tiny.json.

Run once after each intentional change to the serial Phase 2 output schema.
Commit both the input and the output fixture.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "tests" / "probes" / "golden"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a tiny input directory of 1 game with a known move history
    # that yields multiple strong-advantage candidates.
    sys.path.insert(0, str(repo_root))
    from tests.test_strong_advantage_probe_suite import (
        _write_minimal_strong_advantage_game_fixture,
    )
    games_dir = out_dir / "phase2_serial_tiny_input"
    if games_dir.exists():
        shutil.rmtree(games_dir)
    games_dir.mkdir()
    _write_minimal_strong_advantage_game_fixture(games_dir)

    # Run with a stub labeler.
    from scripts.GPU.alphazero import probe_eval
    from scripts.build_probe_suite import _build_arg_parser, _run_strong_advantage

    def stub(state, sims, seed):
        v = 0.6 + 0.001 * (seed % 7)
        t1 = 0.4 + 0.001 * (seed % 5)
        return (v, t1)

    with patch.object(probe_eval, "_default_mcts_labeler", stub), \
         patch.object(probe_eval, "load_network_for_scoring",
                      return_value=(object(), 30, 128, 6)), \
         patch.object(probe_eval, "_set_default_labeler_network", return_value=None), \
         patch.object(probe_eval, "_set_default_labeler_mcts_config", return_value=None):
        target = out_dir / "phase2_serial_tiny.json"
        ap = _build_arg_parser()
        args = ap.parse_args([
            "--tier", "strong_advantage",
            "--input", str(games_dir),
            "--source-iter-range", "0", "0",
            "--label-checkpoint", "fake.safetensors",
            "--label-mcts-sims", "10",
            "--label-mcts-repeats", "2",
            "--out", str(target),
            "--force",
        ])
        rc = _run_strong_advantage(args)
        assert rc == 0

    # Move the draft into the canonical fixture name.
    (out_dir / "phase2_serial_tiny.draft.json").rename(
        out_dir / "phase2_serial_tiny_expected.json"
    )
    print(f"Golden fixture written to {out_dir / 'phase2_serial_tiny_expected.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Generate the fixture**

Run: `.venv/bin/python scripts/probes/generate_golden_phase2_serial_fixture.py`
Expected: a single line of output ending in `Golden fixture written to .../phase2_serial_tiny_expected.json`.

- [ ] **Step 4: Add the regression test**

Append to `tests/test_probe_phase2_parallel.py`:

```python
def test_phase2_serial_unchanged(tmp_path, monkeypatch):
    """Running with default (serial) flags on the frozen tiny input produces
    output byte-identical to the committed golden fixture. Guards against
    accidental drift in serial-mode behavior."""
    from scripts.GPU.alphazero import probe_eval
    from scripts.build_probe_suite import _build_arg_parser, _run_strong_advantage
    import shutil
    import json as _json

    repo_root = Path(__file__).resolve().parent.parent
    golden_dir = repo_root / "tests" / "probes" / "golden"
    golden_input = golden_dir / "phase2_serial_tiny_input"
    expected_path = golden_dir / "phase2_serial_tiny_expected.json"

    games_dir = tmp_path / "games"
    shutil.copytree(golden_input, games_dir)

    def stub(state, sims, seed):
        v = 0.6 + 0.001 * (seed % 7)
        t1 = 0.4 + 0.001 * (seed % 5)
        return (v, t1)

    monkeypatch.setattr(probe_eval, "_default_mcts_labeler", stub)
    monkeypatch.setattr(probe_eval, "load_network_for_scoring",
                        lambda p: (object(), 30, 128, 6))
    monkeypatch.setattr(probe_eval, "_set_default_labeler_network",
                        lambda *a, **kw: None)
    monkeypatch.setattr(probe_eval, "_set_default_labeler_mcts_config",
                        lambda *a, **kw: None)

    target = tmp_path / "out.json"
    ap = _build_arg_parser()
    args = ap.parse_args([
        "--tier", "strong_advantage",
        "--input", str(games_dir),
        "--source-iter-range", "0", "0",
        "--label-checkpoint", "fake.safetensors",
        "--label-mcts-sims", "10",
        "--label-mcts-repeats", "2",
        "--out", str(target),
        "--force",
    ])
    rc = _run_strong_advantage(args)
    assert rc == 0

    actual = (tmp_path / "out.draft.json").read_text()
    expected = expected_path.read_text()

    # phase2_run_stats includes seconds_total / borderline_rerun_seconds which
    # are timing-dependent. Strip them before compare.
    def _strip_timing(blob):
        d = _json.loads(blob)
        stats = d.get("meta", {}).get("phase2_run_stats", {})
        stats.pop("seconds_total", None)
        stats.pop("borderline_rerun_seconds", None)
        return _json.dumps(d, indent=2, sort_keys=True)

    assert _strip_timing(actual) == _strip_timing(expected)
```

- [ ] **Step 5: Run the test and verify it passes**

Run: `.venv/bin/python -m pytest tests/test_probe_phase2_parallel.py::test_phase2_serial_unchanged -v`
Expected: PASS.

- [ ] **Step 6: Run full test suite**

Run: `.venv/bin/python -m pytest tests/test_probe_phase2_parallel.py tests/test_strong_advantage_probe_suite.py tests/test_strong_advantage_diversity_selector.py -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add tests/probes/golden/ scripts/probes/generate_golden_phase2_serial_fixture.py tests/test_probe_phase2_parallel.py
git commit -m "test(probes): tiny golden fixture for serial-mode byte-identity

Adds tests/probes/golden/phase2_serial_tiny_{input,expected}.json plus
a one-off generator. test_phase2_serial_unchanged asserts default flags
produce byte-identical output (modulo timing fields). Guards against
accidental drift in the serial Phase 2 path going forward."
```

---

## Task 8: Update user-facing documentation

**Files:**
- Modify: `docs/probe-suite-generation.md` (knob table at lines ~49–60; new Performance subsection after line ~132; example command around line ~100)
- Modify: `tests/probes/README.md` (one-line cross-reference)
- Modify: `scripts/build_probe_suite.py:1-15` (top docstring)

Spec §14.

- [ ] **Step 1: Update `docs/probe-suite-generation.md` knob table**

Add the seven new flag rows to the existing flag table at lines ~49–60. Insert them after `--max-probes-per-game` in this exact order:

```markdown
| `--label-worker-mode` | serial | Phase 2 execution mode. `serial` (default, byte-reference path) runs in the main process. `process` enables parallel labeling via a process pool with one MLX network per worker. |
| `--label-workers` | 1 | Worker count under `--label-worker-mode=process`. Ignored under `serial` (warning if explicitly set to ≠1). On Apple Silicon start with 2–4. |
| `--mcts-eval-batch-size` | 14 | NN batch size for the labeler's MCTS. Capped at 14 because larger batches have caused Metal hangs; pass `--allow-unsafe-eval-batch` to exceed. |
| `--mcts-stall-flush-sims` | 16 | MCTS stall-flush threshold (see `MCTSConfig`). 0 disables. |
| `--allow-unsafe-eval-batch` | flag | Required to set `--mcts-eval-batch-size > 14`. Intended only for local benchmarking. |
| `--admission-borderline-epsilon` | 0.01 | In process mode, candidates whose phase-2 label is within ε of any admission threshold are re-labeled in the main process to use the serial reference label. 0 disables. |
| `--no-borderline-rerun` | flag | Disable borderline rerun even when ε > 0. Used for benchmarking the raw process-pool path. |
```

- [ ] **Step 2: Add the new Performance subsection**

Find the existing `## Performance` section (around line ~125–132). Add a new subsection AFTER the existing content there:

```markdown
### Parallel labeling

`--label-worker-mode process` runs Phase 2 candidate labeling in a `ProcessPoolExecutor` with one MLX network per worker. Default remains `serial`; parallel mode is fully opt-in.

Recommended starting point on Apple Silicon:

```
--label-worker-mode process --label-workers 2
```

If stable and faster, try `--label-workers 4`. Avoid increasing `--mcts-eval-batch-size` above 14 unless intentionally benchmarking with `--allow-unsafe-eval-batch`. Higher worker counts are not always faster: each process loads its own MLX network and can contend for the Metal scheduler.

`--label-worker-mode process --label-workers 1` is valid (useful for testing worker initialization and deterministic reassembly), but it is not expected to speed up the run.

**Reproducibility under parallel mode.** Serial mode is the strict reference path for mocked/deterministic labelers and the supported strict reproducibility mode for generated artifacts. For real MLX runs, the supported target is identical admitted probe IDs, identical final committed probe IDs, and identical rejection reasons for non-borderline candidates under normal deterministic MLX behavior. Borderline rerun (`--admission-borderline-epsilon`, default 0.01) re-labels threshold-sensitive candidates in the main process so admission decisions match serial. Byte-identical numeric labels are not promised across machines, worker counts, or MLX versions for real runs.

`meta.phase2_run_stats` in the draft JSON records the mode, worker counts (requested and effective), MCTSConfig values, per-status counters, and borderline rerun counters for postmortem reproduction.
```

- [ ] **Step 3: Update the example command (around line ~100)**

Find the existing example block (around line ~96–107) and add a commented-out fast-mode line:

```bash
.venv/bin/python scripts/build_probe_suite.py \
    --tier strong_advantage \
    --source-iter-range 57 58 \
    --label-checkpoint checkpoints/alphazero-v2-staged/model_iter_0059.safetensors \
    --label-mcts-sims 2000 \
    --label-mcts-repeats 2 \
    --max-probes 30
    # Faster (opt-in): --label-worker-mode process --label-workers 4
```

- [ ] **Step 4: Update `tests/probes/README.md`**

Read `tests/probes/README.md`. If it documents the generation command (it likely does), add ONE line cross-referencing the new flags:

```markdown
For parallel-labeling flags (`--label-worker-mode`, `--label-workers`, etc.) see
[`docs/probe-suite-generation.md`](../../docs/probe-suite-generation.md#parallel-labeling).
```

Place it just below wherever the existing `scripts/build_probe_suite.py` invocation is described. Do not duplicate the full knob table.

- [ ] **Step 5: Update the script docstring**

Replace the top docstring of `scripts/build_probe_suite.py` (lines 1–15) with:

```python
"""Tier-parameterized probe suite generator.

Replaces scripts/build_bootstrap_probe_suite.py as the real implementation
(that script is kept as a thin --tier forced shim for muscle memory and
existing CI/cron commands).

Tiers:
  --tier forced            Bootstrap forced suite (existing behavior,
                           writes tests/probes/twixt_probes.json by default).
  --tier strong_advantage  Bootstrap strong-advantage suite (deep-MCTS
                           labeled, light-reviewed). Phases 1/2/3 per
                           docs/superpowers/specs/2026-04-28-...

Both tiers produce byte-identical output for identical inputs in serial
mode. The strong-advantage tier additionally supports an opt-in
process-pool labeling path (`--label-worker-mode process`) with safe
defaults preserving prior byte-identity. See docs/probe-suite-generation.md
for the full operator workflow including parallel-labeling flags.
"""
```

- [ ] **Step 6: Run `--help` manually and skim the docs**

Run: `.venv/bin/python scripts/build_probe_suite.py --tier strong_advantage --help | head -60`
Expected: clean help output, all new flags visible with one-line descriptions.

Open `docs/probe-suite-generation.md` and visually confirm the new rows appear in the knob table and the new Performance subsection reads naturally.

- [ ] **Step 7: Commit**

```bash
git add docs/probe-suite-generation.md tests/probes/README.md scripts/build_probe_suite.py
git commit -m "docs(probes): document Phase 2 parallel-labeling flags

Adds the seven new flags to the knob table in
docs/probe-suite-generation.md, plus a new Parallel labeling
subsection under Performance covering recommended starting values
on Apple Silicon, the byte-reference vs ID-equivalent contract,
and meta.phase2_run_stats. Updates the example command with a
fast-mode comment, adds a cross-reference from
tests/probes/README.md, and updates the script's top docstring.

Spec: docs/superpowers/specs/2026-04-28-probe-phase2-parallel-labeling-design.md §14"
```

---

## Task 9: Add manual real-MLX equivalence script

**Files:**
- Create: `scripts/probes/verify_parallel_equivalence.py`

Spec §12 (manual / out-of-CI).

- [ ] **Step 1: Create the script**

```python
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

    # Label-tolerance comparison on shared IDs.
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
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x scripts/probes/verify_parallel_equivalence.py
```

- [ ] **Step 3: Sanity-check the help text**

Run: `.venv/bin/python scripts/probes/verify_parallel_equivalence.py --help`
Expected: argparse usage output.

- [ ] **Step 4: Commit**

```bash
git add scripts/probes/verify_parallel_equivalence.py
git commit -m "test(probes): manual script to verify serial-vs-process equivalence on real MLX

Out-of-CI. Runs build_probe_suite.py twice (serial then process) on a
real checkpoint with a small --max-probes sample; reports admitted-ID
equivalence and per-field label tolerance. Recommended for one-time
validation after changes to MCTS, the labeler, or worker plumbing.

Spec: docs/superpowers/specs/2026-04-28-probe-phase2-parallel-labeling-design.md §12"
```

---

## Self-review

After all tasks land, sanity-check coverage against the spec:

| Spec section | Implemented in |
|---|---|
| §4 determinism contract | Task 4 (mocked equivalence), Task 5 (borderline rerun guarding decisions), Task 9 (real-MLX manual verification) |
| §5 CLI surface + validation | Task 2 |
| §6 architecture / data flow | Task 4 |
| §7 helper extraction + result dict | Task 3 |
| §8 MCTSConfig wiring + worker init | Task 1, Task 4 |
| §9 borderline rerun + private-key strip | Task 5 |
| §10 instrumentation + meta.phase2_run_stats | Task 6 |
| §11 lifecycle / error handling | Task 4 (init failure, replay/MCTS error isolation) |
| §12 test strategy | All tasks contribute tests; Task 7 adds golden fixture; Task 9 adds manual script |
| §13 future stages | Out of scope (correctly) |
| §14 docs | Task 8 |

If a follow-up reader spots a gap, file it as a small fast-follow task; do not re-open the spec for additions that are already implied (e.g. a missing helper docstring).
