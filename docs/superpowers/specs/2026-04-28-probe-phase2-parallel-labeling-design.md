# Probe Suite Phase 2 — Parallel Labeling

**Date:** 2026-04-28
**Author:** brainstormed with Bill
**Status:** Design approved, awaiting implementation plan
**Touches:** `scripts/build_probe_suite.py`, `scripts/GPU/alphazero/probe_eval.py`, `docs/probe-suite-generation.md`, `tests/probes/README.md`

## 1. Problem

Phase 2 of the strong-advantage probe suite generator is strictly serial. For each candidate it reconstructs a `TwixtState`, calls `label_candidate_with_mcts()`, applies the admission filter, and appends the result. With realistic settings (`--label-mcts-sims 2000+ --label-mcts-repeats 2+`) the loop dominates wallclock, often running for 10s of minutes to multiple hours per generation.

The bottleneck is per-candidate MCTS, which is a mix of Python tree work (GIL-bound) and small batched MLX/Metal forward passes. A process pool that runs N candidates concurrently — each with its own MLX network — should give close-to-linear speedup on Apple Silicon up to the point where worker count saturates the Metal scheduler.

## 2. Goals

1. Add an opt-in process-pool path to Phase 2 of the strong-advantage tier.
2. Preserve byte-identical output for users who do not opt in (default flags unchanged).
3. Preserve the deterministic admitted-probe-ID set under parallel mode by re-running candidates whose parallel-mode label lands within ε of any admission threshold serially in the main process.
4. Expose `MCTSConfig.eval_batch_size` and `MCTSConfig.stall_flush_sims` as user knobs, capped at the known-safe Metal batch size of 14 unless the operator explicitly opts in.
5. Update `docs/probe-suite-generation.md`, `tests/probes/README.md`, and the script docstring so the new knobs are documented in their canonical home.

## 3. Non-goals

- **Forced tier.** It has no Phase 2; this design does not touch it.
- **Cross-candidate batched MLX evaluator.** That's a future stage (see §13).
- **Thread-pool labeling.** MCTS tree work is GIL-bound; threads cannot deliver the speedup we want.
- **Byte-identical numeric labels under real MLX parallel runs.** See §4.

## 4. Determinism contract

Two layers of determinism, treated separately:

- **Structural determinism — strict.** Probe IDs, candidate ordering after sorting, RNG seed derivation, audit row contents (modulo float fields), category/source/ply, selector inputs, and final committed probe IDs are deterministic and process-independent.
- **Numeric determinism — bounded by float tolerance.** `phase2_label` numeric fields (`mean_root_value`, `value_per_run`, `value_stability`, `min_top1_share`) are byte-identical only under mocked-labeler tests. Under real MLX parallel execution they are reproducible within tolerance (`<= 1e-4`).

**Supported user contract:**

- Serial mode is the strict reference path for mocked/deterministic labelers and the supported strict reproducibility mode for generated artifacts.
- For real MLX runs, the supported target is identical admitted probe IDs, identical final committed probe IDs, and identical rejection reasons for non-borderline candidates under normal deterministic MLX behavior. Borderline rerun guards threshold-sensitive cases, but byte-identical numeric labels are not promised across machines, worker counts, or MLX versions.
- Admission-decision drift near threshold boundaries is guarded by the borderline serial-rerun pass (§9). Candidates whose parallel-mode label is within ε of any admission threshold are re-labeled in the main process, so threshold-sensitive admission decisions are exactly the serial-reference answer.

## 5. CLI surface

```
--label-worker-mode {serial,process}        default: process
--label-workers INT (>=1)                    default: 10 (process) / 1 (serial)
--mcts-eval-batch-size INT (>=1)             default: 14
--mcts-stall-flush-sims INT (>=0)            default: 16
--allow-unsafe-eval-batch                    flag,    default: off
--admission-borderline-epsilon FLOAT (>=0)   default: 0.01     (0 disables)
--no-borderline-rerun                        flag,    default: off
```

**Default change post-benchmark:** Originally specified as `serial / 1` for
"zero behavior change for existing users." Updated to `process / 10` after
the M3 Pro benchmark in scripts/probes/benchmark_phase2_knobs.py confirmed
3.19x speedup with admitted-ID equivalence. Tests that rely on serial-mode
byte-identity must pass `--label-worker-mode serial` explicitly.

**Validation (pre-Phase-1):**

- `--label-workers` must be `>= 1`.
- `--mcts-eval-batch-size` must be `>= 1`.
- `--mcts-eval-batch-size > SAFE_METAL_EVAL_BATCH_SIZE_MAX (14)` requires `--allow-unsafe-eval-batch`. Error message:
  > `--mcts-eval-batch-size > 14 is unsafe on Metal/MLX and may hang. Pass --allow-unsafe-eval-batch to benchmark higher values intentionally.`
- `--mcts-stall-flush-sims` must be `>= 0`.
- `--admission-borderline-epsilon` must be `>= 0`.
- If `--label-worker-mode=serial` and `--label-workers != 1`, print warning, set effective workers to 1 (no error). Warning text:
  > `[probe_suite] warning: --label-workers is ignored when --label-worker-mode=serial`
- The constant `SAFE_METAL_EVAL_BATCH_SIZE_MAX = 14` lives at module top of `scripts/build_probe_suite.py` and is referenced by both validation and help text.

**Hardcoded behaviors (no CLI knobs):**

- `multiprocessing` start method: `spawn`. `fork` + MLX is unsafe on macOS.
- Phase-2 progress cadence: existing `max(1, n_total // 20)` heuristic.
- Worker recycling: not enabled. Defer until memory pressure is observed in practice.

## 6. Architecture

```
Phase 1 (unchanged): mine candidates from games -> list[cand]

Phase 2:
  if args.label_worker_mode == "serial":
      results = [_label_one_strong_advantage_candidate(cand, ...) for cand in candidates]
  else:  # "process"
      ctx = multiprocessing.get_context("spawn")
      mcts_cfg_payload = {
          "eval_batch_size": args.mcts_eval_batch_size,
          "stall_flush_sims": args.mcts_stall_flush_sims,
      }
      with ProcessPoolExecutor(
          max_workers=effective_workers,
          mp_context=ctx,
          initializer=_init_label_worker,
          initargs=(str(label_ckpt), mcts_cfg_payload),
      ) as pool:
          futures = [pool.submit(_label_one_strong_advantage_candidate, cand, ...) for cand in candidates]
          results = []
          for fut in as_completed(futures):
              results.append(fut.result())
              # streaming progress logging, no aggregation here

  results.sort(key=lambda r: r["probe_id"])

  if borderline_rerun_enabled:
      for r in results:
          if _is_borderline(r, epsilon, magnitude_threshold, top1_share_floor, stability_cap):
              r = _rerun_in_main_process(r, ...)  # replaces label, re-applies filter

  # Final aggregation happens AFTER reruns. Status is post-rerun status.
  admitted = [r["candidate"] for r in results if r["status"] == "admitted"]
  audit.extend(r["audit_row"] for r in results if r["audit_row"] is not None)

Phase 3 (unchanged): diversity selector
```

The serial path and the process-pool path call the same `_label_one_strong_advantage_candidate` helper. The serial path threads it directly; the process-pool path submits it to the executor.

**Borderline-rerun-enabled** when:
```
args.label_worker_mode == "process"
and args.admission_borderline_epsilon > 0
and not args.no_borderline_rerun
```

## 7. Helper extraction & result dict

The current Phase 2 loop body becomes a pure-ish function:

```python
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
    cand = copy.deepcopy(cand)  # avoid mutation differences between modes
    ...
```

It returns:

```python
{
    "probe_id": str,
    "status": "admitted" | "rejected" | "replay_error" | "mcts_error",
    "candidate": dict | None,         # mutated copy with phase2_label set
    "audit_row": dict | None,         # rejection-style audit row (None for admitted)
    "rejection_reason": str | None,   # for borderline-flip detection
    "phase2_label": dict | None,      # readable without inspecting candidate
    "error_message": str | None,      # populated on replay_error / mcts_error
}
```

Status invariants:

| status | candidate | phase2_label | audit_row | error_message |
|---|---|---|---|---|
| `admitted` | not None | not None | None | None |
| `rejected` | not None | not None | not None | None |
| `replay_error` | None | None | not None | not None |
| `mcts_error` | not None | None | not None | not None |

Replay errors and MCTS errors are caught inside the helper and returned as a result; never raised across the process boundary. Worker survives, pool survives.

## 8. MCTSConfig wiring (probe_eval.py)

Add a module-level companion global, mirroring the existing `_DEFAULT_LABELER_NETWORK` pattern:

```python
_DEFAULT_LABELER_NETWORK = None
_DEFAULT_LABELER_MCTS_CONFIG = None  # NEW

def _set_default_labeler_mcts_config(config) -> None:
    global _DEFAULT_LABELER_MCTS_CONFIG
    _DEFAULT_LABELER_MCTS_CONFIG = config


def _default_mcts_labeler(state, sims, seed):
    if _DEFAULT_LABELER_NETWORK is None:
        raise RuntimeError(...)
    evaluator = LocalGPUEvaluator(_DEFAULT_LABELER_NETWORK)
    if _DEFAULT_LABELER_MCTS_CONFIG is None:
        cfg = MCTSConfig(n_simulations=sims)            # back-compat path
    else:
        cfg = replace(_DEFAULT_LABELER_MCTS_CONFIG, n_simulations=sims)
    mcts = MCTS(evaluator, cfg, rng=random.Random(seed))
    ...
```

`dataclasses.replace(..., n_simulations=sims)` keeps the per-call `sims` argument authoritative — the global config carries only `eval_batch_size` and `stall_flush_sims`, never a frozen `n_simulations`.

The 3-arg `labeler(state, sims, seed)` protocol is preserved. Existing tests that inject `labeler=` callables bypass both globals and need no changes.

**Worker initializer:**

```python
def _init_label_worker(label_checkpoint: str, mcts_cfg_payload: dict):
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

`MCTSConfig` crosses the process boundary as a plain dict, not as a dataclass instance.

**The worker payload includes ONLY the user-tunable fields** — `eval_batch_size` and `stall_flush_sims`. It must NOT include `n_simulations`; that value is per-call and supplied through the labeler protocol's `sims` argument, then overridden onto the global config via `dataclasses.replace(...)` inside `_default_mcts_labeler`. Including `n_simulations` in the worker payload would freeze it incorrectly and is a forward-drift hazard.

**Initializer failure** is unrecoverable. The first `.result()` call surfaces the exception; main aborts with:

```
[probe_suite] ERROR: failed to initialize process label worker from checkpoint <path>
mode=process workers=4
<original exception text>
```

**Main-process setup** (so borderline reruns work):

```python
network, _ic, _h, _nb = load_network_for_scoring(str(label_ckpt))
network.eval()
_set_default_labeler_network(network)
_set_default_labeler_mcts_config(
    MCTSConfig(
        eval_batch_size=args.mcts_eval_batch_size,
        stall_flush_sims=args.mcts_stall_flush_sims,
    )
)
```

## 9. Borderline rerun

**Trigger.** A candidate is borderline if its parallel-mode (post-STM-normalization) `phase2_label` lies within ε of any admission threshold:

```
abs(abs(mean_root_value) - magnitude_threshold) <= ε
abs(min_top1_share        - top1_share_floor)   <= ε
abs(value_stability       - stability_cap)      <= ε
```

Applied to **both admitted and rejected** parallel-mode results — otherwise admission drift is one-sided. Results with `status` of `replay_error` or `mcts_error` carry no `phase2_label` and are excluded from the borderline check; their audit rows are unchanged by the rerun pass.

**Execution.** Re-run `_default_mcts_labeler` synchronously in the main process. Same `rng_seed_base`, same `sims`, same `repeats`, same MCTSConfig (`_DEFAULT_LABELER_MCTS_CONFIG`), same checkpoint already loaded into `_DEFAULT_LABELER_NETWORK`. The rerun executes outside any worker process, so it sees a single MLX context — i.e., the serial reference answer.

**Authority.** The rerun result replaces the parallel result. Admission filter is re-applied **once**. The rerun is final — even if it lands borderline again, accept it (no recursive loop).

**Rerun audit metadata** is attached to the candidate object during rerun so it follows the candidate into whatever final audit row it eventually receives (Phase 2 rejection, diversity drop, or admitted):

```python
cand["_borderline_rerun_audit"] = {
    "borderline_rerun": True,
    "borderline_rerun_reason": ["magnitude", "top1_share"],          # subset of triggers
    "parallel_phase2_label_before_rerun": <pre-rerun label dict>,    # audit-only
    "borderline_rerun_flipped": True | False,
    "parallel_admission_reason": <old reason>,                       # only if flipped
    "serial_rerun_admission_reason": <new reason>,                   # only if flipped
}
```

When any audit row is constructed for that candidate (Phase 2 rejection or selector audit), these fields are merged in. **The committed probe suite stores only the rerun `phase2_label`** — `_borderline_rerun_audit` and `parallel_phase2_label_before_rerun` never land in the committed JSON, only in the audit file.

**Implementation note:** before serializing each entry in `probes_out`, explicitly pop `_borderline_rerun_audit` (and any other private `_`-prefixed key) from the candidate dict. Asserted in `test_borderline_rerun_admitted_to_rejected` and `test_borderline_rerun_rejected_to_admitted` (§12).

If a rerun flipped admission, log:

```
[probe_suite] borderline rerun flipped <probe_id>: <old_reason> -> <new_reason>
```

**Cost.** Worst case = full serial Phase 2 cost added on top of parallel Phase 2 (if every candidate is borderline). Realistic case at ε=0.01: <5–10% of candidates trigger rerun.

## 10. Instrumentation

Phase-2 summary line printed at end of Phase 2:

```
[probe_suite] Phase 2 complete: <n>/<n> labeled (<admitted> admitted, <s>s total)
  mode=process workers_requested=4 workers_effective=4 eval_batch=14 stall_flush=16
  candidates_total=<n> labeled=<n> replay_errors=<n> mcts_errors=<n>
  admitted_before_diversity=<n> rejected=<n>
  borderline_candidates=<n> borderline_reruns=<n> borderline_flips=<n>
  borderline_rerun_seconds=<s>
  Per-reason: <reason>=<n>, ...
```

Same stats land in the draft JSON under a NEW `meta.phase2_run_stats` block (NOT inside `meta.selection_rules`):

```json
{
  "meta": {
    "selection_rules": { ... },
    "phase2_run_stats": {
      "mode": "process",
      "workers_requested": 4,
      "workers_effective": 4,
      "eval_batch_size": 14,
      "stall_flush_sims": 16,
      "candidates_total": 123,
      "labeled": 120,
      "replay_errors": 1,
      "mcts_errors": 2,
      "admitted_before_diversity": 35,
      "rejected": 85,
      "borderline_rerun_enabled": true,
      "admission_borderline_epsilon": 0.01,
      "borderline_candidates": 6,
      "borderline_reruns": 6,
      "borderline_flips": 1,
      "borderline_rerun_seconds": 42.13,
      "seconds_total": 812.44,
      "rejection_reasons": {
        "magnitude_below_threshold": 40,
        "low_top1_share": 30,
        "unstable_value": 15
      }
    }
  }
}
```

`selection_rules` continues to hold deterministic configuration; `phase2_run_stats` holds runtime execution stats. They are kept as siblings to keep the meaning clean.

## 11. Lifecycle / error handling

- **Replay error** in worker (rebuilding `TwixtState` from `move_history`): caught inside helper, returned as `status="replay_error"`. Pool unaffected.
- **MCTS error** in worker: caught inside helper, returned as `status="mcts_error"`. Pool unaffected.
- **Worker initializer failure** (checkpoint load fails): unrecoverable. Surfaces on first `.result()`. Main aborts with the formatted error in §8.
- **KeyboardInterrupt in main**: `pool.shutdown(cancel_futures=True)` (Python ≥3.9; project runs on 3.14). Whatever finished is discarded — we have not reached the sort/aggregate step, so no half-written draft.

## 12. Test strategy

**Pytest (repo CI)** — all using mocked labelers unless noted:

| Test | Mode | Asserts |
|---|---|---|
| `test_phase2_serial_unchanged` | mocked, default flags | byte-identical output to a tiny checked-in golden fixture (NOT the full real committed suite) |
| `test_phase2_serial_vs_process_mocked` | mocked, both modes | byte-identical draft + audit; admitted IDs equal; rejection reasons equal |
| `test_phase2_process_pool_init_failure` | bad checkpoint path | clear error referencing path/mode/workers; exit nonzero; no draft written |
| `test_phase2_replay_error_isolated` | one cand raises during state replay | other candidates still labeled; audit row has `reason="replay_error"` and `error_message` populated |
| `test_phase2_mcts_error_isolated` | one cand raises during MCTS | symmetric, with `reason="mcts_error"` |
| `test_default_mcts_labeler_uses_global_config` | unit | with config global set: `cfg.eval_batch_size==X`, `cfg.stall_flush_sims==Y`, `cfg.n_simulations==sims`; without global: `MCTSConfig(n_simulations=sims)` (back-compat) |
| `test_default_mcts_labeler_keeps_sims_authoritative` | unit | `dataclasses.replace` ensures per-call `sims` overrides any frozen value in the config |
| `test_borderline_rerun_admitted_to_rejected` | mocked flip | flip recorded; final committed suite uses rerun label; audit has `borderline_rerun_flipped=True`; **committed probe JSON contains no `_borderline_rerun_audit` or `parallel_phase2_label_before_rerun` keys** |
| `test_borderline_rerun_rejected_to_admitted` | mocked symmetric flip | same, including the no-private-keys assertion on committed JSON |
| `test_borderline_rerun_disabled` | `--no-borderline-rerun` | no reruns happen; `borderline_reruns=0` |
| `test_borderline_rerun_serial_mode_no_op` | `--label-worker-mode=serial` with ε=0.01 | no reruns; counter=0 |
| `test_borderline_rerun_not_triggered_for_non_borderline` | clearly admitted/rejected candidates in process mode | no reruns; counter=0 |
| `test_admission_borderline_epsilon_zero_disables` | ε=0 in process mode | no reruns; counter=0 |
| `test_unsafe_eval_batch_validation` | `--mcts-eval-batch-size 32` without flag | argparse error; exit nonzero |
| `test_workers_under_serial_mode_warns` | `--label-worker-mode=serial --label-workers=4` | warning printed; `workers_effective=1`; no error |
| `test_parallel_cli_numeric_validation` (parametrized) | invalid values | reject `--label-workers 0`, `--mcts-eval-batch-size 0`, `--mcts-stall-flush-sims -1`, `--admission-borderline-epsilon -0.1` |
| `test_argparse_flags_present` | parser introspection | each new flag exists; default value matches spec; `--label-worker-mode` choices are `{serial, process}`; `--allow-unsafe-eval-batch` and `--no-borderline-rerun` default `False` |

**Manual / out-of-CI** — `scripts/probes/verify_parallel_equivalence.py`:

```
.venv/bin/python scripts/probes/verify_parallel_equivalence.py \
  --input scripts/GPU/logs/games \
  --source-iter-range 57 58 \
  --label-checkpoint <ckpt>.safetensors \
  --sample-candidates 20 \
  --label-workers 4
```

Output:

```
serial_admitted_ids == process_admitted_ids: true/false
serial_final_ids    == process_final_ids:    true/false
max_abs_mean_root_value_diff: ...
max_abs_value_per_run_diff:   ...
max_abs_min_top1_share_diff:  ...
borderline_reruns: ...
borderline_flips:  ...
```

Real-MLX equivalence is intentionally not in CI (slow, machine-dependent).

## 13. Future stages (deferred, for context only)

- **Stage 2:** Adaptive sim tiers — coarse pass at low sims, deep pass only on candidates that pass an early gate.
- **Stage 3:** Cross-candidate batched MLX evaluator — all candidate MCTS leaves go through one shared evaluator that batches across candidates. Best Metal utilization. Requires restructuring MCTS to expose its leaf-eval queue to a shared pool.
- **Stage 4:** Rolling thresholds / dynamic ε based on per-tier label-noise empirical estimates.

None of these are in scope for this PR.

## 14. Doc updates

| File | What changes |
|---|---|
| `docs/probe-suite-generation.md` | (1) Add new flags to the existing knob table at lines ~49–60. (2) Add a new `### Parallel labeling` subsection under `## Performance` (after line ~132) explaining mode flags, the determinism contract from §4, the borderline-rerun behavior, and the `--label-workers 1 + --label-worker-mode process` "plumbing test, not a speedup" caveat. (3) Update the example command around line ~100 to include a commented-out `# Faster (opt-in): --label-worker-mode process --label-workers 4` line. (4) Add a brief "Reproducibility" note: serial mode is the byte-reference path for mocked/deterministic labelers and the supported strict reproducibility mode for generated artifacts; for real MLX runs we promise admitted-ID equivalence and threshold-guarded admission decisions, not absolute byte identity across machines/MLX versions. (5) Add this recommended-starting-values block: |

```
Recommended starting point on Apple Silicon:
  --label-worker-mode process --label-workers 2
If stable and faster, try:
  --label-workers 4
Avoid increasing --mcts-eval-batch-size above 14 unless intentionally
benchmarking with --allow-unsafe-eval-batch.

Higher worker counts are not always faster: each process loads its own
MLX network and can contend for the Metal scheduler.
```

| `tests/probes/README.md` | If it documents the generation command line, mirror the new flags. (Read during implementation; choose between a one-line cross-reference back to `probe-suite-generation.md` or a fuller mirror.) |
| `scripts/build_probe_suite.py` (top docstring) | Mention parallel labeling is now supported under `--tier strong_advantage` with safe defaults preserving prior byte-identity. The argparse `description` keeps using `__doc__.split("\n\n", 1)[0]` so no other plumbing change is needed. |
| argparse `help=` strings | Each new flag gets a one-line `help=` description (auto-surfaces via `--help`). The `--allow-unsafe-eval-batch` help text explicitly mentions Metal hangs. |

**Files NOT updated:**

- `docs/superpowers/specs/2026-04-28-strong-advantage-probe-tier-design.md` — point-in-time tier spec.
- `docs/superpowers/plans/2026-04-28-strong-advantage-probe-tier.md` — point-in-time plan.
- `docs/train-cli.md` — only references the script in passing for trainer-side scoring; behavior there is unchanged.
- `scripts/build_bootstrap_probe_suite.py` — forced-tier shim. No Phase 2.
