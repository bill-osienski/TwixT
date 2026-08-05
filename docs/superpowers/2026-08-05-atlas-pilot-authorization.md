# Atlas Pilot — Operator Authorization

**Issued:** 2026-08-05 · **Scope:** the pilot alone · **Status:** AUTHORIZED

This document authorizes GPU work. It is deliberately narrow, and it is the **only**
atlas authorization currently in force.

**Binding tooling state:** branch `atlas-convergence-diagnostic`, HEAD
`04bc8eccc64d4f48560d5742a7bc8fabf4bed2dd`, clean worktree, full suite
**2624 passed / 4 skipped / 53 deselected / 0 failed**. Qualification recorded in
`docs/superpowers/plans/2026-08-05-atlas-stage5-handoff.md`.

## The checkpoint — the one input the tooling cannot validate

```text
checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors
SHA-1  209cf2d4fd24a48553d259dd71b4954867b9473e        (7,524,333 bytes)
```

Verified against the frozen screen anchor at
`docs/alphazero-value-search-experiment-ledger.md:110`, and re-measured from the file at
authorization time. **`from0409` describes this checkpoint's own origin; it is NOT one of
the later `*-from-calib020-0001` derivatives**, all 36 of which are the rejected
calibration branches this line closed.

**Why this is written down rather than left to the operator.** `measure_provenance`
binds the checkpoint digest *consistently* across the pilot block, the continuation
block, the pilot artifact and the run — but it has no way to know the digest is the
**right** network. Point it at a v13 or v14 derivative and every provenance check
passes, the artifact records a clean chain, and the atlas silently measures a checkpoint
this line already rejected. It is the one operator input no gate can catch, so it is
frozen here by path *and* digest.

## What is authorized

1. **Pilot block generation over the half-open interval `[0, 24)`** of the frozen seed
   range — 24 games, and no more — against the checkpoint frozen above.
2. **The pilot geometry gate** (`build_atlas_corpus pilot-gate`).
3. **The 24-row pilot measurement (`run_atlas run-pilot`) if and only if the gate
   passes.**
4. **Qualification and reporting of the resulting pilot artifact.**

Both preflight invocations in the procedure below are covered, and are zero-GPU.

Nothing else. In particular this authorization does **not** cover generating a single
game with index ≥ 24.

## What remains unauthorized

- **Continuation block generation.** Its size `G_total − 24` is a *function of `N`*, and
  `N` does not exist until the pilot reports.
- **`run_atlas run-final`.** It refuses a non-authoritative or unsized pilot by
  construction; this is the written counterpart of that refusal.
- Any re-run, top-up, or replacement of pilot games.
- Any prototype, any `mcts.py` change, any adoption decision.

**Both stay unauthorized until the pilot produces a valid `N` and its early-widening
result.** Reaching them requires a new written authorization citing those two outputs.

## Frozen parameters — none of these is a decision to make at run time

| parameter | value | source |
|---|---|---|
| board | `active_size = 24` | §3 |
| generation | 400 sims, 280 max moves, batching `(14,48,8)`, **noise ON** | §3 / `PRODUCTION_SETTINGS` |
| ladder | `+400 → +1,200 → +1,600 → +3,200`, **noise OFF** | §4 |
| prefix replay | 400-sim searches | §2b |
| boundary | first flush completion at or after 320 target-search backups | §4 |
| pilot | 24 games, 3 per phase × side cell, **all discovery** | §3 |
| seeds | `game_seed = base_seed + game_idx`; one `random.Random(replay_seed)` per row, never reseeded | §2b / §3 |

The CLI exposes no flag that can change any of them. If a run appears to need one, that
is a protocol question, not a command-line question.

## Procedure

Run `run_atlas emit-runbook --out-dir <dir>` for the exact commands. In outline:

1. **Clean tree, and record HEAD.** No source edit after preflight; no commit between
   generation and qualification.
2. **Checkpoint-only preflight — `run_atlas preflight --checkpoint <net>`, with
   `--pilot-dir` OMITTED.** Measures the worktree, HEAD and checkpoint digest. Exit `0`
   continue · `2` usage · `3` PROVENANCE_FAILURE.
3. **Generate `[0, 24)`.** One block directory, which fails closed if it already exists.
   The generator runs `preflight_source_provenance` itself, **before** constructing the
   evaluator, so a dirty tree or unidentifiable checkpoint costs nothing.
4. **Preflight again, now WITH `--pilot-dir <pilot_block>`**, to bind the generated
   manifest: this is the step that checks the block's recorded digest and HEAD against
   the measured ones.
5. **`build_atlas_corpus pilot-gate`.** A `PHASE_GEOMETRY_NO_GO` means **stop** — not a
   smaller pilot, not replacement games, not reassignment.
6. **`run_atlas run-pilot`**, launched through the detached shell wrapper the runbook
   prints. Launch and wait in **separate** shell invocations: a tool timeout SIGTERMs the
   whole process group when they share one, and `setsid` does not exist on macOS.
7. **Read both sidecars, in a later call.** Do not try to wait on the PID — after
   `disown` a later shell has neither the job table nor a usable job spec, which is
   exactly how Phase 0 lost its exit code.

> **Why preflight runs twice.** `--pilot-dir` names a block that does not exist yet at
> step 2, so a single pre-generation preflight cannot bind a manifest — it would fail on
> a missing directory, and the original one-step ordering in this document was simply not
> executable. `--pilot-dir` is optional on the parser, so the checkpoint-only form needs
> no code change. The two phases check different things and both are required: step 2
> refuses a dirty tree or an unidentifiable checkpoint *before* any GPU time is spent;
> step 4 is the earliest point at which the symmetric digest-and-HEAD binding against the
> generated manifest is even possible.

   ```text
   shell_status + status.json present  -> trust .verdict
   shell_status only                   -> python died before reporting; read run.log
   neither                             -> the wrapper never ran; nothing was measured
   ```

## Stop conditions

| verdict | exit | action |
|---|---:|---|
| `PROVENANCE_FAILURE` | 3 | The run is not reconstructible. Fix and start over. |
| `PHASE_GEOMETRY_NO_GO` | 3 | Stop before the pilot ladder. This is what the gate is for. |
| `ABORTED` | 5 | One or more of the 24 positions was not measured. The pilot is **not** a pilot; diagnose and re-run from a clean state. |
| `UNAVAILABLE` sizing | 5 | An aborted pilot does not size. Do **not** carry a partial `N`. |
| `PROJECTED_CAPACITY_NO_GO` | 0 | A **finding**: the projected corpus exceeds 400 or a class frequency is zero. Stop the atlas here rather than spending the full run. |
| `both_fail` in the early widening check | 0 | A **finding**: close progressive widening **without inventing another shape**. |

Exit `0` with a stopping finding is the protocol working, not a failure. Only usage,
provenance and incompleteness are nonzero.

## What to report

The pilot artifact plus, quoted from it:

- `verdict`, `authoritative`, `assigned`, `measured`, `failed_rows`
- `sizing` — `p_m`, `p_s`, `verdict`, `N`
- `early_widening_check` per shape and `both_fail`, with
  `early_widening_check_authoritative`
- the **three measured unknowns** the pilot exists to close: observed throughput, the
  `remaining` distribution (§4: **a median of zero fails Read-out A's
  controller-deployability claim**), and the inheritance-reset rate
- the observed per-phase ply supply, which is what a runtime projection must be built
  from — **never a scaled smoke**

## Standing rules

- The pilot's 24 rows are **discovery only** and are never eligible for validation.
- **No top-up.** If the pilot is short, it is a no-go, not a shortfall to be filled.
- The pilot is for identity, runtime, class-frequency sizing and the frozen widening kill
  condition **only**. It must not be used to adjust any threshold.
- An undefined statistic is `null`, never `0`, and never a reason to drop a row.
- Amendments precede the work they govern. If the pilot's result suggests a change, the
  change is written **before** the next run, never after seeing a number.
