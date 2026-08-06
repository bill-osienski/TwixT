# Atlas Pilot — Operator Authorization #2 (post-Amendment 5)

**Issued:** 2026-08-05 · **Scope:** the pilot alone · **Status:** AUTHORIZED

Supersedes `2026-08-05-atlas-pilot-authorization.md`, which is **spent**: its block is
permanently retired and its geometry gate returned `PHASE_GEOMETRY_NO_GO`. This is now
the only atlas authorization in force.

## Frozen inputs

| item | value |
|---|---|
| qualified code | `1332bcc` |
| suite at that commit | **2,636 passed / 4 skipped / 53 deselected / 0 failed** |
| checkpoint | `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors` |
| checkpoint SHA-1 | `209cf2d4fd24a48553d259dd71b4954867b9473e` — **unchanged** |
| base seed range | `[20321000, 20321480)` — 480 games, new and non-overlapping |
| **pilot interval** | **`[0, 24)` only — seeds `[20321000, 20321024)`** |
| sampling seed | `20260806` |
| phase rule | Amendment 5: `phase_index = min(3, (4*ply) // n_moves)`, domain `0 ≤ ply < n_moves` |

The checkpoint is frozen by path **and** digest because `measure_provenance` binds the
digest consistently across the chain but cannot know it is the *right* network. A
`*-from-calib020-0001` derivative would pass every provenance check while silently
measuring a checkpoint this line already rejected.

**Qualified code vs. bound HEAD.** `1332bcc` is where the suite was measured. Committing
this document moves HEAD past it; that is documentation-only and leaves `scripts/`
untouched, so the qualification stands. Any commit touching `scripts/` requires
re-running the suite before launch. The run measures and binds whatever HEAD is current
when it starts, which is what the no-edit-after-preflight and
no-commit-between-generation-and-qualification rules keep constant.

## What is authorized

1. **Pilot block generation over `[0, 24)`** of the frozen range — 24 games, and no more.
   **No game with index ≥ 24.**
2. **The pilot geometry gate.**
3. **The 24-row pilot measurement (`run-pilot`) if and only if the gate passes.**
4. **Qualification and reporting of the resulting pilot artifact.**

Both preflight invocations below are covered and are zero-GPU.

## What remains unauthorized

- **Continuation block generation** — its size `G_total − 24` is a function of `N`, which
  does not exist until the pilot reports.
- **`run-final`.**
- **Replacements, top-ups and re-runs.** A no-go is a result, not a retry prompt.
- Any prototype, any `mcts.py` change, any adoption decision.

## The retired corpus

The 24 games at **`[20320000, 20320024)`** are **permanently retired** — geometry-design
evidence only. No position from them may enter discovery or validation, ever, and they
are not to be re-gated under Amendment 5. The new range does not overlap them.

## Procedure

`run_atlas emit-runbook --out-dir <dir>` prints the exact commands. In order:

1. **Clean tree, record HEAD.** No source edit after preflight; no commit between
   generation and qualification.
2. **Checkpoint-only preflight** — `run_atlas preflight --checkpoint <net>`, with
   `--pilot-dir` **omitted**, because the block does not exist yet.
3. **Generate `[0, 24)`.** Its own directory, which fails closed if one exists. The
   generator runs its own source/checkpoint preflight before constructing the evaluator.
4. **Preflight again WITH `--pilot-dir`** — the earliest point at which the symmetric
   digest-and-HEAD binding against the generated manifest is possible.
5. **`build_atlas_corpus pilot-gate`.** A `PHASE_GEOMETRY_NO_GO` means **stop**.
6. **`run_atlas run-pilot`**, through the detached shell wrapper. Launch and wait in
   **separate** shell invocations.
7. **Read both sidecars in a later call.** Do not wait on the PID.

   ```text
   shell_status + status.json   -> trust .verdict
   shell_status only            -> python died before reporting; read run.log
   neither                      -> the wrapper never ran; nothing was measured
   ```

## Stop conditions

| verdict | exit | action |
|---|---:|---|
| `PROVENANCE_FAILURE` | 3 | Not reconstructible. Fix and start over. |
| `PHASE_GEOMETRY_NO_GO` | 3 | Stop before the pilot ladder. Not a smaller pilot, not replacements. |
| `ABORTED` | 5 | A position went unmeasured. The pilot is not a pilot. |
| `UNAVAILABLE` sizing | 5 | An aborted pilot does not size. Do not carry a partial `N`. |
| `PROJECTED_CAPACITY_NO_GO` | 0 | A **finding**: stop the atlas rather than spend the full run. |
| `both_fail` early widening | 0 | A **finding**: close progressive widening without inventing another shape. |

Exit 0 with a stopping finding is the protocol working. Only usage, provenance and
incompleteness are nonzero.

## What to report

The artifact, plus: `verdict`, `authoritative`, `assigned`/`measured`, `failed_rows`;
`sizing` (`p_m`, `p_s`, `verdict`, `N`); `early_widening_check` per shape with
`both_fail` and its authoritativeness; and the three unknowns the pilot exists to close —
observed throughput, the `remaining` distribution (**a median of zero fails Read-out A's
controller-deployability claim**), and the inheritance-reset rate. Also the observed
per-phase ply supply, which is what a runtime projection must be built from.

**Under Amendment 5 the phase distribution is no longer in question** — every game of
8+ moves serves all eight cells — so the ply supply now matters for *cost*, not
feasibility.

## Standing rules

- The pilot's 24 rows are **discovery only**, never eligible for validation.
- **No top-up.** A short pilot is a no-go, not a shortfall to fill.
- The pilot is for identity, runtime, class-frequency sizing and the frozen widening kill
  condition **only**. It must not be used to adjust any threshold.
- An undefined statistic is `null`, never `0`.
- Amendments precede the work they govern. If this pilot's result suggests a change, the
  change is written **before** the next run.
