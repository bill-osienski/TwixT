# Targeted Value Calibration v13 — Asymmetric Gradient-Conflict Projection Design

**Status:** DESIGN — approved + locked by user 2026-07-06; awaiting written-spec review, then writing-plans.
**Date:** 2026-07-06
**Follows:** v12 (asymmetric guardrail hinge, merged @ `2cc4bd1`) and v12b (continuation guardrails, merged @ `7335605` + hardening `4234dc3`). v13 keeps the v12b objective and manifest unchanged and changes only the **update mechanics**: it resolves the A-vs-guardrail gradient conflict on the shared final-block surface.

## Why v13 is a new mechanic, not another row/schedule variant

The v9→v12b sequence established a representational bind: **A (the hard black-predrop correction) needs final-block movement, but final-block movement breaks the B/C/D guardrails.** v9 fixed A by unfreezing `encoder.blocks[5]` and broke B/C/D; v12/v12b reshaped the guardrail loss (one-sided hinge, continuation states) but still apply A and the guardrail as a single summed gradient — so on any step where the A-correction gradient points against the guardrail-hinge gradient, the sum can move the shared final block in a direction that raises guardrail overvalue. v13's hypothesis: **let A move only in directions that do not fight the guardrail.** This is gradient-conflict handling (PCGrad-style projection), not a new objective or manifest — deferred to v13 precisely because it needs gradients for the A and guardrail terms computed separately, which the single bundled backward pass cannot provide.

## Locked scope

- **Objective + manifest + surface unchanged from v12b:** base `calib020_0001`; `--freeze-batchnorm-stats --train-value-head-and-final-block` (trainable surface = `value_head.*` + `encoder.blocks[5].*`); the v12b one-sided black-perspective guardrail hinge; the v12b continuation-guardrail manifest + schedule; `--guardrail-margin 0.10`; no policy CE.
- **No** new manifest, new gate, margin change, schedule change, symmetric PCGrad, or projection over frozen/never-applied leaves.
- **New:** an asymmetric, conflict-only projection of the A gradient away from the guardrail gradient, behind a new flag. Additive — off by default.

## The gradient mechanic (the refinement, mathematically equal to "projected A + guardrail + self-play")

The existing update is one `nn.value_and_grad(network, loss_fn)(network)` over `total_loss = L_S + w·(L_A + L_G)` (self-play + `calibration_loss_weight·(value_term_A + hinge_G)`), giving `g_total = g_S + w·(g_A + g_G)`. Because gradients are linear, projecting A onto G and recombining collapses to a single correction on `g_total`:

```
g_A = ∇ L_A   (unweighted A value term)      # cheap calib-only backward pass
g_G = ∇ L_G   (unweighted guardrail hinge)   # cheap calib-only backward pass

# over the applied surface leaves only:
dot    = Σ_surface ⟨g_A, g_G⟩
normsq = Σ_surface ⟨g_G, g_G⟩
norm_G = sqrt(normsq)
apply  = (dot < 0) and (norm_G > eps)          # eps = 1e-8
c      = dot / (normsq + 1e-12)  if apply else 0.0

g_final[leaf] = g_total[leaf] − calibration_loss_weight · c · g_G[leaf]   # surface leaves only
```

Derivation: `g_A_proj = g_A − c·g_G`, so `g_S + w·g_A_proj + w·g_G = g_total − w·c·g_G`. When `dot ≥ 0` (no conflict), `c = 0` and `g_final = g_total` with **no subtraction performed at all**.

**Weighting (locked, to prevent a double-weight bug):** `g_A` and `g_G` are gradients of the **unweighted** component losses `L_A`, `L_G` (NOT multiplied by `calibration_loss_weight`). `c` is a ratio, so the weight cancels inside it; the single factor of `calibration_loss_weight` (`w`) appears exactly once, in the correction term `w·c·g_G`. This reproduces `g_total`'s `w·(g_A+g_G)` structure exactly.

## Projection scope — the applied trainable surface ONLY

`dot`, `normsq`, and the correction are computed over exactly the leaves that get updated: `value_head.*` and `encoder.blocks[last].*` (`last = len(encoder.blocks) − 1` = 5 in production, 1 in `n_blocks=2` tests). **Not** over the stem, `encoder.blocks[0..4]`, `policy_head`, BN running stats, or any frozen tensor — those never update under the v9 surface, so a conflict measured there would be a false signal. `g_A`/`g_G` are computed via `nn.value_and_grad` over the whole network (which returns the full tree); only the surface leaves are read for the dot/norm and written for the correction.

## Component-loss isolation (`_calibration_component_loss`)

A new calib-only helper forwards **only the calibration positions in eval mode** (the same eval-mode forward the bundled path uses for teacher/guardrail rows — same BN-eval semantics, same positions, same weights) and returns one scalar:

- `component="a_correction"` → the symmetric A **value term** over the `sign == 0` (hard_value) rows only (weight `base_w·(1−|sign|)`).
- `component="guardrail_hinge"` → the one-sided guardrail **hinge** over the `sign != 0` (guardrail) rows only (weight `base_w·|sign|`).

**No self-play batch, no policy CE, no teacher-retention path.** These are exactly the `value_term` and `guardrail_hinge_loss` quantities `alphazero_loss_batch` already computes in its guardrail branch, split so each can be differentiated on its own. **Selection is by the guardrail-sign MASK, never by tag** — the A/value component is weighted `base_w·(1−|sign|)` (so guardrail rows contribute exactly 0) and the guardrail component `base_w·|sign|` (so A rows contribute exactly 0). This avoids tag-specific branching and stays correct under any schedule (in the v13 schedule only `black_predrop_correction` is drawn as A; `red_predrop_severe_root_correction`, also `sign==0` if ever scheduled, would be included automatically). `g_A = nn.value_and_grad(net, λm: _calibration_component_loss(m, …, "a_correction"))(net)`; `g_G` likewise. Because they forward only the ~8-11 calib positions (not the 64-position self-play batch), the two extra backward passes are cheap relative to `g_total`.

## Plug-in point + order (locked)

Inside `train_step`, immediately after `g_total = nn.value_and_grad(...)` (trainer.py:1429) and **before** the `main_grads`/`value_grads` split (trainer.py:1455):

1. Compute `g_total` (unchanged).
2. If the projection flag is enabled **and** A rows **and** guardrail rows were drawn: compute `g_A`, `g_G`.
3. Correct `g_total` on the applied surface leaves (`value_head` + `encoder.blocks[last]`).
4. Split into `main_grads` / `value_grads` (unchanged).
5. Existing clipping (unchanged) — clips the **corrected** gradient, which is what we want.
6. Existing v9-guarded apply (unchanged).

## Flag, guards, and no-op conditions

- **`--post-opening-calibration-gradient-projection`** (store_true, default off) → `post_opening_calibration_gradient_projection` through `train()` → `train_step`.
- **Requires `--train-value-head-and-final-block`.** Enabling it with `--train-value-head-only` is a hard error (the value-head-only surface does not define the A-vs-guardrail final-block conflict).
- **No-op (skip the correction, leave `g_total` exactly unchanged) — telemetry-visible, never an error — when, on a step:**
  - no A (`sign == 0`) rows were drawn → `no_a_steps`;
  - no guardrail (`sign != 0`) rows were drawn → `no_guardrail_steps`;
  - `norm_G ≤ eps` → `tiny_guardrail_steps`;
  - `dot ≥ 0` (no conflict) → `no_conflict_steps`.
  Odd sampling batches are expected; a no-op is the safe behavior.

## Byte-identical / determinism semantics (worded carefully)

- **Flag OFF:** byte-identical to v12/v12b — no new code path executes; the 7/10/13-tuple arities and every value are unchanged, and all pre-existing calibration tests pass unmodified.
- **Flag ON, no conflict on a step (`dot ≥ 0`, or a no-op condition):** the corrected gradient object is **mathematically and elementwise equal to `g_total`** — no projection subtraction is applied to any leaf. We do **not** claim whole-run byte identity in this case: the two extra calib-only forward/backward passes touch MLX execution/cache state, so global run determinism may differ. The guarantee is per-leaf gradient equality, not whole-run identity.

## Telemetry

Core (averaged over steps where projection was **evaluatable** — both A and guardrail rows present and `norm_G > eps`):
- `calib_projection_enabled` (bool), `calib_projection_scope = "value_head_and_final_block"`
- `calib_projection_conflict_steps` (= applied steps; correction applied only on conflict), `calib_projection_conflict_rate` (conflict_steps / evaluatable_steps)
- `calib_projection_dot_avg`, `calib_projection_cos_avg` (`dot/(norm_A·norm_G + 1e-12)`), `calib_projection_c_avg`
- `calib_projection_removed_norm_avg` (`|w·c|·norm_G`, the magnitude subtracted from `g_total`)
- `calib_projection_guardrail_grad_norm_avg` (`norm_G`), `calib_projection_a_grad_norm_avg` (`norm_A`)

Skip counters (distinguish "no conflict" from "could not evaluate"):
- `calib_projection_no_a_steps`, `calib_projection_no_guardrail_steps`, `calib_projection_tiny_guardrail_steps`, `calib_projection_no_conflict_steps`

## Tests (behavioral, not mocks)

**Projection helper (pure function, unit-tested in isolation):**
- exact conflict: hand-built `g_A`, `g_G` pytrees with `dot < 0` → `g_final == g_total − w·c·g_G` on the surface, computed by hand;
- no conflict (`dot ≥ 0`) → `g_final` is elementwise equal to `g_total` (no leaf changed), AND the telemetry object reports `c == 0`, `removed_norm == 0`, `no_conflict_steps += 1`, `conflict_steps` unchanged (guards against a bug where the gradient is correctly left unchanged but the step is miscounted as an applied conflict);
- scope: only `value_head` + `encoder.blocks[last]` leaves change; stem/other-blocks/policy_head leaves are identical to `g_total`;
- epsilon: a near-zero `g_G` (`norm_G ≤ eps`) → no correction (guards div-by-zero / spurious huge `c`).

**Component loss:** `_calibration_component_loss(component="a_correction")` on a mixed batch returns the value term over the sign==0 rows only (guardrail rows contribute 0); `component="guardrail_hinge"` returns the hinge over the sign!=0 rows only; both use the eval-mode forward and carry no policy CE.

**Wiring:** flag off → `train_step` byte-identical tuple + values; flag on with a conflicting mixed calib batch → surface gradient differs from the unprojected gradient and the projection telemetry engages; `--train-value-head-only` + projection flag → `ValueError`.

**Smoke `smoke_v13_gradient_projection.py`:** load the v12b manifest, draw the v12b schedule, run a few `train_step`s with the projection flag on the v9 surface, assert the projection telemetry fields are present and the conflict/skip counters are internally consistent.

## Do NOT touch

The v12b loss/hinge math, `alphazero_loss_batch`'s existing bundled path (v13 adds separate component gradients alongside it, does not alter it), the v12b manifest/builder, mcts.py, continuation_extraction.py, the v8/v9 verifiers, docs/post-game-analysis.md, and the v12/v12b/teacher/continuation loss paths.

## Plan shape (5 tasks)

- T1: `_calibration_component_loss` helper + the pure `project_conflicting_gradient` helper (+ unit tests).
- T2: `train_step` wiring — compute `g_A`/`g_G`, apply the correction before the split, behind the flag; the `--train-value-head-only` mutual-exclusion error (+ wiring/byte-identical tests).
- T3: `train.py` CLI flag `--post-opening-calibration-gradient-projection` + plumb through `train()`.
- T4: projection telemetry (core + skip counters) accumulation + JSON.
- T5: gate-0 smoke + full suite + merge handoff.

## Operator run (USER's, after merge)

The same canonical v12b command (full harness: `--iterations 1 --lr 0.0003 --curriculum-sizes 24 --games-per-iter 100 --simulations 400 --max-moves 280 --batch-size 64 --mcts-eval-batch-size 14 --mcts-pending-virtual-visits 8 --mcts-stall-flush-sims 48 --n-workers 10 --opening-noise-ply 10 --opening-dirichlet-alpha 0.7 --opening-dirichlet-eps 0.35 --max-positions-per-game 280 + resign/adjudicate settings`), the v12b manifest + schedule + `--guardrail-margin 0.10 --freeze-batchnorm-stats --train-value-head-and-final-block`, **plus `--post-opening-calibration-gradient-projection`**, new checkpoint dir. Confirm the projection telemetry engaged (nonzero `conflict_steps` or evaluatable steps; `calib_projection_scope=value_head_and_final_block`); `verify_value_head_and_final_block_checkpoint` exit 0; gates A/B/C/D vs `calib020_0001`. No promotion unless all four pass.

## Interpreting v13

- **A + B/C/D all pass** → the diagnosed gradient conflict was the blocker; promotion match.
- **A holds and B/C/D improve over v12b, conflict_rate > 0** → projection is doing real work; consider a margin/weight tune.
- **conflict_rate ≈ 0** → the failure was not a per-step gradient conflict on the surface (the objectives rarely fight directionally) → the bind is elsewhere (capacity/representation), and projection is the wrong lever.
- **A weakens** → the asymmetric projection is sacrificing too much of A; consider projecting a fraction of the conflicting component, or revisit the surface — before abandoning.
