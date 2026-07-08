# Targeted Value Calibration v13c — Projection-Strength Scalar Design

**Status:** DESIGN — approved + locked by user 2026-07-07; awaiting written-spec review, then writing-plans.
**Date:** 2026-07-07
**Follows:** v13 (asymmetric gradient-conflict projection, merged @ `95f3622` + telemetry-flattening fix `89e2965`). v13 RUN → REJECTED (A pass, B/C/D fail); the telemetry fix now makes `calib_projection_conflict_rate` and the projection counters readable. v13c is the **one final projection branch**: if it fails, the projection line stops.

## Why v13c (vs the already-rejected v13b margin-0.05)

`v13b` (guardrail-margin 0.05, an arg-only run) was rejected: a smaller margin makes **more** guardrail rows active on **every** step, which globally tightens hinge pressure — it weakened A and did not solve C/D. v13c tests a different lever: strengthen the conflict correction **only where the A and guardrail gradients actually conflict**, leaving the margin at 0.10 so no guardrail row is globally tightened. The question v13c answers: *does stronger conflict correction help, without globally increasing hinge pressure?*

## Decision: fold `projection_strength` into the effective projection weight

The v13 correction is `g_final = g_total − weight·c·g_G`, applied by the pure geometric primitive `project_conflicting_gradient` (trainer.py:310) with `weight = calibration_loss_weight` passed at the single call site (trainer.py:1594). v13c keeps that primitive **unchanged** and passes a scaled effective weight:

```
effective_projection_weight = post_opening_calibration_projection_strength * calibration_loss_weight
project_conflicting_gradient(_surf_total, _surf_a, _surf_g, weight=effective_projection_weight)
```

Rationale: `project_conflicting_gradient` stays a pure geometric primitive; `c = dot/(normsq+1e-12)`, the conflict-only gate (`dot < 0` and `norm_G > eps`), and the projected surface are all independent of strength. `strength` scales *only* the magnitude of an already-detected, already-directed correction.

- **At `strength=1.0`:** `effective_projection_weight = calibration_loss_weight`, so the projection update is **numerically identical to v13** (given the same seed/data/order and no unrelated code change — this is a per-update numerical-identity claim, NOT a whole-run checkpoint byte-identity claim across a stochastic training run).
- **At `strength=2.0`:** same conflict detection, same surface, same `c`/`cos`/`dot` geometry — only the applied correction (and `removed_norm`) is 2×, and only on conflicting steps.

## The change (5 edits, no helper signature change)

| File | Change |
|---|---|
| `scripts/GPU/alphazero/train.py` | add `--post-opening-calibration-projection-strength` (float, default `1.0`) + plumb into the `train(...)` call |
| `scripts/GPU/alphazero/trainer.py` `train()` | add param `post_opening_calibration_projection_strength: float = 1.0`; forward it to the calibration-branch `train_step(...)` call |
| `scripts/GPU/alphazero/trainer.py` `train_step` | add param `post_opening_calibration_projection_strength: float = 1.0`; at the projection call site compute `effective_projection_weight = post_opening_calibration_projection_strength * calibration_loss_weight` and pass it as `weight=` — the ONE behavioral line |
| `scripts/GPU/alphazero/trainer.py` telemetry | add `"proj_strength": post_opening_calibration_projection_strength` to the `loss_accumulator` dict; add `"calib_projection_strength"` to the `_teacher_calib_scalars` flattening-mirror tuple (so it reaches `model_iter_*.json`) |
| `scripts/GPU/alphazero/calibration_pool.py` `build_post_opening_calibration_block` | add `"calib_projection_strength": float(loss_accumulator.get("proj_strength", 1.0))` to the `"loss"` dict |

**Do NOT change:** `project_conflicting_gradient`'s signature, the conflict-only gate, `c = dot/(normsq+1e-12)`, the projected surface (`value_head.*` + `encoder.blocks[last].*`), the v12b manifest/schedule, the guardrail margin (stays 0.10), the component-loss helper, or `alphazero_loss_batch`.

## Telemetry requirement (mandatory, both sites)

`calib_projection_strength` MUST be persisted in **both** JSON sites — the nested sidecar `post_opening_calibration.loss` block AND the flattened `model_iter_*.json` row (the `_teacher_calib_scalars` mirror). This is the recurring two-site gotcha: a new calibration telemetry key added to only the sidecar is silently dropped from the per-iteration row (the v13 bug fixed in `89e2965`).

The existing projection telemetry is unchanged in meaning: `calib_projection_c_avg`/`cos_avg`/`dot_avg` stay the **geometric** conflict (independent of strength); `calib_projection_removed_norm_avg` = `|strength·calibration_loss_weight·c|·norm_G` reports the **actual applied** correction magnitude (includes strength). The operator reads both.

## Byte-identical / determinism semantics

- **Projection flag OFF:** byte-identical to v12b (unchanged — the strength arg is never read).
- **Flag ON, `strength=1.0`:** the projection update is numerically identical to v13 per step; the new `calib_projection_strength` telemetry key is additive (defaults 1.0). No whole-run byte-identity is claimed (a stochastic training run differs run-to-run regardless).

## Tests (behavioral, not mocks)

- **Strength scales the correction:** on a conflicting mixed A+guardrail batch on the v9 surface, a `train_step` with `projection_strength=2.0` produces a surface gradient whose deviation from `g_total` (and `removed_norm`) is 2× the `strength=1.0` deviation; `c`/`cos`/`dot` telemetry are identical between the two (geometry unchanged).
- **strength=1.0 == v13:** `projection_strength=1.0` yields the same projected surface gradient as the pre-v13c path (numerically identical update).
- **No-op stays no-op:** on a non-conflicting batch, any strength leaves `g_total` unchanged (strength scales zero).
- **CLI plumb:** `--post-opening-calibration-projection-strength` reaches `train()` → the calibration-branch `train_step(...)`.
- **Telemetry both sites:** `calib_projection_strength` appears in `build_post_opening_calibration_block`'s output (calibration_pool.py) AND in the `_teacher_calib_scalars` flattening mirror (trainer.py) — pinned so it reaches `model_iter_*.json`.

## Operator run (USER's, after merge)

The same canonical v12b/v13 command (full harness + v12b manifest + schedule + `--guardrail-margin 0.10 --freeze-batchnorm-stats --train-value-head-and-final-block --post-opening-calibration-gradient-projection`) **plus `--post-opening-calibration-projection-strength 2.0`**, new checkpoint dir `checkpoints/alphazero-v13c-projection-strength-from-calib020-0001`. Confirm `calib_projection_strength=2.0` in `model_iter_*.json`; nonzero `calib_projection_conflict_steps`; `verify_value_head_and_final_block_checkpoint` exit 0; gates A/B/C/D vs `calib020_0001`.

## Interpreting v13c (final projection-line decision)

- **A holds + B/C/D pass** → stronger conflict correction was the missing piece; promotion match.
- **A holds + B/C/D improve over v13 but still fail (conflict_rate > 0, larger removed_norm)** → projection helps directionally but remains insufficient at strength=2.0. Do NOT continue blind strength/surface tuning without a new written design; v13c is the final projection branch, so the default next step is **v14 = adapter / gated value correction** (a new objective, not another projection knob).
- **A weakens** → strengthening the conflict correction sacrifices too much A even conflict-gated.
- **No change vs v13 / `conflict_rate ≈ 0`** → the A/guardrail gradients rarely conflict directionally on the surface; strengthening a rare correction cannot help ⇒ **the projection line is exhausted; stop it** (the bind is representational/capacity, not a per-step surface gradient conflict).
