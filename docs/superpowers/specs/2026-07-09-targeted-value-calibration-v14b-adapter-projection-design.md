# Targeted Value Calibration v14b — Gradient Projection over the Value-Adapter Surface Design

**Status:** DESIGN — approved by user 2026-07-09 (decisions A–G locked in-session); awaiting written-spec review, then writing-plans.
**Date:** 2026-07-09
**Follows:** v14 (gated value-only adapter surface, merged @ 823525b) and the v13/v13c gradient-projection line. v14 result = **A moved but B/C/D drifted** — the v14b trigger named in v14 spec §12. v14 changed the training *surface*; v14b adds the v13 asymmetric *A-yields-to-guardrail* gradient projection **over that surface**, changing exactly one thing vs v14: projection is now allowed and operates on `{value_head, value_adapter}` instead of `{value_head, encoder.blocks[last]}`.

## Hypothesis / acceptance question

**Does projecting the A-correction gradient to yield to the guardrail gradient — over the value-only adapter surface `{value_head, value_adapter}` — hold B/C/D while keeping the A gain the adapter capacity unlocked?**

- v14 (adapter surface): moved A (capacity) but B/C/D drifted — the value function is nonlocal even value-only.
- v13 (final-block surface): projection is the right *mechanism* (A yields where it conflicts with the guardrail) but the final block was too nonlocal a surface.
- v14b: apply the v13 projection mechanism on v14's isolated, policy-safe surface — conflict-resolution where A and the guardrail disagree, without touching the trunk or policy.

## The change — one combined mode, no new flag

| File | Change |
|---|---|
| `scripts/GPU/alphazero/trainer.py` (`train_step`) | relax the projection guard to allow the adapter surface; select the projected surface by mode (`{value_head, value_adapter}` for v14b); fold `value_adapter_grad_norm` (post-projection) into `_proj_telem` under v14b |
| `scripts/GPU/alphazero/trainer.py` (`train()`) | accumulator branches slot `[13]` by `isinstance(extra, dict)`, not by the v14 flag; reads `value_adapter_grad_norm` from the dict when present |
| `scripts/GPU/alphazero/train.py` | relax any guard tying `--post-opening-calibration-gradient-projection` to the final-block surface so it composes with `--train-value-head-and-value-adapter`; keep the three-surface mutual exclusion |
| `tests/test_v14b_adapter_projection.py` (new) | projection accepted on the adapter surface; surface is `{value_head, value_adapter}`; grad-norm folded + accumulated; self-describing slot compatibility (v13/v14/v14b); verifier exit 0 |

## §A Activation — no new flag

v14b is the **combination** of existing flags:
`--value-adapter --train-value-head-and-value-adapter --post-opening-calibration-gradient-projection` (+ optional `--post-opening-calibration-projection-strength N`, `--value-adapter-bottleneck-width W`), with `--guardrail-margin 0.10 --freeze-batchnorm-stats`.
The projection flag — previously **rejected** on the adapter surface in v14 — now drives projection over `{value_head, value_adapter}`. There is no new CLI flag; v14b is a code change to what the projection flag *does* when the adapter-training flag is set.

## §B Projection guard

The `train_step` projection guard rejects projection **only** for `train_value_head_only` (a single trainable surface — `value_head` — offers no A-vs-guardrail conflict to project). It **allows** both `train_value_head_and_final_block` (v13, existing) and `train_value_head_and_value_adapter` (v14b, new).

## §C Projection surface — surface-agnostic, mode-selected

**Projection remains surface-agnostic; `train_step` chooses which trainable surface to present to `project_conflicting_gradient`.** We are not changing projection math — only the surface dictionary. The conflict-only gate, `c = dot/(normsq + 1e-12)`, `removed_norm`, and all geometry (`dot`/`cos`) are UNCHANGED. Only the surface differs:

- v13/v9 path: `{value_head, encoder.blocks[last]}`
- v14b path: `{value_head, value_adapter}`

The A-correction and guardrail-hinge gradients (the two `nn.value_and_grad` passes on `_calibration_component_loss`, modes `a_correction` / `guardrail_hinge`) already include `value_adapter` grads because the adapter participates in `forward_padded`. `train_step` assembles `_surf_total` / `_surf_a` / `_surf_g` from the `value_head` + `value_adapter` slices, calls the unchanged `project_conflicting_gradient`, and writes the projected grads back to `grads["value_head"]` and `grads["value_adapter"]`.

## §D Grad-norm — folded into the projection dict, post-projection

Under v14b, `value_adapter_grad_norm` is folded into the projection telemetry dict: after the adapter grad norm is computed on the **post-projection** (applied) adapter grad, set `_proj_telem["value_adapter_grad_norm"] = <float>`.

It is the **post-projection / applied** adapter grad norm — the operational signal "how much adapter gradient survived after A yielded to the guardrail." The fold happens whenever projection produced a `_proj_telem` dict under v14b — **including the no-A / no-guardrail / no-conflict skip dicts** — so the accumulator always finds the key (the accumulator's `.get(..., 0.0)` degrades gracefully, but the key should be present on every v14b dict). A pre-projection diagnostic (`value_adapter_grad_norm_pre_projection`) is an OPTIONAL later addition **and MUST NOT be added now** — it would introduce another telemetry field and another interpretation branch for no first-run benefit.

## §E Return slot [13] — self-describing; accumulator branches by type

Slot `[13]` of the guardrail return tuple is **self-describing**:

- **dict** ⇒ projection telemetry (v13; and v14b, where the dict also carries `value_adapter_grad_norm`)
- **float** ⇒ adapter grad norm only (v14, projection off)
- **absent** (13-tuple) ⇒ plain guardrail (no projection, no adapter)

This keeps v13, v14, and v14b compatible **without changing return arity**. `train()`'s accumulator branches by `isinstance(extra, dict)`, NOT by the v14 flag:

```python
extra = _ret[13]
if isinstance(extra, dict):
    proj = extra
    sum_value_adapter_grad_norm += float(extra.get("value_adapter_grad_norm", 0.0))
else:
    proj = None
    sum_value_adapter_grad_norm += float(extra)
```

Byte-identical for v13 (dict, no `value_adapter_grad_norm` key → `+0.0`, sum stays 0) and v14 (float path). Under v14b, both the projection accumulation (conflict / `removed_norm` / …) and the grad-norm accumulation run.

## §F Telemetry — both sites

v14b emits, in BOTH JSON sites (sidecar `build_post_opening_calibration_block` loss block + the flattened `_teacher_calib_scalars` mirror):

- the projection metrics (`calib_projection_*`, as v13)
- `value_adapter_gate` (direct `network.value_adapter.gate` read, as v14)
- `value_adapter_grad_norm` (from `sum_value_adapter_grad_norm`, sourced from the dict under v14b)

plus the run-level bool `train_value_head_and_value_adapter` (as v14). No new telemetry field beyond folding `value_adapter_grad_norm` into the existing projection dict.

## §G Invariants / do NOT change

- `project_conflicting_gradient` — UNCHANGED (surface-agnostic; v14b changes only the surface dict passed in).
- Optimizer routing — UNCHANGED (`opt_main` skipped; one `opt_value.update(value_module, {value_head, value_adapter})` applying the projected value-side grads; no Adam double-step).
- `verify_value_head_and_adapter_checkpoint.py` — UNCHANGED (a v14b checkpoint still changes only `value_head.*` + `value_adapter.*`; projection modifies grads before applying, not the trained surface).
- Byte-identical when projection is off (v14 path intact) and when the adapter / train-adapter flags are off (v13 / v9 / v8 paths intact).
- No change to `_calibration_component_loss`, `alphazero_loss_batch`, the v12b manifest/builder, the guardrail margin, the v12b schedule, `MainModule`, `mcts.py`, `continuation_extraction.py`, the v8/v9 verifiers.

## §H Tests (behavioral)

- **Guard:** projection ACCEPTED (no `ValueError`) when `train_value_head_and_value_adapter` + projection are both set; still REJECTED for `train_value_head_only` + projection.
- **Surface:** on a forced A-vs-guardrail conflict, the projected `value_head` + `value_adapter` grads differ from the unprojected grads; `encoder.blocks[last]` is NOT projected under v14b.
- **Grad-norm fold:** under v14b the slot-`[13]` dict carries `value_adapter_grad_norm` (float); the accumulator reads it; the sidecar + mirror emit it.
- **Self-describing slot:** v13 (dict, no grad_norm key) still accumulates (grad_norm stays 0, byte-identical); v14 (float) path unchanged; v14b (dict + grad_norm) accumulates both.
- **Byte-identical:** projection-off v14 unchanged; v13 final-block projection unchanged.
- **Verifier:** exit 0 on a v14b checkpoint (only `value_head.*` + `value_adapter.*` changed).

## §I Byte-identical / determinism semantics

- **v14b OFF** (projection flag absent): the v14 value-side path + telemetry are byte-identical.
- **v14 OFF** (adapter / train-adapter flags absent): the v13 / v9 projection path is byte-identical.
- No whole-run byte-identity is claimed across a stochastic run (as with every prior branch). At `projection-strength = 1.0` the per-update projection is numerically identical to v14 only where there is no A-vs-guardrail conflict; on conflicting steps it diverges by construction.

## §J Operator run (USER's, after merge)

The v14 command PLUS `--post-opening-calibration-gradient-projection` (+ optional `--post-opening-calibration-projection-strength N`), new checkpoint dir `checkpoints/alphazero-v14b-adapter-projection-from-calib020-0001`. Confirm: `train_value_head_and_value_adapter=true`, projection active (`calib_projection_conflict_steps > 0`), `value_adapter_gate` off 0, `value_adapter_grad_norm` present (post-projection), `verify_value_head_and_adapter_checkpoint` exit 0; then gates A/B/C/D vs `calib020_0001` (no promotion unless all four pass).

## §K Branch order / interpreting v14b

- **A holds + B/C/D recover** → projection over the isolated adapter surface is the fix → promotion match.
- **A regresses** (projection too aggressive) → lower `--post-opening-calibration-projection-strength` before abandoning.
- **B/C/D still drift** → the drift is not gradient-conflict on this surface → reconsider (per-channel gate / richer adapter — a later written design).
- Optional later diagnostic, only if needed: `value_adapter_grad_norm_pre_projection` (do NOT add in this branch).
