# Targeted Value Calibration v14 — Gated Value-Adapter Surface Design

**Status:** DESIGN — approved section-by-section by user 2026-07-08; awaiting written-spec review, then writing-plans.
**Date:** 2026-07-08
**Follows:** the v9–v13 final-block line and the v13/v13b/v13c gradient-projection line. The argument-only projection knobs are exhausted (user-reported): `--guardrail-margin 0.05` → rejected; `--post-opening-calibration-projection-strength 2.0` → best near-pass but still failed gate D; dormant D-root schedule → rejected. The v13 line taught us the **mechanism** helps but **final-block updates are too nonlocal** — moving `encoder.blocks[last]` to fix A drifts the shared value function on B/C/D. v14 keeps the v12b objective and changes exactly one thing: the **training surface**.

## Hypothesis / acceptance question

**Does a value-only adapter — more capacity than value-head-only (v8), policy-isolated, and far less nonlocal than final-block training (v9–v13) — fix gate A without the guardrail drift (B/C/D) that final-block training caused?**

- v8 (value-head-only): protected B/C/D but could **not** move A — the shallow fc1→fc2 MLP on frozen features lacks capacity (representational constraint).
- v9–v13 (final block): could move A but repeatedly caused guardrail drift — the final block is shared by value **and** policy and by all positions, so the update is nonlocal.
- v14: a small value-only feature-correction adapter provides more capacity than value-head-only while touching **only** the value path — no policy contamination, no trunk edit.

v14 is a **code** change (a new third training surface), not an argument variant. The current surfaces are only `value_head_only` (v8) and `value_head_and_final_block` (v9); v14 adds `value_head_and_value_adapter`.

## The change — one new value-only surface, v12b objective unchanged

| File | Change |
|---|---|
| `scripts/GPU/alphazero/network.py` | new `ValueAdapter(nn.Module)` (1×1 bottleneck + folded scalar gate); opt-in construction on `AlphaZeroNetwork` via `create_network(..., value_adapter=..., value_adapter_bottleneck_width=...)`; a `_value_features(features)` helper applied in `forward_padded` **and** the `__call__` empty-moves branch |
| `scripts/GPU/alphazero/trainer.py` | `train_step` + `train()` param `train_value_head_and_value_adapter`; value-side optimizer routing through a `ValueModule` wrapper; mutual-exclusion guard; gate/adapter telemetry in both JSON sites |
| `scripts/GPU/alphazero/train.py` | `--value-adapter`, `--value-adapter-bottleneck-width`, `--train-value-head-and-value-adapter` args + plumb; graft-load assertion |
| `scripts/GPU/alphazero/verify_value_head_and_adapter_checkpoint.py` (new) | tensor-diff verifier: allowed-to-change = `value_head.*` + `value_adapter.*`; all other shared keys byte-identical |
| `tests/test_v14_value_adapter.py` (new) | zero-gate identity, graft-load, surface isolation, hinge-sees-adapter, byte-identical-off, mutual exclusion, verifier |

## §1 Adapter module + insertion point

`ValueAdapter(nn.Module)` operating on channels-last `(B,H,W,C)` encoder features (`C = hidden`, 128 in prod):

```
fc_down = nn.Linear(C, b)          # pointwise over channels == 1×1 conv; matches ValueHead's nn.Linear idiom
fc_up   = nn.Linear(b, C)
gate    = mx.zeros((1,))           # folded scalar gate, init 0.0 (ReZero-style); a named parameter of the adapter
                                   # module, so it saves/loads under the key "value_adapter.gate". Shape (1,) (not
                                   # 0-d) so MLX safetensors save_weights and the verifier see the key cleanly; the
                                   # forward/telemetry read the scalar via gate[0].
__call__(features) -> gate * fc_up(relu(fc_down(features)))
```

- Bottleneck width `b = value_adapter_bottleneck_width` (operator flag); default (`None`/`0`) → `C // 4` (= 32 when C=128).
- **No BatchNorm** in the adapter (keeps running stats out of the value-only surface; no new freeze concerns).
- **No spatial mixing** (pointwise only) — deliberately smaller than the v9 final-block spatial edit that caused drift.
- **Opt-in / byte-identical when off:** the adapter + gate are constructed **only** when `create_network(..., value_adapter=True)`. Base and all existing runs are architecturally unchanged, preserving the v8–v13 "byte-identical when the flag is off" invariant.
- **Identity at init:** with `gate = 0.0`, `value_adapter(features) = 0`, so `features_for_value = features` and the value output is **exactly** the base value. (ReZero bootstrap: at init only the gate receives gradient `∂/∂gate = fc_up(relu(fc_down(features)))`; the mlp params begin training once the gate leaves 0.)

## §2 Forward integration & inference consistency

Add a helper on `AlphaZeroNetwork`:

```
_value_features(features) -> features + value_adapter(features)   # returns features unchanged when adapter absent
```

Call it at **every** site that computes value, so training, batched MCTS, the guardrail hinge, and gate-eval all see the identical adapter-corrected value with **no train/inference skew**:
- `forward_padded` value branch: `value = value_head(_value_features(features), active_size)` (both the plain and `return_value_pretanh` paths).
- `__call__` empty-moves branch (`network.py:589-590`): use `_value_features(features)` too.

The **policy** path keeps raw `features` (unchanged). The guardrail-hinge / A-correction components (`_calibration_component_loss`) call `model.forward_padded`, so they automatically see the adapter — no change needed there.

## §3 Training surface + optimizer routing (MLX-correct)

- New flag `--train-value-head-and-value-adapter` (v14), **mutually exclusive** with `--train-value-head-only` (v8) and `--train-value-head-and-final-block` (v9): `ValueError` in `train_step`, `parser.error` in `train.py`. It also **requires** `--value-adapter` (you cannot train an adapter that was not constructed): `parser.error` if set without it.
- With the adapter present, `grads` (from `nn.value_and_grad`) gains a top-level `value_adapter` key. Route the value-side update through **one** `opt_value.update` on a `ValueModule(value_head, value_adapter)` wrapper (mirrors `MainModule`; single call ⇒ no Adam double-step; trains at `value_lr = learning_rate * value_lr_scale`, 0.1). The value-side grads `{value_head, value_adapter}` are clipped together at `value_grad_max_norm` (the existing value clip).
- `opt_main` is **skipped entirely** (encoder + policy + final block frozen), exactly like v8: encoder/policy grads are still computed and clipped (telemetry unchanged) but never applied.
- **Byte-identical when off:** when the adapter is absent, the value-side update is the existing `opt_value.update(network.value_head, value_grads)` unchanged; `ValueModule` is used **only** when the adapter is present.
- Pair with `--freeze-batchnorm-stats` (BN running stats frozen), as v8/v9 do.

## §4 Checkpoint graft-load

- When the adapter is present, load the base checkpoint with `network.load_weights(base, strict=False)` (the base has no adapter keys). When the adapter is absent, the existing strict `load_weights` path is unchanged (byte-identical off).
- **When the adapter is present, immediately assert the only keys missing from the file are exactly the `value_adapter.*` set** (compare the network's parameter key-set to the checkpoint's key-set). Any other missing/extra key raises — a real load bug cannot hide behind `strict=False`.
- The gate stays `0.0` at load ⇒ the first-iteration value == the base value exactly.

## §5 Objective — unchanged from v12b

- Same v12b manifest, schedule, asymmetric guardrail hinge (`asymmetric_guardrail_retention`), `--guardrail-margin 0.10`, calibration weight. **Gradient projection OFF** — `--post-opening-calibration-gradient-projection` stays wired but is not used in v14 (reserved for v14b). The A hard_value correction + guardrail hinge automatically act on the adapter-corrected value.
- No change to `calibration_pool.py` loss modes, `alphazero_loss_batch`, the v12b manifest/builder, `project_conflicting_gradient`, `_calibration_component_loss`, or the guardrail margin.

## §6 Verifier — `verify_value_head_and_adapter_checkpoint.py` (new)

Tensor-diff two safetensors checkpoints (base vs v14) analogous to `verify_value_head_and_final_block_checkpoint.py`:
- **Allowed-to-change:** `value_head.*` and `value_adapter.*` (the adapter keys are *new*, allowed to appear in the v14 checkpoint only). The gate saves as `value_adapter.gate`, so it is covered by the `value_adapter.*` prefix — the verifier asserts the gate key is present under that prefix; if a future refactor ever hoists the gate to a top-level `value_gate` key, the verifier's allow-set must add it explicitly (the design mandates `value_adapter.gate`).
- **Must be byte-identical:** every shared key that is not `value_head.*` — the entire encoder **including the final block** (catches accidental trunk edits), `policy_head.*`, and **all BatchNorm running stats everywhere** (catches a forgotten `--freeze-batchnorm-stats`).
- Exit codes: `0` pass / `1` leak (a forbidden key changed) / `2` value-path no-op (gate never left 0.0 / adapter identical — collapsed to no correction) / `3` new-key set is not exactly `value_adapter.*`.

## §7 Telemetry

Surface **both** `value_adapter_gate` and `value_adapter_grad_norm` into **both** JSON sites (the recurring two-site gotcha — sidecar `build_post_opening_calibration_block` loss block in `calibration_pool.py` **and** the flattened `model_iter_*.json` `_teacher_calib_scalars` mirror in `trainer.py`), plus the top-level run field `train_value_head_and_value_adapter: bool` in the state dict.

- **`value_adapter_gate`** — the scalar gate value (watch it open from 0.0). Read from `network.value_adapter.gate[0]` at telemetry-build time (network in scope), fed through `loss_accumulator`, mirrored. No change to `train_step`'s return arity.
- **`value_adapter_grad_norm`** — the per-step L2 norm of the adapter grad subtree (gate + fc_down + fc_up), averaged over the iteration (bootstrap "how hard is the adapter being pushed"). Computed in `train_step` (reusing `clip_grad_norm`'s global-norm on `grads["value_adapter"]`) and returned as **one extra trailing float** on the guardrail return tuple — exactly mirroring how `_proj_telem` is appended. This is safe because **projection and v14 are mutually exclusive** (projection is rejected on the v14 surface), so the 14th slot is unambiguous: `train()` disambiguates by the `train_value_head_and_value_adapter` flag it already holds (v14 → the slot is a float grad norm; v13 → the slot is the projection dict; plain guardrail → 13-tuple, unchanged). `train()` accumulates `sum_value_adapter_grad_norm` and divides by steps.

Existing guardrail/A/projection telemetry is unchanged (byte-identical when v14 is off — the extra trailing float appears only when `train_value_head_and_value_adapter` is set).

## §8 Byte-identical / determinism semantics

- **Adapter flag OFF (`--value-adapter` absent):** the network has no adapter; `forward_padded`, the value-side update, and all telemetry are byte-identical to current `main`.
- **Adapter present, gate=0 at init:** value output byte-identical to the base network on the same weights (identity at init).
- No whole-run byte-identity is claimed across a stochastic training run (as with every prior branch).

## §9 Tests (behavioral, not mocks)

- **Zero-gate identity:** an adapter-present network with gate=0 produces `forward_padded` values byte-identical to the no-adapter network on the same loaded weights.
- **Graft-load:** `strict=False` load of a base checkpoint into an adapter network succeeds and leaves the gate at 0.0; the exact-delta assertion passes when only `value_adapter.*` is missing and **raises** when any other key is missing (inject a spurious rename to prove it fails loud).
- **Surface isolation:** after one v14 `train_step` on a mixed self-play + guardrail batch, only `value_head.*` and `value_adapter.*` changed; encoder blocks (incl. the last), policy_head, and BN running stats are byte-identical.
- **Hinge sees adapter:** the guardrail-hinge component value changes when the gate/adapter params change (adapter is in the value path).
- **Byte-identical-off:** with `value_adapter=False` and no v14 flag, the `train_step` value-side path is byte-identical to the current path.
- **Mutual exclusion + dependency:** setting v14 together with v8 or v9 raises; setting v14 without `--value-adapter` raises.
- **Telemetry both sites:** `value_adapter_gate` and `value_adapter_grad_norm` appear in both `build_post_opening_calibration_block`'s output (calibration_pool.py) **and** the flattened `_teacher_calib_scalars` mirror (trainer.py) — pinned so they reach `model_iter_*.json` (the recurring two-site gotcha).
- **Verifier:** exit 0 on a legal value_head+adapter delta; exit 1 on a trunk/policy/BN change; exit 2 on gate-never-moved; exit 3 on an unexpected new-key set.

## §10 Do NOT change

`project_conflicting_gradient`, `_calibration_component_loss`, `alphazero_loss_batch`, the v12b manifest/builder, the guardrail margin (0.10), the v12b schedule, the v8/v9 verifiers, `mcts.py`, `continuation_extraction.py`, `docs/post-game-analysis.md`, `MainModule`. The projection flag stays wired but unused.

## §11 Operator run (USER's, after merge)

The canonical v12b command **minus** `--train-value-head-and-final-block` and (if present) `--post-opening-calibration-gradient-projection`, **plus** `--value-adapter` (+ optional `--value-adapter-bottleneck-width 32`) and `--train-value-head-and-value-adapter`, with `--guardrail-margin 0.10 --freeze-batchnorm-stats`, new checkpoint dir `checkpoints/alphazero-v14-value-adapter-from-calib020-0001`. Confirm `train_value_head_and_value_adapter=true` + `value_adapter_gate` telemetry present (and moving off 0) + `verify_value_head_and_adapter_checkpoint` exit 0; then gates A/B/C/D vs `calib020_0001` (no promotion unless all four pass).

## §12 Branch order & interpreting v14

- **v14:** adapter surface + v12b hinge, projection OFF, scalar gate, width flag. Encoder/policy/final-block/BN frozen.
- **A moves + B/C/D hold** → adapter capacity was the missing piece → promotion match.
- **A moves + B/C/D drift** → the value function itself is nonlocal even value-only → **v14b** = add the v13 gradient projection over the adapter surface `{value_head, value_adapter}` (code).
- **A does not move (underfit)** → first try a wider bottleneck arg-only (`--value-adapter-bottleneck-width 64`); if still underfit → **later branch, new written design** = per-channel gate or a richer adapter (a mechanism change, not a tuning flag).
- **v14c** (objective change) only with a new written reason.
