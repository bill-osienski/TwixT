# Targeted Value Calibration v4 — Teacher-Retention Anchors (Design)

**Created:** 2026-06-29 · **Status:** approved (brainstorming), pending implementation plan · **Scope:** the value-head calibration line of work — fix the black pre-drop overvalue (gate A) without breaking the guardrail families (gates B/C/D).

Supersedes the rejected scalar-MSE-retention branches (calib010 / v2 / v2b / v3 / v3b). Read the experiment ledger first: `docs/2026-06-26-targeted-value-calibration-experiment-ledger.md`. This design implements the ledger's "Next planned hypothesis — v4: Teacher-Retention Anchors."

## 1. Goal & controlled-comparison framing

One falsifiable question:

> **Does teacher-retention preserve B/C/D better than scalar-MSE retention, while holding the same A correction pressure?**

The first v4 run holds **everything** constant versus the rejected v3 run and changes **only the retention objective**. Held constant: base checkpoint, manifest **row set**, tag schedule `2:1:2:1`, global calibration weight `0.01`, batch fraction, **run length (= v3's run length, not a hardcoded iteration count)**, self-play settings, gate manifests, gate sims. Changed: the retention objective moves from scalar-MSE to teacher-retention distillation.

### Why this is not another rejected scalar-retargeting experiment

v3's retention rows were **already** a scalar teacher target — `target_black_value` on those rows was the teacher's **MCTS root value** (`probe_black_root_value`), per the v2/v3 builder. So v4's two precise changes versus v3 are:

1. Retention **value target**: MCTS-root scalar → teacher's **raw-NN value**.
2. **NEW:** retention **policy term** — teacher cross-entropy (≈ KL) to the teacher's full move distribution.

The raw-NN choice is what makes this clean **self-distillation**: base == teacher == `calib020_0001`, so at the first training step the retention value/policy loss is **identically zero** (the network reproduces its own raw outputs); the retention terms only activate as the A-correction term perturbs the shared encoder. **An MCTS-root target would be nonzero at step 0 and contaminate the comparison by dragging the raw head toward search** — turning v4 back into a scalar-retargeting experiment. Raw-NN is therefore not merely cheaper; it is the *correct* target for the self-distillation framing.

This is on the ledger's recommended path (move retention from scalar rows to teacher-retention distillation) and avoids every do-not-repeat item (no global-weight / retention-weight / schedule-ratio sweep; no scalar-MSE-only rows as the main strategy; no promotion match before A/B/C/D pass).

## 2. Acceptance gates (vs current best `calib020_0001`, 400-sim probes)

Unchanged from the ledger. **No promotion match unless all four gates pass.**

| Gate | Family | Pass criteria |
|---|---|---|
| **A** | black pre-drop (frozen-30, held out) | mean ≤ 0.0 **and** severe materially below 43.3% |
| **B** | goal-line | severe 0.0% **and** over ≤ 11.1% |
| **C** | old broad post-opening | severe ≤ 13.3% **and** over ≤ 33.3% **and** mean ≤ +0.099 |
| **D** | red pre-drop | severe = 0.0% **and** mean ≤ 0.0 |

The ledger's severe-overlap analysis motivates the policy term: **C** failures are stable repeat offenders with **high top-1 confidence** (distribution drift, not just value drift) → policy retention should matter; **D** failures are **diffuse / low top-1** → value retention should matter. v4 applies both to all retention rows; the first run uses both together (`teacher_policy_kl_weight = 0.25`).

## 3. Approaches considered

- **Option A — dense teacher policy carried in `PositionRecord.visit_counts`, reusing `make_padded_batch` + `compute_masked_log_probs`.** ✅ **Chosen.** Least invasive. The codebase already represents policy targets as a dense distribution aligned to `legal_moves`; `local_evaluator.infer` already returns priors in that exact order; the self-play policy loss `-Σ pi·log_probs` is already the cross-entropy/KL we need. `state.legal_moves()` is **"sorted for determinism"** (`game/twixt_state.py:212`), so a dense distribution aligned to that order is reproducible build-time ↔ train-time.
- **Option B — sparse teacher-policy JSON with a custom gather/KL path.** Rejected: re-implements masking / log-softmax that already exists; no benefit at this row count.
- **More scalar-MSE retention (more rows / weight tuning).** Rejected by the ledger's do-not-repeat list.

## 4. Architecture

### 4.1 `PositionRecord` — UNCHANGED

The broad self-play data model is not touched. Teacher targets ride in existing fields, populated by the calibration builder/parser:

- `record.outcome` ← value target, **side-to-move perspective**.
  - Correction rows: hard `−0.35` (black-perspective) → side-to-move via the existing `target_in_to_move`.
  - Retention rows: teacher raw-NN value, **already side-to-move** from `infer`.
  - **Perspective rule (no sign bug):** `teacher_value` is stored side-to-move and assigned **directly** to `record.outcome`. Do **NOT** route it through `target_in_to_move` (that helper is for black-perspective hard targets only). On retention rows, `target_black_value` is optional **debug-only** metadata and MUST NOT be used for training.
- `record.visit_counts` ← dense teacher policy aligned to `legal_moves` for retention rows; **zeros** for correction rows. (`make_padded_batch` casts counts to float and normalizes over legal moves, so float priors flow through the `List[int]` annotation unchanged; an all-zero row yields an all-zero `target_pi`.)

### 4.2 `CalibrationSample` — extended (metadata only)

```python
@dataclass(frozen=True)
class CalibrationSample:
    record: PositionRecord
    weight_scale: float = 1.0
    tag: str = ""
    target_black_value: float | None = None
    loss_mode: str = "hard_value"          # "hard_value" | "teacher_retention"
    teacher_value: float | None = None      # side-to-move; telemetry/validation
    teacher_policy_len: int | None = None    # == len(legal_moves); validation
```

### 4.3 Manifest schema

Source = the v3 stratified CSV (`logs/eval/targeted_calibration_v3_strat_from_calib020_0001.csv`, 128 rows: 50 `black_predrop_correction` + 78 retention across `goal_line_retention` 18 / `old_post_opening_retention` 30 / `red_predrop_retention` 30). The builder **appends four columns**; `load_csv_manifest` already passes unknown columns through untouched, so **no loader change is required**.

| New column | Rows | Meaning |
|---|---|---|
| `loss_mode` | all | `hard_value` (correction) or `teacher_retention` (guardrail) |
| `teacher_value` | retention only | teacher raw-NN value, **side-to-move** float |
| `teacher_policy_json` | retention only | dense JSON list of priors aligned to `legal_moves` |
| `teacher_legal_moves_sha1` | retention only | SHA-1 over the canonical `legal_moves` ordering at build time (alignment guard) |

Size: ~78 retention rows × ~400 floats grows the CSV from ~32 KB to a few hundred KB — acceptable. Correction rows leave all four teacher columns blank except `loss_mode=hard_value`.

## 5. Teacher manifest builder

New deterministic script `scripts/GPU/alphazero/build_teacher_calibration_manifest.py`, mirroring `build_targeted_calibration_manifest.py` (argparse, unified columns, no randomness). For each source row:

1. Reconstruct the position via `position_state(replay, position_ply, side_to_move)` — the same path the probe and `build_calibration_position` use.
2. Compute `legal = state.legal_moves()` (sorted/deterministic). Build a single padded batch (board HWC, move rows/cols/mask).
3. Run the **teacher checkpoint** through `LocalGPUEvaluator.infer` — **raw forward, no MCTS** → `priors (1, M)`, `value (1,)`.
4. Emit:
   - Correction rows: pass through unchanged; `loss_mode=hard_value`; teacher columns blank.
   - Retention rows: `loss_mode=teacher_retention`; `teacher_value = value[0]` (side-to-move); `teacher_policy_json = priors[0][:len(legal)]`; `teacher_legal_moves_sha1 = sha1(canonical(legal))`.

Canonical legal-move string for the hash: `";".join(f"{r},{c}" for r, c in legal)` encoded UTF-8 (one definition, shared by builder and loader).

**Inputs / output (first experiment):**
- Source: `logs/eval/targeted_calibration_v3_strat_from_calib020_0001.csv`
- Teacher: `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors` (`calib020_0001`)
- Output: `logs/eval/targeted_calibration_v4_teacher_from_calib020_0001.csv`

`infer` calls `forward_padded`, which canonicalizes internally; policy logits are returned in input-move column order (gather preserves column identity), so `priors[j] ↔ legal[j]`. The same alignment holds at train time.

## 6. Parsing & validation (`calibration_pool.py`)

`build_calibration_position` / `build_calibration_sample` branch on `loss_mode`. A new pool `schema="teacher_retention"` joins `global_target` / `per_row_target`.

**Fail-fast validation at load (before any self-play):**
- Retention rows require finite `teacher_value ∈ [−1.0, 1.0]`.
- `len(teacher_policy_json) == len(legal_moves)` (length guard).
- `teacher_legal_moves_sha1` **recomputed over the reconstructed `legal_moves` must match** the stored hash. This catches a same-length reordering that the length check alone would miss — a silent catastrophic alignment bug. Deterministic legal-move order is an explicit invariant of this design.
- Policy entries ≥ 0 and sum ≈ 1.0 (tol 1e-3).
- `hard_value` rows must have blank/zero teacher policy.

## 7. Loss (`alphazero_loss_batch`)

Keep the calibration forward's logits (today discarded as `_`) **only when a teacher-policy mask is present**; otherwise the path stays byte-identical to today.

Let the calibration minibatch have per-row weight `w_i` (= `weight_scale_i`) and mask `m_i ∈ {0,1}` (1 for `teacher_retention`, 0 for `hard_value`). `cb_targets_i = record.outcome_i`; `cb_target_pi_i` = normalized teacher policy (retention) or all-zero (correction).

```python
cb_logits, cb_values, _ = network.forward_padded(cb_boards, cb_rows, cb_cols, cb_mask, active_size=...)
cb_log_probs = compute_masked_log_probs(cb_logits, cb_mask)        # reuse existing helper

# Value term — ALL calibration rows (hard correction + teacher retention):
per_value_i  = (cb_values - cb_targets) ** 2
value_loss   = Σ_i (w_i · per_value_i)            / max(Σ_i w_i, 1e-8)

# Policy term — teacher_retention rows ONLY (gated by mask):
per_policy_i = -Σ_j cb_target_pi[i,j] · cb_log_probs[i,j]
policy_loss  = Σ_i (w_i · m_i · per_policy_i)     / max(Σ_i (w_i · m_i), 1e-8)

calib_loss   = teacher_value_weight · value_loss + teacher_policy_kl_weight · policy_loss
total_loss   = total_loss + calibration_loss_weight · calib_loss
```

**Edit-driven clarifications (all load-bearing):**

1. **Policy term is cross-entropy, called "KL" operationally.** The implemented term is teacher cross-entropy `−Σ teacher_pi · log cand_pi`, equivalent to `KL(teacher ‖ candidate)` up to the teacher-entropy constant (constant w.r.t. network params → identical gradients). Operator-facing flags/telemetry call it "policy KL."
2. **`teacher_value_weight` scales the ENTIRE calibration value-MSE term** — both hard-correction value-MSE on correction rows **and** teacher-value MSE on retention rows. (Despite the name; the CLI flag keeps the `teacher-value-weight` spelling for consistency with the plan.)
3. **Explicit policy denominator.** The policy weighted-mean denominator is `Σ_i (w_i · m_i)`, **not** `Σ_i w_i` — correction rows must not dilute the retention-policy average.
4. **Mask is the control mechanism.** The policy term is gated explicitly by `teacher_policy_mask`. The all-zero `target_pi` on correction rows is a **secondary safety property, not the control** — never rely on it alone.
5. **`weight_scale` scales both terms.** A row's `weight_scale` means "importance of this row," scaling value and policy together: `row_weight × (value_term + policy_term)`.

**Signature / return.** New args: `calibration_teacher_policy_mask`, `teacher_value_weight`, `teacher_policy_kl_weight`. The calibration return tuple extends to additionally surface `(calib_value_term, calib_policy_kl_term, n_teacher_retention)` for telemetry; the **non-calibration path stays the 7-tuple**, and `train_step` unpacking is updated in lockstep. `teacher_policy_kl_weight = 0` cleanly degrades to value-only (the ablation).

**Regression invariant.** When `calibration_teacher_policy_mask is None` — which §8 guarantees for every `global_target` / `per_row_target` (v2/v3) pool — the loss path is **byte-identical to today**: logits stay discarded, no `cb_log_probs`, no policy term. The decision is gated on `mask is None`, never on inspecting per-row `loss_mode`.

## 8. CLI & trainer wiring

New flags on `train.py` (defaults are the first-experiment values):
- `--post-opening-calibration-teacher-value-weight` (default `1.0`)
- `--post-opening-calibration-teacher-policy-kl-weight` (default `0.25`)

Reused unchanged: `--post-opening-calibration-enabled`, `--post-opening-calibration-manifest`, `--post-opening-calibration-weight` (`0.01`), `--post-opening-calibration-tag-schedule` (`black_predrop_correction=2,goal_line_retention=1,old_post_opening_retention=2,red_predrop_retention=1`).

`train()` builds the `teacher_policy_mask` in the existing per-step sampling block, **only when `pool.schema == "teacher_retention"`** — a 0/1 vector from each sampled row's `loss_mode` (1 for `teacher_retention`, 0 for `hard_value`). For `global_target` / `per_row_target` pools it passes `calibration_teacher_policy_mask=None`, which is what preserves the byte-identical v2/v3 path (§7). This schema gate — **not** the per-row `loss_mode` default — is the control: a v2/v3 manifest whose rows all default to `loss_mode="hard_value"` must still yield `None` (an all-zero mask would wrongly trigger the `cb_log_probs` computation). The mask + the two weights thread into `train_step` / `alphazero_loss_batch`.

## 9. Telemetry

Extend the `post_opening_calibration` sidecar block with `calib_value_term_avg_iter`, `calib_policy_kl_avg_iter`, and `n_teacher_retention_drawn`, keeping the existing `calib_loss_avg_iter`, `calib_mean_value_pred`, `draws_by_tag`, and `calib_n_drawn_by_tag`. Dict-valued telemetry stays in the model_iter JSON + sidecar, never in `metrics.csv` (v3 invariant).

## 10. Tests

- Builder writes the four columns; `teacher_policy_json` length == `legal_moves` and sums ≈ 1.0; `teacher_legal_moves_sha1` matches a freshly recomputed hash; correction rows leave teacher columns blank.
- Parser/loader: retention rows → nonzero `target_pi`; correction rows → zero `target_pi`; load-time validation **rejects** length mismatch, hash mismatch (same-length reorder), out-of-range `teacher_value`, and non-normalized policy.
- Loss: policy term applied to retention rows only (mask correctness); explicit denominator excludes correction rows; `weight_scale` scales both terms; `teacher_policy_kl_weight=0` == value-only.
- **Regression:** v2/v3 (`global_target` / `per_row_target`) manifests produce a byte-identical loss path (no policy term).
- Tag-stratified sampling unchanged; `teacher_policy_mask` built correctly from a stratified draw.

## 11. First experiment + operator gates

Self-distillation run: base = teacher = `calib020_0001`; manifest = `targeted_calibration_v4_teacher_from_calib020_0001.csv`; weights `global 0.01 / teacher_value 1.0 / teacher_policy_kl 0.25`; schedule `2:1:2:1`; **run length = v3's run length**. Then the four 400-sim gates (§2) vs `calib020_0001`. **No promotion match unless all four pass.** A value-only ablation (`teacher_policy_kl_weight 0.0`) is available for free if attribution is needed, but is not the first spend.

On completion (pass or fail), append a row to the experiment ledger and update its do-not-repeat / severe-overlap sections.

## 12. Files touched

| File | Change |
|---|---|
| `scripts/GPU/alphazero/build_teacher_calibration_manifest.py` | **New.** Deterministic teacher-cache builder (raw `infer`); appends `loss_mode` / `teacher_value` / `teacher_policy_json` / `teacher_legal_moves_sha1`. |
| `scripts/GPU/alphazero/calibration_pool.py` | Extend `CalibrationSample`; branch `build_calibration_position`/`build_calibration_sample` on `loss_mode`; `schema="teacher_retention"`; fail-fast validation incl. SHA-1; `split_samples_with_modes` returning the mask; extend telemetry block. |
| `scripts/GPU/alphazero/trainer.py` | Capture calib logits when mask present; value + masked-policy loss with explicit denominators; new args + extended return tuple; `train_step` unpack; build mask + thread weights in `train()`; sidecar fields. |
| `scripts/GPU/alphazero/train.py` | Two new CLI flags + thread into `train()`. |
| `tests/test_calibration_pool.py`, `tests/test_training.py`, `tests/test_calibration_cli_flags.py` (+ a builder test module) | Per §10. |
| `docs/2026-06-26-targeted-value-calibration-experiment-ledger.md` | Append v4 result row after the run. |

## 13. Non-goals (YAGNI)

- No MCTS-derived teacher targets (raw-NN only).
- No schedule / global-weight / retention-weight sweeps (ledger do-not-repeat).
- No `PositionRecord` schema change.
- No promotion match in this spec (gated on A/B/C/D).
- No new gate families; reuse the existing four.

## 14. Approved shape (summary)

```
v4 first experiment =
  same row set / schedule / global weight / run length as v3
  correction rows: hard value target (−0.35 black → side-to-move)
  retention rows: raw-NN teacher value (side-to-move) + dense teacher policy CE/KL
  PositionRecord unchanged; make_padded_batch reused; compute_masked_log_probs reused
  legal-move order pinned by SHA-1 fingerprint
  no MCTS teacher targets; no scalar/schedule/weight sweep
  no match unless A/B/C/D all pass
```
