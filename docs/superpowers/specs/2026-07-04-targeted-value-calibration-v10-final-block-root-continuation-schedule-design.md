# Targeted Value Calibration v10 — Guarded Final-Block + Root/Search-Path Retention (schedule-only) Design

**Status:** DESIGN — awaiting user review. **Config-only experiment: no code, no new manifest, no new verifier.**
**Date:** 2026-07-04
**Follows:** v9 (`--train-value-head-and-final-block`, merged @ b2c26eb) — RUN result below.

## v9 result (authoritative, gate summary.json @ git b2c26eb)

The final-block unfreeze produced the cleanest possible diagnostic: **A crossed the pass line for the first time, but B/C/D — all of which v8 held — broke.**

| gate | base (calib020_0001) | v9 | verdict |
|---|---|---|---|
| **A** black_predrop | mean +0.257, over 50.0%, sev 43.3% | **mean −0.089, over 30.0%, sev 16.7%** | **PASS** |
| **B** goal_line | mean −0.244, over 5.6%, sev 0.0% | mean −0.238, over 22.2%, sev 11.1% | FAIL |
| **C** old_post_opening | mean +0.099, over 33.3%, sev 13.3% | mean +0.067, over 46.7%, sev 30.0% | FAIL |
| **D** red_predrop | mean −0.188, over 13.3%, sev 0.0% | mean −0.115, over 26.7%, sev 20.0% | FAIL |

(v8 value-head-only, for contrast: A mean +0.068 FAIL; B/C/D all PASS. v9 is the mirror image — A fixed, guardrails broken.)

## Coverage diagnostic (2026-07-04) — the pivotal finding

For every v9-failed B/C/D case, checked against the v7 manifest via `continuation_parent_case_id` (not a case-id string split) and filtered to the v9 candidate rows (each `*_probe_cases.csv` holds BOTH base and candidate; the naive read conflates them):

- **Every** failed case (4/4 B, 14/14 C, 8/8 D) already had **continuation/PV retention rows that were scheduled and trained in v7** — zero uncovered, zero needing new extraction.
- ⇒ The continuation/PV-retention lever is **exhausted**: those exact cases were trained with continuation retention and broke anyway. "More continuation extraction" is not the next lever.
- The one **untested** lever is the **root/raw-preservation** tags that exist in the manifest but were **never scheduled**: `goal_line_retention` (18), `old_post_opening_retention` (30), `red_predrop_retention` (30), `red_predrop_root_value_retention` (30). v9 (like v8) trained **continuation-only** for B/C/D.

## v10 hypothesis (narrow, falsifiable)

**Can adding the dormant root/raw-preservation pressure recover B/C/D while keeping v9's A fix — using only a schedule change?**

Keep the v9 mechanics unchanged (`--freeze-batchnorm-stats --train-value-head-and-final-block`, same v7 manifest, weight 0.01). Change **only** the `--post-opening-calibration-tag-schedule` to enable the dormant root tags alongside the existing continuation tags. This is the last untouched scheduling lever under the v9 mechanics.

**This is framed as a cheap falsification test, not the expected solution.** Prior expectation (user): B likely improves, C maybe, D likely still fails, A may weaken. Worth one run before declaring partial-unfreeze dead for this line.

## The v10 schedule (per-tag draws/step; sums to 11/step vs v9's 8/step)

```
black_predrop_correction=2,goal_line_retention=1,goal_line_continuation_retention=1,old_post_opening_retention=1,old_post_opening_continuation_retention=2,red_predrop_root_value_retention=1,red_predrop_continuation_retention=2,red_predrop_severe_root_correction=1
```

| tag | draws/step | loss_mode | preservation kind |
|---|---|---|---|
| black_predrop_correction | 2 | hard_value | A hard correction (unchanged) |
| goal_line_retention | 1 | mcts_root_retention | **B root: value + root-visit policy CE** |
| goal_line_continuation_retention | 1 | searched_continuation_retention | B continuation (value-only) |
| old_post_opening_retention | 1 | mcts_root_retention | **C root: value + root-visit policy CE** |
| old_post_opening_continuation_retention | 2 | searched_continuation_retention | C continuation (value-only) |
| red_predrop_root_value_retention | 1 | searched_continuation_retention (root_value clone) | D root (value-ONLY, no policy CE) |
| red_predrop_continuation_retention | 2 | searched_continuation_retention | D continuation (value-only) |
| red_predrop_severe_root_correction | 1 | hard_value | D sparse severe correction |

## KEY DECISION (defaulted to Option 1 while user away — CONFIRM at review)

For B and C, the **only** root-preservation rows in the manifest are `goal_line_retention` / `old_post_opening_retention`, both `mcts_root_retention`. That loss_mode forces `has_policy_target=True` (calibration_pool.py `__post_init__`) and derives a **root-visit policy CE** target from `root_visits_json` — the **same mechanism deliberately excluded for D** (where `red_predrop_root_value_retention`, a value-only root clone with blank `root_visits_json`, is used instead). There is **no** value-only root variant for B/C in the manifest.

**Chosen (Option 1): accept the policy-CE root rows for B/C.** It is the only schedule-only way to give B/C root preservation, and it honors the user's "B/C get root + continuation" intent. Rationale: v6b's toxicity was **D**-root-visit-policy breaking B/C nonlocally; **B/C**-root-visit-policy applied to B/C themselves is untested. Interpretability caveat baked into the analysis: if B/C break *worse* than v9, root-visit policy CE is a prime suspect.

Rejected: **Drop B/C root** (then B/C get only continuation = unchanged from v9 = already-failed; weak test). **Extract value-only B/C root** (`goal_line_root_value_retention` / `old_post_opening_root_value_retention` don't exist → needs builder + new manifest → no longer the cheap zero-code test).

Explicitly NOT scheduled: `red_predrop_retention` (D's `mcts_root_retention` root-visit-policy rows) — v6b showed D root policy retention toxic; use the value-only `red_predrop_root_value_retention` for D instead.

## Mechanics (all pre-existing — nothing to build)

- Flags: `--freeze-batchnorm-stats --train-value-head-and-final-block` (v9, merged @ b2c26eb).
- Manifest: `logs/eval/targeted_calibration_v7_severe_d_root_correction_from_calib020_0001.csv` (v7, unchanged; all 8 tags present, pools ≫ draw counts).
- Schedule flag: `--post-opening-calibration-tag-schedule` (v3). `sample_by_tag` already consumes per-tag draws/step.
- Weight 0.01; teacher weights at v9 defaults (`teacher_value_weight=1.0`, `teacher_policy_kl_weight=0.25`) — note the policy-KL weight, inert in v9 (no policy rows drawn), is now **active** for the B/C root rows.
- Acceptance verifier: the v9 `verify_value_head_and_final_block_checkpoint.py` still applies unchanged (v10 uses the same flags) — must exit 0.

## Expected telemetry (differs from v8/v9 — the schedule taking effect is verifiable)

- `calib_n_drawn_total` = **1760** (11/step × 160 steps; v9 was 1280 = 8/step).
- `calib_n_drawn_per_step` = **11.0** (v9 8.0).
- `calib_n_drawn_by_tag` = the 8 tags at draws/step × 160: A-corr 320, goal_line_retention 160, goal_line_cont 160, old_post_opening_retention 160, old_post_opening_cont 320, red_predrop_root_value 160, red_predrop_cont 320, red_predrop_severe 160.
- `n_teacher_retention_drawn` ≈ **320** (goal_line_retention + old_post_opening_retention, the 2 policy-CE rows/step × 160) — **was 0 in v8/v9**. This nonzero count is the direct evidence the B/C policy-CE root rows are training.
- `calib_policy_ce_avg_iter` > 0 and `calib_policy_kl_est_avg_iter` > 0 — **were 0.0 in v8/v9** (no policy rows drawn).
- `train_value_head_and_final_block=True`, `unfrozen_block_index=5`, `freeze_batchnorm_stats=True`.

(Step count 160 is this config's value from the v7/v9 run; totals scale as draws/step × steps.)

## Operator run (USER's)

1. **Train** — v9 command, new checkpoint dir, v10 schedule:
```
--checkpoint-dir checkpoints/alphazero-v10-final-block-root-plus-cont-v7-manifest-from-calib020-0001 \
--post-opening-calibration-manifest logs/eval/targeted_calibration_v7_severe_d_root_correction_from_calib020_0001.csv \
--post-opening-calibration-tag-schedule black_predrop_correction=2,goal_line_retention=1,goal_line_continuation_retention=1,old_post_opening_retention=1,old_post_opening_continuation_retention=2,red_predrop_root_value_retention=1,red_predrop_continuation_retention=2,red_predrop_severe_root_correction=1 \
--post-opening-calibration-weight 0.01 \
--freeze-batchnorm-stats --train-value-head-and-final-block
```
2. **Verify telemetry** matches the section above (especially `n_teacher_retention_drawn ≈ 320` and `calib_policy_ce > 0` — proof the B/C root policy-CE rows engaged).
3. **Tensor-diff acceptance** (unchanged v9 verifier), must exit 0:
```
.venv/bin/python -m scripts.GPU.alphazero.verify_value_head_and_final_block_checkpoint \
  --base checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --candidate checkpoints/alphazero-v10-final-block-root-plus-cont-v7-manifest-from-calib020-0001/model_iter_0001.safetensors
```
4. **Gates A/B/C/D** vs `calib020_0001`, `OUT=logs/eval/v10_final_block_root_plus_cont_v7_manifest_from_calib020_0001_gates_400s`. Thresholds unchanged.

## Decision matrix (from user)

- **A pass + B/C/D pass** → candidate earns a promotion match.
- **A pass + D still fails** → partial-unfreeze is likely dead for this line.
- **B/C/D improve but A fails** → root preservation is too dampening (competes with the A correction); no promotion.
- **B/C/D still break** → final-block movement cannot be made safe with the existing retention rows.

Treat v10 as the **last partial-unfreeze attempt** unless it produces a near-miss.

### Interpreting a v10 failure (narrow, not global)

If v10 fails, do **not** conclude "root preservation can never work." The correct, narrow reading is: **the existing v7 root-retention rows — B/C root with root-visit policy CE, D root value-only — were not enough under final-block training.** That leaves a distinct, still-open v11 objective space that v10 does not test: **value-only B/C root clones** (new `goal_line_root_value_retention` / `old_post_opening_root_value_retention` rows, no policy CE) or **explicit delta-to-BASE preservation** (a different objective, not just anchor rows). Those would each require new extraction / a new objective, so they are out of scope for v10 (which stays strictly config-only) and belong to a future v11 if warranted.

## Not doing (explicit)

- No v9b last-2 blocks. No A draw pressure = 3 with v9. No raw-preservation-only-and-expect-D-to-pass (D already shows root-raw preservation is not enough on its own).
- No code, no new manifest, no new extraction, no new verifier. If v10 needs *new* rows (e.g., value-only B/C root), that is a separate, non-cheap experiment (v11), not this one.
