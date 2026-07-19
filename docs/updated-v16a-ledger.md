# Targeted Value Calibration — Experiment Ledger

**Created:** 2026-06-26 · **Status:** active · **Scope:** the value-head calibration line of work (fix the black pre-drop overvalue without breaking the guardrail families).

A durable, append-only record of every value-calibration experiment: what changed, how it scored on the four acceptance gates, what we learned, and what **not** to retry. **Read this before proposing any new calibration knob** — if the change is on the [do-not-repeat](#do-not-repeat-prevents-going-in-circles) list (or another sweep of a knob we've already swept), the answer is probably "no, we already saw the tradeoff."

> **Key conclusion (updated 2026-07-10, post v16a held-out validation):** The A black-pre-drop calibration target is explained as a **400-sim search artifact**, not a stable value-head defect. v14 closed the adapter cleanup line; v15 showed the 400-sim backup came from broad depth-2 frontier optimism; and the budget/trajectory checks showed BASE A collapses with more search (**400/1600/6400 mean +0.2570 → +0.0626 → −0.0451**, gate-over **50.0% → 30.0% → 10.0%**, severe **43.3% → 6.7% → 3.3%**), with the apparent predrop “drop” mostly a selected shallow-search bump (**6400−400 = −0.573 at predrop, −0.001 at drop**). Therefore **value calibration against A remains unjustified**: no v15 Phase 1, no frontier hard-value correction, and no further adapter/projection/schedule cleanup. v16 then falsified c_puct and showed that negative FPU directly reaches the first-touch reply-scanning mechanism on the selected A set: `fpu_value=−0.20` moved mean **+0.2570 → −0.0344** and opponent replies **134.7 → 24.5**. However, the frozen v16a game-held-out test **rejected absolute `fpu_value=−0.20` as a general 400-sim setting**. Across 324 held-out positions it caused **15 new collapsed roots (4.63%)**, and the preregistered stratum reject gate fired in late play: **13/84 = 15.48%** new collapses, including **6/42 = 14.29% late-red** and **7/42 = 16.67% late-black**. Effective children fell **107.58 → 70.92** (−36.66; about −34.1%), top-move flips reached **27.16%**, and late collapsed roots rose **17/84 → 28/84**, despite small central value movement (mean mover delta **+0.0028**, median absolute **0.0180**, p95 absolute **0.2822**). The selected-A result remains valid **mechanistic evidence**, but the fixed absolute candidate does not generalize safely. **Do not run B/C/D or a strength match with `−0.20`.** Next: a read-only postmortem of the 15 new-collapse cases, followed by a new search candidate designed on discovery data only—likely adaptive or parent-relative FPU. The v16a held-out manifest is consumed and must not be used to tune that replacement. The decisive success benchmark remains an equal-checkpoint, equal-400-sim, balanced-color, statistically significant head-to-head strength gain after collateral and guardrail checks pass.

## Before proposing a new calibration experiment

Run this check first (it encodes the [do-not-repeat](#do-not-repeat-prevents-going-in-circles) findings):

1. Check whether it is **only** a global-weight, retention-weight, or schedule-ratio sweep.
2. Check whether it uses **scalar-MSE retention only**.
3. Check whether it requires a **promotion match before A/B/C/D pass**.
4. **If yes to any of the above, stop and justify** why this is *not* repeating a rejected path — in writing, against the [experiment ledger](#experiment-ledger) row that already failed it.

## How to read this

The work targets one known weakness — **A, black pre-drop overvalue** — while holding three fragile "guardrail" families steady: **B goal-line**, **C old broad post-opening**, **D red pre-drop**. Each experiment is scored at 400-sim probes against the current best's baselines.

Per family we track **mean** (mean black-perspective value), **over** (% of positions overvalued), and **severe** (% severely overvalued). On the overvalued families lower is better; the gate pass bars are below.

### The four gates (pass bars vs current best `calib020_0001`)

| Gate | Family | Baseline (current best) | **Pass criteria** |
|---|---|---|---|
| **A** | black pre-drop (frozen-30, held out) | over 50.0% / severe 43.3% / mean +0.257 | mean ≤ 0.0 **and** severe materially below 43.3% |
| **B** | goal-line | over 5.6% / severe 0.0% | severe 0.0% **and** over ≤ 11.1% |
| **C** | old broad post-opening | over 33.3% / severe 13.3% / mean +0.099 | severe ≤ 13.3% **and** over ≤ 33.3% **and** mean ≤ +0.099 |
| **D** | red pre-drop | over 13.3% / severe 0.0% / mean −0.188 | severe = 0.0% **and** mean ≤ 0.0 |

**Promotion rule:** a checkpoint earns a **promotion match** (vs current best) only **after all four gates pass**. No branch below has earned one.

## Current best

**`calib020_0001`** — broad post-opening calibration from `0409`, selected early.

- Gate baselines (its own): A mean +0.257 / over 50.0% / severe 43.3% · B over 5.6% / severe 0.0% · C mean +0.099 / over 33.3% / severe 13.3% · D mean −0.188 / over 13.3% / severe 0.0%.
- **Match:** beat `0379` by **~+80 Elo**.
- **Decision: KEEP.** Strong overall, but a real **black pre-drop (A) weakness** — the thing every branch below tries to fix without breaking B/C/D.


### Implementation finding — BatchNorm calibration confound (2026-06-30)

During v4 gate-0 validation, train-mode BatchNorm was found to make calibration forwards batch-dependent. The v4 manifest initially failed real-checkpoint self-distillation until teacher caching and the teacher-retention training forward were aligned to eval-mode BatchNorm using frozen base running stats. Prior scalar-retention results remain valid for the implementation used, but BatchNorm batch-dependence is now a known confound that may have affected B/C/D retention behavior.

v4 and `v3-frozenBN-control` were both run with `--freeze-batchnorm-stats`. The control result shows BN freezing is required for clean calibration mechanics, but it is **not** sufficient to preserve B/C/D: v3 still passed A while failing B/C/D under frozen BN.

## Experiment ledger

| Experiment | Main change (knobs) | A — black pre-drop | B — goal-line | C — old post-opening | D — red pre-drop | Match | Decision / lesson |
|---|---|---|---|---|---|---|---|
| **calib010** — black-predrop v1 | Train **only** black-predrop correction, weight 0.01 | improved: over ~16–23%, severe ~6.7–10% | borderline / regressed | improved | regressed | Lost badly, **~−95 Elo** | **Reject.** Target fixed, broad play damaged. |
| **v2** — mixed pool | Correction + retention rows; retention_weight **0.5**, global **0.01**, **uniform** sampling | fixed A strongly | **fail** | **fail** | **hard fail** | no match | **Reject.** Correction worked, retention too weak. |
| **v2b** — mixed pool | v2 but retention_weight **2.0**, global 0.01 | fail/borderline @400: mean +0.038, severe 16.7% | **fail:** severe 5.6% | borderline fail: mean +0.109 | **fail:** severe 3.3% | no match | **Reject.** Stronger retention helped some; tradeoff remained. |
| **v3** — tag-stratified | Schedule **2:1:2:1**, retention_weight **1.0**, global 0.01 | **pass:** mean −0.047, severe 10.0% | **pass:** severe 0.0%, over 11.1% | **fail:** mean +0.180, over 40.0%, severe 23.3% | **fail:** severe 10.0% | no match | **Reject.** Stratification fixed mechanics + A/B, but C/D drifted. |
| **v3b** — tag-stratified, lower weight | Same schedule, global **0.005** | weak / fail-ish: mean −0.030, severe 20.0% | **fail:** severe 11.1% | **fail:** mean +0.113, severe 20.0% | **hard fail:** severe 23.3% | no match | **Reject.** Lower scalar weight didn't solve drift. **Stop scalar sweeps.** |
| **v4** — teacher-retention | raw-NN teacher value-MSE + teacher policy CE/KL on retention rows; global 0.01 / value 1.0 / policy 0.25; schedule 2:1:2:1; 1 iter (= v3); freeze_batchnorm_stats=true | **pass:** mean −0.305, over 13.3%, severe 6.7% | **fail:** over 16.7%, severe 11.1% | **fail:** mean +0.029, over 36.7%, severe 23.3% | **fail:** mean −0.038, over 36.7%, severe 16.7% | no match | **Reject.** Teacher-retention preserved clean Gate-0 self-distillation and fixed A, but B/C/D still drifted. No promotion. |
| **v3-frozenBN-control** — scalar-retention BN control | Same as v3 scalar per-row target setup; schedule 2:1:2:1; global 0.01; 1 iter; **freeze_batchnorm_stats=true** | **pass:** mean −0.106, over 20.0%, severe 13.3% | **fail:** over 16.7%, severe 5.6% | **fail:** mean +0.137, over 40.0%, severe 26.7% | **fail:** mean +0.013, over 40.0%, severe 16.7% | no match | **Reject.** Frozen-BN control shows v3 guardrail failure was not primarily a train-mode BatchNorm artifact. Scalar retention still damages B/C/D. |
| **v5** — MCTS-root-visit policy retention | raw teacher value anchor + BASE 400-sim root-visit policy CE on retention rows; global 0.01 / value 1.0 / policy-CE 0.25; schedule 2:1:2:1; freeze_batchnorm_stats=true | pass-ish / improved: mean −0.174, over 20.0%, severe 20.0% | **fail:** mean −0.288, over 16.7%, severe 5.6% | **fail:** mean +0.074, over 40.0%, severe 30.0% (mean passed) | **hard fail:** mean +0.046, over 40.0%, severe 36.7% | no match | **Reject.** Position-level root-visit anchors did not preserve B/C/D after A correction. Diagnose anchor-hold before any next design. |
| **v6** — searched-continuation retention | v5 source + BASE searched continuation/PV rows under B/C/D roots; value-only continuation rows; A hard correction unchanged; schedule 2:1:2:2; freeze_batchnorm_stats=true | **pass / improved:** mean −0.110, over 20.0%, severe 10.0% | **fail:** mean −0.321, over 16.7%, severe 0.0% | **fail:** mean +0.003, over 30.0%, severe 20.0% | **hard fail:** mean +0.150, over 53.3%, severe 30.0% | no match | **Reject.** Continuation rows existed for all failed roots, but D root raw values remained severe or drifted upward. Coverage was not the main problem. |
| **v6b** — D root + continuation hybrid | v6 manifest but schedule also drew `red_predrop_retention=1`, reintroducing D root teacher policy/root-visit CE; schedule 2:1:2:1:2; freeze_batchnorm_stats=true | **pass:** mean −0.308, over 3.3%, severe 3.3% | **fail:** mean −0.240, over 16.7%, severe 5.6% | **hard fail:** mean +0.118, over 56.7%, severe 23.3% | **hard fail:** mean −0.009, over 40.0%, severe 26.7% | no match | **Reject.** D root policy retention helped D only slightly but broke B/C. Root policy CE/KL is toxic as a mixed guardrail strategy. |
| **v6c** — D root value-only + continuation | v6c manifest added 30 depth-0 `red_predrop_root_value_retention` rows; D root rows value-only, no policy/root visits; schedule 2:1:2:1:2; freeze_batchnorm_stats=true | **fail / improved:** mean +0.006, over 30.0%, severe 23.3% | **fail:** mean −0.195, over 16.7%, severe 11.1% | **fail:** mean −0.007, over 36.7%, severe 16.7% | **fail:** mean +0.032, over 33.3%, severe 13.3% | no match | **Reject.** Value-only D root anchoring is less toxic than policy retention but still interferes with B/C and leaves D failing. |
| **v7** — sparse severe-D hard correction | Manifest-only: appended 8 `red_predrop_severe_root_correction` hard-value rows selected by BASE raw severe-overvalue (`target_black_value=-0.35`); v7 schedule 2:1:2:1:2; full-network training; freeze_batchnorm_stats=true | **pass / improved:** mean −0.065, over 26.7%, severe 13.3% | **fail:** mean −0.290, over 16.7%, severe 5.6% | **fail:** mean +0.002, over 30.0%, severe 20.0% | **hard fail:** mean +0.034, over 40.0%, severe 23.3% | no match | **Reject.** Sparse severe-D hard correction did not beat v6c and still broke B/C. Drift map showed nonlocal value-surface movement, not just wrong row selection. |
| **v8** — value-head-only on v7 manifest | Same v7 manifest/schedule, but `--train-value-head-only` skips encoder+policy updates; verifier proved all non-`value_head.*` tensors byte-identical and only 4 value-head tensors changed; freeze_batchnorm_stats=true | **fail / improved:** mean +0.068, over 33.3%, severe 20.0% | **pass:** mean −0.276, over 11.1%, severe 0.0% | **pass:** mean +0.024, over 23.3%, severe 10.0% | **pass:** mean −0.056, over 36.7%, severe 0.0% | no match | **Reject for promotion, but key positive result.** B/C/D passed with value-head-only; A undercorrected. Strong evidence full-network/trunk drift caused earlier guardrail failures. Next: v8b A draw pressure. |
| **v8b** — value-head-only, A draw pressure 3 | Same v7 manifest and value-head-only mechanics as v8, but A schedule raised `black_predrop_correction=2→3`; verifier passed with only 4 value-head tensors changed; freeze_batchnorm_stats=true | **fail / worse than v8:** mean +0.102, over 33.3%, severe 26.7% | **pass:** mean −0.286, over 5.6%, severe 0.0% | **pass / degraded vs v8:** mean +0.086, over 33.3%, severe 13.3% | **pass:** mean −0.096, over 26.7%, severe 0.0% | no match | **Reject.** Higher A draw pressure did not move A and made A/C worse. Raw-A diagnostic showed value-head-only barely moved the A family; the constraint is representational, not sampling. Do not run A=4/A=5 as the next step. |
| **v9** — value head + final residual block | Same v7 manifest and v8 schedule; `--train-value-head-and-final-block` updated only `value_head.*` plus final residual block `encoder.blocks.5` trainable tensors; strict verifier passed and all frozen tensors / BN running stats were byte-identical | **pass:** mean −0.089, over 30.0%, severe 16.7% | **fail:** mean −0.238, over 22.2%, severe 11.1% | **fail:** mean +0.067, over 46.7%, severe 30.0% | **fail:** mean −0.115, over 26.7%, severe 20.0% | no match | **Reject.** Final block gave enough flexibility to fix A, but immediately reintroduced B/C/D guardrail drift. Do not run v9b last-2 blocks as the next step; broader partial unfreeze is expected to worsen this failure mode. |
| **v10** — final block + root/continuation schedule | Config-only from v9: same v7 manifest, same `--train-value-head-and-final-block`, but enabled dormant B/C root-retention tags plus D root-value retention; schedule `2:1:1:1:2:1:2:1` (11 draws/step); telemetry clean (`calib_n_drawn_total=1760`, `n_teacher_retention_drawn=320`, policy CE/KL active) | **pass / near margin:** mean −0.004, over 20.0%, severe 16.7% | **fail / near-pass:** mean −0.195, over 11.1%, severe 5.6% | **pass:** mean +0.016, over 23.3%, severe 10.0% | **fail / near-pass:** mean −0.067, over 26.7%, severe 3.3% | no match | **Reject, but best near-pass.** Root+continuation schedule recovered C and preserved the A fix. Remaining blockers were narrow: B one severe row (`game_000015_ply_19` +0.6435) and D one barely-severe row (`red_loss_game_000752_predrop_ply_70_drop_72` +0.5003). |
| **v10b** — stronger B/D schedule | Config-only from v10, increased `goal_line_retention`, `red_predrop_root_value_retention`, and `red_predrop_severe_root_correction` from 1→2 (14 draws/step); telemetry/verifier clean (`calib_n_drawn_total=2240`, `n_teacher_retention_drawn=480`) | **fail / regressed:** mean +0.095, over 36.7%, severe 30.0% | **pass:** mean −0.310, over 11.1%, severe 0.0% | **fail / regressed:** mean +0.135, over 33.3%, severe 23.3% | **fail / regressed:** mean +0.043, over 23.3%, severe 13.3% | no match | **Reject.** Stronger B/D pressure fixed B but destabilized A/C/D. v10b caused many previously-safe rows to jump upward with high top1 concentration; broad schedule-count pressure is exhausted. If continuing, branch from v10 with surgical value-only rows, not from v10b. |
| **v11** — B surgical value-only root clones | Manifest-only from v10/v7: appended 2 `goal_line_root_value_retention` depth-0 value-only clones for v10 B blockers (`game_000015_ply_19`, `game_000327_ply_63`), replaced B root-policy CE with value-only B root pressure; same final-block update surface; telemetry/verifier clean (`calib_n_drawn_total=1920`, `n_teacher_retention_drawn=160`) | **pass:** mean −0.039, over 30.0%, severe 13.3% | **fail / worse than v10:** mean −0.060, over 22.2%, severe 16.7% | **fail:** mean +0.058, over 23.3%, severe 20.0% | **fail:** mean −0.109, over 30.0%, severe 6.7% | no match | **Reject.** B value-only root clones did not isolate/fix B; B worsened and C/D failed. The v10 B issue was not simply B root-policy CE or missing value-only root preservation. Close the v10/v11 row/schedule branch; next credible path requires a new constraint/objective, not more manifest tweaks. |
| **v12** — asymmetric one-sided guardrail hinge | New objective: `asymmetric_guardrail_retention` one-sided black-perspective hinge on B/C/D root guardrails; no policy CE; A hard correction unchanged; final-block update surface; manifest 136 rows; schedule `2:1:2:2`; telemetry/verifier clean (`calib_n_drawn_total=1120`, hinge active, policy CE/KL 0) | **fail by mean only / near-pass:** mean +0.005, over 20.0%, severe 13.3% | **pass:** mean −0.214, over 5.6%, severe 0.0% | **fail by severe:** mean +0.057, over 23.3%, severe 16.7% | **fail by severe:** mean −0.088, over 23.3%, severe 3.3% | no match | **Reject, but objective is promising.** B was fixed cleanly and the hinge path engaged as intended. A missed by only +0.005 mean, but C still had a broad severe repeat-offender cluster and D had one severe plus many high non-severe over rows. Root-only guardrails are insufficient for C/D; next branch is v12b continuation guardrails, not gradient projection yet. |
| **v12b** — continuation guardrail rows | Same v12 one-sided hinge objective, but loader extension allows `asymmetric_guardrail_retention` rows with `extra_moves_json` to reconstruct searched-continuation states; new builder emitted B/C/D root guardrails plus C/D continuation guardrails; no trainer.py change; manifest 353 rows; schedule `2:1:1:2:1:2`; telemetry/verifier clean (`calib_n_drawn_total=1440`, hinge active, policy CE/KL 0) | **pass / strong:** mean −0.137, over 30.0%, severe 13.3% | **fail by severe:** mean −0.302, over 5.6%, severe 5.6% | **fail by severe:** mean +0.028, over 33.3%, severe 23.3% | **hard fail:** mean −0.093, over 40.0%, severe 16.7% | no match | **Reject.** Continuation guardrails did not solve C/D and regressed B/D relative to v12. C failures were stable repeat offenders (`game_000505`, `000565`, `000619`, `000433`, `000065`, `000309`); D showed broad/diffuse severe drift (`000176`, `000278`, `000780`, `000456`, `000438`). Coverage is no longer the likely missing piece; next branch is v13 gradient-conflict handling/projection, not another schedule, margin, or row-coverage tweak. |
| **v13** — asymmetric gradient-conflict projection | Same v12b manifest/schedule/objective, but split A-correction and guardrail-hinge gradients on the applied surface and project A away from guardrail when `dot(g_A,g_G)<0`; `--freeze-batchnorm-stats --train-value-head-and-final-block --post-opening-calibration-gradient-projection`; telemetry-fixed rerun showed projection engaged (`conflict_rate=28.5%`) | **pass:** mean −0.117, over 23.3%, severe 20.0% | **pass:** mean −0.343, over 5.6%, severe 0.0% | **fail by severe:** mean −0.083, over 26.7%, severe 16.7% | **fail:** mean −0.151, over 36.7%, severe 13.3% | no match | **Reject, but directionally positive.** Projection engaged and fixed A/B while improving C/D shape versus v12b, but C/D severe remained. Initial v13 run had projection telemetry dropped from flattened JSON; telemetry fix made the run interpretable. |
| **v13b** — projection + lower guardrail margin | Same v13 projection mechanics and v12b schedule, but `--guardrail-margin 0.05` to activate guardrails earlier; projection activity rose (`conflict_rate=41.6%`, `active_frac=28.6%`) | **pass but weakened:** mean −0.017, over 36.7%, severe 20.0% | **pass:** mean −0.370, over 11.1%, severe 0.0% | **fail by severe:** mean −0.063, over 26.7%, severe 16.7% | **fail / worse severe:** mean −0.203, over 23.3%, severe 16.7% | no match | **Reject.** Lowering margin globally made more guardrail rows active, but did not solve C/D and weakened A. Margin-tightening is exhausted; do not run 0.025 or broader hinge activation. |
| **v13c** — projection-strength scalar 2.0 | Same v13 projection mechanics, margin restored to 0.10, added `--post-opening-calibration-projection-strength 2.0`; projection strength folds into effective projection weight only when conflict is detected; telemetry clean (`strength=2.0`, `conflict_rate=36.8%`, `removed_norm_avg=0.1292`) | **pass:** mean −0.052, over 13.3%, severe 10.0% | **pass:** mean −0.243, over 0.0%, severe 0.0% | **pass:** mean −0.076, over 6.7%, severe 6.7% | **fail by one severe row:** mean −0.073, over 16.7%, severe 3.3% | no match | **Reject by strict gate, best projection result.** A/B/C all passed; D had one barely-severe repeat outlier (`red_loss_game_000728_predrop_ply_48_drop_50` +0.5441). No promotion because D requires severe 0.0%. v13d is a tightly scoped arg-only D cleanup, not blind strength/margin tuning. |
| **v13d** — v13c + red root guardrail draw 2 | Same v13c mechanics (`projection_strength=2.0`, margin 0.10), same v12b manifest, but schedule changes only `red_predrop_guardrail_retention=1→2` while keeping `red_predrop_continuation_guardrail_retention=2`; telemetry/verifier clean (`calib_n_drawn_total=1600`, `conflict_rate=34.6%`, `removed_norm_avg=0.1802`) | **pass but weakened:** mean −0.083, over 26.7%, severe 23.3% | **fail:** mean −0.265, over 16.7%, severe 5.6% | **fail:** mean −0.013, over 33.3%, severe 20.0% | **hard fail:** mean −0.113, over 33.3%, severe 20.0% | no match | **Reject.** The single D-root cleanup did not clear D; it damaged B/C/D and weakened A relative to v13c. Close the v13 projection/cleanup line. Do not run more root draw pressure, margin tweaks, or projection-strength sweeps without a new design. |
| **v14** — gated value-adapter, projection OFF | New value-only adapter surface (`value_head.*` + `value_adapter.*`, scalar gate, bottleneck 32), encoder/policy/final block/BN frozen; same v12b manifest/schedule/objective, `guardrail_margin=0.10`, projection OFF; telemetry/verifier clean (`value_adapter_gate=0.003018`, `value_adapter_grad_norm=0.001381`, `calib_n_drawn_total=1440`) | **fail / improved:** mean +0.064, over 26.7%, severe 20.0% | **pass:** mean −0.272, over 5.6%, severe 0.0% | **pass:** mean +0.063, over 30.0%, severe 6.7% | **pass but degraded margin:** mean −0.079, over 23.3%, severe 0.0% | no match | **Reject.** Adapter surface is not a no-op: A moved substantially and B/C/D formally held, but A still missed mean ≤ 0.0 and D moved toward black within the pass band. This is not underfit; do not run width 64 next. Next branch: v14b projection ON over `value_head.*` + `value_adapter.*`. |
| **v14b** — value-adapter + projection strength 1.0 | Same v14 adapter surface/objective/schedule, but `--post-opening-calibration-gradient-projection` enabled over `value_head.*` + `value_adapter.*`; bottleneck 32; projection strength default 1.0; verifier passed (`value_head.*` + `value_adapter.*` only; final block byte-identical); telemetry clean after label fix (`conflict_steps=51`, `conflict_rate=39.8%`, `removed_norm_avg=0.0727`, `value_adapter_gate=0.001667`, `value_adapter_grad_norm=0.001518`, `calib_n_drawn_total=1440`) | **fail / near-pass:** mean +0.026, over 26.7%, severe 16.7% | **pass:** mean −0.254, over 11.1%, severe 0.0% | **pass:** mean +0.044, over 23.3%, severe 6.7% | **pass:** mean −0.047, over 23.3%, severe 0.0% | no match | **Reject / best adapter result so far.** Projection improved A while B/C/D stayed inside formal gates, but A still missed mean ≤ 0.0. No promotion. Since projection helped and guardrails held, the one justified follow-up is v14c: same setup with projection strength 2.0; do not redesign or widen before v14c gates. |
| **v14c** — value-adapter + projection strength 2.0 | Same v14b adapter projection setup, but added `--post-opening-calibration-projection-strength 2.0`; label fix confirmed `calib_projection_scope=value_head_and_value_adapter`; verifier passed (`value_head.*` + `value_adapter.*` only; final block byte-identical); telemetry clean (`strength=2.0`, `conflict_steps=49`, `conflict_rate=44.5%`, `removed_norm_avg=0.1329`, `value_adapter_gate=-0.001048`, `value_adapter_grad_norm=0.001371`, `calib_n_drawn_total=1440`) | **fail / regressed vs v14b:** mean +0.060, over 30.0%, severe 20.0% | **pass:** mean −0.261, over 5.6%, severe 0.0% | **pass but at severe cap:** mean +0.052, over 30.0%, severe 13.3% | **pass:** mean −0.056, over 26.7%, severe 0.0% | no match | **Reject.** Strength 2.0 did not push A through; it regressed A versus v14b and narrowed C/D margins. Do not run strength 3.0. Best adapter checkpoint remains v14b. One final easy-lift cleanup is v14d: revert to strength 1.0 and increase only A draw pressure (`black_predrop_correction=2→3`). |
| **v14d** — value-adapter + projection strength 1.0 + A draw 3 | Same v14b adapter projection setup (`projection_strength=1.0`, bottleneck 32, same v12b objective/surface), but increased only `black_predrop_correction=2→3`; all guardrail schedules unchanged; telemetry/verifier clean (`strength=1.0`, `conflict_steps=72`, `conflict_rate=54.5%`, `removed_norm_avg=0.0742`, `value_adapter_gate=-0.000585`, `value_adapter_grad_norm=0.001501`, `calib_n_drawn_total=1600`, A draws 480) | **fail:** mean +0.051, over 33.3%, severe 16.7% | **pass:** mean −0.299, over 5.6%, severe 0.0% | **pass:** mean +0.030, over 30.0%, severe 10.0% | **fail:** mean −0.049, over 40.0%, severe 3.3% | no match | **Reject.** The narrow A-pressure cleanup did not push A through and broke D severe. This closes the argument-only adapter cleanup line: do not run A=4/A=5, strength 3.0, width 64, or another schedule/projection tweak without a new written design. |
| **v15 diagnostics** — A searched-continuation concentration + selected-branch subtree walk | Read-only diagnostics only. Phase 0 walked immediate children under the 30 A roots; Phase 0.5 re-ran deterministic BASE MCTS on the 17 positive A roots, selected positive branches to 90%/max-3, and walked every expanded descendant with PV annotation. No manifest, no training, no checkpoint. | **mechanism clarified:** Phase 0: all 17 positive roots concentrated at selected-child level; Phase 0.5: depth-2 frontier produced broad raw optimism, not child/PV path | n/a | n/a | n/a | no match | **Reject v15 Phase 1 / close v15.** Do not build depth-1 child rows, shallow PV rows, or semi-PV continuation rows. The raw optimism is broad: 5,837 nodes walked, 5,745 raw-scored, median 196 depth-2 nodes/root needed for 70% positive raw mass, median PV share of positive raw mass 0.335%. Next branch must be v16 frontier/tree-level correction or search/prior intervention. |
| **post-v15 budget + trajectory diagnostics** — A search artifact confirmation | Read-only probe/trajectory checks on BASE `calib020_0001`; no checkpoint, no manifest, no training. Re-ran A at 400/1600/6400 sims and compared five high-A predrop→drop trajectories at 400 vs 6400. | **explained artifact:** A mean +0.2570 → +0.0626 → −0.0451 as sims increase; gate-over 50.0% → 30.0% → 10.0%; severe 43.3% → 6.7% → 3.3%. Predrop inflation 6400−400 averaged −0.573; drop-ply inflation −0.001. | n/a | n/a | n/a | no match | **Close A value-calibration line / reject v16 frontier correction before build.** The original “sharp value drop” was mostly a selected 400-sim predrop bump/winner's curse, not a stable value-head overvalue. Next branch is **v16 search reliability** (c_puct first; FPU only if needed), not value calibration. |
| **v16 c_puct falsification diagnostic** — c_puct is not the fix | Read-only search-reliability diagnostic on BASE A probe rows; no checkpoint, no manifest, no training. Integrity check passed: c_puct=1.5 reproduced Phase 0 per-case root values within 1e-6, then only `MCTSConfig.c_puct` varied. | **c_puct worsens A:** c_puct 1.5→0.25 raised mean +0.2570→+0.3778, gate-over 50.0%→60.0%, severe 43.3%→50.0%. Top-child visit share rose 0.474→0.642; top-child visited children rose 134.7→232.9; corr(top_child_n_visited_children, root value)=+0.943. | n/a | n/a | n/a | no match | **Reject c_puct as a fix / continue v16 only via FPU diagnostic.** Lower c_puct funnels visits into the selected root child and increases one-visit opponent-reply frontier scanning. Raising c_puct must not be used to pass A because it would lower the metric by spreading visits onto inferior root moves, not by repairing the frontier. Next search-code lever is an opt-in FPU field with default 0.0 preserving current behavior exactly. |
| **v16 FPU selected-A diagnostic** — first knob that reaches the measured mechanism | Search-code diagnostic on BASE A probe rows; no checkpoint, no manifest, no training. Added opt-in `MCTSConfig.fpu_value` with default 0.0, routing unvisited-child q through the field. Integrity check passed: fpu=0.0 reproduced Phase 0 within 1e-6 and the full suite passed, proving default behavior is byte-identical. | **FPU works on selected A:** fpu 0.0→−0.20 moved mean +0.2570→−0.0344, gate-over 50.0%→6.7%, severe 43.3%→6.7%, and opponent replies scanned 134.7→24.5. fpu −0.35/−0.50 reached severe 0.0% but with much narrower search. `−0.20` was frozen as the single held-out candidate because it was closest to the 6400 reference (mean −0.0451) while preserving more breadth than −0.35/−0.50. | n/a | n/a | n/a | no match | **Promising mechanism / not adoption.** FPU directly suppresses first-touch opponent-reply scanning, the mechanism c_puct could not reach. But the sample is biased by A selection, and FPU gets the value by narrowing search rather than searching deeper. The next rung was the frozen v16a comparison `0.0` vs `−0.20` on a game-held-out neutral sample; see the following row. |
| **v16a held-out FPU validation** — frozen `0.0` vs `−0.20` collateral screen | Search-code diagnostic only on a deterministic stratified, game-held-out, non-selected manifest: 324 positions from 252 games, buckets 40 opening / 100 early-mid / 100 midgame / 84 late, exactly 162 red / 162 black, and zero games shared with A discovery. The 19 winner-null 280-ply state-cap marathons were retained as stressed valid samples. No checkpoint or training changes. | **Not an A gate. Neutral result:** mean mover delta +0.0028, median absolute 0.0180, p95 absolute 0.2822; top-move flips 27.16%; effective children 107.58→70.92 (−36.66, about −34.1%); top-share +0.0716; 15 new collapses / 2 resolved. | n/a | n/a | **Late collateral failure:** 13/84 = 15.48% new collapse; late-red 6/42 = 14.29%, late-black 7/42 = 16.67%; late collapsed roots 17/84→28/84. | no match | **REJECT absolute `fpu_value=−0.20` as a general 400-sim setting.** The preregistered reject rule (any n≥20 stratum with new-collapse rate ≥10%) fired independently for late overall and both late-side strata. Do not proceed to B/C/D or strength evaluation with this candidate. Selected-A success remains mechanistic evidence only. Next: read-only postmortem of the 15 new-collapse cases, then design a new adaptive/parent-relative candidate on discovery data only. The v16a held-out manifest is consumed and must not be used for tuning. |


| **v16 policy-mass successor — reservoir protocol v1** | Search-code successor on unchanged BASE `calib020_0001`: `FPU=Q_parent-r*sqrt(P_explored)`, with completed-visit policy mass. New phase-primary dev-corpus pipeline, immutable 4,800-game reservoir protocol, tuning/frozen whole-game split, matched controls, late-band floors, and end-to-end fingerprints. Frozen generation commit `fca9c0d`; reservoir seed range `[20270000,20274800)`. | **pending.** Selected-A is tuning-only and later must pass the reply-reduction/progress mechanism gate. | **pending guardrail.** | **pending guardrail.** | **pending guardrail.** | pending | **RUNNING (2026-07-16).** Protocol emitted successfully; exact 4,800-game `calib020_0001` vs `0379` replay reservoir is generating with board 24, 400 sims, workers 4, replay capture, and no top-up. No qualification, candidate, frozen-check, A/B/C/D, or strength result has been observed. Do not interpret or promote. |

*(The current best `calib020_0001` is the baseline row — see [Current best](#current-best).)*

## v16 policy-mass successor — reservoir protocol v1 (RUNNING 2026-07-16)

This is the first context-relative successor to rejected v16a. It is a search experiment on unchanged `calib020_0001`, not a new trained checkpoint. The rule under test is:

```text
P_explored = sum(prior(a)) over children with completed backed-up visits
FPU        = Q_parent - r * sqrt(P_explored)
```

Purpose: retain absolute `-0.20`'s ability to suppress selected-A first-touch opponent-reply scanning without reproducing its held-out low-prior collapse failure at near-even, high-branching, flat-policy roots.

Frozen production decisions:

- Generation commit: `fca9c0dc563e47274b71059749ab451fb74e47f1`.
- Checkpoint A / screen anchor: `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`, SHA1 `209cf2d4fd24a48553d259dd71b4954867b9473e`.
- Checkpoint B / reservoir opponent: `checkpoints/alphazero-v2-staged/model_iter_0379.safetensors`, SHA1 `8ad62ac432c35c6ea9b0630b8a2b8c572a0b03a1`.
- Exactly 4,800 games; base seed `20270000`; half-open seed range `[20270000,20274800)`; no top-up.
- Board 24; 400 simulations; MCTS eval batch 14; stall flush 48; opening-temperature selection for 20 plies (`1.0 -> 0.1`); max moves 280; workers 4; replay capture required.
- Selection seed `20260712`.
- Phase-primary final corpus: target/control `180/60`; tuning/frozen `160/80`; each phase gets target `30/15` and control `10/5`.
- Late-target floors: `b300_399 >= 12`, `b200_299 >= 12`.
- Proposal enumerator: side-opposed pair per cell, minimum 12-ply gap, maximum 2 proposals per cell/game; final selection remains globally at most 2 rows/game with whole-game split and side balance.
- New-collapse sub-gate stratum: `ply_bucket`; branching band remains recorded.
- Forbidden/consumed: selected-A manifest and frozen v16a neutral manifest.

Artifact root:

```text
logs/eval/fpu_v16_policy_mass_v2/
```

Machine-authoritative protocol:

```text
logs/eval/fpu_v16_policy_mass_v2/reservoir_v1/reservoir_protocol.json
```

Current observed status:

- Clean `main == origin/main` at `fca9c0d` was confirmed.
- `emit-protocol` succeeded.
- `emit-gen-command` reproduced the reviewed command exactly.
- Reservoir generation started 2026-07-16 and is expected to take approximately 39 hours at historical four-worker throughput.
- The match CLI is silent until completion and writes replay sidecars incrementally.
- **No scientific result exists yet.** Qualification, geometry, screen eligibility, coefficient safety, selected-A progress, frozen performance, B/C/D guardrails, and playing strength are all pending.

Frozen progression/stopping rule:

```text
reservoir qualification
  -> full persisted proposal screen
  -> deterministic 240-row select
  -> absolute_off/r0 tuning controls (r0 must qualify)
  -> frozen five-value grid; choose smallest safe passing r
  -> one isolated 80-position frozen check
  -> cross-matchup + fresh held-out collateral validation
  -> selected-A mechanism + B/C/D guardrails
  -> decisive same-checkpoint 400-sim strength match
```

If the reservoir faithfully matches the protocol but fails geometry, protocol v1 is retired; never append/top-up. If `r0` or every candidate fails, reject the formula family. No self-play adoption before the final strength match passes.

Full reproducible commands and artifact descriptions belong in `docs/post-game-analysis.md` section `v16 Context-Relative Policy-Mass FPU — Operator Runbook`.

## What got better vs worse

**Improved — A (black pre-drop):** targeted correction is **real**. The strongest A correction so far is **v4 teacher-retention**: mean **−0.305**, over **13.3%**, severe **6.7%** (from baseline mean +0.257 / over 50.0% / severe 43.3%). This is an A-only success, not a promotion candidate, because B/C/D failed.

**Worsened / unstable — C (old post-opening):** regresses under every v2/v3 approach. Crucially, **v3 and v3b share 5 severe C cases** — the same positions break regardless of the scalar weight:

- `game_000065_ply_021`
- `game_000309_ply_019`
- `game_000433_ply_029`
- `game_000505_ply_037`
- `game_000565_ply_033`

That overlap means **C is not random eval variance** — it's a stable fragile family that scalar calibration keeps damaging. It needs **direct retention of the current-best behavior** on those positions.

**Worsened / diffuse — D (red pre-drop):** v3/v3b share only **1** severe case:

- `red_loss_game_000728_predrop_ply_48_drop_50`

Low overlap ⇒ D is likely a **broader value-head drift** problem, not a handful of hard positions. A few hardcoded D rows probably won't fix it; it may need broader retention.

## Do-not-repeat (prevents going in circles)

1. **Uniform mixed-pool sampling only.** v2/v2b showed uniform sampling can't reliably separate correction from retention.
2. **Only increasing `retention_weight`.** v2b (2.0) helped some guardrails but weakened A and still failed. → no `retention_weight 3.0` sweep; we've already seen the tradeoff.
3. **Only lowering global calibration weight.** v3b (0.005) did not preserve guardrails and weakened correction. → **stop scalar weight sweeps.**
4. **Promotion matches before A/B/C/D all pass.** Every rejected branch failed gates clearly enough that a match would be wasted compute.
5. **More scalar-MSE-only rows as the main strategy.** The C/D failures show scalar row anchors aren't enough to hold the guardrails.
6. **Rerunning scalar-retention v3 with frozen BatchNorm as the fix.** The `v3-frozenBN-control` still passed A but failed B/C/D, with B/D worse than original v3. BN freezing is required for clean calibration mechanics, but it does not solve guardrail retention.
7. **Another v4 teacher-retention weight/schedule tweak before raw-NN candidate scoring.** v4 included the shared fragile C/D rows, but raw-NN teacher-retention still failed to preserve 400-sim MCTS gate behavior. Before changing teacher value/policy weights or schedule ratios, inspect whether v4 actually matched the raw teacher values on the shared C/D rows. *(Scoring done 2026-07-01 — see #8.)*
8. **Any further raw teacher-retention weight/schedule sweep.** The 2026-07-01 raw-NN focus-row diagnostic shows v4 **mostly matched the raw teacher values** on the shared C/D rows (e.g. `game_000369_ply_051` delta +0.1127 raw vs a severe MCTS gate) while the 400-sim gates still failed — the **objective**, not its weighting, is wrong for the gate. The next branch must target MCTS-root/root-behavior retention.
9. **"Root-value-only retention" as a new branch.** It has already been run: v2/v3 retention rows' `target_black_value` came from `probe_black_root_value` (`build_targeted_calibration_manifest.py:105,137`), i.e. BASE's own 400-sim MCTS root values — and failed B/C/D, including under frozen BN (v3F). Any v5+ proposal whose only value signal is the BASE root value is a v3 rerun. The new signal must be the root **visit distribution** (or deeper tree/path structure), not the root scalar.
10. **Any root-policy weight/schedule sweep, or a new retention design, before the v5 anchor-hold diagnosis.** *(DISCHARGED 2026-07-02 — the path diagnostic ran: anchors HELD, continuations drifted; see the v5 path-diagnostic entry.)* The rule's successor is #11.
11. **Any further root-position-level anchoring as the primary retention strategy.** The v5 path diagnostic proves the mechanism: v5 held its root anchors (dominant moves + visit shares) on the fragile C rows yet stayed severe, because the drift lives in the **searched continuation/child values** one-plus plies below the anchored roots (D top-child NN values +0.03→+0.80). Adding more root rows, sharper root targets, or heavier root weights cannot reach it. Retention designs must anchor **continuation/PV states** (or deeper tree structure), i.e. v6's shape.
12. **Another full-network v6/v7 row-engineering branch as the primary fix.** v6, v6b, v6c, and v7 all used cleaner/more targeted row designs and still failed at least one guardrail. The v7 drift map showed nonlocal value-surface movement even when the selected rows were sensible. Do not add more continuation/root/severe rows under full-network training before changing the training mechanics.
13. **Broad D root retention or sparse severe-D hard correction as a standalone fix.** v6c (30 D root value-only rows) and v7 (8 severe-D hard rows) both moved D in the right mean direction but still failed D and/or broke B/C. D row pressure alone is not enough under full-network training.
14. **Assuming value-head-only is a promotion just because B/C/D pass.** v8 proved value-head-only protects B/C/D, but A still failed. Value-head-only is the active training-mechanics hypothesis, not a promotion candidate until A/B/C/D all pass.
15. **Increasing A draw pressure under value-head-only as the next fix.** v8b raised A schedule mass from 2→3 and made A worse (mean +0.068 → +0.102, severe 20.0% → 26.7%) while raw-A output barely changed. Do not run A=4/A=5 before changing the mechanism.
16. **Broader partial-trunk unfreeze as the next move.** v9 unfreezing just the final residual block fixed A but broke B/C/D. Last-2/final-N unfreeze is expected to increase the same nonlocal guardrail drift unless paired with a new constraint/objective; do not run v9b last-2 as a simple extension.
17. **Another broad v10/v10b schedule-count sweep as the next move.** v10 was a near-pass, but v10b showed that increasing B/D pressure broadly fixes B at the cost of A/C/D. The knobs are coupled; do not keep sweeping tag counts from this family.
18. **Using v10b as the branch point.** v10b is worse than v10 on three gates. Do not build on v10b.
19. **B-only value-root clone surgery as the next fix.** v11 added value-only B root clones for the v10 B blockers and reduced B policy-CE exposure (`n_teacher_retention_drawn` 320→160), but B worsened and C/D failed. The v10 B blocker was not simply B root-policy CE or missing B value-only root preservation.
20. **Any further v10/v11 schedule or manifest-row variant as the next move.** v10 was the near-pass, v10b proved broad pressure is destabilizing, and v11 proved surgical B root-value cloning is insufficient. Existing root/continuation/root-value/severe-row levers are exhausted under final-block training. Do not run v11b, more B clones, more D pressure, or another tag-count variant as the next step.
21. **Treating root-only asymmetric guardrails as sufficient for C/D.** v12 proved the one-sided hinge objective is useful and fixed B, but C still had a broad severe repeat-offender cluster and D retained a severe row plus high non-severe over rows. Do not keep testing root-only guardrail variants as the next step; C/D need continuation guardrails or a stronger constraint.
22. **Jumping directly to gradient projection before testing continuation guardrails.** v12 was too close and too diagnostic: B passed, A nearly passed, and the remaining failures were concentrated in C/D families already known to require searched-continuation coverage. v13 gradient-conflict handling is reserved for after v12b if C/D still break despite continuation guardrails.

23. **Another v12b schedule/margin/row-coverage tweak as the next move.** v12b already applied the one-sided hinge to C/D searched-continuation states and still failed B/C/D, with C stable repeat offenders and D broad severe drift. Do not run v12c with heavier continuation weights, added B continuation guardrails, more C/D rows, or a margin sweep as the primary next branch. The next credible step is gradient-conflict handling/projection.
24. **Rerunning projection variants without flattened projection telemetry.** The first v13 run produced valid gates but dropped `calib_projection_*` from `model_iter_*.json`, making conflict-rate/removed-norm interpretation impossible. Any future projection branch must persist telemetry in both the nested sidecar and flattened per-iteration row before being used for decision-making.
25. **Lowering guardrail margin as the next projection fix.** v13b margin 0.05 increased `guardrail_active_frac` and projection conflict rate but weakened A and did not improve C/D severe. Do not run margin 0.025 or another global hinge-activation sweep.
26. **Promoting a near-pass with one D severe row.** v13c passed A/B/C and missed D by one barely-severe row, but D's pass bar is explicitly `severe=0.0%`. Do not change promotion rules after seeing a near-pass result. One tightly scoped cleanup run was acceptable; promotion still requires all gates.
27. **More v13 projection cleanup after v13d.** v13d was the tightly scoped cleanup run and it failed broadly: B/C/D all failed and D severe rose to 20.0%. Do not keep tuning `red_predrop_guardrail_retention`, projection strength, margin, or tag schedule inside the v13 family. The projection/cleanup line is closed unless a new written design changes the mechanism.
28. **Treating v14 as a promotion candidate or an underfit result.** v14 improved A substantially (mean +0.257→+0.064, severe 43.3%→20.0%) and passed B/C/D by formal gates, but it still failed A by mean. Do not promote it, and do not classify it as no-move/underfit.
29. **Running width 64 immediately after v14.** Width is the underfit lever, but v14 did move A. The blocker was not lack of movement; it was that A remained positive while D's mean margin degraded. Projection over the adapter surface was the right next branch, not wider capacity.
30. **Treating v14b as promotable because B/C/D passed.** v14b is the best adapter result so far and B/C/D passed, but A still missed the formal mean gate (+0.026 > 0.0). No promotion match until A/B/C/D all pass.
31. **Redesigning or widening immediately after v14b before the strength-2 test.** v14b showed projection helps A and does not break formal guardrails. The next single-knob follow-up was v14c (`projection_strength=2.0`), not width 64, per-channel gates, margin changes, or objective changes.
32. **Running projection strength 3.0 after v14c.** v14c strength 2.0 regressed A versus v14b and narrowed C/D margins. Stronger projection is not the cleanup; stop projection-strength escalation.
33. **More than one A-pressure cleanup in the adapter line.** v14d (`black_predrop_correction=2→3` on top of v14b) was the deliberately narrow final cleanup because v14b missed only A mean by +0.026 while B/C/D passed. It failed: A still missed (mean +0.051) and D severe broke to 3.3%. Do not run A=4/A=5 or more guardrail-count tweaks inside the adapter line.
34. **Another argument-only adapter cleanup after v14d.** v14b remains the best adapter near-pass, but v14c (strength 2.0) and v14d (A draw 3) both failed. Do not run strength 3.0, width 64, per-channel gates, margin changes, or objective changes as an incremental tweak. Any continuation must be a new written design with a new mechanism and explicit acceptance/falsification criteria.
35. **Building v15 Phase 1 as depth-1 child rows or shallow PV/path rows.** Phase 0 made the A excess look targetable at the selected-child level, but Phase 0.5 falsified the few-row/PV interpretation. The positive raw mass that MCTS backs up is broad across the depth-2 frontier: median 196 depth-2 nodes per root are needed to cover 70% of positive raw mass, and the median PV share of positive raw mass is only 0.335%. Do not build `black_predrop_continuation_correction` as child/PV rows.
36. **Treating the A overvalue as another raw-root/value-adapter cleanup.** Raw A is already non-positive at BASE and more negative under v14b, while MCTS re-amplifies it from policy-selected frontier states. Do not run more A draw pressure, width/capacity tweaks, projection tweaks, or root raw hard-value rows. The next value-calibration branch must target the frontier distribution explicitly, or the work should redirect to search/prior behavior.
37. **Building v16 as policy-selected frontier hard-value correction.** The post-v15 budget/trajectory diagnostics falsified the remaining value-calibration interpretation. BASE A collapses with more search (400/1600/6400 mean +0.2570 → +0.0626 → −0.0451; gate-over 50.0% → 30.0% → 10.0%; severe 43.3% → 6.7% → 3.3%), and the trajectory check showed the selected predrop ply was inflated by 400-sim search while the drop ply was not. Do not train thousands of depth-2 `hard_value=-0.35` rows; that would gate-fit shallow-search noise.
38. **Treating v16 as another calibration/training branch.** v16 is reserved for search reliability only. No trainer/network/manifest/value-adapter changes under the v16 name.
39. **Using c_puct to fix or gate-pass A.** The v16 c_puct falsification diagnostic reproduced the gate at c_puct=1.5 and then showed lowering c_puct worsens A (mean +0.2570→+0.3778; over 50.0%→60.0%; severe 43.3%→50.0%) by increasing top-child visit share and top-child visited children. Raising c_puct is also disallowed as an A fix because it would reduce the metric by spreading visits onto inferior root moves, not by fixing the depth-2 frontier. c_puct is closed; the only allowed next search-code lever is opt-in FPU with default `0.0` preserving current behavior exactly.
40. **Treating selected-A FPU success as adoption.** The v16 FPU selected-A diagnostic is the first positive search-mechanism result, but it is still on a biased set selected by the flawed 400-sim A statistic. Do not adopt FPU into gates, self-play, or promotion rules based on selected-A results alone. Validate on an unbiased/non-selected sample first, then B/C/D under the same setting, then a head-to-head strength evaluation before any self-play adoption.
41. **Using absolute `fpu_value=−0.20` as a general 400-sim setting, or advancing it to B/C/D or a strength match.** v16a rejected it on the game-held-out sample: late new-collapse rate was 15.48% (13/84), with both late-red and late-black above the preregistered 10% stratum reject bar. The small mean value delta does not rescue a candidate that materially increases late root collapse.
42. **Tuning the replacement FPU candidate on the v16a held-out manifest.** The 324-position v16a sample has been observed and is consumed. Do not select absolute values, formulas, thresholds, or schedules against it. Diagnose the failure read-only, but design/tune any adaptive or parent-relative replacement using discovery data only; use a fresh or separately preregistered confirmatory holdout before adoption.


Also retired as *primary* strategies: global-weight sweeps, retention-weight sweeps, schedule-ratio sweeps, frozen-BN-as-the-fix reruns, raw-teacher weight/schedule tweaks, broad row-engineering, broader partial unfreeze, broad v10/v10b schedule-count sweeps, surgical B value-only root-clone manifest edits, projection-strength escalation, and adapter A-pressure cleanups. The active adapter-cleanup line is closed. The current default is to keep `calib020_0001`; any further calibration work requires a new written design.

## v14 adapter-projection cleanup status (2026-07-09)

### v14c — value-adapter projection strength 2.0 (RUN + REJECTED)

Checkpoint: `checkpoints/alphazero-v14c-value-adapter-projection-strength2-from-calib020-0001/model_iter_0001.safetensors`

Setup: same v14b value-adapter projection surface and v12b manifest/schedule/objective, but `--post-opening-calibration-projection-strength 2.0`. Telemetry/verifier were clean: `train_value_head_and_value_adapter=True`, `train_value_head_and_final_block=False`, `calib_projection_enabled=True`, `calib_projection_scope=value_head_and_value_adapter`, `calib_projection_strength=2.0`, `conflict_steps=49`, `conflict_rate=44.5%`, `removed_norm_avg=0.1329`, `value_adapter_gate=-0.001048`, `value_adapter_grad_norm=0.001371`, `calib_n_drawn_total=1440`, and the adapter verifier passed with only `value_head.*` + `value_adapter.*` changed.

Gate results:
- A black pre-drop: mean +0.060, over 30.0%, severe 20.0% — **FAIL**, and worse than v14b (mean +0.026, severe 16.7%).
- B goal-line: mean −0.261, over 5.6%, severe 0.0% — **PASS**.
- C old post-opening: mean +0.052, over 30.0%, severe 13.3% — **PASS**, but only at the severe cap.
- D red pre-drop: mean −0.056, over 26.7%, severe 0.0% — **PASS** by formal gate.

Decision: **REJECT / no promotion.** Projection strength 2.0 did not fix A and degraded the adapter-line shape versus v14b. Do **not** run strength 3.0. v14b remains the best adapter near-pass.

### v14d — one final narrow A-pressure cleanup (RUN + REJECTED)

Checkpoint: `checkpoints/alphazero-v14d-value-adapter-projection-a3-from-calib020-0001/model_iter_0001.safetensors`

Setup: same v14b value-adapter projection surface and v12b manifest/objective (`projection_strength=1.0`, bottleneck 32), but changed exactly one sampling knob: `black_predrop_correction=2→3`. All guardrail schedules, margin, projection mechanics, objective, and frozen surface were unchanged.

Telemetry/verifier:
- `train_value_head_and_value_adapter=True`, `train_value_head_and_final_block=False`.
- `calib_projection_enabled=True`, `calib_projection_scope=value_head_and_value_adapter`, `calib_projection_strength=1.0`.
- `conflict_steps=72`, `conflict_rate=54.5%`, `removed_norm_avg=0.0742`.
- `value_adapter_gate=-0.000585`, `value_adapter_grad_norm=0.001501`.
- `calib_n_drawn_total=1600`, `calib_n_drawn_per_step=10.0`, with A draws 480 and all guardrail draw counts unchanged from v14b.
- Adapter verifier passed: only `value_head.*` + `value_adapter.*` changed; frozen tensors byte-identical.

Gate results:
- A black pre-drop: mean +0.051, over 33.3%, severe 16.7% — **FAIL**; A did not cross mean ≤0.0.
- B goal-line: mean −0.299, over 5.6%, severe 0.0% — **PASS**.
- C old post-opening: mean +0.030, over 30.0%, severe 10.0% — **PASS**.
- D red pre-drop: mean −0.049, over 40.0%, severe 3.3% — **FAIL**; D severe must be 0.0%.

Decision: **REJECT / no promotion.** Extra A draw pressure did not push A through and broke D severe. This closes the argument-only v14 adapter cleanup line. Do not run A=4/A=5, projection strength 3.0, width 64, or another adapter schedule/projection tweak without a new written design.

## Next-step plan after v16a held-out rejection (2026-07-10)

**Default decision:** keep `calib020_0001` as current best. No v13/v14/v15 checkpoint earned a promotion match; v16/v16a were diagnostic-only and produced no model checkpoint.

**Closed lines:**
- v13 final-block projection cleanup is closed after v13d.
- v14 value-adapter cleanup is closed after v14c/v14d and the raw/MCTS drift diagnostic.
- v15 searched-continuation Phase 1 is closed before implementation: Phase 0.5 proved the optimism is broad depth-2 frontier mass, not a few child/PV states.
- The proposed v16 frontier value-correction design is rejected before build: budget/trajectory diagnostics show the A signal was selected shallow-search inflation, not a valid value-head target.
- c_puct is closed as an A/search-reliability fix.
- **Fixed absolute `fpu_value=−0.20` is closed as a general 400-sim setting.** It reached the selected-A mechanism but failed the preregistered held-out late-collapse gate.

**v16a result:** the frozen `0.0` vs `−0.20` test ran on 324 positions from 252 games, with 40/100/100/84 positions across opening/early-mid/midgame/late, exact red/black balance, and zero A-discovery game overlap. Overall new-collapse rate was 15/324 = 4.63%, just below the 5% overall reject line, but late play failed decisively: 13/84 = 15.48%, with late-red 6/42 = 14.29% and late-black 7/42 = 16.67%. Late collapsed roots rose from 17/84 to 28/84. The candidate also flipped the top move on 27.16% of positions and reduced effective children by 36.66 on average (107.58→70.92, about −34.1%). Central value movement stayed small (mean mover delta +0.0028; median absolute 0.0180; p95 absolute 0.2822), so the failure is search-shape concentration rather than broad mean-value drift.

**Immediate next step — read-only postmortem:** characterize the 15 new-collapse cases without changing the gate or selecting a replacement on this holdout. Report whether the cases cluster in 280-ply state-cap games; whether collapse coincides with top-move changes; baseline/candidate top share, effective children, mover value, root breadth, and opponent-reply count; and relevant legal-move/root-value context. This analysis may explain the failure mechanism but may not tune the next candidate.

**Next candidate design:** retire fixed absolute `−0.20`. A credible continuation should change the mechanism—most likely an adaptive or parent-relative FPU reduction—using the selected-A/discovery corpus and other non-v16a development data only. The v16a manifest is consumed and must not be used to choose the formula or its parameters. Do not run B/C/D or a strength match until a new candidate is frozen and passes an appropriate collateral screen.

**Decisive benchmark remains unchanged:** after collateral and B/C/D guardrail checks pass under an explicitly frozen ship-form search rule, run a same-checkpoint / same-400-sim / balanced-color head-to-head against FPU-off. Adoption requires a statistically significant strength gain; matching 6400-sim A values is mechanistic evidence, not success.

## Raw/MCTS drift diagnostic on A & D (RESOLVED 2026-07-09) — closes the v14 adapter line, sets v15

Ran the line-196 candidate: `eval_raw_nn_position_rows` (raw NN, no MCTS, eval-mode BN) on the A black-pre-drop + D red-pre-drop probe rows for BASE / v14b / v14d, juxtaposed with the 400-sim MCTS gate means. Raw CSV: `logs/eval/v15prep_raw_AD_drift_base_v14b_v14d.csv`.

**A — black pre-drop (want mean ≤ 0):**

| ckpt | raw mean | MCTS mean | search Δ (MCTS−raw) |
|---|---|---|---|
| BASE | −0.015 | +0.257 | +0.272 |
| v14b | −0.178 | +0.026 | +0.204 |
| v14d | −0.070 | +0.051 | +0.121 |

**D — red pre-drop (want mean ≤ 0, severe = 0):**

| ckpt | raw mean | MCTS mean | search Δ |
|---|---|---|---|
| BASE | +0.052 | −0.188 | −0.240 |
| v14b | +0.166 (raw severe 36.7%) | −0.047 | −0.213 |
| v14d | +0.051 | −0.049 | −0.100 |

**Conclusion — A is MCTS/search amplification, NOT raw-value undercorrection:**
- Raw A is already ≤0 at BASE (−0.015); the entire +0.257 gate overvalue is *added by the search* (+0.272). v14b over-corrected raw A to −0.178 yet the search still delivered +0.026 (re-amplified +0.204). The failed v14b/v14d gate is not raw capacity — the search backs up optimistic black continuations from the pre-drop roots.
- The search-Δ itself varies with training (v14b +0.204 → v14d +0.121), so the untapped lever is *reducing the search amplification*, not more raw correction (v14c/v14d proved that non-monotonic).
- D confirms the danger: raw D drift (v14b +0.166 / severe 36.7%) is *masked* by the search (MCTS −0.047) until the mask thins — exactly why "push A harder" broke D severe in v14d.

**Decision:** the v14 value-ADAPTER line did its job as a raw-surface experiment and is **CLOSED**. No width 64, per-channel gate, stronger projection, more A draws, or raw-adapter cleanup. **v15 = A searched-continuation correction on the v14b adapter+projection surface** — correct the child/PV states MCTS uses to produce the +0.20 backup, not the root raw value. Matches the line-291 tree/path hypothesis. First implementation step: a read-only A-continuation *concentration* diagnostic (is the +0.204 from a few child/PV states or broad?) to decide few-rows vs tree/path. Design: `docs/superpowers/specs/2026-07-09-targeted-value-calibration-v15-a-searched-continuation-correction-design.md`.

## v15 searched-continuation diagnostics (RESOLVED 2026-07-09) — closes v15 Phase 1

### Phase 0 — A-continuation concentration diagnostic

Design/plan: `docs/superpowers/specs/2026-07-09-targeted-value-calibration-v15-a-searched-continuation-correction-design.md` and `docs/superpowers/plans/2026-07-09-targeted-value-calibration-v15-phase0-concentration-diagnostic.md`.

Script: `scripts/GPU/alphazero/diagnose_v15_a_continuation_concentration.py`. Output: `logs/eval/v15prep_a_continuation_concentration.csv`.

Read: **concentrated at the selected-child level**. Of the 30 A roots, the 17 with `root_mcts_black_value > 0` were all concentrated. Globally, top-3 children carried 98.1% of all positive backup mass; a top-3-positive-per-root selection yielded 27 depth-1 child branches under the locked 90%/max-3 rule. Integrity checks passed: the sign invariant `sum(visit_share * -child.q_value) == root.q_value` held, and the 30-root MCTS mean reproduced the A gate mean (+0.2570).

Important finding: the depth-1 child raw values were already not the source of the backup. At the top-1 positive child of each overvaluing root, BASE raw black value averaged −0.087 while searched black value averaged +0.619. The +0.706 gap was search below the child, so depth-1 hard-value rows would only help if the value head generalized to deeper leaves.

### Phase 0.5 — selected-branch subtree diagnostic

Design/plan: `docs/superpowers/specs/2026-07-09-targeted-value-calibration-v15-phase05-selected-branch-subtree-diagnostic-design.md` and `docs/superpowers/plans/2026-07-09-targeted-value-calibration-v15-phase05-selected-branch-subtree-diagnostic.md`.

Script: `scripts/GPU/alphazero/diagnose_v15_a_selected_branch_subtrees.py`. Outputs:
- `logs/eval/v15prep_a_selected_branch_subtrees.csv`
- `logs/eval/v15prep_a_selected_branch_subtrees_by_depth_summary.csv`
- closure summaries: `logs/eval/v15prep_a_phase05_per_root_decomposition.csv`, `logs/eval/v15prep_a_phase05_depth2_coverage.csv`, `logs/eval/v15prep_a_phase05_pv_offpv_mass.csv`, `logs/eval/v15prep_a_phase05_closure_summary.txt`

Scope: reran deterministic BASE 400-sim MCTS on the 17 positive A roots, selected positive branches by cumulative positive share ≥0.90 / max 3, then walked every expanded descendant with `visit_count >= 1`, PV annotated. No depth cap, no manifest, no replay JSONs, no training. All 17 roots passed three fail-loud checks: tree reproduction against Phase 0 CSV, contribution invariant, and depth-1 cross-CSV perspective tie.

Result: **Case B — broad frontier optimism**.

- Nodes walked: 5,837.
- Raw-scored nodes: 5,745.
- Terminal nodes: 92.
- Depth-2 frontier: 4,443 nodes, 77.3% of leaf evaluations, mean raw_black(BASE) +0.793, 98.8% raw-positive.
- PV nodes: 139 / 5,745 leaf evaluations (2.4%), mean raw_black(BASE) −0.207.
- Off-PV mean raw_black(BASE): +0.618.
- Depth-2 nodes needed per root to cover 70% positive raw mass: p25 164, median 196, p75 205, max 238.
- PV share of positive raw mass: p25 0.176%, median 0.335%, p75 0.928%, max 23.53%.

Interpretation: Phase 0 proved the A excess is concentrated at which child branch MCTS selects, but Phase 0.5 proved the actual raw optimism being backed up is **not** concentrated on the child/PV path. It is broad across the policy-selected depth-2 frontier. A few-row child/PV continuation manifest would train the wrong states and miss nearly all positive raw mass.

**Decision:** v15 Phase 1 is **rejected before implementation**. Do not build depth-1 child rows, shallow PV/path rows, or semi-PV continuation rows. A temporary v16 frontier/tree-level value design was considered, but the later budget/trajectory diagnostics rejected it before build and redirected v16 to search reliability.

## Post-v15 search-budget + trajectory diagnostics (RESOLVED 2026-07-10) — closes A value calibration, sets v16

### A search-budget sweep on BASE

Ran the same A black-pre-drop probe on BASE / `calib020_0001` at 400, 1600, and 6400 sims. The corrected gate threshold is `over >= 0.25` and `severe >= 0.50`.

| sims | mean black value | gate over (>=0.25) | gate severe (>=0.50) |
|---:|---:|---:|---:|
| 400 | +0.2570 | 50.0% | 43.3% |
| 1600 | +0.0626 | 30.0% | 6.7% |
| 6400 | -0.0451 | 10.0% | 3.3% |

Result: the A gate signal collapses with more search. The earlier quick summary's `positive>0` column was not the gate overvalue metric; the true gate-over rate falls 50.0% -> 30.0% -> 10.0%.

### Predrop trajectory check

Checked five high-A loss games across the predrop->drop window. The 400-sim rerun reproduced the replay-stored `root_value` with max absolute diff 0.0000 across all checked plies, validating the pipeline and perspective. Comparing 6400 vs 400 showed the “sharp value drop” is mostly a predrop bump:

| position in window | mean (6400 - 400) |
|---|---:|
| predrop - 6 | -0.185 |
| predrop - 4 | -0.152 |
| predrop | -0.573 |
| drop ply | -0.001 |
| drop + 2 | +0.035 |
| drop + 4 | +0.012 |

At the predrop ply, four of five sampled high-A cases lost gate-over status at 6400 sims (`000281`, `000259`, `000127`, `000347`); `000611` remained overvalued and matches the full-30 result where only about 10% remain overvalued at 6400.

Interpretation: the original loss-analysis selection maximized `predrop_value - drop_value` using a 400-sim root statistic. Because 400-sim inflation is large at predrop and absent at the drop ply, the selection criterion picked the plies where shallow-search inflation was largest. The “drop” was mostly the selected 400-sim bump unwinding; at 6400 sims the curve is mostly a smooth decline or already non-overvalued.

**Decision:** value calibration against the A signal is not justified. The proposed v16 frontier hard-value correction is rejected before build. Do not train depth-2 frontier rows, child/PV rows, or any new A hard-value manifest from this signal. The remaining well-posed question is search reliability: can 400-sim MCTS be made to behave more like 6400 on these high-branching positions?

**v16:** the first v16 artifact was the read-only c_puct falsification sweep; c_puct is now closed as a fix. Continue only with a minimal opt-in FPU diagnostic if the search-reliability line proceeds.

## v16 c_puct falsification diagnostic (RESOLVED 2026-07-10) — c_puct is not the fix

Ran a read-only c_puct sweep on the 30 A black-pre-drop probe rows using BASE / `calib020_0001`. Integrity check passed: at `c_puct=1.5`, all 30 cases reproduced Phase 0 `root_mcts_black_value` within 1e-6. The sweep then varied only `MCTSConfig.c_puct`.

| c_puct | mean black value | gate over (>=0.25) | gate severe (>=0.50) | root children | top-child children | top-child visit share |
|---:|---:|---:|---:|---:|---:|---:|
| 1.5 | +0.2570 | 50.0% | 43.3% | 80.4 | 134.7 | 0.474 |
| 1.0 | +0.3018 | 53.3% | 46.7% | 82.2 | 160.4 | 0.506 |
| 0.75 | +0.3027 | 53.3% | 40.0% | 82.0 | 175.8 | 0.552 |
| 0.5 | +0.3237 | 56.7% | 43.3% | 79.9 | 197.1 | 0.577 |
| 0.25 | +0.3778 | 60.0% | 50.0% | 79.1 | 232.9 | 0.642 |

Result: c_puct does not merely fail to help; lowering it actively worsens the A metric. The mean black value rose monotonically from +0.2570 to +0.3778 and moved farther away from the 6400-sim reference (mean -0.0451, over 10.0%, severe 3.3%).

Mechanism: lower c_puct increases root concentration (`top_child_visit_share` 0.474 -> 0.642), which sends more simulations into the selected root child. That child then scans more fresh opponent replies (`top_child_n_visited_children` 134.7 -> 232.9), exactly the depth-2 one-visit frontier mechanism implicated by Phase 0.5. Across all case/config rows, `top_child_n_visited_children` correlated with `root_mcts_black_value` at +0.943.

Important non-fix: raising c_puct must not be used to pass A. It can lower the metric by spreading root visits onto inferior black moves, not by repairing the frontier; that would optimize the gate statistic by degrading search.

**Decision:** c_puct is closed as a search-reliability fix. The next possible v16 code change is FPU/unvisited-child initialization, but only as an opt-in diagnostic with byte-identical default behavior and validation beyond the selected A set.

## v16 FPU selected-A diagnostic (RESOLVED 2026-07-10) — first positive search-mechanism result, not adoption

Added an opt-in FPU/unvisited-child initialization knob to MCTS with byte-identical default behavior: `MCTSConfig.fpu_value = 0.0`, and `_select_child` uses that value for unvisited-child q instead of hardcoded `0.0`. The default integrity check passed: `fpu_value=0.0` reproduced Phase 0 per-case root values within 1e-6, and the full suite passed.

Ran the selected A black-pre-drop probe rows at 400 sims across negative FPU values.

| fpu_value | mean black value | gate over (>=0.25) | gate severe (>=0.50) | opponent replies scanned | root moves |
|---:|---:|---:|---:|---:|---:|
| 0.0 | +0.2570 | 50.0% | 43.3% | 134.7 | 80.4 |
| -0.05 | +0.1725 | 40.0% | 26.7% | 102.4 | 67.9 |
| -0.10 | +0.0776 | 23.3% | 13.3% | 60.8 | 54.1 |
| -0.20 | -0.0344 | 6.7% | 6.7% | 24.5 | 33.5 |
| -0.35 | -0.1003 | 6.7% | 0.0% | 8.0 | 19.1 |
| -0.50 | -0.1261 | 3.3% | 0.0% | 6.2 | 10.4 |

Reference: 6400-sim BASE A was mean -0.0451, gate-over 10.0%, severe 3.3%.

Interpretation: FPU is the first tested knob that directly reaches the measured mechanism. Unlike c_puct, which increased top-child reply scanning, negative FPU collapses the opponent's first-touch reply scan and moves the 400-sim metric toward the 6400-sim reference. fpu≈−0.20 is the preferred next validation setting because it is closest to the 6400 mean while preserving more breadth than −0.35/−0.50.

Limitations:
- The sample is biased: these 30 roots were selected by the same 400-sim statistic now known to be inflated.
- FPU reaches the corrected value by narrowing first-touch exploration, while 6400 sims reaches it by deeper search; those are not automatically equivalent.
- The constant `fpu_value` form is diagnostic. A future ship-form may be parent-relative/reduction-based and must be evaluated separately.

**Decision at this stage:** selected-A FPU was promising but not adoptable. Freeze `−0.20` as the single candidate and proceed to a game-held-out v16a comparison against `0.0`; do not tune multiple values on the holdout. The completed v16a result below subsequently rejected the fixed absolute candidate.


## v16a stratified, game-held-out FPU validation (RESOLVED 2026-07-10) — absolute `−0.20` rejected

### Tooling and locked protocol

The v16a tooling generalized the FPU diagnostic without changing the trusted position-reconstruction path or legacy selected-A output. The selected-A path remained byte-identical through golden-output and old-vs-new fake-search comparisons; `fpu_value=0.0` remained the default MCTS behavior. Neutral mode enforced the frozen protocol `0.0` versus `−0.20`, used mover-perspective deltas as primary, and emitted paired search-shape summaries overall and by bucket / side / bucket×side.

The generated neutral manifest contained **324 positions from 252 held-out games**:

| bucket | positions |
|---|---:|
| opening | 40 |
| early-mid | 100 |
| midgame | 100 |
| late | 84 |

The sample was exactly balanced at **162 red / 162 black**, shared **zero games** with the selected-A discovery manifest, and reconstructed every retained row through `position_state`. The 19 `winner=null` games were valid 280-ply `state_cap` marathons and were retained as `game_result=unknown`; they contributed 47 of the 84 late positions and intentionally strengthened the stressed late-game screen.

### Overall paired result

| metric | `fpu=0.0` | `fpu=−0.20` / delta |
|---|---:|---:|
| positions | 324 | 324 |
| mean mover-value delta vs control | — | +0.0028 |
| median absolute mover-value delta | — | 0.0180 |
| p90 / p95 absolute mover-value delta | — | 0.2079 / 0.2822 |
| top-move flip rate | — | 27.16% |
| mean root entropy delta | — | −0.6106 |
| mean effective children | 107.58 | 70.92 (−36.66; about −34.1%) |
| mean visited root children | 132.49 | 90.81 (−41.68) |
| mean top-child reply count | 105.42 | 75.87 (−29.55) |
| stable-top reply-count delta | — | −24.56 |
| mean top-child visit share | 0.4154 | 0.4870 (+0.0716) |
| new / resolved collapses | — | 15 / 2 |
| collapsed-root rate | 7.72% | 11.73% |

The central value distribution did **not** show broad drift: the mean mover delta was nearly zero and the median absolute delta was only 0.018. The dominant effect was search narrowing and concentration.

### Preregistered gate result

The overall new-collapse rate was **15/324 = 4.63%**, narrowly below the 5% overall hard-reject threshold. The preregistered stratum rule nevertheless fired decisively: any stratum with `n ≥ 20` and new-collapse rate `≥10%` is an automatic reject.

| stratum | n | new collapses | rate | top-move flips | effective-children delta | top-share delta |
|---|---:|---:|---:|---:|---:|---:|
| opening | 40 | 0 | 0.00% | 35.00% | −20.39 | +0.0646 |
| early-mid | 100 | 2 | 2.00% | 29.00% | −45.09 | +0.0362 |
| midgame | 100 | 0 | 0.00% | 15.00% | −19.73 | +0.0227 |
| **late** | **84** | **13** | **15.48%** | **35.71%** | **−54.52** | **+0.1753** |
| **late-red** | **42** | **6** | **14.29%** | **38.10%** | **−80.09** | **+0.2099** |
| **late-black** | **42** | **7** | **16.67%** | **33.33%** | **−28.94** | **+0.1407** |

Late collapsed roots rose from **17/84 (20.24%)** under control to **28/84 (33.33%)** under `−0.20`: 13 new collapses, only 2 resolved, for a net increase of 11. Both late-side strata independently exceeded the reject threshold.

### Interpretation and decision

The selected-A mechanism finding remains valid: pessimistic FPU suppresses broad one-touch opponent-reply scanning, which c_puct could not reach. But a fixed absolute `−0.20` applies that suppression too aggressively in ordinary late-game positions, producing materially more near-single-line roots. Matching the 6400-sim A mean was therefore not evidence that the 400-sim search had become generally more reliable.

**Decision: REJECT absolute `MCTSConfig.fpu_value = −0.20` for general 400-sim search and self-play.** Do not run B/C/D or a head-to-head strength match with this candidate. Do not alter the preregistered threshold after observing the near-miss on the overall 5% line; the late-stratum reject is clear and independently repeated across red and black.

**Next:** perform a read-only case-level postmortem on the 15 new collapses, then design a new candidate on discovery data only. The likely direction is an adaptive or parent-relative FPU reduction rather than a fixed absolute value. The v16a manifest is consumed and must not be used to select that replacement.


## Severe-overlap findings (why the next step changes shape)

- **C — stable repeat offenders:** 5 of the severe cases repeat across v3/v3b (listed above). C should be treated as a fixed fragile family needing **direct retention of current-best behavior**, not as eval noise.
- **D — diffuse:** only 1 shared severe case across v3/v3b. D reads as **broad value-head drift**; unlikely to be solved by adding a few hard D rows.

## v4/v3-frozenBN severe-overlap follow-up

The post-v4 overlap check shows mixed failure structure:

- **B goal-line:** no severe-case overlap between v4 and `v3-frozenBN-control`. Treat B as a fragile guardrail, not a fixed-row problem yet.
- **C old post-opening:** 4 shared severe rows repeat across v4 and `v3-frozenBN-control`: `game_000065_ply_021`, `game_000369_ply_051`, `game_000505_ply_037`, `game_000619_ply_061`. This is the strongest stable fragile-family signal.
- **D red pre-drop:** only 1 shared severe row, `red_loss_game_000362_predrop_ply_52_drop_54`, but it shows strong value drift: baseline MCTS +0.198 → v4 MCTS +0.582 → `v3-frozenBN-control` MCTS +0.677. D remains mostly diffuse despite one common failure.

The shared fragile C/D rows were present in the v4 teacher-retention manifest, so v4 did **not** fail because the retention pool missed them. The stronger finding is that **raw-NN teacher retention did not preserve the 400-sim MCTS probe behavior**.

Key examples:
- `game_000065_ply_021`: teacher raw value +0.1105, base MCTS +0.480, v4 MCTS +0.758.
- `game_000369_ply_051`: teacher raw value −0.1389, base MCTS +0.334, v4 MCTS +0.765.
- `red_loss_game_000362_predrop_ply_52_drop_54`: teacher raw value −0.9379, base MCTS +0.198, v4 MCTS +0.582, `v3-frozenBN-control` MCTS +0.677.
- `game_000505_ply_037`: teacher raw value +0.9455 and base MCTS +0.856, so this row is already pro-black under the teacher/baseline and should not be treated as a clean retention failure.

Conclusion: before another branch, inspect raw-NN candidate values on these rows. If v4 matched raw teacher values but MCTS still drifted, the next design should move from raw-NN teacher retention to **MCTS-root retention** or another root-behavior retention objective.

## Resolved diagnostic after v4 and v3-frozenBN-control

Both completed follow-ups are rejects:

- `v3-frozenBN-control` passed A but failed B/C/D, proving that v3's guardrail failure was not primarily a train-mode BatchNorm artifact.
- v4 teacher-retention passed A strongly but failed B/C/D, even though the shared fragile C/D rows were present in the v4 manifest.

The next disciplined step is **not a new training branch**. First, score the shared C/D rows with raw NN-only evaluation for:

- `BASE = calib020_0001`
- `V4 = checkpoints/alphazero-v4-teacher-from-calib020-0001/model_iter_0001.safetensors`
- `V3F = checkpoints/alphazero-v3-frozenBN-control-from-calib020-0001/model_iter_0001.safetensors`

Focus rows:
- `game_000065_ply_021`
- `game_000369_ply_051`
- `game_000619_ply_061`
- `game_000505_ply_037` (diagnostic only; baseline/teacher already high)
- `red_loss_game_000362_predrop_ply_52_drop_54`

Decision value:
- If v4 matched the raw teacher values on these rows but MCTS still drifted, raw-NN teacher retention is the wrong objective for the gate and the next branch should use **MCTS-root/root-behavior retention**.
- If v4 did not match the raw teacher values, inspect loss weighting, masking, and gradient influence before designing a new branch.
- If raw-NN and MCTS disagree systematically on these rows, the gate must be treated as root-search behavior, not just value-head calibration.

Until this raw-NN candidate scoring is done, do **not** run another v3/v4 weight, policy-KL, or schedule sweep.

**→ RESOLVED 2026-07-01** — the scoring is done (next section). The first decision branch holds: **v4 matched the raw teacher values but the MCTS gate still drifted** ⇒ the next design is MCTS-root/root-behavior retention.

## Raw-NN focus-row diagnostic after v4/v3-frozenBN (2026-07-01)

Run via the new read-only `scripts/GPU/alphazero/eval_raw_nn_position_rows.py` CLI (raw NN forward only, no MCTS, eval-mode BatchNorm; plan `docs/superpowers/plans/2026-07-01-eval-raw-nn-position-rows-diagnostic.md`). The diagnostic scored BASE (`calib020_0001`), v4, and `v3-frozenBN-control` on the shared C/D severe rows. **BASE anchors reproduced exactly, validating reconstruction and eval-mode scoring.**

**Result: v4 mostly preserved the raw teacher values, while v3-frozenBN showed large raw drift on the key C rows.**

Key rows:

- `game_000369_ply_051`: BASE raw −0.1389, v4 raw −0.0262 (delta +0.1127, non-severe), v3F raw +0.6670 (delta +0.8059, severe). Yet v4's 400-sim MCTS gate was severe. **This is the cleanest evidence that raw retention held but MCTS/root behavior drifted.**
- `game_000065_ply_021`: BASE raw +0.1105, v4 raw +0.2697 (delta +0.1592, over but non-severe), v3F raw +0.5939 (delta +0.4834, severe).
- `red_loss_game_000362_predrop_ply_52_drop_54`: BASE raw stm −0.9379, v4 raw stm −0.8857 (delta +0.0522), v3F raw stm −0.8219 (delta +0.1160). v4 stayed close to the raw teacher despite failing the MCTS-root gate.
- `game_000619_ply_061` and `game_000505_ply_037` are already raw-severe under BASE, so they are useful diagnostics but not clean examples of newly-created raw drift.

**Conclusion:** v4 did not primarily fail because raw teacher-retention missed or ignored the fragile rows. It mostly preserved raw NN behavior, but that did not preserve the 400-sim MCTS gate behavior. The next branch should move to **MCTS-root/root-behavior retention** or an equivalent root-search preservation objective. Do **not** run another raw teacher-retention weight/schedule sweep as the next step.

## v5 design — MCTS-root-visit policy retention (LOCKED 2026-07-01 · RUN + REJECTED 2026-07-02, result below)

**Key correction that shaped v5 (code-verified):** root-value-only retention is **not new** — `build_targeted_calibration_manifest.py:105,137` set the v2/v3 retention rows' `target_black_value` from `probe_black_root_value`, i.e. **v2/v3 scalar retention already trained the raw value head toward BASE's 400-sim MCTS root values** — and failed B/C/D (twice, incl. frozen-BN control). So "root-value retention only" is v3 with a cleaner name: dead on arrival (do-not-repeat #9).

**The lineage that makes v5 the first genuinely new combination:**

| Branch | Value target | Policy target | Result |
|---|---|---|---|
| v3 / v3F | BASE MCTS-root value | none | failed B/C/D |
| v4 | BASE raw teacher value | BASE raw teacher priors | raw held, MCTS root still failed |
| **v5** | **BASE raw teacher value** | **BASE 400-sim MCTS root visit distribution** | **untested** |

The raw diagnostic showed v4 mostly held raw value, so pushing harder on raw value is not the missing signal; the missing signal is likely the **search-improved root policy**.

**Locked design shape:**

1. **New builder** `scripts/GPU/alphazero/build_mcts_root_retention_manifest.py` — input: source v4/v3-style stratified manifest + BASE checkpoint + gate MCTS config → output v5 manifest. Retention rows append `root_value_stm, root_black_value, root_visits_json, root_legal_moves_sha1, root_sims, root_base_checkpoint, root_seed, root_mcts_eval_batch_size, root_mcts_stall_flush_sims`; correction rows leave them blank.
2. **Root target generation** per retention row: reconstruct via `position_state` → BASE MCTS at 400 sims, `add_noise=False` → dense visit vector aligned to `state.legal_moves()` → normalize to sum 1.0 → dense JSON + legal-move sha1; `root_value_stm`/`root_black_value` stored as metadata. Builder asserts recomputed `root_black_value` ≈ the gate CSV's `probe_black_root_value` where available (the "did we match the gate setup?" check).
3. **Training semantics:** new `loss_mode = mcts_root_retention`, but **no new trainer loss path**: `calibration_pool.build_calibration_position()` parses the mode into the existing v4 teacher-retention tuple shape (`record.outcome` = raw teacher value stm, `record.visit_counts` = normalized BASE root visits, mask present) → the existing 14-tuple masked value + policy-CE path handles it. v2/v3/v4 paths byte-identical when unused.
4. **Value target = raw teacher value, not root value** — avoids repeating v3 and avoids amplifying MCTS root values into the raw head; the diagnostic says v4 held that anchor well, so keep it as a stabilizer.
5. **Policy target = dense normalized root visits, not top-k** — already aligned to legal moves, compatible with the v4 policy-CE machinery, zeros are informative at 400 sims, sha1 validation fits, no lossy top-k reconstruction.
6. **Gate-0 / smoke expectation:** do NOT expect v5 policy loss ≈ 0 at init (root visits are search-improved; raw priors should differ). The v5 smoke instead validates: builder target correctness (recomputed BASE root values match gate CSV values); training mechanics (value term starts ≈ 0 for raw teacher value; policy CE finite and mask-aligned; `legal_moves_sha1` matches; no NaN / shape mismatch / BN train-mode drift).

**Gate:** same A/B/C/D probes vs `calib020_0001`. No promotion unless all four pass.

**Important limitation (record in the v5 plan):** root-visit anchors constrain the candidate's raw policy **at the anchored root positions only**. If gate drift is caused by candidate value/prior changes deeper in the tree, root-visit retention may still fail. If v5 fails with raw value AND root policy held at the anchors, the next hypothesis becomes **tree-level/path-level retention**, not more anchored rows or stronger weights.

### v5 — MCTS-root-visit policy retention (RESULT, 2026-07-02)

Checkpoint: `checkpoints/alphazero-v5-mcts-root-from-calib020-0001/model_iter_0001.safetensors`

Setup: A hard-value correction rows unchanged; B/C/D retention rows used raw teacher value as the value anchor and BASE 400-sim MCTS root visit distribution as the masked policy-CE target. Training used `--freeze-batchnorm-stats`, global calibration weight 0.01, teacher value weight 1.0, root-policy CE weight 0.25, and the 2:1:2:1 tag schedule.

Gate results:
- A black pre-drop: PASS-ish / improved — mean −0.174, over 20.0%, severe 20.0% versus baseline mean +0.257, over 50.0%, severe 43.3%.
- B goal-line: FAIL — mean −0.288, over 16.7%, severe 5.6%. Pass requires severe 0.0% and over ≤ 11.1%.
- C old post-opening: FAIL — mean +0.074, over 40.0%, severe 30.0%. Mean passed, but over/severe failed.
- D red pre-drop: HARD FAIL — mean +0.046, over 40.0%, severe 36.7%. Pass requires severe 0.0% and mean ≤ 0.0%.

Decision: REJECT. No promotion match.

Lesson: Position-level root-visit policy retention did not preserve B/C/D after A correction. v5 tested the hypothesis that v4 failed because it preserved raw priors rather than search-improved root policy; that hypothesis is insufficient. The next step should not be a root-policy weight sweep. First diagnose whether v5 actually held the stored root-policy anchors on the retention rows. If held, the remaining failure points to deeper tree/path-level drift rather than root-row anchoring.

Run telemetry (provenance): `mode=mcts_root_retention`, draws_by_tag 320/160/320/160 (exact 2:1:2:1 over 160 steps), `n_teacher_retention_drawn=640`, `calib_policy_ce_avg_iter=3.83`, `calib_policy_kl_est_avg_iter=1.24` (vs v4's 0.19 — the root-visit target was genuinely non-trivial), `calib_value_term_avg_iter≈0.12`, `freeze_batchnorm_stats=true`.

### v5 path diagnostic — searched continuation drift (2026-07-02)

A gate-faithful path diagnostic was run on six representative failed v5 rows using the same synchronous `MCTS.search` path as the gates/builders. BASE root values matched the stored manifest values exactly, validating the diagnostic.

Findings:
- On C rows (`game_000433`, `game_000065`, `game_000565`), v5 preserved the same dominant root move and similar root visit share:
  - `game_000433`: BASE 19:9 share 0.9975, V5 19:9 share 0.9850.
  - `game_000065`: BASE 13:18 share 0.8800, V5 13:18 share 0.8650.
  - `game_000565`: BASE 21:5 share 1.0000, V5 21:5 share 0.9850.
  Despite this, v5 remained severe/overvalued, showing root-policy retention is insufficient.
- The child/continuation values shifted materially. Example: `game_000565` retained the same root move 21:5, but child NN value moved from BASE −0.4707 to V5 +0.4791.
- On D rows, BASE root visit distributions were diffuse, and v5 child NN values shifted strongly pro-black:
  - `red_loss_000780`: top child NN +0.0976 → +0.8258.
  - `red_loss_000362`: top child NN +0.0322 → +0.8013.
  - `red_loss_000176`: top child NN −0.1810 → +0.8613.

Conclusion: **v5 failed because root-level anchors do not constrain searched continuation values.** The next branch should be **v6 searched-continuation/PV retention**: add child/PV states from BASE MCTS under fragile rows and retain their raw teacher values, with policy retention only where distributions are sharp.

## v6/v7/v8 follow-up results (2026-07-03)

### v6 — searched-continuation/PV retention

Manifest: `logs/eval/targeted_calibration_v6_continuation_from_calib020_0001.csv` (381 rows: 50 hard-value, 78 old root-retention rows, 253 searched-continuation rows). Smoke passed with value-only continuations (`policy_ce=0.0`, 0 policy rows) and schedule draws `2:1:2:2` for A/B-cont/C-cont/D-cont.

Gate results vs `calib020_0001`:
- A black pre-drop: mean −0.110, over 20.0%, severe 10.0% — **PASS / improved**.
- B goal-line: mean −0.321, over 16.7%, severe 0.0% — **FAIL** on over cap.
- C old post-opening: mean +0.003, over 30.0%, severe 20.0% — **FAIL** on severe.
- D red pre-drop: mean +0.150, over 53.3%, severe 30.0% — **HARD FAIL**.

Diagnostic: every failed B/C/D root had continuation rows, so coverage was not the failure. Raw-NN diagnostics showed D root raw values remained severe or drifted upward. Continuation-only was too indirect for D.

### v6b — D root + continuation hybrid

Same v6 manifest, but training also scheduled `red_predrop_retention=1`, reactivating D root policy/root-visit retention. Telemetry confirmed `n_teacher_retention_drawn=160`, `policy_ce=3.809`, `policy_kl_est=1.297`.

Gate results:
- A: mean −0.308, over 3.3%, severe 3.3% — **PASS**.
- B: mean −0.240, over 16.7%, severe 5.6% — **FAIL**.
- C: mean +0.118, over 56.7%, severe 23.3% — **HARD FAIL**.
- D: mean −0.009, over 40.0%, severe 26.7% — **HARD FAIL**.

Lesson: D root policy retention slightly improved D versus v6 but broke B/C badly. Do not reintroduce root policy CE/KL as a mixed guardrail strategy.

### v6c — D root value-only + continuation

Manifest: `logs/eval/targeted_calibration_v6c_d_root_value_only_from_calib020_0001.csv` (411 rows: v6 + 30 depth-0 `red_predrop_root_value_retention` rows). Validation proved D root clones were value-only: `teacher_value` populated, `target_black_value` blank, `teacher_policy_json`/`root_visits_json` blank, `continuation_depth=0`, `continuation_source=root_value`. Smoke passed with `policy_ce=0.0`, 0 policy rows, and schedule `2:1:2:1:2`.

Gate results:
- A: mean +0.006, over 30.0%, severe 23.3% — **FAIL / improved**.
- B: mean −0.195, over 16.7%, severe 11.1% — **FAIL**.
- C: mean −0.007, over 36.7%, severe 16.7% — **FAIL**.
- D: mean +0.032, over 33.3%, severe 13.3% — **FAIL**.

Lesson: value-only D root anchoring is less toxic than policy retention, but still too broad and still interferes with B/C.

### v7 — sparse severe-D hard correction

Manifest-only branch using `logs/eval/targeted_calibration_v7_severe_d_root_correction_from_calib020_0001.csv` (419 rows = v6c 411 + 8 hard-value rows). The 8 rows were selected by BASE raw severe-overvalue among D roots and assigned `target_black_value=-0.35` under tag `red_predrop_severe_root_correction`. Pool smoke passed: draws `2:1:2:1:2`, `policy_mask_sum=0.0`, no policy rows.

Selected severe-D rows: `red_loss_game_000752`, `000438`, `000362`, `000616`, `000408`, `000176`, `000456`, `000432` (BASE raw black approximately +0.62 to +0.97).

Gate results:
- A: mean −0.065, over 26.7%, severe 13.3% — **PASS / improved**.
- B: mean −0.290, over 16.7%, severe 5.6% — **FAIL**.
- C: mean +0.002, over 30.0%, severe 20.0% — **FAIL**.
- D: mean +0.034, over 40.0%, severe 23.3% — **HARD FAIL**.

Drift map: A and D means moved down, but D remained unstable (`up_0.25=12`, `down_0.25=12`) and B goal-line raw values moved upward on 9/18 rows (`mean_delta=+0.2015`). The problem was not just wrong row selection; full-network training moved the value surface nonlocally.

### v8 — value-head-only using v7 manifest

Implementation: `--train-value-head-only` skips the single `opt_main.update(main_module, main_grads)` call while always applying `opt_value.update(network.value_head, value_grads)`. Verifier CLI proved the trained checkpoint changed only the four `value_head.*` tensors; all 88 non-value-head tensors were byte-identical to BASE. Training telemetry: `train_value_head_only=True`, `freeze_batchnorm_stats=True`, `calib_n_drawn_total=1280`, `calib_n_drawn_per_step=8.0`, `policy_ce=0.0`, `n_teacher_retention_drawn=0`.

Gate results:
- A: mean +0.068, over 33.3%, severe 20.0% — **FAIL / improved but undercorrected**.
- B: mean −0.276, over 11.1%, severe 0.0% — **PASS**.
- C: mean +0.024, over 23.3%, severe 10.0% — **PASS**.
- D: mean −0.056, over 36.7%, severe 0.0% — **PASS**.

Decision: **Reject for promotion** because A failed, but this is the most informative positive result of the line: value-head-only protected B/C/D, supporting the hypothesis that full-network/trunk drift caused the v6/v7 guardrail failures.

### v8b — value-head-only, higher A draw pressure (RUN + REJECTED 2026-07-03)

Same v7 manifest + value-head-only mechanics, A draw pressure raised 2→3 (`black_predrop_correction=3,goal_line_continuation_retention=1,old_post_opening_continuation_retention=2,red_predrop_severe_root_correction=1,red_predrop_continuation_retention=2`). Telemetry/verifier were clean: `train_value_head_only=True`, `freeze_batchnorm_stats=True`, `calib_n_drawn_total=1440`, `calib_n_drawn_per_step=9.0`, `policy_ce=0.0`, `n_teacher_retention_drawn=0`, and only the four `value_head.*` tensors changed.

Gate results:
- A: mean +0.102, over 33.3%, severe 26.7% — **FAIL / worse than v8**.
- B: mean −0.286, over 5.6%, severe 0.0% — **PASS**.
- C: mean +0.086, over 33.3%, severe 13.3% — **PASS but worse than v8**.
- D: mean −0.096, over 26.7%, severe 0.0% — **PASS**.

Decision: **REJECT.** Higher A draw pressure did not help A and made A/C worse. This is not a simple "more A mass" problem.

### v8/v8b raw-A diagnostic (why value-head-only can't fix A)

On the 50 `black_predrop_correction` rows, **raw** value-head output barely moved:

| | raw mean | Δ vs BASE | severe raw overvalue |
|---|---|---|---|
| BASE | −0.2469 | — | 20.0% |
| v8 | −0.2533 | −0.0064 | 14.0% |
| v8b | −0.2433 | +0.0035 | 16.0% |

A did **not** fail because MCTS amplified an already-corrected raw value — the raw values themselves scarcely changed. A failed because value-head-only cannot substantially move the worst A raw values with the trunk frozen: `value_head` is a shallow MLP readout (`fc1→fc2`, no conv/BN) on frozen features.

**Conclusion:** v8 proved full-network drift was the main cause of B/C/D breakage (value-head-only preserved B/C/D). But value-head-only is too constrained to fix A. Next hypothesis is **partial unfreeze**: value head + the smallest late representation slice, starting with the final encoder/residual block.

## v9 — value head + final residual block partial unfreeze (RUN + REJECTED 2026-07-03)

Design spec: `docs/superpowers/specs/2026-07-03-targeted-value-calibration-v9-value-head-and-final-block-design.md`; implementation plan: `docs/superpowers/plans/2026-07-03-targeted-value-calibration-v9-value-head-and-final-block.md`.

Setup: same v7 manifest, v8 schedule `black_predrop_correction=2,goal_line_continuation_retention=1,old_post_opening_continuation_retention=2,red_predrop_severe_root_correction=1,red_predrop_continuation_retention=2`, global weight 0.01, `--freeze-batchnorm-stats`, and `--train-value-head-and-final-block`.

Implementation/acceptance proof: v9 skipped the whole-trunk `opt_main.update`, applied exactly one `opt_main.update(network.encoder.blocks[last], main_grads["encoder"]["blocks"][last])`, and always applied `opt_value.update(network.value_head, value_grads)`. Telemetry was clean: `train_value_head_and_final_block=True`, `unfrozen_block_index=5`, `freeze_batchnorm_stats=True`, `calib_n_drawn_total=1280`, `calib_n_drawn_per_step=8.0`, `policy_ce=0.0`, and `n_teacher_retention_drawn=0`. The strict tensor-diff verifier passed: only the 4 `value_head.*` tensors plus the 8 trainable tensors under `encoder.blocks.5.*` changed; all frozen tensors and all BN running stats stayed byte-identical.

Gate results:
- A black pre-drop: mean −0.089, over 30.0%, severe 16.7% — **PASS**.
- B goal-line: mean −0.238, over 22.2%, severe 11.1% — **FAIL**.
- C old post-opening: mean +0.067, over 46.7%, severe 30.0% — **FAIL**.
- D red pre-drop: mean −0.115, over 26.7%, severe 20.0% — **FAIL**.

Decision: **REJECT.** No promotion match.

Lesson: v9 gives the missing representational flexibility that v8 lacked — A passes — but unfreezing even the final residual block is enough to reintroduce the nonlocal B/C/D guardrail drift. The v8/v9 contrast identifies the tradeoff location: value-head-only protects B/C/D but cannot move A; value head + final block moves A but breaks B/C/D. Do **not** run v9b last-2 blocks as the next simple extension; broader partial unfreeze is expected to worsen the same failure mode unless a new constraint/objective is introduced.


## v10/v10b — guarded final block with root/search-path schedule (RUN + REJECTED 2026-07-05)

### v10 — schedule-only root + continuation retention

Design spec: `docs/superpowers/specs/2026-07-04-targeted-value-calibration-v10-final-block-root-continuation-schedule-design.md` (committed @ `91e14ec`).

Setup: same v7 manifest and same v9 update surface (`--freeze-batchnorm-stats --train-value-head-and-final-block`), but changed only `--post-opening-calibration-tag-schedule` to enable dormant root tags alongside already-scheduled continuation tags:

`black_predrop_correction=2,goal_line_retention=1,goal_line_continuation_retention=1,old_post_opening_retention=1,old_post_opening_continuation_retention=2,red_predrop_root_value_retention=1,red_predrop_continuation_retention=2,red_predrop_severe_root_correction=1`

Telemetry/verifier:
- `calib_n_drawn_total=1760`, `calib_n_drawn_per_step=11.0`.
- Draws by tag: A 320, B root 160, B cont 160, C root 160, C cont 320, D root-value 160, D cont 320, D severe 160.
- `n_teacher_retention_drawn=320`, `calib_policy_ce_avg_iter=3.8914`, `calib_policy_kl_est_avg_iter=1.0596`, proving B/C root policy-CE rows fired.
- Strict verifier passed: only `value_head.*` and `encoder.blocks.5.*` trainable tensors changed; all frozen tensors byte-identical.

Gate results:
- A black pre-drop: mean −0.004, over 20.0%, severe 16.7% — **PASS**, but with thin mean margin.
- B goal-line: mean −0.195, over 11.1%, severe 5.6% — **FAIL** by one severe row.
- C old post-opening: mean +0.016, over 23.3%, severe 10.0% — **PASS**.
- D red pre-drop: mean −0.067, over 26.7%, severe 3.3% — **FAIL** by one barely-severe row.

Remaining v10 blockers:
- B severe: `game_000015_ply_19` value +0.6435, top1_share 0.8425. B also had one non-severe over row: `game_000327_ply_63` value +0.3538.
- D severe: `red_loss_game_000752_predrop_ply_70_drop_72` value +0.5003, top1_share 0.055. The row was barely above the severe threshold and diffuse.

Decision: **REJECT.** No promotion match.

Lesson: v10 is the best near-pass in the line. Adding B/C root pressure did **not** make B/C worse; it recovered C fully and left B close. It also preserved the A fix. However, D remained structurally fragile and B/D still had one severe blocker each.

### v10b — stronger B/D schedule

Setup: config-only from v10; increased `goal_line_retention`, `red_predrop_root_value_retention`, and `red_predrop_severe_root_correction` from 1→2:

`black_predrop_correction=2,goal_line_retention=2,goal_line_continuation_retention=1,old_post_opening_retention=1,old_post_opening_continuation_retention=2,red_predrop_root_value_retention=2,red_predrop_continuation_retention=2,red_predrop_severe_root_correction=2`

Telemetry/verifier:
- `calib_n_drawn_total=2240`, `calib_n_drawn_per_step=14.0`.
- Draws by tag: A 320, B root 320, B cont 160, C root 160, C cont 320, D root-value 320, D cont 320, D severe 320.
- `n_teacher_retention_drawn=480`, `calib_policy_ce_avg_iter=4.1862`, `calib_policy_kl_est_avg_iter=1.0937`.
- Strict verifier passed: only `value_head.*` and `encoder.blocks.5.*` trainable tensors changed; all frozen tensors byte-identical.

Gate results:
- A black pre-drop: mean +0.095, over 36.7%, severe 30.0% — **FAIL**.
- B goal-line: mean −0.310, over 11.1%, severe 0.0% — **PASS**.
- C old post-opening: mean +0.135, over 33.3%, severe 23.3% — **FAIL**.
- D red pre-drop: mean +0.043, over 23.3%, severe 13.3% — **FAIL**.

Decision: **REJECT.** No promotion match.

Lesson: v10b fixed B but broke A/C/D. The extra broad B/D pressure did not cleanly solve the remaining blockers; it pushed many previously-safe rows upward and increased search confidence/top1 concentration. Examples:
- A: `black_loss_game_000291` −0.1924 → +0.6422, top1 0.2625 → 0.7775; `black_loss_game_000347` +0.0227 → +0.7694, top1 0.3175 → 0.9325.
- C: `game_000103` −0.3307 → +0.6406, top1 0.2600 → 0.9625; `game_000433` +0.0955 → +0.6995, top1 0.9625 → 0.9875.
- D: `red_loss_game_000362` +0.2579 → +0.6965; `red_loss_game_000578` +0.1619 → +0.5492; `red_loss_game_000780` +0.3815 → +0.5407.

Conclusion: **broad schedule-count pressure is exhausted.** The correct branch point, if continuing, is v10, not v10b. The only disciplined continuation is a surgical manifest edit with value-only rows for the specific v10 blockers (e.g. B `game_000015_ply_19`, possibly B `game_000327_ply_63`, and D `red_loss_game_000752_predrop_ply_70_drop_72`) while keeping the v10 schedule and avoiding any new policy-CE pressure.

## v11 — surgical B value-only root clones (RUN + REJECTED 2026-07-05)

### v11 manifest / training setup

Goal: test whether v10's B blocker was caused by B root-policy CE or missing value-only B root preservation. This was intentionally **manifest-only**, not trainer code and not new MCTS/inference.

Manifest script: `scripts/GPU/alphazero/build_v11_surgical_root_value_manifest.py` created `logs/eval/targeted_calibration_v11_surgical_root_value_from_v10_nearmiss.csv` from the v7 manifest by appending two depth-0 `goal_line_root_value_retention` clones:

- `game_000015_ply_19__root_value` with `teacher_value=0.0469`.
- `game_000327_ply_63__root_value` with `teacher_value=-0.8036`.

Both loaded through `calibration_pool` as `loss_mode=searched_continuation_retention`, `has_policy_target=False`, proving they are value-only and the copied SHA/root reconstruction path is valid.

Training setup: same v10 final-block mechanics (`--freeze-batchnorm-stats --train-value-head-and-final-block`) and a v10-shaped schedule, but with B root-policy CE replaced by B value-only root clones:

`black_predrop_correction=2,goal_line_root_value_retention=2,goal_line_continuation_retention=1,old_post_opening_retention=1,old_post_opening_continuation_retention=2,red_predrop_root_value_retention=1,red_predrop_continuation_retention=2,red_predrop_severe_root_correction=1`

Telemetry/verifier:
- `calib_n_drawn_total=1920`, `calib_n_drawn_per_step=12.0`.
- Draws by tag: A 320, B root-value 320, B continuation 160, C root 160, C continuation 320, D root-value 160, D continuation 320, D severe 160.
- `n_teacher_retention_drawn=160` (down from v10's 320), proving only C root policy-CE rows remained active; B root rows were value-only.
- `calib_policy_ce_avg_iter=2.9096`, `calib_policy_kl_est_avg_iter=1.1022`.
- Strict verifier passed: only `value_head.*` and `encoder.blocks.5.*` trainable tensors changed; all frozen tensors byte-identical.

Gate results:
- A black pre-drop: mean −0.039, over 30.0%, severe 13.3% — **PASS**.
- B goal-line: mean −0.060, over 22.2%, severe 16.7% — **FAIL**, worse than v10.
- C old post-opening: mean +0.058, over 23.3%, severe 20.0% — **FAIL**.
- D red pre-drop: mean −0.109, over 30.0%, severe 6.7% — **FAIL**.

Decision: **REJECT.** No promotion match.

Lesson: v11 did not isolate/fix B. Replacing B root-policy CE with B value-only root clones made B worse and also lost C/D. Therefore v10's B failure was **not** simply caused by B root-policy CE or missing value-only B root preservation. This closes the v10/v11 schedule/manifest branch: existing root/continuation/root-value/severe-row levers cannot safely pass A/B/C/D under final-block training.

## v12 — asymmetric one-sided guardrail hinge (RUN + REJECTED 2026-07-06)

### v12 setup

Goal: introduce a new objective rather than another root/continuation schedule variant. v12 added `loss_mode=asymmetric_guardrail_retention`: a value-only, one-sided hinge that penalizes only candidate drift more pro-black than BASE by more than a margin. The hinge is computed in black perspective via a per-row sign: `relu(sign * (cb_values - cb_targets) - margin)^2`, where `sign=+1` for black-to-move and `-1` for red-to-move.

Manifest: `logs/eval/targeted_calibration_v12_guardrail_from_calib020_0001.csv` (136 rows: 58 hard-value rows + 78 B/C/D root guardrail clones). Guardrail tags: `goal_line_guardrail_retention` 18, `old_post_opening_guardrail_retention` 30, `red_predrop_guardrail_retention` 30.

Training setup: canonical calibration harness from the prior branches (`--iterations 1`, `--lr 0.0003`, `--curriculum-sizes 24`, `--games-per-iter 100`, `--simulations 400`, `--max-moves 280`, `--mcts-eval-batch-size 14`, `--mcts-pending-virtual-visits 8`, `--mcts-stall-flush-sims 48`, `--n-workers 10`, resign/adjudication settings, `--max-positions-per-game 280`) with `--freeze-batchnorm-stats --train-value-head-and-final-block`, target `-0.35`, weight `0.01`, and `--guardrail-margin 0.10`.

Schedule:

`black_predrop_correction=2,goal_line_guardrail_retention=1,old_post_opening_guardrail_retention=2,red_predrop_guardrail_retention=2`

Telemetry/verifier:
- `active_size=24`, `max_moves=280`, `games_per_iter=100`, `mcts_eval_batch_size=14`, `mcts_stall_flush_sims=48`.
- `calib_n_drawn_total=1120`, `calib_n_drawn_per_step=7.0`.
- Draws: A 320, B root guardrail 160, C root guardrail 320, D root guardrail 320.
- `guardrail_hinge_loss=0.02048`, `guardrail_active_frac=0.225`, `guardrail_margin=0.1`.
- `n_teacher_retention_drawn=0`, `calib_policy_ce_avg_iter=0.0`, `calib_policy_kl_est_avg_iter=0.0`, proving no policy CE / teacher-retention path was active.
- Strict verifier passed: only `value_head.*` and `encoder.blocks.5.*` trainable tensors changed; all frozen tensors byte-identical.

### v12 gate results

- A black pre-drop: mean +0.005, over 20.0%, severe 13.3% — **FAIL by mean only** (pass requires mean ≤ 0.0).
- B goal-line: mean −0.214, over 5.6%, severe 0.0% — **PASS**.
- C old post-opening: mean +0.057, over 23.3%, severe 16.7% — **FAIL by severe**.
- D red pre-drop: mean −0.088, over 23.3%, severe 3.3% — **FAIL by severe**.

Decision: **REJECT.** No promotion match.

Lesson: v12 is the strongest evidence so far that the **objective shape matters**. The one-sided hinge fixed B cleanly without policy CE and avoided the v10/v11 B failure mode. However, root-only guardrails were not enough for C/D: C still showed a broad severe repeat-offender cluster (`game_000505`, `game_000565`, `game_000619`, `game_000065`, `game_000433`, etc.), and D had one severe row (`red_loss_game_000362...` +0.5257) plus many elevated non-severe over rows. A nearly passed by aggregate mean but still had severe top rows. The next branch should keep the v12 objective and add searched-continuation guardrail states for C/D before escalating to gradient projection.

## v12b — continuation guardrail rows (RUN + REJECTED 2026-07-06)

v12b reused the v12 objective unchanged and extended the guardrail state coverage. It did **not** change `trainer.py`, add a loss mode, add a CLI flag, change gates, or implement gradient projection.

Implementation:
- Merged/pushed on `origin/main` at `7335605` after v12 (`2cc4bd1`).
- Authoritative suite on merged main: 1354 passed, 0 failures.
- Loader change: in `build_calibration_position`, a guardrail row with non-empty `extra_moves_json` now walks `_apply_extra_moves`, so the hinge applies to the searched continuation board instead of the root. Root guardrail rows with blank `extra_moves_json` remain v12-compatible.
- Builder: `scripts/GPU/alphazero/build_v12b_continuation_guardrail_manifest.py`.
- Smoke: `scripts/GPU/alphazero/smoke_v12b_continuation_guardrail.py`.

Manifest:
- `logs/eval/targeted_calibration_v12b_continuation_guardrail_from_calib020_0001.csv`
- 353 rows total: 58 hard-value, 78 root guardrail, 217 continuation guardrail.
- Counts: `goal_line_guardrail_retention=18`, `old_post_opening_guardrail_retention=30`, `red_predrop_guardrail_retention=30`, `old_post_opening_continuation_guardrail_retention=90`, `red_predrop_continuation_guardrail_retention=127`.
- Schema loaded as `asymmetric_guardrail_retention`.
- Smoke passed: `guardrail_hinge_loss=0.138`, `active_frac=0.429`, `guardrail_n=7`.

Training setup:
- Checkpoint: `checkpoints/alphazero-v12b-continuation-guardrail-from-calib020-0001/model_iter_0001.safetensors`.
- Canonical 24x24 harness, loaded `calib020_0001`, `--guardrail-margin 0.10`, `--freeze-batchnorm-stats`, `--train-value-head-and-final-block`.
- Schedule: `black_predrop_correction=2,goal_line_guardrail_retention=1,old_post_opening_guardrail_retention=1,old_post_opening_continuation_guardrail_retention=2,red_predrop_guardrail_retention=1,red_predrop_continuation_guardrail_retention=2`.

Telemetry/verifier:
- `active_size=24`, `max_moves=280`, `games_per_iter=100`, `mcts_eval_batch_size=14`, `mcts_stall_flush_sims=48`.
- `calib_n_drawn_total=1440`, `calib_n_drawn_per_step=9.0`.
- Draws by tag: A 320, B root guardrail 160, C root guardrail 160, C continuation guardrail 320, D root guardrail 160, D continuation guardrail 320.
- `guardrail_hinge_loss=0.01855`, `guardrail_active_frac=0.299`, `guardrail_margin=0.1`.
- `n_teacher_retention_drawn=0`, `calib_policy_ce_avg_iter=0.0`, `calib_policy_kl_est_avg_iter=0.0`.
- Strict verifier passed: only `value_head.*` and `encoder.blocks.5.*` trainable tensors changed; all frozen tensors byte-identical.

Gate results:
- A black pre-drop: mean −0.137, over 30.0%, severe 13.3% — **PASS**.
- B goal-line: mean −0.302, over 5.6%, severe 5.6% — **FAIL** by severe.
- C old post-opening: mean +0.028, over 33.3%, severe 23.3% — **FAIL** by severe.
- D red pre-drop: mean −0.093, over 40.0%, severe 16.7% — **HARD FAIL**.

Failure-row readout:
- B had one severe row, `game_000299_ply_39` at +0.5807. The old v10/v11 blocker `game_000015_ply_19` was no longer severe, so B remains fragile but not fixed by targeting one historical blocker.
- C severe rows were dominated by stable repeat offenders: `game_000505_ply_037`, `game_000565_ply_033`, `game_000619_ply_061`, `game_000433_ply_029`, `game_000065_ply_021`, `game_000309_ply_019`, plus other high rows.
- D severe rows were broad/diffuse: `red_loss_game_000176`, `000278`, `000780`, `000456`, and `000438` were all severe/high. This matches the historical D pattern as broad value-surface drift, not a single fixed blocker.

Decision: **REJECT.** No promotion match.

Lesson: v12b falsifies the "root-only coverage was the main remaining problem" hypothesis. The one-sided hinge is useful, but adding searched-continuation guardrail coverage for C/D still did not protect B/C/D under value-head + final-block training. The remaining failure is more consistent with **gradient conflict in the shared final block**: A correction needs the final block to move, but the same update directions can still increase guardrail overvalue even when one-sided root and continuation guardrails are present.

## v13 — asymmetric gradient-conflict projection (RUN + REJECTED 2026-07-08)

v13 kept the useful v12b objective/state coverage and changed update mechanics. The core idea was to split calibration minibatch gradients into A correction and guardrail hinge pieces on the applied trainable surface (`value_head.*` + final residual block `encoder.blocks.5.*`). When the A gradient conflicted with the guardrail gradient (`dot(g_A,g_G)<0`), v13 projected the A component away from the guardrail direction before applying the combined surface update.

Setup:
- Base: `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors` (`calib020_0001`).
- Manifest: `logs/eval/targeted_calibration_v12b_continuation_guardrail_from_calib020_0001.csv`.
- Schedule: `black_predrop_correction=2,goal_line_guardrail_retention=1,old_post_opening_guardrail_retention=1,old_post_opening_continuation_guardrail_retention=2,red_predrop_guardrail_retention=1,red_predrop_continuation_guardrail_retention=2`.
- Training surface: `--freeze-batchnorm-stats --train-value-head-and-final-block --post-opening-calibration-gradient-projection`.
- Guardrail margin: 0.10.

Telemetry note: the first v13 checkpoint had projection fields missing from the flattened `model_iter_0001.json`. The code path was present and gates were valid, but conflict-rate/removed-norm could not be read. After fixing the two-site telemetry flattening mirror, v13 was rerun unchanged as `checkpoints/alphazero-v13-gradient-projection-telemetryfix-from-calib020-0001/model_iter_0001.safetensors`.

Telemetry/verifier for the telemetry-fixed run:
- `calib_projection_enabled=True`, `calib_projection_scope=value_head_and_final_block`.
- `calib_projection_conflict_steps=35`, `calib_projection_conflict_rate=0.2846`.
- `calib_projection_removed_norm_avg=0.0903`, `calib_projection_guardrail_grad_norm_avg=4.0431`, `calib_projection_a_grad_norm_avg=13.4087`.
- `guardrail_hinge_loss=0.01999`, `guardrail_active_frac=0.2339`, `guardrail_margin=0.1`.
- `calib_n_drawn_total=1440`, `calib_n_drawn_per_step=9.0`.
- Strict verifier passed: only `value_head.*` and `encoder.blocks.5.*` trainable tensors changed; all frozen tensors byte-identical.

Gate results, telemetry-fixed run:
- A black pre-drop: mean −0.117, over 23.3%, severe 20.0% — **PASS**.
- B goal-line: mean −0.343, over 5.6%, severe 0.0% — **PASS**.
- C old post-opening: mean −0.083, over 26.7%, severe 16.7% — **FAIL** by severe.
- D red pre-drop: mean −0.151, over 36.7%, severe 13.3% — **FAIL**.

Decision: **REJECT.** No promotion match.

Lesson: projection engaged and was directionally useful: A/B passed and C/D improved in shape versus v12b. But C/D still had severe failures, so projection strength or protection needed a cleaner follow-up. The telemetry fix is mandatory for all later projection branches.

## v13b — projection with lower guardrail margin 0.05 (RUN + REJECTED 2026-07-08)

v13b was an arguments-only test of whether v13 failed because the guardrail hinge was not active enough. It kept v13 mechanics and schedule but changed `--guardrail-margin 0.10` to `--guardrail-margin 0.05`.

Telemetry/verifier:
- `guardrail_margin=0.05`, `guardrail_hinge_loss=0.02263`, `guardrail_active_frac=0.2857`.
- `calib_projection_strength` did not exist yet; projection used v13 strength 1.0.
- `calib_projection_conflict_steps=57`, `calib_projection_conflict_rate=0.4161`.
- `calib_projection_removed_norm_avg=0.0991`.
- Strict verifier passed: only `value_head.*` and `encoder.blocks.5.*` trainable tensors changed.

Gate results:
- A black pre-drop: mean −0.017, over 36.7%, severe 20.0% — **PASS**, but weaker than v13.
- B goal-line: mean −0.370, over 11.1%, severe 0.0% — **PASS**.
- C old post-opening: mean −0.063, over 26.7%, severe 16.7% — **FAIL** by severe.
- D red pre-drop: mean −0.203, over 23.3%, severe 16.7% — **FAIL**, worse by severe.

Failure-row readout:
- A retained 19 severe rows, including high rows such as `black_loss_game_000281...` +0.8451 and `black_loss_game_000611...` +0.8037.
- C stayed in the same repeat-offender family (`game_000505`, `000565`, `000619`, `000433`, `000499`, `000065`, `000369`).
- D severe unique rose to 5 rows: `red_loss_game_000362`, `000728`, `000172`, `000176`, `000780`.

Decision: **REJECT.** No promotion match.

Lesson: lowering the margin increased guardrail/projection activity, but did not solve C/D and weakened A. This closes global margin-tightening as a projection fix.

## v13c — projection-strength scalar 2.0 (RUN + REJECTED 2026-07-08)

v13c added one CLI arg: `--post-opening-calibration-projection-strength`. The implementation folds the scalar into the effective projection weight while keeping the helper signature and geometric conflict primitive unchanged:

```
effective_projection_weight = post_opening_calibration_projection_strength * calibration_loss_weight
project_conflicting_gradient(..., weight=effective_projection_weight)
```

The test used `--post-opening-calibration-projection-strength 2.0`, restored `--guardrail-margin 0.10`, and otherwise kept the v13/v12b manifest, schedule, trainable surface, and projection mechanics unchanged.

Telemetry/verifier:
- `guardrail_margin=0.1`, `guardrail_hinge_loss=0.01345`, `guardrail_active_frac=0.2205`.
- `calib_projection_enabled=True`, `calib_projection_scope=value_head_and_final_block`, `calib_projection_strength=2.0`.
- `calib_projection_conflict_steps=46`, `calib_projection_conflict_rate=0.368`.
- `calib_projection_removed_norm_avg=0.1292`, up from v13's 0.0903, proving the stronger correction increased actual applied projection magnitude.
- `calib_projection_guardrail_grad_norm_avg=2.5302`, `calib_projection_a_grad_norm_avg=8.7761`.
- Strict verifier passed: only `value_head.*` and `encoder.blocks.5.*` trainable tensors changed.

Gate results:
- A black pre-drop: mean −0.052, over 13.3%, severe 10.0% — **PASS**.
- B goal-line: mean −0.243, over 0.0%, severe 0.0% — **PASS**.
- C old post-opening: mean −0.076, over 6.7%, severe 6.7% — **PASS**.
- D red pre-drop: mean −0.073, over 16.7%, severe 3.3% — **FAIL** by one severe row.

Failure-row readout:
- Only D severe: `red_loss_game_000728_predrop_ply_48_drop_50` at +0.5441 (3 duplicate raw severe rows in the probe CSV).
- Next D rows were non-severe: `red_loss_game_000362...` +0.4838, `red_loss_game_000752...` +0.3901, `red_loss_game_000176...` +0.3302, `red_loss_game_000780...` +0.3102.

Decision: **REJECT by strict gate.** No promotion match.

Lesson: v13c is the best projection result and proves the projection-strength mechanism helped: A/B/C all passed and D missed by one barely-severe row. But D's gate requires `severe=0.0%`, so v13c cannot be promoted. The only justified continuation is a tightly scoped cleanup of D root guardrail sampling, not blind strength/margin tuning.

## v13d — v13c plus D root guardrail draw 2 (RUN + REJECTED 2026-07-08)

v13d was the only justified argument-only cleanup run after v13c. It kept:
- Base `calib020_0001`.
- v12b manifest.
- `projection_strength=2.0`.
- `guardrail_margin=0.10`.
- `--freeze-batchnorm-stats --train-value-head-and-final-block --post-opening-calibration-gradient-projection`.

Only the tag schedule changed:

```
red_predrop_guardrail_retention=1 -> 2
```

Full v13d schedule:

`black_predrop_correction=2,goal_line_guardrail_retention=1,old_post_opening_guardrail_retention=1,old_post_opening_continuation_guardrail_retention=2,red_predrop_guardrail_retention=2,red_predrop_continuation_guardrail_retention=2`

Checkpoint:
- `checkpoints/alphazero-v13d-projection-strength2-red-root-guardrail2-from-calib020-0001/model_iter_0001.safetensors`.

Telemetry/verifier:
- `guardrail_margin=0.1`, `guardrail_hinge_loss=0.023796`, `guardrail_active_frac=0.213281`.
- `calib_projection_enabled=True`, `calib_projection_scope=value_head_and_final_block`, `calib_projection_strength=2.0`.
- `calib_projection_conflict_steps=46`, `calib_projection_conflict_rate=0.3459`.
- `calib_projection_removed_norm_avg=0.1802`, `calib_projection_guardrail_grad_norm_avg=4.3081`, `calib_projection_a_grad_norm_avg=12.6490`.
- `calib_n_drawn_total=1600`, `calib_n_drawn_per_step=10.0`.
- Draws by tag: A 320, B 160, C root 160, C continuation 320, D root 320, D continuation 320.
- Strict verifier passed: only `value_head.*` and `encoder.blocks.5.*` trainable tensors changed; all frozen tensors byte-identical.

Gate results:
- A black pre-drop: mean −0.083, over 26.7%, severe 23.3% — **PASS**, but weakened versus v13c.
- B goal-line: mean −0.265, over 16.7%, severe 5.6% — **FAIL**.
- C old post-opening: mean −0.013, over 33.3%, severe 20.0% — **FAIL**.
- D red pre-drop: mean −0.113, over 33.3%, severe 20.0% — **HARD FAIL**.

Decision: **REJECT.** No promotion match.

Lesson: the one-row D-root cleanup did not clear D. It damaged B/C/D and weakened A relative to v13c. This closes the v13 projection/cleanup line: do not keep tuning root draw pressure, projection strength, margin, or tag schedule inside the final-block projection family without a new written design.

## v14 — gated value-adapter surface, projection OFF (RUN + REJECTED 2026-07-08)

v14 changed the training surface, not the objective. It tested whether a value-only adapter — more capacity than value-head-only but isolated from policy/trunk updates — could move A without the B/C/D guardrail drift caused by final-block training.

Setup:
- Base: `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors` (`calib020_0001`).
- Checkpoint: `checkpoints/alphazero-v14-value-adapter-from-calib020-0001/model_iter_0001.safetensors`.
- Manifest: `logs/eval/targeted_calibration_v12b_continuation_guardrail_from_calib020_0001.csv`.
- Schedule: `black_predrop_correction=2,goal_line_guardrail_retention=1,old_post_opening_guardrail_retention=1,old_post_opening_continuation_guardrail_retention=2,red_predrop_guardrail_retention=1,red_predrop_continuation_guardrail_retention=2`.
- Objective: v12b asymmetric one-sided guardrail hinge, `guardrail_margin=0.10`.
- Projection: OFF.
- Adapter: `--value-adapter --value-adapter-bottleneck-width 32`.
- Training surface: `--train-value-head-and-value-adapter`; encoder, policy head, final residual block, and BN running stats frozen.

Implementation note: the first gate-eval attempt exposed a scoring loader gap — `probe_eval.load_network_for_scoring()` instantiated a no-adapter network and rejected the v14 checkpoint's `value_adapter.*` keys. The fix was to detect `value_adapter.*` keys in the safetensors file and construct `create_network(..., value_adapter=True)` only for adapter checkpoints. This preserved base/v8-v13 loading behavior.

Telemetry/verifier:
- `train_value_head_and_value_adapter=True`.
- `train_value_head_only=False`, `train_value_head_and_final_block=False`.
- `freeze_batchnorm_stats=True`.
- `value_adapter_gate=0.003017987357452512`.
- `value_adapter_grad_norm=0.0013807759423798416`.
- `guardrail_hinge_loss=0.007941251490490764`, `guardrail_active_frac=0.25803572256118057`.
- `calib_n_drawn_total=1440`, `calib_n_drawn_per_step=9.0`.
- Draws by tag: A 320, B 160, C root 160, C continuation 320, D root 160, D continuation 320.
- `calib_value_term_avg_iter=0.0`, `calib_policy_ce_avg_iter=0.0`, `calib_policy_kl_est_avg_iter=0.0`.
- Verifier passed: 92 base tensors compared; shared frozen set byte-identical; only `value_head.*` and `value_adapter.*` changed; `value_adapter.gate` moved to 0.003018.

Gate results:
- A black pre-drop: baseline mean +0.257 / over 50.0% / severe 43.3%; v14 mean +0.064 / over 26.7% / severe 20.0% — **FAIL**, but strongly improved.
- B goal-line: baseline mean −0.244 / over 5.6% / severe 0.0%; v14 mean −0.272 / over 5.6% / severe 0.0% — **PASS**.
- C old post-opening: baseline mean +0.099 / over 33.3% / severe 13.3%; v14 mean +0.063 / over 30.0% / severe 6.7% — **PASS**.
- D red pre-drop: baseline mean −0.188 / over 13.3% / severe 0.0%; v14 mean −0.079 / over 23.3% / severe 0.0% — **PASS by formal gate**, but with degraded mean/overvalue margin.

Decision: **REJECT.** No promotion match.

Lesson: v14 is not an underfit/no-move result. The adapter surface moved A substantially while B/C/D formally held, so value-only adapter capacity is real and safer than final-block training. However, A still failed by mean and D's margin moved toward black even with the trunk/policy/final block frozen. Width 64 is not the next branch because A did move. The next justified branch is **v14b**: same adapter surface, same v12b objective, projection ON over the adapter value surface (`value_head.*` + `value_adapter.*`), bottleneck 32, projection strength 1.0 first.


## v14b — value-adapter surface + gradient projection, strength 1.0 (RUN + REJECTED 2026-07-09)

v14b tested the planned follow-up to v14: keep the value-only adapter surface and v12b objective, but enable the v13 A-yields-to-guardrail projection over `{value_head, value_adapter}` instead of the final residual block.

Setup:
- Base: `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors` (`calib020_0001`).
- Checkpoint: `checkpoints/alphazero-v14b-value-adapter-projection-from-calib020-0001/model_iter_0001.safetensors`.
- Manifest/schedule/objective: same v12b continuation-guardrail manifest, tag schedule, asymmetric one-sided guardrail hinge, `guardrail_margin=0.10`.
- Adapter: `--value-adapter --value-adapter-bottleneck-width 32`.
- Training surface: `--train-value-head-and-value-adapter`; encoder, policy head, final residual block, and BN running stats frozen.
- Projection: `--post-opening-calibration-gradient-projection`, default `--post-opening-calibration-projection-strength 1.0`.

Implementation notes:
- The first v14b attempt failed at training step 0 because the projection guard still rejected the adapter surface. The implemented fix allowed projection with `--train-value-head-and-value-adapter`, selected the surface `{value_head, value_adapter}`, and left `project_conflicting_gradient` unchanged.
- Slot `[13]` telemetry was made self-describing: dict = projection telemetry (v13/v14b), float = v14 adapter grad norm only. Under v14b, `value_adapter_grad_norm` is folded into the projection dict and is the post-projection/applied adapter grad norm.
- The first completed v14b run exposed a telemetry-label bug: `calib_projection_scope` still said `value_head_and_final_block`. The tensor-diff verifier proved this was a label bug, not a surface leak; the final block stayed byte-identical. The label was fixed for subsequent runs.

Telemetry/verifier:
- `train_value_head_and_value_adapter=True`, `train_value_head_and_final_block=False`, `freeze_batchnorm_stats=True`.
- `value_adapter_gate=0.0016666483134031296`.
- `value_adapter_grad_norm=0.0015184665251581464`.
- `calib_projection_enabled=True`, `calib_projection_conflict_steps=51`, `calib_projection_conflict_rate=0.3984375`, `calib_projection_removed_norm_avg=0.07273300471135768`.
- `calib_projection_guardrail_grad_norm_avg=0.7594021144822769`, `calib_projection_a_grad_norm_avg=9.11618999313071`.
- `guardrail_hinge_loss=0.0032312683621622236`, `guardrail_active_frac=0.22500000698491932`.
- `calib_n_drawn_total=1440`, `calib_n_drawn_per_step=9.0`; draws by tag: A 320, B 160, C root 160, C continuation 320, D root 160, D continuation 320.
- Verifier passed: shared frozen set byte-identical; only `value_head.*` and `value_adapter.*` changed; `value_adapter.gate` moved to 0.001667.

Gate results:
- A black pre-drop: baseline mean +0.257 / over 50.0% / severe 43.3%; v14b mean +0.026 / over 26.7% / severe 16.7% — **FAIL**, but closer than v14.
- B goal-line: baseline mean −0.244 / over 5.6% / severe 0.0%; v14b mean −0.254 / over 11.1% / severe 0.0% — **PASS** at the over cap.
- C old post-opening: baseline mean +0.099 / over 33.3% / severe 13.3%; v14b mean +0.044 / over 23.3% / severe 6.7% — **PASS**.
- D red pre-drop: baseline mean −0.188 / over 13.3% / severe 0.0%; v14b mean −0.047 / over 23.3% / severe 0.0% — **PASS** by formal gate, with degraded mean margin.

Decision: **REJECT / near-pass.** No promotion match because A requires mean ≤ 0.0 and v14b remained positive at +0.026.

Lesson: v14b is the best adapter result so far. Projection helped A relative to v14 (`+0.064 → +0.026`) and B/C/D remained inside formal gates, so the adapter-projection mechanism is directionally correct. Because the only remaining blocker is A mean and projection did not break formal guardrails, the disciplined next step is **v14c**: same setup with `--post-opening-calibration-projection-strength 2.0`. Do not widen the adapter, change margin, or redesign the objective before v14c completes.

## Retired hypothesis — v6 searched-continuation/PV retention

**Working shape (2026-07-02, pre-design):** stop anchoring only the fragile root positions; anchor what search actually visits beneath them.

- **Retention rows:** for each fragile B/C/D row, run BASE MCTS (gate-faithful, 400 sims) and extract **child/PV states** (the searched continuations whose values drifted in v5). Each extracted state becomes its own retention row with a **raw teacher value** anchor (BASE eval-mode forward at that state — the v4/v5 value mechanism that provably holds). **Policy retention only where the visit distribution is sharp** (diffuse D-row distributions gave weak/noisy targets in v5).
- **Correction rows:** unchanged A hard-value family.
- **Manifest encoding:** prefer an `extra_moves_json` column on continuation rows. Reconstruct the source replay prefix with `replay_path + position_ply`, then apply `extra_moves_json` to reach the continuation state. Avoid new sidecar replay files unless `extra_moves_json` becomes too invasive or brittle.
- **State selection:** start conservative. For sharp C rows, keep the fragile root plus the top BASE child / PV line to depth 2–3. For diffuse D rows, add top-k BASE children only where no single PV dominates. Do not extract every visited child until row counts and dilution risk are understood.
- **Training:** should ride the existing retention machinery (`teacher_retention`-style rows over the masked 14-tuple path) — continuation rows are just additional positions.
- **Tags / schedule:** do not hide continuation rows under old root-retention tags. Use separate continuation tags so they can be scheduled and audited independently. Starting schedule candidate: `black_predrop_correction=2,goal_line_root_retention=1,old_post_opening_continuation_retention=2,red_predrop_continuation_retention=2`; adjust only after the builder reports final row counts and tag mass.
- **Gate:** same A/B/C/D probes vs `calib020_0001`. No promotion unless all four pass.


## Code / artifact pointers

- **v2** manifest builder + mixed-pool weighted loss: `scripts/GPU/alphazero/build_targeted_calibration_manifest.py`; operator guide `docs/post-game-analysis.md` §6.
- **v3** tag-stratified sampling: `--post-opening-calibration-tag-schedule` (commits `0c122cb` / `0e0fd24` / `282998d` / `b27d60b` on `main`); telemetry `state.calib_n_drawn_by_tag` + sidecar `post_opening_calibration.draws_by_tag`; operator guide `docs/post-game-analysis.md` §6 (tag-stratified block).
- **v4** teacher-retention builder/smoke/training path: `scripts/GPU/alphazero/build_teacher_calibration_manifest.py`, `scripts/GPU/alphazero/smoke_teacher_calibration_v4.py`, `--post-opening-calibration-teacher-value-weight`, `--post-opening-calibration-teacher-policy-kl-weight`, and `--freeze-batchnorm-stats`.
- **v3-frozenBN-control checkpoint/gates:** `checkpoints/alphazero-v3-frozenBN-control-from-calib020-0001/model_iter_0001.safetensors`, `logs/eval/v3_frozenBN_control_from_calib020_0001_gates_400s/`.
- **v4/v3F severe-overlap review:** `logs/eval/v3f_v4_severe_overlap_review.csv`.
- **Raw-NN focus-row diagnostic:** `scripts/GPU/alphazero/eval_raw_nn_position_rows.py` (+ `tests/test_eval_raw_nn_position_rows.py`, merged to main @ `7064621`); output `logs/eval/v3f_v4_raw_nn_focus_rows.csv`; plan `docs/superpowers/plans/2026-07-01-eval-raw-nn-position-rows-diagnostic.md`.
- **v5 root-retention:** builder `scripts/GPU/alphazero/build_mcts_root_retention_manifest.py` (+ `--gate-checkpoint-label` cross-check), smoke `scripts/GPU/alphazero/smoke_mcts_root_retention_v5.py`, `loss_mode=mcts_root_retention` in `calibration_pool.py`; manifest `logs/eval/targeted_calibration_v5_mcts_root_from_calib020_0001.csv`; checkpoint `checkpoints/alphazero-v5-mcts-root-from-calib020-0001/model_iter_0001.safetensors`; plan `docs/superpowers/plans/2026-07-01-targeted-value-calibration-v5-mcts-root-retention.md`; operator guide `docs/post-game-analysis.md` §8.
- **v6/v6c searched-continuation retention:** builder `scripts/GPU/alphazero/build_searched_continuation_retention_manifest.py`, smoke `scripts/GPU/alphazero/smoke_searched_continuation_retention_v6.py`, `loss_mode=searched_continuation_retention` in `calibration_pool.py`; manifests `logs/eval/targeted_calibration_v6_continuation_from_calib020_0001.csv` and `logs/eval/targeted_calibration_v6c_d_root_value_only_from_calib020_0001.csv`.
- **v7 severe-D hard correction:** manifest-only branch `logs/eval/targeted_calibration_v7_severe_d_root_correction_from_calib020_0001.csv`; severe-D rows selected via `scripts/GPU/alphazero/eval_raw_nn_position_rows.py`; no loader/trainer changes.
- **v8 value-head-only:** CLI flag `--train-value-head-only` in `scripts/GPU/alphazero/train.py` / guard in `scripts/GPU/alphazero/trainer.py`; verifier `scripts/GPU/alphazero/verify_value_head_only_checkpoint.py`; checkpoint `checkpoints/alphazero-v8-value-head-only-v7-manifest-from-calib020-0001/model_iter_0001.safetensors`.
- **v9 value head + final block:** CLI flag `--train-value-head-and-final-block` in `scripts/GPU/alphazero/train.py` / three-way update branch in `scripts/GPU/alphazero/trainer.py`; verifier `scripts/GPU/alphazero/verify_value_head_and_final_block_checkpoint.py`; checkpoint `checkpoints/alphazero-v9-value-head-and-final-block-v7-manifest-from-calib020-0001/model_iter_0001.safetensors`; gates `logs/eval/v9_value_head_and_final_block_v7_manifest_from_calib020_0001_gates_400s`.
- **v10/v10b final-block schedule-only branches:** same v9 flag/verifier and v7 manifest; checkpoints `checkpoints/alphazero-v10-final-block-root-plus-cont-v7-manifest-from-calib020-0001/model_iter_0001.safetensors` and `checkpoints/alphazero-v10b-final-block-root-plus-cont-stronger-bd-v7-manifest-from-calib020-0001/model_iter_0001.safetensors`; gates `logs/eval/v10_final_block_root_plus_cont_v7_manifest_from_calib020_0001_gates_400s` and `logs/eval/v10b_final_block_root_plus_cont_stronger_bd_v7_manifest_from_calib020_0001_gates_400s`; v10 design spec `docs/superpowers/specs/2026-07-04-targeted-value-calibration-v10-final-block-root-continuation-schedule-design.md`.
- **v11 surgical B value-only root clones:** manifest-copy script `scripts/GPU/alphazero/build_v11_surgical_root_value_manifest.py`; manifest `logs/eval/targeted_calibration_v11_surgical_root_value_from_v10_nearmiss.csv`; checkpoint `checkpoints/alphazero-v11-b-root-value-surgical-v10-schedule-from-calib020-0001/model_iter_0001.safetensors`; gates `logs/eval/v11_b_root_value_surgical_v10_schedule_from_calib020_0001_gates_400s`.
- **v12 asymmetric one-sided guardrail hinge:** loss mode `asymmetric_guardrail_retention` in `scripts/GPU/alphazero/calibration_pool.py`; hinge/sign/13-tuple path in `scripts/GPU/alphazero/trainer.py`; CLI `--guardrail-margin` in `scripts/GPU/alphazero/train.py`; builder `scripts/GPU/alphazero/build_v12_guardrail_manifest.py`; smoke `scripts/GPU/alphazero/smoke_asymmetric_guardrail_v12.py`; manifest `logs/eval/targeted_calibration_v12_guardrail_from_calib020_0001.csv`; checkpoint `checkpoints/alphazero-v12-asymmetric-guardrail-hinge-from-calib020-0001/model_iter_0001.safetensors`; gates `logs/eval/v12_asymmetric_guardrail_hinge_from_calib020_0001_gates_400s`.
- **v12b continuation guardrails:** loader gate in `scripts/GPU/alphazero/calibration_pool.py` for `asymmetric_guardrail_retention` rows with non-empty `extra_moves_json`; builder `scripts/GPU/alphazero/build_v12b_continuation_guardrail_manifest.py`; smoke `scripts/GPU/alphazero/smoke_v12b_continuation_guardrail.py`; manifest `logs/eval/targeted_calibration_v12b_continuation_guardrail_from_calib020_0001.csv`; checkpoint target `checkpoints/alphazero-v12b-continuation-guardrail-from-calib020-0001`; gates target `logs/eval/v12b_continuation_guardrail_from_calib020_0001_gates_400s`.
- **v13 gradient-conflict projection:** CLI `--post-opening-calibration-gradient-projection`; projection path in `scripts/GPU/alphazero/trainer.py`; projection telemetry in both `trainer.py` flattened row and `calibration_pool.py` sidecar; smoke `scripts/GPU/alphazero/smoke_v13_gradient_projection.py`; telemetry-fixed checkpoint `checkpoints/alphazero-v13-gradient-projection-telemetryfix-from-calib020-0001/model_iter_0001.safetensors`; gates `logs/eval/v13_gradient_projection_telemetryfix_from_calib020_0001_gates_400s`.
- **v13b margin 0.05:** arg-only projection branch; checkpoint `checkpoints/alphazero-v13b-gradient-projection-margin005-from-calib020-0001/model_iter_0001.safetensors`; gates `logs/eval/v13b_gradient_projection_margin005_from_calib020_0001_gates_400s`.
- **v13c projection-strength scalar:** CLI `--post-opening-calibration-projection-strength`; effective projection weight folds in `projection_strength * calibration_loss_weight`; mandatory telemetry `calib_projection_strength` in sidecar + flattened row; design `docs/superpowers/specs/2026-07-07-targeted-value-calibration-v13c-projection-strength-design.md`; checkpoint `checkpoints/alphazero-v13c-projection-strength-from-calib020-0001/model_iter_0001.safetensors`; gates `logs/eval/v13c_projection_strength2_from_calib020_0001_gates_400s`.
- **v13d red-root cleanup:** arg-only v13c cleanup with `red_predrop_guardrail_retention=2`; checkpoint `checkpoints/alphazero-v13d-projection-strength2-red-root-guardrail2-from-calib020-0001/model_iter_0001.safetensors`; gates `logs/eval/v13d_projection_strength2_red_root_guardrail2_from_calib020_0001_gates_400s`; rejected and closes the v13 projection/cleanup line.
- **v14 adapter line:** value-adapter surface (`value_head.*` + `value_adapter.*`) with scalar gate/bottleneck 32; v14b best near-pass checkpoint `checkpoints/alphazero-v14b-value-adapter-projection-from-calib020-0001/model_iter_0001.safetensors`; v14c/v14d rejected and close cleanup.
- **v15 raw/MCTS + continuation diagnostics:** raw A/D drift CSV `logs/eval/v15prep_raw_AD_drift_base_v14b_v14d.csv`; Phase 0 concentration script `scripts/GPU/alphazero/diagnose_v15_a_continuation_concentration.py` output `logs/eval/v15prep_a_continuation_concentration.csv`; Phase 0.5 subtree script `scripts/GPU/alphazero/diagnose_v15_a_selected_branch_subtrees.py` outputs `logs/eval/v15prep_a_selected_branch_subtrees.csv` and `logs/eval/v15prep_a_selected_branch_subtrees_by_depth_summary.csv`; closure summary files `logs/eval/v15prep_a_phase05_*.csv` / `.txt`; decision: no v15 Phase 1.
- **v16/v16a search-reliability diagnostics:** FPU hook and sweep `scripts/GPU/alphazero/diagnose_fpu_sweep.py`; neutral-manifest builder `scripts/GPU/alphazero/build_v16a_neutral_position_manifest.py`; manifest/meta `logs/eval/v16a_fpu_unbiased/neutral_position_manifest.csv` and `.meta.json`; held-out outputs `neutral_fpu_sweep_cases.csv`, `neutral_fpu_sweep_summary.csv`, `neutral_fpu_sweep_by_stratum.csv`, and `operator_sweep.log` in the same directory; design `docs/superpowers/specs/2026-07-10-v16a-unbiased-fpu-validation-design.md`; result: fixed absolute `−0.20` rejected by the late-stratum collapse gate.
- **v16 context-relative policy-mass successor:** rule/observer/diagnostic `scripts/GPU/alphazero/mcts.py` + `scripts/GPU/alphazero/diagnose_fpu_policy_mass.py`; phase-primary corpus `scripts/GPU/alphazero/fpu_dev_corpus_v2.py`; immutable reservoir protocol/qualification `scripts/GPU/alphazero/fpu_dev_reservoir_protocol.py`; production protocol root `logs/eval/fpu_v16_policy_mass_v2/reservoir_v1/`; design `docs/superpowers/specs/2026-07-10-context-relative-fpu-policy-mass-design.md`; corpus design `docs/superpowers/specs/2026-07-12-fpu-dev-corpus-v2-phase-design.md`; qualification design `docs/superpowers/specs/2026-07-14-fpu-v2-reservoir-protocol-qualification-design.md`; operator guide `docs/post-game-analysis.md` §11; status: reservoir protocol v1 RUNNING, no result yet.

- **Plans:** `docs/superpowers/plans/2026-06-24-targeted-value-calibration-v2.md`, `docs/superpowers/plans/2026-06-25-targeted-value-calibration-v3-tag-stratified-sampling.md`, `docs/superpowers/plans/2026-06-29-targeted-value-calibration-v4-teacher-retention.md`.

## v16 policy-mass successor — reservoir v1 POST-SCREEN GATE-FAIL + role-feasibility repair (2026-07-18/19)

**Outcome of reservoir protocol v1 (updates the RUNNING status above): the 4,800-game reservoir generated and passed `qualify` + `screen`, but `select` GATE-FAILED post-screen.** Kept target capacity by phase: **opening 0 / early_mid 0 / midgame 0 / late 136** (155 kept late-target rows across 86 games under the ≤2/game rule) vs the 45-per-phase demand of the original 240-row allocation. Target geometry is late-only on this net. **No FPU coefficient was tested; no A/B/C/D or strength result exists.** reservoir_v1 artifacts are preserved untouched as immutable discovery evidence.

**Role-feasibility repair (branch `fpu-v2-role-feasibility-repair`, 22 commits fca9c0d→da4dc00; plan `docs/superpowers/plans/2026-07-18-fpu-v2-role-feasibility-repair.md`):** schema-2 config-authoritative `AllocationProfile` (late-only targets), controlled `post-screen-qualify` stage (PASS = exact-selector witness, never capacity bounds), protocol v2 + `run_kind` production/smoke isolation, per-split late-target band minima, authenticated discovery commands, `historical_screen_discovery_v1` identity policy (producer vs analyzer provenance; strict stages unchanged). Suite 2262/0; schema-1 byte-identity pinned by pre-repair goldens. Whole-branch review: ready to merge (0 Critical / 0 Important).

**Task 14 zero-GPU proof on the immutable v1 screen (final rerun r2 under source bytes @ `da4dc00`, clean tree):**

- **Production profile (120 rows) exact-selector PASS, exit 0, deterministic** (r1 pre-cell-order-fix run byte-identical on witness/qualification/profile; superseded r1 reports preserved on disk).
- Final frozen allocation: `target|late` 40 tuning + 20 frozen_check; `control|{opening,early_mid,midgame,late}` 10 tuning + 5 frozen_check each; totals 80/40 tuning/frozen_check, 60/60 target/control.
- Frozen late-target band minima: totals b400_plus ≥ 8 / b300_399 ≥ 12 / b200_299 ≥ 12; per-split tuning {b400 4, b300 8, b200 8} / frozen_check {b400 4, b300 5, b200 5} — **now backed by a constructive witness** (bands realized 11/23/26; per-split tuning 7/15/18, frozen_check 4/8/8).
- **The frozen_check b400_plus minimum is satisfied with ZERO witness slack (4 of 4)** — 11 of the 12 candidate b400 rows (12 games, 7 black/5 red on screen) were selected. This makes Task 16 sizing decisive for fresh-reservoir reliability.
- Old 240-row allocation on the same screen: **GATE_FAIL, exit 4** — all three non-late target cells capacity 0 < demand 45 (confirms 0/0/0/136).
- Committed evidence (`logs/eval/fpu_v16_policy_mass_v2/analysis/`, SHA-1): `production_profile.json` 378d3cdb61c9b113af6cb8d1cf5c8fb41e3f39e2 · `old_allocation_profile.json` d0287df2eb128e5c8fe595897dd203ec0d8a9012 · `production_feasibility_r2.json` 2bef7133939faa9c778dd576155ced3a68a6bf61 · `old_allocation_feasibility_r2.json` 6295e1af467dcfef8019b271947f85334ba5ad66.

**Pending (gated):** Task 15 400-game 24×24 `tooling_smoke` (plumbing proof only — smoke omits production band floors; a smoke sampling GATE_FAIL is a controlled result, never permission to top up or weaken production minima); Task 16 finite-reservoir sizing (299-trial preregistered 95% lower bound ≥ 0.99, next-tier-up rule) + production protocol emission. Smoke/sizing results will be appended here.

---

*Append a new row to the [experiment ledger](#experiment-ledger) and update [do-not-repeat](#do-not-repeat-prevents-going-in-circles) whenever a branch is run and judged. Keep the [key conclusion](#targeted-value-calibration--experiment-ledger) current.*
