# Targeted Value Calibration — Experiment Ledger

**Created:** 2026-06-26 · **Status:** active · **Scope:** the value-head calibration line of work (fix the black pre-drop overvalue without breaking the guardrail families).

A durable, append-only record of every value-calibration experiment: what changed, how it scored on the four acceptance gates, what we learned, and what **not** to retry. **Read this before proposing any new calibration knob** — if the change is on the [do-not-repeat](#do-not-repeat-prevents-going-in-circles) list (or another sweep of a knob we've already swept), the answer is probably "no, we already saw the tradeoff."

> **Key conclusion (updated 2026-07-03, post v8 value-head-only):** Targeted **correction works**, but the successful mechanism changed. Full-network calibration variants kept moving the value surface nonlocally: v6 continuation-only, v6b D root+policy, v6c D root value-only, and v7 sparse severe-D correction all failed at least one guardrail. The v7 drift map showed the core issue: even targeted value pressure caused unrelated B/C/D rows to move upward/downward unpredictably. **v8 value-head-only proved that trunk/representation drift was a major failure mode**: with all non-`value_head.*` tensors byte-identical to BASE, B/C/D passed while A improved but stayed short of the A gate. **v8b (value-head-only, A draw pressure 2→3) RUN + REJECTED 2026-07-03** — did not help A, made A/C worse; the v8/v8b raw-A diagnostic shows raw value-head output barely moved (BASE −0.2469 → v8 −0.2533 / v8b −0.2433), so A's constraint is representational, not sampling. **Next = v9 partial unfreeze: value head + final residual block** (`--train-value-head-and-final-block`, v8 schedule, same v7 manifest; spec `docs/superpowers/specs/2026-07-03-targeted-value-calibration-v9-value-head-and-final-block-design.md`). No promotion match unless A/B/C/D all pass.

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

*(The current best `calib020_0001` is the baseline row — see [Current best](#current-best).)*

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

Also retired as *primary* strategies: global-weight sweeps, retention-weight sweeps, schedule-ratio sweeps, frozen-BN-as-the-fix reruns, and raw-teacher weight/schedule tweaks. The next step is searched-continuation/PV retention (v6), not another root-position-level retention design or raw-objective knob sweep.

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

Same v7 manifest + value-head-only mechanics, A draw pressure raised 2→3 (`black_predrop_correction=3,goal_line_continuation_retention=1,old_post_opening_continuation_retention=2,red_predrop_severe_root_correction=1,red_predrop_continuation_retention=2`). **Result: did not help A and made A/C worse** (per operator; exact per-gate mean/over/severe figures are in the v8b gate CSVs — those are authoritative). Higher A draw pressure on a value-head-only fit does not move A: the constraint is representational, not sampling.

### v8/v8b raw-A diagnostic (why value-head-only can't fix A)

On the 50 `black_predrop_correction` rows, **raw** value-head output barely moved:

| | raw mean | Δ vs BASE | severe raw overvalue |
|---|---|---|---|
| BASE | −0.2469 | — | 20.0% |
| v8 | −0.2533 | −0.0064 | 14.0% |
| v8b | −0.2433 | +0.0035 | 16.0% |

A did **not** fail because MCTS amplified an already-corrected raw value — the raw values themselves scarcely changed. A failed because value-head-only cannot substantially move the worst A raw values with the trunk frozen: `value_head` is a shallow MLP readout (`fc1→fc2`, no conv/BN) on frozen features.

**Conclusion:** v8 proved full-network drift was the main cause of B/C/D breakage (value-head-only preserved B/C/D). But value-head-only is too constrained to fix A. Next hypothesis is **partial unfreeze**: value head + the smallest late representation slice, starting with the final encoder/residual block.

## Current next hypothesis — v9 partial unfreeze (value head + final residual block)

Design spec: `docs/superpowers/specs/2026-07-03-targeted-value-calibration-v9-value-head-and-final-block-design.md`.

New flag `--train-value-head-and-final-block` (mutually exclusive with `--train-value-head-only`): v8's guard (skip the whole-trunk `opt_main.update`) **plus** one extra `opt_main.update(network.encoder.blocks[last], main_grads["encoder"]["blocks"][last])` applying the final block's already-clipped grads to the live block submodule. Unfreezes exactly `value_head.*` + `encoder.blocks[last]` trainable tensors (8); everything else — stem, blocks `0..last-1`, policy head, and **all** BN running stats incl. the final block's — stays byte-identical. Same v7 manifest, **v8 schedule** (NOT v8b's raised A pressure — v8b showed that doesn't help), weight 0.01, both `--freeze-batchnorm-stats` and the new flag. A new strict verifier (`verify_value_head_and_final_block_checkpoint.py`) is the acceptance proof: exit 0 pass / 1 leak / 2 no-op / 3 partial-unfreeze-never-engaged. Gates A/B/C/D vs `calib020_0001`, thresholds unchanged. Interpretation: pass A + hold B/C/D ⇒ drift lives in the earlier trunk; hold B/C/D but still miss A ⇒ one block too few (v9b = last-2); move A but break B/C/D ⇒ partial unfreeze is a dead end.

## Current next hypothesis — v6 searched-continuation/PV retention

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
- **Plans:** `docs/superpowers/plans/2026-06-24-targeted-value-calibration-v2.md`, `docs/superpowers/plans/2026-06-25-targeted-value-calibration-v3-tag-stratified-sampling.md`, `docs/superpowers/plans/2026-06-29-targeted-value-calibration-v4-teacher-retention.md`.

---

*Append a new row to the [experiment ledger](#experiment-ledger) and update [do-not-repeat](#do-not-repeat-prevents-going-in-circles) whenever a branch is run and judged. Keep the [key conclusion](#targeted-value-calibration--experiment-ledger) current.*
