# Targeted Value Calibration — Experiment Ledger

**Created:** 2026-06-26 · **Status:** active · **Scope:** the value-head calibration line of work (fix the black pre-drop overvalue without breaking the guardrail families).

A durable, append-only record of every value-calibration experiment: what changed, how it scored on the four acceptance gates, what we learned, and what **not** to retry. **Read this before proposing any new calibration knob** — if the change is on the [do-not-repeat](#do-not-repeat-prevents-going-in-circles) list (or another sweep of a knob we've already swept), the answer is probably "no, we already saw the tradeoff."

> **Key conclusion (updated 2026-07-02):** Targeted **correction works** — every successful branch can move the black pre-drop family (A). But every tested retention strategy has failed to preserve the guardrail families (B/C/D): scalar root-value targets (v2/v3/v3b/v3F), raw teacher value+priors (v4), and now **position-level root-visit policy anchors (v5, REJECTED 2026-07-02)**. The BN confound is resolved (v3F), and the raw-NN diagnostic showed v4's raw head held while the MCTS root drifted — v5 tested the natural fix (anchor the search-improved root policy) and it was **insufficient**. **Next step is NOT another training run or weight sweep: first diagnose whether v5 actually held its stored root-policy anchors on the retention rows.** Anchors held ⇒ the drift is tree/path-level (deeper than any per-position anchor can reach); anchors not held ⇒ inspect the loss/weighting before concluding anything.

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
10. **Any root-policy weight/schedule sweep, or a new retention design, before the v5 anchor-hold diagnosis.** v5 (root-visit policy anchors, CE weight 0.25) failed B/C/D like every predecessor. Before touching a knob or proposing v6: measure whether the trained v5 checkpoint actually **holds the stored root-policy/value anchors on its own retention rows** (compare candidate raw policy vs stored `root_visits_json`, and candidate raw value vs `teacher_value`, on the v5 manifest — the raw-NN diagnostic CLI + manifest columns make this cheap). Anchors held ⇒ tree/path-level drift is the mechanism and per-position anchoring is a dead end; anchors not held ⇒ the loss/weighting under-trained them and the objective was never really tested.

Also retired as *primary* strategies: global-weight sweeps, retention-weight sweeps, schedule-ratio sweeps, frozen-BN-as-the-fix reruns, and raw-teacher weight/schedule tweaks. The next step is an MCTS-root/root-behavior retention design, not a new raw-objective knob sweep.

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


## Code / artifact pointers

- **v2** manifest builder + mixed-pool weighted loss: `scripts/GPU/alphazero/build_targeted_calibration_manifest.py`; operator guide `docs/post-game-analysis.md` §6.
- **v3** tag-stratified sampling: `--post-opening-calibration-tag-schedule` (commits `0c122cb` / `0e0fd24` / `282998d` / `b27d60b` on `main`); telemetry `state.calib_n_drawn_by_tag` + sidecar `post_opening_calibration.draws_by_tag`; operator guide `docs/post-game-analysis.md` §6 (tag-stratified block).
- **v4** teacher-retention builder/smoke/training path: `scripts/GPU/alphazero/build_teacher_calibration_manifest.py`, `scripts/GPU/alphazero/smoke_teacher_calibration_v4.py`, `--post-opening-calibration-teacher-value-weight`, `--post-opening-calibration-teacher-policy-kl-weight`, and `--freeze-batchnorm-stats`.
- **v3-frozenBN-control checkpoint/gates:** `checkpoints/alphazero-v3-frozenBN-control-from-calib020-0001/model_iter_0001.safetensors`, `logs/eval/v3_frozenBN_control_from_calib020_0001_gates_400s/`.
- **v4/v3F severe-overlap review:** `logs/eval/v3f_v4_severe_overlap_review.csv`.
- **Raw-NN focus-row diagnostic:** `scripts/GPU/alphazero/eval_raw_nn_position_rows.py` (+ `tests/test_eval_raw_nn_position_rows.py`, merged to main @ `7064621`); output `logs/eval/v3f_v4_raw_nn_focus_rows.csv`; plan `docs/superpowers/plans/2026-07-01-eval-raw-nn-position-rows-diagnostic.md`.
- **v5 root-retention:** builder `scripts/GPU/alphazero/build_mcts_root_retention_manifest.py` (+ `--gate-checkpoint-label` cross-check), smoke `scripts/GPU/alphazero/smoke_mcts_root_retention_v5.py`, `loss_mode=mcts_root_retention` in `calibration_pool.py`; manifest `logs/eval/targeted_calibration_v5_root_from_calib020_0001.csv`; checkpoint `checkpoints/alphazero-v5-mcts-root-from-calib020-0001/model_iter_0001.safetensors`; plan `docs/superpowers/plans/2026-07-01-targeted-value-calibration-v5-mcts-root-retention.md`; operator guide `docs/post-game-analysis.md` §8.
- **Plans:** `docs/superpowers/plans/2026-06-24-targeted-value-calibration-v2.md`, `docs/superpowers/plans/2026-06-25-targeted-value-calibration-v3-tag-stratified-sampling.md`, `docs/superpowers/plans/2026-06-29-targeted-value-calibration-v4-teacher-retention.md`.

---

*Append a new row to the [experiment ledger](#experiment-ledger) and update [do-not-repeat](#do-not-repeat-prevents-going-in-circles) whenever a branch is run and judged. Keep the [key conclusion](#targeted-value-calibration--experiment-ledger) current.*
