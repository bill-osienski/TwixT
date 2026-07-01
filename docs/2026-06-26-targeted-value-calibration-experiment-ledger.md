# Targeted Value Calibration — Experiment Ledger

**Created:** 2026-06-26 · **Status:** active · **Scope:** the value-head calibration line of work (fix the black pre-drop overvalue without breaking the guardrail families).

A durable, append-only record of every value-calibration experiment: what changed, how it scored on the four acceptance gates, what we learned, and what **not** to retry. **Read this before proposing any new calibration knob** — if the change is on the [do-not-repeat](#do-not-repeat-prevents-going-in-circles) list (or another sweep of a knob we've already swept), the answer is probably "no, we already saw the tradeoff."

> **Key conclusion (updated 2026-06-30):** Targeted **correction works** — every successful branch can move the black pre-drop family (A), and v4 produced the strongest A correction so far. But every tested retention strategy has still failed to preserve the guardrail families (B/C/D). v4 teacher-retention fixed A but broke B/C/D, so it is rejected. Because train-mode BatchNorm was found to be a calibration confound, the next controlled step is **`v3-frozenBN-control`**, not another weight/schedule sweep.

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

v4 was run with `--freeze-batchnorm-stats`; v2/v3 behavior should not be reinterpreted without a controlled frozen-BN rerun.

## Experiment ledger

| Experiment | Main change (knobs) | A — black pre-drop | B — goal-line | C — old post-opening | D — red pre-drop | Match | Decision / lesson |
|---|---|---|---|---|---|---|---|
| **calib010** — black-predrop v1 | Train **only** black-predrop correction, weight 0.01 | improved: over ~16–23%, severe ~6.7–10% | borderline / regressed | improved | regressed | Lost badly, **~−95 Elo** | **Reject.** Target fixed, broad play damaged. |
| **v2** — mixed pool | Correction + retention rows; retention_weight **0.5**, global **0.01**, **uniform** sampling | fixed A strongly | **fail** | **fail** | **hard fail** | no match | **Reject.** Correction worked, retention too weak. |
| **v2b** — mixed pool | v2 but retention_weight **2.0**, global 0.01 | fail/borderline @400: mean +0.038, severe 16.7% | **fail:** severe 5.6% | borderline fail: mean +0.109 | **fail:** severe 3.3% | no match | **Reject.** Stronger retention helped some; tradeoff remained. |
| **v3** — tag-stratified | Schedule **2:1:2:1**, retention_weight **1.0**, global 0.01 | **pass:** mean −0.047, severe 10.0% | **pass:** severe 0.0%, over 11.1% | **fail:** mean +0.180, over 40.0%, severe 23.3% | **fail:** severe 10.0% | no match | **Reject.** Stratification fixed mechanics + A/B, but C/D drifted. |
| **v3b** — tag-stratified, lower weight | Same schedule, global **0.005** | weak / fail-ish: mean −0.030, severe 20.0% | **fail:** severe 11.1% | **fail:** mean +0.113, severe 20.0% | **hard fail:** severe 23.3% | no match | **Reject.** Lower scalar weight didn't solve drift. **Stop scalar sweeps.** |
| **v4** — teacher-retention | raw-NN teacher value-MSE + teacher policy CE/KL on retention rows; global 0.01 / value 1.0 / policy 0.25; schedule 2:1:2:1; 1 iter (= v3); freeze_batchnorm_stats=true | **pass:** mean −0.305, over 13.3%, severe 6.7% | **fail:** over 16.7%, severe 11.1% | **fail:** mean +0.029, over 36.7%, severe 23.3% | **fail:** mean −0.038, over 36.7%, severe 16.7% | no match | **Reject.** Teacher-retention preserved clean Gate-0 self-distillation and fixed A, but B/C/D still drifted. No promotion. |

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
6. **Another v4 teacher-retention weight/schedule tweak before the frozen-BN control.** v4 fixed A but failed B/C/D. Before changing teacher value/policy weights or schedule ratios, run the controlled `v3-frozenBN-control` to isolate the BatchNorm confound.

Also retired as *primary* strategies: global-weight sweeps, retention-weight sweeps, schedule-ratio sweeps, and ungrounded v4 weight/schedule tweaks. The next step is a controlled historical check, not a new knob sweep.

## Severe-overlap findings (why the next step changes shape)

- **C — stable repeat offenders:** 5 of the severe cases repeat across v3/v3b (listed above). C should be treated as a fixed fragile family needing **direct retention of current-best behavior**, not as eval noise.
- **D — diffuse:** only 1 shared severe case across v3/v3b. D reads as **broad value-head drift**; unlikely to be solved by adding a few hard D rows.

## Current next hypothesis after v4

v4 tested teacher-retention distillation from `calib020_0001` using raw-NN teacher value plus dense teacher policy CE/KL on B/C/D retention rows. It fixed A strongly but still failed B/C/D, so **teacher-retention at the tested settings is rejected**.

The BatchNorm finding means the next disciplined step is not a new v4 weight/schedule change. The next run should be a controlled historical check:

**`v3-frozenBN-control`**
- Same base checkpoint: `calib020_0001`.
- Same v3 scalar-retention manifest and `2:1:2:1` tag schedule.
- Same global calibration weight `0.01`, correction target `−0.35`, one iteration, 100 games, 400 sims, batch 64, lr `0.0003`.
- Add only `--freeze-batchnorm-stats`.
- Score with the same A/B/C/D 400-sim gates.

**Decision value of this control:**
- If `v3-frozenBN-control` still fails like v3, then scalar retention remains rejected under the corrected BN regime.
- If it materially improves B/C/D, BatchNorm batch-dependence was a major confound and the scalar-retention conclusions need a narrower frozen-BN interpretation.
- If it beats v4 on B/C/D while holding A, revisit the teacher-retention design only with a clear reason, not as a sweep.
- If it passes all A/B/C/D gates, then it earns the promotion-match decision point.

Until that control is run, do **not** reinterpret old v2/v3 results as invalid and do **not** start a v4 weight/schedule sweep.

## Code / artifact pointers

- **v2** manifest builder + mixed-pool weighted loss: `scripts/GPU/alphazero/build_targeted_calibration_manifest.py`; operator guide `docs/post-game-analysis.md` §6.
- **v3** tag-stratified sampling: `--post-opening-calibration-tag-schedule` (commits `0c122cb` / `0e0fd24` / `282998d` / `b27d60b` on `main`); telemetry `state.calib_n_drawn_by_tag` + sidecar `post_opening_calibration.draws_by_tag`; operator guide `docs/post-game-analysis.md` §6 (tag-stratified block).
- **v4** teacher-retention builder/smoke/training path: `scripts/GPU/alphazero/build_teacher_calibration_manifest.py`, `scripts/GPU/alphazero/smoke_teacher_calibration_v4.py`, `--post-opening-calibration-teacher-value-weight`, `--post-opening-calibration-teacher-policy-kl-weight`, and `--freeze-batchnorm-stats`.
- **Plans:** `docs/superpowers/plans/2026-06-24-targeted-value-calibration-v2.md`, `docs/superpowers/plans/2026-06-25-targeted-value-calibration-v3-tag-stratified-sampling.md`, `docs/superpowers/plans/2026-06-29-targeted-value-calibration-v4-teacher-retention.md`.

---

*Append a new row to the [experiment ledger](#experiment-ledger) and update [do-not-repeat](#do-not-repeat-prevents-going-in-circles) whenever a branch is run and judged. Keep the [key conclusion](#targeted-value-calibration--experiment-ledger) current.*
