# Reverted Closeout Experiments — Decision Record

**Date:** 2026-05-19
**Context:** Two closeout-side tuning experiments were tried between the 220-229 self-play block and the marathon-termination spec (`2026-05-19-marathon-termination-tuning-design.md`). Both regressed the closeout / long-tail metrics they were intended to improve and have been reverted.

This file is git-tracked so the rollback is visible outside any single Claude session memory and cannot get re-suggested under a different framing without explicit acknowledgement of the prior result.

---

## Stable baseline (do not deviate without new evidence)

For any training block from `model_iter_0229` onward:

```
--closeout-selection-tiebreak-min-value 0.95
--conversion-policy-loss-weight 0.05
```

All other flags inherit the 220-229 launch command (resign / adjudicate / opening / mirror / conversion / etc. unchanged).

---

## Reverted experiment 1 — Fix 2 value gate 0.95 → 0.90

**Flag:** `--closeout-selection-tiebreak-min-value`
**Tried value:** 0.90 (relaxed from baseline 0.95)
**Tried on:** training block after 220-229
**Outcome:** more Fix 2 overrides, but worsened closeout tail and td=2 quality.
**Hypothesis behind the experiment:** the value gate was rejecting positions where the model has the right move in top-5 but isn't 95% confident at the failure ply ("Cluster 1" from the 2026-05-17 manual Family C diagnosis). Relaxing to 0.90 would unlock more overrides.
**Why it failed:** at q ∈ [0.90, 0.95) the model isn't reliable enough for the override to be net-positive. The added overrides included false positives that disrupted otherwise-healthy closeout play.
**Decision:** revert to 0.95. Do NOT re-test 0.90 without a new diagnostic showing why the gate would now behave differently (e.g., after the conversion-policy-loss signal is recalibrated independently).

## Reverted experiment 2 — conversion-policy-loss weight 0.05 → 0.075

**Flag:** `--conversion-policy-loss-weight`
**Tried value:** 0.075 (raised from baseline 0.05)
**Tried on:** training block after 220-229
**Outcome:** did not improve buried td=2 reducers; worsened state-cap / long-tail pressure.
**Hypothesis behind the experiment:** strengthen the conversion signal so the policy learns to consider distance-reducing moves more often when they're outside top-5 ("Cluster 2" from the 2026-05-17 manual Family C diagnosis).
**Why it failed:** the stronger conversion signal apparently pulls the policy away from healthy chain-extension during normal play, increasing state-cap pressure. The td=2 ranking did not improve.
**Decision:** revert to 0.05. Do NOT re-test 0.075 without a new diagnostic showing why the policy signal would now behave differently.

---

## When to revisit either knob

Per the marathon-termination spec, the next round of decisions should be driven by the bucket-count and adjudication-gate-block data, not by re-trying these knobs:

- If the long-tail bucket counts (Spec 2026-05-19-long-tail-bucket-classifier) show `td2_alt_in_top5` rising significantly above its current ~10% share, the Fix 2 gate question may be worth revisiting — but with a NEW diagnostic premise, not the same 0.90 trial.
- If `td2_reducer_buried` rises significantly above its current ~8% share, the conversion-policy-loss question may be worth revisiting — but the 0.075 trial showed the relationship isn't monotonic, so any new attempt needs a smaller step (e.g., 0.06) or a different mechanism (e.g., loss-weighting at td=2 positions only).

---

## Related artifacts

- Long-tail bucket counts: [`docs/superpowers/specs/2026-05-19-long-tail-bucket-classifier-design.md`](../specs/2026-05-19-long-tail-bucket-classifier-design.md)
- Marathon-termination spec: [`docs/superpowers/specs/2026-05-19-marathon-termination-tuning-design.md`](../specs/2026-05-19-marathon-termination-tuning-design.md)
- Marathon-termination plan: [`docs/superpowers/plans/2026-05-19-marathon-termination-tuning.md`](../plans/2026-05-19-marathon-termination-tuning.md)
