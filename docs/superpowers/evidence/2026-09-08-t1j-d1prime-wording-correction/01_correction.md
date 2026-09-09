# CORRECTION to the D1′ outcome wording — 2026-09-08

**Create-only.** The D1 acquisition evidence at
`docs/superpowers/evidence/2026-09-08-t1j-d1-acquisition/` is sealed and is **not** edited by this
correction. Its numbers were independently verified in review — all 221 positions, 1,105 queries,
seed accounting, trace, statistic, interval and manifest — and **the frozen NO_GO verdict stands**.
What was wrong was my prose about what that verdict means.

## What I wrote, and why each is too strong

**1. "Support floors were met, so this is a null, not a shortage of data"**
— `20_outcome.md` §2; the same phrase appears in commit `d75cb71`'s message.

"A null" claims a null-hypothesis result. The interval this analysis produces is an **empirical
stability interval** from stratified whole-game cluster resampling: it measures how much T moves
when the observed games are reweighted. The frozen plan says so explicitly and I restated it
correctly two lines earlier in the same file, then contradicted it. It is **not** a
null-hypothesis confidence interval, so "a null" is not a conclusion this design can reach.

**2. "positions our model marks as mover-fragmentation weaknesses do not show more low-policy-rank
disagreement than their matched controls"** — `20_outcome.md` §2.

The point estimate is **positive**: T = 0.0695. A positive excess was observed. It was small,
below the threshold, and unstable — which is not the same as absent, and "do not show more" says
absent.

## The accurate conclusion

> Support was adequate. The fragmentation cohort showed a small positive matched LPRD excess
> (`0.0695`), but it was below the preregistered `0.15` threshold and unstable under whole-game
> reweighting (`−0.1944` to `0.1827`). Therefore D1′ returned NO_GO.

## What is unchanged

The verdict, every number, the seed reconciliation, and the decision that follows: **no confirmation
run is justified and no training intervention is proposed.** The correction narrows the *claim*, not
the *outcome* — and it narrows it in the direction that makes the hypothesis harder to dismiss, not
easier.

## Where the wrong wording still stands, by design

- `2026-09-08-t1j-d1-acquisition/20_outcome.md`, §2 — sealed evidence, left intact.
- Commit `d75cb71`'s message — history is not rewritten.
- My session summary to the reviewer, which repeated both phrases.

This file is the correction of record for all three. The memory note
`t1j-d1-acquisition-result` carried the same two phrases and **has been corrected in place**, since
it is a working index rather than sealed evidence.
