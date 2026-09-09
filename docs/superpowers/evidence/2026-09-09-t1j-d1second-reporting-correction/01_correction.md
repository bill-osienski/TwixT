# CORRECTION to the D1″ reporting — 2026-09-09

**Create-only.** The sealed execution evidence at
`docs/superpowers/evidence/2026-09-09-t1j-d1second-execution/` is **not edited**, and the analysis is
**not rerun**. Nothing below changes a computed value.

**The primary result is unchanged and verified independently:** T = **−0.0231729**, stability
interval **[−0.0781929, 0.0124378]** over 10,000 replicates, floors passed, verdict **NO_GO**. D1
fragmentation stays closed; no confirmation and no training follow from it.

Two things I reported wrongly.

---

## P1 — A zero override flag does not mean the visit leader was selected

**What I wrote** (`06_outcome.md`, and the same claim in commit `10cab32` and in my summary):

> "The readout never once overrode the visit leader in either role. The second mechanism the plan
> named — the readout declining the move the search preferred — **did not occur at all** in this
> cohort."

**That is false**, and the error is mine: I read a flag as if it measured selection.

**What the flag actually means, verified in the writer's source.** `eval_readout.select` returns
`(move, overrode_leader)`. The frozen configuration is `selection_mode = "opening_temperature"`, and
in that branch the function returns the flag as **`False` unconditionally** — the temperature sample
is taken and the flag is hard-coded `False` beside it. The flag is only ever `True` in
`MODE_HOEFFDING_LCB`, a mode this configuration never uses. So in these 155 rows the flag is a
**constant of the mode**, carrying no information about which move was played.

**What the rank histograms say — reported, not inferred.** These are the stored values, recomputed
from the sealed result and nothing else:

| | positions (n=101) | controls (n=54) |
|---|---|---|
| selected move was the **visit leader** (rank 1) | 22 | 38 |
| selected move was **not** the visit leader | **79 (78.2%)** | **16 (29.6%)** |
| selected move was the **top policy move** | 16 | 23 |
| selected move was **not** the top policy move | 85 (84.2%) | 31 (57.4%) |
| worst selected visit ranks observed | 357, 365, 376, 395 | 11, 13, 139, 196 |

So the honest statement is the opposite of what I wrote: **a non-leader was selected in most position
rows**, sometimes a move with a visit rank in the hundreds — effectively an unvisited move.

⚠ **Stated without inference, deliberately.** The frozen readout samples by temperature, with
`temp_high = 1.0` below ply 20 and `temp_low = 0.1` at or after it, so non-leader selection is an
expected consequence of the configuration rather than a defect on its face. **The
position/control difference above is NOT interpreted here**: the two roles may differ in ply
composition, and this correction does not report a ply breakdown or test any hypothesis about it.
Reading a cause into this table would repeat the mistake it corrects.

## P2 — "an order of magnitude" overstated the gap

**What I wrote:** "the effect is an order of magnitude below the preregistered threshold."

|T| = 0.0232 against a threshold of 0.15 is a factor of about **6.5**, not ten. The accurate
statement: **the point estimate is in the wrong direction and well below the threshold.**

---

## What is unchanged

The verdict, every computed number, the floors, the seed accounting, and the decision that follows:
**the D1 fragmentation line is closed, no confirmation acquisition is proposed, the sealed half stays
sealed, and no training follows.** Both corrections narrow claims I made *about* the result, not the
result.

## Where the wrong wording still stands, by design

- `2026-09-09-t1j-d1second-execution/06_outcome.md` — sealed evidence, left intact.
- Commit `10cab32`'s message — history is not rewritten.
- My summary to the reviewer, which repeated both claims.

This file is the correction of record for all three. The memory note `t1j-d1second-result` carried the
override claim and **has been corrected in place**, being a working index rather than sealed evidence.
