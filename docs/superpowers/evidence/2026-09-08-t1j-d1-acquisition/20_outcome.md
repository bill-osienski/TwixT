# D1 ACQUISITION — RAN ONCE, **COMPLETED**; D1′ outcome **NO_GO**

## 1. The run

| | |
|---|---|
| entry | the public CLI, fresh subprocess, every path literal |
| window | `2026-09-09T00:33:03Z → 00:40:13Z`, **430.4 s** of 5,400 |
| exit | **0** |
| positions | **221 / 221** |
| queries | **1,105 / 1,105** (5 per position, exactly as §12.4 funds them) |
| trace verdict | **OK**, 1,328 lines, one `run_start`, one `run_end` |
| stdout / stderr | both empty |
| surviving processes | **none** |
| gate | restored to False **immediately** after exit, before any analysis |

Bounds in force: per-query 120 s, whole-run 5,400 s, ply cap 280 named explicitly. The cohort was the
fresh 221-row manifest built from the canonical L0 record, verified row-for-row against the pinned
frozen selection and accepted by the public entry's own cohort check.

## 2. The analysis — run once, on the completed record

`analyse_development(report)`, the checked entry that resolves cohort, design, B and PRNG seed itself.

| | |
|---|---|
| **outcome** | **NO_GO** |
| T (Mantel–Haenszel weighted LPRD) | **0.0695** |
| threshold | 0.15 |
| empirical stability interval | **[−0.1944, 0.1827]** |
| replicates / PRNG | 10,000, PCG64, seed 20260907 |
| common-support cells | 21 (8 positive, 9 negative, 4 zero) |
| secondary cohort (decides nothing) | `created_threat`, T = −0.358 |

**Support floors were met, so this is a null, not a shortage of data:** 21 cells ≥ 8, 62 positions
≥ 40, 54 controls ≥ 30, 17 games ≥ 12. NO_GO on **both** conditions — the effect is below the
threshold **and** the stability interval spans zero.

🔑 **What this establishes.** In the frozen §13 development cohort, positions our model marks as
mover-fragmentation weaknesses do **not** show more low-policy-rank disagreement than their matched
controls. Per the frozen plan, GO would have meant only "enough effect and stability to justify ONE
confirmation run" — so NO_GO means **no confirmation run is justified and no training intervention
is proposed**. It is not proof of no population effect. Confirmation data was neither inspected nor
touched.

## 3. Seed reconciliation, from the evidence

All 221 seeds **EXPOSED**, counted from the record: every one of the 221 position entries carries an
incumbent readout, and the readout is what draws the seed. The trace agrees independently
(`seeds_drawn: 221` at `run_end`). The whole block **RETIRED**: a preregistered one-shot schedule
that completed, under L0's rule. `CONSUMED_SEEDS` and the test-only band are unchanged. A spent seed
is now refused by `validate_task_executable`, asserted at that effect. **A future D1 needs a fresh
interval.**

Contrast with 2026-08-28, in one word: that block was retired with **no** exposure claim, because a
VOID wrote no record and claiming 227 draws would have asserted 226 that may never have happened.
Here the record survives, so the count is exact rather than bounded.

## 4. Verification

| | |
|---|---|
| injected-defect controls | **382 / 382 rejected**, 0 not caught, 0 stale |
| clean baseline | **PASS** over 306 target tests |
| every source restored | True |
| post-run suite | **4,154 passed, 4 skipped, 0 failed** (15m21s) |
| six gates | all False |

Five controls were re-aimed once the run changed the facts: the injections "the block is also marked
EXPOSED / RETIRED" became the true state, so the defect is now the opposite one — a spent block left
un-exposed or un-retired — plus a control that deletes the availability check entirely, since one
anchor cannot un-spend two lists.

## 5. Faults of mine this round, corrected

1. 🔴 **My gate-restoration commit was not the one-line change its message implies.** I used
   `git add -A`, which swept the run's record, trace, stdout, stderr, exit code and timing into
   `0277e4a` alongside the gate line. The gate line itself is correct and the artifacts are
   unmodified, but the commit message describes a narrower change than the commit makes. Recorded
   here rather than rewritten, and the message is corrected in the record.
2. 🔴 **A test of mine passed for the wrong reason.** My first "a spent seed cannot be scheduled"
   test built a task missing `reference_sha1` and `anchor_colour`, so it was refused as MALFORMED and
   would have kept passing with the block un-retired. The task is now asserted well formed first, so
   only spentness can refuse it. Exposure is checked before retirement, which is why the message
   names EXPOSED.
3. 🔴 **A fixture lifted one of two spent lists.** `live_block` lifted RETIRED but not EXPOSED, so
   two incumbent mechanism tests began failing on clean source the moment the run exposed the block.
   The clean-baseline check found them. It now lifts both.

## 6. Still excluded

Confirmation inspection or execution, training, another D1 attempt, D2, and the push (58 commits).
