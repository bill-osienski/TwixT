# The L0 canonical 64-game match — RAN ONCE. T1j's observed score: 0.594.

**Date:** 2026-08-27 · **Status:** EXECUTED, once, exit **0**. Gate restored immediately.
· Local, unpushed. · **`[202613000, 202613064)` is SPENT. The match will not run again.**

Basis: plan published at `c7f87c9`, wiring at `f5e3582`. Enabled in `9805c19`, restored in `9483948`.
Evidence: `evidence/2026-08-27-t1j-l0-canonical-match/`.

---

## The result

**T1j's observed score was 38.0 of 64 = 0.5938** against `calib020_0001` at `mdPly` 6.

| | |
|---|---|
| **Hoeffding 95% (PRIMARY)** | **[0.4240, 0.7635]** |
| Wilson 95% (nominal only) | [0.4715, 0.7054] |
| cap terminations | **0** |
| plies min / median / max | 25 / 41 / 88 (cap 280, never approached) |

> **Both intervals include 0.5.** Hoeffding spans [0.4240, 0.7635] and the nominal Wilson interval
> spans [0.4715, 0.7054]; parity sits inside both. So the match gives T1j a **higher point estimate**
> at this setting and **does not establish that T1j is stronger**. Saying otherwise would be reading
> a point estimate as a conclusion, which is precisely what the interval is there to prevent.

All 64 games played. No early stop, no skips, no retries. Exit 0, **stdout and stderr both zero
bytes**, 30m 31s, 2,411 durable records including **2,216 per-ply bindings with zero divergences**.
Exactly two agents per task — one per colour, reused across the game — and every game ended in a win.

### Descriptive only — no interval, no cross-cell comparison

| arm | games | score | rate |
|---|---:|---:|---:|
| `t1j_red` | 32 | 25.0 | 0.781 |
| `t1j_black` | 32 | 13.0 | 0.406 |

Per opening (8 games each): `o2_offcenter` and `o3_low` 0.750 · `o1_center`, `o7_diagonal`,
`o8_contact` 0.625 · `o4_high`, `o6_wide_right` 0.500 · `o5_wide_left` 0.375.

The colour asymmetry is the largest visible effect. It is **not** a preregistered test, carries no
interval, and 32 games per arm is not a basis for a finding. It is recorded because it is there.

## What the larger match actually bought

The screen's strong endpoint read **7.0/8 = 0.875**, early-stopped at the earliest game the rules
allowed. The 64-game rate is **0.594**.

The preregistration said this could happen, before any data existed: an early-stopped 8-game figure
is not an unbiased estimate of a larger match, and the plan recorded that in `08_design_table.txt`
alongside its own wide n=8 interval. **That is what preregistration is for** — the caveat was on
record before the number that vindicated it.

The 0.875 was never wrong; it was a band decision, and the band decision stands. What it was not is a
rate.

## Discipline held

`EARLY_STOP is None` in the run header, no `task_skipped` record exists, and the reporter refuses any
vector that is not all 64 — so no early stop could have produced a report. Zero caps means the
preregistered `CAP_SATURATED_NO_RATE` branch was never reached. A `qualification_receipt` does not
appear: match mode emitted a `match_report`.

**No Elo. No absolute placement.** T1j is uncalibrated. This is an ordering against `calib020_0001` at
400 simulations in this stack, bounded to these eight openings, this colour balance, `mdPly` 6, ply
cap 280, one run. The interval is a 95% bound **under an independence model** — distinct derived
streams rule out accidental reuse, not dependence.

## Seeds reconciled

All 64 played, so all 64 are **exposed** (drawn) *and* **retired** (the one-shot completed) —
contiguous `202613000..202613063`, neighbours free. Registries now hold 129 exposed and 96 retired.

The verdict **reclassifies identically** from the durable records alone — score, both intervals, and
every cell — while `validate_schedule_executable` now refuses the schedule. Spent stops execution,
not reading. That property was designed in during the seed reconciliation and this is the first time
it has been load-bearing on a fresh block.

## Post-run test reconciliation

The 25 failures below were the honest immediate post-run state, and the transcript is preserved
byte-unchanged in `09_full_suite_after_gate_restored.txt`. They were then reconciled on the same
principle the E4 screen established — *spent stops execution, not reading*:

- the four obsolete "reserved/unspent" assertions now check that all 64 seeds are **exposed and
  retired**, that the plan is still **structurally valid**, and that the durable results
  **reclassify identically** (score, both intervals, every cell) — including that both intervals
  contain 0.5;
- a **real-state** command test asserts the spent schedule exits **2** at `schedule` before any
  results file, class directory, agent, RNG, model or Java — and asserts **`spawned == []`**, because
  `repository` (the only check that shells out to git) runs *third* and never executes. An earlier
  form asserted `programs <= {"git"}`, which is **vacuously true of an empty list**; the weaker check
  would have passed however many git calls were made. A fresh-subprocess test confirms the real CLI
  exits 2 and never reaches the authorization message;
- tests of later preconditions, the authorization gate and the enabled-path wiring take an
  `unspent_block` fixture that lifts **eligibility only** — two registry tuples, never either
  authorization gate, asserted on entry and exit; the subprocess variant lifts the same two tuples in
  the child and asserts **both** `L0_EXECUTION_AUTHORIZED` and `SCREEN_AUTHORIZED` are `False` there;
- a registry **snapshot** taken at import, a last-in-file **restoration** test comparing against it,
  and a **mutation control** proving that snapshot check binds.

The real command stays at **exit 2** with the actual L0 block. That refusal is correct and permanent.

The frozen plan's own `"RESERVED, UNSPENT"` and `"NOT EXECUTED"` strings are asserted **deliberately
and left unchanged**: they record the state at preregistration, and their survival is the evidence
that the plan was not rewritten after the results were seen.

## The immediate post-run failures

Recording the seeds makes the schedule unexecutable, which is correct. But 25 tests were written when
the block was unspent. **All 25 are in the two L0/L1 test files; zero elsewhere.** One root cause:
`check_schedule` now refuses, so the command exits 2 where the test expects 5, and the
"still unspent" assertions are false.

That was the state I reported before reconciling, and the transcript stands as evidence of it.

## Where the ladder stands

| gate | question | status |
|---|---|---|
| E1–E3b | artifact, headless, determinism, rules equivalence | PASSED |
| E4 | is T1j in a usable band? | **IN_BAND** at `mdPly` 6 |
| L0 | **what is the rate there?** | **observed 0.594, Hoeffding95 [0.424, 0.764] — includes 0.5** |

The anchor programme's original question — *is there an external reference this stack can be ordered
against* — now has a measured answer at one setting, with the caveat it started with intact. What it
does **not** have is a separation: at 64 games the interval still contains parity, so the honest
summary is that T1j and `calib020_0001` are **not distinguished** by this match, with T1j's point
estimate the higher of the two.
