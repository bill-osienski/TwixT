# The L0 canonical 64-game match — RAN ONCE. T1j scores 0.594.

**Date:** 2026-08-27 · **Status:** EXECUTED, once, exit **0**. Gate restored immediately.
· Local, unpushed. · **`[202613000, 202613064)` is SPENT. The match will not run again.**

Basis: plan published at `c7f87c9`, wiring at `f5e3582`. Enabled in `9805c19`, restored in `9483948`.
Evidence: `evidence/2026-08-27-t1j-l0-canonical-match/`.

---

## The result

**T1j scored 38.0 of 64 = 0.5938** against `calib020_0001` at `mdPly` 6.

| | |
|---|---|
| **Hoeffding 95% (PRIMARY)** | **[0.4240, 0.7635]** |
| Wilson 95% (nominal only) | [0.4715, 0.7054] |
| cap terminations | **0** |
| plies min / median / max | 25 / 41 / 88 (cap 280, never approached) |

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

## 25 tests now fail, and I have not rewritten them

Recording the seeds makes the schedule unexecutable, which is correct. But 25 tests were written when
the block was unspent. **All 25 are in the two L0/L1 test files; zero elsewhere.** One root cause:
`check_schedule` now refuses, so the command exits 2 where the test expects 5, and the
"still unspent" assertions are false.

Whether a spent one-shot should make its own wiring tests unrunnable is a design question. It arose
identically when the E4 screen spent its block, and there it was settled by an explicit ruling rather
than by me choosing. **I have not chosen, and I have not rewritten the tests to match a preference.**
The read path is unaffected and demonstrably works. The seeds are spent either way.

## Where the ladder stands

| gate | question | status |
|---|---|---|
| E1–E3b | artifact, headless, determinism, rules equivalence | PASSED |
| E4 | is T1j in a usable band? | **IN_BAND** at `mdPly` 6 |
| L0 | **what is the rate there?** | **0.594, Hoeffding95 [0.424, 0.764]** |

The anchor programme's original question — *is there an external reference this stack can be ordered
against* — now has a measured answer at one setting, with the caveat it started with intact.
