# E4 canonical screen — RAN ONCE. **Verdict: IN_BAND.**

**Date:** 2026-08-26 · **Status:** the canonical 32-task screen **executed exactly once** from
`a8b3994`. **Exit 0, 9m 48s, 913 durable records, stdout and stderr both empty.** No retry, and
there will be none. The gate was restored to `False` immediately afterwards (`a246a61`).
· **Seeds `[202612128, 202612160)` are SPENT.**

---

## The result

| endpoint | `mdPly` | games | T1j score | caps | decision |
|---|---:|---:|---:|---:|---|
| **weak** | 3 | 16 | **0.0** | 0 | `SATURATED_WEAK` |
| **strong** | 6 | 8 | **7.0** | 0 | `IN_BAND` |

**Joint verdict: `IN_BAND`.** A larger match is permitted at the in-band endpoint.

The early stop fired at the **earliest game it mathematically could** — game 8 of 16 — because at
score 7.0 with zero cap terminations, neither saturation remains reachable: `7 + 8 = 15 < 15.2` and
`7 > 0.8`. Eight tasks were skipped and recorded as such.

**T1j at depth 3 lost every one of its sixteen games**, in both colour arms and across all eight
openings. **At depth 6 it won seven of eight**, four as red and three as black. That is the finding:
`mdPly` is not merely a dial that changes moves — it moves the *result*, decisively. It is exactly
what twixtbot's `trials` was not, and it is why that anchor failed where this one did not.

Every game ended in a **win**. Zero draws, zero cap terminations; game lengths 25–64 plies, median
39, against a cap of 280 that was never approached.

## What the run bound

The header carries the identities, fsynced before setup: repo HEAD `a8b3994…`, plan digest
`f5a21395…`, jar `53ec95e4…`, checkpoint `34c79c0d…`, four JDK components,
`canonical_tasks_executed = 32`, `mode="screen"`, `no_games=False`, `ply_budget=None`. Every ply of
every game was bound through the E3b binder; **829 plies, zero divergences** — a single one would
have aborted the run.

## Closeout, and two things that need your decision

The gate is shut (`a246a61`, one line, nothing else) and the run's JSONL and generated class
directory are preserved unchanged.

Recording the spent seeds — simply true — has a consequence I did **not** anticipate and have **not**
resolved:

**1. The canonical plan is now unloadable.** `load_canonical_plan` → `verify_tasks` →
`validate_schedule` → `validate_task`, which refuses exposed seeds. **60 tests fail — 29 command, 31
runner, and zero elsewhere in the suite.** Whether a spent schedule *should* be permanently
unloadable is a real question: it enforces "never again" exactly, but 60 qualified tests encode the
opposite premise. **I have not chosen, and I have not rewritten the tests to match whichever answer I
preferred.**

**2. A defect, unambiguous either way.** `E4ReferenceError` is not a subclass of `HarnessError`, so
`check_plan` does not catch it and it escapes the CLI as `UNEXPECTED … exit 4`. A refused plan is a
**precondition refusal, exit 2**. Flagged, not fixed — fixing it is beyond this phase.

---

## What this establishes, and what it does not

**Established, bounded:** against `calib020_0001` at 400 simulations, in this stack, under the frozen
schedule and rules — T1j at `mdPly` 3 is **decisively weaker** (0/16), and at `mdPly` 6 lands
**in band** (7/8 over the games played). The dial moves the result.

**Not established, and not claimed:**

- **Absolute placement.** T1j is uncalibrated. This is an **ordering** against these bytes in this
  stack, exactly as E0 warned — not a rating.
- **The strong endpoint's 16-game rate.** Eight games were played; the early stop is by design, so
  the magnitude inside the band is unmeasured. The observed 7/8 sits near its top.
- **Statistical strength.** Sixteen and eight games, with no interval preregistered beyond the band.
- **Anything about the product**, medium difficulty, or any checkpoint other than this one.
