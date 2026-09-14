# CORRECTION to `08_accounting_from_records.txt` and `09_outcome.md` — 2026-09-13

Both files are create-only records of the run and are **not edited**. This states
what they get wrong.

## What I wrote

> A deterministic argmax readout against a deterministic opponent replays one
> game per cell; the 46 repetitions then add nothing.

(`09_outcome.md`; `08_accounting_from_records.txt` says the same in other words,
and so does commit `9cca2d0`'s message.)

## Why it is wrong

**It asserts a mechanism the run does not establish.** Deterministic move
selection does not by itself force every repetition to be identical. The design
expected **seeded search variation** to differentiate the games inside a cell:
each task carries its own seed and its own derived RNG streams, and that was
supposed to be the source of diversity. Argmax removes the readout's randomness,
not the search's.

So the correct statement is narrower and is about *this configuration*, not about
determinism in general:

> **The seeded search variation the design relied on was insufficient here.** In
> 14 of the 15 completed cells it produced no transcript diversity at all — one
> distinct game per 46 repetitions, against a required floor of 42.

Why it was insufficient is **not determined by this run**. It is a question for
the replacement design, not a conclusion from this one.

## A second thing the numbers show, which neither file drew out

`o6_wide_right / t1j_red` produced **46 distinct transcripts** — the diversity
requirement's floor, cleared — and **every one of those games hit the 280-ply cap
with no winner**. Different moves did not give useful variation in *outcomes*.
Transcript distinctness is necessary for the repetitions to carry information; it
is not sufficient. A replacement design that checks only distinctness would pass
this cell and still learn nothing from it.

## What is unchanged

The official outcome is **VOID**, and no strength verdict is derived.

The diversity failure, however, does **not** depend on the VOID: those 14 cells
are **complete** at 46 of 46 games. Finishing the remaining 44 games could not
have changed them. The repetition scheme failed its own diversity requirement on
the evidence in hand, independently of why the run stopped.
