# E3b attempt 3 — post-run provenance correction

**Date:** 2026-08-25 · **Status:** packaging correction to
[attempt 3](2026-08-24-t1j-e3b-attempt3.md), committed at `6377ae5`.
**No fourth qualification run.** Nothing about the attempt-3 result changed, and **no attempt-3
artifact was modified** — `A6b`, `A6c`, `A7` and `90_transcript.txt` are byte-for-byte as committed,
and all **19** of `A7`'s entries still verify.
· **E4 remains unauthorized.**

Added: `B0`–`B4` in `evidence/2026-08-24-t1j-e3b-attempt3/`, with a **separate post-run manifest**
so the run's own manifest is not regenerated.

---

## 1 — The binder covered three of five paths

`A6c` bound only the three Java compile inputs, while the attempt-3 card and commit message said
*the adapter* was bound. `t1j_adapter.py` and `tests/test_t1j_adapter.py` appear in the transcript's
section-4 manifest **and** in `A6`, but `A6c` never compared them. The claim outran the check.

`B0_post_run_binder.py.txt` requires **exactly the five repository paths `A6` records** and, for
each, that three digests agree:

| source | what it is |
|---|---|
| transcript | the digest **recorded as the run happened** |
| `A6` | the digest the evidence file asserts |
| committed blob | `sha256` of `git show 6377ae5:<path>` — **the object store, not the worktree**, so a dirty checkout cannot launder a mismatch |

All five bind. A path appearing more than once in the transcript keeps **every** digest it was seen
with, so an internally inconsistent transcript cannot collapse to a single value — one of the
controls exercises exactly that.

## 2 — The old "negative controls" never ran the checker

`A6c`'s controls compared a deliberately different string against `A6` by hand. That can only show
that two unequal strings are unequal; it says nothing about whether the parsing-and-binding logic
rejects tampered evidence. Every control below **calls `check()` itself**:

```
PASS  untouched evidence accepted (the acceptance control): accepted (0 failure(s))
PASS  transcript digest flipped for the ADAPTER:            rejected (1)
PASS  A6 digest flipped for the ADAPTER:                    rejected (1)
PASS  A6 row for the TESTS file dropped:                    rejected (3)
PASS  transcript line for the ADAPTER removed:              rejected (2)
PASS  committed blob tampered (one byte appended):          rejected (5)
PASS  transcript carries TWO different digests for adapter: rejected (1)
```

Six rejections and one acceptance — a checker that rejected everything would prove nothing either.

**The data was correct both times.** The adapter and test digests always matched across transcript,
`A6` and the committed files; only the checking was inadequate. This is the second time in this
attempt that my check, not my data, was the defect.

## 3 — `git diff --check`: 9 findings, not 18 lines

The attempt-3 commit message said "`git diff --check` reports 18 lines in `90_transcript.txt`",
conflating output lines with findings. `git diff --check` prints **two lines per finding** — the
location, then the offending source line.

`B3_git_diff_check.txt` records the full output and counts computed rather than eyeballed:

| measure | value |
|---|---|
| total output lines | 18 |
| **findings** | **9** |
| distinct files affected | 1 |
| distinct source lines | 9 |

All nine are the transcript's own `    | ` indent prefix applied to blank lines of captured output.
Evidence is create-only; they stand as they ran.

## 4 — The full-suite claim now has evidence

The attempt-3 commit message claimed "full suite 2904 passed, 4 skipped, 0 failed" while the
committed evidence bound only the focused 23-test run. Rather than drop the claim, it is now
evidenced: `B2_full_suite_rerun.txt` is a **labelled post-run re-run**, captured 2026-08-25 at
`HEAD = 6377ae5`, recording the command, the tree hash and the date, and it reproduces the figure
exactly.

```
==== 2904 passed, 4 skipped, 53 deselected, 8 warnings in 584.03s (0:09:44) ====
PYTEST EXIT: 0
```

Stated plainly: **this is a re-run, not the original run's output**, which was not preserved. It
runs the same committed tree, and the two agree.

---

## What is unchanged

Everything the attempt-3 card establishes, and every bound it sets — no unique mapping, no two
independent histories, no conversion from a bare `TwixtState`, no self-contained execution, no
strength, no coverage beyond three positions and 21 cap combinations, and the E0 uncalibrated-anchor
caveat untouched. This correction touches **packaging and record width only**.

`6377ae5`, `3e0e9b7` and `29eae41` stand as committed; this adds a fourth commit rather than
amending any of them.
