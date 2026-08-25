# E4 harness qualification attempt 4 — corrective. RAN, PASSED. No model loaded.

**Date:** 2026-08-25 · **Status:** **RAN and PASSED.** Driver gate exit **0**.
**No model loaded, no T1j, no game, no scheduled seed.**
**Supersedes [attempt 3](2026-08-25-t1j-e4-harness-qualification-attempt3.md)**; attempts 1–3
preserved unchanged. · **The 32-game E4 screen remains UNAUTHORIZED.**

Basis: `main` @ `904d7d6`. Evidence: 8 files plus a self-excluding manifest.
Full suite: **3022 passed, 4 skipped, 0 failed**. 109 harness tests.

---

## 1 — [P1] One endpoint counted as a complete screen

Completeness was checked against **whatever task list was passed**, so one weak task and its result
was accepted as a complete screen — and my own test asserted that. A verdict now binds to the
**verified canonical schedule**: the run must have executed those 32 frozen identities, covering
**both endpoints**, with every task played or skipped by a recorded early stop.

The consequence is deliberate: **a synthetic qualification run can never earn a screen verdict.**

```
PASS  a FULL canonical result set earns a verdict
PASS  a canonical run whose weak endpoint EARLY-STOPPED earns a verdict
PASS  NEGATIVE CONTROL: ONE endpoint only earns NO verdict
PASS  NEGATIVE CONTROL: a subset of the schedule earns NO verdict
PASS  NEGATIVE CONTROL: a single synthetic task earns NO verdict
PASS  a SYNTHETIC run earns a RECEIPT even with both endpoints and every task played
      withheld because: the run executed 2 task(s) that are not the canonical schedule of 32
PASS  ...and both completed games are still recorded
```

Zero tasks, missing, duplicated, alien and unjustified-skip all still earn receipts.

## 2 — [P1] A cleanup failure erased a completed game

You reproduced it: a seven-ply win left **every ply in the log, then an abort, no task result, and
`tasks_played = 0`**. The game had finished; only the record order lost it. The completed game is
now **persisted and counted before cleanup runs**; if cleanup then fails, the record stands and the
abort is appended at run level.

```
PASS  THE COMPLETED GAME SURVIVES — its record is durable
PASS  the win is recorded in full          PASS  the result was written BEFORE the abort
PASS  tasks_played = 1, not 0
```

Cleanup still runs once for every *started* task, aborted or not, and a cleanup failure after an
abort is recorded separately rather than replacing it.

## 3 — [P1] The identity gate had an off switch

`expected is None` returned silently, so **forgetting** to load or pass the evaluator was
indistinguishable from **deliberately disabling** the check — the required-argument lesson from E3b.
For any started task the reference side now **refuses** a missing expected evaluator; only the
classical anchor side bypasses evaluator identity.

```
PASS  NEGATIVE CONTROL: a MISSING expected evaluator is REFUSED on the reference side
PASS  ...and the classical anchor side still bypasses evaluator identity
PASS  NEGATIVE CONTROL: a task naming no reference_colour is refused
PASS  NEGATIVE CONTROL: a STARTED task with no evaluator aborts on the reference side
```

A task that names no `reference_colour` is refused outright rather than defaulting.

## Post-run provenance (added after `b020b0e`)

C-series evidence added separately; the card is amended only with this section. No runtime artifact
changed and `08_MANIFEST` still verifies 7/7.

- **A** — all 8 tracked rows match `git rev-parse b020b0e:<path>`; the checkpoint has no object in
  the commit and is reported *UNTRACKED (gitignored), disk-only*.
- **B** — every file under all prior evidence directories and cards, against the commit that added
  **that file**: **zero changed**.
- **B2 — declared amendments**, now three: the preflight attempt-4 card (`fee7a3b`), the harness
  attempt-2 card (`0a441eb`) and the harness attempt-3 card (`904d7d6`). Each is checked against its
  amending commit **and required to actually differ from its origin**.

Nine controls, all invoking the real checkers, all passing.

---

## What this establishes, and what it does not

**Established:** a screen verdict cannot be drawn from anything but the verified canonical schedule
across both endpoints; a completed game survives a cleanup failure; and the evaluator identity gate
cannot be silently switched off.

**Not established, and not claimed:** anything new about the model — **attempt 4 loaded none**, and
the one real call remains attempt 2's; anything about T1j — **the per-ply binder is still a refusing
stub**, so nothing binds two engines yet; that the agent plays well; game length or screen runtime;
and absolute placement.
