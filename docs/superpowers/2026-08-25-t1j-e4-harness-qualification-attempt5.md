# E4 harness qualification attempt 5 — corrective. RAN, PASSED. No model loaded.

**Date:** 2026-08-25 · **Status:** **RAN and PASSED.** Driver gate exit **0**.
**No model loaded, no T1j, no game, no scheduled seed.**
**Supersedes [attempt 4](2026-08-25-t1j-e4-harness-qualification-attempt4.md)**; attempts 1–4
preserved unchanged. · **The 32-game E4 screen remains UNAUTHORIZED.**

Basis: `main` @ `2723d0e`. Evidence: 8 files plus a self-excluding manifest.
Full suite: **3037 passed, 4 skipped, 0 failed**. 124 harness tests.

---

## 1 — [P1] The schedule was bound by name only

The verdict gate compared **sets of `task_id`**. You reproduced two false accepts, and both are
exactly right: the 32 canonical names **in reverse order**, and the 32 names with the **first task's
seed edited**, each returned `(True, None)`.

The canonical schedule is ordered endpoint, depth, opening, colours, reference identity and seed —
not a bag of names. The gate now re-runs `verify_tasks`, which re-checks the pinned **ordered,
dimension-projected digest**.

```
PASS  NEGATIVE CONTROL: the 32 canonical names in REVERSE ORDER earn NO verdict
      because: the run did not execute the canonical schedule: task digest 1dab9ddf… != pinned
PASS  NEGATIVE CONTROL: an edited endpoint / t1j_mdPly / t1j_mdFixedPly / opening /
      colour_arm / anchor_colour / reference / reference_sha1 / reference_colour / seed
      each earn NO verdict — with task ids unchanged in every case
PASS  the UNTOUCHED canonical schedule earns a verdict
PASS  a canonical run whose weak endpoint EARLY-STOPPED still earns a verdict
```

Ten frozen dimensions, ten controls, each asserting the ids were **unchanged** so the control tests
the digest rather than the names.

## 2 — [P1] A result-write failure skipped cleanup

The completed result was emitted **outside** the cleanup structure, so a failing durable write —
full disk, closed descriptor, fsync error — left control immediately and cleanup never ran,
contradicting the once-per-started-task guarantee.

Recording and cleanup now sit under one structure that **always attempts cleanup**. If the write
fails, the run aborts in a new `result_recording` phase; if cleanup fails too, the **recording
failure stays primary**.

```
PASS  a result-write failure aborts, classified as PHASE_RECORD
PASS  CLEANUP RAN EXACTLY ONCE for the started task (got 1)
PASS  no task_result was written
PASS  an UNRECORDED game is not counted as played
PASS  when BOTH fail, the RECORDING failure is primary
PASS  cleanup was still attempted exactly once
PASS  the ordinary path still cleans up exactly once
```

A game whose result could not be written is **not counted as played** — `tasks_played` stays 0
rather than claiming a game the log does not contain.

## Post-run provenance (added after `f6c8b07`)

C-series evidence added separately; the card is amended only with this section. No runtime artifact
changed and `08_MANIFEST` still verifies 7/7.

- **A** — all 8 tracked rows match `git rev-parse f6c8b07:<path>`; the checkpoint has no object in
  the commit and is reported *UNTRACKED (gitignored), disk-only*.
- **B** — every file under all prior evidence directories and cards, against the commit that added
  **that file**: **zero changed**.
- **B2 — declared amendments**, now four: the preflight attempt-4 card and the harness attempt-2,
  attempt-3 and attempt-4 cards, each amended by its own post-run commit. Each is checked against
  its amending commit **and required to actually differ from its origin**.

Nine controls, all invoking the real checkers, all passing.

---

## What this establishes, and what it does not

**Established:** a screen verdict binds the canonical schedule's order and every frozen dimension,
not merely its task names; and cleanup runs exactly once per started task whether the game
succeeded, aborted, or could not be recorded.

**Not established, and not claimed:** anything new about the model — **attempt 5 loaded none**, and
the one real call remains attempt 2's; anything about T1j — **the per-ply binder is still a refusing
stub**, so nothing binds two engines yet; that the agent plays well; game length or screen runtime;
and absolute placement.
