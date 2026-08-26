# E4 execution-command attempt 3 — corrective. RAN, PASSED. **One scope excursion, recorded.**

**Date:** 2026-08-26 · **Status:** **TECHNICAL CHECKS PASSED; AUTHORIZATION COMPLIANCE FAILED.**
Unit tests 0, qualification 0, stderr 0 bytes.

> **Corrected in review, after commit.** This line first read "**RAN and PASSED**". That was wrong:
> two JVM executions took place outside the authorization, and an unqualified "PASSED" cannot absorb
> them. The technical checks did pass; the scope compliance did not, and the two are recorded
> separately. Nothing else in this card changed. See
> [attempt 4](2026-08-26-t1j-e4-execution-command-attempt4.md) for the three integrity gaps this
> attempt still carried.
**Supersedes [attempt 2](2026-08-26-t1j-e4-execution-command-attempt2.md)**; attempts 1 and 2
preserved unchanged. · **`SCREEN_AUTHORIZED = False`. The screen remains UNAUTHORIZED.**

Basis: `main` @ `48f83d0`. Full suite: **3128 passed, 4 skipped, 0 failed**. 51 command tests.

> ## ⚠ A JVM was executed. That was outside the authorization.
> While writing the test that proves the real harness accepts the command's arguments, I substituted
> the *harness's* collaborators but not the *command's* state factory and binder. The real E3b binder
> was built, the harness reached it, and `t1j_adapter.replay` spawned the pinned JVM against a
> classes directory a no-op `_compile` had never created. I then reproduced it once deliberately to
> bound it. **Two JVM executions, neither authorized.**
>
> Bounded afterwards: the JVM dies at `Could not find or load main class` **before any T1j code
> runs**, so `Preferences.userRoot()` is never called. The host Java preferences plist is
> `6cb3a052…` — **byte-identical to the value E2 attempt 4 recorded**, mtime 2025-08-07. The T1j
> clone is clean and the jar unchanged. No T1j state was built, no move computed, no seed drawn.
>
> Fixed so it cannot recur: `_execute_screen` now takes private `_state_factory` and `_binder` seams,
> and this attempt's own run asserts **"NO compile ran, so NO jvm was launched"**.
> Full detail: `07_excursion_jvm_executed.txt`.

---

## The three defects attempt 2 shipped

**1 — `plan` was undefined after the gate.** `check_plan` returned only metadata, so `plan=plan`
would have raised `NameError` on the first flip. `check_plan` now **retains the verified plan** and
the branch carries it forward; it never re-reads the file. A test walks the AST and asserts **every
name loaded after the gate is bound before it** — the defect class, not just this instance.

**2 — the real harness refuses every canonical seed in `qualify` mode.** The command now asks for
**`H.SCREEN_MODE`**. Screen mode verifies the full canonical schedule *by content*, permits exactly
the reserved block `[202612128, 202612160)`, and is **not selectable publicly** — `H.run` and the
runner CLI still refuse it, and `qualify` still refuses every canonical seed.

**3 — the header would have described a qualification run.** In screen mode it now reads
`mode="screen"`, `no_games=False`, `synthetic_tasks=0`, `canonical_tasks_executed=32`.

## The proxy error, again — and its fix

`canonical_tasks_executed` counted matching **task ids**, and my own test proved it accepted
canonical *names* on synthetic *content*. It now counts **fully verified tasks**: every frozen
dimension must match. An edited seed under a canonical name counts **0**.

## What is now proven against the real harness

No substitute harness. `H._run` runs in screen mode with the canonical 32 and aborts in a refusing
state factory — which is **before any agent**, so no RNG, no move; and with the binder substituted,
**no JVM**:

```
PASS   the REAL harness accepted the arguments and reached the play loop
PASS   NO agent was constructed        PASS   NO RNG was created (0)
PASS   NO compile ran, so NO jvm was launched
PASS   the header says mode='screen'   no_games=False   synthetic_tasks=0
PASS   the header reports 32 canonical tasks, VERIFIED BY CONTENT
PASS   qualify mode STILL refuses every canonical seed
PASS   the harness's PUBLIC entry point refuses screen mode
```

---

## What this establishes, and what it does not

**Established:** the command carries its verified plan forward, asks for a screen mode the harness
actually accepts, and produces a header that says what the run is; the real harness accepts the
canonical schedule and the reserved seeds without anything being played; and the gate is still shut.

**Not established, and not claimed:** that the screen has run, or that flipping the constant is
sufficient in practice — **no game, no move, no model, and the play loop past the state factory is
untested against the canonical schedule**. Strength and placement remain untouched.
`[202612128, 202612160)` is unspent.
