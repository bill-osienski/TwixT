# E4 execution-harness qualification — RAN, PASSED

**Date:** 2026-08-25 · **Status:** **RAN and PASSED.** Driver gate exit **0** — unit tests 0,
qualification 0. **The pinned reference was loaded once and one real agent call was made, on
synthetic seed `90000001`. No game, no T1j, no scheduled seed touched.**
· **The 32-game E4 screen remains UNAUTHORIZED.**

Basis: `main` @ `fee7a3b`. Evidence: `evidence/2026-08-25-t1j-e4-harness-qualification/` — 7 files
plus a self-excluding manifest. Full suite: **2987 passed, 4 skipped, 0 failed**.

---

## The one real call

```
position p06_e3a_center: 6 plies, to_move=red, ply=6
PASS   agent colour red matches side to move
PASS   the single real call completed (validate_ply passed)
PASS   returned move (14, 13) is legal in our engine
PASS   search RNG advanced          PASS   readout RNG advanced
PASS   exactly one move made by this agent
PASS   the compiled graph was built by that inference
PASS   the call used the ONE loaded evaluator
```

`validate_ply` is the strict one — it requires `root_visit_count == 400` exactly, plus a finite root
value and defined means on every visited child. It passed on the first and only attempt; had it
raised, that would have been the qualification result.

## Model load, once

| check | result |
|---|---|
| sha1 before loading | `209cf2d4…` |
| sha256 before loading | `34c79c0d…` |
| size | 7,524,333 bytes |
| `compile=True` | yes (`_use_compile`) |
| graph compiled **before** any inference | no — lazy, as designed |
| graph compiled **after** the one call | yes |

The `compile=True` check is not just the flag: `_compiled_forward` is `None` before the call and
non-`None` after, so compilation demonstrably happened. Memory records why this matters — without
it MLX re-traces per `infer()` and accumulates Metal buffers to exhaustion, which a one-move smoke
would sail through and a long run would die on.

**One evaluator serves every construction.** Six synthetic agents built from the single loaded
evaluator, all holding the identical object by `is`; each seeded distinctly. Negative control: a
rebuilt evaluator is not mistaken for the reused one.

> **A control caught a real defect in my own check.** The reuse counter first used `id()`. CPython
> recycles ids for freed objects, and with short-lived stubs four rebuilt evaluators counted as
> **two** — a rebuild reading as a reuse, precisely the failure the control exists to catch. It now
> holds the objects and compares with `is`.

## The schedule cannot be injected or reshaped

`run(plan_path, results_path, *, mode)` — **paths only**, asserted by signature. No task, callable,
evaluator, cleanup hook, classifier or schedule is reachable through it; the private `_run` carries
those injection points and exists solely to drive fail-closed tests.

The plan is loaded and verified **by the runner itself** against a pinned sha256, and its tasks
against a pinned digest over the *ordered, dimension-projected* task list — order-sensitive, so a
reordering breaks it; value-sensitive, so an edit does; length-sensitive, so an addition, removal or
duplicate does.

Refused, each with its own control: **addition, removal, reordering, duplicate**, and edits to
**endpoint, depth, opening, colour, reference, reference hash, seed**. Also refused: `mode="screen"`
and `mode="games"` — the 32-game screen is not reachable from any mode.

> Each reshaping control first asserts the mutation actually changed the digest. One of them didn't —
> the "endpoint edit" set `strong` on a task already `strong`, a no-op that passed vacuously. The
> guard now fails an invalid control instead.

## Scheduled seeds stayed inert

All **32** canonical seeds refused by the runner; `rng_witness` refuses to draw from any of them;
and after the run, `[202612128, 202612160)` is still **not** exposed. Every agent here used
synthetic seed `90000001`.

## Stub loop, end to end on the real evaluator

Four synthetic tasks: records appended **in order as they happened** (`task_start` → `ply` →
`task_result`), cleanup called once per task, **one evaluator across the whole loop**, verdict
`IN_BAND`.

**Abort:** raised and stopped the run; partial records kept; the abort recorded; **no verdict
emitted from a partial run**. **Partial-result refusal:** an endpoint with fewer than `n` resolved
games is `INCOMPLETE`, and any `INCOMPLETE` endpoint yields `INCONCLUSIVE`. **All five joint
outcomes** reached through the runner's own classifier: `T1J_TOO_STRONG`, `T1J_TOO_WEAK`,
`BRACKETED`, `IN_BAND`, `INCONCLUSIVE`.

## A module that had to move

The frozen decision rules lived only in the preflight's **scratch** harness. The runner importing
them would have meant re-implementing them, with nothing preventing drift — so they are extracted
verbatim into `scripts/GPU/alphazero/e4_screen_rules.py`, and `tests/test_e4_screen_rules.py` pins
them against the same control table the preflight self-test used. One implementation, one source.

**74 harness tests**, no model, no game.

---

## What this establishes, and what it does not

**Established:** the pinned reference loads once with `compile=True` and demonstrably compiles; one
real agent call from a frozen position returns a legal move, passes `validate_ply`, and advances
both generators; one evaluator serves every construction, with rebuilding detected; the canonical
schedule cannot be injected or reshaped and the screen mode is unreachable; scheduled seeds stayed
inert; and recording, abort, cleanup, partial-result refusal and every joint outcome run end to end.

**Not established, and not claimed:** that the agent plays *well*, or plays at all beyond one move —
**one call, one position**; anything about T1j, which this qualification never touched; game length
or screen runtime; and absolute placement — the E0 caveat stands.
