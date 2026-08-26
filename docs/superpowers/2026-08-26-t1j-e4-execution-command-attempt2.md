# E4 execution-command attempt 2 — corrective. RAN, PASSED. The screen is built and dormant.

**Date:** 2026-08-26 · **Status:** **RAN and PASSED.** Unit tests **0**, qualification **0**,
stderr **0 bytes**. **No model, no agent, no RNG, no JVM, no game, no scheduled seed.**
**Supersedes [attempt 1](2026-08-26-t1j-e4-execution-command.md)**, preserved unchanged.
· **`SCREEN_AUTHORIZED = False`. The 32-game screen remains UNAUTHORIZED.**

Basis: `main` @ `48f83d0`. Full suite: **3114 passed, 4 skipped, 0 failed**. 45 command tests.

---

## Why there is an attempt 2

Attempt 1 qualified the **refusal** and nothing else. Its authorized branch was a placeholder: it
created the results file, loaded the evaluator, **assigned a builder it never called**, made an
**unrelated seed-0 RNG**, and raised `NotImplementedError`. It never built the T1j runtime, context,
binder or state factory, never handed the canonical 32 to the harness, and never classified
anything.

**So my claim that only the line-37 change remained was false.** It is corrected here.

## The authorized branch is now real wiring

`_execute_screen` builds the T1j runtime and context, the state factory, the E3b per-ply binder and
the agent factory; loads the pinned reference **once**; and hands the **canonical 32 tasks** to the
qualified harness. Proven with substitutes — nothing enabled, nothing played:

```
PASS   the harness receives the CANONICAL 32 tasks, in order
PASS   ...matching the pinned ordered, dimension-projected digest
PASS   the ONE loaded evaluator is passed through
PASS   _state_factory / _binder / _agent_factory are the qualified make_* products
PASS   ply_cap 280      16 per endpoint      NO qualification budget — plays to terminal
PASS   the harness is given the PATH; it owns recording
PASS   the command itself created NO results file
PASS   the T1j helper is compiled first
PASS   wiring order: ['compile_helper','t1j_runtime','load_evaluator','agent_factory','run_harness']
PASS   the command created NO RNG (random.Random patched): 0
PASS   the authorized branch is NOT REACHED while SCREEN_AUTHORIZED is False
```

Source-level controls assert the branch really references `T1jRuntime`, `IntegrationContext`,
`make_state_factory`, `make_binder`, `make_agent_factory`, `H._run` and `_tasks`, and contains
**no** `NotImplementedError` and **no** `Recorder(`.

**Recording belongs to the harness.** The command passes a path; the harness opens it exclusively
and fsyncs each record. **No run-level RNG exists** — the only generators in a screen are the two
each reference agent derives from its own bound task seed.

Harness exits map one for one: `0→0`, `2→2`, `3→3`, anything else `→4`.

## One more honest correction

The harness header hardcoded `canonical_tasks_executed: 0`. A literal that always says zero would
still say zero on the day the real schedule runs, so it is now **counted** from the intersection of
scheduled and canonical task ids — with a test.

## The claim about the gate, narrowed

Attempt 1 said the constant could not be changed from outside the file. That is too strong: **a
Python module global is rebindable by any code that imports the module.** What holds, and is
demonstrated, is that **no supported input changes it** — no CLI option, no environment variable, no
configuration file, no import-time hook. Five env vars and five flags were each tried in a fresh
subprocess.

## Unchanged from attempt 1, re-run

Six preconditions in fixed order, each immediately fatal, all before the gate; no effectful seam
reached; no results file created; `random.Random` patched showed **0** generators. Eight
per-precondition controls. A fully valid invocation exits **5**; a precondition failure exits **2**.

---

## What this establishes, and what it does not

**Established:** the screen command is complete and dormant — the authorized branch wires every real
collaborator and hands the canonical ordered 32 to the qualified harness, and it is unreachable
while the constant is `False`.

**Not established, and not claimed:** that the screen has *run* — **no game, no move, no model, no
JVM**; that flipping line 42 is sufficient in practice, only that the wiring it reaches is complete
as far as substitutes can show; strength; absolute placement. `[202612128, 202612160)` is untouched.
