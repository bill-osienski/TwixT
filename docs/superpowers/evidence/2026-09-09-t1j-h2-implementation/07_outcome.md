# H2 implementation — CODE AND TESTS ONLY, execution-readiness review

Nothing ran. **All seven gates are False** (H2's is new and shut), the seed block is registered
nowhere, no model was loaded, no JVM started, no query issued, no game played, no seed drawn or
exposed, nothing pushed.

## 1. What exists

| file | role |
|---|---|
| `h2_match_rules.py` | frozen design numbers, the transcript identity, the per-cell screen, the parity verdict, the report |
| `h2_match_plan.py` | the 736-task schedule, positional seeds, the readout mode carried per task |
| `h2_match_runner.py` | the gate, three barriers, identity binding, schedule and seed checks, deadline, the injected play seam |
| `h2_match_command.py` | the wrapper: exit codes and unconditional gate restoration |
| `tests/test_h2_match.py` | **78 tests**, synthetic fixtures and a mocked play seam only |

**Verified against the card, from the modules:** 736 tasks, 16 cells × 46 reps, parity 0.50, decisive
bands 0.4499 / 0.5501, minimum 42 distinct transcripts per cell, 480-minute deadline, 120-second per
call, schedule digest matching its pin, and the wrapper exposing only `--results` and `--trace`.

## 2. The three barriers, separate on purpose

1. **The gate** — `H2_EXECUTION_AUTHORIZED = False`, read by the public entry before anything else. A
   test asserts the real repository state and drives the entry to prove no file is created behind it.
2. **Registration** — reads `ACCOUNTED_SEED_INTERVALS`, never writes it, and checks **every** seed:
   registering all but the last still refuses. A test asserts the block is accounted **nowhere** today.
3. **Output paths** — create-only, and results and trace must canonicalise to two different files.

Registration does not open the gate and cannot: a test registers the block and asserts the gate is
still False.

## 3. What the transcript machinery actually binds

The frozen definition is implemented as written, and the tests are the interesting part:

- 46 records with identical gameplay but different `seed`, `task_id`, `rep` and timing hash to **one**
  transcript, and the cell **fails** — the vacuity the definition exists to prevent.
- Changing **one** played move yields a distinct transcript, so the comparison is not inert in the
  other direction.
- **Removing the first ply** and **removing the final ply** both refuse, which plain contiguity cannot
  see; a duplicate and an out-of-order record refuse too.
- **Flipping every mover** refuses, which an alternation check cannot see — the test asserts the
  flipped sequence still alternates before asserting the refusal.
- A terminal reason outside `("win", "cap")` refuses; there is no resignation.
- A one-collapsed-cell design **fails** while its **global** distinct rate is 93.9%.
- A failing screen leaves `interval` and `score` as `None`: the interval is never computed.

## 4. Verification

| | |
|---|---|
| H2 tests | **78 passed** |
| injected-defect controls | **466 / 466 rejected**, 0 not caught, 0 stale, **0 duplicates** |
| clean baseline | **PASS** over 383 target tests |
| every source restored | True |
| full suite | **4,325 passed, 4 skipped, 0 failed** |
| gates | **all seven False** |

**35 controls are new**, one per frozen property: the parity threshold and the side it is read from,
the sample size, the per-cell threshold, a transcript carrying the seed, a transcript dropping the
moves, contiguity replacing the exact span at each end, movers merely alternating, an unchecked
terminal reason, coerced coordinates, a global rather than per-cell screen, a vacuously-passing
missing cell, the interval computed despite a failing screen, a dropped interval standing, seeds by
membership rather than position, a missing readout mode in the plan and in validation, a short
schedule, a gate defaulting open, an entry that never reads it, a barrier checking one seed, a
disabled barrier, overwritten outputs, one path serving as two, an identity keeping the old readout,
inert settings left looking active, an unchecked model, an unchecked digest, a dead deadline, a
skipped malformed ply, and three wrapper failures.

## 5. Five controls were NOT CAUGHT first time; none was waived

1. **A verdict control aimed at a parameter it could not change.** Reading the wrong interval bound
   still returns `NOT_STRONGER` for a below-parity interval; it is the **inconclusive** case that
   moves. Re-aimed.
2. **A value carried twice.** The report holds the interval's standing at top level *and* inside
   `overall`; the control removed the copy the test does not read. Re-anchored to the one it does.
3. **An injection that injured nothing.** My "skip the malformed ply" defect still called the
   transcript builder behind an `if False`. Rewritten to skip it outright.
4. **A wrapper path no test reached.** With the gate shut, `main` returns on the unauthorized branch
   and never reaches the `finally`, so disabling the finally's check changed nothing. A new test
   drives the gate-open path — the one a real run takes — and another asserts a real open gate in a
   decoy file is left closed after a refusal.
5. **An unreachable `return`.** Restoration's final readback was never exercised: a missing file and a
   two-gate file both refuse earlier. A new test drops the write with a patched `os.replace`, so the
   file still holds an open gate and only reading it back can tell.

## 6. Execution readiness — what is still missing by design

**The play seam is injected and this repository supplies none.** `run_h2` takes `play`, the wrapper
wires nothing to it and says so in its refusal, so the module cannot start a game even with the gate
open. Wiring the seam to the qualified harness belongs to the execution authorization, along with
registering the seed block and opening the gate.
