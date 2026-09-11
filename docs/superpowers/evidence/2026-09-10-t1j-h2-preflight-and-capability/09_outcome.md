# H2 — preflight, terminal ordering, acquisition, capability and wording

Code and tests only. Nothing ran: **all seven gates False**, **0 of 736** H2 seeds accounted, no
model, no JVM, no query, no game, no training, no push.

## 1. The report is now part of the preflight

An existing or aliased `09_report.json` was discovered only **after 736 games** — spending the whole
seed block for a knowable path error. All three outputs are now required, create-only and **pairwise
distinct after canonicalisation**, checked before the worker is spawned and again before the run.

## 2. The terminal `OK` comes after the report is durable

`run_end/OK` was committed **before** the report was built or written, so a report failure exited
VOID while the durable trace said OK. The report is now built, written create-only and **fsynced
inside the protected block**, and only then is the terminal record emitted — carrying the outcome it
committed. The test asserts **no `OK` appears anywhere** in a failed run's trace, not merely that the
last line says VOID: a control that emitted an early OK and let the failure append its VOID satisfied
a last-line assertion while the trace carried both verdicts.

The remaining case the preflight cannot cover — the report path free at the start and **occupied by
the end**, six hours later — is exercised directly by creating it during the final game.

## 3. Output acquisition is incremental

Opening both files before tracking either meant a failure on the second left the first descriptor
untracked. Each is now registered with the exit stack **as it is acquired**.

⚠ **Stated rather than implied:** whether that registration happened is **not observable** in
CPython, because refcounting closes the file object as soon as it goes out of scope. The stack makes
the close deterministic and exception-safe, which is why the code does it, and **no control pretends
to test it**. What is tested is that the failure VOIDs, plays nothing, and leaves no other output.

## 4. A single-use capability replaces the forgeable marker

`H2_SUPERVISED_WORKER=1` was **caller-settable**, so it proved no provenance and contradicted this
module's own claim that no environment variable reaches the path. The supervisor now creates a
**0600 file containing a random token**, passes its path on argv, and the worker **reads and deletes
it** — so a second `--worker` on the same capability refuses, and none outlives the run. A test
asserts the wrapper reads **no environment variable at all**.

⚠ **What this is not:** an authentication boundary. A local caller with write access can fabricate a
file. It removes the accidental bypass and makes the capability single-use; the gate remains what
authorizes a run, and the source says so.

## 5. The degeneracy refusal no longer claims dependence

It said the repetitions "are not independent plays", which the screen **cannot** establish —
independent seeded games can produce identical transcripts, and §3.1 says plainly that the screen does
not test independence. It now says the preregistered **diversity** requirement failed, and says
explicitly what that does not establish.

## Verification

| | |
|---|---|
| H2 tests | **131 passed** (117 before) |
| controls | **512 / 512 rejected**, 0 not caught, 0 stale, 0 duplicates |
| clean baseline | **PASS** over 423 target tests |
| full suite | **4,378 passed, 4 skipped, 0 failed** |

## Five controls were NOT CAUGHT, and two were deleted as unobservable

The preflight **masked** the write-time checks it duplicates, so the report-overwrite and
terminal-ordering controls could not fail until a mid-run race test existed. An aliasing control was
aimed at a presence test. And two properties have no observable consequence: an `fsync` cannot be
detected in-process, and a descriptor's registration cannot be distinguished from refcounting. Both
controls were **deleted with a note** rather than kept as decoration — a control that cannot fail
reports success for work it never did.

## Still not done

Nothing has run, and the seam has never played a game. Registering the block, opening the gate and
executing the match remain a separate authorization.
