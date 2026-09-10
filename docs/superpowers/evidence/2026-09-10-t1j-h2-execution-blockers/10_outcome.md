# H2 — four execution blockers and two hardening gaps closed

Code and tests only. Nothing ran: **all seven gates False**, **0 of 736** H2 seeds accounted, no
model, no JVM, no query, no game, no training, no push.

## 1. The run was guaranteed to fail during setup

`_default_compile` received a **fresh, unstarted** `Deadline`, whose first check raises "the run
deadline was never started" — **after** creating the classes directory. The seam now takes the run's
**one already-started deadline**, refuses an absent or unstarted one before compiling anything, and
the public entry starts that deadline before anything effectful. One deadline, one origin.

## 2. The report was returned and discarded

The worker printed that the verdict was "in" the results file — which holds only per-game rows — and
the frozen `09_report.json` was **never written**. A `REFUSED` report still exited 0 as COMPLETED.

The report is now written **create-only** to the frozen path, and outcomes map deliberately:
completed → 0, degenerate design → 11, cap-saturated → 12, refused → 7. An existing report refuses
rather than being overwritten.

## 3. Interrupt and deadline outcomes disagreed

Every `BaseException` wrote trace verdict `VOID`, including `KeyboardInterrupt`, while the supervisor
returned INTERRUPTED — the durable trace saying the instrument failed while the exit code said the
operator stopped it. The three-way **OK / VOID / INTERRUPTED** contract is restored.

And the cooperative deadline **could not interrupt a blocked game**: it is checked between games while
the outer supervisor waited 481 minutes. The run now arms `d1_probe._supervisor` from the **same
started deadline**, so a hung query is cut off by the run's own clock.

## 4. No cleanup ran between 736 games

H1 clears MLX state after every game; H2 called it **never**. The qualified cleanup now runs in a
`finally` after **every** game, completed or failed, with the count asserted and a failing game proven
to still clean up.

## Hardening

- **The schedule is pinned, not rebuilt.** Comparing a supplied schedule against a fresh build from
  the same live source plan freezes nothing — both sides move together. A **full-field digest** over
  every task field is pinned in the rules and checked.
- **`--worker` is no longer a public bypass.** It ran the match in-process, unbounded, outside the
  restoration boundary. The supervisor marks the child's environment and clears the marker afterwards;
  the worker refuses without it. It is not a gate and grants nothing.

## The tests now DRIVE the production setup

Per the review, the seam is exercised through **mocked effectful boundaries** rather than read: the
toolchain, compile step, evaluator, factories, harness and cleanup are replaced and the seam is then
**run**, so the clock, the argmax config, the cleanup and the game loop are reached together. The
config handed to the incumbent's builder is asserted to carry `selection_mode = "argmax"` with the
simulation count unchanged — the gameplay change, observed rather than inferred.

## Verification

| | |
|---|---|
| H2 tests | **117 passed** (98 before) |
| controls | **501 / 501 rejected**, 0 not caught, 0 stale, 0 duplicates |
| clean baseline | **PASS** over 413 target tests |
| full suite | **4,364 passed, 4 skipped, 0 failed** |

## Six controls were NOT CAUGHT, and two branches were deleted rather than defended

The supervisor's own refusal **masked the cooperative deadline check**, so the deadline control could
not fail: the tests now stand the supervisor down for that assertion and prove its refusal separately.
The field-by-field schedule comparison and the `KeyboardInterrupt` clause both turned out
**unreachable** behind, respectively, the full-field pin and the bare re-raise — deleted, with their
controls, rather than kept as decoration. Three more controls were aimed at tests that could not see
them: the report control at a helper rather than the worker, and the digest control at a path where
any change to the function breaks the pin anyway.

## Still not done

Nothing has run. Registering the block, opening the gate and executing the match remain a separate
authorization, and the seam has still never played a game.
