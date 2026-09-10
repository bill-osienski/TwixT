# H2 implementation repair — seven production-boundary defects closed

Code, tests and one narrow card correction. Nothing ran: **all seven gates False**, the H2 block
accounted nowhere, no model, no JVM, no query, no game, no seed action, no push.

## 1. The mover rule — the one that would have voided half the schedule

`TwixtState.to_move` defaults to red and the runner records `mover` as the side that moved with `ply`
as the count after it, so **recorded odd plies are red and even plies black in every task, in both
arms**. My rule derived the expected mover from `colour_arm`, so **every `t1j_black` transcript would
have VOIDed**.

🔴 **My tests agreed with the bug because they built their fixtures from the same helper.** They now
take their expectation from `TwixtState` itself, stepping the real engine and asserting the rule
against it. `expected_mover` is gone; `colour_at_ply` replaces it, and a test asserts the old name no
longer exists. The card is corrected in the same terms.

## 2. There was no production path, and the public entry could be fed gameplay

`run_h2` took `tasks`, `play`, `identity` and `deadline_s`, so opening the gate would have authorized
**caller-supplied gameplay through the API** while the CLI refused. It now takes **the two output
paths and nothing else**, resolving the schedule, identity, deadline and play seam itself; the seams
survive on the private entry for tests.

The seam is **really wired**: it resolves the pinned toolchain, compiles the helper, builds the
binder, state and agent factories, and calls `e4_screen_runner.play_task`. The one H2 difference is
the config handed to the incumbent's builder — an `EvalConfig` whose `selection_mode` is `argmax`,
which `readout_from_eval_config` turns into the argmax readout. Everything effectful is imported
**inside** the seam, asserted by AST rather than by grep.

⚠ **It has never been exercised end to end.** Only its construction, its wiring and its refusals are
tested.

## 3. Schedule and identity are now bound completely

The design digest covers dimensions only, so a forged `reference_sha256` or forged `rng_streams` kept
the same digest and passed. Every task is now compared **field by field, recursively and
type-strictly** against the schedule this repository builds from the pinned source plan. The identity
is compared the same way — the whole of it, so a changed evaluation batch size, stall-flush count,
noise suppression, RNG mask, readout path or agent lifetime all refuse.

## 4. The wrapper supervises a worker in its own process group

The runner's deadline is polled **between games**, so one blocked game could overrun it indefinitely.
The wrapper now spawns the worker with `supervise` under an outer cap of 28,860 s, kills the whole
group on timeout, forwards an operator interrupt, and gives a timeout, an interrupt and a **surviving
descendant** their own exit codes. A surviving child never accompanies a success.

## 5. The symlink guard and the protected block

`os.path.exists` follows the link, so a **dangling symlink** read as absent and `O_EXCL` would then
fail on the link itself. `lexists` replaces it — H1's own correction, which I had repeated. File
creation moved **inside** the protected block.

## 6. The screen binds its input, and integrity binds before the screen

A 672-row vector with exactly 42 rows per cell passed. The screen now requires **one transcript per
canonical task id** and **exactly 46 games in every cell** before thresholding. And **result integrity
binds first**, so invalid rows are `REFUSED` rather than mislabelled `DEGENERATE DESIGN`.

## 7. The transcript evidence is persisted

Only the task result was written, so the reported diversity could not be recomputed by anyone who was
not there. The plies, the opening bound and the per-game transcript digest are now written to the
results file, and a test **recomputes the whole screen from the persisted digests**. Every mid-run
exit — refusal, crash or interrupt — now leaves a `run_end/VOID`.

## Verification

| | |
|---|---|
| H2 tests | **98 passed** (78 before) |
| controls | **484 / 484 rejected**, 0 not caught, 0 stale, 0 duplicates |
| clean baseline | **PASS** over 399 target tests |
| full suite | **4,345 passed, 4 skipped, 0 failed** |
| gates | all seven False |

## Nine controls were NOT CAUGHT across three rounds, and none was waived

An orphaned control from a rename; a redundant early readout check whose message the test then
asserted; a seam stub the test only read rather than parsed; an eager import the AST check missed
because `from . import x` carries the name in `names`, not `module`; an `argmax_cfg = cfg` alias that
a name-based check could not see; a forged-pin control aimed at a digest test; a per-cell game-count
control aimed at a test that refuses on the row count first; an injection that kept the call it meant
to delete behind an `if False`; and a 16-cell clause made unreachable by the new binding, deleted
rather than re-aimed. **My replacement for that last one was worse than none** — it injected the same
defect as an existing control at a test that could not see it, so it was dropped: one injection, one
owner, one control.

## Still not done

Nothing has run. Registering the block, opening the gate and executing the match remain a separate
authorization.
