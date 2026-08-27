# L1 — execution wiring for the L0 64-game match, QUALIFIED

**Date:** 2026-08-27 · **Status:** QUALIFIED, NOT EXECUTED. **No model, no JVM, no engine game, no
RNG, no download, no install, no draw from `[202613000, 202613064)`.** · Local, unpushed.
**The 64-game match remains separately unauthorized.**

Basis: `main` @ `c7f87c9`, clean. Evidence: `evidence/2026-08-27-t1j-l1-execution-wiring/`.

---

## What L1 adds, and what it refuses to touch

Three new files — `l0_match_runner.py`, `l0_match_command.py`, `tests/test_l0_match_runner.py`.
**Nothing frozen is modified.** `git diff` against `HEAD` is empty for the frozen L0 plan,
`l0_match_rules.py`, `l0_match_plan.py`, and all five E4 screen modules.

The screen's `_run` is bound to the screen: 32 tasks, two endpoints, a band, an early stop. L0 is 64
tasks, one endpoint, no band, no early stop. **Widening `_run` to serve both would put a published
artifact's behaviour behind a mode flag**, and the screen's canonical run is history that must keep
reproducing. So the *loop* is reused and the *schedule policy* is written separately.

| reused, not rebuilt | from |
|---|---|
| `play_task` — opening bound first, one stateful agent per colour per task, move validated before application, both engines bound every ply, external cap | E4 harness |
| `Recorder` — exclusive-create, flushed and fsynced per record | E4 harness |
| `AbortError` + phase classification, `_enforce_evaluator`, the `_refuse_*` defaults | E4 harness |
| T1j runtime, state factory, **E3b per-ply binder**, agent factory | E4 integration |
| reference construction (delegates to the G3 builder) | `e4_screen_reference.build` |
| **all** reporting — rate, intervals, cap policy, claim discipline | `l0_match_rules.match_report` |

The runner computes no statistic of its own; a control that adds one is rejected.

**Deliberately not imported:** `early_in_band_forced`, `saturation_reachable`,
`cap_incompleteness_reachable`, `per_endpoint_decision`, `classify_joint`, `earliest_early_stop`. An
AST test asserts the L0 runner references none of them **and** that they all still exist for the
screen. There is no skip path in the loop at all — `task_skipped` and `stopped` do not appear in the
source.

## Two separate gates

`L0_EXECUTION_AUTHORIZED = False` is L0's **own** constant. The L0 command never reads
`SCREEN_AUTHORIZED` — asserted by AST, because one gate must never be openable by opening the other.
No argv, environment, config or import-time override; the env-var tests compare **stderr byte for
byte** against the unset baseline.

Seven preconditions, each immediately fatal, then the gate:
`plan → schedule → repository → jdk → jar → checkpoint → output_path`.

The real CLI, fresh subprocess, real pinned artifacts (`01`):

```
L0 MATCH NOT AUTHORIZED: L0_EXECUTION_AUTHORIZED is False.
  preconditions completed first: plan, schedule, repository, jdk, jar, checkpoint, output_path
exit: 5          results file created? No such file.   class dir created? No such file.
```

## A real defect, on the first end-to-end run

The reused screen checks raise `e4_screen_command.PreconditionError` — a **different class** from
L0's. Untranslated, `main` did not catch it and a fully understood refusal printed
`UNEXPECTED … exit 4`. Exactly the shape of the E4 `E4ReferenceError` escape. Refusals are now
translated at the delegation boundary: **reusing a check means adopting its failures too.**

## What was qualified

| requirement | how |
|---|---|
| fail-closed identity ordering | the identity header is fsynced **before** setup; an exploding setup still leaves it durable, and the failure is classified `PHASE_SETUP` rather than leaking |
| canonical schedule enforcement | match mode demands the frozen 64 by structure, pinned digest **and** full content against the loaded plan; four edits refused, and **no results file is created** by a refused match |
| all-64 completeness | a 63-row vector is refused; the reporter is the same frozen `match_report` |
| no early stop | 10 straight wins do not stop the run; `may_stop_early` is constant `False`; no skip path exists |
| exclusive durable recording | an existing results file is refused and **left byte-unchanged** |
| integrity aborts | default binder refuses; a binder failure is `PHASE_BIND`; a missing evaluator aborts rather than disabling the check; a cleanup failure after a recorded game is a run-level abort **with the game already persisted** |
| post-run reporting | a qualification run gets a **receipt, never a report**; a synthetic 64-row vector reports rate, Hoeffding (primary) and Wilson (nominal), 8×2 descriptive cells, cap policy |

## The enabled path was not just untested — it was broken

Every command test stopped at the closed gate, so the production assembly behind it was unqualified.
That is the E4 placeholder failure class, repeated: **proving the locked side does nothing does not
prove the enabled side is real.** Opening the gate in process with substitutes and actually calling
the `setup()` closure exposed **three defects that would have killed the first real run**:

| defect | consequence |
|---|---|
| `T1jRuntime(jdk_home=…, jar_path=…, classes_dir=…)` | it takes `java`/`jar`/`classes`/`ply_cap` — `TypeError` in setup |
| `make_state_factory(ctx)` | missing the `openings` it requires — `TypeError` in setup |
| no `trace` entries | the assembly order was unobservable |

The branch now mirrors the qualified screen caller exactly. Nine tests reach it and assert: the
frozen 64 handed over **in order with a matching digest**, `mode=match`, `_ply_cap=280`,
`_ply_budget=None`, **no band and no per-endpoint count**, all seven verified identities passed
through, the trace `compile → t1j_runtime → load_evaluator → agent_factory`, exclusive class-directory
creation, the loaded evaluator handed through by identity, **no RNG**, and **no scheduled seed**.
`subprocess.run` is watched: only `git` from the repository precondition ever appears — never `javac`
or `java`.

> **A control caught a further gap of the same kind.** The collaborator test substituted
> `_state_factory` and `_binder`, and `or` short-circuits — so `INT.make_state_factory(...)` was never
> called and the broken-arguments control passed **vacuously**. A second test now lets the *real*
> constructors run and proves the state factory received the frozen openings: it builds a 6-ply
> opening and rejects an unknown one.

## Mode decides what a reporting refusal means

Every `reported=False` became a `qualification_receipt` with exit 0, in **every** mode. So an
incomplete or malformed canonical match would have failed reporting and still exited successfully,
and the frozen `CAP_SATURATED_NO_RATE` outcome would have been mislabelled as a qualification
artefact.

| scenario | mode | record | exit |
|---|---|---|---|
| complete 64 | match | `match_report` | 0 |
| 33 caps (frozen no-rate rule) | match | `match_outcome` | 0 |
| 63 of 64 played | match | — | **ABORT [classification]** |
| malformed score | match | — | **ABORT [classification]** |
| synthetic (not the design) | qualify | `qualification_receipt` | 0 |

A `qualification_receipt` can never appear in a match run.

**One constraint I did not work around.** The rules module names `CAP_SATURATED_NO_RATE` only as a
literal, and L1 may not modify the frozen statistical rules to add a constant. It is mirrored in the
runner and the duplication is **bound by a test** that runs the real reporter over a cap-heavy vector
and compares, so drift is a test failure rather than a silently mislabelled outcome.

## Two scoping notes, stated rather than left to be found

**The seed-block gate is the third check of the same fact.** Through `_run`, `validate_l0_schedule`
refuses an out-of-block seed and the pinned digest covers `seed` as well, so `_assert_match_seed` is
never reached by that path. It stays as defence in depth — it would catch a schedule satisfying both
while `L0_SEED_BLOCK` itself had changed — and is therefore qualified **directly**.

**The full-content check is masked by the digest.** It is bound directly with the digest re-pinned to
an edited schedule, the realistic failure being a digest constant that drifted. The edit used is a
**seed swap between two tasks**, which survives every structural rule — both seeds in block, all 64
unique, 16 cells of 4 — so the content comparison is the only thing left that can notice.

Neither is an independent gate, and the tests say so. This is the same lesson as L0's: *a layered
gate needs a layered control*, and a check that a run cannot reach must be qualified where it can.

## Controls

**Twenty-six injected defects, each restored and hash-verified, all twenty-six rejected**, baseline 69
(`05`, `06`): the L0 gate opened · the L0 gate reading the *screen's* gate · the gate consulted before
the preconditions · match mode made publicly selectable · the canonical schedule check dropped ·
binding by `task_id` only · qualification accepting a reserved seed · the identity header moved after
setup · a setup failure left unclassified · the result recorded after cleanup · the default binder no
longer refusing · an early stop introduced into the loop · the runner computing its own statistic · a
qualification run emitting a report · delegated refusals no longer translated · the results file no
longer exclusive-create · the seed-block gate removed · **match mode emitting a receipt instead of
aborting** · **a cap-saturated match mislabelled as a receipt** · **the mirrored cap outcome string
drifting** · **`T1jRuntime` built with the wrong keywords** · **the state factory losing its
openings** · **the enabled path handing over synthetic tasks** · **setting a qualification budget** ·
**dropping the verified identities** · **the class directory no longer exclusive**.

Six of these initially failed to bind — two because my injections were too weak to be real defects,
two because an earlier gate masked the one under test. Both causes were fixed rather than the
controls retargeted away.

**Full suite: 3329 passed, 4 skipped, exit 0** against a hash-frozen tree, re-verified identical
afterwards, with no edits during the run.

## What L1 does NOT do

It plays no game, loads no model, starts no JVM, constructs no agent or RNG, and draws from no
scheduled seed. The authorized branch is complete production wiring and **dormant**. Enabling it is a
reviewed one-line change plus a separate authorization.
