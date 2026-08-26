# E4 seed reconciliation — spent stops execution, not reading

**Date:** 2026-08-26 · **Status:** RECONCILED, no execution. **No model, no JVM, no RNG, no
move, no game, no seed drawn, no retry, no push.** · Local, unpushed.

Basis: `main` @ `d50a244`, clean. Evidence: `evidence/2026-08-26-t1j-e4-seed-reconciliation/`.
The canonical run's JSONL and every runtime artifact are **byte-for-byte unchanged**: all 11 entries
of the screen's own `11_MANIFEST.sha256.txt` re-verify `OK`, and git reports no change in that
directory (`09_run_artifacts_unchanged.txt`).

> **A near-miss worth recording.** My first attempt at that check named a manifest that does not
> exist (`A_MANIFEST`) *and* suppressed stderr, so it produced no output — and a "count the non-OK
> lines" test read no output as zero mismatches. **The check had not run.** I reported it as passing
> before catching it. It is the same defect class as everything else on this ladder: a gate that
> exists but does not bind. The re-run above keeps stderr visible and names the real file.

---

## What was wrong

Recording the screen's spent seeds made `load_canonical_plan` refuse the canonical plan, because
one function answered two different questions at once. **60 tests failed**, and a fully understood
refusal escaped the CLI as `UNEXPECTED E4ReferenceError … exit 4`.

Review then found a third problem, older and worse: **a seed that had already been drawn from was
still reusable**, because the registry was a denylist and nobody had added it. Both corrections are
folded in below.

Two decisions settled the first, and neither was mine to make:

1. **The plan stays loadable as immutable historical evidence.** Spent must prevent *execution* —
   not parsing, verification, classification or replay analysis.
2. **The 32 seeds are not all "exposed."** 24 were drawn from. 8 were skipped by the early stop and
   never drawn from. All 32 are retired administratively, because the preregistered one-shot
   schedule completed and partial reuse would be selection bias.

## Two registries, kept apart

| | what it claims | **within the canonical screen block** | why |
|---|---|---|---|
| `EXPOSED_SEED_INTERVALS` | **experimental exposure**: a schedulable seed was drawn from, and so struck off | **24 of the 32**: `[202612128,136)` ∪ `[202612144,160)` | calling all 32 exposed claims a draw that never happened |
| `RETIRED_SEED_INTERVALS` | may **not be used again** — a rule about the future | **all 32** | replaying only the 8 the early stop declined would choose tasks *after seeing the result* |

Those counts are **about the screen's block only**. The registries are workstream-wide, and after
this change hold:

| registry | seeds | contents |
|---|---:|---|
| `EXPOSED_SEED_INTERVALS` | **65** | 24 screen · 32 preflight-attempt-3 witnesses · 4+4 integration attempts · **1 = `90000001`** |
| `RETIRED_SEED_INTERVALS` | **32** | the canonical screen block |
| `TEST_ONLY_SEED_INTERVALS` | 100 | `[90009000,90009100)`, ineligible for any schedule (below) |

**Exposure is not simply "was drawn from."** Test-band seeds are drawn from constantly and never
appear in `EXPOSED`, and defining exposure as *any* draw would contradict that. What makes a draw an
exposure is that it happened to a seed **a schedule could have used**, which strikes that seed off. A
draw inside the test namespace strikes nothing off, because nothing there was ever available to
schedule — the draw is every bit as real, it simply costs nothing.

`seed_is_unavailable` = exposed **or** retired; both refuse execution. `seed_status` reports all four
flags separately, never merged into one word. Every one of the screen's 32 is classified against the
run's own records in `01_seed_taxonomy.txt` — **0 disagreements**, and `202612127` / `202612160`
remain free.

## The seed the registry was built to catch, and had missed

**`90000001` was schedulable and was drawn from twice.** Both draws are preserved, and the second is
cited from the **captured run**, not from the script that configured it:

| draw | where | what it proves |
|---|---|---|
| 1 | `…-preflight-attempt4/06_endpoint_screen_plan.json`, `seed_accounting.witness_demonstration` | an `rng_witness` frozen into the canonical plan — four values from each generator; the values *are* the draw |
| 2 — configured | `…-harness-qualification/04_qualify.py.txt:26, :50, :70` — `SYNTHETIC = 90000001`, `"seed": SYNTHETIC + i`, `REF.build(task(0, …))` | which seed was written down, and that `task(0)` resolves to `90000001` |
| 2 — **executed** | `…-harness-qualification/02_qualification.txt:18–28` | the call **ran**: completed, `(14, 13)` legal, *exactly one move made*, and **"search RNG advanced" + "readout RNG advanced"** |

The source line alone would have proved only the configuration. The run's own record is what proves
both generators moved. It was absent from the registry anyway, and a test of mine asserted **"the
designated test seed stays usable."** That is exactly the reuse condition the registry exists to
prevent. It is now recorded as exposed, and refused by task validation, schedule validation and
`rng_witness` alike.

**The cause was the shape of the check, not the missing entry.** `rng_witness` was a **denylist**: it
refused seeds it had been told about, so any seed nobody thought to list was drawn from silently and
spent without trace. It is now an **allowlist** — witnesses may be taken *only* from
`TEST_ONLY_SEED_INTERVALS`, a band ineligible for any schedule by construction, so a draw there
strikes nothing off. An arbitrary **schedulable** seed (`777777`, `12345678`, …) is now **refused
rather than drawn from and struck off in silence**. Tests draw from `90009001`; a separate
`SCHEDULABLE` constant covers the cases that need `validate_schedule_executable` to accept, and
`rng_witness` refuses *that* too, so it cannot be spent by accident either.

Refusal messages name the reason: a skipped seed is refused as `RETIRED`, a played one as `EXPOSED`.

## Two required functions, not one switch

```
validate_task_structure      validate_task_executable
validate_schedule_structure  validate_schedule_executable
```

Structure asks *is this well formed* and answers the same way forever. Executable asks *may this run
now* and its answer changed the moment the seeds were spent. A `require_unspent=` keyword was
rejected deliberately: **a switch defaults, and a default that can be switched off is the
gate-that-does-not-bind this workstream keeps finding.** Each takes exactly one required parameter,
asserted against the real signatures — the old ambiguous names are **gone**, so every caller must
name its question. Call sites in `07_call_sites.txt`.

| caller | asks |
|---|---|
| `verify_tasks` / `load_canonical_plan` | **structure** |
| `REF.build` (agent construction) | **executable** |
| harness screen mode (`_assert_screen_executable`) | **executable** |
| command `check_schedule` | **executable** |

## Exit 2, never exit 4

`schedule` is now a precondition in its own right, second of seven, consuming the object `check_plan`
verified. `check_plan` catches `E4ReferenceError` **as well as** `HarnessError`, and `verify_tasks`
converts one into the other, so the defect is closed at both layers. Reordering the preconditions
refuses (exit 2) instead of raising `KeyError` (exit 4).

The real CLI, fresh subprocess, real registries (`02_cli_refusal.txt`):

```
PRECONDITION REFUSED: [schedule] seed 202612128 was EXPOSED -- it has been drawn from --
                      and cannot be scheduled
exit: 2                      results file created?  No such file or directory
```

## The plan is still evidence

`03_plan_still_loads.txt` — the spent plan parses, matches both pinned digests, passes structural
validation, and **the completed run reclassifies to the identical verdict** through the harness's own
`classify_run`, with no seed, model or JVM:

```
  weak    played=16  score= 0.0  caps=0  decision=SATURATED_WEAK  recorded=SATURATED_WEAK  MATCH
  strong  played= 8  score= 7.0  caps=0  decision=IN_BAND         recorded=IN_BAND         MATCH
  joint = IN_BAND   recorded = IN_BAND   MATCH
```

That is bound as a test, not just an evidence file.

## The tests, and why the authorization gate is still tested

The 60 failures are gone. Where a test's **premise** changed with the run, the test changed with it
rather than being kept alive by relaxing what the code enforces — `..._accepts_the_canonical_schedule_in_screen_mode`
became `..._refuses_the_SPENT_canonical_schedule_in_screen_mode`, and every gate *before* seed
availability is asserted to still pass, so the refusal is proved to be about the seeds.

The awkward case is the **authorization gate**. Since `schedule` now refuses two gates earlier, every
exit-5 test would have stopped exercising authorization at all — a safety gate whose tests all went
vacuous. So tests whose subject is a later gate get an `unspent_block` fixture that lifts the block's
exposure and retirement **in process only**, never touching `SCREEN_AUTHORIZED`; a fresh-subprocess
variant keeps qualifying exit 5 end to end; and the earlier gate is asserted separately, unpatched.
`test_zzz_the_lift_fixture_never_leaks_into_the_real_registries` runs last and catches a fixture that
failed to restore. The env-var tests now assert something stronger than before: **stderr byte-identical**
to the unset baseline, against *both* gates.

### One defect I introduced and caught

`test_the_scheduled_seed_block_is_never_touched` asserted `not seed_is_exposed(...)` over the block —
which the lift fixture makes true **by construction**. It would have passed whatever the command did.
It now snapshots both registries and asserts they are unchanged, which tests the command rather than
the fixture and holds whatever state they are lifted to. The last control below is its proof: a
command that mutates a registry is rejected, where the vacuous version passed.

## Controls

Eleven injected defects, each restored and hash-verified, every one rejected, plus a passing baseline
(`05`, `06`):

| injected defect | rejected by |
|---|---|
| plan check stops catching `E4ReferenceError` | `..._catches_a_reference_error_too` |
| the two registries merged — all 32 called exposed | `..._records_drawn_and_undrawn_separately` |
| `verify_tasks` asks the executable question | `..._still_loads_and_verifies_after_its_seeds_are_gone` |
| the `schedule` precondition dropped | `..._refused_before_authorization_is_consulted` |
| a `require_unspent` switch reintroduced | `..._no_switch_that_turns_the_seed_check_off` |
| `_verified_plan` leaks `KeyError` | `..._refuses_rather_than_KeyError_...` |
| screen mode stops asking eligibility | `..._refuses_the_SPENT_canonical_schedule_...` |
| the command **mutates** a seed registry | `..._scheduled_seed_block_is_never_touched` |
| `90000001` dropped from the exposed registry | `..._90000001_is_recorded_as_exposed` |
| `rng_witness` reverts to a **denylist** | `..._is_an_ALLOWLIST_so_an_unknown_seed_is_refused` |
| a test-only seed becomes **schedulable** | `..._may_be_built_on_but_never_scheduled` |

**Baseline: 182 passed, exit 0** — the controls are not rejecting everything. Every file is restored
and hash-verified; the tree is byte-identical afterwards.

**Full suite: 3150 passed, 4 skipped, exit 0** (`08_full_suite.txt`), run against a hash-frozen tree
and re-verified identical afterwards. The 60 failures are gone and nothing else moved.

## Housekeeping

`git diff --check` over the whole amended change reports **0 findings in `scripts/` and `tests/`**.
It reports 124 in the evidence directory — trailing whitespace inside verbatim captures (a recorded
diff, pytest output, a CLI transcript). Those are **not** reformatted: an evidence file that has been
tidied is no longer the thing that was captured. (124 findings, from 248 output lines — `--check`
prints two lines per finding.)

## What this does not do

It does not re-run anything, re-open the block, or change the screen's verdict. It does not make any
seed usable again. It does not touch `SCREEN_AUTHORIZED`, which stays `False`. And the E0 caveat is
untouched: **`IN_BAND` is an ordering against `calib020_0001` in this stack, not a placement.**
