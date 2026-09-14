# H3 PILOT — SEED PREPARATION AND REGISTRATION

**RESULT OF RECORD, 2026-09-14.** Every figure below was read from the runs
packaged here, at the moment of recording, not carried forward from a report.

**NOTHING RAN.** No game was played, no seed drawn, no model loaded, no JVM
started. All EIGHT gates are False, `H3_PILOT_EXECUTION_AUTHORIZED` included.
The pilot's three output paths are absent. Pilot execution is a separate
decision and is not requested by this package.

## What was authorized, and what it covers

> "Proceed with pilot seed preparation and registration only… Choose a fresh
> 40-seed interval, re-prove it against the current registries and derived
> streams, register it as accounted, and compute `SEEDED_TASK_DIGEST` from the
> exact 40 seeded tasks. Then verify the real-builder path and run the
> closed-gate pre-run checks against that schedule and unused output paths.
> This authorizes **preparation, not play**."

Commits `f87e5d1..965c9f5` (8), all UNPUSHED. The push stays held.

## 1. THE BLOCK — `[202624000, 202624040)`, ACCOUNTED ONLY

| | |
|---|---|
| accounted | 40 |
| exposed | 0 |
| retired | 0 |
| test-only | 0 |
| in `CONSUMED_SEEDS` | 0 |
| registration barrier | SATISFIED |
| `H3_PILOT_EXECUTION_AUTHORIZED` | **False**, before and after |

Registration is bookkeeping; permission is a separate review. A control asserts
exactly that (`registering the pilot block ALSO opened the execution gate`).

### Collision re-proof v10 — `01_collision_proof_v10.py`, `02_collision_proof_run.txt`
Re-proved against the registries AS THEY NOW STAND, all three spent H2 blocks
included. Exhaustive, not sampled: 6,357 prior seeds over six categories,
31,785 prior values with derivations, **0 direct overlaps, 0 derived-stream
collisions**, own derivations injective (200 = 40 × 5). Seventeen collision
controls and four gap-policy controls: **all REJECTED**.

🔑 The candidate is excluded from the prior set **BY IDENTITY**, in the GAP
check as well as the overlap check — D1 §14's lesson. The gap floor is the
candidate's **own size**, 40, the smallest any round has used because every
earlier candidate was larger; the nearest actual boundary is **1,264** away
with nothing above, reported so a narrow choice could not hide behind a small
threshold.

## 2. THE SCHEDULE — both pins RECOMPUTED, twice

`task_digest` covers EVERY field of every task, so it moves whenever the task
shape moves. It moved twice this step, and both pins were recomputed from the
built schedules each time, never derived from one another:

| | assigning seeds | + the identity fields |
|---|---|---|
| `TASK_DIGEST` (unseeded) | `17516342…0dd43` | **`b972ce46beb637d45a56fe3b002eb88932652fc5134b5db3dbbf6702de192337`** |
| `SEEDED_TASK_DIGEST` | `9ba07945…7e3e8` | **`aa527cc9a1a7b1e657911171c63f19fc006909dd64518bd96de3ce4ddfabfba9`** |

Seeds are POSITIONAL: row *i* carries `202624000 + i`, so a count of seeds
drawn identifies WHICH.

## 3. 🔴 THE DEFECT THE PRE-RUN VERIFICATION FOUND

The verification's one substantive failure was `the registry admits the
schedule for EXECUTION`:

```
E4ReferenceError: task is missing ['reference', 'reference_sha1'];
build_reference_agent would refuse it
```

H3's tasks carried **none** of the fields the qualified construction path reads
off a task. The worse half was not even reached: `e4_screen_integration.
make_agent_factory` subscripts `task["reference_colour"]` DIRECTLY to decide
which side is ours, and `e4_screen_runner._enforce_evaluator` refuses a task
that names none. **Every pilot game would have died at ply 6** with
`[agent_construction] h3pilot-000-p00-inc_red ply 6: 'reference_colour'`.

### Why 597 controls and a green suite said nothing
Every seam test — **including the five that drive the REAL harness** — supplied
its own agent factory:

```python
def agent_for(task, mover):
    return lambda state: sorted(state.legal_moves())[0]
```

It takes the task and reads nothing off it. A test double more permissive than
the thing it stands in for hides exactly the interface it was written to cover.
The fixture now routes the way `make_agent_factory` does, by subscript, and
five existing tests failed until the field existed.

### And the verification was checking its own patch
Its real-builder loop read

```python
task = dict(t, reference="calib020_0001", reference_sha1="209cf2d4…")
```

so it reported **all 40 tasks building through the real builder** while the
SCHEDULED tasks carried neither field and the registry refused every one of
them. A check that supplies what the thing under test is missing is not a
check. It now passes the tasks exactly as the runner will, derives the colour
with the seam's own expression, and asserts FIRST that the scheduled tasks
carry the fields unpatched.

### Two more, found by reading H2's `check_schedule` beside H3's
* H2's ends with `REF.validate_schedule_executable`. H3's did not, so the digest
  pin was the ONLY thing between a frozen schedule and a second run of it. **The
  branch is reachable and it is the case that matters**: the pin fixes the
  schedule forever, but the registry moves — this block becomes EXPOSED and
  RETIRED the moment the pilot draws from it.
* The unseeded schedule matches its OWN pin, so it passes the digest
  comparison. Running the registry check "only when seeds are present" would
  have made seedlessness the way around the registry. It is refused outright.

### What is NOT carried, and why
`reference_sha256` and `rng_streams` are deliberately absent, and a test
records it. Nothing reads either (`_injective_streams` DERIVES the streams from
the seed rather than trusting a copy), and an unread field under a full-field
digest can disagree with the truth unnoticed. The seed→stream binding for this
block is proved in the collision re-proof instead.

## 4. RESULTS OF THE RUNS PACKAGED HERE

| | |
|---|---|
| PRE-RUN VERIFICATION (`06`) | **47/47 PASS, exit 0**, on a clean tree at `965c9f5` |
| FULL SUITE (`05`) | **4,635 passed / 0 failed / 4 skipped / 53 deselected**, 1,025.96 s, exit 0 |
| DEFECT HARNESS (`04`) | **602/602 defects rejected; 0 not caught; 0 indeterminate; 0 stale; 602 distinct injections; 0 duplicate labels; 0 orphan reasons; clean baseline over 491 distinct target tests PASS; PROBLEMS 0**, exit 0 |

## 5. 🔴 THE FIRST HARNESS RUN REPORTED FIVE INDETERMINATE, AND THAT IS THE POINT

Run 1 at `3d9b3f8`: **597/602 rejected, 0 not caught, 0 stale, 0 orphan
reasons, 0 duplicate labels, clean baseline PASS over 491 distinct target
tests — exit 4.** Every one of the five was a consequence of this step's own
changes, and one of them had never demonstrated its claim at all.

| control | why | disposition |
|---|---|---|
| the seeded pin is not recomputed | both digests moved with the task shape | reason re-observed |
| the schedule digest is not compared with the pin | its target tampered an UNSEEDED schedule, so the new seedless refusal fired first | **TARGET REPAIRED** — it now tampers a SEEDED schedule in `ply_cap`, a field the registry ignores, leaving the digest the only thing that can reject it |
| the generated openings are not checked against the frozen pin | **it was passing on the NO-SEED-BLOCK refusal, which fires whether or not the check exists.** Registering a block moved the symptom to the containment boundary | **TARGET REPAIRED** — it now asserts WHICH refusal |
| a half-seeded schedule is admitted | its reason was the "SEEDED_TASK_DIGEST is None" message, recorded while no pin existed | reason re-observed, and the new one is CORRECT: with the pin set, removing the guard hands a half-seeded schedule the seeded pin in silence |
| the runner ignores which pin applies | the unseeded digest inside the message moved | reason re-observed |

Every reason was **observed in a throwaway worktree through the driver's own
classify/evidence/stable path**, never predicted.

Run 2 at `965c9f5`, after the two repairs and five re-declarations, is the run
packaged as `04`: **602/602, PROBLEMS 0, exit 0.**

⚠ Run 1's headline — "597 rejected, 0 not caught" — was NOT the verdict, and
that is the mechanism working as intended. A count of rejections says nothing
about the controls that scored INDETERMINATE, and one of those five had been
scoring for a refusal that says nothing about the defect it names. **This is the
fourth round in which a headline count was not the verdict.**

An earlier run of the harness against `6a88b94` was STOPPED DELIBERATELY once
these changes landed, rather than being reported: it would have described a
superseded tree. Its checkout was removed and the main working tree was
untouched by it — which is this session's containment fix working, and the exact
failure the 2026-09-12 incidents were about.

## 6. SCOPE OF THESE CHECKS — stated with the result

They establish that the block is registered and disjoint, that the seeded
schedule matches its own recomputed pin, that the registry admits it for
EXECUTION, that all 40 registered tasks construct through the REAL builder on
both colour arms with their own seeds, and that the outputs are unused — **on
this working tree, at this commit, with the gate shut throughout**.

They establish NOTHING about H3's question. The pilot has never played a game.
Its launch path has never completed a run, it MAY time out by construction
(7,200 s is a CHOSEN limit, not a bound), and it **cannot produce a strength
verdict** in any case: its outcomes are "authorize design work on a full study"
or "close H3".
