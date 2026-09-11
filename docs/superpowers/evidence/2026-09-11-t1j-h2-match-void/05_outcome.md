# H2 RAN ONCE → **VOID at task 0**. The seam had never played a game, and it could not.

The single authorized match was started and **aborted before its first move**. No verdict exists,
and none is manufactured from a partial run.

| | |
|---|---|
| window | `18:17:05Z → 18:17:06Z`, **1 second** of 480 minutes |
| exit | **3 (VOID)** |
| games completed | **0 of 736** |
| trace | `run_start` → `task_start index 0` → `run_end verdict VOID` |
| results file | one `header` record; **zero** task_result, ply or transcript records |
| gate afterwards | **restored by the wrapper**, unconditionally, on the VOID path |
| surviving JVM | none |

## The cause, exactly

```
AbortError: [agent_construction] h2match-000-...-t1j_red-r0 ply 7:
config is not the frozen research configuration: EvalConfig(... selection_mode='argmax' ...)
```

`twixtbot_g3_reference.build_reference_agent` compares the supplied config with `eval_config()` and
**refuses any difference at all**. H2 *is* a one-field difference. **The qualified construction path
cannot build the H2 incumbent**, so the design as implemented could never have run — not at game 1,
not at game 736.

## 🔴 Why no test caught it, which is the part worth keeping

**Every test of the production seam mocked `build_reference_agent`** — including the ones written
specifically to "drive the production setup through mocked effectful boundaries" rather than inspect
it. Mocking the collaborator that **enforces** a constraint is how the constraint stays invisible.
The mocks proved the seam *called* the builder with an argmax config; they could not prove the
builder would *accept* one, because the stand-in always did.

The behaviour is now pinned by two tests using the **real** builder: one asserting it refuses the
argmax config, one asserting it accepts the frozen config so the refusal is about the change and
nothing else. **When the readout change is made admissible, the first test must be inverted** — its
failure is the reminder.

I recorded the risk before the run ("the seam has NEVER played a game... if it fails at game 1 the
run VOIDs and retires all 736 seeds"). That was the right thing to write down and the wrong thing to
be satisfied with: the risk was not merely possible, it was **certain**, and one unmocked test would
have shown it in a second.

## Seed accounting, from the records

| | |
|---|---|
| **EXPOSED** | **0 of 736** — the builder refused before any agent existed, so no RNG stream was seeded and no seed was drawn. Zero ply records; claiming otherwise would assert draws that did not happen. |
| **RETIRED** | **WHOLE, 736** — a preregistered one-shot schedule was started and did not complete. A spent seed is now refused by `validate_task_executable`, asserted at that effect. |

⚠ **The counter-reading, stated rather than buried.** The rule's reason is that replaying part of a
schedule after seeing where it failed is selection. Here nothing was played and nothing was learned
about any game. I applied the frozen rule anyway, because relaxing one the moment it costs something
is how rules stop binding — **and that decision is the reviewer's to overrule.** A future H2 needs a
fresh interval either way unless it is overruled.

## Verification after the void

| | |
|---|---|
| controls | **523 / 523 rejected**, 0 not caught, 0 stale, 0 duplicates |
| clean baseline | **PASS** over 430 target tests |
| full suite | **4,389 passed, 4 skipped, 0 failed** |
| gates | all seven False |

Four controls are new: the spent block left un-retired, a spent seed still schedulable, the builder's
config check disabled, and the builder refusing even the frozen config so its refusal would say
nothing.

## What H2 needs before it can run again

1. A way for the qualified builder to accept the **argmax readout** — a change to committed,
   qualified code, needing its own authorization and its own controls.
2. A **fresh seed interval**, re-proved and registered, unless the retirement is overruled.
3. Then a new execution authorization.

None of that is started. The push remains unauthorized.
