# The qualified-builder repair — argmax constructs, everything else still refuses

Code and tests only. **The seed retirement stands** (736 accounted, 0 exposed, 736 retired), all
seven gates are False, and no match was run.

## What was wrong

`build_reference_agent` compared the supplied configuration with `eval_config()` and refused **any**
difference. The instinct was right and the check was too blunt: H2 is a one-field change to the
readout, so the qualified path could not build its incumbent, and the authorized 736-game match
**voided at task 0** before a single move.

## The repair, both halves

**One field may vary, and it is named.** `ADMISSIBLE_SELECTION_MODES = ("opening_temperature",
"argmax")`. Every other field — simulations, batch size, stall flush, board size, both temperatures,
the opening-temperature ply count, the move cap — must still equal the frozen research configuration
**exactly and type-strictly**.

🔑 **Admitting a field is not admitting every value of it.** `hoeffding_lcb` is a real, qualified
readout mode and is **refused anyway**: a study that needs it names it in a preregistration and adds
it to that tuple under review.

The comparison is now **field by field**, so a refusal names the field that differs. The old message
said only that something was wrong, which is how a one-field change looked identical to a wholesale
substitution.

## Verified construction, through the real builder

| case | result |
|---|---|
| argmax config | **CONSTRUCTED** — `selection_mode='argmax'`, `readout.mode='argmax'`, sims 400 |
| frozen config (unchanged reference path) | **CONSTRUCTED** — `readout.mode='opening_temperature'` |
| config omitted | falls back to the frozen one |
| `mcts_sims 800` | refused, naming `mcts_sims` |
| `mcts_sims 400.0` — **value equal, type differs** | refused |
| `temp_high 1` — int for float | refused |
| argmax **plus** `mcts_sims 800` | refused |
| `selection_mode 'hoeffding_lcb'` | refused as not admitted |
| a dict instead of an `EvalConfig` | refused |

The readout is asserted to **be** argmax, not merely named it, and the seam test drives the seam's own
closure into the **real** builder — the call the harness makes at ply 7, the one that aborted.

## Two tests deleted, exactly as they promised

The tests that pinned the pre-repair refusal said in their own docstring: *"When the readout change is
made admissible, this test must be inverted, and its failure is the reminder."* The repair made them
fail, on cue, and they are replaced by the section above. **The history is not rewritten** — it lives
in the void evidence and the commit record.

## Verification

| | |
|---|---|
| H2 tests | **162 passed** |
| controls | **531 / 531 rejected**, 0 not caught, 0 stale, 0 duplicates |
| clean baseline | **PASS** over 437 target tests |
| full suite | **4,411 passed, 4 skipped, 0 failed** |
| gates | all seven False |

Ten controls are new, covering both halves: the readout refused again, argmax dropped from the
admitted set, the agent built from the frozen config while the check saw argmax, every other field
admitted, the simulation budget exempted, loose comparison, any string admitted as a mode, a
wrong-typed config, the check skipped, and the fallback removed.

## Three controls were NOT CAUGHT first time, all mine

One rebound a **local** name inside the checker, which the builder never reads again — it injured
nothing; it now attacks the config actually passed to the agent. One was written inverted and still
raised for its target field. And one removed the type check while its test also changed the **value**,
so the refusal happened anyway — a new test differs **only** in type, which is the case the strict
comparison exists for.

## What is still separate

Another match, and the push. Neither is started. H2 would also need a **fresh seed interval**, since
the retirement stands.
