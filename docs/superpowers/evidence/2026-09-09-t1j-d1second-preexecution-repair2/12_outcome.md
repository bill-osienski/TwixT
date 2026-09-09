# D1″ second pre-execution repair — four validation gaps closed, STILL NOT RUN

Code, tests and plan only. **The acquired D1 record was not read**: 19 of 19 manifest entries verify.
No model, no JVM, no query, no seed, no registry edit, no gate change.

## P1 — depth records: container first, then type-strict

`int(d.get("depth", -1))` coerced, so `"6"` and `6.0` passed a type-strict contract, and a non-mapping
entry escaped as `AttributeError`. Now the **container** is checked (a non-empty list), then **every
entry** (a mapping, carrying `depth`, whose `depth` is exactly `int`), and only then is the depth-6
record selected. Every refusal is a named `D1SecondError`.

## P1 — the search budget is bound to the frozen configuration

**The gap:** a forged visit map totalling one passed every internal check, as long as
`root_total_visits` was changed to agree. Nothing tied the root to the search that was supposed to
have run, and the hypothesis is about what the search does to a move *after* the frozen budget is
spent.

`frozen_sim_budget()` reads the simulation count from `d1_probe.frozen_incumbent_identity()` — the
frozen L0 plan's own configuration — and **the module contains no such integer literal**, asserted by
an AST scan rather than a substring search, so the docstrings may still explain the rule in words.

⚠ **A real record that disagrees is a refusal to review, not a tolerance to widen.** No producer-side
assertion pins the visit total to the simulation count in one place, so this binding is stated in the
plan as exact and deliberately fails closed.

## P2 — `top2` is inventory-only, and the code says so where the plan does

The plan's inventory lists it; D1″ reads it nowhere. Rather than leave the exclusion as an absence,
the module declares `UNUSED_OBSERVABLES = ("top2",)` and the plan states that it is **not validated**.
Validating a field no statistic reads would add a refusal path with nothing behind it, and the writer
may legitimately record it as `None`.

## P2 — derived values must agree exactly

`math.isclose` admitted an altered record. `selected_policy_mass` and `root_top1_share` are written
*from* the maps the record preserves, so recomputing them reproduces the same float bit for bit. The
comparison is now exact, with no tolerance frozen anywhere — **zero `isclose` calls remain**, by AST.

## Verification

| | |
|---|---|
| D1″ tests | **93 passed** (78 before) |
| injected-defect controls | **430 / 430 rejected**, 0 not caught, 0 stale, **0 duplicates** |
| clean baseline | **PASS** over 350 target tests |
| every source restored | True |
| full suite | **4,247 passed, 4 skipped, 0 failed** |
| six gates | all False |

## Faults found in my own machinery this round

1. **Two controls were NOT CAUGHT, and both were redundant guards of mine.** `_real` already rejects
   `bool` (`type(True)` is `bool`, not `int`), so the extra `isinstance(x, bool)` clauses were
   unreachable and a control deleting one changed nothing. The duplicates are gone; `_real` is the
   single owner, and the control now weakens *it*, where the defect is observable. A depth test
   matched the word "depth", which the fallback message also contains; it now names the type refusal.
2. **Three inherited controls went stale** on the lines I simplified — reported loudly by the
   harness, re-anchored, and one dropped as superseded.
3. 🔴 **Two inherited controls were byte-identical to two others** — same file, anchor, replacement
   and target test, under different labels. They ran the same experiment twice and the tally counted
   one defect as two. Both removed, and **the harness now refuses duplicates** and prints the count of
   *distinct injections*, so an inflated tally cannot recur. The honest count is **430**, not 432.
4. 🔴 **My first state probe grepped the source** and reported two false alarms, because both matches
   were in comments describing the removed code. A check that greps source is not a check; the
   recorded probe parses the module instead.

## Still not done

The analysis has **not been run on the acquired record**. No D1″ value from real data exists, no
confirmation data has been opened, and no training is proposed.
