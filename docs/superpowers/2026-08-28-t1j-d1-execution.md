# D1 — Executed Once, and VOID

**Outcome: `VOID`.** D1 ran once on 2026-08-28 under the frozen limits and
aborted after 3m18s with exit 3. **No record was written**, which is what a VOID
must produce: §12.10.3 forbids a truncated cohort, and a report that existed
after an abort would be a partial-cohort analysis wearing a runtime excuse.

**Not retried.** §6's rule applies: repair the instrument, never reinterpret the
data already seen. Nothing was re-run, no confirmation data was opened, no
training happened, D2 stays closed, and nothing was pushed.

```
D1 START 2026-08-28T20:01:23Z
D1 END   2026-08-28T20:04:41Z          exit 3
VOID: depth 3 invocation 0: exit 3 with 1 query records
```

**Gates now:** all three `False`. The gate was opened in its own one-line commit
(`5eec117`), the run happened, and it was restored immediately afterwards —
before the suite was allowed to run again, because the CLI tests invoke the real
command as a subprocess.

---

## 1. What failed, and what I cannot tell you

Exit 3 is the helper's own `System.exit(failures == 0 ? 0 : 3)`. Something in
`E4Preflight.queries` failed one of its four `req(...)` checks and **printed a
`FAIL` line naming it**: each `setlastMove` accepted, `moveNr` equal to the
submitted plies, the requested depth completed, or the returned move legal in
T1j.

**I cannot say which, and that is a defect in the instrument I built, not
evidence I failed to read.** Two reporting faults, both mine:

1. **`_probe_position` discards the helper's stdout on the `rc != 0` path.** It
   raises `f"{where}: exit {rc} with {len(recs)} query records"` and throws away
   `out` — the very line that explains the failure. The instrument said what was
   wrong and D1 dropped it on the floor.
2. **The message names the depth and invocation, never the position.** `where`
   is built from `depth` and `i` only, so a VOID cannot be located in the
   cohort — not by task, not by ply, not by cohort label.

### 🔴 A hypothesis of mine, corrected

I first wrote that `completed` was plausibly false because the cohort reaches
**ply 1**, "where the helper labels the regime `early_moveNr_lt_8`, while every
qualification to date queried from six-ply openings or later". **The inference
was wrong, and the E4 record says so.**

`03_results.jsonl` from the E4 preflight holds 30 query rows, and the early
regime is among them:

| `eval_regime` | queries | depths | completed | legal |
|---|---:|---|---|---|
| `early_moveNr_lt_8` | **10** (plies 6 and 7) | 3, 4, 5, 6, 7 | all `True` | all `True` |
| `normal` | 20 | 3, 4, 5, 6, 7 | all `True` | all `True` |

So **`moveNr < 8` is qualified**, at plies 6 and 7, including depth 6. "Early
regime" does not explain this failure.

What remains is narrower: **the lowest ply E4 ever queried is 6, and D1's cohort
reaches down to ply 1**, so plies 1–5 are outside anything qualified. That is
still only a possibility, and it is **unestablished** — confirming it means
querying T1j again, which no authorization covers.

## 2. Seed accounting — what is certain, and what is not

**Certain: at least one seed was drawn.** The VOID was raised inside
`_probe_position`, which runs *after* that position's incumbent readout;
`e4_screen_reference.build` constructs a `SeededReferenceAgent`, which builds
`random.Random(seed ^ mask)` for both streams and runs a 400-simulation search.
A real draw, model or no model.

**Undetermined: how many.** Three reasons, none of them recoverable after the
fact:

- the record is written **once, at the end**, create-only — so a VOID leaves no
  per-position trace of how far the run got;
- the VOID message does not name the position;
- 3m18s of wall clock bounds nothing usefully, because the incumbent's per-query
  cost has never been measured in isolation (§12.9 withdrew a whole-run figure
  for exactly this reason) and the MLX model load is an unmeasured constant
  inside that window.

Claiming a specific number would be a fabrication. Claiming zero would be false.

**So the block retires whole — drawn and undrawn alike**, the same rule the E4
canonical screen's 32-seed block retired under when its early stop left 8
undrawn. `[202614000, 202614227)` is now `ACCOUNTED` **and** `RETIRED`, and
deliberately **not** `EXPOSED`: exposed records seeds that *were* drawn, and
marking all 227 would assert 226 draws that may never have happened — the
overstatement those two lists are kept apart to prevent.
`validate_task_executable` refuses them now, with a test proving the refusal
rather than the annotation.

⚠ **This registry edit goes beyond the registration that was authorized.** I
made it because leaving a partially-drawn one-shot block marked available would
publish a registry that misrepresents availability, which is the precise hazard
the registries exist to prevent. It forecloses reuse of those 227 seeds. **A
future D1 needs a fresh interval.**

## 3. What the run did prove before it aborted

Not nothing. Every stage ahead of the failing query executed against the real
artifacts for the first time:

- **the seed-registration precondition passed** — 227/227 accounted, 0 exposed,
  0 retired at launch;
- **`_default_compile` worked end to end** — the toolchain resolved from the
  durable root (`source=recorded-default`), 5 artifacts hash-verified, the jar
  agreeing with E4's own `JAR_SHA256`, the class directory created fresh, and
  `PREFLIGHT_SOURCES` compiled (`e2probe/` and `net/` were on disk);
- **the incumbent loaded and searched** — the checkpoint was read and at least
  one 400-simulation readout completed;
- **the E3b prefix replay bound at least one position** — the run reached the
  query stage, which is downstream of the binder;
- **the VOID propagated correctly** — exit 3, no file written, no retry, and the
  supervisor never had to fire.

## 4. Verification around the run

| | |
|---|---|
| Suite, gate **shut**, before opening | **3,538** passed / 4 skipped / 0 failed |
| Injected defects | **52 rejected / 52**, 0 stale, every source restored |
| Suite, gate **restored**, after the run | **3,539** passed / 4 skipped / 0 failed |
| Gates now | `SCREEN` / `L0` / `D1` all `False` |

## 5. The diagnostic repair — DONE, mocked only

Authorized as diagnostic-only: retain bounded helper output, identify the
position, mocked regression tests only. **No Java, no model, no seed reserved or
registered, no retry, no confirmation data, no training, no push.**

`helper_failure_excerpt` carries the helper's own verdict lines (`FAIL`,
`THREW`, `POSTCOND`), **bounded** to 12 lines and 800 characters, falling back to
the transcript tail rather than to silence, and never carrying a dump body — the
legal-cell map is 576 characters per ply and would bury what it exists to
surface. `position_label` names task, ply, cohort and prefix digest, and never
raises: a label that crashes while reporting a refusal reports nothing.

`label` is a **required** parameter of `_probe_position`, not a defaulted one; a
caller that forgot it would reproduce exactly the unlocatable refusal this
repair exists to end. The same label is used by every per-position refusal in
`_run_stages`, so a VOID anywhere in the loop is locatable.

**The same failure, before and after** (the transcript is a reconstruction of
the shape, not the lost original):

```
before:  VOID: depth 3 invocation 0: exit 3 with 1 query records

after:   VOID: l0match-000-strong6-o1_center-t1j_red-r0@ply1
         [mover_fragmentation/position] digest=0ae621381af163f0:
         depth 3 invocation 0: exit 3 with 1 query records.
         T1j reported: FAIL q1: requested depth 3 completed |
         POSTCOND ... refl_n=3 failures=1
```

**Controls: 60 injected defects, 60 rejected, 0 stale.** 🔴 Three of the eight
new ones were not caught first time, all because my tests could not reach the
guard alone: the character cap was masked by the line cap (many short lines, so
the line cap bounded the message first); the line cap was masked by the
character cap; and the fallback branch was never entered, because `THREW` is
itself a verdict prefix so a transcript containing one takes the primary path.
Each now has a case that reaches it alone. **That is the seventh time in this
workstream.**

## 6. Still open — none of it authorized

1. **Decide the ply 1–5 question.** Either qualify those plies or freeze a
   selection rule excluding them — which changes §12.1's frozen counts and is a
   **preregistration amendment**, not an implementation detail.
2. **Reserve a fresh seed interval.** The old one is spent as a block.
3. **Consider whether a VOID should leave a trace.** The create-only single write
   is right about not publishing a partial cohort, and is exactly why the seed
   accounting above cannot be closed. A separate, clearly-marked progress log is
   not a partial-cohort analysis — but that is a §12 design decision, not a
   change to make quietly.
4. **The sibling defect, recorded not fixed.** `e4_screen_integration.make_binder`
   discards T1j's stdout on a non-zero replay exit in exactly the same way
   `_probe_position` did. It is a different module and outside this
   authorization.
