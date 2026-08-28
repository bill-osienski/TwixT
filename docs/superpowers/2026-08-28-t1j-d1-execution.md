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

A plausible reading of the four candidates is that `completed` was false —
`completed = uab && cmp == depth + 1`, and the cohort reaches back to **ply 1**,
where the helper itself labels the regime `early_moveNr_lt_8`, while every
qualification to date (E3a, E4, L0) queried from six-ply openings or later.
**That is a hypothesis, not a finding.** Confirming it means querying T1j again,
which is execution this authorization does not cover.

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

## 5. What a repair would need — none of it authorized here

1. **Make the failure legible.** Carry the helper's stdout into the VOID message
   and name the position. Without both, the next VOID is as opaque as this one.
2. **Decide the ply-1 question.** Either establish that the qualified depths hold
   at `moveNr < 8`, or freeze a selection rule that excludes those positions —
   which changes §12.1's frozen counts and is a preregistration amendment, not
   an implementation detail.
3. **Reserve a fresh seed interval.** The old one is spent as a block.
4. **Consider whether a VOID should leave a trace.** The create-only single write
   is correct about not publishing a partial cohort, but it is why the seed
   accounting above cannot be closed. A separate, clearly-marked progress log is
   not a partial-cohort analysis — but that is a design decision for §12, not a
   change to make quietly.
