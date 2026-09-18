# 🔴 H3 SEGMENT 0 — RAN ONCE, VOID. 2026-09-18.

Authorized as a single attempt. **Attempted, VOIDed, and NOT retried.**

```
.venv/bin/python -m scripts.GPU.alphazero.h3_study_command --segment 0
```

| | |
|---|---|
| start → end | `14:24:12Z` → `14:24:13Z` (**1 second**) |
| wrapper exit | **3** (`EXIT_VOID`) |
| stdout | `VOID: FileNotFoundError: [Errno 2] No such file or directory: 'docs/superpowers/evidence/2026-09-15-t1j-h3-study-segment0/04_trace.jsonl'` |
| terminal trace | **NONE — never created** |
| results | **NONE** |
| report | **NONE** |
| completed games | **0 of 148** |
| seeds exposed | **0** |
| gate readback | **`False`** — restored by the wrapper |
| process group | clean; **0** surviving workers, **0** java |

## What happened

`_run_segment_unguarded` opens its trace with `os.open(trace_path, O_EXCL)` and
**never creates the parent directory.** `docs/…/2026-09-15-t1j-h3-study-segment0`
did not exist, so the very first durable write failed.

The order it reached, from the source:

```
_check_segment              ✓
check_seed_registration     ✓
check_output_paths          ✓        (absent paths are what it WANTS)
check_segment_schedule      ✓        (against the pin)
_require_seam_config        ✓        (config object built)
check_incumbent_identity    ✓
deadline.start()            ✓
os.open(trace_path, O_EXCL) ✗  FileNotFoundError — the directory is not there
```

**No model was loaded, no JVM started, no agent was seeded, no game began.**
Zero java processes were observed at any point. Exposure is **0 seeds**.

## 🔴 TWO DEFECTS, AND THE SECOND IS WHY THIS RECORD IS HAND-WRITTEN

**1. The segment runner never creates its output directory.**
`h3_study_generator.write_artifact` does exactly this
(`os.makedirs(os.path.dirname(out_path), exist_ok=True)` before the create-only
open) — the population writer learned it and **the segment runner never did.**
The pilot survived only because its evidence directory already existed.

**2. `h3_study_command` writes NO parent receipt.** `h3_generation_command` has
five: a parent-owned, create-only terminal receipt written after supervision and
gate restoration, added precisely so that *"a durable record on every path"* would
be true. The study command has none.

So a run happened, consumed its authorization, retired a seed quarter — and left
**nothing durable in the repository at all.** The only evidence is the wrapper's
stdout, captured here because it was redirected to a file. Had it gone to a
terminal the run would have been unrecoverable from the record.

That is the defect class this programme keeps finding, in its reporting layer:
the generation path was made to leave a record on every path, and the study path
was never given the same treatment.

## Seed accounting

Segment 0's quarter **`[202626000, 202626148)` RETIRES WHOLE**, per the
authorization: *"Segment 0's block retires whole on start."* The worker was
attempted, so it retires — **EXPOSED 0 / RETIRED WHOLE**, the same accounting H2
attempt 1 took when it VOIDed at task 0.

Segments 1–3 keep their quarters `[202626148, 202626592)`. **A future segment 0
needs a FRESH quarter with its own collision re-proof.**

## Scope

A segment result is not a full-study strength verdict, and this is not even a
segment result. **Nothing about strength was established.** No retry was made.
Segments 1–3 and the push remain unauthorized; the repair is not authorized
either and has not been attempted.
