# ✅ H3 SEGMENT 0 — RAN ONCE, COMPLETED. 2026-09-18.

Authorized as a single execution on the verified tree at `340ff65`.
Run once. **Not retried, and no other segment was run.**

```
.venv/bin/python -m scripts.GPU.alphazero.h3_study_command --segment 0
```

| | |
|---|---|
| preflight, immediately before the gate | **exit 0 — 127 PASS, 0 FAILED, 0 PENDING** |
| wrapper exit | **0** (`EXIT_COMPLETED`) |
| start → end | `20:29:49Z` → `22:36:52Z` — **2.12 h** of a 3.00 h cap |
| terminal record | `{"event": "run_end", "games_completed": 148, "segment": 0, "verdict": "OK"}` |
| games | **148 / 148**, 74 / 74 pairs, 148 distinct `task_id`s |
| timed out | **false** |
| gate readback (from the FILE) | **`False`** — restored by the wrapper |
| process group | `group_cleared: true`; **0** workers, **0** java surviving |
| artifacts | **all three produced** |

## The three artifacts

```
00_launch_receipt.json    709 B    parent-owned, create-only
03_results.jsonl       81,302 B    148 task_results + header + transcripts
04_trace.jsonl         14,494 B    run_start, 148 × task_start/task_done, run_end
09_report.json         22,671 B    74 pairs scored
```

🔑 **The receipt is new machinery and this is its first completed run.** It was
added after segment 0's VOID left no durable record at all; here it records
`outcome COMPLETED`, `exit_code 0`, `worker_exit 0`, `group_cleared true`,
`gate_restored true`, `gate_readback "False"`, the seed block, the segment digest
and the existence of all three artifacts.

## Seed accounting

**`[202628000, 202628148)` — EXPOSED 148 / RETIRED WHOLE.**

Every seed was drawn, and the count is **read from the records** — each
`task_result` carries its own `seed` — not derived from the plan. The 148 seeds
used are exactly the block, contiguous and distinct.

| block | accounted | exposed | retired |
|---|---|---|---|
| segment 0 `[202628000, 202628148)` | 148 | **148** | **148** |
| segment 0's VOIDed quarter `[202626000, 202626148)` | 148 | 0 | 148 |
| segments 1–3 `[202626148, 202626592)` | 444 | 0 | **0** |

🔑 **Segments 1–3 are untouched.** That is the 2026-09-18 isolation repair doing
its job: one segment's consumption costs one segment.

## Outcomes — descriptive, and NOT a verdict

* terminal reasons: **140 win, 8 cap**; `pairs_with_a_cap` 8
* **148 distinct transcripts from 148 games**; 74 distinct openings
* `duplicate_pairs` 0, `within_pair_identical` 0, `shared_continuation_pairs` 0
* timing: mean **51.5 s**, median 41.0 s, max 227.9 s

All five degeneracy gates report CLEAR.

## 🔴 THE REPORT ISSUES NO VERDICT, AND THAT IS CORRECT

```
verdict                 : "NO VERDICT"
verdict_note            : "fewer than the floor of completed pairs"
below_report_floor      : true
interpretation_withheld : true
is_strength_verdict     : false
```

The segment scored **74 pairs against the card's 148-pair report floor**. Its
primary figure (mean 0.743, interval [0.585, 0.901], favouring the incumbent) is
recorded and **explicitly withheld from interpretation**. One segment is a
quarter of the study, and §5.5 forbids reading a partial result as a strength
finding — the machinery enforced that without being asked.

**A segment result is not a full-study strength verdict. Nothing about strength
is established here.**

## Scope

Segments 1–3 and the push remain unauthorized. All ten gates are closed.
