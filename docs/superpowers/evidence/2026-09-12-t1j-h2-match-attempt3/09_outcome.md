# H2 ATTEMPT 3 — RAN ONCE, **VOID**. No verdict. 2026-09-13

One authorized match on `[202622000, 202622736)`. It **VOIDED on its own
28,800 s deadline at game 692 of 736**, wrapper exit **3**.

    run_end: {"error": "AbortError", "event": "run_end",
              "games_completed": 692, "verdict": "VOID"}

    VOID: AbortError: [move] h2match-692-strong6-o8_contact-t1j_black-r2 ply 9:
    black raised whole-run deadline of 28800s exceeded and the run was
    TERMINATED mid-stage. VOID: no partial-cohort analysis is produced.

Start 2026-09-13T14:45:33Z, end 22:45:33Z — the full eight hours.

## THERE IS NO H2 VERDICT, AND NONE IS INFERRED

The design requires all 736 games. The frozen rules produce no verdict from a
VOID and **no report was written** (`09_report.json` is absent). The 692
completed games are **failure evidence** about the launch path and about the
design's structure. No rate, no score, no interval, no per-arm split.

## SEED ACCOUNTING

| | |
|---|---|
| block | `[202622000, 202622736)` — 736 seeds |
| ACCOUNTED | 736 |
| **EXPOSED** | **693** — `[202622000, 202622693)` |
| **RETIRED** | **736 — THE WHOLE BLOCK** |
| not drawn | 43 — `[202622693, 202622736)` |

692 seeds carry a completed game. **The 693rd is claimed as drawn**, and that is
a *different* judgement from attempt 2's, where the uncertain seed emitted a
`task_start` and nothing else. Here the run aborted **inside** task 692 at ply 9:
an agent was built on seed 202622692 and nine plies were played with it. Its ply
records did not survive — they persist per completed game — but the draw did.
An unrecorded draw is still a draw.

Retired whole because a one-shot schedule was started and did not complete.

## 🔴 THE DEADLINE IS NOT THE ONLY THING THAT WENT WRONG

692 games produced **61 distinct transcripts**. Fourteen of the fifteen complete
cells hold **exactly one distinct game in 46 repetitions**.

The frozen degeneracy floor is `MIN_DISTINCT_PER_CELL = 42` of 46. **The gate was
not run** — a VOID writes no report — but these are the counts it would have been
given, and they are 1.

A deterministic argmax readout against a deterministic opponent replays one game
per cell; the 46 repetitions then add nothing. The single exception is
`o6_wide_right / t1j_red`, which has 46 distinct transcripts and is also the only
cell where every game hit the 280-ply cap — all 46 cap terminations in the run.
Those cap games are what consumed the deadline: they run ~6× longer than the
51-ply mean.

**A fourth block would meet the same structure unless the DESIGN changes.** That
is a question about H2's design, not about scheduling or seeds, and it is not
answered here.

## POST-RUN STATE, VERIFIED

* all seven gates **False** — H2's restored by the wrapper's `finally`, not by hand
* **no surviving processes**: no worker, no supervisor, no JVM
* exit **3** (VOID), not 10 — the gate restoration verified its own readback
* outputs create-only and intact; `09_report.json` absent because none was written

## NOT DONE

No retry. No replacement block. No verdict. The push stays held.
