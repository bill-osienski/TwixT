# D1″ pre-execution repair — all three gaps closed, STILL NOT RUN

Code, tests and plan only. **The acquired D1 record was not read**: its 19-entry manifest verifies,
all 19 files. No model, no JVM, no query, no seed, no registry edit, no gate change.

## P1 — a full-record validation pass, before any row is built

`validate_record` checks **every** position before `rows_from_d1_report` builds the first row, so a
record that fails at row 200 fails before row 0 is scored and a partially-computed analysis never
exists. Ten observables are required **by name**. Nothing is coerced.

**Type-strict:** a visit count must be an `int` (so `True` is not one visit) and non-negative; a
policy mass must be a finite non-negative real (`nan`, `inf`, `-0.5` and `True` are all refused); the
override flag must be a `bool`, so a truthy int cannot pass as one; a depth-6 move must be a pair of
`int`s.

**Consistency, not merely presence:** `n_legal`, `root_total_visits`, `selected_visit_count`,
`selected_visit_rank`, `selected_policy_rank`, `selected_policy_mass` and `root_top1_share` are each
**recomputed from the record's own `raw_policy` and `root_visits`** and must agree. An all-zero
policy is refused, because its ranking would be pure tie-break over every legal move.

## P2 — the complement is descriptive, and the plan was wrong

The plan claimed a small policy-eligible complement would yield `NO_GO — insufficient support`.
**It cannot.** The floors count positions, controls, cells and games in common-support cells over the
**unconditional** denominator, exactly as §3 froze; they never count rows with `rank_raw ≤ 5`. A
cohort with ample rows and almost no complement passes every floor.

The corrected statement, now in the plan and asserted by a test: **a small complement mechanically
limits how large `ss` can be, so it produces an ordinary `NO_GO`.** The complement is **reported
descriptively** per role as `[count, rate]`. **The frozen denominator and floors are unchanged** — a
conditional denominator would change the preregistered statistic after the fact.

## P3 — the descriptive summary, frozen exactly

"Summarised by role" was not a specification. The representation is now frozen in the plan, **before
any real value was seen**, and implemented to match: per role, `n`; `overrode_leader` as
`[[false, c], [true, c]]`; `selected_visit_rank` and `selected_policy_rank` as **count histograms**
`[[rank, count], …]` ascending by rank with zero counts omitted; `ss_rate`; and
`complement_rank_raw_le_k`. Both rank fields are now carried onto the analysis row.

**Histograms, not means:** a mean rank cannot distinguish "usually the visit leader, occasionally
something odd" from "drifts everywhere".

## Verification

| | |
|---|---|
| D1″ tests | **78 passed** (39 before this repair) |
| injected-defect controls | **424 / 424 rejected**, 0 not caught, 0 stale |
| clean baseline | **PASS** over 342 target tests |
| every source restored | True |
| full suite | **4,232 passed, 4 skipped, 0 failed** |
| six gates | all False |

**18 controls are new**, including: the entry skipping validation; validation running per-row instead
of as a full pass; each type rule relaxed; each consistency check disabled; a bool passing as the
rank it equals; the complement turned into a floor; the complement never reported; the histogram
averaged, ordered by count, or padded with zero counts; and the row hardcoding either rank field.

## Five controls were NOT CAUGHT first time, and all five were my tests, not the code

1. **A bool that equals the right number.** The consistency test changed a rank to a *wrong* value,
   which a loose `!=` also refuses. `True == 1` is the case only a type-strict comparison catches, so
   a test now sets `True` where the correct rank is 1.
2. **A refusal matched on the wrong word.** The depth-6 type test used an *illegal* move too, and the
   legality message also contains "depth". It now matches "pair of ints".
3. **A histogram whose count order happened to equal its rank order.** With counts `{1: 2, 4: 1}`,
   sorting by descending count gives the same list. A new test uses ranks 1, 5, 5, 9 — rank 1 is the
   rarest, so the orders differ.
4. **Contiguous ranks hid the zero-count padding.** The same new test leaves ranks 2–4 and 6–8 empty,
   so a padded histogram is visibly wrong.
5. **An assertion that matched the defect.** "The row carries both rank fields" asserted both were
   `1` — exactly what a control hardcoding `1` produces. The fixture now gives visit rank 2 and
   policy rank 4.

## Still not done

The analysis has **not been run on the acquired record**. No D1″ value from real data exists, no
confirmation data has been opened, and no training is proposed.
