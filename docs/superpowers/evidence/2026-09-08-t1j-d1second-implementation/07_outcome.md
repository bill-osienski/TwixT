# D1″ implementation — CODE AND TESTS ONLY

**Nothing was executed and nothing was acquired.** No model, no JVM, no query, no game, no seed, no
registry edit, no gate change, no push. The acquired D1 record was **not read**: its 19-entry
manifest still verifies, all 19 files.

## 1. What exists now

| | |
|---|---|
| `scripts/GPU/alphazero/d1second_analysis.py` | the frozen analysis, one checked public entry |
| `tests/test_d1second_analysis.py` | **39 tests**, synthetic fixtures only |
| `d1prime_analysis.matched_statistic` / `stability_interval` | gained an `indicator` parameter |

**The indicator parameter is a deviation from the plan's wording and is reported as one.** §3 said
the machinery would be "reused unchanged"; reuse in fact required naming the field being compared,
because the statistic read `lprd` directly. The parameter **defaults to `lprd`**, a test asserts
D1′'s result is identical with and without it, and all 112 D1′ tests pass untouched. The alternative
— a second copy of the statistic, or D1″ rows carrying `ss` in a field named `lprd` — would have been
a drift risk or a lie in the data.

## 2. The metric, as frozen

`ss := rank_raw(t1j_move_6) ≤ 5 AND rank_visit(t1j_move_6) > 5`

- `rank_visit` takes **the visits and nothing else** — asserted on its signature, because a visit
  ranking that could see the policy would inherit the ordering it is compared against.
- Ties break on canonical `(row, col)`; zero-visit moves are **ranked, not dropped**.
- Every shared constant is **bound, not retyped**: `K_SS is DP.K_LPRD`, `T_THRESHOLD is
  DP.T_THRESHOLD` (0.15, unchanged), `FLOOR is DP.FLOOR`. Only the bootstrap seed differs — 20260908,
  so the replicate draw is not a deterministic repeat of D1′'s over the same games.

## 3. The disjointness precondition

`check_disjoint` is implemented, called by the public entry on every run, and tested three ways:
exhaustively over all 100 rank combinations a ten-move root produces, end to end over 221 synthetic
rows, and by a forged row that must be refused. **It has not been run on the acquired record** — that
is the separate authorization.

## 4. Verification

| | |
|---|---|
| D1″ tests | **39 passed** |
| injected-defect controls | **405 / 405 rejected**, 0 not caught, 0 stale |
| clean baseline | **PASS** over 326 target tests |
| every source restored | True |
| full suite | **4,193 passed, 4 skipped, 0 failed** (16m52s) |
| six gates | all False |

**23 controls are new**, one per frozen property: both `ss` boundaries, `ss` ignoring the visits,
`ss` overlapping `lprd`, the strict variant, tie direction, zero-visit dropping, the same-legal-set
check, the precondition, the entry calling it, the entry regaining knobs, the seed reused from D1′,
the threshold retyped, the floor unenforced, a secondary report deciding the outcome, the readout
summary tolerating an absent field, and the indicator being ignored, defaulted wrongly, or read as
False when missing.

## 5. Three controls were NOT CAUGHT on the first run, and none was waived

1. **A redundant guard.** `suppression_row` refused an empty root, but `rank_visit` refuses it one
   line later anyway, so removing the guard changed nothing observable. A branch no control can
   distinguish proves nothing: **the guard was deleted**, its control re-aimed at the refusal that
   now owns the case, and the test asserts that message.
2. **A test of mine that passed for the wrong reason.** "The entry refuses a report that is not a
   completed D1 run" used `{"positions": []}`, which fails the *later* cohort binding even with the
   contract check deleted. It now uses a complete 221-row report missing exactly one acquisition
   field, and matches on that field's name.
3. **A test of mine that did not discriminate.** `pytest.raises(Exception)` accepted the `KeyError`
   raised once the named refusal was removed. It now requires the named error and the field in the
   message.

**A fourth fault, caught by a test rather than a control:** my own hand-computed expectation was
wrong — in a six-move root a move with four visits ranks *second*, so it is not suppressed by rank at
all. The code was right; the test now builds a root big enough for the case it claims to test.

## 6. What is still not done

The analysis has **not been run on the acquired record**. No D1″ value has been computed from real
data, no confirmation data has been opened, and no training is proposed. Executing the real analysis
is the next separate authorization.
