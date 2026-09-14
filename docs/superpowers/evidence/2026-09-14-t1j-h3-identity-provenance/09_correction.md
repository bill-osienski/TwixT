# CORRECTION — three false claims in this phase's commit messages

Written 2026-09-14, at the end of the provenance repair, against the runs
packaged here. The working agreement requires that a claim in a commit message
which turns out false be said so and corrected; these are corrected here rather
than by rewriting history.

## 1. THE CONTROL TOTALS WERE NEVER MEASURED

`20f2c0e` says "five controls added (612)". `0bd587a` says "controls 612 -> 616".
Both totals are wrong. The measured figures, read from each harness run's own
`distinct injections:` line:

| commit | run | distinct injections |
|---|---|---|
| `965c9f5` (before this phase) | `04_…` of the registration package | **602** |
| `20f2c0e` (identity barrier) | `02_harness_at_the_barrier.txt` | **607** |
| `0bd587a` (object threaded) | `04_harness_threaded_EXIT4.txt` | **611** |
| `3a52f0e` (stale controls repaired) | `06_harness_final.txt` | **610** |

The DELTAS I claimed were right (+5, +4, −1); the TOTALS were not. A single
unmeasured figure — 612, which I obtained by adding five to a number I had
already added five to — propagated through two commit messages and three
reports.

🔑 THIS IS THE EXACT RULE THE WORKING AGREEMENT STATES: *compute counts at the
moment of recording, not merely at generation.* The harness prints the count it
actually injected, on every run. I had it in front of me and wrote arithmetic
instead.

## 2. THE TEST TOTALS WERE COMPUTED THE SAME WAY

`20f2c0e` says "Tests 173 -> 182" and `0bd587a` says "182 -> 186". Only 173 was
ever measured (the three H3 files together, at `965c9f5`). 182 and 186 were
arithmetic; 86 was measured but for `tests/test_h3_pilot_runner.py` ALONE.

Measured at `3a52f0e`, the three H3 files together collect **191** tests. The
honest statement for the phase is **173 → 191**.

## 3. A COMMIT HASH THAT DOES NOT EXIST

`0bd587a` says the preserved harness run finished "before this change (commit
82b6e5f)". **There is no commit 82b6e5f in this repository** (`git cat-file -e`
refuses it). The run it refers to was made at **`20f2c0e`**, committed
2026-09-14 17:19:38 −0400, its output written 17:42.

## What is NOT affected

Every RESULT reported in this package was read from a run's own output, and
none of them changes: the exit-4 run and its three stale controls, the 610/610
run, the suite figures, the 53/53 pre-run verification, and the block's
accounting. The corrections above are to counts and a hash that were narrated
alongside those results, not to the results.
