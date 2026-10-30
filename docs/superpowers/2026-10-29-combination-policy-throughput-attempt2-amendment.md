# Combination policy pilot — the throughput check, attempt 2 (amendment, 2026-10-29)

An amendment to `docs/superpowers/2026-10-28-combination-policy-pilot-card.md` §5 step 2. The card and attempt 1's
evidence are unchanged; everything the card says that this amendment does not change still applies.

## Why

Attempt 1 (ledger entry 41, evidence `dba747bf`) hit its 15-minute ceiling (TERM at 890 s): outer TIMEOUT, record
VOID, 131 of 144 slots started and 113 finished. At that pace 144 games need roughly 19 minutes, so the ceiling, not
the run, was too tight. By the card's rule the pilot closed at step 2, and its block [202696900, 202697044) is EXPOSED
and RETIRED (`e30e277c`). Attempt 1 gives no result on policy learning.

## The exception

The user approves **one** fresh attempt at the throughput check, a narrow exception to the pilot's closure. There is no
further attempt, whatever happens to this one. A started run gets no retry, extension or top-up. An outer TIMEOUT
overrides an inner COMPLETED record and closes the pilot again.

## Unchanged

- The 144 games: the proven agent playing itself (calib020_0001 in eval mode, mirror averaging, root sequential halving
  from ply 20), 24×24, the existing rules, TF32=0.
- 18 workers and the arbiter's inference cap of 14 rows per call.
- All 144 games must finish inside the ceiling, or the run is VOID. Its games are never used.
- What it measures and prices: games/h, the arbiter's mean rows per call, and step 3's ceiling at 1.5× the projection,
  never above 6 h. Step 2's STOP rules.
- The idle gate and the background-compute check before launch, BOINC kept quit, the check after every game, the TF32
  and eval-mode checks, the process-group supervisor with its bounded cleanup observation, and gate restoration.
- The learning target, the training recipe and the evaluation criteria. The CPU timing (entry 40) is not repeated. No
  profiling or parameter sweep.

## Changed

1. **The ceiling:** TERM at 1,790 s, KILL at 1,800 s (30 minutes), cleanup observation bounded through 1,810 s.
2. **Fresh seeds:** a new 144-seed block, from copies of attempt 1's survey and closing procedures under
   `combination-policy-throughput2-seed-{survey,closing}`. The block is chosen by the survey's rule when it runs.
3. **Separate paths:**
   - Plan: `combination-policy-throughput2-run/plan.json`.
   - Wrapper outputs: `combination-policy-throughput2-run/`.
   - Run evidence: `combination-policy-throughput2/`.
   - Run name: `run throughput2`.
4. **Gate 54:** `COMBINATION_POLICY_THROUGHPUT2_AUTHORIZED` in its own module, created closed.
5. **Generation's admission reads attempt 2 only.** Its committed record must be COMPLETED under the card and carry this
   amendment's sha256. Its committed `outer.json` must show success: inner and outer exits 0, outcome COMPLETED, the
   gate restored, survivors empty, the script's own record, the same head. Attempt 1's record and paths never admit
   generation.

## Attempt 1 is preserved

Its VOID record, `outer.json`, games, progress log, closing records and registry entries stay unchanged. Its gate stays
closed, and its paths refuse a second run.

## Not authorized by this amendment

Seed procedures, registration, checkpoint loading, GPU work, gameplay, opening the gate, closing, generation, training
and pushing. Each needs its own approval.
