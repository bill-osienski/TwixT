# Combination policy pilot — generation without a completed throughput check (amendment, 2026-10-30)

An amendment to `docs/superpowers/2026-10-28-combination-policy-pilot-card.md` §5 steps 2–3 and §10 item 9
(admission). It is prospective: it governs the generation run that has not started. The card, the throughput-check
amendment (`2026-10-29-combination-policy-throughput-attempt2-amendment.md`) and both throughput runs' evidence are
unchanged; everything the card says that this amendment does not change still applies.

## Why

Both throughput attempts hit their ceilings:
- **Attempt 1** (ledger entry 41): TERM at 890 s; 131 of 144 slots started and 113 finished.
- **Attempt 2** (ledger entry 42): TERM at 1,790 s; all 144 started and 143 finished. Slot 108 was still in flight.
  Four 280-move state-cap games took 686–1,091 s each.

Both records are VOID, and neither is a throughput pass or a training result. Another 144-game check would not answer
the learning question and could meet the same long-game tail. The user's decision (2026-10-30) is to go straight to
preparing the 1,000-game training corpus.

## The exception

1. **No completed throughput check is required.** Generation (card §5 step 3) is admitted by:
   - the committed CPU timing record (COMPLETED, card §5 step 1);
   - this amendment, pinned by sha256. The run refuses a changed amendment, and its record carries the sha256.
   No throughput record or `outer.json` is read. Both throughput records stay VOID as written, and nothing relabels
   either one as a pass.
2. **The ceiling is six hours** for the one 1,000-game run:
   - TERM at 21,590 s, KILL 10 s later (21,600 s in all);
   - the cleanup observation is bounded through 21,610 s.
   Six hours is the card's existing maximum generation budget (§5 step 2: "never above 6 h"). It is not priced from a
   throughput measurement, and it does not guarantee that the corpus will finish.
3. **A generation timeout is assessed, not automatically abandoned.**
   - A run the ceiling stops is VOID, and its games are not used (card §5 step 3, unchanged).
   - It does not by itself close the learning question. The user then assesses the progress made and the cost of
     continuing, and any continuation is a separate decision.
   - There is no automatic retry, extension or top-up.

## Unchanged

- **The games:** 1,000 games of the proven agent playing itself:
  - calib020_0001 in eval mode with mirror averaging, 200 simulations;
  - root sequential halving from ply 20;
  - 24×24, the existing rules, TF32=0.
- **The run's settings:**
  - 18 workers and the arbiter's cap of 14 rows per call;
  - slot g plays seed base + g;
  - rows go to one create-only file per game outside the repository;
  - the held-out split by slot (≡ 9 mod 10).
- **Seeds (card §5):** a fresh registered 1,000-slot block. Survey, pre-proof, registration and post-proof come before
  the run; K1 → K2 closing comes after, and the block is retired whole. This amendment does not choose the seed method.
- **The launch and run checks:**
  - the idle gate and the background-compute check before launch, with BOINC kept quit;
  - the check after every game, and the TF32 and eval-mode checks;
  - the process-group supervisor with its bounded cleanup observation, and gate restoration;
  - the existing generation gate, `COMBINATION_POLICY_GENERATE_AUTHORIZED`, created closed.
- **Never used for training:** both throughput runs' games, the screen, strength and void-run games, and every other
  source the card excludes (§4).
- **What follows generation:** the model, search, learning target, training recipe, training ceiling (from the CPU
  timing) and strength-evaluation criteria (card §4, §5 steps 4–6).

## Paths

- Plan: `combination-policy-generate-run/plan.json`.
- Wrapper outputs: `combination-policy-generate-run/`. The wrapper is attempt 2's throughput wrapper with only the
  paths, the gate, `run generate` and the ceiling changed.
- Run evidence: `combination-policy-generate/`.
- Rows: `~/twixt-cache/combination-policy-pilot/generate/`.
- Run name: `run generate`.

## Not authorized by this amendment

Opening the gate, checkpoint loading, GPU work, games, closing, training and pushing. Each needs its own approval.
