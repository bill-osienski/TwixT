# Diagnostic O (opening attribution), revised: the frozen plan

**Status:** frozen 2026-11-12, before either model was loaded. It implements the user's revised approval of
2026-11-12, which revises the proposal `docs/superpowers/2026-11-11-htb-next-experiment-proposal.md` §3. The approval
covers **one run** on saved positions, within a 3.2 h ceiling. It plays no game, makes no labels and trains nothing.
Any refusal, stop, timeout or invalid evidence returns for assessment, with no automatic retry. The twixtbot benchmark
and any training need their own proposals.

## 1. Question

The 110 G3 match games with the student (new `hard`, model `78c16a31c3dba37f`) playing Black all began with legacy
`hard`'s (18,19), and the student answered with 72 different replies. So this is a study of the student's responses to
that one opening.

The rules-only replay's race heuristic is a clue about where to look, not a finding. It is where the heuristic first
showed persistent deficits, at plies 3–13 in the 12 losses.

This diagnostic measures three separate things on the student's opening moves:
- the readout;
- the search budget;
- the network's evaluation, the student's against the teacher's.

**It does not assign causes to the losses.** The reasons:
- A low-visit move shows only what the opening sampler chose.
- A different move at 800 simulations shows only sensitivity to the search budget.
- Neither shows that the alternative would have won.

## 2. Positions and inputs (frozen)

- **Positions:** for each of the 110 `docs/superpowers/evidence/htb-g3-match/games/pair_*_game_1.json`, the positions
  before plies 1, 3, …, 19, with Black to move. That is **1,100 positions**, all used regardless of outcome.
- **Game groups:** `docs/superpowers/evidence/htb-diag-o-plan/race_groups.json` (sha256 `947d6d98…`). It was written by
  `race_groups.mjs` from the committed games before any model load.
  - Outcomes: W 96, L 12, D 2.
  - Race group after ply 19, "2 or more pegs behind": W 31, L 12, D 2.

## 3. What is computed per position

All values are from **Black's** side. Raw network values and search values are reported separately.

- **Student stack.** The product's own: `AlphaZeroInference` on the hard model, wrapped in `MirrorAveragingInference`,
  as `server/index.js` and the G3 driver build it. The model id is checked against the hard manifest.

1. **Readout.**
   - The 200-simulation search is replicated from `protocolMove`'s plies-0–19 branch: `RootSequentialHalvingMCTS`,
     cPuct 1.5, `PyRandom(recorded search seed)`. It is followed by `sampleByTemperature` at T 1.0 with
     `PyRandom(recorded readout seed)`.
   - **The sampled move must equal the recorded move.** Any mismatch stops the run before any analysis (exit 3, VOID).
   - Recorded: the 200-simulation **visit leader**, the played move's visit **rank** (1 + the number of moves with more
     visits) and its visit **share**.
   - The visit leader is the move with the most visits. Ties go to the higher raw root prior, then to the lower
     (row, col).
2. **Search budget.**
   - The 800-simulation search uses the same class, network, position and recorded search seed, and so the same tie
     rule. It is compared with the 200-simulation search through their visit leaders.
   - Recorded for each search: the root value, and the search's own mean for the played move and for the leader.
3. **Network evaluation.**
   - **Raw student:** the value at the position; the priors and prior ranks of the candidates; the value of the
     position after each candidate (Red to move there, negated).
   - **Raw teacher:** the pinned twixtbot raw network through the H-TB label path's own wrapper
     (`htb_pilot.TwixtbotTeacher`, `_load_teacher_real`). It is mirror-averaged as in `label_game`, with lockstep state
     equivalence checked on every replay.
   - Recorded for the teacher: the value at the position; its top move; each candidate's probability and rank; the
     value after each candidate.
   - The candidates are the played move, the 200-simulation leader and the 800-simulation leader.
   - **To assess a proposed alternative reply,** the played move and the alternative are compared on their resulting
     positions, by each evaluator.

## 4. Code (frozen)

| File | sha256 |
|---|---|
| `tests/htb_integration/diag_o.mjs` (driver, stages A and B) | `1b488eef2038c79af11ece26580ff4b57cac33b60980273825017c6a68e9752a` |
| `scripts/GPU/alphazero/htb_diag_o_teacher.py` (stage B) | `45664a2b7490a2ef3f1c01b929b1ada121b1ff619a0b75c711d0dbebdc9ebf80` |
| `docs/superpowers/evidence/htb-diag-o-run/01_wrapper.sh` | `dace56a1fcaa02b9ffa92ec0c7856a66d041ac95c35770911a21a6de75a528e6` |

The analysis, frozen: diag_o_analyse.mjs sha256 `3ff51fc822a81e658dc1f6079cd0804eccbc7dc3be1a44955ba144d0bc667eea`.
The wrapper refuses to launch if the analyser differs from it.

## 5. Analysis (frozen)

**The unit is the game.** Losses (L, 12) are compared with wins (W, 96). Draws (D, 2) are shown separately, value by
value, and never pooled. The same comparisons are repeated within the "2 or more behind after ply 19" subgroup (L 12, W
31, D 2).

Per-game metrics, over the game's 10 positions:

| Metric | Definition |
|---|---|
| `R_off` | positions where the played move is not the 200-simulation leader |
| `R_low` | positions where the played move's visit share is below 10% |
| `B_diff` | positions where the 800-simulation leader differs from the 200-simulation leader |
| Values at plies 19 and 9 | raw student, raw teacher, 200-simulation root and 800-simulation root |
| `T_top1` | positions where the played move is the teacher's top move |
| `T_prob_played`, `S_prob_played` | mean probability of the played move, teacher and student |
| Alternatives (`alt200`, `alt800`) | over positions where the leader differs from the played move: the mean of (value after the leader − value after the played move) for the raw student, the raw teacher and that search; also the counts where each evaluator prefers the alternative |

**Uncertainty:**
- A difference in means (L − W) gets a 95% percentile interval from a stratified bootstrap over games: 10,000
  resamples, seed 20261112.
- "At least one" rates get Wilson 95% intervals.
- The AUC of each value at ply 19 (a lower Black value marking a loss) gets a bootstrap interval. The teacher's AUC
  minus the student's AUC is paired on the same resamples.
- A table by ply is descriptive only.

## 6. Signals (frozen): they choose a follow-up and assign no cause

| Signal | Fires when | Follow-up it would justify |
|---|---|---|
| **Readout** | ≥ 8 of the 12 losses have `R_low` ≥ 1, **and** mean `R_low` in L ≥ 2× that in W, **and** the bootstrap interval for the difference is above 0 | A readout-setting change, such as a lower opening temperature, in a fresh match against current new `hard` |
| **Teacher** | Teacher AUC at ply 19 ≥ 0.75, **and** raw student AUC ≤ 0.60, **and** the interval for teacher − student is above 0 | Testing fresh training data from the student's own play, labelled by the teacher, fitted from the current student and tested against it on fresh games |
| **Budget** | ≥ 8 of the 12 losses have `B_diff` ≥ 1, **and** mean `B_diff` in L ≥ 2× that in W, **and** the interval is above 0 | A search-budget change in a fresh match against current new `hard` |

- **If more than one fires:** the order is readout, then teacher, then budget. The readout comes first because a
  sampled move changes every later position; all fired signals are reported.
- **If none fires:** the diagnostic singles out no follow-up, and the proposal's second step (a twixtbot benchmark card)
  stands as the default.
- **The recommendation is a proposal only.** Each follow-up needs its own approval.

## 7. Prediction on record

Unchanged from the proposal, stated before the run, with low confidence:
- The readout explains a minority of the losses (the readout signal does not fire).
- The student's ply-19 value separates losses less well than the teacher's.

## 8. Run procedure and ceiling

1. Commit this plan, the code and the tests, with the gate closed.
2. Create a clean linked worktree from that commit, with `node_modules` APFS-cloned.
3. Make a one-line gate-opening commit (`DIAG_O_AUTHORIZED`). The wrapper refuses any other HEAD.
4. The wrapper then:
   - checks the analyser's sha256 against §4;
   - refuses if Unreal Editor is running; it never stops it;
   - runs the idle gate (outside the ceiling) and the BOINC check;
   - runs the driver under the process-group supervisor: **TERM at 11,510 s, KILL at 11,520 s (3.2 h)**;
   - writes `outer.json`;
   - restores the gate on every exit.
5. The planning estimate is 1,100 × (1.35 s + about 5.4 s) of search plus a few minutes for the teacher: about 2.1 h.
6. The evidence goes to `docs/superpowers/evidence/htb-diag-o/` (`student.jsonl`, `teacher_in.json`, `teacher.json`,
   `analysis.json`, `record.json`, `progress.jsonl`) and `…/htb-diag-o-run/`. It is validated and committed with the
   gate restored, then recorded in the ledger. Nothing is pushed.

**Stops (exit 3, VOID):**
- an input mismatch;
- a reproduction mismatch;
- a teacher lockstep divergence or a non-finite output;
- a missing teacher candidate;
- an analysis error.

**A refusal** (exit 4) launches nothing.
