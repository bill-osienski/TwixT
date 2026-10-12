# Search distillation — Stage A card (DRAFT for review, 2026-10-11)

**Status: draft, revision 2 — design accepted for preparation; scripts written, uncommitted, for review
of their diffs.** Stage A execution, its seed accounting and every commit each need explicit approval. No gate is added or changed. Nothing is pushed.
Capture infrastructure is not extended.

**Question.** Does the label-producing agent (calib020_0001 at 1,600 simulations) beat the same network at
400 simulations under the frozen protocol? This is Stage A of
`docs/superpowers/2026-10-11-search-distillation-proposal.md` (§5), now with a cap from the A1 measurement
(`docs/superpowers/2026-10-11-search-distillation-a1-card.md`, run 09741b3e, closed 6dc4013c).

## 1. What runs

Everything not listed here is A1's, unchanged: the model by absolute path with SHA-1 verified, the main
checkout's environment, a clean linked worktree, `external_learning_stage3.PROTOCOL` imported unchanged,
explicit budgets, `run_game_tasks([task], 1, ...)` once per game in one process, replay sidecars, the
create-only records, the module check after loading.

| Item | Value |
|---|---|
| Agents | `sims1600` (agent A) and `sims400` (agent B): the same checkpoint, budgets 1,600 and 400, both with the opening-temperature readout (20 plies at 1.0, then 0.1). |
| Games | **160 = 80 pairs**, serial. `build_agent_pairing_tasks("stage-a", sims1600, sims400, 160, base_seed=BASE)`: game g has seed BASE + g; pair k = games 2k and 2k + 1; `sims1600` is red in game 2k and black in game 2k + 1. |
| Seeds | A fresh 160-seed block (§4). Every game has its own seed, so pairs are independently seeded; no opening is shared or imposed. |
| Order | Games run in index order. No game is reordered, skipped, replaced or repeated. |
| Cap | **24 hours in total, including loading and cleanup**: `timeout -k 10 86390`, TERM to the whole process group at 86,390 s and KILL 10 s later. No extension. |

## 2. The statistic and the verdict (fixed before any game)

* **Game score** for `sims1600`: win 1, draw 0.5 (no winner, including the 280-ply `state_cap`), loss 0.
* **Pair score** = the mean of the pair's two game scores. **N = 80** pairs.
* **Pass rule (exact):** `mean − sqrt(ln(40) / (2 · 80)) > 0.5`, computed in code by that formula, never
  against a rounded threshold. The margin is about 0.15184. It is the two-sided 95% Hoeffding lower bound
  on pair means in [0, 1].
* **Scoring is by agent identity and colour binding,** never by checkpoint identity (both agents share one
  checkpoint): `winner_agent_id` sims1600 → 1, sims400 → 0, none → 0.5, and the winning colour must be the
  colour the schedule bound to that agent.
* **Attainable boundary:** with 160 games the total points are multiples of 0.5. **104 points gives a mean
  of 0.65 and NON-PASS; 104.5 points gives 0.653125 and PASS.**
* **Power:** a true effect of +25.2 points is a conservative sufficient effect for 80% power; it is not
  necessary.
* **A verdict exists only for a COMPLETED run**: outer exit 0, no survivors, run verdict COMPLETED, every
  per-ply and per-game check passed, `validate_result_set` over the 160 results recorded as passed, and the
  exact schedule: slot g = game g, pair g // 2, sims1600 red on even g, budgets 1,600/400 bound to the
  agents, and every root visit count equal to that colour's budget.
  A run that is stopped, refused, killed or capped, for any reason, gives **no verdict**; its partial
  results are not scored, not extrapolated, and not completed later.
* **Outcomes:** PASS, or NON-PASS. A non-pass is an operational stop for this line. It does not show that
  1,600 is no stronger, nor that distillation is futile. A pass licenses only Stage B planning; it is not
  evidence that distillation will work.
* **Independence is an assumption, not a result.** Distinct seeds and the collision checks do not prove
  statistical independence. The assumption, precisely: both agents are frozen (one checkpoint, fixed
  budgets and readout, no learning or state carried between games); every game starts from a fresh board
  and fresh search and readout generators (`play_eval_game` creates four `random.Random(seed ^ mask)`
  streams per game, one search and one readout stream per colour, from the game's own seed, with the four
  audited masks); and these separate, audited PRNG streams are treated as independent across games. The
  shared evaluator carries no random state between games. **The interval is conditional on this
  assumption.**
* **Distinct transcripts** are counted from a digest of each game's ordered move history (§5), from the
  record alone. The count is descriptive; it is not a gate.
* The verdict comes from a separate analysis script, committed before the run (§6), which reads only the
  committed plan, `outer.json` and `record.json`.

## 3. Runtime and the cap

* **Measured (A1):** 339.5 s and 230.2 s per game, mean 284.81 s. These are amortized whole-game costs
  (about 6.1 and 4.8 ms per simulation). Lazy compilation may explain part of game 0's higher cost; its
  contribution was not isolated.
* **Projection:** 160 × 284.81 s ≈ 45,570 s ≈ **12.7 hours** before headroom.
* **Proposed cap: 24 hours** (about 1.9× the projection).
* **Two timing games cannot guarantee completion.** Game length drives cost. A1's games had 55 and 48
  plies; at about 4.8–6.1 ms per simulation a 280-ply game would take roughly 22–28 minutes. A modest shift
  toward longer games would exhaust the cap. If the cap fires, the result is **no verdict** and the line
  stops for review; there is no extension and no rerun under this card.
* **Operational (accepted):** the wrapper runs `caffeinate -i` inside the timeout, so the machine does not
  idle-sleep during the run. The run is launched detached with `nohup`, by the one frozen command below,
  so that ending the chat session cannot end the run. Progress is watched only by reading
  `progress.jsonl`; no other monitoring is built.

  The frozen launch, run once from the worktree root (it is also in the wrapper's header):

  ```sh
  sh -c 'set -euC; D=docs/superpowers/evidence/search-distillation-stage-a;
         exec 3>"$D/launch.out.txt";
         exec 4>"$D/launch.pid.txt";
         nohup sh "$D/02_stage_a_wrapper.sh" >&3 2>&1 < /dev/null 3>&- 4>&- &
         printf "%s\n" "$!" >&4'
  ```

  Both files are **reserved before the job starts**: fds 3 and 4 are opened create-only (`set -C`, and
  `set -e` stops at the first refusal), so a second launch fails before anything runs. `launch.out.txt`
  then receives the wrapper's own output; `launch.pid.txt` receives the wrapper's PID through the
  already-open fd 4. In the job, stdout and stderr go to fd 3's file, and fds 3 and 4 themselves are closed.

## 4. Seeds — the established method

The A1 method and sequence, unchanged in substance: survey → pre-registration proof → registration →
post-registration proof → run → usage derivation → closing edit. Each step is separately approved.

* **Rule:** `seed_selection.candidate(reserved, 160)`. Because the rule is monotone, the block lies above
  A1's retired block. The survey computes it; no seed is chosen or written here.
* **Controls:** at 160 seeds the pinned planted-literal controls (offsets up to +9) fall inside the real
  block, so **no probe block is used**. Everything else is A1's: the negative, exemption, enumeration and
  state controls, and the two-tree scope (the worktree plus the main checkout).
* **Plan:** `{"kind", "base", "n_slots": 160, "survey_record_sha256"}`.
* **Closing (fixed now):** RETIRE all 160 seeds whatever happens; EXPOSE only the started-game prefix
  (`game_start` events, logged before each game). A stop before game 0 exposes none.

## 5. Recorded, and the exits

As A1, in `docs/superpowers/evidence/search-distillation-stage-a/`, create-only: `launch.out.txt`,
`launch.pid.txt`, `progress.jsonl`, `record.json`, `replays/` (force-added by exact path at commit;
`.gitignore`'s `Replays/` matches case-insensitively), `outer.json` (outer status, wall seconds,
survivors), `run.stdout.txt`, `run.stderr.txt`; after the run, the analysis record
`03_stage_a_analysis_run.txt`.

* **`progress.jsonl` is outcome-blind:** `start`, `loaded`, and per game `game_start` (game, slot, red
  agent) and `game_end` (seconds, plies, recorded budgets). It holds no winner, score or termination
  reason, so the run can be watched for progress and time without seeing results.
* **`record.json`**, written once at the end: as A1 (environment, verified SHA-1, effective protocol,
  modules), plus the 160 results with per-colour `root_total_visits` sets and `transcript_sha256`, the
  sha256 of the game's ordered move history `[[ply, player, row, col], ...]` taken from the replay the
  script already reads, and the result-set check. `strength_verdict` is null there; the verdict is the
  analysis script's.

| Status | Meaning |
|---|---|
| 0 `COMPLETED` | 160 games and every check passed. The analysis may run. |
| 1 | One-shot refusal: an output already exists. Nothing is loaded. |
| 2 `STOP_INTEGRITY` | An `IntegrityError` (per ply, per game, or the result set). No verdict. |
| 3 `STOP_ERROR` | Any other exception. No verdict. |
| 4 `REFUSED` | Before game 0 (checkpoint, plan, seeds, module check). No verdict; no seed exposed. |
| 124 / 137 | The cap fired (TERM / KILL). No verdict. |

## 6. The scripts (written, uncommitted, for review)

Four are copies of A1's committed counterparts, each with a header stating its diff; the review is that
diff. The analysis is new.

* **Survey and proofs**: `evidence/search-distillation-stage-a-seed-survey/01_seed_survey.py`. Diff from
  A1's: `N_SLOTS` 160; the paths; slot lines for 160 games (with the pair); no probe (the planted-literal
  controls run on the real block); no failed-attempt exemption; the plan's kind; the `n_slots + 1` state
  control; the titles.
* **Run**: `evidence/search-distillation-stage-a/01_stage_a_run.py`. Diff: 160 games and slots; the
  paths; outcome-blind `game_end` (no reason); `transcript_sha256`; `validate_result_set` over the whole
  schedule before `COMPLETED`.
* **Wrapper**: `evidence/search-distillation-stage-a/02_stage_a_wrapper.sh`. Diff:
  `timeout -k 10 86390 caffeinate -i ...`; the paths; the header with the frozen launch.
* **Analysis** (new): `evidence/search-distillation-stage-a/03_stage_a_analysis.py`. It refuses (exit 4)
  unless its three inputs are committed and unmodified; gives NO VERDICT (exit 1) on any schedule fault,
  a dirty tree or a failed control; otherwise PASS or NON-PASS (exit 0), with the 80 pair scores, the
  mean, the exact bound and the distinct-transcript count. Its 12 synthetic controls: all wins PASS; all
  draws NON-PASS; 104 points NON-PASS; 104.5 points PASS; those two straddle the exact bound; and NO
  VERDICT for 159 games, a winner agent that is not the winning colour's, budgets not bound to the
  agents, a root visit count off the budget, outer exit 124, survivors, and a STOP_INTEGRITY run.
* **Closing**: `evidence/search-distillation-stage-a-seed-closing/01_derive_usage.py`. Diff: `N_SLOTS`
  160; the paths; 15 synthetic controls generalized to 160 slots (completed with 160 starts; refusal and
  loading timeout expose none; interrupted prefixes of 1 and 74; a kill or integrity stop in the last game
  exposes all 160; and refusals for survivors, a gap, a repeat, 161 starts, exit 0 with 159 starts, a
  refused run that started a game, an unaccounted block and a block already partly retired).

No tests beyond each script's in-record controls and `tests/test_seed_selection.py` (once after
registration, once after closing) are proposed.

## 7. Sequence (as A1 §5)

P0 commit the scripts on `main` → P1 clean worktree → S1 survey → S2 pre-proof → S3 registration + plan +
`test_seed_selection.py` → S4 post-proof → **R** (detached, up to 24 h) with termination confirmed → commit
the outputs → **V** the analysis, once → K1 usage derivation → K2 closing edit + `test_seed_selection.py` →
F fast-forward `main` and remove the worktree. Any non-zero exit, failed control, red test or timeout
stops the sequence for review. The analysis (V) runs only after R's outputs are committed, and only on a
COMPLETED run.

## 8. What Stage A does not establish

Teacher quality for learning, learning benefit, and strength after distillation are all outside Stage A.
A pass says only that the 1,600-simulation agent beats the 400-simulation agent under this protocol, at
this bound. Stages B and C remain provisional and need their own cards.
