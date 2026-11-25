# Benchmark card: did the new Hard close some of the gap to the twixtbot teacher?

**Status:** DRAFT for review, 2026-11-24. **Nothing is built, loaded, generated or played under this card.** All
gates are closed. It needs three approvals before any game: the narrow #52/#195 exception in §2, this card, and,
after a model-free build, one launch review. The smoke and the match are then each run once, each with its own
approval.

## 1. Question, and what this is not

**Question:** under our rules, does the new Hard (`627c8d774df838a8`, deployed locally on 2026-11-24, entry 82)
score higher against the pinned twixtbot teacher than the previous Hard (`78c16a31c3dba37f`)?

- **Primary result:** the **paired score difference** D = new − previous, over the same fresh openings.
- **Reported alongside:** each version's own score against the teacher (draws count ½), and entry 65's 11.8% as
  historical context only.
- **Why this matters.** The H-MX update was accepted on a head-to-head match against the previous Hard (entry 78:
  mean pair score 0.519, decisive pairs 31–15). That match says nothing about distance to the teacher. Entry 65 put
  the previous Hard, as shipped, at 0.118 against this teacher. This card asks whether the update moved Hard toward
  it.

**What this is not:**
- **Not an anchor.** It calibrates nothing and produces no Elo, no absolute strength and no gap size or bound.
- **Not a deployment input.** The deployment stands on entries 78–82 whatever the result.
- **Not a rerun, extension or re-score of entry 65.** The previous Hard is re-measured on fresh openings, inside the
  paired design. Entry 65's numbers are never pooled with these, and its games and openings are not reused.
- **Not twixtbot's strength under its own rules.** The teacher was trained with own-link crossings allowed (entry
  65's card §3); it is measured here as it plays under ours.
- **Not a study of why.** It does not separate the network from search or opening sampling: both versions use the
  same shipped protocol, so the difference is attributable to the models as shipped.

## 2. Ledger items #52 and #195, and the narrow exception this card needs

**What they record.**
- **#52** closed the August 2026 twixtbot anchor pilot. It requires that "a future external anchor must be a
  *different* engine or a rules-matched one, and needs its own scope". Entry 65 ran under a narrow exception to it.
- **#195** closed entry 65's benchmark. Among other things it forbids rerunning, extending, topping up or re-scoring
  it, changing its configuration after the fact, and using twixtbot "as an anchor or play it in any other
  configuration (the #52 exception covered exactly these 4 + 620 games)".

**Why this card is not entry 65 again.**
- **Different question:** a paired comparison of two Hard versions, one of which did not exist on 2026-11-14. Entry
  65 asked whether the student trails its teacher.
- **Fresh openings, a fresh seed namespace and its own game counts** (§5). None of entry 65's games, openings or
  outcomes is reused, pooled or re-scored.
- **The same teacher configuration, unchanged** (§3). It is reused, not varied: nothing here moves toward the
  `trials` ladder or a weaker-anchor hunt that #52 forbids.

**Why it still needs an exception.** It plays new games against twixtbot's network, which #195 limited to entry 65's
4 + 620 games, and which #52 read broadly covers.

**The exception requested, stated narrowly.** Approve twixtbot's pinned network (§3) as an opponent **only**:
- at entry 65's one teacher configuration, fixed here before any game;
- against exactly two subjects, `627c8d774df838a8` and `78c16a31c3dba37f`, each as shipped (§4);
- on this card's openings and seeds (§5);
- for one run of exactly **4 smoke games** (unscored, timing only) and **1,200 scored games** (300 openings × 2
  versions × 2 colours), and no others.

**The exception does not cover:**
- any anchor use: Elo, calibration, or ranking any other checkpoint;
- twixtbot's own player, any `trials` setting or any other teacher configuration;
- any third subject, any other student budget, or changing the openings, counts, pass rule or cap after any result;
- rerunning, extending or topping up this run, or entry 65's;
- any rules change, handicap or ply cap;
- using these games or openings as training data, or as the evaluation set of a fit.

## 3. The teacher (unchanged from entry 65)

| Item | Identity |
|---|---|
| **Engine** | `github.com/stevens68/twixtbot-ui` @ `83749f230a0bae1766b46a05bfde0ed87f0a9a0a`, weights `variables.data` `1958f847…`, `saved_model.pb` `e0c2b882…`, `variables.index` `1d4073c4…`, checked at load |
| **Inference** | the H-TB label path's wrapper (`htb_pilot.TwixtbotTeacher` via `_load_teacher_real`): one raw network call per position, rotation off, softmax over our legal moves, averaged over the position and its left-right reflection (column c → 23 − c) |
| **Move rule** | argmax over our legal moves, ties to the lower (row, col). **No search.** Deterministic |
| **Runtime** | teacher venv: Python 3.13.15, TensorFlow 2.21.0, NumPy 2.5.2 (CPU), clone at `$HOME/twixt-cache/htb-teacher/twixtbot-ui`; versions checked before the model loads (exit 4 on a mismatch) |
| **Lockstep** | entry 65's: twixtbot's `Game` and its reflection kept in lockstep with our `TwixtState`, checked after every ply (`state_divergences` with the `terminal` field left out, §7 of entry 65's card); the full canonical state compared between Node and Python by sha256 after the opening and every ply |

## 4. The two Hard versions (each exactly as shipped)

| | New Hard | Previous Hard |
|---|---|---|
| **Model** | `627c8d774df838a8` (graph `3beaeed1…`, data `cf78b17f…`, manifest `abf3931d…`) | `78c16a31c3dba37f` (ONNX `bc68a274…`, from checkpoint `0e1cb621…`) |
| **Loaded from** | `models/627c8d774df838a8/manifest.json` | `models/78c16a31c3dba37f/manifest.json` |
| **Status** | served by the local product since entry 82 | served before entry 82; kept on disk as the tested rollback |

- **Both** through the product's code: `resolveHardModel` with the explicit manifest, `MirrorAveragingInference`
  over `AlphaZeroInference`, and `protocolMove` at **200 simulations**: cPuct 1.5, T 1.0 sampled readout through ply
  19, root sequential halving from ply 20. This is the shipped protocol for both; only the model differs.
- **Checks before any game:** each model id against its manifest; both artifact files by the product resolver;
  onnxruntime-node **1.23.2** resolved from the launch worktree's own `node_modules` (APFS-cloned before the gate
  commit); the non-finite guard on every evaluation (entry 65's Repair 2).
- **Per move:** exactly one evidence row per played ply; on every Hard move `nSimulations` = 200 and root `visits` =
  200, as entry 65 enforced.

## 5. Openings, colours, seeds and counts

- **Rules:** the empty 24×24 board, Red first, no swap, our rules (`allow_scl=False`, automatic links, no link
  removal). The product's game-end rule alone: play until `TwixtState.isTerminal()`, no ply cap; a win, or a draw
  when no legal moves remain. twixtbot's early-draw flag is recorded only and never ends a game.
- **Openings:** supplied **4-ply** openings (plies 0–3, Red–Black–Red–Black), fresh.
  - **Generation:** opening *i*, ply *k*, takes index
    `int(sha256("hard-teacher-bench/20261124/opening <i>/ply <k>")[:12], 16) mod n`, where *n* is the number of
    our legal moves in rows and columns 4–19 at that ply, in sorted (row, col) order. This is entry 65's rule in a
    new namespace.
  - **Equivalence is by position, not by move list.** Two openings are the same if their positions after ply 3
    (pegs and bridges by colour, side to move) are equal under any of the four symmetries that preserve both
    players' goals: the identity, the left-right reflection (c → 23 − c, the one mirror averaging uses), the
    top-bottom reflection (r → 23 − r) and the 180° rotation. Comparing positions also catches the same pegs played
    in another order.
  - **Within this card:** an opening equivalent to an earlier one is skipped, and the next *i* is used.
  - **Against earlier work:** an opening is skipped if it is equivalent to any opening position in the exclusion
    set of §5.1.
  - The list (300 match openings and 1 smoke opening, in a separate `smoke opening <i>` namespace) is generated and
    committed in the model-free build, with the number of skips of each kind. The analyser re-derives it.
- **Per opening, four games:** the new Hard as Red and as Black, and the previous Hard as Red and as Black, each
  against the teacher.
- **Seeds (common random numbers):** only the Hard side draws randomness. Seed material is
  `sha256("hard-teacher-bench/20261124/<stage>/opening <o>/hard <colour>/ply <ply>")`, hex digits 0–11 the search
  seed and 12–23 the readout seed, as in G3 and entry 65. **The seed does not depend on the version**, so at a given
  opening, colour and ply both versions draw the same numbers. This correlates the two versions' games where their
  play agrees and reduces the variance of D; it biases neither. The build checks that every reachable seed is below
  2^48 and that seeds are distinct across (stage, opening, colour, ply).
- **Game order:** opening by opening, the four games in a fixed order. Outcomes are fixed by the seeds and models
  (the gated H-MX check reproduced 845/845 recorded moves), so order cannot change a result.
- **Counts, fixed:** **300 openings, 1,200 games**; smoke 1 opening, 4 games.

### 5.1 The exclusion set

From a read-only inventory of every supplied-opening set in the repository (2026-11-24). Every set stores openings in
our 0-based (row, col) convention, including the T1j and twixtbot runs. A fresh opening is skipped if its position
after ply 3 is equivalent (§5) to any position below. Play that starts from the empty board (H4, H5, the
mirror-averaging and root-halving matches, checkpoint tournaments, self-play) supplies no opening and adds nothing.

**Required: openings used in evaluation (4-ply).**

| Set | Ledger | Count | Source |
|---|---|---|---|
| Student–teacher benchmark, match and smoke | entry 65 | 200 + 1 | re-derived from `teacher_bench.mjs` `openingList` (namespace `htb-teacher-bench/20261114`), checked against every played game's `opening` field and the recorded list sha `f4d3c11c…` |
| H-OP match and smoke | entries 71–72 | 272 + 2 | `htb-onpolicy-openings/openings.json`, keys `match`, `match_smoke` |
| H-MX match and smoke | entries 77–78 | 400 + 2 | `htb-mix-openings/openings.json`, keys `match`, `match_smoke` |
| Product-stack comparison pool (Aug 2026) | product-model alignment | 210 | `tests/product_match/openings.json` (ids 0–99 played, 200–209 timing, 100–199 never played; all 210 excluded) |
| twixtbot anchor pilot | #52 | 8 | `twixtbot_g3_schedule.py` `OPENINGS` |
| H-TB G3 match and smoke, first 4 plies | entries 62–63 | 222 | `htb-g3-{match,smoke}/games/*.json` `moves[:4]` (played from the empty board; excluded as H-OP and H-MX did) |

**Required: evaluation openings of 6 plies, excluded as prefixes.** A 4-ply opening cannot equal a 6-ply position,
but it can be the start of one. An opening is skipped if its position is equivalent to the position after ply 3 of:

| Set | Count | Source |
|---|---|---|
| T1j E4 screen, L0, H1 and H2 matches | 8 | `2026-08-25-t1j-e4-preflight-attempt4/06_endpoint_screen_plan.json` |
| T1j H3 full study | 296 | `2026-09-17-t1j-h3-study-openings-uniform/01_opening_set.json` (set digest `35932b3f…`) |
| T1j H3 pilot | 20 | **not committed as a list.** The build re-derives it with `h3_pilot_rules.generate_openings(seed=20260913, n=20)` and must reproduce the recorded set digest `4027efe3…`; if it does not, the build stops and returns for review |

**Added by me, for review: openings whose games became training data.** The new Hard was fitted on positions from
H-OP's self-play games, so an opening from that generation could favour it.

| Set | Ledger | Count | Source |
|---|---|---|---|
| H-OP generation, smoke and extension | entries 66–67 | 4,000 + 40 + 2,500 | `htb-onpolicy-openings/openings.json` (`gen`, `gen_smoke`), `gen-extension.json`; only 5,197 were played, but all are excluded, as H-MX did |
| External-engine learning, Stage 1 | #53 | 6 | `external-engine-learning-stage1/record.json` `steps.openings` |

- **In all:** 1,317 evaluation openings of 4 plies, 324 of 6 plies as prefixes, and 6,546 training openings,
  before symmetry. The build reports how many fresh candidates each set removed.
- **Stronger than earlier checks.** H-OP and H-MX compared move lists directly and under the left-right reflection
  only. This card compares positions under all four goal-preserving symmetries, so it also catches the top-bottom
  reflection, the 180° rotation and transposed move orders.
- **Expected effect:** the fresh openings are drawn from about 4 × 10^9 central 4-ply sequences, so I expect few or
  no skips; the count is recorded either way.

## 6. Primary result, uncertainty and the pass rule (fixed before any game)

- **Scores.** A game scores 1 for a Hard win, ½ for a draw and 0 for a loss. For version *v* at opening *o*,
  S_v(o) is the mean of its two games (as Red and as Black). The opening's paired difference is
  D(o) = S_new(o) − S_prev(o), in {−1, −¾, …, ¾, 1}.
- **Primary result:** mean D over the 300 openings.
- **Pass rule (CLOSER):** the **exact one-sided paired sign-flip test** on Σ D(o) gives p ≤ 0.05 for D > 0, and mean
  D > 0.
  - The test conditions on the observed |D(o)| and treats each sign as a fair coin under the null that the two
    versions are exchangeable at every opening. Its null distribution is computed exactly by convolution over the
    integer values 4·|D(o)| (no Monte Carlo).
  - Otherwise the result is **NOT SHOWN**, which is not evidence of no change.
  - **Descriptive flag, not a rule:** if the lower-tail p for D < 0 is ≤ 0.05, the result is reported as LOWER.
- **Uncertainty on D:**
  - **95% Hoeffding interval** (two-sided, distribution-free, range 2): mean D ± √(2 ln 40 / 300) = ± 0.157. It is
    valid but wide.
  - **95% paired bootstrap percentile interval** over openings (100,000 resamples, seed fixed in the build):
    approximate and much narrower. Labelled as approximate.
- **Each version's score:** mean S_v with a two-sided 95% Hoeffding interval over openings, √(ln 40 / 600) = ±
  0.078, the half-width rule entry 65 used. Also by colour, draws separately, game lengths and twixtbot's early-draw
  flag, all descriptive.
- **Historical context:** entry 65's previous Hard as shipped, 0.118 (110 pairs, [−0.011, 0.248]), is printed next to
  the previous Hard's new score. It is not pooled, tested or used to adjust anything. A difference between the two
  is read as sampling plus fresh openings, nothing more.

## 7. Smallest improvement worth detecting, and the number of openings

- **Smallest improvement worth detecting: +0.05 in mean score** (from about 0.118 to about 0.168). That closes about
  an eighth of the distance to 0.5, and about 40% more points per game against the teacher. Anything smaller would
  not be worth knowing at this cost: detecting +0.04 at 80% power needs about 400 openings (about 15.5 h of play).
- **Power (planning simulation, `docs/superpowers/evidence/hard-teacher-bench-plan/power.py`, 3,000 runs per cell;
  output in `power_output.txt` there):**
  - **Model:** the previous Hard's per-colour rates are entry 65 arm P's (as Red 18 W / 6 D / 86 L, as Black
    5 / 0 / 105; Red and Black results were uncorrelated within a pair, r = 0.005). The new version's win
    probabilities are scaled to raise the mean score by δ. A shared latent per (opening, colour) correlates the two
    versions' games with correlation ρ. Arm N and arm P pair scores on common openings correlated at 0.36, so ρ 0.36
    is the central case, 0 the worst and 0.6 the likely one with shared seeds.
  - **Results at 300 openings, sign-flip test** (sign test in brackets):

| | δ = 0 (false CLOSER) | δ = 0.05 | δ = 0.06 |
|---|---|---|---|
| ρ = 0 | 0.043 | **0.83** (0.77) | 0.93 |
| ρ = 0.36 | 0.048 | **0.89** (0.85) | 0.96 |
| ρ = 0.6 | 0.047 | **0.95** (0.93) | 0.99 |

  - At 250 openings power at δ 0.05 is 0.75–0.91; at 350, 0.88–0.97. **300 openings** gives at least 80% in the worst
    case. The sign-flip test was chosen over the sign test because it uses the size of each difference and had
    more power in every cell, at the same size.
  - These figures assume the gain is spread evenly across colours and openings; a gain concentrated in a few
    openings would lower them.

## 8. Timing smoke and the fixed compute cap

- **Planning estimate:** entry 65's arm P played 220 games against this teacher at 34.6 s per game (median 20 s; six
  560–572-ply draws at 415–475 s each). Its Hard moves averaged 1,346 ms (5,445 moves), the teacher's 23 ms, and
  other per-game overhead 0.74 s. Both Hard versions run the same 200-simulation search at matched speed (entry 78:
  search-seconds ratio 1.003). **1,200 games ≈ 41,600 s ≈ 11.6 h**, plus startup.
- **Fixed cap: 24 h for the match** (TERM 86,390 s, KILL 86,400 s), excluding the idle gate's wait. It is never
  raised. A run it stops is VOID, with no partial score. The margin covers a higher draw rate: each extra draw costs
  about 430 s, so the cap holds up to a draw rate of about 11%, against 2.7% in entry 65.
- **Timing smoke:** 1 opening, 4 games (each version as Red and as Black), its own namespace, TERM 3,590 s and KILL
  3,600 s (four 572-ply draws would take about 1,900 s). **Its outcomes are never read, scored or reported.** It
  measures, per stage, the startup seconds and the mean milliseconds per Hard move for each version.
- **Projection:** P = startup + 1,200 × (33.31 s × r + 1.32 s), where r = the smoke's mean Hard-move milliseconds
  (both versions) ÷ 1,346 ms. The 33.31 s and 1.32 s are entry 65 arm P's mean Hard-move seconds and other seconds
  per game. Game lengths come from 220 measured games, not from the smoke's four.
- **LAUNCH_OK iff** 1.5 × P ≤ 86,400 s (that is r ≤ about 1.40) **and** each version's mean Hard-move time is within
  ±10% of the other's. Otherwise **REVIEW**: no launch, and the numbers return to the user. The cap does not change.
- **Machine conditions:** the user's: BOINC quit, Unreal Editor closed (the wrapper refuses while it runs and never
  stops it), the idle gate passed. **The local product server must not be running** during the smoke or the match,
  or the wrapper refuses; stopping and restarting it is the user's call.

## 9. Validity and admission (fail-closed, entry 65's rules)

- **Stops (VOID, no score, no retry):** any lockstep or canonical-state divergence, illegal move, non-finite output,
  model-identity or runtime mismatch, process failure, a missing or duplicate per-ply row, a Hard move without 200
  simulations and 200 visits, or the cap.
- **Admission:** the frozen analyser admits a stage only with `outer.json` COMPLETED (inner and outer exit 0, gate
  restored, no survivors, record written by the script), the stage's ceiling in both records, the same launch commit
  in both, `pgid.txt` equal to the driver's PID, the pinned runtimes and model ids, exactly the scheduled game files,
  each replaying legally from its re-derived opening with re-derived seeds. The match admits only the committed
  smoke evidence and re-derives `timing.json` byte for byte.
- **Shutdown:** the driver sets `process.exitCode`, never calls `process.exit()`, and must pass
  `server/test_ort_shutdown.js` before launch.

## 10. Prediction on record

Mine, before any build or game, low confidence:
- **Previous Hard:** about 0.12 (0.07–0.18), consistent with entry 65.
- **New Hard:** about 0.15 (0.09–0.22).
- **Mean D:** about +0.03 (−0.02 to +0.08). **CLOSER: about a 35% chance.**
- **Basis:** the H-MX fit moved the network markedly toward the teacher on positions from Hard's own play (H-OP
  validation top-1 0.576 → 0.659, KL 0.707 → 0.452, entry 73) and slightly on the original H-TB positions (top-1
  0.651 → 0.660). But its head-to-head gain over the previous Hard was small (0.519, entry 78), and entry 65 found
  that our search closed little of the gap (arm P − arm N +0.070). So I expect a gain, but one near the detection
  threshold.

## 11. What each outcome would support

| Result | Reading | Supports |
|---|---|---|
| **CLOSER** | Under this protocol, the new Hard scores more against the teacher than the previous Hard | A statement that the H-MX update moved Hard toward its teacher in play. As context only, it could inform a future distillation proposal, which needs its own card |
| **NOT SHOWN** | No difference of the planned size was detected | Nothing. Not evidence of no change; no top-up, no re-score |
| **LOWER** (descriptive flag) | The new Hard scored less against the teacher | A note for the ledger only; the deployment decision rests on the head-to-head match |

None of them changes the deployment, opens a gate or authorizes training.

## 12. What a model-free build would contain

- **The opening list** (§5, §5.1), generated and committed with its skip counts, and the seed schedule with its
  distinctness check.
- **A Node driver** (`tests/htb_integration/hard_teacher_bench.mjs`), built from `teacher_bench.mjs`: arm P's code
  path for both versions (explicit manifests), the teacher process, the cross-engine checks, durable per-game records,
  `record.json`, `process.exitCode`.
- **The teacher process:** `htb_bench_teacher.py`'s serve loop and checks, unchanged in behaviour, behind this card's
  own gate.
- **A frozen analyser** (`hard_teacher_bench_analyse.mjs`): admission, the exact sign-flip test, both intervals, the
  per-version scores and `timing`. Its sha256 is written into this card before any model loads.
- **Two wrappers** (smoke, match) on the G3 / entry 65 / H-MX pattern: one-line gate-open commit, idle gate, BOINC
  and Unreal checks, the local-product check, process-group supervisor with TERM and KILL, `outer.json`, gate restored
  on every exit, and the timing rule between stages.
- **Model-free tests:** opening generation and equivalence under all four symmetries, including move-order
  equivalence; the exclusion check against each listed set; seed independence from the version; the sign-flip
  test against brute-force enumeration on small cases, and its boundary; both intervals; admission failures against
  a clean control; both versions checked move for move against `protocolMove` with fake evaluators; the gate refusing
  while closed.

## 13. What this card does not authorize

- Building, loading either model, starting the teacher, generating openings or playing any game before the §2
  exception, approval of this card, and a launch review.
- Stopping or restarting the local product server.
- Any training, labelling or fit; any use of these games as training data.
- Pushing, a hosted deployment or a change to `main`.
- Rerunning, extending or re-scoring the run, or changing a setting after a result.
