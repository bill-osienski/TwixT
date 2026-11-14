# Benchmark card: the H-TB student against its pinned twixtbot teacher, under our rules

**Status:** DRAFT for review, 2026-11-14. **Nothing is built, loaded or played under this card.** It follows
Diagnostic O, whose recorded result stands: **no signal fired** (ledger entry 64, #194). Under that plan's frozen rule
this benchmark card is the default next step. Diagnostic O neither motivates nor shapes it beyond that.

This card needs two approvals before any game: the **#52 exception in §2**, and the card itself. A build would then
return once for launch review, as G3 did.

**Revision 1 (2026-11-14, the user's review).** The user's review keeps both arms, the fixed teacher configuration,
the pair counts and the scoring thresholds. It recommends approving the §2 scope and this revised build together,
followed by one review of the completed build before launch. The changes:
- **Draw check (§6, §7).** The user replayed G3's four draws through the rules code. twixtbot's game declared a draw
  at plies 73–127, while the product played on to ply 572, and every other field agreed throughout. (My replay: the
  flag first appears after plies 73, 80, 95 and 127; it is never set in G3's 216 decisive games.)
  - The product's game-end rule stays authoritative.
  - twixtbot's early-draw flag is recorded separately and never ends a game.
  - One of those replays becomes a regression test.
  - Full canonical state is compared across Node and Python, so a real bridge or winner mismatch still voids the run.
- **Timing (§9).** The four-game smoke has its own ceiling. The projection includes startup and the lockstep checks.
  The 1.5× check is a launch estimate only, since two games per arm can miss long draws; the 6.0 h match cap stays
  fixed.
- **Training decision (§8).** Only arm N TRAILS would justify proposing a fresh distillation experiment. UNRESOLVED
  leaves the question open. Arm P alone cannot say whether search, opening sampling or the network accounts for its
  result.
- **Exception scope (§2):** exactly the four smoke games and the 620 scored games.

**Repair 1 (2026-11-14, the user's launch-check review), before either gate opens.** Unchanged: the four-game smoke,
the 620 scored games, the fixed 6.0 h cap and the arm N decision rule. The changes:
- **Evidence bound to its run** (§7.1), with G3's checks:
  - the inner and outer records name the same launch commit;
  - `pgid.txt` is the driver's PID;
  - the recorded ceiling is the stage's;
  - the match admits only committed smoke evidence.
- **The search budget enforced.** There must be exactly one evidence row per played ply. Every arm P student move
  must record a 200-simulation budget and 200 root visits actually spent.
  - A wrong budget, a duplicate row, a missing row or an extra row is rejected.
  - The earlier model-free smoke fixture played at 64 simulations and still produced a timing estimate. It is now
    refused.
- **Pinned runtimes checked before any model loads** (§4):
  - onnxruntime-node, as the server resolves it;
  - the teacher's Python, TensorFlow and NumPy versions.
  - The product's model resolver already verifies both student artifact files.
- **Smoke outcomes:** replay-validated by the analyser, then excluded from scoring and from every report.
- **The early-draw flag is purely observational** in both lanes. The reflection-lane comparison that could stop the
  run on the flag alone is removed.

## 1. Question, and what this is not

**Question:** does the current H-TB student still trail its teacher under our game rules?

It is asked twice, with one fixed teacher, and each student budget answers its own question (§5):
- **Arm N (network):** does the student's **network** still trail the teacher's network?
- **Arm P (product):** does the student **as shipped in `hard`** still trail that same teacher?

**This is a student–teacher comparison, not an anchor.**
- It calibrates nothing and produces no Elo, no absolute strength and no lower bound on any gap.
- It says nothing about `calib020_0001`, `0379`, legacy `hard` or any other checkpoint.
- Its result may not be used to choose teacher settings for anything later.

## 2. Ledger item #52, and the narrow exception this card needs

**What #52 records.** It covers the 2026-08-24 anchor pilot: 128 games in which twixtbot won every game against
`0379` and `calib020_0001`, at every `trials` setting including 0. #52 then:
- forbids rerunning that pilot, altering its `trials` ladder, relaxing its band, changing or adding openings, granting
  the anchor a rules change or handicap, adding games, and hunting a weaker anchor configuration;
- reads the result as an ordering under that protocol only, and explicitly **not** a rules-matched comparison;
- requires that "a future external anchor must be a *different* engine or a rules-matched one, and needs its own scope".

**Why this card is not that pilot.**
- **Different subject:** the H-TB student, which did not exist in August, rather than our older checkpoints.
- **Different purpose:** a student against the teacher it was distilled from, not calibration against an outside
  reference.
- **Different shape:** no `trials` ladder, no saturation band and no selection rule.
- **Fresh openings and seeds** (§6).

**Why it still needs an exception.** It plays games against the same engine's network, which is not rules-matched.
Read broadly, #52's last requirement covers it.

**The exception requested, stated narrowly.** Approve twixtbot's pinned network (§4) as an opponent **only**:
- as the student's teacher, in this card's two arms;
- at the one teacher configuration in §5, fixed here before any game;
- on the openings and seeds of §6;
- for one run of exactly **4 smoke games** (unscored, for timing only) and **620 scored games** (arm N 400, arm P
  220), and no others.

**The exception does not cover:**
- any anchor use: Elo, calibration, or ranking any other checkpoint;
- twixtbot's own player or any `trials` setting;
- changing the teacher's configuration, the arms, the openings or the game counts after any result;
- rerunning, extending or topping up;
- any rules change or handicap.

**Why the teacher's configuration is not a weaker-anchor hunt.** It is the network's policy exactly as the student
was trained on it: the H-TB label path's mirror-averaged raw output over our legal moves. It is chosen for that reason,
before any game. The August pilot already showed that the unsearched network beat our older checkpoints, so this is
not a setting chosen to lose.

## 3. The teacher's training-rule mismatch

- **The mismatch.** twixtbot's network was trained with **own-link crossings allowed**. Under our rules they are
  forbidden (`allow_scl=False`), and twixtbot's own repository records a game where the network misjudged a position
  for that reason (pilot card, "Rules").
- **What it does to the teacher.** Its policy and value are learned for a game slightly different from ours, so
  under our rules it is a possibly handicapped player, by an unmeasured amount.
- **What it does to the student.** The student was distilled from those outputs, read over our legal moves on
  positions from our rules. It inherits the teacher's judgements, including any rules-driven errors.
- **How to read the result.** It compares the student with **the teacher as it plays under our rules**. It is not
  a measure of the student against twixtbot's strength under its own rules, and it neither measures nor corrects
  the mismatch.
- **Not run:** a `allow_scl=True` arm. Our engine cannot represent own-link crossings, so the two sides' states
  could diverge.

## 4. Identities (fixed before any game)

| Item | Identity |
|---|---|
| **Student network** | product `hard` model `78c16a31c3dba37f` (ONNX sha256 `bc68a2742803a8466856fa353958b168becff1044dd387da001ddcaeb7e54f24`), exported from the H-TB student checkpoint sha256 `0e1cb6211aa0b3e2dec6ab6f39bec74ceb8c472c7dcf603e7d331b1d1c1c34e5`. The model id is checked against the hard manifest at launch, as in G3 and Diagnostic O. |
| **Student inference** | `MirrorAveragingInference(AlphaZeroInference)`, as `server/index.js` builds it; onnxruntime-node 1.23.2. |
| **Teacher** | `github.com/stevens68/twixtbot-ui` @ `83749f230a0bae1766b46a05bfde0ed87f0a9a0a`, with weights `variables.data` sha256 `1958f847…`, `saved_model.pb` `e0c2b882…` and `variables.index` `1d4073c4…` (pilot card, "Identity"). These are checked at load, as in Diagnostic O. |
| **Teacher inference** | the H-TB label path's wrapper (`htb_pilot.TwixtbotTeacher` via `_load_teacher_real`): one raw network call per position with rotation off, softmax over our legal moves, and averaging over the position and its left-right reflection (`reflect_rc`: column c → 23 − c), as in `label_game`. |
| **Teacher runtime** | teacher venv: Python 3.13.15, TensorFlow 2.21.0, NumPy 2.5.2 (CPU), with the clone at `$HOME/twixt-cache/htb-teacher/twixtbot-ui`. |
| **Runtime checks, before any model loads** | onnxruntime-node **1.23.2** (the copy `server/inference.js` resolves) and the teacher's three versions above. They are read from package metadata by each wrapper before launch. The driver checks ORT before loading the student. The teacher process checks its versions and refuses (exit 4) before loading its model, and the driver loads the student only after that check passes. The analyser requires the versions recorded in `record.json` to equal the pins. The student's two artifact files are verified by the product's model resolver. |

## 5. Arms and budgets: each budget for the question it answers

**The teacher is the same in both arms:** the mirror-averaged raw policy, with no search. The move is the argmax over
our legal moves, ties to the lower (row, col). It is deterministic.
- **Why no search for the teacher.** The student was distilled from this exact unsearched output, so it is the
  teacher the question is about.
- **Not used:** twixtbot's own player and its `trials` search. A searched teacher would ask how strong twixtbot's
  engine is, which is not a question about the student. Varying it would also come close to the ladder #52 forbids.

| Arm | Student budget | The question it answers |
|---|---|---|
| **N (network)** | **No search.** The student network's mirror-averaged policy argmax over legal moves, ties to the lower (row, col). Deterministic. | Network against network, on equal footing: the same input symmetry, the same move rule, no search on either side. Does distillation still leave the student's policy short of the policy it imitated? |
| **P (product)** | **As shipped in `hard`.** `protocolMove`: 200 simulations, cPuct 1.5, T 1.0 sampled readout through ply 19, halving from ply 20, with seeded `PyRandom` per move (§6). | Does the player that ships still trail the teacher's network? Our search and readout come on top of the student network. This arm does not match compute. |

**Excluded on purpose:**
- any teacher search;
- a matched-compute arm, because twixtbot's MCTS and ours are not comparable per simulation;
- other student budgets, such as 800 simulations: Diagnostic O's budget signal did not fire, and this card is not a
  budget study.

## 6. Openings, colour balance, seeds and game counts

- **Rules:** the empty 24×24 board, red first, no swap, our rules (`allow_scl=False`, automatic links, no link
  removal). Our `TwixtState` is authoritative.
- **Game end:** the product's rule, as in G3, and it alone: play until Node's `TwixtState.isTerminal()`, with no ply
  cap. A game ends in a win, or in a draw when no legal moves remain.
  - twixtbot's `Game` sets `result = DRAW` early, when its own test decides neither side can still connect. In G3's
    four draws that came at plies 73–127, against the product's 572.
  - **That flag is purely observational.** It never ends a game, is never compared between lanes and is never a
    stop. The first ply at which it is set is recorded per game and per lane (`tb_draw_flag_ply` for the position,
    `tb_draw_flag_ply_reflected` for its reflection), and the position's flag is summarised descriptively (§8).
- **Openings:** both players are deterministic in arm N, so without supplied openings arm N would play only one game
  per colour. So every game starts from a **supplied 4-ply opening** (plies 0–3, red–black–red–black).
  - **Generation:** opening *i*, ply *k*, takes index `int(sha256("htb-teacher-bench/20261114/opening <i>/ply <k>")[:12],
    16) mod n`, where *n* is the number of our legal moves in rows and columns 4–19 at that ply, in sorted order.
  - **Distinctness:** an opening that equals an earlier one, directly or as its left-right reflection, is skipped,
    and the next *i* is used. Under mirror averaging, a reflected opening would replay the reflected game.
  - The list is generated and committed before any model loads, and the analyser re-derives it.
- **Colour balance:** pair *p* plays opening *p* twice: game 0 with the student as Red, game 1 with the student as
  Black.
- **Game counts, fixed:**
  - **Arm N: 200 pairs (400 games)**, on openings 0–199. Deterministic, and cheap (§9).
  - **Arm P: 110 pairs (220 games)**, on openings 0–109, the same openings as arm N's first 110 pairs, so the two arms
    can be compared on common openings. 110 pairs is G3's size.
  - **Smoke:** 1 pair in each arm (4 games) on a separate opening namespace (`…/smoke opening <i>/…`). The analyser
    replay-validates its four games on admission, like the match's (§7.1). Their outcomes are **excluded from scoring
    and from every report**: `timing` reads wall times only. The smoke exists only for the launch estimate in §9.
- **Seeds:** only arm P's student draws randomness.
  - Seed formula: `sha256("htb-teacher-bench/20261114/<stage>/pair <p>/game <g>/ply <ply>")`. Hex digits 0–11 are the
    search seed and digits 12–23 the readout seed, as in G3. `<stage>` is `smoke` or `match`.
  - The `htb-teacher-bench` namespace has not been used before. The build checks that every reachable seed is below
    2^48 and that all are distinct.
  - Arm N and the teacher draw no randomness.

## 7. Lockstep and validity (fail-closed)

- **Teacher state.** The teacher keeps a twixtbot game and its reflection in lockstep with our `TwixtState` and its
  reflection, one move at a time.
- **Check after every ply,** with the adapter's divergence check (`twixtbot_adapter.state_divergences`, no ply cap):
  pegs, bridges, side to move, the full legal set, winner and ply. Both the position and its reflection are checked,
  and the reflection must be the reflection of the position.
  - The adapter's `terminal` comparison is the one field left out. With no cap, it differs only when twixtbot's
    early-draw flag is set (§6), and the rules-level terminal status is already fixed by the winner and the legal set,
    which are compared.
  - Rules-level terminal status (a winner, or no legal moves) is also compared directly on both sides.
- **Cross-engine check: full canonical state.** After the opening and after every ply, the driver (Node, product
  code) and the teacher process (Python, our `TwixtState`, checked against twixtbot as above) each compute a
  canonical state: ply, side to move, pegs by colour, bridges by colour, the sorted legal set, winner and rules-level
  terminal status. They compare its sha256. Any difference stops the run, so a real bridge or winner mismatch voids it.
- **Regression test:** G3's `pair_0057_game_1` (draw flag after ply 73, product end at ply 572) is replayed through
  the teacher process's lockstep and must pass every ply. The same replay checks that the adapter's unfiltered check
  would report the `terminal` difference, and that an injected bridge mismatch stops.
- **Every move,** from either side, must be legal in our engine.
- **Stops (VOID):** any divergence, illegal move, non-finite output, process failure, model-identity mismatch or
  ceiling.
  - A VOID run has **no score**, and no partial result is computed.
  - It is returned for assessment, with no retry.

### 7.1 Admission: the evidence is bound to its run (Repair 1)

The analyser admits a stage, smoke or match, only if every check below holds. Otherwise it exits 3 and computes no
number.
- **The run's records** (G3's binding):
  - `outer.json` says COMPLETED, with inner and outer exits 0, the gate restored, no survivors and the record written
    by the script.
  - The stage's ceiling: TERM = KILL − 10 s, that is 1,790 + 10 s for the smoke and 21,590 + 10 s for the match.
  - `record.json` is COMPLETED, with the stage's `kill_s`.
  - Both records name the same launch commit, and `pgid.txt` is the driver's own PID (`record.json` `pid`).
- **The pins:** `record.json` carries the pinned runtimes (§4) and the student's model id and 200-simulation policy.
- **The games:** exactly the scheduled game files. Each one:
  - replays legally under the product's game-end rule;
  - matches the re-derived opening, colours, result and score;
  - has **exactly one per-move row per played ply** after the opening, in order;
  - has, on every **arm P student row**, the re-derived seeds, `nSimulations` = 200 and `visits` = 200 (the root
    visits the search actually spent);
  - has no seeds, budget or visits on any arm N or teacher row.
- **The match additionally requires the smoke evidence to be committed.** The driver checks with git that it is
  tracked, unchanged at HEAD and complete. The match wrapper checks the same files and the four games, then re-derives
  `timing.json` byte for byte.

## 8. Scoring, classification and the meaning of a sweep (fixed before any game)

- **Score:** the student scores 1 for a win, 0.5 for a draw and 0 for a loss. A pair's score is the mean of its two
  games. Draws are also reported separately, by arm and colour.
- **Interval:** two-sided Hoeffding at 95% over pairs, built from two one-sided bounds at 2.5%. The half-width is
  √(ln 40 / (2n)):
  - **arm N** (n = 200): **0.0960**;
  - **arm P** (n = 110): **0.1295**.
- **Classification, per arm:**
  - **TRAILS:** mean + h < 0.5.
  - **AHEAD:** mean − h > 0.5.
  - **UNRESOLVED:** otherwise. An UNRESOLVED arm does **not** mean equal strength.
  - So arm N can resolve means outside [0.404, 0.596], and arm P means outside [0.371, 0.630].
- **A sweep (0 or 1.000 in an arm)** classifies normally, as TRAILS or AHEAD. It is read only as an **ordering under
  this protocol**:
  - It gives no size of the gap, no Elo and no lower bound.
  - It does not justify adding games, changing either side's budget, or a teacher handicap. Those were excluded
    before any result.
  - It is a result, not a defect. Validity rests on §7 alone.
- **Descriptive, not classified:**
  - the score as Red and as Black;
  - game lengths;
  - arm P − arm N on the 110 common openings.
- **What each outcome would support** (signals for choosing a follow-up, with no causes assigned; each follow-up
  needs its own approval):

| Arm N | Reading | Follow-up it would support |
|---|---|---|
| **TRAILS** | The student's raw policy still trails the policy it learned from, under our rules | Proposing a **fresh distillation experiment**: training data from the student's own play, labelled by this teacher, fitted from the current student, with training positions kept separate from evaluation, and tested against the current student on fresh games |
| **UNRESOLVED** | The question stays open | None from this card |
| **AHEAD** | The student's raw policy no longer trails its teacher under our rules | None toward more distillation from this teacher |

- **Arm P is reported for product context** and does not choose a follow-up. Its result mixes the network, our
  search and the T 1.0 opening sampling, so arm P alone cannot say which of them accounts for it.
- Draws, scores by colour, game lengths, `tb_draw_flag_ply` and arm P − arm N are descriptive.

## 9. Time ceiling (fixed before any game)

- **Ceiling: 6.0 h for the match,** TERM at 21,590 s and KILL at 21,600 s. It is never raised.
  - The idle gate's wait is excluded, as in G3 and Diagnostic O.
  - The machine conditions are the user's: BOINC quit, Unreal Editor closed. The wrapper refuses while Unreal Editor
    is running and never stops it.
- **Planning estimate:**
  - **Arm N:** about 45 ms per ply (two teacher calls of about 19 ms each, as Diagnostic O measured with 119 s for
    6,220 calls, plus checks and two ONNX calls). At about 60 plies, that is about 2.7 s per game, or **about 18 min**
    for 400 games.
  - **Arm P:** G3 measured new `hard` at 1.349 s per move. At about 60 plies (30 student moves), that is about 42 s
    per game, or **about 2.6 h** for 220 games.
  - **Total:** about 2.9 h. Games against the teacher may run longer than G3's (mean about 55 plies), so the ceiling
    covers a mean of about 120 plies.
- **The smoke has its own ceiling:** TERM at 1,790 s and KILL at 1,800 s (30 min). Even if all four smoke games were
  572-ply draws, they would fit with margin.
- **Launch estimate, from the smoke** (4 games, outcomes unread):
  - The projection is: the measured startup (model load, teacher load and the teacher's first reply), plus the arm N
    smoke's mean wall seconds per game × 400, plus the arm P smoke's mean wall seconds per game × 220.
  - The per-game wall time includes every lockstep and canonical-state check, since they run inside the game loop.
  - If **1.5 × the projection > 6.0 h**, the match is **not launched**, and the projection returns for review.
  - The 1.5× check is an **estimate**, not a guarantee: two games per arm can miss long draws.
  - **The 6.0 h match cap stays fixed** whatever the estimate. A match it stops is VOID, with no partial score.
  - `timing.json` is derived from the committed smoke evidence by the frozen analyser, and the match wrapper
    re-derives it and requires a byte-equal result.
- **Shutdown:** every Node driver sets `process.exitCode`, never calls `process.exit()`, and must pass
  `server/test_ort_shutdown.js` before launch.

## 10. Prediction on record

Mine, low confidence, stated before any build or game:
- **Arm N: TRAILS,** with a mean pair score of about 0.25 (I expect between 0.05 and 0.45).
  - Basis: the fit check (entry 52) left the student's top-1 agreement with the teacher at 65.1%, with policy KL
    0.574. A student at that distance would usually lose a network-against-network game.
- **Arm P: UNRESOLVED,** with a mean of about 0.5 (I expect between 0.30 and 0.75).
  - Basis: our search clearly adds strength (G3: 0.93 against legacy `hard` at 800 simulations), but how it compares
    with the teacher's raw policy has never been measured.

## 11. What a build would contain (approved with revision 1)

All model-free until launch:
- **A teacher play process** (`scripts/GPU/alphazero/htb_bench_teacher.py`, teacher venv). It speaks JSON lines with
  three commands: start a game from an opening, apply a move, and report the teacher's move and a digest of the legal
  set. It does incremental lockstep with the reflection, through the label path's wrapper. It has its own gate.
- **A Node driver** (`tests/htb_integration/teacher_bench.mjs`). It covers openings and seeds, the student in both
  arms (product code for arm P), the cross-engine checks, durable per-game records and `record.json`. It sets
  `process.exitCode`.
- **A frozen analyser,** whose sha256 is written into this card before any model loads.
- **A wrapper,** copied from the G3 and Diagnostic O pattern:
  - a one-line gate-open commit, refusing any other HEAD;
  - the idle gate, the BOINC check and the Unreal Editor refusal;
  - a process-group supervisor with TERM and KILL;
  - `outer.json`;
  - gate restoration on every exit;
  - the timing rule between the smoke and the match.
- **Model-free tests:**
  - a fake teacher and the fake evaluator;
  - the protocol arm checked move for move against `protocolMove`;
  - opening generation, distinctness under reflection, seed distinctness, classification boundaries and sweep
    handling;
  - the gate, which must refuse while closed.
- **Then** one launch review, then one run. The ledger is updated when the run is judged.

### 11.1 Built (2026-11-14, revision 1), re-pinned after Repair 1: the frozen files

The first build (`e4aa15b8`) pinned other hashes; Repair 1 changed the analyser, driver, teacher process and both
wrappers, and the table below supersedes them. No file was ever run under the earlier pins.

The analysis, frozen before any model load: teacher_bench_analyse.mjs sha256 `b162323e5db22bd8f1df08d11322feeba5183ce05fca0bf3fae4ec1212b81d4f`.
Both wrappers refuse to launch if the analyser differs from it.

| File | sha256 |
|---|---|
| `tests/htb_integration/teacher_bench.mjs` (driver) | `7ad449a80e545de7626dcedc7799b01323f5244e0082894ea4cdd204fa075bd7` |
| `scripts/GPU/alphazero/htb_bench_teacher.py` (teacher process) | `ea2bab26be1af792616221853cc160bce5d06f91781e3778de843f2881f5c94f` |
| `tests/htb_integration/teacher_bench_authorization.mjs` (both gates closed) | `f1cd212eaa43daff78ea93b3ec4e09e154955040fc2a3be08e3b592caeef340f` |
| `docs/superpowers/evidence/htb-teacher-bench-smoke-run/01_wrapper.sh` | `56fb66de49ca453600f4635890cf891c45595d0f5760a3c5adc9d3cf3fa62d7c` |
| `docs/superpowers/evidence/htb-teacher-bench-match-run/01_wrapper.sh` | `e075a8c2fd5707b4f2eab63be201a7903e3b4daa103e6dfcc51b0d690b1adc10` |

- **Model-free tests:**
  - `tests/htb_integration/test_teacher_bench.mjs`: 17 pass (after Repair 1). Among them:
    - the cross-language canonical digest at every ply of a 572-ply G3 draw and a decisive game;
    - a full smoke schedule through the real Python serve loop, with twixtbot's `Game` in lockstep and a fake
      network;
    - arm P checked move for move against `protocolMove`;
    - run binding: each failure (no outer record, a crash, another launch commit, a wrong `pgid.txt`, the wrong
      ceiling in either record, a wrong ORT or teacher version, a 64-simulation policy) rejected against a clean
      control, with the smoke played at the real 200-simulation budget;
    - per-move evidence: a 64-simulation budget, fewer visits, a duplicate, missing or extra row, and a teacher row
      with a budget are each rejected; the old 64-simulation smoke fixture no longer yields an estimate;
    - committed smoke evidence for the match, and the runtime pins.
  - `tests/test_htb_bench_teacher.py`: 12 pass (after Repair 1). Among them:
    - the G3 `pair_0057_game_1` regression;
    - an own-crossing bridge mismatch that stops;
    - the mirror averaging, which is identical to Diagnostic O's label-path implementation;
    - the early-draw flag set in one lane only, which never stops play and is recorded per lane;
    - a wrong teacher runtime, refused before the teacher model is loaded.
- **Measured check cost (fake network, model-free):** about 5 ms per ply for both lanes' checks in a 41-ply game,
  rising to about 60 ms per ply late in a 572-ply draw (about 35 s for the whole replay). This is small against the
  §9 estimate; the smoke measures the real cost.
- **Each stage, in order:**
  1. a one-line gate-open commit in a clean launch worktree;
  2. its wrapper;
  3. the evidence committed with the gate restored.
  - The smoke's `timing.json` is derived by `teacher_bench_analyse.mjs timing` and committed before the match
    wrapper will run.

## 12. What this card does not authorize

- Loading either model, or playing any game, before the §2 exception, approval of this card and a launch review.
- Any training, labelling or fit. The §8 follow-up rows are proposals only.
- Pushing, deployment or a change to `main`.
- Rerunning, extending or re-scoring the run. Changing a setting after a result.
