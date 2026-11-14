# Benchmark card: the H-TB student against its pinned twixtbot teacher, under our rules

**Status:** DRAFT for review, 2026-11-14. **Nothing is built, loaded or played under this card.** It follows
Diagnostic O, whose recorded result stands: **no signal fired** (ledger entry 64, #194). Under that plan's frozen rule
this benchmark card is the default next step. Diagnostic O neither motivates nor shapes it beyond that.

This card needs two approvals before any game: the **#52 exception in §2**, and the card itself. A build would then
return once for launch review, as G3 did.

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
- for one run of the game counts in §6.

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
- **Game end:** the product's rule, as in G3: play to `isTerminal()`, with no ply cap. A game ends in a win, or in a
  draw when no legal moves remain.
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
  - **Smoke:** 1 pair in each arm (4 games) on a separate opening namespace (`…/smoke opening <i>/…`), excluded from
    the result. Its outcomes are not read or reported; it exists only for the timing rule in §9.
- **Seeds:** only arm P's student draws randomness.
  - Seed formula: `sha256("htb-teacher-bench/20261114/<stage>/pair <p>/game <g>/ply <ply>")`. Hex digits 0–11 are the
    search seed and digits 12–23 the readout seed, as in G3. `<stage>` is `smoke` or `match`.
  - The `htb-teacher-bench` namespace has not been used before. The build checks that every reachable seed is below
    2^48 and that all are distinct.
  - Arm N and the teacher draw no randomness.

## 7. Lockstep and validity (fail-closed)

- **Teacher state.** The teacher keeps a twixtbot game and its reflection in lockstep with our `TwixtState` and its
  reflection, one move at a time.
- **Check after every ply,** with the adapter's divergence check: pegs, bridges, side to move, the full legal set,
  terminal status and winner. Both the position and its reflection are checked.
- **Cross-engine check.** The driver (Node, product code) and the teacher process (Python) compare a digest of the
  legal set every ply. Any difference stops the run.
- **Every move,** from either side, must be legal in our engine.
- **Stops (VOID):** any divergence, illegal move, non-finite output, process failure, model-identity mismatch or
  ceiling.
  - A VOID run has **no score**, and no partial result is computed.
  - It is returned for assessment, with no retry.

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

| Arm N | Arm P | Reading | Follow-up it would support |
|---|---|---|---|
| TRAILS | any | The student's network still trails the policy it imitated, under our rules | A card for **fresh training data from the student's own play, labelled by this teacher**. It would fit from the current student, keep training positions separate from evaluation, and be tested against the current student on fresh games. |
| UNRESOLVED or AHEAD | TRAILS | No evidence that the network trails, but the shipped player does | No further distillation from this teacher; the gap lies elsewhere (a new design, outside this card) |
| UNRESOLVED or AHEAD | UNRESOLVED or AHEAD | No evidence that the student trails its teacher in either form | No further distillation from this teacher; the next work needs a new design |

Arm P is reported for product context. Only arm N bears on whether more distillation from this teacher could pay off.

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
- **Timing rule, from the smoke** (4 games, outcomes unread):
  - The projection is the arm N smoke's seconds per game × 400 plus the arm P smoke's seconds per game × 220.
  - If **1.5 × the projection > 6.0 h**, the match is **not launched**, and the projection returns for review.
  - The ceiling is not changed to fit.
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

## 11. What a build would contain (after approval only)

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

## 12. What this card does not authorize

- Loading either model, or playing any game, before the §2 exception, approval of this card and a launch review.
- Any training, labelling or fit. The §8 follow-up rows are proposals only.
- Pushing, deployment or a change to `main`.
- Rerunning, extending or re-scoring the run. Changing a setting after a result.
