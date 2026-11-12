# After G3: where the H-TB student still loses, and one bounded next experiment

**Status:** a read-only analysis and a proposal, written 2026-11-11. No model was loaded and no game was played. No labels
were made and nothing was trained. Every step below needs its own approval.

**Reference:** the H-TB student, checkpoint sha256 `0e1cb621…`, product model `78c16a31c3dba37f`. In G3 the new
`hard` beat legacy `hard` with 203 wins, 13 losses and 4 draws (ledger entry 63).

## 1. What was read

- **Ledger entries 52–54.** The fit check came out ELIGIBLE: policy KL 0.574 against 2.911, top-1 agreement 65.1% against
  29.7%, value MSE 0.070 against 0.253. Two match results followed: the screen scored 0.953 and the research confirmation
  0.989. The Codex review addendum notes that no ablation was run, so the cause of the gain is unresolved.
- **Ledger entry 63:** the G3 result.
- **The twixtbot anchor run of 2026-08-24** and do-not-repeat item #52. twixtbot won all 128 games against our older
  checkpoints, even with `trials=0` (raw policy, no search). Three limits apply to that run:
  - The comparison is not rules-matched. twixtbot was trained with own-link crossings allowed and played with them
    forbidden.
  - A 1.000 score gives no Elo and no lower bound on the gap.
  - Item #52 bars rerunning that pilot "in this shape", hunting for a weaker anchor configuration, and granting a rules
    change or handicap.

## 2. What the saved G3 games show (rules-only replay)

**Method.** I replayed all 220 games with `server/gameLogic.js`. Every replayed result equals the recorded result. The
scripts and outputs are in `docs/superpowers/evidence/htb-g3-nonwin-replay/`.

**The race measure is my own heuristic and has not been validated.** For each side it counts the fewest extra pegs that
side needs to connect its two edges, ignoring the opponent's replies. Own pegs cost 0 and empty legal holes cost 1. A
link is usable if it is an existing bridge or crosses no existing bridge.

- **The non-wins.**
  - The 13 losses split 1 as Red and 12 as Black.
  - The 4 draws split 2 as Red and 2 as Black. Each draw filled the board at 572 plies. In 3 of the 4, the student was
    the first side left unable to connect.
  - Every non-win in the research confirmation was also as Black (entry 54).
  - Without a swap rule, Red moving first has a structural advantage. Legacy `hard`, for example, won 12 of its 13
    games as Red. So part of this colour skew is expected and is not a student weakness.
- **No recurring position.**
  - Legacy `hard` opened (18,19) in all 110 games in which the student played Black.
  - The student's reply at ply 1 was sampled from 72 different cells.
  - No two losses share more than 3 opening plies, and no reply cell has more than 2 losses.
  - So there is no repeated losing line to target.
- **The losses as Black were decided in the opening.**
  - In all 12, the student fell behind on the race measure for good between ply 3 and ply 13. That is inside plies 0–19,
    where the product samples its move at T 1.0 from a 200-simulation search instead of playing the search's choice.
  - The one loss as Red turned at ply 22.
  - In wins as Black, legacy `hard` fell behind for good at a median of ply 22.
- **Being behind is necessary for a loss here, but not sufficient.** Outcomes of the 110 games with the student as Black,
  by the race measure after ply 19:

  | Student's race position after ply 19 | Wins | Losses | Draws |
  |---|---|---|---|
  | Ahead | 47 | 0 | 0 |
  | Level, or 1 peg behind | 18 | 0 | 0 |
  | 2 or more pegs behind | 31 | **12** | 2 |

  Every loss came from the 45 games in the bottom row, and two thirds of those 45 games were still won.

**What the records cannot show.** No visit counts and no values were saved. The records do save each search's and each
readout's seed, and `protocolMove` takes them, so every student move can be reproduced exactly. So the games cannot
show whether an opening deficit came from:

- **(a) the readout:** a sampled move the search itself did not prefer;
- **(b) the search:** the 200-simulation search preferring a poor move that a deeper search with the same network would
  avoid;
- **(c) the network:** both searches preferring the move, with the value not seeing the deficit.

These three causes call for three different remedies. (a) is a protocol setting and needs no training. (b) concerns
search budget or search targets. (c) is a training target.

## 3. Proposal: one small diagnostic before the twixtbot benchmark

I recommend running **Diagnostic O (opening attribution)** first. It costs about 2 hours, and it decides whether the
next change should be training at all. The twixtbot benchmark you prefer comes second; §4 explains why and what it
would look like.

**What it runs.** It replays the student's Black opening moves (plies 1, 3, …, 19) from **all 110** games in which the
student played Black, which is 1,100 positions. All 110 are used so that the analysis doesn't select games by outcome.
No game is played. On each position it measures three things:

1. **Reproduction.** Run `protocolMove` with the recorded seeds. The move it produces must equal the recorded move, and
   any mismatch stops the diagnostic. Record the visit counts, the played move's visit share and rank, and the root
   value.
2. **Deeper search with the same network.** Run ProtocolMCTS at 800 simulations and take the most-visited move. Record
   that move and its root value.
3. **Teacher evaluation.** Score the position with the pinned twixtbot raw network (mirror-averaged, as in the label
   path). Record the teacher's policy probability and rank for the student's played move, and its value.

**What it measures.** Losses (12) are compared with wins (96), and within the "2 or more behind" group, the 12 losses
with the 31 wins. The comparison is limited to plies 1–19.

- **R (readout):** the share of played moves that were not the 200-simulation search's top choice, and the share with
  a visit share below 10%.
- **S (search):** the share of positions where the 800-simulation choice differs from the 200-simulation choice, and
  how much earlier the 800-simulation root value marks the eventual losses.
- **N (network):** whether the student's ply-19 root value tells eventual losses from wins (AUC), and whether the
  teacher's ply-19 value does (AUC). A paired comparison of the two AUCs follows.

**Decision rule.** These thresholds are proposed here and are fixed only if you approve them.

| Result | Reading | Next change |
|---|---|---|
| R: in ≥ 8 of the 12 losses, the deficit window contains a played move with < 10% visit share, at ≥ 2× the rate in the 31 matched wins | The opening readout causes the losses | **No training.** A readout experiment, such as T 0.5 or playing the search's top move from ply k, measured against the current new `hard` on fresh games |
| Not R; S: the 800-simulation choice differs at ≥ 1 deficit-window move in ≥ 8 of the 12 losses, and its value flags them earlier | The search budget, not the network | Either a product budget change, or search-target training. That training would first have to answer why the search-distillation line was stopped (#69, #87) |
| Neither; N: student AUC ≤ 0.6 while teacher AUC ≥ 0.75 | The student's value has a blind spot on its own Black openings that the teacher doesn't share | **Training:** continue distillation from the current student on the positions it now reaches (§5) |
| Neither, and the teacher's AUC is also ≤ 0.6 | The teacher doesn't know these positions either | Distilling from this teacher won't fix them; a new line is needed |

**Prediction on record.** Low confidence. The readout explains a minority of the losses, not 8 of 12. The student's
ply-19 value separates losses less well than the teacher's.

**Cost.**
- Reproduction: 1,100 searches × 1.35 s, about 25 min.
- Deeper search: 1,100 × about 5.4 s (800 simulations), about 1.7 h.
- Teacher evaluation: 2,200 calls at about 16 ms each (the label run's rate), under 1 min.
- **Total about 2.1 h on CPU**, with a proposed ceiling of 3.2 h (1.5×). No seeds are used; the recorded ones are only
  read.
- Building it is one script, with a check that reproduction is exact.

## 4. The twixtbot benchmark, as the second step

A fresh, colour-balanced benchmark of the student against the pinned twixtbot under our rules answers a different
question: how much of the teacher's strength the student keeps. That decides whether more distillation from this
teacher can still pay off. Diagnostic O asks what costs the student games against today's opponent.

Several things favour running it after O:

- **Rules mismatch.** It is not rules-matched. twixtbot plays with own-link crossings forbidden, which it was not trained
  for, so its result is an ordering under our rules, not a rules-correct strength gap.
- **Item #52.** The benchmark needs a fresh card that measures a new subject (the student). It must use a declared
  budget ladder fixed before any game and fresh seeds, and it must not tune twixtbot's settings.
- **Risk of no information.** Another 1.000 sweep would tell us nothing. The design should therefore include an arm the
  student can plausibly win, such as student at 200 simulations against twixtbot `trials=0`, and state in advance what
  a sweep means.
- **Cost.** Planning estimate: 32 colour-reversed pairs at each of two declared twixtbot budgets (0 and 400 trials)
  makes 128 games. That is about 2.5–3.5 h, judging by the 3.5 h the 2026-08-24 run took up to 1,000 trials. The
  adapter with per-ply state equivalence is already qualified.

If O shows a gap in network values that the teacher fills, the benchmark becomes the natural check on how far that gap
reaches.

## 5. Constraints on any later fit

Any later fit, whatever result prompts it, keeps the conditions you set:

- It starts from the current student (`0e1cb621…`).
- Its training positions are fresh and kept separate from every evaluation set. The 24,000 validation rows are not
  reused for selection (#178), and none of the G3 or Diagnostic O positions are used for training.
- It must show improvement against the current student on fresh games, using a pre-registered colour-reversed rule and
  fresh seeds.

The specific change that O's "network" outcome would support is **on-policy continuation distillation**. The original
fit used positions from `calib020_0001`'s red-first corpus, so the student never trained on the positions it now
reaches itself, especially as Black. The change would label fresh positions from the student's own play with the
teacher, over the whole network and with the same recipe. Nothing else would change, which keeps this a one-factor
comparison against the entry-52 fit.

## 6. Approval needed

- **Diagnostic O:** build the script, then make one run. This loads the student and the teacher, and runs 2,200
  searches and 2,200 teacher calls. No games, labelling or training.
- **The twixtbot benchmark:** a card first, then its own approvals.
