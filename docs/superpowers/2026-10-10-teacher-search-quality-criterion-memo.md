# Teacher search quality — criterion memo (DESIGN ONLY, 2026-10-10)

**Question.** Does deeper twixtbot search produce demonstrably better moves under our rules? Any answer
must rest on a standard that our own rules engine verifies, defined and reviewed before any teacher query.

**Result: blocked.** A verifiable criterion can be defined (§1–§3). On the training-side material it
cannot measure an improvement, because the baseline already makes no mistake the standard can see, and
the material holds just one position where the standard could see one (§4). The pre-query feasibility stop
(§3, S0) fires. Recommendation: **stop. No teacher query.** `calib020_0001` stays the incumbent.

Material used: stage-1 **training** families 0–3 only (8 games, 256 decisions: 128 by the teacher,
128 by the incumbent). Held-out families 4–5, the stage-3 games and every T1j record were not opened.
Every figure comes from our engine alone; no model or teacher was loaded.

## 1. The standard: bounded proofs under our rules

**Judge.** Our `TwixtState` on the official 24×24 board, under protocol semantics:
* Links are added automatically, and any crossing is forbidden, including a crossing of one's own links.
* The ply cap is 280, and reaching it is a draw.

The classifier is the frozen M1 `classify` (`h5_incumbent_tactics`). It was qualified against the
brute-force `exhaustive_scan` (`tactical_candidate`), and re-checked on 66 of these training positions
today, all agreeing.

The teacher's own proof flags and rule flag play no part in the judging.

**Values of a move m at position P** (the mover plays m):
* **WIN3(m):** m wins at once, or leaves the opponent UNDEFENDABLE. The mover then wins within 3 plies
  whatever the reply.
* **LOSS4(m):** after m, either
  * the opponent wins at once, or
  * the opponent has a reply that leaves the mover UNDEFENDABLE.

  Either way the mover loses within 4 plies whatever it does.
* Otherwise **UNRESOLVED**.

**Verified mistake:** the chosen move is LOSS4 while some legal alternative is not LOSS4.
* Replacing it is verifiably never worse.
* It is strictly better unless the alternative also loses beyond the horizon. That cannot be ruled out,
  and it is the standard's stated ceiling.

**Deliberately not counted:** playing a non-WIN3 move when a WIN3 move exists.
* A slower win is not a value error, and a bounded proof cannot tell it from a lost win.
* It is reported descriptively and never counted as an error or a correction.

## 2. How corrections and new mistakes count

Each fixed position has a baseline choice b (100 trials, the stage-1 budget) and a deeper choice d.
Every position falls into exactly one class:

| class | b | d |
|---|---|---|
| **corrected** | verified mistake | not a mistake |
| **introduced** | not a mistake | verified mistake |
| persisted | mistake | mistake |
| unjudgeable change | b ≠ d, neither a mistake | |
| unchanged | b = d, not a mistake | |

* Only corrected and introduced positions carry evidence. Unjudgeable changes are reported and never
  scored. Most of what deeper search changes is expected to land there.
* **Claim rule (its own statistic):** an exact one-sided sign test on the discordant positions, corrected
  against introduced, at α = 0.05. With no introductions this needs **at least 5 corrections**
  (0.5⁵ = 0.031).
* Any introduction is reported beside the test. A positive result means only this: on these positions,
  deeper search avoids verified ≤4-ply losses that the baseline makes. It does not mean strength or
  better play beyond the horizon.

## 3. Position selection, query budget, stop conditions (fixed before any query)

* **Universe.** The decision positions of the stage-1 training families. Both movers count: the teacher
  can be asked what it would play in a position the incumbent faced.
* **Eligible.** A position where a verified mistake is possible: at least one LOSS4 legal move and at
  least one that is not. This is decided by our engine alone, before any query, and uses no teacher
  output.
* **Budget.** Each eligible position is queried once at 100 trials and once at 1,600.
  * Settings: a fresh root, no noise, temperature 0, no rotation, no swap, `smart_root` 0,
    `allow_scl=False`. That is stage 1's configuration except for the trial count.
  * At most 2 queries per eligible position.
  * No games, no seeds and no training.
* **S0, pre-query feasibility.** The baseline's verified mistakes on the eligible set must be able to
  reach 5. That bound is the known baseline mistakes on teacher-to-move positions plus the count of
  eligible incumbent-to-move positions. **Below 5: stop before any query.**
* **S1, during.** VOID, with no retries or substitutions, on any of these:
  * a 100-trial query that does not reproduce the recorded stage-1 move;
  * an illegal move;
  * any query error.
* **S2, outcome.** Any introduction is reported beside the sign test; no position is dropped after
  querying.

## 4. Feasibility on the training-side material: S0 fires

**One ply** (M1 classes):

| mover | QUIET | OWN_WIN | UNDEFENDABLE |
|---|---|---|---|
| teacher | 120 | 8 (all taken) | — |
| incumbent | 116 | — | 8 |

Neither side ever faced a defendable threat.

**Within four plies** (exact, for every chosen move):
* **Teacher:** **0 LOSS4 choices in 128.**
  * A WIN3 move existed in 16 teacher positions, all in the last four plies of each game, and the
    teacher took one every time.
* **Incumbent:** 16 LOSS4 choices.
  * In 15 of them every legal alternative was LOSS4 too, so the game was already lost.
  * One was a verified mistake: `f0-teacher_black`, ply 28.

**Trap probe.** This asks whether a mistake was even available. In every position, the three legal moves
farthest from every peg were tested for LOSS4.
* **No teacher position** produced a hit.
* All 16 incumbent hits were its last two decisions of each game.
* A miss is not a proof that no trap exists. The cheap "null move" proof is unsound under our rules: a
  player's own peg can block that player's own later link.

**Consequence.** The eligible set holds one confirmed position, so at most 1 baseline mistake is possible,
below the 5 that S0 requires.
* In these games the decisive play happens before the 4-ply horizon. The horizon only sees the
  conversion, and the 100-trial teacher already converts perfectly.
* Deeper search has nothing verifiable to correct here. It could only be shown not to introduce
  mistakes, which is not the question.

## 5. The blocker, and what would change it (none recommended now)

**The verifiable horizon is saturated at the baseline budget.** Whatever deeper search adds must act
beyond 4 plies, which is where we cannot verify. The ways around it each need a decision outside this
memo's scope:

1. **A deeper verified horizon** (win in 5, loss in 6).
   * Exhaustive search with this detector costs roughly 500× more per two plies. The 4-ply scan of 256
     positions already took about 1.5 hours.
   * A proof-number solver would be new code that needs its own qualification.
2. **Denser positions from another source**, such as our own self-play replays.
   * That is outside training-side material.
   * Constructed positions would not represent the play we care about.
3. **A non-tactical standard**, such as agreement with a stronger engine. That is not independently
   verifiable under our rules, and T1j stays evaluation-only.

**Recommendation: stop here.** There is no teacher query, probe, game, seed registration, training, gate
opening or push. The successful data pipeline does not by itself justify more training.

Also noted, without interpretation: the teacher won all 8 training games. That is consistent with stronger
play beyond the horizon, but it is not a verified standard for move quality, and it says nothing about
deeper budgets.
