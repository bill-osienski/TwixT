# EXPLORATORY — hypotheses from the H5 weakness review

Every entry separates what is **DEMONSTRATED** (one-ply facts from our rules engine, card §7
C1, with witnesses in `03_mechanical_checks.json`) from what is a **HYPOTHESIS** (not
demonstrated, no rate). Nothing here amends H5's confirmatory result, pools with H4, or
attributes anything to policy, value or search. Each hypothesis is a lead for a separately
carded experiment, never a finding. "Loss" means an incumbent loss; caps are unresolved.

How the inspection was done: every distinct loss and capped trajectory was stepped through
programmatically on its move list (identical to its exported replay, verified 592/592 at
export) — labels, first threats, final positions (`05_inspection_notes.md`). By eye I read
the final boards of all 6 Arm-A loss trajectories, 3 Arm-B loss trajectories (one only to
row 17) and 1 Arm-A cap (`h5s3-62-A`, rows 0–21), a board region of `h5s3-32-B` (rows 12–23,
cols 0–13) at ply 80 with T1j's process records for plies 78–82, and the move lists of the
6 Arm-A losses and `h5s0-26-B`; the targeted counts are in `06_summary.json` /
`06b_edge_holes.json`. The other distinct loss and cap boards were not read by eye.

---

## H1 — the Arm-B "loop": a one-ply win neither engine resolves (13 of 22 Arm-B caps)

**DEMONSTRATED.** In 13 distinct Arm-B capped games (13 games) and 1 Arm-B loss, there are
positions where **T1j (red) had an immediate win** by our rules and played elsewhere — 477
T1j positions in all, 411 in caps — and the **incumbent (black) left a single, defendable
interior hole open** in the same games (MISSED_DEFENCE, 414 positions: 347 in caps with the
reply available, 1 at a capped final ply with the reply unavailable, 66 in the one loss).
None of these incumbent misses was punished on the next ply. T1j's view of each queried
position — pegs, **bridges**, legal moves, terminal flags — equalled ours (the committed
position digests), and its search reported `completed_depth 6`, alpha-beta.

*Example* `h5s3-32-B`: ply 80, incumbent to move, board threat T = (17,7); saving moves
(16,6) and (17,7); it played (11,10). Ply 81, T1j to move, W = (17,7) — two new bridges,
(15,6)–(17,7) and (16,5)–(17,7), cross nothing — it searched (≈17 ms, depth 6) and played
(14,8). In this game T1j had an immediate win available at 100 of its turns before the
280-ply cap (whether always the same hole was not checked).

**HYPOTHESES.** (a) T1j's search does not find or does not choose this completion; the
mechanism (candidate generation, indifference among "won" lines, …) is unknown. (b) The
incumbent's readout does not prioritise blocking a single one-ply threat in these positions.
*Consequence, descriptive only:* these 13 games were scored as caps although, by our rules,
T1j had a win available; H5's result is unchanged and nothing is re-scored. *Test it:* a
carded replay of these exact positions through T1j and the incumbent (positions only).

## H2 — losses arrive without a one-ply warning, mostly as edge threats (SYMMETRIC)

**DEMONSTRATED.** In 45 of 46 distinct losses (A 6/6 = 103 games; B 39/40 = 115 games),
the incumbent's **first** threatened turn was already UNDEFENDABLE and the game ended on the
next ply. Every single-hole case (A 2, B 7) and most multi-hole cases (A 4/4; B 26/32) have
all winning holes on **T1j's own goal edge** — cells the incumbent may not occupy. Examples:
`h5s0-09-A` ply 39, T = (9,0),(13,0); `h5s0-06-A` ply 53, T = (4,23); `h5s1-01-B` ply 36,
T = (23,20).

**But it is symmetric:** T1j's losses end the same way — A 8/8 multi-hole undefendable
endings on the incumbent's edge; B 42/44 multi-hole and 10/10 single-hole on it
(`06b_edge_holes.json`). By D0's lesson this is a property of **losing at TwixT**, not of the
incumbent. **HYPOTHESIS:** the incumbent's decisive mistakes lie two or more plies before
the first threat (letting T1j reach a two-way edge connection). One-ply labels cannot show
it; a two-ply check would need its own card.

## H3 — Arm B: T1j's ladders toward its own edge

**HYPOTHESIS-SUPPORT (stated heuristic, `06_summary.json`).** A "ladder" = a run of one
side's knight-step moves on alternate plies, each answered by an adjacent opponent peg. In
Arm-B losses T1j ran a ladder of ≥ 3 steps in 16/40 distinct trajectories (51/116 games), and
its winning move extended it in 5/40 (20/116); in Arm-B incumbent wins the incumbent did so
in 9/63 (21/158) and 1/63 (4/158). Arm A: rare for both.

*Example* `h5s0-26-B` (×6): T1j's red pegs step (15,5) → … → (22,19) by knight moves during
plies ~9–25, each with an incumbent peg beside it; at ply 34 the single hole (23,17) on red's
goal row is undefendable; T1j completes it at ply 35.

**HYPOTHESIS:** as black, the incumbent follows T1j's ladders toward T1j's goal edge instead
of blocking ahead of them. Not demonstrated; the detector is heuristic and a ladder's
presence does not show it decided the game.

## H4 — T1j's one-ply defensive misses decide some incumbent wins

**DEMONSTRATED.** In 18 distinct incumbent wins (A 16 = 16 games; B 2 = 3 games), T1j's first
threatened turn was a **defendable single interior hole** it did not block; in 19
positions/trajectories the incumbent won on the very next ply. The incumbent had **no**
punished one-ply miss anywhere. Examples: `h5s0-13-A` ply 138, T = (5,6); `h5s2-57-B` ply 45,
T = (15,10). **Not established as an engine difference** (card §6): a descriptive count in
this sample, with the dependence of plies within a game.

## H5 — Arm-A caps are locked boards

**DEMONSTRATED (one-ply).** In all 20 Arm-A caps (20 distinct, frequency 1 each) neither
side ever had an immediate win at any ply. **HYPOTHESIS-SUPPORT:** in all 20, the incumbent's
(red) largest group touches only the bottom edge and T1j's largest only one of its edges; in
`h5s3-62-A` the final board shows 11 incumbent pegs spaced along its own top row (when they
were played was not checked).
**HYPOTHESIS:** once both sides are walled off, neither engine has a plan to break through,
and the incumbent spends moves filling its own region. 13 of 22 Arm-B caps are H1 instead;
the other 9 are one-sided blobs like these.

## Observation, not a cause

In the 6 Arm-A loss trajectories the incumbent's pegs on its own goal rows come late
(plies 19–61 of 40–66), after its structure was set — consequences of losing, not causes
(`06_summary.json`, `armA_loss_goal_row_pegs`).
