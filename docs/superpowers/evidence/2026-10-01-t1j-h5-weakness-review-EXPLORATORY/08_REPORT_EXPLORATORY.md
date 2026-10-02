# EXPLORATORY — H5 post-hoc weakness review: report

🔴 **EXPLORATORY throughout.** This review does not amend, qualify, re-test or add to H5's
confirmatory claim (directional replication, mean 0.5946, nominal 95% [0.5157, 0.6735],
`8ca7278`) or to H4's. No H4–H5 pooling, no interval, test or p-value, no model change or
strength run. "Ours" = the incumbent `calib020_0001`; a **loss** is an incumbent loss; a
**cap** is unresolved — never a loss or a win.

## 1. EXPLORATORY — what ran

* **Card:** `docs/superpowers/2026-10-01-t1j-h5-weakness-review-card.md` (`9693e4f`) with
  §7 clarifications C1–C3 (`97fd909`), committed before any diagnostic.
* **Code:** `scripts/GPU/alphazero/h5_weakness_review.py` + 18 synthetic controls
  (`d8d82aa`; 9/9 injected defects rejected in a disposable worktree).
* **Run:** once, 2026-10-02 03:15:42Z → 03:23:58Z, exit 0, on `d8d82aa`. Integrity: all six
  inputs hash to their pins (`00_inputs.json`), durable, re-derived by `h4_runner.load_games`,
  bound to the manifest schedules, and the per-arm outcomes, cap games, cap-affected pairs and
  unique games by arm reproduce the committed estimate exactly.
* **Timing (C3, frozen):** sample = shortest (31 plies, 0.15 s), median-length (58, 0.29 s),
  longest (280, 12.7 s); projection 739 s for N = 20,786 positions ≤ 7,200 s → the full run
  started and finished: **177/177 distinct trajectories, every pre-move position, 462 s**. No
  sampling, no pruning; the exact-reduction subset assertion held on every played move.

## 2. EXPLORATORY — outcome inventory (`01_inventory.json`)

| arm | incumbent | games | win / loss / cap (games) | distinct trajectories | win / loss / cap (distinct) |
|---|---|---:|---|---:|---|
| A | red (moves first) | 296 | 173 / 103 / 20 | 52 | 26 / 6 / 20 |
| B | black | 296 | 158 / 116 / 22 | 125 | 63 / 40 / 22 |

Arm A is highly concentrated: its 103 losses are **6** distinct trajectories (frequencies
21, 21, 20, 16, 14, 11); every Arm-A cap is distinct.

## 3. EXPLORATORY — Q1: opening and response patterns in losses (`02_prefix_tables.json`)

A prefix **recurs in losses** at ≥ 2 distinct loss trajectories (card §4 Step 2).

* **Arm A:** no prefix of 4–12 plies recurs; each of the 6 loss trajectories has its own
  4-ply opening. At depth 2 only `18,19 9,16` recurs — 31 losses / 21 wins (games), 2 / 1
  (distinct). **The recurrence in Arm A is the whole game**: each losing game is repeated
  11–21 times identically (deterministic incumbent).
* **Arm B:** 12 depth-2 prefixes recur (T1j's first move from the central 6×6, plus the
  incumbent's fixed reply); none at depth ≥ 4. The most lopsided, as win / loss / cap
  (games): `13,13 10,12` 0/11/3, `12,9 8,9` 0/10/0, `9,13 13,13` 0/10/7, `12,10 8,11` 3/8/0,
  `14,12 10,11` 1/8/0, `11,14 8,13` 1/7/0 (full table: `06_summary.json`,
  `armB_depth2_outcomes`). Small cells; descriptive only.

## 4. EXPLORATORY — Q2: demonstrated one-ply errors vs hypotheses (`03_mechanical_checks.json`)

**Rates (C2)** — hits / inspected pre-move positions, by arm and engine; occurrence-weighted
(each game) and distinct (each trajectory once). Witnesses (winning / saving moves, threats
left, reply availability) are persisted per hit.

| arm / engine | view | MISSED_WIN | MISSED_DEFENCE | UNDEFENDABLE | DEFENDED |
|---|---|---|---|---|---|
| A / incumbent | occurrence | 86/11317 | 0/11317 | 103/11317 | 0/11317 |
| A / incumbent | distinct | 6/4642 | 0/4642 | 6/4642 | 0/4642 |
| A / T1j | occurrence | 0/11144 | 18/11144 | 241/11144 | 75/11144 |
| A / T1j | distinct | 0/4616 | 18/4616 | 14/4616 | 4/4616 |
| B / incumbent | occurrence | 3/9903 | 414/9903 | 180/9903 | 15/9903 |
| B / incumbent | distinct | 1/5744 | 414/5744 | 104/5744 | 15/5744 |
| B / T1j | occurrence | 477/10019 | 4/10019 | 157/10019 | 40/10019 |
| B / T1j | distinct | 477/5784 | 3/5784 | 61/5784 | 18/5784 |

No label restates the ending (no label's hits sit at a single distance from the terminal
ply). Positions within a game are not independent, so the per-trajectory view
(`06_summary.json`) matters more than the position rates:

* **The incumbent's one-ply misses were never punished.** Its MISSED_WINs (A 2 trajectories
  / 22 games, B 1 / 3) are all in games it won anyway; its MISSED_DEFENCEs are the Arm-B loop
  (H1) — 0 of them was followed by a T1j win on the next ply.
* **T1j's were.** In 19 distinct incumbent wins T1j missed a defendable one-ply block and lost
  on the very next ply (H4) — always a single **interior** hole.
* **Every incumbent loss but one arrives without a one-ply warning** (45/46 distinct): the
  first threatened turn is already UNDEFENDABLE, usually with the winning holes on T1j's own
  goal edge. **This is symmetric** — T1j's losses end the same way on the incumbent's edge
  (`06b_edge_holes.json`) — so, as D0 found for fragmentation, it is a property of losing at
  TwixT, not of our model (H2).

## 5. EXPLORATORY — Q3: what produces caps (`04_caps.json`, `06_summary.json`)

* **Arm B, 13 of 22 caps — the loop (H1):** T1j (red) had an immediate win by our rules at
  many of its turns and played elsewhere (411 positions in caps), while the incumbent left the
  same kind of single interior hole unblocked (347 positions with the reply available, 1 at a
  capped final ply with it unavailable). T1j's board — bridges included — matched ours at
  every query, and its search completed to depth 6.
* **Arm A, all 20 caps, and Arm B's other 9 — locked boards (H5):** no immediate win for
  either side at any ply (Arm A); in every cap the incumbent's largest group touches only one
  of its goal edges (heuristic profile).
* No two capped trajectories share an exact final-feature profile; prefixes shared by ≥ 2
  distinct capped trajectories exist at every depth (A 1, B 5).

## 6. EXPLORATORY — hypotheses (`07_hypotheses_EXPLORATORY.md`)

H1 the Arm-B loop (T1j does not complete; the incumbent does not block) · H2 losses come
from multi-ply build-ups, symmetric between engines · H3 Arm-B ladders toward T1j's edge
(heuristic) · H4 T1j's punished one-ply defensive misses · H5 locked Arm-A caps. Each states
what is demonstrated and what is not, with files and plies; none is a finding.

## 7. EXPLORATORY — what this adds beyond D0

D0 (L0 match, scripted 6-ply openings, 32 games, no caps) found no signature asymmetric
against us. This review, on 592 empty-board games: confirms D0's symmetry lesson for the way
games end (H2); finds that **no incumbent loss involves a punished one-ply miss**, while 19
incumbent wins do involve one by T1j (H4); and identifies something D0 could not see, because
D0 had no caps — **13 Arm-B games capped although T1j had an immediate win available by our
rules** (H1). The defence label requires a saving move to have existed, which D0's
`ignored_threat` did not.

## 8. EXPLORATORY — limitations

* One-ply only: nothing two or more plies deep is demonstrated (H2, H3 are hypotheses).
* Inspection: every distinct loss and cap trajectory was stepped through programmatically;
  by eye I read 10 final boards (all 6 Arm-A losses, 3 Arm-B losses — one only to row 17 —
  and 1 Arm-A cap to row 21) and one board region of `h5s3-32-B` at ply 80, not all 88.
* Heuristics (ladder detector, goal-touch profile) are stated in code and support hypotheses
  only. Cells are small; Arm A is dominated by six repeated games.
* T1j's internals were not examined (no JVM authorized); why its search skips completions is
  unknown. The live-browser replay check remains unavailable (Chrome not connected).
* No engine difference is established (card §6); counts are descriptive.

## 9. EXPLORATORY — files

`00_inputs.json` · `01_inventory.json` · `02_prefix_tables.json` · `03a_timing.json` ·
`03_mechanical_checks.json` · `04_caps.json` · `05_inspection_notes.md` ·
`06_summarize.py` → `06_summary.json` · `06b_edge_holes.py` → `06b_edge_holes.json` ·
`07_hypotheses_EXPLORATORY.md` · this report.
