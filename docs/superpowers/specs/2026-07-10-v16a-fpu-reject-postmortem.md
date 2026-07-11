# v16a — FPU `-0.20` REJECT + New-Collapse Postmortem (FROZEN)

**Status:** FINAL, read-only. The v16a held-out manifest and all outputs are frozen — they must **not** be used to choose the next FPU value or formula. Run date 2026-07-10 (operator).

## 1. Frozen v16a result

Candidate: absolute `MCTSConfig.fpu_value = -0.20` vs control `0.0`, held-out manifest (324 positions, 252 games, 0 shared with selected-A discovery).

| Metric | Value | Classification |
|---|---:|---|
| Overall new-collapse rate | 15/324 = **4.63%** | below 5% overall |
| **Late new-collapse rate** | **13/84 = 15.48%** | **preregistered AUTOMATIC REJECT** (n≥20 stratum ≥10%) |
| Late-black / late-red new-collapse | 16.67% (n=42) / 14.29% (n=42) | both independently reject |
| Top-move flip rate | 27.16% | inspection |
| Effective children | 107.58 → **70.92** (−34.1%) | inspection |
| Median \|mover-value Δ\| / p95 | 0.018 / 0.282 | safe |
| Mean top-share increase | +0.072 | safe |

**Verdict: REJECT `-0.20` as a global 400-sim / self-play setting.** All gate numbers were independently reproduced from `neutral_fpu_sweep_cases.csv` (they match the tool's strata summary to the digit). Absolute `-0.20` *does* reach the selected-A first-touch mechanism (validated separately) but does **not** generalize safely.

Outputs (frozen): `logs/eval/v16a_fpu_unbiased/neutral_fpu_sweep_{cases,summary,by_stratum}.csv`, `operator_sweep.log`, and the per-case postmortem `v16a_new_collapse_postmortem.csv`.

## 2. New-collapse postmortem (15 cases, read-only)

Failure is **not** value distortion (mover-value shift controlled). It is **late-game / high-branching search concentration**: late collapsed roots go 17→28 of 84 (+11 net; 13 new, 2 resolved). Reconstruction validated (replay `n_legal` == reconstructed for all 15). Answers to the four questions:

**Q1 — Mostly 280-ply marathons?** Over-represented but not exclusive. 13/15 collapses are late (2 are early-mid at ply 25/30 with ~500 legal moves); of the 13 late, **9 are the 280-ply `state_cap` marathons** (69% vs 45% marathon prevalence in the late bucket). The rest are ordinary finished games.

**Q2 — Already concentrated at FPU 0.0?** No. **14/15** had baseline top-share **< 0.85** (real jumps, not trivial 0.95-line crossings); only 1 (early-mid `g622 ply25`) sat in the 0.85–0.94 band, and that one is benign (see Q3). Most collapses jump from near-uniform (baseline effective children 16–204) or moderately narrow (2–5) straight to a single line.

**Q3 — Same move harder, or a new move? (decisive)** 10/15 (67%) **change** the root's chosen move. The tell is the *prior of the move it collapses onto*: **12/15 collapse onto a network-dispreferred move** (prior rank #16–#207; median selected-move prior **0.0032**). Only **3/15** collapse onto the net's **#1-prior** move (`g288 ply92`, `g483 ply30`, `g622 ply25`) — those are benign "commit to the obvious move." So in the large majority, `-0.20` commits the search to a move chosen by **exploration order**, not by policy or search.

**Q4 — Late geometry, or a broader absolute-FPU flaw?** A broader flaw of the *absolute* floor, manifesting through search geometry. All 15 collapse roots are near-even (candidate \|mover value\| ≤ 0.28, median 0.03) with high branching (n_legal 281–503, median 351). When priors are flat and the position is ≈even, an absolute `-0.20` floor sits right at the node's value, so once the first explored child clears `-0.20` every unexplored alternative looks worse and the search stops comparing. These conditions cluster late (marathons stay high-branching and unresolved) but also appear at the 2 early-mid ~500-legal positions.

## 3. Implication (direction only — not a design)

The evidence points at replacing the **absolute assumed value** with a **context-relative** rule — parent-relative FPU reduction and/or prior/branching-aware reduction — because the failure is specifically the fixed floor dominating flat-prior, near-even, high-branching nodes. **This must be designed on selected-A / discovery data or a separate development corpus, never on v16a.** Prefer one interpretable, default-off, byte-identical-when-off change over a broad multi-parameter formula. Then: verify it still suppresses selected-A first-touch scanning, confirm it does not reproduce this late collapse in another form, freeze it, and validate on a **fresh** game-held-out corpus (not v16a) under this same preregistered decision table — before B/C/D and the decisive equal-budget, balanced-color, statistically-significant strength match.
