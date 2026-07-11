# v16a — Stratified, Game-Held-Out, Non-Selected FPU Collateral-Damage Screen (Design)

**Status:** Tooling + manifest SHIPPED (2026-07-10). The FPU validation *sweep* is an operator phase and is deliberately not executed by the implementation.

## Objective

First rung of the FPU validation ladder. The whole "Targeted Value Calibration" line established that gate A's post-opening sharp value drop was a **400-sim search artifact**, not a value-head defect, and that First-Play Urgency (`MCTSConfig.fpu_value`, byte-identical at `0.0`) is the direct lever; the frozen candidate `-0.20` was selected on the A discovery set. v16a asks one narrow question:

> Does the frozen `-0.20` candidate avoid obvious collateral damage on ordinary,
> game-held-out positions — strongly enough to justify proceeding toward B/C/D
> and the real equal-budget strength benchmark?

## Hierarchy (do not conflate)

- **Root goal:** a stronger 400-sim MCTS and better self-play search targets.
- **Decisive benchmark:** a statistically significant, same-checkpoint, equal-400-sim, balanced-color head-to-head strength gain (FPU off vs ship-form).
- **v16a:** *only* a held-out collateral-damage screen. Passing it is a necessary gate, **not** evidence of improvement. Matching the 6400-sim reference and passing these screens is mechanistic evidence, not the benchmark.

## Design decisions

- **Trusted path, unmodified.** The manifest conforms to the canonical `position_probe_cases` schema (`game_idx, case_id, replay_path, position_ply, side_to_move`), so it reconstructs through `load_csv_manifest → search_for_row → position_state` with no reconstruction change. The sweep already touches no A-specific case field except `case_id`.
- **Names.** Manifest uses canonical loader names; the sweep *output* uses the spec names (`game_id←game_idx`, `ply←position_ply`, `top_move←top_child_move`).
- **Mode auto-detected, strict.** A manifest is neutral iff *every* row carries `ply_bucket`; legacy iff none do; mixed/empty raise. Legacy output is byte-identical (golden + pre-branch diff proof).
- **Game-level holdout.** Entire games named in the A discovery manifest are excluded (sourced from the single `DEFAULT_A_MANIFEST` constant), so discovery and validation share zero games (verified: 0 overlap).
- **Mover-perspective primary.** FPU is pessimism for the player to move; black-perspective deltas cancel across colors and would falsely read as harmless. Every value delta is reported in both mover (`_stm_`, primary) and black (`_black_`, continuity) perspective, stratified overall / bucket / side / bucket×side.
- **Frozen protocol, enforced in code.** The tool accepts arbitrary `--fpu-values`, but a neutral manifest defaults to `0.0,-0.20` and rejects other value sets unless `--allow-non-protocol-fpu`.
- **Single top-share denominator.** Top-1 share and the `≥0.95` collapse flag both use `top.visit_count / root.visit_count` (continuity with the selected-A diagnostic); entropy normalizes over the child-visit total.
- **Paired search-shape deltas.** Entropy, effective-children (`exp(entropy)`), visited-children, reply count (all + stable-top), top-share are reported as change-vs-fpu-0, plus `new_collapse`/`resolved_collapse` accounting — breadth count alone can stay flat while concentration moves (the c_puct result).
- **Opening-prefix dedup** (not full board-state dedup): identical move prefixes reach identical states; transpositions are not merged.

## Measured manifest (source: `calib020_0001_vs_0379_800g_w4_seed20115_replay_games.jsonl`)

800 games → 30 A-games excluded → **770 held-out**. Sampled **324 rows**, seed 20260710, exact 162/162 red/black:

| bucket | plies | rows | note |
|---|---|---|---|
| opening | 1–15 | 40 | cap 1 + cross-game prefix dedup |
| early_mid | 16–40 | 100 | cap 2, min-gap 8 |
| midgame | 41–90 | 100 | cap 2, min-gap 8 |
| late | 91+ | 84 | cap 2, min-gap 8; SHORTFALL vs 100 (data-limited: only 48 eligible games; 47 rows come from the 19 `state_cap` 280-ply marathons kept as `game_result=unknown` — the most search-stressed samples). |

Late is intentionally coverage-first: 84 well-separated positions across 48 games beats a round 100 built from correlated plies. Outcomes: red 129 / black 148 / unknown 47.

## Protocol

```
Tool capability:      diagnose_fpu_sweep accepts arbitrary --fpu-values.
v16a validation run:  FROZEN control 0.0 vs candidate -0.20 ONLY (enforced in
                      code: neutral default = 0.0,-0.20; others need
                      --allow-non-protocol-fpu). Screening extra candidates on
                      the holdout is tuning on the holdout.

The held-out MCTS sweep is an operator-run phase and is deliberately not
specified or executed by this implementation plan.
```

## Pre-registered decision table

Confirm BEFORE running the sweep; never after seeing results. Candidate `-0.20` vs control `0.0`, over the held-out set.

**Statistical rule:** only strata with `num_positions ≥ 20` participate in any "stratum" pass/fail; smaller strata are descriptive, inspection-only. **Narrowing is the intended mechanism** — effective-children and reply-count reductions are not failures on their own; they matter only alongside new collapse, extreme concentration, or broad value disruption.

**AUTOMATIC REJECT** if any:
- overall `new_collapse_rate ≥ 0.05`, OR
- any n≥20 stratum `new_collapse_rate ≥ 0.10`, OR
- overall `median_abs_root_value_delta_stm_vs_fpu0 ≥ 0.20`, OR
- overall `p95_abs_root_value_delta_stm_vs_fpu0 ≥ 0.60`, OR
- effective-children reduction ≥ 50% **AND** overall `mean_top_child_visit_share_delta_vs_fpu0 ≥ +0.15` [reduction = `-mean_root_effective_children_delta_vs_fpu0 / mean_root_effective_children(@0.0)`], OR
- integrity, pairing, deterministic-reproduction, or manifest-holdout failure.

**MANDATORY INSPECTION** (case-level review; not auto-reject) if any:
- `new_collapse_rate > 0` but `< 0.05`, OR
- overall `top_move_flip_rate_vs_fpu0 ≥ 0.25`, OR
- any n≥20 stratum `top_move_flip_rate_vs_fpu0 ≥ 0.35`, OR
- effective-children reduction ≥ 30%, OR
- any n≥20 stratum `|mean_root_value_delta_stm_vs_fpu0| ≥ 0.10`, OR
- overall `p95_abs_root_value_delta_stm_vs_fpu0 ≥ 0.35`, OR
- overall `mean_top_child_visit_share_delta_vs_fpu0 ≥ 0.10`, OR
- stable-top opponent-reply reduction ≥ 50% [`= -mean_top_child_children_delta_stable_top_vs_fpu0 / mean_top_child_n_visited_children(@0.0)`].

**SAFE-TO-ADVANCE** (to B/C/D + the equal-budget strength match) only if ALL hold:
- integrity + paired-row checks pass, AND
- `new_collapse_count == 0`, AND
- overall `top_move_flip_rate_vs_fpu0 < 0.25`, AND
- no n≥20 stratum `top_move_flip_rate_vs_fpu0 ≥ 0.35`, AND
- overall `median_abs_root_value_delta_stm_vs_fpu0 < 0.10`, AND
- overall `p95_abs_root_value_delta_stm_vs_fpu0 < 0.35`, AND
- no n≥20 stratum `|mean_root_value_delta_stm_vs_fpu0| ≥ 0.10`, AND
- overall `mean_top_child_visit_share_delta_vs_fpu0 < 0.10`, AND
- no n≥20 stratum `mean_top_child_visit_share_delta_vs_fpu0 ≥ 0.15`.

Do NOT require any effective-children or reply-count reduction to advance; narrowing is the mechanism and is acceptable absent collapse, extreme concentration, or broad value disruption.

"Advance" means "no obvious collateral damage," NOT "improvement." Improvement is decided only by the equal-budget, balanced-color, statistically-significant head-to-head strength match.
