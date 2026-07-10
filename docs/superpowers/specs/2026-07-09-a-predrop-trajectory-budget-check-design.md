# A Pre-Drop Trajectory Budget Check (Design)

**Status:** APPROVED (2026-07-09). Read-only diagnostic. No training, no manifest, no trainer/network/MCTS/loader change.

## 0. The question

The Targeted Value Calibration line (calib010 → v14d) selected its A cases from positions where a **400-sim** replay `root_value` was high and then collapsed ("post-opening sharp value drop"). Phase 0.5 and the operator's budget sweep showed that this statistic is not a value-head property:

- `root.q_value` is the unweighted mean of the raw leaf evaluations backed up through it (validated: mean `raw_black` over 5745 expanded nodes = +0.598 vs measured searched +0.619). `eval_position_probe.py:77,85` makes `probe_black_root_value = to_black(root.q_value)`.
- At 400 sims against ~500 legal moves, the expansion frontier is **89.2% black-to-move depth-2 nodes**, of which 4333/4443 are single-visit. `_select_child` (`mcts.py:955`) hardcodes `q = 0.0` for unvisited children, so the opponent scans ~350 distinct replies once each and never revisits. Those single-visit nodes average `raw_black = +0.803` (99.4% positive) — they are blunders after which black really is winning. The PV reads **−0.207**.
- Budget sweep on BASE over the 30 A roots: mean **+0.2570 (400) → +0.0626 (1600) → −0.0451 (6400)**; gate `over (≥0.25)` **50% → 30% → 10%**; `severe (≥0.50)` **43.3% → 6.7% → 3.3%**.
- `root.q_value` **trains nothing**: the value target is `z`, the game outcome from to-move's POV (`trainer.py:730`). Its only consumers are the gate metric and the per-move `root_value` in replay JSONs, which `eval_loss_replay_analysis.py:134,144` read to select the A cases.

So the phenomenon the whole line has chased was *defined by* the statistic now known to be dominated by single-visit frontier noise. **This diagnostic asks whether the drop exists at all under adequate search**, before any search-config grid (v16) or any further model work.

## 1. Scope

**In scope:** one read-only script that, for 5 representative A loss games, re-searches a 6-ply window spanning `predrop_ply → drop_ply` at 400 and 6400 sims, and records the root value trajectory plus the search's own recommendation at each ply.

**Out of scope:** the v16 search-config grid; any `MCTSConfig`/`mcts.py` change (including FPU, `prior_top_k`, `prior_min_mass`); any manifest, replay JSON, trainer, network, loader, adapter, or projection change; any full tree/raw-frontier walk (Phase 0.5 already did that for the roots). Promotion rules unchanged; the promotion opponent remains `calib020_0001`.

## 2. Cases and window

Five cases, chosen for coverage rather than convenience: `black_loss_game_000281_predrop_ply_19_drop_21`, `..._000259_predrop_ply_35_drop_37`, `..._000127_predrop_ply_33_drop_35` (the three most overvalued roots), `..._000611_predrop_ply_19_drop_21` (the least single-child-dominated root), and `..._000347_predrop_ply_73_drop_75` (a late-ply case, and Phase 0's sign-sanity root).

`drop_ply` is a **column** in the A probe manifest — no case-id parsing. For all five, `drop_ply == predrop_ply + 2`, and every window ply is black-to-move (red moves at even 0-indexed plies).

Window = `{predrop−4, predrop−2, predrop} ∪ {drop, drop+2, drop+4}`, sorted, deduplicated, and clipped to `0 <= ply < len(replay["moves"])`. Clipping removes exactly one position (game 347, ply 79 of 79). **29 positions × 2 budgets = 58 searches** (≈197k sims).

## 3. Method

Per (case, ply, budget): reconstruct the board with `position_state`, seed with `row_seed(CORRECTION_TAG, game_idx, ply, pos_base_seed=20260616, goal_base_seed=20260614)` — the gate's own seed — and run `MCTS(...).search_with_root(state, add_noise=False)` under `EvalConfig(mcts_sims=<budget>, mcts_eval_batch_size=14, mcts_stall_flush_sims=48)` with `_default_evaluator_factory` (train-mode BN, `compile=True`).

**One evaluator is built and reused across both budgets.** Only the `MCTSConfig` differs. This is closer to the gate than rebuilding an evaluator per budget, and it avoids the known MLX `compile=True` sequential-eval gotcha.

`side_to_move` is derived from ply parity (`red` if `ply % 2 == 0` else `black`) and independently asserted by `position_state`, which raises on disagreement.

**Perspective.** `root_black_value = to_black(root_value_stm, side)`. `top_child_q_black = to_black(child.q_value, child.state.to_move)` — the child's own to-move, never the root's side.

**Free integrity check.** The replay JSON stores the original 400-sim `root_value` per move (side-to-move perspective) — the very numbers that defined the drop. The script records `replay_stored_root_black_value = to_black(moves[ply]["root_value"], side)` alongside the fresh 400-sim value and prints the max absolute difference across all rows. Agreement confirms the seed and perspective are right and that the 6400 curve is comparable. This is reported loudly, not asserted: the replay's search used the game's own rng, so rare tie-breaks may differ.

## 4. Output

`logs/eval/v16_a_predrop_trajectory_budget.csv`, one row per (case, ply, budget):

```
case_id, replay_path, budget_sims, ply, predrop_ply, drop_ply,
relative_to_predrop, relative_to_drop, side_to_move,
root_black_value, replay_stored_root_black_value,
top_move, top_visit_share, top_child_q_black, pv_path
```

`logs/eval/v16_a_predrop_trajectory_budget_summary.csv`, one row per (case, budget):

```
case_id, budget_sims, n_pre, n_post, pre_drop_mean, post_drop_mean,
drop_delta, max_pre_drop_value, ply_of_max_pre_drop
```

`pre` is `ply < drop_ply`; `post` is `ply >= drop_ply`; `drop_delta = post_drop_mean − pre_drop_mean`.

## 5. Decision table (the operator's)

| Read | Meaning | Next |
|---|---|---|
| 400 shows the pre-drop spike and collapse; **6400 is flat or already ≤ 0 before the drop** | The A "drop" is a shallow-search artifact | Stop value calibration against this signal. The only open question is whether 400 sims is too shallow for the gate and for self-play |
| Both 400 and 6400 show the same pre-drop spike and collapse | The phenomenon survives deeper search | A search-config (v16) or data/value investigation is justified |
| 6400 reduces severity but preserves a smaller drop | Partly real, partly shallow-search amplified | Search-config testing justified; value calibration still needs caution |

## 6. Note for any future summarizer

The gate's thresholds are `OVERVALUE_THRESHOLD = 0.25` and `SEVERE_OVERVALUE_THRESHOLD = 0.50` (`position_probe_cases.py:18-19`):

```python
over_pct = sum(v >= 0.25 for v in vals) / len(vals) * 100.0
severe_pct = sum(v >= 0.50 for v in vals) / len(vals) * 100.0
```

An earlier ad-hoc summarizer used `v > 0` for `over`, which made the gate's overvalue rate look flat at 50% across budgets when it in fact collapses 50% → 30% → 10%.
