# Targeted Value Calibration v15 — Phase 0.5: Selected-Branch Subtree Diagnostic (Design)

**Status:** APPROVED (2026-07-09). Read-only diagnostic. No manifest, no training, no loader/trainer/network/MCTS change.

**Predecessor:** Phase 0 (`docs/superpowers/plans/2026-07-09-targeted-value-calibration-v15-phase0-concentration-diagnostic.md`, shipped @ `a780ada`).
**Parent spec:** `docs/superpowers/specs/2026-07-09-targeted-value-calibration-v15-a-searched-continuation-correction-design.md`.

---

## 0. Why this exists — the finding that stopped Phase 1

Phase 0 answered its question: the optimistic value backup at the A roots **is** concentrated. All 17 roots with `root_mcts_black_value > 0` (the roots that produce the A gate's +0.257) classify as *concentrated*; the top-3 children carry **98.1%** of all positive backup mass globally. Under the locked selection rule (cumulative positive share ≥ 0.90, max 3 children) that is **17 roots / 27 depth-1 child rows** — 21 at target −0.35, 6 at −0.20 under the parent spec's tiering.

Phase 0 also surfaced, unplanned, the reason those 27 rows should **not** be built yet.

At the top-1 positive child of each overvaluing root, the child's own **raw** black value under BASE is already **−0.087** on average, while its **searched** black value (`−child_q_value`) is **+0.619**. The +0.706 gap is produced by search *below* the depth-1 child.

The MCTS arithmetic makes the consequence concrete. A node's `q_value` is the mean of values backed up through its subtree, and the node's own raw NN evaluation enters exactly once, at expansion. These top children carry `visit_share` 0.9–0.97, i.e. ~350–390 visits. Correcting a child's raw value by Δ therefore moves its backed-up `q` by ≈ Δ/370. Training a depth-1 child from −0.087 to −0.35 shifts its searched value by roughly **0.0007**.

So a depth-1 correction can only help through **generalization** — the value head, taught to output −0.35 on those child boards, must also output lower values on the deep leaf boards inside those subtrees. That is precisely the assumption that failed for root correction across v2–v14, relocated one ply down. Phase 0 neither confirmed nor refuted it.

**Phase 0.5 tests that assumption directly, before any rows are built.** It asks: *where, in the subtree that produces the +0.7, does raw optimism first appear — and is it concentrated on the principal variation or spread across the frontier?*

---

## 1. Scope

**In scope:** one new read-only script that re-runs the deterministic BASE search on the 17 overvaluing A roots, walks the full expanded subtree beneath each selected positive branch, and emits per-node raw/searched values plus a by-depth summary.

**Out of scope, explicitly:** `build_v15_a_continuation_correction_manifest.py`; child replay JSONs; any training manifest; any v15 training command; any change to `calibration_pool.py`, `trainer.py`, `network.py`, `mcts.py`, `continuation_extraction.py`, `probe_eval.py`, `eval_runner.py`, or any manifest/builder. No new loss mode, no loader support for `hard_value` + `extra_moves_json`, no projection code, no verifier change.

The one exception to "touch nothing" is a **behavior-preserving extract-function patch** to the Phase-0 script (§4), so Phase 0.5 reuses the exact search and perspective code Phase 0 validated rather than copying it.

---

## 2. Architecture

One file is modified and two are created:

| File | Role |
|---|---|
| `scripts/GPU/alphazero/diagnose_v15_a_continuation_concentration.py` (modify) | extract two helpers out of `main()`; no behavior change, no CSV schema change |
| `scripts/GPU/alphazero/diagnose_v15_a_selected_branch_subtrees.py` (create) | Phase 0.5: selected-branch subtree walk, PV annotation, by-depth aggregates |
| `tests/test_v15_selected_branch_subtrees.py` (create) | pure logic on synthetic `MCTSNode` trees; no checkpoints, no MCTS |

Flow:

1. Load the Phase-0 CSV (`logs/eval/v15prep_a_continuation_concentration.csv`).
2. Keep roots with `root_mcts_black_value > 0` → 17 roots.
3. `select_positive_branches`: per root, take children in descending `positive_contribution_share` until cumulative ≥ 0.90 or 3 children taken → 27 branches.
4. Reconstruct each root from the A probe manifest and re-run the **deterministic** gate-faithful BASE search (same `row_seed`).
5. Fail-loud integrity checks (§5).
6. For each selected branch, walk **every** descendant with `visit_count ≥ 1`. Annotate the best-child chain from the branch root as the PV.
7. Score each walked node's raw value under BASE and v14b (eval-mode evaluators).
8. Write the per-node CSV and the by-depth summary CSV; print the summary.

**No depth cap.** Each of the 400 simulations expands exactly one leaf, so the entire tree per root is bounded at ~401 expanded nodes and the selected branches are a subset. An arbitrary cutoff would truncate exactly the deep single-visit frontier where optimism most plausibly hides.

---

## 3. Interfaces (verified against source — use exactly)

Imported from the Phase-0 module (`scripts.GPU.alphazero.diagnose_v15_a_continuation_concentration`):

- `per_child_metrics(root) -> list[dict]` — sign-verified in Phase 0; reused for the contribution invariant.
- `classify_concentration(metrics, top_n=3) -> (label, share)`.
- `search_for_row(row, search_fn, *, pos_base_seed, goal_base_seed) -> (state, side, root_value_stm, root)` — **new, extracted in §4**.
- `raw_black_value(state, evaluator) -> float` — **new, extracted in §4**. Equals `to_black(_teacher_infer(state, evaluator)[2], state.to_move)`.
- `_real_search_fn(base_checkpoint, sims, eval_batch_size, stall_flush_sims)` and `_build_raw_evaluator(checkpoint_path)`.

Copied (not imported), because `continuation_extraction.extract_continuations` raises for the A tag and Phase 0 set the precedent of copying its generic helpers:

- `_best_child(node)` — max-visit child, ties broken by lowest encoded move id (`continuation_extraction.py:48`).

MCTS node facts, verified in source:

- `_backup` (`mcts.py:989-1003`) applies exactly **one sign flip per level** on the way up; `TwixtState.apply_move` flips `to_move` every ply. Therefore each node's `value_sum` — and so `q_value` — accumulates in **that node's own to-move perspective**. This is verified, not assumed; nonetheless the CSV carries both a node-perspective and a root-perspective column, and **all analysis uses the root-perspective column**.
- `root.visit_count == Σ child.visit_count` for a `search_with_root` tree, because the root is pre-expanded outside the simulation loop and that expansion value is discarded.
- `node.children` is `Dict[move_id, MCTSNode]`; children are created at parent expansion, so a created child may have `visit_count == 0` and no NN evaluation.

---

## 4. Preparatory patch to the Phase-0 script (behavior-preserving)

Phase 0's `main()` currently inlines the reconstruct-plus-seeded-search block and the raw-value forward. Extract both, unchanged:

```python
def search_for_row(row, search_fn, *, pos_base_seed, goal_base_seed):
    """Reconstruct an A probe row's root state and run the seeded gate-faithful
    search. Returns (state, side, root_value_stm, root)."""

def raw_black_value(state, evaluator) -> float:
    """Raw (non-MCTS) value at `state`, converted to BLACK's perspective."""
```

`main()` then calls them. Constraints: no CSV schema change, no flag change, no behavior change. Verified by the existing five Phase-0 tests plus a `--limit-cases 1` re-run whose rows must match the committed CSV's first root exactly.

Deliberately **not** extracted: `load_phase0_rows`, `group_phase0_by_root`, `select_positive_branches`. These are Phase-0.5 concepts (the 0.90 / max-3 rule originates in the Phase-1 sketch, not Phase 0) and live in the new script.

---

## 5. Integrity checks — all fail-loud

The diagnostic drives a build/no-build decision, so a silent sign or reproduction error is the dominant risk. Three checks, each raising with a loud message:

1. **Tree reproduction.** The fresh `root_mcts_black_value` must equal the Phase-0 CSV's value for that root within `1e-6`. Proves the re-run search reproduced Phase 0's tree. (Same pattern as the v6 builder's source-root cross-check.)
2. **Contribution invariant.** `sum(m["child_contribution_share"] for m in per_child_metrics(root))` must equal `root.q_value` within `1e-6`, on every root — Phase 0's assert, reused.
3. **Cross-CSV perspective tie.** For every depth-1 node, `q_value_root_perspective` must equal `−child_q_value` as recorded in the Phase-0 CSV, within `1e-6`. This ties the two CSVs together and catches a perspective regression in either.

Additionally, every A root must have `side_to_move == "black"`; assert and fail loud otherwise. This is what makes "root perspective" and "black perspective" the same thing throughout.

---

## 6. Per-node CSV

`logs/eval/v15prep_a_selected_branch_subtrees.csv`, one row per expanded node (`visit_count ≥ 1`) beneath a selected branch, including the branch root (the depth-1 child) itself.

```
root_case_id, root_mcts_black_value, root_case_classification,
branch_rank, root_child_move, root_child_positive_contribution_share,
depth_from_root, depth_from_selected_child, path_moves, move_from_parent,
visit_count, visit_share_from_parent, visit_share_from_root,
q_value_node_perspective, q_value_root_perspective,
raw_black_BASE, raw_black_v14b, raw_delta_v14b_minus_BASE,
raw_positive_BASE, raw_positive_v14b,
is_pv_path, pv_depth_index, num_children, unvisited_children_count, is_terminal
```

Definitions:

- `q_value_node_perspective` — `node.q_value` as stored by MCTS for that node.
- `q_value_root_perspective` — `to_black(node.q_value, node.state.to_move)`. **The analysis column.**
- `raw_black_BASE` / `raw_black_v14b` — `raw_black_value(node.state, evaluator)` on the eval-mode BASE and v14b evaluators.
- `raw_delta_v14b_minus_BASE` — `raw_black_v14b − raw_black_BASE`.
- `raw_positive_BASE` / `raw_positive_v14b` — booleans, `raw_black_* > 0`.
- `visit_share_from_parent` — `node.visit_count / node.parent.visit_count`.
- `visit_share_from_root` — `node.visit_count / root.visit_count`.
- `branch_rank` — 1-based rank of the selected child by `positive_contribution_share`.
- `is_pv_path` — node lies on the best-child chain rooted at the **selected child**.
- `pv_depth_index` — index along that chain (branch root = 0); blank off-PV.
- `num_children` — count of created children, including `visit_count == 0` ones.
- `path_moves` — `(r,c)` moves from the **root** to this node.
- `root_case_classification` — carried through verbatim from the Phase-0 CSV, so the file is self-contained when reopened later. Every selected root is `concentrated` by construction (all 17 `root_mcts_black_value > 0` roots classified that way); the column is provenance, not a signal.
- `unvisited_children_count` — created children of this node with `visit_count == 0`. Per-node because the by-depth summary (§7) aggregates it.
- `is_terminal` — `node.state.is_terminal()`.

**Terminal nodes.** A subtree walk reaches terminal nodes; Phase 0 never could, because the depth-1 children of a midgame root are never terminal. A terminal node is visited but never expanded, has no legal moves, and has no NN evaluation — `_teacher_infer` would build zero-width arrays on it. Such nodes are emitted with `is_terminal = True`, `num_children = 0`, their real `visit_count` and `q_value_*`, and **blank** `raw_black_BASE` / `raw_black_v14b` / `raw_delta_v14b_minus_BASE` / `raw_positive_*`. The by-depth summary excludes blank-raw nodes from every `raw_*` statistic (they have no raw value to average) but still counts them in `nodes_count` and `total_visit_share_from_root`, so visit mass is never silently dropped. If terminal nodes carry non-trivial visit mass at some depth, that itself is a finding — the summary's `nodes_count` minus the raw-scored count makes it visible.

---

## 7. By-depth summary

`logs/eval/v15prep_a_selected_branch_subtrees_by_depth_summary.csv`, also printed. Grouped by `depth_from_root`, emitted twice: once over the **full subtree**, once restricted to **PV nodes only**. The full-subtree-versus-PV comparison at each depth is the read.

Per group, for BASE and for v14b:

```
scope (full_subtree | pv_only), depth_from_root,
nodes_count, raw_scored_nodes_count, unvisited_children_count,
total_visit_share_from_root,
mean_raw_black, weighted_mean_raw_black,
pct_raw_positive, pct_visit_mass_raw_positive,
max_raw_black
```

(the four `raw_*` columns and `mean/weighted_mean/max` are each emitted twice, suffixed `_BASE` and `_v14b`)

- `nodes_count` — walked nodes (`visit_count ≥ 1`) at that depth, within scope.
- `raw_scored_nodes_count` — walked nodes at that depth that carry a raw value (i.e. non-terminal). Every `raw_*` statistic below is computed over these only; `nodes_count − raw_scored_nodes_count` is the terminal count.
- `unvisited_children_count` — created-but-never-visited children **of the nodes at that depth**; measures how much of the branching the search declined to explore. Never scored (no NN value exists for them).
- `total_visit_share_from_root` — summed `visit_share_from_root` over the walked nodes at that depth.
- `weighted_mean_raw_black` weights by `visit_share_from_root`.
- `pct_raw_positive` — fraction of walked nodes at that depth with `raw_black > 0`.
- `pct_visit_mass_raw_positive` — fraction of that depth's `total_visit_share_from_root` held by raw-positive nodes.

**`pct_visit_mass_raw_positive` is the decision number, not `pct_raw_positive`.** A thousand single-visit frontier leaves must not outvote one 300-visit node. `pct_raw_positive` is retained as context only.

---

## 8. Decision table (the operator's, after the run)

| Read | Meaning | Next |
|---|---|---|
| Raw rises positive with depth **and** most raw-positive visit mass sits on or near the PV | Optimism is path-concentrated | Build Phase 1 as a **path/PV correction** — the builder scoped previously survives; only `select_branches` changes to emit PV-depth rows instead of depth-1 rows |
| Raw rises positive **broadly** across many sibling/frontier nodes | Few-row Phase 1 is too narrow | Design a tree/frontier-level correction, or stop value calibration |
| Raw stays ≤ 0 across the visited subtree while `q` / root backup stays positive | No state anywhere is raw-optimistic; the +0.7 is a pure search/backup artifact | Value calibration is the wrong lever — redirect to search behavior or training data, and close this branch |

This is the parent spec's "falsify fast" clause, discharged for the cost of one diagnostic run.

---

## 9. Testing

Pure tests on synthetic `MCTSNode` trees (mirroring `tests/test_continuation_extraction.py`'s real-node pattern), no checkpoints and no MCTS:

- `select_positive_branches` reproduces the locked rule: only `root_mcts_black_value > 0` roots; children taken in descending positive share until cumulative ≥ 0.90 or 3 taken; roots with zero positive mass are skipped.
- The subtree walk reaches every descendant with `visit_count ≥ 1` and no others (a `visit_count == 0` created child is excluded; its subtree is not walked).
- PV annotation marks exactly the best-child chain from the selected child, with correct `pv_depth_index`, and marks nothing off-chain.
- `depth_from_root`, `depth_from_selected_child`, `visit_share_from_parent`, and `visit_share_from_root` are correct on a hand-built asymmetric tree.
- The root-value cross-check raises on a mismatched Phase-0 value.
- The by-depth aggregate computes `pct_visit_mass_raw_positive` correctly when raw-positive nodes hold a minority of nodes but a majority of visits (and vice versa) — the case the metric exists to catch.

Never `sys.modules.pop("mlx")`. Judge on the file-scoped run; a fresh worktree lacks gitignored game-log data and shows exactly 14 failed + 6 errors from unrelated fixtures. Authoritative suite on merged main: **1416 passed**.

---

## 10. Cost and inputs

17 seeded 400-sim searches (Phase 0 ran 30) plus roughly 6,800 nodes × 2 checkpoints ≈ 13.6k raw forwards. Phase 0 ran 4.8k forwards and 30 searches in ~7 minutes; expect **8–15 minutes**, in-session.

- Phase-0 CSV: `logs/eval/v15prep_a_continuation_concentration.csv`
- A roots manifest: `logs/eval/loss_analysis_v2_calib020_0001_vs_0379_black/0001_black_post_opening_top30_predrop_probe_manifest.csv`
- BASE: `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`
- v14b: `checkpoints/alphazero-v14b-value-adapter-projection-from-calib020-0001/model_iter_0001.safetensors`
- Gate-faithful search: `EvalConfig(mcts_sims=400, mcts_eval_batch_size=14, mcts_stall_flush_sims=48)`, `_default_evaluator_factory` (train-mode BN, `compile=True`), `MCTS(...).search_with_root(state, add_noise=False)`, `row_seed(CORRECTION_TAG, game_idx, position_ply, pos_base_seed=20260616, goal_base_seed=20260614)`.

**Stop after the CSVs.** The full-subtree-versus-PV read is the operator's; do not design or build Phase 1 without it.
