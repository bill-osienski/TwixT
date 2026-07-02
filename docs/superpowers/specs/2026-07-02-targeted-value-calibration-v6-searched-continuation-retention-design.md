# Targeted Value Calibration v6 — Searched-Continuation / PV Retention (Design)

**Date:** 2026-07-02
**Status:** spec — APPROVED for implementation 2026-07-02 (user review incorporated)
**Ledger:** `docs/2026-06-26-targeted-value-calibration-experiment-ledger-v3f-v4-overlap-updated-v6-prep.md`
**Supersedes as next-branch:** v5 (`mcts_root_retention`, REJECTED 2026-07-02)

## 0. Context and goal

The ledger's v5 path diagnostic identified the retention failure mechanism: root-level
anchors (scalar value, raw teacher value+priors, root-visit policy) held at the fragile
B/C/D root positions while the **searched continuation/child NN values** one-plus plies
below them drifted massively (D top-child NN +0.03 → +0.80). v6 therefore anchors the
**states BASE MCTS actually searches beneath the fragile roots**, not the roots again.

Goal: keep the A (black pre-drop) correction rows unchanged; add continuation retention
rows extracted from BASE's gate-faithful 400-sim search trees under each B/C/D root;
train them with the existing masked teacher-retention machinery; judge with the same
A/B/C/D gates vs `calib020_0001`.

Non-goals (ledger do-not-repeat): any root-row redesign, root-policy CE weight increase,
weight/schedule sweeps, promotion match before all four gates pass, generic doc changes.

## 1. Resolved design decisions

Four points were open or contradictory in the proposed plan. Resolutions (D1–D4 were
taken without live user input and are **flagged for review**):

- **D1 — Value-only continuation rows on the first run.** Each continuation row anchors
  only the BASE raw teacher value (eval-mode forward at the continuation state — the
  mechanism the raw-NN diagnostic proved holds). No policy CE on continuation rows.
  Rationale: the drifting quantity in the diagnostic was child NN *values*; policy
  anchoring already failed at roots twice (v4 raw priors, v5 root visits); subtree visit
  distributions at depth 2–3 rest on tens of visits and make noisy targets. The builder
  exposes `--emit-continuation-policy` (default off) which adds dense eval-mode teacher
  policy (`teacher_policy_json` + `teacher_legal_moves_sha1`) to the same rows; the
  builder is deterministic, so a v6b policy variant is a rerun, not new code.
- **D2 — Extraction rules are tag-based, not sharpness-gated.** The proposed per-row
  sharpness conditions (C ≥ 0.65, D < 0.20) leave rows in [0.20, 0.65) with no rule.
  Instead every row of a family gets its family's extraction shape (§3); the root
  max-visit-share is logged per row and a WARNING is emitted when a row contradicts its
  family's expected shape (a diffuse C root or sharp D root). Telemetry, not a row gate —
  no silent coverage holes.
- **D3 — v5 root-retention rows stay in the manifest, unscheduled.** Verified:
  `CalibrationPool.sample_by_tag` draws only scheduled tags (`calibration_pool.py:269-282`),
  so rows tagged `goal_line_retention` / `old_post_opening_retention` /
  `red_predrop_retention` are inert under the v6 schedule. v6 is continuation-first;
  root retention can be re-enabled later by editing only the schedule string.
- **D4 — `--max-total-continuation-rows 250`, hard-fail.** True worst case is
  B 18×2=36, C 30×3=90, D 30×(3+3)=180 → 306 before the child-PV visit threshold; the
  threshold (§3) is what pulls D toward the intended ~90–120. Exceeding any cap fails
  loudly; the builder never silently trims. **Operator expectation:** if D is denser
  than expected, the first full build may hard-fail at the cap — that is a tuning point
  (raise the cap or `--d-child-pv-min-visits`), not an implementation failure.

## 2. Manifest schema

New rows are appended to the v5 manifest content
(`logs/eval/targeted_calibration_v5_mcts_root_from_calib020_0001.csv` is the `--source`).
Output: `logs/eval/targeted_calibration_v6_continuation_from_calib020_0001.csv`.

Row classes in the v6 manifest:

| class | loss_mode | tag | drawn? |
|---|---|---|---|
| A correction (50) | `hard_value` | `black_predrop_correction` | yes (schedule=2) |
| v5 root retention (78) | `mcts_root_retention` | old `*_retention` tags | no (unscheduled) |
| B continuation (≤36) | `searched_continuation_retention` | `goal_line_continuation_retention` | yes (schedule=1) |
| C continuation (≤90) | `searched_continuation_retention` | `old_post_opening_continuation_retention` | yes (schedule=2) |
| D continuation (≤180, expected ~90–120) | `searched_continuation_retention` | `red_predrop_continuation_retention` | yes (schedule=2) |

New columns (blank on all non-continuation rows):

- **Reconstruction:** `extra_moves_json` — JSON list of `{"row": r, "col": c}` applied
  after the base `position_state(replay, position_ply, side_to_move)` reconstruction;
  `continuation_side_to_move` — expected side to move at the final continuation state
  (the existing `side_to_move` column keeps the ROOT's side so the
  `position_state` assertion at `goal_line_trigger_probe_cases.py:85-88` stays valid);
  `continuation_legal_moves_sha1` — sha1 of the continuation state's legal moves,
  verified at load even when no policy target exists.
- **Provenance/audit:** `continuation_depth` (1-based), `continuation_parent_case_id`
  (root row's `case_id`), `continuation_source` (`pv` | `top_child` | `child_pv`),
  `continuation_path_moves` (human-readable `"19:9>18:11"` string),
  `continuation_tree_visits` (the continuation node's subtree visit count in the root
  search), `continuation_tree_nn_value` (the node's train-mode `nn_value` from the
  search tree — provenance ONLY, never a training target).
- **Teacher fields (reused):** `teacher_value` = eval-mode raw stm value at the
  continuation state via `_teacher_infer`; `teacher_value_source=base_raw_continuation`.
  `teacher_policy_json`/`teacher_legal_moves_sha1` stay blank unless
  `--emit-continuation-policy`.
- **`case_id`:** `<parent_case_id>__cont_<source><depth>_<path>` with filesystem/shell-safe
  path characters (e.g. `game_000433_ply_029__cont_pv2_19-9_18-11`) — unique, greppable,
  parent recoverable, safe if a downstream tool ever embeds case_id in a filename. The
  human-readable `continuation_path_moves` column keeps the `19:9>18:11` form.
- **`weight_scale`:** continuation rows inherit the parent root row's `weight_scale`
  verbatim (1.0 across the current manifest; explicit, never blank — though the pool
  treats blank as 1.0 via `_parse_weight_scale`).

Root-search provenance columns (`root_sims`, `root_seed`, …) are stamped on continuation
rows with the parent root's values.

## 3. State selection (extraction rules)

From each root's single gate-faithful 400-sim BASE search tree:

- **C rows (30):** PV depth 1–3 — follow the max-visit child chain; stop early if a
  node is unexpanded, terminal, or has zero-visit children. Up to 3 rows/root.
- **D rows (30):** top-3 root children by visit count; below each selected child, one
  further PV step **only if that child's subtree visit count ≥ `--d-child-pv-min-visits`
  (default 40 of 400)**. Up to 6 rows/root, expected ~3–4.
- **B rows (18):** PV depth 1–2. Up to 2 rows/root.

Extraction order is stable (PV before top-child, children by descending visits, ties by
encoded move id). Duplicate states cannot arise at these depths under these rules (a
single PV chain; distinct top children; alternating sides at depth 2), but the builder
asserts `case_id` uniqueness across the whole output as a guard.

Caps (all hard-fail): `--max-continuations-per-root` (default 6),
`--max-total-continuation-rows` (default 250).

## 4. Loader changes (`calibration_pool.py`)

1. **`extra_moves_json` support** in `build_calibration_position`: absent/blank column →
   byte-identical current behavior (v2–v5 manifests unchanged). Present → parse JSON
   list of `{row, col}`; after `position_state` returns the root state, apply each move
   via `state.apply_move((row, col))`. Fail loud (ValueError with case_id) on: invalid
   JSON, empty list on a continuation row, illegal move, final `to_move` ≠
   `continuation_side_to_move`, recomputed `legal_moves_sha1` ≠
   `continuation_legal_moves_sha1`.
2. **New loss mode `searched_continuation_retention`** added to `VALID_LOSS_MODES`; the
   per-mode `RETENTION_POLICY_LOSS_MODES` mask test is replaced by a per-row rule
   (bullets below):
   - requires `teacher_value` (stm outcome, exactly like `teacher_retention`);
   - policy is **optional per row**: if `teacher_policy_json` present → dense policy in
     `visit_counts` + sha1 check (existing `_parse_policy_json`), policy mask 1.0;
     if blank → `visit_counts = [0]*len(legal)`, policy mask 0.0.
   - This requires the mask in `split_samples_with_modes` (`calibration_pool.py:322-330`)
     to become per-row (mode ∈ retention set AND policy present) instead of per-mode.
     `teacher_retention`/`mcts_root_retention` rows always have policy, so their mask
     value is unchanged — regression-tested.
3. **Column guards:** continuation rows must have `extra_moves_json` +
   `continuation_side_to_move` + `continuation_legal_moves_sha1` + `teacher_value`,
   and must leave `target_black_value`/`root_visits_json` blank. `hard_value` rows must
   leave all continuation columns blank (extend the existing guard at
   `calibration_pool.py:202-210`).

No trainer loss-math changes: the new mode rides the existing 14-tuple masked path
(`trainer.py:1231-1295`) — `outcome = teacher_value`, eval-mode calibration forward,
value MSE + masked policy CE (CE contributes 0 while all continuation policy masks are 0).
Telemetry (`calib_n_drawn_by_tag`, `calib_value_term`, `calib_policy_ce`, `kl_est`)
works unchanged. Note: `n_teacher_retention` (tuple slot 13) is derived from the policy
mask, so value-only continuation rows will NOT appear in it — acceptance criterion 1
reads continuation draw counts from `calib_n_drawn_by_tag` (the three
`*_continuation_retention` tags), not from `n_teacher_retention_drawn`. No new tuple slot.

## 5. Gate-faithful root-returning search helper

The builder must extract from the SAME search the gates run. `MCTS.search()`
(`mcts.py:410-455`) discards its root; `search_from_root` uses a different (batched
waiter-list) leaf-eval path and is off-limits for target generation.

Change: refactor `MCTS.search()` body into `_search_impl(root_state, add_noise) ->
(visit_counts, q_value, root_node)`; `search()` delegates and returns the first two —
behaviorally byte-identical (test: fixed seed, `search()` results identical pre/post
refactor). A thin public wrapper `search_with_root()` returns all three; the builder
calls it. Node walking uses `MCTSNode.children` (keyed by encoded move id),
`visit_count`, `state`, `nn_value` (`mcts.py:152-194`) — child states are stored on the
nodes, no re-application needed; `extra_moves_json` is decoded from the child move ids.

## 6. Builder

`scripts/GPU/alphazero/build_searched_continuation_retention_manifest.py`, modeled
directly on the v5 builder (`build_mcts_root_retention_manifest.py`):

- **Inputs:** `--source` (v6 uses the v5 output manifest), `--base-checkpoint`, `--out`,
  `--sims 400 --eval-batch-size 14 --stall-flush-sims 48`, seeds
  (`--position-probe-base-seed 20260616`, `--goal-line-base-seed 20260614`), gate
  cross-check flags (`--gate-cases-csv` ×2, `--gate-checkpoint-label`,
  `--gate-tolerance 1e-3`), extraction flags (`--b-pv-depth 2 --c-pv-depth 3
  --d-top-k 3 --d-child-pv-depth 1 --d-child-pv-min-visits 40`), caps (§3),
  `--emit-continuation-policy` (default off), `--limit-cases` (small diagnostic runs).
- **Per source row:** `black_predrop_correction` → pass through unchanged. **Extraction
  sources are exactly the rows with `loss_mode == mcts_root_retention` AND tag in
  {`goal_line_retention`, `old_post_opening_retention`, `red_predrop_retention`}** —
  any other row (hard_value, or continuation rows if the builder is ever rerun on a v6
  output) passes through untouched and is never extracted from; an unknown
  loss_mode/tag combination is a hard error. The qualifying v5 root rows pass through
  unchanged (they stay in the manifest, D3) AND serve as the extraction roots: reconstruct root via `position_state`, seed via the v5
  `row_seed()` formula, run the root-returning gate-faithful search with
  `add_noise=False` (train-mode-BN search evaluator via the same evaluator factory as
  v5), cross-check recomputed `root_black_value` against the gate CSVs (v5's
  `cross_check_gate_values` logic, label-filtered), then extract continuation rows per §3.
- **Teacher values:** per continuation state, `_teacher_infer(node.state,
  raw_evaluator)` where `raw_evaluator` is the separate eval-mode
  `load_network_for_scoring(...); network.eval()` instance (v5's two-evaluator split).
  The tree's `nn_value` is recorded as `continuation_tree_nn_value` provenance only.
- **Determinism:** per-row seeds as above; stable extraction ordering (§3); byte-identical
  output for identical inputs (test).
- **Telemetry:** per-family and total continuation row counts, per-row root
  max-visit-share, D2 shape warnings, and explicit logging of every cap/threshold that
  excluded a candidate row (no silent truncation).

## 7. Smoke

`scripts/GPU/alphazero/smoke_searched_continuation_retention_v6.py` (pattern: v5 smoke):

1. Manifest loads via `CalibrationPool.from_manifest`.
2. Every continuation row reconstructs (base + extra moves) and passes the
   `continuation_side_to_move` + `continuation_legal_moves_sha1` checks.
3. At BASE init with frozen/eval BN: value MSE on continuation rows ≈ 0 (they anchor
   BASE's own eval-mode values); policy CE finite where a policy target exists, exactly
   0 contribution where not; no NaN.
4. Correction rows unchanged (hard_value, no continuation fields).
5. Tag schedule validation: all scheduled tags present; **hard assertion** that a
   sample round's draws-by-tag contain all three `*_continuation_retention` tags with
   positive counts in the scheduled 1:2:2 ratio; startup log reports mode + per-family
   continuation counts.

Expected: `PASS v6 continuation retention mechanics: n_continuation=<N>, value_mse≈0,
policy_ce=0 (value-only run), no NaN`.

## 8. Training, gates, acceptance

Train command = v5 command with three deltas: v6 manifest path, checkpoint dir
`checkpoints/alphazero-v6-continuation-from-calib020-0001`, schedule
`black_predrop_correction=2,goal_line_continuation_retention=1,old_post_opening_continuation_retention=2,red_predrop_continuation_retention=2`.
Unchanged: `--post-opening-calibration-weight 0.01 --post-opening-calibration-target
-0.35 --post-opening-calibration-teacher-value-weight 1.0
--post-opening-calibration-teacher-policy-kl-weight 0.25 --freeze-batchnorm-stats`
(the policy-KL weight is inert while continuation rows are value-only).

Gates: same A/B/C/D 400-sim probes vs `calib020_0001`,
`OUT=logs/eval/v6_continuation_from_calib020_0001_gates_400s`. Acceptance:

1. Build + smoke pass; the training run's `calib_n_drawn_by_tag` telemetry shows all
   three `*_continuation_retention` tags drawn in schedule ratio (hard check — do NOT
   use `n_teacher_retention_drawn`, which is policy-mask-derived and stays 0 on a
   value-only run).
2. **A:** mean ≤ 0.0 and severe materially below 43.3% (baseline).
3. **B:** severe 0.0%, over ≤ 11.1%.
4. **C:** severe ≤ 13.3%, over ≤ 33.3%, mean ≤ +0.099.
5. **D:** severe 0.0%, mean ≤ 0.0.
6. Promotion match ONLY after all four gates pass.

## 9. If v6 fails

Do not sweep weights. Re-run the v5-style path diagnostic on failed B/C/D rows and fork:

- continuation rows absent for the failing subtree → selection issue (depth/top-k too
  conservative) — revisit §3 parameters;
- continuation rows present but their raw values drifted → retention strength/schedule
  issue — first check `calib_value_term` telemetry and per-tag draw counts;
- raw continuation values held but MCTS gates still failed → deeper/broader coverage
  needed (depth 4+, larger top-k, or the D1 policy variant via
  `--emit-continuation-policy`) — NOT a root/weight sweep.

## 10. Implementation order

1. Loader: `extra_moves_json` reconstruction + `searched_continuation_retention` mode +
   per-row policy mask (TDD; regression tests that v2–v5 manifests load byte-identical).
2. `MCTS._search_impl` refactor with byte-identical `search()` test.
3. Builder (TDD; includes determinism, caps, illegal-move, cross-check tests).
4. Smoke script.
5. Diagnostic build on the 6 path-diagnostic rows (`--limit-cases`), inspect by hand.
6. Full build → smoke → 1-iter train → A/B/C/D gates → ledger update.

Per-task commits on a feature branch; FF-merge to main; full suite green before merge.
