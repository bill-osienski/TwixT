# Targeted Value Calibration v12 — Asymmetric One-Sided Guardrail Hinge Design

**Status:** DESIGN — verbally approved + locked by user 2026-07-05; awaiting written-spec review, then writing-plans.
**Date:** 2026-07-05
**Follows:** v10/v10b/v11 (all rejected; branch closed). The v10/v10b/v11 sequence proved existing root/continuation/root-value/severe-row pressure cannot safely pass A/B/C/D under final-block training. The next credible branch must introduce a **new objective**, not another schedule/row variant.

## Why v12 is a new objective, not a rename

The existing calibration value term is already symmetric raw delta-to-BASE: `per_value = (cb_values − cb_targets)²` with `cb_targets` = the stored BASE raw value (trainer.py:1255). v4 (raw-NN), v6 (continuation), and v11 (B root-value clones) all compute this and all failed — do-not-repeat #8: *"the objective, not its weighting, is wrong for the gate."* So v12 does **not** add more symmetric preservation. It changes the **shape** of the guardrail loss.

**Mechanism (Option B, chosen over gradient projection):** every B/C/D guardrail failure is **overvalue drift** (candidate becomes more pro-black than BASE). We do not need to penalize all movement away from BASE — only movement that makes B/C/D too pro-black. So guardrail rows use a **one-sided hinge** instead of symmetric MSE. This stays inside the trainer's single backward pass (`loss_tuple, grads = nn.value_and_grad(network, loss_fn)(network)`, trainer.py:1391), unlike gradient-conflict handling (Option A) which needs separate A/guardrail gradients (2–3 backward passes) — deferred to v13 only if v12 fails while clearly showing gradient conflict is the cause.

## The hinge (black-perspective, computed in stm space)

The gate risk is "too pro-black." The trainer sees **side-to-move** values, and "too pro-black" flips sign by side (C is black-to-move, D red-to-move, B mixed). The correct penalty is in **black** perspective:

```
sign        = +1 if side_to_move == "black" else -1      # per row, from record.to_move
cand_black  = cb_value_stm * sign
guardrail_loss_row = relu(cand_black - target_black - margin) ** 2
```

Because `target_in_to_move` converts a black target to stm by the same `sign` (calibration_pool.py:48–58), `target_black = cb_target_stm * sign`, so the hinge collapses to a form that needs only the per-row sign and the existing stm `cb_targets`:

```
guardrail_loss_row = relu( sign * (cb_values - cb_targets) - margin ) ** 2
```

**Worked check (D, red-to-move):** `red_loss_game_000752…`: teacher_value stm = −0.9729, sign = −1 → BASE black = +0.9729. `sign*(cb_values − cb_targets) − margin = (target_stm − cand_stm) − margin`, which is > 0 exactly when `cand_stm < target_stm − margin`, i.e. the candidate's black value rises **above** BASE — the overvalue direction. Correct. **(C, black-to-move):** sign +1 → fires when `cand_stm > target_stm + margin`. Correct. Without the sign, red-to-move rows would be penalized in the wrong direction — this is the central correctness requirement.

- Candidate **less** pro-black than BASE → no penalty.
- Candidate above BASE but within `margin` → no penalty.
- Candidate more pro-black than BASE by more than `margin` → penalized.

`margin` default **0.10** (black-value units), exposed as `--guardrail-margin`. Do not sweep it unless v12 is a near-miss.

## Locked scope

- **Training surface unchanged:** `--freeze-batchnorm-stats --train-value-head-and-final-block` (v9, merged) — the smallest surface that fixes A.
- **A untouched:** `black_predrop_correction`, `loss_mode=hard_value`, `target_black_value=-0.35`, symmetric. The hinge is guardrail-only.
- **B/C/D:** one-sided black-perspective guardrail hinge, **root rows only**, **no policy CE**, **no symmetric retention on the same states** (guardrail *replaces* the old `*_retention`/root-value tags for B/C/D — running both would double-anchor and reintroduce v10b/v11 over-anchoring).
- **Raw value, not MCTS** (keeps v12 localized/cheap; MCTS/continuation guardrails are v12b/v13 only if root-only gives a clear partial win).

## Target column semantics (explicit — per user)

For `asymmetric_guardrail_retention` rows:
- **`target_black_value` is the BASE raw black-perspective value** — the hinge target.
- **`teacher_value` is provenance only** (the stm anchor it was derived from), **not** the loss target.

This deliberately avoids confusion with the existing side-to-move `teacher_value` used by symmetric retention.

## Files / components

| File | Change |
|---|---|
| `scripts/GPU/alphazero/calibration_pool.py` | add `loss_mode="asymmetric_guardrail_retention"` to `VALID_LOSS_MODES` — `build_calibration_position` needs **no new branch**, guardrail rows fall through the existing default (`hard_value`) branch which already sets `outcome = target_in_to_move(_resolve_target_black(...))` (stm) with zero visit_counts; add a `build_calibration_sample` validation branch (assertions below); `has_policy_target=False`; emit a per-row **guardrail sign vector** (sign ±1 for guardrail rows, 0 otherwise — doubles as the mask) from `record.to_move` + loss_mode, parallel to the existing `teacher_policy_mask` |
| `scripts/GPU/alphazero/build_v12_guardrail_manifest.py` (create) | clone the B/C/D root-retention rows into guardrail rows (below) |
| `scripts/GPU/alphazero/trainer.py` | `alphazero_loss_batch` + `train_step`: new optional `calibration_guardrail_sign` vector + `guardrail_margin` scalar → masked hinge term; guardrail telemetry; **byte-identical when the sign vector is None** (parallels the v4 teacher_policy_mask addition). Plumb `guardrail_margin` through `train()` |

**Loss composition (no double-counting):** when `calibration_guardrail_sign` is present, the symmetric value MSE (trainer.py:1255) is masked to **exclude** guardrail rows (weight `(1 − |sign|)`), and the hinge term covers exactly the guardrail rows (weight `|sign|`). Each calibration row contributes to exactly one of {symmetric MSE, hinge, A hard-value MSE}. `calib_value_term` = symmetric MSE over non-guardrail rows; `guardrail_hinge_loss` = hinge over guardrail rows; both feed `calib_loss`. When the sign vector is `None`, the symmetric term covers all rows exactly as pre-v12 (byte-identical).
| `scripts/GPU/alphazero/train.py` | `--guardrail-margin` (float, default 0.10) + plumb |
| `tests/test_asymmetric_guardrail_*.py` (create) | loss-shape, sign, validation, byte-identical tests |
| `scripts/GPU/alphazero/smoke_asymmetric_guardrail_v12.py` (create) | gate-0 smoke (mirrors prior smokes) |

**Do NOT touch:** existing symmetric/retention/continuation loss paths (additive only), mcts.py, continuation_extraction.py, the v8/v9 verifiers, docs/post-game-analysis.md.

## Loader validation (per user)

`loss_mode == "asymmetric_guardrail_retention"` requires, in `build_calibration_sample` (fail loud):
- `target_black_value` populated (finite in [−1, 1]);
- `teacher_policy_json` blank;
- `root_visits_json` blank;
- resulting `has_policy_target == False`.

`asymmetric_guardrail_retention` is **not** in `RETENTION_POLICY_LOSS_MODES` (so `__post_init__` never forces a policy target) and **not** in `TEACHER_MODE_LOSS_MODES` (no policy CE). It IS added to `VALID_LOSS_MODES`.

## v12 manifest builder

`build_v12_guardrail_manifest.py` clones each B/C/D root-retention row into a guardrail row (pure copy + arithmetic, no reconstruction/MCTS — the BASE anchor already lives in `teacher_value`):

| source tag (`mcts_root_retention`) | new tag (`asymmetric_guardrail_retention`) |
|---|---|
| `goal_line_retention` | `goal_line_guardrail_retention` |
| `old_post_opening_retention` | `old_post_opening_guardrail_retention` |
| `red_predrop_retention` | `red_predrop_guardrail_retention` |

Per clone: `case_id += "__guardrail"`; `tag=<family>_guardrail_retention`; `loss_mode="asymmetric_guardrail_retention"`; `target_black_value = teacher_value_stm * sign` (sign from `side_to_move`); keep `teacher_value` (provenance), `replay_path`, `position_ply`, `side_to_move`; blank `teacher_policy_json`, `root_visits_json`, `root_legal_moves_sha1`, all `continuation_*`, `extra_moves_json`. Validate expected new tag counts (18 / 30 / 30). Input = v7 manifest; output = `logs/eval/targeted_calibration_v12_guardrail_from_calib020_0001.csv`.

## Telemetry

- `guardrail_hinge_loss` — mean masked hinge over guardrail rows.
- `guardrail_active_frac` — fraction of drawn guardrail rows with hinge > 0 (i.e. actually over the margin).
- `guardrail_margin` — the margin used (echo of the flag).
- `guardrail_n_drawn_by_tag` — reuse the existing `calib_n_drawn_by_tag` (the three guardrail tags). `n_teacher_retention_drawn` and `calib_policy_ce`/`kl_est` stay **0** (no policy CE in v12) — that zero is itself a check the guardrail rows carry no policy target.
- Return-arity: guardrail telemetry rides a new gated tuple variant; when no guardrail rows are drawn (`calibration_guardrail_sign is None`), the existing 7/10/14-tuple arities and values are **byte-identical**. Exact tuple layout is a plan detail.

## Tests (behavioral, not mocks)

- `below target → 0`: candidate less pro-black than target → hinge 0.
- `within margin → 0`: candidate above target but within margin → hinge 0.
- `above target+margin → positive`: candidate over the band → hinge > 0, value = `(over)²`.
- `red-to-move sign`: a red-to-move row whose candidate stm is **below** target (→ more pro-black) fires the hinge; a black-to-move row with the same numeric stm relation does not. Directly pins the sign correction.
- `no policy CE`: guardrail rows produce `has_policy_target=False`, policy mask 0, `calib_policy_ce` unaffected.
- `byte-identical when unused`: with `calibration_guardrail_sign=None`, `alphazero_loss_batch`/`train_step` return the pre-v12 tuples unchanged; all pre-existing calibration tests pass.
- loader validation: each of the four assertions raises on violation.

## v12 schedule (starting ratios; adjustable in the plan)

```
black_predrop_correction=2,goal_line_guardrail_retention=1,old_post_opening_guardrail_retention=2,red_predrop_guardrail_retention=2
```
The B/C/D tags are the NEW guardrail-hinge tags — not the old root-policy/symmetric tags.

## Operator run (USER's, after merge)

1. Build the v12 manifest: `.venv/bin/python scripts/GPU/alphazero/build_v12_guardrail_manifest.py` (expect +78 rows: 18+30+30 guardrail).
2. Train: v9 command + `--post-opening-calibration-manifest <v12 csv>` + the v12 schedule + `--guardrail-margin 0.10` + `--freeze-batchnorm-stats --train-value-head-and-final-block`, new checkpoint dir.
3. Telemetry checks: `guardrail_n_drawn_by_tag` nonzero for the three tags; `guardrail_hinge_loss` and `guardrail_active_frac` present; `n_teacher_retention_drawn=0`, `calib_policy_ce=0` (no policy CE); `train_value_head_and_final_block=True`, `unfrozen_block_index=5`.
4. Acceptance verifier: the v9 `verify_value_head_and_final_block_checkpoint.py` (unchanged flags) must exit 0.
5. Gates A/B/C/D vs `calib020_0001`, `OUT=logs/eval/v12_guardrail_from_calib020_0001_gates_400s`. Thresholds unchanged. No promotion unless all four pass.

## Interpreting v12

- **A + B/C/D all pass** → promotion match.
- **A pass, B/C/D near-miss (fewer/less-severe overvalue rows than v10)** → one-sided guardrail is working; consider `--guardrail-margin` tune or continuation guardrails (v12b).
- **A pass, B/C/D still break like v10/v11** → the guardrail loss is being overwhelmed by the A-correction gradient in the shared final block → the diagnosed gradient conflict is real → v13 = gradient-conflict handling (Option A).
- **A weakens** → the hinge is over-constraining; raise margin before abandoning.

Do NOT implement gradient projection (Option A) in v12. It is v13, gated on v12 failing while clearly showing conflicting gradients.
