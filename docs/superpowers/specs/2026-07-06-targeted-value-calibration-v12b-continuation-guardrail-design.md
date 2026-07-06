# Targeted Value Calibration v12b — Continuation Guardrail Rows Design

**Status:** DESIGN — approved + locked by user 2026-07-06; awaiting written-spec review, then writing-plans.
**Date:** 2026-07-06
**Follows:** v12 (asymmetric one-sided black-perspective guardrail hinge — IMPLEMENTED + merged @ `2cc4bd1`; operator run pending). v12b reuses the v12 objective unchanged and only extends *which states* carry the guardrail.

## Why v12b is a loader + builder change, not "builder-only"

v12b's hypothesis is that **C and D need continuation guardrails, not just root guardrails** — the failed B/C/D cases under final-block training (v9/v10/v11) all had *scheduled continuation retention* that couldn't hold under the old symmetric objective, and v12 only applied the new one-sided hinge to **root** rows. v12b applies the same hinge to the **searched-continuation** states as well.

The v12 trainer objective already supports this with zero changes: `loss_mode=asymmetric_guardrail_retention`, `target_black_value`, the per-row black-perspective `guardrail_sign`, the one-sided hinge `relu(sign*(cb_values − cb_targets) − margin)²`, no policy CE, the 13-tuple, and `--guardrail-margin`. **No trainer.py change, no new loss mode, no new CLI flag, no gradient projection, no gate change.**

**But the loader cannot currently express a continuation row as a guardrail row.** `build_calibration_position` (calibration_pool.py) only walks the continuation path (`_apply_extra_moves`, which applies `extra_moves_json` to reach the searched state) when `loss_mode == CONTINUATION_LOSS_MODE`:

```python
# calibration_pool.py:242 (current)
record_ply = position_ply
if loss_mode == CONTINUATION_LOSS_MODE:
    state, n_extra = _apply_extra_moves(state, case)
    record_ply = position_ply + n_extra
```

A `GUARDRAIL_LOSS_MODE` row falls through to the default branch (calibration_pool.py:293–302), which reconstructs the **root** position at `position_ply` and **ignores `extra_moves_json`**. So a "continuation guardrail row" built naively — `loss_mode=asymmetric_guardrail_retention` with `extra_moves_json` preserved — would validate fine but reconstruct the **wrong board** (the root, not the searched continuation). The hinge would then compare the *root's* candidate value against the *continuation's* BASE target — a silently mismatched target/position pair. Preserving the continuation fields in the CSV is necessary but not sufficient; the loader must actually *use* them for guardrail rows.

The fix is one gate condition in `build_calibration_position` (below) — a loader change in calibration_pool.py, **not** the trainer objective.

## The loader change (the entire loader delta)

In `build_calibration_position`, apply the continuation path for continuation rows **or** for guardrail rows that carry a non-empty `extra_moves_json`:

```python
    record_ply = position_ply
    is_guardrail_continuation = (
        loss_mode == GUARDRAIL_LOSS_MODE
        and case.get("extra_moves_json") not in (None, ""))
    if loss_mode == CONTINUATION_LOSS_MODE or is_guardrail_continuation:
        state, n_extra = _apply_extra_moves(state, case)
        record_ply = position_ply + n_extra
```

Properties:
- **Root guardrail rows are unchanged.** They carry blank `extra_moves_json`, so `is_guardrail_continuation` is False and they fall through to the existing root/default branch exactly as in v12 (byte-identical v12 behavior).
- **The presence of `extra_moves_json` cleanly distinguishes** a root guardrail row from a continuation guardrail row — no new flag or field needed.
- **Fail-loud is preserved.** `_apply_extra_moves` already verifies `continuation_side_to_move` and `continuation_legal_moves_sha1` against the reconstructed state and raises on any mismatch, so a malformed continuation guardrail row raises rather than silently mis-reconstructing.
- **No validation-branch change is strictly required.** The `GUARDRAIL_LOSS_MODE` validation in `build_calibration_sample` (calibration_pool.py:329–341) only requires `target_black_value` populated and `teacher_policy_json` / `root_visits_json` blank — all satisfied by a value-only continuation guardrail row. It does not forbid `extra_moves_json`, so continuation guardrail rows pass. (An optional hardening — fail loud early if a guardrail row has `extra_moves_json` populated but `continuation_side_to_move` blank — is redundant with `_apply_extra_moves` and is left out for minimalism.)
- **No `from_manifest` change.** A v12b manifest's modes are `{hard_value, asymmetric_guardrail_retention}` (hard_value stripped before the check) → schema `GUARDRAIL_LOSS_MODE`, already admitted by `_ALLOWED_RETENTION_MODE_SETS`.

## Target / sign semantics (the critical builder correctness point)

After `_apply_extra_moves`, the board sits on the **continuation** state whose side is `continuation_side_to_move` (not the root's `side_to_move`). The loader computes the guardrail sign from `record.to_move` (in `split_samples_with_guardrail`) = the continuation side, and `cb_targets = target_in_to_move(record.to_move, target_black_value)`. So the builder must convert the target using the **continuation** side:

```
continuation guardrail:  sign = +1 if continuation_side_to_move == "black" else -1
                         target_black_value = teacher_value * sign
root guardrail (v12):    sign = +1 if side_to_move == "black" else -1
                         target_black_value = teacher_value * sign
```

Worked round-trip (continuation, depth d): `teacher_value` is the BASE raw value in continuation-stm perspective. `target_black = teacher_value * continuation_sign`. In the loader, `cb_targets = target_in_to_move(continuation_side, target_black) = target_black * continuation_sign = teacher_value` (since `continuation_sign² = 1`) — i.e. `cb_targets` is exactly the continuation stm anchor. The hinge `sign*(cb_values − cb_targets) − margin` with `sign = continuation_sign` reduces to `(cand_black − target_black) − margin`, firing exactly when the candidate's continuation-state black value rises above BASE by more than the margin. Correct one-sided black-perspective overvalue penalty at the continuation state.

**Using the root `side_to_move` for a continuation clone would flip the sign whenever the continuation depth is odd** — the same "wrong direction" bug the black-perspective sign was invented to prevent.

## v12b manifest builder — `build_v12b_continuation_guardrail_manifest.py` (create)

Input = the **v7 manifest** (`logs/eval/targeted_calibration_v7_severe_d_root_correction_from_calib020_0001.csv`), which already holds the root, continuation, D-root-value, A-correction, and severe-D rows. Output = `logs/eval/targeted_calibration_v12b_continuation_guardrail_from_calib020_0001.csv`. Pure copy + arithmetic — no reconstruction/MCTS (the BASE anchor already lives in each parent's `teacher_value`). **Route clones by source `loss_mode`** (this also tightens v12's M1 tag-only-selection minor — a clone is emitted only when the source loss_mode AND tag both match):

| source loss_mode | source tag | → guardrail tag | sign from | continuation fields |
|---|---|---|---|---|
| `mcts_root_retention` | `goal_line_retention` | `goal_line_guardrail_retention` | `side_to_move` | blank |
| `mcts_root_retention` | `old_post_opening_retention` | `old_post_opening_guardrail_retention` | `side_to_move` | blank |
| `mcts_root_retention` | `red_predrop_retention` | `red_predrop_guardrail_retention` | `side_to_move` | blank |
| `searched_continuation_retention` | `old_post_opening_continuation_retention` | `old_post_opening_continuation_guardrail_retention` | `continuation_side_to_move` | **preserved** |
| `searched_continuation_retention` | `red_predrop_continuation_retention` | `red_predrop_continuation_guardrail_retention` | `continuation_side_to_move` | **preserved** |

Per clone (both kinds): `case_id += "__guardrail"`; `loss_mode = "asymmetric_guardrail_retention"`; `target_black_value = float(teacher_value) * sign` (sign per the table); keep `teacher_value` (provenance), `replay_path`, `position_ply`, `side_to_move`; blank the policy/root columns `teacher_policy_json`, `teacher_legal_moves_sha1`, `root_visits_json`, `root_legal_moves_sha1`, and the root_* metadata columns `root_value_stm`, `root_black_value`, `root_sims`, `root_base_checkpoint`, `root_seed`, `root_mcts_eval_batch_size`, `root_mcts_stall_flush_sims` (the v12 blank list + the v12 T2 fix additions).

**Root clones** additionally blank all continuation_* columns and `extra_moves_json` (identical to the v12 builder). **Continuation clones** additionally **preserve** the reconstruction fields: `extra_moves_json`, `continuation_source`, `continuation_depth`, `continuation_parent_case_id`, `continuation_side_to_move`, `continuation_legal_moves_sha1`.

Keep all `hard_value` rows from the v7 manifest (A `black_predrop_correction` + severe-D `red_predrop_severe_root_correction`), same as v12.

### Deliberate drops (locked)
- **`goal_line_continuation_retention` rows are NOT cloned** (B stays root-only). B passed cleanly in v12 (over 5.6% / severe 0.0%); adding B continuation pressure would introduce a new variable and risk over-constraining A/C/D without solving a current blocker. The v7 `goal_line_continuation_retention` rows are dropped from the output.
- **`red_predrop_root_value_retention` (v6c D root-value) rows are NOT cloned.** D root value is already covered by the `red_predrop_retention → red_predrop_guardrail_retention` root clone; cloning the depth-0 root-value rows too would duplicate D root pressure and muddy v12b's specific hypothesis (C/D need *continuation* guardrails, not more root-value pressure). Dropped from the output (same as v12, which kept only hard_value + guardrail clones).

### Validation / expected counts
The builder prints per-guardrail-tag counts. Expected root-guardrail counts match v12 (goal_line 18 / old_post_opening 30 / red_predrop 30). The two continuation-guardrail counts equal the number of `old_post_opening_continuation_retention` / `red_predrop_continuation_retention` rows in the v7 manifest (reported at build time; the operator confirms them). The built manifest must load through `CalibrationPool.from_manifest` with `schema == asymmetric_guardrail_retention`.

## Telemetry

Unchanged from v12: `guardrail_hinge_loss`, `guardrail_active_frac`, `guardrail_margin` present; `guardrail_n_drawn_by_tag` nonzero for the scheduled guardrail tags (now including the two continuation-guardrail tags); `n_teacher_retention_drawn` and `calib_policy_ce`/`kl_est` stay **0** (value-only, no policy CE) — that zero is the same correctness check as v12.

## Tests (behavioral, not mocks)

- **Loader (uses the committed `tests.goal_line_probe_fixtures.legal_replay` fixture, so it runs in a fresh worktree):**
  - A guardrail row **with** a non-empty `extra_moves_json` reconstructs the **continuation** state: `record.ply == position_ply + n_extra`, `record.to_move == continuation_side_to_move`, and `record.outcome == target_in_to_move(continuation_side, target_black_value)` (the continuation stm anchor). Cover both an even-depth and an odd-depth continuation so the sign flip is exercised.
  - A root guardrail row (blank `extra_moves_json`) reconstructs the **root** unchanged (`record.ply == position_ply`, `record.to_move == side_to_move`) — pins the v12 byte-identical behavior.
  - Sign vector from `split_samples_with_guardrail` for a continuation guardrail row equals `+1/−1` by `continuation_side_to_move`.
- **Builder:**
  - `make_continuation_guardrail_clone` computes `target_black_value = teacher_value * sign` with `sign` from `continuation_side_to_move` (assert both a black and a red continuation side, i.e. both depth parities), preserves the six continuation reconstruction fields, blanks `teacher_policy_json`/`root_visits_json`/root_* metadata, keeps `teacher_value`.
  - `make_root_guardrail_clone` still uses `side_to_move` and blanks continuation fields (v12 parity).
  - Clone selection is routed by source `loss_mode` **and** tag: a `searched_continuation_retention` row whose tag is `goal_line_continuation_retention` or `red_predrop_root_value_retention` is dropped (not cloned); a `mcts_root_retention` row whose tag is a root tag is cloned as a root guardrail.
- **Gate-0 smoke `smoke_v12b_continuation_guardrail.py`** (mirrors the v12 smoke, new manifest + v12b schedule): loads the v12b manifest, draws the schedule, asserts guardrail rows drawn (including at least one continuation-guardrail row), runs a `train_step` with the sign vector, asserts the 13-tuple + `guardrail_hinge_loss` present. Requires local replay data; defers to the operator box if absent (same as v12).

## v12b schedule (locked)

```
black_predrop_correction=2,goal_line_guardrail_retention=1,old_post_opening_guardrail_retention=1,old_post_opening_continuation_guardrail_retention=2,red_predrop_guardrail_retention=1,red_predrop_continuation_guardrail_retention=2
```

Shift from v12: the C/D **root** weights drop 2→1 and the freed weight moves to the two **continuation** guardrail tags (2 each); B (`goal_line`) stays root-only at 1; `black_predrop_correction` (A) stays 2.

## Operator run (USER's, after merge)

1. Build the v12b manifest: `.venv/bin/python scripts/GPU/alphazero/build_v12b_continuation_guardrail_manifest.py` (root-guardrail counts 18/30/30 + the two continuation-guardrail counts reported).
2. Train: the v9/v12 command + `--post-opening-calibration-manifest logs/eval/targeted_calibration_v12b_continuation_guardrail_from_calib020_0001.csv` + the v12b schedule + `--guardrail-margin 0.10` + `--freeze-batchnorm-stats --train-value-head-and-final-block`, new checkpoint dir (`checkpoints/alphazero-v12b-continuation-guardrail-from-calib020-0001`).
3. Telemetry: `guardrail_hinge_loss`/`guardrail_active_frac` present, `guardrail_margin=0.10`, `guardrail_n_drawn_by_tag` nonzero for all **five** scheduled guardrail tags, plus `black_predrop_correction` (the A `hard_value` tag) drawn separately, `n_teacher_retention_drawn=0`, `calib_policy_ce=0`; `train_value_head_and_final_block=True`, `unfrozen_block_index=5`.
4. Acceptance verifier: `verify_value_head_and_final_block_checkpoint.py` (unchanged v9 flags) must exit 0.
5. Gates A/B/C/D vs `calib020_0001`, `OUT=logs/eval/v12b_continuation_guardrail_from_calib020_0001_gates_400s`. Thresholds unchanged. No promotion unless all four pass.

## Interpreting v12b

- **A + B/C/D all pass** → promotion match.
- **A pass, C/D improve over v12 (fewer/less-severe overvalue rows)** → continuation guardrails are the missing pressure; consider margin tune or lock v12b.
- **A pass, C/D still break like v12/v10/v11** → the guardrail loss is being overwhelmed by the A-correction gradient in the shared final block even with continuation states covered → the diagnosed gradient conflict is real → **v13 = gradient-conflict handling (Option A)**.
- **A weakens** → the added continuation hinge is over-constraining; raise `--guardrail-margin` before abandoning.

Do NOT implement gradient projection (Option A) in v12b. It remains v13, gated on v12b failing while clearly showing conflicting gradients.
