# Targeted Value Calibration v15 — A Searched-Continuation Correction Design

**Status:** DESIGN — direction agreed by user 2026-07-09 (from the raw/MCTS drift diagnostic); awaiting written-spec review, then a Phase-0 concentration diagnostic, then writing-plans.
**Date:** 2026-07-09
**Follows:** the v14 value-adapter line (v14/v14b/v14c/v14d, all REJECTED). v14b is the best near-pass (A mean +0.026; B/C/D formally held). The raw/MCTS drift diagnostic (ledger, 2026-07-09) proved the remaining A miss is **MCTS/search amplification, not raw-value undercorrection**, closing the raw-surface experiment.

## Hypothesis / acceptance question

**Does correcting the searched-continuation (child/PV) values that MCTS backs up at the A pre-drop roots reduce the +0.20 search amplification enough to pass A (mean ≤ 0), while B/C/D hold — where every root-and-surface value correction (v2–v14d) failed because the search re-derives optimism from uncorrected deeper nodes?**

Evidence (diagnostic): raw A ≈ 0 at BASE (−0.015) but MCTS A = +0.257 — the overvalue is search-added (+0.272). v14b pushed raw A to −0.178 yet MCTS A = +0.026 (search re-amplified +0.204). The search-Δ varies with training (v14b +0.204 → v14d +0.121), so the lever is **reducing the backup**, not more raw correction (v14c/v14d were non-monotonic and broke D). The ledger's line-291 next hypothesis is tree/path-level retention; v15 is the A-specific, correction-targeted form of it.

## The change — correct the children, not the root; reuse the v6 machinery

v15 needs **no new training surface and no new loss**:
- **Surface = v14b, shipped:** `value_head.*` + `value_adapter.*`, bottleneck 32, scalar gate, encoder/policy/final-block/BN frozen, projection ON strength 1.0. (No `network.py`/`trainer.py` training changes.)
- **Objective = v12b, unchanged:** one-sided black-perspective guardrail hinge for B/C/D + `hard_value` A correction. The new A-continuation rows are just **`hard_value` rows applied to child/PV states** instead of the root — already supported by `build_calibration_position` and the loss path.
- **New code is read-only and phased.** **Phase 0** (this plan) is a concentration diagnostic — run BASE MCTS at the A roots, compute per-child visit/contribution shares + child raw values, emit only `v15prep_a_continuation_concentration.csv`; **no manifest, no replay JSONs, no loader touch.** **Phase 1** (deferred until the concentration read) emits the correction manifest via per-row child replay JSONs (Option 1, §1). The tree-walk copies `continuation_extraction`'s generic `_top_children`/`path_moves_of` helpers (`extract_continuations` raises for the A tag). Mirrors v7's manifest-only discipline.

| File | Change |
|---|---|
| `scripts/GPU/alphazero/build_v15_a_continuation_correction_manifest.py` (new) | **Phase 0 (this plan): read-only concentration diagnostic** — per A pre-drop root run BASE 400-sim MCTS, compute per-child `visit_share`/`child_contribution_share` + child raw values under BASE & v14b, emit only `v15prep_a_continuation_concentration.csv`. **Phase 1 (deferred):** emit `hard_value` correction rows (tag `black_predrop_continuation_correction`) via per-row child replay JSONs |
| child replay JSONs + manifest + schedule + `smoke_v15_*.py` | **Phase 1, deferred** — designed after the concentration read (few-row / semi / broad) |

Do NOT change: `continuation_extraction.py`, `_calibration_component_loss`, `alphazero_loss_batch`, `project_conflicting_gradient`, the v12b manifest/builder, the v14b training surface, the verifier, `mcts.py`, `train.py`.

## §0 Phase 0 — A-continuation concentration diagnostic (read-only, BEFORE finalizing the mechanism)

The builder first runs in diagnostic mode. For each of the 30 A pre-drop roots:
- run BASE 400-sim MCTS (gate-faithful: `add_noise=False`, same eval batch/stall config as the gate);
- record the top-visited child/PV states and, per child: `root_case_id`, `child_move`, `visit_share`, `child_depth`, `child_raw_black_value_BASE`, `child_raw_black_value_v14b`, `root_mcts_black_value`, and the child's contribution to the root backup.

**Decision rule (share of the positive backup mass explained by each root's top 1–3 children) — DECIDED:**
- **Concentrated (top 1–3 children ≥ 70%)** → few-row v15: emit only the high-contribution children, target −0.35.
- **Semi-concentrated (40–70%)** → include the top children **plus PV depth-1–2 rows**, tiered target (−0.35 high-contribution / −0.20 secondary per §1).
- **Broad (< 40%)** → do **not** run few-row v15; the optimism is too spread for a few child rows (re-search bypasses them) → write a separate tree/path-level correction design instead.

**Phase 0 is read-only and emits ONLY the concentration diagnostic CSV** `logs/eval/v15prep_a_continuation_concentration.csv` (per-root, per-child: `visit_share`, `child_contribution_share`, depth, raw values BASE/v14b, root MCTS value) + a one-paragraph classification read in the ledger. **It does NOT emit a training manifest and writes no child replay JSONs.** Manifest generation (Phase 1, §1) is deferred until the concentration result chooses few-row / semi-concentrated / broad. No training until this CSV is reviewed.

## §1 A-continuation correction rows (Phase 1, gated on §0)

- **Source rows:** the 30 A `black_predrop_correction` roots (the same probe manifest used by the gate).
- **Extraction:** BASE 400-sim MCTS per root → top child/PV states (count/depth set by §0). Reuse `continuation_extraction.py` (pure tree walk, already merged; do not modify).
- **Target (TIERED, contribution-aware — DECIDED):** define the tier *mechanically* from the Phase-0 metrics, not by hand:
  - `target_black_value = −0.35` if `visit_share ≥ 0.20 OR child_contribution_share ≥ 0.25`
  - `target_black_value = −0.20` otherwise
  - **Emit the low-visit (−0.20) rows ONLY if Phase 0 classifies the backup as *semi-broad*.** If Phase 0 is *concentrated*, emit **only** the high-contribution children at −0.35. If *broad*, do not run few-row v15 at all (§0).
  - Rationale: a continuation correction is more targeted than root correction but still writes into shared value space (v14d showed A pressure can damage D). A flat −0.35 across low-visit children risks over-flattening continuation states that don't actually drive the backup, so low-visit children get the softer −0.20 and only when the backup is spread.
- **Loss mode:** `hard_value` (value-only; no policy CE — keeps `calib_policy_ce/kl` at 0, a correctness check). Value-only by construction, so it composes with the v12b guardrail hinge and the projection surface exactly like the root A rows.
- **Child-board representation (Phase 1, Option 1 — no loader/training/loss change):** manifest rows have no `board_tensor` column (the loader reconstructs boards from `replay_path` + `position_ply`), and `hard_value` rows are forbidden from carrying `extra_moves_json`. So Phase 1 rows are represented as ordinary `hard_value` rows by writing **per-row child replay JSONs**: each child replay contains the original game prefix through the root position plus the selected MCTS child/PV path (`moves = original["moves"][:position_ply] + spec.path_moves`); the row points `replay_path` at that child replay and sets `position_ply = original position_ply + depth`, `side_to_move = continuation side`, `loss_mode = hard_value`, `target_black_value = tiered`, leaving the retention/policy/`extra_moves_json` columns blank. Training then sees a **normal `hard_value` row at the child board** — no `extra_moves_json`, no loader/loss/surface change. (`extract_continuations` raises for the A tag, so the builder copies its generic `_top_children`/`path_moves_of` helpers rather than calling it.)
- **Tag:** `black_predrop_continuation_correction` (distinct from the root `black_predrop_correction`, so the schedule can weight them independently and telemetry separates them).

## §2 Surface & objective (unchanged from v14b)

- Train from BASE `calib020_0001` with `--value-adapter --value-adapter-bottleneck-width 32 --train-value-head-and-value-adapter --post-opening-calibration-gradient-projection` (strength 1.0), `--guardrail-margin 0.10 --freeze-batchnorm-stats`. New checkpoint dir `checkpoints/alphazero-v15-a-continuation-correction-from-calib020-0001`.
- The v12b guardrail hinge protects B/C/D unchanged. Projection stays ON over `{value_head, value_adapter}` (the shipped v14b behavior) so the A-continuation correction still yields to the guardrails where they conflict.

## §3 Schedule (DECIDED)

**Shift one unit of A pressure from root to continuation — do NOT add total A-family pressure.** v14b already over-corrected raw A to −0.178 and the search re-amplified it; v14d *added* root-A pressure and got the bad result (A still failed, D severe broke). So the first v15 keeps total A-family pressure ≈ v14b and only changes *where it lands*:

```
black_predrop_correction=1,
black_predrop_continuation_correction=1,
goal_line_guardrail_retention=1,
old_post_opening_guardrail_retention=1,
old_post_opening_continuation_guardrail_retention=2,
red_predrop_guardrail_retention=1,
red_predrop_continuation_guardrail_retention=2
```

The B/C/D guardrail draws are **exactly** v14b/v12b (do not perturb the protected families). This gives a clean test: if v15 passes, it is because continuation correction changed the searched backup — not because more root A pressure was added. Only if A moves but **under-crosses** while B/C/D hold → v15b may try `black_predrop_continuation_correction=2`.

## §4 Gate, verifier, telemetry

- **Gate:** the same 400-sim A/B/C/D probes vs `calib020_0001`. **No promotion unless all four pass** (A mean ≤ 0 AND severe within cap; D severe = 0.0%).
- **Verifier:** the shipped `verify_value_head_and_adapter_checkpoint.py` (unchanged) — a v15 run still changes only `value_head.*` + `value_adapter.*`.
- **Telemetry:** existing — `calib_projection_scope=value_head_and_value_adapter`, `value_adapter_gate` off 0, projection conflict metrics, plus the new tag's draw count in `calib_n_drawn_by_tag`. No new telemetry fields.

## §5 Falsification & interpretation

- **Falsify fast on the 400-sim gates** (one iteration, same as v14b/v14d). 
- **A moves to ≤ 0 + B/C/D hold** → the search amplification was the blocker and continuation correction reaches it → promotion match.
- **A improves but doesn't cross, and §0 said "concentrated"** → the few-rows form is too shallow → escalate to tree/path-level (deeper/denser child coverage) as v15b.
- **A improves but D severe returns** → the continuation correction is leaking into red-pre-drop neighborhoods → tighten the A-family scoping / lower the tier, do not broaden.
- **A does not move at all** → correcting children still doesn't survive re-search (the sub-search below each child re-amplifies) → the honest conclusion is that 400-sim MCTS optimism at these pre-drop roots is not fixable by value calibration, and the effort redirects (search config, or training-data changes at the pre-drop family). This is the branch that would close the whole calibration line.

## §6 Do NOT (carry the do-not-repeat list forward)

No raw-surface cleanup (v8–v14d exhausted), no width 64, no per-channel gate yet, no stronger projection, no more root A-draw pressure, no margin tweak, no root-value-into-raw-head retention (do-not-repeat #9), no raw-teacher retention weight/schedule sweep (v4 exhausted), no root-only policy retention (v5 exhausted). v15 changes exactly one thing versus v14b: it adds **searched-continuation** (child) correction rows for the A family.
