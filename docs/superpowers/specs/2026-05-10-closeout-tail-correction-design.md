# Closeout Tail Correction — Design Spec

**Date:** 2026-05-10
**Status:** Drafted (awaiting user review)
**Predecessors:**
- [2026-05-03-goal-completion-diagnostics-design.md](2026-05-03-goal-completion-diagnostics-design.md) (Spec 1, shipped)
- [2026-05-05-goal-completion-inline-records-design.md](2026-05-05-goal-completion-inline-records-design.md) (Spec 1.5, shipped)
- [2026-05-06-goal-completion-policy-correction-design.md](2026-05-06-goal-completion-policy-correction-design.md) (Spec 2, shipped — production-verified on 130-139)

**Successor:** Recovery-aware training (Spec 4) only if Fix 3 diagnostics show a clear, addressable failure shape.

---

## 1. Goal

Eliminate the remaining closeout tail after Spec 2. Spec 2 fixed the average policy-prior issue (visit top-5 for endpoint completion rose to 86.3%, reducer to 88.2%; selected completion / reducer rate to ~75%). What remains is tail behavior, not coverage.

### 1.1 Diagnosis from 130-139

Per-ply inspection of the top 9 worst cases shows failures concentrate at **td=1 (one_move_win)**, not at td=2:

- 138/66 ply 54 (td=2): EC policy/visit rank **1**, selected → completes_endpoint, td 2→1 ✓
- 138/66 ply 56 (td=1): same closing move drops to **policy rank 33 / visit rank 173**, selected redundant_reinforcement
- 130/5 ply 56 (td=2): EC rank 1 selected → completes_endpoint, td 2→1 ✓
- 130/5 ply 58 (td=1): EC rank 5 in policy / **visit rank 154**, redundant
- 136/99 ply 78 (td=2): EC rank 1 selected → td 2→1 ✓
- 136/99 ply 80 (td=1): EC rank 6 / visit rank 87, redundant

The td=2 → td=1 transition closes correctly. The td=1 → win transition fails for 6–113 winner moves at q ≈ 0.96+. Same mechanism unifies the high-value-drift cluster (delayed wins) and the state_cap cluster (component eventually broken).

A second cluster (3 of 15 worst cases, including the 224-ply outlier 131/79) is structurally different: dominant component becomes unavailable within ~4 plies of detection, q collapses, the reducer disappears from search too. This is recovery-after-dominant-lost — not the same fix.

### 1.2 Design intent

Three fixes plus a recovery diagnostic, in order of safety and confidence:

1. **Fix 0 — bulk td-before diagnostic.** Prove the failure distribution by `total_goal_distance_before ∈ {1,2,3}` before changing search.
2. **Fix 1 — td=1 root visit forcing.** At MCTS root only, when `total_goal_distance == 1` and at least one endpoint-completion candidate exists, force each candidate to receive ≥ N visits before final selection. PUCT's Q backup carries +1 through subsequent sims, naturally redirecting visits to the winning move.
3. **Fix 2 — narrow selection tie-break (opt-in, off by default).** For positions where a closeout candidate is already in visit top-K but argmax selection chose redundant/off-chain. Defaults off until Fix 1 results are evaluated.
4. **Fix 3 — recovery diagnostic.** Per-game `recovery_event` classification. Diagnostics only; no trainer changes.

**Replay analyzer is in scope and required.** Fix 0 and Fix 3 land in `scripts/twixt_replay_analyzer.py` and are required to produce the §8 success-criteria report. Fix 1's telemetry block also surfaces in the per-iteration sidecar consumed by the analyzer, and the analyzer must learn to summarize it in `report_<range>.txt`. The Fix 1 treatment run cannot be evaluated against §8 metrics without the analyzer changes shipped first.

**Out of scope:**
- Increasing `conversion_policy_loss_weight` or widening to td=3
- New training data sources, curated probes
- Force-selecting moves outside MCTS (a known-winning move could be just played, but we keep the network in the loop on purpose)
- Recovery training of any kind (only diagnostics in this spec)

---

## 2. Architecture & data flow

### 2.1 Affected modules

```
scripts/twixt_replay_analyzer.py
    + td_closeout_breakdown aggregation (Fix 0)
    + recovery_events aggregation and CSV (Fix 3)
    + closeout_td1_visit_forcing sidecar read + report section (Fix 1 telemetry)
    + report sections for all of the above

scripts/GPU/alphazero/mcts.py
    + force_root_visits()                          (Fix 1)
    + apply_closeout_selection_tiebreak()          (Fix 2, opt-in)
    + telemetry accumulators on MCTS instance

scripts/GPU/alphazero/self_play.py
    + call into force_root_visits() after Dirichlet, before main sim loop (Fix 1)
    + call into apply_closeout_selection_tiebreak() after search, before select_move (Fix 2)
    + sidecar telemetry emit                       (Fix 1, Fix 2)

scripts/GPU/alphazero/train.py
    + new CLI flags                                (Fix 1, Fix 2)
```

No changes to `connectivity_diagnostics.py`, `conversion_loss.py`, `closeout_diagnostics.py`, or the network architecture.

### 2.2 Per-ply trigger flow (Fix 1, Fix 2)

```
self_play.play_game()
  for each ply:
    gc_state_full = compute_goal_completion_state(...)           # already done at self_play.py:732
    root = mcts.search_from_root(root, add_noise, ply)
      [inside search_from_root]
      _expand(root)
      _add_dirichlet_noise(root, ply)
      if Fix-1 enabled and gc_state_full.total_goal_distance == 1
         and gc_state_full.endpoint_completion_moves:
          force_root_visits(root, candidates, n)                 # NEW
      main simulation loop (consumes remaining budget)
    visit_counts = collected from root
    if Fix-2 enabled and qualifying:
      visit_counts, override_record = apply_closeout_selection_tiebreak(...)
    move = mcts.select_move(visit_counts, ply)
```

### 2.3 Training-target inclusion

Forced visits are **included** in the `visit_counts` returned by `search_from_root` and therefore in `get_policy_target`. This is intentional: the policy learns the closing pattern through repeated exposure to the forced visit distribution. No `raw_visit_counts` channel is added; if a future ablation needs to compare, it can be done by toggling the flag.

---

## 3. Fix 0 — bulk td-before diagnostic

Pure analyzer change. Reads existing `goal_completion_diagnostics` arrays from per-game JSON files; no new trainer state required.

### 3.1 Aggregation

For each ply record where `goal_completion.total_goal_distance_before` is set and `side_to_move == detected_player`, bucket by `td_before ∈ {1, 2, 3}` and accumulate:

| Field | Definition |
|-------|-----------|
| `records` | count of qualifying rows |
| `high_value_records` | rows with `root_summary.q_value >= 0.95` |
| `selected_completes_endpoint_rate` | fraction with `primary_class == completes_endpoint` |
| `selected_reduces_distance_rate` | fraction with `primary_class == reduces_total_goal_distance` |
| `selected_redundant_rate` | fraction with `primary_class == redundant_reinforcement` |
| `selected_off_chain_rate` | fraction with `primary_class == off_chain` |
| `selected_other_rate` | fraction with `primary_class == other` |
| `endpoint_completion_exists_rate` | fraction where `endpoint_completion_ranking.best_policy_rank is not None` |
| `endpoint_policy_top1_rate` / `_top5_rate` / `_top20_rate` / `_gt20_rate` | rank-bucket fractions, denominator = `endpoint_completion_exists` |
| `endpoint_visit_top1_rate` / `_top5_rate` / `_top20_rate` / `_gt20_rate` | same for visit ranks |
| `distance_reducer_exists_rate` | analogous to endpoint, using `distance_reducing_ranking` |
| `reducer_policy_top{1,5,20}_rate`, `reducer_policy_gt20_rate` | rank buckets |
| `reducer_visit_top{1,5,20}_rate`, `reducer_visit_gt20_rate` | rank buckets |

Aggregated structure attached at `summary.goal_completion.td_closeout_breakdown` in `summary_<range>.json`.

### 3.2 Report section

Added to `report_<range>.txt` under the existing Goal-Completion / Conversion Diagnostics block:

```
Closeout breakdown by total_goal_distance
=========================================
td=1:  records=N  high_value=M
  selected: complete=X%  reduce=Y%  redundant=Z%  off-chain=W%  other=V%
  endpoint exists: P%   policy top5=A%  visit top5=B%  visit >20=C%
  reducer  exists: P'%  policy top5=A'% visit top5=B'% visit >20=C'%
td=2:
  ...
td=3:
  ...
```

### 3.3 CSV (companion)

New file `goal_completion_td_breakdown_<range>.csv` with one row per `td_before` value, columns matching the aggregation table above.

### 3.4 Decision rule

After running Fix 0 on 130-139:

- If td=1 has high redundant/off-chain rate AND endpoint visit top-5 is low (< 50%), Fix 1 is justified.
- If td=1 endpoint visit top-5 is already high (≥ 80%) but selection still drifts, Fix 2 is the right tool.
- If td=1 visit top-5 is low but td=2 visit top-5 is also low, revisit Spec 2 knobs before adding search-side fixes.

---

## 4. Fix 1 — td=1 root visit forcing

### 4.1 Trigger condition

At MCTS root, after `_expand(root)` and `_add_dirichlet_noise(root, ply)`:

```python
trigger_fires = (
    self.config.closeout_td1_visit_forcing_enabled
    and gc_state_full is not None
    and gc_state_full.get("total_goal_distance") == 1
    and gc_state_full.get("endpoint_completion_moves")
    and (
        not self.config.closeout_td1_require_high_value
        or root.q_value >= self.config.closeout_td1_high_value_threshold
    )
)
```

`gc_state_full` is the `compute_goal_completion_state` output already produced at `self_play.py:732`. Passed into `search_from_root` as a new optional kwarg.

### 4.2 Forcing mechanism

```python
def force_root_visits(
    self,
    root: MCTSNode,
    candidate_moves: List[Tuple[int, int]],
    min_visits: int,
    max_candidates: int,
) -> int:
    """Run forced MCTS sims at the root targeted at specific child moves.

    Each forced sim:
      1. Selects the targeted child unconditionally (bypass PUCT at root for this sim).
      2. Descends the rest of the tree by normal PUCT.
      3. Expands/evaluates the leaf and backs up the value through the entire path.

    Forced sims consume from the same n_simulations budget as normal sims.
    Returns the number of forced sims actually executed.
    """
    moves = candidate_moves[:max_candidates]
    forced_count = 0
    for move in moves:
        move_id = encode_move(*move)
        for _ in range(min_visits):
            if forced_count >= self.config.n_simulations:
                return forced_count
            self._run_one_sim_with_root_override(root, move_id)
            forced_count += 1
    return forced_count
```

**Implementation requirement: reuse the existing single-simulation path.** The current MCTS main loop has batching, waiter-list coordination, virtual-visit penalties, and stall-flush logic. Introducing a parallel synchronous eval path risks accidentally bypassing those assumptions (e.g., double-counting visits, racing on shared state, breaking backup semantics).

Preferred shape: identify or extract a shared single-simulation helper from the existing `search_from_root` body and add a `root_move_override: Optional[int]` kwarg. Forced sims call that helper with the override set; normal sims call it with `None`. If the existing loop is too entangled to refactor cleanly in this spec, extract the minimum (descent + expand + backup) and leave the batching scaffolding to the main loop. Do NOT duplicate leaf-eval, backup, virtual-visit, or batching semantics unless a unit test proves equivalence (see §9.1).

After the root child is chosen by override, the rest of the descent is normal PUCT.
- Children corresponding to candidate moves are expanded by `_expand` on first visit, so their priors and Q values are populated correctly.
- After forcing completes, the main simulation loop runs `n_simulations - forced_count` regular PUCT sims. PUCT with Q=+1 backed up through the forced children naturally directs subsequent sims toward those children.

### 4.3 Budget accounting

`n_simulations` (default 400 per training command) stays constant. Forced sims are consumed from the same budget, so the per-position compute does not grow. If a trigger fires with N=8 and max_candidates=4, up to 32 of the 400 sims are forced, leaving ≥ 368 for normal PUCT.

### 4.4 CLI flags

| Flag | Default | Notes |
|------|---------|-------|
| `--closeout-td1-visit-forcing-enabled` | `False` | Defaults off until first eval run |
| `--closeout-td1-min-visits` | `8` | Visits forced per candidate |
| `--closeout-td1-max-forced-moves` | `4` | Caps `len(candidate_moves)` |
| `--closeout-td1-require-high-value` | `False` | If True, gate on `root.q_value` |
| `--closeout-td1-high-value-threshold` | `0.95` | Only used if `require-high-value` |

### 4.5 Telemetry

Per-iteration sidecar at `closeout_td1_visit_forcing`:

```json
{
  "enabled": true,
  "min_visits": 8,
  "max_forced_moves": 4,
  "require_high_value": false,
  "high_value_threshold": 0.95,
  "positions_triggered": 123,
  "positions_skipped_no_candidates": 0,
  "positions_skipped_high_value_gate": 0,
  "forced_sims_total": 984,
  "selected_forced_move_count": 95,
  "selected_forced_move_rate": 0.772,
  "post_force_endpoint_visit_top1_rate": 0.812,
  "post_force_endpoint_visit_top5_rate": 0.940
}
```

The MCTS instance accumulates these counters; the self-play worker drains them into the stats sidecar at the end of each iteration.

### 4.6 Scope

- **Self-play only.** Eval games and inference do not trigger the rule. This isolates the experiment from production performance reporting until the policy has internalized the behavior.
- **Root only.** Forcing inside internal tree nodes is out of scope for this spec.

---

## 5. Fix 2 — narrow selection tie-break (opt-in)

### 5.1 Trigger condition

After `search_from_root` returns and before `select_move`:

```python
tiebreak_fires = (
    self.config.closeout_selection_tiebreak_enabled
    and gc_state_full is not None
    and gc_state_full.get("total_goal_distance") <= self.config.closeout_selection_tiebreak_max_distance
    and root.q_value >= self.config.closeout_selection_tiebreak_min_value
    and any_closeout_candidate_in_visit_topk(
        visit_counts, gc_state_full, self.config.closeout_selection_tiebreak_topk
    )
    and primary_class_of_argmax(visit_counts, ...) in {"redundant_reinforcement", "off_chain", "other"}
)
```

### 5.2 Override priority

1. Endpoint completion move with the highest visit count (tie: random by per-game RNG).
2. Distance-reducing move with the highest visit count.
3. Otherwise no override.

Subject to `min_share`: the chosen candidate must have visit share ≥ `--closeout-selection-tiebreak-min-share`.

### 5.3 Implementation

Returns an updated `visit_counts` dict in which the selected closeout candidate's visit count is bumped to one above the previous argmax. `select_move` then proceeds normally. At temp_low this is equivalent to a hard override (argmax now picks the candidate). At temp_high (still inside opening_noise_ply) the bump shifts the softmax weight but does not force selection — by design, since temp_high windows preserve exploration. The trigger condition still excludes the opening (td<=2 with high q never fires in the first few plies).

### 5.4 CLI flags

| Flag | Default | Notes |
|------|---------|-------|
| `--closeout-selection-tiebreak-enabled` | `False` | Stays off until Fix 1 evaluated |
| `--closeout-selection-tiebreak-max-distance` | `2` | td_before upper bound |
| `--closeout-selection-tiebreak-topk` | `5` | Candidate must be in visit top-K |
| `--closeout-selection-tiebreak-min-value` | `0.95` | Root q gate |
| `--closeout-selection-tiebreak-min-share` | `0.05` | Visit share floor for the override target |

### 5.5 Telemetry

Per-iteration sidecar at `closeout_selection_tiebreak`:

```json
{
  "enabled": true,
  "eligible_positions": 42,
  "overrides": 17,
  "override_rate": 0.405,
  "override_to_endpoint": 12,
  "override_to_reducer": 5,
  "would_have_selected_redundant": 10,
  "would_have_selected_off_chain": 6,
  "would_have_selected_other": 1
}
```

---

## 6. Fix 3 — recovery diagnostic

Pure analyzer addition. No trainer changes, no new training targets.

### 6.1 Event criterion

A game contributes a `recovery_event` row when its `goal_completion_record` satisfies any of:

- `winner_moves_with_dominant_unavailable >= 10`
- `meta.reason == "state_cap"` AND `goal_completion_record.detected == True`
- `meta.reason == "adjudicated"` AND `winner_moves_with_dominant_unavailable >= 5`

### 6.2 Per-event fields

| Field | Source |
|-------|--------|
| `iteration` | game meta |
| `game_id` | record |
| `winner` | record |
| `detected_player` | record |
| `first_detection_ply` | `first_dominant_unclosed_ply` |
| `first_unavailable_ply` | first ply in `goal_completion_diagnostics` where `side_to_move == detected_player` and `goal_completion.total_goal_distance_before > 2` (i.e., component effectively unavailable) |
| `dominant_unavailable_moves` | `winner_moves_with_dominant_unavailable` |
| `latest_largest_component_size` | last `goal_completion.largest_component_size` recorded |
| `latest_total_goal_distance` | last `goal_completion.total_goal_distance_before` recorded |
| `q_at_first_unavailable` | `root_summary.q_value` at `first_unavailable_ply` |
| `q_at_terminal` | `meta.final_root_value` |
| `selected_class_counts_after_first_unavailable` | dict, summed from per-ply records ≥ `first_unavailable_ply` |
| `eventual_outcome` | `win` / `state_cap` / `adjudicated` from `meta.reason` |
| `recovery_class` | bucket assignment per §6.3 |

### 6.3 Recovery class buckets

Assigned in priority order:

| Bucket | Rule |
|--------|------|
| `lost_then_recovered` | `eventual_outcome == "win"` AND `winner_moves_with_dominant_unavailable >= 10` AND later record shows `total_goal_distance <= 2` again |
| `lost_then_won_late` | `eventual_outcome == "win"` AND `conversion_delay_winner_moves >= 30` AND not `lost_then_recovered` |
| `lost_then_state_cap` | `eventual_outcome == "state_cap"` |
| `lost_and_value_collapsed` | `q_at_first_unavailable >= 0.9` AND `q_at_terminal <= 0.5` |
| `lost_but_value_stayed_high` | `q_at_first_unavailable >= 0.9` AND `q_at_terminal >= 0.9` AND not in above |

### 6.4 Outputs

- CSV: `recovery_events_<range>.csv`, one row per event, columns as in §6.2 plus bucket
- Report section in `report_<range>.txt`:

```
Recovery / dominant-component-lost diagnostics
===============================================
Events: N
By outcome:
  lost_then_recovered:        a
  lost_then_won_late:         b
  lost_then_state_cap:        c
  lost_and_value_collapsed:   d
  lost_but_value_stayed_high: e
Median dominant_unavailable_moves: M
Median delay (winner_moves):       D
```

Diagnostic-only; no training signal derived. The aggregate distribution drives whether Spec 4 needs a recovery training mechanism.

---

## 7. Implementation order

Strict ordering — each phase gates the next:

| Phase | Deliverable | Runs / gate |
|-------|------------|-------------|
| 1 | **Fix 0 + Fix 3 analyzer-only** (no trainer code touched) | re-run analyzer on 130-139; gates Phase 2 by confirming td=1 dominates the failure distribution |
| 2 | **Fix 1 MCTS code** — reusing the existing single-simulation / backup path (see §4.2); telemetry accumulators; unit tests including the equivalence test in §9.1 | unit tests pass; smoke run on a constructed td=1 game |
| 3 | **Fix 1 self-play wiring + CLI flags** — defaults off | smoke training run with `--closeout-td1-visit-forcing-enabled` produces well-formed telemetry |
| 4 | **Analyzer additions for Fix 1 telemetry** — read `closeout_td1_visit_forcing` from sidecars, summarize trigger frequency + post-force endpoint visit top-5 in `report_<range>.txt` | analyzer on a 1–2 iteration smoke run shows the new section |
| 5 | **Treatment run** — 10 iterations from `model_iter_0139.safetensors` with Fix 1 enabled (defaults `min_visits=8`, `max_forced_moves=4`); Fix 2 stays off | iters 140-149 complete; sidecars and games captured |
| 6 | **Evaluate against §8 success criteria** using the analyzer | report_140-149 vs report_130-139; telemetry shows trigger fired meaningfully and endpoint visit top-5 rose |
| 7 | **Decision point: Fix 2** — enable only if residual top-K-visible drifts remain | conditional second treatment run |
| 8 | **Recovery training** deferred to Spec 4 brainstorm, gated on Fix 3 outputs | not in this spec |

---

## 8. Success criteria

Compared to 130-139 baseline:

| Metric | 130-139 baseline | Target after Fix 1 |
|--------|------------------|---------------------|
| td=1 selected `completes_endpoint` rate | (filled in after Fix 0 first run) | substantial increase |
| td=1 endpoint visit top-5 rate | (filled in after Fix 0 first run) | sharp increase (≥ 90%) |
| Conversion delay ≥ 10 plies (game count) | 21 | ≤ 10 |
| Conversion delay ≥ 20 plies (game count) | 7 | ≤ 3 |
| state_cap after detection | 4 | ≤ 1 |
| High-value delayed records (policy/MCTS bucket) | 198 | ≤ 80 |
| Game-level high-value delayed | 46 | ≤ 20 |
| Avg plies per game | 53.0 | unchanged ± 5% |
| Probe sign-correct rate | baseline | unchanged ± 2pp |

If td=1 visit top-5 rises sharply but the game-count tails do not move, the residual cause is selection-time (Fix 2) — enable Fix 2 and re-run.

---

## 9. Testing

### 9.1 Unit tests

- `test_td_closeout_breakdown.py` — Fix 0 aggregation over synthetic per-ply records covering all bucket transitions
- `test_force_root_visits.py` — Fix 1 trigger fires only when td=1 and EC moves exist; correct number of forced sims
- `test_forced_root_visit_matches_normal_backup_semantics.py` — **equivalence test.** A single forced sim that overrides the root child to `move_id` produces the same `child.visits` increment and the same backup of leaf value along the path as a hand-constructed normal sim that selects `move_id` by PUCT. Run with a deterministic stub eval and verify identical post-state.
- `test_closeout_selection_tiebreak.py` — Fix 2 override priority; min-share gate; argmax-of-closeout selection
- `test_recovery_events.py` — Fix 3 bucket assignment against fixture records (one per bucket)

### 9.2 Integration smoke test

`scripts/GPU/alphazero/smoke_closeout_td1_visit_forcing.py` — runs a short self-play game with Fix 1 enabled, asserts the trigger fires at least once on a constructed td=1 position and that the resulting telemetry block is well-formed.

### 9.3 Treatment run

Primary evaluation uses the existing 130-139 Spec 2 block as the baseline — its metrics are already produced via `Replays/130-139_Replay`. No separate control rerun is needed for the first read of Fix 1's effect.

Run one treatment block from `model_iter_0139.safetensors` for 10 iterations (140-149) with Fix 1 enabled (`--closeout-td1-visit-forcing-enabled --closeout-td1-min-visits 8 --closeout-td1-max-forced-moves 4`) and all other Spec 2 knobs unchanged. Compare `report_140-149.txt` against `report_130-139.txt` for the §8 metrics.

A same-checkpoint control rerun from `model_iter_0139.safetensors` with Fix 1 disabled is optional and only worth running if the treatment result is ambiguous — e.g., if §8 metrics shift in inconsistent directions or if `closeout_td1_visit_forcing.positions_triggered` indicates the trigger barely fired.

---

## 10. Risks and open considerations

### 10.1 N=8 forced visits may not flip selection alone

At temp_low, `select_move` is argmax-by-visit-count. If a redundant move has already accumulated > 32 visits in early sims of past plies, forcing only 32 visits onto the endpoint move would not flip it. But Fix 1 runs forcing **before** the main simulation loop. Post-forcing, the endpoint child has Q=+1 backed up; PUCT for remaining ~368 sims will preferentially exploit it.

If the first run shows the trigger fires but `selected_forced_move_rate` stays low, raise `--closeout-td1-min-visits` to 16 (no code change needed) before considering more invasive fixes.

### 10.2 Forced visits feed the policy target

Intentional and explicitly desired. The self-correcting effect is the point: over iterations the policy gradually moves probability mass onto the closing move and the forcing becomes less needed. If a future ablation needs unforced visit counts, a `raw_visit_counts` channel can be added later — not in this spec.

### 10.3 Trust in `endpoint_completion_moves` at td=1

The same `compute_goal_completion_state` already drives the Spec 2 conversion aux loss target. If it ever returns a non-winning move as an endpoint completion, Spec 2's training signal would already be wrong. Fix 1 inherits that trust level; no new validation needed.

### 10.4 Sampled diag records limit recovery detail

The embedded `goal_completion_diagnostics` array caps at `max_records_per_game=64` (see `goal_completion_diagnostics_meta`). The 224-ply outlier 131/79 had only 14 of 113 winner moves recorded. Fix 3's recovery classification falls back on whole-game aggregates from `goal_completion_record` for fields like `winner_moves_with_dominant_unavailable`, which are computed before sampling — these stay accurate. The per-ply selected-class detail in §6.2 will be partial for long games. Acceptable for first-pass classification.

### 10.5 Fix 2 may be a near-no-op

Per the 130-139 worst-cases data, only 1 of 15 cases (137/71) sits in the regime where the closeout candidate is in policy AND visit top-5 while selection still drifts. Fix 2's expected impact is small. It ships in this spec as opt-in scaffolding so it's a flag-flip away if Fix 1's residual is dominated by top-K-visible cases. If Fix 0's td=1 breakdown shows that almost all failures have the closeout move outside visit top-5, Fix 2 stays off permanently.

### 10.6 Self-play-only scope of Fix 1

Eval games and the production export path do not trigger Fix 1. This is deliberate: it isolates the experiment and preserves measurement integrity for non-self-play surfaces. If after several iterations the policy has internalized the closing behavior, the flag can be left off for production. If it has not, that itself is a signal worth surfacing — and a candidate for Spec 4 work.
