# Goal-Completion Policy Correction — Design Spec

**Date:** 2026-05-06
**Status:** Approved (brainstorm complete)
**Predecessors:**
- [2026-05-03-goal-completion-diagnostics-design.md](2026-05-03-goal-completion-diagnostics-design.md) (Spec 1, shipped)
- [2026-05-05-goal-completion-inline-records-design.md](2026-05-05-goal-completion-inline-records-design.md) (Spec 1.5, shipped — production-verified on 110-119)

**Successor:** Spec 3 may add curated conversion-probe mining (deferred Phase 5) and/or recovery-bucket aux loss after Spec 2's policy correction is measured.

---

## 1. Goal

Add conversion-aware policy training for dominant-unclosed TwixT positions. Use inline goal-completion diagnostics (Spec 1.5) to identify endpoint-completing and distance-reducing moves, then train the policy head to prefer them directly.

### 1.1 Diagnosis from 110-119

The 110-119 run with Spec 1.5 inline records separates value quality from policy conversion quality:

**Value head is fine.** After detection: `max search_score p50=1.00, p90=1.00`; `mean p50=0.99, p90=1.00`. The model is confident the position is winning.

**Policy head does not rank closeout moves highly.** Phase 3 closeout policy_mcts diagnostics (994 games / 4554 records):

```
Endpoint completion:    policy top1: 0.0%   policy top5: 0.0%
                        visit  top1: 47.2%  visit  top5: 70.4%

Distance reducing:      policy top1: 0.0%   policy top5: 0.0%
                        visit  top1: 55.8%  visit  top5: 79.8%
```

**Selected-move behavior** confirms it:

```
Selected move after detection:
  completes endpoint: 34.2%
  reduces distance:   21.8%
  redundant:          31.7%
  off-chain:          10.7%
  other:               1.6%
```

About 56% of selected closeout moves directly complete or reduce distance; about 42% are redundant or off-chain. MCTS often recovers the closeout move through search; the raw policy prior does not.

### 1.2 Design intent

Spec 2 supervises the policy head directly with an auxiliary cross-entropy loss against an explicit conversion target (completion-weighted higher than reducer-weighted), applied only on closeout-eligible positions. Optional bounded replay sampling boost amplifies exposure. A separate telemetry-only "recovery / extreme-closeout-drift" bucket measures the harder failure mode where the dominant component breaks before completion.

**Out of scope:** value-head changes, MCTS prior pre-bias, recovery-bucket loss, curated probe-set mining (deferred Phase 5).

---

## 2. Architecture & data flow

### 2.1 Five-stage flow

```
[self-play worker — play_game()]
  per ply, after MCTS + move-selection, before apply_move:
    gc_state_cheap = compute_goal_completion_state(state, side, enumerate_moves=False)
    needs_full = (Spec 1.5 conditions) OR
                 (effective_conversion_enabled AND
                  total_goal_distance <= conversion_max_total_goal_distance)
    gc_state_full = compute_goal_completion_state(..., enumerate_moves=True) if needs_full
    --- existing tracker.observe_pre_move(...) and Phase 3 emit unchanged ---

    # NEW: build conversion metadata from gc_state_full
    if effective_conversion_enabled and is_conversion_eligible(gc_state_full, ...):
        position.conversion = {
            "version": 1,
            "total_goal_distance": ...,
            "largest_component_size": ...,
            "endpoint_completion_moves": [list(m) for m in ...],
            "distance_reducing_moves":   [list(m) for m in ...],
            "conversion_category": ...,
            "selected_primary_class": ...,   # filled post-classify; telemetry only
        }

[saver / IPC]
  PositionRecord.to_dict / from_dict round-trips the conversion field.
  GameComplete carries it through worker → trainer pickle.
  Per-game JSONs unchanged — conversion lives only in PositionRecord buffer.

[trainer]
  ReplayBuffer maintains a parallel eligible-index pool, updated on add/evict.
  ReplayBuffer.sample uses stratified draw when boost > 1.0; boost == 1.0
  short-circuits to pure uniform.
  alphazero_loss_batch builds aux_target / aux_mask, computes aux CE against
  the same masked log_probs used by policy CE, mean over eligible only,
  added to total_loss with conversion_loss_weight.

[trainer per-iter sidecar writer]
  + conversion_training block (loss + sample stats + drawn-vs-seen invariant)
  + recovery_or_extreme_closeout_drift block (telemetry only)

[analyzer]
  Spec 1 + 1.5 paths unchanged. The existing policy_mcts_summary report is the
  primary measurement surface for aux-loss effect. Two new minor sections:
  - Conversion-training trend (sidecar roll-up)
  - Recovery / extreme drift trend (sidecar roll-up)
  Plus per-iter CSVs paralleling the existing forced_probe_by_iter pattern.
```

### 2.2 BFS cost invariant

> **At most one full goal-completion computation per ply.** The same `gc_state_full` object is reused by closeout diagnostics, the goal-completion tracker, and conversion metadata. Spec 2 may force a full BFS on plies where Spec 1.5's emit/tracker would not have, but never duplicates one.

When `effective_conversion_enabled=False`, **zero conversion-specific BFS** runs. (Sample boost does not trigger BFS independently — `PositionRecord.conversion` is populated only when `effective_conversion_enabled` is true; see Section 9.4.)

### 2.3 Eligibility guard at the data boundary

`PositionRecord.conversion` is populated only when:
- `effective_conversion_enabled` is true (single source: `--conversion-policy-loss-enabled`), AND
- `is_conversion_eligible(gc_state_full, ...)` returns true.

Eligibility predicate (Section 4) requires non-empty `endpoint_completion_moves` or `distance_reducing_moves`. Empty-target positions never enter the auxiliary loss path.

### 2.4 Pre-move semantics

`PositionRecord.conversion` describes the **pre-move** closeout state of `PositionRecord.to_move` for that exact training position. It is never computed after `state.apply_move(selected_move)`. This invariant is anchor-tested.

### 2.5 Game-level vs per-position metadata separation

`root_value_high_but_delayed` and `dominant_unavailable` are **game-level / mining metadata**, computed at game finalization. They live in `goal_completion_record` (Spec 1.5) and feed the recovery bucket. They are explicitly **not** part of `PositionRecord.conversion` — the auxiliary target depends only on the position's pre-move closeout state, never on game-window context that may not yet exist.

---

## 3. Cost summary

| Configuration | Conversion BFS overhead | Tagging overhead |
|---|---|---|
| Default (loss disabled) | None | None — `PositionRecord.conversion` always None |
| Loss enabled, Spec 1.5 emit on, position eligible | None — reuses `gc_state_full` | One dict allocation + move-list copy |
| Loss enabled, Spec 1.5 emit off, position eligible | One full BFS (would not have run otherwise) | Same |
| Loss enabled, position ineligible | None | None |

The "Spec 1.5 emit off + loss enabled" path is the one new cost surface. It runs full BFS only on plies where `total_goal_distance ≤ conversion_max_total_goal_distance` (default 2). On a typical 1000-game corpus, that's a small fraction of all plies.

---

## 4. Eligibility predicate

### 4.1 Signature (single threshold knob)

```python
# scripts/GPU/alphazero/conversion_loss.py  (new module)

def is_conversion_eligible(
    gc_state_full: Optional[dict],
    *,
    max_total_goal_distance: int,    # CLI default 2
    min_component_size: int,         # CLI: --goal-completion-min-component-size, default 8
) -> bool:
    """Predicate that determines whether a pre-move state qualifies the
    side-to-move's PositionRecord for conversion auxiliary loss.

    Pure dict math — no BFS. Defends against missing/None fields.
    """
    if gc_state_full is None:
        return False
    total = gc_state_full.get("total_goal_distance")
    comp_size = gc_state_full.get("largest_component_size")
    if total is None or comp_size is None:
        return False
    if total > max_total_goal_distance:
        return False
    if comp_size < min_component_size:
        return False
    completion = gc_state_full.get("endpoint_completion_moves") or []
    reducing = gc_state_full.get("distance_reducing_moves") or []
    if not completion and not reducing:
        return False
    return True
```

### 4.2 Side scope

The predicate applies from the **side-to-move perspective**. Eligibility is shape-based, not outcome-based. Positions where the eventual loser had a closeout opportunity and failed are valid training signal — that is exactly the case Spec 2 fixes.

The predicate is symmetric across red/black: any side whose pre-move state matches gets tagged. Winner/loser outcome is reported in per-game records (Spec 1.5) but is not part of training eligibility.

### 4.3 Threshold experiments

| Experiment | `max_total_goal_distance` |
|---|---|
| First run (strongest signal) | 2 |
| Widened (broader-conversion category) | 3 |

Default 2. Single CLI flag (`--conversion-max-total-goal-distance`) widens; no second threshold knob.

---

## 5. PositionRecord schema and IPC

### 5.1 PositionRecord field addition

```python
# scripts/GPU/alphazero/self_play.py

@dataclass
class PositionRecord:
    # ... existing fields ...
    conversion: Optional[dict] = None   # NEW

    # to_dict / from_dict updated to round-trip the field
```

### 5.2 `conversion` dict schema (version 1)

```json
{
  "version": 1,
  "total_goal_distance": 1,
  "largest_component_size": 12,
  "endpoint_completion_moves": [[0, 8]],
  "distance_reducing_moves": [[0, 8], [22, 4]],
  "conversion_category": "two_endpoint_closeout_2ply",
  "selected_primary_class": "redundant_reinforcement"
}
```

| Field | Type | Notes |
|---|---|---|
| `version` | int | 1. Bump on rename / removal / semantic change. |
| `total_goal_distance` | int | 0..max_total_goal_distance |
| `largest_component_size` | int | ≥ min_component_size |
| `endpoint_completion_moves` | list[list[int]] | JSON-friendly `[[r,c], ...]`. Trainer converts to tuples once at batch-build. |
| `distance_reducing_moves` | list[list[int]] | Same encoding. |
| `conversion_category` | str | e.g., `two_endpoint_closeout_2ply`. |
| `selected_primary_class` | str \| null | **Telemetry/mining only.** Aux target builds from move sets regardless. |

### 5.3 IPC carrying

`GameComplete` (frozen IPC dataclass) flows `positions: tuple[dict, ...]` already. The new `conversion` key flows through verbatim — no IPC schema bump beyond the additive field.

Workers and trainer must restart together when this change deploys (same constraint as Spec 1.5 §9.4).

### 5.4 Saver / persisted-buffer compatibility

- **Per-game JSON** (`game_saver.py`): no change. Conversion metadata is an in-buffer training artifact; never persisted to per-game JSON.
- **Replay-buffer save/load**: `to_dict`/`from_dict` round-trip handles the new key. Pre-Spec-2 buffers load with `conversion=None` for every position — no errors, aux loss treats them as ineligible.

### 5.5 Self-play attach point

In `play_game()`'s ply loop, after the existing tracker observation and after `classify_selected_conversion_move` runs:

```python
# Build conversion_meta from the SAME gc_state_full used by tracker / Phase 3.
conversion_meta = None
if effective_conversion_enabled and gc_state_full is not None and is_conversion_eligible(
    gc_state_full,
    max_total_goal_distance=args.conversion_max_total_goal_distance,
    min_component_size=args.goal_completion_min_component_size,
):
    conversion_meta = {
        "version": 1,
        "total_goal_distance":       gc_state_full["total_goal_distance"],
        "largest_component_size":    gc_state_full["largest_component_size"],
        "endpoint_completion_moves": [list(m) for m in (gc_state_full.get("endpoint_completion_moves") or [])],
        "distance_reducing_moves":   [list(m) for m in (gc_state_full.get("distance_reducing_moves") or [])],
        "conversion_category":       gc_state_full.get("category"),
        "selected_primary_class":    None,
    }

if classification_result is not None and conversion_meta is not None:
    conversion_meta["selected_primary_class"] = classification_result.get("primary_class")

position = PositionRecord(..., conversion=conversion_meta)
```

The `gc_state_full` reads on the right-hand side are the same object the tracker and Phase 3 emit consume — fulfilling the "at most one full BFS per ply" invariant.

---

## 6. Auxiliary loss formulation

### 6.1 Per-position target construction

```python
# scripts/GPU/alphazero/conversion_loss.py

def build_conversion_target(
    legal_moves: list[tuple[int, int]],
    completion_moves: set[tuple[int, int]],
    reducing_moves: set[tuple[int, int]],
    *,
    completion_weight: float,    # default 1.0
    reducer_weight: float,       # default 0.35
) -> Optional[np.ndarray]:
    """Return a normalized target distribution over legal_moves (length M),
    or None if the target is empty after legal-move alignment."""
    weights = np.zeros(len(legal_moves), dtype=np.float32)
    for i, m in enumerate(legal_moves):
        if m in completion_moves:
            weights[i] = completion_weight              # disjoint-mass: completion wins ties
        elif m in reducing_moves:
            weights[i] = reducer_weight
    total = float(weights.sum())
    if total <= 0.0:
        return None
    return weights / total
```

**Disjoint-mass rule:** a move that is both endpoint-completing AND distance-reducing receives `completion_weight` (the larger), not the sum. Validation rule: `reducer_weight ≤ completion_weight`.

### 6.2 Batch tensor build

```python
def make_conversion_aux_tensors(
    positions: list[PositionRecord],
    legal_moves_padded: list[list[tuple[int,int]]],   # surfaced from make_padded_batch
    max_moves_cap: int,
    *,
    completion_weight: float = 1.0,
    reducer_weight: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (aux_target, aux_mask) with shapes (B, M_padded) and (B,).

    Padding entries in legal_moves_padded[i] are skipped (per Section 3 lock #2).
    legal_moves_padded[i] is ordered exactly like target_pi[i] columns and
    move_mask[i] — same indexing as the policy CE computation.
    """
    B = len(positions)
    aux_target = np.zeros((B, max_moves_cap), dtype=np.float32)
    aux_mask   = np.zeros((B,), dtype=np.float32)

    for i, p in enumerate(positions):
        if p.conversion is None:
            continue
        completion = {tuple(m) for m in p.conversion.get("endpoint_completion_moves") or ()}
        reducing   = {tuple(m) for m in p.conversion.get("distance_reducing_moves")   or ()}

        weights = np.zeros(max_moves_cap, dtype=np.float32)
        for j, m in enumerate(legal_moves_padded[i]):
            if m is None:                  # padding entry — skip
                continue
            if m in completion:
                weights[j] = completion_weight
            elif m in reducing:
                weights[j] = reducer_weight

        total = float(weights.sum())
        if total <= 0.0:
            continue
        aux_target[i] = weights / total
        aux_mask[i]   = 1.0

    return aux_target, aux_mask
```

### 6.3 Loss math

```python
def alphazero_loss_batch(
    network, positions, *,
    l2_weight=1e-4, value_weight=0.5, max_moves_cap=512, active_size=24,
    progress_weighted=True, progress_weight_floor=0.25,
    conversion_loss_weight: float = 0.0,             # NEW; default 0 = off
    conversion_completion_weight: float = 1.0,       # NEW
    conversion_reducer_weight: float = 0.35,         # NEW
):
    boards, move_rows, move_cols, move_mask, target_pi, outcomes, legal_padded = \
        make_padded_batch(positions, max_moves_cap=max_moves_cap, return_legal=True)

    logits, values, _ = network.forward_padded(...)
    # MASKED log_probs — illegal/padded columns must NOT contribute to either
    # policy_loss or aux_loss. Same shared tensor for both.
    log_probs = logits - mx.logsumexp(logits, axis=1, keepdims=True)   # (B, M)

    # --- existing policy loss (unchanged) ---
    policy_loss = -mx.sum(target_pi * log_probs, axis=1)               # (B,)
    policy_loss = mx.mean(policy_loss)

    # --- value + l2 unchanged ---

    # --- NEW: conversion auxiliary loss (zero-overhead path when disabled) ---
    aux_loss = mx.array(0.0)
    aux_coverage = 0.0
    aux_n_eligible = 0
    if conversion_loss_weight > 0.0:
        aux_target_np, aux_mask_np = make_conversion_aux_tensors(
            positions, legal_padded, max_moves_cap,
            completion_weight=conversion_completion_weight,
            reducer_weight=conversion_reducer_weight,
        )
        aux_target = mx.array(aux_target_np)
        aux_mask   = mx.array(aux_mask_np)

        per_pos_aux = -mx.sum(aux_target * log_probs, axis=1)          # (B,)
        per_pos_aux = aux_mask * per_pos_aux
        n_eligible_arr = mx.sum(aux_mask)
        aux_loss = mx.where(n_eligible_arr > 0,
                            mx.sum(per_pos_aux) / mx.maximum(n_eligible_arr, 1.0),
                            mx.array(0.0))
        aux_n_eligible = int(n_eligible_arr.item())
        aux_coverage = aux_n_eligible / max(len(positions), 1)

    total_loss = (policy_loss
                  + value_weight * value_loss
                  + l2_loss
                  + conversion_loss_weight * aux_loss)

    return total_loss, policy_loss, value_loss, l2_loss, aux_loss, aux_coverage, aux_n_eligible
```

### 6.4 Properties

- **Reuses `log_probs`** — no second forward pass, no second `logsumexp`.
- **Mean over eligible** — `aux_loss` magnitude is per-eligible-position. `conversion_loss_weight` is the only knob controlling relative magnitude vs `policy_loss`. Sample boost (Section 7) addresses *frequency* of eligible positions, not their per-loss weight.
- **Zero-eligible safety** — when no position is eligible, `aux_loss = 0.0` exactly, no NaN.
- **`conversion_loss_weight = 0.0` short-circuits** — `make_conversion_aux_tensors` is never called.
- **`aux_n_eligible` returned as integer** — for exact telemetry. Never reconstructed from a fractional coverage value.

### 6.5 Numerical sanity

With `λ_conv = 0.05`, `completion_weight = 1.0`, `reducer_weight = 0.35`:

When the model assigns near-zero probability to closeout moves (current state), `aux_loss` ≈ 5–10 (CE with all mass on a logit ~0). Multiplied by 0.05 = 0.25–0.5 added to total_loss. Current `policy_loss` ≈ 3–5 in late training, so `0.05 · aux_loss` is ~5–15% of policy loss magnitude when fully wrong, shrinking toward zero as the model learns.

If first-pass logs show sustained `aux_loss > 20`, the eligibility predicate or legal-move alignment is wrong — not the loss math.

---

## 7. Bounded replay sample boost (Track 2)

### 7.1 Eligibility tracking in ReplayBuffer

```python
class ReplayBuffer:
    def __init__(self, max_size: int = 100000):
        self._positions: list[PositionRecord] = []
        self._eligible_idxs: list[int] = []      # NEW — index pool
        self._eligible_pos: dict[int, int] = {}  # NEW — idx -> position in pool
        # ... existing ring buffer ...

    def _eligible_add(self, idx: int) -> None:
        if idx in self._eligible_pos:
            return
        self._eligible_pos[idx] = len(self._eligible_idxs)
        self._eligible_idxs.append(idx)

    def _eligible_remove(self, idx: int) -> None:
        pos = self._eligible_pos.pop(idx, None)
        if pos is None:
            return
        last = self._eligible_idxs.pop()
        if pos < len(self._eligible_idxs):
            self._eligible_idxs[pos] = last
            self._eligible_pos[last] = pos

    def _update_eligible_index(self, idx: int, p: PositionRecord) -> None:
        if p.conversion is not None:
            self._eligible_add(idx)
        else:
            self._eligible_remove(idx)
```

Index pool with swap-delete supports O(1) add/remove/sample. The eligibility predicate is not re-evaluated at sample time — `position.conversion is not None` is the cached truth (the self-play guard already enforced eligibility at write time).

### 7.2 Active-size filtering cost

First implementation filters eligible indices by `active_size` at sample time (O(E) per call, where E is current eligible-pool size). E is expected small. If profiling shows sampling overhead, promote to per-active-size index pools (`_eligible_idxs_by_size: dict[int, IndexPool]`).

### 7.3 Stratified `sample()`

```python
def sample(
    self,
    batch_size: int,
    rng: random.Random | None = None,
    active_size: int | None = None,
    *,
    conversion_sample_boost: float = 1.0,
    conversion_max_batch_fraction: float = 0.15,
) -> list[PositionRecord]:
    """
    boost == 1.0 (DEFAULT): pure uniform — eligibility ignored, cap not consulted.
        First-experiment invariant: identical behavior to pre-Spec-2 sample().

    boost > 1.0: stratified.
        cap_count = math.floor(batch_size * conversion_max_batch_fraction)
        if cap_count == 0:
            return uniform sample (boost has no room to act)

        natural_expectation = batch_size * (eligible_count / total_filtered_count)
        target_eligible = min(
            math.ceil(natural_expectation * conversion_sample_boost),  # ceil to avoid round-to-zero
            cap_count,
            eligible_count,
            batch_size,
        )

        Draw target_eligible uniformly from eligible pool (active_size-filtered),
        without replacement.
        Fill remaining (batch_size - target_eligible) uniformly from
        non-eligible pool (active_size-filtered), without replacement.
        Shuffle assembled batch.

        INVARIANT: a position appears at most once in the returned batch.
    """
```

### 7.4 First-experiment invariant

The first experiment (per the experiment recipes in Section 9) runs `--conversion-sample-boost 1.0`. With `boost == 1.0`, sampling short-circuits to the existing pure-uniform path — eligibility set is not consulted, cap is not consulted, no behavior delta. Aux loss is the sole experimental variable.

### 7.5 Curriculum interaction

```
candidates_eligible_at_size = eligible_idxs ∩ positions_at_active_size
candidates_other_at_size    = positions_at_active_size \ eligible_idxs
```

Size-12 batches never draw size-24 eligible positions.

### 7.6 Edge cases

| Condition | Behavior |
|---|---|
| `eligible_count == 0` at requested active_size | Pure uniform; log once per iter; `boost_inactive_steps += 1` |
| `eligible_count > 0` but less than `target_eligible` | Take all eligible; fill from non-eligible |
| `cap_count == 0` (cap × batch_size < 1) | Boost disabled for that batch |
| `cap > 1.0` | Validation error at startup |

### 7.7 Sampler telemetry

`ReplayBuffer.last_sample_stats`:

```python
@dataclass
class SampleStats:
    batch_size: int
    eligible_drawn: int
    cap_was_binding: bool
    boost_was_inactive: bool
```

The trainer accumulates per-iter:

```json
"sample_stats": {
  "eligible_drawn_total": int,
  "eligible_drawn_fraction": float,
  "cap_was_binding_steps": int,
  "boost_inactive_steps": int
}
```

---

## 8. Sidecar telemetry

### 8.1 `conversion_training` block

Final shape (always emitted, schema stable across enabled/disabled):

```json
"conversion_training": {
  "version": 1,
  "enabled": true,
  "config": {
    "configured_loss_weight": 0.05,
    "effective_loss_weight":  0.05,
    "completion_weight":      1.0,
    "reducer_weight":         0.35,
    "max_total_goal_distance": 2,
    "min_component_size":     8,
    "sample_boost":           1.0,
    "max_batch_fraction":     0.15
  },
  "buffer": {
    "eligible_positions_in_buffer":      1234,
    "eligible_position_rate":            0.0247,
    "eligible_positions_at_active_size": 980,
    "eligible_rate_at_active_size":      0.0312
  },
  "loss": {
    "aux_loss_avg_iter":                 3.2419,
    "aux_target_coverage_rate":          0.0289,
    "aux_positions_seen_in_training":    7423,
    "aux_positions_fraction_in_batches": 0.0289
  },
  "sample_stats": {
    "eligible_drawn_total":     7423,
    "eligible_drawn_fraction":  0.0289,
    "cap_was_binding_steps":    0,
    "boost_inactive_steps":     0
  },
  "consistency": {
    "drawn_vs_seen_match":  true,
    "drawn_minus_seen":     0
  }
}
```

When `effective_conversion_enabled = false`:

```json
"conversion_training": {
  "version": 1,
  "enabled": false,
  "config": {
    "configured_loss_weight": 0.05,
    "effective_loss_weight":  0.0,
    "...": "..."
  },
  "buffer": { "...all zeros or values..." },
  "loss": {
    "aux_loss_avg_iter": 0.0,
    "aux_target_coverage_rate": 0.0,
    "aux_positions_seen_in_training": 0,
    "aux_positions_fraction_in_batches": 0.0
  },
  "sample_stats": { "...zeros..." },
  "consistency": { "drawn_vs_seen_match": true, "drawn_minus_seen": 0 }
}
```

### 8.2 Drawn-vs-seen consistency invariant

- `drawn = sample_accumulator.eligible_drawn_total` — sampler's running tally
- `seen = loss_accumulator.aux_positions_seen_in_training` — sum of `aux_n_eligible` returned by `train_step`
- `drawn_minus_seen = drawn - seen`
- `drawn_vs_seen_match = (drawn_minus_seen == 0)`

Mismatch indicates one of: legal-move alignment bug, active-size filter divergence, stale conversion metadata, cross-curriculum buffer drift. The match field is sticky — once an iter sees a divergence, it is flagged. A warning is logged with non-zero delta.

### 8.3 `recovery_or_extreme_closeout_drift` block

Telemetry-only, default-on. Reads existing `goal_completion_record` fields — no new computation.

```json
"recovery_or_extreme_closeout_drift": {
  "version": 1,
  "enabled": true,
  "config": {
    "dominant_unavailable_moves_threshold": 10,
    "delay_threshold":                       20
  },
  "games_total":                100,
  "detected_games":              98,
  "count":                        2,
  "rate":                         0.02,
  "rate_among_detected":          0.0204,
  "dominant_unavailable_moves": {
    "p50": 0, "p90": 4, "p95": 9, "max": 22, "mean": 1.7
  },
  "trigger_breakdown": {
    "dominant_unavailable_only":      1,
    "delay_ge_threshold_only":        0,
    "state_cap_after_detection_only": 1,
    "multiple_triggers":              0
  }
}
```

### 8.4 Recovery predicate

```python
def is_recovery_or_extreme_closeout_drift(
    record: dict,
    *,
    du_threshold: int,
    delay_threshold: int,
) -> bool:
    if not record.get("detected"):
        return False
    # du_moves: explicit fallback chain — never silently zero on Class 2
    du_moves = (
        record.get("winner_moves_with_dominant_unavailable")
        if record.get("outcome_class") == 1
        else record.get("dominant_unavailable_moves")    # Class 2 fallback if tracker emits this
    )
    if du_moves is not None and du_moves >= du_threshold:
        return True
    delay = record.get("conversion_delay_plies")
    if delay is not None and delay >= delay_threshold:
        return True
    if record.get("outcome_class") == 2 and record.get("detected"):
        return True   # state_cap after detection
    return False
```

Class 2 records contribute through `state_cap_after_detection` clause unless the tracker explicitly populates a side-specific `dominant_unavailable_moves` field. The fallback chain prevents silent-zero attribution.

### 8.5 `trigger_breakdown` semantics

Mutually exclusive partition:
- `dominant_unavailable_only` — only the DU clause fired
- `delay_ge_threshold_only` — only the delay clause fired
- `state_cap_after_detection_only` — only the cap clause fired
- `multiple_triggers` — ≥ 2 clauses fired

`count = sum(trigger_breakdown values)`.

### 8.6 Where these blocks are written

In the trainer's per-iter sidecar writer, after Spec 1.5's `goal_completion_summary`. Both helpers are pure, located in:

```
scripts/GPU/alphazero/conversion_telemetry.py
  - build_conversion_training_block(config, buffer_stats, loss_accumulator, sample_accumulator)
  - build_recovery_block(records, *, du_threshold, delay_threshold)
  - is_recovery_or_extreme_closeout_drift(record, *, du_threshold, delay_threshold)
```

### 8.7 Flat CSV fields (per-iter trend)

Following the existing `fps_*` / `sas_*` pattern in `metrics.csv`:

| Field | Source |
|---|---|
| `cnv_enabled` | `conversion_training.enabled` (0/1) |
| `cnv_loss_weight` | `effective_loss_weight` |
| `cnv_aux_loss_avg` | `loss.aux_loss_avg_iter` |
| `cnv_aux_coverage` | `loss.aux_target_coverage_rate` |
| `cnv_aux_seen` | `loss.aux_positions_seen_in_training` |
| `cnv_eligible_in_buf` | `buffer.eligible_positions_in_buffer` |
| `cnv_eligible_at_size` | `buffer.eligible_positions_at_active_size` |
| `cnv_drawn_total` | `sample_stats.eligible_drawn_total` |
| `cnv_drawn_vs_seen_ok` | `consistency.drawn_vs_seen_match` (0/1) |
| `rcv_count` | `recovery.count` |
| `rcv_rate` | `recovery.rate` |
| `rcv_du_p90` | `recovery.dominant_unavailable_moves.p90` |

### 8.8 Analyzer surfacing (minimal)

Aux-loss effect is measured through the **existing** `policy_mcts_summary` analyzer report. Spec 2 adds two narrow read-only sections:

**Conversion-training trend** in `report.txt`:

```
── Conversion-training trend ─────────────────────────────────
Iters covered:   120-129
Aux loss weight: 0.05 (constant)
Aux loss (avg):  3.24 → 2.87 → 2.51 → ... → 1.62
Coverage rate:   2.9% → 3.1% → 3.0% → ... → 3.3%
Drawn vs seen:   ✓ all iters consistent
──────────────────────────────────────────────────────────────
```

**Recovery / extreme-closeout-drift trend**:

```
── Recovery / extreme-closeout-drift (telemetry only) ────────
Iters covered:        120-129
Recovery count/iter:  18 → 15 → 14 → ... → 9
Recovery rate:        1.8% → 1.5% → 1.4% → ... → 0.9%
DU moves p90:         4 → 3 → 3 → ... → 2
──────────────────────────────────────────────────────────────
```

Plus per-iter CSVs `conversion_training_by_iter.csv` and `recovery_or_extreme_closeout_drift_by_iter.csv`, paralleling `forced_probe_by_iter.csv`.

---

## 9. CLI surface and config defaults

### 9.1 Flags

#### Track 1 — auxiliary loss (Phase 2)

| Flag | Default | Validation | Purpose |
|---|---|---|---|
| `--conversion-policy-loss-enabled` | `False` | bool | Master switch. `effective_conversion_enabled` derives from this flag only. |
| `--conversion-policy-loss-weight` | `0.05` | `> 0.0` (when enabled); errors if enabled+weight≤0 | λ in `total_loss = ... + λ · aux_loss`. |
| `--conversion-completion-weight` | `1.0` | `> 0.0` | Target weight for `endpoint_completion_moves`. |
| `--conversion-reducer-weight` | `0.35` | `≥ 0.0`, `≤ completion_weight` | Target weight for `distance_reducing_moves`. |
| `--conversion-max-total-goal-distance` | `2` | `1 ≤ x ≤ 3` | Single eligibility threshold. First experiment 2, widen to 3. |

#### Track 2 — sample boost (Phase 3)

| Flag | Default | Validation | Purpose |
|---|---|---|---|
| `--conversion-sample-boost` | `1.0` | `≥ 1.0` | `1.0` short-circuits to pure uniform. |
| `--conversion-max-batch-fraction` | `0.15` | `0.0 ≤ x ≤ 1.0` | Hard cap on eligible fraction per batch. |

#### Eligibility (shared with Spec 1.5)

| Flag | Default | Source |
|---|---|---|
| `--goal-completion-min-component-size` | `8` | Existing Spec 1.5 flag — single source of truth. |
| `--goal-completion-max-depth` | `3` | Existing Spec 1.5 flag — same. |

#### Track 4 — recovery / extreme-closeout-drift telemetry

| Flag | Default | Validation | Purpose |
|---|---|---|---|
| `--recovery-bucket-enabled` | `True` | bool | Telemetry-only. Off ⇒ block emitted as `{"enabled": false}` shell. |
| `--recovery-dominant-unavailable-threshold` | `10` | `≥ 1` | DU-moves threshold. |
| `--recovery-delay-threshold` | `20` | `≥ 1` | `conversion_delay_plies` threshold. |

### 9.2 Effective-config derivation

```python
effective_conversion_enabled = args.conversion_policy_loss_enabled
effective_loss_weight = (
    args.conversion_policy_loss_weight
    if effective_conversion_enabled
    else 0.0
)
```

Sidecar emits both `configured_loss_weight` and `effective_loss_weight` for clarity across run configs.

### 9.3 Validation rules (trainer startup)

```python
def _validate_conversion_args(args):
    if args.conversion_policy_loss_enabled and args.conversion_policy_loss_weight <= 0.0:
        parser.error(
            "--conversion-policy-loss-enabled requires "
            "--conversion-policy-loss-weight > 0.0. "
            "For dry-run telemetry without loss, omit --conversion-policy-loss-enabled."
        )
    if args.conversion_completion_weight <= 0.0:
        parser.error("--conversion-completion-weight must be > 0.0")
    if args.conversion_reducer_weight < 0.0:
        parser.error("--conversion-reducer-weight must be >= 0.0")
    if args.conversion_reducer_weight > args.conversion_completion_weight:
        parser.error(
            "--conversion-reducer-weight must be <= --conversion-completion-weight "
            f"(got reducer={args.conversion_reducer_weight}, "
            f"completion={args.conversion_completion_weight})."
        )
    if not (1 <= args.conversion_max_total_goal_distance <= 3):
        parser.error("--conversion-max-total-goal-distance must be in [1, 3]")
    if args.conversion_sample_boost < 1.0:
        parser.error(
            "--conversion-sample-boost must be >= 1.0 "
            "(use --conversion-policy-loss-enabled to disable conversion entirely)"
        )
    if not (0.0 <= args.conversion_max_batch_fraction <= 1.0):
        parser.error("--conversion-max-batch-fraction must be in [0.0, 1.0]")

    # Cross-flag warning
    if (not args.conversion_policy_loss_enabled
            and args.conversion_sample_boost > 1.0):
        print(
            "[WARN] --conversion-sample-boost > 1.0 has no effect when "
            "--conversion-policy-loss-enabled is off. Sample boost stays inactive "
            "and PositionRecord.conversion stays unpopulated."
        )

    if args.recovery_dominant_unavailable_threshold < 1:
        parser.error("--recovery-dominant-unavailable-threshold must be >= 1")
    if args.recovery_delay_threshold < 1:
        parser.error("--recovery-delay-threshold must be >= 1")
```

### 9.4 Default-config behavior

With **no** `--conversion-*` or `--recovery-*` flags:

- Conversion auxiliary loss: **off** (`effective_conversion_enabled=False`).
- Conversion eligibility tagging: **off** (`PositionRecord.conversion=None` for every position).
- Sample boost: inert (would short-circuit to uniform anyway at `boost=1.0`).
- Recovery bucket telemetry: **on** (free — reads existing tracker fields).
- Sidecar shape stable: `conversion_training` block emitted with `enabled=false`; `recovery_or_extreme_closeout_drift` populated.

Behavior identical to pre-Spec-2.

### 9.5 First-experiment recipe

```bash
python -m scripts.GPU.alphazero.train \
    --iterations 10 \
    --conversion-policy-loss-enabled \
    --conversion-policy-loss-weight 0.05 \
    --conversion-completion-weight 1.0 \
    --conversion-reducer-weight 0.35 \
    --conversion-sample-boost 1.0 \
    --conversion-max-batch-fraction 0.15 \
    --conversion-max-total-goal-distance 2
```

`boost=1.0` short-circuits to pure uniform sampling. Aux loss is the sole experimental variable. Recovery telemetry runs in parallel.

### 9.6 Second-experiment recipe (add boost)

```bash
# ... same as experiment 1 ...
    --conversion-sample-boost 2.0    # was 1.0
```

Single-flag delta. Drawn-vs-seen consistency check is the cheap canary that boost is changing batch composition as expected.

### 9.7 Third-experiment recipe (widen eligibility)

```bash
# ... same as experiment 2 ...
    --conversion-max-total-goal-distance 3    # was 2
```

Captures `total_goal_distance == 3` cases (broader-conversion category).

### 9.8 Trainer startup banner

When enabled:

```
Conversion auxiliary loss: enabled (weight=0.05)
  Target weights:        completion=1.0, reducer=0.35
  Eligibility:           total_goal_distance <= 2, min_component_size >= 8
  Sample boost:          1.0 (disabled — pure uniform)
  Max batch fraction:    0.15 (cap inert at boost=1.0)
Recovery / extreme-closeout-drift: enabled (du_threshold=10, delay_threshold=20)
```

When disabled:

```
Conversion auxiliary loss: disabled
Recovery / extreme-closeout-drift: enabled (du_threshold=10, delay_threshold=20)
```

---

## 10. Success criteria

### 10.1 Baseline (110-119)

| Metric | Baseline |
|---|---|
| Endpoint completion policy top5 | 0.0% |
| Distance reducing policy top5 | 0.0% |
| Selected completes endpoint | 34.2% |
| Selected reduces distance | 21.8% |
| Selected redundant | 31.7% |
| Selected off-chain | 10.7% |
| Selected other | 1.6% |
| Games delay ≥ 10 | 65 |
| Games delay ≥ 20 | 16 |
| Game-level high-value delayed | 136 |
| Detailed high-value delayed records | 511 |

### 10.2 First 10-iteration target (after Phase 2 ships)

| Metric | Target |
|---|---|
| Endpoint completion policy top5 | > 30–40% |
| Distance reducing policy top5 | > 40–50% |
| Selected redundant | < 25% |
| Selected off-chain | < 8% |
| Games delay ≥ 10 | meaningfully lower |
| Games delay ≥ 20 | no worse, ideally lower |

Policy ranking should move first; delay metrics may lag while the replay buffer turns over.

### 10.3 Longer-term target

| Metric | Target |
|---|---|
| Endpoint completion policy top5 | > 60% |
| Distance reducing policy top5 | > 70% |
| Selected redundant | < 20% |
| Selected off-chain | < 5% |
| Games delay ≥ 10 | cut by ~50% |
| Games delay ≥ 20 | single digits or near zero |

### 10.4 Guardrail metrics (must not degrade)

- `forced_probe sign_correct`
- `strong_advantage_probe sign_correct`
- Opening corner / edge rates
- Average plies
- Draw / state_cap rate
- Red/black balance
- Overall value calibration
- Policy entropy

The fix should improve closeout behavior without making openings rigid or harming general play.

### 10.5 First-run interpretation note

For the first enabled 10-iteration run, judge wiring first by:
- `aux_target_coverage_rate` (training pipeline is alive)
- `aux_positions_seen_in_training` (loss had batch exposure)
- Policy top5 movement (the actual goal)

Do not over-interpret `delay_ge_10` / `delay_ge_20` until the replay buffer contains enough conversion-tagged positions (~5–10 iterations for typical configs, depending on `buffer_size / games_per_iter / positions_per_game`).

---

## 11. Test strategy

~55 new tests across 7 new test files plus extensions, with 9 anchor tests for critical invariants. Comparable to Spec 1.5's ~50 tests.

### 11.1 New test files

#### `tests/test_conversion_eligibility.py` — predicate (Section 4)

- `test_eligible_with_two_endpoint_closeout`
- `test_ineligible_when_total_distance_above_threshold`
- `test_ineligible_when_component_too_small`
- `test_ineligible_when_no_completion_or_reducer_moves`
- `test_ineligible_when_gc_state_full_is_none`
- `test_ineligible_when_total_distance_is_none`

#### `tests/test_conversion_target.py` — target construction (Section 6)

- `test_target_normalizes_to_unit_sum`
- `test_target_assigns_completion_weight_to_completion_moves`
- `test_target_assigns_reducer_weight_to_reducer_only_moves`
- `test_target_disjoint_mass_rule_completion_wins`
- `test_target_zero_for_other_legal_moves`
- **`test_conversion_aux_target_aligns_with_legal_move_order`** *(anchor)* — `legal=[(1,2),(3,4),(5,6)], completion=[(5,6)], reducer=[(1,2)]` → `target=[0.35/1.35, 0.0, 1.0/1.35]`
- `test_target_returns_none_when_no_completion_or_reducer_in_legal_moves`

#### `tests/test_conversion_aux_tensors.py` — batch tensors (Section 6)

- `test_aux_tensor_shape_matches_target_pi`
- `test_aux_mask_zero_for_ineligible_positions`
- `test_aux_mask_zero_when_target_returns_none`
- `test_aux_tensor_skips_padding_columns`
- `test_aux_tensor_aligns_with_target_pi_columns`

#### `tests/test_conversion_loss.py` — loss math (Section 6)

- `test_aux_loss_zero_when_all_ineligible`
- **`test_aux_loss_uses_masked_log_probs`** *(anchor)*
- `test_aux_loss_mean_over_eligible_only`
- **`test_aux_loss_returns_n_eligible_as_int`** *(anchor)*
- `test_total_loss_includes_aux_term_when_enabled`
- `test_total_loss_excludes_aux_when_weight_zero`
- `test_aux_loss_matches_hand_computed_ce_on_fixture`

#### `tests/test_replay_buffer_conversion.py` — sampling (Section 7)

- `test_replay_buffer_eligible_index_tracks_evictions`
- `test_replay_buffer_eligible_index_swap_delete_correctness`
- **`test_sample_boost_1_is_pure_uniform`** *(anchor)*
- `test_sample_boost_2_produces_at_most_cap_fraction`
- `test_sample_boost_uses_ceil_rounding_for_target` — fixture: `batch_size=16, natural_expectation < 1, cap_count ≥ 1, boost > 1` → expect `eligible_drawn == 1`
- `test_sample_falls_back_to_uniform_when_eligible_pool_empty`
- `test_sample_active_size_intersects_eligibility`
- **`test_sample_no_duplicate_positions_with_two_strata`** *(anchor)*

#### `tests/test_conversion_telemetry.py` — sidecar blocks (Section 8)

- `test_conversion_training_block_schema_when_disabled` — keys present, `configured_loss_weight=0.05, effective_loss_weight=0.0`
- `test_conversion_training_block_schema_when_enabled` — `configured == effective == 0.05`
- `test_conversion_training_block_disabled_emits_zero_telemetry`
- **`test_drawn_vs_seen_match_flags_divergence`** *(anchor)*
- `test_drawn_vs_seen_match_naming_correctness` — sampler→drawn, loss→seen
- `test_recovery_predicate_three_triggers`
- `test_recovery_predicate_state_cap_after_detection_required_for_class2`
- `test_recovery_block_class2_dominant_unavailable_handling`
- `test_recovery_block_excludes_undetected_games`
- `test_recovery_block_percentiles_handcrafted`
- `test_recovery_rate_denominators` — `rate=count/games_total`, `rate_among_detected=count/detected_games`
- `test_recovery_block_renamed_to_extreme_closeout_drift` — sidecar key is `recovery_or_extreme_closeout_drift`

#### `tests/test_position_record_conversion.py` — IPC + persistence (Section 5)

- `test_position_record_conversion_round_trip_dict`
- `test_position_record_conversion_none_when_loss_disabled`
- **`test_position_record_conversion_pre_move_invariant`** *(anchor)*
- `test_position_record_buffer_load_with_old_no_conversion_field`
- `test_game_complete_ipc_carries_conversion`

#### `tests/test_conversion_cli_config.py` — CLI/config invariants (Section 9)

- **`test_conversion_disabled_by_default_effective_weight_zero`** *(anchor)*
- `test_conversion_enabled_uses_configured_weight`
- `test_conversion_enabled_with_zero_weight_errors`
- `test_sample_boost_without_loss_warns_and_tagging_stays_off`
- `test_reducer_weight_greater_than_completion_weight_errors`
- `test_conversion_max_total_goal_distance_bounds`

### 11.2 Extended existing test files

#### `tests/test_self_play_goal_completion_integration.py`

- `test_play_game_attaches_conversion_when_enabled_and_eligible`
- **`test_play_game_no_conversion_metadata_when_loss_disabled`** *(anchor)* — counts conversion-specific BFS calls (not absolute BFS) under default config; uses `goal_completion_record_enabled=False, goal_completion_emit_enabled=False, conversion_policy_loss_enabled=False` to isolate
- `test_play_game_no_extra_bfs_when_spec_15_already_running_full_state`
- `test_play_game_conversion_enabled_computes_full_state_when_emit_disabled` — confirms aux loss path can force a full BFS when Spec 1.5 emit is off

#### `tests/test_trainer_loss.py`

- `test_trainer_runs_with_conversion_enabled_smoke`

### 11.3 Anchor tests (9, mandatory)

| Anchor | Pins |
|---|---|
| `test_conversion_aux_target_aligns_with_legal_move_order` | Section 6 alignment invariant |
| `test_aux_loss_uses_masked_log_probs` | Section 6 padding-mask invariant |
| `test_aux_loss_returns_n_eligible_as_int` | Section 6 integer-tally invariant |
| `test_sample_boost_1_is_pure_uniform` | Section 7 first-experiment invariant |
| `test_sample_no_duplicate_positions_with_two_strata` | Section 7 no-replacement invariant |
| `test_drawn_vs_seen_match_flags_divergence` | Section 8 cross-boundary invariant |
| `test_position_record_conversion_pre_move_invariant` | Section 5 attach-point invariant |
| `test_play_game_no_conversion_metadata_when_loss_disabled` | Section 9 default-off invariant |
| `test_conversion_disabled_by_default_effective_weight_zero` | Section 9 effective-weight invariant |

### 11.4 Test fixtures

- Reuse Spec 1.5's synthetic 8–12 ply fixtures where possible.
- One real `iter_0110_game_*.json` (already used by Spec 1.5 perf-regression test) extended with conversion field for round-trip tests.
- Two new synthetic positions, one per stratum, for the alignment anchor.
- Hand-computed CE values for `test_aux_loss_matches_hand_computed_ce_on_fixture` (tolerance 1e-5 for MLX numerical drift).

### 11.5 Test scope total

| File | New tests | Anchors |
|---|---|---|
| test_conversion_eligibility.py | 6 | 0 |
| test_conversion_target.py | 7 | 1 |
| test_conversion_aux_tensors.py | 5 | 0 |
| test_conversion_loss.py | 7 | 2 |
| test_replay_buffer_conversion.py | 8 | 2 |
| test_conversion_telemetry.py | 12 | 1 |
| test_position_record_conversion.py | 5 | 1 |
| test_conversion_cli_config.py | 6 | 1 |
| Extensions to existing files | 6 | 1 |
| **Total** | **62** | **9** |

---

## 12. Implementation phases

Single spec, ordered for risk minimization. Each phase is one or more commits with TDD.

### 12.1 Phase 1 — Data plumbing (no behavior change)

1. Eligibility predicate module (`scripts/GPU/alphazero/conversion_loss.py`) — `is_conversion_eligible` only. Tests: `test_conversion_eligibility.py`.
2. PositionRecord schema addition; `to_dict`/`from_dict` round-trip. Tests: `test_position_record_conversion.py`.
3. IPC carrying — `GameComplete` flow; pickle round-trip test.
4. Self-play attach point in `play_game()` — populate `position.conversion` when `effective_conversion_enabled`. Reuse existing `gc_state_full`. Anchors: `test_position_record_conversion_pre_move_invariant`, `test_play_game_no_conversion_metadata_when_loss_disabled`.
5. CLI flag scaffolding with default-off. Banner additions. Anchor: `test_conversion_disabled_by_default_effective_weight_zero`. Tests: `test_conversion_cli_config.py`.

After Phase 1: trainer accepts new flags; default behavior identical to pre-Spec-2; default-off invariant tested. Buffers can be re-saved/loaded with the new field.

### 12.2 Phase 2 — Auxiliary loss (the actual policy correction)

6. Target construction (`build_conversion_target`). Tests: `test_conversion_target.py`. Anchor: `test_conversion_aux_target_aligns_with_legal_move_order`.
7. Batch tensor build (`make_conversion_aux_tensors`); surface `legal_moves_padded` from `make_padded_batch`. Tests: `test_conversion_aux_tensors.py`.
8. Loss math — extend `alphazero_loss_batch` and `train_step` return signature with `aux_loss`, `aux_coverage`, `aux_n_eligible`. Tests: `test_conversion_loss.py`. Anchors: `test_aux_loss_uses_masked_log_probs`, `test_aux_loss_returns_n_eligible_as_int`.
9. Trainer accumulators + sidecar `conversion_training` block (`build_conversion_training_block`). Stable schema enabled/disabled.
10. Phase-3-emit-disabled cost-path test — `test_play_game_conversion_enabled_computes_full_state_when_emit_disabled`.
11. End-to-end smoke — 1 iter, 2 games, batch=8 with first-experiment recipe. Sidecar populated, banner correct.

After Phase 2: first experiment shippable. Run 10 iterations, compare to 110-119 baseline via existing `policy_mcts_summary` analyzer report.

### 12.3 Phase 3 — Bounded replay boost

12. ReplayBuffer eligible-index tracking (index pool with swap-delete; `_update_eligible_index` on add/evict). Tests: `test_replay_buffer_conversion.py` (eligibility-tracking subset).
13. Stratified `sample()` — boost=1.0 short-circuit, ceil rounding, no-replacement invariant, active_size intersection. Anchors: `test_sample_boost_1_is_pure_uniform`, `test_sample_no_duplicate_positions_with_two_strata`.
14. Sample-stats accumulator + sidecar `sample_stats` sub-block. Drawn-vs-seen consistency check with warning. Anchor: `test_drawn_vs_seen_match_flags_divergence`.
15. End-to-end check — second-experiment recipe (`--conversion-sample-boost 2.0`).

After Phase 3: second experiment shippable.

### 12.4 Phase 4 — Recovery / extreme-closeout-drift telemetry (independent)

16. Recovery predicate + block builder (`is_recovery_or_extreme_closeout_drift`, `build_recovery_block`). Reads existing `goal_completion_record` fields. Tests: `test_conversion_telemetry.py` (recovery subset).
17. Trainer sidecar wiring. Default-on (free).
18. Analyzer surfacing — minimal trend section in `report.txt` and `recovery_or_extreme_closeout_drift_by_iter.csv`.

Phase 4 is independent of the loss machinery. Could land before Phase 2 if desired; sequenced after Phase 1's CLI scaffolding only because the trainer wiring shares the same sidecar writer.

### 12.5 Phase 5 — Curated probe set + mining (deferred)

Deferred to Spec 3 or a Spec 2 follow-up. Stub:

- `tests/probes/conversion_probes.json` curated from worst-cases CSV with manual review.
- `run_conversion_probes_inline(network, probes)` — policy-rank-of-closeout-move metric.
- Slots into existing tier-keyed `_run_inline_probe_eval` wrapper.
- Sidecar gains `probe_summary["goal_completion"]` and flat `gcps_*` fields.

Lands once Phase 2's effect is measured and a curated probe set is worth the operational cost.

### 12.6 Sequencing summary

- **Strict order:** Phase 1 → Phase 2 → Phase 3.
- **Independent:** Phase 4 (any time after Phase 1's CLI scaffolding).
- **Deferred:** Phase 5 (post-Spec 2).

---

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Aux loss destabilizes policy training | `loss_weight=0.05` is small; first experiment runs `boost=1.0` so eligible position rate stays natural. Rollback: omit `--conversion-policy-loss-enabled`. Watch guardrail metrics (Section 10.4). |
| Aux target mis-aligned with legal_moves order — silent training drift | Anchor test `test_conversion_aux_target_aligns_with_legal_move_order` + drawn-vs-seen consistency invariant catch alignment bugs cross-boundary. |
| Phase 1 changes IPC schema — workers and trainer must restart together | Same constraint as Spec 1.5; documented; no graceful in-place upgrade. |
| Eligibility predicate too narrow — `total ≤ 2` produces too few eligible positions | Two escape hatches: widen to `total ≤ 3` via single flag (third experiment); or amplify with `sample_boost` (second experiment). |
| `gc_state_full` enumeration wrong on edges | Same enumeration drives Spec 1.5 Phase 3 emit; bugs would have surfaced there. New tests add per-position move-set fixtures. |
| Sampling boost reorders the buffer — eligible positions cluster, biasing optimizer | Section 7.3: stratified draw is followed by an explicit shuffle. Test pins this. |
| Recovery predicate undercounts Class 2 — `winner_moves_with_dominant_unavailable` is null on capped games | Section 8.4 explicitly defines fallback chain; Class 2 contributes through `state_cap_after_detection`. Test `test_recovery_block_class2_dominant_unavailable_handling`. |
| Aux loss with weight=0 still allocates tensors — perf regression | `if conversion_loss_weight > 0.0` guard short-circuits `make_conversion_aux_tensors`. Test `test_total_loss_excludes_aux_when_weight_zero`. |

---

## 14. Migration / backwards compat

### 14.1 Pre-existing buffers

Buffers persisted before Spec 2 load with `position.conversion = None` for every position. Aux loss treats them as ineligible; no errors. Re-running training populates `conversion` on **new** positions only — buffer mixes old (no metadata) and new until enough new positions evict the old.

The first 1–2 iterations after enabling conversion loss have a smaller eligible pool than steady state. The drawn-vs-seen invariant still holds; aux loss has less to consume. After buffer turnover (~5–10 iterations for typical configs), eligible pool reaches steady state. Sidecar `eligible_position_rate` makes this visible.

**Operator note:** Don't judge first-2-iter aux-loss magnitude until buffer turnover completes. Per Section 10.5: judge wiring first by coverage and policy top5 movement; delay metrics may lag.

### 14.2 Pre-existing per-game JSONs

Unchanged. `goal_completion_record` (Spec 1.5) is the only goal-completion artifact in per-game JSONs; conversion metadata lives only in the in-memory PositionRecord buffer.

### 14.3 Schema versioning

- `PositionRecord.conversion["version"] = 1`
- `conversion_training["version"] = 1`
- `recovery_or_extreme_closeout_drift["version"] = 1`

Bump on rename / removal / semantic change. Pure-additive optional fields do not bump. Aggregator/analyzer reads `version` defensively; warns on unknown versions and proceeds with best-effort field reads.

---

## 15. Rollback plan

Three independent kill switches, smallest blast radius first:

1. **Omit `--conversion-policy-loss-enabled`** (default state) — disables aux loss math, disables PositionRecord conversion tagging, zero overhead, zero behavior delta vs pre-Spec-2.

   Setting `--conversion-policy-loss-weight 0.0` while leaving `--conversion-policy-loss-enabled` set fails validation (Section 9.3). Rollback uses the enable flag, not the weight.

2. **`--conversion-sample-boost 1.0`** (default) — disables stratified sampling. Aux loss continues if enabled. Used to isolate "loss is the problem" from "sampling is the problem."

3. **`--recovery-bucket-enabled false`** — disables recovery telemetry. Independent of the loss machinery.

If a structural bug appears (drawn-vs-seen mismatches, unexpected aux loss values):
- Remove `--conversion-policy-loss-enabled` from the run config. Training reverts to pre-Spec-2 behavior immediately.
- Buffer/replay state is preserved — old conversion metadata becomes inert; new positions stop tagging.
- Code remains in place for fix; no rollback PR needed.

---

## 16. Out of scope

- **Curated conversion probe set + tier-keyed inline evaluator** — Phase 5, deferred.
- **Per-side or per-category asymmetric weighting** — completion/reducer weights are global. Future spec could vary by `category` (e.g., `two_endpoint_closeout_2ply` vs `distance_eq_3_broader_conversion`).
- **Recovery-bucket aux loss** — explicitly punted to Spec 3. Spec 2 only measures.
- **Curriculum integration tuning** — eligibility intersects active_size at sample time; existing curriculum machinery is honored. Cross-curriculum behavior surfaces in flat CSV; future tuning if needed.
- **Value-head changes** — explicitly out. Value head is fine; do not touch `--value-weight`, `--value-lr-scale`, `--value-grad-max-norm`.
- **MCTS prior pre-bias** — adding closeout-move prior boost at MCTS root (analogous to `root_edge_band_penalty` but pro-closeout). Could be considered if Spec 2's policy correction is insufficient.
- **Backfilling conversion metadata into pre-Spec-2 per-game JSONs** — replay buffer is the only consumer; backfill is unnecessary.

---

## 17. Bottom line

Spec 2 is a policy-prior correction, not a value-head correction.

The model already knows closeout positions are winning (`search_score p50=0.99` after detection). The missing skill is:

> When a winning chain can cross the goal line, rank the crossing or distance-reducing move highly instead of reinforcing the already-winning chain.

MCTS already finds many of these moves through search despite the poor policy prior. Spec 2 supervises the policy head to learn them directly via an auxiliary cross-entropy loss against an explicit conversion target, applied only on closeout-eligible positions. Optional bounded sample boost amplifies exposure. A telemetry-only recovery bucket measures the harder dominant-lost failure mode for a future spec.
