# Recovery / Re-targeting Diagnostic — Design Spec

**Date:** 2026-05-12
**Status:** Drafted (awaiting user review)
**Predecessors:**
- [2026-05-10-closeout-tail-correction-design.md](2026-05-10-closeout-tail-correction-design.md) (Spec 3: Fix 1 shipped + Fix 2 shipped pending 170-179 trend confirmation)

**Successor:** Spec 5 (Recovery-aware training) only if §11 decision rule signals a systemic, addressable failure shape.

---

## 1. Goal & Scope

### 1.1 Goal

Detect and characterize self-play games where a side, after its position collapses or its dominant chain is contested, fails to re-target — continuing to play locally-connecting moves that don't reduce its own goal distance, block the opponent, or build a credible alternate route. Surface the signal in per-iteration sidecars, the analyzer report, and a worst-cases CSV.

Diagnostic-only. No change to MCTS, training targets, or move selection.

### 1.2 Anchor case

`iter_0169_game_022`: Black's value collapsed from +0.30 to −0.40 in one own-ply, then stayed below −0.75 for 10 plies. The post-collapse Black moves were local to existing black structure and appeared to keep extending/connecting fragments rather than reducing Black's own goal distance, blocking Red's eventual closeout, or building a credible alternate route under the proposed diagnostic definitions.

Spec 3 §1.1 identified two failure clusters in the 130-139 worst cases. Fix 1 + Fix 2 (Spec 3) address the first (td=1 visit forcing and td=2 closeout selection tie-break). The second cluster — "dominant component becomes unavailable within ~4 plies of detection, q collapses, the reducer disappears from search too" — is structurally distinct and observed from the *loser's* MCTS state, which the existing winner-scoped diagnostics cannot see. This spec adds the diagnostic surface for it.

### 1.3 In scope

- Per-move trigger detection and classification, watched independently for each side
- Per-side, per-game record emitted only when the side's window opens
- Per-iteration sidecar aggregation
- Analyzer report section and two CSVs (`recovery_retargeting_by_iter`, `recovery_retargeting_worst_cases`)
- Defense classification on by default, with explicit opt-out
- IPC + trainer transport plumbing, mirroring Fix 1 / Fix 2 (reference commits: `52a51bdf5`, `2c5f56b71`, `d788023f4`)
- Unit tests including a synthetic anchor fixture reproducing the game-22 pattern

### 1.4 Out of scope

- Any change to move selection, MCTS priors, training targets, or auxiliary losses
- Resignation/adjudication tuning
- Recovery-aware training (deferred to Spec 5 only if §11 outcomes warrant it)
- Loser-side "wrong target" cases (high top1_share commitment to a wrong fragment) — different failure family

### 1.5 Relationship to existing diagnostics

- `recovery_or_extreme_closeout_drift` is winner-side, aggregated from `goal_completion_record`. Different signal.
- `recovery_events` (Spec 3 Fix 3, §6 of that spec) is scoped to dominant-unavailable in the winner. Different scope.
- This diagnostic observes loser-side MCTS state — neither of the above sees it.

The new sidecar key is `recovery_retargeting_summary`; the new per-game JSON key is `recovery_retargeting_record`. Distinct names, no collision.

---

## 2. Trigger

The diagnostic watches each side independently. The tracker maintains, per side, the previous own-move's `search_score` (initially `None`). On a side's own move, the trigger fires when either condition holds, with the joint top1_share gate applying to both:

```
diffuse_root = root_top1_share <= diffuse_root_top1_threshold     # default 0.20

delta_precursor =
    previous_own_search_score is not None
    AND (previous_own_search_score - current_search_score) >= delta_threshold   # default 0.50
    AND current_search_score <= delta_max_current_score                          # default -0.30
    AND diffuse_root

steady_state =
    current_search_score <= collapse_value_threshold                             # default -0.75
    AND diffuse_root

triggered = delta_precursor OR steady_state
```

`trigger_reason` per triggered move is one of `"delta_precursor"`, `"steady_state"`, `"both"`.

Severity sub-flags (recorded but not part of the trigger):

```
is_severe_collapse = current_search_score <= severe_collapse_value_threshold     # default -0.90
is_very_diffuse    = root_top1_share        <= very_diffuse_root_top1_threshold  # default 0.15
```

### 2.1 Score source

`current_search_score` is the root q-value from the side-to-move's perspective at the end of MCTS search, before any temperature sampling. Same source as the existing `move_root_values` list already populated in `self_play.py` around line 834. `root_top1_share` is the existing per-move top-1 visit share already collected into `move_top1_shares`. No new MCTS instrumentation needed.

### 2.2 Trigger window scope

The window for a side opens at the first triggered ply for that side and stays open until game end OR until that side resigns, whichever comes first. v1 windows do not close before game end/resignation. If this produces overly fat records in practice, a future version may add `close_after_n_recovered_own_moves`.

Once the window opens, every own-move of that side from then on is recorded and classified, even if the trigger condition is false on that ply. Plies before the first trigger are not recorded for that side. Each per-move record carries `triggered_this_ply: bool` and `trigger_reason: null | "delta_precursor" | "steady_state" | "both"`.

Non-triggered plies inside an open window are still classified and counted (with a flag `triggered_this_ply=false`), so the report can distinguish steady collapse from post-collapse recovery attempts.

### 2.3 Missing-signal behavior

If `current_search_score` or `root_top1_share` is missing for a ply, trigger evaluation is skipped. `previous_own_search_score` is updated only from valid `current_search_score` values. If the side is already in-window, missing-signal plies are counted separately and skipped for move-classification rates.

Counters:
- `missing_search_score_moves`
- `missing_root_top1_share_moves`
- `missing_signal_moves` (sum of the above, recorded for convenience)

`missing_signal_moves` are excluded from `selected_class_counts` and all class-rate denominators.

### 2.4 Per-side tracker state

- `previous_own_search_score: Optional[float]` — updated each valid own-move
- `triggered: bool` — whether the window has opened
- `first_trigger_ply: Optional[int]`
- `first_trigger_reason: Optional[str]`
- Counters and per-move records (§4)

---

## 3. Per-move classifier

### 3.1 Core component definitions

Used by every bucket below. Computed for the side-to-move on every classified ply.

```
own_components_before
  All same-color bridge-connected components on the board before the move
  is applied. A component is a set of pegs joined via TwixT bridges
  (knight-shape: (±1, ±2) or (±2, ±1) with no enemy peg blocking the
  bridge). Computed from the board state for the side-to-move.

dominant_component_before
  The largest component in own_components_before by peg count. Ties broken
  deterministically by the lexicographically-smallest peg coordinate
  contained in the component.

selected_component_after
  The component containing the placed peg AFTER the move is applied. At
  minimum, contains {selected_move}; may absorb one or more prior
  components if the placed peg bridges into them.

extends(component)
  True iff component.pegs ⊆ selected_component_after.pegs

opens_new_component
  True iff selected_component_after.pegs intersects no prior component
  (i.e., the placed peg is bridge-isolated from all prior same-color pegs).

merges_components
  True iff selected_component_after extends two or more prior components.

merges_dominant_with_alternate
  True iff selected_component_after extends dominant_component_before AND
  also extends at least one other prior component.

local_to_existing
  True iff the placed peg (r, c) has at least one prior same-color peg
  (r', c') such that (abs(r-r'), abs(c-c')) ∈ {(1, 2), (2, 1)} — TwixT
  bridge knight distance. The flag does NOT require that an actual bridge
  forms (the bridge could be blocked by an enemy peg). The flag is about
  proximity to bridge-able structure, not bridge formation.

own_total_goal_distance_before / _after
  From compute_goal_completion_state(state, side_to_move, enumerate_moves=False).
  None if not computable.

own_largest_component_size_before
  max(c.size for c in own_components_before).

own_largest_component_size_after
  Largest same-color bridge-connected component size computed from the
  post-move state.

opponent_total_goal_distance_before / _after
  From compute_goal_completion_state(state, opponent, enumerate_moves=False).
  Only computed when classify_defense=True (default).
```

### 3.2 Bucket priority order

The move classifies as the FIRST bucket whose rule fires:

```
1. blocks_opponent_closeout
2. reduces_own_goal_distance
3. starts_or_extends_alternate_component
4. connects_to_existing_component
5. improves_own_largest_component
6. redundant_local_reinforcement
7. off_plan_or_unclear
```

### 3.3 Bucket rules

1. **`blocks_opponent_closeout`** (only when `classify_defense=True`): `opponent_total_goal_distance_before` is not None AND ≤ 2 AND (`opponent_total_goal_distance_after` is None OR > the before value).

2. **`reduces_own_goal_distance`**: `own_total_goal_distance_before` and `_after` both not None AND `_after < _before`.

3. **`starts_or_extends_alternate_component`**: `selected_component_after` does NOT extend `dominant_component_before` AND (`opens_new_component` OR `selected_component_after` extends exactly one prior non-dominant component OR `selected_component_after` merges two or more prior non-dominant components) AND `selected_component_after.size >= alternate_component_min_size` (default 4).

4. **`connects_to_existing_component`**: `selected_component_after` extends ANY prior same-color component (dominant or otherwise), AND no higher-priority bucket fired.

5. **`improves_own_largest_component`**: `own_largest_component_size_after > own_largest_component_size_before` AND no higher-priority bucket fired.

6. **`redundant_local_reinforcement`**: `local_to_existing` is True AND `own_total_goal_distance` did NOT improve AND opponent closeout was NOT blocked AND no alternate-component bucket fired AND `own_largest_component_size` did NOT improve.

7. **`off_plan_or_unclear`**: fallback when none of 1–6 fired (or when classification raises after the trigger fired; see §6.3).

### 3.4 Per-move flags

Recorded on each sampled per-move entry, in addition to `primary_class`. Flags are descriptive and non-exclusive. `primary_class` is exclusive and priority-ordered. A move may have `extends_dominant_component=True` while `primary_class="reduces_own_goal_distance"`; this is expected because strategic progress takes priority over structural labels.

```
opens_new_component:                bool
merges_components:                  bool
merges_dominant_with_alternate:     bool
extends_dominant_component:         bool
local_to_existing:                  bool
blocked_opponent_closeout:          bool   # only set when classify_defense=True
own_total_goal_distance_before:     Optional[int]
own_total_goal_distance_after:      Optional[int]
own_largest_component_size_before:  int
own_largest_component_size_after:   int
opponent_total_goal_distance_before: Optional[int]   # only when classify_defense=True
opponent_total_goal_distance_after:  Optional[int]   # only when classify_defense=True
```

### 3.5 Rollups

Computed at game-end from the per-move primary_class counts:

```
constructive_recovery_moves =
    reduces_own_goal_distance
  + starts_or_extends_alternate_component

defensive_moves =
    blocks_opponent_closeout

structural_connection_moves =
    connects_to_existing_component
  + improves_own_largest_component

local_drift_moves =
    redundant_local_reinforcement
  + off_plan_or_unclear
```

The four rollups partition the same denominator (`classified_in_window_moves`) and should sum to 1.0 except for floating-point rounding. The report shows both the granular primary_class counts and these rollups. `structural_connection_moves` are not counted as constructive recovery unless they also reduce own goal distance or build a qualifying alternate component.

---

## 4. Per-game record schema

Emitted on the saved game JSON as top-level key `recovery_retargeting_record`, present only when at least one side had its window opened. Mirrors how `goal_completion_record` is conditionally written.

```jsonc
"recovery_retargeting_record": {
  "version": 1,
  "iteration": 170,
  "game_idx": 22,
  "game_id": "game_022",
  "winner": "red",
  "loser": "black",
  "starting_player": "red",
  "n_moves": 65,
  "reason": "win",
  "classifier_error_count": 0,
  "config": {
    "collapse_value_threshold":           -0.75,
    "severe_collapse_value_threshold":    -0.90,
    "diffuse_root_top1_threshold":         0.20,
    "very_diffuse_root_top1_threshold":    0.15,
    "delta_threshold":                     0.50,
    "delta_max_current_score":            -0.30,
    "alternate_component_min_size":          4,
    "classify_defense":                   true
  },
  "triggered_sides":      ["black"],
  "first_trigger_ply":     44,
  "first_trigger_side":    "black",
  "first_trigger_reason": "delta_precursor",
  "side_records": {
    "red":   { "triggered": false, "classifier_error_count": 0 },
    "black": {
      "triggered": true,
      "first_trigger_ply":     44,
      "first_trigger_reason": "delta_precursor",
      "classifier_error_count": 0,

      "in_window_own_moves":           11,
      "triggered_own_moves":           11,
      "non_triggered_in_window_moves":  0,
      "missing_signal_moves":           0,
      "missing_search_score_moves":     0,
      "missing_root_top1_share_moves":  0,

      "trigger_reason_counts": {
        "delta_precursor": 1,
        "steady_state":   10,
        "both":            0
      },

      "severe_collapse_moves":  6,
      "very_diffuse_moves":     8,

      "mean_search_score_triggered_plies":   -0.83,
      "min_search_score_triggered_plies":    -0.99,
      "max_search_score_triggered_plies":    -0.40,
      "mean_root_top1_share_triggered_plies": 0.12,

      "classified_in_window_moves": 11,
      "selected_class_counts": {
        "blocks_opponent_closeout":               1,
        "reduces_own_goal_distance":              0,
        "starts_or_extends_alternate_component":  0,
        "connects_to_existing_component":         3,
        "improves_own_largest_component":         2,
        "redundant_local_reinforcement":          5,
        "off_plan_or_unclear":                    0
      },

      "constructive_recovery_moves":   0,
      "defensive_moves":               1,
      "structural_connection_moves":   5,
      "local_drift_moves":             5,

      "constructive_recovery_rate":   0.000,
      "defensive_rate":               0.091,
      "structural_connection_rate":   0.455,
      "local_drift_rate":             0.455,

      "sampled_moves_count":   11,
      "sampled_moves_cap":     32,
      "sampled_moves_dropped":  0,
      "sample_all_moves":      false,
      "sampled_moves":         [ ... see §4.1 ... ]
    }
  }
}
```

### 4.1 Per-move sampled detail

Bounded list at `recovery_retargeting_record.side_records.<side>.sampled_moves`, capped at `max_sampled_moves_per_side` (default 32, override via `sample_all_moves=True`). If the cap is exceeded, entries are retained in priority order, then ply order. **No random sampling.**

Sampling priority within a side's window:

1. Every triggered ply within the first 4 plies after `first_trigger_ply` (the inflection region).
2. Every severe-collapse ply (`is_severe_collapse=True`).
3. Remaining plies in window order until the cap is hit.

Sampled entries may include `triggered_this_ply=false` entries when they occur after the window opens; those entries have `trigger_reason=null` but still receive `primary_class` classification if search_score/root_top1_share are present.

Each sampled entry contains:

```jsonc
{
  "ply": 44,
  "triggered_this_ply": true,
  "trigger_reason": "delta_precursor",
  "current_search_score": -0.40,
  "previous_own_search_score": 0.30,
  "search_score_delta": -0.70,
  "root_top1_share": 0.12,
  "is_severe_collapse": false,
  "is_very_diffuse": true,
  "primary_class": "redundant_local_reinforcement",
  "selected_move": [2, 11],
  "flags": {
    "opens_new_component":              false,
    "merges_components":                false,
    "merges_dominant_with_alternate":   false,
    "extends_dominant_component":        true,
    "local_to_existing":                 true,
    "blocked_opponent_closeout":        false
  },
  "own_total_goal_distance_before":      6,
  "own_total_goal_distance_after":       6,
  "own_largest_component_size_before":  14,
  "own_largest_component_size_after":   15,
  "opponent_total_goal_distance_before": 4,
  "opponent_total_goal_distance_after":  4
}
```

### 4.2 Notes on the schema

- `loser` is `None` for draws; both `side_records` keys are still present so a draw with one side collapsed still records cleanly.
- `config` echoes the thresholds at time of run. Future cross-iteration comparisons can detect drift.
- Stored rates are convenience fields only. Integer counts are authoritative; the analyzer recomputes all corpus-level rates from counts.
- `selected_class_counts` denominator is `classified_in_window_moves` = sum(`selected_class_counts`). The four rollup rates partition that denominator.

---

## 5. Runtime integration

### 5.1 New module

`scripts/GPU/alphazero/recovery_retargeting_diagnostics.py`. Self-contained; depends only on `connectivity_diagnostics.compute_goal_completion_state` and the board-state helpers already used by `closeout_diagnostics`.

```python
@dataclass
class RecoveryRetargetingConfig:
    enabled: bool = True
    collapse_value_threshold: float = -0.75
    severe_collapse_value_threshold: float = -0.90
    diffuse_root_top1_threshold: float = 0.20
    very_diffuse_root_top1_threshold: float = 0.15
    delta_threshold: float = 0.50
    delta_max_current_score: float = -0.30
    alternate_component_min_size: int = 4
    classify_defense: bool = True
    max_sampled_moves_per_side: int = 32
    sample_all_moves: bool = False
```

11 fields total.

```python
class RecoveryRetargetingTracker:
    """Per-game tracker. One instance per game; lifecycle matches play_game."""

    def __init__(self, config: RecoveryRetargetingConfig): ...

    def observe_move(
        self,
        *,
        state_before,
        selected_move: Tuple[int, int],
        ply: int,
        side_to_move: str,
        search_score: Optional[float],
        root_top1_share: Optional[float],
    ) -> None: ...

    def finalize_game(
        self,
        *,
        iteration: int,
        game_idx: int,
        game_id: str,
        winner: Optional[str],
        starting_player: str,
        n_moves: int,
        reason: str,
    ) -> Optional[dict]: ...
```

### 5.2 `observe_move` semantics

`observe_move` is called after MCTS search and final move selection, before the main game state applies the move. The tracker first evaluates the trigger using the provided `search_score`/`root_top1_share` and its per-side `previous_own_search_score`. If the side newly triggers or is already in-window, the tracker computes `state_after` internally (`state_after = state_before.apply_move(selected_move)`) and classifies the selected move. If the side is not in-window and does not trigger, no `state_after` or classifier work is performed.

Cost control note: the double `apply_move` call (tracker + game loop) only fires on in-window plies. With expected trigger rate ~10–15% of loser plies × ~50 plies/game × 100 games/iter ≈ 500–750 extra `apply_move` calls per iter. Negligible vs. 400-sim MCTS.

### 5.3 Defensive behavior

If `search_score` or `root_top1_share` is missing, trigger evaluation is skipped. `previous_own_search_score` is updated only from valid `current_search_score` values. If the side is already in-window, increment `missing_signal_moves` and do not classify the move.

If classification itself raises after a valid in-window ply, increment `classifier_error_count` (both top-level and per-side) and classify that ply as `off_plan_or_unclear`. Log a single warning per game (not per ply).

### 5.4 Hook point in `play_game`

The hook lives in the existing per-ply loop, between MCTS search and the move-application call. Variables it needs (`root_value`, the per-move `top1_share`) are already in scope:

```python
visit_counts, root_value, root = mcts.search_from_root(...)
# ... existing Fix 1 / Fix 2 logic (closeout_td1, tiebreak) ...
move = mcts.select_move(visit_counts, ply)
top1_share = _compute_top1_share(visit_counts)

if recovery_tracker is not None and recovery_tracker.config.enabled:
    recovery_tracker.observe_move(
        state_before=state,
        selected_move=move,
        ply=ply,
        side_to_move=state.to_move,
        search_score=root_value,
        root_top1_share=top1_share,
    )

state = state.apply_move(move)
```

### 5.5 Game-end wiring (self_play.py)

After the existing `goal_completion_record` finalization at line ~1296:

```python
recovery_retargeting_record = (
    recovery_tracker.finalize_game(
        iteration=0,                       # populated downstream like goal_completion_record
        game_idx=game_id,                  # play_game's local counter (dispatch-order);
                                           # game_saver overrides to save-order via §5.8
        game_id=f"game_{game_id:03d}",
        winner=winner,
        starting_player=start_player,
        n_moves=len(move_history),
        reason=_gc_reason_for_record,
    )
    if recovery_tracker is not None else None
)
```

The `game_id` here is `play_game`'s function parameter (the dispatch-order counter). The save-order reconciliation in §5.8 overrides the persisted record's `game_idx` and `game_id` to match the saver's authoritative counter. This is exactly the pattern `goal_completion_record` follows post-`32c4966a6`.

On `GameRecord` add:

```python
recovery_retargeting_record: Optional[dict] = None
```

### 5.6 IPC + trainer transport

Mirror the Fix 2 pattern from `d788023f4` exactly. Required changes:

- `GameRecord.recovery_retargeting_record` field (defaults None)
- `ipc_messages.GameComplete.recovery_retargeting_record` field (defaults None)
- `self_play_worker.py` forwards `game.recovery_retargeting_record` into `GameComplete` parallel to existing Fix 1/Fix 2 forwarding (direct attribute access; field is declared on GameRecord)
- `trainer.train()` kwargs: 11 new params (the 11 config fields above)
- `trainer.train()` threads them into a new `RecoveryRetargetingConfig` passed as a kwarg into `play_game` (NOT into `MCTSConfig` — this tracker is play_game-local, not MCTS-internal)
- `trainer.train()` adds `all_recovery_retargeting_records: list = []` at both inner-parallel and outer scopes
- IPC append branch: `msg.recovery_retargeting_record`
- Serial-path append: defensive `getattr(game, "recovery_retargeting_record", None)`
- `_inject_iteration` pattern extended to also inject `iteration` into `recovery_retargeting_record`
- Sidecar emit: `_sidecar["recovery_retargeting_summary"] = aggregate_recovery_retargeting_records(all_recovery_retargeting_records, games_total=<iter game count>)` immediately after the existing Fix 2 sidecar emit
- `train.py`: new CLI flags per §7.1 wired into `train_kwargs.update(...)`
- Startup banner block in `trainer.py` immediately after the Fix 2 banner

### 5.7 Sidecar aggregation consumes records, not tracker state

Trainer sidecar aggregation consumes the finalized `recovery_retargeting_record` dicts, not live tracker objects. Workers serialize records into `GameComplete`; the trainer collects them; the aggregator runs at iteration end.

### 5.8 Game-saver fix already in place

The `32c4966a6` `game_saver` reconciliation handles `goal_completion_record.game_idx`. The same pattern extends to `recovery_retargeting_record.game_idx` and `game_id`. Add to the conditional block in `game_saver.save_game_replay`:

```python
if recovery_retargeting_record is not None:
    record["recovery_retargeting_record"] = {
        **recovery_retargeting_record,
        "game_idx": game_idx,
        "game_id": f"game_{game_idx:03d}",
    }
```

---

## 6. Sidecar aggregation, analyzer report, and CSVs

### 6.1 Per-iteration sidecar

Trainer writes one block per training iteration into `iter_NNNN_stats.json`, parallel to Fix 1 / Fix 2 telemetry:

```jsonc
"recovery_retargeting_summary": {
  "version": 1,
  "enabled": true,
  "config": {
    "collapse_value_threshold":           -0.75,
    "severe_collapse_value_threshold":    -0.90,
    "diffuse_root_top1_threshold":         0.20,
    "very_diffuse_root_top1_threshold":    0.15,
    "delta_threshold":                     0.50,
    "delta_max_current_score":            -0.30,
    "alternate_component_min_size":          4,
    "classify_defense":                   true
  },

  "games_total":               100,
  "games_triggered":            17,
  "trigger_rate":                0.170,
  "triggered_loser_side":       16,
  "triggered_winner_side":       2,
  "triggered_loser_side_per_triggered_game":  0.941,
  "triggered_winner_side_per_triggered_game": 0.118,

  "in_window_own_moves_total":           162,
  "triggered_own_moves_total":           142,
  "non_triggered_in_window_moves_total":  20,
  "missing_signal_moves_total":            0,
  "severe_collapse_moves_total":          61,
  "very_diffuse_moves_total":             48,

  "trigger_reason_counts_total": {
    "delta_precursor":  17,
    "steady_state":    101,
    "both":              4
  },

  "classified_in_window_moves_total": 162,
  "selected_class_counts_total": {
    "blocks_opponent_closeout":              13,
    "reduces_own_goal_distance":              7,
    "starts_or_extends_alternate_component":  5,
    "connects_to_existing_component":         29,
    "improves_own_largest_component":         20,
    "redundant_local_reinforcement":          70,
    "off_plan_or_unclear":                    18
  },

  "selected_class_rates_total": {
    "blocks_opponent_closeout":              0.080,
    "reduces_own_goal_distance":             0.043,
    "starts_or_extends_alternate_component": 0.031,
    "connects_to_existing_component":        0.179,
    "improves_own_largest_component":        0.123,
    "redundant_local_reinforcement":         0.432,
    "off_plan_or_unclear":                   0.111
  },

  "constructive_recovery_rate":   0.074,
  "defensive_rate":               0.080,
  "structural_connection_rate":   0.302,
  "local_drift_rate":             0.543,

  "schema_integrity": {
    "skipped_unknown_version_count":  0,
    "skipped_config_mismatch_count":  0,
    "classifier_error_count_total":   0
  }
}
```

### 6.2 Aggregator signature

```python
def aggregate_recovery_retargeting_records(
    records: list[dict],
    *,
    games_total: int,
    config: dict | None = None,
) -> dict
```

`games_total` is the iteration's full game count (passed by trainer); records only exist when triggered. Without `games_total`, the aggregator can't emit `trigger_rate`.

### 6.3 Aggregator semantics

- Sums integer counts across per-game records; recomputes all rates from counts.
- Skips records with `version != 1`; increments `schema_integrity.skipped_unknown_version_count`.
- Per-iteration: asserts all records' configs match the first record's config; mismatches increment `schema_integrity.skipped_config_mismatch_count` and the mismatched record is skipped.
- The first non-skipped record's `config` block becomes the sidecar's `config` block.

### 6.4 Analyzer cross-iteration rollup

When the analyzer rolls up `recovery_retargeting_summary` across iterations (in `analyze()`), mixed configs are **NOT** skipped by default. Aggregate counts as-is; print a "mixed config" warning in `report_<range>.txt` and list the configs by iteration. This avoids silently dropping useful records if thresholds are tuned between training runs.

### 6.5 Analyzer report section

Added to `report_<range>.txt` immediately after the existing `Closeout selection tie-break` block:

```
Recovery / Re-targeting Diagnostics
===================================
Iters covered: 170-179  enabled=True  defense_classifier=on
Config: collapse_value<=-0.75  diffuse_root_top1<=0.20  delta>=0.50 with current<=-0.30
Triggered games:           143 / 1000 (14.3%)
  side was eventual loser: 136 / 143 (95.1%)
  side was eventual winner:  9 / 143 (6.3%)
In-window own moves:       1,284
  triggered:               1,108
  non-triggered in-window:   176
  missing-signal:              0
Severity:
  severe collapse:           522 plies
  very diffuse root:         914 plies
Trigger composition:
  delta_precursor:           177
  steady_state:              859
  both:                       72
Move-class composition (denominator: classified in-window):
  blocks opponent closeout:              8.1%   (104)
  reduces own goal distance:             4.3%   ( 55)
  starts/extends alternate component:    3.2%   ( 41)
  connects to existing component:       18.0%   (231)
  improves own largest component:       12.4%   (159)
  redundant local reinforcement:        42.7%   (548)
  off-plan or unclear:                  11.3%   (146)
Rollup:
  constructive recovery:                 7.5%
  defense:                               8.1%
  structural connection:                30.4%
  local drift / unclear:                54.0%
Schema integrity:
  classifier_error_count:                0
  records skipped (unknown version):     0
  records skipped (config mismatch):     0
Worst cases: recovery_retargeting_worst_cases.csv
```

If `classify_defense` was false during the run, an explicit warning line replaces the defense row:
```
  defense:                  N/A (defense classification disabled — local drift may include defensive moves)
```

If configs differ across iterations in the rolled-up range:
```
Mixed config across iters covered:
  iter 170-174: collapse_value<=-0.75 delta>=0.50
  iter 175-179: collapse_value<=-0.70 delta>=0.40
  WARNING: rates aggregate across config changes; treat with care.
```

### 6.6 `recovery_retargeting_by_iter.csv`

One row per iteration in the analyzed range. Columns:

```
iteration, games_total, games_triggered, trigger_rate,
triggered_loser_side, triggered_winner_side,
triggered_loser_side_per_triggered_game,
in_window_own_moves_total, triggered_own_moves_total,
severe_collapse_moves_total, very_diffuse_moves_total,
classified_in_window_moves_total, classifier_error_count_total,
constructive_recovery_rate, defensive_rate,
structural_connection_rate, local_drift_rate,
redundant_local_reinforcement_rate, off_plan_or_unclear_rate,
trigger_delta_precursor_count, trigger_steady_state_count, trigger_both_count
```

### 6.7 `recovery_retargeting_worst_cases.csv`

Top N games (N = `--recovery-retargeting-worst-cases-top-k` per §7.1.1, default 25) sorted by:

```
(local_drift_moves DESC, in_window_own_moves DESC, min_search_score_triggered_plies ASC)
```

**Two-row split for two-triggered-sides games.** If both sides triggered in a game, the CSV writes one row per triggered side. The `triggered_side` column distinguishes them. The sort key applies per-row.

Columns:

```
iteration, game_idx, game_id, winner, loser, reason, n_moves,
triggered_side, first_trigger_ply, first_trigger_reason,
in_window_own_moves, triggered_own_moves,
severe_collapse_moves, very_diffuse_moves,
classified_in_window_moves, missing_signal_moves,
blocks_opponent_closeout_moves, reduces_own_goal_distance_moves,
starts_or_extends_alternate_component_moves,
connects_to_existing_component_moves, improves_own_largest_component_moves,
redundant_local_reinforcement_moves, off_plan_or_unclear_moves,
constructive_recovery_moves, defensive_moves,
structural_connection_moves, local_drift_moves,
local_drift_rate, constructive_recovery_rate,
mean_search_score_triggered_plies, min_search_score_triggered_plies,
max_search_score_triggered_plies, mean_root_top1_share_triggered_plies
```

`iteration` and `game_idx` correctly point at the right file via the `32c4966a6` reconciliation. Analyzer joins on `(iteration, game_idx)` against `meta.iteration` and `meta.game_idx` in the per-game JSONs.

---

## 7. CLI flags & startup banner

### 7.1 Flags

Added to `scripts/GPU/alphazero/train.py` after the Fix 2 flag block. Recovery diagnostic defaults to ON; an explicit disable flag is provided:

```
--recovery-retargeting-disabled                          store_true; default False (i.e. diagnostic is ON)
--recovery-retargeting-collapse-value-threshold          float, default -0.75
--recovery-retargeting-severe-value-threshold            float, default -0.90
--recovery-retargeting-diffuse-root-top1-threshold       float, default  0.20
--recovery-retargeting-very-diffuse-root-top1-threshold  float, default  0.15
--recovery-retargeting-delta-threshold                   float, default  0.50
--recovery-retargeting-delta-max-current-score           float, default -0.30
--recovery-retargeting-alternate-component-min-size      int,   default  4
--recovery-retargeting-classify-defense                  store_true target; default True
--recovery-retargeting-no-classify-defense               store_false on the above (mutex pair)
--recovery-retargeting-max-sampled-moves-per-side        int,   default  32
--recovery-retargeting-sample-all-moves                  store_true, default False
```

### 7.1.1 Analyzer CLI flag

Added to `scripts/twixt_replay_analyzer.py` (NOT to `train.py` — this flag tunes the analyzer's CSV output, not the training run):

```
--recovery-retargeting-worst-cases-top-k                 int,   default 25
```

Matches the convention used by `goal_completion_worst_cases_top_k`.

### 7.2 Validation

Applied in `train.py` after argparse, before constructing `RecoveryRetargetingConfig`. Uses `parser.error(...)` or `raise ValueError` in config construction — never raw `assert` (which is disabled under Python optimized mode):

```
collapse_value_threshold < delta_max_current_score
severe_collapse_value_threshold <= collapse_value_threshold
very_diffuse_root_top1_threshold <= diffuse_root_top1_threshold
0.0 <= diffuse_root_top1_threshold <= 1.0
0.0 <= very_diffuse_root_top1_threshold <= 1.0
delta_threshold > 0
alternate_component_min_size >= 1
max_sampled_moves_per_side >= 0
```

Fail-fast at startup; never let an out-of-band config silently produce nonsense data over 24h of GPU.

### 7.3 Startup banner

Added to `trainer.py` immediately after the Fix 2 banner block:

```
  Recovery / re-targeting diagnostics: enabled
    collapse_value <=         -0.75
    severe_value <=           -0.90
    diffuse_root_top1 <=       0.20
    delta >=                   0.50
    delta_max_current_score:  -0.30
    alternate_component_min_size:  4
    classify_defense:          on
    sample_all_moves:          off  (cap=32 per side)
```

If `--recovery-retargeting-no-classify-defense` is passed:

```
    classify_defense:          off  (WARNING: local_drift may include defensive moves)
```

If `--recovery-retargeting-disabled`:

```
  Recovery / re-targeting diagnostics: disabled
```

The banner is the canary for "flag-not-parsed / flag-not-wired" bugs. Same role it played for Fix 1 (`1f0045070`) and Fix 2 (Task 23 Step 7 of the `d788023` plan).

---

## 8. Tests

`tests/test_recovery_retargeting_diagnostics.py`. Synthetic fixtures; do NOT depend on full game JSONs.

### 8.1 Trigger

- `test_steady_state_trigger_fires_when_score_and_top1_both_low`
- `test_steady_state_does_not_fire_when_score_bad_but_root_confident`
- `test_steady_state_does_not_fire_when_root_diffuse_but_score_ok`
- `test_delta_precursor_fires_on_sharp_drop`
- `test_delta_precursor_guard_blocks_when_current_score_still_positive`
- `test_trigger_reason_both_when_both_paths_fire`
- `test_window_stays_open_across_non_triggered_plies`
- `test_window_does_not_close_within_game`

### 8.2 Missing signal

- `test_missing_search_score_does_not_fire_trigger_and_does_not_update_previous`
- `test_missing_root_top1_share_does_not_fire_trigger`
- `test_in_window_missing_signal_increments_missing_signal_moves_and_skips_classification`

### 8.3 Classifier

- `test_classifies_blocks_opponent_closeout`
- `test_classifies_reduces_own_goal_distance`
- `test_classifies_starts_or_extends_alternate_component_via_opens_new`
- `test_classifies_starts_or_extends_alternate_via_extending_non_dominant`
- `test_classifies_connects_to_existing_component`
- `test_classifies_improves_own_largest_component`
- `test_classifies_redundant_local_reinforcement`
- `test_classifies_off_plan_or_unclear_fallback`
- `test_priority_defense_beats_reduces_goal_distance`
- `test_priority_reduces_goal_distance_beats_alternate_component`
- `test_extends_dominant_with_reduces_distance_flags_extends_dominant_true_classifies_as_reduces`
- `test_local_to_existing_uses_knight_not_chebyshev`
- `test_merges_dominant_with_alternate_flag_set_but_primary_class_routes_correctly`

### 8.4 Defense classifier opt-out

- `test_classify_defense_default_on`
- `test_defensive_move_classifies_before_local_reinforcement`
- `test_classify_defense_disabled_skips_bfs_and_marks_report_warning`
- `test_classify_defense_uses_enumerate_moves_false`

### 8.5 Finalize game

- `test_finalize_emits_record_only_when_at_least_one_side_triggered`
- `test_finalize_returns_none_when_no_side_triggered`
- `test_per_side_record_includes_classified_in_window_moves_denominator`
- `test_per_side_record_rates_partition_to_one_modulo_rounding`
- `test_sampled_moves_priority_first_4_after_trigger_then_severe_then_window_order`
- `test_sampled_moves_dropped_count_when_cap_exceeded`
- `test_sample_all_moves_disables_cap`

### 8.6 Aggregator + sidecar

- `test_aggregator_sums_counts_and_recomputes_rates`
- `test_aggregator_passes_games_total_through_to_sidecar`
- `test_aggregator_skips_unknown_version_and_counts_skip`
- `test_aggregator_per_iteration_warns_on_config_mismatch_and_skips`
- `test_aggregator_cross_iteration_does_not_skip_on_config_mismatch_but_records_mixed_config`

### 8.7 Analyzer + CSV

- `test_analyzer_report_includes_recovery_retargeting_section_when_sidecars_present`
- `test_analyzer_report_warning_when_classify_defense_was_off_in_any_iter`
- `test_analyzer_by_iter_csv_columns_and_one_row_per_iteration`
- `test_analyzer_worst_cases_csv_two_rows_for_dual_triggered_game`
- `test_analyzer_worst_cases_top_k_arg_caps_row_count`

### 8.8 Anchor regression

- `test_iter_0169_game_022_synthetic_fixture_classifies_local_drift` — small synthetic board, NOT the full game JSON. Constructs a board state where a same-color peg has multiple knight-distance neighbors, an opponent close to td=2, and a move that:
  - is `local_to_existing`
  - does NOT reduce own goal_distance
  - does NOT block opponent closeout (opponent td unchanged)
  - does NOT start an alternate-component
  - does NOT improve `own_largest_component_size`
  → must classify as `redundant_local_reinforcement`.

---

## 9. Verification workflow

1. **Unit tests pass.** `.venv/bin/pytest tests/test_recovery_retargeting_diagnostics.py -v`
2. **Full regression suite still passes.** `.venv/bin/pytest tests/test_mcts*.py tests/test_analyzer_*.py tests/test_self_play_closeout*.py tests/test_train_closeout*.py tests/test_game_saver*.py tests/test_recovery_retargeting_diagnostics.py -q` — confirms no regression to Fix 1 / Fix 2 / game-saver fix paths. Baseline: 168 passed, 2 skipped. Expected after this spec: 168 + (count of new tests) passed, 2 skipped.
3. **CLI smoke.** `.venv/bin/python -m scripts.GPU.alphazero.train --help | grep recovery-retargeting` — confirm the key flags appear: `--recovery-retargeting-disabled`, `--recovery-retargeting-collapse-value-threshold`, `--recovery-retargeting-diffuse-root-top1-threshold`, `--recovery-retargeting-no-classify-defense`, `--recovery-retargeting-worst-cases-top-k`. Do NOT lock the exact flag count — depends on argparse output format.
4. **5–10 game smoke run.** Resume from `model_iter_0169.safetensors`, run 1 iteration × 5 games (~10 minutes). Inspect:
   - Banner shows all three diagnostic blocks (Fix 1, Fix 2, recovery) as enabled.
   - Sidecar `recovery_retargeting_summary` block is well-formed regardless of trigger rate. If no game triggers, the sidecar should still contain `recovery_retargeting_summary` with `games_triggered=0` and `enabled=true`. If a trigger appears, inspect at least one `recovery_retargeting_record`.
5. **Launch the 170–179 production block** with all flags at defaults. Total run time ≈ 24h matches 160-169's profile.
6. **Post-run analysis.** Stage 170-179 games + sidecars into `Replays/170-179/`, run the analyzer, inspect:
   - `Recovery / Re-targeting Diagnostics` section in `report_170-179.txt`
   - `recovery_retargeting_worst_cases_170-179.csv` — sort confirms; sanity-check a top row by opening the game file.
   - `recovery_retargeting_by_iter_170-179.csv` — per-iter trend.
   - Existing Fix 1 / Fix 2 / Spec 3 metrics — confirm no regression vs 160-169.

---

## 10. Out-of-scope work (deferred)

- Mid-game intervention when the diagnostic fires (Dirichlet noise widening, temperature adjustment) — Spec 5 candidate.
- Loser-side recovery-aware aux loss — Spec 5 candidate.
- Curated recovery probes for the loser side — Spec 5 candidate.
- "Wrong target" loser-side cases (high top1_share commitment to a wrong fragment) — different failure family; future spec.

No interventions ship in this diagnostic spec regardless of §11's outcome.

---

## 11. Decision rule (post-170-179)

Based on the aggregate sidecar over 170-179:

| Observation | Implication |
|---|---|
| `trigger_rate ≥ 10%` AND `triggered_loser_side_per_triggered_game ≥ 80%` AND `local_drift_rate ≥ 40%` | Pattern confirmed as common and loser-side-dominant. Spec 5 brainstorm (intervention) is warranted. |
| `trigger_rate ≥ 10%` AND `constructive_recovery_rate ≥ 30%` | Sides DO try to re-target after collapse; "confused connector" is a minority pattern. Spec 5 not urgent. |
| `trigger_rate < 5%` | Pattern is rare; observed game-22 example is an outlier. No follow-up. |
| `local_drift_rate ≥ 30%` AND `structural_connection_rate ≥ 30%` AND `redundant_local_reinforcement_rate` is low | Review sampled moves before calling it confused drift; the classifier may be grouping legitimate rebuilding as structural connection. Hand-review required before designing intervention. |
| `classifier_error_count_total > 1%` of `in_window_moves_total` | Implementation bug. Fix the diagnostic before drawing conclusions. |
| `triggered_winner_side_per_triggered_game > 20%` | Surprising — investigate. May indicate trigger thresholds are too sensitive (winners shouldn't collapse). |

The decision rule is consulted by hand after the 170-179 analyzer pass. No automated gating.

---

## 12. Reference: file paths

- `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py` — new module
- `scripts/GPU/alphazero/self_play.py` — `play_game` hook + `GameRecord` field + game-end finalize call
- `scripts/GPU/alphazero/ipc_messages.py` — `GameComplete` field
- `scripts/GPU/alphazero/self_play_worker.py` — IPC forwarding
- `scripts/GPU/alphazero/trainer.py` — `train()` kwargs, IPC append, serial append, `_inject_iteration` extension, sidecar emit, startup banner
- `scripts/GPU/alphazero/train.py` — CLI flags, validation, `train_kwargs.update`
- `scripts/GPU/alphazero/game_saver.py` — `recovery_retargeting_record.game_idx` / `game_id` reconciliation
- `scripts/twixt_replay_analyzer.py` — `aggregate_recovery_retargeting_records` + report formatter + CSV writers, wired into `analyze()`
- `tests/test_recovery_retargeting_diagnostics.py` — new test file
