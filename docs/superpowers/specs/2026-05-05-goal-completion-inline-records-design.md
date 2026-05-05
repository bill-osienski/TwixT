# Goal-Completion Inline Records — Design Spec

**Date:** 2026-05-05
**Status:** Approved (brainstorm complete)
**Predecessor:** [2026-05-03-goal-completion-diagnostics-design.md](2026-05-03-goal-completion-diagnostics-design.md) (Spec 1, shipped)
**Successor:** Spec 2 (Phases 5–6: closeout-aware probe tier + training correction knobs) — to be designed after this spec ships and 1–2 fresh training iterations carry inline records.

---

## 1. Goal

Move per-game goal-completion classification from replay-side BFS aggregation into self-play inline emission. The analyzer becomes a consumer of pre-computed records; it never recomputes goal-completion state by default.

**Why this matters:** The analyzer's current Phase 2 path walks every replay's move history, calls `compute_goal_completion_state` per ply, and runs `classify_selected_conversion_move` for every winner ply post-detection. On a 1010-game corpus this takes 2+ hours of pure-Python BFS work, blocking analytical iteration on Spec 2 design.

**The architecture invariant:**

> Goal-completion record generation is part of self-play, not analysis. The analyzer never recomputes goal-completion state by default; it consumes persisted per-game records and per-iteration sidecar summaries. Recompute exists only as an explicit `--goal-completion-recompute` debug/backfill mode.

**Sequencing:** Spec 1.5 ships → run a few iterations → review records → start Spec 2 brainstorm with fast feedback loops.

---

## 2. Why pre-move detection

Pre-move detection measures the first ply where the player to move already has a closeout-shaped position and therefore has an opportunity to finish or reduce distance. Earlier post-move detection (Spec 1 Phase 2) measured the ply that *created* the closeout, which is useful structurally but less useful for judging whether the model missed a conversion opportunity. Because of this semantic change, `first_dominant_unclosed_ply` may shift later by roughly one same-side move compared with older recompute reports — this is expected and not a model regression.

Both the inline tracker and the `--goal-completion-recompute` legacy walker use pre-move semantics, so cross-path comparisons are apples-to-apples.

---

## 3. Architecture & data flow

```
[self-play worker]
  per ply (after MCTS, after move selection, before apply_move):
    gc_state_cheap = compute_goal_completion_state(state, side, enumerate_moves=False)
    decide need_full = (Phase 3 emit needs it) OR (tracker needs classification)
    gc_state_full = upgrade with enumerate_moves=True if needed
    Phase 3 closeout_diagnostics consumes gc_state_full when emit enabled
    tracker.observe_pre_move(state, ply, side_to_move, selected_move,
                             search_score, gc_state_cheap, gc_state_full)
  game end:
    record = tracker.finalize_game(...)
    record attached to GameRecord.goal_completion_record

[trainer]
  reconstitutes GameRecord (incl. goal_completion_record)
  saver writes top-level "goal_completion_record" key in per-game JSON
  per-iteration aggregator: aggregate_goal_completion_records(records, config)
  writes "goal_completion_summary" block into iter_NNNN_stats.json sidecar

[analyzer]
  default path: read goal_completion_record from per-game JSONs, aggregate
                via shared aggregator. Sidecar summaries used for validation.
  no replay reconstruction, no BFS, no compute_goal_completion_state
  worst-cases CSV populated from per-game records
  --goal-completion-recompute: opt-in legacy path (pre-move semantics)
  --goal-completion-recompute-validate: full-corpus comparison of inline vs recomputed
```

**Cost summary:**

- One cheap BFS per ply when `goal_completion_record_enabled=True` (default). Already paid by Phase 3 too — no duplication.
- One full BFS per ply when `gc_state_full` is needed: post-detection on the detected side, or `total_goal_distance ≤ emit_threshold` (Phase 3 emit on). Strictly cheaper than the analyzer's current per-ply enumeration.
- Zero goal-completion overhead when both `record_enabled=False` and `emit_enabled=False`.

---

## 4. Per-game record schema

**Field**: `replay["goal_completion_record"]` — single dict per game, present on every game when `goal_completion_record_enabled=True`.

**Single schema, nullable fields, `outcome_class` discriminator.** All games emit the same shape; analyzer partitions by `outcome_class`.

### 4.1 Class 1 — decisive winner (`outcome_class == 1`)

```json
{
  "version": 1,
  "game_id": "iter_0112_game_034",
  "iteration": 112,
  "game_idx": 34,
  "winner": "red",
  "detected_player": "red",
  "starting_player": "red",
  "n_moves": 59,
  "reason": "win",
  "outcome_class": 1,
  "scope": "winner",
  "ever_distance_le_2": true,
  "ever_distance_le_3": true,
  "min_total_goal_distance": 2,
  "detected": true,
  "first_dominant_unclosed_ply": 37,
  "first_total_goal_distance": 2,
  "first_category": "two_endpoint_closeout_2ply",
  "first_largest_component_size": 12,
  "first_endpoint_distances": {"top": 0, "bottom": 1},
  "actual_terminal_ply": 59,
  "actual_win_ply": 59,
  "conversion_delay_plies": 22,
  "conversion_delay_winner_moves": 11,
  "cap_delay_proxy_plies": null,
  "winner_moves_in_watch_window": 12,
  "winner_moves_with_dominant_component": 12,
  "winner_moves_with_dominant_unavailable": 0,
  "primary_class_counts": {
    "completes_endpoint": 2,
    "reduces_total_goal_distance": 0,
    "redundant_reinforcement": 8,
    "off_chain": 2,
    "other": 0
  },
  "max_search_score_after_detection": 0.9987,
  "mean_search_score_after_detection": 0.97,
  "high_value_after_detection_plies": 12,
  "root_value_high_but_delayed": true,
  "search_score_coverage_in_watch_window": 12
}
```

> **Naming note**: Internally, `_SideAccumulator.moves_after_detection` accumulates the focal side's post-detection move count. At Class 1 finalize, this value maps to `winner_moves_in_watch_window` in the persisted record (preserving continuity with the existing analyzer's report shape). The persisted record uses the analyzer-facing name only.

### 4.2 Class 2 — capped / timeout / board-full (`outcome_class == 2`)

```json
{
  "version": 1,
  "game_id": "iter_0112_game_034",
  "iteration": 112,
  "game_idx": 34,
  "winner": null,
  "detected_player": "red",
  "starting_player": "red",
  "n_moves": 280,
  "reason": "state_cap",
  "outcome_class": 2,
  "scope": "both_sides",
  "ever_distance_le_2": true,
  "ever_distance_le_3": true,
  "min_total_goal_distance": 2,
  "detected": true,
  "first_dominant_unclosed_ply": 220,
  "first_total_goal_distance": 2,
  "first_category": "two_endpoint_closeout_2ply",
  "first_largest_component_size": 14,
  "first_endpoint_distances": {"top": 0, "bottom": 1},
  "actual_terminal_ply": 280,
  "actual_win_ply": null,
  "conversion_delay_plies": null,
  "conversion_delay_winner_moves": null,
  "cap_delay_proxy_plies": 60,
  "winner_moves_in_watch_window": null,
  "winner_moves_with_dominant_component": null,
  "winner_moves_with_dominant_unavailable": null,
  "primary_class_counts": null,
  "max_search_score_after_detection": null,
  "mean_search_score_after_detection": null,
  "high_value_after_detection_plies": null,
  "root_value_high_but_delayed": null,
  "search_score_coverage_in_watch_window": null
}
```

### 4.3 Class 3 — excluded (`outcome_class == 3`)

```json
{
  "version": 1,
  "game_id": "iter_0112_game_034",
  "iteration": 112,
  "game_idx": 34,
  "winner": null,
  "detected_player": null,
  "starting_player": "red",
  "n_moves": 0,
  "reason": "unknown",
  "outcome_class": 3,
  "scope": "excluded",
  "detected": false,
  "actual_terminal_ply": 0,
  "actual_win_ply": null,
  "conversion_delay_plies": null,
  "cap_delay_proxy_plies": null
}
```

### 4.4 Field reference

| Field | Type | Class 1 | Class 2 | Class 3 | Meaning |
|---|---|---|---|---|---|
| `version` | int | 1 | 1 | 1 | Schema version. |
| `game_id`, `iteration`, `game_idx` | str/int | populated | populated | populated | Identity. |
| `winner` | "red"/"black"/null | populated | null | null | Game outcome. |
| `detected_player` | "red"/"black"/null | = winner | focal side | null | Player whose detection drives this record. |
| `outcome_class` | 1/2/3 | 1 | 2 | 3 | Population discriminator. |
| `scope` | "winner"/"both_sides"/"excluded" | "winner" | "both_sides" | "excluded" | Scope of fields. |
| `ever_distance_le_2/le_3` | bool | bool | bool | false | Population-coverage flags. |
| `min_total_goal_distance` | int/null | int | int | null | Min seen for focal side. |
| `detected` | bool | true if ever ≤ detection_threshold | same | false | Detection state. |
| `first_dominant_unclosed_ply` | int/null | first pre-move ply detected | first pre-move ply detected | null | **Pre-move semantics.** |
| `first_total_goal_distance` | int/null | populated | populated | null | At first detection. |
| `first_category` | str/null | populated | populated | null | e.g., `two_endpoint_closeout_2ply`. |
| `first_largest_component_size` | int/null | populated | populated | null | Component size at first detection. |
| `first_endpoint_distances` | dict/null | populated | populated | null | `{"top": 0, "bottom": 1}` or `{"left", "right"}`. |
| `actual_terminal_ply` | int | n_moves | n_moves | 0 | Last ply. |
| `actual_win_ply` | int/null | terminal | null | null | Win ply (Class 1 only). |
| `conversion_delay_plies` | int/null | terminal − first_detected | null | null | Class 1 only. |
| `conversion_delay_winner_moves` | int/null | winner-only count | null | null | Class 1 only. |
| `cap_delay_proxy_plies` | int/null | null | terminal − first_detected | null | Class 2 only. |
| `winner_moves_in_watch_window` | int/null | populated | null | null | Class 1: focal-side post-detection move count (incl. detection ply). |
| `winner_moves_with_dominant_component`, `winner_moves_with_dominant_unavailable` | int/null | populated | null | null | Class 1 split of the watch window. |
| `primary_class_counts` | dict/null | populated | null | null | Class 1 only. Sum across watch window. |
| `*_after_detection` (search_score) | float/int/null | populated | null | null | Class 1 only. |
| `root_value_high_but_delayed` | bool/null | bool | null | null | `high_value_after_detection_plies ≥ 1 AND conversion_delay_plies ≥ high_value_delay_threshold_plies`. |

---

## 5. Per-iteration sidecar block

**Location**: `iter_NNNN_stats.json["goal_completion_summary"]`. Trainer writes once per iteration.

```json
{
  "version": 1,
  "config": {
    "detection_threshold": 2,
    "emit_threshold": 3,
    "high_value_threshold": 0.9,
    "high_value_delay_threshold_plies": 6,
    "max_depth": 3,
    "min_component_size": 8
  },
  "diagnostics_coverage": {
    "games_total": 100,
    "games_with_record": 100,
    "coverage_rate": 1.0,
    "games_class_1": 95,
    "games_class_2": 5,
    "games_class_3": 0
  },
  "main_population": {
    "n": 95,
    "games_with_dominant_unclosed": 95,
    "games_with_total_distance_le_2": 95,
    "games_with_total_distance_le_3": 95,
    "detected": 95,
    "detection_rate": 1.0,
    "min_total_goal_distance": {"p10": 0, "p50": 0, "p90": 1, "min": 0},
    "conversion_delay_plies": {"p50": 4, "p90": 10, "p95": 18, "max": 28, "mean": 6.3},
    "conversion_delay_winner_moves": {"p50": 2, "p90": 5, "max": 14, "mean": 3.1},
    "primary_class_rates": {
      "completes_endpoint": 0.415,
      "reduces_total_goal_distance": 0.222,
      "redundant_reinforcement": 0.280,
      "off_chain": 0.058,
      "other": 0.026,
      "dominant_unavailable": 0.010
    },
    "search_score_after_detection": {
      "max":  {"p50": 1.00, "p90": 1.00, "max": 1.00},
      "mean": {"p50": 0.98, "p90": 1.00, "max": 1.00}
    },
    "bad_cases": {
      "delay_ge_10": 15,
      "delay_ge_20": 4,
      "high_value_after_detection_plies_total": 247,
      "root_value_high_but_delayed": 14
    }
  },
  "capped_population": {
    "n": 5,
    "detected": 5,
    "cap_delay_proxy_plies": {"p50": 40, "p90": 60, "max": 60},
    "first_detector_side": {"red": 3, "black": 2}
  },
  "excluded_population": {"n": 0}
}
```

The summary is computed by the shared `aggregate_goal_completion_records()` function (Section 7). The same function is called by the analyzer for cross-iteration roll-up — the summary shape is identical at any scope.

---

## 6. Inline tracker module

**Location**: new module `scripts/GPU/alphazero/goal_completion_tracker.py`.

### 6.1 Public surface

```python
@dataclass
class _SideAccumulator:
    detected: bool = False
    first_dominant_unclosed_ply: Optional[int] = None
    first_total_goal_distance: Optional[int] = None
    first_category: Optional[str] = None
    first_largest_component_size: Optional[int] = None
    first_endpoint_distances: Optional[dict] = None
    primary_class_counts: dict = field(default_factory=_zero_class_counts)
    moves_after_detection: int = 0
    moves_with_dominant_component: int = 0
    moves_with_dominant_unavailable: int = 0
    search_scores_after_detection: list = field(default_factory=list)
    high_value_after_detection_plies: int = 0
    min_total_goal_distance: Optional[int] = None
    ever_distance_le_2: bool = False
    ever_distance_le_3: bool = False


@dataclass
class GoalCompletionGameTracker:
    enabled: bool = True
    detection_threshold: int = 2
    high_value_threshold: float = 0.9
    high_value_delay_threshold_plies: int = 6
    max_depth: int = 3
    min_component_size: int = 8
    red: _SideAccumulator = field(default_factory=_SideAccumulator)
    black: _SideAccumulator = field(default_factory=_SideAccumulator)

    def is_detected(self, side: str) -> bool: ...

    def observe_pre_move(
        self,
        *,
        state: TwixtState,
        ply: int,
        side_to_move: str,
        selected_move: Tuple[int, int],
        search_score: Optional[float],
        gc_state_cheap: Optional[dict],
        gc_state_full: Optional[dict],
    ) -> None: ...

    def finalize_game(
        self,
        *,
        winner: Optional[str],
        reason: str,
        n_moves: int,
        starting_player: str,
        iteration: int,
        game_idx: int,
        game_id: str,
    ) -> Optional[dict]: ...
```

### 6.2 `observe_pre_move` semantics

Pseudocode (reference):

```python
def observe_pre_move(self, *, state, ply, side_to_move, selected_move,
                     search_score, gc_state_cheap, gc_state_full):
    if not self.enabled:
        return
    acc = self.red if side_to_move == "red" else self.black

    # 1. Update side_to_move's coverage flags from the cheap state.
    if gc_state_cheap is not None:
        total = gc_state_cheap.get("total_goal_distance")
        if total is not None:
            if acc.min_total_goal_distance is None or total < acc.min_total_goal_distance:
                acc.min_total_goal_distance = total
            if total <= 2: acc.ever_distance_le_2 = True
            if total <= 3: acc.ever_distance_le_3 = True

    # 2. Snapshot pre-this-ply detection state. This determines whether the
    #    selected move counts as a "post-detection" move. With pre-move
    #    semantics, the detection ply itself is post-detection.
    was_detected_before = acc.detected

    # 3. Update detection: if not already detected and this side's pre-move
    #    state has a dominant-unclosed component within threshold, set
    #    detection now.
    if not acc.detected and gc_state_cheap is not None:
        total = gc_state_cheap.get("total_goal_distance")
        if total is not None and total <= self.detection_threshold:
            acc.detected = True
            acc.first_dominant_unclosed_ply = ply
            acc.first_total_goal_distance = total
            acc.first_category = gc_state_cheap.get("category")
            acc.first_endpoint_distances = gc_state_cheap.get("endpoint_distances")
            comp = gc_state_cheap.get("component_pegs")
            acc.first_largest_component_size = len(comp) if comp else None

    # 4. If detected (either before or just now), classify selected move.
    #    Pre-move semantics: the detection ply's move IS classified.
    if acc.detected:
        acc.moves_after_detection += 1

        if gc_state_cheap is None:
            acc.moves_with_dominant_unavailable += 1
        elif gc_state_full is None:
            # We needed full state to classify but caller didn't upgrade.
            # This should not happen if Section 3's need_full logic is correct;
            # treat defensively as dominant_unavailable.
            acc.moves_with_dominant_unavailable += 1
        else:
            acc.moves_with_dominant_component += 1
            cls = classify_selected_conversion_move(
                state, side_to_move, selected_move, gc_state_full,
                max_depth=self.max_depth,
                min_component_size=self.min_component_size,
            )
            primary = cls.get("primary_class", "other")
            if primary in acc.primary_class_counts:
                acc.primary_class_counts[primary] += 1
            else:
                acc.primary_class_counts["other"] += 1

        if search_score is not None:
            ss = float(search_score)
            acc.search_scores_after_detection.append(ss)
            if ss >= self.high_value_threshold:
                acc.high_value_after_detection_plies += 1
```

### 6.3 `finalize_game` semantics

```python
def finalize_game(self, *, winner, reason, n_moves, starting_player,
                  iteration, game_idx, game_id):
    if not self.enabled:
        return None

    outcome_class = _classify_outcome(winner, reason)  # 1, 2, or 3

    if outcome_class == 1:
        focal = self.red if winner == "red" else self.black
        return _build_class1_record(
            focal=focal, winner=winner, reason=reason, n_moves=n_moves,
            starting_player=starting_player, iteration=iteration,
            game_idx=game_idx, game_id=game_id,
            high_value_delay_threshold_plies=self.high_value_delay_threshold_plies,
        )

    if outcome_class == 2:
        focal_side, focal = _pick_class2_focal(self.red, self.black)
        return _build_class2_record(
            focal=focal, focal_side=focal_side, reason=reason,
            n_moves=n_moves, starting_player=starting_player,
            iteration=iteration, game_idx=game_idx, game_id=game_id,
        )

    # Class 3: minimal record.
    return _build_class3_record(
        reason=reason, n_moves=n_moves, starting_player=starting_player,
        iteration=iteration, game_idx=game_idx, game_id=game_id,
    )


def _classify_outcome(winner, reason):
    if winner in ("red", "black"):
        return 1
    if reason in ("state_cap", "timeout", "board_full"):
        return 2
    return 3


def _pick_class2_focal(red_acc, black_acc):
    """Tie-break: earliest first_dominant_unclosed_ply →
    lower first_total_goal_distance → red before black."""
    candidates = []
    if red_acc.detected: candidates.append(("red", red_acc))
    if black_acc.detected: candidates.append(("black", black_acc))
    if not candidates:
        # Neither detected — return red (or black) accumulator with detected=false.
        return "red", red_acc
    candidates.sort(key=lambda c: (
        c[1].first_dominant_unclosed_ply,
        c[1].first_total_goal_distance or 9999,
        0 if c[0] == "red" else 1,
    ))
    return candidates[0]
```

### 6.4 Class 1 finalize edge: detected side ≠ winner

If detection on the winner's side never occurred (winner.detected == False), the Class 1 record reports `detected: false` with all post-detection fields null. The fact that the *opponent* may have been detected is not surfaced in Class 1 (winner-perspective scope). This matches the Spec 1 analyzer's existing behavior.

### 6.5 Tracker disabled

`enabled=False`: `observe_pre_move` is a no-op, `finalize_game` returns `None`. Worker ships `goal_completion_record=None`, saver omits the JSON key, trainer aggregator skips. Zero BFS overhead.

---

## 7. Shared aggregator

**Location**: new module `scripts/GPU/alphazero/goal_completion_aggregator.py`.

### 7.1 Public surface

```python
def aggregate_goal_completion_records(
    records: list[dict | None],
    config: dict,
    games_total: int | None = None,
) -> dict:
    """Aggregate per-game records into a goal_completion_summary block.

    Pure function — no I/O, no BFS. Same shape at any scope (per-iter for
    trainer, cross-iter for analyzer).
    """
```

### 7.2 Reference implementation

```python
def aggregate_goal_completion_records(records, config, games_total=None):
    games_total = games_total if games_total is not None else len(records)
    valid = [_normalize_record(r) for r in records if r is not None]

    main = [r for r in valid if r["outcome_class"] == 1]
    capped = [r for r in valid if r["outcome_class"] == 2]
    excluded = [r for r in valid if r["outcome_class"] == 3]

    return {
        "version": 1,
        "config": dict(config),
        "diagnostics_coverage": {
            "games_total": games_total,
            "games_with_record": len(valid),
            "coverage_rate": (len(valid) / games_total) if games_total else 0.0,
            "games_class_1": len(main),
            "games_class_2": len(capped),
            "games_class_3": len(excluded),
        },
        "main_population": _summarize_main_population(main, config),
        "capped_population": _summarize_capped_population(capped),
        "excluded_population": {"n": len(excluded)},
    }


def _normalize_record(r: dict) -> dict:
    """Forward/backward-tolerant normalization at the function boundary."""
    return {
        "version": int(r.get("version", 1)),
        "outcome_class": int(r.get("outcome_class", 3)),
        "reason": r.get("reason") or "unknown",
        "winner": r.get("winner"),
        "detected_player": r.get("detected_player"),
        "detected": bool(r.get("detected", False)),
        "ever_distance_le_2": bool(r.get("ever_distance_le_2", False)),
        "ever_distance_le_3": bool(r.get("ever_distance_le_3", False)),
        "min_total_goal_distance": r.get("min_total_goal_distance"),
        "first_dominant_unclosed_ply": r.get("first_dominant_unclosed_ply"),
        "first_total_goal_distance": r.get("first_total_goal_distance"),
        "first_category": r.get("first_category"),
        "actual_terminal_ply": r.get("actual_terminal_ply"),
        "actual_win_ply": r.get("actual_win_ply"),
        "conversion_delay_plies": r.get("conversion_delay_plies"),
        "conversion_delay_winner_moves": r.get("conversion_delay_winner_moves"),
        "cap_delay_proxy_plies": r.get("cap_delay_proxy_plies"),
        "primary_class_counts": r.get("primary_class_counts") or _zero_class_counts(),
        "max_search_score_after_detection": r.get("max_search_score_after_detection"),
        "mean_search_score_after_detection": r.get("mean_search_score_after_detection"),
        "high_value_after_detection_plies": r.get("high_value_after_detection_plies"),
        "root_value_high_but_delayed": r.get("root_value_high_but_delayed"),
        "winner_moves_in_watch_window": r.get("winner_moves_in_watch_window"),
        "winner_moves_with_dominant_component": r.get("winner_moves_with_dominant_component"),
        "winner_moves_with_dominant_unavailable": r.get("winner_moves_with_dominant_unavailable"),
        "search_score_coverage_in_watch_window": r.get("search_score_coverage_in_watch_window"),
    }
```

### 7.3 Population helpers

`_summarize_main_population` and `_summarize_capped_population` migrate from `twixt_replay_analyzer.py:1191`/`:1377` into this module. Their signatures change to consume normalized dicts instead of dataclass instances. The output shape (`main_population{...}`, `capped_population{...}`) is preserved exactly so existing report formatters continue to work.

The shared module exports:

```python
__all__ = [
    "aggregate_goal_completion_records",
    "_summarize_main_population",
    "_summarize_capped_population",
    "_normalize_record",
    "_zero_class_counts",
]
```

---

## 8. Self-play integration

**File modified**: `scripts/GPU/alphazero/self_play.py`.

### 8.1 `play_game()` lifecycle

```python
def play_game(
    *,
    # ... existing args ...
    goal_completion_record_enabled: bool = True,
    goal_completion_emit_enabled: bool = True,
    goal_completion_detection_threshold: int = 2,
    goal_completion_emit_threshold: int = 3,             # renamed from broadcast_threshold
    high_value_threshold: float = 0.9,
    high_value_delay_threshold_plies: int = 6,
    goal_completion_max_depth: int = 3,
    goal_completion_min_component_size: int = 8,
):
    # ... setup ...

    # Validate invariant.
    if goal_completion_detection_threshold > goal_completion_emit_threshold:
        raise ValueError(
            "detection_threshold must be <= emit_threshold "
            f"(got {goal_completion_detection_threshold} > {goal_completion_emit_threshold})"
        )

    gc_tracker = GoalCompletionGameTracker(
        enabled=goal_completion_record_enabled,
        detection_threshold=goal_completion_detection_threshold,
        high_value_threshold=high_value_threshold,
        high_value_delay_threshold_plies=high_value_delay_threshold_plies,
        max_depth=goal_completion_max_depth,
        min_component_size=goal_completion_min_component_size,
    )

    # ... game loop:
    while not state.is_terminal():
        side = state.to_move

        # 1. MCTS search.
        root = mcts.search(state, ...)
        # 2. Capture per-move metrics (search_score, root_top1_share).
        search_score = _root_value_for_side(root, side)
        root_top1_share = _root_top1_share(root)
        # 3. Select move.
        selected_move = mcts.select_move(root, ...)

        # 4. Compute gc_state cheaply (single source of truth for the ply).
        gc_state_cheap = compute_goal_completion_state(
            state, side,
            max_depth=goal_completion_max_depth,
            min_component_size=goal_completion_min_component_size,
            enumerate_moves=False,
        ) if (goal_completion_record_enabled or goal_completion_emit_enabled) else None

        # 5. Decide whether to upgrade to full state.
        needs_phase3_full = (
            goal_completion_emit_enabled
            and gc_state_cheap is not None
            and gc_state_cheap.get("total_goal_distance") is not None
            and gc_state_cheap["total_goal_distance"] <= goal_completion_emit_threshold
        )
        needs_tracker_full = (
            gc_tracker.enabled
            and gc_state_cheap is not None
            and gc_state_cheap.get("total_goal_distance") is not None
            and (
                gc_tracker.is_detected(side)
                or gc_state_cheap["total_goal_distance"] <= gc_tracker.detection_threshold
            )
        )
        gc_state_full = None
        if needs_phase3_full or needs_tracker_full:
            gc_state_full = compute_goal_completion_state(
                state, side,
                max_depth=goal_completion_max_depth,
                min_component_size=goal_completion_min_component_size,
                enumerate_moves=True,
            )

        # 6. Phase 3 closeout_diagnostics consumes gc_state_full when emit on.
        if goal_completion_emit_enabled:
            _phase3_observe(...)  # existing closeout_diagnostics hooks

        # 7. Tracker observes — even when Phase 3 emit is off.
        gc_tracker.observe_pre_move(
            state=state,
            ply=current_ply,
            side_to_move=side,
            selected_move=selected_move,
            search_score=search_score,
            gc_state_cheap=gc_state_cheap,
            gc_state_full=gc_state_full,
        )

        # 8. Append history / per-move metadata.
        # 9. state = state.apply_move(selected_move).

    # Game end.
    record = GameRecord(...)
    record.goal_completion_record = gc_tracker.finalize_game(
        winner=winner, reason=reason, n_moves=len(record.moves),
        starting_player=record.starting_player,
        iteration=iteration, game_idx=game_idx, game_id=record.game_id,
    )
    return record
```

### 8.2 BFS reuse contract

| Phase 3 emit | record_enabled | per-ply cheap | per-ply full |
|---|---|---|---|
| on | on | always | when total ≤ emit_threshold OR (detected side OR total ≤ detection_threshold) |
| off | on | always | when (detected side OR total ≤ detection_threshold) |
| on | off | always | when total ≤ emit_threshold |
| off | off | never | never |

The cheap call is shared whenever either is on. Full calls overlap heavily on post-detection plies of decisive games.

---

## 9. Saver / IPC

### 9.1 Dataclass changes

```python
# GameRecord (mutable):
@dataclass
class GameRecord:
    # ... existing fields ...
    goal_completion_diagnostics: List[dict] = field(default_factory=list)
    goal_completion_diagnostics_meta: Optional[dict] = None
    goal_completion_record: Optional[dict] = None     # NEW

# GameComplete (frozen, IPC):
@dataclass(frozen=True)
class GameComplete:
    # ... existing fields ...
    goal_completion_diagnostics: tuple = ()
    goal_completion_diagnostics_meta: Optional[dict] = None
    goal_completion_record: Optional[dict] = None     # NEW
```

### 9.2 Conversion helpers

```python
# GameRecord -> GameComplete (worker-side)
GameComplete(
    # ... existing fields ...
    goal_completion_diagnostics=tuple(record.goal_completion_diagnostics or ()),
    goal_completion_diagnostics_meta=record.goal_completion_diagnostics_meta,
    goal_completion_record=record.goal_completion_record,
)

# GameComplete -> GameRecord (trainer-side)
GameRecord(
    # ... existing fields ...
    goal_completion_diagnostics=list(gc.goal_completion_diagnostics or ()),
    goal_completion_diagnostics_meta=gc.goal_completion_diagnostics_meta,
    goal_completion_record=gc.goal_completion_record,
)
```

### 9.3 Saver signature

```python
def save_game_replay(
    *,
    # ... existing args ...
    goal_completion_diagnostics: Optional[List[dict]] = None,
    goal_completion_diagnostics_meta: Optional[dict] = None,
    goal_completion_record: Optional[dict] = None,    # NEW kwarg
    # ...
) -> None:
    payload = { ... }
    if goal_completion_diagnostics:
        payload["goal_completion_diagnostics"] = goal_completion_diagnostics
    if goal_completion_diagnostics_meta is not None:
        payload["goal_completion_diagnostics_meta"] = goal_completion_diagnostics_meta
    if goal_completion_record is not None:
        payload["goal_completion_record"] = goal_completion_record
    # ... write JSON ...
```

The three keys are independent: any subset can be present.

### 9.4 IPC compatibility

Workers and trainer must restart together after this schema change (pickled `GameComplete` crossing the process boundary). No graceful in-place upgrade.

### 9.5 Versioning

`goal_completion_record["version"] = 1`. Bump on rename / removal / semantic change. Pure-additive new optional fields do not bump. Aggregator reads `version` defensively.

---

## 10. Trainer aggregation

### 10.1 Hook location

In the trainer's per-iteration sidecar writer, after all worker games complete:

```python
gc_records = [g.goal_completion_record for g in completed_games]
gc_summary = aggregate_goal_completion_records(
    gc_records,
    config=goal_completion_config_snapshot(),
    games_total=len(completed_games),
)
sidecar["goal_completion_summary"] = gc_summary
```

Pure dict math — no BFS, no replay walking. Same call-site context as `aggregate_per_game_stats()`.

### 10.2 Config snapshot

The trainer captures the resolved config (defaults + overrides) once per iteration and embeds it under `goal_completion_summary["config"]`. Analyzer echoes it into the report header so users see which thresholds produced the numbers.

If multiple iterations have differing configs, the analyzer warns and emits the per-iter configs in the report.

---

## 11. Analyzer changes

**File modified**: `scripts/twixt_replay_analyzer.py`.

### 11.1 Default path (no flags)

```python
# Per-game records are the canonical top-line source.
# Sidecars are read for validation and iteration telemetry.
sidecar_summaries = {
    it: sidecar.get("goal_completion_summary")
    for it, sidecar in relevant_sidecars.items()
    if sidecar.get("goal_completion_summary") is not None
}
per_game_records = [r.get("goal_completion_record") for r in replays]

gc_top_line = aggregate_goal_completion_records(
    per_game_records,
    config=_resolved_config_for_report(sidecar_summaries),
    games_total=len(replays),
)

# Validation: reconcile sidecar n vs records found per iteration.
_warn_on_sidecar_record_mismatch(sidecar_summaries, replays)
_warn_on_version_mismatch(per_game_records, sidecar_summaries)

# Worst-cases CSV from per-game records.
write_goal_completion_worst_cases_csv(replays, gc_top_line, ...)

# Report formatter unchanged — consumes the same summary shape.
lines = format_goal_completion_report(gc_top_line)
```

### 11.2 Worst-cases CSV

```python
def sort_delay_plies(rec):
    if rec is None:
        return -1
    if rec.get("outcome_class") == 1:
        return rec.get("conversion_delay_plies") or 0
    if rec.get("outcome_class") == 2:
        return rec.get("cap_delay_proxy_plies") or 0
    return -1

records_with_replays = [
    (r, replay)
    for r, replay in zip(per_game_records, replays)
    if r is not None
]
records_with_replays.sort(key=lambda pair: -sort_delay_plies(pair[0]))
top_n = records_with_replays[:max_worst_cases]
# Write CSV columns from r fields directly.
```

### 11.3 Missing-record behavior

```python
missing_count = sum(1 for r in per_game_records if r is None)
if missing_count == len(per_game_records):
    warn(
        f"{missing_count}/{len(per_game_records)} replays missing "
        f"goal_completion_record. Goal-completion report skipped. "
        f"Run with --goal-completion-recompute or rerun training with "
        f"goal_completion_record_enabled."
    )
elif missing_count > 0:
    examples = [r["game_id"] for r, replay in zip(per_game_records, replays)
                if r is None][:3]
    warn(
        f"{missing_count}/{len(per_game_records)} replays missing "
        f"goal_completion_record. Examples: {', '.join(examples)}."
    )
```

### 11.4 Sidecar/replay mismatch warning

```python
# Per iteration: compare sidecar n vs records found in replays.
for it, summary in sidecar_summaries.items():
    sidecar_n = summary["diagnostics_coverage"]["games_with_record"]
    replay_n = sum(
        1 for r, replay in zip(per_game_records, replays)
        if r is not None and replay.get("iteration") == it
    )
    if sidecar_n != replay_n:
        warn(
            f"Goal-completion sidecar/replay mismatch for iter {it:04d}: "
            f"sidecar games_with_record={sidecar_n}, replay records found={replay_n}. "
            f"Using per-game records as canonical analyzer source."
        )
```

### 11.5 `--goal-completion-recompute` legacy fallback

When the flag is set, the analyzer takes the moved-out legacy walker:

```python
# Legacy module: scripts/GPU/alphazero/goal_completion_recompute.py
# Houses _build_class1_per_game_record, _build_class2_per_game_record,
# _build_class3_per_game_record. Pre-move detection semantics applied.
from scripts.GPU.alphazero.goal_completion_recompute import (
    recompute_goal_completion_records_from_replays,
)
recomputed_records = recompute_goal_completion_records_from_replays(replays, config)

# When mixed with inline records (some games have them, some don't):
merged = [
    inline if inline is not None else recomputed
    for inline, recomputed in zip(per_game_records, recomputed_records)
]
gc_top_line = aggregate_goal_completion_records(merged, config, games_total=len(replays))
```

### 11.6 `--goal-completion-recompute-validate`

Implies `--goal-completion-recompute`. Runs full corpus, compares inline vs recomputed records per game on key fields:

```python
key_fields = [
    "outcome_class",
    "detected",
    "detected_player",
    "first_dominant_unclosed_ply",
    "first_total_goal_distance",
    "first_category",
    "conversion_delay_plies",
    "conversion_delay_winner_moves",
    "cap_delay_proxy_plies",
    "primary_class_counts",
    "root_value_high_but_delayed",
]
float_fields = [
    "max_search_score_after_detection",
    "mean_search_score_after_detection",
]
TOLERANCE = 1e-6
```

Emits per-game divergence report (or "all match" when truly aligned).

This flag is intentionally expensive. For sampled validation, run the analyzer on a smaller input set.

### 11.7 Code paths retired / moved

| Symbol | Before | After |
|---|---|---|
| `aggregate_goal_completion_diagnostics` (analyzer) | replay-walking aggregator | replaced by `aggregate_goal_completion_records` from shared module |
| `_build_class1_per_game_record` (analyzer) | analyzer-internal | moved to `goal_completion_recompute.py`; pre-move semantics |
| `_build_class2_per_game_record` (analyzer) | analyzer-internal | moved to `goal_completion_recompute.py`; pre-move semantics |
| `_summarize_main_population` (analyzer) | analyzer-internal | moved to `goal_completion_aggregator.py`; consumes normalized dicts |
| `_summarize_capped_population` (analyzer) | analyzer-internal | moved to `goal_completion_aggregator.py`; consumes normalized dicts |
| `format_goal_completion_report` | unchanged | unchanged (reads same summary shape) |
| `format_policy_mcts_closeout_report` | unchanged | unchanged |
| `write_goal_completion_worst_cases_csv` | walks replays + records | reads records only |

---

## 12. CLI surface and config defaults

### 12.1 Trainer / `play_game()` flags

| Flag | Default | Purpose |
|---|---|---|
| `--goal-completion-record-enabled` | `True` | Compact per-game record emission (cheap BFS per ply). |
| `--goal-completion-emit-enabled` | `True` | Phase 3 detailed per-ply closeout records (storage + ranking cost). |
| `--goal-completion-detection-threshold` | `2` | total_goal_distance ≤ this → detected. |
| `--goal-completion-emit-threshold` | `3` | total_goal_distance ≤ this → Phase 3 emit (and full-state computation). Renamed from `--goal-completion-broadcast-threshold`. |
| `--high-value-threshold` | `0.9` | search_score threshold for "high-value" classification. |
| `--high-value-delay-threshold-plies` | `6` | conversion delay threshold for `root_value_high_but_delayed`. |
| `--goal-completion-max-depth` | `3` | BFS depth for component_goal_distances. |
| `--goal-completion-min-component-size` | `8` | min component size for dominant-component classification. |

Both `record_enabled` and `emit_enabled` default `True`. Setting both `False` → zero goal-completion overhead.

**Invariant**: `detection_threshold ≤ emit_threshold` (validated at trainer startup; ValueError on violation).

**Rename**: `goal_completion_broadcast_threshold` → `goal_completion_emit_threshold` across `closeout_diagnostics.py`, `self_play.py`, `trainer.py`, CLI flag, and tests. Clean break — no backwards-compat alias.

### 12.2 Analyzer flags

| Flag | Default | Purpose |
|---|---|---|
| `--goal-completion-recompute` | off | Use legacy replay walker (pre-move semantics). Used for old corpora and back-fill. |
| `--goal-completion-recompute-validate` | off | With recompute on, also load inline records and report per-field divergence. Implies `--goal-completion-recompute`. Full-corpus, intentionally expensive. |

### 12.3 Threshold cheat-sheet

- **detection_threshold**: When the player to move first has a strict closeout-shaped position. Drives `detected`, `conversion_delay_plies`, bad cases, worst-cases ranking.
- **emit_threshold**: When detailed per-ply closeout diagnostics are emitted. Must be ≥ detection_threshold so strict closeout plies are eligible for detailed capture.
- **high_value_threshold**: Whether `search_score` is considered "the model knows this is winning."
- **high_value_delay_threshold_plies**: Threshold for marking a delayed closeout as a bad case (`root_value_high_but_delayed`).
- **max_depth**: How far the bridge-reachable goal-distance BFS looks.
- **min_component_size**: Minimum component size to count as dominant (filters out small local fragments).

---

## 13. Migration / backwards compat

### 13.1 Treatment of pre-existing corpora (110-119)

- The 3-hour smoke run already in flight (with `--no-connectivity`, current Spec 1 Phase 2 code) is the reference snapshot for "what 110-119 looked like under the old design."
- After Spec 1.5 ships, further analysis of 110-119 uses `--goal-completion-recompute` (pre-move semantics, same slow walker). Numbers shift by ~1 ply per game on `first_dominant_unclosed_ply` — not a regression.
- **No JSON write-back migration**: don't backfill `goal_completion_record` into the 1010 old per-game JSONs. Old replays stay old; new replays carry inline records.
- Spec 2 design uses 110-119 as **directional signal** (the policy-blind-to-closeout pattern is robust). Once Spec 1.5 lands, run 1–2 fresh iterations with inline records and use those for any quantitative tuning targets in Spec 2.

### 13.2 Mixed corpora during cut-over

Realistic scenario: `analyzer --input Replays/108-119` where 108-110 lack records and 111-119 have them.

- Aggregator sees `list[dict | None]` with mixed presence.
- Coverage report: `games_with_record: 783, games_total: 1010, coverage_rate: 0.78`.
- Top-line summary computed from records that exist.
- Worst-cases CSV draws from records that exist.
- One aggregated warning with examples.
- `--goal-completion-recompute` fills the gap; analyzer merges inline-where-present, recomputed-where-absent.

### 13.3 Migration table

| Corpus type | Recommended path |
|---|---|
| New runs with records | Default analyzer path (no flags). |
| Old runs without records | Use `--goal-completion-recompute` only if goal-completion metrics are needed. |
| Mixed old/new | Default uses records and reports coverage; `--goal-completion-recompute` fills missing. |
| Debugging tracker correctness | `--goal-completion-recompute --goal-completion-recompute-validate`. |
| Need fast report only | Skip recompute; accept coverage warning. |

### 13.4 Schema versioning policy

- Current: `goal_completion_record["version"] = 1` and `goal_completion_summary["version"] = 1`.
- Bump on field rename / removal / semantic change. Pure additive optional fields do not bump.
- Aggregator reads `version` defensively; on unknown versions, warn and proceed with best-effort field reads.
- **Cross-version mismatch rule**: if per-game records and sidecar summaries have different versions, analyzer treats per-game records as canonical, warns about the mismatch, and rolls up from records.

### 13.5 Rollback plan

If the inline tracker has a bug:

- Revert worker hook to no-op by setting `goal_completion_record_enabled=False` default.
- Replays continue to be written without the new field.
- Analyzer falls back to "missing record" warning + `--goal-completion-recompute` for analysis.
- No data loss; return to slow-but-known path.

Independent kill switches:
- `record_enabled=False`: disables compact tracking + cheap BFS.
- `emit_enabled=False`: only disables detailed closeout records (Phase 3); compact records still emit.

---

## 14. Test strategy

### 14.1 New test files

**`tests/test_goal_completion_tracker.py`** — tracker unit tests
- Detection semantics (pre-move): synthetic 8-move game; assert `first_dominant_unclosed_ply` lands at the first ply where the side has the closeout pre-move.
- **Anchor: `test_tracker_premove_detection_classifies_detection_ply_move`** (named, non-optional). Protects pre-move semantic shift from regression.
- Detection coverage flags: `ever_distance_le_2/le_3`, `min_total_goal_distance`.
- Watch-window classification: synthetic states for each `primary_class` outcome.
- Dual-side tracking: both sides reach dominant-unclosed independently.
- Class 1 finalize: winner = detected side.
- Class 1 finalize edge: detected side ≠ winner (rare).
- Class 2 finalize: capped game, focal-side tie-break (earliest ply → lower distance → red).
- Class 3 finalize: corrupt/excluded.
- Tracker disabled: `enabled=False` → observe no-op, finalize returns None.
- No-full-state path: tracker handles `gc_state_full=None` defensively (counts as `dominant_unavailable`).

**`tests/test_goal_completion_aggregator.py`** — aggregator unit tests
- Empty input.
- Mixed Nones (`coverage_rate < 1.0`).
- Single-iter Class 1 only.
- Mixed populations.
- Percentile correctness (handcrafted record set with known delays).
- Cross-iteration roll-up: 10 single-iter aggregations vs 1 cross-iter aggregation on the same data → all stats match.
- Schema versioning: records with `version: 2` (unknown) → warn, best-effort.
- Naming check: `games_with_dominant_unclosed`, `games_with_total_distance_le_2/le_3` keys present.

**`tests/test_goal_completion_save_load.py`** — IPC + saver round-trip
- `GameRecord` → `GameComplete` (tuple-normalized) → pickle → `GameRecord` reconstituted.
- Saver writes `goal_completion_record` top-level key when present; omits when None.
- Loader: `replay.get("goal_completion_record")` returns dict or None.

**`tests/test_self_play_goal_completion_integration.py`** — end-to-end self-play
- Deterministic small `play_game()` reaches dominant-unclosed; assert record fields.
- `goal_completion_record_enabled=False`: assert `record.goal_completion_record is None`.
- `emit_enabled=False, record_enabled=True`: record populated, `goal_completion_diagnostics` empty.

**`tests/test_analyzer_goal_completion_records.py`** — analyzer record consumption
- Fixture: 3 per-game JSONs with pre-computed records + 1 sidecar.
- Default mode → assert summary matches inline records (no recompute).
- Worst-cases CSV from records.
- Coverage 100%.
- Mixed corpus: aggregated warning, coverage < 100%.
- Sidecar/replay mismatch: warn, don't fail.
- Version mismatch (record v1, sidecar v2): warn; records canonical.
- **Anchor: `test_analyzer_default_path_does_not_recompute_goal_completion`** — monkeypatches `compute_goal_completion_state` and `_build_class*_per_game_record`; asserts none called when records present. Structural anti-regression guard.

**`tests/test_analyzer_goal_completion_recompute.py`** — recompute fallback
- Fixtures without records → `--goal-completion-recompute` produces summary using pre-move semantics.
- Validation flag: `--goal-completion-recompute-validate` produces per-field divergence report.
- Game 097 anchor (pre-move) recompute test.

**`tests/test_analyzer_per_ply_perf_regression.py`** — perf regression guard
- 50 small synthetic fixture JSONs with `goal_completion_record` pre-populated.
- Default analyzer path completes in < 5 seconds.
- Generous absolute bound; structural guard above is the real anti-regression.

### 14.2 Updated test files

- `test_connectivity_goal_completion.py`: keep all 21 tests (helper semantics unchanged). Game 097 anchor renamed to clarify it's the structural (post-move) anchor; the new pre-move tracker anchor lives in the tracker test file.
- `test_analyzer_goal_completion.py`: tests that exercise `_build_class1_per_game_record` move into `test_analyzer_goal_completion_recompute.py` (those code paths are now flag-gated). Tests for shared aggregator helpers move into `test_goal_completion_aggregator.py`.

### 14.3 Test data fixtures

- Reuse existing fixture games where possible.
- New synthetic 8–12 ply games for tracker-specific edge cases.
- One real `iter_0110_game_*.json` for the perf regression test.

### 14.4 Total scope

~50 new tests, plus updates/migrations of existing tests. Compare to Spec 1's ~81 tests across 19 commits.

---

## 15. Implementation order

Single phase, ordered for risk minimization (TDD throughout):

1. **Tracker module** (`goal_completion_tracker.py`) — pure data, easiest to test. Lock pre-move semantics with anchor test.
2. **Shared aggregator** (`goal_completion_aggregator.py`) — pure functions. Move `_summarize_*` helpers from analyzer; switch to dict-typed input.
3. **IPC + saver changes** — add `goal_completion_record` field on dataclasses; saver kwarg; round-trip test.
4. **Self-play integration** (`self_play.py`) — wire tracker into `play_game()` with the BFS reuse contract. Rename `broadcast_threshold` → `emit_threshold` here too.
5. **Trainer per-iteration aggregation hook** — call shared aggregator at sidecar write time.
6. **Analyzer record-consumption path** — rewrite default path; keep replay loader unchanged; test reads pre-computed records.
7. **Recompute fallback module** (`goal_completion_recompute.py`) — move legacy walkers; pre-move semantics; CLI flag wiring.
8. **Validation flag** — `--goal-completion-recompute-validate` per-field comparison.
9. **Migration documentation + threshold cheat-sheet** — added to spec/README.
10. **End-to-end smoke test** — run analyzer on a fresh small training run; confirm fast path produces records, sidecar summary aggregates, worst-cases CSV populates.

Each step is one commit with TDD. The order keeps invariants stable: aggregator works before integration, IPC works before integration, tracker works before integration. By step 5 the analyzer can consume records; legacy fallback is added in steps 7–8.

---

## 16. Risks

- **Worker/trainer schema skew during deploy**: handled by coordinated restart (Section 9.4).
- **Pre-move semantic shift confuses users**: mitigated by spec preamble, threshold cheat-sheet, and migration table; recompute path uses same semantics so internal comparisons stay consistent.
- **Hidden BFS cost in worker**: cheap BFS per ply with `record_enabled=True` is the new baseline. If it shows up as a self-play perf regression, the kill switch (`record_enabled=False`) restores prior cost; the slow analyzer recompute path remains the user's escape hatch.
- **Aggregator divergence between trainer and analyzer**: avoided by sharing `aggregate_goal_completion_records()` across both. Sidecar/replay mismatch warning catches drift loudly.
- **Schema evolution**: version field + defensive reads + cross-version mismatch rule (Section 13.4) keep readers robust.

---

## 17. Out of scope

- **Phase 5/6 (probe tier + training knobs)**: deferred to Spec 2.
- **Phase 3 detailed-record aggregation moving inline**: not in this spec. The analyzer continues to aggregate Phase 3's per-ply `goal_completion_diagnostics` at read-time; that's already cheap dict iteration (no BFS), so no perf concern.
- **Migration tooling for old per-game JSONs**: not in this spec. Old replays remain old.
- **Removing the recompute path**: kept indefinitely as a debug/backfill tool.
