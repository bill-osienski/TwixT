# Goal-Completion / Conversion Diagnostics — Design

**Date:** 2026-05-03
**Author:** brainstormed with Bill
**Status:** Design approved, ready for implementation planning
**Touches:** `scripts/GPU/alphazero/{self_play,ipc_messages,self_play_worker,game_saver,trainer,connectivity_diagnostics}.py`, possibly a new `scripts/GPU/alphazero/closeout_diagnostics.py`, `scripts/twixt_replay_analyzer.py`, and five new test files.
**Predecessors:**
- `docs/superpowers/specs/2026-04-29-per-game-stats-persistence-design.md` — per-game persistence pattern reused (per-move plumbing mirrors per-game plumbing).
- `docs/superpowers/specs/2026-04-29-analyzer-per-game-stats-design.md` — analyzer-surfacing pattern reused (coverage maps, partial-coverage rendering).
- `docs/superpowers/specs/2026-04-28-strong-advantage-probe-tier-design.md` — Phase 4 of this spec wires the trainer + analyzer paths that predecessor designed.

**Successor:** Spec 2 (Phases 5–6: goal-completion probe tier + training correction knobs) — to be brainstormed after at least one 10-iter run with Phase 0–4 diagnostics on disk.

---

## 1. Problem

Self-play games like `iter_0108_game_097` show a specific failure mode: the value head correctly evaluates the position as winning (`final_root_value=0.9987`), but the policy fails to commit to closing moves and the winner drifts for many redundant plies before the game terminates. In game 097, Red had a continuous knight-bridge chain at turn 35 with a 2-ply forced win available (top endpoint distance 1, bottom endpoint distance 1, no opposing interference at either endpoint) and still played 22 more plies — 10 redundant pegs — before finally closing on turns 57 and 59.

The current per-iter telemetry can't quantify this:
- No per-move root value on disk for plies past the opening-diagnostics window. The saved-game `moves[].search_score` field exists in the schema but is unpopulated (`game_saver.py:71` literally writes `None`).
- No replay-side analysis of structural closeout availability vs realized closeout. `connectivity_diagnostics.py` computes largest components and goal-touching components but does not measure bridge-reachable distance to the goal lines.
- No surfacing of the data in `summary.json` / `report.txt` (the operator's primary review artifact for 10-iter runs).
- The strong-advantage probe tier is fully designed and the probe set was promoted on 2026-04-28, but its telemetry never reaches the trainer sidecar — `trainer.py:2959` hard-wires `strong_advantage_summary=None` when calling `build_probe_summary_block`, so even the existing measurement is invisible across iterations.

## 2. Goals

Add an analyzer-and-reporting-first diagnostics pipeline that quantifies, for every 10-iter run:

1. How often the eventual winner reaches a dominant unclosed structure.
2. At what ply the structure first appears.
3. How many plies pass between detection and the actual game termination.
4. What fraction of the winner's post-detection moves complete endpoints / reduce distance / are redundant / are off-chain.
5. How often the model "knew it was won" (high `search_score`) but drifted anyway.
6. Whether the failure attributes to value, policy, or MCTS selection — by capturing inline closeout root diagnostics with policy/MCTS top-K detail on closeout-shaped plies.
7. The strong-advantage probe tier's per-iter trend, surfaced through the same `summary.json` / `report.txt` operator review path.

## 3. Non-goals

- No new probe-set creation or relabeling in this spec. This spec only surfaces the existing strong-advantage tier according to the predecessor spec; any probe promotion / input-file change is treated as prerequisite work or handled outside this spec.
- No goal-completion probe tier (Spec 2).
- No training-loop changes — conversion curriculum, conversion policy loss weight, dominant-unclosed reweighting, value-weight tuning (Spec 2).
- No backfill of old replay JSONs to the new schema. Old replays remain readable; analyzer treats absent fields as not-covered (same convention as the per-game-stats predecessor).
- No Replay.html or other UI consumer changes.
- No new heatmap / matplotlib plots or per-iter trend charts.
- No new CSV emitters beyond `goal_completion_worst_cases.csv` and the predecessor's `strong_advantage_probe_by_iter.csv`.
- No widening of replay-buffer / training-position schema (`PositionRecord` unchanged).

## 4. Architecture overview

The change runs through five planes, in dependency order. Each plane is touched once or twice with well-bounded edits.

### 4.1 Five planes

1. **Per-move data plane** *(Phase 0).* Self-play collects two parallel lists alongside `move_history` (`move_root_values`, `move_top1_shares`). They flow through `GameRecord` (in-process path) and `ipc_messages.GameComplete` (worker path) into the routing helpers `_save_game_from_record` / `_save_game_from_ipc`, then into a widened `save_game_replay(...)` signature that populates `moves[i]["search_score"]` and `moves[i]["root_top1_share"]`.

2. **Connectivity helper plane** *(Phase 1).* `scripts/GPU/alphazero/connectivity_diagnostics.py` gains three pure functions — `component_goal_distances`, `compute_goal_completion_state`, `classify_selected_conversion_move` — using `TwixtState`'s engine semantics (`apply_move()`, `_find_new_bridges`, `_proper_intersect_knight`) for crossing-aware BFS. Pure replay-side helpers; no self-play coupling.

3. **Replay-aggregation plane** *(Phases 0/2/4 surfacing).* `scripts/twixt_replay_analyzer.py` gains `aggregate_per_move_stats`, `aggregate_goal_completion_diagnostics`, and tier-keyed `probe_summary` reading; matching `format_*_report` helpers; and the `goal_completion_worst_cases.csv` writer.

4. **Strong-advantage telemetry plane** *(Phase 4).* Trainer extracts `tiers.get("strong_advantage")` from the inline probe-eval call and feeds it into `build_probe_summary_block(...)` (the second argument, currently hard-wired to `None`). Sidecar gains a legacy `strong_advantage_probe_summary` block alongside the existing `forced_probe_summary` for one release cycle. New flat CSV fields `sas_*` mirror the existing `fps_*`. Analyzer reads tier-keyed sidecar fields with legacy fallback and produces parallel summary block + report section + by-iter CSV.

5. **Inline closeout-diagnostics plane** *(Phase 3).* `play_game()` adds a per-ply call to `compute_goal_completion_state(state, state.to_move, ...)` after MCTS search, gated on `total_goal_distance ≤ emit_threshold` AND `largest_component_size ≥ emit_min_component`. When the gate fires, a new `build_closeout_diagnostic_partial / finalize` pair (split around move selection) produces a record with goal-completion sub-blocks, selected-move classification, and per-completion-move policy/visit ranking. Records collect into `goal_completion_diagnostics: []` on the saved game JSON, with a meta echo. Defensive try/except around the entire capture path.

### 4.2 Two-threshold split

Replay-side conversion-delay metrics default to **strict `total_goal_distance ≤ 2`**. Inline root diagnostics default to **broader capture at `total_goal_distance ≤ 3`** so policy/MCTS evidence is available for both strict closeouts and near-closeouts. Both thresholds are flag-configurable; both `le_2` and `le_3` buckets are reported regardless.

### 4.3 Safety invariant

All inline diagnostic capture is best-effort and must never affect move selection, training targets, or game termination. Failures increment counters in `goal_completion_diagnostics_meta` and emit stderr warnings; they are never raised into the training path.

### 4.4 Data-flow summary

| Artifact | Phase 0 adds | Phase 2 adds | Phase 3 adds | Phase 4 adds |
|---|---|---|---|---|
| `iter_NNNN_game_NNN.json` (saved game) | `moves[i].search_score`, `moves[i].root_top1_share` | — | `goal_completion_diagnostics`, `goal_completion_diagnostics_meta` | — |
| `iter_NNNN_stats.json` (per-iter sidecar) | — | — | — | `strong_advantage_probe_summary`; populated `probe_summary.strong_advantage` |
| `summary.json` | `per_move_stats` block | `goal_completion` block | `goal_completion.diagnostics_coverage`, `goal_completion.policy_mcts_summary` sub-blocks | `strong_advantage_probe.{by_iter, latest}` block |
| `report.txt` | Per-move stats section | Goal-Completion / Conversion Diagnostics section | Policy/MCTS closeout behavior section | Strong-Advantage Probe Sign-Agree section |
| New CSVs | — | `goal_completion_worst_cases.csv` | — | `strong_advantage_probe_by_iter.csv` |

---

## 5. Phase 0 — Per-move search_score + root_top1_share

### 5.1 Schema (additive to existing `moves[]` element)

```jsonc
{
  "turn": 37,
  "player": "red",
  "row": 15,
  "col": 11,
  "bridges_created": [],
  "heuristics": {},
  "search_score": 0.9821,         // root q_value from side_to_move perspective, captured after MCTS search and before move selection. Range typically [-1, 1] under MCTS invariants. null for old replays / non-MCTS games.
  "root_top1_share": 0.0875       // max(child visits) / sum(child visits) at root. (0, 1] when populated; null when total_visits == 0 or unavailable.
}
```

`search_score` is unconditionally side-to-move-perspective. Phase 2 handles winner-perspective conversion when interpreting "high winner confidence" vs "loser confidence of losing." Old on-disk replays are missing both keys; the analyzer treats absence as not-covered.

### 5.2 Self-play capture (`scripts/GPU/alphazero/self_play.py`)

After `mcts.search_from_root(...)` returns and at the **same point where `move_history.append(selected_move)` happens** (so the resign-by-loser branch, which breaks the loop without playing a move, doesn't add a phantom entry), append to two parallel `list[Optional[float]]` accumulators:

```python
move_root_values.append(float(root_value) if root_value is not None else None)
total = sum(visit_counts.values()) if visit_counts else 0
top1  = max(visit_counts.values()) if visit_counts else 0
move_top1_shares.append(float(top1 / total) if total > 0 else None)
```

Lists end length-equal to `move_history` on every code path.

### 5.3 Plumbing

| Plane | Change |
|---|---|
| `self_play.GameRecord` | `move_root_values: List[Optional[float]] = field(default_factory=list)`, `move_top1_shares: List[Optional[float]] = field(default_factory=list)`. Populated at `play_game()` return. |
| `ipc_messages.GameComplete` | `move_root_values: Optional[List[Optional[float]]] = None`, `move_top1_shares: Optional[List[Optional[float]]] = None`. Pickle-safe; defaults preserve existing IPC contract. |
| `self_play_worker.py` | Worker-side accumulators populated identically; passed through `GameComplete` construction. |
| `game_saver.save_game_replay(...)` | Two new kwargs. Inside the existing per-move construction loop, index-into the parallel list defensively (excess `move_history` entries → `null`, excess parallel entries → silently ignored). On length mismatch, write a stderr warning and increment a per-game counter (do not raise). |
| `trainer.py` routing helpers | `_save_game_from_record` / `_save_game_from_ipc` (already extracted by predecessor spec) each gain a two-line forwarding addition. |

### 5.4 Analyzer surfacing — `aggregate_per_move_stats(replays)`

New top-level helper near `aggregate_per_game_stats`. Returns:

```jsonc
{
  "n_games_total": 1000,
  "n_moves_total": 56425,
  "coverage": {
    "search_score":    56000,    // move-count denominator, not game-count
    "root_top1_share": 56000
  },
  "search_score": {                                     // null when coverage.search_score == 0
    "mean":      0.18,
    "p50":       0.21,
    "p90":       0.93,
    "p95":       0.97,
    "min":      -1.00,
    "max":       1.00,
    "mean_abs":  0.51                                   // mean of absolute values; endgame-decisiveness proxy
  },
  "root_top1_share": {                                  // null when coverage.root_top1_share == 0
    "mean": 0.32, "p50": 0.28, "p90": 0.61, "p95": 0.78, "min": 0.04, "max": 1.00
  }
}
```

Single linear pass over all `replay["moves"]` arrays. Old replays contribute zero coverage; their moves are excluded from distributions, not zero-weighted.

### 5.5 Report addition (placed immediately before the existing Per-game stats section)

```
Per-move stats (n=56,000 / 56,425 moves carry new fields):
  search_score:    mean=0.18 p50=0.21 p90=0.93 p95=0.97 (range [-1.00, 1.00], mean_abs=0.51)
  root_top1_share: mean=0.32 p50=0.28 p90=0.61 p95=0.78 min=0.04
  Coverage:        search_score=56000/56425 root_top1_share=56000/56425
```

Empty/partial/uniform-coverage rules:
- Coverage line suppressed only when **both** fields have **full** coverage over all moves.
- Field with zero coverage → its line omitted (not "n/a").
- All zero (`n_moves_with_any_stats == 0`) → short fallback: `Per-move stats: no moves carry new fields (all replays predate persistence change).`

### 5.6 Tests (Phase 0) — 11 tests

1. `test_save_game_replay_writes_per_move_fields_when_lists_populated`
2. `test_save_game_replay_per_move_fields_null_when_lists_absent`
3. `test_save_game_replay_per_move_fields_handle_short_parallel_list`
4. `test_save_game_replay_per_move_fields_ignores_long_parallel_list`
5. `test_aggregate_per_move_stats_zero_coverage_for_old_replays`
6. `test_aggregate_per_move_stats_full_coverage_distributions_correct`
7. `test_aggregate_per_move_stats_partial_coverage_excludes_missing_not_zero`
8. `test_format_per_move_stats_report_uniform_coverage_suppresses_coverage_line`
9. `test_format_per_move_stats_report_zero_coverage_short_message`
10. `test_in_process_play_game_returns_per_move_lists_aligned_with_history`
11. `test_resign_path_does_not_append_phantom_per_move_entries`

Plus a pickle-roundtrip assertion that `GameComplete` preserves the two new optional list fields.

### 5.7 Implementation sequencing (Phase 0) — 5 commits

1. `feat(saver): per-move search_score and root_top1_share kwargs` — `game_saver.py` only. Tests #1–#4.
2. `feat(self-play): collect per-ply root value and top1 share, append to parallel lists` — `self_play.py` only, in-process path. Tests #10, #11.
3. `feat(ipc): pass per-move lists through GameComplete` — `ipc_messages.py` + `self_play_worker.py`. Pickle roundtrip test.
4. `feat(trainer): route per-move lists through save helpers` — two-line additions to `_save_game_from_record` and `_save_game_from_ipc`.
5. `feat(analyzer): aggregate_per_move_stats and report rendering` — `twixt_replay_analyzer.py`. Tests #5–#9.

---

## 6. Phase 1 — Bridge-reachable endpoint distance helpers

All additions to `scripts/GPU/alphazero/connectivity_diagnostics.py`. Pure functions; no caller wiring (Phase 2 wires them).

### 6.1 Function signatures

```python
def component_goal_distances(
    state: TwixtState,
    player: str,
    component: frozenset[tuple[int, int]],
    max_depth: int = 3,
) -> dict:
    """
    Shortest bridge-extension distance from `component` to each of player's
    two goal sides. For red: {"top", "bottom"}. For black: {"left", "right"}.
    Each value: int in [0, max_depth] or None.

    Distance N = minimum fresh legal peg placements such that after those
    placements the resulting connected component (using TwixtState bridge
    rules including crossing checks) contains a peg on the goal side.
    Existing same-color pegs are absorbable at cost 0 only when (a) already
    connected by `state.bridges`, or (b) become connected via the new
    bridges created by a fresh hypothetical placement per `apply_move()`
    semantics.
    """

def compute_goal_completion_state(
    state: TwixtState,
    player: str,
    max_depth: int = 3,
    min_component_size: int = 8,
) -> Optional[dict]:
    """
    Best dominant-unclosed component for `player`, or None if no component
    meets `min_component_size`. Selection rule: smallest total_goal_distance;
    tie-break by largest component size; tie-break by deterministic peg
    ordering (min-corner of the component peg set).

    Returns:
      component_pegs:             frozenset[(r, c)]
      largest_component_size:     int
      endpoint_distances:         dict[str, int|None]  # {"top","bottom"} or {"left","right"}
      total_goal_distance:        int | None           # None if either endpoint is None
      touches_goal_a, touches_goal_b: bool
      endpoint_completion_moves:  list[(r, c)]         # placements that drop a non-zero endpoint to 0
      distance_reducing_moves:    list[(r, c)]         # placements that strictly reduce total
      category:                   str
      max_depth:                  int                  # echo
    """

def classify_selected_conversion_move(
    state_before: TwixtState,
    player: str,
    selected_move: tuple[int, int],
    goal_state_before: dict,
    max_depth: int = 3,
    min_component_size: int = 8,
) -> dict:
    """
    Classify a selected move against the pre-move dominant-unclosed state.
    Returns:
      completes_endpoint:           bool   # raw, can co-occur with reduces_total
      reduces_total_goal_distance:  bool
      is_redundant_reinforcement:   bool   # bridgeable to dominant component but no distance reduction
      is_off_chain:                 bool   # not knight-adjacent to extended component, no reduction
      primary_class: "completes_endpoint" |
                     "reduces_total_goal_distance" |
                     "redundant_reinforcement" |
                     "off_chain" |
                     "other"                          # for report rate-summing (always 100%)
      total_goal_distance_before:   int | None
      total_goal_distance_after:    int | None
    """
```

### 6.2 Category enum (six entries)

| Category | Definition |
|---|---|
| `"already_won"` | `total_goal_distance == 0` (terminal — caller skips) |
| `"one_move_win"` | `total_goal_distance == 1` |
| `"two_endpoint_closeout_2ply"` | endpoint distances == 1 and 1 (Game 097 case) |
| `"one_endpoint_distance_2"` | one endpoint touching (0), other distance == 2 |
| `"broader_conversion"` | `total_goal_distance ≤ max_depth`, none of the above |
| `"not_reachable"` | `total_goal_distance is None` (internal — `compute_goal_completion_state` returns `None` instead) |

### 6.3 Algorithm — bridge-legality-aware BFS

For each candidate component meeting `min_component_size`:

1. **Frontier seed**: all pegs in the component at cost 0; absorb same-color out-of-component pegs that are *already* in the same `state.bridges`-connected component (degenerate: this is just the existing component).
2. **Per-endpoint BFS**: target = "any cell on goal_side that is part of the (extended) component." Layer L corresponds to "L fresh placements made." Expand a frontier node by enumerating cells `(r, c)` where:
   - cell is empty,
   - `state.is_valid_placement(r, c)` for `player` (corner/edge color rules),
   - `dataclasses.replace(state, to_move=player).apply_move((r, c))` succeeds and results in a bridge that connects `(r, c)` to the frontier node (verified against the engine's actual `_find_new_bridges` output, not pure knight geometry).
3. Add `(r, c)` to the next layer at cost +1; transitively absorb same-color out-of-component pegs that become connected through the newly created bridges (cost 0).
4. **Termination**: stop when a frontier cell **is** on goal_side. Distance = layer count of the terminating cell.
5. **No path within `max_depth`**: return `None` for that endpoint.

`endpoint_completion_moves` is the set of fresh placements `(r, c)` at layer L=1 from which a goal-side cell is reachable in 0 additional placements.

`distance_reducing_moves` is the set of fresh placements where, after hypothetical apply, `total_goal_distance` strictly decreases. Computed by post-hoc evaluation: candidate set is "any legal placement within knight distance of an in-component peg." Worst case ~30–50 candidates per ply. Sub-millisecond on typical positions.

### 6.4 Engine-faithful hypothetical placements

```python
hypo_state    = dataclasses.replace(state, to_move=player).apply_move((r, c))
new_bridges   = hypo_state.bridges - state.bridges
new_component = hypo_state._get_connected_component((r, c), player)
```

Three properties hold by engine guarantee: placement legality, bridge creation rules, and crossing-check semantics. Helper-level bridge-legality calls (`_proper_intersect_knight`) are allowed *only* as an optimization to short-circuit obviously-blocked candidates without copying state — they may never alter rules. Final accept/reject decisions go through `apply_move()` semantics.

### 6.5 Edge cases

| Case | Behavior |
|---|---|
| Empty board / no pegs of `player` | `compute_goal_completion_state` → `None` |
| All components below `min_component_size` | `compute_goal_completion_state` → `None` |
| Component already touching both goals (terminal) | Returns state with `total_goal_distance == 0`, `category == "already_won"`. Callers skip. |
| Multiple equal-distance components | Tie-break by size desc, then by deterministic peg ordering |
| Hypothetical placement that produces a winning component | `endpoint_completion_moves` includes the winning move; `distance_reducing_moves` is a strict superset |
| Bridge-legality check on hypothetical placement | `_proper_intersect_knight` semantics; bridge only forms if no intersection with existing bridges |
| `min_component_size = 8` default | Configurable via `--goal-completion-emit-min-component` (Phase 3) |

### 6.6 Tests (Phase 1) — 19 tests

1. `test_component_goal_distances_distance_zero_already_touching`
2. `test_component_goal_distances_distance_one_via_fresh_placement_on_goal_line`
3. `test_component_goal_distances_distance_one_via_isolated_existing_goal_line_peg_with_bridgeable_connector`
4. `test_component_goal_distances_distance_two_two_hop_chain`
5. `test_component_goal_distances_blocked_by_intersecting_bridge_takes_alternative_or_none`
6. `test_component_goal_distances_unreachable_within_max_depth_returns_none`
7. `test_component_goal_distances_skips_invalid_placements`
8. `test_compute_goal_completion_state_picks_smallest_distance_then_largest_size`
9. `test_compute_goal_completion_state_returns_none_below_min_component_size`
10. `test_compute_goal_completion_state_endpoint_completion_moves_exact_set`
11. `test_compute_goal_completion_state_distance_reducing_is_superset_of_endpoint_completion`
12. `test_compute_goal_completion_state_categories_partition_correctly`
13. `test_classify_selected_completes_and_reduces_both_true_primary_class_is_completes`
14. `test_classify_selected_reduces_distance_only_primary_class_is_reduces`
15. `test_classify_selected_redundant_reinforcement_bridgeable_to_component_no_distance_reduction`
16. `test_classify_selected_off_chain_when_no_knight_neighbor_in_extended_component`
17. **`test_compute_goal_completion_state_game097_turn35_canonical`** — replay first 35 moves of `iter_0108_game_097`; assert `total_goal_distance == 2`, `endpoint_distances == {"top": 1, "bottom": 1}`, `category == "two_endpoint_closeout_2ply"`, `(0, 8) in endpoint_completion_moves` and `(23, 6) in endpoint_completion_moves`. **Spec anchor.**
18. `test_existing_same_color_goal_peg_requires_actual_or_new_bridge_connection` — guards against ambient bridge absorption.
19. `test_classify_selected_primary_class_other_for_adjacent_nonreducing_nonredundant_move` — guards against forcing ambiguous moves.

### 6.7 Implementation sequencing (Phase 1) — 2 commits

1. `feat(connectivity): component_goal_distances with apply_move-faithful BFS` — adds `component_goal_distances` and a small private `_apply_hypothetical(state, player, move)` wrapper. Tests #1–#7.
2. `feat(connectivity): compute_goal_completion_state + classify_selected_conversion_move with primary_class` — adds the snapshot helper and the classifier. Tests #8–#19.

---

## 7. Phase 2 — Replay-side aggregation, summary, report, worst-cases CSV

All additions to `scripts/twixt_replay_analyzer.py`. Pure post-hoc analysis using the Phase 1 helpers. No self-play coupling.

### 7.1 Outcome-class taxonomy

| Class | `meta.reason` values | Scope | Counted in main metrics? |
|---|---|---|---|
| 1 (decisive) | `win`, `resign`, `adjudicated` | eventual winner only | yes |
| 2 (capped) | `state_cap`, `timeout`, `timeout_selfplay`, `board_full` | both sides (no winner) | bad-cases only |
| 3 (excluded) | `draw`, `unknown`, anything else | n/a | no |

Class 1 resign games: the loser's resign happens after their own MCTS produced `root_value ≤ -0.97`; this is a clean decisive resolution. `actual_terminal_ply = n_moves`; do not estimate hypothetical remaining drift.

### 7.2 Per-game record (intermediate)

```jsonc
{
  "game_id": "iter_0108_game_097",
  "iteration": 108, "game_idx": 97,
  "winner": "red",                       // null on Class 2
  "starting_player": "red",
  "n_moves": 59,
  "reason": "win",
  "outcome_class": 1,                    // 1 | 2 | 3
  "scope": "winner",                     // 1 → winner; 2 → both_sides
  "detected_player": "red",              // for Class 2: first-detected side; for Class 1: winner

  "ever_distance_le_2": true,            // requires min_component_size at the same ply
  "ever_distance_le_3": true,
  "min_total_goal_distance":  2,

  "detected": true,
  "first_dominant_unclosed_ply": 35,
  "first_total_goal_distance":  2,
  "first_category": "two_endpoint_closeout_2ply",
  "actual_terminal_ply": 59,             // = n_moves on Class 1 and Class 2
  "actual_win_ply":     59,              // CSV-compat alias; null when winner is null

  // Class 1 only:
  "conversion_delay_plies":         24,  // strictly post-detection: actual_terminal_ply - first_dominant_unclosed_ply
  "conversion_delay_winner_moves":  12,  // count of winner moves strictly after detection through terminal

  // Class 2 only:
  "cap_delay_after_detection_plies": null,

  // Class 1 only (per-winner-move classification within watch window):
  "winner_moves_in_watch_window":          12,
  "winner_moves_with_dominant_component":  12,
  "winner_moves_with_dominant_unavailable": 0,
  "primary_class_counts": {                                // sum to winner_moves_with_dominant_component
    "completes_endpoint":          2,
    "reduces_total_goal_distance": 0,
    "redundant_reinforcement":     8,
    "off_chain":                   2,
    "other":                       0
  },

  // Per-move-stats sub-block (null when search_score coverage is 0 in the watch window):
  "max_search_score_after_detection":  0.9987,
  "mean_search_score_after_detection": 0.97,
  "high_value_after_detection_plies":  12,                 // post-detection winner plies with search_score >= --goal-completion-high-value-threshold
  "root_value_high_but_delayed":       true,               // Class 1 + detected + ≥1 high-value post-detection winner ply + conversion_delay_plies >= 10
  "search_score_coverage_in_watch_window": 12
}
```

**Watch window**: detection is post-move; classification opens on the in-scope side's *subsequent* moves. For Game 097 detection at turn 35 → watch window opens at turn 37. Per-winner-move classification re-runs `compute_goal_completion_state(state_before, winner)`. If it returns `None`, the move counts toward `winner_moves_with_dominant_unavailable` and **not** toward `primary_class_counts`. Rate sums use `winner_moves_with_dominant_component` as denominator.

**Detection-on-terminal**: `conversion_delay_plies = 0`, `conversion_delay_winner_moves = 0` (strictly-after-detection semantics).

**Class 2 detected_player tie-break**: lower `total_goal_distance` first, then larger component size, then `"red"` before `"black"` (deterministic).

### 7.3 Aggregation function

```python
def aggregate_goal_completion_diagnostics(
    replays: List[dict],
    max_depth: int = 3,
    min_component_size: int = 8,
    detection_threshold: int = 2,
    high_value_threshold: float = 0.9,
    high_value_delay_threshold_plies: int = 10,
    worst_cases_top_k: int = 25,
) -> dict:
    """Per-game compute → bucket by outcome_class → return summary block."""
```

One pass over replays. For each replay: classify outcome, replay the move history with `TwixtState`, compute per-ply `compute_goal_completion_state` for the in-scope side(s), build the per-game record, accumulate into population buckets.

### 7.4 `summary.json["goal_completion"]` shape

```jsonc
"goal_completion": {
  "config": {
    "max_depth": 3,
    "min_component_size": 8,
    "detection_threshold": 2,
    "high_value_threshold": 0.9,
    "high_value_delay_threshold_plies": 10,
    "worst_cases_top_k": 25
  },

  "main_population": {                                       // Class 1
    "scope": "decisive_winner_only",
    "games": 990,
    "games_with_dominant_unclosed": 312,
    "games_with_total_distance_le_2": 142,
    "games_with_total_distance_le_3": 287,
    "detected": 142,
    "conversion_delay_plies":        {"p50": 4, "p90": 12, "p95": 18, "max": 24, "mean": 5.6},
    "conversion_delay_winner_moves": {"p50": 2, "p90":  6, "max":  12, "mean": 2.8},
    "move_quality_after_detection": {                        // pooled across all detected Class 1 games
      "completes_endpoint_rate":          0.27,
      "reduces_total_goal_distance_rate": 0.06,
      "redundant_reinforcement_rate":     0.51,
      "off_chain_rate":                   0.12,
      "other_rate":                       0.04,
      "dominant_unavailable_rate":        0.00              // separate from primary_class rates
    },
    "high_value_diagnostics": {
      "search_score_coverage_pct":        100.0,
      "max_search_score_after_detection": {"p50": 0.86, "p90": 0.99, "max": 1.00},
      "mean_search_score_after_detection":{"p50": 0.62, "p90": 0.94, "max": 0.99}
    },
    "bad_cases": {
      "delay_ge_10_plies":           18,
      "delay_ge_20_plies":            3,
      "root_value_high_but_delayed": 14
    }
  },

  "capped_population": {                                     // Class 2
    "scope": "both_sides",
    "games": 8,
    "games_with_dominant_unclosed": 3,
    "detected_before_cap": 3,
    "cap_delay_after_detection_plies": {"p50": 22, "p90": 38, "max": 51},
    "bad_cases": {
      "state_cap_after_detection":   2,
      "timeout_after_detection":     1,
      "board_full_after_detection":  0
    }
  },

  "excluded_population": {
    "games": 2
  },

  "diagnostics_coverage": {                                  // populated by Phase 3
    "games_with_diagnostics":     0,
    "total_records":              0,
    "coverage_pct_of_decisive_games": 0.0,
    "error_count":                0,
    "version":                    1
  }

  // Phase 3 also adds "policy_mcts_summary" sub-block (see §8.6)
}
```

### 7.5 `report.txt` section (immediately after Per-game stats)

```
Goal-Completion / Conversion Diagnostics
========================================
Config: detection<=2 / max_depth=3 / min_component=8 / high_value>=0.9
Population split: 990 decisive / 8 capped / 2 excluded

Main (decisive wins, winner-only):
  Dominant-unclosed reached: 312 / 990 (31.5%)
    Strict closeout (<=2): 142    Broader (<=3): 287
  Detected (gate=<=2): 142
  Conversion delay:
    plies:        p50=4 p90=12 p95=18 max=24 mean=5.6
    winner moves: p50=2 p90=6  max=12 mean=2.8
  Move quality after detection (pooled):
    endpoint completion: 27.0%
    distance reducing:    6.0%
    redundant reinforce: 51.0%
    off-chain:           12.0%
    other:                4.0%
    dominant unavailable: 0.0%
  High value after detection:
    max search_score:  p50=0.86 p90=0.99 max=1.00
    mean search_score: p50=0.62 p90=0.94 max=0.99
  Bad cases:
    delay >=10 plies:               18
    delay >=20 plies:                3
    high value but delayed:         14

Capped (state_cap / timeout / board_full):
  Games:                              8
  Dominant unclosed before cap:       3
  Cap delay after detection:
    plies: p50=22 p90=38 max=51
  Bad cases:
    state_cap after detection:        2
    timeout after detection:          1
    board_full after detection:       0

Worst cases (top 25 by conversion_delay_plies → goal_completion_worst_cases.csv):
  iter_108_game_097 reason=win detected=35 end=59 delay=24 redundant=8 off_chain=2
  iter_092_game_054 reason=win detected=27 end=48 delay=21 redundant=7 off_chain=3
  ...
```

Empty/partial-coverage rules:
- `main_population.games == 0` → render header + `No decisive games in this run.`
- Per-move search_score coverage 0 across all detected games → suppress the `High value after detection` block and `high value but delayed` line; keep the rest.
- Total detection (Class 1 + Class 2) is zero → render header + `No dominant-unclosed positions detected this run.` and stop.

### 7.6 `goal_completion_worst_cases.csv`

Schema:
```
iteration,game_idx,game_id,winner,starting_player,n_moves,reason,
detected_player,
first_dominant_unclosed_ply,first_total_goal_distance,first_category,
actual_win_ply,conversion_delay_plies,conversion_delay_winner_moves,
distance_reducing_moves,endpoint_completion_moves,redundant_reinforcement_moves,off_chain_moves,other_moves,dominant_unavailable_moves,
max_search_score_after_detection,mean_search_score_after_detection,
high_value_after_detection_plies,root_value_high_but_delayed,
state_cap_after_detection,timeout_after_detection,board_full_after_detection,
outcome_class,scope
```

For Class 2 rows: `winner = null`, `actual_win_ply = null`, `conversion_delay_plies` reused for the Class 2 sort (carries `cap_delay_after_detection_plies`); `outcome_class = 2` and `scope = both_sides` make the semantics unambiguous.

Sort: `conversion_delay_plies DESC, redundant_reinforcement_moves DESC, iteration ASC`. Top-K configurable via `--goal-completion-worst-cases-top-k` (default 25). Both Class 1 and Class 2 detected games eligible; sort is over the unified pool.

### 7.7 Edge cases

| Case | Behavior |
|---|---|
| Replay loaded with no `meta.reason` | classified as Class 3 (excluded) |
| `reason == "win"` but `winner == null` | warning logged; treated as Class 3 (corrupt record) |
| Detection ply == `n_moves` (detected on the winning move itself) | `conversion_delay_plies = 0`, `conversion_delay_winner_moves = 0` |
| Class 1 game where structure dissolves before win | watch window still extends to `actual_terminal_ply`; `winner_moves_with_dominant_unavailable` counts the dissolved-state moves |
| Class 2 game where neither side reaches dominant-unclosed | not counted in `detected_before_cap`; counted in `capped_population.games` only |
| Old replays without per-move search_score | `max_*`, `mean_*`, `high_value_*`, `root_value_high_but_delayed` all `null`; `search_score_coverage_in_watch_window` reflects this |
| Replay with no moves array | counted in `n_games_total` but excluded from goal_completion analysis |
| `meta.starting_player` absent | falls back to `replay["starting_player"]` per existing analyzer convention |

### 7.8 Tests (Phase 2) — 22 tests

1. `test_aggregate_empty_replays_returns_zero_block`
2. `test_aggregate_class1_detected_simple_2ply_closeout` *(Game-097-style fixture)*
3. `test_aggregate_class1_undetected_when_min_component_size_unmet`
4. `test_aggregate_class1_undetected_when_distance_above_threshold`
5. `test_aggregate_class1_first_dominant_unclosed_ply_locks_at_first_occurrence`
6. `test_aggregate_class1_watch_window_classifies_each_winner_move_into_primary_class`
7. `test_aggregate_class1_dominant_unavailable_counted_separately_from_primary_class`
8. `test_aggregate_class1_high_value_after_detection_uses_search_score_threshold`
9. `test_aggregate_class1_root_value_high_but_delayed_requires_both_high_value_and_delay`
10. `test_aggregate_class2_state_cap_with_detected_dominant_increments_bad_case`
11. `test_aggregate_class2_no_detection_excluded_from_detected_count`
12. `test_aggregate_class2_uses_both_sides_scope_for_detection`
13. `test_aggregate_class3_draw_reason_excluded`
14. `test_aggregate_outcome_class_partition_sums_to_n_games_total`
15. `test_aggregate_le_2_and_le_3_buckets_independent_of_detection_threshold`
16. `test_aggregate_pooled_rates_use_winner_moves_with_dominant_denominator`
17. `test_aggregate_old_replays_without_search_score_yield_null_high_value_fields`
18. `test_aggregate_worst_cases_csv_sort_order_correct`
19. `test_aggregate_worst_cases_csv_top_k_respects_flag`
20. `test_aggregate_worst_cases_csv_class2_rows_have_null_winner_and_win_ply`
21. `test_format_goal_completion_report_full_population_renders_all_sections`
22. `test_format_goal_completion_report_zero_detection_short_message`

### 7.9 Implementation sequencing (Phase 2) — 4 commits

1. `feat(analyzer): aggregate_goal_completion_diagnostics scaffolding + Class 1 detection` — Tests #1–#9.
2. `feat(analyzer): Class 2 capped population + Class 3 exclusion` — Tests #10–#15.
3. `feat(analyzer): goal_completion summary block + report rendering` — Tests #16, #17, #21, #22.
4. `feat(analyzer): goal_completion_worst_cases.csv writer` — Tests #18–#20.

---

## 8. Phase 3 — Inline closeout root diagnostics

Inline emission during self-play. Most complex phase; touches the self-play hot path. Strict adherence to the safety invariant (§4.3).

### 8.1 Hook in `play_game()` (`scripts/GPU/alphazero/self_play.py:546`)

After `mcts.search_from_root(...)` and after the existing opening_diagnostics block:

```python
# --- partial closeout-diagnostic capture (best-effort) ---
gc_state = None
partial_diag = None
if cfg.goal_completion_emit_enabled:
    if len(goal_completion_diagnostics) >= cfg.goal_completion_max_records_per_game:
        goal_completion_diagnostics_meta["records_dropped_by_cap"] += 1
    else:
        try:
            gc_state = compute_goal_completion_state(
                state, state.to_move,
                max_depth=cfg.goal_completion_max_depth,
                min_component_size=cfg.goal_completion_emit_min_component,
            )
            if (gc_state is not None
                    and gc_state["total_goal_distance"] is not None
                    and gc_state["total_goal_distance"] <= cfg.goal_completion_emit_threshold):
                if root.priors_raw is None:
                    goal_completion_diagnostics_meta["skipped_missing_priors_count"] += 1
                else:
                    partial_diag = build_closeout_diagnostic_partial(...)
        except Exception as e:
            goal_completion_diagnostics_meta["error_count"] += 1
            sys.stderr.write(f"[closeout-diag] ply={ply} error: {e}\n")

# --- existing resign check (unchanged) ---
# If resign branch fires while partial_diag is not None,
#   goal_completion_diagnostics_meta["resign_dropped_partial_count"] += 1
#   partial_diag is not finalized.

# --- move selection (existing) ---
selected_move = select_move(...)

# --- finalize closeout diagnostic ---
if partial_diag is not None:
    try:
        full_diag = finalize_closeout_diagnostic(
            partial_diag,
            state_before=state, player=state.to_move,
            selected_move=selected_move,
            goal_state_before=gc_state,
        )
        goal_completion_diagnostics.append(full_diag)
    except Exception as e:
        goal_completion_diagnostics_meta["error_count"] += 1
        sys.stderr.write(f"[closeout-diag] ply={ply} finalize error: {e}\n")

# --- existing position record + apply move + history append ---
```

Partial built before the resign check; finalize built after move selection. Cap-check before the BFS so a pathological game that satisfies `distance ≤ 3` for many plies doesn't bloat the saved JSON.

### 8.2 Module placement for closeout-diagnostic helpers

Preferred: a new `scripts/GPU/alphazero/closeout_diagnostics.py` module (parallel to `opening_diagnostics.py`). Final placement TBD by writing-plans — opening_diagnostics.py is concept-specific and adding closeout-only fields to it would muddy that module.

The new module exposes:
- `build_closeout_diagnostic_partial(...)` — pre-move-selection portion (root_summary, goal_completion sub-block, completion-move ranking).
- `finalize_closeout_diagnostic(...)` — adds selected_move + classification.

Internally `build_closeout_diagnostic_partial` calls `opening_diagnostics.build_root_diagnostic(...)` for the region/policy/visit detail, then wraps it with closeout-specific sub-blocks. **`build_root_diagnostic` is not extended.**

### 8.3 Per-record schema (post-finalize)

```jsonc
{
  "ply": 36,
  "side_to_move": "red",
  "active_size": 24,

  "root_summary": {
    "visit_count": 400,
    "q_value":  0.98,                                  // side_to_move perspective; no winner-conversion needed for inline records
    "nn_value": 0.97
  },

  "goal_completion": {
    "max_depth": 3,
    "total_goal_distance_before": 2,
    "endpoint_distances": {"top": 1, "bottom": 1},
    "largest_component_size": 11,
    "category": "two_endpoint_closeout_2ply",
    "endpoint_completion_moves": [[0, 8], [23, 6]],
    "distance_reducing_moves":   [[0, 8], [23, 6], [22, 4]]   // null when skip_distance_reducing flag set
  },

  "endpoint_completion_ranking": {                            // null when no endpoint_completion_moves
    "best_policy_rank":     12,
    "best_policy_prob":      0.004,
    "best_visit_rank":       8,
    "best_completion_visit_share": 0.04,                      // max visit share among endpoint_completion_moves
    "any_in_policy_top5":    false,
    "any_in_visit_top5":     false
  },
  "distance_reducing_ranking": {                              // null when distance_reducing_moves null/empty
    "best_policy_rank":      9,
    "best_policy_prob":      0.012,
    "best_visit_rank":       6,
    "best_visit_share":      0.07,
    "any_in_policy_top5":    false,
    "any_in_visit_top5":     false
  },

  "selected_move": [15, 11],
  "selected_move_classification": {
    "completes_endpoint":          false,
    "reduces_total_goal_distance": false,
    "is_redundant_reinforcement":  false,
    "is_off_chain":                true,
    "primary_class":               "off_chain",
    "total_goal_distance_before":  2,
    "total_goal_distance_after":   2
  }
}
```

### 8.4 `goal_completion_diagnostics_meta`

```jsonc
{
  "enabled": true,
  "max_depth": 3,
  "emit_threshold": 3,
  "emit_min_component_size": 8,
  "max_records_per_game": 64,
  "skip_distance_reducing": false,
  "diagnostic_version": 1,
  "computed_inline": true,
  "selection_perspective": "side_to_move",
  "storage": "in_game_json",
  "error_count": 0,
  "resign_dropped_partial_count": 0,
  "skipped_missing_priors_count": 0,
  "records_dropped_by_cap": 0
}
```

When `cfg.goal_completion_emit_enabled` is `False`, **neither** `goal_completion_diagnostics` nor `_meta` is written to the saved game JSON — no schema noise on disabled runs.

### 8.5 Plumbing

| Plane | Change |
|---|---|
| `self_play.GameRecord` | Two new fields: `goal_completion_diagnostics: List[dict] = field(default_factory=list)`, `goal_completion_diagnostics_meta: Optional[dict] = None`. |
| `ipc_messages.GameComplete` | Same two optional fields. Pickle-safe. |
| `self_play_worker.py` | Worker accumulates diagnostics during its `play_game` call; `GameComplete` carries them. |
| `game_saver.save_game_replay(...)` | Two new kwargs; written as top-level JSON keys (parallel to `opening_diagnostics` / `opening_diagnostics_meta`). Both keys absent when `meta is None`. |
| `trainer.py` routing helpers | One-line forwards in `_save_game_from_record` / `_save_game_from_ipc`. |

### 8.6 Analyzer Phase 3 surfacing

`aggregate_goal_completion_diagnostics(replays)` from §7.3 grows two sub-blocks:

**`goal_completion.diagnostics_coverage`** (already shape-allocated in Phase 2; now populated):

```jsonc
"diagnostics_coverage": {
  "games_with_diagnostics":          31,
  "total_records":                   84,
  "coverage_pct_of_decisive_games":  3.13,
  "error_count":                      0,
  "resign_dropped_partial_count":     0,
  "skipped_missing_priors_count":     0,
  "records_dropped_by_cap":           0,
  "version":                          1
}
```

**`goal_completion.policy_mcts_summary`** (new, pools across all closeout records):

```jsonc
"policy_mcts_summary": {
  "n_records": 84,
  "endpoint_completion_ranking": {
    "n_rankable":           51,                          // records with non-null endpoint_completion_ranking
    "policy_top1_rate":     0.45,
    "policy_top5_rate":     0.72,
    "visit_top1_rate":      0.30,
    "visit_top5_rate":      0.65
  },
  "distance_reducing_ranking": {
    "n_rankable":           70,
    "policy_top1_rate":     0.51,
    "policy_top5_rate":     0.78,
    "visit_top1_rate":      0.36,
    "visit_top5_rate":      0.71
  },
  "selected_primary_class_rates": {
    "completes_endpoint":          0.18,
    "reduces_total_goal_distance": 0.06,
    "redundant_reinforcement":     0.55,
    "off_chain":                   0.15,
    "other":                       0.06
  },
  "high_value_delayed_closeouts": 17,                    // records with q_value (side_to_move perspective) >= 0.9 AND primary_class ∈ {redundant_reinforcement, off_chain, other} AND total_goal_distance_before <= 2
  "by_distance": {
    "distance_le_2": { "n": 51, ... },
    "distance_eq_3": { "n": 33, ... }
  }
}
```

`n_rankable` is the denominator for top-K rates; `n_records` is the overall record count. Both reported.

### 8.7 Report addition (immediately after the main goal-completion section)

```
Policy/MCTS closeout behavior (n=84 records across 31 games):
  Coverage:                        31 / 990 decisive games (3.1%); 0 capture errors
  Endpoint-completion ranking (n_rankable=51):
    best completion in policy top1: 45.0%   policy top5: 72.0%
    best completion in visit top1:  30.0%   visit top5:  65.0%
  Distance-reducing ranking (n_rankable=70):
    best reducer in policy top1:    51.0%   policy top5: 78.0%
    best reducer in visit top1:     36.0%   visit top5:  71.0%
  Selected (primary class):
    completes endpoint:    18.0%
    reduces distance:       6.0%
    redundant:             55.0%
    off-chain:             15.0%
    other:                  6.0%
  High-value delayed closeouts:    17
  By distance:
    le_2 (n=51): policy_top5=78.4% mcts_top5=72.5% selected_completes=21.6%
    eq_3 (n=33): policy_top5=63.6% mcts_top5=54.5% selected_completes=12.1%
```

When `total_records == 0`: render `Coverage: 0 / N decisive games (0.0%); 0 capture errors. No closeout records captured this run.` and stop.

### 8.8 CLI flags

```
--goal-completion-emit-enabled            (bool, default True; set False to fully disable inline capture)
--goal-completion-emit-threshold          (int, default 3)
--goal-completion-emit-min-component      (int, default 8)
--goal-completion-max-depth               (int, default 3)            # shared with analyzer
--goal-completion-skip-distance-reducing  (bool, default False)
--goal-completion-max-records-per-game    (int, default 64)
```

### 8.9 Edge cases

| Case | Behavior |
|---|---|
| Resign branch fires after partial built | `resign_dropped_partial_count` incremented; partial dropped (no finalize possible without selected_move). Not an error. |
| Adjudication-resolution at ply with dominant-unclosed | Adjudication doesn't change `selected_move`; capture proceeds normally. |
| `priors_raw is None` | `skipped_missing_priors_count` incremented; capture skipped this ply. Not an error. |
| Capture exception | `error_count` incremented; stderr warning; training path unaffected. |
| `endpoint_completion_moves` empty | `endpoint_completion_ranking = null`; record still saved; analyzer's `n_rankable` excludes it. |
| `distance_reducing_moves` skipped (flag) | `distance_reducing_ranking = null`; report shows note when `>50%` of records have null distance_reducing_ranking. |
| Old replay without `goal_completion_diagnostics` | `coverage_pct_of_decisive_games = 0`; `policy_mcts_summary = null`; report shows zero-coverage line. |
| Records-per-game cap reached | `records_dropped_by_cap` incremented; remaining plies in that game capture nothing further. |

### 8.10 Tests (Phase 3) — 18 tests

1. `test_build_closeout_diagnostic_partial_includes_root_summary_and_goal_completion`
2. `test_build_closeout_diagnostic_partial_endpoint_completion_ranking_uses_priors_and_visits`
3. `test_build_closeout_diagnostic_partial_no_endpoint_completion_moves_yields_null_ranking`
4. `test_build_closeout_diagnostic_partial_distance_reducing_ranking_separate_from_endpoint`
5. `test_build_closeout_diagnostic_partial_skip_flag_nulls_distance_reducing`
6. `test_finalize_closeout_diagnostic_classifies_selected_move_via_classify_helper`
7. `test_play_game_emits_diagnostic_when_threshold_and_component_size_satisfied`
8. `test_play_game_skips_emission_below_emit_threshold`
9. `test_play_game_skips_emission_below_min_component_size`
10. `test_play_game_skips_emission_when_emit_enabled_false_meta_block_absent`
11. `test_play_game_diagnostic_exception_increments_error_count_no_crash`
12. `test_play_game_diagnostic_meta_records_config_echo_and_counters`
13. `test_play_game_records_dropped_by_cap_when_max_records_per_game_reached`
14. `test_play_game_resign_dropped_partial_count_incremented_when_resign_after_partial`
15. `test_save_game_replay_writes_goal_completion_diagnostics_array_and_meta_keys`
16. `test_save_game_replay_omits_diagnostic_keys_when_meta_none`
17. `test_aggregate_diagnostics_coverage_counts_games_with_records`
18. `test_aggregate_policy_mcts_summary_pools_records_correctly_by_distance`

### 8.11 Implementation sequencing (Phase 3) — 5 commits

1. `feat(closeout-diag): build_closeout_diagnostic_partial + finalize, pure functions` — new module composing `build_root_diagnostic` with closeout sub-blocks. Tests #1–#6.
2. `feat(self-play): inline goal-completion-state computation + defensive partial capture` — Tests #7–#10, #14.
3. `feat(self-play): finalize closeout diagnostic + counters wiring` — error_count, resign_dropped_partial_count, skipped_missing_priors_count, records_dropped_by_cap. Tests #11–#13.
4. `feat(saver/ipc): thread goal_completion_diagnostics + meta through GameRecord, GameComplete, save_game_replay` — Tests #15–#16.
5. `feat(analyzer): diagnostics_coverage + policy_mcts_summary + report rendering` — Tests #17–#18.

---

## 9. Phase 4 — Strong-advantage probe telemetry surfacing

Implements `docs/superpowers/specs/2026-04-28-strong-advantage-probe-tier-design.md` §6–§7 verbatim. The probe set itself was promoted on 2026-04-28; this phase wires its trainer-side extraction and analyzer-side surfacing.

### 9.1 Trainer changes (`scripts/GPU/alphazero/trainer.py`)

1. Inline probe loop (~line 2802) currently does `forced_probe_summary = tiers.get("forced")`. Add:
   ```python
   strong_advantage_probe_summary = tiers.get("strong_advantage")
   ```

2. Sidecar write (~line 2956): change `build_probe_summary_block(forced_summary=..., strong_advantage_summary=None)` to pass the new variable. Add `"strong_advantage_probe_summary": strong_advantage_probe_summary,` legacy field alongside the existing `forced_probe_summary` (one-release dual-emit window per the predecessor spec).

3. CSV flat fields (~line 3408–3415):
   ```python
   "sas_n":                         (strong_advantage_probe_summary or {}).get("n"),
   "sas_sign_correct_pct":          (strong_advantage_probe_summary or {}).get("sign_correct_pct"),
   "sas_median_abs_v":              (strong_advantage_probe_summary or {}).get("median_abs_v"),
   "sas_delta_sign_correct_pct":    (strong_advantage_probe_summary or {}).get("delta_sign_correct_pct"),
   "sas_rolling5_sign_correct_pct": (strong_advantage_probe_summary or {}).get("rolling5_sign_correct_pct"),
   ```

### 9.2 Analyzer changes (`scripts/twixt_replay_analyzer.py`)

1. Sidecar reader: prefer tier-keyed `sc.get("probe_summary", {}).get("<tier>")` when present; fall back to legacy `sc.get("forced_probe_summary")` / `sc.get("strong_advantage_probe_summary")` flat fields. Tier-name loop over `["forced", "strong_advantage"]`.

2. `agg["strong_advantage_probe_by_iter"]` (list, iter-ordered) and `agg["strong_advantage_probe_latest"]` (dict). Same shape as existing `forced_probe_*`.

3. `summary.json["strong_advantage_probe"] = {by_iter, latest}` parallel to existing `forced_probe`.

4. `report.txt` section after the existing Forced-probe section:
   ```
   Strong-Advantage Probe Sign-Agree
   =================================
   Latest iter NNNN:
     n=N, sign_correct=K (P%), median |v|=V
   Delta vs prev: ±X.X pp sign-correct, ±Y.YY median |v|
   Rolling-5: P% sign-correct, median |v|=V
   Per-iter table: strong_advantage_probe_by_iter.csv
   ```

5. `strong_advantage_probe_by_iter.csv` with columns `iteration, n, n_skipped_size, sign_correct, sign_correct_pct, median_abs_v, delta_sign_correct_pct, delta_median_abs_v, rolling5_sign_correct_pct, rolling5_median_abs_v` (matches existing forced CSV).

### 9.3 Tests

Predecessor spec §9 already specifies `tests/test_strong_advantage_analyzer_aggregation.py` (7 tests). Add three tests beyond:

1. `test_trainer_writes_strong_advantage_probe_summary_alongside_forced` — sidecar dual-emit.
2. `test_trainer_csv_emits_sas_flat_fields` — flat CSV columns populated.
3. `test_analyzer_prefers_probe_summary_block_over_legacy_flat_fields` — precedence.

### 9.4 Implementation sequencing (Phase 4) — 3 commits

1. `feat(trainer): extract strong_advantage tier and populate probe_summary + sidecar + CSV` — three trainer-side edits + flat CSV fields + the three new trainer tests.
2. `feat(analyzer): tier-keyed probe_summary reader with legacy fallback` — sidecar parsing + by_iter/latest aggregation.
3. `feat(analyzer): strong_advantage_probe summary block + report section + by_iter.csv` — surfacing.

---

## 10. Implementation sequencing across phases

Listed in **intended implementation order**, not numeric phase order:

| Order | Phase | Commits | Tests | Risk |
|---|---|---|---|---|
| 1 | 0 — per-move `search_score` + `root_top1_share` | 5 | 11 | Low (additive schema, plumbing) |
| 2 | 1 — connectivity helpers | 2 | 19 | Low (pure functions, no callers) |
| 3 | 2 — replay-side aggregation, summary, report, worst-cases CSV | 4 | 22 | Medium (analyzer integration) |
| 4 | 4 — strong-advantage telemetry | 3 | predecessor §9 (7) + 3 = 10 | Low (predecessor design already done) |
| 5 | 3 — inline closeout diagnostics | 5 | 18 | High (touches self-play hot path) |
| **Total** | | **19** | **~80** | |

**Rationale for ordering**:
- Phase 0 first because per-move data feeds Phase 2 metrics (`high_value_after_detection`) and Phase 3 cross-validation.
- Phase 1 before Phase 2 because Phase 2 depends on the helpers.
- Phase 2 before Phase 4 to keep the diagnostics work continuous.
- Phase 4 before Phase 3 because Phase 4 is independent and lower-risk; demoting Phase 3 (highest risk: hot-path modification) to last means each previous milestone has been validated end-to-end before we modify self-play.

## 11. Test plan summary

Final test-file layout is for writing-plans to settle; the spec defines counts per phase, not exact filenames. Reasonable starting layout:

| File | Phase(s) | Tests |
|---|---|---|
| `tests/test_game_saver_per_move_fields.py` (new) | 0 | 4 (saver tests #1–#4) |
| `tests/test_self_play_per_move_capture.py` (new) | 0 | 2 (in-process capture + resign branch, tests #10–#11) |
| `tests/test_analyzer_per_move_stats.py` (new) | 0 | 5 (aggregate + report tests #5–#9) |
| `tests/test_connectivity_goal_completion.py` (new) | 1 | 19 |
| `tests/test_analyzer_goal_completion.py` (new) | 2 | 22 |
| `tests/test_self_play_closeout_diagnostics.py` (new) | 3 | 14 (build + finalize + self-play hook + saver/IPC) |
| `tests/test_analyzer_closeout_diagnostics.py` (new) | 3 | 4 (analyzer surfacing for closeout: tests #17–#18 plus two report tests) |
| `tests/test_strong_advantage_analyzer_aggregation.py` (modified, predecessor) | 4 | predecessor 7 + 3 = 10 |

Total: **~80** new or new-subsection tests across the listed files.

## 12. Verification

```bash
# After Phase 0:
.venv/bin/python -m pytest tests/test_game_saver_per_move_fields.py tests/test_self_play_per_move_capture.py -v

# After Phase 1:
.venv/bin/python -m pytest tests/test_connectivity_goal_completion.py -v

# After Phase 2:
.venv/bin/python -m pytest tests/test_analyzer_goal_completion.py -v
.venv/bin/python -m pytest tests/test_analyzer_phase2_sidecar_fields.py tests/test_analyzer_phase2_smoke.py tests/test_analyzer_per_game_stats.py -v   # regression

# After Phase 4:
.venv/bin/python -m pytest tests/test_strong_advantage_analyzer_aggregation.py -v

# After Phase 3 (final):
.venv/bin/python -m pytest tests/test_self_play_closeout_diagnostics.py tests/test_analyzer_goal_completion.py -v
.venv/bin/python -m pytest tests/ -k "self_play or trainer or analyzer or connectivity" -v   # broad regression

# Manual: short live self-play run, inspect saved JSON for new fields, run analyzer end-to-end, inspect summary.json + report.txt for the new sections
```

Each phase's verification gate must be green before the next phase commits.

## 13. Out of scope (this spec)

- New probe-set creation, relabeling, or promotion (Spec 2 if needed).
- Goal-completion probe tier (Spec 2).
- Training-loop changes — conversion curriculum, conversion policy loss weight, dominant-unclosed reweighting, value-weight tuning (Spec 2).
- Replay.html or other UI consumer changes.
- Backfill of old replay JSONs to the new schema.
- Per-iter trend chart matplotlib figures.
- New connectivity-diagnostics ply-bucket aggregations beyond the existing `aggregate_connectivity_by_ply`.
- Widening of replay-buffer / training-position schema (`PositionRecord` unchanged).

## 14. Open questions

None. All design decisions resolved during brainstorming.

## 15. CLI flags introduced

Nine total, with `--goal-completion-max-depth` shared by analyzer and self-play config plumbing:

```
--goal-completion-detection-threshold      (int,   default 2)         # Phase 2 (analyzer)
--goal-completion-high-value-threshold     (float, default 0.9)       # Phase 2 (analyzer)
--goal-completion-worst-cases-top-k        (int,   default 25)        # Phase 2 (analyzer)
--goal-completion-emit-enabled             (bool,  default True)      # Phase 3 (self-play)
--goal-completion-emit-threshold           (int,   default 3)         # Phase 3 (self-play)
--goal-completion-emit-min-component       (int,   default 8)         # Phase 3 (self-play)
--goal-completion-max-depth                (int,   default 3)         # shared: Phase 1 helper / Phase 2 analyzer / Phase 3 self-play
--goal-completion-skip-distance-reducing   (bool,  default False)     # Phase 3 (self-play perf escape)
--goal-completion-max-records-per-game     (int,   default 64)        # Phase 3 (self-play safety cap)
```
