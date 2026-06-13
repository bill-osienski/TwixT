# Goal-Completion / Conversion Diagnostics — Planning Snippets

Context pack for planning the goal-completion / conversion diagnostics work and the strong-advantage probe activation. Triggered by analysis of `iter_0108_game_097.json`, where Red had a forced win at turn 35 (chain from row 1 to row 20, no Black interference at either endpoint) but drifted for 22 plies before closing.

Snippets 1-5 are the minimum needed for the diagnostics plan. 6-7 are for probe activation. 8-9 are for later training-correction knobs.

---

## (1) `iter_0108_game_097.json` — metadata + moves 30-59

```jsonc
{
  "id": "iter_0108_game_097",
  "timestamp": "2026-05-02T06:35:40.600054+00:00",
  "config_hash": "alphazero",
  "depth": 400,
  "seed": 97,
  "winner": "red",
  "starting_player": "red",
  "moves": [
    /* moves 1-29 elided for brevity — full game is in chat history */
    {"turn": 30, "player": "black", "row":  9, "col": 22, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 31, "player": "red",   "row": 10, "col":  4, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 32, "player": "black", "row":  8, "col": 20, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 33, "player": "red",   "row":  7, "col":  5, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 34, "player": "black", "row":  7, "col": 23, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 35, "player": "red",   "row":  1, "col":  6, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 36, "player": "black", "row":  7, "col": 14, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 37, "player": "red",   "row": 15, "col": 11, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 38, "player": "black", "row": 18, "col": 11, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 39, "player": "red",   "row": 17, "col": 15, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 40, "player": "black", "row": 20, "col": 15, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 41, "player": "red",   "row": 16, "col":  3, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 42, "player": "black", "row": 19, "col": 17, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 43, "player": "red",   "row": 22, "col":  4, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 44, "player": "black", "row": 20, "col": 19, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 45, "player": "red",   "row": 14, "col":  2, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 46, "player": "black", "row": 18, "col": 15, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 47, "player": "red",   "row":  8, "col":  3, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 48, "player": "black", "row": 21, "col": 17, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 49, "player": "red",   "row":  5, "col":  4, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 50, "player": "black", "row": 19, "col": 21, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 51, "player": "red",   "row": 18, "col":  6, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 52, "player": "black", "row": 20, "col": 23, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 53, "player": "red",   "row": 20, "col":  3, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 54, "player": "black", "row": 19, "col":  9, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 55, "player": "red",   "row": 10, "col":  2, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 56, "player": "black", "row": 21, "col": 21, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 57, "player": "red",   "row": 23, "col":  6, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 58, "player": "black", "row": 22, "col": 19, "bridges_created": [], "heuristics": {}, "search_score": null},
    {"turn": 59, "player": "red",   "row":  0, "col":  8, "bridges_created": [], "heuristics": {}, "search_score": null}
  ],
  "meta": {
    "board_size": 24,
    "mode": "alphazero",
    "reason": "win",
    "iteration": 108,
    "game_idx": 97,
    "simulations": 400,
    "n_moves": 59,
    "starting_player": "red",
    "worker_id": 3,
    "wall_time_s": 32.058578916825354,
    "adjudication_block_reason": null,
    "final_root_value": 0.9987027534928778,
    "final_top1_share": 0.5,
    "compute": {"leaf_evals": 11036, "backups": 23600, "nn_batches": 937}
  },
  "opening_diagnostics": [/* per-ply MCTS detail for first 6 plies — see snippet 5 for schema */],
  "opening_diagnostics_meta": { /* config echo, see snippet 5 */ }
}
```

**Note on `bridges_created` and `search_score`:** both fields are present in the schema but never populated (saved as `[]` and `null` respectively). The trainer doesn't currently fill these. This is the per-move-detail gap that any goal-completion diagnostic will need to start filling.

---

## (2) Rule helpers — `scripts/GPU/alphazero/game/twixt_state.py`

```python
# Constants (top of file)
BOARD_SIZE = 24
MAX_PLIES = 600
KNIGHT_MOVES = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]

@dataclass
class TwixtState:
    """Immutable-style TwixT state. Methods return new objects (no mutation)."""
    board_size: int = 24
    active_size: int = 24      # curriculum (≤ board_size)
    to_move: str = "red"
    pegs: Dict[Pos, str] = field(default_factory=dict)
    bridges: Set[Bridge] = field(default_factory=set)
    ply: int = 0
    max_plies_limit: Optional[int] = None

    def is_valid_placement(self, row, col):
        """Rules:
        1. Within [0, active_size)
        2. Empty
        3. Active-region corners forbidden
        4. Red cannot place on left/right edges of active region
        5. Black cannot place on top/bottom edges of active region
        """
        # ... (full body in file lines 170-204)

    def legal_moves(self) -> List[Pos]:
        """Return all valid (row,col) for current player, sorted."""
        moves = []
        for row in range(self.active_size):
            for col in range(self.active_size):
                if self.is_valid_placement(row, col):
                    moves.append((row, col))
        return moves

    def apply_move(self, move: Pos) -> TwixtState:
        """Validates, places peg, finds new bridges (knight-move neighbors of
        same color whose proposed bridge does not properly-intersect any
        existing bridge), switches to_move. Returns new state.
        Raises ValueError if illegal."""
        # ... (full body in file lines 316-355)

    # Bridge legality: knight-move + same color + not already present + does
    # not properly-intersect any opposing or same-color existing bridge.
    # Implementation in _find_new_bridges + _proper_intersect_knight (lines 222-314).

    def _check_win(self, player: str) -> bool:
        """For 'red': BFS from pegs on row 0; win if any reaches row active-1.
        For 'black': BFS from pegs on col 0; win if any reaches col active-1.
        Uses self.bridges as adjacency."""
        # ... (full body in file lines ~480-517)

    def winner(self) -> Optional[str]:
        if self._check_win("red"):   return "red"
        if self._check_win("black"): return "black"
        return None

    def is_terminal(self) -> bool:
        if self.winner() is not None: return True
        if self.max_plies_limit and self.ply >= self.max_plies_limit: return True
        if self.ply >= MAX_PLIES: return True
        # plus board-full check via legal_moves()
```

**Key facts for goal-completion logic:**
- Red goal lines: row 0 (top) and row 23 (bottom). Red CAN place on rows 0, 23 (only restricted from cols 0, 23).
- Black goal lines: col 0 (left) and col 23 (right). Black CAN place on cols 0, 23 (only restricted from rows 0, 23).
- Active-region corners forbidden for everyone.
- "Win" = a single connected component (via `bridges`) touches BOTH of that color's goal lines.

---

## (3) Connectivity helpers — `scripts/GPU/alphazero/connectivity_diagnostics.py`

Full file (102 lines):

```python
"""Connectivity-aware replay diagnostics — Phase 1 of the retrain design spec.

Computes per-position Twixt-structural stats (goal-touching components,
largest component size, etc.) from game JSON move histories, then aggregates
by ply bucket + outcome for analyzer-side reporting.
"""
from __future__ import annotations
from typing import Dict, List
from collections import defaultdict

from .game.twixt_state import TwixtState


def compute_position_connectivity(state: TwixtState) -> Dict[str, object]:
    """Per-position connectivity stats using the shared connectivity_masks helper."""
    out: Dict[str, object] = {}

    for player, prefix, goal1_name, goal2_name in (
        ("red", "red", "top", "bottom"),
        ("black", "black", "left", "right"),
    ):
        m_g1, m_g2, m_both = state.connectivity_masks(player)
        out[f"{prefix}_has_{goal1_name}_component"] = bool(m_g1.sum() > 0)
        out[f"{prefix}_has_{goal2_name}_component"] = bool(m_g2.sum() > 0)

        # Largest component size
        pegs_of = [(r, c) for (r, c), col in state.pegs.items() if col == player]
        seen = set()
        sizes = []
        for peg in pegs_of:
            if peg in seen:
                continue
            comp = state._get_connected_component(peg, player)
            sizes.append(len(comp))
            seen.update(comp)
        out[f"{prefix}_largest_component_size"] = max(sizes) if sizes else 0

        # Number of goal-touching components (0, 1, or 2)
        seen = set()
        touching_count = 0
        for peg in pegs_of:
            if peg in seen:
                continue
            comp = state._get_connected_component(peg, player)
            seen.update(comp)
            if player == "red":
                touches = any(r == 0 or r == state.active_size - 1 for (r, _) in comp)
            else:
                touches = any(c == 0 or c == state.active_size - 1 for (_, c) in comp)
            if touches:
                touching_count += 1
        out[f"{prefix}_n_goal_touching_components"] = min(touching_count, 2)

    return out


def aggregate_connectivity_by_ply(game_records: List[dict], ply_buckets) -> List[dict]:
    """Bucket per-position stats by (ply_bucket, color, outcome)."""
    buckets: Dict = defaultdict(lambda: defaultdict(list))
    for gr in game_records:
        move_history = [(int(m["row"]), int(m["col"])) for m in (gr.get("moves") or [])]
        active = (gr.get("meta") or {}).get("board_size", 24)
        start_player = gr.get("starting_player") or (gr.get("meta") or {}).get("starting_player", "red")
        winner = gr.get("winner", "draw")
        state = TwixtState(active_size=active, to_move=start_player)

        for ply, (r, c) in enumerate(move_history):
            state = state.apply_move((r, c))
            stats = compute_position_connectivity(state)

            bucket_label = "other"
            for lo, hi, label in ply_buckets:
                if lo <= ply + 1 <= hi:
                    bucket_label = label
                    break

            key = (bucket_label, winner)
            buckets[key]["red_largest_component_size"].append(stats["red_largest_component_size"])
            buckets[key]["black_largest_component_size"].append(stats["black_largest_component_size"])
            buckets[key]["red_has_top_component"].append(int(stats["red_has_top_component"]))
            buckets[key]["red_has_bottom_component"].append(int(stats["red_has_bottom_component"]))
            buckets[key]["black_has_left_component"].append(int(stats["black_has_left_component"]))
            buckets[key]["black_has_right_component"].append(int(stats["black_has_right_component"]))
            buckets[key]["red_n_goal_touching_components"].append(stats["red_n_goal_touching_components"])
            buckets[key]["black_n_goal_touching_components"].append(stats["black_n_goal_touching_components"])

    rows = []
    for (bucket_label, outcome), data in sorted(buckets.items()):
        if not data.get("red_largest_component_size"):
            continue
        n = len(data["red_largest_component_size"])
        row = {"ply_bucket": bucket_label, "outcome": outcome, "n": n}
        for k, vs in data.items():
            row[f"mean_{k}"] = round(sum(vs) / n, 3)
        rows.append(row)
    return rows
```

The key primitives it uses are on `TwixtState`:
- `state.connectivity_masks(player) -> (mask_goal1, mask_goal2, mask_both)` — per-cell BFS reachability masks for each goal line and their intersection
- `state._get_connected_component(peg, player) -> Set[Pos]` — BFS over `state.bridges`

For the goal-completion diagnostic, the natural extension is a new function in this file:

```python
def compute_goal_completion_distance(state: TwixtState, player: str) -> int | None:
    """Minimum number of additional pegs of `player` needed to win.
    Returns None if not computable (e.g., other side already won).

    Approach: BFS from each peg of `player`'s components to the two goal lines
    counting "missing knight-bridge endpoints" along the shortest extending
    path. Captures the 2-peg-to-win case in game 097 turn 35 → returns 2.
    """


def is_dominant_unclosed(state: TwixtState, player: str,
                         largest_component_threshold: int = 8,
                         max_distance: int = 3) -> bool:
    """Position has goal_completion_distance(player) <= max_distance AND
    that player's largest goal-touching component >= threshold AND
    the opponent has no goal-touching component (or one but small)."""
```

Both would slot into `aggregate_connectivity_by_ply` to produce per-iter buckets that surface in `summary.json` and `report.txt`.

### Tensor-side classifier (separate code path)

`_classify_position_from_tensor` in `scripts/GPU/alphazero/trainer.py:533` drives `sanity_by_connectivity` from raw NHWC tensors at training time:

```python
def _classify_position_from_tensor(
    board_tensor: np.ndarray, winning_size_threshold: int = 8
) -> str:
    """Bucket a position into 'winning_structure' or 'no_winning_structure'.

    Uses channels 24-29 (Phase 2 connectivity masks) directly — no state
    reconstruction needed.

    Channel layout:
        24: red_connected_to_top
        25: red_connected_to_bottom
        26: red_connected_to_both    (terminal-only)
        27: black_connected_to_left
        28: black_connected_to_right
        29: black_connected_to_both  (terminal-only)

    Bucket rule:
        winning_structure  = either color has a goal-touching component AND
                             (≥ winning_size_threshold pegs in that component
                              OR pegs touching both goal edges).
    """
    if board_tensor.ndim != 3 or board_tensor.shape[-1] < 30:
        return "unknown"
    s_red_top = float(np.sum(board_tensor[..., 24]))
    s_red_bot = float(np.sum(board_tensor[..., 25]))
    s_blk_left = float(np.sum(board_tensor[..., 27]))
    s_blk_right = float(np.sum(board_tensor[..., 28]))
    red_winning = (
        s_red_top >= winning_size_threshold
        or s_red_bot >= winning_size_threshold
        or (s_red_top > 0 and s_red_bot > 0)
    )
    black_winning = (
        s_blk_left >= winning_size_threshold
        or s_blk_right >= winning_size_threshold
        or (s_blk_left > 0 and s_blk_right > 0)
    )
    if red_winning or black_winning:
        return "winning_structure"
    return "no_winning_structure"
```

For the goal-completion diagnostic at analysis time, the `TwixtState`-based connectivity_diagnostics path is the right starting point because it can replay full move histories.

---

## (4) Replay analyzer entry points — `scripts/twixt_replay_analyzer.py`

```python
def analyze(replays: List[dict],
            out_dir: str,
            board_size_override: Optional[int],
            edge_pad: int, opening_k: int, opening_geom_kmax: int,
            near_corner_radius: int, edge_band_width: int,
            buckets_spec: str, window: int,
            run_config: Optional[dict] = None,
            meta: Optional[dict] = None,
            sidecars: Optional[Dict[int, dict]] = None,
            no_plots: bool = False,
            dump_root_child_per_game: bool = False,
            out_suffix: Optional[str] = None,
            calibrate: bool = False,
            calibrate_weights: Optional[str] = None,
            no_connectivity: bool = False,
            args: Optional[argparse.Namespace] = None) -> None:
    """Main analyzer flow. ~1000 lines. Outline:
       1. iterate replays — accumulate per-replay rows, opening counts,
          edge/corner rates, heatmaps, opening sequences
       2. compute opening drift (KL divergence between game windows)
       3. aggregate sidecars (if present) → sc_agg
       4. aggregate per-game stats → per_game_stats_val
       5. opening_diagnostics aggregation (build_root_diagnostic outputs)
       6. connectivity_diagnostics (uses connectivity_diagnostics.aggregate_connectivity_by_ply)
       7. replay_probe_scoring (network forward over winning-replay positions)
       8. value_calibration (network forward + bucketed calibration)
       9. assemble `summary` dict, dump to summary_<suffix>.json
      10. assemble `lines` list, dump to report_<suffix>.txt
      11. write CSVs (replay_summary, opening_*, replay_cap_by_iter, etc.)
      12. write heatmap PNGs (gated by no_plots)"""

# The summary/report assembly is where per_game_stats was wired in:
# - summary dict literal at ~line 2300 (where summary["per_game_stats"] was added)
# - report-text assembly at ~line 2610 (where format_per_game_stats_report() is invoked,
#   correctly OUTSIDE the `if use_sidecar:` block per the Task 5 fix).
# Same pattern applies for any new "goal_completion" block.

def main():
    ap = argparse.ArgumentParser(description="Analyze TwixT self-play replay JSONs...")
    ap.add_argument("--input", nargs="+", required=True, ...)
    ap.add_argument("--out", required=True, ...)
    # 30+ flags — board_size, edge_pad, opening_k, ply_buckets, window,
    # run-config, meta, no_plots, calibrate, calibrate_weights,
    # winning_structure_min_size, no-connectivity, weights, checkpoint-dir,
    # probe-scoring-disable, calibration-disable, etc.
    args = ap.parse_args()
    replays = load_replays(args.input)
    sidecars = load_sidecars(args.input)
    analyze(replays=replays, out_dir=args.out, sidecars=sidecars, ...)
```

### Insertion points for a `goal_completion` block

| Stage | Where | What to add |
|---|---|---|
| Aggregation | call site near per_game_stats_val (~line 2285) | `goal_completion_val = aggregate_goal_completion_diagnostics(replays)` |
| summary.json | dict literal entry (~line 2310) | `"goal_completion": goal_completion_val,` |
| report.txt | after Per-game stats section (~line 2615) | `lines.extend(format_goal_completion_report(summary["goal_completion"]))` |

### Existing format_*_report functions (style reference)

- `format_replay_cap_report` — line 1045
- `format_per_game_stats_report` — line 1241
- `format_connectivity_diagnostics_report` — line 1372
- `format_sanity_by_connectivity_report` — line 1396
- `format_tier_probe_report` — line 1454
- `format_forced_probe_report` — line 1515
- `format_value_calibration_report` — line 1520
- `format_replay_probe_scoring_report` — line 1555

---

## (5) Sidecar / root-child diagnostic schema

Function: `build_root_diagnostic(...)` in `scripts/GPU/alphazero/opening_diagnostics.py:130`. Writes one entry per ply during the diagnostic window.

### Output schema (one entry per ply, only first ~6 plies of each game)

```jsonc
// Each entry inside meta.opening_diagnostics array:
{
  "ply": 0,
  "side_to_move": "black",
  "penalties_active": {"edge_band": false, "near_corner": false},
  "effective_near_corner_penalty": 0.0,
  "near_corner_penalty_source": "none",
  "config": { /* echo of run penalty config */ },
  "legal_moves_total": 528,
  "legal_move_counts":   {"near_corner": 24,  "edge_band": 108, "interior": 396},
  "raw_mass":            {"near_corner": 0.006, "edge_band": 0.020, "interior": 0.974},
  "penalized_mass":      {"near_corner": 0.026, "edge_band": 0.107, "interior": 0.866},
  "visit_mass":          {"near_corner": 0.003, "edge_band": 0.000, "interior": 0.997},
  "raw_top1":       {"move": [19, 18], "share": 0.146, "is_edge_band": false, "is_near_corner": false, "primary_region": "interior"},
  "penalized_top1": {"move": [19, 18], "share": 0.073, ...},
  "visit_top1":     {"move": [19,  5], "share": 0.153, ...},
  "root_summary": {
    "visit_count": 400,
    "q_value": -0.0647,
    "nn_value": -0.05    // present in newer schema; pre-feature games may omit
  },
  "top_children": [/* top-k child detail: visits, q, prior, region */]
}
```

### Function signature

```python
def build_root_diagnostic(
    ply: int,
    side_to_move: str,
    visit_counts: Dict[Tuple[int, int], int],
    priors_raw: Dict,
    priors_adjusted: Dict,
    board_size: int,
    band_width: int,
    corner_radius: int,
    edge_penalty: float,
    corner_penalty: float,
    edge_penalty_ply: int,
    corner_penalty_ply: int,
    decode_fn,
    corner_penalty_early: float = 0.0,
    corner_penalty_early_plies: int = 0,
) -> dict:
    """Build a single per-root diagnostic record.

    Inputs:
        visit_counts: Dict[(row, col) -> visits] from MCTS
        priors_raw: raw NN output (root.priors_raw)
        priors_adjusted: post-root-adjustment (Dirichlet noise + penalties)
        decode_fn: move_id -> (r, c) for key normalization

    Output: dict matching the schema above, including raw/penalized/visit
    mass distributions over (near_corner, edge_band, interior) regions and
    top-1 per stage with region classification.
    """
```

### Schema is per-ply but currently only emitted for plies 0..diag_end_ply

- The window is configurable (typically 4-6 plies)
- Diagnostic window control: `compute_diagnostic_end_ply(edge_penalty_ply, corner_penalty_ply, floor=4, extra=2)`

For per-move detail across the **full game** — this is the natural extension point. Same `build_root_diagnostic` schema but emitted for every ply (or filtered to plies where `is_dominant_unclosed(state, player)` is true). The per-game JSON would gain a parallel `goal_completion_diagnostics: [...]` array OR a sidecar `iter_NNNN_game_NNN_trace.json`.

### Writer hookup

The existing diagnostic window writer is at `scripts/GPU/alphazero/self_play.py:552` inside the per-ply loop, gated on `ply < diag_end_ply`. To add per-ply records for the whole game (or filtered to "dominant unclosed" positions), wrap the same `build_root_diagnostic(...)` call in a parallel emission path keyed off the `is_dominant_unclosed` predicate.

---

## (6) Probe file schema and loader

```jsonc
// tests/probes/twixt_probes.json AND strong_advantage_probes.draft.json
{
  "meta": {
    "type": "...",
    "tier": "forced" | "strong_advantage",
    "not_gate_suite": true,                  // for strong_advantage draft
    "review_mode": "light",
    "reviewer": "<name>",
    "reviewed_at_utc": "...",
    "generator": "...",
    "generator_version": "...",
    "selection_rules": { ... }
  },
  "probes": [
    {
      "id": "iter_0058_game_044_ply066_chain_advantage_central_red",
      "category": "chain_advantage_central_red",
      "confidence": "strong_advantage",        // or "forced"
      "side_to_move": "red",
      "expected_value_sign": 1,                 // +1 = side_to_move winning
      "active_size": 24,
      "ply": 66,
      "move_history": [[4,4], [8,5], ..., [r, c]],   // replay sequence to set up the position
      "source_game": "iter_0058_game_044.json",
      "source_ply": 66,
      "starting_player": "red",
      "phase1_features": { /* mining-time features used to qualify the candidate */ },
      "phase2_label":    { /* deep-MCTS labeling: mean_root_value, value_per_run,
                             value_stability, min_top1_share, etc. */ }
    },
    /* ... 29 more probes for strong_advantage; 30 for forced ... */
  ]
}
```

### Loader/evaluator entry points — `scripts/GPU/alphazero/probe_eval.py`

```python
def _replay_probe(probe: dict) -> TwixtState:
    """Reconstruct TwixtState from probe['move_history'] + probe['active_size']
    + probe['starting_player']."""

def _eval_probe(probe: dict, evaluator: LocalGPUEvaluator, sims: int) -> dict:
    """Run network on the replayed state. Returns:
       { id, expected_value_sign, predicted_value, sign_correct, abs_v, ... }"""

def run_forced_probes_inline(
    probes_file: str,
    evaluator: LocalGPUEvaluator,
    sims: int = 0,
    iteration: int | None = None,
    rolling_window_size: int = 5,
) -> dict:
    """Loads probes file, runs evaluator on each, aggregates by tier:
       returns {n, sign_correct, sign_correct_pct, median_abs_v,
                delta_sign_correct_pct (vs prior iter),
                rolling5_sign_correct_pct, ...}
       The current trainer only consumes the 'forced' tier slice of this."""

def _aggregate(rows: list[dict]) -> dict:
    """Tier-by-tier aggregation; returns {forced: {...}, strong_advantage: {...}}.
       The forced slice is what gets persisted to forced_probe_summary today.
       The strong_advantage slice is computed but never written — a one-line
       trainer change would expose it as strong_advantage_probe_summary."""

# Other relevant functions in probe_eval.py:
# - extract_forced_probes_from_games (line 322)
# - extract_strong_advantage_candidates (line 740)
# - compute_phase1_features (line 578)
# - is_forced_within_k (line 709)
# - label_candidate_with_mcts (line 906)
# - apply_admission_filter (line 1010)
```

---

## (7) Trainer hook for probe evaluation — `scripts/GPU/alphazero/trainer.py`

```python
# At iteration startup (~line 1775 / before games begin):
#   - load probes file (--probes path arg, default tests/probes/twixt_probes.json)
#   - cache the probes dict for per-iter reuse

# Inside the per-iteration block (~line 2745-2802):
forced_probe_summary: Optional[dict] = None
if probes_file_loaded and inline_eval_enabled:
    # call run_forced_probes_inline → get tier dict
    tiers = probe_eval.run_forced_probes_inline(probes_file, evaluator, sims)
    forced_probe_summary = tiers.get("forced")
    # NOTE: tiers also contains "strong_advantage" but it is currently
    # NOT extracted. The fix to expose it is:
    # strong_advantage_probe_summary = tiers.get("strong_advantage")

# Sidecar write (~line 2956):
sidecar = {
    ...
    "forced_probe_summary": forced_probe_summary,
    # NEW would be:
    # "strong_advantage_probe_summary": strong_advantage_probe_summary,
    ...
    "sanity_by_connectivity": v_stats.get("sanity_by_connectivity"),
    ...
}

# CSV / metrics flat fields (~line 3408-3415):
"fps_n": (forced_probe_summary or {}).get("n"),
"fps_sign_correct_pct": (forced_probe_summary or {}).get("sign_correct_pct"),
"fps_median_abs_v": (forced_probe_summary or {}).get("median_abs_v"),
"fps_delta_sign_correct_pct": ...,
"fps_rolling5_sign_correct_pct": ...,
# Add parallel `sas_*` fields for strong_advantage tier.
```

`replay_probe_scoring` lives in the analyzer (not the trainer) — it scores the model against winning replays at the per-bucket level. Code is in `scripts/twixt_replay_analyzer.py` near line 1819 (search for `replay_probe_scoring`); the helper that does the actual scoring is in `scripts/GPU/alphazero/probe_eval.py` and `scripts/GPU/alphazero/value_calibration.py`.

---

## (8) Loss + train step — `scripts/GPU/alphazero/trainer.py:1010-1110`

```python
def compute_loss_combined(
    network, positions, *, l2_weight, value_weight, max_moves_cap,
    active_size, progress_weighted, progress_weight_floor,
):
    """Returns (total_loss, policy_loss, value_loss, l2_loss).
       total_loss MUST be first for nn.value_and_grad()."""
    boards, move_rows, move_cols, move_mask, target_pi, outcomes = make_padded_batch(
        positions, max_moves_cap=max_moves_cap
    )
    plies_np       = np.array([getattr(p, "ply", 0) for p in positions], dtype=np.int32)
    game_n_moves_np = np.array([getattr(p, "game_n_moves", None) or 1 for p in positions], dtype=np.int32)

    # Single batched forward pass with curriculum active_size
    logits, values, _ = network.forward_padded(
        boards, move_rows, move_cols, move_mask, active_size=active_size
    )

    # Policy: cross entropy, masked logits already have -1e9 outside legal moves
    log_probs    = logits - mx.logsumexp(logits, axis=1, keepdims=True)   # (B, M)
    policy_loss  = -mx.sum(target_pi * log_probs, axis=1)                  # (B,)
    policy_loss  = mx.mean(policy_loss)

    # Value: progress-weighted MSE OR plain MSE
    if progress_weighted:
        value_loss = _compute_progress_weighted_value_loss(
            values, outcomes, plies_np, game_n_moves_np, floor=progress_weight_floor
        )
    else:
        value_loss = mx.mean((values - outcomes) ** 2)

    # L2 regularization over all params
    l2_loss = sum(mx.sum(p**2) for _, p in flatten_params(network.parameters()))
    l2_loss = l2_weight * l2_loss

    total_loss = policy_loss + value_weight * value_loss + l2_loss
    return total_loss, policy_loss, value_loss, l2_loss


def train_step(
    network, main_module, opt_main, opt_value,
    batch: List[PositionRecord],
    l2_weight=1e-4,
    value_weight=0.5,
    max_moves_cap=512,
    active_size=24,
    value_grad_max_norm=0.5,
    progress_weighted=True,
    progress_weight_floor=0.25,
    ...
):
    """One optimizer step: value_and_grad on compute_loss_combined,
       split-LR (main vs value head), value-grad clipping, etc."""
```

### `make_padded_batch(positions, max_moves_cap)` builds:

| Tensor | Shape | What it carries |
|---|---|---|
| `boards` | (B, H, W, C) | peg & link & connectivity tensors |
| `move_rows` | (B, M_padded) int | legal move row indices padded to `max_moves_cap` |
| `move_cols` | (B, M_padded) int | legal move col indices padded |
| `move_mask` | (B, M_padded) float | 1.0 where legal, 0.0 padding |
| `target_pi` | (B, M_padded) float | visit counts normalized to a probability distribution over legal moves; pad slots are 0 |
| `outcomes` | (B,) float in {-1, 0, +1} | from `to_move`'s perspective |

### Progress-weighted value loss formula

```
weight(ply, game_n_moves) = max(floor, ply / game_n_moves)
value_loss = sum(w * (v - z)^2) / sum(w)
```

Late-game positions weigh more — closes the value-head signal at endgame.

---

## (9) Replay buffer / sample schema

`scripts/GPU/alphazero/self_play.py:290`:

```python
@dataclass
class PositionRecord:
    """Single training position from self-play.
    Stored in MLX-native NHWC layout to avoid transpose during training."""
    board_tensor: np.ndarray             # (H, W, C) - 30 channels (peg/link/dist + connectivity-Phase2)
    to_move: str                          # "red" or "black" — explicit, NOT inferred from index
    legal_moves: List[Tuple[int, int]]    # available (row, col) at this position
    visit_counts: List[int]               # MCTS visits, same order as legal_moves
    outcome: Optional[float] = None       # +1 if to_move won the game, -1 lost, 0 draw
    active_size: int = 24                 # curriculum size at the time
    ply: int = 0                          # ply at which this position occurred (0 = first move)
    game_n_moves: Optional[int] = None    # total plies played in the source game

    # to_dict / from_dict serializers exist for buffer persistence
```

`scripts/GPU/alphazero/trainer.py:1145`:

```python
class ReplayBuffer:
    """Fixed-size ring buffer of PositionRecord. Oldest overwritten when full."""
    def __init__(self, max_size: int = 100000): ...
    def add_game(self, game: GameRecord): ...                  # iterates game.positions
    def add_positions(self, positions: List[PositionRecord]): ...
    def sample(self, batch_size, rng=None, active_size=None) -> List[PositionRecord]:
        """Uniform sampling. If active_size given, filters to matching positions
           (curriculum learning needs same-size batches)."""
```

### Where conversion-curriculum hooks would slot in

| Hook | What it changes | Where |
|---|---|---|
| `conversion_policy_loss_weight` | Reweight policy_loss for "dominant unclosed" positions | `compute_loss_combined` — multiply per-position policy CE by weight before mean |
| `conversion_samples_per_iter` | Mix curated dominant-unclosed positions into each training batch at fixed proportion | `ReplayBuffer.sample` — accept a parallel "curated buffer" and stratified-sample |
| `conversion_curriculum_path` | Path to a JSON like `tests/probes/conversion_curriculum.json` (similar shape to the probe files: `move_history` + `expected closing move`) | New loader, similar to `_replay_probe` in `probe_eval.py` |
| Per-position `is_dominant_unclosed` flag | Tag each PositionRecord at self-play time so the trainer can filter | Add `is_dominant_unclosed: bool = False` to `PositionRecord` and set it in `play_game()` using `connectivity_diagnostics.is_dominant_unclosed(state, winner_side)` |

---

## Cheap quick-win flagged out-of-band

The `search_score` field already exists in the saved-game `moves[]` schema but is unpopulated. Filling it with the per-move `root_value` (which is in MCTS memory at every ply) would be a 5-line change in `play_game()` and would cost roughly nothing — it's the cheapest path to "review moves and scores at a granular level" for the full game, before any of the heavier diagnostics.

Worth bundling into snippet-1's follow-up if writing a goal-completion plan.

---

## Diagnostic findings on `iter_0108_game_097` (reference)

After turn 35, Red had a continuous knight-bridge chain anchored at (1,6) at the top and (20,5) at the bottom:

```
(1,6) ⇌ (3,5) ⇌ (5,6) ⇌ (7,5)/(7,7) ⇌ (9,6) ⇌ (11,5) ⇌ (12,3) ⇌ (14,4) ⇌ (16,5) ⇌ (18,4) ⇌ (20,5)
```

**Black interference around either endpoint:**
- Near rows 0-2: closest Black peg is (5,9). Nothing within bridging distance of cols 2-8.
- Near rows 22-23: closest Black pegs are (20,15), (20,19), (20,23), (21,17), (22,19). All cols ≥ 15, far from Red's chain at col 5.

**Shortest win from turn 36:** 2 plies for Red — e.g., turn 37: (0,8); turn 39: (23,6). Black cannot stop either bridge.

**What Red actually played** (turns 37-55): 10 redundant pegs before finally cashing in on turns 57 and 59.

| Turn | Move | What it does |
|---|---|---|
| 37 | (15,11) | mid-board, helps neither endpoint |
| 39 | (17,15) | bottom-right edge area, not on main chain |
| 41 | (16,3) | redundant — row 16 already connected via (16,5) |
| **43** | **(22,4)** | **extends toward row 23 — actually useful** |
| 45 | (14,2) | redundant interior |
| 47 | (8,3) | redundant interior |
| 49 | (5,4) | redundant — row 5 already covered by (5,6) |
| 51 | (18,6) | redundant — (18,4) already connected |
| 53 | (20,3) | redundant — (20,5) already connected |
| 55 | (10,2) | redundant — (10,4) already connected |
| **57** | **(23,6)** | **finally completes the bottom hop** |
| **59** | **(0,8)** | **finally completes the top hop — WIN** |

### Meta confirms the failure mode

```
final_root_value: 0.9987    # MCTS at turn 59 was 99.9% sure Red wins
final_top1_share: 0.50      # but visit distribution still split 50/50
n_moves: 59
simulations: 400
```

The value head **knew the game was won** the whole time — but the policy never strongly committed to the closing move. Classic "value head says winning everywhere, but no signal differentiates close-out from drift" — exactly the failure mode the strong-advantage probe tier was designed to catch (and is currently inactive in training because the draft was never `--promote`d to `tests/probes/twixt_probes.json`).
