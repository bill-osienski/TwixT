# Connectivity-Retrain Design

**Date:** 2026-04-19
**Status:** Approved for implementation planning
**Spec owner:** bill-osienski

## 1. Problem

The iter-0999 AlphaZero checkpoint exhibits value-head blindness on near-win connectivity states. Observed failure: a browser position with Red's near-complete chain rendered NN value = 0% for Red (single forward pass) and MCTS root value = 39% for Red (after search). A human would score this position ≥95% for Red.

Training telemetry corroborates a global issue: `v_sign_agree` plateaued at ~0.67 for the final ~500 iterations despite continued game generation. That is only marginally better than random on a binary classifier.

## 2. Root-cause diagnosis

Three contributing factors, ranked by leverage:

1. **Input-tensor blindness to connectivity.** Channels 19–22 (`CHANNEL_RED_TOP_DIST`, `CHANNEL_RED_BOTTOM_DIST`, `CHANNEL_BLACK_LEFT_DIST`, `CHANNEL_BLACK_RIGHT_DIST`) are labeled "distance" but actually encode a linear row/col position ramp: `tensor[...DIST, r, c] = 1.0 - r / max_idx`. Every cell in row `r` gets the same value whether or not there is a peg connected to the goal edge. The network must infer connectivity purely from bridge-direction channels 2–17, which is poorly suited to global graph reachability over a 24×24 board with a shallow conv stack.

2. **Under-weighted value head.** `value_weight=0.25` combined with typical loss magnitudes (policy ≈ 3.2, value ≈ 0.3) gives the value head roughly 3% of the scalar objective. Value gradients are starved relative to policy gradients.

3. **Uniform value supervision by ply.** Every position in a game receives the same final-outcome label, regardless of whether the position is two plies from terminal or 150 plies from terminal. Early-game positions carry more label noise than late-game positions, but train with equal value-loss weight.

Not a primary cause: resign/adjudication (working — iter-948 sidecar shows 69% resigns, 7% adjudications, only 2% timeouts); opening regime (already tuned via Phase 2 early-override work); curriculum (already completed to 24×24 by iter ~97).

## 3. Approach: staged clean retrain (Option A)

Clean retrain from scratch. Launch one training run that is both "the validation retrain" and "the full retrain": gate evaluation at iter 150–300 determines whether the run continues to a full 1000+ iters or aborts.

**Design principle: one variable family changed at a time.** The self-play regime (replay cap, adjudication threshold, resign settings, near-corner penalty, edge-band penalty, dirichlet parameters, curriculum configuration, learning rate, L2, hidden, n_blocks) stays constant at current values. The only changes are:

1. NN input tensor: 24 → 30 channels (6 new connectivity channels)
2. `value_weight`: 0.25 → 0.5
3. New progress-weighted value loss (default ON, floor 0.25)

## 4. Phased rollout

| Phase | Name | Purpose | Stop condition |
|---|---|---|---|
| 0 | Probe suite build | Script candidate generation → user curation → commit `tests/probes/twixt_probes.json` **and** run iter-0999 baseline scoring | 50–80 curated probes committed + baseline CSV + baseline summary committed |
| 1 | Diagnostic infra | Probe runner, connectivity-aware replay diagnostics, value calibration by position type, replay-composition extensions. | Telemetry outputs verified on iter-0999 checkpoint |
| 2 | Architecture + training changes | Add 6 connectivity channels (24→30), bump `value_weight` 0.25→0.5, add progress-weighted value loss | Unit tests pass + 8×8 curriculum smoke run succeeds |
| 3 | Staged validation retrain | Launch fresh-weights training using existing self-play knobs. Full probe suite every 25 iters; forced-tier NN-only probe sample every iter. | Gate evaluable at ≥150 iters; mandatory at 300 iters |
| 4 | Gate evaluation | Apply probe / replay-value / health gates (Section 7) | PROMOTE or ABORT |
| 5 | Full retrain | Training run continues uninterrupted past the gate (no restart — Phase 3 and Phase 5 are the same run; "promotion" means "don't stop at 150–300 iters, keep going"). Target ≥1000 iters for feature parity with iter-0999. | Training stabilizes or stopped manually |

**Invariant across phases 0–2.** Iter-0999 checkpoint is untouched. Existing self-play regime stays constant: replay cap 64 / endgame keep 16, adjudication threshold 0.20, resign settings, near-corner penalty 0.60 for ply<14 with early 0.90 for ply<2, edge-band penalty 0.75 for ply<16, curriculum sizes/thresholds.

**Rollback.** If gate ABORTs at Phase 4, iter-0999 remains the production model. No code reverts needed — all changes are additive (new channels, new files, new diagnostic sections). Pre-Phase-2 sidecars render "not available" for new sections.

**Out of scope (follow-up specs).** (a) A-vs-B channel-layout ablation after first successful retrain; (b) auxiliary "moves-to-win" training head; (c) graph-NN encoder; (d) `distance_to_connect` structural reachability metric (deferred from Phase 1 — add later if simpler structural metrics prove insufficient).

## 5. Architecture — connectivity channels

### 5.1 Channel layout

Input tensor goes from `NUM_CHANNELS=24` to `30`.

```
Channel 24: red_connected_to_top         — binary per peg
Channel 25: red_connected_to_bottom      — binary per peg
Channel 26: red_connected_to_both        — binary per peg (= AND of 24,25)
Channel 27: black_connected_to_left      — binary per peg
Channel 28: black_connected_to_right     — binary per peg
Channel 29: black_connected_to_both      — binary per peg (= AND of 27,28)
```

A channel is `1.0` on cell (r, c) iff there is a peg of the corresponding color at (r, c) **and** the peg's bridge-connected component touches the named goal edge. `0.0` elsewhere (including empty cells and wrong-color pegs). Channels respect `active_size` (0-padded outside the active region, same convention as existing peg/bridge channels).

### 5.2 Computation — reuse existing connectivity semantics

**Implementation requirement:** the same connectivity graph used by `winner()` / `_get_connected_component()` in `twixt_state.py` must be the source of truth for these channels. No separate DSU. This prevents subtle drift between "what game logic thinks is connected" and "what feature code thinks is connected."

Computed in `to_tensor()`: for each player independently, group pegs into components (via existing BFS over `self.bridges`), flag each component with `touches_goal1` / `touches_goal2`, then materialize the three per-color masks from the flags.

Expected invariant: `*_connected_to_both` is all-zeros on every legal non-terminal state and non-empty on exactly the winning player's component at a terminal state.

### 5.3 Downstream impacts

- `network.py`: first conv layer `in_channels=NUM_CHANNELS` already reads the constant, so bumping `NUM_CHANNELS=30` cascades through. Hidden size / n_blocks unchanged.
- `export_onnx.py`: produces a 30-channel-input ONNX automatically.
- `server/inference.js` + browser-side tensor construction: JS must emit the 6 new channels in the exact same order and semantics as Python, respecting `active_size` zero-padding. The existing JS DSU (`assets/js/game/rollbackDSU.js`) can be reused for connectivity.
- `PositionRecord` schema unchanged by connectivity channels (tensor is always regenerated from state at load time).
- Schema version: input layout is a breaking change. Iter-0999 (24-channel) cannot be loaded into the 30-channel network. Expected under Option A.

### 5.4 Performance

Worst case ~100 pegs + ~300 bridges on a 24×24 board. Per-player BFS is negligible (<1 ms). Additive `to_tensor()` cost is immaterial. No caching needed. MPS memory not currently a bottleneck; 25% input-tensor size increase is acceptable.

## 6. Training changes

### 6.1 `value_weight`: 0.25 → 0.5

Single default change in `trainer.py::train()`. Existing `--value-weight` CLI flag still overrides. Existing value-weight warmup ramp (`curr_value_weight = 0.05 → 0.10 → target`) stays as-is; it now ramps to `0.5` over the first handful of iterations.

### 6.2 Progress-weighted value loss

Per-sample value-loss weight based on position progress through its source game. Positions closer to terminal are weighted more heavily (sharper labels).

**Formulation:**

```
progress      = clip(ply / max(game_n_moves - 1, 1), 0.0, 1.0)
sample_weight = progress_weight_floor + (1 - progress_weight_floor) * progress
value_loss    = sum(sample_weight * (values - outcomes)^2) / sum(sample_weight)
```

**Critical implementation detail:** use the normalized weighted mean (`sum(w * err²) / sum(w)`), not the unnormalized `mean(w * err²)`. This keeps loss scale stable across weight-profile changes.

**Edge cases:**

- `game_n_moves <= 1` → progress clamps to 1.0 via the `max(game_n_moves - 1, 1)` denominator.
- `progress_weight_floor = 1.0` reproduces unweighted MSE exactly (used in unit test + available as CLI escape hatch).

**Required `PositionRecord` extensions:**

```python
@dataclass
class PositionRecord:
    ...
    ply: int                            # ply at which this position occurred
    game_n_moves: Optional[int] = None  # total plies in the source game
```

`ply` and `game_n_moves` are set by `play_game()` at outcome-assignment time, same pass where `outcome` is stamped.

**CLI:** `--progress-weighted-value-loss` (default ON), `--progress-weight-floor` (default 0.25).

### 6.3 Hyperparameters explicitly NOT changing

For unambiguous interpretation of the validation gate, the following stay at current values:

| Param | Value | Source |
|---|---|---|
| `c_puct` | 1.5 | MCTSConfig default |
| `dirichlet_alpha` / `dirichlet_eps` | 0.3 / 0.25 | "" |
| `opening_dirichlet_*` / `opening_noise_ply` | current | "" |
| `root_near_corner_penalty` / `_ply` | 0.60 / 14 | CLI |
| `root_near_corner_penalty_early` / `_plies` | 0.90 / 2 | CLI |
| `root_edge_band_penalty` / `_ply` / `_width` | 0.75 / 16 / 2 | CLI |
| `max_positions_per_game` / `endgame_keep_positions` | 64 / 16 | CLI |
| `adjudicate_*` | current | CLI |
| `resign_*` | current | CLI |
| learning rate, L2, hidden, n_blocks, eval_batch_size | current | CLI/defaults |
| curriculum sizes / window / thresholds | current | CLI |

Phase 3 invocation uses the same `train.py` CLI as today, adding only `--value-weight 0.5` (or letting the new default ride) and `--progress-weighted-value-loss`. Checkpoint dir: `checkpoints/alphazero-v2-staged/`. On promotion: `checkpoints/alphazero-v2/`. Iter-0999 stays at `checkpoints/alphazero-fresh/`.

## 7. Validation gate

### 7.1 Pass/fail checklist

**PROMOTE to full retrain iff all three groups pass after ≥150 iters (and by 300 iters).**

**Probe-suite gate:**

- [ ] `forced` tier: ≥95% sign-correct
- [ ] `forced` tier: median `|nn_value|` ≥ 0.75
- [ ] `strong_advantage` tier: ≥80% sign-correct
- [ ] `strong_advantage` tier: median `|nn_value|` ≥ 0.45
- [ ] Overall: ≥85% sign-correct
- [ ] Improvement vs iter-0999 baseline: ≥+10 pp overall **OR** ≥+15 pp on forced tier

**Replay-value gate:**

- [ ] `v_sign_agree` sustained ≥ 0.75 over final 20% of staged iters (not a single lucky spike)
- [ ] Final `v_sign_agree` ≥ baseline + 0.05 (baseline ≈ 0.67 from iter-0999, so target ≥ 0.72 at minimum)

**Health gate (any failure → ABORT regardless of above):**

- [ ] `avg_plies` ≤ 220
- [ ] `resign_rate` ≥ 45%
- [ ] `timeout_rate` ≤ 3%
- [ ] `v_pred_std` > 0.30
- [ ] `v_frac_sat` < 0.05
- [ ] No NaN; no collapse to constant predictions

### 7.2 Cadence & duration

- Full probe suite: every 25 iters
- `forced`-tier NN-only sample (cheap, 10–20 forward passes): every iter
- Gate evaluable at ≥150 iters, mandatory by 300 iters
- If ≥150 iters reached but gate fails: continue to 300, re-evaluate. If still fails at 300: halt, report, spec update required before next attempt.

## 8. Probe suite (Phase 0)

### 8.1 Storage

```
tests/probes/
  twixt_probes.json     # curated permanent suite, versioned in git
  README.md             # category definitions, labeling rules, add-probe process
  candidates.json       # (gitignored) intermediate pre-curation output
```

### 8.2 Schema

Each probe entry:

```json
{
  "id": "nw-red-001",
  "category": "near_win_red",
  "confidence": "forced",
  "side_to_move": "red",
  "expected_value_sign": 1,
  "expected_value_min": 0.75,
  "expected_value_max": null,
  "active_size": 24,
  "ply": 42,
  "move_history": [[0, 3], [23, 20], [1, 5], ...],
  "source_game": "scripts/GPU/logs/games/iter_0820_game_014.json",
  "source_ply": 42,
  "peg_counts": {"red": 22, "black": 19},
  "mirror_of": null,
  "evaluation_modes": ["nn_only", "mcts"],
  "note": "Red has single chain reaching row 0 to row 21, one bridge from bottom"
}
```

**Canonical state** = `move_history` replayed from game start against a fresh `TwixtState`. Auxiliary metadata (`active_size`, `ply`, `source_game`, `source_ply`, `peg_counts`) is stored for inspection/debug only; if it disagrees with the replayed state, the replayed state wins.

### 8.3 Categories

| Category | Min | Max |
|---|---:|---:|
| `near_win_red` | 10 | 15 |
| `near_win_black` | 10 | 15 |
| `blocked_or_trap` | 8 | 10 |
| `false_positive_connectivity` | 5 | 10 |
| `dense_but_disconnected` | 8 | 10 |
| `central_win` | 8 | 10 |
| `edge_corner_legitimate` | 5 | 10 |
| `symmetric_sanity` | 5 | 10 |
| **Total** | **59** | **90** |

Target curated size: 50–80 (some categories may end up at category min; `unclear_do_not_use` entries are discarded).

### 8.4 Confidence tiers

- `forced` — unambiguously winning/losing (1–2 moves from terminal or obvious structural lock)
- `strong_advantage` — clearly better but not forced
- `unclear_do_not_use` — reviewer couldn't decide; discarded from final suite

**Labeling rule (README):** if reviewers disagree on a candidate's tier, default to `unclear_do_not_use`. Do not force resolution.

### 8.5 Curation workflow

1. **Sampler** (`scripts/build_probe_candidates.py`) reads `scripts/GPU/logs/games/*.json`, applies per-category heuristic rules (using the Section 5 connectivity routine), emits `tests/probes/candidates.json` with ~150–250 candidates grouped by category, each with annotated heuristic reason.
2. **User review pass** (~30–60 min): assigns confidence tiers, edits notes, discards `unclear_do_not_use` entries, caps per-category counts.
3. **Final commit**: `tests/probes/twixt_probes.json` + `README.md`.
4. **Baseline scoring**: run probe evaluator against iter-0999. Commit `checkpoints/alphazero-fresh/probe_eval_iter_0999_baseline.csv` and `_baseline.json` to the repo.

### 8.6 Evaluator

New tool: `scripts/GPU/alphazero/probe_eval.py`

```
python -m scripts.GPU.alphazero.probe_eval \
    --weights checkpoints/alphazero-fresh/model_iter_0999.safetensors \
    --probes tests/probes/twixt_probes.json \
    --sims 200 \
    --out checkpoints/alphazero-fresh/probe_eval_iter_0999_baseline.csv
```

**Dual-format contract:** the runner must auto-detect `NUM_CHANNELS` from the checkpoint (or checkpoint metadata) and instantiate the matching network. A single runner supports both 24-channel (iter-0999 and earlier) and 30-channel (retrain onward) checkpoints. This is a hard requirement — baseline scoring and staged-retrain scoring must use the same tool.

Per probe, records:

| Field | Definition |
|---|---|
| `nn_value` | Raw forward-pass value from side-to-move |
| `mcts_root_value` | Backed-up root Q after `--sims` simulations |
| `mcts_top_move` | Argmax of visit counts |
| `mcts_top_share` | Top move's fraction of total visits |
| `sign_correct_nn` | `sign(nn_value) == expected_value_sign` |
| `sign_correct_mcts` | `sign(mcts_root_value) == expected_value_sign` |
| `nn_magnitude` | `|nn_value|` |
| `magnitude_in_band` | `expected_value_min ≤ |nn_value|` (iff set); and `|nn_value| ≤ expected_value_max` (iff set) |
| `search_corrected` | NN wrong but MCTS right |
| `both_wrong` | NN and MCTS both wrong |

Also writes per-run aggregate: `probe_eval_summary_iter_NNNN.json` with per-tier sign-correct rates, median magnitudes, failures grouped by category. This summary is what the gate reads.

### 8.7 Training-loop integration

- **Every 25 iters**: full suite via MCTS + NN; emits per-iter CSV + summary JSON.
- **Every iter**: `forced`-tier NN-only sample (no MCTS); emits a slim per-iter row.
- **Iter-0999 baseline (Phase 0)**: run once, commit CSV + JSON to repo as gate reference.

## 9. Diagnostic infrastructure (Phase 1)

### 9.1 Connectivity-aware replay diagnostics (new)

Computed from sampled replay positions via the same `_get_connected_component` used in Section 5.2.

Per position:

| Field | Type |
|---|---|
| `red_has_top_component` | bool |
| `red_has_bottom_component` | bool |
| `black_has_left_component` | bool |
| `black_has_right_component` | bool |
| `red_largest_component_size` | int |
| `black_largest_component_size` | int |
| `red_n_goal_touching_components` | {0,1,2} |
| `black_n_goal_touching_components` | {0,1,2} |

Aggregated by ply bucket and by final game outcome (red-win / black-win / draw).

**Artifacts:**

- `connectivity_by_ply_<suffix>.csv` — one row per (ply_bucket, color, outcome)
- `connectivity_summary_<suffix>.json` — structural rollup, correlation with NN value / MCTS value / actual winner
- `report.txt` section: `Connectivity Diagnostics`

**Module:** new file `scripts/GPU/alphazero/connectivity_diagnostics.py`. Called once per analyzer run. Always enabled (cheap).

### 9.2 Value calibration by position type (new)

Re-bucketed value-head sanity stats, covering the blind-spot categories.

**Buckets:**

| Bucket | Definition |
|---|---|
| `red_winning_structure` | Red has a goal-touching component AND (largest red component size ≥ 8 pegs OR has two red goal-touching components). Threshold configurable via `--winning-structure-min-size`, default 8. |
| `black_winning_structure` | Symmetric |
| `balanced_no_winning_structure` | Neither side has a goal-touching component matching the above |
| `early_game` | position_ply < 20% × game_n_moves |
| `mid_game` | 20% ≤ position_ply < 70% |
| `late_game` | position_ply ≥ 70% |
| `short_source_game` | game_n_moves ≤ 80 |
| `long_source_game` | game_n_moves > 200 |

**Per bucket:**

| Metric | Definition |
|---|---|
| `n` | position count |
| `sign_agree` | `sign(nn_value) == sign(outcome)` rate |
| `mse` | `mean((nn_value - outcome)^2)` |
| `pred_mean` | mean `nn_value` |
| `outcome_mean` | mean `outcome` |
| `calibration_bins` | 5 reliability-diagram bins (configurable via `--calibration-bins`) |

**Artifacts:**

- `value_calibration_<suffix>.csv` — one row per bucket (flat stats)
- `value_calibration_bins_<suffix>.csv` — per-bucket calibration-bin detail
- `report.txt` section: `Value Head Calibration by Position Type`

**Module:** new file `scripts/GPU/alphazero/value_calibration.py`. Requires loading a checkpoint and scoring positions — not free. Gated behind `--calibrate` flag; uses latest checkpoint in checkpoint dir unless `--calibrate-weights <path>` given. Sample size via `--calibration-sample N` (default 1000).

### 9.3 Search-vs-NN disagreement

Already captured by probe evaluator (`search_corrected`, `both_wrong` flags). Aggregate in probe summary: "fraction of probes where search corrected NN" vs "fraction where both wrong." No separate artifact.

### 9.4 Replay composition extensions

Extend `replay_cap_by_iter.csv` (Phase 4 stayover) with new columns:

- `positions_by_termination_{win,resign,adjudicated,timeout}`
- `positions_in_short_games` / `positions_in_long_games` (using 80-ply and 200-ply cutoffs matching 9.2)

Requires the trainer to track termination type per contributed position — already available in `GameRecord`, just needs aggregation through the replay-cap pipeline (~20 lines in `trainer.py`).

### 9.5 Analyzer CLI additions

```
--probes <path>                    # emit probe-eval section from sidecar probe data
--calibrate                        # run 9.2 value-calibration-by-position-type
--calibrate-weights <path>         # explicit weights for --calibrate
--calibration-sample N             # default 1000
--calibration-bins N               # default 5
--winning-structure-min-size N     # default 8; threshold for 9.2 buckets
--no-connectivity                  # skip 9.1 (speed)
```

Defaults: probe-eval + connectivity + replay-composition always on. Calibration requires explicit `--calibrate`.

### 9.6 Backward compatibility

All new sections use the established pattern: pre-Phase-1 runs render "(not available)" lines in `report.txt`, produce empty dicts in `summary.json`, emit no new CSV if no data. Consistent with how `root_child_diagnostics` and `early_override_summary` already degrade.

## 10. Testing

### 10.1 Connectivity channels

1. **Parity with `winner()`**: ~100 positions across red-win / black-win / non-terminal — `*_connected_to_both` non-empty iff that color won.
2. **Isolated peg**: single red peg, zero bridges → all 3 red channels zero (unless peg is literally on row 0, in which case `red_connected_to_top` = 1 at that cell).
3. **Single chain**: red chain row 0 → row `active-1` → every peg in chain has all three red channels = 1; pegs not in chain = 0.
4. **Active-size respect**: for `active_size=8`, channels 24–29 are zero outside the 8×8 region.
5. **JS/Python parity**: same state yields same 30-channel tensor bit-for-bit (extend `run_encoding_parity.py`).
6. **Mirrored-state parity** (NEW): mirror a state left-right, swap red/black semantics where the mirror implies it; verify connectivity-channel semantics transform correctly.

### 10.2 Progress-weighted value loss

7. **Floor=1.0 reproduces unweighted MSE** exactly.
8. **Scale invariance**: scaling all per-sample weights by 10× does not change the loss value (validates normalized weighted mean).
9. **Edge case** `game_n_moves ≤ 1`: denominator clamp yields `progress = 1.0`.

### 10.3 Probe suite

10. **Schema validation**: every entry in `tests/probes/twixt_probes.json` passes JSON-schema check — required fields, valid category, valid confidence tier, `active_size` in allowed range, optional fields have correct types.
11. **Replay correctness**: replaying `move_history` produces the recorded `active_size`, `ply`, `peg_counts`; no illegal moves.

### 10.4 Analyzer end-to-end

12. **Synthetic smoke**: minimal synthetic dataset containing Phase 2 fields + Phase 1 probe data; verify all new CSVs, new `summary.json` keys, new `report.txt` sections are produced without error. Reuses existing E2E smoke pattern.

## 11. File layout

### 11.1 New files

```
docs/superpowers/specs/2026-04-19-connectivity-retrain-design.md   (this doc)
scripts/build_probe_candidates.py
scripts/GPU/alphazero/probe_eval.py
scripts/GPU/alphazero/connectivity_diagnostics.py
scripts/GPU/alphazero/value_calibration.py
tests/probes/twixt_probes.json
tests/probes/README.md
tests/test_connectivity_channels.py
tests/test_progress_weighted_loss.py
tests/test_probe_suite_schema.py
tests/test_analyzer_phase2_smoke.py
```

### 11.2 Modified files

```
scripts/GPU/alphazero/
  game/twixt_state.py                # NUM_CHANNELS 24→30, to_tensor() extension
  self_play.py                       # PositionRecord.ply + game_n_moves
  trainer.py                         # value_weight default, progress-weighted loss,
                                     # termination-type aggregation in replay-cap
  train.py                           # --progress-weighted-value-loss, --progress-weight-floor
scripts/
  twixt_replay_analyzer.py           # wire in connectivity + calibration + probe sections
assets/js/
  <tensor construction module>       # 6 new channel construction (JS parity)
tests/
  run_encoding_parity.py             # extend for 30 channels
  test_tensor_repr.py                # extend for connectivity channels
```

### 11.3 Generated (gitignored)

```
tests/probes/candidates.json                                 # pre-curation intermediate
checkpoints/alphazero-v2-staged/                             # staged retrain output
checkpoints/alphazero-v2/                                    # promoted retrain output
```

### 11.4 Committed baseline artifacts

```
checkpoints/alphazero-fresh/probe_eval_iter_0999_baseline.csv
checkpoints/alphazero-fresh/probe_eval_iter_0999_baseline.json
```

## 12. Rollout risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Probe suite labels biased | Medium | Reviewer-disagreement rule defaults to `unclear_do_not_use`. Suite versioned in repo. |
| 30-channel tensor slows training unacceptably | Low | MPS memory not currently a bottleneck. Measure on smoke run. Revert to B/C channel layouts if severe. |
| New features hurt early training | Medium | Validation gate catches this. Value-weight warmup still active. 8×8 smoke run before 24×24 commit. |
| Progress-weighted loss hurts value head on short games | Low-Medium | Floor=0.25 conservative. `--progress-weight-floor 1.0` CLI escape hatch reproduces old behavior. |
| JS/Python channel drift | Medium-High | Parity tests are mandatory. Added to CI-style smoke. |
| Staged retrain passes gate but full retrain stalls | Low | Can stop at any iter; iter-0999 stays as rollback. |
| Self-play regime change sneaks in alongside architecture change | Low | Section 6.3 explicit "NOT changing" table. Restated here: 24×24 self-play regime is fixed for this retrain. Any regime change is a separate, future spec. |
| Probe runner fails on one of the two checkpoint formats | High-if-not-tested | Dual-format contract (Section 8.6) is a hard requirement with explicit tests for both format detections. |

## 13. Success criteria summary

The retrain is successful iff all of:

1. Phase 0 probe suite committed with 50–80 entries covering 8 categories
2. Phase 0 iter-0999 baseline scored and committed
3. Phases 1–2 deliverables pass all tests (Section 10)
4. Phase 3 staged retrain reaches ≥150 iters without health-guard failures
5. Phase 4 gate evaluation returns PROMOTE
6. Phase 5 full retrain reaches ≥1000 iters with value-head signal sustained

The retrain is a failure iff Phase 4 returns ABORT and analysis does not yield a clear next-spec direction. That triggers a re-scope: possibly new channel design (Option B/C), possibly a different training change (e.g. auxiliary moves-to-win head), possibly probe suite revision.

## 14. Next step

Invoke `superpowers:writing-plans` to produce the phased implementation plan from this spec.
