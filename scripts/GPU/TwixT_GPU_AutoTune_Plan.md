# TwixT GPU AutoTune Rebuild Plan (Python + MLX on Apple Silicon)

/scripts/GPU/
├── __init__.py
├── __main__.py
├── cli.py                    # Single entry point
├── config/
│   ├── knobs.py              # Extensible knob registry + core bands + hash fields
│   └── search_config.py      # Load/save search.json (preserve unknown keys)
├── game/
│   ├── board.py              # Peg placement + legal moves (stub -> full rules)
│   ├── bridge.py             # Bridge creation + crossing detection (stub -> full)
│   ├── state.py              # GameState dataclass
│   └── rules.py              # Apply moves + win detection (stub -> BFS graph)
├── ai/
│   ├── heuristics.py         # Feature extraction (stub -> 28+ features)
│   ├── search.py             # Minimax alpha-beta (stub -> full)
│   ├── move_ordering.py      # Top-N ordering (stub -> MLX-batched scoring)
│   └── value_model.py        # MLX inference (stub -> full)
├── tuning/
│   ├── hasher.py             # Stable config hashing (ordered fields)
│   ├── state.py              # Unified durable state (tuning_state.json)
│   ├── sweep.py              # Bucket generation (v1 implemented; expand to full buckets)
│   ├── validation.py         # Combine depths + gating helpers
│   ├── ridge.py              # MLX ridge regression + predicted-bias gate (CPU fallback)
│   └── loop.py               # Orchestration + ranking + validation queue
├── selfplay/
│   ├── engine.py             # Game simulator interface (placeholder today)
│   ├── parallel.py           # Batched runner + JSONL logging + replay output
│   └── results.py            # Aggregation + bias
└── replay/
    ├── recorder.py           # Writes game-*.json per run
    ├── format.py             # JSON schema + dataclasses
    └── viewer.py             # Interactive CLI replay (ASCII board)

## 1) Goal
Rebuild the TwixT **auto-tuning loop** so it runs end-to-end from a **single CLI** under `scripts/GPU/`, uses **MLX (Metal) on M3** for the heavy numerical work, and produces **replayable games** (self-play + validation) as first-class artifacts.

**Outputs**
- Updated `assets/js/ai/search.json` with the winning config.
- Sweep logs + validation logs (JSONL).
- Durable state file (registry + streaks + pending queue + model state).
- Replay game records (JSON) suitable for CLI playback and (optionally) JS visualization.

## 2) Non‑negotiables (lessons learned to preserve)
### Hash stability
- Compute config hashes from a **fixed, ordered knob list** (never from dict iteration order).
- Store the ordered knob list + schema version in code; add unit tests so hash cannot silently change.

### Buckets & sampling
- Preserve core buckets: `soft-best`, `trend`, `best`, `niche`.
- Sensitive knobs (span/coverage) must clamp to **core bands** in the core buckets to prevent d2 blowups.
- Each cycle’s `explore` must reserve slots for **fixed probes** (edges ±5, blackSpan ±0.05, plus coverage probes).
- Category weights must guarantee those probes always appear.

### Validation gating
- Require multiple **60/60** passes (streak) + a **macro (1200/1200)** gate before declaring success.
- Use predicted-bias gating when correlation is trustworthy; if the gate becomes too strict, fall back to **core-clamped** candidates rather than stalling.

### Correlation & data hygiene
- Rolling window with **decay**.
- **Up-weight probes**, down-weight/ignore low-info samples (draw-heavy, near-zero bias).
- Registry statuses are durable and case-insensitive: `UNTESTED/SHORTLIST/VALIDATING/STABLE/RETIRED`.
- Never re-queue `RETIRED` or `STABLE`. Preserve streaks; reset on failure; retire chronic failures.

## 3) Design decision to maximize speed (and correctness)
The fastest path on Apple Silicon is to:
- Keep **discrete game rules** (legal moves, crossings, win detection) correct and deterministic in Python.
- Push **numerical evaluation** to MLX: batch heuristic feature extraction, move scoring, value model inference, ridge regression.

This avoids trying to “GPU-ize” inherently branchy geometry/crossing logic too early while still capturing most of the speedup.

## 4) Proposed unified architecture
### A. Core packages (under `/scripts/GPU/`)
- `config/` — knob registry + search.json load/save.
- `game/` — rules engine (board, bridges, crossing detection, win detection).
- `ai/` — heuristics + search + move ordering + value model (MLX where applicable).
- `selfplay/` — single/parallel game runners + aggregations.
- `tuning/` — hashing, sweep gen, ranking, validation gating, ridge regression, main loop.
- `replay/` — recorder + JSON schema + viewer(s).

### B. Unified CLI (single entry point)
`python -m scripts.GPU.cli <subcommand>`

Recommended subcommands:
- `init` — create/verify logs/state folders; optionally seed RNG.
- `suggest` — generate a sweep (bucket quotas, probes, hashing).
- `sweep` — run 10/10 (or configurable) games at depths (e.g., 2 & 3) for all suggested configs; write sweep results.
- `rank` — rank by weighted parity loss across depths; shortlist.
- `validate` — run 60/60 validations for queued hashes; update streaks; apply macro gate.
- `loop` — end-to-end orchestration until “value-ready” config found.
- `replay <game.json>` — interactive playback of a recorded game.
- `export --hash <h>` — write winning knobs to `search.json`.

## 5) Observability (so you can “check in on progress”)
### Always-on progress telemetry
- Emit a concise progress line every N games:
  - cycle, config hash, depth, game count, win/loss/draw, current parity score
- Write a machine-readable `logs/progress.jsonl` stream (optional).

### Replay and review
- Record **at least**: moves, player, chosen move score, optional feature deltas, seed, depth, config hash.
- Keep replay artifacts organized by hash:
  - `logs/games/<hash>/game-<uuid>.json`
- Provide:
  - CLI viewer (ASCII) now
  - Optional JS/HTML viewer later for rich visualization (recommended, since you already have JS UI for humans)

## 6) Data formats (durable + append-friendly)
Use **JSONL** for event-like streams (sweeps/validations/progress) so partial runs don’t corrupt files.

### State file (single source of truth)
`logs/tuning_state.json`
- hash_registry: status, streak, validation_count, last_seen, etc.
- pending_validation queue
- knob_stats & optional freeze list
- best score + cycles_since_improvement
- correlation model state (rolling window metadata)

### Sweep results (append-only)
`logs/sweep-results.jsonl`
- includes config hash, knob values, depths, results, parity metrics, tags (probe, bucket)

### Validation results (append-only)
`logs/validation-results.jsonl`
- includes config hash, run size (60/60), pass/fail, updated streak, macro status

### Replay record (per game)
`logs/games/<hash>/game-<uuid>.json`
- id, timestamp, seed, config hash, depth, winner
- move list: turn, player, row/col, optional bridges created, move_score, optional heuristic snapshot

## 7) Phased implementation (with “correctness first” guardrails)
### Phase 1 — Game engine foundation
- Port board, bridge creation, crossing detection, win detection, legal move generation.
- Add unit tests + deterministic seeds.
- Add a “golden position” suite comparing Python vs JS on:
  - legal moves count
  - crossing legality outcomes
  - win detection
  - (optional) move chosen by a fixed shallow heuristic

### Phase 2 — Heuristics port (vectorizable)
- Port the 28+ features.
- Create a batched MLX path:
  - Given a position + list of candidate moves, compute feature matrix and scores in one GPU call.
- Keep a CPU fallback (numpy) for dev environments without MLX.

### Phase 3 — Search + value model
- Alpha-beta minimax depths 2–4.
- Move ordering: score all moves, keep top-N (N configurable; depends on board state).
- Value model: MLX logistic regression or small MLP (whatever matches your current “value-model.json” format).

### Phase 4 — Self-play (batched)
- Parallel runner that batches:
  - many games
  - many positions per ply (for GPU evaluation)
- Deterministic seeds per (hash, depth, game_index) for reproducibility.

### Phase 5 — Tuning logic (port from autoTune)
- Knob specs + core bands.
- Stable hashing.
- Bucket generation + fixed probes quotas.
- Ranking by parity loss (weighted depths).
- Predicted bias gate from ridge/correlation model.
- Validation streak + macro gate; retire chronic failures.

### Phase 6 — Replay system
- Recorder integrated into engine.
- CLI replay viewer (step, back, jump to turn N).
- Optional: export to a JS-friendly format so the existing UI can replay the same games.

### Phase 7 — CLI polish + optimization pass
- Improve usability (default paths, “resume from state”, crash-safe).
- Profile bottlenecks.
- Decide if any **Swift/Metal** hot path is needed (likely only if rules/crossing becomes dominant).

## 8) Adding knobs safely (extensibility)
- `config/knobs.py` is the single registry:
  - name, discrete values (or range+step), default, category, sensitivity (core-banded?), probe strategy.
- Hashing uses the ordered knob list; adding a knob should:
  1) require explicitly placing it in the ordered list
  2) bump a `HASH_SCHEMA_VERSION`
  3) add a migration note (old state files remain readable)
- Sweep generator automatically picks it up via registry + category weights.

## 9) Performance targets (pragmatic)
- Primary speedups come from **batch evaluation**:
  - move scoring + heuristic features
  - value inference
  - ridge regression
- Expect large gains even if game legality rules stay on CPU.
- Early benchmark target: 10/10 sweep at depths (2,3) **< 2 minutes** on M3 (real number will depend on move branching & top-N).

## 10) Immediate next actions
1. Lock hash schema: ordered knob list + unit test.
2. Implement replay schema + recorder (so every run is debuggable).
3. Port rules engine with correctness tests (Python vs JS).
4. Replace placeholders in self-play with the real engine (even CPU-only initially).
5. Only then: MLX acceleration for heuristics/value/ridge.

---
## Appendix: Recommended extra improvements beyond the current plan
- **Seed discipline**: derive per-game RNG seed from `(config_hash, depth, game_index)` so any game is reproducible.
- **Crash resilience**: write sweep/validation results incrementally (JSONL), and keep `tuning_state.json` updated with atomic writes.
- **Dashboard (optional)**: a tiny local HTML page that tails JSONL logs and shows cycle progress + top configs.
- **Compatibility adapter (temporary)**: allow calling existing JS sim for correctness checks early in the port.
