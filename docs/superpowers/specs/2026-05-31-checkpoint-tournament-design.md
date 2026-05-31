# Checkpoint Tournament System — Design

**Date:** 2026-05-31
**Status:** Approved design, pending implementation
**Author:** brainstormed with Claude

## Purpose

Self-play diagnostics tell us whether training is *stable* (closeouts sane,
state-caps rare, openings diverse). They do **not** tell us whether
`model_iter_0419` is actually *stronger* than `model_iter_0379`.

This system answers that directly: it freezes two (or more) checkpoints and
plays them against each other under controlled, exploration-free conditions to
estimate relative strength. The motivating use is **plateau detection** — so we
keep training when it's still gaining and stop injecting noise once it isn't.

It is **pure evaluation**: no backprop, no training-noise, no checkpoint writes.

## Non-goals (v1)

- **No adjudication.** A game that reaches the move cap with no connection
  winner is scored as a draw (`state_cap`), reported in its own bucket, scored
  0.5. (Rationale below — two-model adjudication is biased; deferred.)
- **No resign.** Games end on a real connection win or a state-cap. Resign only
  saves compute and would add a confounder. (Deferred.)
- **No SPRT / sequential stopping.** Fixed game counts per pairing in v1.
- **No reuse of the training inference-server / `play_game` path.** See
  "Why a standalone loop" below.
- **No crash-resumable JSONL shards.** Results flow through a queue; the parent
  writes final JSON/CSV. (Deferred to v2 — only needed for 10k+ game or long
  unattended runs.)

## Key design decisions (and why)

### 1. Standalone eval game loop — do NOT refactor `self_play.play_game`

The termination rules live inside `self_play.play_game()`, which is built
around a **single** MCTS + tree-reuse for both sides and is entangled with
training diagnostics (goal-completion tracking, mirror augmentation, recovery
re-targeting, conversion loss). A two-model match cannot reuse that loop
cleanly.

Measurement of the "rules" that aren't already shared:

| Rule | Location | Real logic | Notes |
|------|----------|-----------|-------|
| winner | `TwixtState.winner()` | already a method | **already shared — zero risk** |
| state-cap / board-full | `self_play.py:1167-1276` | ~15 lines | `winner is None and ply>=max_moves` → state_cap |
| adjudicate | `self_play.py:1182-1260` | ~25 lines core | rest is `ADJ_DEBUG` + diagnostic-field wiring |
| resign | `self_play.py:929-985` | ~30 lines | optional; not used in v1 |

The one rule genuinely worth sharing — `state.winner()` — is **already**
shared. State-cap is trivially identical. Adjudication is the only nontrivial
piece, and in a two-model game it *must* diverge from self-play anyway (in
self-play one model adjudicates; in a tournament, the side-to-move's model is
biased toward declaring itself the winner). So extracting it would share the
~25 lines we'd have to change regardless.

**Decision:** write a clean standalone two-evaluator loop. **Zero changes** to
`self_play.py`, `mcts.py`, or any training code.

### 2. No adjudication in v1 — state-caps are draws

In self-play, the single model evaluates a capped position unambiguously. In a
tournament, whichever model adjudicates is biased toward itself, and balanced
colors do not wash this out (the bias tracks model optimism, not strength).

A fair version exists (evaluate with *both* models; declare a winner only if
both agree on sign and clear the magnitude threshold; else draw), but it is
deferred. v1 counts every capped game as a `state_cap` draw, scored 0.5,
reported in its own bucket. If state-caps turn out to be a meaningful fraction
of tournament games, revisit with agreement-based adjudication.

### 3. Controlled variation: opening temperature, then sharp

Fully deterministic play (pure argmax, no noise) makes every game with the same
color assignment **byte-identical** — 400 games would collapse to 2 distinct
games. A tournament needs a controlled source of variation.

We reuse the standard AlphaZero approach already present in `MCTSConfig`:
`temp_high=1.0` for the first `temp_threshold_ply` (~20) plies → diverse
openings; then `temp_low=0.1` → near-deterministic sharp play. Per-game seed
drives the opening divergence into distinct lines. This is **separate** from
Dirichlet exploration noise, which stays **off** (`add_noise=False`). Mirror
augmentation **off**. Exposed via `--opening-temp-plies / --temp-high
/ --temp-low` for tuning.

### 4. Controlled colors: red always moves first; balance which model is red

`start_player` is fixed to `"red"` every game (red has the fixed first-move
advantage). We balance *which model* plays red by `game_idx` parity. This
separates the color/first-move advantage (cancels across a balanced match) from
model strength. By-color stats are still tracked and a `color_bias` sanity
metric is reported.

### 5. One flat task queue, one global worker pool — never nested pools

Two levels of parallelism exist: across games within a pairing, and across
pairings in a tournament. Parallelizing both with separate pools would give
`W_tourney × W_match` processes contending for one Metal GPU and a fixed core
count → oversubscription, which on Metal turns "parallel" into slower and less
stable.

**Decision:** unify both scripts on a single work-queue primitive. The match
script builds one pairing's tasks; the tournament builds *all* pairings into
one flat list; both call the same `run_game_tasks` with the same pool. This
mirrors the trainer's dynamic-counter worker pool
(`trainer.py:1553-1638`):

- `mp.get_context("spawn")` — **mandatory on macOS**; Metal/MLX is not
  fork-safe.
- **Dynamic scheduling** via a shared atomic counter (`ctx.Value("i", 0)`) so
  workers *pull* the next task — natural load-balancing, no per-pairing barrier.
- Explicit `WorkerDone` signals (not `Queue.empty()`).

Properties: one parallelism knob (`--workers`); dynamic work-stealing across all
games; no per-pairing barrier; per-worker checkpoint cache; reproducible seeds
independent of worker assignment; same engine for match and tournament.

## Architecture

Three new files. **Zero changes** to training code.

```
scripts/GPU/alphazero/
  eval_runner.py                  # shared flat task queue + global worker pool
  eval_elo.py                     # pure stats: score, Elo, draw-aware CI, verdict
  eval_checkpoint_match.py        # builds one pairing's tasks, calls run_game_tasks
  eval_checkpoint_tournament.py   # builds all pairings' tasks, calls run_game_tasks,
                                  #   groups results by pairing_id
```

Reused as-is (imported, not copied):

- `probe_eval.load_network_for_scoring(path)` → auto-detects 24/30-channel.
- `LocalGPUEvaluator(net)` → inference wrapper (one-liner, per `probe_eval`).
- `MCTS(evaluator, cfg, rng)`, `mcts.search(state, add_noise=False)`,
  `mcts.select_move(counts, ply)`.
- `TwixtState` (`apply_move`, `is_terminal`, `winner`, `legal_moves`, `to_move`).
- `MCTSConfig` for sims / batch / stall-flush / temperature knobs.

### `eval_runner.py`

```python
@dataclass(frozen=True)
class EvalGameTask:
    task_id: int            # globally unique across the whole run
    pairing_id: str         # e.g. "0419_vs_0379"
    game_idx: int           # local within the pairing
    red_checkpoint: str
    black_checkpoint: str
    seed: int

@dataclass
class EvalGameResult:
    task_id: int
    pairing_id: str
    game_idx: int
    red_checkpoint: str
    black_checkpoint: str
    winner: str | None              # "red" | "black" | None
    winner_checkpoint: str | None
    reason: str                     # "win" | "state_cap" | "board_full"
    n_moves: int
    red_score: float                # 1.0 / 0.0 / 0.5
    black_score: float              # 1.0 - red_score

@dataclass(frozen=True)
class EvalConfig:                   # pickle-safe; passed to workers under spawn
    board_size: int
    mcts_sims: int
    mcts_eval_batch_size: int
    mcts_stall_flush_sims: int
    opening_temp_plies: int
    temp_high: float
    temp_low: float
    max_moves: int

def run_game_tasks(tasks: list[EvalGameTask], workers: int,
                   config: EvalConfig) -> list[EvalGameResult]:
    """Execute tasks and return results sorted by (pairing_id, game_idx).

    workers == 1: run sequentially in-process through the same per-game
                  function path (no spawn).
    workers  > 1: mp.get_context("spawn"); shared atomic next-task counter;
                  workers pull tasks until drained; results via queue;
                  explicit WorkerDone; parent joins with timeout.
    """
```

Worker per-process checkpoint cache:

```python
model_cache: dict[str, Evaluator] = {}   # ckpt_path -> LocalGPUEvaluator
# A worker that pulls 0419/0379 then 0419/0339 loads 0419 once, reuses it.
# ~7.5MB/model; caching all of a 4-checkpoint tournament is ~30MB.
```

Per-game function (pure, deterministic given seed):

```python
def play_eval_game(red_eval, black_eval, config, seed) -> (winner, reason, n_moves):
    rng = random.Random(seed)                       # shared by both MCTS
    mcts_red   = MCTS(red_eval,   cfg_from(config), rng)
    mcts_black = MCTS(black_eval, cfg_from(config), rng)
    state = TwixtState(active_size=config.board_size,
                       to_move="red", max_plies_limit=config.max_moves)
    ply = 0
    while not state.is_terminal() and ply < config.max_moves:
        mcts = mcts_red if state.to_move == "red" else mcts_black
        counts, _ = mcts.search(state, add_noise=False)   # fresh root each ply
        move = mcts.select_move(counts, ply)              # opening temp -> sharp
        state = state.apply_move(move); ply += 1
    winner = state.winner() if state.is_terminal() else None
    if winner is not None:                 reason = "win"
    elif not state.legal_moves():          reason = "board_full"   # rare draw
    else:                                  reason = "state_cap"
    return winner, reason, ply
```

Note: uses `mcts.search` (fresh root per ply), **not**
`search_from_root`/`advance_root` — tree reuse assumes one model owns the tree,
which breaks across two models. Fresh-search-per-ply is correct; the only cost
is no subtree carryover (acceptable for eval).

### Determinism rule

`seed` is **task-derived, never worker-derived**:

```
seed = base_seed + pairing_index * 1_000_000 + game_idx
```

Invariant: *same tournament schedule + same `base_seed` ⇒ same games,
regardless of `--workers` or task scheduling order.* This is a required test.

### Task construction

Match (one pairing A vs B):

```
pairing_id = "<A>_vs_<B>"
game_idx even -> red=A, black=B
game_idx odd  -> red=B, black=A      # balanced colors
```

Tournament: every pairing built into one flat list; `task_id` globally unique;
`pairing_id` identifies the pair; `game_idx` local within the pairing. Results
grouped by `pairing_id` afterward.

### Results handling

Each worker puts a compact `EvalGameResult` on a results queue (one small dict
per game, unlike training's heavy position streams). Parent collects until all
`WorkerDone` received, then **sorts by `(pairing_id, game_idx)` before writing**
so output is deterministic even though execution order is not.

## Statistics (`eval_elo.py` — pure, no game/MLX deps)

Scoring (decisive win = 1, state-cap and board-full = 0.5):

- `score_A = wins_A + 0.5 * (state_caps + board_full)`,
  `score_rate_A = score_A / games`
- **Elo:** `elo = 400 * log10(p/(1-p))`, with `p` clamped to
  `[1/(2N), 1 - 1/(2N)]` so a clean sweep gives a large-but-finite Elo.
- **CI — draw-aware (trinomial), not Wilson:** outcomes are `{0, 0.5, 1}`, so a
  Bernoulli/Wilson interval (0/1 only) is the wrong model.
  `var = [w·(1-m)² + d·(0.5-m)² + l·(0-m)²] / N`, `SE = sqrt(var/N)`,
  `CI95 = m ± 1.96·SE`; endpoints mapped through the Elo formula → Elo CI.
- **Verdict:** `≥0.55 stronger / 0.52–0.55 weak signal / 0.48–0.52 tied /
  <0.48 worse`.
- **By-color + bias:** per pairing, `A_as_red {games,wins,losses,caps,
  score_rate}` and `A_as_black {…}`; plus `color_bias.red_win_rate_decisive`
  so a red/black first-move imbalance isn't misread as model strength.

Pure helpers:

```python
score_rate(wins, draws_plus_caps, total) -> float
elo_diff(score_rate, n) -> float                 # clamped
score_ci_trinomial(w, d, l, z=1.96) -> (lo, hi)  # on score-rate
elo_ci(w, d, l) -> (lo, hi)                       # CI endpoints -> Elo
verdict(score_rate) -> str
```

## CLI & output

### Match

```
python -m scripts.GPU.alphazero.eval_checkpoint_match \
  --checkpoint-a checkpoints/alphazero-v2-staged/model_iter_0419.safetensors \
  --checkpoint-b checkpoints/alphazero-v2-staged/model_iter_0379.safetensors \
  --games 400 --board-size 24 \
  --mcts-sims 400 --mcts-eval-batch-size 14 --mcts-stall-flush-sims 48 \
  --opening-temp-plies 20 --temp-high 1.0 --temp-low 0.1 --max-moves 200 \
  --workers 6 --base-seed 12345 \
  --output logs/eval/0419_vs_0379.json
```

Writes:
- `<output stem>_games.jsonl` — per-game `EvalGameResult`s, sorted by
  `(pairing_id, game_idx)`.
- `<output>` — summary JSON:

```json
{
  "pairing_id": "0419_vs_0379",
  "checkpoint_a": ".../model_iter_0419.safetensors",
  "checkpoint_b": ".../model_iter_0379.safetensors",
  "games": 400,
  "a_wins": 223, "b_wins": 169, "state_caps": 8, "board_full": 0,
  "a_score": 227.0, "a_score_rate": 0.5675,
  "elo_estimate": 47.1, "elo_ci95": [12.3, 82.0],
  "score_rate_ci95": [0.517, 0.617],
  "verdict": "stronger",
  "a_as_red":   {"games":200,"wins":...,"losses":...,"caps":...,"score_rate":...},
  "a_as_black": {"games":200,"wins":...,"losses":...,"caps":...,"score_rate":...},
  "color_bias": {"red_win_rate_decisive": 0.51},
  "avg_plies": 58.2,
  "config": {"mcts_sims":400,"mcts_eval_batch_size":14,"mcts_stall_flush_sims":48,
             "opening_temp_plies":20,"temp_high":1.0,"temp_low":0.1,
             "max_moves":200,"board_size":24,"base_seed":12345,"workers":6},
  "git_commit": "<sha>", "generated_at": "<iso8601>"
}
```

### Tournament

```
python -m scripts.GPU.alphazero.eval_checkpoint_tournament \
  --checkpoints-dir checkpoints/alphazero-v2-staged \
  --pairings 0419:0379,0419:0339,0419:0299,0379:0339 \
  --games 400 --board-size 24 \
  --mcts-sims 400 --mcts-eval-batch-size 14 --mcts-stall-flush-sims 48 \
  --opening-temp-plies 20 --temp-high 1.0 --temp-low 0.1 --max-moves 200 \
  --workers 6 --base-seed 12345 \
  --output-dir logs/eval/tournament_<id>/
```

- `--pairings A:B,A:C,...` explicit pairs (short iter ids resolve against
  `--checkpoints-dir`); **or** `--round-robin` builds all `C(n,2)` pairs from a
  `--checkpoints` list.
- Builds **one** flat task list across all pairings → **one** `run_game_tasks`
  call → groups results by `pairing_id`.
- Writes per-pairing summary JSON files + a combined `tournament.json` + prints
  a sorted table to stdout.

### Cross-format support

A and B auto-detect input channels independently, so a 24-channel (≤iter-0999)
vs 30-channel (post-retrain) match works — useful for measuring strength across
the Phase-2 retrain boundary.

## Testing

All game-logic tests use a **fake deterministic evaluator** (fixed priors/value)
— fast, no checkpoint or GPU, matching how `probe_eval`/MCTS are already tested.

Required (architecture — from review):
1. `workers=1` and `workers=2` produce the same sorted task identities/results
   with a fake deterministic evaluator.
2. Tournament task builder produces balanced red/black assignments per pairing.
3. Seeds are stable and independent of worker assignment.
4. No nested-pool path: tournament calls `run_game_tasks` exactly once.
5. Worker checkpoint cache loads each checkpoint at most once per worker
   (fake-cache test).

Required (stats + game logic):
6. `eval_elo.py` units: 60% → ≈+70 Elo (±tol); `p=1` clamped finite;
   symmetric `elo_A = -elo_B`; trinomial CI on known counts.
7. `play_eval_game` determinism: same seed → identical move sequence.
8. Color/winner bookkeeping: A-as-red win and A-as-black win both credit A;
   state-cap → 0.5/0.5.
9. End-to-end smoke: 2-game match, fake evaluator, small board; summary fields
   exist and counts sum correctly.

## Edge cases & error handling

- **Fail fast before spawning:** missing checkpoint path, unreadable
  safetensors, empty pairing list, `games < 2`.
- `workers = min(workers, len(tasks))` (mirrors trainer's `n_spawn`).
- **Dead-worker detection:** parent tracks `WorkerDone` count and joins with a
  timeout; a worker dying without `WorkerDone` → loud failure, not a hang.
- **Elo at extremes:** the `p` clamp above.
- **Output dir** created if absent; summary always written even if a pairing is
  degenerate (all caps).

## First runs to validate the system

The four pairings from the original proposal:

```
0419 vs 0379    # adjacent — is recent training still gaining?
0419 vs 0339
0419 vs 0299
0379 vs 0339
```

Interpretation:
- 0419 beats all clearly → keep training.
- 0419 ≈ 0379 but beats 0339/0299 → plateauing around 0379–0419.
- 0419 loses to 0379 → freeze 0379 as current best.
```
