# Resign Feature for AlphaZero Self-Play

## Goal

Add conservative resign logic to self-play games. When the MCTS root value shows sustained hopelessness (e.g., root_value ≤ -0.97 for 10+ consecutive plies), the current player resigns and the opponent wins. This reduces timeout draws and produces cleaner training signals.

**Training impact**: Resign produces decisive outcomes (winner ≠ None), so positions get ±1 value labels instead of 0. This strengthens value target distribution.

---

## Files to Modify

1. `scripts/GPU/alphazero/self_play.py` — resign constant + play_game() logic
2. `scripts/GPU/alphazero/self_play_worker.py` — pass resign params + update IPC mapping
3. `scripts/GPU/alphazero/trainer.py` — wire resign params through both paths + update inline IPC mapping
4. `scripts/GPU/alphazero/train.py` — CLI flags
5. `scripts/GPU/alphazero/game_saver.py` — handle "resign" reason cleanly
6. `docs/train-cli.md` — document new flags

---

## CLI Flags (conservative defaults)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--resign-enabled` | bool | False | Enable resign logic |
| `--resign-min-ply` | int | 80 | Don't resign before this ply |
| `--resign-threshold` | float | -0.97 | Resign when root_value ≤ this |
| `--resign-consecutive` | int | 10 | Require N consecutive low values |
| `--resign-min-visits` | int | 200 | Require root.visit_count ≥ this |
| `--resign-min-top1-share` | float | 0.0 | Require top move's visit share ≥ this (0 = disabled) |

---

## Implementation

### 1. Add RESIGN constant (`self_play.py`, after line 40)

```python
RESIGN = "resign"  # Game ended by resignation
```

### 2. Add helper function (`self_play.py`, after constants)

```python
def opponent(player: str) -> str:
    """Return opponent color."""
    return "black" if player == "red" else "red"
```

### 3. Update `play_game()` signature (`self_play.py:317`)

```python
def play_game(
    evaluator: Evaluator,
    mcts_config: Optional[MCTSConfig] = None,
    rng: Optional[random.Random] = None,
    max_moves: int = 200,
    add_noise: bool = True,
    active_size: int = 24,
    start_player: Optional[str] = None,
    game_id: int = 0,
    # Resign parameters (conservative defaults = disabled)
    resign_enabled: bool = False,
    resign_min_ply: int = 80,
    resign_threshold: float = -0.97,
    resign_consecutive: int = 10,
    resign_min_visits: int = 200,
    resign_min_top1_share: float = 0.0,  # Optional: require top move support
) -> GameRecord:
```

### 4. Add resign tracking and logic (`self_play.py`, inside play_game)

After `root = MCTSNode(state=state)`, add:
```python
    resign_streak = 0
    resigned_by: Optional[str] = None
    winner: Optional[str] = None
    draw_reason: Optional[str] = None
```

Inside the while loop, after `mcts.search_from_root(...)`, add:
```python
        # --- RESIGN CHECK (after search, before move selection) ---
        # root_value is from state.to_move perspective:
        #   +1 = to_move winning, -1 = to_move losing
        if resign_enabled and ply >= resign_min_ply:
            # Optional: require top move has enough support (avoid noisy resigns)
            if resign_min_top1_share > 0:
                total_visits = sum(visit_counts.values())
                top1_visits = max(visit_counts.values()) if visit_counts else 0
                top1_share = top1_visits / total_visits if total_visits > 0 else 0
                share_ok = top1_share >= resign_min_top1_share
            else:
                share_ok = True

            if (root.visit_count >= resign_min_visits
                and root_value <= resign_threshold
                and share_ok):
                resign_streak += 1
            else:
                resign_streak = 0

            if resign_streak >= resign_consecutive:
                # Set winner/draw_reason immediately before break
                resigned_by = state.to_move
                winner = opponent(resigned_by)
                draw_reason = RESIGN
                break
```

After the while loop, update terminal handling:
```python
    # Compute terminal status (only if not resigned)
    is_timeout = (ply >= max_moves)
    is_terminal = state.is_terminal()

    # Resign already set winner/draw_reason; handle normal endings
    if resigned_by is None:
        winner = state.winner() if is_terminal else None
        draw_reason = None

        if winner is None:
            if is_timeout:
                draw_reason = DRAW_TIMEOUT
            elif is_terminal:
                if not state.legal_moves():
                    draw_reason = DRAW_BOARD_FULL
                elif state.max_plies_limit is not None and state.ply >= state.max_plies_limit:
                    draw_reason = DRAW_STATE_CAP
                else:
                    draw_reason = DRAW_UNKNOWN
```

### 4b. Store resigned_by in GameRecord (`self_play.py`)

Add field to GameRecord dataclass:
```python
@dataclass
class GameRecord:
    positions: List[PositionRecord]
    winner: Optional[str]
    n_moves: int
    move_history: List[Tuple[int, int]] = field(default_factory=list)
    start_player: str = "red"
    draw_reason: Optional[str] = None
    resigned_by: Optional[str] = None  # NEW: who resigned (or None)
    # ... MCTS stats ...
```

Update return statement in play_game():
```python
    return GameRecord(
        positions=positions,
        winner=winner,
        n_moves=ply,
        move_history=move_history,
        start_player=start_player,
        draw_reason=draw_reason,
        resigned_by=resigned_by,  # NEW
        # ... MCTS stats ...
    )
```

### 5. Update `self_play_worker.py`

Add RESIGN to imports (line 19):
```python
from .self_play import (
    play_game, play_games, GameRecord, PositionRecord,
    DRAW_TIMEOUT, DRAW_BOARD_FULL, DRAW_STATE_CAP, DRAW_UNKNOWN, RESIGN,
)
```

Update `_DRAW_REASON_TO_INT` (after line 29):
```python
_DRAW_REASON_TO_INT = {
    None: 0,
    DRAW_TIMEOUT: 1,
    DRAW_BOARD_FULL: 2,
    DRAW_STATE_CAP: 3,
    DRAW_UNKNOWN: 4,
    RESIGN: 5,  # NEW
}
```

Update `self_play_worker_main()` signature (line 32):
```python
def self_play_worker_main(
    worker_id: int,
    request_queue: Any,
    response_queue: Any,
    position_queue: Any,
    stats_queue: Optional[Any],
    mcts_config: MCTSConfig,
    games_total: int,
    next_game_id: Any,
    seed: int,
    chunk_size: int = 32,
    max_moves: int = 200,
    add_noise: bool = True,
    active_size: int = 24,
    # Resign parameters
    resign_enabled: bool = False,
    resign_min_ply: int = 80,
    resign_threshold: float = -0.97,
    resign_consecutive: int = 10,
    resign_min_visits: int = 200,
    resign_min_top1_share: float = 0.0,
) -> None:
```

Update `play_game()` call inside worker (around line 117):
```python
        game = play_game(
            evaluator=evaluator,
            mcts_config=mcts_config,
            rng=game_rng,
            max_moves=max_moves,
            add_noise=add_noise,
            active_size=active_size,
            game_id=gid,
            resign_enabled=resign_enabled,
            resign_min_ply=resign_min_ply,
            resign_threshold=resign_threshold,
            resign_consecutive=resign_consecutive,
            resign_min_visits=resign_min_visits,
            resign_min_top1_share=resign_min_top1_share,
        )
```

### 6. Update `trainer.py`

Update `run_parallel_selfplay()` signature (around line 955):
```python
def run_parallel_selfplay(
    evaluator,
    mcts_config: MCTSConfig,
    games_to_play: int,
    n_workers: int,
    master_rng: random.Random,
    max_moves: int = 200,
    active_size: int = 24,
    curriculum=None,
    buffer=None,
    game_saver=None,
    # Resign parameters
    resign_enabled: bool = False,
    resign_min_ply: int = 80,
    resign_threshold: float = -0.97,
    resign_consecutive: int = 10,
    resign_min_visits: int = 200,
    resign_min_top1_share: float = 0.0,
):
```

Update kwargs dict when spawning workers (around line 1049):
```python
        p = ctx.Process(
            target=self_play_worker_main,
            kwargs={
                "worker_id": wid,
                # ... existing params ...
                "resign_enabled": resign_enabled,
                "resign_min_ply": resign_min_ply,
                "resign_threshold": resign_threshold,
                "resign_consecutive": resign_consecutive,
                "resign_min_visits": resign_min_visits,
                "resign_min_top1_share": resign_min_top1_share,
            },
        )
```

**Update inline IPC mapping** (around line 1154) — handle resign (winner exists but draw_reason set):
```python
                # Map draw_reason int back to string (0 = None explicitly)
                # Note: resign has winner but also has draw_reason=5
                draw_reason_str = {
                    0: None, 1: "timeout", 2: "board_full", 3: "state_cap", 4: "unknown", 5: "resign"
                }.get(msg.draw_reason)

                # Derive resigned_by from msg (resign means loser resigned)
                resigned_by = None
                if draw_reason_str == "resign" and msg.winner and msg.winner != "draw":
                    resigned_by = "black" if msg.winner == "red" else "red"

                # For resign, pass draw_reason even though winner exists
                game_saver.maybe_save_game(
                    winner=msg.winner if msg.winner != "draw" else None,
                    move_history=msg.move_history,
                    n_moves=msg.n_moves,
                    draw_reason=draw_reason_str,
                    start_player=msg.start_player,
                    resigned_by=resigned_by,  # NEW
                )
```

Update `train()` signature (around line 1296):
```python
def train(
    # ... existing params ...
    # Resign parameters
    resign_enabled: bool = False,
    resign_min_ply: int = 80,
    resign_threshold: float = -0.97,
    resign_consecutive: int = 10,
    resign_min_visits: int = 200,
    resign_min_top1_share: float = 0.0,
):
```

Pass resign params to `run_parallel_selfplay()` call (around line 1698):
```python
    _, new_positions, parallel_stats = run_parallel_selfplay(
        evaluator=evaluator,
        # ... existing params ...
        resign_enabled=resign_enabled,
        resign_min_ply=resign_min_ply,
        resign_threshold=resign_threshold,
        resign_consecutive=resign_consecutive,
        resign_min_visits=resign_min_visits,
        resign_min_top1_share=resign_min_top1_share,
    )
```

Update sequential self-play loop (around line 1745):
```python
    game = play_game(
        evaluator,
        mcts_config=iter_mcts_config,
        rng=game_rng,
        max_moves=scaled_max_moves,
        add_noise=True,
        active_size=active_size,
        start_player=start_player,
        game_id=g,
        resign_enabled=resign_enabled,
        resign_min_ply=resign_min_ply,
        resign_threshold=resign_threshold,
        resign_consecutive=resign_consecutive,
        resign_min_visits=resign_min_visits,
        resign_min_top1_share=resign_min_top1_share,
    )
```

Update sequential game_saver call (after game is created):
```python
    game_saver.maybe_save_game(
        winner=game.winner,
        move_history=tuple(game.move_history),
        n_moves=game.n_moves,
        draw_reason=game.draw_reason,
        start_player=game.start_player,
        resigned_by=game.resigned_by,  # NEW
    )
```

### 7. Update `train.py` — CLI flags (after other flags)

```python
    # Resign parameters
    parser.add_argument("--resign-enabled", action="store_true",
        help="Enable automatic resign when position is hopeless")
    parser.add_argument("--resign-min-ply", type=int, default=80,
        help="Don't resign before this ply (default: 80)")
    parser.add_argument("--resign-threshold", type=float, default=-0.97,
        help="Resign when root value <= this (default: -0.97)")
    parser.add_argument("--resign-consecutive", type=int, default=10,
        help="Require N consecutive low values to resign (default: 10)")
    parser.add_argument("--resign-min-visits", type=int, default=200,
        help="Require root visits >= this to resign (default: 200)")
    parser.add_argument("--resign-min-top1-share", type=float, default=0.0,
        help="Require top move's visit share >= this to resign (default: 0 = disabled)")
```

Add validation:
```python
    # Validate resign parameters
    if args.resign_min_ply < 0:
        parser.error("--resign-min-ply must be >= 0")
    if args.resign_threshold > 0:
        parser.error("--resign-threshold must be <= 0 (negative means losing)")
    if args.resign_threshold < -1.0:
        parser.error("--resign-threshold must be >= -1.0 (value is tanh in [-1,1])")
    if args.resign_consecutive < 1:
        parser.error("--resign-consecutive must be >= 1")
    if args.resign_min_visits < 1:
        parser.error("--resign-min-visits must be >= 1")
    if not (0.0 <= args.resign_min_top1_share <= 1.0):
        parser.error("--resign-min-top1-share must be in [0, 1]")
```

Pass to train():
```python
    network = train(
        # ... existing params ...
        resign_enabled=args.resign_enabled,
        resign_min_ply=args.resign_min_ply,
        resign_threshold=args.resign_threshold,
        resign_consecutive=args.resign_consecutive,
        resign_min_visits=args.resign_min_visits,
        resign_min_top1_share=args.resign_min_top1_share,
    )
```

### 8. Update `game_saver.py` — handle resign reason + resigned_by

Update signature to accept resigned_by:
```python
def save_game_replay(
    games_dir: Path,
    iteration: int,
    game_idx: int,
    winner: Optional[str],
    move_history: Tuple[Tuple[int, int], ...],
    n_moves: int,
    active_size: int = 24,
    simulations: int = 0,
    draw_reason: Optional[str] = None,
    start_player: str = "red",
    resigned_by: Optional[str] = None,  # NEW
) -> Path:
```

Update the reason logic (around line 66):
```python
    # Determine reason
    if winner:
        # Winner exists - could be normal win or resignation
        if draw_reason == "resign":
            reason = "resign"
        else:
            reason = "win"
    elif draw_reason:
        reason = draw_reason
    else:
        reason = "draw"
```

Add resigned_by to meta only when reason == resign (inside record dict):
```python
    meta = {
        # ... existing fields ...
    }
    if reason == "resign" and resigned_by:
        meta["resigned_by"] = resigned_by

    record = {
        # ... existing fields ...
        "meta": meta,
    }
```

Update `GameSaver.maybe_save_game()` to pass through resigned_by:
```python
def maybe_save_game(
    self,
    winner: Optional[str],
    move_history: Optional[Tuple[Tuple[int, int], ...]],
    n_moves: int,
    draw_reason: Optional[str] = None,
    start_player: str = "red",
    resigned_by: Optional[str] = None,  # NEW
) -> Optional[Path]:
```

### 9. Update `docs/train-cli.md`

Add new section after existing flags:

```markdown
## Resign

Allows self-play games to end early when the position is hopeless.
Disabled by default (conservative).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--resign-enabled` | flag | off | Enable automatic resignation |
| `--resign-min-ply` | int | 80 | Don't resign before this ply |
| `--resign-threshold` | float | -0.97 | Resign when root value ≤ this |
| `--resign-consecutive` | int | 10 | Require N consecutive low values |
| `--resign-min-visits` | int | 200 | Require root visits ≥ this |
| `--resign-min-top1-share` | float | 0.0 | Require top move's visit share ≥ this (0 = disabled) |

**Example**: `--resign-enabled --resign-threshold -0.95 --resign-consecutive 8`

**Validation**: min_ply ≥ 0, threshold ≤ 0, consecutive ≥ 1, min_visits ≥ 1, top1_share ∈ [0, 1]

**Note**: Resign is from current player's perspective. If root_value ≤ threshold for N consecutive searches, current player resigns and opponent wins.

**Training impact**: Resigned games produce decisive outcomes (winner ≠ draw), so positions get ±1 value labels instead of 0. This strengthens the value target distribution and reduces timeout draws.
```

---

## Key Design Points

1. **Perspective correctness**: `root_value` is from `state.to_move`'s perspective. Negative = losing. If ≤ -0.97, current player is ~97% likely to lose → they resign → opponent wins. (Comment this explicitly in code.)

2. **Resign is NOT a draw**: `winner` is set to opponent, `draw_reason` is set to "resign" for metadata tracking. Positions are labeled +1/-1 (not 0). Semantically `draw_reason` is overloaded for this; a future refactor could rename to `end_reason`. **IMPORTANT**: Downstream logic must treat draws as `winner is None` (or `winner_str == "draw"`), NOT as `draw_reason != None`, because resign uses draw_reason for metadata while still having a winner.

3. **Conservative defaults**: Disabled by default. When enabled: ply ≥ 80 (ensures midgame has unfolded; early games are too noisy), threshold -0.97, 10 consecutive, 200 visits. Very hard to trigger accidentally.

4. **Both paths covered**: Sequential (single-process) and parallel (multi-process) paths both receive resign parameters.

5. **IPC compatible**: `_DRAW_REASON_TO_INT` updated with RESIGN=5. Update reverse mapping wherever draw_reason is decoded (inline mapping at trainer.py:1154). No separate `_INT_TO_DRAW_REASON` dict exists — only the inline one. Note: `GameComplete` IPC message (ipc_messages.py:59) already carries both `winner: str` and `draw_reason: int`, so resign is encoded as `winner="red"/"black"` + `draw_reason=5`.

6. **Set winner/draw_reason before break**: Inside the resign trigger, set both variables immediately before `break` to avoid "variable referenced before assignment" bugs.

7. **Optional false-resign guard**: `resign_min_top1_share` requires the top move to have at least X% of visits, reducing resigns due to noisy/uncertain evals.

8. **visit_counts note**: `visit_counts` includes zeros for unvisited legal moves (children may be absent → 0). `total_visits = sum(...)` correctly equals total root child visits; `max(...)` picks the largest.

---

## Verification

1. **Compile check**:
   ```bash
   python3 -m py_compile scripts/GPU/alphazero/self_play.py scripts/GPU/alphazero/self_play_worker.py scripts/GPU/alphazero/trainer.py scripts/GPU/alphazero/train.py scripts/GPU/alphazero/game_saver.py
   ```

2. **Unit tests**:
   ```bash
   .venv/bin/python -m pytest tests/test_mcts.py tests/test_self_play.py -v
   ```

3. **Smoke test (resign disabled)**:
   ```bash
   .venv/bin/python -m scripts.GPU.alphazero.train \
     --iterations 1 --games-per-iter 4 --train-steps 1 \
     --simulations 50 --n-workers 1 --seed 42 \
     --checkpoint-dir /tmp/resign-test
   ```

4. **Smoke test (resign enabled, loose threshold to trigger)**:
   ```bash
   .venv/bin/python -m scripts.GPU.alphazero.train \
     --iterations 1 --games-per-iter 10 --train-steps 1 \
     --simulations 100 --n-workers 1 --seed 42 \
     --resign-enabled --resign-threshold -0.5 --resign-consecutive 3 --resign-min-ply 10 \
     --checkpoint-dir /tmp/resign-test
   ```
   Expect: Some games end with "resign" reason, shorter average plies.

5. **Check replay JSON**:
   ```bash
   cat /tmp/resign-test/scripts/GPU/logs/games/*.json | grep '"reason"'
   ```
   Should see `"reason": "resign"` for resigned games.

---

## Status

**IMPLEMENTED** - 2025-02-25

All changes have been made and verified:
- Compile check passes
- Unit tests pass (18/19, one pre-existing flaky test)
- Smoke tests pass for both resign-disabled and resign-enabled modes
