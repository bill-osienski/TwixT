# Plan: Fix Timeout Draws + Curriculum Stalling

## Goals (Definition of Done)
1. No more "fake draws" caused by undersized max_moves
2. Self-play and game-state terminal logic share the same cap
3. Training output shows: total draws, timeout draws, board-full draws, state-cap draws, unknown draws
4. Curriculum promotion uses true draws, not timeout artifacts
5. Works for all curriculum sizes including final 24x24

---

## Phase 1: Make Move Limits Consistent

### 1.1 Add `max_plies_limit` to TwixtState

**File:** `scripts/GPU/alphazero/game/twixt_state.py`

Add field to dataclass (around line 135):
```python
max_plies_limit: Optional[int] = None  # if set, state becomes terminal at this ply
```

Update imports at top if needed:
```python
from typing import Dict, List, Optional, Set, Tuple
```

### 1.2 Update `copy()` to carry the field

**File:** `scripts/GPU/alphazero/game/twixt_state.py:145-154`

```python
def copy(self) -> TwixtState:
    return TwixtState(
        board_size=self.board_size,
        active_size=self.active_size,
        to_move=self.to_move,
        pegs=dict(self.pegs),
        bridges=set(self.bridges),
        ply=self.ply,
        max_plies_limit=self.max_plies_limit,  # ADD THIS
    )
```

### 1.3 Update `is_terminal()` to use dynamic cap first

**File:** `scripts/GPU/alphazero/game/twixt_state.py:459-479`

Change from:
```python
if self.ply >= MAX_PLIES:
    return True
```

To:
```python
# Dynamic cap (from self-play) takes precedence
if self.max_plies_limit is not None and self.ply >= self.max_plies_limit:
    return True

# Safety cap (should never fire in normal training)
if self.ply >= MAX_PLIES:
    return True
```

### 1.4 Raise MAX_PLIES safety cap

**File:** `scripts/GPU/alphazero/game/twixt_state.py:24`

Change from:
```python
MAX_PLIES = 200
```

To:
```python
# Safety clamp: should never fire if max_plies_limit is used during self-play/training.
MAX_PLIES = 600
```

---

## Phase 2: Add Draw Reason Instrumentation

### 2.1 Add draw reason constants and field

**File:** `scripts/GPU/alphazero/self_play.py` (near top, after imports)

Add constants to avoid string typos across files:
```python
# Draw reason constants (used in GameRecord, curriculum, trainer)
DRAW_TIMEOUT = "timeout_selfplay"
DRAW_BOARD_FULL = "terminal_board_full"
DRAW_STATE_CAP = "terminal_state_cap"
DRAW_UNKNOWN = "terminal_unknown"
```

Add field to GameRecord (around line 96-109):
```python
draw_reason: Optional[str] = None  # DRAW_TIMEOUT, DRAW_BOARD_FULL, DRAW_STATE_CAP, DRAW_UNKNOWN
```

### 2.2 Construct state with same cap as self-play loop

**File:** `scripts/GPU/alphazero/self_play.py:164`

Change from:
```python
state = TwixtState(active_size=active_size, to_move=start_player)
```

To:
```python
state = TwixtState(
    active_size=active_size,
    to_move=start_player,
    max_plies_limit=max_moves,  # Unify cap with self-play loop
)

# Invariant: caps must match (catch divergence bugs early)
assert state.max_plies_limit == max_moves, "State cap must match self-play cap"
```

Note: `apply_move()` uses `self.copy()` which now carries `max_plies_limit`, so MCTS-created states inherit the cap automatically.

### 2.3 Compute winner + draw_reason after loop (CORRECTED)

**File:** `scripts/GPU/alphazero/self_play.py:209-223`

Replace:
```python
if state.is_terminal():
    winner = state.winner()
else:
    winner = None  # Max moves reached = draw
```

With (using ply for timeout detection, not state.is_terminal()):
```python
# Compute these once to avoid refactor bugs
is_timeout = (ply >= max_moves)
is_terminal = state.is_terminal()
winner = state.winner() if is_terminal else None
draw_reason = None

if winner is None:
    # No winner - determine draw reason
    # Check ply first (authoritative for timeout)
    if is_timeout:
        draw_reason = DRAW_TIMEOUT
    elif is_terminal:
        # State is terminal but no winner - why?
        if not state.legal_moves():
            draw_reason = DRAW_BOARD_FULL
        elif state.max_plies_limit is not None and state.ply >= state.max_plies_limit:
            draw_reason = DRAW_STATE_CAP
        else:
            draw_reason = DRAW_UNKNOWN
```

**Why:** "timeout_selfplay" means "we hit the loop cap (ply >= max_moves)", regardless of what `state.is_terminal()` says. This gives consistent attribution.

### 2.4 Return draw_reason in GameRecord

**File:** `scripts/GPU/alphazero/self_play.py:224-239`

Add `draw_reason=draw_reason` to the return statement. Use `n_moves=ply` (what self-play actually ran, authoritative).

```python
return GameRecord(
    positions=positions,
    winner=winner,
    n_moves=ply,  # ply is authoritative (not state.ply)
    draw_reason=draw_reason,
    move_history=move_history,
    ...
)
```

---

## Phase 3: Fix max_moves Policy

### 3.1 Add per-size table

**File:** `scripts/GPU/alphazero/trainer.py` (near top, after imports)

```python
# Per-size max moves table (tuned to give Black time to convert)
# NOTE: These are PLIES (half-moves), not full moves. TwixT alternates turns,
# so 175 plies = ~87 moves per player at size 16.
MAX_MOVES_TABLE = {
    8:  90,
    10: 110,
    12: 130,
    16: 175,
    20: 225,
    24: 280,
}

def get_scaled_max_moves(active_size: int, fallback_mult: float = 10.0) -> int:
    """Get max moves for curriculum size."""
    return MAX_MOVES_TABLE.get(active_size, int(fallback_mult * active_size))
```

### 3.2 Replace hardcoded formula

**File:** `scripts/GPU/alphazero/trainer.py:401`

Change from:
```python
scaled_max_moves = 6 * active_size  # Scale max_moves with board size
```

To:
```python
scaled_max_moves = get_scaled_max_moves(active_size)
```

---

## Phase 4: Curriculum Ignores Timeout Draws

### 4.0 Import draw reason constants

**File:** `scripts/GPU/alphazero/curriculum.py` (at top)

```python
from .self_play import DRAW_TIMEOUT, DRAW_BOARD_FULL, DRAW_STATE_CAP, DRAW_UNKNOWN
```

**Note:** If you hit circular-import issues (self_play.py importing curriculum.py indirectly), move constants to a tiny `scripts/GPU/alphazero/constants.py` and have everyone import from there. Not required if current imports are stable.

### 4.1 Change history to store (winner, draw_reason) tuples

**File:** `scripts/GPU/alphazero/curriculum.py`

Change `_history` type (line 46):
```python
_history: List[Tuple[Optional[str], Optional[str]]] = field(default_factory=list)
# Each entry: (winner, draw_reason) e.g. ("red", None), (None, "timeout_selfplay")
```

### 4.2 Update `record_game()` signature

**File:** `scripts/GPU/alphazero/curriculum.py:61-70`

Change from:
```python
def record_game(self, winner: Optional[str]) -> None:
    self._history.append(winner)
```

To:
```python
def record_game(self, winner: Optional[str], draw_reason: Optional[str] = None) -> None:
    self._history.append((winner, draw_reason))
```

### 4.3 Update `get_metrics()` to separate draw types (ENHANCED)

**File:** `scripts/GPU/alphazero/curriculum.py:72-107`

Replace with:
```python
def get_metrics(self) -> dict:
    if not self._history:
        # No games yet - use 0.0 for rates to avoid confusing "100% draws" printouts
        return {
            "red_wins": 0, "black_wins": 0,
            "draws": 0, "timeout_draws": 0, "board_full_draws": 0,
            "state_cap_draws": 0, "unknown_draws": 0,
            "total": 0,
            "draw_rate": 0.0, "draw_rate_true": 0.0, "draw_rate_timeout": 0.0,
            "red_win_rate": 0.0, "black_win_rate": 0.0,
        }

    red_wins = sum(1 for w, _ in self._history if w == "red")
    black_wins = sum(1 for w, _ in self._history if w == "black")

    # Break down draws by reason (using constants)
    timeout_draws = sum(1 for w, dr in self._history if w is None and dr == DRAW_TIMEOUT)
    board_full_draws = sum(1 for w, dr in self._history if w is None and dr == DRAW_BOARD_FULL)
    state_cap_draws = sum(1 for w, dr in self._history if w is None and dr == DRAW_STATE_CAP)
    unknown_draws = sum(1 for w, dr in self._history if w is None and dr == DRAW_UNKNOWN)

    draws = timeout_draws + board_full_draws + state_cap_draws + unknown_draws
    true_draws = draws - timeout_draws  # Non-timeout draws
    total = len(self._history)

    decisive = red_wins + black_wins
    draw_rate = draws / total if total > 0 else 0.0
    draw_rate_true = true_draws / total if total > 0 else 0.0
    draw_rate_timeout = timeout_draws / total if total > 0 else 0.0
    # Note: win rates are conditional on decisive games (excludes draws)
    red_win_rate = red_wins / decisive if decisive > 0 else 0.0
    black_win_rate = black_wins / decisive if decisive > 0 else 0.0

    return {
        "red_wins": red_wins, "black_wins": black_wins,
        "draws": draws, "timeout_draws": timeout_draws,
        "board_full_draws": board_full_draws, "state_cap_draws": state_cap_draws,
        "unknown_draws": unknown_draws,
        "total": total,
        "draw_rate": draw_rate, "draw_rate_true": draw_rate_true,
        "draw_rate_timeout": draw_rate_timeout,
        "red_win_rate": red_win_rate, "black_win_rate": black_win_rate,
    }
```

### 4.4 Update `should_promote()` to use true draws

**File:** `scripts/GPU/alphazero/curriculum.py:127-129`

Change from:
```python
if metrics["draw_rate"] > self.draw_threshold:
    return False
```

To:
```python
if metrics["draw_rate_true"] > self.draw_threshold:
    return False
```

### 4.5 Update trainer to import constants and pass draw_reason

**File:** `scripts/GPU/alphazero/trainer.py` (at top, update import)

```python
from .self_play import play_game, DRAW_TIMEOUT, DRAW_BOARD_FULL, DRAW_STATE_CAP, DRAW_UNKNOWN
```

**File:** `scripts/GPU/alphazero/trainer.py:458-467`

Change from:
```python
if game.winner == "red":
    red_wins += 1
    curriculum.record_game("red")
elif game.winner == "black":
    black_wins += 1
    curriculum.record_game("black")
else:
    draws += 1
    curriculum.record_game(None)
```

To (always pass both winner and draw_reason, using constants):
```python
# Track results for display
if game.winner == "red":
    red_wins += 1
elif game.winner == "black":
    black_wins += 1
else:
    draws += 1
    # Track draw breakdown for display (using constants)
    if game.draw_reason == DRAW_TIMEOUT:
        timeout_draws += 1
    elif game.draw_reason == DRAW_BOARD_FULL:
        board_full_draws += 1
    elif game.draw_reason == DRAW_STATE_CAP:
        state_cap_draws += 1
    else:
        unknown_draws += 1

# Always record to curriculum (draw_reason is None for wins)
curriculum.record_game(game.winner, game.draw_reason)
```

Also add to initialization block (around line 415):
```python
timeout_draws = 0
board_full_draws = 0
state_cap_draws = 0
unknown_draws = 0
```

---

## Phase 5: Trainer Output Improvements (ENHANCED)

### 5.1 Update stats printing with full breakdown

**File:** `scripts/GPU/alphazero/trainer.py:483-492`

Change:
```python
print(f"  Results: Red={red_wins}, Black={black_wins}, Draws={draws}")
```

To:
```python
print(f"  Results: Red={red_wins}, Black={black_wins}, Draws={draws}")
print(f"    Draw breakdown: timeout={timeout_draws}, board_full={board_full_draws}, "
      f"state_cap={state_cap_draws}, unknown={unknown_draws}")
```

### 5.2 Add curriculum metrics to promotion output

**File:** `scripts/GPU/alphazero/trainer.py:496-501`

Change:
```python
print(f"\n*** CURRICULUM PROMOTED: active_size={curriculum.active_size} ***")
print(f"    Previous metrics: draw_rate={metrics['draw_rate']:.1%}")
```

To:
```python
print(f"\n*** CURRICULUM PROMOTED: active_size={curriculum.active_size} ***")
print(f"    Previous metrics: draw_rate_true={metrics['draw_rate_true']:.1%}, "
      f"timeout_rate={metrics['draw_rate_timeout']:.1%}")
```

---

## Phase 6: Verification

### 6.1 Micro sanity run
```bash
.venv/bin/python -u -m scripts.GPU.alphazero.train \
    --iterations 1 --games-per-iter 2 --simulations 100 \
    --curriculum-sizes 16 --train-steps 0 2>&1
```

Confirm:
- No crashes
- draw_reason populated in output
- "Draw breakdown:" line appears with timeout/board_full/state_cap/unknown

### 6.2 Focused tuning on size 16
```bash
.venv/bin/python -u -m scripts.GPU.alphazero.train \
    --iterations 5 --games-per-iter 25 --simulations 400 \
    --curriculum-sizes 16 --train-steps 50 2>&1 | tee logs/test-size16.log
```

Watch for:
- timeout_draws should be low (may not be zero - see note below)
- **state_cap_draws MUST be zero** (this is the key invariant - caps match)
- Black wins appearing
- Avg plies below cap

**Important invariant note:**
- `timeout_draws` may not be literally zero even when correct (ply can hit max_moves before state.is_terminal() fires depending on timing)
- `state_cap_draws == 0` is the real invariant (if >0, caps diverged somehow)

**"We fixed it" signature:**
- timeout_draws should drop sharply compared to before (was ~30% of games)
- avg_plies should rise (previously-timeouted games now finish naturally)
- Black wins should appear (those timeout draws were Black-winning positions)

**If timeout_draws still high at size 16:**
- First knob to turn: bump 16: 175 → 200 (or 220)
- No code changes needed, just update MAX_MOVES_TABLE

---

## Implementation Checklist

- [ ] Phase 1.1: Add `max_plies_limit` field to TwixtState
- [ ] Phase 1.2: Update `copy()` to carry field
- [ ] Phase 1.3: Update `is_terminal()` for dynamic cap
- [ ] Phase 1.4: Raise MAX_PLIES to 600 with comment
- [ ] Phase 2.1: Add draw reason constants + `draw_reason` field to GameRecord
- [ ] Phase 2.2: Pass `max_plies_limit` when creating state + add assert
- [ ] Phase 2.3: Compute draw_reason after game loop (ply-based timeout detection)
- [ ] Phase 2.4: Return draw_reason in GameRecord
- [ ] Phase 3.1: Add MAX_MOVES_TABLE and helper function
- [ ] Phase 3.2: Replace formula with table lookup
- [ ] Phase 4.0: Import draw reason constants in curriculum.py
- [ ] Phase 4.1: Change curriculum history to tuples
- [ ] Phase 4.2: Update `record_game()` signature
- [ ] Phase 4.3: Update `get_metrics()` for full draw breakdown (using constants)
- [ ] Phase 4.4: Update `should_promote()` for true draws
- [ ] Phase 4.5: Import constants in trainer.py + pass (winner, draw_reason)
- [ ] Phase 5.1: Update Results printout with full breakdown
- [ ] Phase 5.2: Update promotion metrics printout
- [ ] Phase 6.1: Run micro sanity test
- [ ] Phase 6.2: Run focused size-16 test

---

## Gotchas to Avoid

**A: Don't forget `max_plies_limit` in `copy()`**
- If forgotten, MCTS-created states lose the cap and `is_terminal()` becomes inconsistent

**B: Ensure state cap and self-play cap match**
- We set `max_plies_limit=max_moves` in state construction
- If they differ, "timeout_selfplay" and "terminal_state_cap" will disagree

**C: Use `ply` for timeout detection, not `state.is_terminal()`**
- "timeout_selfplay" means the loop cap was hit
- Check `ply >= max_moves` first, then check state terminal reasons

---

## Max Moves Table (Starting Values)

**Note:** These are PLIES (half-moves), not full moves. TwixT alternates turns.

| Size | Max Plies | Per Player | Rationale |
|------|-----------|------------|-----------|
| 8    | 90        | ~45 moves  | ~11x size |
| 10   | 110       | ~55 moves  | ~11x size |
| 12   | 130       | ~65 moves  | ~11x size |
| 16   | 175       | ~87 moves  | ~11x size |
| 20   | 225       | ~112 moves | ~11x size |
| 24   | 280       | ~140 moves | ~12x size |

Tune based on timeout_draw_rate:
- If > 10%: increase cap by 25-50
- If avg plies far below cap: you're fine
- If state_cap_draws > 0: bug (caps should match)
