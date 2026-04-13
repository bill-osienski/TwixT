# Adjudicate Timeout Games Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a self-play game hits max_moves without a terminal winner, run one final deterministic MCTS search and convert clearly-decided positions into decisive outcomes instead of timeout draws.

**Architecture:** Adjudication runs at the end of `play_game()` in `self_play.py`, after the game loop exits due to `ply >= max_moves`. It performs one `mcts.search_from_root(root, add_noise=False)` call to get a stable `root_value`, then applies confidence gates (min_ply, threshold, min_visits, optional top1_share) to decide if the position is clearly won/lost. The feature follows the same wiring pattern as resign: constant + play_game params -> worker IPC int code -> trainer counters/printing -> CLI flags -> game_saver mapping -> docs.

**Tech Stack:** Python (self-play/training pipeline), MLX (MCTS neural network evaluation), pytest (tests)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `scripts/GPU/alphazero/self_play.py` | Modify | Add `ADJUDICATED` constant, adjudication params to `play_game()`, adjudication logic in end-of-game block |
| `scripts/GPU/alphazero/self_play_worker.py` | Modify | Pass adjudication params through to `play_game()`, add int code 6 to `_DRAW_REASON_TO_INT` |
| `scripts/GPU/alphazero/ipc_messages.py` | Modify | Update `GameComplete.draw_reason` docstring to include code 6=adjudicated |
| `scripts/GPU/alphazero/trainer.py` | Modify | Add adjudication params to `run_parallel_selfplay()` and `train()`, add counters, update `process_stats_message()`, update sequential path, add print line, update stats dict |
| `scripts/GPU/alphazero/train.py` | Modify | Add 5 CLI flags with validation, pass to `train()` |
| `scripts/GPU/alphazero/game_saver.py` | Modify | Update reason mapping for `draw_reason == "adjudicated"` |
| `docs/train-cli.md` | Modify | Add Adjudication section between Resign and Mirror Augmentation |
| `tests/test_self_play.py` | Modify | Add adjudication unit tests |

---

### Task 1: Add ADJUDICATED constant and import in self_play.py

**Files:**
- Modify: `scripts/GPU/alphazero/self_play.py:37-44` (constants block)

- [ ] **Step 1: Add the ADJUDICATED constant**

After the existing `RESIGN = "resign"` line (line 44), add:

```python
# Adjudication constant (game hit max_moves; winner assigned by final MCTS eval)
ADJUDICATED = "adjudicated"
```

- [ ] **Step 2: Verify import works**

Run: `.venv/bin/python -c "from scripts.GPU.alphazero.self_play import ADJUDICATED; print(ADJUDICATED)"`
Expected: `adjudicated`

- [ ] **Step 3: Commit**

```bash
git add scripts/GPU/alphazero/self_play.py
git commit -m "feat(adjudicate): add ADJUDICATED constant to self_play.py"
```

---

### Task 2: Add adjudication parameters to play_game() signature

**Files:**
- Modify: `scripts/GPU/alphazero/self_play.py:336-353` (play_game signature)

- [ ] **Step 1: Extend play_game() signature**

After the resign parameters block (line 352), add:

```python
    # Adjudication-at-timeout parameters (disabled by default)
    adjudicate_enabled: bool = False,
    adjudicate_min_ply: int = 120,
    adjudicate_threshold: float = 0.90,
    adjudicate_min_visits: int = 200,
    adjudicate_min_top1_share: float = 0.0,
```

Update the docstring (after line 370) to include:

```python
        adjudicate_enabled: Enable timeout adjudication (default: False)
        adjudicate_min_ply: Don't adjudicate before this ply (default: 120)
        adjudicate_threshold: Absolute root_value threshold for adjudication (default: 0.90)
        adjudicate_min_visits: Require root.visit_count >= this (default: 200)
        adjudicate_min_top1_share: Require top move's visit share >= this (default: 0.0 = disabled)
```

- [ ] **Step 2: Verify no import/syntax errors**

Run: `.venv/bin/python -c "from scripts.GPU.alphazero.self_play import play_game; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/GPU/alphazero/self_play.py
git commit -m "feat(adjudicate): add adjudication params to play_game() signature"
```

---

### Task 3: Implement adjudication logic in end-of-game block

**Files:**
- Modify: `scripts/GPU/alphazero/self_play.py:525-559` (end-of-game block)

- [ ] **Step 1: Replace the timeout branch with adjudication logic**

The current code at lines 533-537 is:

```python
        if winner is None:
            # No winner - determine draw reason
            # Check ply first (authoritative for timeout)
            if is_timeout:
                draw_reason = DRAW_TIMEOUT
```

Replace the `if is_timeout:` branch (keep the outer `if winner is None:` and `# No winner` comment) with:

First, **before** the end-of-game block (near where `resigned_by`, `winner`, `draw_reason` are initialized around line 399-401), add sentinel values for adjudication diagnostics:

```python
    adj_root_value = None   # Final MCTS eval at cap (for diagnostics)
    adj_top1_share = None   # Top-1 visit share at cap (for diagnostics)
```

Then replace the `if is_timeout:` branch:

```python
            if is_timeout:
                # INVARIANT: adjudicate only when winner is None
                # (guaranteed here by outer `if winner is None:` guard)
                draw_reason = None  # Reset before adjudication attempt
                # --- ADJUDICATE TIMEOUT (optional) ---
                if adjudicate_enabled and ply >= adjudicate_min_ply:
                    # Run a final deterministic search at the cap state
                    adj_visit_counts, adj_root_value, adj_root = mcts.search_from_root(
                        root, add_noise=False, ply=ply
                    )

                    # Confidence gates
                    visits_ok = (adj_root.visit_count >= adjudicate_min_visits)

                    if adjudicate_min_top1_share > 0 and adj_visit_counts:
                        total_visits = sum(adj_visit_counts.values())
                        top1_visits = max(adj_visit_counts.values())
                        adj_top1_share = (top1_visits / total_visits) if total_visits > 0 else 0.0
                        top1_ok = (adj_top1_share >= adjudicate_min_top1_share)
                    else:
                        top1_ok = True

                    # Decide winner if confident enough
                    if visits_ok and top1_ok:
                        if adj_root_value >= adjudicate_threshold:
                            winner = state.to_move
                            draw_reason = ADJUDICATED
                        elif adj_root_value <= -adjudicate_threshold:
                            winner = opponent(state.to_move)
                            draw_reason = ADJUDICATED

                # Fall back to timeout if not adjudicated (or adjudication disabled/skipped)
                if draw_reason is None:
                    draw_reason = DRAW_TIMEOUT
```

**Key correctness notes:**
- `draw_reason = None` reset is safe because it's inside `if is_timeout:`, so non-timeout terminal draw reasons (board_full, state_cap) are never masked.
- `adj_root_value` and `adj_top1_share` are initialized to `None` outside the block, so the diagnostic print is always safe.
- `root` is synced with `state` from the last loop iteration (tree reuse). This is consistent.
- `add_noise=False` is critical -- adjudication must be deterministic.
- `adj_root_value` is from `state.to_move` perspective: `+1` = to_move winning, `-1` = to_move losing.
- The `elif` branches for `DRAW_BOARD_FULL`, `DRAW_STATE_CAP`, `DRAW_UNKNOWN` remain unchanged after this block.
- The `if draw_reason is None` fallback is now **outside** the `if adjudicate_enabled` block, so it also catches the case where adjudication is disabled or ply < min_ply.
- **Do NOT set `resigned_by` during adjudication.** It must stay `None`. If future work needs "who was favored at cap", add a separate field (e.g. `adjudicated_from_to_move`) -- never overload `resigned_by`.

- [ ] **Step 2: Update the diagnostic print for adjudicated games**

After the existing timeout diagnostic (line 557-559), add:

```python
    if draw_reason == ADJUDICATED and adj_root_value is not None:
        top1_str = f", top1={adj_top1_share:.2f}" if adj_top1_share is not None else ""
        print(f"  ADJUDICATED: plies={ply}, winner={winner}, root_value={adj_root_value:.3f}{top1_str}")
```

The `adj_root_value is not None` guard is belt-and-suspenders since `ADJUDICATED` is only set when `adj_root_value` is assigned, but it prevents any future refactoring from causing a crash. Including `top1_share` when computed helps with threshold tuning (same pattern as resign diagnostics).

- [ ] **Step 3: Verify syntax**

Run: `.venv/bin/python -c "from scripts.GPU.alphazero.self_play import play_game; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add scripts/GPU/alphazero/self_play.py
git commit -m "feat(adjudicate): implement adjudication logic in play_game end-of-game block"
```

---

### Task 4: Wire adjudication through self_play_worker.py

**Files:**
- Modify: `scripts/GPU/alphazero/self_play_worker.py:19-33` (imports + reason map)
- Modify: `scripts/GPU/alphazero/self_play_worker.py:36-59` (self_play_worker_main signature)
- Modify: `scripts/GPU/alphazero/self_play_worker.py:94-116` (_worker_loop signature)
- Modify: `scripts/GPU/alphazero/self_play_worker.py:139-154` (play_game call)

- [ ] **Step 1: Add ADJUDICATED import and int code**

Add `ADJUDICATED` to the **existing** `.self_play` import list (do not remove current imports like `play_game`, `PositionRecord`, etc.). The result should look like:

```python
from .self_play import (
    play_game, PositionRecord,
    DRAW_TIMEOUT, DRAW_BOARD_FULL, DRAW_STATE_CAP, DRAW_UNKNOWN, RESIGN,
    ADJUDICATED,
)
```

At line 33, add to `_DRAW_REASON_TO_INT`:

```python
    ADJUDICATED: 6,  # Adjudicated at timeout (has winner, decisive)
```

- [ ] **Step 2: Add adjudication params to self_play_worker_main()**

After the resign parameters (line 58), add:

```python
    # Adjudication parameters
    adjudicate_enabled: bool = False,
    adjudicate_min_ply: int = 120,
    adjudicate_threshold: float = 0.90,
    adjudicate_min_visits: int = 200,
    adjudicate_min_top1_share: float = 0.0,
```

Pass them through in the `_worker_loop()` call (line 81-87):

```python
        _worker_loop(
            worker_id, request_queue, response_queue, position_queue,
            stats_queue, mcts_config, games_total, next_game_id, seed,
            chunk_size, max_moves, add_noise, active_size,
            resign_enabled, resign_min_ply, resign_threshold,
            resign_window, resign_k, resign_min_visits, resign_min_top1_share,
            adjudicate_enabled, adjudicate_min_ply, adjudicate_threshold,
            adjudicate_min_visits, adjudicate_min_top1_share,
        )
```

- [ ] **Step 3: Add adjudication params to _worker_loop()**

After the resign parameters (line 115), add:

```python
    # Adjudication parameters
    adjudicate_enabled: bool,
    adjudicate_min_ply: int,
    adjudicate_threshold: float,
    adjudicate_min_visits: int,
    adjudicate_min_top1_share: float,
```

Pass them through in the `play_game()` call (line 139-154):

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
            resign_window=resign_window,
            resign_k=resign_k,
            resign_min_visits=resign_min_visits,
            resign_min_top1_share=resign_min_top1_share,
            adjudicate_enabled=adjudicate_enabled,
            adjudicate_min_ply=adjudicate_min_ply,
            adjudicate_threshold=adjudicate_threshold,
            adjudicate_min_visits=adjudicate_min_visits,
            adjudicate_min_top1_share=adjudicate_min_top1_share,
        )
```

- [ ] **Step 4: Simplify draw_reason_int mapping**

Replace the entire `draw_reason_int` computation block (around lines 174-184) with a simpler, safer pattern that automatically handles resign, adjudicated, and all draw reasons:

```python
            # Compute draw_reason_int from game.draw_reason (handles all cases)
            draw_reason_int = _DRAW_REASON_TO_INT.get(game.draw_reason, 0)
            # Defensive: if draw_reason is set but not in dict, force unknown
            if game.draw_reason is not None and draw_reason_int == 0:
                draw_reason_int = _DRAW_REASON_TO_INT[DRAW_UNKNOWN]
```

This automatically supports:
- normal wins -> draw_reason None -> 0
- resign/adjudicated -> code 5/6 even though winner exists
- timeout draw -> code 1
- board_full/state_cap/unknown -> codes 2/3/4
- unknown future reason string not yet in dict -> falls back to unknown (4) instead of silently becoming 0

- [ ] **Step 5: Verify import works**

Run: `.venv/bin/python -c "from scripts.GPU.alphazero.self_play_worker import self_play_worker_main; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/self_play_worker.py
git commit -m "feat(adjudicate): wire adjudication params through self_play_worker"
```

---

### Task 5: Update ipc_messages.py docstring

**Files:**
- Modify: `scripts/GPU/alphazero/ipc_messages.py:63` (GameComplete.draw_reason docstring)

- [ ] **Step 1: Update the draw_reason docstring**

Change line 63 from:

```python
    draw_reason: int  # 0=none, 1=timeout, 2=board_full, 3=state_cap, 4=unknown
```

to:

```python
    draw_reason: int  # 0=none, 1=timeout, 2=board_full, 3=state_cap, 4=unknown, 5=resign, 6=adjudicated
```

- [ ] **Step 2: Commit**

```bash
git add scripts/GPU/alphazero/ipc_messages.py
git commit -m "docs(adjudicate): update GameComplete.draw_reason docstring with code 6"
```

---

### Task 6: Update game_saver.py reason mapping

**Files:**
- Modify: `scripts/GPU/alphazero/game_saver.py:67-76` (reason mapping)

- [ ] **Step 1: Update the reason mapping**

Change the existing block (lines 67-76) from:

```python
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

to:

```python
    if winner:
        # Winner exists - could be normal win, resignation, or adjudication
        if draw_reason == "resign":
            reason = "resign"
        elif draw_reason == "adjudicated":
            reason = "adjudicated"
        else:
            reason = "win"
    elif draw_reason:
        reason = draw_reason
    else:
        reason = "draw"
```

- [ ] **Step 2: Verify import works**

Run: `.venv/bin/python -c "from scripts.GPU.alphazero.game_saver import save_game_replay; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/GPU/alphazero/game_saver.py
git commit -m "feat(adjudicate): update game_saver reason mapping for adjudicated games"
```

---

### Task 7: Wire adjudication through trainer.py - parallel path

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py:955-974` (run_parallel_selfplay signature)
- Modify: `scripts/GPU/alphazero/trainer.py:1003` (imports)
- Modify: `scripts/GPU/alphazero/trainer.py:1057-1082` (worker spawn kwargs)
- Modify: `scripts/GPU/alphazero/trainer.py:1098-1127` (tracker vars)
- Modify: `scripts/GPU/alphazero/trainer.py:1130-1141` (nonlocal declarations)
- Modify: `scripts/GPU/alphazero/trainer.py:1145-1221` (process_stats_message)
- Modify: `scripts/GPU/alphazero/trainer.py:1327-1360` (stats dict)

- [ ] **Step 1: Add adjudication params to run_parallel_selfplay() signature**

After the resign params (line 973), add:

```python
    # Adjudication parameters
    adjudicate_enabled: bool = False,
    adjudicate_min_ply: int = 120,
    adjudicate_threshold: float = 0.90,
    adjudicate_min_visits: int = 200,
    adjudicate_min_top1_share: float = 0.0,
```

- [ ] **Step 2: Add ADJUDICATED to the import**

At line 1003, add `ADJUDICATED` to the import:

```python
    from .self_play import DRAW_TIMEOUT, DRAW_BOARD_FULL, DRAW_STATE_CAP, DRAW_UNKNOWN, RESIGN, ADJUDICATED
```

- [ ] **Step 3: Add adjudication params to worker spawn kwargs**

After the resign kwargs (line 1081), add:

```python
                # Adjudication parameters
                "adjudicate_enabled": adjudicate_enabled,
                "adjudicate_min_ply": adjudicate_min_ply,
                "adjudicate_threshold": adjudicate_threshold,
                "adjudicate_min_visits": adjudicate_min_visits,
                "adjudicate_min_top1_share": adjudicate_min_top1_share,
```

- [ ] **Step 4: Add adjudication counter variables**

After the resign tracking vars (line 1109), add:

```python
    # Adjudication tracking (decisive, not draw)
    adjudicated_games = 0
    adjudicated_red_wins = 0
    adjudicated_black_wins = 0
```

- [ ] **Step 5: Add nonlocal declarations**

In `process_stats_message()`, add to the nonlocal block (after line 1134):

```python
        nonlocal adjudicated_games, adjudicated_red_wins, adjudicated_black_wins
```

- [ ] **Step 6: Add adjudication tracking to process_stats_message()**

After the resign tracking block (around line 1174), add:

```python
            # Track adjudication (separate from draw breakdown, code 6)
            if msg.draw_reason == 6:
                adjudicated_games += 1
                if msg.winner == "red":
                    adjudicated_red_wins += 1
                elif msg.winner == "black":
                    adjudicated_black_wins += 1
```

- [ ] **Step 7: Update draw_reason_str mapping in game_saver block**

At line 1206, add `6: "adjudicated"` to the mapping dict:

```python
                draw_reason_str = {
                    0: None, 1: "timeout", 2: "board_full", 3: "state_cap", 4: "unknown", 5: "resign", 6: "adjudicated"
                }.get(msg.draw_reason)
```

No change needed for the `resigned_by` derivation block below (lines 1210-1212) -- adjudicated games have no `resigned_by`, so it stays `None` and `game_saver.maybe_save_game()` handles that correctly.

- [ ] **Step 8: Add adjudication to stats dict**

After the resign entries in the stats dict (line 1340), add:

```python
        "adjudicated_games": adjudicated_games,
        "adjudicated_red_wins": adjudicated_red_wins,
        "adjudicated_black_wins": adjudicated_black_wins,
```

- [ ] **Step 9: Verify syntax**

Run: `.venv/bin/python -c "from scripts.GPU.alphazero.trainer import run_parallel_selfplay; print('OK')"`
Expected: `OK`

- [ ] **Step 10: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py
git commit -m "feat(adjudicate): wire adjudication through parallel self-play path in trainer"
```

---

### Task 8: Wire adjudication through trainer.py - train() signature and sequential path

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py:1365-1425` (train() signature)
- Modify: `scripts/GPU/alphazero/trainer.py:1740-1760` (sequential counter init, near where `rg_` counters init)
- Modify: `scripts/GPU/alphazero/trainer.py:1781-1802` (parallel self-play call)
- Modify: `scripts/GPU/alphazero/trainer.py:1804-1836` (parallel stats unpack)
- Modify: `scripts/GPU/alphazero/trainer.py:1846-1940` (sequential self-play path)
- Modify: `scripts/GPU/alphazero/trainer.py:1968-1993` (print summary)

- [ ] **Step 1: Add adjudication params to train() signature**

After the resign params (line 1424), add:

```python
    # Adjudication parameters (disabled by default)
    adjudicate_enabled: bool = False,
    adjudicate_min_ply: int = 120,
    adjudicate_threshold: float = 0.90,
    adjudicate_min_visits: int = 200,
    adjudicate_min_top1_share: float = 0.0,
```

- [ ] **Step 2: Initialize adjudication counters in the per-iteration block**

Near where `resign_games = 0` is initialized (around line 1740-1760, in the per-iteration reset block), add:

```python
            adjudicated_games = 0
            adjudicated_red_wins = 0
            adjudicated_black_wins = 0
```

- [ ] **Step 3: Pass adjudication params to run_parallel_selfplay()**

After the resign kwargs in the parallel call (line 1801), add:

```python
                    adjudicate_enabled=adjudicate_enabled,
                    adjudicate_min_ply=adjudicate_min_ply,
                    adjudicate_threshold=adjudicate_threshold,
                    adjudicate_min_visits=adjudicate_min_visits,
                    adjudicate_min_top1_share=adjudicate_min_top1_share,
```

- [ ] **Step 4: Unpack adjudication stats from parallel path**

After the resign stats unpack (line 1816), add:

```python
                adjudicated_games = parallel_stats.get("adjudicated_games", 0)
                adjudicated_red_wins = parallel_stats.get("adjudicated_red_wins", 0)
                adjudicated_black_wins = parallel_stats.get("adjudicated_black_wins", 0)
```

- [ ] **Step 5: Pass adjudication params to sequential play_game()**

After the resign kwargs in the sequential call (line 1871), add:

```python
                        adjudicate_enabled=adjudicate_enabled,
                        adjudicate_min_ply=adjudicate_min_ply,
                        adjudicate_threshold=adjudicate_threshold,
                        adjudicate_min_visits=adjudicate_min_visits,
                        adjudicate_min_top1_share=adjudicate_min_top1_share,
```

- [ ] **Step 6: Add adjudication counting in sequential path**

After the resign tracking block (around line 1916), add:

```python
                    # Track adjudication (separate from draw, decisive)
                    if game.draw_reason == ADJUDICATED:
                        adjudicated_games += 1
                        if game.winner == "red":
                            adjudicated_red_wins += 1
                        elif game.winner == "black":
                            adjudicated_black_wins += 1
```

**IMPORTANT (import scope):** The sequential path uses `RESIGN` (line 1910) and `DRAW_TIMEOUT` etc. The import at line 1003 is inside `run_parallel_selfplay()` and does NOT help `train()`. There must be another import that brings these into `train()` scope -- find it (likely module-level in trainer.py) and add `ADJUDICATED` there. If no module-level import exists, add one:

```python
from .self_play import (
    DRAW_TIMEOUT, DRAW_BOARD_FULL, DRAW_STATE_CAP, DRAW_UNKNOWN, RESIGN, ADJUDICATED,
)
```

Prefer module-level import to avoid the same constant being imported in multiple function scopes.

- [ ] **Step 7: Add adjudication print line and gate diagnostics**

After the resign print (line 1975), add:

```python
            if adjudicated_games > 0 or (adjudicate_enabled and timeout_draws > 0):
                print(f"    Adjudicated: {adjudicated_games} (red_wins={adjudicated_red_wins}, black_wins={adjudicated_black_wins}, remaining_timeouts={timeout_draws})")
```

This shows both the adjudication count AND remaining timeouts, making threshold tuning much easier. If adjudication is enabled but adjudicated_games==0, you still see the line showing remaining timeouts. We don't claim "attempts" because not all timeouts are eligible (some may be below `adjudicate_min_ply`).

- [ ] **Step 8: Verify syntax**

Run: `.venv/bin/python -c "from scripts.GPU.alphazero.trainer import train; print('OK')"`
Expected: `OK`

- [ ] **Step 9: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py
git commit -m "feat(adjudicate): wire adjudication through train() and sequential self-play path"
```

---

### Task 9: Add CLI flags in train.py

**Files:**
- Modify: `scripts/GPU/alphazero/train.py:262-277` (after resign CLI block)
- Modify: `scripts/GPU/alphazero/train.py:334-350` (after resign validation block)
- Modify: `scripts/GPU/alphazero/train.py:391-448` (train() call kwargs)

- [ ] **Step 1: Add CLI argument definitions**

After the resign arguments block (after line 276), add:

```python
    # Adjudication parameters (disabled by default)
    parser.add_argument("--adjudicate-enabled", action="store_true",
        help="Enable timeout adjudication (assign winner at max_moves using MCTS eval)")
    parser.add_argument("--adjudicate-min-ply", type=int, default=120,
        help="Don't adjudicate before this ply (default: 120)")
    parser.add_argument("--adjudicate-threshold", type=float, default=0.90,
        help="Adjudicate when |root_value| >= this (default: 0.90)")
    parser.add_argument("--adjudicate-min-visits", type=int, default=200,
        help="Require root visits >= this to adjudicate (default: 200)")
    parser.add_argument("--adjudicate-min-top1-share", type=float, default=0.0,
        help="Require top move's visit share >= this to adjudicate (default: 0 = disabled)")
```

- [ ] **Step 2: Add validation**

After the resign validation block (after line 350), add:

```python
    # Validate adjudication parameters
    if args.adjudicate_min_ply < 0:
        parser.error("--adjudicate-min-ply must be >= 0")
    if not (0 <= args.adjudicate_threshold <= 1):
        parser.error("--adjudicate-threshold must be in [0, 1]")
    if args.adjudicate_min_visits < 1:
        parser.error("--adjudicate-min-visits must be >= 1")
    if not (0.0 <= args.adjudicate_min_top1_share <= 1.0):
        parser.error("--adjudicate-min-top1-share must be in [0, 1]")
```

- [ ] **Step 3: Pass args to train()**

After the resign kwargs in the `train()` call (around line 447), add:

```python
        # Adjudication parameters
        adjudicate_enabled=args.adjudicate_enabled,
        adjudicate_min_ply=args.adjudicate_min_ply,
        adjudicate_threshold=args.adjudicate_threshold,
        adjudicate_min_visits=args.adjudicate_min_visits,
        adjudicate_min_top1_share=args.adjudicate_min_top1_share,
```

- [ ] **Step 4: Verify CLI help works**

Run: `.venv/bin/python -m scripts.GPU.alphazero.train --help | grep adjudicate`
Expected: All 5 flags should appear.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/train.py
git commit -m "feat(adjudicate): add CLI flags for adjudication with validation"
```

---

### Task 10: Add adjudication tests to test_self_play.py

**Files:**
- Modify: `tests/test_self_play.py`

- [ ] **Step 1: Write test for adjudication with real game (loose threshold)**

Add before `main()`:

```python
def test_adjudication_wiring():
    """Test that adjudication parameters are accepted and produce valid outcomes."""
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.mcts import MCTSConfig
    from scripts.GPU.alphazero.self_play import play_game, ADJUDICATED, DRAW_TIMEOUT

    network = create_network(hidden=64, n_blocks=2)
    evaluator = LocalGPUEvaluator(network)
    config = MCTSConfig(n_simulations=10)

    # Play with adjudication enabled + permissive gates
    # Random untrained network may or may not cross threshold, so we test
    # structural correctness, not "must adjudicate"
    game = play_game(
        evaluator,
        mcts_config=config,
        max_moves=10,
        add_noise=False,
        adjudicate_enabled=True,
        adjudicate_min_ply=0,
        adjudicate_threshold=0.10,  # Very loose
        adjudicate_min_visits=1,
        adjudicate_min_top1_share=0.0,
    )

    # Game should complete without error
    assert game.n_moves > 0
    assert game.winner in ("red", "black", None)

    # Validate structural correctness based on what actually happened
    if game.draw_reason == ADJUDICATED:
        # Adjudicated: must have a decisive winner and ±1 outcomes
        assert game.winner in ("red", "black"), "Adjudicated game must have a winner"
        for pos in game.positions:
            assert pos.outcome in (1.0, -1.0), f"Adjudicated game should have ±1 outcomes, got {pos.outcome}"
    elif game.draw_reason == DRAW_TIMEOUT:
        # Timeout draw: winner must be None, outcomes must be 0.0
        assert game.winner is None, "Timeout draw should have no winner"
        for pos in game.positions:
            assert pos.outcome == 0.0, f"Timeout draw should have 0.0 outcomes, got {pos.outcome}"
    else:
        # Early terminal win is also valid (game ended before cap)
        assert game.winner in ("red", "black", None)

    print(f"PASS: Adjudication wiring (moves={game.n_moves}, winner={game.winner}, reason={game.draw_reason})")
    return True


def test_adjudication_disabled_by_default():
    """Test that adjudication is off by default (no change to existing behavior)."""
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.mcts import MCTSConfig
    from scripts.GPU.alphazero.self_play import play_game, ADJUDICATED

    network = create_network(hidden=64, n_blocks=2)
    evaluator = LocalGPUEvaluator(network)
    config = MCTSConfig(n_simulations=10)

    # Play without adjudication (default)
    game = play_game(
        evaluator,
        mcts_config=config,
        max_moves=10,
        add_noise=False,
    )

    # Should never have adjudicated reason when disabled
    assert game.draw_reason != ADJUDICATED, "Adjudication should not fire when disabled"

    print(f"PASS: Adjudication disabled by default (reason={game.draw_reason})")
    return True
```

- [ ] **Step 2: Add tests to the main() test list**

In `main()`, add to the `tests` list:

```python
        test_adjudication_wiring,
        test_adjudication_disabled_by_default,
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python tests/test_self_play.py`
Expected: All tests PASS (including the 2 new adjudication tests).

- [ ] **Step 4: Commit**

```bash
git add tests/test_self_play.py
git commit -m "test(adjudicate): add adjudication wiring and disabled-by-default tests"
```

---

### Task 11: Update docs/train-cli.md

**Files:**
- Modify: `docs/train-cli.md:140-142` (between Resign and Mirror Augmentation sections)

- [ ] **Step 1: Add Adjudication section**

After the Resign section's last line (line 140), before `## Mirror Augmentation` (line 142), insert:

```markdown

## Adjudication

Converts timeout draws into decisive outcomes when the final position is clearly won/lost.
When a game hits `max_moves` without a terminal winner, one final deterministic MCTS search evaluates the position.
If the root value exceeds the threshold (and confidence gates pass), a winner is assigned.
Disabled by default (conservative).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--adjudicate-enabled` | flag | off | Enable timeout adjudication |
| `--adjudicate-min-ply` | int | 120 | Don't adjudicate before this ply (same unit as max_moves) |
| `--adjudicate-threshold` | float | 0.90 | Adjudicate when |root_value| >= this |
| `--adjudicate-min-visits` | int | 200 | Require root visits >= this |
| `--adjudicate-min-top1-share` | float | 0.0 | Require top move's visit share >= this (0 = disabled) |

**Units**: `adjudicate-min-ply` is measured in plies (one ply = one player's move), the same unit as `max-moves`.

**Example**: `--adjudicate-enabled --adjudicate-threshold 0.92 --adjudicate-min-ply 160`

**Validation**: min_ply >= 0, threshold in [0, 1], min_visits >= 1, top1_share in [0, 1]

**Note**: Adjudication uses the last MCTS root (tree-reused from the final game move) with `add_noise=False` for a deterministic evaluation. The root_value is from `state.to_move` perspective: if `root_value >= +T`, current player wins; if `root_value <= -T`, opponent wins.

**Training impact**: Adjudicated games produce decisive outcomes (winner != draw), so positions get +/-1 value labels instead of 0. This strengthens the value target distribution and reduces timeout draws without raising max_moves. In saved replays, `meta.reason = "adjudicated"` distinguishes these from normal wins.
```

- [ ] **Step 2: Commit**

```bash
git add docs/train-cli.md
git commit -m "docs(adjudicate): add Adjudication section to train-cli.md"
```

---

### Task 12: End-to-end smoke test

**Files:** None (verification only)

- [ ] **Step 1: Run with adjudication disabled (baseline)**

Run:
```bash
.venv/bin/python -m scripts.GPU.alphazero.train \
  --iterations 1 --games-per-iter 4 --train-steps 1 \
  --simulations 50 --n-workers 1 --seed 42
```

Expected: No "Adjudicated" line in output. Normal behavior.

- [ ] **Step 2: Run with adjudication enabled (loose threshold to force firing)**

Run:
```bash
.venv/bin/python -m scripts.GPU.alphazero.train \
  --iterations 1 --games-per-iter 4 --train-steps 1 \
  --simulations 50 --n-workers 1 --seed 42 \
  --adjudicate-enabled --adjudicate-threshold 0.10 \
  --adjudicate-min-ply 0 --adjudicate-min-visits 1
```

Expected: Some games show `ADJUDICATED:` diagnostic line. Iteration summary shows `Adjudicated: N` line with N > 0.

- [ ] **Step 3: Verify CLI validation**

Run: `.venv/bin/python -m scripts.GPU.alphazero.train --adjudicate-threshold 1.5`
Expected: Error message about threshold.

Run: `.venv/bin/python -m scripts.GPU.alphazero.train --adjudicate-min-visits 0`
Expected: Error message about min_visits.
