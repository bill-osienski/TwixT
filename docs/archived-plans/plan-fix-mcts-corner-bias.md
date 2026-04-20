# Fix MCTS Corner Bias - Random Tie-Breaking

## Status: IMPLEMENTED

## Problem
The MCTS implementation has deterministic tie-breaking that causes corner/edge bias:
1. `select_move()` uses lexicographic sort when visit counts tie
2. `_expand_batch()` creates priors dict in row-major order (insertion order preserved)
3. `_select_child()` uses "first wins" when PUCT scores tie

All three cause (0,0), (0,1), etc. to be favored when scores are equal.

---

## File Modified
`scripts/GPU/alphazero/mcts.py`

---

## Fix #1: Random Tie-Break in `select_move()`

**What changed:**
- Added defensive guard at top: fail fast if called with empty `visit_counts`
- Replaced lexicographic sort tie-break with `self.rng.choice(best_moves)`
- Uses per-game RNG for reproducibility per seed

---

## Fix #2: Randomize Move Ordering in `_expand_batch()`

**What changed:**
- Shuffle `(move_id, prior)` pairs before building the priors dict
- This randomizes dict insertion order, preventing row-major bias when iterating priors
- Uses `self.rng` for reproducibility per seed
- NOTE: Shuffles pairs (not `moves_id` alone) to keep priors aligned with move_ids

---

## Fix #3: Random Tie-Break in `_select_child()`

**What changed:**
- Added two early asserts: `is_expanded` then `priors` for clearer error messages
- `eps = 1e-8` catches "effectively tied" scores from float jitter
- Collects all tied moves into `best_moves` list instead of "first wins"
- Uses `self.rng.choice()` (per-game RNG) for reproducible random tie-breaking
- `child` can still be `None` (lazy creation) — callers already handle this

---

## Test Updates (`tests/test_mcts.py`)

- Renamed `test_lexicographic_tiebreak` → `test_random_tiebreak`: asserts move is any of the tied moves
- Fixed all 7 pre-existing test failures: wrapped `AlphaZeroNetwork` in `LocalGPUEvaluator`
- Fixed `test_puct_formula`: manually creates children (lazy creation means `_expand` doesn't)
- Fixed `test_deterministic_selection`: tests seed reproducibility instead of repeated-call identity

---

## Expected Behavior After Fixes

- **Within an iteration:** Openings should diversify (no longer "all same corner")
- **Reproducibility preserved:** Same game_id + seed → same game (per-game RNG)
- **No performance impact:** Random tie-break is O(k) where k = number of tied moves
