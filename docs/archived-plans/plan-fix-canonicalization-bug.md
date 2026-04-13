# Plan: Fix Canonicalization Active Region Bug

## Problem Summary

**Root cause identified:** Board rotation uses full 24×24, but move coord rotation uses `active_size`.

```python
# Board rotation (current - WRONG):
boards_rot = mx.transpose(boards, (0, 2, 1, 3))  # swap H,W
boards_rot = boards_rot[:, :, ::-1, :]           # reverse -> (c, 23-r)

# Move rotation (correct):
cols_rot = active_size - 1 - move_rows           # -> (c, active_size-1-r)
```

For a peg at (4, 2) with `active_size=8`:
- Board rotation: (4,2) → (2, **19**) - uses full 24
- Move rotation: (4,2) → (2, **3**) - uses active_size=8

Result: Policy head gathers features at (2,3) but the peg is at (2,19). Network sees empty space where there should be pegs. This corrupts black-to-move positions whenever `active_size < board_size` (i.e., during curriculum), explaining the 87% Red-dominant iterations.

---

## Changes

### 1. Fix `canonicalize_batch()` - Rotate Only Active Region

**File:** `scripts/GPU/alphazero/network.py`
**Location:** Lines 106-110 (Step 1 rotation)

**Replace:**
```python
    # Step 1: Rotate spatial planes 90° CW for entire batch
    # CW rotation: (r, c) -> (c, N-1-r)
    # Implementation: transpose (H,W)->(W,H), then reverse axis=2 (the new "row" axis) -> CW
    boards_rot = mx.transpose(boards, (0, 2, 1, 3))  # (B, W, H, C) - swaps H and W
    boards_rot = boards_rot[:, :, ::-1, :]  # reverse along axis=2 -> gives N-1-r
```

**With:**
```python
    # Step 1: Rotate ONLY the active region 90° CW: (r, c) -> (c, S-1-r)
    # This ensures board rotation matches move coord rotation (both use active_size)
    S = active_size

    active = boards[:, :S, :S, :]                  # (B, S, S, C)
    active_rot = mx.transpose(active, (0, 2, 1, 3))  # swap H,W within active
    active_rot = active_rot[:, :, ::-1, :]         # CW within SxS

    # Paste rotated active region back into HxW tensor (zeros elsewhere)
    B, H, W, C = boards.shape

    if W > S:
        right_pad = mx.zeros((B, S, W - S, C), dtype=boards.dtype)
        top = mx.concatenate([active_rot, right_pad], axis=2)
    else:
        top = active_rot

    if H > S:
        bottom = mx.zeros((B, H - S, W, C), dtype=boards.dtype)
        boards_rot = mx.concatenate([top, bottom], axis=1)
    else:
        boards_rot = top
```

**Keep Step 2+ exactly as-is:** swap pegs, permute links, remap distances, force CH_TO_MOVE=1.

**Rationale:** Zeros outside active region prevents phantom signals from leaking into inference/training.

---

### 2. Verify Fix with Diagnostic

**File:** `scripts/GPU/alphazero/network.py`

The diagnostic prints two locations:
1. **Example move** (first legal move) - may be empty (0.0/0.0) if that cell has no pegs
2. **Occupied peg** (first peg found) - **must** show the peg at correct rotated position

After the fix, the "OCCUPIED" line should change from:
```
OCCUPIED orig[4,2]: black=0.0, red=1.0
OCCUPIED canon[2,3]: cur(ch0)=0.0, opp(ch1)=0.0  ← BUG (peg vanished)
```

To:
```
OCCUPIED orig[4,2]: black=0.0, red=1.0
OCCUPIED canon[2,3]: cur(ch0)=0.0, opp(ch1)=1.0  ← FIXED (red peg now visible as opp)
WRONG_LOC canon[2,19]: cur(ch0)=0.0, opp(ch1)=0.0  ← Confirms old bug location is now empty
```

**Enhancement:** Add diagnostic line for the "wrong" location. Compute exactly as the buggy code did:
- `wrong_r = c`
- `wrong_c = (H - 1) - r` (using original board H, not active_size)

This ensures the diagnostic catches the same bug class even if board size changes.

---

### 3. Add Unit-Style Invariants (Recommended)

**File:** `tests/test_canonicalization.py` (new file)

**Invariant 0: CH_TO_MOVE forced to 1 after canonicalization (cheap, catches regressions)**
```python
def test_to_move_forced_to_one():
    """After canonicalization, CH_TO_MOVE must be 1.0 everywhere (model assumes canonical red)."""
    # For both red-to-move and black-to-move boards:
    # - Canonicalize
    # - Assert boards_out[..., CH_TO_MOVE] == 1.0 throughout active region
    # Catches accidental regressions where ones18 isn't applied.
```

**Invariant 1: Active-region peg count preserved**
```python
def test_peg_count_preserved():
    """Peg count in [0:S,0:S] must match before/after canonicalization."""
    # For black-to-move boards:
    # - count pegs in [0:S,0:S] before
    # - count pegs in [0:S,0:S] after (in swapped channels)
    # - they must match
```

**Invariant 2: Rotated moves stay in active region**
```python
def test_rotated_moves_in_bounds():
    """All rotated moves must satisfy 0 <= r' < S and 0 <= c' < S."""
    # For black-to-move:
    # - all (r', c') must be within [0, active_size)
```

**Invariant 3: Single-peg coordinate mapping (HIGHEST VALUE - would have caught this bug)**
```python
def test_single_peg_coordinate_mapping():
    """Verify peg at (r,c) appears at (c, S-1-r) after canonicalization."""
    # Test multiple coordinates to catch off-by-one errors:
    # - center-ish: (4, 2)
    # - near edges (but legal): (1, 5), (6, 1)
    # - 2-3 random points within active region
    #
    # For each:
    # - Create empty board with single red peg at (r, c)
    # - IMPORTANT: Set CH_TO_MOVE=0 (black-to-move) to trigger rotation
    # - Sanity-check that is_black becomes True inside canonicalize_batch
    # - Canonicalize
    # - Assert peg appears at (c, S-1-r) in opp_peg channel (ch1 after swap)
    # - Assert original location (r, c) in canonical board is empty
    # - Assert WRONG location (c, H-1-r) is also empty (catches full-board rotation bug)
```

**Invariant 4: "Gather sees the peg" integration test (end-to-end sanity)**
```python
def test_gather_sees_peg():
    """Policy gather at rotated move coord must see the actual peg features."""
    # This tests the exact assumption that broke: board and move transforms must match.
    # - Build a black-to-move board with one known peg at (r, c) inside active region
    # - Build moves=[(r, c)] with just that coordinate
    # - Run forward_padded() (or canonicalize_batch + policy head gather path)
    # - Assert the feature gathered at rotated move coord is non-zero in expected peg channel
    # Fast, deterministic, prevents future regressions if rotation code changes.
```

---

### 4. Sanity Check Link Permutation (5-minute check)

After fixing board rotation, verify link channels aren't accidentally broken:
- Build a state with one known bridge direction (e.g., `(dr,dc)=(2,1)` = channel 0)
- Canonicalize black-to-move
- Confirm it lands in the channel predicted by `LINK_PERM_CW`
- **Test BOTH endpoints:** Encoding marks source in `dir_offset` and dest in `rev_dir_offset`. Verify both endpoints land in expected permuted channels after rotation (catches "half right" permutation bugs).

This ensures the link permutation logic was correctly designed for active-region rotation.

---

### 5. Remove Diagnostic After Verification

Once the fix is confirmed working, remove or disable the `_CANON_DIAG_LOGGED` diagnostic code to avoid performance overhead in production training.

---

## Verification

### Step 1: Run diagnostic test
```bash
.venv/bin/python -u -m scripts.GPU.alphazero.train \
    --iterations 1 --games-per-iter 2 --train-steps 10 \
    --checkpoint-dir /tmp/canon-fix-test 2>&1 | grep -A8 "CANON_DIAG"
```

**Expected output after fix:**
```
CANON_DIAG: active_size=8
  move_orig=(X,Y) -> move_rot=(Y',X')
  orig_board[X,Y]: black_peg=0.0, red_peg=0.0
  canon_board[Y',X']: cur_peg(ch0)=0.0, opp_peg(ch1)=0.0  ← empty move cell, fine
  OCCUPIED orig[4,2]: black=0.0, red=1.0
  OCCUPIED canon[2,3]: cur(ch0)=0.0, opp(ch1)=1.0  ← KEY: red peg visible as opp
```

### Step 2: Small self-play experiment (20-50 iterations)
```bash
.venv/bin/python -u -m scripts.GPU.alphazero.train \
    --iterations 50 --games-per-iter 25 --train-steps 100 \
    --checkpoint-dir checkpoints/canon-fix-test
```

**Success criteria (realistic expectations):**
- Black win% should materially increase (e.g., ~22% → >30-35%)
- "Iterations where Red won more" should fall from 87% toward ~50%
- May not reach perfect 50/50 in 50 iters due to curriculum/exploration effects

**Note:** First few iterations may show volatility (even worse draw rate or weird imbalance) because network weights were trained on corrupted black positions and now suddenly see correct ones. This is normal. Look at the **trend** over ~20-50 iters.

### Step 3: Check iteration-level balance
```bash
# After 50 iterations, check win ratios
cat checkpoints/canon-fix-test/games.md | tail -20
```

---

## Success Criteria

1. **Immediate:** Diagnostic shows `opp(ch1)=1.0` for occupied cells after rotation (not 0.0)
2. **Short-term:** Black win% rises materially (>30%) in self-play
3. **Medium-term:** Iteration-level Red dominance drops from 87% toward balanced
4. **Long-term:** Red/Black balance approaches 50/50 over extended training

---

## Notes

**active_size is per-batch:** Currently `canonicalize_batch()` takes `active_size: int`, assuming all samples in the batch share the same size (true for curriculum per-iteration). If mixed sizes are ever needed, this would need to become per-sample. For now: fine as-is.

**All channels rotated:** The slice `boards[:, :S, :S, :]` includes pegs, links, to_move, distances, phase. That's correct. (CH_TO_MOVE is overwritten to ones anyway.)

**Value head and zeros:** The value head already uses `active_size` for masked pooling, so zeros outside the active region are handled correctly. No additional changes needed.

**MLX slice assignment (PREFERRED if supported):**
```python
boards_rot = mx.zeros_like(boards)
boards_rot = boards_rot.at[:, :S, :S, :].set(active_rot)  # or similar MLX idiom
```
This is simpler and avoids two classic concat-pad footguns:
- Mixing up axis 1 vs 2 on padding
- Accidentally padding with wrong dtype/shape when C changes

If MLX doesn't support clean indexed set, concat-pad is correct but ensure:
- **Width/right pad on axis=2** (the W dimension)
- **Height/bottom pad on axis=1** (the H dimension)

**Diagnostic should print H, W, S** once to make wrong-loc math easy to verify.

**Distance planes rotate correctly:** The active slice includes CH_RED_TOP/BOTTOM and CH_BLACK_LEFT/RIGHT. These are spatial (distance-to-edge per cell) and should rotate with the board. Step 2 then remaps them (black-left→top, etc.). This is consistent.

**Code path guardrails:**
- Both training and MCTS inference use the same canonicalization (via `forward_padded()`) - verified correct.
- CH_TO_MOVE detection uses `< 0.5` threshold. If there's ever noisy floats, clamp when generating tensors. (Not currently an issue.)

---

## Cleanup After Verification

1. Remove `_CANON_DIAG_LOGGED` flag and diagnostic block from `network.py`
2. Delete temporary checkpoint directories (`/tmp/canon-*`)
