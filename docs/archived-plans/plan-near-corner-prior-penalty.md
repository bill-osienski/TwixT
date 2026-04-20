# Near-Corner Prior Penalty (Combined Root Shaping)

**Status:** Implemented

## Goal

Add near-corner penalty to complement the existing edge-band penalty. Both penalties will be applied in a single combined loop post-Dirichlet-noise with one renormalization. Use **max-penalty** approach (no double-penalizing overlaps) as the safer default.

---

## Files Modified

1. `scripts/GPU/alphazero/mcts.py` — helper function + MCTSConfig fields + combined penalty block
2. `scripts/GPU/alphazero/train.py` — 3 CLI flags + validation
3. `scripts/GPU/alphazero/trainer.py` — 3 params in `train()` signature + add to overrides dict
4. `docs/train-cli.md` — document new flags

---

## Implementation

### 1. Helper function for near-corner (mcts.py, next to `_is_edge_band`)

```python
def _is_near_corner_cheb(r: int, c: int, S: int, R: int) -> bool:
    """True if (r,c) is within Chebyshev distance <= R of any corner on an SxS board."""
    if R <= 0:
        return False
    corners = ((0, 0), (0, S - 1), (S - 1, 0), (S - 1, S - 1))
    for rr, cc in corners:
        if max(abs(r - rr), abs(c - cc)) <= R:
            return True
    return False
```

### 2. MCTSConfig — new fields (after edge-band fields)

```python
    # Near-corner prior penalty (root_near_corner_penalty=0 disables)
    root_near_corner_penalty: float = 0.0     # λc in exp(-λc)
    root_near_corner_penalty_ply: int = 0     # apply for ply < this
    root_near_corner_radius: int = 2          # Chebyshev radius R
```

### 3. Replace edge-band penalty block with combined block (mcts.py)

Replace the current edge-band penalty block with:

```python
        # --- Root prior shaping (post-noise): near-corner + edge-band (plies < ply limits) ---
        S = BOARD_W  # 24

        edge_pen = self.config.root_edge_band_penalty
        edge_ply = self.config.root_edge_band_penalty_ply
        band = self.config.root_edge_band_width

        corner_pen = self.config.root_near_corner_penalty
        corner_ply = self.config.root_near_corner_penalty_ply
        R = self.config.root_near_corner_radius

        apply_edge = edge_pen > 0.0 and edge_ply > 0 and ply < edge_ply
        apply_corner = corner_pen > 0.0 and corner_ply > 0 and ply < corner_ply and R > 0

        if apply_edge or apply_corner:
            edge_mult = math.exp(-edge_pen) if apply_edge else 1.0
            corner_mult = math.exp(-corner_pen) if apply_corner else 1.0

            total = 0.0
            edge_count = 0
            corner_count = 0
            edge_mass = 0.0
            corner_mass = 0.0

            # IMPORTANT: iterate stable ids; don't iterate dict view while mutating
            for mid in move_ids:
                p = root.priors[mid]
                r, c = decode_move(mid)

                in_edge = apply_edge and _is_edge_band(r, c, S, band)
                in_corner = apply_corner and _is_near_corner_cheb(r, c, S, R)

                # Max-penalty (no double-penalize overlaps)
                mult = 1.0
                if in_edge:
                    mult = min(mult, edge_mult)
                if in_corner:
                    mult = min(mult, corner_mult)

                if mult != 1.0:
                    p *= mult
                    root.priors[mid] = p

                total += p

                if in_edge:
                    edge_count += 1
                    edge_mass += p
                if in_corner:
                    corner_count += 1
                    corner_mass += p

            if total > 1e-12:
                inv = 1.0 / total
                for mid in move_ids:
                    root.priors[mid] *= inv
                edge_mass *= inv
                corner_mass *= inv

            if _OPENDBG:
                if apply_edge:
                    print(f"[EDGEBAND] ply={ply}: {edge_count}/{len(move_ids)} in band, mass={edge_mass:.3f}, penalty={edge_pen}, B={band}")
                if apply_corner:
                    print(f"[NEARCORNER] ply={ply}: {corner_count}/{len(move_ids)} in R, mass={corner_mass:.3f}, penalty={corner_pen}, R={R}")
        # --- end root prior shaping ---
```

### 4. CLI flags (train.py, after edge-band flags)

```python
    # Near-corner prior penalty
    parser.add_argument("--root-near-corner-penalty", type=float, default=None,
        help="Near-corner prior penalty λ. If set, prior *= exp(-λ) for near-corner moves")
    parser.add_argument("--root-near-corner-penalty-ply", type=int, default=None,
        help="Apply near-corner penalty for ply < this value")
    parser.add_argument("--root-near-corner-radius", type=int, default=None,
        help="Near-corner Chebyshev radius (MCTSConfig default: 2)")
```

### 5. Validation (train.py, after edge-band validation)

```python
    # Validate near-corner penalty
    if args.root_near_corner_penalty is not None and args.root_near_corner_penalty < 0:
        parser.error("--root-near-corner-penalty must be >= 0")
    if args.root_near_corner_penalty_ply is not None and args.root_near_corner_penalty_ply < 0:
        parser.error("--root-near-corner-penalty-ply must be >= 0")
    if args.root_near_corner_radius is not None and args.root_near_corner_radius < 1:
        parser.error("--root-near-corner-radius must be >= 1")
    if args.root_near_corner_radius is not None and args.root_near_corner_radius >= 12:
        parser.error("--root-near-corner-radius must be < 12 for a 24x24 board")
```

### 6. Pass to train() (train.py, after edge-band params)

```python
        # Near-corner prior penalty
        root_near_corner_penalty=args.root_near_corner_penalty,
        root_near_corner_penalty_ply=args.root_near_corner_penalty_ply,
        root_near_corner_radius=args.root_near_corner_radius,
```

### 7. trainer.py — add to train() signature (after edge-band params)

```python
    # Near-corner prior penalty (None = use MCTSConfig defaults)
    root_near_corner_penalty: Optional[float] = None,
    root_near_corner_penalty_ply: Optional[int] = None,
    root_near_corner_radius: Optional[int] = None,
```

### 8. trainer.py — add to mcts_exploration_overrides dict

```python
    if root_near_corner_penalty is not None:
        mcts_exploration_overrides["root_near_corner_penalty"] = root_near_corner_penalty
    if root_near_corner_penalty_ply is not None:
        mcts_exploration_overrides["root_near_corner_penalty_ply"] = root_near_corner_penalty_ply
    if root_near_corner_radius is not None:
        mcts_exploration_overrides["root_near_corner_radius"] = root_near_corner_radius
```

### 9. docs/train-cli.md — add new section (after Edge-Band section)

```markdown
## Near-Corner Prior Penalty

Applies a multiplicative penalty to near-corner moves in the root prior for plies < N.
Uses Chebyshev distance (max of row/col distance) to determine corner proximity.
Disabled by default.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--root-near-corner-penalty` | float | None (MCTSConfig: 0) | Penalty strength λ. Prior is multiplied by exp(-λ) for near-corner moves |
| `--root-near-corner-penalty-ply` | int | None (MCTSConfig: 0) | Apply penalty for plies < this value |
| `--root-near-corner-radius` | int | None (MCTSConfig: 2) | Chebyshev radius R. A cell is near-corner if max(|r-corner_r|, |c-corner_c|) <= R for any corner |

**Example**: `--root-near-corner-penalty 0.25 --root-near-corner-penalty-ply 6 --root-near-corner-radius 2`

**Validation**: penalty ≥ 0, ply ≥ 0, 1 ≤ radius < 12

**Note**: When both edge-band and near-corner penalties are active, overlapping cells receive the **max penalty** (not double-penalized).
```

---

## Verification

1. **Compile check**:
   ```bash
   python3 -m py_compile scripts/GPU/alphazero/mcts.py scripts/GPU/alphazero/train.py scripts/GPU/alphazero/trainer.py
   ```

2. **Unit test**:
   ```bash
   .venv/bin/python -m pytest tests/test_mcts.py -v
   ```

3. **Smoke run** with both penalties enabled:
   ```bash
   .venv/bin/python -m scripts.GPU.alphazero.train \
     --iterations 2 \
     --games-per-iter 4 \
     --train-steps 1 \
     --simulations 50 \
     --n-workers 1 \
     --seed 42 \
     --opening-debug \
     --root-edge-band-penalty 0.25 \
     --root-edge-band-penalty-ply 4 \
     --root-edge-band-width 2 \
     --root-near-corner-penalty 0.25 \
     --root-near-corner-penalty-ply 6 \
     --root-near-corner-radius 2
   ```
   Confirm: Both `[EDGEBAND]` and `[NEARCORNER]` log lines appear.

---

## Key Design Decisions

1. **Max-penalty (not product)**: Overlapping cells get `min(edge_mult, corner_mult)` — avoids double-whacking cells that are both edge-band and near-corner.

2. **Post-noise application**: Both penalties apply after Dirichlet noise, consistent with current edge-band behavior.

3. **Single loop, single renormalization**: More efficient and avoids accumulating rounding errors.

4. **Chebyshev distance**: Uses `max(|dr|, |dc|)` rather than Manhattan — creates square corner regions rather than diamond shapes.

5. **Stable iteration**: Uses `move_ids` (snapshot list) for iteration, not `root.priors.items()`, to avoid dict mutation during iteration and maintain consistent ordering with Dirichlet noise application.

6. **Overlap note**: Near-corner (R=2) is NOT a strict subset of edge-band (B=2). Example: (2,2) is near-corner but not in edge-band. Some overlap exists at actual corners, but they're distinct regions.
