# MLX Memory Management for AlphaZero Training

This document describes memory management strategies implemented to prevent Metal resource exhaustion during AlphaZero self-play and training on Apple Silicon.

## Problem Summary

When running AlphaZero training with MLX on Apple Silicon, we encountered:

```
RuntimeError: [metal::malloc] Resource limit (499000) exceeded.
```

This error occurs when Metal's resource handle table fills up, typically from:
1. Accumulating many small tensor allocations in tight loops
2. BatchNorm layers updating running statistics during inference
3. Not releasing MLX array references promptly

The crash happened in two places:
- **Training**: Loss function looping over 64 positions, each creating many tensors
- **Self-play**: MCTS running thousands of forward passes per game

## Fixes Implemented

### 1. Vectorized Training Loss (`trainer.py`)

**Problem**: Original `alphazero_loss()` looped over each position in the batch, creating ~30,000 small tensor allocations per training step.

**Solution**: Replaced with `alphazero_loss_batch()` using padded tensors:

```python
def make_padded_batch(positions, max_moves_cap=512):
    """Prepare batched tensors with padded moves."""
    # Stack all boards: (B, H, W, C)
    # Pad moves to uniform length M
    # Return: boards, move_rows, move_cols, move_mask, target_pi, outcomes

def alphazero_loss_batch(network, positions, l2_weight=1e-4, max_moves_cap=512):
    """Single batched forward pass instead of per-position loop."""
    boards, move_rows, move_cols, move_mask, target_pi, outcomes = make_padded_batch(positions)
    logits, values = network.forward_padded(boards, move_rows, move_cols, move_mask)
    # Compute loss in one graph
```

**Result**: Training reduced from ~30,000 allocations to ~10 per step.

### 2. Vectorized PolicyHead (`network.py`)

**Problem**: Original `PolicyHead.__call__()` looped over 200-480 moves per position.

**Solution**: Added `forward_padded()` with vectorized gather:

```python
class PolicyHead(nn.Module):
    def __init__(self, in_channels=128, hidden=64):
        ...
        self._neg_inf_f32 = mx.array(NEG_INF)  # Pre-allocate constant

    def forward_padded(self, features, move_rows, move_cols, move_mask):
        x = nn.relu(self.bn(self.conv(features)))

        B, M = move_rows.shape
        if B == 1:
            # Fast path for MCTS: avoid mx.arange allocation
            gathered = x[0, move_rows[0], move_cols[0], :]
            h = nn.relu(self.fc(gathered))
            logits = self.out(h).squeeze(-1)
            logits = mx.where(move_mask[0] > 0.5, logits, self._neg_inf_f32)
            return logits[None, :]

        # General B>1 path for training
        b_idx = mx.arange(B, dtype=move_rows.dtype)[:, None]
        gathered = x[b_idx, move_rows, move_cols, :]
        ...
```

**Key optimizations**:
- B==1 fast path avoids `mx.arange()` allocation per call
- Pre-allocated `_neg_inf_f32` constant avoids per-call allocation
- Single vectorized gather instead of loop

### 3. Eval/Train Mode Switching (`trainer.py`)

**Problem**: BatchNorm updates running statistics on every forward pass in train mode. During MCTS (thousands of forwards), this accumulates Metal resources.

**Solution**: Switch to eval mode during self-play:

```python
for iteration in range(n_iterations):
    # Self-play: freeze BN behavior
    network.eval()
    for g in range(games_per_iteration):
        game = play_game(network, ...)

    # Training: BN updates enabled
    network.train()
    for step in range(train_steps):
        loss = train_step(network, ...)
```

### 4. Cache Clearing Cadence in MCTS (`mcts.py`)

**Problem**: Even with optimizations, a single MCTS search (50-200 simulations) accumulates resources over thousands of `_expand()` calls.

**Solution**: Clear cache every N expands:

```python
class MCTS:
    def __init__(self, ...):
        ...
        self._expand_calls = 0

    def _expand(self, node):
        self._expand_calls += 1
        ...
        # After converting to Python and deleting MLX arrays
        del board, policy_logits, value, priors

        # Drain Metal resources periodically
        if (self._expand_calls % 32) == 0:
            mx.clear_cache()
```

**Tuning**: 32 is a balance between stability and performance. Increase if stable, decrease if still crashing.

### 5. Explicit Array Cleanup in MCTS (`mcts.py`)

**Problem**: MLX arrays held in Python variables keep Metal resources alive.

**Solution**: Convert to Python immediately and delete references:

```python
def _expand(self, node):
    policy_logits, value = self.network(board, moves)
    priors = self._stable_softmax(policy_logits)
    mx.eval(priors, value)  # Single sync point

    # Bulk transfer to Python (avoids N sync points from .item() loop)
    priors_list = priors.tolist()
    node.priors = {m: float(p) for m, p in zip(moves, priors_list)}
    node.nn_value = float(value.item())

    # Explicitly drop references
    del board, policy_logits, value, priors
```

### 6. Game-Level Cleanup (`trainer.py`, `self_play.py`)

**Solution**: After each game, clean up Python references and Metal cache:

```python
game = play_game(network, ...)
gc.collect()
mx.clear_cache()
```

## Configuration

### Cache Limit

Set in `trainer.py`:
```python
mx.set_cache_limit(2 * 1024 * 1024 * 1024)  # 2GB default
```

This limits MLX's cache size but doesn't prevent resource handle exhaustion.

### Memory Telemetry

Enabled during training to monitor memory:
```python
active_mb = mx.get_active_memory() / (1024 * 1024)
cache_mb = mx.get_cache_memory() / (1024 * 1024)
print(f"GPU: {active_mb:.0f}MB active, {cache_mb:.0f}MB cache")
```

## Remaining Considerations

### If Still Crashing

If crashes persist with larger networks (128 hidden, 6 blocks), consider:

1. **BN Fusion for Inference**: Fold BatchNorm into Conv weights for inference path
2. **Replace BatchNorm with GroupNorm**: GroupNorm has no running statistics
3. **Reduce Network Size**: Use 64 hidden, 4 blocks (known stable)
4. **Reduce MCTS Simulations**: 50 instead of 200

### Performance vs Stability Tradeoffs

| Setting | Stable | Fast | Notes |
|---------|--------|------|-------|
| Cache clear every 16 expands | More | Less | For unstable configs |
| Cache clear every 32 expands | Good | Good | Default |
| Cache clear every 64 expands | Less | More | For stable configs |
| gc.collect() every game | More | Less | Helps with Python refs |
| gc.collect() every N games | Less | More | After stabilizing |

## File Changes Summary

| File | Changes |
|------|---------|
| `scripts/GPU/alphazero/network.py` | Added `forward_padded()`, B==1 fast path, pre-allocated constants |
| `scripts/GPU/alphazero/trainer.py` | Vectorized loss, eval/train switching, gc.collect(), telemetry |
| `scripts/GPU/alphazero/mcts.py` | Cache clearing cadence, explicit del, bulk tolist() transfer |
| `scripts/GPU/alphazero/self_play.py` | gc.collect() + mx.clear_cache() after games |

## References

- MLX Memory Management: `mx.clear_cache()`, `mx.set_cache_limit()`, `mx.get_active_memory()`
- Metal Resource Limits: Error `[metal::malloc] Resource limit (N) exceeded`
- BatchNorm in Inference: Use `model.eval()` to freeze running statistics
