# Two-Optimizer Module Update Pattern

## Problem

When using two separate optimizers in MLX (e.g., different learning rates for encoder+policy vs value head), passing sliced parameter dicts to `optimizer.update()` may not reliably mutate the original network:

```python
# RISKY: May update detached tree!
params = network.parameters()
main_params = {"encoder": params["encoder"], "policy_head": params["policy_head"]}
opt_main.update(main_params, main_grads)
```

## Solution: MainModule Wrapper

Use a wrapper `nn.Module` that holds references to the live encoder/policy_head:

```python
class MainModule(nn.Module):
    """Holds references to the live encoder + policy_head modules."""
    def __init__(self, encoder: nn.Module, policy_head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.policy_head = policy_head
```

**Key insight:** This stores references, not clones. `main_module.encoder` IS `network.encoder`.

## Why Not Two Separate update() Calls?

If MLX increments Adam's internal step counter per `update()` call, doing:

```python
opt_main.update(network.encoder, encoder_grads)
opt_main.update(network.policy_head, policy_grads)  # Would "double-step" opt_main!
```

...would effectively double-step the optimizer each training step. The wrapper avoids this.

## Implementation

### 1. Create wrapper after network

```python
network = create_network(hidden=hidden, n_blocks=n_blocks)
main_module = MainModule(network.encoder, network.policy_head)

opt_main = optim.Adam(learning_rate=learning_rate)
opt_value = optim.Adam(learning_rate=learning_rate * value_lr_scale)
```

### 2. In train_step(), slice grads only (not params)

```python
loss_tuple, grads = nn.value_and_grad(network, loss_fn)(network)

# Slice GRADS only into module-shaped trees
main_grads = {
    "encoder": grads["encoder"],
    "policy_head": grads["policy_head"],
}
value_grads = grads["value_head"]

# Separate clipping
main_grads, _ = clip_grad_norm(main_grads, max_norm=1.0)
value_grads, _ = clip_grad_norm(value_grads, max_norm=0.5)

# Update REAL modules (guaranteed to mutate network)
opt_main.update(main_module, main_grads)
opt_value.update(network.value_head, value_grads)
```

## Comparison

| Old Approach | New Approach |
|--------------|--------------|
| Pass sliced dict to `update()` | Pass real `nn.Module` to `update()` |
| Hope MLX mutates original network | MLX guarantees mutation via module reference |
| Adam may track wrong param tree | Adam tracks exactly what it should |

## Verification

To verify updates land in the live model, use a checksum on first step:

```python
def first_array_leaf(tree):
    """Return first mx.array found in nested dict/list/tuple tree."""
    if tree is None:
        return None
    if isinstance(tree, mx.array):
        return tree
    if isinstance(tree, dict):
        for v in tree.values():
            out = first_array_leaf(v)
            if out is not None:
                return out
    if isinstance(tree, (list, tuple)):
        for v in tree:
            out = first_array_leaf(v)
            if out is not None:
                return out
    return None

# In training loop:
if step == 0 and iteration == 0:
    leaf = first_array_leaf(network.encoder.parameters())
    assert leaf is not None
    print(f"DEBUG encoder checksum: {float(mx.sum(leaf).item()):.6f}")
```

- Within a run: checksum should change after step 0 (proving weights updated)
- Across two runs with same seed: post-step-0 checksum should match (proving determinism)

## Files

- `scripts/GPU/alphazero/trainer.py`: Contains `MainModule` class and `train_step()` implementation
