# Targeted Value Calibration v14 — Gated Value-Adapter Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third training surface — `value_head + value_adapter` — an opt-in, policy-isolated, value-only 1×1-bottleneck adapter with a scalar gate (init 0), trained under the unchanged v12b guardrail-hinge objective, to test whether value-only capacity fixes gate A without the nonlocal B/C/D drift final-block training caused.

**Architecture:** A `ValueAdapter` module inserts `features_for_value = features + gate·adapter(features)` in the value path only (policy path untouched); it is constructed only when `--value-adapter` is set (byte-identical off) and is identity at init (gate=0). A new `--train-value-head-and-value-adapter` flag routes the value-side optimizer through a `ValueModule(value_head, value_adapter)` wrapper (one `opt_value.update`, no Adam double-step) while `opt_main` is skipped (encoder/policy/final-block frozen, exactly like v8). The v12b manifest/schedule/hinge/margin and projection code are unchanged; projection is rejected on the adapter surface (reserved for v14b).

**Tech Stack:** Python 3.14 / MLX, pytest.

**Spec:** `docs/superpowers/specs/2026-07-08-targeted-value-calibration-v14-value-adapter-design.md` (APPROVED — do not redesign).

## Global Constraints

- Python: always `.venv/bin/python`; tests `.venv/bin/python -m pytest <file> -v`; full-suite baseline on merged main after v13c + test-fix: **1372 passed**.
- NEVER `sys.modules.pop("mlx")` in tests.
- **Opt-in / byte-identical when off:** with `--value-adapter` absent, `self.value_adapter is None`; `_value_features` returns features unchanged; `network.parameters()` has no `value_adapter.*` key; the `train_step` value-side path and all save/load/telemetry are byte-identical to current `main`.
- **Identity at init:** adapter present + `gate=0.0` ⇒ `forward_padded` value byte-identical to the base (no-adapter) value on the same weights.
- **v14 surface** = `value_head.*` + `value_adapter.*` (incl. `value_adapter.gate`); `opt_main` skipped (like v8); encoder/policy/final-block frozen; pair with `--freeze-batchnorm-stats`.
- **Objective UNCHANGED:** v12b manifest/schedule, asymmetric guardrail hinge, `guardrail_margin` default 0.10, calibration weight. Gradient projection is OFF and is **rejected** on the v14 surface (`ValueError`) — projection over the adapter surface is v14b.
- **Gate:** key `value_adapter.gate`, stored `mx.zeros((1,))` (shape `(1,)`, not 0-d). Telemetry `value_adapter_gate` MUST appear in BOTH JSON sites (sidecar `build_post_opening_calibration_block` loss block in `calibration_pool.py` AND the flattened `_teacher_calib_scalars` mirror tuple in `trainer.py`). Run-level bool `train_value_head_and_value_adapter` in the state dict.
- **Mutual exclusion / dependency:** at most one of `--train-value-head-only` / `--train-value-head-and-final-block` / `--train-value-head-and-value-adapter`; the last requires `--value-adapter`.
- Do NOT change: `project_conflicting_gradient`, `_calibration_component_loss`, `alphazero_loss_batch`, the v12b manifest/builder, the guardrail margin, the v12b schedule, `verify_value_head_only_checkpoint.py`, `verify_value_head_and_final_block_checkpoint.py`, `MainModule`, `mcts.py`, `continuation_extraction.py`, `docs/post-game-analysis.md`.
- Worktree `feature/tvc-v14-value-adapter`; symlink `.venv`; FF-merge (no `--no-ff`, never force-push); trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; file-scoped `git add`; **locate code by content, not line numbers**. Fresh worktree lacks gitignored data → whole-repo suite there = 14 failed + 6 errors; judge tasks file-scoped; authoritative suite on merged main.

## File Structure

| File | Role |
|---|---|
| `scripts/GPU/alphazero/network.py` (modify) | `ValueAdapter` module; `AlphaZeroNetwork` opt-in (`value_adapter`, `value_adapter_bottleneck_width`); `_value_features` helper + `forward_padded`/`__call__` integration; `create_network` flags |
| `scripts/GPU/alphazero/trainer.py` (modify) | `ValueModule`; `train_step` v14 param + value-side routing + guards; `train()` params + `create_network` flags + `value_module` + both `train_step` calls + graft-load helper + telemetry + state flag |
| `scripts/GPU/alphazero/calibration_pool.py` (modify) | `value_adapter_gate` in `build_post_opening_calibration_block` loss block |
| `scripts/GPU/alphazero/train.py` (modify) | `--value-adapter` / `--value-adapter-bottleneck-width` / `--train-value-head-and-value-adapter` args + `parser.error` guards + plumb |
| `scripts/GPU/alphazero/verify_value_head_and_adapter_checkpoint.py` (create) | tensor-diff verifier, exit 0/1/2/3 |
| `tests/test_v14_value_adapter.py` (create; extended per task) | all behavioral tests |

**Task → work-item map:** T1 = network module + forward; T2 = trainer surface routing; T3 = graft-load; T4 = telemetry + CLI; T5 = verifier; T6 = full suite + merge (controller-run).

---

### Task 1: `ValueAdapter` module + network opt-in + forward integration (network.py)

**Files:**
- Modify: `scripts/GPU/alphazero/network.py`
- Test: `tests/test_v14_value_adapter.py` (create)

**Interfaces:**
- Produces: `ValueAdapter(channels, bottleneck_width=None)` with `.fc_down`/`.fc_up`/`.gate` (mx.array shape `(1,)`) and `__call__(features)->gate*fc_up(relu(fc_down(features)))`. `AlphaZeroNetwork(..., value_adapter: bool=False, value_adapter_bottleneck_width: Optional[int]=None)` with `self.value_adapter` (a `ValueAdapter` or `None`) and `_value_features(features)`. `create_network(..., value_adapter=False, value_adapter_bottleneck_width=None)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_v14_value_adapter.py`:

```python
"""v14: value-only feature-correction adapter (1x1 bottleneck + scalar gate,
init 0) in the value path only. Opt-in, byte-identical off, identity at init."""
import numpy as np
import pytest
import mlx.core as mx
from mlx.utils import tree_flatten

from scripts.GPU.alphazero.network import create_network, canonicalize_batch


def test_adapter_absent_by_default():
    net = create_network(hidden=64, n_blocks=2)
    assert net.value_adapter is None
    keys = {k for k, _ in tree_flatten(net.parameters())}
    assert not any(k.startswith("value_adapter") for k in keys)
    feats = mx.random.normal((1, 24, 24, 64))
    assert mx.array_equal(net._value_features(feats), feats).item()  # identity when absent


def test_gate_key_present_and_shape_and_default_width():
    net = create_network(hidden=64, n_blocks=2, value_adapter=True)
    keys = {k for k, _ in tree_flatten(net.parameters())}
    assert "value_adapter.gate" in keys                 # saves under value_adapter.*
    assert net.value_adapter.gate.shape == (1,)          # not 0-d (safetensors-safe)
    assert float(net.value_adapter.gate[0]) == 0.0       # init 0
    assert net.value_adapter.fc_down.weight.shape[0] == 64 // 4   # nn.Linear weight is (out,in)


def test_bottleneck_width_override():
    net = create_network(hidden=64, n_blocks=2, value_adapter=True,
                         value_adapter_bottleneck_width=8)
    assert net.value_adapter.fc_down.weight.shape[0] == 8


def test_zero_gate_value_features_identity():
    net = create_network(hidden=64, n_blocks=2, value_adapter=True)
    feats = mx.random.normal((2, 24, 24, 64))
    assert mx.array_equal(net._value_features(feats), feats).item()   # gate 0 -> identity


def test_nonzero_gate_changes_value_features():
    net = create_network(hidden=64, n_blocks=2, value_adapter=True)
    net.value_adapter.gate = mx.array([1.0])
    feats = mx.random.normal((2, 24, 24, 64))
    assert not mx.array_equal(net._value_features(feats), feats).item()


def _board_moves():
    board = mx.zeros((1, 24, 24, 30))
    rows = mx.zeros((1, 2), dtype=mx.int32)
    cols = mx.zeros((1, 2), dtype=mx.int32)
    mask = mx.ones((1, 2))
    return board, rows, cols, mask


def test_forward_padded_gate_zero_matches_raw_value_head():
    net = create_network(hidden=64, n_blocks=2, value_adapter=True)
    assert float(net.value_adapter.gate[0]) == 0.0
    board, rows, cols, mask = _board_moves()
    _, v_fwd, _ = net.forward_padded(board, rows, cols, mask, 24)
    cb, cr, cc, cm = canonicalize_batch(board, rows, cols, mask, 24)
    v_base = net.value_head(net.encoder(cb), 24)          # base path, no adapter
    assert mx.allclose(v_fwd, v_base).item()              # identity at init through forward_padded


def test_forward_padded_value_reflects_gate():
    net = create_network(hidden=64, n_blocks=2, value_adapter=True)
    board, rows, cols, mask = _board_moves()
    _, v0, _ = net.forward_padded(board, rows, cols, mask, 24)
    net.value_adapter.gate = mx.array([2.0])
    _, v1, _ = net.forward_padded(board, rows, cols, mask, 24)
    assert not mx.allclose(v0, v1).item()                 # forward_padded actually applies the adapter
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_v14_value_adapter.py -v`
Expected: FAIL — `TypeError: create_network() got an unexpected keyword argument 'value_adapter'` (and `AttributeError: ... 'value_adapter'`).

- [ ] **Step 3: Implement**

**(a)** In `network.py`, add the `ValueAdapter` class immediately **before** `class AlphaZeroNetwork(nn.Module):` (locate that class line):

```python
class ValueAdapter(nn.Module):
    """v14: small value-only feature-correction adapter (pointwise 1x1 bottleneck
    + folded scalar gate). Inserted between the shared encoder features and the
    value head so it corrects ONLY the value path (policy path untouched).

    __call__(features) -> gate * fc_up(relu(fc_down(features)))   over (B,H,W,C).
    The gate is init 0.0 (identity at init; ReZero-style bootstrap) and stored as
    shape (1,) so it saves/loads under the key "value_adapter.gate" (MLX
    safetensors are cleanest with >=1-d arrays).
    """

    def __init__(self, channels: int, bottleneck_width: Optional[int] = None):
        super().__init__()
        b = bottleneck_width if bottleneck_width else channels // 4
        self.fc_down = nn.Linear(channels, b)
        self.fc_up = nn.Linear(b, channels)
        self.gate = mx.zeros((1,))

    def __call__(self, features: mx.array) -> mx.array:
        h = nn.relu(self.fc_down(features))
        return self.gate * self.fc_up(h)
```

**(b)** Extend `AlphaZeroNetwork.__init__`. Locate:

```python
    def __init__(
        self,
        in_channels: int = NUM_CHANNELS,
        hidden: int = 128,
        n_blocks: int = 6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.encoder = BoardEncoder(in_channels, hidden, n_blocks)
        self.policy_head = PolicyHead(hidden)
        self.value_head = ValueHead(hidden)
```

Replace with:

```python
    def __init__(
        self,
        in_channels: int = NUM_CHANNELS,
        hidden: int = 128,
        n_blocks: int = 6,
        value_adapter: bool = False,
        value_adapter_bottleneck_width: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.encoder = BoardEncoder(in_channels, hidden, n_blocks)
        self.policy_head = PolicyHead(hidden)
        self.value_head = ValueHead(hidden)
        # v14: opt-in value-only adapter (None when off -> byte-identical, no
        # value_adapter.* params).
        self.value_adapter = (
            ValueAdapter(hidden, value_adapter_bottleneck_width)
            if value_adapter else None)

    def _value_features(self, features: mx.array) -> mx.array:
        """v14: value-only adapter correction (identity when the adapter is absent)."""
        if self.value_adapter is None:
            return features
        return features + self.value_adapter(features)
```

**(c)** In `AlphaZeroNetwork.forward_padded`, locate:

```python
        if return_value_pretanh:
            value, pretanh = self.value_head(features, active_size, return_pretanh=True)
            return policy_logits, value, pretanh

        value = self.value_head(features, active_size)  # (B,)
        return policy_logits, value, None
```

Replace with:

```python
        value_feats = self._value_features(features)  # v14: value-only adapter
        if return_value_pretanh:
            value, pretanh = self.value_head(value_feats, active_size, return_pretanh=True)
            return policy_logits, value, pretanh

        value = self.value_head(value_feats, active_size)  # (B,)
        return policy_logits, value, None
```

**(d)** In `AlphaZeroNetwork.__call__`, the empty-moves branch, locate:

```python
            features = self.encoder(board_canon)
            value = self.value_head(features, active_size)
            return mx.array([]), value
```

Replace with:

```python
            features = self.encoder(board_canon)
            value = self.value_head(self._value_features(features), active_size)  # v14
            return mx.array([]), value
```

**(e)** Extend `create_network`. Locate:

```python
def create_network(
    hidden: int = 128,
    n_blocks: int = 6,
    in_channels: Optional[int] = None,
) -> AlphaZeroNetwork:
```

Replace the signature with:

```python
def create_network(
    hidden: int = 128,
    n_blocks: int = 6,
    in_channels: Optional[int] = None,
    value_adapter: bool = False,
    value_adapter_bottleneck_width: Optional[int] = None,
) -> AlphaZeroNetwork:
```

And locate the return:

```python
    return AlphaZeroNetwork(
        in_channels=in_channels,
        hidden=hidden,
        n_blocks=n_blocks,
    )
```

Replace with:

```python
    return AlphaZeroNetwork(
        in_channels=in_channels,
        hidden=hidden,
        n_blocks=n_blocks,
        value_adapter=value_adapter,
        value_adapter_bottleneck_width=value_adapter_bottleneck_width,
    )
```

(`Optional` is already imported in network.py — it is used in the existing `create_network`/`forward_padded` signatures. `nn`, `mx` are imported.)

- [ ] **Step 4: Run the new tests**

Run: `.venv/bin/python -m pytest tests/test_v14_value_adapter.py -v`
Expected: ALL 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/network.py tests/test_v14_value_adapter.py
git commit -m "feat(network): v14 ValueAdapter module + opt-in value-only adapter surface

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: training surface routing — `ValueModule` + `train_step`/`train()` (trainer.py)

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py`
- Test: `tests/test_v14_value_adapter.py` (append)

**Interfaces:**
- Consumes: Task 1's `create_network(..., value_adapter=True)`, `network.value_adapter`, `network._value_features`.
- Produces: `ValueModule(value_head, value_adapter)`; `train_step(..., train_value_head_and_value_adapter=False, value_module=None)`; `train(..., value_adapter=False, value_adapter_bottleneck_width=None, train_value_head_and_value_adapter=False)`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_v14_value_adapter.py`)

```python
import mlx.optimizers as optim
from scripts.GPU.alphazero.trainer import (
    MainModule, ValueModule, train_step, freeze_batchnorm_running_stats)
from scripts.GPU.alphazero.self_play import PositionRecord


def _pos():
    return PositionRecord(board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
                          to_move="red", legal_moves=[(0, 0), (1, 1), (2, 2)],
                          visit_counts=[10, 5, 3], outcome=1.0, active_size=24,
                          ply=0, game_n_moves=10)


def _adapter_net():
    net = create_network(hidden=64, n_blocks=2, value_adapter=True)
    freeze_batchnorm_running_stats(net)
    return net


def _v14_kwargs(net):
    return dict(
        network=net, main_module=MainModule(net.encoder, net.policy_head),
        opt_main=optim.Adam(learning_rate=1e-3), opt_value=optim.Adam(learning_rate=1e-3),
        batch=[_pos() for _ in range(3)],
        train_value_head_and_value_adapter=True,
        value_module=ValueModule(net.value_head, net.value_adapter))


def test_v14_mutually_exclusive_with_v8():
    net = _adapter_net()
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_step(**{**_v14_kwargs(net), "train_value_head_only": True})


def test_v14_mutually_exclusive_with_v9():
    net = _adapter_net()
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_step(**{**_v14_kwargs(net), "train_value_head_and_final_block": True})


def test_projection_rejected_on_adapter_surface():
    net = _adapter_net()
    with pytest.raises(ValueError, match="requires --train-value-head-and-final-block"):
        train_step(**{**_v14_kwargs(net),
                      "post_opening_calibration_gradient_projection": True})


def test_v14_surface_isolation():
    net = _adapter_net()
    before = {k: np.array(v) for k, v in tree_flatten(net.parameters())}
    train_step(**_v14_kwargs(net))
    after = {k: np.array(v) for k, v in tree_flatten(net.parameters())}
    for k in after:
        changed = not np.array_equal(before[k], after[k])
        if k.startswith("value_head.") or k.startswith("value_adapter."):
            continue                                   # allowed to change
        assert not changed, f"frozen tensor changed under v14: {k}"
    # the gate received gradient and moved off 0 (v14 actually engaged)
    assert not np.array_equal(before["value_adapter.gate"], after["value_adapter.gate"])
    # value head trained too
    assert any(not np.array_equal(before[k], after[k])
               for k in after if k.startswith("value_head."))


def test_guardrail_hinge_sees_adapter():
    # the v12b guardrail hinge (via model.forward_padded) reads the
    # adapter-corrected value — changing the gate changes the hinge. This calls
    # _calibration_component_loss (do-not-change) only to observe it.
    from scripts.GPU.alphazero.trainer import _calibration_component_loss
    from scripts.GPU.alphazero.calibration_pool import target_in_to_move
    net = _adapter_net()
    row = PositionRecord(board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
                         to_move="black", legal_moves=[(0, 0), (1, 1)],
                         visit_counts=[0, 0], outcome=target_in_to_move("black", -0.9),
                         active_size=24, ply=20, game_n_moves=None)
    sign = np.array([1.0], dtype=np.float32)
    h0 = float(_calibration_component_loss(net, [row], None, sign, 0.10, "guardrail_hinge").item())
    net.value_adapter.gate = mx.array([3.0])
    h1 = float(_calibration_component_loss(net, [row], None, sign, 0.10, "guardrail_hinge").item())
    assert h0 != h1
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_v14_value_adapter.py -k "v14 or projection_rejected or hinge_sees" -v`
Expected: FAIL — `ImportError: cannot import name 'ValueModule'` (then, once that exists, `TypeError: train_step() got an unexpected keyword argument 'train_value_head_and_value_adapter'`).

- [ ] **Step 3: Implement**

**(a)** Add `ValueModule` immediately after `class MainModule(nn.Module):` ends (locate the line `        self.policy_head = policy_head` that closes `MainModule.__init__`, and insert after it, before the `# Per-size max moves table` comment):

```python
class ValueModule(nn.Module):
    """Holds references to the live value_head + value_adapter (v14).

    Used so opt_value updates value_head AND the value adapter in ONE update()
    call (no Adam double-step), mirroring MainModule for the value side.
    """
    def __init__(self, value_head: nn.Module, value_adapter: nn.Module):
        super().__init__()
        self.value_head = value_head
        self.value_adapter = value_adapter
```

**(b)** `train_step` signature. Locate `    post_opening_calibration_projection_strength: float = 1.0,   # v13c` (the last param before `) -> tuple:`) and add below it:

```python
    train_value_head_and_value_adapter: bool = False,   # v14
    value_module=None,                                   # v14: ValueModule(value_head, value_adapter)
```

**(c)** Mutual-exclusion guard. Locate:

```python
    if train_value_head_only and train_value_head_and_final_block:
        raise ValueError(
            "train_value_head_only and train_value_head_and_final_block are "
            "mutually exclusive")
```

Add immediately below it:

```python
    if train_value_head_and_value_adapter and (
            train_value_head_only or train_value_head_and_final_block):
        raise ValueError(
            "train_value_head_and_value_adapter is mutually exclusive with "
            "train_value_head_only and train_value_head_and_final_block")
```

**(d)** Projection guard. Locate:

```python
        if train_value_head_only:
            raise ValueError(
                "post_opening_calibration_gradient_projection requires "
                "--train-value-head-and-final-block (the value-head-only surface "
                "does not define the A-vs-guardrail final-block conflict)")
```

Replace with:

```python
        if train_value_head_only or train_value_head_and_value_adapter:
            raise ValueError(
                "post_opening_calibration_gradient_projection requires "
                "--train-value-head-and-final-block (the value-head-only and "
                "value-adapter surfaces do not define the A-vs-guardrail "
                "final-block conflict; adapter-surface projection is v14b)")
```

**(e)** Value-side grad extraction. Locate:

```python
    value_grads = grads["value_head"]
```

Replace with:

```python
    if train_value_head_and_value_adapter:
        value_grads = {"value_head": grads["value_head"],
                       "value_adapter": grads["value_adapter"]}
    else:
        value_grads = grads["value_head"]
```

**(f)** Update region. Locate:

```python
    # Update REAL modules (guaranteed to mutate network)
    if train_value_head_only:
        # v8: encoder+policy grads are computed and clipped (telemetry
        # unchanged) but never applied — only the value head trains.
        pass
```

Replace with:

```python
    # Update REAL modules (guaranteed to mutate network)
    if train_value_head_only or train_value_head_and_value_adapter:
        # v8 / v14: encoder+policy grads are computed and clipped (telemetry
        # unchanged) but never applied — only the value side (head, and for v14
        # the adapter) trains.
        pass
```

Then locate:

```python
    opt_value.update(network.value_head, value_grads)
```

Replace with:

```python
    if train_value_head_and_value_adapter:
        opt_value.update(value_module, value_grads)   # v14: value_head + value_adapter, one update
    else:
        opt_value.update(network.value_head, value_grads)
```

**(g)** `train()` signature. Locate the `def train(` param `    post_opening_calibration_projection_strength: float = 1.0,   # v13c` (immediately above `) -> AlphaZeroNetwork:`) and add below it:

```python
    value_adapter: bool = False,                              # v14
    value_adapter_bottleneck_width: Optional[int] = None,     # v14
    train_value_head_and_value_adapter: bool = False,         # v14
```

**(h)** `create_network` call in `train()`. Locate:

```python
    # Create network
    network = create_network(hidden=hidden, n_blocks=n_blocks)
```

Replace with:

```python
    # Create network
    network = create_network(hidden=hidden, n_blocks=n_blocks,
                             value_adapter=value_adapter,
                             value_adapter_bottleneck_width=value_adapter_bottleneck_width)
```

**(i)** `value_module` construction. Locate:

```python
    # Create wrapper module that references encoder + policy_head
    # This ensures opt_main.update() mutates the live network params
    main_module = MainModule(network.encoder, network.policy_head)
```

Add immediately below it:

```python
    # v14: value-side wrapper so opt_value updates value_head + value_adapter in
    # one call (None off).
    value_module = (ValueModule(network.value_head, network.value_adapter)
                    if train_value_head_and_value_adapter else None)
```

**(j)** Forward the two new args in **both** `train_step(...)` calls in `train()`.

In the CALIBRATION-branch call, locate:

```python
                                post_opening_calibration_projection_strength=post_opening_calibration_projection_strength,
                            )
```

Replace with:

```python
                                post_opening_calibration_projection_strength=post_opening_calibration_projection_strength,
                                train_value_head_and_value_adapter=train_value_head_and_value_adapter,
                                value_module=value_module,
                            )
```

In the non-calibration branch call, locate:

```python
                                train_value_head_only=train_value_head_only,
                                train_value_head_and_final_block=train_value_head_and_final_block,
                            )
```

Replace with:

```python
                                train_value_head_only=train_value_head_only,
                                train_value_head_and_final_block=train_value_head_and_final_block,
                                train_value_head_and_value_adapter=train_value_head_and_value_adapter,
                                value_module=value_module,
                            )
```

(`Optional` is already imported in trainer.py.)

- [ ] **Step 4: Run the new tests + regression**

Run: `.venv/bin/python -m pytest tests/test_v14_value_adapter.py tests/test_training.py -v`
Expected: ALL PASS (v14 mutual-exclusion/projection/surface-isolation green; existing training tests unchanged — byte-identical off).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py tests/test_v14_value_adapter.py
git commit -m "feat(training): v14 value-adapter training surface (ValueModule routing + guards)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: graft-load (strict-except-adapter) in `train()` load path (trainer.py)

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py`
- Test: `tests/test_v14_value_adapter.py` (append)

**Interfaces:**
- Produces: module-level `_load_base_weights_grafting_adapter(network, path)`; the `train()` load path uses it when `value_adapter` is set.

- [ ] **Step 1: Write the failing tests** (append)

```python
from scripts.GPU.alphazero.trainer import _load_base_weights_grafting_adapter


def test_graft_load_succeeds_and_keeps_gate_zero(tmp_path):
    base = create_network(hidden=64, n_blocks=2)             # no adapter
    p = str(tmp_path / "base.safetensors")
    base.save_weights(p)
    adapter_net = create_network(hidden=64, n_blocks=2, value_adapter=True)
    _load_base_weights_grafting_adapter(adapter_net, p)
    assert float(adapter_net.value_adapter.gate[0]) == 0.0   # adapter untouched (fresh)
    assert mx.array_equal(adapter_net.value_head.fc1.weight,
                          base.value_head.fc1.weight).item()  # shared weights grafted


def test_graft_load_fails_loud_on_unexpected_missing(tmp_path):
    base = create_network(hidden=64, n_blocks=2)
    d = dict(tree_flatten(base.parameters()))
    d.pop(next(k for k in d if k.startswith("value_head.")))  # drop a non-adapter key
    p = str(tmp_path / "broken.safetensors")
    mx.save_safetensors(p, d)
    adapter_net = create_network(hidden=64, n_blocks=2, value_adapter=True)
    with pytest.raises(ValueError, match="graft-load mismatch"):
        _load_base_weights_grafting_adapter(adapter_net, p)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_v14_value_adapter.py -k graft -v`
Expected: FAIL — `ImportError: cannot import name '_load_base_weights_grafting_adapter'`.

- [ ] **Step 3: Implement**

**(a)** Add the helper. Locate `def freeze_batchnorm_running_stats(network) -> int:` near the top of `trainer.py` and add the following function immediately **above** it (both are module-level helpers):

```python
def _load_base_weights_grafting_adapter(network, path: str) -> None:
    """v14: load a base checkpoint (which has NO value_adapter.* keys) into an
    adapter-augmented network with strict=False, asserting the ONLY keys missing
    from the file are exactly the value_adapter.* set — so a real load bug cannot
    hide behind strict=False. The gate stays at its 0.0 init after the graft."""
    file_keys = set(mx.load(str(path)).keys())
    net_keys = {k for k, _ in tree_flatten(network.parameters())}
    missing = net_keys - file_keys                       # in net, absent from file
    extra = file_keys - net_keys                          # in file, not in net
    expected_missing = {k for k in net_keys if k.startswith("value_adapter.")}
    if missing != expected_missing or extra:
        raise ValueError(
            "graft-load mismatch (adapter present): "
            f"missing-from-file={sorted(missing)}, "
            f"expected-missing={sorted(expected_missing)}, "
            f"extra-in-file={sorted(extra)}")
    network.load_weights(str(path), strict=False)
```

Ensure `tree_flatten` is importable in trainer.py: if the file does not already `from mlx.utils import tree_flatten`, add that import next to the other `mlx` imports at the top of the file. (Confirm by searching for `tree_flatten` before adding.)

**(b)** Use it in the load path. Locate:

```python
    # Load weights-only if specified (no state restore)
    if load_weights_from:
        network.load_weights(load_weights_from)
        print(f"Loaded weights-only from {load_weights_from} (no state restored)")
```

Replace with:

```python
    # Load weights-only if specified (no state restore)
    if load_weights_from:
        if value_adapter:
            _load_base_weights_grafting_adapter(network, load_weights_from)  # v14 strict-except-adapter
        else:
            network.load_weights(load_weights_from)
        print(f"Loaded weights-only from {load_weights_from} (no state restored)")
```

- [ ] **Step 4: Run the new tests**

Run: `.venv/bin/python -m pytest tests/test_v14_value_adapter.py -k graft -v`
Expected: BOTH PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py tests/test_v14_value_adapter.py
git commit -m "feat(training): v14 graft-load (strict-except-adapter, fail-loud on other key mismatch)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: telemetry (both sites) + CLI (trainer.py + calibration_pool.py + train.py)

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py`, `scripts/GPU/alphazero/calibration_pool.py`, `scripts/GPU/alphazero/train.py`
- Test: `tests/test_v14_value_adapter.py` (append)

**Interfaces:**
- Consumes: Task 2's `train()` params + `network.value_adapter.gate`.
- Produces: `value_adapter_gate` in `build_post_opening_calibration_block` loss dict + the `_teacher_calib_scalars` mirror; `train_value_head_and_value_adapter` in the state dict; `--value-adapter`, `--value-adapter-bottleneck-width`, `--train-value-head-and-value-adapter` CLI args + plumb + `parser.error` guards.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_build_block_emits_value_adapter_gate():
    from scripts.GPU.alphazero.calibration_pool import build_post_opening_calibration_block
    block = build_post_opening_calibration_block(
        config={}, enabled=True,
        loss_accumulator={"steps_done": 1, "value_adapter_gate": 0.37})
    assert block["loss"]["value_adapter_gate"] == pytest.approx(0.37)


def test_cli_and_telemetry_wiring():
    from scripts.GPU.alphazero import train as train_mod
    from scripts.GPU.alphazero import trainer as trainer_mod
    from scripts.GPU.alphazero import calibration_pool as cp_mod
    tsrc = open(train_mod.__file__).read()
    assert '"--value-adapter"' in tsrc
    assert '"--value-adapter-bottleneck-width"' in tsrc
    assert '"--train-value-head-and-value-adapter"' in tsrc
    assert "value_adapter=args.value_adapter," in tsrc
    assert ("train_value_head_and_value_adapter="
            "args.train_value_head_and_value_adapter,") in tsrc
    assert "requires --value-adapter" in tsrc                       # dependency guard
    rsrc = open(trainer_mod.__file__).read()
    assert '"value_adapter_gate"' in rsrc                            # loss_accumulator + mirror tuple
    assert '"train_value_head_and_value_adapter": train_value_head_and_value_adapter,' in rsrc
    csrc = open(cp_mod.__file__).read()
    assert '"value_adapter_gate"' in csrc                            # sidecar loss block
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_v14_value_adapter.py -k "telemetry or build_block" -v`
Expected: FAIL (missing keys / source strings).

- [ ] **Step 3: Implement**

**(a)** `trainer.py` — `loss_accumulator` dict. Locate:

```python
                        "guardrail_margin": post_opening_guardrail_margin,
```

Add immediately below it:

```python
                        "value_adapter_gate": (
                            float(network.value_adapter.gate[0])
                            if train_value_head_and_value_adapter else 0.0),
```

**(b)** `trainer.py` — flattening mirror tuple. Locate:

```python
                    "calib_projection_no_conflict_steps", "calib_projection_strength")
                if k in _poc_loss}
```

Replace with:

```python
                    "calib_projection_no_conflict_steps", "calib_projection_strength",
                    "value_adapter_gate")
                if k in _poc_loss}
```

**(c)** `trainer.py` — state dict run flag. Locate:

```python
            "train_value_head_and_final_block": train_value_head_and_final_block,
            "unfrozen_block_index": (
                len(network.encoder.blocks) - 1
                if train_value_head_and_final_block else None),
```

Add immediately below the `"unfrozen_block_index": (...)` entry:

```python
            # v14: whether only value_head + value_adapter trained.
            "train_value_head_and_value_adapter": train_value_head_and_value_adapter,
```

**(d)** `calibration_pool.py` — `build_post_opening_calibration_block` loss dict. Locate:

```python
            "guardrail_margin":
                float(loss_accumulator.get("guardrail_margin", 0.0)),
```

Add immediately below it:

```python
            "value_adapter_gate":
                float(loss_accumulator.get("value_adapter_gate", 0.0)),
```

**(e)** `train.py` — CLI args. Locate the `--post-opening-calibration-projection-strength` argument definition and add these three sibling args immediately after its full `parser.add_argument(...)` call:

```python
    parser.add_argument("--value-adapter", action="store_true",
        help="v14: build a value-only feature-correction adapter (1x1 bottleneck "
             "+ scalar gate init 0) between the encoder and the value head. Off "
             "by default (byte-identical). Required by "
             "--train-value-head-and-value-adapter.")
    parser.add_argument("--value-adapter-bottleneck-width", type=int, default=None,
        help="v14: adapter bottleneck width. Default (None) = hidden // 4.")
    parser.add_argument("--train-value-head-and-value-adapter", action="store_true",
        help="v14: train only value_head.* + value_adapter.* (skip the whole-trunk "
             "opt_main update; encoder/policy/final-block frozen). Mutually "
             "exclusive with --train-value-head-only / "
             "--train-value-head-and-final-block. Requires --value-adapter. Pair "
             "with --freeze-batchnorm-stats.")
```

**(f)** `train.py` — `parser.error` guards. Locate:

```python
    if args.train_value_head_only and args.train_value_head_and_final_block:
        parser.error("--train-value-head-only and "
                     "--train-value-head-and-final-block are mutually exclusive")
```

Replace with:

```python
    if sum([args.train_value_head_only,
            args.train_value_head_and_final_block,
            args.train_value_head_and_value_adapter]) > 1:
        parser.error("--train-value-head-only, --train-value-head-and-final-block, "
                     "and --train-value-head-and-value-adapter are mutually exclusive")
    if args.train_value_head_and_value_adapter and not args.value_adapter:
        parser.error("--train-value-head-and-value-adapter requires --value-adapter")
```

**(g)** `train.py` — plumb into `train(...)`. Locate:

```python
        post_opening_calibration_projection_strength=args.post_opening_calibration_projection_strength,
    ))
```

Replace with:

```python
        post_opening_calibration_projection_strength=args.post_opening_calibration_projection_strength,
        value_adapter=args.value_adapter,
        value_adapter_bottleneck_width=args.value_adapter_bottleneck_width,
        train_value_head_and_value_adapter=args.train_value_head_and_value_adapter,
    ))
```

- [ ] **Step 4: Run the new tests + regression**

Run: `.venv/bin/python -m pytest tests/test_v14_value_adapter.py tests/test_calibration_pool.py tests/test_calibration_cli_flags.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py scripts/GPU/alphazero/calibration_pool.py scripts/GPU/alphazero/train.py tests/test_v14_value_adapter.py
git commit -m "feat(training): v14 value_adapter_gate telemetry (both sites) + CLI flags

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: verifier `verify_value_head_and_adapter_checkpoint.py`

**Files:**
- Create: `scripts/GPU/alphazero/verify_value_head_and_adapter_checkpoint.py`
- Test: `tests/test_v14_verify_adapter_checkpoint.py` (create)

**Interfaces:**
- Produces: `compare_value_head_and_adapter(base_path, candidate_path) -> dict`; `main(argv) -> int` (exit 0 pass / 1 leak / 2 value-path no-op / 3 unexpected new-key set).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_v14_verify_adapter_checkpoint.py`:

```python
"""v14 verifier: a --train-value-head-and-value-adapter checkpoint changed ONLY
value_head.* + value_adapter.* vs its base; everything else byte-identical."""
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.verify_value_head_and_adapter_checkpoint import main


def _save(net, path):
    net.save_weights(str(path))


def _mutate(d, key, add):
    d = dict(d)
    d[key] = d[key] + add
    return d


def test_exit0_legal_value_head_and_adapter_delta(tmp_path):
    base = create_network(hidden=64, n_blocks=2)
    bp = tmp_path / "base.safetensors"
    _save(base, bp)
    # candidate = adapter net loaded from base, with value_head + gate moved
    cand = create_network(hidden=64, n_blocks=2, value_adapter=True)
    cand.load_weights(str(bp), strict=False)
    cand.value_adapter.gate = mx.array([0.5])
    cand.value_head.fc1.weight = cand.value_head.fc1.weight + 0.01
    cp = tmp_path / "cand.safetensors"
    _save(cand, cp)
    assert main(["--base", str(bp), "--candidate", str(cp)]) == 0


def test_exit1_frozen_leak(tmp_path):
    base = create_network(hidden=64, n_blocks=2)
    bp = tmp_path / "base.safetensors"
    _save(base, bp)
    cand = create_network(hidden=64, n_blocks=2, value_adapter=True)
    cand.load_weights(str(bp), strict=False)
    cand.value_adapter.gate = mx.array([0.5])
    cand.value_head.fc1.weight = cand.value_head.fc1.weight + 0.01
    # LEAK: also change a policy tensor via the flattened dict
    d = dict(tree_flatten(cand.parameters()))
    pk = next(k for k in d if k.startswith("policy_head."))
    d = _mutate(d, pk, 0.02)
    cp = tmp_path / "cand.safetensors"
    mx.save_safetensors(str(cp), d)
    assert main(["--base", str(bp), "--candidate", str(cp)]) == 1


def test_exit2_gate_never_moved(tmp_path):
    base = create_network(hidden=64, n_blocks=2)
    bp = tmp_path / "base.safetensors"
    _save(base, bp)
    cand = create_network(hidden=64, n_blocks=2, value_adapter=True)
    cand.load_weights(str(bp), strict=False)     # gate stays 0, value_head unchanged
    cp = tmp_path / "cand.safetensors"
    _save(cand, cp)
    assert main(["--base", str(bp), "--candidate", str(cp)]) == 2


def test_exit3_no_adapter_keys(tmp_path):
    base = create_network(hidden=64, n_blocks=2)
    bp = tmp_path / "base.safetensors"
    _save(base, bp)
    cand = create_network(hidden=64, n_blocks=2)     # NO adapter -> no new keys
    d = dict(tree_flatten(cand.parameters()))
    vk = next(k for k in d if k.startswith("value_head."))
    d = _mutate(d, vk, 0.01)
    cp = tmp_path / "cand.safetensors"
    mx.save_safetensors(str(cp), d)
    assert main(["--base", str(bp), "--candidate", str(cp)]) == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_v14_verify_adapter_checkpoint.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.GPU.alphazero.verify_value_head_and_adapter_checkpoint'`.

- [ ] **Step 3: Implement**

Create `scripts/GPU/alphazero/verify_value_head_and_adapter_checkpoint.py`:

```python
"""v14 acceptance check: prove a --train-value-head-and-value-adapter run touched
ONLY value_head.* and the (new) value_adapter.* tensors.

Compares two safetensors checkpoints. The base has NO value_adapter.* keys; the
candidate adds exactly the value_adapter.* set. Allowed to change: value_head.*
and value_adapter.*. Everything else that the two checkpoints share — the stem,
ALL residual blocks (including the final one), the policy head, and ALL
BatchNorm running stats anywhere — must be byte-identical:
  exit 0  PASS: shared frozen set byte-identical; value head AND the gate moved
  exit 1  FAIL: a shared frozen tensor changed (a running-stat leak means
          --freeze-batchnorm-stats was missing/ineffective — run is invalid)
  exit 2  FAIL: no value_head tensor changed, or the gate never left 0.0 — the
          value-path correction never engaged (no-op)
  exit 3  FAIL: candidate-only keys are not exactly value_adapter.*, or the base
          has keys the candidate lacks (wrong architecture / flag mis-plumbed)
"""
from __future__ import annotations

import argparse
import sys

import mlx.core as mx

GATE_KEY = "value_adapter.gate"


def _is_running_stat(key: str) -> bool:
    return key.endswith(".running_mean") or key.endswith(".running_var")


def compare_value_head_and_adapter(base_path: str, candidate_path: str) -> dict:
    base = mx.load(str(base_path))
    cand = mx.load(str(candidate_path))
    base_keys, cand_keys = set(base), set(cand)
    new_keys = cand_keys - base_keys              # expected: exactly value_adapter.*
    missing = base_keys - cand_keys               # expected: empty
    unexpected_new = {k for k in new_keys if not k.startswith("value_adapter.")}
    adapter_keys = {k for k in new_keys if k.startswith("value_adapter.")}
    frozen_diffs, value_head_deltas, adapter_deltas = [], {}, {}
    for k in sorted(base_keys & cand_keys):
        if k.startswith("value_head."):
            delta = mx.abs(cand[k].astype(mx.float32) - base[k].astype(mx.float32))
            value_head_deltas[k] = float(delta.max().item()) if delta.size else 0.0
        elif not bool(mx.array_equal(base[k], cand[k]).item()):
            frozen_diffs.append(k)
    for k in sorted(adapter_keys):
        adapter_deltas[k] = float(mx.abs(cand[k].astype(mx.float32)).max().item())
    gate_abs = (float(mx.abs(cand[GATE_KEY]).max().item())
                if GATE_KEY in cand else 0.0)
    return {"frozen_diffs": frozen_diffs, "value_head_deltas": value_head_deltas,
            "adapter_deltas": adapter_deltas, "missing": sorted(missing),
            "unexpected_new": sorted(unexpected_new),
            "adapter_keys": sorted(adapter_keys), "gate_abs": gate_abs,
            "n_tensors": len(base_keys)}


def _changed(deltas: dict) -> bool:
    return bool(deltas) and max(deltas.values()) > 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Verify a --train-value-head-and-value-adapter checkpoint "
                    "changed ONLY value_head.* and value_adapter.* vs its base.")
    ap.add_argument("--base", required=True)
    ap.add_argument("--candidate", required=True)
    args = ap.parse_args(argv)
    r = compare_value_head_and_adapter(args.base, args.candidate)
    for k, d in sorted(r["value_head_deltas"].items()):
        print(f"{k}: max|delta| = {d:.3e}")
    for k, d in sorted(r["adapter_deltas"].items()):
        print(f"{k}: max|abs| = {d:.3e}")
    if r["missing"] or r["unexpected_new"] or not r["adapter_keys"]:
        print(f"FAIL: unexpected key sets — missing-in-candidate={r['missing']}, "
              f"non-adapter-new={r['unexpected_new']}, "
              f"adapter-keys-present={bool(r['adapter_keys'])}")
        return 3
    if r["frozen_diffs"]:
        print(f"FAIL: {len(r['frozen_diffs'])} shared frozen tensor(s) changed "
              f"(allowed: value_head.* + value_adapter.*):")
        for k in r["frozen_diffs"]:
            print(f"  LEAK: {k}")
        return 1
    if not _changed(r["value_head_deltas"]) or r["gate_abs"] == 0.0:
        print(f"FAIL: value-path no-op (value_head changed={_changed(r['value_head_deltas'])}, "
              f"gate |abs|={r['gate_abs']:.3e}) — the adapter correction never engaged")
        return 2
    print(f"PASS: {r['n_tensors']} base tensors; shared frozen set byte-identical; "
          f"value head + value_adapter (gate |abs|={r['gate_abs']:.3e}) trained")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the new tests**

Run: `.venv/bin/python -m pytest tests/test_v14_verify_adapter_checkpoint.py -v`
Expected: ALL 4 PASS (exit 0/1/2/3). (The test file has no `cand.policy_head.update(...)` line — the exit-1 leak is injected by mutating a `policy_head.*` key in the flattened dict before `mx.save_safetensors`.)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/verify_value_head_and_adapter_checkpoint.py tests/test_v14_verify_adapter_checkpoint.py
git commit -m "feat(training): v14 verify_value_head_and_adapter_checkpoint (exit 0/1/2/3)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: full-suite verification + merge handoff (controller-run)

- [ ] **Step 1: Full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected in the worktree: baseline-passed + the new v14 tests, with EXACTLY the known 14 failed + 6 errors (missing gitignored data). Authoritative check (1372 + new v14 tests, 0 failures) on merged main before push.

- [ ] **Step 2: Hand off to merge**

FF-merge to main, authoritative suite on merged main, push (superpowers:finishing-a-development-branch). STOP after push — the operator run is the USER's (see below).

---

## Operator run (USER's, after merge) — from the spec §11

The canonical v12b command **minus** `--train-value-head-and-final-block` and (if present) `--post-opening-calibration-gradient-projection`, **plus** `--value-adapter` (+ optional `--value-adapter-bottleneck-width 32`) and `--train-value-head-and-value-adapter`, with `--guardrail-margin 0.10 --freeze-batchnorm-stats`, new checkpoint dir `checkpoints/alphazero-v14-value-adapter-from-calib020-0001`. Confirm `train_value_head_and_value_adapter=true` + `value_adapter_gate` telemetry present (and moving off 0) + `verify_value_head_and_adapter_checkpoint` exit 0; then gates A/B/C/D vs `calib020_0001` (no promotion unless all four pass). Interpretation: A moves + B/C/D hold → adapter capacity was the missing piece (promotion match); A moves + B/C/D drift → v14b adds projection over the adapter surface; A does not move → wider bottleneck arg-only (`--value-adapter-bottleneck-width 64`), then per-channel gate as a later written design.
```
