# Targeted Value Calibration v14 — Gated Value-Adapter Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third training surface — `value_head + value_adapter` — an opt-in, policy-isolated, value-only 1×1-bottleneck adapter with a scalar gate (init 0), trained under the unchanged v12b guardrail-hinge objective, to test whether value-only capacity fixes gate A without the nonlocal B/C/D drift final-block training caused.

**Architecture:** A `ValueAdapter` module inserts `features_for_value = features + value_adapter(features)` (adapter = `gate·fc_up(relu(fc_down(features)))`) in the value path only (policy untouched); it is constructed only when `--value-adapter` is set (byte-identical off) and is identity at init (gate=0). A new `--train-value-head-and-value-adapter` flag routes the value-side optimizer through a `ValueModule(value_head, value_adapter)` wrapper (one `opt_value.update`, no Adam double-step) while `opt_main` is skipped (encoder/policy/final-block frozen, like v8). The v12b manifest/schedule/hinge/margin are unchanged; projection is rejected on the adapter surface (v14b).

**Tech Stack:** Python 3.14 / MLX, pytest.

**Spec:** `docs/superpowers/specs/2026-07-08-targeted-value-calibration-v14-value-adapter-design.md` (APPROVED — do not redesign).

## Global Constraints

- Python: always `.venv/bin/python`; tests `.venv/bin/python -m pytest <file> -v`; full-suite baseline on merged main = **1372 passed**.
- NEVER `sys.modules.pop("mlx")` in tests.
- **INVARIANT A — `--value-adapter` absent ⇒ no change:** `self.value_adapter is None`; `_value_features` returns features unchanged; `network.parameters()` has no `value_adapter.*` key; the load path, the `train_step` value-side/update path, and all telemetry are byte-identical to current `main`; existing v8–v13 behavior unchanged. (Every new code branch is gated on a flag defaulting to `False`/`None`.)
- **INVARIANT B — `--value-adapter` present + gate=0:** the value output equals the base value at init (identity); the **policy** output is independent of the adapter/gate always; after training, ONLY `value_head.*` + `value_adapter.*` may change (encoder incl. final block, policy_head, and all BN running stats byte-identical).
- **v14 surface** = `value_head.*` + `value_adapter.*` (incl. `value_adapter.gate`); `opt_main` skipped (like v8); pair with `--freeze-batchnorm-stats`.
- **Objective UNCHANGED:** v12b manifest/schedule, asymmetric guardrail hinge, `guardrail_margin` default 0.10, calibration weight. Gradient projection is OFF and **rejected** on the v14 surface (`ValueError`).
- **Gate:** key `value_adapter.gate`, stored `mx.zeros((1,))` (shape `(1,)`). Telemetry `value_adapter_gate` AND `value_adapter_grad_norm` MUST appear in BOTH JSON sites (sidecar `build_post_opening_calibration_block` loss block + the flattened `_teacher_calib_scalars` mirror tuple). Run-level bool `train_value_head_and_value_adapter` in the state dict.
- **Mutual exclusion / dependency:** at most one of `--train-value-head-only` / `--train-value-head-and-final-block` / `--train-value-head-and-value-adapter`; the last requires `--value-adapter`.
- Do NOT change: `project_conflicting_gradient`, `_calibration_component_loss`, `alphazero_loss_batch`, the v12b manifest/builder, the guardrail margin, the v12b schedule, `verify_value_head_only_checkpoint.py`, `verify_value_head_and_final_block_checkpoint.py`, `MainModule`, `mcts.py`, `continuation_extraction.py`, `docs/post-game-analysis.md`.
- Worktree `feature/tvc-v14-value-adapter`; symlink `.venv`; FF-merge (no `--no-ff`, never force-push); trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; file-scoped `git add`; **locate code by content, not line numbers**. Fresh worktree lacks gitignored data → whole-repo suite there = 14 failed + 6 errors; judge tasks file-scoped; authoritative suite on merged main.

## File Structure

| File | Role |
|---|---|
| `scripts/GPU/alphazero/network.py` (modify) | `ValueAdapter`; `AlphaZeroNetwork` opt-in + `_value_features` + `forward_padded`/`__call__` integration; `create_network` flags |
| `scripts/GPU/alphazero/trainer.py` (modify) | `_load_base_weights_grafting_adapter` helper (T1); `ValueModule` + `train_step` routing/guards/grad-norm + `train()` params/wiring/graft-wiring + telemetry (T2) |
| `scripts/GPU/alphazero/calibration_pool.py` (modify) | `value_adapter_gate` + `value_adapter_grad_norm` in `build_post_opening_calibration_block` (T2) |
| `scripts/GPU/alphazero/train.py` (modify) | v14 CLI args + `parser.error` guards + plumb (T2) |
| `scripts/GPU/alphazero/verify_value_head_and_adapter_checkpoint.py` (create) | verifier, exit 0/1/2/3 (T3) |
| `scripts/GPU/alphazero/smoke_v14_value_adapter.py` (create) | tiny end-to-end v14 smoke (T3) |
| `tests/test_v14_value_adapter.py` (create; T1 + T2) / `tests/test_v14_verify_adapter_checkpoint.py` (create; T3) | behavioral tests |

**Task → work-item map:** T1 = network module + forward integration + graft-load helper. T2 = training surface (CLI + ValueModule + routing + mutual-exclusion + telemetry both sites) + train() wiring. T3 = verifier + smoke + full suite + merge + operator run.

---

### Task 1: Network + checkpoint graft (network.py + trainer.py graft helper)

**Files:**
- Modify: `scripts/GPU/alphazero/network.py`, `scripts/GPU/alphazero/trainer.py`
- Test: `tests/test_v14_value_adapter.py` (create)

**Interfaces:**
- Produces: `ValueAdapter(channels, bottleneck_width=None)` with `.fc_down`/`.fc_up`/`.gate` (mx.array shape `(1,)`) and `__call__(features)->gate*fc_up(relu(fc_down(features)))`. `AlphaZeroNetwork(..., value_adapter=False, value_adapter_bottleneck_width=None)` with `self.value_adapter` (`ValueAdapter` or `None`) + `_value_features(features)`. `create_network(..., value_adapter=False, value_adapter_bottleneck_width=None)`. `_load_base_weights_grafting_adapter(network, path)` (trainer.py, module-level).

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
from scripts.GPU.alphazero.trainer import _load_base_weights_grafting_adapter


def test_adapter_absent_by_default():
    net = create_network(hidden=64, n_blocks=2)
    assert net.value_adapter is None
    keys = {k for k, _ in tree_flatten(net.parameters())}
    assert not any(k.startswith("value_adapter") for k in keys)
    feats = mx.random.normal((1, 24, 24, 64))
    assert mx.array_equal(net._value_features(feats), feats).item()   # identity when absent


def test_gate_key_present_and_shape_and_default_width():
    net = create_network(hidden=64, n_blocks=2, value_adapter=True)
    keys = {k for k, _ in tree_flatten(net.parameters())}
    assert "value_adapter.gate" in keys                  # saves under value_adapter.*
    assert net.value_adapter.gate.shape == (1,)           # not 0-d (safetensors-safe)
    assert float(net.value_adapter.gate[0]) == 0.0        # init 0
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
    return (mx.zeros((1, 24, 24, 30)), mx.zeros((1, 2), dtype=mx.int32),
            mx.zeros((1, 2), dtype=mx.int32), mx.ones((1, 2)))


def test_forward_padded_gate_zero_matches_raw_value_head():
    net = create_network(hidden=64, n_blocks=2, value_adapter=True)
    assert float(net.value_adapter.gate[0]) == 0.0
    board, rows, cols, mask = _board_moves()
    _, v_fwd, _ = net.forward_padded(board, rows, cols, mask, 24)
    cb, cr, cc, cm = canonicalize_batch(board, rows, cols, mask, 24)
    v_base = net.value_head(net.encoder(cb), 24)          # base path, no adapter
    assert mx.allclose(v_fwd, v_base).item()              # INVARIANT B: value == base at init


def test_forward_padded_value_reflects_gate():
    net = create_network(hidden=64, n_blocks=2, value_adapter=True)
    board, rows, cols, mask = _board_moves()
    _, v0, _ = net.forward_padded(board, rows, cols, mask, 24)
    net.value_adapter.gate = mx.array([2.0])
    _, v1, _ = net.forward_padded(board, rows, cols, mask, 24)
    assert not mx.allclose(v0, v1).item()                 # forward_padded actually applies the adapter


def test_policy_unaffected_by_adapter():
    # INVARIANT B: the policy path uses raw features -> independent of the gate.
    net = create_network(hidden=64, n_blocks=2, value_adapter=True)
    board, rows, cols, mask = _board_moves()
    p0, _, _ = net.forward_padded(board, rows, cols, mask, 24)
    net.value_adapter.gate = mx.array([5.0])
    p1, _, _ = net.forward_padded(board, rows, cols, mask, 24)
    assert mx.array_equal(p0, p1).item()                  # policy identical regardless of gate


def test_graft_load_succeeds_and_keeps_gate_zero(tmp_path):
    base = create_network(hidden=64, n_blocks=2)          # no adapter
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

Run: `.venv/bin/python -m pytest tests/test_v14_value_adapter.py -v`
Expected: FAIL — `ImportError: cannot import name '_load_base_weights_grafting_adapter'` (then, once network changes land, `TypeError: create_network() got an unexpected keyword argument 'value_adapter'`).

- [ ] **Step 3: Implement (network.py)**

**(a)** Add `ValueAdapter` immediately **before** `class AlphaZeroNetwork(nn.Module):`:

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

**(c)** In `forward_padded`, locate:

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

**(d)** In `__call__`, the empty-moves branch, locate:

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

**(e)** Extend `create_network`. Locate its signature `def create_network(\n    hidden: int = 128,\n    n_blocks: int = 6,\n    in_channels: Optional[int] = None,\n) -> AlphaZeroNetwork:` and add the two params before `) -> AlphaZeroNetwork:`:

```python
    value_adapter: bool = False,
    value_adapter_bottleneck_width: Optional[int] = None,
```

Then locate the return `return AlphaZeroNetwork(\n        in_channels=in_channels,\n        hidden=hidden,\n        n_blocks=n_blocks,\n    )` and add before `    )`:

```python
        value_adapter=value_adapter,
        value_adapter_bottleneck_width=value_adapter_bottleneck_width,
```

(`Optional`, `nn`, `mx` are already imported in network.py.)

- [ ] **Step 4: Implement (trainer.py graft helper)**

Locate `def freeze_batchnorm_running_stats(network) -> int:` near the top of `trainer.py` and add this module-level function immediately **above** it:

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

Ensure `tree_flatten` is importable in trainer.py: search for `tree_flatten`; if not already imported, add `from mlx.utils import tree_flatten` with the other top-of-file `mlx` imports.

- [ ] **Step 5: Run the new tests**

Run: `.venv/bin/python -m pytest tests/test_v14_value_adapter.py -v`
Expected: ALL 10 PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/network.py scripts/GPU/alphazero/trainer.py tests/test_v14_value_adapter.py
git commit -m "feat(network): v14 ValueAdapter module + opt-in value-only surface + graft-load

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Training surface + telemetry (trainer.py + calibration_pool.py + train.py)

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py`, `scripts/GPU/alphazero/calibration_pool.py`, `scripts/GPU/alphazero/train.py`
- Test: `tests/test_v14_value_adapter.py` (append)

**Interfaces:**
- Consumes: Task 1's `create_network(value_adapter=...)`, `network.value_adapter`, `_load_base_weights_grafting_adapter`.
- Produces: `ValueModule(value_head, value_adapter)`; `train_step(..., train_value_head_and_value_adapter=False, value_module=None)`; `train(..., value_adapter=False, value_adapter_bottleneck_width=None, train_value_head_and_value_adapter=False)`; `value_adapter_gate` + `value_adapter_grad_norm` telemetry both sites; CLI `--value-adapter` / `--value-adapter-bottleneck-width` / `--train-value-head-and-value-adapter`.

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
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_step(**{**_v14_kwargs(_adapter_net()), "train_value_head_only": True})


def test_v14_mutually_exclusive_with_v9():
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_step(**{**_v14_kwargs(_adapter_net()), "train_value_head_and_final_block": True})


def test_projection_rejected_on_adapter_surface():
    with pytest.raises(ValueError, match="requires --train-value-head-and-final-block"):
        train_step(**{**_v14_kwargs(_adapter_net()),
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
    assert not np.array_equal(before["value_adapter.gate"], after["value_adapter.gate"])  # gate moved
    assert any(not np.array_equal(before[k], after[k])
               for k in after if k.startswith("value_head."))                             # value head trained


def test_guardrail_hinge_sees_adapter():
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
    assert h0 != h1                                     # the hinge reads the adapter-corrected value


def test_build_block_emits_gate_and_grad_norm():
    from scripts.GPU.alphazero.calibration_pool import build_post_opening_calibration_block
    block = build_post_opening_calibration_block(
        config={}, enabled=True,
        loss_accumulator={"steps_done": 2, "value_adapter_gate": 0.37,
                          "sum_value_adapter_grad_norm": 0.5})
    assert block["loss"]["value_adapter_gate"] == pytest.approx(0.37)
    assert block["loss"]["value_adapter_grad_norm"] == pytest.approx(0.25)   # 0.5 / 2 steps


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
    assert "requires --value-adapter" in tsrc
    rsrc = open(trainer_mod.__file__).read()
    assert '"value_adapter_gate"' in rsrc
    assert '"sum_value_adapter_grad_norm"' in rsrc
    assert '"value_adapter_grad_norm"' in rsrc            # mirror tuple
    assert '"train_value_head_and_value_adapter": train_value_head_and_value_adapter,' in rsrc
    csrc = open(cp_mod.__file__).read()
    assert '"value_adapter_gate"' in csrc
    assert '"value_adapter_grad_norm"' in csrc
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_v14_value_adapter.py -k "v14 or projection_rejected or hinge_sees or build_block or wiring" -v`
Expected: FAIL — `ImportError: cannot import name 'ValueModule'`, then `TypeError` on the new `train_step` kwargs.

- [ ] **Step 3: Implement (trainer.py `ValueModule` + `train_step`)**

**(a)** Add `ValueModule` immediately after `class MainModule(nn.Module):` ends (after its `self.policy_head = policy_head` line, before the `# Per-size max moves table` comment):

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

**(b)** `train_step` signature. Locate `    post_opening_calibration_projection_strength: float = 1.0,   # v13c` (last param before `) -> tuple:`) and add below it:

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

**(e)** Value-side grad extraction + adapter grad norm. Locate:

```python
    value_grads = grads["value_head"]
```

Replace with:

```python
    if train_value_head_and_value_adapter:
        value_grads = {"value_head": grads["value_head"],
                       "value_adapter": grads["value_adapter"]}
        _, _agn = clip_grad_norm(grads["value_adapter"], max_norm=1e9)  # raw norm (no real clip)
        _adapter_grad_norm = float(_agn.item())
    else:
        value_grads = grads["value_head"]
        _adapter_grad_norm = 0.0
```

**(f)** Update region — opt_main skip. Locate:

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

**(g)** Update region — value update. Locate:

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

**(h)** Guardrail return — append the adapter grad norm for v14. Locate:

```python
    if calib_active and guardrail_mode:
        _guard_ret = (
            float(total_loss.item()), float(policy_loss.item()),
            float(value_loss.item()), float(l2_loss.item()),
            float(aux_loss.item()), float(aux_coverage), int(aux_n_eligible),
            float(calib_loss.item()), float(calib_value_mean.item()), int(calib_n),
            float(guardrail_hinge_loss.item()), float(guardrail_active_frac.item()),
            int(guardrail_n),
        )
        if _proj_telem is not None:
            return _guard_ret + (_proj_telem,)
        return _guard_ret
```

Replace with:

```python
    if calib_active and guardrail_mode:
        _guard_ret = (
            float(total_loss.item()), float(policy_loss.item()),
            float(value_loss.item()), float(l2_loss.item()),
            float(aux_loss.item()), float(aux_coverage), int(aux_n_eligible),
            float(calib_loss.item()), float(calib_value_mean.item()), int(calib_n),
            float(guardrail_hinge_loss.item()), float(guardrail_active_frac.item()),
            int(guardrail_n),
        )
        if _proj_telem is not None:
            return _guard_ret + (_proj_telem,)          # v13: dict at [13]
        if train_value_head_and_value_adapter:
            return _guard_ret + (_adapter_grad_norm,)   # v14: float at [13] (projection is off)
        return _guard_ret
```

- [ ] **Step 4: Implement (trainer.py `train()` wiring + telemetry)**

**(i)** `train()` signature. Locate `    post_opening_calibration_projection_strength: float = 1.0,   # v13c` (above `) -> AlphaZeroNetwork:`) and add below it:

```python
    value_adapter: bool = False,                              # v14
    value_adapter_bottleneck_width: Optional[int] = None,     # v14
    train_value_head_and_value_adapter: bool = False,         # v14
```

**(j)** `create_network` call. Locate:

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

**(k)** `value_module` construction. Locate:

```python
    main_module = MainModule(network.encoder, network.policy_head)
```

Add immediately below it:

```python
    # v14: value-side wrapper so opt_value updates value_head + value_adapter in
    # one call (None off).
    value_module = (ValueModule(network.value_head, network.value_adapter)
                    if train_value_head_and_value_adapter else None)
```

**(l)** Graft-load wiring. Locate:

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

**(m)** Per-iteration sum init. Locate:

```python
                sum_guardrail_active_frac = 0.0
```

Add immediately below it:

```python
                sum_value_adapter_grad_norm = 0.0   # v14
```

**(n)** Accumulate the adapter grad norm. Locate the guardrail 14-tuple accumulation block:

```python
                            if (_calib_guard_sign is not None
                                    and len(_ret) == 14):
                                sum_guardrail_hinge_loss += _ret[10]
                                sum_guardrail_active_frac += _ret[11]
                                proj = _ret[13]
                                if proj["skip_reason"] == "no_a":
```

Replace the head of it (down to `proj = _ret[13]`) with a flag-disambiguated form:

```python
                            if (_calib_guard_sign is not None
                                    and len(_ret) == 14):
                                sum_guardrail_hinge_loss += _ret[10]
                                sum_guardrail_active_frac += _ret[11]
                                if train_value_head_and_value_adapter:
                                    # v14: the 14th slot is the adapter grad norm
                                    # (float), NOT a projection dict.
                                    sum_value_adapter_grad_norm += float(_ret[13])
                                    proj = None
                                else:
                                    proj = _ret[13]
                                if proj is not None and proj["skip_reason"] == "no_a":
```

Then update the remaining `elif`/`else` chain that follows so it is guarded by `proj is not None`. Locate the chain:

```python
                                elif proj["skip_reason"] == "no_guardrail":
                                    proj_no_guardrail_steps += 1
                                elif proj["skip_reason"] == "tiny_guardrail":
                                    proj_tiny_guardrail_steps += 1
                                elif proj["conflict"]:
                                    proj_conflict_steps += 1
                                    sum_proj_dot += proj["dot"]
                                    sum_proj_cos += proj["cos"]
                                    sum_proj_c += proj["c"]
                                    sum_proj_removed_norm += proj["removed_norm"]
                                    sum_proj_norm_g += proj["norm_G"]
                                    sum_proj_norm_a += proj["norm_A"]
                                else:
                                    proj_no_conflict_steps += 1
                                    sum_proj_dot += proj["dot"]
                                    sum_proj_cos += proj["cos"]
                                    sum_proj_norm_g += proj["norm_G"]
                                    sum_proj_norm_a += proj["norm_A"]
```

Replace `elif proj["conflict"]:` with `elif proj is not None and proj["conflict"]:`, and the trailing `else:` with `elif proj is not None:`. (The `no_a`/`no_guardrail`/`tiny_guardrail` branches already begin with the `proj is not None and proj["skip_reason"] == ...` guard from edit (n) head, so make each of them `elif proj is not None and proj["skip_reason"] == ...`.) Final chain:

```python
                                if proj is not None and proj["skip_reason"] == "no_a":
                                    proj_no_a_steps += 1
                                elif proj is not None and proj["skip_reason"] == "no_guardrail":
                                    proj_no_guardrail_steps += 1
                                elif proj is not None and proj["skip_reason"] == "tiny_guardrail":
                                    proj_tiny_guardrail_steps += 1
                                elif proj is not None and proj["conflict"]:
                                    proj_conflict_steps += 1
                                    sum_proj_dot += proj["dot"]
                                    sum_proj_cos += proj["cos"]
                                    sum_proj_c += proj["c"]
                                    sum_proj_removed_norm += proj["removed_norm"]
                                    sum_proj_norm_g += proj["norm_G"]
                                    sum_proj_norm_a += proj["norm_A"]
                                elif proj is not None:
                                    proj_no_conflict_steps += 1
                                    sum_proj_dot += proj["dot"]
                                    sum_proj_cos += proj["cos"]
                                    sum_proj_norm_g += proj["norm_G"]
                                    sum_proj_norm_a += proj["norm_A"]
```

**(o)** `loss_accumulator` dict. Locate `                        "guardrail_margin": post_opening_guardrail_margin,` and add below it:

```python
                        "value_adapter_gate": (
                            float(network.value_adapter.gate[0])
                            if train_value_head_and_value_adapter else 0.0),
                        "sum_value_adapter_grad_norm": sum_value_adapter_grad_norm,
```

**(p)** Flattening mirror tuple. Locate:

```python
                    "calib_projection_no_conflict_steps", "calib_projection_strength")
                if k in _poc_loss}
```

Replace with:

```python
                    "calib_projection_no_conflict_steps", "calib_projection_strength",
                    "value_adapter_gate", "value_adapter_grad_norm")
                if k in _poc_loss}
```

**(q)** State dict run flag. Locate:

```python
            "unfrozen_block_index": (
                len(network.encoder.blocks) - 1
                if train_value_head_and_final_block else None),
```

Add immediately below it:

```python
            # v14: whether only value_head + value_adapter trained.
            "train_value_head_and_value_adapter": train_value_head_and_value_adapter,
```

- [ ] **Step 5: Implement (calibration_pool.py telemetry)**

Locate:

```python
            "guardrail_margin":
                float(loss_accumulator.get("guardrail_margin", 0.0)),
```

Add immediately below it:

```python
            "value_adapter_gate":
                float(loss_accumulator.get("value_adapter_gate", 0.0)),
            "value_adapter_grad_norm": (
                float(loss_accumulator.get("sum_value_adapter_grad_norm", 0.0))
                / max(int(loss_accumulator.get("steps_done", 0)), 1)),
```

- [ ] **Step 6: Implement (train.py CLI + guards + plumb)**

**(r)** CLI args. Locate the `--post-opening-calibration-projection-strength` `parser.add_argument(...)` block and add after it:

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

**(s)** `parser.error` guards. Locate:

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

**(t)** Plumb into `train(...)`. Locate:

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

- [ ] **Step 7: Run the new tests + regression**

Run: `.venv/bin/python -m pytest tests/test_v14_value_adapter.py tests/test_training.py tests/test_calibration_pool.py tests/test_calibration_cli_flags.py tests/test_gradient_projection.py tests/test_v13_projection_wiring.py -v`
Expected: ALL PASS (v14 green; v13 projection/wiring + training + calibration unchanged — the `len(_ret)==14` disambiguation is byte-identical when v14 is off).

- [ ] **Step 8: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py scripts/GPU/alphazero/calibration_pool.py scripts/GPU/alphazero/train.py tests/test_v14_value_adapter.py
git commit -m "feat(training): v14 value-adapter training surface + gate/grad_norm telemetry + CLI

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Verifier + smoke + tests + operator run

**Files:**
- Create: `scripts/GPU/alphazero/verify_value_head_and_adapter_checkpoint.py`, `scripts/GPU/alphazero/smoke_v14_value_adapter.py`
- Test: `tests/test_v14_verify_adapter_checkpoint.py` (create)

**Interfaces:**
- Consumes: Task 1's `create_network(value_adapter=True)`; Task 2's `train()`/CLI.
- Produces: `compare_value_head_and_adapter(base, candidate) -> dict`; verifier `main(argv)->int` (0/1/2/3); a runnable v14 smoke.

- [ ] **Step 1: Write the failing verifier tests**

Create `tests/test_v14_verify_adapter_checkpoint.py`:

```python
"""v14 verifier: a --train-value-head-and-value-adapter checkpoint changed ONLY
value_head.* + value_adapter.* vs its base; everything else byte-identical."""
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.verify_value_head_and_adapter_checkpoint import main


def _mutate(d, key, add):
    d = dict(d)
    d[key] = d[key] + add
    return d


def test_exit0_legal_value_head_and_adapter_delta(tmp_path):
    base = create_network(hidden=64, n_blocks=2)
    bp = tmp_path / "base.safetensors"
    base.save_weights(str(bp))
    cand = create_network(hidden=64, n_blocks=2, value_adapter=True)
    cand.load_weights(str(bp), strict=False)
    cand.value_adapter.gate = mx.array([0.5])
    cand.value_head.fc1.weight = cand.value_head.fc1.weight + 0.01
    cp = tmp_path / "cand.safetensors"
    cand.save_weights(str(cp))
    assert main(["--base", str(bp), "--candidate", str(cp)]) == 0


def test_exit1_frozen_leak(tmp_path):
    base = create_network(hidden=64, n_blocks=2)
    bp = tmp_path / "base.safetensors"
    base.save_weights(str(bp))
    cand = create_network(hidden=64, n_blocks=2, value_adapter=True)
    cand.load_weights(str(bp), strict=False)
    cand.value_adapter.gate = mx.array([0.5])
    cand.value_head.fc1.weight = cand.value_head.fc1.weight + 0.01
    d = dict(tree_flatten(cand.parameters()))
    pk = next(k for k in d if k.startswith("policy_head."))    # LEAK a policy tensor
    d = _mutate(d, pk, 0.02)
    cp = tmp_path / "cand.safetensors"
    mx.save_safetensors(str(cp), d)
    assert main(["--base", str(bp), "--candidate", str(cp)]) == 1


def test_exit2_gate_never_moved(tmp_path):
    base = create_network(hidden=64, n_blocks=2)
    bp = tmp_path / "base.safetensors"
    base.save_weights(str(bp))
    cand = create_network(hidden=64, n_blocks=2, value_adapter=True)
    cand.load_weights(str(bp), strict=False)                    # gate stays 0, value_head unchanged
    cp = tmp_path / "cand.safetensors"
    cand.save_weights(str(cp))
    assert main(["--base", str(bp), "--candidate", str(cp)]) == 2


def test_exit3_no_adapter_keys(tmp_path):
    base = create_network(hidden=64, n_blocks=2)
    bp = tmp_path / "base.safetensors"
    base.save_weights(str(bp))
    cand = create_network(hidden=64, n_blocks=2)                # NO adapter -> no new keys
    d = dict(tree_flatten(cand.parameters()))
    vk = next(k for k in d if k.startswith("value_head."))
    d = _mutate(d, vk, 0.01)
    cp = tmp_path / "cand.safetensors"
    mx.save_safetensors(str(cp), d)
    assert main(["--base", str(bp), "--candidate", str(cp)]) == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_v14_verify_adapter_checkpoint.py -v`
Expected: FAIL — `ModuleNotFoundError: ... verify_value_head_and_adapter_checkpoint`.

- [ ] **Step 3: Implement the verifier**

Create `scripts/GPU/alphazero/verify_value_head_and_adapter_checkpoint.py`:

```python
"""v14 acceptance check: prove a --train-value-head-and-value-adapter run touched
ONLY value_head.* and the (new) value_adapter.* tensors.

Compares two safetensors checkpoints. The base has NO value_adapter.* keys; the
candidate adds exactly the value_adapter.* set. Allowed to change: value_head.*
and value_adapter.*. Everything the two checkpoints SHARE — the stem, ALL
residual blocks (including the final one), the policy head, and ALL BatchNorm
running stats anywhere — must be byte-identical:
  exit 0  PASS: shared frozen set byte-identical; value head AND the gate moved
  exit 1  FAIL: a shared frozen tensor changed (a running-stat leak means
          --freeze-batchnorm-stats was missing/ineffective — run is invalid)
  exit 2  FAIL: no value_head tensor changed, or the gate never left 0.0 — the
          value-path correction never engaged (no-op)
  exit 3  FAIL: candidate-only keys are not exactly value_adapter.*, the base has
          keys the candidate lacks, or the candidate has no value_adapter.* keys
          at all (wrong architecture / flag mis-plumbed)
"""
from __future__ import annotations

import argparse
import sys

import mlx.core as mx

GATE_KEY = "value_adapter.gate"


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
        print(f"FAIL: value-path no-op (value_head changed="
              f"{_changed(r['value_head_deltas'])}, gate |abs|={r['gate_abs']:.3e}) "
              f"— the adapter correction never engaged")
        return 2
    print(f"PASS: {r['n_tensors']} base tensors; shared frozen set byte-identical; "
          f"value head + value_adapter (gate |abs|={r['gate_abs']:.3e}) trained")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the verifier tests**

Run: `.venv/bin/python -m pytest tests/test_v14_verify_adapter_checkpoint.py -v`
Expected: ALL 4 PASS (exit 0/1/2/3).

- [ ] **Step 5: Write the v14 smoke**

Create `scripts/GPU/alphazero/smoke_v14_value_adapter.py` — a tiny end-to-end check (no gitignored data): build an adapter net, freeze BN, run a handful of `train_step`s with a hand-built guardrail calibration batch, then assert the gate left 0 and the surface stayed isolated. Print `SMOKE OK` on success, exit non-zero on failure.

```python
"""Tiny end-to-end v14 smoke: adapter net + guardrail calibration -> gate opens,
surface isolated. Runs on synthetic data (no gitignored fixtures)."""
import sys
import numpy as np
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord
from scripts.GPU.alphazero.calibration_pool import target_in_to_move
from scripts.GPU.alphazero.trainer import (
    MainModule, ValueModule, train_step, freeze_batchnorm_running_stats)


def _pos():
    return PositionRecord(board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
                          to_move="red", legal_moves=[(0, 0), (1, 1), (2, 2)],
                          visit_counts=[10, 5, 3], outcome=1.0, active_size=24,
                          ply=0, game_n_moves=10)


def _guard_row(target_black):
    return PositionRecord(board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
                          to_move="black", legal_moves=[(0, 0), (1, 1)],
                          visit_counts=[0, 0], outcome=target_in_to_move("black", target_black),
                          active_size=24, ply=20, game_n_moves=None)


def main() -> int:
    mx.random.seed(0)
    net = create_network(hidden=64, n_blocks=2, value_adapter=True)
    freeze_batchnorm_running_stats(net)
    mm = MainModule(net.encoder, net.policy_head)
    vm = ValueModule(net.value_head, net.value_adapter)
    om, ov = optim.Adam(learning_rate=1e-2), optim.Adam(learning_rate=1e-2)
    calib = [_guard_row(0.9), _guard_row(-0.9)]
    sign = np.array([1.0, 1.0], dtype=np.float32)
    before = {k: np.array(v) for k, v in tree_flatten(net.parameters())}
    for _ in range(10):
        train_step(network=net, main_module=mm, opt_main=om, opt_value=ov,
                   batch=[_pos() for _ in range(3)], calibration_positions=calib,
                   calibration_loss_weight=0.01, calibration_guardrail_sign=sign,
                   guardrail_margin=0.10, train_value_head_and_value_adapter=True,
                   value_module=vm)
    after = {k: np.array(v) for k, v in tree_flatten(net.parameters())}
    gate = float(net.value_adapter.gate[0])
    if gate == 0.0:
        print(f"SMOKE FAIL: gate never left 0 (gate={gate})")
        return 1
    for k in after:
        if k.startswith("value_head.") or k.startswith("value_adapter."):
            continue
        if not np.array_equal(before[k], after[k]):
            print(f"SMOKE FAIL: frozen tensor changed: {k}")
            return 1
    print(f"SMOKE OK: gate={gate:.4f}; surface isolated (value_head + value_adapter only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the smoke**

Run: `.venv/bin/python -m scripts.GPU.alphazero.smoke_v14_value_adapter`
Expected: prints `SMOKE OK: gate=...; surface isolated ...`, exit 0.

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/verify_value_head_and_adapter_checkpoint.py scripts/GPU/alphazero/smoke_v14_value_adapter.py tests/test_v14_verify_adapter_checkpoint.py
git commit -m "feat(training): v14 verifier + end-to-end smoke

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 8: Full suite + merge (controller-run)**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected in the worktree: baseline-passed + the new v14 tests, with EXACTLY the known 14 failed + 6 errors (missing gitignored data). Then FF-merge to main, authoritative suite on merged main (1372 + new v14 tests, 0 failures), push (superpowers:finishing-a-development-branch). STOP after push — the operator run is the USER's.

---

## Operator run (USER's, after merge) — from the spec §11

The canonical v12b command **minus** `--train-value-head-and-final-block` and (if present) `--post-opening-calibration-gradient-projection`, **plus** `--value-adapter` (+ optional `--value-adapter-bottleneck-width 32`) and `--train-value-head-and-value-adapter`, with `--guardrail-margin 0.10 --freeze-batchnorm-stats`, new checkpoint dir `checkpoints/alphazero-v14-value-adapter-from-calib020-0001`. Confirm `train_value_head_and_value_adapter=true` + `value_adapter_gate` moving off 0 (+ `value_adapter_grad_norm` present) + `verify_value_head_and_adapter_checkpoint` exit 0; then gates A/B/C/D vs `calib020_0001` (no promotion unless all four pass). Interpretation: A moves + B/C/D hold → adapter capacity was the missing piece (promotion match); A moves + B/C/D drift → v14b adds projection over the adapter surface; A does not move → wider bottleneck arg-only (`--value-adapter-bottleneck-width 64`), then per-channel gate as a later written design.
```
