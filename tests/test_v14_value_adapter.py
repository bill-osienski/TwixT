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
