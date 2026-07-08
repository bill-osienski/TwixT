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
