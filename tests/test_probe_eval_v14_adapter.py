"""probe_eval loader must reconstruct v14 value-adapter checkpoints correctly.

The eval/probe loader (_load_network, shared by eval_runner / eval_position_probe
/ eval_raw_nn_position_rows / the manifest builders) rebuilds the network from a
checkpoint independently of training. For a v14 checkpoint it must build the
adapter — at the checkpoint's ACTUAL bottleneck width, not just hidden//4 — or the
value path would silently drop the adapter correction (wrong value) or strict-load
fail. For base / pre-v14 checkpoints it must stay adapter-free (byte-identical
behavior).
"""
import mlx.core as mx

from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.probe_eval import _load_network, _detect_value_adapter


def test_detect_value_adapter_true_only_when_keys_present(tmp_path):
    v14 = create_network(hidden=64, n_blocks=2, value_adapter=True)
    base = create_network(hidden=64, n_blocks=2)
    pv, pb = str(tmp_path / "v14.safetensors"), str(tmp_path / "base.safetensors")
    v14.save_weights(pv)
    base.save_weights(pb)
    assert _detect_value_adapter(pv) is True
    assert _detect_value_adapter(pb) is False


def test_load_network_builds_adapter_at_nondefault_width(tmp_path):
    # width 8 != hidden//4 (16): a loader that ignored the checkpoint width and
    # fell back to the default would strict-load-fail here — so this pins the
    # width-detection hardening, not just adapter presence.
    net = create_network(hidden=64, n_blocks=2, value_adapter=True,
                         value_adapter_bottleneck_width=8)
    p = str(tmp_path / "v14_w8.safetensors")
    net.save_weights(p)
    loaded, _in_ch, _h, _nb = _load_network(p, hidden=64, n_blocks=2, verbose=False)
    assert loaded.value_adapter is not None
    assert loaded.value_adapter.fc_down.weight.shape[0] == 8       # width detected
    # strict load succeeded ⇒ the grafted value_head + adapter weights match the file
    assert mx.array_equal(loaded.value_adapter.fc_down.weight,
                          net.value_adapter.fc_down.weight).item()


def test_load_network_base_checkpoint_has_no_adapter(tmp_path):
    net = create_network(hidden=64, n_blocks=2)                    # no adapter
    p = str(tmp_path / "base.safetensors")
    net.save_weights(p)
    loaded, _in_ch, _h, _nb = _load_network(p, hidden=64, n_blocks=2, verbose=False)
    assert loaded.value_adapter is None                           # old behavior preserved
