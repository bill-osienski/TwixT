"""probe_eval's loader returns the network in evaluation mode (BatchNorm on running statistics), for both
the public load_network_for_scoring and the probe CLI's direct _load_network call."""
import mlx.nn as nn

from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.probe_eval import _load_network, load_network_for_scoring


def _in_eval_mode(net):
    mods = net.named_modules()
    assert any(isinstance(m, nn.BatchNorm) for _, m in mods)
    return not net.training and not any(m.training for _, m in mods)


def test_loaders_return_eval_mode(tmp_path):
    p = str(tmp_path / "w.safetensors")
    create_network().save_weights(p)
    assert _in_eval_mode(load_network_for_scoring(p)[0])
    assert _in_eval_mode(_load_network(p, verbose=False)[0])
