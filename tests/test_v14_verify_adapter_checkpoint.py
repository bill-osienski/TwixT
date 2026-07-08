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
