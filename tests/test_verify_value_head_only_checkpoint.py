import mlx.core as mx
import pytest

from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.verify_value_head_only_checkpoint import (
    compare_value_head_only, main)


def _save(net, path):
    net.save_weights(str(path))
    return str(path)


@pytest.fixture()
def base_and_net(tmp_path):
    net = create_network(hidden=64, n_blocks=2)
    return _save(net, tmp_path / "base.safetensors"), net


def _bump_value_head(net):
    net.value_head.fc2.weight = net.value_head.fc2.weight + 0.01


def _bump_encoder(net):
    net.encoder.conv1.weight = net.encoder.conv1.weight + 0.01


def test_value_head_only_change_passes(tmp_path, base_and_net):
    base, net = base_and_net
    _bump_value_head(net)
    cand = _save(net, tmp_path / "cand.safetensors")
    report = compare_value_head_only(base, cand)
    assert report["frozen_diffs"] == []
    assert max(report["value_deltas"].values()) > 0
    # 4 value-head tensors (fc1.weight/bias, fc2.weight/bias) — the property
    # that matters; do NOT pin the total tensor count (schema drift is the
    # checkpoint format's concern, not this verifier's).
    assert len(report["value_deltas"]) == 4
    assert report["n_tensors"] > 0
    assert main(["--base", base, "--candidate", cand]) == 0


def test_encoder_change_fails_exit_1(tmp_path, base_and_net):
    base, net = base_and_net
    _bump_value_head(net)
    _bump_encoder(net)
    cand = _save(net, tmp_path / "cand.safetensors")
    report = compare_value_head_only(base, cand)
    assert "encoder.conv1.weight" in report["frozen_diffs"]
    assert main(["--base", base, "--candidate", cand]) == 1


def test_identical_checkpoints_exit_2(tmp_path, base_and_net):
    base, net = base_and_net
    cand = _save(net, tmp_path / "cand.safetensors")
    assert main(["--base", base, "--candidate", cand]) == 2


def test_key_set_mismatch_raises(tmp_path, base_and_net):
    base, _ = base_and_net
    other = create_network(hidden=64, n_blocks=4)      # different key set
    cand = _save(other, tmp_path / "cand.safetensors")
    with pytest.raises(ValueError, match="key"):
        compare_value_head_only(base, cand)
