import mlx.core as mx
import pytest

from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.verify_value_head_and_final_block_checkpoint import (
    compare_value_head_and_final_block, main)


def _save(net, path):
    net.save_weights(str(path))
    return str(path)


@pytest.fixture()
def base_and_net(tmp_path):
    net = create_network(hidden=64, n_blocks=2)     # last block = blocks.1
    return _save(net, tmp_path / "base.safetensors"), net


def _bump_value_head(net):
    net.value_head.fc2.weight = net.value_head.fc2.weight + 0.01


def _bump_final_block(net):
    last = len(net.encoder.blocks) - 1
    b = net.encoder.blocks[last]
    b.conv1.weight = b.conv1.weight + 0.01


def _bump_early_block(net):
    net.encoder.blocks[0].conv1.weight = net.encoder.blocks[0].conv1.weight + 0.01


def _bump_policy(net):
    net.policy_head.conv.weight = net.policy_head.conv.weight + 0.01


def _bump_final_block_running_stat(net):
    last = len(net.encoder.blocks) - 1
    b = net.encoder.blocks[last]
    b.bn1.running_mean = b.bn1.running_mean + 0.01


def test_value_head_and_final_block_change_passes(tmp_path, base_and_net):
    base, net = base_and_net
    _bump_value_head(net)
    _bump_final_block(net)
    cand = _save(net, tmp_path / "cand.safetensors")
    report = compare_value_head_and_final_block(base, cand)
    assert report["frozen_diffs"] == []
    assert max(report["value_head_deltas"].values()) > 0
    assert max(report["final_block_deltas"].values()) > 0
    # 4 value-head tensors + 8 final-block trainable tensors (running stats
    # excluded) — the properties that matter; do NOT pin the total count.
    assert len(report["value_head_deltas"]) == 4
    assert len(report["final_block_deltas"]) == 8
    assert report["last_block_index"] == 1
    assert report["n_tensors"] > 0
    assert main(["--base", base, "--candidate", cand]) == 0


def test_early_block_change_fails_exit_1(tmp_path, base_and_net):
    base, net = base_and_net
    _bump_value_head(net); _bump_final_block(net); _bump_early_block(net)
    cand = _save(net, tmp_path / "cand.safetensors")
    report = compare_value_head_and_final_block(base, cand)
    assert "encoder.blocks.0.conv1.weight" in report["frozen_diffs"]
    assert main(["--base", base, "--candidate", cand]) == 1


def test_policy_change_fails_exit_1(tmp_path, base_and_net):
    base, net = base_and_net
    _bump_value_head(net); _bump_final_block(net); _bump_policy(net)
    cand = _save(net, tmp_path / "cand.safetensors")
    report = compare_value_head_and_final_block(base, cand)
    assert any(k.startswith("policy_head.") for k in report["frozen_diffs"])
    assert main(["--base", base, "--candidate", cand]) == 1


def test_final_block_running_stat_change_fails_exit_1(tmp_path, base_and_net):
    """A forgotten --freeze-batchnorm-stats moves the final block's running
    stats; those are NOT in the allowed set, so they must leak → exit 1."""
    base, net = base_and_net
    _bump_value_head(net); _bump_final_block(net)
    _bump_final_block_running_stat(net)
    cand = _save(net, tmp_path / "cand.safetensors")
    report = compare_value_head_and_final_block(base, cand)
    assert "encoder.blocks.1.bn1.running_mean" in report["frozen_diffs"]
    assert main(["--base", base, "--candidate", cand]) == 1


def test_identical_checkpoints_exit_2(tmp_path, base_and_net):
    base, net = base_and_net
    cand = _save(net, tmp_path / "cand.safetensors")
    assert main(["--base", base, "--candidate", cand]) == 2


def test_value_head_only_no_final_block_exit_3(tmp_path, base_and_net):
    """Value head moved but the final block did not — partial unfreeze never
    engaged (v9 collapsed to v8)."""
    base, net = base_and_net
    _bump_value_head(net)
    cand = _save(net, tmp_path / "cand.safetensors")
    report = compare_value_head_and_final_block(base, cand)
    assert max(report["value_head_deltas"].values()) > 0
    assert max(report["final_block_deltas"].values()) == 0
    assert main(["--base", base, "--candidate", cand]) == 3


def test_key_set_mismatch_raises(tmp_path, base_and_net):
    base, _ = base_and_net
    other = create_network(hidden=64, n_blocks=4)   # different key set
    cand = _save(other, tmp_path / "cand.safetensors")
    with pytest.raises(ValueError, match="key"):
        compare_value_head_and_final_block(base, cand)
