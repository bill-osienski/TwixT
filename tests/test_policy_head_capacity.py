"""Policy-head capacity pilot: the isolated residual head and the arms' trainable surfaces (CPU, random weights;
no checkpoint, search, game or real-target training). 24x24 positions from the rules engine."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from scripts.GPU.alphazero import policy_head_capacity as PHC  # noqa: E402
from scripts.GPU.alphazero.game import TwixtState  # noqa: E402
from scripts.GPU.alphazero.network import AlphaZeroNetwork, PolicyHead  # noqa: E402


def _batch():
    """Red's 24x24 opening (528 legal moves) and black to move after one red peg, padded to 528."""
    red = TwixtState(active_size=24, to_move="red", max_plies_limit=280)
    black = red.apply_move((12, 12))
    states = [red, black]
    legal = [s.legal_moves() for s in states]
    M = max(map(len, legal))
    assert len(legal[0]) == 528 and len(legal[1]) < M and black.to_move == "black"
    rows, cols, mask = (np.zeros((2, M), np.int32) for _ in range(3))
    for i, moves in enumerate(legal):
        rows[i, :len(moves)], cols[i, :len(moves)], mask[i, :len(moves)] = [m[0] for m in moves], [m[1] for m in moves], 1
    boards = np.stack([s.to_tensor().transpose(1, 2, 0) for s in states]).astype(np.float32)
    return mx.array(boards), mx.array(rows), mx.array(cols), mx.array(mask.astype(np.float32)), [len(m) for m in legal]


@pytest.fixture(scope="module")
def reference():
    mx.random.seed(0)
    net = AlphaZeroNetwork()
    net.policy_head.bn.running_mean = mx.array([0.3, -0.2])     # non-default, so copying them is observable
    net.policy_head.bn.running_var = mx.array([1.7, 0.6])
    net.eval()
    return net


def _clone(net):
    twin = AlphaZeroNetwork()
    twin.update(net.parameters())
    twin.eval()
    return twin


def _forward(net, batch):
    boards, rows, cols, mask, _ = batch
    logits, value, _ = net.forward_padded(boards, rows, cols, mask, active_size=24)
    return np.asarray(logits), np.asarray(value)


def _snapshot(net):
    return {k: np.array(v) for k, v in tree_flatten(net.parameters())}


@pytest.mark.parametrize("arm", ["C", "R"])
def test_each_arm_starts_exactly_equal_to_the_incumbent_head(reference, arm):
    batch = _batch()
    want_logits, want_value = _forward(reference, batch)
    net = _clone(reference)
    PHC.install(net, arm)
    got_logits, got_value = _forward(net, batch)
    np.testing.assert_array_equal(got_logits, want_logits)
    np.testing.assert_array_equal(got_value, want_value)
    np.testing.assert_array_equal(np.asarray(net.policy_head.bn.running_mean), np.float32([0.3, -0.2]))


def test_the_residual_path_is_live_and_keeps_full_legal_support(reference):
    batch = _batch()
    base, _ = _forward(reference, batch)
    net = _clone(reference)
    PHC.install(net, "R")
    mx.random.seed(1)
    net.policy_head.extra_out.weight = mx.random.normal(net.policy_head.extra_out.weight.shape)
    logits, _ = _forward(net, batch)
    for i, n in enumerate(batch[4]):
        assert np.all(np.isfinite(logits[i, :n])) and np.all(logits[i, :n] > -1e8)
        assert np.all(logits[i, n:] == np.float32(-1e9))
        assert not np.array_equal(logits[i, :n], base[i, :n])
        p = np.exp(logits[i, :n] - logits[i, :n].max())
        p /= p.sum()
        assert np.all(p > 0) and abs(p.sum() - 1) < 1e-5


@pytest.mark.parametrize("arm", ["C", "R"])
def test_two_steps_change_exactly_the_arms_surface(reference, arm):
    boards, rows, cols, mask, ns = batch = _batch()
    target = np.zeros(mask.shape, np.float32)
    for i, n in enumerate(ns):
        target[i, :n] = np.arange(1, n + 1, dtype=np.float32)
        target[i, :n] /= target[i, :n].sum()
    target = mx.array(target)
    net = _clone(reference)
    names = PHC.install(net, arm)
    before = _snapshot(net)

    def loss(model):
        logits, _, _ = model.forward_padded(boards, rows, cols, mask, active_size=24)
        return -mx.sum(target * (logits - mx.logsumexp(logits, axis=-1, keepdims=True))) / logits.shape[0]

    step = nn.value_and_grad(net, loss)
    opt = optim.Adam(learning_rate=1e-2)
    for _ in range(2):
        _, grads = step(net)
        opt.update(net, grads)
        mx.eval(net.parameters(), opt.state)
    after = _snapshot(net)
    assert set(after) == set(before)
    changed = {k for k in before if not np.array_equal(before[k], after[k])}
    assert changed == set(names) == set(PHC.SURFACES[arm][0])
    stats = [k for k in before if k.endswith(("running_mean", "running_var"))]
    assert len(stats) >= 4 and not changed & set(stats)
    assert net.training is False and net.policy_head.training is False
    assert _forward(net, batch)[0].shape == mask.shape


def test_refusals(reference):
    net = _clone(reference)
    with pytest.raises(ValueError, match="arm must be"):
        PHC.install(net, "X")
    PHC.install(net, "R")
    with pytest.raises(ValueError, match="installed twice"):
        PHC.install(net, "C")
    with pytest.raises(TypeError, match="not for search"):
        net.policy_head(None, [(0, 1)])
    assert type(reference.policy_head) is PolicyHead      # installing on a clone left the source alone
