"""POLICY-HEAD CAPACITY PILOT -- the isolated residual head (arm R) and the trainable surfaces of arms R and C.
Not production, not wired into any caller; no checkpoint, search or training here.

The existing PolicyHead scores a move from two numbers per cell (1x1 conv 128 -> 2, BN, ReLU, then 2 -> 64 -> 1).
ResidualPolicyHead keeps that path unchanged and adds a second one that reads the move cell's full encoder
features: logit = existing(features) + extra_out(relu(extra_fc(features at the move))), with extra_fc 128 -> 64
and extra_out 64 -> 1. extra_out starts at zero, so R starts exactly equal to the head it copies.

R's starting residual tensors come from an explicit seed (install(network, "R", init_seed=...)): extra_fc from
numpy.random.default_rng(seed), uniform on [-1/sqrt(128), 1/sqrt(128)] (MLX's default Linear range), weight then
bias, float32; extra_out zero. They never depend on MLX's global random state, so two installs with the same seed
start bitwise identical whatever else was drawn in between; residual_sha256() names them.

Surfaces (install() refuses anything else):
* C trains only the existing head's eight weight/bias arrays (519 parameters);
* R trains those 519 plus extra_fc / extra_out (8,321) = 8,840.
Everything else -- encoder, value head and every BatchNorm running statistic -- stays frozen, and the network is
left in eval mode. 24x24 enforcement belongs to the caller that builds positions.
"""
import hashlib
import math
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from .network import PolicyHead

EXTRA_HIDDEN = 64
BASE = tuple(sorted(f"policy_head.{m}.{p}" for m in ("conv", "bn", "fc", "out") for p in ("weight", "bias")))
EXTRA = tuple(sorted(f"policy_head.{m}.{p}" for m in ("extra_fc", "extra_out") for p in ("weight", "bias")))
SURFACES = {"C": (BASE, 519), "R": (tuple(sorted(BASE + EXTRA)), 519 + 8321)}
EXTRA_ARRAYS = ("extra_fc.weight", "extra_fc.bias", "extra_out.weight", "extra_out.bias")


def residual_init(seed: int, in_channels: int = 128) -> Dict[str, np.ndarray]:
    """R's starting residual tensors for `seed` (see the module docstring)."""
    if type(seed) is not int:
        raise ValueError(f"arm R needs an integer init_seed, got {seed!r}")
    rng, k = np.random.default_rng(seed), math.sqrt(1.0 / in_channels)
    weight = rng.uniform(-k, k, (EXTRA_HIDDEN, in_channels)).astype(np.float32)
    bias = rng.uniform(-k, k, (EXTRA_HIDDEN,)).astype(np.float32)
    return {"extra_fc.weight": weight, "extra_fc.bias": bias,
            "extra_out.weight": np.zeros((1, EXTRA_HIDDEN), np.float32), "extra_out.bias": np.zeros((1,), np.float32)}


def residual_sha256(arrays: Dict[str, np.ndarray]) -> str:
    """SHA-256 over the four residual arrays in EXTRA_ARRAYS order: name, dtype, shape, then the bytes."""
    h = hashlib.sha256()
    for name in EXTRA_ARRAYS:
        a = np.ascontiguousarray(arrays[name])
        h.update(f"{name}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def residual_arrays(head) -> Dict[str, np.ndarray]:
    params = dict(tree_flatten(head.parameters()))
    return {k: np.array(params[k]) for k in EXTRA_ARRAYS}


class ResidualPolicyHead(PolicyHead):
    def __init__(self, init_seed: int, in_channels: int = 128, hidden: int = 64):
        super().__init__(in_channels, hidden)
        self.extra_fc = nn.Linear(in_channels, EXTRA_HIDDEN)      # shapes only; the values are replaced below
        self.extra_out = nn.Linear(EXTRA_HIDDEN, 1)
        init = residual_init(init_seed, in_channels)
        self.extra_fc.weight, self.extra_fc.bias = mx.array(init["extra_fc.weight"]), mx.array(init["extra_fc.bias"])
        self.extra_out.weight, self.extra_out.bias = mx.array(init["extra_out.weight"]), mx.array(init["extra_out.bias"])

    def forward_padded(self, features, move_rows, move_cols, move_mask):
        base = super().forward_padded(features, move_rows, move_cols, move_mask)
        B = move_rows.shape[0]
        gathered = features[mx.arange(B, dtype=move_rows.dtype)[:, None], move_rows, move_cols, :]   # (B, M, C)
        extra = self.extra_out(nn.relu(self.extra_fc(gathered))).squeeze(-1)                         # (B, M)
        return mx.where(move_mask > 0.5, base + extra, self._neg_inf_f32)

    def __call__(self, *_a, **_k):
        raise TypeError("the capacity pilot's head is not for search")


def install(network, arm: str, *, init_seed: Optional[int] = None) -> List[str]:
    """Give `network` arm C's or arm R's policy head and freeze all but that arm's surface; returns its names.
    R copies the existing head's arrays (BN running statistics included) into a ResidualPolicyHead whose residual
    path starts at residual_init(init_seed); R requires an integer init_seed and C refuses one."""
    if arm not in SURFACES:
        raise ValueError(f"arm must be one of {sorted(SURFACES)}, got {arm!r}")
    if arm == "C" and init_seed is not None:
        raise ValueError("arm C has no residual path: init_seed must be None")
    if arm == "R" and type(init_seed) is not int:
        raise ValueError(f"arm R needs an integer init_seed, got {init_seed!r}")
    if type(network.policy_head) is not PolicyHead:
        raise ValueError("the network's policy head is not the original PolicyHead (installed twice?)")
    if arm == "R":
        head = ResidualPolicyHead(init_seed)
        head.update(network.policy_head.parameters())
        network.policy_head = head
    network.eval()
    network.freeze()
    network.policy_head.unfreeze()
    network.policy_head.bn.freeze(keys=["running_mean", "running_var"])
    flat = tree_flatten(network.trainable_parameters())
    names, size = tuple(sorted(k for k, _ in flat)), sum(v.size for _, v in flat)
    if (names, size) != SURFACES[arm]:
        raise ValueError(f"arm {arm}'s trainable surface is {size} parameters {names}, not {SURFACES[arm]}")
    return list(names)

