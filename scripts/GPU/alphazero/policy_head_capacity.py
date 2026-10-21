"""POLICY-HEAD CAPACITY PILOT -- the isolated residual head (arm R) and the trainable surfaces of arms R and C.
Not production, not wired into any caller; no checkpoint, search or training here.

The existing PolicyHead scores a move from two numbers per cell (1x1 conv 128 -> 2, BN, ReLU, then 2 -> 64 -> 1).
ResidualPolicyHead keeps that path unchanged and adds a second one that reads the move cell's full encoder
features: logit = existing(features) + extra_out(relu(extra_fc(features at the move))), with extra_fc 128 -> 64
and extra_out 64 -> 1. extra_out starts at zero, so R starts exactly equal to the head it copies.

Surfaces (install() refuses anything else):
* C trains only the existing head's eight weight/bias arrays (519 parameters);
* R trains those 519 plus extra_fc / extra_out (8,321) = 8,840.
Everything else -- encoder, value head and every BatchNorm running statistic -- stays frozen, and the network is
left in eval mode. 24x24 enforcement belongs to the caller that builds positions.
"""
from typing import List

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .network import PolicyHead

EXTRA_HIDDEN = 64
BASE = tuple(sorted(f"policy_head.{m}.{p}" for m in ("conv", "bn", "fc", "out") for p in ("weight", "bias")))
EXTRA = tuple(sorted(f"policy_head.{m}.{p}" for m in ("extra_fc", "extra_out") for p in ("weight", "bias")))
SURFACES = {"C": (BASE, 519), "R": (tuple(sorted(BASE + EXTRA)), 519 + 8321)}


class ResidualPolicyHead(PolicyHead):
    def __init__(self, in_channels: int = 128, hidden: int = 64):
        super().__init__(in_channels, hidden)
        self.extra_fc = nn.Linear(in_channels, EXTRA_HIDDEN)
        self.extra_out = nn.Linear(EXTRA_HIDDEN, 1)
        self.extra_out.weight = mx.zeros_like(self.extra_out.weight)
        self.extra_out.bias = mx.zeros_like(self.extra_out.bias)

    def forward_padded(self, features, move_rows, move_cols, move_mask):
        base = super().forward_padded(features, move_rows, move_cols, move_mask)
        B = move_rows.shape[0]
        gathered = features[mx.arange(B, dtype=move_rows.dtype)[:, None], move_rows, move_cols, :]   # (B, M, C)
        extra = self.extra_out(nn.relu(self.extra_fc(gathered))).squeeze(-1)                         # (B, M)
        return mx.where(move_mask > 0.5, base + extra, self._neg_inf_f32)

    def __call__(self, *_a, **_k):
        raise TypeError("the capacity pilot's head is not for search")


def install(network, arm: str) -> List[str]:
    """Give `network` arm C's or arm R's policy head and freeze all but that arm's surface; returns its names.
    R copies the existing head's arrays (BN running statistics included) into a ResidualPolicyHead."""
    if arm not in SURFACES:
        raise ValueError(f"arm must be one of {sorted(SURFACES)}, got {arm!r}")
    if type(network.policy_head) is not PolicyHead:
        raise ValueError("the network's policy head is not the original PolicyHead (installed twice?)")
    if arm == "R":
        head = ResidualPolicyHead()
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

