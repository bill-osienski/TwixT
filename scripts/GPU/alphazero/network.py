"""AlphaZero network with dual policy/value heads.

Architecture:
- BoardEncoder: CNN + 6 ResBlocks (NUM_CHANNELS input, 128 hidden)
- PolicyHead: Gather-based, outputs one logit per legal move
- ValueHead: Predicts win probability in [-1, 1]

Input: (B, H, W, C) tensor where C=NUM_CHANNELS (MLX channels-last format)
Output: policy logits (N,) and value scalar

TENSOR LAYOUT CONTRACT:
- MLX uses NHWC (channels-last): (B, H, W, C)
- All indexing uses x[batch, row, col, channel]
- PyTorch/ONNX use NCHW - conversion happens ONLY in export_onnx.py
- See docs/alphazero-twixt.md "Tensor Layout Contract" for details
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .game import BOARD_SIZE, NUM_CHANNELS


# =============================================================================
# Canonicalization: Make network always see "current player" connecting top↔bottom
# =============================================================================

# Channel layout constants (prevents off-by-one bugs on refactor)
CH_RED_PEG = 0
CH_BLACK_PEG = 1
CH_RED_LINK0 = 2          # 2..9  (8 directions)
CH_BLACK_LINK0 = 10       # 10..17 (8 directions)
CH_TO_MOVE = 18
CH_RED_TOP = 19
CH_RED_BOTTOM = 20
CH_BLACK_LEFT = 21
CH_BLACK_RIGHT = 22
CH_PHASE = 23
# Connectivity channels (Task 11, NUM_CHANNELS=30)
CH_RED_CONN_TOP = 24
CH_RED_CONN_BOTTOM = 25
CH_RED_CONN_BOTH = 26
CH_BLACK_CONN_LEFT = 27
CH_BLACK_CONN_RIGHT = 28
CH_BLACK_CONN_BOTH = 29

# 90° CW rotation: (dr, dc) → (dc, -dr)
# LINK_PERM_CW[old_channel] = new_channel
LINK_PERM_CW = [6, 7, 0, 1, 2, 3, 4, 5]

# Inverse: INV_LINK_PERM_CW[new_channel] = old_channel
INV_LINK_PERM_CW = [0] * 8
for _old, _new in enumerate(LINK_PERM_CW):
    INV_LINK_PERM_CW[_new] = _old
# Result: [2, 3, 4, 5, 6, 7, 0, 1]

# Precompute channel index tensors (avoid per-call allocation, ~30k calls/iter)
# IDX_* are mx.array so mx.take stays device-side (no CPU→GPU transfer per call)
IDX_CUR_LINKS = mx.array(
    [CH_BLACK_LINK0 + INV_LINK_PERM_CW[d] for d in range(8)], dtype=mx.int32
)
IDX_OPP_LINKS = mx.array(
    [CH_RED_LINK0 + INV_LINK_PERM_CW[d] for d in range(8)], dtype=mx.int32
)


def canonicalize_batch(
    boards: mx.array,
    move_rows: mx.array,
    move_cols: mx.array,
    move_mask: mx.array,
    active_size: int,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """Canonicalize batch so network always sees current-player connecting top↔bottom.

    For red-to-move: no change (red already connects top↔bottom)
    For black-to-move: rotate 90° CW, swap colors, permute link channels

    Args:
        boards: (B, H, W, C) tensor, H=W=24, C=NUM_CHANNELS
        move_rows: (B, M) legal move row coords
        move_cols: (B, M) legal move col coords
        move_mask: (B, M) valid move mask (1=valid, 0=padded)
        active_size: curriculum board size (e.g., 20)

    Returns:
        Tuple of (boards, move_rows, move_cols, move_mask) - canonicalized
    """
    B, H, W, C = boards.shape

    # Detect black-to-move: CH_TO_MOVE == 0 means black to move
    # CH_TO_MOVE is uniform across spatial dims, sample [0,0]
    is_black = boards[:, 0, 0, CH_TO_MOVE] < 0.5  # (B,)

    # NOTE: No early exit - mx.any() returns mx.array, not bool, and would
    # force a device→host sync. Always compute canonical form, then select.

    # Shared ones tensor for CH_TO_MOVE (reused for both paths)
    ones18 = mx.ones((B, H, W, 1), dtype=boards.dtype)

    # === Build red-fixed boards (CH_TO_MOVE forced to 1, but no rotation) ===
    boards_red_fixed = mx.concatenate([
        boards[:, :, :, :CH_TO_MOVE],
        ones18,
        boards[:, :, :, CH_TO_MOVE + 1:],
    ], axis=3)

    # === Build black-canonical boards ===

    # Step 1: Rotate ONLY the active region 90° CW: (r, c) -> (c, S-1-r)
    # This ensures board rotation matches move coord rotation (both use active_size)
    # BUG FIX: Previously rotated full 24x24, causing (r,c) -> (c, 23-r) mismatch
    S = active_size

    active = boards[:, :S, :S, :]                    # (B, S, S, C)
    active_rot = mx.transpose(active, (0, 2, 1, 3))  # swap H,W within active
    active_rot = active_rot[:, :, ::-1, :]           # CW within SxS

    # Paste rotated active region back into HxW tensor (zeros elsewhere)
    if W > S:
        right_pad = mx.zeros((B, S, W - S, C), dtype=boards.dtype)
        top = mx.concatenate([active_rot, right_pad], axis=2)  # width pad on axis=2
    else:
        top = active_rot

    if H > S:
        bottom = mx.zeros((B, H - S, W, C), dtype=boards.dtype)
        boards_rot = mx.concatenate([top, bottom], axis=1)  # height pad on axis=1
    else:
        boards_rot = top

    # Step 2: Build canonical channel order for black-to-move
    # Channel mapping:
    #   new 0  = old 1   (black pegs -> current pegs)
    #   new 1  = old 0   (red pegs -> opponent pegs)
    #   new 2-9  = old 10-17 (black links -> current links), permuted
    #   new 10-17 = old 2-9 (red links -> opponent links), permuted
    #   new 18 = ones
    #   new 19 = old 21  (black-left dist -> current-top)
    #   new 20 = old 22  (black-right dist -> current-bottom)
    #   new 21 = old 20  (red-bottom dist -> opp-left)
    #   new 22 = old 19  (red-top dist -> opp-right)
    #   new 23 = old 23  (phase unchanged)
    #   new 24 = old 27  (black_conn_left -> current_conn_top)
    #   new 25 = old 28  (black_conn_right -> current_conn_bottom)
    #   new 26 = old 29  (black_conn_both -> current_conn_both)
    #   new 27 = old 25  (red_conn_bottom -> opp_conn_left)
    #   new 28 = old 24  (red_conn_top -> opp_conn_right)
    #   new 29 = old 26  (red_conn_both -> opp_conn_both)

    # Pegs: swap red/black
    ch_pegs = mx.concatenate([
        boards_rot[:, :, :, CH_BLACK_PEG:CH_BLACK_PEG + 1],   # new 0 = old 1 (black pegs)
        boards_rot[:, :, :, CH_RED_PEG:CH_RED_PEG + 1],       # new 1 = old 0 (red pegs)
    ], axis=3)

    # Current player links (from old black links ch 10-17, permuted)
    # Use precomputed IDX_CUR_LINKS for efficient gather (no per-call allocation)
    ch_cur_links = mx.take(boards_rot, IDX_CUR_LINKS, axis=3)  # (B, H, W, 8)

    # Opponent links (from old red links ch 2-9, permuted)
    ch_opp_links = mx.take(boards_rot, IDX_OPP_LINKS, axis=3)  # (B, H, W, 8)

    # Channel 18: reuse shared ones tensor
    ch_18 = ones18

    # Distance channels remapped
    ch_dists = mx.concatenate([
        boards_rot[:, :, :, CH_BLACK_LEFT:CH_BLACK_LEFT + 1],    # new 19 (black-left -> top)
        boards_rot[:, :, :, CH_BLACK_RIGHT:CH_BLACK_RIGHT + 1],  # new 20 (black-right -> bottom)
        boards_rot[:, :, :, CH_RED_BOTTOM:CH_RED_BOTTOM + 1],    # new 21 (red-bottom -> left)
        boards_rot[:, :, :, CH_RED_TOP:CH_RED_TOP + 1],          # new 22 (red-top -> right)
    ], axis=3)

    # Phase unchanged
    ch_phase = boards_rot[:, :, :, CH_PHASE:CH_PHASE + 1]

    # Connectivity channels remapped (same semantics as distance swap):
    # new 24 (current_conn_top)    <- old 27 (black_conn_left), rotated
    # new 25 (current_conn_bottom) <- old 28 (black_conn_right), rotated
    # new 26 (current_conn_both)   <- old 29 (black_conn_both), rotated
    # new 27 (opp_conn_left)       <- old 25 (red_conn_bottom), rotated
    # new 28 (opp_conn_right)      <- old 24 (red_conn_top), rotated
    # new 29 (opp_conn_both)       <- old 26 (red_conn_both), rotated
    ch_conn = mx.concatenate([
        boards_rot[:, :, :, CH_BLACK_CONN_LEFT:CH_BLACK_CONN_LEFT + 1],    # new 24
        boards_rot[:, :, :, CH_BLACK_CONN_RIGHT:CH_BLACK_CONN_RIGHT + 1],  # new 25
        boards_rot[:, :, :, CH_BLACK_CONN_BOTH:CH_BLACK_CONN_BOTH + 1],    # new 26
        boards_rot[:, :, :, CH_RED_CONN_BOTTOM:CH_RED_CONN_BOTTOM + 1],    # new 27
        boards_rot[:, :, :, CH_RED_CONN_TOP:CH_RED_CONN_TOP + 1],          # new 28
        boards_rot[:, :, :, CH_RED_CONN_BOTH:CH_RED_CONN_BOTH + 1],        # new 29
    ], axis=3)

    # Assemble canonical board
    boards_black_canon = mx.concatenate([
        ch_pegs,       # 0-1
        ch_cur_links,  # 2-9
        ch_opp_links,  # 10-17
        ch_18,         # 18
        ch_dists,      # 19-22
        ch_phase,      # 23
        ch_conn,       # 24-29
    ], axis=3)

    # Step 3: Select per-sample using mx.where (no branching)
    # Select between boards_black_canon (for black) and boards_red_fixed (for red)
    # Both have CH_TO_MOVE forced to 1, avoiding a second concat
    sel = is_black[:, None, None, None]  # (B, 1, 1, 1) for broadcasting
    boards_out = mx.where(sel, boards_black_canon, boards_red_fixed)

    # === Rotate move coordinates ===
    # CW rotation: (r, c) -> (c, active_size - 1 - r)
    # Ensure int32 dtype in coord math
    rows_rot = move_cols
    a = mx.array(active_size - 1, dtype=move_rows.dtype)
    cols_rot = a - move_rows

    # Only apply to valid black-to-move entries
    valid = (move_mask > 0.5) & is_black[:, None]  # (B, M)
    move_rows_out = mx.where(valid, rows_rot, move_rows)
    move_cols_out = mx.where(valid, cols_rot, move_cols)

    # Mask unchanged
    return boards_out, move_rows_out, move_cols_out, move_mask


class ResBlock(nn.Module):
    """Residual block with skip connection.

    Architecture: conv -> bn -> relu -> conv -> bn -> + residual -> relu
    Input/Output: (B, H, W, C) - channels-last format
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(channels)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return nn.relu(x + residual)


class BoardEncoder(nn.Module):
    """CNN encoder for board state.

    Architecture:
    - Initial conv 3x3 with batch norm
    - N residual blocks (default 6)

    Input: (B, H, W, C) where C=in_channels (channels-last, NHWC)
    Output: (B, H, W, hidden) feature maps (channels-last)
    """

    def __init__(
        self,
        in_channels: int = NUM_CHANNELS,
        hidden: int = 128,
        n_blocks: int = 6,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(hidden)

        self.blocks = [ResBlock(hidden) for _ in range(n_blocks)]

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: (B, H, W, C) board tensor (channels-last)

        Returns:
            (B, H, W, hidden) encoded features (channels-last)
        """
        x = nn.relu(self.bn1(self.conv1(x)))
        for block in self.blocks:
            x = block(x)

        return x  # (B, H, W, hidden)


NEG_INF = -1e9  # Constant for masked logits


class PolicyHead(nn.Module):
    """Gather-based policy head - outputs one logit per legal move.

    Supports both:
    - Batched padded moves via forward_padded() for training
    - Single-position list API via __call__() for MCTS inference

    Architecture:
    - 1x1 conv to reduce channels (128 -> 2)
    - Vectorized gather at move locations
    - FC -> relu -> FC -> logit

    Input: (B, H, W, C) features (channels-last)
    """

    def __init__(self, in_channels: int = 128, hidden: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2, kernel_size=1)
        self.bn = nn.BatchNorm(2)
        self.fc = nn.Linear(2, hidden)
        self.out = nn.Linear(hidden, 1)
        # Pre-allocate constant to avoid per-call allocation
        self._neg_inf_f32 = mx.array(NEG_INF, dtype=mx.float32)

    def forward_padded(
        self,
        features: mx.array,   # (B, H, W, C)
        move_rows: mx.array,  # (B, M) int32
        move_cols: mx.array,  # (B, M) int32
        move_mask: mx.array,  # (B, M) float32 1/0
    ) -> mx.array:
        """Batched forward with padded moves.

        Args:
            features: (B, H, W, C) encoded board features
            move_rows: (B, M) row indices for each move
            move_cols: (B, M) column indices for each move
            move_mask: (B, M) 1.0 for valid moves, 0.0 for padding

        Returns:
            logits: (B, M) with NEG_INF for padded positions
        """
        # Reduce channels: (B, H, W, C) -> (B, H, W, 2)
        x = nn.relu(self.bn(self.conv(features)))  # NHWC

        B, M = move_rows.shape

        if B == 1:
            # Fast path for MCTS inference: avoid mx.arange allocation
            gathered = x[0, move_rows[0], move_cols[0], :]    # (M, 2)
            h = nn.relu(self.fc(gathered))                    # (M, hidden)
            logits = self.out(h).squeeze(-1)                  # (M,)
            logits = mx.where(move_mask[0] > 0.5, logits, self._neg_inf_f32)
            return logits[None, :]                            # (1, M)

        # General B>1 path for training
        b_idx = mx.arange(B, dtype=move_rows.dtype)[:, None]  # (B, 1)
        gathered = x[b_idx, move_rows, move_cols, :]          # (B, M, 2)

        # MLP over last dim; flatten then reshape
        h = nn.relu(self.fc(gathered.reshape(B * M, 2)))      # (B*M, hidden)
        logits = self.out(h).reshape(B, M)                    # (B, M)

        # Mask padded positions to NEG_INF (so softmax ignores them)
        logits = mx.where(move_mask > 0.5, logits, self._neg_inf_f32)
        return logits

    def __call__(
        self,
        features: mx.array,
        moves: List[Tuple[int, int]],
    ) -> mx.array:
        """Single-position API for MCTS inference.

        Args:
            features: (1, H, W, C) encoded board features
            moves: List of (row, col) legal moves

        Returns:
            (N,) logits, one per move
        """
        N = len(moves)
        if N == 0:
            return mx.array([])

        # Convert to padded format and call vectorized forward
        rows = mx.array([r for r, _ in moves], dtype=mx.int32)[None, :]  # (1, N)
        cols = mx.array([c for _, c in moves], dtype=mx.int32)[None, :]  # (1, N)
        mask = mx.ones((1, N), dtype=mx.float32)

        logits = self.forward_padded(features, rows, cols, mask)  # (1, N)
        return logits[0]  # (N,)


class ValueHead(nn.Module):
    """Value head with curriculum-aware masked pooling.

    Architecture (global pooling based - layout agnostic):
    - Masked global average pool + masked global max pool over spatial dims
    - Concatenate pooled features -> (B, 2*C)
    - FC -> relu -> FC -> tanh

    When active_size < 24, only pools over the active region to avoid
    padded zeros corrupting the value estimate:
    - Avg pool: zeros would bias mean toward neutral
    - Max pool: zeros could become max if real features are negative

    Output: scalar in [-1, 1] where +1 = current player wins
    Input: (B, H, W, C) features (channels-last, NHWC)
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden: int = 256,
    ):
        super().__init__()
        # 2*in_channels because we concat avg and max pooled features
        self.fc1 = nn.Linear(2 * in_channels, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def __call__(
        self, features: mx.array, active_size: int = 24, return_pretanh: bool = False
    ) -> mx.array:
        """Forward pass with masked pooling.

        Args:
            features: (B, H, W, C) encoded board (channels-last, NHWC)
            active_size: Active board region (1 to 24). Only pool within
                         rows/cols [0, active_size). Default 24 = full board.
            return_pretanh: If True, return (value, pretanh) for diagnostics

        Returns:
            (B,) value in [-1, 1], or (value, pretanh) tuple if return_pretanh=True
        """
        B, H, W, C = features.shape

        # Build spatial mask: 1 inside active area, 0 outside
        # Shape: (1, H, W, 1) for broadcasting with (B, H, W, C)
        rows = mx.arange(H)[None, :, None, None]  # (1, H, 1, 1)
        cols = mx.arange(W)[None, None, :, None]  # (1, 1, W, 1)
        mask = ((rows < active_size) & (cols < active_size)).astype(features.dtype)
        # mask shape: (1, H, W, 1)

        # Masked average pooling: sum / count (not mean over full 24×24)
        # CAREFUL with shapes: mask sum is (1, 1), need to broadcast correctly
        masked = features * mask  # (B, H, W, C)
        denom = mx.sum(mask, axis=(1, 2))  # (1, 1)
        denom = denom[:, 0] + 1e-8  # (1,) - flatten to 1D
        avg_pool = mx.sum(masked, axis=(1, 2)) / denom[:, None]  # (B, C)

        # Masked max pooling: set outside-active to large negative before max
        # Use dtype-safe neg_inf to avoid backend broadcasting quirks
        neg_inf = mx.full(features.shape, -1e9, dtype=features.dtype)
        masked_for_max = mx.where(mask > 0.5, features, neg_inf)  # (B, H, W, C)
        max_pool = mx.max(masked_for_max, axis=(1, 2))  # (B, C)

        # Concatenate pooled features, force float32 throughout
        x = mx.concatenate([avg_pool, max_pool], axis=-1).astype(mx.float32)  # (B, 2*C)

        # MLP to scalar (explicit fp32 after each layer for stability)
        h = nn.relu(self.fc1(x)).astype(mx.float32)
        pre = self.fc2(h).astype(mx.float32)

        # Soft clamp: bounds pretanh to (-CLAMP, CLAMP) but preserves gradients
        # tanh(x/C)*C approaches ±C asymptotically with gradient that decays but never zeros
        # Hard clip kills gradients at boundary; soft clamp allows learning to continue
        # Note: this compresses mid-range (e.g., pre=10 -> ~7.6), which is fine
        PRETANH_CLAMP = 10.0
        pre = PRETANH_CLAMP * mx.tanh(pre / PRETANH_CLAMP)

        # Robust (B,) output - handle (B,1), (B,), or unexpected shapes
        if pre.ndim == 2 and pre.shape[1] == 1:
            pre_1d = pre[:, 0]
        elif pre.ndim == 1:
            pre_1d = pre
        else:
            # Unexpected shape; last-resort flatten to (B,) using known B
            pre_1d = pre.reshape((B,))
        value = mx.tanh(pre_1d)

        if return_pretanh:
            return value, pre_1d
        return value


class ValueAdapter(nn.Module):
    """v14: small value-only feature-correction adapter (pointwise 1x1 bottleneck
    + folded scalar gate). Inserted between the shared encoder features and the
    value head so it corrects ONLY the value path (policy path untouched).

    __call__(features) -> gate * fc_up(relu(fc_down(features)))   over (B,H,W,C).
    The gate is init 0.0 (identity at init; ReZero-style bootstrap) and stored as
    shape (1,) so it saves/loads under the key "value_adapter.gate" (MLX
    safetensors are cleanest with >=1-d arrays).
    """

    def __init__(self, channels: int, bottleneck_width: Optional[int] = None):
        super().__init__()
        b = bottleneck_width if bottleneck_width else channels // 4
        self.fc_down = nn.Linear(channels, b)
        self.fc_up = nn.Linear(b, channels)
        self.gate = mx.zeros((1,))

    def __call__(self, features: mx.array) -> mx.array:
        h = nn.relu(self.fc_down(features))
        return self.gate * self.fc_up(h)


class AlphaZeroNetwork(nn.Module):
    """Combined network with shared encoder (in_channels defaults to NUM_CHANNELS).

    Architecture:
    - Shared BoardEncoder (CNN + ResBlocks)
    - PolicyHead (gather-based, N logits)
    - ValueHead (scalar output)

    Supports both:
    - Batched padded forward via forward_padded() for training
    - Single-position list API via __call__() for MCTS inference

    Input: (B, H, W, C) board tensor, list of legal moves
    Output: policy logits (N,), value scalar
    """

    def __init__(
        self,
        in_channels: int = NUM_CHANNELS,
        hidden: int = 128,
        n_blocks: int = 6,
        value_adapter: bool = False,
        value_adapter_bottleneck_width: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.encoder = BoardEncoder(in_channels, hidden, n_blocks)
        self.policy_head = PolicyHead(hidden)
        self.value_head = ValueHead(hidden)
        # v14: opt-in value-only adapter (None when off -> byte-identical, no
        # value_adapter.* params).
        self.value_adapter = (
            ValueAdapter(hidden, value_adapter_bottleneck_width)
            if value_adapter else None)

    def _value_features(self, features: mx.array) -> mx.array:
        """v14: value-only adapter correction (identity when the adapter is absent)."""
        if self.value_adapter is None:
            return features
        return features + self.value_adapter(features)

    def forward_padded(
        self,
        board: mx.array,       # (B, H, W, C)
        move_rows: mx.array,   # (B, M)
        move_cols: mx.array,   # (B, M)
        move_mask: mx.array,   # (B, M)
        active_size: int = 24,  # Curriculum board size
        return_value_pretanh: bool = False,  # For diagnostics
    ) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
        """Batched forward with padded moves for training/MCTS.

        IMPORTANT: Input is canonicalized so network always sees current player
        connecting top↔bottom, regardless of actual color.

        Args:
            board: (B, H, W, C) board tensors
            move_rows: (B, M) row indices
            move_cols: (B, M) column indices
            move_mask: (B, M) valid move mask
            active_size: Curriculum board size for masked pooling (default 24)
            return_value_pretanh: If True, also return pre-tanh values for diagnostics

        Returns:
            (policy_logits, value, pretanh) - pretanh is None unless return_value_pretanh=True
        """
        # === CANONICALIZE INPUT ===
        board, move_rows, move_cols, move_mask = canonicalize_batch(
            board, move_rows, move_cols, move_mask, active_size
        )

        features = self.encoder(board)  # (B, H, W, hidden)
        policy_logits = self.policy_head.forward_padded(
            features, move_rows, move_cols, move_mask
        )  # (B, M)

        value_feats = self._value_features(features)  # v14: value-only adapter
        if return_value_pretanh:
            value, pretanh = self.value_head(value_feats, active_size, return_pretanh=True)
            return policy_logits, value, pretanh

        value = self.value_head(value_feats, active_size)  # (B,)
        return policy_logits, value, None

    def __call__(
        self,
        board: mx.array,
        moves: List[Tuple[int, int]],
        active_size: int = 24,  # Curriculum board size
    ) -> Tuple[mx.array, mx.array]:
        """Single-position forward for MCTS inference.

        Canonicalizes input identically to batched path via forward_padded().

        Args:
            board: (B, H, W, C) or (H, W, C) board tensor (channels-last)
            moves: List of (row, col) legal moves
            active_size: Curriculum board size for masked pooling (default 24)

        Returns:
            policy: (N,) logits for each move
            value: scalar in [-1, 1]
        """
        # Ensure 4D input (some call sites may pass (H,W,C) instead of (1,H,W,C))
        if board.ndim == 3:
            board = board[None, ...]

        # Layout assertion: MLX uses NHWC (B, H, W, C)
        assert board.ndim == 4, f"Expected 4D tensor, got {board.ndim}D"
        assert board.shape[3] == self.in_channels, (
            f"Expected {self.in_channels} channels in last dim (NHWC), "
            f"got shape {board.shape}. Did you pass NCHW by mistake?"
        )

        N = len(moves)
        if N == 0:
            # Handle empty moves - still need to canonicalize for value head
            board_canon, _, _, _ = canonicalize_batch(
                board,
                mx.zeros((1, 1), dtype=mx.int32),
                mx.zeros((1, 1), dtype=mx.int32),
                mx.zeros((1, 1), dtype=mx.float32),
                active_size,
            )
            features = self.encoder(board_canon)
            value = self.value_head(self._value_features(features), active_size)  # v14
            return mx.array([]), value

        # Build padded format for single position
        move_rows = mx.array([[r for r, c in moves]], dtype=mx.int32)  # (1, N)
        move_cols = mx.array([[c for r, c in moves]], dtype=mx.int32)  # (1, N)
        move_mask = mx.ones((1, N), dtype=mx.float32)

        # Use forward_padded (which canonicalizes)
        policy_logits, value, _ = self.forward_padded(
            board, move_rows, move_cols, move_mask, active_size
        )

        # Remove batch dim from both outputs
        # value is now always (B,) shape from updated ValueHead
        return policy_logits[0], value[0]

    def evaluate(
        self,
        board: mx.array,
        moves: List[Tuple[int, int]],
        active_size: int = 24,  # Curriculum board size
    ) -> Tuple[mx.array, float]:
        """Evaluate position (convenience method).

        Args:
            board: (B, H, W, C) board tensor (channels-last)
            moves: List of (row, col) legal moves
            active_size: Curriculum board size for masked pooling (default 24)

        Returns:
            priors: (N,) softmax probabilities
            value: float in [-1, 1]
        """
        policy_logits, value = self(board, moves, active_size)

        # Convert logits to probabilities
        if policy_logits.size > 0:
            priors = mx.softmax(policy_logits)
        else:
            priors = policy_logits

        return priors, float(value)


def create_network(
    hidden: int = 128,
    n_blocks: int = 6,
    in_channels: Optional[int] = None,
    value_adapter: bool = False,
    value_adapter_bottleneck_width: Optional[int] = None,
) -> AlphaZeroNetwork:
    """Build an AlphaZero network.

    Args:
        hidden: Hidden channels in encoder (default 128)
        n_blocks: Number of residual blocks (default 6)
        in_channels: input channel count. None (default) uses the module's
            current NUM_CHANNELS. Explicit int (e.g. 24) lets callers
            instantiate a network matching a historical checkpoint format.

    Returns:
        AlphaZeroNetwork instance
    """
    if in_channels is None:
        in_channels = NUM_CHANNELS
    return AlphaZeroNetwork(
        in_channels=in_channels,
        hidden=hidden,
        n_blocks=n_blocks,
        value_adapter=value_adapter,
        value_adapter_bottleneck_width=value_adapter_bottleneck_width,
    )


def state_to_input(state: "TwixtState") -> mx.array:
    """Convert TwixtState to network input tensor.

    Args:
        state: TwixtState instance

    Returns:
        (1, H, W, C) MLX array ready for network
    """
    import numpy as np

    # Get numpy tensor (C, H, W)
    tensor = state.to_tensor()

    # Transpose to (H, W, C) and add batch dimension
    tensor = np.transpose(tensor, (1, 2, 0))  # (H, W, C)

    return mx.array(tensor[None, ...])  # (1, H, W, C)


def state_to_input_batch(states: list) -> mx.array:
    """Convert multiple TwixtStates to batched network input.

    Args:
        states: List of TwixtState instances

    Returns:
        (B, H, W, C) MLX array ready for network
    """
    import numpy as np

    boards = []
    for state in states:
        tensor = state.to_tensor()  # (C, H, W) numpy
        tensor = np.transpose(tensor, (1, 2, 0))  # (H, W, C)
        boards.append(tensor)

    boards_np = np.stack(boards, axis=0)  # (B, H, W, C)
    return mx.array(boards_np)
