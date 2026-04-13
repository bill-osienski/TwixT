# AlphaZero Implementation Plan for TwixT

## Overview

Replace hand-tuned heuristics with self-play reinforcement learning using the AlphaZero approach:
- **Training**: MLX on Mac GPU
- **Inference**: ONNX on CPU (Node.js server) with JS heuristics fallback
- **Frontend**: Browser-based with difficulty levels and win prediction bar

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         TRAINING (Mac)                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Self-Play  │───▶│   Replay    │───▶│   MLX Training      │  │
│  │   (MCTS)    │    │   Buffer    │    │   (GPU)             │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────┘
                                │ Export
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      INFERENCE (Server)                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   ONNX      │───▶│    MCTS     │───▶│   Express API       │  │
│  │   Runtime   │    │   Search    │    │   (Node.js)         │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FRONTEND (Browser)                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  AlphaZero  │◀──▶│  Heuristics │    │   Win Prediction    │  │
│  │   Client    │    │   Fallback  │    │   Bar               │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tensor Layout Contract

This section answers the critical layout questions. **Read this before touching any tensor code.**

### Q1: What is the canonical internal layout in MLX?

**Answer: All MLX tensors are NHWC (B, H, W, C) end-to-end.**

- `state_to_input()` outputs `(1, 24, 24, 24)` in NHWC
- Encoder outputs `(B, H, W, hidden)` in NHWC
- PolicyHead gathers with `x[0, row, col, :]`
- ValueHead flattens after 1x1 conv (layout-agnostic when C=1)

### Q2: Where is the single place we convert layouts for ONNX export?

**Answer: `export_onnx.py` converts NHWC → NCHW exactly once.**

- MLX model weights are transposed during export (conv kernels)
- The PyTorch export wrapper expects NCHW input
- Node.js prepares NCHW directly (no runtime transpose in inference)
- **Never convert anywhere else** - double-transpose bugs are silent and deadly

### Q3: What layout does ONNX expect, and what does Node feed?

**Answer: ONNX expects NCHW. Node must feed `[1, C, H, W]`.**

| Component | Layout | Shape | Indexing |
|-----------|--------|-------|----------|
| **MLX (training)** | NHWC | `(1, 24, 24, 24)` | `x[0, r, c, :]` |
| **PyTorch (export)** | NCHW | `(1, 24, 24, 24)` | `x[0, :, r, c]` |
| **ONNX model** | NCHW | `(1, 24, 24, 24)` | Standard for ORT |
| **Node.js input** | NCHW | `(1, 24, 24, 24)` | Must match ONNX |

If Node feeds NHWC to an NCHW model, inference silently produces garbage.

### Q4: Did we fix all gather/indexing sites?

**Answer: Yes. Verified sites:**

| Location | Code | Layout-Sensitive? |
|----------|------|-------------------|
| PolicyHead gather | `x[0, row, col, :]` | Yes - would break with NCHW |
| ValueHead pooling | `mean(x, axis=(1,2))` + `max(x, axis=(1,2))` | **No** - truly layout-agnostic |
| ResBlocks | No manual indexing | Conv handles it |
| BoardEncoder | No manual indexing | Conv handles it |

**ValueHead uses global pooling (mean + max)**: This is truly layout-agnostic because pooling over spatial dimensions produces the same result regardless of memory order. The old flatten-based approach would silently poison training if layout was wrong.

### Q5: Did parity tests fail for the right reason and now pass?

**Answer: Encoding parity (Phase 1) passes. ONNX parity (Phase 6) not yet implemented.**

Current tests:
- `test_layout_sanity`: PolicyHead tested directly with synthetic features, verifies NHWC gather
- `run_encoding_parity.py`: 69 positions match between Python and Node.js tensors
- Network assertion: Catches wrong channel count at forward() entry

Needed in Phase 6:
- MLX forward vs ONNX forward match (< 1e-4 diff) on 10+ boards
- Test with varying move counts to catch padding bugs

---

### Pre-Commit Checklist (Answer Before Each Phase)

#### 1. Where exactly is NHWC→NCHW done for ONNX?

| Step | Location | What Happens |
|------|----------|--------------|
| Weight export | `export_onnx.py` | Conv kernel weights transposed once |
| Runtime input | `server/inference.js` | Node.js converts `toTensor()` HWC → CHW in evaluate() |

**Critical details:**
- **No data-layout transpose inside ONNX graph.** The ONNX model expects NCHW input; Node.js must provide it.
- **`verify_export.py` must transpose the same way Node does.** Otherwise parity test uses different conversion path than runtime.
- **Conv kernel layout**: MLX stores `(O, KH, KW, I)`, PyTorch expects `(O, I, KH, KW)`. If your MLX checkpoint already uses `(O, I, KH, KW)`, skip transpose or you'll double-transpose. Parity test will catch this.

#### 2. Do we have a deterministic gather unit test that fails if indexing is wrong?

**Yes**: `tests/test_network.py::test_layout_sanity`
- Bypasses encoder, tests PolicyHead directly with synthetic features
- Creates `[1,0]` at (5,10) and `[0,1]` at (10,5)
- Asserts different logits when querying each position
- Would fail immediately if gather used `x[0, :, r, c]` instead of `x[0, r, c, :]`

#### 3. Do we have an MLX vs ONNX parity test on multiple boards AND multiple move counts?

**Not yet.** This is Phase 6 (`verify_export.py`). Required tests:
- [ ] 10+ random boards with different move counts (5, 50, 200, 500)
- [ ] Policy logits match < 1e-4
- [ ] Value matches < 1e-4
- [ ] Masked positions have logits ≤ -1e8 (effectively zero probability)
- [ ] **Move-order invariance**: Run ONNX twice with same moves in shuffled order, re-associate logits by (row,col), verify they match. Catches "sorted in one place but not another" bugs.

#### 4. Do we have a masking test for padded moves?

**Not yet.** This is Phase 6. Required tests:
- [ ] Positions 0..N-1 have real logits (finite, reasonable range)
- [ ] Positions N..511 have logits ≤ -1e8 (strict: exactly -1e9, tolerant: ≤ -1e8 for ORT variance)
- [ ] Softmax over valid moves sums to 1.0 (within tolerance)

### Self-Defense Assertions

**In `network.py` (MLX)**:
```python
assert board.shape[3] == NUM_CHANNELS, "Expected NHWC, got wrong channels"
```

**In `export_onnx.py` (Phase 6, to be added)**:
```python
# Before permute
assert mlx_input.shape == (B, H, W, C), "Input must be NHWC"
# After permute
assert torch_input.shape == (B, C, H, W), "Output must be NCHW"
```

### Layout Sanity Test

`tests/test_network.py::test_layout_sanity` is a **tripwire test** that fails 100% of the time if gather indexing is wrong:

1. **Non-square dims**: Uses `(1, 7, 11, 5)` NHWC so shapes differ from NCHW `(1, 5, 7, 11)`
2. **Deterministic weights**: All conv/fc weights set to ones, so output is predictable
3. **Position-sensitive**: Verifies different input positions produce different outputs

This catches both "wrong axis order" and "wrong indexing convention" bugs.

---

## Critical Conventions (Must Follow)

These conventions ensure correctness and parity between Python training and Node.js inference:

1. **MCTS Leaf Rule**: `_expand()` always evaluates the leaf node with NN. Never return `node.qValue` for already-expanded nodes - unexpanded children of expanded nodes need evaluation.

2. **Single NN Eval Per Expansion**: Store both `node.priors` and `node.nnValue` during expansion. Backup uses stored `nnValue`, not a second NN call.

3. **Node.js Rules Parity**: `server/gameLogic.js` must produce identical legal moves and win detection as Python training code. Test exhaustively.

4. **Cache Key Ordering**: Sort moves before hashing to ensure order-independent cache hits: `moves.sort((a,b) => a.row - b.row || a.col - b.col)`.

5. **Terminal Value Convention**: From perspective of player-to-move:
   - `0` if draw
   - `+1.0` if winner == to_move
   - `-1.0` if winner != to_move

   **Draw detection**: `winner()` returns `null/None` for non-terminal positions (game ongoing). A **draw** is detected separately: `isTerminal() == true && winner() == null`. This happens when no legal moves remain but no path is complete. Do NOT confuse "no winner yet" with "draw".

   **TwixT note**: In TwixT, terminal nodes are reached when someone just completed a winning path. The winner is usually the opponent of `to_move` (the player who just moved), so the value for `to_move` is typically `-1`. Draw is rare but possible if the board fills.

6. **Board Encoding**: 24-channel tensor encoding (matching existing engine). Both Python and Node.js must use identical encoding. Channel layout defined in `toTensor()`.

7. **Cache Performance**: Board stored/hashed as `Uint8Array`; use FNV-1a loop hashing (no spread operators); cache is LRU eviction. Optionally support "value-only by board" cache for positions where move-set is implied.

8. **No Spread Operators in Hot Paths**: Use explicit loops for `Math.max`, `Math.min`, softmax, etc. Spread on large arrays can blow the stack.

---

## Board Encoding Specification (24 Channels)

Both Python training and Node.js inference **must use identical encoding**. Mismatch here breaks parity silently.

| Channel | Description |
|---------|-------------|
| 0 | Red pegs (1 where red peg exists, 0 elsewhere) |
| 1 | Black pegs (1 where black peg exists, 0 elsewhere) |
| 2 | Red links NNE direction (+2 row, +1 col) |
| 3 | Red links ENE direction (+1 row, +2 col) |
| 4 | Red links ESE direction (-1 row, +2 col) |
| 5 | Red links SSE direction (-2 row, +1 col) |
| 6 | Red links SSW direction (-2 row, -1 col) |
| 7 | Red links WSW direction (-1 row, -2 col) |
| 8 | Red links WNW direction (+1 row, -2 col) |
| 9 | Red links NNW direction (+2 row, -1 col) |
| 10-17 | Black links (same 8 directions as red) |
| 18 | Current player indicator (1 if red to move, 0 if black) |
| 19 | Red top edge distance (normalized 0-1, closer = higher) |
| 20 | Red bottom edge distance (normalized 0-1) |
| 21 | Black left edge distance (normalized 0-1) |
| 22 | Black right edge distance (normalized 0-1) |
| 23 | Move number / game phase (normalized 0-1, e.g., ply/200) |

**Link encoding**: For each link between pegs at (r1,c1) and (r2,c2), mark a 1 at **both** endpoints in the appropriate direction channel. This makes links visible from either end.

**Edge distance**: For cell (r, c):
- Red top: `1 - r / (size - 1)`
- Red bottom: `r / (size - 1)`
- Black left: `1 - c / (size - 1)`
- Black right: `c / (size - 1)`

**Implementation checklist**:
- [ ] Python `TwixtState.to_tensor()` produces (24, 24, 24) array
- [ ] Node.js `TwixtState.toTensor()` produces identical output
- [ ] Write parity test comparing Python and Node outputs on same position

**BLOCKER**: Training and inference are blocked until `to_tensor()`/`toTensor()` are fully implemented with all 24 channels and parity-tested. Training on a partial or mismatched encoding will produce a useless model.

---

## Known TODOs (Must Complete Before Training)

The following are explicitly marked TODO in this plan and **must be implemented before training begins**:

| Location | TODO | Impact if skipped |
|----------|------|-------------------|
| `server/gameLogic.js` → `toTensor()` | Fill channels 2-23 (links, edges, meta) | Node.js inference will produce garbage |
| `server/gameLogic.js` → `_crossesExistingLink()` | Implement segment intersection | Illegal links allowed, game rules broken |
| `server/gameLogic.js` → `_checkWin()` | Implement path finding (BFS/union-find) | Games never terminate properly |
| Python `TwixtState.to_tensor()` | Implement full 24-channel encoding | Training data will be wrong |
| Python + Node.js | Define draw semantics: draw = terminal with no winner, caused by (a) no legal moves (board fills / all cells blocked), or (b) ply ≥ `MAX_PLIES` (forced draw, even if legal moves exist). Both implementations must use same `MAX_PLIES` constant. | Python/Node disagree on terminal states |
| `export_onnx.py` | **(Optional)** Vectorize policy-head gather: replace `for i in range(512)` loop with batched `index_select`. | ORT latency may be high (~512 ops vs 1) |

**Parity tests will fail** until all of the above are implemented identically in Python and Node.js. This is expected — the plan documents the architecture, not a working implementation.

---

## Phase 1: Network Architecture (Python/MLX)

### File: `scripts/GPU/alphazero/network.py`

```python
"""AlphaZero network with dual policy/value heads."""
import mlx.core as mx
import mlx.nn as nn
from typing import List, Tuple

class BoardEncoder(nn.Module):
    """CNN encoder for board state (24-channel input).

    Note: MLX Conv2d uses channels-last format (B, H, W, C), not channels-first.
    """

    def __init__(self, in_channels: int = 24, hidden: int = 128, n_blocks: int = 6):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(hidden)

        self.blocks = []
        for _ in range(n_blocks):
            self.blocks.append(ResBlock(hidden))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C) - MLX uses channels-last format
        x = nn.relu(self.bn1(self.conv1(x)))
        for block in self.blocks:
            x = block(x)
        return x  # (B, H, W, hidden)


class ResBlock(nn.Module):
    """Residual block with skip connection."""

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


class PolicyHead(nn.Module):
    """Gather-based policy head - outputs one logit per legal move.

    Uses feature gathering to extract (row, col) features, then projects
    to scalar logits. This avoids 24x24 output with masking.

    Note: MLX Conv2d uses channels-last format (B, H, W, C).
    """

    def __init__(self, in_channels: int = 128, hidden: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2, kernel_size=1)
        self.bn = nn.BatchNorm(2)
        self.fc = nn.Linear(2, hidden)
        self.out = nn.Linear(hidden, 1)

    def __call__(self, features: mx.array, moves: List[Tuple[int, int]]) -> mx.array:
        """
        Args:
            features: (B, H, W, C) encoded board features (B=1 for inference)
            moves: List of (row, col) legal moves

        Returns:
            (N,) logits, one per move
        """
        # Reduce channels: (B, H, W, C) -> (B, H, W, 2)
        x = nn.relu(self.bn(self.conv(features)))

        # Gather features at move locations
        # x is (1, H, W, 2), extract features at each (row, col)
        logits = []
        for row, col in moves:
            # Get 2-channel feature at (row, col)
            feat = x[0, row, col, :]  # (2,) - channels-last indexing
            h = nn.relu(self.fc(feat))
            logit = self.out(h)
            logits.append(logit.squeeze())

        return mx.stack(logits)  # (N,)


class ValueHead(nn.Module):
    """Value head - predicts win probability for current player.

    Uses global pooling (mean + max) instead of flatten.
    This is LAYOUT-AGNOSTIC: pooling over spatial dims produces the same
    result regardless of NHWC vs NCHW memory order. Flatten-based
    approaches silently break if layout is wrong.

    Input: (B, H, W, C) features (channels-last, NHWC)
    """

    def __init__(self, in_channels: int = 128, hidden: int = 256):
        super().__init__()
        # 2*in_channels because we concat avg and max pooled features
        self.fc1 = nn.Linear(2 * in_channels, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def __call__(self, features: mx.array) -> mx.array:
        """
        Args:
            features: (B, H, W, C) encoded board - channels-last

        Returns:
            (B,) value in [-1, 1]
        """
        # Global pooling over spatial dimensions (H, W) = axes (1, 2)
        avg_pool = mx.mean(features, axis=(1, 2))  # (B, C)
        max_pool = mx.max(features, axis=(1, 2))   # (B, C)

        # Concatenate pooled features
        x = mx.concatenate([avg_pool, max_pool], axis=-1)  # (B, 2*C)

        # MLP to scalar
        x = nn.relu(self.fc1(x))
        return mx.tanh(self.fc2(x)).squeeze()  # (B,) or scalar


class AlphaZeroNetwork(nn.Module):
    """Combined network with shared encoder (24-channel input)."""

    def __init__(self, in_channels: int = 24, hidden: int = 128, n_blocks: int = 6):
        super().__init__()
        self.encoder = BoardEncoder(in_channels, hidden, n_blocks)
        self.policy_head = PolicyHead(hidden)
        self.value_head = ValueHead(hidden)

    def __call__(
        self,
        board: mx.array,
        moves: List[Tuple[int, int]]
    ) -> Tuple[mx.array, mx.array]:
        """
        Args:
            board: (B, H, W, C) board tensor
            moves: List of (row, col) legal moves

        Returns:
            policy: (N,) logits for each move
            value: scalar in [-1, 1]
        """
        features = self.encoder(board)
        policy = self.policy_head(features, moves)
        value = self.value_head(features)
        return policy, value


def create_network() -> AlphaZeroNetwork:
    """Create default network (24-channel input)."""
    return AlphaZeroNetwork(in_channels=24, hidden=128, n_blocks=6)
```

---

## Phase 2: MCTS Implementation (Python)

### File: `scripts/GPU/alphazero/mcts.py`

```python
"""MCTS with PUCT selection and neural network evaluation."""
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mlx.core as mx

from .network import AlphaZeroNetwork


@dataclass
class MCTSConfig:
    """MCTS hyperparameters."""
    c_puct: float = 1.5              # Exploration constant
    n_simulations: int = 800         # Simulations per move
    dirichlet_alpha: float = 0.3     # Dirichlet noise parameter
    dirichlet_eps: float = 0.25      # Noise mixing weight
    temp_threshold_ply: int = 20     # Plies before temperature drops
    temp_high: float = 1.0           # Early game temperature
    temp_low: float = 0.1            # Late game temperature


@dataclass
class MCTSNode:
    """Node in MCTS tree."""
    state: "GameState"
    parent: Optional["MCTSNode"] = None
    move: Optional[Tuple[int, int]] = None  # Move that led here

    # Statistics
    visit_count: int = 0
    value_sum: float = 0.0

    # NN outputs (set during expansion)
    priors: Optional[Dict[Tuple[int, int], float]] = None  # move -> prior
    nn_value: Optional[float] = None  # Value from NN (stored, not re-evaluated)

    # Children
    children: Dict[Tuple[int, int], "MCTSNode"] = field(default_factory=dict)

    @property
    def q_value(self) -> float:
        """Mean action value."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        """Node has been evaluated by NN."""
        return self.priors is not None


class MCTS:
    """Monte Carlo Tree Search with neural network guidance."""

    def __init__(
        self,
        network: AlphaZeroNetwork,
        config: MCTSConfig = None,
        rng: Optional[random.Random] = None,
    ):
        self.network = network
        self.config = config or MCTSConfig()
        self.rng = rng or random.Random()

    def search(self, root_state: "GameState") -> Tuple[Dict[Tuple[int, int], int], float]:
        """Run MCTS from given state.

        Args:
            root_state: Current game state

        Returns:
            visit_counts: Dict mapping move -> raw visit count
            root_value: Estimated value of position
        """
        root = MCTSNode(state=root_state)

        # Expand root and add Dirichlet noise
        self._expand(root)
        self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(self.config.n_simulations):
            node = root
            search_path = [node]

            # SELECT: traverse tree using PUCT
            while node.is_expanded and not node.state.is_terminal():
                move, node = self._select_child(node)
                search_path.append(node)

            # EXPAND & EVALUATE: single NN call, store both priors and value
            if not node.state.is_terminal():
                value = self._expand(node)
            else:
                # Terminal node: explicit value assignment
                winner = node.state.winner()
                if winner is None:
                    value = 0.0  # Draw
                elif winner == node.state.to_move:
                    value = 1.0  # Current player won
                else:
                    value = -1.0  # Current player lost

            # BACKUP: propagate value up the tree
            self._backup(search_path, value)

        # Return raw visit counts (not normalized)
        visit_counts = {
            move: child.visit_count
            for move, child in root.children.items()
        }

        return visit_counts, root.q_value

    def _expand(self, node: MCTSNode) -> float:
        """Expand node: run NN, store priors and value, create children.

        Returns:
            value: NN value estimate for this position (stored in node.nn_value)
        """
        state = node.state
        moves = state.legal_moves()

        # Prepare input
        board_tensor = mx.array(state.to_tensor())[None, ...]  # Add batch dim

        # Single NN call - get both policy and value
        policy_logits, value = self.network(board_tensor, moves)
        mx.eval(policy_logits, value)

        # Convert logits to priors via softmax
        priors = self._stable_softmax(policy_logits)

        # Store priors as dict
        node.priors = {
            move: priors[i].item()
            for i, move in enumerate(moves)
        }

        # Store NN value (used in backup, avoids second NN call)
        node.nn_value = value.item()

        # Create child nodes (unexpanded)
        for move in moves:
            child_state = state.apply_move(move)
            node.children[move] = MCTSNode(
                state=child_state,
                parent=node,
                move=move,
            )

        return node.nn_value

    def _select_child(self, node: MCTSNode) -> Tuple[Tuple[int, int], MCTSNode]:
        """Select child using PUCT formula.

        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        c = self.config.c_puct
        sqrt_parent = math.sqrt(node.visit_count + 1)

        best_score = float("-inf")
        best_move = None
        best_child = None

        for move, child in node.children.items():
            prior = node.priors[move]

            # Q from child's perspective (negate for opponent)
            q = -child.q_value if child.visit_count > 0 else 0.0

            # PUCT exploration bonus
            u = c * prior * sqrt_parent / (1 + child.visit_count)

            score = q + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def _backup(self, search_path: List[MCTSNode], leaf_value: float):
        """Propagate value up the search path.

        Value alternates sign as we go up (opponent's loss is our gain).
        """
        value = leaf_value
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for parent (opponent's perspective)

    def _add_dirichlet_noise(self, root: MCTSNode):
        """Add Dirichlet noise to root priors for exploration."""
        if not root.priors:
            return

        moves = list(root.priors.keys())
        n = len(moves)

        # Generate Dirichlet noise
        noise = self.rng.gammavariate
        samples = [noise(self.config.dirichlet_alpha, 1.0) for _ in range(n)]
        total = sum(samples)
        noise_probs = [s / total for s in samples]

        # Mix with original priors
        eps = self.config.dirichlet_eps
        for i, move in enumerate(moves):
            root.priors[move] = (1 - eps) * root.priors[move] + eps * noise_probs[i]

    def _stable_softmax(self, logits: mx.array) -> mx.array:
        """Numerically stable softmax."""
        shifted = logits - mx.max(logits)
        exp_shifted = mx.exp(shifted)
        return exp_shifted / mx.sum(exp_shifted)

    def select_move(
        self,
        visit_counts: Dict[Tuple[int, int], int],
        ply: int,
    ) -> Tuple[int, int]:
        """Select move from visit counts using temperature.

        Args:
            visit_counts: Dict mapping move -> visit count
            ply: Current ply number

        Returns:
            Selected move
        """
        # Determine temperature
        if ply < self.config.temp_threshold_ply:
            temp = self.config.temp_high
        else:
            temp = self.config.temp_low

        moves = list(visit_counts.keys())
        counts = [visit_counts[m] for m in moves]

        if temp < 0.01:
            # Deterministic: pick highest visit count with stable tie-break
            max_count = max(counts)
            best_moves = [m for m, c in zip(moves, counts) if c == max_count]
            # Lexicographic tie-break
            best_moves.sort(key=lambda m: (m[0], m[1]))
            return best_moves[0]

        # Softmax over visit counts with temperature
        log_counts = [math.log(c + 1e-8) / temp for c in counts]
        max_log = max(log_counts)
        exp_counts = [math.exp(lc - max_log) for lc in log_counts]
        total = sum(exp_counts)
        probs = [e / total for e in exp_counts]

        # Sample
        r = self.rng.random()
        cumsum = 0.0
        for move, prob in zip(moves, probs):
            cumsum += prob
            if r <= cumsum:
                return move

        return moves[-1]
```

---

## Phase 3: Self-Play & Training

### File: `scripts/GPU/alphazero/self_play.py`

```python
"""Self-play game generation."""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import random

from .mcts import MCTS, MCTSConfig
from .network import AlphaZeroNetwork


@dataclass
class PositionRecord:
    """Single training position from self-play.

    IMPORTANT: to_move is stored explicitly, NOT inferred from move index.
    This ensures correct value targets even with non-standard starting positions.
    """
    board_tensor: List[List[List[float]]]  # (H, W, C)
    to_move: str                            # "red" or "black" - explicit, not inferred
    legal_moves: List[Tuple[int, int]]      # List of (row, col)
    visit_counts: List[int]                 # Raw visit counts (same order as legal_moves)
    outcome: Optional[float] = None         # +1 if to_move won, -1 if lost, 0 draw


@dataclass
class GameRecord:
    """Complete self-play game."""
    positions: List[PositionRecord]
    winner: Optional[str]  # "red", "black", or None for draw
    n_moves: int


def play_game(
    network: AlphaZeroNetwork,
    mcts_config: MCTSConfig = None,
    rng: Optional[random.Random] = None,
    max_moves: int = 200,
) -> GameRecord:
    """Play one self-play game.

    Args:
        network: Neural network for evaluation
        mcts_config: MCTS configuration
        rng: Random number generator
        max_moves: Maximum moves before declaring draw

    Returns:
        GameRecord with all positions and outcome
    """
    from ..game.twixt_state import TwixtState  # Local import to avoid cycles

    mcts = MCTS(network, mcts_config, rng)
    state = TwixtState.initial()
    positions = []

    ply = 0
    while not state.is_terminal() and ply < max_moves:
        # Run MCTS
        visit_counts, _ = mcts.search(state)

        # Record position with explicit to_move
        moves = list(visit_counts.keys())
        counts = [visit_counts[m] for m in moves]

        positions.append(PositionRecord(
            board_tensor=state.to_tensor(),
            to_move=state.to_move,  # Explicit, not inferred from ply
            legal_moves=moves,
            visit_counts=counts,  # Raw counts, not normalized
        ))

        # Select and apply move
        move = mcts.select_move(visit_counts, ply)
        state = state.apply_move(move)
        ply += 1

    # Determine winner
    winner = state.winner() if state.is_terminal() else None

    # Assign outcomes to positions
    for pos in positions:
        if winner is None:
            pos.outcome = 0.0  # Draw
        elif winner == pos.to_move:
            pos.outcome = 1.0  # Player at this position won
        else:
            pos.outcome = -1.0  # Player at this position lost

    return GameRecord(
        positions=positions,
        winner=winner,
        n_moves=ply,
    )
```

### File: `scripts/GPU/alphazero/trainer.py`

```python
"""Training loop for AlphaZero."""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import List
import math

from .network import AlphaZeroNetwork
from .self_play import PositionRecord


def stable_softmax(x: mx.array) -> mx.array:
    """Numerically stable softmax."""
    shifted = x - mx.max(x)
    exp_shifted = mx.exp(shifted)
    return exp_shifted / mx.sum(exp_shifted)


def alphazero_loss(
    network: AlphaZeroNetwork,
    positions: List[PositionRecord],
    l2_weight: float = 1e-4,
) -> mx.array:
    """Combined policy + value + L2 loss.

    Loss = cross_entropy(pi, p) + MSE(z, v) + L2

    Where:
        pi = MCTS visit distribution (normalized)
        p = network policy output
        z = game outcome from perspective of to_move
        v = network value output
    """
    total_policy_loss = mx.array(0.0)
    total_value_loss = mx.array(0.0)

    for pos in positions:
        board = mx.array(pos.board_tensor)[None, ...]  # (1, H, W, C)

        # Forward pass
        policy_logits, value = network(board, pos.legal_moves)

        # Policy target: normalize visit counts to distribution (with epsilon for safety)
        counts = mx.array(pos.visit_counts, dtype=mx.float32)
        target_policy = counts / (mx.sum(counts) + 1e-8)  # Defensive normalization

        # Cross-entropy loss: -sum(target * log(pred))
        pred_log_probs = policy_logits - mx.logsumexp(policy_logits)  # log_softmax
        policy_loss = -mx.sum(target_policy * pred_log_probs)

        # Value loss: MSE
        value_loss = (value - pos.outcome) ** 2

        total_policy_loss = total_policy_loss + policy_loss
        total_value_loss = total_value_loss + value_loss

    n = len(positions)
    avg_policy_loss = total_policy_loss / n
    avg_value_loss = total_value_loss / n

    # L2 regularization on all parameters
    l2_loss = mx.array(0.0)
    for name, param in network.parameters().items():
        l2_loss = l2_loss + mx.sum(param ** 2)
    l2_loss = l2_weight * l2_loss

    return avg_policy_loss + avg_value_loss + l2_loss


def train_step(
    network: AlphaZeroNetwork,
    optimizer: optim.Optimizer,
    batch: List[PositionRecord],
) -> float:
    """Single training step.

    Returns:
        Loss value
    """
    def loss_fn(model):
        return alphazero_loss(model, batch)

    loss, grads = nn.value_and_grad(network, loss_fn)(network)
    optimizer.update(network, grads)
    mx.eval(network.parameters(), optimizer.state, loss)

    return loss.item()


class ReplayBuffer:
    """Fixed-size buffer of training positions with uniform sampling."""

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer: List[PositionRecord] = []
        self.index = 0

    def add_game(self, game: "GameRecord"):
        """Add all positions from a game to the buffer."""
        for pos in game.positions:
            if len(self.buffer) < self.max_size:
                self.buffer.append(pos)
            else:
                # Overwrite oldest
                self.buffer[self.index] = pos
            self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size: int) -> List[PositionRecord]:
        """Sample random batch from buffer."""
        import random
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


def train(
    n_iterations: int = 100,
    games_per_iteration: int = 25,
    train_steps_per_iteration: int = 100,
    batch_size: int = 64,
    buffer_size: int = 100000,
    checkpoint_dir: str = "checkpoints/alphazero",
    mcts_simulations: int = 800,
    learning_rate: float = 1e-3,
    resume_from: Optional[str] = None,
):
    """Full AlphaZero training loop.

    Each iteration:
    1. Self-play: generate games with current network
    2. Add positions to replay buffer
    3. Train on random batches from buffer
    4. Checkpoint

    Args:
        n_iterations: Total training iterations
        games_per_iteration: Self-play games per iteration
        train_steps_per_iteration: Gradient updates per iteration
        batch_size: Positions per training step
        buffer_size: Max replay buffer capacity
        checkpoint_dir: Where to save checkpoints
        mcts_simulations: MCTS simulations per move
        learning_rate: Optimizer learning rate
        resume_from: Path to checkpoint to resume from
    """
    import os
    import json
    import random
    from pathlib import Path

    from .network import create_network
    from .self_play import play_game
    from .mcts import MCTSConfig

    # Setup
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    network = create_network()
    optimizer = optim.Adam(learning_rate=learning_rate)
    buffer = ReplayBuffer(max_size=buffer_size)
    mcts_config = MCTSConfig(n_simulations=mcts_simulations)

    start_iteration = 0

    # Resume from checkpoint if specified
    if resume_from:
        network.load_weights(resume_from)
        # Load buffer and iteration from state file if exists
        state_path = Path(resume_from).with_suffix('.json')
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
                start_iteration = state.get('iteration', 0)
        print(f"Resumed from {resume_from}, iteration {start_iteration}")

    print(f"Starting training: {n_iterations} iterations")
    print(f"  Games/iter: {games_per_iteration}")
    print(f"  Train steps/iter: {train_steps_per_iteration}")
    print(f"  Buffer size: {buffer_size}")

    for iteration in range(start_iteration, n_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{n_iterations}")
        print(f"{'='*60}")

        # 1. Self-play
        print(f"\nSelf-play: generating {games_per_iteration} games...")
        rng = random.Random(iteration * 1000)
        games_generated = 0
        positions_added = 0

        for g in range(games_per_iteration):
            game = play_game(network, mcts_config, rng)
            buffer.add_game(game)
            games_generated += 1
            positions_added += len(game.positions)

            if (g + 1) % 5 == 0:
                print(f"  Games: {g+1}/{games_per_iteration}, "
                      f"Buffer: {len(buffer)} positions")

        print(f"  Generated {games_generated} games, {positions_added} positions")
        print(f"  Buffer size: {len(buffer)}")

        # 2. Training
        if len(buffer) >= batch_size:
            print(f"\nTraining: {train_steps_per_iteration} steps...")
            total_loss = 0.0

            for step in range(train_steps_per_iteration):
                batch = buffer.sample(batch_size)
                loss = train_step(network, optimizer, batch)
                total_loss += loss

                if (step + 1) % 20 == 0:
                    avg_loss = total_loss / (step + 1)
                    print(f"  Step {step+1}/{train_steps_per_iteration}, "
                          f"Loss: {avg_loss:.4f}")

            print(f"  Average loss: {total_loss / train_steps_per_iteration:.4f}")
        else:
            print(f"\nSkipping training (buffer has {len(buffer)} < {batch_size})")

        # 3. Checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"model_iter_{iteration+1:04d}.safetensors")
        network.save_weights(ckpt_path)

        state = {
            'iteration': iteration + 1,
            'buffer_size': len(buffer),
            'games_total': (iteration + 1) * games_per_iteration,
        }
        with open(ckpt_path.replace('.safetensors', '.json'), 'w') as f:
            json.dump(state, f, indent=2)

        print(f"\nCheckpoint saved: {ckpt_path}")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")

    return network
```

---

## Phase 4: ONNX Export

### File: `scripts/GPU/alphazero/export_onnx.py`

```python
"""Export MLX model to ONNX for Node.js inference.

Strategy: Pad moves to 512 with -1e9 masking for invalid positions.
"""
import torch
import torch.nn as tnn
import numpy as np
from typing import Dict, List, Tuple

# MLX -> PyTorch parameter name mapping
PARAM_MAP = {
    # Encoder
    "encoder.conv1.weight": "encoder.conv1.weight",
    "encoder.conv1.bias": "encoder.conv1.bias",
    "encoder.bn1.weight": "encoder.bn1.weight",
    "encoder.bn1.bias": "encoder.bn1.bias",
    "encoder.bn1.running_mean": "encoder.bn1.running_mean",
    "encoder.bn1.running_var": "encoder.bn1.running_var",
    # ResBlocks: encoder.blocks.{i}.conv{1,2}, bn{1,2}
    # PolicyHead
    "policy_head.conv.weight": "policy_head.conv.weight",
    "policy_head.conv.bias": "policy_head.conv.bias",
    "policy_head.bn.weight": "policy_head.bn.weight",
    "policy_head.bn.bias": "policy_head.bn.bias",
    "policy_head.fc.weight": "policy_head.fc.weight",
    "policy_head.fc.bias": "policy_head.fc.bias",
    "policy_head.out.weight": "policy_head.out.weight",
    "policy_head.out.bias": "policy_head.out.bias",
    # ValueHead (global pooling - no conv/bn layers)
    "value_head.fc1.weight": "value_head.fc1.weight",
    "value_head.fc1.bias": "value_head.fc1.bias",
    "value_head.fc2.weight": "value_head.fc2.weight",
    "value_head.fc2.bias": "value_head.fc2.bias",
}


class OnnxAlphaZero(tnn.Module):
    """PyTorch model matching MLX architecture for ONNX export.

    Uses pad-to-512 strategy with masking for variable move counts.
    24-channel input to match training encoding.
    """

    def __init__(self, hidden: int = 128, n_blocks: int = 6, max_moves: int = 512):
        super().__init__()
        self.max_moves = max_moves

        # Encoder (24-channel input)
        self.encoder_conv1 = tnn.Conv2d(24, hidden, 3, padding=1)
        self.encoder_bn1 = tnn.BatchNorm2d(hidden)

        self.res_blocks = tnn.ModuleList()
        for _ in range(n_blocks):
            self.res_blocks.append(tnn.Sequential(
                tnn.Conv2d(hidden, hidden, 3, padding=1),
                tnn.BatchNorm2d(hidden),
                tnn.ReLU(),
                tnn.Conv2d(hidden, hidden, 3, padding=1),
                tnn.BatchNorm2d(hidden),
            ))

        # Policy head
        self.policy_conv = tnn.Conv2d(hidden, 2, 1)
        self.policy_bn = tnn.BatchNorm2d(2)
        self.policy_fc = tnn.Linear(2, 64)
        self.policy_out = tnn.Linear(64, 1)

        # Value head (global pooling - layout agnostic)
        # 2*hidden because we concat avg and max pooled features
        self.value_fc1 = tnn.Linear(2 * hidden, 256)
        self.value_fc2 = tnn.Linear(256, 1)

    def forward(
        self,
        board: torch.Tensor,      # (1, C, H, W) = (1, 24, 24, 24) NCHW format!
        move_rows: torch.Tensor,  # (512,) padded row indices
        move_cols: torch.Tensor,  # (512,) padded col indices
        move_mask: torch.Tensor,  # (512,) 1.0 for valid, 0.0 for padding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        IMPORTANT: PyTorch uses NCHW layout. Node.js must feed (1, 24, 24, 24)
        where dim order is (batch, channels, height, width).

        Returns:
            policy_logits: (512,) with -1e9 for invalid moves
            value: scalar in [-1, 1]
        """
        # Encode - expects NCHW input
        x = torch.relu(self.encoder_bn1(self.encoder_conv1(board)))
        for block in self.res_blocks:
            x = torch.relu(block(x) + x)

        # Policy: gather features at move locations
        policy_feat = torch.relu(self.policy_bn(self.policy_conv(x)))  # (1, 2, 24, 24)

        # Gather: extract (2,) feature vector at each (row, col)
        # policy_feat is (1, 2, H, W)
        #
        # WARNING: v1 export uses a Python for-loop over 512 moves. This works
        # but generates a chunky ONNX graph (512 separate gather ops) which can
        # be slow in ONNX Runtime. For production, prefer vectorized gather:
        #   gathered = policy_feat[0, :, move_rows, move_cols]  # (2, 512)
        #   gathered = gathered.T  # (512, 2)
        #   ... then batch FC layers
        # This produces a single gather op and batched matmuls.
        raw_logits = []
        for i in range(self.max_moves):
            r, c = move_rows[i], move_cols[i]
            feat = policy_feat[0, :, r, c]  # (2,)
            h = torch.relu(self.policy_fc(feat))
            logit = self.policy_out(h)
            raw_logits.append(logit)

        raw_logits = torch.stack(raw_logits).squeeze(-1)  # (512,)

        # Apply mask: -1e9 for invalid positions
        masked_logits = torch.where(
            move_mask > 0.5,
            raw_logits,
            torch.full_like(raw_logits, -1e9)
        )

        # Value head (global pooling - layout agnostic)
        # x is (1, C, H, W) from encoder
        # Pool over spatial dims (2, 3) in NCHW format
        avg_pool = torch.mean(x, dim=(2, 3))  # (B, C)
        max_pool = torch.amax(x, dim=(2, 3))  # (B, C)
        v = torch.cat([avg_pool, max_pool], dim=-1)  # (B, 2*C)

        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return masked_logits, value.squeeze()


def convert_weights(mlx_params: Dict, pytorch_model: OnnxAlphaZero) -> None:
    """Copy weights from MLX to PyTorch model.

    Handles:
    - Conv weight layout: MLX (out, in, H, W) -> PyTorch (out, in, H, W) (same)
    - BatchNorm running stats from MLX
    """
    state_dict = pytorch_model.state_dict()

    for mlx_name, torch_name in PARAM_MAP.items():
        if mlx_name in mlx_params:
            param = np.array(mlx_params[mlx_name])
            state_dict[torch_name] = torch.from_numpy(param)

    # Handle ResBlock parameters dynamically
    for i in range(len(pytorch_model.res_blocks)):
        mlx_prefix = f"encoder.blocks.{i}"
        torch_prefix = f"res_blocks.{i}"

        for j in [1, 2]:  # conv1, conv2
            for suffix in ["weight", "bias"]:
                mlx_key = f"{mlx_prefix}.conv{j}.{suffix}"
                torch_key = f"{torch_prefix}.{(j-1)*3}.{suffix}"
                if mlx_key in mlx_params:
                    state_dict[torch_key] = torch.from_numpy(np.array(mlx_params[mlx_key]))

            # BatchNorm
            bn_idx = (j - 1) * 3 + 1
            for suffix in ["weight", "bias", "running_mean", "running_var"]:
                mlx_key = f"{mlx_prefix}.bn{j}.{suffix}"
                torch_key = f"{torch_prefix}.{bn_idx}.{suffix}"
                if mlx_key in mlx_params:
                    state_dict[torch_key] = torch.from_numpy(np.array(mlx_params[mlx_key]))

    pytorch_model.load_state_dict(state_dict)


def export_to_onnx(mlx_model, output_path: str):
    """Export MLX model to ONNX.

    Args:
        mlx_model: Trained MLX AlphaZeroNetwork
        output_path: Path for .onnx file
    """
    # Create PyTorch model
    pytorch_model = OnnxAlphaZero()
    pytorch_model.eval()

    # Copy weights
    mlx_params = dict(mlx_model.parameters())
    convert_weights(mlx_params, pytorch_model)

    # Create dummy inputs
    board = torch.randn(1, 24, 24, 24)  # 24 channels
    move_rows = torch.zeros(512, dtype=torch.long)
    move_cols = torch.zeros(512, dtype=torch.long)
    move_mask = torch.zeros(512)
    move_mask[:10] = 1.0  # Pretend 10 valid moves

    # Export
    torch.onnx.export(
        pytorch_model,
        (board, move_rows, move_cols, move_mask),
        output_path,
        input_names=["board", "move_rows", "move_cols", "move_mask"],
        output_names=["policy_logits", "value"],
        dynamic_axes=None,  # Fixed sizes for simplicity
        opset_version=13,
    )

    print(f"Exported ONNX model to {output_path}")
```

### File: `scripts/GPU/alphazero/verify_export.py`

```python
"""Verify ONNX export matches MLX model outputs.

Run on multiple boards with different move counts to catch edge cases.
"""
import numpy as np
import onnxruntime as ort

def verify_forward_parity(mlx_model, onnx_path: str, test_boards: list):
    """Compare MLX and ONNX outputs on test positions.

    Args:
        mlx_model: MLX AlphaZeroNetwork
        onnx_path: Path to exported ONNX model
        test_boards: List of (board_tensor, legal_moves) tuples

    Raises:
        AssertionError if outputs differ by more than tolerance
    """
    import mlx.core as mx

    session = ort.InferenceSession(onnx_path)

    for i, (board_np, moves) in enumerate(test_boards):
        # MLX forward
        board_mlx = mx.array(board_np)[None, ...]
        policy_mlx, value_mlx = mlx_model(board_mlx, moves)
        mx.eval(policy_mlx, value_mlx)

        # ONNX forward (prepare padded inputs)
        board_onnx = np.transpose(board_np, (2, 0, 1))[None, ...].astype(np.float32)

        move_rows = np.zeros(512, dtype=np.int64)
        move_cols = np.zeros(512, dtype=np.int64)
        move_mask = np.zeros(512, dtype=np.float32)

        for j, (r, c) in enumerate(moves):
            move_rows[j] = r
            move_cols[j] = c
            move_mask[j] = 1.0

        policy_onnx, value_onnx = session.run(
            None,
            {
                "board": board_onnx,
                "move_rows": move_rows,
                "move_cols": move_cols,
                "move_mask": move_mask,
            }
        )

        # Compare (only valid moves for policy)
        policy_mlx_np = np.array(policy_mlx)
        policy_onnx_valid = policy_onnx[:len(moves)]

        policy_diff = np.max(np.abs(policy_mlx_np - policy_onnx_valid))
        value_diff = abs(float(value_mlx) - float(value_onnx))

        print(f"Board {i}: policy_diff={policy_diff:.6f}, value_diff={value_diff:.6f}")

        assert policy_diff < 1e-4, f"Policy mismatch on board {i}: {policy_diff}"
        assert value_diff < 1e-4, f"Value mismatch on board {i}: {value_diff}"

    print(f"All {len(test_boards)} boards passed parity check!")
```

---

## Phase 5: Node.js Server

### File: `server/gameLogic.js`

```javascript
/**
 * TwixT game rules for Node.js server.
 *
 * CRITICAL: This must produce identical legal moves and win detection
 * as the Python training code. Test exhaustively with comparison scripts.
 */

class TwixtState {
  constructor(board = null, toMove = 'red', links = null) {
    this.size = 24;
    this.board = board || this._emptyBoard();
    this.toMove = toMove;
    this.links = links || { red: [], black: [] };
  }

  _emptyBoard() {
    return Array(this.size).fill(null).map(() => Array(this.size).fill(null));
  }

  legalMoves() {
    const moves = [];
    for (let r = 0; r < this.size; r++) {
      for (let c = 0; c < this.size; c++) {
        if (this._isLegalMove(r, c)) {
          moves.push({ row: r, col: c });
        }
      }
    }
    return moves;
  }

  _isLegalMove(r, c) {
    // Empty cell
    if (this.board[r][c] !== null) return false;

    // Corner exclusion
    if ((r === 0 || r === this.size - 1) && (c === 0 || c === this.size - 1)) {
      return false;
    }

    // Row exclusion for colors
    if (this.toMove === 'red') {
      if (r === 0 || r === this.size - 1) return false;
    } else {
      if (c === 0 || c === this.size - 1) return false;
    }

    return true;
  }

  applyMove(move) {
    const newBoard = this.board.map(row => [...row]);
    newBoard[move.row][move.col] = this.toMove;

    const newLinks = {
      red: [...this.links.red],
      black: [...this.links.black],
    };

    // Add new links (knight-move connections)
    const newLinksList = this._findNewLinks(move.row, move.col, this.toMove, newBoard);
    newLinks[this.toMove] = [...newLinks[this.toMove], ...newLinksList];

    return new TwixtState(
      newBoard,
      this.toMove === 'red' ? 'black' : 'red',
      newLinks
    );
  }

  _findNewLinks(r, c, color, board) {
    // Knight move offsets
    const offsets = [
      [-2, -1], [-2, 1], [-1, -2], [-1, 2],
      [1, -2], [1, 2], [2, -1], [2, 1]
    ];

    const links = [];
    for (const [dr, dc] of offsets) {
      const nr = r + dr;
      const nc = c + dc;

      if (nr >= 0 && nr < this.size && nc >= 0 && nc < this.size) {
        if (board[nr][nc] === color) {
          // Check for crossing links (simplified - full impl needed)
          const link = [[r, c], [nr, nc]];
          if (!this._crossesExistingLink(link, color)) {
            links.push(link);
          }
        }
      }
    }
    return links;
  }

  _crossesExistingLink(newLink, color) {
    // Full link crossing detection needed
    // Placeholder - implement segment intersection
    return false;
  }

  isTerminal() {
    return this.winner() !== null || this.legalMoves().length === 0;
  }

  winner() {
    // Check if red connects top to bottom (columns 0 to size-1)
    // Check if black connects left to right (rows 0 to size-1)
    // Use union-find or BFS on links

    if (this._checkWin('red')) return 'red';
    if (this._checkWin('black')) return 'black';
    return null;
  }

  _checkWin(color) {
    // Placeholder - implement proper path finding
    // Red wins: path from row 0 to row size-1
    // Black wins: path from col 0 to col size-1
    return false;
  }

  toTensor() {
    // 24-channel tensor encoding - MUST match Python training exactly.
    // Channel layout (example - adapt to match existing engine):
    //   0: red pegs
    //   1: black pegs
    //   2-9: red link directions (8 knight-move directions)
    //   10-17: black link directions
    //   18-23: additional features (to_move, edge distances, etc.)
    //
    // TODO: Import actual encoding from existing engine's toTensor()
    // to ensure parity with Python training code.
    const channels = 24;
    const tensor = [];
    for (let r = 0; r < this.size; r++) {
      const row = [];
      for (let c = 0; c < this.size; c++) {
        const cell = new Array(channels).fill(0);
        cell[0] = this.board[r][c] === 'red' ? 1 : 0;
        cell[1] = this.board[r][c] === 'black' ? 1 : 0;
        // TODO: Fill remaining channels to match training encoding
        row.push(cell);
      }
      tensor.push(row);
    }
    return tensor;
  }
}

module.exports = { TwixtState };
```

### File: `server/cache.js`

```javascript
/**
 * Position cache with board+moves hash key.
 *
 * Uses:
 * - Uint8Array for board representation (fast)
 * - FNV-1a hashing (fast, good distribution)
 * - LRU eviction (Map iteration order)
 * - Sorted moves for order-independent cache hits
 *
 * Optionally supports "value-only by board" caching when move-set is implied.
 */

// FNV-1a constants (32-bit)
const FNV_OFFSET = 2166136261;
const FNV_PRIME = 16777619;

class BoardMovesCache {
  constructor(maxSize = 10000) {
    this.cache = new Map();
    this.maxSize = maxSize;
  }

  /**
   * Convert board to Uint8Array for fast hashing.
   * 0 = empty, 1 = red, 2 = black
   */
  _boardToUint8(board) {
    const size = board.length;
    const arr = new Uint8Array(size * size);
    let idx = 0;
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        const val = board[r][c];
        arr[idx++] = val === 'red' ? 1 : val === 'black' ? 2 : 0;
      }
    }
    return arr;
  }

  /**
   * FNV-1a hash for Uint8Array (fast loop, no spread).
   */
  _fnv1a(data) {
    let hash = FNV_OFFSET;
    for (let i = 0; i < data.length; i++) {
      hash ^= data[i];
      hash = Math.imul(hash, FNV_PRIME);
    }
    return hash >>> 0;  // Convert to unsigned
  }

  _hashBoard(board) {
    const arr = this._boardToUint8(board);
    return this._fnv1a(arr);
  }

  _hashMoves(moves) {
    // CRITICAL: Sort moves for order-independent hashing
    const sorted = [...moves].sort((a, b) => a.row - b.row || a.col - b.col);

    // Pack moves into Uint8Array (2 bytes per move: row, col)
    const arr = new Uint8Array(sorted.length * 2);
    for (let i = 0; i < sorted.length; i++) {
      arr[i * 2] = sorted[i].row;
      arr[i * 2 + 1] = sorted[i].col;
    }
    return this._fnv1a(arr);
  }

  makeKey(board, moves) {
    const boardHash = this._hashBoard(board);
    const movesHash = this._hashMoves(moves);
    return `${boardHash}:${movesHash}`;
  }

  /**
   * Key for value-only caching (when move-set is implied by board).
   */
  makeBoardOnlyKey(board) {
    return `v:${this._hashBoard(board)}`;
  }

  get(board, moves) {
    const key = this.makeKey(board, moves);
    const value = this.cache.get(key);

    // LRU: move to end on access
    if (value !== undefined) {
      this.cache.delete(key);
      this.cache.set(key, value);
    }
    return value;
  }

  set(board, moves, value) {
    const key = this.makeKey(board, moves);

    // If key exists, delete first (for LRU ordering)
    if (this.cache.has(key)) {
      this.cache.delete(key);
    }

    // LRU eviction: remove oldest (first) entry
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }

    this.cache.set(key, value);
  }

  clear() {
    this.cache.clear();
  }
}

module.exports = { BoardMovesCache };
```

### File: `server/inference.js`

```javascript
/**
 * ONNX model wrapper for inference.
 */
const ort = require('onnxruntime-node');

class AlphaZeroInference {
  constructor(modelPath) {
    this.modelPath = modelPath;
    this.session = null;
    this.maxMoves = 512;
  }

  async load() {
    this.session = await ort.InferenceSession.create(this.modelPath);
  }

  async evaluate(boardTensor, moves) {
    /**
     * Evaluate position with neural network.
     *
     * Args:
     *   boardTensor: (24, 24, 24) board representation (24 channels)
     *   moves: Array of {row, col} legal moves
     *
     * Returns:
     *   { priors: Map<string, number>, value: number }
     *   priors maps "row,col" -> probability (after softmax)
     *   value in [-1, 1]
     */

    const numChannels = 24;

    // CRITICAL: Convert from toTensor() NHWC output to ONNX NCHW input
    // boardTensor is [H][W][C] from toTensor()
    // ONNX expects [C][H][W] flattened as (1, C, H, W)
    // This is the ONLY place Node.js does layout conversion
    const board = new Float32Array(1 * numChannels * 24 * 24);
    for (let c = 0; c < numChannels; c++) {
      for (let r = 0; r < 24; r++) {
        for (let col = 0; col < 24; col++) {
          board[c * 24 * 24 + r * 24 + col] = boardTensor[r][col][c];
        }
      }
    }

    // Prepare move arrays (padded to 512)
    const moveRows = new BigInt64Array(this.maxMoves);
    const moveCols = new BigInt64Array(this.maxMoves);
    const moveMask = new Float32Array(this.maxMoves);

    for (let i = 0; i < moves.length && i < this.maxMoves; i++) {
      moveRows[i] = BigInt(moves[i].row);
      moveCols[i] = BigInt(moves[i].col);
      moveMask[i] = 1.0;
    }

    // Run inference
    const feeds = {
      board: new ort.Tensor('float32', board, [1, 24, 24, 24]),
      move_rows: new ort.Tensor('int64', moveRows, [this.maxMoves]),
      move_cols: new ort.Tensor('int64', moveCols, [this.maxMoves]),
      move_mask: new ort.Tensor('float32', moveMask, [this.maxMoves]),
    };

    const results = await this.session.run(feeds);

    // Extract logits (only valid moves)
    const logits = results.policy_logits.data.slice(0, moves.length);
    const value = results.value.data[0];

    // Return raw logits (caller can softmax if needed)
    // MCTS uses logits directly for priors
    const priors = new Map();

    // Softmax for priors (used in MCTS) - loop max, no spread operator
    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxLogit) maxLogit = logits[i];
    }

    let sumExp = 0;
    const exps = [];
    for (let i = 0; i < logits.length; i++) {
      const exp = Math.exp(logits[i] - maxLogit);
      exps.push(exp);
      sumExp += exp;
    }

    for (let i = 0; i < moves.length; i++) {
      const key = `${moves[i].row},${moves[i].col}`;
      priors.set(key, exps[i] / sumExp);
    }

    return { priors, value };
  }
}

module.exports = { AlphaZeroInference };
```

### File: `server/mcts.js`

```javascript
/**
 * MCTS for Node.js server inference.
 *
 * Key differences from Python training MCTS:
 * - Uses ONNX inference instead of MLX
 * - Simpler (no Dirichlet noise for inference)
 * - Fixed simulation count based on difficulty
 */

class MCTSNode {
  constructor(state, parent = null, move = null) {
    this.state = state;
    this.parent = parent;
    this.move = move;

    this.visitCount = 0;
    this.valueSum = 0;

    this.priors = null;      // Map<"row,col", prior>
    this.nnValue = null;     // Stored NN value (single eval per expansion)
    this.children = new Map(); // Map<"row,col", MCTSNode>
  }

  get qValue() {
    return this.visitCount === 0 ? 0 : this.valueSum / this.visitCount;
  }

  get isExpanded() {
    return this.priors !== null;
  }
}


class MCTS {
  constructor(inference, config = {}) {
    this.inference = inference;
    this.cPuct = config.cPuct || 1.5;
    this.nSimulations = config.nSimulations || 200;
  }

  async search(rootState) {
    /**
     * Run MCTS from given state.
     *
     * Returns:
     *   { visitCounts: Map<"row,col", count>, rootValue: number }
     */
    const root = new MCTSNode(rootState);

    // Expand root
    await this._expand(root);

    // Run simulations
    for (let i = 0; i < this.nSimulations; i++) {
      let node = root;
      const searchPath = [node];

      // SELECT: traverse using PUCT
      while (node.isExpanded && !node.state.isTerminal()) {
        const [move, child] = this._selectChild(node);
        node = child;
        searchPath.push(node);
      }

      // EXPAND & EVALUATE
      let value;
      if (node.state.isTerminal()) {
        // Terminal: explicit value
        const winner = node.state.winner();
        if (winner === null) {
          value = 0;
        } else if (winner === node.state.toMove) {
          value = 1;
        } else {
          value = -1;
        }
      } else {
        // Non-terminal leaf: expand and get NN value
        value = await this._expand(node);
      }

      // BACKUP
      this._backup(searchPath, value);
    }

    // Collect visit counts
    const visitCounts = new Map();
    for (const [moveKey, child] of root.children) {
      visitCounts.set(moveKey, child.visitCount);
    }

    return { visitCounts, rootValue: root.qValue };
  }

  async _expand(node) {
    /**
     * Expand node: single NN eval, store priors and value.
     * Returns: NN value for backup
     */
    const moves = node.state.legalMoves();
    const boardTensor = node.state.toTensor();

    // Single NN call
    const { priors, value } = await this.inference.evaluate(boardTensor, moves);

    // Store on node (avoids second NN call in backup)
    node.priors = priors;
    node.nnValue = value;

    // Create children (unexpanded)
    for (const move of moves) {
      const key = `${move.row},${move.col}`;
      const childState = node.state.applyMove(move);
      node.children.set(key, new MCTSNode(childState, node, move));
    }

    return value;
  }

  _selectChild(node) {
    /**
     * Select child using PUCT: Q + c * P * sqrt(N) / (1 + N_child)
     */
    const sqrtParent = Math.sqrt(node.visitCount + 1);

    let bestScore = -Infinity;
    let bestMove = null;
    let bestChild = null;

    for (const [moveKey, child] of node.children) {
      const prior = node.priors.get(moveKey) || 0;

      // Q from child perspective (negate)
      const q = child.visitCount > 0 ? -child.qValue : 0;

      // PUCT bonus
      const u = this.cPuct * prior * sqrtParent / (1 + child.visitCount);

      const score = q + u;
      if (score > bestScore) {
        bestScore = score;
        bestMove = moveKey;
        bestChild = child;
      }
    }

    return [bestMove, bestChild];
  }

  _backup(searchPath, leafValue) {
    /**
     * Propagate value up, alternating sign.
     */
    let value = leafValue;
    for (let i = searchPath.length - 1; i >= 0; i--) {
      const node = searchPath[i];
      node.visitCount += 1;
      node.valueSum += value;
      value = -value;
    }
  }

  selectMove(visitCounts, temperature = 0.1) {
    /**
     * Select move from visit counts.
     *
     * temperature=0: deterministic (highest count)
     * temperature>0: sample proportional to count^(1/temp)
     */
    const moves = Array.from(visitCounts.keys());
    const counts = moves.map(m => visitCounts.get(m));

    if (temperature < 0.01) {
      // Deterministic with lexicographic tie-break
      let maxCount = -1;
      let bestMove = null;

      for (let i = 0; i < moves.length; i++) {
        if (counts[i] > maxCount) {
          maxCount = counts[i];
          bestMove = moves[i];
        } else if (counts[i] === maxCount) {
          // Lexicographic tie-break
          if (moves[i] < bestMove) {
            bestMove = moves[i];
          }
        }
      }
      return bestMove;
    }

    // Temperature sampling - loop max, no spread operator
    const logCounts = counts.map(c => Math.log(c + 1e-8) / temperature);
    let maxLog = -Infinity;
    for (let i = 0; i < logCounts.length; i++) {
      if (logCounts[i] > maxLog) maxLog = logCounts[i];
    }

    let sumExp = 0;
    const exps = [];
    for (let i = 0; i < logCounts.length; i++) {
      const exp = Math.exp(logCounts[i] - maxLog);
      exps.push(exp);
      sumExp += exp;
    }

    let r = Math.random();
    for (let i = 0; i < moves.length; i++) {
      r -= exps[i] / sumExp;
      if (r <= 0) return moves[i];
    }

    return moves[moves.length - 1];
  }
}

module.exports = { MCTS, MCTSNode };
```

### File: `server/index.js`

```javascript
/**
 * Express server for AlphaZero inference API.
 */
const express = require('express');
const cors = require('cors');
const { AlphaZeroInference } = require('./inference');
const { MCTS } = require('./mcts');
const { TwixtState } = require('./gameLogic');
const { BoardMovesCache } = require('./cache');

const app = express();
app.use(cors());
app.use(express.json());

// Global instances
let inference = null;
let cache = null;

// Difficulty -> simulations mapping
const DIFFICULTY_SIMS = {
  easy: 50,
  medium: 200,
  hard: 800,
};

app.post('/api/move', async (req, res) => {
  try {
    const { board, toMove, links, difficulty = 'medium' } = req.body;

    // Reconstruct state
    const state = new TwixtState(board, toMove, links);
    const moves = state.legalMoves();

    if (moves.length === 0) {
      return res.json({ error: 'no_legal_moves' });
    }

    // Check cache
    const cached = cache.get(board, moves);
    if (cached) {
      return res.json(cached);
    }

    // Run MCTS
    const nSims = DIFFICULTY_SIMS[difficulty] || 200;
    const mcts = new MCTS(inference, { nSimulations: nSims });

    const { visitCounts, rootValue } = await mcts.search(state);

    // Select move
    const temperature = difficulty === 'easy' ? 0.5 : 0.1;
    const moveKey = mcts.selectMove(visitCounts, temperature);
    const [row, col] = moveKey.split(',').map(Number);

    const result = {
      move: { row, col },
      value: rootValue,
      visits: Object.fromEntries(visitCounts),
    };

    // Cache result
    cache.set(board, moves, result);

    res.json(result);
  } catch (err) {
    console.error('Error:', err);
    res.status(500).json({ error: err.message });
  }
});

/**
 * Evaluate position without selecting a move.
 * Used for win prediction bar updates.
 */
app.post('/api/evaluate', async (req, res) => {
  try {
    const { board, toMove, links } = req.body;

    const state = new TwixtState(board, toMove, links);
    const moves = state.legalMoves();

    if (moves.length === 0) {
      // Terminal position
      const winner = state.winner();
      const value = winner === toMove ? 1 : winner ? -1 : 0;
      return res.json({ value, terminal: true });
    }

    // Single NN evaluation (no MCTS for speed)
    const boardTensor = state.toTensor();
    const { value } = await inference.evaluate(boardTensor, moves);

    res.json({ value, terminal: false });
  } catch (err) {
    console.error('Evaluate error:', err);
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', modelLoaded: inference !== null });
});

async function main() {
  const modelPath = process.env.MODEL_PATH || './model.onnx';
  const port = process.env.PORT || 3001;

  console.log('Loading ONNX model...');
  inference = new AlphaZeroInference(modelPath);
  await inference.load();

  cache = new BoardMovesCache(10000);

  app.listen(port, () => {
    console.log(`AlphaZero server running on port ${port}`);
  });
}

main().catch(console.error);
```

---

## Phase 6: Frontend Integration

### File: `assets/js/ai/alphaZeroClient.js`

```javascript
/**
 * AlphaZero client with server fallback to heuristics.
 */

export class AlphaZeroClient {
  constructor(serverUrl = 'http://localhost:3001') {
    this.serverUrl = serverUrl;
    this.timeout = 2000;  // 2 second timeout, then fallback
    this.available = null; // null = unknown, true/false = checked
  }

  async checkAvailability() {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 1000);

      const response = await fetch(`${this.serverUrl}/api/health`, {
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      const data = await response.json();
      this.available = data.status === 'ok' && data.modelLoaded;
      return this.available;
    } catch {
      this.available = false;
      return false;
    }
  }

  async getMove(gameState, difficulty = 'medium') {
    /**
     * Get move from AlphaZero server, with timeout fallback to heuristics.
     *
     * Returns: { move: {row, col}, value: number, source: 'alphazero'|'heuristics' }
     */

    // Check availability on first call
    if (this.available === null) {
      await this.checkAvailability();
    }

    if (!this.available) {
      return this._fallbackToHeuristics(gameState, difficulty);
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await fetch(`${this.serverUrl}/api/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          board: gameState.board,
          toMove: gameState.toMove,
          links: gameState.links,
          difficulty,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      return {
        move: data.move,
        value: data.value,
        source: 'alphazero',
      };
    } catch (err) {
      console.warn('AlphaZero server unavailable, falling back to heuristics:', err.message);
      this.available = false;  // Disable for future calls
      return this._fallbackToHeuristics(gameState, difficulty);
    }
  }

  _fallbackToHeuristics(gameState, difficulty) {
    // Import existing heuristics AI
    // This would call the existing search.js getBestMove
    const { TwixtAI } = window.TwixtAI || {};

    if (!TwixtAI) {
      throw new Error('Heuristics fallback not available');
    }

    const ai = new TwixtAI(gameState);
    const move = ai.getBestMove(difficulty);

    return {
      move,
      value: null,  // Heuristics don't provide value
      source: 'heuristics',
    };
  }

  /**
   * Evaluate position for win bar (no move selection).
   * Returns value in [-1, 1] where +1 = current player winning.
   */
  async evaluate(gameState) {
    if (!this.available) {
      return null;  // Can't evaluate without server
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 500);  // Fast timeout for UI

      const response = await fetch(`${this.serverUrl}/api/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          board: gameState.board,
          toMove: gameState.toMove,
          links: gameState.links,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) return null;

      const data = await response.json();
      return data.value;
    } catch {
      return null;
    }
  }
}

// Singleton instance
export const alphaZero = new AlphaZeroClient();
```

### File: `assets/js/ui/winBar.js`

```javascript
/**
 * Win prediction bar component.
 *
 * Displays neural network's evaluation as a visual bar:
 * - Red side = NN thinks red is winning
 * - Black side = NN thinks black is winning
 * - Center = even position
 */

export class WinBar {
  constructor(containerId = 'win-bar') {
    this.container = document.getElementById(containerId);
    this.value = 0;  // -1 to +1, from red's perspective
    this.enabled = false;

    if (this.container) {
      this._createElements();
    }
  }

  _createElements() {
    this.container.innerHTML = `
      <div class="win-bar-wrapper">
        <span class="win-bar-label red-label">Red</span>
        <div class="win-bar-track">
          <div class="win-bar-fill red-fill"></div>
          <div class="win-bar-fill black-fill"></div>
          <div class="win-bar-center"></div>
        </div>
        <span class="win-bar-label black-label">Black</span>
      </div>
      <div class="win-bar-percentage"></div>
    `;

    this.redFill = this.container.querySelector('.red-fill');
    this.blackFill = this.container.querySelector('.black-fill');
    this.percentageEl = this.container.querySelector('.win-bar-percentage');
  }

  /**
   * Update the bar with a new evaluation.
   * @param {number} value - NN value from red's perspective (-1 to +1)
   * @param {string} toMove - Current player ('red' or 'black')
   */
  update(value, toMove) {
    if (!this.container || value === null) return;

    // Convert to red's perspective if needed
    const redValue = toMove === 'red' ? value : -value;
    this.value = redValue;

    // Calculate fill percentages (value -1 to +1 -> 0% to 100%)
    const redPercent = Math.max(0, redValue) * 50;  // 0-50%
    const blackPercent = Math.max(0, -redValue) * 50;  // 0-50%

    this.redFill.style.width = `${50 + redPercent}%`;
    this.blackFill.style.width = `${50 + blackPercent}%`;

    // Show win percentage
    const winPercent = ((redValue + 1) / 2 * 100).toFixed(0);
    this.percentageEl.textContent = `Red: ${winPercent}%`;
  }

  /**
   * Enable/disable the win bar.
   */
  setEnabled(enabled) {
    this.enabled = enabled;
    if (this.container) {
      this.container.style.display = enabled ? 'block' : 'none';
    }
  }

  /**
   * Clear the bar (unknown position).
   */
  clear() {
    this.value = 0;
    if (this.redFill) this.redFill.style.width = '50%';
    if (this.blackFill) this.blackFill.style.width = '50%';
    if (this.percentageEl) this.percentageEl.textContent = '';
  }
}

// CSS (add to your stylesheet)
/*
.win-bar-wrapper {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 10px 0;
}

.win-bar-track {
  flex: 1;
  height: 20px;
  background: #333;
  border-radius: 4px;
  position: relative;
  overflow: hidden;
}

.win-bar-fill {
  position: absolute;
  top: 0;
  height: 100%;
  transition: width 0.3s ease;
}

.red-fill {
  left: 0;
  background: linear-gradient(to right, #c44, #e55);
}

.black-fill {
  right: 0;
  background: linear-gradient(to left, #444, #666);
}

.win-bar-center {
  position: absolute;
  left: 50%;
  top: 0;
  width: 2px;
  height: 100%;
  background: #fff;
  transform: translateX(-50%);
}

.win-bar-label {
  font-size: 12px;
  font-weight: bold;
  width: 40px;
}

.red-label { color: #e55; }
.black-label { color: #888; }

.win-bar-percentage {
  text-align: center;
  font-size: 11px;
  color: #999;
  margin-top: 4px;
}
*/
```

### Integration Example

```javascript
// In your game controller:
import { alphaZero } from './ai/alphaZeroClient.js';
import { WinBar } from './ui/winBar.js';

const winBar = new WinBar('win-bar');

// After each move, update the win bar
async function onPositionChanged(gameState) {
  // Only update if AlphaZero is available
  if (alphaZero.available) {
    winBar.setEnabled(true);
    const value = await alphaZero.evaluate(gameState);
    if (value !== null) {
      winBar.update(value, gameState.toMove);
    }
  } else {
    winBar.setEnabled(false);
  }
}

// On game start
winBar.clear();

// On AI move received (already has value from /api/move)
function onAIMoveReceived(result, gameState) {
  if (result.value !== null) {
    winBar.update(result.value, gameState.toMove);
  }
}
```

---

## Acceptance Criteria Checklist

### Training (Python)
- [ ] Network produces policy logits and value for any legal position
- [ ] MCTS runs correct number of simulations with PUCT selection
- [ ] Single NN eval per expansion (priors + value stored on node)
- [ ] Self-play generates games with explicit `to_move` in PositionRecord
- [ ] Replay buffer accumulates positions across iterations
- [ ] Training converges (loss decreases over epochs)
- [ ] Checkpoints save/load correctly (model + state)

### Encoding Parity
- [ ] Python `to_tensor()` returns (24, 24, 24) array matching spec
- [ ] Node.js `toTensor()` returns identical output for same position
- [ ] All 24 channels implemented identically (pegs, links, edges, meta)
- [ ] Write automated parity test comparing outputs

### Export
- [ ] ONNX export runs without errors
- [ ] Verify script confirms forward parity (<1e-4 difference)
- [ ] Multi-board verification (different move counts)
- [ ] 24-channel input shape matches training

### Server (Node.js)
- [ ] MCTS applies moves and evaluates children correctly
- [ ] Cache hits on repeated positions (order-independent, LRU)
- [ ] `/api/move` returns move + value within timeout
- [ ] `/api/evaluate` returns value for win bar (fast, no MCTS)
- [ ] `/api/health` reports model status

### Frontend
- [ ] Client uses server when available
- [ ] Timeout triggers heuristics fallback
- [ ] Win prediction bar displays evaluation
- [ ] Win bar updates after each move
- [ ] Bar disabled gracefully when server unavailable

### Parity
- [ ] Python and Node.js produce same move sequence on deterministic test positions
- [ ] Node.js `gameLogic.js` matches Python legal moves exactly
- [ ] Node.js `toTensor()` matches Python `to_tensor()` exactly

---

## File Summary

| File | Purpose |
|------|---------|
| `scripts/GPU/alphazero/network.py` | MLX neural network (24-channel input) |
| `scripts/GPU/alphazero/mcts.py` | Python MCTS for training |
| `scripts/GPU/alphazero/self_play.py` | Game generation |
| `scripts/GPU/alphazero/trainer.py` | Training loop + replay buffer + orchestrator |
| `scripts/GPU/alphazero/export_onnx.py` | MLX → ONNX conversion |
| `scripts/GPU/alphazero/verify_export.py` | Export parity testing |
| `server/gameLogic.js` | TwixT rules for Node (must match Python) |
| `server/cache.js` | Position cache (Uint8Array, FNV-1a, LRU) |
| `server/inference.js` | ONNX wrapper |
| `server/mcts.js` | Node.js MCTS |
| `server/index.js` | Express API (/move, /evaluate, /health) |
| `assets/js/ai/alphaZeroClient.js` | Frontend client with evaluate() |
| `assets/js/ui/winBar.js` | Win prediction bar component |
