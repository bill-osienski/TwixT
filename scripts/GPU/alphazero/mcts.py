"""MCTS with PUCT selection and neural network evaluation.

This implements AlphaZero-style Monte Carlo Tree Search:
- Neural network guides search via policy priors and value estimates
- PUCT formula balances exploration and exploitation
- Dirichlet noise at root for exploration during training
- Temperature-based move selection

Critical Conventions (MUST follow for correctness):
1. Leaf eval rule: _expand() always calls NN for new nodes
2. Single NN eval: Store both priors and nn_value during expansion
3. Terminal values from perspective of to_move:
   - +1.0 if winner == to_move (current player won)
   - -1.0 if winner != to_move (current player lost)
   - 0.0 if draw
4. Backup sign flip: Value alternates sign going up the tree
5. PUCT uses sqrt(N+1) for numerical stability when N=0

CPU-SAFE: No MLX imports in this file. Safe for worker processes.
All GPU operations are delegated to the Evaluator (see local_evaluator.py).
DO NOT add `import mlx` here - it will break multi-process workers.
"""
from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .game import TwixtState
from .evaluator import Evaluator
# Canonical penalty selector — single source of truth shared with the
# diagnostic builder so stored `effective_near_corner_penalty` always matches
# what MCTS actually applied.
from .opening_diagnostics import effective_near_corner_penalty

_OPENDBG = os.environ.get("TWIXT_OPENING_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")

# Guard against accidental MLX import (breaks multi-process workers)
# Only warn in worker processes (main process imports MLX intentionally)
# Set TWIXT_WARN_MLX_IMPORT_ORDER=1 to force-enable for debugging
if "mlx" in sys.modules:
    import multiprocessing
    force_warn = os.getenv("TWIXT_WARN_MLX_IMPORT_ORDER", "0") == "1"
    is_worker = multiprocessing.current_process().name != "MainProcess"
    if force_warn or is_worker:
        import warnings
        warnings.warn(
            "MLX was imported before mcts.py - this may cause issues in worker processes. "
            "Ensure evaluator handles all GPU operations.",
            RuntimeWarning,
            stacklevel=2,
        )


# =============================================================================
# Move encoding helpers (Stage 4.1 optimization)
# =============================================================================
# Use constant board width (24) for encoding - avoids collisions if active_size changes.
# Valid even when active_size < 24 because legal_moves() won't emit out-of-range moves.
# Sanity-checked via active_size bounds in visit_counts decode.
BOARD_W = 24


def encode_move(r: int, c: int) -> int:
    """Encode (row, col) as single int for faster dict ops."""
    return r * BOARD_W + c


def decode_move(move_id: int) -> tuple[int, int]:
    """Decode move_id back to (row, col)."""
    return divmod(move_id, BOARD_W)


def _is_edge_band(r: int, c: int, S: int, B: int) -> bool:
    """Return True if (r, c) is in the edge band of an SxS board with width B."""
    return r < B or r >= S - B or c < B or c >= S - B


def _is_near_corner_cheb(r: int, c: int, S: int, R: int) -> bool:
    """True if (r,c) is within Chebyshev distance <= R of any corner on an SxS board."""
    if R <= 0:
        return False
    corners = ((0, 0), (0, S - 1), (S - 1, 0), (S - 1, S - 1))
    for rr, cc in corners:
        if max(abs(r - rr), abs(c - cc)) <= R:
            return True
    return False


@dataclass
class MCTSConfig:
    """MCTS hyperparameters."""

    c_puct: float = 1.5  # Exploration constant in PUCT
    n_simulations: int = 800  # Simulations per move
    dirichlet_alpha: float = 0.3  # Dirichlet noise parameter
    dirichlet_eps: float = 0.25  # Noise mixing weight (0 = no noise)
    temp_threshold_ply: int = 20  # Plies before temperature drops
    temp_high: float = 1.0  # Early game temperature
    temp_low: float = 0.1  # Late game temperature
    eval_batch_size: int = 14  # Leaves per NN batch (reduced from 16 to prevent Metal GPU hangs)
    pending_virtual_visits: int = 8  # Virtual visits added to pending leaves
    stall_flush_sims: int = 16  # Flush if no NEW pending leaf in N sims (0 = disabled)
    # Opening exploration boost (opening_noise_ply=0 disables)
    opening_noise_ply: int = 0
    opening_dirichlet_alpha: float = 1.0
    opening_dirichlet_eps: float = 0.5
    # Edge-band prior penalty (root_edge_band_penalty=0 disables)
    root_edge_band_penalty: float = 0.0      # λ in exp(-λ) multiplier
    root_edge_band_penalty_ply: int = 0      # apply for ply < this
    root_edge_band_width: int = 2            # band width B (r < B or r >= S-B, etc.)
    # Near-corner prior penalty (root_near_corner_penalty=0 disables)
    root_near_corner_penalty: float = 0.0     # λc in exp(-λc)
    root_near_corner_penalty_ply: int = 0     # apply for ply < this
    root_near_corner_radius: int = 2          # Chebyshev radius R
    # Early-only near-corner override (Phase 2):
    # When active (both values > 0), replaces `root_near_corner_penalty` for
    # plies < `root_near_corner_penalty_early_plies`. The baseline penalty
    # then continues to apply for plies in [early_plies, penalty_ply).
    # Use case: root-search diagnostics showed q overrides the root prior at
    # ply 0-1 while the broader penalty window is still useful for later plies.
    root_near_corner_penalty_early: float = 0.0
    root_near_corner_penalty_early_plies: int = 0
    # Spec 3 Fix 1 — td=1 root visit forcing
    closeout_td1_visit_forcing_enabled: bool = False
    closeout_td1_min_visits: int = 8
    closeout_td1_max_forced_moves: int = 4
    closeout_td1_require_high_value: bool = False
    closeout_td1_high_value_threshold: float = 0.95

    def __post_init__(self):
        if self.eval_batch_size < 1:
            raise ValueError("eval_batch_size must be >= 1")
        if self.pending_virtual_visits < 0:
            raise ValueError("pending_virtual_visits must be >= 0")
        if self.stall_flush_sims < 0:
            raise ValueError("stall_flush_sims must be >= 0")


@dataclass
class MCTSNode:
    """Node in MCTS tree.

    Attributes:
        state: Game state at this node
        parent: Parent node (None for root)
        move: Encoded move_id (int) that led to this node from parent
        visit_count: Number of times node was visited
        value_sum: Sum of backed-up values
        priors: Dict mapping move_id (int) -> prior probability (may have noise applied)
        priors_raw: Original priors before noise (for tree reuse)
        nn_value: Value estimate from NN (stored during expansion)
        children: Dict mapping move_id (int) -> child node
    """

    state: TwixtState
    parent: Optional[MCTSNode] = None
    move: Optional[int] = None  # Encoded move_id, not (row, col) tuple

    # Statistics
    visit_count: int = 0
    value_sum: float = 0.0

    # NN outputs (set during expansion)
    # Keys are encoded move_ids (int), not (row, col) tuples
    priors: Optional[Dict[int, float]] = None
    priors_raw: Optional[Dict[int, float]] = None  # Original priors
    nn_value: Optional[float] = None

    # Children - keys are encoded move_ids (int)
    children: Dict[int, MCTSNode] = field(default_factory=dict)

    @property
    def q_value(self) -> float:
        """Mean action value Q(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        """True if node has been evaluated by NN."""
        return self.priors is not None


class MCTS:
    """Monte Carlo Tree Search with neural network guidance.

    Usage:
        mcts = MCTS(evaluator, config)
        visit_counts, root_value = mcts.search(state)
        move = mcts.select_move(visit_counts, state.ply)
    """

    def __init__(
        self,
        evaluator: Evaluator,
        config: Optional[MCTSConfig] = None,
        rng: Optional[random.Random] = None,
    ):
        """Initialize MCTS.

        Args:
            evaluator: Evaluator for leaf node evaluation (implements Evaluator protocol)
            config: MCTS hyperparameters (uses defaults if None)
            rng: Random number generator (uses new Random if None)
        """
        self.evaluator = evaluator
        self.config = config or MCTSConfig()
        self.rng = rng or random.Random()

        # For testing: track NN calls
        self._nn_call_count = 0
        # For resource management: track expand calls for cache clearing cadence
        self._expand_calls = 0
        # For diagnostics: track actual NN batch invocations (vs logical leaf evals)
        self._nn_batches = 0

        # Waiter list diagnostics
        self._total_backups = 0  # Total backups performed (must be ~games*plies*sims)
        self._total_waiters_backed_up = 0  # For avg_waiters calculation
        self._unique_leaves_expanded = 0  # Unique leaves evaluated
        self._max_waiters_on_any_leaf = 0  # Max waiters on single leaf (dogpile detector)
        self._flush_full = 0  # Batch-full flushes (healthy)
        self._flush_stall = 0  # Stall flushes (tree narrowed)
        self._flush_tail = 0  # Tail flushes (end of sims)

        # Final-root snapshot for per-game stats persistence (spec 2026-04-29).
        # Updated at the end of every successful root search; trainer reads
        # these after the game's last move.
        self._final_root_value: Optional[float] = None
        self._final_top1_share: Optional[float] = None

    def _capture_final_root_stats(self, root: MCTSNode) -> None:
        """Snapshot root.q_value and top child visit share after a search.

        Pure observation — does not mutate the tree, RNG, or counters.
        Sets self._final_root_value and self._final_top1_share for the trainer
        to read after the game's last move. Both values are coerced to Python
        float so JSON serialization downstream is straightforward.
        """
        value = getattr(root, "q_value", None)
        self._final_root_value = float(value) if value is not None else None
        children = list(getattr(root, "children", {}).values())
        if not children:
            self._final_top1_share = None
            return
        total_visits = sum(getattr(c, "visit_count", 0) for c in children)
        if total_visits <= 0:
            self._final_top1_share = None
            return
        top_visits = max(getattr(c, "visit_count", 0) for c in children)
        self._final_top1_share = float(top_visits / total_visits)

    def search(
        self,
        root_state: TwixtState,
        add_noise: bool = True,
    ) -> Tuple[Dict[Tuple[int, int], int], float]:
        """Run MCTS from given state.

        Args:
            root_state: Current game state
            add_noise: Whether to add Dirichlet noise at root (for training)

        Returns:
            visit_counts: Dict mapping (row, col) tuple -> visit count (decoded for callers)
            root_value: Estimated value of position for current player
        """
        root = MCTSNode(state=root_state)

        # Expand root node
        self._expand(root)

        # Add Dirichlet noise for exploration (during training)
        # Note: For fresh root, priors = priors_raw (same object), so reset is implicit
        if add_noise:
            self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(self.config.n_simulations):
            node = root
            search_path = [node]

            # SELECT with LAZY CHILD CREATION
            while node.is_expanded and not node.state.is_terminal():
                move_id, child = self._select_child(node)

                # Instantiate child if missing (lazy creation)
                if child is None:
                    # Decode move_id to (row, col) for apply_move
                    r, c = decode_move(move_id)
                    child = MCTSNode(
                        state=node.state.apply_move((r, c)),
                        parent=node,
                        move=move_id,
                    )
                    node.children[move_id] = child

                search_path.append(child)
                node = child

            # EXPAND & EVALUATE
            if not node.state.is_terminal():
                # Non-terminal leaf: expand with NN
                value = self._expand(node)
            else:
                # Terminal node: explicit value assignment
                value = self._terminal_value(node.state)

            # BACKUP: propagate value up the tree
            self._backup(search_path, value)

        # Build visit_counts from ALL legal moves (not root.priors)
        visit_counts: Dict[Tuple[int, int], int] = {}
        for (r, c) in root.state.legal_moves():
            move_id = encode_move(r, c)
            child = root.children.get(move_id)
            visit_counts[(r, c)] = child.visit_count if child else 0

        # Debug sanity check (catches encoding bugs)
        if __debug__:
            active = root.state.active_size
            for (r, c) in visit_counts.keys():
                assert 0 <= r < active and 0 <= c < active, f"Bad move {(r,c)} for active_size={active}"

        # Snapshot final-root stats for per-game persistence (spec 2026-04-29).
        self._capture_final_root_stats(root)

        return visit_counts, root.q_value

    def search_from_root(
        self,
        root: MCTSNode,
        add_noise: bool = True,
        ply: int = 0,
    ) -> Tuple[Dict[Tuple[int, int], int], float, MCTSNode]:
        """Run MCTS with batched leaf evaluation using waiter lists.

        Key design: Multiple sims can wait on the same pending leaf. When the
        leaf is expanded, ALL waiters are backed up with the returned value.
        This enables effective batching without flushing on duplicates.

        Args:
            root: Existing root node to search from
            add_noise: Whether to add Dirichlet noise at root (for training)

        Returns:
            visit_counts: Dict mapping (row, col) tuple -> visit count (decoded for callers)
            root_value: Q-value estimate at root
            root: The root node (for caller to keep reference)
        """
        # Expand root if not already expanded (single expand, not batched)
        if not root.is_expanded:
            self._expand(root)

        # Add noise - _add_dirichlet_noise copies from priors_raw internally
        if add_noise:
            self._add_dirichlet_noise(root, ply)

        # Define batch_size once at top
        batch_size = self.config.eval_batch_size

        # Option B: Store (node_id, node) pairs to avoid recomputing id() at flush
        # Note: pending_node_ids contains id(node), NOT move_ids - different int domains!
        pending_nodes: List[Tuple[int, MCTSNode]] = []  # (node_id, node), unique leaves
        pending_waiters: Dict[int, List[List[MCTSNode]]] = {}  # node_id -> list of paths
        pending_node_ids: Set[int] = set()  # For virtual visit penalty in _select_child
        stall_count: int = 0  # Sims since we added a NEW pending leaf

        for sim in range(self.config.n_simulations):
            node = root
            search_path = [node]

            # SELECT with LAZY CHILD CREATION
            while node.is_expanded and not node.state.is_terminal():
                move_id, child = self._select_child(node, pending_node_ids)

                # Instantiate child if missing (critical for tree reuse!)
                if child is None:
                    # Decode move_id to (row, col) for apply_move
                    r, c = decode_move(move_id)
                    child = MCTSNode(
                        state=node.state.apply_move((r, c)),
                        parent=node,
                        move=move_id,
                    )
                    node.children[move_id] = child

                search_path.append(child)
                node = child

            # QUEUE or IMMEDIATE BACKUP
            if node.state.is_terminal():
                # Terminal: backup immediately
                value = self._terminal_value(node.state)
                self._backup(search_path, value)
                stall_count = 0  # Reset to prevent stall flush during terminal-heavy search
            else:
                # Non-terminal leaf: queue for batch expansion
                assert not node.is_expanded  # Guard: catch accidental double-expands
                node_id = id(node)

                # Add this sim's path as a "waiter" on that leaf
                if node_id not in pending_waiters:
                    # First time this leaf is pending: create waiters list and queue once
                    pending_waiters[node_id] = [search_path]
                    pending_nodes.append((node_id, node))  # Option B: store (id, node)
                    pending_node_ids.add(node_id)
                    stall_count = 0  # Reset: we added a new pending leaf
                else:
                    # Duplicate pending leaf: add to waiters (NEVER flush here!)
                    pending_waiters[node_id].append(search_path)
                    stall_count += 1  # No new pending leaf this sim

                # Flush when batch is full
                if len(pending_nodes) >= batch_size:
                    self._flush_pending_batch(pending_nodes, pending_waiters)
                    self._flush_full += 1
                    pending_nodes.clear()
                    pending_waiters.clear()
                    pending_node_ids.clear()
                    stall_count = 0

                # Stall flush: tree narrowed, not finding new leaves
                # Note: stall_flush_sims == 0 means disabled
                elif (
                    self.config.stall_flush_sims > 0
                    and stall_count >= self.config.stall_flush_sims
                    and pending_nodes
                ):
                    self._flush_pending_batch(pending_nodes, pending_waiters)
                    self._flush_stall += 1
                    pending_nodes.clear()
                    pending_waiters.clear()
                    pending_node_ids.clear()
                    stall_count = 0

        # Flush remaining (tail flush)
        if pending_nodes:
            self._flush_pending_batch(pending_nodes, pending_waiters)
            self._flush_tail += 1
            pending_nodes.clear()
            pending_waiters.clear()
            pending_node_ids.clear()

        # Build visit_counts from ALL legal moves (not root.priors)
        visit_counts: Dict[Tuple[int, int], int] = {}
        for (r, c) in root.state.legal_moves():
            move_id = encode_move(r, c)
            child = root.children.get(move_id)
            visit_counts[(r, c)] = child.visit_count if child else 0

        # Debug sanity check (catches encoding bugs)
        if __debug__:
            active = root.state.active_size
            for (r, c) in visit_counts.keys():
                assert 0 <= r < active and 0 <= c < active, f"Bad move {(r,c)} for active_size={active}"

        # Snapshot final-root stats for per-game persistence (spec 2026-04-29).
        self._capture_final_root_stats(root)

        return visit_counts, root.q_value, root

    def advance_root(self, root: MCTSNode, move: Tuple[int, int]) -> MCTSNode:
        """Advance root to child after move is played.

        Returns the child node. If child doesn't exist, creates it.
        Detaches from parent to allow GC of old tree.

        Args:
            root: Current root node
            move: Move that was played as (row, col) tuple

        Returns:
            New root node (the child corresponding to the move)
        """
        # Encode the move to match internal int keys
        move_id = encode_move(move[0], move[1])

        if move_id in root.children:
            new_root = root.children[move_id]
        else:
            # Move wasn't explored - create fresh node
            new_root = MCTSNode(state=root.state.apply_move(move), move=move_id)

        # Detach from parent to allow GC of old tree
        new_root.parent = None
        return new_root

    def _expand(self, node: MCTSNode) -> float:
        """Expand node: evaluate via evaluator, store priors and value.

        Delegates to _expand_batch for actual evaluation.
        DOES NOT create child nodes - those are created lazily in the search loop.

        Args:
            node: Node to expand

        Returns:
            value: Value estimate for this position
        """
        # Delegate to batch path (handles all evaluator interaction)
        values = self._expand_batch([node])
        return values[0]

    def _expand_batch(self, nodes: List[MCTSNode]) -> List[float]:
        """Expand multiple leaf nodes in one evaluator batch.

        Args:
            nodes: List of unexpanded leaf nodes

        Returns:
            List of values for each leaf (for caller to backup)
        """
        if not nodes:
            return []

        # Assert all nodes are unexpanded (invariant check)
        for n in nodes:
            if n.is_expanded:
                raise ValueError("_expand_batch received already-expanded node")

        B = len(nodes)
        self._nn_batches += 1
        self._nn_call_count += B  # Logical leaf evals
        self._expand_calls += B

        # Collect states and moves for each leaf
        states = [n.state for n in nodes]
        # Get (row, col) tuples from legal_moves()
        moves_lists_rc = [s.legal_moves() for s in states]

        # Encode moves as int keys (using constant BOARD_W)
        moves_lists_id = [
            [encode_move(r, c) for (r, c) in moves_rc]
            for moves_rc in moves_lists_rc
        ]

        # Handle edge case: any node with no legal moves
        for i, (node, moves_id) in enumerate(zip(nodes, moves_lists_id)):
            if not moves_id:
                node.priors = {}
                node.priors_raw = {}
                node.nn_value = 0.0

        # Filter to nodes with moves (keep RC and ID lists aligned!)
        valid_indices = [i for i, m in enumerate(moves_lists_rc) if m]
        if not valid_indices:
            return [0.0] * B

        valid_nodes = [nodes[i] for i in valid_indices]
        valid_states = [states[i] for i in valid_indices]
        valid_moves_rc = [moves_lists_rc[i] for i in valid_indices]
        valid_moves_id = [moves_lists_id[i] for i in valid_indices]

        # Assert homogeneous active_size (curriculum invariant)
        active_sizes = {s.active_size for s in valid_states}
        if len(active_sizes) != 1:
            raise ValueError(f"Mixed active_size in batch: {active_sizes}")
        active_size = valid_states[0].active_size

        # Build batched numpy arrays for evaluator
        B_valid = len(valid_nodes)

        # Convert states to board tensors (B', H, W, C) - inline numpy conversion
        boards_list = []
        for state in valid_states:
            tensor = self.evaluator.build_input_tensor(state)  # (C, H, W) numpy
            tensor = np.transpose(tensor, (1, 2, 0))  # (H, W, C)
            boards_list.append(tensor)
        boards_np = np.stack(boards_list, axis=0).astype(np.float32)  # (B', H, W, C)

        # Pad moves to max length (use RC for tensor building)
        max_M = max(len(m) for m in valid_moves_rc)

        move_rows_np = np.zeros((B_valid, max_M), dtype=np.int32)
        move_cols_np = np.zeros((B_valid, max_M), dtype=np.int32)
        move_mask_np = np.zeros((B_valid, max_M), dtype=np.float32)

        for b, moves_rc in enumerate(valid_moves_rc):
            for j, (r, c) in enumerate(moves_rc):
                move_rows_np[b, j] = r
                move_cols_np[b, j] = c
                move_mask_np[b, j] = 1.0

        # Call evaluator (returns numpy arrays)
        priors_np, values_np = self.evaluator.infer(
            boards_np, move_rows_np, move_cols_np, move_mask_np, active_size
        )

        # Assign to nodes using int keys (move_id)
        for b, (node, moves_id) in enumerate(zip(valid_nodes, valid_moves_id)):
            # Shuffle (move_id, prior) pairs to randomize dict insertion order
            # This prevents row-major bias when iterating priors
            # Uses self.rng for reproducibility per seed
            # NOTE: Shuffle pairs, NOT moves_id, to keep priors aligned with move_ids
            pairs = [(moves_id[j], float(priors_np[b, j])) for j in range(len(moves_id))]
            self.rng.shuffle(pairs)
            raw_priors = {mid: p for (mid, p) in pairs}
            node.priors_raw = raw_priors
            node.priors = raw_priors  # Same object - NO copy here!
            node.nn_value = float(values_np[b])

            # Debug invariant check (catches key type bugs)
            if __debug__ and node.priors:
                k = next(iter(node.priors.keys()))
                assert isinstance(k, int), f"Expected int move_id keys, got {type(k)}"

        # Build return values (including 0.0 for invalid nodes)
        result = [0.0] * B
        for i, idx in enumerate(valid_indices):
            result[idx] = nodes[idx].nn_value

        return result

    def _flush_pending_batch(
        self,
        pending_nodes: List[Tuple[int, MCTSNode]],
        pending_waiters: Dict[int, List[List[MCTSNode]]],
    ) -> None:
        """Expand each unique pending leaf once, then backup all waiter paths.

        Args:
            pending_nodes: List of (node_id, node) pairs (unique leaves in order)
            pending_waiters: Dict mapping node_id -> list of search paths waiting
        """
        # Sanity check: all pending leaves must still be unexpanded
        for node_id, leaf in pending_nodes:
            if leaf.is_expanded:
                raise ValueError("Pending leaf is already expanded before flush")

        # Extract nodes in order for the actual batched NN call
        leaves = [leaf for (_, leaf) in pending_nodes]

        values = self._expand_batch(leaves)

        # Diagnostics: track waiters
        max_waiters = 0

        # Backup all waiters per leaf, using the stored node_id (Option B)
        for (node_id, _leaf), value in zip(pending_nodes, values):
            waiters = pending_waiters.get(node_id, [])
            w = len(waiters)
            max_waiters = max(max_waiters, w)
            self._total_waiters_backed_up += w
            for path in waiters:
                self._backup(path, value)

        # Diagnostics
        self._unique_leaves_expanded += len(pending_nodes)
        self._max_waiters_on_any_leaf = max(self._max_waiters_on_any_leaf, max_waiters)

    def _select_child(
        self, node: MCTSNode, pending_node_ids: Optional[Set[int]] = None
    ) -> Tuple[int, Optional[MCTSNode]]:
        """Select best move using PUCT formula.

        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s) + 1) / (1 + N(s,a))

        Does NOT create children - returns (move_id, existing_child_or_None).
        Caller is responsible for instantiating missing children.

        Note: Using sqrt(N+1) instead of sqrt(N) for numerical stability
        when parent visit count is 0.

        Args:
            node: Parent node to select from
            pending_node_ids: Set of node ids (id(node)) currently pending evaluation
                              (for virtual visit penalty). NOT move_ids!

        Returns:
            (move_id, child_or_None): Selected move_id (int) and child if exists
        """
        # Early asserts for debugging: catch invariant violations closer to source
        assert node.is_expanded, "_select_child called on unexpanded node"
        assert node.priors, "Expanded node has empty priors"

        c = self.config.c_puct
        sqrt_parent = math.sqrt(node.visit_count + 1)

        best_score = float("-inf")
        best_moves = []  # List of (move_id, child) tuples
        eps = 1e-8  # Tie tolerance (slightly larger to catch float jitter)

        for move_id, prior in node.priors.items():
            child = node.children.get(move_id)

            # Q-value from child's perspective (negate because opponent)
            if child is not None and child.visit_count > 0:
                q = -child.q_value
                child_visits = child.visit_count
            else:
                q = 0.0
                child_visits = 0

            # Virtual visit penalty for pending leaves
            is_pending = (
                pending_node_ids is not None
                and child is not None
                and (not child.is_expanded)
                and id(child) in pending_node_ids
            )
            if is_pending:
                child_visits += self.config.pending_virtual_visits

            # PUCT exploration bonus
            u = c * prior * sqrt_parent / (1 + child_visits)
            score = q + u

            # Collect ties with epsilon tolerance
            if score > best_score + eps:
                best_score = score
                best_moves = [(move_id, child)]
            elif abs(score - best_score) <= eps:
                best_moves.append((move_id, child))

        # Fail-fast if no moves (should never happen after asserts)
        assert best_moves, "No selectable moves in _select_child"

        # Random tie-break using per-game RNG (reproducible per seed)
        chosen_move_id, chosen_child = self.rng.choice(best_moves)
        return chosen_move_id, chosen_child

    def _backup(self, search_path: List[MCTSNode], leaf_value: float) -> None:
        """Propagate value up the search path.

        Value alternates sign as we go up (opponent's loss is our gain).

        Args:
            search_path: Path from root to leaf
            leaf_value: Value at leaf node (from perspective of leaf's to_move)
        """
        self._total_backups += 1  # Track total backups for diagnostics
        value = leaf_value
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for parent (opponent's perspective)

    def _add_dirichlet_noise(self, root: MCTSNode, ply: int = 0) -> None:
        """Add Dirichlet noise to root priors for exploration.

        This encourages exploration of diverse moves during training.
        IMPORTANT: This function COPIES priors_raw before mutating to avoid
        noise accumulation across moves.

        Args:
            root: Root node to add noise to
            ply: Current game ply (for opening noise boost)
        """
        if not root.priors_raw:
            return

        # COPY priors_raw before mutating (prevents noise accumulation)
        assert root.priors_raw is not None
        root.priors = dict(root.priors_raw)  # Copy only for root

        # Use priors_raw keys for stable ordering / coverage
        move_ids = list(root.priors_raw.keys())
        n = len(move_ids)
        if n == 0:
            return

        # Ply-conditional alpha/eps for opening exploration boost
        boosted = self.config.opening_noise_ply > 0 and ply < self.config.opening_noise_ply
        if boosted:
            alpha = self.config.opening_dirichlet_alpha
            eps = self.config.opening_dirichlet_eps
            if _OPENDBG:
                print(f"[OPENNOISE] Boosted ply={ply}: alpha={alpha}, eps={eps}")
        else:
            alpha = self.config.dirichlet_alpha
            eps = self.config.dirichlet_eps

        # Generate Dirichlet noise using gamma distribution
        # Dirichlet(alpha) = normalize(Gamma(alpha, 1), ...)
        samples = [self.rng.gammavariate(alpha, 1.0) for _ in range(n)]
        total = sum(samples)
        if total < 1e-8:
            # Fallback to uniform if samples are too small
            noise_probs = [1.0 / n] * n
        else:
            noise_probs = [s / total for s in samples]

        # Mix with original priors: (1-eps) * prior + eps * noise
        for i, move_id in enumerate(move_ids):
            root.priors[move_id] = (1 - eps) * root.priors[move_id] + eps * noise_probs[i]

        # --- Root prior shaping (post-noise): near-corner + edge-band (plies < ply limits) ---
        S = BOARD_W  # 24

        edge_pen = self.config.root_edge_band_penalty
        edge_ply = self.config.root_edge_band_penalty_ply
        band = self.config.root_edge_band_width

        # Resolve effective near-corner penalty via the canonical selector.
        # When an early override is active for this ply it replaces the
        # baseline value; otherwise the baseline applies (or nothing, if the
        # baseline window is over). Read `effective_near_corner_penalty`
        # docstring for precedence details.
        effective_corner_pen = effective_near_corner_penalty(
            ply=ply,
            corner_penalty=self.config.root_near_corner_penalty,
            corner_penalty_ply=self.config.root_near_corner_penalty_ply,
            corner_penalty_early=self.config.root_near_corner_penalty_early,
            corner_penalty_early_plies=self.config.root_near_corner_penalty_early_plies,
        )
        R = self.config.root_near_corner_radius

        apply_edge = edge_pen > 0.0 and edge_ply > 0 and ply < edge_ply
        apply_corner = effective_corner_pen > 0.0 and R > 0

        if apply_edge or apply_corner:
            edge_mult = math.exp(-edge_pen) if apply_edge else 1.0
            corner_mult = math.exp(-effective_corner_pen) if apply_corner else 1.0

            total = 0.0
            edge_count = 0
            corner_count = 0
            edge_mass = 0.0
            corner_mass = 0.0

            # IMPORTANT: iterate stable ids; don't iterate dict view while mutating
            for mid in move_ids:
                p = root.priors[mid]
                r, c = decode_move(mid)

                in_edge = apply_edge and _is_edge_band(r, c, S, band)
                in_corner = apply_corner and _is_near_corner_cheb(r, c, S, R)

                # Max-penalty (no double-penalize overlaps)
                mult = 1.0
                if in_edge:
                    mult = min(mult, edge_mult)
                if in_corner:
                    mult = min(mult, corner_mult)

                if mult != 1.0:
                    p *= mult
                    root.priors[mid] = p

                total += p

                if in_edge:
                    edge_count += 1
                    edge_mass += p
                if in_corner:
                    corner_count += 1
                    corner_mass += p

            if total > 1e-12:
                inv = 1.0 / total
                for mid in move_ids:
                    root.priors[mid] *= inv
                edge_mass *= inv
                corner_mass *= inv

            if _OPENDBG:
                if apply_edge:
                    print(f"[EDGEBAND] ply={ply}: {edge_count}/{len(move_ids)} in band, mass={edge_mass:.3f}, penalty={edge_pen}, B={band}")
                if apply_corner:
                    # Flag when the early override is the one in effect so
                    # diagnostic logs stay unambiguous.
                    early_active = (
                        self.config.root_near_corner_penalty_early > 0.0
                        and self.config.root_near_corner_penalty_early_plies > 0
                        and ply < self.config.root_near_corner_penalty_early_plies
                    )
                    src = "EARLY" if early_active else "BASE"
                    print(f"[NEARCORNER] ply={ply}: {corner_count}/{len(move_ids)} in R, mass={corner_mass:.3f}, penalty={effective_corner_pen} ({src}), R={R}")
        # --- end root prior shaping ---

    def _terminal_value(self, state: TwixtState) -> float:
        """Get value for terminal state from perspective of to_move.

        Convention:
        - +1.0 if winner == to_move (current player won)
        - -1.0 if winner != to_move (current player lost)
        - 0.0 if draw

        Note: In TwixT, when a terminal state is reached, the winner is
        typically the player who just moved (opponent of to_move), so the
        value is usually -1.0 for the player to move.
        """
        winner = state.winner()
        if winner is None:
            return 0.0  # Draw
        elif winner == state.to_move:
            return 1.0  # Current player won
        else:
            return -1.0  # Current player lost

    def select_move(
        self,
        visit_counts: Dict[Tuple[int, int], int],
        ply: int,
    ) -> Tuple[int, int]:
        """Select move from visit counts using temperature.

        Args:
            visit_counts: Dict mapping move -> visit count
            ply: Current ply number (for temperature selection)

        Returns:
            Selected move (row, col)
        """
        # Defensive guard: fail fast if called with no moves
        if not visit_counts:
            raise AssertionError("select_move called with empty visit_counts")

        # Determine temperature based on game phase
        if ply < self.config.temp_threshold_ply:
            temp = self.config.temp_high
        else:
            temp = self.config.temp_low

        moves = list(visit_counts.keys())
        counts = [visit_counts[m] for m in moves]

        if temp < 0.01:
            # Random tie-break using per-game RNG (reproducible per seed)
            max_count = max(counts)
            best_moves = [m for m, c in zip(moves, counts) if c == max_count]
            return self.rng.choice(best_moves)

        # Temperature-scaled softmax over visit counts
        # Use log(count) / temp then softmax for numerical stability
        log_counts = [math.log(c + 1e-8) / temp for c in counts]
        max_log = max(log_counts)
        exp_counts = [math.exp(lc - max_log) for lc in log_counts]
        total = sum(exp_counts)
        probs = [e / total for e in exp_counts]

        # Sample from distribution
        r = self.rng.random()
        cumsum = 0.0
        for move, prob in zip(moves, probs):
            cumsum += prob
            if r <= cumsum:
                return move

        return moves[-1]  # Fallback (numerical edge case)

    def get_policy_target(
        self,
        visit_counts: Dict[Tuple[int, int], int],
    ) -> Dict[Tuple[int, int], float]:
        """Convert visit counts to policy target (normalized).

        Args:
            visit_counts: Dict mapping move -> visit count

        Returns:
            Dict mapping move -> probability (sums to 1)
        """
        total = sum(visit_counts.values())
        if total == 0:
            # Uniform distribution
            n = len(visit_counts)
            return {m: 1.0 / n for m in visit_counts}

        return {m: c / total for m, c in visit_counts.items()}
