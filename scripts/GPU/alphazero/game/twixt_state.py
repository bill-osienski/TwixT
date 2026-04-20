"""TwixT game state for AlphaZero training.

This implementation must exactly match the Node.js version in server/gameLogic.js
for parity between training (Python) and inference (Node.js).

Constants:
    BOARD_SIZE: 24x24 board
    MAX_PLIES: 200 (forced draw after this many moves)

Draw semantics:
    - (a) No legal moves (board fills), OR
    - (b) ply >= MAX_PLIES (forced draw, even if moves exist)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import deque

import numpy as np

# Game constants (MUST match Node.js)
BOARD_SIZE = 24
# Safety clamp: should never fire if max_plies_limit is used during self-play/training.
MAX_PLIES = 600

# Type aliases
Pos = Tuple[int, int]  # (row, col)
Bridge = Tuple[Pos, Pos]  # ((r1,c1), (r2,c2)) with canonical ordering

# Knight-move offsets for TwixT bridges
KNIGHT_MOVES = [
    (-2, -1), (-2, 1),
    (-1, -2), (-1, 2),
    (1, -2), (1, 2),
    (2, -1), (2, 1),
]

# Direction channel mapping for tensor encoding
# Maps (dr, dc) delta to channel offset (0-7)
# Channels 2-9 for red links, 10-17 for black links
# Direction names: NNE, ENE, ESE, SSE, SSW, WSW, WNW, NNW
DIRECTION_TO_CHANNEL = {
    (2, 1): 0,    # NNE: +2 row, +1 col
    (1, 2): 1,    # ENE: +1 row, +2 col
    (-1, 2): 2,   # ESE: -1 row, +2 col
    (-2, 1): 3,   # SSE: -2 row, +1 col
    (-2, -1): 4,  # SSW: -2 row, -1 col
    (-1, -2): 5,  # WSW: -1 row, -2 col
    (1, -2): 6,   # WNW: +1 row, -2 col
    (2, -1): 7,   # NNW: +2 row, -1 col
}

# Channel indices for tensor encoding
CHANNEL_RED_PEGS = 0
CHANNEL_BLACK_PEGS = 1
CHANNEL_RED_LINKS_START = 2    # 2-9 (8 directions)
CHANNEL_BLACK_LINKS_START = 10  # 10-17 (8 directions)
CHANNEL_CURRENT_PLAYER = 18
CHANNEL_RED_TOP_DIST = 19
CHANNEL_RED_BOTTOM_DIST = 20
CHANNEL_BLACK_LEFT_DIST = 21
CHANNEL_BLACK_RIGHT_DIST = 22
CHANNEL_MOVE_NUMBER = 23
# Phase 2 connectivity channels (see spec 2026-04-19)
CHANNEL_RED_CONN_TOP = 24
CHANNEL_RED_CONN_BOTTOM = 25
CHANNEL_RED_CONN_BOTH = 26
CHANNEL_BLACK_CONN_LEFT = 27
CHANNEL_BLACK_CONN_RIGHT = 28
CHANNEL_BLACK_CONN_BOTH = 29

# Pre-Phase-2 channel count, preserved for backward-compat with iter-0999
# (24-channel) checkpoints. See to_tensor_v1() below.
NUM_CHANNELS_V1 = 24
NUM_CHANNELS = 30


def _canonical_bridge(p1: Pos, p2: Pos) -> Bridge:
    """Return bridge endpoints in canonical order (smaller pos first)."""
    return (p1, p2) if p1 < p2 else (p2, p1)


def _orient(ax: int, ay: int, bx: int, by: int, cx: int, cy: int) -> int:
    """Orientation test for three points.

    Returns:
        1 if CCW (counter-clockwise)
        -1 if CW (clockwise)
        0 if collinear
    """
    v = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if v > 0:
        return 1
    elif v < 0:
        return -1
    return 0


def _proper_intersect_knight(
    x1: int, y1: int, x2: int, y2: int,
    x3: int, y3: int, x4: int, y4: int
) -> bool:
    """Fast proper intersection test for TwixT knight-edges.

    For knight-move segments (delta ±1,±2 or ±2,±1):
    - Collinear overlaps cannot happen between distinct knight-edges
    - No interior lattice points exist (gcd(1,2)=1)
    - Only need pure orientation test

    Returns True if segments properly cross.
    """
    o1 = _orient(x1, y1, x2, y2, x3, y3)
    o2 = _orient(x1, y1, x2, y2, x4, y4)
    if o1 == 0 or o2 == 0 or o1 == o2:
        return False

    o3 = _orient(x3, y3, x4, y4, x1, y1)
    o4 = _orient(x3, y3, x4, y4, x2, y2)
    if o3 == 0 or o4 == 0 or o3 == o4:
        return False

    return True


@dataclass
class TwixtState:
    """Immutable-style TwixT game state.

    Methods return new state objects rather than mutating in place.
    This is important for MCTS tree search.

    Attributes:
        board_size: Physical tensor dimension (always 24)
        active_size: Playable region dimension (<= board_size)
                     Used for curriculum learning. Edges are at 0 and active_size-1.
        to_move: Current player ("red" or "black")
        pegs: Dict mapping (row, col) -> player color
        bridges: Set of canonical bridge tuples
        ply: Number of moves made (0-indexed)
    """
    board_size: int = BOARD_SIZE       # Physical tensor size (always 24)
    active_size: int = BOARD_SIZE      # Curriculum size (<= board_size)
    to_move: str = "red"  # Red moves first
    pegs: Dict[Pos, str] = field(default_factory=dict)
    bridges: Set[Bridge] = field(default_factory=set)
    ply: int = 0
    max_plies_limit: Optional[int] = None  # if set, state becomes terminal at this ply

    def __post_init__(self):
        """Validate active_size."""
        if not (1 <= self.active_size <= self.board_size):
            raise ValueError(
                f"active_size must be in [1, {self.board_size}], "
                f"got {self.active_size}"
            )

    def copy(self) -> TwixtState:
        """Create a deep copy of the state."""
        return TwixtState(
            board_size=self.board_size,
            active_size=self.active_size,
            to_move=self.to_move,
            pegs=dict(self.pegs),
            bridges=set(self.bridges),
            ply=self.ply,
            max_plies_limit=self.max_plies_limit,
        )

    def is_valid_placement(self, row: int, col: int) -> bool:
        """Check if a peg placement is valid for current player.

        Rules (with curriculum active_size):
        1. Cell must be within [0, active_size) × [0, active_size)
        2. Cell must be empty
        3. Corners of ACTIVE region are forbidden
        4. Red cannot place on left/right edges of ACTIVE region
        5. Black cannot place on top/bottom edges of ACTIVE region
        """
        active = self.active_size

        # Out of active bounds
        if row < 0 or row >= active or col < 0 or col >= active:
            return False

        # Occupied
        if (row, col) in self.pegs:
            return False

        # Corners of ACTIVE region forbidden (not board_size corners)
        if (row == 0 or row == active - 1) and (col == 0 or col == active - 1):
            return False

        # Edge restrictions by player (using active_size)
        if self.to_move == "red":
            # Red connects top<->bottom; cannot place on left/right edges
            if col == 0 or col == active - 1:
                return False
        else:
            # Black connects left<->right; cannot place on top/bottom edges
            if row == 0 or row == active - 1:
                return False

        return True

    def legal_moves(self) -> List[Pos]:
        """Return all valid move positions for current player.

        Only considers positions within [0, active_size) × [0, active_size).

        Returns:
            List of (row, col) tuples, sorted for determinism.
        """
        moves = []
        active = self.active_size
        for row in range(active):
            for col in range(active):
                if self.is_valid_placement(row, col):
                    moves.append((row, col))
        return moves

    def _find_new_bridges(self, row: int, col: int, player: str) -> List[Bridge]:
        """Find all new bridges created by placing a peg at (row, col).

        A bridge is created when:
        1. The knight-move neighbor is within active region
        2. The knight-move neighbor has a peg of the same player
        3. The bridge doesn't already exist
        4. The bridge doesn't cross any existing bridge

        Args:
            row, col: Position of the newly placed peg
            player: Color of the peg ("red" or "black")

        Returns:
            List of new bridges (canonical form)
        """
        new_bridges = []
        active = self.active_size

        for dr, dc in KNIGHT_MOVES:
            r2 = row + dr
            c2 = col + dc

            # In ACTIVE bounds? (not board_size)
            if r2 < 0 or r2 >= active or c2 < 0 or c2 >= active:
                continue

            # Same player's peg at other end?
            if self.pegs.get((r2, c2)) != player:
                continue

            # Bridge already exists?
            bridge = _canonical_bridge((row, col), (r2, c2))
            if bridge in self.bridges:
                continue

            # Would cross an existing bridge?
            if self._crosses_existing_bridge(row, col, r2, c2):
                continue

            new_bridges.append(bridge)

        return new_bridges

    def _crosses_existing_bridge(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """Check if a candidate bridge (r1,c1)-(r2,c2) crosses any existing bridge.

        Uses bbox rejection for efficiency, then proper intersection test.
        Shared endpoints are legal (not a crossing).

        Uses x=col, y=row convention to match JS.
        """
        if not self.bridges:
            return False

        # Candidate endpoints (x=col, y=row to match JS)
        a1x, a1y = c1, r1
        a2x, a2y = c2, r2

        # Candidate bbox
        a_minx = min(a1x, a2x)
        a_maxx = max(a1x, a2x)
        a_miny = min(a1y, a2y)
        a_maxy = max(a1y, a2y)

        for (br1, bc1), (br2, bc2) in self.bridges:
            # Note: bridge endpoints are (row, col) tuples

            # Check shared endpoints (legal, not a crossing)
            if (r1 == br1 and c1 == bc1) or (r1 == br2 and c1 == bc2) or \
               (r2 == br1 and c2 == bc1) or (r2 == br2 and c2 == bc2):
                continue

            # Bridge endpoints in x,y convention
            b1x, b1y = bc1, br1
            b2x, b2y = bc2, br2

            # Bbox rejection
            b_minx = min(b1x, b2x)
            b_maxx = max(b1x, b2x)
            if b_maxx < a_minx or b_minx > a_maxx:
                continue

            b_miny = min(b1y, b2y)
            b_maxy = max(b1y, b2y)
            if b_maxy < a_miny or b_miny > a_maxy:
                continue

            # Proper intersection test
            if _proper_intersect_knight(a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y):
                return True

        return False

    def apply_move(self, move: Pos) -> TwixtState:
        """Apply a move and return a new state.

        VALIDATES the move - raises ValueError if illegal.
        This catches bugs that could silently poison training data.

        Args:
            move: (row, col) position to place peg

        Returns:
            New TwixtState with move applied

        Raises:
            ValueError: If move is illegal for current active_size/player
        """
        row, col = move

        # CRITICAL: Validate move is legal for current active_size
        if not self.is_valid_placement(row, col):
            raise ValueError(
                f"Illegal move {move} for active_size={self.active_size}, "
                f"to_move={self.to_move}"
            )

        player = self.to_move

        # Create new state
        new_state = self.copy()
        new_state.pegs[(row, col)] = player
        new_state.ply += 1

        # Find and add new bridges
        new_bridges = self._find_new_bridges_on_new_state(new_state, row, col, player)
        for bridge in new_bridges:
            new_state.bridges.add(bridge)

        # Switch player
        new_state.to_move = "black" if player == "red" else "red"

        return new_state

    def _find_new_bridges_on_new_state(
        self, new_state: TwixtState, row: int, col: int, player: str
    ) -> List[Bridge]:
        """Find bridges using the new state's peg positions.

        This is needed because we check against pegs that include the new peg.
        Uses active_size for bounds checking.
        """
        new_bridges = []
        active = self.active_size

        for dr, dc in KNIGHT_MOVES:
            r2 = row + dr
            c2 = col + dc

            # In ACTIVE bounds? (not board_size)
            if r2 < 0 or r2 >= active or c2 < 0 or c2 >= active:
                continue

            if new_state.pegs.get((r2, c2)) != player:
                continue

            bridge = _canonical_bridge((row, col), (r2, c2))
            if bridge in new_state.bridges:
                continue

            # Check crossing against new_state's bridges (which don't include this yet)
            if self._crosses_existing_bridge(row, col, r2, c2):
                continue

            new_bridges.append(bridge)

        return new_bridges

    def _get_connected_component(self, start: Pos, player: str) -> Set[Pos]:
        """Get all positions connected to start via same-player bridges.

        Uses BFS traversal through bridges.
        """
        visited = set()
        component = set()
        queue = deque([start])

        while queue:
            pos = queue.popleft()
            if pos in visited:
                continue
            if self.pegs.get(pos) != player:
                continue

            visited.add(pos)
            component.add(pos)

            # Find neighbors through bridges
            for bridge in self.bridges:
                (p1r, p1c), (p2r, p2c) = bridge

                # Check if this bridge connects to our position
                nr, nc = None, None
                if (p1r, p1c) == pos:
                    nr, nc = p2r, p2c
                elif (p2r, p2c) == pos:
                    nr, nc = p1r, p1c
                else:
                    continue

                # Bridge must belong to same player (check one endpoint is enough)
                if self.pegs.get((p1r, p1c)) != player:
                    continue

                if (nr, nc) not in visited:
                    queue.append((nr, nc))

        return component

    def connectivity_masks(self, player: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (touches_goal1, touches_goal2, touches_both) masks for `player`.

        Each mask is shape (active_size, active_size), dtype float32, value
        1.0 on cells where `player` has a peg whose bridge-connected component
        touches the named goal edge; 0.0 elsewhere.

        Uses the exact same connectivity graph as `winner()` via
        `_get_connected_component`, so feature-side and game-logic-side
        connectivity can never drift.

        For red: goal1 = row 0 (top), goal2 = row active_size-1 (bottom).
        For black: goal1 = col 0 (left), goal2 = col active_size-1 (right).
        """
        active = self.active_size
        m_g1 = np.zeros((active, active), dtype=np.float32)
        m_g2 = np.zeros((active, active), dtype=np.float32)
        m_both = np.zeros((active, active), dtype=np.float32)

        # Collect player's pegs
        player_pegs = [(r, c) for (r, c), col in self.pegs.items() if col == player]
        if not player_pegs:
            return m_g1, m_g2, m_both

        # Goal-edge predicates per player
        if player == "red":
            on_g1 = lambda r, c: r == 0
            on_g2 = lambda r, c: r == active - 1
        else:  # black
            on_g1 = lambda r, c: c == 0
            on_g2 = lambda r, c: c == active - 1

        # Bucket pegs into components (via existing BFS). Pegs already seen by
        # a prior BFS are tagged so we don't recompute.
        seen: Set[Pos] = set()
        components: List[Set[Pos]] = []
        for peg in player_pegs:
            if peg in seen:
                continue
            comp = self._get_connected_component(peg, player)
            components.append(comp)
            seen.update(comp)

        # Per component: does it touch goal1? goal2? Then mark all its pegs.
        for comp in components:
            touches_g1 = any(on_g1(r, c) for (r, c) in comp)
            touches_g2 = any(on_g2(r, c) for (r, c) in comp)
            for (r, c) in comp:
                if touches_g1:
                    m_g1[r, c] = 1.0
                if touches_g2:
                    m_g2[r, c] = 1.0
                if touches_g1 and touches_g2:
                    m_both[r, c] = 1.0

        return m_g1, m_g2, m_both

    def _check_win(self, player: str) -> bool:
        """Check if player has won (connected their two edges).

        With curriculum:
        - Red wins: path from row 0 to row (active_size - 1)
        - Black wins: path from col 0 to col (active_size - 1)
        """
        active = self.active_size

        if player == "red":
            # Check each peg on top edge (row 0) within active region
            for col in range(active):
                if self.pegs.get((0, col)) == "red":
                    component = self._get_connected_component((0, col), "red")
                    # Check if any peg in component is on bottom edge (row active-1)
                    for r, c in component:
                        if r == active - 1:
                            return True
        else:
            # Check each peg on left edge (col 0) within active region
            for row in range(active):
                if self.pegs.get((row, 0)) == "black":
                    component = self._get_connected_component((row, 0), "black")
                    # Check if any peg in component is on right edge (col active-1)
                    for r, c in component:
                        if c == active - 1:
                            return True

        return False

    def winner(self) -> Optional[str]:
        """Return the winner if any, else None.

        Checks both players for a winning path.
        """
        if self._check_win("red"):
            return "red"
        if self._check_win("black"):
            return "black"
        return None

    def is_terminal(self) -> bool:
        """Check if game is over (win or draw).

        Terminal conditions:
        1. A player has won
        2. No legal moves remain (board full in playable area)
        3. ply >= max_plies_limit (dynamic cap from self-play)
        4. ply >= MAX_PLIES (safety cap, should never fire in training)
        """
        # Check for winner
        if self.winner() is not None:
            return True

        # Dynamic cap (from self-play) takes precedence
        if self.max_plies_limit is not None and self.ply >= self.max_plies_limit:
            return True

        # Safety cap (should never fire in normal training)
        if self.ply >= MAX_PLIES:
            return True

        # No legal moves = draw
        if not self.legal_moves():
            return True

        return False

    def game_result(self) -> Optional[str]:
        """Return game result: "red", "black", or "draw".

        Returns None if game is not terminal.
        """
        if not self.is_terminal():
            return None

        w = self.winner()
        if w is not None:
            return w

        return "draw"

    def __hash__(self) -> int:
        """Hash for use in caches/sets."""
        return hash((
            self.to_move,
            frozenset(self.pegs.items()),
            frozenset(self.bridges),
        ))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TwixtState):
            return False
        return (
            self.to_move == other.to_move and
            self.pegs == other.pegs and
            self.bridges == other.bridges
        )

    def __repr__(self) -> str:
        return (
            f"TwixtState(to_move={self.to_move!r}, "
            f"ply={self.ply}, "
            f"pegs={len(self.pegs)}, "
            f"bridges={len(self.bridges)})"
        )

    @classmethod
    def from_moves(cls, moves: List[Pos], active_size: int = BOARD_SIZE) -> TwixtState:
        """Create a state by applying a sequence of moves.

        Args:
            moves: List of (row, col) positions in order
            active_size: Playable region size for curriculum

        Returns:
            TwixtState after all moves applied
        """
        state = cls(active_size=active_size)
        for move in moves:
            state = state.apply_move(move)
        return state

    def to_dict(self) -> dict:
        """Serialize state to dict (for JSON compatibility)."""
        return {
            "board_size": self.board_size,
            "active_size": self.active_size,
            "to_move": self.to_move,
            "pegs": {f"{r},{c}": p for (r, c), p in self.pegs.items()},
            "bridges": [
                [[r1, c1], [r2, c2]]
                for (r1, c1), (r2, c2) in sorted(self.bridges)
            ],
            "ply": self.ply,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TwixtState:
        """Deserialize state from dict."""
        pegs = {}
        for key, player in d["pegs"].items():
            r, c = map(int, key.split(","))
            pegs[(r, c)] = player

        bridges = set()
        for (p1, p2) in d["bridges"]:
            bridges.add(_canonical_bridge(tuple(p1), tuple(p2)))

        board_size = d.get("board_size", BOARD_SIZE)
        return cls(
            board_size=board_size,
            active_size=d.get("active_size", board_size),
            to_move=d["to_move"],
            pegs=pegs,
            bridges=bridges,
            ply=d.get("ply", len(pegs)),
        )

    def to_tensor(self) -> np.ndarray:
        """Convert state to 30-channel tensor for neural network input.

        Returns:
            numpy array of shape (30, 24, 24) = (channels, rows, cols)

        CURRICULUM NOTE:
            - Playable region is [0, active_size) × [0, active_size)
            - Padded region (outside active_size) is zeroed
            - Edge distance channels use active_size as the boundary, not 24

        Channel layout:
            0: Red pegs (1 where red peg exists)
            1: Black pegs (1 where black peg exists)
            2-9: Red link directions (8 knight-move directions)
            10-17: Black link directions (8 knight-move directions)
            18: Current player indicator (1 if red to move, 0 if black)
            19: Red top edge distance (normalized 0-1, closer to row 0 = higher)
            20: Red bottom edge distance (normalized 0-1, closer to row active_size-1 = higher)
            21: Black left edge distance (normalized 0-1, closer to col 0 = higher)
            22: Black right edge distance (normalized 0-1, closer to col active_size-1 = higher)
            23: Move number / game phase (ply / MAX_PLIES, normalized 0-1)
            24: Red connected to top edge (1 on pegs whose component touches row 0)
            25: Red connected to bottom edge (1 on pegs whose component touches row active-1)
            26: Red connected to both edges (1 on pegs whose component touches top AND bottom)
            27: Black connected to left edge (1 on pegs whose component touches col 0)
            28: Black connected to right edge (1 on pegs whose component touches col active-1)
            29: Black connected to both edges (1 on pegs whose component touches left AND right)

        Link encoding: For each link, mark 1 at BOTH endpoints in the
        appropriate direction channel. This makes links visible from either end.
        """
        size = self.board_size        # Physical tensor size (always 24)
        active = self.active_size     # Curriculum playable region
        tensor = np.zeros((NUM_CHANNELS, size, size), dtype=np.float32)

        # Channel 0-1: Peg positions (only within active region can have pegs)
        for (r, c), player in self.pegs.items():
            if player == "red":
                tensor[CHANNEL_RED_PEGS, r, c] = 1.0
            else:
                tensor[CHANNEL_BLACK_PEGS, r, c] = 1.0

        # Channels 2-17: Link directions (same as before, pegs constrained to active)
        for (r1, c1), (r2, c2) in self.bridges:
            # Determine player from peg color at first endpoint
            player = self.pegs.get((r1, c1))
            if player is None:
                continue  # Should not happen with valid bridges

            # Calculate direction from endpoint 1 to endpoint 2
            dr = r2 - r1
            dc = c2 - c1
            dir_offset = DIRECTION_TO_CHANNEL.get((dr, dc))
            if dir_offset is None:
                continue  # Should not happen with valid knight moves

            # Calculate reverse direction (endpoint 2 to endpoint 1)
            rev_dr = -dr
            rev_dc = -dc
            rev_dir_offset = DIRECTION_TO_CHANNEL.get((rev_dr, rev_dc))

            # Determine base channel for this player's links
            if player == "red":
                base_channel = CHANNEL_RED_LINKS_START
            else:
                base_channel = CHANNEL_BLACK_LINKS_START

            # Mark 1 at BOTH endpoints in appropriate direction channels
            tensor[base_channel + dir_offset, r1, c1] = 1.0
            if rev_dir_offset is not None:
                tensor[base_channel + rev_dir_offset, r2, c2] = 1.0

        # Channel 18: Current player indicator (fill only active region)
        if self.to_move == "red":
            tensor[CHANNEL_CURRENT_PLAYER, :active, :active] = 1.0
        # else: already 0.0

        # Channels 19-22: Edge distances USING active_size (CHANGED for curriculum)
        # Goal edges are at 0 and active_size-1, not 0 and 23
        max_idx = max(1, active - 1)  # Avoid div-by-zero for active_size=1
        for r in range(active):
            for c in range(active):
                # Red top edge distance: closer to row 0 = higher value
                tensor[CHANNEL_RED_TOP_DIST, r, c] = 1.0 - r / max_idx
                # Red bottom edge distance: closer to row (active-1) = higher value
                tensor[CHANNEL_RED_BOTTOM_DIST, r, c] = r / max_idx
                # Black left edge distance: closer to col 0 = higher value
                tensor[CHANNEL_BLACK_LEFT_DIST, r, c] = 1.0 - c / max_idx
                # Black right edge distance: closer to col (active-1) = higher value
                tensor[CHANNEL_BLACK_RIGHT_DIST, r, c] = c / max_idx

        # Channel 23: Move number / game phase (fill only active region)
        tensor[CHANNEL_MOVE_NUMBER, :active, :active] = self.ply / MAX_PLIES

        # Channels 24-29: Connectivity masks (Phase 2 — see spec 2026-04-19)
        # Uses the same connectivity graph as winner() for feature/game-logic parity.
        m_red_top, m_red_bot, m_red_both = self.connectivity_masks("red")
        m_blk_left, m_blk_right, m_blk_both = self.connectivity_masks("black")
        tensor[CHANNEL_RED_CONN_TOP, :active, :active] = m_red_top
        tensor[CHANNEL_RED_CONN_BOTTOM, :active, :active] = m_red_bot
        tensor[CHANNEL_RED_CONN_BOTH, :active, :active] = m_red_both
        tensor[CHANNEL_BLACK_CONN_LEFT, :active, :active] = m_blk_left
        tensor[CHANNEL_BLACK_CONN_RIGHT, :active, :active] = m_blk_right
        tensor[CHANNEL_BLACK_CONN_BOTH, :active, :active] = m_blk_both

        # IMPORTANT: Regions outside active_size are already zeros (from np.zeros)
        # This is intentional for consistent curriculum training

        return tensor


def to_tensor_v1(state: "TwixtState") -> np.ndarray:
    """Produce a 24-channel tensor matching the pre-Phase-2 layout.

    Used by probe_eval.py and any other tool that needs to evaluate a
    24-channel checkpoint using the current codebase. The output matches
    exactly what the pre-Phase-2 to_tensor() would have produced:
    channels 0-23 only, no connectivity channels.
    """
    # Build the full 30-channel tensor, then slice off the last 6.
    # Safe because channels 0-23 are identical in both formats.
    full = state.to_tensor()
    return full[:NUM_CHANNELS_V1].copy()
