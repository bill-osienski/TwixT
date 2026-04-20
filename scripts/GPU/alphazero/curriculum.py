"""Curriculum manager for AlphaZero training.

Automatically promotes to larger board sizes based on win rate metrics.
This solves the "all-draws" problem where training on 24x24 never produces
terminal wins because games hit max_moves before anyone wins.

Strategy:
- Start with small boards (8x8) where games complete in ~20-30 moves
- Network learns "what winning looks like" on small boards
- Automatically promote to larger boards when criteria are met
- Value signal propagates progressively to full 24x24

Key insight: Keep tensor shape fixed at 24x24 but limit playable region
to active_size via TwixtState.active_size field.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .self_play import DRAW_TIMEOUT, DRAW_BOARD_FULL, DRAW_STATE_CAP, DRAW_UNKNOWN


@dataclass
class CurriculumManager:
    """Manages curriculum progression through board sizes.

    Promotion criteria (all must be met over rolling window):
    1. Draw rate < threshold (games are decisive)
    2. Both colors have won at least once (no degenerate strategy)
    3. Win rate reasonably balanced (no color dominance)

    Attributes:
        sizes: Tuple of board sizes to progress through
        window: Number of games to consider for metrics
        draw_threshold: Max draw rate to allow promotion (default 0.3 = 30%)
        min_wins_each: Min wins required for each color before promotion
        idx: Current index into sizes tuple
    """

    sizes: Tuple[int, ...] = (8, 10, 12, 16, 20, 24)
    window: int = 200
    draw_threshold: float = 0.3
    min_wins_each: int = 5
    idx: int = 0

    # Rolling window of recent results: (winner, draw_reason) tuples
    # e.g. ("red", None), (None, "timeout_selfplay")
    _history: List[Tuple[Optional[str], Optional[str]]] = field(default_factory=list)

    # Stability guard: require criteria met 2 consecutive times
    _promote_streak: int = 0

    @property
    def active_size(self) -> int:
        """Current curriculum board size."""
        return self.sizes[self.idx]

    @property
    def is_final(self) -> bool:
        """True if at final (largest) board size."""
        return self.idx >= len(self.sizes) - 1

    def record_game(self, winner: Optional[str], draw_reason: Optional[str] = None) -> None:
        """Record a game result.

        Args:
            winner: "red", "black", or None for draw
            draw_reason: If winner is None, one of DRAW_TIMEOUT, DRAW_BOARD_FULL, etc.
        """
        self._history.append((winner, draw_reason))
        # Keep only last `window` games
        if len(self._history) > self.window:
            self._history = self._history[-self.window :]

    def get_metrics(self) -> dict:
        """Calculate metrics over rolling window.

        Returns:
            Dict with red_wins, black_wins, draws (by type), draw_rate, etc.
        """
        if not self._history:
            # No games yet - use 0.0 for rates to avoid confusing "100% draws" printouts
            return {
                "red_wins": 0, "black_wins": 0,
                "draws": 0, "timeout_draws": 0, "board_full_draws": 0,
                "state_cap_draws": 0, "unknown_draws": 0,
                "total": 0,
                "draw_rate": 0.0, "draw_rate_true": 0.0,
                "timeout_rate": 0.0,  # Canonical name
                "draw_rate_timeout": 0.0,  # Compat alias (remove after 1-2 runs)
                "red_win_rate": 0.0, "black_win_rate": 0.0,
            }

        red_wins = sum(1 for w, _ in self._history if w == "red")
        black_wins = sum(1 for w, _ in self._history if w == "black")

        # Break down draws by reason (using constants)
        timeout_draws = sum(1 for w, dr in self._history if w is None and dr == DRAW_TIMEOUT)
        board_full_draws = sum(1 for w, dr in self._history if w is None and dr == DRAW_BOARD_FULL)
        state_cap_draws = sum(1 for w, dr in self._history if w is None and dr == DRAW_STATE_CAP)
        unknown_draws = sum(1 for w, dr in self._history if w is None and dr == DRAW_UNKNOWN)

        draws = timeout_draws + board_full_draws + state_cap_draws + unknown_draws
        true_draws = draws - timeout_draws  # Non-timeout draws
        total = len(self._history)

        decisive = red_wins + black_wins
        draw_rate = draws / total if total > 0 else 0.0
        draw_rate_true = true_draws / total if total > 0 else 0.0
        draw_rate_timeout = timeout_draws / total if total > 0 else 0.0
        # Note: win rates are conditional on decisive games (excludes draws)
        red_win_rate = red_wins / decisive if decisive > 0 else 0.0
        black_win_rate = black_wins / decisive if decisive > 0 else 0.0

        return {
            "red_wins": red_wins, "black_wins": black_wins,
            "draws": draws, "timeout_draws": timeout_draws,
            "board_full_draws": board_full_draws, "state_cap_draws": state_cap_draws,
            "unknown_draws": unknown_draws,
            "total": total,
            "draw_rate": draw_rate, "draw_rate_true": draw_rate_true,
            "timeout_rate": draw_rate_timeout,  # Canonical name
            "draw_rate_timeout": draw_rate_timeout,  # Compat alias (remove after 1-2 runs)
            "red_win_rate": red_win_rate, "black_win_rate": black_win_rate,
        }

    def should_promote(self) -> bool:
        """Check if ready to promote to next size.

        Criteria:
        1. Not already at final size
        2. Have enough games in window
        3. Draw rate below threshold
        4. Both colors have won at least min_wins_each
        5. Timeout rate below 15% (don't promote if timeouts dominate)
        """
        if self.is_final:
            return False

        if len(self._history) < self.window // 2:
            # Need at least half window of data
            return False

        metrics = self.get_metrics()

        # Check draw rate (true draws only, ignoring timeouts)
        if metrics["draw_rate_true"] > self.draw_threshold:
            return False

        # Check both colors winning
        if metrics["red_wins"] < self.min_wins_each:
            return False
        if metrics["black_wins"] < self.min_wins_each:
            return False

        # Check timeout rate - don't promote if timeouts dominate
        if metrics["timeout_rate"] > 0.15:
            return False

        return True

    def maybe_promote(self) -> bool:
        """Promote to next size if criteria met for 2 consecutive checks.

        Uses a stability guard to prevent premature promotion on noise.

        Returns:
            True if promotion occurred
        """
        if self.should_promote():
            self._promote_streak += 1
            if self._promote_streak >= 2:
                self.idx += 1
                # Clear history and streak for new size
                self.reset_history()
                return True
        else:
            self._promote_streak = 0
        return False

    def reset_history(self) -> None:
        """Clear game history and promotion streak (call after size change)."""
        self._history = []
        self._promote_streak = 0

    def demote(self) -> bool:
        """Demote to previous (smaller) board size.

        Returns:
            True if demotion occurred, False if already at minimum size.
        """
        if self.idx <= 0:
            return False
        self.idx -= 1
        self.reset_history()
        return True

    def to_dict(self) -> dict:
        """Serialize for checkpoint."""
        return {
            "sizes": list(self.sizes),
            "window": self.window,
            "draw_threshold": self.draw_threshold,
            "min_wins_each": self.min_wins_each,
            "idx": self.idx,
            "history": self._history,
            "promote_streak": self._promote_streak,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CurriculumManager:
        """Deserialize from checkpoint."""
        cm = cls(
            sizes=tuple(d.get("sizes", (8, 10, 12, 16, 20, 24))),
            window=d.get("window", 200),
            draw_threshold=d.get("draw_threshold", 0.3),
            min_wins_each=d.get("min_wins_each", 5),
            idx=d.get("idx", 0),
        )
        # Handle both old format (list of str/None) and new format (list of tuples)
        raw_history = d.get("history", [])
        if raw_history and isinstance(raw_history[0], (list, tuple)):
            # New format: list of [winner, draw_reason] - convert to tuples
            cm._history = [(tuple(h) if h else (None, None)) for h in raw_history]
        else:
            # Old format: list of winners only - convert to (winner, None) tuples
            cm._history = [(w, None) for w in raw_history]
        cm._promote_streak = d.get("promote_streak", 0)
        return cm

    def __repr__(self) -> str:
        metrics = self.get_metrics()
        return (
            f"CurriculumManager(active_size={self.active_size}, "
            f"idx={self.idx}/{len(self.sizes)-1}, "
            f"history={metrics['total']}, "
            f"draw_rate={metrics['draw_rate']:.1%})"
        )
