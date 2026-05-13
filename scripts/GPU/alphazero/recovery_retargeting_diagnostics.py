"""Spec 4 — Recovery / re-targeting diagnostic.

Per-side, per-game tracker that detects collapse/re-targeting failure
patterns from MCTS root values and visit-share concentration. Diagnostic
only: does not affect MCTS, selection, or training targets.

Spec: docs/superpowers/specs/2026-05-12-recovery-retargeting-diagnostic-design.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RecoveryRetargetingConfig:
    enabled: bool = True
    collapse_value_threshold: float = -0.75
    severe_collapse_value_threshold: float = -0.90
    diffuse_root_top1_threshold: float = 0.20
    very_diffuse_root_top1_threshold: float = 0.15
    delta_threshold: float = 0.50
    delta_max_current_score: float = -0.30
    alternate_component_min_size: int = 4
    classify_defense: bool = True
    max_sampled_moves_per_side: int = 32
    sample_all_moves: bool = False


def validate_config(cfg: RecoveryRetargetingConfig) -> None:
    """Raise ValueError on out-of-band config. Called once at startup."""
    if not (cfg.collapse_value_threshold < cfg.delta_max_current_score):
        raise ValueError(
            f"collapse_value_threshold ({cfg.collapse_value_threshold}) must be "
            f"strictly less than delta_max_current_score ({cfg.delta_max_current_score}) "
            f"so the delta path doesn't subsume the steady-state path"
        )
    if not (cfg.severe_collapse_value_threshold <= cfg.collapse_value_threshold):
        raise ValueError(
            f"severe_collapse_value_threshold ({cfg.severe_collapse_value_threshold}) "
            f"must be <= collapse_value_threshold ({cfg.collapse_value_threshold})"
        )
    if not (cfg.very_diffuse_root_top1_threshold <= cfg.diffuse_root_top1_threshold):
        raise ValueError(
            f"very_diffuse_root_top1_threshold ({cfg.very_diffuse_root_top1_threshold}) "
            f"must be <= diffuse_root_top1_threshold ({cfg.diffuse_root_top1_threshold})"
        )
    if not (0.0 <= cfg.diffuse_root_top1_threshold <= 1.0):
        raise ValueError(
            f"diffuse_root_top1_threshold ({cfg.diffuse_root_top1_threshold}) must be in [0, 1]"
        )
    if not (0.0 <= cfg.very_diffuse_root_top1_threshold <= 1.0):
        raise ValueError(
            f"very_diffuse_root_top1_threshold ({cfg.very_diffuse_root_top1_threshold}) must be in [0, 1]"
        )
    if not (cfg.delta_threshold > 0):
        raise ValueError(f"delta_threshold ({cfg.delta_threshold}) must be > 0")
    if not (cfg.alternate_component_min_size >= 1):
        raise ValueError(
            f"alternate_component_min_size ({cfg.alternate_component_min_size}) must be >= 1"
        )
    if not (cfg.max_sampled_moves_per_side >= 0):
        raise ValueError(
            f"max_sampled_moves_per_side ({cfg.max_sampled_moves_per_side}) must be >= 0"
        )


# ---------------------------------------------------------------------------
# Component analysis helpers
# ---------------------------------------------------------------------------

_KNIGHT_OFFSETS = ((1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1))


def knight_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
    """The 8 TwixT knight-distance offsets from (r, c). No bounds check."""
    return [(r + dr, c + dc) for dr, dc in _KNIGHT_OFFSETS]


def find_components(state, side: str) -> List[frozenset]:
    """All same-color bridge-connected components for `side` on the current state.

    Uses state._get_connected_component which respects enemy-blocking of bridges
    on real states. The _StubState fixture in tests provides a simpler
    knight-neighbor walk; that's sufficient for unit testing the classifier
    logic without a full Twixt board.
    """
    pegs_of = [p for p, color in state.pegs.items() if color == side]
    seen: set = set()
    components: List[frozenset] = []
    for peg in pegs_of:
        if peg in seen:
            continue
        comp = frozenset(state._get_connected_component(peg, side))
        if not comp:
            comp = frozenset({peg})
        seen.update(comp)
        components.append(comp)
    return components


def is_local_to_existing(state, side: str, move: Tuple[int, int]) -> bool:
    """True iff `move` is at TwixT knight distance of at least one same-color peg.

    Bridge-formability is NOT required; the flag is about proximity to
    bridge-able structure, per spec §3.1.
    """
    r, c = move
    for (nr, nc) in knight_neighbors(r, c):
        if state.pegs.get((nr, nc)) == side:
            return True
    return False


def selected_component_after(state_after, side: str, move: Tuple[int, int]) -> frozenset:
    """The component containing `move` in the POST-MOVE state.

    Caller is responsible for constructing `state_after` (via state.apply_move
    or equivalent). This helper performs no state mutation — making it safe
    to call inside the per-ply hook without copying or restoring board state.
    """
    comp = frozenset(state_after._get_connected_component(move, side))
    if not comp:
        comp = frozenset({move})
    return comp
