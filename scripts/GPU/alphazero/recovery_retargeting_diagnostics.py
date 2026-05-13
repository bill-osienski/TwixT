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
