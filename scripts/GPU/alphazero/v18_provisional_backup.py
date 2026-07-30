"""v18 depth-2 provisional backup -- the SINGLE implementation of the formula.

Spec: docs/superpowers/specs/2026-07-29-v18-depth2-provisional-backup-design.md
Sec 3.1 (formula), Sec 3.2 (eligibility), Sec 4.2 (helper contract).

Pure: no node traversal, no evaluator access, no RNG, no state mutation. Safe to
import from the preflight, the screen, the diagnostic and (later) from `mcts.py`.
Spec Sec 4.2 requires exactly one implementation: import this, never copy it.

PERSPECTIVE CONTRACT -- stated once, unmistakably:

    raw_leaf_value   is in the LEAF's to-move perspective
    raw_parent_value is in the PARENT's to-move perspective (the opposite side)

Worked example from the measured A pattern (spec Sec 1.1). The top positive
depth-1 child has raw BLACK value about -0.087, but it is RED to move, so
`parent.nn_value` holds +0.087. The leaf below is black to move at +0.793:

    baseline = -(+0.087) = -0.087
    residual = 0.793 - (-0.087) = 0.880
    backup at cap 0.50 = -0.087 + 0.50 = 0.413

A depth-2 leaf shares the root's side to move, so the returned backup value is in
the same perspective as `raw_leaf_value` and drops straight into the existing
sign-alternating `MCTS._backup` path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Explicit identity sentinel. Spec Sec 4.1: two tanh values differ by at most
# 2.0, so this cap can never bind -- but the real identity guarantee is the
# structural branch at the call site, which returns before reaching this module.
IDENTITY_CAP: float = 2.0

# Spec Sec 7, weakest candidate first, matching the weakest-first ladder.
CAP_GRID: tuple[float, ...] = (1.25, 1.00, 0.75, 0.50)


@dataclass(frozen=True)
class ProvisionalBackup:
    """One clip decision.

    `backup_value` enters `_backup`. The other three fields exist so the call
    site fills Sec 4.4's telemetry columns WITHOUT recomputing the residual -- a
    second computation is a second implementation.

    `clip_direction` is +1 when an unusually POSITIVE first estimate was pulled
    down, -1 when an unusually NEGATIVE one was pulled up, 0 when the cap did not
    bind. `clipped_amount` is always non-negative.
    """

    backup_value: float
    residual: float
    clipped_amount: float
    clip_direction: int


def _check_cap(cap: float) -> None:
    # bool subclasses int, so `True` would silently act as cap 1.0.
    if isinstance(cap, bool):
        raise ValueError(f"cap must be a float, not a bool: {cap!r}")
    if not isinstance(cap, (int, float)) or not math.isfinite(cap):
        raise ValueError(f"cap must be finite: {cap!r}")
    if cap <= 0.0 or cap > IDENTITY_CAP:
        raise ValueError(f"cap must satisfy 0.0 < cap <= {IDENTITY_CAP}; got {cap!r}")


def provisional_depth2_backup_value(
    raw_leaf_value: float,
    raw_parent_value: float,
    cap: float,
) -> ProvisionalBackup:
    """Clip a depth-2 leaf's raw value toward its parent's raw value.

    Spec Sec 3.1. The comparison is STRICT: `abs(residual) == cap` does not clip.
    """
    _check_cap(cap)
    if isinstance(raw_leaf_value, bool) or isinstance(raw_parent_value, bool):
        raise ValueError("raw values must be floats, not bools")
    if not math.isfinite(raw_leaf_value) or not math.isfinite(raw_parent_value):
        # Spec Sec 3.2: fail loudly. Bypassing would conceal corruption.
        raise ValueError(
            f"nonfinite raw value: leaf={raw_leaf_value!r} parent={raw_parent_value!r}")

    baseline = -raw_parent_value          # parent value in the LEAF's perspective
    residual = raw_leaf_value - baseline

    if residual > cap:
        return ProvisionalBackup(baseline + cap, residual, residual - cap, 1)
    if residual < -cap:
        return ProvisionalBackup(baseline - cap, residual, -cap - residual, -1)

    # Spec Sec 3.1: return the ORIGINAL value, never `baseline + residual`, so an
    # unbound row carries no rounding difference from shipped.
    return ProvisionalBackup(raw_leaf_value, residual, 0.0, 0)
