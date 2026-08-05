"""Atlas Read-out C producer -- design section 8, EXECUTION-FROZEN.

Accumulates counters ONLINE. No event objects, no event logs, no node mutation.

The per-node prior-rank cache keys `id(node)` because `MCTSNode` is a plain
`@dataclass` and therefore unhashable, and it MUST be cleared at every
`advance_root`: detaching a subtree frees `id()` values for reuse, so a
longer-lived cache would silently return another node's ranks.

CPU-SAFE: stdlib only, no MLX, no scipy.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

# Frozen widening shapes (design section 8). Matched at the root, divergent below.
WIDENING_SHAPES: Tuple[Tuple[str, float, float], ...] = (
    ("c4a05", 4.0, 0.5),
    ("c13a03", 13.0, 0.3),
)

# Frozen: at least 10% of first-touch selections outside the admitted set.
MEANINGFUL_INTERVENTION_FIRST_TOUCH_RATE = 0.10

DEPTH_BUCKETS = ("0", "1", "2+")


def _bucket(depth: int) -> str:
    return "0" if depth == 0 else ("1" if depth == 1 else "2+")


def k_of_n(n: int, c: float, alpha: float, n_legal: int) -> int:
    """K(n) = min(n_legal, max(1, ceil(C * n^alpha)))."""
    return min(n_legal, max(1, math.ceil(c * (n ** alpha))))


def n_admit(rank: int, c: float, alpha: float, n_legal: int) -> int:
    """min { n >= 0 integer : K(n) >= rank }, computed as a SEARCH.

    The closed form ceil((rank/C)^(1/alpha)) is WRONG: it inverts C*n^alpha >= r
    and discards the ceil inside K. Counterexample (C=4, alpha=0.5, r=9): the
    closed form gives 6, but K(5) = ceil(4*sqrt(5)) = 9 >= 9, so the answer is 5.
    """
    n = 0
    while k_of_n(n, c, alpha, n_legal) < rank:
        n += 1
    return n


def _empty_cell() -> Dict[str, Any]:
    return {
        "eligible_events": 0,
        "outside_events": 0,
        "first_touch_events": 0,
        "first_touch_outside_events": 0,
        "excluded_prior_mass": 0.0,
    }


class SelectionTracer:
    """One tracer per row. Counters only -- no events retained."""

    def __init__(self) -> None:
        # move -> rank, cumulative prior mass, total prior mass.
        self._cache: Dict[int, Tuple[Dict[int, int], List[float], float]] = {}
        self._cells: Dict[str, Dict[str, Dict[str, Any]]] = {
            shape: {b: _empty_cell() for b in ("overall",) + DEPTH_BUCKETS}
            for shape, _c, _a in WIDENING_SHAPES
        }
        self._forced_bypass: Dict[str, Dict[str, int]] = {
            shape: {"events": 0, "outside_events": 0}
            for shape, _c, _a in WIDENING_SHAPES
        }
        self._within_forced_events = 0

    # -- cache ---------------------------------------------------------
    def _ranks_for(self, parent: Any) -> Tuple[Dict[int, int], List[float], float]:
        """Prior rank (adjusted prior DESCENDING, move-ID ASCENDING) plus the
        cumulative prior mass, which is what makes event-weighted excluded mass
        cheap: mass outside top-K is `total - cum[K]`.
        """
        key = id(parent)
        hit = self._cache.get(key)
        if hit is None:
            ordered = sorted(parent.priors.items(), key=lambda kv: (-kv[1], kv[0]))
            ranks = {mv: i + 1 for i, (mv, _p) in enumerate(ordered)}
            cum, run = [0.0], 0.0
            for _mv, pr in ordered:
                run += float(pr)
                cum.append(run)
            hit = (ranks, cum, run)
            self._cache[key] = hit
        return hit

    def clear_node_cache(self) -> None:
        """MUST be called at every `advance_root` -- see the module docstring."""
        self._cache.clear()

    # -- hook ----------------------------------------------------------
    def on_select_child(self, parent, selected_move, existing_child, depth,
                        parent_completed_visits, root_override,
                        within_forced_simulation) -> None:
        ranks, cum, total_mass = self._ranks_for(parent)
        rank = ranks[selected_move]
        n_legal = len(parent.priors)
        first_touch = existing_child is None
        if within_forced_simulation:
            self._within_forced_events += 1

        for shape, c, alpha in WIDENING_SHAPES:
            k = k_of_n(parent_completed_visits, c, alpha, n_legal)
            outside = rank > k
            if root_override:
                # Bypasses widening: excluded from the primary denominator,
                # reported separately (design section 8).
                self._forced_bypass[shape]["events"] += 1
                if outside:
                    self._forced_bypass[shape]["outside_events"] += 1
                continue
            # Event-weighted excluded mass: the total prior mass OUTSIDE the
            # admitted top-K set at this event -- NOT the selected move's prior.
            # The selected-move reading reports ZERO for an event whose selected
            # move is admitted while most of the distribution is excluded.
            excluded_mass = total_mass - cum[min(k, len(cum) - 1)]
            for key in ("overall", _bucket(depth)):
                cell = self._cells[shape][key]
                cell["eligible_events"] += 1
                cell["excluded_prior_mass"] += excluded_mass
                if outside:
                    cell["outside_events"] += 1
                if first_touch:
                    cell["first_touch_events"] += 1
                    if outside:
                        cell["first_touch_outside_events"] += 1

    # -- output --------------------------------------------------------
    @staticmethod
    def _rate(num: int, den: int) -> Optional[float]:
        """Zero denominator -> None. Never 0.0, never False."""
        return None if den == 0 else num / den

    def snapshot(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "by_shape": {},
            "within_forced_events": self._within_forced_events,
        }
        for shape, _c, _a in WIDENING_SHAPES:
            cells = {}
            for key, cell in self._cells[shape].items():
                cells[key] = dict(cell)
                cells[key]["outside_rate"] = self._rate(
                    cell["outside_events"], cell["eligible_events"])
                cells[key]["first_touch_outside_rate"] = self._rate(
                    cell["first_touch_outside_events"], cell["first_touch_events"])
                # Reported as an event-wise MEAN, per the spec amendment.
                cells[key]["mean_excluded_prior_mass"] = self._rate(
                    cell["excluded_prior_mass"], cell["eligible_events"])
            ft_rate = cells["overall"]["first_touch_outside_rate"]
            bypass = dict(self._forced_bypass[shape])
            out["by_shape"][shape] = {
                **cells,
                "forced_root_bypass_events": bypass["events"],
                "forced_root_bypass_outside_events": bypass["outside_events"],
                "forced_root_bypass_outside_rate": self._rate(
                    bypass["outside_events"], bypass["events"]),
                # None, never False, when the denominator is empty.
                "meaningfully_affected": (
                    None if ft_rate is None
                    else ft_rate >= MEANINGFUL_INTERVENTION_FIRST_TOUCH_RATE
                ),
            }
        return out
