from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class KnobSpec:
    """Defines a single tunable parameter.

    Rules of thumb:
    - Keep `values` discrete and ordered.
    - Keep the ordered hash field list (`HASH_FIELDS`) stable over time.
    """

    name: str
    values: List[float]
    default: float
    category: str = "misc"

    def validate(self) -> None:
        if self.default not in self.values:
            raise ValueError(f"Default {self.default!r} not present in values for {self.name}")


def _frange(start: float, stop: float, step: float) -> List[float]:
    # Inclusive range with stable rounding.
    vals: List[float] = []
    x = start
    i = 0
    while x <= stop + 1e-12:
        vals.append(round(x, 10))
        i += 1
        x = start + i * step
    return vals


# -----------------------------------------------------------------------------
# Stable knob registry
# -----------------------------------------------------------------------------

# The *ordered* list of knobs that participate in hashing.
# Do not reorder without migrating historical hashes/logs.
HASH_FIELDS: List[str] = [
    "firstEdgeRed",
    "firstEdgeBlack",
    "finishPenalty",
    "redFinishPenaltyFactor",
    "blackFinishScaleMultiplier",
    "redSpanGainMultiplier",
    "blackSpanGainMultiplier",
    "redDoubleCoverageBonus",
    "blackDoubleCoverageScale",
    "gapDecayScale",
    "connectorBonusScale",
    "finishBonusScale",
]


def _ivals(a: int, b: int, step: int) -> List[float]:
    return [float(x) for x in range(a, b + 1, step)]


# Defaults match current JS search.json values
KNOB_SPECS: Dict[str, KnobSpec] = {
    "firstEdgeRed": KnobSpec("firstEdgeRed", _ivals(400, 440, 5), 415.0, category="edge"),
    "firstEdgeBlack": KnobSpec("firstEdgeBlack", _ivals(435, 465, 5), 450.0, category="edge"),
    "finishPenalty": KnobSpec("finishPenalty", _ivals(1161, 1221, 20), 1181.0, category="finish"),
    "redFinishPenaltyFactor": KnobSpec("redFinishPenaltyFactor", _frange(0.15, 0.5, 0.05), 0.2, category="finish"),
    "blackFinishScaleMultiplier": KnobSpec("blackFinishScaleMultiplier", _frange(0.85, 1.15, 0.05), 1.0, category="finish"),
    "redSpanGainMultiplier": KnobSpec("redSpanGainMultiplier", _frange(0.8, 1.2, 0.05), 0.95, category="span"),
    "blackSpanGainMultiplier": KnobSpec("blackSpanGainMultiplier", _frange(0.85, 1.15, 0.05), 1.0, category="span"),
    "redDoubleCoverageBonus": KnobSpec("redDoubleCoverageBonus", _ivals(400, 1000, 100), 600.0, category="coverage"),
    "blackDoubleCoverageScale": KnobSpec("blackDoubleCoverageScale", _frange(0.7, 1.0, 0.05), 0.85, category="coverage"),
    "gapDecayScale": KnobSpec("gapDecayScale", _frange(0.9, 1.1, 0.02), 1.0, category="shape"),
    "connectorBonusScale": KnobSpec("connectorBonusScale", _frange(0.9, 1.1, 0.02), 1.0, category="shape"),
    "finishBonusScale": KnobSpec("finishBonusScale", _frange(0.9, 1.1, 0.02), 1.0, category="finish"),
}


# Core bands used for clamping in sensitive buckets (centered on JS defaults).
CORE_BANDS: Dict[str, Tuple[float, float]] = {
    "redSpanGainMultiplier": (0.9, 1.0),
    "blackSpanGainMultiplier": (0.95, 1.05),
    "redDoubleCoverageBonus": (500.0, 700.0),
    "blackDoubleCoverageScale": (0.8, 0.9),
    "redFinishPenaltyFactor": (0.15, 0.3),
    "blackFinishScaleMultiplier": (0.95, 1.05),
}


def validate_registry() -> None:
    """Sanity checks that should hold for every run."""
    for name in HASH_FIELDS:
        if name not in KNOB_SPECS:
            raise ValueError(f"HASH_FIELDS contains {name!r} but no spec exists")
    for spec in KNOB_SPECS.values():
        spec.validate()


def defaults() -> Dict[str, float]:
    return {k: float(v.default) for k, v in KNOB_SPECS.items()}


def clamp_to_core(knobs: Dict[str, float], fields: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """Clamp selected knobs to their core band.

    - If `fields` is None, clamps every knob with a core band.
    - Returns a *copy*.
    """

    out = dict(knobs)
    fset = set(fields) if fields is not None else set(CORE_BANDS.keys())

    for name, (lo, hi) in CORE_BANDS.items():
        if name not in fset or name not in out:
            continue
        val = float(out[name])
        if val < lo:
            out[name] = lo
        elif val > hi:
            out[name] = hi
    return out


def nearest_allowed(name: str, val: float) -> float:
    spec = KNOB_SPECS[name]
    best = spec.values[0]
    best_d = abs(best - val)
    for v in spec.values[1:]:
        d = abs(v - val)
        if d < best_d:
            best = v
            best_d = d
    return float(best)


def normalize_to_allowed(knobs: Dict[str, float]) -> Dict[str, float]:
    """Snap all known knobs to their nearest allowed discrete values."""
    out = dict(knobs)
    for name, spec in KNOB_SPECS.items():
        if name not in out:
            out[name] = float(spec.default)
            continue
        out[name] = nearest_allowed(name, float(out[name]))
    return out
