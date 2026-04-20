from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..config.knobs import HASH_FIELDS, KNOB_SPECS, clamp_to_core, normalize_to_allowed


@dataclass
class SweepCandidate:
    knobs: Dict[str, float]
    tag: str  # e.g., soft-best / trend / explore / fixed_probe / mutate


def _mutate_one(rng: random.Random, base: Dict[str, float], *, max_changes: int = 3) -> Dict[str, float]:
    out = dict(base)
    n = rng.randint(1, max_changes)
    choices = rng.sample(list(KNOB_SPECS.keys()), k=min(n, len(KNOB_SPECS)))
    for name in choices:
        spec = KNOB_SPECS[name]
        out[name] = float(rng.choice(spec.values))
    return normalize_to_allowed(out)


def _random_candidate(rng: random.Random) -> Dict[str, float]:
    return {name: float(rng.choice(spec.values)) for name, spec in KNOB_SPECS.items()}


def generate_sweep(
    base: Dict[str, float],
    *,
    total: int = 24,
    seed: int,
    fixed_slots: int = 6,
    mutate_slots: int = 10,
) -> List[SweepCandidate]:
    """Generate a cycle's candidate knob configs.

    This is a simplified v1 that preserves your *core behaviors*:
    - reserved fixed probes
    - mutation slots
    - the rest explore/random

    As we port your exact bucket logic (soft-best/trend/best/niche/explore/mutate),
    this function becomes a thin translation of the old `suggest` command.
    """

    rng = random.Random(seed)
    base = normalize_to_allowed(base)

    cands: List[SweepCandidate] = []

    # ---------------------------------------------------------------------
    # Fixed probes (guaranteed each cycle)
    # ---------------------------------------------------------------------
    probes: List[Dict[str, float]] = []

    # Edge probes (±5)
    for delta in (-5, 5):
        k = dict(base)
        k["firstEdgeRed"] = float(base.get("firstEdgeRed", 0.0) + delta)
        probes.append(k)

        k2 = dict(base)
        k2["firstEdgeBlack"] = float(base.get("firstEdgeBlack", 0.0) + delta)
        probes.append(k2)

    # black span probes (±0.05)
    for delta in (-0.05, 0.05):
        k = dict(base)
        k["blackSpanGainMultiplier"] = float(base.get("blackSpanGainMultiplier", 0.0) + delta)
        probes.append(k)

    # coverage probes (min/max of core band as anchors)
    k_min = dict(base)
    k_min["redDoubleCoverageBonus"] = 600.0
    k_max = dict(base)
    k_max["redDoubleCoverageBonus"] = 1000.0
    probes.extend([k_min, k_max])

    # clamp sensitive knobs back to core
    for p in probes[:fixed_slots]:
        cands.append(SweepCandidate(knobs=clamp_to_core(normalize_to_allowed(p)), tag="fixed_probe"))

    # ---------------------------------------------------------------------
    # Mutations (roam full range)
    # ---------------------------------------------------------------------
    for _ in range(max(0, mutate_slots)):
        if len(cands) >= total:
            break
        cands.append(SweepCandidate(knobs=_mutate_one(rng, base), tag="mutate"))

    # ---------------------------------------------------------------------
    # Explore/random fill
    # ---------------------------------------------------------------------
    while len(cands) < total:
        k = _random_candidate(rng)
        cands.append(SweepCandidate(knobs=normalize_to_allowed(k), tag="explore"))

    return cands[:total]
