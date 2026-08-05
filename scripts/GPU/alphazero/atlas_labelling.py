"""Atlas labelling and capacity sizing -- design sections 5 and 3, FROZEN.

Pure: consumes Stage 3 LegResult rows and nothing else, so the whole stage
qualifies on synthetic input with no reservoir and no GPU.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

# Frozen section 5 thresholds.
STABLE_VALUE_TOL = 0.10          # |V3200 - V6400|
STABLE_TOP_TWO_MARGIN = 0.05     # normalized, at 6400
MISLEADING_VALUE_GAP = 0.25      # |V400 - V6400|
STABLE_NEGATIVE_VALUE_TOL = 0.10

# Frozen section 3 sizing.
PILOT_GAMES = 24
DISCOVERY_FRACTION = 0.6
VALIDATION_FRACTION = 0.4
MARGIN = 1.20
MIN_VALIDATION_MISLEADING = 20
MIN_VALIDATION_STABLE_NEGATIVE = 25
ALLOWED_N = (200, 240, 280, 320, 360, 400)


def _by_b(legs: Sequence[Any]) -> Dict[int, Any]:
    d = {l.nominal_B: l for l in legs}
    missing = {400, 1600, 3200, 6400} - set(d)
    if missing:
        raise ValueError(f"missing rungs {sorted(missing)}; all four are required")
    return d


def stable_reference(legs: Sequence[Any]) -> Dict[str, Any]:
    """Section 5: 3,200 and 6,400 agree on the move, their values are within
    0.10, and the 6,400 top-two margin is at least 0.05.

    Without the 3,200 rung there is no stability check at all and "6,400 is
    truth" stops being falsifiable -- which is why the ladder carries it.
    """
    d = _by_b(legs)
    moves_agree = d[3200].selected_move == d[6400].selected_move
    value_close = abs(d[3200].root_value - d[6400].root_value) <= STABLE_VALUE_TOL
    margin = d[6400].top_two_margin
    margin_ok = margin is not None and margin >= STABLE_TOP_TWO_MARGIN
    return {
        "stable": bool(moves_agree and value_close and margin_ok),
        "moves_agree": moves_agree, "value_close": value_close,
        "margin_ok": margin_ok, "top_two_margin": margin,
        "stable_deep_move": d[6400].selected_move if moves_agree else None,
    }


def classify_row(legs: Sequence[Any]) -> str:
    d = _by_b(legs)
    ref = stable_reference(legs)
    if not ref["stable"]:
        return "no_stable_reference"
    deep = ref["stable_deep_move"]
    value_gap = abs(d[400].root_value - d[6400].root_value)
    same_move = d[400].selected_move == deep
    # Misleading is an OR; stable-negative is an AND. The asymmetry is
    # deliberate and the ambiguous band between them is kept, not forced.
    if (value_gap >= MISLEADING_VALUE_GAP) or (not same_move):
        return "misleading"
    if same_move and value_gap <= STABLE_NEGATIVE_VALUE_TOL:
        return "stable_negative"
    return "ambiguous"


def class_counts(rows: Sequence[Sequence[Any]]) -> Dict[str, int]:
    """Counts, with the misleading components reported separately (section 5)."""
    c = {"misleading": 0, "stable_negative": 0, "ambiguous": 0,
         "no_stable_reference": 0, "misleading_by_value": 0,
         "misleading_by_move": 0}
    for legs in rows:
        label = classify_row(legs)
        c[label] += 1
        if label == "misleading":
            d = _by_b(legs)
            deep = stable_reference(legs)["stable_deep_move"]
            if abs(d[400].root_value - d[6400].root_value) >= MISLEADING_VALUE_GAP:
                c["misleading_by_value"] += 1
            if d[400].selected_move != deep:
                c["misleading_by_move"] += 1
    return c


def _round_up_40(x: float) -> int:
    return int(math.ceil(x / 40.0) * 40)


def size_from_pilot(counts: Dict[str, int], pilot_n: int = PILOT_GAMES
                    ) -> Dict[str, Any]:
    """Section 3's frozen staged sizing. Fails CLOSED on a zero frequency or a
    requirement above 400 -- never defaults a number."""
    p_m = counts.get("misleading", 0) / pilot_n
    p_s = counts.get("stable_negative", 0) / pilot_n
    base = {"p_m": p_m, "p_s": p_s}
    if p_m == 0 or p_s == 0:
        return {**base, "verdict": "PROJECTED_CAPACITY_NO_GO", "N": None,
                "reason": "a pilot class frequency is zero; no N can satisfy it"}
    need = max(MARGIN * MIN_VALIDATION_MISLEADING / (VALIDATION_FRACTION * p_m),
               MARGIN * MIN_VALIDATION_STABLE_NEGATIVE / (VALIDATION_FRACTION * p_s))
    n = _round_up_40(need)
    if n > max(ALLOWED_N):
        return {**base, "verdict": "PROJECTED_CAPACITY_NO_GO", "N": None,
                "required": n,
                "reason": f"required N {n} exceeds the frozen maximum {max(ALLOWED_N)}"}
    return {**base, "verdict": "OK", "N": max(n, min(ALLOWED_N)), "required": n}


def final_capacity_gate(counts: Dict[str, int]) -> Dict[str, Any]:
    """Section 3: the completed VALIDATION split must hold >=20 misleading and
    >=25 stable-negative. Otherwise the atlas ends as an operational capacity
    failure -- do not weaken labels, move ambiguous rows, or add games."""
    short = []
    if counts.get("misleading", 0) < MIN_VALIDATION_MISLEADING:
        short.append("misleading")
    if counts.get("stable_negative", 0) < MIN_VALIDATION_STABLE_NEGATIVE:
        short.append("stable_negative")
    return {"verdict": "CAPACITY_FAILURE" if short else "OK",
            "short_of": short, "counts": dict(counts)}
