"""Atlas Read-out B -- design section 7, FROZEN.

Calibration, not a hypothesis: does an inherited collateral gate fire on changes
that move TOWARD the stable deeper reference?

Rows carry phase and flat/near-even facts, because a bare list of LegResults
cannot identify the strata section 7 requires.

The outcome is "needs review", never "invalid".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .atlas_labelling import stable_reference

COLLAPSE_TOP_SHARE = 0.95
VALUE_CONVERGENCE_TOL = 0.10
HALF = 0.5
MIN_ELIGIBLE_TRIGGERS = 10
MIN_CONVERGENT_RATE = 0.75
BASE_RATE_MARGIN = 0.15
GATE_NAMES = ("new_collapse", "lower_prior_flip",
              "effective_children_drop", "top_share_increase")


def _by_b(legs: Sequence[Any]) -> Dict[int, Any]:
    return {l.nominal_B: l for l in legs}


def gate_triggers(legs: Sequence[Any], hi: int = 1600) -> Dict[str, bool]:
    """Historical metrics on the 400 -> `hi` transition.

    `hi` is a parameter so the required 400 -> 6,400 natural-convergence report
    reuses this code rather than duplicating it.
    """
    d = _by_b(legs)
    a, b = d[400], d[hi]
    return {
        "new_collapse": (a.top_share is not None and b.top_share is not None
                         and a.top_share < COLLAPSE_TOP_SHARE
                         and b.top_share >= COLLAPSE_TOP_SHARE),
        "lower_prior_flip": (a.selected_move != b.selected_move
                             and a.selected_move_prior_rank is not None
                             and b.selected_move_prior_rank is not None
                             and b.selected_move_prior_rank
                             > a.selected_move_prior_rank),
        "effective_children_drop": (a.effective_children is not None
                                    and b.effective_children is not None
                                    and b.effective_children
                                    < a.effective_children),
        "top_share_increase": (a.top_share is not None and b.top_share is not None
                               and b.top_share > a.top_share),
    }


COMPOUND_EFF_CHILDREN_REDUCTION = 0.50
COMPOUND_TOP_SHARE_INCREASE = 0.15


def compound_narrowing(rows: Sequence[Dict[str, Any]], hi: int = 1600
                       ) -> Optional[bool]:
    """Section 7's compound condition -- an AGGREGATE over the cohort, not a
    per-row boolean (spec amendment).

    mean effective-children reduction >= 0.50 AND mean top-share increase
    >= 0.15. A per-row directional test would fire on any row that narrowed at
    all, however slightly, and would report compound narrowing where the
    historical gate saw none.

    None when the cohort has no defined aggregate -- inapplicable is not failing.
    """
    # COMPLETE CASE: both means must describe the SAME cohort. Accumulating them
    # independently lets a partially missing row contribute to one mean and not
    # the other, so the two would summarize different row sets.
    reductions, increases = [], []
    for row in rows:
        d = _by_b(row["legs"])
        a, b = d[400], d[hi]
        if (a.effective_children is None or b.effective_children is None
                or a.effective_children <= 0
                or a.top_share is None or b.top_share is None):
            return None            # any incomplete row makes the aggregate None
        reductions.append((a.effective_children - b.effective_children)
                          / a.effective_children)
        increases.append(b.top_share - a.top_share)
    if not reductions:
        return None
    mean_red = sum(reductions) / len(reductions)
    mean_inc = sum(increases) / len(increases)
    return bool(mean_red >= COMPOUND_EFF_CHILDREN_REDUCTION
                and mean_inc >= COMPOUND_TOP_SHARE_INCREASE)


def closes_half(m400: Optional[float], m1600: Optional[float],
                D: Optional[float]) -> bool:
    """|m400 - D| > 0 AND |m1600 - D| <= 0.5 * |m400 - D|.

    The `> 0` guard matters: with no gap, "closes half" is vacuous and must not
    fire rather than firing trivially.
    """
    if m400 is None or m1600 is None or D is None:
        return False
    gap = abs(m400 - D)
    if gap <= 0:
        return False
    return abs(m1600 - D) <= HALF * gap


def convergent(legs: Sequence[Any], ref: Dict[str, Any]) -> Dict[str, Any]:
    """The FROZEN section 7 predicate. Persistence is a JOINT requirement."""
    d = _by_b(legs)
    deep = ref.get("stable_deep_move")
    move_conv = d[400].selected_move != deep and d[1600].selected_move == deep
    value_conv = (abs(d[1600].root_value - d[6400].root_value)
                  <= abs(d[400].root_value - d[6400].root_value)
                  - VALUE_CONVERGENCE_TOL)
    # SAME metric toward BOTH deep rungs. The disjunction is over METRICS, not
    # rungs: mixing one metric's 3,200 agreement with another's 6,400 is not
    # evidence.
    ts = (closes_half(d[400].top_share, d[1600].top_share, d[3200].top_share)
          and closes_half(d[400].top_share, d[1600].top_share, d[6400].top_share))
    ec = (closes_half(d[400].effective_children, d[1600].effective_children,
                      d[3200].effective_children)
          and closes_half(d[400].effective_children, d[1600].effective_children,
                          d[6400].effective_children))
    dist_conv = (d[1600].selected_move == deep) and (ts or ec)
    persistent = (d[1600].selected_move == d[3200].selected_move
                  == d[6400].selected_move)
    return {"move_convergent": move_conv, "value_convergent": value_conv,
            "dist_convergent": dist_conv, "persistent": persistent,
            "convergent": bool(persistent
                               and (move_conv or value_conv or dist_conv))}


def calibrate_gate(rows: Sequence[Dict[str, Any]], gate_name: str
                   ) -> Dict[str, Any]:
    """Section 7's frozen "needs review" rule, on the ELIGIBLE denominator."""
    total = eligible = confirmed = eligible_rows = base_conv = 0
    for row in rows:
        legs = row["legs"]
        ref = stable_reference(legs)
        fired = gate_triggers(legs)[gate_name]
        if fired:
            total += 1
        if not ref["stable"]:
            continue                      # unclassifiable: excluded, not counted
        eligible_rows += 1
        conv = convergent(legs, ref)["convergent"]
        base_conv += 1 if conv else 0
        if fired:
            eligible += 1
            confirmed += 1 if conv else 0

    rate = (confirmed / eligible) if eligible else None
    base_rate = (base_conv / eligible_rows) if eligible_rows else None
    needs_review = (eligible >= MIN_ELIGIBLE_TRIGGERS
                    and rate is not None and rate >= MIN_CONVERGENT_RATE
                    and base_rate is not None
                    and (rate - base_rate) >= BASE_RATE_MARGIN)
    return {"gate": gate_name, "total_triggers": total,
            "eligible_triggers": eligible,
            "eligible_trigger_fraction": ((eligible / total) if total else None),
            "confirmed_convergent": confirmed, "convergent_rate": rate,
            "base_convergent_rate": base_rate,
            # "needs review" means the gate structure must be reviewed and
            # frozen before it judges another prototype. It does NOT mean the
            # gate is invalid and does not authorize deleting or relaxing it.
            "verdict": "needs review" if needs_review else "no finding"}


def natural_convergence_report(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Section 7: the same metrics at 400 -> 6,400, to show the SCALE of natural
    deeper-search change.

    This is the natural-convergence reference distribution. It is explicitly
    NOT causal evidence that a same-budget intervention is safe.
    """
    n = len(rows)
    counts = {g: 0 for g in GATE_NAMES}
    for row in rows:
        t = gate_triggers(row["legs"], hi=6400)
        for g in GATE_NAMES:
            counts[g] += 1 if t[g] else 0
    return {"transition": "400->6400", "n_rows": n,
            "trigger_counts": counts,
            "trigger_rates": {g: (c / n if n else None) for g, c in counts.items()},
            "is_causal_evidence": False}


def by_stratum_summary(rows: Sequence[Dict[str, Any]], gate_name: str
                       ) -> Dict[str, Any]:
    """Overall plus late / flat-policy / near-even. Section 7 creates NO
    per-stratum acceptance gate, so the strata carry counts only."""
    def strip(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in d.items() if k != "verdict"}

    out: Dict[str, Any] = {"overall": calibrate_gate(rows, gate_name)}
    for name, pred in (("late", lambda r: r["phase"] == "late"),
                       ("flat_policy", lambda r: r["flat_policy"]),
                       ("near_even", lambda r: r["near_even"])):
        subset = [r for r in rows if pred(r)]
        out[name] = strip(calibrate_gate(subset, gate_name)) if subset else {
            "n_rows": 0, "convergent_rate": None}
    return out
