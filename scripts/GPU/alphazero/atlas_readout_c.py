"""Atlas Read-out C -- design section 8 and amendments 6a / 4, FROZEN.

Counterfactual COVERAGE analysis. It cannot prove progressive widening would
improve search, because applying widening changes the later tree.

Everything here is evaluated at TWO instants -- the batch-safe boundary and
nominal B = 400 -- because those are the horizons a widening rule would actually
see. The 6,400 tree is never a retention horizon, and no function here can reach
one: the producer does not emit deep-rung visit counts at all.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .selection_tracer import (
    DEPTH_BUCKETS, MEANINGFUL_INTERVENTION_FIRST_TOUCH_RATE, WIDENING_SHAPES,
    k_of_n,
)
# DEPTH_NAMES ("root", "reply", "two_ply") is the reference line's own vocabulary
# and belongs with the producer that emits it. Imported rather than restated:
# a second copy of a frozen constant is how the two drift apart.
from .warm_prefix_replay import DEPTH_NAMES

RETENTION_ROOT_BAR = 0.95
RETENTION_DEPTH1_BAR = 0.90
MISLEADING_INTERVENTION_BAR = 0.50
STABLE_INTERVENTION_CEILING = 0.25
FLAT_ENTROPY_BAR = 0.90
FLAT_TOP_PRIOR_BAR = 0.025

STRATA = ("late", "near_even", "root_flat",
          "locally_flat_depth1", "locally_flat_depth2")

# Amendment 4: retention is judged at BOTH instants, and the B=400 intervention
# drives the feasibility bars while the boundary one is reported.
INSTANTS = ("at_boundary", "at_400")
GATING_INSTANT = "at_400"

# Section 8's floors are about "stable deep root moves" and "stable depth-1
# replies", so only rows that HAVE a stable deep reference may contribute to
# them. A `no_stable_reference` row's two deep rungs disagree; its reference
# line is not a stable deep move and cannot be evidence of retaining one.
#
# An explicit ALLOW-list, never `label != "no_stable_reference"`: a label added
# later would silently be admitted by the deny-list form.
STABLE_REFERENCE_LABELS = ("misleading", "stable_negative", "ambiguous")

# The rates the validation verdict requires. `descendant_retention` is NOT one:
# it breaks exact ties in shape selection and is not a bar, so an undefined one
# must not turn a passing aggregate INCONCLUSIVE.
REQUIRED_RATES = ("root_retention", "depth1_retention",
                  "misleading_intervention", "stable_intervention")
_BARS = {"root_retention": (RETENTION_ROOT_BAR, "floor"),
         "depth1_retention": (RETENTION_DEPTH1_BAR, "floor"),
         "misleading_intervention": (MISLEADING_INTERVENTION_BAR, "floor"),
         "stable_intervention": (STABLE_INTERVENTION_CEILING, "ceiling")}


def static_retention(root_priors: Dict[int, float],
                     required_moves: Sequence[int], n_at_selection: int,
                     shape: Tuple[str, float, float]) -> Dict[str, Any]:
    """Would the moves stable deeper search requires have been admitted?

    `n_at_selection` is the parent's EFFECTIVE completed visit count -- at a warm
    root that is I + N_actual, never the nominal 320 (amendment 6a). Retention is
    evaluated under K(n), the narrower conservative set, so a pass is safe.
    """
    _name, c, alpha = shape
    n_legal = len(root_priors)
    k = k_of_n(n_at_selection, c, alpha, n_legal)
    order = sorted(root_priors.items(), key=lambda kv: (-kv[1], kv[0]))
    rank = {mv: i + 1 for i, (mv, _p) in enumerate(order)}
    if not required_moves:
        return {"retained": 0, "required": 0, "rate": None, "k": k}
    retained = sum(1 for mv in required_moves if rank.get(mv, n_legal + 1) <= k)
    return {"retained": retained, "required": len(required_moves),
            "rate": retained / len(required_moves), "k": k}


def edge_retention(edge: Dict[str, Any],
                   parent_visits: Dict[Tuple[int, ...], int],
                   shape: Tuple[str, float, float]) -> Dict[str, Any]:
    """Would this required edge have been admitted AT THIS INSTANT?

    `n` is the parent's effective completed visit count read from the instant's
    OWN map -- `I + N_actual` at the boundary, `I + 400` after leg 1; never the
    nominal 320 and never the final 6,400 tree (amendments 6a and 4).

    A path ABSENT from the map has ZERO visits. `K(0) = min(n_legal, max(1, 0))
    = 1` through the `max(1, ...)` floor, so rank 1 is still admitted there --
    a real admission, not a vacuous one.
    """
    n = parent_visits.get(edge["parent_path"], 0)
    r = static_retention(edge["parent_priors"], [edge["move"]], n, shape)
    return {"retained": r["retained"] == 1, "k": r["k"], "n": n,
            "depth": edge["depth"]}


def intervention_from_snapshots(snapshots: Dict[str, Any], shape_key: str,
                                *, instant: str) -> Dict[str, Any]:
    """Meaningful intervention with the DIRECTIONAL lag bound, at ONE instant.

    The lag is conservative for retention and ANTI-conservative for
    intervention, so the threshold must also pass under K(n+14) -- a counter the
    tracer PRODUCES, never a caller-supplied number. Passing only under K(n) is
    INCONCLUSIVE, not a pass.

    `instant` is keyword-only with NO default. Amendment 4 gates on B=400 and
    reports the boundary separately, and a defaulted instant is exactly how a
    caller silently gets the other one.
    """
    snap = snapshots.get(instant)
    if snap is None:
        # A row whose boundary never fired has no snapshot at that instant.
        # That is not a rate of zero and not an intervention of False.
        return {"first_touch_outside_rate": None, "lagged_rate": None,
                "meaningfully_affected": None, "verdict": "NO_SNAPSHOT"}
    cell = snap["by_shape"][shape_key]["overall"]
    ft = cell["first_touch_events"]
    if not ft:
        return {"first_touch_outside_rate": None, "lagged_rate": None,
                "meaningfully_affected": None, "verdict": "NO_EVENTS"}
    rate = cell["first_touch_outside_events"] / ft
    lagged = cell["lagged_first_touch_outside_events"] / ft
    ok = rate >= MEANINGFUL_INTERVENTION_FIRST_TOUCH_RATE
    lagged_ok = lagged >= MEANINGFUL_INTERVENTION_FIRST_TOUCH_RATE
    if ok and lagged_ok:
        verdict, affected = "OK", True
    elif ok:
        # None, never False: undecided is not a measured negative.
        verdict, affected = "INCONCLUSIVE", None
    else:
        verdict, affected = "OK", False
    return {"first_touch_outside_rate": rate, "lagged_rate": lagged,
            "meaningfully_affected": affected, "verdict": verdict}


def classify_strata(row: Dict[str, Any]) -> Set[str]:
    """Frozen strata from the ROW. Flat-policy status is recomputed LOCALLY
    along the reference line, not inherited from the root."""
    s: Set[str] = set()
    if row.get("phase") == "late":
        s.add("late")
    if row.get("near_even"):
        s.add("near_even")
    if row.get("flat_policy"):
        s.add("root_flat")
    return s


def classify_edge_strata(edge: Dict[str, Any]) -> Set[str]:
    """Locally-flat strata are EDGE-LEVEL, not row-level.

    Under the union-of-two-deep-lines rule a single row can contain both flat
    and non-flat reference parents. A row-level "any parent is flat" flag would
    pool their retention into one number and hide exactly the contrast the
    stratum exists to expose. Each deduplicated required edge is classified
    using ITS OWN parent priors.
    """
    s: Set[str] = set()
    priors = edge.get("parent_priors")
    if not priors or not is_flat(priors):
        return s
    # EXPLICIT branches. An `else depth2` fallthrough would map a flat ROOT edge
    # (depth 0) -- and any malformed or missing depth -- to locally_flat_depth2,
    # inventing a stratum membership the edge does not have.
    depth = edge.get("depth")
    if depth == 1:
        s.add("locally_flat_depth1")
    elif depth == 2:
        s.add("locally_flat_depth2")
    return s


def is_flat(priors: Dict[int, float]) -> bool:
    """The FROZEN flat-policy definition: normalized entropy >= 0.90 AND top
    prior <= 0.025. Checking only the top prior misclassifies a concentrated
    low-top distribution as flat -- both halves are required.

    PUBLIC because the root-level `flat_policy` row fact applies the same
    predicate (atlas_row_facts). One frozen definition, one implementation:
    a second copy is how the root stratum and the local strata drift apart.
    """
    vals = [p for p in priors.values() if p > 0]
    if len(vals) < 2:
        return False
    s = sum(vals)
    norm = [v / s for v in vals]
    entropy = (-sum(q * math.log(q) for q in norm)) / math.log(len(priors))
    return entropy >= FLAT_ENTROPY_BAR and max(norm) <= FLAT_TOP_PRIOR_BAR


def _rate(num: float, den: float) -> Optional[float]:
    """Zero denominator -> None. Never 0.0, never False."""
    return (num / den) if den else None


def _pair() -> Dict[str, int]:
    return {"retained": 0, "required": 0}


def _empty_counters() -> Dict[str, Any]:
    def z():
        return {"eligible_events": 0, "outside_events": 0,
                "first_touch_events": 0, "first_touch_outside_events": 0,
                "lagged_first_touch_outside_events": 0,
                "excluded_prior_mass": 0.0}
    return {**z(), "by_depth": {b: z() for b in DEPTH_BUCKETS},
            "forced_root_bypass_events": 0,
            "forced_root_bypass_outside_events": 0,
            "within_forced_events": 0, "missing_snapshots": 0}


def _pool_counters(acc: Dict[str, Any], snap: Optional[Dict[str, Any]],
                   shape_key: str) -> None:
    """Section 8's online aggregates, pooled across the cohort.

    SUM numerators and denominators; never average per-row rates. A mean of
    means would weigh a 10-event row the same as a 990-event one.
    """
    if snap is None:
        acc["missing_snapshots"] += 1
        return
    block = snap["by_shape"][shape_key]
    for key in ("overall",) + DEPTH_BUCKETS:
        cell = block[key]
        target = acc if key == "overall" else acc["by_depth"][key]
        for f in ("eligible_events", "outside_events", "first_touch_events",
                  "first_touch_outside_events",
                  "lagged_first_touch_outside_events", "excluded_prior_mass"):
            target[f] += cell[f]
    # Forced-root bypasses are reported SEPARATELY and never enter the primary
    # intervention denominator (design section 8).
    acc["forced_root_bypass_events"] += block["forced_root_bypass_events"]
    acc["forced_root_bypass_outside_events"] += block[
        "forced_root_bypass_outside_events"]
    acc["within_forced_events"] += snap["within_forced_events"]


def _finalize_counters(acc: Dict[str, Any]) -> Dict[str, Any]:
    def rates(c):
        c["outside_rate"] = _rate(c["outside_events"], c["eligible_events"])
        c["first_touch_outside_rate"] = _rate(c["first_touch_outside_events"],
                                              c["first_touch_events"])
        c["lagged_first_touch_outside_rate"] = _rate(
            c["lagged_first_touch_outside_events"], c["first_touch_events"])
        # EVENT-WEIGHTED: total mass outside top-K over total eligible events.
        c["mean_excluded_prior_mass"] = _rate(c["excluded_prior_mass"],
                                              c["eligible_events"])
    rates(acc)
    for cell in acc["by_depth"].values():
        rates(cell)
    acc["forced_root_bypass_outside_rate"] = _rate(
        acc["forced_root_bypass_outside_events"],
        acc["forced_root_bypass_events"])
    return acc


def aggregate_shape(rows: Sequence[Dict[str, Any]],
                    shape: Tuple[str, float, float]) -> Dict[str, Any]:
    """Fold per-row results into the rates section 8 gates, AT BOTH INSTANTS.

    Retention runs over the DEDUPLICATED UNION of required edges from both deep
    lines (amendment 4), each evaluated against that instant's own parent-visit
    map. INCONCLUSIVE rows are excluded from the intervention denominator and
    counted separately -- folding them in as either outcome would invent a
    measurement. If a denominator empties, the rate is None and the shape CANNOT
    pass.
    """
    name = shape[0]
    acc = {inst: {"by_role": {r: _pair() for r in DEPTH_NAMES},
                  "by_stratum": {s: _pair() for s in STRATA},
                  "retention_rows": 0,
                  "mis_num": 0, "mis_den": 0, "stab_num": 0, "stab_den": 0,
                  "inconclusive": 0}
           for inst in INSTANTS}
    counters = {inst: _empty_counters() for inst in INSTANTS}
    agreement = {d: {"agree": 0, "disagree": 0,
                     "single_line": 0, "absent_both": 0}
                 for d in DEPTH_NAMES}
    rows_without_stable_reference = 0

    for row in rows:
        snaps = row["snapshots"]
        merged = snaps["reference_lines"]["merged"]
        row_strata = classify_strata(row)
        # Agreement is reported over EVERY row, including unstable ones: a row
        # can be no_stable_reference because of the value gap or the top-two
        # margin while its deep root moves agree, so agreement and stability
        # are different facts and pooling all rows is the honest report.
        for depth_name, a in merged["agreement"].items():
            agreement[depth_name][a["state"]] += 1

        # Section 8's floors concern STABLE deep moves. A row whose deep rungs
        # never agreed has no stable deep move, so it contributes no required
        # edges -- but its selection events still count, because those describe
        # what widening would have done regardless of label.
        stable = row["label"] in STABLE_REFERENCE_LABELS
        if not stable:
            rows_without_stable_reference += 1

        for inst in INSTANTS:
            a = acc[inst]
            visits = (snaps.get("parent_visits") or {}).get(inst) or {}
            if stable:
                a["retention_rows"] += 1
                for edge in merged["required_edges"]:
                    if edge["depth"] >= len(DEPTH_NAMES):
                        continue                 # beyond the two-ply horizon
                    res = edge_retention(edge, visits, shape)
                    # Flat-policy status is recomputed LOCALLY per edge; row
                    # strata apply to every edge of that row.
                    buckets = [a["by_role"][DEPTH_NAMES[edge["depth"]]]]
                    buckets += [a["by_stratum"][s]
                                for s in row_strata | classify_edge_strata(edge)]
                    for bucket in buckets:
                        bucket["required"] += 1
                        bucket["retained"] += 1 if res["retained"] else 0

            iv = intervention_from_snapshots(snaps, name, instant=inst)
            if iv["meaningfully_affected"] is None:
                a["inconclusive"] += 1
            elif row["label"] == "misleading":
                a["mis_den"] += 1
                a["mis_num"] += 1 if iv["meaningfully_affected"] else 0
            elif row["label"] == "stable_negative":
                a["stab_den"] += 1
                a["stab_num"] += 1 if iv["meaningfully_affected"] else 0

            _pool_counters(counters[inst], snaps.get(inst), name)

    def with_rate(p):
        return {**p, "rate": _rate(p["retained"], p["required"])}

    instants: Dict[str, Any] = {}
    for inst in INSTANTS:
        a = acc[inst]
        roles = {r: with_rate(p) for r, p in a["by_role"].items()}
        instants[inst] = {
            "by_role": roles,
            "root_retention": roles["root"]["rate"],
            "depth1_retention": roles["reply"]["rate"],
            "descendant_retention": roles["two_ply"]["rate"],
            "by_stratum": {s: with_rate(p) for s, p in a["by_stratum"].items()},
            # How many rows the retention bars actually rest on.
            "retention_rows": a["retention_rows"],
            "misleading_intervention": _rate(a["mis_num"], a["mis_den"]),
            "stable_intervention": _rate(a["stab_num"], a["stab_den"]),
            "misleading_denominator": a["mis_den"],
            "stable_denominator": a["stab_den"],
            "inconclusive": a["inconclusive"],
        }

    def worst(field):
        """Amendment 4: the floors must pass at BOTH instants, and
        `min(a, b) >= bar` is exactly that. None if either is undefined, since
        an undefined rate is not a satisfied bar."""
        vals = [instants[i][field] for i in INSTANTS]
        return None if any(v is None for v in vals) else min(vals)

    gated = instants[GATING_INSTANT]
    return {
        "shape": name,
        "gated_on": GATING_INSTANT,
        "instants": instants,
        "counters": {i: _finalize_counters(counters[i]) for i in INSTANTS},
        # Reported, never gated (amendment 4). single_line and absent_both are
        # DIFFERENT missingness states and neither enters the denominator.
        "agreement": {d: {**v, "agree_rate": _rate(v["agree"],
                                                   v["agree"] + v["disagree"])}
                      for d, v in agreement.items()},
        # The retention bars rest only on stable-reference-eligible rows; this
        # says how many were set aside, so a floor computed over three rows is
        # not mistaken for one computed over the corpus.
        "rows_without_stable_reference": rows_without_stable_reference,
        "retention_rows": instants[GATING_INSTANT]["retention_rows"],
        # Hoisted for select_shape / validation_verdict: retention is the WORSE
        # instant, intervention is the B=400 number.
        "root_retention": worst("root_retention"),
        "depth1_retention": worst("depth1_retention"),
        "descendant_retention": worst("descendant_retention"),
        "misleading_intervention": gated["misleading_intervention"],
        "stable_intervention": gated["stable_intervention"],
        "misleading_denominator": gated["misleading_denominator"],
        "stable_denominator": gated["stable_denominator"],
        "inconclusive": gated["inconclusive"],
    }


def validation_verdict(aggregate: Dict[str, Any]) -> Dict[str, Any]:
    """The frozen three-way precedence (amendment 6a):

        1. FAIL          -- any DEFINED rate misses its bar
        2. INCONCLUSIVE  -- otherwise, any required rate is UNDEFINED
        3. PASS          -- otherwise

    ORDERED, because a result can hold both a defined miss and an undefined rate
    and would otherwise satisfy two verdicts at once. A defined miss is evidence
    and outranks a gap in the evidence; an undefined rate is not a satisfied bar
    and is not a measured miss either.
    """
    failed: List[str] = []
    undefined: List[str] = []
    for rate_name in REQUIRED_RATES:
        value = aggregate.get(rate_name)
        bar, kind = _BARS[rate_name]
        if value is None:
            undefined.append(rate_name)
        elif (value < bar) if kind == "floor" else (value > bar):
            failed.append(rate_name)
    verdict = "FAIL" if failed else ("INCONCLUSIVE" if undefined else "PASS")
    return {"verdict": verdict, "failed": failed, "undefined": undefined}


def select_shape(per_shape: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Frozen LEXICOGRAPHIC order: retention floors -> stable-intervention
    ceiling -> higher misleading intervention -> tie on descendant retention.

    A None rate cannot pass: an undefined denominator is not a satisfied bar.
    """
    def passes(v):
        return (v.get("root_retention") is not None
                and v["root_retention"] >= RETENTION_ROOT_BAR
                and v.get("depth1_retention") is not None
                and v["depth1_retention"] >= RETENTION_DEPTH1_BAR
                and v.get("stable_intervention") is not None
                and v["stable_intervention"] <= STABLE_INTERVENTION_CEILING
                and v.get("misleading_intervention") is not None)

    survivors = {k: v for k, v in per_shape.items() if passes(v)}
    if not survivors:
        return {"selected": None, "verdict": "NO_SHAPE_PASSES",
                "considered": sorted(per_shape)}
    best = max(survivors, key=lambda k: (
        survivors[k]["misleading_intervention"],
        survivors[k].get("descendant_retention") or 0.0))
    return {"selected": best, "verdict": "OK", "survivors": sorted(survivors),
            "considered": sorted(per_shape)}


def select_on_discovery_validate_on_selected(
        discovery: Sequence[Dict[str, Any]],
        validation: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Choose on DISCOVERY, then evaluate ONLY the selected shape on VALIDATION.

    Evaluating both on validation would let a shape be chosen for looking good
    on the split that judges it.
    """
    disc = {s[0]: aggregate_shape(discovery, s) for s in WIDENING_SHAPES}
    chosen = select_shape(disc)
    if chosen["selected"] is None:
        # Nothing was validated, so there is no validation aggregate to judge.
        # NO_SHAPE_PASSES is a selection outcome, not an INCONCLUSIVE
        # validation, and must not be reported as one.
        return {**chosen, "selected_on": "discovery", "validated": {},
                "validation_verdict": None}
    shape = next(s for s in WIDENING_SHAPES if s[0] == chosen["selected"])
    validated = aggregate_shape(validation, shape)
    return {**chosen, "selected_on": "discovery",
            "discovery": disc,
            "validated": {chosen["selected"]: validated},
            # Amendment 6a: the validation aggregate is JUDGED, not merely
            # computed.
            "validation_verdict": validation_verdict(validated)}
