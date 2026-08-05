"""Atlas Read-out A -- design section 6 and amendment 6a, FROZEN.

Consumes Task 0's FROZEN CAPTURES, never a live root: the ladder mutates that
root through all four legs, so a later read describes the 6,400 tree.

Pure: every input is a plain dict, so this qualifies on synthetic rows.
"""
from __future__ import annotations

import math
import random
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

FEATURE_NAMES: Tuple[str, ...] = (
    "one_visit_backup_share",
    "depth3plus_backup_fraction",
    "leader_visit_margin",
    "root_policy_entropy",
    "leader_breadth",
)


def collect_features(capture_start: Dict[str, Any],
                     capture_boundary: Dict[str, Any],
                     n_actual: int) -> Dict[str, Optional[float]]:
    """The five frozen features. Undefined -> None, never 0.0."""
    delta = capture_boundary["D3"] - capture_start["D3"]
    if delta < 0 or delta > max(n_actual, 0):
        raise ValueError(
            f"backup accounting invariant violated: D3 delta {delta} outside "
            f"[0, {n_actual}]; the row must fail rather than be recorded")

    n_vis = capture_boundary["n_visited_children"]
    top = capture_boundary["top_child_visits"]
    second = capture_boundary["second_child_visits"]
    total = capture_boundary["total_child_visits"]
    return {
        "one_visit_backup_share": ((capture_boundary["one_visit_children"] / n_vis)
                                   if n_vis else None),
        "depth3plus_backup_fraction": ((delta / n_actual) if n_actual else None),
        "leader_visit_margin": (((top - second) / total)
                                if (top is not None and second is not None
                                    and total) else None),
        "root_policy_entropy": capture_boundary["policy_entropy"],
        "leader_breadth": capture_boundary["leader_breadth"],
    }


LABEL_TO_Y = {"misleading": 1, "stable_negative": 0}

MIN_VALIDATION_MISLEADING = 20
MIN_VALIDATION_STABLE_NEGATIVE = 25
AUC_BAR = 0.75
AUC_LOWER_BOUND_BAR = 0.60
MAX_FLAG_RATE = 0.25
MIN_PRECISION = 0.60
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260804


def prepare_rows(rows: Sequence[Dict[str, Any]],
                 feature_key: str = "features") -> Dict[str, Any]:
    """String labels -> numeric, ineligible classes dropped, missing-feature
    rows REJECTED and counted (section 6a).

    `feature_key` selects which frozen capture to read, so the SAME pipeline
    serves both feature sets. `kept_indices` is what lets the dual pipeline
    report whether the two sets ran on the same rows.
    """
    feats, y, kept = [], [], []
    dropped = rejected = 0
    for i, r in enumerate(rows):
        if r["label"] not in LABEL_TO_Y:
            dropped += 1                       # ambiguous / no_stable_reference
            continue
        # A MISSING capture rejects the row exactly like a missing feature:
        # `or {}` makes every feature None rather than raising a KeyError that a
        # caller might be tempted to catch and default.
        f = r.get(feature_key) or {}
        if any(f.get(k) is None for k in FEATURE_NAMES):
            rejected += 1
            continue
        feats.append(f)
        y.append(LABEL_TO_Y[r["label"]])
        kept.append(i)
    return {"features": feats, "y": y, "kept_indices": kept,
            "dropped_ineligible": dropped, "rejected_missing_features": rejected}


def standardize(rows, feature_names: Sequence[str] = FEATURE_NAMES,
                stats: Optional[Dict[str, Tuple[float, float]]] = None):
    """Z-score; `stats` learned on DISCOVERY only. A missing feature raises --
    imputing the mean would fabricate a maximally-uninformative observation."""
    for i, r in enumerate(rows):
        missing = [f for f in feature_names if r.get(f) is None]
        if missing:
            raise ValueError(f"row {i} is missing features {missing}; "
                             f"rows with undefined features are rejected")
    if stats is None:
        stats = {}
        for f in feature_names:
            vals = [r[f] for r in rows]
            mu = statistics.fmean(vals) if vals else 0.0
            sd = statistics.pstdev(vals) if len(vals) > 1 else 1.0
            stats[f] = (mu, sd if sd else 1.0)
    X = [[(r[f] - stats[f][0]) / stats[f][1] for f in feature_names]
         for r in rows]
    return X, stats


def fit_ridge_logistic(X, y, l2: float = 1.0, iters: int = 2000,
                       lr: float = 0.1) -> Dict[str, Any]:
    """Frozen hyperparameters (section 6a). stdlib only; no numpy or scipy."""
    n_f = len(X[0]) if X else 0
    w, b, n = [0.0] * n_f, 0.0, (len(X) or 1)
    for _ in range(iters):
        gw, gb = [0.0] * n_f, 0.0
        for xi, yi in zip(X, y):
            z = b + sum(wj * xj for wj, xj in zip(w, xi))
            p = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))
            err = p - yi
            for j in range(n_f):
                gw[j] += err * xi[j]
            gb += err
        w = [wj - lr * (gw[j] / n + l2 * wj / n) for j, wj in enumerate(w)]
        b -= lr * gb / n

    def predict(x):
        z = b + sum(wj * xj for wj, xj in zip(w, x))
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))

    return {"w": w, "b": b, "predict": predict}


def auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Rank AUC, ties at 0.5. None when a class is absent -- never a defaulted
    0.5, which would look like a real chance result."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > q else 0.5 if p == q else 0.0)
               for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def bootstrap_auc_lower_bound(scores, labels, seed: int = BOOTSTRAP_SEED,
                              replicates: int = BOOTSTRAP_REPLICATES,
                              alpha: float = 0.05) -> Optional[float]:
    if auc(scores, labels) is None:
        return None
    rng = random.Random(seed)
    n = len(scores)
    vals = []
    for _ in range(replicates):
        idx = [rng.randrange(n) for _ in range(n)]
        a = auc([scores[i] for i in idx], [labels[i] for i in idx])
        if a is not None:
            vals.append(a)
    if not vals:
        return None
    vals.sort()
    return vals[int(alpha * len(vals))]


def evaluate_detector(discovery: Sequence[Dict[str, Any]],
                      validation: Sequence[Dict[str, Any]],
                      seed: int = BOOTSTRAP_SEED,
                      replicates: int = BOOTSTRAP_REPLICATES,
                      feature_key: str = "features") -> Dict[str, Any]:
    """Section 6's frozen bars. Fails CLOSED, and capacity is checked AFTER
    missing-feature rejection.

    `seed` and `replicates` default to the FROZEN values and are parameters only
    so CPU tests can run a cheap bootstrap; nothing on the measurement path
    passes either. `feature_key` is what makes this the same pipeline for both
    feature sets rather than a second implementation of it.
    """
    d = prepare_rows(discovery, feature_key)
    v = prepare_rows(validation, feature_key)
    v_pos, v_neg = v["y"].count(1), v["y"].count(0)
    base = {"feature_set": feature_key,
            "n_misleading": v_pos, "n_stable_negative": v_neg,
            "rejected_missing_features": v["rejected_missing_features"],
            "dropped_ineligible": v["dropped_ineligible"],
            "auc": None, "auc_lower_bound": None}
    if v_pos < MIN_VALIDATION_MISLEADING or v_neg < MIN_VALIDATION_STABLE_NEGATIVE:
        return {**base, "verdict": "INSUFFICIENT_CLASSES",
                "reason": "validation split cannot support its own gate"}
    if d["y"].count(1) == 0 or d["y"].count(0) == 0:
        return {**base, "verdict": "INSUFFICIENT_DISCOVERY_CLASSES",
                "reason": "cannot fit with a single discovery class"}

    Xd, stats = standardize(d["features"])
    model = fit_ridge_logistic(Xd, d["y"])
    Xv, _ = standardize(v["features"], stats=stats)
    sv = [model["predict"](x) for x in Xv]

    a = auc(sv, v["y"])
    lb = bootstrap_auc_lower_bound(sv, v["y"], seed=seed, replicates=replicates)
    sd = sorted((model["predict"](x) for x in Xd), reverse=True)
    thr = sd[max(0, int(MAX_FLAG_RATE * len(sd)) - 1)] if sd else 1.0
    flagged = [(s, y) for s, y in zip(sv, v["y"]) if s >= thr]
    flag_rate = len(flagged) / len(sv) if sv else None
    precision = (sum(y for _s, y in flagged) / len(flagged)) if flagged else None

    passed = (a is not None and a >= AUC_BAR and lb is not None
              and lb >= AUC_LOWER_BOUND_BAR and flag_rate is not None
              and flag_rate <= MAX_FLAG_RATE and precision is not None
              and precision >= MIN_PRECISION)
    return {**base, "verdict": "PASS" if passed else "FAIL", "auc": a,
            "auc_lower_bound": lb, "flag_rate": flag_rate,
            "precision": precision, "threshold": thr}


def deployability(remaining_values: Sequence[int],
                  strata: Optional[Dict[str, Sequence[int]]] = None
                  ) -> Dict[str, Any]:
    """Section 6: remaining == 0 is non-actionable; a MEDIAN of zero fails the
    controller-deployability claim. Strata are REPORTED, never gated."""
    def summarize(vals: Sequence[int]) -> Dict[str, Any]:
        if not vals:
            return {"n": 0, "median_remaining": None,
                    "zero_budget_fraction": None, "quartiles": None}
        s = sorted(vals)
        return {"n": len(s), "median_remaining": statistics.median(s),
                "zero_budget_fraction": sum(1 for x in s if x == 0) / len(s),
                "quartiles": (statistics.quantiles(s, n=4, method="inclusive")
                              if len(s) >= 2 else None)}

    overall = summarize(remaining_values)
    med = overall["median_remaining"]
    verdict = ("NO_ROWS" if med is None
               else "NOT_DEPLOYABLE" if med == 0 else "DEPLOYABLE")
    return {**overall, "verdict": verdict,
            "by_stratum": {k: summarize(v) for k, v in (strata or {}).items()}}


# -- section 6a: Read-out A runs on BOTH feature sets -------------------------

FEATURE_SETS = ("features_at_boundary", "features_at_400")
AUTHORITATIVE_FEATURE_SET = "features_at_boundary"
INSUFFICIENCY_VERDICTS = ("INSUFFICIENT_CLASSES",
                          "INSUFFICIENT_DISCOVERY_CLASSES")


def evaluate_detector_both(discovery: Sequence[Dict[str, Any]],
                           validation: Sequence[Dict[str, Any]],
                           seed: int = BOOTSTRAP_SEED,
                           replicates: int = BOOTSTRAP_REPLICATES
                           ) -> Dict[str, Any]:
    """The IDENTICAL pipeline on both frozen feature sets (amendment 6a).

    The boundary remains AUTHORITATIVE -- it is the only instant at which a
    controller could still act on the result.

        boundary INSUFFICIENT_*                                -> as itself
        boundary PASS                                          -> PASS
        boundary FAIL, B=400 PASS, boundary rejected NO rows    -> LATE_ONLY_SEPARATION
        boundary FAIL, B=400 PASS, boundary rejected ANY row    -> FAIL + blocked
        boundary FAIL, B=400 not PASS                          -> FAIL

    LATE_ONLY_SEPARATION means the information exists but arrives too late to
    allocate the remaining budget -- section 6's stated FAILURE condition, not a
    success.
    """
    per = {fs: evaluate_detector(discovery, validation, seed=seed,
                                 replicates=replicates, feature_key=fs)
           for fs in FEATURE_SETS}
    auth = per[AUTHORITATIVE_FEATURE_SET]["verdict"]
    late = per["features_at_400"]["verdict"]

    # Amendment 4 lists a MISSING-FEATURE REJECTION alongside the two
    # insufficiency verdicts as something that cannot establish lateness.
    # Rejections shrink the boundary sample and a smaller sample is MORE likely
    # to miss the bars, so a rejection is exactly what could manufacture the
    # FAIL half of a lateness finding. Counted in BOTH splits: discovery
    # rejections change the fitted model, validation rejections change what it
    # is scored on.
    boundary_rejections = sum(
        prepare_rows(split, feature_key=AUTHORITATIVE_FEATURE_SET
                     )["rejected_missing_features"]
        for split in (discovery, validation))

    blocked = None
    if auth in INSUFFICIENCY_VERDICTS:
        # An ABSENCE of evidence, never evidence about timing.
        verdict = auth
    elif auth == "PASS":
        verdict = "PASS"
    elif late == "PASS" and boundary_rejections == 0:
        verdict = "LATE_ONLY_SEPARATION"          # boundary FAIL and 400 PASS
    elif late == "PASS":
        # Reported as the boundary's OWN result, inside the frozen verdict
        # vocabulary, with the reason recorded rather than silently dropped.
        verdict, blocked = "FAIL", "boundary_missing_feature_rejections"
    else:
        verdict = "FAIL"

    def overlap(split):
        kept = {fs: set(prepare_rows(split, feature_key=fs)["kept_indices"])
                for fs in FEATURE_SETS}
        a, b = kept[FEATURE_SETS[0]], kept[FEATURE_SETS[1]]
        return {"n_common": len(a & b), "identical": a == b}

    return {
        "verdict": verdict,
        "authoritative": AUTHORITATIVE_FEATURE_SET,
        "per_feature_set": per,
        "boundary_rejections": boundary_rejections,
        "lateness_blocked_by": blocked,
        # REPORTED, never gated. BOTH splits: a validation-only overlap would
        # call two models identical when they were fitted on different rows.
        "row_overlap": {"discovery": overlap(discovery),
                        "validation": overlap(validation)},
    }
