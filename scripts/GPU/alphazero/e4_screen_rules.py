"""The E4 endpoint screen's FROZEN decision rules, as committed code.

These are the rules preregistered and self-tested by the E4 preflight (attempt 4,
preregistration sha256 d41da183f7503c539d1809f4f233c3f956ec6959a236a187c5f142e839438971).
They lived only in the preflight's scratch harness, which would have forced the
execution runner to re-implement them, with nothing preventing drift. Extracted
here VERBATIM from that qualified file so there is ONE implementation, with
tests/test_e4_screen_rules.py pinning it against the same control table the
preflight self-test used.

Nothing here plays, scores or seeds anything: it classifies a result vector.
"""

DECISIONS = ("SATURATED_STRONG", "SATURATED_WEAK", "IN_BAND", "INCOMPLETE")


def per_endpoint_decision(score, played, n, band, cap_terminations=0):
    """T1j's decision at ONE endpoint. `score` counts draws as 0.5.

    A draw is always a ply-cap termination -- TwixT cannot draw by rule -- so an
    endpoint where most games hit the cap is INCOMPLETE, not scored.
    """
    lo, hi = band[0] * n, band[1] * n
    if cap_terminations * 2 > n:
        return "INCOMPLETE"
    if played < n:
        # only a FORCED IN_BAND may stop early -- and forcing it takes more than
        # ruling out score saturation; see early_in_band_forced
        if early_in_band_forced(score, played, n, band, cap_terminations):
            return "IN_BAND"
        return "INCOMPLETE"
    if score >= hi:
        return "SATURATED_STRONG"
    if score <= lo:
        return "SATURATED_WEAK"
    return "IN_BAND"


def saturation_reachable(score, played, n, band):
    """Can EITHER SCORE saturation verdict still be reached from here?"""
    lo, hi = band[0] * n, band[1] * n
    strong_possible = score + (n - played) >= hi
    weak_possible = score <= lo
    return strong_possible or weak_possible


def cap_incompleteness_reachable(played, n, cap_terminations):
    """Could this endpoint still end INCOMPLETE through ply-cap terminations?

    The cap-saturation abort fires at cap_terminations * 2 > n. Every unplayed
    game could still hit the cap, so the abort remains reachable while
    cap_terminations + (n - played) > n // 2.
    """
    return cap_terminations + (n - played) > n // 2


def early_in_band_forced(score, played, n, band, cap_terminations=0):
    """Is IN_BAND the ONLY outcome still reachable?

    Attempt 2 checked score saturation alone and concluded IN_BAND was forced
    after one win and one loss. It is NOT: with 14 games unplayed, nine of them
    could still terminate at the ply cap, firing the cap-saturation abort and
    making the endpoint INCOMPLETE. Both routes out of IN_BAND must be closed.
    """
    if played >= n:
        return False                       # not an EARLY stop
    return (not saturation_reachable(score, played, n, band)
            and not cap_incompleteness_reachable(played, n, cap_terminations))


def earliest_early_stop(n, band):
    """The first game at which an IN_BAND early stop can possibly be forced.

    Reported rather than asserted: the number follows from n and the band, and
    stating it wrongly is exactly the defect this function exists to prevent.
    """
    for played in range(1, n + 1):
        for half in range(0, 2 * played + 1):
            if early_in_band_forced(half / 2.0, played, n, band, 0):
                return played, half / 2.0
    return None, None


JOINT = {
    ("SATURATED_STRONG", "SATURATED_STRONG"): "T1J_TOO_STRONG",
    ("SATURATED_WEAK", "SATURATED_WEAK"): "T1J_TOO_WEAK",
    ("SATURATED_WEAK", "SATURATED_STRONG"): "BRACKETED",
    ("SATURATED_STRONG", "SATURATED_WEAK"): "BRACKETED",
}


def classify_joint(decision_weak, decision_strong):
    """TOTAL over all 16 combinations of DECISIONS; asserts its own totality."""
    for d in (decision_weak, decision_strong):
        if d not in DECISIONS:
            raise ValueError(f"unknown endpoint decision {d!r}")
    if "INCOMPLETE" in (decision_weak, decision_strong):
        return "INCONCLUSIVE"
    if "IN_BAND" in (decision_weak, decision_strong):
        return "IN_BAND"
    return JOINT[(decision_weak, decision_strong)]


LARGER_MATCH_PERMITTED = ("IN_BAND", "BRACKETED", "INCONCLUSIVE")


def joint_truth_table():
    """Every combination, so the frozen plan carries the whole mapping."""
    return {f"{a}|{b}": classify_joint(a, b) for a in DECISIONS for b in DECISIONS}
