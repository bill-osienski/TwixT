"""H1 head-to-head viability screen: design, verdict and reporting. NO EXECUTION.

H1 asks ONE question the card states before any data exists: **is `calib020_0001`
worth targeted improvement against T1j?** L0 measured T1j at 0.5938 over 64 games
with a 95% Hoeffding interval of [0.4240, 0.7635]. Parity is inside that interval
AND SO IS 0.75, so L0 can exclude neither parity nor a rout. H1 is the same
protocol at 224 games, sized so that 0.75 falls outside the interval for any rate
near L0's estimate.

WHAT IS REUSED, AND WHY NOTHING IS RETYPED
------------------------------------------
Everything H1 shares with L0 is READ from `l0_match_rules` rather than restated:
the scoring rule, both intervals, the result contract, the ply cap, alpha, and the
outcome vocabulary. Restating them would create a second source that looks
identical until it isn't -- the failure this workstream keeps finding.

⚠ AN HONEST LIMIT ON THAT CLAIM. For a small integer, equality cannot distinguish
"read from L0" from "retyped as 8": CPython interns both, so no runtime test can
tell them apart. H1 therefore does not declare the shared scalars at all -- it
re-exports L0's objects, so a retype would have to add a constant AND change its
use sites. The values that actually matter, the eight openings and the reference
identity, are bound to the sha256-pinned source plan and to the task digest, and
those bindings ARE checkable.

WHAT H1 DOES NOT REUSE
----------------------
`match_report` is L0's, and reports L0's estimand over L0's 64 games. H1 has a
verdict L0 does not have, so it has its own reporter. `bind_results` IS shared,
after being parameterised by a REQUIRED `Design` -- see `l0_match_rules.Design`
for why that parameter has no default.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from . import l0_match_rules as L0

#: Shared with L0, RE-EXPORTED rather than restated. See the module docstring.
T1J_MDPLY = L0.T1J_MDPLY
PLY_CAP = L0.PLY_CAP
SCORE_T1J_WIN = L0.SCORE_T1J_WIN
SCORE_DRAW = L0.SCORE_DRAW
SCORE_T1J_LOSS = L0.SCORE_T1J_LOSS
Z_95 = L0.Z_95
ALPHA = L0.ALPHA
WINNERS = L0.WINNERS
TERMINAL_REASONS = L0.TERMINAL_REASONS
N_OPENINGS = L0.N_OPENINGS
N_ARMS = L0.N_ARMS

#: H1's OWN size. This is the only design number that differs from L0, and the
#: card justifies it: at n=224 the Hoeffding half-width is 0.0907, so 0.75 sits
#: outside the interval for any observed rate below 0.6593. n=128 would resolve
#: only below 0.6300 -- barely clear of L0's own point estimate of 0.5938.
N_REPS = 14
N_GAMES = N_OPENINGS * N_ARMS * N_REPS                  # 224

#: PAPER-RESERVED by the H1 card, DELIBERATELY UNREGISTERED: it appears in no
#: registry tuple.
#:
#: 🔴 CORRECTED CLAIM. This comment previously said "so a run is refused by the
#: seed barrier as well as by the gate". BOTH OF THOSE BARRIERS ARE ABSENT: H1
#: has no runner, so it has no gate, and nothing checks this block against
#: ACCOUNTED_SEED_INTERVALS -- `validate_task_executable` asks about consumed,
#: exposed and retired seeds, NOT accounted ones, and ACCEPTS an unregistered
#: H1 seed today. The protection was written as present fact while being purely
#: prospective, which is the exact defect class this workstream tracks.
#:
#: What is TRUE today: there is nothing to run, so there is nothing to refuse.
#: An H1 RUNNER PHASE MUST ADD BOTH -- a gate defaulting to False and a
#: registration precondition modelled on `d1_probe._check_seed_registration` --
#: and neither exists until it does.
#: Defined HERE rather than in the plan module -- L0 keeps its block in the plan,
#: but H1's abort rule has to be RUNNABLE, and the plan module imports these
#: rules, so the rules cannot import the block back.
H1_SEED_BLOCK = (202616000, 202616224)


def seed_is_outside_the_reserved_block(seed: Any) -> bool:
    """The seed abort rule AS A PREDICATE. Prose cannot be run, and a rule that
    is only prose is a rule nothing checks."""
    lo, hi = H1_SEED_BLOCK
    return not lo <= int(seed) < hi

#: THE PREDECLARED VIABILITY THRESHOLD, fixed before any H1 data exists.
#:
#: At a 0.75 score rate T1j wins three games in four. Converted ONCE, for sizing
#: only, that is ~191 Elo -- against interventions in this project's own history
#: that moved TENS of Elo (frozen-parent -23.5, parent-replay -47.2). A gap that
#: size is not what "targeted improvement" plausibly closes.
#:
#: ⚠ That Elo conversion sized the threshold and is a FORBIDDEN CLAIM in the
#: report. It says nothing about what happened.
VIABILITY_THRESHOLD = 0.75

#: Cap policy, inherited: caps never stop the match -- that would be an early
#: stop -- and the decision is made afterwards. A rate over mostly-unresolved
#: games measures the cap, not the players.
CAP_NO_RATE_THRESHOLD = N_GAMES // 2                    # 112; "more than half" is > this

#: There is no early stop, for the reason L0 recorded: the canonical screen
#: early-stopped at 7.0/8 = 0.875 and the full 64-game rate came back 0.594. An
#: early stop cannot bias a BAND decision and does bias a RATE -- and H1's verdict
#: is a rate question.
EARLY_STOP = None

#: Named here so a runner cannot quietly acquire an early stop by importing the
#: screen's rules instead. The H1 protocol names this function; it has one answer.
MUST_NEVER_BE_CALLED_ON_AN_H1_RUN = (
    "e4_screen_rules.may_stop_early",
    "e4_screen_rules.endpoint_decision",
    "e4_screen_rules.band_verdict",
    "e4_screen_rules.joint_verdict",
    "e4_screen_rules.should_continue",
)


def may_stop_early(*_args: Any, **_kwargs: Any) -> bool:
    """Always False. H1 plays all 224 games."""
    return False


#: INTEGRITY failures only. A COMPLETED GAME IS A RESULT, whoever wins -- the
#: distinction D1 destroyed by conflating FAIL with VOID, and the reason the
#: low-ply qualification had to exist. A loss is data; only the instrument can
#: VOID.
#: 🔴 COMPOSED, NOT APPENDED. The first version was `L0.L0_ABORT_RULES + (...)`,
#: which left L0's "any seed outside the reserved L0 block" active alongside
#: H1's -- and EVERY VALID H1 SEED IS OUTSIDE THE L0 BLOCK, so the rule set
#: aborted every game it governed. `L0.abort_rules` takes the block name, so H1
#: gets exactly one seed rule and it names H1's block.
H1_ABORT_RULES = L0.abort_rules("H1") + (
    "the whole-run wall-clock cap being exceeded",
)

#: NOT abort rules, named so they cannot be quietly reintroduced.
#:
#: 🔴 NOT `L0.NOT_ABORT_RULES`. Inherited whole, it imported two statements that
#: are FALSE OF H1 straight into the frozen artifact: that caps never stop an
#: "L0 match", and that the design "has no band to saturate". H1 HAS a band --
#: the 0.75 viability threshold -- and a cap-saturated no-rate branch past 112.
#: The PROHIBITIONS are identical in substance; only the reasons differ, and a
#: reason that describes the wrong match is a false statement in the protocol.
NOT_ABORT_RULES = (
    f"cap-termination saturation: caps never stop an H1 match -- all {N_GAMES} games "
    f"are played. Past {CAP_NO_RATE_THRESHOLD} of them the report is "
    f"CAP_SATURATED_NO_RATE, which is a POST-MATCH outcome and not a stop",
    f"score saturation: H1 HAS a band -- the {VIABILITY_THRESHOLD} viability "
    f"threshold -- but it is read off the interval only AFTER all {N_GAMES} games are "
    f"played, and is never an early-stop criterion. A verdict reachable early is a "
    f"verdict biased by when someone chose to look",
    L0.EARLY_STOP_NOT_ABORT_RULE,
)


def decisive_bands(n: int = N_GAMES) -> tuple:
    """The observed rates at which the verdict is forced, DERIVED not typed.

    Returned from the primary interval itself, so the card's 0.6593 / 0.8407
    cannot drift from the interval that actually decides. Below the first band the
    verdict is VIABLE; at or above the second it is NOT_VIABLE; between them the
    interval straddles the threshold and the answer is INCONCLUSIVE.
    """
    half_width = L0.hoeffding_interval(n / 2, n)[1] - 0.5
    return (VIABILITY_THRESHOLD - half_width, VIABILITY_THRESHOLD + half_width)


def viability_verdict(lo: float, hi: float) -> str:
    """Where VIABILITY_THRESHOLD falls relative to the interval.

    ⚠ THE DECISION IS DIRECTIONAL; THE INTERVAL IS NOT. Only the T1j-dominates
    direction returns NOT_VIABLE, but both bounds come from a TWO-SIDED interval
    at alpha=0.05, so each bound individually carries at least 97.5% one-sided
    coverage. Reading a one-sided rule off a two-sided interval is CONSERVATIVE,
    and the card explicitly declines to narrow to the one-sided width.
    """
    if hi < VIABILITY_THRESHOLD:
        return "VIABLE"
    if lo >= VIABILITY_THRESHOLD:
        return "NOT_VIABLE"
    return "INCONCLUSIVE"


#: THE FROZEN H1 SCHEDULE IDENTITY, pinned once the plan is built. It lives in
#: the rules layer for L0's reason: `bind_results` must verify it, and the plan
#: module imports these rules, so the rules cannot import the plan back.
#: 🔴 While this module was being written this constant held a HAND-TYPED
#: 64-hex string that looked entirely plausible and matched nothing. It is now
#: the digest of the built schedule, and a test recomputes it from the tasks:
#: a pinned digest never checked against the artifact it pins is decoration.
H1_TASK_DIGEST = "058e62400849e17f0485c2505f562d948e06548ad33ec21ea7fd81854e8e8a21"

#: WHICH design is being bound. Required at every `bind_results` call.
H1_DESIGN = L0.Design(name="H1", n_openings=N_OPENINGS, n_arms=N_ARMS,
                      n_reps=N_REPS, n_games=N_GAMES, task_digest=H1_TASK_DIGEST)

#: The quantity being estimated, written down before any data exists.
ESTIMAND = (
    "the equally weighted mean, over the 16 opening/colour cells, of T1j's expected "
    "score against calib020_0001 at mdPly 6, the expectation being over ENGINE "
    "RANDOMNESS ONLY. The design is balanced (14 games x 16 cells), so the plain mean "
    "of the 224 scores is that equally weighted cell mean. It is NOT the expected "
    "score against TwixT openings in general: the 8 openings were fixed in advance "
    "and are part of the estimand, not a sample from a population of openings.")

#: Repeated in every report, because the interval is conditional on it.
INDEPENDENCE_CAVEAT = (
    "the 224 games are MODELLED as independent; this is an assumption, not a "
    "measurement. What was verified is that the tasks derive 448 distinct generator "
    "streams -- two per game, search and readout -- colliding with no stream used "
    "before, which rules out accidental stream REUSE only. The seeds are fixed "
    "consecutive integers, the derivation is a fixed XOR, and T1j seeds its own "
    "Zobrist table from an unseeded Random per process, which this design neither "
    "controls nor observes.")

#: L0's list inherited WHOLE, plus two H1-specific prohibitions. Kept in committed
#: code, not only in the card, so a reporting script cannot claim what the protocol
#: forbids without editing a module.
#: 🔴 NOT `L0.FORBIDDEN_CLAIMS + (...)`. TWO of L0's prohibitions state
#: DENOMINATORS -- "8 games per opening and 32 per colour arm" and "the 64 games
#: ARE independent" -- and H1's cells hold 28 and 112 over 224 games. Inheriting
#: them verbatim would have put false denominators into H1's report, which is
#: precisely what those prohibitions exist to prevent. `L0.forbidden_claims`
#: rebuilds the list for THIS design, in the order L0 froze it.
FORBIDDEN_CLAIMS = L0.forbidden_claims(N_GAMES, N_OPENINGS, N_ARMS) + (
    "any pooling of H1 with L0, or any combined rate across the two matches; they "
    "differ in n, were preregistered separately, and pooling after seeing both is a "
    "choice made with the data in hand",
    "any use of the ~191 Elo figure that SIZED the 0.75 threshold to describe the "
    "result; it justified where the line sits, before any data, and is not a finding",
)


def viability_report(results: Sequence[Dict[str, Any]],
                     tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The whole H1 report and its verdict, or a refusal.

    The verdict is computed from the PRIMARY interval only. Wilson is reported
    alongside and decides nothing: L0's card recorded why at length, and that
    correction is inherited rather than re-derived.
    """
    pairs, why = L0.bind_results(results, tasks, design=H1_DESIGN)
    if pairs is None:
        return {"reported": False, "reason": why}

    overall = L0._summary(pairs)
    caps = overall["cap_terminations"]
    if caps > CAP_NO_RATE_THRESHOLD:
        return {"reported": False, "outcome": "CAP_SATURATED_NO_RATE",
                "reason": (f"{caps} of {N_GAMES} games terminated at the ply cap, more "
                           f"than the preregistered threshold of {CAP_NO_RATE_THRESHOLD}; "
                           f"the positions did not resolve and there is no rate to "
                           f"report, and so no viability verdict"),
                "cap_terminations": caps, "games": overall["games"]}

    score = overall["t1j_score"]
    hl, hh = L0.hoeffding_interval(score, N_GAMES)
    wl, wh = L0.wilson_interval(score, N_GAMES)
    verdict = viability_verdict(hl, hh)
    lo_band, hi_band = decisive_bands(N_GAMES)
    overall.update({
        "ci95_hoeffding": [hl, hh],
        "estimand": ESTIMAND,
        "ci95_hoeffding_method": (
            "PRIMARY, and the ONLY interval the verdict reads. Hoeffding, closed "
            "form, half-width sqrt(ln(2/alpha)/(2n)), alpha = %.15g, TWO-SIDED. A 95%% "
            "bound UNDER AN INDEPENDENCE MODEL." % ALPHA),
        "independence_is_modelled_not_measured": INDEPENDENCE_CAVEAT,
        "ci95_wilson": [wl, wh],
        "ci95_wilson_method": (
            "SECONDARY, NOMINAL, APPROXIMATE, and DECIDES NOTHING. Wilson score "
            "interval, z = %.15g. Assumes Bernoulli trials, which a TwixT game is "
            "not. NOT conservative and not described as such." % Z_95),
        "variance_deficit_descriptive": L0.variance_deficit(caps, N_GAMES),
        "cap_warning": caps > 0,
    })
    return {
        "reported": True,
        "verdict": verdict,
        "verdict_rule": (
            f"VIABLE if the two-sided 95%% Hoeffding UPPER bound < "
            f"{VIABILITY_THRESHOLD}; NOT_VIABLE if the LOWER bound >= "
            f"{VIABILITY_THRESHOLD}; INCONCLUSIVE otherwise. Predeclared before any "
            f"H1 data existed. The decision is directional; the interval is not."),
        "decisive_bands": {"viable_below": lo_band, "not_viable_at_or_above": hi_band},
        "overall": overall,
        "by_colour_arm": {k: L0._summary(v) for k, v in
                          sorted(L0._cell_pairs(pairs, "colour_arm").items())},
        "by_opening": {k: L0._summary(v) for k, v in
                       sorted(L0._cell_pairs(pairs, "opening").items())},
        "descriptive_only": ["by_colour_arm", "by_opening"],
        "bounded_to": (
            "these 8 frozen openings, this colour balance, mdPly 6, "
            "calib020_0001 at 400 simulations, ply cap 280, one run of 224 games"),
        "forbidden_claims": FORBIDDEN_CLAIMS,
    }
