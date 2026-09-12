"""H2 — the deterministic-readout head-to-head: FROZEN RULES.

The card is `docs/superpowers/2026-09-09-t1j-h2-deterministic-readout-card.md`
(frozen at 5aa5db4). H2 asks whether T1j is stronger than our incumbent **in the
deterministic visit-count-argmax configuration** — not stronger than every
deterministic use of the model, a claim nothing establishes.

🔴 NOTHING HERE RUNS A GAME. This module holds the design constants, the frozen
transcript identity, the per-cell degeneracy screen, the parity verdict and the
report. The gate lives in `h2_match_runner`; the seed block below is reserved ON
PAPER and registered nowhere.

WHAT DIFFERS FROM H1, and it is deliberately short:
  * ONE GAMEPLAY-RULE CHANGE -- the incumbent readout is `argmax`, not
    `opening_temperature`. Only this changes what either side does at the board.
  * the sample size (736, not 224), the seed block, the incumbent identity and
    the output locations also differ; those are not gameplay.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import l0_match_rules as L0

#: SHARED WITH L0, RE-EXPORTED RATHER THAN RESTATED, so a change to the shared
#: rule cannot leave H2 describing a different one.
PLY_CAP = L0.PLY_CAP
SCORE_DRAW = L0.SCORE_DRAW
ALPHA = L0.ALPHA
WINNERS = L0.WINNERS
#: EXACTLY TWO. `l0_match_rules` refuses anything else, and THIS PROTOCOL HAS NO
#: RESIGNATION -- the card's first draft listed one, which would have written a
#: state the runner cannot produce into the definition of a transcript.
TERMINAL_REASONS = L0.TERMINAL_REASONS
N_OPENINGS = L0.N_OPENINGS
N_ARMS = L0.N_ARMS

#: H2's own size. 46 x 16 = 736 gives a Hoeffding half-width of 0.0500603, so the
#: verdict is forced only below 0.4499 or above 0.5501. H1's 224 gave 0.0907,
#: which cannot separate parity from a moderate edge -- the ambiguity that left
#: H1 inconclusive.
N_REPS = 46
N_GAMES = N_OPENINGS * N_ARMS * N_REPS                  # 736

#: THE GAMEPLAY-RULE CHANGE, and the only one. Under argmax the incumbent plays
#: the visit-count maximum; `temp_high`, `temp_low` and `opening_temp_plies`
#: become INERT and are recorded as NOT APPLICABLE, never as "unchanged" -- a
#: setting that no longer acts is not the same setting.
SELECTION_MODE = "argmax"
INERT_UNDER_ARGMAX = ("temp_high", "temp_low", "opening_temp_plies")

#: Unchanged from H1, restated only where the report must carry them.
T1J_MDPLY = 6
MCTS_SIMS = 400

#: ATTEMPT 1's BLOCK. Registered 2026-09-11, then RETIRED WHOLE the same day when
#: the single authorized match VOIDed at task 0 -- `build_reference_agent` refused
#: H2's own readout, so no agent was built and NO SEED WAS DRAWN (exposed 0).
#: Kept under its own name so the voided attempt's plan stays loadable AS A RECORD
#: and is refused for EXECUTION by the registry, never by a digest mismatch alone.
#: ⚠ The retirement was REVIEWED AND KEPT on 2026-09-11: nothing was learned about
#: gameplay, and fresh seeds cost less than a retrospective exception.
H2_ATTEMPT1_SEED_BLOCK = (202618000, 202618000 + N_GAMES)   # [202618000, 202618736)

#: ATTEMPT 2's BLOCK, SPENT. Reserved 2026-09-11 for the retry against the repaired
#: builder, then ACCOUNTED 736 / EXPOSED 383 / RETIRED WHOLE on 2026-09-12 -- not by
#: an authorized match but by the INCIDENT: an injected-defect control deleted
#: `check_gate()` from `run_h2` and played 383 of the 736 games. Kept under its own
#: name so that run's records stay verifiable AS EVIDENCE and are refused for
#: EXECUTION by the registry.
#: ⚠ Seed 202620383 is deliberately NOT claimed as drawn -- task 383 emitted a trace
#: `task_start` only -- and the whole-block retirement covers that uncertainty
#: instead of a claim either way.
H2_ATTEMPT2_SEED_BLOCK = (202620000, 202620000 + N_GAMES)   # [202620000, 202620736)

#: ATTEMPT 3 (2026-09-12): a FRESH interval for the match H2 still has not had.
#: Re-proved against the registries AS THEY STAND -- attempts 1 and 2 both spent and
#: both now reachable as controls -- plus every paper reservation no registry can see
#: and every derived stream. The gap floor is the candidate's OWN SIZE (736); the
#: nearest other boundary is 1,264 below and there is nothing above.
#:
#: 🔴 RESERVED ON PAPER until the seed-preparation step registers it. Registering is
#: a reviewed edit under its own authorization, so a block that is never authorized
#: costs nothing to abandon. `h2_match_runner.check_seed_registration` REFUSES a run
#: while it is absent from `ACCOUNTED_SEED_INTERVALS`, and the gate refuses first.
H2_SEED_BLOCK = (202622000, 202622000 + N_GAMES)        # [202622000, 202622736)

#: THE FROZEN H2 SCHEDULE IDENTITY, pinned once the plan is built. It lives here
#: for L0's reason: `bind_results` must verify it, and the plan module imports
#: these rules, so the rules cannot import the plan back. Recomputed by a test
#: from the built schedule -- a pinned digest never checked against the artifact
#: it pins is decoration.
#: ATTEMPT 1's digests, kept for the voided attempt's record. Its schedule must
#: stay verifiable AS A RECORD: a spent plan is refused for execution by the
#: registry, and a digest that no longer matches anything would hide that.
H2_ATTEMPT1_TASK_DIGEST = "3c0a0ae12c61dae69b134b30d0f4adb7dc97b9e46478cebe679784996d7172d2"
H2_ATTEMPT1_FULL_TASK_DIGEST = ("08d44fb84851e6fe91ca6510192ef7f3dd"
                                "57d7ea0997e59d4e4d5d7c60384235")

#: ATTEMPT 2's digests, kept for the incident's record: its 383 preserved games must
#: stay verifiable against the schedule they came from.
H2_ATTEMPT2_TASK_DIGEST = "ef68cb9962d1a8d42092ddf329dbdcb27a51f5594567e3f21c14cbe35c435c69"

#: ATTEMPT 3's digests: the SAME 736 tasks with seeds from the fresh block, so only
#: `seed`, the derived `rng_streams` and the task ids differ. Recomputed by a test
#: from the built schedule, as attempts 1 and 2 were.
H2_TASK_DIGEST = "4cec38c75a4297d1b494df1c5754215962b138767d8f4c38fc4996078b94c332"

#: 🔴 AND A DIGEST OVER **EVERY FIELD**, because the one above does not cover them.
#: `l0_task_digest` projects the DESIGN DIMENSIONS, so a forged `reference_sha256`
#: or forged `rng_streams` leaves it unchanged. Comparing a supplied schedule with
#: a fresh build from the same live source plan does not fix that either: both
#: sides move together if the source changes. This pins the whole artifact.
H2_ATTEMPT2_FULL_TASK_DIGEST = ("e513b479824c069f02b984ace95799d3c9fefdf"
                                "ac48c80f2f8154bb7bf830a65")
H2_FULL_TASK_DIGEST = ("2d330ede735b578ea83ab68ebbb22611a7"
                       "b1e41c1af64ca9f98c36ee84a3e2da")


def h2_full_task_digest(tasks: Sequence[Mapping[str, Any]]) -> str:
    """sha256 over EVERY field of every task, in order. Nothing projected away."""
    return hashlib.sha256(json.dumps([dict(t) for t in tasks], sort_keys=True,
                                     separators=(",", ":"),
                                     default=str).encode()).hexdigest()

#: THE PRIMARY QUESTION IS PARITY. Not 0.75 -- that asked whether T1j is far
#: enough ahead to justify targeted work, a different question that may be
#: REPORTED here but can never replace this one.
PARITY = 0.50

#: Reported for continuity with H1 and DECIDING NOTHING.
H1_INVESTMENT_THRESHOLD = 0.75

#: The per-cell degeneracy screen (card §2.1): 90% applied WHERE IT BINDS. A
#: GLOBAL rule does not bind -- one wholly collapsed cell beside fifteen distinct
#: ones gives 691/736 = 93.9% and passes, while a sixteenth of the design carries
#: nothing at all.
MIN_DISTINCT_PER_CELL = 42                              # of N_REPS = 46

#: Cap policy, inherited: caps never stop the match -- that would be an early stop
#: -- and the decision is made afterwards. A rate over mostly-unresolved games
#: measures the cap, not the players.
CAP_NO_RATE_THRESHOLD = N_GAMES // 2                    # 368; "more than half" is > this

#: No early stop, for L0's reason: the canonical screen early-stopped at 0.875 and
#: the full match came back 0.594. An early stop cannot bias a BAND decision and
#: does bias a RATE -- and this verdict is a rate question.
EARLY_STOP = None


def may_stop_early(*_args: Any, **_kwargs: Any) -> bool:
    """Always False. H2 plays all 736 games."""
    return False


def seed_is_outside_the_reserved_block(seed: Any) -> bool:
    """The seed abort rule AS A PREDICATE: prose cannot be run."""
    lo, hi = H2_SEED_BLOCK
    return not lo <= int(seed) < hi


# ───────────────────────── the frozen transcript identity ───────────────────

class H2RulesError(ValueError):
    """A refusal from H2's rules."""


#: THE GAMEPLAY FIELDS, and nothing else (card §2.2). Hash the whole record
#: instead and `seed`, `task_id` and `rep` make all 46 games of a cell unique even
#: when the gameplay is byte-identical -- the screen would report 46 of 46
#: distinct for a cell that played ONE game 46 times and would pass every design
#: it exists to catch.
TRANSCRIPT_EXCLUDES = (
    "task_id", "rep", "seed", "rng_streams", "opening", "colour_arm",
    "timestamp", "elapsed_s", "wall_clock", "root_visits", "root_value",
    "policy_rank", "readout_overrode_leader", "top2", "digest", "identity",
    "plan_sha256", "engine_version",
)


def _int(value: Any, what: str) -> int:
    """TYPE-STRICT: `True` is not 1 and `"11"` is not 11."""
    if type(value) is not int:
        raise H2RulesError(f"{what} is {value!r} ({type(value).__name__}); an int is required")
    return value


#: THE PRODUCTION GAME ALWAYS STARTS RED. `TwixtState.to_move` defaults to "red"
#: ("Red moves first"), the runner records `mover` as the side that moved and
#: `ply` as the count AFTER the move, so recorded ODD plies are red and EVEN plies
#: are black -- in every task, in both arms.
STARTING_COLOUR = "red"

#: 🔴 WHAT `colour_arm` DOES, AND WHAT IT DOES NOT. It assigns SYSTEMS to colours
#: -- `t1j_red` means T1j plays red and our incumbent plays black -- and it has NO
#: bearing on turn order. My first version derived the expected mover from the
#: ARM, so every `t1j_black` task expected black at ply 7 while the real state says
#: red: HALF THE SCHEDULE WOULD HAVE VOIDED. The synthetic tests passed because
#: they built their fixtures from that same helper, so the test and the code shared
#: one bug and agreed with each other. The test now derives its expectation from
#: `TwixtState` itself.
ARM_ASSIGNS_SYSTEMS_NOT_TURN_ORDER = (
    "colour_arm names which SYSTEM plays which COLOUR; red moves first in every "
    "game regardless. Turn order comes from the frozen starting colour and ply "
    "parity, never from the arm.")


def colour_at_ply(ply: int) -> str:
    """WHOSE MOVE recorded ply `ply` IS: odd = red, even = black, always.

    🔑 NOT "the movers alternate". Flipping EVERY mover in a game preserves
    alternation while swapping which side played every move, turning one
    transcript silently into a different one. Parity from a fixed starting colour
    is what a flipped game disagrees with at ply one.
    """
    return STARTING_COLOUR if _int(ply, "ply") % 2 == 1 else (
        "black" if STARTING_COLOUR == "red" else "red")


def transcript(plies: Sequence[Mapping[str, Any]], result: Mapping[str, Any], *,
               opening_bound: int) -> Tuple[Any, ...]:
    """THE FROZEN TRANSCRIPT: the type-strict ordered post-opening
    `(mover, row, col)` moves, then the terminal reason and the winner.

    EXACTLY that. Every excluded field is excluded because including it would let
    the SCHEDULE supply the diversity the screen is meant to measure.

    Refuses -- never returns a novel value -- when the record is malformed:

    * the ply sequence must be EXACTLY `opening_bound + 1 ... result["plies"]`,
      each index once, in increasing order. 🔴 "Contiguous from the first
      post-opening ply" was not enough at either end: a sequence missing its FIRST
      record is still contiguous from whatever survives, and one missing its LAST
      is still contiguous up to there. Both endpoints are anchored to records the
      run itself wrote, so a truncation is a count mismatch, not a judgement call.
    * every mover must EQUAL the colour whose turn that ply is (`colour_at_ply`),
      which depends on PLY PARITY and not on the colour arm.
    * the terminal reason must be one of the two the protocol has.
    """
    lo = _int(opening_bound, "opening_bound") + 1
    hi = _int(result.get("plies"), "task_result.plies")
    want = list(range(lo, hi + 1))
    got = [_int(p.get("ply"), "ply") for p in plies]
    if got != want:
        raise H2RulesError(
            f"the ply sequence is {got[:4]}...{got[-4:] if len(got) > 4 else ''} "
            f"({len(got)} records) but the run's own records declare exactly "
            f"{lo}..{hi} ({len(want)} records): a missing, duplicated or "
            f"out-of-order ply makes the transcript undefined, and an undefined "
            f"transcript is never hashed into a novel value.")
    reason = result.get("terminal_reason")
    if reason not in TERMINAL_REASONS:
        raise H2RulesError(
            f"terminal_reason {reason!r} is not one of {TERMINAL_REASONS}; this "
            f"protocol has no resignation and no other terminal state")
    winner = result.get("winner")
    if winner is not None and winner not in WINNERS:
        raise H2RulesError(f"winner {winner!r} is not one of {WINNERS} or None")
    out: List[Any] = []
    for p in plies:
        ply = _int(p["ply"], "ply")
        mover = p.get("mover")
        want_mover = colour_at_ply(ply)
        if mover != want_mover:
            raise H2RulesError(
                f"ply {ply} records mover {mover!r} but the game's turn order gives "
                f"{want_mover!r}; movers are bound to PLY PARITY from a fixed "
                f"starting colour, not merely alternating and not to the arm")
        move = p.get("move")
        if not isinstance(move, (list, tuple)) or len(move) != 2:
            raise H2RulesError(f"ply {ply}: move {move!r} is not a (row, col) pair")
        out.append((mover, _int(move[0], "row"), _int(move[1], "col")))
    out.append(("terminal", reason, winner))
    return tuple(out)


def transcript_digest(t: Sequence[Any]) -> str:
    """A stable digest of a transcript, for counting distinct ones."""
    return hashlib.sha256(json.dumps(list(t), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


# ───────────────────── the per-cell degeneracy screen ───────────────────────

def cell_key(task: Mapping[str, Any]) -> Tuple[str, str]:
    return (str(task["opening"]), str(task["colour_arm"]))


def degeneracy_screen(per_game: Sequence[Mapping[str, Any]], *,
                      canonical_task_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Distinct transcripts WITHIN EACH of the 16 cells (card §2.1).

    `per_game` carries one entry per game: `opening`, `colour_arm` and the
    transcript digest. Returns every cell's count -- ALL SIXTEEN, not a global
    figure and not only the minimum, so a reader can see which cell was thin.

    🔴 NO "effective n = 16" CLAIM IS MADE. That holds only if ALL sixteen cells
    collapse; a partial collapse costs an amount this design does not quantify,
    which is why the screen REFUSES rather than adjusts.
    """
    # 🔴 THE INPUT IS BOUND FIRST. Thresholding alone accepted 672 rows -- exactly
    # 42 per cell -- because "16 cells and >= 42 distinct" is true of a set that is
    # missing 64 games. A screen that does not know how many games there were
    # cannot say how many were distinct.
    if canonical_task_ids is not None:
        want = list(canonical_task_ids)
        got = [g.get("task_id") for g in per_game]
        if len(got) != len(want) or sorted(map(str, got)) != sorted(map(str, want)):
            raise H2RulesError(
                f"the transcript vector holds {len(got)} rows for {len(want)} canonical "
                f"tasks and its task ids are not exactly theirs; one transcript per "
                f"task, no more and no fewer.")
    elif len(per_game) != N_GAMES:
        raise H2RulesError(
            f"the transcript vector holds {len(per_game)} rows, not {N_GAMES}")

    counts: Dict[Tuple[str, str], set] = {}
    totals: Counter = Counter()
    for g in per_game:
        key = (str(g["opening"]), str(g["colour_arm"]))
        counts.setdefault(key, set()).add(g["transcript_digest"])
        totals[key] += 1
    cells = []
    for key in sorted(counts):
        n_distinct, n_games = len(counts[key]), totals[key]
        cells.append({"cell": list(key), "n_games": n_games, "n_distinct": n_distinct,
                      "passes": n_distinct >= MIN_DISTINCT_PER_CELL})
    # EVERY cell must hold exactly the frozen number of GAMES as well: a cell with
    # 42 games all distinct is not a cell with 46 games of which 42 differ.
    short = [c["cell"] for c in cells if c["n_games"] != N_REPS]
    if short:
        raise H2RulesError(
            f"{len(short)} cell(s) do not hold exactly {N_REPS} games: {short[:3]}")
    failing = [c for c in cells if not c["passes"]]
    return {
        "min_distinct_required": MIN_DISTINCT_PER_CELL,
        "reps_per_cell": N_REPS,
        "cells": cells,
        "n_cells": len(cells),
        "failing_cells": [c["cell"] for c in failing],
        # 🔴 `and len(cells) == 16` STOOD HERE AND WAS DELETED: once the vector is
        # bound to 736 rows AND every cell must hold exactly 46 games, sixteen cells
        # follow arithmetically, so no control could distinguish the clause. A
        # branch nothing can reach proves nothing; the binding above owns the rule.
        "passes": not failing,
        "global_distinct_rate_NOT_THE_RULE": (
            sum(c["n_distinct"] for c in cells) / sum(c["n_games"] for c in cells)
            if cells else None),
    }


# ───────────────────────────── the parity verdict ───────────────────────────

def decisive_bands(n: int = N_GAMES) -> Tuple[float, float]:
    """The observed scores at which the verdict is forced, DERIVED not typed."""
    half_width = L0.hoeffding_interval(n / 2, n)[1] - 0.5
    return (PARITY - half_width, PARITY + half_width)


def parity_verdict(lo: float, hi: float) -> str:
    """Where PARITY falls relative to the interval. ONE RULE, ONE THRESHOLD."""
    if lo > PARITY:
        return "T1J_STRONGER"
    if hi < PARITY:
        return "NOT_STRONGER"
    return "INCONCLUSIVE"


#: The interval's standing, carried in the REPORT and not only in the card. A
#: claim that lives only in a document is one a reader of the result never sees.
INTERVAL_STANDING = (
    "NOMINAL UNDER THE INDEPENDENCE MODEL. Hoeffding is distribution-free about "
    "the SHAPE of a bounded outcome, not about dependence between games. These "
    "games run on predetermined pseudorandom seed streams against a deterministic "
    "opponent over a design that repeats each opening 46 times, so independence is "
    "a MODEL being relied upon, not a fact this design establishes. The per-cell "
    "transcript screen does NOT validate it: distinct transcripts detect the same "
    "game replayed and nothing more.")

FORBIDDEN_CLAIMS = L0.forbidden_claims(N_GAMES, N_OPENINGS, N_ARMS) + (
    "any pooling of H2 with either H1 attempt or with L0",
    "any causal claim from comparing H2's rate with H1's: the readout AND the seeds "
    "differ, so the comparison is confounded by construction and may be displayed "
    "beside its confound but never interpreted as the effect of the readout",
    "that a NOT_STRONGER result makes H1's 0.6763 attributable to the readout",
    "that visit-count argmax is the strongest deterministic use of the model; it is "
    "one qualified deterministic rule and nothing here establishes it as the best",
    "any coverage guarantee for the interval, which is nominal under the "
    "independence model this design does not establish",
    "any training decision; a T1J_STRONGER outcome makes a training question worth "
    "DESIGNING and authorizes nothing",
)


# ─────────────────────────────── the report ─────────────────────────────────

#: WHICH design is being bound. Required at every `bind_results` call. The digest
#: is pinned by `h2_match_plan` and recomputed by a test from the built schedule:
#: a pinned digest never checked against the artifact it pins is decoration.
def h2_design(task_digest: str) -> Any:
    return L0.Design(name="H2", n_openings=N_OPENINGS, n_arms=N_ARMS,
                     n_reps=N_REPS, n_games=N_GAMES, task_digest=task_digest)


def h2_report(results: Sequence[Mapping[str, Any]], tasks: Sequence[Mapping[str, Any]],
              per_game: Sequence[Mapping[str, Any]], *, task_digest: str) -> Dict[str, Any]:
    """The whole H2 report and its verdict, or a refusal.

    🔑 THE ORDER IS THE POINT. The per-cell degeneracy screen runs FIRST and a
    failing cell PREVENTS the interval from being computed at all -- a screen that
    runs after the number it guards is decoration. The refusal carries the failing
    cells by name and every cell's count.
    """
    # 🔴 RESULT INTEGRITY BINDS FIRST. Screening first let a schedule of INVALID
    # results be reported as `DEGENERATE DESIGN` -- a data-integrity failure wearing
    # a design-validity name. The bind refuses on its own terms; only then does the
    # screen speak, and only then can the interval be computed.
    pairs, why = L0.bind_results(list(results), list(tasks), design=h2_design(task_digest))
    if pairs is None:
        return {"reported": False, "reason": why, "outcome": "REFUSED",
                "verdict": "REFUSED", "degeneracy_screen": None,
                "interval": None, "score": None,
                "interval_standing": INTERVAL_STANDING,
                "forbidden_claims": FORBIDDEN_CLAIMS}

    screen = degeneracy_screen(per_game,
                               canonical_task_ids=[t["task_id"] for t in tasks])
    if not screen["passes"]:
        return {
            "reported": False,
            "outcome": "INCONCLUSIVE — DEGENERATE DESIGN",
            "verdict": "INCONCLUSIVE — DEGENERATE DESIGN",
            "reason": (f"{len(screen['failing_cells'])} of {screen['n_cells']} cell(s) "
                       f"hold fewer than {MIN_DISTINCT_PER_CELL} distinct transcripts "
                       f"of {N_REPS}: {screen['failing_cells']}. The primary interval "
                       f"is NOT computed: the preregistered DIVERSITY screen failed. "
                       f"⚠ This does NOT establish that the games were statistically "
                       f"non-independent -- independent seeded games can produce "
                       f"identical transcripts, and §3.1 says plainly that this screen "
                       f"cannot test independence. What failed is the design's own "
                       f"diversity requirement."),
            "degeneracy_screen": screen,
            "interval": None, "score": None,
            "interval_standing": INTERVAL_STANDING,
            "forbidden_claims": FORBIDDEN_CLAIMS,
        }

    overall = L0._summary(pairs)
    caps = overall["cap_terminations"]
    if caps > CAP_NO_RATE_THRESHOLD:
        return {"reported": False, "outcome": "CAP_SATURATED_NO_RATE",
                "verdict": "CAP_SATURATED_NO_RATE",
                "reason": (f"{caps} of {N_GAMES} games terminated at the ply cap, more "
                           f"than the preregistered threshold of {CAP_NO_RATE_THRESHOLD}; "
                           f"the positions did not resolve, so there is no rate and no "
                           f"parity verdict"),
                "cap_terminations": caps, "games": overall["games"],
                "degeneracy_screen": screen, "interval": None, "score": None,
                "interval_standing": INTERVAL_STANDING,
                "forbidden_claims": FORBIDDEN_CLAIMS}

    score = overall["t1j_score"]
    hl, hh = L0.hoeffding_interval(score, N_GAMES)
    wl, wh = L0.wilson_interval(score, N_GAMES)
    verdict = parity_verdict(hl, hh)
    lo_band, hi_band = decisive_bands(N_GAMES)
    overall.update({
        "ci95_hoeffding": [hl, hh],
        "ci95_hoeffding_method": (
            "PRIMARY, and the ONLY interval the verdict reads. Hoeffding, closed form, "
            "half-width sqrt(ln(2/alpha)/(2n)), alpha = %.15g, TWO-SIDED." % ALPHA),
        "interval_standing": INTERVAL_STANDING,
        "ci95_wilson": [wl, wh],
        "ci95_wilson_method": (
            "SECONDARY, NOMINAL, APPROXIMATE, and DECIDES NOTHING. Wilson assumes "
            "Bernoulli trials, which a TwixT game is not. 🔴 It may not replace "
            "Hoeffding after the result is seen -- the error H1's closure named."),
        "variance_deficit_descriptive": L0.variance_deficit(caps, N_GAMES),
        "cap_warning": caps > 0,
    })
    return {
        "reported": True,
        "design": "H2",
        "selection_mode": SELECTION_MODE,
        "inert_under_argmax": list(INERT_UNDER_ARGMAX),
        "n_games": N_GAMES,
        "task_digest": task_digest,
        "score": score / N_GAMES,
        "interval": [hl, hh],
        "interval_standing": INTERVAL_STANDING,
        "outcome": verdict,
        "verdict": verdict,
        "verdict_rule": (
            f"T1J_STRONGER if the two-sided 95% Hoeffding LOWER bound > {PARITY}; "
            f"NOT_STRONGER if the UPPER bound < {PARITY}; INCONCLUSIVE otherwise. "
            f"Predeclared before any H2 data existed."),
        "parity": PARITY,
        "decisive_bands": {"not_stronger_below": lo_band,
                           "t1j_stronger_above": hi_band},
        "degeneracy_screen": screen,
        "overall": overall,
        "by_colour_arm": {k: L0._summary(v) for k, v in
                          sorted(L0._cell_pairs(pairs, "colour_arm").items())},
        "by_opening": {k: L0._summary(v) for k, v in
                       sorted(L0._cell_pairs(pairs, "opening").items())},
        "descriptive_only": ["by_colour_arm", "by_opening"],
        "reported_for_continuity_only": {
            "h1_investment_threshold": H1_INVESTMENT_THRESHOLD,
            "note": ("REPORTED, NEVER DECISIVE. 0.75 asked whether T1j is far enough "
                     "ahead to justify targeted work; H2's question is parity, and "
                     "this figure cannot change the outcome above."),
        },
        "bounded_to": (
            "these 8 frozen openings, this colour balance, mdPly 6, calib020_0001 at "
            "400 simulations with the ARGMAX readout, ply cap 280, one run of 736 games"),
        "forbidden_claims": FORBIDDEN_CLAIMS,
    }
