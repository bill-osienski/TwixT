"""The L0 larger match's FROZEN decision and reporting rules, as committed code.

L0 is a PREREGISTRATION. Nothing here plays, scores or seeds anything: it turns a
completed result vector into a report. It is committed before the match is
authorized so that authorizing it later cannot become an occasion to choose a
statistic.

WHAT L0 IS, AND HOW IT DIFFERS FROM THE E4 SCREEN
--------------------------------------------------
The screen asked a yes/no question -- is T1j inside a usable band -- and was
allowed to stop early once the answer was forced. L0 asks a MEASUREMENT question:
what is T1j's score rate at mdPly 6 against calib020_0001, over this fixed design.

So the screen's machinery is deliberately NOT reused here:

  * NO EARLY STOP. All 64 games are played. An early stop selects a stopping
    point on the data, which biases a rate estimate even when it cannot bias a
    band decision. `e4_screen_rules.early_in_band_forced` and its helpers must
    never be called on an L0 run, and `may_stop_early` below always answers no.
  * NO BAND, NO VERDICT. There is no SATURATED/IN_BAND classification, because
    there is no decision to make. There is a rate and an interval.
  * ONE RUN. A second run of the same design would need a second seed block and
    would turn one preregistered measurement into a choice among measurements.

The only permitted early termination is a FAIL-CLOSED INTEGRITY ABORT: the run
stops and reports NO rate at all. An abort is not a short match; it is not a
match.

THE INTERVAL: A VALID BOUND FIRST, A NOMINAL ONE ALONGSIDE
----------------------------------------------------------
The house method elsewhere is a bootstrap. A bootstrap needs a generator and a
recorded resampling seed, and L0 is forbidden from constructing one. That is a
real gain rather than a constraint: both intervals below are CLOSED FORM, so the
reported numbers are reproducible from the recorded scores alone, with no
resampling seed to record, lose, or quietly change.

THE ESTIMAND, STATED BEFORE THE INTERVAL
The quantity being estimated is the EQUALLY WEIGHTED MEAN, over the 16
opening/colour cells, of T1j's expected score against calib020_0001 at mdPly 6,
where the expectation is over ENGINE RANDOMNESS ONLY -- our agent's opening
temperature and MCTS tie-breaks, plus whatever T1j itself contributes. The design
is balanced (4 games in each of 16 cells), so the plain mean of the 64 scores IS
that equally weighted cell mean; no reweighting is applied or needed.

It is NOT the expected score against TwixT openings in general. Eight openings
were fixed in advance and are part of the estimand, not a sample from a
population of openings.

PRIMARY: `hoeffding_interval`. Hoeffding's inequality bounds the deviation of a
mean of INDEPENDENT variables BOUNDED IN [0, 1]. It requires neither a Bernoulli
outcome nor identical distributions -- which matters here, because a TwixT game
scores 1.0/0.5/0.0 and the eight openings are not the same distribution.

  INDEPENDENCE IS A MODEL, NOT A MEASURED FACT, AND THE INTERVAL IS CONDITIONAL
  ON IT. What was actually verified is that the 64 tasks derive 128 DISTINCT
  generator streams, none colliding with any stream this workstream has used
  before. That rules out accidental stream REUSE. It does not prove statistical
  independence: the seeds are fixed consecutive integers, the derivation is a
  fixed XOR, and T1j seeds its own `Zobrist` table from an unseeded `Random` per
  process, which no part of this design controls or observes. So the interval is
  a 95% bound UNDER AN INDEPENDENCE MODEL, and the report says so in those words.
  Coverage is at least nominal GIVEN that model; the model itself is assumed.

It is wide, and that width is the honest price of a distribution-free guarantee
over a 64-game design.

SECONDARY: `wilson_interval`, reported as NOMINAL AND APPROXIMATE.

  A PREVIOUS VERSION OF THIS MODULE CLAIMED WILSON WAS "PROVABLY CONSERVATIVE"
  HERE. THAT WAS WRONG, and wrong in a way worth recording, because the argument
  looked like a proof. It showed that draws lower the variance --

      Var(X) = 0.25b + P(X=1) - p^2 = p(1-p) - 0.25b  <=  p(1-p)

  -- which is true, and then asserted coverage from it, which does not follow. A
  variance inequality is not a coverage statement. Wilson's coverage oscillates
  with n and p and dips below nominal on its own, draws or no draws: at n = 64 and
  a true rate of 0.002 its exact coverage is 87.97%, and across p = 0.001..0.999
  it is below 95% at 43.3% of values, including 94.01% at p = 0.5. The
  exhaustive test that "pinned" the claim enumerated observed compositions, which
  says nothing about repeated sampling, and used the OBSERVED draw count as if it
  were the true draw probability.

  So Wilson stays -- it is the conventional figure and it is far tighter -- but it
  is labelled nominal, it is never called conservative, and it is not the primary.

`variance_deficit` is retained as a DESCRIPTIVE property of the observed outcome
mix. It is not a coverage guarantee and the report does not present it as one.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Sequence

#: The design. 8 frozen openings x 2 colour arms x 4 repetitions.
N_OPENINGS = 8
N_ARMS = 2
N_REPS = 4
N_GAMES = N_OPENINGS * N_ARMS * N_REPS          # 64

#: The single endpoint. The screen's weak endpoint (mdPly 3) is not measured
#: again: it saturated at 0.0/16 and a rate there needs no more precision.
T1J_MDPLY = 6

#: Unchanged from the E4 screen, and deliberately so.
PLY_CAP = 280
SCORE_T1J_WIN = 1.0
SCORE_DRAW = 0.5
SCORE_T1J_LOSS = 0.0

#: 95%. Two-sided normal quantile, written out so the number in the report and
#: the number in the code cannot drift apart.
Z_95 = 1.959963984540054
ALPHA = 0.05

#: Recorded outcome vocabulary, taken from the E4 screen's OWN durable records
#: rather than invented here: `winner` is a COLOUR, and t1j_points is 1.0 exactly
#: when the winner is the anchor's colour. An earlier version of `score_game`
#: compared winner to the string "t1j", which no recorded row has ever contained.
WINNERS = ("red", "black", None)
TERMINAL_REASONS = ("win", "cap")

#: Cap policy. A cap termination means the position did not resolve inside 280
#: plies. Caps do NOT stop the match -- that would be an early stop -- so all 64
#: games are always played. The decision is made afterwards, and is preregistered:
#: more than half unresolved and there is no rate to report, because a rate over
#: mostly-unresolved games measures the cap, not the players.
CAP_NO_RATE_THRESHOLD = N_GAMES // 2            # 32; "more than half" is > this

#: L0's OWN abort rules. The screen's list is deliberately NOT imported: it
#: contains a STATISTICAL abort (more than half of an endpoint cap-terminating
#: makes it INCOMPLETE) which contradicts L0's no-early-stop rule and would give a
#: runner two conflicting authorities. Everything below is an INTEGRITY failure.
L0_ABORT_RULES = (
    "any per-ply state divergence between the two engines",
    "any T1j query that does not complete its requested depth",
    "any illegal move, or the null sentinel, from either side",
    "any postcondition failure: a Window/Frame, a non-headless jvm, a mutated host "
    "preference store, or an unauthorized reflective access",
    "any artifact identity mismatch: jar, JDK component, or checkpoint sha",
    "any seed outside the reserved L0 block, or any seed used twice",
    "any failure to write or fsync a durable record",
)

#: NOT an abort rule, and named here so it cannot be quietly reintroduced.
NOT_ABORT_RULES = (
    "cap-termination saturation: caps never stop an L0 match; see CAP_NO_RATE_THRESHOLD",
    "score saturation: L0 measures a rate and has no band to saturate",
    "any early stop of any kind",
)

#: There is no early stop. This is a constant, not a tunable.
EARLY_STOP = None


def may_stop_early(*_args: Any, **_kwargs: Any) -> bool:
    """Always False. L0 plays all 64 games.

    Present as a callable so a runner cannot silently acquire an early stop by
    importing the screen's rules instead: the L0 protocol names this function,
    and this function has one answer.
    """
    return False


def score_game(winner, anchor_colour: str, terminal_reason: str) -> float:
    """T1j's points for one game. `winner` is a COLOUR, as the records store it.

    TwixT cannot draw by rule, so a draw is ALWAYS a ply-cap termination. T1j
    scores 1.0 exactly when the winning colour is the colour T1j was playing.
    """
    if terminal_reason not in TERMINAL_REASONS:
        raise ValueError(f"unknown terminal_reason {terminal_reason!r}")
    if anchor_colour not in ("red", "black"):
        raise ValueError(f"anchor_colour must be red or black, got {anchor_colour!r}")
    if terminal_reason == "cap":
        return SCORE_DRAW
    if winner not in ("red", "black"):
        raise ValueError(f"a win needs a winning colour, got {winner!r}")
    return SCORE_T1J_WIN if winner == anchor_colour else SCORE_T1J_LOSS


def hoeffding_interval(score: float, n: int, alpha: float = ALPHA):
    """THE PRIMARY INTERVAL. Valid for INDEPENDENT outcomes bounded in [0, 1].

    Hoeffding: P(|Xbar - mu| >= t) <= 2 exp(-2 n t^2), so the half-width at
    confidence 1 - alpha is sqrt(ln(2/alpha) / (2n)).

    It assumes independence and boundedness and NOTHING ELSE -- not a Bernoulli
    outcome, not identical distributions. Both matter here: a game scores
    1.0/0.5/0.0, and the eight openings are eight different distributions.

    Coverage is at least 1 - alpha GIVEN INDEPENDENCE, which is the claim Wilson
    could not support even given independence. Independence itself is MODELLED,
    not established: distinct derived streams rule out accidental reuse, and
    nothing here rules out dependence through fixed consecutive seeds or through
    T1j's own unseeded per-process randomness.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 <= score <= n:
        raise ValueError(f"score {score} outside [0, {n}]")
    p = score / n
    t = math.sqrt(math.log(2.0 / alpha) / (2.0 * n))
    return (max(0.0, p - t), min(1.0, p + t))


def wilson_interval(score: float, n: int, z: float = Z_95):
    """SECONDARY, NOMINAL, APPROXIMATE. Wilson score interval for `score / n`.

    Deterministic: no generator, no resampling, no seed. At score 0 the lower
    bound is exactly 0 and at score n the upper bound is exactly 1.

    NOT a guaranteed 95% interval, and never described as conservative. It assumes
    Bernoulli trials; a TwixT game is not one, the openings are not identically
    distributed, and Wilson's own coverage oscillates below nominal regardless --
    87.97% at n = 64, p = 0.002. Reported alongside `hoeffding_interval` because
    it is the conventional figure and far tighter, not because it is valid here.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 <= score <= n:
        raise ValueError(f"score {score} outside [0, {n}]")
    p = score / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    lo, hi = centre - half, centre + half
    # In exact arithmetic centre == half at p = 0, so lo is 0; in floating point
    # it lands a few ulps away (3.5e-18 was observed). The docstring above
    # promises exactness, so make it exact rather than let the claim be false by
    # a rounding error. Same at the top end by reflection.
    if score == 0:
        lo = 0.0
    if score == n:
        hi = 1.0
    return (max(0.0, lo), min(1.0, hi))


def variance_deficit(draws: int, n: int) -> float:
    """DESCRIPTIVE ONLY. How much less variance the OBSERVED mix has.

    Exactly 0.25 * draws / n^2 on the rate scale, computed from the OBSERVED draw
    count -- which is not the true draw probability, and this is not a coverage
    statement. It described a real property and was then used to support a
    conservativeness claim it cannot support; it is kept, and demoted.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return 0.25 * draws / (n * n)


#: THE FROZEN SCHEDULE IDENTITY. It lives here, in the rules layer, rather than
#: in l0_match_plan, for one reason: `bind_results` must verify it, and the plan
#: module imports these rules, so the rules module cannot import the plan back.
#: Putting the digest above the reporter left the reporter unable to check it --
#: and a reporter that cannot check the schedule will report on any schedule.
L0_TASK_DIMENSIONS = ("task_id", "endpoint", "t1j_mdPly", "t1j_mdFixedPly", "opening",
                      "colour_arm", "rep", "anchor_colour", "reference",
                      "reference_sha1", "reference_colour", "seed")
L0_TASK_DIGEST = "193d66bf5f1e4dca52dd967530182566176466ef8ff79d5151b403b2634085bc"


def l0_task_digest(tasks: Sequence[Dict[str, Any]]) -> str:
    """Ordered, dimension-projected sha256 over the schedule.

    Deliberately separate from `e4_screen_runner.task_digest`, whose dimension
    tuple is pinned by the canonical screen's published digest: widening that to
    take an argument would put a frozen artifact's identity behind a parameter.
    """
    try:
        payload = json.dumps([[t[k] for k in L0_TASK_DIMENSIONS] for t in tasks],
                             separators=(",", ":"))
    except KeyError as e:
        raise ValueError(f"task is missing dimension {e}") from None
    return hashlib.sha256(payload.encode()).hexdigest()


#: Fields a task_result must carry, and the task dimensions it must agree with.
RESULT_FIELDS = ("task_id", "winner", "terminal_reason", "t1j_points", "plies", "seed")
BOUND_DIMENSIONS = ("seed", "endpoint", "t1j_mdPly", "anchor_colour")


def validate_result(row: Dict[str, Any], task: Dict[str, Any]) -> None:
    """Bind ONE result to ITS canonical task, and check the outcome is possible.

    An earlier version checked only that the 64 task_ids were present. Sixty-four
    canonical names carrying t1j_points = 0.25, plies = -7 and invented openings
    were accepted and turned into a rate and an interval. A name is not a result.

    Note what is NOT read off the row: opening, colour_arm and rep come from the
    CANONICAL TASK, so a result cannot misreport which cell it belongs to. The
    screen's own records do not carry those fields at all.
    """
    missing = [f for f in RESULT_FIELDS if f not in row]
    if missing:
        raise ValueError(f"{row.get('task_id')}: result is missing {missing}")
    for dim in BOUND_DIMENSIONS:
        if dim in row and row[dim] != task[dim]:
            raise ValueError(f"{task['task_id']}: result {dim}={row[dim]!r} != "
                             f"scheduled {task[dim]!r}")
    if row["terminal_reason"] not in TERMINAL_REASONS:
        raise ValueError(f"{task['task_id']}: terminal_reason "
                         f"{row['terminal_reason']!r} not in {TERMINAL_REASONS}")
    if row["winner"] not in WINNERS:
        raise ValueError(f"{task['task_id']}: winner {row['winner']!r} not in {WINNERS}")
    plies = row["plies"]
    if not isinstance(plies, int) or isinstance(plies, bool):
        raise ValueError(f"{task['task_id']}: plies {plies!r} is not an integer")
    if not 1 <= plies <= PLY_CAP:
        raise ValueError(f"{task['task_id']}: plies {plies} outside [1, {PLY_CAP}]")
    # THE PRODUCTION ORDERING, matched exactly. e4_screen_runner checks
    # state.winner() BEFORE the cap, so a natural win ON ply 280 is a WIN, not a
    # cap; and a cap records winner=None. An earlier version equated plies ==
    # PLY_CAP with terminal_reason == "cap", which rejected the legitimate win at
    # the cap and accepted an impossible cap with a winning colour.
    if row["terminal_reason"] == "cap":
        if plies != PLY_CAP:
            raise ValueError(f"{task['task_id']}: a cap termination is recorded only at "
                             f"ply {PLY_CAP}, got {plies}")
        if row["winner"] is not None:
            raise ValueError(f"{task['task_id']}: a cap termination has no winner, got "
                             f"{row['winner']!r}; the runner records winner=None")
    else:                                            # "win"
        if row["winner"] not in ("red", "black"):
            raise ValueError(f"{task['task_id']}: a win requires a winning colour, got "
                             f"{row['winner']!r}")
    # TYPE-SAFE AND FAIL-CLOSED. An earlier version compared float(got) to the
    # expected score, which accepted True (float(True) == 1.0) and the string
    # "1.0" as wins, and raised an uncaught TypeError on None or a list -- a
    # traceback instead of a refused report. bool is checked FIRST because
    # isinstance(True, int) is True in Python.
    got = row["t1j_points"]
    if isinstance(got, bool):
        raise ValueError(f"{task['task_id']}: t1j_points {got!r} is a bool, not a score")
    if not isinstance(got, (int, float)):
        raise ValueError(f"{task['task_id']}: t1j_points {got!r} is "
                         f"{type(got).__name__}, not a JSON number")
    if not math.isfinite(got):
        raise ValueError(f"{task['task_id']}: t1j_points {got!r} is not finite")
    expected = score_game(row["winner"], task["anchor_colour"], row["terminal_reason"])
    if float(got) != expected:
        raise ValueError(f"{task['task_id']}: t1j_points {got!r} != {expected} implied by "
                         f"winner={row['winner']!r}, anchor={task['anchor_colour']!r}, "
                         f"terminal_reason={row['terminal_reason']!r}")


def bind_results(results: Sequence[Dict[str, Any]], tasks: Sequence[Dict[str, Any]]):
    """Pair every result with its canonical task, or refuse. Returns (pairs, reason).

    THE FAIL-CLOSED BOUNDARY IS HERE, AROUND THE WHOLE BINDING PATH, not merely
    around `validate_result`. An earlier version guarded only the per-result call,
    so malformed shapes escaped as tracebacks BEFORE reaching it:

      * a None result           -> AttributeError at r.get("task_id")
      * a non-dict task         -> TypeError while building by_id
      * task_ids None + "alien" -> TypeError sorting heterogeneous values

    Each is now refused explicitly, with a message naming the offending position,
    and the blanket below is the last resort for whatever has not been thought of.
    """
    try:
        return _bind_results(results, tasks)
    except Exception as e:                               # noqa: BLE001
        return None, f"malformed input rejected ({type(e).__name__}: {e})"


def _shape_errors(results, tasks):
    """Container and identity TYPES, checked before any comprehension or set op.

    Everything downstream indexes, hashes and sorts these values, and all three
    raise on the wrong type. Validating shape first keeps the refusal message
    actionable instead of leaving it to the blanket.
    """
    if not isinstance(tasks, (list, tuple)):
        return f"the schedule is {type(tasks).__name__}, expected a list of tasks"
    if not isinstance(results, (list, tuple)):
        return f"the results are {type(results).__name__}, expected a list of records"
    for i, t in enumerate(tasks):
        if not isinstance(t, dict):
            return f"schedule entry {i} is {type(t).__name__}, expected an object"
        if not isinstance(t.get("task_id"), str):
            return (f"schedule entry {i} has task_id {t.get('task_id')!r}, "
                    f"expected a string")
    for i, r in enumerate(results):
        if not isinstance(r, dict):
            return f"result {i} is {type(r).__name__}, expected an object"
        if not isinstance(r.get("task_id"), str):
            return f"result {i} has task_id {r.get('task_id')!r}, expected a string"
    return None


def _bind_results(results, tasks):
    shape = _shape_errors(results, tasks)
    if shape is not None:
        return None, shape
    # Past this point every task_id is a string, so the set operations and the
    # sorts below cannot raise on mixed types.
    by_id = {t["task_id"]: t for t in tasks}
    if len(by_id) != len(tasks):
        return None, "the schedule contains duplicate task identities"
    if len(tasks) != N_GAMES:
        return None, f"the design names {len(tasks)} tasks, expected {N_GAMES}"
    # THE SCHEDULE ITSELF MUST BE THE FROZEN ONE. Without this, a caller could
    # hand over a task list with an opening renamed, supply matching results, and
    # get a report about an invented cell -- reproduced, and the reason this
    # check is here rather than only in the plan loader.
    try:
        digest = l0_task_digest(tasks)
    except ValueError as e:
        return None, str(e)
    if digest != L0_TASK_DIGEST:
        return None, (f"the schedule is not the frozen L0 design: task digest {digest} "
                      f"!= pinned {L0_TASK_DIGEST}")
    got = [r["task_id"] for r in results]
    if len(set(got)) != len(got):
        return None, "a task identity appears more than once in the results"
    alien = sorted(set(got) - set(by_id))
    if alien:
        return None, f"results contain identities not in the design: {alien[:3]}"
    missing = sorted(set(by_id) - set(got))
    if missing:
        return None, f"{len(missing)} game(s) unplayed: {missing[:3]}"
    pairs = [(r, by_id[r["task_id"]]) for r in results]
    for row, task in pairs:
        try:
            validate_result(row, task)
        except ValueError as e:
            return None, str(e)
        except Exception as e:                       # noqa: BLE001
            return None, (f"{task['task_id']}: malformed result rejected "
                          f"({type(e).__name__}: {e})")
    cells: Dict[Any, int] = {}
    for _, task in pairs:
        key = (task["opening"], task["colour_arm"])
        cells[key] = cells.get(key, 0) + 1
    if len(cells) != N_OPENINGS * N_ARMS:
        return None, f"{len(cells)} opening/colour cells, expected {N_OPENINGS * N_ARMS}"
    bad = {k: v for k, v in cells.items() if v != N_REPS}
    if bad:
        return None, f"cells without exactly {N_REPS} results: {sorted(bad.items())[:3]}"
    return pairs, None


def _cell_pairs(pairs, key):
    """Group by a CANONICAL TASK field, never by a field read off the result."""
    out: Dict[Any, List[Any]] = {}
    for row, task in pairs:
        out.setdefault(task[key], []).append((row, task))
    return out


def _summary(pairs):
    rows = [r for r, _ in pairs]
    score = sum(float(r["t1j_points"]) for r in rows)
    caps = sum(1 for r in rows if r["terminal_reason"] == "cap")
    plies = [int(r["plies"]) for r in rows]
    return {
        "games": len(rows),
        "t1j_score": score,
        "t1j_rate": (score / len(rows)) if rows else None,
        "cap_terminations": caps,
        "plies_min": min(plies) if plies else None,
        "plies_median": sorted(plies)[len(plies) // 2] if plies else None,
        "plies_max": max(plies) if plies else None,
    }


def match_report(results: Sequence[Dict[str, Any]], tasks: Sequence[Dict[str, Any]]):
    """The WHOLE report, or a refusal. Takes the CANONICAL TASKS, not just names.

    Two inferential numbers, in order of authority:
      * `ci95_hoeffding` -- PRIMARY, valid for independent bounded outcomes.
      * `ci95_wilson`    -- NOMINAL and APPROXIMATE, reported for convention.

    Everything else is DESCRIPTIVE: the per-opening and per-colour breakdowns have
    8 and 32 games behind them, carry no interval, and support no comparison
    between cells. They exist to show the shape of the result.
    """
    pairs, why = bind_results(results, tasks)
    if pairs is None:
        return {"reported": False, "reason": why}

    overall = _summary(pairs)
    caps = overall["cap_terminations"]
    if caps > CAP_NO_RATE_THRESHOLD:
        # PREREGISTERED, and applied only AFTER all 64 games are played: caps never
        # stop the match. A rate over mostly-unresolved games measures the cap.
        return {"reported": False, "outcome": "CAP_SATURATED_NO_RATE",
                "reason": (f"{caps} of {N_GAMES} games terminated at the ply cap, more "
                           f"than the preregistered threshold of {CAP_NO_RATE_THRESHOLD}; "
                           f"the positions did not resolve and there is no rate to report"),
                "cap_terminations": caps, "games": overall["games"]}

    score = overall["t1j_score"]
    hl, hh = hoeffding_interval(score, N_GAMES)
    wl, wh = wilson_interval(score, N_GAMES)
    overall.update({
        "ci95_hoeffding": [hl, hh],
        "estimand": ESTIMAND,
        "ci95_hoeffding_method": (
            "PRIMARY. Hoeffding, closed form, half-width sqrt(ln(2/alpha)/(2n)), "
            "alpha = %.15g. A 95%% bound UNDER AN INDEPENDENCE MODEL. It needs no "
            "Bernoulli outcome and no identical distributions, but it does assume "
            "independence, which is MODELLED here and not established: distinct "
            "derived streams rule out accidental stream reuse only." % ALPHA),
        "independence_is_modelled_not_measured": INDEPENDENCE_CAVEAT,
        "ci95_wilson": [wl, wh],
        "ci95_wilson_method": (
            "SECONDARY, NOMINAL, APPROXIMATE. Wilson score interval, z = %.15g. Assumes "
            "Bernoulli trials, which a TwixT game is not, and its coverage dips below "
            "nominal on its own (87.97%% at n=64, p=0.002). NOT conservative and not "
            "described as such." % Z_95),
        "variance_deficit_descriptive": variance_deficit(caps, N_GAMES),
        "cap_warning": caps > 0,
    })
    return {
        "reported": True,
        "overall": overall,
        "by_colour_arm": {k: _summary(v) for k, v in
                          sorted(_cell_pairs(pairs, "colour_arm").items())},
        "by_opening": {k: _summary(v) for k, v in
                       sorted(_cell_pairs(pairs, "opening").items())},
        "descriptive_only": ["by_colour_arm", "by_opening"],
        "bounded_to": (
            "these 8 frozen openings, this colour balance, mdPly 6, "
            "calib020_0001 at 400 simulations, ply cap 280, one run of 64 games"),
        "forbidden_claims": FORBIDDEN_CLAIMS,
    }


#: The quantity being estimated, written down before any data exists.
ESTIMAND = (
    "the equally weighted mean, over the 16 opening/colour cells, of T1j's expected "
    "score against calib020_0001 at mdPly 6, the expectation being over ENGINE "
    "RANDOMNESS ONLY. The design is balanced (4 games x 16 cells), so the plain mean "
    "of the 64 scores is that equally weighted cell mean. It is NOT the expected score "
    "against TwixT openings in general: the 8 openings were fixed in advance and are "
    "part of the estimand, not a sample from a population of openings.")

#: Repeated in every report, because the interval is conditional on it.
INDEPENDENCE_CAVEAT = (
    "the 64 games are MODELLED as independent; this is an assumption, not a "
    "measurement. What was verified is that the tasks derive 128 distinct generator "
    "streams that collide with no stream used before, which rules out accidental "
    "stream REUSE only. The seeds are fixed consecutive integers, the derivation is a "
    "fixed XOR, and T1j seeds its own Zobrist table from an unseeded Random per "
    "process, which this design neither controls nor observes.")

#: Stated in the rules, not only in the card, so a reporting script cannot claim
#: what the protocol forbids without editing committed code.
FORBIDDEN_CLAIMS = (
    "any Elo figure, or any conversion of this rate into one",
    "any absolute strength placement -- T1j is uncalibrated, so this is an "
    "ORDERING against calib020_0001 in this stack, not a placement",
    "any generalisation beyond these 8 openings or beyond mdPly 6",
    "any per-cell comparison presented as a finding: 8 games per opening and 32 "
    "per colour arm carry no interval here and were not preregistered as tests",
    "any claim that a wider or narrower interval would follow from more games "
    "not actually played",
    "any description of the Wilson interval as conservative, exact or guaranteed; "
    "it is nominal, and hoeffding_interval is the primary",
    "any re-run, pooled or otherwise, of this design on these seeds",
    "any statement that the 64 games ARE independent, or that independence was "
    "verified; distinct streams rule out reuse, not dependence",
)
