"""v18 preflight sizing -- Task 8 Step 3.

Three things live here and nothing else: the v18 role predicates resolved at a
measured cutoff, the tier-ladder DRIVER, and the record emitter.

**No selection algorithm.** The exact selector is
`fpu_dev_corpus_v2.sample_v2_rows` under the schema-5 profile Task 8 Step 1
added, called once per trial. A second selector here would be a second answer
to "does this geometry fill", and the two would drift.

**Why the trial loop lives here** (plan revision 36, items 129-130). Role
assignment is PER RESERVOIR: step 2's representative quota is drawn before the
selector runs, and spec Sec 9.2.2's exact per-role side geometry is verified
after it. `sizing_analysis_core` takes rows that are already role-labelled and
decides success internally, so it can express neither -- and a quota drawn once
over the whole census leaves ~4 representatives in a 200-game subset, failing
every tier. The loop here is the v2 repair's discipline -- whole games, the same
seeded key construction, 299 trials, the imported exact Clopper-Pearson rule --
around those two extra conditions.

**No restated thresholds.** Every number comes from `v18_preflight_criteria`.
The role predicates are `criteria.classify_role`, called -- not copied: the plan
records that revision 3 restated the flip-control rule here as "and/or" and
weakened a frozen AND in the process.

Nothing in this module measures anything. It consumes an already-measured
census and emits a record that forbids scientific interpretation.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from . import v18_preflight_criteria as criteria
from .fpu_dev_reservoir_protocol import canonical_json_bytes
from .fpu_dev_corpus_v2 import (
    MAX_PER_GAME,
    MIN_PLY_GAP,
    PHASES,
    SIDE_TOL,
    V18_SCHEMA,
    _binomial_lower_bound,
    allocation_for_schema,
    band_of,
    parse_allocation_profile,
    post_screen_qualification_report,
    roles_for_schema,
    sample_v2_rows,
)

# The frozen schema-5 run kind. Named once; `parse_allocation_profile` refuses
# any other value under schema 5, so a drift here fails at the profile.
RUN_KIND = "v18_preflight_sizing"

# `classify_role`'s labels are Task 5's vocabulary; the corpus cells are Task
# 8's. Asserted equal below so the two cannot drift into a silent mis-cell.
ROLE_TO_CELL = {
    "target": "target",
    "representative": "representative",
    "identity": "identity_witness",
    "flip": "flip_control",
}
assert set(ROLE_TO_CELL.values()) == set(roles_for_schema(V18_SCHEMA)), ROLE_TO_CELL
assert set(ROLE_TO_CELL) | {"unassigned"} == set(criteria.ROLE_LABELS), ROLE_TO_CELL

# The sentinel `role_predicates` must resolve. A rename that leaves it
# unresolved would ship a predicate comparing exposure against a STRING.
_CUTOFF_SENTINEL = "EXPOSURE_CUTOFF"


# ---------------------------------------------------------------------------
# The exposure cutoff -- derived from the matched cohort, never from A rows.
# ---------------------------------------------------------------------------

def matched_control_rows(cohort: Sequence[Dict],
                         census_rows: Sequence[Dict]) -> List[Dict]:
    """The cohort's 30 identities resolved back to their CENSUS rows.

    The matched-cohort artifact deliberately does not carry exposure: every
    exposure column is in `v18_cohort_matcher.FORBIDDEN_ROW_FIELDS` and is
    projected away before any cost is computed, so selecting on it cannot fit
    the threshold to its own sample. The join is therefore mandatory, and its
    key is the frozen one -- game CONTENT sha1 plus ply, never `game_idx`,
    which is reservoir-local.
    """
    by_identity = {(r["game_content_sha1"], r["position_ply"]): r
                   for r in census_rows}
    resolved = []
    for row in cohort:
        key = (row["game_content_sha1"], row["position_ply"])
        if key not in by_identity:
            raise ValueError(
                f"matched cohort row {key} is not in the census: the cohort and "
                f"the census do not describe one measurement")
        found = by_identity[key]
        if found["canonical_state_sha1"] != row["canonical_state_sha1"]:
            raise ValueError(
                f"cohort row {key} has canonical state "
                f"{row['canonical_state_sha1']} but the census row at that "
                f"identity is {found['canonical_state_sha1']}")
        resolved.append(found)
    return resolved


def exposure_cutoff(control_rows: Sequence[Dict]) -> float:
    """EXPOSURE_CUTOFF: the nearest-rank 0.90 quantile of primary exposure over
    the matched cohort. The quantile itself is `criteria.nearest_rank_quantile`,
    so the no-interpolation rule has exactly one implementation and the result
    is always an OBSERVED datum -- a cutoff no row attains selects nothing.
    """
    rule = criteria.EXPOSURE_CUTOFF_RULE
    if rule["uses_a_rows"] is not False:
        raise ValueError("EXPOSURE_CUTOFF_RULE no longer excludes A rows")
    offenders = sorted({r["population"] for r in control_rows
                        if r.get("population") == "selected_a"})
    if offenders:
        raise ValueError(
            "A rows reached the exposure cutoff population: the cutoff is "
            "derived from the matched control cohort only (spec Sec 2.2.3); A "
            "rows establish reach, never a threshold")
    if len(control_rows) != rule["n"]:
        raise ValueError(
            f"exposure cutoff needs exactly {rule['n']} matched control rows, "
            f"got {len(control_rows)}")
    # The frozen total ordering, applied before the quantile so a reordered
    # input cannot move a tie.
    ordered = sorted(control_rows,
                     key=lambda r: tuple(r[k] for k in rule["deterministic_ordering"]))
    return criteria.nearest_rank_quantile(
        [r[rule["statistic"]] for r in ordered], rule["quantile"])


# ---------------------------------------------------------------------------
# Role predicates and the role PARTITION.
# ---------------------------------------------------------------------------

def role_predicates(cutoff: float) -> Dict[str, Any]:
    """The frozen `ROLE_ASSIGNMENT` with `EXPOSURE_CUTOFF` resolved.

    Nothing else is transformed. This is the object the record publishes, so a
    reader can see the exact predicate set the classification ran under.
    """
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)):
        raise ValueError(f"exposure cutoff must be numeric, got {cutoff!r}")
    resolved: Dict[str, Any] = {}
    substitutions = 0
    for name, spec in criteria.ROLE_ASSIGNMENT["roles"].items():
        conditions = []
        for condition in spec["conditions"]:
            condition = dict(condition)
            if condition.get("value") == _CUTOFF_SENTINEL:
                condition["value"] = float(cutoff)
                substitutions += 1
            conditions.append(condition)
        resolved[name] = {"step": spec["step"], "conditions": tuple(conditions),
                          "uses_absolute_residual_magnitude":
                              spec["uses_absolute_residual_magnitude"]}
    if substitutions != 1:
        raise ValueError(
            f"expected exactly one {_CUTOFF_SENTINEL} placeholder in the frozen "
            f"role conditions, resolved {substitutions}: an unresolved "
            f"placeholder would compare exposure against a string")
    return {
        "exposure_cutoff": float(cutoff),
        "order": criteria.ROLE_ASSIGNMENT["order"],
        "on_shortfall": criteria.ROLE_ASSIGNMENT["on_shortfall"],
        "revisit_representatives_after_residual_roles":
            criteria.ROLE_ASSIGNMENT["revisit_representatives_after_residual_roles"],
        "roles": resolved,
        # IMPORTED whole, never restated: the operator is AND.
        "flip_control_exposure": criteria.FLIP_CONTROL_EXPOSURE,
        "identity_witness": criteria.IDENTITY_WITNESS,
        "representative_ordering_key": _REPRESENTATIVE_ORDER_KEY,
    }


# Residual-INDEPENDENT by construction: a canonical state hash cannot be
# computed from a residual. Ordering representatives by anything measured would
# condition the sample on the very quantity they control for.
_REPRESENTATIVE_ORDER_KEY = "canonical_state_sha1"


class RepresentativeShortfall(ValueError):
    """A reservoir cannot fill the representative quota.

    The ONE condition the trial loop is allowed to convert into a failed trial.
    Everything else -- a missing census field, a non-monotone clip series -- is
    corrupt EVIDENCE, and counting it as a failed trial would move the measured
    sizing probability instead of invalidating the measurement.
    """


def validate_census_rows(rows: Sequence[Dict]) -> None:
    """Every row carries the fields the predicates read, and clip counts that
    are physically possible. Run ONCE, before any sampling."""
    for row in rows:
        missing = sorted(criteria.REQUIRED_CENSUS_FIELDS - set(row))
        if missing:
            raise ValueError(f"census row is missing required field(s): {missing}")
        _assert_cap_monotone(row)


def _assert_cap_monotone(row: Dict) -> None:
    """A stronger cap can never clip FEWER leaves than a weaker one.

    Clipping is strict at `abs(residual) > cap`, so `would_clip` is
    non-increasing in the cap. This is what makes both step-3 predicates mean
    what they claim: `would_clip_0.5 == 0` is byte-identity at EVERY grid cap,
    and flip exposure at 1.25 is exposure at every stronger cap. A census that
    violates it is corrupt, and accepting it would make both claims vacuous.
    """
    counts = [(cap, row[f"would_clip_{cap}"]) for cap in
              sorted((str(c) for c in criteria.CAP_GRID), key=float)]
    for (weak_cap, weak), (strong_cap, strong) in zip(counts[1:], counts):
        if strong < weak:
            raise ValueError(
                f"census row {row.get('canonical_state_sha1')} clips {strong} "
                f"leaves at cap {strong_cap} but {weak} at the weaker cap "
                f"{weak_cap}: clipping is monotone in the cap, so this row is "
                f"corrupt")


# Exact per-role side geometry, spec Sec 9.2.2. DERIVED from the frozen
# allocation -- each role's rows split evenly -- and pinned against the spec's
# own literal table below, so neither can drift without the other.
def _exact_role_sides() -> Dict[str, Tuple[int, int]]:
    totals: Dict[str, int] = {}
    for (role, _phase), counts in allocation_for_schema(V18_SCHEMA).items():
        totals[role] = totals.get(role, 0) + sum(counts.values())
    sides = {}
    for role, total in totals.items():
        if total % 2:
            raise ValueError(f"role {role} has {total} rows: exact side balance "
                             f"is impossible on an odd total")
        sides[role] = (total // 2, total // 2)
    return sides


ROLE_SIDE_GEOMETRY = _exact_role_sides()
# Spec Sec 9.2.2, verbatim. A derivation that silently stopped matching the
# spec would be a derivation of the wrong thing.
assert ROLE_SIDE_GEOMETRY == {"target": (8, 8), "identity_witness": (2, 2),
                              "flip_control": (2, 2),
                              "representative": (8, 8)}, ROLE_SIDE_GEOMETRY


def assert_role_side_balance(manifest: Sequence[Dict]) -> None:
    """EXACT per-role red/black balance on the produced manifest.

    The profile's `side_tol` constrains only the single split's AGGREGATE count,
    so a manifest can be 20/20 overall while every role inside it is badly
    skewed -- 12/4 targets against 4/12 representatives passes `side_tol` and
    fails spec Sec 9.2.2. This is the binding rule; the aggregate follows from
    it, never the other way round.
    """
    for role, (red, black) in sorted(ROLE_SIDE_GEOMETRY.items()):
        rows = [r for r in manifest if r["role"] == role]
        actual = (sum(1 for r in rows if r["side"] == "red"),
                  sum(1 for r in rows if r["side"] == "black"))
        if actual != (red, black):
            raise ValueError(
                f"role side balance: {role} is {actual[0]} red / {actual[1]} "
                f"black, spec Sec 9.2.2 requires {red}/{black}")


def _draw_representatives(candidates: Sequence[Dict]) -> List[Dict]:
    """Step 2's QUOTA: exactly four per phase, side-steered, canonical order.

    `representative_selected` is the OUTCOME of this draw, not a synonym for
    eligibility. Labelling every candidate a representative would consume every
    near-even non-target row and leave identity and flip drawing from nothing
    but rows the near-even rule excludes -- which measures a role geometry the
    corpus will never have.

    The candidate set is conditioned on target status and near-evenness and on
    NOTHING residual, so the ordering key leads with the canonical state hash:
    ordering by anything measured would condition the sample on the quantity
    these rows exist to control for.

    The draw carries the CORPUS GEOMETRY -- `MAX_PER_GAME` and `MIN_PLY_GAP`.
    The sixteen rows drawn here are the only representatives the selector will
    ever see, with zero slack, so a draw that ignores the per-game cap or the
    ply separation hands it an inadmissible set and guarantees a failure that
    says nothing about the reservoir.

    Side steering chooses among equally admissible candidates; the exact 8/8
    requirement is verified on the manifest, never assumed here.
    """
    quota = {phase: counts["all"] for (role, phase), counts
             in allocation_for_schema(V18_SCHEMA).items() if role == "representative"}
    by_phase: Dict[str, List[Dict]] = {phase: [] for phase in quota}
    for row in candidates:
        if row["phase"] in by_phase:
            by_phase[row["phase"]].append(row)
    drawn: List[Dict] = []
    sides = {"red": 0, "black": 0}
    taken_plies: Dict[str, List[int]] = {}          # per game, across ALL phases
    for phase in PHASES:
        if phase not in quota:
            continue
        pool = sorted(by_phase[phase], key=_representative_order_key)
        taken = 0
        while taken < quota[phase]:
            # CENSUS vocabulary: the side field is `side_to_move`. `side` is the
            # SELECTOR row's key, and these rows have not been mapped yet.
            behind = "red" if sides["red"] <= sides["black"] else "black"
            admissible = [r for r in pool if _fits_representative_geometry(r, taken_plies)]
            pick = next((r for r in admissible if r["side_to_move"] == behind), None)
            if pick is None:
                pick = admissible[0] if admissible else None
            if pick is None:
                raise RepresentativeShortfall(
                    f"representative shortfall: phase {phase} offers "
                    f"{len(by_phase[phase])} candidate(s), none of them still "
                    f"admissible, against a quota of {quota[phase]}. The frozen "
                    f"rule on a shortfall is STOP")
            pool.remove(pick)
            sides[pick["side_to_move"]] += 1
            taken_plies.setdefault(pick["game_content_sha1"], []).append(
                pick["position_ply"])
            drawn.append(pick)
            taken += 1
    return drawn


def _representative_order_key(row: Dict) -> Tuple[str, str, int]:
    """A TOTAL order. A bare canonical-hash sort leaves ties to input order,
    which is not a deterministic rule -- and input order is whatever the census
    reader happened to yield."""
    return (row["canonical_state_sha1"], row["game_content_sha1"],
            row["position_ply"])


def _fits_representative_geometry(row: Dict, taken_plies: Dict[str, List[int]]) -> bool:
    """The frozen corpus geometry, applied while drawing: at most
    `MAX_PER_GAME` rows from one game, and no two of them within
    `MIN_PLY_GAP` plies."""
    plies = taken_plies.get(row["game_content_sha1"], ())
    if len(plies) >= MAX_PER_GAME:
        return False
    return all(abs(row["position_ply"] - ply) >= MIN_PLY_GAP for ply in plies)


def classify_rows(rows: Sequence[Dict],
                  predicates: Dict[str, Any]) -> Dict[str, Any]:
    """Assign every census row exactly one role, by the frozen ORDER.

    The order is what makes the roles a PARTITION -- the predicates alone
    overlap, and a row can satisfy both target exposure and flip exposure.

    Step 2 is a QUOTA, not a label: exactly sixteen representatives are drawn,
    four per phase, from a candidate set conditioned on target status and
    near-evenness alone. Everything the quota does not take falls through to
    step 3, so a surplus near-even row can still become an identity witness or
    a flip control.

    This runs on ONE candidate reservoir. The ladder calls it per trial, because
    "can this reservoir yield the corpus" is a question about the whole frozen
    procedure, not about a labelling computed once over the full census.
    """
    validate_census_rows(rows)
    return _assign_roles(rows, predicates)


def _assign_roles(rows: Sequence[Dict],
                  predicates: Dict[str, Any]) -> Dict[str, Any]:
    """`classify_rows` without the census validation.

    Split out for ONE reason: the ladder validates the whole census once, before
    sampling, and must not re-validate 3,600 rows on each of 2,093 trials -- nor
    silently turn a validation failure into a failed trial. See
    `validate_census_rows`.
    """
    cutoff = predicates["exposure_cutoff"]
    near_even_max = criteria.NEAR_EVEN["value"]
    by_role: Dict[str, List[Dict]] = {label: [] for label in criteria.ROLE_LABELS}

    # Step 1: targets, and the step-2 candidate set they define.
    roles = {id(row): criteria.classify_role(row, cutoff) for row in rows}
    candidates = [row for row in rows
                  if roles[id(row)] != "target"
                  and abs(row["root_value_stm"]) <= near_even_max]
    # Step 2: the quota. Its winners -- and only its winners -- are
    # representatives; the rest keep their step-3 classification.
    for row in _draw_representatives(candidates):
        roles[id(row)] = criteria.classify_role(row, cutoff,
                                                representative_selected=True)

    selector: List[Dict] = []
    for row in rows:
        role = roles[id(row)]
        by_role[role].append(row)
        if role != "unassigned":
            selector.append({
                # Game identity is the replay CONTENT hash. `game_idx` is
                # reservoir-local: it invents overlaps and misses renumbering.
                "game_idx": row["game_content_sha1"],
                "ply": row["position_ply"],
                "side": row["side_to_move"],
                "role": ROLE_TO_CELL[role],
                "phase": row["phase"],
                "band": band_of(row["n_legal"]),
                "canonical_sha1": row["canonical_state_sha1"],
            })
    assigned = sum(len(v) for v in by_role.values())
    if assigned != len(rows):
        raise ValueError(f"role assignment covered {assigned} of {len(rows)} rows")
    return {
        "exposure_cutoff": cutoff,
        "counts": {label: len(by_role[label]) for label in criteria.ROLE_LABELS},
        "by_role": by_role,
        "selector_rows": selector,
    }


# ---------------------------------------------------------------------------
# The tier ladder. Owns the trial loop; imports the exact selector and the
# exact binomial bound, and implements neither.
# ---------------------------------------------------------------------------

def schema5_profile():
    """The frozen 40-row v18 allocation, built through the real parser.

    Going through `parse_allocation_profile` rather than constructing an
    `AllocationProfile` directly is deliberate: the parser is where the frozen
    per-cell allocation, the single split and the v18 run identity are
    ENFORCED, so a drift in any of them fails here rather than three stages
    later inside a trial.
    """
    return parse_allocation_profile({
        "config_schema_version": V18_SCHEMA,
        "run_kind": RUN_KIND,
        "scientific_interpretation_forbidden": True,
        "phase_allocation": {f"{role}|{phase}": dict(counts) for (role, phase),
                             counts in allocation_for_schema(V18_SCHEMA).items()},
        "late_floors": {},
        "late_target_band_minima": {},
        "corpus_size": sum(sum(c.values()) for c
                           in allocation_for_schema(V18_SCHEMA).values()),
        # v2's shared constants, imported -- the same 2-per-game / 12-ply /
        # side-tolerance discipline `PER_GAME["future_corpus"]` records.
        "max_per_game": MAX_PER_GAME,
        "min_ply_gap": MIN_PLY_GAP,
        "side_tol": SIDE_TOL,
    }, source="v18_selector_sizing")


def sizing_ladder(rows: Sequence[Dict], predicates: Dict[str, Any], *,
                  all_game_ids: Sequence[str],
                  tiers: Sequence[int] = None, trials: int = None,
                  seed: int = None) -> List[Dict[str, Any]]:
    """Whole-game subsampling over the frozen tier ladder.

    `all_game_ids` is the COMPLETE universe from the frozen Task 4 record,
    including games that yielded zero classified rows -- dropping them would
    bias success upward. It is required rather than derived from `rows` for
    exactly that reason.

    Success is an exact-selector witness -- qualification PASS, a complete
    manifest, and the frozen per-role side geometry -- never a capacity bound.
    The full-universe tier is ONE degenerate trial, because drawing 800 from 800
    returns the same set every time.
    """
    sizing = criteria.SIZING
    tiers = tuple(sizing["probabilistic_tiers"]) + (sizing["degenerate_tier"],) \
        if tiers is None else tuple(tiers)
    trials = sizing["trials_per_probabilistic_tier"] if trials is None else trials
    seed = sizing["seed"] if seed is None else seed
    profile = schema5_profile()

    # ONCE, before any sampling. Corrupt evidence must invalidate the
    # measurement, not lower the estimated success rate by presenting itself as
    # a reservoir that failed to yield a corpus.
    validate_census_rows(rows)

    # Bucket by game ONCE. A row whose game is outside the frozen universe
    # raises here rather than being silently sampled.
    by_game: Dict[str, List[Dict]] = {gi: [] for gi in all_game_ids}
    for row in rows:
        by_game[row["game_content_sha1"]].append(row)
    games = sorted(by_game)

    ladder = []
    for count in tiers:
        if count > len(games):
            ladder.append({"n_games": count, "meets_criterion": False,
                           "skipped": f"only {len(games)} games in the universe"})
            continue
        # Drawing every game from the universe returns the same set each time.
        n_trials = 1 if count == len(games) else trials
        successes, reasons = 0, {}
        for index in range(n_trials):
            # The v2 repair's key construction, verbatim: whole games, one
            # seeded stream per (seed, tier, trial), so a ladder is reproducible
            # from the frozen seed alone.
            rng = random.Random(f"sizing:{seed}:{count}:{index}")
            subset = set(rng.sample(games, count))
            sub_rows = [r for gi in subset for r in by_game[gi]]
            reason = _run_trial(sub_rows, predicates, profile, seed)
            if reason is None:
                successes += 1
            else:
                reasons[reason] = reasons.get(reason, 0) + 1
        lower = _binomial_lower_bound(successes, n_trials, sizing["alpha"])
        entry = {
            "n_games": count,
            "n_trials": n_trials,
            "n_successes": successes,
            "success_rate": successes / n_trials,
            "lower_bound_95": lower,
            "meets_criterion": lower >= sizing["minimum_lower_bound"],
            "degenerate_full_universe": count == len(games),
            "failure_reasons": dict(sorted(reasons.items())),
        }
        entry["witness_trial_index"] = _witness_trial_index(entry)
        ladder.append(entry)
    return ladder


def _run_trial(sub_rows: Sequence[Dict], predicates: Dict[str, Any], profile,
               selection_seed: int):
    """One trial = the WHOLE frozen procedure on one candidate reservoir.

    Returns None on success, or the reason it failed. Success is an
    exact-selector witness that also satisfies the frozen role side geometry --
    a capacity bound is never a witness, and neither is a filled manifest whose
    roles are internally skewed.

    Only a `RepresentativeShortfall` is a trial FAILURE. The census was
    validated before sampling, so any other exception escaping here is a defect
    or corrupt evidence, and it propagates: no record, rather than a quietly
    depressed success rate.
    """
    try:
        classification = _assign_roles(sub_rows, predicates)
    except RepresentativeShortfall as exc:
        return f"classify: {_first_line(exc)}"
    selector_rows = classification["selector_rows"]
    report = post_screen_qualification_report(selector_rows, profile)
    if report["status"] != "PASS":
        return f"qualify: {report['binding_constraint']}"
    try:
        # THE one exact selector. This module implements no second one.
        manifest, _stats = sample_v2_rows(selector_rows, seed=selection_seed,
                                          alloc=profile)
    except ValueError as exc:
        return f"select: {_first_line(exc)}"
    try:
        assert_role_side_balance(manifest)
    except ValueError as exc:
        return f"side: {_first_line(exc)}"
    return None


def _first_line(exc: Exception) -> str:
    return str(exc).splitlines()[0][:120]


def _witness_trial_index(entry: Dict[str, Any]):
    """The reproducible successful draw, named by its TRIAL INDEX.

    Not the content-SHA prefix: a prefix can fail even when every random trial
    passes, so it cannot be promised as the witness. No re-run is needed to find
    one, because passing at the frozen trial count REQUIRES every trial to have
    succeeded -- a single failure drops the exact bound below
    `SIZING["minimum_lower_bound"]`, which
    `test_the_frozen_trial_count_is_the_smallest_that_can_pass` pins. The
    assertion below is what keeps that reasoning honest if the rule ever moves.
    """
    if not entry["meets_criterion"]:
        return None
    if entry["n_successes"] != entry["n_trials"]:
        raise ValueError(
            f"a tier passed with {entry['n_successes']} of {entry['n_trials']} "
            f"trials: trial 0 is no longer guaranteed to be a witness, so the "
            f"successful index must be recorded by the trial loop itself")
    return 0


def smallest_qualifying_tier(ladder: Sequence[Dict[str, Any]]):
    """The smallest PROBABILISTICALLY passing tier, or None.

    The degenerate full-universe tier is excluded: one trial cannot estimate a
    success probability, so it can serve as an operational size but never as the
    qualifying evidence.
    """
    passing = [t["n_games"] for t in ladder
               if t.get("meets_criterion") and not t.get("degenerate_full_universe")]
    return min(passing) if passing else None


def recommended_operational_size(ladder: Sequence[Dict[str, Any]]):
    """The next-tier-up margin rule: one tier above the smallest qualifying one.

    Returns None when nothing qualifies (SIZING_FAILS) and when the qualifying
    tier is already the largest -- a margin that does not exist is not reported
    as one.
    """
    smallest = smallest_qualifying_tier(ladder)
    if smallest is None:
        return None
    tiers = [t["n_games"] for t in ladder]
    above = [t for t in tiers if t > smallest]
    return min(above) if above else None


def operating_characteristics(ladder: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per COUNT GATE: what that gate's evidence can and cannot support.

    `successes_required_to_pass` is what turns a bare success count into an
    operating characteristic -- it says how much of the gate's budget the
    observed run actually consumed, and it is None for a gate that cannot pass
    at its trial count no matter what (the degenerate tier).
    """
    table: Dict[str, Any] = {}
    for tier in ladder:
        if "skipped" in tier:
            table[str(tier["n_games"])] = {"skipped": tier["skipped"]}
            continue
        required = _successes_required(tier["n_trials"])
        table[str(tier["n_games"])] = {
            "n_trials": tier["n_trials"],
            "n_successes": tier["n_successes"],
            "success_rate": tier["success_rate"],
            "lower_bound_95": tier["lower_bound_95"],
            "successes_required_to_pass": required,
            "shortfall": (None if required is None
                          else max(0, required - tier["n_successes"])),
            "meets_criterion": tier["meets_criterion"],
            "degenerate_full_universe": tier["degenerate_full_universe"],
        }
    return table


def _successes_required(n_trials: int):
    """Smallest k whose exact one-sided lower bound clears the frozen rule.

    Binary search, not a scan: the bound is non-decreasing in k (P(X >= k) falls
    as k rises), and the bound itself is O(n) per evaluation.
    """
    alpha = criteria.SIZING["alpha"]
    target = criteria.SIZING["minimum_lower_bound"]
    if _binomial_lower_bound(n_trials, n_trials, alpha) < target:
        return None                       # unattainable at this trial count
    lo, hi = 0, n_trials
    while lo < hi:
        mid = (lo + hi) // 2
        if _binomial_lower_bound(mid, n_trials, alpha) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---------------------------------------------------------------------------
# The record.
# ---------------------------------------------------------------------------

_REQUIRED_BINDINGS = ("criteria_sha1", "universe_sha1", "census_sha1",
                      "matched_cohort_sha1")

# Canonical lowercase hex. Length alone authenticates nothing: `"z" * 40` is
# forty characters and identifies no artifact that can ever exist.
_SHA1_RE = re.compile(r"\A[0-9a-f]{40}\Z")


def _reconcile_ladder(ladder: Sequence[Dict[str, Any]]) -> None:
    """RECOMPUTE every derived ladder field from the two primitive counts.

    Checking the tier names alone lets a hand-edited ladder publish
    `0/299 successes` with `meets_criterion: True` and a witness index, and the
    record would report SIZING_PASSES over it. Success rate, the exact
    Clopper-Pearson bound, the pass flag, the degenerate flag, the frozen trial
    count and the witness index are all functions of `(n_successes, n_trials,
    n_games)` -- so each is recomputed here and compared, never trusted.
    """
    sizing = criteria.SIZING
    frozen = list(sizing["probabilistic_tiers"]) + [sizing["degenerate_tier"]]
    tiers = [t["n_games"] for t in ladder]
    if tiers != frozen:
        raise ValueError(
            f"refusing to emit: the ladder covers {tiers}, the frozen ladder is "
            f"{frozen}. A record may not claim frozen coverage it does not have")
    for tier in ladder:
        where = f"tier {tier['n_games']}"
        degenerate = tier["n_games"] == sizing["degenerate_tier"]
        n, k = tier["n_trials"], tier["n_successes"]
        if not isinstance(n, int) or not isinstance(k, int) or not 0 <= k <= n:
            raise ValueError(
                f"refusing to emit: {where} reports {k} successes in {n} trials")
        expected_trials = 1 if degenerate else sizing["trials_per_probabilistic_tier"]
        if n != expected_trials:
            raise ValueError(
                f"refusing to emit: {where} ran {n} trials, the frozen count is "
                f"{expected_trials}")
        if tier["degenerate_full_universe"] is not degenerate:
            raise ValueError(
                f"refusing to emit: {where} claims degenerate="
                f"{tier['degenerate_full_universe']}; only the full-universe "
                f"tier is degenerate")
        reasons = tier["failure_reasons"]
        if (not isinstance(reasons, dict)
                or any(not isinstance(c, int) or isinstance(c, bool) or c <= 0
                       for c in reasons.values())
                or sum(reasons.values()) != n - k):
            raise ValueError(
                f"refusing to emit: {where} reports {k} of {n} successes but "
                f"failure reasons totalling {sum(reasons.values()) if isinstance(reasons, dict) else reasons!r}; "
                f"every failed trial has exactly one recorded reason")
        expected = {
            "success_rate": k / n,
            "lower_bound_95": _binomial_lower_bound(k, n, sizing["alpha"]),
        }
        for field, value in expected.items():
            if not math.isclose(tier[field], value, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    f"refusing to emit: {where} reports {field}={tier[field]}, "
                    f"recomputing it from {k}/{n} gives {value}")
        passes = expected["lower_bound_95"] >= sizing["minimum_lower_bound"]
        if tier["meets_criterion"] is not passes:
            raise ValueError(
                f"refusing to emit: {where} claims meets_criterion="
                f"{tier['meets_criterion']}, but a lower bound of "
                f"{expected['lower_bound_95']} from {k}/{n} gives {passes}")
        witness = 0 if passes else None
        if tier["witness_trial_index"] != witness:
            raise ValueError(
                f"refusing to emit: {where} names witness trial "
                f"{tier['witness_trial_index']}, the pass state implies "
                f"{witness}")


def emit_sizing_record(path: str, *, cutoff: float, classification: Dict[str, Any],
                       ladder: Sequence[Dict[str, Any]],
                       predicates: Dict[str, Any],
                       bindings: Dict[str, str]) -> str:
    """Write the canonical sizing record; return its SHA-1.

    Every input is bound by SHA-1, and every published number is RECONCILED
    against the objects it claims to summarize before a byte is written -- a
    record is evidence only if it cannot describe a run that did not happen.
    """
    missing = sorted(set(_REQUIRED_BINDINGS) - set(bindings))
    if missing:
        raise ValueError(f"missing input binding(s): {missing}")
    for name in _REQUIRED_BINDINGS:
        value = bindings[name]
        if not isinstance(value, str) or not _SHA1_RE.match(value):
            raise ValueError(
                f"{name} must be a canonical lowercase hex sha1, got {value!r}")
    if classification["exposure_cutoff"] != cutoff:
        raise ValueError(
            f"refusing to emit: the classification ran at cutoff "
            f"{classification['exposure_cutoff']}, the record claims {cutoff}")
    if predicates["exposure_cutoff"] != cutoff:
        raise ValueError(
            f"refusing to emit: the predicates resolve cutoff "
            f"{predicates['exposure_cutoff']}, the record claims {cutoff}")
    _reconcile_ladder(ladder)

    smallest = smallest_qualifying_tier(ladder)
    payload = {
        "run_kind": RUN_KIND,
        "scientific_interpretation_forbidden": True,
        "scope_boundary": criteria.SCOPE_BOUNDARY,
        "exposure_cutoff": cutoff,
        "exposure_cutoff_rule": criteria.EXPOSURE_CUTOFF_RULE,
        "role_predicates": predicates,
        "role_counts": classification["counts"],
        "n_classified_rows": sum(classification["counts"].values()),
        "n_selector_rows": len(classification["selector_rows"]),
        "sizing": criteria.SIZING,
        "profile": schema5_profile().fingerprint(),
        "ladder": list(ladder),
        "operating_characteristics": operating_characteristics(ladder),
        "smallest_qualifying_tier": smallest,
        "recommended_operational_size": recommended_operational_size(ladder),
        "sizing_status": "SIZING_PASSES" if smallest is not None else "SIZING_FAILS",
        **{name: bindings[name] for name in _REQUIRED_BINDINGS},
    }
    raw = canonical_json_bytes(payload)
    Path(path).write_bytes(raw)
    return hashlib.sha1(raw).hexdigest()
