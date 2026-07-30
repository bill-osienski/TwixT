"""v18 frozen preflight criteria -- THE PREREGISTRATION.

Spec: docs/superpowers/specs/2026-07-29-v18-depth2-provisional-backup-design.md
Sec 2.2 / 2.3.  Plan: Task 5.

Every numeric threshold, formula and decision rule the preflight will apply,
written down BEFORE any of them is measured. Nothing here reads a measurement
artifact; the operating-characteristic generator below is pure simulation.

Each constant carries `"basis"`:

    "policy"    -- a margin we chose, defensible but not derived
    "normative" -- a requirement we impose on the mechanism
    "measured"  -- a number taken from a recorded prior experiment

A reader must be able to tell an imposed number from a derived one without
reading the plan, which is why the label is data rather than prose.

`emit_frozen_criteria` is NOT called at implementation time; the binding
artifact is emitted later, at a clean HEAD (plan Execution Phase step 2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from statistics import NormalDist
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .fpu_dev_reservoir_protocol import canonical_json_bytes
from . import fpu_provenance

# ---------------------------------------------------------------------------
# Scope. Every flag is False: this module is a preregistration, and holding it
# authorizes nothing.
# ---------------------------------------------------------------------------

SCOPE_BOUNDARY = {
    "mcts_py_edit_authorized": False,
    "positive_cap_search_authorized": False,
    "scientific_acceptance_run_authorized": False,
    "commit_authorized": False,
    "later_stage_authorized": False,
}

SPEC_REVISION = 3
STRONGEST_CAP = 0.50
WEAKEST_CAP = 1.25
CAP_GRID = (1.25, 1.00, 0.75, 0.50)

# The primary statistic's census column, defined ONCE and referenced everywhere
# it is named: CENSUS_SCHEMA, REQUIRED_CENSUS_FIELDS, the target role predicate,
# SEPARATION, EXPOSURE_CUTOFF_RULE and `classify_role`. A second spelling of this
# name is how a gate comes to reference a column the census never emits.
PRIMARY_EXPOSURE_COLUMN = "exposure_primary_0.50"

# ---------------------------------------------------------------------------
# (a) ONE exposure formula, fixed a priori.
#
# Revision 1's first-past-the-post rule let selected-A choose the selector by
# A-vs-control AUC, contradicting "A may only demonstrate reach". The primary is
# therefore frozen before measurement, and the diagnostics cannot overturn it.
# ---------------------------------------------------------------------------

PRIMARY_EXPOSURE_FORMULA = {
    "name": "contribution_weighted_positive_mass",
    "census_column": PRIMARY_EXPOSURE_COLUMN,
    "expression": (
        "sum over eligible depth-2 leaves of "
        "(terminating_backups(leaf) / root.visit_count) * max(0, residual - cap)"),
    "evaluated_at_cap": STRONGEST_CAP,
    "cap_role": "strongest",
    "population": "eligible depth-2 leaves of one position's tree",
    "row_unit": "position",
    "chosen_a_priori": True,
    "basis": "policy",
    "rationale": (
        "the statistic most directly aligned with the claimed backup-mass "
        "mechanism: it weights each positive over-cap residual by the share of "
        "root backups that actually flow through that leaf"),
    "uses_absolute_residual_magnitude": False,
}

DESCRIPTIVE_EXPOSURE_FORMULAS = {
    "positive_count": {
        "expression": "count of eligible depth-2 leaves with residual > cap",
        "role": "descriptive_diagnostic",
        "can_rescue_primary_failure": False,
        "basis": "policy",
    },
    "positive_clipped_mass": {
        "expression": "sum over eligible depth-2 leaves of max(0, residual - cap)",
        "role": "descriptive_diagnostic",
        "can_rescue_primary_failure": False,
        "basis": "policy",
    },
}

# ---------------------------------------------------------------------------
# Selection predicates. Numeric before any residual is measured.
# ---------------------------------------------------------------------------

PHASE_WINDOWS = {
    "opening": (0, 30),
    "early_mid": (31, 60),
    "midgame": (61, 90),
    "late": (91, None),
    "basis": "policy",
}

NEAR_EVEN = {
    # The census emits the SIGNED value; near-even applies abs() as a transform.
    "variable": "root_value_stm",
    "transform": "abs",
    "op": "<=",
    "value": 0.30,
    "basis": "measured",
    "rationale": (
        "anchored to the observed failure regime: every one of v16a's 15 new "
        "collapses was near-even, abs(stm) <= 0.28, median 0.03"),
}

MIN_ELIGIBLE_DEPTH2_LEAVES = 50

BRANCHING = {
    "n_legal": "RECORDED as a stratum, never gated",
    "gated": False,
    "basis": "policy",
    "rationale": "spec Sec 9.2.4: branching bands recorded, not post-hoc gated",
}

# ---------------------------------------------------------------------------
# CANONICAL FIELD CONTRACT.
#
# Every predicate below names a column of Task 7's frozen census_positions.csv
# and nothing else. The criteria module and the measurement CLI must not hold
# two vocabularies for one quantity -- that is how a selector silently reads a
# column that was never emitted.
#
# Note the deliberate spelling difference: `exposure_primary_0.50` carries two
# decimals because that is the frozen column name, while the per-cap triples
# render their cap with `str(cap)`, giving 0.5 / 0.75 / 1.0 / 1.25. Do not tidy
# either into the other; both are load-bearing joins.
# ---------------------------------------------------------------------------

CENSUS_SCHEMA = (
    "population", "source_universe_ordinal", "game_content_sha1", "game_idx",
    "position_ply", "side_to_move", "canonical_state_sha1", "phase",
    "root_value_stm", "n_legal", "eligible_depth2_leaves", "replies",
    "explored_replies", "depth_ge3_backups", "depth_ge3_fraction",
    "follow_up_visits_per_reply", "positive_mass", "negative_mass",
    "sign_dominance", "terminal_depth2", "total_depth2",
    PRIMARY_EXPOSURE_COLUMN, "exposure_descriptive_count",
    "exposure_descriptive_clipped_mass",
) + tuple(
    f"{prefix}_{cap}"
    for cap in (str(c) for c in CAP_GRID)
    for prefix in ("would_clip", "clipped_amount", "revisit_to_depth3_rate")
) + ("seed",)

# Exactly the columns `classify_role` reads. Asserted to be a subset of
# CENSUS_SCHEMA, so the criteria module and its producer cannot drift apart.
REQUIRED_CENSUS_FIELDS = frozenset({
    PRIMARY_EXPOSURE_COLUMN,
    "sign_dominance",
    "root_value_stm",
    "eligible_depth2_leaves",
    "would_clip_1.25",
    "clipped_amount_1.25",
    "would_clip_0.5",
})

# The columns that express ABSOLUTE residual magnitude. Target selection may
# never touch these -- Sec 1.3's directional prediction lives there. Flip
# controls and identity witnesses legitimately do, which is what keeps the
# prohibition testable rather than vacuous.
ABSOLUTE_RESIDUAL_VARIABLES = frozenset({
    "would_clip_1.25",
    "clipped_amount_1.25",
    "would_clip_0.5",
})

FLIP_CONTROL_EXPOSURE = {
    "operator": "AND",
    "conditions": (
        {"variable": "would_clip_1.25", "op": ">=", "value": 3, "binding": True},
        {"variable": "clipped_amount_1.25", "op": ">=", "value": 0.50,
         "binding": True},
    ),
    "basis": "policy",
    "rationale": "exposure at the weakest cap implies exposure at every stronger cap",
}

IDENTITY_WITNESS = {
    "variable": "would_clip_0.5",
    "op": "==",
    "value": 0,
    "equivalent_to": "max(abs(eligible depth-2 residual)) <= 0.50",
    "basis": "policy",
    "rationale": (
        "EXACTLY equivalent to a residual maximum at or below 0.50, because the "
        "clip rule is STRICT: a leaf clips at 0.50 iff abs(residual) > 0.50, so "
        "a zero count is precisely the statement that no residual exceeds it. "
        "Stated as a count because the census emits counts -- a residual "
        "maximum is not a column and must not be invented as one. Such a tree "
        "is byte-identical to shipped at every grid cap"),
}

# Thresholds that are DERIVED at measurement time by a frozen rule rather than
# written as a literal. A selection predicate may name one of these; it may not
# carry prose.
DERIVED_THRESHOLD_RULES = {
    "EXPOSURE_CUTOFF": {
        "derivation": (
            "nearest-rank 0.90 quantile of exposure(row, 0.50) over the 30-row "
            "matched cohort"),
        "basis": "policy",
    },
}

ROLE_LABELS = ("target", "representative", "identity", "flip", "unassigned")

ROLE_ASSIGNMENT = {
    "order": ("target", "representative", "identity_and_flip", "shortfall"),
    "on_shortfall": "STOP",
    "roles": {
        "target": {
            "step": 1,
            "conditions": (
                {"variable": PRIMARY_EXPOSURE_COLUMN, "op": ">=",
                 "value": "EXPOSURE_CUTOFF"},
                {"variable": "sign_dominance", "op": ">=", "value": 0.80},
                {"variable": "root_value_stm", "transform": "abs", "op": "<=",
                 "value": 0.30},
                {"variable": "eligible_depth2_leaves", "op": ">=", "value": 50},
                {"variable": "is_flip_control", "op": "==", "value": False},
            ),
            "uses_absolute_residual_magnitude": False,
            "rationale": (
                "the explicit NOT keeps flip priority: flip controls are the "
                "scarcer role and the one that makes the matched-control gate "
                "non-vacuous, so starving them to feed targets is the worse "
                "failure"),
        },
        "representative": {
            "step": 2,
            "conditions": (
                {"variable": "is_target", "op": "==", "value": False},
                {"variable": "root_value_stm", "transform": "abs", "op": "<=",
                 "value": 0.30},
            ),
            "uses_absolute_residual_magnitude": False,
            "drawn_by": "exact phase/side quotas, canonical_state_sha1 ascending",
            "inspects_identity_or_flip_eligibility": False,
            "rationale": (
                "chosen BEFORE identity and flip exist, so the selection "
                "conditions on target status and nothing else; hash-ordering a "
                "residual-conditioned candidate set would not be "
                "residual-independent"),
        },
        "identity_witness": {
            "step": 3,
            "conditions": (
                {"variable": "would_clip_0.5", "op": "==", "value": 0},
            ),
            "uses_absolute_residual_magnitude": True,
        },
        "flip_control": {
            "step": 3,
            "conditions": FLIP_CONTROL_EXPOSURE["conditions"],
            "uses_absolute_residual_magnitude": True,
        },
    },
    "revisit_representatives_after_residual_roles": False,
}

MATCHING = {
    "cardinality": {"n_a": 30, "n_c": 30},
    "on_short_cohort": "PREFLIGHT_FAIL",
    "algorithm": "rectangular_hungarian_minimum_cost",
    "greedy_nearest_neighbour_forbidden": True,
    "greedy_rationale": (
        "greedy nearest-neighbour can fail on A-row ordering even when a "
        "complete valid matching exists, turning a solvable problem into a "
        "spurious preflight failure"),
    "inadmissible_pair_cost": "infinite",
    "cost": (
        "sum of per-variable normalized absolute differences, each divided by "
        "its own tolerance so all terms are in [0, 1]"),
    # `abs_root_value_stm` is the ONE derived variable: a TRANSFORM of a census
    # column rather than a column itself, because the matcher pairs on
    # magnitude, so a +0.2 A row and a -0.2 control are a legitimate pair.
    # `side_to_move` is matched exactly and carries the direction -- it is a
    # real column and is named as one, not aliased to "side".
    "variables": ("phase", "side_to_move", "abs_root_value_stm", "n_legal",
                  "eligible_depth2_leaves"),
    "derived_variables": {"abs_root_value_stm": "abs(root_value_stm)"},
    "census_columns": ("phase", "side_to_move", "root_value_stm", "n_legal",
                       "eligible_depth2_leaves"),
    "tolerances": {
        "phase": "exact",
        "side_to_move": "exact",
        "abs_root_value_stm": 0.10,
        "n_legal": 50,
        "eligible_depth2_leaves": 40,
    },
    # The determinism contract, SCOPED. The earlier single `tie_breaking` tuple
    # read as a global ordering and could not distinguish within-game selection
    # from equal-cost assignment resolution, so an emitted artifact carrying it
    # claimed more than the matcher can deliver.
    "determinism": {
        "a_row_order": ("canonical_state_sha1", "game_content_sha1",
                        "position_ply"),
        "game_column_order": ("game_content_sha1",),
        "within_game_position_order": ("cost", "canonical_state_sha1",
                                       "game_content_sha1", "position_ply"),
        "equal_cost_assignment_resolution": (
            "deterministic Hungarian traversal under the frozen a_row_order and "
            "game_column_order"),
        "global_lexicographic_minimum": False,
        "rationale": (
            "the scientific requirement is a deterministic, order-independent, "
            "residual-blind minimum-cost matching. Across game columns an "
            "equal-cost tie resolves by game_column_order, NOT by "
            "within_game_position_order, and no global lexicographic minimum is "
            "claimed -- ties are between equally admissible controls, so only "
            "the reproducibility of the choice carries content"),
    },
    "basis": "policy",
}

PER_GAME = {
    "controls": {
        "max_positions_per_game": 1,
        "rationale_removes_within_game_correlation": True,
        "consequence": "no game-clustered bootstrap is required",
    },
    "future_corpus": {
        "max_positions_per_game": 2,
        "min_ply_separation": 12,
    },
    "basis": "policy",
}

# ---------------------------------------------------------------------------
# Universe, census and sizing -- imported by Tasks 4, 4b and 8.
# ---------------------------------------------------------------------------

UNIVERSE = {
    "order_of_operations": (
        "1. authenticate the complete source",
        "2. remove FORBIDDEN WHOLE GAMES -- replay CONTENT-SHA exclusions ONLY",
        "3. sort survivors by replay content SHA-1 ascending, take EXACTLY the "
        "first 800 -- this fixes all_game_ids",
        "4. enumerate census positions INSIDE those 800 games",
        "5. apply POSITION exclusions (canonical hashes), which never remove a "
        "game from all_game_ids",
    ),
    "n_games": 800,
    "on_insufficient_games": "STOP",
    "zero_yield_games": "RETAINED in all_game_ids",
    "zero_yield_rationale": (
        "sizing_analysis_core's all_game_ids is the COMPLETE reservoir "
        "universe including games that yielded ZERO kept rows -- excluding them "
        "would bias success upward"),
    "position_exclusions_never_remove_a_game": True,
    "basis": "policy",
}

CENSUS = {
    "positions_per_game": 6,
    "phase_strata": {"opening": 1, "early_mid": 1, "midgame": 1, "late": 3},
    "position_rule": (
        "within each phase, over that phase's ASCENDING DISTINCT qualifying "
        "plies, take the nearest-rank quantile at index ceil(q * n), 1-indexed, "
        "without replacement"),
    "quantiles": {
        "opening": (0.5,), "early_mid": (0.5,), "midgame": (0.5,),
        "late": (0.25, 0.50, 0.75),
    },
    "missing_phases": "contribute zero and are REPORTED per game; never backfilled",
    "max_total_searches": 4800,
    "on_exceeding_max": "abort before the evaluator loads",
    "ordering": ("game_content_sha1_ascending", "ply_ascending"),
    "basis": "policy",
    "rationale": (
        "six evenly spaced GLOBAL ply quantiles can omit a narrow late interval "
        "entirely, or overrepresent late positions in long games; the selector "
        "geometry is phase-sensitive, so a phase-biased census biases sizing"),
}

SIZING = {
    "probabilistic_tiers": (200, 300, 400, 500, 600, 700),
    "degenerate_tier": 800,
    "trials_per_probabilistic_tier": 299,
    "trial_draw": "a random subset of that tier's size from the 800-game universe",
    "seed": 20260729,
    "success_criterion": (
        "an EXACT-SELECTOR witness filling the complete four-role geometry -- "
        "never a capacity bound"),
    "tier_passes_iff": "_binomial_lower_bound(k, 299, alpha=0.05) >= 0.99",
    "binomial_lower_bound_source": "fpu_dev_corpus_v2 (:3876)",
    "binomial_lower_bound_note": (
        "the exact ONE-SIDED Clopper-Pearson lower bound at alpha 0.05; 299 "
        "all-success gives 0.99003 >= 0.99, 298 gives 0.98999. Do not "
        "re-derive it and do not call it a 95% interval"),
    "tier_800_is_one_degenerate_trial": True,
    "tier_800_rationale": (
        "drawing 800 games from an 800-game universe returns the same set every "
        "time, so 299 repetitions cannot estimate a binomial success probability"),
    "on_700_not_qualifying": "SIZING_FAILS",
    "reported_size": "the smallest probabilistically passing tier, then the "
                     "next-tier-up margin rule",
    "reported_witness": (
        "drawn from a SUCCESSFUL frozen-seed trial at that tier, identified by "
        "its trial index -- NOT the content-SHA prefix, which can fail even "
        "when every random trial passes"),
    "basis": "policy",
}

# ---------------------------------------------------------------------------
# The measured criteria.
# ---------------------------------------------------------------------------

SIGN_DOMINANCE = {
    "min": 0.80,
    "formula": "positive_mass / (positive_mass + negative_mass)",
    "positive_mass": "sum(max(0,  residual_i))",
    "negative_mass": "sum(max(0, -residual_i))",
    "on_zero_denominator": 0.0,
    "on_zero_denominator_role": "ineligible_as_target",
    "basis": "policy",
}

REACH = {
    "min": 0.50,
    "population": "a_rows",
    "establishes": "reach_only",
    "aggregation": "pooled",
    "numerator": (
        "sum over A rows, over eligible leaves with residual > 0.50, of "
        "terminating_backups(leaf) * max(0, leaf.nn_value)"),
    "denominator": "the same sum over ALL eligible leaves",
    "on_zero_denominator": "PREFLIGHT_FAIL",
    "basis": "policy",
    "rationale": (
        "pooled, never a mean of per-row ratios: averaging ratios lets a row "
        "with almost no backup mass carry the same weight as one carrying most "
        "of it"),
}

TERMINAL_FRACTION = {
    "max": 0.10,
    "numerator": "depth-2 nodes with visit_count > 0 that are terminal",
    "denominator": (
        "depth-2 nodes with visit_count > 0 (terminal + eligible + synthetic)"),
    "aggregation": "pooled",
    "also_reported_per_population": True,
    "basis": "policy",
}

SEPARATION = {
    "row_unit": "position",
    "n_a": 30,
    "n_c": 30,
    "positive_class": "the 30 A rows",
    "negative_class": "the 30 MATCHED cohort rows",
    "statistic": PRIMARY_EXPOSURE_COLUMN,
    "estimator": "mann_whitney_u_over_n_a_times_n_c",
    "tie_contribution": 0.5,
    "weighting": "none",
    "min_auc": 0.70,
    "min_lower_bound": 0.5,
    "lower_bound_quantile": 0.05,
    "lower_bound_sided": "one_sided_95",
    # Frozen BEFORE the number is known, so it cannot be renegotiated after.
    "on_failure_means": (
        "required A-vs-matched-control selectivity was not established"),
    "on_failure_does_not_mean": "no effect exists",
    "on_failure_rationale": (
        "the approved operating characteristics give 51.7% power when the true "
        "AUC equals the 0.70 threshold, so a boundary miss is near a coin flip "
        "and carries no evidential weight against the mechanism; any verdict "
        "text reading a separation failure as refutation is wrong"),
    "lower_bound_rationale": (
        "one-sided q = 0.05 is chosen over two-sided q = 0.025 because only the "
        "lower endpoint gates, and it matches the one-sided Clopper-Pearson "
        "convention used by sizing. Revision 5 wrote rank 0.95*(n-1), which is "
        "the 95th PERCENTILE -- the UPPER endpoint -- not a lower bound"),
    "bootstrap": {
        "replicates": 10000,
        "seed": 20260729,
        "stratified": True,
    },
    "basis": "policy",
}

PROSPECTIVE_TARGET_SUBSET = "prospective_target_subset_of_broad_non_a_census"

R_MIN_RULE = {
    "population": PROSPECTIVE_TARGET_SUBSET,
    "population_definition": (
        "broad non-A census rows with exposure(row, 0.50) >= EXPOSURE_CUTOFF"),
    "excludes_a_rows": True,
    "subset_floor": 16,
    "on_subset_below_floor": "PREFLIGHT_FAIL",
    "subset_below_floor_reason": "prospective_target_subset_below_floor",
    "caps": CAP_GRID,
    "evaluated_at": "every_grid_cap",
    "aggregation": (
        "pooled(c) = (sum predicted_shipped_replies - sum "
        "predicted_capped_replies) / sum predicted_shipped_replies"),
    "fail_rule": "no grid cap has pooled(c) > 0",
    "fail_reason": "mechanism_not_predicted_to_act_at_any_cap",
    "derivation": "let c* be the WEAKEST cap with pooled(c*) > 0; "
                  "R_min = max(R_MIN_FLOOR, 0.5 * pooled(c*))",
    "floor": 0.01,
    "floor_basis": "normative",
    "floor_never_rescues_nonpositive": True,
    "on_nonpositive": "PREFLIGHT_FAIL",
    "on_nonpositive_reason": "mechanism_not_predicted_to_act_at_any_cap",
    "on_nonpositive_per_cap": "cap_ineligible_to_define_r_min",
    "weak_cap_under_reach": "ADVANCE",
    "basis": "policy",
    "rationale": (
        "small or zero reply reduction on ordinary controls is exactly the "
        "selectivity v18 claims, so penalising it would reward an "
        "indiscriminate mechanism; and under-reach at the weakest cap is a "
        "Sec 7 ADVANCE outcome, never a family rejection"),
}

HISTORICAL_ANCHORS = {
    "v16a_heldout_-0.20": {
        "reply_reduction": 0.28027286567513765,
        "exact_form": "1 - 24583/34156",
        "artifact_sha1": "6d15c7dd15bdc8e8a983700f536950bcc9830019",
        "basis": "measured",
    },
    "v17_weakest_r_0.15": {
        "reply_reduction": 0.23358985966500678,
        "exact_form": "516/2209",
        "artifact_sha1": "af7778c84e1ea04f463febfc615e5363400d6aad",
        "basis": "measured",
    },
    "v16_selected_a_fpu_-0.20": {
        "reply_reduction": 0.81836179163573375,
        "exact_form": "3307/4041",
        "artifact_sha1": "f201f0f25b868e5c4c7103992054c7b4df5074d1",
        "basis": "measured",
    },
}

R_MAX_SAFETY_FACTOR = 0.5

R_MAX_RULE = {
    "value": min(a["reply_reduction"] for a in HISTORICAL_ANCHORS.values()
                 ) * R_MAX_SAFETY_FACTOR,
    "safety_factor": R_MAX_SAFETY_FACTOR,
    "derivation": "min(anchor reply_reduction) * 0.5",
    "anchors_may_only": "lower_r_max",
    "on_r_min_ge_r_max": "PREFLIGHT_FAIL",
    "basis": "policy",
    "rationale": (
        "a deliberately conservative POLICY MARGIN, not an empirical "
        "derivation; spec Sec 2.1.1 permits anchors to LOWER R_max and to do "
        "nothing else. An empty band is unsatisfiable, not a pass"),
}

REVISIT_FORM_CRITERION = {
    "population": PROSPECTIVE_TARGET_SUBSET,
    "paired_if_fraction_at_least": 0.75,
    "min_would_clip_leaves": 5,
    "cap": WEAKEST_CAP,
    "otherwise": "candidate_only_floor",
    "basis": "policy",
    "rationale": (
        "the same population as R_min, and for the same reason: the 30-row "
        "matched cohort's top decile is about three rows"),
}

CONVERSION_EFFICIENCY_MIN = {
    "value": 0.5,
    "basis": "normative",
    "derived_bound_withdrawn": True,
    "rationale": (
        "a budget-conversion requirement we impose, not a bound we derive. "
        "Revision 1 claimed the ratio is bounded near 1.0 because one "
        "un-scanned reply frees exactly one simulation; that is withdrawn, "
        "since root allocation and stable-leader-subtree traffic both change "
        "under the cap, so gained deep backups are not conserved one-for-one "
        "against lost replies and the ratio is not bounded that way"),
}

MIN_LOST_REPLIES = {"development": 20, "held_out": 30}

STABLE_LEADER_MIN_FRACTION = 0.75

EXPOSURE_CUTOFF_RULE = {
    "population": "matched_cohort",
    "n": 30,
    "uses_a_rows": False,
    "statistic": PRIMARY_EXPOSURE_COLUMN,
    "quantile": 0.90,
    "method": "nearest_rank",
    "interpolation": False,
    "nearest_rank_definition": "the ceil(0.90 * n)-th smallest value, 1-indexed",
    "target_predicate": "exposure >= cutoff",
    "ties": "admit",
    "deterministic_ordering": ("canonical_state_sha1", "game_idx", "position_ply"),
    "basis": "policy",
}

BASIS_INDEX = {
    "PRIMARY_EXPOSURE_FORMULA": "policy",
    "SIGN_DOMINANCE.min": "policy",
    "REACH.min": "policy",
    "TERMINAL_FRACTION.max": "policy",
    "SEPARATION.min_auc": "policy",
    "R_MAX_RULE.value": "policy",
    "R_MIN_RULE.floor": "normative",
    "CONVERSION_EFFICIENCY_MIN": "normative",
    "MIN_LOST_REPLIES": "normative",
    "STABLE_LEADER_MIN_FRACTION": "normative",
    "NEAR_EVEN.value": "measured",
    "HISTORICAL_ANCHORS": "measured",
}

# ---------------------------------------------------------------------------
# Mechanical helpers. Each one implements a rule stated above exactly once.
# ---------------------------------------------------------------------------


def nearest_rank_quantile(values: Iterable[float], q: float) -> float:
    """The ceil(q * n)-th smallest value, 1-indexed. No interpolation, so the
    result is always an OBSERVED datum -- a cutoff no row attains is not a
    usable selection threshold."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("nearest-rank quantile of an empty population")
    rank = math.ceil(q * len(ordered))
    return ordered[max(1, rank) - 1]


def meets_exposure_cutoff(exposure: float, cutoff: float) -> bool:
    """Ties ADMITTED, per EXPOSURE_CUTOFF_RULE. `>` would silently drop the row
    sitting exactly on a cutoff that nearest-rank guarantees exists."""
    return exposure >= cutoff


def pooled_ratio(pairs: Sequence[Tuple[float, float]]) -> float:
    """sum(numerators) / sum(denominators) -- NOT the mean of per-pair ratios."""
    denominator = sum(d for _n, d in pairs)
    if denominator == 0:
        raise ValueError("pooled ratio with a zero denominator: PREFLIGHT_FAIL")
    return sum(n for n, _d in pairs) / denominator


def classify_role(row: Dict, exposure_cutoff: float, *,
                  representative_selected: bool = False) -> str:
    """Assign ONE role by the frozen ORDER, which is what makes the roles a
    partition -- the predicates alone overlap.

    `row` is a `census_positions.csv` record: the column names below are read
    DIRECTLY, with no alias layer, so this function and its producer share one
    vocabulary. See `REQUIRED_CENSUS_FIELDS`.

    `representative_selected` is the step-2 quota outcome, supplied by the
    caller: whether a non-target row is actually drawn depends on phase/side
    quotas that live in the corpus geometry, not here. What Task 5 freezes is
    the PRECEDENCE, and that is what this function implements.
    """
    is_flip = (row["would_clip_1.25"] >= 3
               and row["clipped_amount_1.25"] >= 0.50)

    is_target = (
        meets_exposure_cutoff(row[PRIMARY_EXPOSURE_COLUMN], exposure_cutoff)
        and row["sign_dominance"] >= SIGN_DOMINANCE["min"]
        and abs(row["root_value_stm"]) <= NEAR_EVEN["value"]
        and row["eligible_depth2_leaves"] >= MIN_ELIGIBLE_DEPTH2_LEAVES
        and not is_flip)
    if is_target:
        return "target"
    if representative_selected:
        return "representative"
    # Step 3, internally disjoint by construction: `would_clip_0.5 == 0` says no
    # residual exceeds 0.50, which forbids the three-above-1.25 flip predicate.
    if row["would_clip_0.5"] == IDENTITY_WITNESS["value"]:
        return "identity"
    if is_flip:
        return "flip"
    return "unassigned"


def weakest_positive_cap(pooled_by_cap: Dict[float, float]) -> float:
    """c* -- the WEAKEST cap with pooled(c) > 0. A nonpositive weakest cap makes
    that cap ineligible and ADVANCES to the next; it is not a rejection."""
    for cap in sorted(pooled_by_cap, reverse=True):
        if pooled_by_cap[cap] > 0.0:
            return cap
    raise ValueError(R_MIN_RULE["fail_reason"])


def r_min_from_pooled(pooled_by_cap: Dict[float, float]) -> float:
    """R_min = max(floor, 0.5 * pooled(c*)).

    The floor raises a small POSITIVE prediction. It never lifts a nonpositive
    one: `weakest_positive_cap` raises first, so no cap with pooled <= 0 can
    ever define R_min.
    """
    cap = weakest_positive_cap(pooled_by_cap)
    return max(R_MIN_RULE["floor"], 0.5 * pooled_by_cap[cap])


def r_band_is_satisfiable(r_min: float, r_max: float) -> bool:
    """An empty band is unsatisfiable, never a pass."""
    return r_min < r_max


def revisit_form(would_clip_counts: Sequence[int]) -> str:
    """Paired iff at least 75% of the prospective target subset carries >= 5
    shipped would_clip leaves at the WEAKEST cap.

    The PROSPECTIVE-TARGET SUBSET FLOOR binds here, not merely at R_min:
    refusing only an EMPTY subset would let a 3-row population decide the form
    of the whole criterion on 2 dense rows, which is exactly the vacuity the
    16-row floor exists to prevent.
    """
    floor = R_MIN_RULE["subset_floor"]
    if len(would_clip_counts) < floor:
        raise ValueError(
            f"{R_MIN_RULE['subset_below_floor_reason']}: "
            f"{len(would_clip_counts)} rows < the frozen floor of {floor}")
    dense = sum(1 for n in would_clip_counts
                if n >= REVISIT_FORM_CRITERION["min_would_clip_leaves"])
    fraction = dense / len(would_clip_counts)
    if fraction >= REVISIT_FORM_CRITERION["paired_if_fraction_at_least"]:
        return "paired"
    return REVISIT_FORM_CRITERION["otherwise"]


def mann_whitney_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """U / (n_A * n_C), ties contributing exactly 0.5."""
    if not len(positive) or not len(negative):
        raise ValueError("AUC needs a nonempty positive and negative class")
    greater = ties = 0
    for a in positive:
        for c in negative:
            if a > c:
                greater += 1
            elif a == c:
                ties += 1
    return (greater + 0.5 * ties) / (len(positive) * len(negative))


def protocol_quantile(values: Sequence[float], q: float) -> float:
    """The protocol's sorted linear-interpolation convention at rank q*(n-1).

    NOTE the rank rule: `q * (n - 1)`, not `q * n`. With q = 0.05 this is the
    ONE-SIDED 95% LOWER bound. Revision 5 wrote 0.95*(n-1), which names the
    UPPER endpoint -- the opposite of a lower bound.
    """
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile of an empty vector")
    if len(ordered) == 1:
        return ordered[0]
    rank = q * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (rank - low) * (ordered[high] - ordered[low])


def auc_lower_bound(bootstrap_aucs: Sequence[float]) -> float:
    """The frozen one-sided 95% lower bound of a bootstrap AUC distribution."""
    return protocol_quantile(bootstrap_aucs, SEPARATION["lower_bound_quantile"])


# ---------------------------------------------------------------------------
# Operating characteristics -- PURE SIMULATION.
#
# "True AUC 0.70" does not identify a distribution, and distributions with the
# same AUC give materially different power, tie rates and bootstrap behaviour.
# The data-generating model is therefore frozen here, before the table is run.
# ---------------------------------------------------------------------------

AUC_OC_MODEL = {
    "family": "equal_variance_gaussian_location_shift",
    "control": "N(0, 1)",
    "treatment": "N(delta, 1)",
    "delta": "sqrt(2) * Phi^-1(AUC)",
    "n_a": 30,
    "n_c": 30,
    "outer_datasets": 2000,
    "bootstrap_replicates": 10000,
    "outer_seed": 20260731,
    "bootstrap_seed": 20260732,
    "seed_rationale": (
        "SEPARATE streams, so the bootstrap cannot correlate with dataset "
        "generation"),
    "percentile_q": 0.05,
    "percentile_rank_rule": "q * (n - 1)",
    "ties": (
        "contribute exactly 0.5 to the Mann-Whitney statistic; the continuous "
        "family makes exact ties measure-zero, so the tie path is exercised by "
        "a separate unit test, not by the table"),
    "auc_points": (0.50, 0.60, 0.70, 0.80, 0.90),
    "reported": "power at 0.60 / 0.70 / 0.80 / 0.90, false-pass at 0.50, each "
                "with its Monte Carlo standard error",
    "label": (
        "model-specific operating characteristics under an equal-variance "
        "Gaussian shift -- NOT distribution-free power. Real exposure "
        "distributions are skewed and bounded below at zero, so these numbers "
        "bound intuition, not truth"),
    "basis": "policy",
}


def auc_to_gaussian_delta(auc: float) -> float:
    """delta = sqrt(2) * Phi^-1(AUC). AUC 0.70 -> 0.7416, 0.80 -> 1.1902."""
    if not 0.0 < auc < 1.0:
        raise ValueError(f"AUC must lie strictly in (0, 1): {auc!r}")
    return math.sqrt(2.0) * NormalDist().inv_cdf(auc)


def _stream(seed: int, auc: float, index: int) -> np.random.Generator:
    # Keyed by (stream seed, AUC point, dataset index) so no two cells share
    # draws and the whole table is reproducible from the two frozen seeds.
    return np.random.default_rng([seed, int(round(auc * 1_000_000)), index])


def simulate_dataset(auc: float, index: int):
    """One Monte Carlo dataset from the frozen DGP: `(treatment, control)`."""
    rng = _stream(AUC_OC_MODEL["outer_seed"], auc, index)
    delta = auc_to_gaussian_delta(auc)
    treatment = rng.normal(delta, 1.0, AUC_OC_MODEL["n_a"])
    control = rng.normal(0.0, 1.0, AUC_OC_MODEL["n_c"])
    return treatment, control


def bootstrap_aucs(treatment, control, replicates: int, index: int,
                   auc: float = 0.0):
    """Stratified bootstrap AUCs -- resample each class to its own size."""
    rng = _stream(AUC_OC_MODEL["bootstrap_seed"], auc, index)
    n_a, n_c = len(treatment), len(control)
    a = np.asarray(treatment)[rng.integers(0, n_a, size=(replicates, n_a))]
    c = np.asarray(control)[rng.integers(0, n_c, size=(replicates, n_c))]
    diff = a[:, :, None] - c[:, None, :]
    greater = (diff > 0).sum(axis=(1, 2))
    ties = (diff == 0).sum(axis=(1, 2))
    return (greater + 0.5 * ties) / float(n_a * n_c)


def generate_auc_oc_table(auc_points=None, outer_datasets=None,
                          bootstrap_replicates=None) -> Dict:
    """Power and false-pass for the frozen SEPARATION rule under the frozen DGP.

    PURE SIMULATION: reads no census, no cohort and no preflight artifact.
    """
    auc_points = tuple(auc_points if auc_points is not None
                       else AUC_OC_MODEL["auc_points"])
    outer = int(outer_datasets if outer_datasets is not None
                else AUC_OC_MODEL["outer_datasets"])
    replicates = int(bootstrap_replicates if bootstrap_replicates is not None
                     else AUC_OC_MODEL["bootstrap_replicates"])

    rows: List[Dict] = []
    for auc in auc_points:
        passes = 0
        point_pass = 0
        bound_pass = 0
        for index in range(outer):
            treatment, control = simulate_dataset(auc, index)
            point = mann_whitney_auc(treatment, control)
            bound = auc_lower_bound(
                bootstrap_aucs(treatment, control, replicates, index, auc))
            hit_point = point >= SEPARATION["min_auc"]
            hit_bound = bound >= SEPARATION["min_lower_bound"]
            point_pass += int(hit_point)
            bound_pass += int(hit_bound)
            passes += int(hit_point and hit_bound)
        rate = passes / outer
        rows.append({
            "true_auc": auc,
            "delta": auc_to_gaussian_delta(auc),
            "outcome": "false_pass_rate" if auc <= 0.5 else "power",
            "pass_rate": rate,
            "mc_standard_error": math.sqrt(max(rate * (1.0 - rate), 0.0) / outer),
            "point_estimate_gate_rate": point_pass / outer,
            "lower_bound_gate_rate": bound_pass / outer,
        })
    return {
        "model": AUC_OC_MODEL,
        "rule": {"min_auc": SEPARATION["min_auc"],
                 "min_lower_bound": SEPARATION["min_lower_bound"],
                 "lower_bound_quantile": SEPARATION["lower_bound_quantile"]},
        "outer_datasets": outer,
        "bootstrap_replicates": replicates,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Presentation and emission.
# ---------------------------------------------------------------------------


def as_dict() -> Dict:
    """Every frozen constant, deterministically. Runtime provenance belongs to
    emission, so it is absent here and this is safe to print and diff."""
    return {
        "run_kind": "preregistration",
        "scientific_interpretation_forbidden": True,
        "spec_revision": SPEC_REVISION,
        "mcts_py_unmodified": True,
        "scope_boundary": SCOPE_BOUNDARY,
        "cap_grid": list(CAP_GRID),
        "primary_exposure_formula": PRIMARY_EXPOSURE_FORMULA,
        "descriptive_exposure_formulas": DESCRIPTIVE_EXPOSURE_FORMULAS,
        "exposure_cutoff_rule": EXPOSURE_CUTOFF_RULE,
        "derived_threshold_rules": DERIVED_THRESHOLD_RULES,
        "role_assignment": ROLE_ASSIGNMENT,
        "role_labels": list(ROLE_LABELS),
        "census_schema": list(CENSUS_SCHEMA),
        "primary_exposure_column": PRIMARY_EXPOSURE_COLUMN,
        "required_census_fields": sorted(REQUIRED_CENSUS_FIELDS),
        "absolute_residual_variables": sorted(ABSOLUTE_RESIDUAL_VARIABLES),
        "phase_windows": PHASE_WINDOWS,
        "near_even": NEAR_EVEN,
        "min_eligible_depth2_leaves": MIN_ELIGIBLE_DEPTH2_LEAVES,
        "branching": BRANCHING,
        "flip_control_exposure": FLIP_CONTROL_EXPOSURE,
        "identity_witness": IDENTITY_WITNESS,
        "matching": MATCHING,
        "per_game": PER_GAME,
        "universe": UNIVERSE,
        "census": CENSUS,
        "sizing": SIZING,
        "sign_dominance": SIGN_DOMINANCE,
        "reach": REACH,
        "terminal_fraction": TERMINAL_FRACTION,
        "separation": SEPARATION,
        "r_min_rule": R_MIN_RULE,
        "r_max_rule": R_MAX_RULE,
        "historical_anchors": HISTORICAL_ANCHORS,
        "revisit_form_criterion": REVISIT_FORM_CRITERION,
        "conversion_efficiency_min": CONVERSION_EFFICIENCY_MIN,
        "min_lost_replies": MIN_LOST_REPLIES,
        "stable_leader_min_fraction": STABLE_LEADER_MIN_FRACTION,
        "prospective_target_subset_floor": R_MIN_RULE["subset_floor"],
        "auc_oc_model": AUC_OC_MODEL,
        "basis_index": BASIS_INDEX,
    }


def emit_frozen_criteria(path: str) -> str:
    """Write the preregistration canonically and return its SHA-1.

    NOT called at implementation time -- the binding artifact is emitted in the
    Execution Phase, at a clean HEAD, so no measurement can precede it.
    """
    payload = as_dict()
    payload["git_commit"] = fpu_provenance.git_commit()
    payload["worktree_clean"] = fpu_provenance.worktree_clean()
    raw = canonical_json_bytes(payload)
    with open(path, "wb") as handle:
        handle.write(raw)
    return hashlib.sha1(raw).hexdigest()


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auc-oc-table", action="store_true",
                        help="generate the pure-simulation operating-characteristic table")
    parser.add_argument("--print", action="store_true",
                        help="print every frozen constant without emitting an artifact")
    args = parser.parse_args()
    if args.auc_oc_table:
        print(json.dumps(generate_auc_oc_table(), indent=2, sort_keys=True))
    if args.print or not args.auc_oc_table:
        print(json.dumps(as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
