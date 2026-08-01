"""v18 frozen preflight criteria -- the preregistration. Plan Task 5.

Pure data plus pure simulation. No measurement artifact is read and none is
emitted into the repository: the one test that exercises `emit_frozen_criteria`
writes into pytest's `tmp_path` and asserts the canonical logs path is untouched.
"""
import builtins
import hashlib
import json
import math
import pathlib
import random
from statistics import NormalDist

import pytest

from scripts.GPU.alphazero import v18_preflight_criteria as C


def census_row(*, exposure, sign_dominance, root_value_stm, would_clip_125,
               clipped_amount_125, would_clip_05, eligible_depth2_leaves=90):
    """One `census_positions.csv` record, using the FROZEN column names.

    Every key here is a member of `C.CENSUS_SCHEMA`; the helper exists so no
    test can quietly invent a column the measurement never emits.
    """
    row = {
        "population": "census",
        "source_universe_ordinal": 7,
        "game_content_sha1": "0" * 40,
        "game_idx": 3,
        "position_ply": 44,
        "side_to_move": "black",
        "canonical_state_sha1": "1" * 40,
        "phase": "early_mid",
        "root_value_stm": root_value_stm,
        "n_legal": 480,
        "eligible_depth2_leaves": eligible_depth2_leaves,
        "replies": 12,
        "explored_replies": 9,
        "depth_ge3_backups": 120,
        "depth_ge3_fraction": 0.3,
        "follow_up_visits_per_reply": 2.5,
        "positive_mass": 4.0,
        "negative_mass": 1.0,
        "sign_dominance": sign_dominance,
        "terminal_depth2": 1,
        "total_depth2": 40,
        "exposure_primary_0.50": exposure,
        "exposure_descriptive_count": 5,
        "exposure_descriptive_clipped_mass": 2.0,
        "would_clip_1.25": would_clip_125,
        "clipped_amount_1.25": clipped_amount_125,
        "revisit_to_depth3_rate_1.25": 0.5,
        "would_clip_1.0": 4,
        "clipped_amount_1.0": 1.0,
        "revisit_to_depth3_rate_1.0": 0.5,
        "would_clip_0.75": 6,
        "clipped_amount_0.75": 1.5,
        "revisit_to_depth3_rate_0.75": 0.5,
        "would_clip_0.5": would_clip_05,
        "clipped_amount_0.5": 2.0,
        "revisit_to_depth3_rate_0.5": 0.5,
        "seed": 20260730,
    }
    assert set(row) == set(C.CENSUS_SCHEMA), (
        set(row) ^ set(C.CENSUS_SCHEMA))
    return row


# --- (a) the single frozen exposure formula --------------------------------

def test_primary_formula_is_frozen_and_single():
    p = C.PRIMARY_EXPOSURE_FORMULA
    assert p["name"] == "contribution_weighted_positive_mass"
    assert p["evaluated_at_cap"] == 0.50
    assert p["cap_role"] == "strongest"
    assert p["chosen_a_priori"] is True
    assert p["basis"] == "policy"
    # Exactly one primary, and it is not also listed as a descriptive rescue.
    assert p["name"] not in C.DESCRIPTIVE_EXPOSURE_FORMULAS


def test_descriptive_formulas_cannot_rescue():
    assert C.DESCRIPTIVE_EXPOSURE_FORMULAS, "the diagnostics must exist to be bounded"
    for name, spec in C.DESCRIPTIVE_EXPOSURE_FORMULAS.items():
        assert "can_rescue_primary_failure" in spec, name
        assert spec["can_rescue_primary_failure"] is False, name
        assert spec["role"] == "descriptive_diagnostic", name


# --- role predicates --------------------------------------------------------

def test_target_selection_never_uses_absolute_residuals():
    # Sec 1.3's directional prediction lives in TARGET selection, so absolute
    # residual magnitude may not enter there. Flip controls legitimately use it.
    target = C.ROLE_ASSIGNMENT["roles"]["target"]
    target_vars = {c["variable"] for c in target["conditions"]}
    assert not (target_vars & C.ABSOLUTE_RESIDUAL_VARIABLES)
    assert target["uses_absolute_residual_magnitude"] is False
    # Non-vacuity: the frozen name set is neither empty nor misspelt, because a
    # role that SHOULD match it does.
    flip = C.ROLE_ASSIGNMENT["roles"]["flip_control"]
    flip_vars = {c["variable"] for c in flip["conditions"]}
    assert flip_vars & C.ABSOLUTE_RESIDUAL_VARIABLES
    assert flip["uses_absolute_residual_magnitude"] is True


def test_flip_control_exposure_is_an_AND_of_two_frozen_constants():
    f = C.FLIP_CONTROL_EXPOSURE
    assert f["operator"] == "AND"
    assert len(f["conditions"]) == 2
    by_var = {c["variable"]: c for c in f["conditions"]}
    assert by_var["would_clip_1.25"]["value"] == 3
    assert by_var["would_clip_1.25"]["op"] == ">="
    assert by_var["clipped_amount_1.25"]["value"] == 0.50
    assert by_var["clipped_amount_1.25"]["op"] == ">="
    # Both name frozen census columns, not classifier-local aliases.
    assert set(by_var) <= set(C.CENSUS_SCHEMA)
    for cond in f["conditions"]:
        assert cond["binding"] is True, cond


def test_every_selection_predicate_is_numeric():
    # No prose thresholds anywhere in a selection rule. A string value is only
    # admissible when it names a threshold DERIVED by a frozen rule.
    for role, spec in C.ROLE_ASSIGNMENT["roles"].items():
        for cond in spec["conditions"]:
            value = cond["value"]
            if isinstance(value, str):
                assert value in C.DERIVED_THRESHOLD_RULES, (role, cond)
                assert value.isupper(), (role, cond)
            else:
                assert isinstance(value, (int, float, bool)), (role, cond)
    for name in C.DERIVED_THRESHOLD_RULES:
        assert C.DERIVED_THRESHOLD_RULES[name]["derivation"], name


def test_role_assignment_is_total_and_exclusive():
    # Renamed from test_role_predicates_are_mutually_exclusive_on_every_row:
    # the formulas alone do NOT establish exclusivity, the frozen ORDER does.
    assert C.ROLE_ASSIGNMENT["order"] == (
        "target", "representative", "identity_and_flip", "shortfall")
    assert C.ROLE_ASSIGNMENT["on_shortfall"] == "STOP"

    cutoff = 1.0
    # Deliberate overlaps: rows that satisfy more than one role's predicate.
    # Column names are the frozen census schema, not classifier-local aliases.
    both_target_and_flip = census_row(
        exposure=5.0, sign_dominance=0.95, root_value_stm=0.05,
        would_clip_125=4, clipped_amount_125=2.0, would_clip_05=9)
    plain_target = census_row(
        exposure=5.0, sign_dominance=0.95, root_value_stm=0.05,
        would_clip_125=0, clipped_amount_125=0.0, would_clip_05=2)
    identity_row = census_row(
        exposure=0.0, sign_dominance=0.0, root_value_stm=0.05,
        would_clip_125=0, clipped_amount_125=0.0, would_clip_05=0)
    nothing_row = census_row(
        exposure=0.0, sign_dominance=0.1, root_value_stm=-0.9,
        eligible_depth2_leaves=3,
        would_clip_125=0, clipped_amount_125=0.0, would_clip_05=1)

    # A row satisfying BOTH target and flip must go to flip: the explicit NOT in
    # the target predicate keeps flip priority, because flip is the scarcer role.
    assert C.classify_role(both_target_and_flip, cutoff) == "flip"
    assert C.classify_role(plain_target, cutoff) == "target"
    assert C.classify_role(identity_row, cutoff) == "identity"
    assert C.classify_role(nothing_row, cutoff) == "unassigned"

    # Representative is drawn by quota at step 2 from NON-TARGET rows, and wins
    # over the step-3 residual roles for the row it takes.
    assert C.classify_role(identity_row, cutoff, representative_selected=True) == "representative"
    # ... but never displaces a target, which is assigned at step 1.
    assert C.classify_role(plain_target, cutoff, representative_selected=True) == "target"

    # Totality and exclusivity: every row gets exactly one label from the frozen set.
    rows = [both_target_and_flip, plain_target, identity_row, nothing_row]
    labels = [C.classify_role(r, cutoff) for r in rows]
    assert all(label in C.ROLE_LABELS for label in labels)
    # Identity and flip cannot both hold: would_clip_0.5 == 0 says no residual
    # exceeds 0.50, which forbids three leaves above 1.25. Step 3 is internally
    # disjoint by construction, not by assignment order.
    for row in rows:
        identity = row["would_clip_0.5"] == 0
        flip = (row["would_clip_1.25"] >= 3 and row["clipped_amount_1.25"] >= 0.50)
        assert not (identity and flip), row


# --- matching and per-game caps ---------------------------------------------

def test_matching_tolerances_are_frozen_for_every_variable():
    m = C.MATCHING
    tol = m["tolerances"]
    assert tol["phase"] == "exact"
    assert tol["side_to_move"] == "exact"
    assert tol["abs_root_value_stm"] == 0.10
    assert tol["n_legal"] == 50
    assert tol["eligible_depth2_leaves"] == 40
    # Every matching variable carries a tolerance; none is left unstated.
    assert set(tol) == set(m["variables"])
    # Every matching input is either a real census column or a DECLARED
    # transform of one. `side` was an undeclared alias for `side_to_move`.
    assert set(m["variables"]) <= (
        set(C.CENSUS_SCHEMA) | set(m["derived_variables"]))
    assert "side" not in m["variables"]
    assert "side" not in m["tolerances"]
    assert m["derived_variables"] == {"abs_root_value_stm": "abs(root_value_stm)"}
    assert set(m["census_columns"]) <= set(C.CENSUS_SCHEMA)
    assert m["cardinality"] == {"n_a": 30, "n_c": 30}
    assert m["algorithm"] == "rectangular_hungarian_minimum_cost"
    assert m["greedy_nearest_neighbour_forbidden"] is True
    assert m["on_short_cohort"] == "PREFLIGHT_FAIL"
    assert m["inadmissible_pair_cost"] == "infinite"
    # The determinism contract is SCOPED: a single tie-break tuple could not
    # distinguish within-game selection from equal-cost assignment resolution,
    # so an artifact carrying it over-claimed.
    assert "tie_breaking" not in m, "the ambiguous single tuple must be gone"
    d = m["determinism"]
    assert d["a_row_order"] == ("canonical_state_sha1", "game_content_sha1",
                                "position_ply")
    assert d["game_column_order"] == ("game_content_sha1",)
    assert d["within_game_position_order"] == (
        "cost", "canonical_state_sha1", "game_content_sha1", "position_ply")
    assert d["equal_cost_assignment_resolution"]
    assert d["global_lexicographic_minimum"] is False
    # The emitted preregistration carries the scoped block, not the old tuple.
    emitted = C.as_dict()["matching"]
    assert emitted["determinism"] is d
    assert "tie_breaking" not in emitted


def test_per_game_caps_differ_for_controls_and_corpus():
    controls = C.PER_GAME["controls"]
    corpus = C.PER_GAME["future_corpus"]
    assert controls["max_positions_per_game"] == 1
    assert corpus["max_positions_per_game"] == 2
    assert corpus["min_ply_separation"] == 12
    assert controls["max_positions_per_game"] != corpus["max_positions_per_game"]
    assert controls["rationale_removes_within_game_correlation"] is True


# --- the exposure cutoff ----------------------------------------------------

def test_exposure_cutoff_rule_is_control_only_with_nearest_rank():
    r = C.EXPOSURE_CUTOFF_RULE
    assert r["population"] == "matched_cohort"
    assert r["n"] == 30
    assert r["uses_a_rows"] is False
    assert r["quantile"] == 0.90
    assert r["method"] == "nearest_rank"
    assert r["interpolation"] is False
    assert r["statistic"] == C.PRIMARY_EXPOSURE_COLUMN
    # nearest-rank is ceil(q*n), 1-indexed, so the cutoff is always an observed
    # datum rather than an interpolated value that no row attains.
    assert C.nearest_rank_quantile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.90) == 9
    assert C.nearest_rank_quantile([5.0], 0.90) == 5.0
    observed = [3.0, 1.0, 2.0]
    assert C.nearest_rank_quantile(observed, 0.90) in observed


def test_exposure_cutoff_tie_convention_is_admit_and_ordering_is_total():
    r = C.EXPOSURE_CUTOFF_RULE
    assert r["target_predicate"] == "exposure >= cutoff"
    assert r["ties"] == "admit"
    assert r["deterministic_ordering"] == (
        "canonical_state_sha1", "game_idx", "position_ply")
    assert len(set(r["deterministic_ordering"])) == 3
    # ">=" admits a row sitting exactly on the cutoff; ">" would silently drop it.
    assert C.meets_exposure_cutoff(2.0, 2.0) is True
    assert C.meets_exposure_cutoff(1.999, 2.0) is False


# --- the pooled/denominator-naming criteria ---------------------------------

def test_sign_dominance_formula_names_its_zero_denominator_behaviour():
    s = C.SIGN_DOMINANCE
    assert s["min"] == 0.80
    assert s["formula"] == "positive_mass / (positive_mass + negative_mass)"
    assert s["on_zero_denominator"] == 0.0
    assert s["on_zero_denominator_role"] == "ineligible_as_target"


def test_reach_is_pooled_not_a_mean_of_ratios():
    r = C.REACH
    assert r["min"] == 0.50
    assert r["aggregation"] == "pooled"
    assert r["aggregation"] != "mean_of_per_row_ratios"
    assert r["population"] == "a_rows"
    assert r["establishes"] == "reach_only"
    assert r["on_zero_denominator"] == "PREFLIGHT_FAIL"
    assert "sum over A rows" in r["numerator"]
    assert "ALL eligible leaves" in r["denominator"]
    # Pooling is not averaging: on unequal row weights the two disagree, which
    # is exactly why the rule names one of them.
    assert C.pooled_ratio([(1.0, 10.0), (9.0, 10.0)]) == pytest.approx(0.5)
    assert C.pooled_ratio([(1.0, 1.0), (0.0, 99.0)]) == pytest.approx(0.01)


def test_terminal_fraction_names_its_denominator():
    t = C.TERMINAL_FRACTION
    assert t["max"] == 0.10
    assert t["numerator"] == "depth-2 nodes with visit_count > 0 that are terminal"
    assert t["denominator"] == (
        "depth-2 nodes with visit_count > 0 (terminal + eligible + synthetic)")
    assert t["aggregation"] == "pooled"
    assert t["also_reported_per_population"] is True


def test_separation_declares_row_unit_ties_weighting_and_bootstrap():
    s = C.SEPARATION
    assert s["row_unit"] == "position"
    assert s["row_unit"] != "leaf"
    assert s["n_a"] == 30 and s["n_c"] == 30
    assert s["statistic"] == C.PRIMARY_EXPOSURE_COLUMN
    assert s["estimator"] == "mann_whitney_u_over_n_a_times_n_c"
    assert s["tie_contribution"] == 0.5
    assert s["weighting"] == "none"
    assert s["min_auc"] == 0.70
    assert s["min_lower_bound"] == 0.5
    assert s["bootstrap"]["replicates"] == 10000
    assert s["bootstrap"]["seed"] == 20260729
    assert s["bootstrap"]["stratified"] is True
    assert s["lower_bound_quantile"] == 0.05
    assert s["lower_bound_sided"] == "one_sided_95"
    # The tie path is exercised here rather than by the continuous-family table.
    assert C.mann_whitney_auc([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.5)
    assert C.mann_whitney_auc([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert C.mann_whitney_auc([1.0, 2.0], [3.0, 4.0]) == 0.0
    assert C.mann_whitney_auc([2.0], [1.0, 2.0, 3.0]) == pytest.approx(0.5)


# --- R_min ------------------------------------------------------------------

def test_r_min_population_is_the_prospective_target_subset():
    r = C.R_MIN_RULE
    assert r["population"] == "prospective_target_subset_of_broad_non_a_census"
    assert r["excludes_a_rows"] is True
    assert r["population"] != "matched_cohort"
    assert r["population"] != "all_ordinary_controls"
    assert r["subset_floor"] == 16
    assert r["on_subset_below_floor"] == "PREFLIGHT_FAIL"
    assert r["subset_below_floor_reason"] == "prospective_target_subset_below_floor"


def test_r_min_is_evaluated_at_every_cap_not_only_the_weakest():
    assert C.R_MIN_RULE["caps"] == (1.25, 1.00, 0.75, 0.50)
    assert len(C.R_MIN_RULE["caps"]) == 4
    assert C.R_MIN_RULE["evaluated_at"] == "every_grid_cap"


def test_r_min_fails_only_when_no_cap_predicts_positive_conversion():
    r = C.R_MIN_RULE
    assert r["fail_rule"] == "no grid cap has pooled(c) > 0"
    # One positive cap is enough, even if the weakest is nonpositive.
    assert C.r_min_from_pooled({1.25: -0.1, 1.00: 0.4, 0.75: 0.4, 0.50: 0.9}) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        C.r_min_from_pooled({1.25: -0.1, 1.00: 0.0, 0.75: -0.3, 0.50: 0.0})


def test_weak_cap_under_reach_is_an_advance_outcome_not_a_rejection():
    r = C.R_MIN_RULE
    assert r["weak_cap_under_reach"] == "ADVANCE"
    assert r["weak_cap_under_reach"] != "PREFLIGHT_FAIL"
    # c* is the WEAKEST cap with a positive pooled value, so a nonpositive
    # weakest cap advances to the next one instead of rejecting the family.
    assert C.weakest_positive_cap({1.25: -0.1, 1.00: 0.4, 0.75: 0.8, 0.50: 0.9}) == 1.00


def test_r_min_failure_reason_is_the_frozen_string():
    assert C.R_MIN_RULE["fail_reason"] == "mechanism_not_predicted_to_act_at_any_cap"
    # Revision 3's verdict test used the obsolete shorter string; they must not drift.
    assert C.R_MIN_RULE["fail_reason"] != "mechanism_not_predicted_to_act"


def test_r_min_rule_fails_rather_than_floors_a_nonpositive_prediction():
    r = C.R_MIN_RULE
    assert r["on_nonpositive"] == "PREFLIGHT_FAIL"
    assert isinstance(r["on_nonpositive_reason"], str) and r["on_nonpositive_reason"]
    assert not isinstance(r["on_nonpositive"], (int, float))
    assert r["floor"] == 0.01
    assert r["floor_basis"] == "normative"
    assert r["floor_never_rescues_nonpositive"] is True
    assert r["on_nonpositive_per_cap"] == "cap_ineligible_to_define_r_min"
    # The floor raises a small positive prediction, but never lifts a negative
    # one into a positive R_min.
    assert C.r_min_from_pooled({1.25: 0.001, 1.00: 0.5, 0.75: 0.5, 0.50: 0.5}) == 0.01
    with pytest.raises(ValueError):
        C.r_min_from_pooled({1.25: -1.0, 1.00: -1.0, 0.75: -1.0, 0.50: -1.0})


# --- R_max and the band -----------------------------------------------------

def test_r_max_is_a_policy_margin_and_strictly_below_every_anchor():
    r = C.R_MAX_RULE
    assert r["basis"] == "policy"
    assert r["basis"] != "measured"
    assert r["safety_factor"] == 0.5
    assert r["value"] == 0.11679492983250339
    anchors = [a["reply_reduction"] for a in C.HISTORICAL_ANCHORS.values()]
    assert r["value"] == min(anchors) * 0.5
    for anchor in anchors:
        assert r["value"] < anchor
    assert r["anchors_may_only"] == "lower_r_max"


def test_empty_r_band_is_a_failure_not_a_pass():
    assert C.R_MAX_RULE["on_r_min_ge_r_max"] == "PREFLIGHT_FAIL"
    assert C.r_band_is_satisfiable(0.05, 0.11679492983250339) is True
    assert C.r_band_is_satisfiable(0.11679492983250339, 0.11679492983250339) is False
    assert C.r_band_is_satisfiable(0.5, 0.11679492983250339) is False


# --- revisit form, per-stage constants, anchors -----------------------------

def test_revisit_density_population_is_the_prospective_target_subset():
    r = C.REVISIT_FORM_CRITERION
    assert r["population"] == "prospective_target_subset_of_broad_non_a_census"
    assert r["population"] == C.R_MIN_RULE["population"]
    assert r["paired_if_fraction_at_least"] == 0.75
    assert r["min_would_clip_leaves"] == 5
    assert r["cap"] == 1.25
    assert r["otherwise"] == "candidate_only_floor"


def test_min_lost_replies_has_separate_development_and_heldout_values():
    assert C.MIN_LOST_REPLIES == {"development": 20, "held_out": 30}
    assert C.STABLE_LEADER_MIN_FRACTION == 0.75
    # 0.75 of the per-stage target counts: 12/16 development, 18/24 held-out.
    assert math.floor(0.75 * 16) == 12
    assert math.floor(0.75 * 24) == 18


def test_conversion_efficiency_is_labelled_normative_not_derived():
    c = C.CONVERSION_EFFICIENCY_MIN
    assert c["value"] == 0.5
    assert c["basis"] == "normative"
    assert c["basis"] != "measured"
    assert c["derived_bound_withdrawn"] is True
    assert "not bounded" in c["rationale"] or "not a bound" in c["rationale"]


def test_historical_anchors_carry_exact_values_and_artifact_sha1s():
    a = C.HISTORICAL_ANCHORS
    assert set(a) == {"v16a_heldout_-0.20", "v17_weakest_r_0.15",
                      "v16_selected_a_fpu_-0.20"}
    assert a["v16a_heldout_-0.20"]["reply_reduction"] == 0.28027286567513765
    assert a["v17_weakest_r_0.15"]["reply_reduction"] == 0.23358985966500678
    assert a["v16_selected_a_fpu_-0.20"]["reply_reduction"] == 0.81836179163573375
    # Exact rational forms, never a rounded decimal.
    assert a["v16a_heldout_-0.20"]["reply_reduction"] == 1 - 24583 / 34156
    assert a["v17_weakest_r_0.15"]["reply_reduction"] == 516 / 2209
    assert a["v16_selected_a_fpu_-0.20"]["reply_reduction"] == 3307 / 4041
    assert a["v16a_heldout_-0.20"]["artifact_sha1"] == "6d15c7dd15bdc8e8a983700f536950bcc9830019"
    assert a["v17_weakest_r_0.15"]["artifact_sha1"] == "af7778c84e1ea04f463febfc615e5363400d6aad"
    assert a["v16_selected_a_fpu_-0.20"]["artifact_sha1"] == "f201f0f25b868e5c4c7103992054c7b4df5074d1"
    for name, anchor in a.items():
        assert len(anchor["artifact_sha1"]) == 40, name
        assert int(anchor["artifact_sha1"], 16) >= 0, name


# --- emission ---------------------------------------------------------------

def test_emit_frozen_criteria_is_byte_reproducible(tmp_path):
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    sha_a = C.emit_frozen_criteria(str(first))
    sha_b = C.emit_frozen_criteria(str(second))
    assert first.read_bytes() == second.read_bytes()
    assert sha_a == sha_b == hashlib.sha1(first.read_bytes()).hexdigest()
    # No timestamp leaked into the canonical bytes.
    assert b"timestamp" not in first.read_bytes()
    # Nothing was written to the real preflight directory by this test.
    assert not pathlib.Path(
        "logs/eval/v18_depth2_provisional_backup/v18_preflight_criteria.json").exists()


def test_frozen_criteria_stamps_scope_boundary_and_forbids_interpretation():
    d = C.as_dict()
    assert d["scope_boundary"] == C.SCOPE_BOUNDARY
    assert set(C.SCOPE_BOUNDARY) == {
        "mcts_py_edit_authorized", "positive_cap_search_authorized",
        "scientific_acceptance_run_authorized", "commit_authorized",
        "later_stage_authorized"}
    assert all(v is False for v in C.SCOPE_BOUNDARY.values())
    assert d["run_kind"] == "preregistration"
    assert d["scientific_interpretation_forbidden"] is True
    assert d["spec_revision"] == 3
    assert d["mcts_py_unmodified"] is True
    # as_dict is deterministic: runtime provenance belongs to emission only.
    assert C.as_dict() == d
    assert "git_commit" not in d


# --- AUC tail and the operating-characteristic generator --------------------

def test_seed_policy_is_asymmetric_and_frozen_in_the_criteria():
    """The census cannot use the historical XOR rule: game_idx and position_ply
    are both < 1024, so it admits at most 1024 distinct values and the measured
    1,974-row census collapses to 841 -- 1,133 forced duplicates."""
    s = C.SEED_POLICY
    assert s["selected_a"]["rule"] == "historical_xor"
    assert s["selected_a"]["base"] == 20260616
    assert s["selected_a"]["require_unique"] is False
    assert s["selected_a"]["unique_seeds"] == 27
    assert s["selected_a"]["duplicate_groups"] == 3
    assert "historical provenance" in s["selected_a"]["duplicates_are"]
    assert s["census"]["rule"] == "sha1_digest"
    assert s["census"]["base"] == 20260730
    assert s["census"]["domain_tag"] == "v18_preflight_census_seed_v1"
    assert s["census"]["require_unique"] is True
    assert "game_content_sha1" in s["census"]["expression"]
    assert s["cross_population_disjoint_required"] is True
    assert s["checked_before"] == "evaluator_construction"
    # The policy is EMITTED, not merely held in the measurement module.
    assert C.as_dict()["seed_policy"] is s


def test_lower_tail_quantile_is_q_0_05():
    assert C.SEPARATION["lower_bound_quantile"] == 0.05
    assert C.AUC_OC_MODEL["percentile_q"] == 0.05
    assert C.AUC_OC_MODEL["percentile_q"] != 0.95
    assert C.AUC_OC_MODEL["percentile_rank_rule"] == "q * (n - 1)"


def test_quantile_interpolation_on_a_known_vector():
    # n = 11, so rank = q * 10 and the arithmetic is hand-checkable.
    vector = [float(i) for i in range(11)]
    assert C.protocol_quantile(vector, 0.05) == pytest.approx(0.5)    # rank 0.5
    assert C.protocol_quantile(vector, 0.50) == pytest.approx(5.0)    # rank 5.0
    assert C.protocol_quantile(vector, 0.95) == pytest.approx(9.5)    # rank 9.5
    # The two tails must not coincide, or the regression this guards is untestable.
    assert C.protocol_quantile(vector, 0.05) != C.protocol_quantile(vector, 0.95)
    # Fractional rank interpolates linearly: n = 4, rank = 0.05 * 3 = 0.15.
    assert C.protocol_quantile([1.0, 2.0, 3.0, 4.0], 0.05) == pytest.approx(1.15)
    # Input order must not matter; the convention sorts first.
    shuffled = list(vector)
    random.Random(0).shuffle(shuffled)
    assert C.protocol_quantile(shuffled, 0.05) == pytest.approx(0.5)
    assert C.protocol_quantile([7.0], 0.05) == 7.0


def test_upper_quantile_cannot_pass_as_the_lower_bound():
    vector = [0.1 * i for i in range(21)]
    low = C.protocol_quantile(vector, 0.05)
    high = C.protocol_quantile(vector, 0.95)
    assert low < high
    # The frozen rule must select the LOWER endpoint. Revision 5 wrote
    # rank 0.95*(n-1), which names the upper one.
    assert C.auc_lower_bound(vector) == pytest.approx(low)
    assert C.auc_lower_bound(vector) != pytest.approx(high)


def test_gaussian_dgp_delta_mapping_is_sqrt2_probit_of_auc():
    assert C.auc_to_gaussian_delta(0.50) == 0.0
    assert C.auc_to_gaussian_delta(0.70) == pytest.approx(0.7416, abs=5e-5)
    assert C.auc_to_gaussian_delta(0.80) == pytest.approx(1.1902, abs=5e-5)
    assert C.auc_to_gaussian_delta(0.70) == pytest.approx(
        math.sqrt(2.0) * NormalDist().inv_cdf(0.70), abs=1e-12)
    assert C.AUC_OC_MODEL["family"] == "equal_variance_gaussian_location_shift"
    assert C.AUC_OC_MODEL["n_a"] == 30 and C.AUC_OC_MODEL["n_c"] == 30
    assert "NOT distribution-free" in C.AUC_OC_MODEL["label"]


def test_outer_and_bootstrap_streams_are_separate_and_deterministic():
    assert C.AUC_OC_MODEL["outer_seed"] == 20260731
    assert C.AUC_OC_MODEL["bootstrap_seed"] == 20260732
    assert C.AUC_OC_MODEL["outer_seed"] != C.AUC_OC_MODEL["bootstrap_seed"]
    a_one, c_one = C.simulate_dataset(0.70, 0)
    a_two, c_two = C.simulate_dataset(0.70, 0)
    assert list(a_one) == list(a_two) and list(c_one) == list(c_two)
    a_next, _ = C.simulate_dataset(0.70, 1)
    assert list(a_one) != list(a_next)
    # A different true AUC must not reuse the same draws.
    a_other, _ = C.simulate_dataset(0.80, 0)
    assert list(a_one) != list(a_other)
    # The bootstrap stream is keyed independently of the outer stream.
    boot_one = C.bootstrap_aucs(a_one, c_one, replicates=32, index=0)
    boot_two = C.bootstrap_aucs(a_one, c_one, replicates=32, index=0)
    assert list(boot_one) == list(boot_two)
    assert list(boot_one) != list(C.bootstrap_aucs(a_one, c_one, replicates=32, index=1))


def test_oc_table_is_reproducible_across_two_generations():
    kwargs = dict(auc_points=(0.70,), outer_datasets=4, bootstrap_replicates=64)
    assert C.generate_auc_oc_table(**kwargs) == C.generate_auc_oc_table(**kwargs)


def test_oc_generator_reads_no_measurement_artifact(monkeypatch):
    # Pure simulation: no census, no cohort, no preflight artifact. If the
    # generator ever grows a data dependency, this fails loudly.
    def refuse(*args, **kwargs):
        raise AssertionError("the OC generator must not touch the filesystem")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(pathlib.Path, "open", refuse)
    monkeypatch.setattr(pathlib.Path, "read_bytes", refuse)
    monkeypatch.setattr(pathlib.Path, "read_text", refuse)
    table = C.generate_auc_oc_table(
        auc_points=(0.70,), outer_datasets=2, bootstrap_replicates=32)
    assert table["rows"], "the table must actually have been produced"


# --- integration with the producer of the rows these criteria classify ------

def test_classifier_consumes_the_frozen_census_schema():
    # The defect this pins: the classifier once read `exposure`,
    # `abs_root_value_stm`, `count_abs_residual_over_125`,
    # `clipped_amount_at_125` and `max_abs_eligible_residual` -- four aliases
    # and one column the census never emits at all.
    assert C.REQUIRED_CENSUS_FIELDS <= set(C.CENSUS_SCHEMA)
    assert "max_abs_eligible_residual" not in C.CENSUS_SCHEMA
    for alias in ("exposure", "abs_root_value_stm",
                  "count_abs_residual_over_125", "clipped_amount_at_125"):
        assert alias not in C.CENSUS_SCHEMA

    # Every variable named by a role predicate is either a census column or a
    # derived boolean the assignment order computes.
    derived = {"is_flip_control", "is_target"}
    for role, spec in C.ROLE_ASSIGNMENT["roles"].items():
        for cond in spec["conditions"]:
            assert (cond["variable"] in C.CENSUS_SCHEMA
                    or cond["variable"] in derived), (role, cond)

    # A row in the EXACT schema flows through the classifier untouched.
    row = census_row(exposure=9.0, sign_dominance=0.99, root_value_stm=-0.02,
                     would_clip_125=0, clipped_amount_125=0.0, would_clip_05=7)
    assert C.classify_role(row, 1.0) == "target"
    # And the classifier reads nothing outside its declared contract.
    trimmed = {k: v for k, v in row.items() if k in C.REQUIRED_CENSUS_FIELDS}
    assert C.classify_role(trimmed, 1.0) == "target"


def test_primary_exposure_column_is_defined_once_and_used_everywhere():
    # The defect this pins: SEPARATION and EXPOSURE_CUTOFF_RULE named
    # `exposure_at_cap_0.50`, a column the census never emits, so any consumer
    # indexing rows by that contract would have missed the authenticated census.
    assert C.PRIMARY_EXPOSURE_COLUMN in C.CENSUS_SCHEMA
    assert C.SEPARATION["statistic"] == C.PRIMARY_EXPOSURE_COLUMN
    assert C.EXPOSURE_CUTOFF_RULE["statistic"] == C.PRIMARY_EXPOSURE_COLUMN
    assert C.PRIMARY_EXPOSURE_FORMULA["census_column"] == C.PRIMARY_EXPOSURE_COLUMN
    assert C.PRIMARY_EXPOSURE_COLUMN in C.REQUIRED_CENSUS_FIELDS
    target_vars = {c["variable"]
                   for c in C.ROLE_ASSIGNMENT["roles"]["target"]["conditions"]}
    assert C.PRIMARY_EXPOSURE_COLUMN in target_vars
    assert "exposure_at_cap_0.50" not in C.CENSUS_SCHEMA
    # And the retired spelling survives nowhere in the emitted preregistration.
    assert "exposure_at_cap_0.50" not in json.dumps(C.as_dict())


def test_identity_is_would_clip_0_5_equals_zero():
    assert C.IDENTITY_WITNESS["variable"] == "would_clip_0.5"
    assert C.IDENTITY_WITNESS["op"] == "=="
    assert C.IDENTITY_WITNESS["value"] == 0
    assert C.IDENTITY_WITNESS["equivalent_to"] == (
        "max(abs(eligible depth-2 residual)) <= 0.50")
    base = dict(exposure=0.0, sign_dominance=0.0, root_value_stm=0.01,
                would_clip_125=0, clipped_amount_125=0.0)
    # Zero leaves clipped at the strongest cap == no residual exceeds 0.50,
    # because the clip rule is strict.
    assert C.classify_role(census_row(would_clip_05=0, **base), 1.0) == "identity"
    # One leaf over 0.50 disqualifies the witness.
    assert C.classify_role(census_row(would_clip_05=1, **base), 1.0) != "identity"


def test_revisit_form_refuses_a_subset_below_the_floor():
    floor = C.R_MIN_RULE["subset_floor"]
    assert floor == 16
    # 12/16 dense = 0.75, exactly the frozen fraction, admitted.
    assert C.revisit_form([6] * 12 + [1] * 4) == "paired"
    # 11/16 = 0.6875 falls short.
    assert C.revisit_form([6] * 11 + [1] * 5) == "candidate_only_floor"
    # 15 rows cannot decide the criterion at all, however dense.
    with pytest.raises(ValueError) as excinfo:
        C.revisit_form([6] * 15)
    assert C.R_MIN_RULE["subset_below_floor_reason"] in str(excinfo.value)
    with pytest.raises(ValueError):
        C.revisit_form([])


def test_separation_failure_interpretation_is_frozen():
    s = C.SEPARATION
    assert s["on_failure_means"] == (
        "required A-vs-matched-control selectivity was not established")
    assert s["on_failure_does_not_mean"] == "no effect exists"
    assert "no effect" not in s["on_failure_means"]
    assert s["on_failure_rationale"]


def test_sizing_states_its_pass_rule_as_numbers_not_only_prose():
    """Revision 36: `alpha` and the lower-bound floor are consumed by Task 8's
    ladder, its record reconciliation and its count-gate arithmetic. Stating
    them only inside `tier_passes_iff` forced all three to restate the numbers,
    which is how a threshold silently acquires two values."""
    s = C.SIZING
    assert s["alpha"] == 0.05
    assert s["minimum_lower_bound"] == 0.99
    # The prose and the numbers must keep saying the same thing.
    assert f"alpha={s['alpha']}" in s["tier_passes_iff"]
    assert f">= {s['minimum_lower_bound']}" in s["tier_passes_iff"]
    assert str(s["trials_per_probabilistic_tier"]) in s["tier_passes_iff"]


def test_the_frozen_trial_count_is_the_smallest_that_can_pass():
    """299 all-success clears the floor and 298 of 299 does not -- which is why
    a passing tier implies every trial succeeded."""
    from scripts.GPU.alphazero.fpu_dev_corpus_v2 import _binomial_lower_bound
    n, alpha, floor = (C.SIZING["trials_per_probabilistic_tier"],
                       C.SIZING["alpha"], C.SIZING["minimum_lower_bound"])
    assert _binomial_lower_bound(n, n, alpha) >= floor
    assert _binomial_lower_bound(n - 1, n, alpha) < floor
    assert _binomial_lower_bound(n - 1, n - 1, alpha) < floor
