"""Atlas Stage 4, Tasks 2 and 3 -- Read-out A.

Features from the FROZEN captures, the ridge classifier and its frozen bars,
deployability, and amendment 6a's dual boundary/B=400 pipeline.

Pure: every input is a plain dict. No reservoir, no checkpoint, no MLX.
"""
import math

import pytest

from scripts.GPU.alphazero.atlas_readout_a import FEATURE_NAMES, collect_features


def _cap(D3=0, root_visits=400, total=390, top=200, second=100, n_vis=30,
         one_vis=5, leader=7, breadth=17, entropy=0.8, n_legal=500):
    return {"D3": D3, "root_visits": root_visits, "total_child_visits": total,
            "top_child_visits": top, "second_child_visits": second,
            "n_visited_children": n_vis, "one_visit_children": one_vis,
            "leader_move": leader, "leader_breadth": breadth,
            "policy_entropy": entropy, "n_legal": n_legal}


def test_exactly_five_frozen_features():
    assert len(FEATURE_NAMES) == 5
    assert set(FEATURE_NAMES) == {
        "one_visit_backup_share", "depth3plus_backup_fraction",
        "leader_visit_margin", "root_policy_entropy", "leader_breadth"}


def test_depth_feature_uses_the_two_point_D3_accounting():
    """(D3(boundary) - D3(start)) / N_actual -- NOT selection events."""
    f = collect_features(_cap(D3=40), _cap(D3=140), n_actual=326)
    assert f["depth3plus_backup_fraction"] == pytest.approx(100 / 326)


def test_the_backup_invariant_is_enforced_here_too():
    with pytest.raises(ValueError):
        collect_features(_cap(D3=140), _cap(D3=40), n_actual=326)      # negative
    with pytest.raises(ValueError):
        collect_features(_cap(D3=0), _cap(D3=999), n_actual=326)       # > N_actual


def test_remaining_features_come_from_the_boundary_capture():
    f = collect_features(_cap(D3=0), _cap(D3=10, one_vis=6, n_vis=30, top=200,
                                          second=100, total=400, entropy=0.77,
                                          breadth=17), n_actual=326)
    assert f["one_visit_backup_share"] == pytest.approx(6 / 30)
    assert f["leader_visit_margin"] == pytest.approx((200 - 100) / 400)
    assert f["root_policy_entropy"] == pytest.approx(0.77)
    assert f["leader_breadth"] == 17


def test_undefined_features_are_None_not_zero():
    f = collect_features(
        _cap(D3=0),
        _cap(D3=0, n_vis=0, one_vis=0, top=None, second=None, total=0,
             entropy=None, breadth=None),
        n_actual=326)
    for k in ("one_visit_backup_share", "leader_visit_margin",
              "root_policy_entropy", "leader_breadth"):
        assert f[k] is None, f"{k} must be None when undefined"


def test_a_single_visited_child_has_no_margin():
    f = collect_features(_cap(D3=0), _cap(D3=0, n_vis=1, second=None),
                         n_actual=326)
    assert f["leader_visit_margin"] is None


def test_zero_N_actual_yields_None_not_a_division_error():
    f = collect_features(_cap(D3=0), _cap(D3=0), n_actual=0)
    assert f["depth3plus_backup_fraction"] is None


# -- Task 3: classifier, string labels, bars, deployability, dual pipeline ----

from scripts.GPU.alphazero.atlas_readout_a import (
    AUTHORITATIVE_FEATURE_SET, FEATURE_SETS, INSUFFICIENCY_VERDICTS, LABEL_TO_Y,
    auc, bootstrap_auc_lower_bound, deployability, evaluate_detector,
    evaluate_detector_both, fit_ridge_logistic, prepare_rows, standardize,
)


def _row(label, **feats):
    base = {k: 0.5 for k in FEATURE_NAMES}
    base.update(feats)
    return {"label": label, "features": base}


def test_string_labels_map_explicitly_and_other_classes_are_dropped():
    assert LABEL_TO_Y == {"misleading": 1, "stable_negative": 0}
    r = prepare_rows([_row("misleading"), _row("stable_negative"),
                      _row("ambiguous"), _row("no_stable_reference")])
    assert r["y"] == [1, 0]
    assert r["dropped_ineligible"] == 2


def test_rows_with_a_missing_feature_are_REJECTED_and_reported():
    bad = _row("misleading"); bad["features"]["leader_breadth"] = None
    r = prepare_rows([bad, _row("stable_negative")])
    assert r["rejected_missing_features"] == 1
    assert len(r["y"]) == 1


def test_capacity_is_rechecked_AFTER_rejection():
    """Rejection is exactly what can push a split below its own gate."""
    rows = [_row("misleading") for _ in range(20)] + \
           [_row("stable_negative") for _ in range(25)]
    rows[0]["features"]["root_policy_entropy"] = None       # one rejection
    r = evaluate_detector(discovery=rows, validation=rows)
    assert r["verdict"] == "INSUFFICIENT_CLASSES"
    assert r["n_misleading"] == 19


def test_fitting_requires_both_DISCOVERY_classes():
    disc = [_row("misleading") for _ in range(30)]           # one class only
    val = [_row("misleading") for _ in range(20)] + \
          [_row("stable_negative") for _ in range(25)]
    r = evaluate_detector(discovery=disc, validation=val)
    assert r["verdict"] == "INSUFFICIENT_DISCOVERY_CLASSES"


def test_auc_is_one_for_perfect_separation_and_half_for_none():
    assert auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert auc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == pytest.approx(0.5)


def test_auc_is_None_when_a_class_is_absent():
    assert auc([0.1, 0.2], [0, 0]) is None       # None, never a defaulted 0.5


def test_standardization_stats_come_from_discovery_only():
    disc = [{"a": 1.0}, {"a": 3.0}]
    _z, stats = standardize(disc, feature_names=("a",))
    z_val, stats2 = standardize([{"a": 5.0}], feature_names=("a",), stats=stats)
    assert stats2 == stats
    assert z_val[0][0] == pytest.approx((5.0 - 2.0) / stats["a"][1])


def test_standardize_rejects_a_missing_feature_rather_than_imputing():
    with pytest.raises(ValueError, match="missing features"):
        standardize([{"a": None}], feature_names=("a",))


def test_ridge_logistic_separates_a_linearly_separable_set():
    X, y = [[-2.0], [-1.0], [1.0], [2.0]], [0, 0, 1, 1]
    model = fit_ridge_logistic(X, y)
    assert auc([model["predict"](x) for x in X], y) == pytest.approx(1.0)


def test_deployability_fails_when_the_MEDIAN_remaining_is_zero():
    r = deployability([0, 0, 0, 40, 60])
    assert r["median_remaining"] == 0 and r["verdict"] == "NOT_DEPLOYABLE"
    assert r["zero_budget_fraction"] == pytest.approx(3 / 5)


def test_deployability_passes_with_a_positive_median_and_reports_quartiles():
    r = deployability([10, 40, 60, 70, 80])
    assert r["verdict"] == "DEPLOYABLE" and len(r["quartiles"]) == 3


def test_deployability_reports_strata_without_gating_on_them():
    r = deployability([0, 40, 60], strata={"late": [0, 0], "midgame": [60]})
    assert set(r["by_stratum"]) == {"late", "midgame"}
    assert "verdict" not in r["by_stratum"]["late"]


def test_deployability_of_an_empty_set_is_None_not_zero():
    r = deployability([])
    assert r["median_remaining"] is None and r["zero_budget_fraction"] is None
    assert r["verdict"] == "NO_ROWS"


# -- section 6a: the IDENTICAL pipeline on BOTH feature sets ------------------

def _dual_rows(boundary_separates, four_hundred_separates,
               n_misleading=20, n_stable=60):
    """20 misleading + 60 stable-negative: the smallest set that satisfies the
    frozen capacity gate (>=20 / >=25) AND can clear the <=25% flag-rate bar,
    since 20/80 is exactly 0.25. At 45 rows a perfect detector would fail its
    own flag rate at 44%, and the test would pin the wrong thing.

    The class counts are parameters because the ceiling binds them: a perfect
    detector flags every positive, so `n_misleading / (n_misleading + n_stable)`
    must stay <= 0.25 for a PASS to be reachable at all. Adding positives to a
    fixture is therefore NOT a safe way to grow it.
    """
    rows = []
    for i in range(n_misleading + n_stable):
        label = "misleading" if i < n_misleading else "stable_negative"
        hi = 1.0 if label == "misleading" else 0.0
        rows.append({
            "label": label,
            "features_at_boundary": {k: (hi if boundary_separates else 0.5)
                                     for k in FEATURE_NAMES},
            "features_at_400": {k: (hi if four_hundred_separates else 0.5)
                                for k in FEATURE_NAMES},
        })
    return rows


def test_the_pipeline_runs_on_both_frozen_feature_sets():
    assert FEATURE_SETS == ("features_at_boundary", "features_at_400")
    assert AUTHORITATIVE_FEATURE_SET == "features_at_boundary"
    rows = _dual_rows(True, True)
    r = evaluate_detector_both(rows, rows, replicates=64)
    assert set(r["per_feature_set"]) == set(FEATURE_SETS)
    assert r["authoritative"] == AUTHORITATIVE_FEATURE_SET


def test_the_boundary_result_is_authoritative_when_it_passes():
    rows = _dual_rows(True, False)
    r = evaluate_detector_both(rows, rows, replicates=64)
    assert r["per_feature_set"]["features_at_boundary"]["verdict"] == "PASS"
    assert r["per_feature_set"]["features_at_400"]["verdict"] == "FAIL"
    assert r["verdict"] == "PASS"            # a failing B=400 cannot demote it


def test_LATE_ONLY_SEPARATION_requires_boundary_FAIL_and_400_PASS():
    """Both, exactly. It is section 6's stated FAILURE condition: the
    information exists but arrives too late to allocate the last 80 sims."""
    rows = _dual_rows(False, True)
    r = evaluate_detector_both(rows, rows, replicates=64)
    assert r["per_feature_set"]["features_at_boundary"]["verdict"] == "FAIL"
    assert r["per_feature_set"]["features_at_400"]["verdict"] == "PASS"
    assert r["verdict"] == "LATE_ONLY_SEPARATION"


def test_both_sets_failing_is_a_plain_FAIL():
    rows = _dual_rows(False, False)
    assert evaluate_detector_both(rows, rows, replicates=64)["verdict"] == "FAIL"


def test_a_boundary_insufficiency_is_reported_as_itself_not_as_lateness():
    """Absence of evidence is never evidence of timing: a B=400 PASS must not
    promote an insufficient boundary result to LATE_ONLY_SEPARATION."""
    rows = _dual_rows(False, True)
    rows[0]["features_at_boundary"]["leader_breadth"] = None   # 19 misleading
    r = evaluate_detector_both(rows, rows, replicates=64)
    assert r["per_feature_set"]["features_at_boundary"]["verdict"] in \
        INSUFFICIENCY_VERDICTS
    assert r["per_feature_set"]["features_at_400"]["verdict"] == "PASS"
    assert r["verdict"] == "INSUFFICIENT_CLASSES"


def test_a_boundary_REJECTION_blocks_LATE_ONLY_SEPARATION():
    """Amendment 4 lists a missing-feature rejection alongside the two
    insufficiency verdicts as something that cannot establish lateness.

    Rejections shrink the boundary sample, and a smaller sample is MORE likely
    to miss the AUC and bootstrap bars -- so a rejection is exactly what could
    manufacture the FAIL half of a lateness finding.

    Sizing is exact and both halves are load-bearing. 21/84 keeps the B=400
    flag rate at exactly the 0.25 ceiling, so a perfect detector still PASSES;
    rejecting one boundary misleading row leaves 20/83, which still clears the
    >=20 / >=25 capacity gate, so the boundary reaches the bars and returns a
    genuine FAIL rather than an INSUFFICIENT_CLASSES. Adding three MISLEADING
    rows instead would give 23/83 = 27.7% and the B=400 half could never pass.
    """
    rows = _dual_rows(False, True, n_misleading=21, n_stable=63)
    rows[0]["features_at_boundary"]["leader_breadth"] = None
    r = evaluate_detector_both(rows, rows, replicates=64)
    assert r["per_feature_set"]["features_at_boundary"]["verdict"] == "FAIL"
    assert r["per_feature_set"]["features_at_400"]["verdict"] == "PASS"
    # Reported as the boundary's OWN result, not promoted to a timing finding.
    assert r["verdict"] == "FAIL"
    assert r["lateness_blocked_by"] == "boundary_missing_feature_rejections"


def test_row_overlap_is_reported_for_DISCOVERY_and_validation_separately():
    """Missing discovery features make the two models train on different rows,
    which a validation-only overlap would report as identical row sets."""
    rows = _dual_rows(True, True)
    disc = [dict(r, features_at_400=dict(r["features_at_400"])) for r in rows]
    disc[0]["features_at_400"]["root_policy_entropy"] = None
    r = evaluate_detector_both(disc, rows, replicates=64)
    assert r["per_feature_set"]["features_at_400"]["rejected_missing_features"] == 0
    assert r["row_overlap"]["validation"]["identical"] is True
    assert r["row_overlap"]["discovery"]["identical"] is False
    assert r["row_overlap"]["discovery"]["n_common"] == 79
