"""Pins the extracted E4 decision rules to the behaviour the preflight qualified.

The rules were extracted verbatim from the preflight's scratch harness so the
execution runner would not re-implement them. This file is what keeps the two in
agreement: it is the preflight self-test's control table, run against the
committed module.
"""
import pytest

from scripts.GPU.alphazero import e4_screen_rules as R

N, BAND = 16, [0.05, 0.95]


@pytest.mark.parametrize("score,played,caps,want", [
    (16.0, 16, 0, "SATURATED_STRONG"),
    (15.5, 16, 0, "SATURATED_STRONG"),      # >= 15.2
    (15.0, 16, 0, "IN_BAND"),               # below 15.2
    (0.0, 16, 0, "SATURATED_WEAK"),
    (0.5, 16, 0, "SATURATED_WEAK"),         # <= 0.8
    (1.0, 16, 0, "IN_BAND"),                # above 0.8
    (8.0, 16, 0, "IN_BAND"),                # all draws
    (0.0, 4, 0, "INCOMPLETE"),              # unfinished, saturation reachable
    (1.0, 2, 0, "INCOMPLETE"),              # cap route still open at game 2
    (4.0, 8, 0, "IN_BAND"),                 # early stop fires at game 8
    (4.0, 8, 1, "INCOMPLETE"),              # one cap reopens the cap route
    (7.5, 8, 0, "INCOMPLETE"),              # strong saturation still reachable
    (8.0, 16, 9, "INCOMPLETE"),             # cap-saturation abort
])
def test_per_endpoint_decision_table(score, played, caps, want):
    assert R.per_endpoint_decision(score, played, N, BAND, caps) == want


def test_early_stop_asymmetry_and_earliest_game():
    assert not R.saturation_reachable(1.0, 2, N, BAND)          # score route closed
    assert R.cap_incompleteness_reachable(2, N, 0)              # cap route open
    assert not R.early_in_band_forced(1.0, 2, N, BAND, 0)
    assert R.early_in_band_forced(4.0, 8, N, BAND, 0)
    assert not R.early_in_band_forced(8.0, 16, N, BAND, 0)      # not an EARLY stop
    assert R.earliest_early_stop(N, BAND) == (8, 1.0)
    assert R.saturation_reachable(0.0, 15, N, BAND)             # never forced before 16


@pytest.mark.parametrize("a,b,want", [
    ("IN_BAND", "IN_BAND", "IN_BAND"),
    ("SATURATED_STRONG", "IN_BAND", "IN_BAND"),
    ("SATURATED_WEAK", "SATURATED_STRONG", "BRACKETED"),
    ("SATURATED_STRONG", "SATURATED_WEAK", "BRACKETED"),
    ("SATURATED_STRONG", "SATURATED_STRONG", "T1J_TOO_STRONG"),
    ("SATURATED_WEAK", "SATURATED_WEAK", "T1J_TOO_WEAK"),
    ("INCOMPLETE", "IN_BAND", "INCONCLUSIVE"),
    ("IN_BAND", "INCOMPLETE", "INCONCLUSIVE"),
])
def test_joint_classifier(a, b, want):
    assert R.classify_joint(a, b) == want


def test_joint_classifier_is_total_and_rejects_nonsense():
    table = R.joint_truth_table()
    assert len(table) == len(R.DECISIONS) ** 2 == 16
    assert all(v for v in table.values())
    with pytest.raises(ValueError):
        R.classify_joint("NONSENSE", "IN_BAND")


def test_incomplete_takes_priority_over_in_band():
    """The report once claimed IN_BAND wins; the code is conservative."""
    assert R.classify_joint("IN_BAND", "INCOMPLETE") == "INCONCLUSIVE"
    assert R.LARGER_MATCH_PERMITTED == ("IN_BAND", "BRACKETED", "INCONCLUSIVE")
