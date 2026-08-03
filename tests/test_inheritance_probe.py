import pytest

from scripts.GPU.alphazero.inheritance_probe import (
    DECISION_POINT_SIMS,
    SearchRow,
    inherited_fraction_320,
    phase_for_ply,
)


@pytest.mark.parametrize(
    "ply,expected",
    [
        (0, "opening"), (30, "opening"),
        (31, "early_mid"), (60, "early_mid"),
        (61, "midgame"), (90, "midgame"),
        (91, "late"), (279, "late"),
    ],
)
def test_phase_boundaries_are_exact(ply, expected):
    assert phase_for_ply(ply) == expected


def test_negative_ply_is_rejected():
    with pytest.raises(ValueError):
        phase_for_ply(-1)


def test_inherited_fraction_uses_the_320_decision_point():
    assert DECISION_POINT_SIMS == 320
    assert inherited_fraction_320(0) == 0.0
    assert inherited_fraction_320(166) == pytest.approx(166 / 486)
    assert inherited_fraction_320(320) == pytest.approx(0.5)


def test_negative_starting_visits_is_rejected():
    with pytest.raises(ValueError):
        inherited_fraction_320(-1)


def test_search_row_computes_its_own_fraction():
    row = SearchRow.build(
        ply=95, starting_visits=166, starting_visited_children=41, forced_count=0
    )
    assert row.phase == "late"
    assert row.inherited_fraction_320 == pytest.approx(166 / 486)
    assert row.played_child_visits is None
