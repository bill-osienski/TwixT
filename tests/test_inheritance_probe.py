import pytest

from scripts.GPU.alphazero.inheritance_probe import (
    DECISION_POINT_SIMS,
    InheritanceProbeConfig,
    InheritanceProbeTracker,
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


class _FakeChild:
    def __init__(self, visit_count):
        self.visit_count = visit_count


class _FakeRoot:
    """Minimal stand-in exposing only what the tracker reads."""

    def __init__(self, visit_count, child_visits):
        self.visit_count = visit_count
        self.children = {i: _FakeChild(v) for i, v in enumerate(child_visits)}


def _ply(tracker, ply, root, before, after, played):
    tracker.observe_search_start(ply=ply, root=root, forced_sims_total=before)
    tracker.observe_search_end(forced_sims_total=after)
    tracker.observe_played_child(visits=played)


def test_tracker_records_one_row_per_search():
    tracker = InheritanceProbeTracker(InheritanceProbeConfig())
    _ply(tracker, 0, _FakeRoot(0, []), 0, 0, 140)
    _ply(tracker, 1, _FakeRoot(140, [3, 0, 9]), 0, 0, None)
    assert len(tracker.rows) == 2
    assert tracker.rows[0].starting_visits == 0
    assert tracker.rows[0].played_child_visits == 140
    assert tracker.rows[1].starting_visited_children == 2
    assert tracker.rows[1].played_child_visits is None


def test_forced_count_belongs_to_the_search_that_produced_it():
    tracker = InheritanceProbeTracker(InheritanceProbeConfig())
    _ply(tracker, 0, _FakeRoot(0, []), 0, 7, 1)
    _ply(tracker, 1, _FakeRoot(1, []), 7, 7, 1)
    _ply(tracker, 2, _FakeRoot(1, []), 7, 10, 1)
    assert [row.forced_count for row in tracker.rows] == [7, 0, 3]


def test_counter_going_backwards_fails_loud():
    tracker = InheritanceProbeTracker(InheritanceProbeConfig())
    tracker.observe_search_start(
        ply=0, root=_FakeRoot(0, []), forced_sims_total=9
    )
    with pytest.raises(ValueError, match="went backwards"):
        tracker.observe_search_end(forced_sims_total=2)


def test_out_of_order_calls_fail_loud():
    tracker = InheritanceProbeTracker(InheritanceProbeConfig())
    with pytest.raises(RuntimeError):
        tracker.observe_search_end(forced_sims_total=0)
    with pytest.raises(RuntimeError):
        tracker.observe_played_child(visits=1)
    tracker.observe_search_start(
        ply=0, root=_FakeRoot(0, []), forced_sims_total=0
    )
    with pytest.raises(RuntimeError):
        tracker.observe_search_start(
            ply=1, root=_FakeRoot(0, []), forced_sims_total=0
        )
    with pytest.raises(RuntimeError):
        tracker.observe_played_child(visits=1)


def test_disabled_tracker_records_nothing():
    tracker = InheritanceProbeTracker(InheritanceProbeConfig(enabled=False))
    _ply(tracker, 0, _FakeRoot(0, []), 0, 5, 5)
    assert tracker.rows == []
