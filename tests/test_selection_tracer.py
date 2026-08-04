"""Atlas Stage 1 -- selection tracer unit tests."""
import pytest

from scripts.GPU.alphazero.selection_tracer import (
    WIDENING_SHAPES,
    SelectionTracer,
    k_of_n,
    n_admit,
)


def test_frozen_shapes_are_pinned():
    assert WIDENING_SHAPES == (("c4a05", 4.0, 0.5), ("c13a03", 13.0, 0.3))


@pytest.mark.parametrize("n,expected", [(400, 80), (105, 41), (20, 18), (5, 9), (1, 4)])
def test_k_of_n_shape_a(n, expected):
    assert k_of_n(n, 4.0, 0.5, n_legal=10_000) == expected


@pytest.mark.parametrize("n,expected", [(400, 79), (105, 53), (20, 32), (5, 22), (1, 13)])
def test_k_of_n_shape_b(n, expected):
    assert k_of_n(n, 13.0, 0.3, n_legal=10_000) == expected


def test_k_is_clamped_by_n_legal_and_floored_at_one():
    assert k_of_n(400, 4.0, 0.5, n_legal=12) == 12
    assert k_of_n(0, 4.0, 0.5, n_legal=500) == 1


def test_n_admit_is_a_search_not_a_closed_form():
    """The closed form ceil((r/C)^(1/alpha)) is WRONG -- it discards the ceil
    inside K. At (C=4, alpha=0.5, r=9) it returns 6, but K(5)=9 >= 9."""
    assert n_admit(9, 4.0, 0.5, n_legal=500) == 5
    # Rank 1 is admitted at n=0 via the max(1, ...) floor.
    assert n_admit(1, 4.0, 0.5, n_legal=500) == 0


class _Node:
    def __init__(self, priors, visit_count=0):
        self.priors = priors
        self.visit_count = visit_count
        self.children = {}


def test_rank_order_is_prior_desc_then_move_id_asc():
    t = SelectionTracer()
    parent = _Node({7: 0.5, 3: 0.5, 9: 0.2})   # 3 and 7 tie on prior
    assert t._ranks_for(parent) == {3: 1, 7: 2, 9: 3}


def test_cache_is_cleared_on_demand():
    t = SelectionTracer()
    t._ranks_for(_Node({1: 1.0}))
    assert t._cache
    t.clear_node_cache()
    assert not t._cache


def test_zero_denominator_yields_none_not_zero():
    snap = SelectionTracer().snapshot()
    for shape, _c, _a in WIDENING_SHAPES:
        cell = snap["by_shape"][shape]["overall"]
        assert cell["outside_rate"] is None
        assert cell["first_touch_outside_rate"] is None
        assert snap["by_shape"][shape]["meaningfully_affected"] is None


def test_forced_root_overrides_leave_the_primary_denominator():
    t = SelectionTracer()
    parent = _Node({1: 0.9, 2: 0.1}, visit_count=5)
    t.on_select_child(parent=parent, selected_move=2, existing_child=None, depth=0,
                      parent_completed_visits=5, root_override=True,
                      within_forced_simulation=True)
    snap = t.snapshot()
    for shape, _c, _a in WIDENING_SHAPES:
        assert snap["by_shape"][shape]["overall"]["eligible_events"] == 0
        assert snap["by_shape"][shape]["forced_root_bypass_events"] == 1


def test_outside_k_counts_and_excluded_mass_accumulate():
    """A rank-9 selection at n=5: K(5)=9 for shape A (inside) and 22 for shape B
    (inside). A rank-10 selection at n=5 is outside shape A but inside B."""
    t = SelectionTracer()
    priors = {mv: 1.0 / (mv + 1) for mv in range(20)}   # strictly decreasing
    parent = _Node(priors, visit_count=5)
    rank10_move = sorted(priors.items(), key=lambda kv: (-kv[1], kv[0]))[9][0]
    t.on_select_child(parent=parent, selected_move=rank10_move, existing_child=None,
                      depth=1, parent_completed_visits=5, root_override=False,
                      within_forced_simulation=False)
    snap = t.snapshot()
    a = snap["by_shape"]["c4a05"]["overall"]
    b = snap["by_shape"]["c13a03"]["overall"]
    assert a["outside_events"] == 1 and a["excluded_prior_mass"] > 0
    assert b["outside_events"] == 0 and b["excluded_prior_mass"] == 0.0
    # Depth bucketing: this was depth 1.
    assert snap["by_shape"]["c4a05"]["1"]["eligible_events"] == 1
    assert snap["by_shape"]["c4a05"]["0"]["eligible_events"] == 0
