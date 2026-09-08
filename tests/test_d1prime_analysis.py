"""D1' analysis -- the FROZEN plan (docs/superpowers/2026-09-07-t1j-d1prime-diagnostic-proposal.md)
implemented against SYNTHETIC cohorts and MOCKED readouts only.

Nothing here reads a real D1 record, a real confirmation game, a model or a JVM.
The only randomness is the plan's analysis-only PRNG (PCG64, seed 20260907) on
synthetic fixtures. Every §9 control of the plan has a test below whose failure
mode is the named defect.
"""
import inspect
import itertools

import numpy as np
import pytest

from scripts.GPU.alphazero import d1prime_analysis as DP

OPENINGS = ("o1_center", "o2_offcenter", "o3_low", "o4_high",
            "o5_wide_left", "o6_wide_right", "o7_diagonal", "o8_contact")
ARMS = ("t1j_red", "t1j_black")
PHASES = ("opening", "early", "middle", "late")


def _gid(opening, arm, rep):
    return f"g-{opening}-{arm}-r{rep}"


def _design(reps=(0, 1)):
    """A synthetic L0-shaped design: opening-major, then arm, then rep."""
    return [{"task_id": _gid(o, a, r), "opening": o, "colour_arm": a, "rep": r}
            for o in OPENINGS for a in ARMS for r in reps]


_digest_counter = itertools.count(1)


def _row(opening, arm, phase, role, task_id, lprd, *, sig="mover_fragmentation",
         ply=10, digest=None, agree=False):
    return {"task_id": task_id, "opening": opening, "colour_arm": arm, "phase": phase,
            "signature": sig, "role": role, "ply": ply,
            "digest": digest or f"d{next(_digest_counter):06d}",
            "lprd": bool(lprd), "agree": bool(agree)}


def _cohort(cells, *, pos_rate, ctl_rate, per_role=3, reps=(0, 1), sig="mover_fragmentation"):
    """`per_role` positions and controls in every cell, spread over BOTH games of
    the cell's stratum, with exact within-cell LPRD rates."""
    rows = []
    for (o, a, ph) in cells:
        for role, rate in (("position", pos_rate), ("control", ctl_rate)):
            n_true = round(rate * per_role)
            for i in range(per_role):
                rows.append(_row(o, a, ph, role, _gid(o, a, reps[i % len(reps)]),
                                 i < n_true, sig=sig))
    return rows


ALL_16 = [(o, a, "middle") for o in OPENINGS for a in ARMS]
STRATA = None   # built lazily from _design()


def _strata():
    return DP.design_strata(_design(), reps=(0, 1))


# ─────────────────────────── frozen constants ───────────────────────────────

def test_the_frozen_constants_match_the_plan():
    assert DP.K_LPRD == 5
    assert DP.T_THRESHOLD == 0.15
    assert DP.B_REPLICATES == 10_000
    assert DP.BOOTSTRAP_SEED == 20260907
    assert DP.FLOOR == {"cells": 8, "positions": 40, "controls": 30, "games": 12}
    assert DP.CONFIRMATION_CEILING == 240
    assert DP.CONFIRMATION_MIN_PLY == 5
    assert DP.PRIMARY_COHORT == "mover_fragmentation"
    assert DP.SECONDARY_COHORT == "created_threat"
    assert DP.ARMS == ("t1j_red", "t1j_black")


# ─────────────────────────── LPRD and ranks ─────────────────────────────────

def test_rank_raw_orders_by_mass_then_breaks_ties_by_row_col_only():
    policy = {"3,3": 0.2, "1,1": 0.2, "2,2": 0.6, "0,5": 0.2}
    ranks = DP.rank_raw(policy)
    assert ranks[(2, 2)] == 1
    assert ranks[(0, 5)] == 2 and ranks[(1, 1)] == 3 and ranks[(3, 3)] == 4
    # BY CONSTRUCTION never by visits: the function has no visits parameter
    assert list(inspect.signature(DP.rank_raw).parameters) == ["policy"]


def _d1_position(*, policy, our_move, t1j_move_6, visits=None, overrode=False, **labels):
    """A D1 per-position record in the shape `d1_probe` writes (mocked)."""
    r, c = our_move
    rec = {"row": r, "col": c, "raw_policy": dict(policy),
           "root_visits": visits or {k: 1 for k in policy}, "overrode_leader": overrode}
    depths = [{"depth": 3, "move": [t1j_move_6[0], t1j_move_6[1]]},
              {"depth": 6, "move": [t1j_move_6[0], t1j_move_6[1]]}]
    base = {"task_id": "g-o1_center-t1j_red-r0", "opening": "o1_center",
            "colour_arm": "t1j_red", "phase": "middle", "signature": "mover_fragmentation",
            "role": "position", "ply": 10, "digest": "dX"}
    base.update(labels)
    return {**base, "incumbent": rec, "depths": depths}


def _policy_with_ranks(n=10):
    """Moves (0,0)..(0,n-1) with strictly decreasing mass: rank == col + 1."""
    return {f"0,{i}": (n - i) / 100 for i in range(n)}


def test_lprd_is_true_when_t1j_move_ranks_below_fifth_and_false_otherwise():
    pol = _policy_with_ranks()
    assert DP.position_row(_d1_position(policy=pol, our_move=(0, 0), t1j_move_6=(0, 6)))["lprd"] is True
    assert DP.position_row(_d1_position(policy=pol, our_move=(0, 0), t1j_move_6=(0, 4)))["lprd"] is False
    assert DP.position_row(_d1_position(policy=pol, our_move=(0, 0), t1j_move_6=(0, 5)))["lprd"] is True


def test_agree_AND_lprd_can_both_be_true_neither_implies_the_other():
    """§3 correction (1), the positive test: our 400-sim readout chose the move
    the raw policy ranks 7th, and T1j chose the same move."""
    pol = _policy_with_ranks()
    row = DP.position_row(_d1_position(policy=pol, our_move=(0, 6), t1j_move_6=(0, 6)))
    assert row["rank_t1j"] == 7
    assert row["lprd"] is True and row["agree"] is True
    row2 = DP.position_row(_d1_position(policy=pol, our_move=(0, 0), t1j_move_6=(0, 6)))
    assert row2["lprd"] is True and row2["agree"] is False
    row3 = DP.position_row(_d1_position(policy=pol, our_move=(0, 1), t1j_move_6=(0, 1)))
    assert row3["lprd"] is False and row3["agree"] is True


def test_position_row_reads_the_mdPly_6_move_not_the_depth_3_move():
    pol = _policy_with_ranks()
    pos = _d1_position(policy=pol, our_move=(0, 0), t1j_move_6=(0, 8))
    pos["depths"][0]["move"] = [0, 1]                  # depth 3 disagrees; irrelevant
    assert DP.position_row(pos)["t1j_move_6"] == (0, 8)


def test_position_row_refuses_a_t1j_move_outside_the_legal_policy():
    pol = _policy_with_ranks()
    with pytest.raises(DP.D1PrimeError, match="not in the raw policy"):
        DP.position_row(_d1_position(policy=pol, our_move=(0, 0), t1j_move_6=(9, 9)))


def test_rows_from_d1_report_maps_every_position_and_keeps_labels():
    pol = _policy_with_ranks()
    report = {"positions": [
        _d1_position(policy=pol, our_move=(0, 0), t1j_move_6=(0, 6), task_id="tA", role="position"),
        _d1_position(policy=pol, our_move=(0, 0), t1j_move_6=(0, 1), task_id="tB", role="control")]}
    rows = DP.rows_from_d1_report(report)
    assert [r["task_id"] for r in rows] == ["tA", "tB"]
    assert [r["lprd"] for r in rows] == [True, False]
    assert rows[0]["role"] == "position" and rows[1]["role"] == "control"
    assert "visit_share_t1j" in rows[0] and "overrode_leader" in rows[0]


# ───────────────────── the matched statistic T over common support ──────────

def test_MH_weighted_statistic_known_answer_two_cells():
    """Cell A: 3 positions (all lprd) vs 1 control (not) -> diff 1.0, w = 3*1/4 = 0.75.
    Cell B: 2 vs 2, both rates 0.5 -> diff 0, w = 2*2/4 = 1.0.
    T = (0.75*1.0 + 1.0*0) / 1.75 = 0.428571..."""
    A, B = ("o1_center", "t1j_red", "middle"), ("o2_offcenter", "t1j_red", "middle")
    rows = [_row(*A, "position", "gA", True) for _ in range(3)] + [_row(*A, "control", "gA", False)]
    rows += [_row(*B, "position", "gB", True), _row(*B, "position", "gB", False),
             _row(*B, "control", "gB", True), _row(*B, "control", "gB", False)]
    st = DP.matched_statistic(rows, cohort="mover_fragmentation")
    assert st["T"] == pytest.approx(0.75 / 1.75)
    assert st["n_cs_cells"] == 2 and st["n_positions_cs"] == 5 and st["n_controls_cs"] == 3
    weights = {tuple(c["cell"]): c["weight"] for c in st["cells"]}
    assert weights[A] == pytest.approx(0.75) and weights[B] == pytest.approx(1.0)


def test_common_support_BINDS_an_excess_that_lives_only_in_control_less_cells_is_nothing():
    """§9: a cohort whose excess is entirely in cells without controls -> T over
    common support is 0; the unmatched rows are reported, never pooled."""
    cs = [(o, "t1j_red", "middle") for o in OPENINGS]
    rows = _cohort(cs, pos_rate=1 / 3, ctl_rate=1 / 3)            # matched: no excess
    for o in OPENINGS:                                               # unmatched: all lprd
        rows += [_row(o, "t1j_black", "late", "position", _gid(o, "t1j_black", 0), True)
                 for _ in range(3)]
    st = DP.matched_statistic(rows, cohort="mover_fragmentation")
    assert st["T"] == pytest.approx(0.0)
    assert st["unmatched"]["n_positions"] == 24 and st["unmatched"]["lprd_rate"] == 1.0
    assert st["n_cs_cells"] == 8


def test_the_statistic_is_None_when_no_cell_has_both_roles():
    rows = [_row("o1_center", "t1j_red", "middle", "position", "g", True)]
    assert DP.matched_statistic(rows, cohort="mover_fragmentation")["T"] is None


def test_the_statistic_only_reads_rows_of_the_requested_cohort():
    rows = _cohort(ALL_16, pos_rate=1.0, ctl_rate=0.0, sig="created_threat")
    assert DP.matched_statistic(rows, cohort="mover_fragmentation")["T"] is None
    assert DP.matched_statistic(rows, cohort="created_threat")["T"] == pytest.approx(1.0)


# ───────────────────────── eligibility floors ───────────────────────────────

def _floor_cohort(n_cells, n_pos, n_ctl, n_games):
    """Exactly n_cells common-support cells with n_pos positions and n_ctl controls
    in total, from exactly n_games distinct games. Rows outside common support
    are added to make the game count independent of the cell count."""
    cells = [(OPENINGS[i % 8], ARMS[(i // 8) % 2], PHASES[(i // 16) % 4]) for i in range(n_cells)]
    games = [f"game{i}" for i in range(n_games)]
    rows = []
    for i in range(n_pos):
        rows.append(_row(*cells[i % n_cells], "position", games[i % n_games], i % 2))
    for i in range(n_ctl):
        rows.append(_row(*cells[i % n_cells], "control", games[i % n_games], i % 2))
    return rows


@pytest.mark.parametrize("cells,pos,ctl,games,ok", [
    (8, 40, 30, 12, True),
    (7, 40, 30, 12, False), (8, 39, 30, 12, False), (8, 40, 29, 12, False), (8, 40, 30, 11, False),
])
def test_the_eligibility_floor_accepts_AT_the_floor_and_refuses_one_below(cells, pos, ctl, games, ok):
    st = DP.matched_statistic(_floor_cohort(cells, pos, ctl, games), cohort="mover_fragmentation")
    met, detail = DP.floors_met(st)
    assert met is ok, detail
    assert set(detail) == {"cells", "positions", "controls", "games"}


def test_contributing_games_count_rows_of_EITHER_role_in_common_support_only():
    """Interpretation recorded: a game contributes if it holds >= 1 row of either
    role in a common-support cell; unmatched rows do not make a game contribute."""
    A = ("o1_center", "t1j_red", "middle")
    rows = [_row(*A, "position", "gP", True), _row(*A, "control", "gC", False),
            _row("o2_offcenter", "t1j_black", "late", "position", "gU", True)]   # unmatched
    st = DP.matched_statistic(rows, cohort="mover_fragmentation")
    assert st["games_cs"] == 2


# ───────────────── stratified whole-game resampling, frozen ─────────────────

def test_design_strata_are_the_16_opening_x_arm_cells_in_frozen_order_with_2_games_each():
    strata = _strata()
    assert [s["key"] for s in strata] == [(o, a) for o in OPENINGS for a in ARMS]
    assert all(len(s["games"]) == 2 for s in strata)
    assert strata[0]["games"] == [_gid("o1_center", "t1j_red", 0), _gid("o1_center", "t1j_red", 1)]


def test_design_strata_refuse_a_stratum_without_exactly_two_games():
    with pytest.raises(DP.D1PrimeError, match="exactly 2"):
        DP.design_strata(_design(reps=(0, 1, 2)), reps=(0, 1, 2))


def _parallel_draws(seed, strata, B):
    """The plan's draw sequence, re-derived independently: one Generator, per
    stratum `integers(0, 2, size=2)` in stratum order, replicates in sequence."""
    rng = np.random.default_rng(seed)
    return [[rng.integers(0, 2, size=2).tolist() for _ in strata] for _ in range(B)]


def test_resampling_draws_whole_games_within_strata_with_the_pinned_PRNG_and_order():
    strata = _strata()
    rows = _cohort(ALL_16, pos_rate=2 / 3, ctl_rate=1 / 3)
    B = 5
    draws = _parallel_draws(DP.BOOTSTRAP_SEED, strata, B)
    reps = DP.replicates(rows, strata, B=B, seed=DP.BOOTSTRAP_SEED)
    assert len(reps) == B
    for b in range(B):
        # expected multiset of games: for each stratum, the two drawn games
        expected = []
        for s, (i, j) in zip(strata, draws[b]):
            expected += [s["games"][i], s["games"][j]]
        got = [g for g in reps[b]["games"]]
        assert got == expected, b
        # rows travel WHOLE and with MULTIPLICITY
        by_game = {}
        for r in rows:
            by_game.setdefault(r["task_id"], []).append(r)
        want_rows = sum((by_game.get(g, []) for g in expected), [])
        assert len(reps[b]["rows"]) == len(want_rows)
        assert sorted(r["digest"] for r in reps[b]["rows"]) == sorted(r["digest"] for r in want_rows)


def test_multiplicity_is_RETAINED_a_game_drawn_twice_counts_twice_and_nothing_is_rededuplicated():
    strata = _strata()[:1]
    A = ("o1_center", "t1j_red", "middle")
    gA, gB = strata[0]["games"]
    rows = [_row(*A, "position", gA, True), _row(*A, "position", gA, False),
            _row(*A, "control", gB, False), _row(*A, "control", gA, True)]
    rng = np.random.default_rng(0)
    rep = DP.resample_once(rows, strata, rng)
    n_from_A = rep["games"].count(gA)
    assert len(rep["rows"]) == n_from_A * 3 + rep["games"].count(gB) * 1
    if n_from_A == 2:
        assert sorted(r["digest"] for r in rep["rows"]).count(rows[0]["digest"]) == 2


def test_a_replicate_that_loses_a_role_in_a_cell_drops_that_cell_and_is_counted():
    """Positions in game A only, controls in game B only, one stratum: the draw
    AA (or BB) leaves the cell with one role -> that replicate's T is undefined;
    it is COUNTED, never redrawn or dropped, and the outcome says so."""
    strata = _strata()[:1]
    A = ("o1_center", "t1j_red", "middle")
    gA, gB = strata[0]["games"]
    rows = [_row(*A, "position", gA, True) for _ in range(3)] + \
           [_row(*A, "control", gB, False) for _ in range(3)]
    B = 200
    draws = _parallel_draws(DP.BOOTSTRAP_SEED, strata, B)
    expected_undefined = sum(1 for d in draws if d[0][0] == d[0][1])
    si = DP.stability_interval(rows, strata, cohort="mover_fragmentation", B=B,
                               seed=DP.BOOTSTRAP_SEED)
    assert si["undefined"] == expected_undefined > 0
    assert len(si["replicates"]) == B, "replicates were dropped or redrawn"
    assert si["interval"] == "UNDEFINED"
    assert si["defined_cells"] == {0: expected_undefined, 1: B - expected_undefined}


def test_the_stability_interval_is_the_type_7_central_95_percent_range():
    assert DP.central_interval([1, 2, 3, 4]) == pytest.approx((1.075, 3.925))
    assert DP.central_interval(list(range(1, 101))) == pytest.approx((3.475, 97.525))


def test_two_runs_produce_IDENTICAL_replicate_vectors():
    strata = _strata()
    rows = _cohort(ALL_16, pos_rate=2 / 3, ctl_rate=1 / 3)
    a = DP.stability_interval(rows, strata, cohort="mover_fragmentation", B=50)
    b = DP.stability_interval(rows, strata, cohort="mover_fragmentation", B=50)
    assert a["replicates"] == b["replicates"]
    assert a["prng"] == {"bit_generator": "PCG64", "seed": 20260907, "B": 50}


def test_a_perfectly_balanced_cohort_gives_a_degenerate_interval_at_T():
    """Both roles in both games of every stratum with exact rates: every
    reweighting reproduces the same cell rates, so every replicate equals T."""
    strata = _strata()
    rows = _cohort(ALL_16, pos_rate=1.0, ctl_rate=2 / 3, per_role=6)
    si = DP.stability_interval(rows, strata, cohort="mover_fragmentation", B=100)
    assert si["T"] == pytest.approx(1 / 3)
    assert si["interval"] == pytest.approx((1 / 3, 1 / 3))
    assert si["undefined"] == 0


# ─────────────────────── development decision rule ──────────────────────────

def _decide(rows, **kw):
    return DP.development_decision(rows, _strata(), B=100, **kw)


def test_a_symmetric_synthetic_cohort_is_NO_GO():
    rows = _cohort(ALL_16, pos_rate=1 / 3, ctl_rate=1 / 3)
    d = _decide(rows)
    assert d["outcome"] == "NO_GO" and d["T"] == pytest.approx(0.0)


def test_an_injected_excess_of_0_30_at_signature_positions_is_GO():
    rows = _cohort(ALL_16, pos_rate=2 / 3, ctl_rate=1 / 3, per_role=6)   # 16 cells, 96/96, 32 games
    d = _decide(rows)
    assert d["T"] == pytest.approx(1 / 3)
    assert d["interval"][0] > 0
    assert d["outcome"] == "GO"


def test_the_same_excess_at_CONTROLS_is_NO_GO_direction_binds():
    rows = _cohort(ALL_16, pos_rate=1 / 3, ctl_rate=2 / 3, per_role=6)
    d = _decide(rows)
    assert d["T"] == pytest.approx(-1 / 3) and d["outcome"] == "NO_GO"


def test_an_excess_AT_OR_ABOVE_0_15_is_GO_and_one_below_is_NO_GO_despite_a_positive_lower_bound():
    rows = _cohort(ALL_16, pos_rate=0.5, ctl_rate=1 / 3, per_role=6)       # 3/6 vs 2/6 = +0.1667
    d = _decide(rows)
    assert d["T"] == pytest.approx(1 / 6) and d["outcome"] == "GO"
    rows = _cohort(ALL_16, pos_rate=0.5, ctl_rate=5 / 12, per_role=12)     # 6/12 vs 5/12 = +0.0833
    d = _decide(rows)
    assert d["T"] == pytest.approx(1 / 12) and d["interval"][0] > 0 and d["outcome"] == "NO_GO"


def test_an_effect_carried_by_ONE_game_has_a_lower_bound_at_zero_and_is_NO_GO():
    """T >= 0.15 but unstable: six strata (12 games) meet the floor; five are
    balanced (diff 0); in the sixth, every position in game r0 is lprd and every
    one in r1 is not, controls never are -- so replicates drawing r1 twice put
    that stratum's four cells at 0 and T at 0. About a quarter of replicates do,
    so the 2.5% quantile is 0: the lower bound is NOT > 0 -> NO_GO. A rule that
    read only T would say GO."""
    strata = _strata()[:6]
    rows = []
    for s in strata[:5]:
        (o, a), (gA, gB) = s["key"], s["games"]
        for ph in PHASES:
            rows += [_row(o, a, ph, "position", g, i % 2) for i, g in enumerate([gA, gB, gA])]
            rows += [_row(o, a, ph, "control", g, i % 2) for i, g in enumerate([gA, gB, gA])]
    (o, a), (gA, gB) = strata[5]["key"], strata[5]["games"]
    for ph in PHASES:
        rows += [_row(o, a, ph, "position", gA, True) for _ in range(9)]
        rows += [_row(o, a, ph, "position", gB, False) for _ in range(9)]
        rows += [_row(o, a, ph, "control", gA, False) for _ in range(9)]
        rows += [_row(o, a, ph, "control", gB, False) for _ in range(9)]
    d = DP.development_decision(rows, strata, B=400)
    assert d["T"] == pytest.approx(18 / 66)         # 4 cells x w=9 x 0.5 = 18, over 4x9 + 20x1.5 = 66
    assert d["interval"][0] == pytest.approx(0.0)
    assert d["outcome"] == "NO_GO"


def test_insufficient_support_is_its_own_NO_GO_and_computes_no_interval():
    rows = _floor_cohort(7, 40, 30, 12)
    d = _decide(rows)
    assert d["outcome"] == "NO_GO — insufficient support"
    assert d["interval"] is None and d["floor"]["cells"] == 7


def test_bootstrap_undefined_is_its_own_NO_GO():
    """Six strata (12 games, the floor); in each, positions only in game r0 and
    controls only in game r1, four phase cells per stratum (24 cells). A
    replicate is undefined iff EVERY stratum draws AA or BB (prob 2^-6 each).
    The count is re-derived from the pinned draw sequence, must be > 0, and no
    replicate may be dropped or redrawn."""
    strata = _strata()[:6]
    rows = []
    for s in strata:
        (o, a), (gA, gB) = s["key"], s["games"]
        for ph in PHASES:
            rows += [_row(o, a, ph, "position", gA, i % 2) for i in range(3)]
            rows += [_row(o, a, ph, "control", gB, (i + 1) % 2) for i in range(3)]
    st = DP.matched_statistic(rows, cohort="mover_fragmentation")
    assert DP.floors_met(st)[0], "the fixture must meet the floor so the bootstrap branch is reached"
    B = 500
    draws = _parallel_draws(DP.BOOTSTRAP_SEED, strata, B)
    expected_undefined = sum(1 for d in draws if all(i == j for i, j in d))
    assert expected_undefined > 0, "the pinned seed must yield an undefined replicate for this test to bind"
    d = DP.development_decision(rows, strata, B=B)
    assert d["outcome"] == "NO_GO — bootstrap undefined" and d["interval"] == "UNDEFINED"
    assert d["stability"]["undefined"] == expected_undefined
    assert len(d["stability"]["replicates"]) == B


def test_the_secondary_cohort_is_reported_but_cannot_produce_GO():
    rows = _cohort(ALL_16, pos_rate=1 / 3, ctl_rate=1 / 3)                        # primary: nothing
    rows += _cohort(ALL_16, pos_rate=1.0, ctl_rate=0.0, per_role=6, sig="created_threat")
    d = _decide(rows)
    assert d["outcome"] == "NO_GO"
    assert d["secondary"]["T"] == pytest.approx(1.0)


# ─────────────────────── confirmation decision rule ─────────────────────────

def _confirm(rows, **kw):
    return DP.confirmation_decision(rows, _strata(), B=100, **kw)


def test_confirmation_GO_needs_both_arms_defined_and_positive():
    rows = _cohort(ALL_16, pos_rate=2 / 3, ctl_rate=1 / 3, per_role=6)
    d = _confirm(rows)
    assert d["outcome"] == "GO"
    assert d["arms"]["t1j_red"] == pytest.approx(1 / 3) and d["arms"]["t1j_black"] == pytest.approx(1 / 3)


def test_a_ONE_ARM_effect_cannot_confirm():
    red = [(o, "t1j_red", "middle") for o in OPENINGS]
    black = [(o, "t1j_black", "middle") for o in OPENINGS]
    rows = _cohort(red, pos_rate=1.0, ctl_rate=0.0, per_role=6) + \
           _cohort(black, pos_rate=1 / 3, ctl_rate=1 / 3, per_role=6)
    d = _confirm(rows)
    assert d["T"] > 0.15 and d["interval"][0] > 0
    assert d["arms"]["t1j_black"] == pytest.approx(0.0)
    assert d["outcome"] == "NO_GO"


def test_an_UNDEFINED_arm_is_NO_GO_arm_undefined_never_zero_never_skipped():
    """No common-support cell in the black arm: its T does not exist. The
    outcome must say exactly that -- not plain NO_GO (treated as zero) and not
    GO (skipped)."""
    red = [(o, "t1j_red", "middle") for o in OPENINGS]
    rows = _cohort(red, pos_rate=1.0, ctl_rate=0.0, per_role=6)
    # black arm: positions only -> no common support there
    for o in OPENINGS:
        rows += [_row(o, "t1j_black", "middle", "position", _gid(o, "t1j_black", 0), True)]
    d = _confirm(rows)
    assert d["arms"]["t1j_black"] is None
    assert d["outcome"] == "NO_GO — arm undefined"


# ───────────────── confirmation selection: filter first, select once ────────

def _cand(task_id, ply, opening, arm, phase, digest, *, frag=False, threat=False, ours=True):
    return {"task_id": task_id, "ply": ply, "opening": opening, "colour_arm": arm,
            "phase": phase, "digest": digest, "incumbent_to_move": ours,
            "mover_more_fragmented": frag, "created_threat": threat}


def test_rows_below_ply_5_are_ineligible_for_BOTH_roles_before_anything_else():
    c = ("o1_center", "t1j_red", "opening")
    cands = [_cand("g1", 3, *c, "a", frag=True), _cand("g1", 4, *c, "b", frag=False),
             _cand("g1", 5, *c, "c", frag=True), _cand("g1", 6, *c, "d", frag=False)]
    out = DP.confirmation_select(cands, seen_digests=set())
    assert {r["digest"] for r in out["rows"]} == {"c", "d"}
    assert out["removed"]["ply_lt_5"] == 2


def test_within_half_dedup_keeps_the_earliest_by_task_id_then_ply():
    c = ("o1_center", "t1j_red", "middle")
    cands = [_cand("g2", 10, *c, "same", frag=True), _cand("g1", 12, *c, "same", frag=True)]
    out = DP.confirmation_select(cands, seen_digests=set())
    assert [(r["task_id"], r["ply"]) for r in out["rows"]] == [("g1", 12)]
    assert out["removed"]["within_half_dup"] == 1


def test_cross_half_duplicates_are_REMOVED_BEFORE_the_cap_so_they_consume_no_slot():
    """§9: a cell with four candidates whose EARLIEST digest was seen in
    development: filter first, then select once -> the three later rows are
    selected. A cap-first defect would select the seen row, remove it, and keep two."""
    c = ("o1_center", "t1j_red", "middle")
    cands = [_cand("g1", 10 + i, *c, f"x{i}", frag=True) for i in range(4)]
    out = DP.confirmation_select(cands, seen_digests={"x0"})
    assert {r["digest"] for r in out["rows"]} == {"x1", "x2", "x3"}
    assert out["removed"]["seen_in_development"] == 1


def test_controls_come_from_the_selected_positions_cells_only_with_the_same_cap():
    c1, c2 = ("o1_center", "t1j_red", "middle"), ("o2_offcenter", "t1j_red", "middle")
    cands = [_cand("g1", 10 + i, *c1, f"p{i}", frag=True) for i in range(5)]
    cands += [_cand("g1", 20 + i, *c1, f"c{i}", frag=False) for i in range(5)]
    cands += [_cand("g1", 30 + i, *c2, f"o{i}", frag=False) for i in range(2)]   # no position in c2
    out = DP.confirmation_select(cands, seen_digests=set())
    frag = [r for r in out["rows"] if r["signature"] == "mover_fragmentation"]
    assert sum(r["role"] == "position" for r in frag) == 3
    assert sum(r["role"] == "control" for r in frag) == 3
    assert all(tuple(r[k] for k in ("opening", "colour_arm", "phase")) == c1 for r in frag)


def test_non_incumbent_rows_never_enter_selection():
    c = ("o1_center", "t1j_red", "middle")
    cands = [_cand("g1", 10, *c, "t", frag=True, ours=False)]
    assert DP.confirmation_select(cands, seen_digests=set())["rows"] == []


def test_the_ceiling_refuses_241_rows_and_accepts_240_before_any_seed():
    def cands(n_cells):
        out = []
        for i in range(n_cells):
            cell = (OPENINGS[i % 8], ARMS[(i // 8) % 2], PHASES[(i // 16) % 4])
            for j in range(3):
                out.append(_cand(f"g{i}", 10 + j, *cell, f"p{i}-{j}", frag=True))
                out.append(_cand(f"g{i}", 20 + j, *cell, f"c{i}-{j}", frag=False))
        return out
    ok = DP.confirmation_select(cands(40), seen_digests=set())        # 40 cells x 6 = 240
    assert ok["counts"]["total"] == 240
    with pytest.raises(DP.D1PrimeRefused, match="240"):
        DP.confirmation_select(cands(40) + [_cand("gx", 10, "o1_center", "t1j_red", "late", "extra", frag=True)],
                               seen_digests=set())


# ─────────────────────── cohort binding, type-strict ────────────────────────

def test_the_analysis_refuses_a_cohort_that_is_not_the_frozen_one_field_by_field():
    frozen = [dict(_row("o1_center", "t1j_red", "middle", "position", "g", True), ply=10)]
    ok = [dict(frozen[0])]
    DP.check_cohort(ok, frozen)                                    # identical: accepted
    for k, v in (("opening", "o2_offcenter"), ("role", "control"), ("ply", "10"),
                 ("ply", 10.0),                                  # == 10, but is not an int
                 ("ply", 11), ("signature", "created_threat"), ("digest", "other")):
        bad = [dict(frozen[0], **{k: v})]
        with pytest.raises(DP.D1PrimeError, match="frozen"):
            DP.check_cohort(bad, frozen)
    with pytest.raises(DP.D1PrimeError, match="frozen"):
        DP.check_cohort(ok + ok, frozen)                           # wrong count


def test_the_analysis_touches_no_model_no_java_and_no_seed_registry():
    import pathlib
    src = pathlib.Path(DP.__file__).read_text(encoding="utf-8")
    for forbidden in ("_default_load_evaluator", "subprocess", "t1j_adapter", "ACCOUNTED_SEED_INTERVALS",
                      "seed_is_accounted", "run_d1", "game_features", "D1_EXECUTION_AUTHORIZED"):
        assert forbidden not in src, forbidden
