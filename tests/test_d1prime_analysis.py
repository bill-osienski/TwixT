"""D1' analysis -- the FROZEN plan (docs/superpowers/2026-09-07-t1j-d1prime-diagnostic-proposal.md)
implemented against SYNTHETIC cohorts and MOCKED readouts only.

Nothing here reads a real D1 record, a real confirmation game, a model or a JVM.
The only randomness is the plan's analysis-only PRNG (PCG64, seed 20260907) on
synthetic fixtures. Every §9 control of the plan has a test below whose failure
mode is the named defect.
"""
import inspect
import itertools
import re
from collections import Counter

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
         ply=10, digest=None, agree=False, rep=None):
    """A cohort row. `rep` is derived from the game id (`...-r<n>`) as the
    canonical selection rows carry it, so the design binding has something to
    bind; rows built with arbitrary game names default to 0."""
    if rep is None:
        m = re.search(r"-r(\d+)$", str(task_id))
        rep = int(m.group(1)) if m else 0
    return {"task_id": task_id, "opening": opening, "colour_arm": arm, "phase": phase,
            "signature": sig, "role": role, "ply": ply, "rep": rep,
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


def _incumbent_record(*, policy, our_move, visits=None, overrode=False):
    """The incumbent readout AS `d1_probe.IncumbentReadout` WRITES IT: the real
    `eval_replay.ply_record` (synthetic values, no model) plus the same
    `record.update(...)` d1_probe applies. Inventing the keys here is how a
    field-name mismatch hides -- so the writer is real and only its inputs are
    synthetic."""
    from scripts.GPU.alphazero import eval_replay
    counts = {_move(k): int(v) for k, v in (visits or {k: 1 for k in policy}).items()}
    rec = eval_replay.ply_record(10, "red", tuple(our_move), counts, 0.1,
                                 top2=None, overrode_leader=overrode)
    rec.update({"seed": 1, "streams": {}, "raw_policy": dict(policy),
                "root_visits": {f"{r},{c}": v for (r, c), v in sorted(counts.items())},
                "selected_policy_rank": 1, "selected_policy_mass": 0.1})
    return rec


def _move(key):
    if isinstance(key, str):
        r, c = key.split(",")
        return (int(r), int(c))
    return (int(key[0]), int(key[1]))


def _d1_position(*, policy, our_move, t1j_move_6, visits=None, overrode=False, **labels):
    """A D1 per-position record in the shape `d1_probe` writes."""
    rec = _incumbent_record(policy=policy, our_move=our_move, visits=visits, overrode=overrode)
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


def test_the_override_flag_is_read_from_the_REAL_writers_field_name():
    """🔴 [P2] Production writes `readout_overrode_leader` (eval_replay.ply_record);
    revision 1 read `overrode_leader` and turned a production True into None. The
    fixture is built by the real writer, so the name cannot drift unnoticed."""
    from scripts.GPU.alphazero import eval_replay
    written = eval_replay.ply_record(10, "red", (0, 0), {(0, 0): 5, (0, 1): 3}, 0.1,
                                     overrode_leader=True)
    assert "readout_overrode_leader" in written and "overrode_leader" not in written
    pol = _policy_with_ranks()
    row = DP.position_row(_d1_position(policy=pol, our_move=(0, 0), t1j_move_6=(0, 6),
                                       overrode=True))
    assert row["overrode_leader"] is True
    row2 = DP.position_row(_d1_position(policy=pol, our_move=(0, 0), t1j_move_6=(0, 6),
                                        overrode=False))
    assert row2["overrode_leader"] is False


def test_a_record_without_the_override_field_is_REFUSED_not_silently_None():
    pol = _policy_with_ranks()
    pos = _d1_position(policy=pol, our_move=(0, 0), t1j_move_6=(0, 6))
    del pos["incumbent"]["readout_overrode_leader"]
    with pytest.raises(DP.D1PrimeError, match="readout_overrode_leader"):
        DP.position_row(pos)


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
    return DP._development_decision(rows, _strata(), B=100, **kw)


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
    d = DP._development_decision(rows, strata, B=400)
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
    d = DP._development_decision(rows, strata, B=B)
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
    return DP._confirmation_decision(rows, _strata(), B=100, **kw)


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


def test_an_undefined_arm_is_named_EVEN_WHEN_support_is_insufficient():
    """🔴 [P2] The generic insufficient-support result was returned BEFORE the arm
    check, so an undefined arm could receive the wrong named outcome. The frozen
    plan requires `NO_GO — arm undefined` for that condition; it is checked first."""
    red = [(o, "t1j_red", "middle") for o in OPENINGS[:4]]        # 4 cells: below the floor
    rows = _cohort(red, pos_rate=1.0, ctl_rate=0.0, per_role=3)
    for o in OPENINGS[:4]:                                        # black: positions only
        rows += [_row(o, "t1j_black", "middle", "position", _gid(o, "t1j_black", 0), True)]
    st = DP.matched_statistic(rows, cohort="mover_fragmentation")
    assert not DP.floors_met(st)[0], "the fixture must be below the floor"
    assert DP.matched_statistic(rows, cohort="mover_fragmentation", arm="t1j_black")["T"] is None
    d = _confirm(rows)
    assert d["outcome"] == "NO_GO — arm undefined", d["outcome"]
    assert d["arms"]["t1j_black"] is None


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
    """A confirmation candidate in the PRODUCTION schema: the mover is identified
    by `system` ("ours" / "t1j"), as `d1_selection` records it via `D0.moved_by`."""
    return {"task_id": task_id, "ply": ply, "opening": opening, "colour_arm": arm,
            "phase": phase, "digest": digest, "system": "ours" if ours else "t1j",
            "mover_more_fragmented": frag, "created_threat": threat}


def test_the_selector_uses_the_PRODUCTION_system_field_and_refuses_a_row_without_it():
    """🔴 [P1] Revision 1 read `incumbent_to_move` and DEFAULTED a missing flag to
    True, so a production row with `system="t1j"` was selected. Eligibility is
    read from `system`, and a row carrying neither is REFUSED -- never accepted
    by default."""
    c = ("o1_center", "t1j_red", "middle")
    ours = _cand("g1", 10, *c, "a", frag=True)
    theirs = _cand("g1", 11, *c, "b", frag=True, ours=False)
    assert ours["system"] == "ours" and theirs["system"] == "t1j"
    out = DP.confirmation_select([ours, theirs], seen_digests=set())
    assert {r["digest"] for r in out["rows"]} == {"a"}
    assert out["removed"]["not_incumbent_to_move"] == 1
    no_field = {k: v for k, v in ours.items() if k != "system"}
    with pytest.raises(DP.D1PrimeError, match="system"):
        DP.confirmation_select([no_field], seen_digests=set())
    with pytest.raises(DP.D1PrimeError, match="system"):
        DP.confirmation_select([dict(ours, system="incumbent")], seen_digests=set())


def test_the_selector_matches_D0s_ONE_DEFINITION_of_which_engine_moved():
    """`system` is produced by `d0_postmortem.moved_by`; the selector must agree
    with it rather than restate the cut."""
    from scripts.GPU.alphazero import d0_postmortem as D0
    assert D0.moved_by("t1j_red", "red") == "t1j" and D0.moved_by("t1j_red", "black") == "ours"
    c = ("o1_center", "t1j_red", "middle")
    rows = [dict(_cand("g1", 10 + i, *c, f"d{i}", frag=True),
                 system=D0.moved_by("t1j_red", mover))
            for i, mover in enumerate(("red", "black"))]
    out = DP.confirmation_select(rows, seen_digests=set())
    assert {r["digest"] for r in out["rows"]} == {"d1"}


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


# ────────────── the CHECKED entry point: binding before calculation ─────────

MANIFEST_FIELDS = DP.COHORT_BINDING_FIELDS + ("rep",)      # as select_all's rows carry them


def _manifest(rows):
    return [{k: r[k] for k in MANIFEST_FIELDS} for r in rows]


def _frozen_and_rows(pos_rate=2 / 3, ctl_rate=1 / 3, per_role=6):
    rows = _cohort(ALL_16, pos_rate=pos_rate, ctl_rate=ctl_rate, per_role=per_role)
    return rows, _manifest(rows)


def _checked(rows, frozen, tasks, kernel=None, reps=(0, 1), B=50):
    return DP._analyse(rows, frozen_cohort=frozen, tasks=tasks, reps=reps,
                       kernel=kernel or DP._development_decision, B=B, seed=DP.BOOTSTRAP_SEED)


def test_the_binding_layer_binds_the_cohort_BEFORE_any_calculation():
    """🔴 [P1] `_development_decision` computes from whatever rows it is handed:
    a cohort with every phase forged produced GO while `check_cohort` refused the
    same rows. `_analyse` binds first, so a forged label cannot reach the
    statistic -- and the production entry resolves what it binds against."""
    rows, frozen = _frozen_and_rows()
    assert _checked(rows, frozen, _design())["outcome"] == "GO"
    forged = [dict(r, phase="late") for r in rows]
    assert DP._development_decision(forged, _strata(), B=50)["outcome"] == "GO"   # the kernel does not bind
    with pytest.raises(DP.D1PrimeError, match="frozen"):
        _checked(forged, frozen, _design())


def test_the_binding_layer_FIXES_the_primary_hypothesis():
    """The secondary cohort cannot be made primary: neither entry takes a cohort
    parameter, and a `created_threat`-only cohort is reported as secondary."""
    for fn in (DP.analyse_development, DP.analyse_confirmation):
        assert "cohort" not in inspect.signature(fn).parameters
    rows = _cohort(ALL_16, pos_rate=1.0, ctl_rate=0.0, per_role=6, sig="created_threat")
    out = _checked(rows, _manifest(rows), _design())
    assert out["cohort"] == DP.PRIMARY_COHORT
    assert out["outcome"] == "NO_GO — insufficient support"
    assert out["secondary"]["T"] == pytest.approx(1.0)


def test_the_binding_layer_builds_strata_from_the_DESIGN_not_from_the_rows():
    """A game that contributed no selected row still exists in its stratum and
    must be drawable; strata inferred from rows would silently drop it."""
    rows, frozen = _frozen_and_rows()
    thin = [r for r in rows if not r["task_id"].endswith("-r1")]
    thin_frozen = _manifest(thin)
    out = _checked(thin, thin_frozen, _design())
    assert out["strata"] == 16
    assert {g for s in DP.design_strata(_design(), reps=(0, 1)) for g in s["games"]} > \
        {r["task_id"] for r in thin}


def test_the_CONFIRMATION_kernel_binds_and_keeps_the_arm_rules():
    rows = _cohort(ALL_16, pos_rate=2 / 3, ctl_rate=1 / 3, per_role=6, reps=(2, 3))
    frozen = _manifest(rows)
    out = _checked(rows, frozen, _design(reps=(2, 3)), DP._confirmation_decision, reps=(2, 3))
    assert out["outcome"] == "GO" and set(out["arms"]) == set(DP.ARMS)
    assert out["bound"]["reps"] == [2, 3]
    with pytest.raises(DP.D1PrimeError, match="frozen"):
        _checked([dict(r, role="control") for r in rows], frozen, _design(reps=(2, 3)),
                 DP._confirmation_decision, reps=(2, 3))


def test_a_cohort_whose_GAMES_are_not_in_the_designs_strata_is_REFUSED():
    """Found by a fixture of mine that passed the wrong half's design: every
    replicate would be empty and the run would report "bootstrap undefined" --
    an instrument mismatch wearing a result's name."""
    rows, frozen = _frozen_and_rows()                       # games from reps (0, 1)
    with pytest.raises(DP.D1PrimeError, match="not in the design"):
        _checked(rows, frozen, _design(reps=(2, 3)), DP._confirmation_decision, reps=(2, 3))


def test_the_confirmation_entry_takes_a_MANIFEST_PATH_and_no_numeric_knobs(tmp_path, canonical):
    """🔑 There is no canonical confirmation cohort to resolve: it is produced at
    §6.2 step 8, a gated step that has not run. So the cohort must be a WRITTEN
    ARTIFACT a review can pin -- not a list a caller assembles -- and its sha256
    is recorded. No B, seed or reps parameter exists."""
    import hashlib, json as _json
    params = list(inspect.signature(DP.analyse_confirmation).parameters)
    assert params == ["d1_report", "cohort_manifest_path"], params
    # confirmation rows must live in the HOLDOUT half's games (reps 2 and 3)
    conf_tasks = [t for t in canonical["tasks"] if int(t["rep"]) in (2, 3)]
    by_stratum = {}
    for t in conf_tasks:
        by_stratum.setdefault((t["opening"], t["colour_arm"]), []).append(t)
    rows = []
    for (o, a), ts in by_stratum.items():
        for role in ("position", "control"):
            for i in range(6):
                task = ts[i % 2]
                rows.append(_row(o, a, "middle", role, task["task_id"],
                                 i < (6 if role == "position" else 2)))
    manifest = tmp_path / "cohort.json"
    manifest.write_bytes(_json.dumps({"rows": _manifest(rows)}).encode())
    rep = _report_for(rows, lprd_positions=1.0, lprd_controls=0.0)
    out = DP.analyse_confirmation(rep, cohort_manifest_path=str(manifest))
    assert out["resolved"]["cohort_manifest_sha256"] == \
        hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert out["prng"]["B"] == DP.B_REPLICATES
    forged = tmp_path / "forged.json"
    forged.write_bytes(_json.dumps({"rows": [dict(m, phase="late") for m in _manifest(rows)]}).encode())
    with pytest.raises(DP.D1PrimeError, match="frozen"):
        DP.analyse_confirmation(rep, cohort_manifest_path=str(forged))


def test_the_entries_report_what_they_bound():
    rows, frozen = _frozen_and_rows()
    out = _checked(rows, frozen, _design())
    assert out["bound"] == {"rows": len(rows), "cohort": DP.PRIMARY_COHORT,
                            "reps": [0, 1], "strata": 16}
    assert out["claim"].startswith("T describes matched LPRD in the frozen selected cohort")


# ───────── the PRODUCTION entry: resolves the canonical cohort itself ───────

@pytest.fixture(scope="module")
def canonical():
    """The canonical §13 cohort and design, resolved the way production must."""
    from scripts.GPU.alphazero import d0_postmortem as D0, d1_selection as SEL, l0_match_plan as L0P
    bound = D0.bind_record(DP.L0_RECORD_REL, DP.L0_PLAN_REL)
    rows = SEL.select_all(bound)["positions"]
    return {"rows": rows, "tasks": L0P.load_l0_plan(DP.L0_PLAN_REL)["tasks"]}


def _report_for(rows, *, lprd_positions, lprd_controls):
    """A D1 report of MOCKED readouts carrying real cohort labels: the T1j move
    ranks 7th (lprd) or 1st (not), per role, at the requested rates."""
    pol = _policy_with_ranks()
    out = []
    per_role = {}
    for r in rows:
        i = per_role.get(r["role"], 0)
        per_role[r["role"]] = i + 1
        rate = lprd_positions if r["role"] == "position" else lprd_controls
        lprd = (i % 100) < round(rate * 100)
        labels = {k: r[k] for k in DP.COHORT_BINDING_FIELDS}
        out.append(_d1_position(policy=pol, our_move=(0, 0),
                                t1j_move_6=(0, 6) if lprd else (0, 0), **labels))
    return {"positions": out}


def test_the_production_entry_EXPOSES_NO_KNOBS_at_all():
    """🔴 [P1] A caller could pass forged rows WITH a matching forged
    frozen_cohort, and override B, the PRNG seed and the repetitions -- a GO from
    a changed protocol wearing a frozen one's name. The production entry takes
    the D1 report and nothing else."""
    params = list(inspect.signature(DP.analyse_development).parameters)
    assert params == ["d1_report"], params
    for banned in ("rows", "frozen_cohort", "tasks", "reps", "B", "seed", "cohort"):
        assert banned not in params, banned


def test_the_production_entry_RESOLVES_the_canonical_cohort_and_design_itself(canonical):
    """It does not restate the selection: the cohort it binds against is exactly
    `d1_selection.select_all` over the digest-bound L0 record, and the design is
    the frozen L0 plan's tasks."""
    rep = _report_for(canonical["rows"], lprd_positions=1.0, lprd_controls=0.0)
    out = DP.analyse_development(rep)
    assert out["bound"] == {"rows": 221, "cohort": DP.PRIMARY_COHORT,
                            "reps": [0, 1], "strata": 16}
    assert out["resolved"]["cohort_source"] == "d1_selection.select_all"
    assert out["resolved"]["n_positions"] == 221
    assert out["resolved"]["record"] == DP.L0_RECORD_REL
    assert out["resolved"]["plan"] == DP.L0_PLAN_REL
    assert out["prng"] == {"bit_generator": "PCG64", "seed": DP.BOOTSTRAP_SEED,
                           "B": DP.B_REPLICATES}


def test_the_production_entry_REFUSES_a_report_that_is_not_the_canonical_cohort(canonical):
    """Forged rows can no longer come with a matching forged manifest: there is
    no manifest parameter, and the resolved cohort refuses them."""
    rep = _report_for(canonical["rows"], lprd_positions=1.0, lprd_controls=0.0)
    forged = {"positions": [dict(p, phase="late") for p in rep["positions"]]}
    with pytest.raises(DP.D1PrimeError, match="frozen"):
        DP.analyse_development(forged)
    short = {"positions": rep["positions"][:-1]}
    with pytest.raises(DP.D1PrimeError, match="frozen"):
        DP.analyse_development(short)


def test_the_production_entry_uses_the_FROZEN_B_and_seed(monkeypatch, canonical):
    seen = {}
    real = DP.stability_interval

    def spy(rows, strata, **kw):
        seen.update(kw)
        return real(rows, strata, **kw)

    monkeypatch.setattr(DP, "stability_interval", spy)
    DP.analyse_development(_report_for(canonical["rows"], lprd_positions=1.0, lprd_controls=0.0))
    assert seen["B"] == DP.B_REPLICATES == 10_000 and seen["seed"] == DP.BOOTSTRAP_SEED


# ───────────── design binding: task metadata, not just membership ───────────

def test_a_design_whose_task_METADATA_disagrees_with_the_rows_is_REFUSED():
    """🔴 [P1] Two complete opening strata swapped in the design, counts still
    valid: membership-only checking accepted it and returned GO. Each row's task
    id is bound to the design's opening, colour arm and repetition."""
    rows, frozen = _frozen_and_rows()
    design = _design()
    swapped = []
    for t in design:
        o = t["opening"]
        o = "o2_offcenter" if o == "o1_center" else ("o1_center" if o == "o2_offcenter" else o)
        swapped.append(dict(t, opening=o))
    assert sorted(t["task_id"] for t in swapped) == sorted(t["task_id"] for t in design)
    assert Counter(t["opening"] for t in swapped) == Counter(t["opening"] for t in design)
    with pytest.raises(DP.D1PrimeError, match="disagrees with the design"):
        DP._analyse(rows, frozen_cohort=frozen, tasks=swapped, reps=(0, 1),
                    kernel=DP._development_decision, B=20, seed=1)


def test_a_design_that_RELABELS_reps_within_a_stratum_is_REFUSED():
    """Strata stay well formed (2 games each) and every task id is present, so
    only the metadata binding can catch it: each game's `rep` is swapped with the
    other game's in its stratum."""
    rows, frozen = _frozen_and_rows()
    design = [dict(t, rep=1 - int(t["rep"])) for t in _design()]
    with pytest.raises(DP.D1PrimeError, match="disagrees with the design"):
        _checked(rows, frozen, design)


@pytest.mark.parametrize("value", ["0", 0.0, True])
def test_the_design_binding_is_TYPE_STRICT_on_rep(value):
    """`"0"`, `0.0` and `True` all compare equal to a rep of 0 under `==`; the
    binding requires the same TYPE, so none of them passes as the design's rep."""
    rows, frozen = _frozen_and_rows()
    design = [dict(t, rep=value) if int(t["rep"]) == 0 else t for t in _design()]
    with pytest.raises(DP.D1PrimeError, match="disagrees with the design"):
        _checked(rows, frozen, design)


def test_a_row_that_does_not_CARRY_a_bound_design_field_is_REFUSED():
    """A guard that skips a field the row omits is a guard the row escapes by
    omission -- found when synthetic rows without `rep` bound nothing."""
    rows, frozen = _frozen_and_rows()
    stripped = [{k: v for k, v in m.items() if k != "rep"} for m in frozen]
    with pytest.raises(DP.D1PrimeError, match="does not carry 'rep'"):
        _checked(rows, stripped, _design())


def test_a_single_task_relabelled_is_caught_by_the_STRATA_construction_first():
    """Recorded so the division of labour is explicit: moving ONE task to another
    opening leaves a stratum with 1 game and another with 3, which
    `design_strata` refuses before the metadata check is reached. Both refusals
    are correct; this pins which one fires."""
    rows, frozen = _frozen_and_rows()
    design = [dict(t, opening="o3_low") if t["task_id"] == rows[0]["task_id"] else t
              for t in _design()]
    with pytest.raises(DP.D1PrimeError, match="exactly 2"):
        _checked(rows, frozen, design)


def test_rows_carry_rep_and_it_is_bound_too(canonical):
    assert all("rep" in r for r in canonical["rows"])
    by_id = {t["task_id"]: t for t in canonical["tasks"]}
    assert all(r["rep"] == by_id[r["task_id"]]["rep"] for r in canonical["rows"])


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
