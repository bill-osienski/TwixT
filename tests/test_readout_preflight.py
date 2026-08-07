"""Frozen preflight gate tests.

All fixtures are SYNTHETIC. No Candidate telemetry exists or is authorized.
"""
import pytest

from scripts.GPU.alphazero.readout_preflight import (
    evaluate_gates, nonleader_selection_report, preflight_stats,
)


def _ply(ply, player, top2, overrode=False, rank=1, n_legal=None):
    if n_legal is None:
        n_legal = len(top2) if isinstance(top2, list) else 2
    return {"ply": ply, "player": player, "row": 1, "col": 1,
            "selected_visit_rank": rank, "n_legal": n_legal,
            "top2": top2, "readout_overrode_leader": overrode}


def _t2(nl=190, nc=40, ql=-0.3, qc=0.0):
    """Leader/challenger telemetry in ROOT perspective (`ql`, `qc`)."""
    return [
        {"row": 2, "col": 2, "completed_visit_count": nl,
         "q_value_child_perspective": -ql, "q_value_root_perspective": ql},
        {"row": 1, "col": 1, "completed_visit_count": nc,
         "q_value_child_perspective": -qc, "q_value_root_perspective": qc},
    ]


def _replay(game_idx, red_agent_id, moves):
    return {"schema_version": 2, "game_idx": game_idx,
            "red_agent_id": red_agent_id, "black_agent_id": "control",
            "moves": moves}


# --- population -------------------------------------------------------------


def test_population_excludes_opening_plies():
    r = _replay(0, "candidate", [
        _ply(0, "red", _t2()), _ply(19, "red", _t2()), _ply(20, "red", _t2()),
    ])
    s = preflight_stats([r], agent_id="candidate", opening_temp_plies=20)
    assert s["population_plies"] == 1


def test_population_excludes_the_other_agents_turns():
    r = _replay(0, "candidate", [
        _ply(20, "red", _t2()), _ply(21, "black", _t2()), _ply(22, "red", _t2()),
    ])
    s = preflight_stats([r], agent_id="candidate", opening_temp_plies=20)
    assert s["population_plies"] == 2


def test_ineligible_plies_stay_in_the_denominator_as_no_override():
    # Challenger below MIN_CHILD_VISITS: ineligible, but still counted.
    r = _replay(0, "candidate", [
        _ply(20, "red", _t2(nc=3)), _ply(22, "red", _t2()),
    ])
    s = preflight_stats([r], agent_id="candidate", opening_temp_plies=20)
    assert s["population_plies"] == 2
    assert s["eligible_plies"] == 1


def test_override_rate_uses_the_full_population():
    moves = [_ply(20 + 2 * i, "red", _t2(nc=3)) for i in range(10)]
    moves[0] = _ply(20, "red", _t2())          # the one that overrides
    r = _replay(0, "candidate", moves)
    s = preflight_stats([r], agent_id="candidate", opening_temp_plies=20)
    assert s["overrides"] == 1
    assert s["override_rate"] == pytest.approx(0.1)


def test_the_agent_can_play_black(_unused=None):
    r = _replay(0, "control", [_ply(20, "black", _t2())])
    r["black_agent_id"] = "candidate"
    s = preflight_stats([r], agent_id="candidate", opening_temp_plies=20)
    assert s["population_plies"] == 1
    assert s["colour_split"] == {"red": 0.0, "black": 1.0}


# --- corrupt vs undefined telemetry ----------------------------------------


def test_undefined_q_is_counted_and_reported_not_silently_dropped():
    t2 = _t2()
    t2[0]["q_value_root_perspective"] = None
    r = _replay(0, "candidate", [_ply(20, "red", t2)])
    s = preflight_stats([r], agent_id="candidate", opening_temp_plies=20)
    assert s["undefined_q_plies"] == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_q_is_corrupt_not_merely_undefined(bad):
    t2 = _t2()
    t2[0]["q_value_root_perspective"] = bad
    r = _replay(0, "candidate", [_ply(20, "red", t2)])
    s = preflight_stats([r], agent_id="candidate", opening_temp_plies=20)
    assert s["undefined_q_plies"] == 1, f"{bad!r} must be caught, not scored"


def test_a_none_mean_on_an_unvisited_child_is_not_corrupt():
    t2 = _t2()
    t2[1]["completed_visit_count"] = 0
    t2[1]["q_value_child_perspective"] = None
    t2[1]["q_value_root_perspective"] = None
    r = _replay(0, "candidate", [_ply(20, "red", t2)])
    s = preflight_stats([r], agent_id="candidate", opening_temp_plies=20)
    assert s["undefined_q_plies"] == 0
    assert s["population_plies"] == 1
    assert s["eligible_plies"] == 0     # challenger has 0 visits


# --- agent presence ---------------------------------------------------------


def test_PARTIAL_agent_absence_raises():
    """CONSTRUCTED: one replay contains the agent, one does not.

    Skipping the odd one out and scoring the rest would produce a
    clean-looking report over a population that quietly lost games -- exactly
    the fail-open the identity contract exists to prevent.
    """
    present = _replay(0, "candidate", [_ply(20, "red", _t2())])
    absent = _replay(1, "someone_else", [_ply(20, "red", _t2())])
    absent["black_agent_id"] = "another"
    with pytest.raises(ValueError, match="absent from 1 of 2"):
        preflight_stats([present, absent], agent_id="candidate",
                        opening_temp_plies=20)


def test_the_missing_game_ids_are_named_in_the_error():
    replays = [_replay(0, "candidate", [_ply(20, "red", _t2())])]
    for gid in (7, 9):
        r = _replay(gid, "someone_else", [_ply(20, "red", _t2())])
        r["black_agent_id"] = "another"
        replays.append(r)
    with pytest.raises(ValueError, match=r"\[7, 9\]"):
        preflight_stats(replays, agent_id="candidate", opening_temp_plies=20)


def test_an_agent_absent_from_every_replay_raises():
    r = _replay(0, "someone_else", [_ply(20, "red", _t2())])
    r["black_agent_id"] = "another"
    with pytest.raises(ValueError, match="absent from 1 of 1"):
        preflight_stats([r], agent_id="candidate", opening_temp_plies=20)


def test_old_schema_replays_are_rejected_not_silently_scored():
    r = _replay(0, "candidate", [_ply(20, "red", _t2())])
    r["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version"):
        preflight_stats([r], agent_id="candidate", opening_temp_plies=20)


# --- frozen gates -----------------------------------------------------------


def test_gate_closes_on_a_near_no_op():
    g = evaluate_gates({"population_plies": 1000, "overrides": 4,
                        "override_rate": 0.004, "max_single_game_share": 0.2,
                        "undefined_q_plies": 0})
    assert g["passed"] is False
    assert "override_rate_floor" in g["failed_gates"]


def test_gate_closes_when_the_rule_is_not_conservative():
    g = evaluate_gates({"population_plies": 1000, "overrides": 200,
                        "override_rate": 0.20, "max_single_game_share": 0.2,
                        "undefined_q_plies": 0})
    assert g["passed"] is False
    assert "override_rate_ceiling" in g["failed_gates"]


def test_gate_closes_on_single_game_concentration():
    g = evaluate_gates({"population_plies": 1000, "overrides": 50,
                        "override_rate": 0.05, "max_single_game_share": 0.6,
                        "undefined_q_plies": 0})
    assert g["passed"] is False
    assert "single_game_concentration" in g["failed_gates"]


def test_gate_halts_on_undefined_q():
    g = evaluate_gates({"population_plies": 1000, "overrides": 50,
                        "override_rate": 0.05, "max_single_game_share": 0.2,
                        "undefined_q_plies": 1})
    assert g["passed"] is False
    assert "undefined_q" in g["failed_gates"]


def test_gate_closes_on_an_empty_population():
    g = evaluate_gates({"population_plies": 0, "overrides": 0,
                        "override_rate": None, "max_single_game_share": None,
                        "undefined_q_plies": 0})
    assert g["passed"] is False
    assert "empty_population" in g["failed_gates"]


def test_gate_passes_in_the_frozen_band():
    g = evaluate_gates({"population_plies": 1000, "overrides": 50,
                        "override_rate": 0.05, "max_single_game_share": 0.2,
                        "undefined_q_plies": 0})
    assert g["passed"] is True
    assert g["failed_gates"] == []


def test_the_frozen_thresholds_are_reported_with_the_verdict():
    g = evaluate_gates({"population_plies": 1000, "overrides": 50,
                        "override_rate": 0.05, "max_single_game_share": 0.2,
                        "undefined_q_plies": 0})
    assert g["thresholds"] == {
        "override_rate_floor": 0.005,
        "override_rate_ceiling": 0.15,
        "single_game_share_ceiling": 0.50,
    }


def test_colour_split_is_descriptive_and_never_a_gate():
    g = evaluate_gates({"population_plies": 1000, "overrides": 50,
                        "override_rate": 0.05, "max_single_game_share": 0.2,
                        "undefined_q_plies": 0,
                        "colour_split": {"red": 1.0, "black": 0.0}})
    assert g["passed"] is True


def test_boundary_values_are_inside_the_band():
    # The gates are strict comparisons: exactly at a threshold must PASS.
    for rate in (0.005, 0.15):
        g = evaluate_gates({"population_plies": 1000, "overrides": 5,
                            "override_rate": rate,
                            "max_single_game_share": 0.50,
                            "undefined_q_plies": 0})
        assert g["passed"] is True, f"rate {rate} should be inside the band"


# --- descriptive outputs ----------------------------------------------------


def test_descriptive_outputs_are_present():
    moves = [_ply(20, "red", _t2()), _ply(80, "red", _t2()),
             _ply(120, "red", _t2(nc=3))]
    s = preflight_stats([_replay(0, "candidate", moves)],
                        agent_id="candidate", opening_temp_plies=20)
    assert set(s["override_rate_by_ply_bucket"]) >= {"20-39", "70-109", "110+"}
    for bucket in s["override_rate_by_ply_bucket"].values():
        assert {"plies", "overrides", "rate"} <= set(bucket)
    assert set(s["challenger_visits_at_override"]) == {"n", "min", "median", "max"}
    assert isinstance(s["per_game_override_counts"], dict)


def test_challenger_visit_summary_is_null_when_nothing_overrode():
    r = _replay(0, "candidate", [_ply(20, "red", _t2(nc=3))])
    s = preflight_stats([r], agent_id="candidate", opening_temp_plies=20)
    assert s["overrides"] == 0
    cv = s["challenger_visits_at_override"]
    assert cv["n"] == 0
    assert cv["min"] is None and cv["median"] is None and cv["max"] is None


def test_nonleader_selection_splits_at_the_opening_boundary():
    def ply_ranked(ply, rank):
        return _ply(ply, "red", _t2(), rank=rank)

    r = _replay(0, "candidate", [
        ply_ranked(0, 3), ply_ranked(5, 1),      # opening: 1 of 2 non-leader
        ply_ranked(20, 1), ply_ranked(25, 1),    # post: 0 of 2 non-leader
    ])
    rep = nonleader_selection_report([r], "candidate", opening_temp_plies=20)
    assert rep["opening"]["rate"] == pytest.approx(0.5)
    assert rep["post_opening"]["rate"] == pytest.approx(0.0)


def test_nonleader_report_raises_on_missing_rank():
    rec = _ply(20, "red", _t2())
    rec.pop("selected_visit_rank")
    with pytest.raises(ValueError, match="selected_visit_rank"):
        nonleader_selection_report([_replay(0, "candidate", [rec])],
                                   "candidate", opening_temp_plies=20)


def test_nonleader_rate_is_none_not_zero_for_an_empty_bucket():
    r = _replay(0, "candidate", [_ply(20, "red", _t2(), rank=1)])
    rep = nonleader_selection_report([r], "candidate", opening_temp_plies=20)
    assert rep["opening"]["plies"] == 0
    assert rep["opening"]["rate"] is None


# --- telemetry completeness: FAIL CLOSED ------------------------------------


def test_missing_top2_raises_rather_than_scoring_no_override():
    """CONSTRUCTED: schema 2 uses None for "NOT CAPTURED". Falling through as
    "no override" would lower the override rate on absent data and could close
    Candidate 2 against the floor gate."""
    rec = _ply(20, "red", None, n_legal=30)
    with pytest.raises(ValueError, match="not a list"):
        preflight_stats([_replay(0, "candidate", [rec])],
                        agent_id="candidate", opening_temp_plies=20)


def test_absent_top2_key_raises():
    rec = _ply(20, "red", _t2())
    rec.pop("top2")
    with pytest.raises(ValueError, match="top2"):
        preflight_stats([_replay(0, "candidate", [rec])],
                        agent_id="candidate", opening_temp_plies=20)


def test_empty_top2_list_raises_when_moves_were_legal():
    rec = _ply(20, "red", [], n_legal=30)
    with pytest.raises(ValueError, match="expected 2 for n_legal=30"):
        preflight_stats([_replay(0, "candidate", [rec])],
                        agent_id="candidate", opening_temp_plies=20)


def test_a_short_top2_raises_when_two_moves_were_legal():
    rec = _ply(20, "red", [_t2()[0]], n_legal=30)
    with pytest.raises(ValueError, match="expected 2 for n_legal=30"):
        preflight_stats([_replay(0, "candidate", [rec])],
                        agent_id="candidate", opening_temp_plies=20)


def test_the_error_names_the_game_and_ply():
    rec = _ply(37, "red", None, n_legal=30)
    with pytest.raises(ValueError, match="game 4 ply 37"):
        preflight_stats([_replay(4, "candidate", [rec])],
                        agent_id="candidate", opening_temp_plies=20)


def test_missing_n_legal_raises():
    rec = _ply(20, "red", _t2())
    rec.pop("n_legal")
    with pytest.raises(ValueError, match="n_legal"):
        preflight_stats([_replay(0, "candidate", [rec])],
                        agent_id="candidate", opening_temp_plies=20)


def test_ONE_legal_move_is_legitimately_ineligible_and_stays_counted():
    """The rule needs a challenger, so a forced move cannot override -- but it
    is real play and must remain in the denominator, not vanish."""
    forced = _ply(20, "red", [_t2()[0]], n_legal=1)
    normal = _ply(22, "red", _t2())
    s = preflight_stats([_replay(0, "candidate", [forced, normal])],
                        agent_id="candidate", opening_temp_plies=20)
    assert s["population_plies"] == 2
    assert s["eligible_plies"] == 1
    assert s["overrides"] == 1
    assert s["override_rate"] == pytest.approx(0.5)
