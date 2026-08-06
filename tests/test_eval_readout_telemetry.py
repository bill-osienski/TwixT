"""Telemetry contract tests: perspective, undefined values, schema version."""
import pytest

from scripts.GPU.alphazero import eval_readout as R
from scripts.GPU.alphazero.eval_replay import REPLAY_SCHEMA_VERSION, ply_record


def _top2():
    return [
        R.ChildStat((2, 2), 190, 0.30, -0.30),
        R.ChildStat((1, 1), 40, -0.05, 0.05),
    ]


def test_schema_version_is_bumped_for_top2():
    assert REPLAY_SCHEMA_VERSION == 2


def test_ply_record_without_top2_keeps_the_field_null_not_empty():
    rec = ply_record(0, "red", (2, 2), {(2, 2): 5, (1, 1): 3}, 0.1)
    assert rec["top2"] is None
    assert rec["readout_overrode_leader"] is False


def test_ply_record_emits_both_perspectives():
    rec = ply_record(21, "red", (2, 2), {(2, 2): 190, (1, 1): 40}, 0.1,
                     top2=_top2())
    a, b = rec["top2"]
    assert a["completed_visit_count"] == 190
    assert a["q_value_child_perspective"] == pytest.approx(0.30)
    assert a["q_value_root_perspective"] == pytest.approx(-0.30)
    assert b["q_value_root_perspective"] == pytest.approx(0.05)


def test_ply_record_top2_carries_the_move_coordinates():
    rec = ply_record(21, "red", (2, 2), {(2, 2): 190, (1, 1): 40}, 0.1,
                     top2=_top2())
    assert (rec["top2"][0]["row"], rec["top2"][0]["col"]) == (2, 2)
    assert (rec["top2"][1]["row"], rec["top2"][1]["col"]) == (1, 1)


def test_ply_record_preserves_undefined_q_as_null():
    top2 = [R.ChildStat((2, 2), 190, 0.3, -0.3), R.ChildStat((1, 1), 0, None, None)]
    rec = ply_record(21, "red", (2, 2), {(2, 2): 190, (1, 1): 0}, 0.1, top2=top2)
    assert rec["top2"][1]["q_value_child_perspective"] is None
    assert rec["top2"][1]["q_value_root_perspective"] is None


def test_ply_record_records_the_override_flag():
    rec = ply_record(21, "red", (1, 1), {(2, 2): 190, (1, 1): 40}, 0.1,
                     top2=_top2(), overrode_leader=True)
    assert rec["readout_overrode_leader"] is True


def test_ply_record_still_fails_loud_on_a_move_outside_the_counts():
    with pytest.raises(ValueError):
        ply_record(0, "red", (9, 9), {(2, 2): 5}, 0.1)


def test_ply_record_still_fails_loud_on_empty_counts():
    with pytest.raises(ValueError):
        ply_record(0, "red", (2, 2), {}, 0.1)


def test_legacy_fields_are_unchanged():
    # B2 is additive: nothing that already existed may shift.
    rec = ply_record(7, "black", (1, 1), {(2, 2): 190, (1, 1): 40}, -0.25)
    assert rec["ply"] == 7
    assert rec["player"] == "black"
    assert (rec["row"], rec["col"]) == (1, 1)
    assert rec["root_value"] == pytest.approx(-0.25)
    assert rec["root_top1_share"] == pytest.approx(190 / 230)
    assert rec["selected_visit_rank"] == 2
    assert rec["selected_visit_count"] == 40
    assert rec["root_total_visits"] == 230
    assert rec["n_legal"] == 2
