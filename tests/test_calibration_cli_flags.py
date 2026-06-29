import pytest

from scripts.GPU.alphazero.train import (
    build_arg_parser, parse_calibration_tag_schedule,
)


def test_calibration_flag_defaults():
    args = build_arg_parser().parse_args([])
    assert args.post_opening_calibration_enabled is False
    assert args.post_opening_calibration_manifest is None
    assert args.post_opening_calibration_target == -0.50
    assert args.post_opening_calibration_weight == 0.02
    assert args.post_opening_calibration_batch_fraction == 0.10
    assert args.post_opening_calibration_tag_schedule is None


def test_calibration_flags_set():
    args = build_arg_parser().parse_args([
        "--post-opening-calibration-enabled",
        "--post-opening-calibration-manifest", "train.csv",
        "--post-opening-calibration-weight", "0.05",
        "--post-opening-calibration-target", "-0.35",
        "--post-opening-calibration-batch-fraction", "0.15",
    ])
    assert args.post_opening_calibration_enabled is True
    assert args.post_opening_calibration_manifest == "train.csv"
    assert args.post_opening_calibration_weight == 0.05
    assert args.post_opening_calibration_target == -0.35
    assert args.post_opening_calibration_batch_fraction == 0.15


def test_calibration_tag_schedule_flag_parsed_raw():
    args = build_arg_parser().parse_args([
        "--post-opening-calibration-tag-schedule",
        "black_predrop_correction=2,goal_line_retention=1",
    ])
    assert (args.post_opening_calibration_tag_schedule
            == "black_predrop_correction=2,goal_line_retention=1")


def test_parse_calibration_tag_schedule_none():
    assert parse_calibration_tag_schedule(None) is None
    assert parse_calibration_tag_schedule("") is None


def test_parse_calibration_tag_schedule_valid_ordered():
    out = parse_calibration_tag_schedule(
        "black_predrop_correction=2,goal_line_retention=1,"
        "old_post_opening_retention=2,red_predrop_retention=1")
    assert out == {"black_predrop_correction": 2, "goal_line_retention": 1,
                   "old_post_opening_retention": 2, "red_predrop_retention": 1}
    assert list(out) == ["black_predrop_correction", "goal_line_retention",
                         "old_post_opening_retention", "red_predrop_retention"]


def test_parse_calibration_tag_schedule_missing_equals_raises():
    with pytest.raises(ValueError):
        parse_calibration_tag_schedule("black_predrop_correction")


def test_parse_calibration_tag_schedule_negative_raises():
    with pytest.raises(ValueError):
        parse_calibration_tag_schedule("black_predrop_correction=-1")


def test_parse_calibration_tag_schedule_duplicate_raises():
    with pytest.raises(ValueError):
        parse_calibration_tag_schedule(
            "black_predrop_correction=2,black_predrop_correction=1")


def test_parse_calibration_tag_schedule_zero_total_raises():
    with pytest.raises(ValueError):
        parse_calibration_tag_schedule("black_predrop_correction=0")


def test_calibration_teacher_weight_flag_defaults():
    args = build_arg_parser().parse_args([])
    assert args.post_opening_calibration_teacher_value_weight == 1.0
    assert args.post_opening_calibration_teacher_policy_kl_weight == 0.25


def test_calibration_teacher_weight_flags_set():
    args = build_arg_parser().parse_args([
        "--post-opening-calibration-teacher-value-weight", "0.5",
        "--post-opening-calibration-teacher-policy-kl-weight", "0.0",
    ])
    assert args.post_opening_calibration_teacher_value_weight == 0.5
    assert args.post_opening_calibration_teacher_policy_kl_weight == 0.0
