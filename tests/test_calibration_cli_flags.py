from scripts.GPU.alphazero.train import build_arg_parser


def test_calibration_flag_defaults():
    args = build_arg_parser().parse_args([])
    assert args.post_opening_calibration_enabled is False
    assert args.post_opening_calibration_manifest is None
    assert args.post_opening_calibration_target == -0.50
    assert args.post_opening_calibration_weight == 0.02
    assert args.post_opening_calibration_batch_fraction == 0.10


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
