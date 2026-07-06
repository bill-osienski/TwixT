"""Wiring pins for the guardrail path inside the 4000-line train loop
(precedent: tests/test_train_value_head_and_final_block.py)."""
from scripts.GPU.alphazero import trainer as trainer_mod
from scripts.GPU.alphazero import train as train_mod


def test_train_loop_selects_guardrail_split_and_forwards_sign():
    src = open(trainer_mod.__file__).read()
    assert "split_samples_with_guardrail" in src
    assert "GUARDRAIL_LOSS_MODE" in src
    assert "calibration_guardrail_sign=_calib_guard_sign," in src
    assert "guardrail_margin=post_opening_guardrail_margin," in src
    # telemetry accumulation + JSON
    assert "sum_guardrail_hinge_loss" in src
    assert '"guardrail_hinge_loss"' in src
    assert '"guardrail_active_frac"' in src
    assert '"guardrail_margin"' in src


def test_cli_guardrail_margin_flag_and_plumb():
    src = open(train_mod.__file__).read()
    assert '"--guardrail-margin"' in src
    assert "post_opening_guardrail_margin=args.guardrail_margin," in src
