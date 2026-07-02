"""The train-loop schema gate must route continuation pools down the masked
teacher-mode path. Source-level check (the gate is inline in a 4000-line
function; mirrors test_builder_module_defers_heavy_imports's source-inspection
style) plus a set-membership check."""
import re

from scripts.GPU.alphazero import trainer as trainer_mod
from scripts.GPU.alphazero.calibration_pool import (
    CONTINUATION_LOSS_MODE, TEACHER_MODE_LOSS_MODES)


def test_continuation_mode_is_teacher_mode():
    assert CONTINUATION_LOSS_MODE in TEACHER_MODE_LOSS_MODES


def test_trainer_gates_on_teacher_mode_loss_modes():
    src = open(trainer_mod.__file__).read()
    assert len(re.findall(r"schema in TEACHER_MODE_LOSS_MODES", src)) == 2, (
        "both schema gates (setup print + train-step mask split) must use "
        "TEACHER_MODE_LOSS_MODES")
    assert "schema in RETENTION_POLICY_LOSS_MODES" not in src
