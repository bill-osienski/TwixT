"""Single source of truth for training-surface labels.

training_surface_label() feeds BOTH the startup banner and the projection-scope
telemetry, so a new surface can never leave one of them stale (the recurring
v13-era hardcoded-label bug: --help, docstring, calib_projection_scope, missing
v14 banner). This behavioral test is the forcing function — a new surface with no
label branch here fails a test instead of shipping a wrong string.
"""
import pytest

from scripts.GPU.alphazero.trainer import training_surface_label


@pytest.mark.parametrize("flags,expected", [
    (dict(train_value_head_and_value_adapter=True), "value_head_and_value_adapter"),
    (dict(train_value_head_and_final_block=True), "value_head_and_final_block"),
    (dict(train_value_head_only=True), "value_head_only"),
    (dict(), "all_trainable"),                       # full training, no restriction
])
def test_training_surface_label(flags, expected):
    assert training_surface_label(**flags) == expected


def test_labels_are_distinct():
    labels = {
        training_surface_label(train_value_head_and_value_adapter=True),
        training_surface_label(train_value_head_and_final_block=True),
        training_surface_label(train_value_head_only=True),
        training_surface_label(),
    }
    assert len(labels) == 4                          # every surface has its own name


def test_startup_banner_and_projection_status_wired_to_the_helper():
    # The banner + the projection strength/scope line both live in train() and are
    # awkward to drive without a full training loop; pin that they exist and derive
    # from the shared label so future edits don't silently drop them.
    from scripts.GPU.alphazero import trainer as trainer_mod
    src = open(trainer_mod.__file__).read()
    assert "TRAIN SURFACE:" in src                                    # unified surface banner
    assert "Gradient projection: ON (strength=" in src               # prints strength...
    assert "scope={_surface}" in src                                 # ...and scope, from the helper
    assert src.count("training_surface_label(") >= 3                 # def + banner + proj_scope
