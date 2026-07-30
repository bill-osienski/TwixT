"""v18 clip formula -- spec Sec 3.1 / 4.2. Pure; no MCTS, no GPU."""
import pytest

from scripts.GPU.alphazero.v18_provisional_backup import (
    CAP_GRID,
    IDENTITY_CAP,
    ProvisionalBackup,
    provisional_depth2_backup_value,
)

# The observed A pattern, spec Sec 1.1. The depth-1 node is RED to move, so its
# stored parent-to-move value is +0.087 even though its black value is -0.087.
A_PARENT_TO_MOVE = 0.087
A_LEAF_TO_MOVE = 0.793
A_RESIDUAL = 0.880
A_BACKUP_AT_050 = 0.413


def test_no_clip_returns_the_original_leaf_object_value():
    out = provisional_depth2_backup_value(0.20, A_PARENT_TO_MOVE, 1.25)
    assert out.backup_value == 0.20          # exact, not baseline+residual
    assert out.clipped_amount == 0.0
    assert out.clip_direction == 0


def test_no_clip_does_not_reconstruct_arithmetically():
    leaf, parent = 0.1 + 0.2, 0.3
    out = provisional_depth2_backup_value(leaf, parent, 2.0 - 1e-9)
    assert repr(out.backup_value) == repr(leaf)


def test_positive_clip_matches_the_observed_a_pattern():
    out = provisional_depth2_backup_value(A_LEAF_TO_MOVE, A_PARENT_TO_MOVE, 0.50)
    assert out.residual == pytest.approx(A_RESIDUAL, abs=1e-12)
    assert out.clip_direction == 1
    assert out.backup_value == pytest.approx(A_BACKUP_AT_050, abs=1e-12)
    assert out.clipped_amount == pytest.approx(A_RESIDUAL - 0.50, abs=1e-12)


def test_the_observed_a_residual_does_not_bind_at_the_weakest_cap():
    # 0.880 < 1.25. This is why the ladder expects the weakest cap to under-reach
    # rather than to be unsafe -- spec Sec 7 routes that to "advance", not "reject".
    out = provisional_depth2_backup_value(A_LEAF_TO_MOVE, A_PARENT_TO_MOVE, 1.25)
    assert out.clip_direction == 0


def test_negative_clip_is_symmetric_in_magnitude():
    out = provisional_depth2_backup_value(-A_LEAF_TO_MOVE, -A_PARENT_TO_MOVE, 0.50)
    assert out.residual == pytest.approx(-A_RESIDUAL, abs=1e-12)
    assert out.clip_direction == -1
    assert out.backup_value == pytest.approx(-A_BACKUP_AT_050, abs=1e-12)
    assert out.clipped_amount == pytest.approx(A_RESIDUAL - 0.50, abs=1e-12)


def test_boundary_is_strict_so_abs_residual_equal_cap_does_not_clip():
    # Spec Sec 9.2.1 identity witnesses (max|residual| <= 0.50 vs strongest cap
    # 0.50) depend on this boundary being exact.
    out = provisional_depth2_backup_value(0.50, 0.0, 0.50)
    assert out.residual == 0.50
    assert out.clip_direction == 0
    assert out.backup_value == 0.50


def test_perspective_reversal_symmetry_over_the_whole_grid():
    for cap in CAP_GRID:
        for leaf, parent in [(0.9, -0.1), (-0.4, 0.7), (0.05, 0.05), (1.0, -1.0)]:
            a = provisional_depth2_backup_value(leaf, parent, cap)
            b = provisional_depth2_backup_value(-leaf, -parent, cap)
            assert b.backup_value == pytest.approx(-a.backup_value, abs=1e-15)
            assert b.residual == pytest.approx(-a.residual, abs=1e-15)
            assert b.clipped_amount == pytest.approx(a.clipped_amount, abs=1e-15)
            assert b.clip_direction == -a.clip_direction


def test_identity_cap_cannot_bind_on_any_tanh_pair():
    for leaf in (-1.0, -0.3, 0.0, 0.3, 1.0):
        for parent in (-1.0, -0.3, 0.0, 0.3, 1.0):
            out = provisional_depth2_backup_value(leaf, parent, IDENTITY_CAP)
            assert out.clip_direction == 0
            assert out.backup_value == leaf


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_inputs_raise_rather_than_bypass(bad):
    with pytest.raises(ValueError):
        provisional_depth2_backup_value(bad, 0.0, 1.0)
    with pytest.raises(ValueError):
        provisional_depth2_backup_value(0.0, bad, 1.0)


@pytest.mark.parametrize("bad", [0.0, -0.5, 2.0001, float("nan")])
def test_out_of_range_cap_rejected(bad):
    with pytest.raises(ValueError):
        provisional_depth2_backup_value(0.5, 0.0, bad)


@pytest.mark.parametrize("bad", [True, False])
def test_boolean_cap_rejected(bad):
    with pytest.raises(ValueError):
        provisional_depth2_backup_value(0.5, 0.0, bad)


def test_result_is_immutable():
    out = provisional_depth2_backup_value(0.9, -0.1, 0.5)
    assert isinstance(out, ProvisionalBackup)
    with pytest.raises(Exception):
        out.backup_value = 0.0
