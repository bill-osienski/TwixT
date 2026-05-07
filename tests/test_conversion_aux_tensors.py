# tests/test_conversion_aux_tensors.py
"""Batch-level conversion aux tensor tests (Spec 2 §6.2)."""
import numpy as np
from scripts.GPU.alphazero.conversion_loss import make_conversion_aux_tensors


def _pos(conversion=None):
    """Lightweight stand-in for PositionRecord — only fields used by
    make_conversion_aux_tensors are required."""
    class _P:
        pass
    p = _P()
    p.conversion = conversion
    return p


def test_aux_tensor_shape_matches_target_pi():
    positions = [_pos(conversion=None) for _ in range(3)]
    legal_padded = [[(0, 0), None, None]] * 3   # M_padded = 3
    aux_target, aux_mask = make_conversion_aux_tensors(
        positions, legal_padded, max_moves_cap=3,
    )
    assert aux_target.shape == (3, 3)
    assert aux_mask.shape == (3,)


def test_aux_mask_zero_for_ineligible_positions():
    positions = [_pos(conversion=None), _pos(conversion=None)]
    legal_padded = [[(0, 0)], [(1, 1)]]
    aux_target, aux_mask = make_conversion_aux_tensors(
        positions, legal_padded, max_moves_cap=4,
    )
    assert aux_mask.tolist() == [0.0, 0.0]
    assert aux_target.sum() == 0.0


def test_aux_mask_zero_when_target_returns_none():
    """conversion present but no completion/reducer move appears in legal_padded."""
    conv = {
        "endpoint_completion_moves": [[99, 99]],   # not in legal
        "distance_reducing_moves":   [[88, 88]],   # not in legal
    }
    positions = [_pos(conversion=conv)]
    legal_padded = [[(0, 0), (5, 5)]]
    aux_target, aux_mask = make_conversion_aux_tensors(
        positions, legal_padded, max_moves_cap=4,
    )
    assert aux_mask.tolist() == [0.0]
    assert aux_target[0].sum() == 0.0


def test_aux_tensor_skips_padding_columns():
    """legal_padded entries equal to None must not contribute to weights
    (Spec 2 §6.2 + §3 lock #2)."""
    conv = {
        "endpoint_completion_moves": [[0, 0]],
        "distance_reducing_moves":   [],
    }
    positions = [_pos(conversion=conv)]
    legal_padded = [[(0, 0), None, None, None]]   # only first slot is real
    aux_target, aux_mask = make_conversion_aux_tensors(
        positions, legal_padded, max_moves_cap=4,
    )
    assert aux_mask.tolist() == [1.0]
    np.testing.assert_allclose(aux_target[0], [1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_aux_tensor_aligns_with_target_pi_columns():
    """Per-position aux_target column j references the same legal_padded[i][j]
    that target_pi[i][j] would reference."""
    conv = {
        "endpoint_completion_moves": [[5, 6]],
        "distance_reducing_moves":   [[1, 2]],
    }
    positions = [_pos(conversion=conv)]
    legal_padded = [[(1, 2), (3, 4), (5, 6), None]]   # match the spec anchor fixture
    aux_target, aux_mask = make_conversion_aux_tensors(
        positions, legal_padded, max_moves_cap=4,
        completion_weight=1.0, reducer_weight=0.35,
    )
    assert aux_mask.tolist() == [1.0]
    np.testing.assert_allclose(
        aux_target[0],
        [0.35 / 1.35, 0.0, 1.0 / 1.35, 0.0],
        atol=1e-6,
    )
