"""make_padded_batch must not truncate a 24x24 position (CPU, no network): the opening has 528 legal moves, and the
ordinary training chain's default cap was 512, which dropped the last 16 moves and renormalized what was left."""
import inspect

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

from scripts.GPU.alphazero import trainer as T  # noqa: E402
from scripts.GPU.alphazero.game import TwixtState  # noqa: E402
from scripts.GPU.alphazero.self_play import PositionRecord  # noqa: E402


def _record(moves, counts):
    return PositionRecord(board_tensor=np.zeros((24, 24, 1), np.float32), to_move="red",
                          legal_moves=list(moves), visit_counts=list(counts), outcome=0.0, active_size=24)


def _arrays(batch):
    _b, rows, cols, mask, pi, _z = batch
    return np.asarray(rows), np.asarray(cols), np.asarray(mask), np.asarray(pi)


def test_the_24x24_opening_keeps_every_legal_move_and_its_target_mass():
    moves = TwixtState(active_size=24, to_move="red", max_plies_limit=280).legal_moves()
    assert len(moves) == 528
    counts = [0] * 528
    counts[0], counts[527] = 300, 100
    rows, cols, mask, pi = _arrays(T.make_padded_batch([_record(moves, counts)]))
    assert pi.shape == (1, 528) and mask.sum() == 528
    assert list(zip(rows[0].tolist(), cols[0].tolist())) == [tuple(m) for m in moves]
    expected = np.zeros(528, np.float32)
    expected[0], expected[527] = 0.75, 0.25
    np.testing.assert_allclose(pi[0], expected, rtol=0, atol=1e-7)


def test_a_small_batch_matches_hand_computed_targets():
    a = _record([(1, 2), (3, 4), (5, 6)], [1, 0, 3])
    b = _record([(7, 8), (9, 10)], [2, 2])
    rows, cols, mask, pi = _arrays(T.make_padded_batch([a, b]))
    np.testing.assert_array_equal(rows, [[1, 3, 5], [7, 9, 0]])
    np.testing.assert_array_equal(cols, [[2, 4, 6], [8, 10, 0]])
    np.testing.assert_array_equal(mask, [[1, 1, 1], [1, 1, 0]])
    np.testing.assert_allclose(pi, [[0.25, 0.0, 0.75], [0.5, 0.5, 0.0]], rtol=0, atol=1e-7)


def test_an_explicit_cap_is_still_honoured():
    moves = TwixtState(active_size=24, to_move="red", max_plies_limit=280).legal_moves()
    counts = [0] * 528
    counts[0], counts[527] = 300, 100
    _r, _c, mask, pi = _arrays(T.make_padded_batch([_record(moves, counts)], max_moves_cap=512))
    assert pi.shape == (1, 512) and mask.sum() == 512 and pi[0, 0] == 1.0


@pytest.mark.parametrize("fn", ["make_padded_batch", "_calibration_component_loss", "summarize_value_sanity",
                                "alphazero_loss_batch", "train_step"])
def test_the_ordinary_chain_defaults_to_the_full_board(fn):
    assert inspect.signature(getattr(T, fn)).parameters["max_moves_cap"].default == 576
