"""PositionRecord.conversion round-trip (Spec 2 §5)."""
import numpy as np
from scripts.GPU.alphazero.self_play import PositionRecord


def _make_position(conversion=None):
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move="red",
        legal_moves=[(0, 8), (5, 5), (22, 4)],
        visit_counts=[10, 5, 3],
        outcome=1.0,
        active_size=24,
        ply=37,
        game_n_moves=59,
        conversion=conversion,
    )


def test_position_record_conversion_round_trip_dict():
    conv = {
        "version": 1,
        "total_goal_distance": 2,
        "largest_component_size": 12,
        "endpoint_completion_moves": [[0, 8]],
        "distance_reducing_moves":   [[22, 4]],
        "conversion_category": "two_endpoint_closeout_2ply",
        "selected_primary_class": "redundant_reinforcement",
    }
    p = _make_position(conversion=conv)
    d = p.to_dict()
    p2 = PositionRecord.from_dict(d)
    assert p2.conversion == conv


def test_position_record_conversion_defaults_to_none():
    p = _make_position(conversion=None)
    d = p.to_dict()
    p2 = PositionRecord.from_dict(d)
    assert p2.conversion is None


def test_position_record_buffer_load_with_old_no_conversion_field():
    """Pre-Spec-2 buffers: dict has no 'conversion' key. from_dict must default to None."""
    legacy_dict = {
        "board_tensor": np.zeros((24, 24, 30), dtype=np.float32).tolist(),
        "to_move": "red",
        "legal_moves": [(0, 8)],
        "visit_counts": [1],
        "outcome": 1.0,
        "active_size": 24,
        "ply": 0,
        "game_n_moves": 1,
        # no 'conversion' key
    }
    p = PositionRecord.from_dict(legacy_dict)
    assert p.conversion is None
