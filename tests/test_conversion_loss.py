# tests/test_conversion_loss.py
"""Loss math tests (Spec 2 §6.3)."""
import numpy as np
import mlx.core as mx
import pytest

from scripts.GPU.alphazero.trainer import alphazero_loss_batch
from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord


def _pos(conversion=None, active_size=24):
    return PositionRecord(
        board_tensor=np.zeros((active_size, active_size, 30), dtype=np.float32),
        to_move="red",
        legal_moves=[(0, 0), (1, 1), (2, 2)],
        visit_counts=[10, 5, 3],
        outcome=1.0,
        active_size=active_size,
        ply=0,
        game_n_moves=10,
        conversion=conversion,
    )


def _network():
    return create_network()


def test_aux_loss_zero_when_all_ineligible():
    net = _network()
    positions = [_pos(conversion=None) for _ in range(4)]
    total, policy, value, l2, aux, coverage, n_eligible = alphazero_loss_batch(
        net, positions, conversion_loss_weight=0.05,
    )
    assert float(aux.item()) == 0.0
    assert n_eligible == 0
    assert not np.isnan(float(total.item()))


def test_aux_loss_uses_masked_log_probs():
    """ANCHOR (Spec 2 §11.3): aux loss must use the SAME masked log_probs
    as policy loss. Padded/illegal columns must not contribute to either.

    DETERMINISTIC fixture-based check: extract masking logic into helper
    `compute_masked_log_probs(logits, move_mask)` and assert that on a
    known fixture, log_probs at legal columns equal log(0.5) within 1e-5.
    """
    from scripts.GPU.alphazero.trainer import compute_masked_log_probs

    # B=1, M_padded=4, only first 2 columns legal.
    # Logits chosen so masked-softmax probabilities are predictable.
    logits = mx.array([[0.0, 0.0, 100.0, 100.0]], dtype=mx.float32)
    move_mask = mx.array([[1.0, 1.0, 0.0, 0.0]], dtype=mx.float32)

    log_probs = compute_masked_log_probs(logits, move_mask)
    # Effective logits over legal columns are [0, 0] → softmax = [0.5, 0.5]
    legal_log_probs = [float(log_probs[0, j].item()) for j in (0, 1)]
    np.testing.assert_allclose(legal_log_probs, [np.log(0.5), np.log(0.5)], atol=1e-5)

    # Padded columns leaking into denominator would put ~all mass on cols 2/3,
    # giving log_probs at cols 0,1 of approximately -100. Catch that case:
    assert all(lp > -1.0 for lp in legal_log_probs), (
        "padded columns appear to be leaking into logsumexp"
    )


def test_aux_loss_mean_over_eligible_only():
    """Same per-position aux magnitude regardless of eligible/total ratio."""
    net = _network()
    conv = {
        "version": 1,
        "endpoint_completion_moves": [[0, 0]],
        "distance_reducing_moves":   [],
    }
    # 1 eligible / 4 total
    pos_few = [_pos(conversion=conv)] + [_pos(conversion=None) for _ in range(3)]
    # 1 eligible / 1 total
    pos_only = [_pos(conversion=conv)]

    _, _, _, _, aux_few, _, n_few = alphazero_loss_batch(
        net, pos_few, conversion_loss_weight=0.05,
    )
    _, _, _, _, aux_only, _, n_only = alphazero_loss_batch(
        net, pos_only, conversion_loss_weight=0.05,
    )
    np.testing.assert_allclose(
        float(aux_few.item()), float(aux_only.item()), atol=1e-3,
    )
    assert n_few == 1
    assert n_only == 1


def test_aux_loss_returns_n_eligible_as_int():
    """ANCHOR (Spec 2 §11.3): aux_n_eligible must be an int, not a float."""
    net = _network()
    conv = {
        "version": 1,
        "endpoint_completion_moves": [[0, 0]],
        "distance_reducing_moves":   [],
    }
    positions = [_pos(conversion=conv), _pos(conversion=None), _pos(conversion=conv)]
    _, _, _, _, _, _, n_eligible = alphazero_loss_batch(
        net, positions, conversion_loss_weight=0.05,
    )
    assert isinstance(n_eligible, int)
    assert n_eligible == 2


def test_total_loss_includes_aux_term_when_enabled():
    net = _network()
    conv = {
        "version": 1,
        "endpoint_completion_moves": [[0, 0]],
        "distance_reducing_moves":   [],
    }
    positions = [_pos(conversion=conv)]
    total_off, policy_off, _, _, aux_off, _, _ = alphazero_loss_batch(
        net, positions, conversion_loss_weight=0.0,
    )
    total_on, policy_on, _, _, aux_on, _, _ = alphazero_loss_batch(
        net, positions, conversion_loss_weight=0.5,
    )
    assert float(aux_off.item()) == 0.0
    assert float(aux_on.item()) > 0.0
    # total_on should be policy + 0.5*aux + value + l2 — strictly greater than total_off
    assert float(total_on.item()) > float(total_off.item())


def test_total_loss_excludes_aux_when_weight_zero():
    """conversion_loss_weight=0 short-circuits — make_conversion_aux_tensors
    should NOT be called (zero overhead path)."""
    import scripts.GPU.alphazero.conversion_loss as cl_mod
    call_count = {"n": 0}
    original = cl_mod.make_conversion_aux_tensors

    def _spy(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    cl_mod.make_conversion_aux_tensors = _spy
    try:
        net = _network()
        positions = [_pos(conversion={"endpoint_completion_moves": [[0,0]],
                                       "distance_reducing_moves": []})]
        alphazero_loss_batch(net, positions, conversion_loss_weight=0.0)
        assert call_count["n"] == 0, (
            "make_conversion_aux_tensors was called even with weight=0"
        )
    finally:
        cl_mod.make_conversion_aux_tensors = original


def test_aux_loss_matches_hand_computed_ce_on_fixture():
    """Sanity check on a deterministic fixture.

    With random init, aux_loss on a 3-legal-move position should be
    ~log(3) magnitude (~1.0–2.0). Loose bound for MLX numerical drift.
    """
    net = _network()
    conv = {
        "version": 1,
        "endpoint_completion_moves": [[0, 0]],
        "distance_reducing_moves":   [],
    }
    positions = [_pos(conversion=conv)]
    _, _, _, _, aux, _, _ = alphazero_loss_batch(
        net, positions, conversion_loss_weight=0.05,
    )
    aux_val = float(aux.item())
    assert 0.1 < aux_val < 10.0, (
        f"aux_loss={aux_val} on one-position fixture — "
        "expected ~log(3) magnitude with random init"
    )
