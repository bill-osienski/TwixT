"""v13 gradient-conflict projection: a pure surface-pytree projection helper and
an eval-mode calibration-only component-loss helper."""
import numpy as np
import mlx.core as mx
import pytest

from scripts.GPU.alphazero.trainer import (
    project_conflicting_gradient, _calibration_component_loss,
    freeze_batchnorm_running_stats)
from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord
from scripts.GPU.alphazero.calibration_pool import target_in_to_move


def _surf(vh, blk):
    return {"value_head": {"w": mx.array(vh, dtype=mx.float32)},
            "block": {"w": mx.array(blk, dtype=mx.float32)}}


def test_projection_conflict_subtracts():
    # g_A and g_G anti-parallel on value_head -> dot<0 -> correct g_total.
    surf_total = _surf([1.0, 0.0], [0.0])
    surf_A = _surf([1.0, 0.0], [0.0])
    surf_G = _surf([-1.0, 0.0], [0.0])
    out, telem = project_conflicting_gradient(surf_total, surf_A, surf_G, weight=0.01)
    assert telem["conflict"] is True and telem["skip_reason"] is None
    assert telem["c"] == pytest.approx(-1.0, abs=1e-6)         # dot/(normsq+eps)
    # g_final = total - weight*c*G = [1,0] - 0.01*(-1)*[-1,0] = [0.99, 0]
    assert float(out["value_head"]["w"][0].item()) == pytest.approx(0.99, abs=1e-5)
    assert telem["removed_norm"] == pytest.approx(0.01, abs=1e-6)   # |w*c|*norm_G


def test_projection_no_conflict_unchanged_and_telemetry():
    # dot>=0 -> no correction, g_final IS g_total, telemetry counts non-conflict.
    surf_total = _surf([1.0, 2.0], [3.0])
    surf_A = _surf([1.0, 0.0], [0.0])
    surf_G = _surf([1.0, 0.0], [0.0])                          # dot = +1
    out, telem = project_conflicting_gradient(surf_total, surf_A, surf_G, weight=0.01)
    assert telem["evaluated"] is True and telem["conflict"] is False
    assert telem["skip_reason"] is None                       # a genuine no-conflict
    assert telem["c"] == 0.0 and telem["removed_norm"] == 0.0
    assert out is surf_total                                   # unchanged object


def test_projection_tiny_guardrail_skipped():
    # anti-parallel but ||g_G|| below eps -> skip (tiny_guardrail), unchanged.
    surf_total = _surf([1.0], [0.0])
    surf_A = _surf([1.0], [0.0])
    surf_G = _surf([-1e-12], [0.0])                            # dot<0 but norm_G<eps
    out, telem = project_conflicting_gradient(surf_total, surf_A, surf_G, weight=0.01)
    assert telem["conflict"] is False and telem["skip_reason"] == "tiny_guardrail"
    assert telem["c"] == 0.0 and out is surf_total


def _row(to_move, target_black):
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move=to_move, legal_moves=[(0, 0), (1, 1)], visit_counts=[0, 0],
        outcome=target_in_to_move(to_move, target_black), active_size=24,
        ply=20, game_n_moves=None)


def test_component_loss_masks_by_sign():
    # a_correction ignores guardrail rows; guardrail_hinge ignores A rows.
    # Frozen/eval BN makes the per-row value batch-independent, so the mixed-batch
    # component equals the solo-batch component exactly.
    net = create_network(hidden=64, n_blocks=2)
    freeze_batchnorm_running_stats(net)
    A = _row("black", -0.35)                                   # sign 0 (hard_value)
    G = _row("red", -0.9)                                      # sign +... guardrail
    sign_ag = np.array([0.0, 1.0], dtype=np.float32)
    la_mixed = _calibration_component_loss(net, [A, G], None, sign_ag, 0.10, "a_correction")
    la_solo = _calibration_component_loss(net, [A], None, np.array([0.0], np.float32), 0.10, "a_correction")
    assert float(la_mixed.item()) == pytest.approx(float(la_solo.item()), abs=1e-5)
    lg_mixed = _calibration_component_loss(net, [A, G], None, sign_ag, 0.10, "guardrail_hinge")
    lg_solo = _calibration_component_loss(net, [G], None, np.array([1.0], np.float32), 0.10, "guardrail_hinge")
    assert float(lg_mixed.item()) == pytest.approx(float(lg_solo.item()), abs=1e-5)


def test_component_loss_rejects_unknown_component():
    net = create_network(hidden=64, n_blocks=2)
    freeze_batchnorm_running_stats(net)
    with pytest.raises(ValueError, match="component"):
        _calibration_component_loss(net, [_row("black", -0.35)], None,
                                    np.array([0.0], np.float32), 0.10, "bogus")
