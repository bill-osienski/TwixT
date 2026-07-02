import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from scripts.GPU.alphazero.trainer import (
    alphazero_loss_batch, train_step, MainModule, flatten_params,
    CALIB_VALUE_TERM_IDX,
)
from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord


def _main_pos():
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move="red", legal_moves=[(0, 0), (1, 1), (2, 2)],
        visit_counts=[10, 5, 3], outcome=1.0, active_size=24,
        ply=0, game_n_moves=10,
    )


def _calib_pos(target=-0.5):
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move="black", legal_moves=[(0, 0), (1, 1)],
        visit_counts=[0, 0], outcome=target, active_size=24,
        ply=20, game_n_moves=None,
    )


def test_disabled_returns_seven_tuple():
    net = create_network(hidden=64, n_blocks=2)
    out = alphazero_loss_batch(net, [_main_pos() for _ in range(3)])
    assert len(out) == 7


def test_zero_weight_is_inert_seven_tuple():
    net = create_network(hidden=64, n_blocks=2)
    pos = [_main_pos() for _ in range(3)]
    out = alphazero_loss_batch(
        net, pos,
        calibration_positions=[_calib_pos()],
        calibration_loss_weight=0.0,
    )
    # Zero weight means calibration is inactive: no extended tuple, no leaked term.
    assert len(out) == 7


def test_enabled_returns_ten_tuple_and_adds_weighted_mse():
    net = create_network(hidden=64, n_blocks=2)
    pos = [_main_pos() for _ in range(3)]
    calib = [_calib_pos(-0.5), _calib_pos(-0.5)]
    value_weight = 0.5
    conversion_weight = 0.0
    calibration_weight = 0.02
    out = alphazero_loss_batch(
        net,
        pos,
        value_weight=value_weight,
        conversion_loss_weight=conversion_weight,
        calibration_positions=calib,
        calibration_loss_weight=calibration_weight,
    )
    assert len(out) == 10
    (
        total,
        policy_loss,
        value_loss,
        l2_loss,
        aux_loss,
        _aux_coverage,
        _aux_n_eligible,
        calib_loss,
        _calib_value_mean,
        calib_n,
    ) = out
    assert calib_n == 2
    expected_total = (
        policy_loss
        + value_weight * value_loss
        + l2_loss
        + conversion_weight * aux_loss
        + calibration_weight * calib_loss
    )
    np.testing.assert_allclose(
        float(total.item()),
        float(expected_total.item()),
        atol=1e-6,
    )


def _vh_gnorm(grads):
    return sum(float(mx.sum(mx.abs(p)).item())
               for _, p in flatten_params(grads["value_head"]))


def test_calibration_gradient_reaches_value_head():
    net = create_network(hidden=64, n_blocks=2)
    pos = [_main_pos() for _ in range(3)]
    calib = [_calib_pos(-0.5)]

    def off(m):
        return alphazero_loss_batch(m, pos, value_weight=0.0, l2_weight=0.0)

    def on(m):
        return alphazero_loss_batch(m, pos, value_weight=0.0, l2_weight=0.0,
                                    calibration_positions=calib,
                                    calibration_loss_weight=0.02)

    _, g_off = nn_value_and_grad(net, off)
    _, g_on = nn_value_and_grad(net, on)
    # Essential claim: calibration drives a value-head gradient the disabled
    # path does not. Assert the margin (robust to MLX init/path noise) rather
    # than relying solely on exact-zero for the disabled case.
    assert _vh_gnorm(g_on) > 1e-6
    assert _vh_gnorm(g_on) > _vh_gnorm(g_off)


def nn_value_and_grad(net, fn):
    import mlx.nn as nn
    return nn.value_and_grad(net, fn)(net)


def test_weighted_calibration_loss_matches_formula():
    """calib_loss == Σ(wᵢ·mseᵢ)/Σ(wᵢ). Recover per-sample mse with one-hot weights,
    then verify the [1,3] weighting (network is deterministic across forward-only calls)."""
    net = create_network(hidden=64, n_blocks=2)
    pos = [_main_pos() for _ in range(3)]
    calib = [_calib_pos(-0.5), _calib_pos(0.25)]  # distinct targets → distinct mse

    def calib_loss(weights):
        out = alphazero_loss_batch(
            net, pos, calibration_positions=calib,
            calibration_weights=np.array(weights, dtype=np.float32),
            calibration_loss_weight=0.02)
        return float(out[7].item())  # index 7 = calib_loss

    mse0 = calib_loss([1.0, 0.0])
    mse1 = calib_loss([0.0, 1.0])
    expected = (1.0 * mse0 + 3.0 * mse1) / (1.0 + 3.0)
    np.testing.assert_allclose(calib_loss([1.0, 3.0]), expected, rtol=1e-5)


def test_equal_weights_equal_unweighted_mean():
    net = create_network(hidden=64, n_blocks=2)
    pos = [_main_pos() for _ in range(3)]
    calib = [_calib_pos(-0.5), _calib_pos(0.25)]

    def calib_loss(weights):
        out = alphazero_loss_batch(
            net, pos, calibration_positions=calib,
            calibration_weights=(None if weights is None
                                 else np.array(weights, dtype=np.float32)),
            calibration_loss_weight=0.02)
        return float(out[7].item())

    np.testing.assert_allclose(calib_loss([2.0, 2.0]), calib_loss(None), rtol=1e-6)


def test_all_zero_calibration_weights_are_finite_zero_loss():
    """Σw==0 → mx.maximum(Σw, 1e-8) keeps it finite (0.0), never NaN.
    Explicit zero-weight rows are now allowed, so pin this behavior."""
    net = create_network(hidden=64, n_blocks=2)
    pos = [_main_pos() for _ in range(3)]
    calib = [_calib_pos(-0.5), _calib_pos(0.25)]
    out = alphazero_loss_batch(
        net, pos, calibration_positions=calib,
        calibration_weights=np.array([0.0, 0.0], dtype=np.float32),
        calibration_loss_weight=0.02)
    calib_loss = float(out[7].item())
    assert np.isfinite(calib_loss)
    assert calib_loss == 0.0


def test_train_step_arity_disabled_and_enabled():
    net = create_network(hidden=64, n_blocks=2)
    mm = MainModule(net.encoder, net.policy_head)
    opt_main = optim.Adam(learning_rate=1e-3)
    opt_value = optim.Adam(learning_rate=1e-4)
    pos = [_main_pos() for _ in range(3)]

    off = train_step(network=net, main_module=mm, opt_main=opt_main,
                     opt_value=opt_value, batch=pos)
    assert len(off) == 7

    on = train_step(network=net, main_module=mm, opt_main=opt_main,
                    opt_value=opt_value, batch=pos,
                    calibration_positions=[_calib_pos()],
                    calibration_loss_weight=0.02)
    assert len(on) == 10
    assert all(np.isfinite(x) for x in on[:9])
    assert on[9] == 1


def test_train_step_accepts_weights_returns_ten_tuple():
    net = create_network(hidden=64, n_blocks=2)
    mm = MainModule(net.encoder, net.policy_head)
    opt_main = optim.Adam(learning_rate=1e-3)
    opt_value = optim.Adam(learning_rate=1e-4)
    pos = [_main_pos() for _ in range(3)]
    out = train_step(network=net, main_module=mm, opt_main=opt_main, opt_value=opt_value,
                     batch=pos, calibration_positions=[_calib_pos(-0.5), _calib_pos(0.25)],
                     calibration_weights=np.array([1.0, 3.0], dtype=np.float32),
                     calibration_loss_weight=0.02)
    assert len(out) == 10
    assert all(np.isfinite(x) for x in out[:9])
    assert out[9] == 2


def _teacher_calib_pos(value=0.2):
    # 2 legal moves, uniform teacher policy in visit_counts.
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move="black", legal_moves=[(0, 0), (1, 1)],
        visit_counts=[0.5, 0.5], outcome=value, active_size=24,
        ply=20, game_n_moves=None,
    )


def test_mask_none_stays_ten_tuple_regression():
    net = create_network(hidden=64, n_blocks=2)
    out = alphazero_loss_batch(
        net, [_main_pos() for _ in range(3)],
        calibration_positions=[_calib_pos(-0.5)],
        calibration_loss_weight=0.02,
        calibration_teacher_policy_mask=None,     # v2/v3 path
    )
    assert len(out) == 10                          # unchanged shape


def test_teacher_mode_returns_fourteen_tuple():
    net = create_network(hidden=64, n_blocks=2)
    calib = [_calib_pos(-0.5), _teacher_calib_pos(0.2)]   # 1 correction, 1 retention
    out = alphazero_loss_batch(
        net, [_main_pos() for _ in range(3)],
        calibration_positions=calib,
        calibration_weights=np.array([1.0, 1.0], dtype=np.float32),
        calibration_loss_weight=0.01,
        calibration_teacher_policy_mask=np.array([0.0, 1.0], dtype=np.float32),
        teacher_value_weight=1.0, teacher_policy_kl_weight=0.25,
    )
    assert len(out) == 14
    (_, _, _, _, _, _, _, _, _, _,
     value_term, policy_ce, policy_kl_est, n_ret) = out
    assert n_ret == 1                              # one retention row
    assert float(policy_ce) >= float(policy_kl_est) - 1e-5   # CE >= CE - H  (H >= 0)
    assert float(policy_kl_est) >= -1e-4           # KL is non-negative


def test_policy_ce_zero_when_no_retention_rows():
    net = create_network(hidden=64, n_blocks=2)
    out = alphazero_loss_batch(
        net, [_main_pos() for _ in range(3)],
        calibration_positions=[_calib_pos(-0.5)],
        calibration_weights=np.array([1.0], dtype=np.float32),
        calibration_loss_weight=0.01,
        calibration_teacher_policy_mask=np.array([0.0], dtype=np.float32),  # no retention
        teacher_value_weight=1.0, teacher_policy_kl_weight=0.25,
    )
    assert len(out) == 14
    *_head, value_term, policy_ce, policy_kl_est, n_ret = out   # named, not magic indices
    assert abs(float(policy_ce)) < 1e-6            # policy_ce == 0 (guarded denominator, no retention rows)


def test_train_step_teacher_mode_returns_fourteen_floats():
    net = create_network(hidden=64, n_blocks=2)
    main = MainModule(net.encoder, net.policy_head)
    opt_main = optim.Adam(learning_rate=1e-3)
    opt_value = optim.Adam(learning_rate=1e-3)
    out = train_step(
        network=net, main_module=main, opt_main=opt_main, opt_value=opt_value,
        batch=[_main_pos() for _ in range(3)],
        calibration_positions=[_calib_pos(-0.5), _teacher_calib_pos(0.2)],
        calibration_weights=np.array([1.0, 1.0], dtype=np.float32),
        calibration_loss_weight=0.01,
        calibration_teacher_policy_mask=np.array([0.0, 1.0], dtype=np.float32),
        teacher_value_weight=1.0, teacher_policy_kl_weight=0.25,
    )
    assert len(out) == 14
    assert all(np.isfinite(x) for x in out)


def test_make_padded_batch_correction_vs_retention_target_pi():
    # spec §10: bridge between parsing and loss — correction rows produce a
    # zero target_pi, retention rows a normalized one, padded/masked columns no mass.
    from scripts.GPU.alphazero.trainer import make_padded_batch
    corr = _calib_pos(-0.5)                        # 2 legal moves, visit_counts [0, 0]
    ret3 = PositionRecord(                          # 3 legal moves, uniform teacher policy
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move="black", legal_moves=[(0, 0), (1, 1), (2, 2)],
        visit_counts=[1 / 3, 1 / 3, 1 / 3], outcome=0.2, active_size=24,
        ply=20, game_n_moves=None)
    _, _, _, mask, target_pi, _ = make_padded_batch([corr, ret3])
    tp = np.array(target_pi.tolist())
    msk = np.array(mask.tolist())
    assert tp.shape[1] == msk.shape[1] == 3        # target_pi width == padded legal dim
    assert np.allclose(tp[0], 0.0)                 # correction row: all-zero target
    np.testing.assert_allclose(tp[1].sum(), 1.0, atol=1e-6)  # retention row sums to 1
    assert tp[0, 2] == 0.0 and msk[0, 2] == 0.0    # corr's padded slot: masked, no mass


def _freeze_bn(net):
    """momentum=0 → running stats never update (frozen at base); train-mode
    normalization still uses batch stats. Lets eval-mode forwards read fixed
    base running stats with no drift, isolating the calibration eval-toggle."""
    for _, m in net.named_modules():
        if isinstance(m, nn.BatchNorm):
            m.momentum = 0.0


def _distinct_teacher_pos(board_fill, moves):
    n = len(moves)
    return PositionRecord(
        board_tensor=np.full((24, 24, 30), board_fill, dtype=np.float32),
        to_move="black", legal_moves=list(moves),
        visit_counts=[1.0 / n] * n, outcome=0.0, active_size=24,
        ply=20, game_n_moves=None)


def _eval_value(net, rec):
    net.eval()
    n = len(rec.legal_moves)
    rr = np.zeros((1, n), np.int32); cc = np.zeros((1, n), np.int32); mm = np.ones((1, n), np.float32)
    for j, (r, c) in enumerate(rec.legal_moves):
        rr[0, j], cc[0, j] = r, c
    _, v, _ = net.forward_padded(mx.array(rec.board_tensor[None]),
                                 mx.array(rr), mx.array(cc), mx.array(mm), active_size=24)
    return float(np.array(v)[0])


def test_teacher_calib_forward_uses_eval_running_stats():
    """The network uses BatchNorm, so a TRAIN-mode batched calibration forward
    normalizes by per-batch statistics (batch-dependent). The teacher-path
    calibration forward must instead run in EVAL mode (running stats) so its
    value reproduces the per-position eval-cached teacher target — even with the
    net in TRAIN mode and a multi-board batch. (BN frozen here so eval reads
    fixed base stats; the ignored main-batch forward cannot perturb them.)"""
    net = create_network(hidden=64, n_blocks=2)
    _freeze_bn(net)
    a = _distinct_teacher_pos(0.0, [(0, 0), (1, 1)])
    b = _distinct_teacher_pos(1.0, [(0, 0), (1, 1), (2, 2)])   # different board → batch stats differ
    a.outcome = _eval_value(net, a)                            # eval-mode (running-stats) teacher value
    b.outcome = _eval_value(net, b)
    net.train()                                               # net in TRAIN mode (like training)
    out = alphazero_loss_batch(
        net, [_main_pos() for _ in range(3)],
        calibration_positions=[a, b],
        calibration_weights=np.array([1.0, 1.0], dtype=np.float32),
        calibration_loss_weight=1.0,
        calibration_teacher_policy_mask=np.array([1.0, 1.0], dtype=np.float32),
        teacher_value_weight=1.0, teacher_policy_kl_weight=0.0)
    assert float(out[CALIB_VALUE_TERM_IDX]) < 1e-5


def test_freeze_batchnorm_running_stats():
    """freeze_batchnorm_running_stats sets momentum=0 on every BatchNorm so the
    running stats stay frozen at their loaded (base) values; a train-mode forward
    no longer moves them. (Train-mode normalization still uses batch stats.)"""
    from scripts.GPU.alphazero.trainer import freeze_batchnorm_running_stats
    net = create_network(hidden=64, n_blocks=2)
    n_frozen = freeze_batchnorm_running_stats(net)
    assert n_frozen >= 1
    assert all(m.momentum == 0.0 for _, m in net.named_modules()
               if isinstance(m, nn.BatchNorm))
    bn = net.encoder.blocks[0].bn1
    rm0 = np.array(bn.running_mean).copy()
    net.train()
    # Non-zero input so activations (hence batch stats) are non-zero regardless of
    # conv-bias init — an UNfrozen BN would move running_mean here; momentum=0 must not.
    b = np.ones((5, 24, 24, 30), np.float32)
    rr = np.zeros((5, 3), np.int32); cc = np.zeros((5, 3), np.int32); mm = np.ones((5, 3), np.float32)
    net.forward_padded(mx.array(b), mx.array(rr), mx.array(cc), mx.array(mm), active_size=24)
    assert np.allclose(rm0, np.array(bn.running_mean))


def test_root_retention_flows_masked_policy_ce_path(tmp_path):
    """mcts_root_retention samples must produce mask=1.0 and drive the 14-tuple
    masked policy-CE path with a finite, positive CE (root visits != raw priors
    for a fresh network, so CE > 0 is expected — NOT ~0 like v4 self-distillation)."""
    import json as _json
    import math as _math
    from scripts.GPU.alphazero.calibration_pool import (
        build_calibration_sample, legal_moves_sha1, split_samples_with_modes)
    from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
    from tests.goal_line_probe_fixtures import legal_replay

    replay = legal_replay(9, game_idx=1)
    rp = tmp_path / "game_000001.json"
    rp.write_text(_json.dumps(replay))
    state = position_state(replay, 5, "black")
    legal = state.legal_moves()
    n = len(legal)
    # A sharp (non-uniform) root-visit target so CE > H(target) is comfortably > 0.
    visits = [0.0] * n
    visits[0] = 0.9
    if n > 1:
        visits[1] = 0.1
    case = {"game_idx": "1", "case_id": "root1", "replay_path": str(rp),
            "position_ply": "5", "side_to_move": "black",
            "tag": "old_post_opening_retention", "weight_scale": "1.0",
            "loss_mode": "mcts_root_retention", "teacher_value": "0.2",
            "root_visits_json": _json.dumps(visits),
            "root_legal_moves_sha1": legal_moves_sha1(legal)}
    sample = build_calibration_sample(case, calibration_target=-0.35)
    records, weights, mask = split_samples_with_modes([sample], has_weight_scale=True)
    assert mask.tolist() == [1.0]

    net = create_network(hidden=64, n_blocks=2)
    out = alphazero_loss_batch(
        net, records,
        calibration_positions=records,
        calibration_weights=weights,
        calibration_loss_weight=1.0,
        calibration_teacher_policy_mask=mask,
        teacher_value_weight=1.0, teacher_policy_kl_weight=0.25,
    )
    assert len(out) == 14                                  # 14-tuple teacher path
    ce = float(out[11])                                    # CALIB_POLICY_CE_IDX
    assert _math.isfinite(ce) and ce > 0.0
    assert int(out[13]) == 1                               # n_retention counts the root row
