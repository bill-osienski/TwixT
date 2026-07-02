import csv
import json

import numpy as np
import pytest

from scripts.GPU.alphazero.calibration_pool import (
    CONTINUATION_LOSS_MODE, legal_moves_sha1)
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
from scripts.GPU.alphazero.smoke_searched_continuation_retention_v6 import (
    V6_TAG_SCHEDULE, assert_continuation_retention_mechanics)
from tests.goal_line_probe_fixtures import legal_replay


def _manifest(tmp_path, teacher_value):
    """1 hard_value row + 1 inert root row + 3 continuation rows (one per
    continuation tag), all on the same tiny replay."""
    rp = tmp_path / "game_000001.json"
    replay = legal_replay(9, game_idx=1)
    rp.write_text(json.dumps(replay))
    state = position_state(replay, 5, "black")
    legal = state.legal_moves()
    m1 = legal[0]
    s1 = state.apply_move(m1)
    dense = [0.0] * len(legal); dense[0] = 1.0
    common = {"game_idx": "1", "replay_path": str(rp), "position_ply": "5",
              "side_to_move": "black", "weight_scale": "1.0"}
    cont_common = {
        "loss_mode": CONTINUATION_LOSS_MODE,
        "teacher_value": repr(teacher_value),
        "extra_moves_json": json.dumps([{"row": m1[0], "col": m1[1]}]),
        "continuation_side_to_move": s1.to_move,
        "continuation_legal_moves_sha1": legal_moves_sha1(s1.legal_moves()),
        **common}
    rows = [
        {"case_id": "corr1", "tag": "black_predrop_correction",
         "loss_mode": "hard_value", "target_black_value": "-0.35", **common},
        {"case_id": "root1", "tag": "old_post_opening_retention",
         "loss_mode": "mcts_root_retention", "teacher_value": repr(teacher_value),
         "root_visits_json": json.dumps(dense),
         "root_legal_moves_sha1": legal_moves_sha1(legal), **common},
        {"case_id": "b1", "tag": "goal_line_continuation_retention", **cont_common},
        {"case_id": "c1", "tag": "old_post_opening_continuation_retention", **cont_common},
        {"case_id": "d1", "tag": "red_predrop_continuation_retention", **cont_common},
    ]
    fields = sorted({k for r in rows for k in r})
    p = tmp_path / "v6_manifest.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return p


def _network_value(net):
    """The network's actual eval-mode stm value at the continuation state —
    write it back as teacher_value so the anchor reproduces exactly."""
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.build_teacher_calibration_manifest import _teacher_infer
    replay = legal_replay(9, game_idx=1)
    state = position_state(replay, 5, "black")
    s1 = state.apply_move(state.legal_moves()[0])
    prev = net.training
    net.eval()
    try:
        _, _, v = _teacher_infer(s1, LocalGPUEvaluator(net))
    finally:
        net.train(prev)
    return float(v)


def test_smoke_passes_on_reproducing_anchor(tmp_path):
    from scripts.GPU.alphazero.network import create_network
    net = create_network(hidden=64, n_blocks=2)
    v = _network_value(net)
    p = _manifest(tmp_path, teacher_value=v)
    report = assert_continuation_retention_mechanics(net, str(p))
    assert report["n_continuation"] == 3
    assert report["policy_ce"] == 0.0                  # value-only: mask all zero
    assert report["n_policy_rows"] == 0
    assert abs(report["value_mse"]) < 1e-4
    assert report["draws_by_tag"] == V6_TAG_SCHEDULE   # hard schedule assertion


def test_smoke_fails_on_drifted_anchor(tmp_path):
    from scripts.GPU.alphazero.network import create_network
    net = create_network(hidden=64, n_blocks=2)
    v = _network_value(net)
    drifted = max(-1.0, min(1.0, v - 0.5))
    p = _manifest(tmp_path, teacher_value=drifted)
    with pytest.raises(AssertionError, match="value"):
        assert_continuation_retention_mechanics(net, str(p))


def test_smoke_fails_on_wrong_schema(tmp_path):
    """A v5-only manifest (no continuation rows) must be rejected."""
    from scripts.GPU.alphazero.network import create_network
    rp = tmp_path / "game_000001.json"
    replay = legal_replay(9, game_idx=1)
    rp.write_text(json.dumps(replay))
    state = position_state(replay, 5, "black")
    legal = state.legal_moves()
    dense = [0.0] * len(legal); dense[0] = 1.0
    rows = [{"game_idx": "1", "case_id": "root1", "replay_path": str(rp),
             "position_ply": "5", "side_to_move": "black",
             "tag": "old_post_opening_retention",
             "loss_mode": "mcts_root_retention", "teacher_value": "0.0",
             "root_visits_json": json.dumps(dense),
             "root_legal_moves_sha1": legal_moves_sha1(legal)}]
    fields = sorted({k for r in rows for k in r})
    p = tmp_path / "v5_only.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    net = create_network(hidden=64, n_blocks=2)
    with pytest.raises(AssertionError, match="schema"):
        assert_continuation_retention_mechanics(net, str(p))
