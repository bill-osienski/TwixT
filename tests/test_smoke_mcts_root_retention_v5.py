import csv
import json

import numpy as np
import pytest

from scripts.GPU.alphazero.build_mcts_root_retention_manifest import build_rows
from tests.goal_line_probe_fixtures import legal_replay


def _sharp_search(state, seed):
    legal = state.legal_moves()
    counts = {m: 0 for m in legal}
    counts[legal[0]] = 400                     # sharp target != raw priors
    return counts, 0.3


def _manifest_from_net(tmp_path, net):
    """1 retention row whose raw anchor comes from THIS network (gate-0: base==candidate)."""
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    rp = tmp_path / "game_000001.json"
    rp.write_text(json.dumps(legal_replay(9, game_idx=1)))
    net.eval()                                  # raw anchor cached in eval mode (mirrors builder main())
    rows = build_rows(
        [{"game_idx": "1", "case_id": "r1", "replay_path": str(rp),
          "position_ply": "5", "side_to_move": "black",
          "tag": "old_post_opening_retention", "weight_scale": "1.0"}],
        LocalGPUEvaluator(net), _sharp_search,
        sims=400, base_checkpoint="in-memory",
        pos_base_seed=20260616, goal_base_seed=20260614,
        eval_batch_size=14, stall_flush_sims=48)
    man = tmp_path / "v5.csv"
    with man.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return man


def test_root_retention_mechanics_gate0(tmp_path):
    from scripts.GPU.alphazero.smoke_mcts_root_retention_v5 import (
        assert_root_retention_mechanics)
    from scripts.GPU.alphazero.network import create_network

    net = create_network(hidden=64, n_blocks=2)
    man = _manifest_from_net(tmp_path, net)
    net.train()                                 # production-faithful: net in TRAIN at smoke time
    stats = assert_root_retention_mechanics(net, str(man), value_tol=1e-4)
    assert abs(stats["value_mse"]) < 1e-4       # raw anchor reproduces (eval-mode forward)
    assert np.isfinite(stats["policy_ce"]) and stats["policy_ce"] > 0.0
    assert stats["kl_est"] > 1e-3               # sharp target vs raw priors: genuinely > 0
    assert stats["n_retention"] == 1


def test_smoke_rejects_wrong_schema(tmp_path):
    from scripts.GPU.alphazero.smoke_mcts_root_retention_v5 import (
        assert_root_retention_mechanics)
    from scripts.GPU.alphazero.network import create_network
    man = tmp_path / "hard.csv"
    rp = tmp_path / "game_000001.json"
    rp.write_text(json.dumps(legal_replay(9, game_idx=1)))
    with man.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "game_idx", "case_id", "replay_path", "position_ply",
            "side_to_move", "target_black_value"])
        w.writeheader()
        w.writerow({"game_idx": "1", "case_id": "h1", "replay_path": str(rp),
                    "position_ply": "5", "side_to_move": "black",
                    "target_black_value": "-0.35"})
    with pytest.raises(AssertionError, match="schema"):
        assert_root_retention_mechanics(create_network(hidden=64, n_blocks=2), str(man))
