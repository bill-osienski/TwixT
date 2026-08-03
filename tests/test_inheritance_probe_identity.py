"""Identity and integration qualification for the opt-in inheritance probe."""
import random

from scripts.GPU.alphazero.inheritance_probe import InheritanceProbeConfig
from scripts.GPU.alphazero.mcts import MCTSConfig
from scripts.GPU.alphazero.self_play import play_game
from tests.eval_fakes import FakeEvaluator


def _shipped_batching_config(n_simulations: int = 64) -> MCTSConfig:
    return MCTSConfig(
        n_simulations=n_simulations,
        eval_batch_size=14,
        stall_flush_sims=48,
        pending_virtual_visits=8,
    )


def _play(probe_config, seed: int = 20260803):
    return play_game(
        evaluator=FakeEvaluator(value=0.0),
        mcts_config=_shipped_batching_config(),
        rng=random.Random(seed),
        max_moves=8,
        add_noise=False,
        active_size=24,
        game_id=1,
        inheritance_probe_config=probe_config,
    )


def test_probe_on_records_one_row_per_ply():
    record = _play(InheritanceProbeConfig())
    rows = record.inheritance_probe_record["rows"]
    assert len(rows) >= 2
    assert [row["ply"] for row in rows] == list(range(len(rows)))
    assert rows[0]["starting_visits"] == 0
    assert rows[0]["inherited_fraction_320"] == 0.0
    assert all("played_child_visits" in row for row in rows)


def test_probe_off_leaves_the_record_untouched():
    assert _play(None).inheritance_probe_record is None
