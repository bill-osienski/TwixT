import json

from scripts.GPU.alphazero.eval_checkpoint_match import run_match, replay_dir_for
from scripts.GPU.alphazero.eval_runner import EvalConfig
from tests.eval_fakes import fake_evaluator_factory


def _tiny_cfg():
    return EvalConfig(board_size=8, mcts_sims=8, mcts_eval_batch_size=4,
                      mcts_stall_flush_sims=4, opening_temp_plies=4,
                      temp_high=1.0, temp_low=0.1, max_moves=12)


def test_replay_dir_for_off_returns_none():
    assert replay_dir_for("logs/eval/m.json", None, False) is None


def test_replay_dir_for_default_derives_from_output_stem():
    assert replay_dir_for("logs/eval/m.json", None, True) == "logs/eval/m_replays"


def test_replay_dir_for_explicit_overrides_default():
    assert replay_dir_for("logs/eval/m.json", "/tmp/rr", True) == "/tmp/rr"


def test_run_match_without_replays_leaves_replay_path_null(tmp_path):
    out = tmp_path / "m.json"
    run_match("A", "B", games=2, base_seed=5, config=_tiny_cfg(), workers=1,
              output=str(out), evaluator_factory=fake_evaluator_factory)
    rows = [json.loads(line) for line in (tmp_path / "m_games.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(r["replay_path"] is None for r in rows)
    assert not (tmp_path / "m_replays").exists()


def test_run_match_with_replays_writes_sidecars_and_links(tmp_path):
    out = tmp_path / "m.json"
    rd = str(tmp_path / "m_replays")
    run_match("A", "B", games=2, base_seed=5, config=_tiny_cfg(), workers=1,
              output=str(out), evaluator_factory=fake_evaluator_factory,
              replay_dir=rd)
    rows = [json.loads(line) for line in (tmp_path / "m_games.jsonl").read_text().splitlines()]
    assert all(r["replay_path"] for r in rows)
    for r in rows:
        assert (tmp_path / "m_replays" / f"game_{r['game_idx']:06d}.json").exists()
    rep = json.loads((tmp_path / "m_replays" / "game_000000.json").read_text())
    assert rep["schema_version"] == 1
    assert len(rep["moves"]) == rep["n_moves"]
    assert rep["seed"] == 5  # base_seed + offset(0) + game_idx(0)
