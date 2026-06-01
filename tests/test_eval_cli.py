import json

from scripts.GPU.alphazero.eval_runner import EvalConfig
from scripts.GPU.alphazero.eval_checkpoint_match import run_match
from tests.eval_fakes import fake_evaluator_factory


def _tiny_cfg():
    return EvalConfig(board_size=8, mcts_sims=8, mcts_eval_batch_size=4,
                      mcts_stall_flush_sims=4, opening_temp_plies=4,
                      temp_high=1.0, temp_low=0.1, max_moves=12)


def test_run_match_two_games_writes_outputs(tmp_path):
    out = tmp_path / "m.json"
    summary = run_match(
        a_ckpt="A", b_ckpt="B", games=2, base_seed=1, config=_tiny_cfg(),
        workers=1, output=str(out), evaluator_factory=fake_evaluator_factory,
    )
    # Summary fields present and internally consistent.
    assert summary["games"] == 2
    assert summary["a_wins"] + summary["b_wins"] + summary["state_caps"] \
        + summary["board_full"] == 2
    assert "elo_estimate" in summary and "a_as_red" in summary

    # Files written: summary JSON + per-game JSONL.
    assert out.exists()
    games_file = tmp_path / "m_games.jsonl"
    assert games_file.exists()
    lines = games_file.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert {"task_id", "pairing_id", "game_idx", "winner", "reason"} <= rec.keys()


def test_run_match_pairing_id_default(tmp_path):
    out = tmp_path / "m.json"
    s = run_match(a_ckpt="checkpoints/x/model_iter_0419.safetensors",
                  b_ckpt="checkpoints/x/model_iter_0379.safetensors",
                  games=2, base_seed=0, config=_tiny_cfg(), workers=1,
                  output=str(out), evaluator_factory=fake_evaluator_factory)
    assert s["pairing_id"] == "0419_vs_0379"


import scripts.GPU.alphazero.eval_checkpoint_tournament as tourney_mod
from scripts.GPU.alphazero.eval_checkpoint_tournament import (
    run_tournament, parse_pairings,
)


def test_parse_pairings_resolves_ids():
    pairs = parse_pairings("0419:0379,0419:0339", "checkpoints/x")
    assert pairs == [
        ("checkpoints/x/model_iter_0419.safetensors",
         "checkpoints/x/model_iter_0379.safetensors"),
        ("checkpoints/x/model_iter_0419.safetensors",
         "checkpoints/x/model_iter_0339.safetensors"),
    ]


def test_run_tournament_calls_run_game_tasks_once(tmp_path, monkeypatch):
    calls = {"n": 0}
    real = tourney_mod.run_game_tasks

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(tourney_mod, "run_game_tasks", counting)

    pairings = [("A", "B"), ("A", "C")]
    out = run_tournament(
        pairings=pairings, games=2, base_seed=0, config=_tiny_cfg(),
        workers=1, output_dir=str(tmp_path),
        evaluator_factory=fake_evaluator_factory,
    )
    assert calls["n"] == 1                       # ONE pool, no nested calls
    assert len(out["pairings"]) == 2
    assert (tmp_path / "tournament.json").exists()


def test_run_tournament_writes_per_pairing_files(tmp_path):
    out = run_tournament(
        pairings=[("A", "B")], games=2, base_seed=0, config=_tiny_cfg(),
        workers=1, output_dir=str(tmp_path),
        evaluator_factory=fake_evaluator_factory,
    )
    assert (tmp_path / "A_vs_B.json").exists()
