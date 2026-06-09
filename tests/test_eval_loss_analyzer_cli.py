import json
from pathlib import Path

from scripts.GPU.alphazero.eval_loss_analyzer import main

A = "ckpts/model_iter_0399.safetensors"
B = "ckpts/model_iter_0379.safetensors"


def _row(game_idx, red, black, winner, reason="win", n=50, task_id=0,
         pairing_id="0399_vs_0379"):
    if winner == "red":
        rs, bs, wc = 1.0, 0.0, red
    elif winner == "black":
        rs, bs, wc = 0.0, 1.0, black
    else:
        rs, bs, wc = 0.5, 0.5, None
    return {
        "task_id": task_id, "pairing_id": pairing_id, "game_idx": game_idx,
        "red_checkpoint": red, "black_checkpoint": black,
        "winner": winner, "winner_checkpoint": wc, "reason": reason,
        "n_moves": n, "red_score": rs, "black_score": bs,
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_cli_writes_outputs_and_combined(tmp_path, capsys):
    # match 1: A loses badly (B wins all 4)
    m1 = tmp_path / "weak_0399_vs_0379_games.jsonl"
    _write_jsonl(m1, [_row(i, A, B, "black", n=30 + i) for i in range(4)])
    # match 2: A wins all 4
    m2 = tmp_path / "strong_0399_vs_0379_games.jsonl"
    _write_jsonl(m2, [_row(i, A, B, "red", n=30 + i) for i in range(4)])
    out = tmp_path / "loss_analysis"

    rc = main(["--games-jsonl", str(m1), "--games-jsonl", str(m2),
               "--output-dir", str(out)])
    assert rc == 0

    for stem in ("weak_0399_vs_0379", "strong_0399_vs_0379"):
        assert (out / f"{stem}_loss_summary.json").exists()
        assert (out / f"{stem}_by_color.csv").exists()
        assert (out / f"{stem}_by_length.csv").exists()
        assert (out / f"{stem}_worst_losses.csv").exists()

    combined = (out / "combined_branch_comparison.csv").read_text().splitlines()
    # header + 2 rows, strong first (higher a_score_rate)
    assert combined[1].startswith("strong_0399_vs_0379")
    assert combined[2].startswith("weak_0399_vs_0379")

    summary = json.loads((out / "weak_0399_vs_0379_loss_summary.json").read_text())
    assert summary["verdict"] == "worse"
    assert "LOSS ANALYSIS" in capsys.readouterr().out


def test_cli_skips_self_match(tmp_path, capsys):
    m = tmp_path / "0419_vs_0419_sanity_games.jsonl"
    # self-match rows: same checkpoint both seats, scored as draws (a red
    # "win" against yourself is semantically odd). Never validated — the CLI
    # skips on resolved a == b before validate_rows runs.
    _write_jsonl(
        m,
        [_row(i, A, A, None, reason="state_cap", n=280, pairing_id="0399_vs_0399")
         for i in range(2)],
    )
    rc = main(["--games-jsonl", str(m), "--output-dir", str(tmp_path / "out")])
    assert rc == 0
    assert "self-match" in capsys.readouterr().out
    assert not (tmp_path / "out" / "combined_branch_comparison.csv").exists()


def test_cli_no_inputs_returns_2(tmp_path, capsys):
    rc = main(["--output-dir", str(tmp_path / "out")])
    assert rc == 2
    assert "no input files" in capsys.readouterr().err
