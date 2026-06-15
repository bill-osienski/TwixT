import csv
import json
from pathlib import Path

import pytest

from scripts.GPU.alphazero.generate_goal_line_trigger_probe_manifest import (
    build_manifest, main as gen_main,
)
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import DEFAULT_SELECTION

CANON_DIR = Path("logs/eval/loss_analysis_v2_1")
CANON_CANDIDATES = CANON_DIR / "goal_line_trigger_probe_candidates.csv"
CANON_MANIFEST = CANON_DIR / "goal_line_trigger_probe_manifest.json"

_CAND_HEADER = [
    "game_idx", "rank", "n_moves", "collapse_type", "largest_drop_phase",
    "trigger_zone", "prev_black_ply", "prev_black_row", "prev_black_col",
    "prev_black_value", "prev_black_top1", "trigger_red_ply", "trigger_red_row",
    "trigger_red_col", "trigger_red_value", "trigger_red_top1", "drop_black_ply",
    "drop_black_row", "drop_black_col", "drop_black_value", "drop_black_top1",
    "drop_amount", "replay_path",
]


def _write_candidates(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CAND_HEADER)
        w.writeheader()
        w.writerows(rows)


def _row(**over):
    base = dict(zip(_CAND_HEADER, [
        "769", "4", "45", "sharp_value_drop", "post_opening", "red_goal_band_3",
        "39", "21", "21", "0.88", "0.885", "40", "22", "22", "0.65", "0.955",
        "41", "18", "6", "-0.46", "0.08", "-1.34",
        "logs/eval/x_replays/game_000769.json"]))
    base.update(over)
    return base


def test_build_manifest_shape_and_selection_echo():
    rows = [_row(), _row(game_idx="1", prev_black_value="0.10")]  # 2nd filtered out
    m = build_manifest(rows, DEFAULT_SELECTION, "src.csv")
    assert m["schema_version"] == 1
    assert m["name"] == "goal_line_trigger_black_defense_probe"
    assert m["source"] == "src.csv"
    assert m["selection"] == DEFAULT_SELECTION
    assert m["num_cases"] == 1 and len(m["cases"]) == 1
    assert m["cases"][0]["game_idx"] == 769


def test_generator_cli_writes_manifest(tmp_path):
    csv_path = tmp_path / "cand.csv"
    _write_candidates(csv_path, [_row(), _row(game_idx="2")])
    out = tmp_path / "manifest.json"
    rc = gen_main(["--from-candidates-csv", str(csv_path), "--output", str(out)])
    assert rc == 0
    m = json.loads(out.read_text())
    assert m["num_cases"] == 2 and {c["game_idx"] for c in m["cases"]} == {769, 2}


@pytest.mark.skipif(not CANON_CANDIDATES.exists() or not CANON_MANIFEST.exists(),
                    reason="canonical loss_analysis_v2_1 artifacts not present")
def test_generator_reproduces_canonical_manifest(tmp_path):
    out = tmp_path / "regenerated.json"
    rc = gen_main(["--from-candidates-csv", str(CANON_CANDIDATES), "--output", str(out)])
    assert rc == 0
    got = json.loads(out.read_text())
    want = json.loads(CANON_MANIFEST.read_text())
    got_keys = [(c["game_idx"], c["position_ply"]) for c in got["cases"]]
    want_keys = [(c["game_idx"], c["position_ply"]) for c in want["cases"]]
    assert got_keys == want_keys and got["num_cases"] == want["num_cases"] == 18
