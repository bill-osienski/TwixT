import csv
import json
from pathlib import Path

import pytest

from scripts.GPU.alphazero.generate_goal_line_trigger_probe_manifest import (
    build_manifest, main as gen_main,
)
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import DEFAULT_SELECTION
from scripts.GPU.alphazero.eval_goal_line_trigger_probe import main as probe_main
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import EXPECTED_PROBLEM
from tests.eval_fakes import FakeEvaluator
from tests.goal_line_probe_fixtures import legal_replay

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


def _fake_factory(path):
    # Two distinct constant evaluators so the probe must produce different
    # per-checkpoint readouts. NOTE: FakeEvaluator's constant negates/clamps
    # through the negamax backup (+0.9 leaf -> root -0.9; <=0 leaf -> root ~0.0),
    # so these do NOT model real over/under-valuation. The real 0399-vs-0379
    # direction is the operator acceptance run (Task 6), not a fake unit test.
    return FakeEvaluator(value=0.9 if "0399" in path else 0.0)


def _write_probe_inputs(tmp_path, position_plies=(5, 7)):
    """Write sidecars + a manifest + two dummy checkpoint files; return paths."""
    rdir = tmp_path / "replays"
    rdir.mkdir()
    cases = []
    for i, pp in enumerate(position_plies):
        replay = legal_replay(pp + 3, game_idx=i)     # ensure n_moves > position_ply
        assert replay["moves"][pp]["player"] == "black"  # pp must be black's turn
        rpath = rdir / f"game_{i:06d}.json"
        rpath.write_text(json.dumps(replay))
        cases.append({
            "game_idx": i, "rank": i + 1, "replay_path": str(rpath),
            "position_ply": pp, "side_to_move": "black",
            "expected_problem": EXPECTED_PROBLEM, "trigger_red_ply": pp + 1,
            "trigger_red_move": {"row": 0, "col": 1}, "trigger_zone": "red_goal_row_exact",
            "baseline_black_prev_value": 0.7, "baseline_black_prev_top1": 0.9,
            "drop_black_ply": pp + 2, "drop_black_value": -0.5, "drop_amount": -1.2,
        })
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "name": "goal_line_trigger_black_defense_probe",
        "selection": DEFAULT_SELECTION, "num_cases": len(cases), "cases": cases}))
    ck = tmp_path / "ckpts"
    ck.mkdir()
    a = ck / "model_iter_0379.safetensors"; a.write_text("x")
    b = ck / "model_iter_0399.safetensors"; b.write_text("x")
    return manifest, a, b


def _run(tmp_path, manifest, a, b, outdir):
    return probe_main(
        ["--manifest", str(manifest), "--checkpoint", str(a), "--checkpoint", str(b),
         "--output-dir", str(outdir), "--mcts-sims", "12"],
        evaluator_factory=_fake_factory)


def test_probe_writes_summary_and_cases(tmp_path, capsys):
    manifest, a, b = _write_probe_inputs(tmp_path)
    out = tmp_path / "out"
    assert _run(tmp_path, manifest, a, b, out) == 0
    summary = json.loads((out / "goal_line_trigger_probe_summary.json").read_text())
    assert summary["num_cases"] == 2 and summary["mcts_sims"] == 12
    assert set(summary["checkpoints"]) == {"0379", "0399"}
    rows = list(csv.DictReader((out / "goal_line_trigger_probe_cases.csv").open()))
    assert len(rows) == 4                                   # 2 checkpoints x 2 cases
    assert {"checkpoint", "case_id", "probe_black_root_value", "probe_top1_share",
            "black_overvalue", "baseline_black_prev_value"} <= set(rows[0].keys())


def test_probe_distinguishes_checkpoints(tmp_path):
    manifest, a, b = _write_probe_inputs(tmp_path)
    out = tmp_path / "out"
    _run(tmp_path, manifest, a, b, out)
    s = json.loads((out / "goal_line_trigger_probe_summary.json").read_text())["checkpoints"]
    # Different evaluators -> different per-checkpoint readouts (the comparison
    # machinery works). Exact root values are an MCTS detail; the directional
    # 0399-overvalues-more-than-0379 readout is operator acceptance (Task 6),
    # not a constant-fake unit test.
    assert s["0399"]["mean_black_root_value"] != s["0379"]["mean_black_root_value"]


def test_probe_is_deterministic(tmp_path):
    manifest, a, b = _write_probe_inputs(tmp_path)
    o1, o2 = tmp_path / "o1", tmp_path / "o2"
    _run(tmp_path, manifest, a, b, o1)
    _run(tmp_path, manifest, a, b, o2)
    assert (o1 / "goal_line_trigger_probe_cases.csv").read_text() == \
           (o2 / "goal_line_trigger_probe_cases.csv").read_text()


def test_probe_missing_checkpoint_returns_2(tmp_path):
    manifest, a, _b = _write_probe_inputs(tmp_path)
    rc = probe_main(["--manifest", str(manifest), "--checkpoint", str(a),
                     "--checkpoint", str(tmp_path / "nope.safetensors"),
                     "--output-dir", str(tmp_path / "o")], evaluator_factory=_fake_factory)
    assert rc == 2
    assert not (tmp_path / "o").exists()


def test_probe_out_of_range_position_ply_raises(tmp_path):
    manifest, a, b = _write_probe_inputs(tmp_path)
    data = json.loads(manifest.read_text())
    data["cases"][0]["position_ply"] = 999
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="out of range"):
        _run(tmp_path, manifest, a, b, tmp_path / "o")


def test_probe_disambiguates_colliding_short_ids(tmp_path):
    # Two checkpoints with the SAME iter number from different run dirs must not
    # collide into one summary key.
    manifest, _a, _b = _write_probe_inputs(tmp_path)
    d1 = tmp_path / "runA"; d1.mkdir()
    c1 = d1 / "model_iter_0399.safetensors"; c1.write_text("x")
    d2 = tmp_path / "runB"; d2.mkdir()
    c2 = d2 / "model_iter_0399.safetensors"; c2.write_text("x")
    out = tmp_path / "out"
    rc = probe_main(["--manifest", str(manifest), "--checkpoint", str(c1),
                     "--checkpoint", str(c2), "--output-dir", str(out),
                     "--mcts-sims", "12"], evaluator_factory=_fake_factory)
    assert rc == 0
    s = json.loads((out / "goal_line_trigger_probe_summary.json").read_text())
    assert len(s["checkpoints"]) == 2                 # both kept, not overwritten
    assert set(s["checkpoints"]) == {"runA:0399", "runB:0399"}


def test_probe_missing_manifest_returns_2(tmp_path):
    _m, a, b = _write_probe_inputs(tmp_path)
    rc = probe_main(["--manifest", str(tmp_path / "nope.json"),
                     "--checkpoint", str(a), "--checkpoint", str(b),
                     "--output-dir", str(tmp_path / "o")], evaluator_factory=_fake_factory)
    assert rc == 2
    assert not (tmp_path / "o").exists()
