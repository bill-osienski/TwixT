import csv
import json

import numpy as np
import pytest

from scripts.GPU.alphazero import eval_raw_nn_position_rows as R
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
from tests.goal_line_probe_fixtures import legal_replay


class _FakeEval:
    """Deterministic stand-in for LocalGPUEvaluator: uniform priors + fixed value. No MCTS."""

    def __init__(self, value=0.2):
        self._value = value

    def build_input_tensor(self, state):
        return state.to_tensor()

    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        b, m = move_mask.shape
        priors = move_mask / np.maximum(move_mask.sum(axis=1, keepdims=True), 1.0)
        values = np.full((b,), self._value, dtype=np.float32)
        return priors.astype(np.float32), values


def _replay_file(tmp_path, n=9, game_idx=1):
    rp = tmp_path / f"game_{game_idx:06d}.json"
    rp.write_text(json.dumps(legal_replay(n, game_idx=game_idx)))
    return rp


def _case(rp, case_id, ply, side, **extra):
    base = {
        "game_idx": "1", "case_id": case_id, "replay_path": str(rp),
        "position_ply": str(ply), "side_to_move": side,
    }
    base.update(extra)
    return base


def test_to_black_flips_red_to_move():
    assert R.to_black(0.7, "black") == pytest.approx(0.7)
    assert R.to_black(0.7, "red") == pytest.approx(-0.7)
    with pytest.raises(ValueError):
        R.to_black(0.1, "green")


def test_score_row_red_to_move_flips_black_value(tmp_path):
    rp = _replay_file(tmp_path)
    row = R.score_row(_FakeEval(value=0.2), _case(rp, "red1", 4, "red"))  # 4 moves -> red to move
    assert row["raw_value_stm"] == pytest.approx(0.2)
    assert row["raw_black_value"] == pytest.approx(-0.2)     # red-to-move: black = -stm
    assert row["overvalue"] is False and row["severe_overvalue"] is False


def test_score_row_black_overvalue_flags_and_top1(tmp_path):
    rp = _replay_file(tmp_path)
    row = R.score_row(_FakeEval(value=0.6), _case(rp, "b1", 5, "black"))  # 5 moves -> black
    assert row["raw_black_value"] == pytest.approx(0.6)
    assert row["overvalue"] is True and row["severe_overvalue"] is True   # 0.6 >= 0.50
    assert ":" in row["top1_move"] and 0.0 < row["top1_prob"] <= 1.0


def test_raw_nn_rows_scores_with_local_evaluator(tmp_path):
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator

    rp = _replay_file(tmp_path)
    net = create_network(hidden=64, n_blocks=2)
    net.eval()
    ev = LocalGPUEvaluator(net)
    row = R.score_row(ev, _case(rp, "ret1", 5, "black"))
    # score_row must apply NO transform to the stm value beyond the shared infer wrapper.
    state = position_state(legal_replay(9, game_idx=1), 5, "black")
    _, _, value = R._teacher_infer(state, ev)   # _teacher_infer wraps evaluator.infer (the "direct infer")
    assert row["raw_value_stm"] == pytest.approx(value, abs=1e-6)
    assert row["raw_black_value"] == pytest.approx(value, abs=1e-6)   # black-to-move: no flip


def test_score_row_side_to_move_mismatch_raises(tmp_path):
    rp = _replay_file(tmp_path)
    with pytest.raises(ValueError, match="side_to_move"):
        R.score_row(_FakeEval(), _case(rp, "bad", 4, "black"))   # ply 4 -> red; claims black


def test_delta_vs_teacher_uses_manifest_value_stm_perspective():
    rows = [
        {"checkpoint": "/ck/base.st", "case_id": "x", "raw_value_stm": 0.10, "teacher_value": "-0.50"},
        {"checkpoint": "/ck/cand.st", "case_id": "x", "raw_value_stm": 0.30, "teacher_value": "-0.50"},
    ]
    R.resolve_deltas(rows, base_checkpoint="/ck/base.st")
    cand = next(r for r in rows if r["checkpoint"] == "/ck/cand.st")
    assert cand["teacher_value_source"] == "manifest"
    assert cand["teacher_value"] == pytest.approx(-0.50)
    assert cand["value_delta_vs_teacher"] == pytest.approx(0.30 - (-0.50))  # stm - stm, NO flip
    assert cand["abs_value_delta_vs_teacher"] == pytest.approx(0.80)


def test_delta_vs_teacher_falls_back_to_base_when_no_manifest_value():
    rows = [
        {"checkpoint": "/ck/base.st", "case_id": "y", "raw_value_stm": 0.11, "teacher_value": ""},
        {"checkpoint": "/ck/v4.st",   "case_id": "y", "raw_value_stm": 0.42, "teacher_value": ""},
    ]
    R.resolve_deltas(rows, base_checkpoint="/ck/base.st")
    base = next(r for r in rows if r["checkpoint"] == "/ck/base.st")
    v4 = next(r for r in rows if r["checkpoint"] == "/ck/v4.st")
    assert v4["teacher_value_source"] == "base_checkpoint"
    assert v4["teacher_value"] == pytest.approx(0.11)             # BASE raw is the teacher
    assert v4["value_delta_vs_teacher"] == pytest.approx(0.31)    # 0.42 - 0.11
    assert base["value_delta_vs_teacher"] == pytest.approx(0.0)   # base vs itself


def _write_manifest(path, cases):
    cols = [
        "game_idx", "case_id", "replay_path", "position_ply", "side_to_move",
        "tag", "target_black_value", "teacher_value", "source", "source_rank",
        "loss_mode", "teacher_legal_moves_sha1",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for c in cases:
            w.writerow(c)


def test_case_id_filter(tmp_path):
    rp = _replay_file(tmp_path)
    man = tmp_path / "m.csv"
    _write_manifest(man, [_case(rp, "keep", 5, "black"), _case(rp, "drop", 5, "black")])
    cases = R.load_and_filter_cases([str(man)], case_ids={"keep"})
    assert [c["case_id"] for c in cases] == ["keep"]


def test_union_across_manifests_and_dedup(tmp_path):
    rp = _replay_file(tmp_path)
    m1, m2 = tmp_path / "m1.csv", tmp_path / "m2.csv"
    _write_manifest(m1, [_case(rp, "a", 5, "black")])
    _write_manifest(m2, [_case(rp, "a", 5, "black"), _case(rp, "b", 4, "red")])  # 'a' duplicated
    cases = R.load_and_filter_cases([str(m1), str(m2)])
    assert sorted(c["case_id"] for c in cases) == ["a", "b"]   # dedup 'a'


def test_tag_and_limit_filters(tmp_path):
    rp = _replay_file(tmp_path)
    man = tmp_path / "m.csv"
    _write_manifest(man, [
        _case(rp, "c1", 5, "black", tag="old_post_opening_retention"),
        _case(rp, "c2", 5, "black", tag="goal_line_retention"),
    ])
    assert [c["case_id"] for c in R.load_and_filter_cases([str(man)], tags={"goal_line_retention"})] == ["c2"]
    assert len(R.load_and_filter_cases([str(man)], limit=1)) == 1


def test_main_end_to_end_with_fake_factory(tmp_path):
    rp = _replay_file(tmp_path)
    man = tmp_path / "m.csv"
    _write_manifest(man, [
        _case(rp, "b1", 5, "black", teacher_value="-0.50"),  # manifest teacher present
        _case(rp, "r1", 4, "red"),                           # no teacher -> base fallback
    ])
    base = tmp_path / "base.safetensors"; base.write_text("x")
    cand = tmp_path / "cand.safetensors"; cand.write_text("x")
    out = tmp_path / "out.csv"

    rc = R.main(
        ["--manifest", str(man),
         "--checkpoint", str(base), "--checkpoint", str(cand),
         "--out", str(out)],
        evaluator_factory=lambda ckpt: _FakeEval(value=0.2 if "base" in ckpt else 0.4),
    )
    assert rc == 0

    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4                                     # 2 checkpoints x 2 cases
    for col in ["raw_value_stm", "raw_black_value", "value_delta_vs_teacher",
                "teacher_value_source", "top1_move", "top1_prob",
                "overvalue", "severe_overvalue"]:
        assert col in rows[0]

    b1_cand = next(r for r in rows if r["case_id"] == "b1" and r["checkpoint"] == str(cand))
    assert b1_cand["teacher_value_source"] == "manifest"
    assert float(b1_cand["value_delta_vs_teacher"]) == pytest.approx(0.4 - (-0.5))  # 0.9

    r1_cand = next(r for r in rows if r["case_id"] == "r1" and r["checkpoint"] == str(cand))
    assert r1_cand["teacher_value_source"] == "base_checkpoint"
    assert float(r1_cand["value_delta_vs_teacher"]) == pytest.approx(0.4 - 0.2)     # stm-space, 0.2
    assert float(r1_cand["raw_black_value"]) == pytest.approx(-0.4)                 # red -> flip


def test_case_id_filter_reflected_in_output(tmp_path):
    rp = _replay_file(tmp_path)
    man = tmp_path / "m.csv"
    _write_manifest(man, [_case(rp, "keep", 5, "black"), _case(rp, "drop", 5, "black")])
    base = tmp_path / "base.safetensors"; base.write_text("x")
    out = tmp_path / "out.csv"
    rc = R.main(
        ["--manifest", str(man), "--checkpoint", str(base),
         "--case-id", "keep", "--out", str(out)],
        evaluator_factory=lambda ckpt: _FakeEval(),
    )
    assert rc == 0
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert {r["case_id"] for r in rows} == {"keep"}


def test_main_side_to_move_mismatch_raises(tmp_path):
    rp = _replay_file(tmp_path)
    man = tmp_path / "m.csv"
    _write_manifest(man, [_case(rp, "bad", 4, "black")])      # ply 4 -> red; claims black
    base = tmp_path / "base.safetensors"; base.write_text("x")
    out = tmp_path / "o.csv"
    with pytest.raises(ValueError, match="side_to_move"):
        R.main(["--manifest", str(man), "--checkpoint", str(base), "--out", str(out)],
               evaluator_factory=lambda ckpt: _FakeEval())


def test_module_does_not_import_mcts():
    import importlib
    mod = importlib.import_module("scripts.GPU.alphazero.eval_raw_nn_position_rows")
    src = open(mod.__file__).read()
    assert "from .mcts" not in src and "MCTS(" not in src
