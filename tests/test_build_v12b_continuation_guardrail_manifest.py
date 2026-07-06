import csv

from scripts.GPU.alphazero.build_v12b_continuation_guardrail_manifest import (
    make_root_guardrail_clone, make_continuation_guardrail_clone,
    ROOT_TO_GUARDRAIL_TAG, CONTINUATION_TO_GUARDRAIL_TAG)


def _root_parent(side, tv):
    return {"case_id": "game_1_ply_9", "tag": "goal_line_retention",
            "loss_mode": "mcts_root_retention", "side_to_move": side,
            "teacher_value": tv, "target_black_value": "",
            "root_visits_json": "[0.5,0.5]", "root_legal_moves_sha1": "abc",
            "root_black_value": "0.83", "teacher_policy_json": "",
            "extra_moves_json": "", "continuation_side_to_move": ""}


def _cont_parent(cont_side, tv, depth="2"):
    return {"case_id": "game_1_ply_9__cont_pv2",
            "tag": "old_post_opening_continuation_retention",
            "loss_mode": "searched_continuation_retention", "side_to_move": "black",
            "teacher_value": tv, "target_black_value": "",
            "extra_moves_json": '[{"row":3,"col":4},{"row":5,"col":6}]',
            "continuation_side_to_move": cont_side,
            "continuation_legal_moves_sha1": "abc123", "continuation_depth": depth,
            "continuation_parent_case_id": "game_1_ply_9", "continuation_source": "pv",
            "continuation_path_moves": "d4 f6", "continuation_tree_visits": "400",
            "continuation_tree_nn_value": "0.31",
            "teacher_policy_json": "", "root_visits_json": ""}


def test_root_clone_uses_root_side_and_blanks_continuation():
    b = make_root_guardrail_clone(_root_parent("black", "0.30"))
    assert b["loss_mode"] == "asymmetric_guardrail_retention"
    assert b["tag"] == "goal_line_guardrail_retention"
    assert b["case_id"] == "game_1_ply_9__guardrail"
    assert float(b["target_black_value"]) == 0.30              # black root: +tv
    r = make_root_guardrail_clone(_root_parent("red", "-0.97"))
    assert float(r["target_black_value"]) == 0.97             # red root: -tv
    for row in (b, r):
        assert row["extra_moves_json"] == ""
        assert row["continuation_side_to_move"] == ""
        assert row["root_black_value"] == ""
        assert row["teacher_policy_json"] == ""


def test_continuation_clone_uses_continuation_side_and_preserves_reconstruction():
    # continuation side red (odd depth) -> sign -1 -> target_black = tv * -1
    c = make_continuation_guardrail_clone(_cont_parent("red", "-0.40", depth="1"))
    assert c["loss_mode"] == "asymmetric_guardrail_retention"
    assert c["tag"] == "old_post_opening_continuation_guardrail_retention"
    assert c["case_id"] == "game_1_ply_9__cont_pv2__guardrail"
    assert float(c["target_black_value"]) == 0.40            # -0.40 * -1
    # continuation side black (even depth) -> sign +1
    c2 = make_continuation_guardrail_clone(_cont_parent("black", "0.22", depth="2"))
    assert float(c2["target_black_value"]) == 0.22           # 0.22 * +1
    for row in (c, c2):
        # reconstruction/identity fields PRESERVED
        assert row["extra_moves_json"] == '[{"row":3,"col":4},{"row":5,"col":6}]'
        assert row["continuation_legal_moves_sha1"] == "abc123"
        assert row["continuation_source"] == "pv"
        assert row["continuation_depth"] in ("1", "2")
        # policy/root/search-scalar fields BLANKED
        assert row["teacher_policy_json"] == ""
        assert row["root_visits_json"] == ""
        assert row["continuation_tree_visits"] == ""
        assert row["continuation_tree_nn_value"] == ""
        # teacher_value kept as provenance
        assert row["teacher_value"] in ("-0.40", "0.22")


def test_tag_maps():
    assert ROOT_TO_GUARDRAIL_TAG == {
        "goal_line_retention": "goal_line_guardrail_retention",
        "old_post_opening_retention": "old_post_opening_guardrail_retention",
        "red_predrop_retention": "red_predrop_guardrail_retention"}
    assert CONTINUATION_TO_GUARDRAIL_TAG == {
        "old_post_opening_continuation_retention":
            "old_post_opening_continuation_guardrail_retention",
        "red_predrop_continuation_retention":
            "red_predrop_continuation_guardrail_retention"}


def _write_csv(tmp_path, rows):
    fields = sorted({k for r in rows for k in r})
    p = tmp_path / "v7.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return p


def test_main_routes_and_drops(tmp_path, monkeypatch):
    import scripts.GPU.alphazero.build_v12b_continuation_guardrail_manifest as m
    rows = [
        {"case_id": "a", "tag": "black_predrop_correction", "loss_mode": "hard_value",
         "target_black_value": "-0.35", "teacher_value": "", "side_to_move": "black"},
        {"case_id": "sev", "tag": "red_predrop_severe_root_correction",
         "loss_mode": "hard_value", "target_black_value": "-0.35",
         "teacher_value": "", "side_to_move": "red"},
        {"case_id": "broot", "tag": "goal_line_retention",
         "loss_mode": "mcts_root_retention", "teacher_value": "0.2",
         "side_to_move": "black"},
        {"case_id": "ccont", "tag": "old_post_opening_continuation_retention",
         "loss_mode": "searched_continuation_retention", "teacher_value": "0.1",
         "side_to_move": "black", "continuation_side_to_move": "red",
         "extra_moves_json": '[{"row":1,"col":1}]',
         "continuation_legal_moves_sha1": "x"},
        {"case_id": "glcont", "tag": "goal_line_continuation_retention",   # DROPPED
         "loss_mode": "searched_continuation_retention", "teacher_value": "0.1",
         "side_to_move": "black", "continuation_side_to_move": "red"},
        {"case_id": "drv", "tag": "red_predrop_root_value_retention",      # DROPPED
         "loss_mode": "searched_continuation_retention", "teacher_value": "0.4",
         "side_to_move": "black", "continuation_side_to_move": "black",
         "extra_moves_json": ""},
    ]
    inp = _write_csv(tmp_path, rows)
    outp = tmp_path / "v12b.csv"
    monkeypatch.setattr("sys.argv", ["prog", "--input", str(inp), "--output", str(outp)])
    m.main()
    with outp.open(newline="") as f:
        out = list(csv.DictReader(f))
    assert sorted(r["tag"] for r in out) == sorted([
        "black_predrop_correction", "red_predrop_severe_root_correction",
        "goal_line_guardrail_retention",
        "old_post_opening_continuation_guardrail_retention"])
    cc = [r for r in out
          if r["tag"] == "old_post_opening_continuation_guardrail_retention"][0]
    assert cc["extra_moves_json"] == '[{"row":1,"col":1}]'
    assert cc["loss_mode"] == "asymmetric_guardrail_retention"
