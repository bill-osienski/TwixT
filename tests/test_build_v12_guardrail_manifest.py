import csv
from scripts.GPU.alphazero.build_v12_guardrail_manifest import (
    make_guardrail_clone, SOURCE_TO_GUARDRAIL_TAG)


def _parent(side, tv):
    return {"case_id": "game_1_ply_9", "tag": "goal_line_retention",
            "loss_mode": "mcts_root_retention", "side_to_move": side,
            "teacher_value": tv, "target_black_value": "",
            "root_visits_json": "[0.5,0.5]", "root_legal_moves_sha1": "abc",
            "teacher_policy_json": "", "extra_moves_json": "",
            "root_black_value": "0.83", "root_value_stm": "0.80",
            "root_sims": "400"}


def test_clone_converts_stm_teacher_value_to_black_target():
    # black to move: black target == stm teacher value
    b = make_guardrail_clone(_parent("black", "0.30"))
    assert b["loss_mode"] == "asymmetric_guardrail_retention"
    assert b["tag"] == "goal_line_guardrail_retention"
    assert b["case_id"] == "game_1_ply_9__guardrail"
    assert float(b["target_black_value"]) == 0.30
    # red to move: black target == -stm teacher value
    r = make_guardrail_clone(_parent("red", "-0.97"))
    assert float(r["target_black_value"]) == 0.97
    # value-only: policy/root blanked
    for row in (b, r):
        assert row["teacher_policy_json"] == ""
        assert row["root_visits_json"] == ""
        # root-search metadata blanked (no stale MCTS anchors leak)
        assert row["root_black_value"] == ""
        assert row["root_value_stm"] == ""
        assert row["root_sims"] == ""
        assert row["root_legal_moves_sha1"] == ""


def test_tag_map_covers_bcd_roots():
    assert SOURCE_TO_GUARDRAIL_TAG == {
        "goal_line_retention": "goal_line_guardrail_retention",
        "old_post_opening_retention": "old_post_opening_guardrail_retention",
        "red_predrop_retention": "red_predrop_guardrail_retention"}
