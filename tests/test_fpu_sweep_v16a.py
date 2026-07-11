import pytest
from scripts.GPU.alphazero.diagnose_fpu_sweep import (
    _parse_args, manifest_is_neutral, resolve_integrity_csv, resolve_fpu_values,
    resolve_output_paths, PROTOCOL_FPUS, DEFAULT_A_MANIFEST, DEFAULT_PHASE0_CSV,
    DEFAULT_OUT, DEFAULT_FPUS)


def test_cli_aliases_and_none_sentinels():
    assert _parse_args(["--manifest", "m"]).manifest == "m"
    assert _parse_args(["--a-manifest", "m"]).manifest == "m"
    assert _parse_args(["--integrity-csv", "i"]).integrity_csv == "i"
    assert _parse_args(["--phase0-csv", "i"]).integrity_csv == "i"
    a = _parse_args([])
    assert a.integrity_csv is None and a.fpu_values is None
    assert a.out is None and a.summary_out is None and a.strata_summary_out is None
    assert a.skip_integrity_check is False and a.allow_non_protocol_fpu is False


def test_manifest_is_neutral_is_strict():
    assert manifest_is_neutral([{"ply_bucket": "midgame"}, {"ply_bucket": "late"}]) is True
    assert manifest_is_neutral([{"case_id": "c"}, {"case_id": "d"}]) is False
    with pytest.raises(ValueError):
        manifest_is_neutral([{"ply_bucket": "late"}, {"case_id": "d"}])   # mixed
    with pytest.raises(ValueError):
        manifest_is_neutral([])                                           # empty


def test_resolve_integrity_conditional():
    assert resolve_integrity_csv(None, False, False, DEFAULT_PHASE0_CSV) == DEFAULT_PHASE0_CSV
    assert resolve_integrity_csv(None, False, True, DEFAULT_PHASE0_CSV) is None
    assert resolve_integrity_csv("x", False, True, DEFAULT_PHASE0_CSV) == "x"
    assert resolve_integrity_csv(None, True, False, DEFAULT_PHASE0_CSV) is None


def test_resolve_fpu_values_frozen_protocol():
    assert resolve_fpu_values(None, True, False) == PROTOCOL_FPUS       # neutral default
    assert resolve_fpu_values(None, False, False) == [float(x) for x in DEFAULT_FPUS.split(",")]
    assert resolve_fpu_values("0.0,-0.20", True, False) == [0.0, -0.20]
    with pytest.raises(SystemExit):
        resolve_fpu_values("0.0,-0.10,-0.20", True, False)             # non-protocol, no override
    assert resolve_fpu_values("0.0,-0.10,-0.20", True, True) == [0.0, -0.10, -0.20]
    with pytest.raises(SystemExit):
        resolve_fpu_values("-0.20", True, True)                        # missing baseline 0.0


def test_resolve_output_paths_mode_scoped():
    lo = resolve_output_paths(None, None, None, "a/x.csv", False)
    assert lo[0] == DEFAULT_OUT                                         # legacy -> A defaults
    ne = resolve_output_paths(None, None, None, "logs/eval/v16a/m.csv", True)
    assert ne == ("logs/eval/v16a/neutral_fpu_sweep_cases.csv",
                  "logs/eval/v16a/neutral_fpu_sweep_summary.csv",
                  "logs/eval/v16a/neutral_fpu_sweep_by_stratum.csv")
    assert resolve_output_paths("o", "s", "t", "m.csv", True) == ("o", "s", "t")  # explicit wins


import math
from scripts.GPU.alphazero.diagnose_fpu_sweep import (
    visit_entropy, enrich_with_deltas, GENERIC_CASE_FIELDNAMES)


def test_visit_entropy():
    assert abs(visit_entropy([5, 5, 5, 5]) - math.log(4)) < 1e-12
    assert visit_entropy([10]) == 0.0 and visit_entropy([]) == 0.0


def _rich(fpu, cid, stm, blk, top, rootc, topc, share, eff, ent, col):
    return {"fpu_value": fpu, "case_id": cid, "root_mcts_stm_value": stm,
            "root_mcts_black_value": blk, "top_child_move": top,
            "root_n_visited_children": rootc, "top_child_n_visited_children": topc,
            "top_child_visit_share": share, "root_effective_children": eff,
            "root_visit_entropy": ent, "root_collapsed_ge_0_95": col}


def test_enrich_mover_black_shape_and_collapse_deltas():
    rows = [_rich(0.0, "A", 0.20, -0.20, "3:4", 5, 200, 0.60, 6.0, 1.8, False),
            _rich(-0.2, "A", 0.05, -0.05, "3:4", 8, 120, 0.97, 2.0, 0.3, True)]
    enrich_with_deltas(rows)
    c = rows[1]
    assert abs(c["root_value_delta_stm_vs_fpu0"] - (-0.15)) < 1e-12
    assert abs(c["root_value_delta_black_vs_fpu0"] - 0.15) < 1e-12
    assert c["root_children_delta_vs_fpu0"] == 3
    assert c["top_child_children_delta_vs_fpu0"] == -80
    assert abs(c["root_effective_children_delta_vs_fpu0"] - (-4.0)) < 1e-12
    assert abs(c["root_visit_entropy_delta_vs_fpu0"] - (-1.5)) < 1e-12
    assert c["new_collapse_vs_fpu0"] is True and c["resolved_collapse_vs_fpu0"] is False
    assert rows[0]["new_collapse_vs_fpu0"] is False


def test_enrich_resolved_collapse_and_blank_share():
    rows = [_rich(0.0, "A", 0.2, -0.2, "", 5, 0, "", 1.0, 0.0, True),
            _rich(-0.2, "A", 0.2, -0.2, "9:9", 5, 200, 0.6, 3.0, 1.0, False)]
    enrich_with_deltas(rows)
    assert rows[1]["resolved_collapse_vs_fpu0"] is True
    assert rows[1]["top_move_changed_vs_fpu0"] is True
    assert rows[1]["top_child_visit_share_delta_vs_fpu0"] == ""    # baseline blank


def test_generic_case_fieldnames_no_redundant_top1_share():
    assert "root_top1_visit_share" not in GENERIC_CASE_FIELDNAMES
    for k in ("root_mcts_stm_value", "top_child_visit_share", "root_collapsed_ge_0_95",
              "root_value_delta_stm_vs_fpu0", "new_collapse_vs_fpu0"):
        assert k in GENERIC_CASE_FIELDNAMES


from scripts.GPU.alphazero.diagnose_fpu_sweep import (
    _percentile, _delta_metrics, summarize_grouped,
    GENERIC_SUMMARY_FIELDNAMES, STRATA_SUMMARY_FIELDNAMES)


def test_percentile():
    assert abs(_percentile([0, 1, 2, 3, 4], 90) - 3.6) < 1e-12
    assert _percentile([7.0], 95) == 7.0


def _e(cid, bucket, side, stm, blk, changed, rootc, topc, share, eff, ent, col,
       ecd=0.0, entd=0.0, tccd=0, newc=False, resc=False, rcd=0, tcsd=0.0):
    return {"fpu_value": -0.2, "case_id": cid, "ply_bucket": bucket,
            "side_to_move": side, "root_value_delta_stm_vs_fpu0": stm,
            "root_value_delta_black_vs_fpu0": blk, "top_move_changed_vs_fpu0": changed,
            "root_n_visited_children": rootc, "top_child_n_visited_children": topc,
            "top_child_visit_share": share, "root_effective_children": eff,
            "root_visit_entropy": ent, "root_collapsed_ge_0_95": col,
            "root_effective_children_delta_vs_fpu0": ecd,
            "root_visit_entropy_delta_vs_fpu0": entd,
            "top_child_children_delta_vs_fpu0": tccd,
            "root_children_delta_vs_fpu0": rcd,
            "top_child_visit_share_delta_vs_fpu0": tcsd,
            "new_collapse_vs_fpu0": newc, "resolved_collapse_vs_fpu0": resc}


def test_black_cancels_mover_preserved():
    rows = [_e(f"c{i}", "midgame", s, -0.10, (-0.10 if s == "black" else 0.10),
               False, 6, 100, 0.5, 5, 1.5, False)
            for i, s in enumerate(["black", "red", "black", "red"])]
    m = _delta_metrics(rows)
    assert abs(m["mean_root_value_delta_black_vs_fpu0"]) < 1e-12       # cancels
    assert abs(m["mean_root_value_delta_stm_vs_fpu0"] - (-0.10)) < 1e-12  # preserved


def test_paired_shape_deltas_and_collapse_counts_and_stable_top():
    rows = [_e("a", "late", "black", -0.2, -0.2, False, 6, 100, 0.96, 4, 1.0, True,
               ecd=-2.0, tccd=-50, newc=True),
            _e("b", "late", "red", 0.1, -0.1, True, 4, 300, 0.5, 8, 2.0, False,
               ecd=1.0, tccd=+30, resc=True)]
    m = _delta_metrics(rows)
    assert m["new_collapse_count"] == 1 and m["resolved_collapse_count"] == 1
    assert abs(m["new_collapse_rate"] - 0.5) < 1e-12
    assert abs(m["mean_root_effective_children_delta_vs_fpu0"] - (-0.5)) < 1e-12
    # stable-top paired reply delta uses only the unchanged-top row (a): -50
    assert abs(m["mean_top_child_children_delta_stable_top_vs_fpu0"] - (-50)) < 1e-12
    assert abs(m["mean_top_child_children_delta_vs_fpu0"] - (-10)) < 1e-12


def test_summarize_grouped_strata():
    rows = [_e("a", "midgame", "black", -0.2, -0.2, True, 6, 100, 0.5, 5, 1.5, False),
            _e("b", "midgame", "red", 0.0, 0.0, False, 4, 200, 0.5, 6, 1.6, False),
            _e("c", "late", "black", 0.4, 0.4, True, 2, 300, 0.7, 3, 1.0, False)]
    assert [g["group"] for g in summarize_grouped(rows, "bucket")] == ["midgame", "late"]
    assert {g["group"] for g in summarize_grouped(rows, "side")} == {"black", "red"}
    assert {g["group"] for g in summarize_grouped(rows, "bucket_x_side")} == {
        "midgame|black", "midgame|red", "late|black"}
    assert STRATA_SUMMARY_FIELDNAMES[:3] == ["fpu_value", "group_kind", "group"]
    assert "mean_root_value_delta_stm_vs_fpu0" in GENERIC_SUMMARY_FIELDNAMES


import csv as _csv
from scripts.GPU.alphazero.diagnose_fpu_sweep import (
    _legacy_case_row, _generic_case_row, _write_csv, FIELDNAMES)


def test_legacy_projection_exact_columns():
    rich = {k: 0 for k in FIELDNAMES}
    rich.update({"fpu_value": 0.0, "case_id": "A", "extra_key": 99})
    assert list(_legacy_case_row(rich).keys()) == FIELDNAMES     # no extras -> DictWriter safe


def test_legacy_case_csv_golden_bytes(tmp_path):
    r = {"fpu_value": 0.0, "case_id": "game_x", "root_mcts_black_value": 0.5,
         "gate_over_ge_0_25": True, "gate_severe_ge_0_50": False,
         "root_n_visited_children": 3, "top_child_move": "12:8",
         "top_child_visit_share": 0.75, "top_child_q_black": -0.25,
         "top_child_n_visited_children": 42}
    p = tmp_path / "c.csv"
    _write_csv(str(p), FIELDNAMES, [_legacy_case_row(r)])
    expected = (
        "fpu_value,case_id,root_mcts_black_value,gate_over_ge_0_25,"
        "gate_severe_ge_0_50,root_n_visited_children,top_child_move,"
        "top_child_visit_share,top_child_q_black,top_child_n_visited_children\r\n"
        "0.0,game_x,0.5,True,False,3,12:8,0.75,-0.25,42\r\n")
    assert p.read_bytes() == expected.encode()


import types
from pathlib import Path as _P
import scripts.GPU.alphazero.diagnose_fpu_sweep as sweep
from scripts.GPU.alphazero.mcts import MCTSNode, encode_move


def _fake_root(root_stm, children):
    """children: [(rc, visits, q_value, to_move)]. Root has visit_count = SIMS so
    the sim-count guard passes; children give _best_child / entropy something to
    read. q_value = value_sum / visit_count, so value_sum = q * v."""
    root = MCTSNode(state=types.SimpleNamespace(to_move="black"), visit_count=sweep.SIMS)
    for rc, v, q, tm in children:
        ch = MCTSNode(state=types.SimpleNamespace(to_move=tm), parent=root,
                      move=encode_move(*rc), visit_count=v, value_sum=q * v)
        root.children[ch.move] = ch
    return root, root_stm


_FAKE = {
    ("game_000005_ply_020", 0.0): _fake_root(0.5, [((12, 8), 300, -0.25, "red"), ((1, 1), 100, 0.1, "red")]),
    ("game_000005_ply_020", -0.20): _fake_root(0.3, [((12, 8), 260, -0.10, "red"), ((1, 1), 140, 0.2, "red")]),
}


def _fake_search_for_row(case, fn, **kw):
    root, stm = _FAKE[(case["case_id"], fn)]
    return None, case["side_to_move"], stm, root


def _legacy_manifest(tmp_path):
    p = tmp_path / "legacy.csv"
    p.write_text("game_idx,case_id,replay_path,position_ply,side_to_move\n"
                 "5,game_000005_ply_020,r.json,20,black\n")
    return str(p)


def test_main_legacy_end_to_end_matches_golden(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "_search_fns", lambda *a, **k: {x: x for x in a[1]})
    monkeypatch.setattr(sweep, "search_for_row", _fake_search_for_row)
    monkeypatch.setattr(sweep, "_phase0_baseline", lambda p: {"game_000005_ply_020": 0.5})
    out, summ = tmp_path / "cases.csv", tmp_path / "summary.csv"
    rc = sweep.main(["--manifest", _legacy_manifest(tmp_path), "--fpu-values", "0.0,-0.20",
                     "--integrity-csv", "dummy", "--out", str(out), "--summary-out", str(summ)])
    assert rc == 0
    golden = _P("tests/golden")
    assert out.read_bytes() == (golden / "fpu_sweep_legacy_cases.csv").read_bytes()
    assert summ.read_bytes() == (golden / "fpu_sweep_legacy_summary.csv").read_bytes()


def test_main_legacy_integrity_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "_search_fns", lambda *a, **k: {x: x for x in a[1]})
    monkeypatch.setattr(sweep, "search_for_row", _fake_search_for_row)
    monkeypatch.setattr(sweep, "_phase0_baseline", lambda p: {"game_000005_ply_020": 0.99})
    with pytest.raises(SystemExit):
        sweep.main(["--manifest", _legacy_manifest(tmp_path), "--fpu-values", "0.0,-0.20",
                    "--integrity-csv", "dummy", "--out", str(tmp_path / "o.csv"),
                    "--summary-out", str(tmp_path / "s.csv")])
