"""Card §15's memory budget and revised fit projection: recomputed from the committed timing records, and each
per-unit allowance checked against a fresh tracemalloc measurement of the real structures on fakes (no model)."""
import gc
import json
import pathlib
import sys
import tracemalloc

import numpy as np
import pytest

from scripts.GPU.alphazero import htb_budget as B
from scripts.GPU.alphazero import htb_pilot as H
from scripts.GPU.alphazero import htb_timing as T

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_htb_pilot import FakeTeacher, ids_in, random_game   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
EV = ROOT / T.EVIDENCE


def _recs():
    return (json.loads((EV / "label2" / "record.json").read_text()),
            json.loads((EV / "fit" / "record.json").read_text()))


def _peak(fn):
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    out = fn()
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    return out, peak


def _games(n=12, plies=60):
    return [{"id": g, "starting_player": "red", "moves": random_game(3000 + i, plies),
             "players": [("red", "black")[k % 2] for k in range(plies)]} for i, g in enumerate(ids_in("train", n))]


def test_THE_BUDGETS_AND_THE_PROJECTION_RECOMPUTE_FROM_THE_COMMITTED_RECORDS():
    lab, fit = _recs()
    fb = B.fit_budget(fit)
    assert fb["conservative_sum_bytes"] == sum(fb["terms_bytes"].values())
    assert fb["proposed_memory_required_bytes"] == 12 * B.GIB >= fb["conservative_sum_bytes"]
    assert fb["phase_peak_bytes"] < fb["conservative_sum_bytes"] < fb["as_built_estimate_bytes"]
    assert fb["terms_bytes"]["compact_labels"] == 264_000 * (576 * 4 + 8 + 300)
    assert fb["feature_map_bytes"] == 64 * 24 * 24 * 128 * 4
    assert fb["terms_bytes"]["networks_optimizer_gradients_snapshots"] == 4 * fit["measurements"]["networks_bytes"]
    assert fb["conservative_sum_bytes"] < min(fb["available_bytes_observed"])
    old = B.label_budget(lab, fit, compact=False)                   # the previous label path, as budgeted 2026-11-04
    assert old["terms_bytes"]["labelled_rows_of_the_larger_arm"] == 240_000 * 56_000
    assert old["conservative_sum_bytes"] == sum(old["terms_bytes"].values()) == 23_562_292_049
    lb = B.label_budget(lab, fit)                                   # card §16: compact as produced
    assert "labelled_rows_of_the_larger_arm" not in lb["terms_bytes"] and lb["compact"] is True
    assert lb["terms_bytes"]["compact_labels_of_the_larger_arm"] == 240_000 * (2_333 + 300 + 200)
    assert lb["conservative_sum_bytes"] == sum(lb["terms_bytes"].values()) == 10_840_404_049
    assert lb["proposed_memory_required_bytes"] == H.LABEL_MEMORY_REQUIRED == 11 * B.GIB
    assert fb["proposed_memory_required_bytes"] == H.FIT_MEMORY_REQUIRED
    inv = json.loads((ROOT / H.C0_INVENTORY).read_text())
    assert inv["eligible"]["plies_per_game"]["max"] == B.MAX_GAME_PLIES
    cfg = T.TimingConfig(root=ROOT, data_root=ROOT)
    p = B.batched_fit_projection(fit, cfg)
    m = fit["measurements"]
    r = m["lazy_board_s_per_row"] + m["examples_s_per_row"]
    assert p["rebuild_s_per_row_estimate"] == r and p["rows_built_in_updates"] == 15_000 * 32
    assert "planning estimate" in p["basis"] and "upper bound" not in p["basis"]
    u = m["updates"]
    assert p["terms"]["updates_s"] == round(u["warmup_s"] + 14_990 * u["steady_mean_s"] + 480_000 * r, 1)
    assert p["projected_s"] == 12_301.3 and (p["term_s"], p["kill_s"], p["cleanup_through_s"]) == (18_590, 18_600, 18_610)
    assert p["terms"]["updates_s"] - T.project_fit(m, cfg)["terms"]["updates_s"] > fit["memory_estimate"]["lazy_boards_projected_s"]


def test_A_LABEL_ROW_AS_RUN_LABEL_HOLDS_IT_IS_INSIDE_ITS_ALLOWANCE():
    games = _games()
    want = [(g, [{"ply": k, "key": H.state_key(s)} for k, s in H.positions(g["moves"]) if k % 3 == 0]) for g in games]
    rows, peak = _peak(lambda: [r for g, w in want for r in H.label_game(g["id"], g["moves"], w, FakeTeacher(smooth=True))])
    gc.collect()
    tracemalloc.start()
    b = tracemalloc.get_traced_memory()[0]
    kept = [r for g, w in want for r in H.label_game(g["id"], g["moves"], w, FakeTeacher(smooth=True))]
    per_row = (tracemalloc.get_traced_memory()[0] - b) / len(kept)
    tracemalloc.stop()
    assert np.mean([len(r["legal"]) for r in kept]) > 450                  # near the real ~500 legal moves
    assert 30_000 < per_row <= B.LABEL_ROW_B


def test_COMPACT_METADATA_SELECTION_AND_POOL_ARE_INSIDE_THEIR_ALLOWANCES():
    games = _games()
    plies = sum(len(g["moves"]) for g in games)
    _, sel_peak = _peak(lambda: H.select(games, n_train=50, n_val=0))
    assert sel_peak / plies <= B.SELECT_B_PER_POSITION
    keys = [H.state_key(s) for g in games for _, s in H.positions(g["moves"])]
    meta, b = _peak(lambda: [{"game_id": games[i % len(games)]["id"] + "", "ply": i % 300, "key": k[:] + ""}
                             for i, k in enumerate(keys)])
    assert b / len(meta) <= B.META_ROW_B
    raw = []                                                   # pool entries from freshly parsed corpus-shaped JSON
    for g in games:
        raw.append(json.loads(json.dumps({"id": g["id"], "starting_player": "red", "moves": [
            {"turn": k + 1, "player": p, "row": r, "col": c, "heuristics": {}, "search_score": 0.0}
            for k, ((r, c), p) in enumerate(zip(g["moves"], g["players"]))]})))
    pool, b = _peak(lambda: H.pool_from(raw, arms=("train",)))
    assert b / plies <= B.POOL_B_PER_MOVE


def test_A_BUILT_EXAMPLE_ON_FAKES_IS_INSIDE_THE_TEMPORARY_EXAMPLE_ALLOWANCE():
    _, fit = _recs()
    games = _games(4)
    rows = [{"game_id": g["id"], "ply": k, "key": H.state_key(s), "value": 0.0,
             "target576": np.full(H.PADDED, 1.0 / H.PADDED)}
            for g in games for k, s in H.positions(g["moves"]) if k % 7 == 3]
    ex, b = _peak(lambda: H.build_examples(rows, {g["id"]: g["moves"] for g in games}))
    assert b / len(ex) <= fit["measurements"]["example_bytes_traced"] * B.EXAMPLE_MARGIN


def test_A_COMPACT_LABEL_ROW_IS_INSIDE_ITS_ALLOWANCE():
    games = _games()
    sel = [{"game_id": g["id"], "ply": k, "key": H.state_key(s)} for g in games for k, s in H.positions(g["moves"])]
    sel = list({r["key"]: r for r in sel}.values())
    arrays, b = _peak(lambda: H.LabelArrays(sel))
    assert H.PADDED * 4 < b / len(sel) <= B.COMPACT_ROW_B + B.META_ROW_B + B.KEY_INDEX_B
