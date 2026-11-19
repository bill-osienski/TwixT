"""H-MX model-free tests (card docs/superpowers/2026-11-18-htb-mix-distillation-card.md, revision 1).

No real model is loaded and no gate is opened: the fit runs on tiny networks and fake label sets written here, and
the export on a fake runner. One read-only test loads the four REAL cached label sets (no model) to show they are
reusable as written and that the mix selection holds on them.
"""
import hashlib
import json
import pathlib

import mlx.core as mx
import numpy as np
import pytest

from scripts.GPU.alphazero import htb_mix as M
from scripts.GPU.alphazero import htb_mix_export as XM
from scripts.GPU.alphazero import htb_mix_recovery as R
from scripts.GPU.alphazero import htb_onpolicy as H
from scripts.GPU.alphazero import htb_onpolicy_export as X
from scripts.GPU.alphazero import htb_pilot as P
from tests import test_htb_pilot as TP

ROOT = pathlib.Path(__file__).resolve().parents[1]
CURRENT = ROOT / "models/78c16a31c3dba37f"
GATE_FILES = {"fit": "htb_mix_fit_authorization.py", "export": "htb_mix_export_authorization.py"}


@pytest.fixture(autouse=True)
def _cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


# ─────────────────────────────── card, gates, frozen numbers ───────────────────────────────

def test_THE_CARD_IS_THE_PINNED_REVISION_1():
    assert hashlib.sha256((ROOT / M.CARD).read_bytes()).hexdigest() == M.CARD_SHA256
    assert "**Status:** FROZEN at revision 1" in (ROOT / M.CARD).read_text()


def test_THE_FROZEN_NUMBERS_ARE_THE_CARDS():
    assert (M.N_OP, M.N_TB, M.TB_ROWS, M.TB_DROPPED, M.TB_POOL) == (120_000, 120_000, 240_000, 11, 239_989)
    assert (M.HALF, M.UPDATES, M.RECIPE) == (16, 7_500, H.RECIPE) and M.RECIPE == P.Recipe()
    assert (M.M1_FACTOR, M.M4_FACTOR, M.M5_DROP, M.M6_FACTOR, M.S_REL_TOL) == (0.90, 1.05, 0.010, 1.15, 1e-9)
    assert M.KILL_S == {"fit": 11_700, "export": 3_600} and M.GRACE_S == 10
    assert M.PAIRS == {"match": 400, "match smoke": 2} and M.NS == "htb-mix/20261118"
    assert M.STUDENT == H.STUDENT and M.FIT_TAG == "htb-mix-fit/1"
    assert M.Config().updates == 7_500 and M.FIT_MEMORY_REQUIRED == 12 * 2 ** 30


def test_THE_PINNED_INPUTS_ARE_THE_COMMITTED_FILES():
    for s in M.SOURCES.values():
        for rel, want in ((s["label_record"], s["label_record_sha256"]), (s["selection"], s["selection_sha256"])):
            assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == want
        rec = json.loads((ROOT / s["label_record"]).read_text())
        assert {e["arm"]: e["manifest_sha256"] for e in rec["labels"]} == s["manifests"] and rec["counts"] == s["counts"]
    for rel, want in ((M.OP_GAMES, M.OP_GAMES_SHA256), (M.OP_FIT_RECORD, M.OP_FIT_RECORD_SHA256), *M.OP_OPENINGS.items()):
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == want


@pytest.mark.parametrize("stage", sorted(GATE_FILES))
def test_EVERY_GATE_IS_CLOSED_AND_ITS_FILE_HOLDS_ONLY_THE_DECLARATION(stage):
    text = (ROOT / "scripts/GPU/alphazero" / GATE_FILES[stage]).read_text()
    code = [ln for ln in text.split('"""')[-1].splitlines() if ln.strip()]
    assert code == [f"{M.GATES[stage][1]} = False"] and M.gate_open(stage) is False


def test_EVERY_ENTRY_REFUSES_WHILE_ITS_GATE_IS_CLOSED_AND_WRITES_NOTHING(capsys):
    """The cache is compared before and after, not asserted absent: it exists once a run has prepared it."""
    cache_before = M.CACHE.exists()
    assert M.main(["run", "fit"]) == P.EXIT["REFUSED"]
    assert XM.main(["run", "export"]) == P.EXIT["REFUSED"]
    err = capsys.readouterr().err
    assert "HTB_MIX_FIT_AUTHORIZED is closed" in err and "HTB_MIX_EXPORT_AUTHORIZED is closed" in err
    assert not (ROOT / M.EV["export"]).exists()
    assert M.CACHE.exists() == cache_before


# ─────────────────────────────── openings and seeds (card §9) ───────────────────────────────

def test_THE_COMMITTED_OPENINGS_RE_DERIVE_BYTE_EQUAL_AND_ARE_PINNED():
    raw = (ROOT / M.OPENINGS_REL).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == M.OPENINGS_SHA256
    assert M.dump_openings(M.build_openings()) == raw


def test_NO_OPENING_REPEATS_OR_MEETS_A_BENCHMARK_G3_OR_H_OP_OPENING_AND_NOTHING_WAS_SKIPPED():
    o = json.loads((ROOT / M.OPENINGS_REL).read_text())
    assert len(o["match"]) == 400 and len(o["match_smoke"]) == 2
    forbidden = M.forbidden_openings()
    assert len(forbidden) == 201 + 222 + 6_814 and o["excluded"]["h_op_openings"] == 6_814
    assert H.opening_problems({"match": o["match"], "match_smoke": o["match_smoke"]}, forbidden) == []
    assert [e["index"] for e in o["match"]] == list(range(400))          # the exclusions hit nothing
    assert all(e["moves"] == [list(m) for m in H.opening_at(M.NS, "match opening", e["index"])] for e in o["match"][:20])
    probs = H.opening_problems({"match": o["match"][:3]}, [o["match"][1]["moves"]])
    assert probs and "match[1]" in probs[0]                               # the check catches a forbidden opening


def test_AN_H_OP_OPENING_IS_EXCLUDED_DIRECTLY_AND_REFLECTED(monkeypatch):
    first = H.opening_at(M.NS, "match opening", 0)
    monkeypatch.setattr(M, "op_openings", lambda root=M.ROOT: [[list(H.reflect_rc(m)) for m in first]])
    o = M.build_openings(n_match=3, n_smoke=1)
    assert o["match"][0]["index"] == 1


def test_SEEDS_ARE_UNIQUE_AND_THE_DRIVERS_FORMULA():
    s = M.seed_uniqueness()
    assert s["derivations"] == 402 * 2 * 572 == s["distinct_pairs"] and s["repeated_values"] == 0
    assert next(iter(M.seed_strings())) == "htb-mix/20261118/match/pair 0/game 0/ply 4"


def test_THE_OPENINGS_COMMAND_RE_DERIVES_AND_CHECKS(capsys):
    assert M.main(["openings"]) == 0
    assert "re-derives byte-equal" in capsys.readouterr().out


# ─────────────────────────────── the mix selection (card §5, §6) ───────────────────────────────

def _keys(n_op=8, n_opv=3, n_tb=10, n_tbv=3):
    k = lambda p, n: [hashlib.sha256(f"{p}{i}".encode()).hexdigest() for i in range(n)]      # noqa: E731
    return {"op_train": k("o", n_op), "op_val": k("ov", n_opv), "tb_train": k("t", n_tb), "tb_val": k("tv", n_tbv)}


def _ex(*keys):
    return H.Exclusions(set(keys), set(), set(), set())


def test_THE_MIX_SELECTION_DROPS_E1_E3_FROM_H_TB_AND_FIXES_BOTH_ORDERS():
    keys = _keys()
    drop = keys["tb_train"][3]
    sel = M.mix_selection(keys, _ex(drop), n_op=8, n_tb=8, tb_pool=9)
    assert sorted(sel["op_order"]) == list(range(8)) and len(sel["tb_order"]) == 8 and 3 not in sel["tb_order"]
    want_op = sorted(range(8), key=lambda i: hashlib.sha256(f"htb-mix-fit/1|op|{keys['op_train'][i]}".encode()).hexdigest())
    want_tb = sorted([i for i in range(10) if i != 3],
                     key=lambda i: hashlib.sha256(f"htb-mix-fit/1|tb|{keys['tb_train'][i]}".encode()).hexdigest())[:8]
    assert sel["op_order"] == want_op and sel["tb_order"] == want_tb
    r = sel["record"]
    assert r["tb_dropped"] == 1 and r["tb_pool"] == 9 and not any(r["overlap"].values()) and len(r["overlap"]) == 6
    assert sel == M.mix_selection(keys, _ex(drop), n_op=8, n_tb=8, tb_pool=9)            # deterministic


@pytest.mark.parametrize("a, b", [("op_train", "op_val"), ("op_train", "tb_train"), ("op_train", "tb_val"),
                                  ("op_val", "tb_train"), ("op_val", "tb_val"), ("tb_train", "tb_val")])
def test_ANY_SHARED_KEY_BETWEEN_TWO_SETS_STOPS(a, b):
    keys = _keys()
    keys[b] = [keys[a][0], *keys[b][1:]]
    with pytest.raises(P.Stop, match=f"{a}&{b}"):
        M.mix_selection(keys, _ex(), n_op=8, n_tb=8, tb_pool=10)


def test_A_MIRRORED_POSITION_IS_THE_SAME_KEY():
    moves = TP.random_game(5, 20)
    s = H.replay(moves)[-1]
    r = H.replay([H.reflect_rc(m) for m in moves])[-1]
    assert P.state_key(s) == P.state_key(r)                    # so a mirrored copy across sets is an overlap STOP


@pytest.mark.parametrize("where", ["op_train", "op_val", "tb_val"])
def test_AN_E1_E3_KEY_OUTSIDE_THE_H_TB_TRAINING_ROWS_STOPS(where):
    keys = _keys()
    with pytest.raises(P.Stop, match="E1-E3 keys outside"):
        M.mix_selection(keys, _ex(keys[where][0]), n_op=8, n_tb=8, tb_pool=10)


def test_A_REPEATED_KEY_OR_THE_WRONG_COUNTS_STOP():
    keys = _keys()
    bad = {**keys, "tb_val": [keys["tb_val"][0]] * 2}
    with pytest.raises(P.Stop, match="tb_val repeats a key"):
        M.mix_selection(bad, _ex(), n_op=8, n_tb=8, tb_pool=10)
    with pytest.raises(P.Stop, match="H-TB pool is 10 rows"):
        M.mix_selection(keys, _ex(), n_op=8, n_tb=8, tb_pool=9)
    with pytest.raises(P.Stop, match="8 H-OP training rows"):
        M.mix_selection(keys, _ex(), n_op=16, n_tb=8, tb_pool=10)
    with pytest.raises(P.Stop, match="smaller than"):
        M.mix_selection(keys, _ex(), n_op=8, n_tb=11, tb_pool=10)


@pytest.mark.skipif(not all((pathlib.Path(s["cache"]) / "labels-train" / "manifest.json").exists()
                            for s in M.SOURCES.values()), reason="the cached label sets are not on this machine")
def test_THE_REAL_CACHED_SETS_ARE_REUSABLE_AS_WRITTEN_AND_THE_MIX_SELECTION_HOLDS():
    """Read-only: no model. Card §4-§5 on the real files: every manifest, shard and row list verified, the rows the
    committed selections' in order, zero key overlap among the four sets, 11 E1-E3 hits dropped -> 239,989."""
    cfg = M.Config()
    rows = M.load_sources(cfg)
    keys = {f"{s}_{a}": [m["key"] for m in rows[s][a][0]] for s in ("op", "tb") for a in ("train", "val")}
    assert {k: len(v) for k, v in keys.items()} == {"op_train": 120_000, "op_val": 12_000, "tb_train": 240_000,
                                                    "tb_val": 24_000}
    sel = M.mix_selection(keys, H.build_exclusions(), M.N_OP, M.N_TB, M.TB_POOL)
    r = sel["record"]
    assert not any(r["overlap"].values()) and r["tb_dropped"] == 11 and r["tb_pool"] == 239_989
    assert r["op"]["n"] == r["tb"]["n"] == 120_000 and len(M.mixed_plan(None, None, sel["op_order"], sel["tb_order"],
                                                                        M.HALF)) == 7_500


# ─────────────────────────────── the mixed fit (card §6) ───────────────────────────────

def test_THE_MIXED_FIT_IS_H_OPS_FIT_BITWISE_GIVEN_H_OPS_OWN_BATCHES():
    ex = TP._examples()[:16]
    # one epoch: H-MX's plan uses each row once (fit_mixed refuses a repeat), so the loop is compared over one pass
    recipe = P.Recipe(epochs=1, rows_per_batch=4, learning_rate=1e-2)
    ref = TP.tiny_net()
    a, b = TP.twin(ref), TP.twin(ref)
    ra = H.fit(a, ref, ex, recipe, tag=M.FIT_TAG)
    rb = M.fit_mixed(b, ref, M.single_plan(P._Built(ex), recipe, M.FIT_TAG), recipe)
    assert ra["student_digest"] == rb["student_digest"] and ra["updates"] == rb["updates"] == 4
    with pytest.raises(P.Stop, match="repeats"):                          # H-OP's two passes are not H-MX's plan
        M.fit_mixed(TP.twin(ref), ref, M.single_plan(P._Built(ex), P.Recipe(epochs=2, rows_per_batch=4), M.FIT_TAG),
                    P.Recipe(epochs=2, rows_per_batch=4))
    assert [x["loss"] for x in ra["log"]] == [x["loss"] for x in rb["log"]]


def test_EACH_MIXED_BATCH_IS_HALF_H_OP_THEN_HALF_H_TB_AND_THE_FIT_MOVES_EVERY_PART():
    ex = TP._examples(n_games=4)
    op, tb = P._Built(ex[:16]), P._Built(ex[16:32])
    plan = M.mixed_plan(op, tb, list(range(16)), list(range(16)), half=4)
    assert len(plan) == 4 and all([(s is op, len(i)) for s, i in b[:1]] == [(True, 4)] and b[1][0] is tb for b in plan)
    ref = TP.tiny_net(1)
    r = M.fit_mixed(TP.twin(ref), ref, plan, P.Recipe(epochs=1, rows_per_batch=8, learning_rate=1e-2))
    assert r["updates"] == 4 and r["rows_per_source_per_batch"] == [4, 4] and r["epoch_tag"] == "htb-mix-fit/1"
    assert r["surface_tensors_changed"] > 0 and r["running_statistics_unchanged"] > 0
    with pytest.raises(P.Stop, match="equal batches"):
        M.mixed_plan(op, tb, list(range(16)), list(range(12)), half=4)


def test_THE_MIXED_FIT_REFUSES_A_REPEATED_ROW_A_RAGGED_BATCH_AND_A_NON_COPY():
    ex = TP._examples(n_games=4)
    op, tb = P._Built(ex[:16]), P._Built(ex[16:32])
    recipe = P.Recipe(epochs=1, rows_per_batch=8, learning_rate=1e-2)
    ref = TP.tiny_net(2)
    with pytest.raises(P.Stop, match="repeats"):
        M.fit_mixed(TP.twin(ref), ref, [[(op, [0, 1, 2, 3]), (tb, [0, 1, 2, 3])], [(op, [3, 4, 5, 6]), (tb, [4, 5, 6, 7])]],
                    recipe)
    with pytest.raises(P.Stop, match="does not hold 8 rows"):
        M.fit_mixed(TP.twin(ref), ref, [[(op, [0, 1, 2]), (tb, [0, 1, 2, 3])]], recipe)
    with pytest.raises(P.Stop, match="exact copy"):
        M.fit_mixed(TP.tiny_net(3), ref, M.mixed_plan(op, tb, list(range(8)), list(range(8)), 4), recipe)
    with pytest.raises(P.Stop, match="same object"):
        M.fit_mixed(ref, ref, M.mixed_plan(op, tb, list(range(8)), list(range(8)), 4), recipe)


def test_THE_START_CHECK_SCORES_THE_FIRST_MIXED_ROWS():
    ex = TP._examples(n_games=4)
    op, tb = P._Built(ex[:16]), P._Built(ex[16:32])
    plan = M.mixed_plan(op, tb, list(range(16)), list(range(16)), half=4)
    ref = TP.tiny_net(4)
    s = M.start_check(ref, plan, n=12)
    first = op.take([0, 1, 2, 3]) + tb.take([0, 1, 2, 3]) + op.take([4, 5, 6, 7])
    assert s == P.score(ref, first) and s["n"] == 12


# ─────────────────────────────── the one-time fit screens (card §7) ───────────────────────────────

S_OP = {"kl": 0.70735, "top1": 0.57608, "value_mse": 0.025782}
S_TB = {"kl": 0.57449, "top1": 0.65142, "value_mse": 0.070317}
C_OK_OP = {"kl": 0.5, "top1": 0.65, "value_mse": 0.02}
C_OK_TB = {"kl": 0.58, "top1": 0.645, "value_mse": 0.075}


def test_A_CHALLENGER_INSIDE_EVERY_BAR_IS_ELIGIBLE_AND_EVERY_MOVEMENT_IS_REPORTED():
    v = M.gate_verdict(C_OK_OP, S_OP, C_OK_TB, S_TB)
    assert v["verdict"] == "ELIGIBLE" and v["held"] == ["M1", "M2", "M3", "M4", "M5", "M6"] and v["failed"] == []
    assert set(v["movement"]) == {"op_val", "tb_val"} and v["movement"]["tb_val"]["kl"]["direction"] == "away from the teacher"
    assert v["bars"]["M4"] == pytest.approx(0.6032145) and v["bars"]["M6"] == pytest.approx(0.08086455)


@pytest.mark.parametrize("which, c_op, c_tb", [
    ("M1", {**C_OK_OP, "kl": 0.90 * 0.70735 * 1.0001}, C_OK_TB),
    ("M2", {**C_OK_OP, "top1": 0.57608}, C_OK_TB),                       # strict
    ("M3", {**C_OK_OP, "value_mse": 0.0257821}, C_OK_TB),
    ("M4", C_OK_OP, {**C_OK_TB, "kl": 1.05 * 0.57449 * 1.0001}),
    ("M5", C_OK_OP, {**C_OK_TB, "top1": 0.65142 - 0.0101}),
    ("M6", C_OK_OP, {**C_OK_TB, "value_mse": 1.15 * 0.070317 * 1.0001}),
])
def test_EACH_SCREEN_FAILS_ON_ITS_OWN(which, c_op, c_tb):
    v = M.gate_verdict(c_op, S_OP, c_tb, S_TB)
    assert v["failed"] == [which] and v["verdict"] == M.GATE_MISSED
    assert M.GATE_MISSED == "STOP: the mixed challenger missed the fit gate"


def test_THE_BARS_ARE_INCLUSIVE_WHERE_THE_CARD_SAYS_AT_MOST_OR_AT_LEAST():
    c_op = {"kl": 0.5 * 0.9, "top1": 0.61, "value_mse": 0.05}
    c_tb = {"kl": 1.0 * 1.05, "top1": 0.6 - 0.010, "value_mse": 0.1 * 1.15}
    v = M.gate_verdict(c_op, {"kl": 0.5, "top1": 0.6, "value_mse": 0.05}, c_tb, {"kl": 1.0, "top1": 0.6, "value_mse": 0.1})
    assert v["verdict"] == "ELIGIBLE"


def test_H_OPS_OWN_CHALLENGER_WOULD_PASS_M1_TO_M3_AND_FAIL_M4_TO_M6():
    """Card §7: the gate H-OP did not have. Entry 69's recorded scores, read from the pinned H-OP fit record."""
    r = M.op_fit_scores(M.Config(committed=H.committed))
    s, c = r["S"], r["H_OP_challenger"]
    v = M.gate_verdict(c["op_val"], s["op_val"], c["tb_val"], s["tb_val"])
    assert v["held"] == ["M1", "M2", "M3"] and v["failed"] == ["M4", "M5", "M6"]


def test_S_MUST_REPRODUCE_H_OPS_RECORDED_SCORES():
    want = {"op_val": {**S_OP, "n": 12_000}, "tb_val": {**S_TB, "n": 24_000}}
    assert M.s_reproduction(want, want)["max_relative_difference"] == 0
    off = {"op_val": dict(want["op_val"]), "tb_val": {**want["tb_val"], "kl": S_TB["kl"] * (1 + 2e-9)}}
    with pytest.raises(P.Stop, match="tb_val kl .* does not reproduce"):
        M.s_reproduction(off, want)
    with pytest.raises(P.Stop, match="scored 11999 op_val rows"):
        M.s_reproduction({**want, "op_val": {**want["op_val"], "n": 11_999}}, want)


def test_SCORING_IS_BITWISE_REPEATABLE_ON_THE_CPU():
    """Card §7: the reproduction check's tolerance stays 1e-9 only if MLX CPU scoring repeats exactly."""
    ex = TP._examples(n_games=3)
    net = TP.tiny_net(5)
    assert P.score(net, ex) == P.score(net, ex) == P.score(TP.twin(net), ex)


# ─────────────────────────────── the fit run, end to end on fakes ───────────────────────────────

def _tree(tmp_path):
    root = tmp_path / "root"
    (root / M.CARD).parent.mkdir(parents=True, exist_ok=True)
    (root / M.CARD).write_bytes((ROOT / M.CARD).read_bytes())
    return root


def _fake_source(root, name, seed0, n_games, val_games, cache):
    """A cached label set as the label stages write it: shards + manifests, a selection and a COMPLETED record."""
    games, rows, seen = {}, {"train": [], "val": []}, set()
    for i in range(n_games):
        gid = f"{name}_{i:03d}"
        moves = TP.random_game(seed0 + i, 40)
        games[gid] = moves
        arm = "val" if i >= n_games - val_games else "train"
        want = [{"ply": k, "key": P.state_key(s)} for k, s in P.positions(moves) if k >= 6 and k % 3 == 0]
        want = [w for w in want if w["key"] not in seen and not seen.add(w["key"])]
        rows[arm] += P.label_game(gid, moves, want, TP.FakeTeacher(smooth=True))
    rows["train"] = rows["train"][:len(rows["train"]) // 4 * 4]           # whole mixed batches of 4 + 4
    man = {}
    for arm in ("train", "val"):
        P.write_label_shards(rows[arm], cache / f"labels-{arm}", shard_rows=17)
        man[arm] = hashlib.sha256((cache / f"labels-{arm}" / "manifest.json").read_bytes()).hexdigest()
    sel = {"rows": {a: [{"game_id": r["game_id"], "ply": r["ply"], "key": r["key"]} for r in rows[a]] for a in rows}}
    sel_rel, rec_rel = f"ev/{name}/selection.json", f"ev/{name}/record.json"
    (root / sel_rel).parent.mkdir(parents=True, exist_ok=True)
    (root / sel_rel).write_text(json.dumps(sel))
    sel_sha = hashlib.sha256((root / sel_rel).read_bytes()).hexdigest()
    counts = {a: len(rows[a]) for a in rows}
    (root / rec_rel).write_text(json.dumps({"status": "COMPLETED", "selection_sha256": sel_sha, "counts": counts,
                                            "labels": [{"arm": a, "manifest_sha256": man[a], "n": counts[a]}
                                                       for a in ("train", "val")]}))
    src = {"label_record": rec_rel, "label_record_sha256": hashlib.sha256((root / rec_rel).read_bytes()).hexdigest(),
           "selection": sel_rel, "selection_sha256": sel_sha, "manifests": man, "counts": counts, "cache": cache}
    return src, games


def _fit_setup(tmp_path, s_perturb=0.0):
    root = _tree(tmp_path)
    op, op_games = _fake_source(root, "op", 300, 8, 2, tmp_path / "op-cache")
    tb, tb_games = _fake_source(root, "tb", 500, 10, 2, tmp_path / "tb-cache")
    n_op = op["counts"]["train"]
    base = TP.tiny_net(7)
    cfg = M.Config(root=root, data_root=root, cache=root / "cache", sources={"op": op, "tb": tb}, n_op=n_op, n_tb=n_op,
                   tb_pool=tb["counts"]["train"], recipe=P.Recipe(epochs=1, rows_per_batch=8, learning_rate=1e-2),
                   start_rows=16, committed=lambda r, rel: (r / rel).exists(), evidence_problems=lambda r, rels: [],
                   head=lambda r: "f" * 40, gate_open=lambda s: True, exclusions=lambda r: H.Exclusions.empty(),
                   tf32_required=False, available_memory=lambda: {"available_bytes": 1 << 40})
    # H-OP's fit record for these fakes: S's scores exactly as the run will compute them (or perturbed)
    rows = M.load_sources(cfg)
    st = {f"{s}_val": P.ExampleStore(*rows[s]["val"], g) for s, g in (("op", op_games), ("tb", tb_games))}
    sv = {a: P.score(TP.twin(base), st[a]) for a in ("op_val", "tb_val")}
    sv["tb_val"] = {**sv["tb_val"], "kl": sv["tb_val"]["kl"] * (1 + s_perturb)}
    rec = {"student_val": sv["op_val"], "challenger_val": sv["op_val"],
           "retention": {"student": sv["tb_val"], "challenger": sv["tb_val"]}}
    (root / M.OP_FIT_RECORD).parent.mkdir(parents=True, exist_ok=True)
    (root / M.OP_FIT_RECORD).write_text(json.dumps(rec))
    cfg.pins = {M.OP_GAMES: "unused", M.OP_FIT_RECORD: hashlib.sha256((root / M.OP_FIT_RECORD).read_bytes()).hexdigest()}
    return cfg, base, {"op": op_games, "tb": tb_games}


def test_THE_FIT_RUN_BINDS_BOTH_SOURCES_REPRODUCES_S_FITS_THE_MIX_SCREENS_AND_SAVES(tmp_path):
    cfg, base, games = _fit_setup(tmp_path)
    saved = {}

    def save(net, path):
        saved["digest"] = P._digest(net)
        return {"path": str(path), "state_digest": saved["digest"]}
    assert M.run_fit(cfg, load_student=lambda: TP.twin(base), games=lambda c: games, save=save) == 0
    rec = json.loads((cfg.ev("fit") / "record.json").read_text())
    assert rec["status"] == "COMPLETED" and rec["kind"] == "htb-mix-fit-record/1" and rec["kill_s"] == 11_700
    assert rec["fit"]["updates"] == cfg.updates == len(range(0, cfg.n_op, 4)) and rec["fit"]["rows_per_source_per_batch"] == [4, 4]
    assert rec["s_reproduction"]["max_relative_difference"] == 0 and rec["start"]["n"] == 16
    assert not any(rec["selection"]["overlap"].values()) and rec["selection"]["tb"]["n"] == cfg.n_tb
    assert rec["verdict"] == rec["gate"]["verdict"] in ("ELIGIBLE", M.GATE_MISSED)
    assert set(rec["gate"]["movement"]) == {"op_val", "tb_val"} and "never gated" in rec["descriptive"]["note"]
    assert rec["challenger"]["state_digest"] == saved["digest"] == rec["fit"]["student_digest"]
    with pytest.raises(P.Refused, match="destination already exists"):
        M.run_fit(cfg, load_student=lambda: TP.twin(base), games=lambda c: games, save=save)


def test_S_THAT_DOES_NOT_REPRODUCE_STOPS_BEFORE_ANY_UPDATE_AND_SAVES_NOTHING(tmp_path):
    cfg, base, games = _fit_setup(tmp_path, s_perturb=1e-6)
    saved = []
    assert M.run_fit(cfg, load_student=lambda: TP.twin(base), games=lambda c: games,
                     save=lambda n, p: saved.append(p)) == P.EXIT["STOP"]
    rec = json.loads((cfg.ev("fit") / "record.json").read_text())
    assert rec["status"] == "STOP" and rec["stop"]["step"] == "reproduction" and "fit" not in rec and saved == []


def test_THE_FIT_REFUSES_UNPINNED_LABELS_BEFORE_LOADING_ANYTHING(tmp_path):
    cfg, base, games = _fit_setup(tmp_path)
    rel = cfg.sources["tb"]["label_record"]
    (cfg.root / rel).write_text((cfg.root / rel).read_text() + " ")
    boom = lambda *a: (_ for _ in ()).throw(AssertionError("loaded"))           # noqa: E731
    with pytest.raises(P.Refused, match="not the pinned file"):
        M.run_fit(cfg, load_student=boom, games=boom, save=boom)
    assert not cfg.ev("fit").exists()


def test_A_TAMPERED_SHARD_OR_REORDERED_ROWS_ARE_REFUSED(tmp_path):
    cfg, base, games = _fit_setup(tmp_path)
    shard = pathlib.Path(cfg.sources["op"]["cache"]) / "labels-val" / "labels-00000.npz"
    raw = bytearray(shard.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    shard.write_bytes(bytes(raw))
    with pytest.raises(P.Refused, match="changed since labelling"):
        M.load_sources(cfg)


# ─────────────────────────────── wrappers (card §11) ───────────────────────────────

@pytest.mark.parametrize("stage", ["fit", "export"])
def test_EACH_WRAPPER_IS_ITS_DERIVATION_WITH_THE_CARDS_CEILING(stage):
    from tests import htb_mix_wrapper_spec as W

    text = W.wrapper_path(stage).read_text()
    assert text == W.derive(stage)
    assert f"TERM_S={M.KILL_S[stage] - 10}\n" in text and "GRACE_S=10\n" in text
    assert f"GATE={M.GATES[stage][1]}\n" in text and f"GATE_REL=scripts/GPU/alphazero/{M.GATES[stage][0]}.py\n" in text
    assert f"RUN_REL={M.EV[stage]}\n" in text and f"OUT_REL={M.OUTER[stage]}\n" in text
    module = "htb_mix_export" if stage == "export" else "htb_mix"
    assert f'-m "$MODULE" run {stage}' in text and f"MODULE=scripts.GPU.alphazero.{module}\n" in text
    assert f"RECORD_KIND=htb-mix-{stage}-record/1\n" in text and "htb-onpolicy" not in text.split("set -u\n", 1)[1]


# ─────────────────────────────── export and parity (card §8) ───────────────────────────────

def _cfg(root):
    return M.Config(root=root, data_root=root, cache=root / "cache", committed=lambda r, rel: (r / rel).exists(),
                    evidence_problems=lambda r, rels: [], head=lambda r: "f" * 40, gate_open=lambda s: True,
                    exclusions=lambda r: H.Exclusions.empty(), tf32_required=False)


def _stage_evidence(cfg, stage, rec_extra=None, **outer_over):
    ev = cfg.root / M.EV[stage]
    ev.mkdir(parents=True, exist_ok=True)
    rec = {"status": "COMPLETED", "card_sha256": M.CARD_SHA256, "pid": 4242, "head": "f" * 40,
           "kill_s": M.KILL_S[stage], **(rec_extra or {})}
    (ev / "record.json").write_text(json.dumps(rec))
    out = cfg.root / M.OUTER[stage]
    out.mkdir(parents=True, exist_ok=True)
    (out / "outer.json").write_text(json.dumps({**dict(H.OUTER_OK), "head": rec["head"], "term_s": M.KILL_S[stage] - 10,
                                                "grace_s": 10, **outer_over}))
    (out / "pgid.txt").write_text(f"{rec['pid']}\n")
    for f in ("idle_check.txt", "bg_check.txt"):
        (out / f).write_text("ok\n")


def _recovery_evidence(cfg, tmp_path, rec_over=None, ch_over=None, **outer_over):
    """A clean, COMPLETED recovery (amendment §6) with a verified challenger file: what the export admits."""
    ch = tmp_path / "challenger.safetensors"
    ch.write_bytes(b"mixed challenger weights")
    ev, out = cfg.root / R.EV, cfg.root / R.OUTER
    ev.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"status": "COMPLETED", "kind": R.RECORD_KIND, "card_sha256": M.CARD_SHA256,
           "amendment_sha256": R.AMENDMENT_SHA256, "pid": 4242, "head": "f" * 40, "kill_s": R.KILL_S,
           "links": {"failed_record_sha256": R.FAILED["record_sha256"]}, "verdict": "ELIGIBLE",
           "fit": {"student_digest": R.TARGET},
           "challenger": {"path": str(ch), "sha256": hashlib.sha256(ch.read_bytes()).hexdigest(), "state_digest": R.TARGET,
                          "reload_state_digest": R.TARGET, "verified": True, **(ch_over or {})}, **(rec_over or {})}
    (ev / "record.json").write_text(json.dumps(rec))
    (out / "outer.json").write_text(json.dumps({**dict(H.OUTER_OK), "head": rec["head"], "term_s": R.KILL_S - 10,
                                                "grace_s": 10, **outer_over}))
    (out / "pgid.txt").write_text(f"{rec['pid']}\n")
    for f in ("idle_check.txt", "bg_check.txt"):
        (out / f).write_text("ok\n")
    return ch


def _fake_runner(fail=None, graph=CURRENT / "model.onnx"):
    calls = []

    def run(name, argv, env, cwd, log):
        calls.append(name)
        log.write_text(f"{name}\n")
        if name == "export":
            out = pathlib.Path(argv[argv.index("--output") + 1])
            out.write_bytes(graph.read_bytes())
            (out.parent / "model.onnx.data").write_bytes(b"new weights")
        for flag, f in (("s_python", "python_side.json"), ("s_node", "node_side.json"), ("v3_python", "v3_python.json"),
                        ("v3_node", "v3_node.json")):
            if name == flag:
                pathlib.Path([a for a in argv if a.endswith(f)][0]).write_text("{}")
        if name in ("s_compare", "v3_compare"):
            pathlib.Path(argv[-1]).write_text(json.dumps({"verdict": "PASS"}))
        return 1 if name == fail else 0
    return run, calls


def _xargs(tmp_path):
    return {"py": pathlib.Path("/py"), "export_root": tmp_path / "export", "scratch": tmp_path / "scratch",
            "versions": lambda r: "test runtime"}


def _current(root):
    (root / "models/78c16a31c3dba37f").mkdir(parents=True)
    for f in ("model.onnx", "manifest.json"):
        (root / "models/78c16a31c3dba37f" / f).write_bytes((CURRENT / f).read_bytes())


def test_THE_EXPORT_QUALIFIES_AN_ELIGIBLE_MIXED_CHALLENGER_WITH_H_OPS_UNMODIFIED_STEPS(tmp_path):
    root = _tree(tmp_path)
    _current(root)
    cfg = _cfg(root)
    _recovery_evidence(cfg, tmp_path)
    run, calls = _fake_runner()
    assert XM.run_export(cfg, runner=run, **_xargs(tmp_path)) == 0
    rec = json.loads((root / M.EV["export"] / "record.json").read_text())
    assert rec["status"] == "COMPLETED" and rec["verdict"] == "PARITY_PASS" and rec["kind"] == "htb-mix-export-record/1"
    assert rec["kill_s"] == 3_600 and rec["card_sha256"] == M.CARD_SHA256
    c = rec["challenger"]
    m = json.loads((pathlib.Path(c["dir"]) / "manifest.json").read_text())
    assert m["model_id"] == c["model_id"] == X.model_id(c["graph_sha256"], c["external_data_sha256"]) != X.CURRENT_ID
    assert "H-MX challenger" in m["description"] and "H-OP" not in m["description"] and "htb_mix_export" in m["provenance_note"]
    assert m["contract"] == json.loads((CURRENT / "manifest.json").read_text())["contract"]
    assert calls == ["export", "canary_python", "canary_node", "s_python", "s_node", "s_compare", "v3_python", "v3_node",
                     "v3_compare"]
    assert [n for n, *_ in X.commands(root, pathlib.Path("/py"), {"path": "p", "sha256": "s", "state_digest": "d"},
                                       tmp_path, tmp_path, tmp_path)] == calls[1:]


@pytest.mark.parametrize("fail", ["canary_python", "s_compare", "v3_compare"])
def test_ANY_FAILED_EXPORT_STEP_STOPS_THERE_WITHOUT_RETRY(tmp_path, fail):
    root = _tree(tmp_path)
    _current(root)
    cfg = _cfg(root)
    _recovery_evidence(cfg, tmp_path)
    run, calls = _fake_runner(fail=fail)
    assert XM.run_export(cfg, runner=run, **_xargs(tmp_path)) == P.EXIT["STOP"]
    rec = json.loads((root / M.EV["export"] / "record.json").read_text())
    assert rec["status"] == "STOP" and rec["stop"]["step"] == fail and calls.count(fail) == 1 and calls[-1] == fail


def test_A_DIFFERENT_GRAPH_STOPS_BEFORE_ANY_PARITY_STEP(tmp_path):
    root = _tree(tmp_path)
    _current(root)
    cfg = _cfg(root)
    _recovery_evidence(cfg, tmp_path)
    run, calls = _fake_runner(graph=ROOT / "tests/fixtures/onnx_shutdown/tiny_add.onnx")
    assert XM.run_export(cfg, runner=run, **_xargs(tmp_path)) == P.EXIT["STOP"]
    assert json.loads((root / M.EV["export"] / "record.json").read_text())["stop"]["step"] == "graph" and calls == ["export"]


def test_THE_EXPORT_REFUSES_A_MISSED_SCREEN_A_CHANGED_CHALLENGER_OR_AN_OUTER_TIMEOUT(tmp_path):
    run, calls = _fake_runner()
    cfg = _cfg(_tree(tmp_path / "a"))
    _recovery_evidence(cfg, tmp_path / "a", rec_over={"verdict": M.GATE_MISSED})
    with pytest.raises(P.Refused, match="only an ELIGIBLE, verified challenger"):
        XM.run_export(cfg, runner=run, **_xargs(tmp_path / "a"))
    cfg = _cfg(_tree(tmp_path / "b"))
    _recovery_evidence(cfg, tmp_path / "b").write_bytes(b"tampered")
    with pytest.raises(P.Refused, match="not the one the recovery saved and verified"):
        XM.run_export(cfg, runner=run, **_xargs(tmp_path / "b"))
    cfg = _cfg(_tree(tmp_path / "c"))
    _recovery_evidence(cfg, tmp_path / "c", inner_exit=124, outcome="TIMEOUT", outer_exit=124)
    with pytest.raises(P.Refused, match=r"htb-mix-recovery-run/outer.json is not clean"):
        XM.run_export(cfg, runner=run, **_xargs(tmp_path / "c"))
    assert calls == []


@pytest.mark.parametrize("fault", ["term_s", "kill_s", "pgid", "head", "bg_check.txt", "survivors", "card"])
def test_THE_FIT_HANDOFF_CHECKS_COMMIT_PID_CLEANUP_CEILING_AND_CARD(tmp_path, fault):
    cfg = _cfg(_tree(tmp_path))
    _stage_evidence(cfg, "fit")
    assert M.stage_admission(cfg, "fit")["head"] == "f" * 40
    out = cfg.root / M.OUTER["fit"]
    if fault == "term_s":
        _stage_evidence(cfg, "fit", term_s=99)
    elif fault == "kill_s":
        _stage_evidence(cfg, "fit", rec_extra={"kill_s": 11_701})
    elif fault == "pgid":
        (out / "pgid.txt").write_text("1\n")
    elif fault == "head":
        _stage_evidence(cfg, "fit", head="e" * 40)
    elif fault == "bg_check.txt":
        (out / fault).unlink()
    elif fault == "survivors":
        _stage_evidence(cfg, "fit", survivors="unconfirmed cleanup: process group 4242")
    else:
        _stage_evidence(cfg, "fit", rec_extra={"card_sha256": H.CARD_SHA256})          # an H-OP record
    want = {"term_s": "ceiling", "kill_s": "ceiling", "pgid": "pgid", "head": "launch commit",
            "bg_check.txt": "absent, uncommitted", "survivors": "not clean", "card": "not a COMPLETED record of this card"}
    with pytest.raises(P.Refused, match=want[fault]):
        M.stage_admission(cfg, "fit")


def test_H_OP_FILES_ARE_NOT_EDITED_BY_THE_BUILD():
    """Card §2 (#200): the H-MX build adds files; H-OP's and H-TB's stay byte-identical to their pinned evidence."""
    pins = {"tests/htb_integration/onpolicy_match_analyse.mjs": "1876706130de56fe81ea44c8ef513fe2802440c0fa3d2c77eafce8ca07e6f7b7"}
    for rel, want in pins.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == want


def test_AN_OVERLAP_FOUND_AT_FIT_START_IS_A_RECORDED_STOP_BEFORE_ANY_MODEL_LOADS(tmp_path):
    cfg, base, games = _fit_setup(tmp_path)
    rows = M.load_sources(cfg)
    hit = rows["tb"]["train"][0][0]["key"]                     # an H-TB training key, made an E1-E3 key
    cfg.exclusions = lambda r: H.Exclusions({hit}, set(), set(), set())
    boom = lambda *a: (_ for _ in ()).throw(AssertionError("loaded"))           # noqa: E731
    assert M.run_fit(cfg, load_student=boom, games=boom, save=boom) == P.EXIT["STOP"]
    rec = json.loads((cfg.ev("fit") / "record.json").read_text())
    assert rec["status"] == "STOP" and rec["stop"]["step"] == "exclusions" and "pool" in rec["stop"]["detail"]


@pytest.mark.parametrize("stage, term", [("match-smoke", 1_790), ("match", 64_790)])
def test_EACH_MATCH_WRAPPER_IS_ITS_DERIVATION_AND_PINS_THE_FROZEN_ANALYSER(stage, term):
    from tests import htb_mix_match_wrapper_spec as W

    text = W.wrapper_path(stage).read_text()
    assert text == W.derive(stage) and f"TERM_S={term}\n" in text
    analyser = hashlib.sha256((ROOT / "tests/htb_integration/mix_match_analyse.mjs").read_bytes()).hexdigest()
    assert f"ANALYSER_SHA256={analyser}\n" in text
    assert "MIX_MATCH" in text and "ONPOLICY_MATCH" not in text.split("set -u\n", 1)[1]
