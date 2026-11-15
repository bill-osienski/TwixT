"""H-OP model-free tests (card docs/superpowers/2026-11-14-htb-onpolicy-distillation-card.md, revision 3).

No checkpoint, no TensorFlow, no GPU: fake evaluators, fake teachers, tiny CPU networks and synthetic smoke games. The
lockstep tests use twixtbot's pure-Python `Game` from the pinned clone and skip only if the clone is absent.
"""
import hashlib
import importlib.util
import json
import math
import pathlib
import random

import mlx.core as mx
import numpy as np
import pytest

from scripts.GPU.alphazero import htb_onpolicy as H
from scripts.GPU.alphazero import htb_pilot as P
from scripts.GPU.alphazero.game.twixt_state import TwixtState
from tests import test_htb_pilot as TP

ROOT = pathlib.Path(__file__).resolve().parents[1]
GATE_FILES = {"gen-smoke": "htb_onpolicy_gen_smoke_authorization.py", "generate": "htb_onpolicy_generate_authorization.py",
              "label": "htb_onpolicy_label_authorization.py", "fit": "htb_onpolicy_fit_authorization.py",
              "export": "htb_onpolicy_export_authorization.py"}


@pytest.fixture(autouse=True)
def _cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


# ─────────────────────────────── card, gates, frozen numbers ───────────────────────────────

def test_THE_CARD_IS_THE_PINNED_REVISION_3():
    raw = (ROOT / H.CARD).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == H.CARD_SHA256
    assert b"(revision 3)" in raw.splitlines()[0]


def test_THE_FROZEN_NUMBERS_ARE_THE_CARDS():
    assert (H.N_TRAIN, H.N_VAL, H.STOP, H.G_MAX, H.SMOKE_GAMES) == (120_000, 12_000, {"train": 138_000, "val": 13_800},
                                                                    4_000, 40)
    assert (H.GEN_CAP_S, H.SMOKE_CAP_S, H.TRUNC, H.Z_ONE_SIDED, H.F_MIN, H.MIN_SMOKE_PLIES) == \
        (144_000, 3_600, 150, 1.645, 0.90, 800)
    assert H.RECIPE == P.Recipe() and H.UPDATES == 7_500 and H.I1_FACTOR == 0.90 and H.SIMS == 200
    assert (H.SELECT_TAG, H.FIT_TAG) == ("htb-onpolicy-select/1", "htb-onpolicy-fit/1")
    assert H.LABEL_MEMORY_REQUIRED == 5_905_580_032 and H.FIT_MEMORY_REQUIRED == 12 * (1 << 30)
    assert H.STUDENT["sha256"].startswith("0e1cb621") and H.STUDENT["state_digest"].startswith("586dca4f")


@pytest.mark.parametrize("stage", sorted(GATE_FILES))
def test_EVERY_GATE_IS_CLOSED_AND_ITS_FILE_HOLDS_ONLY_THE_DECLARATION(stage):
    assert H.gate_open(stage) is False
    src = (ROOT / "scripts/GPU/alphazero" / GATE_FILES[stage]).read_text()
    code = [ln for ln in src.split('"""')[-1].splitlines() if ln.strip()]
    assert code == [f"{H.GATES[stage][1]} = False"]


@pytest.mark.parametrize("argv", [["run", "gen-smoke"], ["run", "generate"], ["run", "label", "--clone-root", "x"],
                                  ["run", "fit"]])
def test_EVERY_ENTRY_REFUSES_WHILE_ITS_GATE_IS_CLOSED_AND_WRITES_NOTHING(argv, capsys):
    before = {p for p in (ROOT / "docs/superpowers/evidence").iterdir() if p.name.startswith("htb-onpolicy")}
    assert H.main(argv) == P.EXIT["REFUSED"]
    assert "AUTHORIZED is closed" in capsys.readouterr().err
    after = {p for p in (ROOT / "docs/superpowers/evidence").iterdir() if p.name.startswith("htb-onpolicy")}
    assert after == before and not H.CACHE.exists()


# ─────────────────────────────── openings, seeds, arms ───────────────────────────────

def test_THE_PORT_REPRODUCES_THE_BENCHMARKS_OPENINGS_FROM_ITS_COMMITTED_GAMES():
    ours = {o["index"]: o["moves"] for o in H.opening_list(H.BENCH_NS, "opening", 200)}
    seen = {}
    for f in (ROOT / "docs/superpowers/evidence/htb-teacher-bench-match/games").glob("*.json"):
        g = json.loads(f.read_text())
        seen[g["opening_index"]] = g["opening"]
    assert seen == ours and len(ours) == 200
    smoke = json.loads(next((ROOT / "docs/superpowers/evidence/htb-teacher-bench-smoke/games").glob("*.json")).read_text())
    assert H.opening_list(H.BENCH_NS, "smoke opening", 1)[0]["moves"] == smoke["opening"]


def test_THE_COMMITTED_OPENINGS_RE_DERIVE_BYTE_EQUAL_AND_ARE_PINNED():
    raw = (ROOT / H.OPENINGS_REL).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == H.OPENINGS_SHA256
    assert (json.dumps(H.build_openings(), sort_keys=True, indent=1) + "\n").encode() == raw


def test_NO_OPENING_REPEATS_ACROSS_OR_WITHIN_LISTS_OR_MEETS_A_BENCHMARK_OR_G3_OPENING():
    o = json.loads((ROOT / H.OPENINGS_REL).read_text())
    assert [len(o[k]) for k in ("gen", "gen_smoke", "match", "match_smoke")] == [4000, 40, 272, 2]
    forbidden = {H._seq(m) for m in H.bench_openings() + H.g3_openings()}
    assert len(H.g3_openings()) == 222 and len(H.bench_openings()) == 201
    seen = set()
    for k in ("gen", "gen_smoke", "match", "match_smoke"):
        for e in o[k]:
            d, r = H._seq(e["moves"]), H._reflected(e["moves"])
            assert d not in forbidden and r not in forbidden and d not in seen and r not in seen, (k, e)
            seen.add(d)
            assert all(H.CENTRAL_MIN <= a <= H.CENTRAL_MAX and H.CENTRAL_MIN <= b <= H.CENTRAL_MAX for a, b in e["moves"])


def test_AN_EXCLUDED_OPENING_IS_SKIPPED_DIRECTLY_AND_REFLECTED():
    first = H.opening_at(H.NS, "gen opening", 0)
    assert H.opening_list(H.NS, "gen opening", 1, [first])[0]["index"] == 1
    assert H.opening_list(H.NS, "gen opening", 1, [list(H._reflected(first))])[0]["index"] == 1


def test_SEEDS_ARE_THE_FORMULA_48_BIT_AND_DISTINCT():
    h = hashlib.sha256(f"{H.NS}/gen/game 7/ply 12".encode()).hexdigest()
    assert H.move_seeds("gen", 7, 12) == {"search": int(h[:12], 16), "readout": int(h[12:24], 16)}
    seeds = [H.move_seeds(p, g, k)[s] for p in ("gen", "gen smoke") for g in range(40) for k in range(4, 600)
             for s in ("search", "readout")]
    assert len(set(seeds)) == len(seeds) and max(seeds) < 2 ** 48


def test_THE_RUN_SPLITS_9_TO_1_AND_THE_SMOKE_ALTERNATES_20_20():
    assert [H.run_arm(i) for i in range(20)].count("val") == 2 and H.run_arm(9) == H.run_arm(19) == "val"
    assert [H.smoke_arm(j) for j in range(40)] == ["train", "val"] * 20
    assert (H.n_tr(2555), H.n_val(2555)) == (2300, 255)


def test_GAMES_FOR_IS_THE_SMALLEST_G_UNDER_THE_9_TO_1_SPLIT():
    for k_target in range(1, 400):
        for u in (1.0, 3.0, 7.5):
            g = H.games_for("train", k_target, u)
            need = math.ceil(k_target / u)
            assert H.n_tr(g) >= need and H.n_tr(g - 1) < need
            gv = H.games_for("val", k_target, u)
            assert H.n_val(gv) >= need and H.n_val(gv - 1) < need
    assert H.games_for("val", 10, 0.0) == math.inf and H.games_for("train", 10, -1.0) == math.inf


# ─────────────────────────────── the smoke's yield verdict (card §4.4) ───────────────────────────────

def _rows(arms_x):
    return [{"index": j, "arm": a, "x": x, "y": x} for j, (a, x) in enumerate(arms_x)]


def test_REGRESSION_36_4_FORMULA_TEST_RETURNS_REVIEW_ON_VALIDATION():
    v = H.yield_verdict(_rows([("train", 60)] * 36 + [("val", 20)] * 4), t0=10, w=1, complete=True)
    assert v["pools"]["train"]["u"] == 60 and v["pools"]["val"]["u"] == 20
    assert v["games_to_stop_targets"]["by_pool"] == {"train": 2555, "val": 6900}
    assert v["games_to_stop_targets"]["G_stop"] == 6900 and v["games_to_stop_targets"]["set_by"] == "val"
    assert v["sigma"] == pytest.approx(12.153, abs=5e-4)
    assert v["pools"]["val"]["u_lb"] == pytest.approx(10.004, abs=5e-4)
    assert v["games_to_exact_counts"]["G_N"] == 12000 and v["games_to_exact_counts"]["set_by"] == "val"
    assert v["T"] == 6910 and v["failed"] == ["Y3", "Y4"] and v["verdict"] == "REVIEW"


def _smoke_games(lengths):
    """Synthetic smoke games, alternating 20/20, every searched position a distinct key."""
    return [{"index": j, "arm": H.smoke_arm(j), "keys": [f"k{j}-{p}" for p in range(n)]} for j, n in enumerate(lengths)]


def test_FIXTURE_20_20_THE_60_20_EXAMPLE_IN_THE_ACTUAL_SMOKE_RETURNS_REVIEW():
    rows = H.smoke_rows(_smoke_games([60 if j % 2 == 0 else 20 for j in range(40)]), H.Exclusions.empty())
    v = H.yield_verdict(rows, t0=10, w=1, complete=True)
    assert [v["pools"][a]["games"] for a in ("train", "val")] == [20, 20]
    assert v["pools"]["train"]["u"] == 60 and v["pools"]["val"]["u"] == 20
    assert v["games_to_stop_targets"]["G_stop"] == 6900 and v["games_to_stop_targets"]["set_by"] == "val"
    assert v["sigma"] == pytest.approx(20.255, abs=5e-4)
    assert v["pools"]["val"]["u_lb"] == pytest.approx(12.550, abs=5e-4)
    assert v["games_to_exact_counts"]["G_N"] == 9570 and v["games_to_exact_counts"]["set_by"] == "val"
    assert v["failed"] == ["Y3", "Y4"] and v["verdict"] == "REVIEW"


def test_FIXTURE_20_20_ALL_GAMES_AT_60_PLIES_IS_LAUNCH_OK():
    rows = H.smoke_rows(_smoke_games([60] * 40), H.Exclusions.empty())
    v = H.yield_verdict(rows, t0=60, w=33, complete=True)
    assert v["games_to_stop_targets"]["G_stop"] == 2555 and v["games_to_stop_targets"]["set_by"] == "train"
    assert v["games_to_exact_counts"]["G_N"] == 2222 and v["games_to_exact_counts"]["set_by"] == "train"
    assert v["T"] == 84_375 and v["launch_estimate"] == pytest.approx(126_562.5)
    assert v["failed"] == [] and v["verdict"] == "LAUNCH_OK"


def test_Y1_Y2_Y5_EACH_FAIL_ON_THEIR_OWN():
    rows = H.smoke_rows(_smoke_games([60] * 40), H.Exclusions.empty())
    assert H.yield_verdict(rows, 60, 33, complete=False)["failed"] == ["Y1"]
    assert H.yield_verdict(rows, 60, 60, complete=True)["failed"] == ["Y5"]
    short = H.smoke_rows(_smoke_games([19] * 40), H.Exclusions.empty())
    assert "Y1" in H.yield_verdict(short, 1, 1, True)["failed"]                 # 760 searched plies < 800
    dup = [{"index": j, "arm": H.smoke_arm(j), "keys": [f"k{j}-{p % 50}" for p in range(60)]} for j in range(40)]
    v = H.yield_verdict(H.smoke_rows(dup, H.Exclusions.empty()), 60, 1, True)
    assert v["pools"]["train"]["f"] == pytest.approx(50 / 60) and "Y2" in v["failed"]


def test_E4_COUNTS_FOR_TRAINING_BUT_NOT_FOR_VALIDATION():
    games = _smoke_games([60] * 4)
    ex = H.Exclusions(set(), set(), set(), {"k0-0", "k0-1", "k1-0", "k1-1"})
    rows = {r["index"]: r["y"] for r in H.smoke_rows(games, ex)}
    assert rows[0] == 60 and rows[1] == 58                                        # game 0 train, game 1 val
    e1 = H.Exclusions({"k0-0", "k1-0"}, set(), set(), set())
    rows = {r["index"]: r["y"] for r in H.smoke_rows(games, e1)}
    assert rows[0] == 59 and rows[1] == 59
    c = H.PoolCounter(ex)
    assert c.add("train", games[0]["keys"]) == {"train": 60, "val": 0}
    assert c.add("val", games[1]["keys"]) == {"train": 60, "val": 58}


def test_A_VALIDATION_KEY_FOUND_IN_A_LATER_TRAINING_GAME_IS_REMOVED():
    games = [{"index": 0, "arm": "train", "keys": ["a", "b"]}, {"index": 1, "arm": "val", "keys": ["c", "d", "e"]},
             {"index": 2, "arm": "train", "keys": ["d", "f"]}]
    rows = {r["index"]: r["y"] for r in H.smoke_rows(games, H.Exclusions.empty())}
    assert rows == {0: 2, 1: 2, 2: 2}                                             # d left validation
    c = H.PoolCounter(H.Exclusions.empty())
    c.add("train", ["a", "b"])
    assert c.add("val", ["c", "d", "e"])["val"] == 3
    assert c.add("train", ["d", "f"]) == {"train": 4, "val": 2}                  # the run's counter drops it too
    assert c.add("val", ["d", "a", "g"])["val"] == 3                             # never counted again


def test_THE_STOP_RULE_NEEDS_BOTH_TARGETS(monkeypatch):
    monkeypatch.setattr(H, "STOP", {"train": 3, "val": 2})
    c = H.PoolCounter(H.Exclusions.empty())
    c.add("train", ["a", "b", "c"])
    assert not c.reached()
    c.add("val", ["x", "a"])
    assert not c.reached()
    c.add("val", ["y"])
    assert c.reached()


def test_THE_YIELD_RECORD_SERIALIZES_INFINITE_COUNTS():
    rows = H.smoke_rows([{"index": 0, "arm": "train", "keys": ["a"]}, {"index": 1, "arm": "val", "keys": ["a"]}],
                        H.Exclusions.empty())
    v = H.yield_verdict(rows, 1, 1, True)
    assert v["games_to_exact_counts"]["G_N"] == math.inf
    assert json.loads(H.dump_yield(v))["games_to_exact_counts"]["G_N"] == "inf"


# ─────────────────────────────── selection (card §5) ───────────────────────────────

def _gen_games(n, plies=30, seed0=100):
    out = []
    for i in range(n):
        opening = H.opening_at(H.NS, "gen opening", i)
        s = H.replay(opening)[-1]
        moves = [list(m) for m in opening]
        rng = random.Random(seed0 + i)
        while len(moves) < plies and not H.natural_terminal(s):
            m = sorted(s.legal_moves())[rng.randrange(len(s.legal_moves()))]
            s = s.apply_move(m)
            moves.append(list(m))
        out.append({"index": i, "arm": H.run_arm(i), "moves": moves})
    return out


def test_SELECTION_KEEPS_GAMES_SEPARATE_EXCLUDES_AND_IS_DETERMINISTIC():
    games = _gen_games(20)
    games[9]["moves"] = games[3]["moves"]                     # a validation game repeating a training game
    k_train = H.searched_keys(games[0]["moves"])
    ex = H.Exclusions({k_train[0]}, set(), set(), {H.searched_keys(games[19]["moves"])[2]})
    sel = H.select(games, ex, n_train=200, n_val=20)
    assert sel == H.select(games, ex, n_train=200, n_val=20)
    train = {r["key"] for r in sel["rows"]["train"]}
    val = {r["key"] for r in sel["rows"]["val"]}
    assert not train & val and k_train[0] not in train
    assert sel["exclusion_hits"]["val"]["in_training_games"] >= 26 and sel["exclusion_hits"]["val"]["E4"] == 1
    for r in sel["rows"]["train"] + sel["rows"]["val"]:
        g = games[int(r["game_id"].split("_")[1])]
        assert H.run_arm(g["index"]) == ("train" if r in sel["rows"]["train"] else "val")
        assert P.state_key(H.replay(g["moves"])[r["ply"]]) == r["key"] and r["ply"] >= 4
    with pytest.raises(P.Refused, match="no fallback"):
        H.select(games, ex, n_train=10_000, n_val=20)
    games[4]["arm"] = "val"
    with pytest.raises(P.Stop, match="the rule says"):
        H.select(games, ex, n_train=10, n_val=2)


# ─────────────────────────────── the generation move: G1's fixtures ───────────────────────────────

def _fixtures():
    spec = importlib.util.spec_from_file_location("g1_fixtures", ROOT / "tests/fixtures/htb_integration/generate_fixtures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, json.loads((ROOT / "tests/fixtures/htb_integration/search.json").read_text())


def test_THE_GENERATION_MOVE_REPRODUCES_G1S_FIXTURES_SEED_FOR_SEED():
    """The driver's move function, against the fixtures G1 used to tie protocolMove to the research code."""
    from scripts.GPU.alphazero.mirror_averaging import MirrorAveragingEvaluator

    mod, fx = _fixtures()
    fake = mod.FakeEvaluator()
    evs = {"fake": fake, "mirror_fake": MirrorAveragingEvaluator(fake)}
    assert len(fx["cases"]) == 27
    for c in fx["cases"]:
        s = mod.state_from(c["moves"])
        r = H.research_move(evs[c["evaluator"]], s, {"search": c["search_seed"], "readout": c["readout_seed"] or 0},
                            n_sims=c["n_sims"])
        want = c["result"]["readout_move"] if c["kind"] == "plain" else c["result"]["choice"]
        assert f"{r['move'][0]},{r['move'][1]}" == want, c["id"]
        assert r["halving"] is (c["kind"] == "halving") and r["visits"] == c["n_sims"] == r["n_sims"], c["id"]


# ─────────────────────────────── the generation loop (fake seams) ───────────────────────────────

class _Net:
    pass


class _Base:
    def __init__(self):
        self.network = _Net()


def _seams(move=None, digest=None, bg_ok=True):
    def fake_move(ev, s, seeds):
        legal = sorted(s.legal_moves())
        return {"move": legal[seeds["search"] % len(legal)], "visits": 200, "n_sims": 200, "halving": s.ply >= 20}
    t = {"now": 0.0}

    def clock():
        t["now"] += 0.5
        return t["now"]
    return H.GenSeams(versions=lambda: dict(H.VERSIONS), student_sha256=lambda: H.STUDENT["sha256"], load=_Base,
                      digest=digest or (lambda n: H.STUDENT["state_digest"]), require_eval=lambda ev, w: None,
                      background=lambda: {"ok": bg_ok}, wrap=lambda b: b, move=move or fake_move, clock=clock)


def _run_generate(tmp_path, stage, seams, ex=None, monkeypatch=None):
    out = tmp_path / stage
    out.mkdir()
    rec, events = {}, []
    o = json.loads((ROOT / H.OPENINGS_REL).read_text())
    H.generate(stage, seams, out, rec, lambda e, **k: events.append((e, k)), ex or H.Exclusions.empty(),
               o["gen_smoke" if stage == "gen-smoke" else "gen"], t_start=0.0)
    return rec, H.read_games(out / "games.jsonl"), events


def test_THE_SMOKE_PLAYS_40_ALTERNATING_GAMES_TO_THE_PRODUCTS_END_AND_TIMES_THE_WHOLE_LOOP(tmp_path):
    rec, games, events = _run_generate(tmp_path, "gen-smoke", _seams())
    assert rec["games"] == 40 and len(games) == 40 and rec["stopped_by"] == "smoke"
    o = json.loads((ROOT / H.OPENINGS_REL).read_text())["gen_smoke"]
    assert H.check_games(games, o, H.smoke_arm) == []
    assert [g["arm"] for g in games] == ["train", "val"] * 20
    assert rec["startup_s"] > 0 and rec["w"] == rec["loop_seconds"] / 40
    assert rec["background_checks"] == 41 and sum(e == "game_end" for e, _ in events) == 40


def test_THE_RUN_STOPS_ON_BOTH_POSITION_COUNTS(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "STOP", {"train": 300, "val": 30})
    rec, games, _ = _run_generate(tmp_path, "generate", _seams())
    assert rec["stopped_by"] == "targets" and rec["counts"]["train"] >= 300 and rec["counts"]["val"] >= 30
    prev = games[-2]["counts_after"]
    assert prev["train"] < 300 or prev["val"] < 30                     # it stopped at the FIRST game reaching both


def test_THE_RUN_STOPS_AT_G_MAX(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "G_MAX", 12)
    rec, games, _ = _run_generate(tmp_path, "generate", _seams())
    assert rec["stopped_by"] == "g_max" and rec["games"] == 12 and [g["arm"] for g in games].count("val") == 1


@pytest.mark.parametrize("bad, step", [({"visits": 199}, "budget"), ({"n_sims": 64, "visits": 64}, "budget"),
                                       ({"move": (0, 0)}, "illegal")])
def test_A_SHORT_BUDGET_OR_AN_ILLEGAL_MOVE_STOPS(tmp_path, bad, step):
    def move(ev, s, seeds):
        legal = sorted(s.legal_moves())
        return {"move": legal[0], "visits": 200, "n_sims": 200, "halving": False, **bad}
    with pytest.raises(P.Stop) as e:
        _run_generate(tmp_path, "gen-smoke", _seams(move=move))
    assert e.value.step == step


def test_A_CHANGED_STUDENT_OR_BOINC_STOPS(tmp_path):
    with pytest.raises(P.Stop, match="state digest"):
        _run_generate(tmp_path, "gen-smoke", _seams(digest=lambda n: "0" * 64))
    (tmp_path / "b").mkdir()
    with pytest.raises(P.Stop, match="BOINC"):
        _run_generate(tmp_path / "b", "gen-smoke", _seams(bg_ok=False))


def test_REPLAY_VALIDATION_CATCHES_A_TAMPERED_GAME(tmp_path):
    _rec, games, _ = _run_generate(tmp_path, "gen-smoke", _seams())
    o = json.loads((ROOT / H.OPENINGS_REL).read_text())["gen_smoke"]
    games[3]["per_move"] = games[3]["per_move"][:-1]
    games[5]["moves"] = games[5]["moves"][:-1]
    games[6]["arm"] = "val"                                  # game 6 is training-like in the 20/20 smoke
    faults = H.check_games(games, o, H.smoke_arm)
    assert {f.split(":")[0] for f in faults} == {"game 3", "game 5", "game 6"}



# ─────────────────────────────── evidence chain: smoke -> yield -> admission ───────────────────────────────

def _tree(tmp_path):
    """A throwaway root with the card and the openings, every file treated as committed."""
    root = tmp_path / "root"
    for rel in (H.CARD, H.OPENINGS_REL):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes((ROOT / rel).read_bytes())
    return root


def _cfg(root, **kw):
    return H.Config(root=root, data_root=root, cache=root / "cache", committed=lambda r, rel: True,
                    head=lambda r: "f" * 40, gate_open=lambda s: True, exclusions=lambda r: H.Exclusions.empty(),
                    tf32_required=False, **kw)


def _stage_evidence(cfg, stage, rec_extra=None, outer_extra=None):
    ev = cfg.root / H.EV[stage]
    ev.mkdir(parents=True, exist_ok=True)
    rec = {"status": "COMPLETED", "card_sha256": H.CARD_SHA256, "pid": 4242, "head": "f" * 40, **(rec_extra or {})}
    (ev / "record.json").write_text(json.dumps(rec))
    out = cfg.root / H.OUTER[stage]
    out.mkdir(parents=True, exist_ok=True)
    (out / "outer.json").write_text(json.dumps({**dict(H.OUTER_OK), "head": "f" * 40, **(outer_extra or {})}))
    (out / "pgid.txt").write_text("4242\n")


def test_THE_SMOKE_RUNS_THROUGH_THE_GATED_ENTRY_AND_ITS_YIELD_RE_DERIVES(tmp_path):
    root = _tree(tmp_path)
    cfg = _cfg(root)
    assert H.run_gen("gen-smoke", cfg, _seams(), t_start=0.0) == 0
    rec = json.loads((root / H.EV["gen-smoke"] / "record.json").read_text())
    assert rec["status"] == "COMPLETED" and rec["games"] == 40 and rec["head"] == "f" * 40
    with pytest.raises(P.Refused, match="destination already exists"):
        H.run_gen("gen-smoke", cfg, _seams(), t_start=0.0)
    v, raw = H.derive_yield(cfg)
    assert v["complete"] is True and v["faults"] == [] and v["pools"]["train"]["games"] == 20
    assert v["pools"]["train"]["f"] == pytest.approx(1.0, abs=0.05)              # fake games barely repeat
    assert raw == H.derive_yield(cfg)[1]


def test_GENERATION_ADMITS_ONLY_A_CLEAN_SMOKE_WITH_A_BYTE_EQUAL_LAUNCH_OK_YIELD(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    cfg = _cfg(root)
    assert H.run_gen("gen-smoke", cfg, _seams(), t_start=0.0) == 0
    out = root / H.OUTER["gen-smoke"]
    out.mkdir(parents=True)
    rec = json.loads((root / H.EV["gen-smoke"] / "record.json").read_text())
    (out / "outer.json").write_text(json.dumps({**dict(H.OUTER_OK), "head": rec["head"]}))
    (out / "pgid.txt").write_text(f"{rec['pid']}\n")
    v, raw = H.derive_yield(cfg)
    (root / H.YIELD_REL).write_bytes(raw)
    if v["verdict"] == "LAUNCH_OK":
        assert H.generation_admission(cfg, H.Exclusions.empty())["yield_sha256"] == hashlib.sha256(raw).hexdigest()
    else:                                                    # the fake loop's timing can fail Y5: REVIEW refuses
        with pytest.raises(P.Refused, match="no launch"):
            H.generation_admission(cfg, H.Exclusions.empty())
    monkeypatch.setattr(H, "yield_verdict", lambda *a, **k: {**v, "verdict": "LAUNCH_OK", "failed": []})
    (root / H.YIELD_REL).write_bytes(H.derive_yield(cfg)[1])
    assert H.generation_admission(cfg, H.Exclusions.empty())
    (root / H.YIELD_REL).write_bytes(raw + b" ")
    with pytest.raises(P.Refused, match="byte-equal"):
        H.generation_admission(cfg, H.Exclusions.empty())
    (root / H.YIELD_REL).write_bytes(H.derive_yield(cfg)[1])
    (out / "pgid.txt").write_text("1\n")
    with pytest.raises(P.Refused, match="pgid"):
        H.generation_admission(cfg, H.Exclusions.empty())
    (out / "pgid.txt").write_text(f"{rec['pid']}\n")
    (out / "outer.json").write_text(json.dumps({**dict(H.OUTER_OK), "head": rec["head"], "gate_restored": False}))
    with pytest.raises(P.Refused, match="not clean"):
        H.generation_admission(cfg, H.Exclusions.empty())


def test_A_REVIEW_YIELD_REFUSES_THE_RUN(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    cfg = _cfg(root)
    assert H.run_gen("gen-smoke", cfg, _seams(), t_start=0.0) == 0
    rec = json.loads((root / H.EV["gen-smoke"] / "record.json").read_text())
    out = root / H.OUTER["gen-smoke"]
    out.mkdir(parents=True)
    (out / "outer.json").write_text(json.dumps({**dict(H.OUTER_OK), "head": rec["head"]}))
    (out / "pgid.txt").write_text(f"{rec['pid']}\n")
    monkeypatch.setattr(H, "yield_verdict", lambda *a, **k: {"verdict": "REVIEW", "failed": ["Y3", "Y4"]})
    (root / H.YIELD_REL).write_bytes(H.derive_yield(cfg)[1])
    with pytest.raises(P.Refused, match=r"REVIEW, failed \['Y3', 'Y4'\]: no launch"):
        H.run_gen("generate", cfg, _seams(), t_start=0.0)
    assert not (root / H.EV["generate"]).exists()


# ─────────────────────────────── labels (card §6) ───────────────────────────────

CLONE = pathlib.Path("~/twixt-cache/htb-teacher/twixtbot-ui").expanduser()
needs_clone = pytest.mark.skipif(not (CLONE / "src/backend/twixt.py").exists(), reason="pinned twixtbot clone absent")


@needs_clone
def test_THE_LABEL_LOCKSTEP_IS_THE_BENCHMARKS_A_G3_DRAW_LABELS_WHERE_THE_FIRST_CHECK_STOPPED():
    """G3 pair_0057_game_1 (572 plies): twixtbot flags a draw after ply 73. The first fit's unfiltered check would
    STOP labelling there; the H-OP teacher's check labels the whole game and still catches a real divergence."""
    from tests import test_htb_bench_teacher as TB

    twixt = TB._twixt()[0]

    class Teacher(TB.FakeTeacher):
        rt = {"twixt": twixt}
        divergences = H.OnpolicyTeacher.divergences

    class Old(TB.FakeTeacher):
        rt = {"twixt": twixt}
        divergences = P.TwixtbotTeacher.divergences

    moves = json.loads(TB.G3_DRAW.read_text())["moves"]
    states = H.replay(moves)
    rows = [{"ply": k, "key": P.state_key(states[k])} for k in (10, 72, 74, 300, 571)]
    out = P.label_game("g3", moves, rows, Teacher())
    assert [r["ply"] for r in out] == [10, 72, 74, 300, 571]
    with pytest.raises(P.Stop, match="lockstep"):
        P.label_game("g3", moves, rows, Old())
    bad = Teacher(allow_scl=True)                       # a lane that forms a bridge ours forbids
    m = [(10, 10), (5, 5), (12, 11), (5, 8), (10, 11), (5, 12), (12, 10), (6, 6)]
    with pytest.raises(P.Stop, match="lockstep"):
        P.label_game("x", m, [{"ply": 7, "key": P.state_key(H.replay(m)[7])}], bad)


def _generation(cfg, n=30):
    games = _gen_games(n, plies=40)
    ev = cfg.root / H.EV["generate"]
    ev.mkdir(parents=True, exist_ok=True)
    with open(ev / "games.jsonl", "w") as fh:
        for g in games:
            row = {**g, "plies": len(g["moves"]), "searched": len(g["moves"]) - 4,
                   "per_move": [{"ply": k, "visits": 200} for k in range(4, len(g["moves"]))]}
            fh.write(json.dumps(row) + "\n")
    _stage_evidence(cfg, "generate")
    return games


def test_THE_LABEL_RUN_SELECTS_LABELS_AND_BINDS(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    cfg = _cfg(root, n_train=256, n_val=32, shard_rows=100, available_memory=lambda: {"available_bytes": 1 << 40})
    _generation(cfg)
    monkeypatch.setattr(H, "check_games", lambda games, o, arm_of: [])      # synthetic games are not the frozen ones
    teacher = TP.FakeTeacher()
    assert H.run_label("unused", cfg, load_teacher=lambda c: (teacher, {"fake": True})) == 0
    rec = json.loads((root / H.EV["label"] / "record.json").read_text())
    assert rec["status"] == "COMPLETED" and [e["n"] for e in rec["labels"]] == [256, 32]
    assert teacher.calls == 2 * (256 + 32)
    label, sel = H.check_label_binding(cfg)
    rows = H.load_bound_labels(cfg, label, sel)
    assert len(rows["train"][0]) == 256 and rows["val"][1].shape == (32, P.PADDED)
    with pytest.raises(P.Refused, match="destination already exists"):
        H.run_label("unused", cfg, load_teacher=lambda c: (teacher, {}))


def test_THE_LABEL_RUN_REFUSES_WITHOUT_A_CLEAN_GENERATION(tmp_path):
    root = _tree(tmp_path)
    cfg = _cfg(root, available_memory=lambda: {"available_bytes": 1 << 40})
    with pytest.raises(P.Refused, match="absent, uncommitted or changed"):
        H.run_label("x", cfg, load_teacher=lambda c: (_ for _ in ()).throw(AssertionError("loaded")))
    _generation(cfg)
    (root / H.OUTER["generate"] / "outer.json").write_text(json.dumps({**dict(H.OUTER_OK), "outer_exit": 137,
                                                                         "head": "f" * 40}))
    with pytest.raises(P.Refused, match="not clean"):
        H.run_label("x", cfg, load_teacher=lambda c: (_ for _ in ()).throw(AssertionError("loaded")))
    assert not (root / H.EV["label"]).exists()


# ─────────────────────────────── the fit and the gate (card §7, §8) ───────────────────────────────

def test_THE_FIT_COPY_IS_P_FIT_BITWISE_GIVEN_ITS_NAMESPACE_AND_DIFFERS_ONLY_IN_ORDER_WITH_OURS():
    ex = TP._examples()[:16]
    recipe = P.Recipe(epochs=2, rows_per_batch=4, learning_rate=1e-2)
    ref = TP.tiny_net()
    a, b, c = TP.twin(ref), TP.twin(ref), TP.twin(ref)
    ra, rb, rc = P.fit(a, ref, ex, recipe), H.fit(b, ref, ex, recipe, tag=P.FIT_TAG), H.fit(c, ref, ex, recipe)
    assert ra["student_digest"] == rb["student_digest"] and [x["loss"] for x in ra["log"]] == [x["loss"] for x in rb["log"]]
    assert rc["epoch_tag"] == "htb-onpolicy-fit/1" and rc["student_digest"] != ra["student_digest"]
    keys = [e["key"] for e in ex]
    assert H.epoch_order(keys, 0, P.FIT_TAG) == P.epoch_order(keys, 0) != H.epoch_order(keys, 0)


@pytest.mark.parametrize("c, s, want, failed", [
    ({"kl": 0.9, "top1": 0.61, "value_mse": 0.05}, {"kl": 1.0, "top1": 0.6, "value_mse": 0.05}, "ELIGIBLE", []),
    ({"kl": 0.91, "top1": 0.61, "value_mse": 0.04}, {"kl": 1.0, "top1": 0.6, "value_mse": 0.05}, H.GATE_MISSED, ["I1"]),
    ({"kl": 0.5, "top1": 0.6, "value_mse": 0.04}, {"kl": 1.0, "top1": 0.6, "value_mse": 0.05}, H.GATE_MISSED, ["I2"]),
    ({"kl": 0.5, "top1": 0.7, "value_mse": 0.0501}, {"kl": 1.0, "top1": 0.6, "value_mse": 0.05}, H.GATE_MISSED, ["I3"]),
])
def test_THE_GATE_IS_I1_TO_I3_AND_A_MISS_REPORTS_EVERY_MOVEMENT(c, s, want, failed):
    v = H.gate_verdict(c, s)
    assert v["verdict"] == want and v["failed"] == failed
    assert set(v["movement"]) == {"kl", "top1", "value_mse"}
    m = v["movement"]
    assert m["kl"]["C_minus_S"] == pytest.approx(c["kl"] - s["kl"]) and m["kl"]["C_over_S"] == pytest.approx(c["kl"] / s["kl"])
    if c["kl"] < s["kl"]:
        assert m["kl"]["direction"] == "toward the teacher"
    if c["top1"] == s["top1"]:
        assert m["top1"]["direction"] == "unchanged"
    if c["value_mse"] > s["value_mse"]:
        assert m["value_mse"]["direction"] == "away from the teacher"
    assert H.GATE_MISSED == "STOP: the challenger missed the imitation gate"


def test_THE_FIT_RUN_BINDS_LABELS_SCORES_BOTH_AND_SAVES_THE_CHALLENGER(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    cfg = _cfg(root, n_train=64, n_val=16, shard_rows=50, start_rows=32,
               recipe=P.Recipe(epochs=1, rows_per_batch=8, learning_rate=1e-2),
               available_memory=lambda: {"available_bytes": 1 << 40})
    _generation(cfg)
    monkeypatch.setattr(H, "check_games", lambda games, o, arm_of: [])
    assert H.run_label("u", cfg, load_teacher=lambda c: (TP.FakeTeacher(smooth=True), {})) == 0
    base = TP.tiny_net(3)
    saved = {}

    def save(net, path):
        saved["digest"] = P._digest(net)
        return {"path": str(path), "state_digest": saved["digest"]}
    old = TP._examples(n_games=2)[:8]
    assert H.run_fit(cfg, load_student=lambda: TP.twin(base), retention=lambda c: old, save=save) == 0
    rec = json.loads((root / H.EV["fit"] / "record.json").read_text())
    assert rec["status"] == "COMPLETED" and rec["fit"]["updates"] == 8 and rec["epoch_tag"] == "htb-onpolicy-fit/1"
    assert rec["gate"]["verdict"] in ("ELIGIBLE", H.GATE_MISSED) and set(rec["gate"]["movement"]) == {"kl", "top1",
                                                                                                         "value_mse"}
    assert rec["verdict"] == rec["gate"]["verdict"] and rec["challenger"]["state_digest"] == saved["digest"]
    assert rec["retention"]["student"]["n"] == 8 and "descriptive only" in rec["retention"]["note"]
    with pytest.raises(P.Refused, match="destination already exists"):
        H.run_fit(cfg, load_student=lambda: TP.twin(base), retention=lambda c: old, save=save)


def test_THE_FIT_REFUSES_UNBOUND_LABELS_BEFORE_LOADING_ANYTHING(tmp_path):
    root = _tree(tmp_path)
    cfg = _cfg(root, available_memory=lambda: {"available_bytes": 1 << 40})
    boom = lambda *a: (_ for _ in ()).throw(AssertionError("loaded"))           # noqa: E731
    with pytest.raises(P.Refused, match="absent, uncommitted or changed"):
        H.run_fit(cfg, load_student=boom, retention=boom, save=boom)
    assert not (root / H.EV["fit"]).exists()


# ─────────────────────────────── wrappers (card §12) ───────────────────────────────

@pytest.mark.parametrize("stage", ["gen-smoke", "generate", "label", "fit", "export"])
def test_EACH_WRAPPER_IS_ITS_DERIVATION_WITH_THE_CARDS_CEILING(stage):
    from tests import htb_onpolicy_wrapper_spec as W

    text = W.wrapper_path(stage).read_text()
    assert text == W.derive(stage)
    kill = {"gen-smoke": H.SMOKE_CAP_S, "generate": H.GEN_CAP_S, "label": 6_600, "fit": 11_700, "export": 3_600}[stage]
    assert f"TERM_S={kill - 10}\n" in text and f"GATE={H.GATES[stage][1]}\n" in text
    assert f"GATE_REL=scripts/GPU/alphazero/{H.GATES[stage][0]}.py\n" in text
    assert f"RUN_REL={H.EV[stage]}\n" in text and f"OUT_REL={H.OUTER[stage]}\n" in text
    cmd = {"gen-smoke": "run gen-smoke", "generate": "run generate", "label": 'run label --clone-root "$CKPT"',
           "fit": "run fit", "export": "run export"}[stage]
    module = "htb_onpolicy_export" if stage == "export" else "htb_onpolicy"
    assert f'-m "$MODULE" {cmd}' in text and f"MODULE=scripts.GPU.alphazero.{module}\n" in text
    assert ("teacher-venv/bin/python" in text) is (stage == "label")


# ─────────────────────────────── export and parity (card §9) ───────────────────────────────

from scripts.GPU.alphazero import htb_onpolicy_export as X  # noqa: E402

CURRENT = ROOT / "models/78c16a31c3dba37f"


def test_THE_MODEL_ID_IS_THE_PRODUCTS_AND_REPRODUCES_THE_CURRENT_PAIR():
    m = json.loads((CURRENT / "manifest.json").read_text())
    assert X.model_id(m["graph"]["sha256"], m["external_data"]["sha256"]) == "78c16a31c3dba37f" == X.CURRENT_ID


def test_THE_STRUCTURAL_CHECK_SEES_OPS_AND_SHAPES_NOT_WEIGHTS():
    cur = X.graph_signature(CURRENT / "model.onnx")
    assert cur == X.graph_signature(CURRENT / "model.onnx") and len(cur["ops"]) > 20 and cur["initializers"]
    assert X.graph_signature(ROOT / "tests/fixtures/onnx_shutdown/tiny_add.onnx") != cur


def test_THE_MANIFEST_KEEPS_THE_CONTRACT_AND_CHANGES_ONLY_IDENTITY_AND_PROVENANCE():
    tmpl = json.loads((CURRENT / "manifest.json").read_text())
    m = X.build_manifest(tmpl, "0123456789abcdef", {"size_bytes": 1, "sha256": "a" * 64}, {"size_bytes": 2, "sha256": "b" * 64},
                         {"path": "/x/challenger.safetensors", "sha1": "c" * 40, "sha256": "d" * 64, "state_digest": "e" * 64},
                         "f" * 40, "python 3.14.10")
    assert m["contract"] == tmpl["contract"] and m["manifest_version"] == tmpl["manifest_version"]
    assert m["model_id"] == "0123456789abcdef" and m["provenance"]["source_checkpoint_sha256"] == "d" * 64
    assert "--weights /x/challenger.safetensors" in m["provenance"]["exporter_config"]
    assert tmpl["model_id"] == "78c16a31c3dba37f"                                     # the template is untouched


def test_THE_QUALIFICATION_USES_THE_UNMODIFIED_TOOLS_AND_THE_ISOLATED_RUNTIME(tmp_path):
    ch = {"path": "/c.safetensors", "sha256": "s" * 64, "state_digest": "d" * 64}
    cmds = X.commands(ROOT, pathlib.Path("/py"), ch, tmp_path / "pair", tmp_path / "scratch", tmp_path / "ev")
    names = [c[0] for c in cmds]
    assert names == ["canary_python", "canary_node", "s_python", "s_node", "s_compare", "v3_python", "v3_node", "v3_compare"]
    by = {n: a for n, a, _ in cmds}
    assert by["canary_python"][0] == by["s_python"][0] == str(X.ISOLATED_PY)
    assert by["s_python"][1] == "tests/parity/run_python_side.py" and by["s_compare"][1] == "tests/parity/compare.mjs"
    assert by["v3_compare"][1] == "tests/parity/htb_mirror/compare.mjs"                # the reviewed comparator
    assert by["v3_python"][-3:] == ["/c.safetensors", "s" * 64, "d" * 64]
    assert X.export_command(pathlib.Path("/py"), "/c.safetensors", tmp_path)[2:] == [
        "-m", "scripts.GPU.alphazero.export_onnx", "--weights", "/c.safetensors", "--output", str(tmp_path / "model.onnx")]


def test_THE_V3_COPIES_ARE_THEIR_DERIVATION():
    from tests import htb_onpolicy_v3_spec as V

    for name in V.SUBS:
        assert (ROOT / V.DST / name).read_text() == V.derive(name)


def _fit_evidence(cfg, tmp_path, verdict="ELIGIBLE"):
    ch = tmp_path / "challenger.safetensors"
    ch.write_bytes(b"challenger weights")
    _stage_evidence(cfg, "fit", rec_extra={"verdict": verdict, "challenger": {
        "path": str(ch), "sha256": hashlib.sha256(ch.read_bytes()).hexdigest(), "state_digest": "9" * 64}})
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


def test_THE_EXPORT_QUALIFIES_AN_ELIGIBLE_CHALLENGER_AND_RECORDS_WHAT_THE_MATCH_ADMITS(tmp_path):
    root = _tree(tmp_path)
    (root / "models/78c16a31c3dba37f").mkdir(parents=True)
    for f in ("model.onnx", "manifest.json"):
        (root / "models/78c16a31c3dba37f" / f).write_bytes((CURRENT / f).read_bytes())
    cfg = _cfg(root)
    _fit_evidence(cfg, tmp_path)
    run, calls = _fake_runner()
    assert X.run_export(cfg, runner=run, **_xargs(tmp_path)) == 0
    rec = json.loads((root / H.EV["export"] / "record.json").read_text())
    assert rec["status"] == "COMPLETED" and rec["verdict"] == "PARITY_PASS" and rec["kind"] == X.RECORD_KIND
    c = rec["challenger"]
    assert c["model_id"] != X.CURRENT_ID and c["dir"] == str(tmp_path / "export" / c["model_id"])
    m = json.loads((pathlib.Path(c["dir"]) / "manifest.json").read_text())
    assert m["model_id"] == c["model_id"] == X.model_id(c["graph_sha256"], c["external_data_sha256"])
    assert calls == ["export", "canary_python", "canary_node", "s_python", "s_node", "s_compare", "v3_python", "v3_node",
                     "v3_compare"]
    assert (root / H.EV["export"] / "v3_node.json.gz").exists()


@pytest.mark.parametrize("fail", ["canary_python", "s_compare", "v3_compare"])
def test_ANY_FAILED_STEP_STOPS_THERE_WITHOUT_RETRY(tmp_path, fail):
    root = _tree(tmp_path)
    (root / "models/78c16a31c3dba37f").mkdir(parents=True)
    for f in ("model.onnx", "manifest.json"):
        (root / "models/78c16a31c3dba37f" / f).write_bytes((CURRENT / f).read_bytes())
    cfg = _cfg(root)
    _fit_evidence(cfg, tmp_path)
    run, calls = _fake_runner(fail=fail)
    assert X.run_export(cfg, runner=run, **_xargs(tmp_path)) == P.EXIT["STOP"]
    rec = json.loads((root / H.EV["export"] / "record.json").read_text())
    assert rec["status"] == "STOP" and rec["stop"]["step"] == fail and "verdict" not in rec
    assert calls.count(fail) == 1 and calls[-1] == fail


def test_A_DIFFERENT_GRAPH_STOPS_BEFORE_ANY_PARITY_STEP(tmp_path):
    root = _tree(tmp_path)
    (root / "models/78c16a31c3dba37f").mkdir(parents=True)
    for f in ("model.onnx", "manifest.json"):
        (root / "models/78c16a31c3dba37f" / f).write_bytes((CURRENT / f).read_bytes())
    cfg = _cfg(root)
    _fit_evidence(cfg, tmp_path)
    run, calls = _fake_runner(graph=ROOT / "tests/fixtures/onnx_shutdown/tiny_add.onnx")
    assert X.run_export(cfg, runner=run, **_xargs(tmp_path)) == P.EXIT["STOP"]
    rec = json.loads((root / H.EV["export"] / "record.json").read_text())
    assert rec["stop"]["step"] == "graph" and calls == ["export"] and not (tmp_path / "export").exists()


def test_THE_EXPORT_REFUSES_A_MISSED_GATE_OR_A_CHANGED_CHALLENGER_BEFORE_ANYTHING_RUNS(tmp_path):
    root = _tree(tmp_path)
    cfg = _cfg(root)
    ch = _fit_evidence(cfg, tmp_path, verdict=H.GATE_MISSED)
    run, calls = _fake_runner()
    with pytest.raises(P.Refused, match="only an ELIGIBLE challenger"):
        X.run_export(cfg, runner=run, **_xargs(tmp_path))
    root2 = _tree(tmp_path / "two")
    cfg2 = _cfg(root2)
    ch = _fit_evidence(cfg2, tmp_path / "two")
    ch.write_bytes(b"tampered")
    with pytest.raises(P.Refused, match="not the one the fit saved"):
        X.run_export(cfg2, runner=run, **_xargs(tmp_path / "two"))
    assert calls == [] and not (root / H.EV["export"]).exists()
    assert X.main(["run", "export"]) == P.EXIT["REFUSED"]                          # the real gate is closed
