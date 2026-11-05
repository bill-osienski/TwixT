"""H-TB bounded timing: focused fake checks. No checkpoint, corpus, TensorFlow or GPU; MLX pinned to the CPU.

The gate is opened only IN MEMORY (monkeypatch); the file stays closed.
"""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import mlx.core as mx
import numpy as np
import pytest

from scripts.GPU.alphazero import htb_pilot as H
from scripts.GPU.alphazero import htb_timing as T
from scripts.GPU.alphazero import htb_timing_authorization as GATE
from scripts.GPU.alphazero import idle_gate

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import htb_timing_wrapper_spec as W                       # noqa: E402
from test_htb_pilot import FakeTeacher, _Spy, ids_in, random_game, tiny_net   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


@pytest.fixture
def opened(monkeypatch):
    monkeypatch.setattr(GATE, "HTB_TIMING_AUTHORIZED", True)
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")


def _train_games(n=8):
    return [{"id": gid, "moves": random_game(600 + i, 32)} for i, gid in enumerate(ids_in("train", n))]


class SlowTeacher(FakeTeacher):
    def evaluate(self, tb, colour, legal):
        time.sleep(0.002)
        return super().evaluate(tb, colour, legal)


def _tcfg(tmp_path, games=None, **kw):
    root = tmp_path / "repo"
    for rel in (H.CARD, T.PLAN, H.C0_INVENTORY):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, root / rel)
    g = games if games is not None else _train_games()
    base = dict(root=root, data_root=root, cache=tmp_path / "cache", committed=lambda r, rel: rel in (H.CARD, T.PLAN),
                games_train=_Spy(lambda r: g), load_teacher=_Spy(lambda c: (SlowTeacher(smooth=True), {"fake": 1})),
                load_reference=_Spy(lambda r: tiny_net(0)),
                available_memory=lambda: {"physical_bytes": 48 << 30, "available_bytes": 20 << 30, "page_size": 16384},
                forbid_modules=("a_module_nothing_imports",),
                key_games=3, teacher_games=2, warmup_calls=2, fit_first_game=2, fit_rows=16, memory_rows=4,
                updates=8, warmup_updates=2, reload_shards=2, shard_rows=5, rows_per_batch=4)
    base.update(kw)
    return T.TimingConfig(**base)


def _rec(cfg, kind):
    return json.loads((cfg.evidence(kind) / "record.json").read_text())


# ─────────────────────────────── gate, plan, ceilings ───────────────────────────────

def test_THE_TIMING_GATE_IS_CLOSED_AND_BOTH_ENTRIES_REFUSE(capsys):
    assert GATE.HTB_TIMING_AUTHORIZED is False
    assert T.main(["time", "label", "--clone-root", "/nonexistent"]) == 2
    assert T.main(["time", "fit"]) == 2
    assert capsys.readouterr().err.count("HTB_TIMING_AUTHORIZED is closed") == 2
    # re-anchored at the ONE T-label run (gate opening 87c5365c; ledger #172): it STOPPED at selection on a
    # black-first game; T-fit has never run
    ev = ROOT / T.EVIDENCE
    assert sorted(x.name for x in ev.iterdir()) == ["label"]
    assert sorted(x.name for x in (ev / "label").iterdir()) == ["record.json", "started.json"]
    rec = json.loads((ev / "label" / "record.json").read_text())
    assert rec["status"] == "STOP" and rec["stop"] == {"step": "corpus",
                                                        "detail": "ply 0: recorded player black but red to move"}
    assert "measurements" not in rec and "projection" not in rec
    outer = json.loads((ROOT / "docs/superpowers/evidence/htb-timing-label-run/outer.json").read_text())
    assert (outer["inner_exit"], outer["outer_exit"], outer["gate_restored"], outer["survivors"]) == (3, 3, True, "")
    assert not (ROOT / "docs/superpowers/evidence/htb-timing-fit-run/outer.json").exists()


def test_THE_PLAN_IS_PINNED_AND_THE_CEILINGS_ARE_THE_USERS_EXACTLY():
    assert hashlib.sha256((ROOT / T.PLAN).read_bytes()).hexdigest() == T.PLAN_SHA256
    assert T.CEILINGS == {"label": {"term_s": 1190, "kill_s": 1200, "cleanup_through_s": 1210},
                          "fit": {"term_s": 1790, "kill_s": 1800, "cleanup_through_s": 1810}}
    assert T.EXECUTION_ALLOWANCE_S == 3000
    assert idle_gate.MAX_WAIT_S == 1800.0                       # idle waiting: separate, per wrapper
    d = T.TimingConfig.__dataclass_fields__
    assert (d["key_games"].default, d["teacher_games"].default, d["fit_rows"].default, d["updates"].default,
            d["warmup_updates"].default, d["warmup_calls"].default) == (200, 40, 640, 60, 10, 20)


# ─────────────────────────────── projections (pure arithmetic) ───────────────────────────────

def test_THE_LABEL_PROJECTION_PRICES_EVERY_TERM_AND_COUNTS_TEACHER_TIME_ONCE():
    cfg = T.TimingConfig(root=ROOT, data_root=ROOT)
    c0 = {"train": {"games": 10, "positions": 900}, "val": {"games": 1, "positions": 100}}
    m = {"teacher_load_s": 5.0, "corpus_auth_s": 60.0, "selection_s": 2.0, "selection_positions": 100,
         "teacher_calls": {"calls": 100, "warmup_calls": 20, "warmup_s": 4.0, "steady_mean_s": 0.001},
         "replay_overhead_s": 1.0, "plies_replayed": 1000, "shard_write_s": 0.5, "selection_serialization_s": 3.0}
    p = T.project_label(m, c0, cfg)
    calls = 2 * (240_000 + 24_000)
    want = {"teacher_load_s": 5.0, "corpus_auth_s": 60.0, "selection_s": 20.0,
            "teacher_calls_s": round(4.0 + (calls - 20) * 0.001, 1), "replay_overhead_s": 2.0,
            "shard_writes_s": 16.5, "selection_serialization_s": 3.0}
    assert p["terms"] == want and p["basis"]["shards"] == 33 and p["basis"]["teacher_calls"] == 528_000
    total = 5 + 60 + 20 + 4.0 + (calls - 20) * 0.001 + 2 + 16.5 + 3
    assert p["projected_s"] == round(total, 1)
    assert p["kill_s"] == 1200 and p["term_s"] == 1190 and p["cleanup_through_s"] == 1210   # 1.5 x 638.5 = 957.7 -> 1200
    assert p["unpriced"]


def test_THE_FIT_PROJECTION_AND_THE_MEMORY_ESTIMATE():
    cfg = T.TimingConfig(root=ROOT, data_root=ROOT)
    m = {"corpus_auth_s": 60.0, "reload_s_per_shard": 1.0, "examples_s_per_row": 0.001, "load_s": [2.0, 1.0],
         "score_s_per_row_network": 0.002, "save_s": 0.5,
         "updates": {"updates": 60, "warmup_updates": 10, "warmup_s": 10.0, "steady_mean_s": 0.5}}
    p = T.project_fit(m, cfg)
    assert p["terms"] == {"corpus_auth_s": 60.0, "label_reload_s": 33.0, "examples_s": 264.0, "model_loads_s": 3.0,
                          "start_check_s": round(0.002 * 2048, 1), "validation_scoring_s": 96.0,
                          "updates_s": 10.0 + 14_990 * 0.5, "save_s": 0.5}
    assert p["kill_s"] % 600 == 0 and p["kill_s"] >= 1.5 * p["projected_s"] > p["kill_s"] - 600
    mm = {"board_bytes": 69_120, "example_bytes_traced": 140_000, "label_row_bytes": 4_808, "networks_bytes": 15_000_000,
          "memory_before": {"physical_bytes": 51_539_607_552, "available_bytes": 19_844_907_008},
          "lazy_board_s_per_row": 0.004}
    e = T.memory_estimate(mm, cfg)
    assert e["boards_only_bytes"] == 2 * 69_120 * 264_000 == 36_495_360_000            # the user's ~36.5 GB
    assert e["estimate_bytes"] == 140_000 * 264_000 + 4_808 * 264_000 + 15_000_000
    assert e["fits_available"] is False and e["fits_physical"] is True
    assert e["lazy_boards_projected_s"] == pytest.approx(0.004 * 15_000 * 32)           # per row covers both orientations


# ─────────────────────────────── the entries on fakes ───────────────────────────────

def test_T_LABEL_COMPLETES_ON_FAKES_TRAIN_ONLY_AND_RECORDS_NO_OUTPUTS(tmp_path, opened):
    cfg = _tcfg(tmp_path)
    assert T.time_label("/fake-clone", cfg) == 0
    rec = _rec(cfg, "label")
    assert rec["status"] == "COMPLETED" and rec["ceiling"] == T.CEILINGS["label"]
    m = rec["measurements"]
    tc = m["teacher_calls"]
    assert tc["calls"] > tc["warmup_calls"] == 2 and m["plies_replayed"] > 0 and m["selection_positions"] > 0
    assert 0 <= m["replay_overhead_s"] < m["label_game_total_s"]         # teacher time is NOT inside the overhead
    assert set(rec["projection"]["terms"]) == {"teacher_load_s", "corpus_auth_s", "selection_s", "teacher_calls_s",
                                               "replay_overhead_s", "shard_writes_s", "selection_serialization_s"}
    T.assert_no_outputs(rec)
    text = (cfg.evidence("label") / "record.json").read_text()
    assert '"value"' not in text and '"target' not in text


def test_T_LABEL_STOPS_ON_A_NON_TRAIN_GAME_AND_ON_A_FORBIDDEN_IMPORT(tmp_path, opened):
    val = [{"id": ids_in("val", 1)[0], "moves": random_game(700, 20)}]
    cfg = _tcfg(tmp_path / "a", games=_train_games(4) + val)
    assert T.time_label("/fake", cfg) == 3
    assert _rec(cfg, "label")["stop"]["step"] == "arm"
    cfg = _tcfg(tmp_path / "b", forbid_modules=("numpy",))
    assert T.time_label("/fake", cfg) == 3
    assert _rec(cfg, "label")["stop"]["step"] == "isolation"


@pytest.mark.parametrize("kind", ["label", "fit"])
@pytest.mark.parametrize("where", ["evidence", "cache"])
def test_A_TIMING_REFUSES_BEFORE_ANY_LOAD_WHEN_A_DESTINATION_EXISTS(tmp_path, opened, capsys, kind, where):
    cfg = _tcfg(tmp_path)
    (cfg.evidence(kind) if where == "evidence" else cfg.cache_dir(kind)).mkdir(parents=True)
    argv = ["time", kind] + (["--clone-root", "/fake"] if kind == "label" else [])
    assert T.main(argv, cfg) == 2 and "already exists" in capsys.readouterr().err
    assert cfg.load_teacher.calls == cfg.games_train.calls == cfg.load_reference.calls == 0
    assert not (cfg.evidence(kind) / "record.json").exists()


def test_A_TIMING_REFUSES_AN_UNCOMMITTED_OR_CHANGED_PLAN(tmp_path, opened, capsys):
    cfg = _tcfg(tmp_path, committed=lambda r, rel: rel == H.CARD)
    assert T.main(["time", "fit"], cfg) == 2 and "plan is not committed" in capsys.readouterr().err
    cfg = _tcfg(tmp_path / "x")
    with open(cfg.root / T.PLAN, "a") as f:
        f.write("\n")
    assert T.main(["time", "fit"], cfg) == 2 and "plan is not the pinned one" in capsys.readouterr().err
    assert cfg.games_train.calls == cfg.load_reference.calls == 0


def test_T_FIT_COMPLETES_ON_FAKES_PRICES_THE_RUN_AND_ESTIMATES_MEMORY(tmp_path, opened):
    cfg = _tcfg(tmp_path)
    assert T.time_fit(cfg) == 0
    rec = _rec(cfg, "fit")
    assert rec["status"] == "COMPLETED" and rec["ceiling"] == T.CEILINGS["fit"]
    m = rec["measurements"]
    assert m["updates"]["updates"] == 8 and m["updates"]["warmup_updates"] == 2 and cfg.load_reference.calls == 2
    assert m["board_bytes"] == 69_120 and m["example_bytes_traced"] >= 2 * 69_120
    assert set(rec["projection"]["terms"]) == {"corpus_auth_s", "label_reload_s", "examples_s", "model_loads_s",
                                               "start_check_s", "validation_scoring_s", "updates_s", "save_s"}
    mem = rec["memory_estimate"]
    assert mem["boards_only_bytes"] == 2 * 69_120 * 264_000 and mem["fits_available"] is False
    T.assert_no_outputs(rec)
    assert (cfg.cache_dir("fit") / "student-timing.safetensors").is_file()


def test_T_FIT_KEEPS_DISTINCT_STATES_WHEN_GAMES_SHARE_AN_OPENING(tmp_path, opened):
    shared = random_game(800, 4)                              # every game repeats the same first 4 moves
    games = []
    for i, gid in enumerate(ids_in("train", 8)):
        s = H.TwixtState.from_moves(shared)
        moves = list(shared)
        for m in random_game(810 + i, 40):
            if m in s.legal_moves() and s.winner() is None:
                moves.append(m)
                s = s.apply_move(m)
        games.append({"id": gid, "moves": moves})
    cfg = _tcfg(tmp_path, games=games)
    assert T.time_fit(cfg) == 0, _rec(cfg, "fit").get("stop")


def test_THE_OUTPUT_CHECK_FINDS_A_NESTED_OUTPUT_FIELD():
    with pytest.raises(H.Stop, match="loss"):
        T.assert_no_outputs({"measurements": {"updates": [{"s": 1.0, "loss": 2.0}]}})


# ─────────────────────────────── the wrappers ───────────────────────────────

@pytest.mark.parametrize("kind", ["label", "fit"])
def test_EACH_WRAPPER_IS_THE_DECLARED_DERIVATION_OF_THE_CYCLE2_WRAPPER(kind):
    src = (ROOT / W.SOURCE).read_bytes()
    assert hashlib.sha256(src).hexdigest() == W.SOURCE_SHA256
    p = ROOT / W.TARGETS[kind]
    assert p.read_text() == W.derive(kind, src.decode())
    assert os.access(p, os.X_OK)
    assert subprocess.run(["bash", "-n", str(p)]).returncode == 0
    text = p.read_text()
    term = T.CEILINGS[kind]["term_s"]
    assert f"\nTERM_S={term}\n" in text and "\nGRACE_S=10\n" in text
    assert term + 10 == T.CEILINGS[kind]["kill_s"] and term + 20 == T.CEILINGS[kind]["cleanup_through_s"]
    assert f'-m "$MODULE" time {kind}' in text and "GATE=HTB_TIMING_AUTHORIZED\n" in text


def test_THE_LABEL_OUTER_RECORD_NAMES_THE_TEACHER_INTERPRETER():
    """The user's 2026-11-04 correction: T-label launches $TPY, so outer.json's "python" is $TPY; the project
    interpreter that runs only the idle and background checks is recorded as "checks_python"."""
    text = (ROOT / W.TARGETS["label"]).read_text()
    rec = [l for l in text.splitlines() if '> "$OUT_REL/outer.json"' in l]
    assert rec == ['      "$W" "$HEAD_C" "$PRE" "$TPY" "$PY" "$CKPT" > "$OUT_REL/outer.json" || final=5']
    assert '"python": "%s", "checks_python": "%s", "clone_root": "%s"}' in text
    assert 'PY=""; TPY=""; CKPT=""' in text                                    # defined under set -u before any refusal
    fit = (ROOT / W.TARGETS["fit"]).read_text()
    assert "TPY" not in fit and '"$PRE" "$PY" "$CKPT" > "$OUT_REL/outer.json"' in fit
