"""External-engine learning STAGE 3 (preliminary evaluation) -- SYNTHETIC checks only.

No held-out logit is computed and no game is played between real checkpoints here. The held-out comparison
and the screen run on synthetic logits and synthetic results; the games path runs the real evaluation runner
with GPU-free fake evaluators on a tiny budget. The only real inputs read are the committed manifest and the
stage-1/stage-2 evidence it binds (hashes, row metadata and the held-out rows' rebuild -- no forward pass).
"""
import ast
import importlib.util
import json
import math
import pathlib
import shutil

import pytest

from scripts.GPU.alphazero import external_learning as EL
from scripts.GPU.alphazero import external_learning_stage2 as S2
from scripts.GPU.alphazero import external_learning_stage3 as S3
from scripts.GPU.alphazero import external_learning_stage3_authorization as GATE
from scripts.GPU.alphazero import eval_runner as ER

SYNTHETIC = 1_000_000                                    # never a 2026-band seed
GAMES4 = ["g0", "g1", "g2", "g3"]
needs_mlx = pytest.mark.skipif(importlib.util.find_spec("mlx") is None, reason="needs MLX")


def _tree(rel):
    root = S3.ROOT / rel
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.exists() else None


def _rows(per_game, n_legal=3):
    """Synthetic held-out rows: `per_game[i]` rows in game i, the teacher's choice always index 0."""
    return [{"row_id": f"{g}-{k}", "game_id": g, "teacher_colour": "red" if i % 2 else "black",
             "legal_moves": [[0, j] for j in range(n_legal)], "chosen_index": 0}
            for i, (g, n) in enumerate(zip(GAMES4, per_game)) for k in range(n)]


# ─────────────────────────────── the gate, the manifest, the environment ───────────────────────────────

def test_the_gate_is_closed_and_both_entries_refuse_changing_nothing(tmp_path):
    assert GATE.EXTERNAL_LEARNING_STAGE3_AUTHORIZED is False
    before = [_tree(S3.RUN_DIR), _tree(S3.EVIDENCE_DIR), (S3.ROOT / S3.CLEANUP_FAILED).exists()]
    assert S3.main(["run"]) == S3.EXIT_REFUSED
    assert S3.main(["_child", "--run-dir", str(tmp_path)]) == S3.EXIT_REFUSED
    assert [_tree(S3.RUN_DIR), _tree(S3.EVIDENCE_DIR), (S3.ROOT / S3.CLEANUP_FAILED).exists()] == before
    assert list(tmp_path.iterdir()) == []


def test_the_gate_module_holds_only_its_declaration():
    tree = ast.parse(pathlib.Path(GATE.__file__).read_text())
    assert [type(n).__name__ for n in tree.body] == ["Expr", "Assign"]


def test_the_committed_manifest_is_PINNED_and_DERIVED_and_freezes_the_cards_protocol():
    m = S3.check_manifest()
    assert m["games"]["eval_config"] == {"board_size": 24, "mcts_sims": 400, "mcts_eval_batch_size": 14,
                                         "mcts_stall_flush_sims": 48, "selection_mode": "opening_temperature",
                                         "opening_temp_plies": 20, "temp_high": 1.0, "temp_low": 0.1,
                                         "max_moves": 280}
    assert (m["games"]["games"], m["games"]["pairs"], m["games"]["workers"], m["games"]["add_noise"]) == (32, 16, 1, False)
    assert m["games"]["mcts_config"]["n_simulations"] == 400 and m["games"]["mcts_config"]["opening_noise_ply"] == 0
    assert m["ceiling_s"] == 7200 and round(m["screen"]["half_width"], 4) == 0.3395
    assert round(0.5 - m["screen"]["half_width"], 4) == 0.1605
    rec2 = json.loads((S3.ROOT / S2.EVIDENCE_DIR / "record.json").read_text())
    assert m["stage2"]["control"]["beta"] == rec2["steps"]["control"]["beta"]                 # frozen, not refitted
    assert m["stage2"]["candidate"] == {"path": f"{S2.EVIDENCE_DIR}/candidate.safetensors",
                                        "sha256": rec2["steps"]["candidate_sha256"]}
    assert m["stage2"]["candidate"]["sha256"].startswith("e73e80b4")
    assert m["incumbent"]["sha1"] == "209cf2d4fd24a48553d259dd71b4954867b9473e"
    assert m["stage1"]["heldout"]["games"] == ["f4-teacher_black", "f4-teacher_red", "f5-teacher_black",
                                               "f5-teacher_red"] and m["stage1"]["heldout"]["rows"] == 64
    assert m["game_seeds"] == {"slots": 32, "plan": S3.PLAN}


def test_the_environment_is_PINNED_to_stage_2s_and_the_provenance_keeps_stage_1s_difference():
    m = S3.check_manifest()
    env2 = S3.recorded_environment((S3.ROOT / S2.EVIDENCE_DIR / "environment.txt").read_text())
    assert {k: m["environment"][k] for k in env2} == env2 == m["environment_provenance"]["stage2"]
    p1 = m["environment_provenance"]["stage1"]
    assert p1["python"].startswith("3.13.") and (p1["mlx"], p1["numpy"]) != (env2["mlx"], env2["numpy"])
    want = {k: m["environment"][k] for k in ("python", "pip_freeze_sha256", "mlx", "numpy")}
    assert S3.check_environment(m, now=want) == want
    for key, value in (("python", "3.13.15"), ("pip_freeze_sha256", "0" * 64), ("mlx", "0.32.1")):
        with pytest.raises(S3.Stop, match="not the pinned one"):
            S3.check_environment(m, now={**want, key: value})


def _evidence_copy(tmp_path):
    for rel in (S3.MANIFEST, EL.EVIDENCE_DIR + "/MANIFEST.sha256", EL.EVIDENCE_DIR + "/finalized.json",
                EL.EVIDENCE_DIR + "/" + S2.STAGE1_ROWS,
                "docs/superpowers/evidence/external-engine-learning-setup/03_venv_freeze.txt",
                *(f"{S2.EVIDENCE_DIR}/{f}" for f in ("MANIFEST.sha256", "record.json", "environment.txt", S2.CANDIDATE))):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(S3.ROOT / rel, tmp_path / rel)
    return tmp_path


def test_a_changed_manifest_beta_candidate_or_config_STOPS(tmp_path, monkeypatch):
    root = _evidence_copy(tmp_path)
    assert S3.check_manifest(root)["stage2"]["control"]["beta"] > 0
    rec = root / S2.EVIDENCE_DIR / "record.json"
    good = rec.read_bytes()
    r = json.loads(good)
    r["steps"]["control"]["beta"] += 1e-9                                       # a refitted beta
    rec.write_text(json.dumps(r))
    with pytest.raises(S3.Stop, match="not the one this module derives"):
        S3.check_manifest(root)
    rec.write_bytes(good)
    cand = root / S2.EVIDENCE_DIR / S2.CANDIDATE
    cand.write_bytes(cand.read_bytes()[:-1] + b"\0")                            # another checkpoint
    with pytest.raises(S3.Stop, match="not the one stage 2 saved"):
        S3.check_manifest(root)
    shutil.copyfile(S3.ROOT / S2.EVIDENCE_DIR / S2.CANDIDATE, cand)
    monkeypatch.setitem(S3.PROTOCOL, "mcts_sims", 401)                          # another game config
    with pytest.raises(S3.Stop, match="not the one this module derives"):
        S3.check_manifest(root)
    monkeypatch.setitem(S3.PROTOCOL, "mcts_sims", 400)
    real = ER.cfg_from
    monkeypatch.setattr(ER, "cfg_from", lambda c: real(c).__class__(**{**real(c).__dict__, "c_puct": 2.0}))
    with pytest.raises(S3.Stop, match="not the one this module derives"):     # a drifted search default
        S3.check_manifest(root)
    monkeypatch.setattr(ER, "cfg_from", real)
    (root / S3.MANIFEST).write_bytes((root / S3.MANIFEST).read_bytes().replace(b'"ceiling_s": 7200', b'"ceiling_s": 7201'))
    with pytest.raises(S3.Stop, match="not the pinned bytes"):
        S3.check_manifest(root)


def test_the_REAL_held_out_rows_are_the_four_sealed_games_rebuilt_and_tampering_STOPS(tmp_path):
    """The run's own loader on the committed evidence (row metadata and rebuild only; no forward pass)."""
    m = S3.check_manifest()
    rows = S3.load_heldout(m)
    assert len(rows) == 64 and {r["split"] for r in rows} == {"heldout"}
    assert sorted({r["game_id"] for r in rows}) == m["stage1"]["heldout"]["games"]
    for bad in ({**m["stage1"]["heldout"], "games": m["stage1"]["heldout"]["games"][:3]},
                {**m["stage1"]["heldout"], "rows": 63}, {**m["stage1"]["heldout"], "row_ids_sha256": "0" * 64}):
        with pytest.raises(S3.Stop, match="not the manifest's sealed games"):
            S3.load_heldout({**m, "stage1": {**m["stage1"], "heldout": bad}})
    root = _evidence_copy(tmp_path)
    p = root / EL.EVIDENCE_DIR / S2.STAGE1_ROWS
    p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises(S3.Stop, match="is not the file stage 1's manifest hashed"):
        S3.load_heldout(m, root)


# ─────────────────────────────── the plan: seeds ───────────────────────────────

def test_check_plan_refuses_an_unregistered_or_spent_seed_and_any_tampering(monkeypatch):
    from scripts.GPU.alphazero import e4_screen_reference as REG

    plan = S3.build_plan(SYNTHETIC, "a" * 64)
    assert plan["manifest_sha256"] == S3.MANIFEST_SHA256 and plan["n_slots"] == 32
    assert [g["seed"] for g in plan["games"]] == list(range(SYNTHETIC, SYNTHETIC + 32))
    with pytest.raises(S3.Stop, match="not ACCOUNTED"):
        S3.check_plan(plan)                                                     # the real registry
    monkeypatch.setattr(REG, "seed_is_accounted", lambda s: True)
    monkeypatch.setattr(REG, "seed_is_unavailable", lambda s: False)
    assert S3.check_plan(json.loads(json.dumps(plan))) == plan
    swapped = json.loads(json.dumps(plan))
    swapped["games"][3]["seed"], swapped["games"][4]["seed"] = swapped["games"][4]["seed"], swapped["games"][3]["seed"]
    for bad in ({**plan, "base": float(SYNTHETIC)}, {**plan, "manifest_sha256": "0" * 64}, {**plan, "n_slots": 31},
                {**plan, "survey_record_sha256": None}, {**plan, "extra": 1}, swapped, {**plan, "games": plan["games"][:-1]}):
        with pytest.raises(S3.Stop):
            S3.check_plan(bad)
    monkeypatch.setattr(REG, "seed_is_accounted", lambda s: s != SYNTHETIC + 31)
    with pytest.raises(S3.Stop, match="not ACCOUNTED"):
        S3.check_plan(plan)
    monkeypatch.setattr(REG, "seed_is_accounted", lambda s: True)
    monkeypatch.setattr(REG, "seed_is_unavailable", lambda s: s == SYNTHETIC + 7)
    with pytest.raises(S3.Stop, match="spent"):
        S3.check_plan(plan)


def test_the_tasks_ARE_the_plans_slots_colour_balanced_and_any_mismatch_STOPS():
    m = S3.check_manifest()
    plan = S3.build_plan(SYNTHETIC, "a" * 64)
    tasks = S3.game_tasks(plan, m)
    assert [t.seed for t in tasks] == list(range(SYNTHETIC, SYNTHETIC + 32))
    assert [t.red_agent.agent_id for t in tasks] == ["candidate", "incumbent"] * 16
    cand = str(S3.ROOT / m["stage2"]["candidate"]["path"])
    inc = str(S3.ROOT / m["incumbent"]["path"])
    assert {(t.red_checkpoint, t.black_checkpoint) for t in tasks} == {(cand, inc), (inc, cand)}
    assert {t.red_agent.readout for t in tasks} == {ER.readout_from_eval_config(S3.eval_config())}
    shifted = json.loads(json.dumps(plan))
    shifted["games"][0]["seed"] += 1
    with pytest.raises(S3.Stop, match="not the plan's seed/colour slots"):
        S3.game_tasks(shifted, m)
    recoloured = json.loads(json.dumps(plan))
    recoloured["games"][0]["candidate"] = "black"
    with pytest.raises(S3.Stop, match="not the plan's seed/colour slots"):
        S3.game_tasks(recoloured, m)


# ─────────────────────────────── checkpoints ───────────────────────────────

def test_only_the_two_PINNED_checkpoints_load_each_verified_once(tmp_path, monkeypatch):
    a, b = tmp_path / "cand.safetensors", tmp_path / "inc.safetensors"
    a.write_bytes(b"candidate bytes")
    b.write_bytes(b"incumbent bytes")
    import hashlib
    pinned = {str(a): ("sha256", hashlib.sha256(b"candidate bytes").hexdigest()),
              str(b): ("sha1", hashlib.sha1(b"incumbent bytes").hexdigest())}
    loads = []
    monkeypatch.setattr(ER, "_default_evaluator_factory", lambda p: loads.append(p) or object())
    factory = S3.pinned_factory(pinned)
    assert factory(str(a)) is factory(str(a)) and factory(str(b)) is not None and loads == [str(a), str(b)]
    with pytest.raises(S3.Stop, match="not a pinned stage-3 checkpoint"):
        S3.pinned_factory(pinned)(str(tmp_path / "other.safetensors"))
    for path, raw in ((a, b"candidate byteS"), (b, b"incumbent byteS")):
        good = path.read_bytes()
        path.write_bytes(raw)
        with pytest.raises(S3.Stop, match="is not the pinned"):
            S3.pinned_factory(pinned)(str(path))
        path.write_bytes(good)
    assert loads == [str(a), str(b)]                                           # nothing else was ever loaded


def test_the_real_pins_name_the_committed_candidate_and_the_incumbent():
    m = S3.check_manifest()
    pinned = S3.pins(m)
    for path in pinned:
        S3.verify_checkpoint(path, pinned)                                      # bytes only; nothing loaded
    assert sorted(kind for kind, _ in pinned.values()) == ["sha1", "sha256"]


# ─────────────────────────────── the held-out comparison ───────────────────────────────

GOOD, MID, BAD = [5.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 4.0, 0.0]


def test_the_four_games_weigh_EQUALLY_a_pooled_row_mean_would_have_passed():
    """Game g0 holds 20 rows the candidate gets right; g1..g3 one row each it gets badly wrong. Pooled over
    rows the candidate's CE beats both comparators; per game (each game once) it does not."""
    rows = _rows([20, 1, 1, 1])
    inc = [MID] * len(rows)
    cand = [GOOD] * 20 + [BAD] * 3
    out = S3.heldout_comparison(rows, inc, cand, 0.56, GAMES4)
    ce = lambda z: math.log(sum(math.exp(x) for x in z)) - z[0]
    pooled = sum(ce(z) for z in cand) / len(cand)
    assert pooled < ce([0.56 * x for x in MID]) < ce(MID)                       # rows pooled: would pass
    per = [out["per_game"][g]["candidate_ce"] for g in GAMES4]
    assert out["game_means"]["candidate_ce"] == pytest.approx(sum(per) / 4)
    assert out["game_means"]["candidate_ce"] > out["game_means"]["incumbent_ce"] and out["passes"] is False
    assert [out["per_game"][g]["rows"] for g in GAMES4] == [20, 1, 1, 1]


def test_a_GENUINE_improvement_on_every_game_passes_and_reports_every_card_quantity():
    rows = _rows([3, 2, 4, 1])
    out = S3.heldout_comparison(rows, [MID] * 10, [GOOD] * 10, 0.56, GAMES4)
    assert out["passes"] is True
    g = out["per_game"]["g0"]
    assert set(g) == {"rows", "teacher_colour", "incumbent_ce", "control_ce", "candidate_ce", "incumbent_top1",
                      "candidate_top1", "incumbent_rank", "candidate_rank", "move_disagreements"}
    assert (g["incumbent_rank"], g["candidate_rank"], g["move_disagreements"]) == (2, 1, 3)
    assert set(out["paired"]["g3"]) == {"candidate_minus_incumbent_ce", "candidate_minus_control_ce",
                                        "candidate_minus_incumbent_top1"}
    assert set(out["colour_means_descriptive"]) == {"red", "black"}


def test_PURE_temperature_rescaling_cannot_pass_however_much_it_lowers_CE():
    rows = _rows([2, 2, 2, 2])
    inc = [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]] * 4                               # half right, half wrong
    sharp = [[3.0 * x for x in z] for z in inc]
    out = S3.heldout_comparison(rows, inc, sharp, 0.56, GAMES4)
    assert out["game_means"]["candidate_top1"] == out["game_means"]["incumbent_top1"] and out["passes"] is False


def test_a_MISSING_EXTRA_or_UNEVEN_held_out_input_STOPS():
    rows = _rows([2, 2, 2, 2])
    z = [MID] * 8
    with pytest.raises(S3.Stop, match="exactly the 4 frozen held-out games"):
        S3.heldout_comparison(rows[:6], z[:6], z[:6], 0.56, GAMES4)            # g3 missing
    with pytest.raises(S3.Stop, match="exactly the 4 frozen held-out games"):
        S3.heldout_comparison(rows, z, z, 0.56, GAMES4[:3] + ["g9"])
    with pytest.raises(S3.Stop, match="does not cover its legal moves"):
        S3.heldout_comparison(rows, z, z[:7], 0.56, GAMES4)
    with pytest.raises(S3.Stop, match="nonfinite"):
        S3.heldout_comparison(rows, z, [[math.nan, 0.0, 0.0]] + z[1:], 0.56, GAMES4)


# ─────────────────────────────── the stage: held-out failure prevents games ───────────────────────────────

def _wire(monkeypatch, cand_logits, plays, m=None, stub_games=True):
    """The stage with synthetic held-out inputs; game-side entry points record whether they were reached."""
    rows = _rows([2, 2, 2, 2])
    m = m or {"stage1": {"heldout": {"games": GAMES4}}, "stage2": {"candidate": {"path": "/c.safetensors"},
                                                                   "control": {"beta": 0.56}},
              "incumbent": {"path": "/i.safetensors"}}
    inc = str(S3.ROOT / m["incumbent"]["path"])
    monkeypatch.setattr(EL, "_committed", lambda rel: True)
    monkeypatch.setattr(S3, "load_plan", lambda: ({"n_slots": 32}, "p" * 64))
    monkeypatch.setattr(S3, "check_manifest", lambda: m)
    monkeypatch.setattr(S3, "check_environment", lambda man: {"python": "x"})
    monkeypatch.setattr(S3, "load_heldout", lambda man: rows)
    monkeypatch.setattr(S3, "load_network", lambda path, pinned: path)
    monkeypatch.setattr(S2, "records_of", lambda rs: rs)
    monkeypatch.setattr(S2, "forward_rows", lambda net, recs: ([MID] * 8 if net == inc else cand_logits, []))
    if stub_games:
        monkeypatch.setattr(S3, "pins", lambda man: {})
        monkeypatch.setattr(S3, "game_tasks", lambda plan, man: plays.append("tasks") or [])
        monkeypatch.setattr(S3, "play_games", lambda *a, **k: plays.append("games") or [])
    return m


@pytest.mark.parametrize("cand", [[MID] * 8, [[0.56 * x for x in MID]] * 8, [BAD] * 8], ids=["same", "control", "worse"])
def test_FAILED_held_out_criteria_prevent_every_game(monkeypatch, tmp_path, cand):
    plays = []
    _wire(monkeypatch, cand, plays)
    rec = {"steps": {}}
    with pytest.raises(S3.HeldoutFailed):
        S3.stage3(tmp_path, rec)
    assert plays == [] and rec["steps"]["heldout"]["passes"] is False and "tasks" not in rec["steps"]


def test_the_CHILD_records_STOP_HELDOUT_exit_1_and_a_passing_comparison_reaches_the_games(monkeypatch, tmp_path):
    plays = []
    _wire(monkeypatch, [MID] * 8, plays)
    monkeypatch.setattr(GATE, "EXTERNAL_LEARNING_STAGE3_AUTHORIZED", True)        # in-process only
    assert S3._child(tmp_path) == S3.EXIT_STOP_HELDOUT == 1
    rec = json.loads((tmp_path / "record.json").read_text())
    assert rec["verdict"] == "STOP_HELDOUT" and rec["stop"]["step"] == "heldout" and plays == []
    assert S3.exit_code_for(tmp_path / "record.json", "EXITED", 1) == 1
    assert S3.exit_code_for(tmp_path / "record.json", "EXITED", 0) == S3.EXIT_STOP     # disagreement: never a pass
    run2 = tmp_path / "second"
    run2.mkdir()
    _wire(monkeypatch, [GOOD] * 8, plays)
    S3._child(run2)                                                             # passes, then the (stubbed) games
    assert plays == ["tasks", "games"]


def test_a_runner_INTEGRITY_fault_is_the_cards_STOP_exit_2(monkeypatch, tmp_path):
    from scripts.GPU.alphazero.eval_integrity import IntegrityError

    def fault(*a, **k):
        raise IntegrityError("task 3: game ended in unknown_error")
    _wire(monkeypatch, [GOOD] * 8, [])
    monkeypatch.setattr(S3, "game_tasks", lambda plan, man: [])
    monkeypatch.setattr(S3, "play_games", fault)
    monkeypatch.setattr(GATE, "EXTERNAL_LEARNING_STAGE3_AUTHORIZED", True)        # in-process only
    assert S3._child(tmp_path) == S3.EXIT_STOP == 2
    rec = json.loads((tmp_path / "record.json").read_text())
    assert (rec["verdict"], rec["stop"]["step"]) == ("STOP", "integrity") and "unknown_error" in rec["stop"]["detail"]
    assert "screen" not in rec["steps"]


# ─────────────────────────────── the screen ───────────────────────────────

def _results(cand_scores, reasons=None):
    """32 results from the runner's own result builder; the candidate is red on even games."""
    m = S3.check_manifest()
    tasks = S3.game_tasks(S3.build_plan(SYNTHETIC, "a" * 64), m)
    out = []
    for t, s, why in zip(tasks, cand_scores, reasons or ["win"] * 32):
        cand_red = t.red_agent.agent_id == "candidate"
        winner = None if s == 0.5 else ("red" if (s == 1.0) == cand_red else "black")
        out.append(ER.make_result(t, winner, why, 40))
    return out, tasks


def test_the_screen_scores_the_candidate_in_BOTH_colours_over_16_pairs_and_the_cards_veto():
    from scripts.GPU.alphazero.eval_integrity import validate_result_set

    res, tasks = _results([1.0] * 32)
    validate_result_set(res, tasks, "candidate", "incumbent")
    s = S3.screen_summary(res)
    assert (s["candidate_wins"], s["mean_pair_score"], s["outcome"]) == (32, 1.0, "NOT_VETOED")
    s = S3.screen_summary(_results([0.0] * 32)[0])
    assert (s["candidate_losses"], s["outcome"]) == (32, "VETO") and round(s["interval_unclipped"][1], 4) == 0.3395
    five, six = [1.0] * 5 + [0.0] * 27, [1.0] * 6 + [0.0] * 26                  # 5/32 = 0.156 < 0.1605 < 6/32
    assert S3.screen_summary(_results(five)[0])["outcome"] == "VETO"
    assert S3.screen_summary(_results(six)[0])["outcome"] == "NOT_VETOED"
    black_only = [0.0, 1.0] * 16                                               # wins only as black
    assert S3.screen_summary(_results(black_only)[0])["pair_means"] == [0.5] * 16


def test_more_than_half_CAPS_is_UNINTERPRETABLE_and_an_incomplete_cohort_has_no_rate():
    caps17 = _results([0.5] * 17 + [1.0] * 15, ["state_cap"] * 17 + ["win"] * 15)[0]
    s = S3.screen_summary(caps17)
    assert (s["caps"], s["outcome"]) == (17, "UNINTERPRETABLE")
    caps16 = _results([0.5] * 16 + [1.0] * 16, ["state_cap"] * 16 + ["win"] * 16)[0]
    assert S3.screen_summary(caps16)["outcome"] == "NOT_VETOED"
    with pytest.raises(S3.Stop, match="incomplete cohort"):
        S3.screen_summary(caps16[:-1])


# ─────────────────────────────── the games path, end to end, on fakes ───────────────────────────────

@needs_mlx
def test_the_WHOLE_stage_with_FAKE_evaluators_plays_the_plans_games_through_the_runner(monkeypatch, tmp_path):
    """The real runner (workers 1, replays captured, per-game binding checks) with GPU-free evaluators on a
    tiny budget: 4 games / 2 pairs, 24x24, 4 simulations, a 6-ply cap."""
    import hashlib

    from tests.eval_fakes import fake_evaluator_factory

    for name, value in (("GAMES", 4), ("PAIRS", 2), ("N_SLOTS", 4)):
        monkeypatch.setattr(S3, name, value)
    monkeypatch.setattr(S3, "PROTOCOL", {**S3.PROTOCOL, "mcts_sims": 4, "mcts_eval_batch_size": 2,
                                          "mcts_stall_flush_sims": 2, "opening_temp_plies": 2, "max_moves": 6})
    cand, inc = tmp_path / "c.safetensors", tmp_path / "i.safetensors"
    cand.write_bytes(b"c")
    inc.write_bytes(b"i")
    m = {"stage1": {"heldout": {"games": GAMES4}},
         "stage2": {"candidate": {"path": str(cand), "sha256": hashlib.sha256(b"c").hexdigest()},
                    "control": {"beta": 0.56}},
         "incumbent": {"path": str(inc), "sha1": hashlib.sha1(b"i").hexdigest()}}
    _wire(monkeypatch, [GOOD] * 8, [], m=m, stub_games=False)                  # real pins, tasks and games
    plan = S3.build_plan(SYNTHETIC, "a" * 64)
    monkeypatch.setattr(S3, "load_plan", lambda: (plan, "p" * 64))
    monkeypatch.setattr(ER, "_default_evaluator_factory", fake_evaluator_factory)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rec = {"steps": {}}
    S3.stage3(run_dir, rec)
    s = rec["steps"]
    assert list(s) == ["plan", "manifest_sha256", "environment", "heldout_rows", "heldout", "tasks", "games", "screen"]
    assert [g["seed"] for g in s["games"]] == list(range(SYNTHETIC, SYNTHETIC + 4))
    assert [g["red_agent_id"] for g in s["games"]] == ["candidate", "incumbent"] * 2
    assert sorted(p.name for p in (run_dir / S3.REPLAYS).iterdir()) == [f"game_{i:06d}.json" for i in range(4)]
    events = [json.loads(line) for line in (run_dir / "progress.jsonl").read_text().splitlines()]
    assert [e["game"] for e in events if e["event"] == "game_end"] == [0, 1, 2, 3]
    assert len(s["screen"]["pair_means"]) == 2
