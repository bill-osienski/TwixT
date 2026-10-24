"""The corrected-configuration trial's fresh held-out set (`scripts/GPU/alphazero/search_distillation_d2_heldout.py`):
one phase of all-held-out games, its manifest, the setting and version records, eligibility and stops, through the
TEST seams -- real MCTS, the repo's CPU FakeEvaluator, a small network that is only hashed, 8 simulations and four games
(not the card's 12 at 1,600). The public entry's refusals, the pins, and fresh processes (no MLX on import; the real
setting, gate, card and code checks).

🔴 EVERY TEST IS ISOLATED, WHATEVER THE GATE SAYS (ledger entry 28, #74): the autouse fixture redirects the evidence and
held-out directories and REPLACES the module's `conclude` with a stub, so nothing reached through the public entry can
play a game; the tests call the real boundary as `REAL_CONCLUDE`, with fixture seams only. The fresh-process checks
redirect the outputs and stub `conclude` too."""
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import sys

import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

from scripts.GPU.alphazero import search_distillation_d2 as D  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_heldout as H  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_heldout_authorization as GATE  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_measure as M  # noqa: E402
from scripts.GPU.alphazero import self_play as sp  # noqa: E402
from scripts.GPU.alphazero import training_replay_capture as C  # noqa: E402
from tests.test_search_distillation_d2_measure import _seams  # noqa: E402

REAL_CONCLUDE = H.conclude
REAL_PARENT = H.HELDOUT_PARENT
SPEC = {"games": 4, "n_simulations": 8, "phase_bytes": 64 << 20}
PLAN = {"base": 100, "n_slots": 4}
OK = lambda: dict(H.VERSIONS)                                            # noqa: E731


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    reached = []
    monkeypatch.setattr(sp, "_MIRROR_PROB", 0.0)
    monkeypatch.setattr(H, "EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setattr(H, "HELDOUT_PARENT", tmp_path / "cache")
    monkeypatch.setattr(H, "conclude", lambda *a, **k: reached.append(1) or (H.EXIT_DONE, {
        "status": "STUB", "head": None, "heldout_dir": "stub"}))
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    monkeypatch.setattr(H, "_mlx_loaded", lambda: False)
    (tmp_path / "cache").mkdir()
    return {"tmp": tmp_path, "reached": reached}


@pytest.fixture
def opened(monkeypatch, isolated):
    monkeypatch.setattr(GATE, "SEARCH_DISTILLATION_D2_HELDOUT_AUTHORIZED", True)
    return isolated


def _checkpoint() -> pathlib.Path:
    common = subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=H.ROOT,
                                     text=True).strip()
    return pathlib.Path(common).parent / M.CHECKPOINT


def _progress(out):
    return [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(sp, "_MIRROR_PROB", 0.0)
    mp.setenv("MLX_ENABLE_TF32", "0")
    root = tmp_path_factory.mktemp("heldout")
    code, rec = REAL_CONCLUDE(SPEC, _seams(read_plan=lambda: PLAN), root / "ev", root / "set", head="e" * 40,
                              versions_of=OK)
    mp.undo()
    return code, rec, root


# ─────────────────────────────── the run, through the test seams ───────────────────────────────

def test_A_COMPLETED_RUN_is_eligible_and_its_manifest_binds_the_one_all_held_out_phase(completed):
    code, rec, root = completed
    assert code == H.EXIT_DONE and rec["status"] == "COMPLETED" and rec["eligible"] is True, rec.get("void")
    assert json.loads((root / "ev" / "record.json").read_text()) == json.loads(json.dumps(rec))
    raw = (root / "ev" / "heldout.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == rec["heldout_sha256"]
    man = json.loads(raw)
    assert man["kind"] == H.MANIFEST_KIND and man["eligible"] is True and man["held_out_games"] == [0, 1, 2, 3]
    assert man["heldout_dir"] == str(root / "set") and man["phase"]["games"] == [0, 1, 2, 3]
    assert man["model_digest"] == rec["model_digest_before"] == rec["model_digest_after"]
    d = root / "set" / H.PHASE_DIR
    assert hashlib.sha256((d / "finalized.json").read_bytes()).hexdigest() == man["phase"]["finalized_sha256"]
    for f in C.PHASE_FILES:                                      # the evidence copies are the set's bytes
        assert (root / "ev" / "phases" / H.PHASE_DIR / f).read_bytes() == (d / f).read_bytes()
    _ctx, records, _ids = D.load_records(str(d))
    assert len(records) == man["phase"]["records"] == rec["totals"]["records"]
    assert [g["seed"] for g in rec["games"]] == [100, 101, 102, 103] and all(g["held_out"] for g in rec["games"])
    assert man["seeds"] == {"base": 100, "n_slots": 4} and rec["versions"] == H.VERSIONS
    events = [e["event"] for e in _progress(root / "ev")]
    assert events.count("game_start") == 4 and events.count("phase_end") == 1 and events[-1] == "end"


def test_THE_SETTING_IS_RECORDED_AT_START_AND_END_in_the_record_and_the_manifest(completed):
    _, rec, root = completed
    assert rec["setting"] == rec["setting_at_end"] == {"MLX_ENABLE_TF32": "0"}
    assert json.loads((root / "ev" / "heldout.json").read_text())["setting"] == {"MLX_ENABLE_TF32": "0"}
    assert _progress(root / "ev")[0] == {**_progress(root / "ev")[0], "event": "start",
                                         "setting": {"MLX_ENABLE_TF32": "0"}}


def test_THE_FROZEN_SPEC_IS_THE_CARD_S():
    assert H.FROZEN == {"games": 12, "n_simulations": 1600, "phase_bytes": 100 << 20}
    assert H.FROZEN["phase_bytes"] >= 12 * C.ENVELOPE_MAX_BYTES
    assert H.VERSIONS == {"mlx": "0.30.3", "numpy": "2.4.1", "python": "3.14.9"}
    assert (H.SETTING, H.SETTING_VALUE) == ("MLX_ENABLE_TF32", "0")
    assert REAL_PARENT.is_relative_to(os.path.expanduser("~")) and not REAL_PARENT.is_relative_to(H.ROOT)
    assert H.PLAN == "docs/superpowers/evidence/search-distillation-d2-heldout-run/plan.json"


@pytest.mark.parametrize("case, step", [("versions", "versions"), ("plan", "seeds"), ("seed", "seeds"),
                                        ("mirror", "mirror"), ("checkpoint", "checkpoint")])
def test_A_STOP_BEFORE_PLAY_writes_no_phase(tmp_path, case, step):
    over, versions_of = {"read_plan": lambda: PLAN}, OK
    if case == "versions":
        versions_of = lambda: {**H.VERSIONS, "mlx": "0.30.4"}           # noqa: E731
    elif case == "plan":
        over["read_plan"] = lambda: {"base": 100, "n_slots": 5}
    elif case == "seed":
        over["seed_ok"] = lambda s: s != 103
    elif case == "mirror":
        over["mirror_prob"] = lambda: 0.5
    elif case == "checkpoint":
        over["checkpoint_sha1"] = lambda: "0" * 40
    code, rec = REAL_CONCLUDE(SPEC, _seams(**over), tmp_path / "ev", tmp_path / "set", versions_of=versions_of)
    assert code == H.EXIT_VOID and rec["void"]["step"] == step and list((tmp_path / "set").iterdir()) == []
    assert rec["eligible"] is False and not (tmp_path / "ev" / "heldout.json").exists()


def test_SIGTERM_mid_phase_is_a_VOID_not_eligible_with_no_manifest_and_the_partial_games_kept(tmp_path):
    def play(evaluator, **kw):
        if kw["game_id"] == 2:
            os.kill(os.getpid(), signal.SIGTERM)
        return sp.play_game(evaluator, **kw)

    code, rec = REAL_CONCLUDE(SPEC, _seams(read_plan=lambda: PLAN, play_game=play), tmp_path / "ev",
                              tmp_path / "set", versions_of=OK)
    assert code == H.EXIT_VOID and rec["void"]["step"] == "terminated" and rec["eligible"] is False
    assert (tmp_path / "set" / H.PHASE_DIR / "games" / "game_1.json").exists()
    assert not (tmp_path / "set" / H.PHASE_DIR / "finalized.json").exists()
    assert not (tmp_path / "ev" / "heldout.json").exists() and len(rec["games"]) == 2
    assert json.loads((tmp_path / "ev" / "record.json").read_text())["status"] == "VOID"


def test_A_CAPTURE_FAILURE_stops_at_once_without_retry_or_top_up(tmp_path):
    calls = []

    def drifting(evaluator, **kw):
        calls.append(kw["game_id"])
        game = sp.play_game(evaluator, **kw)
        if kw["game_id"] == 1:
            evaluator.network.policy_head.bn.running_mean = mx.array([0.31, -0.2])
        return game

    code, rec = REAL_CONCLUDE(SPEC, _seams(read_plan=lambda: PLAN, play_game=drifting), tmp_path / "ev",
                              tmp_path / "set", versions_of=OK)
    assert code == H.EXIT_VOID and "no longer matches" in rec["void"]["detail"] and calls == [0, 1, 2, 3]
    assert not (tmp_path / "set" / H.PHASE_DIR / "finalized.json").exists()
    assert not (tmp_path / "ev" / "heldout.json").exists()


def test_EXISTING_OUTPUTS_ARE_NEVER_REUSED(tmp_path):
    (tmp_path / "set").mkdir()
    with pytest.raises(FileExistsError):
        REAL_CONCLUDE(SPEC, _seams(read_plan=lambda: PLAN), tmp_path / "ev", tmp_path / "set", versions_of=OK)


# ─────────────────────────────── refusals and pins ───────────────────────────────

def test_THE_GATE_IS_CLOSED_IN_THE_SOURCE_and_a_closed_gate_refuses(isolated, capsys):
    assert GATE.SEARCH_DISTILLATION_D2_HELDOUT_AUTHORIZED is False
    assert H.main(["run", "--checkpoint", str(_checkpoint())]) == H.EXIT_REFUSED
    assert "is closed" in capsys.readouterr().err and isolated["reached"] == []


def test_NO_TEST_HERE_CAN_REACH_THE_REAL_EVIDENCE_OR_CACHE(isolated):
    assert H.EVIDENCE_DIR.startswith(str(isolated["tmp"])) and H.HELDOUT_PARENT.is_relative_to(isolated["tmp"])
    assert H.conclude is not REAL_CONCLUDE


def test_ANYTHING_BUT_run_WITH_A_CHECKPOINT_IS_USAGE():
    assert H.main([]) == H.main(["run"]) == H.main(["go", "--checkpoint", "x"]) == H.EXIT_USAGE


def test_THE_PINS_ARE_THE_FILES_ON_DISK():
    assert M._sha((H.ROOT / H.CARD).read_bytes()) == H.CARD_SHA256
    for rel, sha in H.PINNED.items():
        assert M._sha((H.ROOT / rel).read_bytes()) == sha, rel
    assert set(H.CODE.values()) == set(H.PINNED)


@pytest.mark.parametrize("value", [None, "1", "", " 0"])
def test_ANY_SETTING_BUT_EXACTLY_0_REFUSES_even_with_the_gate_open(opened, monkeypatch, value, capsys):
    if value is None:
        monkeypatch.delenv("MLX_ENABLE_TF32")
    else:
        monkeypatch.setenv("MLX_ENABLE_TF32", value)
    assert H.main(["run", "--checkpoint", str(_checkpoint())]) == H.EXIT_REFUSED
    assert "MLX_ENABLE_TF32" in capsys.readouterr().err and opened["reached"] == []


@pytest.mark.parametrize("case, says", [("mlx", "imported"), ("card", "pinned card"), ("code", "pinned bytes"),
                                        ("relative", "absolute"), ("evidence", "one-shot"),
                                        ("parent", "not a directory")])
def test_EACH_REFUSAL_BEFORE_THE_BOUNDARY(opened, monkeypatch, capsys, case, says):
    ckpt = str(_checkpoint())
    if case == "mlx":
        monkeypatch.setattr(H, "_mlx_loaded", lambda: True)
    elif case == "card":
        monkeypatch.setattr(H, "CARD_SHA256", "0" * 64)
    elif case == "code":
        monkeypatch.setitem(H.PINNED, H.CODE["generate"], "0" * 64)
    elif case == "relative":
        ckpt = "checkpoints/x.safetensors"
    elif case == "evidence":
        (opened["tmp"] / "evidence").mkdir()
    elif case == "parent":
        monkeypatch.setattr(H, "HELDOUT_PARENT", opened["tmp"] / "absent")
    assert H.main(["run", "--checkpoint", ckpt]) == H.EXIT_REFUSED
    assert says in capsys.readouterr().err and opened["reached"] == []


def test_WITH_EVERY_CHECK_PASSING_THE_PUBLIC_ENTRY_REACHES_THE_BOUNDARY_ONCE(opened):
    assert H.main(["run", "--checkpoint", str(_checkpoint())]) == H.EXIT_DONE and opened["reached"] == [1]


# ─────────────────────────────── fresh processes ───────────────────────────────

PRELUDE = """
import pathlib, sys, tempfile
from scripts.GPU.alphazero import search_distillation_d2_heldout as H
from scripts.GPU.alphazero import search_distillation_d2_heldout_authorization as GATE
print("MLX_AFTER_IMPORT", H._mlx_loaded(), flush=True)
_tmp = pathlib.Path(tempfile.mkdtemp())
H.EVIDENCE_DIR = str(_tmp / "evidence")
H.HELDOUT_PARENT = _tmp
def stub(spec, seams, out, heldout, head=None):     # never a real run, whatever a mutant does above it
    print("CONCLUDE_REACHED", flush=True)
    return 0, {"status": "STUB", "head": head, "heldout_dir": str(heldout)}
H.conclude = stub
"""


def _fresh(body, setting):
    env = {k: v for k, v in os.environ.items() if k != "MLX_ENABLE_TF32"}
    env.update(PYTHONPATH=str(H.ROOT), PYTHONDONTWRITEBYTECODE="1")
    if setting is not None:
        env["MLX_ENABLE_TF32"] = setting
    argv = repr(["run", "--checkpoint", str(_checkpoint())])
    return subprocess.run([sys.executable, "-c", PRELUDE + body + f"\nsys.exit(H.main({argv}))\n"], cwd=H.ROOT,
                          env=env, capture_output=True, text=True, timeout=120)


def test_FRESH_importing_the_entry_loads_no_MLX_and_the_real_checks_pass_to_the_gate():
    p = _fresh("", "0")
    assert "MLX_AFTER_IMPORT False" in p.stdout, p.stdout + p.stderr
    assert p.returncode == 4 and "is closed" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


@pytest.mark.parametrize("setting", [None, "1"])
def test_FRESH_without_the_setting_the_real_check_refuses(setting):
    p = _fresh("GATE.SEARCH_DISTILLATION_D2_HELDOUT_AUTHORIZED = True", setting)
    assert p.returncode == 4 and "MLX_ENABLE_TF32" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


def test_FRESH_MLX_imported_before_the_check_refuses():
    p = _fresh("GATE.SEARCH_DISTILLATION_D2_HELDOUT_AUTHORIZED = True\nimport mlx.core", "0")
    assert p.returncode == 4 and "imported" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


def test_FRESH_with_the_setting_and_the_gate_open_every_real_check_passes():
    p = _fresh("GATE.SEARCH_DISTILLATION_D2_HELDOUT_AUTHORIZED = True", "0")
    assert p.returncode == 0 and "CONCLUDE_REACHED" in p.stdout, p.stderr


def test_THE_REAL_VERSIONS_ARE_THE_CARD_S():
    assert H.versions() == H.VERSIONS


# ─────────────────────────────── the REAL registered plan ───────────────────────────────

def test_REAL_THE_REGISTERED_PLAN_through_the_entry_s_own_seams_is_12_usable_seeds_and_the_survey_s_build_plan():
    """After S3 (registration) and S4 (post-registration proof): the plan the public entry will read, and every seed
    the entry's own seed check will judge, in the REAL registry. Read-only; nothing is drawn or exposed."""
    import importlib.util
    seams = H._real_seams(_checkpoint())
    plan = seams.read_plan()
    assert plan["n_slots"] == H.FROZEN["games"] == 12 and type(plan["base"]) is int
    assert all(seams.seed_ok(plan["base"] + s) for s in range(12))
    here = H.ROOT / "docs/superpowers/evidence/search-distillation-d2-heldout-seed-survey"
    spec = importlib.util.spec_from_file_location("_heldout_survey", here / "01_seed_survey.py")
    survey = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(survey)
    record = (here / "02_seed_survey_run.txt").read_bytes()
    assert plan == survey.build_plan(plan["base"], hashlib.sha256(record).hexdigest())
    assert f"block = [{plan['base']}, {plan['base'] + 12})" in record.decode()
    from scripts.GPU.alphazero import e4_screen_reference as REG
    assert REG.ACCOUNTED_SEED_INTERVALS[-1] == (plan["base"], plan["base"] + 12)
    assert not REG.seed_is_unavailable(plan["base"]) and not REG.seed_is_test_only(plan["base"] + 11)
