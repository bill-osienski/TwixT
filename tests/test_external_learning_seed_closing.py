"""External-engine learning stage 1's CLOSING SEED ACCOUNTING -- the derivation, SYNTHETIC ONLY.

The derivation script is imported by path and never run against the real evidence here. Its whitelisted
reads, its reconciliation, its controls and its entry are exercised on synthetic worlds; guarded mappings
prove that no field outside the whitelist is touched. No band value is spelled.
"""
import importlib.util
import json
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/superpowers/evidence/external-engine-learning-seed-closing/01_derive_usage.py"


@pytest.fixture(scope="module")
def M():
    spec = importlib.util.spec_from_file_location("_el_seed_closing", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Guarded(dict):
    """A mapping that FAILS the test when a key outside `allowed` is read."""
    def __init__(self, data, allowed):
        super().__init__(data)
        self.allowed = set(allowed)

    def __getitem__(self, k):
        if k not in self.allowed:
            raise AssertionError(f"read outside the whitelist: {k!r}")
        return super().__getitem__(k)

    def get(self, k, default=None):
        return self[k] if k in self else default

    def __iter__(self):
        raise AssertionError("iterated a guarded mapping")

    def items(self):
        raise AssertionError("iterated a guarded mapping")

    def values(self):
        raise AssertionError("iterated a guarded mapping")


# ─────────────────────────────── the script ───────────────────────────────

def test_the_script_and_this_file_SPELL_no_band_literal():
    spec = importlib.util.spec_from_file_location("_s", ROOT / "docs/superpowers/evidence/"
                                                  "tactical-candidate-seed-survey/01_seed_survey.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    for path in (SCRIPT, pathlib.Path(__file__)):
        assert S.scan(path.read_text()) == {}, path


def test_the_PATHS_and_the_plan_PIN(M):
    from scripts.GPU.alphazero import external_learning as EL
    assert (M.ROOT / M.HERE / "01_derive_usage.py").resolve() == SCRIPT
    assert M.OUT == f"{M.HERE}/02_derive_usage_run.txt"
    assert (M.RUN, M.PLAN) == (EL.EVIDENCE_DIR, EL.PLAN)
    assert M._sha((ROOT / M.PLAN).read_bytes()) == M.PLAN_SHA256


def test_the_RUN_controls_all_PASS(M):
    results = M._controls()
    assert len(results) >= 20 and [label for label, ok, _ in results if not ok] == []


# ─────────────────────────────── the whitelisted reads ───────────────────────────────

def test_seed_fields_reads_ONLY_game_id_family_and_seed(M):
    game = Guarded({"game_id": "f0-r", "family": 0, "seed": 7, "moves": [[1, 1]], "winner": "red", "result": "x",
                    "rows": []}, ("game_id", "family", "seed"))
    assert M.seed_fields(game) == {"game_id": "f0-r", "family": 0, "seed": 7}


def test_game_starts_reads_ONLY_the_event_and_a_game_starts_identity(M, monkeypatch):
    entries = [{"event": "step", "step": "plan", "value": {"x": 1}},
               {"event": "game_start", "game_id": "f0-r", "seed": 3, "opening": [], "family": 0},
               {"event": "move", "game_id": "f0-r", "move": [1, 1], "ply": 4},
               {"event": "game_start", "game_id": "f0-b", "seed": 4, "opening": [], "family": 0},
               {"event": "step", "step": "games", "value": [{"result": "x"}]}]
    lines = [json.dumps(e) for e in entries] + [""]
    real = json.loads

    def loads(s):
        e = real(s)
        return Guarded(e, ("event", "game_id", "seed") if e["event"] == "game_start" else ("event",))
    monkeypatch.setattr(M.json, "loads", loads)
    assert M.game_starts(lines) == [("f0-r", 3), ("f0-b", 4)]


def test_registry_intervals_PARSES_the_four_and_refuses_a_double_assignment(M):
    src = ("ACCOUNTED_SEED_INTERVALS = ((1, 3), (5, 9))\nEXPOSED_SEED_INTERVALS = ((1, 3),)\n"
           "RETIRED_SEED_INTERVALS = ()\nTEST_ONLY_SEED_INTERVALS = ((-9, -1),)\n")
    assert M.registry_intervals(src)["ACCOUNTED_SEED_INTERVALS"] == ((1, 3), (5, 9))
    with pytest.raises(M.Refused, match="exactly once"):
        M.registry_intervals(src + "EXPOSED_SEED_INTERVALS = ()\n")


# ─────────────────────────────── the reconciliation ───────────────────────────────

def test_a_SOUND_world_reconciles_to_its_block(M):
    faults, block = M.reconcile(*M._world(families=6, lo=700))
    assert faults == [] and block == (700, 718)


@pytest.mark.parametrize("i,edit,words", [
    (1, lambda v: v[:-1], "game starts are not the plan's"),
    (1, lambda v: v + v[-1:], "a game started more than once"),
    (2, lambda v: v[1:], "the game files' seed fields are not the plan's games"),
    (4, lambda v: {**v, "0": [[9, 9]]}, "a recorded opening is not the one its plan seed draws"),
    (3, lambda v: {k: x for k, x in v.items() if k != "0"}, "the record's openings are not the plan's families"),
    (6, lambda v: ("EXITED", None), "not finalized with its group reaped"),
])
def test_each_fault_is_NAMED(M, i, edit, words):
    w = list(M._world())
    w[i] = edit(w[i])
    faults, _block = M.reconcile(*w)
    assert any(words in f for f in faults), faults


def test_main_takes_no_arguments(M):
    with pytest.raises(M.Refused, match="no arguments"):
        M.main(["x", "y"])


def test_INPUTS_refuse_an_existing_record_and_a_dirty_tree(M, monkeypatch, tmp_path):
    for args in (["init", "-q"], ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                                  "commit", "-q", "--allow-empty", "-m", "base"]):
        assert subprocess.run(["git", "-C", str(tmp_path), *args], capture_output=True).returncode == 0
    monkeypatch.setattr(M, "ROOT", tmp_path)
    (tmp_path / "stray.txt").write_text("x")
    with pytest.raises(M.Refused, match="not clean"):
        M._inputs()
    (tmp_path / M.HERE).mkdir(parents=True)
    (tmp_path / M.OUT).write_text("x")
    with pytest.raises(M.Refused, match="runs once"):
        M._inputs()


# ─────────────────────────────── the inputs, on a synthetic evidence tree ───────────────────────────────

@pytest.fixture
def evidence(M, monkeypatch, tmp_path):
    """The run's layout with the REAL committed plan (its pin must hold), synthetic games and journal carrying
    the plan's seeds, openings drawn from the plan's opening seeds, a manifest, and a registry ACCOUNTING the
    plan's block -- committed in a temporary repository."""
    from scripts.GPU.alphazero import external_learning as EL
    from scripts.GPU.alphazero import h4_pilot_feasibility as F
    plan_raw = (ROOT / M.PLAN).read_bytes()
    plan = json.loads(plan_raw)
    (tmp_path / M.PLAN).parent.mkdir(parents=True)
    (tmp_path / M.PLAN).write_bytes(plan_raw)
    d = tmp_path / M.RUN
    (d / "games").mkdir(parents=True)
    (tmp_path / M.HERE).mkdir(parents=True)                                    # the script's own directory
    journal = []
    for g in plan["games"]:
        (d / "games" / f"{g['game_id']}.json").write_text(json.dumps({**g, "moves": [], "winner": None}))
        journal.append(json.dumps({"event": "game_start", **g, "opening": []}))
        journal.append(json.dumps({"event": "move", "game_id": g["game_id"], "move": [1, 1]}))
    openings = {str(o["family"]): [list(m) for m in EL.generate_opening(o["opening_seed"])] for o in plan["openings"]}
    (d / "record.json").write_text(json.dumps({"steps": {"plan": {"sha256": M.PLAN_SHA256}, "openings": openings}}))
    (d / "finalized.json").write_text(json.dumps({"status": "EXITED", "group_reaped": True, "exit_code": 0,
                                                  "verdict": "PASS"}))
    (d / "progress.jsonl").write_text("\n".join(journal) + "\n")
    lo, hi = EL.plan_block(plan)
    (tmp_path / M.REGISTRY).parent.mkdir(parents=True)
    (tmp_path / M.REGISTRY).write_text(f"ACCOUNTED_SEED_INTERVALS = ((1, 2), ({lo}, {hi}))\n"
                                       "EXPOSED_SEED_INTERVALS = ((1, 2),)\nRETIRED_SEED_INTERVALS = ((1, 2),)\n"
                                       "TEST_ONLY_SEED_INTERVALS = ()\n")

    def manifest():
        lines = [f"{M._sha(p.read_bytes())}  {p.relative_to(d)}\n" for p in sorted(d.rglob("*"))
                 if p.is_file() and p.name != "MANIFEST.sha256"]
        (d / "MANIFEST.sha256").write_text("".join(lines))

    def commit():
        for args in (["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                                     "-c", "core.hooksPath=/dev/null", "commit", "-q", "-m", "x"]):
            assert subprocess.run(["git", "-C", str(tmp_path), *args], capture_output=True).returncode == 0
    manifest()
    assert subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], capture_output=True).returncode == 0
    commit()
    durable = []
    monkeypatch.setattr(M, "ROOT", tmp_path)
    monkeypatch.setattr(F, "check_durable", lambda paths, _fixture: durable.append(list(paths)))
    return tmp_path, plan, (lo, hi), commit, manifest, durable


def test_INPUTS_read_a_sound_tree_and_it_RECONCILES(M, evidence):
    _root, plan, block, _commit, _manifest, durable = evidence
    x = M._inputs()
    assert len(x["starts"]) == len(x["games"]) == len(plan["games"]) == x["files"] - 6
    assert len(durable) == 1 and len(durable[0]) == x["files"]                   # every input, durability-checked
    assert M.reconcile(x["plan"], x["starts"], x["games"], x["recorded"], x["drawn"], x["registries"],
                       x["final"]) == ([], block)


def test_INPUTS_refuse_a_run_file_that_is_not_the_one_the_MANIFEST_HASHED(M, evidence):
    root, plan, _block, commit, _manifest, _d = evidence
    gid = plan["games"][3]["game_id"]
    f = root / M.RUN / "games" / f"{gid}.json"
    f.write_text(f.read_text().replace('"moves": []', '"moves": [[1, 1]]'))
    commit()
    with pytest.raises(M.Refused, match=f"games/{gid}.json is not the file the manifest hashed"):
        M._inputs()


def test_INPUTS_refuse_a_plan_that_is_NOT_THE_PIN_and_a_run_of_another_plan(M, evidence):
    root, plan, _block, commit, manifest, _d = evidence
    rec = json.loads((root / M.RUN / "record.json").read_text())
    rec["steps"]["plan"]["sha256"] = "0" * 64
    (root / M.RUN / "record.json").write_text(json.dumps(rec))
    manifest()
    commit()
    with pytest.raises(M.Refused, match="ran another plan"):
        M._inputs()
    (root / M.PLAN).write_text(json.dumps(plan))                                # same content, other bytes
    commit()
    with pytest.raises(M.Refused, match="not stage 1's committed plan"):
        M._inputs()


def test_INPUTS_refuse_an_UNCOMMITTED_input(M, evidence, monkeypatch):
    from scripts.GPU.alphazero import h4_pilot_feasibility as F

    def refuse(paths, _fixture):
        raise F.H4AnalysisRefused("not durable here")
    monkeypatch.setattr(F, "check_durable", refuse)
    with pytest.raises(M.Refused, match="not committed and unmodified: not durable here"):
        M._inputs()


def test_MAIN_on_a_sound_tree_RECONCILES_and_writes_its_record_ONCE(M, evidence, monkeypatch):
    root, _plan, block, _commit, _manifest, _d = evidence
    monkeypatch.setattr(M, "_LINES", [])
    assert M.main(["script"]) == 0
    rec = (root / M.OUT).read_text()
    assert f"the block the used seeds make: {block};" in rec and "faults: none" in rec
    assert "VERDICT: RECONCILES" in rec and "EXPOSED WHOLE and RETIRED WHOLE" in rec
    with pytest.raises(FileExistsError):
        M._write(root / M.OUT)                                                   # create-only


def test_MAIN_on_a_block_already_EXPOSED_does_NOT_reconcile(M, evidence, monkeypatch):
    root, _plan, (lo, hi), commit, _manifest, _d = evidence
    reg = root / M.REGISTRY
    reg.write_text(reg.read_text().replace("EXPOSED_SEED_INTERVALS = ((1, 2),)",
                                           f"EXPOSED_SEED_INTERVALS = ((1, 2), ({lo}, {lo + 1}))"))
    commit()
    monkeypatch.setattr(M, "_LINES", [])
    assert M.main(["script"]) == 1
    rec = (root / M.OUT).read_text()
    assert "already in EXPOSED_SEED_INTERVALS" in rec and "VERDICT: DOES NOT RECONCILE" in rec


# ─────────────────────────────── THE CLOSED STATE: the one closing edit (2026-10-09) ───────────────────────────────

def _closed_block(M):
    """The block, READ from the committed derivation record -- never spelled here."""
    import re
    rec = (ROOT / M.HERE / "02_derive_usage_run.txt").read_text()
    assert "VERDICT: RECONCILES" in rec
    m = re.search(r"^the block the used seeds make: \((\d+), (\d+)\);", rec, re.M)
    return int(m.group(1)), int(m.group(2))


def test_the_stage1_block_is_ACCOUNTED_EXPOSED_and_RETIRED_exactly_as_itself(M):
    from scripts.GPU.alphazero import e4_screen_reference as REF
    lo, hi = block = _closed_block(M)
    for name in ("ACCOUNTED_SEED_INTERVALS", "EXPOSED_SEED_INTERVALS", "RETIRED_SEED_INTERVALS"):
        intervals = getattr(REF, name)
        short = name.split("_")[0]
        assert intervals.count(block) == 1, f"the stage-1 block is not registered as exactly itself in {short}"
        assert [iv for iv in intervals if iv[0] < hi and lo < iv[1]] == [block], \
            f"another interval overlaps the stage-1 block in {short}"
    ex, rt = REF.EXPOSED_SEED_INTERVALS, REF.RETIRED_SEED_INTERVALS
    assert ex[ex.index(block):] == rt[rt.index(block):] and all(iv[0] >= hi for iv in ex[ex.index(block) + 1:]), \
        "the stage-1 block is not the LAST entry of EXPOSED and RETIRED"
    assert not any(iv[0] <= lo - 1 < iv[1] or iv[0] <= hi < iv[1] for name in
                   ("ACCOUNTED_SEED_INTERVALS", "EXPOSED_SEED_INTERVALS", "RETIRED_SEED_INTERVALS", "TEST_ONLY_SEED_INTERVALS")
                   for iv in getattr(REF, name)), "a registry interval reaches just outside the stage-1 block"
    for seed in range(lo, hi):
        st = REF.seed_status(seed)
        assert (st["accounted"], st["exposed"], st["retired"], st["test_only"]) == (True, True, True, False), \
            f"stage-1 seed lo+{seed - lo} is not accounted, exposed and retired"


def test_every_stage1_seed_now_FAULTS_as_exposed_and_retired(M):
    from scripts.GPU.alphazero import h4_runner as R4
    lo, hi = _closed_block(M)
    assert R4.seed_registry_faults([{"seed": s} for s in range(lo, hi)]) == \
        [f"{s}: exposed, retired" for s in range(lo, hi)], "a stage-1 seed does not fault as exactly exposed and retired"


def test_check_plan_REFUSES_the_committed_stage1_plan_as_SPENT(M):
    from scripts.GPU.alphazero import external_learning as EL
    plan = json.loads((ROOT / M.PLAN).read_bytes())
    assert EL.plan_block(plan) == _closed_block(M)
    with pytest.raises(EL.Stage1Stop, match="is spent"):
        EL.check_plan(plan)
