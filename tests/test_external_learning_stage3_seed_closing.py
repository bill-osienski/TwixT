"""External-engine learning stage 3's CLOSING SEED ACCOUNTING -- the derivation, SYNTHETIC ONLY.

The derivation script is imported by path and never run against the real evidence here. Its whitelisted
reads, its reconciliation (interrupted games count as used), its controls and its entry are exercised on
synthetic worlds; guarded mappings prove that no field outside the whitelist is touched. No band value is spelled.
"""
import importlib.util
import json
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/superpowers/evidence/external-engine-learning-stage3-seed-closing/01_derive_usage.py"


@pytest.fixture(scope="module")
def M():
    spec = importlib.util.spec_from_file_location("_el3_seed_closing", SCRIPT)
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


def test_the_script_and_this_file_SPELL_no_band_literal():
    spec = importlib.util.spec_from_file_location("_s", ROOT / "docs/superpowers/evidence/"
                                                  "tactical-candidate-seed-survey/01_seed_survey.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    for path in (SCRIPT, pathlib.Path(__file__)):
        assert S.scan(path.read_text()) == {}, path


def test_the_PATHS_and_the_plan_PIN(M):
    from scripts.GPU.alphazero import external_learning_stage3 as S3
    assert (M.ROOT / M.HERE / "01_derive_usage.py").resolve() == SCRIPT
    assert M.OUT == f"{M.HERE}/02_derive_usage_run.txt"
    assert (M.RUN, M.PLAN, S3.REPLAYS) == (S3.EVIDENCE_DIR, S3.PLAN, "games")
    assert M._sha((ROOT / M.PLAN).read_bytes()) == M.PLAN_SHA256


def test_the_RUN_controls_all_PASS(M):
    results = M._controls()
    assert len(results) >= 20 and [label for label, ok, _ in results if not ok] == []


def test_seed_fields_reads_ONLY_game_idx_and_seed(M):
    replay = Guarded({"game_idx": 3, "seed": 7, "task_id": 3, "moves": [[1, 1]], "winner": "red"},
                     ("game_idx", "seed"))
    assert M.seed_fields(replay) == {"game": 3, "seed": 7}


def test_game_starts_reads_ONLY_the_event_and_a_game_starts_identity(M, monkeypatch):
    entries = [{"event": "step", "step": "heldout", "value": {"passes": True}},
               {"event": "game_start", "game": 0, "seed": 3, "red": "candidate"},
               {"event": "game_end", "game": 0, "winner_agent_id": "candidate", "reason": "win"},
               {"event": "game_start", "game": 1, "seed": 4, "red": "incumbent"},
               {"event": "step", "step": "screen", "value": {"outcome": "x"}}]
    lines = [json.dumps(e) for e in entries] + [""]
    real = json.loads

    def loads(s):
        e = real(s)
        return Guarded(e, ("event", "game", "seed") if e["event"] == "game_start" else ("event",))
    monkeypatch.setattr(M.json, "loads", loads)
    assert M.game_starts(lines) == [(0, 3), (1, 4)]


def test_registry_intervals_PARSES_the_four_and_refuses_a_double_assignment(M):
    src = ("ACCOUNTED_SEED_INTERVALS = ((1, 3), (5, 9))\nEXPOSED_SEED_INTERVALS = ((1, 3),)\n"
           "RETIRED_SEED_INTERVALS = ()\nTEST_ONLY_SEED_INTERVALS = ((-9, -1),)\n")
    assert M.registry_intervals(src)["ACCOUNTED_SEED_INTERVALS"] == ((1, 3), (5, 9))
    with pytest.raises(M.Refused, match="exactly once"):
        M.registry_intervals(src + "EXPOSED_SEED_INTERVALS = ()\n")


@pytest.mark.parametrize("k,status,exposed", [(32, "EXITED", (700, 732)), (5, "TIMEOUT", (700, 705)),
                                              (0, "EXITED", None)])
def test_EXPOSURE_is_every_STARTED_game_and_RETIREMENT_the_whole_block(M, k, status, exposed):
    plan, starts, replays, regs, _f = M._world(n=32, lo=700, k=k)
    faults, exp, ret = M.reconcile(plan, starts, replays[:max(k - 1, 0)], regs, (status, True))  # last one unfinished
    assert (faults, exp, ret) == ([], exposed, (700, 732))


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


@pytest.fixture
def evidence(M, monkeypatch, tmp_path):
    """The run's layout with the REAL committed plan (its pin must hold), synthetic replays and journal
    carrying the plan's seeds, a manifest, and a registry ACCOUNTING the plan's block -- committed in a
    temporary repository."""
    from scripts.GPU.alphazero import h4_pilot_feasibility as F
    plan_raw = (ROOT / M.PLAN).read_bytes()
    plan = json.loads(plan_raw)
    (tmp_path / M.PLAN).parent.mkdir(parents=True)
    (tmp_path / M.PLAN).write_bytes(plan_raw)
    d = tmp_path / M.RUN
    (d / "games").mkdir(parents=True)
    (tmp_path / M.HERE).mkdir(parents=True)
    journal = []
    for g in plan["games"]:
        (d / "games" / f"game_{g['game']:06d}.json").write_text(json.dumps(
            {"game_idx": g["game"], "task_id": g["game"], "seed": g["seed"], "moves": [], "winner": None}))
        journal.append(json.dumps({"event": "game_start", "game": g["game"], "seed": g["seed"], "red": "x"}))
        journal.append(json.dumps({"event": "game_end", "game": g["game"], "winner_agent_id": "x"}))
    (d / "record.json").write_text(json.dumps({"steps": {"plan": {"sha256": M.PLAN_SHA256}}}))
    (d / "finalized.json").write_text(json.dumps({"status": "EXITED", "group_reaped": True, "exit_code": 0,
                                                  "verdict": "SCREEN_COMPLETED"}))
    (d / "progress.jsonl").write_text("\n".join(journal) + "\n")
    lo, hi = plan["base"], plan["base"] + plan["n_slots"]
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
    assert len(x["starts"]) == len(x["replays"]) == len(plan["games"]) == x["files"] - 6
    assert len(durable) == 1 and len(durable[0]) == x["files"]
    assert M.reconcile(x["plan"], x["starts"], x["replays"], x["registries"], x["final"]) == ([], block, block)


def test_INPUTS_refuse_a_run_file_that_is_not_the_one_the_MANIFEST_HASHED(M, evidence):
    root, _plan, _block, commit, _manifest, _d = evidence
    f = root / M.RUN / "games" / "game_000003.json"
    f.write_text(f.read_text().replace('"moves": []', '"moves": [[1, 1]]'))
    commit()
    with pytest.raises(M.Refused, match="games/game_000003.json is not the file the manifest hashed"):
        M._inputs()


def test_INPUTS_refuse_a_plan_that_is_NOT_THE_PIN_and_a_run_of_another_plan(M, evidence):
    root, plan, _block, commit, manifest, _d = evidence
    (root / M.RUN / "record.json").write_text(json.dumps({"steps": {"plan": {"sha256": "0" * 64}}}))
    manifest()
    commit()
    with pytest.raises(M.Refused, match="ran another plan"):
        M._inputs()
    (root / M.PLAN).write_text(json.dumps(plan))                                # same content, other bytes
    commit()
    with pytest.raises(M.Refused, match="not stage 3's committed plan"):
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
    assert f"seeds used (every started game): {block}; the plan's block: {block}" in rec and "faults: none" in rec
    assert f"VERDICT: RECONCILES -- EXPOSED {block}; RETIRED the whole block {block}" in rec
    with pytest.raises(FileExistsError):
        M._write(root / M.OUT)


def test_MAIN_on_a_block_already_EXPOSED_does_NOT_reconcile(M, evidence, monkeypatch):
    root, _plan, (lo, _hi), commit, _manifest, _d = evidence
    reg = root / M.REGISTRY
    reg.write_text(reg.read_text().replace("EXPOSED_SEED_INTERVALS = ((1, 2),)",
                                           f"EXPOSED_SEED_INTERVALS = ((1, 2), ({lo}, {lo + 1}))"))
    commit()
    monkeypatch.setattr(M, "_LINES", [])
    assert M.main(["script"]) == 1
    rec = (root / M.OUT).read_text()
    assert "already in EXPOSED_SEED_INTERVALS" in rec and "VERDICT: DOES NOT RECONCILE" in rec


# ─────────────────────────────── THE CLOSED STATE: the one closing edit (2026-10-10) ───────────────────────────────

def _closed_block(M):
    """The block, READ from the committed derivation record -- never spelled here."""
    import re
    rec = (ROOT / M.HERE / "02_derive_usage_run.txt").read_text()
    assert "VERDICT: RECONCILES" in rec
    m = re.search(r"^seeds used \(every started game\): \((\d+), (\d+)\); the plan's block: \((\d+), (\d+)\)$", rec, re.M)
    assert m.group(1, 2) == m.group(3, 4), "the closing exposes less than the block: this edit assumed all started"
    return int(m.group(1)), int(m.group(2))


def test_the_stage3_block_is_ACCOUNTED_EXPOSED_and_RETIRED_exactly_as_itself(M):
    from scripts.GPU.alphazero import e4_screen_reference as REF
    lo, hi = block = _closed_block(M)
    assert hi == lo + 32
    for name in ("ACCOUNTED_SEED_INTERVALS", "EXPOSED_SEED_INTERVALS", "RETIRED_SEED_INTERVALS"):
        intervals = getattr(REF, name)
        short = name.split("_")[0]
        assert intervals.count(block) == 1, f"the stage-3 block is not registered as exactly itself in {short}"
        assert [iv for iv in intervals if iv[0] < hi and lo < iv[1]] == [block], \
            f"another interval overlaps the stage-3 block in {short}"
    ex, rt = REF.EXPOSED_SEED_INTERVALS, REF.RETIRED_SEED_INTERVALS
    assert ex[ex.index(block):] == rt[rt.index(block):] and all(iv[0] >= hi for iv in ex[ex.index(block) + 1:]), \
        "the stage-3 block is not the LAST entry of EXPOSED and RETIRED"
    for s in (lo, hi - 1):
        st = REF.seed_status(s)
        assert (st["accounted"], st["exposed"], st["retired"], st["test_only"]) == (True, True, True, False), \
            "the stage-3 block is not accounted, exposed and retired"


def test_the_stage3_block_now_FAULTS_and_check_plan_REFUSES_the_committed_plan_as_SPENT(M):
    from scripts.GPU.alphazero import external_learning_stage3 as S3
    from scripts.GPU.alphazero import h4_runner as R4
    lo, hi = _closed_block(M)
    assert R4.seed_registry_faults([{"seed": lo}, {"seed": hi - 1}]) == [f"{lo}: exposed, retired",
                                                                        f"{hi - 1}: exposed, retired"]
    plan = json.loads((ROOT / M.PLAN).read_bytes())
    assert plan["base"] == lo
    with pytest.raises(S3.Stop, match="already spent"):
        S3.check_plan(plan)
