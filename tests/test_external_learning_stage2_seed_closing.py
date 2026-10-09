"""External-engine learning stage 2's CLOSING SEED ACCOUNTING -- the derivation, SYNTHETIC ONLY.

The derivation script is imported by path and never run against the real evidence here. Its whitelisted
reads, its reconciliation, its controls and its entry are exercised on synthetic worlds. No band value is
spelled.
"""
import importlib.util
import json
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/superpowers/evidence/external-engine-learning-stage2-seed-closing/01_derive_usage.py"
SYNTHETIC = 1_000_000                                    # never a 2026-band seed


@pytest.fixture(scope="module")
def M():
    spec = importlib.util.spec_from_file_location("_el2_seed_closing", SCRIPT)
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

    def __iter__(self):
        raise AssertionError("iterated a guarded mapping")


def test_the_script_and_this_file_SPELL_no_band_literal():
    spec = importlib.util.spec_from_file_location("_s", ROOT / "docs/superpowers/evidence/"
                                                  "tactical-candidate-seed-survey/01_seed_survey.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    for path in (SCRIPT, pathlib.Path(__file__)):
        assert S.scan(path.read_text()) == {}, path


def test_the_PATHS_and_the_plan_PIN(M):
    from scripts.GPU.alphazero import external_learning_stage2 as S2
    assert (M.ROOT / M.HERE / "01_derive_usage.py").resolve() == SCRIPT
    assert (M.RUN, M.PLAN, M.OUT) == (S2.EVIDENCE_DIR, S2.PLAN, f"{M.HERE}/02_derive_usage_run.txt")
    assert M._sha((ROOT / M.PLAN).read_bytes()) == M.PLAN_SHA256


def test_the_RUN_controls_all_PASS(M):
    results = M._controls()
    assert len(results) >= 17 and [label for label, ok, _ in results if not ok] == []


def test_record_fields_reads_ONLY_the_named_fields(M):
    steps = Guarded({"plan": Guarded({"sha256": "p", "n_slots": 1}, ("sha256",)), "schedule_sha256": "s",
                     "inputs": Guarded({"train_rows": 9, "rows_sha256": "r"}, ("train_rows",)),
                     "training": Guarded({"steps": 4, "last_loss": 1.0}, ("steps",)), "policy_change": {}},
                    ("plan", "schedule_sha256", "inputs", "training"))
    rec = Guarded({"verdict": "PASS", "steps": steps, "card": "c"}, ("verdict", "steps"))
    assert M.record_fields(rec) == {"verdict": "PASS", "plan_sha256": "p", "schedule_sha256": "s", "train_rows": 9,
                                    "steps": 4}


def test_train_steps_reads_ONLY_the_event_and_a_train_steps_number(M, monkeypatch):
    entries = [{"event": "step", "step": "control", "value": {}}, {"event": "train_step", "step": 0, "loss": 2.0},
               {"event": "train_step", "step": 1, "loss": 1.0, "grad_norm": 3.0}]
    lines = [json.dumps(e) for e in entries] + [""]
    real = json.loads
    monkeypatch.setattr(M.json, "loads", lambda s: (lambda e: Guarded(
        e, ("event", "step") if e["event"] == "train_step" else ("event",)))(real(s)))
    assert M.train_steps(lines) == [0, 1]


@pytest.mark.parametrize("i,edit,words", [
    (1, lambda r: {**r, "plan_sha256": "0" * 64}, "ran another plan"),
    (2, lambda s: "t" * 64, "the schedule the plan's seed draws is not the run's"),
    (3, lambda v: v[:-1], "logged train steps"),
    (5, lambda r: {**r, "ACCOUNTED_SEED_INTERVALS": ()}, "not ACCOUNTED exactly as itself"),
    (6, lambda f: ("EXITED", False), "not finalized with its group reaped"),
])
def test_each_fault_is_NAMED(M, i, edit, words):
    w = list(M._world())
    w[i] = edit(w[i])
    faults, _block = M.reconcile(*w)
    assert any(words in f for f in faults), faults


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
    """The run's layout with the REAL committed plan (its pin must hold), a record whose schedule hash is the
    one the plan's seed draws, 128 logged steps, a manifest, and a registry ACCOUNTING the seed -- committed
    in a temporary repository."""
    from scripts.GPU.alphazero import external_learning_stage2 as S2
    from scripts.GPU.alphazero import h4_pilot_feasibility as F
    plan_raw = (ROOT / M.PLAN).read_bytes()
    plan = json.loads(plan_raw)
    (tmp_path / M.PLAN).parent.mkdir(parents=True)
    (tmp_path / M.PLAN).write_bytes(plan_raw)
    (tmp_path / M.HERE).mkdir(parents=True)
    d = tmp_path / M.RUN
    d.mkdir(parents=True)
    sched = M._jsha(S2.sampling_schedule(plan["seed"], 128).tolist())
    (d / "record.json").write_text(json.dumps({"verdict": "PASS", "steps": {
        "plan": {"sha256": M.PLAN_SHA256}, "schedule_sha256": sched, "inputs": {"train_rows": 128},
        "training": {"steps": 128}}}))
    (d / "finalized.json").write_text(json.dumps({"status": "EXITED", "group_reaped": True}))
    (d / "progress.jsonl").write_text("".join(json.dumps({"event": "train_step", "step": k, "loss": 1.0}) + "\n"
                                              for k in range(128)))
    seed = plan["seed"]
    (tmp_path / M.REGISTRY).parent.mkdir(parents=True)
    (tmp_path / M.REGISTRY).write_text(f"ACCOUNTED_SEED_INTERVALS = ((1, 2), ({seed}, {seed + 1}))\n"
                                       "EXPOSED_SEED_INTERVALS = ((1, 2),)\nRETIRED_SEED_INTERVALS = ((1, 2),)\n"
                                       "TEST_ONLY_SEED_INTERVALS = ()\n")

    def manifest():
        (d / "MANIFEST.sha256").write_text("".join(f"{M._sha((d / n).read_bytes())}  {n}\n"
                                                   for n in ("finalized.json", "progress.jsonl", "record.json")))

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
    return tmp_path, plan, (seed, seed + 1), commit, manifest, durable


def test_MAIN_on_a_sound_tree_RECONCILES_and_writes_its_record_ONCE(M, evidence, monkeypatch):
    root, _plan, block, _commit, _manifest, durable = evidence
    monkeypatch.setattr(M, "_LINES", [])
    assert M.main(["script"]) == 0
    assert len(durable) == 1 and len(durable[0]) == 6
    rec = (root / M.OUT).read_text()
    assert f"the block of the seed: {block};" in rec and "faults: none" in rec and "VERDICT: RECONCILES" in rec
    with pytest.raises(FileExistsError):
        M._write(root / M.OUT)


def test_MAIN_on_another_schedule_does_NOT_reconcile(M, evidence, monkeypatch):
    root, _plan, _block, commit, manifest, _d = evidence
    rec = json.loads((root / M.RUN / "record.json").read_text())
    rec["steps"]["schedule_sha256"] = "0" * 64
    (root / M.RUN / "record.json").write_text(json.dumps(rec))
    manifest()
    commit()
    monkeypatch.setattr(M, "_LINES", [])
    assert M.main(["script"]) == 1
    assert "VERDICT: DOES NOT RECONCILE" in (root / M.OUT).read_text()


def test_INPUTS_refuse_a_file_the_manifest_did_not_hash_and_a_plan_that_is_not_the_pin(M, evidence):
    root, plan, _block, commit, _manifest, _d = evidence
    (root / M.RUN / "progress.jsonl").write_text("")
    commit()
    with pytest.raises(M.Refused, match="progress.jsonl is not the file the manifest hashed"):
        M._inputs()
    (root / M.PLAN).write_text(json.dumps(plan))                                # same content, other bytes
    commit()
    with pytest.raises(M.Refused, match="not stage 2's committed plan"):
        M._inputs()


def test_INPUTS_refuse_an_UNCOMMITTED_input(M, evidence, monkeypatch):
    from scripts.GPU.alphazero import h4_pilot_feasibility as F

    def refuse(paths, _fixture):
        raise F.H4AnalysisRefused("not durable here")
    monkeypatch.setattr(F, "check_durable", refuse)
    with pytest.raises(M.Refused, match="not committed and unmodified: not durable here"):
        M._inputs()


# ─────────────────────────────── THE CLOSED STATE: the one closing edit (2026-10-09) ───────────────────────────────

def _closed_block(M):
    """The block, READ from the committed derivation record -- never spelled here."""
    import re
    rec = (ROOT / M.HERE / "02_derive_usage_run.txt").read_text()
    assert "VERDICT: RECONCILES" in rec
    m = re.search(r"^the block of the seed: \((\d+), (\d+)\);", rec, re.M)
    return int(m.group(1)), int(m.group(2))


def test_the_stage2_seed_is_ACCOUNTED_EXPOSED_and_RETIRED_exactly_as_itself(M):
    from scripts.GPU.alphazero import e4_screen_reference as REF
    lo, hi = block = _closed_block(M)
    assert hi == lo + 1
    for name in ("ACCOUNTED_SEED_INTERVALS", "EXPOSED_SEED_INTERVALS", "RETIRED_SEED_INTERVALS"):
        intervals = getattr(REF, name)
        short = name.split("_")[0]
        assert intervals.count(block) == 1, f"the stage-2 seed is not registered as exactly itself in {short}"
        assert [iv for iv in intervals if iv[0] < hi and lo < iv[1]] == [block], \
            f"another interval overlaps the stage-2 seed in {short}"
    ex, rt = REF.EXPOSED_SEED_INTERVALS, REF.RETIRED_SEED_INTERVALS
    assert ex[ex.index(block):] == rt[rt.index(block):] and all(iv[0] >= hi for iv in ex[ex.index(block) + 1:]), \
        "the stage-2 seed is not the LAST entry of EXPOSED and RETIRED"
    st = REF.seed_status(lo)
    assert (st["accounted"], st["exposed"], st["retired"], st["test_only"]) == (True, True, True, False), \
        "the stage-2 seed is not accounted, exposed and retired"


def test_the_stage2_seed_now_FAULTS_and_check_plan_REFUSES_the_committed_plan_as_SPENT(M):
    from scripts.GPU.alphazero import external_learning_stage2 as S2
    from scripts.GPU.alphazero import h4_runner as R4
    lo, _hi = _closed_block(M)
    assert R4.seed_registry_faults([{"seed": lo}]) == [f"{lo}: exposed, retired"]
    plan = json.loads((ROOT / M.PLAN).read_bytes())
    assert plan["seed"] == lo
    with pytest.raises(S2.Stop, match="is spent"):
        S2.check_plan(plan)
