"""The tactical-candidate pilot's CLOSING SEED ACCOUNTING (card §18 A8.2) -- the derivation, SYNTHETIC ONLY.

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
SCRIPT = ROOT / "docs/superpowers/evidence/2026-10-08-tactical-candidate-seed-closing/01_derive_usage.py"


@pytest.fixture(scope="module")
def M():
    spec = importlib.util.spec_from_file_location("_seed_closing", SCRIPT)
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
    assert (M.ROOT / M.HERE / "01_derive_usage.py").resolve() == SCRIPT
    assert M.OUT == f"{M.HERE}/02_derive_usage_run.txt"
    assert M._sha((ROOT / M.PLAN).read_bytes()) == M.PLAN_SHA256


def test_the_RUN_controls_all_PASS(M):
    results = M._controls()
    assert len(results) >= 18 and [label for label, ok, _ in results if not ok] == []


# ─────────────────────────────── the whitelisted reads ───────────────────────────────

def test_seed_fields_reads_ONLY_pair_arm_and_the_two_colour_seeds(M):
    game = Guarded({"pair_id": "p", "arm": "B", "moves": [[1, 1]], "terminal": {"reason": "win"}, "witnesses": [],
                    "seeds": Guarded({"red": 7, "black": 8}, ("red", "black"))}, ("pair_id", "arm", "seeds"))
    assert M.seed_fields(game) == {"pair_id": "p", "arm": "B", "seeds": {"red": 7, "black": 8}}


def test_game_starts_reads_ONLY_the_step_and_a_game_starts_identity(M, monkeypatch):
    entries = [{"step": "model_load", "observation": {"x": 1}},
               {"step": "game_start", "observation": {"pair_id": "p", "arm": "A"}},
               {"step": "game", "observation": {"digest": "d", "overrides": 0, "seconds": 1.0}},
               {"step": "game_start", "observation": {"pair_id": "p", "arm": "B"}},
               {"step": "pair", "observation": {"fired": False}}]
    lines = [json.dumps(e) for e in entries] + [""]
    real = json.loads

    def loads(s):
        e = real(s)
        obs_keys = ("pair_id", "arm") if e["step"] == "game_start" else ()
        return Guarded({"step": e["step"], "observation": Guarded(e["observation"], obs_keys)}, ("step", "observation"))
    monkeypatch.setattr(M.json, "loads", loads)
    assert M.game_starts(lines) == [("p", "A"), ("p", "B")]


def test_registry_intervals_PARSES_the_four_and_refuses_a_double_assignment(M):
    src = ("ACCOUNTED_SEED_INTERVALS = ((1, 3), (5, 9))\nEXPOSED_SEED_INTERVALS = ((1, 3),)\n"
           "RETIRED_SEED_INTERVALS = ()\nTEST_ONLY_SEED_INTERVALS = ((-9, -1),)\n")
    assert M.registry_intervals(src)["ACCOUNTED_SEED_INTERVALS"] == ((1, 3), (5, 9))
    with pytest.raises(M.Refused, match="exactly once"):
        M.registry_intervals(src + "EXPOSED_SEED_INTERVALS = ()\n")
    with pytest.raises(M.Refused, match="exactly once"):
        M.registry_intervals(src.replace("RETIRED_SEED_INTERVALS = ()\n", ""))


# ─────────────────────────────── the reconciliation ───────────────────────────────

def test_a_SOUND_world_reconciles_to_its_block(M):
    faults, block = M.reconcile(*M._world(n_pairs=16, lo=700))
    assert faults == [] and block == (700, 732)


@pytest.mark.parametrize("i,edit,words", [
    (1, lambda v: v[:-1], "game starts are not the plan's"),
    (1, lambda v: v + v[-1:], "a game started more than once"),
    (2, lambda v: v[1:], "the record's games are not the started games"),
    (3, lambda v: v[::-1], "the game files are not the record's games"),
    (5, lambda v: ("COMPLETE", "STOP", "COMPLETE"), "the run is not COMPLETE"),
])
def test_an_IDENTITY_or_STATUS_mismatch_is_a_FAULT(M, i, edit, words):
    w = list(M._world())
    w[i] = edit(w[i])
    assert any(words in f for f in M.reconcile(*w)[0])


def test_a_game_whose_seeds_are_not_its_PLAN_PAIRS_is_a_FAULT_even_when_both_arms_agree(M):
    w = list(M._world())
    w[3] = [dict(s, seeds=dict(s["seeds"], black=999)) if s["pair_id"] == "p-1" else s for s in w[3]]
    faults = M.reconcile(*w)[0]
    assert faults == ["game p-1-A: its seeds are not its plan pair's", "game p-1-B: its seeds are not its plan pair's"]


def test_colour_seeds_NOT_reused_across_the_two_arms_is_a_FAULT(M):
    w = list(M._world())
    w[3] = [dict(s, seeds={"red": s["seeds"]["black"], "black": s["seeds"]["red"]}) if s["arm"] == "B" else s
            for s in w[3]]
    faults = M.reconcile(*w)[0]
    assert "pair p-0: its colour seeds are not reused across both arms" in faults


@pytest.mark.parametrize("name,value,words", [
    ("ACCOUNTED_SEED_INTERVALS", ((100, 132),), "not ACCOUNTED exactly as itself"),
    ("ACCOUNTED_SEED_INTERVALS", ((100, 132), (500, 506), (500, 506)), "not ACCOUNTED exactly as itself"),
    ("ACCOUNTED_SEED_INTERVALS", ((100, 132), (500, 507)), "not ACCOUNTED exactly as itself"),
    ("EXPOSED_SEED_INTERVALS", ((505, 600),), "already in EXPOSED_SEED_INTERVALS"),
    ("RETIRED_SEED_INTERVALS", ((400, 501),), "already in RETIRED_SEED_INTERVALS"),
])
def test_the_REGISTRY_state_before_the_edit_is_required(M, name, value, words):
    w = list(M._world())
    w[4] = dict(w[4], **{name: value})
    assert any(words in f for f in M.reconcile(*w)[0])


@pytest.mark.parametrize("value", [((490, 500),), ((506, 520),)])
def test_an_ADJACENT_interval_is_no_overlap(M, value):
    w = list(M._world())
    w[4] = dict(w[4], EXPOSED_SEED_INTERVALS=value, RETIRED_SEED_INTERVALS=value)
    assert M.reconcile(*w)[0] == []


def test_seeds_that_are_NOT_one_contiguous_block_are_a_FAULT(M):
    w = list(M._world())
    w[0] = {"pairs": [dict(p, seeds=dict(p["seeds"], black=p["seeds"]["black"] + (2 if p["pair_id"] == "p-2" else 0)))
                      for p in w[0]["pairs"]]}
    w[3] = [dict(s, seeds=dict(next(p["seeds"] for p in w[0]["pairs"] if p["pair_id"] == s["pair_id"]))) for s in w[3]]
    assert any("not one contiguous block of 6" in f for f in M.reconcile(*w)[0])


# ─────────────────────────────── the entry ───────────────────────────────

def test_the_ENTRY_takes_no_arguments_and_reads_nothing_first(M, monkeypatch):
    monkeypatch.setattr(M, "_inputs", lambda: (_ for _ in ()).throw(AssertionError("read before refusing")))
    with pytest.raises(M.Refused, match="no arguments"):
        M.main(["x", "--force"])


@pytest.fixture
def entry(M, monkeypatch, tmp_path):
    (tmp_path / M.HERE).mkdir(parents=True)
    monkeypatch.setattr(M, "ROOT", tmp_path)
    monkeypatch.setattr(M, "_LINES", [])
    plan, starts, games, seeds, regs, statuses = M._world()
    x = {"head": "h", "plan": plan, "starts": starts, "games": games, "seeds": seeds, "registries": regs,
         "statuses": statuses, "files": 9}
    monkeypatch.setattr(M, "_inputs", lambda: x)
    return tmp_path, x


def test_a_RECONCILING_world_exits_0_and_records_EXPOSED_and_RETIRED_WHOLE(M, entry):
    root, _x = entry
    assert M.main(["x"]) == 0
    assert "VERDICT: RECONCILES -- all 6 seeds of (500, 506) started" in (root / M.OUT).read_text()


def test_a_FAULT_exits_1_and_records_NO_edit(M, entry):
    root, x = entry
    x["starts"] = x["starts"][:-1]
    assert M.main(["x"]) == 1
    assert "VERDICT: DOES NOT RECONCILE -- no registry edit" in (root / M.OUT).read_text()


def test_a_FAILED_control_exits_1(M, entry, monkeypatch):
    root, _x = entry
    monkeypatch.setattr(M, "_controls", lambda: [("c", True, ""), ("d", False, "")])
    assert M.main(["x"]) == 1 and "DOES NOT RECONCILE" in (root / M.OUT).read_text()


def test_NO_controls_never_reconcile(M, entry, monkeypatch):
    monkeypatch.setattr(M, "_controls", lambda: [])
    assert M.main(["x"]) == 1


def test_the_RECORD_is_CREATE_ONLY(M, monkeypatch, tmp_path):
    monkeypatch.setattr(M, "_LINES", ["a"])
    M._write(tmp_path / "r.txt")
    with pytest.raises(FileExistsError):
        M._write(tmp_path / "r.txt")


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
    """The pilot's layout with the REAL committed plan (its pin must hold), synthetic games carrying the plan's
    seeds, and a registry ACCOUNTING the plan's block -- committed in a temporary repository."""
    from scripts.GPU.alphazero import h4_pilot_feasibility as F
    plan_raw = (ROOT / M.PLAN).read_bytes()
    plan = json.loads(plan_raw)
    (tmp_path / M.PLAN).parent.mkdir(parents=True)
    (tmp_path / M.PLAN).write_bytes(plan_raw)
    d = tmp_path / M.PILOT
    (d / "games").mkdir(parents=True)
    games, journal = [], []
    for p in plan["pairs"]:
        for arm in M.ARMS:
            raw = (json.dumps({"pair_id": p["pair_id"], "arm": arm, "seeds": p["seeds"], "moves": [],
                               "terminal": {"reason": "cap", "ply": 0}}) + "\n").encode()
            name = f"{p['pair_id']}-{arm}.json"
            (d / "games" / name).write_bytes(raw)
            games.append({"pair_id": p["pair_id"], "arm": arm, "digest": "d", "file": name, "sha256": M._sha(raw)})
            journal.append(json.dumps({"step": "game_start", "observation": {"pair_id": p["pair_id"], "arm": arm}}))
    (d / "record.json").write_text(json.dumps({"status": "COMPLETE", "plan": plan, "games": games}))
    (d / "watchdog.json").write_text(json.dumps({"status": "COMPLETE", "record_status": "COMPLETE"}))
    (d / "log.jsonl").write_text("\n".join(journal) + "\n")
    seeds = sorted(s for p in plan["pairs"] for s in p["seeds"].values())
    (tmp_path / M.REGISTRY).parent.mkdir(parents=True)
    (tmp_path / M.REGISTRY).write_text(f"ACCOUNTED_SEED_INTERVALS = ((1, 2), ({seeds[0]}, {seeds[-1] + 1}))\n"
                                       "EXPOSED_SEED_INTERVALS = ((1, 2),)\nRETIRED_SEED_INTERVALS = ((1, 2),)\n"
                                       "TEST_ONLY_SEED_INTERVALS = ()\n")

    def commit():
        for args in (["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                                     "-c", "core.hooksPath=/dev/null", "commit", "-q", "-m", "x"]):
            assert subprocess.run(["git", "-C", str(tmp_path), *args], capture_output=True).returncode == 0
    assert subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], capture_output=True).returncode == 0
    commit()
    durable = []
    monkeypatch.setattr(M, "ROOT", tmp_path)
    monkeypatch.setattr(F, "check_durable", lambda paths, _fixture: durable.append(list(paths)))
    return tmp_path, plan, (seeds[0], seeds[-1] + 1), commit, durable


def test_INPUTS_read_a_sound_tree_and_it_RECONCILES(M, evidence):
    _root, plan, block, _commit, durable = evidence
    x = M._inputs()
    assert len(x["starts"]) == len(x["seeds"]) == len(x["games"]) == 2 * len(plan["pairs"]) == x["files"] - 5
    assert len(durable) == 1 and len(durable[0]) == x["files"]                   # every input, durability-checked
    assert M.reconcile(x["plan"], x["starts"], x["games"], x["seeds"], x["registries"], x["statuses"]) == ([], block)


def test_INPUTS_refuse_a_game_file_that_is_not_the_one_HASHED(M, evidence):
    root, _plan, _block, commit, _d = evidence
    f = root / M.PILOT / "games" / "pilot-03-B.json"
    f.write_text(f.read_text().replace('"moves": []', '"moves": [[1, 1]]'))
    commit()
    with pytest.raises(M.Refused, match="pilot-03-B.json is not the file the record hashed"):
        M._inputs()


def test_INPUTS_refuse_a_plan_that_is_NOT_THE_PIN_and_a_record_of_another_plan(M, evidence):
    root, plan, _block, commit, _d = evidence
    rec = json.loads((root / M.PILOT / "record.json").read_text())
    (root / M.PILOT / "record.json").write_text(json.dumps(dict(rec, plan={"pairs": plan["pairs"][:15]})))
    commit()
    with pytest.raises(M.Refused, match="ran another plan"):
        M._inputs()
    (root / M.PLAN).write_text(json.dumps(plan))                                # same content, other bytes
    commit()
    with pytest.raises(M.Refused, match="not the pilot's committed plan"):
        M._inputs()


def test_INPUTS_refuse_an_UNCOMMITTED_input(M, evidence, monkeypatch):
    from scripts.GPU.alphazero import h4_pilot_feasibility as F

    def refuse(paths, _fixture):
        raise F.H4AnalysisRefused("not durable here")
    monkeypatch.setattr(F, "check_durable", refuse)
    with pytest.raises(M.Refused, match="not committed and unmodified: not durable here"):
        M._inputs()


# ─────────────────────────────── THE CLOSED STATE: the one closing edit (2026-10-08) ───────────────────────────────

def _closed_block(M):
    """The block, READ from the committed derivation record -- never spelled here."""
    import re
    rec = (ROOT / M.HERE / "02_derive_usage_run.txt").read_text()
    assert "VERDICT: RECONCILES" in rec
    m = re.search(r"^the block the started seeds make: \((\d+), (\d+)\);", rec, re.M)
    return int(m.group(1)), int(m.group(2))


def test_the_pilot_block_is_ACCOUNTED_EXPOSED_and_RETIRED_exactly_as_itself_and_LAST(M):
    from scripts.GPU.alphazero import e4_screen_reference as REF
    lo, hi = block = _closed_block(M)
    for name in ("ACCOUNTED_SEED_INTERVALS", "EXPOSED_SEED_INTERVALS", "RETIRED_SEED_INTERVALS"):
        intervals = getattr(REF, name)
        short = name.split("_")[0]
        assert intervals.count(block) == 1, f"the pilot block is not registered as exactly itself in {short}"
        assert [iv for iv in intervals if iv[0] < hi and lo < iv[1]] == [block], \
            f"another interval overlaps the pilot block in {short}"
    # RE-ANCHORED 2026-10-09 (external-engine learning closing edit): appended LAST at its own closing, the
    # pilot block is now followed by a LATER closing -- in both lists alike, and only by higher blocks.
    ex, rt = REF.EXPOSED_SEED_INTERVALS, REF.RETIRED_SEED_INTERVALS
    assert ex[ex.index(block):] == rt[rt.index(block):] and all(iv[0] >= hi for iv in ex[ex.index(block) + 1:]), \
        "the pilot block is not the LAST entry of EXPOSED and RETIRED"
    assert not any(iv[0] <= lo - 1 < iv[1] or iv[0] <= hi < iv[1] for name in
                   ("ACCOUNTED_SEED_INTERVALS", "EXPOSED_SEED_INTERVALS", "RETIRED_SEED_INTERVALS", "TEST_ONLY_SEED_INTERVALS")
                   for iv in getattr(REF, name)), "a registry interval reaches just outside the pilot block"
    for seed in range(lo, hi):
        st = REF.seed_status(seed)
        assert (st["accounted"], st["exposed"], st["retired"], st["test_only"]) == (True, True, True, False), \
            f"pilot seed lo+{seed - lo} is not accounted, exposed and retired"


def test_every_pilot_seed_now_FAULTS_as_exposed_and_retired(M):
    from scripts.GPU.alphazero import h4_runner as R4
    lo, hi = _closed_block(M)
    assert R4.seed_registry_faults([{"seed": s} for s in range(lo, hi)]) == \
        [f"{s}: exposed, retired" for s in range(lo, hi)], "a pilot seed does not fault as exactly exposed and retired"


def test_R1_REFUSES_the_committed_pilot_plan_at_the_shared_registry(M):
    from scripts.GPU.alphazero import candidate_match_runner as CM
    plan = json.loads((ROOT / M.PLAN).read_bytes())
    try:
        CM.check_plan(plan, mode="pilot")
    except CM.MatchRefused as e:
        assert "seeds not registered for use, or spent" in str(e) and "exposed, retired" in str(e), \
            "R1 refused the spent pilot plan for another reason"
    else:
        raise AssertionError("R1 accepted the spent pilot plan")
