"""L1: execution-wiring qualification for the L0 64-game match. NO EXECUTION.

No model, no jvm, no engine game, no download, no install, and NO DRAW from the
reserved block [202613000, 202613064). Every seed used here is inside
TEST_ONLY_SEED_INTERVALS, which no schedule may contain by construction. The last
test asserts the reserved block is still untouched when this file finishes.
"""
import ast
import inspect
import json
import os

import pytest

from scripts.GPU.alphazero import e4_screen_reference as REF
from scripts.GPU.alphazero import e4_screen_rules as SCREEN_RULES
from scripts.GPU.alphazero import e4_screen_command as SCREEN_CMD
from scripts.GPU.alphazero import e4_screen_runner as H
from scripts.GPU.alphazero import l0_match_command as C
from scripts.GPU.alphazero import l0_match_plan as P
from scripts.GPU.alphazero import l0_match_rules as RULES
from scripts.GPU.alphazero import l0_match_runner as R

PLAN_PATH = P.L0_PLAN_REL
TEST_SEED = 90009010                      # inside TEST_ONLY_SEED_INTERVALS
SHA1 = REF.REFERENCE_CHECKPOINTS["calib020_0001"]["sha1"] if hasattr(
    REF, "REFERENCE_CHECKPOINTS") else None


@pytest.fixture(scope="module")
def frozen():
    return P.load_l0_plan()


def syn(i=0, opening="o1_center", arm="t1j_red", seed=None):
    """A synthetic task on a TEST-ONLY seed. Never a scheduled seed."""
    from scripts.GPU.alphazero.twixtbot_g3_schedule import REFERENCE_CHECKPOINTS
    anchor = "red" if arm == "t1j_red" else "black"
    t = {"task_id": f"syn-{i:03d}", "endpoint": "strong", "t1j_mdPly": 6,
         "t1j_mdFixedPly": True, "opening": opening, "colour_arm": arm, "rep": i % 4,
         "anchor_colour": anchor, "reference": "calib020_0001",
         "reference_sha1": REFERENCE_CHECKPOINTS["calib020_0001"]["sha1"],
         "seed": TEST_SEED + i if seed is None else seed}
    t["reference_colour"] = REF.reference_colour(t)
    return t


class _EndState:
    """A synthetic TERMINAL state. No board, no engine, no move is ever made."""

    def __init__(self, ply=40, winner="red"):
        self.ply, self._w, self.to_move = ply, winner, "red"

    def winner(self):
        return self._w

    def legal_moves(self):
        raise AssertionError("a terminal state was asked for moves")

    def apply_move(self, m):
        raise AssertionError("a terminal state was moved")


def no_agents(task, mover, evaluator=None):
    raise AssertionError("an agent was constructed during L1")


def rows(path):
    return [json.loads(l) for l in open(path)]


# --- the L0 gate is closed, and it is L0's OWN --------------------------------

def test_the_l0_execution_gate_is_false():
    assert C.L0_EXECUTION_AUTHORIZED is False


def test_exactly_one_assignment_to_the_l0_gate_and_it_is_false():
    src = inspect.getsource(C)
    assigns = [n for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.Assign)
               for t in n.targets
               if isinstance(t, ast.Name) and t.id == "L0_EXECUTION_AUTHORIZED"]
    assert len(assigns) == 1
    assert assigns[0].value.value is False


def test_the_l0_gate_does_not_read_the_screens_gate():
    """One gate must never be openable by opening the other."""
    src = inspect.getsource(C)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Attribute) and node.attr == "SCREEN_AUTHORIZED":
            pytest.fail("the L0 command reads the screen's gate")
        if isinstance(node, ast.Name) and node.id == "SCREEN_AUTHORIZED":
            pytest.fail("the L0 command references SCREEN_AUTHORIZED")


def test_the_l0_command_reads_no_environment_and_no_config():
    """AST, not a grep: a comment saying 'reads no environment' is not behaviour."""
    offences = []
    for node in ast.walk(ast.parse(inspect.getsource(C))):
        if isinstance(node, ast.Attribute) and node.attr in {"environ", "environb", "getenv"}:
            offences.append(f"attribute {node.attr} line {node.lineno}")
        if isinstance(node, ast.Name) and node.id in {"getenv", "environ"}:
            offences.append(f"name {node.id} line {node.lineno}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", None) or ""
            names = [a.name for a in node.names]
            for bad in ("configparser", "dotenv", "yaml", "tomllib", "tomli"):
                if bad in mod or any(bad in n for n in names):
                    offences.append(f"import {bad} line {node.lineno}")
    assert offences == [], offences


def test_match_mode_is_not_publicly_selectable(tmp_path):
    with pytest.raises(H.HarnessError, match="UNAUTHORIZED"):
        R.run(PLAN_PATH, str(tmp_path / "a.jsonl"), mode=R.MATCH_MODE)
    assert R.MODES == ("qualify",)
    params = inspect.signature(R.run).parameters
    assert set(params) == {"plan_path", "results_path", "mode"}


# --- the screen's frozen behaviour is untouched ------------------------------

def test_the_l0_runner_imports_no_screen_decision_rule():
    """No band, no saturation, no early stop, no joint classifier."""
    banned = {"early_in_band_forced", "saturation_reachable", "per_endpoint_decision",
              "cap_incompleteness_reachable", "classify_joint", "earliest_early_stop"}
    tree = ast.parse(inspect.getsource(R))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(a.name for a in node.names)
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        if isinstance(node, ast.Name):
            names.add(node.id)
    assert not (names & banned), sorted(names & banned)
    for fn in banned:                       # they must still exist, for the screen
        assert hasattr(SCREEN_RULES, fn)


def test_the_screen_harness_constants_are_untouched():
    """L1 must not widen the screen. These are the screen's frozen numbers."""
    assert H.CANONICAL_N_TASKS == 32
    assert H.CANONICAL_N_PER_ENDPOINT == 16
    assert H.BAND == (0.05, 0.95)
    assert H.MODES == ("qualify",) and H.SCREEN_MODE == "screen"
    assert H.SCREEN_SEED_BLOCK == (202612128, 202612160)


# --- identity ordering: header durable BEFORE any effect ---------------------

def test_the_identity_header_is_durable_before_setup_runs(tmp_path):
    out = tmp_path / "r.jsonl"
    ident = {"plan": {"plan_sha256": "abc"}, "jar": {"sha256": "def"}}

    def exploding_setup():
        raise RuntimeError("setup blew up after the header")

    with pytest.raises(H.AbortError) as e:
        R._run(PLAN_PATH, str(out), mode="qualify", _tasks=[syn(0)],
               _identity=ident, _setup=exploding_setup)
    assert e.value.phase == H.PHASE_SETUP
    recs = rows(out)
    assert recs[0]["record_type"] == "run_header"
    assert recs[0]["identity"] == ident, "identities must survive a setup failure"
    assert recs[-1]["record_type"] == "abort" and recs[-1]["phase"] == H.PHASE_SETUP
    assert not any(r["record_type"] == "setup_complete" for r in recs)


def test_a_setup_failure_is_classified_not_leaked(tmp_path):
    with pytest.raises(H.AbortError) as e:
        R._run(PLAN_PATH, str(tmp_path / "r.jsonl"), mode="qualify", _tasks=[syn(0)],
               _setup=lambda: (_ for _ in ()).throw(ValueError("boom")))
    assert e.value.phase == H.PHASE_SETUP and "ValueError" in e.value.message


# --- exclusive, durable recording -------------------------------------------

def test_the_results_file_is_exclusive_create(tmp_path):
    out = tmp_path / "r.jsonl"
    out.write_text("stale\n")
    with pytest.raises(H.HarnessError, match="already exists"):
        R._run(PLAN_PATH, str(out), mode="qualify", _tasks=[syn(0)])
    assert out.read_text() == "stale\n", "an existing file must be left alone"


def test_the_runner_uses_the_qualified_recorder_not_its_own():
    src = inspect.getsource(R)
    assert "H.Recorder(" in src
    assert "class Recorder" not in src, "L1 must not reimplement the recorder"


# --- integrity aborts --------------------------------------------------------

def test_the_default_binder_refuses(tmp_path):
    """A run that forgets to wire the binder must abort, not play unbound."""
    with pytest.raises(H.AbortError) as e:
        R._run(PLAN_PATH, str(tmp_path / "r.jsonl"), mode="qualify", _tasks=[syn(0)],
               _state_factory=lambda t: _EndState(), _agent_factory=no_agents,
               _cleanup=lambda: None)
    assert e.value.phase == H.PHASE_BIND


def test_the_default_state_factory_and_agent_factory_refuse(tmp_path):
    with pytest.raises(H.AbortError) as e:
        R._run(PLAN_PATH, str(tmp_path / "a.jsonl"), mode="qualify", _tasks=[syn(0)],
               _binder=lambda *a, **k: None, _cleanup=lambda: None)
    assert e.value.phase in (H.PHASE_PRECONDITION, H.PHASE_FACTORY)


def test_a_missing_evaluator_aborts_rather_than_disabling_the_check(tmp_path):
    """No off switch: forgetting the evaluator must not look like disabling it."""
    class _Agent:
        pass

    with pytest.raises(H.AbortError) as e:
        R._run(PLAN_PATH, str(tmp_path / "r.jsonl"), mode="qualify",
               _tasks=[syn(0)],
               _state_factory=lambda t: _MoveOnce(),
               _agent_factory=lambda t, m, ev: _Agent(),
               _binder=lambda *a, **k: None, _evaluator=None, _cleanup=lambda: None)
    assert e.value.phase == H.PHASE_FACTORY
    assert "identity cannot be checked" in e.value.message


class _MoveOnce:
    """One legal move, then terminal -- enough to force an agent construction.

    `to_move` matters: the evaluator gate binds ONLY the reference side, so a
    state whose mover is the classical anchor never reaches it. That scoping is
    deliberate (T1j holds no evaluator and never will) and this default exercises
    the gate rather than skipping past it.
    """

    def __init__(self, ply=10, to_move="black"):
        self.ply, self.to_move = ply, to_move

    def winner(self):
        return None

    def legal_moves(self):
        return [(1, 1)]

    def apply_move(self, m):
        return _EndState(ply=self.ply + 1, winner="red")


def test_a_binder_failure_is_classified_as_a_binding_abort(tmp_path):
    def bad_binder(*a, **k):
        raise ValueError("engines diverged")

    with pytest.raises(H.AbortError) as e:
        R._run(PLAN_PATH, str(tmp_path / "r.jsonl"), mode="qualify", _tasks=[syn(0)],
               _state_factory=lambda t: _EndState(), _agent_factory=no_agents,
               _binder=bad_binder, _cleanup=lambda: None)
    assert e.value.phase == H.PHASE_BIND and "engines diverged" in e.value.message


def test_a_cleanup_failure_after_a_recorded_game_is_a_run_level_abort(tmp_path):
    out = tmp_path / "r.jsonl"

    def bad_cleanup():
        raise RuntimeError("cleanup failed")

    with pytest.raises(H.AbortError) as e:
        R._run(PLAN_PATH, str(out), mode="qualify", _tasks=[syn(0)],
               _state_factory=lambda t: _EndState(), _agent_factory=no_agents,
               _binder=lambda *a, **k: None, _cleanup=bad_cleanup)
    assert e.value.phase == H.PHASE_CLEANUP
    recs = rows(out)
    assert any(r["record_type"] == "task_result" for r in recs), (
        "the finished game must be persisted BEFORE cleanup runs")


# --- no early stop, and every scheduled task is played -----------------------

def test_every_scheduled_task_is_played_and_none_is_skipped(tmp_path):
    out = tmp_path / "r.jsonl"
    tasks = [syn(i) for i in range(6)]
    rc = R._run(PLAN_PATH, str(out), mode="qualify", _tasks=tasks,
                _state_factory=lambda t: _EndState(), _agent_factory=no_agents,
                _binder=lambda *a, **k: None, _cleanup=lambda: None)
    assert rc == H.EXIT_OK
    recs = rows(out)
    played = [r for r in recs if r["record_type"] == "task_result"]
    assert len(played) == 6
    assert not any(r["record_type"] in ("task_skipped", "early_stop") for r in recs)
    assert recs[0]["early_stop"] is None and recs[0]["n_games"] == 64


def test_a_run_of_all_wins_does_not_stop_early(tmp_path):
    """The screen would have early-stopped on a streak like this. L0 must not."""
    out = tmp_path / "r.jsonl"
    tasks = [syn(i) for i in range(10)]
    R._run(PLAN_PATH, str(out), mode="qualify", _tasks=tasks,
           _state_factory=lambda t: _EndState(winner=t["anchor_colour"]),
           _agent_factory=no_agents, _binder=lambda *a, **k: None,
           _cleanup=lambda: None)
    played = [r for r in rows(out) if r["record_type"] == "task_result"]
    assert len(played) == 10 and all(r["t1j_points"] == 1.0 for r in played)
    assert RULES.may_stop_early(10.0, 10, 64) is False


def test_the_runner_has_no_skip_path_at_all():
    src = inspect.getsource(R)
    assert "task_skipped" not in src
    assert "early_stop" in src, "the header records that there is none"
    assert "stopped" not in src


# --- canonical schedule enforcement (match mode, never executed) -------------

def test_match_mode_requires_the_frozen_schedule(tmp_path, frozen):
    """Synthetic tasks must be refused BEFORE the recorder is created."""
    out = tmp_path / "r.jsonl"
    with pytest.raises(H.HarnessError, match="expected exactly 64"):
        R._run(PLAN_PATH, str(out), mode=R.MATCH_MODE, _tasks=[syn(0)],
               _state_factory=lambda t: _EndState(), _agent_factory=no_agents,
               _binder=lambda *a, **k: None, _cleanup=lambda: None)
    assert not out.exists(), "a refused match must create no results file"


@pytest.mark.parametrize("mutate,match", [
    (lambda ts: ts[:-1], "expected exactly 64"),
    (lambda ts: ts[::-1], "digest"),
    (lambda ts: [dict(t, seed=t["seed"] + 1000) if i == 0 else t
                 for i, t in enumerate(ts)], "outside"),
    (lambda ts: [dict(t, opening="zzz") if i == 0 else t
                 for i, t in enumerate(ts)], "opening/colour cells"),
    (lambda ts: [dict(t, rep=3) if i == 0 else t
                 for i, t in enumerate(ts)], "repetitions"),
])
def test_match_mode_refuses_every_schedule_edit(tmp_path, frozen, mutate, match):
    out = tmp_path / "r.jsonl"
    with pytest.raises(H.HarnessError, match=match):
        R._run(PLAN_PATH, str(out), mode=R.MATCH_MODE,
               _tasks=mutate([dict(t) for t in frozen["tasks"]]),
               _state_factory=lambda t: _EndState(), _agent_factory=no_agents,
               _binder=lambda *a, **k: None, _cleanup=lambda: None)
    assert not out.exists()


def test_the_seed_block_gate_is_defence_in_depth_and_binds_directly(frozen):
    """Honest scoping: through `_run` this gate is the THIRD check of the same fact.

    `validate_l0_schedule` already refuses a seed outside the block, and the task
    digest covers `seed` as well, so an edited seed never reaches `_assert_match_seed`
    via a run. It stays as defence in depth -- it would catch a schedule that
    satisfied both while `L0_SEED_BLOCK` itself had been changed -- and is therefore
    qualified DIRECTLY here rather than through a path that cannot reach it.
    """
    for t in frozen["tasks"]:
        R._assert_match_seed(t)                       # the whole frozen 64 pass
    with pytest.raises(H.HarnessError, match="outside the reserved L0 block"):
        R._assert_match_seed(dict(frozen["tasks"][0], seed=202612128))
    with pytest.raises(H.HarnessError, match="outside the reserved L0 block"):
        R._assert_match_seed(dict(frozen["tasks"][0], seed=P.L0_SEED_BLOCK[1]))


def test_match_mode_binds_full_content_not_task_ids(tmp_path, frozen, monkeypatch):
    """Canonical NAMES on synthetic CONTENT must not pass.

    Through `_run` the pinned digest catches an edited task first, so this check
    is masked there. It is bound DIRECTLY, with the digest re-pinned to the edited
    schedule so the content comparison against the loaded plan is the only gate
    left standing -- the realistic failure being a digest constant that drifted.
    """
    tasks = [dict(t) for t in frozen["tasks"]]
    # SWAP two seeds. Every structural rule still holds -- both seeds are in the
    # block, all 64 remain unique, the 16 cells still hold 4 reps each -- so the
    # only thing left that can notice is the content comparison against the plan.
    tasks[0], tasks[4] = (dict(tasks[0], seed=tasks[4]["seed"]),
                          dict(tasks[4], seed=tasks[0]["seed"]))
    for t in (tasks[0], tasks[4]):
        t["rng_streams"] = REF.rng_stream_seeds(t)
    monkeypatch.setattr(RULES, "L0_TASK_DIGEST", RULES.l0_task_digest(tasks))
    with pytest.raises(H.HarnessError, match="does not match the frozen plan"):
        R._verify_match_schedule(tasks, frozen)
    # the control: the real 64 pass the same call
    monkeypatch.undo()
    R._verify_match_schedule([dict(t) for t in frozen["tasks"]], frozen)


def test_qualification_refuses_a_reserved_or_spent_seed(tmp_path, frozen):
    """Qualify mode runs on test-only seeds; a scheduled seed is refused."""
    for seed in (202613000, 202612128, 90000001):
        with pytest.raises(H.HarnessError, match="accounted, exposed or retired"):
            R._run(PLAN_PATH, str(tmp_path / f"r{seed}.jsonl"), mode="qualify",
                   _tasks=[syn(0, seed=seed)],
                   _state_factory=lambda t: _EndState(), _agent_factory=no_agents,
                   _binder=lambda *a, **k: None, _cleanup=lambda: None)


def test_a_test_only_seed_is_accepted_by_qualification(tmp_path):
    """The control: a gate that refuses every seed proves nothing."""
    assert REF.seed_is_test_only(TEST_SEED)
    rc = R._run(PLAN_PATH, str(tmp_path / "r.jsonl"), mode="qualify", _tasks=[syn(0)],
                _state_factory=lambda t: _EndState(), _agent_factory=no_agents,
                _binder=lambda *a, **k: None, _cleanup=lambda: None)
    assert rc == H.EXIT_OK


# --- post-run reporting ------------------------------------------------------

def test_a_qualification_run_gets_a_RECEIPT_never_a_report(tmp_path):
    out = tmp_path / "r.jsonl"
    R._run(PLAN_PATH, str(out), mode="qualify", _tasks=[syn(i) for i in range(4)],
           _state_factory=lambda t: _EndState(), _agent_factory=no_agents,
           _binder=lambda *a, **k: None, _cleanup=lambda: None)
    recs = rows(out)
    receipts = [r for r in recs if r["record_type"] == "qualification_receipt"]
    assert len(receipts) == 1
    assert receipts[0]["tasks_played"] == 4
    assert "not in the design" in receipts[0]["report_withheld"]
    assert not any(r["record_type"] == "match_report" for r in recs)


def test_the_runner_computes_no_statistic_of_its_own():
    """All reporting is delegated to the frozen rules."""
    src = inspect.getsource(R)
    assert "RULES.match_report(" in src
    for forbidden in ("wilson", "hoeffding", "sqrt", "/ N_GAMES", "sum("):
        assert forbidden not in src, forbidden


def test_the_reporter_would_report_a_complete_frozen_match(frozen):
    """Reporting is qualified on a synthetic RESULT VECTOR, not by playing.

    This is the one place L1 exercises the report path end to end, and it uses no
    runner, no state, no agent and no seed: just 64 rows shaped as play_task emits
    them, bound to the frozen tasks.
    """
    result_rows = []
    for i, t in enumerate(frozen["tasks"]):
        win = i % 8 != 0
        other = "black" if t["anchor_colour"] == "red" else "red"
        result_rows.append({"task_id": t["task_id"], "seed": t["seed"], "plies": 40,
                            "terminal_reason": "win",
                            "winner": t["anchor_colour"] if win else other,
                            "t1j_points": 1.0 if win else 0.0})
    rep = RULES.match_report(result_rows, frozen["tasks"])
    assert rep["reported"] is True
    assert rep["overall"]["games"] == 64 and rep["overall"]["t1j_score"] == 56.0
    assert "ci95_hoeffding" in rep["overall"] and "ci95_wilson" in rep["overall"]


def test_an_incomplete_frozen_match_is_not_reported(frozen):
    result_rows = [{"task_id": t["task_id"], "seed": t["seed"], "plies": 40,
                    "terminal_reason": "win", "winner": t["anchor_colour"],
                    "t1j_points": 1.0} for t in frozen["tasks"][:63]]
    rep = RULES.match_report(result_rows, frozen["tasks"])
    assert rep["reported"] is False and "unplayed" in rep["reason"]


# --- the command: preconditions, then the gate -------------------------------

def _repo(root):
    dest = root / os.path.dirname(PLAN_PATH)
    dest.mkdir(parents=True)
    (dest / os.path.basename(PLAN_PATH)).write_bytes(open(PLAN_PATH, "rb").read())
    (root / ".gitignore").write_text("checkpoints/\n")
    import subprocess
    for cmd in (["init", "-q"], ["add", "-A"],
                ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "plan"]):
        subprocess.run(["git", "-C", str(root), *cmd], check=True, capture_output=True)
    ck = root / C.CANONICAL_CHECKPOINT_REL
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.write_bytes(open("checkpoints/alphazero-v2-calib020-from0409/"
                        "model_iter_0001.safetensors", "rb").read())
    return root


@pytest.fixture(scope="module")
def clean_repo(tmp_path_factory):
    return _repo(tmp_path_factory.mktemp("l0repo") / "r")


# The toolchain is resolved through an explicit setting and hash-verified before
# use; it is NOT committed. These constants previously pointed into a session
# scratchpad under /private/tmp, which was cleaned -- taking 41 qualification
# tests red with it. See scripts/GPU/alphazero/t1j_toolchain.py.
from scripts.GPU.alphazero import t1j_toolchain as TC

_TOOLCHAIN = TC.verified_paths()
JDK = _TOOLCHAIN["jdk_home"]
JAR = _TOOLCHAIN["jar"]


def args(clean_repo, tmp_path, **over):
    a = dict(plan_path=str(clean_repo / PLAN_PATH), results_path=str(tmp_path / "o.jsonl"),
             repo_root=str(clean_repo), jdk_home=JDK, jar_path=JAR,
             checkpoint_path=str(clean_repo / C.CANONICAL_CHECKPOINT_REL))
    a.update(over)
    return a


#: A SNAPSHOT of the seed registries, taken at import. Every lift below is
#: measured against this, and `test_zzz_the_registries_are_restored` compares the
#: live registries to it at the end of the file.
_REGISTRY_SNAPSHOT = {
    "exposed": REF.EXPOSED_SEED_INTERVALS,
    "retired": REF.RETIRED_SEED_INTERVALS,
    "test_only": REF.TEST_ONLY_SEED_INTERVALS,
    "accounted": REF.ACCOUNTED_SEED_INTERVALS,
}


def _registries_match_snapshot():
    return (REF.EXPOSED_SEED_INTERVALS == _REGISTRY_SNAPSHOT["exposed"]
            and REF.RETIRED_SEED_INTERVALS == _REGISTRY_SNAPSHOT["retired"]
            and REF.TEST_ONLY_SEED_INTERVALS == _REGISTRY_SNAPSHOT["test_only"]
            and REF.ACCOUNTED_SEED_INTERVALS == _REGISTRY_SNAPSHOT["accounted"])


@pytest.fixture
def unspent_block(monkeypatch):
    """Lifts the L0 block's ELIGIBILITY in process, and nothing else.

    WHY THIS IS NOT RELAXING A GATE. The match ran on 2026-08-27, so
    `check_schedule` now refuses the spent block before authorization is ever
    consulted. Left alone, every test of a LATER precondition, of the
    AUTHORIZATION gate, and of the ENABLED-PATH wiring would stop passing for a
    reason that has nothing to do with what it tests -- and a safety gate whose
    tests all went vacuous is the failure this workstream keeps finding. The
    real-state refusal is asserted separately and unpatched, in
    `test_the_spent_schedule_is_refused_with_exit_2_before_any_effect`.

    IT TOUCHES ELIGIBILITY ONLY. It never changes L0_EXECUTION_AUTHORIZED and
    never changes SCREEN_AUTHORIZED -- asserted on the way in and on the way out.
    """
    gate_l0, gate_screen = C.L0_EXECUTION_AUTHORIZED, SCREEN_CMD.SCREEN_AUTHORIZED
    keep_exposed = tuple(iv for iv in REF.EXPOSED_SEED_INTERVALS if iv != P.L0_SEED_BLOCK)
    keep_retired = tuple(iv for iv in REF.RETIRED_SEED_INTERVALS if iv != P.L0_SEED_BLOCK)
    monkeypatch.setattr(REF, "EXPOSED_SEED_INTERVALS", keep_exposed)
    monkeypatch.setattr(REF, "RETIRED_SEED_INTERVALS", keep_retired)
    assert not REF.seed_is_unavailable(P.L0_SEED_BLOCK[0])
    assert C.L0_EXECUTION_AUTHORIZED is gate_l0 is False
    assert SCREEN_CMD.SCREEN_AUTHORIZED is gate_screen is False
    yield
    # THE FIXTURE'S OWN CONTRACT, not the test's. It patched exactly the two
    # eligibility registries and nothing else, so the screen's gate -- which no L0
    # test ever touches -- must still be closed. The L0 gate is deliberately NOT
    # asserted here: an enabled-path test may lift it in process, and
    # `test_the_gate_is_restored_after_the_enabled_path_tests` plus the last-in-file
    # restoration test cover that separately.
    assert SCREEN_CMD.SCREEN_AUTHORIZED is False, "the lift must never touch a gate"
    assert REF.TEST_ONLY_SEED_INTERVALS == _REGISTRY_SNAPSHOT["test_only"]
    assert REF.ACCOUNTED_SEED_INTERVALS == _REGISTRY_SNAPSHOT["accounted"]


def test_the_lift_fixture_changes_eligibility_and_nothing_else(unspent_block):
    """The fixture's own contract, asserted rather than assumed."""
    assert not REF.seed_is_exposed(P.L0_SEED_BLOCK[0])
    assert not REF.seed_is_retired(P.L0_SEED_BLOCK[0])
    assert REF.seed_is_accounted(P.L0_SEED_BLOCK[0]), "reservation is untouched"
    assert REF.TEST_ONLY_SEED_INTERVALS == _REGISTRY_SNAPSHOT["test_only"]
    # other experiments' seeds are unaffected
    assert REF.seed_is_exposed(202612128) and REF.seed_is_retired(202612128)
    assert REF.seed_is_exposed(90000001)
    assert C.L0_EXECUTION_AUTHORIZED is False


def test_the_spent_schedule_is_refused_with_exit_2_before_any_effect(
        tmp_path, clean_repo):
    """REAL STATE, UNPATCHED. The block is spent, so execution is refused.

    This is now the command's permanent behaviour and it must NOT reach the
    closed-gate exit 5 with the actual L0 block. Nothing effectful may happen
    first: no results file, no class directory, no agent, no RNG, no model, no
    Java.
    """
    import random
    import subprocess as _sub
    made, spawned = [], []

    class _Counting(random.Random):
        def __init__(self, *a, **kw):
            made.append(a)
            super().__init__(*a, **kw)

    real_run = _sub.run
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(random, "Random", _Counting)
        mp.setattr(_sub, "run", lambda *a, **k: (spawned.append(a[0] if a else k.get("args")),
                                                 real_run(*a, **k))[1])
        trace = []
        log = []
        out = tmp_path / "must_not_exist.jsonl"
        with pytest.raises(C.PreconditionError) as e:
            C._run_match(**args(clean_repo, tmp_path, results_path=str(out)),
                         _trace=trace,
                         _load_evaluator=lambda r: log.append("evaluator"),
                         _build_agent=lambda *a, **k: log.append("agent"),
                         _run_harness=lambda *a, **k: log.append("harness"),
                         _compile=lambda *a, **k: log.append("compile"))
    assert e.value.which == "schedule"
    assert "EXPOSED" in e.value.message
    assert trace == ["plan"], "the plan still parses; the schedule is what refuses"
    assert log == [], "nothing effectful may be reached"
    assert not out.exists() and not (tmp_path / "must_not_exist.jsonl.t1j_classes").exists()
    assert made == [], "no RNG"
    # NOTHING is spawned. The refusal happens at `schedule`, precondition 2, and
    # `repository` -- the only check that shells out to git -- never runs. An
    # earlier version asserted `programs <= {"git"}`, which is VACUOUSLY TRUE of
    # an empty list and would have passed however many git calls were made.
    assert spawned == [], f"no subprocess may run before the refusal: {spawned}"


def test_the_real_cli_exits_2_not_5_on_the_spent_block(tmp_path, clean_repo):
    """Fresh subprocess, real registries. Exit 2 is correct and permanent."""
    r = cli(*cli_args(tmp_path, clean_repo))
    assert r.returncode == C.EXIT_PRECONDITION, r.stderr[:300]
    assert "PRECONDITION REFUSED" in r.stderr and "schedule" in r.stderr
    assert "UNEXPECTED" not in r.stderr
    assert "L0 MATCH NOT AUTHORIZED" not in r.stderr, (
        "the spent block must not reach the authorization gate")
    assert not (tmp_path / "o.jsonl").exists()


def test_all_seven_preconditions_complete_before_the_refusal(tmp_path, clean_repo,
                                                             unspent_block):
    trace = []
    calls = []
    with pytest.raises(C.AuthorizationError) as e:
        C._run_match(**args(clean_repo, tmp_path), _trace=trace,
                     _load_evaluator=lambda r: calls.append("evaluator"),
                     _build_agent=lambda *a, **k: calls.append("agent"),
                     _run_harness=lambda *a, **k: calls.append("harness"),
                     _compile=lambda *a, **k: calls.append("compile"))
    assert trace == list(C.PRECONDITIONS)
    assert e.value.completed_preconditions == list(C.PRECONDITIONS)
    assert calls == [], "no effect may be reached while unauthorized"


def test_no_results_file_or_class_dir_is_created_while_unauthorized(tmp_path, clean_repo, unspent_block):
    out = tmp_path / "must_not_exist.jsonl"
    with pytest.raises(C.AuthorizationError):
        C._run_match(**args(clean_repo, tmp_path, results_path=str(out)))
    assert not out.exists()
    assert not (tmp_path / "must_not_exist.jsonl.t1j_classes").exists()


def test_no_rng_is_constructed_anywhere_while_unauthorized(tmp_path, clean_repo,
                                                           monkeypatch, unspent_block):
    import random
    made = []

    class _Counting(random.Random):
        def __init__(self, *a, **kw):
            made.append(a)
            super().__init__(*a, **kw)

    monkeypatch.setattr(random, "Random", _Counting)
    with pytest.raises(C.AuthorizationError):
        C._run_match(**args(clean_repo, tmp_path))
    assert made == [], f"{len(made)} RNG(s) constructed before the refusal"


@pytest.mark.parametrize("over,which,completed", [
    ({"plan_path": "/nonexistent/plan.json"}, "plan", []),
    ({"jdk_home": "/nonexistent"}, "jdk", ["plan", "schedule", "repository"]),
    ({"jar_path": "/nonexistent"}, "jar", ["plan", "schedule", "repository", "jdk"]),
])
def test_each_precondition_is_immediately_fatal_in_order(tmp_path, clean_repo,
                                                         over, which, completed, unspent_block):
    trace = []
    with pytest.raises(C.PreconditionError) as e:
        C._run_match(**args(clean_repo, tmp_path, **over), _trace=trace)
    assert e.value.which == which
    assert trace == completed


def test_a_delegated_refusal_is_translated_not_leaked(tmp_path, clean_repo):
    """The reused screen checks raise a DIFFERENT PreconditionError class.

    Untranslated, `main` did not catch it and a fully understood refusal printed
    UNEXPECTED with exit 4 -- observed on the first end-to-end run.
    """
    with pytest.raises(C.PreconditionError):
        C.check_jdk("/nonexistent")
    with pytest.raises(C.PreconditionError):
        C.check_jar("/nonexistent")
    with pytest.raises(C.PreconditionError):
        C.check_output_path(str(tmp_path))


def test_the_verified_plan_object_is_carried_not_reread(tmp_path, clean_repo, unspent_block):
    rec = C.check_plan(str(clean_repo / PLAN_PATH))
    assert rec["n_tasks"] == 64 and rec["plan"]["n_tasks"] == 64
    assert C.check_schedule(rec["plan"])["executable"] is True
    with pytest.raises(C.PreconditionError, match="before the plan was verified"):
        C._verified_plan({})


# --- the CLI, in fresh subprocesses ------------------------------------------

def cli(*a, env=None):
    import subprocess
    import sys
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run([sys.executable, "-m", "scripts.GPU.alphazero.l0_match_command",
                           *a], capture_output=True, text=True, env=e)


def cli_args(tmp_path, repo):
    return ("--plan", str(repo / PLAN_PATH), "--results", str(tmp_path / "o.jsonl"),
            "--repo", str(repo), "--jdk", JDK, "--jar", JAR,
            "--checkpoint", str(repo / C.CANONICAL_CHECKPOINT_REL))


def test_cli_help_documents_the_ban():
    r = cli("--help")
    assert r.returncode == 0 and "NOT AUTHORIZED" in r.stdout


#: Runs the REAL `main` in a FRESH process with the L0 block's ELIGIBILITY lifted,
#: so the AUTHORIZATION gate is what refuses. It never touches
#: L0_EXECUTION_AUTHORIZED or SCREEN_AUTHORIZED, which are read from source as
#: always -- the lift is two registry tuples and nothing else.
_LIFT = (
    "import sys;"
    "from scripts.GPU.alphazero import e4_screen_reference as R;"
    "from scripts.GPU.alphazero import l0_match_plan as P;"
    "from scripts.GPU.alphazero import l0_match_command as C;"
    "from scripts.GPU.alphazero import e4_screen_command as SC;"
    "R.EXPOSED_SEED_INTERVALS=tuple(i for i in R.EXPOSED_SEED_INTERVALS"
    " if i != P.L0_SEED_BLOCK);"
    "R.RETIRED_SEED_INTERVALS=tuple(i for i in R.RETIRED_SEED_INTERVALS"
    " if i != P.L0_SEED_BLOCK);"
    "assert C.L0_EXECUTION_AUTHORIZED is False;"
    "assert SC.SCREEN_AUTHORIZED is False;"
    "sys.exit(C.main(sys.argv[1:]))"
)


def gate_cli(*a, env=None):
    import subprocess
    import sys
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run([sys.executable, "-c", _LIFT, *a], capture_output=True,
                          text=True, env=e)


def test_the_authorization_gate_still_exits_5_in_a_fresh_process(tmp_path, clean_repo):
    """With eligibility lifted in the child, the GATE is what refuses.

    The spent block otherwise stops the command two gates earlier, which would
    leave exit 5 -- the primary safety property -- untested from a real process.
    """
    r = gate_cli(*cli_args(tmp_path, clean_repo))
    assert r.returncode == C.EXIT_UNAUTHORIZED, r.stderr[:300]
    assert "L0 MATCH NOT AUTHORIZED" in r.stderr
    for name in C.PRECONDITIONS:
        assert name in r.stderr
    assert not (tmp_path / "o.jsonl").exists()


@pytest.mark.parametrize("env", [
    {"L0_EXECUTION_AUTHORIZED": "1"}, {"AUTHORIZED": "yes"}, {"FORCE": "1"},
    {"SCREEN_AUTHORIZED": "1"}, {"PYTHONOPTIMIZE": "1"},
])
def test_no_environment_variable_enables_the_match(tmp_path, clean_repo, env):
    """Asserted against BOTH refusals, so no env var can be excused by either.

    The plain CLI stops at `schedule` (the block is spent); the lifted one reaches
    authorization. An environment variable must change neither, and the
    comparison is against the unset baseline BYTE FOR BYTE.
    """
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    base = cli(*cli_args(tmp_path / "a", clean_repo))
    with_env = cli(*cli_args(tmp_path / "a", clean_repo), env=env)
    assert with_env.returncode == base.returncode == C.EXIT_PRECONDITION, (env, with_env.stderr[:200])
    assert with_env.stderr == base.stderr, env

    gbase = gate_cli(*cli_args(tmp_path / "b", clean_repo))
    genv = gate_cli(*cli_args(tmp_path / "b", clean_repo), env=env)
    assert genv.returncode == gbase.returncode == C.EXIT_UNAUTHORIZED, (env, genv.stderr[:200])
    assert genv.stderr == gbase.stderr, env
    assert not (tmp_path / "a" / "o.jsonl").exists()
    assert not (tmp_path / "b" / "o.jsonl").exists()


@pytest.mark.parametrize("flag", ["--authorized", "--force", "--yes", "--enable-match"])
def test_no_command_line_flag_is_accepted(tmp_path, clean_repo, flag):
    r = cli(*cli_args(tmp_path, clean_repo), flag)
    assert r.returncode == 2 and "unrecognized arguments" in r.stderr


def test_cli_precondition_failure_exits_2_not_4(tmp_path, clean_repo):
    r = cli("--plan", str(clean_repo / PLAN_PATH), "--results", str(tmp_path / "o.jsonl"),
            "--repo", str(clean_repo), "--jdk", "/nonexistent", "--jar", JAR,
            "--checkpoint", str(clean_repo / C.CANONICAL_CHECKPOINT_REL))
    assert r.returncode == C.EXIT_PRECONDITION
    assert "PRECONDITION REFUSED" in r.stderr
    assert "UNEXPECTED" not in r.stderr


# --- L1 spends nothing -------------------------------------------------------

# --- THE ENABLED PATH, with substitutes only ---------------------------------
# Proving the locked side does nothing does not prove the enabled side is real.
# Everything below temporarily opens the L0 gate IN PROCESS and substitutes every
# effectful collaborator. No Java, no model load, no game, no RNG, no scheduled
# seed. `subprocess.run` is WATCHED to prove no javac or jvm was spawned.

class _Captured:
    def __init__(self):
        self.args = None
        self.kw = None

    def __call__(self, *a, **kw):
        self.args, self.kw = a, kw
        return H.EXIT_OK


def enabled(tmp_path, clean_repo, monkeypatch, **over):
    """Open the gate in process, substitute everything effectful, and run."""
    monkeypatch.setattr(C, "L0_EXECUTION_AUTHORIZED", True)
    cap = _Captured()
    sentinel = object()
    compiled = []
    kw = dict(args(clean_repo, tmp_path),
              _compile=lambda javac, jar, out: compiled.append((javac, jar, out)) or
              {"compile_inputs": {}, "compiled_classes": {}},
              _load_evaluator=lambda repo: sentinel,
              _build_agent=lambda task, *, evaluator: ("agent", task["task_id"]),
              _run_harness=cap,
              _state_factory=lambda t: _EndState(),
              _binder=lambda *a, **k: None,
              _cleanup=lambda: None)
    kw.update(over)
    rc = C._run_match(**kw)
    return rc, cap, sentinel, compiled


def test_the_enabled_path_hands_the_harness_the_frozen_64(tmp_path, clean_repo,
                                                          monkeypatch, frozen, unspent_block):
    rc, cap, sentinel, _ = enabled(tmp_path, clean_repo, monkeypatch)
    assert rc == C.EXIT_OK
    assert cap.args[2:] == () and len(cap.args) == 2, "paths are positional"
    assert cap.kw["mode"] == R.MATCH_MODE
    handed = cap.kw["_tasks"]
    assert len(handed) == 64
    assert [t["task_id"] for t in handed] == [t["task_id"] for t in frozen["tasks"]]
    assert RULES.l0_task_digest(handed) == RULES.L0_TASK_DIGEST


def test_the_enabled_path_uses_the_frozen_settings_not_a_budget(tmp_path, clean_repo,
                                                                monkeypatch, unspent_block):
    _, cap, _, _ = enabled(tmp_path, clean_repo, monkeypatch)
    assert cap.kw["_ply_cap"] == RULES.PLY_CAP == 280
    assert cap.kw["_ply_budget"] is None, "a match plays to a terminal state"
    assert "_n_per_endpoint" not in cap.kw, "L0 has no per-endpoint count"
    assert "_band" not in cap.kw, "L0 has no band"


def test_the_enabled_path_passes_the_verified_identities(tmp_path, clean_repo,
                                                         monkeypatch, unspent_block):
    _, cap, _, _ = enabled(tmp_path, clean_repo, monkeypatch)
    ident = cap.kw["_identity"]
    assert set(ident) == set(C.PRECONDITIONS)
    assert ident["plan"]["task_digest"] == RULES.L0_TASK_DIGEST
    assert ident["jar"]["sha256"] == C.JAR_SHA256
    assert ident["checkpoint"]["sha256"] == C.CHECKPOINT_SHA256
    assert len(ident["jdk"]["components"]) == 4
    assert ident["repository"]["worktree"] == "clean"


def test_the_enabled_path_setup_builds_the_qualified_collaborators(
        tmp_path, clean_repo, monkeypatch, unspent_block):
    """The setup closure is CALLED. Three defects lived here behind the gate:
    T1jRuntime with wrong keywords, make_state_factory without its openings, and
    no trace at all."""
    import subprocess as _sub
    spawned = []
    real_run = _sub.run
    monkeypatch.setattr(_sub, "run",
                        lambda *a, **k: (spawned.append(a[0] if a else k.get("args")),
                                         real_run(*a, **k))[1])
    trace = []
    rc, cap, sentinel, compiled = enabled(tmp_path, clean_repo, monkeypatch, _trace=trace)
    collab = cap.kw["_setup"]()                      # <-- the branch under test
    assert set(collab) >= {"state_factory", "binder", "agent_factory", "evaluator",
                           "cleanup", "artifacts"}
    assert collab["evaluator"] is sentinel, "the loaded evaluator is handed through"
    assert callable(collab["state_factory"]) and callable(collab["binder"])
    assert callable(collab["agent_factory"])
    assert trace[-4:] == ["compile", "t1j_runtime", "load_evaluator", "agent_factory"]
    # the class directory is created exclusively, under the results path
    cd = collab["artifacts"]["classes_dir"]
    assert cd == str(tmp_path / "o.jsonl") + ".t1j_classes" and os.path.isdir(cd)
    assert compiled and compiled[0][0].endswith("/bin/javac")
    # git IS spawned, by the repository precondition, and that is expected. What
    # must never appear is javac or java: assert the ALLOWED set rather than
    # "nothing ran", which was too broad to be true and would have been silenced
    # by relaxing it to a bare count.
    programs = {os.path.basename(str(cmd[0])) for cmd in spawned if cmd}
    assert programs <= {"git"}, f"only git may be spawned, got {sorted(programs)}"
    assert not any(os.path.basename(str(part)) in {"javac", "java"}
                   for cmd in spawned for part in cmd), spawned


def test_the_enabled_path_builds_the_REAL_state_factory_and_binder(
        tmp_path, clean_repo, monkeypatch, unspent_block):
    """Without substituting them, so the real constructors actually run.

    The previous test substituted `_state_factory` and `_binder`, and `or`
    short-circuits -- so `INT.make_state_factory(...)` was never called and a
    control that broke its arguments passed vacuously. Constructing these is
    inert: both return closures and neither touches Java.
    """
    import subprocess as _sub
    spawned = []
    real_run = _sub.run
    monkeypatch.setattr(_sub, "run",
                        lambda *a, **k: (spawned.append(a[0] if a else k.get("args")),
                                         real_run(*a, **k))[1])
    rc, cap, sentinel, _ = enabled(tmp_path, clean_repo, monkeypatch,
                                   _state_factory=None, _binder=None)
    collab = cap.kw["_setup"]()                  # the REAL constructors run here
    assert callable(collab["state_factory"]) and callable(collab["binder"])
    # the state factory really did receive the frozen openings
    from scripts.GPU.alphazero import e4_screen_integration as INT
    frozen_plan = P.load_l0_plan()
    with pytest.raises(H.AbortError, match="unknown opening"):
        collab["state_factory"]({"task_id": "x", "opening": "not_an_opening"})
    st = collab["state_factory"](dict(frozen_plan["tasks"][0]))
    assert st.ply == frozen_plan["opening_plies"] == 6
    programs = {os.path.basename(str(cmd[0])) for cmd in spawned if cmd}
    assert programs <= {"git"}, sorted(programs)


def test_the_enabled_path_setup_refuses_an_existing_class_directory(
        tmp_path, clean_repo, monkeypatch, unspent_block):
    rc, cap, _, _ = enabled(tmp_path, clean_repo, monkeypatch)
    setup = cap.kw["_setup"]
    setup()
    with pytest.raises(H.AbortError) as e:
        setup()                                      # a second must refuse, not merge
    assert e.value.phase == H.PHASE_SETUP and "already exists" in e.value.message


def test_the_enabled_path_constructs_no_rng(tmp_path, clean_repo, monkeypatch, unspent_block):
    import random
    made = []

    class _Counting(random.Random):
        def __init__(self, *a, **kw):
            made.append(a)
            super().__init__(*a, **kw)

    monkeypatch.setattr(random, "Random", _Counting)
    rc, cap, _, _ = enabled(tmp_path, clean_repo, monkeypatch)
    cap.kw["_setup"]()
    assert made == [], f"{len(made)} RNG(s) constructed on the enabled path"


def test_the_enabled_path_touches_no_scheduled_seed(tmp_path, clean_repo, monkeypatch,
                                                    unspent_block):
    """It draws from nothing: the harness is substituted, so no game is played.

    Under the lift the block reads unexposed, so exposure alone proves little
    here. What binds is that the registries are exactly what the fixture set --
    the enabled path modified neither.
    """
    before = (REF.EXPOSED_SEED_INTERVALS, REF.RETIRED_SEED_INTERVALS)
    rc, cap, _, _ = enabled(tmp_path, clean_repo, monkeypatch)
    cap.kw["_setup"]()
    assert (REF.EXPOSED_SEED_INTERVALS, REF.RETIRED_SEED_INTERVALS) == before


def test_the_gate_is_restored_after_the_enabled_path_tests():
    """monkeypatch undoes the lift; this catches a leak."""
    assert C.L0_EXECUTION_AUTHORIZED is False


# --- match mode: receipts are exclusive to qualify ---------------------------

def _frozen_rows(frozen, wins_every=8, caps=0):
    out = []
    for i, t in enumerate(frozen["tasks"]):
        if i < caps:
            out.append({"task_id": t["task_id"], "seed": t["seed"],
                        "plies": RULES.PLY_CAP, "terminal_reason": "cap",
                        "winner": None, "t1j_points": 0.5})
            continue
        win = i % wins_every != 0
        other = "black" if t["anchor_colour"] == "red" else "red"
        out.append({"task_id": t["task_id"], "seed": t["seed"], "plies": 40,
                    "terminal_reason": "win",
                    "winner": t["anchor_colour"] if win else other,
                    "t1j_points": 1.0 if win else 0.0})
    return out


class _Rec:
    def __init__(self):
        self.records = []

    def emit(self, r):
        self.records.append(r)

    def emit_terminal(self, r):
        self.records.append(r)
        return None


def test_the_cap_saturated_outcome_string_matches_what_the_rules_emit(frozen):
    """Binds the runner's MIRRORED constant to what the frozen reporter emits.

    The rules module names it only as a literal and L1 may not modify the frozen
    statistical rules to add a constant, so the duplication is bound by a test.
    """
    rep = RULES.match_report(_frozen_rows(frozen, caps=33), frozen["tasks"])
    assert rep["reported"] is False
    assert rep["outcome"] == R.CAP_SATURATED_NO_RATE


def test_match_mode_records_cap_saturation_as_a_MATCH_OUTCOME_and_exits_0(frozen):
    rec = _Rec()
    rc = R._report(rec, frozen, frozen["tasks"], _frozen_rows(frozen, caps=33),
                   R.MATCH_MODE, cleanups=64)
    assert rc == H.EXIT_OK
    kinds = [r["record_type"] for r in rec.records]
    assert kinds == ["match_outcome"]
    assert "qualification_receipt" not in kinds
    assert rec.records[0]["outcome"] == R.CAP_SATURATED_NO_RATE
    assert rec.records[0]["cap_terminations"] == 33


def test_match_mode_aborts_on_any_other_reporting_refusal(frozen):
    """An incomplete or malformed canonical match must NOT exit 0."""
    for rows_, why in ((_frozen_rows(frozen)[:63], "unplayed"),
                       ([dict(r, t1j_points=0.25) if i == 0 else r
                         for i, r in enumerate(_frozen_rows(frozen))], "t1j_points"),
                       ([], "unplayed")):
        rec = _Rec()
        with pytest.raises(H.AbortError) as e:
            R._report(rec, frozen, frozen["tasks"], rows_, R.MATCH_MODE, cleanups=0)
        assert e.value.phase == H.PHASE_CLASSIFY
        assert why in e.value.message
        assert not any(r["record_type"] == "qualification_receipt" for r in rec.records)


def test_match_mode_reports_normally_when_the_match_is_whole(frozen):
    """The control: match mode must not have become a blanket abort."""
    rec = _Rec()
    rc = R._report(rec, frozen, frozen["tasks"], _frozen_rows(frozen), R.MATCH_MODE,
                   cleanups=64)
    assert rc == H.EXIT_OK
    assert [r["record_type"] for r in rec.records] == ["match_report"]
    assert rec.records[0]["overall"]["t1j_score"] == 56.0


def test_a_qualification_receipt_can_never_appear_in_match_mode(frozen):
    src = inspect.getsource(R._report)
    assert "qualification_receipt" in src
    for rows_ in (_frozen_rows(frozen)[:10], _frozen_rows(frozen, caps=33),
                  _frozen_rows(frozen)):
        rec = _Rec()
        try:
            R._report(rec, frozen, frozen["tasks"], rows_, R.MATCH_MODE, cleanups=0)
        except H.AbortError:
            pass
        assert not any(r["record_type"] == "qualification_receipt"
                       for r in rec.records), rows_[:1]


def test_qualify_mode_still_gets_a_receipt_and_exits_0(frozen):
    rec = _Rec()
    rc = R._report(rec, frozen, [syn(0)], [], "qualify", cleanups=0)
    assert rc == H.EXIT_OK
    assert [r["record_type"] for r in rec.records] == ["qualification_receipt"]


def test_zzz_the_registries_are_restored(tmp_path, clean_repo):
    """Runs last. Every lift above must have been undone, exactly.

    Compared against the snapshot taken at import, not against a re-derived
    expectation -- a check that recomputes what it is checking cannot fail.
    """
    assert _registries_match_snapshot(), "a lift leaked out of its fixture"
    assert P.L0_SEED_BLOCK in REF.EXPOSED_SEED_INTERVALS
    assert P.L0_SEED_BLOCK in REF.RETIRED_SEED_INTERVALS
    for seed in range(*P.L0_SEED_BLOCK):
        assert REF.seed_is_exposed(seed) and REF.seed_is_retired(seed)
    assert C.L0_EXECUTION_AUTHORIZED is False
    assert SCREEN_CMD.SCREEN_AUTHORIZED is False
    # and the real command still refuses the spent block, unpatched
    with pytest.raises(C.PreconditionError) as e:
        C._run_match(**args(clean_repo, tmp_path))
    assert e.value.which == "schedule"


def test_zzz_the_snapshot_assertion_actually_binds(monkeypatch):
    """THE MUTATION CONTROL. A snapshot check that cannot fail proves nothing."""
    assert _registries_match_snapshot()
    monkeypatch.setattr(REF, "EXPOSED_SEED_INTERVALS",
                        REF.EXPOSED_SEED_INTERVALS + ((1, 2),))
    assert not _registries_match_snapshot(), "the snapshot check must notice a change"
    monkeypatch.undo()
    assert _registries_match_snapshot()
