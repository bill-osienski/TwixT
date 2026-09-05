"""The H1 runner. THE MATCH IS NOT AUTHORIZED AND IS NEVER RUN HERE.

No model is loaded, no JVM started, no game played, no seed drawn, no registry
edited. Every collaborator is a mock injected through the private seam, and the
match path is reached only with the gate and the registry monkeypatched INSIDE a
test -- never by editing either for real.
"""
import json
import os

import pytest

from scripts.GPU.alphazero import e4_screen_reference as REF
from scripts.GPU.alphazero import e4_screen_runner as H
from scripts.GPU.alphazero import h1_viability_plan as PLAN
from scripts.GPU.alphazero import h1_viability_rules as RULES
from scripts.GPU.alphazero import h1_viability_runner as RUN
from scripts.GPU.alphazero import t1j_adapter as A
from scripts.GPU.alphazero import void_trace as VT


# ══════════════════════════ the gate, as published ═══════════════════════════

def test_the_gate_is_false_as_published():
    assert RUN.H1_EXECUTION_AUTHORIZED is False


def test_the_PUBLIC_entry_point_reaches_the_GATE_not_a_mode_list():
    """🔴 [P1] The match must be publicly REACHABLE, refused by the gate.

    An earlier version listed only "qualify", so `mode="match"` was rejected by a
    MODE LIST before either barrier was consulted, and every match test had to
    reach the private `_run`. Opening the gate and registering the block must be
    SUFFICIENT to start the match through `run()`; a hidden mode list would be a
    third barrier nobody can see the state of.
    """
    assert RUN.MATCH_MODE in RUN.MODES
    with pytest.raises(RUN.H1Error, match="H1_EXECUTION_AUTHORIZED is False"):
        RUN.run("/nonexistent/results.jsonl", mode=RUN.MATCH_MODE)


def test_an_unknown_mode_is_still_refused():
    with pytest.raises(RUN.H1Error, match="not permitted"):
        RUN.run("/nonexistent/results.jsonl", mode="whatever")


def test_the_public_match_path_supplies_the_PRODUCTION_setup(monkeypatch, tmp_path):
    """🔴 [P1] On the real path `_setup` was None, leaving the REFUSING binder,
    agent factory and state factory in place -- a match would have aborted on its
    first ply, and no successful test could reveal it because they all replace
    `play_task`."""
    seen = {}
    monkeypatch.setattr(RUN, "_run", lambda *a, **k: seen.update(k) or 0)
    RUN.run(str(tmp_path / "r.jsonl"), mode=RUN.MATCH_MODE)
    assert seen["_setup_factory"] is RUN._production_setup
    RUN.run(str(tmp_path / "q.jsonl"), mode="qualify")
    assert seen["_setup_factory"] is None, \
        "qualification must not build production collaborators"


def test_the_public_entry_point_takes_no_plan_and_no_collaborators():
    import inspect
    params = set(inspect.signature(RUN.run).parameters)
    assert params == {"results_path", "mode", "trace_path"}


def test_this_module_reads_NO_OTHER_experiments_gate():
    src = open("scripts/GPU/alphazero/h1_viability_runner.py").read()
    for other in ("D1_EXECUTION_AUTHORIZED", "L0_EXECUTION_AUTHORIZED",
                  "SCREEN_AUTHORIZED", "LOWPLY_QUALIFICATION_AUTHORIZED"):
        assert other not in src, other


# ═══════════════════ 🔴 BOTH BARRIERS FIRE BEFORE ANYTHING ═══════════════════

@pytest.fixture
def spy(monkeypatch, tmp_path):
    """Records whether ANY effectful step was reached."""
    seen = {"recorder": 0, "plan_load": 0, "setup": 0, "play": 0}

    real_recorder = H.Recorder

    class SpyRecorder(real_recorder):
        def __init__(self, path):
            seen["recorder"] += 1
            super().__init__(path)

    monkeypatch.setattr(H, "Recorder", SpyRecorder)
    monkeypatch.setattr(RUN.H, "Recorder", SpyRecorder)

    real_load = PLAN.load_h1_plan

    def spy_load(path=PLAN.H1_PLAN_REL):
        seen["plan_load"] += 1
        return real_load(path)

    monkeypatch.setattr(RUN.PLAN, "load_h1_plan", spy_load)
    monkeypatch.setattr(H, "play_task",
                        lambda **kw: seen.__setitem__("play", seen["play"] + 1))
    monkeypatch.setattr(RUN.H, "play_task",
                        lambda **kw: seen.__setitem__("play", seen["play"] + 1))
    seen["out"] = str(tmp_path / "results.jsonl")
    return seen


def _setup_that_must_not_run(seen):
    def _s():
        seen["setup"] += 1
        raise AssertionError("setup ran despite a closed barrier")
    return _s


def test_THE_GATE_fires_before_plan_load_setup_recorder_or_play(spy, tmp_path):
    """Barrier 1, and the ORDER is the guarantee: a refusal that has already
    created a results file has not refused."""
    with pytest.raises(RUN.H1Error, match="NOT AUTHORIZED"):
        RUN._run(spy["out"], mode=RUN.MATCH_MODE,
                 trace_path=str(tmp_path / "auto_trace_1.jsonl"), _setup=_setup_that_must_not_run(spy))
    assert spy["plan_load"] == 0, "the plan was read before the gate"
    assert spy["setup"] == 0, "setup ran before the gate"
    assert spy["recorder"] == 0, "the recorder was constructed before the gate"
    assert spy["play"] == 0, "a game was played before the gate"
    assert not os.path.exists(spy["out"]), "a results file exists after a refusal"


def test_THE_REGISTRATION_BARRIER_fires_before_plan_load_setup_recorder_or_play(
        spy, monkeypatch, tmp_path):
    """Barrier 2, with the gate OPEN, so it is reached alone. Opening one barrier
    must not open the other."""
    monkeypatch.setattr(RUN, "H1_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS", _registry_without_h1())
    with pytest.raises(RUN.H1Error, match="not registered"):
        RUN._run(spy["out"], mode=RUN.MATCH_MODE,
                 trace_path=str(tmp_path / "auto_trace_2.jsonl"), _setup=_setup_that_must_not_run(spy))
    assert spy["plan_load"] == 0 and spy["setup"] == 0
    assert spy["recorder"] == 0 and spy["play"] == 0
    assert not os.path.exists(spy["out"])


def test_REGISTERING_THE_BLOCK_DOES_NOT_OPEN_THE_GATE(spy, monkeypatch):
    """The two are independent, and this is now the REAL repository state: the
    block is registered and the gate is still shut. Nothing is patched."""
    RUN.check_seed_registration()                     # barrier 2 REALLY satisfied
    assert RUN.H1_EXECUTION_AUTHORIZED is False
    with pytest.raises(RUN.H1Error, match="NOT AUTHORIZED"):
        RUN._run(spy["out"], mode=RUN.MATCH_MODE)
    assert spy["recorder"] == 0 and not os.path.exists(spy["out"])


def test_a_PARTLY_registered_block_is_still_refused(monkeypatch):
    """🔴 BOTH ENDPOINTS REGISTERED, A HOLE IN THE MIDDLE.

    The first version registered [lo, hi-1), which leaves hi-1 unaccounted -- so
    an endpoints-only check ALSO refused, and the control that reduced the scan
    to the endpoints was NOT CAUGHT. Only a hole strictly inside the block can
    tell a full scan from an endpoint check.
    """
    lo, hi = RULES.H1_SEED_BLOCK
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS",
                        _registry_without_h1() + ((lo, lo + 1), (hi - 1, hi)))
    assert REF.seed_is_accounted(lo) and REF.seed_is_accounted(hi - 1)
    assert not REF.seed_is_accounted(lo + 1)          # the hole
    with pytest.raises(RUN.H1Error, match="not registered"):
        RUN.check_seed_registration()


def test_the_registration_check_READS_the_registry_and_never_writes_it(monkeypatch):
    """Driven through the REFUSING branch -- the one that could be tempted to fix
    what it found. The block is registered now, so the refusal is arranged by
    stripping it."""
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS", _registry_without_h1())
    before = tuple(REF.ACCOUNTED_SEED_INTERVALS)
    with pytest.raises(RUN.H1Error, match="not registered"):
        RUN.check_seed_registration()
    assert tuple(REF.ACCOUNTED_SEED_INTERVALS) == before


# ═══════════════ the frozen design: 224, no early stop, timeouts ═════════════

@pytest.fixture(scope="module")
def frozen():
    return PLAN.load_h1_plan()


def _registry_without_h1():
    """The ACCOUNTED tuple as it stood BEFORE the seed-preparation authorization.

    The negative controls below need a registry that does NOT contain the block.
    It is really registered now, so they must REMOVE it rather than pretend --
    and this asserts the removal happened, so a control cannot go vacuous if the
    interval is ever restructured.
    """
    out = tuple(i for i in REF.ACCOUNTED_SEED_INTERVALS
                if tuple(i) != tuple(RULES.H1_SEED_BLOCK))
    assert len(out) == len(REF.ACCOUNTED_SEED_INTERVALS) - 1, \
        "the H1 block is not in ACCOUNTED as a single interval; these controls are stale"
    return out


#: A SNAPSHOT of the seed registries at import, so the lift below is measured
#: against something and `test_zzz_the_registries_are_restored` can prove it was
#: undone.
_REGISTRY_SNAPSHOT = {
    "exposed": REF.EXPOSED_SEED_INTERVALS,
    "retired": REF.RETIRED_SEED_INTERVALS,
    "accounted": REF.ACCOUNTED_SEED_INTERVALS,
}


@pytest.fixture
def unspent_block(monkeypatch):
    """Lifts the H1 block's ELIGIBILITY in process, and nothing else.

    WHY THIS IS NOT RELAXING A GATE. The match ran on 2026-09-05 and VOIDED, so
    the block is now EXPOSED (60 seeds drawn) and RETIRED (all 224), and
    `validate_schedule_executable` refuses it before authorization is ever
    consulted. Left alone, every test of a LATER precondition, of the GATE, and
    of the enabled-path wiring would stop passing for a reason that has nothing
    to do with what it tests -- and a safety gate whose tests all went vacuous is
    the failure this workstream keeps finding.

    The real-state refusal is asserted SEPARATELY and UNPATCHED, in
    `test_the_SPENT_schedule_is_refused_in_the_REAL_state`.

    IT TOUCHES ELIGIBILITY ONLY -- never H1_EXECUTION_AUTHORIZED, asserted on the
    way in and on the way out.
    """
    gate = RUN.H1_EXECUTION_AUTHORIZED
    blk = tuple(RULES.H1_SEED_BLOCK)
    monkeypatch.setattr(REF, "EXPOSED_SEED_INTERVALS",
                        tuple(i for i in REF.EXPOSED_SEED_INTERVALS
                              if tuple(i) != (blk[0], blk[0] + 60)))
    monkeypatch.setattr(REF, "RETIRED_SEED_INTERVALS",
                        tuple(i for i in REF.RETIRED_SEED_INTERVALS
                              if tuple(i) != blk))
    assert not REF.seed_is_unavailable(blk[0]), "the lift did not take"
    assert RUN.H1_EXECUTION_AUTHORIZED is gate is False
    yield
    assert RUN.H1_EXECUTION_AUTHORIZED is False or gate is False


def test_the_SPENT_schedule_is_refused_in_the_REAL_state(tmp_path, monkeypatch):
    """UNPATCHED. The block is spent, so the match cannot run again even with the
    gate open -- which is the protection the fixture above deliberately lifts."""
    monkeypatch.setattr(RUN, "H1_EXECUTION_AUTHORIZED", True)
    with pytest.raises(RUN.H1Error, match="may not be executed"):
        RUN._run(str(tmp_path / "r.jsonl"), mode=RUN.MATCH_MODE,
                 trace_path=str(tmp_path / "t.jsonl"), _supervise=False)


def test_zzz_the_registries_are_restored():
    """Every lift above is undone. Runs last by name."""
    assert REF.EXPOSED_SEED_INTERVALS == _REGISTRY_SNAPSHOT["exposed"]
    assert REF.RETIRED_SEED_INTERVALS == _REGISTRY_SNAPSHOT["retired"]
    assert REF.ACCOUNTED_SEED_INTERVALS == _REGISTRY_SNAPSHOT["accounted"]


@pytest.fixture
def openable(monkeypatch, unspent_block):
    """Gate open, INSIDE the test only. The real gate constant is never edited.

    ⚠ INVERTED by the seed-preparation authorization: this used to patch the
    block into ACCOUNTED as well. The block is REGISTERED now, so arranging it
    would hide a later de-registration behind a fixture that silently supplies
    what it is meant to depend on. It asserts the fact instead -- and fails, on
    purpose, the day the block leaves the registry.
    """
    assert tuple(RULES.H1_SEED_BLOCK) in {tuple(i)
                                          for i in REF.ACCOUNTED_SEED_INTERVALS}, \
        "the H1 block is no longer registered; this fixture must arrange again"
    monkeypatch.setattr(RUN, "H1_EXECUTION_AUTHORIZED", True)


def _rows(tasks, t1j_wins):
    out = []
    for i, t in enumerate(tasks):
        won = i < t1j_wins
        other = "black" if t["anchor_colour"] == "red" else "red"
        out.append({"task_id": t["task_id"], "seed": t["seed"], "plies": 40,
                    "terminal_reason": "win",
                    "winner": t["anchor_colour"] if won else other,
                    "t1j_points": 1.0 if won else 0.0})
    return out


def _drive(monkeypatch, tasks, t1j_wins):
    """Make play_task emit the i-th prepared outcome. No agent, no state, no seed."""
    rows = {r["task_id"]: r for r in _rows(tasks, t1j_wins)}

    def fake_play(*, task, **kw):
        r = rows[task["task_id"]]
        return {"winner": r["winner"], "terminal_reason": r["terminal_reason"],
                "plies": r["plies"], "t1j_points": r["t1j_points"]}

    monkeypatch.setattr(RUN.H, "play_task", fake_play)


def test_a_complete_match_reports_and_carries_the_frozen_verdict(
        tmp_path, monkeypatch, openable, frozen):
    _drive(monkeypatch, frozen["tasks"], 133)          # 133/224 = 0.5938, L0's rate
    out = str(tmp_path / "r.jsonl")
    assert RUN._run(out, mode=RUN.MATCH_MODE,
                 trace_path=str(tmp_path / "auto_trace_4.jsonl"), _supervise=False,
                    _cleanup=lambda: None) == H.EXIT_OK
    recs = [json.loads(l) for l in open(out)]
    rep = [r for r in recs if r["record_type"] == "viability_report"]
    assert len(rep) == 1
    assert rep[0]["verdict"] == "VIABLE"
    assert rep[0]["overall"]["games"] == 224
    assert len([r for r in recs if r["record_type"] == "task_result"]) == 224


def test_the_runner_computes_NO_STATISTIC_of_its_own():
    import inspect
    src = inspect.getsource(RUN)
    assert "RULES.viability_report(" in src
    for forbidden in ("wilson", "hoeffding", "sqrt", "viability_verdict",
                      "VIABILITY_THRESHOLD"):
        assert forbidden not in src, forbidden


def test_there_is_no_early_stop_and_no_skip_path():
    import inspect
    src = inspect.getsource(RUN)
    assert RULES.EARLY_STOP is None and RULES.may_stop_early() is False
    # ⚠ Forbid the CALL, not the NAME. A blanket ban on the substring
    # forbade the module docstring from EXPLAINING the prohibition it
    # enforces -- the first version of this test failed on its own module
    # saying "may_stop_early is a constant False".
    assert "may_stop_early(" not in src
    for forbidden in ("task_skipped", "\n            break", "\n            continue"):
        assert forbidden not in src, forbidden


def test_the_screen_decision_functions_still_exist_and_are_not_imported():
    """The prohibition must not rot into a reference to nothing."""
    from scripts.GPU.alphazero import e4_screen_rules as SR
    import inspect
    src = inspect.getsource(RUN)
    for dotted in RULES.MUST_NEVER_BE_CALLED_ON_AN_H1_RUN:
        name = dotted.split(".")[1]
        assert hasattr(SR, name), f"{dotted} no longer exists; the ban is vacuous"
        assert name not in src, dotted


def test_the_frozen_limits_are_the_cards():
    assert RUN.RUN_DEADLINE_S == 180 * 60
    assert RUN.PER_CALL_TIMEOUT_S == 120
    assert RULES.N_GAMES == 224


def test_the_run_header_records_the_limits_it_ran_under(
        tmp_path, monkeypatch, openable, frozen):
    _drive(monkeypatch, frozen["tasks"], 100)
    out = str(tmp_path / "r.jsonl")
    RUN._run(out, mode=RUN.MATCH_MODE,
                 trace_path=str(tmp_path / "auto_trace_5.jsonl"), _supervise=False, _cleanup=lambda: None)
    hdr = json.loads(open(out).readline())
    assert hdr["record_type"] == "run_header"
    assert hdr["per_call_timeout_s"] == 120
    assert hdr["run_deadline_s"] == 180 * 60
    assert hdr["n_games"] == 224 and hdr["early_stop"] is None
    assert hdr["task_digest"] == RULES.H1_TASK_DIGEST
    assert hdr["plan_sha256"] == PLAN.H1_PLAN_SHA256
    assert hdr["seed_block"] == list(RULES.H1_SEED_BLOCK)


# ══════════════ create-only outputs and the non-analytic VOID trace ══════════

def test_the_trace_file_is_CREATE_ONLY(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("")
    with pytest.raises(FileExistsError):
        with RUN._trace_file(str(p)):
            pass


# NOTE: the create-only results check lives in
# `test_a_PREEXISTING_output_path_is_a_PRECONDITION_refusal`, which asserts the
# EXACT exception type. The version here used `pytest.raises(Exception)` and so
# could not tell a precondition refusal from a VOID -- which is precisely the
# mismatch review found.


def test_the_trace_carries_ONLY_counters_and_closed_enums():
    assert RUN.TRACE_SCHEMA == "h1-void-trace/1"
    assert set(RUN.H1_TRACE.fields) == {
        "event", "ts", "schema", "n_games", "index", "games_completed", "verdict"}
    for forbidden in ("task_id", "seed", "opening", "colour_arm", "winner",
                      "t1j_points", "rate", "verdict_value", "score"):
        assert forbidden not in RUN.H1_TRACE.fields, forbidden


@pytest.mark.parametrize("event,fields", [
    ("run_start", {"schema": "h1-void-trace/1", "n_games": 224, "note": "x"}),
    ("task_start", {"index": "game=(11,11)"}),
    ("task_done", {"index": 0, "games_completed": True}),
    ("run_end", {"verdict": "PROBABLY", "games_completed": 0}),
    ("exfiltrate", {"index": 0}),
    ("task_done", {"index": 224, "games_completed": 0}),
    ("run_start", {"schema": "h1-void-trace/2", "n_games": 1}),
])
def test_the_trace_refuses_anything_that_could_carry_a_MEASUREMENT(event, fields):
    with pytest.raises(RUN.H1Error):
        VT.check(RUN.H1_TRACE, event, fields)


def test_a_refusal_writes_NOTHING_to_the_trace(tmp_path):
    p = str(tmp_path / "t.jsonl")
    with RUN._trace_file(p) as fh:
        with pytest.raises(RUN.H1Error):
            RUN._trace(fh, event="task_start", index=0, leak="secret")
    assert open(p).read() == "", "a refusal that has already appended has not refused"


def test_the_trace_records_HOW_FAR_a_void_run_got_and_nothing_else(
        tmp_path, monkeypatch, openable, frozen):
    """The gap D1's VOID left: no record of where it died."""
    calls = {"n": 0}

    def explode(*, task, **kw):
        calls["n"] += 1
        if calls["n"] > 3:
            raise H.AbortError(H.PHASE_SETUP, "instrument failure")
        return {"winner": task["anchor_colour"], "terminal_reason": "win",
                "plies": 40, "t1j_points": 1.0}

    monkeypatch.setattr(RUN.H, "play_task", explode)
    out, tr = str(tmp_path / "r.jsonl"), str(tmp_path / "t.jsonl")
    with pytest.raises(H.AbortError):
        RUN._run(out, mode=RUN.MATCH_MODE, trace_path=tr, _supervise=False,
                 _cleanup=lambda: None)
    lines = [json.loads(l) for l in open(tr)]
    assert lines[0]["event"] == "run_start" and lines[0]["n_games"] == 224
    assert lines[-1]["event"] == "run_end"
    assert lines[-1]["verdict"] == "VOID"
    assert lines[-1]["games_completed"] == 3      # died on the 4th game
    assert all(set(l) <= set(RUN.H1_TRACE.fields) for l in lines)
    assert "viability_report" not in open(out).read()


# ═══════════════════════════ the 180-minute deadline ═════════════════════════

def test_a_deadline_breach_is_a_VOID_with_no_report(tmp_path, monkeypatch,
                                                    openable, frozen):
    from scripts.GPU.alphazero import d1_probe as D1
    clock = {"t": 0.0}
    dl = D1.Deadline(RUN.RUN_DEADLINE_S, clock=lambda: clock["t"])
    dl.start()
    _drive(monkeypatch, frozen["tasks"], 10)

    real = RUN.H.play_task

    def slow(**kw):
        clock["t"] += RUN.RUN_DEADLINE_S / 2
        return real(**kw)

    monkeypatch.setattr(RUN.H, "play_task", slow)
    out, tr = str(tmp_path / "r.jsonl"), str(tmp_path / "t.jsonl")
    with pytest.raises(RUN.H1VoidError, match="VOID"):
        RUN._run(out, mode=RUN.MATCH_MODE, trace_path=tr, _deadline=dl,
                 _supervise=False, _cleanup=lambda: None)
    assert "viability_report" not in open(out).read()
    assert json.loads(open(tr).read().splitlines()[-1])["verdict"] == "VOID"


def test_the_COOPERATIVE_check_does_not_translate_and_the_boundary_does(monkeypatch):
    """ONE translation point. `_check_deadline` raises D1's type; the run
    boundary converts it, so cooperative and asynchronous breaches leave by the
    same door. Two conversions meant the inner one could never fire, and a
    control removing it was NOT CAUGHT until they were collapsed."""
    from scripts.GPU.alphazero import d1_probe as D1
    dl = D1.Deadline(1.0, clock=lambda: 0.0)
    dl.start()
    monkeypatch.setattr(dl, "elapsed", lambda: 999.0)
    with pytest.raises(D1.D1VoidError):
        RUN._check_deadline(dl, "here")          # unconverted, by design


# ══════════════════ match mode loads ONLY the pinned v3 plan ═════════════════

def test_match_mode_REFUSES_a_supplied_plan_path(tmp_path, openable):
    with pytest.raises(RUN.H1Error, match="pinned v3 plan only"):
        RUN._run(str(tmp_path / "r.jsonl"), mode=RUN.MATCH_MODE,
                 trace_path=str(tmp_path / "auto_trace_6.jsonl"),
                 _plan_path=str(tmp_path / "other.json"))


def test_match_mode_loads_the_pinned_plan_without_being_told_where(
        tmp_path, monkeypatch, openable, frozen):
    seen = {}
    real = PLAN.load_h1_plan
    monkeypatch.setattr(RUN.PLAN, "load_h1_plan",
                        lambda p=PLAN.H1_PLAN_REL: (seen.setdefault("p", p), real(p))[1])
    _drive(monkeypatch, frozen["tasks"], 10)
    RUN._run(str(tmp_path / "r.jsonl"), mode=RUN.MATCH_MODE,
                 trace_path=str(tmp_path / "auto_trace_7.jsonl"), _supervise=False,
             _cleanup=lambda: None)
    assert seen["p"] == PLAN.H1_PLAN_REL


def test_a_schedule_that_is_not_the_frozen_224_is_refused(tmp_path, openable, frozen):
    short = list(frozen["tasks"])[:-1]
    with pytest.raises(RUN.H1Error, match="expected exactly"):
        RUN._run(str(tmp_path / "r.jsonl"), mode=RUN.MATCH_MODE,
                 trace_path=str(tmp_path / "auto_trace_8.jsonl"), _tasks=short)


def test_a_task_renamed_onto_synthetic_CONTENT_is_refused(tmp_path, openable, frozen):
    """Canonical NAMES attached to synthetic CONTENT counted in an earlier
    workstream. The binding is by FULL content, not by task_id.

    🔴 REACHES THE CONTENT CHECK ALONE. The first version mutated `opening`,
    which is a DIGEST dimension -- so the digest comparison above caught it and
    a control narrowing this check back to the digest dimensions was NOT CAUGHT.
    `reference_sha256` is NOT in the digest and NOT checked by the plan
    validator (both verified), so only the full-key comparison can refuse it.
    """
    tasks = [dict(t) for t in frozen["tasks"]]
    tasks[7] = dict(tasks[7], reference_sha256="deadbeef" * 8)
    assert RULES.L0.l0_task_digest(tasks) == RULES.H1_TASK_DIGEST   # digest blind
    PLAN.validate_h1_schedule(tasks)                                # validator blind
    with pytest.raises(RUN.H1Error, match="does not match the frozen plan"):
        RUN._run(str(tmp_path / "r.jsonl"), mode=RUN.MATCH_MODE,
                 trace_path=str(tmp_path / "auto_trace_9.jsonl"), _tasks=tasks)


def test_a_seed_outside_the_reserved_block_is_refused(tmp_path, openable, frozen):
    """Asserted at the PLAN VALIDATOR, which is the single enforcement point.

    🔴 Through the runner alone this could not isolate the block check: a changed
    seed changes the DIGEST, and L0's block is additionally exposed and retired,
    so two other guards fire first and a control disabling the block check was
    NOT CAUGHT. 202617000 is outside every registry, so only the block check can
    refuse it.
    """
    tasks = [dict(t) for t in frozen["tasks"]]
    tasks[3] = dict(tasks[3], seed=202617000)
    with pytest.raises(PLAN.H1PlanError, match="outside"):
        PLAN.validate_h1_schedule(tasks)
    # and the runner refuses it too, by whichever guard reaches it first
    with pytest.raises(RUN.H1Error):
        RUN._run(str(tmp_path / "r.jsonl"), mode=RUN.MATCH_MODE,
                 trace_path=str(tmp_path / "auto_trace_10.jsonl"), _tasks=tasks)


# ══════════════════════ the preregistered no-rate outcome ════════════════════

def test_the_cap_saturated_outcome_string_matches_what_the_rules_EMIT(frozen):
    """Mirrored, not imported -- so it is bound by running the real reporter over
    a cap-heavy vector rather than by trusting the literal."""
    rows = _rows(frozen["tasks"], 0)
    for r in rows[:113]:
        r.update(winner=None, terminal_reason="cap", t1j_points=0.5,
                 plies=RULES.PLY_CAP)
    rep = RULES.viability_report(rows, frozen["tasks"])
    assert rep["reported"] is False
    assert rep["outcome"] == RUN.CAP_SATURATED_NO_RATE


def test_cap_saturation_exits_OK_and_publishes_NO_VERDICT(
        tmp_path, monkeypatch, openable, frozen):
    def capped(*, task, **kw):
        i = int(task["task_id"].split("-")[1])
        if i < 113:
            return {"winner": None, "terminal_reason": "cap",
                    "plies": RULES.PLY_CAP, "t1j_points": 0.5}
        other = "black" if task["anchor_colour"] == "red" else "red"
        return {"winner": other, "terminal_reason": "win", "plies": 40,
                "t1j_points": 0.0}

    monkeypatch.setattr(RUN.H, "play_task", capped)
    out = str(tmp_path / "r.jsonl")
    assert RUN._run(out, mode=RUN.MATCH_MODE,
                 trace_path=str(tmp_path / "auto_trace_11.jsonl"), _supervise=False,
                    _cleanup=lambda: None) == H.EXIT_OK
    recs = [json.loads(l) for l in open(out)]
    outcomes = [r for r in recs if r["record_type"] == "match_outcome"]
    assert len(outcomes) == 1
    assert outcomes[0]["outcome"] == RUN.CAP_SATURATED_NO_RATE
    assert not any(r["record_type"] == "viability_report" for r in recs)
    assert all("verdict" not in r for r in recs)


# ═══════════ [P1] the production setup, mocked at the real boundaries ════════

def test_the_production_setup_wires_every_qualified_collaborator(monkeypatch,
                                                                 tmp_path, frozen):
    """Mocked at the PROCESS and MODEL boundaries only: no javac, no JVM, no
    checkpoint. Everything between is the real wiring."""
    from scripts.GPU.alphazero import e4_screen_command as SCREEN_CMD
    from scripts.GPU.alphazero import e4_screen_integration as INT
    from scripts.GPU.alphazero import t1j_toolchain as TC
    from scripts.GPU.alphazero import d1_probe as D1

    calls = {}
    monkeypatch.setattr(TC, "verified_paths",
                        lambda *a, **k: {"jdk_home": "/jdk", "jar": "/t1j.jar",
                                         "root": "/root", "source": "s", "verified": True})
    def _compile(dl, *, paths):
        calls["compile"] = paths
        calls["compile_deadline"] = dl
        return {"jar_sha256": "x"}

    monkeypatch.setattr(D1, "_default_compile", _compile)
    def _runtime(**kw):
        calls["runtime"] = kw
        return "RUNTIME"

    monkeypatch.setattr(INT, "T1jRuntime", _runtime)
    def _load(root):
        calls["evaluator_loads"] = calls.get("evaluator_loads", 0) + 1
        return "EVAL"

    monkeypatch.setattr(SCREEN_CMD, "_default_load_evaluator", _load)
    monkeypatch.setattr(INT, "make_state_factory", lambda openings, ctx: "STATE")
    monkeypatch.setattr(INT, "make_binder", lambda runtime, ctx: "BINDER")
    def _agents(**kw):
        calls["agent_kw"] = kw
        return "AGENTS"

    monkeypatch.setattr(INT, "make_agent_factory", _agents)

    dl = D1.Deadline(RUN.RUN_DEADLINE_S)
    dl.start()
    got = RUN._production_setup(str(tmp_path / "r.jsonl"), dl)()
    assert calls["compile_deadline"] is dl, \
        "compilation must use the RUN's clock, not a second one"
    assert got["state_factory"] == "STATE"
    assert got["binder"] == "BINDER"            # the E3b binder
    assert got["agent_factory"] == "AGENTS"
    assert got["evaluator"] == "EVAL"
    assert callable(got["cleanup"])
    # T1j gets the frozen 120-second timeout and the frozen ply cap
    assert calls["runtime"]["timeout_s"] == 120
    assert calls["runtime"]["ply_cap"] == RULES.PLY_CAP
    # the toolchain is RESOLVED, not supplied, and compiled against
    assert calls["compile"].jar == "/t1j.jar"
    assert calls["compile"].java == "/jdk/bin/java"
    assert calls["compile"].ply_cap == RULES.PLY_CAP
    assert got["artifacts"]["per_call_timeout_s"] == 120
    # 🔴 THE AGENT'S OWN QUERY TIMEOUT. `T1jRuntime.timeout_s` bounds REPLAY;
    # `make_agent_factory(t1j_timeout_s=...)` is a SEPARATE argument defaulting
    # to None, and omitting it left the agent querying with no timeout at all
    # while the runtime truthfully reported 120.
    assert calls["agent_kw"]["t1j_timeout_s"] == 120


def test_the_production_setup_loads_the_incumbent_EXACTLY_ONCE(monkeypatch, tmp_path):
    from scripts.GPU.alphazero import e4_screen_command as SCREEN_CMD
    from scripts.GPU.alphazero import e4_screen_integration as INT
    from scripts.GPU.alphazero import t1j_toolchain as TC
    from scripts.GPU.alphazero import d1_probe as D1
    n = {"loads": 0}
    monkeypatch.setattr(TC, "verified_paths", lambda *a, **k: {
        "jdk_home": "/jdk", "jar": "/j.jar", "root": "/r", "source": "s", "verified": True})
    monkeypatch.setattr(D1, "_default_compile", lambda dl, *, paths: {})
    monkeypatch.setattr(INT, "T1jRuntime", lambda **kw: "R")
    monkeypatch.setattr(INT, "make_state_factory", lambda o, c: "S")
    monkeypatch.setattr(INT, "make_binder", lambda r, c: "B")
    monkeypatch.setattr(INT, "make_agent_factory", lambda **kw: "A")

    def load(root):
        n["loads"] += 1
        return "EVAL"

    monkeypatch.setattr(SCREEN_CMD, "_default_load_evaluator", load)
    dl2 = D1.Deadline(RUN.RUN_DEADLINE_S)
    dl2.start()
    RUN._production_setup(str(tmp_path / "r.jsonl"), dl2)()
    assert n["loads"] == 1, "the incumbent must be loaded once for the whole match"


# ═════════════ [P1] the ASYNCHRONOUS alarm must leave as H1's VOID ═══════════

def test_the_SIGALRM_supervisor_raises_H1s_void_not_D1s(tmp_path, monkeypatch,
                                                        openable, frozen):
    """🔴 The REAL alarm, not `_supervise=False`.

    `_check_deadline` translates only the COOPERATIVE breach. The reused
    supervisor fires SIGALRM from inside whatever blocking call is running, and
    that D1VoidError escaped this runner under the wrong experiment's type.
    """
    import time
    from scripts.GPU.alphazero import d1_probe as D1

    dl = D1.Deadline(0.25)
    dl.start()

    def blocks(**kw):
        time.sleep(5)                       # interrupted by the alarm, not by us
        raise AssertionError("the supervisor did not fire")

    monkeypatch.setattr(RUN.H, "play_task", blocks)
    out, tr = str(tmp_path / "r.jsonl"), str(tmp_path / "t.jsonl")
    with pytest.raises(RUN.H1VoidError) as exc:
        RUN._run(out, mode=RUN.MATCH_MODE, trace_path=tr, _deadline=dl,
                 _supervise=True, _cleanup=lambda: None)
    assert not isinstance(exc.value, D1.D1VoidError) or isinstance(exc.value,
                                                                   RUN.H1VoidError)
    assert type(exc.value) is RUN.H1VoidError
    assert json.loads(open(tr).read().splitlines()[-1])["verdict"] == "VOID"


# ═════════════════════ [P1] the VOID record contract ═════════════════════════

@pytest.fixture
def voided(tmp_path, monkeypatch, openable, frozen):
    """A match that dies on game 4 with a helper-style failure."""
    calls = {"n": 0}

    def die(*, task, rec=None, **kw):
        calls["n"] += 1
        if calls["n"] > 3:
            err = A.HelperOutputError("helper said no", "FAIL line 1\nFAIL line 2\n")
            raise err
        return {"winner": task["anchor_colour"], "terminal_reason": "win",
                "plies": 40, "t1j_points": 1.0}

    monkeypatch.setattr(RUN.H, "play_task", die)
    out, tr = str(tmp_path / "r.jsonl"), str(tmp_path / "t.jsonl")
    with pytest.raises(Exception):
        RUN._run(out, mode=RUN.MATCH_MODE, trace_path=tr, _supervise=False,
                 _cleanup=lambda: None)
    return {"recs": [json.loads(l) for l in open(out)], "out": out, "tr": tr}


def test_a_VOID_publishes_NO_REPORT_AND_NO_VERDICT(voided):
    kinds = {r["record_type"] for r in voided["recs"]}
    assert "viability_report" not in kinds
    assert "match_outcome" not in kinds
    assert not any("verdict" in r for r in voided["recs"]
                   if r["record_type"] != "void_diagnostic")


def test_a_VOID_RETAINS_the_raw_rows_it_already_wrote_and_says_so(voided):
    """THE CONTRACT, stated in the module and asserted here: prior rows are KEPT.
    Destroying fsynced evidence on failure would be the worse defect."""
    results = [r for r in voided["recs"] if r["record_type"] == "task_result"]
    assert len(results) == 3, "the three completed games must survive the VOID"
    src = open("scripts/GPU/alphazero/h1_viability_runner.py").read()
    assert "THE VOID RECORD CONTRACT" in src


def test_the_retained_rows_CANNOT_become_a_result(voided, frozen):
    """Enforced structurally, not asked for: the reporter refuses any vector that
    is not all 224, so a partial match cannot yield a rate or a verdict."""
    partial = [{k: v for k, v in r.items() if k != "record_type"}
               for r in voided["recs"] if r["record_type"] == "task_result"]
    rep = RULES.viability_report(partial, frozen["tasks"])
    assert rep["reported"] is False and "unplayed" in rep["reason"]
    assert "verdict" not in rep


def test_the_VOID_diagnostic_NAMES_THE_POSITION_structurally(voided):
    """🔴 D1's VOID named a depth and an invocation and nothing else, so its
    cause could not be settled and still cannot be. Every item the card requires
    is a FIELD here, not prose a human must parse."""
    diag = [r for r in voided["recs"] if r["record_type"] == "void_diagnostic"]
    assert len(diag) == 1
    d = diag[0]
    for field in ("task_id", "index", "opening", "colour_arm", "rep", "seed",
                  "ply", "helper_excerpt", "error_type", "games_completed"):
        assert field in d, field
    assert d["index"] == 3 and d["games_completed"] == 3
    assert d["opening"] and d["colour_arm"] and d["seed"]
    assert d["error_type"] == "HelperOutputError"
    assert "FAIL line 1" in d["helper_excerpt"]


def test_the_VOID_diagnostic_excerpt_is_BOUNDED(tmp_path, monkeypatch, openable,
                                                frozen):
    """A diagnostic must not become the channel the trace refuses to be."""
    huge = "X" * 100_000 + "\n"

    def die(*, task, **kw):
        raise A.HelperOutputError("boom", huge * 50)

    monkeypatch.setattr(RUN.H, "play_task", die)
    out = str(tmp_path / "r.jsonl")
    with pytest.raises(Exception):
        RUN._run(out, mode=RUN.MATCH_MODE,
                 trace_path=str(tmp_path / "auto_trace_12.jsonl"), _supervise=False, _cleanup=lambda: None)
    d = [json.loads(l) for l in open(out)
         if json.loads(l)["record_type"] == "void_diagnostic"][0]
    assert len(d["helper_excerpt"]) <= A.FAILURE_EXCERPT_CHARS + 200


def test_the_VOID_diagnostic_records_the_PLY_it_died_on(tmp_path, monkeypatch,
                                                        openable, frozen):
    """`play_task` reports plies only in an OUTCOME, which a VOID never produces,
    so the durable `ply` records are the only place that number exists."""
    def die(*, task, rec, **kw):
        rec.emit({"record_type": "ply", "task_id": task["task_id"], "ply": 17})
        raise RuntimeError("mid-game instrument failure")

    monkeypatch.setattr(RUN.H, "play_task", die)
    out = str(tmp_path / "r.jsonl")
    with pytest.raises(Exception):
        RUN._run(out, mode=RUN.MATCH_MODE,
                 trace_path=str(tmp_path / "auto_trace_13.jsonl"), _supervise=False, _cleanup=lambda: None)
    d = [json.loads(l) for l in open(out)
         if json.loads(l)["record_type"] == "void_diagnostic"][0]
    assert d["ply"] == 17


# ═══════ [P1] real play_task, only the SUBPROCESS mocked: excerpt and ply ════

def _real_wiring(monkeypatch, replay_stdout=None, query_raises=None):
    """A REAL binder and REAL agents over a small board, with `subprocess.run`
    the only thing mocked. The earlier tests made `play_task` raise
    HelperOutputError directly and so never met the AbortError wrapper that real
    failures pass through."""
    import subprocess
    from scripts.GPU.alphazero import e4_screen_integration as INT
    from scripts.GPU.alphazero.game.twixt_state import TwixtState

    runtime = INT.T1jRuntime(java="/j", jar="/x.jar", classes="/c",
                             ply_cap=280, timeout_s=RUN.PER_CALL_TIMEOUT_S)
    ctx = INT.IntegrationContext()

    def fake_run(args, **kw):
        if "replay" in args and replay_stdout is not None:
            return subprocess.CompletedProcess(args, 0, replay_stdout, "")
        raise AssertionError(f"unexpected subprocess call: {args}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return {"binder": INT.make_binder(runtime, ctx),
            "state_factory": lambda task: TwixtState(active_size=6, to_move="red")}


def test_a_REAL_helper_failure_keeps_a_BOUNDED_excerpt(tmp_path, monkeypatch,
                                                       openable, frozen):
    """🔴 Real query/replay failures arrive as H.AbortError, which has NO stdout.
    Reading `error.stdout` alone produced `helper_excerpt: null` for exactly the
    failures the diagnostic exists for."""
    wiring = _real_wiring(monkeypatch, replay_stdout="GARBAGE NOT A DUMP\n")
    out = str(tmp_path / "r.jsonl")
    with pytest.raises(H.AbortError):
        RUN._run(out, mode=RUN.MATCH_MODE,
                 trace_path=str(tmp_path / "auto_trace_14.jsonl"), _supervise=False, _cleanup=lambda: None,
                 _binder=wiring["binder"], _state_factory=wiring["state_factory"],
                 _agent_factory=lambda t, m, e: (lambda s: (2, 2)))
    d = [json.loads(l) for l in open(out)
         if json.loads(l)["record_type"] == "void_diagnostic"][0]
    assert d["helper_excerpt"], "a real helper failure lost its transcript"
    assert len(d["helper_excerpt"]) <= A.FAILURE_EXCERPT_CHARS + 200
    assert d["error_type"] == "AbortError"


def test_the_excerpt_walks_the_SUPPRESSED_context_chain():
    """`_bind` re-raises `from None`, which clears __cause__ but leaves
    __context__ set. The transcript is still reachable, and this pins that."""
    try:
        try:
            raise A.HelperOutputError("bad", "FAIL: postcond\nsecond line\n")
        except Exception:
            raise H.AbortError("bind", "binder raised HelperOutputError") from None
    except H.AbortError as ab:
        assert ab.__cause__ is None
        got = RUN._bounded_excerpt(ab)
    assert "FAIL: postcond" in got


def test_the_excerpt_falls_back_to_the_bounded_ABORT_MESSAGE():
    """A postcondition or illegal-move abort has no transcript at all; the
    integration layer put the detail in the message, so that is what is kept."""
    got = RUN._bounded_excerpt(H.AbortError("move", "X" * 100_000))
    assert got and len(got) <= A.FAILURE_EXCERPT_CHARS


def test_a_failure_on_the_FIRST_searched_move_still_names_a_PLY(
        tmp_path, monkeypatch, openable, frozen):
    """🔴 The tracker read `ply` records only, so a failure before the first one
    recorded ply=None. `opening_bound` carries the opening's ply and is now read."""
    def die_on_first_move(*, task, rec, **kw):
        rec.emit({"record_type": "opening_bound", "task_id": task["task_id"],
                  "ply": 6, "opening": task["opening"]})
        raise H.AbortError(H.PHASE_MOVE, "T1j returned the null sentinel")

    monkeypatch.setattr(RUN.H, "play_task", die_on_first_move)
    out = str(tmp_path / "r.jsonl")
    with pytest.raises(H.AbortError):
        RUN._run(out, mode=RUN.MATCH_MODE,
                 trace_path=str(tmp_path / "auto_trace_15.jsonl"), _supervise=False, _cleanup=lambda: None)
    d = [json.loads(l) for l in open(out)
         if json.loads(l)["record_type"] == "void_diagnostic"][0]
    assert d["ply"] == 6, "a first-move failure must still name the position"


def test_a_BINDER_failure_after_a_move_names_the_ply_it_was_ATTEMPTING(
        tmp_path, monkeypatch, openable, frozen):
    """🔴 A binder failure happens AFTER the move applies but BEFORE that move's
    `ply` record, so counting records reported the PREVIOUS ply. `note_bind` runs
    before the binder can fail."""
    seen = {}

    def binder(task, state, ply, move=None):
        seen["ply"] = ply
        if move is not None:
            raise H.AbortError(H.PHASE_BIND, f"divergence at ply {ply}")

    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    out = str(tmp_path / "r.jsonl")
    with pytest.raises(H.AbortError):
        RUN._run(out, mode=RUN.MATCH_MODE,
                 trace_path=str(tmp_path / "auto_trace_16.jsonl"), _supervise=False, _cleanup=lambda: None,
                 _binder=binder,
                 _state_factory=lambda task: TwixtState(active_size=6, to_move="red"),
                 _agent_factory=lambda t, m, e: (lambda s: (2, 2)))
    d = [json.loads(l) for l in open(out)
         if json.loads(l)["record_type"] == "void_diagnostic"][0]
    assert d["ply"] == seen["ply"], "the diagnostic named a ply the binder was not on"


# ═════════ [P1] recorder creation is under the protected boundary ════════════

def test_a_PREEXISTING_output_path_is_a_PRECONDITION_refusal(tmp_path, openable):
    """🔴 THE EXCEPTION AND THE DURABLE VERDICT MUST AGREE.

    This used to write a VOID trace and re-raise `H.HarnessError` unchanged: the
    file said the instrument failed mid-run, the exception said the run was never
    well formed. Both cannot be true. A path that already exists is a
    PRECONDITION failure -- nothing ran -- so it refuses beside the barriers,
    writes NO trace, and touches nothing.
    """
    out = tmp_path / "r.jsonl"
    out.write_text("")
    tr = str(tmp_path / "t.jsonl")
    with pytest.raises(RUN.H1Error, match="already exists") as exc:
        RUN._run(str(out), mode=RUN.MATCH_MODE, trace_path=tr, _supervise=False,
                 _cleanup=lambda: None)
    assert type(exc.value) is RUN.H1Error, "a precondition refusal is not a VOID"
    assert not os.path.exists(tr), "a refused run opened a trace"
    assert out.read_text() == ""


def test_a_PREEXISTING_TRACE_path_is_refused_too(tmp_path, openable):
    tr = tmp_path / "t.jsonl"
    tr.write_text("")
    with pytest.raises(RUN.H1Error, match="already exists"):
        RUN._run(str(tmp_path / "r.jsonl"), mode=RUN.MATCH_MODE, trace_path=str(tr),
                 _supervise=False, _cleanup=lambda: None)
    assert not os.path.exists(str(tmp_path / "r.jsonl"))


def test_a_MIDRUN_recorder_failure_is_a_VOID_and_the_TRACE_AGREES(
        tmp_path, monkeypatch, openable):
    """What remains after the preflight -- a full disk, a revoked permission --
    IS the instrument failing mid-run. The trace says VOID and now so does the
    exception type."""
    def explode(path):
        raise H.HarnessError("cannot create the results file: [Errno 28] no space")

    monkeypatch.setattr(RUN, "_PlyCountingRecorder", explode)
    out, tr = str(tmp_path / "r.jsonl"), str(tmp_path / "t.jsonl")
    with pytest.raises(RUN.H1VoidError) as exc:
        RUN._run(out, mode=RUN.MATCH_MODE, trace_path=tr, _supervise=False,
                 _cleanup=lambda: None)
    assert type(exc.value) is RUN.H1VoidError
    lines = [json.loads(l) for l in open(tr)]
    assert [l["event"] for l in lines] == ["run_start", "run_end"]
    assert lines[-1]["verdict"] == "VOID" and lines[-1]["games_completed"] == 0
    assert not os.path.exists(out)


def test_the_setup_FACTORY_is_invoked_with_the_running_clock(tmp_path, monkeypatch,
                                                             openable, frozen):
    """The factory exists so the setup can be bound to the clock that is ALREADY
    RUNNING. Asserting only that `run()` passes a factory would not notice
    `_run` never calling it -- a control proved exactly that."""
    seen = {}

    def factory(results_path, deadline):
        seen["path"] = results_path
        seen["deadline"] = deadline
        return lambda: {"agent_factory": lambda *a: None, "state_factory": lambda t: None,
                        "binder": lambda *a, **k: None, "evaluator": None,
                        "cleanup": lambda: None, "artifacts": {}}

    _drive(monkeypatch, frozen["tasks"], 5)
    out = str(tmp_path / "r.jsonl")
    RUN._run(out, mode=RUN.MATCH_MODE,
                 trace_path=str(tmp_path / "auto_trace_17.jsonl"), _supervise=False, _setup_factory=factory)
    assert seen["path"] == out
    assert seen["deadline"].started, "the setup was bound to a clock that never started"
    assert seen["deadline"].limit_s == RUN.RUN_DEADLINE_S
    hdr = json.loads(open(out).readline())
    assert hdr["run_deadline_s"] == seen["deadline"].limit_s   # one clock, one origin


# ═════════ [P1] a match needs a trace, and two DISTINCT output files ═════════

def test_MATCH_MODE_REQUIRES_a_trace_path(tmp_path, openable):
    """🔴 The card freezes a create-only VOID trace, and match mode accepted
    `trace_path=None` -- a run could reach the games with the one record a VOID
    depends on absent. Enforced in `_run`, not only in `run()`, so the private
    seam cannot skip a card requirement."""
    out = str(tmp_path / "r.jsonl")
    with pytest.raises(RUN.H1Error, match="requires a trace path") as exc:
        RUN._run(out, mode=RUN.MATCH_MODE, _supervise=False, _cleanup=lambda: None)
    assert type(exc.value) is RUN.H1Error
    assert os.listdir(tmp_path) == [], "a refused run created a file"


def test_QUALIFICATION_does_not_require_a_trace(tmp_path):
    """The requirement is the MATCH's: qualification plays no scheduled game."""
    RUN.check_output_paths(str(tmp_path / "q.jsonl"), None, require_trace=False)


def test_THE_TWO_OUTPUT_PATHS_MUST_BE_DIFFERENT_FILES(tmp_path, openable):
    """🔴 Given one path twice, the trace CREATED it and the recorder then
    refused it -- turning a naming slip into a VOID that retires all 224 seeds."""
    same = str(tmp_path / "both.jsonl")
    with pytest.raises(RUN.H1Error, match="same file") as exc:
        RUN._run(same, mode=RUN.MATCH_MODE, trace_path=same, _supervise=False,
                 _cleanup=lambda: None)
    assert type(exc.value) is RUN.H1Error
    assert os.listdir(tmp_path) == [], "a refused run created a file"


@pytest.mark.parametrize("second", ["./{name}", "{name}", "sub/../{name}"])
def test_TWO_NAMES_FOR_ONE_FILE_are_refused_after_canonicalisation(tmp_path, second):
    """"r.jsonl", "./r.jsonl" and a path through a symlink are one file under
    three names, so the comparison canonicalises before deciding."""
    (tmp_path / "sub").mkdir()
    results = str(tmp_path / "r.jsonl")
    trace = str(tmp_path / second.format(name="r.jsonl"))
    with pytest.raises(RUN.H1Error, match="same file"):
        RUN.check_output_paths(results, trace, require_trace=True)


def test_a_SYMLINKED_trace_path_is_refused_too(tmp_path):
    real = tmp_path / "real.jsonl"
    link = tmp_path / "link.jsonl"
    real.write_text("")
    link.symlink_to(real)
    real.unlink()                       # a DANGLING link: neither path "exists"
    assert not os.path.exists(str(link))
    with pytest.raises(RUN.H1Error, match="same file"):
        RUN.check_output_paths(str(real), str(link), require_trace=True)


def test_the_distinctness_check_runs_BEFORE_either_file_is_opened(tmp_path, openable):
    """Order again: the refusal must precede creation, not clean up after it."""
    same = str(tmp_path / "x.jsonl")
    with pytest.raises(RUN.H1Error):
        RUN._run(same, mode=RUN.MATCH_MODE, trace_path=same, _supervise=False,
                 _cleanup=lambda: None)
    assert not os.path.exists(same)


def _dangling(path):
    """A symlink whose target does not exist: `exists` says False, `lexists` says
    True, and create-exclusive open refuses it."""
    os.symlink("/nonexistent/target", path)
    assert not os.path.exists(path) and os.path.lexists(path)
    return path


def test_a_DANGLING_RESULTS_link_is_refused_by_the_precondition(tmp_path, openable):
    """🔴 `os.path.exists` FOLLOWS the link, so a dangling one read as absent --
    the precheck passed, the trace opened, and the recorder produced a VOID.
    A knowable path condition must not cost the seed block.

    The earlier symlink test cannot prove this: both its names resolve to one
    target, so the SAME-FILE guard fires first.
    """
    results = _dangling(str(tmp_path / "r.jsonl"))
    trace = str(tmp_path / "t.jsonl")                 # a genuinely different file
    assert RUN._canonical(results) != RUN._canonical(trace)   # same-file guard idle
    with pytest.raises(RUN.H1Error, match="already exists") as exc:
        RUN._run(results, mode=RUN.MATCH_MODE, trace_path=trace, _supervise=False,
                 _cleanup=lambda: None)
    assert type(exc.value) is RUN.H1Error, "a knowable path condition became a VOID"
    assert "dangling symlink" in str(exc.value)
    assert not os.path.exists(trace), "a refused run opened the trace"
    assert sorted(os.listdir(tmp_path)) == ["r.jsonl"]        # only the link we made


def test_a_DANGLING_TRACE_link_is_refused_by_the_precondition(tmp_path, openable):
    """The same check, on the other output name -- neither is covered by the other."""
    results = str(tmp_path / "r.jsonl")
    trace = _dangling(str(tmp_path / "t.jsonl"))
    assert RUN._canonical(results) != RUN._canonical(trace)
    with pytest.raises(RUN.H1Error, match="already exists") as exc:
        RUN._run(results, mode=RUN.MATCH_MODE, trace_path=trace, _supervise=False,
                 _cleanup=lambda: None)
    assert type(exc.value) is RUN.H1Error
    assert not os.path.exists(results)
    assert sorted(os.listdir(tmp_path)) == ["t.jsonl"]


def test_the_precondition_agrees_with_what_CREATE_EXCLUSIVE_open_would_do(tmp_path):
    """The check exists to predict the open. Asserted against the real syscall."""
    for name in ("plain", "dangling"):
        path = str(tmp_path / f"{name}.jsonl")
        if name == "dangling":
            _dangling(path)
        else:
            open(path, "x").close()
        refused_by_check = False
        try:
            RUN.check_output_paths(path, str(tmp_path / "other.jsonl"),
                                   require_trace=True)
        except RUN.H1Error:
            refused_by_check = True
        refused_by_open = False
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644))
        except FileExistsError:
            refused_by_open = True
        assert refused_by_check == refused_by_open is True, name


def test_the_public_docstring_DESCRIBES_THE_TESTED_BEHAVIOUR():
    """An operator-facing description that contradicts the code is a defect in
    the same family as a guard that cannot fire: it documents a protection that
    is not there, and the reader has no way to tell."""
    doc = RUN.run.__doc__
    # The withdrawn claim may appear ONLY as a QUOTED correction -- the same
    # treatment the low-ply card's "boundary lies between" wording gets. Strip
    # the quotation, then require the live text not to make the claim.
    quoted = 'this said "MATCH MODE IS NOT SELECTABLE HERE"'
    assert quoted in doc, "the correction should record what it corrects"
    live = doc.replace(quoted, "")
    assert "NOT SELECTABLE" not in live.upper(), "the live text still denies it"
    assert "SELECTABLE, DELIBERATELY" in doc
    assert "H1_EXECUTION_AUTHORIZED" in doc, "the docstring must name what refuses"
    # and the behaviour it describes is the behaviour that is tested
    assert RUN.MATCH_MODE in RUN.MODES
    with pytest.raises(RUN.H1Error, match="H1_EXECUTION_AUTHORIZED is False"):
        RUN.run("/nonexistent/r.jsonl", mode=RUN.MATCH_MODE,
                trace_path="/nonexistent/t.jsonl")
