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
from scripts.GPU.alphazero import void_trace as VT


# ══════════════════════════ the gate, as published ═══════════════════════════

def test_the_gate_is_false_as_published():
    assert RUN.H1_EXECUTION_AUTHORIZED is False


def test_the_public_entry_point_cannot_select_match_mode():
    with pytest.raises(RUN.H1Error, match="UNAUTHORIZED"):
        RUN.run("/nonexistent/results.jsonl", mode=RUN.MATCH_MODE)


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


def test_THE_GATE_fires_before_plan_load_setup_recorder_or_play(spy):
    """Barrier 1, and the ORDER is the guarantee: a refusal that has already
    created a results file has not refused."""
    with pytest.raises(RUN.H1Error, match="NOT AUTHORIZED"):
        RUN._run(spy["out"], mode=RUN.MATCH_MODE, _setup=_setup_that_must_not_run(spy))
    assert spy["plan_load"] == 0, "the plan was read before the gate"
    assert spy["setup"] == 0, "setup ran before the gate"
    assert spy["recorder"] == 0, "the recorder was constructed before the gate"
    assert spy["play"] == 0, "a game was played before the gate"
    assert not os.path.exists(spy["out"]), "a results file exists after a refusal"


def test_THE_REGISTRATION_BARRIER_fires_before_plan_load_setup_recorder_or_play(
        spy, monkeypatch):
    """Barrier 2, with the gate OPEN, so it is reached alone. Opening one barrier
    must not open the other."""
    monkeypatch.setattr(RUN, "H1_EXECUTION_AUTHORIZED", True)
    with pytest.raises(RUN.H1Error, match="not registered"):
        RUN._run(spy["out"], mode=RUN.MATCH_MODE, _setup=_setup_that_must_not_run(spy))
    assert spy["plan_load"] == 0 and spy["setup"] == 0
    assert spy["recorder"] == 0 and spy["play"] == 0
    assert not os.path.exists(spy["out"])


def test_REGISTERING_THE_BLOCK_DOES_NOT_OPEN_THE_GATE(spy, monkeypatch):
    """The two are independent. Registration alone still refuses."""
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS",
                        REF.ACCOUNTED_SEED_INTERVALS + (RULES.H1_SEED_BLOCK,))
    RUN.check_seed_registration()                     # barrier 2 now satisfied
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
                        REF.ACCOUNTED_SEED_INTERVALS + ((lo, lo + 1), (hi - 1, hi)))
    assert REF.seed_is_accounted(lo) and REF.seed_is_accounted(hi - 1)
    assert not REF.seed_is_accounted(lo + 1)          # the hole
    with pytest.raises(RUN.H1Error, match="not registered"):
        RUN.check_seed_registration()


def test_the_registration_check_READS_the_registry_and_never_writes_it(monkeypatch):
    before = tuple(REF.ACCOUNTED_SEED_INTERVALS)
    with pytest.raises(RUN.H1Error):
        RUN.check_seed_registration()
    assert tuple(REF.ACCOUNTED_SEED_INTERVALS) == before


# ═══════════════ the frozen design: 224, no early stop, timeouts ═════════════

@pytest.fixture(scope="module")
def frozen():
    return PLAN.load_h1_plan()


@pytest.fixture
def openable(monkeypatch):
    """Gate open and block registered, INSIDE the test only. Neither the real
    gate constant nor the real registry is ever edited."""
    monkeypatch.setattr(RUN, "H1_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS",
                        REF.ACCOUNTED_SEED_INTERVALS + (RULES.H1_SEED_BLOCK,))


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
    assert RUN._run(out, mode=RUN.MATCH_MODE, _supervise=False,
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
    RUN._run(out, mode=RUN.MATCH_MODE, _supervise=False, _cleanup=lambda: None)
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


def test_the_results_file_is_CREATE_ONLY(tmp_path, monkeypatch, openable, frozen):
    out = tmp_path / "r.jsonl"
    out.write_text("")
    _drive(monkeypatch, frozen["tasks"], 10)
    with pytest.raises(Exception):
        RUN._run(str(out), mode=RUN.MATCH_MODE, _supervise=False, _cleanup=lambda: None)
    assert out.read_text() == ""          # untouched


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


def test_the_deadline_translates_D1s_error_into_H1s_own(monkeypatch):
    from scripts.GPU.alphazero import d1_probe as D1
    dl = D1.Deadline(1.0, clock=lambda: 0.0)
    dl.start()
    monkeypatch.setattr(dl, "elapsed", lambda: 999.0)
    with pytest.raises(RUN.H1VoidError):
        RUN._check_deadline(dl, "here")


# ══════════════════ match mode loads ONLY the pinned v3 plan ═════════════════

def test_match_mode_REFUSES_a_supplied_plan_path(tmp_path, openable):
    with pytest.raises(RUN.H1Error, match="pinned v3 plan only"):
        RUN._run(str(tmp_path / "r.jsonl"), mode=RUN.MATCH_MODE,
                 _plan_path=str(tmp_path / "other.json"))


def test_match_mode_loads_the_pinned_plan_without_being_told_where(
        tmp_path, monkeypatch, openable, frozen):
    seen = {}
    real = PLAN.load_h1_plan
    monkeypatch.setattr(RUN.PLAN, "load_h1_plan",
                        lambda p=PLAN.H1_PLAN_REL: (seen.setdefault("p", p), real(p))[1])
    _drive(monkeypatch, frozen["tasks"], 10)
    RUN._run(str(tmp_path / "r.jsonl"), mode=RUN.MATCH_MODE, _supervise=False,
             _cleanup=lambda: None)
    assert seen["p"] == PLAN.H1_PLAN_REL


def test_a_schedule_that_is_not_the_frozen_224_is_refused(tmp_path, openable, frozen):
    short = list(frozen["tasks"])[:-1]
    with pytest.raises(RUN.H1Error, match="expected exactly"):
        RUN._run(str(tmp_path / "r.jsonl"), mode=RUN.MATCH_MODE, _tasks=short)


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
        RUN._run(str(tmp_path / "r.jsonl"), mode=RUN.MATCH_MODE, _tasks=tasks)


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
        RUN._run(str(tmp_path / "r.jsonl"), mode=RUN.MATCH_MODE, _tasks=tasks)


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
    assert RUN._run(out, mode=RUN.MATCH_MODE, _supervise=False,
                    _cleanup=lambda: None) == H.EXIT_OK
    recs = [json.loads(l) for l in open(out)]
    outcomes = [r for r in recs if r["record_type"] == "match_outcome"]
    assert len(outcomes) == 1
    assert outcomes[0]["outcome"] == RUN.CAP_SATURATED_NO_RATE
    assert not any(r["record_type"] == "viability_report" for r in recs)
    assert all("verdict" not in r for r in recs)
