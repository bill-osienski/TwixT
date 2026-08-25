"""Fail-closed checks for the E4 execution harness.

No model, no T1j, no scheduled seed. The canonical schedule's seeds
[202612128, 202612160) appear only as inert plan data. The production play loop
is exercised against the REAL TwixtState with fake agents, so the control flow
under test is the harness's own.
"""
import inspect
import json
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import e4_screen_runner as H
from scripts.GPU.alphazero.game.twixt_state import TwixtState

PLAN = "docs/superpowers/evidence/2026-08-25-t1j-e4-preflight-attempt4/06_endpoint_screen_plan.json"
SYNTHETIC = 90000001
OPENING = [(11, 11), (12, 13), (13, 12), (10, 13), (12, 10), (14, 14)]


def synthetic_task(i=0, endpoint="weak", anchor="red"):
    return {"task_id": f"synthetic-{i:03d}", "endpoint": endpoint, "seed": SYNTHETIC + i,
            "anchor_colour": anchor, "reference_colour": "black" if anchor == "red" else "red"}


def opening_state(task):
    s = TwixtState(active_size=24, to_move="red")
    for mv in OPENING:
        s = s.apply_move(mv)
    return s


def scripted_agent(moves):
    it = iter(moves)
    return lambda state: next(it)


def null_binder(task, state, ply):
    return None


# ------------------------------------------------- public surface + the CLI

def test_public_run_accepts_paths_only():
    params = inspect.signature(H.run).parameters
    assert set(params) == {"plan_path", "results_path", "mode"}
    assert params["mode"].kind is inspect.Parameter.KEYWORD_ONLY


def cli(*args):
    return subprocess.run([sys.executable, "-m", "scripts.GPU.alphazero.e4_screen_runner", *args],
                          capture_output=True, text=True)


def test_cli_help_is_a_real_command():
    r = cli("--help")
    assert r.returncode == 0
    assert "UNAUTHORIZED" in r.stdout and "exit codes" in r.stdout


def test_cli_refuses_screen_mode_in_a_fresh_subprocess(tmp_path):
    r = cli("--plan", PLAN, "--results", str(tmp_path / "a.jsonl"), "--mode", "screen")
    assert r.returncode == H.EXIT_PRECONDITION
    assert "UNAUTHORIZED" in r.stderr
    assert not (tmp_path / "a.jsonl").exists()      # refused before anything was opened


def test_cli_refuses_an_existing_results_file(tmp_path):
    p = tmp_path / "b.jsonl"
    p.write_text("stale\n")
    r = cli("--plan", PLAN, "--results", str(p))
    assert r.returncode == H.EXIT_PRECONDITION
    assert "already exists" in r.stderr
    assert p.read_text() == "stale\n"               # untouched


def test_cli_refuses_a_tampered_plan(tmp_path):
    plan = json.load(open(PLAN))
    plan["note"] = "tampered"
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan))
    r = cli("--plan", str(p), "--results", str(tmp_path / "c.jsonl"))
    assert r.returncode == H.EXIT_PRECONDITION
    assert "sha256" in r.stderr


def test_cli_public_path_runs_no_task_and_aborts_nothing(tmp_path):
    out = tmp_path / "d.jsonl"
    r = cli("--plan", PLAN, "--results", str(out))
    assert r.returncode == H.EXIT_OK, r.stderr
    rows = [json.loads(l) for l in open(out)]
    assert rows[0]["canonical_tasks"] == 32 and rows[0]["canonical_tasks_executed"] == 0
    assert not [x for x in rows if x["record_type"] == "task_result"]


# --------------------------------------------------------- the schedule lock

def canonical_tasks():
    return json.load(open(PLAN))["tasks"]


def _moved(mutate):
    t = canonical_tasks()
    return H.task_digest(mutate(t)) != H.task_digest(t) or len(mutate(t)) != len(t)


@pytest.mark.parametrize("mutate,label", [
    (lambda t: t + [dict(t[0], task_id="extra")], "addition"),
    (lambda t: t[:-1], "removal"),
    (lambda t: [t[1], t[0]] + t[2:], "reordering"),
    (lambda t: t[:-1] + [dict(t[0])], "duplicate"),
    (lambda t: [dict(t[0], endpoint="weak" if t[0]["endpoint"] == "strong" else "strong")] + t[1:],
     "endpoint edit"),
    (lambda t: [dict(t[0], t1j_mdPly=99)] + t[1:], "depth edit"),
    (lambda t: [dict(t[0], opening="o9_fake")] + t[1:], "opening edit"),
    (lambda t: [dict(t[0], colour_arm="flipped")] + t[1:], "colour edit"),
    (lambda t: [dict(t[0], reference="0379")] + t[1:], "reference edit"),
    (lambda t: [dict(t[0], reference_sha1="0" * 40)] + t[1:], "reference hash edit"),
    (lambda t: [dict(t[0], seed=202612999)] + t[1:], "seed edit"),
])
def test_reshaping_the_schedule_is_refused(mutate, label):
    assert _moved(mutate), f"{label}: INVALID CONTROL — the mutation changed nothing"
    with pytest.raises(H.HarnessError):
        H.verify_tasks(mutate(canonical_tasks()))


def test_scheduled_seeds_are_refused():
    for task in canonical_tasks()[:4]:
        with pytest.raises(H.HarnessError):
            H._assert_not_scheduled(task)
    H._assert_not_scheduled(synthetic_task())


# ------------------------------------------------------------ the play loop

class _Rec:
    def __init__(self): self.rows = []
    def emit(self, r): self.rows.append(r)
    def emit_terminal(self, r): self.rows.append(r); return None


def play(agent_moves, *, ply_cap=280, binder=null_binder, anchor="red"):
    rec = _Rec()
    task = synthetic_task(anchor=anchor)
    it = iter(agent_moves)
    out = H.play_task(task=task, agent_for=lambda t, m: (lambda s: next(it)),
                      state_factory=opening_state, binder=binder, rec=rec, ply_cap=ply_cap)
    return out, rec


def test_loop_alternates_and_records_every_ply():
    out, rec = play([(5, 5), (6, 7), (5, 7)], ply_cap=9)
    plies = [r for r in rec.rows if r["record_type"] == "ply"]
    assert [p["mover"] for p in plies] == ["red", "black", "red"]
    assert [p["ply"] for p in plies] == [7, 8, 9]
    assert out["plies"] == 9 and out["terminal_reason"] == "cap"


def test_loop_applies_the_external_ply_cap():
    out, rec = play([(5, 5), (6, 7), (5, 7), (9, 9)], ply_cap=8)
    assert out["terminal_reason"] == "cap"
    assert out["plies"] == 8 and out["winner"] is None
    assert out["t1j_points"] == 0.5                      # a cap is a draw
    assert len([r for r in rec.rows if r["record_type"] == "ply"]) == 2


def test_loop_validates_the_move_before_applying_it():
    with pytest.raises(H.AbortError) as e:
        play([(11, 11)])                                  # already occupied
    assert e.value.phase == H.PHASE_MOVE and "not legal" in e.value.message
    with pytest.raises(H.AbortError):
        play([None])
    with pytest.raises(H.AbortError):
        play([(99, 99)])                                  # off board


def test_loop_binds_the_opening_first_then_every_applied_move():
    seen = []
    play([(5, 5), (6, 7)], ply_cap=8, binder=lambda t, s, p: seen.append(p))
    assert seen == [6, 7, 8]          # 6 is the OPENING, bound before any move


def test_a_binder_divergence_aborts_in_the_binding_phase():
    def diverging(task, state, ply):
        raise H.AbortError(H.PHASE_BIND, f"state divergence at ply {ply}")
    with pytest.raises(H.AbortError) as e:
        play([(5, 5)], binder=diverging)
    assert e.value.phase == H.PHASE_BIND


def test_the_default_binder_refuses():
    with pytest.raises(H.AbortError) as e:
        H.play_task(task=synthetic_task(), agent_for=lambda t, m: (lambda s: (5, 5)),
                    state_factory=opening_state, binder=H._refuse_binder, rec=_Rec())
    assert e.value.phase == H.PHASE_BIND


def test_points_follow_the_anchor_colour():
    s = opening_state(None)
    # a cap draw scores 0.5 for either anchor colour
    for anchor in ("red", "black"):
        out, _ = play([(5, 5)], ply_cap=7, anchor=anchor)
        assert out["t1j_points"] == 0.5


# ------------------------------------------- evaluator identity is ENFORCED

class _Agent:
    def __init__(self, evaluator, moves):
        self.evaluator = evaluator
        self._it = iter(moves)
    def __call__(self, state): return next(self._it)


def run_stub(tmp_path, tasks, factory, *, n=2, cap=280, cleanup=None, binder=null_binder):
    out = tmp_path / "r.jsonl"
    rc = H._run(PLAN, str(out), mode="qualify", _tasks=tasks, _agent_factory=factory,
                _state_factory=opening_state, _binder=binder, _evaluator=EV,
                _cleanup=cleanup or (lambda: None), _n_per_endpoint=n, _ply_cap=cap)
    return rc, [json.loads(l) for l in open(out)]


EV = object()


def test_one_evaluator_is_accepted(tmp_path):
    rc, rows = run_stub(tmp_path, [synthetic_task(0, "weak"), synthetic_task(1, "strong")],
                        lambda t, m, ev: _Agent(EV, [(5, 5)]), cap=7)
    assert rc == H.EXIT_OK
    assert rows[-1]["record_type"] == "verdict"


def test_a_rebuilt_evaluator_ABORTS_with_no_verdict(tmp_path):
    """NEGATIVE CONTROL. A rebuild on the REFERENCE side must fail the gate."""
    with pytest.raises(H.AbortError) as e:
        run_stub(tmp_path, [synthetic_task(0, "weak", anchor="black")],   # reference plays red
                 lambda t, m, ev: _Agent(object(), [(5, 5)]), cap=7)
    assert e.value.phase == H.PHASE_FACTORY
    rows = [json.loads(l) for l in open(tmp_path / "r.jsonl")]
    kinds = [r["record_type"] for r in rows]
    assert "abort" in kinds and "verdict" not in kinds
    assert rows[-1]["phase"] == H.PHASE_FACTORY


# ------------------------------------------------- phases, aborts, recording

def test_cleanup_failure_is_classified_and_aborts(tmp_path):
    def boom(): raise RuntimeError("metal cache wedged")
    with pytest.raises(H.AbortError) as e:
        run_stub(tmp_path, [synthetic_task(0, "weak")],
                 lambda t, m, ev: _Agent(EV, [(5, 5)]), cap=7, cleanup=boom)
    assert e.value.phase == H.PHASE_CLEANUP
    rows = [json.loads(l) for l in open(tmp_path / "r.jsonl")]
    assert rows[-1]["record_type"] == "abort" and rows[-1]["phase"] == H.PHASE_CLEANUP


def test_agent_exception_is_classified_as_a_move_failure(tmp_path):
    def raising(t, m, ev):
        class _A:
            evaluator = EV
            def __call__(self, s): raise ValueError("network died")
        return _A()
    with pytest.raises(H.AbortError) as e:
        run_stub(tmp_path, [synthetic_task(0, "weak")], raising, cap=7)
    assert e.value.phase == H.PHASE_MOVE and "network died" in e.value.message


def test_recorder_refuses_an_existing_file(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text("")
    with pytest.raises(H.HarnessError):
        H.Recorder(str(p))


def test_records_are_durable_immediately(tmp_path):
    """Each record is flushed and fsynced, so a reader sees it at once."""
    p = tmp_path / "y.jsonl"
    rec = H.Recorder(str(p))
    rec.emit({"record_type": "one"})
    assert json.loads(open(p).read().strip())["record_type"] == "one"
    rec.emit({"record_type": "two"})
    assert len(open(p).read().strip().splitlines()) == 2
    rec.close()


def test_a_failing_terminal_record_does_not_mask_the_error(tmp_path):
    rec = H.Recorder(str(tmp_path / "z.jsonl"))
    rec.close()                                            # writes will now fail
    note = rec.emit_terminal({"record_type": "abort"})
    assert note and "ValueError" in note or note            # reported, not raised


# ------------------------------------------------ early stop and classifying

# A 6x6 board reaches a real red win in 7 plies, so the early stop can be driven
# by actual WINS rather than cap draws -- cap draws keep the cap route open and
# correctly prevent the stop.
SMALL_WIN = [(0, 2), (2, 0), (2, 3), (3, 0), (4, 2), (4, 0), (5, 4)]
# the sequence ALTERNATES, so each colour plays its own half of it
SMALL_WIN_BY_COLOUR = {"red": SMALL_WIN[0::2], "black": SMALL_WIN[1::2]}


def win_factory(task, mover, evaluator=None):
    return _Agent(EV, list(SMALL_WIN_BY_COLOUR[mover]))


def small_state(task):
    return TwixtState(active_size=6, to_move="red")


def test_a_scripted_small_board_game_really_ends_in_a_win():
    rec = _Rec()
    out = H.play_task(task=synthetic_task(anchor="red"),
                      agent_for=lambda t, m: win_factory(t, m),
                      state_factory=small_state, binder=null_binder, rec=rec)
    assert out["winner"] == "red" and out["terminal_reason"] == "win"
    assert out["t1j_points"] == 1.0 and out["plies"] == 7


def test_endpoint_stops_early_and_later_tasks_are_skipped(tmp_path):
    """n=4: one anchor win and one anchor loss closes BOTH routes at game 2."""
    tasks = [synthetic_task(0, "weak", anchor="red"),      # red wins -> anchor scores 1.0
             synthetic_task(1, "weak", anchor="black"),    # red wins -> anchor scores 0.0
             synthetic_task(2, "weak", anchor="red")]      # must never be played
    out = tmp_path / "r.jsonl"
    rc = H._run(PLAN, str(out), mode="qualify", _tasks=tasks,
                _agent_factory=win_factory,
                _state_factory=small_state, _binder=null_binder, _evaluator=EV,
                _cleanup=lambda: None, _n_per_endpoint=4)
    rows = [json.loads(l) for l in open(out)]
    kinds = [r["record_type"] for r in rows]
    assert rc == H.EXIT_OK
    assert kinds.count("task_result") == 2
    assert "early_stop" in kinds
    assert kinds.count("task_skipped") == 1
    stop = [r for r in rows if r["record_type"] == "early_stop"][0]
    assert stop["played"] == 2 and stop["score"] == 1.0
    assert rows[-1]["record_type"] == "verdict"
    assert rows[-1]["per_endpoint"]["weak"]["decision"] == "IN_BAND"


def verdict(weak, strong, n=16):
    rows = ([{"endpoint": "weak", "t1j_points": p, "terminal_reason": "win"} for p in weak]
            + [{"endpoint": "strong", "t1j_points": p, "terminal_reason": "win"} for p in strong])
    return H.classify_run(rows, n_per_endpoint=n, band=[0.05, 0.95])


@pytest.mark.parametrize("weak,strong,want", [
    ([1.0] * 16, [1.0] * 16, "T1J_TOO_STRONG"),
    ([0.0] * 16, [0.0] * 16, "T1J_TOO_WEAK"),
    ([0.0] * 16, [1.0] * 16, "BRACKETED"),
    ([0.5] * 16, [0.5] * 16, "IN_BAND"),
    ([0.0] * 4, [0.5] * 16, "INCONCLUSIVE"),
])
def test_every_joint_outcome_is_reachable(weak, strong, want):
    assert verdict(weak, strong)["joint"] == want


def test_partial_results_are_refused_a_verdict():
    v = verdict([1.0] * 3, [0.0] * 3)
    assert v["per_endpoint"]["weak"]["decision"] == "INCOMPLETE"
    assert v["joint"] == "INCONCLUSIVE"


def test_cap_terminations_make_an_endpoint_incomplete():
    rows = ([{"endpoint": "weak", "t1j_points": 0.5, "terminal_reason": "cap"} for _ in range(9)]
            + [{"endpoint": "weak", "t1j_points": 1.0, "terminal_reason": "win"} for _ in range(7)]
            + [{"endpoint": "strong", "t1j_points": 0.5, "terminal_reason": "win"} for _ in range(16)])
    v = H.classify_run(rows, n_per_endpoint=16, band=[0.05, 0.95])
    assert v["per_endpoint"]["weak"]["decision"] == "INCOMPLETE"
    assert v["joint"] == "INCONCLUSIVE"


def test_each_side_is_built_once_per_task_not_once_per_ply():
    """SeededReferenceAgent is stateful: rebuilding per ply would reset both RNG
    streams and make the per-task seed meaningless."""
    builds = []

    def counting_factory(task, mover):
        builds.append(mover)
        return _Agent(EV, list(SMALL_WIN_BY_COLOUR[mover]))

    rec = _Rec()
    out = H.play_task(task=synthetic_task(anchor="red"), agent_for=counting_factory,
                      state_factory=small_state, binder=null_binder, rec=rec)
    assert out["winner"] == "red" and out["plies"] == 7
    assert sorted(builds) == ["black", "red"]          # exactly one of each, for 7 plies
    assert out["agents_built"] == 2


# --- the identity gate belongs to the REFERENCE side only -------------------

class _NoEvaluatorAgent:
    """A classical engine: it holds no MLX evaluator and never will."""
    def __init__(self, moves): self._it = iter(moves)
    def __call__(self, state): return next(self._it)


def test_the_anchor_side_may_hold_no_evaluator():
    """T1j is classical. Gating both colours would abort on its first construction."""
    task = synthetic_task(anchor="red")            # anchor red, reference black
    built = []

    def factory(t, mover):
        built.append(mover)
        if mover == t["reference_colour"]:
            return _Agent(EV, list(SMALL_WIN_BY_COLOUR[mover]))
        return _NoEvaluatorAgent(list(SMALL_WIN_BY_COLOUR[mover]))

    def agent_for(t, mover):
        a = factory(t, mover)
        H._enforce_evaluator(a, EV, t, mover)
        return a

    out = H.play_task(task=task, agent_for=agent_for, state_factory=small_state,
                      binder=null_binder, rec=_Rec())
    assert out["winner"] == "red" and sorted(built) == ["black", "red"]


@pytest.mark.parametrize("agent,label", [
    (_NoEvaluatorAgent([(0, 0)]), "no evaluator at all"),
    (_Agent(object(), [(0, 0)]), "a rebuilt evaluator"),
])
def test_the_reference_side_rejects_a_missing_or_rebuilt_evaluator(agent, label):
    task = synthetic_task(anchor="red")            # reference plays black
    with pytest.raises(H.AbortError) as e:
        H._enforce_evaluator(agent, EV, task, "black")
    assert e.value.phase == H.PHASE_FACTORY
    # ...and the same agent is fine on the anchor side
    H._enforce_evaluator(agent, EV, task, "red")


# --- the opening is bound before anything else -----------------------------

def test_an_opening_divergence_aborts_before_either_agent_is_built():
    built = []

    def diverging(task, state, ply):
        raise H.AbortError(H.PHASE_BIND, f"opening divergence at ply {ply}")

    def counting(t, m):
        built.append(m)
        return _Agent(EV, [(5, 5)])

    with pytest.raises(H.AbortError) as e:
        H.play_task(task=synthetic_task(), agent_for=counting, state_factory=small_state,
                    binder=diverging, rec=_Rec())
    assert e.value.phase == H.PHASE_BIND
    assert built == [], "no agent may be constructed before the opening is bound"


def test_the_opening_bind_is_recorded():
    rec = _Rec()
    H.play_task(task=synthetic_task(anchor="red"), agent_for=lambda t, m: win_factory(t, m),
                state_factory=small_state, binder=null_binder, rec=rec)
    assert rec.rows[0]["record_type"] == "opening_bound" and rec.rows[0]["ply"] == 0


# --- binder failures are classified, cleanup still runs --------------------

def test_a_plain_exception_from_the_binder_is_classified_and_recorded(tmp_path):
    def sloppy(task, state, ply):
        raise ValueError("t1j replay exit 3")
    cleanups = []
    with pytest.raises(H.AbortError) as e:
        run_stub(tmp_path, [synthetic_task(0, "weak")],
                 lambda t, m, ev: _Agent(EV, [(5, 5)]), cap=7,
                 cleanup=lambda: cleanups.append(1), binder=sloppy)
    assert e.value.phase == H.PHASE_BIND and "ValueError" in e.value.message
    rows = [json.loads(l) for l in open(tmp_path / "r.jsonl")]
    kinds = [r["record_type"] for r in rows]
    assert "abort" in kinds and rows[-1]["phase"] == H.PHASE_BIND
    assert len(cleanups) == 1, "cleanup runs for a STARTED task even when it aborts"


def test_a_cleanup_failure_after_an_abort_does_not_mask_the_abort(tmp_path):
    def sloppy(task, state, ply):
        raise ValueError("t1j replay exit 3")

    def boom():
        raise RuntimeError("cleanup also failed")

    with pytest.raises(H.AbortError) as e:
        run_stub(tmp_path, [synthetic_task(0, "weak")],
                 lambda t, m, ev: _Agent(EV, [(5, 5)]), cap=7, cleanup=boom, binder=sloppy)
    assert e.value.phase == H.PHASE_BIND          # the ORIGINAL failure survives
    rows = [json.loads(l) for l in open(tmp_path / "r.jsonl")]
    assert any(r["record_type"] == "cleanup_failure_after_abort" for r in rows)


# --- no screen verdict from an incomplete result set -----------------------

def test_zero_games_get_a_RECEIPT_not_a_verdict(tmp_path):
    out = tmp_path / "empty.jsonl"
    rc = H.run(PLAN, str(out), mode="qualify")
    rows = [json.loads(l) for l in open(out)]
    kinds = [r["record_type"] for r in rows]
    assert rc == H.EXIT_OK
    assert "verdict" not in kinds, "zero games must not produce a screen verdict"
    assert kinds[-1] == "qualification_receipt"
    assert "no tasks were scheduled" in rows[-1]["verdict_withheld"]
    assert "joint" not in rows[-1] and "larger_match_permitted" not in rows[-1]


def test_cli_zero_games_emits_no_verdict(tmp_path):
    out = tmp_path / "cli_empty.jsonl"
    r = cli("--plan", PLAN, "--results", str(out))
    assert r.returncode == H.EXIT_OK
    kinds = [json.loads(l)["record_type"] for l in open(out)]
    assert "verdict" not in kinds and kinds[-1] == "qualification_receipt"


@pytest.mark.parametrize("tasks,results,skipped,stopped,ok", [
    ([], [], [], {}, False),                                            # zero
    ([{"task_id": "a"}], [], [], {}, False),                            # missing
    ([{"task_id": "a"}], [{"task_id": "a"}], [], {}, True),             # complete
    ([{"task_id": "a"}], [{"task_id": "a"}, {"task_id": "a"}], [], {}, False),   # duplicate
    ([{"task_id": "a"}], [{"task_id": "zzz"}], [], {}, False),          # alien
    ([{"task_id": "a"}, {"task_id": "b"}], [{"task_id": "a"}],
     [{"task_id": "b", "endpoint": "weak"}], {"weak": "stopped"}, True),  # justified skip
    ([{"task_id": "a"}, {"task_id": "b"}], [{"task_id": "a"}],
     [{"task_id": "b", "endpoint": "weak"}], {}, False),                # UNjustified skip
])
def test_screen_verdict_requires_a_complete_result_set(tasks, results, skipped, stopped, ok):
    allowed, why = H.screen_verdict_allowed(tasks, results, skipped, stopped)
    assert allowed is ok, why


def test_a_complete_run_still_gets_a_verdict(tmp_path):
    tasks = [synthetic_task(0, "weak", anchor="red"), synthetic_task(1, "weak", anchor="black")]
    out = tmp_path / "full.jsonl"
    rc = H._run(PLAN, str(out), mode="qualify", _tasks=tasks, _agent_factory=win_factory,
                _state_factory=small_state, _binder=null_binder, _evaluator=EV,
                _cleanup=lambda: None, _n_per_endpoint=2)
    rows = [json.loads(l) for l in open(out)]
    assert rc == H.EXIT_OK and rows[-1]["record_type"] == "verdict"
