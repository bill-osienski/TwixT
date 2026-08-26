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

from scripts.GPU.alphazero import e4_screen_reference as REF
from scripts.GPU.alphazero import e4_screen_runner as H
from scripts.GPU.alphazero.game.twixt_state import TwixtState

PLAN = "docs/superpowers/evidence/2026-08-25-t1j-e4-preflight-attempt4/06_endpoint_screen_plan.json"
SYNTHETIC = 90009001                    # TEST_ONLY_SEED_INTERVALS: never schedulable
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


def null_binder(task, state, ply, move=None):
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
    play([(5, 5), (6, 7)], ply_cap=8, binder=lambda t, s, p, m=None: seen.append(p))
    assert seen == [6, 7, 8]          # 6 is the OPENING, bound before any move


def test_a_binder_divergence_aborts_in_the_binding_phase():
    def diverging(task, state, ply, move=None):
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
    # a synthetic run is not the canonical schedule, so it earns a RECEIPT
    assert rows[-1]["record_type"] == "qualification_receipt"
    assert [r["record_type"] for r in rows].count("task_result") == 2


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
    # the early stop fired and was recorded; the VERDICT is still withheld,
    # because synthetic identities are not the canonical schedule
    assert rows[-1]["record_type"] == "qualification_receipt"


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

    def diverging(task, state, ply, move=None):
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
    def sloppy(task, state, ply, move=None):
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
    def sloppy(task, state, ply, move=None):
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


CANON = canonical_tasks()


def res(t):
    return {"task_id": t["task_id"], "endpoint": t["endpoint"], "t1j_points": 0.5,
            "terminal_reason": "win"}


def test_a_full_canonical_result_set_gets_a_verdict():
    allowed, why = H.screen_verdict_allowed(CANON, CANON, [res(t) for t in CANON], [], {})
    assert allowed, why


def test_a_canonical_run_with_justified_skips_gets_a_verdict():
    played = [res(t) for t in CANON if t["endpoint"] == "strong"]
    skipped = [{"task_id": t["task_id"], "endpoint": "weak"}
               for t in CANON if t["endpoint"] == "weak"]
    allowed, why = H.screen_verdict_allowed(CANON, CANON, played, skipped, {"weak": "stopped"})
    assert allowed, why


@pytest.mark.parametrize("label,tasks,results,skipped,stopped", [
    ("zero tasks", [], [], [], {}),
    ("ONE endpoint only", [t for t in CANON if t["endpoint"] == "weak"],
     [res(t) for t in CANON if t["endpoint"] == "weak"], [], {}),
    ("a subset of the schedule", CANON[:4], [res(t) for t in CANON[:4]], [], {}),
    ("a single synthetic task", [{"task_id": "synthetic-000", "endpoint": "weak"}],
     [{"task_id": "synthetic-000", "endpoint": "weak", "t1j_points": 1.0,
       "terminal_reason": "win"}], [], {}),
    ("a missing result", CANON, [res(t) for t in CANON[:-1]], [], {}),
    ("a duplicated result", CANON, [res(t) for t in CANON] + [res(CANON[0])], [], {}),
    ("an alien identity", CANON,
     [res(t) for t in CANON[:-1]] + [{"task_id": "zz", "endpoint": "weak",
                                      "t1j_points": 1.0, "terminal_reason": "win"}], [], {}),
    ("an UNJUSTIFIED skip", CANON, [res(t) for t in CANON[:-1]],
     [{"task_id": CANON[-1]["task_id"], "endpoint": CANON[-1]["endpoint"]}], {}),
])
def test_no_screen_verdict_unless_it_binds_the_canonical_schedule(
        label, tasks, results, skipped, stopped):
    allowed, why = H.screen_verdict_allowed(CANON, tasks, results, skipped, stopped)
    assert not allowed, f"{label} was wrongly accepted"
    assert why


def test_a_synthetic_run_gets_a_RECEIPT_not_a_verdict(tmp_path):
    """Synthetic identities are not the canonical schedule, so a screen verdict
    can never be drawn from a qualification run."""
    tasks = [synthetic_task(0, "weak", anchor="red"), synthetic_task(1, "strong", anchor="black")]
    out = tmp_path / "synthetic.jsonl"
    rc = H._run(PLAN, str(out), mode="qualify", _tasks=tasks, _agent_factory=win_factory,
                _state_factory=small_state, _binder=null_binder, _evaluator=EV,
                _cleanup=lambda: None, _n_per_endpoint=2)
    rows = [json.loads(l) for l in open(out)]
    assert rc == H.EXIT_OK
    assert rows[-1]["record_type"] == "qualification_receipt"
    assert "not the canonical schedule" in rows[-1]["verdict_withheld"]
    assert [r["record_type"] for r in rows].count("task_result") == 2


# --- a completed game survives a cleanup failure ---------------------------

def test_a_cleanup_failure_does_not_erase_a_completed_game(tmp_path):
    """The game finished. Its record must be durable BEFORE cleanup runs."""
    def boom():
        raise RuntimeError("metal cache wedged")

    out = tmp_path / "cleanup_after_win.jsonl"
    with pytest.raises(H.AbortError) as e:
        H._run(PLAN, str(out), mode="qualify", _tasks=[synthetic_task(0, "weak", anchor="red")],
               _agent_factory=win_factory, _state_factory=small_state, _binder=null_binder,
               _evaluator=EV, _cleanup=boom, _n_per_endpoint=4)
    assert e.value.phase == H.PHASE_CLEANUP
    rows = [json.loads(l) for l in open(out)]
    kinds = [r["record_type"] for r in rows]
    result = [r for r in rows if r["record_type"] == "task_result"]
    assert len(result) == 1, "the completed game must survive the cleanup failure"
    assert result[0]["winner"] == "red" and result[0]["plies"] == 7
    assert kinds.index("task_result") < kinds.index("abort")
    assert rows[-1]["record_type"] == "abort" and rows[-1]["tasks_played"] == 1


# --- the identity gate has no off switch -----------------------------------

def test_the_reference_side_rejects_a_MISSING_expected_evaluator():
    """Forgetting to pass the evaluator must not look like disabling the check."""
    task = synthetic_task(anchor="red")               # reference plays black
    with pytest.raises(H.AbortError) as e:
        H._enforce_evaluator(_Agent(EV, [(0, 0)]), None, task, "black")
    assert e.value.phase == H.PHASE_FACTORY and "no expected evaluator" in e.value.message
    H._enforce_evaluator(_NoEvaluatorAgent([(0, 0)]), None, task, "red")   # anchor side is fine


def test_a_task_without_a_reference_colour_is_refused():
    with pytest.raises(H.AbortError):
        H._enforce_evaluator(_Agent(EV, [(0, 0)]), EV, {"task_id": "x"}, "red")


def test_a_run_with_no_evaluator_aborts_on_the_reference_side(tmp_path):
    out = tmp_path / "noev.jsonl"
    with pytest.raises(H.AbortError) as e:
        H._run(PLAN, str(out), mode="qualify", _tasks=[synthetic_task(0, "weak", anchor="red")],
               _agent_factory=win_factory, _state_factory=small_state, _binder=null_binder,
               _evaluator=None, _cleanup=lambda: None, _n_per_endpoint=4)
    assert e.value.phase == H.PHASE_FACTORY


# --- the verdict binds the ORDERED schedule and every frozen dimension ------

def _canon_res():
    return [res(t) for t in CANON]


def test_reversed_canonical_order_earns_no_verdict():
    """The 32 canonical names, all present, in reverse. Sets alone accepted this."""
    rev = list(reversed(CANON))
    allowed, why = H.screen_verdict_allowed(CANON, rev, [res(t) for t in rev], [], {})
    assert not allowed and "canonical schedule" in why


@pytest.mark.parametrize("field,value", [
    ("endpoint", "weak"),
    ("t1j_mdPly", 99),
    ("t1j_mdFixedPly", False),
    ("opening", "o9_fake"),
    ("colour_arm", "flipped"),
    ("anchor_colour", "black"),
    ("reference", "0379"),
    ("reference_sha1", "0" * 40),
    ("reference_colour", "red"),
    ("seed", 202612999),
])
def test_editing_any_frozen_dimension_earns_no_verdict(field, value):
    """Task ids unchanged; one frozen dimension edited. Sets alone accepted this."""
    edited = [dict(CANON[0], **{field: value})] + CANON[1:]
    assert [t["task_id"] for t in edited] == [t["task_id"] for t in CANON]
    if edited[0][field] == CANON[0][field]:
        pytest.skip(f"{field} already equals {value!r}; not a control")
    allowed, why = H.screen_verdict_allowed(CANON, edited, [res(t) for t in edited], [], {})
    assert not allowed, f"an edited {field} was wrongly accepted"


def test_the_untouched_canonical_schedule_still_earns_a_verdict():
    allowed, why = H.screen_verdict_allowed(CANON, CANON, _canon_res(), [], {})
    assert allowed, why


# --- a failure to record the result must not skip cleanup ------------------

class _FailOnTaskResult(H.Recorder):
    """Durable for everything except the record that matters most."""

    def emit(self, record):
        if record.get("record_type") == "task_result":
            raise OSError("no space left on device")
        return super().emit(record)


def test_a_result_write_failure_still_runs_cleanup_exactly_once(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "Recorder", _FailOnTaskResult)
    cleanups = []
    with pytest.raises(H.AbortError) as e:
        H._run(PLAN, str(tmp_path / "wf.jsonl"), mode="qualify",
               _tasks=[synthetic_task(0, "weak", anchor="red")],
               _agent_factory=win_factory, _state_factory=small_state, _binder=null_binder,
               _evaluator=EV, _cleanup=lambda: cleanups.append(1), _n_per_endpoint=4)
    assert e.value.phase == H.PHASE_RECORD
    assert "could not be recorded" in e.value.message
    assert len(cleanups) == 1, "cleanup must run exactly once for a STARTED task"


def test_a_result_write_failure_keeps_the_write_error_as_primary(tmp_path, monkeypatch):
    """If cleanup ALSO fails, the recording failure is what surfaces."""
    monkeypatch.setattr(H, "Recorder", _FailOnTaskResult)

    def boom():
        raise RuntimeError("cleanup also failed")

    with pytest.raises(H.AbortError) as e:
        H._run(PLAN, str(tmp_path / "wf2.jsonl"), mode="qualify",
               _tasks=[synthetic_task(0, "weak", anchor="red")],
               _agent_factory=win_factory, _state_factory=small_state, _binder=null_binder,
               _evaluator=EV, _cleanup=boom, _n_per_endpoint=4)
    assert e.value.phase == H.PHASE_RECORD


def test_an_unrecorded_game_is_not_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "Recorder", _FailOnTaskResult)
    with pytest.raises(H.AbortError):
        H._run(PLAN, str(tmp_path / "wf3.jsonl"), mode="qualify",
               _tasks=[synthetic_task(0, "weak", anchor="red")],
               _agent_factory=win_factory, _state_factory=small_state, _binder=null_binder,
               _evaluator=EV, _cleanup=lambda: None, _n_per_endpoint=4)
    rows = [json.loads(l) for l in open(tmp_path / "wf3.jsonl")]
    kinds = [r["record_type"] for r in rows]
    assert "task_result" not in kinds
    assert rows[-1]["record_type"] == "abort" and rows[-1]["tasks_played"] == 0


def test_the_header_counts_canonical_tasks_by_CONTENT_not_by_name(tmp_path):
    """Counting task_ids was a proxy: canonical NAMES on synthetic CONTENT counted."""
    borrowed = [dict(synthetic_task(i, "weak", anchor="red"), task_id=CANON[i]["task_id"])
                for i in range(2)]
    assert [t["task_id"] for t in borrowed] == [CANON[i]["task_id"] for i in range(2)]
    assert H.count_canonical(borrowed, CANON) == 0, "borrowed NAMES must not count"
    assert H.count_canonical(CANON, CANON) == 32
    assert H.count_canonical(CANON[:5], CANON) == 5
    assert H.count_canonical([dict(CANON[0], seed=1)], CANON) == 0, "an edited seed must not count"

    out = tmp_path / "hdr.jsonl"
    rc = H._run(PLAN, str(out), mode="qualify", _tasks=borrowed, _agent_factory=win_factory,
                _state_factory=small_state, _binder=null_binder, _evaluator=EV,
                _cleanup=lambda: None, _n_per_endpoint=16)
    hdr = json.loads(open(out).readline())
    assert rc == H.EXIT_OK
    assert hdr["canonical_tasks_executed"] == 0
    assert hdr["no_games"] is True and hdr["synthetic_tasks"] == 2


# --- screen mode: reachable only privately, and only with the real schedule -

def test_screen_mode_is_not_selectable_from_the_public_entry_point(tmp_path):
    with pytest.raises(H.HarnessError):
        H.run(PLAN, str(tmp_path / "s.jsonl"), mode="screen")
    assert not (tmp_path / "s.jsonl").exists()


def test_the_cli_still_refuses_screen_mode(tmp_path):
    r = cli("--plan", PLAN, "--results", str(tmp_path / "s.jsonl"), "--mode", "screen")
    assert r.returncode == H.EXIT_PRECONDITION and "UNAUTHORIZED" in r.stderr


class _StopAtStateFactory(Exception):
    pass


def refusing_state_factory(task):
    """Aborts before any agent is built, so no RNG and no move can occur."""
    raise H.AbortError(H.PHASE_PRECONDITION, "qualification stop: no game may be played")


def test_the_REAL_harness_refuses_the_SPENT_canonical_schedule_in_screen_mode(tmp_path):
    """POST-RUN STATE. The screen ran once on 2026-08-26; it may not run again.

    This test used to assert the opposite -- that the harness ACCEPTED this
    schedule -- and it was right until the schedule was executed. Its premise
    changed when the run happened, so the test changes with it rather than being
    kept alive by relaxing what the harness enforces.

    Every gate BEFORE seed availability is asserted to still pass, so the refusal
    is proved to be about the seeds and not about a digest that quietly broke.
    """
    out = tmp_path / "screen.jsonl"
    built = []

    H.verify_tasks(CANON)                                 # structure: still binds
    assert H.count_canonical(CANON, json.load(open(PLAN))["tasks"]) == H.CANONICAL_N_TASKS
    for t in CANON:
        H._assert_screen_seed(t)                          # still the reserved block

    with pytest.raises(H.HarnessError, match="may not be executed"):
        H._assert_screen_executable(CANON)                # THIS is what refuses now

    with pytest.raises(H.HarnessError) as e:
        H._run(PLAN, str(out), mode=H.SCREEN_MODE, _tasks=CANON,
               _agent_factory=lambda t, m, ev: built.append(m),
               _state_factory=refusing_state_factory, _binder=null_binder, _evaluator=EV,
               _cleanup=lambda: None, _n_per_endpoint=16, _ply_cap=280)
    assert "may not be executed" in str(e.value)
    assert built == [], "no agent may be constructed"
    assert not out.exists(), "a refused screen creates no results file at all"


def test_screen_mode_still_accepts_a_structurally_identical_UNSPENT_schedule(tmp_path):
    """THE CONTROL. The refusal above must be the seeds, not screen mode itself.

    A gate that rejects everything proves nothing. This rebuilds the canonical
    tasks against seeds that are inside the reserved block but neither drawn from
    nor retired -- which no longer exists, since the whole block retired -- so it
    asserts the reachable half instead: with retirement lifted for this call, the
    identical schedule passes the very check that just refused it.
    """
    unspent = [dict(t) for t in CANON]
    original = REF.RETIRED_SEED_INTERVALS
    original_exposed = REF.EXPOSED_SEED_INTERVALS
    try:
        REF.RETIRED_SEED_INTERVALS = ()
        REF.EXPOSED_SEED_INTERVALS = tuple(
            iv for iv in original_exposed if iv[0] not in (202612128, 202612144))
        H._assert_screen_executable(unspent)              # passes, same tasks
    finally:
        REF.RETIRED_SEED_INTERVALS = original
        REF.EXPOSED_SEED_INTERVALS = original_exposed
    assert REF.RETIRED_SEED_INTERVALS == original
    with pytest.raises(H.HarnessError):                   # and refuses again after
        H._assert_screen_executable(unspent)


def test_qualify_mode_still_refuses_every_canonical_seed(tmp_path):
    with pytest.raises(H.HarnessError) as e:
        H._run(PLAN, str(tmp_path / "q.jsonl"), mode="qualify", _tasks=CANON[:1],
               _agent_factory=win_factory, _state_factory=small_state, _binder=null_binder,
               _evaluator=EV, _cleanup=lambda: None)
    assert "accounted, exposed or retired" in str(e.value)


@pytest.mark.parametrize("mutate,label", [
    (lambda t: t[:31], "a short schedule"),
    (lambda t: list(reversed(t)), "a reordered schedule"),
    (lambda t: [dict(t[0], seed=999)] + t[1:], "an off-block seed"),
    (lambda t: [dict(t[0], opening="o9_fake")] + t[1:], "edited content"),
])
def test_screen_mode_refuses_anything_but_the_verified_canonical_schedule(
        tmp_path, mutate, label):
    with pytest.raises(H.HarnessError):
        H._run(PLAN, str(tmp_path / f"x.jsonl"), mode=H.SCREEN_MODE, _tasks=mutate(list(CANON)),
               _agent_factory=win_factory, _state_factory=refusing_state_factory,
               _binder=null_binder, _evaluator=EV, _cleanup=lambda: None, _n_per_endpoint=16)


def test_screen_mode_refuses_a_seed_outside_the_reserved_block():
    assert H.SCREEN_SEED_BLOCK == (202612128, 202612160)
    with pytest.raises(H.HarnessError):
        H._assert_screen_seed({"task_id": "x", "seed": 202612127})
    with pytest.raises(H.HarnessError):
        H._assert_screen_seed({"task_id": "x", "seed": 202612160})
    H._assert_screen_seed({"task_id": "x", "seed": 202612128})


def test_the_completed_screen_can_still_be_reclassified_from_its_records():
    """REPLAY ANALYSIS after the seeds are spent. 'Spent' must not mean unreadable.

    The canonical run's durable JSONL is reclassified with the harness's own
    classifier and must reproduce the verdict the run recorded. Nothing here
    needs a seed, a model or a JVM -- which is the property the split protects.
    """
    run = ("docs/superpowers/evidence/2026-08-26-t1j-e4-canonical-screen/"
           "07_e4_screen_results.jsonl")
    rows = [json.loads(l) for l in open(run)]
    results = [r for r in rows if r["record_type"] == "task_result"]
    recorded = [r for r in rows if r["record_type"] == "verdict"][-1]

    again = H.classify_run(results, n_per_endpoint=H.CANONICAL_N_PER_ENDPOINT, band=H.BAND)
    assert again["per_endpoint"] == recorded["per_endpoint"]
    assert again["joint"] == recorded["joint"] == "IN_BAND"
    assert again["larger_match_permitted"] == recorded["larger_match_permitted"] is True
    assert again["per_endpoint"]["weak"] == {
        "played": 16, "score": 0.0, "cap_terminations": 0, "decision": "SATURATED_WEAK"}
    assert again["per_endpoint"]["strong"] == {
        "played": 8, "score": 7.0, "cap_terminations": 0, "decision": "IN_BAND"}
