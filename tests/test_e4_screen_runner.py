"""Fail-closed checks for the E4 execution harness. No model, no game, no seed.

Everything here runs on SYNTHETIC seeds. The canonical schedule's seeds
[202612128, 202612160) appear only as inert plan data: no test constructs a
generator from one, and the harness refuses to.
"""
import inspect
import json

import pytest

from scripts.GPU.alphazero import e4_screen_runner as H

PLAN = "docs/superpowers/evidence/2026-08-25-t1j-e4-preflight-attempt4/06_endpoint_screen_plan.json"
SYNTHETIC = 90000001


def synthetic_task(i=0, endpoint="weak", points=1.0, reason="win", abort=None):
    return {"task_id": f"synthetic-{i:03d}", "endpoint": endpoint,
            "seed": SYNTHETIC + i, "_points": points, "_reason": reason, "_abort": abort}


def stub_factory(evaluator):
    class _Agent:
        def __init__(self): self.evaluator = evaluator
    return lambda task, ev: _Agent()


def stub_opponent(task, agent, rec):
    if task.get("_abort"):
        return {"abort": task["_abort"]}
    rec.emit({"record_type": "ply", "task_id": task["task_id"], "ply": 0})
    return {"t1j_points": task["_points"], "terminal_reason": task["_reason"], "plies": 1}


# --- the public surface accepts paths only ---------------------------------

def test_public_run_accepts_paths_only():
    params = inspect.signature(H.run).parameters
    assert set(params) == {"plan_path", "results_path", "mode"}
    for forbidden in ("task", "tasks", "agent", "factory", "evaluator",
                      "cleanup", "classifier", "schedule", "opponent"):
        assert not any(forbidden in p for p in params), forbidden
    assert params["mode"].kind is inspect.Parameter.KEYWORD_ONLY


def test_screen_mode_is_unauthorized(tmp_path):
    for mode in ("screen", "games", "", None):
        with pytest.raises(H.HarnessError):
            H.run(PLAN, str(tmp_path / "r.jsonl"), mode=mode)


def test_public_run_builds_nothing_without_injection(tmp_path):
    """With no stubs there is no agent factory, so the loop cannot run tasks."""
    assert H.run(PLAN, str(tmp_path / "r.jsonl"), mode="qualify") == 0
    rows = [json.loads(l) for l in open(tmp_path / "r.jsonl")]
    assert rows[0]["record_type"] == "run_header"
    assert rows[0]["canonical_tasks"] == 32
    assert rows[0]["canonical_tasks_executed"] == 0
    assert not [r for r in rows if r["record_type"] == "task_result"]


# --- the schedule cannot be injected or reshaped ---------------------------

def canonical_tasks():
    return json.load(open(PLAN))["tasks"]


def test_canonical_plan_verifies():
    plan = H.load_canonical_plan(PLAN)
    assert len(plan["tasks"]) == H.CANONICAL_N_TASKS == 32


def test_tampered_plan_file_is_refused(tmp_path):
    raw = json.load(open(PLAN))
    raw["note"] = "tampered"
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(raw))
    with pytest.raises(H.HarnessError):
        H.load_canonical_plan(str(p))


def _moved(mutate):
    """A mutation that changes nothing is not a control."""
    t = canonical_tasks()
    return H.task_digest(mutate(t)) != H.task_digest(t) or len(mutate(t)) != len(t)


@pytest.mark.parametrize("mutate,label", [
    (lambda t: t + [dict(t[0], task_id="extra")], "addition"),
    (lambda t: t[:-1], "removal"),
    (lambda t: [t[1], t[0]] + t[2:], "reordering"),
    (lambda t: t[:-1] + [dict(t[0])], "duplicate"),
    (lambda t: [dict(t[0], endpoint="weak" if t[0]["endpoint"] == "strong" else "strong")]
     + t[1:], "endpoint edit"),
    (lambda t: [dict(t[0], t1j_mdPly=7)] + t[1:], "depth edit"),
    (lambda t: [dict(t[0], opening="o9_fake")] + t[1:], "opening edit"),
    (lambda t: [dict(t[0], colour_arm="t1j_black")] + t[1:], "colour edit"),
    (lambda t: [dict(t[0], reference="0379")] + t[1:], "reference edit"),
    (lambda t: [dict(t[0], reference_sha1="0" * 40)] + t[1:], "reference hash edit"),
    (lambda t: [dict(t[0], seed=202612999)] + t[1:], "seed edit"),
])
def test_reshaping_the_schedule_is_refused(mutate, label):
    assert _moved(mutate), f"{label}: INVALID CONTROL — the mutation changed nothing"
    with pytest.raises(H.HarnessError):
        H.verify_tasks(mutate(canonical_tasks()))


def test_task_digest_is_order_sensitive():
    t = canonical_tasks()
    assert H.task_digest(t) == H.CANONICAL_TASK_DIGEST
    assert H.task_digest([t[1], t[0]] + t[2:]) != H.CANONICAL_TASK_DIGEST


# --- scheduled seeds are inert ---------------------------------------------

def test_scheduled_seeds_are_refused_by_the_runner():
    for task in canonical_tasks()[:4]:
        with pytest.raises(H.HarnessError):
            H._assert_not_scheduled(task)


def test_synthetic_seeds_are_allowed():
    H._assert_not_scheduled(synthetic_task())


def test_running_a_canonical_task_is_refused(tmp_path):
    with pytest.raises(H.HarnessError):
        H._run(PLAN, str(tmp_path / "r.jsonl"), mode="qualify",
               _tasks=canonical_tasks()[:1], _agent_factory=stub_factory(object()),
               _opponent=stub_opponent, _cleanup=lambda: None)


# --- the stub loop: recording, cleanup, evaluator reuse, abort --------------

def run_stub(tmp_path, tasks, evaluator=None, factory=None, cleanup=None, n=2):
    ev = evaluator if evaluator is not None else object()
    calls = []
    out = tmp_path / "r.jsonl"
    rc = H._run(PLAN, str(out), mode="qualify", _tasks=tasks,
                _agent_factory=factory or stub_factory(ev),
                _opponent=stub_opponent,
                _cleanup=cleanup or (lambda: calls.append(1)),
                _n_per_endpoint=n)
    return rc, [json.loads(l) for l in open(out)], calls


def test_records_are_appended_as_they_happen(tmp_path):
    tasks = [synthetic_task(0, "weak"), synthetic_task(1, "weak", 0.0, "loss")]
    rc, rows, calls = run_stub(tmp_path, tasks)
    assert rc == 0
    kinds = [r["record_type"] for r in rows]
    assert kinds[0] == "run_header" and kinds[-1] == "verdict"
    assert kinds.count("task_start") == kinds.count("task_result") == 2
    assert kinds.index("task_start") < kinds.index("ply") < kinds.index("task_result")
    assert len(calls) == 2                       # cleanup once per task


def test_one_evaluator_serves_every_construction(tmp_path):
    ev = object()
    _, rows, _ = run_stub(tmp_path, [synthetic_task(i, "weak") for i in range(4)], evaluator=ev)
    assert rows[-1]["distinct_evaluators"] == 1


def test_a_rebuilt_evaluator_is_detected(tmp_path):
    """NEGATIVE CONTROL. A factory that hands back a fresh evaluator each time
    must show up as more than one distinct instance."""
    class _Agent:
        def __init__(self): self.evaluator = object()      # rebuilt every call
    _, rows, _ = run_stub(tmp_path, [synthetic_task(i, "weak") for i in range(4)],
                          factory=lambda task, ev: _Agent())
    assert rows[-1]["distinct_evaluators"] == 4 != 1


def test_abort_stops_the_run_and_keeps_partial_records(tmp_path):
    tasks = [synthetic_task(0, "weak"), synthetic_task(1, "weak", abort="divergence at ply 7")]
    with pytest.raises(H.HarnessError):
        run_stub(tmp_path, tasks)
    rows = [json.loads(l) for l in open(tmp_path / "r.jsonl")]
    kinds = [r["record_type"] for r in rows]
    assert "abort" in kinds
    assert kinds.count("task_result") == 1          # the completed one persisted
    assert "verdict" not in kinds                   # NO verdict from a partial run


# --- classification, including partial-result refusal ----------------------

def result(endpoint, points, reason="win"):
    return {"endpoint": endpoint, "t1j_points": points, "terminal_reason": reason}


def verdict(weak, strong, n=16):
    rows = [result("weak", p) for p in weak] + [result("strong", p) for p in strong]
    return H.classify_run(rows, n_per_endpoint=n, band=[0.05, 0.95])


@pytest.mark.parametrize("weak,strong,want", [
    ([1.0] * 16, [1.0] * 16, "T1J_TOO_STRONG"),
    ([0.0] * 16, [0.0] * 16, "T1J_TOO_WEAK"),
    ([0.0] * 16, [1.0] * 16, "BRACKETED"),
    ([0.5] * 16, [0.5] * 16, "IN_BAND"),
    ([0.0] * 4, [0.5] * 16, "INCONCLUSIVE"),        # weak incomplete
])
def test_every_joint_outcome_is_reachable(weak, strong, want):
    assert verdict(weak, strong)["joint"] == want


def test_partial_results_are_refused_a_verdict():
    v = verdict([1.0] * 3, [0.0] * 3)
    assert v["per_endpoint"]["weak"]["decision"] == "INCOMPLETE"
    assert v["joint"] == "INCONCLUSIVE"


def test_cap_terminations_make_an_endpoint_incomplete():
    rows = [result("weak", 0.5, "cap") for _ in range(9)] + \
           [result("weak", 1.0) for _ in range(7)] + [result("strong", 0.5) for _ in range(16)]
    v = H.classify_run(rows, n_per_endpoint=16, band=[0.05, 0.95])
    assert v["per_endpoint"]["weak"]["decision"] == "INCOMPLETE"
    assert v["joint"] == "INCONCLUSIVE" and v["larger_match_permitted"] is True
