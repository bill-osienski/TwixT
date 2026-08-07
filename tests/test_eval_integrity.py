"""Zero-tolerance integrity checks (design spec section 8.3)."""
import pathlib

import pytest

from scripts.GPU.alphazero import eval_readout as R
from scripts.GPU.alphazero.eval_integrity import (
    IntegrityError, validate_game_binding, validate_ply, validate_result_set,
    validate_seed_intervals,
)
from scripts.GPU.alphazero.eval_runner import (
    AgentSpec, EvalGameTask, build_agent_pairing_tasks, build_pairing_tasks,
    make_result,
)

CKPT = str(pathlib.Path(__file__).parent / "eval_fakes.py")
A = AgentSpec("candidate", CKPT, R.ReadoutConfig(mode=R.MODE_ARGMAX))
B = AgentSpec("control", CKPT, R.ReadoutConfig(mode=R.MODE_OPENING_TEMPERATURE))


def _t2(nl=190, nc=40):
    return [R.ChildStat((2, 2), nl, 0.3, -0.3),
            R.ChildStat((1, 1), nc, -0.05, 0.05)]


# --- validate_ply ----------------------------------------------------------


def test_valid_ply_passes():
    validate_ply(5, expected_sims=400, root_visit_count=400,
                 root_value=0.12, top2=_t2())


def test_budget_mismatch_raises():
    with pytest.raises(IntegrityError, match="budget"):
        validate_ply(5, expected_sims=400, root_visit_count=399,
                     root_value=0.12, top2=_t2())


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_root_value_raises(bad):
    with pytest.raises(IntegrityError, match="root_value"):
        validate_ply(5, expected_sims=400, root_visit_count=400,
                     root_value=bad, top2=_t2())


def test_non_finite_q_on_a_VISITED_child_raises():
    bad = [R.ChildStat((2, 2), 190, float("nan"), float("nan")),
           R.ChildStat((1, 1), 40, -0.05, 0.05)]
    with pytest.raises(IntegrityError, match="q_value"):
        validate_ply(5, expected_sims=400, root_visit_count=400,
                     root_value=0.1, top2=bad)


def test_none_q_on_an_UNVISITED_child_is_allowed():
    # None on a zero-visit child is an UNDEFINED mean, not corrupt telemetry.
    ok = [R.ChildStat((2, 2), 190, 0.3, -0.3),
          R.ChildStat((1, 1), 0, None, None)]
    validate_ply(5, expected_sims=400, root_visit_count=400,
                 root_value=0.1, top2=ok)


def test_none_q_on_a_VISITED_child_raises():
    bad = [R.ChildStat((2, 2), 190, None, None),
           R.ChildStat((1, 1), 40, -0.05, 0.05)]
    with pytest.raises(IntegrityError, match="q_value"):
        validate_ply(5, expected_sims=400, root_visit_count=400,
                     root_value=0.1, top2=bad)


def test_empty_top2_is_allowed_the_root_may_have_no_children():
    validate_ply(5, expected_sims=400, root_visit_count=400,
                 root_value=0.1, top2=[])


# --- validate_seed_intervals ----------------------------------------------


def test_adjacent_half_open_intervals_do_not_overlap():
    validate_seed_intervals([64, 128], [[0, 64]])
    validate_seed_intervals([0, 64], [[64, 128]])


@pytest.mark.parametrize("prior", [
    [60, 70],      # straddles the start
    [120, 200],    # straddles the end
    [0, 1000],     # contains it
    [70, 80],      # contained by it
    [64, 128],     # identical
])
def test_overlapping_intervals_raise(prior):
    with pytest.raises(ValueError, match="overlap"):
        validate_seed_intervals([64, 128], [prior])


def test_empty_or_reversed_CURRENT_interval_raises():
    with pytest.raises(ValueError, match="current.*empty or reversed"):
        validate_seed_intervals([100, 100], [])
    with pytest.raises(ValueError, match="current.*empty or reversed"):
        validate_seed_intervals([200, 100], [])


def test_a_reversed_PRIOR_interval_raises():
    """CONSTRUCTED: [200, 100) is malformed history. Checking only
    current-versus-each-prior would let it through, because a reversed
    interval overlaps nothing."""
    with pytest.raises(ValueError, match=r"prior\[0\].*empty or reversed"):
        validate_seed_intervals([1000, 1064], [[200, 100]])


def test_an_empty_PRIOR_interval_raises():
    with pytest.raises(ValueError, match=r"prior\[1\].*empty or reversed"):
        validate_seed_intervals([1000, 1064], [[0, 64], [300, 300]])


def test_two_PRIORS_overlapping_each_other_raises():
    """CONSTRUCTED: both priors are disjoint from the current interval, so
    only a whole-set pairwise check can catch them overlapping each other."""
    with pytest.raises(ValueError, match=r"prior\[0\].*overlaps prior\[1\]"):
        validate_seed_intervals([1000, 1064], [[0, 100], [50, 150]])


def test_a_fully_valid_set_passes():
    validate_seed_intervals([1000, 1064], [[0, 64], [64, 128], [500, 600]])


# --- validate_game_binding -------------------------------------------------


def test_game_binding_accepts_a_correct_result():
    tasks = build_agent_pairing_tasks("p", A, B, 4, 100)
    validate_game_binding(make_result(tasks[0], "red", "win", 40), tasks[0])


def test_game_binding_is_validated_immediately_not_at_the_end():
    """Spec 8.3 requires an IMMEDIATE stop, so the per-game guard must reject
    a single bad result on its own, without needing the rest of the run."""
    tasks = build_agent_pairing_tasks("p", A, B, 4, 100)
    bad = make_result(tasks[0], "red", "win", 40)
    bad.red_readout = dict(bad.black_readout)
    with pytest.raises(IntegrityError, match="configuration"):
        validate_game_binding(bad, tasks[0])


def test_game_binding_rejects_unknown_error_on_its_own():
    tasks = build_agent_pairing_tasks("p", A, B, 2, 100)
    r = make_result(tasks[0], None, "unknown_error", 40)
    with pytest.raises(IntegrityError, match="unknown_error"):
        validate_game_binding(r, tasks[0])


def test_game_binding_rejects_a_swapped_agent_id():
    tasks = build_agent_pairing_tasks("p", A, B, 2, 100)
    r = make_result(tasks[0], "red", "win", 40)
    r.red_agent_id, r.black_agent_id = r.black_agent_id, r.red_agent_id
    with pytest.raises(IntegrityError, match="does not match the task binding"):
        validate_game_binding(r, tasks[0])


def test_game_binding_rejects_one_agent_holding_both_colours():
    """CONSTRUCTED: a malformed TASK that binds the same agent to both
    colours.

    Mutating a well-formed result instead would trip the id-vs-binding check
    first and never reach this guard, so the mutation has to be in the task.
    Both ids then match their binding and only the both-colours check can fire.
    """
    task = EvalGameTask(task_id=0, pairing_id="p", game_idx=0,
                        red_checkpoint=CKPT, black_checkpoint=CKPT,
                        seed=100, red_agent=A, black_agent=A)
    r = make_result(task, "red", "win", 40)
    with pytest.raises(IntegrityError, match="both colours"):
        validate_game_binding(r, task)


def test_game_binding_leaves_LEGACY_results_alone():
    """CONSTRUCTED: legacy checkpoint tasks carry no AgentSpec, and
    eval_checkpoint_match's behaviour must not change. The guard returns
    without inspecting them -- including their termination reason."""
    tasks = build_pairing_tasks("p", "a.safetensors", "b.safetensors", 2, 100, 0)
    validate_game_binding(make_result(tasks[0], "red", "win", 40), tasks[0])
    validate_game_binding(make_result(tasks[0], None, "unknown_error", 40),
                          tasks[0])


# --- validate_result_set ---------------------------------------------------


def _ok_set(n=4):
    tasks = build_agent_pairing_tasks("p", A, B, n, 100)
    results = [make_result(t, "red", "win", 40) for t in tasks]
    return results, tasks


def test_valid_result_set_passes():
    results, tasks = _ok_set()
    validate_result_set(results, tasks, "candidate", "control")


def test_unknown_error_raises():
    results, tasks = _ok_set()
    results[1].reason = "unknown_error"
    with pytest.raises(IntegrityError, match="unknown_error"):
        validate_result_set(results, tasks, "candidate", "control")


def test_missing_result_raises():
    results, tasks = _ok_set()
    with pytest.raises(IntegrityError, match="incomplete"):
        validate_result_set(results[:-1], tasks, "candidate", "control")


def test_duplicate_task_id_raises():
    """Duplicates are reported as duplicates, not as the binding mismatch a
    collided lookup would produce. The structural checks run first for exactly
    this reason."""
    results, tasks = _ok_set()
    results[1].task_id = results[0].task_id
    with pytest.raises(IntegrityError, match="duplicate"):
        validate_result_set(results, tasks, "candidate", "control")


def test_unexpected_agent_id_raises():
    results, tasks = _ok_set()
    results[0].red_agent_id = "impostor"
    with pytest.raises(IntegrityError, match="agent"):
        validate_result_set(results, tasks, "candidate", "control")


def test_a_SYSTEMATIC_config_leak_is_caught():
    """The decisive case: every row carries the wrong config CONSISTENTLY.

    A consistency-only check (does this agent always play the same readout?)
    passes here, because the wrong config is applied uniformly. Only a
    comparison against the TASK's expected config catches it -- and a
    systematic leak is exactly the failure that would silently invalidate a
    whole match.
    """
    results, tasks = _ok_set()
    wrong = {"mode": R.MODE_HOEFFDING_LCB, "opening_temp_plies": 20,
             "temp_high": 1.0, "temp_low": 0.1}
    for r in results:
        if r.red_agent_id == "candidate":
            r.red_readout = dict(wrong)
        else:
            r.black_readout = dict(wrong)
    with pytest.raises(IntegrityError, match="configuration"):
        validate_result_set(results, tasks, "candidate", "control")


def test_colour_imbalance_raises():
    """CONSTRUCTED: hand-built tasks that never swap colours.

    Every result matches its task binding exactly, so validate_game_binding is
    satisfied and ONLY the whole-run colour-balance check can fire. Mutating a
    balanced result set instead would trip the binding guard first and prove
    nothing about colour balance.
    """
    tasks = [
        EvalGameTask(task_id=i, pairing_id="p", game_idx=i,
                     red_checkpoint=CKPT, black_checkpoint=CKPT,
                     seed=100 + i, red_agent=A, black_agent=B)
        for i in range(4)
    ]
    results = [make_result(t, "red", "win", 40) for t in tasks]
    # Sanity: the bindings really are clean, so the guard below is isolated.
    for r, t in zip(results, tasks):
        validate_game_binding(r, t)
    with pytest.raises(IntegrityError, match="colour balance"):
        validate_result_set(results, tasks, "candidate", "control")
