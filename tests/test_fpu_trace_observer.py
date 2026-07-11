"""Tests for the per-completed-simulation observer hook + canonical
visit-leader (Task 3): `visit_leader_move`, the guarded `_backup` observer
tail, `MCTS.__init__(..., observer=None)`, and the joint (Task 0-3)
byte-identical-off proof against the pre-branch golden.

Frozen design ref: docs/superpowers/specs/2026-07-10-context-relative-fpu-policy-mass-design.md
§2 ("Optional observer on the single real 400-sim run") + "Isolation".

Direct `_backup` tests below use hand-built synthetic `MCTSNode` trees + a
CPU stub value fn (same pattern as `tests/test_fpu_value.py` /
`tests/test_fpu_policy_mass_rule.py`) -- no real search/GPU/MLX involved.
The pre-expansion, integration, and golden-off tests drive the real CPU
FakeEvaluator search harness (`tests/fpu_search_fixture.py`).
"""
import json
import random

from scripts.GPU.alphazero.mcts import (
    MCTS,
    MCTSConfig,
    MCTSNode,
    encode_move,
    visit_leader_move,
)
from tests.fpu_search_fixture import run_search


def _stub_value_fn():
    def f(state):
        return {}, 0.0
    return f


class _Spy:
    def __init__(self): self.calls = []
    def on_root_simulation(self, count, root, move, leader): self.calls.append((count, move, leader))


# ---------------------------------------------------------------------------
# visit_leader_move -- canonical MCTS visit-leader comparator. Kept
# IDENTICAL to continuation_extraction._best_child (max visit_count, ties
# broken by lowest encoded move id); not imported from there directly since
# continuation_extraction already does `from .mcts import decode_move` and
# importing it back would be circular.
# ---------------------------------------------------------------------------

def test_visit_leader_move_picks_max_visit_child():
    root = MCTSNode(state=None)
    A, B = encode_move(0, 0), encode_move(1, 1)
    root.priors = {A: 0.5, B: 0.5}
    root.children[A] = MCTSNode(state=None, parent=root, move=A, visit_count=3)
    root.children[B] = MCTSNode(state=None, parent=root, move=B, visit_count=7)
    assert visit_leader_move(root) == B


def test_visit_leader_move_ties_break_to_lowest_move_id():
    root = MCTSNode(state=None)
    A, B = encode_move(0, 0), encode_move(5, 5)
    assert A < B                                    # sanity: tie-break needs a real ordering
    root.priors = {A: 0.5, B: 0.5}
    root.children[A] = MCTSNode(state=None, parent=root, move=A, visit_count=4)
    root.children[B] = MCTSNode(state=None, parent=root, move=B, visit_count=4)
    assert visit_leader_move(root) == A


def test_visit_leader_move_none_when_no_visited_child():
    root = MCTSNode(state=None)
    A = encode_move(0, 0)
    root.priors = {A: 1.0}
    root.children[A] = MCTSNode(state=None, parent=root, move=A, visit_count=0)  # pending, not a leader
    assert visit_leader_move(root) is None


def test_visit_leader_move_none_when_no_children_at_all():
    root = MCTSNode(state=None)
    root.priors = {}
    assert visit_leader_move(root) is None


# ---------------------------------------------------------------------------
# Direct _backup unit tests -- hand-built two-node search_path, no full
# search. Exercises the guarded observer tail in isolation.
# ---------------------------------------------------------------------------

def _root_with_one_child(visit_count=0):
    root = MCTSNode(state=None)
    A = encode_move(2, 3)
    root.priors = {A: 1.0}
    child = MCTSNode(state=None, parent=root, move=A, visit_count=visit_count)
    root.children[A] = child
    return root, child, A


def test_backup_calls_observer_exactly_once():
    spy = _Spy()
    m = MCTS(_stub_value_fn(), MCTSConfig(n_simulations=1), random.Random(0), observer=spy)
    root, child, _A = _root_with_one_child()
    m._backup([root, child], 0.5)
    assert len(spy.calls) == 1


def test_backup_two_calls_produce_two_observer_calls_with_incrementing_count():
    spy = _Spy()
    m = MCTS(_stub_value_fn(), MCTSConfig(n_simulations=1), random.Random(0), observer=spy)
    root, child, _A = _root_with_one_child()
    m._backup([root, child], 0.5)
    m._backup([root, child], -0.2)
    assert len(spy.calls) == 2
    assert [c for c, _, _ in spy.calls] == [1, 2]


def test_backup_reports_updated_root_move_from_search_path():
    spy = _Spy()
    m = MCTS(_stub_value_fn(), MCTSConfig(n_simulations=1), random.Random(0), observer=spy)
    root, child, A = _root_with_one_child()
    m._backup([root, child], 0.3)
    count, move, _leader = spy.calls[0]
    assert count == 1
    assert move == A == child.move


def test_backup_reports_none_move_for_length_one_search_path():
    # A bare-root path (len(search_path) == 1) is the defensive branch noted
    # in the plan: search_with_root's real per-sim path never produces this
    # (root is expanded before the sim loop -- see the pre-expansion
    # invariant test below), but _backup must still handle it correctly for
    # any other caller.
    spy = _Spy()
    m = MCTS(_stub_value_fn(), MCTSConfig(n_simulations=1), random.Random(0), observer=spy)
    root = MCTSNode(state=None)
    m._backup([root], 0.7)
    count, move, _leader = spy.calls[0]
    assert count == 1
    assert move is None


def test_backup_reports_leader_using_post_backup_visit_counts():
    # visit_leader_move(root) must reflect the visit_count bump _backup just
    # applied -- the observer fires AFTER the loop, not before.
    spy = _Spy()
    m = MCTS(_stub_value_fn(), MCTSConfig(n_simulations=1), random.Random(0), observer=spy)
    root, child, A = _root_with_one_child(visit_count=0)
    assert visit_leader_move(root) is None            # pre-backup: child has 0 visits, no leader yet
    m._backup([root, child], 0.4)
    _count, _move, leader = spy.calls[0]
    assert leader == A == visit_leader_move(root)      # post-backup: child now has 1 visit


def test_backup_observer_off_does_not_raise_and_creates_no_attr():
    m = MCTS(_stub_value_fn(), MCTSConfig(n_simulations=1), random.Random(0))  # observer defaults to None
    assert not hasattr(m, "_observer_completed_count")    # fix 5b: attr never created when off
    root, child, _A = _root_with_one_child()
    m._backup([root, child], 0.5)                         # must not raise
    assert not hasattr(m, "_observer_completed_count")     # still absent after a real backup
    assert root.visit_count == 1 and child.visit_count == 1   # backup itself still worked normally


# ---------------------------------------------------------------------------
# Root pre-expansion invariant (fix 5a) -- search_with_root calls
# _expand(root) before the sim loop, so sim 1 already has a real root move
# (never None on this path; the None branch in _backup is defensive for
# other callers, exercised directly above).
# ---------------------------------------------------------------------------

def test_search_with_root_pre_expands_root_so_first_sim_has_a_root_move():
    spy = _Spy(); _out, root, _m = run_search(n_sims=1, observer=spy)
    assert len(spy.calls) == 1 and spy.calls[0][0] == 1
    assert spy.calls[0][1] is not None            # proves root was expanded before sim 1


# ---------------------------------------------------------------------------
# Integration (fix 2/5a) -- full 200-sim search: exactly one callback per
# completed sim, no gaps/dups, every move is None or a legal root move, and
# the final callback's leader matches visit_leader_move(root) on the
# finished tree.
# ---------------------------------------------------------------------------

def test_one_callback_per_completed_sim():
    spy = _Spy(); _out, root, _m = run_search(n_sims=200, observer=spy)
    assert [c for c, _, _ in spy.calls] == list(range(1, 201))     # exactly 1..n, no gaps/dups
    legal = set(root.priors.keys())
    assert all((m is None) or (m in legal) for _, m, _ in spy.calls)   # None allowed by contract
    assert spy.calls[-1][2] == visit_leader_move(root)            # final leader == final visit leader


# ---------------------------------------------------------------------------
# Observer-off reproduces the pre-branch golden -- the joint Task 0-3
# byte-identity proof. Default MCTSConfig() (fpu_policy_mass_reduction=None)
# + observer=None must be bit-for-bit identical to the golden captured in
# Task 0 from UNMODIFIED mcts.py.
# ---------------------------------------------------------------------------

def test_observer_off_matches_prebranch_golden():
    out, _r, _m = run_search()                    # default config, observer None
    assert out == json.load(open("tests/golden/fpu_prebranch_search.json"))


# ===========================================================================
# FpuTraceObserver (Task 7) -- the diagnostic-side observer that consumes the
# `on_root_simulation(count, root, updated_root_move, current_root_leader_move)`
# callback INCREMENTALLY (no 200-child rescan/sim). Records: first-visit sim
# per root move; explored-mass 25/50/75% crossings; leader-change timeline
# (leader taken from the PASSED arg, never recomputed); final-leader
# last-takeover == stabilization; a `None` root move ignored for first-visit/
# mass but still counted; end-state selected-move prior + rank + top share.
# Driven by direct synthetic call sequences AND the real CPU FakeEvaluator
# search (no GPU/MLX).
# ===========================================================================
from scripts.GPU.alphazero.diagnose_fpu_policy_mass import FpuTraceObserver
from scripts.GPU.alphazero.mcts import visit_leader_move as _vlm


def _root_with_priors(priors_by_move, *, visits=None, root_visits=0):
    """Synthetic root: `priors_by_move` = {move_id: prior}. `visits` (optional)
    = {move_id: visit_count} instantiates children with those counts so the
    finalize-time top-share / prior-rank read a realistic end state."""
    root = MCTSNode(state=None, visit_count=root_visits)
    root.priors = dict(priors_by_move)
    for mid in priors_by_move:
        vc = (visits or {}).get(mid, 0)
        if vc:
            root.children[mid] = MCTSNode(state=None, parent=root, move=mid, visit_count=vc)
    return root


def test_observer_first_visit_sims_and_mass_crossings():
    A, B, C = encode_move(0, 0), encode_move(1, 1), encode_move(2, 2)
    root = _root_with_priors({A: 0.5, B: 0.3, C: 0.2})
    obs = FpuTraceObserver()
    obs.on_root_simulation(1, root, A, A)      # A first visited -> mass 0.5 (>=0.25,>=0.50)
    obs.on_root_simulation(2, root, A, A)      # revisit A -> no new first-visit, mass unchanged
    obs.on_root_simulation(3, root, B, A)      # B -> mass 0.8 (>=0.75)
    obs.on_root_simulation(4, root, C, A)      # C -> mass 1.0
    assert obs.first_visit_sim == {A: 1, B: 3, C: 4}
    assert obs.mass_cross_sim == {0.25: 1, 0.50: 1, 0.75: 3}


def test_observer_leader_timeline_and_stabilization():
    A, B = encode_move(0, 0), encode_move(1, 1)
    root = _root_with_priors({A: 0.6, B: 0.4})
    obs = FpuTraceObserver()
    for count, leader in enumerate([A, A, B, B, A], start=1):
        obs.on_root_simulation(count, root, A, leader)     # move fixed; leader is the PASSED arg
    assert obs.leader_timeline == [(1, A), (3, B), (5, A)]  # change-points only
    assert obs.final_leader_move == A
    assert obs.stabilization_sim == 5                      # final leader's LAST takeover


def test_observer_takes_leader_from_passed_arg_not_recomputed():
    # Build an end state whose ACTUAL visit-leader is B, but PASS A as the
    # leader every sim -> the observer must report A (proving it never recomputes).
    A, B = encode_move(0, 0), encode_move(4, 4)
    root = _root_with_priors({A: 0.5, B: 0.5}, visits={A: 3, B: 9}, root_visits=12)
    assert _vlm(root) == B                                  # the recomputed leader would be B
    obs = FpuTraceObserver()
    for count in range(1, 6):
        obs.on_root_simulation(count, root, A, A)           # pass A, contradicting the tree
    assert obs.final_leader_move == A
    assert obs.leader_timeline == [(1, A)]


def test_observer_ignores_none_root_move_but_counts_the_sim():
    A = encode_move(0, 0)
    root = _root_with_priors({A: 0.5, encode_move(1, 1): 0.5})
    obs = FpuTraceObserver()
    obs.on_root_simulation(1, root, None, None)             # None: no first-visit, no mass
    obs.on_root_simulation(2, root, A, A)                   # A first visited AT sim 2, not 1
    assert obs.first_visit_sim == {A: 2}                    # None never created a phantom move
    assert abs(obs.explored_mass - 0.5) < 1e-12             # only A's prior counted
    assert obs.completed_simulation_count == 2              # counter still advanced past the None sim


def test_observer_finalize_selected_move_prior_rank_and_top_share():
    # Final leader is the LOW-prior move C (prior 0.2 -> rank 3); its subtree
    # dominates the visits (top share 0.8). This is the lock-in geometry §6.1
    # is built to catch.
    A, B, C = encode_move(0, 0), encode_move(1, 1), encode_move(2, 2)
    root = _root_with_priors({A: 0.5, B: 0.3, C: 0.2},
                             visits={A: 1, B: 1, C: 8}, root_visits=10)
    obs = FpuTraceObserver()
    obs.on_root_simulation(1, root, A, A)
    obs.on_root_simulation(2, root, B, A)
    obs.on_root_simulation(3, root, C, C)          # C takes over and holds
    res = obs.result()
    assert res["selected_move_prior"] == 0.2
    assert res["selected_move_prior_rank"] == 3
    assert abs(res["final_root_top_share"] - 0.8) < 1e-12
    assert res["stabilization_sim"] == 3
    assert abs(res["explored_mass_at_stabilization"] - 1.0) < 1e-12   # A+B+C all visited by sim 3


def test_observer_over_real_cpu_search_is_consistent():
    obs = FpuTraceObserver(); spy = _Spy()

    class _Fan:                                     # fan the callback to both observers
        def on_root_simulation(self, *a):
            obs.on_root_simulation(*a); spy.on_root_simulation(*a)

    _out, root, _m = run_search(n_sims=200, observer=_Fan())
    assert obs.completed_simulation_count == 200
    # final leader agrees with the real visit-leader and the spy's last leader
    assert obs.final_leader_move == visit_leader_move(root) == spy.calls[-1][2]
    assert 1 <= obs.stabilization_sim <= 200
    legal = set(root.priors.keys())
    assert set(obs.first_visit_sim).issubset(legal)
    assert all(1 <= s <= 200 for s in obs.first_visit_sim.values())
    # uniform priors + fpu=0 visit every root move once before revisiting, so all
    # three mass thresholds are crossed, in order
    xs = obs.mass_cross_sim
    assert xs[0.25] is not None and xs[0.50] is not None and xs[0.75] is not None
    assert xs[0.25] <= xs[0.50] <= xs[0.75]
    res = obs.result()
    assert res["selected_move_prior_rank"] >= 1
    assert 0.0 <= res["final_root_top_share"] <= 1.0
    # explored mass is the raw summed prior of first-visited children (mirrors
    # mcts.explored_policy_mass, which is unclamped -- policy_mass_fpu clamps
    # only at point of use); float summation of ~N uniform priors can land a
    # hair over 1.0, so tolerate epsilon rather than clamp away information.
    assert 0.0 <= res["explored_mass_at_stabilization"] <= 1.0 + 1e-6
