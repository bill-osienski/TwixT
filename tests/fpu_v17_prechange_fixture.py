"""v17 Task 1 pre-change identity goldens, captured from UNMODIFIED `mcts.py`.

Frozen design ref:
`docs/superpowers/specs/2026-07-24-v17-baseline-preserving-policy-mass-fpu-design.md`
(SHA-1 `944f358c0e3ef66503d2cbb56e31dabd145bafc2`) §2.2 "Exact-zero identity is
structural", whose required proofs are:

  (i)  synthetic-tree selected moves and selection traces, and
  (ii) fixed CPU-search visit counts, root value, tree signature, and callback
       sequence.

`cpu_search_signature` covers (ii) and `synthetic_selection_trace` covers (i);
`prechange_goldens` composes both into the single golden JSON.

Every search here runs at the §2.4 frozen batching triple
`(eval_batch_size, stall_flush_sims, pending_virtual_visits) = (14, 48, 8)`.
`MCTSConfig.stall_flush_sims` defaults to `16`, so `48` is set EXPLICITLY --
never inherited. (On this small fixture 16 and 48 happen to produce the same
search; the triple is pinned because §2.4 makes batching part of the v17
mechanism, not because this fixture discriminates it.)

Task 3 re-runs these against `tests/golden/fpu_v17_prechange_search.json`:

  * new field `None`  -> `prechange_goldens()` must reproduce the whole golden.
  * new field `0.0`   -> `cpu_search_signature(fpu_shipped_policy_mass_reduction=0.0)`
    must reproduce the golden's `cpu_search` block. The synthetic sweep is not
    re-runnable at `0.0` because design §2.1 rejects a non-`None` new field
    unless `fpu_value == 0.0`, and the sweep deliberately varies `fpu_value`.

This module reads only existing helpers; it defines no MCTS behaviour of its
own. It is deliberately coupled to `test_fpu_policy_mass_rule`'s pinned
synthetic tree -- if that tree changes, this golden SHOULD break, because the
identity basis changed.
"""
import dataclasses
import random

from tests.fpu_search_fixture import run_search
from tests.test_fpu_policy_mass_rule import _synthetic_root_for_policy_mass

from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig, visit_leader_move

# design §2.4 -- frozen for every v17 search in every stage
BATCHING = {"eval_batch_size": 14, "stall_flush_sims": 48, "pending_virtual_visits": 8}

# `fpu_value` points bracketing the pinned synthetic tree's X/Y decision
# boundary. From that tree's docstring: score_X = 0.10149255578531499 and
# u_Y = 0.15074813431681333, so Y loses to X once fpu_value drops below
# 0.10149255578531499 - 0.15074813431681333 = -0.04925557853149834. Sweeping
# `fpu_value` walks the REAL boundary using the real `_select_child`, so the
# trace needs no reimplementation of the PUCT arithmetic.
FPU_TRACE_POINTS = (-0.50, -0.20, -0.10, -0.06, -0.05, -0.0492, -0.02, 0.0, 0.10)


class _CallbackSpy:
    """Records the `on_root_simulation` sequence. Same shape as
    `tests/test_fpu_trace_observer.py::_Spy`, but stores JSON-ready lists."""

    def __init__(self):
        self.calls = []

    def on_root_simulation(self, count, root, move, leader):
        self.calls.append([int(count),
                           None if move is None else int(move),
                           None if leader is None else int(leader)])


def v17_config(**overrides):
    """`MCTSConfig` at the frozen §2.4 batching triple, plus `overrides`."""
    return dataclasses.replace(MCTSConfig(), **BATCHING, **overrides)


def _stub_value_fn():
    def f(state):
        return {}, 0.0
    return f


def synthetic_selection_trace(**config_overrides):
    """§2.2 proof (i): the shipped `fpu_value` -> selected-move map on the
    pinned synthetic tree, plus the formula inputs that determine it."""
    trace = []
    for fpu in FPU_TRACE_POINTS:
        root, _X, _Y, _Z = _synthetic_root_for_policy_mass()   # fresh tree per point
        cfg = v17_config(n_simulations=1, c_puct=1.5, fpu_value=fpu,
                         **config_overrides)
        chosen, _child = MCTS(_stub_value_fn(), cfg, random.Random(0))._select_child(root)
        trace.append([float(fpu).hex(), int(chosen)])
    return trace


def cpu_search_signature(**config_overrides):
    """§2.2 proof (ii). Runs the fixed CPU search twice -- observer OFF (the
    signature proper, per plan Task 1 step 2) and observer ON (for the callback
    sequence) -- and asserts the observer does not perturb the search."""
    cfg = v17_config(**config_overrides)
    off, root, _m = run_search(config=cfg)
    spy = _CallbackSpy()
    on, _root_on, _m2 = run_search(config=cfg, observer=spy)
    assert off == on, "observer changed the search; it must be read-only"
    return {
        "batching": [cfg.eval_batch_size, cfg.stall_flush_sims,
                     cfg.pending_virtual_visits],
        "search": off,
        "selected_move": visit_leader_move(root),
        # tree signature: per-child completed visits + bit-exact Q
        "tree": [[int(mid), int(ch.visit_count), float(ch.q_value).hex()]
                 for mid, ch in sorted(root.children.items())],
        "callbacks": spy.calls,
    }


def prechange_goldens(**config_overrides):
    """The complete Task 1 golden payload."""
    return {
        "cpu_search": cpu_search_signature(**config_overrides),
        "synthetic_selection_trace": synthetic_selection_trace(**config_overrides),
    }
