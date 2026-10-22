"""Real-search counter probe (plan R3, spec 8): the real MCTS.search_from_root with the repo's deterministic CPU
FakeEvaluator (uniform priors, value 0; no network). Three 24x24 searches of 16 simulations each (48 total):
fresh root, reused root with inherited visits, fixed near-terminal position. Observations are printed BEFORE the
contract assertions, so a mismatch is reported with its numbers."""
import json
import math
import random

from scripts.GPU.alphazero.game.twixt_state import TwixtState
from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig, MCTSNode, decode_move
from tests.fpu_search_fixture import FakeEvaluator
from tests.test_training_replay_capture import BLACK_FAR, RED_CHAIN

SIMS = 16


def _children(root):
    return {decode_move(k): c.visit_count for k, c in root.children.items()}


FLUSHES = ("_flush_full", "_flush_tail", "_flush_stall")


def _terminal_children(root):
    """Root children whose position is terminal (a win or the cap), with their completed visit counts."""
    return {decode_move(k): c.visit_count for k, c in root.children.items() if c.state.is_terminal()}


def _search(case, mcts, root, ply):
    rb, cb = root.visit_count, _children(root)
    forced0 = mcts._closeout_td1_forced_sims_total
    flush0 = {f: getattr(mcts, f) for f in FLUSHES}
    counts, value, out = mcts.search_from_root(root, add_noise=True, ply=ply, gc_state_full=None)
    ra, ca = out.visit_count, _children(out)
    flush = {f.lstrip("_"): getattr(mcts, f) - flush0[f] for f in FLUSHES}
    term = _terminal_children(out)
    obs = {"case": case, "same_root_object": out is root, "root_before": rb, "root_after": ra,
           "root_delta": ra - rb, "child_sum_before": sum(cb.values()), "child_sum_after": sum(ca.values()),
           "child_sum_delta": sum(ca.values()) - sum(cb.values()), "children_before": len(cb),
           "children_after": len(ca), "forced_delta": mcts._closeout_td1_forced_sims_total - forced0,
           "root_value": float(value), "nonzero_targets": sum(1 for v in counts.values() if v),
           "legal": len(out.state.legal_moves()), "flush_deltas": flush,
           "inherited_child_mass": sum(cb.values()),
           "terminal_children": {f"{r},{c}": v for (r, c), v in sorted(term.items())},
           "terminal_child_visits": sum(term.values()),
           "unexercised": [k for k, v in flush.items() if v == 0]
                          + ([] if sum(term.values()) else ["terminal_child_visits"])
                          + ([] if sum(cb.values()) else ["inherited_child_mass"])}
    print("search-probe", json.dumps(obs))
    return counts, out, cb, ca, obs


def _check_target(counts, out, cb, ca, obs):
    legal = out.state.legal_moves()
    assert list(counts) == legal, obs                                     # full legal order, in order
    assert all(type(v) is int and v >= 0 for v in counts.values()), obs
    assert counts == {m: ca.get(m, 0) for m in legal}, obs                # target == completed child counts
    assert set(ca) <= set(legal), obs
    assert all(ca.get(m, 0) >= v for m, v in cb.items()), obs             # no child count decreased
    assert obs["same_root_object"] and math.isfinite(obs["root_value"]) and obs["forced_delta"] == 0, obs


def _check_budget(obs):
    # The capture contract under test (R3): every simulation completes and is backed up once through the root
    # and once through exactly one root child. A failure here is the STOP condition.
    assert obs["root_delta"] == SIMS, obs
    assert obs["child_sum_delta"] == SIMS, obs


def _chain_state():
    s = TwixtState(active_size=24, to_move="red", max_plies_limit=280)
    for a, b in zip(RED_CHAIN, BLACK_FAR):
        s = s.apply_move(a).apply_move(b)
    assert [x for x in s.legal_moves() if s.apply_move(x).winner() == "red"] == [(23, 4), (23, 8)]
    return s


def test_real_search_counters_on_three_fixed_roots():
    mcts = MCTS(FakeEvaluator(), MCTSConfig(n_simulations=SIMS), random.Random(3))
    searches = 0

    fresh = MCTSNode(state=TwixtState(active_size=24, to_move="red", max_plies_limit=280))
    counts, out, cb, ca, obs = _search("fresh", mcts, fresh, 0); searches += 1
    _check_target(counts, out, cb, ca, obs)
    assert obs["root_before"] == 0 and obs["child_sum_before"] == 0, obs
    _check_budget(obs)

    legal = out.state.legal_moves()
    move = max(legal, key=lambda m: (ca.get(m, 0), -legal.index(m)))      # most visited; ties by legal order
    reused = mcts.advance_root(out, move)
    assert reused.visit_count == ca[move] and reused.visit_count >= 1     # tree reuse carries the visits
    counts, out, cb, ca, obs = _search("reused", mcts, reused, 1); searches += 1
    _check_target(counts, out, cb, ca, obs)
    _check_budget(obs)

    counts, out, cb, ca, obs = _search("near_terminal", mcts, MCTSNode(state=_chain_state()), 24); searches += 1
    _check_target(counts, out, cb, ca, obs)
    _check_budget(obs)

    assert searches * SIMS == 48
