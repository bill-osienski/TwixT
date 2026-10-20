"""Focused CPU checks for the root-only sequential-halving prototype (scripts/GPU/alphazero/root_sequential_halving.py):
simulation accounting, terminal results, legal moves, descent below the root, and the unchanged plies 0-19.
A stub evaluator stands in for the network; nothing loads a model."""
import math
import random

import numpy as np
import pytest

from scripts.GPU.alphazero import root_sequential_halving as RSH
from scripts.GPU.alphazero.game import TwixtState
from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig, encode_move


class Stub:
    """Priors proportional to weight(row, col); value(state) from the side to move."""

    def __init__(self, weight=lambda r, c: 1.0, value=lambda s: 0.0):
        self.weight, self.value, self.pending = weight, value, []

    def build_input_tensor(self, state):
        self.pending.append(state)
        return np.zeros((1, state.active_size, state.active_size), dtype=np.float32)

    def infer(self, boards, rows, cols, mask, active_size):
        states, self.pending = self.pending, []
        assert len(states) == boards.shape[0]
        w = np.array([[self.weight(int(r), int(c)) for r, c in zip(rr, cc)] for rr, cc in zip(rows, cols)]) * mask
        return (w / w.sum(axis=1, keepdims=True)).astype(np.float32), \
            np.array([self.value(s) for s in states], dtype=np.float32)


def played(plies):
    s = TwixtState(active_size=24, to_move="red", max_plies_limit=280)
    for i in range(plies):
        moves = s.legal_moves()
        s = s.apply_move(moves[(i * 97) % len(moves)])
    assert not s.is_terminal()
    return s


def restricted(state, k):
    moves = sorted(state.legal_moves())[:k]
    state.legal_moves = lambda: list(moves)
    return moves


def search(state, stub=None, cls=RSH.RootSequentialHalvingMCTS, sims=400, seed=7):
    mcts = cls(stub or Stub(), MCTSConfig(n_simulations=sims), rng=random.Random(seed))
    counts, value, root = mcts.search_with_root(state, add_noise=False)
    return mcts, counts, value, root


@pytest.mark.parametrize("m, expected", [
    (16, [97] * 2 + [43] * 2 + [18] * 4 + [6] * 8),
    (15, [100] * 2 + [43] * 2 + [18] * 4 + [6] * 7),
    (7, [120, 119] + [52] * 2 + [19] * 3),        # 135 left for the final pair: the remainder to the top-ranked
    (3, [167] * 2 + [66]),
    (1, [400]),
])
def test_accounting_ties_resolve_by_row_col(m, expected):
    state = played(RSH.FROM_PLY)
    moves = restricted(state, m)
    mcts, counts, _, root = search(state)
    assert root.visit_count == 400 == sum(c.visit_count for c in root.children.values())
    assert [counts[mv] for mv in moves] == expected          # uniform priors, zero values: (row, col) order
    assert mcts.last_choice == moves[0]
    assert mcts.last_rounds[0] == moves


def test_accounting_on_the_full_board():
    state = played(RSH.FROM_PLY + 3)
    mcts, counts, _, root = search(state, Stub(weight=lambda r, c: 1.0 + (r * 31 + c * 17) % 23))
    assert root.visit_count == 400 and sum(counts.values()) == 400
    assert sorted((v for v in counts.values() if v), reverse=True) == [97] * 2 + [43] * 2 + [18] * 4 + [6] * 8
    assert [len(r) for r in mcts.last_rounds] == [16, 8, 4, 2]
    assert all(set(b) <= set(a) for a, b in zip(mcts.last_rounds, mcts.last_rounds[1:]))
    prior = root.priors
    top16 = sorted(prior, key=lambda mid: (-prior[mid], divmod(mid, 24)))[:16]
    assert {encode_move(*mv) for mv in mcts.last_rounds[0]} == set(top16)
    assert mcts.last_choice in state.legal_moves() and mcts.last_choice in mcts.last_rounds[-1]


def test_value_overrides_a_lower_prior():
    state = played(RSH.FROM_PLY)
    moves = restricted(state, 16)
    root_player, target = state.to_move, moves[-1]               # the lowest-prior candidate

    def value(s):                                                # target on the board: good for the root player
        if s.pegs.get(target) is None:
            return 0.0
        return 0.8 if s.to_move == root_player else -0.8

    mcts, *_ = search(state, Stub(weight=lambda r, c: 1.0 if (r, c) == target else 2.0, value=value))
    assert mcts.last_choice == target


def test_score_formula_and_zero_prior():
    assert RSH.score(0.25, 0.0, 6) == pytest.approx(math.log(0.25) + 56 * 0.5)
    assert RSH.score(0.0, 1.0, 0) == pytest.approx(math.log(RSH.PRIOR_FLOOR) + 50.0)


def winning_position():
    """Red pegs (2k, 5+k), k = 0..11, a linked chain from row 0 to row 22; black far left. Red to move at ply 24
    wins with (23, 18)."""
    s = TwixtState(active_size=24, to_move="red", max_plies_limit=280)
    for k in range(12):
        s = s.apply_move((2 * k, 5 + k)).apply_move((k + 1, 1))
    assert s.to_move == "red" and s.ply == 24 and not s.is_terminal()
    end = s.apply_move((23, 18))
    assert end.is_terminal() and end.winner() == "red"
    return s


def test_terminal_win_is_backed_up_and_chosen():
    state = winning_position()
    moves = sorted(state.legal_moves())[:2] + [(23, 18)]
    state.legal_moves = lambda: list(moves)
    mcts, counts, _, root = search(state)
    child = root.children[encode_move(23, 18)]
    assert -child.q_value == 1.0 and child.visit_count == counts[(23, 18)] >= 66
    assert mcts.last_choice == (23, 18)


def test_known_limit_winning_move_outside_the_candidates_is_never_searched():
    state = winning_position()
    mcts, counts, *_ = search(state)                             # uniform priors: (23, 18) is not in the top 16
    assert counts[(23, 18)] == 0 and mcts.last_choice != (23, 18)


def test_root_selection_never_uses_puct_below_root_always_does():
    state = played(RSH.FROM_PLY)
    mcts = RSH.RootSequentialHalvingMCTS(Stub(), MCTSConfig(n_simulations=400), rng=random.Random(7))
    seen, real = [], mcts._select_child
    mcts._select_child = lambda node: (seen.append(node.parent is None), real(node))[1]
    mcts.search_with_root(state, add_noise=False)
    assert seen and not any(seen)


def test_plies_before_20_are_the_ordinary_search():
    for plies in (0, RSH.FROM_PLY - 1):
        a = search(played(plies), sims=120, cls=MCTS)
        b = search(played(plies), sims=120)
        assert a[1] == b[1] and a[2] == b[2] and b[0].last_choice is None and b[0].last_rounds is None


def test_refusals():
    with pytest.raises(ValueError, match="root noise"):
        RSH.RootSequentialHalvingMCTS(Stub(), MCTSConfig(n_simulations=400)).search_with_root(
            played(RSH.FROM_PLY), add_noise=True)
    small = TwixtState(active_size=12, to_move="red", max_plies_limit=280)
    for i in range(RSH.FROM_PLY):
        small = small.apply_move(small.legal_moves()[(i * 7) % len(small.legal_moves())])
    with pytest.raises(ValueError, match="24x24 only"):
        RSH.RootSequentialHalvingMCTS(Stub(), MCTSConfig(n_simulations=400)).search_with_root(small, add_noise=False)
    with pytest.raises(ValueError, match="cannot visit"):
        RSH.RootSequentialHalvingMCTS(Stub(), MCTSConfig(n_simulations=40)).search_with_root(
            played(RSH.FROM_PLY), add_noise=False)
