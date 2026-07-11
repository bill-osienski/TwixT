"""Tests for the complete-state canonical hash (`canonical_state_sha1`).

Frozen design ref: docs/superpowers/specs/2026-07-10-context-relative-fpu-policy-mass-design.md
Plan Task 4: complete-state canonical hash used by later tasks (corpus
disjointness, controls join key) to dedupe/join positions on FUTURE-PLAY
equivalence: equal hash <=> identical side-to-move, legal moves, terminal
result, and NN input tensor; transpositions that reach the same state hash;
and any change to a future-relevant field changes the hash.
"""
import dataclasses
import numpy as np
from scripts.GPU.alphazero.game.twixt_state import TwixtState
from scripts.GPU.alphazero.fpu_state_hash import canonical_state_sha1


def _play(moves, active_size=10, max_plies=None):
    s = TwixtState(active_size=active_size, to_move="red", max_plies_limit=max_plies)
    for m in moves: s = s.apply_move(m)
    return s


def test_equal_hash_implies_equal_behavior_and_nninput():
    a, b = _play([(3, 3), (5, 5), (4, 6)]), _play([(3, 3), (5, 5), (4, 6)])
    assert canonical_state_sha1(a) == canonical_state_sha1(b)
    assert a.to_move == b.to_move and set(a.legal_moves()) == set(b.legal_moves())
    assert a.is_terminal() == b.is_terminal() and a.winner() == b.winner()
    assert np.array_equal(a.to_tensor(), b.to_tensor())


def test_transposition_same_state_same_hash():
    # FIX 6: reorder ONLY within each player's turns (alternation preserves ownership).
    # red gets {(2,2),(2,7)}, black gets {(7,7),(7,2)} in BOTH orders; interior, no bridges.
    a = _play([(2, 2), (7, 7), (2, 7), (7, 2)])       # red A, black B, red C, black D
    b = _play([(2, 7), (7, 2), (2, 2), (7, 7)])       # red C, black D, red A, black B
    assert canonical_state_sha1(a) == canonical_state_sha1(b)
    assert a.pegs == b.pegs and a.bridges == b.bridges
    assert a.to_move == b.to_move and set(a.legal_moves()) == set(b.legal_moves())
    assert a.is_terminal() == b.is_terminal() and np.array_equal(a.to_tensor(), b.to_tensor())


def test_each_future_relevant_field_changes_hash():
    base = _play([(3, 3), (5, 5)])
    assert canonical_state_sha1(base) != canonical_state_sha1(_play([(3, 3), (6, 6)]))            # pegs
    assert canonical_state_sha1(base) != canonical_state_sha1(_play([(3, 3), (5, 5)], active_size=12))
    assert canonical_state_sha1(base) != canonical_state_sha1(_play([(3, 3), (5, 5)], max_plies=40))
    flip = dataclasses.replace(base, to_move=("black" if base.to_move == "red" else "red"))
    assert canonical_state_sha1(base) != canonical_state_sha1(flip)                               # side
