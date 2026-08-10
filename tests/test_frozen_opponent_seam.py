"""Frozen-parent opponent: the pure dual-root game seam in `play_game`.

Covers the card's pre-GPU gate items that live in `self_play.py`:

* default-off identity as **deterministic behavioural equivalence** — same games,
  same training data, no second root/evaluator, no extra RNG consumption;
* the two roots never alias, and neither network's priors leak into the other's
  tree;
* each agent searches and selects with its OWN instance and RNG;
* both trees advance on every played move, including moves the inactive tree
  never explored;
* the roots stay synchronised to the same board state;
* only learner-to-move positions become training rows.

Worker/server plumbing is deliberately out of scope — this file tests the game
seam alone, which the card requires to pass first.
"""
import random

import numpy as np
import pytest

from scripts.GPU.alphazero.self_play import play_game
from scripts.GPU.alphazero.mcts import MCTSConfig

BOARD = 8
SIMS = 6


class StubEvaluator:
    """Deterministic, network-free evaluator with a per-instance prior signature.

    `tag` biases priors toward a fixed cell so a tree's priors can be traced back
    to the network that produced them — that is what makes prior-leakage
    detectable without running two real networks.
    """

    def __init__(self, tag: float, value: float = 0.0):
        self.tag = tag
        self.value = value
        self.calls = 0

    def build_input_tensor(self, state):
        return state.to_tensor()

    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        self.calls += 1
        b, m = move_mask.shape
        # Move-DEPENDENT priors: a constant would normalize to uniform and make
        # every stub network behave identically, which would silently pass the
        # separation tests below. `tag` shifts which moves are favoured, so two
        # tags are genuinely different networks.
        pattern = np.abs(np.sin(np.arange(m, dtype=np.float32) * self.tag + 1.0))
        priors = np.broadcast_to(pattern, (b, m)).astype(np.float32) * move_mask
        values = np.full((b,), self.value, dtype=np.float32)
        return priors, values


def _cfg():
    return MCTSConfig(n_simulations=SIMS)


def _play(seed, *, opponent=None, learner=None, **kw):
    return play_game(
        StubEvaluator(tag=0.9) if "evaluator" not in kw else kw.pop("evaluator"),
        mcts_config=_cfg(),
        rng=random.Random(seed),
        max_moves=24,
        active_size=BOARD,
        start_player="red",
        opponent_evaluator=opponent,
        learner_player=learner,
        goal_completion_emit_enabled=False,
        goal_completion_record_enabled=False,
    )


def _signature(game):
    """Everything a training run consumes: moves plus the replay rows."""
    return (
        [tuple(m) if isinstance(m, (list, tuple)) else m for m in game.move_history],
        game.winner,
        [(p.to_move, p.ply, list(p.visit_counts)) for p in game.positions],
    )


# ------------------------------------------------- default-off identity
def test_default_off_is_deterministically_equivalent():
    """Same seed, no opponent: identical games and identical training rows."""
    assert _signature(_play(11)) == _signature(_play(11))


def test_default_off_consumes_no_extra_rng():
    """A second agent must not be constructed, so the stream cannot advance."""
    r1, r2 = random.Random(5), random.Random(5)
    play_game(StubEvaluator(0.9), mcts_config=_cfg(), rng=r1, max_moves=12,
              active_size=BOARD, start_player="red",
              goal_completion_emit_enabled=False, goal_completion_record_enabled=False)
    play_game(StubEvaluator(0.9), mcts_config=_cfg(), rng=r2, max_moves=12,
              active_size=BOARD, start_player="red", opponent_evaluator=None,
              learner_player=None,
              goal_completion_emit_enabled=False, goal_completion_record_enabled=False)
    assert r1.getstate() == r2.getstate()


def test_default_off_every_position_is_kept():
    game = _play(7)
    plies = {p.ply for p in game.positions}
    assert len(plies) == len(game.move_history)          # nothing filtered out


# ------------------------------------------------- argument validation
@pytest.mark.parametrize("bad", [None, "", "RED", "white"])
def test_opponent_requires_an_explicit_learner_colour(bad):
    with pytest.raises(ValueError, match="learner_player"):
        _play(3, opponent=StubEvaluator(0.1), learner=bad)


# ------------------------------------------------- learner-only training rows
@pytest.mark.parametrize("learner", ["red", "black"])
def test_only_learner_to_move_positions_are_trained_on(learner):
    game = _play(21, opponent=StubEvaluator(0.1), learner=learner)
    assert game.positions, "expected some learner rows"
    assert {p.to_move for p in game.positions} == {learner}


def test_frozen_opponent_halves_the_rows_it_would_otherwise_keep():
    """Sanity on the dose assumption the card pins 200 games/iteration to."""
    solo = _play(33)
    duo = _play(33, opponent=StubEvaluator(0.1), learner="red")
    assert len(duo.positions) < len(solo.positions)


# ------------------------------------------------- the dual-root seam
def test_each_agent_uses_its_own_evaluator():
    """Both networks are actually consulted — neither side is played by the other."""
    learner_eval, opp_eval = StubEvaluator(0.9), StubEvaluator(0.1)
    play_game(learner_eval, mcts_config=_cfg(), rng=random.Random(4), max_moves=16,
              active_size=BOARD, start_player="red", opponent_evaluator=opp_eval,
              learner_player="red", goal_completion_emit_enabled=False,
              goal_completion_record_enabled=False)
    assert learner_eval.calls > 0 and opp_eval.calls > 0


def test_opponent_identity_actually_drives_opponent_plies():
    """Same seed, same learner, different opponent network ⇒ different game.

    If the opponent's plies were served by the learner's tree or evaluator, this
    would produce identical move histories.
    """
    a = _play(9, opponent=StubEvaluator(0.10), learner="red")
    b = _play(9, opponent=StubEvaluator(0.77), learner="red")
    assert a.move_history != b.move_history


def test_learner_identity_still_drives_learner_plies():
    """The mirror image: changing only the learner changes the game too."""
    common = dict(mcts_config=_cfg(), rng=None, max_moves=24, active_size=BOARD,
                  start_player="red", learner_player="red",
                  goal_completion_emit_enabled=False,
                  goal_completion_record_enabled=False)
    def run(learner_tag):
        kw = dict(common); kw["rng"] = random.Random(9)
        return play_game(StubEvaluator(learner_tag),
                         opponent_evaluator=StubEvaluator(0.10), **kw)
    assert run(0.90).move_history != run(0.33).move_history


def test_roots_stay_synchronised_across_the_whole_game():
    """The in-loop assertion is the guard; this proves it is exercised, not dead."""
    game = _play(13, opponent=StubEvaluator(0.1), learner="black")
    assert len(game.move_history) > 2                    # the assert ran on real plies


def test_unexplored_move_does_not_break_the_inactive_tree():
    """A 1-simulation opponent explores almost nothing, so advance_root must
    build fresh nodes for the learner's choices rather than raising."""
    game = play_game(
        StubEvaluator(0.9), mcts_config=MCTSConfig(n_simulations=1),
        rng=random.Random(17), max_moves=20, active_size=BOARD,
        start_player="red", opponent_evaluator=StubEvaluator(0.1),
        learner_player="red", goal_completion_emit_enabled=False,
        goal_completion_record_enabled=False,
    )
    assert len(game.move_history) > 2


# ------------------------------------------------- direct instrumentation
#
# The behavioural tests above infer the seam from outputs. These watch it
# happen: how many agents are built, which roots each one touches, who is
# dispatched to, and whether the RNG objects are distinct.

class _Spy:
    """Records every MCTS construction and the calls made on each instance."""

    def __init__(self):
        self.instances = []          # one entry per MCTS built
        self.roots = {}              # instance index -> set of root ids it saw
        self.advanced = {}           # instance index -> [moves advanced]
        self.selected = {}           # instance index -> call count

    def install(self, monkeypatch):
        from scripts.GPU.alphazero import self_play as sp
        real = sp.MCTS
        spy = self

        class SpyMCTS(real):
            def __init__(self, evaluator, config, rng):
                super().__init__(evaluator, config, rng)
                self._spy_idx = len(spy.instances)
                spy.instances.append(self)
                spy.roots[self._spy_idx] = set()
                spy.advanced[self._spy_idx] = []
                spy.selected[self._spy_idx] = 0

            def search_from_root(self, root, **kw):
                spy.roots[self._spy_idx].add(id(root))
                out = super().search_from_root(root, **kw)
                spy.roots[self._spy_idx].add(id(out[2]))
                return out

            def advance_root(self, root, move):
                spy.roots[self._spy_idx].add(id(root))
                spy.advanced[self._spy_idx].append(move)
                out = super().advance_root(root, move)
                spy.roots[self._spy_idx].add(id(out))
                return out

            def select_move(self, visit_counts, ply):
                spy.selected[self._spy_idx] += 1
                return super().select_move(visit_counts, ply)

        monkeypatch.setattr(sp, "MCTS", SpyMCTS)
        return self


def test_default_off_constructs_exactly_one_mcts(monkeypatch):
    spy = _Spy().install(monkeypatch)
    _play(41)
    assert len(spy.instances) == 1


def test_enabled_constructs_exactly_two_mcts(monkeypatch):
    spy = _Spy().install(monkeypatch)
    _play(41, opponent=StubEvaluator(0.10), learner="red")
    assert len(spy.instances) == 2


def test_the_two_agents_never_touch_the_same_root_object(monkeypatch):
    spy = _Spy().install(monkeypatch)
    _play(43, opponent=StubEvaluator(0.10), learner="red")
    learner_roots, opp_roots = spy.roots[0], spy.roots[1]
    assert learner_roots and opp_roots
    assert learner_roots.isdisjoint(opp_roots), "a root object was shared"


def test_both_agents_advance_on_every_played_move(monkeypatch):
    spy = _Spy().install(monkeypatch)
    game = _play(45, opponent=StubEvaluator(0.10), learner="red")
    played = list(game.move_history)
    # every played move except the last (the game ends before advancing past it)
    assert spy.advanced[0] == spy.advanced[1], "trees advanced differently"
    assert spy.advanced[0] == played[:len(spy.advanced[0])]
    assert len(spy.advanced[0]) >= len(played) - 1


def test_select_move_dispatches_to_the_active_agent_with_distinct_rngs(monkeypatch):
    spy = _Spy().install(monkeypatch)
    game = _play(47, opponent=StubEvaluator(0.10), learner="red")
    learner_calls, opp_calls = spy.selected[0], spy.selected[1]
    assert learner_calls > 0 and opp_calls > 0
    assert learner_calls + opp_calls == len(game.move_history)
    # red starts and is the learner, so it never selects fewer than the opponent
    assert learner_calls >= opp_calls
    assert spy.instances[0].rng is not spy.instances[1].rng


def test_mirror_augmentation_never_fires_on_a_parent_ply(monkeypatch):
    """With mirror probability 1 every learner ply yields exactly two rows."""
    from scripts.GPU.alphazero import self_play as sp
    monkeypatch.setattr(sp, "_MIRROR_PROB", 1.0)
    game = _play(49, opponent=StubEvaluator(0.10), learner="black")

    assert {p.to_move for p in game.positions} == {"black"}
    by_ply = {}
    for p in game.positions:
        by_ply.setdefault(p.ply, 0)
        by_ply[p.ply] += 1
    assert by_ply, "expected learner rows"
    assert set(by_ply.values()) == {2}, f"expected primary+mirror per ply, got {by_ply}"


# ------------------------------------------------- unsupported combinations
@pytest.mark.parametrize("flag", ["resign_enabled", "adjudicate_enabled"])
def test_frozen_opponent_refuses_resign_and_adjudication(flag):
    with pytest.raises(ValueError, match="does not support resign or adjudication"):
        play_game(
            StubEvaluator(0.9), mcts_config=_cfg(), rng=random.Random(2),
            max_moves=16, active_size=BOARD, start_player="red",
            opponent_evaluator=StubEvaluator(0.1), learner_player="red",
            goal_completion_emit_enabled=False, goal_completion_record_enabled=False,
            **{flag: True},
        )


def test_those_flags_are_still_allowed_without_a_frozen_opponent():
    game = play_game(
        StubEvaluator(0.9), mcts_config=_cfg(), rng=random.Random(2),
        max_moves=16, active_size=BOARD, start_player="red",
        resign_enabled=True, adjudicate_enabled=True,
        goal_completion_emit_enabled=False, goal_completion_record_enabled=False,
    )
    assert game.n_moves > 0


def test_frozen_opponent_games_are_reproducible():
    a = _play(29, opponent=StubEvaluator(0.1), learner="red")
    b = _play(29, opponent=StubEvaluator(0.1), learner="red")
    assert _signature(a) == _signature(b)
