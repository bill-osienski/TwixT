"""`--replay-warmup-games`: the four mechanical checks, and nothing else.

1. default-off identity  — 0 (and the default) is a no-op: no self-play, no RNG
   consumption, no buffer change, and the call site in `train()` is guarded.
2. exactly N            — N games are requested and their positions land in the buffer.
3. no weight change     — warmup leaves every network parameter identical.
4. ordering             — the buffer is populated before the first optimizer step.

Plus RNG continuity: warmup consumes the *existing* master stream in place, so
iteration 0 continues from the advanced state and can never replay warmup games.

Real self-play is far too slow for a unit test, so these monkeypatch the module
attribute `trainer.run_parallel_selfplay` — a reach no production caller can make.
"""
import inspect
import random

import numpy as np
import pytest
from mlx.utils import tree_flatten

from scripts.GPU.alphazero import trainer
from scripts.GPU.alphazero.trainer import ReplayBuffer, run_replay_warmup, train
from scripts.GPU.alphazero.self_play import PositionRecord
from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator

POSITIONS_PER_GAME = 3


def _pos():
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move="red", legal_moves=[(0, 0), (1, 1)], visit_counts=[7, 3],
        outcome=1.0, active_size=24, ply=0, game_n_moves=10)


class _FakeSelfPlay:
    """Stands in for run_parallel_selfplay: records its call and fills the buffer.

    Draws from master_rng once per game, exactly as the real parallel path seeds
    its workers, so RNG-continuity assertions mean something.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, *, games_to_play, n_workers, master_rng, buffer, **kwargs):
        self.calls.append({"games_to_play": games_to_play, "n_workers": n_workers,
                           "master_rng": master_rng, "kwargs": kwargs})
        for _ in range(games_to_play):
            master_rng.randint(0, 2 ** 31)
            buffer.add_positions([_pos() for _ in range(POSITIONS_PER_GAME)])
        return [], [], {}


@pytest.fixture
def fake(monkeypatch):
    f = _FakeSelfPlay()
    monkeypatch.setattr(trainer, "run_parallel_selfplay", f)
    return f


def _warm(n, fake_unused=None, *, n_workers=4, buffer=None, master_rng=None, **kw):
    buffer = buffer if buffer is not None else ReplayBuffer(max_size=1000)
    master_rng = master_rng if master_rng is not None else random.Random(7)
    added = run_replay_warmup(n, n_workers=n_workers, buffer=buffer,
                              master_rng=master_rng, **kw)
    return added, buffer, master_rng


# ---------------------------------------------------------------- check 1
def test_zero_games_is_a_no_op(fake):
    rng = random.Random(7)
    state_before = rng.getstate()
    added, buffer, _ = _warm(0, buffer=ReplayBuffer(max_size=1000), master_rng=rng)

    assert added == 0
    assert len(buffer) == 0
    assert fake.calls == []                    # no self-play at all
    assert rng.getstate() == state_before      # stream untouched


def test_negative_games_is_also_a_no_op(fake):
    added, buffer, _ = _warm(-5)
    assert (added, len(buffer), fake.calls) == (0, 0, [])


def test_train_defaults_the_flag_off():
    assert inspect.signature(train).parameters["replay_warmup_games"].default == 0


def test_train_call_site_is_guarded_and_runs_once():
    """Default-off identity at the call site: nothing below it can run at 0."""
    src = inspect.getsource(train)
    assert "run_replay_warmup(" in src
    guard = "if iteration == start_iteration and replay_warmup_games > 0:"
    assert guard in src
    # the guard must precede the call it protects
    assert src.index(guard) < src.index("run_replay_warmup(\n")


# ---------------------------------------------------------------- check 2
def test_exactly_n_games_requested_and_stored(fake):
    added, buffer, _ = _warm(12)

    assert len(fake.calls) == 1
    assert fake.calls[0]["games_to_play"] == 12          # not 11, not 13
    assert added == 12 * POSITIONS_PER_GAME
    assert len(buffer) == 12 * POSITIONS_PER_GAME


def test_return_value_counts_only_what_warmup_added(fake):
    buffer = ReplayBuffer(max_size=1000)
    buffer.add_positions([_pos(), _pos()])               # pre-existing content
    added, buffer, _ = _warm(4, buffer=buffer)

    assert added == 4 * POSITIONS_PER_GAME               # excludes the 2 already there
    assert len(buffer) == 2 + 4 * POSITIONS_PER_GAME


# ---------------------------------------------------------------- check 3
def test_warmup_changes_no_weights(fake):
    net = create_network(hidden=64, n_blocks=2)
    before = {k: np.array(v) for k, v in tree_flatten(net.parameters())}

    _warm(6, evaluator=LocalGPUEvaluator(net))

    after = {k: np.array(v) for k, v in tree_flatten(net.parameters())}
    assert before.keys() == after.keys()
    for k in before:
        assert np.array_equal(before[k], after[k]), f"warmup moved {k}"


# ---------------------------------------------------------------- check 4
def test_buffer_is_populated_by_warmup(fake):
    _, buffer, _ = _warm(5)
    assert len(buffer) > 0


def test_warmup_precedes_the_first_optimizer_step_in_train():
    """Ordering, structurally: warmup is called before the training phase."""
    src = inspect.getsource(train)
    warmup_at = src.index("run_replay_warmup(\n")
    training_at = src.index('print(f"\\nTraining: {scaled_train_steps} steps')
    selfplay_at = src.index("# 1. Self-play (inference mode")
    assert warmup_at < selfplay_at < training_at


# ------------------------------------------------- pass-through wiring
def _call_site_kwargs():
    """Keyword names passed to run_replay_warmup at the train() call site."""
    src = inspect.getsource(train)
    start = src.index("run_replay_warmup(\n") + len("run_replay_warmup(")
    depth, end = 1, start
    while depth:
        end += 1
        depth += (src[end] == "(") - (src[end] == ")")
    return {line.strip().split("=")[0]
            for line in src[start:end].splitlines()
            if "=" in line and not line.strip().startswith("#")}


def test_call_site_kwargs_all_reach_a_real_parameter():
    """A typo here would only surface an hour into a real run, not in the suite."""
    warmup_params = set(inspect.signature(run_replay_warmup).parameters)
    selfplay_params = set(inspect.signature(trainer.run_parallel_selfplay).parameters)
    passed = _call_site_kwargs()

    assert passed, "failed to parse the call site"
    unknown = passed - warmup_params - selfplay_params
    assert not unknown, f"call site passes unknown kwargs: {sorted(unknown)}"


def test_call_site_supplies_every_required_selfplay_parameter():
    sig = inspect.signature(trainer.run_parallel_selfplay)
    required = {n for n, p in sig.parameters.items()
                if p.default is inspect.Parameter.empty
                and p.kind is not inspect.Parameter.VAR_KEYWORD}
    # run_replay_warmup supplies these three itself
    supplied = _call_site_kwargs() | {"games_to_play", "n_workers", "master_rng", "buffer"}

    assert not (required - supplied), f"unsupplied: {sorted(required - supplied)}"


# ---------------------------------------------------------------- RNG continuity
def test_warmup_consumes_the_existing_master_stream_in_place(fake):
    rng = random.Random(20260809)
    before = rng.getstate()
    _warm(3, master_rng=rng)

    assert fake.calls[0]["master_rng"] is rng     # same object, not a copy
    assert rng.getstate() != before               # advanced, not reset


def test_iteration_zero_continues_from_the_advanced_stream(fake):
    """Warmup then training must not draw the same seeds."""
    rng = random.Random(20260809)
    _warm(3, master_rng=rng)
    after_warmup = [rng.randint(0, 2 ** 31) for _ in range(3)]

    fresh = random.Random(20260809)
    replayed = [fresh.randint(0, 2 ** 31) for _ in range(3)]

    assert after_warmup != replayed               # no overlap with warmup's draws


def test_warmup_never_reseeds():
    assert "random.Random(" not in inspect.getsource(run_replay_warmup)


# ---------------------------------------------------------------- refusal
def test_sequential_worker_count_is_refused_not_silently_skipped(fake):
    with pytest.raises(ValueError, match="n_workers >= 2"):
        _warm(10, n_workers=1)
    assert fake.calls == []


def test_refusal_does_not_apply_when_warmup_is_off(fake):
    added, _, _ = _warm(0, n_workers=1)           # off beats the precondition
    assert added == 0
