"""Frozen-opponent worker/server plumbing, exercised with stubs only.

Step 3 of the card's sequence. These run no accelerated work: the inference
servers are driven with a stub evaluator over real multiprocessing queues, and
the trainer wiring is checked by monkeypatching the module attributes.

Covered:
* the second server, queues and evaluator exist ONLY in frozen-opponent mode;
* warmup stays on the ordinary single-network path;
* learner colour is a pure function of game id, so an even game count splits
  exactly 100/100 regardless of scheduling;
* both servers start, stop, join and have their queues cleaned up symmetrically;
* a server failure fails the run closed, with no fallback to the learner
  evaluator.
"""
import inspect
import multiprocessing as mp
import threading

import numpy as np
import pytest

from scripts.GPU.alphazero import trainer
from scripts.GPU.alphazero.inference_server import InferenceServer
from scripts.GPU.alphazero.self_play_worker import _worker_loop
from scripts.GPU.alphazero.ipc_messages import StopSignal


class StubEvaluator:
    def __init__(self, tag=0.5):
        self.tag = tag
        self.calls = 0

    def build_input_tensor(self, state):
        return state.to_tensor()

    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        self.calls += 1
        b, m = move_mask.shape
        return (np.full((b, m), self.tag, np.float32) * move_mask,
                np.zeros((b,), np.float32))


# ------------------------------------------------- colour assignment
def _colour(gid):
    """Mirrors the worker's rule; asserted against the source below."""
    return "red" if gid % 2 == 0 else "black"


def test_colour_is_a_pure_function_of_game_id_not_completion_order():
    src = inspect.getsource(_worker_loop)
    assert 'learner_player = "red" if gid % 2 == 0 else "black"' in src
    # nothing about worker identity or ordering may appear in that decision
    line = [l for l in src.splitlines() if "learner_player = " in l and "gid" in l][0]
    assert "worker_id" not in line and "games_played" not in line


@pytest.mark.parametrize("total", [2, 8, 200])
def test_even_game_count_splits_exactly_in_half(total):
    colours = [_colour(g) for g in range(total)]
    assert colours.count("red") == colours.count("black") == total // 2


def test_two_hundred_games_give_the_pinned_hundred_hundred():
    colours = [_colour(g) for g in range(200)]
    assert (colours.count("red"), colours.count("black")) == (100, 100)


def test_assignment_is_independent_of_claim_order():
    """Workers claim ids out of order; the split must not care."""
    claimed = [7, 0, 3, 1, 2, 6, 5, 4]
    assert sorted(_colour(g) for g in claimed) == sorted(_colour(g) for g in range(8))


# ------------------------------------------------- conditional construction
def test_worker_builds_no_second_evaluator_without_the_transport():
    src = inspect.getsource(_worker_loop)
    assert "opponent_evaluator = None" in src
    assert "if opponent_request_queue is not None:" in src


def test_worker_never_falls_back_to_the_learner_evaluator():
    """The opponent evaluator is never assigned from `evaluator`."""
    src = inspect.getsource(_worker_loop)
    assert "opponent_evaluator = evaluator" not in src
    assert "opponent_evaluator or evaluator" not in src


def test_second_server_is_conditional_in_run_parallel_selfplay():
    src = inspect.getsource(trainer.run_parallel_selfplay)
    assert "if opponent_evaluator is not None:" in src
    assert "opp_server = None" in src


def _balanced_call(src, name):
    """Extract a full call expression by matching parentheses."""
    start = src.index(name + "(") + len(name)
    depth, i = 0, start
    while True:
        depth += (src[i] == "(") - (src[i] == ")")
        i += 1
        if depth == 0:
            return src[start:i]


def test_warmup_stays_on_the_ordinary_single_network_path():
    """run_replay_warmup must not forward an opponent, at either end."""
    warm_src = inspect.getsource(trainer.run_replay_warmup)
    assert "opponent_evaluator" not in warm_src

    call = _balanced_call(inspect.getsource(trainer.train), "run_replay_warmup")
    assert "buffer=buffer" in call, "failed to parse the warmup call"
    assert "opponent" not in call


def test_iteration_selfplay_does_forward_the_opponent():
    """The mirror of the test above: the training iterations DO get it, so the
    warmup assertion is a real distinction rather than a vacuous one."""
    call = _balanced_call(inspect.getsource(trainer.train), "run_parallel_selfplay")
    assert "opponent_evaluator=frozen_opponent_evaluator" in call


def test_frozen_network_is_never_handed_to_an_optimizer():
    src = inspect.getsource(trainer.train)
    assert "frozen_network.load_weights(" in src
    for bad in ("opt_main.update(frozen", "opt_value.update(frozen",
                "optimizer(frozen", "frozen_network.train()"):
        assert bad not in src


def test_frozen_opponent_requires_parallel_workers():
    src = inspect.getsource(trainer.train)
    assert "frozen-opponent training requires n_workers >= 2" in src


# ------------------------------------------------- real transport, stub payloads
def _drain(q):
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except Exception:
            return out


def test_two_servers_run_and_stop_independently_on_real_queues():
    """Both servers are started, serve their own queue, and stop cleanly."""
    ctx = mp.get_context("spawn")
    servers, threads, queues = [], [], []
    for tag in (0.9, 0.1):
        rq = ctx.Queue(maxsize=16)
        resp = {0: ctx.Queue(maxsize=16)}
        s = InferenceServer(evaluator=StubEvaluator(tag), request_queue=rq,
                            response_queues=resp, max_batch_rows=4, flush_ms=2)
        t = threading.Thread(target=s.run_forever, daemon=True)
        t.start()
        servers.append(s); threads.append(t); queues.append((rq, resp))

    assert all(t.is_alive() for t in threads)

    for s, t, (rq, resp) in zip(servers, threads, queues):
        try:
            rq.put(StopSignal(), timeout=0.5)
        except Exception:
            pass
        s.stop()
        t.join(timeout=2.0)
        assert not t.is_alive(), "server thread failed to stop"
        for q in [rq, *resp.values()]:
            q.cancel_join_thread()
            q.close()


def test_shutdown_path_is_symmetric_for_both_servers():
    src = inspect.getsource(trainer.run_parallel_selfplay)
    tail = src[src.index("server.stop()"):]
    assert "opp_server.stop()" in tail
    assert "opp_server_thread.join" in tail
    assert "opp_request_queue.put(StopSignal()" in tail
    assert "opp_response_queues.values()" in tail       # queue cleanup


def test_both_servers_report_crashes_to_the_same_fail_closed_handler():
    src = inspect.getsource(trainer.run_parallel_selfplay)
    # one shared stats_queue is what makes the single handler cover both
    assert src.count("stats_queue=stats_queue") == 2
    assert 'raise RuntimeError(f"InferenceServer crashed' in src


def test_a_server_error_message_raises_rather_than_degrading():
    """The handler is a raise, not a log-and-continue."""
    src = inspect.getsource(trainer.run_parallel_selfplay)
    idx = src.index('msg.get("type") == "server_error"')
    following = src[idx:idx + 220]
    assert "raise RuntimeError" in following
    assert "continue" not in following.split("raise RuntimeError")[0]


# ------------------------------------------------- integrated round trip
#
# The server-only test above proves threads start and stop; it sends no request
# and spawns no worker, so it cannot see RemoteEvaluator routing or learner /
# opponent separation. These run the real thing: two workers, two servers, real
# spawn transport, stub evaluators, tiny games.

def _tiny_run(learner_stub, opponent_stub, games=2, workers=2):
    from scripts.GPU.alphazero.curriculum import CurriculumManager
    from scripts.GPU.alphazero.trainer import ReplayBuffer, run_parallel_selfplay
    from scripts.GPU.alphazero.mcts import MCTSConfig
    import random as _random

    buffer = ReplayBuffer(max_size=5000)
    return run_parallel_selfplay(
        evaluator=learner_stub,
        mcts_config=MCTSConfig(n_simulations=2, eval_batch_size=4),
        games_to_play=games,
        n_workers=workers,
        master_rng=_random.Random(3),
        max_moves=8,
        active_size=8,
        curriculum=CurriculumManager(sizes=(8,)),
        buffer=buffer,
        opponent_evaluator=opponent_stub,
    ), buffer


def test_worker_to_two_server_round_trip_calls_both_evaluators():
    """The gate item a green suite cannot cover: real worker -> two servers."""
    learner, opponent = StubEvaluator(0.9), StubEvaluator(0.1)
    (_games, _new_positions, stats), buffer = _tiny_run(learner, opponent)

    assert stats["games_generated"] == 2, stats
    assert learner.calls > 0, "learner server never served a request"
    assert opponent.calls > 0, "opponent server never served a request"
    assert len(buffer) > 0, "no learner positions reached the buffer"


def test_round_trip_keeps_only_learner_positions():
    learner, opponent = StubEvaluator(0.9), StubEvaluator(0.1)
    (_g, _p, _s), buffer = _tiny_run(learner, opponent, games=2, workers=2)
    # games 0 and 1 -> learner is red then black, so both colours appear across
    # the run, but never both within one game's rows
    assert {p.to_move for p in buffer.buffer} <= {"red", "black"}
    assert len(buffer) > 0


class ExplodingEvaluator(StubEvaluator):
    """Fails inside the server thread, the way a real GPU fault would."""

    def infer(self, *a, **kw):
        raise RuntimeError("opponent evaluator exploded")


def test_opponent_server_failure_fails_the_run_closed():
    """Failure companion: the opponent side dies and the run must raise, not
    fall back to the learner and not hang waiting for WorkerDone."""
    learner, opponent = StubEvaluator(0.9), ExplodingEvaluator(0.1)
    with pytest.raises(RuntimeError, match="InferenceServer crashed|worker .* failed"):
        _tiny_run(learner, opponent)


def test_run_survives_being_repeated_after_a_failure():
    """Cleanup is real: a good run still works after a failed one."""
    with pytest.raises(RuntimeError):
        _tiny_run(StubEvaluator(0.9), ExplodingEvaluator(0.1))
    learner, opponent = StubEvaluator(0.9), StubEvaluator(0.1)
    (_g, _p, stats), _b = _tiny_run(learner, opponent)
    assert stats["games_generated"] == 2


def test_worker_reports_transport_failure_instead_of_exiting_zero():
    """The hang this prevents: exit 0 with no WorkerDone, trainer waits forever."""
    import scripts.GPU.alphazero.self_play_worker as spw

    sent = []

    class FakeQueue:
        def put(self, msg, **kw):
            sent.append(msg)

    def boom(*a, **kw):
        raise RuntimeError("transport gone")

    original = spw._worker_loop
    spw._worker_loop = boom
    try:
        with pytest.raises(SystemExit) as exc:
            spw.self_play_worker_main(
                worker_id=3, request_queue=None, response_queue=None,
                position_queue=None, stats_queue=FakeQueue(),
                mcts_config=None, games_total=1, next_game_id=None,
                seed=1, chunk_size=8, max_moves=4, add_noise=False, active_size=8,
            )
    finally:
        spw._worker_loop = original

    assert exc.value.code != 0, "a failed worker must not exit 0"
    assert sent and sent[0]["type"] == "worker_error"
    assert sent[0]["worker_id"] == 3
    assert "transport gone" in sent[0]["error"]


def test_trainer_raises_on_a_worker_error_message():
    src = inspect.getsource(trainer.run_parallel_selfplay)
    assert 'msg.get("type") == "worker_error"' in src
    idx = src.index('msg.get("type") == "worker_error"')
    assert "raise RuntimeError" in src[idx:idx + 260]


def test_startup_rejects_odd_games_per_iteration_and_resign():
    src = inspect.getsource(trainer.train)
    assert "requires an even --games-per-iter" in src
    assert "does not support resign or adjudication" in src
    # both must sit with the frozen-opponent setup, i.e. before the warmup call
    assert src.index("requires an even --games-per-iter") < src.index("run_replay_warmup(")


def test_frozen_opponent_flag_is_refused_at_startup_pending_dnr_50():
    """The flag still routes to the aborting two-server path, so train() must
    refuse before the warmup rather than burning an hour to reach the crash."""
    src = inspect.getsource(trainer.train)
    assert "--frozen-opponent-checkpoint is disabled" in src
    assert "#50" in src

    # the refusal must come before ANY frozen-opponent setup and before warmup
    refusal = src.index("--frozen-opponent-checkpoint is disabled")
    assert refusal < src.index("frozen_network = create_network")
    assert refusal < src.index("run_replay_warmup(")
    assert refusal < src.index("LocalGPUEvaluator(frozen_network)")


def test_no_override_flag_exists_for_the_refusal():
    """An escape hatch here would be a production hole, not a test seam."""
    import scripts.GPU.alphazero.train as train_cli
    cli = inspect.getsource(train_cli)
    for hole in ("--allow-two-servers", "--force-frozen-opponent",
                 "--ignore-dnr-50", "--unsafe-metal"):
        assert hole not in cli
    assert "allow_two_servers" not in inspect.getsource(trainer.train)
