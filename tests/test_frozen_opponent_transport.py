"""Frozen-opponent worker/server plumbing, exercised with stubs only.

Step 3 of the card's sequence. These run no accelerated work: the inference
servers are driven with a stub evaluator over real multiprocessing queues, and
the trainer wiring is checked by monkeypatching the module attributes.

Covered:
* the opponent model, its response queue and its evaluator exist ONLY in
  frozen-opponent mode -- there is never a second server;
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
from scripts.GPU.alphazero.ipc_messages import (
    StopSignal, InferenceRequest, DEFAULT_MODEL_ID, OPPONENT_MODEL_ID)


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


def test_exactly_one_server_serves_both_models():
    """Replaces test_second_server_is_conditional_*: there is never a second."""
    src = inspect.getsource(trainer.run_parallel_selfplay)
    assert src.count("InferenceServer(") == 1
    for gone in ("opp_server", "opp_request_queue = ctx.Queue", "opp_response_queues ="):
        assert gone not in src, f"two-server remnant: {gone}"
    assert "evaluators=_evaluators" in src


def test_response_queues_are_addressed_by_worker_and_model():
    src = inspect.getsource(trainer.run_parallel_selfplay)
    assert "(wid, DEFAULT_MODEL_ID): ctx.Queue" in src
    assert "(wid, OPPONENT_MODEL_ID): ctx.Queue" in src
    # one request queue, shared by both models
    assert src.count("request_queue = ctx.Queue") == 1


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


def test_single_server_lifecycle_with_two_models():
    """Replaces test_two_servers_run_and_stop_*: ONE server, two evaluators."""
    ctx = mp.get_context("spawn")
    rq = ctx.Queue(maxsize=16)
    resp = {(0, DEFAULT_MODEL_ID): ctx.Queue(maxsize=16),
            (0, OPPONENT_MODEL_ID): ctx.Queue(maxsize=16)}
    server = InferenceServer(
        evaluators={DEFAULT_MODEL_ID: StubEvaluator(0.9),
                    OPPONENT_MODEL_ID: StubEvaluator(0.1)},
        request_queue=rq, response_queues=resp, max_batch_rows=14, flush_ms=2)
    t = threading.Thread(target=server.run_forever, daemon=True)
    t.start()
    assert t.is_alive()

    try:
        rq.put(StopSignal(), timeout=0.5)
    except Exception:
        pass
    server.stop()
    t.join(timeout=2.0)
    assert not t.is_alive(), "server thread failed to stop"
    for q in [rq, *resp.values()]:
        q.cancel_join_thread(); q.close()


def test_shutdown_cleans_every_model_addressed_queue():
    """Replaces test_shutdown_path_is_symmetric_*: one lifecycle, whole map."""
    src = inspect.getsource(trainer.run_parallel_selfplay)
    tail = src[src.index("server.stop()"):]
    assert "server_thread.join" in tail
    assert "response_queues.values()" in tail          # every (worker, model) queue
    assert "opp_server.stop()" not in tail             # no second lifecycle


def test_server_requires_at_least_one_evaluator():
    with pytest.raises(ValueError, match="at least one evaluator"):
        InferenceServer(evaluators={}, request_queue=None, response_queues={})


def test_either_model_failure_reaches_the_fail_closed_handler():
    """Replaces test_both_servers_report_crashes_*: one server, one handler,
    and a failure in EITHER model surfaces through it."""
    src = inspect.getsource(trainer.run_parallel_selfplay)
    assert src.count("stats_queue=stats_queue") == 1   # one server now
    assert 'raise RuntimeError(f"InferenceServer crashed' in src
    assert 'msg.get("type") == "worker_error"' in src


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
# opponent separation. These run the real thing: two workers, ONE arbiter, real
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


def test_worker_to_single_arbiter_round_trip():
    """Real workers -> ONE arbiter -> two models -> buffer."""
    learner, opponent = StubEvaluator(0.9), StubEvaluator(0.1)
    (_games, _new_positions, stats), buffer = _tiny_run(learner, opponent)

    assert stats["games_generated"] == 2, stats
    assert learner.calls > 0, "learner server never served a request"
    assert opponent.calls > 0, "opponent server never served a request"
    assert len(buffer) > 0, "no learner positions reached the buffer"


def test_single_game_round_trip_yields_exactly_one_colour():
    """Strengthened replacement for the vacuous {red, black}-subset assertion.

    One game has one learner colour, so EVERY buffered row must share it. This
    fails if a single opponent-to-move position leaks into training.
    """
    learner, opponent = StubEvaluator(0.9), StubEvaluator(0.1)
    (_g, _p, _s), buffer = _tiny_run(learner, opponent, games=1, workers=2)
    colours = {r.to_move for r in buffer.buffer}
    assert buffer.buffer, "expected learner rows"
    assert len(colours) == 1, f"one game produced two colours of row: {colours}"


def test_colour_alternates_across_games_in_the_buffer():
    """Companion: over games 0 and 1 both colours appear, so the single-colour
    assertion above is a per-game property and not an artefact of never
    switching."""
    (_g, _p, _s), buffer = _tiny_run(StubEvaluator(0.9), StubEvaluator(0.1),
                                     games=2, workers=2)
    assert {r.to_move for r in buffer.buffer} == {"red", "black"}


class ExplodingEvaluator(StubEvaluator):
    """Fails inside the server thread, the way a real GPU fault would."""

    def infer(self, *a, **kw):
        raise RuntimeError("opponent evaluator exploded")


@pytest.mark.parametrize("exploding_slot", ["learner", "opponent"])
def test_either_model_failure_fails_the_run_closed(exploding_slot):
    """Behavioural half of the 'either model' gate -- not source inspection.

    Whichever model dies, the run must raise rather than fall back to the other
    or hang waiting for a WorkerDone that never comes.
    """
    if exploding_slot == "learner":
        learner, opponent = ExplodingEvaluator(0.9), StubEvaluator(0.1)
    else:
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


# ------------------------------------------------- one owner, and no fallback
def test_both_models_are_served_on_exactly_one_thread():
    """Invariant 1, instrumented: two evaluators, ONE thread ever calls them.

    This is the whole point of the arbiter -- two owners abort the device (#50).
    """
    import threading as _th
    seen_threads = set()

    class ThreadRecordingStub(StubEvaluator):
        def infer(self, *a, **kw):
            seen_threads.add(_th.get_ident())
            return super().infer(*a, **kw)

    ctx = mp.get_context("spawn")
    rq = ctx.Queue(maxsize=64)
    resp = {(0, DEFAULT_MODEL_ID): ctx.Queue(maxsize=64),
            (0, OPPONENT_MODEL_ID): ctx.Queue(maxsize=64)}
    a, b = ThreadRecordingStub(0.9), ThreadRecordingStub(0.1)
    server = InferenceServer(evaluators={DEFAULT_MODEL_ID: a, OPPONENT_MODEL_ID: b},
                             request_queue=rq, response_queues=resp,
                             max_batch_rows=14, flush_ms=2)
    t = threading.Thread(target=server.run_forever, daemon=True); t.start()

    def _req(model_id, rid):
        return InferenceRequest(
            worker_id=0, request_id=rid,
            boards=np.zeros((2, 8, 8, 30), np.float32),
            move_rows=np.zeros((2, 4), np.int32), move_cols=np.zeros((2, 4), np.int32),
            move_mask=np.ones((2, 4), np.float32), active_size=8, model_id=model_id)

    for i in range(6):
        rq.put(_req(DEFAULT_MODEL_ID if i % 2 == 0 else OPPONENT_MODEL_ID, i + 1))
    # consume AND check: a response must carry its own model's signature
    for key, expect in ((( 0, DEFAULT_MODEL_ID), 0.9), ((0, OPPONENT_MODEL_ID), 0.1)):
        r = resp[key].get(timeout=10)
        assert np.allclose(r.priors[r.priors > 0], expect), f"{key} got foreign values"

    server.stop(); t.join(timeout=2.0)
    assert a.calls > 0 and b.calls > 0, "both models must be exercised"
    assert len(seen_threads) == 1, f"more than one thread touched a model: {seen_threads}"
    for q in [rq, *resp.values()]:
        q.cancel_join_thread(); q.close()


def test_unknown_model_id_raises_and_never_falls_back():
    from scripts.GPU.alphazero.inference_server import InferenceServer as S
    only = StubEvaluator(0.9)
    server = S(evaluators={DEFAULT_MODEL_ID: only}, request_queue=None,
               response_queues={}, max_batch_rows=14, flush_ms=2)
    req = InferenceRequest(
        worker_id=0, request_id=1,
        boards=np.zeros((1, 8, 8, 30), np.float32),
        move_rows=np.zeros((1, 4), np.int32), move_cols=np.zeros((1, 4), np.int32),
        move_mask=np.ones((1, 4), np.float32), active_size=8, model_id="nope")
    with pytest.raises(KeyError, match="unknown model_id"):
        server._flush([req])
    assert only.calls == 0, "fell back to the only registered evaluator"


def test_missing_response_route_raises_rather_than_misrouting():
    """A reply with nowhere to go must fail, not land in the wrong queue."""
    server = InferenceServer(
        evaluators={DEFAULT_MODEL_ID: StubEvaluator(0.9)},
        request_queue=None, response_queues={}, max_batch_rows=14, flush_ms=2)
    req = InferenceRequest(
        worker_id=7, request_id=1,
        boards=np.zeros((1, 8, 8, 30), np.float32),
        move_rows=np.zeros((1, 4), np.int32), move_cols=np.zeros((1, 4), np.int32),
        move_mask=np.ones((1, 4), np.float32), active_size=8,
        model_id=DEFAULT_MODEL_ID)
    with pytest.raises(KeyError):
        server._flush([req])


def test_default_off_registers_one_model_and_one_queue_per_worker():
    """One-model equivalence: no extra queues, no extra models."""
    src = inspect.getsource(trainer.run_parallel_selfplay)
    assert "if opponent_evaluator is not None:" in src
    # opponent queues and the opponent model are both behind that guard
    idx = src.index("_evaluators = {DEFAULT_MODEL_ID: evaluator}")
    assert "OPPONENT_MODEL_ID] = opponent_evaluator" in src[idx:idx + 260]


class RecordingStub(StubEvaluator):
    """Returns a value distinguishable per model, and records what it was given."""

    def __init__(self, tag):
        super().__init__(tag)
        self.seen_boards = []

    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        self.seen_boards.append(np.array(boards, copy=True))
        b, m = move_mask.shape
        return (np.full((b, m), self.tag, np.float32) * move_mask,
                np.full((b,), self.tag, np.float32))


def test_one_flush_with_two_models_routes_values_and_inputs_correctly():
    """The preregistered routing+grouping gate, asserted on VALUES not counts.

    Both requests share worker_id AND request_id, so only the (worker_id,
    model_id) key can disambiguate them -- exactly the collision that made a
    shared response queue a silent contamination bug. Both go through ONE
    _flush, so grouping by model is exercised too.
    """
    B, M, SIZE = 3, 4, 8
    learner, opponent = RecordingStub(0.25), RecordingStub(0.75)
    qs = {(0, DEFAULT_MODEL_ID): __import__("queue").Queue(),
          (0, OPPONENT_MODEL_ID): __import__("queue").Queue()}
    server = InferenceServer(
        evaluators={DEFAULT_MODEL_ID: learner, OPPONENT_MODEL_ID: opponent},
        request_queue=None, response_queues=qs, max_batch_rows=64, flush_ms=2)

    def _req(model_id, fill):
        return InferenceRequest(
            worker_id=0, request_id=1,                 # SAME id on purpose
            boards=np.full((B, SIZE, SIZE, 30), fill, np.float32),
            move_rows=np.zeros((B, M), np.int32), move_cols=np.zeros((B, M), np.int32),
            move_mask=np.ones((B, M), np.float32), active_size=SIZE, model_id=model_id)

    server._flush([_req(DEFAULT_MODEL_ID, 1.0), _req(OPPONENT_MODEL_ID, 2.0)])

    # 1. each response landed in its OWN model-addressed queue, with that
    #    model's distinguishable values -- not the other's
    lr = qs[(0, DEFAULT_MODEL_ID)].get_nowait()
    orr = qs[(0, OPPONENT_MODEL_ID)].get_nowait()
    assert np.allclose(lr.values, 0.25), f"learner queue got {lr.values[:1]}"
    assert np.allclose(orr.values, 0.75), f"opponent queue got {orr.values[:1]}"
    assert qs[(0, DEFAULT_MODEL_ID)].empty() and qs[(0, OPPONENT_MODEL_ID)].empty()

    # 2. each evaluator saw ONLY its own input -- no cross-feeding
    assert len(learner.seen_boards) == 1 and len(opponent.seen_boards) == 1
    assert np.allclose(learner.seen_boards[0], 1.0)
    assert np.allclose(opponent.seen_boards[0], 2.0)

    # 3. never mixed in one forward pass: one batch each, exact rows
    tel = server.model_telemetry()
    for model in (DEFAULT_MODEL_ID, OPPONENT_MODEL_ID):
        assert tel[model] == {"requests": 1, "rows": B, "batches": 1}, tel[model]


def test_per_model_telemetry_is_exposed():
    server = InferenceServer(
        evaluators={DEFAULT_MODEL_ID: StubEvaluator(0.9),
                    OPPONENT_MODEL_ID: StubEvaluator(0.1)},
        request_queue=None, response_queues={}, max_batch_rows=14, flush_ms=2)
    tel = server.model_telemetry()
    assert set(tel) == {DEFAULT_MODEL_ID, OPPONENT_MODEL_ID}
    assert all(set(v) == {"requests", "rows", "batches"} for v in tel.values())
    assert all(v["batches"] == 0 for v in tel.values())
