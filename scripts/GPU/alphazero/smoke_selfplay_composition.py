"""Real-GPU composition smoke: self-play over the single arbiter.

The arbiter smoke proved device ownership and routing with SYNTHETIC requests.
This one drives the path that actually matters --

    run_parallel_selfplay -> self_play_worker -> dual-root play_game -> MCTS
      -> two RemoteEvaluators -> the arbiter

-- with real games, so variable-size batches, tree reuse and backpressure meet
Metal for the first time. Crucially, real MCTS produces batches BELOW the row
cap, so mixed-model grouping inside one flush can finally occur.

Mixed grouping is MEASURED, not inferred. "A flush with more than one request"
proves nothing -- both could be the learner's. The instrumented server counts
`multi_request_flushes` and, separately, `mixed_model_flushes` (a pending flush
holding two distinct model ids). Only the latter is evidence.

Worker guarantee, stated honestly: run_parallel_selfplay cannot return normally
until every expected WorkerDone message arrives, and a worker failure surfaces as
worker_error -> RuntimeError. It does NOT expose the child processes' OS exit
codes, so this smoke claims the former and never the latter.

Three outcomes: PASS · NO_EXPOSURE (everything else fine but zero mixed flushes:
the condition was never exercised -- not an arbiter or Metal failure) · FAIL.

Exit: 0 PASS · 1 FAIL · 2 NO_EXPOSURE · 3 artifact exists · 4 sha mismatch ·
142 timeout · -6/134 SIGABRT · anything else invalid.

    python -m scripts.GPU.alphazero.smoke_selfplay_composition
"""
from __future__ import annotations

import os

# MUST precede any import of self_play: _MIRROR_PROB is read at import time.
os.environ["TWIXT_MIRROR_PROB"] = "0.0"

import argparse          # noqa: E402
import hashlib           # noqa: E402
import json              # noqa: E402
import random            # noqa: E402
import signal            # noqa: E402
import sys               # noqa: E402
import threading         # noqa: E402
import time              # noqa: E402
from pathlib import Path  # noqa: E402

# ---------------------------------------------------------------- frozen dose
CHECKPOINT = "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"
CHECKPOINT_SHA1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"
HIDDEN, BLOCKS = 128, 6
GAMES = 4                 # even, so the colour split is exact
WORKERS = 2
SIMULATIONS = 32
ACTIVE_SIZE = 24
MAX_MOVES = 60
MAX_BATCH_ROWS, FLUSH_MS = 14, 2
MASTER_SEED = 20260812
MIRROR_PROB = "0.0"
TIMEOUT_S = 1800
OUT_JSON = Path("logs/eval/composition_smoke.json")
OUT_EXIT = Path("logs/eval/composition_smoke.exit")

EXIT_PASS, EXIT_FAIL, EXIT_NO_EXPOSURE = 0, 1, 2


def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _ThreadRecordingEvaluator:
    """Records which thread reaches a model. One owner or the run fails."""

    def __init__(self, inner, seen: set):
        self._inner = inner
        self._seen = seen

    def build_input_tensor(self, state):
        return self._inner.build_input_tensor(state)

    def infer(self, *args, **kwargs):
        self._seen.add(threading.get_ident())
        return self._inner.infer(*args, **kwargs)


class _RecordingBuffer:
    """Stands in for ReplayBuffer, keeping each streamed chunk intact.

    Chunk structure is the evidence: every chunk must hold exactly one colour,
    and exactly four chunks may begin a game.
    """

    def __init__(self):
        self.chunks = []

    def add_positions(self, positions):
        self.chunks.append(list(positions))

    def add_game(self, game):                      # not used by the parallel path
        self.chunks.append(list(game.positions))

    def __len__(self):
        return sum(len(c) for c in self.chunks)


def _instrumented_server_class(base):
    """Subclass counting pending flushes by model composition, and recording
    its own serving thread so shutdown becomes measurable from outside."""
    holder = []

    class _Instrumented(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.multi_request_flushes = 0
            self.mixed_model_flushes = 0
            self.serving_thread = None
            self.run_forever_exited = False
            holder.append(self)

        def run_forever(self):
            self.serving_thread = threading.current_thread()
            try:
                return super().run_forever()
            finally:
                self.run_forever_exited = True

        def _flush(self, batch):
            # Inspect the PENDING batch, before grouping splits it by model.
            if len(batch) > 1:
                self.multi_request_flushes += 1
            if len({r.model_id for r in batch}) > 1:
                self.mixed_model_flushes += 1
            return super()._flush(batch)

    return _Instrumented, holder


def run() -> dict:
    from . import trainer as trainer_mod
    from .curriculum import CurriculumManager
    from .inference_server import InferenceServer
    from .local_evaluator import LocalGPUEvaluator
    from .mcts import MCTSConfig
    from .network import create_network
    from .ipc_messages import DEFAULT_MODEL_ID, OPPONENT_MODEL_ID
    from .self_play import _MIRROR_PROB as effective_mirror_prob

    seen_threads: set = set()

    def _build():
        net = create_network(hidden=HIDDEN, n_blocks=BLOCKS)
        net.load_weights(CHECKPOINT)
        net.eval()
        return _ThreadRecordingEvaluator(LocalGPUEvaluator(net, compile=False), seen_threads)

    learner_eval, opponent_eval = _build(), _build()

    # run_parallel_selfplay imports InferenceServer LOCALLY from its own module
    # (trainer.py:2035), so the trainer has no such attribute and patching it
    # would intercept nothing -- it would raise AttributeError before any game.
    # Patch the source module instead; the function-local import resolves it at
    # call time.
    from . import inference_server as inference_server_mod
    instrumented, holder = _instrumented_server_class(InferenceServer)
    original_server = inference_server_mod.InferenceServer
    inference_server_mod.InferenceServer = instrumented

    buffer = _RecordingBuffer()
    t0 = time.perf_counter()
    error = None
    stats = {}
    try:
        _games, _new_positions, stats = trainer_mod.run_parallel_selfplay(
            evaluator=learner_eval,
            mcts_config=MCTSConfig(n_simulations=SIMULATIONS,
                                   eval_batch_size=MAX_BATCH_ROWS),
            games_to_play=GAMES,
            n_workers=WORKERS,
            master_rng=random.Random(MASTER_SEED),
            max_moves=MAX_MOVES,
            active_size=ACTIVE_SIZE,
            curriculum=CurriculumManager(sizes=(ACTIVE_SIZE,)),
            buffer=buffer,
            opponent_evaluator=opponent_eval,
        )
    except Exception as exc:                       # noqa: BLE001 - reported, fatal
        error = f"{type(exc).__name__}: {exc}"
    finally:
        inference_server_mod.InferenceServer = original_server
    elapsed = time.perf_counter() - t0

    server = holder[0] if holder else None
    telemetry = server.model_telemetry() if server else {}

    # --- chunk-level evidence ---
    chunk_colours = [sorted({p.to_move for p in c}) for c in buffer.chunks if c]
    single_colour_chunks = all(len(c) == 1 for c in chunk_colours)
    starts = [c[0] for c in buffer.chunks if c and c[0].ply in (0, 1)]
    start_colours = sorted(p.to_move for p in starts)

    report = {
        "checkpoint": CHECKPOINT,
        "checkpoint_sha1_expected": CHECKPOINT_SHA1,
        "dose": {"games": GAMES, "workers": WORKERS, "simulations": SIMULATIONS,
                 "active_size": ACTIVE_SIZE, "max_moves": MAX_MOVES,
                 "max_batch_rows": MAX_BATCH_ROWS, "flush_ms": FLUSH_MS,
                 "master_seed": MASTER_SEED, "mirror_prob": MIRROR_PROB},
        "mirror_prob_effective": effective_mirror_prob,
        "error": error,
        "games_generated": stats.get("games_generated"),
        "chunks": len(buffer.chunks),
        "positions": len(buffer),
        "every_chunk_single_colour": single_colour_chunks,
        "game_start_chunks": len(starts),
        "game_start_colours": start_colours,
        "telemetry": telemetry,
        "inference_threads_observed": len(seen_threads),
        "server_thread_recorded": bool(server is not None and server.serving_thread),
        "server_run_forever_exited": bool(server is not None and server.run_forever_exited),
        "server_thread_stopped": bool(
            server is not None and server.serving_thread is not None
            and not server.serving_thread.is_alive()),
        "multi_request_flushes": getattr(server, "multi_request_flushes", None),
        "mixed_model_flushes": getattr(server, "mixed_model_flushes", None),
        "elapsed_s": round(elapsed, 2),
    }

    both_served = bool(telemetry) and all(
        telemetry.get(m, {}).get("requests", 0) > 0
        for m in (DEFAULT_MODEL_ID, OPPONENT_MODEL_ID))

    core_ok = bool(
        error is None
        and stats.get("games_generated") == GAMES
        and single_colour_chunks
        and len(starts) == GAMES
        and start_colours == ["black", "black", "red", "red"]
        and len(seen_threads) == 1
        and both_served
        and effective_mirror_prob == 0.0
        # Shutdown is measured, not assumed: the thread was recorded, its
        # finally ran, and it is no longer alive after the call returned.
        and report["server_thread_recorded"]
        and report["server_run_forever_exited"]
        and report["server_thread_stopped"]
    )
    mixed = report["mixed_model_flushes"] or 0

    report["core_ok"] = core_ok
    if core_ok and mixed >= 1:
        report["OUTCOME"] = "PASS"
    elif core_ok:
        # The central condition was never exercised. NOT an arbiter or Metal
        # failure -- scheduling simply never co-queued both models.
        report["OUTCOME"] = "NO_EXPOSURE"
    else:
        report["OUTCOME"] = "FAIL"
    return report


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    for path in (OUT_JSON, OUT_EXIT):
        if path.exists():
            print(f"REFUSE: {path} already exists")
            return 3

    actual = _sha1_file(CHECKPOINT)
    if actual != CHECKPOINT_SHA1:
        print(f"REFUSE: sha1 {actual} != expected {CHECKPOINT_SHA1}")
        return 4

    signal.alarm(TIMEOUT_S)
    report = run()
    signal.alarm(0)

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"\nOUTCOME={report['OUTCOME']}  written={OUT_JSON}")
    return {"PASS": EXIT_PASS, "NO_EXPOSURE": EXIT_NO_EXPOSURE}.get(
        report["OUTCOME"], EXIT_FAIL)


if __name__ == "__main__":
    sys.exit(main())
