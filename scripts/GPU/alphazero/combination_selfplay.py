"""THE COMBINATION POLICY PILOT'S SELF-PLAY -- card `docs/superpowers/2026-10-28-combination-policy-pilot-card.md`
§3 and §4: the proven agent playing itself, in CPU worker processes behind the one Metal-owning inference arbiter.
Nothing runs on import, and importing this module loads no MLX.

THE GAME (`play_slot`) is `mirror_halving_combination.play_game` -- the strength match's own loop, called, never
copied -- with the combination agent on BOTH sides: `MirrorAveragingEvaluator` over the given evaluator,
`RootSequentialHalvingMCTS` at 200 simulations, Stage B's PROTOCOL (24x24, 280 plies, the opening-temperature
readout, no root noise). The loop derives the match's four streams from the game's seed (searches seed ^ 0xA5A5A5 /
^ 0x5A5A5A, readouts ^ 0xC3C3C3 / ^ 0x3C3C3C), reads `last_choice` from ply 20 and the readout before. The one
addition is `capturing`, a subclass of the halving search that hands each completed search, unchanged, to a sink:
each row's target comes from the very root that chose the move (`halving_policy_target.search_row`).

THE SEEDS: slot g plays seed slot_base + g, whichever worker claims it. Workers claim slots from one shared counter;
no worker seed exists, and `play_slot` takes no RNG.

THE DRIVER (`generate`): one `InferenceServer` thread in the calling process owns the device (#50) and batches across
workers up to `max_batch_rows` rows per call; `n_workers` spawned processes each wrap a `RemoteEvaluator`, give that
INSTANCE `network = None` (MirrorAveragingEvaluator reads `base.network`, which the remote evaluator lacks; no class
is edited), and report "hello" (their MLX_ENABLE_TF32 and whether MLX is loaded -- it must not be), "start" BEFORE
each claimed game, then "game" with its record and rows. The calling thread never touches MLX while the server runs.
Any worker or server failure raises; the caller's record boundary turns it into a VOID.
"""
from __future__ import annotations

import os
import queue
import signal
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import halving_policy_target as T
from . import mirror_halving_combination as K

SETTING = "MLX_ENABLE_TF32"
M_PAD, N_CAND = 576, 16                        # the policy width cap (ffe6b342); the halving candidates
QUEUE_SIZES = {"request": 256, "response": 64, "result": 128, "stats": 128}      # run_parallel_selfplay's


def capturing(sink: Callable[[Any, Any], None]):
    """A RootSequentialHalvingMCTS whose every completed halving search is handed to sink(search, root)."""
    from .root_sequential_halving import RootSequentialHalvingMCTS

    class CapturingHalvingMCTS(RootSequentialHalvingMCTS):
        def search_with_root(self, root_state, add_noise=True):
            out = super().search_with_root(root_state, add_noise=add_noise)
            if self.last_choice is not None:
                sink(self, out[2])
            return out

    return CapturingHalvingMCTS


def protocol_config():
    """Stage B's PROTOCOL as the match reads it; refuses anything but 24x24, 280 plies and the frozen variant."""
    from . import eval_runner as ER
    from . import root_sequential_halving as RSH
    from .external_learning_stage3 import PROTOCOL
    K.check_variant(RSH)
    config = ER.EvalConfig(**PROTOCOL)
    if (config.board_size, config.max_moves) != (24, 280):
        raise ValueError("PROTOCOL is not 24x24 / 280 plies")
    return config


def pack_rows(rows: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """One game's rows as fixed-width arrays (legal moves padded to M_PAD, candidates to N_CAND)."""
    k = len(rows)
    out = {"ply": np.zeros(k, np.int32), "n_legal": np.zeros(k, np.int32), "n_max": np.zeros(k, np.int32),
           "choice": np.zeros(k, np.int32), "n_rounds": np.zeros(k, np.int32),
           "rounds": np.zeros((k, N_CAND), np.int32), "prior": np.zeros((k, M_PAD), np.float32),
           "target": np.zeros((k, M_PAD), np.float64), "cand": np.full((k, N_CAND), -1, np.int32),
           "q": np.full((k, N_CAND), np.nan, np.float64), "visits": np.zeros((k, N_CAND), np.int32)}
    for i, r in enumerate(rows):
        n, c = r["n_legal"], len(r["cand"])
        if n > M_PAD or c > N_CAND:
            raise ValueError(f"ply {r['ply']}: {n} legal moves or {c} candidates exceed {M_PAD} / {N_CAND}")
        for key in ("ply", "n_legal", "n_max", "choice", "n_rounds"):
            out[key][i] = r[key]
        out["prior"][i, :n], out["target"][i, :n] = r["prior"], r["target"]
        out["cand"][i, :c], out["q"][i, :c], out["visits"][i, :c] = r["cand"], r["q"], r["visits"]
        out["rounds"][i, :c] = r["rounds"]
    return out


def play_slot(evaluator: Any, seed: int) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """One game of the combination agent against itself at `seed`: (game record, packed rows)."""
    from . import eval_runner as ER
    from .mirror_averaging import MirrorAveragingEvaluator

    if type(seed) is not int:
        raise ValueError(f"the seed must be an int, got {seed!r}")
    config = protocol_config()
    readout = ER.readout_from_eval_config(config)
    avg = MirrorAveragingEvaluator(evaluator)
    rows: List[Dict[str, Any]] = []
    cls = capturing(lambda search, root: rows.append(T.search_row(search, root)))
    winner, reason, n_moves, records = K.play_game(
        avg, avg, config, seed, capture=True, red_readout=readout, black_readout=readout,
        red_sims=K.CANDIDATE_SIMS, black_sims=K.CANDIDATE_SIMS, red_cls=cls, black_cls=cls)
    moves = [(r["row"], r["col"]) for r in records]
    for row in rows:
        if records[row["ply"]]["ply"] != row["ply"]:
            raise ValueError(f"ply {row['ply']}: the records are out of order")
    game = {"seed": seed, "moves": moves, "winner": winner, "reason": reason, "n_moves": n_moves,
            "n_rows": len(rows), "row_plies": [r["ply"] for r in rows]}
    return game, pack_rows(rows)


def _mlx_loaded() -> bool:
    return any(m == "mlx" or m.startswith("mlx.") for m in sys.modules)


def worker_main(worker_id: int, request_queue: Any, response_queue: Any, result_queue: Any, next_slot: Any,
                n_slots: int, slot_base: int, play: Optional[Callable[..., Any]] = None) -> None:
    """A CPU worker: claim slots until none remain; report hello, then start before and game after each one."""
    from .remote_evaluator import RemoteEvaluator

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        play = play or play_slot
        result_queue.put(("hello", worker_id, os.environ.get(SETTING), _mlx_loaded()))
        evaluator = RemoteEvaluator(worker_id, request_queue, response_queue)
        evaluator.network = None          # instance attribute: MirrorAveragingEvaluator reads base.network
        while True:
            with next_slot.get_lock():
                g = next_slot.value
                if g >= n_slots:
                    break
                next_slot.value += 1
            result_queue.put(("start", worker_id, g))
            t = time.perf_counter()
            game, rows = play(evaluator, slot_base + g)
            result_queue.put(("game", worker_id, g, game, rows, time.perf_counter() - t))
        result_queue.put(("done", worker_id))
    except BaseException as e:            # fail loud: the driver must never wait on a dead worker
        try:
            result_queue.put(("error", worker_id, f"{type(e).__name__}: {e}"))
        except Exception:
            pass
        sys.exit(1)


class GenerationFailed(RuntimeError):
    """A worker, the server or the protocol failed; the caller records a VOID."""


def generate(evaluator: Any, n_slots: int, slot_base: int, n_workers: int, max_batch_rows: int,
             on_start: Callable[[int, int], None], on_game: Callable[..., None],
             play: Optional[Callable[..., Any]] = None, poll_s: float = 1.0) -> Dict[str, Any]:
    """Play slots 0..n_slots-1 (seed slot_base + g) on n_workers spawned CPU workers behind one InferenceServer
    thread that serves `evaluator`. Calls on_start(slot, worker) when a worker reports a claimed slot, and
    on_game(slot, worker, game, rows, seconds) when it ends. Returns the claim count, the workers' hellos and the
    server's telemetry; raises GenerationFailed on any failure (workers are always stopped)."""
    import multiprocessing as mp
    import threading

    from .inference_server import InferenceServer
    from .ipc_messages import DEFAULT_MODEL_ID, StopSignal

    if n_slots < 1 or n_workers < 1 or max_batch_rows < 1:
        raise ValueError(f"n_slots {n_slots}, n_workers {n_workers}, max_batch_rows {max_batch_rows}")
    ctx = mp.get_context("spawn")
    n_spawn = min(n_workers, n_slots)
    request_queue = ctx.Queue(maxsize=QUEUE_SIZES["request"])
    response_queues = {(w, DEFAULT_MODEL_ID): ctx.Queue(maxsize=QUEUE_SIZES["response"]) for w in range(n_spawn)}
    result_queue = ctx.Queue(maxsize=QUEUE_SIZES["result"])
    stats_queue = ctx.Queue(maxsize=QUEUE_SIZES["stats"])
    next_slot = ctx.Value("i", 0)
    server = InferenceServer({DEFAULT_MODEL_ID: evaluator}, request_queue, response_queues,
                             max_batch_rows=max_batch_rows, flush_ms=2, stats_queue=stats_queue)
    thread = threading.Thread(target=server.run_forever, daemon=True)
    thread.start()
    workers = []
    hellos: Dict[int, Dict[str, Any]] = {}
    done, started, ended = set(), [], []
    try:
        for w in range(n_spawn):
            p = ctx.Process(target=worker_main, args=(w, request_queue, response_queues[(w, DEFAULT_MODEL_ID)],
                                                      result_queue, next_slot, n_slots, slot_base, play))
            p.start()
            workers.append(p)
        while len(done) < n_spawn:
            try:
                msg = result_queue.get(timeout=poll_s)
            except queue.Empty:
                _check_alive(server, thread, stats_queue, workers, done)
                continue
            kind, w = msg[0], msg[1]
            if kind == "hello":
                hellos[w] = {"setting": msg[2], "mlx_loaded": msg[3]}
                if msg[2] != "0" or msg[3]:
                    raise GenerationFailed(f"worker {w}: {SETTING}={msg[2]!r}, MLX loaded {msg[3]}")
            elif kind == "start":
                started.append(msg[2])
                on_start(msg[2], w)
            elif kind == "game":
                ended.append(msg[2])
                on_game(msg[2], w, msg[3], msg[4], msg[5])
            elif kind == "done":
                done.add(w)
            elif kind == "error":
                raise GenerationFailed(f"worker {w} failed: {msg[2]}")
            else:
                raise GenerationFailed(f"unknown message {kind!r} from worker {w}")
        request_queue.put(StopSignal())
        thread.join(timeout=30)
        if thread.is_alive() or server._error is not None:
            raise GenerationFailed(f"the inference server did not stop cleanly: {server._error!r}")
        for p in workers:
            p.join(timeout=30)
        if any(p.exitcode != 0 for p in workers):
            raise GenerationFailed(f"worker exit codes {[p.exitcode for p in workers]}")
        return {"claims": next_slot.value, "started": sorted(started), "ended": sorted(ended), "hellos": hellos,
                "workers": n_spawn, "server": server.model_telemetry()[DEFAULT_MODEL_ID],
                "max_batch_rows": max_batch_rows}
    finally:
        server.stop()
        for p in workers:
            if p.is_alive():
                p.terminate()
        for p in workers:
            p.join(timeout=10)


def _check_alive(server: Any, thread: Any, stats_queue: Any, workers: List[Any], done: set) -> None:
    try:
        msg = stats_queue.get_nowait()
    except queue.Empty:
        msg = None
    if isinstance(msg, dict) and msg.get("type") == "server_error":
        raise GenerationFailed(f"the inference server crashed: {msg.get('error')}")
    if not thread.is_alive():
        raise GenerationFailed(f"the inference server thread stopped: {server._error!r}")
    #: a worker exits 0 only after queueing "done", which may still be in flight; any other exit is a failure
    dead = [w for w, p in enumerate(workers) if w not in done and not p.is_alive() and p.exitcode != 0]
    if dead:
        raise GenerationFailed(f"workers {dead} died (exit codes {[workers[w].exitcode for w in dead]})")
