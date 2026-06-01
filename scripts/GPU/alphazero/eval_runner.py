"""Shared executor for checkpoint-tournament games.

A flat list of EvalGameTask is drained by run_game_tasks, which both the
match and tournament scripts call. workers==1 runs in-process; workers>1
uses a spawn worker pool with a shared atomic task counter (the trainer
idiom). Determinism is task-derived: same base_seed + schedule => same
games regardless of worker count.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import random
from dataclasses import dataclass
from typing import Callable, Optional

from .game.twixt_state import TwixtState
from .mcts import MCTS, MCTSConfig

# game_idx and pairing offsets share this stride; games-per-pairing must
# stay below it so task_ids/seeds never collide across pairings.
GAMES_PER_PAIRING_LIMIT = 1_000_000

EvaluatorFactory = Callable[[str], object]


@dataclass(frozen=True)
class EvalGameTask:
    task_id: int
    pairing_id: str
    game_idx: int
    red_checkpoint: str
    black_checkpoint: str
    seed: int


@dataclass
class EvalGameResult:
    task_id: int
    pairing_id: str
    game_idx: int
    red_checkpoint: str
    black_checkpoint: str
    winner: Optional[str]            # "red" | "black" | None
    winner_checkpoint: Optional[str]
    reason: str                      # "win"|"state_cap"|"board_full"|"unknown_error"
    n_moves: int
    red_score: float
    black_score: float


@dataclass(frozen=True)
class EvalConfig:
    board_size: int = 24
    mcts_sims: int = 400              # SIMS_TABLE[24]
    mcts_eval_batch_size: int = 14
    mcts_stall_flush_sims: int = 48
    selection_mode: str = "opening_temperature"   # or "argmax"
    opening_temp_plies: int = 20
    temp_high: float = 1.0
    temp_low: float = 0.1
    max_moves: int = 280             # MAX_MOVES_TABLE[24]


@dataclass(frozen=True)
class _WorkerDone:
    worker_id: int


@dataclass(frozen=True)
class _WorkerFailed:
    worker_id: int
    error: str


def cfg_from(config: EvalConfig) -> MCTSConfig:
    """Map EvalConfig -> MCTSConfig. argmax mode zeroes temps to hit
    select_move's deterministic argmax branch."""
    if config.selection_mode == "argmax":
        th, tl = 0.0, 0.0
    elif config.selection_mode == "opening_temperature":
        th, tl = config.temp_high, config.temp_low
    else:
        raise ValueError(f"unknown selection_mode {config.selection_mode!r}")
    return MCTSConfig(
        n_simulations=config.mcts_sims,
        eval_batch_size=config.mcts_eval_batch_size,
        stall_flush_sims=config.mcts_stall_flush_sims,
        temp_threshold_ply=config.opening_temp_plies,
        temp_high=th,
        temp_low=tl,
    )


def play_eval_game(red_eval, black_eval, config: EvalConfig, seed: int):
    """Play one A-vs-B game. Returns (winner, reason, n_moves).

    Independent per-side child RNGs (seed-derived) avoid cross-player
    coupling. Classification is explicit via winner()/ply/legal_moves --
    is_terminal() conflates win/cap/board-full (twixt_state.py:549-574).
    """
    mcts_red = MCTS(red_eval, cfg_from(config), random.Random(seed ^ 0xA5A5A5))
    mcts_black = MCTS(black_eval, cfg_from(config), random.Random(seed ^ 0x5A5A5A))
    state = TwixtState(active_size=config.board_size, to_move="red",
                       max_plies_limit=config.max_moves)
    ply = 0
    # Explicit loop condition -- do NOT gate on is_terminal(), which conflates
    # win / cap / board-full. Continue while there is no real winner, the ply
    # cap is not reached, and legal moves remain.
    while state.winner() is None and ply < config.max_moves and state.legal_moves():
        mcts = mcts_red if state.to_move == "red" else mcts_black
        counts, _ = mcts.search(state, add_noise=False)
        move = mcts.select_move(counts, ply)
        state = state.apply_move(move)
        ply += 1
    winner = state.winner()
    if winner is not None:
        reason = "win"
    elif ply >= config.max_moves:
        reason = "state_cap"
    elif not state.legal_moves():
        reason = "board_full"
    else:
        reason = "unknown_error"
    return winner, reason, ply


def make_result(task: EvalGameTask, winner, reason, n_moves) -> EvalGameResult:
    """Build a result, mapping winner color -> checkpoint and 0/0.5/1 scores."""
    if winner == "red":
        red_score, black_score, winner_ckpt = 1.0, 0.0, task.red_checkpoint
    elif winner == "black":
        red_score, black_score, winner_ckpt = 0.0, 1.0, task.black_checkpoint
    else:
        red_score, black_score, winner_ckpt = 0.5, 0.5, None
    return EvalGameResult(
        task_id=task.task_id, pairing_id=task.pairing_id, game_idx=task.game_idx,
        red_checkpoint=task.red_checkpoint, black_checkpoint=task.black_checkpoint,
        winner=winner, winner_checkpoint=winner_ckpt, reason=reason,
        n_moves=n_moves, red_score=red_score, black_score=black_score,
    )


def build_pairing_tasks(pairing_id, a_ckpt, b_ckpt, games, base_seed, pairing_index):
    """Balanced-color tasks for one pairing. Even game_idx -> red=A; odd -> red=B.
    task_id and seed are task-derived (stable across worker counts)."""
    if games < 2:
        raise ValueError("games must be >= 2")
    if games % 2 != 0:
        # Color balancing assigns A=red on even game_idx, A=black on odd.
        # An odd count gives one model an extra red game -> biased.
        raise ValueError("games must be even for balanced colors")
    if games >= GAMES_PER_PAIRING_LIMIT:
        raise ValueError(f"games must be < {GAMES_PER_PAIRING_LIMIT}")
    offset = pairing_index * GAMES_PER_PAIRING_LIMIT
    tasks = []
    for g in range(games):
        red, black = (a_ckpt, b_ckpt) if g % 2 == 0 else (b_ckpt, a_ckpt)
        tasks.append(EvalGameTask(
            task_id=offset + g, pairing_id=pairing_id, game_idx=g,
            red_checkpoint=red, black_checkpoint=black, seed=base_seed + offset + g,
        ))
    return tasks


def short_id(token: str) -> str:
    """Derive a short iter id from a checkpoint path, or pass a bare id through.

    Lives here (low-level shared module) so eval_summary and the tournament/
    match scripts all import it from one place — avoids a circular import.
    """
    base = os.path.basename(token)
    if base.startswith("model_iter_") and base.endswith(".safetensors"):
        return base[len("model_iter_"):-len(".safetensors")]
    return token


def resolve_checkpoint(token: str, checkpoints_dir: str) -> str:
    """Resolve a token to a checkpoint path.

    A path (contains os.sep or ends with .safetensors) passes through; a bare
    iter id resolves to <dir>/model_iter_<id>.safetensors.
    """
    if os.sep in token or token.endswith(".safetensors"):
        return token
    return os.path.join(checkpoints_dir, f"model_iter_{token}.safetensors")


def _default_evaluator_factory(path: str):
    """Real loader: auto-detects 24/30-channel, wraps in LocalGPUEvaluator.
    Imported lazily so fake-evaluator tests need no MLX.

    compile=True: reuses the MLX computation graph across calls to prevent
    Metal resource exhaustion during long sequential eval runs (see
    local_evaluator module docstring for details).
    """
    from .probe_eval import load_network_for_scoring
    from .local_evaluator import LocalGPUEvaluator
    net, _in_ch, _hidden, _blocks = load_network_for_scoring(path, verbose=False)
    return LocalGPUEvaluator(net, compile=True)


def _sorted(results):
    return sorted(results, key=lambda r: (r.pairing_id, r.game_idx))


def _make_cache(factory):
    cache: dict = {}

    def get_eval(path):
        ev = cache.get(path)
        if ev is None:
            ev = factory(path)
            cache[path] = ev
        return ev

    return get_eval


def _run_sequential(tasks, config, factory):
    import gc

    import mlx.core as mx

    get_eval = _make_cache(factory)
    results = []
    for task in tasks:
        red = get_eval(task.red_checkpoint)
        black = get_eval(task.black_checkpoint)
        winner, reason, nm = play_eval_game(red, black, config, task.seed)
        results.append(make_result(task, winner, reason, nm))
        # Flush pending MLX lazy ops and release cached Metal buffers between
        # games to stay within Metal's resource limit (trainer.py:3169-3173).
        mx.eval()
        gc.collect()
        mx.clear_cache()
    return _sorted(results)


def _worker_main(worker_id, tasks, config, factory, next_idx, result_q):
    """Pull tasks via the shared atomic counter; per-process checkpoint cache.

    On any exception, send a _WorkerFailed sentinel so the parent fails
    promptly instead of waiting out the stall timeout.
    """
    import traceback
    get_eval = _make_cache(factory)
    n = len(tasks)
    try:
        while True:
            with next_idx.get_lock():
                i = next_idx.value
                if i >= n:
                    break
                next_idx.value = i + 1
            task = tasks[i]
            red = get_eval(task.red_checkpoint)
            black = get_eval(task.black_checkpoint)
            winner, reason, nm = play_eval_game(red, black, config, task.seed)
            result_q.put(make_result(task, winner, reason, nm))
    except Exception as e:
        result_q.put(_WorkerFailed(worker_id, f"{e!r}\n{traceback.format_exc()}"))
        return
    result_q.put(_WorkerDone(worker_id))


def _run_parallel(tasks, workers, config, factory):
    """Spawn pool (macOS-mandatory). Shared next-task counter, results via
    queue, explicit WorkerDone, parent joins with timeout (no silent hang).
    A _WorkerFailed sentinel surfaces a crashed worker promptly."""
    ctx = mp.get_context("spawn")
    next_idx = ctx.Value("i", 0)
    result_q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker_main,
                    args=(wid, tasks, config, factory, next_idx, result_q))
        for wid in range(workers)
    ]
    for p in procs:
        p.start()

    def _terminate_all():
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=5)

    GET_TIMEOUT = 600  # seconds without progress => assume stall
    results = []
    done = 0
    while done < workers:
        try:
            msg = result_q.get(timeout=GET_TIMEOUT)
        except queue.Empty:
            dead = [(p.pid, p.exitcode) for p in procs
                    if p.exitcode not in (None, 0)]
            _terminate_all()
            raise RuntimeError(
                f"eval workers stalled (>{GET_TIMEOUT}s, no result); "
                f"crashed={dead}"
            )
        if isinstance(msg, _WorkerFailed):
            _terminate_all()
            raise RuntimeError(
                f"eval worker {msg.worker_id} crashed: {msg.error}"
            )
        if isinstance(msg, _WorkerDone):
            done += 1
        else:
            results.append(msg)

    for p in procs:
        p.join(timeout=GET_TIMEOUT)

    if len(results) != len(tasks):
        raise RuntimeError(
            f"expected {len(tasks)} results, collected {len(results)}"
        )
    return _sorted(results)


def run_game_tasks(tasks, workers: int, config: EvalConfig,
                   evaluator_factory: Optional[EvaluatorFactory] = None):
    """Execute tasks; return results sorted by (pairing_id, game_idx).

    workers<=1 runs in-process. workers>1 uses a spawn worker pool with a
    shared atomic task counter (dynamic work-stealing).

    NOTE: when workers>1, evaluator_factory must be a MODULE-LEVEL picklable
    callable (it is sent to spawned workers). Lambdas/closures will fail to
    pickle. The default real loader and the test fakes satisfy this.
    """
    factory = evaluator_factory or _default_evaluator_factory
    if not tasks:
        return []
    workers = min(workers, len(tasks))
    if workers <= 1:
        return _run_sequential(tasks, config, factory)
    return _run_parallel(tasks, workers, config, factory)
