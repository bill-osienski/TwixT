"""Shared executor for checkpoint-tournament games.

A flat list of EvalGameTask is drained by run_game_tasks, which both the
match and tournament scripts call. workers==1 runs in-process; workers>1
uses a spawn worker pool with a shared atomic task counter (the trainer
idiom). Determinism is task-derived: same base_seed + schedule => same
games regardless of worker count.
"""
from __future__ import annotations

import dataclasses
import multiprocessing as mp
import os
import queue
import random
from dataclasses import dataclass
from typing import Callable, Optional

from .game.twixt_state import TwixtState
from .mcts import MCTS, MCTSConfig
from .eval_replay import ply_record, build_replay_dict, write_replay

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
    # Optional per-agent search config. None (default) => both sides use
    # cfg_from(config), i.e. the historical symmetric behaviour, byte-for-byte.
    # Set when two agents share identical checkpoint bytes but must search
    # differently (v17: same net, different FPU coefficient) -- a case the
    # checkpoint-swap color balance alone cannot express, because swapping a
    # path for itself is a no-op.
    red_mcts: Optional[MCTSConfig] = None
    black_mcts: Optional[MCTSConfig] = None
    # Agent identity, independent of checkpoint. Two agents on ONE checkpoint
    # are distinguishable only by this; without it every downstream score
    # collapses to "self-match" and no strength statistic can be computed.
    red_agent: Optional[str] = None
    black_agent: Optional[str] = None


def is_agent_task(task: EvalGameTask) -> bool:
    """True once a task carries any agent-mode field.

    Deliberately ANY, not all: a task that is half-configured is exactly the
    silent-fallback case validation has to catch, so it must count as agent
    mode rather than slipping back into the legacy path.
    """
    return any(v is not None for v in
               (task.red_mcts, task.black_mcts, task.red_agent, task.black_agent))


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
    replay_path: Optional[str] = None
    # Agent identity carried through to scoring. None on the legacy path, and
    # omitted entirely from the per-game JSONL there (see _write_outputs), so
    # existing artifacts keep their exact bytes.
    red_agent: Optional[str] = None
    black_agent: Optional[str] = None
    winner_agent: Optional[str] = None


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


def play_eval_game(red_eval, black_eval, config: EvalConfig, seed: int,
                   capture: bool = False,
                   red_mcts: Optional[MCTSConfig] = None,
                   black_mcts: Optional[MCTSConfig] = None):
    """Play one A-vs-B game. Returns (winner, reason, n_moves, records).

    `records` is None unless capture=True, in which case it is a list of
    ply_record dicts (one per ply). Capturing reads already-computed search
    outputs only — no extra search calls, no RNG draws — so game outcomes are
    identical with capture on or off.

    `red_mcts`/`black_mcts` override that side's search config. Each defaults to
    None => cfg_from(config), which is exactly what both sides used before this
    parameter existed. Seeds are unchanged and stay side-derived, so overriding
    one side does not disturb the other side's RNG stream.
    """
    red_cfg = cfg_from(config) if red_mcts is None else red_mcts
    black_cfg = cfg_from(config) if black_mcts is None else black_mcts
    mcts_red = MCTS(red_eval, red_cfg, random.Random(seed ^ 0xA5A5A5))
    mcts_black = MCTS(black_eval, black_cfg, random.Random(seed ^ 0x5A5A5A))
    state = TwixtState(active_size=config.board_size, to_move="red",
                       max_plies_limit=config.max_moves)
    ply = 0
    records = [] if capture else None
    while state.winner() is None and ply < config.max_moves and state.legal_moves():
        mcts = mcts_red if state.to_move == "red" else mcts_black
        counts, root_value = mcts.search(state, add_noise=False)
        move = mcts.select_move(counts, ply)
        if capture:
            records.append(ply_record(ply, state.to_move, move, counts, root_value))
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
    return winner, reason, ply, records


def make_result(task: EvalGameTask, winner, reason, n_moves,
                replay_path=None) -> EvalGameResult:
    """Build a result, mapping winner color -> checkpoint and 0/0.5/1 scores."""
    if winner == "red":
        red_score, black_score, winner_ckpt = 1.0, 0.0, task.red_checkpoint
        winner_agent = task.red_agent
    elif winner == "black":
        red_score, black_score, winner_ckpt = 0.0, 1.0, task.black_checkpoint
        winner_agent = task.black_agent
    else:
        red_score, black_score, winner_ckpt = 0.5, 0.5, None
        winner_agent = None
    return EvalGameResult(
        task_id=task.task_id, pairing_id=task.pairing_id, game_idx=task.game_idx,
        red_checkpoint=task.red_checkpoint, black_checkpoint=task.black_checkpoint,
        winner=winner, winner_checkpoint=winner_ckpt, reason=reason,
        n_moves=n_moves, red_score=red_score, black_score=black_score,
        replay_path=replay_path,
        red_agent=task.red_agent, black_agent=task.black_agent,
        winner_agent=winner_agent,
    )


def build_pairing_tasks(pairing_id, a_ckpt, b_ckpt, games, base_seed, pairing_index,
                        a_mcts: Optional[MCTSConfig] = None,
                        b_mcts: Optional[MCTSConfig] = None,
                        a_agent: Optional[str] = None,
                        b_agent: Optional[str] = None):
    """Balanced-color tasks for one pairing. Even game_idx -> red=A; odd -> red=B.
    task_id and seed are task-derived (stable across worker counts).

    An agent is the TRIPLE (id, checkpoint, search config). The color swap below
    moves all three together in one expression, so nothing can be left attached
    to a color instead of to its agent. The id is what makes two agents on ONE
    checkpoint distinguishable downstream.

    Supplying any of `a_mcts`/`b_mcts`/`a_agent`/`b_agent` turns on agent mode,
    which requires BOTH search configs explicitly: an omitted config would
    silently mean "the base", indistinguishable from one that was dropped by
    mistake. Ids default to "A"/"B". All four omitted => legacy tasks, identical
    to before.
    """
    agent_mode = any(v is not None for v in (a_mcts, b_mcts, a_agent, b_agent))
    if agent_mode:
        if a_mcts is None or b_mcts is None:
            raise ValueError(
                "agent mode requires BOTH a_mcts and b_mcts; pass the base "
                "config explicitly for an agent that is not varying, so that "
                "'uses the base' is never confused with 'config was dropped'")
        a_agent = "A" if a_agent is None else a_agent
        b_agent = "B" if b_agent is None else b_agent
        for agent_id in (a_agent, b_agent):
            # Ids become JSON object keys in the match provenance. A non-string
            # would be silently coerced there (1 -> "1") and no longer match the
            # id on the task, so refuse it at construction.
            if not isinstance(agent_id, str) or not agent_id:
                raise ValueError(
                    f"agent id must be a non-empty string, got {agent_id!r}")
        if a_agent == b_agent:
            raise ValueError(
                f"agents must have distinct ids, got {a_agent!r} for both")
    if games < 2:
        raise ValueError("games must be >= 2")
    if games % 2 != 0:
        # Color balancing assigns A=red on even game_idx, A=black on odd.
        # An odd count gives one model an extra red game -> biased.
        raise ValueError("games must be even for balanced colors")
    if games >= GAMES_PER_PAIRING_LIMIT:
        raise ValueError(f"games must be < {GAMES_PER_PAIRING_LIMIT}")
    offset = pairing_index * GAMES_PER_PAIRING_LIMIT
    agent_a = (a_ckpt, a_mcts, a_agent)
    agent_b = (b_ckpt, b_mcts, b_agent)
    tasks = []
    for g in range(games):
        # One swap of whole agents -- never a separate swap of checkpoints, of
        # configs and of ids, which could drift out of step.
        (red, red_cfg, red_id), (black, black_cfg, black_id) = (
            (agent_a, agent_b) if g % 2 == 0 else (agent_b, agent_a))
        tasks.append(EvalGameTask(
            task_id=offset + g, pairing_id=pairing_id, game_idx=g,
            red_checkpoint=red, black_checkpoint=black, seed=base_seed + offset + g,
            red_mcts=red_cfg, black_mcts=black_cfg,
            red_agent=red_id, black_agent=black_id,
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
    # Eval workers opt into compile=True to avoid repeated MLX/Metal resource churn.
    # Training path keeps LocalGPUEvaluator default compile=False.
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


def require_agent_config_consistency(base: MCTSConfig,
                                     a_mcts: Optional[MCTSConfig],
                                     b_mcts: Optional[MCTSConfig],
                                     allow_differ=(),
                                     labels=("a", "b")) -> None:
    """Refuse per-agent configs that differ from `base` outside `allow_differ`.

    Two agents in one match must be comparable: everything about the search --
    simulation count, batching, temperatures -- has to match the base config,
    or a measured difference cannot be attributed to the field under study.
    `allow_differ` names the fields the caller is deliberately varying.

    Generic on purpose: the caller supplies the policy (which field is under
    study), so this module needs no knowledge of any particular experiment.
    Call it at task-construction time -- before any evaluator is loaded -- so a
    misconfigured run costs nothing.
    """
    allowed = set(allow_differ)
    if unknown := sorted(allowed - {f.name for f in dataclasses.fields(base)}):
        raise ValueError(
            f"allow_differ names fields that are not on MCTSConfig: {unknown}")
    base_fields = dataclasses.asdict(base)
    for label, cfg in zip(labels, (a_mcts, b_mcts)):
        if cfg is None:
            continue
        if not isinstance(cfg, MCTSConfig):
            raise TypeError(
                f"agent {label} config must be an MCTSConfig, got "
                f"{type(cfg).__name__}")
        differing = sorted(
            name for name, value in dataclasses.asdict(cfg).items()
            if name not in allowed and base_fields.get(name) != value)
        if differing:
            raise ValueError(
                f"agent {label} search config differs from the base config in "
                f"{differing}, which is not in allow_differ={sorted(allowed)}; "
                f"agents must be identical apart from the field under study")


def require_consistent_agent_tasks(tasks, base: MCTSConfig, allow_differ=(),
                                   config_validator=None) -> None:
    """Validate a COMPLETE task list under agent mode, before any evaluator load.

    Per-task checking is not enough. The failures that matter are properties of
    the whole list: one task quietly left on the base config, an agent whose
    checkpoint or config drifts between games, a color assignment that stops
    alternating. Each of those keeps every individual task well-formed while
    destroying the comparison, so the list is validated as a unit.

    `config_validator` is called on the base config and on every agent's config
    — the caller's chance to enforce constraints this module has no business
    knowing (v17 passes the frozen batching triple validator). It runs even
    with no agent tasks, because a wrong base is wrong either way.
    """
    if config_validator is not None:
        config_validator(base)
    configured = [t for t in tasks if is_agent_task(t)]
    if not configured:
        return
    if len(configured) != len(tasks):
        plain = sorted(t.task_id for t in tasks if not is_agent_task(t))
        raise ValueError(
            f"agent mode is active but tasks {plain} carry no agent fields; "
            f"they would silently run base-vs-base and break the comparison")

    by_pairing: dict = {}
    for task in tasks:
        by_pairing.setdefault(task.pairing_id, []).append(task)

    for pairing_id, group in by_pairing.items():
        agents: dict = {}
        for task in group:
            sides = ((task.red_agent, task.red_checkpoint, task.red_mcts),
                     (task.black_agent, task.black_checkpoint, task.black_mcts))
            for agent_id, ckpt, cfg in sides:
                if agent_id is None or cfg is None:
                    raise ValueError(
                        f"task {task.task_id} is only half-configured "
                        f"(agent={agent_id!r}, config set={cfg is not None}); "
                        f"agent mode requires both on both sides")
                if agents.setdefault(agent_id, (ckpt, cfg)) != (ckpt, cfg):
                    raise ValueError(
                        f"agent {agent_id!r} is not defined consistently across "
                        f"pairing {pairing_id!r}: its checkpoint or search "
                        f"config changes between games")
            if task.red_agent == task.black_agent:
                raise ValueError(
                    f"task {task.task_id} has agent {task.red_agent!r} on both "
                    f"sides")
        if len(agents) != 2:
            raise ValueError(
                f"pairing {pairing_id!r} must have exactly 2 agents, found "
                f"{sorted(agents)}")

        # Exact color assignment: the agent that is red on even game_idx must be
        # red on every even game_idx and black on every odd one.
        ordered = sorted(group, key=lambda t: t.game_idx)
        even = [t for t in ordered if t.game_idx % 2 == 0]
        if not even:
            raise ValueError(
                f"pairing {pairing_id!r} has no even game_idx; color balance "
                f"cannot be established")
        first_red = even[0].red_agent
        other = next(a for a in agents if a != first_red)
        for task in ordered:
            expected = first_red if task.game_idx % 2 == 0 else other
            if task.red_agent != expected:
                raise ValueError(
                    f"task {task.task_id} (game_idx {task.game_idx}) has "
                    f"{task.red_agent!r} as red; exact color balance requires "
                    f"{expected!r}")
        red_counts = [sum(1 for t in ordered if t.red_agent == a) for a in agents]
        if len(set(red_counts)) != 1:
            raise ValueError(
                f"pairing {pairing_id!r} color balance is uneven: "
                f"{dict(zip(agents, red_counts))}")

        for agent_id, (_ckpt, cfg) in sorted(agents.items()):
            require_agent_config_consistency(base, cfg, None,
                                             allow_differ=allow_differ,
                                             labels=(agent_id, None))
            if config_validator is not None:
                config_validator(cfg)


def _play_and_build_result(task, red, black, config, capture, replay_dir):
    """Play one game and build its result, writing a replay sidecar when
    capturing. Shared by the sequential and worker loops (both single-process)."""
    winner, reason, nm, records = play_eval_game(
        red, black, config, task.seed, capture=capture,
        red_mcts=task.red_mcts, black_mcts=task.black_mcts)
    result = make_result(task, winner, reason, nm)
    if records is not None:
        result.replay_path = write_replay(
            replay_dir,
            build_replay_dict(result, task.seed, config.board_size, records))
    return result


def _run_sequential(tasks, config, factory, replay_dir=None):
    import gc

    import mlx.core as mx

    capture = replay_dir is not None
    get_eval = _make_cache(factory)
    results = []
    for task in tasks:
        red = get_eval(task.red_checkpoint)
        black = get_eval(task.black_checkpoint)
        results.append(
            _play_and_build_result(task, red, black, config, capture, replay_dir))
        # Flush pending MLX lazy ops and release cached Metal buffers between
        # games to stay within Metal's resource limit (trainer.py:3169-3173).
        mx.eval()
        gc.collect()
        mx.clear_cache()
    return _sorted(results)


def _worker_main(worker_id, tasks, config, factory, next_idx, result_q,
                 replay_dir=None):
    """Pull tasks via the shared atomic counter; per-process checkpoint cache.

    On any exception, send a _WorkerFailed sentinel so the parent fails
    promptly instead of waiting out the stall timeout.
    """
    import traceback
    capture = replay_dir is not None
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
            result_q.put(
                _play_and_build_result(task, red, black, config, capture, replay_dir))
    except Exception as e:
        result_q.put(_WorkerFailed(worker_id, f"{e!r}\n{traceback.format_exc()}"))
        return
    result_q.put(_WorkerDone(worker_id))


def _run_parallel(tasks, workers, config, factory, replay_dir=None):
    """Spawn pool (macOS-mandatory). Shared next-task counter, results via
    queue, explicit WorkerDone, parent joins with timeout (no silent hang).
    A _WorkerFailed sentinel surfaces a crashed worker promptly."""
    ctx = mp.get_context("spawn")
    next_idx = ctx.Value("i", 0)
    result_q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker_main,
                    args=(wid, tasks, config, factory, next_idx, result_q,
                          replay_dir))
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
                f"crashed={dead}. If this is a Metal/MLX resource limit, "
                f"re-run with --workers 1 (sequential is fully valid, just slower)."
            )
        if isinstance(msg, _WorkerFailed):
            _terminate_all()
            raise RuntimeError(
                f"eval worker {msg.worker_id} crashed: {msg.error}\n"
                f"If this is a Metal/MLX resource limit, re-run with --workers 1 "
                f"(sequential is fully valid, just slower)."
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
                   evaluator_factory: Optional[EvaluatorFactory] = None,
                   replay_dir: Optional[str] = None,
                   allow_differ=(), config_validator=None):
    """Execute tasks; return results sorted by (pairing_id, game_idx).

    workers<=1 runs in-process. workers>1 uses a spawn worker pool with a
    shared atomic task counter (dynamic work-stealing).

    When replay_dir is set, each game writes a per-ply replay sidecar into it
    (worker-safe: each game writes its own file) and the result's replay_path
    is filled in; otherwise replay_path stays None.

    NOTE: when workers>1, evaluator_factory must be a MODULE-LEVEL picklable
    callable (it is sent to spawned workers). Lambdas/closures will fail to
    pickle. The default real loader and the test fakes satisfy this.

    The complete task list is validated here, before any evaluator loads. This
    is the backstop for callers that build tasks themselves rather than going
    through run_match; the default empty `allow_differ` refuses ANY divergence,
    so a direct caller has to state which field it is varying instead of
    getting silence. `config_validator` lets the caller additionally constrain
    the base and every agent config (v17: the frozen batching triple).
    """
    if not tasks:
        return []
    require_consistent_agent_tasks(tasks, cfg_from(config),
                                   allow_differ=allow_differ,
                                   config_validator=config_validator)
    factory = evaluator_factory or _default_evaluator_factory
    workers = min(workers, len(tasks))
    if workers <= 1:
        return _run_sequential(tasks, config, factory, replay_dir)
    return _run_parallel(tasks, workers, config, factory, replay_dir)
