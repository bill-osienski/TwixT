from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..replay.recorder import ReplayPaths, new_record, write_game_record
from ..tuning.hasher import config_hash
from ..utils.jsonl import append_jsonl
from .engine import SimOutcome, TwixtSimulator
from .results import GameSummary, summarize


# Default worker count based on available cores
# For M3 with 18 GPU cores, we want enough workers to keep GPU busy
# but not so many that we cause memory pressure
DEFAULT_WORKERS = min(12, max(4, (os.cpu_count() or 4) - 2))


def _play_one_game(args: Tuple[Dict[str, float], int, int, int, str]) -> dict:
    """Worker function for parallel game execution.

    Args:
        args: Tuple of (knobs, seed, depth, board_size, mode)

    Returns:
        Dict with game results (serializable for multiprocessing)
    """
    knobs, seed, depth, board_size, mode = args
    sim = TwixtSimulator(board_size=board_size)
    outcome = sim.play_one(knobs, seed=seed, depth=depth, mode=mode)

    # Convert to serializable dict (can't pickle dataclasses with complex fields)
    return {
        "seed": seed,
        "winner": outcome.winner,
        "total_moves": outcome.total_moves,
        "reason": outcome.reason,
        "starting_player": outcome.starting_player,
        "stats": outcome.stats,
        "moves": [
            {
                "turn": m.turn,
                "player": m.player,
                "row": m.row,
                "col": m.col,
                "search_score": m.search_score,
                "bridges_created": m.bridges_created,
            }
            for m in outcome.moves
        ],
    }


def _outcome_from_dict(d: dict) -> SimOutcome:
    """Reconstruct SimOutcome from serialized dict."""
    from ..replay.format import Move

    moves = [
        Move(
            turn=m["turn"],
            player=m["player"],
            row=m["row"],
            col=m["col"],
            search_score=m.get("search_score"),
            bridges_created=m.get("bridges_created"),
        )
        for m in d["moves"]
    ]
    return SimOutcome(
        winner=d["winner"],
        moves=moves,
        total_moves=d["total_moves"],
        reason=d["reason"],
        starting_player=d.get("starting_player", "red"),
        stats=d.get("stats", {}),
    )


def run_games_parallel(
    *,
    knobs: Dict[str, float],
    depth: int,
    games: int,
    seed: int,
    board_size: int = 24,
    workers: int = DEFAULT_WORKERS,
    mode: str = "training",
) -> List[SimOutcome]:
    """Run N games in parallel using multiprocessing.

    This is the key to utilizing multiple GPU cores effectively.
    Each worker runs a game independently, and the GPU acceleration
    in heuristics evaluation is parallelized across workers.

    Args:
        knobs: Configuration knobs
        depth: Search depth
        games: Number of games to run
        seed: Base random seed
        board_size: Board size
        workers: Number of parallel workers

    Returns:
        List of SimOutcome for each game
    """
    # Prepare arguments for each game
    args_list = [(knobs, seed + i, depth, board_size, mode) for i in range(games)]

    # Use ProcessPoolExecutor for true parallelism (bypasses GIL)
    outcomes: List[SimOutcome] = []

    # For small batches, sequential may be faster (avoid process spawn overhead)
    if games <= 2 or workers <= 1:
        sim = TwixtSimulator(board_size=board_size)
        return sim.play_batch(knobs, seeds=[seed + i for i in range(games)], depth=depth, mode=mode)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_play_one_game, args): args[1] for args in args_list}

        results_by_seed: Dict[int, SimOutcome] = {}
        for future in as_completed(futures):
            seed_val = futures[future]
            try:
                result_dict = future.result()
                results_by_seed[seed_val] = _outcome_from_dict(result_dict)
            except Exception as e:
                # On error, create a draw outcome
                print(f"  Warning: Game seed={seed_val} failed: {e}")
                results_by_seed[seed_val] = SimOutcome(
                    winner="draw",
                    moves=[],
                    total_moves=0,
                    reason="error",
                )

    # Return in original seed order
    return [results_by_seed[seed + i] for i in range(games)]


def run_games(
    *,
    sim: TwixtSimulator,
    knobs: Dict[str, float],
    depth: int,
    games: int,
    seed: int,
    tag: str,
    results_jsonl: Path,
    games_dir: Path,
    predicted_bias: Optional[float] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    parallel: bool = True,
    workers: int = DEFAULT_WORKERS,
    mode: str = "training",
) -> GameSummary:
    """Run N games and write:
    - a JSONL summary row (for sweep/validation aggregation)
    - per-game replay JSON

    Args:
        sim: TwixtSimulator instance (used for board_size, ignored if parallel=True)
        knobs: Configuration knobs
        depth: Search depth
        games: Number of games
        seed: Base random seed
        tag: Tag for logging (e.g., "sweep", "validation")
        results_jsonl: Path to append JSONL results
        games_dir: Directory for game replay files
        predicted_bias: Predicted bias from ridge model
        extra_meta: Extra metadata for replay files
        parallel: Use parallel execution (default True)
        workers: Number of parallel workers
    """

    h = config_hash(knobs)
    seeds = [seed + i for i in range(games)]
    t0 = time.time()

    if parallel and games > 2:
        outcomes = run_games_parallel(
            knobs=knobs,
            depth=depth,
            games=games,
            seed=seed,
            board_size=sim.board_size,
            workers=workers,
            mode=mode,
        )
    else:
        outcomes = sim.play_batch(knobs, seeds=seeds, depth=depth, mode=mode)

    dt = time.time() - t0

    # Write replay files
    rp = ReplayPaths(games_dir=games_dir)
    for s, out in zip(seeds, outcomes):
        rec = new_record(
            config_hash=h,
            depth=depth,
            seed=s,
            winner=out.winner,
            moves=out.moves,
            meta={
                "tag": tag,
                "predicted_bias": predicted_bias,
                "elapsed_s": dt / max(1, games),
                "board_size": sim.board_size,
                "mode": mode,
                "stats": out.stats,
                **(extra_meta or {}),
            },
        )
        write_game_record(rp, rec)

    summ = summarize(outcomes)
    reason_counts = {
        "win": sum(1 for o in outcomes if o.reason == "win"),
        "stall": sum(1 for o in outcomes if o.reason == "stall"),
        "max_moves": sum(1 for o in outcomes if o.reason == "max_moves"),
        "no_moves": sum(1 for o in outcomes if o.reason == "no_moves"),
        "error": sum(1 for o in outcomes if o.reason == "error"),
    }
    avg_moves = (
        sum(int(o.total_moves) for o in outcomes) / max(1, len(outcomes))
    )
    avg_stagnation_max = (
        sum((o.stats or {}).get("stagnation_max", 0) for o in outcomes) / max(1, len(outcomes))
    )
    avg_progress_events = (
        sum((o.stats or {}).get("progress_events", 0) for o in outcomes) / max(1, len(outcomes))
    )
    avg_opening_random = (
        sum((o.stats or {}).get("opening_random_moves", 0) for o in outcomes) / max(1, len(outcomes))
    )
    score_vals = [
        (o.stats or {}).get("avg_search_score")
        for o in outcomes
        if (o.stats or {}).get("avg_search_score") is not None
    ]
    avg_search_score = sum(score_vals) / max(1, len(score_vals)) if score_vals else None

    append_jsonl(
        results_jsonl,
        {
            "hash": h,
            "tag": tag,
            "depth": int(depth),
            "games": summ.games,
            "red": summ.red,
            "black": summ.black,
            "draws": summ.draws,
            "bias": summ.bias,
            "predicted_bias": predicted_bias,
            "reasons": reason_counts,
            "avg_moves": avg_moves,
            "stats": {
                "avg_stagnation_max": avg_stagnation_max,
                "avg_progress_events": avg_progress_events,
                "avg_opening_random_moves": avg_opening_random,
                "avg_search_score": avg_search_score,
            },
            "knobs": knobs,
            "seed": seed,
            "elapsed_s": dt,
            "workers": workers if parallel else 1,
            "mode": mode,
        },
    )

    return summ
