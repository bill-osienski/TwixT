"""MIRROR AVERAGING + ROOT SEQUENTIAL HALVING -- the one fixed combination, shared by its two gated entries (C1,
`mirror_halving_c1`; the screen and the strength comparison, `mirror_halving_match`). Nothing runs on import, and
importing this module loads no MLX.

Card: `docs/superpowers/2026-10-26-mirror-averaging-plus-root-halving-combination-card.md` (§2 the candidate, §4 the
accounting, §5 the steps). The candidate, frozen without tuning:
* evaluator: `mirror_averaging.MirrorAveragingEvaluator` over its own eval-mode load of calib020_0001;
* search: `root_sequential_halving.RootSequentialHalvingMCTS` (ordinary search and readout through ply 19, the
  unchanged halving rule and its `last_choice` from ply 20) at CANDIDATE_SIMS = 200;
* opponent: the unchanged incumbent, `mcts.MCTS` over a plain evaluator at INCUMBENT_SIMS = 400.

THE GAME LOOP (play_game / play_and_build) is `docs/superpowers/evidence/root-halving-match/01_match.py`'s, copied
unchanged: a copy of eval_runner.play_eval_game / _play_and_build_result with three marked differences (each colour's
MCTS class from its agent; the move is the search's last_choice when it supplies one, else the readout; an optional
on_search callback). The averaging needs no loop change: the candidate's evaluator is the wrapper.

COST (card §4): a network evaluation is one position evaluated by an UNDERLYING network call (`instrument` wraps the
plain evaluator's infer, which the averaging wrapper calls twice per leaf); search seconds are the wall time inside
search_with_root. Each guard is a RATIO OF TOTALS, candidate over incumbent.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

CARD = "docs/superpowers/2026-10-26-mirror-averaging-plus-root-halving-combination-card.md"
CANDIDATE, INCUMBENT = "candidate", "incumbent"
CANDIDATE_SIMS, INCUMBENT_SIMS = 200, 400
#: root_sequential_halving's frozen constants, as the card names them (checked, never set, by `check_variant`)
VARIANT = {"FROM_PLY": 20, "CANDIDATES": 16, "C_VISIT": 50.0, "C_SCALE": 1.0, "PRIOR_FLOOR": 1e-12, "BOARD": 24}
GUARDS = {"evaluations": 1.05, "search_seconds": 1.10}
SCREEN = {"pairs": 32, "stop_below": 0.40}
STRENGTH = {"pairs": 509, "alpha": 0.05, "null": 0.5}
COST_KEYS = ("searches", "halving_searches", "search_seconds", "network_infer_calls", "network_positions",
             "network_seconds")


class PassThroughEvaluator:
    """The identity control's wrapper (card §5, C1 control (b)): every call goes to `base`, unchanged."""

    def __init__(self, base):
        self.base = base
        self.network = getattr(base, "network", None)

    def build_input_tensor(self, state):
        return self.base.build_input_tensor(state)

    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        return self.base.infer(boards, move_rows, move_cols, move_mask, active_size)


def check_variant(rsh) -> Dict[str, Any]:
    """The halving module's constants must be the card's; returns them as read."""
    got = {k: getattr(rsh, k) for k in VARIANT}
    if got != VARIANT:
        raise ValueError(f"root_sequential_halving's constants {got} are not the card's {VARIANT}")
    return got


def candidate_search(base, config, rng):
    """The candidate's search object: halving over the averaged evaluator at 200 simulations."""
    from .eval_runner import cfg_from
    from .mirror_averaging import MirrorAveragingEvaluator
    from .root_sequential_halving import RootSequentialHalvingMCTS
    return RootSequentialHalvingMCTS(MirrorAveragingEvaluator(base), cfg_from(config, CANDIDATE_SIMS), rng)


def incumbent_search(base, config, rng):
    from .eval_runner import cfg_from
    from .mcts import MCTS
    return MCTS(base, cfg_from(config, INCUMBENT_SIMS), rng)


# ─────────────────────────────── the game loop (copied unchanged) ───────────────────────────────

def play_game(red_eval, black_eval, config, seed, capture=False, red_readout=None, black_readout=None,
              red_sims=None, black_sims=None, red_cls=None, black_cls=None, on_search=None):
    """eval_runner.play_eval_game, copied; the three differences are marked DIFF."""
    from scripts.GPU.alphazero import eval_readout
    from scripts.GPU.alphazero.eval_integrity import validate_ply
    from scripts.GPU.alphazero.eval_replay import ply_record
    from scripts.GPU.alphazero.eval_runner import cfg_from, readout_from_eval_config, root_child_stats
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    from scripts.GPU.alphazero.mcts import MCTS

    red_rd = red_readout or readout_from_eval_config(config)
    black_rd = black_readout or readout_from_eval_config(config)
    red_n = config.mcts_sims if red_sims is None else red_sims
    black_n = config.mcts_sims if black_sims is None else black_sims
    mcts_red = (red_cls or MCTS)(red_eval, cfg_from(config, red_n), random.Random(seed ^ 0xA5A5A5))        # DIFF
    mcts_black = (black_cls or MCTS)(black_eval, cfg_from(config, black_n), random.Random(seed ^ 0x5A5A5A))  # DIFF
    readout_rng_red = random.Random(seed ^ 0xC3C3C3)
    readout_rng_black = random.Random(seed ^ 0x3C3C3C)
    state = TwixtState(active_size=config.board_size, to_move="red",
                       max_plies_limit=config.max_moves)
    ply = 0
    records = [] if capture else None
    while state.winner() is None and ply < config.max_moves and state.legal_moves():
        is_red = state.to_move == "red"
        mcts = mcts_red if is_red else mcts_black
        rdt = red_rd if is_red else black_rd
        rng = readout_rng_red if is_red else readout_rng_black
        t = time.perf_counter()                                                                     # DIFF
        counts, root_value, root = mcts.search_with_root(state, add_noise=False)
        choice = getattr(mcts, "last_choice", None)                                                 # DIFF
        if on_search is not None:                                                                   # DIFF
            on_search(is_red, time.perf_counter() - t, choice is not None)
        top2 = eval_readout.top_two(root_child_stats(counts, root))
        validate_ply(ply, red_n if is_red else black_n, root.visit_count,
                     root_value, top2)
        if choice is None:                                                                          # DIFF
            move, overrode = eval_readout.select(counts, ply, rdt, rng, top2=top2)
        else:
            move, overrode = choice, False
        if capture:
            records.append(ply_record(ply, state.to_move, move, counts,
                                      root_value, top2=top2,
                                      overrode_leader=overrode))
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


def play_and_build(task, red, black, config, capture, replay_dir, classes=None, on_search=None):
    """eval_runner._play_and_build_result, copied; it calls play_game with each agent's MCTS class (DIFF)."""
    from scripts.GPU.alphazero.eval_integrity import validate_game_binding
    from scripts.GPU.alphazero.eval_replay import build_replay_dict, write_replay
    from scripts.GPU.alphazero.eval_runner import agent_sims, make_result

    classes = classes or {}
    red_rd = task.red_agent.readout if task.red_agent is not None else None
    black_rd = task.black_agent.readout if task.black_agent is not None else None
    winner, reason, nm, records = play_game(
        red, black, config, task.seed, capture=capture,
        red_readout=red_rd, black_readout=black_rd,
        red_sims=agent_sims(task.red_agent, config),
        black_sims=agent_sims(task.black_agent, config),
        red_cls=classes.get(task.red_agent.agent_id), black_cls=classes.get(task.black_agent.agent_id),
        on_search=on_search)
    result = make_result(task, winner, reason, nm, config=config)
    validate_game_binding(result, task)
    if records is not None:
        result.replay_path = write_replay(
            replay_dir,
            build_replay_dict(result, task.seed, config.board_size, records))
    return result


# ─────────────────────────────── cost (card §4) ───────────────────────────────

def new_cost() -> Dict[str, float]:
    return dict.fromkeys(COST_KEYS, 0)


def instrument(base, cost: Dict[str, float]) -> None:
    """Count every UNDERLYING network call of `base` into `cost` (read-only: the outputs pass through unchanged)."""
    real = base.infer

    def net_infer(boards, *a, **kw):
        t = time.perf_counter()
        out = real(boards, *a, **kw)
        cost["network_seconds"] += time.perf_counter() - t
        cost["network_infer_calls"] += 1
        cost["network_positions"] += int(boards.shape[0])
        return out

    base.infer = net_infer


def delta(after: Mapping[str, float], before: Mapping[str, float]) -> Dict[str, float]:
    return {k: after[k] - before[k] for k in COST_KEYS}


def guards(candidate: Mapping[str, float], incumbent: Mapping[str, float]) -> Dict[str, Any]:
    """Ratios of TOTALS, candidate over incumbent, against the frozen guards (card §4)."""
    if incumbent["network_positions"] <= 0 or incumbent["search_seconds"] <= 0:
        raise ValueError(f"the incumbent's totals are not positive: {dict(incumbent)}")
    ev = candidate["network_positions"] / incumbent["network_positions"]
    sec = candidate["search_seconds"] / incumbent["search_seconds"]
    return {"evaluations_ratio": ev, "search_seconds_ratio": sec,
            "evaluations_within": ev <= GUARDS["evaluations"], "search_seconds_within": sec <= GUARDS["search_seconds"],
            "guards": dict(GUARDS)}


# ─────────────────────────────── scores and verdicts (card §5) ───────────────────────────────

def pair_scores(results: Sequence[Any], n_pairs: int) -> list:
    """The candidate's score per colour-reversed pair (games 2k, 2k+1); a state cap or other draw scores 0.5."""
    if sorted(r.game_idx for r in results) != list(range(2 * n_pairs)):
        raise ValueError(f"incomplete cohort: {len(results)} of {2 * n_pairs} games -- no score")
    s = {r.game_idx: r.red_score if r.red_agent_id == CANDIDATE else r.black_score for r in results}
    return [(s[2 * k] + s[2 * k + 1]) / 2 for k in range(n_pairs)]


def screen_verdict(pairs: Sequence[float]) -> Dict[str, Any]:
    """Gross regression only: STOP below 0.40; otherwise CONTINUE. No strength claim either way."""
    if len(pairs) != SCREEN["pairs"]:
        raise ValueError(f"{len(pairs)} pairs, not the screen's {SCREEN['pairs']}")
    mean = math.fsum(pairs) / len(pairs)
    return {"pairs": len(pairs), "mean_pair_score": mean, "stop_below": SCREEN["stop_below"],
            "verdict": "STOP" if mean < SCREEN["stop_below"] else "CONTINUE",
            "note": "gross-regression check only; not a strength result"}


def hoeffding_lower(pairs: Sequence[float], alpha: float) -> float:
    """mean - sqrt(ln(2 / alpha) / (2 n)), in float64: the exact formula of card §5 T (pair scores in [0, 1])."""
    n = len(pairs)
    return math.fsum(pairs) / n - math.sqrt(math.log(2.0 / alpha) / (2.0 * n))


def strength_verdict(pairs: Sequence[float], g: Mapping[str, Any]) -> Dict[str, Any]:
    """PASS iff the exact Hoeffding lower bound exceeds 0.5 AND both guards hold over the whole run."""
    if len(pairs) != STRENGTH["pairs"]:
        raise ValueError(f"{len(pairs)} pairs, not the strength comparison's {STRENGTH['pairs']}")
    lower = hoeffding_lower(pairs, STRENGTH["alpha"])
    above = lower > STRENGTH["null"]
    matched = bool(g["evaluations_within"] and g["search_seconds_within"])
    verdict = "PASS" if above and matched else "NOT_COMPUTE_MATCHED" if above else "NON_PASS"
    return {"pairs": len(pairs), "mean_pair_score": math.fsum(pairs) / len(pairs), "hoeffding_lower": lower,
            "alpha": STRENGTH["alpha"], "bound_above_null": above, "compute_matched": matched, "verdict": verdict,
            "note": "pairs assumed independent; a non-pass is not designed to reliably detect modest gains"}


def readout_choice(counts, root, ply, readout, rng: random.Random) -> Optional[tuple]:
    """What the existing readout would choose from this search (diagnostics only; uses its own RNG)."""
    from scripts.GPU.alphazero import eval_readout
    from scripts.GPU.alphazero.eval_runner import root_child_stats
    top2 = eval_readout.top_two(root_child_stats(counts, root))
    return eval_readout.select(counts, ply, readout, rng, top2=top2)[0]


def timed_search(make: Callable[[random.Random], Any], state, seed: int):
    """One search with a fresh random.Random(seed): (search object, counts, root value, root, seconds)."""
    m = make(random.Random(seed))
    t = time.perf_counter()
    counts, value, root = m.search_with_root(state, add_noise=False)
    return m, counts, value, root, time.perf_counter() - t
