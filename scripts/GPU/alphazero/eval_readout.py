"""Final move readout for checkpoint-eval games.

Pure: no MLX, no MCTS, no game engine. This module owns every readout rule the
evaluation harness can play, including the FROZEN Hoeffding-LCB override
(design spec section 7.4, frozen 2026-08-06).

Why this lives outside MCTS: `mcts.MCTS` draws prior-shuffle, PUCT tie-break
and move readout from ONE `self.rng`, so changing the readout changes the
generator state entering every subsequent search. Evaluation therefore selects
moves here, with its own RNG stream, and never calls `mcts.select_move`.
Self-play still calls `mcts.select_move` and is unaffected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

Move = Tuple[int, int]

# --- FROZEN constants: do not change (design spec section 7.4) -------------
VALUE_RANGE = 2.0       # backed-up MCTS values lie in [-1, 1]
DELTA = 0.05            # confidence level; sets the radius SCALE only.
                        # NOT the match's statistical alpha, and it carries no
                        # repeated-decision guarantee across many positions.
MIN_CHILD_VISITS = 8    # smallest n with hoeffding_radius(n) <= 1.0, from the
                        # preregistered "radius no wider than half the value
                        # range" requirement. The requirement is a judgement;
                        # only the arithmetic that follows is forced.
_HOEFFDING_NUM = math.log(2.0 / DELTA) / 2.0    # == 1.844439...

MODE_OPENING_TEMPERATURE = "opening_temperature"
MODE_ARGMAX = "argmax"
MODE_HOEFFDING_LCB = "hoeffding_lcb"
MODES = (MODE_OPENING_TEMPERATURE, MODE_ARGMAX, MODE_HOEFFDING_LCB)

# Below this, select_move's own deterministic branch takes over. Matches
# mcts.select_move's threshold so the two agree on what "temperature 0" means.
_DETERMINISTIC_TEMP = 0.01


@dataclass(frozen=True)
class ReadoutConfig:
    """How one agent turns a completed search into a played move."""
    mode: str = MODE_OPENING_TEMPERATURE
    opening_temp_plies: int = 20
    temp_high: float = 1.0
    temp_low: float = 0.1

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(
                f"unknown readout mode {self.mode!r}; expected one of {MODES}")


@dataclass(frozen=True)
class ChildStat:
    """One root child's completed statistics.

    `q_child` is the child's own stored mean; `q_root` is the MOVER's
    perspective and equals `-q_child` (mcts.py:1122). Both are None when the
    mean is UNDEFINED (zero completed visits, or a non-finite value) --
    never 0.0, which is what `MCTSNode.q_value` returns at visit_count == 0.
    """
    move: Move
    visits: int
    q_child: Optional[float]
    q_root: Optional[float]


def hoeffding_radius(n: int) -> float:
    """Hoeffding half-width for the mean of `n` observations spanning
    VALUE_RANGE.

    NOTE: MCTS backups are adaptively sampled and correlated, not i.i.d., so
    this is a principled UNFITTED radius, not a valid confidence guarantee.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return VALUE_RANGE * math.sqrt(_HOEFFDING_NUM / n)


def top_two(stats: Dict[Move, Tuple[int, Optional[float]]]) -> List[ChildStat]:
    """The two children with the highest completed visit counts.

    `stats` maps move -> (completed_visits, child_perspective_mean_or_None).
    Ordering is (-visits, move), so visit ties break in canonical numeric
    (row, col) order. Returns 0, 1 or 2 entries.
    """
    ordered = sorted(stats.items(), key=lambda kv: (-kv[1][0], kv[0]))
    out: List[ChildStat] = []
    for move, (visits, q_child) in ordered[:2]:
        defined = visits > 0 and q_child is not None and math.isfinite(q_child)
        q_c = float(q_child) if defined else None
        out.append(ChildStat(move=move, visits=int(visits), q_child=q_c,
                             q_root=(None if q_c is None else -q_c)))
    return out


def lcb_override(top2: List[ChildStat]) -> Optional[Move]:
    """The FROZEN rule. Returns the challenger's move iff it overrides the
    visit leader; otherwise None, meaning play the leader.

    This is a conservative RANKING HEURISTIC. It does not establish at 95%
    confidence that the challenger is the better move.
    """
    if len(top2) < 2:
        return None
    leader, challenger = top2[0], top2[1]
    if leader.visits < MIN_CHILD_VISITS or challenger.visits < MIN_CHILD_VISITS:
        return None
    if leader.q_root is None or challenger.q_root is None:
        return None
    lcb_leader = leader.q_root - hoeffding_radius(leader.visits)
    lcb_challenger = challenger.q_root - hoeffding_radius(challenger.visits)
    return challenger.move if lcb_challenger > lcb_leader else None


def _argmax(counts: Dict[Move, int]) -> Move:
    """Visit-count argmax with a deterministic canonical (row, col) tie-break.
    No RNG: the readout stream is never consumed on this path."""
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def _sample_by_temperature(counts: Dict[Move, int], temp: float, rng) -> Move:
    """Sample proportional to count^(1/temp).

    Mirrors mcts.select_move's log-count softmax exactly, including the 1e-8
    floor and the max-subtraction, so the two implementations agree on the
    distribution (they do NOT agree game-for-game, because the streams differ).
    """
    moves = list(counts.keys())
    log_counts = [math.log(counts[m] + 1e-8) / temp for m in moves]
    max_log = max(log_counts)
    exp_counts = [math.exp(lc - max_log) for lc in log_counts]
    total = sum(exp_counts)
    r = rng.random()
    cumsum = 0.0
    for move, e in zip(moves, exp_counts):
        cumsum += e / total
        if r <= cumsum:
            return move
    return moves[-1]


def _opening_move(counts: Dict[Move, int], temp: float, rng) -> Move:
    if temp < _DETERMINISTIC_TEMP:
        return _argmax(counts)
    return _sample_by_temperature(counts, temp, rng)


def select(counts: Dict[Move, int], ply: int, readout: ReadoutConfig, rng,
           top2: Optional[List[ChildStat]] = None) -> Tuple[Move, bool]:
    """Pick the played move. Returns (move, overrode_visit_leader).

    `rng` MUST be the readout stream, never an MCTS search stream.
    `top2` is required by MODE_HOEFFDING_LCB after the opening, ignored
    otherwise.
    """
    if not counts:
        raise ValueError("select called with empty visit counts")

    if readout.mode == MODE_ARGMAX:
        return _argmax(counts), False

    if readout.mode == MODE_OPENING_TEMPERATURE:
        temp = (readout.temp_high if ply < readout.opening_temp_plies
                else readout.temp_low)
        return _opening_move(counts, temp, rng), False

    # MODE_HOEFFDING_LCB: sample the opening for match diversity, then play
    # visit argmax with the frozen override.
    if ply < readout.opening_temp_plies:
        return _opening_move(counts, readout.temp_high, rng), False
    if top2 is None:
        raise ValueError(
            "hoeffding_lcb readout requires top2 child stats after the opening")
    leader = _argmax(counts)
    override = lcb_override(top2)
    if override is not None and override != leader:
        return override, True
    return leader, False
