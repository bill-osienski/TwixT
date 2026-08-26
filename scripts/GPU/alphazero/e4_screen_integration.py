"""Wire the REAL T1j engine and the REAL reference agent into the E4 harness.

**NO SCREEN.** This module supplies the three components the harness leaves
abstract -- a state factory, a per-ply binder, and an agent factory -- so that a
qualification run can drive both engines against each other for a handful of
plies. The canonical 32 tasks, the reserved seeds and any complete benchmark game
remain out of scope; the harness's own gates enforce that, not this file.

WHY A MOVE LOG. Our TwixtState keeps no ordered history -- E3b established that --
and T1j is only ever advanced by replaying an ordered sequence through its own
``Match.setlastMove``. So a per-task move log is the shared spine: the state
factory seeds it with the opening, the binder appends each move as it is applied,
and the T1j agent replays it to reach the position it must move from. The harness
passes the move to the binder precisely so this log can exist.

WHAT THE BINDER CHECKS, EVERY PLY. Pegs, bridges, side to move, independently
derived ply, the full legal-move set, terminal state with winner attribution, and
T1j's ORDERED HISTORY read back through its own accessors -- plus the helper's
POSTCOND surface: headless, zero Window/Frame, host preferences unchanged, only
authorized reflection. The first divergence aborts; nothing is repaired.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import t1j_adapter as A
from .e4_screen_runner import AbortError, PHASE_BIND, PHASE_MOVE, PHASE_PRECONDITION


class IntegrationContext:
    """One task's shared move log. Rebuilt by the state factory per task."""

    def __init__(self) -> None:
        self.task_id: Optional[str] = None
        self.moves: List[Tuple[int, int]] = []
        self.binds = 0
        self.t1j_queries = 0

    def reset(self, task_id: str, opening: Sequence[Tuple[int, int]]) -> None:
        self.task_id = task_id
        self.moves = [tuple(m) for m in opening]
        self.binds = 0
        self.t1j_queries = 0


class T1jRuntime:
    """The pinned runtime. Identity is the caller's business; this just carries it."""

    def __init__(self, *, java: str, jar: str, classes: str, ply_cap: int):
        self.java, self.jar, self.classes, self.ply_cap = java, jar, classes, ply_cap


def make_state_factory(openings: Dict[str, Sequence[Tuple[int, int]]],
                       ctx: IntegrationContext, *, board_size: int = A.BOARD_N) -> Callable:
    """Build the opening position and seed the move log."""
    from .game.twixt_state import TwixtState

    def state_factory(task: Dict[str, Any]):
        name = task["opening"]
        if name not in openings:
            raise AbortError(PHASE_PRECONDITION, f"{task['task_id']}: unknown opening {name!r}")
        moves = [tuple(m) for m in openings[name]]
        state = TwixtState(active_size=board_size, to_move="red")
        for mv in moves:
            if mv not in set(state.legal_moves()):
                raise AbortError(PHASE_PRECONDITION,
                                 f"{task['task_id']}: opening move {mv} is illegal at ply "
                                 f"{state.ply}")
            state = state.apply_move(mv)
        ctx.reset(task["task_id"], moves)
        return state

    return state_factory


def make_binder(runtime: T1jRuntime, ctx: IntegrationContext) -> Callable:
    """The E3b per-ply binder. Aborts on the FIRST divergence."""

    def binder(task: Dict[str, Any], state, ply: int, move=None) -> None:
        if move is not None:
            ctx.moves.append(tuple(move))
        where = "opening" if move is None else f"ply {ply}"
        if len(ctx.moves) != state.ply:
            raise AbortError(PHASE_BIND,
                             f"{task['task_id']} {where}: the move log holds {len(ctx.moves)} "
                             f"moves but our ply is {state.ply}")

        plies, rc, out = A.replay(ctx.moves, ply_cap=runtime.ply_cap, java=runtime.java,
                                  jar=runtime.jar, classes=runtime.classes)
        if rc != 0:
            raise AbortError(PHASE_BIND, f"{task['task_id']} {where}: T1j replay exit {rc}")
        posts = A.parse_postconds(out)
        if len(posts) != 1 or not posts[0].clean:
            raise AbortError(PHASE_BIND,
                             f"{task['task_id']} {where}: T1j postconditions not clean: {posts}")
        if len(plies) != state.ply + 1:
            raise AbortError(PHASE_BIND,
                             f"{task['task_id']} {where}: T1j reported {len(plies)} plies, "
                             f"expected {state.ply + 1}")
        tp = plies[-1]

        opegs, obr = A.our_snapshot(state)
        div: List[str] = []
        if opegs != tp.pegs:
            div.append(f"pegs (ours-only {sorted(opegs - tp.pegs)[:2]}, "
                       f"t1j-only {sorted(tp.pegs - opegs)[:2]})")
        if obr != tp.bridges:
            div.append(f"bridges (ours-only {sorted(obr - tp.bridges)[:2]}, "
                       f"t1j-only {sorted(tp.bridges - obr)[:2]})")
        if tp.next_player != A.PLAYER_TO_T1J[state.to_move]:
            div.append(f"side to move {tp.next_player} != {A.PLAYER_TO_T1J[state.to_move]}")
        if tp.ply != state.ply:
            div.append(f"ply T1j {tp.ply} != ours {state.ply}")
        ours_legal = {A.to_t1j(r, c) for (r, c) in state.legal_moves()}
        if ours_legal != tp.legal:
            div.append(f"legal set |ours|={len(ours_legal)} |t1j|={len(tp.legal)}")
        ow = state.winner()
        if {"Y": tp.term_y, "X": tp.term_x} != {"Y": ow == "red", "X": ow == "black"}:
            div.append(f"terminal T1j Y={tp.term_y} X={tp.term_x} != ours {ow}")
        submitted = tuple(A.to_t1j(*m) for m in ctx.moves)
        if tuple(tp.history) != submitted:
            div.append(f"T1j history {list(tp.history)[-2:]} != submitted {list(submitted)[-2:]}")
        if div:
            raise AbortError(PHASE_BIND, f"{task['task_id']} {where}: " + "; ".join(div))
        ctx.binds += 1

    return binder


class T1jAgent:
    """The anchor. Classical: it holds NO evaluator, by design."""

    def __init__(self, *, runtime: T1jRuntime, ctx: IntegrationContext, depth: int,
                 colour: str, timeout_s: Optional[float] = None):
        self.runtime, self.ctx, self.depth, self.colour = runtime, ctx, depth, colour
        self.timeout_s = timeout_s
        self.moves_made = 0
        self.last_completed_depth: Optional[int] = None

    def __call__(self, state) -> Tuple[int, int]:
        if state.to_move != self.colour:
            raise AbortError(PHASE_MOVE,
                             f"T1j asked to move as {self.colour} but {state.to_move} is to move")
        if len(self.ctx.moves) != state.ply:
            raise AbortError(PHASE_MOVE,
                             f"the move log holds {len(self.ctx.moves)} moves but our ply is "
                             f"{state.ply}; T1j would search a different position")
        recs, dumps, rc, out = A.query(self.ctx.moves, depth=self.depth, java=self.runtime.java,
                                       jar=self.runtime.jar, classes=self.runtime.classes,
                                       timeout_s=self.timeout_s)
        self.ctx.t1j_queries += 1
        if rc != 0 or len(recs) != 1:
            raise AbortError(PHASE_MOVE, f"T1j query exit {rc} with {len(recs)} record(s)")
        posts = A.parse_postconds(out)
        if len(posts) != 1 or not posts[0].clean:
            raise AbortError(PHASE_MOVE, f"T1j query postconditions not clean: {posts}")
        r = recs[0]
        if not r.completed or r.requested_depth != self.depth:
            raise AbortError(PHASE_MOVE,
                             f"T1j did not complete depth {self.depth} "
                             f"(currentMaxPly={r.current_max_ply}, usealphabeta={r.usealphabeta})")
        if r.null_sentinel or r.move is None or not r.legal:
            raise AbortError(PHASE_MOVE, f"T1j returned an unusable move {r.move}")
        self.moves_made += 1
        self.last_completed_depth = r.completed_depth
        return r.move


def make_agent_factory(*, runtime: T1jRuntime, ctx: IntegrationContext, evaluator,
                       reference_build: Callable, t1j_timeout_s: Optional[float] = None
                       ) -> Callable:
    """`(task, mover) -> agent`. The reference on its colour, T1j on the other."""

    def agent_factory(task: Dict[str, Any], mover: str, _evaluator=None):
        if mover == task["reference_colour"]:
            return reference_build(task, evaluator=evaluator)
        return T1jAgent(runtime=runtime, ctx=ctx, depth=int(task["t1j_mdPly"]),
                        colour=mover, timeout_s=t1j_timeout_s)

    return agent_factory
