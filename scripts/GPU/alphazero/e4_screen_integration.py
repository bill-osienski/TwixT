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

import hashlib
import os

from . import t1j_adapter as A
from .e4_screen_runner import AbortError, PHASE_BIND, PHASE_MOVE, PHASE_PRECONDITION

#: Reflection counts, MEASURED not assumed. E3bDump's replay mode reflects once
#: (freshMatch's single nextPlayer write). E4Preflight reflects three times per
#: query: that write plus the two FindMove reads. A count is checked at every
#: caller, because PostCond.clean only proves the field NAMES were authorized --
#: a repeated or missing authorized access would pass it.
REPLAY_REFL_N = 1
QUERY_REFL_N = 3

#: The qualified JDK, by component. Presence of a path named for Temurin 17 does
#: not bind the runtime that actually executed anything.
PINNED_JDK = {
    "bin/java": "af8b122943345320b179c75c3404d56a981017739746b75f9caf583632f0bea0",
    "bin/javac": "6f5159301c750bba340390eda5fdd4a0959445355f97c40aa9c2addb00ede5ab",
    "lib/modules": "28745573641057e822a972f223fc8e40db5c9df11ae3f7402764780afc2f1951",
    "release": "cb6064fe4d7b87d9fbb8b8c7702047044d1bbeac38e0c5217f595579b6cc764b",
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_jdk_identity(jdk_home: str, pinned: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Bind the RUNTIME, not a directory name. Raises on any mismatch."""
    pinned = PINNED_JDK if pinned is None else pinned
    seen = {}
    for rel, want in sorted(pinned.items()):
        path = os.path.join(jdk_home, rel)
        if not os.path.isfile(path):
            raise AbortError(PHASE_PRECONDITION, f"pinned JDK component missing: {rel}")
        got = _sha256(path)
        seen[rel] = got
        if got != want:
            raise AbortError(PHASE_PRECONDITION,
                             f"JDK {rel} sha256 {got} != pinned {want}")
    return seen


def check_postcond(out: str, *, expected_refl: int, where: str, phase: str) -> Any:
    """The helper's own safety surface, INCLUDING the exact reflection count."""
    posts = A.parse_postconds(out)
    if len(posts) != 1:
        raise AbortError(phase, f"{where}: {len(posts)} POSTCOND lines, expected exactly 1")
    p = posts[0]
    if not p.clean:
        raise AbortError(phase, f"{where}: T1j postconditions not clean: {p}")
    if p.refl_n != expected_refl:
        raise AbortError(phase,
                         f"{where}: {p.refl_n} reflective accesses, expected exactly "
                         f"{expected_refl}; authorized NAMES are not an authorized COUNT")
    return p


def compare_state(state, tp, moves: Sequence[Tuple[int, int]]) -> List[str]:
    """Divergences between OUR state and a T1j ply record. ONE implementation.

    Used by the per-ply binder AND by the agent's check of the position the search
    jvm actually reconstructed, so the two can never drift apart.
    """
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
    submitted = tuple(A.to_t1j(*m) for m in moves)
    if tuple(tp.history) != submitted:
        div.append(f"history {list(tp.history)[-2:]} != submitted {list(submitted)[-2:]}")
    return div


class IntegrationContext:
    """One task's shared move log. Rebuilt by the state factory per task."""

    def __init__(self) -> None:
        self.task_id: Optional[str] = None
        self.moves: List[Tuple[int, int]] = []
        #: PER TASK and NEVER cleared. An earlier version kept bare counters and
        #: reset them per task, so a cross-task total read the last task's numbers.
        self.stats: Dict[str, Dict[str, int]] = {}

    def reset(self, task_id: str, opening: Sequence[Tuple[int, int]]) -> None:
        self.task_id = task_id
        self.moves = [tuple(m) for m in opening]
        self.stats.setdefault(task_id, {"binds": 0, "t1j_queries": 0, "searched_binds": 0})

    def bump(self, key: str) -> None:
        self.stats[self.task_id][key] += 1

    def total(self, key: str) -> int:
        return sum(v[key] for v in self.stats.values())


class T1jRuntime:
    """The pinned runtime. Identity is the caller's business; this just carries it."""

    def __init__(self, *, java: str, jar: str, classes: str, ply_cap: int,
                 timeout_s: float):
        """``ply_cap`` and ``timeout_s`` are both REQUIRED, for the same reason.

        A missing cap silently defaults further up the stack, and a missing
        timeout silently restores unbounded waiting at ``subprocess.run``. The
        adapter refuses ``None`` for either; this refuses omitting them.
        """
        if timeout_s is None:
            raise TypeError("timeout_s is required: an unbounded replay never returns")
        self.java, self.jar, self.classes, self.ply_cap = java, jar, classes, ply_cap
        self.timeout_s = timeout_s


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
                                  jar=runtime.jar, classes=runtime.classes,
                                  timeout_s=runtime.timeout_s)
        if rc != 0:
            raise AbortError(PHASE_BIND, f"{task['task_id']} {where}: T1j replay exit {rc}")
        check_postcond(out, expected_refl=REPLAY_REFL_N,
                       where=f"{task['task_id']} {where} replay", phase=PHASE_BIND)
        if len(plies) != state.ply + 1:
            raise AbortError(PHASE_BIND,
                             f"{task['task_id']} {where}: T1j reported {len(plies)} plies, "
                             f"expected {state.ply + 1}")
        tp = plies[-1]
        div = compare_state(state, tp, ctx.moves)
        if div:
            raise AbortError(PHASE_BIND, f"{task['task_id']} {where}: " + "; ".join(div))
        ctx.bump("binds")

    return binder


class T1jAgent:
    """The anchor. Classical: it holds NO evaluator, by design.

    THE POSITION THE SEARCH JVM RECONSTRUCTED IS RE-BOUND before its move is
    accepted. The per-ply binder proves that *a* jvm can rebuild the history; it
    says nothing about the jvm that actually searched. Those are different
    processes, and only this check ties the returned move to our position.
    """

    def __init__(self, *, runtime: T1jRuntime, ctx: IntegrationContext, depth: int,
                 colour: str, timeout_s: Optional[float] = None, _query: Optional[Callable] = None):
        self.runtime, self.ctx, self.depth, self.colour = runtime, ctx, depth, colour
        self.timeout_s = timeout_s
        self._query = _query or A.query          # private seam, for fail-closed tests
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
        recs, dumps, rc, out = self._query(
            self.ctx.moves, depth=self.depth, java=self.runtime.java, jar=self.runtime.jar,
            classes=self.runtime.classes, timeout_s=self.timeout_s)
        self.ctx.bump("t1j_queries")
        where = f"{self.ctx.task_id} query at ply {state.ply}"
        if rc != 0 or len(recs) != 1:
            raise AbortError(PHASE_MOVE, f"{where}: exit {rc} with {len(recs)} record(s)")
        check_postcond(out, expected_refl=QUERY_REFL_N, where=where, phase=PHASE_MOVE)

        # THE SEARCHED POSITION, re-bound against ours before the move is used.
        if len(dumps) != 1:
            raise AbortError(PHASE_MOVE,
                             f"{where}: {len(dumps)} searched-position dumps, expected 1")
        div = compare_state(state, dumps[0], self.ctx.moves)
        if div:
            raise AbortError(PHASE_MOVE,
                             f"{where}: the SEARCH jvm reconstructed a different position: "
                             + "; ".join(div))
        self.ctx.bump("searched_binds")

        r = recs[0]
        if not r.completed or r.requested_depth != self.depth:
            raise AbortError(PHASE_MOVE,
                             f"{where}: T1j did not complete depth {self.depth} "
                             f"(currentMaxPly={r.current_max_ply}, usealphabeta={r.usealphabeta})")
        if r.null_sentinel or r.move is None or not r.legal:
            raise AbortError(PHASE_MOVE, f"{where}: unusable move {r.move}")
        if r.move not in set(state.legal_moves()):
            raise AbortError(PHASE_MOVE,
                             f"{where}: T1j returned {r.move}, illegal in OUR engine")
        self.moves_made += 1
        self.last_completed_depth = r.completed_depth
        return r.move


def make_agent_factory(*, runtime: T1jRuntime, ctx: IntegrationContext, evaluator,
                       reference_build: Callable, t1j_timeout_s: Optional[float] = None,
                       _query: Optional[Callable] = None) -> Callable:
    """`(task, mover) -> agent`. The reference on its colour, T1j on the other."""

    def agent_factory(task: Dict[str, Any], mover: str, _evaluator=None):
        if mover == task["reference_colour"]:
            return reference_build(task, evaluator=evaluator)
        return T1jAgent(runtime=runtime, ctx=ctx, depth=int(task["t1j_mdPly"]),
                        colour=mover, timeout_s=t1j_timeout_s, _query=_query)

    return agent_factory
