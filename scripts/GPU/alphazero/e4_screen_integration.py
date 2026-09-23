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
import json
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


def _is_h4(runtime) -> bool:
    """The H4 acceptance mode. Anything that does not say True is the DEFAULT
    path -- which is today's fail-closed behaviour, so absence switches nothing off."""
    return getattr(runtime, "h4_acceptance", False) is True


class IntegrationContext:
    """One task's shared move log. Rebuilt by the state factory per task."""

    def __init__(self) -> None:
        self.task_id: Optional[str] = None
        self.moves: List[Tuple[int, int]] = []
        #: PER TASK and NEVER cleared. An earlier version kept bare counters and
        #: reset them per task, so a cross-task total read the last task's numbers.
        self.stats: Dict[str, Dict[str, int]] = {}
        #: §4B (card §3.1.1): the runtime the first binder/agent registered.
        self.runtime: Any = None
        #: §4B (card §5.1): H4-mode process observations, ONE BUCKET PER TASK,
        #: never erased by `reset` -- the same rule `stats` learned the hard way.
        self.processes: Dict[str, List[Dict[str, Any]]] = {}

    def reset(self, task_id: str, opening: Sequence[Tuple[int, int]]) -> None:
        if _is_h4(self.runtime):
            if task_id in self.processes:
                raise AbortError(PHASE_PRECONDITION,
                                 f"task_id {task_id!r} was already used on this H4 "
                                 f"context; two tasks may not share a record bucket")
            self.processes[task_id] = []
        self.task_id = task_id
        self.moves = [tuple(m) for m in opening]
        self.stats.setdefault(task_id, {"binds": 0, "t1j_queries": 0, "searched_binds": 0})

    def bind_runtime(self, runtime) -> None:
        """Card §3.1.1: every runtime is COMPARED with the first one registered.

        Identity is REQUIRED whenever either side is H4 -- so an H4 agent can never
        meet a default binder, or the reverse, or a second H4 runtime. A
        default/default mismatch stays PERMITTED: that is today's behaviour, and
        callers that accept a `_binder` override were qualified with it.
        """
        if self.runtime is None:
            self.runtime = runtime
            return
        if runtime is self.runtime:
            return
        if _is_h4(runtime) or _is_h4(self.runtime):
            raise AbortError(PHASE_PRECONDITION,
                             f"runtime mismatch on one context: registered "
                             f"h4_acceptance={_is_h4(self.runtime)}, presented "
                             f"h4_acceptance={_is_h4(runtime)}, and they are different "
                             f"objects. An H4 binder and agent must share ONE runtime.")

    def observe(self, obs: Dict[str, Any], outcome: str,
                refused_at: Optional[str] = None, reason: Optional[str] = None) -> None:
        """Append ONE H4 observation for a subprocess that RETURNED (card §5)."""
        bucket = self.processes.setdefault(self.task_id, [])
        obs.update(outcome=outcome, refused_at=refused_at, reason=reason,
                   ordinal=len(bucket), task_id=self.task_id)
        bucket.append(obs)

    def bump(self, key: str) -> None:
        self.stats[self.task_id][key] += 1

    def total(self, key: str) -> int:
        return sum(v[key] for v in self.stats.values())


class T1jRuntime:
    """The pinned runtime. Identity is the caller's business; this just carries it."""

    def __init__(self, *, java: str, jar: str, classes: str, ply_cap: int,
                 timeout_s: float, h4_acceptance: bool = False):
        """``ply_cap`` and ``timeout_s`` are both REQUIRED, for the same reason.

        A missing cap silently defaults further up the stack, and a missing
        timeout silently restores unbounded waiting at ``subprocess.run``. The
        adapter refuses ``None`` for either; this refuses omitting them.
        """
        if timeout_s is None:
            raise TypeError("timeout_s is required: an unbounded replay never returns")
        self.java, self.jar, self.classes, self.ply_cap = java, jar, classes, ply_cap
        self.timeout_s = timeout_s
        #: §4B: THE ONLY switch. Default False is today's fail-closed behaviour.
        self.h4_acceptance = h4_acceptance is True


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
    ctx.bind_runtime(runtime)

    def binder(task: Dict[str, Any], state, ply: int, move=None) -> None:
        if move is not None:
            ctx.moves.append(tuple(move))
        where = "opening" if move is None else f"ply {ply}"
        if len(ctx.moves) != state.ply:
            raise AbortError(PHASE_BIND,
                             f"{task['task_id']} {where}: the move log holds {len(ctx.moves)} "
                             f"moves but our ply is {state.ply}")
        if _is_h4(runtime):
            return _bind_h4(runtime, ctx, task, state, where)

        plies, rc, out = A.replay(ctx.moves, ply_cap=runtime.ply_cap, java=runtime.java,
                                  jar=runtime.jar, classes=runtime.classes,
                                  timeout_s=runtime.timeout_s)
        if rc != 0:
            # THE TRANSCRIPT IS THE DIAGNOSIS. This used to say only "replay exit
            # {rc}" and drop `out`, the same defect that left a real D1 abort
            # unexplained: E3bDump exits non-zero with its `failures` counter set
            # and prints a FAIL line naming the check, and that line went in the
            # bin. Bounded, because a dump carries a 576-character legal-cell map
            # per ply and would bury what the excerpt exists to surface.
            raise AbortError(PHASE_BIND,
                             f"{task['task_id']} {where}: T1j replay exit {rc}. "
                             f"T1j reported: {A.helper_failure_excerpt(out)}")
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
        ctx.bind_runtime(runtime)
        if _is_h4(runtime):
            from . import h4_repair_qualification as H4RQ
            if depth != H4RQ.DEPTH:
                raise AbortError(PHASE_PRECONDITION,
                                 f"H4 mode is qualified at depth {H4RQ.DEPTH} only; got "
                                 f"{depth}. The shared classifier is depth-specific.")
            if timeout_s is None:
                raise AbortError(PHASE_PRECONDITION,
                                 "H4 mode refuses an unbounded QUERY timeout "
                                 "(T1jAgent.timeout_s / t1j_timeout_s is None); the "
                                 "runtime's replay timeout is a different setting")

    def __call__(self, state) -> Tuple[int, int]:
        if state.to_move != self.colour:
            raise AbortError(PHASE_MOVE,
                             f"T1j asked to move as {self.colour} but {state.to_move} is to move")
        if len(self.ctx.moves) != state.ply:
            raise AbortError(PHASE_MOVE,
                             f"the move log holds {len(self.ctx.moves)} moves but our ply is "
                             f"{state.ply}; T1j would search a different position")
        if _is_h4(self.runtime):
            return self._call_h4(state)
        recs, dumps, rc, out = self._query(
            self.ctx.moves, depth=self.depth, java=self.runtime.java, jar=self.runtime.jar,
            classes=self.runtime.classes, timeout_s=self.timeout_s)
        self.ctx.bump("t1j_queries")
        where = f"{self.ctx.task_id} query at ply {state.ply}"
        if rc != 0 or len(recs) != 1:
            # As on the binder's replay path: the helper's own FAIL line is the
            # diagnosis, and dropping it is what left a real D1 abort
            # unexplained. Bounded, and the dump body never travels.
            #
            # 🔴 THE FULL TRANSCRIPT TRAVELS AS THE CAUSE. The excerpt above is
            # bounded for humans; the structured POSTCOND observation (2026-09-05)
            # sits LAST on the transcript and review showed the bound dropping it.
            # A diagnostic reads the chained `stdout` and parses BEFORE bounding.
            message = (f"{where}: exit {rc} with {len(recs)} record(s). "
                       f"T1j reported: {A.helper_failure_excerpt(out)}")
            raise AbortError(PHASE_MOVE, message) from A.HelperOutputError(message, out)
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

    def _call_h4(self, state) -> Tuple[int, int]:
        """§4B card §4: the H4-mode path, in its frozen order.

        `rc`, `failures` and `completed` are judged in ONE place -- the qualified
        classifier -- which REPLACES the three default refusal sites rather than
        relaxing them one at a time (card §2).
        """
        from . import h4_repair_qualification as H4RQ
        ctx, ply = self.ctx, state.ply
        where = f"{ctx.task_id} h4 query at ply {ply}"
        obs = _h4_obs("query", ply)
        try:
            recs, dumps, rc, out = self._query(
                ctx.moves, depth=self.depth, java=self.runtime.java, jar=self.runtime.jar,
                classes=self.runtime.classes, timeout_s=self.timeout_s,
                inject_matchdata=True)
        except A.HelperOutputError as e:          # returned, but unreadable -> VOID
            ctx.observe(obs, "unreadable", "query", str(e))
            raise
        # (subprocess.TimeoutExpired propagates UNCONVERTED: nothing returned,
        # so there is nothing to record -- the runner's VOID names the call.)
        ctx.bump("t1j_queries")
        obs["return_code"] = rc
        step = "proc"
        try:
            _h4_proc(obs, out, where, PHASE_MOVE)
            step = "postcond"
            post = _h4_postcond(obs, out, where, PHASE_MOVE, H4RQ.QUERY_REFL_N_OPTIN)
            step = "matchdata"
            _h4_matchdata(obs, out, where)
            step = "query_record"
            if len(recs) != 1:
                _h4_refuse(PHASE_MOVE, where, f"{len(recs)} query records, expected 1", out)
            r = recs[0]
            obs["telemetry"] = {"usealphabeta": r.usealphabeta,
                                "current_max_ply": r.current_max_ply,
                                "completed": r.completed,
                                "completed_depth": r.completed_depth,
                                "move_nr": r.move_nr, "q": r.q,
                                "requested_depth": r.requested_depth,
                                "elapsed_us": r.elapsed_us}
            ours = A.PLAYER_TO_T1J[state.to_move]
            for ok, msg in ((r.q == 1, f"query index q={r.q}, expected 1"),
                            (r.move_nr == ply, f"moveNr={r.move_nr} but our ply is {ply}"),
                            (r.to_move == ours, f"to_move={r.to_move!r} but ours is {ours!r}"),
                            (r.requested_depth == self.depth,
                             f"requested depth {r.requested_depth}, not {self.depth}")):
                if not ok:
                    _h4_refuse(PHASE_MOVE, where, msg, out)
            step = "dump"
            if len(dumps) != 1:
                _h4_refuse(PHASE_MOVE, where,
                           f"{len(dumps)} searched-position dumps, expected 1", out)
            _h4_coherence(obs, state, dumps[0], ctx.moves)
            div = compare_state(state, dumps[0], ctx.moves)
            if div:
                _h4_refuse(PHASE_MOVE, where, "the query jvm reconstructed a different "
                           "position: " + "; ".join(div), out)
            ctx.bump("searched_binds")
            step = "classify"
            try:
                source = _h4_classifier()(ply=ply, rec=r, exit_status=rc,
                                          failures=post.failures)
            except H4RQ.H4RQStop as e:
                _h4_refuse(PHASE_MOVE, where, e.message, out)
            obs["source"] = source
            step = "move"
            obs["move"] = None if r.move is None else list(r.move)
            if r.null_sentinel or r.move is None or not r.legal:
                _h4_refuse(PHASE_MOVE, where, f"unusable move {r.move}", out)
            if r.move not in set(state.legal_moves()):
                _h4_refuse(PHASE_MOVE, where, f"{r.move} is illegal in OUR engine", out)
        except AbortError as e:                    # semantic refusal -> STOP
            ctx.observe(obs, "refused", step, e.message)
            raise
        except (ValueError, KeyError) as e:        # a line would not parse -> VOID
            ctx.observe(obs, "unreadable", step, str(e))
            raise
        ctx.observe(obs, "accepted")
        self.moves_made += 1
        self.last_completed_depth = r.completed_depth
        return r.move


# ─────────────────────── §4B: the H4-mode helpers (card §4, §5) ───────────────────────

def _h4_classifier():
    """The QUALIFIED classifier, looked up AT CALL TIME and never copied.

    `h4_repair_qualification` imports this module at import time, so importing
    it back at module level would be a cycle. Looking it up per call also means
    it is always THE object the CLEAN qualification exercised.
    """
    from . import h4_repair_qualification as H4RQ
    return H4RQ.classify_reply


def _h4_obs(role: str, ply: int) -> Dict[str, Any]:
    """A fresh observation. Fields the path does not reach stay None -- never guessed."""
    return {"role": role, "board_ply": ply, "return_code": None, "proc": None,
            "postcond": None, "source": None, "matchdata": None, "move": None,
            "telemetry": None, "t1j_position_digest": None,
            "expected_position_digest": None}


# ── H4 runner card §4: the DURABLE coherence field. Additive and observational:
# computed from a dump the adapter already parsed, BEFORE `compare_state` judges
# it, and never consulted by any accept/refuse decision.

def position_payload(tp) -> Dict[str, Any]:
    """T1j's reported position, as the canonical payload -- EXACTLY the fields
    `compare_state` compares, each in a canonical order."""
    return {"ply": tp.ply, "next_player": tp.next_player,
            "term_y": bool(tp.term_y), "term_x": bool(tp.term_x),
            "pegs": sorted(tp.pegs), "bridges": sorted(tp.bridges),
            "legal": sorted([list(xy) for xy in tp.legal]),
            "history": [list(xy) for xy in tp.history]}


def expected_payload(state, moves: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
    """OUR position, in the same vocabulary. Recomputable from the moves alone."""
    pegs, bridges = A.our_snapshot(state)
    winner = state.winner()
    return {"ply": state.ply, "next_player": A.PLAYER_TO_T1J[state.to_move],
            "term_y": winner == "red", "term_x": winner == "black",
            "pegs": sorted(pegs), "bridges": sorted(bridges),
            "legal": sorted([list(A.to_t1j(r, c)) for (r, c) in state.legal_moves()]),
            "history": [list(A.to_t1j(*m)) for m in moves]}


def position_digest(payload: Dict[str, Any]) -> str:
    """The payload's INTEGRITY FINGERPRINT. Payload equality is what matches
    `compare_state`; digest equality implies it only under SHA-256's collision
    resistance (runner card §4)."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _h4_coherence(obs: Dict[str, Any], state, tp, moves) -> None:
    obs["t1j_position_digest"] = position_digest(position_payload(tp))
    obs["expected_position_digest"] = position_digest(expected_payload(state, moves))


def _h4_refuse(phase: str, where: str, message: str, out: str) -> None:
    """A SEMANTIC refusal: AbortError carrying the FULL transcript as its cause."""
    msg = f"{where}: {message}"
    raise AbortError(phase, msg) from A.HelperOutputError(msg, out)


def _h4_proc(obs: Dict[str, Any], out: str, where: str, phase: str) -> None:
    procs = A.parse_procs(out)
    if len(procs) != 1:
        _h4_refuse(phase, where, f"{len(procs)} PROC lines, expected exactly 1 per jvm", out)
    p = procs[0]
    obs["proc"] = {"pid": p.pid, "java_version": p.java_version, "vm": p.vm,
                   "headless": p.headless, "prefs_factory": p.prefs_factory}


def _h4_postcond(obs: Dict[str, Any], out: str, where: str, phase: str,
                 expected_refl: int):
    """The SAFETY SURFACE, field by field. `failures` is recorded, never judged here:
    `PostCond.clean` folds it in, and that is the refusal site this path replaces."""
    posts = A.parse_postconds(out)
    if len(posts) != 1:
        _h4_refuse(phase, where, f"{len(posts)} POSTCOND lines, expected exactly 1", out)
    p = posts[0]
    obs["postcond"] = {k: getattr(p, k) for k in (
        "no_throw", "windows", "frames", "headless", "prefs_ok", "refl_ok",
        "refl_n", "failures")}
    for ok, msg in ((p.no_throw, "the helper threw"),
                    (p.windows == 0, f"{p.windows} windows opened"),
                    (p.frames == 0, f"{p.frames} frames opened"),
                    (p.headless, "not headless"),
                    (p.prefs_ok, "the preferences surface was disturbed"),
                    (p.refl_ok, "the reflective-access check failed"),
                    (p.refl_n == expected_refl,
                     f"refl_n={p.refl_n}, expected exactly {expected_refl}")):
        if not ok:
            _h4_refuse(phase, where, f"safety surface not clean: {msg}", out)
    return p


def _h4_matchdata(obs: Dict[str, Any], out: str, where: str) -> None:
    mds = A.parse_matchdata(out)
    if len(mds) != 1:
        _h4_refuse(PHASE_MOVE, where, f"{len(mds)} MATCHDATA lines, expected exactly 1", out)
    md = mds[0]
    obs["matchdata"] = {"pie_rule": md.pie_rule, "xsize": md.xsize, "ysize": md.ysize,
                        "ystarts": md.ystarts, "identity": md.identity}
    if md.pie_rule or md.xsize != A.BOARD_N or md.ysize != A.BOARD_N or not md.ystarts:
        _h4_refuse(PHASE_MOVE, where, f"the injected MatchData did not read back as "
                   f"frozen: {obs['matchdata']}", out)
    if not md.identity:
        _h4_refuse(PHASE_MOVE, where, "MATCHDATA identity=false -- getMatchData() did "
                   "not return the injected object", out)


def _bind_h4(runtime, ctx: IntegrationContext, task: Dict[str, Any], state,
             where: str) -> None:
    """§4B card §4: the H4 binder. Every default replay check, plus exactly one
    PROC, an observation for every returning replay, and the FULL stdout on
    every refusal -- through its own statements, so the default path is untouched."""
    where = f"{task['task_id']} {where} h4 replay"
    obs = _h4_obs("replay", state.ply)
    try:
        plies, rc, out = A.replay(ctx.moves, ply_cap=runtime.ply_cap, java=runtime.java,
                                  jar=runtime.jar, classes=runtime.classes,
                                  timeout_s=runtime.timeout_s)
    except A.HelperOutputError as e:
        ctx.observe(obs, "unreadable", "replay", str(e))
        raise
    obs["return_code"] = rc
    step = "proc"
    try:
        _h4_proc(obs, out, where, PHASE_BIND)
        step = "exit"
        if rc != 0:
            _h4_refuse(PHASE_BIND, where, f"T1j replay exit {rc}", out)
        step = "postcond"
        p = _h4_postcond(obs, out, where, PHASE_BIND, REPLAY_REFL_N)
        if p.failures != 0:
            _h4_refuse(PHASE_BIND, where, f"failures={p.failures} on a replay", out)
        step = "plies"
        if len(plies) != state.ply + 1:
            _h4_refuse(PHASE_BIND, where, f"T1j reported {len(plies)} plies, expected "
                       f"{state.ply + 1}", out)
        step = "state"
        _h4_coherence(obs, state, plies[-1], ctx.moves)
        div = compare_state(state, plies[-1], ctx.moves)
        if div:
            _h4_refuse(PHASE_BIND, where, "; ".join(div), out)
    except AbortError as e:
        ctx.observe(obs, "refused", step, e.message)
        raise
    except (ValueError, KeyError) as e:
        ctx.observe(obs, "unreadable", step, str(e))
        raise
    ctx.observe(obs, "accepted")
    ctx.bump("binds")


def make_agent_factory(*, runtime: T1jRuntime, ctx: IntegrationContext, evaluator,
                       reference_build: Callable, t1j_timeout_s: Optional[float] = None,
                       _query: Optional[Callable] = None) -> Callable:
    """`(task, mover) -> agent`. The reference on its colour, T1j on the other."""
    ctx.bind_runtime(runtime)

    def agent_factory(task: Dict[str, Any], mover: str, _evaluator=None):
        if mover == task["reference_colour"]:
            return reference_build(task, evaluator=evaluator)
        return T1jAgent(runtime=runtime, ctx=ctx, depth=int(task["t1j_mdPly"]),
                        colour=mover, timeout_s=t1j_timeout_s, _query=_query)

    return agent_factory
