"""Low-ply T1j qualification -- the GATED RUNNER. IT IS NOT AUTHORIZED.

Frozen by `docs/superpowers/2026-08-28-t1j-lowply-qualification-card.md`. Nothing
here loads a model, draws or registers a seed, reads confirmation data, trains,
or touches D1's experiment. It queries T1j and nothing else, which is why it
needs no seed interval at all: it never invokes the incumbent, so it never
derives an RNG stream.

🔑 THE ONE THING THAT MAKES THIS DIFFERENT FROM D1: **FAIL IS A RESULT, NOT AN
ABORT.** D1 conflated them -- a T1j reply that did not complete its requested
depth aborted the whole run, so the observation could not be recorded, and that
is precisely why the low-ply question is still open. Here, T1j failing to
complete IS THE MEASUREMENT: the run continues and records every reply.

VOID is reserved for the INSTRUMENT: identity mismatch, a per-call timeout, a
deadline breach, output the adapter cannot parse into a reply at all, or a
refusal about the harness rather than about T1j.

WHY D1'S TIMING MACHINERY IS IMPORTED RATHER THAN RESTATED. `Deadline`,
`QueryBudget` and `_supervisor` took three rounds to get right -- one clock and
one origin, arming from the remaining time, refusing an unstarted deadline, and
refusing a non-positive remaining because `setitimer(..., 0)` DISABLES the timer.
A second copy of that would be a second thing to get wrong. What is imported is
the TIMING, never D1's gate: this module reads its OWN constant and no other
experiment's, and a test asserts the others appear nowhere in this file.
"""
from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import e4_screen_integration as INT
from . import t1j_adapter as A
from .d1_probe import D1Error, D1VoidError
from .d1_probe import Deadline, QueryBudget, T1jPaths, _supervisor
from .d1_probe import _default_compile as _compile_helper_verified
from .e4_screen_runner import AbortError
from .t1j_toolchain import ToolchainError

Pos = Tuple[int, int]

#: THE LOW-PLY QUALIFICATION IS UNAUTHORIZED. Changing this is a reviewed code
#: change. Read directly, at BOTH public entry points -- `run_qualification` and
#: `main` -- because gating only the CLI protects nothing: a direct Python caller
#: reaches the runner without passing any gate. No supported override exists --
#: not argv, not the environment, not a configuration file, not an import hook.
#:
#: This is the qualification's OWN gate. It reads no other experiment's, and a
#: test asserts the others appear nowhere in this file: one gate must never be
#: openable by opening another.
LOWPLY_QUALIFICATION_AUTHORIZED = False

#: Frozen in the card. Nine prefixes, both qualified depths, two invocations per
#: depth as SEPARATE processes -- the only way to vary T1j's per-process Zobrist
#: salt, which is why `repeats>1` is prohibited: it reuses one process's salt and
#: so agrees by construction.
DEPTHS = (3, 6)
INVOCATIONS_PER_DEPTH = 2
N_PREFIXES = 9
QUERY_CAP = N_PREFIXES * len(DEPTHS) * INVOCATIONS_PER_DEPTH      # 36
PER_CALL_TIMEOUT_S = 120
RUN_DEADLINE_S = 900

#: The frozen input, pinned by content. A qualification that read a different
#: prefix list would be answering a different question.
FROZEN_PREFIXES_REL = ("docs/superpowers/evidence/2026-08-28-t1j-lowply-qualification/"
                       "01_frozen_prefixes.json")
FROZEN_PREFIXES_SHA256 = "a9054cb2d56cf75f554292e47ac7e5b86a0c74f54b255ee9147fb65f92207acf"


class LowPlyError(Exception):
    """A refusal by the harness. Never a statement about T1j."""


class LowPlyVoidError(LowPlyError):
    """The run is VOID: the instrument failed, so nothing was measured."""


def digest_of(state) -> str:
    """The card's positions carry §12.2 digests; this recomputes the same one."""
    from .d1_selection import canonical_digest
    return canonical_digest(state)


def load_frozen_prefixes(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """The nine frozen prefixes, VERIFIED BY HASH before they are returned.

    The card's whole claim to be frozen rests on this file, so a run that read a
    silently different one would answer a different question while looking
    identical. An explicit path is hashed against the SAME pin: the argument
    relocates the file, it does not choose a different one.
    """
    p = FROZEN_PREFIXES_REL if path is None else path
    try:
        raw = open(p, "rb").read()
    except OSError as e:
        raise LowPlyError(f"cannot read the frozen prefixes: {e}") from None
    import hashlib
    got = hashlib.sha256(raw).hexdigest()
    if got != FROZEN_PREFIXES_SHA256:
        raise LowPlyError(
            f"{p}: sha256 {got} != the pinned {FROZEN_PREFIXES_SHA256}. The card is "
            f"bound to that exact content, so a different file answers a different "
            f"question while looking identical.")
    return json.loads(raw)["prefixes"]


def prefix_label(p: Dict[str, Any]) -> str:
    """Which prefix an observation is about. The D1 VOID could not say."""
    digest = str(p.get("digest") or "")
    return (f"{p.get('task_id', '?')}@ply{p.get('ply', '?')} "
            f"[{p.get('signature') or '?'}/{p.get('role') or '?'}] "
            f"digest={digest[:16] or '?'}")


def _replay_state(prefix: Sequence[Pos], *, where: str):
    from .game.twixt_state import TwixtState
    state = TwixtState(active_size=A.BOARD_N, to_move="red")
    for i, move in enumerate(prefix):
        if tuple(move) not in set(state.legal_moves()):
            raise LowPlyVoidError(
                f"{where}: prefix move {i} {tuple(move)} is illegal at ply {state.ply}. "
                f"The frozen input is wrong, which is an instrument failure: VOID.")
        state = state.apply_move(tuple(move))
    return state


def _observe_reply(rec, dumps, out, state, moves, *, depth: int,
                   where: str) -> Dict[str, Any]:
    """ONE reply, described. Its shortcomings are RECORDED, not raised.

    Everything the card lists under PASS is checked here and any shortfall is
    appended to `failures`. None of it aborts, because every one of these is a
    statement ABOUT T1J, and refusing to record it is how D1 lost the answer.
    """
    failures: List[str] = []
    if not rec.completed or rec.completed_depth != depth:
        failures.append(f"did not complete depth {depth} "
                        f"(completed={rec.completed}, completed_depth={rec.completed_depth})")
    if rec.requested_depth != depth:
        failures.append(f"reply reports requested depth {rec.requested_depth}, not {depth}")
    if rec.null_sentinel or rec.move is None:
        failures.append("returned the null sentinel, not a move")
    if not rec.legal:
        failures.append("T1j reports its own move illegal")
    if rec.move is not None and rec.move not in set(state.legal_moves()):
        failures.append(f"returned {rec.move}, illegal in OUR engine")

    posts = A.parse_postconds(out)
    post = None
    if len(posts) != 1:
        failures.append(f"{len(posts)} POSTCOND lines, expected exactly 1")
    else:
        post = posts[0]
        if not post.clean:
            failures.append(f"postcondition surface not clean: {post}")
        if post.refl_n != INT.QUERY_REFL_N:
            failures.append(f"{post.refl_n} reflective accesses, expected exactly "
                            f"{INT.QUERY_REFL_N}")

    searched = None
    if len(dumps) != 1:
        failures.append(f"{len(dumps)} searched-position dumps, expected exactly 1")
    else:
        div = INT.compare_state(state, dumps[0], moves)
        if div:
            failures.append("the SEARCH jvm reconstructed a different position: "
                            + "; ".join(div))
        searched = {"ply": dumps[0].ply, "next_player": dumps[0].next_player,
                    "n_pegs": len(dumps[0].pegs), "n_bridges": len(dumps[0].bridges),
                    "n_legal": len(dumps[0].legal),
                    "history_len": len(dumps[0].history)}

    return {"where": where, "move": list(rec.move) if rec.move else None,
            "requested_depth": rec.requested_depth,
            "completed_depth": rec.completed_depth, "completed": rec.completed,
            "legal": rec.legal, "null_sentinel": rec.null_sentinel,
            "to_move": rec.to_move, "current_max_ply": rec.current_max_ply,
            "usealphabeta": rec.usealphabeta, "eval_regime": rec.eval_regime,
            "elapsed_us": rec.elapsed_us,
            "postcond": (None if post is None else
                         {"clean": post.clean, "refl_n": post.refl_n,
                          "failures": post.failures}),
            "searched_state": searched, "failures": failures}


def _query_once(*, moves: Sequence[Pos], depth: int, paths: T1jPaths, state,
                budget: QueryBudget, label: str, index: int) -> Dict[str, Any]:
    where = f"{label}: depth {depth} invocation {index}"
    budget.spend(1)
    try:
        recs, dumps, rc, out = A.query(
            [tuple(m) for m in moves], depth=depth, java=paths.java, jar=paths.jar,
            classes=paths.classes, repeats=1, timeout_s=PER_CALL_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        raise LowPlyVoidError(
            f"{where}: T1j did not answer within {PER_CALL_TIMEOUT_S}s ({e}). "
            f"A hung process is an instrument failure: VOID.") from None

    # A NON-ZERO EXIT IS EXPECTED HERE AND IS NOT A VOID. E4Preflight exits 3
    # when its own `failures` counter is set -- which is exactly the condition
    # being measured. What makes a reply unusable is having no reply at all.
    if len(recs) != 1:
        raise LowPlyVoidError(
            f"{where}: exit {rc} and no usable query record ({len(recs)} parsed). "
            f"T1j reported: {A.helper_failure_excerpt(out)}. VOID.")

    obs = _observe_reply(recs[0], dumps, out, state, moves, depth=depth, where=where)
    obs["exit_status"] = rc
    obs["helper_report"] = A.helper_failure_excerpt(out)
    return obs


def _bind_prefix(binder: Callable, ctx, *, label: str, state,
                 prefix: Sequence[Pos]) -> Dict[str, Any]:
    """The E3b replay bind. A divergence is RECORDED, per the card's PASS list.

    A TIMEOUT is different and voids: that is the instrument, not T1j's answer.
    """
    ctx.reset(label, [tuple(m) for m in prefix])
    try:
        binder({"task_id": label}, state, state.ply)
    except subprocess.TimeoutExpired as e:
        raise LowPlyVoidError(
            f"{label}: the prefix replay did not answer within "
            f"{PER_CALL_TIMEOUT_S}s ({e}): VOID.") from None
    except AbortError as e:
        return {"bound": False, "detail": e.message}
    return {"bound": True, "detail": None}


def run_qualification(*, prefixes: Sequence[Dict[str, Any]], paths: T1jPaths,
                      out_path: str, deadline: Optional[Deadline] = None,
                      budget: Optional[QueryBudget] = None,
                      _compile: Optional[Callable] = None) -> Dict[str, Any]:
    """PUBLIC ENTRY. Refuses while the gate is shut, before anything happens."""
    if not LOWPLY_QUALIFICATION_AUTHORIZED:
        raise LowPlyError(
            "the low-ply qualification is UNAUTHORIZED. Gating only the CLI would "
            "protect nothing: a direct Python caller reaches this runner without "
            "passing it. Nothing has been compiled, queried or written.")
    return _run_unguarded(prefixes=prefixes, paths=paths, out_path=out_path,
                          deadline=deadline, budget=budget, _compile=_compile)


def _run_unguarded(*, prefixes, paths, out_path, deadline=None, budget=None,
                   _compile=None):
    """Everything below the gate. PRIVATE, and never a way around the gate.

    It exists so the machinery can be tested WITHOUT lifting the gate in a
    fixture -- a fixture that flips an execution gate IS the gate failing.
    """
    deadline = deadline or Deadline(limit_s=RUN_DEADLINE_S)
    budget = budget or QueryBudget(cap=QUERY_CAP)
    compile_fn = (_compile if _compile is not None
                  else functools.partial(_compile_helper_verified, paths=paths))
    # The clock starts, THEN the supervisor arms from that same clock's remaining
    # time, so the enforced window and the reported one share one origin.
    deadline.start()
    try:
        with _supervisor(deadline):
            return _stages(prefixes, paths, out_path, deadline, budget, compile_fn)
    except LowPlyError:
        raise
    except D1VoidError as e:
        # The deadline, the supervisor's SIGALRM and the budget are D1's classes
        # because the machinery is. REUSING A CHECK MEANS ADOPTING ITS FAILURES:
        # untranslated these escape `main`'s handlers and a fully understood
        # refusal reports as UNEXPECTED, exit 4, instead of VOID, exit 3.
        raise LowPlyVoidError(str(e)) from None
    except D1Error as e:
        # D1BudgetError subclasses this, so one branch covers the budget too. A
        # separate arm for it was redundant -- no test could reach it alone,
        # which is a reason to delete a branch, not a reason to prove it.
        raise LowPlyError(str(e)) from None


def _stages(prefixes, paths, out_path, deadline, budget, compile_fn):
    try:
        artifacts = compile_fn(deadline)
    except AbortError as e:
        raise LowPlyVoidError(f"helper compilation refused: {e.message}. VOID.") from None
    except (ToolchainError, D1Error) as e:
        # Identity is the instrument. A jar or JDK that does not verify, or a
        # javac that fails, says nothing about T1j at low ply.
        raise LowPlyVoidError(f"toolchain or compilation failed: {e}. VOID.") from None
    deadline.check("after helper compilation")

    runtime = INT.T1jRuntime(java=paths.java, jar=paths.jar, classes=paths.classes,
                             ply_cap=paths.ply_cap, timeout_s=PER_CALL_TIMEOUT_S)
    ctx = INT.IntegrationContext()
    binder = INT.make_binder(runtime, ctx)

    out: List[Dict[str, Any]] = []
    n_failures = 0
    for p in prefixes:
        label = prefix_label(p)
        deadline.check(f"before {label}")
        moves = [tuple(m) for m in p["prefix"]]
        state = _replay_state(moves, where=label)
        got = digest_of(state)
        if got != p.get("digest"):
            raise LowPlyVoidError(
                f"{label}: the frozen prefix replays to digest {got}, not the recorded "
                f"{p.get('digest')!r}. The input describes a different position: VOID.")

        bind = _bind_prefix(binder, ctx, label=label, state=state, prefix=moves)
        deadline.check(f"after binding {label}")

        depths: List[Dict[str, Any]] = []
        for depth in DEPTHS:
            invs = [_query_once(moves=moves, depth=depth, paths=paths, state=state,
                                budget=budget, label=label, index=i)
                    for i in range(INVOCATIONS_PER_DEPTH)]
            deadline.check(f"after {label} depth {depth}")
            fails = list(dict.fromkeys(f for inv in invs for f in inv["failures"]))
            # Cross-process agreement, on the fields 12.7 names. A disagreement
            # here is a statement about T1j's per-process Zobrist salt, so it is
            # RECORDED like any other shortfall.
            keys = [(tuple(i["move"]) if i["move"] else None, i["legal"],
                     i["requested_depth"], i["completed_depth"]) for i in invs]
            agree = len(set(keys)) == 1
            if not agree:
                fails.append(f"the {INVOCATIONS_PER_DEPTH} independent JVM invocations "
                             f"disagree: {keys}")
            depths.append({"depth": depth, "invocations": invs, "agree": agree,
                           "failures": fails})

        prefix_fails = ([] if bind["bound"] else [f"E3b replay did not bind: {bind['detail']}"])
        prefix_fails += [f for d in depths for f in d["failures"]]
        n_failures += len(prefix_fails)
        out.append({"task_id": p.get("task_id"), "ply": p.get("ply"),
                    "digest": p.get("digest"), "prefix": [list(m) for m in moves],
                    "opening": p.get("opening"), "colour_arm": p.get("colour_arm"),
                    "signature": p.get("signature"), "role": p.get("role"),
                    "bind": bind, "depths": depths, "failures": prefix_fails,
                    "passed": not prefix_fails})

    deadline.check("before writing the record")
    report = {
        "verdict": "PASS" if n_failures == 0 else "FAIL",
        "n_prefixes": len(out), "n_failures": n_failures,
        "queries_spent": budget.spent, "query_cap": budget.cap,
        "per_call_timeout_s": PER_CALL_TIMEOUT_S,
        "run_deadline_s": deadline.limit_s, "elapsed_s": deadline.elapsed(),
        "depths": list(DEPTHS), "invocations_per_depth": INVOCATIONS_PER_DEPTH,
        "toolchain_identity": artifacts, "prefixes": out,
    }
    fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    return report


EXIT_OK = 0
EXIT_VOID = 3
EXIT_UNEXPECTED = 4
EXIT_UNAUTHORIZED = 5


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI. Refuses while the gate is shut, BEFORE touching anything.

    This is the SECOND of the two guard reads; `run_qualification` carries the
    other. Nothing reads an environment variable, a flag or a configuration file
    to reach the gate.

    NOTE ON THE EXIT STATUS: a `FAIL` verdict exits 0. It is a RESULT, and the
    record carries it. Only the instrument failing is a non-zero abort.
    """
    import argparse
    ap = argparse.ArgumentParser(
        prog="lowply_qualification",
        description="Low-ply T1j qualification. IT IS NOT AUTHORIZED.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefixes")
    ap.add_argument("--java")
    ap.add_argument("--jar")
    ap.add_argument("--classes")
    ap.add_argument("--ply-cap", type=int, default=None)
    a = ap.parse_args(argv)

    if not LOWPLY_QUALIFICATION_AUTHORIZED:
        print("the low-ply qualification is UNAUTHORIZED. No JVM was started, no "
              "position queried, and no file was written.", file=sys.stderr)
        return EXIT_UNAUTHORIZED

    try:                                                      # pragma: no cover
        run_qualification(
            prefixes=load_frozen_prefixes(a.prefixes),
            paths=T1jPaths(java=a.java, jar=a.jar, classes=a.classes,
                           ply_cap=a.ply_cap),
            out_path=a.out)
    except LowPlyVoidError as e:                              # pragma: no cover
        print(f"VOID: {e}", file=sys.stderr)
        return EXIT_VOID
    except LowPlyError as e:                                  # pragma: no cover
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_UNEXPECTED
    return EXIT_OK                                            # pragma: no cover


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
