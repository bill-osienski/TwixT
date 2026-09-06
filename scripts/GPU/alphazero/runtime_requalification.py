"""Java-only RUNTIME REQUALIFICATION of the edited E4Preflight helper -- the GATED
RUNNER. IT IS NOT AUTHORIZED.

Frozen by `docs/superpowers/2026-09-06-t1j-runtime-requalification-card.md`.
The 2026-09-05 additive edit to `E4Preflight.java` (sha fa662339...) was
compile-only verified on 2026-09-06 (buildability). This is the SMALL runtime
sample that would establish OPERATION on the eight frozen screen openings --
including the exact position where the H1 match VOIDed -- and that the new
POSTCOND observation fields survive the REAL adapter/diagnostic path.

WHAT A PASS WOULD AND WOULD NOT ESTABLISH. It establishes that this build
binds, searches depth 6, returns a legal move, re-binds its searched position,
and reports a clean preference surface WITH its own before/after values, on
these eight cases, once each. It does NOT explain the 2026-09-05 failure and
does NOT prove it cannot recur: a shared-directory race is not ruled out by
eight clean samples.

THE SHAPE IS THE LOW-PLY RUNNER'S, BECAUSE IT WAS RIGHT ONCE: FAIL IS A
RESULT, NOT AN ABORT. A T1j reply that does not complete, an illegal move, a
searched-position mismatch, a dirty preference surface -- each is RECORDED and
the run continues. VOID is the INSTRUMENT: identity mismatch, a timeout, a
deadline breach, unparseable output, a reply that could not have come from the
compiled class, or a refusal about the harness.

WHAT IS NEW HERE, AND WHY:
  * THE QUERY GOES THROUGH THE REAL `T1jAgent`, not `A.query` directly, so an
    unclean reply arrives as the same AbortError-from-HelperOutputError chain
    the H1 match's diagnostic reads -- and the record is built with the SAME
    two helpers (`_bounded_excerpt`, `_helper_text`). That is the verification
    the card asks for: the observation survives THAT path, not a test double.
  * THE OBSERVATION'S ABSENCE IS A VOID. The compiled class emits the four
    fields on every POSTCOND line; a reply without them did not come from it.
  * AN OUTER SUPERVISOR KILLS THE WHOLE PROCESS GROUP on timeout. The
    compile-only driver's `subprocess.run(timeout=)` killed the Python worker
    and not necessarily its javac child (review, 2026-09-06). Here the worker
    is started in its OWN SESSION and the group is terminated, then killed,
    then verified empty.
  * EXIT CODES MEAN ONE THING EACH, and a FAIL result is 2, not 0. The
    2026-09-05 wrapper's unconditional exit 0 hid a VOID from anything that
    read only the status.

Preference checks are INTACT: `prefs_ok` is the helper's own verdict, `clean`
still requires it, and a dirty surface is a recorded FAIL on that prefix.

D1's timing machinery (`Deadline`, `QueryBudget`, `_supervisor`) is imported,
never restated; D1's gate is not. This module reads its OWN gate and no other.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import e4_screen_integration as INT
from . import h1_viability_plan as PLAN
from . import h1_viability_runner as DIAG
from . import t1j_adapter as A
from .d1_probe import D1Error, D1VoidError
from .d1_probe import Deadline, QueryBudget, T1jPaths, _supervisor
from .d1_probe import _default_compile as _compile_helper_verified
from .e4_screen_runner import AbortError
from .t1j_toolchain import ToolchainError

Pos = Tuple[int, int]
MODULE = "scripts.GPU.alphazero.runtime_requalification"

#: THE RUNTIME REQUALIFICATION IS UNAUTHORIZED. Changing this is a reviewed code
#: change. Read at ALL THREE public entries -- `main`, `worker_main` and
#: `run_requalification` -- because the supervisor spawns the worker as its own
#: process, and a worker that trusted its parent to have checked would be a gate
#: with a way around it. No override exists: not argv, not the environment, not
#: a configuration file. This is the requalification's OWN gate.
RUNTIME_REQUAL_AUTHORIZED = False

#: Frozen in the card. Eight prefixes = the eight sha-pinned screen openings at
#: the opening ply, ONE depth (the match's mdPly 6), ONE invocation each.
DEPTH = 6
INVOCATIONS_PER_PREFIX = 1
N_PREFIXES = 8
QUERY_CAP = N_PREFIXES * INVOCATIONS_PER_PREFIX          # 8 T1j queries
REPLAY_LAUNCHES = N_PREFIXES                             # 8 E3bDump replays
HELPER_LAUNCHES = QUERY_CAP + REPLAY_LAUNCHES            # 16 processes running the jar
JAVA_PROCESSES = HELPER_LAUNCHES + 1                     # 17: + ONE javac
PER_CALL_TIMEOUT_S = 120
RUN_DEADLINE_S = 900
#: The OUTER, process-tree cap exceeds the inner deadline by this much, so the
#: inner supervisor gets to report a breach as a VOID before the outer one
#: kills the group -- the outer cap is the backstop, not the first responder.
SUPERVISOR_GRACE_S = 60
#: The qualified replay cap, as the match runs use it.
PLY_CAP = 280

#: The frozen input, pinned by content, and cross-checked against the pinned
#: screen plan it was derived from: two frozen sources, one statement.
FROZEN_PREFIXES_REL = ("docs/superpowers/evidence/2026-09-06-t1j-runtime-requalification/"
                       "01_frozen_prefixes.json")
FROZEN_PREFIXES_SHA256 = "86b9e73801b6d643625845aa9b6b82052c1993436c691b34d2ad53401f09a305"
#: The opening whose ply-6 position the 2026-09-05 H1 match VOIDed on.
H1_FAILED_OPENING = "o3_low"

EXIT_PASS = 0
EXIT_FAIL = 2            # a RESULT: T1j fell short on >= 1 prefix; the record says where
EXIT_VOID = 3            # the instrument failed; nothing was measured
EXIT_UNEXPECTED = 4
EXIT_UNAUTHORIZED = 5
EXIT_TIMEOUT = 6         # the outer supervisor killed the process group
EXIT_REFUSED = 7         # a precondition refusal by the harness


class RequalError(Exception):
    """A refusal by the harness. Never a statement about T1j."""


class RequalVoidError(RequalError):
    """The run is VOID: the instrument failed, so nothing was measured."""


# ───────────────────────────── the frozen input ─────────────────────────────

def load_frozen_prefixes(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """The eight frozen prefixes, VERIFIED BY HASH and AGAINST THE PLAN.

    An explicit path is hashed against the SAME pin: the argument relocates the
    file, it does not choose a different one. Then every opening is compared
    with the sha-pinned screen plan's -- a pinned file whose content drifted
    from the pinned plan would pass a hash check alone.
    """
    p = FROZEN_PREFIXES_REL if path is None else path
    try:
        raw = open(p, "rb").read()
    except OSError as e:
        raise RequalError(f"cannot read the frozen prefixes: {e}") from None
    got = hashlib.sha256(raw).hexdigest()
    if got != FROZEN_PREFIXES_SHA256:
        raise RequalError(
            f"{p}: sha256 {got} != the pinned {FROZEN_PREFIXES_SHA256}. The card is "
            f"bound to that exact content; a different file answers a different "
            f"question while looking identical.")
    doc = json.loads(raw)
    prefixes = doc["prefixes"]
    openings = PLAN.load_source_plan()["openings"]
    ours = [(p_["opening"], [list(m) for m in p_["prefix"]]) for p_ in prefixes]
    plans = [(name, [list(m) for m in mv]) for name, mv in openings.items()]
    if ours != plans:
        raise RequalError(
            "the frozen prefix file and the pinned screen plan disagree on the "
            "openings; two frozen sources must say one thing.")
    if len(prefixes) != N_PREFIXES:
        raise RequalError(f"{len(prefixes)} prefixes, the card freezes {N_PREFIXES}")
    if [p_["opening"] for p_ in prefixes if p_.get("h1_failed")] != [H1_FAILED_OPENING]:
        raise RequalError(f"exactly one prefix must be flagged as the H1 failure, "
                          f"{H1_FAILED_OPENING}")
    return prefixes


def prefix_label(p: Dict[str, Any]) -> str:
    return (f"{p.get('opening', '?')}@ply{p.get('ply', '?')}"
            f"{' [H1-FAILED]' if p.get('h1_failed') else ''} "
            f"digest={str(p.get('digest') or '')[:16] or '?'}")


def digest_of(state) -> str:
    from .d1_selection import canonical_digest
    return canonical_digest(state)


def _replay_state(prefix: Sequence[Pos], *, where: str):
    from .game.twixt_state import TwixtState
    state = TwixtState(active_size=A.BOARD_N, to_move="red")
    for i, move in enumerate(prefix):
        if tuple(move) not in set(state.legal_moves()):
            raise RequalVoidError(
                f"{where}: prefix move {i} {tuple(move)} is illegal at ply {state.ply}. "
                f"The frozen input is wrong, which is an instrument failure: VOID.")
        state = state.apply_move(tuple(move))
    return state


# ───────────────────────────── one reply, described ─────────────────────────

def _observe_reply(rec, dumps, out, state, moves, *, where: str) -> Dict[str, Any]:
    """ONE reply, described. T1j's shortcomings are RECORDED, not raised.

    Two things ARE raised, because they are about the instrument: a POSTCOND
    line that cannot be read, and a POSTCOND line WITHOUT the observation the
    compiled class is known to emit.
    """
    failures: List[str] = []
    if not rec.completed or rec.completed_depth != DEPTH:
        failures.append(f"did not complete depth {DEPTH} "
                        f"(completed={rec.completed}, completed_depth={rec.completed_depth})")
    if rec.requested_depth != DEPTH:
        failures.append(f"reply reports requested depth {rec.requested_depth}, not {DEPTH}")
    if rec.null_sentinel or rec.move is None:
        failures.append("returned the null sentinel, not a move")
    if not rec.legal:
        failures.append("T1j reports its own move illegal")
    if rec.move is not None and rec.move not in set(state.legal_moves()):
        failures.append(f"returned {rec.move}, illegal in OUR engine")

    try:
        posts = A.parse_postconds(out)
    except (ValueError, KeyError) as e:
        raise RequalVoidError(
            f"{where}: the helper's POSTCOND output could not be parsed: {e}. "
            f"T1j reported: {A.helper_failure_excerpt(out)}. VOID.") from None
    if len(posts) != 1:
        raise RequalVoidError(
            f"{where}: {len(posts)} POSTCOND lines, expected exactly 1. VOID.")
    post = posts[0]
    observed = {"prefs_before": post.prefs_before, "prefs_after": post.prefs_after,
                "count_before": post.count_before, "count_after": post.count_after}
    if all(v is None for v in observed.values()):
        raise RequalVoidError(
            f"{where}: the POSTCOND line carries no preference observation. The "
            f"compiled E4Preflight emits prefs_before/prefs_after/count_before/"
            f"count_after on every line, so this reply did not come from the class "
            f"the run compiled: instrument, VOID.")
    # THE DIAGNOSTIC READER MUST AGREE WITH THE PARSER. Two readers of one line.
    prefs_observation = A.postcond_prefs_observation(out)
    if prefs_observation != {"prefs_ok": post.prefs_ok, **observed}:
        raise RequalVoidError(
            f"{where}: the parser and the diagnostic reader disagree on the "
            f"preference observation ({prefs_observation} vs {observed}). VOID.")
    if not post.clean:
        failures.append(f"postcondition surface not clean (preference or other): {post}")
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
                    "n_legal": len(dumps[0].legal), "history_len": len(dumps[0].history)}

    return {"where": where, "move": list(rec.move) if rec.move else None,
            "requested_depth": rec.requested_depth,
            "completed_depth": rec.completed_depth, "completed": rec.completed,
            "legal": rec.legal, "null_sentinel": rec.null_sentinel,
            "to_move": rec.to_move, "current_max_ply": rec.current_max_ply,
            "usealphabeta": rec.usealphabeta, "eval_regime": rec.eval_regime,
            "elapsed_us": rec.elapsed_us,
            "postcond": {"clean": post.clean, "prefs_ok": post.prefs_ok,
                         "refl_n": post.refl_n, "failures": post.failures,
                         "no_throw": post.no_throw, "windows": post.windows,
                         "frames": post.frames, "headless": post.headless,
                         "refl_ok": post.refl_ok, **observed},
            "prefs_observation": prefs_observation,
            "searched_state": searched, "failures": failures}


def _query_once(*, runtime, ctx, state, moves: Sequence[Pos], budget: QueryBudget,
                where: str) -> Dict[str, Any]:
    """ONE query, THROUGH THE REAL AGENT, with the reply kept for the record.

    `T1jAgent` is the object the H1 match queries with. Its `_query` seam is
    used to KEEP the raw reply (the agent returns only a move) -- it still calls
    `A.query`, the one qualified query path. An `AbortError` is what a real
    shortfall looks like on the match path, so it is caught HERE and turned into
    a recorded FAIL, with the diagnostic built by the H1 runner's own helpers.
    """
    box: Dict[str, Any] = {}

    def keep(qmoves, **kw):
        recs, dumps, rc, out = A.query(qmoves, **kw)
        box.update(recs=recs, dumps=dumps, rc=rc, out=out)
        return recs, dumps, rc, out

    budget.spend(1)
    agent = INT.T1jAgent(runtime=runtime, ctx=ctx, depth=DEPTH, colour=state.to_move,
                         timeout_s=PER_CALL_TIMEOUT_S, _query=keep)
    abort: Optional[AbortError] = None
    try:
        agent(state)
    except subprocess.TimeoutExpired as e:
        raise RequalVoidError(
            f"{where}: T1j did not answer within {PER_CALL_TIMEOUT_S}s ({e}). "
            f"A hung process is an instrument failure: VOID.") from None
    except A.HelperOutputError as e:
        raise RequalVoidError(
            f"{where}: the helper's output could not be parsed ({e}). "
            f"T1j reported: {A.helper_failure_excerpt(e.stdout)}. VOID.") from None
    except AbortError as e:
        abort = e

    if "out" not in box:
        raise RequalVoidError(f"{where}: no reply reached the adapter: VOID.")
    # A NON-ZERO EXIT IS EXPECTED AND IS NOT A VOID: E4Preflight exits 3 when
    # its own `failures` counter is set, which is the condition being measured.
    if len(box["recs"]) != 1:
        raise RequalVoidError(
            f"{where}: exit {box['rc']} and no usable query record "
            f"({len(box['recs'])} parsed). T1j reported: "
            f"{A.helper_failure_excerpt(box['out'])}. VOID.")

    obs = _observe_reply(box["recs"][0], box["dumps"], box["out"], state, moves,
                         where=where)
    obs["exit_status"] = box["rc"]
    if abort is None:
        obs.update(abort=None, helper_excerpt=None, helper_prefs_observed=None)
    else:
        # THE H1 DIAGNOSTIC PATH, verbatim: bounded excerpt from the chain, and
        # the observation parsed from the fullest text reachable.
        obs.update(abort={"phase": abort.phase, "message": abort.message},
                   helper_excerpt=DIAG._bounded_excerpt(abort),
                   helper_prefs_observed=A.postcond_prefs_observation(
                       DIAG._helper_text(abort) or ""))
        if not obs["failures"]:
            obs["failures"].append(f"the agent refused the reply: [{abort.phase}] "
                                   f"{abort.message}")
    return obs


def _bind_prefix(binder: Callable, ctx, *, label: str, state,
                 prefix: Sequence[Pos]) -> Dict[str, Any]:
    """The E3b replay bind. A divergence is RECORDED; a timeout VOIDs."""
    ctx.reset(label, [tuple(m) for m in prefix])
    try:
        binder({"task_id": label}, state, state.ply)
    except subprocess.TimeoutExpired as e:
        raise RequalVoidError(
            f"{label}: the prefix replay did not answer within "
            f"{PER_CALL_TIMEOUT_S}s ({e}): VOID.") from None
    except A.HelperOutputError as e:
        raise RequalVoidError(
            f"{label}: the replay output could not be parsed ({e}). "
            f"T1j reported: {A.helper_failure_excerpt(e.stdout)}. VOID.") from None
    except AbortError as e:
        return {"bound": False, "detail": e.message}
    return {"bound": True, "detail": None}


# ───────────────────────────────── the run ──────────────────────────────────

def run_requalification(*, prefixes: Sequence[Dict[str, Any]], paths: T1jPaths,
                        out_path: str, deadline: Optional[Deadline] = None,
                        budget: Optional[QueryBudget] = None,
                        _compile: Optional[Callable] = None) -> Dict[str, Any]:
    """PUBLIC ENTRY. Refuses while the gate is shut, before anything happens.
    Then requires EXACTLY the frozen eight, in order."""
    if not RUNTIME_REQUAL_AUTHORIZED:
        raise RequalError(
            "the runtime requalification is UNAUTHORIZED. Gating only the CLI would "
            "protect nothing: a direct Python caller reaches this runner without "
            "passing it. Nothing has been compiled, queried or written.")
    frozen = load_frozen_prefixes()
    if [p["digest"] for p in prefixes] != [p["digest"] for p in frozen]:
        raise RequalError("the run must execute EXACTLY the frozen eight prefixes, in "
                          "the frozen order; a subset or a reordering answers a "
                          "different question.")
    return _run_unguarded(prefixes=prefixes, paths=paths, out_path=out_path,
                          deadline=deadline, budget=budget, _compile=_compile)


def _run_unguarded(*, prefixes, paths, out_path, deadline=None, budget=None,
                   _compile=None):
    """Everything below the gate. PRIVATE, and never a way around the gate."""
    deadline = deadline or Deadline(limit_s=RUN_DEADLINE_S)
    budget = budget or QueryBudget(cap=QUERY_CAP)
    compile_fn = (_compile if _compile is not None
                  else functools.partial(_compile_helper_verified, paths=paths))
    deadline.start()
    try:
        with _supervisor(deadline):
            return _stages(prefixes, paths, out_path, deadline, budget, compile_fn)
    except RequalError:
        raise
    except D1VoidError as e:
        raise RequalVoidError(str(e)) from None
    except D1Error as e:
        raise RequalError(str(e)) from None


def _stages(prefixes, paths, out_path, deadline, budget, compile_fn):
    if os.path.lexists(out_path):
        raise FileExistsError(f"the record path already exists: {out_path}")
    try:
        artifacts = compile_fn(deadline)
    except AbortError as e:
        raise RequalVoidError(f"helper compilation refused: {e.message}. VOID.") from None
    except (ToolchainError, D1Error) as e:
        raise RequalVoidError(f"toolchain or compilation failed: {e}. VOID.") from None
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
            raise RequalVoidError(
                f"{label}: the frozen prefix replays to digest {got}, not the recorded "
                f"{p.get('digest')!r}. The input describes a different position: VOID.")
        bind = _bind_prefix(binder, ctx, label=label, state=state, prefix=moves)
        deadline.check(f"after binding {label}")
        query = _query_once(runtime=runtime, ctx=ctx, state=state, moves=moves,
                            budget=budget, where=f"{label}: depth {DEPTH}")
        deadline.check(f"after querying {label}")
        fails = ([] if bind["bound"] else [f"E3b replay did not bind: {bind['detail']}"])
        fails += query["failures"]
        n_failures += len(fails)
        out.append({"opening": p.get("opening"), "ply": p.get("ply"),
                    "digest": p.get("digest"), "prefix": [list(m) for m in moves],
                    "h1_failed": bool(p.get("h1_failed")),
                    "bind": bind, "query": query, "failures": fails,
                    "passed": not fails})

    deadline.check("before writing the record")
    failed = [r for r in out if r["h1_failed"]]
    report = {
        "verdict": "PASS" if n_failures == 0 else "FAIL",
        "establishes": ("operation of THIS build on these prefixes, once each; NOT an "
                        "explanation of the 2026-09-05 failure and NOT proof it "
                        "cannot recur"),
        "n_prefixes": len(out), "n_failures": n_failures,
        "queries_spent": budget.spent, "query_cap": budget.cap,
        "per_call_timeout_s": PER_CALL_TIMEOUT_S,
        "run_deadline_s": deadline.limit_s, "elapsed_s": deadline.elapsed(),
        "depth": DEPTH, "invocations_per_prefix": INVOCATIONS_PER_PREFIX,
        "h1_failed_prefix": (None if not failed else
                             {"opening": failed[0]["opening"], "passed": failed[0]["passed"],
                              "failures": failed[0]["failures"]}),
        "toolchain_identity": artifacts, "prefixes": out,
    }
    fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    return report


# ───────────────── the OUTER supervisor: a whole process group ──────────────

def supervise(cmd: Sequence[str], *, timeout_s: float,
              kill_grace_s: float = 5.0) -> Dict[str, Any]:
    """Run `cmd` in ITS OWN SESSION and, on timeout, terminate the WHOLE GROUP.

    `subprocess.run(timeout=)` kills only the direct child. A Python worker
    whose javac or java child outlives it is exactly what the compile-only
    driver could not rule out. `start_new_session=True` makes the worker the
    leader of a fresh process group, so `killpg` reaches every descendant that
    did not itself change group -- the JDK does not. SIGTERM first, a grace
    period, then SIGKILL, then the group is PROBED (signal 0) until it is empty
    or a bounded wait runs out; the result says which.
    """
    p = subprocess.Popen(list(cmd), start_new_session=True)
    pgid = p.pid                                   # its own session leader
    timed_out = False
    try:
        rc = p.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _killpg(pgid, signal.SIGTERM)
        try:
            p.wait(timeout=kill_grace_s)
        except subprocess.TimeoutExpired:
            pass
        _killpg(pgid, signal.SIGKILL)
        p.wait()
        rc = EXIT_TIMEOUT
    return {"exit_code": rc, "timed_out": timed_out,
            "group_cleared": _group_cleared(pgid, wait_s=kill_grace_s + 3.0)}


def _killpg(pgid: int, sig) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _group_cleared(pgid: int, *, wait_s: float) -> bool:
    """True once NO process remains in the group (killpg(…, 0) -> ESRCH)."""
    end = time.monotonic() + wait_s
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        if time.monotonic() >= end:
            return False
        time.sleep(0.05)


# ────────────────────────────────── the CLI ─────────────────────────────────

def _parser():
    import argparse
    ap = argparse.ArgumentParser(
        prog="runtime_requalification",
        description="Java-only runtime requalification of E4Preflight. NOT AUTHORIZED.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefixes")
    ap.add_argument("--java")
    ap.add_argument("--jar")
    ap.add_argument("--classes")
    ap.add_argument("--ply-cap", type=int, default=PLY_CAP)
    ap.add_argument("--worker", action="store_true",
                    help="internal: run the stages in this process (spawned by main)")
    return ap


def _resolved_paths(a) -> T1jPaths:
    """The VERIFIED toolchain's paths, never retyped; classes beside the record."""
    if a.java and a.jar and a.classes:
        return T1jPaths(java=a.java, jar=a.jar, classes=a.classes, ply_cap=a.ply_cap)
    from . import t1j_toolchain as TC
    tc = TC.verified_paths()
    return T1jPaths(java=os.path.join(tc["jdk_home"], "bin", "java"), jar=tc["jar"],
                    classes=a.out + ".t1j_classes", ply_cap=a.ply_cap)


def worker_main(argv: Optional[Sequence[str]] = None) -> int:
    """The stages, in THIS process. Reads the gate itself: it does not trust
    the parent. Exit codes: PASS 0 / FAIL 2 / VOID 3 / UNEXPECTED 4 /
    UNAUTHORIZED 5 / REFUSED 7."""
    a = _parser().parse_args(argv)
    if not RUNTIME_REQUAL_AUTHORIZED:
        print("the runtime requalification is UNAUTHORIZED. No JVM was started, no "
              "position queried, and no file was written.", file=sys.stderr)
        return EXIT_UNAUTHORIZED
    try:
        report = run_requalification(prefixes=load_frozen_prefixes(a.prefixes),
                                     paths=_resolved_paths(a), out_path=a.out)
    except RequalVoidError as e:
        print(f"VOID: {e}", file=sys.stderr)
        return EXIT_VOID
    except RequalError as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as e:                                    # noqa: BLE001
        print(f"UNEXPECTED {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_UNEXPECTED
    print(f"{report['verdict']}: {report['n_failures']} failure(s) over "
          f"{report['n_prefixes']} prefixes, {report['queries_spent']} queries")
    return EXIT_PASS if report["verdict"] == "PASS" else EXIT_FAIL


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI. Refuses while the gate is shut, BEFORE spawning anything. Otherwise
    supervises a WORKER subprocess under the outer process-tree cap and passes
    its exit code through; a kill is EXIT_TIMEOUT."""
    argv = list(sys.argv[1:] if argv is None else argv)
    a = _parser().parse_args(argv)
    if a.worker:
        return worker_main(argv)
    if not RUNTIME_REQUAL_AUTHORIZED:
        print("the runtime requalification is UNAUTHORIZED. No worker was spawned, "
              "no JVM started, and no file written.", file=sys.stderr)
        return EXIT_UNAUTHORIZED
    r = supervise([sys.executable, "-m", MODULE, "--worker", *argv],
                  timeout_s=RUN_DEADLINE_S + SUPERVISOR_GRACE_S, kill_grace_s=5.0)
    if r["timed_out"]:
        print(f"TIMEOUT: the worker exceeded {RUN_DEADLINE_S + SUPERVISOR_GRACE_S}s; "
              f"its process group was killed (cleared={r['group_cleared']}).",
              file=sys.stderr)
    return r["exit_code"]


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
