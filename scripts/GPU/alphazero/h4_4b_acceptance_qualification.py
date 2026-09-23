"""H4 §4B ACCEPTANCE-MODE QUALIFICATION -- THE GATED RUNNER. IT IS NOT AUTHORIZED.

Frozen by `docs/superpowers/2026-09-22-t1j-h4-4b-acceptance-qualification-card.md`.
🔴 THE CARD IS THE AUTHORITY. If this runner disagrees with it, the card wins and
the disagreement is a design amendment, not a code fix.

WHAT IT ESTABLISHES: that the PRODUCTION adapter -- `make_binder` and the
`T1jAgent` built by `make_agent_factory`, on one H4-mode `T1jRuntime` -- accepts
the helper's replies at the 16 frozen positions, refuses none of them, names
which routine answered, and records one PROC per JVM. It drives the production
agent and binder DIRECTLY over the pinned matrix, never through the game loop:
no game is played, no seed drawn, no score computed.

🔴 STOP AND VOID ARE DECIDED BY THE TYPE RAISED (card §8). An `AbortError` from
the adapter is a STOP -- a result about the ADAPTER, never about T1j -- even when
its cause is a `HelperOutputError` carrying the transcript. A timeout, or output
a parser cannot read, arrives UNCONVERTED and is a VOID.
"""
from __future__ import annotations

import functools
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import e4_screen_integration as INT
from . import h4_4a_characterization as H4A
from . import h4_repair_qualification as H4RQ
from .d1_probe import D1Error, D1VoidError, Deadline, QueryBudget, T1jPaths, _supervisor
from .d1_probe import _default_compile as _compile_helper_verified
from .e4_screen_runner import AbortError
from .t1j_toolchain import ToolchainError

#: THE §4B QUALIFICATION IS NOT AUTHORIZED. Changing this is a reviewed one-line
#: code change. Read directly at BOTH public entry points -- `run_qualification`
#: and `main` -- because gating only the CLI protects nothing. No supported
#: override exists: not argv, not the environment, not a configuration file, not
#: an import hook. This is the qualification's OWN gate; it reads no other.
H4_4B_ACCEPTANCE_QUALIFICATION_AUTHORIZED = True

#: Consumed from the repair qualification, never retyped (card §3.3, §7).
DEPTH = H4RQ.DEPTH
PER_CALL_TIMEOUT_S = H4RQ.PER_CALL_TIMEOUT_S
RUN_DEADLINE_S = H4RQ.RUN_DEADLINE_S
COLOURS = ("red", "black")


class H4BError(Exception):
    """A refusal by the harness before it starts. Never a statement about T1j."""


class H4BStop(H4BError):
    """A frozen STOP fired (card §8). A RESULT; it ends §4B."""


class H4BVoidError(H4BError):
    """VOID: the instrument was unreadable, so nothing was measured."""


def _task(task_id: str, t1j_colour: str) -> Dict[str, Any]:
    """The task shape `make_agent_factory` reads: T1j plays `t1j_colour`."""
    ref = "black" if t1j_colour == "red" else "red"
    return {"task_id": task_id, "reference_colour": ref, "t1j_mdPly": DEPTH}


def _no_reference(task, evaluator=None):
    raise H4BError("§4B builds no reference agent; the factory must never ask for one")


def build_adapter(paths: T1jPaths):
    """The production adapter on ONE H4 runtime and ONE context (card §7).

    Built BEFORE the destination is claimed and before compilation, so every
    construction refusal -- depth, query timeout, pairing -- writes nothing and
    starts no JVM.
    """
    ctx = INT.IntegrationContext()
    runtime = INT.T1jRuntime(java=paths.java, jar=paths.jar, classes=paths.classes,
                             ply_cap=paths.ply_cap, timeout_s=PER_CALL_TIMEOUT_S,
                             h4_acceptance=True)
    binder = INT.make_binder(runtime, ctx)
    factory = INT.make_agent_factory(runtime=runtime, ctx=ctx, evaluator=None,
                                     reference_build=_no_reference,
                                     t1j_timeout_s=PER_CALL_TIMEOUT_S)
    agents = {c: factory(_task("construction", c), c) for c in COLOURS}
    return ctx, runtime, binder, agents


def all_records(ctx) -> List[Dict[str, Any]]:
    return [r for bucket in ctx.processes.values() for r in bucket]


def run_qualification(*, paths: T1jPaths, out_path: str,
                      prefixes: Optional[Sequence[Dict[str, Any]]] = None,
                      deadline: Optional[Deadline] = None,
                      _compile: Optional[Callable] = None) -> Dict[str, Any]:
    """PUBLIC ENTRY. Refuses while the gate is shut, before anything happens."""
    if not H4_4B_ACCEPTANCE_QUALIFICATION_AUTHORIZED:
        raise H4BError(
            "the H4 §4B acceptance qualification is UNAUTHORIZED. Nothing has been "
            "built, compiled, queried or written.")
    return _run_unguarded(paths=paths, out_path=out_path, prefixes=prefixes,
                          deadline=deadline, _compile=_compile)


def _run_unguarded(*, paths, out_path, prefixes=None, deadline=None, _compile=None):
    """Everything below the gate. PRIVATE, and never a way around the gate."""
    try:
        matrix = H4RQ.derive_matrix(prefixes)
        digest = H4RQ.verify_matrix(matrix)
    except H4RQ.H4RQError as e:
        raise H4BError(f"the frozen matrix did not reproduce: {e}") from None
    caps = H4RQ.derived_caps(matrix)
    try:
        ctx, runtime, binder, agents = build_adapter(paths)
    except AbortError as e:
        raise H4BError(f"the adapter refused construction: {e}") from None
    deadline = deadline or Deadline(limit_s=RUN_DEADLINE_S)
    budget = QueryBudget(cap=caps["total"])
    compile_fn = (_compile if _compile is not None
                  else functools.partial(_compile_helper_verified, paths=paths))
    H4RQ._claim(out_path)              # BEFORE compilation, BEFORE any JVM
    deadline.start()
    state_box: Dict[str, Any] = {"row": None}
    try:
        with _supervisor(deadline):
            return _run_matrix(matrix, digest, caps, ctx, binder, agents, out_path,
                               deadline, budget, compile_fn, state_box)
    except AbortError as e:
        cause = e.__cause__
        _write_ended(out_path, "STOP", e, getattr(cause, "stdout", None), ctx,
                     state_box, digest, caps, budget, deadline)
        raise H4BStop(str(e)) from e
    except H4BStop as e:
        _write_ended(out_path, "STOP", e, None, ctx, state_box, digest, caps,
                     budget, deadline)
        raise
    except (subprocess.TimeoutExpired, ValueError, KeyError, D1VoidError, D1Error,
            ToolchainError) as e:
        _write_ended(out_path, "VOID", e, getattr(e, "stdout", None), ctx, state_box,
                     digest, caps, budget, deadline)
        raise H4BVoidError(f"{type(e).__name__}: {e}") from e


def _write_ended(out_path, verdict, exc, stdout, ctx, state_box, digest, caps,
                 budget, deadline) -> None:
    """A STOP or VOID leaves a DURABLE record -- retries are forbidden, so this
    one observation of the failure is the only one there will ever be."""
    records = all_records(ctx)
    H4RQ._write_record(out_path, {
        "stage": "h4_4b_acceptance_qualification",
        "verdict": verdict,
        "reason": str(exc),
        "exception": type(exc).__name__,
        "failing_position": state_box["row"],
        "stdout": stdout if isinstance(stdout, str) else None,
        "matrix_sha256": digest,
        "derived_caps": caps,
        "subprocesses_spent": budget.spent,
        "records_completed": len(records),
        "records": records,
        "elapsed_s": deadline.elapsed() if deadline.started else None,
        "scope": ("A STOP or VOID ENDS §4B under its card: no repair, no retry, no "
                  "reinterpretation, no second run. The records below are those "
                  "completed before it and are NOT a partial qualification."),
    })


def _run_matrix(matrix, digest, caps, ctx, binder, agents, out_path, deadline,
                budget, compile_fn, state_box):
    try:
        artifacts = compile_fn(deadline)
    except (ToolchainError, D1Error) as e:
        raise D1VoidError(f"toolchain or compilation failed: {e}") from None
    deadline.check("after helper compilation")

    for row in matrix:
        state_box["row"] = {k: row[k] for k in ("family", "ply", "prefix")}
        label = f"{row['family']}@ply{row['ply']}"
        deadline.check(f"before {label}")
        moves = [tuple(m) for m in row["prefix"]]
        state = H4A._state_for(moves, where=label)
        if state.ply != int(row["ply"]):
            raise D1VoidError(f"{label}: replays to ply {state.ply}, not {row['ply']}")
        task_id = f"h4_4b-{row['family']}-ply{row['ply']}"
        ctx.reset(task_id, moves)
        budget.spend(1)
        binder({"task_id": task_id}, state, state.ply, None)
        for rep in range(H4RQ.repetitions_for(int(row["ply"]))):
            budget.spend(1)
            agents[state.to_move](state)
            deadline.check(f"after {label} query rep{rep}")

    state_box["row"] = None
    records = all_records(ctx)
    pids = {r["proc"]["pid"] for r in records if r.get("proc")}
    for ok, msg in ((budget.spent == caps["total"],
                     f"spent {budget.spent} subprocesses, expected the derived {caps['total']}"),
                    (len(records) == caps["total"],
                     f"{len(records)} process records, expected {caps['total']}"),
                    (all(r["outcome"] == "accepted" for r in records),
                     "a record that is not accepted reached the end of the run"),
                    (len(pids) == caps["total"],
                     f"{len(pids)} distinct pids, expected {caps['total']}")):
        if not ok:
            raise H4BStop(msg)

    by_ply: Dict[str, Dict[str, int]] = {}
    for r in records:
        if r["role"] == "query":
            cell = by_ply.setdefault(str(r["board_ply"]), {})
            cell[r["source"]] = cell.get(r["source"], 0) + 1
    report = {
        "stage": "h4_4b_acceptance_qualification",
        "verdict": "CLEAN",
        "matrix_sha256": digest,
        "n_positions": len(matrix),
        "derived_caps": caps,
        "subprocesses_spent": budget.spent,
        "n_records": len(records),
        "distinct_pids": len(pids),
        "depth": DEPTH,
        "per_call_timeout_s": PER_CALL_TIMEOUT_S,
        "run_deadline_s": deadline.limit_s,
        "elapsed_s": deadline.elapsed(),
        "source_by_ply": by_ply,
        "toolchain_identity": artifacts,
        "records": records,
        "scope": (
            "The 16 frozen positions, depth 6, driven through the PRODUCTION agent "
            "and binder in H4 mode -- never the game loop. Establishes that the "
            "adapter accepts these replies, refuses none, names the answering "
            "routine and records one PROC per JVM. native_initial_fifth_or_more is "
            "exercised only by constructed tests. No game, no seed, no score."),
    }
    H4RQ._write_record(out_path, report)
    return report


EXIT_OK = 0
EXIT_STOP = 2
EXIT_VOID = 3
EXIT_UNEXPECTED = 4
EXIT_UNAUTHORIZED = 5


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI. Refuses while the gate is shut, BEFORE touching anything."""
    import argparse
    ap = argparse.ArgumentParser(
        prog="h4_4b_acceptance_qualification",
        description="H4 §4B acceptance-mode qualification. IT IS NOT AUTHORIZED.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", required=True)
    a = ap.parse_args(argv)

    if not H4_4B_ACCEPTANCE_QUALIFICATION_AUTHORIZED:
        print("the H4 §4B acceptance qualification is UNAUTHORIZED. No JVM was "
              "started, nothing was built, and no file was written.", file=sys.stderr)
        return EXIT_UNAUTHORIZED

    try:                                                      # pragma: no cover
        run_qualification(paths=H4RQ.resolve_paths(a.classes), out_path=a.out)
    except H4BStop as e:                                      # pragma: no cover
        print(f"STOP: {e}", file=sys.stderr)
        return EXIT_STOP
    except H4BVoidError as e:                                 # pragma: no cover
        print(f"VOID: {e}", file=sys.stderr)
        return EXIT_VOID
    except H4BError as e:                                     # pragma: no cover
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_UNEXPECTED
    except Exception as e:                                    # noqa: BLE001
        print(f"UNEXPECTED {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_UNEXPECTED
    return EXIT_OK                                            # pragma: no cover


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
