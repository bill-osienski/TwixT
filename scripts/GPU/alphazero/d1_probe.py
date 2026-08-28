"""D1 -- same-position interrogation: THE FAIL-CLOSED MACHINERY ONLY.

NOTHING HERE EXECUTES D1. No model is loaded, no JVM started, no seed registered
or drawn, no position queried, no game played. `D1_EXECUTION_AUTHORIZED` is
False and every entry point refuses while it is.

The limits are frozen in plan section 12.10 and duplicated here as constants that
the tests pin against the plan's values:

  * 120 s per T1j query -- the qualified E4 preflight limit;
  * 90 minutes whole-run wall clock, on a MONOTONIC clock started before helper
    compilation, because a per-call timeout cannot see compilation, replay,
    binding, incumbent search or output;
  * 1,135 queries = 227 x (1 incumbent + 2 depth-3 + 2 depth-6).

WHY THE TIMEOUT IS PASSED AND RE-PASSED EXPLICITLY. Between a caller and the
process boundary there are three hops that each default to None --
`make_agent_factory(t1j_timeout_s=None)`, `T1jAgent(timeout_s=None)` and
`t1j_adapter.query(timeout_s=None)` -- and `subprocess.run(timeout=None)` waits
forever. A single unforwarded hop silently restores unbounded waiting and raises
nothing, so the tests assert arrival at `subprocess.run`, never at the call site.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import t1j_adapter as A

Pos = Tuple[int, int]

#: D1 EXECUTION IS UNAUTHORIZED. Changing this is a reviewed code change. Read
#: directly, in ONE place. No supported override exists -- not argv, not the
#: environment, not a configuration file, not an import hook.
#:
#: This is D1's OWN gate. `l0_match_command.L0_EXECUTION_AUTHORIZED` and
#: `e4_screen_command.SCREEN_AUTHORIZED` guard DIFFERENT experiments and nothing
#: here reads either: one gate must never be openable by opening another.
D1_EXECUTION_AUTHORIZED = False

#: Frozen in plan section 12.10.
PER_QUERY_TIMEOUT_S = 120
RUN_DEADLINE_S = 90 * 60
QUERY_CAP = 1135
N_POSITIONS = 227
T1J_DEPTHS = (3, 6)
INVOCATIONS_PER_DEPTH = 2
#: Reserved in 12.5 and DELIBERATELY NOT REGISTERED in any seed registry.
SEED_INTERVAL = (202614000, 202614227)


class D1Error(Exception):
    """Base for every D1 refusal."""


class D1VoidError(D1Error):
    """The run is VOID. Never a truncated cohort, never a partial analysis."""


class D1BudgetError(D1Error):
    """The frozen query ceiling would be exceeded."""


@dataclass(frozen=True)
class T1jPaths:
    java: str
    jar: str
    classes: str


class Deadline:
    """Whole-run wall clock on a MONOTONIC source.

    Started explicitly, before helper compilation, so the window covers
    compilation, replay, binding, incumbent work and output -- not just the T1j
    calls a per-query timeout can see.
    """

    def __init__(self, limit_s: float = RUN_DEADLINE_S,
                 clock: Callable[[], float] = time.monotonic):
        self.limit_s = limit_s
        self._clock = clock
        self._t0: Optional[float] = None

    def start(self) -> "Deadline":
        if self._t0 is not None:
            raise D1Error("deadline already started")
        self._t0 = self._clock()
        return self

    @property
    def started(self) -> bool:
        return self._t0 is not None

    def elapsed(self) -> float:
        if self._t0 is None:
            raise D1Error("deadline was never started")
        return self._clock() - self._t0

    def check(self, where: str) -> None:
        """Raise VOID if the window has closed. Called at every stage boundary."""
        if self._t0 is None:
            raise D1VoidError(f"{where}: the run deadline was never started")
        el = self.elapsed()
        if el > self.limit_s:
            raise D1VoidError(
                f"{where}: whole-run deadline exceeded ({el:.1f}s > {self.limit_s}s). "
                f"The run is VOID: no partial-cohort analysis is produced.")


class QueryBudget:
    """The frozen ceiling on queries MADE. It bounds no duration; see Deadline."""

    def __init__(self, cap: int = QUERY_CAP):
        self.cap = cap
        self.spent = 0

    def spend(self, n: int = 1) -> None:
        if self.spent + n > self.cap:
            raise D1BudgetError(
                f"query budget exhausted: {self.spent} + {n} would exceed the frozen "
                f"ceiling of {self.cap}")
        self.spent += n


#: Fields the two independent invocations must agree on (plan 12.7).
AGREEMENT_FIELDS = ("move", "legal", "requested_depth", "completed_depth")


def probe_position(*, moves: Sequence[Pos], depth: int, paths: T1jPaths,
                   budget: QueryBudget, deadline: Deadline) -> Dict[str, Any]:
    """Query ONE prefix at ONE depth, twice, in two separate JVM processes.

    `repeats=1` on both calls is load-bearing: `repeats>1` runs the repeats
    INSIDE ONE JVM, reusing that process's single unseeded Zobrist salt, so it
    cannot vary the cross-process variable the check exists to test.
    """
    if deadline.started:
        deadline.check(f"before querying depth {depth}")
    results = []
    for i in range(INVOCATIONS_PER_DEPTH):
        budget.spend(1)
        try:
            recs, dumps, rc, out = A.query(
                list(moves), depth=depth, java=paths.java, jar=paths.jar,
                classes=paths.classes, repeats=1, timeout_s=PER_QUERY_TIMEOUT_S)
        except subprocess.TimeoutExpired as e:
            raise D1VoidError(
                f"depth {depth} invocation {i}: T1j query timed out after "
                f"{PER_QUERY_TIMEOUT_S}s ({e}). The run is VOID: no partial-cohort "
                f"analysis is produced.") from None
        if rc != 0 or len(recs) != 1:
            raise D1VoidError(
                f"depth {depth} invocation {i}: exit {rc} with {len(recs)} query records")
        results.append((recs[0], dumps))
        if deadline.started:
            deadline.check(f"after depth {depth} invocation {i}")
    a, b = results
    for f in AGREEMENT_FIELDS:
        if getattr(a[0], f) != getattr(b[0], f):
            raise D1VoidError(
                f"depth {depth}: the two independent JVM invocations disagree on {f!r} "
                f"({getattr(a[0], f)!r} vs {getattr(b[0], f)!r}). Two Zobrist salts, two "
                f"answers: VOID, never averaged and never tie-broken.")
    if a[1] != b[1]:
        raise D1VoidError(
            f"depth {depth}: the two invocations replayed different states: VOID.")
    return {"depth": depth, "record": a[0], "dump": a[1],
            "invocations": INVOCATIONS_PER_DEPTH}


def _check_seed(seed: Any) -> int:
    lo, hi = SEED_INTERVAL
    if not isinstance(seed, int) or isinstance(seed, bool) or not lo <= seed < hi:
        raise D1VoidError(
            f"seed {seed!r} is outside the reserved diagnostic interval [{lo}, {hi}). "
            f"D1 may draw from nothing else.")
    return seed


def _check_prefix(pos: Dict[str, Any]) -> List[Pos]:
    """The retained FULL move prefix -- never a digest.

    The E3b adapter advances T1j only by replaying an ordered move sequence; a
    canonical state digest is a deduplication label and cannot be replayed.
    """
    prefix = pos.get("prefix")
    if not isinstance(prefix, (list, tuple)) or not prefix:
        raise D1VoidError(f"{pos.get('task_id')}: no retained move prefix")
    if any(not (isinstance(m, (list, tuple)) and len(m) == 2) for m in prefix):
        raise D1VoidError(f"{pos.get('task_id')}: prefix is not a move sequence")
    ply = pos.get("ply")
    if ply is not None and len(prefix) != ply:
        raise D1VoidError(
            f"{pos.get('task_id')}: prefix holds {len(prefix)} moves but the position is "
            f"at ply {ply}; T1j would search a different position")
    return [tuple(m) for m in prefix]


def run_d1(*, positions: Sequence[Dict[str, Any]], paths: T1jPaths, out_path: str,
           deadline: Optional[Deadline] = None, budget: Optional[QueryBudget] = None,
           _compile: Optional[Callable] = None) -> Dict[str, Any]:
    """Orchestrate the T1j side of D1 under both frozen limits.

    THE DEADLINE STARTS FIRST, before compilation, so the window covers every
    stage. Any VOID propagates as an exception and NOTHING is written: a report
    that exists after a breach is a partial-cohort analysis wearing a runtime
    excuse.
    """
    deadline = (deadline or Deadline()).start()
    budget = budget or QueryBudget()
    compile_fn = _compile if _compile is not None else _default_compile

    compile_fn(deadline)
    deadline.check("after helper compilation")

    out: List[Dict[str, Any]] = []
    for pos in positions:
        _check_seed(pos.get("seed"))
        prefix = _check_prefix(pos)
        deadline.check(f"before position {pos.get('task_id')}@{pos.get('ply')}")
        per_depth = [probe_position(moves=prefix, depth=d, paths=paths,
                                    budget=budget, deadline=deadline)
                     for d in T1J_DEPTHS]
        out.append({"task_id": pos.get("task_id"), "ply": pos.get("ply"),
                    "seed": pos["seed"], "prefix_len": len(prefix),
                    "depths": [{"depth": r["depth"],
                                "move": list(r["record"].move) if r["record"].move else None,
                                "completed_depth": r["record"].completed_depth,
                                "invocations": r["invocations"]} for r in per_depth]})
    deadline.check("before writing the report")

    report = {"n_positions": len(out), "queries_spent": budget.spent,
              "query_cap": budget.cap, "per_query_timeout_s": PER_QUERY_TIMEOUT_S,
              "run_deadline_s": deadline.limit_s, "elapsed_s": deadline.elapsed(),
              "seed_interval": list(SEED_INTERVAL), "positions": out}
    fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    return report


def _default_compile(deadline: Deadline) -> None:
    """Production compile step. Unreachable while the gate is shut."""
    raise D1Error("D1 execution is unauthorized; no helper is compiled")


EXIT_OK = 0
EXIT_VOID = 3
EXIT_UNEXPECTED = 4
EXIT_UNAUTHORIZED = 5


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI. Refuses while the gate is shut, BEFORE touching anything.

    The gate is read exactly once, here, and nothing reads an environment
    variable, a flag or a config file to reach it: opening D1 is a reviewed
    one-line change to `D1_EXECUTION_AUTHORIZED` plus a separate authorization.
    """
    import argparse
    ap = argparse.ArgumentParser(description="D1 same-position interrogation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions")
    ap.add_argument("--java")
    ap.add_argument("--jar")
    ap.add_argument("--classes")
    a = ap.parse_args(argv)

    if not D1_EXECUTION_AUTHORIZED:
        print("D1 execution is UNAUTHORIZED. No model was loaded, no JVM started, "
              "no seed drawn, no position queried, and no file was written.",
              file=sys.stderr)
        return EXIT_UNAUTHORIZED

    try:                                                      # pragma: no cover
        run_d1(positions=json.load(open(a.positions, encoding="utf-8")),
               paths=T1jPaths(java=a.java, jar=a.jar, classes=a.classes),
               out_path=a.out)
    except D1VoidError as e:                                  # pragma: no cover
        print(f"VOID: {e}", file=sys.stderr)
        return EXIT_VOID
    except D1Error as e:                                      # pragma: no cover
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_UNEXPECTED
    return EXIT_OK                                            # pragma: no cover


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
