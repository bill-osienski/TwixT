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

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import d1_selection as SEL
from . import e4_screen_integration as INT
from . import t1j_adapter as A
from .e4_screen_runner import AbortError

Pos = Tuple[int, int]

#: D1 EXECUTION IS UNAUTHORIZED. Changing this is a reviewed code change. Read
#: directly, at BOTH public entry points -- `run_d1` and `main` -- because gating
#: only the CLI protected nothing: a direct Python caller reached the runner
#: without passing any gate. A test counts the Load-context reads and requires at
#: least two, so dropping either one fails. No supported override exists -- not
#: argv, not the environment, not a configuration file, not an import hook.
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
    """The pinned runtime, with BOTH bounds carried explicitly.

    `ply_cap` has no default here on purpose. Plan 5.5 [A1] records that
    `play_task` declares `ply_cap: int = PLY_CAP`, so an omitted cap is silently
    defaulted rather than refused -- the hazard is the silence. A cap that must
    be named cannot be forgotten quietly.
    """
    java: str
    jar: str
    classes: str
    ply_cap: int


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


@contextlib.contextmanager
def _supervisor(deadline: "Deadline"):
    """Terminate the run at the deadline even if a stage is BLOCKED.

    `Deadline.check` is cooperative: it runs between stages and cannot interrupt
    a hung compilation, replay, readout or write. SIGALRM can, because it
    interrupts the blocking call itself.

    FAILS CLOSED. If the timer cannot be armed -- not the main thread, or no
    SIGALRM on this platform -- the run REFUSES rather than silently falling back
    to cooperative-only checking, which is how a safeguard becomes decorative.

    ponytail: SIGALRM interrupts blocking syscalls and Python bytecode, not a
    C extension that never returns; a subprocess-level supervisor is the upgrade
    if that ever becomes the failure mode.
    """
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not \
            threading.main_thread():
        raise D1Error(
            "the whole-run supervisor cannot be armed here (needs SIGALRM on the main "
            "thread); refusing rather than degrading to cooperative checks only")

    def _fire(_signum, _frame):
        raise D1VoidError(
            f"whole-run deadline of {deadline.limit_s}s exceeded and the run was "
            f"TERMINATED mid-stage. VOID: no partial-cohort analysis is produced.")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, deadline.limit_s)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


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


def _validate_reply(rec, dumps, *, depth: int, n_moves: int, where: str) -> None:
    """Each reply must be valid ON ITS OWN. AGREEMENT IS NOT VALIDITY.

    Two equally invalid answers agree perfectly, and an earlier version of this
    module accepted exactly that: it compared the pair and never asked whether
    either was usable. Plan 12.7 requires legality, completion at the requested
    depth, a real move, and a dump -- of every reply.
    """
    if rec.null_sentinel or rec.move is None:
        raise D1VoidError(f"{where}: T1j returned the null sentinel, not a move: VOID")
    if not rec.legal:
        raise D1VoidError(f"{where}: T1j returned an illegal move {rec.move}: VOID")
    if not rec.completed:
        raise D1VoidError(f"{where}: T1j did not complete its search: VOID")
    if rec.requested_depth != depth or rec.completed_depth != depth:
        raise D1VoidError(
            f"{where}: requested depth {depth} but the reply reports requested="
            f"{rec.requested_depth} completed depth={rec.completed_depth}: VOID")
    # A MISSING DUMP IS NOT AN AGREEING DUMP. Two empty dumps compare equal, so
    # without a cardinality check an absent dump passed the pair comparison.
    if not dumps:
        raise D1VoidError(f"{where}: the reply carried no state dump at all: VOID")
    if dumps[-1].ply != n_moves:
        raise D1VoidError(
            f"{where}: the dump ends at ply {dumps[-1].ply} but the retained prefix "
            f"holds {n_moves} moves, so T1j searched a different position: VOID")


#: Fields the two independent invocations must agree on (plan 12.7).
AGREEMENT_FIELDS = ("move", "legal", "requested_depth", "completed_depth")


def _probe_position(*, moves: Sequence[Pos], depth: int, paths: T1jPaths,
                    budget: QueryBudget, deadline: Deadline) -> Dict[str, Any]:
    """Query ONE prefix at ONE depth, twice, in two separate JVM processes.

    PRIVATE. It was public, which reopened the very hole the CLI gate was meant
    to close: a direct caller could issue T1j queries without passing any gate.
    `run_d1` is the single public execution entry and it checks the gate first.

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
        _validate_reply(recs[0], dumps, depth=depth, n_moves=len(moves),
                        where=f"depth {depth} invocation {i}")
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


def _check_seed_registration() -> None:
    """The reserved block must be REGISTERED before D1 uses any of it (12.5).

    READS the registry; never writes one. Registering the interval is a reviewed
    edit to `e4_screen_reference.ACCOUNTED_SEED_INTERVALS` and belongs to the D1
    EXECUTION authorization -- "a reserved-on-paper block that is never
    authorized must cost nothing to abandon". A runtime mutation would make the
    registry a thing the run can grant itself, which is the same shape as a gate
    that opens its own gate.

    EVERY seed in the block is checked, not the endpoints: a partial registration
    would otherwise pass and then draw an unaccounted seed halfway through.

    Availability -- exposed, retired, already consumed -- is a DIFFERENT question
    and is asked per position by `e4_screen_reference.validate_task_executable`
    when the incumbent agent is built. This asks only whether the block has been
    accounted for at all.
    """
    from . import e4_screen_reference as REF
    lo, hi = SEED_INTERVAL
    missing = [s for s in range(lo, hi) if not REF.seed_is_accounted(s)]
    if missing:
        raise D1Error(
            f"the D1 diagnostic seed block [{lo}, {hi}) is not registered: "
            f"{len(missing)} of {hi - lo} seeds are absent from "
            f"ACCOUNTED_SEED_INTERVALS (first {missing[0]}). Registering it is a "
            f"reviewed edit to that registry, part of the D1 EXECUTION "
            f"authorization; nothing here writes a registry at runtime.")


def _replay_prefix(prefix: Sequence[Pos], *, where: str):
    """OUR state at the retained position, replayed move by move.

    Legality is checked against our own engine before T1j ever sees the
    sequence, so an unplayable prefix is refused here rather than diagnosed
    from a divergence report later.
    """
    from .game.twixt_state import TwixtState
    state = TwixtState(active_size=A.BOARD_N, to_move="red")
    for i, move in enumerate(prefix):
        if move not in set(state.legal_moves()):
            raise D1VoidError(
                f"{where}: prefix move {i} {move} is illegal at ply {state.ply}: VOID")
        state = state.apply_move(move)
    return state


def _check_digest(state, pos: Dict[str, Any], *, where: str) -> None:
    """12.7: a prefix that does not replay to its recorded digest is a VOID."""
    got = SEL.canonical_digest(state)
    if got != pos.get("digest"):
        raise D1VoidError(
            f"{where}: the retained prefix replays to digest {got}, not the recorded "
            f"{pos.get('digest')!r}. It is a different position: VOID.")


def _bind_prefix(binder: Callable, ctx, *, task_id: str, state,
                 prefix: Sequence[Pos]) -> None:
    """5.5: reuse the E3b binder for every replayed position, unchanged.

    The whole prefix is submitted at once, which is the binder's `move=None`
    opening path -- one replay, one postcondition surface, one comparison of
    pegs, bridges, side to move, ply, the full legal set, terminal attribution
    and T1j's own ordered history.

    THE TRANSLATION IS THE POINT. `make_binder` raises `e4_screen_runner
    .AbortError`, which is not a `D1Error`; untranslated it escapes `main`'s
    handlers and a fully understood refusal is reported as UNEXPECTED, exit 4,
    instead of VOID, exit 3. `l0_match_command._delegate` exists because the
    same defect shipped once already: reusing a check means adopting its
    failures too.
    """
    ctx.reset(task_id, prefix)
    try:
        binder({"task_id": task_id}, state, state.ply)
    except AbortError as e:
        raise D1VoidError(
            f"{task_id}: the E3b binder refused the replayed prefix [{e.phase}]: "
            f"{e.message}. VOID: no partial-cohort analysis is produced.") from None
    except subprocess.TimeoutExpired as e:
        # The binder's replay is a T1j process too, and 12.10.3 makes a breach of
        # either frozen limit a VOID. Without this the raw TimeoutExpired escapes
        # `main`'s handlers as UNEXPECTED, exit 4 -- the binding path would have
        # been the one place a timeout was not a VOID.
        raise D1VoidError(
            f"{task_id}: the E3b prefix replay timed out after "
            f"{PER_QUERY_TIMEOUT_S}s ({e}). The run is VOID: no partial-cohort "
            f"analysis is produced.") from None


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
           repo_root: str = ".",
           deadline: Optional[Deadline] = None, budget: Optional[QueryBudget] = None,
           _compile: Optional[Callable] = None,
           _incumbent: Optional[Callable] = None) -> Dict[str, Any]:
    """Orchestrate the T1j side of D1 under both frozen limits.

    THE DEADLINE STARTS FIRST, before compilation, so the window covers every
    stage. Any VOID propagates as an exception and NOTHING is written: a report
    that exists after a breach is a partial-cohort analysis wearing a runtime
    excuse.
    """
    if not D1_EXECUTION_AUTHORIZED:
        raise D1Error(
            "D1 execution is UNAUTHORIZED. The CLI gate alone protected nothing: a "
            "direct Python caller reached this runner without passing it. Nothing has "
            "been compiled, queried or written.")
    return _run_d1_unguarded(positions=positions, paths=paths, out_path=out_path,
                             repo_root=repo_root, deadline=deadline, budget=budget,
                             _compile=_compile, _incumbent=_incumbent)


def _run_d1_unguarded(*, positions, paths, out_path, repo_root=".", deadline=None,
                      budget=None, _compile=None, _incumbent=None):
    """Everything below the gate. PRIVATE, and never a way around `run_d1`.

    It exists so the machinery can be tested WITHOUT lifting the gate in a
    fixture -- a fixture that flips an execution gate is the gate failing.
    `run_d1` is the only public entry and it checks the gate first; the Load-context
    AST test asserts both public entries read it.
    """
    deadline = deadline or Deadline()
    budget = budget or QueryBudget()
    compile_fn = _compile if _compile is not None else _default_compile
    # Constructing this reads the frozen L0 plan and nothing else; the EVALUATOR
    # is loaded on first use, which is after the registration check and after
    # compilation -- "loaded once, before the first position" (12.6).
    incumbent = _incumbent if _incumbent is not None else _Incumbent(repo_root)
    with _supervisor(deadline):
        return _run_stages(positions, paths, out_path, deadline, budget, compile_fn,
                           incumbent)


def _run_stages(positions, paths, out_path, deadline, budget, compile_fn, incumbent):
    # BEFORE the clock and before anything is compiled: an unregistered seed
    # block must cost nothing, and a refusal after compilation has already
    # spent time and written a class directory.
    _check_seed_registration()

    deadline.start()
    compile_fn(deadline)
    deadline.check("after helper compilation")

    # ONE runtime, ONE context, ONE binder for the whole run -- the E3b
    # components exactly as the qualified commands construct them.
    runtime = INT.T1jRuntime(java=paths.java, jar=paths.jar, classes=paths.classes,
                             ply_cap=paths.ply_cap, timeout_s=PER_QUERY_TIMEOUT_S)
    ctx = INT.IntegrationContext()
    binder = INT.make_binder(runtime, ctx)

    out: List[Dict[str, Any]] = []
    for pos in positions:
        _check_seed(pos.get("seed"))
        prefix = _check_prefix(pos)
        where = f"{pos.get('task_id')}@{pos.get('ply')}"
        deadline.check(f"before position {where}")
        state = _replay_prefix(prefix, where=where)
        _check_digest(state, pos, where=where)
        _bind_prefix(binder, ctx, task_id=where, state=state, prefix=prefix)
        deadline.check(f"after binding {where}")
        ours = incumbent(pos=pos, state=state, budget=budget)
        deadline.check(f"after the incumbent readout at {where}")
        per_depth = [_probe_position(moves=prefix, depth=d, paths=paths,
                                    budget=budget, deadline=deadline)
                     for d in T1J_DEPTHS]
        out.append({"task_id": pos.get("task_id"), "ply": pos.get("ply"),
                    "seed": pos["seed"], "prefix_len": len(prefix),
                    "digest": pos.get("digest"), "incumbent": ours,
                    "depths": [{"depth": r["depth"],
                                "move": list(r["record"].move) if r["record"].move else None,
                                "completed_depth": r["record"].completed_depth,
                                "invocations": r["invocations"]} for r in per_depth]})
    deadline.check("before writing the report")

    report = {"n_positions": len(out), "queries_spent": budget.spent,
              "incumbent_identity": getattr(incumbent, "identity", None),
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


def frozen_incumbent_identity(plan_path: Optional[str] = None) -> Dict[str, Any]:
    """12.6's incumbent, READ FROM THE FROZEN L0 PLAN and never retyped.

    `load_l0_plan` verifies the plan's sha256 before returning it, so the
    checkpoint name and sha1 here are the ones the canonical match actually
    played -- not a string copied into this file that could drift from it. The
    search and readout settings come from `e4_screen_reference.frozen_settings`,
    which reads them off the qualified path itself for the same reason.
    """
    from . import e4_screen_reference as REF
    from . import l0_match_plan as PLAN
    plan = PLAN.load_l0_plan(plan_path or PLAN.L0_PLAN_REL)
    pairs = {(t["reference"], t["reference_sha1"]) for t in plan["tasks"]}
    if len(pairs) != 1:
        raise D1Error(
            f"the frozen L0 plan names {len(pairs)} incumbent references {sorted(pairs)}; "
            f"D1 interrogates one incumbent and will not choose between them")
    reference, sha1 = pairs.pop()
    return {"reference": reference, "reference_sha1": sha1,
            "plan_sha256": PLAN.L0_PLAN_SHA256, **REF.frozen_settings()}


def _raw_policy(root) -> Dict[Pos, float]:
    """The raw legal-move policy the search ACTUALLY used, off the root node.

    12.6 suppresses noise at the call site with `add_noise=False`, which leaves
    `priors` and `priors_raw` the same mapping. If they differ, this root is not
    the root the frozen settings describe and the record would misdescribe it.
    """
    from .mcts import decode_move
    if not root.priors_raw:
        raise D1VoidError("the search root carries no raw policy: VOID")
    if root.priors != root.priors_raw:
        raise D1VoidError(
            "the search root's priors differ from its raw priors, so root noise "
            "reached a search 12.6 specifies with add_noise=False: VOID")
    return {decode_move(mid): float(p) for mid, p in root.priors_raw.items()}


def _policy_ranks(policy: Dict[Pos, float]) -> Dict[Pos, int]:
    """Rank by descending mass, ties by ascending (row, col).

    The SAME tie-break `eval_replay.ply_record` uses for the visit rank, so the
    two ranks are comparable. This is a policy rank and not a second definition
    of the visit rank, which stays where it already lives.
    """
    ranked = sorted(policy.items(), key=lambda kv: (-kv[1], kv[0]))
    return {move: i + 1 for i, (move, _mass) in enumerate(ranked)}


class _Incumbent:
    """5.2 capture at 12.6's exact settings. ONE evaluator for the whole run.

    The evaluator is loaded on FIRST USE and never again -- not at a cohort
    boundary, not at a cell boundary, not between the two T1j depths. Each
    position gets a fresh `SeededReferenceAgent`, which is stateless with
    respect to the evaluator: it carries that position's seed and nothing else,
    and it neither reloads, rebuilds, recompiles nor re-seeds the evaluator.

    PRIVATE, and reachable only from `_run_stages`, which `run_d1` gates. The
    structural test skips private names, so it does NOT scan this class today --
    what it does is walk public CLASSES as well as public functions, so making
    this one public would immediately require it to read the gate itself. The
    privacy is the protection here; the test is what stops the privacy from
    being dropped quietly.
    """

    def __init__(self, repo_root: str = ".", *, plan_path: Optional[str] = None,
                 _load: Optional[Callable] = None, _build: Optional[Callable] = None):
        from . import e4_screen_command as SCREEN_CMD
        from . import e4_screen_reference as REF
        self.repo_root = repo_root
        self.identity = frozen_incumbent_identity(plan_path)
        self._load = _load or SCREEN_CMD._default_load_evaluator
        self._build = _build or REF.build
        self._evaluator = None

    def _task(self, pos: Dict[str, Any], state) -> Dict[str, Any]:
        """The task shape `build_reference_agent` requires.

        D1 plays no game and has NO ANCHOR, but the qualified builder derives
        our colour as the opposite of `anchor_colour`. Setting it to the
        opposite of the side to move therefore puts our incumbent on the colour
        that is to move -- and keeps `validate_task_executable`'s seed checks on
        the path, which calling the builder directly would have skipped.
        """
        return {"seed": _check_seed(pos.get("seed")),
                "reference": self.identity["reference"],
                "reference_sha1": self.identity["reference_sha1"],
                "anchor_colour": "black" if state.to_move == "red" else "red"}

    def __call__(self, *, pos: Dict[str, Any], state, budget: QueryBudget) -> Dict[str, Any]:
        from . import e4_screen_reference as REF
        from . import eval_replay

        budget.spend(1)                      # 12.4 funds ONE readout per position
        task = self._task(pos, state)
        if self._evaluator is None:
            self._evaluator = self._load(self.repo_root)
        agent = self._build(task, evaluator=self._evaluator, capture=True)
        move = agent(state)
        cap = agent.last_capture
        if cap is None:
            raise D1VoidError(
                f"the incumbent returned {move} but captured nothing; 5.2's "
                f"observables would be absent from the record: VOID")

        record = eval_replay.ply_record(
            cap["ply"], cap["player"], cap["move"], cap["counts"], cap["root_value"],
            top2=cap["top2"], overrode_leader=cap["overrode_leader"])
        policy = _raw_policy(cap["root"])
        if set(policy) != set(cap["counts"]):
            raise D1VoidError(
                f"the raw policy covers {len(policy)} moves and the visit counts "
                f"{len(cap['counts'])}; they must be the same legal set: VOID")
        ranks = _policy_ranks(policy)
        record.update({
            "seed": task["seed"],
            "streams": REF.rng_stream_seeds(task),
            "raw_policy": {f"{r},{c}": m for (r, c), m in sorted(policy.items())},
            "root_visits": {f"{r},{c}": v for (r, c), v in sorted(cap["counts"].items())},
            "selected_policy_rank": ranks[cap["move"]],
            "selected_policy_mass": policy[cap["move"]],
        })
        return record


EXIT_OK = 0
EXIT_VOID = 3
EXIT_UNEXPECTED = 4
EXIT_UNAUTHORIZED = 5


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI. Refuses while the gate is shut, BEFORE touching anything.

    This is the SECOND of the two guard reads; `run_d1` carries the other, and
    both are required -- gating only the CLI left the public runner reachable
    directly. Nothing reads an environment variable, a flag or a config file to
    reach the gate: opening D1 is a reviewed one-line change to
    `D1_EXECUTION_AUTHORIZED` plus a separate authorization.
    """
    import argparse
    ap = argparse.ArgumentParser(description="D1 same-position interrogation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions")
    ap.add_argument("--java")
    ap.add_argument("--jar")
    ap.add_argument("--classes")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--ply-cap", type=int, default=None,
                    help="REQUIRED for a run: 5.5 [A1] forbids a silently defaulted cap")
    a = ap.parse_args(argv)

    if not D1_EXECUTION_AUTHORIZED:
        print("D1 execution is UNAUTHORIZED. No model was loaded, no JVM started, "
              "no seed drawn, no position queried, and no file was written.",
              file=sys.stderr)
        return EXIT_UNAUTHORIZED

    try:                                                      # pragma: no cover
        run_d1(positions=json.load(open(a.positions, encoding="utf-8")),
               paths=T1jPaths(java=a.java, jar=a.jar, classes=a.classes,
                              ply_cap=a.ply_cap),
               out_path=a.out, repo_root=a.repo)
    except D1VoidError as e:                                  # pragma: no cover
        print(f"VOID: {e}", file=sys.stderr)
        return EXIT_VOID
    except D1Error as e:                                      # pragma: no cover
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_UNEXPECTED
    return EXIT_OK                                            # pragma: no cover


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
