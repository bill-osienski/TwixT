"""H5 STUDY RUNNER. GATED; NOT AUTHORIZED.

Frozen by `docs/superpowers/2026-09-27-t1j-h5-implementation-card.md` §2, under
the H5 replication card (`2026-09-25-t1j-h5-replication-card.md`, with §12).
🔴 THE CARDS ARE THE AUTHORITY: where this module and a card disagree, the card
wins and the disagreement is an amendment, not a code fix.

🔴 H5 EDITS NO H4 FILE (card §0). It plays through H4's runner by IMPORT: every
game-level function -- `play_game`, `check_game`, the agent and binder wrappers,
the classification, the reader -- is H4's own object, called unchanged. Only two
H4 functions read H4 module globals H5 must replace, and they are COPIED here
VERBATIM: `header_candidate` and `_run_unguarded`. Their text is H4's; the
substitution happens through THIS module's namespace (`DESIGNS`,
`bound_identity`, `header_candidate`), and a test requires their ASTs to equal
H4's and every other free name to resolve to H4's very object.

🔴 STUDY MODE (card §2.3): refuses while `SEED_BLOCKS` is empty; accepts NO
injected seam, not even paths; reads only a COMMITTED manifest; segment k > 0
runs only after segment k - 1 is committed and complete. FIXTURE MODE: negative
synthetic seeds, a stub incumbent, header `H5_SYNTHETIC_FIXTURE`, `evidence:
false`. The gate lives in `h5_study_authorization.py`, OUTSIDE this hashed file.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from . import e4_screen_integration as INT
from . import h4_pilot_feasibility as F
from . import h4_runner as R4
from .d1_probe import D1Error, D1VoidError, Deadline, T1jPaths, _supervisor
from .d1_probe import _default_compile as _compile_helper_verified
from .h4_runner import (EMPTY_BOARD, PLY_CAP, QUERY_TIMEOUT_S,
                        REPLAY_TIMEOUT_S, SEGMENT_DEADLINE_S, H4RunError, H4RunVoid,
                        Results, _Stash, _canon, _stdout_of, _trace, _verified_t1j_paths,
                        binding_differences, check_destinations, check_incumbent_identity,
                        check_schedule, claim_directory, classify, completed_counts,
                        frozen_argmax_config, frozen_incumbent_identity,
                        load_production_evaluator, make_schedule, play_game,
                        production_incumbent_build, schedule_digest, seed_registry_faults,
                        split_toolchain)

#: THE H5 STUDY IS NOT AUTHORIZED. The gate is `AUTH.H5_STUDY_EXECUTION_AUTHORIZED`,
#: in its own unhashed module, read AS AN ATTRIBUTE AT CALL TIME at BOTH public
#: entries -- `run_study` and `main` -- in EVERY mode, fixture included. No
#: override exists: not argv, not the environment, not a file, not an import hook.
from . import h5_study_authorization as AUTH

#: Card §2.1: the mode decides the design label, the seed sign and the evidence
#: flag. There is NO pilot mode (replication card §7).
DESIGNS = {"fixture": "H5_SYNTHETIC_FIXTURE", "study": "H5_STUDY"}
FIXTURE_DESIGN = DESIGNS["fixture"]
STUDY_DESIGN = DESIGNS["study"]

REPO_ROOT = R4.REPO_ROOT
#: Card §2.1: SEVEN bound cards -- H5's two and H4's five, incorporated by reference.
CARDS = ("docs/superpowers/2026-09-25-t1j-h5-replication-card.md",
         "docs/superpowers/2026-09-27-t1j-h5-implementation-card.md",
         *R4.CARDS)
#: 🔴 Card §2.4: EVERY first-party file reachable by import at any depth from this
#: runner and `h5_production_qualification.py`, package `__init__`s included, minus
#: the H5 gate modules. H4 gate modules on the path ARE listed: opening an H4 gate
#: during H5 changes H5's bound `code`. A test recomputes the closure and requires
#: equality with this tuple.
CODE = (
    "scripts/GPU/__init__.py",
    "scripts/GPU/alphazero/__init__.py",
    "scripts/GPU/alphazero/d0_postmortem.py",
    "scripts/GPU/alphazero/d1_probe.py",
    "scripts/GPU/alphazero/d1_selection.py",
    "scripts/GPU/alphazero/e4_screen_command.py",
    "scripts/GPU/alphazero/e4_screen_integration.py",
    "scripts/GPU/alphazero/e4_screen_reference.py",
    "scripts/GPU/alphazero/e4_screen_rules.py",
    "scripts/GPU/alphazero/e4_screen_runner.py",
    "scripts/GPU/alphazero/eval_integrity.py",
    "scripts/GPU/alphazero/eval_readout.py",
    "scripts/GPU/alphazero/eval_replay.py",
    "scripts/GPU/alphazero/eval_runner.py",
    "scripts/GPU/alphazero/evaluator.py",
    "scripts/GPU/alphazero/fpu_state_hash.py",
    "scripts/GPU/alphazero/game/__init__.py",
    "scripts/GPU/alphazero/game/twixt_state.py",
    "scripts/GPU/alphazero/h2_match_plan.py",
    "scripts/GPU/alphazero/h2_match_rules.py",
    "scripts/GPU/alphazero/h2_match_runner.py",
    "scripts/GPU/alphazero/h4_4a_characterization.py",
    "scripts/GPU/alphazero/h4_pilot_authorization.py",
    "scripts/GPU/alphazero/h4_pilot_feasibility.py",
    "scripts/GPU/alphazero/h4_production_qualification.py",
    "scripts/GPU/alphazero/h4_production_qualification_authorization.py",
    "scripts/GPU/alphazero/h4_repair_qualification.py",
    "scripts/GPU/alphazero/h4_runner.py",
    "scripts/GPU/alphazero/h5_production_qualification.py",
    "scripts/GPU/alphazero/h5_runner.py",
    "scripts/GPU/alphazero/l0_match_plan.py",
    "scripts/GPU/alphazero/l0_match_rules.py",
    "scripts/GPU/alphazero/local_evaluator.py",
    "scripts/GPU/alphazero/mcts.py",
    "scripts/GPU/alphazero/network.py",
    "scripts/GPU/alphazero/opening_diagnostics.py",
    "scripts/GPU/alphazero/probe_eval.py",
    "scripts/GPU/alphazero/t1j_adapter.py",
    "scripts/GPU/alphazero/t1j_toolchain.py",
    "scripts/GPU/alphazero/twixtbot_g3_reference.py",
    "scripts/GPU/alphazero/twixtbot_g3_schedule.py",
    "scripts/GPU/alphazero/void_trace.py",
)
#: Card §1: the only files on the path left out of `code`.
GATE_MODULES = ("scripts/GPU/alphazero/h5_study_authorization.py",
                "scripts/GPU/alphazero/h5_production_qualification_authorization.py")

#: 🔴 Card §2.1, §7, §11-§12: the four step-5 survey blocks, registered at step 6
#: after the frozen collision proof's `pre` run was CLEAN
#: (evidence/t1j-h5-seed-registration/02_*), each ACCOUNTED in e4_screen_reference.
#: ONE line, in exactly this annotated form: the proof's A3 shape rule parses it.
#: Study mode refuses while it is empty.
SEED_BLOCKS: Dict[int, tuple] = {0: (202_650_000, 202_650_148), 1: (202_652_000, 202_652_148), 2: (202_654_000, 202_654_148), 3: (202_656_000, 202_656_148)}
SEGMENTS = (0, 1, 2, 3)
PAIRS_PER_SEGMENT = 74
#: Replication card §8: FIXED names, repo-relative, create-only; the one-shot use
#: record of each segment until H5 closes.
EVIDENCE_DIRS = {k: f"docs/superpowers/evidence/t1j-h5-study-segment{k}" for k in SEGMENTS}
PAIR_TAG = "h5s"

EXIT_OK = R4.EXIT_OK
EXIT_VOID = R4.EXIT_VOID
EXIT_REFUSED = R4.EXIT_REFUSED
EXIT_UNAUTHORIZED = R4.EXIT_UNAUTHORIZED


# ───────────────────────── the header, bound to H5's cards and code ─────────────────────────

def bound_identity(incumbent_identity: Mapping[str, Any],
                   toolchain_content: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """H4's `bound_identity`, CALLED UNCHANGED -- so `t1j_runtime`, key
    `h4_acceptance` included, is built by H4's own code -- with `cards` and `code`
    replaced by H5's, hashed from the files NOW."""
    return {**R4.bound_identity(incumbent_identity, toolchain_content),
            "cards": {c: R4._sha256_or_none(REPO_ROOT / c) for c in CARDS},
            "code": {m: R4._sha256_or_none(REPO_ROOT / m) for m in CODE}}


def header_candidate(*, mode: str, segment: int, schedule: Sequence[Mapping[str, Any]],
                     incumbent_identity: Mapping[str, Any],
                     toolchain_content: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return {"record_type": "header", "design": DESIGNS[mode], "evidence": mode != "fixture",
            "segment": segment, "schedule_digest": schedule_digest(schedule),
            "seeds": [t["seed"] for t in schedule],
            **bound_identity(incumbent_identity, toolchain_content)}


# ─────────────────────── the canonical schedule and the manifest ───────────────────────

def canonical_schedule(segment: int, reference: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Replication card §2: pair k of the block gets lo+2k (Arm A), lo+2k+1 (Arm B)."""
    if segment not in SEED_BLOCKS:
        raise H4RunError(f"no registered H5 block for segment {segment}")
    lo, hi = SEED_BLOCKS[segment]
    pairs = [(f"{PAIR_TAG}{segment}-{k:02d}", lo + 2 * k, lo + 2 * k + 1)
             for k in range((hi - lo) // 2)]
    return make_schedule(pairs, mode="study", reference=reference)


def manifest_entry(manifest: str, *, segment: int) -> Dict[str, Any]:
    try:
        with open(manifest, encoding="utf-8") as fh:
            m = json.load(fh)
    except (OSError, ValueError) as e:
        raise H4RunError(f"manifest {manifest} is unreadable: {e}") from None
    if not isinstance(m, dict) or m.get("stage") != STUDY_DESIGN:
        raise H4RunError(f"{manifest} is not a {STUDY_DESIGN} manifest")
    entries = m.get("segments")
    if not isinstance(entries, list) or not 0 <= segment < len(entries) \
            or entries[segment].get("segment") != segment:
        raise H4RunError(f"{manifest} has no entry for segment {segment}")
    return entries[segment]


def _durable(paths: Sequence[str], what: str) -> None:
    """The H4 analysis card §2.1 rule, by import: tracked at HEAD in THIS
    repository, unmodified."""
    try:
        F.check_durable(list(paths), _fixture=False)
    except F.H4AnalysisRefused as e:
        raise H4RunError(f"{what}: {e}") from None


#: Implementation card §10 A1: the ONLY fields the predecessor check keeps, per
#: record type -- identities and structure, never an outcome. Disjoint from
#: `h4_runner.BLIND_FIELDS` (a test holds it to that).
PREDECESSOR_VIEW = {
    "header": ("design", "evidence", "segment", "schedule_digest", "seeds",
               "incumbent_identity", "t1j_runtime", "cards", "code"),
    "game_start": ("task_id", "pair_id", "arm", "seed", "game_index"),
    "ply": ("task_id",), "process": ("task_id",), "game_result": ("task_id",),
    "segment_end": ("complete", "games_completed"), "run_void": ()}
START_KEYS = ("task_id", "pair_id", "arm", "seed", "game_index")


def predecessor_views(path: str) -> List[Dict[str, Any]]:
    """Each record reduced to its whitelisted view AS SOON AS IT IS PARSED. The
    blind is PROCEDURAL: parsing reads the bytes that carry outcomes; nothing
    here inspects, computes from or outputs an outcome value."""
    views = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            kind = r.get("record_type") if isinstance(r, dict) else None
            keep = PREDECESSOR_VIEW.get(kind)
            views.append({"record_type": kind} if keep is None else
                         {"record_type": kind, **{k: r.get(k) for k in keep}})
    return views


def structural_faults(views: Sequence[Mapping[str, Any]], entry: Mapping[str, Any]) -> List[str]:
    """Card §10 A1: the file against ITS manifest entry -- header, then for each
    scheduled task in order one game_start, its own ply/process records, one
    game_result, then one complete segment_end. Returns the first fault."""
    schedule = entry["schedule"]
    if not views or views[0]["record_type"] != "header":
        return ["the first record is not a header"]
    head = views[0]
    if (head["design"], head["evidence"], head["segment"]) != \
            (STUDY_DESIGN, True, entry["segment"]):
        return [f"the header is {head['design']!r}/evidence={head['evidence']!r}/segment "
                f"{head['segment']!r}, not {STUDY_DESIGN}/True/{entry['segment']}"]
    unbound = [f for f in R4.MANIFEST_BOUND if _canon(head[f]) != _canon(entry.get(f))]
    if unbound:
        return [f"the header's {unbound} differ from manifest entry {entry['segment']}"]
    i = 1
    for n, task in enumerate(schedule):
        want = {**{k: task[k] for k in START_KEYS[:-1]}, "game_index": n}
        if i >= len(views) or views[i]["record_type"] != "game_start":
            return [f"task {n} ({task['task_id']}): no game_start where one is due"]
        got = {k: views[i][k] for k in START_KEYS}
        if _canon(got) != _canon(want):
            return [f"task {n}: game_start {got} is not the schedule's {want}"]
        i += 1
        while i < len(views) and views[i]["record_type"] in ("ply", "process"):
            if views[i]["task_id"] != task["task_id"]:
                return [f"task {n}: a {views[i]['record_type']} record of "
                        f"{views[i]['task_id']!r} inside {task['task_id']!r}"]
            i += 1
        if i >= len(views) or views[i]["record_type"] != "game_result" \
                or views[i]["task_id"] != task["task_id"]:
            got = (f"{views[i]['record_type']} of {views[i].get('task_id')!r}"
                   if i < len(views) else "end of file")
            return [f"task {n} ({task['task_id']}): {got} where its own game_result is due"]
        i += 1
    rest = views[i:]
    if [v["record_type"] for v in rest] != ["segment_end"]:
        return [f"after the last game: {[v['record_type'] for v in rest]}, not exactly "
                f"one segment_end"]
    end = rest[0]
    if end["complete"] is not True or end["games_completed"] != len(schedule):
        return [f"segment_end complete={end['complete']!r} games_completed="
                f"{end['games_completed']!r}, not True/{len(schedule)}"]
    return []


def check_predecessor(manifest: str, segment: int) -> None:
    """Card §10 A1: segment k > 0 runs only after segment k - 1's results are
    COMMITTED, UNMODIFIED and STRUCTURALLY COMPLETE against manifest entry k - 1."""
    if segment == 0:
        return
    prev = os.path.join(REPO_ROOT, EVIDENCE_DIRS[segment - 1], "results.jsonl")
    _durable([prev], f"segment {segment} needs segment {segment - 1} committed first")
    entry = manifest_entry(manifest, segment=segment - 1)
    try:
        faults = structural_faults(predecessor_views(prev), entry)
    except (OSError, ValueError, KeyError, TypeError) as e:
        faults = [f"unreadable: {type(e).__name__}"]
    if faults:
        raise H4RunError(f"segment {segment} needs segment {segment - 1} STRUCTURALLY "
                         f"COMPLETE: {faults[0]}")


# ──────────────────────────────── the public entries ────────────────────────────────

def run_study(*, mode: str, out_dir: str, deadline_s: float,
              paths: Optional[T1jPaths] = None, classes: Optional[str] = None,
              schedule: Optional[Sequence[Mapping[str, Any]]] = None,
              manifest: Optional[str] = None,
              incumbent_build: Optional[Callable] = None, evaluator: Any = None,
              incumbent_identity: Optional[Mapping[str, Any]] = None, segment: int = 0,
              _compile: Optional[Callable] = None,
              _now: Callable[[], float] = time.monotonic) -> Dict[str, Any]:
    """PUBLIC ENTRY. Refuses while the gate is shut, in EVERY mode."""
    if not AUTH.H5_STUDY_EXECUTION_AUTHORIZED:
        raise H4RunError("the H5 study is UNAUTHORIZED. Nothing has been built, "
                         "compiled, played or written.")
    if mode not in DESIGNS:
        raise H4RunError(f"unknown mode {mode!r}; one of {sorted(DESIGNS)}")
    if mode != "fixture":
        return _production(out_dir=out_dir, classes=classes, deadline_s=deadline_s,
                           manifest=manifest, segment=segment, _now=_now,
                           injected={"paths": paths, "schedule": schedule,
                                     "incumbent_build": incumbent_build,
                                     "evaluator": evaluator,
                                     "incumbent_identity": incumbent_identity,
                                     "_compile": _compile})
    if manifest is not None or classes is not None or paths is None:
        raise H4RunError("fixture mode takes stub paths and binds to no manifest")
    check_schedule(schedule, mode=mode)
    check_destinations(out_dir, paths.classes, mode=mode)
    if incumbent_build is None or incumbent_identity is None:
        raise H4RunError("fixture mode needs the STUB incumbent and its identity")
    if _compile is None:
        raise H4RunError("fixture mode never compiles for real; pass the stub compile")
    return _run_unguarded(mode=mode, schedule=schedule, out_dir=out_dir, paths=paths,
                          deadline_s=deadline_s, incumbent_build=incumbent_build,
                          evaluator=evaluator, incumbent_identity=incumbent_identity,
                          segment=segment, _compile=_compile, _now=_now)


def _production(*, out_dir, classes, deadline_s, manifest, segment, _now, injected):
    """Card §2.3: study mode. Every refusal before the claim writes NOTHING; every
    bound field that needs no compile is checked before the claim, the toolchain
    resolved or the checkpoint loaded."""
    if not SEED_BLOCKS:
        raise H4RunError("no H5 seed block is registered: SEED_BLOCKS is empty until "
                         "step 6, and study mode refuses while it is")
    seams = sorted(k for k, v in injected.items() if v is not None)
    if seams:
        raise H4RunError(f"study mode accepts no injected seam, got {seams}: an entry "
                         f"that accepts its own seams is not a gate")
    if manifest is None or classes is None:
        raise H4RunError("study mode plays only a committed manifest's schedule, into "
                         "a named class directory")
    if segment not in SEGMENTS:
        raise H4RunError(f"no H5 segment {segment!r}; one of {list(SEGMENTS)}")
    _durable([manifest], "the manifest is not durable")
    entry = manifest_entry(manifest, segment=segment)
    check_predecessor(manifest, segment)
    schedule = entry.get("schedule")
    check_schedule(schedule, mode="study")
    config = frozen_argmax_config()
    identity = frozen_incumbent_identity(config, design=STUDY_DESIGN)
    check_incumbent_identity(identity, config, design=STUDY_DESIGN)
    wrong_ref = [t["task_id"] for t in schedule
                 if (t.get("reference"), t.get("reference_sha1"))
                 != (identity["reference"], identity["reference_sha1"])]
    if wrong_ref:
        raise H4RunError(f"tasks {wrong_ref[:3]} name another incumbent reference")
    if _canon(list(schedule)) != _canon(canonical_schedule(segment, identity)):
        raise H4RunError(f"the manifest's schedule is not the CANONICAL H5 segment "
                         f"{segment} schedule (block, pair ids, arm seeds, reference)")
    faults = seed_registry_faults(schedule)
    if faults:
        raise H4RunError(f"seeds fail the shared registry: {faults[:3]}")
    want_dir = EVIDENCE_DIRS[segment]
    if entry.get("evidence_dir") != want_dir or os.path.realpath(out_dir) != \
            os.path.realpath(os.path.join(REPO_ROOT, want_dir)):
        raise H4RunError(f"H5 segment {segment} writes ONLY to its manifest's "
                         f"evidence_dir {want_dir!r}, not {out_dir!r}")
    if os.path.lexists(out_dir):
        raise H4RunError(f"{out_dir} is occupied: this segment has already been run, "
                         f"or a VOID left it -- a run is one-shot")
    head = header_candidate(mode="study", segment=segment, schedule=schedule,
                            incumbent_identity=identity, toolchain_content=None)
    bad = binding_differences(head, entry, stage=STUDY_DESIGN, with_toolchain=False)
    if bad:
        raise H4RunError(f"the run would not bind to its manifest: {bad} differ. Nothing "
                         f"was claimed, loaded, compiled or played.")
    paths = _verified_t1j_paths(classes)
    check_destinations(out_dir, paths.classes, mode="study")
    evaluator = load_production_evaluator(identity)
    return _run_unguarded(mode="study", schedule=schedule, out_dir=out_dir, paths=paths,
                          deadline_s=deadline_s,
                          incumbent_build=production_incumbent_build(config),
                          evaluator=evaluator, incumbent_identity=identity,
                          segment=segment, _compile=None, _now=_now, entry=entry)


# ──────── the H4 run loop -- COPIED VERBATIM from `h4_runner._run_unguarded` (card §2.3) ────────

def _run_unguarded(*, mode, schedule, out_dir, paths, deadline_s, incumbent_build,
                   evaluator, incumbent_identity, segment, _compile, _now, entry=None):
    t_start = _now()
    claim_directory(out_dir)                      # THE claim: before any draw or JVM
    results = Results(os.path.join(out_dir, "results.jsonl"))
    trace = os.fdopen(os.open(os.path.join(out_dir, "trace.jsonl"),
                              os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), "w")
    written: Dict[str, List[Dict[str, Any]]] = {"ply": [], "process": [], "game_result": []}
    stash = _Stash()
    where: Dict[str, Any] = {"stage": "setup", "task": None}
    deadline = Deadline(limit_s=deadline_s)
    try:
        ctx = INT.IntegrationContext()
        runtime = INT.T1jRuntime(java=paths.java, jar=paths.jar, classes=paths.classes,
                                 ply_cap=PLY_CAP, timeout_s=REPLAY_TIMEOUT_S,
                                 h4_acceptance=True)
        binder = INT.make_binder(runtime, ctx)
        base_factory = INT.make_agent_factory(runtime=runtime, ctx=ctx, evaluator=evaluator,
                                              reference_build=incumbent_build,
                                              t1j_timeout_s=QUERY_TIMEOUT_S)
        state_factory = INT.make_state_factory({EMPTY_BOARD: []}, ctx)
        deadline.start()
        with _supervisor(deadline):
            where["stage"] = "compile"
            compile_fn = _compile or (lambda d: _compile_helper_verified(d, paths=paths))
            content, local = split_toolchain(compile_fn(deadline))
            header = header_candidate(mode=mode, segment=segment, schedule=schedule,
                                      incumbent_identity=incumbent_identity,
                                      toolchain_content=content)
            if entry is not None:
                bad = binding_differences(header, entry, stage=DESIGNS[mode],
                                          with_toolchain=True)
                if bad:
                    raise H4RunVoid("unexpected", f"after compiling, the header does not "
                                    f"bind to its manifest: {bad} differ. No game was "
                                    f"played (card §12.6.3)")
            results.emit({**header, "t1j_local": local})
            setup_s = _now() - t_start
            _trace(trace, {"event": "run_start", "n_games": len(schedule),
                           "design": DESIGNS[mode]})
            for i, task in enumerate(schedule):
                where.update(stage="game", task=task)
                deadline.check(f"before game {i}")
                _trace(trace, {"event": "game_start", "index": i,
                               "task_id": task["task_id"]})
                res = play_game(task={**task, "game_index": i}, ctx=ctx, binder=binder,
                                base_factory=base_factory, state_factory=state_factory,
                                results=results, written=written, stash=stash, now=_now)
                _trace(trace, {"event": "game_done", "index": i,
                               "terminal_reason": res["terminal_reason"],
                               "plies": res["plies"], "elapsed_s": res["elapsed_s"]})
            where.update(stage="post_game_check", task=None)
            end = {"record_type": "segment_end", "games_completed": len(schedule),
                   "complete": True, "setup_s": setup_s, "total_s": _now() - t_start}
            results.emit(end)
            _trace(trace, {"event": "run_end", "games_completed": len(schedule)})
            return {"games_completed": len(schedule), "results": results.path,
                    "complete": True}
    except BaseException as raised:              # noqa: BLE001 -- recorded, re-raised
        original = stash.exc or raised
        if isinstance(raised, H4RunVoid):
            original = raised
        stage = where["stage"]
        if isinstance(original, D1VoidError):
            stage = "deadline"
        task = where["task"] or {}
        void = {"record_type": "run_void", "stage": stage,
                "task_id": task.get("task_id"), "pair_id": task.get("pair_id"),
                "arm": task.get("arm"),
                "ply": (written["ply"][-1]["ply"] if task and written["ply"]
                        and written["ply"][-1]["task_id"] == task.get("task_id") else None),
                "classification": classify(original),
                "exception": type(original).__name__, "reason": str(original),
                "stdout": _stdout_of(original),
                "completed": completed_counts(results.path),
                "elapsed_s": _now() - t_start}
        try:
            results.emit(void)
            _trace(trace, {"event": "run_void", "classification": void["classification"]})
        except Exception:                          # noqa: BLE001 -- never mask the cause
            pass
        if isinstance(raised, (KeyboardInterrupt, SystemExit)):
            raise
        raise H4RunVoid(void["classification"], str(original)) from original
    finally:
        results.close()
        trace.close()


# ─────────────────────────────────── the CLI ───────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI. Refuses while the gate is shut, BEFORE touching anything."""
    import argparse
    ap = argparse.ArgumentParser(prog="h5_runner",
                                 description="H5 study runner. IT IS NOT AUTHORIZED.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--segment", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--classes", required=True)
    a = ap.parse_args(argv)
    if not AUTH.H5_STUDY_EXECUTION_AUTHORIZED:
        print("the H5 study is UNAUTHORIZED. No JVM was started, nothing was played, "
              "and no file was written.", file=sys.stderr)
        return EXIT_UNAUTHORIZED
    try:
        out = run_study(mode="study", manifest=a.manifest, segment=a.segment,
                        out_dir=a.out_dir, classes=a.classes,
                        deadline_s=SEGMENT_DEADLINE_S)
    except H4RunVoid as e:
        print(f"VOID: {e}", file=sys.stderr)
        return EXIT_VOID
    except (H4RunError, D1Error) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"complete: {out['games_completed']} games -> {out['results']}")
    return EXIT_OK


if __name__ == "__main__":                                                        # pragma: no cover
    raise SystemExit(main())
