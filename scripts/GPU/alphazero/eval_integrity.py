"""Zero-tolerance integrity checks for readout matches.

Design spec section 8.3 freezes these conditions as immediate stops. Every
check FAILS CLOSED: an unverifiable condition raises rather than warning and
continuing. A run that trips any of these is not a result.

Scope note: illegal moves and crashes are NOT raised here -- they already
abort through `TwixtState.apply_move` and the worker `_WorkerFailed` path.
`IntegrityError` covers budget mismatch, corrupt required telemetry,
binding/configuration faults, `unknown_error`, and incomplete or duplicate
result sets. Seed-interval reuse raises `ValueError`, because it is a
configuration error caught before anything runs, not a run-time integrity
fault.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Dict


class IntegrityError(RuntimeError):
    """A frozen zero-tolerance condition was observed."""


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def validate_ply(ply: int, expected_sims: int, root_visit_count: int,
                 root_value, top2) -> None:
    """Per-ply guard. Raises on budget mismatch or corrupt required telemetry.

    A None mean on a ZERO-visit child is undefined, not corrupt, and passes.
    A None or non-finite mean on a VISITED child is corrupt and raises.
    """
    if root_visit_count != expected_sims:
        raise IntegrityError(
            f"ply {ply}: simulation budget mismatch -- root completed "
            f"{root_visit_count} visits, expected exactly {expected_sims}")
    if not _finite(root_value):
        raise IntegrityError(f"ply {ply}: root_value is not finite ({root_value!r})")
    for stat in top2 or []:
        if stat.visits <= 0:
            continue
        for label, q in (("q_value_child_perspective", stat.q_child),
                         ("q_value_root_perspective", stat.q_root)):
            if not _finite(q):
                raise IntegrityError(
                    f"ply {ply}: child {stat.move} has {stat.visits} visits but "
                    f"{label} is {q!r}; a visited child's mean is defined")


def validate_seed_intervals(current, priors) -> None:
    """Refuse a seed interval that overlaps any already-consumed interval.

    Intervals are HALF-OPEN [start, end): [0, 64) and [64, 128) are adjacent,
    not overlapping. Reusing seeds would silently correlate a "fresh" run with
    an earlier one, which is the kind of contamination that is invisible in
    the result and fatal to it.

    EVERY interval is validated, and the WHOLE set is checked pairwise -- not
    just current-versus-each-prior. A reversed prior, or two priors that
    overlap each other, means the recorded history is wrong, and a history
    that cannot be trusted cannot establish that this run is fresh.
    """
    labelled = [("current", list(current))]
    labelled += [(f"prior[{i}]", list(p)) for i, p in enumerate(priors)]

    for label, (start, end) in labelled:
        if end <= start:
            raise ValueError(
                f"{label} seed interval [{start}, {end}) is empty or reversed")

    for i in range(len(labelled)):
        label_a, (start_a, end_a) = labelled[i]
        for j in range(i + 1, len(labelled)):
            label_b, (start_b, end_b) = labelled[j]
            if start_a < end_b and start_b < end_a:
                raise ValueError(
                    f"{label_a} [{start_a}, {end_a}) overlaps {label_b} "
                    f"[{start_b}, {end_b}); seeds may not be reused")


def validate_game_binding(result, task) -> None:
    """Per-game guard, applied AS EACH GAME FINISHES so a fault stops the run
    immediately rather than after every game has been played (spec 8.3).

    The decisive check is that each colour's recorded readout equals the
    TASK's expected config. Checking only that an agent is self-consistent
    across games would pass a SYSTEMATIC leak, where every row carries the
    same wrong configuration.

    LEGACY tasks carry no AgentSpec and are returned untouched, including
    their termination reason: eval_checkpoint_match's behaviour must not
    change, and this guard is shared through _play_and_build_result.
    """
    if task.red_agent is None or task.black_agent is None:
        return

    if result.reason == "unknown_error":
        raise IntegrityError(f"task {result.task_id}: game ended in unknown_error")

    for colour, agent_id, readout, spec in (
            ("red", result.red_agent_id, result.red_readout, task.red_agent),
            ("black", result.black_agent_id, result.black_readout,
             task.black_agent)):
        if agent_id != spec.agent_id:
            raise IntegrityError(
                f"task {result.task_id}: {colour} agent id {agent_id!r} does "
                f"not match the task binding {spec.agent_id!r}")
        expected = asdict(spec.readout)
        if readout != expected:
            raise IntegrityError(
                f"task {result.task_id}: configuration mismatch for {colour} "
                f"agent {agent_id!r} -- recorded {readout!r}, task specifies "
                f"{expected!r}")

    if result.red_agent_id == result.black_agent_id:
        raise IntegrityError(
            f"task {result.task_id}: agent {result.red_agent_id!r} holds both "
            f"colours")


def validate_result_set(results, tasks, agent_a_id: str,
                        agent_b_id: str) -> None:
    """Whole-run guard, applied before any statistic is computed.

    Re-runs the per-game binding check, because a result set may reach here
    from a path that did not call validate_game_binding, and then adds the
    checks that only make sense over the whole run.
    """
    expected_ids = {agent_a_id, agent_b_id}

    # STRUCTURAL checks first. A duplicate or missing task_id makes the
    # result->task lookup below meaningless, so reporting a binding mismatch
    # ahead of it would name the wrong defect.
    bad = [r.task_id for r in results if r.reason == "unknown_error"]
    if bad:
        raise IntegrityError(
            f"{len(bad)} game(s) ended in unknown_error: {sorted(bad)[:10]}")

    if len(results) != len(tasks):
        raise IntegrityError(
            f"incomplete run: {len(results)} results for {len(tasks)} tasks")

    seen: Dict[int, int] = {}
    for r in results:
        seen[r.task_id] = seen.get(r.task_id, 0) + 1
    dupes = sorted(t for t, n in seen.items() if n > 1)
    if dupes:
        raise IntegrityError(f"duplicate task_ids in results: {dupes[:10]}")
    missing = sorted({t.task_id for t in tasks} - set(seen))
    if missing:
        raise IntegrityError(f"incomplete run: missing task_ids {missing[:10]}")

    # Only now is the lookup one-to-one and the binding check meaningful.
    by_task_all = {t.task_id: t for t in tasks}
    for r in results:
        validate_game_binding(r, by_task_all[r.task_id])

    red_counts: Dict[str, int] = {agent_a_id: 0, agent_b_id: 0}
    for r in results:
        got = {r.red_agent_id, r.black_agent_id}
        if None in got or not got <= expected_ids:
            raise IntegrityError(
                f"task {r.task_id}: unexpected agent ids "
                f"{sorted(map(str, got))}, expected {sorted(expected_ids)}")
        red_counts[r.red_agent_id] += 1

    if red_counts[agent_a_id] != red_counts[agent_b_id]:
        raise IntegrityError(
            f"colour balance broken: {agent_a_id} was red "
            f"{red_counts[agent_a_id]} times, {agent_b_id} "
            f"{red_counts[agent_b_id]}")
