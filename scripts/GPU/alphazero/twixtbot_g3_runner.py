"""The concrete, fail-closed G3 runner. PREPARATION ONLY — not authorized to run.

Pilot card: docs/superpowers/2026-08-22-twixtbot-anchor-pilot-card.md.

WHY THIS EXISTS. `run_schedule` took an arbitrary callback, so nothing bound each
task to its models, performed the between-game Metal cleanup, wrote results
durably, or computed the two score rates. A callback that did none of it
satisfied the old signature. This module is the single command that does, and
every dependency is constructed FROM THE TASK rather than accepted from a caller.

FAIL-CLOSED AT THREE LEVELS:
  * a move fails            -> HarnessAbort inside play_game
  * a game aborts           -> the schedule STOPS, remaining tasks unplayed
  * construction fails      -> a structured aborted record, not an escaping
                               exception. Construction happens outside
                               play_game, so it needs its own guard.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Sequence

from . import twixtbot_g3_harness as H
from . import twixtbot_g3_reference as RF
from .twixtbot_g3_schedule import (
    ANCHOR_SETTINGS, CONSUMED_SEEDS, PLY_CAP, RESERVED_SEEDS, TRIALS_LADDER,
)

#: G3's pass condition. Non-saturation ONLY; ordering is descriptive.
NON_SATURATION_BAND = (0.15, 0.85)


class RunnerError(Exception):
    """A binding the runner refuses to proceed without."""


def assert_anchor_settings(kwargs: Dict, task: dict, ct) -> None:
    """Every frozen anchor setting, checked against the task. No exceptions."""
    want = {
        "model": "model/pb",
        "trials": task["trials"],
        "temperature": ANCHOR_SETTINGS["temperature"],
        "add_noise": ANCHOR_SETTINGS["add_noise"],
        "rotation": ct.ROT_OFF,
        "allow_swap": ANCHOR_SETTINGS["allow_swap"],
    }
    for k, v in want.items():
        if k not in kwargs:
            raise RunnerError(f"anchor kwargs missing {k}")
        if kwargs[k] != v:
            raise RunnerError(f"anchor {k}={kwargs[k]!r}, frozen value is {v!r}")
    extra = set(kwargs) - set(want)
    if extra:
        raise RunnerError(f"anchor kwargs carry unfrozen keys: {sorted(extra)}")


def build_task_bindings(*, task, ct, nnmplayer, evaluator_cache: Dict, repo_root: str):
    """Construct BOTH sides from the task, asserting every identity.

    Returns (player_factory, reference_agent). Raises RunnerError/ReferenceError
    on any mismatch; the caller turns that into a structured aborted record.
    """
    if task["seed"] in CONSUMED_SEEDS:
        raise RunnerError(f"task seed {task['seed']} is recorded as consumed")
    lo, hi = RESERVED_SEEDS
    if not (lo <= task["seed"] < hi):
        raise RunnerError(f"task seed {task['seed']} outside the reserved interval")
    if task["trials"] not in TRIALS_LADDER:
        raise RunnerError(f"trials {task['trials']} is not on the frozen ladder")
    if task["ply_cap"] != PLY_CAP:
        raise RunnerError(f"task ply_cap {task['ply_cap']} != {PLY_CAP}")

    kwargs = H.anchor_player_kwargs(task["trials"], ct)
    assert_anchor_settings(kwargs, task, ct)

    ref = task["reference"]
    if ref not in evaluator_cache:
        evaluator_cache[ref] = RF.load_reference_evaluator(ref, repo_root)
    evaluator = evaluator_cache[ref]

    reference_colour = "black" if task["anchor_colour"] == "red" else "red"
    agent = RF.build_reference_agent(
        task=task, evaluator=evaluator, colour=reference_colour
    )
    return (lambda: nnmplayer.Player(**kwargs)), agent


def _append_durably(path: Optional[str], record: dict) -> None:
    """Write one result and fsync it. A crash must not lose finished games."""
    if path is None:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_g3(
    *,
    tasks: Sequence[dict],
    twixt,
    Point,
    ct,
    TwixtState,
    nnmplayer,
    repo_root: str,
    results_path: Optional[str] = None,
    play_game: Callable = H.play_game,
    cleanup: Callable[[], None] = RF.between_games_cleanup,
) -> dict:
    """Run the schedule. Stops at the FIRST aborted task. Not authorized to run."""
    evaluator_cache: Dict = {}
    results: List[dict] = []

    for task in tasks:
        try:
            player_factory, agent = build_task_bindings(
                task=task, ct=ct, nnmplayer=nnmplayer,
                evaluator_cache=evaluator_cache, repo_root=repo_root,
            )
        except BaseException as e:                              # noqa: BLE001
            # Construction sits OUTSIDE play_game, so without this a binding
            # failure would escape with no record at all.
            rec = {
                "task_index": task.get("task_index"), "seed": task.get("seed"),
                "trials": task.get("trials"), "reference": task.get("reference"),
                "opening_id": task.get("opening_id"),
                "colour_arm": task.get("colour_arm"),
                "aborted": True, "abort_reason": "engine_exception",
                "abort_detail": f"binding: {type(e).__name__}: {e}",
                "result": None, "winner": None, "plies": 0, "moves": [],
            }
            results.append(rec)
            _append_durably(results_path, rec)
            return _stopped(results, task, rec, tasks)

        rec = play_game(
            task=task, twixt=twixt, Point=Point, ct=ct, TwixtState=TwixtState,
            player_factory=player_factory, reference_agent=agent, ply_cap=PLY_CAP,
        )
        results.append(rec)
        _append_durably(results_path, rec)
        cleanup()                                # Metal buffers, every game

        if rec.get("aborted"):
            return _stopped(results, task, rec, tasks)

    return {"completed": True, "stopped_at_task_index": None, "stopped_reason": None,
            "n_played": len(results), "n_remaining": 0, "results": results,
            "summary": summarise(results)}


def _stopped(results, task, rec, tasks) -> dict:
    return {
        "completed": False,
        "stopped_at_task_index": task.get("task_index"),
        "stopped_reason": rec.get("abort_reason"),
        "stopped_detail": rec.get("abort_detail"),
        "n_played": len(results),
        "n_remaining": len(tasks) - len(results),
        "results": results,
        # No summary on an incomplete run: scoring a filtered sample would read
        # as a result.
        "summary": None,
    }


def summarise(results: Sequence[dict]) -> dict:
    """The two score rates per trials setting, and the lowest passing setting.

    Score is the ANCHOR's, from the anchor's perspective: win 1, draw 0.5,
    loss 0. Pass = non-saturation against BOTH references. Ordering is recorded
    DESCRIPTIVELY and is not part of the pass condition.
    """
    if any(r.get("aborted") for r in results):
        raise RunnerError("refusing to summarise a run containing an aborted game")

    per: Dict[int, Dict[str, dict]] = {}
    for r in results:
        cell = per.setdefault(r["trials"], {}).setdefault(
            r["reference"], {"n": 0, "score": 0.0})
        cell["n"] += 1
        cell["score"] += {"anchor": 1.0, "reference": 0.0}.get(r["result"], 0.5)

    lo, hi = NON_SATURATION_BAND
    out: Dict[int, dict] = {}
    for trials, refs in per.items():
        rates = {ref: (c["score"] / c["n"] if c["n"] else None) for ref, c in refs.items()}
        unsat = all(v is not None and lo <= v <= hi for v in rates.values())
        out[trials] = {
            "score_rates": rates,
            "games": {ref: c["n"] for ref, c in refs.items()},
            "non_saturated": unsat,
            # descriptive only, never a gate
            "ordering_note": _ordering(rates),
        }
    passing = sorted(t for t, v in out.items() if v["non_saturated"])
    return {
        "band": list(NON_SATURATION_BAND),
        "per_trials": out,
        "passing_trials": passing,
        "selected_trials": passing[0] if passing else None,
        "pass_condition": "non-saturation against BOTH references; ordering is not a gate",
    }


def _ordering(rates: Dict[str, Optional[float]]) -> str:
    a, b = rates.get("0379"), rates.get("calib020_0001")
    if a is None or b is None:
        return "incomplete"
    if a == b:
        return "flat (descriptive only)"
    return ("anchor scores higher vs 0379 than vs calib020_0001"
            if a > b else
            "anchor scores higher vs calib020_0001 than vs 0379") + " (descriptive only)"
