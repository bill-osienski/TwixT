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


def build_task_bindings(*, task, ct, nnmplayer, evaluator_cache: Dict, repo_root: str,
                        anchor_evaluator=None):
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
    # One shared anchor evaluator when the production path supplies it: a fresh
    # NNEvaluater per game would rebuild a TensorFlow SavedModel and session each
    # time. `evaluator` is a collaborator, not a frozen SETTING, so it is passed
    # separately and assert_anchor_settings still sees only the frozen keys.
    if anchor_evaluator is not None:
        return (lambda: nnmplayer.Player(evaluator=anchor_evaluator, **kwargs)), agent
    return (lambda: nnmplayer.Player(**kwargs)), agent


def _append_durably(path: Optional[str], record: dict) -> None:
    """Write one result and fsync it. A crash must not lose finished games."""
    if path is None:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


#: The only results a finished game may carry.
VALID_RESULTS = ("anchor", "reference", "draw", "draw_ply_cap")


def _identity(rec: dict) -> tuple:
    return (rec.get("task_index"), rec.get("seed"), rec.get("trials"),
            rec.get("reference"), rec.get("opening_id"), rec.get("colour_arm"))


def canonical_identities() -> set:
    """The exact 128 result identities a complete G3 run must produce."""
    from .twixtbot_g3_schedule import enumerate_tasks
    return {_identity(t) for t in enumerate_tasks()}


def assert_canonical_results(results: Sequence[dict]) -> None:
    """The result set must BE the canonical schedule. Nothing less counts.

    An earlier summarise() checked only for aborted records, so `all()` over the
    references actually present was vacuously true: a single draw against 0379
    selected trials=100 as passing "against BOTH references". Presence is now
    required, not assumed.
    """
    from .twixtbot_g3_schedule import REFERENCES, TRIALS_LADDER

    ids = [_identity(r) for r in results]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise RunnerError(f"duplicate result identities: {dupes[:3]}")
    got, want = set(ids), canonical_identities()
    if got != want:
        raise RunnerError(
            f"result set is not the canonical schedule: {len(want - got)} missing, "
            f"{len(got - want)} unexpected (have {len(got)} of {len(want)})"
        )
    for r in results:
        if r.get("aborted"):
            raise RunnerError(f"task {r.get('task_index')} aborted: {r.get('abort_reason')}")
        if r.get("result") not in VALID_RESULTS:
            raise RunnerError(
                f"task {r.get('task_index')} has result {r.get('result')!r}, "
                f"expected one of {VALID_RESULTS}"
            )
    for trials in TRIALS_LADDER:
        for ref in REFERENCES:
            n = len([r for r in results if r["trials"] == trials and r["reference"] == ref])
            if n != 16:
                raise RunnerError(f"trials={trials} ref={ref}: {n} games, expected 16")


def _prepare_results_path(results_path) -> str:
    """A NEW durable path is mandatory. No silent in-memory-only run."""
    if not isinstance(results_path, str) or not results_path:
        raise RunnerError("results_path is required; a run that persists nothing is not a run")
    if os.path.exists(results_path):
        raise RunnerError(f"results_path already exists: {results_path}")
    parent = os.path.dirname(os.path.abspath(results_path))
    if not os.path.isdir(parent):
        raise RunnerError(f"results directory does not exist: {parent}")
    return results_path


# --- anchor runtime identity -------------------------------------------------

def verify_anchor_runtime(clone_root: str) -> dict:
    """Prove the anchor runtime is the PINNED one, before any seed is touched.

    run_g3 previously accepted twixt/Point/ct/TwixtState/nnmplayer from its
    caller while loading the anchor from the relative path "model/pb". Nothing
    checked that those modules came from the pinned clone, that cwd was the clone
    root, or that the model bytes still matched -- so a different checkout or a
    fake Player satisfied the "production" entry point.
    """
    import hashlib
    import subprocess

    from .twixtbot_g3_schedule import ANCHOR

    root = os.path.realpath(clone_root)
    if not os.path.isdir(os.path.join(root, ".git")):
        raise RunnerError(f"not a git checkout: {root}")

    def git(*args):
        return subprocess.run(["git", "-C", root, *args], capture_output=True,
                              text=True, check=True).stdout.strip()

    head = git("rev-parse", "HEAD")
    if head != ANCHOR["commit"]:
        raise RunnerError(f"anchor clone is at {head}, pinned commit is {ANCHOR['commit']}")
    dirty = git("status", "--porcelain")
    if dirty:
        raise RunnerError(f"anchor clone is not clean:\n{dirty}")

    hashes = {
        "model/pb/variables/variables.data-00000-of-00001": ANCHOR["weights_sha256"],
        "model/pb/saved_model.pb": ANCHOR["saved_model_sha256"],
        "model/pb/variables/variables.index": ANCHOR["variables_index_sha256"],
    }
    for rel, want in hashes.items():
        h = hashlib.sha256()
        with open(os.path.join(root, rel), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != want:
            raise RunnerError(f"{rel}: {h.hexdigest()} != pinned {want}")

    return {"clone_root": root, "commit": head, "model_artifacts": len(hashes)}


def _import_anchor_runtime(clone_root: str, repo_root: str):
    """Import the engine modules FROM the verified clone, and prove their origin."""
    import importlib
    import sys

    root = os.path.realpath(clone_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    # NNEvaluater does os.path.join(os.getcwd(), model), so cwd IS part of the
    # anchor's identity. Set it here rather than trusting the caller's shell.
    os.chdir(root)

    twixt = importlib.import_module("src.backend.twixt")
    point = importlib.import_module("src.backend.point")
    ct = importlib.import_module("src.constants")
    nnmplayer = importlib.import_module("src.backend.nnmplayer")
    nneval = importlib.import_module("src.backend.nneval")
    state_mod = importlib.import_module("scripts.GPU.alphazero.game.twixt_state")

    for mod, base, label in ((twixt, root, "twixt"), (ct, root, "constants"),
                             (nnmplayer, root, "nnmplayer"), (nneval, root, "nneval"),
                             (state_mod, os.path.realpath(repo_root), "TwixtState")):
        origin = os.path.realpath(getattr(mod, "__file__", "") or "")
        if not origin.startswith(base + os.sep):
            raise RunnerError(f"{label} was imported from {origin}, not from {base}")
    if os.path.realpath(os.getcwd()) != root:
        raise RunnerError(f"cwd is {os.getcwd()}, expected the clone root {root}")

    return {"twixt": twixt, "Point": point.Point, "ct": ct, "nnmplayer": nnmplayer,
            "nneval": nneval, "TwixtState": state_mod.TwixtState}


# --- the production entry point ----------------------------------------------

#: Exit codes for the qualified command. The gate outcome IS the exit status.
EXIT_GATE_PASSED = 0
EXIT_GATE_FAILED = 1        # completed, but no trials setting passes
EXIT_RUN_ABORTED = 2        # stopped early; no verdict
EXIT_PRECONDITION = 3       # runtime identity / path / schedule refused


def run_g3(*, clone_root: str, repo_root: str, results_path: str) -> dict:
    """THE production entry point. NOT authorized to run.

    Takes only paths. It verifies the anchor runtime, imports the engines from
    the verified clone, enumerates and validates the canonical 128 tasks, loads
    ONE shared anchor evaluator, and requires a new durable results path.
    """
    from .twixtbot_g3_schedule import enumerate_tasks, schedule_invariants

    identity = verify_anchor_runtime(clone_root)
    rt = _import_anchor_runtime(clone_root, repo_root)

    tasks = enumerate_tasks()
    bad = schedule_invariants(tasks)
    if bad:
        raise RunnerError(f"the schedule is not valid: {bad}")
    path = _prepare_results_path(results_path)

    # ONE shared anchor evaluator for the whole run. The old factory built a new
    # NNEvaluater -- a fresh TensorFlow SavedModel and session -- for every game,
    # unlike G2's qualified evaluator-reuse path.
    shared_anchor_eval = rt["nneval"].NNEvaluater(ANCHOR_MODEL_DIR)

    out = _run_tasks(
        tasks=tasks, twixt=rt["twixt"], Point=rt["Point"], ct=rt["ct"],
        TwixtState=rt["TwixtState"], nnmplayer=rt["nnmplayer"], repo_root=repo_root,
        results_path=path, play_game=H.play_game, cleanup=RF.between_games_cleanup,
        require_canonical=True, anchor_evaluator=shared_anchor_eval,
    )
    out["anchor_identity"] = identity
    _write_terminal_record(path, out)
    return out


ANCHOR_MODEL_DIR = "model/pb"


def _write_terminal_record(path: str, out: dict) -> None:
    """Persist the VERDICT, not just the games.

    Without this the canonical assertion and the summary -- including
    selected_trials=None when nothing passes -- existed only in a returned Python
    object, which a caller could ignore.
    """
    _append_durably(path, {
        "record_type": "terminal",
        "completed": out.get("completed"),
        "stopped_at_task_index": out.get("stopped_at_task_index"),
        "stopped_reason": out.get("stopped_reason"),
        "games_played": out.get("games_played"),
        "tasks_remaining": out.get("tasks_remaining"),
        "summary": out.get("summary"),
        "gate": gate_verdict(out),
    })


def gate_verdict(out: dict) -> str:
    if not out.get("completed"):
        return "ABORTED"
    summary = out.get("summary") or {}
    return "PASSED" if summary.get("selected_trials") is not None else "FAILED"


def exit_code_for(out: dict) -> int:
    return {"PASSED": EXIT_GATE_PASSED, "FAILED": EXIT_GATE_FAILED,
            "ABORTED": EXIT_RUN_ABORTED}[gate_verdict(out)]


def main(argv=None) -> int:
    """The qualified command. Its EXIT STATUS binds the G3 outcome."""
    import argparse

    ap = argparse.ArgumentParser(description="Run the G3 anchor calibration schedule.")
    ap.add_argument("--clone-root", required=True)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--results-path", required=True)
    args = ap.parse_args(argv)
    try:
        out = run_g3(clone_root=args.clone_root, repo_root=args.repo_root,
                     results_path=args.results_path)
    except (RunnerError, RF.ReferenceError) as e:
        print(f"PRECONDITION REFUSED: {e}")
        return EXIT_PRECONDITION
    verdict = gate_verdict(out)
    print(f"G3 {verdict}: games_played={out.get('games_played')} "
          f"selected_trials={(out.get('summary') or {}).get('selected_trials')}")
    return exit_code_for(out)


def _run_tasks(
    *, tasks, twixt, Point, ct, TwixtState, nnmplayer, repo_root, results_path,
    play_game=H.play_game, cleanup=RF.between_games_cleanup, require_canonical=True,
    anchor_evaluator=None,
) -> dict:
    """Private. Injectable ONLY so the gates above can be negative-tested."""
    evaluator_cache: Dict = {}
    results: List[dict] = []
    games_played = 0

    for task in tasks:
        try:
            player_factory, agent = build_task_bindings(
                task=task, ct=ct, nnmplayer=nnmplayer,
                evaluator_cache=evaluator_cache, repo_root=repo_root,
                anchor_evaluator=anchor_evaluator,
            )
        except BaseException as e:                              # noqa: BLE001
            rec = _abort_record(task, "engine_exception", f"binding: {type(e).__name__}: {e}")
            results.append(rec)
            _append_durably(results_path, rec)
            return _stopped(results, task, rec, tasks, games_played)

        rec = play_game(
            task=task, twixt=twixt, Point=Point, ct=ct, TwixtState=TwixtState,
            player_factory=player_factory, reference_agent=agent, ply_cap=PLY_CAP,
        )
        results.append(rec)
        _append_durably(results_path, rec)
        if not rec.get("aborted"):
            games_played += 1

        if rec.get("aborted"):
            return _stopped(results, task, rec, tasks, games_played)

        try:
            cleanup()
        except BaseException as e:                              # noqa: BLE001
            stop = _abort_record(task, "engine_exception",
                                 f"between-game cleanup: {type(e).__name__}: {e}")
            stop["cleanup_failure"] = True
            stop["record_type"] = "run_stop"
            results.append(stop)
            _append_durably(results_path, stop)
            return _stopped(results, task, stop, tasks, games_played)

    summary = summarise(results) if require_canonical else summarise(results, strict=False)
    return {"completed": True, "stopped_at_task_index": None, "stopped_reason": None,
            "games_played": games_played, "tasks_remaining": 0, "results": results,
            "results_path": results_path, "summary": summary}


def _abort_record(task: dict, reason: str, detail: str) -> dict:
    return {
        "task_index": task.get("task_index"), "seed": task.get("seed"),
        "trials": task.get("trials"), "reference": task.get("reference"),
        "opening_id": task.get("opening_id"), "colour_arm": task.get("colour_arm"),
        "aborted": True, "abort_reason": reason, "abort_detail": detail,
        "result": None, "winner": None, "plies": 0, "moves": [],
    }


def _stopped(results, task, rec, tasks, games_played: int) -> dict:
    """Counts are of GAMES, not records.

    A cleanup failure appends both the completed game and a run-level stop record
    for the same task, so len(results) over-counts: with three tasks and cleanup
    failing after task 0 it reported n_played=2 / n_remaining=1, when one game had
    been played and two tasks remained. Stop records are excluded here.
    """
    stops = sum(1 for r in results if r.get("record_type") == "run_stop")
    tasks_attempted = len(results) - stops
    return {
        "completed": False,
        "stopped_at_task_index": task.get("task_index"),
        "stopped_reason": rec.get("abort_reason"),
        "stopped_detail": rec.get("abort_detail"),
        "games_played": games_played,
        "tasks_attempted": tasks_attempted,
        "tasks_remaining": len(tasks) - tasks_attempted,
        "results": results,
        # No summary on an incomplete run: scoring a filtered sample would read
        # as a result.
        "summary": None,
    }


def summarise(results: Sequence[dict], *, strict: bool = True) -> dict:
    """The two score rates per trials setting, and the lowest passing setting.

    Score is the ANCHOR's: win 1, draw 0.5, loss 0. Pass = non-saturation against
    BOTH references. Ordering is recorded DESCRIPTIVELY and is not a gate.

    `strict` requires the result set to BE the canonical 128-task schedule. It
    defaults to True and is set False only by the private test helper.
    """
    if strict:
        assert_canonical_results(results)
    elif any(r.get("aborted") for r in results):
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
