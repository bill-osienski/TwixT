"""Two-agent, one-checkpoint readout match.

Both competitors load the SAME checkpoint and differ only in how they turn a
completed search into a played move. Scoring is by agent identity, because
checkpoint keying cannot separate them (eval_summary returns None for score
rate, Elo and CIs whenever checkpoint_a == checkpoint_b).

The historical checkpoint-vs-checkpoint CLI (eval_checkpoint_match.py) is
deliberately untouched.

NO GPU RUN IS AUTHORIZED BY THIS TOOL. Running it requires a separate written
authorization naming the exact scope, seed interval and game count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone

from .eval_readout import (
    MODE_ARGMAX, MODE_HOEFFDING_LCB, MODE_OPENING_TEMPERATURE, ReadoutConfig,
)
from .eval_runner import (
    AgentSpec, EvalConfig, build_agent_pairing_tasks, run_game_tasks,
)
from .eval_summary import summarize_agent_match

CANDIDATE_ID = "candidate"
CONTROL_ID = "control"

READOUT_NAMES = ("tournament", "argmax", "opening_then_argmax", "hoeffding_lcb")


def readout_config_from_name(name, opening_temp_plies, temp_high, temp_low):
    """Map a CLI name to a ReadoutConfig.

    tournament          -- the shipped tournament readout (temp_high then temp_low)
    argmax              -- all-ply visit argmax (Candidate 1)
    opening_then_argmax -- temp_high opening, then argmax (Candidate 2 CONTROL)
    hoeffding_lcb       -- temp_high opening, then argmax + frozen override
                           (Candidate 2 CANDIDATE)
    """
    if name == "tournament":
        return ReadoutConfig(mode=MODE_OPENING_TEMPERATURE,
                             opening_temp_plies=opening_temp_plies,
                             temp_high=temp_high, temp_low=temp_low)
    if name == "argmax":
        return ReadoutConfig(mode=MODE_ARGMAX)
    if name == "opening_then_argmax":
        return ReadoutConfig(mode=MODE_OPENING_TEMPERATURE,
                             opening_temp_plies=opening_temp_plies,
                             temp_high=temp_high, temp_low=0.0)
    if name == "hoeffding_lcb":
        return ReadoutConfig(mode=MODE_HOEFFDING_LCB,
                             opening_temp_plies=opening_temp_plies,
                             temp_high=temp_high, temp_low=temp_low)
    raise ValueError(f"unknown readout name {name!r}; expected {READOUT_NAMES}")


def _source_repo_dir():
    """The repository containing the EXECUTING source.

    Provenance must describe the code that actually ran. Anchoring to the
    process CWD, or to a caller-supplied path, would let dirty engine code run
    while a pristine unrelated repository was recorded.
    """
    return os.path.dirname(os.path.abspath(__file__))


def _git_provenance(repo_dir):
    """Commit and worktree state for `repo_dir`. Fails closed.

    A DIRTY WORKTREE IS REFUSED OUTRIGHT, with no override. Recording
    `git status --porcelain` would name the changed files but not their
    contents, so a "dirty but documented" run is still not reproducible from
    its own provenance.

    PRIVATE. `repo_dir` exists so this function can be unit-tested against
    constructed repositories. It is NOT reachable from run_readout_match's
    signature -- production always passes _source_repo_dir(), so no caller can
    aim provenance at a different repository.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.PIPE, cwd=repo_dir
        ).decode().strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.PIPE, cwd=repo_dir
        ).decode()
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            f"cannot establish git provenance ({e!r}); a run whose code state "
            f"is unknown is not reproducible") from e
    if porcelain.strip():
        raise RuntimeError(
            "worktree is dirty; a recorded run must come from a committed "
            "state. Recording the porcelain status would name the changed "
            "files but not their contents, so it cannot substitute for a "
            f"commit. Commit or stash first:\n{porcelain}")
    return {"git_commit": commit, "worktree_clean": True}


def _sha1(path):
    """Checkpoint hash. Fails closed: the spec requires a cryptographic hash,
    so an unreadable checkpoint aborts the run rather than recording None."""
    h = hashlib.sha1()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError as e:
        raise RuntimeError(
            f"cannot hash checkpoint {path!r} ({e!r}); provenance requires a "
            f"checkpoint hash") from e
    return h.hexdigest()


def _write_outputs(output, summary, results):
    out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)
    stem, _ext = os.path.splitext(output)
    with open(f"{stem}_games.jsonl", "w") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r)) + "\n")
    with open(output, "w") as fh:
        json.dump(summary, fh, indent=2)


def run_readout_match(checkpoint, candidate_readout, control_readout, games,
                      base_seed, config: EvalConfig, workers, output,
                      pairing_id=None, evaluator_factory=None, replay_dir=None,
                      prior_seed_intervals=()):
    """Run one candidate-vs-control readout match. Returns the summary dict.

    `prior_seed_intervals` is every interval already consumed by this line of
    work, as half-open [start, end) pairs. B6 RECORDS them; Task B7 adds
    eval_integrity.validate_seed_intervals and ENFORCES disjointness here.

    There is deliberately NO repository parameter: provenance is anchored to
    the repository containing this source file, so a run can never record a
    different repository than the one it executed from.

    Provenance runs BEFORE any game, so an unreproducible configuration costs
    zero GPU time.
    """
    if candidate_readout == control_readout:
        raise ValueError("candidate and control readouts are identical; "
                         "this match would measure nothing")
    provenance = _git_provenance(_source_repo_dir())
    checkpoint_sha1 = _sha1(checkpoint)
    priors = [list(iv) for iv in prior_seed_intervals]
    started = time.monotonic()
    if pairing_id is None:
        pairing_id = f"{CANDIDATE_ID}_vs_{CONTROL_ID}"
    candidate = AgentSpec(CANDIDATE_ID, checkpoint, candidate_readout)
    control = AgentSpec(CONTROL_ID, checkpoint, control_readout)
    tasks = build_agent_pairing_tasks(pairing_id, candidate, control, games,
                                      base_seed)
    results = run_game_tasks(tasks, workers=workers, config=config,
                             evaluator_factory=evaluator_factory,
                             replay_dir=replay_dir)
    config_dict = {
        **asdict(config),
        "base_seed": base_seed,
        "workers": workers,
        # COMPLETE configs: mode alone cannot distinguish `tournament` from
        # `opening_then_argmax` (both mode "opening_temperature").
        "candidate_readout": asdict(candidate_readout),
        "control_readout": asdict(control_readout),
        "checkpoint": checkpoint,
        "checkpoint_sha1": checkpoint_sha1,
        # HALF-OPEN [start, end): game g uses seed base_seed + g for
        # g in range(games), so `end` is the first UNUSED seed. A later run
        # proving disjointness must compare against this convention.
        "seed_interval": [base_seed, base_seed + games],
        "seed_interval_convention": "half_open_[start,end)",
        # Spec section 10 requires the intervals of every prior run, so a
        # reader can verify disjointness without external bookkeeping.
        "prior_seed_intervals": priors,
        # Search and readout draw from separate streams (spec section 7.1).
        # Recorded so a reader can reconstruct any game exactly.
        "rng_derivation": {
            "search_red": "seed ^ 0xA5A5A5",
            "search_black": "seed ^ 0x5A5A5A",
            "readout_red": "seed ^ 0xC3C3C3",
            "readout_black": "seed ^ 0x3C3C3C",
            "game_seed": "base_seed + game_idx",
        },
    }
    summary = summarize_agent_match(results, CANDIDATE_ID, CONTROL_ID,
                                    pairing_id, config_dict)
    summary.update(provenance)
    summary["wall_clock_seconds"] = time.monotonic() - started
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    if output:
        _write_outputs(output, summary, results)
    return summary


def _build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Two-agent readout match on ONE checkpoint. "
                    "Requires a separate written GPU authorization.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--candidate-readout", required=True, choices=READOUT_NAMES)
    ap.add_argument("--control-readout", required=True, choices=READOUT_NAMES)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--base-seed", type=int, required=True)
    ap.add_argument("--board-size", type=int, default=24)
    ap.add_argument("--mcts-sims", type=int, default=400)
    ap.add_argument("--mcts-eval-batch-size", type=int, default=14)
    ap.add_argument("--mcts-stall-flush-sims", type=int, default=48)
    ap.add_argument("--opening-temp-plies", type=int, default=20)
    ap.add_argument("--temp-high", type=float, default=1.0)
    ap.add_argument("--temp-low", type=float, default=0.1)
    ap.add_argument("--max-moves", type=int, default=280)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--replay-dir", default=None,
                    help="capture per-ply replays incl. top-two child "
                         "visits/Q (required to feed the preflight)")
    ap.add_argument("--prior-seed-interval", action="append", default=[],
                    metavar="START:END",
                    help="a half-open [START,END) seed interval already "
                         "consumed by this line of work. Repeatable. Recorded "
                         "in the summary so disjointness is verifiable from "
                         "the artifact alone.")
    ap.add_argument("--output", required=True)
    return ap


def _parse_interval(text):
    try:
        start, end = text.split(":")
        return [int(start), int(end)]
    except ValueError as e:
        raise SystemExit(
            f"bad --prior-seed-interval {text!r}; expected START:END") from e


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    config = EvalConfig(
        board_size=args.board_size, mcts_sims=args.mcts_sims,
        mcts_eval_batch_size=args.mcts_eval_batch_size,
        mcts_stall_flush_sims=args.mcts_stall_flush_sims,
        opening_temp_plies=args.opening_temp_plies,
        temp_high=args.temp_high, temp_low=args.temp_low,
        max_moves=args.max_moves,
    )

    def _mk(name):
        return readout_config_from_name(
            name, args.opening_temp_plies, args.temp_high, args.temp_low)

    summary = run_readout_match(
        checkpoint=args.checkpoint,
        candidate_readout=_mk(args.candidate_readout),
        control_readout=_mk(args.control_readout),
        games=args.games, base_seed=args.base_seed, config=config,
        workers=args.workers, output=args.output, replay_dir=args.replay_dir,
        prior_seed_intervals=[_parse_interval(t)
                              for t in args.prior_seed_interval],
    )
    print(f"{summary['pairing_id']}: a_score_rate={summary['a_score_rate']:.4f} "
          f"elo={summary['elo_estimate']:.1f} CI95={summary['elo_ci95']} "
          f"verdict={summary['verdict']}")


if __name__ == "__main__":
    main()
