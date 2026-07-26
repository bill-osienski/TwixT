"""Checkpoint match: one A-vs-B pairing.

Builds one pairing's balanced-color tasks, runs them through the shared
eval_runner pool, aggregates a summary, and writes per-game JSONL + summary
JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, fields
from datetime import datetime, timezone

from .eval_runner import (
    EvalConfig, EvalGameResult, build_pairing_tasks, interpretation_forbidden,
    run_game_tasks, short_id, validate_labels,
)
from .eval_summary import summarize_match


def build_match_tasks(a_ckpt, b_ckpt, games, base_seed, pairing_id,
                      a_mcts=None, b_mcts=None, a_agent=None, b_agent=None):
    """Tasks for a single pairing (pairing_index fixed at 0)."""
    return build_pairing_tasks(pairing_id, a_ckpt, b_ckpt, games, base_seed,
                               pairing_index=0, a_mcts=a_mcts, b_mcts=b_mcts,
                               a_agent=a_agent, b_agent=b_agent)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


# Result fields introduced with agent mode. Emitted only when the run actually
# used agent identity, so legacy per-game JSONL keeps its exact bytes.
_AGENT_RESULT_FIELDS = ("red_agent", "black_agent", "winner_agent")


def _result_row(result, labels=None):
    row = asdict(result)
    if result.red_agent is None and result.black_agent is None:
        for name in _AGENT_RESULT_FIELDS:
            row.pop(name)
    if labels:
        row.update(labels)
    return row


def _write_outputs(output, summary, results, labels=None):
    out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)
    stem, _ext = os.path.splitext(output)
    games_path = f"{stem}_games.jsonl"
    with open(games_path, "w") as fh:
        for r in results:  # already sorted by (pairing_id, game_idx)
            fh.write(json.dumps(_result_row(r, labels)) + "\n")
    with open(output, "w") as fh:
        json.dump(summary, fh, indent=2)


def replay_dir_for(output, replay_dir_arg, save_enabled):
    """Resolve the replay output dir. None when capture is off; else the
    explicit --replay-dir, else <output-stem>_replays."""
    if not save_enabled:
        return None
    if replay_dir_arg:
        return replay_dir_arg
    stem, _ext = os.path.splitext(output)
    return f"{stem}_replays"


def run_match(a_ckpt, b_ckpt, games, base_seed, config, workers, output,
              pairing_id=None, evaluator_factory=None, replay_dir=None,
              a_mcts=None, b_mcts=None, a_agent=None, b_agent=None,
              allow_differ=(), config_validator=None, labels=None):
    """Run a full match and write outputs. Returns the summary dict.

    `a_mcts`/`b_mcts` give A and B their own search configs — for two agents on
    the same checkpoint that must search differently. `a_agent`/`b_agent` name
    them (default "A"/"B"); that identity is what lets the summary score two
    agents sharing one checkpoint instead of nulling every statistic as a
    self-match.

    `allow_differ` names the fields under study and `config_validator`
    constrains the base and both agent configs (v17 passes the frozen batching
    triple validator). run_game_tasks applies both to the complete task list
    before it loads any evaluator, so a misconfigured match fails without
    touching the GPU. Those checks are deliberately left in the shared runner
    rather than repeated here: it is the one point every caller routes through,
    including those that build tasks themselves.

    All agent arguments default to None, which reproduces the symmetric path
    exactly, summary and per-game bytes included.
    """
    # Validated ONCE, here at the public boundary, against the fields labels
    # are stamped over -- so no downstream writer has to re-check them.
    labels = validate_labels(
        labels, native_fields=[f.name for f in fields(EvalGameResult)])
    if pairing_id is None:
        pairing_id = f"{short_id(a_ckpt)}_vs_{short_id(b_ckpt)}"
    tasks = build_match_tasks(a_ckpt, b_ckpt, games, base_seed, pairing_id,
                              a_mcts=a_mcts, b_mcts=b_mcts,
                              a_agent=a_agent, b_agent=b_agent)
    results = run_game_tasks(tasks, workers=workers, config=config,
                             evaluator_factory=evaluator_factory,
                             replay_dir=replay_dir, allow_differ=allow_differ,
                             config_validator=config_validator, labels=labels)
    config_dict = {**asdict(config), "base_seed": base_seed, "workers": workers}
    if config.mcts_pending_virtual_visits is None:
        # Omitted entirely when unset, so a caller that never asked for it gets
        # byte-identical recorded config. When set, the EFFECTIVE value is
        # recorded rather than left to be inferred from an MCTSConfig default.
        config_dict.pop("mcts_pending_virtual_visits")
    # Agent mode is decided by the tasks, not by which argument was passed, so
    # the recorded provenance always describes what actually ran.
    first = tasks[0]
    agent_mode = first.red_agent is not None
    if agent_mode:
        # Complete effective search config per agent. Recorded only when the
        # feature is used, so the symmetric path's bytes are untouched.
        #
        # Agent ids are caller-controlled, so they get their own nested
        # namespace: sharing a dict with metadata would let an agent named
        # "allow_differ" overwrite its own config and silently break complete
        # provenance. Nesting makes that collision structurally impossible
        # rather than relying on a reserved-name list.
        by_id = {t.red_agent: t.red_mcts for t in tasks}
        by_id.update({t.black_agent: t.black_mcts for t in tasks})
        config_dict["agent_mcts"] = {
            "agents": {aid: asdict(cfg) for aid, cfg in sorted(by_id.items())},
            "allow_differ": sorted(allow_differ),
        }
    summary = summarize_match(
        results, a_ckpt, b_ckpt, pairing_id, config_dict,
        a_agent=first.red_agent if agent_mode else None,
        b_agent=first.black_agent if agent_mode else None)
    if labels:
        # Stamped BEFORE the summary is written and hashed, so downstream
        # qualification and every recorded identity cover the labelled bytes.
        summary.update(labels)
    summary["git_commit"] = _git_commit()
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    if output:
        _write_outputs(output, summary, results, labels)
    return summary


def _build_arg_parser():
    ap = argparse.ArgumentParser(description="Run a checkpoint A-vs-B match.")
    ap.add_argument("--checkpoint-a", required=True)
    ap.add_argument("--checkpoint-b", required=True)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--board-size", type=int, default=24)
    ap.add_argument("--mcts-sims", type=int, default=400)
    ap.add_argument("--mcts-eval-batch-size", type=int, default=14)
    ap.add_argument("--mcts-stall-flush-sims", type=int, default=48)
    ap.add_argument("--selection-mode", default="opening_temperature",
                    choices=["opening_temperature", "argmax"])
    ap.add_argument("--opening-temp-plies", type=int, default=20)
    ap.add_argument("--temp-high", type=float, default=1.0)
    ap.add_argument("--temp-low", type=float, default=0.1)
    ap.add_argument("--max-moves", type=int, default=280)
    ap.add_argument("--workers", type=int, default=1,
                    help="default 1. Values >1 spawn multiple MLX processes and "
                         "may exceed Metal resource limits on some Macs; run a "
                         "small --workers 2 probe before a large parallel run.")
    ap.add_argument("--base-seed", type=int, default=12345)
    ap.add_argument("--save-eval-replays", action="store_true",
                    help="write a per-ply replay sidecar per game and link it "
                         "from each *_games.jsonl row (default off).")
    ap.add_argument("--replay-dir", default=None,
                    help="replay output dir (default <output-stem>_replays); "
                         "only used with --save-eval-replays.")
    ap.add_argument("--mcts-pending-virtual-visits", type=int, default=None,
                    help="state the value explicitly instead of inheriting "
                         "MCTSConfig's default. Omitted by default, which "
                         "leaves artifact bytes unchanged.")
    ap.add_argument("--require-batching-triple", default=None,
                    help="assert the EFFECTIVE (eval_batch_size, "
                         "stall_flush_sims, pending_virtual_visits) equals "
                         "this comma-separated triple, BEFORE any evaluator "
                         "loads. A mismatch in any element refuses.")
    ap.add_argument("--run-kind", default=None,
                    help="stamp run_kind onto the summary, every per-game JSONL "
                         "row and every replay sidecar. Omitted by default, "
                         "which leaves artifact bytes exactly as before.")
    ap.add_argument("--scientific-interpretation",
                    choices=["forbidden", "allowed"], default=None,
                    help="REQUIRED with --run-kind. There is no default: a "
                         "silent 'allowed' would mislabel a tooling/smoke "
                         "artifact as interpretable.")
    ap.add_argument("--output", required=True)
    return ap


def _config_from_args(args) -> EvalConfig:
    return EvalConfig(
        mcts_pending_virtual_visits=args.mcts_pending_virtual_visits,
        board_size=args.board_size, mcts_sims=args.mcts_sims,
        mcts_eval_batch_size=args.mcts_eval_batch_size,
        mcts_stall_flush_sims=args.mcts_stall_flush_sims,
        selection_mode=args.selection_mode,
        opening_temp_plies=args.opening_temp_plies,
        temp_high=args.temp_high, temp_low=args.temp_low,
        max_moves=args.max_moves,
    )


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if args.scientific_interpretation and not args.run_kind:
        raise SystemExit("--scientific-interpretation requires --run-kind "
                         "(a label without a run kind is not a complete stamp)")
    if args.run_kind and not args.scientific_interpretation:
        raise SystemExit(
            "--run-kind requires --scientific-interpretation "
            "{forbidden,allowed}: defaulting it would let a tooling_smoke run "
            "emit scientific_interpretation_forbidden=false")
    if args.require_batching_triple:
        # BEFORE checkpoint resolution and before any evaluator load.
        from .eval_runner import cfg_from as _cfg_from
        want = tuple(int(x) for x in args.require_batching_triple.split(","))
        if len(want) != 3:
            raise SystemExit("--require-batching-triple needs three values")
        if args.mcts_pending_virtual_visits is None:
            # Comparing the EFFECTIVE triple alone cannot tell "stated 8" from
            # "inherited 8" -- both land on 8 today. Design 2.4 requires
            # explicit derivation, so a run asserting the triple must also
            # state the value it is asserting.
            raise SystemExit(
                "--require-batching-triple requires "
                "--mcts-pending-virtual-visits: without it the value is "
                "inherited from MCTSConfig's default, which is not the "
                "explicit derivation the frozen design requires")
        eff = _cfg_from(_config_from_args(args))
        got = (eff.eval_batch_size, eff.stall_flush_sims,
               eff.pending_virtual_visits)
        if got != want:
            raise SystemExit(
                f"effective batching triple {got} != required {want}; results "
                f"at a different triple are incomparable, not merely slower")
    labels = None
    if args.run_kind:
        # DERIVED from the run kind, never taken from the flag: the flag only
        # states the operator's intent, and a mismatch is refused below.
        declared = args.scientific_interpretation == "forbidden"
        try:
            derived = interpretation_forbidden(args.run_kind)
        except ValueError as exc:
            raise SystemExit(f"--run-kind: {exc}")
        if declared != derived:
            raise SystemExit(
                f"--scientific-interpretation "
                f"{args.scientific_interpretation!r} contradicts --run-kind "
                f"{args.run_kind!r}, which requires "
                f"{'forbidden' if derived else 'allowed'}")
        labels = {"run_kind": args.run_kind,
                  "scientific_interpretation_forbidden": derived}
    for path in (args.checkpoint_a, args.checkpoint_b):
        if not os.path.exists(path):
            raise SystemExit(f"checkpoint not found: {path}")
    replay_dir = replay_dir_for(args.output, args.replay_dir, args.save_eval_replays)
    summary = run_match(
        a_ckpt=args.checkpoint_a, b_ckpt=args.checkpoint_b, games=args.games,
        base_seed=args.base_seed, config=_config_from_args(args),
        workers=args.workers, output=args.output, replay_dir=replay_dir,
        labels=labels,
    )
    if summary.get("self_match"):
        cb = summary["color_bias"]["red_win_rate_decisive"]
        print(f"{summary['pairing_id']}: SELF-MATCH — per-checkpoint score "
              f"undefined; red_win_rate_decisive={cb} "
              f"(see color_bias / a_as_red+black are null by design)")
    else:
        print(f"{summary['pairing_id']}: a_score_rate={summary['a_score_rate']:.4f} "
              f"elo={summary['elo_estimate']:.1f} "
              f"CI95={summary['elo_ci95']} verdict={summary['verdict']}")


if __name__ == "__main__":
    main()
