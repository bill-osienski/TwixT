"""Tier-parameterized probe suite generator.

Replaces scripts/build_bootstrap_probe_suite.py as the real implementation
(that script is kept as a thin --tier forced shim for muscle memory and
existing CI/cron commands).

Tiers:
  --tier forced            Bootstrap forced suite (existing behavior,
                           writes tests/probes/twixt_probes.json by default).
  --tier strong_advantage  Bootstrap strong-advantage suite (deep-MCTS
                           labeled, light-reviewed). Phases 1/2/3 per
                           docs/superpowers/specs/2026-04-28-...

Both tiers produce byte-identical output for identical inputs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


# Maximum MCTS evaluator batch size known stable on Apple Metal/MLX.
# scripts/GPU/alphazero/mcts.py:106 documents that batches > this value have
# previously caused Metal GPU hangs. The probe builder caps --mcts-eval-batch-size
# at this value unless --allow-unsafe-eval-batch is passed.
SAFE_METAL_EVAL_BATCH_SIZE_MAX = 14


def _init_label_worker(label_checkpoint: str, mcts_cfg_payload: dict) -> None:
    """ProcessPoolExecutor initializer: load network and register MCTSConfig.

    Each worker process holds its own MLX network (own MLX context) and its
    own copy of the registered MCTSConfig. n_simulations is per-call and
    NOT in the payload — see spec §8.
    """
    from scripts.GPU.alphazero.probe_eval import (
        load_network_for_scoring,
        _set_default_labeler_network,
        _set_default_labeler_mcts_config,
    )
    from scripts.GPU.alphazero.mcts import MCTSConfig
    network, _ic, _h, _nb = load_network_for_scoring(label_checkpoint)
    network.eval()
    _set_default_labeler_network(network)
    _set_default_labeler_mcts_config(MCTSConfig(**mcts_cfg_payload))


# --- Diversity selector constants and helpers ---

MIN_PLY_SEPARATION_SAME_GAME = 3
"""Same-game probes must be at least this many plies apart. Tied to the
current K-range [3, 8]: with span 5, separation 3 admits at most 2 plies
per game, matching the default --max-probes-per-game cap."""

CATEGORY_ITERATION_ORDER = (
    "chain_advantage_central_red",
    "chain_advantage_central_black",
    "chain_advantage_edge_red",
    "chain_advantage_edge_black",
)
"""Fixed canonical order for round-robin category fill. Empty buckets
are skipped at iteration time. See spec §5.4."""


def _diversity_sort_key(cand: dict) -> tuple:
    """Stage-2 rank key: structural-first, Phase-2 secondary, source order
    as final determinism guarantee. Lower tuple sorts first. See spec §4.2."""
    p1 = cand["phase1_features"]
    p2 = cand["phase2_label"]
    try:
        iter_num = int(cand["source_game"].split("_")[1])
    except (IndexError, ValueError):
        iter_num = 0
    return (
        -p1["cc_size"],
        -p1["axis_span_margin"],
        -p1["cc_axis_span"],
        -p2["min_top1_share"],
        p2["value_stability"],
        -iter_num,
        -cand["source_ply"],
        cand["source_game"],
    )


def _find_near_duplicate_keeper(cand: dict, kept: list) -> dict | None:
    """Rule A — Near-duplicate. Returns the matching kept candidate or None.

    Same source_game AND same category AND |Δcc_size| < 2 AND
    |Δaxis_span_margin| < 0.05. Multiple matches: smallest source_ply
    (deterministic). See spec §4.2.
    """
    cand_p1 = cand["phase1_features"]
    matches = [
        k for k in kept
        if k["source_game"] == cand["source_game"]
        and k["category"] == cand["category"]
        and abs(k["phase1_features"]["cc_size"] - cand_p1["cc_size"]) < 2
        and abs(k["phase1_features"]["axis_span_margin"] - cand_p1["axis_span_margin"]) < 0.05
    ]
    if not matches:
        return None
    return min(matches, key=lambda k: k["source_ply"])


def _find_ply_too_close_keeper(cand: dict, kept: list, rank_index: dict) -> dict | None:
    """Rule B — Ply-too-close. Returns the blocking kept candidate or None.

    Same source_game AND |Δsource_ply| < MIN_PLY_SEPARATION_SAME_GAME (any
    category — Rule B is category-agnostic).

    Tiered tie-break:
      1. Closest kept sibling (smallest |Δsource_ply|).
      2. Better Stage-2 rank (smaller rank_index value).
      3. Smallest source_ply.

    rank_index: maps id(cand) to its position in its category's Stage-2
    sort order. The selector builds this once after Stage 2.
    Precondition: rank_index must contain id(k) for every candidate in
    kept. Missing keys raise KeyError — the selector is responsible for
    populating rank_index for all candidates before any are moved to kept.
    See spec §4.2.
    """
    matches = [
        k for k in kept
        if k["source_game"] == cand["source_game"]
        and abs(k["source_ply"] - cand["source_ply"]) < MIN_PLY_SEPARATION_SAME_GAME
    ]
    if not matches:
        return None
    return min(
        matches,
        key=lambda k: (
            abs(k["source_ply"] - cand["source_ply"]),
            rank_index[id(k)],
            k["source_ply"],
        ),
    )


def _find_per_game_cap_keeper(cand: dict, kept: list, cap: int) -> dict | None:
    """Rule C — Per-game cap. Returns the smallest-source_ply keeper from
    the same game when the cap is reached (len(from_game) >= cap), else
    None. Counted total across all categories. The helper is called
    BEFORE the candidate enters kept, so "reached" (not "exceeded") is
    the firing condition. See spec §4.2 / §5.2.
    """
    from_game = [k for k in kept if k["source_game"] == cand["source_game"]]
    if len(from_game) < cap:
        return None
    return min(from_game, key=lambda k: k["source_ply"])


def _select_diverse_admitted_candidates(
    admitted: list,
    audit: list,
    *,
    max_probes: int,
    max_probes_per_game: int,
) -> list:
    """Post-Phase-2 diversity-aware selector. Replaces the simple
    `admitted[: max_probes]` slice with a category round-robin walk
    applying near-duplicate, ply-separation, and per-game cap rules.

    Mutates `audit` in place: appends one row per CONSIDERED candidate
    (reason="admitted" if kept, reason="diversity_*" if dropped). Returns
    the kept list in selection order.

    Audit-coverage policy (Option A — by design): once `max_probes` is
    reached, the round-robin terminates and any remaining post-Phase-2
    candidates are NOT visited and therefore get NO audit row. The audit
    is exhaustive over CONSIDERED candidates, not over ALL Phase-2
    survivors. Rationale: an unvisited candidate is not "evicted by a
    rule" — it simply lost the race for a finite suite slot. Tagging
    every Phase-2 survivor with a "would have been considered next"
    pseudo-reason adds noise without diagnostic value. Operators who
    need a total Phase-2-admit count should derive it externally
    (e.g., len(admitted) before this function is called), not by
    counting audit rows.

    See spec §4.2 for the algorithm and §7 for audit semantics.
    """
    # Stage 1: bucket by category.
    buckets = {cat: [] for cat in CATEGORY_ITERATION_ORDER}
    for cand in admitted:
        cat = cand["category"]
        if cat in buckets:
            buckets[cat].append(cand)

    # Stage 2: rank within each category.
    for cat in buckets:
        buckets[cat].sort(key=_diversity_sort_key)

    # Build rank_index: id(cand) → rank position in its category.
    # Used by Rule B's tie-break (better Stage-2 rank wins).
    rank_index = {}
    for cands in buckets.values():
        for i, c in enumerate(cands):
            rank_index[id(c)] = i

    # Stage 3: round-robin walk with suppression rules.
    kept = []
    cursors = {cat: 0 for cat in CATEGORY_ITERATION_ORDER}

    while len(kept) < max_probes:
        progressed = False
        for cat in CATEGORY_ITERATION_ORDER:
            if len(kept) >= max_probes:
                break
            if cursors[cat] >= len(buckets[cat]):
                continue
            progressed = True
            cand = buckets[cat][cursors[cat]]
            cursors[cat] += 1

            audit_base = {
                "source_game": cand["source_game"],
                "source_ply": cand["source_ply"],
                "phase1_features": cand["phase1_features"],
                "phase2_label": cand["phase2_label"],
            }

            # Rule A: near-duplicate.
            keeper = _find_near_duplicate_keeper(cand, kept)
            if keeper is not None:
                row = {
                    **audit_base,
                    "reason": "diversity_near_duplicate",
                    "kept_instead_source_ply": keeper["source_ply"],
                }
                _merge_borderline_audit(row, cand)
                audit.append(row)
                continue

            # Rule B: ply-too-close.
            keeper = _find_ply_too_close_keeper(cand, kept, rank_index)
            if keeper is not None:
                row = {
                    **audit_base,
                    "reason": "diversity_ply_too_close",
                    "kept_instead_source_ply": keeper["source_ply"],
                }
                _merge_borderline_audit(row, cand)
                audit.append(row)
                continue

            # Rule C: per-game cap.
            keeper = _find_per_game_cap_keeper(cand, kept, max_probes_per_game)
            if keeper is not None:
                row = {
                    **audit_base,
                    "reason": "diversity_per_game_cap",
                    "kept_instead_source_ply": keeper["source_ply"],
                }
                _merge_borderline_audit(row, cand)
                audit.append(row)
                continue

            # Admit.
            kept.append(cand)
            row = {**audit_base, "reason": "admitted"}
            _merge_borderline_audit(row, cand)
            audit.append(row)

        if not progressed:
            break

    return kept


# --- Tier dispatch ---

def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--tier", choices=["forced", "strong_advantage"], required=True)
    ap.add_argument("--input", default="scripts/GPU/logs/games")
    ap.add_argument("--source-iter-range", nargs=2, type=int,
                    metavar=("MIN", "MAX"))
    ap.add_argument("--out", default=None,
                    help="Output path. Defaults: forced -> tests/probes/twixt_probes.json, "
                         "strong_advantage -> tests/probes/strong_advantage_probes.json")
    ap.add_argument("--samples-per-bucket", type=int, default=12)
    ap.add_argument("--max-probes", type=int, default=30)
    ap.add_argument("--max-probes-per-game", type=int, default=2,
                    help="Maximum number of admitted probes from any single "
                         "source game. Counts total across all 4 categories. "
                         "Default 2. Strong-advantage tier only.")

    # strong_advantage-specific flags (ignored for forced)
    ap.add_argument("--label-checkpoint", default=None)
    ap.add_argument("--label-mcts-sims", type=int, default=10000)
    ap.add_argument("--label-mcts-repeats", type=int, default=3)
    ap.add_argument("--magnitude-threshold", type=float, default=0.45)
    ap.add_argument("--top1-share-floor", type=float, default=0.15)
    ap.add_argument("--stability-cap", type=float, default=0.15)
    ap.add_argument("--promote", action="store_true",
                    help="Promote *.draft.json to committed file")
    ap.add_argument("--reviewer", default=None,
                    help="Reviewer name, required with --promote")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing draft or committed file")

    # Phase 2 parallel-labeling flags (strong_advantage tier only)
    ap.add_argument("--label-worker-mode", choices=["serial", "process"],
                    default="serial",
                    help="Phase 2 execution mode. Default 'serial' is the "
                         "byte-reference path. 'process' enables a process pool.")
    ap.add_argument("--label-workers", type=int, default=1,
                    help="Worker count under --label-worker-mode=process. "
                         "Ignored under serial. Apple Silicon: start with 2-4.")
    ap.add_argument("--mcts-eval-batch-size", type=int, default=14,
                    help=(f"NN batch size for the labeler's MCTS. Capped at "
                          f"{SAFE_METAL_EVAL_BATCH_SIZE_MAX} because larger "
                          "batches have caused Metal hangs; pass "
                          "--allow-unsafe-eval-batch to exceed."))
    ap.add_argument("--mcts-stall-flush-sims", type=int, default=16,
                    help="MCTS stall-flush threshold (see MCTSConfig). 0 disables.")
    ap.add_argument("--allow-unsafe-eval-batch", action="store_true",
                    help="Required to set --mcts-eval-batch-size > "
                         f"{SAFE_METAL_EVAL_BATCH_SIZE_MAX}. Benchmark only.")
    ap.add_argument("--admission-borderline-epsilon", type=float, default=0.01,
                    help="In process mode, candidates whose phase-2 label is "
                         "within epsilon of any admission threshold are "
                         "re-labeled in the main process to use the serial "
                         "reference label. 0 disables.")
    ap.add_argument("--no-borderline-rerun", action="store_true",
                    help="Disable borderline rerun even when epsilon > 0.")
    return ap


def _validate_parallel_args(ap: argparse.ArgumentParser, args) -> None:
    """Validate the new Phase 2 parallel flags. Calls ap.error() on failure."""
    if args.label_workers < 1:
        ap.error("--label-workers must be >= 1")
    if args.mcts_eval_batch_size < 1:
        ap.error("--mcts-eval-batch-size must be >= 1")
    if (args.mcts_eval_batch_size > SAFE_METAL_EVAL_BATCH_SIZE_MAX
            and not args.allow_unsafe_eval_batch):
        ap.error(
            f"--mcts-eval-batch-size > {SAFE_METAL_EVAL_BATCH_SIZE_MAX} "
            "is unsafe on Metal/MLX and may hang. "
            "Pass --allow-unsafe-eval-batch to benchmark higher values intentionally."
        )
    if args.mcts_stall_flush_sims < 0:
        ap.error("--mcts-stall-flush-sims must be >= 0")
    if args.admission_borderline_epsilon < 0:
        ap.error("--admission-borderline-epsilon must be >= 0")


def main() -> int:
    ap = _build_arg_parser()
    args = ap.parse_args()
    _validate_parallel_args(ap, args)

    # Workers under serial mode: warn if explicitly set to anything other than 1.
    if args.label_worker_mode == "serial" and args.label_workers != 1:
        print("[probe_suite] warning: --label-workers is ignored when "
              "--label-worker-mode=serial", file=sys.stderr)
        args.label_workers = 1

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if args.tier == "forced":
        return _run_forced(args)
    elif args.tier == "strong_advantage":
        return _run_strong_advantage(args)
    else:
        print(f"[probe_suite] ERROR: unknown tier {args.tier}", file=sys.stderr)
        return 2


# --- Forced tier (lifted from build_bootstrap_probe_suite.py) ---

def _run_forced(args) -> int:
    if args.out is None:
        args.out = "tests/probes/twixt_probes.json"
    if args.source_iter_range is None:
        print("[probe_suite] ERROR: --source-iter-range required for --tier forced",
              file=sys.stderr)
        return 2

    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games

    min_iter, max_iter = args.source_iter_range
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"[probe_suite] ERROR: --input path is not a directory: {input_dir}",
              file=sys.stderr)
        return 2

    games: list[dict] = []
    for fp in sorted(input_dir.glob("iter_*_game_*.json")):
        with open(fp) as f:
            try:
                g = json.load(f)
            except json.JSONDecodeError:
                continue
        iteration = (g.get("meta") or {}).get("iteration")
        if iteration is None or not (min_iter <= iteration <= max_iter):
            continue
        games.append(g)

    probes = extract_forced_probes_from_games(
        games,
        active_size=24,
        k_plies=2,
        winner_reasons=frozenset({"win"}),
        dedupe_exact=True,
        dedupe_mirror=True,
        max_probes=None,
    )

    # Interleave-then-truncate: balance must survive truncation.
    # extract_forced_probes_from_games already returned each color's probes
    # in canonical sort order. We merge red/black greedily into `balanced`,
    # at each step taking the color with the better sort key AS LONG AS
    # the ≤ 2:1 balance rule would still hold. Stop at max_probes.
    #
    # An earlier version applied a pre-truncation cap and then truncated,
    # but the final truncation could skew the output (e.g., all top-N
    # probes came from the same color when the most recent iters favored
    # that color). Interleaving closes that gap.

    def _sort_key(p: dict) -> tuple:
        basename = p["source_game"]
        try:
            iter_num = int(basename.split("_")[1])
        except (IndexError, ValueError):
            iter_num = 0
        return (-iter_num, -p["source_ply"], basename)

    red = [p for p in probes if p["category"] == "near_win_red"]
    black = [p for p in probes if p["category"] == "near_win_black"]

    balanced: list[dict] = []
    ri = bi = 0
    red_count = black_count = 0
    while len(balanced) < args.max_probes:
        can_red = ri < len(red) and red_count + 1 <= 2 * max(black_count, 1)
        can_black = bi < len(black) and black_count + 1 <= 2 * max(red_count, 1)
        if not can_red and not can_black:
            break
        if can_red and can_black:
            if _sort_key(red[ri]) <= _sort_key(black[bi]):
                balanced.append(red[ri]); ri += 1; red_count += 1
            else:
                balanced.append(black[bi]); bi += 1; black_count += 1
        elif can_red:
            balanced.append(red[ri]); ri += 1; red_count += 1
        else:
            balanced.append(black[bi]); bi += 1; black_count += 1

    balanced.sort(key=_sort_key)

    payload = {
        "meta": {
            "type": "bootstrap_rule_selected",
            "not_gate_suite": True,
            "note": ("Rule-selected bootstrap suite for trainer-side inline "
                     "telemetry and practical regression monitoring. NOT the "
                     "spec §7 review-curated gate suite — see "
                     "tests/probes/README.md for the distinction."),
            "generator": "scripts/build_bootstrap_probe_suite.py",
            "generator_version": 1,
            "selection_rules": {
                "board_size": 24,
                "winner_reasons": ["win"],
                "k_plies_from_terminal": 2,
                "dedup": "exact + 4-form-mirror-canonical",
                "source_iter_range": [min_iter, max_iter],
            },
        },
        "probes": balanced,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")

    print(f"[probe_suite] wrote {len(balanced)} forced probes to {out_path}")
    return 0


# --- Strong-advantage tier ---


def _probe_id_and_seed_base(cand: dict) -> tuple[str, int]:
    """Compute the deterministic (probe_id, rng_seed_base) pair for a candidate.

    Stable across processes because hashlib.sha256 is not subject to
    Python's randomized hash().
    """
    probe_id = _probe_id_for(cand)
    seed_base = int.from_bytes(
        hashlib.sha256(probe_id.encode("utf-8")).digest()[:4],
        "big",
    )
    return probe_id, seed_base


def _phase2_aggregate(results: list[dict]) -> tuple[list, list]:
    """Sort Phase 2 result dicts by probe_id and partition into
    (admitted candidates, audit rows). Idempotent on already-sorted input.
    Used by both serial and process modes, and called once AFTER the
    borderline-rerun pass (Task 5) so post-rerun statuses drive the
    partition. Spec §6."""
    results.sort(key=lambda r: r["probe_id"])
    admitted = [r["candidate"] for r in results if r["status"] == "admitted"]
    audit_rows = [r["audit_row"] for r in results if r["audit_row"] is not None]
    return admitted, audit_rows


def _label_one_strong_advantage_candidate(
    cand: dict,
    *,
    label_ckpt_name: str,
    sims: int,
    repeats: int,
    magnitude_threshold: float,
    top1_share_floor: float,
    stability_cap: float,
) -> dict:
    """Phase 2 per-candidate labeling helper. Used by both the serial loop
    and the process-pool path.

    Returns a structured result dict. See spec §7.
    """
    from scripts.GPU.alphazero.probe_eval import (
        label_candidate_with_mcts,
        apply_admission_filter,
    )
    from scripts.GPU.alphazero.game.twixt_state import TwixtState

    cand = copy.deepcopy(cand)
    probe_id, seed_base = _probe_id_and_seed_base(cand)

    # Replay candidate moves into a TwixtState.
    try:
        state = TwixtState(active_size=24, to_move=cand["starting_player"])
        for r, c in cand["move_history"]:
            state = state.apply_move((r, c))
    except Exception as exc:
        return {
            "probe_id": probe_id,
            "status": "replay_error",
            "candidate": None,
            "audit_row": {
                "source_game": cand["source_game"],
                "source_ply": cand["source_ply"],
                "phase1_features": cand["phase1_features"],
                "reason": "replay_error",
            },
            "rejection_reason": "replay_error",
            "phase2_label": None,
            "error_message": f"{type(exc).__name__}: {exc}",
        }

    # Run MCTS labeling.
    try:
        label = label_candidate_with_mcts(
            state,
            sims=sims,
            repeats=repeats,
            rng_seed_base=seed_base,
        )
    except Exception as exc:
        return {
            "probe_id": probe_id,
            "status": "mcts_error",
            "candidate": cand,
            "audit_row": {
                "source_game": cand["source_game"],
                "source_ply": cand["source_ply"],
                "phase1_features": cand["phase1_features"],
                "reason": "mcts_error",
            },
            "rejection_reason": "mcts_error",
            "phase2_label": None,
            "error_message": f"{type(exc).__name__}: {exc}",
        }

    # Normalize STM perspective (red-perspective for downstream consumers).
    stm = _stm_at_ply(cand)
    if stm == "black":
        label["mean_root_value"] = -label["mean_root_value"]
        label["value_per_run"] = [-v for v in label["value_per_run"]]

    cand["phase2_label"] = label
    ok, reason = apply_admission_filter(
        cand,
        magnitude_threshold=magnitude_threshold,
        top1_share_floor=top1_share_floor,
        stability_cap=stability_cap,
    )
    cand["phase2_label"]["label_checkpoint"] = label_ckpt_name

    if ok:
        return {
            "probe_id": probe_id,
            "status": "admitted",
            "candidate": cand,
            "audit_row": None,
            "rejection_reason": None,
            "phase2_label": cand["phase2_label"],
            "error_message": None,
        }
    return {
        "probe_id": probe_id,
        "status": "rejected",
        "candidate": cand,
        "audit_row": {
            "source_game": cand["source_game"],
            "source_ply": cand["source_ply"],
            "phase1_features": cand["phase1_features"],
            "phase2_label": cand["phase2_label"],
            "reason": reason,
        },
        "rejection_reason": reason,
        "phase2_label": cand["phase2_label"],
        "error_message": None,
    }


_BORDERLINE_TRIGGERS = ("magnitude", "top1_share", "stability")


def _is_borderline(
    label: dict,
    *,
    epsilon: float,
    magnitude_threshold: float,
    top1_share_floor: float,
    stability_cap: float,
) -> list[str]:
    """Return the list of triggers (subset of _BORDERLINE_TRIGGERS) for which
    the label is within epsilon of the corresponding admission threshold.
    Empty list means not borderline. Spec §9.
    """
    triggers = []
    if abs(abs(label["mean_root_value"]) - magnitude_threshold) <= epsilon:
        triggers.append("magnitude")
    if abs(label["min_top1_share"] - top1_share_floor) <= epsilon:
        triggers.append("top1_share")
    if abs(label["value_stability"] - stability_cap) <= epsilon:
        triggers.append("stability")
    return triggers


def _run_borderline_reruns(
    results: list[dict],
    *,
    epsilon: float,
    magnitude_threshold: float,
    top1_share_floor: float,
    stability_cap: float,
    label_ckpt_name: str,
    sims: int,
    repeats: int,
) -> dict:
    """Re-label borderline candidates synchronously in the main process.
    Mutates `results` in place. Returns counters for instrumentation:
        {"candidates": N, "reruns": N, "flips": N, "seconds": float}

    Spec §9. Excludes replay_error / mcts_error rows (phase2_label is None).
    The rerun result replaces the parallel result and admission filter is
    re-applied once. Audit metadata records flips.
    """
    import time as _time
    counters = {"candidates": 0, "reruns": 0, "flips": 0, "seconds": 0.0}
    t0 = _time.time()
    for r in results:
        if r["phase2_label"] is None:
            continue  # replay/mcts errors carry no label
        triggers = _is_borderline(
            r["phase2_label"],
            epsilon=epsilon,
            magnitude_threshold=magnitude_threshold,
            top1_share_floor=top1_share_floor,
            stability_cap=stability_cap,
        )
        if not triggers:
            continue
        counters["candidates"] += 1
        # Re-execute the helper in the main process. Same seed/sims/cfg.
        # Use the candidate object that came back from the worker (already
        # deepcopy'd inside the helper, but we deepcopy again to keep the
        # pre-rerun label intact for audit metadata).
        cand = copy.deepcopy(r["candidate"])
        # Strip the prior phase2_label so the helper re-labels cleanly.
        cand.pop("phase2_label", None)
        rerun = _label_one_strong_advantage_candidate(
            cand,
            label_ckpt_name=label_ckpt_name,
            sims=sims,
            repeats=repeats,
            magnitude_threshold=magnitude_threshold,
            top1_share_floor=top1_share_floor,
            stability_cap=stability_cap,
        )
        # Spec §9: rerun must preserve probe identity. The helper
        # recomputes seed_base from sha256(probe_id) and builds the same
        # probe_id internally, so this should always hold. Guarded raise
        # (not assert) so the invariant survives `python -O`.
        if rerun["probe_id"] != r["probe_id"]:
            raise RuntimeError(
                f"borderline rerun changed probe_id: "
                f"{r['probe_id']} -> {rerun['probe_id']}"
            )
        counters["reruns"] += 1
        flipped = rerun["status"] != r["status"]
        if flipped:
            counters["flips"] += 1
            print(
                f"[probe_suite] borderline rerun flipped {r['probe_id']}: "
                f"{r['rejection_reason'] or 'admitted'} -> "
                f"{rerun['rejection_reason'] or 'admitted'}",
                file=sys.stderr,
            )

        rerun_audit_meta = {
            "borderline_rerun": True,
            "borderline_rerun_reason": triggers,
            "parallel_phase2_label_before_rerun": r["phase2_label"],
            "borderline_rerun_flipped": flipped,
        }
        if flipped:
            rerun_audit_meta["parallel_admission_reason"] = (
                r["rejection_reason"] or "admitted"
            )
            rerun_audit_meta["serial_rerun_admission_reason"] = (
                rerun["rejection_reason"] or "admitted"
            )

        # Attach to the candidate so any audit row built downstream
        # (selector audit) merges these fields.
        if rerun["candidate"] is not None:
            rerun["candidate"]["_borderline_rerun_audit"] = rerun_audit_meta

        # Replace the parallel result with the rerun result. Audit row, if
        # any, also takes the rerun's content with the rerun metadata merged.
        if rerun["audit_row"] is not None:
            rerun["audit_row"].update(rerun_audit_meta)
        # Update the result entry in-place.
        r.update(rerun)

    counters["seconds"] = _time.time() - t0
    return counters


def _merge_borderline_audit(audit_row: dict, cand: dict) -> dict:
    """If cand has _borderline_rerun_audit, merge those keys into audit_row.
    Returns the (potentially mutated) audit_row. Spec §9.
    """
    rerun_meta = cand.get("_borderline_rerun_audit")
    if rerun_meta:
        audit_row.update(rerun_meta)
    return audit_row


def _print_phase2_progress(idx: int, n_total: int, n_admitted: int,
                           t_start: float) -> None:
    import time as _time
    elapsed = _time.time() - t_start
    if idx > 0:
        rate = idx / elapsed
        eta_s = (n_total - idx) / rate if rate > 0 else 0.0
        eta_str = f"ETA {eta_s/60:.1f}m" if eta_s < 3600 else f"ETA {eta_s/3600:.1f}h"
    else:
        eta_str = "ETA --"
    print(
        f"[probe_suite] Phase 2: {idx}/{n_total} labeled "
        f"({n_admitted} admitted, {elapsed:.0f}s elapsed, {eta_str})",
        flush=True,
    )


def _run_strong_advantage(args) -> int:
    if args.out is None:
        args.out = "tests/probes/strong_advantage_probes.json"

    if args.promote:
        return _run_promote(args)

    if args.label_checkpoint is None:
        print("[probe_suite] ERROR: --label-checkpoint required for "
              "--tier strong_advantage (when not --promote).", file=sys.stderr)
        return 2
    if args.source_iter_range is None:
        print("[probe_suite] ERROR: --source-iter-range required for "
              "--tier strong_advantage.", file=sys.stderr)
        return 2

    label_ckpt = Path(args.label_checkpoint)
    if not label_ckpt.exists():
        print(f"[probe_suite] ERROR: --label-checkpoint not found: {label_ckpt}",
              file=sys.stderr)
        return 2

    out_path = Path(args.out)
    draft_path = out_path.with_suffix(".draft.json")
    audit_path = out_path.parent / "candidates_strong_advantage.json"
    if draft_path.exists() and not args.force:
        print(f"[probe_suite] ERROR: draft already exists: {draft_path}\n"
              f"  Pass --force to overwrite, or delete the existing draft.",
              file=sys.stderr)
        return 2

    from scripts.GPU.alphazero.probe_eval import (
        extract_strong_advantage_candidates,
        _set_default_labeler_network,
        _set_default_labeler_mcts_config,
        load_network_for_scoring,
    )
    from scripts.GPU.alphazero.mcts import MCTSConfig

    # Phase 1: load games, mine candidates.
    min_iter, max_iter = args.source_iter_range
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"[probe_suite] ERROR: --input not a directory: {input_dir}",
              file=sys.stderr)
        return 2
    games = []
    for fp in sorted(input_dir.glob("iter_*_game_*.json")):
        with open(fp) as f:
            try:
                g = json.load(f)
            except json.JSONDecodeError:
                continue
        iteration = (g.get("meta") or {}).get("iteration")
        if iteration is None or not (min_iter <= iteration <= max_iter):
            continue
        g["source_game"] = fp.stem
        games.append(g)

    candidates, audit = extract_strong_advantage_candidates(games)
    print(f"[probe_suite] Phase 1: {len(candidates)} candidates from "
          f"{len(games)} games")

    # Phase 2: load network, label each candidate, apply admission filter.
    # IMPORTANT: this generator currently supports ONLY labeling checkpoints
    # built with create_network defaults (hidden=128, n_blocks=6).
    # load_network_for_scoring auto-detects input channels (24 vs 30) but
    # does NOT auto-detect hidden/n_blocks. To label against a checkpoint
    # with a different architecture, this generator must first be extended
    # with --hidden/--blocks flags (follow-up); the call below will
    # otherwise raise a tensor-shape mismatch and abort the run.
    #
    # In process mode the main process loads the network ONLY IFF the
    # borderline-rerun pass will fire (epsilon > 0 AND --no-borderline-rerun
    # is off). Otherwise workers load their own copies via
    # _init_label_worker and the main process needs no MLX context. Spec §9.
    mcts_cfg_payload = {
        "eval_batch_size": args.mcts_eval_batch_size,
        "stall_flush_sims": args.mcts_stall_flush_sims,
    }
    rerun_enabled = (
        args.label_worker_mode == "process"
        and args.admission_borderline_epsilon > 0
        and not args.no_borderline_rerun
    )
    need_main_process_labeler = (
        args.label_worker_mode == "serial" or rerun_enabled
    )
    if need_main_process_labeler:
        try:
            network, _ic, _h, _nb = load_network_for_scoring(str(label_ckpt))
            network.eval()
        except Exception as exc:
            print(f"[probe_suite] ERROR: failed to load main-process labeler "
                  f"network from checkpoint {label_ckpt}\n"
                  f"  mode={args.label_worker_mode} "
                  f"rerun_enabled={rerun_enabled}\n"
                  f"  {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        _set_default_labeler_network(network)
        _set_default_labeler_mcts_config(MCTSConfig(**mcts_cfg_payload))

    admitted = []
    import time as _time
    n_total = len(candidates)
    # Cadence: at small batches, every candidate; at big batches, every 5%.
    progress_every = max(1, n_total // 20)
    t_phase2_start = _time.time()

    helper_kwargs = dict(
        label_ckpt_name=label_ckpt.name,
        sims=args.label_mcts_sims,
        repeats=args.label_mcts_repeats,
        magnitude_threshold=args.magnitude_threshold,
        top1_share_floor=args.top1_share_floor,
        stability_cap=args.stability_cap,
    )

    if args.label_worker_mode == "serial":
        results = []
        admitted_so_far = 0
        for idx, cand in enumerate(candidates):
            if idx % progress_every == 0:
                _print_phase2_progress(idx, n_total, admitted_so_far,
                                       t_phase2_start)
            r = _label_one_strong_advantage_candidate(cand, **helper_kwargs)
            results.append(r)
            if r["status"] == "admitted":
                admitted_so_far += 1
    else:  # "process"
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = _mp.get_context("spawn")
        # NOTE: the bare name `_init_label_worker` is resolved through the
        # module globals of `scripts.build_probe_suite` at this exact moment,
        # so tests can monkeypatch `bps._init_label_worker` to a top-level
        # test helper and have ProcessPoolExecutor pickle the patched
        # function (by qualified name) for the worker. Do NOT replace the
        # bare name with `from scripts.build_probe_suite import
        # _init_label_worker`; that would freeze the binding and break the
        # smoke test in tests/test_probe_phase2_parallel.py.
        try:
            with ProcessPoolExecutor(
                max_workers=args.label_workers,
                mp_context=ctx,
                initializer=_init_label_worker,
                initargs=(str(label_ckpt), mcts_cfg_payload),
            ) as pool:
                futures = [
                    pool.submit(_label_one_strong_advantage_candidate, cand,
                                **helper_kwargs)
                    for cand in candidates
                ]
                results = []
                completed = 0
                for fut in as_completed(futures):
                    results.append(fut.result())
                    completed += 1
                    if completed % progress_every == 0:
                        _print_phase2_progress(completed, n_total,
                                               sum(1 for r in results
                                                   if r["status"] == "admitted"),
                                               t_phase2_start)
        except Exception as exc:
            print(f"[probe_suite] ERROR: failed to initialize process label "
                  f"worker from checkpoint {label_ckpt}\n"
                  f"  mode={args.label_worker_mode} workers={args.label_workers}\n"
                  f"  {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    # Borderline serial-rerun pass (Task 5 / spec §9). In process mode
    # with epsilon > 0 and --no-borderline-rerun off, candidates whose
    # parallel-mode label is within epsilon of any admission threshold
    # are re-labeled synchronously in the main process; the rerun result
    # replaces the parallel result. For serial / disabled paths the
    # counters are zero-filled. `rerun_enabled` and `rerun_counters` are
    # consumed by Task 6's instrumentation (meta.phase2_run_stats).
    if rerun_enabled:
        rerun_counters = _run_borderline_reruns(
            results,
            epsilon=args.admission_borderline_epsilon,
            magnitude_threshold=args.magnitude_threshold,
            top1_share_floor=args.top1_share_floor,
            stability_cap=args.stability_cap,
            label_ckpt_name=label_ckpt.name,
            sims=args.label_mcts_sims,
            repeats=args.label_mcts_repeats,
        )
    else:
        rerun_counters = {"candidates": 0, "reruns": 0, "flips": 0,
                          "seconds": 0.0}

    # _phase2_aggregate sorts AFTER reruns so post-rerun statuses drive
    # the admitted/audit partition. Spec §6.
    admitted, new_audit_rows = _phase2_aggregate(results)
    audit.extend(new_audit_rows)
    for r in results:
        if r["status"] in ("replay_error", "mcts_error"):
            ar = r["audit_row"]
            kind = "state replay error" if r["status"] == "replay_error" else "MCTS error"
            print(f"[probe_suite] WARN: {kind} on "
                  f"{ar['source_game']} ply {ar['source_ply']}: "
                  f"{r['error_message']}", file=sys.stderr)

    # Final Phase 2 summary so the operator sees a clean breakdown.
    # NOTE: at this point in the pipeline, audit contains only Phase-2
    # rejection rows (the diversity selector hasn't run yet, so no
    # reason="admitted" or reason="diversity_*" rows exist). We
    # explicitly seed the breakdown with the Phase-2-admitted count
    # (`len(admitted)`) so the operator sees the full picture.
    phase2_elapsed = _time.time() - t_phase2_start
    from collections import Counter as _Counter
    reason_breakdown = _Counter(
        a["reason"] for a in audit if "phase2_label" in a or a["reason"] in
        ("mcts_error", "replay_error",
         "sign_mismatch", "magnitude_below_threshold", "low_top1_share",
         "unstable_value", "position_already_forced")
    )
    reason_breakdown["admitted"] = len(admitted)
    breakdown_str = ", ".join(f"{r}={n}" for r, n in reason_breakdown.most_common())
    print(
        f"[probe_suite] Phase 2 complete: {n_total}/{n_total} labeled "
        f"({len(admitted)} admitted, {phase2_elapsed:.0f}s total)\n"
        f"  Per-reason: {breakdown_str}",
        flush=True,
    )

    if not admitted:
        # No reason="admitted" rows can exist in audit at this point
        # (Phase 2 only writes rejections; the selector hasn't run).
        from collections import Counter
        reason_counts = Counter(a["reason"] for a in audit)
        msg = ", ".join(f"{r}: {n}" for r, n in reason_counts.most_common())
        print(f"[probe_suite] ERROR: 0 admitted probes overall.\n"
              f"  Drop reasons: {msg}", file=sys.stderr)
        return 1

    admitted = _select_diverse_admitted_candidates(
        admitted,
        audit,
        max_probes=args.max_probes,
        max_probes_per_game=args.max_probes_per_game,
    )

    probes_out = []
    for cand in admitted:
        probes_out.append({
            "id": _probe_id_for(cand),
            "category": cand["category"],
            "confidence": "strong_advantage",
            "side_to_move": _stm_at_ply(cand),
            "expected_value_sign": 1 if cand["winner"] == "red" else -1,
            "active_size": 24,
            "ply": cand["ply"],
            "move_history": cand["move_history"],
            "source_game": cand["source_game"],
            "source_ply": cand["source_ply"],
            "starting_player": cand["starting_player"],
            "phase1_features": cand["phase1_features"],
            "phase2_label": cand["phase2_label"],
        })

    import hashlib
    ckpt_hash = hashlib.sha256(label_ckpt.read_bytes()).hexdigest()
    payload = {
        "meta": {
            "type": "bootstrap_rule_selected",
            "tier": "strong_advantage",
            "not_gate_suite": True,
            "review_mode": "draft",
            "reviewer": None,
            "reviewed_at_utc": None,
            "generator": "scripts/build_probe_suite.py",
            "generator_version": 1,
            "selection_rules": {
                "board_size": 24,
                "winner_reasons": ["win"],
                "k_plies_from_terminal_range": [3, 8],
                "phase1_thresholds": {
                    "min_cc_size": 10,
                    "min_cc_axis_span": 0.55,
                    "min_axis_span_margin": 0.10,
                    "require_cc_touches_own_goal": True,
                    "exclude_forced_within_2": True,
                },
                "phase2_thresholds": {
                    "label_mcts_sims": args.label_mcts_sims,
                    "label_mcts_repeats": args.label_mcts_repeats,
                    "min_magnitude": args.magnitude_threshold,
                    "min_top1_share": args.top1_share_floor,
                    "max_value_stability": args.stability_cap,
                    "require_sign_match_source_winner": True,
                },
                "label_checkpoint": str(label_ckpt),
                "label_checkpoint_sha256": ckpt_hash,
                "source_iter_range": [min_iter, max_iter],
                "dedup": "exact + 4-form-mirror-canonical",
                "category_min_count": 5,
                "max_probes_per_game": args.max_probes_per_game,
                "min_ply_separation_same_game": MIN_PLY_SEPARATION_SAME_GAME,
                "category_iteration_order": list(CATEGORY_ITERATION_ORDER),
                "diversity_quality_key_order": [
                    "phase1_features.cc_size desc",
                    "phase1_features.axis_span_margin desc",
                    "phase1_features.cc_axis_span desc",
                    "phase2_label.min_top1_share desc",
                    "phase2_label.value_stability asc",
                    "default_sort_key (-iter, -source_ply, source_game)",
                ],
            },
        },
        "probes": probes_out,
    }

    draft_path.parent.mkdir(parents=True, exist_ok=True)
    with open(draft_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
    with open(audit_path, "w") as f:
        json.dump({"audit": audit}, f, indent=2, sort_keys=False)
        f.write("\n")

    print(f"[probe_suite] wrote {len(probes_out)} candidates to draft "
          f"{draft_path}\n  audit: {audit_path}\n"
          f"  Next: review the draft, then run --promote --reviewer NAME "
          f"(lands in Task 2.5b).")
    return 0


def _run_promote(args) -> int:
    """Promote a *.draft.json to the committed file.

    Stamps meta.review_mode="light_review", meta.reviewer, and
    meta.reviewed_at_utc. Refuses to overwrite an existing committed
    file unless --force is passed.
    """
    if not args.reviewer:
        print("[probe_suite] ERROR: --reviewer required with --promote",
              file=sys.stderr)
        return 2
    out_path = Path(args.out)
    draft_path = out_path.with_suffix(".draft.json")
    if not draft_path.exists():
        print(f"[probe_suite] ERROR: no draft to promote at {draft_path}",
              file=sys.stderr)
        return 2
    if out_path.exists() and not args.force:
        print(f"[probe_suite] ERROR: committed file exists: {out_path}\n"
              f"  Pass --force to overwrite (deliberate re-promotion).",
              file=sys.stderr)
        return 2

    payload = json.loads(draft_path.read_text())
    payload["meta"]["review_mode"] = "light_review"
    payload["meta"]["reviewer"] = args.reviewer
    # ISO 8601 UTC with explicit Z suffix; matches probe_eval.py convention
    # (datetime.utcnow() is deprecated on Python 3.14+).
    payload["meta"]["reviewed_at_utc"] = (
        datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"[probe_suite] promoted {draft_path} -> {out_path} "
          f"(reviewer={args.reviewer})")
    return 0


def _probe_id_for(cand: dict) -> str:
    """Deterministic probe ID: iter_NNNN_game_MMM_plyNNN_<category>."""
    return (
        f"{cand['source_game']}_ply{cand['source_ply']:03d}_{cand['category']}"
    )


def _stm_at_ply(cand: dict) -> str:
    """Whose turn it is at the candidate position (the side ABOUT to move)."""
    plies_played = cand["source_ply"]
    starting = cand["starting_player"]
    if plies_played % 2 == 0:
        return starting
    return "black" if starting == "red" else "red"


def main_with_args(argv: list) -> int:
    """Test entrypoint: invokes main() with explicit args (sys.argv-style)."""
    saved = sys.argv
    sys.argv = ["build_probe_suite.py", *argv]
    try:
        return main() or 0
    finally:
        sys.argv = saved


if __name__ == "__main__":
    sys.exit(main() or 0)
