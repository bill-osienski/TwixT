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
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


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


# --- Tier dispatch ---

def main() -> int:
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

    args = ap.parse_args()

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
        label_candidate_with_mcts,
        apply_admission_filter,
        _set_default_labeler_network,
        load_network_for_scoring,
    )
    from scripts.GPU.alphazero.game.twixt_state import TwixtState

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
    network, _ic, _h, _nb = load_network_for_scoring(str(label_ckpt))
    network.eval()
    _set_default_labeler_network(network)

    admitted = []
    import hashlib
    import time as _time
    n_total = len(candidates)
    # Cadence: at small batches, every candidate; at big batches, every 5%.
    progress_every = max(1, n_total // 20)
    t_phase2_start = _time.time()
    for idx, cand in enumerate(candidates):
        if idx % progress_every == 0:
            elapsed = _time.time() - t_phase2_start
            n_admitted = len(admitted)
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

        try:
            state = TwixtState(active_size=24, to_move=cand["starting_player"])
            for r, c in cand["move_history"]:
                state = state.apply_move((r, c))
        except Exception as exc:
            print(f"[probe_suite] WARN: state replay error on "
                  f"{cand['source_game']} ply {cand['source_ply']}: {exc}",
                  file=sys.stderr)
            audit.append({
                "source_game": cand["source_game"],
                "source_ply": cand["source_ply"],
                "phase1_features": cand["phase1_features"],
                "reason": "replay_error",
            })
            continue

        # Stable seed: SHA-256 of probe ID, first 4 bytes as big-endian int.
        # Python's built-in hash() is process-randomized and would break
        # byte-reproducibility across runs.
        seed_base = int.from_bytes(
            hashlib.sha256(_probe_id_for(cand).encode("utf-8")).digest()[:4],
            "big",
        )

        try:
            label = label_candidate_with_mcts(
                state,
                sims=args.label_mcts_sims,
                repeats=args.label_mcts_repeats,
                rng_seed_base=seed_base,
            )
        except Exception as exc:
            print(f"[probe_suite] WARN: MCTS error on {cand['source_game']} "
                  f"ply {cand['source_ply']}: {exc}", file=sys.stderr)
            audit.append({
                "source_game": cand["source_game"],
                "source_ply": cand["source_ply"],
                "phase1_features": cand["phase1_features"],
                "reason": "mcts_error",
            })
            continue

        # Normalize labeler output from STM-perspective to red-perspective
        # before storing into phase2_label. apply_admission_filter (and
        # everything downstream that compares against expected_value_sign)
        # operates in red-perspective. The candidate's STM at this ply is
        # `_stm_at_ply(cand)`; if black, negate the value fields.
        stm = _stm_at_ply(cand)
        if stm == "black":
            label["mean_root_value"] = -label["mean_root_value"]
            label["value_per_run"] = [-v for v in label["value_per_run"]]
            # value_stability is max-min, sign-invariant — leave as-is.
            # min_top1_share is a probability — sign-invariant — leave as-is.

        cand["phase2_label"] = label
        ok, reason = apply_admission_filter(
            cand,
            magnitude_threshold=args.magnitude_threshold,
            top1_share_floor=args.top1_share_floor,
            stability_cap=args.stability_cap,
        )
        cand["phase2_label"]["label_checkpoint"] = label_ckpt.name
        audit.append({
            "source_game": cand["source_game"],
            "source_ply": cand["source_ply"],
            "phase1_features": cand["phase1_features"],
            "phase2_label": cand["phase2_label"],
            "reason": reason,
        })
        if ok:
            admitted.append(cand)

    # Final Phase 2 summary so the operator sees a clean breakdown.
    phase2_elapsed = _time.time() - t_phase2_start
    from collections import Counter as _Counter
    reason_breakdown = _Counter(
        a["reason"] for a in audit if "phase2_label" in a or a["reason"] in
        ("mcts_error", "replay_error", "admitted",
         "sign_mismatch", "magnitude_below_threshold", "low_top1_share",
         "unstable_value", "position_already_forced")
    )
    breakdown_str = ", ".join(f"{r}={n}" for r, n in reason_breakdown.most_common())
    print(
        f"[probe_suite] Phase 2 complete: {n_total}/{n_total} labeled "
        f"({len(admitted)} admitted, {phase2_elapsed:.0f}s total)\n"
        f"  Per-reason: {breakdown_str}",
        flush=True,
    )

    if not admitted:
        from collections import Counter
        reason_counts = Counter(a["reason"] for a in audit if a["reason"] != "admitted")
        msg = ", ".join(f"{r}: {n}" for r, n in reason_counts.most_common())
        print(f"[probe_suite] ERROR: 0 admitted probes overall.\n"
              f"  Drop reasons: {msg}", file=sys.stderr)
        return 1

    admitted = admitted[: args.max_probes]

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
