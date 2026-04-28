"""Probe evaluator — run the curated probe suite against a checkpoint.

Produces per-probe CSV + aggregate JSON. Supports both 24-channel (iter-999
and earlier) and 30-channel (post-retrain) networks via auto-detection of
the checkpoint's first-conv-layer input channel count.

Formal runs require an explicit --weights path. Interactive "latest
checkpoint" convenience mode prints the resolved path before proceeding.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone

import numpy as np
import mlx.core as mx

from .game.twixt_state import TwixtState
from .mcts import MCTS, MCTSConfig
from .local_evaluator import LocalGPUEvaluator


def _detect_input_channels(weights_path: str) -> int:
    """Inspect a safetensors checkpoint to learn the first conv layer's input channels.

    Uses MLX's loader (no safetensors package dependency). The encoder's first
    conv weight is stored as (out_channels, kH, kW, in_channels) in MLX NHWC.
    """
    weights = mx.load(weights_path)
    # Canonical first-conv key for this architecture
    key = "encoder.conv1.weight"
    if key not in weights:
        raise RuntimeError(
            f"Could not find {key} in {weights_path}; cannot auto-detect input channels."
        )
    shape = weights[key].shape
    if len(shape) != 4:
        raise RuntimeError(
            f"Expected 4D conv weight at {key}, got shape {shape}."
        )
    # MLX NHWC: (out_channels, kH, kW, in_channels)
    return shape[-1]


def _load_network(
    weights_path: str,
    hidden: int | None = None,
    n_blocks: int | None = None,
    verbose: bool = True,
):
    """Load a network, auto-detecting input channel format.

    Dual-format contract: supports both pre-Phase-2 24-channel checkpoints
    (iter-0999 and earlier) and post-Phase-2 30-channel checkpoints via the
    `in_channels` parameter on create_network.

    Architecture (hidden / n_blocks): defaults inherited from
    `create_network()` — currently 128 and 6 respectively. These ARE the
    locked canonical training architecture for both 24ch and 30ch runs
    (see Task 15 trainer defaults). Override via `--hidden` / `--n-blocks`
    if a checkpoint from a future architecture variant needs to be evaluated.
    Both values are also echoed into the aggregate JSON metadata so any
    mismatch is visible in artifacts.
    """
    from .network import create_network

    in_channels = _detect_input_channels(weights_path)
    if verbose:
        print(f"[probe_eval] Detected {in_channels}-channel checkpoint at {weights_path}")
    if in_channels not in (24, 30):
        raise RuntimeError(
            f"Unsupported channel count {in_channels} in {weights_path}. "
            f"Expected 24 (pre-Phase-2) or 30 (post-Phase-2)."
        )
    # Instantiate using create_network defaults unless caller overrides.
    # Do NOT duplicate the hidden=128, n_blocks=6 literals here — a future
    # architecture bump should change the default in one place (create_network)
    # and have it flow through to all evaluators.
    kwargs = {"in_channels": in_channels}
    if hidden is not None:
        kwargs["hidden"] = hidden
    if n_blocks is not None:
        kwargs["n_blocks"] = n_blocks
    net = create_network(**kwargs)
    # Resolve the actual hidden / n_blocks used so we can echo them in metadata
    actual_hidden = hidden if hidden is not None else _default_create_network_param("hidden")
    actual_n_blocks = n_blocks if n_blocks is not None else _default_create_network_param("n_blocks")
    if verbose:
        print(f"[probe_eval] Network architecture: hidden={actual_hidden}, n_blocks={actual_n_blocks}")
    net.load_weights(weights_path)
    return net, in_channels, actual_hidden, actual_n_blocks


def load_network_for_scoring(weights_path: str, verbose: bool = False):
    """Public wrapper over _load_network.

    Provides a stable import symbol for the trainer, analyzer, and bootstrap
    generator to share. Returns (network, in_channels, hidden, n_blocks) with
    auto-detection of 24-channel vs 30-channel checkpoints.

    See _load_network for full docstring.
    """
    return _load_network(weights_path, hidden=None, n_blocks=None, verbose=verbose)


def _default_create_network_param(name: str):
    """Introspect create_network's default for a named param. Used to echo
    the actual architecture in metadata when the user didn't override."""
    import inspect
    from .network import create_network
    sig = inspect.signature(create_network)
    return sig.parameters[name].default


def _replay_probe(probe: dict) -> TwixtState:
    """Replay a probe's move_history from an empty state.

    The initial `to_move` is taken from the probe's `starting_player` field
    (defaults to "red" for probes predating the schema update — see the
    bootstrap suite generator and extract_forced_probes_from_games, which
    now emit this field explicitly). Side-to-move alternates with each
    applied move, so mismatched starting_player produces illegal-move
    errors — we raise loudly with the probe id rather than mask.
    """
    starting_player = probe.get("starting_player", "red")
    state = TwixtState(active_size=probe["active_size"], to_move=starting_player)
    for ply_idx, move in enumerate(probe["move_history"]):
        r, c = int(move[0]), int(move[1])
        try:
            state = state.apply_move((r, c))
        except ValueError as e:
            raise ValueError(
                f"probe {probe.get('id', '<unknown>')!r} failed to replay at "
                f"ply {ply_idx} move=({r}, {c}) starting_player={starting_player!r}: {e}"
            ) from e
    return state


def _eval_probe(probe: dict, evaluator: LocalGPUEvaluator, sims: int) -> dict:
    """Evaluate one probe: get NN value + run MCTS with `sims` sims.

    The evaluator builds the input tensor in the correct 24ch / 30ch format
    based on its network's `in_channels`.
    """
    state = _replay_probe(probe)

    # NN-only value: single forward pass from the state's side-to-move perspective.
    # Tensor format is selected by the evaluator to match the network's input.
    tensor = evaluator.build_input_tensor(state)  # (C, H, W)
    tensor = np.transpose(tensor, (1, 2, 0))  # (H, W, C)
    boards_np = np.expand_dims(tensor.astype(np.float32), axis=0)
    moves = state.legal_moves()
    move_rows_np = np.array([[m[0] for m in moves]], dtype=np.int32)
    move_cols_np = np.array([[m[1] for m in moves]], dtype=np.int32)
    move_mask_np = np.ones((1, len(moves)), dtype=np.float32)
    priors_np, values_np = evaluator.infer(
        boards_np, move_rows_np, move_cols_np, move_mask_np, state.active_size
    )
    nn_value = float(values_np[0])  # From side_to_move perspective

    # MCTS: run sims; root_value also from side_to_move perspective
    mcts_root_value = None
    mcts_top_move = None
    mcts_top_share = None
    if sims > 0:
        cfg = MCTSConfig(n_simulations=sims)
        mcts = MCTS(evaluator, cfg, rng=random.Random(42))
        visit_counts, root_value = mcts.search(state, add_noise=False)
        mcts_root_value = float(root_value)
        if visit_counts:
            top = max(visit_counts.items(), key=lambda kv: kv[1])
            mcts_top_move = list(top[0])
            total = sum(visit_counts.values())
            mcts_top_share = top[1] / total if total > 0 else 0.0

    # Convert to red-perspective for consistency (spec convention)
    if state.to_move == "black":
        nn_value = -nn_value
        if mcts_root_value is not None:
            mcts_root_value = -mcts_root_value

    # Score against expected
    exp_sign = probe.get("expected_value_sign", 0)
    sign_correct_nn = int((exp_sign > 0 and nn_value > 0) or
                          (exp_sign < 0 and nn_value < 0) or
                          (exp_sign == 0 and abs(nn_value) < 0.1))
    sign_correct_mcts = 0
    if mcts_root_value is not None:
        sign_correct_mcts = int((exp_sign > 0 and mcts_root_value > 0) or
                                (exp_sign < 0 and mcts_root_value < 0) or
                                (exp_sign == 0 and abs(mcts_root_value) < 0.1))

    # Magnitude checks
    min_mag = probe.get("expected_value_min")
    max_mag = probe.get("expected_value_max")
    mag_ok = True
    if min_mag is not None:
        mag_ok = mag_ok and abs(nn_value) >= min_mag
    if max_mag is not None:
        mag_ok = mag_ok and abs(nn_value) <= max_mag

    # Search-corrected / both-wrong flags
    search_corrected = int(sign_correct_mcts == 1 and sign_correct_nn == 0)
    both_wrong = int(sign_correct_mcts == 0 and sign_correct_nn == 0)

    return {
        "probe_id": probe["id"],
        "category": probe["category"],
        "confidence": probe["confidence"],
        "expected_value_sign": exp_sign,
        "nn_value": round(nn_value, 4),
        "mcts_root_value": round(mcts_root_value, 4) if mcts_root_value is not None else None,
        "mcts_top_move": mcts_top_move,
        "mcts_top_share": round(mcts_top_share, 4) if mcts_top_share is not None else None,
        "sign_correct_nn": sign_correct_nn,
        "sign_correct_mcts": sign_correct_mcts,
        "nn_magnitude": round(abs(nn_value), 4),
        "magnitude_in_band": int(mag_ok),
        "search_corrected": search_corrected,
        "both_wrong": both_wrong,
    }


def run_forced_probes_inline(
    network,
    probes: list[dict],
    active_size: int | None = None,
) -> dict:
    """Evaluate forced-tier probes against an in-memory network (NN-only, no MCTS).

    Designed for trainer-side per-iteration invocation — no disk I/O, no
    process spawning. Uses the same NN-forward path as _eval_probe but skips
    MCTS entirely (the `sims=0` branch).

    Args:
        network: AlphaZeroNetwork instance (already initialized / trained)
        probes: list of probe dicts (pre-filtered to confidence=='forced')
        active_size: if set, only evaluate probes whose active_size matches
            (skip probes that don't apply to the current curriculum size)

    Returns:
        Dict with aggregates:
          - n (int): probes evaluated (post active_size filter)
          - n_skipped_size (int): probes skipped because active_size mismatch
          - sign_correct (int): count of NN-correct probes
          - sign_correct_pct (float): sign_correct / n, or None if n==0
          - median_abs_v (float): median of |nn_value|, or None if n==0
          - nn_values (list[float]): per-probe NN values (red-perspective)
          - expected_signs (list[int]): per-probe expected sign
    """
    # Filter by active_size if requested
    applicable = probes
    n_skipped_size = 0
    if active_size is not None:
        applicable = [p for p in probes if p.get("active_size") == active_size]
        n_skipped_size = len(probes) - len(applicable)

    if not applicable:
        return {
            "n": 0,
            "n_skipped_size": n_skipped_size,
            "sign_correct": 0,
            "sign_correct_pct": None,
            "median_abs_v": None,
            "nn_values": [],
            "expected_signs": [],
        }

    evaluator = LocalGPUEvaluator(network)
    nn_values: list[float] = []
    expected_signs: list[int] = []
    sign_correct = 0

    for probe in applicable:
        state = _replay_probe(probe)
        tensor = evaluator.build_input_tensor(state)
        tensor = np.transpose(tensor, (1, 2, 0))
        boards_np = np.expand_dims(tensor.astype(np.float32), axis=0)
        moves = state.legal_moves()
        move_rows_np = np.array([[m[0] for m in moves]], dtype=np.int32)
        move_cols_np = np.array([[m[1] for m in moves]], dtype=np.int32)
        move_mask_np = np.ones((1, len(moves)), dtype=np.float32)
        _, values_np = evaluator.infer(
            boards_np, move_rows_np, move_cols_np, move_mask_np, state.active_size
        )
        nn_value = float(values_np[0])
        # Red-perspective convention (matches _eval_probe)
        if state.to_move == "black":
            nn_value = -nn_value
        nn_values.append(nn_value)

        exp_sign = probe.get("expected_value_sign", 0)
        expected_signs.append(exp_sign)
        if (exp_sign > 0 and nn_value > 0) or \
           (exp_sign < 0 and nn_value < 0) or \
           (exp_sign == 0 and abs(nn_value) < 0.1):
            sign_correct += 1

    n = len(applicable)
    abs_values = sorted(abs(v) for v in nn_values)
    median_abs_v = abs_values[n // 2] if n else None
    # Use simple midpoint (not interpolated median) — matches probe_eval CSV convention
    if n >= 2 and n % 2 == 0:
        median_abs_v = 0.5 * (abs_values[n // 2 - 1] + abs_values[n // 2])

    return {
        "n": n,
        "n_skipped_size": n_skipped_size,
        "sign_correct": sign_correct,
        "sign_correct_pct": round(sign_correct / n, 4) if n else None,
        "median_abs_v": round(median_abs_v, 4) if median_abs_v is not None else None,
        "nn_values": [round(v, 4) for v in nn_values],
        "expected_signs": expected_signs,
    }


def extract_forced_probes_from_games(
    games: list[dict],
    active_size: int = 24,
    k_plies: int = 2,
    winner_reasons: frozenset = frozenset({"win"}),
    dedupe_exact: bool = True,
    dedupe_mirror: bool = True,
    max_probes: int | None = None,
) -> list[dict]:
    """Extract near-terminal forced probes from parsed game JSONs.

    See spec §4.1 for full semantics. This is the shared primitive used by
    both the analyzer (replay-derived probe scoring) and the bootstrap
    probe suite generator.
    """
    probes: list[dict] = []

    for game in games:
        meta = game.get("meta") or {}
        if meta.get("board_size") != active_size:
            continue
        if meta.get("reason") not in winner_reasons:
            continue
        winner = game.get("winner")
        if winner not in ("red", "black"):
            continue

        moves_list = game.get("moves") or []
        n_moves = len(moves_list)
        if n_moves < k_plies + 1:
            continue

        # Build move_history as list of [r, c] pairs (ply-ordered).
        # Support both on-disk canonical schema (separate row/col fields; see
        # scripts/GPU/replay/format.py::Move) and legacy tuple schema
        # (nested `move: [r, c]`) used by some test fixtures.
        def _move_rc(m: dict) -> list[int]:
            if "row" in m and "col" in m:
                return [int(m["row"]), int(m["col"])]
            return list(m["move"])
        move_history = [_move_rc(m) for m in moves_list]
        source_game_basename = game.get("id") or f"iter_{meta.get('iteration', 0):04d}_game_{meta.get('game_idx', 0):03d}"
        source_iteration = meta.get("iteration", 0)
        category = f"near_win_{winner}"

        # Perspective convention (shared with run_forced_probes_inline and
        # _eval_probe): expected_value_sign is stored in RED-PERSPECTIVE.
        #   +1 = red wins from red's point of view
        #   -1 = black wins from red's point of view
        # The scoring code converts raw nn_value to red-perspective by negating
        # it when state.to_move == "black" (probe_eval.py:264-266), so this
        # comparison is apples-to-apples.
        expected_value_sign = +1 if winner == "red" else -1
        starting_player = game.get("starting_player") or "red"

        # Emit K probes at plies n_moves-1, n_moves-2, ..., n_moves-k_plies (down to 0).
        for k in range(1, k_plies + 1):
            ply = n_moves - k
            if ply < 0:
                continue
            # Side-to-move at this ply: starting_player if ply%2==0 else the other.
            side_to_move = starting_player if ply % 2 == 0 else ("black" if starting_player == "red" else "red")
            probe = {
                "id": f"{source_game_basename}_ply{ply:03d}_{winner}",
                "category": category,
                "confidence": "forced",
                "side_to_move": side_to_move,
                "expected_value_sign": expected_value_sign,
                "active_size": active_size,
                "ply": ply,
                "move_history": move_history[:ply],
                "source_game": source_game_basename,
                "source_ply": ply,
                # starting_player required by _replay_probe to correctly
                # initialize TwixtState.to_move — games that began with
                # black would otherwise fail replay when re-constructed
                # with the default red-starts assumption.
                "starting_player": starting_player,
                "_source_iteration": source_iteration,  # sort-only; stripped before return
            }
            probes.append(probe)

    # Dedup: exact + 4-form mirror canonical (spec §4.1).
    if dedupe_exact or dedupe_mirror:
        seen_keys: set = set()
        deduped: list[dict] = []

        def _canon_key(move_history: list[list[int]], N: int, use_mirror: bool) -> tuple:
            # Always include the identity form.
            forms = [tuple(tuple(m) for m in move_history)]
            if use_mirror:
                # Horizontal: (r, c) → (r, N-1-c)
                forms.append(tuple((r, N - 1 - c) for (r, c) in move_history))
                # Vertical: (r, c) → (N-1-r, c)
                forms.append(tuple((N - 1 - r, c) for (r, c) in move_history))
                # 180°: (r, c) → (N-1-r, N-1-c)
                forms.append(tuple((N - 1 - r, N - 1 - c) for (r, c) in move_history))
            return min(forms)  # lex-smallest is the canonical form

        N = active_size
        for p in probes:
            key = _canon_key(p["move_history"], N, dedupe_mirror)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(p)
        probes = deduped
    # Sort: source_iteration desc, source_ply desc, source_game basename asc
    # (tiebreaker). Always applied, even when max_probes is None, so downstream
    # consumers (per-probe CSV) see deterministic order.
    probes.sort(key=lambda p: (
        -p["_source_iteration"],
        -p["source_ply"],
        p["source_game"],
    ))

    # Truncate to max_probes after sorting.
    if max_probes is not None and len(probes) > max_probes:
        probes = probes[:max_probes]

    # Strip internal sort-only keys before returning.
    for p in probes:
        p.pop("_source_iteration", None)
    return probes


def _aggregate(rows: list[dict]) -> dict:
    """Per-tier and per-category aggregation."""
    from statistics import median
    def pct(xs, n):
        return round(sum(xs) / n, 3) if n else 0.0

    forced = [r for r in rows if r["confidence"] == "forced"]
    strong = [r for r in rows if r["confidence"] == "strong_advantage"]
    overall = rows

    def bucket_stats(bucket):
        n = len(bucket)
        if n == 0:
            return {"n": 0}
        # Did MCTS actually run? (sims=0 mode leaves mcts_root_value as None)
        any_mcts = any(r.get("mcts_root_value") is not None for r in bucket)
        stats = {
            "n": n,
            "sign_correct_nn_rate": pct([r["sign_correct_nn"] for r in bucket], n),
            "median_nn_magnitude": round(median([r["nn_magnitude"] for r in bucket]), 3),
            "magnitude_in_band_rate": pct([r["magnitude_in_band"] for r in bucket], n),
        }
        if any_mcts:
            stats["sign_correct_mcts_rate"] = pct([r["sign_correct_mcts"] for r in bucket], n)
            stats["search_corrected_rate"] = pct([r["search_corrected"] for r in bucket], n)
            stats["both_wrong_rate"] = pct([r["both_wrong"] for r in bucket], n)
        else:
            stats["sign_correct_mcts_rate"] = None
            stats["search_corrected_rate"] = None
            stats["both_wrong_rate"] = None
        return stats

    by_category = {}
    for cat in {r["category"] for r in rows}:
        by_category[cat] = bucket_stats([r for r in rows if r["category"] == cat])

    return {
        "forced": bucket_stats(forced),
        "strong_advantage": bucket_stats(strong),
        "overall": bucket_stats(overall),
        "by_category": by_category,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate a model against the probe suite.")
    ap.add_argument("--weights", required=True,
                    help="Path to .safetensors checkpoint. REQUIRED for formal runs.")
    ap.add_argument("--probes", default="tests/probes/twixt_probes.json",
                    help="Path to probe suite JSON")
    ap.add_argument("--sims", type=int, default=200,
                    help="MCTS sims per probe (0 to skip MCTS and do NN-only)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--forced-only", action="store_true",
                    help="Evaluate only forced-tier probes (cheap per-iter sampling mode)")
    # Architecture overrides — defaults inherited from create_network() so
    # one canonical source of truth. Only override when evaluating a
    # non-canonical architecture variant.
    ap.add_argument("--hidden", type=int, default=None,
                    help="Override hidden channel count (default: create_network default)")
    ap.add_argument("--n-blocks", type=int, default=None,
                    help="Override residual block count (default: create_network default)")
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        print(f"[ERROR] weights file not found: {args.weights}", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(args.probes):
        print(f"[ERROR] probes file not found: {args.probes}", file=sys.stderr)
        sys.exit(2)

    print(f"[probe_eval] weights: {os.path.abspath(args.weights)}")
    print(f"[probe_eval] probes:  {os.path.abspath(args.probes)}")
    print(f"[probe_eval] sims:    {args.sims}")

    probes_data = json.loads(open(args.probes).read())
    probes = probes_data.get("probes") or probes_data.get("candidates") or []
    if args.forced_only:
        probes = [p for p in probes if p.get("confidence") == "forced"]
        print(f"[probe_eval] forced-only mode: {len(probes)} probes")

    net, in_channels, hidden, n_blocks = _load_network(
        args.weights, hidden=args.hidden, n_blocks=args.n_blocks
    )
    evaluator = LocalGPUEvaluator(net)

    rows = []
    for i, probe in enumerate(probes):
        row = _eval_probe(probe, evaluator, args.sims)
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  evaluated {i+1}/{len(probes)}")

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"[probe_eval] wrote per-probe CSV: {args.out}")

    # Write aggregate JSON
    agg = _aggregate(rows)
    agg_meta = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "weights": os.path.abspath(args.weights),
        "probes": os.path.abspath(args.probes),
        "checkpoint_format": f"{in_channels}-channel",
        "network_architecture": {"hidden": hidden, "n_blocks": n_blocks},
        "probes_total": len(rows),
        "sims": args.sims,
        "forced_only": args.forced_only,
        "aggregate": agg,
    }
    json_out = args.out.rsplit(".", 1)[0] + ".json"
    with open(json_out, "w") as f:
        json.dump(agg_meta, f, indent=2)
    print(f"[probe_eval] wrote aggregate JSON: {json_out}")


if __name__ == "__main__":
    main()


# ============================================================
# Strong-advantage probe tier — Phase 1 structural features
# ============================================================

def compute_phase1_features(state, winner: str) -> dict:
    """Compute Phase-1 structural features for the eventual winner of a game.

    Used by the strong_advantage probe-suite generator to filter candidate
    positions before deep-MCTS labeling. See spec
    docs/superpowers/specs/2026-04-28-strong-advantage-probe-tier-design.md
    Phase 1.

    Args:
        state: TwixtState at the candidate position (winner has not yet won).
        winner: "red" or "black" — the side that wins the source game.

    Returns:
        dict with keys:
          cc_size: int — size of the largest same-color connected component
            for `winner`.
          cc_axis_span: float — fraction of `winner`'s goal axis the largest
            CC spans. Red goal axis is rows; black is cols. Range [0, 1].
          cc_touches_own_goal: bool — True if the largest CC touches at
            least one of `winner`'s two goal edges.
            (red: row 0 or row 23; black: col 0 or col 23).
          axis_span_margin: float — winner_cc_axis_span - loser_cc_axis_span.
            Negative if the loser is more advanced.
          centroid_chebyshev_from_center: int — Chebyshev distance of the
            winner's CC centroid from the board center (11.5, 11.5).
          forced_within_2: bool — Conservative: True if `winner` has an
            immediate (1-ply) winning move. The dict key reserves the
            "within 2" name as a forward-compat slot — today's
            implementation is a 1-ply scan (see is_forced_within_k for
            the rationale). Safe direction for the admission filter
            ("exclude already-forced"): under-reports forced positions,
            so any borderline cases will be filtered by Phase 2 MCTS.
    """
    loser = "black" if winner == "red" else "red"
    winner_pegs = _collect_pegs(state, winner)
    loser_pegs = _collect_pegs(state, loser)

    win_cc, win_span = _largest_connected_component(state, winner_pegs, winner)
    _, lose_span = _largest_connected_component(state, loser_pegs, loser)

    if not win_cc:
        return {
            "cc_size": 0,
            "cc_axis_span": 0.0,
            "cc_touches_own_goal": False,
            "axis_span_margin": -lose_span,
            "centroid_chebyshev_from_center": 23,
            "forced_within_2": False,
        }

    # Goal-touching: does the largest CC touch a goal-axis edge for winner?
    if winner == "red":
        touches = any(r == 0 or r == 23 for r, _ in win_cc)
    else:
        touches = any(c == 0 or c == 23 for _, c in win_cc)

    # Centroid Chebyshev distance from board center (11.5, 11.5).
    avg_r = sum(r for r, _ in win_cc) / len(win_cc)
    avg_c = sum(c for _, c in win_cc) / len(win_cc)
    cheb = int(round(max(abs(avg_r - 11.5), abs(avg_c - 11.5))))

    return {
        "cc_size": len(win_cc),
        "cc_axis_span": round(win_span, 4),
        "cc_touches_own_goal": touches,
        "axis_span_margin": round(win_span - lose_span, 4),
        "centroid_chebyshev_from_center": cheb,
        "forced_within_2": is_forced_within_k(state, winner, k=2),
    }


def _collect_pegs(state, color: str) -> list:
    """Return [(r, c), ...] of all pegs of `color` on the board."""
    return [(r, c) for (r, c), col in state.pegs.items() if col == color]


def _largest_connected_component(state, pegs: list, color: str) -> tuple:
    """Return (cc_cells, axis_span) for the largest knight-bridged component
    of `color`. axis_span is the fraction of `color`'s goal axis the
    component spans (red: row range / 23; black: col range / 23).
    """
    if not pegs:
        return [], 0.0
    bridges = _bridges_for_color(state, color)
    peg_set = set(pegs)
    adj = {p: set() for p in pegs}
    for a, b in bridges:
        if a in peg_set and b in peg_set:
            adj[a].add(b)
            adj[b].add(a)

    seen = set()
    components = []
    for p in pegs:
        if p in seen:
            continue
        stack = [p]
        comp = []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(adj[x] - seen)
        components.append(comp)

    largest = max(components, key=len)
    if color == "red":
        rows = [r for r, _ in largest]
        span = (max(rows) - min(rows)) / 23.0
    else:
        cols = [c for _, c in largest]
        span = (max(cols) - min(cols)) / 23.0
    return largest, span


def _bridges_for_color(state, color: str) -> list:
    """Return [(p1, p2), ...] of every knight-bridge currently held by
    `color` on `state`.

    state.bridges is a set of Bridge tuples ((r1,c1),(r2,c2)); the color
    is determined by looking up the peg owner at one endpoint.
    """
    out = []
    for p1, p2 in state.bridges:
        if state.pegs.get(p1) == color:
            out.append((p1, p2))
    return out


def is_forced_within_k(state, player: str, k: int = 1) -> bool:
    """True if `player` (whose turn it is, or hypothetically) can force a
    win within k plies of play.

    Conservative implementation: only does a 1-ply lookahead — returns
    True iff `player` has any legal move that immediately wins. For k>1
    this is a lower bound (under-reports forced positions), which is
    safe for the strong_advantage filter ("exclude already-forced") —
    we'd rather over-admit a not-quite-forced candidate (Phase 2 MCTS
    will filter it) than under-admit a genuinely strong-advantage one.

    A future tightening can extend to a true negamax k-ply search; the
    interface accepts k for forward-compat.
    """
    if state.to_move != player:
        # Conservative: when it's not `player`'s turn, return False without
        # considering opponent responses. For the admission filter this is
        # safe (we under-report forced, so any borderline cases pass through
        # to Phase 2 MCTS which catches them). A true k>=2 negamax would
        # need to enumerate opponent replies — out of scope for v1.
        return False
    for move in state.legal_moves():
        try:
            next_state = state.apply_move(move)
        except Exception:
            continue
        if next_state.is_terminal() and next_state.winner() == player:
            return True
    return False


def extract_strong_advantage_candidates(
    games: list,
    *,
    k_plies_range: tuple = (3, 8),
    min_cc_size: int = 10,
    min_cc_axis_span: float = 0.55,
    min_axis_span_margin: float = 0.10,
    require_cc_touches_own_goal: bool = True,
    exclude_forced_within_2: bool = True,
    category_min_count: int = 5,
) -> tuple:
    """Phase-1 candidate mining for the strong_advantage probe tier.

    Walks each decisive game, samples positions at terminal_ply - K for K
    in k_plies_range (inclusive on both ends), computes structural features
    on each, and applies the Phase-1 admission gate. See spec Phase 1.

    Args:
        games: list of game-record dicts (must contain `moves`, `winner`,
            `meta.reason` or fallback `winner_reason`, optionally `starting_player`).
        k_plies_range: (min_K, max_K) plies before terminal to sample.
        min_cc_size, min_cc_axis_span, min_axis_span_margin: Phase-1
            heuristic thresholds.
        require_cc_touches_own_goal, exclude_forced_within_2: gate flags.
        category_min_count: warning threshold; if any of the 4 categories
            ends up with fewer surviving candidates than this, a warning
            is printed (the candidate list is returned regardless).

    Returns:
        (candidates, audit) where:
          candidates: list of dicts with `move_history`, `ply`, `winner`,
            `category`, `phase1_features`, `source_game`, `source_ply`,
            `starting_player`. Sorted by (-iter, -source_ply, source_game)
            for deterministic order.
          audit: list of dicts with `source_game`, `source_ply`, `reason`,
            and `phase1_features`. One entry per dropped candidate; the
            audit row is also written for ADMITTED candidates with reason
            "admitted" so the audit captures the full provenance.
    """
    from .game.twixt_state import TwixtState

    candidates = []
    audit = []

    for game in games:
        # Canonical schema (game_saver.py): meta.reason holds the win/draw/timeout
        # status. Older fixtures may put it at top level as winner_reason — fall
        # back for back-compat.
        meta = game.get("meta") or {}
        reason = meta.get("reason") or game.get("winner_reason")
        if reason != "win":
            continue
        winner = game.get("winner")
        if winner not in ("red", "black"):
            continue
        moves_list = game.get("moves") or []
        if not moves_list:
            continue
        terminal_ply = len(moves_list)
        starting_player = game.get("starting_player", "red")
        source_game = game.get("source_game") or _derive_source_game_basename(game)

        for k in range(k_plies_range[0], k_plies_range[1] + 1):
            target_ply = terminal_ply - k
            if target_ply < 1:
                continue

            state = TwixtState(active_size=24, to_move=starting_player)
            for i in range(target_ply):
                m = moves_list[i]
                state = state.apply_move((m["row"], m["col"]))

            feats = compute_phase1_features(state, winner=winner)
            base_audit = {
                "source_game": source_game,
                "source_ply": target_ply,
                "phase1_features": feats,
            }

            if feats["cc_size"] < min_cc_size:
                audit.append({**base_audit, "reason": "phase1_cc_size"})
                continue
            if feats["cc_axis_span"] < min_cc_axis_span:
                audit.append({**base_audit, "reason": "phase1_axis_span"})
                continue
            if feats["axis_span_margin"] < min_axis_span_margin:
                audit.append({**base_audit, "reason": "phase1_axis_span_margin"})
                continue
            if require_cc_touches_own_goal and not feats["cc_touches_own_goal"]:
                audit.append({**base_audit, "reason": "phase1_no_goal_touch"})
                continue
            if exclude_forced_within_2 and feats["forced_within_2"]:
                audit.append({**base_audit, "reason": "phase1_already_forced"})
                continue

            cheb = feats["centroid_chebyshev_from_center"]
            if 7 <= cheb <= 8:
                audit.append({**base_audit, "reason": "category_midband"})
                continue

            if cheb <= 6:
                category = f"chain_advantage_central_{winner}"
            else:  # cheb >= 9
                category = f"chain_advantage_edge_{winner}"

            cand = {
                "move_history": [(m["row"], m["col"]) for m in moves_list[:target_ply]],
                "ply": target_ply,
                "winner": winner,
                "category": category,
                "phase1_features": feats,
                "source_game": source_game,
                "source_ply": target_ply,
                "starting_player": starting_player,
            }
            candidates.append(cand)
            audit.append({**base_audit, "reason": "admitted"})

    def _sort_key(c: dict) -> tuple:
        try:
            iter_num = int(c["source_game"].split("_")[1])
        except (IndexError, ValueError):
            iter_num = 0
        return (-iter_num, -c["source_ply"], c["source_game"])

    candidates.sort(key=_sort_key)

    for cat in [
        "chain_advantage_central_red",
        "chain_advantage_central_black",
        "chain_advantage_edge_red",
        "chain_advantage_edge_black",
    ]:
        n = sum(1 for c in candidates if c["category"] == cat)
        if n < category_min_count:
            import sys
            print(
                f"[probe_suite] WARNING: category {cat} has {n} candidates "
                f"(< {category_min_count}); broaden --source-iter-range or "
                f"relax thresholds.",
                file=sys.stderr,
            )

    return candidates, audit


def _derive_source_game_basename(game: dict) -> str:
    """Best-effort recovery of the source_game basename from a game dict.

    Prefers the explicit `id` field (canonical for production game records;
    see scripts/GPU/alphazero/game_saver.py which stamps it as
    'iter_NNNN_game_MMM'). Falls back to building from meta.iteration +
    meta.game_idx if id is missing (test fixtures, older schemas).
    """
    if game.get("id"):
        return game["id"]
    meta = game.get("meta") or {}
    iteration = meta.get("iteration", 0)
    # Canonical key is `game_idx` (game_saver.py:88), NOT `game_index`.
    game_idx = meta.get("game_idx", 0)
    return f"iter_{iteration:04d}_game_{game_idx:03d}"


# ============================================================
# Strong-advantage probe tier — Phase 2 deep-MCTS labeling
# ============================================================

def label_candidate_with_mcts(
    state,
    *,
    sims: int,
    repeats: int,
    rng_seed_base: int,
    labeler=None,
) -> dict:
    """Phase-2 deep-MCTS labeling for one candidate position.

    Runs MCTS at `sims` simulations × `repeats` repeats with different RNG
    seeds per repeat. Aggregates per-run results.

    Args:
        state: TwixtState at the candidate position.
        sims: simulations per MCTS run.
        repeats: number of repeated MCTS runs.
        rng_seed_base: integer seed; per-run seed = rng_seed_base ^ repeat_idx.
        labeler: optional callable (state, sims, seed) -> (root_value,
            top1_share). If None, uses the production deep-MCTS labeler
            from `_default_mcts_labeler`. Tests inject a stub here.

    Returns:
        dict with mean_root_value, value_per_run, value_stability,
        min_top1_share, label_mcts_sims, label_mcts_repeats, rng_seed_base.
    """
    if labeler is None:
        labeler = _default_mcts_labeler

    values = []
    top1_shares = []
    for repeat_idx in range(repeats):
        seed = rng_seed_base ^ repeat_idx
        v, t1 = labeler(state, sims, seed)
        values.append(v)
        top1_shares.append(t1)

    return {
        "mean_root_value": round(sum(values) / len(values), 6),
        "value_per_run": [round(v, 6) for v in values],
        "value_stability": round(max(values) - min(values), 6),
        "min_top1_share": round(min(top1_shares), 6),
        "label_mcts_sims": sims,
        "label_mcts_repeats": repeats,
        "rng_seed_base": rng_seed_base,
    }


def _default_mcts_labeler(state, sims, seed):
    """Production deep-MCTS labeler. Uses the network registered via
    _set_default_labeler_network() and runs MCTS at the given sim count
    and seed.

    Returns (root_value_from_stm_perspective, top1_visit_share).

    Tests should pass an explicit `labeler=` rather than rely on this.
    """
    if _DEFAULT_LABELER_NETWORK is None:
        raise RuntimeError(
            "Default MCTS labeler called without a registered network. "
            "Either pass labeler= explicitly or call "
            "_set_default_labeler_network() first."
        )
    # MCTS interface (verified against scripts/GPU/alphazero/mcts.py):
    #   MCTS(evaluator, cfg, rng=...)   constructor
    #   mcts.search(state, add_noise=False) -> (visit_counts dict, root_value)
    #   visit_counts: Dict[(r, c), int]
    #   root_value: float (from state.to_move's perspective)
    evaluator = LocalGPUEvaluator(_DEFAULT_LABELER_NETWORK)
    cfg = MCTSConfig(n_simulations=sims)
    mcts = MCTS(evaluator, cfg, rng=random.Random(seed))
    visit_counts, root_value = mcts.search(state, add_noise=False)
    if not visit_counts:
        return float(root_value), 0.0
    total = sum(visit_counts.values()) or 1
    top1 = max(visit_counts.values())
    return float(root_value), top1 / total


_DEFAULT_LABELER_NETWORK = None


def _set_default_labeler_network(network) -> None:
    """Register the production network for `_default_mcts_labeler`."""
    global _DEFAULT_LABELER_NETWORK
    _DEFAULT_LABELER_NETWORK = network


# ============================================================
# Strong-advantage probe tier — Phase 2 admission filter
# ============================================================

def apply_admission_filter(
    candidate: dict,
    *,
    magnitude_threshold: float,
    top1_share_floor: float,
    stability_cap: float,
) -> tuple:
    """Phase-2 admission gate. Returns (admitted: bool, reason: str).

    Reason is one of:
      "admitted", "sign_mismatch", "magnitude_below_threshold",
      "low_top1_share", "unstable_value", "position_already_forced".

    Order of checks matters only for the audit reason — first failing
    clause is reported. Sign-match is checked first because it's the
    cross-check against the source-game winner.
    """
    label = candidate["phase2_label"]
    feats = candidate["phase1_features"]
    winner = candidate["winner"]
    expected_sign = 1 if winner == "red" else -1

    if (label["mean_root_value"] >= 0) != (expected_sign == 1):
        return False, "sign_mismatch"
    if abs(label["mean_root_value"]) < magnitude_threshold:
        return False, "magnitude_below_threshold"
    if label["min_top1_share"] < top1_share_floor:
        return False, "low_top1_share"
    if label["value_stability"] > stability_cap:
        return False, "unstable_value"
    if feats.get("forced_within_2"):
        return False, "position_already_forced"
    return True, "admitted"
