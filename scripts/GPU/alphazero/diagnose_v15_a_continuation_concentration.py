"""v15 Phase-0 (READ-ONLY) diagnostic: A-continuation concentration.

The "Targeted Value Calibration" line established that the remaining A-gate
miss is MCTS/SEARCH AMPLIFICATION, not raw-value undercorrection (raw A is
already <= 0 at BASE, but the 400-sim MCTS gate sees the value rise by
roughly +0.20 to +0.27). v15 will eventually correct the searched CHILD
states MCTS backs up rather than the root's raw value. This module is
Phase 0 only: it measures whether the optimistic value backup at each A
root is concentrated in a few children (a few-row correction would be
viable in Phase 1) or spread broadly across many children (a tree/path-level
design would be needed instead).

READ-ONLY: this script only reads the BASE + v14b checkpoints and the A
probe manifest, and writes exactly one diagnostic CSV. It does not modify
mcts.py, continuation_extraction.py, calibration_pool.py, eval_runner.py,
probe_eval.py, trainer.py, network.py, or any manifest/builder. No training,
no manifest, no child replay JSONs — that is Phase 1.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from .build_mcts_root_retention_manifest import CORRECTION_TAG, row_seed
from .build_teacher_calibration_manifest import _teacher_infer
from .eval_raw_nn_position_rows import to_black
from .goal_line_trigger_probe_cases import position_state
from .mcts import decode_move, encode_move
from .position_probe_cases import load_csv_manifest

FIELDNAMES = [
    "root_case_id", "child_move", "depth", "visit_count", "visit_share",
    "child_contribution_share", "positive_contribution_share", "child_q_value",
    "child_raw_black_BASE", "child_raw_black_v14b", "root_mcts_black_value",
    "root_case_classification", "root_top3_positive_share",
]

DEFAULT_BASE_CHECKPOINT = "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"
DEFAULT_V14B_CHECKPOINT = (
    "checkpoints/alphazero-v14b-value-adapter-projection-from-calib020-0001/"
    "model_iter_0001.safetensors")
DEFAULT_A_MANIFEST = (
    "logs/eval/loss_analysis_v2_calib020_0001_vs_0379_black/"
    "0001_black_post_opening_top30_predrop_probe_manifest.csv")
DEFAULT_OUT = "logs/eval/v15prep_a_continuation_concentration.csv"


def per_child_metrics(root) -> list[dict]:
    """Per root-child visit_share + contribution (root perspective). Contributions
    sum to root.q_value; visit_shares sum to 1. positive_contribution_share is each
    child's fraction of the POSITIVE backup mass (negative/defensive children -> 0),
    for sorting within a root. Children with 0 visits are dropped."""
    from scripts.GPU.alphazero.mcts import decode_move
    total = root.visit_count or sum(c.visit_count for c in root.children.values())
    out = []
    for move_id, ch in root.children.items():
        if ch.visit_count <= 0:
            continue
        vs = ch.visit_count / total
        out.append({
            "move": list(decode_move(move_id)),
            "visit_count": ch.visit_count,
            "visit_share": vs,
            "q_value": ch.q_value,
            "child_contribution_share": vs * (-ch.q_value),   # root perspective
        })
    total_pos = sum(r["child_contribution_share"] for r in out
                    if r["child_contribution_share"] > 0)
    for r in out:
        r["positive_contribution_share"] = (
            max(r["child_contribution_share"], 0.0) / total_pos if total_pos > 0 else 0.0)
    return out


def classify_concentration(metrics: list[dict], top_n: int = 3) -> tuple[str, float]:
    """Share of the POSITIVE backup mass explained by the top_n highest-contribution
    children. >=0.70 concentrated / 0.40-0.70 semi / <0.40 broad (spec §0)."""
    pos = [m["child_contribution_share"] for m in metrics if m["child_contribution_share"] > 0]
    total_pos = sum(pos)
    if total_pos <= 0:
        return "broad", 0.0
    top = sum(sorted(pos, reverse=True)[:top_n])
    share = top / total_pos
    label = "concentrated" if share >= 0.70 else ("semi" if share >= 0.40 else "broad")
    return label, share


def path_moves_of(node) -> tuple:
    """(r, c) moves from the root to this node, via parent links.

    Copied (NOT imported) from continuation_extraction.path_moves_of. That
    module's entry point, extract_continuations, raises ValueError for any
    tag outside the B/C/D families (FAMILY_BY_SOURCE_TAG) -- the A family
    (CORRECTION_TAG == "black_predrop_correction") is not one of them -- so
    this read-only Phase-0 diagnostic stays decoupled from the tag-gated
    extraction machinery and reuses only the generic parent-chain walk.
    """
    moves = []
    while node.parent is not None:
        moves.append(decode_move(node.move))
        node = node.parent
    return tuple(reversed(moves))


def _real_search_fn(base_checkpoint: str, sims: int, eval_batch_size: int,
                    stall_flush_sims: int):
    """Gate-faithful root-returning search. Heavy imports deferred (mirrors
    build_searched_continuation_retention_manifest._real_search_fn -- same
    evaluator/config as the v5/v6 builders and the A/B/C/D gate probes)."""
    from .eval_runner import EvalConfig, cfg_from, _default_evaluator_factory
    from .mcts import MCTS
    evaluator = _default_evaluator_factory(base_checkpoint)
    cfg = cfg_from(EvalConfig(mcts_sims=sims,
                              mcts_eval_batch_size=eval_batch_size,
                              mcts_stall_flush_sims=stall_flush_sims))

    def search_fn(state, seed):
        return MCTS(evaluator, cfg, random.Random(seed)).search_with_root(
            state, add_noise=False)

    return search_fn


def _build_raw_evaluator(checkpoint_path: str):
    """EVAL-mode (net.eval()) LocalGPUEvaluator for raw (non-MCTS) child
    scoring. Heavy imports deferred to match the codebase's fakes-in-tests
    convention. load_network_for_scoring auto-detects a value adapter from
    the checkpoint's own keys, so this works unmodified for BASE (no
    adapter) and v14b (has one) alike -- do not hand-build the network."""
    from .local_evaluator import LocalGPUEvaluator
    from .probe_eval import load_network_for_scoring
    net, *_ = load_network_for_scoring(checkpoint_path)
    net.eval()
    return LocalGPUEvaluator(net)


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description="v15 Phase-0 (READ-ONLY) diagnostic: measure whether the "
                     "optimistic MCTS value backup at each A-family "
                     "(black_predrop_correction) root is concentrated in a "
                     "few children or spread broadly, to decide whether the "
                     "Phase-1 searched-continuation correction can be a "
                     "few-row fix or needs a tree/path-level design. Reads "
                     "the BASE + v14b checkpoints and the A probe manifest; "
                     "writes one concentration CSV. No training, no "
                     "manifest, no child replay JSONs.")
    ap.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT,
                    help="searched (gate-faithful 400-sim MCTS) AND raw-scored "
                         "(eval-mode) checkpoint.")
    ap.add_argument("--v14b-checkpoint", default=DEFAULT_V14B_CHECKPOINT,
                    help="raw-scored (eval-mode) only checkpoint; never searched.")
    ap.add_argument("--a-manifest", default=DEFAULT_A_MANIFEST,
                    help="A-family (black_predrop_correction) probe manifest "
                         "csv; each row is one root to reconstruct + search.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--eval-batch-size", type=int, default=14)
    ap.add_argument("--stall-flush-sims", type=int, default=48)
    ap.add_argument("--position-probe-base-seed", type=int, default=20260616)
    ap.add_argument("--goal-line-base-seed", type=int, default=20260614,
                    help="unused by A rows (row_seed's goal-line branch never "
                         "fires for CORRECTION_TAG); kept for row_seed parity.")
    ap.add_argument("--limit-cases", type=int, default=None,
                    help="process only the first N manifest rows (smoke testing).")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    rows = load_csv_manifest(args.a_manifest)["cases"]
    if args.limit_cases is not None:
        rows = rows[:args.limit_cases]

    base_raw_evaluator = _build_raw_evaluator(args.base_checkpoint)
    v14b_raw_evaluator = _build_raw_evaluator(args.v14b_checkpoint)
    search_fn = _real_search_fn(args.base_checkpoint, args.sims,
                                args.eval_batch_size, args.stall_flush_sims)

    out_rows = []
    root_labels = []
    for i, row in enumerate(rows):
        case_id = row["case_id"]
        replay = json.loads(Path(row["replay_path"]).read_text())
        ply = int(float(row["position_ply"]))
        side = row["side_to_move"]
        state = position_state(replay, ply, side)
        seed = row_seed(CORRECTION_TAG, row["game_idx"], ply,
                        pos_base_seed=args.position_probe_base_seed,
                        goal_base_seed=args.goal_line_base_seed)
        counts, root_value_stm, root = search_fn(state, seed)

        metrics = per_child_metrics(root)
        _sum = sum(m["child_contribution_share"] for m in metrics)
        if i == 0:
            # MUST-FIX real-root sign sanity check (brief Step 5): the
            # contribution metric's sign/invariant was derived, not yet
            # verified against a real MCTSNode. Verify + print loudly before
            # trusting any concentration read from this run.
            assert abs(_sum - root.q_value) < 1e-6, (
                f"contribution invariant broken: sum={_sum:+.6f} != root.q_value={root.q_value:+.6f} "
                f"(check the (-child.q_value) sign)")
            assert abs(root.q_value - root_value_stm) < 1e-6, (
                f"SIGN/PERSPECTIVE MISMATCH: root.q_value={root.q_value:+.6f} != "
                f"root_value_stm={root_value_stm:+.6f}; the contribution metric is likely sign-flipped "
                f"(sum matches -root.q_value) — DO NOT trust the concentration read")
            print(f"[v15 phase0] sign sanity OK on {case_id}: sum(contrib)={_sum:+.4f} "
                  f"== root.q={root.q_value:+.4f} == root_value_stm={root_value_stm:+.4f}")
        else:
            # Keep the invariant assert on every root; fail loud if violated.
            assert abs(_sum - root.q_value) < 1e-6, (
                f"contribution invariant broken on {case_id}: sum={_sum:+.6f} != "
                f"root.q_value={root.q_value:+.6f} (check the (-child.q_value) sign)")

        label, share = classify_concentration(metrics)
        root_labels.append(label)
        root_mcts_black_value = to_black(root_value_stm, side)
        print(f"[v15 phase0] {case_id}: {label} (top3_positive_share={share:.3f}, "
              f"n_children={len(metrics)}, root_mcts_black_value={root_mcts_black_value:+.4f})")

        for m in metrics:
            child = root.children[encode_move(*m["move"])]
            depth = len(path_moves_of(child))
            _, _, child_raw_stm_base = _teacher_infer(child.state, base_raw_evaluator)
            _, _, child_raw_stm_v14b = _teacher_infer(child.state, v14b_raw_evaluator)
            out_rows.append({
                "root_case_id": case_id,
                "child_move": f"{m['move'][0]}:{m['move'][1]}",
                "depth": depth,
                "visit_count": m["visit_count"],
                "visit_share": m["visit_share"],
                "child_contribution_share": m["child_contribution_share"],
                "positive_contribution_share": m["positive_contribution_share"],
                "child_q_value": m["q_value"],
                "child_raw_black_BASE": to_black(child_raw_stm_base, child.state.to_move),
                "child_raw_black_v14b": to_black(child_raw_stm_v14b, child.state.to_move),
                "root_mcts_black_value": root_mcts_black_value,
                "root_case_classification": label,
                "root_top3_positive_share": share,
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(out_rows)

    agg = Counter(root_labels)
    print(f"[v15 phase0] aggregate across {len(root_labels)} roots: {dict(agg)}")
    print(f"wrote {len(out_rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
