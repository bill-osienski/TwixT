"""v15 Phase-0.5 (READ-ONLY) diagnostic: selected-branch subtree walk.

Phase 0 established that the optimistic MCTS backup at the A roots is
concentrated: all 17 roots with root_mcts_black_value > 0 classify as
"concentrated", and the top-3 children carry 98.1% of the positive backup
mass. It also found the reason not to build depth-1 correction rows: at the
top child of each overvaluing root the RAW black value is already -0.087
under BASE while its SEARCHED value is +0.619. A node's own NN evaluation
enters its q exactly once, so a child with ~370 visits moves its searched q
by only ~delta/370 -- correcting a depth-1 child from -0.087 to -0.35 shifts
its backed-up value by ~0.0007. Depth-1 correction can therefore only work
through generalization to the deep leaves, the same assumption that failed
for root correction across v2-v14.

This script tests that assumption. It re-runs the deterministic BASE search
on the overvaluing roots, walks EVERY expanded descendant (visit_count >= 1)
beneath each selected positive branch, and records each node's raw value
(BASE and v14b) alongside its searched q, with the principal variation
annotated as a column. The by-depth summary is emitted twice -- over the full
subtree and over PV nodes only -- and the comparison answers: is the raw
optimism path-concentrated, spread across the frontier, or absent entirely?

pct_visit_mass_raw_positive is the decision metric, NOT pct_raw_positive: a
thousand single-visit frontier leaves must not outvote one 300-visit node.

READ-ONLY: reads the BASE + v14b checkpoints, the A probe manifest, and the
Phase-0 CSV; writes two diagnostic CSVs. No manifest, no child replay JSONs,
no training, and no change to mcts.py, calibration_pool.py, trainer.py,
network.py, probe_eval.py, eval_runner.py, or continuation_extraction.py.
"""
from __future__ import annotations

import argparse
import csv
from collections import OrderedDict, defaultdict
from pathlib import Path

from .diagnose_v15_a_continuation_concentration import (
    DEFAULT_A_MANIFEST, DEFAULT_BASE_CHECKPOINT, DEFAULT_V14B_CHECKPOINT,
    _build_raw_evaluator, _real_search_fn, path_moves_of, per_child_metrics,
    raw_black_value, search_for_row)
from .eval_raw_nn_position_rows import to_black
from .mcts import decode_move, encode_move
from .position_probe_cases import load_csv_manifest

DEFAULT_PHASE0_CSV = "logs/eval/v15prep_a_continuation_concentration.csv"
DEFAULT_OUT = "logs/eval/v15prep_a_selected_branch_subtrees.csv"
DEFAULT_SUMMARY_OUT = (
    "logs/eval/v15prep_a_selected_branch_subtrees_by_depth_summary.csv")

FIELDNAMES = [
    "root_case_id", "root_mcts_black_value", "root_case_classification",
    "branch_rank", "root_child_move", "root_child_positive_contribution_share",
    "depth_from_root", "depth_from_selected_child", "path_moves",
    "move_from_parent", "visit_count", "visit_share_from_parent",
    "visit_share_from_root", "q_value_node_perspective",
    "q_value_root_perspective", "raw_black_BASE", "raw_black_v14b",
    "raw_delta_v14b_minus_BASE", "raw_positive_BASE", "raw_positive_v14b",
    "is_pv_path", "pv_depth_index", "num_children", "unvisited_children_count",
    "is_terminal",
]

SUMMARY_FIELDNAMES = (
    ["scope", "depth_from_root", "nodes_count", "raw_scored_nodes_count",
     "unvisited_children_count", "total_visit_share_from_root"]
    + [f"{stat}_{tag}"
       for tag in ("BASE", "v14b")
       for stat in ("mean_raw_black", "weighted_mean_raw_black",
                    "pct_raw_positive", "pct_visit_mass_raw_positive",
                    "max_raw_black")])

TOLERANCE = 1e-6


def _best_child(node):
    """Max-visit child (ties: lowest encoded move id); None if no visited child.

    COPIED from continuation_extraction._best_child: extract_continuations
    raises for the A tag, so that module cannot be used here, and Phase 0 set
    the precedent of copying its generic helpers rather than modifying it.
    """
    visited = [c for c in node.children.values() if c.visit_count > 0]
    if not visited:
        return None
    return min(visited, key=lambda c: (-c.visit_count, c.move))


def load_phase0_rows(csv_path) -> list[dict]:
    """Rows of the Phase-0 concentration CSV, values as strings."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def group_phase0_by_root(rows) -> "OrderedDict[str, list[dict]]":
    """Phase-0 rows grouped by root_case_id, preserving file order."""
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for r in rows:
        groups.setdefault(r["root_case_id"], []).append(r)
    return groups


def select_positive_branches(groups, *, cum_threshold: float = 0.90,
                             max_children: int = 3) -> list:
    """[(root_case_id, [phase-0 child rows in rank order])] for the roots that
    actually overvalue (root_mcts_black_value > 0). Within a root, children
    with positive contribution are ranked by positive_contribution_share
    descending and taken until the cumulative share reaches cum_threshold or
    max_children have been taken. Roots with no positive backup mass are
    skipped -- Phase 0 labels those "broad" with share 0.0, which conflates
    "not overvalued at all" with "overvalued broadly"."""
    out = []
    for cid, rows in groups.items():
        if float(rows[0]["root_mcts_black_value"]) <= 0:
            continue
        positive = [r for r in rows
                    if float(r["child_contribution_share"]) > 0]
        if not positive:
            continue
        positive.sort(key=lambda r: -float(r["positive_contribution_share"]))
        picked, cum = [], 0.0
        for r in positive:
            picked.append(r)
            cum += float(r["positive_contribution_share"])
            if cum >= cum_threshold or len(picked) >= max_children:
                break
        out.append((cid, picked))
    return out


def walk_subtree(branch_root) -> list:
    """Every descendant of branch_root with visit_count >= 1, including
    branch_root itself. Iterative (no recursion depth limit). Unvisited
    children are excluded and never descended into -- MCTS creates them at
    parent expansion but never evaluates them, so they carry no NN value."""
    out, stack = [], [branch_root]
    while stack:
        node = stack.pop()
        if node.visit_count < 1:
            continue
        out.append(node)
        stack.extend(node.children.values())
    return out


def pv_chain(branch_root) -> dict:
    """{id(node): pv_depth_index} along the best-child chain rooted at
    branch_root (branch_root itself is index 0)."""
    chain, node, i = {}, branch_root, 0
    while node is not None:
        chain[id(node)] = i
        node = _best_child(node)
        i += 1
    return chain


def node_metrics(node, root, branch_root, pv_index) -> dict:
    """Tree-derived per-node fields (no NN evaluation). q_value_root_perspective
    is the analysis column: node.q_value is stored in the node's OWN to-move
    perspective (mcts._backup flips the sign once per level, and apply_move
    flips to_move every ply), so converting with node.state.to_move lands every
    node in black's perspective -- which is the root's, since every A root is
    black to move."""
    # node_metrics is only called for selected branch descendants, never for the
    # true MCTS root, so node.parent is guaranteed non-None below.
    path = path_moves_of(node)
    branch_depth = len(path_moves_of(branch_root))
    return {
        "depth_from_root": len(path),
        "depth_from_selected_child": len(path) - branch_depth,
        "path_moves": " ".join(f"{r}:{c}" for r, c in path),
        "move_from_parent": "{}:{}".format(*decode_move(node.move)),
        "visit_count": node.visit_count,
        "visit_share_from_parent": node.visit_count / node.parent.visit_count,
        "visit_share_from_root": node.visit_count / root.visit_count,
        "q_value_node_perspective": node.q_value,
        "q_value_root_perspective": to_black(node.q_value, node.state.to_move),
        "is_pv_path": pv_index is not None,
        "pv_depth_index": "" if pv_index is None else pv_index,
        "num_children": len(node.children),
        "unvisited_children_count": sum(
            1 for c in node.children.values() if c.visit_count < 1),
        "is_terminal": node.state.is_terminal(),
    }


def aggregate_by_depth(rows, scope: str) -> list[dict]:
    """Per-depth summary over `rows` (already filtered to `scope`). Terminal
    nodes carry no raw value (blank cells) and are excluded from every raw_*
    statistic, but still counted in nodes_count and total_visit_share_from_root
    so visit mass is never silently dropped. pct_visit_mass_raw_positive is
    normalized by the SCORED visit mass, so a terminal node cannot dilute the
    decision metric."""
    by_depth = defaultdict(list)
    for r in rows:
        by_depth[int(r["depth_from_root"])].append(r)

    out = []
    for depth in sorted(by_depth):
        group = by_depth[depth]
        rec = {
            "scope": scope,
            "depth_from_root": depth,
            "nodes_count": len(group),
            "unvisited_children_count": sum(
                int(r["unvisited_children_count"]) for r in group),
            "total_visit_share_from_root": sum(
                float(r["visit_share_from_root"]) for r in group),
        }
        scored = [r for r in group if r["raw_black_BASE"] != ""]
        rec["raw_scored_nodes_count"] = len(scored)
        scored_mass = sum(float(r["visit_share_from_root"]) for r in scored)

        for tag in ("BASE", "v14b"):
            if not scored:
                for stat in ("mean_raw_black", "weighted_mean_raw_black",
                             "pct_raw_positive", "pct_visit_mass_raw_positive",
                             "max_raw_black"):
                    rec[f"{stat}_{tag}"] = ""
                continue
            vals = [float(r[f"raw_black_{tag}"]) for r in scored]
            masses = [float(r["visit_share_from_root"]) for r in scored]
            weighted = sum(m * v for m, v in zip(masses, vals))
            pos_mass = sum(m for m, v in zip(masses, vals) if v > 0)
            rec[f"mean_raw_black_{tag}"] = sum(vals) / len(vals)
            rec[f"weighted_mean_raw_black_{tag}"] = (
                weighted / scored_mass if scored_mass > 0 else 0.0)
            rec[f"pct_raw_positive_{tag}"] = (
                sum(1 for v in vals if v > 0) / len(vals))
            rec[f"pct_visit_mass_raw_positive_{tag}"] = (
                pos_mass / scored_mass if scored_mass > 0 else 0.0)
            rec[f"max_raw_black_{tag}"] = max(vals)
        out.append(rec)
    return out


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description="v15 Phase-0.5 (READ-ONLY) diagnostic: walk the full "
                    "expanded subtree under each selected positive branch of "
                    "the overvaluing A roots, recording raw (BASE + v14b) and "
                    "searched values per node with PV annotation, to decide "
                    "whether Phase-1 correction should be PV/path-level, "
                    "frontier/tree-level, or should not be built. Reads the "
                    "checkpoints, the A probe manifest, and the Phase-0 CSV; "
                    "writes two diagnostic CSVs. No manifest, no replay JSONs, "
                    "no training.")
    ap.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT,
                    help="searched (gate-faithful 400-sim MCTS) AND raw-scored "
                         "(eval-mode) checkpoint.")
    ap.add_argument("--v14b-checkpoint", default=DEFAULT_V14B_CHECKPOINT,
                    help="raw-scored (eval-mode) only checkpoint; never searched.")
    ap.add_argument("--a-manifest", default=DEFAULT_A_MANIFEST)
    ap.add_argument("--phase0-csv", default=DEFAULT_PHASE0_CSV,
                    help="the Phase-0 concentration CSV; supplies the roots, "
                         "the selected children, and the cross-check values.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    ap.add_argument("--cum-threshold", type=float, default=0.90)
    ap.add_argument("--max-children", type=int, default=3)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--eval-batch-size", type=int, default=14)
    ap.add_argument("--stall-flush-sims", type=int, default=48)
    ap.add_argument("--position-probe-base-seed", type=int, default=20260616)
    ap.add_argument("--goal-line-base-seed", type=int, default=20260614)
    ap.add_argument("--limit-roots", type=int, default=None,
                    help="process only the first N selected roots (smoke testing).")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    groups = group_phase0_by_root(load_phase0_rows(args.phase0_csv))
    branches = select_positive_branches(groups,
                                        cum_threshold=args.cum_threshold,
                                        max_children=args.max_children)
    if args.limit_roots is not None:
        branches = branches[:args.limit_roots]
    manifest = {r["case_id"]: r
                for r in load_csv_manifest(args.a_manifest)["cases"]}

    base_ev = _build_raw_evaluator(args.base_checkpoint)
    v14b_ev = _build_raw_evaluator(args.v14b_checkpoint)
    search_fn = _real_search_fn(args.base_checkpoint, args.sims,
                                args.eval_batch_size, args.stall_flush_sims)

    out_rows = []
    for cid, picked in branches:
        state, side, root_value_stm, root = search_for_row(
            manifest[cid], search_fn,
            pos_base_seed=args.position_probe_base_seed,
            goal_base_seed=args.goal_line_base_seed)

        # Every A root is black to move; this is what makes "root perspective"
        # and "black perspective" the same thing in every column below.
        if side != "black":
            raise SystemExit(f"{cid}: expected side_to_move 'black', got {side!r}")

        # CHECK 1 -- tree reproduction: the fresh search must reproduce Phase 0's.
        fresh_black = to_black(root_value_stm, side)
        csv_black = float(picked[0]["root_mcts_black_value"])
        if abs(fresh_black - csv_black) > TOLERANCE:
            raise SystemExit(
                f"{cid}: TREE NOT REPRODUCED: fresh root_mcts_black_value="
                f"{fresh_black:+.6f} != Phase-0 CSV {csv_black:+.6f}; the "
                f"search config or seed drifted -- DO NOT trust this run")

        # CHECK 2 -- Phase 0's contribution invariant, on every root.
        metrics = per_child_metrics(root)
        _sum = sum(m["child_contribution_share"] for m in metrics)
        if abs(_sum - root.q_value) > TOLERANCE:
            raise SystemExit(
                f"{cid}: contribution invariant broken: sum={_sum:+.6f} != "
                f"root.q_value={root.q_value:+.6f} (check the (-child.q_value) sign)")

        for rank, prow in enumerate(picked, start=1):
            move_rc = tuple(int(x) for x in prow["child_move"].split(":"))
            branch_root = root.children[encode_move(*move_rc)]

            # CHECK 3 -- cross-CSV perspective tie: the depth-1 node's
            # root-perspective q must equal -child_q_value from Phase 0.
            qrp = to_black(branch_root.q_value, branch_root.state.to_move)
            expected = -float(prow["child_q_value"])
            if abs(qrp - expected) > TOLERANCE:
                raise SystemExit(
                    f"{cid} child {prow['child_move']}: PERSPECTIVE MISMATCH: "
                    f"q_value_root_perspective={qrp:+.6f} != -child_q_value="
                    f"{expected:+.6f} -- the to_black conversion drifted between "
                    f"Phase 0 and Phase 0.5; DO NOT trust this run")

            chain = pv_chain(branch_root)
            for node in walk_subtree(branch_root):
                m = node_metrics(node, root, branch_root, chain.get(id(node)))
                if m["is_terminal"]:
                    raw_base = raw_v14b = delta = ""
                    pos_base = pos_v14b = ""
                else:
                    raw_base = raw_black_value(node.state, base_ev)
                    raw_v14b = raw_black_value(node.state, v14b_ev)
                    delta = raw_v14b - raw_base
                    pos_base, pos_v14b = raw_base > 0, raw_v14b > 0
                out_rows.append({
                    "root_case_id": cid,
                    "root_mcts_black_value": fresh_black,
                    "root_case_classification": prow["root_case_classification"],
                    "branch_rank": rank,
                    "root_child_move": prow["child_move"],
                    "root_child_positive_contribution_share":
                        float(prow["positive_contribution_share"]),
                    "raw_black_BASE": raw_base,
                    "raw_black_v14b": raw_v14b,
                    "raw_delta_v14b_minus_BASE": delta,
                    "raw_positive_BASE": pos_base,
                    "raw_positive_v14b": pos_v14b,
                    **m,
                })
        print(f"[v15 phase0.5] {cid}: {len(picked)} branch(es), "
              f"root_mcts_black_value={fresh_black:+.4f}, checks OK")

    out_rows.sort(key=lambda r: (r["root_case_id"], r["branch_rank"],
                                 r["depth_from_root"], -r["visit_count"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(out_rows)

    summary = (aggregate_by_depth(out_rows, "full_subtree")
               + aggregate_by_depth([r for r in out_rows if r["is_pv_path"]],
                                    "pv_only"))
    with open(args.summary_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        w.writeheader()
        w.writerows(summary)

    print(f"\n[v15 phase0.5] by-depth summary "
          f"(pct_visit_mass_raw_positive is the decision metric):")
    for rec in summary:
        print(f"  {rec['scope']:<13} d={rec['depth_from_root']:<3} "
              f"n={rec['nodes_count']:<5} scored={rec['raw_scored_nodes_count']:<5} "
              f"vmass={rec['total_visit_share_from_root']:.3f} "
              f"wmean_raw_BASE={rec['weighted_mean_raw_black_BASE']} "
              f"pct_vmass_raw_pos_BASE={rec['pct_visit_mass_raw_positive_BASE']}")
    print(f"\nwrote {len(out_rows)} node rows -> {args.out}")
    print(f"wrote {len(summary)} summary rows -> {args.summary_out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
