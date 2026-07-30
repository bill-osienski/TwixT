"""v18 read-only tree walker -- derives every v18 metric from a FINISHED tree.

Spec: docs/superpowers/specs/2026-07-29-v18-depth2-provisional-backup-design.md
Sec 4.4 (telemetry columns), Sec 10.1.1 (depth accounting).

READ-ONLY. Nothing here mutates a node, runs a simulation, touches an evaluator
or reads a cap out of `MCTSConfig`. It answers "what WOULD a cap have done to
this tree", which is why it is meaningful on a shipped tree that never clipped.

Every clip decision comes from `v18_provisional_backup`; this module never
re-derives the formula (spec Sec 4.2: exactly one implementation).

Perspective: `parent.nn_value` is parent-to-move, `leaf.nn_value` is
leaf-to-move, and the residual is `leaf - (-parent)`. See the helper module.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .mcts import MCTSNode, visit_leader_move
from .v18_provisional_backup import (
    IDENTITY_CAP,
    ProvisionalBackup,
    provisional_depth2_backup_value,
)


def terminating_backups(node: MCTSNode) -> int:
    """Simulations that ENDED at this node.

    Exact on the synchronous path, which adds no virtual visits to
    `visit_count`; the batched path is out of v18 scope (spec Sec 4.3).
    """
    return node.visit_count - sum(c.visit_count for c in node.children.values())


def _iter_with_depth(root: MCTSNode):
    """Breadth-first `(node, depth)` over the whole tree, insertion-ordered."""
    frontier = [(root, 0)]
    while frontier:
        nxt = []
        for node, depth in frontier:
            yield node, depth
            nxt.extend((child, depth + 1) for child in node.children.values())
        frontier = nxt


def depth_terminating_histogram(root: MCTSNode) -> Dict[int, int]:
    """Depth -> simulations terminating there. Sums to `root.visit_count`:
    every node's visits are counted once and subtracted once as a child."""
    hist: Dict[int, int] = {}
    for node, depth in _iter_with_depth(root):
        hist[depth] = hist.get(depth, 0) + terminating_backups(node)
    return hist


def _is_finite_value(value) -> bool:
    # bool is an int subclass; a bool nn_value is corruption, not a value.
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _leaf_eligible(parent: MCTSNode, leaf: MCTSNode) -> bool:
    """Spec Sec 3.2. A bypassed leaf is not an error -- it backs up unchanged."""
    if leaf.state.is_terminal():
        return False
    if not leaf.priors:                    # no legal moves -> synthetic nn_value
        return False
    return _is_finite_value(leaf.nn_value) and _is_finite_value(parent.nn_value)


def eligible_depth2_pairs(root: MCTSNode) -> List[Tuple[MCTSNode, MCTSNode]]:
    """`(parent, leaf)` for every clip-eligible depth-2 leaf.

    Path is exactly `[root, parent, leaf]` -- depth >= 3 is never eligible.
    """
    pairs: List[Tuple[MCTSNode, MCTSNode]] = []
    for parent in root.children.values():
        if not parent.children:
            # No depth-2 leaves below it, so its priors are never consulted.
            # This branch also covers a terminal depth-1 node, which is never
            # expanded and therefore legitimately carries no priors.
            continue
        if not parent.priors:
            raise ValueError(
                f"depth-1 node move={parent.move!r} has {len(parent.children)} "
                "children but empty priors -- corruption, not an eligibility "
                "skip (spec Sec 3.2)")
        for leaf in parent.children.values():
            if _leaf_eligible(parent, leaf):
                pairs.append((parent, leaf))
    return pairs


def _decide(parent: MCTSNode, leaf: MCTSNode, cap: float) -> ProvisionalBackup:
    return provisional_depth2_backup_value(leaf.nn_value, parent.nn_value, cap)


def _decisions(root: MCTSNode,
               cap: float) -> List[Tuple[MCTSNode, MCTSNode, ProvisionalBackup]]:
    """One clip decision per eligible leaf. The single traversal every cap-aware
    metric below is built on, so none of them restates the formula."""
    return [(parent, leaf, _decide(parent, leaf, cap))
            for parent, leaf in eligible_depth2_pairs(root)]


def residual(parent: MCTSNode, leaf: MCTSNode) -> float:
    """Signed residual, from the single formula implementation. `IDENTITY_CAP`
    cannot bind, so this is the raw residual rather than a clipped one."""
    return _decide(parent, leaf, IDENTITY_CAP).residual


def would_clip(root: MCTSNode, cap: float) -> List[MCTSNode]:
    """Eligible leaves this cap WOULD have clipped. Well defined -- and
    generally non-empty -- on a shipped tree, which clipped nothing."""
    return [leaf for _p, leaf, d in _decisions(root, cap) if d.clip_direction]


def leader(root: MCTSNode) -> Optional[MCTSNode]:
    """The visit-leading depth-1 child, by the canonical v17 comparator."""
    move_id = visit_leader_move(root)
    return None if move_id is None else root.children[move_id]


def replies(leader_node: MCTSNode) -> int:
    """Breadth statistic imported from v17: visited children, terminals
    INCLUDED. Deliberately not the eligibility predicate."""
    return sum(1 for c in leader_node.children.values() if c.visit_count > 0)


def explored_replies(leader_node: MCTSNode) -> List[MCTSNode]:
    """Visited replies a cap could act on: nonterminal, expanded, finite."""
    return [c for c in leader_node.children.values()
            if c.visit_count > 0 and not c.state.is_terminal() and c.priors
            and _is_finite_value(c.nn_value)]


def follow_up_visits_per_explored_reply(leader_node: MCTSNode) -> float:
    """Mean visits AFTER the first touch. Zero explored replies makes the
    statistic undefined, not zero -- averaging it in would bias the arm."""
    explored = explored_replies(leader_node)
    if not explored:
        raise ValueError(
            "no explored replies under the leader: follow-up visits per reply "
            "is undefined, not 0.0")
    return sum(c.visit_count - 1 for c in explored) / len(explored)


def revisit_to_depth3_rate(root: MCTSNode, cap: float) -> float:
    """Fraction of the would-clip population that was visited again deeply."""
    population = would_clip(root, cap)
    if not population:
        raise ValueError(
            f"no leaf would clip at cap {cap!r}: the revisit rate is undefined, "
            "not 0.0")
    revisited = sum(1 for leaf in population
                    if any(c.visit_count > 0 for c in leaf.children.values()))
    return revisited / len(population)


def positive_mass(root: MCTSNode) -> float:
    return sum(max(0.0, residual(p, l)) for p, l in eligible_depth2_pairs(root))


def negative_mass(root: MCTSNode) -> float:
    return sum(max(0.0, -residual(p, l)) for p, l in eligible_depth2_pairs(root))


def sign_dominance(root: MCTSNode) -> float:
    """Share of residual mass that is positive. A tree with no residual mass at
    all scores 0.0 rather than raising: it is genuinely undominated."""
    pos, neg = positive_mass(root), negative_mass(root)
    total = pos + neg
    return pos / total if total > 0.0 else 0.0


def _require_searched(root: MCTSNode) -> int:
    if root.visit_count <= 0:
        raise ValueError("root has no visits: this tree was never searched")
    return root.visit_count


def contribution_weighted_positive_mass(root: MCTSNode, cap: float) -> float:
    """Positive clipped mass weighted by each leaf's share of the budget, so a
    large residual on a leaf that terminated one simulation counts once."""
    budget = _require_searched(root)
    return sum((terminating_backups(leaf) / budget) * d.clipped_amount
               for _p, leaf, d in _decisions(root, cap) if d.clip_direction == 1)


def exposed_positive_backup_mass(root: MCTSNode,
                                 cap: float) -> Tuple[float, float]:
    """`(numerator, denominator)`, NOT the ratio: the caller pools across rows
    rather than averaging per-row ratios.

    Denominator: `terminating_backups * max(0, raw_leaf)` over ALL eligible
    leaves. Numerator: the same weight restricted to leaves this cap clips down.
    """
    numerator = denominator = 0.0
    for _p, leaf, d in _decisions(root, cap):
        weight = terminating_backups(leaf) * max(0.0, leaf.nn_value)
        denominator += weight
        if d.clip_direction == 1:
            numerator += weight
    return numerator, denominator


def terminal_depth2_counts(root: MCTSNode) -> Tuple[int, int]:
    """`(terminal_visited, all_visited)` over the depth-2 children of every
    depth-1 node. Terminals are the population a cap can never touch."""
    terminal = total = 0
    for parent in root.children.values():
        for leaf in parent.children.values():
            if leaf.visit_count <= 0:
                continue
            total += 1
            if leaf.state.is_terminal():
                terminal += 1
    return terminal, total


def walk(root: MCTSNode, caps) -> Dict:
    """Every documented metric for one finished tree, JSON-serialisable and
    deterministic. `per_cap` is keyed by `str(cap)`.

    Undefined statistics are emitted as `null`, never as a fabricated 0.0: at a
    cap no leaf reaches, the revisit rate has an empty denominator.
    """
    budget = _require_searched(root)
    histogram = depth_terminating_histogram(root)
    depth_ge3 = sum(n for depth, n in histogram.items() if depth >= 3)
    leader_node = leader(root)
    explored = [] if leader_node is None else explored_replies(leader_node)
    terminal_d2, total_d2 = terminal_depth2_counts(root)

    record: Dict = {
        "root_visit_count": int(budget),
        "depth_terminating_histogram": {
            int(d): int(n) for d, n in sorted(histogram.items())},
        "depth_ge3_backups": int(depth_ge3),
        "depth_ge3_fraction": depth_ge3 / budget,
        "leader_move": None if leader_node is None else int(leader_node.move),
        "replies": None if leader_node is None else replies(leader_node),
        "explored_replies": len(explored),
        "follow_up_visits_per_reply": (
            follow_up_visits_per_explored_reply(leader_node) if explored else None),
        "eligible_depth2_leaves": len(eligible_depth2_pairs(root)),
        "positive_mass": positive_mass(root),
        "negative_mass": negative_mass(root),
        "sign_dominance": sign_dominance(root),
        "terminal_depth2": terminal_d2,
        "total_depth2": total_d2,
        "per_cap": {},
    }

    for cap in caps:
        clipped = [(leaf, d) for _p, leaf, d in _decisions(root, cap)
                   if d.clip_direction]
        numerator, denominator = exposed_positive_backup_mass(root, cap)
        record["per_cap"][str(cap)] = {
            "would_clip_count": len(clipped),
            "clipped_amount_total": sum(d.clipped_amount for _l, d in clipped),
            "positive_count": sum(1 for _l, d in clipped if d.clip_direction == 1),
            "negative_count": sum(1 for _l, d in clipped if d.clip_direction == -1),
            "revisit_to_depth3_rate": (
                revisit_to_depth3_rate(root, cap) if clipped else None),
            "contribution_weighted_positive_mass":
                contribution_weighted_positive_mass(root, cap),
            "exposed_positive_mass_numerator": numerator,
            "exposed_positive_mass_denominator": denominator,
        }
    return record
