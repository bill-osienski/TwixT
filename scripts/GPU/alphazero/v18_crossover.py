"""v18 static first-order crossover analysis -- spec Sec 2.2.1.

Answers one narrow question about a FINISHED shipped tree: at a depth-1 node,
how many unvisited priors outscore the best visited child under `_select_child`,
and how does that count move if the depth-2 backups had been clipped?

**WHAT THIS ANALYSIS CANNOT DO.** It is FIRST-ORDER and STATIC. It substitutes a
clipped initial backup into the tree that shipped search actually produced, and
re-scores selection once. It does **not** reproduce sequential selection once the
candidate tree diverges: a real capped run would visit different nodes, expand
different leaves and accumulate different statistics from the first clipped
backup onward. Consequently it **cannot empirically derive conversion
efficiency** -- how many predicted reply reductions become real depth-3 visits --
and no downstream task may read it as if it had. It predicts a direction and a
first-order magnitude, nothing more.

**The delta is SIGNED.** The clip is symmetric, so a large negative residual
raises a visited child's counterfactual value in the leaf's perspective, which
LOWERS its parent-side score and makes unvisited replies relatively more
attractive -- more scanning, not less. A negative `predicted_reply_reduction` is
a meaningful scientific result, not invalid input. Nothing here clamps it.

Read-only: no node is mutated, no search runs, no evaluator is touched.
"""
from __future__ import annotations

import math
from typing import Dict, List

from .mcts import MCTSNode
from .v18_provisional_backup import provisional_depth2_backup_value
from .v18_tree_walk import eligible_depth2_pairs

# The only search execution mode on which the counterfactual substitution below
# is exact. See `assert_synchronous_tree`.
SYNCHRONOUS_MODE = "synchronous"


def assert_synchronous_tree(root: MCTSNode, expected_sims: int, *,
                            search_execution_mode) -> None:
    """Refuse a tree whose provenance does not license the substitution.

    Spec Sec 2.2.1. BOTH conditions are required:

      1. `root.visit_count == expected_sims`
      2. `search_execution_mode == "synchronous"` exactly

    **The visit count alone is not evidence of provenance.** Both search entry
    points back up exactly one path per simulation, so a batched-waiter tree
    satisfies condition 1 identically -- the count proves the simulation BUDGET
    and nothing about the route. The condition that actually matters is
    invisible in the count: `search_from_root` backs up ALL waiters on a pending
    leaf with the same expansion value (`mcts.py:595-606`), so a leaf with `k`
    waiters accumulates `k * nn_value` into `value_sum` while
    `counterfactual_child_q` substitutes exactly one -- wrong by
    `(k-1) * (backup_value - nn_value)`, silently. `search_with_root` is "the
    same synchronous per-sim path as search(); NOT search_from_root's batched
    waiter path" (`mcts.py:528-535`), and is the only path on which the
    substitution is exact.

    `search_execution_mode` is therefore an INPUT, proven by the caller's route
    (the measurement CLI asserts it against a constant bound to its single
    `search_with_root` call site). Nothing here may infer it from node state.

    It is required and keyword-only with NO default: a default of
    `"synchronous"` would reinstate the very hole this closes, silently
    asserting the safe value for every caller that forgot the argument.
    """
    if search_execution_mode != SYNCHRONOUS_MODE:
        raise ValueError(
            f"refusing a tree from search_execution_mode "
            f"{search_execution_mode!r}: the counterfactual substitution is "
            f"exact only on {SYNCHRONOUS_MODE!r} provenance, and the visit "
            f"count cannot distinguish the two paths")
    if root.visit_count != expected_sims:
        raise ValueError(
            f"root.visit_count {root.visit_count} != expected_sims "
            f"{expected_sims}: this tree did not run the frozen budget")


def counterfactual_child_q(parent: MCTSNode, child: MCTSNode,
                           cap: float) -> float:
    """Child Q under the counterfactual that its FIRST backup had been clipped.

    A visited child's q_value is a running mean, not its raw evaluation, so the
    clipped value must be substituted into the accumulated sum. EXACT because
    the expansion backup contributed precisely `child.nn_value` to
    `child.value_sum` (mcts.py:1145-1148) -- see `assert_synchronous_tree` for
    the prerequisite that makes that true. A batched-waiter tree can back one
    expansion to several waiters, and the substitution would NOT be exact there.
    """
    pb = provisional_depth2_backup_value(child.nn_value, parent.nn_value, cap)
    return (child.value_sum - child.nn_value + pb.backup_value) / child.visit_count


def _is_synthetic(node: MCTSNode) -> bool:
    """Expanded with no legal moves: `_expand_batch` gives it a synthetic
    nn_value 0.0 (mcts.py:925-927). A cap never acts on such a leaf."""
    return not node.priors


def crossover_at_node(parent: MCTSNode, cap: float, c_puct: float) -> Dict:
    """First-order reply-scanning prediction at one depth-1 node.

    Mirrors `_select_child` (mcts.py:1062, 1091-1114) exactly:

        sqrt_parent          = sqrt(parent.visit_count + 1)
        visited_score(child) = -child.q + c_puct * prior * sqrt_parent
                                          / (1 + child.visit_count)
        unvisited_score(m)   = fpu(0.0) + c_puct * prior[m] * sqrt_parent

    `-child.q` is the perspective flip `_select_child` applies; the shipped FPU
    for an unvisited move is `MCTSConfig.fpu_value == 0.0`. Candidates come from
    `parent.priors`, because that is what `_select_child` iterates.

    Returns the best visited score under shipped and under the cap, the count of
    unvisited priors exceeding each, and the exclusion counts.
    """
    sqrt_parent = math.sqrt(parent.visit_count + 1)
    priors = parent.priors or {}

    excluded_terminal = excluded_synthetic = 0
    shipped_best = capped_best = float("-inf")

    for move_id, child in parent.children.items():
        if child.visit_count <= 0:
            continue
        if child.state.is_terminal():
            excluded_terminal += 1
            continue
        if _is_synthetic(child):
            excluded_synthetic += 1
            continue
        u = c_puct * priors.get(move_id, 0.0) * sqrt_parent / (1 + child.visit_count)
        shipped_best = max(shipped_best, -child.q_value + u)
        capped_best = max(
            capped_best, -counterfactual_child_q(parent, child, cap) + u)

    unvisited = [prior for move_id, prior in priors.items()
                 if move_id not in parent.children
                 or parent.children[move_id].visit_count <= 0]

    def _exceeding(best: float) -> int:
        # No eligible visited child means no crossover to measure at this node.
        if best == float("-inf"):
            return 0
        return sum(1 for prior in unvisited
                   if c_puct * prior * sqrt_parent > best)

    return {
        "shipped_best_visited_score": shipped_best,
        "capped_best_visited_score": capped_best,
        "predicted_shipped_replies": _exceeding(shipped_best),
        "predicted_capped_replies": _exceeding(capped_best),
        "excluded_terminal": excluded_terminal,
        "excluded_synthetic": excluded_synthetic,
    }


def crossover_for_tree(root: MCTSNode, cap: float, c_puct: float) -> Dict:
    """Sum the per-node prediction over eligible depth-1 nodes.

    Eligible = carries at least one clip-eligible depth-2 leaf, which is exactly
    `v18_tree_walk.eligible_depth2_pairs` -- the same eligibility the walker
    uses, never a second definition.

    `predicted_reply_reduction` is SIGNED and never clamped. A zero shipped
    denominator raises rather than returning 0.0: no reply scanning to reduce
    makes the ratio undefined, and a fabricated zero would read as "the cap
    changed nothing".
    """
    seen: List[MCTSNode] = []
    for parent, _leaf in eligible_depth2_pairs(root):
        if not any(parent is p for p in seen):
            seen.append(parent)

    per_node = [crossover_at_node(parent, cap, c_puct) for parent in seen]
    shipped = sum(n["predicted_shipped_replies"] for n in per_node)
    capped = sum(n["predicted_capped_replies"] for n in per_node)

    if shipped == 0:
        raise ValueError(
            "predicted_shipped_replies is 0: the reply-reduction ratio is "
            "undefined, not 0.0 -- there is no scanning for a cap to reduce")

    return {
        "predicted_shipped_replies": shipped,
        "predicted_capped_replies": capped,
        "predicted_reply_delta": shipped - capped,
        "predicted_reply_reduction": (shipped - capped) / shipped,
        "per_node": per_node,
    }
