"""Pure tree-walk extraction of searched-continuation states (v6).

Consumes a root MCTSNode from MCTS.search_with_root (gate-faithful search) and
returns ContinuationSpecs per the spec's tag-based rules:
  B goal_line_retention          -> PV depth 1-2
  C old_post_opening_retention   -> PV depth 1-3
  D red_predrop_retention        -> top-k root children (+1 PV step below a
                                    child only when its subtree visits pass
                                    d_child_pv_min_visits)
Eligibility: a node is extractable only if it was expanded during search
(is_expanded) and is non-terminal (has legal moves). tree_nn_value comes from
the TRAIN-mode search evaluator — provenance only, never a training target.
NO MLX imports here: the builder's tests run with fakes.
"""
from __future__ import annotations

from dataclasses import dataclass

from .mcts import decode_move

FAMILY_BY_SOURCE_TAG = {
    "goal_line_retention": "B",
    "old_post_opening_retention": "C",
    "red_predrop_retention": "D",
}
CONTINUATION_TAG_BY_SOURCE_TAG = {
    "goal_line_retention": "goal_line_continuation_retention",
    "old_post_opening_retention": "old_post_opening_continuation_retention",
    "red_predrop_retention": "red_predrop_continuation_retention",
}


@dataclass(frozen=True)
class ContinuationSpec:
    path_moves: tuple            # ((r, c), ...) root -> continuation
    source: str                  # "pv" | "top_child" | "child_pv"
    depth: int                   # == len(path_moves)
    tree_visits: int
    tree_nn_value: float | None  # train-mode BN; provenance ONLY
    state: object                # TwixtState at the continuation


def _eligible(node) -> bool:
    """Expanded during search and non-terminal."""
    return node.is_expanded and len(node.state.legal_moves()) > 0


def _best_child(node):
    """Max-visit child (ties: lowest encoded move id); None if no visited child."""
    visited = [c for c in node.children.values() if c.visit_count > 0]
    if not visited:
        return None
    return min(visited, key=lambda c: (-c.visit_count, c.move))


def _top_children(node, k: int) -> list:
    visited = [c for c in node.children.values() if c.visit_count > 0]
    return sorted(visited, key=lambda c: (-c.visit_count, c.move))[:k]


def path_moves_of(node) -> tuple:
    """(r, c) moves from the root to this node, via parent links."""
    moves = []
    while node.parent is not None:
        moves.append(decode_move(node.move))
        node = node.parent
    return tuple(reversed(moves))


def format_path_moves(path_moves) -> str:
    return ">".join(f"{r}:{c}" for r, c in path_moves)


def case_path_token(path_moves) -> str:
    return "_".join(f"{r}-{c}" for r, c in path_moves)


def continuation_case_id(parent_case_id: str, spec: ContinuationSpec) -> str:
    return (f"{parent_case_id}__cont_{spec.source}{spec.depth}_"
            f"{case_path_token(spec.path_moves)}")


def root_max_visit_share(root) -> float:
    total = sum(c.visit_count for c in root.children.values())
    if total <= 0:
        return 0.0
    return max(c.visit_count for c in root.children.values()) / total


def _spec_for(node, source: str) -> ContinuationSpec:
    path = path_moves_of(node)
    return ContinuationSpec(path_moves=path, source=source, depth=len(path),
                            tree_visits=node.visit_count,
                            tree_nn_value=node.nn_value, state=node.state)


def _pv_specs(root, max_depth: int) -> list:
    specs, node = [], root
    for _ in range(max_depth):
        child = _best_child(node)
        if child is None or not _eligible(child):
            break
        specs.append(_spec_for(child, "pv"))
        node = child
    return specs


def extract_continuations(root, source_tag: str, *, b_pv_depth: int = 2,
                          c_pv_depth: int = 3, d_top_k: int = 3,
                          d_child_pv_depth: int = 1,
                          d_child_pv_min_visits: int = 40,
                          max_per_root: int = 6) -> list:
    family = FAMILY_BY_SOURCE_TAG.get(source_tag)
    if family is None:
        raise ValueError(f"not an extraction-source tag: {source_tag!r}")
    if family == "B":
        specs = _pv_specs(root, b_pv_depth)
    elif family == "C":
        specs = _pv_specs(root, c_pv_depth)
    else:                                   # D
        specs = []
        for child in _top_children(root, d_top_k):
            if not _eligible(child):
                continue
            specs.append(_spec_for(child, "top_child"))
            if child.visit_count < d_child_pv_min_visits:
                continue
            node = child
            for _ in range(d_child_pv_depth):
                grand = _best_child(node)
                if grand is None or not _eligible(grand):
                    break
                specs.append(_spec_for(grand, "child_pv"))
                node = grand
    if len(specs) > max_per_root:
        raise ValueError(
            f"{source_tag}: {len(specs)} continuations exceed max_per_root "
            f"{max_per_root}")
    return specs
