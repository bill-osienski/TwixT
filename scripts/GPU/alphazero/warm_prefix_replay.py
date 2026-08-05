"""Atlas warm-prefix replay and additive ladder -- design sections 2b and 4.

Phase 0 returned WARM_START_REQUIRED, so the atlas probes a root carrying
trajectory-compounded inheritance rather than a fresh one. This module replays a
corpus game's recorded moves through shipped searches, stops before the sampled
target, and runs the frozen four-leg additive ladder on the resulting warm root.

Immediate-parent replay is NOT sufficient and is not implemented: inheritance
compounds, so one parent search cannot reproduce a tree carried across a full
trajectory.

CPU-SAFE at import: no MLX, no scipy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .corpus_geometry import GameMeta
from .mcts import MCTSNode, encode_move, visit_leader_move

BOUNDARY_THRESHOLD = 320          # design section 4
LEG_INCREMENTS = (400, 1200, 1600, 3200)
NOMINAL_B = (400, 1600, 3200, 6400)


# ---------------------------------------------------------------------------
# Prefix replay
# ---------------------------------------------------------------------------

@dataclass
class PrefixStep:
    ply: int
    forced_move: int
    forced_child_visits: Optional[int]     # None == child absent, never 0
    inheritance_reset: bool
    zero_effective_inheritance: bool
    state_agrees: bool = False             # canonical agreement after advance


@dataclass
class PrefixResult:
    root: MCTSNode
    inherited_I: int
    steps: List[PrefixStep] = field(default_factory=list)
    reset_count: int = 0
    reset_rate: Optional[float] = None
    last_reset_ply: Optional[int] = None
    cache_clears: int = 0                  # one per advance_root, counted


def replay_seed_for(meta: GameMeta, base_seed: int) -> int:
    """Section 2b: replay_seed = base_seed + game_idx, VERIFIED against the
    sidecar's recorded seed rather than assumed."""
    want = base_seed + meta.game_id
    if meta.seed != want:
        raise ValueError(
            f"game {meta.game_id}: sidecar seed {meta.seed} != base_seed + "
            f"game_idx ({want}); the frozen replay seed identity is violated")
    return want


def replay_prefix(mcts, meta: GameMeta, move_history: Sequence[Tuple[int, int]],
                  target_ply: int, active_size: int = 24) -> PrefixResult:
    """Force `move_history[:target_ply]` through shipped searches.

    `mcts` MUST already carry random.Random(replay_seed); this function never
    constructs one, because section 2b requires a single stream across the
    prefix and every ladder leg.
    """
    from .game.twixt_state import TwixtState

    if target_ply < 0 or target_ply > len(move_history):
        raise ValueError(
            f"target_ply {target_ply} outside history of {len(move_history)} moves")
    if meta.n_moves != len(move_history):
        raise ValueError(
            f"metadata says n_moves={meta.n_moves} but history has "
            f"{len(move_history)} moves; the sidecar and replay disagree")

    root = MCTSNode(state=TwixtState(active_size=active_size,
                                     to_move=meta.start_player))
    steps: List[PrefixStep] = []
    cache_clears = 0
    for ply in range(target_ply):
        mcts.search_from_root(root, add_noise=False, ply=ply)
        move = tuple(move_history[ply])

        # Section 2b step 2: legality, then state agreement after the advance.
        if move not in root.state.legal_moves():
            raise ValueError(
                f"ply {ply}: recorded move {move} is not legal in the replayed "
                f"state; the replay has diverged from the source game")
        expected_state = root.state.apply_move(move)

        child = root.children.get(encode_move(move[0], move[1]))
        visits = None if child is None else child.visit_count
        reset = child is None

        root = mcts.advance_root(root, move)
        # Canonical agreement over (to_move, pegs, bridges) -- an inherited
        # child's state must equal the independently applied move. A silent
        # divergence here would make every downstream measurement describe a
        # different game.
        if root.state != expected_state or hash(root.state) != hash(expected_state):
            raise ValueError(
                f"ply {ply}: state disagreement after advance_root; the "
                f"inherited child does not match the applied recorded move")

        steps.append(PrefixStep(
            ply=ply, forced_move=encode_move(move[0], move[1]),
            forced_child_visits=visits, inheritance_reset=reset,
            # Unions absent-or-zero WITHOUT collapsing the pair: the fields
            # above keep None and 0 distinct.
            zero_effective_inheritance=(visits is None or visits == 0),
            state_agrees=True,
        ))

        # Cache lifetime (design section 8): a detached subtree frees id()
        # values for reuse, so a longer-lived rank cache would silently return
        # another node's ranks. Cleared at EVERY advance, and counted so a test
        # can prove every boundary cleared -- final emptiness alone would pass
        # if only the last advance did.
        tracer = getattr(mcts, "_selection_observer", None)
        if tracer is not None and hasattr(tracer, "clear_node_cache"):
            tracer.clear_node_cache()
            cache_clears += 1

    resets = [s.ply for s in steps if s.inheritance_reset]
    return PrefixResult(
        root=root, inherited_I=root.visit_count, steps=steps,
        reset_count=len(resets),
        reset_rate=(len(resets) / len(steps)) if steps else None,
        last_reset_ply=(resets[-1] if resets else None),
        cache_clears=cache_clears,
    )


# ---------------------------------------------------------------------------
# Frozen captures (design section 6a) and the reference-line contract (4)
# ---------------------------------------------------------------------------

def capture_tree_state(root: MCTSNode) -> Dict[str, Any]:
    """The COMPLETE compact capture (section 6a), taken at ONE instant.

    D3 sums visit_count over nodes at depth EXACTLY 3 below `root`: every backup
    reaching depth >=3 passes through exactly one of them, so the sum counts each
    such backup once. Selection events cannot substitute -- they are edge
    traversals, and one deep simulation emits several.

    Everything else here exists because the ladder mutates this root in place:
    after leg 4 it describes 6,400 simulations, so any value not frozen now is
    unrecoverable.
    """
    d3 = 0
    frontier: List[Tuple[MCTSNode, int]] = [(root, 0)]
    while frontier:
        node, depth = frontier.pop()
        if depth == 3:
            d3 += node.visit_count
            continue                     # deeper nodes are already counted here
        if depth < 3:
            frontier.extend((c, depth + 1) for c in node.children.values())

    visited = [c for c in root.children.values() if c.visit_count > 0]
    counts = sorted((c.visit_count for c in visited), reverse=True)
    leader = visit_leader_move(root)
    leader_breadth = (len(root.children[leader].children)
                      if leader is not None and leader in root.children else None)

    priors = [p for p in (root.priors or {}).values() if p > 0]
    entropy = None
    if len(priors) >= 2:
        s = sum(priors)
        norm = [p / s for p in priors]
        # Normalized by log(n_legal), the existing convention.
        entropy = (-sum(q * math.log(q) for q in norm)) / math.log(len(root.priors))

    return {
        "D3": d3,
        # K(n)'s EFFECTIVE n at a warm root -- I + backups, not the nominal budget.
        "root_visits": root.visit_count,
        "total_child_visits": sum(counts),
        "top_child_visits": counts[0] if counts else None,
        "second_child_visits": counts[1] if len(counts) >= 2 else None,
        "n_visited_children": len(visited),
        "one_visit_children": sum(1 for c in visited if c.visit_count == 1),
        "leader_move": leader,
        # Children OF THE LEADER, not of the root -- a different quantity.
        "leader_breadth": leader_breadth,
        "policy_entropy": entropy,
        "n_legal": len(root.priors or {}),
    }


def capture_parent_visits(root: MCTSNode, max_depth: int = 2
                          ) -> Dict[Tuple[int, ...], int]:
    """Visit counts for EVERY existing path through depth `max_depth`, at ONE
    instant (amendment 4).

    A complete map rather than a lookup of the reference line's paths, because
    the union of required edges is not known until leg 4 while this is captured
    during leg 1. A path ABSENT from the map has zero visits, where
    K(0) = min(n_legal, max(1, 0)) = 1 admits only rank 1.

    Bounded by the nodes that exist, not by the branching factor: at most
    `I + N_actual + 1` paths can exist at this instant, so this is hundreds of
    entries -- not the 250k a 500-wide two-ply enumeration would suggest.

    Keys are TUPLES, the natural type here. `build_atlas_corpus._jsonable`
    normalizes them at the artifact boundary; the pure modules keep tuples.
    """
    out: Dict[Tuple[int, ...], int] = {}
    frontier: List[Tuple[MCTSNode, Tuple[int, ...]]] = [(root, ())]
    while frontier:
        node, path = frontier.pop()
        out[path] = node.visit_count
        if len(path) < max_depth:
            frontier.extend((c, path + (mv,)) for mv, c in node.children.items())
    return out


def deep_reference_line(root: MCTSNode) -> Dict[str, Any]:
    """The two-ply reference line at ONE instant (amendment 4).

    Called at the end of leg 3 and again at the end of leg 4: the ladder is
    additive on ONE tree, so after leg 4 the 3,200 state exists nowhere and a
    single post-ladder summary could only ever describe 6,400.

    Emits EDGES -- `(parent_path, move)` plus the parent's priors, which is what
    ranks are computed from. It deliberately does NOT emit the parent's visit
    count: that is a deep-rung number, and retention must key on the boundary and
    B=400 maps instead. Not producing it is what makes the earlier defect
    unreachable rather than merely discouraged.
    """
    edges: List[Dict[str, Any]] = []
    node: Optional[MCTSNode] = root
    path: Tuple[int, ...] = ()
    for _ in range(3):                       # root move, reply, two-ply
        if node is None:
            break
        mv = visit_leader_move(node)         # canonical: ties by lowest move id
        if mv is None:
            break                            # nothing below here has a visit
        edges.append({
            "parent_path": path, "move": mv, "depth": len(path),
            "parent_priors": dict(node.priors or {}),
        })
        path = path + (mv,)
        node = node.children.get(mv)
    return {"edges": edges, "moves": [e["move"] for e in edges]}


DEPTH_NAMES = ("root", "reply", "two_ply")


def merge_reference_lines(line_3200: Dict[str, Any], line_6400: Dict[str, Any]
                          ) -> Dict[str, Any]:
    """The DEDUPLICATED UNION of both deep lines' edges (amendment 4).

    Agreements collapse to one edge; disagreements retain BOTH. Neither rung is
    declared truth and neither is called stable -- the point of the 3,200/6,400
    pair is that agreement is a FINDING, not an assumption.
    """
    by_key: Dict[Tuple[Tuple[int, ...], int], Dict[str, Any]] = {}
    priors_by_parent: Dict[Tuple[int, ...], Dict[int, float]] = {}
    at_depth: Dict[int, Dict[int, Tuple[Tuple[int, ...], int]]] = {}

    for rung, line in ((3200, line_3200), (6400, line_6400)):
        for e in line["edges"]:
            path = e["parent_path"]
            # Priors cannot change between rungs under the frozen
            # add_noise=False ladder, so ASSERT rather than assume. Keyed on the
            # PARENT, not the edge: two different edges can share a parent, and
            # that is exactly the case worth checking.
            prev = priors_by_parent.get(path)
            if prev is not None and prev != e["parent_priors"]:
                raise ValueError(
                    f"parent priors differ between deep rungs at path {path}; "
                    f"under the frozen add_noise=False ladder they cannot")
            priors_by_parent[path] = e["parent_priors"]

            key = (path, e["move"])
            if key in by_key:
                by_key[key]["sources"] = by_key[key]["sources"] + (rung,)
            else:
                by_key[key] = {**e, "sources": (rung,)}
            at_depth.setdefault(e["depth"], {})[rung] = key

    agreement: Dict[str, Any] = {}
    for depth, name in enumerate(DEPTH_NAMES):
        seen = at_depth.get(depth, {})
        in32, in64 = 3200 in seen, 6400 in seen
        if in32 and in64:
            # Agreement is equality of the COMPLETE edge: the same move id
            # under a different parent is a different edge.
            state = "agree" if seen[3200] == seen[6400] else "disagree"
        elif in32 or in64:
            # Amendment 4's "present in only one line": counted separately,
            # outside the agreement denominator. There is nothing to compare.
            state = "single_line"
        else:
            # Neither line reached this depth. A DIFFERENT missingness state --
            # collapsing it into single_line would report a comparison that was
            # never even half-available. Also outside the denominator.
            state = "absent_both"
        agreement[name] = {"in_3200": in32, "in_6400": in64, "state": state}

    return {
        # Sorted by (parent_path, move) so the union is reproducible run to run.
        "required_edges": [by_key[k] for k in sorted(by_key)],
        "agreement": agreement,
    }


def check_backup_invariant(d3_start: int, d3_boundary: int,
                           n_actual: int) -> bool:
    """Section 6a. A violation is a broken accounting, not a datum."""
    delta = d3_boundary - d3_start
    if delta < 0 or delta > n_actual:
        raise ValueError(
            f"backup accounting invariant violated: D3 delta {delta} outside "
            f"[0, {n_actual}]; the row must fail rather than be recorded")
    return True


# ---------------------------------------------------------------------------
# Batch-safe boundary
# ---------------------------------------------------------------------------

@dataclass
class BoundaryRecord:
    N_actual: int
    overshoot: int
    remaining: int
    flush_type: str


class BatchSafeBoundaryObserver:
    """Captures the FIRST flush completion at or after `threshold` completed
    TARGET-search backups (design section 4).

    A raw backup count will not do: at backup 320 the search is mid-flush, with
    expansions from later batch members already visible and up to
    eval_batch_size simulations queued as unredirectable waiters. Right after a
    flush's clears, the in-flight set is provably empty -- the only quiescent
    point in the loop.
    """

    def __init__(self, inherited_I: int, threshold: int = BOUNDARY_THRESHOLD,
                 leg_B: int = 400, tracer=None) -> None:
        if inherited_I < 0:
            raise ValueError("inherited_I must be non-negative")
        self._I = inherited_I
        self._threshold = threshold
        self._leg_B = leg_B
        self._tracer = tracer
        self.record: Optional[BoundaryRecord] = None
        # Section 8's FIRST frozen snapshot, taken at exactly the quiescent
        # boundary moment. Taking it later would describe a different tree.
        self.tracer_snapshot_at_boundary: Optional[Dict[str, Any]] = None
        # The boundary instant is the OBSERVER's to freeze: by the time
        # run_additive_ladder regains control, leg 1 has already finished.
        self.capture_at_boundary: Optional[Dict[str, Any]] = None
        self.parent_visits_at_boundary: Optional[Dict[Tuple[int, ...], int]] = None

    def on_flush_complete(self, flush_type: str, root: Any) -> None:
        if self.record is not None:
            return                                # first one wins
        n_actual = root.visit_count - self._I      # excludes inherited visits
        if n_actual < self._threshold:
            return
        assert self._threshold <= n_actual <= self._leg_B, (
            f"N_actual {n_actual} outside the frozen "
            f"[{self._threshold}, {self._leg_B}] range")
        self.record = BoundaryRecord(
            N_actual=n_actual,
            overshoot=n_actual - self._threshold,
            remaining=self._leg_B - n_actual,
            flush_type=flush_type,
        )
        if self._tracer is not None:
            self.tracer_snapshot_at_boundary = self._tracer.snapshot()
        # Frozen at the SAME instant as the tracer snapshot, by value.
        self.capture_at_boundary = capture_tree_state(root)
        self.parent_visits_at_boundary = capture_parent_visits(root)


# ---------------------------------------------------------------------------
# Additive ladder
# ---------------------------------------------------------------------------

@dataclass
class LegResult:
    """A rung's evidence, captured BEFORE the tree advances past it.

    The ladder is additive on ONE tree, so by the end of leg 4 the 400/1,600/3,200
    states no longer exist anywhere. Everything sections 5 and 7 need must be
    frozen here or it is unrecoverable: section 5 wants V_B and the 6,400 top-two
    margin; section 7 wants effective children, top share, and the selected
    move's prior RANK for the lower-prior-flip gate.
    """
    nominal_B: int
    inherited_I: int
    effective: int
    root_value: float
    selected_move: Optional[int]
    selected_move_prior_rank: Optional[int]
    top_share: Optional[float]
    top_two_margin: Optional[float]
    effective_children: Optional[float]
    n_visited_children: int
    visit_counts: Dict[int, int]           # compact: NONZERO entries only


def _root_summary(root: MCTSNode, visit_counts: Dict[Any, int],
                  selected_move: Optional[int]) -> Dict[str, Any]:
    """Derive section 5 / section 7 metrics from a root, at this rung.

    Undefined statistics are None, never 0.0 -- an empty or single-child
    distribution has no top-two margin, and that is a different fact from a
    margin of zero.
    """
    nonzero = {encode_move(r, c): v for (r, c), v in visit_counts.items() if v > 0}
    total = sum(nonzero.values())
    ordered = sorted(nonzero.values(), reverse=True)

    top_share = (ordered[0] / total) if (ordered and total) else None
    top_two_margin = (((ordered[0] - ordered[1]) / total)
                      if (len(ordered) >= 2 and total) else None)
    if total:
        # exp(entropy) -- the section 7 "effective children" metric.
        ent = -sum((v / total) * math.log(v / total) for v in nonzero.values())
        eff_children = math.exp(ent)
    else:
        eff_children = None

    # Prior rank of the selected move: adjusted prior DESCENDING, move-ID
    # ASCENDING -- the same frozen order the selection tracer uses.
    rank = None
    if selected_move is not None and root.priors:
        order = sorted(root.priors.items(), key=lambda kv: (-kv[1], kv[0]))
        for i, (mv, _p) in enumerate(order, start=1):
            if mv == selected_move:
                rank = i
                break

    return {
        "visit_counts": nonzero,
        "n_visited_children": len(nonzero),
        "top_share": top_share,
        "top_two_margin": top_two_margin,
        "effective_children": eff_children,
        "selected_move_prior_rank": rank,
    }


def run_additive_ladder(mcts, root: MCTSNode, inherited_I: int, ply: int,
                        boundary_observer=None, target_tracer=None,
                        increments: Sequence[int] = LEG_INCREMENTS
                        ) -> Tuple[List[LegResult], Dict[str, Any]]:
    """Four ADDITIVE legs on ONE tree. `mcts` keeps its RNG throughout.

    Returns (legs, snapshots). `snapshots` carries section 8's two frozen
    target-search tracer snapshots: "at_boundary" (taken by the boundary
    observer at N_actual) and "at_400" (taken here immediately after leg 1).

    `target_tracer` MUST be a fresh tracer attached AFTER prefix replay. A tracer
    that ran through the prefix would have its counters contaminated by unrelated
    searches, and section 8's statistics are about the TARGET search alone.

    The boundary observer is attached for leg 1 only: the 320-completion prefix
    lives inside the first 400-simulation leg (design section 4).
    """
    if len(increments) != 4:
        raise ValueError(f"the frozen ladder has four legs, got {len(increments)}")

    if target_tracer is not None:
        # Section 8's statistics are about the TARGET search alone. A tracer that
        # ran through prefix replay carries counters from unrelated searches, and
        # nothing downstream could tell the difference -- so refuse it here.
        if getattr(mcts, "_selection_observer", None) is not target_tracer:
            raise ValueError(
                "target_tracer must be the MCTS's current selection observer; "
                "attach it AFTER prefix replay")
        pre = target_tracer.snapshot()
        if any(pre["by_shape"][s]["overall"]["eligible_events"]
               for s in pre["by_shape"]):
            raise ValueError(
                "target_tracer is not empty: it already accumulated events, so "
                "its snapshots would be contaminated by prefix replay")

    legs: List[LegResult] = []
    snapshots: Dict[str, Any] = {
        "at_boundary": None, "at_400": None,
        # Frozen captures and amendment 4's producer output. Every entry is
        # taken at the instant it names -- the ladder mutates ONE tree, so a
        # value not frozen there is unrecoverable.
        "captures": {"at_start": capture_tree_state(root),
                     "at_boundary": None, "at_400": None},
        "parent_visits": {"at_boundary": None, "at_400": None},
        "reference_lines": {"at_3200": None, "at_6400": None, "merged": None},
    }
    running_B = 0
    original_n = mcts.config.n_simulations
    original_flush_obs = getattr(mcts, "_flush_observer", None)
    try:
        for leg_idx, inc in enumerate(increments):
            mcts.config.n_simulations = inc
            # Leg 1 only -- the 320 prefix is inside the first leg.
            mcts._flush_observer = boundary_observer if leg_idx == 0 else None
            visit_counts, root_value, root = mcts.search_from_root(
                root, add_noise=False, ply=ply)
            running_B += inc
            # CANONICAL leader: max visits, ties by lowest encoded move id.
            # max(visit_counts, key=...) would break ties by dict insertion
            # order, so stable-reference labels and lower-prior-flip metrics
            # could change between runs. Section 9 names this trap explicitly.
            sel = visit_leader_move(root)
            summary = _root_summary(root, visit_counts, sel)
            legs.append(LegResult(
                nominal_B=running_B, inherited_I=inherited_I,
                effective=inherited_I + running_B,
                root_value=float(root_value), selected_move=sel,
                selected_move_prior_rank=summary["selected_move_prior_rank"],
                top_share=summary["top_share"],
                top_two_margin=summary["top_two_margin"],
                effective_children=summary["effective_children"],
                n_visited_children=summary["n_visited_children"],
                visit_counts=summary["visit_counts"],
            ))
            if leg_idx == 0:
                if target_tracer is not None:
                    # Section 8: the SECOND frozen snapshot, at nominal B = 400.
                    snapshots["at_400"] = target_tracer.snapshot()
                snapshots["captures"]["at_400"] = capture_tree_state(root)
                snapshots["parent_visits"]["at_400"] = capture_parent_visits(root)
            # Select the deep rungs by LEG INDEX, never by running_B: CPU tests
            # run tiny increments where running_B never reaches 3,200, and a
            # budget test would silently capture no deep line at all.
            if leg_idx == 2:
                snapshots["reference_lines"]["at_3200"] = deep_reference_line(root)
            elif leg_idx == 3:
                snapshots["reference_lines"]["at_6400"] = deep_reference_line(root)
    finally:
        mcts.config.n_simulations = original_n
        mcts._flush_observer = original_flush_obs
    if boundary_observer is not None:
        snapshots["at_boundary"] = boundary_observer.tracer_snapshot_at_boundary
        snapshots["captures"]["at_boundary"] = boundary_observer.capture_at_boundary
        snapshots["parent_visits"]["at_boundary"] = (
            boundary_observer.parent_visits_at_boundary)
    lines = snapshots["reference_lines"]
    if lines["at_3200"] is not None and lines["at_6400"] is not None:
        lines["merged"] = merge_reference_lines(lines["at_3200"],
                                                lines["at_6400"])
    return legs, snapshots


# ---------------------------------------------------------------------------
# Runtime projection
# ---------------------------------------------------------------------------

def project_runtime(rows: int, mean_prefix_plies: float,
                    tracer_overhead: float = 0.010) -> Dict[str, Any]:
    """Simulation-count projection for the atlas run.

    `tracer_overhead` defaults to Stage 1's MEASURED +1.0% (400 sims, min-of-5,
    FakeEvaluator). It is an UPPER BOUND: FakeEvaluator isolates tracer cost from
    NN cost, so a real evaluator's inference dominates and the true figure is
    lower. `mean_prefix_plies` has no default -- it must be measured from the
    corpus's actual ply distribution, never assumed (design section 4).
    """
    ladder = sum(LEG_INCREMENTS)
    prefix = int(round(mean_prefix_plies * 400))
    return {
        "rows": rows,
        "ladder_sims_per_row": ladder,
        "prefix_sims_per_row": prefix,
        "sims_per_row": ladder + prefix,
        "total_sims": rows * (ladder + prefix),
        "dominant_term": "prefix_replay" if prefix > ladder else "ladder",
        "tracer_overhead": tracer_overhead,
        "tracer_overhead_is_upper_bound": True,
        "note": "Prefix cost scales with target ply and is dominated by late "
                "rows. Derive mean_prefix_plies from the corpus's observed "
                "per-phase ply supply, never from a smoke.",
    }
