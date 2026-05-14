"""Spec 4 — Recovery / re-targeting diagnostic.

Per-side, per-game tracker that detects collapse/re-targeting failure
patterns from MCTS root values and visit-share concentration. Diagnostic
only: does not affect MCTS, selection, or training targets.

Spec: docs/superpowers/specs/2026-05-12-recovery-retargeting-diagnostic-design.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RecoveryRetargetingConfig:
    enabled: bool = True
    collapse_value_threshold: float = -0.75
    severe_collapse_value_threshold: float = -0.90
    diffuse_root_top1_threshold: float = 0.20
    very_diffuse_root_top1_threshold: float = 0.15
    delta_threshold: float = 0.50
    delta_max_current_score: float = -0.30
    alternate_component_min_size: int = 4
    classify_defense: bool = True
    max_sampled_moves_per_side: int = 32
    sample_all_moves: bool = False


def validate_config(cfg: RecoveryRetargetingConfig) -> None:
    """Raise ValueError on out-of-band config. Called once at startup."""
    if not (cfg.collapse_value_threshold < cfg.delta_max_current_score):
        raise ValueError(
            f"collapse_value_threshold ({cfg.collapse_value_threshold}) must be "
            f"strictly less than delta_max_current_score ({cfg.delta_max_current_score}) "
            f"so the delta path doesn't subsume the steady-state path"
        )
    if not (cfg.severe_collapse_value_threshold <= cfg.collapse_value_threshold):
        raise ValueError(
            f"severe_collapse_value_threshold ({cfg.severe_collapse_value_threshold}) "
            f"must be <= collapse_value_threshold ({cfg.collapse_value_threshold})"
        )
    if not (cfg.very_diffuse_root_top1_threshold <= cfg.diffuse_root_top1_threshold):
        raise ValueError(
            f"very_diffuse_root_top1_threshold ({cfg.very_diffuse_root_top1_threshold}) "
            f"must be <= diffuse_root_top1_threshold ({cfg.diffuse_root_top1_threshold})"
        )
    if not (0.0 <= cfg.diffuse_root_top1_threshold <= 1.0):
        raise ValueError(
            f"diffuse_root_top1_threshold ({cfg.diffuse_root_top1_threshold}) must be in [0, 1]"
        )
    if not (0.0 <= cfg.very_diffuse_root_top1_threshold <= 1.0):
        raise ValueError(
            f"very_diffuse_root_top1_threshold ({cfg.very_diffuse_root_top1_threshold}) must be in [0, 1]"
        )
    if not (cfg.delta_threshold > 0):
        raise ValueError(f"delta_threshold ({cfg.delta_threshold}) must be > 0")
    if not (cfg.alternate_component_min_size >= 1):
        raise ValueError(
            f"alternate_component_min_size ({cfg.alternate_component_min_size}) must be >= 1"
        )
    if not (cfg.max_sampled_moves_per_side >= 0):
        raise ValueError(
            f"max_sampled_moves_per_side ({cfg.max_sampled_moves_per_side}) must be >= 0"
        )


# ---------------------------------------------------------------------------
# Component analysis helpers
# ---------------------------------------------------------------------------

_KNIGHT_OFFSETS = ((1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1))


def knight_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
    """The 8 TwixT knight-distance offsets from (r, c). No bounds check."""
    return [(r + dr, c + dc) for dr, dc in _KNIGHT_OFFSETS]


def find_components(state, side: str) -> List[frozenset]:
    """All same-color bridge-connected components for `side` on the current state.

    Uses state._get_connected_component which respects enemy-blocking of bridges
    on real states. The _StubState fixture in tests provides a simpler
    knight-neighbor walk; that's sufficient for unit testing the classifier
    logic without a full Twixt board.
    """
    pegs_of = [p for p, color in state.pegs.items() if color == side]
    seen: set = set()
    components: List[frozenset] = []
    for peg in pegs_of:
        if peg in seen:
            continue
        comp = frozenset(state._get_connected_component(peg, side))
        if not comp:
            comp = frozenset({peg})
        seen.update(comp)
        components.append(comp)
    return components


def is_local_to_existing(state, side: str, move: Tuple[int, int]) -> bool:
    """True iff `move` is at TwixT knight distance of at least one same-color peg.

    Bridge-formability is NOT required; the flag is about proximity to
    bridge-able structure, per spec §3.1.
    """
    r, c = move
    for (nr, nc) in knight_neighbors(r, c):
        if state.pegs.get((nr, nc)) == side:
            return True
    return False


def selected_component_after(state_after, side: str, move: Tuple[int, int]) -> frozenset:
    """The component containing `move` in the POST-MOVE state.

    Caller is responsible for constructing `state_after` (via state.apply_move
    or equivalent). This helper performs no state mutation — making it safe
    to call inside the per-ply hook without copying or restoring board state.
    """
    comp = frozenset(state_after._get_connected_component(move, side))
    if not comp:
        comp = frozenset({move})
    return comp


# ---------------------------------------------------------------------------
# Trigger evaluation
# ---------------------------------------------------------------------------

def evaluate_trigger(
    *,
    current_search_score: Optional[float],
    root_top1_share: Optional[float],
    previous_own_search_score: Optional[float],
    config: RecoveryRetargetingConfig,
) -> dict:
    """Per-ply trigger decision. Pure function. Spec §2.

    Returns:
        {
          "triggered": bool,
          "trigger_reason": None | "delta_precursor" | "steady_state" | "both",
          "is_severe_collapse": bool,
          "is_very_diffuse": bool,
          "missing_search_score": bool,
          "missing_root_top1_share": bool,
          "search_score_delta": Optional[float],
        }
    """
    missing_search_score = current_search_score is None
    missing_root_top1_share = root_top1_share is None
    if missing_search_score or missing_root_top1_share:
        return {
            "triggered": False,
            "trigger_reason": None,
            "is_severe_collapse": False,
            "is_very_diffuse": False,
            "missing_search_score": missing_search_score,
            "missing_root_top1_share": missing_root_top1_share,
            "search_score_delta": None,
        }

    diffuse_root = root_top1_share <= config.diffuse_root_top1_threshold

    delta_value = (
        previous_own_search_score - current_search_score
        if previous_own_search_score is not None else None
    )
    delta_precursor = (
        delta_value is not None
        and delta_value >= config.delta_threshold
        and current_search_score <= config.delta_max_current_score
        and diffuse_root
    )

    steady_state = (
        current_search_score <= config.collapse_value_threshold
        and diffuse_root
    )

    if delta_precursor and steady_state:
        trigger_reason = "both"
    elif delta_precursor:
        trigger_reason = "delta_precursor"
    elif steady_state:
        trigger_reason = "steady_state"
    else:
        trigger_reason = None

    return {
        "triggered": trigger_reason is not None,
        "trigger_reason": trigger_reason,
        "is_severe_collapse": current_search_score <= config.severe_collapse_value_threshold,
        "is_very_diffuse": root_top1_share <= config.very_diffuse_root_top1_threshold,
        "missing_search_score": False,
        "missing_root_top1_share": False,
        "search_score_delta": delta_value,
    }


# ---------------------------------------------------------------------------
# Primary-class classifier
# ---------------------------------------------------------------------------

PRIMARY_CLASSES = (
    "blocks_opponent_closeout",
    "reduces_own_goal_distance",
    "starts_or_extends_alternate_component",
    "connects_to_existing_component",
    "improves_own_largest_component",
    "redundant_local_reinforcement",
    "off_plan_or_unclear",
)


def _dominant_component(components: List[frozenset]) -> Optional[frozenset]:
    """Largest component by size; tie-break by lexicographically-smallest peg."""
    if not components:
        return None
    # find_components guarantees non-empty components, so no min(c) guard needed.
    return max(components, key=lambda c: (len(c), -min(c)[0], -min(c)[1]))


def classify_move(
    *,
    state_before,
    state_after,
    side: str,
    move: Tuple[int, int],
    own_total_goal_distance_before: Optional[int],
    own_total_goal_distance_after: Optional[int],
    opponent_total_goal_distance_before: Optional[int],
    opponent_total_goal_distance_after: Optional[int],
    classify_defense: bool,
    alternate_component_min_size: int,
) -> dict:
    """Classify a single move into one of PRIMARY_CLASSES. Spec §3.

    Both `state_before` (pre-move) and `state_after` (post-move) are passed
    in; the classifier never mutates either. Caller computes state_after
    once via state.apply_move(move) and reuses it.

    Returns:
        {
          "primary_class": str,
          "flags": {
            "opens_new_component": bool,
            "merges_components": bool,
            "merges_dominant_with_alternate": bool,
            "extends_dominant_component": bool,
            "local_to_existing": bool,
            "blocked_opponent_closeout": bool,
          },
          "own_largest_component_size_before": int,
          "own_largest_component_size_after": int,
        }
    """
    own_components_before = find_components(state_before, side)
    dominant_before = _dominant_component(own_components_before)
    selected_after = selected_component_after(state_after, side, move)
    local_to_existing = is_local_to_existing(state_before, side, move)

    prior_components_extended = [c for c in own_components_before if c <= selected_after]
    opens_new = len(prior_components_extended) == 0
    merges = len(prior_components_extended) >= 2
    extends_dominant = dominant_before is not None and (dominant_before <= selected_after)
    merges_dom_alt = extends_dominant and merges
    extends_only_non_dominant = (
        len(prior_components_extended) == 1
        and not extends_dominant
    )

    largest_before = max((len(c) for c in own_components_before), default=0)
    own_components_after = find_components(state_after, side)
    largest_after = max((len(c) for c in own_components_after), default=0)

    # Defense check (priority 1, only when classify_defense=True).
    blocked_opp = False
    if classify_defense and opponent_total_goal_distance_before is not None and opponent_total_goal_distance_before <= 2:
        if (opponent_total_goal_distance_after is None
                or opponent_total_goal_distance_after > opponent_total_goal_distance_before):
            blocked_opp = True

    flags = {
        "opens_new_component":            opens_new,
        "merges_components":              merges,
        "merges_dominant_with_alternate": merges_dom_alt,
        "extends_dominant_component":     extends_dominant,
        "local_to_existing":              local_to_existing,
        "blocked_opponent_closeout":      blocked_opp,
    }

    # Priority-ordered classification.
    if blocked_opp:
        primary = "blocks_opponent_closeout"
    elif (own_total_goal_distance_before is not None
          and own_total_goal_distance_after is not None
          and own_total_goal_distance_after < own_total_goal_distance_before):
        primary = "reduces_own_goal_distance"
    elif (
        not extends_dominant
        and (opens_new or extends_only_non_dominant or merges)
        and len(selected_after) >= alternate_component_min_size
    ):
        primary = "starts_or_extends_alternate_component"
    elif len(prior_components_extended) >= 1:
        primary = "connects_to_existing_component"
    elif largest_after > largest_before:
        primary = "improves_own_largest_component"
    elif (local_to_existing
          and (own_total_goal_distance_before is None
               or own_total_goal_distance_after is None
               or own_total_goal_distance_after >= own_total_goal_distance_before)):
        primary = "redundant_local_reinforcement"
    else:
        primary = "off_plan_or_unclear"

    return {
        "primary_class": primary,
        "flags": flags,
        "own_largest_component_size_before": largest_before,
        "own_largest_component_size_after": largest_after,
    }


# ---------------------------------------------------------------------------
# Rollup mapping (single source of truth for spec §3.5 / §6.1 partitions)
# ---------------------------------------------------------------------------

def _bucket_rollup(counts: Dict[str, int], *, denom: int) -> dict:
    """Map per-class counts into the four spec §3.5 rollups + their rates.

    counts: a {class -> int} dict over PRIMARY_CLASSES.
    denom:  classified_in_window_moves (use 1 when zero so rates are 0.0).

    Used by both RecoveryRetargetingTracker.finalize_game (per-game record)
    and aggregate_recovery_retargeting_records (per-iter sidecar) so the
    bucket→class mapping lives in one place.
    """
    constructive = counts["reduces_own_goal_distance"] + counts["starts_or_extends_alternate_component"]
    defensive    = counts["blocks_opponent_closeout"]
    structural   = counts["connects_to_existing_component"] + counts["improves_own_largest_component"]
    local_drift  = counts["redundant_local_reinforcement"] + counts["off_plan_or_unclear"]
    return {
        "constructive_recovery_moves":  constructive,
        "defensive_moves":              defensive,
        "structural_connection_moves":  structural,
        "local_drift_moves":            local_drift,
        "constructive_recovery_rate":   round(constructive / denom, 3),
        "defensive_rate":               round(defensive / denom, 3),
        "structural_connection_rate":   round(structural / denom, 3),
        "local_drift_rate":             round(local_drift / denom, 3),
    }


# ---------------------------------------------------------------------------
# Per-game tracker
# ---------------------------------------------------------------------------

@dataclass
class _SideAccumulator:
    triggered: bool = False
    first_trigger_ply: Optional[int] = None
    first_trigger_reason: Optional[str] = None
    previous_own_search_score: Optional[float] = None

    in_window_own_moves: int = 0
    triggered_own_moves: int = 0
    non_triggered_in_window_moves: int = 0
    missing_signal_moves: int = 0
    missing_search_score_moves: int = 0
    missing_root_top1_share_moves: int = 0

    trigger_reason_counts: Dict[str, int] = field(
        default_factory=lambda: {"delta_precursor": 0, "steady_state": 0, "both": 0}
    )
    severe_collapse_moves: int = 0
    very_diffuse_moves: int = 0

    triggered_scores: List[float] = field(default_factory=list)
    triggered_top1_shares: List[float] = field(default_factory=list)

    selected_class_counts: Dict[str, int] = field(
        default_factory=lambda: {c: 0 for c in PRIMARY_CLASSES}
    )

    sampled_moves: List[dict] = field(default_factory=list)
    sampled_moves_dropped: int = 0
    classifier_error_count: int = 0


class RecoveryRetargetingTracker:
    """Per-game tracker. One instance per game; lifecycle matches play_game.

    gc_state_provider: callable(state, side, enumerate_moves=False) -> dict|None
        Matches connectivity_diagnostics.compute_goal_completion_state.
    """

    def __init__(self, config: RecoveryRetargetingConfig, gc_state_provider):
        self.config = config
        self._gc_state_provider = gc_state_provider
        self._sides: Dict[str, _SideAccumulator] = {
            "red":   _SideAccumulator(),
            "black": _SideAccumulator(),
        }
        self._warned_classifier_error = False

    def observe_move(
        self,
        *,
        state_before,
        selected_move: Tuple[int, int],
        ply: int,
        side_to_move: str,
        search_score: Optional[float],
        root_top1_share: Optional[float],
    ) -> None:
        # Defensive: enabled=False trackers no-op even if invoked.
        if not self.config.enabled:
            return

        side_acc = self._sides[side_to_move]
        opponent = "black" if side_to_move == "red" else "red"

        # Capture previous score BEFORE updating it.
        prev_score = side_acc.previous_own_search_score

        trig = evaluate_trigger(
            current_search_score=search_score,
            root_top1_share=root_top1_share,
            previous_own_search_score=prev_score,
            config=self.config,
        )

        missing = trig["missing_search_score"] or trig["missing_root_top1_share"]

        # Side not in-window and didn't trigger: just update prev_score and return.
        if not side_acc.triggered and not trig["triggered"]:
            if not trig["missing_search_score"]:
                side_acc.previous_own_search_score = search_score
            return

        # First-time trigger: open the window.
        if not side_acc.triggered and trig["triggered"]:
            side_acc.triggered = True
            side_acc.first_trigger_ply = ply
            side_acc.first_trigger_reason = trig["trigger_reason"]

        # in_window_own_moves counts every own-move once window opens.
        side_acc.in_window_own_moves += 1

        if missing:
            side_acc.missing_signal_moves += 1
            if trig["missing_search_score"]:
                side_acc.missing_search_score_moves += 1
            if trig["missing_root_top1_share"]:
                side_acc.missing_root_top1_share_moves += 1
            # Per spec §2.3: update from a valid current_search_score even if
            # root_top1_share is missing. Skip update only when search_score is None.
            if not trig["missing_search_score"]:
                side_acc.previous_own_search_score = search_score
            return

        # Valid signal: classify and update bookkeeping.
        if trig["triggered"]:
            side_acc.triggered_own_moves += 1
            side_acc.trigger_reason_counts[trig["trigger_reason"]] += 1
            side_acc.triggered_scores.append(search_score)
            side_acc.triggered_top1_shares.append(root_top1_share)
            if trig["is_severe_collapse"]:
                side_acc.severe_collapse_moves += 1
            if trig["is_very_diffuse"]:
                side_acc.very_diffuse_moves += 1
        else:
            side_acc.non_triggered_in_window_moves += 1

        # Compute state_after ONCE via state.apply_move. No mutation.
        try:
            state_after = state_before.apply_move(selected_move)
            own_gc_before = self._gc_state_provider(state_before, side_to_move, enumerate_moves=False)
            own_gc_after = self._gc_state_provider(state_after, side_to_move, enumerate_moves=False)
            opp_gc_before = None
            opp_gc_after = None
            if self.config.classify_defense:
                opp_gc_before = self._gc_state_provider(state_before, opponent, enumerate_moves=False)
                opp_gc_after = self._gc_state_provider(state_after, opponent, enumerate_moves=False)

            own_td_before = (own_gc_before or {}).get("total_goal_distance")
            own_td_after = (own_gc_after or {}).get("total_goal_distance")
            opp_td_before = (opp_gc_before or {}).get("total_goal_distance")
            opp_td_after = (opp_gc_after or {}).get("total_goal_distance")

            cls = classify_move(
                state_before=state_before,
                state_after=state_after,
                side=side_to_move,
                move=selected_move,
                own_total_goal_distance_before=own_td_before,
                own_total_goal_distance_after=own_td_after,
                opponent_total_goal_distance_before=opp_td_before,
                opponent_total_goal_distance_after=opp_td_after,
                classify_defense=self.config.classify_defense,
                alternate_component_min_size=self.config.alternate_component_min_size,
            )
            side_acc.selected_class_counts[cls["primary_class"]] += 1
            primary_class = cls["primary_class"]
            flags = cls["flags"]
            own_lcs_before = cls["own_largest_component_size_before"]
            own_lcs_after = cls["own_largest_component_size_after"]
        except Exception:
            side_acc.classifier_error_count += 1
            if not self._warned_classifier_error:
                import logging
                logging.getLogger(__name__).warning(
                    "recovery_retargeting classifier raised; recording as off_plan_or_unclear"
                )
                self._warned_classifier_error = True
            side_acc.selected_class_counts["off_plan_or_unclear"] += 1
            primary_class = "off_plan_or_unclear"
            flags = {
                "opens_new_component": False, "merges_components": False,
                "merges_dominant_with_alternate": False, "extends_dominant_component": False,
                "local_to_existing": False, "blocked_opponent_closeout": False,
            }
            own_lcs_before = 0
            own_lcs_after = 0
            own_td_before = None
            own_td_after = None
            opp_td_before = None
            opp_td_after = None

        # Sampled-moves recording.
        own_move_ordinal = side_acc.in_window_own_moves  # 1-based.
        entry = {
            "ply": ply,
            "in_window_own_move_index": own_move_ordinal,
            "triggered_this_ply": trig["triggered"],
            "trigger_reason": trig["trigger_reason"],
            "current_search_score": search_score,
            "previous_own_search_score": prev_score,
            "search_score_delta": trig["search_score_delta"],
            "root_top1_share": root_top1_share,
            "is_severe_collapse": trig["is_severe_collapse"],
            "is_very_diffuse": trig["is_very_diffuse"],
            "primary_class": primary_class,
            "selected_move": list(selected_move),
            "flags": flags,
            "own_total_goal_distance_before": own_td_before,
            "own_total_goal_distance_after": own_td_after,
            "own_largest_component_size_before": own_lcs_before,
            "own_largest_component_size_after": own_lcs_after,
            "opponent_total_goal_distance_before": opp_td_before,
            "opponent_total_goal_distance_after": opp_td_after,
        }
        self._maybe_record_sample(side_acc, entry)

        # Update previous_own_search_score AFTER entry is built.
        side_acc.previous_own_search_score = search_score

    def _maybe_record_sample(self, side_acc: _SideAccumulator, entry: dict) -> None:
        if self.config.sample_all_moves:
            side_acc.sampled_moves.append(entry)
            return
        cap = self.config.max_sampled_moves_per_side
        if cap <= 0:
            side_acc.sampled_moves_dropped += 1
            return
        # Priority 1 (highest): first 4 own-moves in window.
        # Priority 2: severe-collapse plies.
        # Priority 3 (lowest): everything else, in window order.
        side_acc.sampled_moves.append(entry)
        if len(side_acc.sampled_moves) > cap:
            def _priority(e):
                if e.get("in_window_own_move_index", 10**9) <= 4:
                    return 0
                if e["is_severe_collapse"]:
                    return 1
                return 2
            worst_idx = max(
                range(len(side_acc.sampled_moves)),
                key=lambda i: (_priority(side_acc.sampled_moves[i]), side_acc.sampled_moves[i]["ply"]),
            )
            side_acc.sampled_moves.pop(worst_idx)
            side_acc.sampled_moves_dropped += 1

    def side_snapshot(self, side: str) -> dict:
        """Test helper: snapshot of per-side accumulator state."""
        a = self._sides[side]
        return {
            "triggered": a.triggered,
            "first_trigger_ply": a.first_trigger_ply,
            "first_trigger_reason": a.first_trigger_reason,
            "in_window_own_moves": a.in_window_own_moves,
            "triggered_own_moves": a.triggered_own_moves,
            "non_triggered_in_window_moves": a.non_triggered_in_window_moves,
            "missing_signal_moves": a.missing_signal_moves,
            "missing_search_score_moves": a.missing_search_score_moves,
            "missing_root_top1_share_moves": a.missing_root_top1_share_moves,
            "selected_class_counts": dict(a.selected_class_counts),
            "classifier_error_count": a.classifier_error_count,
        }

    def _build_side_record(self, a: _SideAccumulator) -> dict:
        """Build the per-side dict for a triggered side. Spec §4 schema."""
        classified = sum(a.selected_class_counts.values())
        denom = classified if classified > 0 else 1
        scores = a.triggered_scores
        shares = a.triggered_top1_shares
        return {
            "triggered":              True,
            "first_trigger_ply":      a.first_trigger_ply,
            "first_trigger_reason":   a.first_trigger_reason,
            "classifier_error_count": a.classifier_error_count,

            "in_window_own_moves":             a.in_window_own_moves,
            "triggered_own_moves":             a.triggered_own_moves,
            "non_triggered_in_window_moves":   a.non_triggered_in_window_moves,
            "missing_signal_moves":            a.missing_signal_moves,
            "missing_search_score_moves":      a.missing_search_score_moves,
            "missing_root_top1_share_moves":   a.missing_root_top1_share_moves,

            "trigger_reason_counts":  dict(a.trigger_reason_counts),
            "severe_collapse_moves":  a.severe_collapse_moves,
            "very_diffuse_moves":     a.very_diffuse_moves,

            "mean_search_score_triggered_plies":    round(sum(scores) / len(scores), 3) if scores else None,
            "min_search_score_triggered_plies":     round(min(scores), 3) if scores else None,
            "max_search_score_triggered_plies":     round(max(scores), 3) if scores else None,
            "mean_root_top1_share_triggered_plies": round(sum(shares) / len(shares), 3) if shares else None,

            "classified_in_window_moves": classified,
            "selected_class_counts":      dict(a.selected_class_counts),

            **_bucket_rollup(a.selected_class_counts, denom=denom),

            "sampled_moves_count":   len(a.sampled_moves),
            "sampled_moves_cap":     self.config.max_sampled_moves_per_side,
            "sampled_moves_dropped": a.sampled_moves_dropped,
            "sample_all_moves":      self.config.sample_all_moves,
            "sampled_moves":         list(a.sampled_moves),
        }

    def finalize_game(
        self,
        *,
        iteration: int,
        game_idx: int,
        game_id: str,
        winner: Optional[str],
        starting_player: str,
        n_moves: int,
        reason: str,
    ) -> Optional[dict]:
        """Emit per-game record per Spec §4 if any side opened a window. Else None."""
        triggered_sides = [s for s, a in self._sides.items() if a.triggered]
        if not triggered_sides:
            return None

        loser = None
        if winner == "red":
            loser = "black"
        elif winner == "black":
            loser = "red"

        first_acc = min(
            (a for a in self._sides.values() if a.triggered),
            key=lambda a: a.first_trigger_ply if a.first_trigger_ply is not None else 10**9,
        )
        first_trigger_ply = first_acc.first_trigger_ply
        first_trigger_side = next(s for s, a in self._sides.items() if a is first_acc)
        first_trigger_reason = first_acc.first_trigger_reason

        side_records: Dict[str, dict] = {}
        total_classifier_errors = 0
        for side in ("red", "black"):
            a = self._sides[side]
            total_classifier_errors += a.classifier_error_count
            if not a.triggered:
                side_records[side] = {"triggered": False, "classifier_error_count": a.classifier_error_count}
                continue
            side_records[side] = self._build_side_record(a)

        return {
            "version": 1,
            "iteration": iteration,
            "game_idx": game_idx,
            "game_id": game_id,
            "winner": winner,
            "loser": loser,
            "starting_player": starting_player,
            "n_moves": n_moves,
            "reason": reason,
            "classifier_error_count": total_classifier_errors,
            "config": {
                "collapse_value_threshold":          self.config.collapse_value_threshold,
                "severe_collapse_value_threshold":   self.config.severe_collapse_value_threshold,
                "diffuse_root_top1_threshold":       self.config.diffuse_root_top1_threshold,
                "very_diffuse_root_top1_threshold":  self.config.very_diffuse_root_top1_threshold,
                "delta_threshold":                   self.config.delta_threshold,
                "delta_max_current_score":           self.config.delta_max_current_score,
                "alternate_component_min_size":      self.config.alternate_component_min_size,
                "classify_defense":                  self.config.classify_defense,
            },
            "triggered_sides":      triggered_sides,
            "first_trigger_ply":    first_trigger_ply,
            "first_trigger_side":   first_trigger_side,
            "first_trigger_reason": first_trigger_reason,
            "side_records":         side_records,
        }


# ---------------------------------------------------------------------------
# Per-iteration aggregator
# ---------------------------------------------------------------------------

_AGG_COUNT_KEYS = (
    "in_window_own_moves",
    "triggered_own_moves",
    "non_triggered_in_window_moves",
    "missing_signal_moves",
    "severe_collapse_moves",
    "very_diffuse_moves",
)


def aggregate_recovery_retargeting_records(
    records: List[dict],
    *,
    games_total: int,
    config: Optional[dict] = None,
) -> dict:
    """Aggregate per-game records into a per-iteration sidecar summary. Spec §6.

    `games_total` is the iteration's full game count; records exist only when
    at least one side triggered.

    Per-iteration semantics: all records must share the same config block.
    Cross-iteration use is handled by the analyzer's analyze() loop, not here.
    """
    skipped_unknown_version = 0
    skipped_config_mismatch = 0
    accepted: List[dict] = []
    canonical_config = config

    for rec in records:
        if rec is None:
            continue
        if rec.get("version") != 1:
            skipped_unknown_version += 1
            continue
        cfg = rec.get("config") or {}
        if canonical_config is None:
            canonical_config = cfg
        elif cfg != canonical_config:
            skipped_config_mismatch += 1
            continue
        accepted.append(rec)

    games_triggered = len(accepted)
    triggered_loser_side = 0
    triggered_winner_side = 0
    sums_total = {k + "_total": 0 for k in _AGG_COUNT_KEYS}
    selected_class_totals = {c: 0 for c in PRIMARY_CLASSES}
    trigger_reason_totals = {"delta_precursor": 0, "steady_state": 0, "both": 0}
    classifier_error_total = 0

    for rec in accepted:
        classifier_error_total += int(rec.get("classifier_error_count", 0))
        winner = rec.get("winner")
        loser = rec.get("loser")
        for side, sr in (rec.get("side_records") or {}).items():
            if not sr or not sr.get("triggered"):
                continue
            if side == loser:
                triggered_loser_side += 1
            elif side == winner:
                triggered_winner_side += 1
            for k in _AGG_COUNT_KEYS:
                sums_total[k + "_total"] += int(sr.get(k, 0) or 0)
            for cls, count in (sr.get("selected_class_counts") or {}).items():
                if cls in selected_class_totals:
                    selected_class_totals[cls] += int(count or 0)
            for reason, count in (sr.get("trigger_reason_counts") or {}).items():
                if reason in trigger_reason_totals:
                    trigger_reason_totals[reason] += int(count or 0)

    classified_total = sum(selected_class_totals.values())
    denom = classified_total if classified_total > 0 else 1
    selected_class_rates = {
        cls: round(count / denom, 3) for cls, count in selected_class_totals.items()
    }
    rollup = _bucket_rollup(selected_class_totals, denom=denom)

    return {
        "version": 1,
        "enabled": True,
        "config": canonical_config or {},
        "games_total": games_total,
        "games_triggered": games_triggered,
        "trigger_rate": round(games_triggered / games_total, 3) if games_total > 0 else 0.0,
        "triggered_loser_side": triggered_loser_side,
        "triggered_winner_side": triggered_winner_side,
        "triggered_loser_side_per_triggered_game": round(triggered_loser_side / games_triggered, 3) if games_triggered > 0 else 0.0,
        "triggered_winner_side_per_triggered_game": round(triggered_winner_side / games_triggered, 3) if games_triggered > 0 else 0.0,
        **sums_total,
        "trigger_reason_counts_total": trigger_reason_totals,
        "classified_in_window_moves_total": classified_total,
        "selected_class_counts_total": selected_class_totals,
        "selected_class_rates_total": selected_class_rates,
        # Per-iter sidecar emits only the four rollup *rates* (not the *_moves
        # totals — those are at game scope, not iter scope).
        "constructive_recovery_rate":  rollup["constructive_recovery_rate"],
        "defensive_rate":              rollup["defensive_rate"],
        "structural_connection_rate":  rollup["structural_connection_rate"],
        "local_drift_rate":            rollup["local_drift_rate"],
        "schema_integrity": {
            "skipped_unknown_version_count": skipped_unknown_version,
            "skipped_config_mismatch_count": skipped_config_mismatch,
            "classifier_error_count_total": classifier_error_total,
        },
    }


# ---------------------------------------------------------------------------
# Side-split aggregation (Spec 2026-05-13 filtered side-split view)
# ---------------------------------------------------------------------------


def _side_bucket_for_record(record: dict, side: str) -> str:
    """Map a triggered side to its eventual-game-outcome bucket.

    Returns one of: 'eventual_loser', 'eventual_winner', 'state_cap_or_draw'.
    Buckets are by *eventual game outcome*, not by side at the trigger ply.
    Draws and state-caps land in 'state_cap_or_draw' for both sides.
    """
    winner = record.get("winner")
    loser = record.get("loser")
    if winner is None:
        return "state_cap_or_draw"
    if side == loser:
        return "eventual_loser"
    if side == winner:
        return "eventual_winner"
    return "state_cap_or_draw"
