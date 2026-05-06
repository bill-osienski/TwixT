"""Legacy replay walker for goal-completion (spec §11.5).

Runs only behind --goal-completion-recompute. Adopts pre-move detection
semantics so its outputs are directly comparable with inline records.
"""
from __future__ import annotations

from typing import List, Optional

from .connectivity_diagnostics import (
    compute_goal_completion_state,
    classify_selected_conversion_move,
)
from .game.twixt_state import TwixtState
from .goal_completion_aggregator import (
    aggregate_goal_completion_records,
)


def recompute_goal_completion_records_from_replays(
    replays: list, config: dict,
) -> List[Optional[dict]]:
    """Walk each replay's move history and produce a goal_completion_record.

    Pre-move detection semantics: a side is detected on the first ply
    where it is to move and its pre-move state already has
    total_goal_distance <= detection_threshold. The selected move on
    that ply IS classified.
    """
    detection_threshold = int(config.get("detection_threshold", 2))
    max_depth = int(config.get("max_depth", 3))
    min_component_size = int(config.get("min_component_size", 8))
    high_value_threshold = float(config.get("high_value_threshold", 0.9))
    high_value_delay_plies = int(config.get("high_value_delay_threshold_plies", 6))

    out: List[Optional[dict]] = []
    for replay in replays:
        try:
            rec = _walk_replay(
                replay,
                detection_threshold=detection_threshold,
                max_depth=max_depth,
                min_component_size=min_component_size,
                high_value_threshold=high_value_threshold,
                high_value_delay_plies=high_value_delay_plies,
            )
            out.append(rec)
        except Exception as e:
            import sys
            sys.stderr.write(
                f"[recompute] iter_{replay.get('iteration')}_game_"
                f"{replay.get('game_idx')}: {e!r}\n"
            )
            out.append(None)
    return out


def _walk_replay(
    replay: dict,
    *,
    detection_threshold: int,
    max_depth: int,
    min_component_size: int,
    high_value_threshold: float,
    high_value_delay_plies: int,
) -> Optional[dict]:
    """Re-derive a goal_completion_record from raw move history.

    Mirrors GoalCompletionGameTracker but operates on a stored replay
    (no live MCTS state). Pre-move detection semantics: the dominant
    component is checked on the state BEFORE each move is applied.
    """
    from .goal_completion_tracker import GoalCompletionGameTracker

    moves = replay.get("moves") or []
    starting_player = replay.get("starting_player", "red")
    active = (replay.get("meta") or {}).get("board_size", 24)
    winner = replay.get("winner")
    if winner not in ("red", "black"):
        winner = None
    reason = (replay.get("meta") or {}).get("reason", "win" if winner else "unknown")

    tracker = GoalCompletionGameTracker(
        enabled=True,
        detection_threshold=detection_threshold,
        high_value_threshold=high_value_threshold,
        high_value_delay_threshold_plies=high_value_delay_plies,
        max_depth=max_depth,
        min_component_size=min_component_size,
    )

    state = TwixtState(active_size=active, to_move=starting_player)
    for i, m in enumerate(moves):
        side = m.get("player") or state.to_move
        sel = (int(m["row"]), int(m["col"]))
        # Pre-move state: compute goal-completion state for side_to_move
        # BEFORE applying selected move.
        try:
            gc_cheap = compute_goal_completion_state(
                state, side,
                max_depth=max_depth,
                min_component_size=min_component_size,
                enumerate_moves=False,
            )
        except Exception:
            gc_cheap = None

        gc_full = None
        if gc_cheap is not None:
            total = gc_cheap.get("total_goal_distance")
            if total is not None and (
                tracker.is_detected(side) or total <= detection_threshold
            ):
                try:
                    gc_full = compute_goal_completion_state(
                        state, side,
                        max_depth=max_depth,
                        min_component_size=min_component_size,
                        enumerate_moves=True,
                    )
                except Exception:
                    gc_full = None

        ss = m.get("search_score")
        tracker.observe_pre_move(
            state=state, ply=i + 1, side_to_move=side,
            selected_move=sel,
            search_score=float(ss) if ss is not None else None,
            gc_state_cheap=gc_cheap, gc_state_full=gc_full,
        )

        try:
            state = state.apply_move(sel)
        except Exception:
            return None  # Corrupt replay -> bubble out

    # Map replay reason to tracker outcome reasons.
    return tracker.finalize_game(
        winner=winner,
        reason=reason,
        n_moves=len(moves),
        starting_player=starting_player,
        iteration=int(replay.get("iteration", 0)),
        game_idx=int(replay.get("game_idx", 0)),
        game_id=(replay.get("goal_completion_record") or {}).get("game_id")
                or f"iter_{int(replay.get('iteration', 0)):04d}_game_{int(replay.get('game_idx', 0)):03d}",
    )


_KEY_FIELDS = (
    "outcome_class",
    "detected",
    "detected_player",
    "first_dominant_unclosed_ply",
    "first_total_goal_distance",
    "first_category",
    "conversion_delay_plies",
    "conversion_delay_winner_moves",
    "cap_delay_proxy_plies",
    "primary_class_counts",
    "root_value_high_but_delayed",
)
_FLOAT_FIELDS = (
    "max_search_score_after_detection",
    "mean_search_score_after_detection",
)
_FLOAT_TOLERANCE = 1e-6


def compare_records_for_validation(
    inline: Optional[dict], recomputed: Optional[dict],
) -> dict:
    """Per-field divergence report (spec §11.6)."""
    if inline is None and recomputed is None:
        return {}
    if inline is None or recomputed is None:
        return {"presence": (inline is not None, recomputed is not None)}
    div: dict = {}
    for k in _KEY_FIELDS:
        a, b = inline.get(k), recomputed.get(k)
        if a != b:
            div[k] = (a, b)
    for k in _FLOAT_FIELDS:
        a, b = inline.get(k), recomputed.get(k)
        if a is None and b is None:
            continue
        if a is None or b is None or abs(float(a) - float(b)) > _FLOAT_TOLERANCE:
            div[k] = (a, b)
    return div
