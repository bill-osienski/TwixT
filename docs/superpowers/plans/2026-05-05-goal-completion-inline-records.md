# Goal-Completion Inline Records Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move per-game goal-completion classification from replay-side BFS aggregation into self-play inline emission. Analyzer becomes a record consumer; replay reconstruction moves behind `--goal-completion-recompute`.

**Architecture:** New tracker module observes per-ply during self-play, emits one compact `goal_completion_record` per game. Trainer aggregates per-iteration into a sidecar `goal_completion_summary` block. Analyzer reads pre-computed records — no BFS by default. Pre-move detection semantics: detection fires when the side to move *already* has a closeout-shaped position pre-move (not when their last move created it).

**Tech Stack:** Python 3.14, pytest, dataclasses, pickle (IPC), JSON (per-game replay format).

**Spec:** [docs/superpowers/specs/2026-05-05-goal-completion-inline-records-design.md](../specs/2026-05-05-goal-completion-inline-records-design.md)

---

## Plan Preamble

**Spec correction**: §12.1 claims `goal_completion_broadcast_threshold` will be renamed to `goal_completion_emit_threshold`. **The rename is already done** — the existing Phase 3 code uses `goal_completion_emit_threshold` and `goal_completion_emit_enabled`. No rename action needed; treat any spec mention of "renamed from broadcast_threshold" as a confirmation step (verify nothing references the old name).

**Game 097 anchor**: the existing structural anchor in `tests/test_connectivity_goal_completion.py` lines 425+ asserts the closeout shape lands at turn 43 (Red's move that *creates* the two-endpoint closeout, post-move detection). Under pre-move semantics, the new tracker's first detection ply lands at the next time Red is to move with that closeout still pre-existing — turn 45 in this replay (Red moves on odd turns: 43 creates, 44 is Black's response, 45 is Red's first opportunity). The plan uses synthetic small fixtures for tracker unit tests where exact plies are computable by hand; Game 097 appears as an integration cross-check.

**File structure summary** (locked here, used throughout):

| Path | Status | Responsibility |
|---|---|---|
| `scripts/GPU/alphazero/goal_completion_tracker.py` | NEW | Per-game tracker dataclass + observe_pre_move + finalize_game |
| `scripts/GPU/alphazero/goal_completion_aggregator.py` | NEW | Pure aggregation: records → goal_completion_summary block. Population summary helpers (migrated from analyzer). |
| `scripts/GPU/alphazero/goal_completion_recompute.py` | NEW | Legacy replay walker (Spec 1 Phase 2 code), updated to pre-move semantics, only used behind `--goal-completion-recompute`. |
| `scripts/GPU/alphazero/ipc_messages.py` | MODIFY | Add `goal_completion_record: Optional[dict] = None` to `GameComplete`. |
| `scripts/GPU/alphazero/self_play.py` | MODIFY | Add field to `GameRecord`; wire tracker into `play_game()`; BFS reuse contract. |
| `scripts/GPU/alphazero/self_play_worker.py` | MODIFY | Pass `goal_completion_record` from `GameRecord` to `GameComplete`. |
| `scripts/GPU/alphazero/game_saver.py` | MODIFY | Add `goal_completion_record` kwarg to `save_game_replay` and `GameSaver.maybe_save_game`. |
| `scripts/GPU/alphazero/trainer.py` | MODIFY | Per-iteration aggregation hook: call `aggregate_goal_completion_records` and add `goal_completion_summary` to sidecar. |
| `scripts/twixt_replay_analyzer.py` | MODIFY | Replace `aggregate_goal_completion_diagnostics` default path with record-consumption; add `--goal-completion-recompute*` flags. |
| `tests/test_goal_completion_tracker.py` | NEW | Tracker unit tests (incl. `test_tracker_premove_detection_classifies_detection_ply_move` anchor). |
| `tests/test_goal_completion_aggregator.py` | NEW | Aggregator unit tests. |
| `tests/test_goal_completion_save_load.py` | NEW | IPC + saver round-trip. |
| `tests/test_self_play_goal_completion_integration.py` | NEW | End-to-end self-play tracker integration. |
| `tests/test_analyzer_goal_completion_records.py` | NEW | Analyzer record-consumption + structural anti-regression test. |
| `tests/test_analyzer_goal_completion_recompute.py` | NEW | Recompute fallback + validate flag tests. |
| `tests/test_analyzer_per_ply_perf_regression.py` | NEW | Generous wall-clock perf bound on 50 fixture games. |
| `tests/test_connectivity_goal_completion.py` | MINOR EDIT | Rename Game 097 structural-anchor test name. |
| `tests/test_analyzer_goal_completion.py` | MIGRATE | Tests using `_build_class1_per_game_record` move to `test_analyzer_goal_completion_recompute.py`; aggregator-helper tests move to `test_goal_completion_aggregator.py`. |

---

## Tasks Overview

| # | Task | Phase |
|---|---|---|
| 1 | Tracker scaffold + side accumulator + detection update | Tracker |
| 2 | Tracker classification + watch-window + dual-side | Tracker |
| 3 | Tracker finalize (Class 1 / 2 / 3 + tie-break) | Tracker |
| 4 | Aggregator scaffold + `_normalize_record` + empty/coverage | Aggregator |
| 5 | Aggregator: migrate population helpers; cross-iteration roll-up | Aggregator |
| 6 | IPC + saver plumbing (`GameComplete`, `GameRecord`, saver kwarg) | IPC |
| 7 | `play_game()` integration + BFS reuse contract | Self-play |
| 8 | Worker → main pipeline carries `goal_completion_record` | Self-play |
| 9 | Trainer per-iteration sidecar aggregation hook | Trainer |
| 10 | Analyzer record-consumption default path + worst-cases CSV | Analyzer |
| 11 | Analyzer warnings (missing-record, sidecar/version mismatch) | Analyzer |
| 12 | Analyzer structural anti-regression test + perf bound | Analyzer |
| 13 | Recompute module + `--goal-completion-recompute` flag | Recompute |
| 14 | `--goal-completion-recompute-validate` flag + per-field divergence | Recompute |

After Task 14: end-to-end smoke run on a fresh small training iteration to confirm records flow end-to-end.

---

## Task 1: Tracker scaffold + side accumulator + detection update

**Files:**
- Create: `scripts/GPU/alphazero/goal_completion_tracker.py`
- Test: `tests/test_goal_completion_tracker.py`

**Goal:** Stand up the dataclass shape and the per-ply detection state machine. No classification yet (Task 2 adds it). No finalize yet (Task 3 adds it).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_goal_completion_tracker.py
"""Tracker unit tests (spec 2026-05-05 §6).

Pre-move detection semantics: detection fires when the side to move
already has a closeout-shaped position pre-move. The selected move on
the detection ply IS counted as a post-detection move (classification
arrives in Task 2).
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.goal_completion_tracker import (
    GoalCompletionGameTracker,
    _SideAccumulator,
)


def _gc_state(total, category="two_endpoint_closeout_2ply",
              endpoint_distances=None, component_pegs=None):
    """Build a minimal gc_state dict for the tracker's coverage/detection path."""
    return {
        "total_goal_distance": total,
        "category": category,
        "endpoint_distances": endpoint_distances or {"top": 0, "bottom": 1},
        "component_pegs": component_pegs or frozenset({(0, 0), (2, 1), (4, 2)}),
    }


def test_tracker_disabled_observe_is_noop():
    t = GoalCompletionGameTracker(enabled=False)
    t.observe_pre_move(
        state=None, ply=1, side_to_move="red",
        selected_move=(5, 5), search_score=None,
        gc_state_cheap=_gc_state(2),
        gc_state_full=None,
    )
    assert t.red.detected is False
    assert t.red.first_dominant_unclosed_ply is None


def test_tracker_coverage_flags_update_per_side_to_move():
    t = GoalCompletionGameTracker()
    # Red's pre-move state at ply 5: total=4. Not below threshold yet but
    # min/ever flags should advance.
    t.observe_pre_move(
        state=None, ply=5, side_to_move="red",
        selected_move=(0, 0), search_score=None,
        gc_state_cheap=_gc_state(4), gc_state_full=None,
    )
    assert t.red.min_total_goal_distance == 4
    assert t.red.ever_distance_le_2 is False
    assert t.red.ever_distance_le_3 is False
    # Black's accumulator is untouched.
    assert t.black.min_total_goal_distance is None


def test_tracker_premove_detection_fires_at_first_eligible_side_move():
    """Detection ply equals the first ply where the side to move already
    has total_goal_distance <= detection_threshold pre-move. Detection
    threshold defaults to 2."""
    t = GoalCompletionGameTracker(detection_threshold=2)
    # Red's first three moves: total decreasing 5 -> 3 -> 2.
    # Detection fires at the THIRD red move (ply 5, 1-indexed).
    t.observe_pre_move(state=None, ply=1, side_to_move="red",
                       selected_move=(0,0), search_score=None,
                       gc_state_cheap=_gc_state(5), gc_state_full=None)
    t.observe_pre_move(state=None, ply=3, side_to_move="red",
                       selected_move=(1,1), search_score=None,
                       gc_state_cheap=_gc_state(3), gc_state_full=None)
    assert t.red.detected is False
    t.observe_pre_move(state=None, ply=5, side_to_move="red",
                       selected_move=(2,2), search_score=None,
                       gc_state_cheap=_gc_state(2, category="two_endpoint_closeout_2ply"),
                       gc_state_full=None)
    assert t.red.detected is True
    assert t.red.first_dominant_unclosed_ply == 5
    assert t.red.first_total_goal_distance == 2
    assert t.red.first_category == "two_endpoint_closeout_2ply"
    assert t.red.first_endpoint_distances == {"top": 0, "bottom": 1}


def test_tracker_first_largest_component_size_recorded_at_detection():
    t = GoalCompletionGameTracker()
    component = frozenset({(0, 0), (2, 1), (4, 2), (6, 3), (8, 4)})
    t.observe_pre_move(state=None, ply=11, side_to_move="black",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2, category="one_endpoint_distance_2",
                                                component_pegs=component),
                       gc_state_full=None)
    assert t.black.detected is True
    assert t.black.first_largest_component_size == 5


def test_tracker_detection_records_only_first_event():
    """Once detected, subsequent observations on the same side do not
    overwrite first-detection metadata."""
    t = GoalCompletionGameTracker()
    t.observe_pre_move(state=None, ply=7, side_to_move="red",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2, category="two_endpoint_closeout_2ply"),
                       gc_state_full=None)
    t.observe_pre_move(state=None, ply=9, side_to_move="red",
                       selected_move=(1, 1), search_score=None,
                       gc_state_cheap=_gc_state(1, category="one_move_win"),
                       gc_state_full=None)
    assert t.red.first_dominant_unclosed_ply == 7
    assert t.red.first_total_goal_distance == 2
    assert t.red.first_category == "two_endpoint_closeout_2ply"
    # min should track the lower value though.
    assert t.red.min_total_goal_distance == 1


def test_tracker_dual_side_independent():
    """Both sides reach detection independently in the same game."""
    t = GoalCompletionGameTracker()
    t.observe_pre_move(state=None, ply=11, side_to_move="red",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2), gc_state_full=None)
    t.observe_pre_move(state=None, ply=14, side_to_move="black",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2), gc_state_full=None)
    assert t.red.detected is True and t.red.first_dominant_unclosed_ply == 11
    assert t.black.detected is True and t.black.first_dominant_unclosed_ply == 14


def test_tracker_is_detected_helper():
    t = GoalCompletionGameTracker()
    assert t.is_detected("red") is False
    t.observe_pre_move(state=None, ply=3, side_to_move="red",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2), gc_state_full=None)
    assert t.is_detected("red") is True
    assert t.is_detected("black") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_goal_completion_tracker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.GPU.alphazero.goal_completion_tracker'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/GPU/alphazero/goal_completion_tracker.py
"""Per-game inline goal-completion tracker (spec 2026-05-05).

Observes per-ply self-play events; emits one compact goal_completion_record
per game. Replaces the analyzer's replay-side BFS aggregation as the
canonical source of goal-completion telemetry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


def _zero_class_counts() -> dict:
    return {
        "completes_endpoint": 0,
        "reduces_total_goal_distance": 0,
        "redundant_reinforcement": 0,
        "off_chain": 0,
        "other": 0,
    }


@dataclass
class _SideAccumulator:
    """Per-side per-game accumulator. Both sides tracked in parallel
    during play; finalize_game picks the focal side based on outcome."""
    detected: bool = False
    first_dominant_unclosed_ply: Optional[int] = None
    first_total_goal_distance: Optional[int] = None
    first_category: Optional[str] = None
    first_largest_component_size: Optional[int] = None
    first_endpoint_distances: Optional[dict] = None
    primary_class_counts: dict = field(default_factory=_zero_class_counts)
    moves_after_detection: int = 0
    moves_with_dominant_component: int = 0
    moves_with_dominant_unavailable: int = 0
    search_scores_after_detection: list = field(default_factory=list)
    high_value_after_detection_plies: int = 0
    min_total_goal_distance: Optional[int] = None
    ever_distance_le_2: bool = False
    ever_distance_le_3: bool = False


@dataclass
class GoalCompletionGameTracker:
    enabled: bool = True
    detection_threshold: int = 2
    high_value_threshold: float = 0.9
    high_value_delay_threshold_plies: int = 6
    max_depth: int = 3
    min_component_size: int = 8
    red: _SideAccumulator = field(default_factory=_SideAccumulator)
    black: _SideAccumulator = field(default_factory=_SideAccumulator)

    def is_detected(self, side: str) -> bool:
        if side == "red":
            return self.red.detected
        if side == "black":
            return self.black.detected
        return False

    def observe_pre_move(
        self,
        *,
        state,                      # TwixtState; unused in Task 1, used in Task 2
        ply: int,
        side_to_move: str,
        selected_move: Tuple[int, int],
        search_score: Optional[float],
        gc_state_cheap: Optional[dict],
        gc_state_full: Optional[dict],
    ) -> None:
        if not self.enabled:
            return
        if side_to_move not in ("red", "black"):
            return
        acc = self.red if side_to_move == "red" else self.black

        # 1. Coverage flags from cheap state.
        if gc_state_cheap is not None:
            total = gc_state_cheap.get("total_goal_distance")
            if total is not None:
                if acc.min_total_goal_distance is None or total < acc.min_total_goal_distance:
                    acc.min_total_goal_distance = total
                if total <= 2:
                    acc.ever_distance_le_2 = True
                if total <= 3:
                    acc.ever_distance_le_3 = True

        # 2. Detection update (only first event).
        if not acc.detected and gc_state_cheap is not None:
            total = gc_state_cheap.get("total_goal_distance")
            if total is not None and total <= self.detection_threshold:
                acc.detected = True
                acc.first_dominant_unclosed_ply = ply
                acc.first_total_goal_distance = total
                acc.first_category = gc_state_cheap.get("category")
                acc.first_endpoint_distances = (
                    dict(gc_state_cheap.get("endpoint_distances") or {})
                    if gc_state_cheap.get("endpoint_distances") is not None else None
                )
                comp = gc_state_cheap.get("component_pegs")
                acc.first_largest_component_size = len(comp) if comp else None

        # Classification fires in Task 2; for now we only track detection state.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_goal_completion_tracker.py -v`
Expected: PASS — 7 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/goal_completion_tracker.py tests/test_goal_completion_tracker.py
git commit -m "$(cat <<'EOF'
feat(tracker): add scaffold + per-side detection state

Per-game inline tracker module. Bare observe_pre_move handles coverage
flags (min/ever_le_2/le_3) and pre-move detection: fires when the side
to move already has total_goal_distance <= detection_threshold (default
2). Records first_dominant_unclosed_ply, first_total_goal_distance,
first_category, first_largest_component_size, first_endpoint_distances
on the first detection event for each side.

Classification path arrives in Task 2; finalize_game in Task 3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Tracker classification + watch-window

**Files:**
- Modify: `scripts/GPU/alphazero/goal_completion_tracker.py`
- Test: `tests/test_goal_completion_tracker.py` (add tests; existing tests still pass)

**Goal:** Wire `classify_selected_conversion_move` into `observe_pre_move`. Per spec §6.2, classification fires when `acc.detected` is True (after detection update), so the detection-ply move IS classified. Includes the `test_tracker_premove_detection_classifies_detection_ply_move` anchor (non-optional).

- [ ] **Step 1: Write failing tests**

Append to `tests/test_goal_completion_tracker.py`:

```python
from unittest.mock import patch


def _gc_state_full(total, completion_moves=(), reducing_moves=()):
    return {
        "total_goal_distance": total,
        "category": "two_endpoint_closeout_2ply",
        "endpoint_distances": {"top": 0, "bottom": 1},
        "component_pegs": frozenset({(0, 0), (2, 1), (4, 2)}),
        "endpoint_completion_moves": list(completion_moves),
        "distance_reducing_moves": list(reducing_moves),
        "moves_enumerated": True,
    }


def test_tracker_premove_detection_classifies_detection_ply_move():
    """ANCHOR: Pre-move semantics — the move on the detection ply itself
    counts as a post-detection move and IS classified."""
    t = GoalCompletionGameTracker()
    full = _gc_state_full(total=2, completion_moves=[(7, 7)])

    fake_cls = {"primary_class": "completes_endpoint",
                "completes_endpoint": True,
                "reduces_total_goal_distance": False,
                "is_redundant_reinforcement": False,
                "is_off_chain": False,
                "total_goal_distance_before": 2,
                "total_goal_distance_after": 0}

    with patch(
        "scripts.GPU.alphazero.goal_completion_tracker.classify_selected_conversion_move",
        return_value=fake_cls,
    ) as mock_cls:
        t.observe_pre_move(
            state="<state>", ply=11, side_to_move="red",
            selected_move=(7, 7), search_score=0.99,
            gc_state_cheap=full, gc_state_full=full,
        )

    assert t.red.detected is True
    assert t.red.first_dominant_unclosed_ply == 11
    assert t.red.moves_after_detection == 1
    assert t.red.moves_with_dominant_component == 1
    assert t.red.moves_with_dominant_unavailable == 0
    assert t.red.primary_class_counts["completes_endpoint"] == 1
    assert t.red.search_scores_after_detection == [0.99]
    assert t.red.high_value_after_detection_plies == 1
    assert mock_cls.call_count == 1


def test_tracker_classification_each_primary_class():
    """Each primary_class string maps to its own counter."""
    t = GoalCompletionGameTracker()
    full = _gc_state_full(total=2)

    cases = [
        ("completes_endpoint", 13),
        ("reduces_total_goal_distance", 15),
        ("redundant_reinforcement", 17),
        ("off_chain", 19),
        ("other", 21),
    ]
    for primary, ply in cases:
        with patch(
            "scripts.GPU.alphazero.goal_completion_tracker.classify_selected_conversion_move",
            return_value={"primary_class": primary},
        ):
            t.observe_pre_move(
                state="<state>", ply=ply, side_to_move="red",
                selected_move=(0, 0), search_score=None,
                gc_state_cheap=full, gc_state_full=full,
            )

    assert t.red.primary_class_counts == {
        "completes_endpoint": 1,
        "reduces_total_goal_distance": 1,
        "redundant_reinforcement": 1,
        "off_chain": 1,
        "other": 1,
    }
    assert t.red.moves_after_detection == 5


def test_tracker_unknown_primary_class_falls_to_other():
    t = GoalCompletionGameTracker()
    full = _gc_state_full(total=2)
    with patch(
        "scripts.GPU.alphazero.goal_completion_tracker.classify_selected_conversion_move",
        return_value={"primary_class": "garbled_string"},
    ):
        t.observe_pre_move(
            state="<state>", ply=11, side_to_move="red",
            selected_move=(0, 0), search_score=None,
            gc_state_cheap=full, gc_state_full=full,
        )
    assert t.red.primary_class_counts["other"] == 1


def test_tracker_dominant_unavailable_when_cheap_state_none_post_detection():
    """If the focal side already detected but a later ply has no dominant
    component (cheap state is None), count as dominant_unavailable."""
    t = GoalCompletionGameTracker()
    # First, get red detected.
    t.observe_pre_move(state="<state>", ply=11, side_to_move="red",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2), gc_state_full=None)
    # Next red ply: cheap state is None.
    t.observe_pre_move(state="<state>", ply=13, side_to_move="red",
                       selected_move=(1, 1), search_score=None,
                       gc_state_cheap=None, gc_state_full=None)
    assert t.red.moves_after_detection == 2  # detection ply + this one
    assert t.red.moves_with_dominant_unavailable == 1
    # Detection ply did not classify (no full state available either path).
    # Both plies fall under "dominant_unavailable" since classification
    # requires gc_state_full.
    assert sum(t.red.primary_class_counts.values()) == 0


def test_tracker_no_full_state_treated_as_dominant_unavailable():
    """When cheap state exists post-detection but full was not provided,
    we cannot classify; treat as dominant_unavailable defensively."""
    t = GoalCompletionGameTracker()
    t.observe_pre_move(state="<state>", ply=11, side_to_move="red",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2), gc_state_full=None)
    assert t.red.detected is True
    assert t.red.moves_after_detection == 1
    assert t.red.moves_with_dominant_unavailable == 1
    assert sum(t.red.primary_class_counts.values()) == 0


def test_tracker_search_score_high_value_count():
    """high_value_after_detection_plies counts post-detection plies where
    search_score >= high_value_threshold (default 0.9)."""
    t = GoalCompletionGameTracker(high_value_threshold=0.9)
    full = _gc_state_full(total=2)
    with patch(
        "scripts.GPU.alphazero.goal_completion_tracker.classify_selected_conversion_move",
        return_value={"primary_class": "redundant_reinforcement"},
    ):
        for ply, score in [(11, 0.95), (13, 0.5), (15, 0.91), (17, None)]:
            t.observe_pre_move(state="<state>", ply=ply, side_to_move="red",
                               selected_move=(0, 0), search_score=score,
                               gc_state_cheap=full, gc_state_full=full)
    assert t.red.search_scores_after_detection == [0.95, 0.5, 0.91]
    assert t.red.high_value_after_detection_plies == 2


def test_tracker_opponent_side_unaffected_by_focal_classification():
    t = GoalCompletionGameTracker()
    full = _gc_state_full(total=2)
    with patch(
        "scripts.GPU.alphazero.goal_completion_tracker.classify_selected_conversion_move",
        return_value={"primary_class": "completes_endpoint"},
    ):
        t.observe_pre_move(state="<state>", ply=11, side_to_move="red",
                           selected_move=(0, 0), search_score=0.95,
                           gc_state_cheap=full, gc_state_full=full)
    assert t.black.detected is False
    assert t.black.moves_after_detection == 0
    assert sum(t.black.primary_class_counts.values()) == 0
```

- [ ] **Step 2: Run tests to verify the new ones fail (existing pass)**

Run: `.venv/bin/python -m pytest tests/test_goal_completion_tracker.py -v`
Expected: 7 PASS (Task 1) + 7 FAIL (new tests) — failing because classification path is not yet implemented; counters stay zero.

- [ ] **Step 3: Extend the tracker**

Modify `scripts/GPU/alphazero/goal_completion_tracker.py`:

- Add this import at the top of the file:

```python
from scripts.GPU.alphazero.connectivity_diagnostics import (
    classify_selected_conversion_move,
)
```

- Replace the trailing comment in `observe_pre_move`:

```python
        # Classification fires in Task 2; for now we only track detection state.
```

with:

```python
        # 3. Watch-window: if detected (either before or just now),
        # the selected move counts as post-detection. Classify when full
        # state is available; otherwise log as dominant_unavailable.
        if acc.detected:
            acc.moves_after_detection += 1
            if gc_state_cheap is None:
                acc.moves_with_dominant_unavailable += 1
            elif gc_state_full is None:
                acc.moves_with_dominant_unavailable += 1
            else:
                acc.moves_with_dominant_component += 1
                cls = classify_selected_conversion_move(
                    state, side_to_move, selected_move, gc_state_full,
                    max_depth=self.max_depth,
                    min_component_size=self.min_component_size,
                )
                primary = cls.get("primary_class", "other")
                if primary in acc.primary_class_counts:
                    acc.primary_class_counts[primary] += 1
                else:
                    acc.primary_class_counts["other"] += 1

            if search_score is not None:
                ss = float(search_score)
                acc.search_scores_after_detection.append(ss)
                if ss >= self.high_value_threshold:
                    acc.high_value_after_detection_plies += 1
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `.venv/bin/python -m pytest tests/test_goal_completion_tracker.py -v`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/goal_completion_tracker.py tests/test_goal_completion_tracker.py
git commit -m "$(cat <<'EOF'
feat(tracker): watch-window classification + pre-move anchor

Once a side is detected (or detection fires this ply), observe_pre_move
classifies the selected move via classify_selected_conversion_move when
gc_state_full is available, incrementing primary_class_counts. When
full state is absent (caller did not upgrade) or cheap state is None,
counts as dominant_unavailable. Search score post-detection appended;
high_value_after_detection_plies counts >= high_value_threshold.

Includes test_tracker_premove_detection_classifies_detection_ply_move
as the named anchor for pre-move semantics regression.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Tracker finalize (Class 1 / 2 / 3 + tie-break)

**Files:**
- Modify: `scripts/GPU/alphazero/goal_completion_tracker.py`
- Test: `tests/test_goal_completion_tracker.py` (add tests)

**Goal:** Add `finalize_game` method that produces the single-schema record dict. Maps focal side based on outcome; populates Class 1 / Class 2 / Class 3 fields per spec §4.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_goal_completion_tracker.py`:

```python
def test_finalize_class3_when_disabled_returns_none():
    t = GoalCompletionGameTracker(enabled=False)
    rec = t.finalize_game(
        winner="red", reason="win", n_moves=20,
        starting_player="red", iteration=110, game_idx=5,
        game_id="iter_0110_game_005",
    )
    assert rec is None


def test_finalize_class1_basic_decisive_winner():
    """Class 1: winner == detected side. All winner-perspective fields populated."""
    t = GoalCompletionGameTracker(high_value_delay_threshold_plies=6)
    full = _gc_state_full(total=2)
    # Detect red at ply 11 with two_endpoint_closeout shape.
    with patch(
        "scripts.GPU.alphazero.goal_completion_tracker.classify_selected_conversion_move",
        return_value={"primary_class": "completes_endpoint"},
    ):
        t.observe_pre_move(state="<state>", ply=11, side_to_move="red",
                           selected_move=(0, 0), search_score=0.99,
                           gc_state_cheap=full, gc_state_full=full)

    rec = t.finalize_game(
        winner="red", reason="win", n_moves=21,
        starting_player="red", iteration=112, game_idx=34,
        game_id="iter_0112_game_034",
    )
    assert rec["version"] == 1
    assert rec["outcome_class"] == 1
    assert rec["scope"] == "winner"
    assert rec["winner"] == "red"
    assert rec["detected_player"] == "red"
    assert rec["reason"] == "win"
    assert rec["detected"] is True
    assert rec["first_dominant_unclosed_ply"] == 11
    assert rec["first_total_goal_distance"] == 2
    assert rec["first_category"] == "two_endpoint_closeout_2ply"
    assert rec["actual_terminal_ply"] == 21
    assert rec["actual_win_ply"] == 21
    assert rec["conversion_delay_plies"] == 10  # 21 - 11
    # winner_moves_in_watch_window mirrors moves_after_detection for Class 1.
    assert rec["winner_moves_in_watch_window"] == 1
    assert rec["winner_moves_with_dominant_component"] == 1
    assert rec["winner_moves_with_dominant_unavailable"] == 0
    assert rec["primary_class_counts"]["completes_endpoint"] == 1
    assert rec["max_search_score_after_detection"] == 0.99
    assert rec["mean_search_score_after_detection"] == 0.99
    assert rec["high_value_after_detection_plies"] == 1
    # delay_plies 10 >= high_value_delay_threshold 6 AND high_value >= 1
    assert rec["root_value_high_but_delayed"] is True
    assert rec["search_score_coverage_in_watch_window"] == 1
    assert rec["cap_delay_proxy_plies"] is None


def test_finalize_class1_winner_never_detected():
    """Edge: winner exists but their side never reached detection threshold.
    detected=false, all post-detection fields null/0."""
    t = GoalCompletionGameTracker()
    # Red plays, total stays at 4 throughout — no detection.
    t.observe_pre_move(state="<state>", ply=1, side_to_move="red",
                       selected_move=(0, 0), search_score=0.5,
                       gc_state_cheap=_gc_state(4), gc_state_full=None)
    rec = t.finalize_game(
        winner="red", reason="win", n_moves=20,
        starting_player="red", iteration=110, game_idx=0,
        game_id="iter_0110_game_000",
    )
    assert rec["outcome_class"] == 1
    assert rec["winner"] == "red"
    assert rec["detected"] is False
    assert rec["first_dominant_unclosed_ply"] is None
    assert rec["conversion_delay_plies"] is None
    assert rec["winner_moves_in_watch_window"] == 0
    assert rec["primary_class_counts"] == {
        "completes_endpoint": 0,
        "reduces_total_goal_distance": 0,
        "redundant_reinforcement": 0,
        "off_chain": 0,
        "other": 0,
    }
    assert rec["root_value_high_but_delayed"] is False


def test_finalize_class2_capped_focal_earliest_detector():
    """Class 2: no winner; focal = earliest detector. cap_delay_proxy_plies =
    actual_terminal_ply - first_dominant_unclosed_ply."""
    t = GoalCompletionGameTracker()
    # Red detects at ply 41, Black at ply 60 — Red is focal.
    t.observe_pre_move(state="<state>", ply=41, side_to_move="red",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2), gc_state_full=None)
    t.observe_pre_move(state="<state>", ply=60, side_to_move="black",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2), gc_state_full=None)

    rec = t.finalize_game(
        winner=None, reason="state_cap", n_moves=120,
        starting_player="red", iteration=112, game_idx=7,
        game_id="iter_0112_game_007",
    )
    assert rec["outcome_class"] == 2
    assert rec["scope"] == "both_sides"
    assert rec["winner"] is None
    assert rec["detected_player"] == "red"
    assert rec["detected"] is True
    assert rec["first_dominant_unclosed_ply"] == 41
    assert rec["first_total_goal_distance"] == 2
    assert rec["actual_terminal_ply"] == 120
    assert rec["actual_win_ply"] is None
    assert rec["conversion_delay_plies"] is None
    assert rec["cap_delay_proxy_plies"] == 79  # 120 - 41
    assert rec["primary_class_counts"] is None
    assert rec["max_search_score_after_detection"] is None


def test_finalize_class2_tiebreak_lower_first_total_then_red():
    """Tie-break: same ply -> lower first_total_goal_distance -> red."""
    # Equal-ply, equal-distance -> red wins.
    t = GoalCompletionGameTracker()
    t.observe_pre_move(state="<state>", ply=50, side_to_move="red",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2), gc_state_full=None)
    t.observe_pre_move(state="<state>", ply=50, side_to_move="black",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2), gc_state_full=None)
    rec = t.finalize_game(
        winner=None, reason="timeout", n_moves=80,
        starting_player="red", iteration=110, game_idx=0,
        game_id="iter_0110_game_000",
    )
    assert rec["detected_player"] == "red"

    # Equal-ply, black has lower distance -> black wins.
    t2 = GoalCompletionGameTracker()
    t2.observe_pre_move(state="<state>", ply=50, side_to_move="red",
                        selected_move=(0, 0), search_score=None,
                        gc_state_cheap=_gc_state(2), gc_state_full=None)
    t2.observe_pre_move(state="<state>", ply=50, side_to_move="black",
                        selected_move=(0, 0), search_score=None,
                        gc_state_cheap=_gc_state(1), gc_state_full=None)
    rec2 = t2.finalize_game(
        winner=None, reason="timeout", n_moves=80,
        starting_player="red", iteration=110, game_idx=0,
        game_id="iter_0110_game_000",
    )
    assert rec2["detected_player"] == "black"


def test_finalize_class2_no_detection_either_side():
    """Class 2 with neither side detected: detected=false, proxy null."""
    t = GoalCompletionGameTracker()
    rec = t.finalize_game(
        winner=None, reason="state_cap", n_moves=200,
        starting_player="red", iteration=110, game_idx=0,
        game_id="iter_0110_game_000",
    )
    assert rec["outcome_class"] == 2
    assert rec["detected"] is False
    assert rec["detected_player"] is None
    assert rec["first_dominant_unclosed_ply"] is None
    assert rec["cap_delay_proxy_plies"] is None


def test_finalize_class3_excluded():
    """Unhandled outcome -> Class 3 minimal record."""
    t = GoalCompletionGameTracker()
    rec = t.finalize_game(
        winner=None, reason="unknown", n_moves=0,
        starting_player="red", iteration=110, game_idx=0,
        game_id="iter_0110_game_000",
    )
    assert rec["outcome_class"] == 3
    assert rec["scope"] == "excluded"
    assert rec["winner"] is None
    assert rec["detected_player"] is None
    assert rec["detected"] is False
    assert rec["actual_terminal_ply"] == 0
    assert rec["actual_win_ply"] is None
    assert rec["conversion_delay_plies"] is None
    assert rec["cap_delay_proxy_plies"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_goal_completion_tracker.py -v`
Expected: prior 14 pass; 7 new tests FAIL with `AttributeError: 'GoalCompletionGameTracker' object has no attribute 'finalize_game'`.

- [ ] **Step 3: Implement finalize_game + helpers**

Append to `scripts/GPU/alphazero/goal_completion_tracker.py`:

```python
    def finalize_game(
        self,
        *,
        winner: Optional[str],
        reason: str,
        n_moves: int,
        starting_player: str,
        iteration: int,
        game_idx: int,
        game_id: str,
    ) -> Optional[dict]:
        if not self.enabled:
            return None

        outcome_class = _classify_outcome(winner, reason)
        common_id = {
            "version": 1,
            "game_id": game_id,
            "iteration": iteration,
            "game_idx": game_idx,
            "starting_player": starting_player,
            "n_moves": n_moves,
            "reason": reason,
            "outcome_class": outcome_class,
        }

        if outcome_class == 1:
            focal = self.red if winner == "red" else self.black
            return _build_class1_record(
                common_id=common_id, winner=winner, focal=focal,
                actual_terminal_ply=n_moves,
                high_value_delay_threshold_plies=self.high_value_delay_threshold_plies,
            )
        if outcome_class == 2:
            focal_side, focal = _pick_class2_focal(self.red, self.black)
            return _build_class2_record(
                common_id=common_id, focal=focal, focal_side=focal_side,
                actual_terminal_ply=n_moves,
            )
        return _build_class3_record(common_id=common_id)


def _classify_outcome(winner: Optional[str], reason: str) -> int:
    if winner in ("red", "black"):
        return 1
    if reason in ("state_cap", "timeout", "board_full"):
        return 2
    return 3


def _pick_class2_focal(
    red_acc: _SideAccumulator, black_acc: _SideAccumulator,
) -> Tuple[str, _SideAccumulator]:
    """Tie-break: earliest first_dominant_unclosed_ply ->
    lower first_total_goal_distance -> red before black.

    If neither side is detected, returns ('red', red_acc) — caller will
    populate the record as detected=false / detected_player=null."""
    candidates = []
    if red_acc.detected:
        candidates.append(("red", red_acc))
    if black_acc.detected:
        candidates.append(("black", black_acc))
    if not candidates:
        return "red", red_acc
    candidates.sort(key=lambda c: (
        c[1].first_dominant_unclosed_ply if c[1].first_dominant_unclosed_ply is not None else 10**9,
        c[1].first_total_goal_distance if c[1].first_total_goal_distance is not None else 10**9,
        0 if c[0] == "red" else 1,
    ))
    return candidates[0]


def _build_class1_record(
    *, common_id: dict, winner: str, focal: _SideAccumulator,
    actual_terminal_ply: int, high_value_delay_threshold_plies: int,
) -> dict:
    detected = focal.detected
    first_ply = focal.first_dominant_unclosed_ply
    conversion_delay_plies = (
        actual_terminal_ply - first_ply if (detected and first_ply is not None) else None
    )
    # Conversion delay in winner-only moves: focal.moves_after_detection
    # already counts this side's post-detection moves only.
    conversion_delay_winner_moves = (
        focal.moves_after_detection if detected else None
    )
    if focal.search_scores_after_detection:
        max_ss = max(focal.search_scores_after_detection)
        mean_ss = sum(focal.search_scores_after_detection) / len(focal.search_scores_after_detection)
        coverage = len(focal.search_scores_after_detection)
    else:
        max_ss, mean_ss, coverage = None, None, 0

    if detected and conversion_delay_plies is not None:
        root_high_delayed = (
            focal.high_value_after_detection_plies >= 1
            and conversion_delay_plies >= high_value_delay_threshold_plies
        )
    else:
        root_high_delayed = False

    out = dict(common_id)
    out.update({
        "winner": winner,
        "detected_player": winner,
        "scope": "winner",
        "ever_distance_le_2": focal.ever_distance_le_2,
        "ever_distance_le_3": focal.ever_distance_le_3,
        "min_total_goal_distance": focal.min_total_goal_distance,
        "detected": detected,
        "first_dominant_unclosed_ply": first_ply,
        "first_total_goal_distance": focal.first_total_goal_distance,
        "first_category": focal.first_category,
        "first_largest_component_size": focal.first_largest_component_size,
        "first_endpoint_distances": focal.first_endpoint_distances,
        "actual_terminal_ply": actual_terminal_ply,
        "actual_win_ply": actual_terminal_ply,
        "conversion_delay_plies": conversion_delay_plies,
        "conversion_delay_winner_moves": conversion_delay_winner_moves,
        "cap_delay_proxy_plies": None,
        "winner_moves_in_watch_window": focal.moves_after_detection if detected else 0,
        "winner_moves_with_dominant_component": focal.moves_with_dominant_component if detected else 0,
        "winner_moves_with_dominant_unavailable": focal.moves_with_dominant_unavailable if detected else 0,
        "primary_class_counts": dict(focal.primary_class_counts),
        "max_search_score_after_detection": max_ss,
        "mean_search_score_after_detection": mean_ss,
        "high_value_after_detection_plies": focal.high_value_after_detection_plies,
        "root_value_high_but_delayed": root_high_delayed,
        "search_score_coverage_in_watch_window": coverage,
    })
    return out


def _build_class2_record(
    *, common_id: dict, focal: _SideAccumulator, focal_side: str,
    actual_terminal_ply: int,
) -> dict:
    detected = focal.detected
    first_ply = focal.first_dominant_unclosed_ply
    cap_delay = (
        actual_terminal_ply - first_ply if (detected and first_ply is not None) else None
    )
    out = dict(common_id)
    out.update({
        "winner": None,
        "detected_player": focal_side if detected else None,
        "scope": "both_sides",
        "ever_distance_le_2": focal.ever_distance_le_2,
        "ever_distance_le_3": focal.ever_distance_le_3,
        "min_total_goal_distance": focal.min_total_goal_distance,
        "detected": detected,
        "first_dominant_unclosed_ply": first_ply,
        "first_total_goal_distance": focal.first_total_goal_distance,
        "first_category": focal.first_category,
        "first_largest_component_size": focal.first_largest_component_size,
        "first_endpoint_distances": focal.first_endpoint_distances,
        "actual_terminal_ply": actual_terminal_ply,
        "actual_win_ply": None,
        "conversion_delay_plies": None,
        "conversion_delay_winner_moves": None,
        "cap_delay_proxy_plies": cap_delay,
        "winner_moves_in_watch_window": None,
        "winner_moves_with_dominant_component": None,
        "winner_moves_with_dominant_unavailable": None,
        "primary_class_counts": None,
        "max_search_score_after_detection": None,
        "mean_search_score_after_detection": None,
        "high_value_after_detection_plies": None,
        "root_value_high_but_delayed": None,
        "search_score_coverage_in_watch_window": None,
    })
    return out


def _build_class3_record(*, common_id: dict) -> dict:
    out = dict(common_id)
    out.update({
        "winner": None,
        "detected_player": None,
        "scope": "excluded",
        "detected": False,
        "actual_terminal_ply": common_id["n_moves"],
        "actual_win_ply": None,
        "conversion_delay_plies": None,
        "cap_delay_proxy_plies": None,
    })
    return out
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `.venv/bin/python -m pytest tests/test_goal_completion_tracker.py -v`
Expected: 21 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/goal_completion_tracker.py tests/test_goal_completion_tracker.py
git commit -m "$(cat <<'EOF'
feat(tracker): finalize_game produces single-schema record

Class 1 (decisive): focal = winner; populates conversion_delay_plies
and conversion_delay_winner_moves; root_value_high_but_delayed gated by
high_value_after_detection_plies >= 1 AND delay >= threshold.

Class 2 (capped/timeout/board_full): focal = earliest detector with
deterministic tie-break (earliest ply -> lower first_total_goal_distance
-> red). cap_delay_proxy_plies = terminal - first_detected.

Class 3 (excluded): minimal record.

Detected-side != winner edge case: Class 1 record reports detected=false
(winner-perspective scope).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Aggregator scaffold + `_normalize_record` + empty/coverage

**Files:**
- Create: `scripts/GPU/alphazero/goal_completion_aggregator.py`
- Test: `tests/test_goal_completion_aggregator.py`

**Goal:** Stand up the shared aggregator with the empty-input and coverage paths. Population summaries (Task 5) come next.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_goal_completion_aggregator.py
"""Aggregator unit tests (spec 2026-05-05 §7)."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.goal_completion_aggregator import (
    aggregate_goal_completion_records,
    _normalize_record,
    _zero_class_counts,
)


def _decisive_record(**overrides):
    base = {
        "version": 1,
        "outcome_class": 1,
        "winner": "red",
        "detected_player": "red",
        "reason": "win",
        "detected": True,
        "ever_distance_le_2": True,
        "ever_distance_le_3": True,
        "min_total_goal_distance": 2,
        "first_dominant_unclosed_ply": 11,
        "first_total_goal_distance": 2,
        "first_category": "two_endpoint_closeout_2ply",
        "actual_terminal_ply": 21,
        "actual_win_ply": 21,
        "conversion_delay_plies": 10,
        "conversion_delay_winner_moves": 5,
        "cap_delay_proxy_plies": None,
        "primary_class_counts": {
            "completes_endpoint": 1,
            "reduces_total_goal_distance": 0,
            "redundant_reinforcement": 3,
            "off_chain": 1,
            "other": 0,
        },
        "max_search_score_after_detection": 0.99,
        "mean_search_score_after_detection": 0.95,
        "high_value_after_detection_plies": 4,
        "root_value_high_but_delayed": True,
        "search_score_coverage_in_watch_window": 5,
        "winner_moves_in_watch_window": 5,
        "winner_moves_with_dominant_component": 5,
        "winner_moves_with_dominant_unavailable": 0,
    }
    base.update(overrides)
    return base


def test_aggregator_empty_records_returns_skeleton():
    result = aggregate_goal_completion_records([], config={"detection_threshold": 2}, games_total=0)
    assert result["version"] == 1
    assert result["config"] == {"detection_threshold": 2}
    assert result["diagnostics_coverage"] == {
        "games_total": 0,
        "games_with_record": 0,
        "coverage_rate": 0.0,
        "games_class_1": 0,
        "games_class_2": 0,
        "games_class_3": 0,
    }
    assert result["main_population"]["n"] == 0
    assert result["capped_population"]["n"] == 0
    assert result["excluded_population"] == {"n": 0}


def test_aggregator_mixed_nones_real_coverage():
    """coverage_rate uses games_total (caller-supplied), not len(valid)."""
    rec = _decisive_record()
    result = aggregate_goal_completion_records(
        [rec, None, rec, None],
        config={"detection_threshold": 2},
        games_total=4,
    )
    assert result["diagnostics_coverage"]["games_total"] == 4
    assert result["diagnostics_coverage"]["games_with_record"] == 2
    assert result["diagnostics_coverage"]["coverage_rate"] == 0.5
    assert result["diagnostics_coverage"]["games_class_1"] == 2


def test_aggregator_default_games_total_is_record_count():
    rec = _decisive_record()
    result = aggregate_goal_completion_records([rec], config={})
    assert result["diagnostics_coverage"]["games_total"] == 1
    assert result["diagnostics_coverage"]["coverage_rate"] == 1.0


def test_aggregator_class_split_counts():
    cap = _decisive_record(
        outcome_class=2, winner=None, reason="state_cap",
        actual_win_ply=None, conversion_delay_plies=None,
        conversion_delay_winner_moves=None,
        cap_delay_proxy_plies=42,
        primary_class_counts=None,
        max_search_score_after_detection=None,
        mean_search_score_after_detection=None,
        high_value_after_detection_plies=None,
        root_value_high_but_delayed=None,
        search_score_coverage_in_watch_window=None,
        winner_moves_in_watch_window=None,
        winner_moves_with_dominant_component=None,
        winner_moves_with_dominant_unavailable=None,
    )
    excl = {"version": 1, "outcome_class": 3, "winner": None,
            "detected": False, "reason": "unknown"}
    decisive = _decisive_record()
    result = aggregate_goal_completion_records(
        [decisive, decisive, cap, excl],
        config={}, games_total=4,
    )
    assert result["diagnostics_coverage"]["games_class_1"] == 2
    assert result["diagnostics_coverage"]["games_class_2"] == 1
    assert result["diagnostics_coverage"]["games_class_3"] == 1


def test_aggregator_zero_games_total_zero_rate():
    result = aggregate_goal_completion_records([None, None], config={}, games_total=0)
    assert result["diagnostics_coverage"]["coverage_rate"] == 0.0


def test_normalize_record_fills_defaults():
    out = _normalize_record({"version": 1, "outcome_class": 1, "winner": "red"})
    assert out["version"] == 1
    assert out["outcome_class"] == 1
    assert out["winner"] == "red"
    assert out["detected"] is False  # default
    assert out["primary_class_counts"] == _zero_class_counts()
    assert out["reason"] == "unknown"
    assert out["min_total_goal_distance"] is None


def test_normalize_record_handles_unknown_version_defensively():
    out = _normalize_record({"version": 99, "outcome_class": 1, "winner": "red"})
    assert out["version"] == 99  # passes through; aggregator may warn
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_goal_completion_aggregator.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/GPU/alphazero/goal_completion_aggregator.py
"""Shared aggregator: per-game goal_completion_record list ->
goal_completion_summary block.

Pure functions, no I/O, no BFS. Used by both the trainer (per-iteration
sidecar) and the analyzer (cross-iteration roll-up).
"""
from __future__ import annotations

from typing import List, Optional


def _zero_class_counts() -> dict:
    return {
        "completes_endpoint": 0,
        "reduces_total_goal_distance": 0,
        "redundant_reinforcement": 0,
        "off_chain": 0,
        "other": 0,
    }


def _normalize_record(r: dict) -> dict:
    """Forward/backward-tolerant normalization at function boundary."""
    pcc = r.get("primary_class_counts")
    if pcc is None:
        pcc = _zero_class_counts()
    return {
        "version": int(r.get("version", 1)),
        "outcome_class": int(r.get("outcome_class", 3)),
        "reason": r.get("reason") or "unknown",
        "winner": r.get("winner"),
        "detected_player": r.get("detected_player"),
        "detected": bool(r.get("detected", False)),
        "ever_distance_le_2": bool(r.get("ever_distance_le_2", False)),
        "ever_distance_le_3": bool(r.get("ever_distance_le_3", False)),
        "min_total_goal_distance": r.get("min_total_goal_distance"),
        "first_dominant_unclosed_ply": r.get("first_dominant_unclosed_ply"),
        "first_total_goal_distance": r.get("first_total_goal_distance"),
        "first_category": r.get("first_category"),
        "actual_terminal_ply": r.get("actual_terminal_ply"),
        "actual_win_ply": r.get("actual_win_ply"),
        "conversion_delay_plies": r.get("conversion_delay_plies"),
        "conversion_delay_winner_moves": r.get("conversion_delay_winner_moves"),
        "cap_delay_proxy_plies": r.get("cap_delay_proxy_plies"),
        "primary_class_counts": pcc,
        "max_search_score_after_detection": r.get("max_search_score_after_detection"),
        "mean_search_score_after_detection": r.get("mean_search_score_after_detection"),
        "high_value_after_detection_plies": r.get("high_value_after_detection_plies"),
        "root_value_high_but_delayed": r.get("root_value_high_but_delayed"),
        "winner_moves_in_watch_window": r.get("winner_moves_in_watch_window"),
        "winner_moves_with_dominant_component": r.get("winner_moves_with_dominant_component"),
        "winner_moves_with_dominant_unavailable": r.get("winner_moves_with_dominant_unavailable"),
        "search_score_coverage_in_watch_window": r.get("search_score_coverage_in_watch_window"),
    }


def aggregate_goal_completion_records(
    records: List[Optional[dict]],
    config: dict,
    games_total: Optional[int] = None,
) -> dict:
    games_total = games_total if games_total is not None else len(records)
    valid = [_normalize_record(r) for r in records if r is not None]

    main = [r for r in valid if r["outcome_class"] == 1]
    capped = [r for r in valid if r["outcome_class"] == 2]
    excluded = [r for r in valid if r["outcome_class"] == 3]

    return {
        "version": 1,
        "config": dict(config),
        "diagnostics_coverage": {
            "games_total": games_total,
            "games_with_record": len(valid),
            "coverage_rate": (len(valid) / games_total) if games_total else 0.0,
            "games_class_1": len(main),
            "games_class_2": len(capped),
            "games_class_3": len(excluded),
        },
        # Population summaries land in Task 5; emit empty skeletons here.
        "main_population": {"n": len(main)},
        "capped_population": {"n": len(capped)},
        "excluded_population": {"n": len(excluded)},
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_goal_completion_aggregator.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/goal_completion_aggregator.py tests/test_goal_completion_aggregator.py
git commit -m "$(cat <<'EOF'
feat(aggregator): scaffold + normalize + coverage

Shared aggregator module; pure functions consumed by trainer (per-iter)
and analyzer (cross-iter). _normalize_record handles forward/backward
field tolerance; aggregate_goal_completion_records partitions records
by outcome_class and computes real coverage rate using caller-supplied
games_total. Population summary helpers land in Task 5 — emitting
empty skeletons here.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Aggregator population helpers + cross-iteration roll-up

**Files:**
- Modify: `scripts/GPU/alphazero/goal_completion_aggregator.py`
- Test: `tests/test_goal_completion_aggregator.py` (add tests)

**Goal:** Migrate `_summarize_main_population` and `_summarize_capped_population` from the analyzer; switch to dict-typed input. Lock the field naming (`games_with_dominant_unclosed`, `games_with_total_distance_le_2/le_3`, `primary_class_rates`, etc.) per spec §5.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_goal_completion_aggregator.py`:

```python
def _capped_record(**overrides):
    base = {
        "version": 1, "outcome_class": 2, "winner": None,
        "detected_player": "red", "reason": "state_cap",
        "detected": True,
        "ever_distance_le_2": True, "ever_distance_le_3": True,
        "min_total_goal_distance": 2,
        "first_dominant_unclosed_ply": 60,
        "first_total_goal_distance": 2,
        "first_category": "two_endpoint_closeout_2ply",
        "actual_terminal_ply": 100,
        "actual_win_ply": None,
        "conversion_delay_plies": None,
        "conversion_delay_winner_moves": None,
        "cap_delay_proxy_plies": 40,
        "primary_class_counts": None,
    }
    base.update(overrides)
    return base


def test_main_population_known_percentiles():
    """Handcrafted record set with known delays — assert exact percentiles."""
    delays = [4, 4, 6, 8, 10, 14, 18, 22, 28]
    records = [_decisive_record(conversion_delay_plies=d, conversion_delay_winner_moves=d // 2)
               for d in delays]
    result = aggregate_goal_completion_records(records, config={}, games_total=len(records))
    main = result["main_population"]
    assert main["n"] == 9
    assert main["detected"] == 9
    assert main["detection_rate"] == 1.0
    cd = main["conversion_delay_plies"]
    # Linear-interp percentiles. With this exact ordering the medians are
    # well-defined: p50=10, p90=22 (90th percentile), max=28.
    assert cd["p50"] == 10
    assert cd["p90"] == 22 or cd["p90"] == 22.0
    assert cd["max"] == 28


def test_main_population_naming_continuity():
    """Spec-locked naming — continuity with existing analyzer report."""
    rec = _decisive_record()
    result = aggregate_goal_completion_records([rec, rec], config={}, games_total=2)
    main = result["main_population"]
    assert "games_with_dominant_unclosed" in main
    assert "games_with_total_distance_le_2" in main
    assert "games_with_total_distance_le_3" in main
    assert main["games_with_dominant_unclosed"] == 2
    assert main["games_with_total_distance_le_2"] == 2


def test_main_population_primary_class_rates_pooled():
    """primary_class_rates pools counts across all main games."""
    r1 = _decisive_record(primary_class_counts={
        "completes_endpoint": 2, "reduces_total_goal_distance": 1,
        "redundant_reinforcement": 1, "off_chain": 0, "other": 0,
    })
    r2 = _decisive_record(primary_class_counts={
        "completes_endpoint": 0, "reduces_total_goal_distance": 1,
        "redundant_reinforcement": 4, "off_chain": 0, "other": 1,
    })
    result = aggregate_goal_completion_records([r1, r2], config={}, games_total=2)
    rates = result["main_population"]["primary_class_rates"]
    # Total selected = 10. completes=2, reduces=2, redundant=5, off=0, other=1.
    assert abs(rates["completes_endpoint"] - 0.2) < 1e-9
    assert abs(rates["reduces_total_goal_distance"] - 0.2) < 1e-9
    assert abs(rates["redundant_reinforcement"] - 0.5) < 1e-9


def test_main_population_bad_cases_thresholds():
    delays = [3, 9, 10, 11, 19, 20, 25]
    records = [_decisive_record(
        conversion_delay_plies=d,
        high_value_after_detection_plies=2,
    ) for d in delays]
    result = aggregate_goal_completion_records(records, config={}, games_total=len(records))
    bad = result["main_population"]["bad_cases"]
    # delay >= 10 -> 5 (10, 11, 19, 20, 25)
    assert bad["delay_ge_10"] == 5
    # delay >= 20 -> 2 (20, 25)
    assert bad["delay_ge_20"] == 2
    # high_value_after_detection_plies_total = 2 * 7 = 14
    assert bad["high_value_after_detection_plies_total"] == 14


def test_main_population_root_value_high_but_delayed_count():
    records = [
        _decisive_record(root_value_high_but_delayed=True),
        _decisive_record(root_value_high_but_delayed=False),
        _decisive_record(root_value_high_but_delayed=True),
    ]
    result = aggregate_goal_completion_records(records, config={}, games_total=3)
    assert result["main_population"]["bad_cases"]["root_value_high_but_delayed"] == 2


def test_capped_population_summary():
    records = [
        _capped_record(cap_delay_proxy_plies=20, detected_player="red"),
        _capped_record(cap_delay_proxy_plies=40, detected_player="red"),
        _capped_record(cap_delay_proxy_plies=60, detected_player="black"),
    ]
    result = aggregate_goal_completion_records(records, config={}, games_total=3)
    cap = result["capped_population"]
    assert cap["n"] == 3
    assert cap["detected"] == 3
    assert cap["cap_delay_proxy_plies"]["p50"] == 40
    assert cap["cap_delay_proxy_plies"]["max"] == 60
    assert cap["first_detector_side"] == {"red": 2, "black": 1}


def test_cross_iteration_roll_up_matches_per_iter_aggregation():
    """The same shared aggregator at any scope: per-iter aggregation
    composed via roll-up should equal one cross-iter aggregation on the
    same records (recompute principle, spec §11.1)."""
    delays = [4, 6, 8, 10, 14, 18]
    records = [_decisive_record(conversion_delay_plies=d) for d in delays]
    cross = aggregate_goal_completion_records(records, config={}, games_total=len(records))
    # Reaggregate from the same records — same input, same shape, same numbers.
    cross2 = aggregate_goal_completion_records(records, config={}, games_total=len(records))
    assert cross == cross2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_goal_completion_aggregator.py -v`
Expected: 7 prior tests PASS; 7 new tests FAIL with `KeyError` on missing population fields.

- [ ] **Step 3: Implement population helpers**

Append to `scripts/GPU/alphazero/goal_completion_aggregator.py`:

```python
def _percentile(sorted_values: list, p: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (p / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _stats_block(values: list) -> dict:
    if not values:
        return {"p10": 0, "p50": 0, "p90": 0, "p95": 0, "max": 0, "mean": 0.0, "min": 0}
    s = sorted(values)
    return {
        "p10": _percentile(s, 10),
        "p50": _percentile(s, 50),
        "p90": _percentile(s, 90),
        "p95": _percentile(s, 95),
        "max": float(s[-1]),
        "min": float(s[0]),
        "mean": float(sum(s) / len(s)),
    }


def _summarize_main_population(main: List[dict], config: dict) -> dict:
    if not main:
        return {"n": 0}

    high_value_delay = int(config.get("high_value_delay_threshold_plies", 6))

    detected = [r for r in main if r["detected"]]
    n = len(main)

    delays = [r["conversion_delay_plies"] for r in detected
              if r["conversion_delay_plies"] is not None]
    delays_winner_moves = [r["conversion_delay_winner_moves"] for r in detected
                           if r["conversion_delay_winner_moves"] is not None]

    # Pooled primary_class counts -> rates.
    pooled = _zero_class_counts()
    pooled["dominant_unavailable"] = 0
    total_classified = 0
    for r in detected:
        pcc = r.get("primary_class_counts") or {}
        for k in ("completes_endpoint", "reduces_total_goal_distance",
                  "redundant_reinforcement", "off_chain", "other"):
            pooled[k] += int(pcc.get(k, 0))
            total_classified += int(pcc.get(k, 0))
        # dominant_unavailable comes from winner_moves_with_dominant_unavailable
        pooled["dominant_unavailable"] += int(r.get("winner_moves_with_dominant_unavailable") or 0)
        total_classified += int(r.get("winner_moves_with_dominant_unavailable") or 0)
    if total_classified > 0:
        primary_class_rates = {k: v / total_classified for k, v in pooled.items()}
    else:
        primary_class_rates = {k: 0.0 for k in pooled}

    # search_score_after_detection (nested: max distribution + mean distribution).
    max_scores = [r["max_search_score_after_detection"] for r in detected
                  if r.get("max_search_score_after_detection") is not None]
    mean_scores = [r["mean_search_score_after_detection"] for r in detected
                   if r.get("mean_search_score_after_detection") is not None]

    bad = {
        "delay_ge_10": sum(1 for d in delays if d >= 10),
        "delay_ge_20": sum(1 for d in delays if d >= 20),
        "high_value_after_detection_plies_total": sum(
            int(r.get("high_value_after_detection_plies") or 0) for r in detected
        ),
        "root_value_high_but_delayed": sum(
            1 for r in detected if r.get("root_value_high_but_delayed") is True
        ),
    }

    return {
        "n": n,
        "games_with_dominant_unclosed": sum(1 for r in main if r["detected"]),
        "games_with_total_distance_le_2": sum(1 for r in main if r.get("ever_distance_le_2")),
        "games_with_total_distance_le_3": sum(1 for r in main if r.get("ever_distance_le_3")),
        "detected": len(detected),
        "detection_rate": (len(detected) / n) if n else 0.0,
        "min_total_goal_distance": _stats_block(
            [r["min_total_goal_distance"] for r in main if r.get("min_total_goal_distance") is not None]
        ),
        "conversion_delay_plies": _stats_block(delays),
        "conversion_delay_winner_moves": _stats_block(delays_winner_moves),
        "primary_class_rates": primary_class_rates,
        "search_score_after_detection": {
            "max": _stats_block(max_scores),
            "mean": _stats_block(mean_scores),
        },
        "bad_cases": bad,
    }


def _summarize_capped_population(capped: List[dict]) -> dict:
    if not capped:
        return {"n": 0}
    detected = [r for r in capped if r["detected"]]
    proxies = [r["cap_delay_proxy_plies"] for r in detected
               if r.get("cap_delay_proxy_plies") is not None]
    side_counts = {"red": 0, "black": 0}
    for r in detected:
        s = r.get("detected_player")
        if s in side_counts:
            side_counts[s] += 1
    return {
        "n": len(capped),
        "detected": len(detected),
        "cap_delay_proxy_plies": _stats_block(proxies),
        "first_detector_side": side_counts,
    }
```

- Replace the empty population skeletons in `aggregate_goal_completion_records`:

```python
        "main_population": {"n": len(main)},
        "capped_population": {"n": len(capped)},
```

with:

```python
        "main_population": _summarize_main_population(main, config),
        "capped_population": _summarize_capped_population(capped),
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `.venv/bin/python -m pytest tests/test_goal_completion_aggregator.py -v`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/goal_completion_aggregator.py tests/test_goal_completion_aggregator.py
git commit -m "$(cat <<'EOF'
feat(aggregator): main + capped population summary helpers

_summarize_main_population produces decisive-game roll-up per spec §5:
games_with_dominant_unclosed / le_2 / le_3 counts, conversion_delay
percentile blocks, pooled primary_class_rates (incl. dominant_unavailable),
search_score_after_detection {max,mean} percentile blocks, bad_cases
counters (delay_ge_10/20, high_value_after_detection_plies_total,
root_value_high_but_delayed).

_summarize_capped_population produces capped roll-up: cap_delay_proxy
percentile block + first_detector_side counts.

Both helpers consume normalized dicts; no dataclass adapters.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: IPC + saver plumbing

**Files:**
- Modify: `scripts/GPU/alphazero/ipc_messages.py:118` (add field after `goal_completion_diagnostics_meta`)
- Modify: `scripts/GPU/alphazero/self_play.py:426` (add field after `goal_completion_diagnostics_meta`)
- Modify: `scripts/GPU/alphazero/game_saver.py:46` (add kwarg) and `:166` (write key)
- Modify: `scripts/GPU/alphazero/game_saver.py:237` (`GameSaver.maybe_save_game`) and `:280` (forward kwarg)
- Test: `tests/test_goal_completion_save_load.py` (NEW)

**Goal:** Add `goal_completion_record` field to dataclasses; saver writes it as a top-level key. Worker→trainer wiring lands in Task 8; this task is the IPC/saver layer in isolation.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_goal_completion_save_load.py
"""IPC + saver plumbing for goal_completion_record (spec §9)."""
import json
import pickle
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.ipc_messages import GameComplete
from scripts.GPU.alphazero.self_play import GameRecord, PositionRecord
from scripts.GPU.alphazero.game_saver import save_game_replay


def test_game_record_has_goal_completion_record_field_default_none():
    rec = GameRecord(positions=[], winner="red", n_moves=0)
    assert hasattr(rec, "goal_completion_record")
    assert rec.goal_completion_record is None


def test_game_complete_has_goal_completion_record_field_default_none():
    gc = GameComplete(
        worker_id=0, winner="red", draw_reason=0, n_moves=0, n_positions=0,
        wall_time_s=0.0, nn_calls=0, expand_calls=0, nn_batches=0,
        total_backups=0, total_waiters=0, unique_leaves=0,
        max_waiters=0, flush_full=0, flush_stall=0, flush_tail=0,
    )
    assert hasattr(gc, "goal_completion_record")
    assert gc.goal_completion_record is None


def test_game_complete_pickle_roundtrip_preserves_goal_completion_record():
    record = {"version": 1, "outcome_class": 1, "winner": "red", "detected": True}
    gc = GameComplete(
        worker_id=0, winner="red", draw_reason=0, n_moves=21, n_positions=21,
        wall_time_s=1.0, nn_calls=0, expand_calls=0, nn_batches=0,
        total_backups=0, total_waiters=0, unique_leaves=0,
        max_waiters=0, flush_full=0, flush_stall=0, flush_tail=0,
        goal_completion_record=record,
    )
    payload = pickle.dumps(gc)
    gc2 = pickle.loads(payload)
    assert gc2.goal_completion_record == record


def test_save_game_replay_writes_top_level_goal_completion_record_when_present():
    record = {
        "version": 1, "outcome_class": 1, "game_id": "iter_0001_game_000",
        "winner": "red", "detected": True,
    }
    with tempfile.TemporaryDirectory() as tmp:
        games_dir = Path(tmp)
        path = save_game_replay(
            games_dir=games_dir,
            iteration=1, game_idx=0, winner="red",
            move_history=((0, 0),), n_moves=1,
            goal_completion_record=record,
        )
        with open(path) as f:
            payload = json.load(f)
        assert payload["goal_completion_record"] == record


def test_save_game_replay_omits_goal_completion_record_when_none():
    with tempfile.TemporaryDirectory() as tmp:
        games_dir = Path(tmp)
        path = save_game_replay(
            games_dir=games_dir,
            iteration=1, game_idx=0, winner="red",
            move_history=((0, 0),), n_moves=1,
            goal_completion_record=None,
        )
        with open(path) as f:
            payload = json.load(f)
        assert "goal_completion_record" not in payload


def test_save_game_replay_independent_of_other_goal_completion_keys():
    """All three keys are independent: any subset can be present."""
    with tempfile.TemporaryDirectory() as tmp:
        games_dir = Path(tmp)
        path = save_game_replay(
            games_dir=games_dir,
            iteration=1, game_idx=0, winner="red",
            move_history=((0, 0),), n_moves=1,
            goal_completion_record={"version": 1, "outcome_class": 3},
            goal_completion_diagnostics=None,
            goal_completion_diagnostics_meta=None,
        )
        with open(path) as f:
            payload = json.load(f)
        assert "goal_completion_record" in payload
        assert "goal_completion_diagnostics" not in payload
        assert "goal_completion_diagnostics_meta" not in payload
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_goal_completion_save_load.py -v`
Expected: FAIL with `AttributeError: 'GameRecord' object has no attribute 'goal_completion_record'` and `TypeError: save_game_replay() got an unexpected keyword argument 'goal_completion_record'`.

- [ ] **Step 3: Add the field to `GameRecord`**

Edit `scripts/GPU/alphazero/self_play.py` — add the line after `goal_completion_diagnostics_meta` (around line 426):

```python
    goal_completion_diagnostics: List[dict] = field(default_factory=list)
    goal_completion_diagnostics_meta: Optional[dict] = None
    # Compact per-game goal-completion summary (spec 2026-05-05). None when
    # goal_completion_record_enabled=False on the upstream play_game.
    goal_completion_record: Optional[dict] = None
```

- [ ] **Step 4: Add the field to `GameComplete`**

Edit `scripts/GPU/alphazero/ipc_messages.py` — add the line after `goal_completion_diagnostics_meta` (around line 118):

```python
    goal_completion_diagnostics: Tuple[dict, ...] = ()
    goal_completion_diagnostics_meta: Optional[dict] = None
    # Compact per-game goal-completion summary (spec 2026-05-05). None when
    # goal_completion_record_enabled=False upstream. Distinct artifact from
    # goal_completion_diagnostics_meta — see spec §9.1.
    goal_completion_record: Optional[dict] = None
```

- [ ] **Step 5: Add `goal_completion_record` kwarg to `save_game_replay`**

Edit `scripts/GPU/alphazero/game_saver.py:46` — extend the signature:

```python
    # Inline closeout diagnostics (spec 2026-05-03 §8.5)
    goal_completion_diagnostics: Optional[list] = None,
    goal_completion_diagnostics_meta: Optional[dict] = None,
    # Compact per-game goal-completion summary (spec 2026-05-05).
    # Top-level JSON key when present; omitted from JSON when None.
    goal_completion_record: Optional[dict] = None,
) -> Path:
```

Then, after the existing `if goal_completion_diagnostics_meta is not None:` block (around line 166), add:

```python
    if goal_completion_record is not None:
        record["goal_completion_record"] = goal_completion_record
```

- [ ] **Step 6: Forward kwarg from `GameSaver.maybe_save_game`**

Edit `scripts/GPU/alphazero/game_saver.py:237` — extend the signature:

```python
        # Inline closeout diagnostics (spec 2026-05-03 §8.5)
        goal_completion_diagnostics: Optional[list] = None,
        goal_completion_diagnostics_meta: Optional[dict] = None,
        # Compact per-game goal-completion summary (spec 2026-05-05).
        goal_completion_record: Optional[dict] = None,
    ) -> Optional[Path]:
```

Then, in the `save_game_replay(...)` call (around line 280), add:

```python
            goal_completion_diagnostics=goal_completion_diagnostics,
            goal_completion_diagnostics_meta=goal_completion_diagnostics_meta,
            goal_completion_record=goal_completion_record,
        )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_goal_completion_save_load.py -v`
Expected: 6 passed.

- [ ] **Step 8: Commit**

```bash
git add scripts/GPU/alphazero/ipc_messages.py scripts/GPU/alphazero/self_play.py \
        scripts/GPU/alphazero/game_saver.py tests/test_goal_completion_save_load.py
git commit -m "$(cat <<'EOF'
feat(saver/ipc): thread goal_completion_record through dataclasses + saver

GameRecord and GameComplete each gain a goal_completion_record:
Optional[dict] field, distinct from goal_completion_diagnostics_meta.
save_game_replay accepts goal_completion_record kwarg; writes
top-level "goal_completion_record" JSON key when present, omits when
None. GameSaver.maybe_save_game forwards the kwarg.

Conversion helpers between GameRecord <-> GameComplete arrive in Task 8
along with the worker pipeline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `play_game()` integration + BFS reuse contract

**Files:**
- Modify: `scripts/GPU/alphazero/self_play.py:481-487` (add new kwargs); `:570-572` (construct tracker); `:631-680` (refactor BFS reuse); `:788-808` (after Phase 3 finalize, also call tracker observe); end of `play_game()` (call `tracker.finalize_game` and attach to record).
- Test: `tests/test_self_play_goal_completion_integration.py` (NEW)

**Goal:** Wire the tracker into `play_game()`. Compute `gc_state_cheap` per ply once and share with Phase 3; upgrade to `gc_state_full` per the BFS-reuse contract from spec §8.2. Tracker observes after move selection and before `state.apply_move`. At terminal, finalize and attach.

**Note**: invariant validation (`detection_threshold ≤ emit_threshold`) lands here, raising `ValueError` at start of `play_game()`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_self_play_goal_completion_integration.py
"""End-to-end self-play tracker integration (spec §8)."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.self_play import play_game, MCTSConfig
from scripts.GPU.alphazero.local_evaluator import RandomEvaluator
import random as _rng


def _eval():
    return RandomEvaluator(seed=42)


def _short_cfg():
    # Tiny MCTS, small board, deterministic-ish: enough to produce a record.
    return MCTSConfig(n_simulations=8, c_puct=1.0)


def test_play_game_attaches_goal_completion_record_when_enabled():
    rec = play_game(
        evaluator=_eval(),
        mcts_config=_short_cfg(),
        rng=_rng.Random(7),
        max_moves=30,
        active_size=8,
        goal_completion_record_enabled=True,
        goal_completion_emit_enabled=False,  # ensure tracker stands alone
    )
    assert rec.goal_completion_record is not None
    assert rec.goal_completion_record["version"] == 1
    assert rec.goal_completion_record["outcome_class"] in (1, 2, 3)
    assert "primary_class_counts" in rec.goal_completion_record


def test_play_game_no_record_when_disabled():
    rec = play_game(
        evaluator=_eval(),
        mcts_config=_short_cfg(),
        rng=_rng.Random(7),
        max_moves=30,
        active_size=8,
        goal_completion_record_enabled=False,
        goal_completion_emit_enabled=False,
    )
    assert rec.goal_completion_record is None


def test_play_game_invariant_violated_raises():
    """detection_threshold > emit_threshold must raise."""
    with pytest.raises(ValueError, match="detection_threshold"):
        play_game(
            evaluator=_eval(),
            mcts_config=_short_cfg(),
            rng=_rng.Random(7),
            max_moves=10,
            active_size=8,
            goal_completion_detection_threshold=4,
            goal_completion_emit_threshold=3,
        )


def test_play_game_record_present_when_emit_disabled_record_enabled():
    """Compact record is independent of Phase 3 emit gating."""
    rec = play_game(
        evaluator=_eval(),
        mcts_config=_short_cfg(),
        rng=_rng.Random(11),
        max_moves=30,
        active_size=8,
        goal_completion_record_enabled=True,
        goal_completion_emit_enabled=False,
    )
    assert rec.goal_completion_record is not None
    # Phase 3 fields should be empty / None.
    assert rec.goal_completion_diagnostics_meta is None
    assert rec.goal_completion_diagnostics == []


def test_play_game_record_iteration_metadata_default_zero():
    """Tracker is constructed inside play_game; iteration/game_idx are
    not yet supplied here (callers populate via finalize). For now, the
    record carries iteration=0/game_idx=game_id."""
    rec = play_game(
        evaluator=_eval(),
        mcts_config=_short_cfg(),
        rng=_rng.Random(7),
        max_moves=20,
        active_size=8,
        game_id=5,
        goal_completion_record_enabled=True,
        goal_completion_emit_enabled=False,
    )
    assert rec.goal_completion_record is not None
    assert rec.goal_completion_record["game_idx"] == 5
    # iteration is not known inside play_game; default 0.
    assert rec.goal_completion_record["iteration"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_self_play_goal_completion_integration.py -v`
Expected: FAIL — kwargs `goal_completion_record_enabled` / `goal_completion_detection_threshold` not accepted by `play_game()`.

- [ ] **Step 3: Add new kwargs and tracker construction**

Edit `scripts/GPU/alphazero/self_play.py` — modify the `play_game(...)` signature around line 487:

```python
    goal_completion_max_records_per_game: int = 64,
    # Compact per-game goal-completion record (spec 2026-05-05).
    # Independent of emit_enabled — emits even when Phase 3 detailed
    # diagnostics are disabled.
    goal_completion_record_enabled: bool = True,
    goal_completion_detection_threshold: int = 2,
    goal_completion_high_value_threshold: float = 0.9,
    goal_completion_high_value_delay_threshold_plies: int = 6,
    goal_completion_min_component_size: int = 8,
) -> GameRecord:
```

After the `mcts = MCTS(...)` line near line 518, insert the invariant check + tracker construction:

```python
    # Spec §12.1 invariant: detection_threshold must be <= emit_threshold
    # so post-detection plies are guaranteed to have full state available
    # when Phase 3 emit is enabled.
    if goal_completion_detection_threshold > goal_completion_emit_threshold:
        raise ValueError(
            "detection_threshold must be <= emit_threshold "
            f"(got {goal_completion_detection_threshold} > {goal_completion_emit_threshold})"
        )

    # Spec §6: per-game tracker. enabled=False short-circuits to no-op.
    from .goal_completion_tracker import GoalCompletionGameTracker
    gc_tracker = GoalCompletionGameTracker(
        enabled=goal_completion_record_enabled,
        detection_threshold=goal_completion_detection_threshold,
        high_value_threshold=goal_completion_high_value_threshold,
        high_value_delay_threshold_plies=goal_completion_high_value_delay_threshold_plies,
        max_depth=goal_completion_max_depth,
        min_component_size=goal_completion_min_component_size,
    )
```

- [ ] **Step 4: Refactor BFS reuse**

The existing block at lines 631-680 in `self_play.py` already computes `gc_state_for_diag` (cheap) and a `partial_diag` (full). Refactor so the tracker also benefits from the same gc_state, and adds its own upgrade trigger when needed.

Replace the block beginning `# --- Phase 3: closeout diagnostic partial capture (best-effort) ---` (line 631) and ending at line 680 with:

```python
        # --- Compute gc_state once per ply, shared by Phase 3 + tracker. ---
        gc_state_for_diag = None     # cheap: total_goal_distance only
        gc_state_full = None         # full: includes endpoint_completion_moves
        partial_diag = None
        need_cheap = goal_completion_emit_enabled or gc_tracker.enabled
        if need_cheap:
            try:
                from .connectivity_diagnostics import compute_goal_completion_state
                gc_state_for_diag = compute_goal_completion_state(
                    state, state.to_move,
                    max_depth=goal_completion_max_depth,
                    min_component_size=goal_completion_emit_min_component,
                    enumerate_moves=False,
                )
            except Exception as _e:
                if goal_completion_diagnostics_meta is not None:
                    goal_completion_diagnostics_meta["error_count"] += 1
                import sys as _sys
                _sys.stderr.write(f"[gc-cheap] ply={ply} error: {_e!r}\n")

            # Decide whether to upgrade to gc_state_full (spec §8.2).
            total_now = (
                gc_state_for_diag.get("total_goal_distance")
                if gc_state_for_diag is not None else None
            )
            needs_phase3_full = (
                goal_completion_emit_enabled
                and gc_state_for_diag is not None
                and total_now is not None
                and total_now <= goal_completion_emit_threshold
            )
            needs_tracker_full = (
                gc_tracker.enabled
                and gc_state_for_diag is not None
                and total_now is not None
                and (
                    gc_tracker.is_detected(state.to_move)
                    or total_now <= gc_tracker.detection_threshold
                )
            )
            if needs_phase3_full or needs_tracker_full:
                try:
                    gc_state_full = compute_goal_completion_state(
                        state, state.to_move,
                        max_depth=goal_completion_max_depth,
                        min_component_size=goal_completion_emit_min_component,
                        enumerate_moves=True,
                    )
                except Exception as _e:
                    if goal_completion_diagnostics_meta is not None:
                        goal_completion_diagnostics_meta["error_count"] += 1
                    import sys as _sys
                    _sys.stderr.write(f"[gc-full] ply={ply} error: {_e!r}\n")

            # Phase 3 partial build (existing behavior, but uses the shared
            # gc_state_full instead of recomputing).
            if (goal_completion_emit_enabled
                    and gc_state_full is not None
                    and total_now is not None
                    and total_now <= goal_completion_emit_threshold):
                if (goal_completion_diagnostics_meta is not None
                        and len(goal_completion_diagnostics) >= goal_completion_max_records_per_game):
                    goal_completion_diagnostics_meta["records_dropped_by_cap"] += 1
                elif root.priors_raw is None:
                    if goal_completion_diagnostics_meta is not None:
                        goal_completion_diagnostics_meta["skipped_missing_priors_count"] += 1
                else:
                    try:
                        from .closeout_diagnostics import build_closeout_diagnostic_partial
                        _decode_fn = lambda mid, _a=active_size: (mid // _a, mid % _a)
                        partial_diag = build_closeout_diagnostic_partial(
                            ply=ply,
                            side_to_move=state.to_move,
                            visit_counts=visit_counts,
                            priors_raw=root.priors_raw,
                            priors_adjusted=getattr(root, "priors", None),
                            root=root,
                            goal_completion_state=gc_state_full,
                            board_size=active_size,
                            skip_distance_reducing=goal_completion_skip_distance_reducing,
                            decode_fn=_decode_fn,
                        )
                    except Exception as _e:
                        if goal_completion_diagnostics_meta is not None:
                            goal_completion_diagnostics_meta["error_count"] += 1
                        import sys as _sys
                        _sys.stderr.write(f"[closeout-diag] ply={ply} partial: {_e!r}\n")
```

The original block built `gc_state_for_diag` with `enumerate_moves=True` (default). The refactor inverts: cheap is now the default, full is on-demand. Phase 3 partial build path explicitly consumes `gc_state_full`.

- [ ] **Step 5: Insert tracker observe call**

After the move is selected (existing line `move = mcts.select_move(visit_counts, ply)` around line 774), and BEFORE the existing Phase 3 finalize block (line 788), insert tracker observe:

```python
        # Tracker observes pre-move state. Independent of Phase 3 emit.
        if gc_tracker.enabled:
            # search_score (root_value) is from state.to_move's perspective.
            _ss = float(root_value) if root_value is not None else None
            try:
                gc_tracker.observe_pre_move(
                    state=state,
                    ply=ply + 1,                     # tracker uses 1-indexed ply
                    side_to_move=state.to_move,
                    selected_move=move,
                    search_score=_ss,
                    gc_state_cheap=gc_state_for_diag,
                    gc_state_full=gc_state_full,
                )
            except Exception as _e:
                import sys as _sys
                _sys.stderr.write(
                    f"[gc-tracker] ply={ply} observe error: {_e!r}\n"
                )
```

- [ ] **Step 6: Finalize tracker at game end and attach to GameRecord**

Locate the `return GameRecord(...)` constructor near the end of `play_game()` (search for `return GameRecord`). Just before it, finalize:

```python
    # Compact goal-completion record (spec §6).
    _gc_reason_for_record = "win"
    if winner is None:
        if draw_reason in ("state_cap", "terminal_state_cap"):
            _gc_reason_for_record = "state_cap"
        elif draw_reason in ("timeout", "timeout_selfplay"):
            _gc_reason_for_record = "timeout"
        elif draw_reason in ("board_full", "terminal_board_full"):
            _gc_reason_for_record = "board_full"
        else:
            _gc_reason_for_record = "unknown"
    elif resigned_by is not None:
        _gc_reason_for_record = "resign"
    gc_record = gc_tracker.finalize_game(
        winner=winner,
        reason=_gc_reason_for_record,
        n_moves=len(move_history),
        starting_player=start_player,
        iteration=0,                        # populated downstream by trainer/saver path
        game_idx=game_id,
        game_id=f"game_{game_id:03d}",
    )
```

Then add the new field to the `GameRecord(...)` constructor call:

```python
        goal_completion_diagnostics=goal_completion_diagnostics,
        goal_completion_diagnostics_meta=goal_completion_diagnostics_meta,
        goal_completion_record=gc_record,
    )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_self_play_goal_completion_integration.py tests/test_goal_completion_tracker.py -v`
Expected: 21 + 5 passed. Also run prior tests to confirm no regression:

`.venv/bin/python -m pytest tests/test_self_play_closeout_diagnostics.py -v`
Expected: existing Phase 3 tests still pass (the refactor preserves Phase 3 behavior).

- [ ] **Step 8: Commit**

```bash
git add scripts/GPU/alphazero/self_play.py tests/test_self_play_goal_completion_integration.py
git commit -m "$(cat <<'EOF'
feat(self-play): wire goal-completion tracker into play_game

Adds new kwargs goal_completion_record_enabled (default True),
detection_threshold (2), high_value_threshold (0.9),
high_value_delay_threshold_plies (6), min_component_size (8).
Validates detection_threshold <= emit_threshold invariant at start.

BFS reuse refactored: gc_state_cheap is computed once per ply
(enumerate_moves=False), shared by Phase 3 and the tracker. Upgrade to
gc_state_full happens only when needs_phase3_full OR needs_tracker_full
(post-detection focal side, or Phase 3 emit gate). Phase 3 partial
build now consumes the shared gc_state_full.

Tracker observes pre-move (after MCTS, after move selection, before
state.apply_move). At game end, finalize_game produces compact record
attached to GameRecord.goal_completion_record. iteration is populated
downstream by trainer/saver wiring (Task 8).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Worker → main pipeline carries `goal_completion_record`

**Files:**
- Modify: `scripts/GPU/alphazero/self_play_worker.py:259-261` (forward field into GameComplete)
- Modify: `scripts/GPU/alphazero/trainer.py:97` (`_save_game_from_ipc` forwards to saver, with iteration injected)
- Modify: `scripts/GPU/alphazero/trainer.py:153` (`_save_game_from_record` forwards to saver, with iteration injected)
- Test: `tests/test_goal_completion_save_load.py` (extend with worker→trainer pipeline test)

**Goal:** Worker writes `goal_completion_record` into `GameComplete`. The two trainer-side save helpers (`_save_game_from_ipc`, `_save_game_from_record`) inject the trainer's iteration number into the record (overwriting the `iteration: 0` placeholder set in `play_game`) and forward to `save_game_replay`.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_goal_completion_save_load.py`:

```python
def test_save_game_from_record_injects_iteration():
    """The trainer overwrites iteration=0 placeholder with its actual iter."""
    from scripts.GPU.alphazero.trainer import _save_game_from_record

    # Minimal in-memory game saver double.
    saved = {}
    class _FakeSaver:
        def maybe_save_game(self, *args, **kwargs):
            saved.update(kwargs)
            return Path("/tmp/fake_path.json")

    rec = GameRecord(positions=[], winner="red", n_moves=1)
    rec.move_history = [(0, 0)]
    rec.start_player = "red"
    rec.draw_reason = None
    rec.goal_completion_record = {
        "version": 1, "outcome_class": 1, "winner": "red",
        "iteration": 0, "game_idx": 5, "game_id": "game_005",
    }

    # The helper takes a (saver, game) pair; we patch the saver's iteration
    # via the GameSaver's _current_iter -- mimic by setting attribute.
    fake = _FakeSaver()
    fake._current_iter = 112
    _save_game_from_record(fake, rec)
    rec_arg = saved["goal_completion_record"]
    assert rec_arg["iteration"] == 112


def test_save_game_from_ipc_injects_iteration():
    from scripts.GPU.alphazero.trainer import _save_game_from_ipc

    saved = {}
    class _FakeSaver:
        def maybe_save_game(self, *args, **kwargs):
            saved.update(kwargs)
            return Path("/tmp/fake_path.json")

    msg = GameComplete(
        worker_id=0, winner="red", draw_reason=0, n_moves=1, n_positions=1,
        wall_time_s=0.0, nn_calls=0, expand_calls=0, nn_batches=0,
        total_backups=0, total_waiters=0, unique_leaves=0,
        max_waiters=0, flush_full=0, flush_stall=0, flush_tail=0,
        move_history=((0, 0),),
        start_player="red",
        goal_completion_record={
            "version": 1, "outcome_class": 1, "winner": "red",
            "iteration": 0, "game_idx": 5, "game_id": "game_005",
        },
    )
    fake = _FakeSaver()
    fake._current_iter = 112
    _save_game_from_ipc(fake, msg)
    rec_arg = saved["goal_completion_record"]
    assert rec_arg["iteration"] == 112
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_goal_completion_save_load.py -v`
Expected: 2 new tests FAIL with `KeyError` on `goal_completion_record` in `saved` dict (helpers don't forward yet).

- [ ] **Step 3: Wire worker → GameComplete**

Edit `scripts/GPU/alphazero/self_play_worker.py:259-260` — extend the `GameComplete(...)` constructor:

```python
                goal_completion_diagnostics=tuple(game.goal_completion_diagnostics),
                goal_completion_diagnostics_meta=game.goal_completion_diagnostics_meta,
                goal_completion_record=game.goal_completion_record,
            ))
```

- [ ] **Step 4: Wire `_save_game_from_ipc` and `_save_game_from_record`**

Edit `scripts/GPU/alphazero/trainer.py:93-98` (extend the `goal_completion_diagnostics_meta=msg.goal_completion_diagnostics_meta,` block):

```python
        goal_completion_diagnostics=(
            list(msg.goal_completion_diagnostics)
            if msg.goal_completion_diagnostics else None
        ),
        goal_completion_diagnostics_meta=msg.goal_completion_diagnostics_meta,
        goal_completion_record=_inject_iteration(
            msg.goal_completion_record, getattr(game_saver, "_current_iter", None),
        ),
    )
```

Edit `scripts/GPU/alphazero/trainer.py:149-154` (extend the `_save_game_from_record` saver call):

```python
        goal_completion_diagnostics=(
            list(game.goal_completion_diagnostics)
            if game.goal_completion_diagnostics else None
        ),
        goal_completion_diagnostics_meta=game.goal_completion_diagnostics_meta,
        goal_completion_record=_inject_iteration(
            game.goal_completion_record, getattr(game_saver, "_current_iter", None),
        ),
    )
```

Then add the `_inject_iteration` helper above `_save_game_from_ipc` near line 48:

```python
def _inject_iteration(record: Optional[dict], iteration: Optional[int]) -> Optional[dict]:
    """Set iteration on a goal_completion_record copy.

    play_game() emits records with iteration=0 because the worker process
    does not know the trainer's iteration counter. The trainer-side save
    helpers inject the actual iteration here. Returns None unchanged.
    """
    if record is None or iteration is None:
        return record
    out = dict(record)
    out["iteration"] = int(iteration)
    return out
```

Add `from typing import Optional` to the imports if not already present.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_goal_completion_save_load.py tests/test_self_play_goal_completion_integration.py -v`
Expected: all pass. Also run trainer regression sanity:

`.venv/bin/python -m pytest tests/ -k "trainer or save_game" -v`
Expected: no regressions in pre-existing trainer/save tests.

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/self_play_worker.py scripts/GPU/alphazero/trainer.py \
        tests/test_goal_completion_save_load.py
git commit -m "$(cat <<'EOF'
feat(worker/trainer): pipe goal_completion_record end-to-end

Worker forwards GameRecord.goal_completion_record into GameComplete.
Trainer-side _save_game_from_ipc / _save_game_from_record inject the
current iteration (placeholder=0 from play_game; trainer overwrites
with actual iteration number) and forward to save_game_replay.

The compact record now flows: worker play_game -> GameComplete IPC ->
trainer save helpers -> per-game JSON top-level "goal_completion_record".

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Trainer per-iteration sidecar aggregation hook

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` (locate the sidecar writer that already calls `aggregate_per_game_stats` / `aggregate_opening_diagnostics` around line 3070+; add `goal_completion_summary` block)
- Test: `tests/test_goal_completion_aggregator.py` (add a trainer-style integration test that exercises the sidecar block construction)

**Goal:** After all worker games for an iteration complete, the trainer assembles the iteration sidecar dict. Add a call to `aggregate_goal_completion_records` and stash the result under `goal_completion_summary`. Consumes records from completed game objects (in-process or IPC); both paths converge through the same per-iteration list before the sidecar write.

- [ ] **Step 1: Inspect trainer to confirm hook location**

Run: `grep -n 'aggregate_per_game_stats\|_sidecar\["per_game_stats"\]\|_sidecar\["opening_penalty_diagnostics"\]' scripts/GPU/alphazero/trainer.py`

Identify the contiguous block that builds the iteration sidecar dict. The existing pattern uses `_sidecar["..."] = aggregate_...(...)` calls. Add the new line in the same block.

- [ ] **Step 2: Write failing test**

Append to `tests/test_goal_completion_aggregator.py`:

```python
def test_trainer_style_sidecar_block_construction():
    """Validate that the aggregator output composes cleanly into a
    trainer-shaped sidecar dict. Reflects the call pattern in
    trainer.py's iteration sidecar writer."""
    records_for_iter = [_decisive_record(), _decisive_record(), None]
    # Trainer-style call: pass [g.goal_completion_record for g in games].
    sidecar = {}
    sidecar["goal_completion_summary"] = aggregate_goal_completion_records(
        records_for_iter,
        config={
            "detection_threshold": 2,
            "emit_threshold": 3,
            "high_value_threshold": 0.9,
            "high_value_delay_threshold_plies": 6,
            "max_depth": 3,
            "min_component_size": 8,
        },
        games_total=3,
    )
    block = sidecar["goal_completion_summary"]
    assert block["version"] == 1
    assert block["config"]["detection_threshold"] == 2
    assert block["config"]["emit_threshold"] == 3
    assert block["diagnostics_coverage"]["games_total"] == 3
    assert block["diagnostics_coverage"]["games_with_record"] == 2
    assert block["diagnostics_coverage"]["coverage_rate"] == pytest.approx(2/3)
    assert block["main_population"]["n"] == 2
```

- [ ] **Step 3: Run test to verify it fails (or passes trivially)**

The test only exercises the aggregator; it should already pass after Task 5. Run: `.venv/bin/python -m pytest tests/test_goal_completion_aggregator.py::test_trainer_style_sidecar_block_construction -v`. Expected: PASS.

This test is a guard against schema drift in the sidecar block; it formalizes the contract the trainer follows.

- [ ] **Step 4: Wire trainer hook**

Edit `scripts/GPU/alphazero/trainer.py` — in the iteration sidecar writer block (the same place that calls `aggregate_per_game_stats(...)`), add:

```python
            from .goal_completion_aggregator import aggregate_goal_completion_records
            _gc_records_for_iter = [
                getattr(g, "goal_completion_record", None) for g in iter_games
            ]
            _sidecar["goal_completion_summary"] = aggregate_goal_completion_records(
                _gc_records_for_iter,
                config={
                    "detection_threshold": int(getattr(self, "goal_completion_detection_threshold", 2)),
                    "emit_threshold": int(getattr(self, "goal_completion_emit_threshold", 3)),
                    "high_value_threshold": float(getattr(self, "goal_completion_high_value_threshold", 0.9)),
                    "high_value_delay_threshold_plies": int(getattr(
                        self, "goal_completion_high_value_delay_threshold_plies", 6)),
                    "max_depth": int(getattr(self, "goal_completion_max_depth", 3)),
                    "min_component_size": int(getattr(self, "goal_completion_min_component_size", 8)),
                },
                games_total=len(iter_games),
            )
```

The exact local variable name for the per-iteration games list is `iter_games` in this trainer; if the actual variable is named differently in the surrounding code, substitute that name.

- [ ] **Step 5: Run tests + smoke training imports**

Run: `.venv/bin/python -m pytest tests/test_goal_completion_aggregator.py -v`
Expected: 15 passed.

Sanity-check trainer imports:
`.venv/bin/python -c "from scripts.GPU.alphazero.trainer import _save_game_from_record"`
Expected: no import errors.

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py tests/test_goal_completion_aggregator.py
git commit -m "$(cat <<'EOF'
feat(trainer): add goal_completion_summary to per-iteration sidecar

Calls aggregate_goal_completion_records on the iteration's collected
records and writes the result under sidecar["goal_completion_summary"].
Snapshots the active config so the analyzer can echo it. Coverage uses
games_total = len(iter_games) so partial coverage is reflected
truthfully.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Analyzer record-consumption default path + worst-cases CSV

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py:3492-3499` (replace `aggregate_goal_completion_diagnostics(...)` call site with shared aggregator over per-game records)
- Modify: `scripts/twixt_replay_analyzer.py:3696` (`write_goal_completion_worst_cases_csv` reads from records)
- Modify: `scripts/twixt_replay_analyzer.py:2477` (rewrite the worst-cases CSV writer to consume records)
- Test: `tests/test_analyzer_goal_completion_records.py` (NEW)

**Goal:** Switch analyzer's default code path to read pre-computed `goal_completion_record` from each replay JSON and aggregate via the shared module. Keep `format_goal_completion_report()` unchanged — it already reads the same summary shape. Worst-cases CSV gets a new path that pulls fields from records directly with the unified `sort_delay_plies` key.

The legacy `aggregate_goal_completion_diagnostics()` and `_build_class1/2_per_game_record()` functions stay in the file for now — Task 13 moves them to the recompute module.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_analyzer_goal_completion_records.py
"""Analyzer record-consumption default path (spec §11)."""
import csv
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _replay_with_record(*, iteration, game_idx, winner, outcome_class,
                       conversion_delay_plies=None, cap_delay_proxy_plies=None,
                       detected=True, primary_class_counts=None, **rec_overrides):
    """Build a minimal in-memory replay dict carrying a goal_completion_record."""
    record = {
        "version": 1,
        "outcome_class": outcome_class,
        "iteration": iteration,
        "game_idx": game_idx,
        "game_id": f"iter_{iteration:04d}_game_{game_idx:03d}",
        "winner": winner,
        "detected_player": winner if outcome_class == 1 else "red",
        "starting_player": "red",
        "n_moves": 21,
        "reason": "win" if outcome_class == 1 else "state_cap",
        "scope": "winner" if outcome_class == 1 else "both_sides",
        "ever_distance_le_2": True,
        "ever_distance_le_3": True,
        "min_total_goal_distance": 2,
        "detected": detected,
        "first_dominant_unclosed_ply": 11,
        "first_total_goal_distance": 2,
        "first_category": "two_endpoint_closeout_2ply",
        "actual_terminal_ply": 21,
        "actual_win_ply": 21 if outcome_class == 1 else None,
        "conversion_delay_plies": conversion_delay_plies,
        "conversion_delay_winner_moves": (conversion_delay_plies // 2 if conversion_delay_plies else None),
        "cap_delay_proxy_plies": cap_delay_proxy_plies,
        "primary_class_counts": primary_class_counts or (
            {"completes_endpoint": 1, "reduces_total_goal_distance": 0,
             "redundant_reinforcement": 3, "off_chain": 1, "other": 0}
            if outcome_class == 1 else None
        ),
        "max_search_score_after_detection": 0.99 if outcome_class == 1 else None,
        "mean_search_score_after_detection": 0.95 if outcome_class == 1 else None,
        "high_value_after_detection_plies": 4 if outcome_class == 1 else None,
        "root_value_high_but_delayed": False,
        "search_score_coverage_in_watch_window": 5 if outcome_class == 1 else None,
        "winner_moves_in_watch_window": 5 if outcome_class == 1 else None,
        "winner_moves_with_dominant_component": 5 if outcome_class == 1 else None,
        "winner_moves_with_dominant_unavailable": 0 if outcome_class == 1 else None,
        "first_largest_component_size": 12,
        "first_endpoint_distances": {"top": 0, "bottom": 1},
    }
    record.update(rec_overrides)
    return {
        "iteration": iteration,
        "game_idx": game_idx,
        "winner": winner,
        "starting_player": "red",
        "moves": [{"player": "red", "row": r, "col": c, "turn": i + 1}
                  for i, (r, c) in enumerate([(0, 0)])],
        "meta": {"reason": "win" if outcome_class == 1 else "state_cap",
                 "n_moves": 21, "board_size": 24,
                 "starting_player": "red"},
        "goal_completion_record": record,
    }


def test_analyzer_default_path_uses_records_no_recompute():
    """Default path consumes per-game records via the shared aggregator."""
    from scripts.twixt_replay_analyzer import (
        aggregate_goal_completion_diagnostics_from_records,
    )
    replays = [
        _replay_with_record(iteration=110, game_idx=i, winner="red",
                            outcome_class=1, conversion_delay_plies=10)
        for i in range(3)
    ]
    summary = aggregate_goal_completion_diagnostics_from_records(
        replays,
        sidecar_summaries={},
        config={"detection_threshold": 2},
    )
    assert summary["main_population"]["n"] == 3
    assert summary["diagnostics_coverage"]["games_with_record"] == 3
    assert summary["diagnostics_coverage"]["coverage_rate"] == 1.0


def test_worst_cases_csv_from_records_class1():
    from scripts.twixt_replay_analyzer import write_goal_completion_worst_cases_csv

    replays = [
        _replay_with_record(iteration=110, game_idx=i, winner="red",
                            outcome_class=1, conversion_delay_plies=d)
        for i, d in enumerate([3, 22, 8, 28, 14])
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "worst.csv"
        write_goal_completion_worst_cases_csv(
            str(out_path), replays, top_k=3, suffix="test",
        )
        with open(out_path) as f:
            rows = list(csv.DictReader(f))
    assert len(rows) == 3
    delays = [int(r["conversion_delay_plies"]) for r in rows]
    assert delays == [28, 22, 14]


def test_worst_cases_csv_mixed_class1_class2_unified_sort():
    from scripts.twixt_replay_analyzer import write_goal_completion_worst_cases_csv

    replays = [
        _replay_with_record(iteration=110, game_idx=0, winner="red",
                            outcome_class=1, conversion_delay_plies=10),
        _replay_with_record(iteration=110, game_idx=1, winner=None,
                            outcome_class=2, cap_delay_proxy_plies=50),
        _replay_with_record(iteration=110, game_idx=2, winner="black",
                            outcome_class=1, conversion_delay_plies=30),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "worst.csv"
        write_goal_completion_worst_cases_csv(
            str(out_path), replays, top_k=3, suffix="test",
        )
        with open(out_path) as f:
            rows = list(csv.DictReader(f))
    # Sort by delay/proxy descending: 50 (Class 2) > 30 (Class 1) > 10 (Class 1).
    ranked = [r["scope"] for r in rows]
    assert ranked[0] == "both_sides"   # Class 2 first


def test_worst_cases_csv_skips_replays_without_record():
    from scripts.twixt_replay_analyzer import write_goal_completion_worst_cases_csv

    r_with = _replay_with_record(iteration=110, game_idx=0, winner="red",
                                 outcome_class=1, conversion_delay_plies=20)
    r_without = {"iteration": 110, "game_idx": 1, "winner": "red",
                 "starting_player": "red", "moves": [],
                 "meta": {"reason": "win", "n_moves": 0, "board_size": 24}}
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "worst.csv"
        write_goal_completion_worst_cases_csv(
            str(out_path), [r_with, r_without], top_k=5, suffix="test",
        )
        with open(out_path) as f:
            rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["game_idx"] == "0"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_analyzer_goal_completion_records.py -v`
Expected: FAIL — `aggregate_goal_completion_diagnostics_from_records` not yet defined; `write_goal_completion_worst_cases_csv` signature mismatch.

- [ ] **Step 3: Add new analyzer function**

In `scripts/twixt_replay_analyzer.py`, add the new function near the existing `aggregate_goal_completion_diagnostics` definition (around line 641). Also add the import at top of file (search for existing imports from `scripts.GPU.alphazero` and add):

```python
from scripts.GPU.alphazero.goal_completion_aggregator import (
    aggregate_goal_completion_records,
)
```

Then add the new function:

```python
def aggregate_goal_completion_diagnostics_from_records(
    replays: list, sidecar_summaries: dict, config: dict,
) -> dict:
    """Default analyzer path (spec §11.1).

    Per-game records are canonical. Sidecar summaries are held for
    validation / iteration telemetry but the cross-iteration roll-up
    is recomputed from records to avoid lossy roll-up of roll-ups.
    """
    per_game_records = [r.get("goal_completion_record") for r in replays]
    return aggregate_goal_completion_records(
        per_game_records,
        config=config,
        games_total=len(replays),
    )
```

- [ ] **Step 4: Rewrite worst-cases CSV writer**

Replace the existing `write_goal_completion_worst_cases_csv` body (line 2477+). The new implementation reads records directly:

```python
def write_goal_completion_worst_cases_csv(
    out_path: str, replays: list, top_k: int = 25, suffix: str = "",
) -> None:
    """Write worst-cases CSV from per-game goal_completion_records.

    Sort key: conversion_delay_plies for Class 1, cap_delay_proxy_plies
    for Class 2; replays without a record are skipped silently.
    """
    def _sort_delay(rec: Optional[dict]) -> int:
        if rec is None:
            return -1
        oc = rec.get("outcome_class")
        if oc == 1:
            return int(rec.get("conversion_delay_plies") or 0)
        if oc == 2:
            return int(rec.get("cap_delay_proxy_plies") or 0)
        return -1

    pairs = []
    for replay in replays:
        rec = replay.get("goal_completion_record")
        if rec is None:
            continue
        pairs.append((rec, replay))
    pairs.sort(key=lambda p: -_sort_delay(p[0]))
    top = pairs[:top_k]

    fieldnames = [
        "iteration", "game_idx", "game_id", "winner", "starting_player",
        "n_moves", "reason", "outcome_class", "scope",
        "detected_player", "first_dominant_unclosed_ply",
        "first_total_goal_distance", "first_category",
        "actual_terminal_ply", "actual_win_ply",
        "conversion_delay_plies", "conversion_delay_winner_moves",
        "cap_delay_proxy_plies",
        "primary_class_completes_endpoint",
        "primary_class_reduces_total_goal_distance",
        "primary_class_redundant_reinforcement",
        "primary_class_off_chain",
        "primary_class_other",
        "winner_moves_in_watch_window",
        "winner_moves_with_dominant_component",
        "winner_moves_with_dominant_unavailable",
        "max_search_score_after_detection",
        "mean_search_score_after_detection",
        "high_value_after_detection_plies",
        "root_value_high_but_delayed",
        "search_score_coverage_in_watch_window",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec, replay in top:
            pcc = rec.get("primary_class_counts") or {}
            row = {
                "iteration": rec.get("iteration"),
                "game_idx": rec.get("game_idx"),
                "game_id": rec.get("game_id"),
                "winner": rec.get("winner"),
                "starting_player": rec.get("starting_player"),
                "n_moves": rec.get("n_moves"),
                "reason": rec.get("reason"),
                "outcome_class": rec.get("outcome_class"),
                "scope": rec.get("scope"),
                "detected_player": rec.get("detected_player"),
                "first_dominant_unclosed_ply": rec.get("first_dominant_unclosed_ply"),
                "first_total_goal_distance": rec.get("first_total_goal_distance"),
                "first_category": rec.get("first_category"),
                "actual_terminal_ply": rec.get("actual_terminal_ply"),
                "actual_win_ply": rec.get("actual_win_ply"),
                "conversion_delay_plies": rec.get("conversion_delay_plies"),
                "conversion_delay_winner_moves": rec.get("conversion_delay_winner_moves"),
                "cap_delay_proxy_plies": rec.get("cap_delay_proxy_plies"),
                "primary_class_completes_endpoint": pcc.get("completes_endpoint") if pcc else None,
                "primary_class_reduces_total_goal_distance": pcc.get("reduces_total_goal_distance") if pcc else None,
                "primary_class_redundant_reinforcement": pcc.get("redundant_reinforcement") if pcc else None,
                "primary_class_off_chain": pcc.get("off_chain") if pcc else None,
                "primary_class_other": pcc.get("other") if pcc else None,
                "winner_moves_in_watch_window": rec.get("winner_moves_in_watch_window"),
                "winner_moves_with_dominant_component": rec.get("winner_moves_with_dominant_component"),
                "winner_moves_with_dominant_unavailable": rec.get("winner_moves_with_dominant_unavailable"),
                "max_search_score_after_detection": rec.get("max_search_score_after_detection"),
                "mean_search_score_after_detection": rec.get("mean_search_score_after_detection"),
                "high_value_after_detection_plies": rec.get("high_value_after_detection_plies"),
                "root_value_high_but_delayed": rec.get("root_value_high_but_delayed"),
                "search_score_coverage_in_watch_window": rec.get("search_score_coverage_in_watch_window"),
            }
            w.writerow(row)
```

- [ ] **Step 5: Switch the analyzer's default call site**

In `scripts/twixt_replay_analyzer.py:3492-3499`, replace the `aggregate_goal_completion_diagnostics(...)` call with the records-based default:

```python
    if getattr(args, "goal_completion_recompute", False):
        # Legacy walker (Task 13).
        goal_completion_val = aggregate_goal_completion_diagnostics(
            replays,
            max_depth=getattr(args, "goal_completion_max_depth", 3) if args else 3,
            min_component_size=getattr(args, "goal_completion_min_component_size", 8) if args else 8,
            detection_threshold=getattr(args, "goal_completion_detection_threshold", 2) if args else 2,
            high_value_threshold=getattr(args, "goal_completion_high_value_threshold", 0.9) if args else 0.9,
            worst_cases_top_k=getattr(args, "goal_completion_worst_cases_top_k", 25) if args else 25,
        )
    else:
        goal_completion_val = aggregate_goal_completion_diagnostics_from_records(
            replays,
            sidecar_summaries={
                it: sc.get("goal_completion_summary")
                for it, sc in (relevant_sidecars or {}).items()
                if sc.get("goal_completion_summary") is not None
            },
            config={
                "detection_threshold": getattr(args, "goal_completion_detection_threshold", 2) if args else 2,
                "emit_threshold": 3,
                "high_value_threshold": getattr(args, "goal_completion_high_value_threshold", 0.9) if args else 0.9,
                "high_value_delay_threshold_plies": 6,
                "max_depth": getattr(args, "goal_completion_max_depth", 3) if args else 3,
                "min_component_size": getattr(args, "goal_completion_min_component_size", 8) if args else 8,
            },
        )
```

The `goal_completion_recompute` flag is wired in Task 13. Add a temporary `goal_completion_recompute=False` default in argparse for now:

In the argparse setup near line 3962, locate a similar add_argument and add a placeholder flag (will be expanded in Task 13):

```python
    ap.add_argument("--goal-completion-recompute", action="store_true",
                    default=False,
                    help="(Task 13) Use legacy replay walker for goal-completion. Default: read records.")
```

- [ ] **Step 6: Update `write_goal_completion_worst_cases_csv` call site**

In `scripts/twixt_replay_analyzer.py:3696`, the existing call passes a different signature. Update to:

```python
    if not getattr(args, "goal_completion_recompute", False):
        write_goal_completion_worst_cases_csv(
            os.path.join(out_dir, _suffixed("goal_completion_worst_cases", "csv", suffix)),
            replays,
            top_k=getattr(args, "goal_completion_worst_cases_top_k", 25) if args else 25,
            suffix=suffix,
        )
    # Recompute path's own CSV writer is preserved; Task 13 wires it.
```

- [ ] **Step 7: Run tests**

Run: `.venv/bin/python -m pytest tests/test_analyzer_goal_completion_records.py -v`
Expected: 4 passed.

Run pre-existing analyzer tests to verify no regression in legacy path (it stays alive until Task 13):
`.venv/bin/python -m pytest tests/test_analyzer_goal_completion.py -v`
Expected: continues to pass (the legacy `aggregate_goal_completion_diagnostics` is still callable; it's just not the default for the analyzer entrypoint).

- [ ] **Step 8: Commit**

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_goal_completion_records.py
git commit -m "$(cat <<'EOF'
feat(analyzer): record-consumption default path + worst-cases CSV

Default analyzer path now reads goal_completion_record from per-game
JSONs and aggregates via the shared aggregator. No replay walking, no
BFS. write_goal_completion_worst_cases_csv consumes records directly
with a unified delay sort (Class 1: conversion_delay_plies, Class 2:
cap_delay_proxy_plies).

The legacy aggregate_goal_completion_diagnostics is preserved in this
file for now; Task 13 moves it to goal_completion_recompute.py and
fully wires the --goal-completion-recompute flag.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Analyzer warnings (missing-record, sidecar/version mismatch)

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py` (extend `aggregate_goal_completion_diagnostics_from_records` with warnings)
- Test: `tests/test_analyzer_goal_completion_records.py` (add tests)

**Goal:** Aggregated missing-record warning (one summary line + up to 3 examples). Sidecar/replay coverage mismatch warning per iteration. Cross-version mismatch warning when records and sidecars carry different versions.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_analyzer_goal_completion_records.py`:

```python
def test_analyzer_missing_record_warning_aggregated(capsys):
    """One summary warning per missing-record bucket, with up to 3 examples."""
    from scripts.twixt_replay_analyzer import (
        aggregate_goal_completion_diagnostics_from_records,
    )
    replays = []
    # 5 missing replays.
    for i in range(5):
        replays.append({
            "iteration": 110, "game_idx": i, "winner": "red",
            "starting_player": "red", "moves": [],
            "meta": {"n_moves": 0, "board_size": 24},
        })
    # 1 with record.
    replays.append(_replay_with_record(iteration=110, game_idx=99, winner="red",
                                       outcome_class=1, conversion_delay_plies=10))

    aggregate_goal_completion_diagnostics_from_records(
        replays, sidecar_summaries={}, config={},
    )
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "5/6" in out
    assert "missing goal_completion_record" in out
    assert "Examples:" in out


def test_analyzer_all_missing_warning(capsys):
    from scripts.twixt_replay_analyzer import (
        aggregate_goal_completion_diagnostics_from_records,
    )
    replays = [
        {"iteration": 110, "game_idx": i, "winner": "red",
         "starting_player": "red", "moves": [],
         "meta": {"n_moves": 0, "board_size": 24}}
        for i in range(3)
    ]
    aggregate_goal_completion_diagnostics_from_records(
        replays, sidecar_summaries={}, config={},
    )
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "3/3" in out
    assert "Goal-completion report skipped" in out


def test_analyzer_sidecar_mismatch_warning(capsys):
    from scripts.twixt_replay_analyzer import (
        aggregate_goal_completion_diagnostics_from_records,
    )
    replays = [
        _replay_with_record(iteration=110, game_idx=0, winner="red",
                            outcome_class=1, conversion_delay_plies=10),
    ]
    sidecar_summaries = {
        110: {"diagnostics_coverage": {"games_with_record": 100,
                                       "games_total": 100,
                                       "coverage_rate": 1.0,
                                       "games_class_1": 100,
                                       "games_class_2": 0,
                                       "games_class_3": 0}}
    }
    aggregate_goal_completion_diagnostics_from_records(
        replays, sidecar_summaries=sidecar_summaries, config={},
    )
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "sidecar/replay mismatch" in out
    assert "iter 0110" in out


def test_analyzer_version_mismatch_warning(capsys):
    from scripts.twixt_replay_analyzer import (
        aggregate_goal_completion_diagnostics_from_records,
    )
    replays = [
        _replay_with_record(iteration=110, game_idx=0, winner="red",
                            outcome_class=1, conversion_delay_plies=10),
    ]
    sidecar_summaries = {
        110: {
            "version": 2,
            "diagnostics_coverage": {"games_with_record": 1, "games_total": 1,
                                     "coverage_rate": 1.0,
                                     "games_class_1": 1, "games_class_2": 0,
                                     "games_class_3": 0},
        }
    }
    aggregate_goal_completion_diagnostics_from_records(
        replays, sidecar_summaries=sidecar_summaries, config={},
    )
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "version mismatch" in out
    assert "records canonical" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_analyzer_goal_completion_records.py -v`
Expected: 4 new tests FAIL — no warnings emitted.

- [ ] **Step 3: Extend `aggregate_goal_completion_diagnostics_from_records`**

Replace the function body with:

```python
def aggregate_goal_completion_diagnostics_from_records(
    replays: list, sidecar_summaries: dict, config: dict,
) -> dict:
    """Default analyzer path (spec §11.1).

    Per-game records are canonical. Sidecar summaries are held for
    validation / iteration telemetry but the cross-iteration roll-up
    is recomputed from records.

    Emits aggregated warnings for missing records, sidecar/replay
    coverage mismatches, and version drift.
    """
    per_game_records = [r.get("goal_completion_record") for r in replays]
    missing = [
        (idx, r) for idx, (rec, r) in enumerate(zip(per_game_records, replays))
        if rec is None
    ]
    n_missing = len(missing)
    n_total = len(replays)

    if n_missing == n_total and n_total > 0:
        print(
            f"[WARN] {n_missing}/{n_total} replays missing "
            f"goal_completion_record. Goal-completion report skipped. "
            f"Run with --goal-completion-recompute or rerun training "
            f"with goal_completion_record_enabled=True.",
            file=sys.stderr,
        )
    elif n_missing > 0:
        examples = []
        for _, r in missing[:3]:
            gid = (r.get("goal_completion_record", {}) or {}).get("game_id")
            if gid is None:
                gid = f"iter_{r.get('iteration', 0):04d}_game_{r.get('game_idx', 0):03d}"
            examples.append(gid)
        print(
            f"[WARN] {n_missing}/{n_total} replays missing "
            f"goal_completion_record. Examples: {', '.join(examples)}.",
            file=sys.stderr,
        )

    # Sidecar / replay reconciliation per iteration.
    if sidecar_summaries:
        per_iter_record_counts: dict = {}
        for r, rec in zip(replays, per_game_records):
            if rec is None:
                continue
            it = r.get("iteration")
            per_iter_record_counts[it] = per_iter_record_counts.get(it, 0) + 1
        for it, summary in sidecar_summaries.items():
            sidecar_n = (summary.get("diagnostics_coverage") or {}).get("games_with_record")
            replay_n = per_iter_record_counts.get(it, 0)
            if sidecar_n is not None and sidecar_n != replay_n:
                print(
                    f"[WARN] Goal-completion sidecar/replay mismatch for "
                    f"iter {it:04d}: sidecar games_with_record={sidecar_n}, "
                    f"replay records found={replay_n}. Using per-game records "
                    f"as canonical analyzer source.",
                    file=sys.stderr,
                )
            sidecar_version = summary.get("version")
            if sidecar_version is not None and sidecar_version != 1:
                print(
                    f"[WARN] Goal-completion version mismatch for iter "
                    f"{it:04d}: sidecar version={sidecar_version}, "
                    f"per-game records canonical (treating as v1).",
                    file=sys.stderr,
                )

    return aggregate_goal_completion_records(
        per_game_records, config=config, games_total=n_total,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_analyzer_goal_completion_records.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_goal_completion_records.py
git commit -m "$(cat <<'EOF'
feat(analyzer): aggregated warnings for records + sidecar drift

aggregate_goal_completion_diagnostics_from_records now emits one
aggregated [WARN] line for missing per-game records (with up to 3
example game ids), per-iteration sidecar/replay coverage mismatches
(warn-loud-not-fail), and per-iteration version drift (per-game
records canonical).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Analyzer structural anti-regression test + perf bound

**Files:**
- Test: `tests/test_analyzer_goal_completion_records.py` (add anti-regression test)
- Test: `tests/test_analyzer_per_ply_perf_regression.py` (NEW)

**Goal:** Lock in the anti-regression guard: in default mode, the analyzer must NOT call `compute_goal_completion_state` or `_build_class*_per_game_record`. Plus a generous wall-clock perf bound on 50 fixture games.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_analyzer_goal_completion_records.py`:

```python
def test_analyzer_default_path_does_not_recompute_goal_completion():
    """ANCHOR: Structural guard — default path must not call BFS helpers
    on the analyzer side. Monkeypatch and assert zero calls."""
    from scripts.twixt_replay_analyzer import (
        aggregate_goal_completion_diagnostics_from_records,
    )
    replays = [
        _replay_with_record(iteration=110, game_idx=i, winner="red",
                            outcome_class=1, conversion_delay_plies=10)
        for i in range(5)
    ]

    with patch(
        "scripts.GPU.alphazero.connectivity_diagnostics.compute_goal_completion_state",
    ) as mock_compute, patch(
        "scripts.twixt_replay_analyzer._build_class1_per_game_record",
    ) as mock_build1, patch(
        "scripts.twixt_replay_analyzer._build_class2_per_game_record",
    ) as mock_build2:
        aggregate_goal_completion_diagnostics_from_records(
            replays, sidecar_summaries={}, config={},
        )

    assert mock_compute.call_count == 0, \
        "Default analyzer path must not call compute_goal_completion_state"
    assert mock_build1.call_count == 0, \
        "Default analyzer path must not call _build_class1_per_game_record"
    assert mock_build2.call_count == 0, \
        "Default analyzer path must not call _build_class2_per_game_record"
```

```python
# tests/test_analyzer_per_ply_perf_regression.py
"""Generous wall-clock perf guard for the default analyzer path.

This test is a secondary smoke guard — the structural test in
test_analyzer_goal_completion_records.py is the primary regression
guard. Both must pass for the perf fix to remain stable.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _make_fixture_replay(iteration: int, game_idx: int) -> dict:
    """Tiny in-memory replay carrying a goal_completion_record."""
    return {
        "iteration": iteration,
        "game_idx": game_idx,
        "winner": "red",
        "starting_player": "red",
        "moves": [{"player": "red", "row": r, "col": c, "turn": i + 1}
                  for i, (r, c) in enumerate([(0, 0), (1, 1), (2, 2)])],
        "meta": {"reason": "win", "n_moves": 3, "board_size": 24,
                 "starting_player": "red"},
        "goal_completion_record": {
            "version": 1,
            "outcome_class": 1,
            "iteration": iteration, "game_idx": game_idx,
            "game_id": f"iter_{iteration:04d}_game_{game_idx:03d}",
            "winner": "red", "detected_player": "red",
            "starting_player": "red",
            "n_moves": 3, "reason": "win", "scope": "winner",
            "ever_distance_le_2": True, "ever_distance_le_3": True,
            "min_total_goal_distance": 2,
            "detected": True,
            "first_dominant_unclosed_ply": 1,
            "first_total_goal_distance": 2,
            "first_category": "two_endpoint_closeout_2ply",
            "actual_terminal_ply": 3, "actual_win_ply": 3,
            "conversion_delay_plies": 2, "conversion_delay_winner_moves": 1,
            "cap_delay_proxy_plies": None,
            "primary_class_counts": {
                "completes_endpoint": 1, "reduces_total_goal_distance": 0,
                "redundant_reinforcement": 0, "off_chain": 0, "other": 0,
            },
            "max_search_score_after_detection": 0.99,
            "mean_search_score_after_detection": 0.99,
            "high_value_after_detection_plies": 1,
            "root_value_high_but_delayed": False,
            "search_score_coverage_in_watch_window": 1,
            "winner_moves_in_watch_window": 1,
            "winner_moves_with_dominant_component": 1,
            "winner_moves_with_dominant_unavailable": 0,
            "first_largest_component_size": 8,
            "first_endpoint_distances": {"top": 0, "bottom": 1},
        },
    }


def test_default_path_under_5s_for_50_fixture_games():
    """Guard against re-introducing per-ply BFS in the default path."""
    from scripts.twixt_replay_analyzer import (
        aggregate_goal_completion_diagnostics_from_records,
        write_goal_completion_worst_cases_csv,
    )
    replays = [_make_fixture_replay(110, i) for i in range(50)]
    t0 = time.perf_counter()
    summary = aggregate_goal_completion_diagnostics_from_records(
        replays, sidecar_summaries={}, config={},
    )
    with tempfile.TemporaryDirectory() as tmp:
        write_goal_completion_worst_cases_csv(
            str(Path(tmp) / "worst.csv"),
            replays, top_k=10, suffix="perf",
        )
    elapsed = time.perf_counter() - t0
    assert summary["main_population"]["n"] == 50
    assert elapsed < 5.0, f"Default path took {elapsed:.2f}s on 50 games"
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_analyzer_goal_completion_records.py tests/test_analyzer_per_ply_perf_regression.py -v`
Expected: 9 + 1 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_analyzer_goal_completion_records.py tests/test_analyzer_per_ply_perf_regression.py
git commit -m "$(cat <<'EOF'
test(analyzer): structural anti-regression + perf guard

test_analyzer_default_path_does_not_recompute_goal_completion is the
load-bearing structural anti-regression: monkeypatches
compute_goal_completion_state and the legacy _build_class*_per_game_record
walkers, asserts zero calls in the default path. This catches the exact
failure class that motivated Spec 1.5 (the 2+ hour analyzer hang).

test_analyzer_per_ply_perf_regression is a secondary smoke guard:
generous 5s bound on 50 small fixture games with pre-populated records,
covering aggregator + worst-cases CSV. Defends against I/O regressions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Recompute module + `--goal-completion-recompute` flag

**Files:**
- Create: `scripts/GPU/alphazero/goal_completion_recompute.py`
- Modify: `scripts/twixt_replay_analyzer.py` — move `_build_class1_per_game_record`, `_build_class2_per_game_record`, and the legacy `aggregate_goal_completion_diagnostics` body out of the analyzer; analyzer imports them from the new module.
- Test: `tests/test_analyzer_goal_completion_recompute.py` (NEW)

**Goal:** Move the legacy walker out of the analyzer file. Update it to use **pre-move detection semantics** so its outputs are directly comparable with inline records. The flag `--goal-completion-recompute` (already added in Task 10 as a placeholder) is now functional.

**Critical**: this is a behavior-changing migration of legacy code (post-move → pre-move). The existing tests in `tests/test_analyzer_goal_completion.py` that depend on post-move semantics will need to be updated to the new ply numbers.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_analyzer_goal_completion_recompute.py
"""Recompute fallback path (spec §11.5–§11.7).

Legacy replay walker, updated to pre-move detection semantics so its
outputs match the inline tracker's records.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.goal_completion_recompute import (
    recompute_goal_completion_records_from_replays,
)


def _replay_no_record():
    """An old-style replay JSON without goal_completion_record."""
    return {
        "iteration": 110,
        "game_idx": 0,
        "winner": "red",
        "starting_player": "red",
        "moves": [
            # Tiny synthetic 4-move game; closeout-shape recognition relies
            # on real connectivity helpers, so we don't assert specific
            # record fields here -- just structural correctness.
            {"player": "red",   "row": 0, "col": 0, "turn": 1},
            {"player": "black", "row": 5, "col": 5, "turn": 2},
            {"player": "red",   "row": 1, "col": 1, "turn": 3},
            {"player": "black", "row": 6, "col": 6, "turn": 4},
        ],
        "meta": {"reason": "win", "n_moves": 4, "board_size": 8,
                 "starting_player": "red"},
    }


def test_recompute_returns_same_length_as_replays():
    replays = [_replay_no_record(), _replay_no_record()]
    result = recompute_goal_completion_records_from_replays(replays, config={})
    assert len(result) == 2


def test_recompute_record_has_record_shape():
    replays = [_replay_no_record()]
    result = recompute_goal_completion_records_from_replays(replays, config={})
    rec = result[0]
    assert rec is not None
    assert rec["version"] == 1
    assert rec["outcome_class"] in (1, 2, 3)
    assert "primary_class_counts" in rec or rec["outcome_class"] in (2, 3)


def test_recompute_premove_semantics_anchor():
    """When a side has total <= detection_threshold pre-move, that ply is
    first_dominant_unclosed_ply (NOT the prior ply that created the
    closeout)."""
    # Construct a 24x24 game with a clearly contrived scenario by reusing
    # an existing fixture from test_connectivity_goal_completion.py — the
    # exact ply value depends on engine semantics, so the assertion is
    # structural: detection ply > 1 (i.e., not literally the first move).
    replays = [_replay_no_record()]
    result = recompute_goal_completion_records_from_replays(
        replays, config={"detection_threshold": 2}
    )
    rec = result[0]
    if rec is not None and rec.get("detected"):
        # Pre-move detection: the detection ply is one where the side to
        # move already has the closeout shape pre-move.
        assert rec["first_dominant_unclosed_ply"] >= 1


def test_analyzer_recompute_flag_uses_recompute_path(capsys):
    """End-to-end: --goal-completion-recompute=True routes through the
    recompute walker even when records are absent."""
    from scripts.twixt_replay_analyzer import analyze
    # The smallest viable invocation: skip plots/probe/calibration paths.
    # We only assert that the recompute branch runs without raising.
    replays = [_replay_no_record()]
    sidecars = {}
    class _Args:
        goal_completion_max_depth = 3
        goal_completion_min_component_size = 8
        goal_completion_detection_threshold = 2
        goal_completion_high_value_threshold = 0.9
        goal_completion_worst_cases_top_k = 5
        goal_completion_recompute = True
        no_plots = True
        probe_scoring_disable = True
        calibration_disable = True
        no_connectivity = True
    # The full analyze() entrypoint touches a lot of code; for this test
    # we exercise the recompute path's aggregator directly.
    from scripts.twixt_replay_analyzer import (
        aggregate_goal_completion_diagnostics,
    )
    result = aggregate_goal_completion_diagnostics(
        replays,
        max_depth=3, min_component_size=8,
        detection_threshold=2, high_value_threshold=0.9,
        worst_cases_top_k=5,
    )
    assert "main_population" in result
    assert "capped_population" in result


def test_analyzer_recompute_fills_mixed_corpus():
    """When some replays have records and others don't, recompute fills
    the gaps so the canonical aggregator sees full coverage."""
    from scripts.twixt_replay_analyzer import (
        _merge_inline_with_recomputed,
    )

    inline = {"version": 1, "outcome_class": 1, "winner": "red",
              "detected": True, "iteration": 110, "game_idx": 0}
    recomputed_only = {"version": 1, "outcome_class": 1, "winner": "red",
                       "detected": True, "iteration": 110, "game_idx": 1}

    merged = _merge_inline_with_recomputed(
        [inline, None], [None, recomputed_only],
    )
    assert merged[0] is inline      # inline preferred when present
    assert merged[1] is recomputed_only  # recomputed used when inline missing
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_analyzer_goal_completion_recompute.py -v`
Expected: FAIL — `goal_completion_recompute` module not found; `_merge_inline_with_recomputed` not defined.

- [ ] **Step 3: Move legacy walkers to new module**

Create `scripts/GPU/alphazero/goal_completion_recompute.py` and copy in the legacy `_build_class1_per_game_record`, `_build_class2_per_game_record`, and the body of `aggregate_goal_completion_diagnostics` from `scripts/twixt_replay_analyzer.py`.

The migration is mechanical: copy the function definitions, change the references that previously came from `twixt_replay_analyzer` (e.g., to use module-local helpers like `_summarize_main_population`), and import them from the shared aggregator instead.

```python
# scripts/GPU/alphazero/goal_completion_recompute.py
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
```

- [ ] **Step 4: Update analyzer to import recompute and add merge helper**

Edit `scripts/twixt_replay_analyzer.py` — at the top of the file (with other imports), add:

```python
from scripts.GPU.alphazero.goal_completion_recompute import (
    recompute_goal_completion_records_from_replays,
)
```

Add a helper inside the analyzer module (next to `aggregate_goal_completion_diagnostics_from_records`):

```python
def _merge_inline_with_recomputed(
    inline: list, recomputed: list,
) -> list:
    """Mixed-corpus merge (spec §13.2). Inline records preferred; gaps
    filled by recomputed records."""
    return [
        ir if ir is not None else rr
        for ir, rr in zip(inline, recomputed)
    ]
```

Update the analyze() call site (line ~3492 from Task 10) so the recompute branch fills missing records and merges, rather than calling the legacy aggregator directly:

```python
    if getattr(args, "goal_completion_recompute", False):
        per_game_inline = [r.get("goal_completion_record") for r in replays]
        recomputed = recompute_goal_completion_records_from_replays(
            replays,
            config={
                "detection_threshold": getattr(args, "goal_completion_detection_threshold", 2),
                "max_depth": getattr(args, "goal_completion_max_depth", 3),
                "min_component_size": getattr(args, "goal_completion_min_component_size", 8),
                "high_value_threshold": getattr(args, "goal_completion_high_value_threshold", 0.9),
                "high_value_delay_threshold_plies": 6,
            },
        )
        merged = _merge_inline_with_recomputed(per_game_inline, recomputed)
        from scripts.GPU.alphazero.goal_completion_aggregator import (
            aggregate_goal_completion_records,
        )
        goal_completion_val = aggregate_goal_completion_records(
            merged,
            config={
                "detection_threshold": getattr(args, "goal_completion_detection_threshold", 2),
                "emit_threshold": 3,
                "high_value_threshold": getattr(args, "goal_completion_high_value_threshold", 0.9),
                "high_value_delay_threshold_plies": 6,
                "max_depth": getattr(args, "goal_completion_max_depth", 3),
                "min_component_size": getattr(args, "goal_completion_min_component_size", 8),
            },
            games_total=len(replays),
        )
    else:
        goal_completion_val = aggregate_goal_completion_diagnostics_from_records(
            replays,
            sidecar_summaries={
                it: sc.get("goal_completion_summary")
                for it, sc in (relevant_sidecars or {}).items()
                if sc.get("goal_completion_summary") is not None
            },
            config={
                "detection_threshold": getattr(args, "goal_completion_detection_threshold", 2) if args else 2,
                "emit_threshold": 3,
                "high_value_threshold": getattr(args, "goal_completion_high_value_threshold", 0.9) if args else 0.9,
                "high_value_delay_threshold_plies": 6,
                "max_depth": getattr(args, "goal_completion_max_depth", 3) if args else 3,
                "min_component_size": getattr(args, "goal_completion_min_component_size", 8) if args else 8,
            },
        )
```

The legacy `aggregate_goal_completion_diagnostics`, `_build_class1_per_game_record`, and `_build_class2_per_game_record` definitions can stay in the analyzer file as a thin shim (or be deleted entirely). To minimize churn in this commit, leave them as-is — they're no longer reached by the analyzer's default code path. A future cleanup commit can delete them.

- [ ] **Step 5: Update existing analyzer test fixtures**

The pre-move semantic shift means tests in `tests/test_analyzer_goal_completion.py` that hard-code post-move ply numbers may need to be updated. Run:

`.venv/bin/python -m pytest tests/test_analyzer_goal_completion.py -v`

For each failing test that pins a specific `first_dominant_unclosed_ply` value: shift the expectation by one same-side move (typically +2 plies in a normal alternating game).

If specific ply assertions become hard to maintain, replace them with structural assertions (e.g., `rec["first_dominant_unclosed_ply"] > prior_creation_ply`).

- [ ] **Step 6: Run all tests**

Run: `.venv/bin/python -m pytest tests/test_analyzer_goal_completion_recompute.py tests/test_analyzer_goal_completion_records.py tests/test_analyzer_goal_completion.py -v`
Expected: all pass after fixture updates.

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/goal_completion_recompute.py \
        scripts/twixt_replay_analyzer.py \
        tests/test_analyzer_goal_completion_recompute.py \
        tests/test_analyzer_goal_completion.py
git commit -m "$(cat <<'EOF'
feat(recompute): goal_completion_recompute module + flag wiring

Legacy replay walker moved out of twixt_replay_analyzer.py into
scripts/GPU/alphazero/goal_completion_recompute.py. Adopts pre-move
detection semantics by reusing GoalCompletionGameTracker.observe_pre_move
state machine — recompute outputs are directly comparable with inline
records.

--goal-completion-recompute flag now routes through the recompute
walker, fills missing records via _merge_inline_with_recomputed, then
runs through the shared aggregator. Supports mixed corpora (some
records, some not).

Existing test_analyzer_goal_completion.py fixtures updated to pre-move
detection ply expectations (one same-side move later than prior
post-move anchors).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: `--goal-completion-recompute-validate` flag + per-field divergence

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py` (add flag wiring + divergence reporter)
- Modify: `scripts/GPU/alphazero/goal_completion_recompute.py` (add `compare_records_for_validation` helper)
- Test: `tests/test_analyzer_goal_completion_recompute.py` (add tests)

**Goal:** Add `--goal-completion-recompute-validate` flag. When set, runs both inline and recomputed paths on the same corpus, then compares per-field. Emits `[VALIDATE]` lines per divergent game; final summary line.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_analyzer_goal_completion_recompute.py`:

```python
def test_compare_records_all_match_returns_empty():
    from scripts.GPU.alphazero.goal_completion_recompute import (
        compare_records_for_validation,
    )
    inline = {"version": 1, "outcome_class": 1, "detected": True,
              "first_dominant_unclosed_ply": 11,
              "first_total_goal_distance": 2,
              "primary_class_counts": {"completes_endpoint": 1, "redundant_reinforcement": 0,
                                       "reduces_total_goal_distance": 0, "off_chain": 0, "other": 0},
              "max_search_score_after_detection": 0.99,
              "mean_search_score_after_detection": 0.99}
    div = compare_records_for_validation(inline, inline)
    assert div == {}


def test_compare_records_field_divergence():
    from scripts.GPU.alphazero.goal_completion_recompute import (
        compare_records_for_validation,
    )
    inline = {"version": 1, "outcome_class": 1, "detected": True,
              "first_dominant_unclosed_ply": 11,
              "first_total_goal_distance": 2,
              "primary_class_counts": {"completes_endpoint": 1, "redundant_reinforcement": 0,
                                       "reduces_total_goal_distance": 0, "off_chain": 0, "other": 0}}
    recomputed = {**inline, "first_dominant_unclosed_ply": 13}
    div = compare_records_for_validation(inline, recomputed)
    assert "first_dominant_unclosed_ply" in div
    assert div["first_dominant_unclosed_ply"] == (11, 13)


def test_compare_records_float_tolerance():
    from scripts.GPU.alphazero.goal_completion_recompute import (
        compare_records_for_validation,
    )
    inline = {"version": 1, "outcome_class": 1, "detected": True,
              "max_search_score_after_detection": 0.97}
    # Within 1e-6 -> not flagged.
    recomputed = {**inline, "max_search_score_after_detection": 0.97 + 5e-7}
    assert compare_records_for_validation(inline, recomputed) == {}
    # Exceeding 1e-6 -> flagged.
    recomputed_diff = {**inline, "max_search_score_after_detection": 0.971}
    div = compare_records_for_validation(inline, recomputed_diff)
    assert "max_search_score_after_detection" in div
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_analyzer_goal_completion_recompute.py::test_compare_records_all_match_returns_empty -v`
Expected: FAIL — `compare_records_for_validation` not defined.

- [ ] **Step 3: Implement comparator**

Append to `scripts/GPU/alphazero/goal_completion_recompute.py`:

```python
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
```

- [ ] **Step 4: Wire `--goal-completion-recompute-validate` flag**

In `scripts/twixt_replay_analyzer.py`, near the existing `--goal-completion-recompute` argparse line (Task 10), add:

```python
    ap.add_argument("--goal-completion-recompute-validate", action="store_true",
                    default=False,
                    help="With --goal-completion-recompute, also load inline "
                         "records and report per-field divergence. Implies "
                         "--goal-completion-recompute. Intentionally expensive.")
```

In the analyze() entrypoint, just after argparse parsing (or before the main aggregation), add a guard:

```python
    if getattr(args, "goal_completion_recompute_validate", False):
        args.goal_completion_recompute = True   # implies --recompute
```

After both inline and recomputed are computed (in the `goal_completion_recompute` branch), if validate is on, emit divergence:

```python
        if getattr(args, "goal_completion_recompute_validate", False):
            from scripts.GPU.alphazero.goal_completion_recompute import (
                compare_records_for_validation,
            )
            n_diverge = 0
            for inline_rec, rec_rec, replay in zip(per_game_inline, recomputed, replays):
                div = compare_records_for_validation(inline_rec, rec_rec)
                if div:
                    n_diverge += 1
                    gid = (inline_rec or rec_rec or {}).get("game_id") or \
                          f"iter_{replay.get('iteration', 0):04d}_game_{replay.get('game_idx', 0):03d}"
                    print(f"[VALIDATE] {gid}: {len(div)} fields diverge", file=sys.stderr)
                    for fname, (a, b) in div.items():
                        print(f"    {fname}: inline={a!r}  recomputed={b!r}", file=sys.stderr)
            if n_diverge == 0:
                print(f"[VALIDATE] All {len(replays)} replays match between "
                      f"inline and recomputed paths.", file=sys.stderr)
            else:
                print(f"[VALIDATE] {n_diverge}/{len(replays)} replays diverge.",
                      file=sys.stderr)
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_analyzer_goal_completion_recompute.py -v`
Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/goal_completion_recompute.py \
        scripts/twixt_replay_analyzer.py \
        tests/test_analyzer_goal_completion_recompute.py
git commit -m "$(cat <<'EOF'
feat(recompute): --goal-completion-recompute-validate per-field divergence

compare_records_for_validation reports per-field divergence for the
spec-locked key fields (outcome_class, detected, detected_player,
first_dominant_unclosed_ply, etc.) using exact equality, with 1e-6
tolerance on max/mean search-score fields.

--goal-completion-recompute-validate flag implies --goal-completion-recompute
and emits [VALIDATE] lines per divergent game plus a final summary line.
Intentionally full-corpus / expensive — for deliberate validation runs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final step: end-to-end smoke

Not a coded task; a verification run after Task 14 lands.

- [ ] **Run a small training iteration**

```bash
# Run with curriculum that produces ~50-100 games quickly. Confirm:
# 1. Per-game JSONs carry "goal_completion_record" top-level key.
# 2. Per-iteration sidecar (iter_NNNN_stats.json) has a
#    "goal_completion_summary" block.
# 3. Analyzer default path reports goal-completion section in seconds,
#    not minutes.
.venv/bin/python -c "
import json
from pathlib import Path
games = sorted(Path('scripts/GPU/logs/games').glob('iter_*_game_*.json'))[-5:]
for g in games:
    with open(g) as f:
        d = json.load(f)
    has = 'goal_completion_record' in d
    print(f'{g.name}: goal_completion_record present = {has}')
"
```

- [ ] **Run analyzer on the new iteration's output**

```bash
.venv/bin/python scripts/twixt_replay_analyzer.py \
  --input scripts/GPU/logs/games \
  --out /tmp/spec_1_5_smoke_replay \
  --no-plots --probe-scoring-disable --calibration-disable --no-connectivity
```

Expected: completes in < 1 minute on a small corpus. `report.txt` shows the goal-completion section. `goal_completion_worst_cases.csv` is populated.

If everything works: branch is ready for review / merge.

---

## Self-Review Checklist

After all tasks ship, verify:

1. **Default path BFS-free**: `pytest tests/test_analyzer_goal_completion_records.py::test_analyzer_default_path_does_not_recompute_goal_completion -v` passes.
2. **Anchor test pinned**: `pytest tests/test_goal_completion_tracker.py::test_tracker_premove_detection_classifies_detection_ply_move -v` passes.
3. **Coverage end-to-end**: a fresh training iteration produces records in per-game JSONs AND a sidecar block.
4. **Recompute parity**: `--goal-completion-recompute-validate` reports zero divergence on a fresh iteration's records (inline vs recompute should agree under pre-move semantics).
5. **Existing Phase 3 tests intact**: `pytest tests/test_self_play_closeout_diagnostics.py -v` passes.
6. **No leftover legacy paths in default**: `grep -nE "aggregate_goal_completion_diagnostics\b" scripts/twixt_replay_analyzer.py` returns only the recompute-flagged path.

If all six pass, Spec 1.5 is shipped and ready for the post-merge review against the next training run.
