# Recovery / Re-targeting Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a diagnostic-only telemetry stream that detects games where a side, after its position collapses, fails to re-target — surfacing the data in per-iteration sidecars, the analyzer report, and a worst-cases CSV.

**Architecture:** A new `recovery_retargeting_diagnostics` module owns the per-game tracker, classifier, and per-iter aggregator. `self_play.play_game` hosts the per-ply hook; the IPC + trainer transport mirrors the Fix 2 pattern from commit `d788023f4`. Default-on at 11 config parameters; explicit `--recovery-retargeting-disabled` opt-out.

**Tech Stack:** Python 3.14, pytest, MLX-backed self-play infrastructure. Uses existing `connectivity_diagnostics.compute_goal_completion_state` and `state._get_connected_component`.

**Spec reference:** `docs/superpowers/specs/2026-05-12-recovery-retargeting-diagnostic-design.md`

**Current baseline:** 168 passed, 2 skipped (`tests/test_mcts*.py tests/test_analyzer_*.py tests/test_self_play_closeout*.py tests/test_train_closeout*.py tests/test_game_saver*.py`).

---

## File Structure

**New files:**
- `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py` — config dataclass, component helpers, trigger, classifier, tracker, aggregator
- `tests/test_recovery_retargeting_diagnostics.py` — unit tests

**Modified files:**
- `scripts/GPU/alphazero/self_play.py` — `play_game` hook + `GameRecord` field + finalize call
- `scripts/GPU/alphazero/ipc_messages.py` — `GameComplete.recovery_retargeting_record` field
- `scripts/GPU/alphazero/self_play_worker.py` — IPC forwarding
- `scripts/GPU/alphazero/trainer.py` — `train()` kwargs, IPC append, serial append, `_inject_iteration` extension, sidecar emit, startup banner
- `scripts/GPU/alphazero/train.py` — CLI flags, validation, `train_kwargs.update`
- `scripts/GPU/alphazero/game_saver.py` — `recovery_retargeting_record.game_idx` / `game_id` reconciliation
- `scripts/twixt_replay_analyzer.py` — imports aggregator, adds report formatter + 2 CSV writers, wired into `analyze()`, plus `--recovery-retargeting-worst-cases-top-k` flag

---

# Phase 1 — Core diagnostics module

## Task 1: Create `RecoveryRetargetingConfig` + validation

**Files:**
- Create: `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py`
- Test: `tests/test_recovery_retargeting_diagnostics.py`

- [ ] **Step 1: Write the config tests**

Create `tests/test_recovery_retargeting_diagnostics.py`:

```python
"""Tests for Spec 4 recovery / re-targeting diagnostic."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from scripts.GPU.alphazero.recovery_retargeting_diagnostics import (
    RecoveryRetargetingConfig,
    validate_config,
)


def test_config_defaults_match_spec():
    c = RecoveryRetargetingConfig()
    assert c.enabled is True
    assert c.collapse_value_threshold == -0.75
    assert c.severe_collapse_value_threshold == -0.90
    assert c.diffuse_root_top1_threshold == 0.20
    assert c.very_diffuse_root_top1_threshold == 0.15
    assert c.delta_threshold == 0.50
    assert c.delta_max_current_score == -0.30
    assert c.alternate_component_min_size == 4
    assert c.classify_defense is True
    assert c.max_sampled_moves_per_side == 32
    assert c.sample_all_moves is False


def test_validate_collapse_lt_delta_max_current_score():
    cfg = RecoveryRetargetingConfig(collapse_value_threshold=-0.30, delta_max_current_score=-0.30)
    with pytest.raises(ValueError, match="collapse_value_threshold"):
        validate_config(cfg)


def test_validate_severe_le_collapse():
    cfg = RecoveryRetargetingConfig(collapse_value_threshold=-0.75, severe_collapse_value_threshold=-0.50)
    with pytest.raises(ValueError, match="severe_collapse_value_threshold"):
        validate_config(cfg)


def test_validate_very_diffuse_le_diffuse():
    cfg = RecoveryRetargetingConfig(diffuse_root_top1_threshold=0.20, very_diffuse_root_top1_threshold=0.30)
    with pytest.raises(ValueError, match="very_diffuse_root_top1_threshold"):
        validate_config(cfg)


def test_validate_top1_range():
    with pytest.raises(ValueError, match="diffuse_root_top1_threshold"):
        validate_config(RecoveryRetargetingConfig(diffuse_root_top1_threshold=1.5))


def test_validate_delta_positive():
    with pytest.raises(ValueError, match="delta_threshold"):
        validate_config(RecoveryRetargetingConfig(delta_threshold=0.0))


def test_validate_alternate_component_min_size_positive():
    with pytest.raises(ValueError, match="alternate_component_min_size"):
        validate_config(RecoveryRetargetingConfig(alternate_component_min_size=0))


def test_validate_max_sampled_non_negative():
    with pytest.raises(ValueError, match="max_sampled_moves_per_side"):
        validate_config(RecoveryRetargetingConfig(max_sampled_moves_per_side=-1))


def test_validate_default_config_passes():
    validate_config(RecoveryRetargetingConfig())   # must not raise
```

- [ ] **Step 2: Run → fail (module doesn't exist)**

```bash
.venv/bin/pytest tests/test_recovery_retargeting_diagnostics.py -v
```

Expected: ModuleNotFoundError on `scripts.GPU.alphazero.recovery_retargeting_diagnostics`.

- [ ] **Step 3: Create the module with the config + validation**

Create `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py`:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_recovery_retargeting_diagnostics.py -v
```

Expected: 9 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/recovery_retargeting_diagnostics.py tests/test_recovery_retargeting_diagnostics.py
git commit -m "feat(recovery_retargeting): config dataclass + validation (Spec 4)"
```

---

## Task 2: Implement component analysis helpers

**Files:**
- Modify: `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py`
- Test: `tests/test_recovery_retargeting_diagnostics.py`

- [ ] **Step 1: Write tests for `find_components`, `local_to_existing`, `selected_component_after`**

Append to `tests/test_recovery_retargeting_diagnostics.py`:

```python
from scripts.GPU.alphazero.recovery_retargeting_diagnostics import (
    find_components,
    is_local_to_existing,
    knight_neighbors,
    selected_component_after,
)


class _StubState:
    """Minimal state shim: exposes .pegs dict, apply_move, _get_connected_component."""
    def __init__(self, pegs_dict, to_move="black"):
        # pegs_dict: {(r, c): "red" | "black"}
        self.pegs = dict(pegs_dict)
        self.to_move = to_move

    def apply_move(self, move):
        """Return a NEW _StubState with `move` placed for the current side.

        The real TwixtState.apply_move alternates to_move; the stub mirrors that.
        Tests that need a specific side-to-move should construct the stub with
        the desired to_move and call apply_move once.
        """
        new_pegs = dict(self.pegs)
        new_pegs[move] = self.to_move
        return _StubState(new_pegs, to_move="red" if self.to_move == "black" else "black")

    def _get_connected_component(self, peg, side):
        # BFS over knight-distance neighbors of the same color, no enemy blocking check
        # (sufficient for unit tests; real state has full enemy-block logic)
        if peg not in self.pegs or self.pegs[peg] != side:
            return frozenset()
        visited = {peg}
        frontier = [peg]
        while frontier:
            cur = frontier.pop()
            for n in knight_neighbors(*cur):
                if n in self.pegs and self.pegs[n] == side and n not in visited:
                    visited.add(n)
                    frontier.append(n)
        return frozenset(visited)


def _state_after(state_before, side, move):
    """Test helper: build a new _StubState representing state_before + move for side."""
    new_pegs = dict(state_before.pegs)
    new_pegs[move] = side
    return _StubState(new_pegs)


def test_knight_neighbors_returns_8_offsets():
    n = set(knight_neighbors(5, 5))
    assert n == {(3, 4), (3, 6), (4, 3), (4, 7), (6, 3), (6, 7), (7, 4), (7, 6)}


def test_find_components_groups_by_bridge_connectivity():
    # Two black pegs at knight distance form one component; a third isolated peg is its own component.
    state = _StubState({(0, 0): "black", (1, 2): "black", (10, 10): "black"})
    comps = find_components(state, "black")
    assert len(comps) == 2
    sizes = sorted(len(c) for c in comps)
    assert sizes == [1, 2]


def test_find_components_skips_other_color():
    state = _StubState({(0, 0): "black", (1, 2): "red"})
    comps = find_components(state, "black")
    assert len(comps) == 1
    assert next(iter(comps)) == frozenset({(0, 0)})


def test_is_local_to_existing_true_when_knight_neighbor_exists():
    state = _StubState({(0, 0): "black"})
    assert is_local_to_existing(state, "black", (1, 2)) is True
    assert is_local_to_existing(state, "black", (2, 1)) is True


def test_is_local_to_existing_false_when_no_same_color_knight_neighbor():
    state = _StubState({(0, 0): "black"})
    # (2, 2) is Chebyshev-2 from (0, 0) but NOT knight-distance.
    assert is_local_to_existing(state, "black", (2, 2)) is False


def test_is_local_to_existing_ignores_other_color():
    state = _StubState({(1, 2): "red"})
    assert is_local_to_existing(state, "black", (0, 0)) is False


def test_selected_component_after_includes_new_peg_and_merged_components():
    """Caller passes state_after (post-move). Helper does NOT mutate state."""
    # Two prior black pegs at (0, 0) and (4, 0). (2, 1) is knight-distance from both.
    state_before = _StubState({(0, 0): "black", (4, 0): "black"})
    state_after = _state_after(state_before, "black", (2, 1))
    comp_after = selected_component_after(state_after, "black", (2, 1))
    assert (0, 0) in comp_after
    assert (4, 0) in comp_after
    assert (2, 1) in comp_after
    assert len(comp_after) == 3


def test_selected_component_after_uses_post_move_state_without_mutation():
    """The helper must NOT mutate state_after.pegs (or any state)."""
    state_before = _StubState({(0, 0): "black", (4, 0): "black"})
    state_after = _state_after(state_before, "black", (2, 1))
    pegs_before_call = dict(state_after.pegs)
    selected_component_after(state_after, "black", (2, 1))
    assert state_after.pegs == pegs_before_call
    # state_before is untouched (it never received the move).
    assert (2, 1) not in state_before.pegs
```

- [ ] **Step 2: Run → fail**

```bash
.venv/bin/pytest tests/test_recovery_retargeting_diagnostics.py -v
```

Expected: ImportError on the new helper names.

- [ ] **Step 3: Implement the helpers**

Append to `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py`:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_recovery_retargeting_diagnostics.py -v
```

Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/recovery_retargeting_diagnostics.py tests/test_recovery_retargeting_diagnostics.py
git commit -m "feat(recovery_retargeting): component analysis helpers + knight-distance locality (Spec 4 §3.1)"
```

---

## Task 3: Implement trigger evaluation

**Files:**
- Modify: `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py`
- Test: `tests/test_recovery_retargeting_diagnostics.py`

- [ ] **Step 1: Write trigger tests**

Append to `tests/test_recovery_retargeting_diagnostics.py`:

```python
from scripts.GPU.alphazero.recovery_retargeting_diagnostics import evaluate_trigger


def _cfg(**overrides):
    return RecoveryRetargetingConfig(**overrides)


def test_steady_state_trigger_fires_when_score_and_top1_both_low():
    r = evaluate_trigger(
        current_search_score=-0.80, root_top1_share=0.10,
        previous_own_search_score=-0.70, config=_cfg(),
    )
    assert r["triggered"] is True
    assert r["trigger_reason"] == "steady_state"


def test_steady_state_does_not_fire_when_score_bad_but_root_confident():
    r = evaluate_trigger(
        current_search_score=-0.80, root_top1_share=0.40,
        previous_own_search_score=None, config=_cfg(),
    )
    assert r["triggered"] is False
    assert r["trigger_reason"] is None


def test_steady_state_does_not_fire_when_root_diffuse_but_score_ok():
    r = evaluate_trigger(
        current_search_score=-0.20, root_top1_share=0.10,
        previous_own_search_score=None, config=_cfg(),
    )
    assert r["triggered"] is False


def test_delta_precursor_fires_on_sharp_drop():
    r = evaluate_trigger(
        current_search_score=-0.40, root_top1_share=0.12,
        previous_own_search_score=0.30, config=_cfg(),
    )
    assert r["triggered"] is True
    assert r["trigger_reason"] == "delta_precursor"


def test_delta_precursor_guard_blocks_when_current_score_still_positive():
    # Drop from +0.95 to +0.40 = delta 0.55 >= 0.50, top1 diffuse, but current > -0.30 guard.
    r = evaluate_trigger(
        current_search_score=0.40, root_top1_share=0.10,
        previous_own_search_score=0.95, config=_cfg(),
    )
    assert r["triggered"] is False


def test_trigger_reason_both_when_both_paths_fire():
    # current=-0.80 (steady fires) AND previous=-0.20 → delta=0.60 → delta also fires
    r = evaluate_trigger(
        current_search_score=-0.80, root_top1_share=0.10,
        previous_own_search_score=-0.20, config=_cfg(),
    )
    assert r["triggered"] is True
    assert r["trigger_reason"] == "both"


def test_missing_search_score_skips_trigger():
    r = evaluate_trigger(
        current_search_score=None, root_top1_share=0.10,
        previous_own_search_score=-0.30, config=_cfg(),
    )
    assert r["triggered"] is False
    assert r["missing_search_score"] is True


def test_missing_root_top1_share_skips_trigger():
    r = evaluate_trigger(
        current_search_score=-0.80, root_top1_share=None,
        previous_own_search_score=None, config=_cfg(),
    )
    assert r["triggered"] is False
    assert r["missing_root_top1_share"] is True


def test_severity_flags_reflect_current_score_and_share():
    r = evaluate_trigger(
        current_search_score=-0.95, root_top1_share=0.10,
        previous_own_search_score=None, config=_cfg(),
    )
    assert r["triggered"] is True
    assert r["is_severe_collapse"] is True
    assert r["is_very_diffuse"] is True
```

- [ ] **Step 2: Run → fail**

```bash
.venv/bin/pytest tests/test_recovery_retargeting_diagnostics.py::test_steady_state_trigger_fires_when_score_and_top1_both_low -v
```

Expected: ImportError on `evaluate_trigger`.

- [ ] **Step 3: Implement the trigger**

Append to `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py`:

```python
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
        previous_own_search_score is not None
        and delta_value is not None
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
        triggered = True
    elif delta_precursor:
        trigger_reason = "delta_precursor"
        triggered = True
    elif steady_state:
        trigger_reason = "steady_state"
        triggered = True
    else:
        trigger_reason = None
        triggered = False

    return {
        "triggered": triggered,
        "trigger_reason": trigger_reason,
        "is_severe_collapse": current_search_score <= config.severe_collapse_value_threshold,
        "is_very_diffuse": root_top1_share <= config.very_diffuse_root_top1_threshold,
        "missing_search_score": False,
        "missing_root_top1_share": False,
        "search_score_delta": delta_value,
    }
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_recovery_retargeting_diagnostics.py -v
```

Expected: all 18 tests PASSED (9 from Task 1 + 6 from Task 2 + 9 here).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/recovery_retargeting_diagnostics.py tests/test_recovery_retargeting_diagnostics.py
git commit -m "feat(recovery_retargeting): trigger evaluation (delta-precursor + steady-state, Spec 4 §2)"
```

---

## Task 4: Implement primary-class classifier

**Files:**
- Modify: `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py`
- Test: `tests/test_recovery_retargeting_diagnostics.py`

- [ ] **Step 1: Write classifier tests**

Append to `tests/test_recovery_retargeting_diagnostics.py`:

```python
from scripts.GPU.alphazero.recovery_retargeting_diagnostics import classify_move


def _classify(
    state_before, side, move,
    own_td_before, own_td_after,
    opp_td_before=None, opp_td_after=None,
    classify_defense=True,
    alternate_component_min_size=4,
    state_after=None,
):
    """Test harness wrapper. Caller may pass state_after explicitly to override
    the default (state_before + move) — useful for testing isolated-bridge scenarios."""
    if state_after is None:
        state_after = _state_after(state_before, side, move)
    return classify_move(
        state_before=state_before,
        state_after=state_after,
        side=side,
        move=move,
        own_total_goal_distance_before=own_td_before,
        own_total_goal_distance_after=own_td_after,
        opponent_total_goal_distance_before=opp_td_before,
        opponent_total_goal_distance_after=opp_td_after,
        classify_defense=classify_defense,
        alternate_component_min_size=alternate_component_min_size,
    )


def test_classify_move_does_not_mutate_state_before():
    """classify_move must not mutate state_before.pegs in any code path."""
    state_before = _StubState({(0, 0): "black", (1, 2): "black"})
    state_after = _state_after(state_before, "black", (5, 5))
    pegs_snapshot = dict(state_before.pegs)
    classify_move(
        state_before=state_before, state_after=state_after,
        side="black", move=(5, 5),
        own_total_goal_distance_before=4, own_total_goal_distance_after=4,
        opponent_total_goal_distance_before=None,
        opponent_total_goal_distance_after=None,
        classify_defense=True, alternate_component_min_size=4,
    )
    assert state_before.pegs == pegs_snapshot


def test_classifies_blocks_opponent_closeout():
    state = _StubState({(0, 0): "black"})
    r = _classify(state, "black", (5, 5),
                  own_td_before=6, own_td_after=6,
                  opp_td_before=2, opp_td_after=3)
    assert r["primary_class"] == "blocks_opponent_closeout"
    assert r["flags"]["blocked_opponent_closeout"] is True


def test_classifies_reduces_own_goal_distance():
    state = _StubState({(0, 0): "black"})
    r = _classify(state, "black", (5, 5),
                  own_td_before=4, own_td_after=3)
    assert r["primary_class"] == "reduces_own_goal_distance"


def test_priority_defense_beats_reduces_goal_distance():
    state = _StubState({(0, 0): "black"})
    r = _classify(state, "black", (5, 5),
                  own_td_before=4, own_td_after=3,         # reduces own goal distance
                  opp_td_before=2, opp_td_after=3)         # also blocks opponent
    assert r["primary_class"] == "blocks_opponent_closeout"


def test_classifies_starts_or_extends_alternate_via_opens_new():
    # Dominant black component at (0,0)-(1,2) size 2; move at (10,10) opens new component size 1.
    # alternate_component_min_size=1 to make this test independent of default.
    state = _StubState({(0, 0): "black", (1, 2): "black"})
    r = _classify(state, "black", (10, 10),
                  own_td_before=5, own_td_after=5,
                  alternate_component_min_size=1)
    assert r["primary_class"] == "starts_or_extends_alternate_component"
    assert r["flags"]["opens_new_component"] is True


def test_classifies_connects_to_existing_component():
    # Move bridges to dominant component but does NOT reduce td.
    state = _StubState({(0, 0): "black", (1, 2): "black", (3, 1): "black"})
    # New move at (4, 3) is knight-from (3, 1) and joins dominant.
    r = _classify(state, "black", (4, 3),
                  own_td_before=5, own_td_after=5)
    assert r["primary_class"] == "connects_to_existing_component"
    assert r["flags"]["extends_dominant_component"] is True


def test_classifies_redundant_local_reinforcement():
    # Move is local (knight-distance) to a same-color peg, but the simulated
    # bridge is blocked (e.g., enemy peg between them), so the move does NOT
    # actually join the component. Use _IsolateState for both before & after.
    class _IsolateState(_StubState):
        def _get_connected_component(self, peg, side):
            if peg not in self.pegs or self.pegs[peg] != side:
                return frozenset()
            return frozenset({peg})

    state_before = _IsolateState({(0, 0): "black"})
    new_pegs = dict(state_before.pegs)
    new_pegs[(1, 2)] = "black"
    state_after = _IsolateState(new_pegs)
    r = _classify(state_before, "black", (1, 2),                # knight-local to (0, 0)
                  own_td_before=5, own_td_after=5,
                  state_after=state_after)
    assert r["primary_class"] == "redundant_local_reinforcement"
    assert r["flags"]["local_to_existing"] is True
    assert r["flags"]["extends_dominant_component"] is False


def test_classifies_off_plan_or_unclear_fallback():
    state = _StubState({(0, 0): "black"})
    # Move is far away (not local), no td change, no defense.
    r = _classify(state, "black", (15, 15),
                  own_td_before=5, own_td_after=5)
    assert r["primary_class"] == "off_plan_or_unclear"


def test_local_to_existing_uses_knight_not_chebyshev():
    # (2, 2) is Chebyshev-2 from (0, 0) but NOT knight-2.
    class _IsolateState(_StubState):
        def _get_connected_component(self, peg, side):
            if peg not in self.pegs or self.pegs[peg] != side:
                return frozenset()
            return frozenset({peg})

    state_before = _IsolateState({(0, 0): "black"})
    new_pegs = dict(state_before.pegs)
    new_pegs[(2, 2)] = "black"
    state_after = _IsolateState(new_pegs)
    r = _classify(state_before, "black", (2, 2),
                  own_td_before=5, own_td_after=5,
                  state_after=state_after)
    assert r["flags"]["local_to_existing"] is False
    assert r["primary_class"] == "off_plan_or_unclear"


def test_classify_defense_disabled_never_returns_blocks_opponent_closeout():
    state = _StubState({(0, 0): "black"})
    r = _classify(state, "black", (5, 5),
                  own_td_before=4, own_td_after=3,
                  opp_td_before=2, opp_td_after=3,
                  classify_defense=False)
    assert r["primary_class"] == "reduces_own_goal_distance"
    assert r["flags"]["blocked_opponent_closeout"] is False
```

- [ ] **Step 2: Run → fail**

```bash
.venv/bin/pytest tests/test_recovery_retargeting_diagnostics.py::test_classifies_blocks_opponent_closeout -v
```

Expected: ImportError on `classify_move`.

- [ ] **Step 3: Implement the classifier**

Append to `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py`:

```python
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
    return max(components, key=lambda c: (len(c), -min(c)[0] if c else 0, -min(c)[1] if c else 0))


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
    local_flag = is_local_to_existing(state_before, side, move)

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
        "local_to_existing":              local_flag,
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
    elif (local_flag
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
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_recovery_retargeting_diagnostics.py -v
```

Expected: all PASSED (27 tests so far).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/recovery_retargeting_diagnostics.py tests/test_recovery_retargeting_diagnostics.py
git commit -m "feat(recovery_retargeting): primary-class classifier with priority order (Spec 4 §3)"
```

---

# Phase 2 — Tracker

## Task 5: Implement `RecoveryRetargetingTracker.observe_move`

**Files:**
- Modify: `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py`
- Test: `tests/test_recovery_retargeting_diagnostics.py`

- [ ] **Step 1: Write tracker tests**

Append to `tests/test_recovery_retargeting_diagnostics.py`:

```python
from scripts.GPU.alphazero.recovery_retargeting_diagnostics import RecoveryRetargetingTracker


def _gc_stub(td_before, td_after):
    """Helper to build a goal-completion-state provider that returns fixed tds."""
    calls = {"n": 0}
    def provider(state, side, enumerate_moves=False):
        calls["n"] += 1
        # Alternate before/after based on call order.
        # In tests we usually need before/after pairs.
        return {"total_goal_distance": td_before if calls["n"] % 2 == 1 else td_after}
    return provider


def test_observe_move_not_in_window_no_classify():
    # Trigger doesn't fire (score too high), tracker doesn't classify.
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(
        state_before=state, selected_move=(5, 5), ply=10, side_to_move="black",
        search_score=+0.20, root_top1_share=0.30,
    )
    snap = tracker.side_snapshot("black")
    assert snap["triggered"] is False
    assert snap["in_window_own_moves"] == 0


def test_observe_move_opens_window_on_trigger():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(
        state_before=state, selected_move=(5, 5), ply=44, side_to_move="black",
        search_score=-0.85, root_top1_share=0.12,
    )
    snap = tracker.side_snapshot("black")
    assert snap["triggered"] is True
    assert snap["first_trigger_ply"] == 44
    assert snap["first_trigger_reason"] == "steady_state"
    assert snap["in_window_own_moves"] == 1
    assert snap["triggered_own_moves"] == 1


def test_observe_move_window_stays_open_across_non_triggered_plies():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    # Open window:
    tracker.observe_move(state, (5, 5), 44, "black", -0.85, 0.12)
    # Non-triggered own move (score recovered, top1 still diffuse but score above threshold):
    tracker.observe_move(state, (6, 6), 46, "black", -0.20, 0.30)
    snap = tracker.side_snapshot("black")
    assert snap["in_window_own_moves"] == 2
    assert snap["triggered_own_moves"] == 1
    assert snap["non_triggered_in_window_moves"] == 1


def test_observe_move_missing_signal_in_window_counts_separately():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(state, (5, 5), 44, "black", -0.85, 0.12)
    tracker.observe_move(state, (6, 6), 46, "black", None, 0.20)
    snap = tracker.side_snapshot("black")
    assert snap["missing_signal_moves"] == 1
    assert snap["missing_search_score_moves"] == 1
    # Missing-signal plies don't get classified.
    assert sum(snap["selected_class_counts"].values()) == 1


def test_observe_move_other_side_does_not_affect_window():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(state, (5, 5), 44, "black", -0.85, 0.12)
    # Red's move is irrelevant to Black's window.
    tracker.observe_move(state, (6, 6), 45, "red", -0.85, 0.12)
    snap = tracker.side_snapshot("black")
    assert snap["in_window_own_moves"] == 1
    red_snap = tracker.side_snapshot("red")
    assert red_snap["triggered"] is True
    assert red_snap["in_window_own_moves"] == 1


def test_observe_move_does_not_mutate_state_before():
    """observe_move must not mutate state.pegs in any code path.
    Real TwixtState.apply_move returns a new state; we verify the stub here."""
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black", (1, 2): "black"})
    pegs_snapshot = dict(state.pegs)
    # Trigger fires (steady_state) so classification path runs.
    tracker.observe_move(
        state_before=state, selected_move=(5, 5), ply=44, side_to_move="black",
        search_score=-0.85, root_top1_share=0.12,
    )
    assert state.pegs == pegs_snapshot


def test_observe_move_in_window_includes_missing_signal_in_count():
    """in_window_own_moves counts every own-move after window opens,
    including missing-signal plies. classified_in_window_moves (at finalize)
    excludes missing-signal plies."""
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(state, (5, 5), 44, "black", -0.85, 0.12)   # opens window, triggered
    tracker.observe_move(state, (6, 6), 46, "black", None, 0.20)    # missing-signal in-window
    tracker.observe_move(state, (7, 7), 48, "black", -0.80, 0.10)   # triggered
    snap = tracker.side_snapshot("black")
    assert snap["in_window_own_moves"] == 3
    assert snap["missing_signal_moves"] == 1
    # selected_class_counts holds the classified subset (2 plies, not 3).
    assert sum(snap["selected_class_counts"].values()) == 2


def test_observe_move_sampled_entry_previous_score_is_pre_current():
    """The sampled entry's previous_own_search_score must reflect the score from
    the prior own-move, NOT the current score (which gets stored on the tracker
    only AFTER the entry is built)."""
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    # First triggered ply: previous_own_search_score is None.
    tracker.observe_move(state, (5, 5), 44, "black", -0.85, 0.12)
    # Second triggered ply: previous_own_search_score must be -0.85, NOT -0.99.
    tracker.observe_move(state, (6, 6), 46, "black", -0.99, 0.10)
    side_acc = tracker._sides["black"]
    entry_46 = next(e for e in side_acc.sampled_moves if e["ply"] == 46)
    assert entry_46["previous_own_search_score"] == -0.85
    assert entry_46["current_search_score"] == -0.99


def test_observe_move_disabled_via_config_is_no_op():
    """If config.enabled is False the tracker is not constructed by self_play
    in the first place. But ensure the tracker itself also no-ops if invoked
    despite enabled=False, so an integration bug doesn't silently corrupt state."""
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(enabled=False),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(state, (5, 5), 44, "black", -0.85, 0.12)
    snap = tracker.side_snapshot("black")
    assert snap["triggered"] is False
    rec = tracker.finalize_game(
        iteration=170, game_idx=0, game_id="game_000",
        winner="red", starting_player="red", n_moves=65, reason="win",
    )
    assert rec is None
```

- [ ] **Step 2: Run → fail**

```bash
.venv/bin/pytest tests/test_recovery_retargeting_diagnostics.py::test_observe_move_not_in_window_no_classify -v
```

Expected: ImportError on `RecoveryRetargetingTracker`.

- [ ] **Step 3: Implement the tracker**

Append to `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py`:

```python
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

    The tracker receives one observe_move call per ply with the side-to-move's
    score/share. It maintains per-side state (previous_own_search_score and
    accumulators) and emits the per-game record at finalize_game.

    gc_state_provider: callable(state, side, enumerate_moves=False) -> dict|None
        Matches the signature of connectivity_diagnostics.compute_goal_completion_state.
        Tests use a stub; production self_play passes the real helper.
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
        # Defensive: if a caller invokes a tracker whose config has enabled=False,
        # no-op rather than collecting data. Production code in self_play.py also
        # checks enabled before constructing the tracker, but this guard prevents
        # silent corruption if that check is ever bypassed.
        if not self.config.enabled:
            return

        side_acc = self._sides[side_to_move]
        opponent = "black" if side_to_move == "red" else "red"

        # Capture previous score BEFORE updating it, so trigger evaluation and
        # the sampled-entry record both see the pre-current value.
        prev_score = side_acc.previous_own_search_score

        trig = evaluate_trigger(
            current_search_score=search_score,
            root_top1_share=root_top1_share,
            previous_own_search_score=prev_score,
            config=self.config,
        )

        # Track missing-signal independently of in-window status.
        missing = trig["missing_search_score"] or trig["missing_root_top1_share"]

        # If side is not in-window and didn't just trigger, no further work.
        # Still update previous_own_search_score so future delta-precursor
        # evaluations see the current valid score.
        if not side_acc.triggered and not trig["triggered"]:
            if not trig["missing_search_score"]:
                side_acc.previous_own_search_score = search_score
            return

        # First-time trigger: open the window.
        if not side_acc.triggered and trig["triggered"]:
            side_acc.triggered = True
            side_acc.first_trigger_ply = ply
            side_acc.first_trigger_reason = trig["trigger_reason"]

        # Every own-move inside the window counts toward in_window_own_moves
        # (matches the spec name's literal meaning). The classified-vs-missing
        # split lives in classified_in_window_moves at finalize.
        side_acc.in_window_own_moves += 1

        if missing:
            side_acc.missing_signal_moves += 1
            if trig["missing_search_score"]:
                side_acc.missing_search_score_moves += 1
            if trig["missing_root_top1_share"]:
                side_acc.missing_root_top1_share_moves += 1
            # No classification on missing-signal plies. previous_own_search_score
            # is NOT updated (per spec §2.3 — only valid scores update it).
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

        # Compute state_after ONCE via the state's own apply_move. No mutation.
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

        # Sampled-moves recording. Capture in-window-own-move ordinal BEFORE
        # updating previous_own_search_score so the entry's prev_score reflects
        # the value used by the trigger.
        own_move_ordinal = side_acc.in_window_own_moves  # 1-based: this ply IS the Nth in-window own move.
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

        # Update previous_own_search_score AFTER the sampled entry is built.
        side_acc.previous_own_search_score = search_score

    def _maybe_record_sample(self, side_acc: _SideAccumulator, entry: dict) -> None:
        if self.config.sample_all_moves:
            side_acc.sampled_moves.append(entry)
            return
        cap = self.config.max_sampled_moves_per_side
        if cap <= 0:
            side_acc.sampled_moves_dropped += 1
            return
        # Priority 1 (highest): first 4 own-moves in the window — the inflection region.
        # Priority 2: severe-collapse plies.
        # Priority 3 (lowest): everything else, in window order.
        # Implemented as: insert in window order; if over cap, drop lowest-priority entries.
        side_acc.sampled_moves.append(entry)
        if len(side_acc.sampled_moves) > cap:
            def _priority(e):
                # 0 = highest priority; 2 = lowest.
                if e.get("in_window_own_move_index", 10**9) <= 4:
                    return 0
                if e["is_severe_collapse"]:
                    return 1
                return 2
            # Find the rightmost (latest-ply) lowest-priority entry; drop it.
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
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_recovery_retargeting_diagnostics.py -v
```

Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/recovery_retargeting_diagnostics.py tests/test_recovery_retargeting_diagnostics.py
git commit -m "feat(recovery_retargeting): tracker.observe_move (window state + classifier integration, Spec 4 §5)"
```

---

## Task 6: Implement `RecoveryRetargetingTracker.finalize_game`

**Files:**
- Modify: `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py`
- Test: `tests/test_recovery_retargeting_diagnostics.py`

- [ ] **Step 1: Write finalize tests**

Append to `tests/test_recovery_retargeting_diagnostics.py`:

```python
def test_finalize_returns_none_when_no_side_triggered():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(state, (5, 5), 10, "black", +0.20, 0.30)
    rec = tracker.finalize_game(
        iteration=0, game_idx=0, game_id="game_000",
        winner="red", starting_player="red", n_moves=65, reason="win",
    )
    assert rec is None


def test_finalize_emits_record_when_one_side_triggered():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(state, (5, 5), 44, "black", -0.85, 0.12)
    rec = tracker.finalize_game(
        iteration=170, game_idx=22, game_id="game_022",
        winner="red", starting_player="red", n_moves=65, reason="win",
    )
    assert rec is not None
    assert rec["version"] == 1
    assert rec["iteration"] == 170
    assert rec["game_idx"] == 22
    assert rec["game_id"] == "game_022"
    assert rec["winner"] == "red"
    assert rec["loser"] == "black"
    assert rec["triggered_sides"] == ["black"]
    assert rec["first_trigger_ply"] == 44
    assert rec["first_trigger_side"] == "black"
    assert rec["first_trigger_reason"] == "steady_state"
    black_rec = rec["side_records"]["black"]
    assert black_rec["triggered"] is True
    assert black_rec["classified_in_window_moves"] == 1
    # Rollups partition the denominator.
    rollup_sum = (
        black_rec["constructive_recovery_moves"]
        + black_rec["defensive_moves"]
        + black_rec["structural_connection_moves"]
        + black_rec["local_drift_moves"]
    )
    assert rollup_sum == black_rec["classified_in_window_moves"]


def test_finalize_loser_is_none_on_draw():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(state, (5, 5), 44, "black", -0.85, 0.12)
    rec = tracker.finalize_game(
        iteration=170, game_idx=22, game_id="game_022",
        winner=None, starting_player="red", n_moves=65, reason="board_full",
    )
    assert rec["loser"] is None


def test_finalize_includes_config_block():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    tracker.observe_move(state, (5, 5), 44, "black", -0.85, 0.12)
    rec = tracker.finalize_game(
        iteration=170, game_idx=22, game_id="game_022",
        winner="red", starting_player="red", n_moves=65, reason="win",
    )
    cfg = rec["config"]
    assert cfg["collapse_value_threshold"] == -0.75
    assert cfg["classify_defense"] is True


def test_finalize_sampled_moves_metadata():
    tracker = RecoveryRetargetingTracker(
        config=RecoveryRetargetingConfig(max_sampled_moves_per_side=2),
        gc_state_provider=lambda *a, **kw: {"total_goal_distance": 5},
    )
    state = _StubState({(0, 0): "black"})
    for ply, mv in [(44, (5, 5)), (46, (6, 6)), (48, (7, 7)), (50, (8, 8))]:
        tracker.observe_move(state, mv, ply, "black", -0.85, 0.12)
    rec = tracker.finalize_game(
        iteration=170, game_idx=22, game_id="game_022",
        winner="red", starting_player="red", n_moves=65, reason="win",
    )
    black_rec = rec["side_records"]["black"]
    assert black_rec["sampled_moves_count"] == 2
    assert black_rec["sampled_moves_cap"] == 2
    assert black_rec["sampled_moves_dropped"] == 2
```

- [ ] **Step 2: Run → fail**

```bash
.venv/bin/pytest tests/test_recovery_retargeting_diagnostics.py::test_finalize_returns_none_when_no_side_triggered -v
```

Expected: AttributeError (`finalize_game` doesn't exist).

- [ ] **Step 3: Implement finalize_game**

Append to `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py`:

```python
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

        # First-trigger metadata across both sides (by ply).
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
            if not a.triggered:
                side_records[side] = {"triggered": False, "classifier_error_count": a.classifier_error_count}
                total_classifier_errors += a.classifier_error_count
                continue
            classified = sum(a.selected_class_counts.values())
            counts = a.selected_class_counts
            constructive = counts["reduces_own_goal_distance"] + counts["starts_or_extends_alternate_component"]
            defensive = counts["blocks_opponent_closeout"]
            structural = counts["connects_to_existing_component"] + counts["improves_own_largest_component"]
            local_drift = counts["redundant_local_reinforcement"] + counts["off_plan_or_unclear"]
            denom = classified if classified > 0 else 1
            side_records[side] = {
                "triggered":            True,
                "first_trigger_ply":    a.first_trigger_ply,
                "first_trigger_reason": a.first_trigger_reason,
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

                "mean_search_score_triggered_plies":   round(sum(a.triggered_scores) / len(a.triggered_scores), 3) if a.triggered_scores else None,
                "min_search_score_triggered_plies":    round(min(a.triggered_scores), 3) if a.triggered_scores else None,
                "max_search_score_triggered_plies":    round(max(a.triggered_scores), 3) if a.triggered_scores else None,
                "mean_root_top1_share_triggered_plies": round(sum(a.triggered_top1_shares) / len(a.triggered_top1_shares), 3) if a.triggered_top1_shares else None,

                "classified_in_window_moves": classified,
                "selected_class_counts":      dict(a.selected_class_counts),

                "constructive_recovery_moves":  constructive,
                "defensive_moves":              defensive,
                "structural_connection_moves":  structural,
                "local_drift_moves":            local_drift,

                "constructive_recovery_rate":  round(constructive / denom, 3),
                "defensive_rate":              round(defensive / denom, 3),
                "structural_connection_rate":  round(structural / denom, 3),
                "local_drift_rate":            round(local_drift / denom, 3),

                "sampled_moves_count":   len(a.sampled_moves),
                "sampled_moves_cap":     self.config.max_sampled_moves_per_side,
                "sampled_moves_dropped": a.sampled_moves_dropped,
                "sample_all_moves":      self.config.sample_all_moves,
                "sampled_moves":         list(a.sampled_moves),
            }
            total_classifier_errors += a.classifier_error_count

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
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_recovery_retargeting_diagnostics.py -v
```

Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/recovery_retargeting_diagnostics.py tests/test_recovery_retargeting_diagnostics.py
git commit -m "feat(recovery_retargeting): finalize_game emits per-game record (Spec 4 §4)"
```

---

# Phase 3 — Aggregator + analyzer outputs

## Task 7: Implement `aggregate_recovery_retargeting_records`

**Files:**
- Modify: `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py`
- Test: `tests/test_recovery_retargeting_diagnostics.py`

- [ ] **Step 1: Write aggregator tests**

Append to `tests/test_recovery_retargeting_diagnostics.py`:

```python
from scripts.GPU.alphazero.recovery_retargeting_diagnostics import (
    aggregate_recovery_retargeting_records,
)


def _record(side="black", classified=10, classes=None, in_window=10, triggered=8, severe=4, very_diffuse=6):
    classes = classes or {"redundant_local_reinforcement": classified}
    counts = {c: 0 for c in PRIMARY_CLASSES}
    counts.update(classes)
    return {
        "version": 1,
        "iteration": 170, "game_idx": 0, "game_id": "game_000",
        "winner": "red", "loser": side,
        "triggered_sides": [side],
        "side_records": {
            "red": {"triggered": False, "classifier_error_count": 0} if side != "red" else None,
            "black": {"triggered": False, "classifier_error_count": 0} if side != "black" else None,
            side: {
                "triggered": True,
                "in_window_own_moves": in_window,
                "triggered_own_moves": triggered,
                "non_triggered_in_window_moves": in_window - triggered,
                "missing_signal_moves": 0,
                "severe_collapse_moves": severe,
                "very_diffuse_moves": very_diffuse,
                "trigger_reason_counts": {"delta_precursor": 1, "steady_state": triggered - 1, "both": 0},
                "classified_in_window_moves": classified,
                "selected_class_counts": counts,
                "constructive_recovery_moves": counts.get("reduces_own_goal_distance", 0) + counts.get("starts_or_extends_alternate_component", 0),
                "defensive_moves": counts.get("blocks_opponent_closeout", 0),
                "structural_connection_moves": counts.get("connects_to_existing_component", 0) + counts.get("improves_own_largest_component", 0),
                "local_drift_moves": counts.get("redundant_local_reinforcement", 0) + counts.get("off_plan_or_unclear", 0),
                "classifier_error_count": 0,
            },
        },
        "classifier_error_count": 0,
        "config": {
            "collapse_value_threshold": -0.75,
            "severe_collapse_value_threshold": -0.90,
            "diffuse_root_top1_threshold": 0.20,
            "very_diffuse_root_top1_threshold": 0.15,
            "delta_threshold": 0.50,
            "delta_max_current_score": -0.30,
            "alternate_component_min_size": 4,
            "classify_defense": True,
        },
    }


def test_aggregator_sums_counts_and_recomputes_rates():
    recs = [_record(), _record()]
    s = aggregate_recovery_retargeting_records(recs, games_total=100)
    assert s["version"] == 1
    assert s["games_total"] == 100
    assert s["games_triggered"] == 2
    assert s["triggered_own_moves_total"] == 16
    assert s["in_window_own_moves_total"] == 20
    assert s["selected_class_counts_total"]["redundant_local_reinforcement"] == 20
    # All weight in redundant_local_reinforcement; local_drift_rate must be 1.0
    assert s["local_drift_rate"] == 1.0


def test_aggregator_returns_empty_summary_when_no_records():
    s = aggregate_recovery_retargeting_records([], games_total=100)
    assert s["games_total"] == 100
    assert s["games_triggered"] == 0
    assert s["trigger_rate"] == 0.0


def test_aggregator_empty_records_emits_enabled_summary_with_zero_trigger_rate():
    """5-game smoke run where no game triggers must still produce a well-formed
    sidecar block (the analyzer's report relies on the block existing)."""
    s = aggregate_recovery_retargeting_records([], games_total=5)
    assert s["version"] == 1
    assert s["enabled"] is True
    assert s["games_total"] == 5
    assert s["games_triggered"] == 0
    assert s["trigger_rate"] == 0.0
    assert s["in_window_own_moves_total"] == 0
    assert s["classified_in_window_moves_total"] == 0
    # Schema-integrity block is always present.
    assert s["schema_integrity"]["classifier_error_count_total"] == 0


def test_aggregator_skips_unknown_version():
    rec = _record()
    rec["version"] = 99
    s = aggregate_recovery_retargeting_records([_record(), rec], games_total=100)
    assert s["games_triggered"] == 1
    assert s["schema_integrity"]["skipped_unknown_version_count"] == 1


def test_aggregator_skips_config_mismatch():
    a = _record()
    b = _record()
    b["config"]["collapse_value_threshold"] = -0.50
    s = aggregate_recovery_retargeting_records([a, b], games_total=100)
    assert s["games_triggered"] == 1
    assert s["schema_integrity"]["skipped_config_mismatch_count"] == 1
```

- [ ] **Step 2: Run → fail**

```bash
.venv/bin/pytest tests/test_recovery_retargeting_diagnostics.py::test_aggregator_sums_counts_and_recomputes_rates -v
```

Expected: ImportError on `aggregate_recovery_retargeting_records`.

- [ ] **Step 3: Implement the aggregator**

Append to `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py`:

```python
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
    expected_config: Optional[dict] = None,
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
    canonical_config = expected_config

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
    constructive = selected_class_totals["reduces_own_goal_distance"] + selected_class_totals["starts_or_extends_alternate_component"]
    defensive = selected_class_totals["blocks_opponent_closeout"]
    structural = selected_class_totals["connects_to_existing_component"] + selected_class_totals["improves_own_largest_component"]
    local_drift = selected_class_totals["redundant_local_reinforcement"] + selected_class_totals["off_plan_or_unclear"]

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
        "constructive_recovery_rate": round(constructive / denom, 3),
        "defensive_rate": round(defensive / denom, 3),
        "structural_connection_rate": round(structural / denom, 3),
        "local_drift_rate": round(local_drift / denom, 3),
        "schema_integrity": {
            "skipped_unknown_version_count": skipped_unknown_version,
            "skipped_config_mismatch_count": skipped_config_mismatch,
            "classifier_error_count_total": classifier_error_total,
        },
    }
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_recovery_retargeting_diagnostics.py -v
```

Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/recovery_retargeting_diagnostics.py tests/test_recovery_retargeting_diagnostics.py
git commit -m "feat(recovery_retargeting): per-iter aggregator with schema-integrity counters (Spec 4 §6.1-6.3)"
```

---

## Task 8: Analyzer report formatter

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py`
- Test: `tests/test_analyzer_recovery_retargeting.py`

- [ ] **Step 1: Write the formatter test**

Create `tests/test_analyzer_recovery_retargeting.py`:

```python
"""Tests for Spec 4 analyzer surface."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.twixt_replay_analyzer import format_recovery_retargeting_report


def _summary(**overrides):
    base = {
        "version": 1,
        "enabled": True,
        "config": {
            "collapse_value_threshold": -0.75,
            "severe_collapse_value_threshold": -0.90,
            "diffuse_root_top1_threshold": 0.20,
            "very_diffuse_root_top1_threshold": 0.15,
            "delta_threshold": 0.50,
            "delta_max_current_score": -0.30,
            "alternate_component_min_size": 4,
            "classify_defense": True,
        },
        "games_total": 1000, "games_triggered": 143,
        "trigger_rate": 0.143,
        "triggered_loser_side": 136, "triggered_winner_side": 9,
        "triggered_loser_side_per_triggered_game": 0.951,
        "triggered_winner_side_per_triggered_game": 0.063,
        "in_window_own_moves_total": 1284,
        "triggered_own_moves_total": 1108,
        "non_triggered_in_window_moves_total": 176,
        "missing_signal_moves_total": 0,
        "severe_collapse_moves_total": 522,
        "very_diffuse_moves_total": 914,
        "trigger_reason_counts_total": {"delta_precursor": 177, "steady_state": 859, "both": 72},
        "classified_in_window_moves_total": 1284,
        "selected_class_counts_total": {
            "blocks_opponent_closeout": 104, "reduces_own_goal_distance": 55,
            "starts_or_extends_alternate_component": 41,
            "connects_to_existing_component": 231, "improves_own_largest_component": 159,
            "redundant_local_reinforcement": 548, "off_plan_or_unclear": 146,
        },
        "constructive_recovery_rate": 0.075,
        "defensive_rate": 0.081,
        "structural_connection_rate": 0.304,
        "local_drift_rate": 0.540,
        "schema_integrity": {
            "skipped_unknown_version_count": 0,
            "skipped_config_mismatch_count": 0,
            "classifier_error_count_total": 0,
        },
        "iters_covered": [170, 179],
    }
    base.update(overrides)
    return base


def test_format_emits_section_header_and_key_lines():
    lines = format_recovery_retargeting_report(_summary())
    body = "\n".join(lines)
    assert "Recovery / Re-targeting Diagnostics" in body
    assert "Triggered games:" in body
    assert "constructive recovery:" in body
    assert "local drift / unclear:" in body


def test_format_warns_when_classify_defense_off():
    s = _summary()
    s["config"] = dict(s["config"])
    s["config"]["classify_defense"] = False
    body = "\n".join(format_recovery_retargeting_report(s))
    assert "defense classification disabled" in body


def test_format_returns_empty_when_summary_is_none_or_empty():
    assert format_recovery_retargeting_report(None) == []
    assert format_recovery_retargeting_report({}) == []
```

- [ ] **Step 2: Run → fail**

```bash
.venv/bin/pytest tests/test_analyzer_recovery_retargeting.py -v
```

Expected: ImportError on `format_recovery_retargeting_report`.

- [ ] **Step 3: Implement the formatter in `scripts/twixt_replay_analyzer.py`**

Add (immediately after `format_closeout_selection_tiebreak_report`, near line ~2461):

```python
def format_recovery_retargeting_report(summary: Optional[dict]) -> list:
    """Format the recovery / re-targeting telemetry section. Spec 4 §6.5."""
    if not summary:
        return []
    cfg = summary.get("config") or {}
    classify_defense_on = bool(cfg.get("classify_defense", True))

    def _pct(x):
        return f"{(x or 0.0) * 100.0:.1f}%"

    lines = []
    lines.append("Recovery / Re-targeting Diagnostics")
    lines.append("===================================")
    iters = summary.get("iters_covered") or []
    if iters:
        lines.append(
            f"Iters covered: {min(iters)}-{max(iters)}  enabled={summary.get('enabled')}  "
            f"defense_classifier={'on' if classify_defense_on else 'off'}"
        )
    lines.append(
        f"Config: collapse_value<={cfg.get('collapse_value_threshold')}  "
        f"diffuse_root_top1<={cfg.get('diffuse_root_top1_threshold')}  "
        f"delta>={cfg.get('delta_threshold')} with current<={cfg.get('delta_max_current_score')}"
    )
    games_total = summary.get("games_total", 0)
    games_triggered = summary.get("games_triggered", 0)
    lines.append(f"Triggered games:           {games_triggered} / {games_total} ({_pct(summary.get('trigger_rate'))})")
    lines.append(f"  side was eventual loser: {summary.get('triggered_loser_side', 0)} / {games_triggered} ({_pct(summary.get('triggered_loser_side_per_triggered_game'))})")
    lines.append(f"  side was eventual winner:{summary.get('triggered_winner_side', 0):4d} / {games_triggered} ({_pct(summary.get('triggered_winner_side_per_triggered_game'))})")
    in_window = summary.get("in_window_own_moves_total", 0)
    lines.append(f"In-window own moves:       {in_window}")
    lines.append(f"  triggered:               {summary.get('triggered_own_moves_total', 0)}")
    lines.append(f"  non-triggered in-window: {summary.get('non_triggered_in_window_moves_total', 0)}")
    lines.append(f"  missing-signal:          {summary.get('missing_signal_moves_total', 0)}")
    lines.append(f"Severity:")
    lines.append(f"  severe collapse:         {summary.get('severe_collapse_moves_total', 0)} plies")
    lines.append(f"  very diffuse root:       {summary.get('very_diffuse_moves_total', 0)} plies")
    trc = summary.get("trigger_reason_counts_total") or {}
    lines.append("Trigger composition:")
    lines.append(f"  delta_precursor:         {trc.get('delta_precursor', 0)}")
    lines.append(f"  steady_state:            {trc.get('steady_state', 0)}")
    lines.append(f"  both:                    {trc.get('both', 0)}")
    classified = summary.get("classified_in_window_moves_total", 0)
    counts = summary.get("selected_class_counts_total") or {}
    rates = summary.get("selected_class_rates_total") or {}
    lines.append("Move-class composition (denominator: classified in-window):")
    for cls, label in (
        ("blocks_opponent_closeout",              "blocks opponent closeout:"),
        ("reduces_own_goal_distance",             "reduces own goal distance:"),
        ("starts_or_extends_alternate_component", "starts/extends alternate component:"),
        ("connects_to_existing_component",        "connects to existing component:"),
        ("improves_own_largest_component",        "improves own largest component:"),
        ("redundant_local_reinforcement",         "redundant local reinforcement:"),
        ("off_plan_or_unclear",                   "off-plan or unclear:"),
    ):
        lines.append(f"  {label:42s} {_pct(rates.get(cls)):>6s}   ({counts.get(cls, 0)})")
    lines.append("Rollup:")
    lines.append(f"  constructive recovery:                 {_pct(summary.get('constructive_recovery_rate'))}")
    if classify_defense_on:
        lines.append(f"  defense:                               {_pct(summary.get('defensive_rate'))}")
    else:
        lines.append(f"  defense:                  N/A (defense classification disabled — local drift may include defensive moves)")
    lines.append(f"  structural connection:                 {_pct(summary.get('structural_connection_rate'))}")
    lines.append(f"  local drift / unclear:                 {_pct(summary.get('local_drift_rate'))}")
    si = summary.get("schema_integrity") or {}
    lines.append("Schema integrity:")
    lines.append(f"  classifier_error_count:                {si.get('classifier_error_count_total', 0)}")
    lines.append(f"  records skipped (unknown version):     {si.get('skipped_unknown_version_count', 0)}")
    lines.append(f"  records skipped (config mismatch):     {si.get('skipped_config_mismatch_count', 0)}")
    lines.append("Worst cases: recovery_retargeting_worst_cases.csv")
    return lines
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_analyzer_recovery_retargeting.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_recovery_retargeting.py
git commit -m "feat(analyzer): recovery / re-targeting report formatter (Spec 4 §6.5)"
```

---

## Task 9: `recovery_retargeting_by_iter.csv` writer

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py`
- Test: `tests/test_analyzer_recovery_retargeting.py`

- [ ] **Step 1: Write CSV test**

Append to `tests/test_analyzer_recovery_retargeting.py`:

```python
import csv

from scripts.twixt_replay_analyzer import write_recovery_retargeting_by_iter_csv


def test_by_iter_csv_one_row_per_iter(tmp_path):
    per_iter = {
        170: _summary(games_total=100, games_triggered=14),
        171: _summary(games_total=100, games_triggered=20),
    }
    out = tmp_path / "recovery_retargeting_by_iter.csv"
    write_recovery_retargeting_by_iter_csv(str(out), per_iter)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2
    assert int(rows[0]["iteration"]) == 170
    assert int(rows[0]["games_triggered"]) == 14
    assert int(rows[1]["games_triggered"]) == 20
    assert "local_drift_rate" in rows[0]
```

- [ ] **Step 2: Run → fail**

```bash
.venv/bin/pytest tests/test_analyzer_recovery_retargeting.py::test_by_iter_csv_one_row_per_iter -v
```

Expected: ImportError on `write_recovery_retargeting_by_iter_csv`.

- [ ] **Step 3: Implement the by-iter writer**

Add to `scripts/twixt_replay_analyzer.py` (after the report formatter):

```python
def write_recovery_retargeting_by_iter_csv(out_path: str, per_iter_summaries: dict) -> str:
    """Write recovery_retargeting_by_iter.csv. Spec 4 §6.6.

    per_iter_summaries: dict mapping iteration -> per-iter summary dict (output
    of aggregate_recovery_retargeting_records).
    """
    import csv
    fields = [
        "iteration",
        "games_total", "games_triggered", "trigger_rate",
        "triggered_loser_side", "triggered_winner_side",
        "triggered_loser_side_per_triggered_game",
        "in_window_own_moves_total", "triggered_own_moves_total",
        "severe_collapse_moves_total", "very_diffuse_moves_total",
        "classified_in_window_moves_total", "classifier_error_count_total",
        "constructive_recovery_rate", "defensive_rate",
        "structural_connection_rate", "local_drift_rate",
        "redundant_local_reinforcement_rate", "off_plan_or_unclear_rate",
        "trigger_delta_precursor_count", "trigger_steady_state_count", "trigger_both_count",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for it in sorted(per_iter_summaries.keys()):
            s = per_iter_summaries[it] or {}
            rates = s.get("selected_class_rates_total") or {}
            trc = s.get("trigger_reason_counts_total") or {}
            si = s.get("schema_integrity") or {}
            w.writerow({
                "iteration": it,
                "games_total": s.get("games_total", 0),
                "games_triggered": s.get("games_triggered", 0),
                "trigger_rate": s.get("trigger_rate", 0.0),
                "triggered_loser_side": s.get("triggered_loser_side", 0),
                "triggered_winner_side": s.get("triggered_winner_side", 0),
                "triggered_loser_side_per_triggered_game": s.get("triggered_loser_side_per_triggered_game", 0.0),
                "in_window_own_moves_total": s.get("in_window_own_moves_total", 0),
                "triggered_own_moves_total": s.get("triggered_own_moves_total", 0),
                "severe_collapse_moves_total": s.get("severe_collapse_moves_total", 0),
                "very_diffuse_moves_total": s.get("very_diffuse_moves_total", 0),
                "classified_in_window_moves_total": s.get("classified_in_window_moves_total", 0),
                "classifier_error_count_total": si.get("classifier_error_count_total", 0),
                "constructive_recovery_rate": s.get("constructive_recovery_rate", 0.0),
                "defensive_rate": s.get("defensive_rate", 0.0),
                "structural_connection_rate": s.get("structural_connection_rate", 0.0),
                "local_drift_rate": s.get("local_drift_rate", 0.0),
                "redundant_local_reinforcement_rate": rates.get("redundant_local_reinforcement", 0.0),
                "off_plan_or_unclear_rate": rates.get("off_plan_or_unclear", 0.0),
                "trigger_delta_precursor_count": trc.get("delta_precursor", 0),
                "trigger_steady_state_count": trc.get("steady_state", 0),
                "trigger_both_count": trc.get("both", 0),
            })
    return out_path
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_analyzer_recovery_retargeting.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_recovery_retargeting.py
git commit -m "feat(analyzer): recovery_retargeting_by_iter.csv writer (Spec 4 §6.6)"
```

---

## Task 10: `recovery_retargeting_worst_cases.csv` writer

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py`
- Test: `tests/test_analyzer_recovery_retargeting.py`

- [ ] **Step 1: Write worst-cases tests**

Append to `tests/test_analyzer_recovery_retargeting.py`:

```python
from scripts.twixt_replay_analyzer import write_recovery_retargeting_worst_cases_csv


def _per_game_rec(iteration, game_idx, sides_triggered, local_drift_moves, in_window):
    side_records = {"red": {"triggered": False}, "black": {"triggered": False}}
    for side in sides_triggered:
        side_records[side] = {
            "triggered": True,
            "first_trigger_ply": 44, "first_trigger_reason": "steady_state",
            "in_window_own_moves": in_window, "triggered_own_moves": in_window,
            "severe_collapse_moves": 0, "very_diffuse_moves": 0,
            "classified_in_window_moves": in_window, "missing_signal_moves": 0,
            "selected_class_counts": {
                "blocks_opponent_closeout": 0, "reduces_own_goal_distance": 0,
                "starts_or_extends_alternate_component": 0,
                "connects_to_existing_component": 0, "improves_own_largest_component": 0,
                "redundant_local_reinforcement": local_drift_moves,
                "off_plan_or_unclear": 0,
            },
            "constructive_recovery_moves": 0, "defensive_moves": 0,
            "structural_connection_moves": 0, "local_drift_moves": local_drift_moves,
            "local_drift_rate": 1.0, "constructive_recovery_rate": 0.0,
            "mean_search_score_triggered_plies": -0.85,
            "min_search_score_triggered_plies": -0.99,
            "max_search_score_triggered_plies": -0.75,
            "mean_root_top1_share_triggered_plies": 0.12,
        }
    return {
        "iteration": iteration, "game_idx": game_idx, "game_id": f"game_{game_idx:03d}",
        "winner": "red", "loser": "black", "n_moves": 65, "reason": "win",
        "triggered_sides": sides_triggered, "side_records": side_records,
    }


def test_worst_cases_csv_sort_order_and_topk(tmp_path):
    out = tmp_path / "recovery_retargeting_worst_cases.csv"
    records = [
        _per_game_rec(170, 0, ["black"], local_drift_moves=2, in_window=2),
        _per_game_rec(170, 1, ["black"], local_drift_moves=15, in_window=15),
        _per_game_rec(170, 2, ["black"], local_drift_moves=8, in_window=8),
    ]
    write_recovery_retargeting_worst_cases_csv(str(out), records, top_k=2)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2
    # Sort: local_drift_moves DESC. Top two are 15, then 8.
    assert int(rows[0]["local_drift_moves"]) == 15
    assert int(rows[1]["local_drift_moves"]) == 8


def test_worst_cases_csv_two_rows_for_dual_triggered_game(tmp_path):
    out = tmp_path / "recovery_retargeting_worst_cases.csv"
    records = [_per_game_rec(170, 0, ["black", "red"], local_drift_moves=5, in_window=5)]
    write_recovery_retargeting_worst_cases_csv(str(out), records, top_k=25)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2
    sides = sorted(r["triggered_side"] for r in rows)
    assert sides == ["black", "red"]
```

- [ ] **Step 2: Run → fail**

```bash
.venv/bin/pytest tests/test_analyzer_recovery_retargeting.py::test_worst_cases_csv_sort_order_and_topk -v
```

Expected: ImportError on `write_recovery_retargeting_worst_cases_csv`.

- [ ] **Step 3: Implement the worst-cases writer**

Append to `scripts/twixt_replay_analyzer.py` (after the by-iter writer):

```python
def write_recovery_retargeting_worst_cases_csv(
    out_path: str, records: list, *, top_k: int = 25,
) -> str:
    """Write recovery_retargeting_worst_cases.csv. Spec 4 §6.7.

    One row per triggered side; sorted by (local_drift_moves DESC,
    in_window_own_moves DESC, min_search_score_triggered_plies ASC).
    """
    import csv
    rows = []
    for rec in records:
        if not rec:
            continue
        for side in rec.get("triggered_sides") or []:
            sr = (rec.get("side_records") or {}).get(side) or {}
            counts = sr.get("selected_class_counts") or {}
            rows.append({
                "iteration": rec.get("iteration"),
                "game_idx": rec.get("game_idx"),
                "game_id": rec.get("game_id"),
                "winner": rec.get("winner"),
                "loser": rec.get("loser"),
                "reason": rec.get("reason"),
                "n_moves": rec.get("n_moves"),
                "triggered_side": side,
                "first_trigger_ply": sr.get("first_trigger_ply"),
                "first_trigger_reason": sr.get("first_trigger_reason"),
                "in_window_own_moves": sr.get("in_window_own_moves", 0),
                "triggered_own_moves": sr.get("triggered_own_moves", 0),
                "severe_collapse_moves": sr.get("severe_collapse_moves", 0),
                "very_diffuse_moves": sr.get("very_diffuse_moves", 0),
                "classified_in_window_moves": sr.get("classified_in_window_moves", 0),
                "missing_signal_moves": sr.get("missing_signal_moves", 0),
                "blocks_opponent_closeout_moves":              counts.get("blocks_opponent_closeout", 0),
                "reduces_own_goal_distance_moves":             counts.get("reduces_own_goal_distance", 0),
                "starts_or_extends_alternate_component_moves": counts.get("starts_or_extends_alternate_component", 0),
                "connects_to_existing_component_moves":        counts.get("connects_to_existing_component", 0),
                "improves_own_largest_component_moves":        counts.get("improves_own_largest_component", 0),
                "redundant_local_reinforcement_moves":         counts.get("redundant_local_reinforcement", 0),
                "off_plan_or_unclear_moves":                   counts.get("off_plan_or_unclear", 0),
                "constructive_recovery_moves": sr.get("constructive_recovery_moves", 0),
                "defensive_moves":             sr.get("defensive_moves", 0),
                "structural_connection_moves": sr.get("structural_connection_moves", 0),
                "local_drift_moves":           sr.get("local_drift_moves", 0),
                "local_drift_rate":            sr.get("local_drift_rate", 0.0),
                "constructive_recovery_rate":  sr.get("constructive_recovery_rate", 0.0),
                "mean_search_score_triggered_plies": sr.get("mean_search_score_triggered_plies"),
                "min_search_score_triggered_plies":  sr.get("min_search_score_triggered_plies"),
                "max_search_score_triggered_plies":  sr.get("max_search_score_triggered_plies"),
                "mean_root_top1_share_triggered_plies": sr.get("mean_root_top1_share_triggered_plies"),
            })
    rows.sort(
        key=lambda r: (
            -int(r["local_drift_moves"] or 0),
            -int(r["in_window_own_moves"] or 0),
            float(r["min_search_score_triggered_plies"]) if r.get("min_search_score_triggered_plies") is not None else 0.0,
        )
    )
    rows = rows[:max(0, int(top_k))]
    fields = list(rows[0].keys()) if rows else [
        "iteration","game_idx","game_id","winner","loser","reason","n_moves",
        "triggered_side","first_trigger_ply","first_trigger_reason",
        "in_window_own_moves","triggered_own_moves",
        "severe_collapse_moves","very_diffuse_moves",
        "classified_in_window_moves","missing_signal_moves",
        "blocks_opponent_closeout_moves","reduces_own_goal_distance_moves",
        "starts_or_extends_alternate_component_moves",
        "connects_to_existing_component_moves","improves_own_largest_component_moves",
        "redundant_local_reinforcement_moves","off_plan_or_unclear_moves",
        "constructive_recovery_moves","defensive_moves",
        "structural_connection_moves","local_drift_moves",
        "local_drift_rate","constructive_recovery_rate",
        "mean_search_score_triggered_plies","min_search_score_triggered_plies",
        "max_search_score_triggered_plies","mean_root_top1_share_triggered_plies",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out_path
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_analyzer_recovery_retargeting.py -v
```

Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_recovery_retargeting.py
git commit -m "feat(analyzer): recovery_retargeting_worst_cases.csv with two-row split for dual-triggered games (Spec 4 §6.7)"
```

---

## Task 11: Wire analyzer pieces into `analyze()`

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py`

- [ ] **Step 1: Find the wiring site for Fix 2**

```bash
grep -n "aggregate_closeout_selection_tiebreak\|format_closeout_selection_tiebreak_report\|recovery_retargeting" scripts/twixt_replay_analyzer.py | head -10
```

Locate the block (around line 4759-4764) where the Fix 2 telemetry is wired into `analyze()`.

- [ ] **Step 2: Add the analyzer-side CLI flag**

Find the analyzer argparse section (search for `argparse.ArgumentParser`) and add immediately after the closeout-related flags:

```python
    parser.add_argument(
        "--recovery-retargeting-worst-cases-top-k",
        type=int, default=25,
        help="Max rows in recovery_retargeting_worst_cases CSV (Spec 4)",
    )
```

- [ ] **Step 3: Wire into `analyze()` after the Fix 2 block**

In `analyze()`, immediately after the Fix 2 sidecar/report block (around line 4764), add:

```python
    # Spec 4 — recovery / re-targeting diagnostic.
    from scripts.GPU.alphazero.recovery_retargeting_diagnostics import (
        aggregate_recovery_retargeting_records,
    )

    # Per-iter summaries: read from sidecars; analyzer cross-iter rollup
    # tolerates config drift (warns, does not skip) per spec §6.4.
    per_iter_rr = {}
    for it, sc in (relevant_sidecars or {}).items():
        block = (sc or {}).get("recovery_retargeting_summary")
        if isinstance(block, dict):
            per_iter_rr[it] = block

    # Cross-iter rollup: build a synthetic summary by re-aggregating from
    # per-game records (collected by analyze() upstream as `replays`).
    rr_records = [r.get("recovery_retargeting_record") for r in replays
                  if isinstance(r, dict) and r.get("recovery_retargeting_record")]
    rr_games_total = len(replays) if replays else 0
    rr_summary = aggregate_recovery_retargeting_records(
        rr_records, games_total=rr_games_total,
    )
    rr_summary["iters_covered"] = sorted(per_iter_rr.keys())

    # Detect mixed configs across iterations (warn, do not skip — spec §6.4).
    iter_configs = {it: (sc or {}).get("config") for it, sc in per_iter_rr.items()}
    distinct_configs = []
    for cfg in iter_configs.values():
        if cfg and cfg not in distinct_configs:
            distinct_configs.append(cfg)
    rr_summary["mixed_config_across_iters"] = len(distinct_configs) > 1

    if rr_summary.get("games_total"):
        lines.extend([""])
        lines.extend(format_recovery_retargeting_report(rr_summary))
        if rr_summary.get("mixed_config_across_iters"):
            lines.append("")
            lines.append(f"Mixed config across iters covered ({len(distinct_configs)} distinct configs).")
            lines.append(f"WARNING: rates aggregate across config changes; treat with care.")
        summary["recovery_retargeting"] = rr_summary

        # CSVs (analyzer §6.6 + §6.7).
        from scripts.twixt_replay_analyzer import (  # local import to avoid circular ref
            write_recovery_retargeting_by_iter_csv,
            write_recovery_retargeting_worst_cases_csv,
        )
        rr_by_iter_path = os.path.join(out_dir, _suffixed("recovery_retargeting_by_iter", "csv", suffix))
        write_recovery_retargeting_by_iter_csv(rr_by_iter_path, per_iter_rr)
        rr_worst_path = os.path.join(out_dir, _suffixed("recovery_retargeting_worst_cases", "csv", suffix))
        write_recovery_retargeting_worst_cases_csv(
            rr_worst_path, rr_records, top_k=args.recovery_retargeting_worst_cases_top_k,
        )
```

NOTE: The `args.recovery_retargeting_worst_cases_top_k` reference assumes `analyze()` has access to argparse args. If it does not, thread the value through as a function parameter with default 25.

- [ ] **Step 4: Smoke test the analyzer help**

```bash
.venv/bin/python scripts/twixt_replay_analyzer.py --help | grep recovery-retargeting
```

Expected: at least `--recovery-retargeting-worst-cases-top-k` listed.

- [ ] **Step 5: Run the full analyzer test suite to confirm no regression**

```bash
.venv/bin/pytest tests/test_analyzer_*.py -q
```

Expected: previous passing tests still pass plus the new ones.

- [ ] **Step 6: Commit**

```bash
git add scripts/twixt_replay_analyzer.py
git commit -m "feat(analyzer): wire recovery_retargeting summary + CSVs into analyze() (Spec 4 §6)"
```

---

# Phase 4 — Self-play integration

## Task 12: Add tracker hook in `play_game`

**Files:**
- Modify: `scripts/GPU/alphazero/self_play.py`

- [ ] **Step 1: Add `recovery_retargeting_config` parameter to `play_game`**

Find the `play_game` signature (around line 571) and add a new keyword parameter after the existing closeout knobs:

```python
def play_game(
    ...,
    recovery_retargeting_config: Optional["RecoveryRetargetingConfig"] = None,
    ...,
):
```

Import at the top of self_play.py:

```python
from scripts.GPU.alphazero.recovery_retargeting_diagnostics import (
    RecoveryRetargetingConfig,
    RecoveryRetargetingTracker,
)
from scripts.GPU.alphazero.connectivity_diagnostics import compute_goal_completion_state as _compute_goal_completion_state
```

- [ ] **Step 2: Instantiate the tracker once per game**

After the existing tracker/diagnostic initializations (around the start of `play_game`), add:

```python
    recovery_tracker = None
    if recovery_retargeting_config is not None and recovery_retargeting_config.enabled:
        recovery_tracker = RecoveryRetargetingTracker(
            config=recovery_retargeting_config,
            gc_state_provider=_compute_goal_completion_state,
        )
```

- [ ] **Step 3: Find the move-selection point and add the observe_move call**

Find where `move = mcts.select_move(visit_counts, ply)` is called (around line 1052). Compute `top1_share` from `visit_counts` if not already available, and call the tracker BEFORE applying the move:

```python
        # Spec 4 — recovery / re-targeting diagnostic per-ply hook.
        if recovery_tracker is not None:
            if visit_counts:
                total_visits = sum(visit_counts.values()) or 1
                top1_share = max(visit_counts.values()) / total_visits
            else:
                top1_share = None
            recovery_tracker.observe_move(
                state_before=state,
                selected_move=move,
                ply=ply,
                side_to_move=state.to_move,
                search_score=root_value,
                root_top1_share=top1_share,
            )
```

- [ ] **Step 4: Verify a smoke run launches**

```bash
.venv/bin/python -c "from scripts.GPU.alphazero.self_play import play_game; print('import ok')"
```

Expected: `import ok`.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/self_play.py
git commit -m "feat(self_play): recovery-retargeting tracker per-ply hook (Spec 4 §5.4)"
```

---

## Task 13: Add `GameRecord.recovery_retargeting_record` + finalize call

**Files:**
- Modify: `scripts/GPU/alphazero/self_play.py`

- [ ] **Step 1: Add the field to `GameRecord`**

In `self_play.py`, find the `GameRecord` dataclass (around line 410) and add after the existing `closeout_tiebreak_telemetry` field:

```python
    # Compact per-game recovery / re-targeting record (Spec 4 §4).
    # Top-level JSON key when present; omitted from JSON when None.
    recovery_retargeting_record: Optional[dict] = None
```

- [ ] **Step 2: Add the finalize call after goal_completion_record**

Find the `gc_record = gc_tracker.finalize_game(...)` block (around line 1296). Immediately after, add:

```python
    recovery_retargeting_record = (
        recovery_tracker.finalize_game(
            iteration=0,                       # populated downstream like goal_completion_record
            game_idx=game_id,                  # play_game's local counter; saver overrides per §5.8
            game_id=f"game_{game_id:03d}",
            winner=winner,
            starting_player=start_player,
            n_moves=len(move_history),
            reason=_gc_reason_for_record,
        )
        if recovery_tracker is not None else None
    )
```

- [ ] **Step 3: Pass it into the `GameRecord` constructor**

Find the `return GameRecord(...)` call (around line 1306) and add at the end:

```python
        recovery_retargeting_record=recovery_retargeting_record,
```

- [ ] **Step 4: Confirm no syntax errors**

```bash
.venv/bin/python -c "import scripts.GPU.alphazero.self_play; print('ok')"
```

Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/self_play.py
git commit -m "feat(self_play): GameRecord.recovery_retargeting_record + finalize call (Spec 4 §5.5)"
```

---

# Phase 5 — IPC + trainer transport

## Task 14: Add `GameComplete.recovery_retargeting_record` IPC field

**Files:**
- Modify: `scripts/GPU/alphazero/ipc_messages.py`

- [ ] **Step 1: Find the existing `closeout_tiebreak_telemetry` field**

```bash
grep -n "closeout_tiebreak_telemetry\|closeout_td1_telemetry" scripts/GPU/alphazero/ipc_messages.py
```

- [ ] **Step 2: Add the parallel field on `GameComplete`**

After the existing `closeout_tiebreak_telemetry: Optional[dict] = None` line, add:

```python
    # Spec 4 — recovery / re-targeting per-game record (None if no side triggered).
    recovery_retargeting_record: Optional[dict] = None
```

- [ ] **Step 3: Commit**

```bash
git add scripts/GPU/alphazero/ipc_messages.py
git commit -m "feat(ipc): GameComplete.recovery_retargeting_record field (Spec 4 §5.6)"
```

---

## Task 15: `self_play_worker.py` IPC forwarding

**Files:**
- Modify: `scripts/GPU/alphazero/self_play_worker.py`

- [ ] **Step 1: Find the existing GameComplete construction**

```bash
grep -n "closeout_tiebreak_telemetry\|closeout_td1_telemetry" scripts/GPU/alphazero/self_play_worker.py
```

Locate the lines (around 271-272) where Fix 1 and Fix 2 are forwarded.

- [ ] **Step 2: Add the new forwarding line**

After `closeout_tiebreak_telemetry=game.closeout_tiebreak_telemetry,` add:

```python
                recovery_retargeting_record=game.recovery_retargeting_record,
```

(Direct attribute access matches Fix 2's pattern — the field is declared on `GameRecord` with `None` default.)

- [ ] **Step 3: Commit**

```bash
git add scripts/GPU/alphazero/self_play_worker.py
git commit -m "feat(self_play_worker): forward recovery_retargeting_record via GameComplete (Spec 4 §5.6)"
```

---

## Task 16: Trainer kwargs, `_inject_iteration` extension, and config threading

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py`

- [ ] **Step 1: Find the Fix 2 kwargs and threading sites**

```bash
grep -n "closeout_selection_tiebreak_enabled\|all_closeout_tiebreak_telemetry\|_inject_iteration" scripts/GPU/alphazero/trainer.py | head -20
```

- [ ] **Step 2: Add 11 new `train()` kwargs (Spec §5.1 config fields)**

After the existing Fix 2 kwargs in `train()`, add:

```python
    # Spec 4 — recovery / re-targeting diagnostic config.
    recovery_retargeting_enabled: bool = True,
    recovery_retargeting_collapse_value_threshold: float = -0.75,
    recovery_retargeting_severe_value_threshold: float = -0.90,
    recovery_retargeting_diffuse_root_top1_threshold: float = 0.20,
    recovery_retargeting_very_diffuse_root_top1_threshold: float = 0.15,
    recovery_retargeting_delta_threshold: float = 0.50,
    recovery_retargeting_delta_max_current_score: float = -0.30,
    recovery_retargeting_alternate_component_min_size: int = 4,
    recovery_retargeting_classify_defense: bool = True,
    recovery_retargeting_max_sampled_moves_per_side: int = 32,
    recovery_retargeting_sample_all_moves: bool = False,
```

- [ ] **Step 3: Construct `RecoveryRetargetingConfig` once at startup**

Import at the top of `trainer.py`:

```python
from scripts.GPU.alphazero.recovery_retargeting_diagnostics import (
    RecoveryRetargetingConfig,
    aggregate_recovery_retargeting_records,
    validate_config as _validate_recovery_retargeting_config,
)
```

In `train()` before the first MCTSConfig construction, add:

```python
    recovery_retargeting_config = RecoveryRetargetingConfig(
        enabled=recovery_retargeting_enabled,
        collapse_value_threshold=recovery_retargeting_collapse_value_threshold,
        severe_collapse_value_threshold=recovery_retargeting_severe_value_threshold,
        diffuse_root_top1_threshold=recovery_retargeting_diffuse_root_top1_threshold,
        very_diffuse_root_top1_threshold=recovery_retargeting_very_diffuse_root_top1_threshold,
        delta_threshold=recovery_retargeting_delta_threshold,
        delta_max_current_score=recovery_retargeting_delta_max_current_score,
        alternate_component_min_size=recovery_retargeting_alternate_component_min_size,
        classify_defense=recovery_retargeting_classify_defense,
        max_sampled_moves_per_side=recovery_retargeting_max_sampled_moves_per_side,
        sample_all_moves=recovery_retargeting_sample_all_moves,
    )
    _validate_recovery_retargeting_config(recovery_retargeting_config)
```

- [ ] **Step 4: Pass `recovery_retargeting_config` into both `play_game` call sites**

Find both call sites (parallel-path and serial-path) by searching for `play_game(`. At each, add the kwarg:

```python
        recovery_retargeting_config=recovery_retargeting_config,
```

- [ ] **Step 5: Extend `_inject_iteration`**

Find `_inject_iteration` (around line 50). Update to also inject iteration into `recovery_retargeting_record`:

```python
def _inject_iteration(record: Optional[dict], iteration: Optional[int]) -> Optional[dict]:
    """Set iteration on a goal_completion_record OR recovery_retargeting_record copy."""
    if record is None or iteration is None:
        return record
    return {**record, "iteration": iteration}
```

(Function shape is unchanged; just confirm the docstring reflects the dual use.)

In `_save_game_from_ipc` and `_save_game_from_record` (lines ~113 and ~172), find the existing `goal_completion_record=_inject_iteration(...)` lines and add a sibling:

```python
        recovery_retargeting_record=_inject_iteration(
            msg.recovery_retargeting_record, getattr(game_saver, "_current_iter", None),
        ),
```

(replace `msg` with `game` in `_save_game_from_record`).

- [ ] **Step 6: Confirm no syntax errors**

```bash
.venv/bin/python -c "from scripts.GPU.alphazero.trainer import train; print('ok')"
```

Expected: `ok`.

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py
git commit -m "feat(trainer): recovery-retargeting kwargs + RecoveryRetargetingConfig + _inject_iteration extension (Spec 4 §5.6)"
```

---

## Task 17: Trainer IPC append, serial append, sidecar emit, startup banner

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py`

- [ ] **Step 1: Find the Fix 2 collection sites**

```bash
grep -n "all_closeout_tiebreak_telemetry\|closeout_selection_tiebreak" scripts/GPU/alphazero/trainer.py | head -20
```

The four sites of interest (per d788023f4 plan):
- Inner-iter list init
- Parallel-IPC branch append (around trainer.py:1864-1872)
- Parallel-to-merged forwarding (around 2039-2046 and 2852-2856)
- Serial-path append (around 3028-3030 / 3078-3084)
- Sidecar emit (around 3718-3726)

- [ ] **Step 2: Add `all_recovery_retargeting_records` list at both scopes**

In `run_parallel_selfplay()` (or whatever wraps the parallel branch), find:

```python
    all_closeout_tiebreak_telemetry: list = []
```

Add immediately after:

```python
    all_recovery_retargeting_records: list = []
```

And in the inner-iter scope, find the same `all_closeout_tiebreak_telemetry: list = []` and add the same line.

- [ ] **Step 3: IPC branch append**

Find the IPC branch where `msg.closeout_tiebreak_telemetry` is appended (around line 1869-1871). Immediately after, add:

```python
            if getattr(msg, "recovery_retargeting_record", None) is not None:
                all_recovery_retargeting_records.append(msg.recovery_retargeting_record)
```

- [ ] **Step 4: Parallel-to-merged forwarding**

Find the parallel_stats dict return (around line 2043-2046) and add:

```python
        "all_recovery_retargeting_records": list(all_recovery_retargeting_records),
```

In the parallel-merge consumption point (around line 2852-2900), find the equivalent for tiebreak and add:

```python
            _par_rr = parallel_stats.get("all_recovery_retargeting_records", [])
            if _par_rr:
                all_recovery_retargeting_records.extend(_par_rr)
```

- [ ] **Step 5: Serial-path append**

Find the serial-path collection (around line 3078-3084) where `game.closeout_tiebreak_telemetry` is appended via `getattr`. Add the parallel line:

```python
                rr_rec = getattr(game, "recovery_retargeting_record", None)
                if rr_rec is not None:
                    all_recovery_retargeting_records.append(rr_rec)
```

- [ ] **Step 6: Sidecar emit**

Find the sidecar emit for Fix 2 (around line 3722-3726):

```python
            _sidecar["closeout_selection_tiebreak"] = _merge_closeout_tiebreak_telemetry(
                all_closeout_tiebreak_telemetry
            )
```

Immediately after, add:

```python
            _sidecar["recovery_retargeting_summary"] = aggregate_recovery_retargeting_records(
                all_recovery_retargeting_records, games_total=len(games_for_this_iter or [])
            )
```

(Replace `games_for_this_iter` with whatever local variable holds the iteration's full game list — match the Fix 2 pattern.)

- [ ] **Step 7: Startup banner**

Find the Fix 2 banner block (around line 2548-2555). Immediately after the Fix 2 banner's `else` branch, add:

```python
    if recovery_retargeting_enabled:
        print(f"  Recovery / re-targeting diagnostics: enabled")
        print(f"    collapse_value <=         {recovery_retargeting_collapse_value_threshold:.2f}")
        print(f"    severe_value <=           {recovery_retargeting_severe_value_threshold:.2f}")
        print(f"    diffuse_root_top1 <=       {recovery_retargeting_diffuse_root_top1_threshold:.2f}")
        print(f"    delta >=                   {recovery_retargeting_delta_threshold:.2f}")
        print(f"    delta_max_current_score:  {recovery_retargeting_delta_max_current_score:.2f}")
        print(f"    alternate_component_min_size:  {recovery_retargeting_alternate_component_min_size}")
        if recovery_retargeting_classify_defense:
            print(f"    classify_defense:          on")
        else:
            print(f"    classify_defense:          off  (WARNING: local_drift may include defensive moves)")
        if recovery_retargeting_sample_all_moves:
            print(f"    sample_all_moves:          on")
        else:
            print(f"    sample_all_moves:          off  (cap={recovery_retargeting_max_sampled_moves_per_side} per side)")
    else:
        print(f"  Recovery / re-targeting diagnostics: disabled")
```

- [ ] **Step 8: Confirm no syntax errors**

```bash
.venv/bin/python -c "from scripts.GPU.alphazero.trainer import train; print('ok')"
```

Expected: `ok`.

- [ ] **Step 9: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py
git commit -m "feat(trainer): recovery_retargeting IPC append + sidecar emit + startup banner (Spec 4 §5.6, §7.3)"
```

---

# Phase 6 — Game-saver fix + train CLI

## Task 18: Extend `game_saver` reconciliation for `recovery_retargeting_record`

**Files:**
- Modify: `scripts/GPU/alphazero/game_saver.py`
- Test: `tests/test_game_saver_recovery_retargeting_id_fields.py`

- [ ] **Step 1: Write the test**

Create `tests/test_game_saver_recovery_retargeting_id_fields.py`:

```python
"""save_game_replay must overwrite recovery_retargeting_record's game_idx/game_id
with the saver's authoritative values, mirroring the 32c4966a6 fix for
goal_completion_record."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.game_saver import save_game_replay


def _read(path):
    with open(path) as f:
        return json.load(f)


def test_save_overrides_recovery_record_game_idx_and_game_id(tmp_path):
    bogus = {
        "version": 1,
        "iteration": 170,
        "game_idx": 13,             # dispatch-order, wrong
        "game_id": "game_013",
        "winner": "red", "loser": "black",
        "triggered_sides": ["black"],
        "side_records": {"red": {"triggered": False}, "black": {"triggered": True}},
    }
    out = save_game_replay(
        games_dir=tmp_path,
        iteration=170,
        game_idx=22,               # save-order, authoritative
        winner="red",
        move_history=((4, 19), (12, 12)),
        n_moves=2,
        recovery_retargeting_record=bogus,
    )
    saved = _read(out)
    assert saved["recovery_retargeting_record"]["game_idx"] == 22
    assert saved["recovery_retargeting_record"]["game_id"] == "game_022"
    assert saved["recovery_retargeting_record"]["triggered_sides"] == ["black"]


def test_save_does_not_mutate_caller_recovery_record(tmp_path):
    caller_view = {"iteration": 170, "game_idx": 13, "game_id": "game_013"}
    save_game_replay(
        games_dir=tmp_path, iteration=170, game_idx=22, winner="red",
        move_history=((4, 19),), n_moves=1,
        recovery_retargeting_record=caller_view,
    )
    assert caller_view["game_idx"] == 13
    assert caller_view["game_id"] == "game_013"


def test_save_recovery_record_none_is_unchanged(tmp_path):
    out = save_game_replay(
        games_dir=tmp_path, iteration=170, game_idx=22, winner=None,
        move_history=(), n_moves=0, recovery_retargeting_record=None,
    )
    saved = _read(out)
    assert "recovery_retargeting_record" not in saved
```

- [ ] **Step 2: Run → fail (parameter doesn't exist)**

```bash
.venv/bin/pytest tests/test_game_saver_recovery_retargeting_id_fields.py -v
```

Expected: TypeError on unknown kwarg `recovery_retargeting_record`.

- [ ] **Step 3: Add the parameter and the override block**

In `scripts/GPU/alphazero/game_saver.py`:

After the existing `goal_completion_record` parameter, add:

```python
    # Compact per-game recovery / re-targeting record (spec 2026-05-12 §5.8).
    recovery_retargeting_record: Optional[dict] = None,
```

After the existing `if goal_completion_record is not None: ...` reconciliation block, add:

```python
    if recovery_retargeting_record is not None:
        # play_game writes a dispatch-order counter into game_idx/game_id;
        # save-order counter is authoritative. Override on a shallow copy
        # (no in-place mutation — trainer collects records via IPC before save).
        record["recovery_retargeting_record"] = {
            **recovery_retargeting_record,
            "game_idx": game_idx,
            "game_id": f"game_{game_idx:03d}",
        }
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_game_saver_recovery_retargeting_id_fields.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Wire the new param into `_save_game_from_ipc` and `_save_game_from_record` in `trainer.py`**

In `trainer.py`, find both functions and add:

```python
        recovery_retargeting_record=_inject_iteration(
            msg.recovery_retargeting_record, getattr(game_saver, "_current_iter", None),
        ),
```

(use `game.recovery_retargeting_record` in `_save_game_from_record`).

Also find the `game_saver.maybe_save_game(...)` call in `_save_game_from_record` and add the parallel line in the kwargs list.

For the IPC path, find where `save_game_replay` is called in trainer's IPC processing branch and add the kwarg.

- [ ] **Step 6: Run tests + verify full suite**

```bash
.venv/bin/pytest tests/test_game_saver*.py -v
```

Expected: all PASSED.

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/game_saver.py scripts/GPU/alphazero/trainer.py tests/test_game_saver_recovery_retargeting_id_fields.py
git commit -m "feat(game_saver,trainer): reconcile recovery_retargeting_record.game_idx with save-order (Spec 4 §5.8)"
```

---

## Task 19: `train.py` CLI flags + validation

**Files:**
- Modify: `scripts/GPU/alphazero/train.py`
- Test: `tests/test_train_recovery_retargeting_cli.py`

- [ ] **Step 1: Write CLI smoke test**

Create `tests/test_train_recovery_retargeting_cli.py`:

```python
"""Tests for Spec 4 train.py CLI surface."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _build_parser():
    """Helper that recreates the train.py argparse with the new flags only."""
    from scripts.GPU.alphazero.train import _add_recovery_retargeting_args
    parser = argparse.ArgumentParser()
    _add_recovery_retargeting_args(parser)
    return parser


def test_default_flags_enable_diagnostic_off_by_disable_flag():
    p = _build_parser()
    args = p.parse_args([])
    # By default, --recovery-retargeting-disabled is False, so diagnostic is enabled.
    assert args.recovery_retargeting_disabled is False
    assert args.recovery_retargeting_classify_defense is True


def test_disable_flag_turns_off():
    p = _build_parser()
    args = p.parse_args(["--recovery-retargeting-disabled"])
    assert args.recovery_retargeting_disabled is True


def test_no_classify_defense_flag_turns_off():
    p = _build_parser()
    args = p.parse_args(["--recovery-retargeting-no-classify-defense"])
    assert args.recovery_retargeting_classify_defense is False


def test_threshold_overrides_parse():
    p = _build_parser()
    args = p.parse_args([
        "--recovery-retargeting-collapse-value-threshold", "-0.60",
        "--recovery-retargeting-delta-threshold", "0.40",
    ])
    assert args.recovery_retargeting_collapse_value_threshold == -0.60
    assert args.recovery_retargeting_delta_threshold == 0.40
```

- [ ] **Step 2: Run → fail**

```bash
.venv/bin/pytest tests/test_train_recovery_retargeting_cli.py -v
```

Expected: ImportError on `_add_recovery_retargeting_args`.

- [ ] **Step 3: Add the helper and call it in train.py**

In `scripts/GPU/alphazero/train.py`, immediately after the Fix 2 flag block, add:

```python
def _add_recovery_retargeting_args(parser):
    """Spec 4 recovery / re-targeting diagnostic CLI flags."""
    parser.add_argument("--recovery-retargeting-disabled", action="store_true",
                        help="Disable the diagnostic. Default: enabled.")
    parser.add_argument("--recovery-retargeting-collapse-value-threshold", type=float, default=-0.75)
    parser.add_argument("--recovery-retargeting-severe-value-threshold", type=float, default=-0.90)
    parser.add_argument("--recovery-retargeting-diffuse-root-top1-threshold", type=float, default=0.20)
    parser.add_argument("--recovery-retargeting-very-diffuse-root-top1-threshold", type=float, default=0.15)
    parser.add_argument("--recovery-retargeting-delta-threshold", type=float, default=0.50)
    parser.add_argument("--recovery-retargeting-delta-max-current-score", type=float, default=-0.30)
    parser.add_argument("--recovery-retargeting-alternate-component-min-size", type=int, default=4)
    classify_group = parser.add_mutually_exclusive_group()
    classify_group.add_argument("--recovery-retargeting-classify-defense", dest="recovery_retargeting_classify_defense", action="store_true", default=True)
    classify_group.add_argument("--recovery-retargeting-no-classify-defense", dest="recovery_retargeting_classify_defense", action="store_false")
    parser.add_argument("--recovery-retargeting-max-sampled-moves-per-side", type=int, default=32)
    parser.add_argument("--recovery-retargeting-sample-all-moves", action="store_true", default=False)
```

Call `_add_recovery_retargeting_args(parser)` immediately after the Fix 2 flag additions in the main parser construction.

- [ ] **Step 4: Wire flags into `train_kwargs.update(...)` mapping**

Find where Fix 2 flags are mapped into `train_kwargs` (search `closeout_selection_tiebreak_enabled=`). After that block, add:

```python
    train_kwargs.update(
        recovery_retargeting_enabled=not args.recovery_retargeting_disabled,
        recovery_retargeting_collapse_value_threshold=args.recovery_retargeting_collapse_value_threshold,
        recovery_retargeting_severe_value_threshold=args.recovery_retargeting_severe_value_threshold,
        recovery_retargeting_diffuse_root_top1_threshold=args.recovery_retargeting_diffuse_root_top1_threshold,
        recovery_retargeting_very_diffuse_root_top1_threshold=args.recovery_retargeting_very_diffuse_root_top1_threshold,
        recovery_retargeting_delta_threshold=args.recovery_retargeting_delta_threshold,
        recovery_retargeting_delta_max_current_score=args.recovery_retargeting_delta_max_current_score,
        recovery_retargeting_alternate_component_min_size=args.recovery_retargeting_alternate_component_min_size,
        recovery_retargeting_classify_defense=args.recovery_retargeting_classify_defense,
        recovery_retargeting_max_sampled_moves_per_side=args.recovery_retargeting_max_sampled_moves_per_side,
        recovery_retargeting_sample_all_moves=args.recovery_retargeting_sample_all_moves,
    )
```

- [ ] **Step 5: Run tests + verify CLI help**

```bash
.venv/bin/pytest tests/test_train_recovery_retargeting_cli.py -v
.venv/bin/python -m scripts.GPU.alphazero.train --help | grep recovery-retargeting
```

Expected: 4 tests PASSED. Help output shows at least these flags: `--recovery-retargeting-disabled`, `--recovery-retargeting-collapse-value-threshold`, `--recovery-retargeting-diffuse-root-top1-threshold`, `--recovery-retargeting-no-classify-defense`. The `--recovery-retargeting-worst-cases-top-k` flag must NOT appear (it's analyzer-side per Task 11).

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/train.py tests/test_train_recovery_retargeting_cli.py
git commit -m "feat(train): recovery-retargeting CLI flags + train_kwargs wiring (Spec 4 §7.1)"
```

---

# Phase 7 — Smoke and production launch

## Task 20: 5-game smoke run

**Files:** none (manual verification)

- [ ] **Step 1: Run the full regression suite**

```bash
.venv/bin/pytest tests/test_mcts*.py tests/test_analyzer_*.py tests/test_self_play_closeout*.py tests/test_train_closeout*.py tests/test_game_saver*.py tests/test_recovery_retargeting_diagnostics.py tests/test_analyzer_recovery_retargeting.py tests/test_train_recovery_retargeting_cli.py tests/test_game_saver_recovery_retargeting_id_fields.py -q
```

Expected: all PASSED (168 baseline + new recovery-retargeting tests + game-saver tests + train CLI tests). 2 known skips remain.

- [ ] **Step 2: Train CLI smoke**

```bash
.venv/bin/python -m scripts.GPU.alphazero.train --help | grep recovery-retargeting
```

Confirm these train-side flags appear:
- `--recovery-retargeting-disabled`
- `--recovery-retargeting-collapse-value-threshold`
- `--recovery-retargeting-diffuse-root-top1-threshold`
- `--recovery-retargeting-no-classify-defense`

`--recovery-retargeting-worst-cases-top-k` must NOT appear here.

- [ ] **Step 3: Analyzer CLI smoke**

```bash
.venv/bin/python scripts/twixt_replay_analyzer.py --help | grep recovery-retargeting
```

Confirm:
- `--recovery-retargeting-worst-cases-top-k`

- [ ] **Step 4: Launch 1-iteration × 5-game smoke**

```bash
.venv/bin/python -m scripts.GPU.alphazero.train \
  --resume checkpoints/alphazero-v2-staged/model_iter_0169.safetensors \
  --iterations 170 \
  --games-per-iter 5 \
  --checkpoint-dir /tmp/recovery_smoke_ckpt \
  --value-weight 0.5 --value-lr-scale 0.0025 --value-grad-max-norm 0.05 \
  --progress-weighted-value-loss --progress-weight-floor 0.25 \
  --n-workers 2 --mcts-eval-batch-size 14 --mcts-stall-flush-sims 48 \
  --closeout-td1-visit-forcing-enabled \
  --closeout-selection-tiebreak-enabled
```

Expected output:

- Banner shows ALL THREE diagnostic blocks enabled:
  ```
  Closeout td=1 visit forcing: enabled
  Closeout selection tie-break: enabled
  Recovery / re-targeting diagnostics: enabled
  ```
- Run completes in ~10 minutes.

- [ ] **Step 5: Inspect sidecar and game JSONs**

```bash
ls -la scripts/GPU/logs/games/iter_0169_stats.json
.venv/bin/python -c "import json; s = json.load(open('scripts/GPU/logs/games/iter_0169_stats.json')); print(json.dumps(s.get('recovery_retargeting_summary', {}), indent=2)[:500])"
```

Expected: sidecar contains `recovery_retargeting_summary` with `version=1`, `enabled=True`, and `games_total=5`. If no game triggered, `games_triggered=0` and the section is still well-formed. If a game triggered, inspect one game JSON for the `recovery_retargeting_record` key.

- [ ] **Step 6: Tear down smoke artifacts**

```bash
rm -rf /tmp/recovery_smoke_ckpt
# Keep the iter_0169_stats.json + iter_0169_game_*.json files only if 170-179 launches against them.
```

- [ ] **Step 7: Commit nothing (this is a verification task)**

---

## Task 21: 170-179 production launch (USER-GATED)

**Files:** none (training run)

**This task is user-gated.** Present the verified launch command to the user; do NOT launch training without confirmation.

- [ ] **Step 1: Verify the resume checkpoint exists**

```bash
ls -la checkpoints/alphazero-v2-staged/model_iter_0169.safetensors
```

- [ ] **Step 2: Present this launch command to the user**

```bash
.venv/bin/python -m scripts.GPU.alphazero.train \
  --resume checkpoints/alphazero-v2-staged/model_iter_0169.safetensors \
  --iterations 179 \
  --games-per-iter 100 \
  --checkpoint-dir checkpoints/alphazero-v2-staged \
  --value-weight 0.5 \
  --value-lr-scale 0.0025 \
  --value-grad-max-norm 0.05 \
  --progress-weighted-value-loss \
  --progress-weight-floor 0.25 \
  --n-workers 10 \
  --mcts-eval-batch-size 14 \
  --mcts-stall-flush-sims 48 \
  --opening-noise-ply 10 \
  --opening-dirichlet-alpha 0.7 \
  --opening-dirichlet-eps 0.50 \
  --mirror-prob 0.5 \
  --resign-enabled \
  --resign-min-ply 80 \
  --resign-threshold -0.945 \
  --resign-window 12 \
  --resign-k 4 \
  --resign-min-visits 200 \
  --resign-min-top1-share 0.102 \
  --adjudicate-enabled \
  --adjudicate-min-ply 240 \
  --adjudicate-threshold 0.20 \
  --adjudicate-min-visits 200 \
  --adjudicate-min-top1-share 0.13 \
  --max-positions-per-game 64 \
  --endgame-keep-positions 16 \
  --conversion-policy-loss-enabled \
  --conversion-policy-loss-weight 0.05 \
  --conversion-completion-weight 1.0 \
  --conversion-reducer-weight 0.35 \
  --conversion-max-total-goal-distance 2 \
  --conversion-sample-boost 2.0 \
  --conversion-max-batch-fraction 0.15 \
  --closeout-td1-visit-forcing-enabled \
  --closeout-td1-min-visits 8 \
  --closeout-td1-max-forced-moves 4 \
  --closeout-selection-tiebreak-enabled \
  --closeout-selection-tiebreak-max-distance 2 \
  --closeout-selection-tiebreak-topk 5 \
  --closeout-selection-tiebreak-min-value 0.95 \
  --closeout-selection-tiebreak-min-share 0.05
```

(Recovery diagnostic flags are all at defaults, so no `--recovery-retargeting-*` flag is required.)

Verify the startup banner shows ALL THREE diagnostic blocks enabled before letting the run proceed. If any says `disabled`, abort and investigate.

- [ ] **Step 3: Post-run analysis**

After 170-179 completes:

```bash
mkdir -p Replays/170-179
for f in scripts/GPU/logs/games/iter_0169_game_*.json scripts/GPU/logs/games/iter_017?_game_*.json; do
  ln -sf "../../$f" "Replays/170-179/$(basename $f)" 2>/dev/null
done
for f in scripts/GPU/logs/games/iter_0169_stats.json scripts/GPU/logs/games/iter_017?_stats.json; do
  ln -sf "../../$f" "Replays/170-179/$(basename $f)" 2>/dev/null
done
.venv/bin/python ./scripts/twixt_replay_analyzer.py --input Replays/170-179 --out Replays/170-179_Replay
```

Inspect:
- `Recovery / Re-targeting Diagnostics` section in `Replays/170-179_Replay/report_170-179.txt`
- `Replays/170-179_Replay/recovery_retargeting_worst_cases_170-179.csv`
- `Replays/170-179_Replay/recovery_retargeting_by_iter_170-179.csv`

Apply spec §11 decision rule.

---

# Self-Review notes (engineer-facing)

- If any task fails its tests, do NOT introduce ad-hoc try/except blocks to mask the failure. Root-cause the issue.
- The component-extraction tests use a `_StubState` shim. The real `TwixtState._get_connected_component` respects enemy-blocked bridges; the stub does not. This is acceptable because the classifier tests target classification logic, not bridge-blocking. Tests that need true bridge semantics should be added as integration tests in a follow-up if needed.
- The trainer integration (Task 16/17) mirrors Fix 1 / Fix 2 transport patterns. Use `git show 52a51bdf5 2c5f56b71 d788023f4` to see the exact line ranges if anything looks unclear.
- Task 11's wiring uses `args.recovery_retargeting_worst_cases_top_k` inside `analyze()`. If `analyze()` doesn't have argparse args in scope, thread the value through as a function parameter with default 25. Do not break the existing `analyze()` signature for other callers.
- Recovery diagnostic data is generated only at runtime — Task 20's smoke run is the verification gate. If the smoke shows the sidecar block but no `recovery_retargeting_record` on any game JSON (because no game triggered), that is acceptable; rerun on more games to surface a trigger or trust that 170-179's larger sample will produce them.
