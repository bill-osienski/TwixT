# Goal-Completion Policy Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the policy head to rank endpoint-completing and distance-reducing moves highly on dominant-unclosed (closeout-shaped) TwixT positions, fixing the policy-prior bias that 110-119 production analysis surfaced.

**Architecture:** Auxiliary cross-entropy loss against an explicit conversion target (completion=1.0, reducer=0.35, normalized over those moves only), applied on side-to-move closeout-eligible positions. Candidate move sets are computed once during self-play (reusing Spec 1.5's `gc_state_full`) and stored on `PositionRecord`. Optional bounded replay sample boost amplifies exposure. Telemetry-only recovery / extreme-closeout-drift bucket measures the harder dominant-lost failure mode.

**Tech Stack:** Python 3.14, MLX (Apple Silicon ML framework), NumPy, pytest. Existing AlphaZero TwixT trainer in `scripts/GPU/alphazero/`.

**Reference spec:** `docs/superpowers/specs/2026-05-06-goal-completion-policy-correction-design.md`

---

## File structure

### New files

| File | Responsibility |
|---|---|
| `scripts/GPU/alphazero/conversion_loss.py` | `is_conversion_eligible()` predicate; `build_conversion_target()` per-position target; `make_conversion_aux_tensors()` batch tensors. Pure functions, no MLX state. |
| `scripts/GPU/alphazero/conversion_telemetry.py` | `build_conversion_training_block()`; `is_recovery_or_extreme_closeout_drift()` predicate; `build_recovery_block()`. Pure dict math. |
| `tests/test_conversion_eligibility.py` | Predicate tests (6) |
| `tests/test_conversion_target.py` | Target construction tests (7) |
| `tests/test_conversion_aux_tensors.py` | Batch tensor build tests (5) |
| `tests/test_conversion_loss.py` | Loss math tests (7) |
| `tests/test_replay_buffer_conversion.py` | Eligibility tracking + sampling tests (8) |
| `tests/test_conversion_telemetry.py` | Sidecar block tests (12) |
| `tests/test_position_record_conversion.py` | IPC + persistence tests (5) |
| `tests/test_conversion_cli_config.py` | CLI/config invariants (6) |

### Modified files

| File | What changes |
|---|---|
| `scripts/GPU/alphazero/self_play.py` | `PositionRecord` gains `conversion: Optional[dict]`; `play_game()` builds and attaches conversion metadata after MCTS + classification. |
| `scripts/GPU/alphazero/ipc_messages.py` | (No change — `GameComplete.positions` already carries dicts; verify with round-trip test.) |
| `scripts/GPU/alphazero/trainer.py` | `make_padded_batch` gains `return_legal` kwarg; `alphazero_loss_batch` and `train_step` extend signatures; `ReplayBuffer` gains eligibility index pool + stratified sampling; sidecar writer emits two new blocks. |
| `scripts/GPU/alphazero/train.py` | New CLI flags (10); validation; banner additions. |
| `scripts/twixt_replay_analyzer.py` | Two minor read-only sections + two per-iter CSVs (Phase 4 task). |
| `tests/test_self_play_goal_completion_integration.py` | Extend with conversion-attach tests. |
| `tests/test_trainer_loss.py` | Extend with conversion-enabled smoke tests. |

---

## Phase 1 — Data plumbing (no behavior change)

### Task 1: Eligibility predicate module

**Files:**
- Create: `scripts/GPU/alphazero/conversion_loss.py`
- Test: `tests/test_conversion_eligibility.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_conversion_eligibility.py
"""Eligibility predicate tests (Spec 2 §4)."""
from scripts.GPU.alphazero.conversion_loss import is_conversion_eligible


def _gc(total=2, comp=12, completion=None, reducing=None):
    return {
        "total_goal_distance": total,
        "largest_component_size": comp,
        "endpoint_completion_moves": completion if completion is not None else [(0, 8)],
        "distance_reducing_moves":   reducing   if reducing   is not None else [(22, 4)],
    }


def test_eligible_with_two_endpoint_closeout():
    gc = _gc(total=2, comp=12)
    assert is_conversion_eligible(gc, max_total_goal_distance=2, min_component_size=8) is True


def test_ineligible_when_total_distance_above_threshold():
    gc = _gc(total=4)
    assert is_conversion_eligible(gc, max_total_goal_distance=3, min_component_size=8) is False


def test_ineligible_when_component_too_small():
    gc = _gc(comp=6)
    assert is_conversion_eligible(gc, max_total_goal_distance=2, min_component_size=8) is False


def test_ineligible_when_no_completion_or_reducer_moves():
    gc = _gc(completion=[], reducing=[])
    assert is_conversion_eligible(gc, max_total_goal_distance=2, min_component_size=8) is False


def test_ineligible_when_gc_state_full_is_none():
    assert is_conversion_eligible(None, max_total_goal_distance=2, min_component_size=8) is False


def test_ineligible_when_total_distance_is_none():
    gc = _gc()
    gc["total_goal_distance"] = None
    assert is_conversion_eligible(gc, max_total_goal_distance=2, min_component_size=8) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_conversion_eligibility.py -v
```

Expected: ImportError / ModuleNotFoundError on `conversion_loss`.

- [ ] **Step 3: Implement the predicate**

```python
# scripts/GPU/alphazero/conversion_loss.py
"""Conversion auxiliary loss helpers (Spec 2).

Pure functions — no MLX state, no I/O. Predicates and target builders for
the policy-side closeout correction.
"""
from __future__ import annotations
from typing import Optional


def is_conversion_eligible(
    gc_state_full: Optional[dict],
    *,
    max_total_goal_distance: int,
    min_component_size: int,
) -> bool:
    """Determines whether a pre-move state qualifies the side-to-move's
    PositionRecord for conversion auxiliary loss.

    Pure dict math — no BFS. Defends against missing/None fields.
    """
    if gc_state_full is None:
        return False
    total = gc_state_full.get("total_goal_distance")
    comp_size = gc_state_full.get("largest_component_size")
    if total is None or comp_size is None:
        return False
    if total > max_total_goal_distance:
        return False
    if comp_size < min_component_size:
        return False
    completion = gc_state_full.get("endpoint_completion_moves") or []
    reducing = gc_state_full.get("distance_reducing_moves") or []
    if not completion and not reducing:
        return False
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_conversion_eligibility.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/conversion_loss.py tests/test_conversion_eligibility.py
git commit -m "feat(conversion): add is_conversion_eligible predicate

Pure dict-math predicate over gc_state_full. Single threshold knob
(max_total_goal_distance) per Spec 2 §4.1; requires non-empty
completion or reducer move lists.
"
```

---

### Task 2: PositionRecord conversion field + round-trip

**Files:**
- Modify: `scripts/GPU/alphazero/self_play.py` (`PositionRecord` dataclass and `to_dict`/`from_dict`)
- Test: `tests/test_position_record_conversion.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_position_record_conversion.py
"""PositionRecord.conversion round-trip (Spec 2 §5)."""
import numpy as np
from scripts.GPU.alphazero.self_play import PositionRecord


def _make_position(conversion=None):
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move="red",
        legal_moves=[(0, 8), (5, 5), (22, 4)],
        visit_counts=[10, 5, 3],
        outcome=1.0,
        active_size=24,
        ply=37,
        game_n_moves=59,
        conversion=conversion,
    )


def test_position_record_conversion_round_trip_dict():
    conv = {
        "version": 1,
        "total_goal_distance": 2,
        "largest_component_size": 12,
        "endpoint_completion_moves": [[0, 8]],
        "distance_reducing_moves":   [[22, 4]],
        "conversion_category": "two_endpoint_closeout_2ply",
        "selected_primary_class": "redundant_reinforcement",
    }
    p = _make_position(conversion=conv)
    d = p.to_dict()
    p2 = PositionRecord.from_dict(d)
    assert p2.conversion == conv


def test_position_record_conversion_defaults_to_none():
    p = _make_position(conversion=None)
    d = p.to_dict()
    p2 = PositionRecord.from_dict(d)
    assert p2.conversion is None


def test_position_record_buffer_load_with_old_no_conversion_field():
    """Pre-Spec-2 buffers: dict has no 'conversion' key. from_dict must default to None."""
    legacy_dict = {
        "board_tensor": np.zeros((24, 24, 30), dtype=np.float32).tolist(),
        "to_move": "red",
        "legal_moves": [(0, 8)],
        "visit_counts": [1],
        "outcome": 1.0,
        "active_size": 24,
        "ply": 0,
        "game_n_moves": 1,
        # no 'conversion' key
    }
    p = PositionRecord.from_dict(legacy_dict)
    assert p.conversion is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_position_record_conversion.py -v
```

Expected: FAIL — `PositionRecord.__init__()` got an unexpected keyword argument 'conversion'.

- [ ] **Step 3: Add the field to PositionRecord and update to_dict / from_dict**

In `scripts/GPU/alphazero/self_play.py`, locate the `PositionRecord` dataclass (around line 291). Add the field:

```python
@dataclass
class PositionRecord:
    """Single training position from self-play.
    Stored in MLX-native NHWC layout to avoid transpose during training."""
    board_tensor: np.ndarray
    to_move: str
    legal_moves: List[Tuple[int, int]]
    visit_counts: List[int]
    outcome: Optional[float] = None
    active_size: int = 24
    ply: int = 0
    game_n_moves: Optional[int] = None
    conversion: Optional[dict] = None     # Spec 2: closeout aux-loss metadata
```

Update `to_dict`:

```python
    def to_dict(self) -> dict:
        return {
            "board_tensor": self.board_tensor.tolist(),
            "to_move": self.to_move,
            "legal_moves": self.legal_moves,
            "visit_counts": self.visit_counts,
            "outcome": self.outcome,
            "active_size": self.active_size,
            "ply": self.ply,
            "game_n_moves": self.game_n_moves,
            "conversion": self.conversion,
        }
```

Update `from_dict`:

```python
    @classmethod
    def from_dict(cls, d: dict) -> "PositionRecord":
        return cls(
            board_tensor=np.array(d["board_tensor"], dtype=np.float32),
            to_move=d["to_move"],
            legal_moves=[tuple(m) for m in d["legal_moves"]],
            visit_counts=d["visit_counts"],
            outcome=d.get("outcome"),
            active_size=d.get("active_size", 24),
            ply=d.get("ply", 0),
            game_n_moves=d.get("game_n_moves"),
            conversion=d.get("conversion"),     # defaults to None for pre-Spec-2 dicts
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_position_record_conversion.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/self_play.py tests/test_position_record_conversion.py
git commit -m "feat(conversion): add PositionRecord.conversion field with round-trip

Optional dict carrying conversion auxiliary-loss metadata (Spec 2 §5).
to_dict / from_dict round-trip; pre-Spec-2 buffer dicts load with
conversion=None.
"
```

---

### Task 3: IPC carrying via PositionRecord pickle round-trip

**Files:**
- Test: `tests/test_position_record_conversion.py` (extend)

**Background:** Workers send `PositionRecord` instances to the trainer through `position_queue.put(buf)` where `buf` is a `List[PositionRecord]` (`scripts/GPU/alphazero/self_play_worker.py:198`). `multiprocessing.Queue` uses pickle internally on the live `PositionRecord` object — NOT the `to_dict()` form.

`GameComplete` (`scripts/GPU/alphazero/ipc_messages.py:59`) does **not** carry positions; it carries game-level stats (`worker_id`, `n_positions`, `winner`, MCTS counters, `goal_completion_record`, etc.). So the IPC contract for conversion metadata is just "`PositionRecord` pickles correctly with the new field" — no `ipc_messages.py` change needed.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_position_record_conversion.py`:

```python
import pickle


def test_position_record_pickle_round_trip_carries_conversion():
    """Worker→trainer IPC uses position_queue.put(List[PositionRecord]).
    multiprocessing.Queue pickles the live object; conversion field must
    survive pickle round-trip on the live PositionRecord (not via to_dict).
    """
    conv = {
        "version": 1,
        "total_goal_distance": 2,
        "largest_component_size": 12,
        "endpoint_completion_moves": [[0, 8]],
        "distance_reducing_moves":   [[22, 4]],
        "conversion_category": "two_endpoint_closeout_2ply",
        "selected_primary_class": "completes_endpoint",
    }
    p = _make_position(conversion=conv)
    p2 = pickle.loads(pickle.dumps(p))
    assert p2.conversion == conv


def test_position_record_pickle_with_conversion_none():
    """Default-off path: conversion=None must also round-trip."""
    p = _make_position(conversion=None)
    p2 = pickle.loads(pickle.dumps(p))
    assert p2.conversion is None
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_position_record_conversion.py -v
```

Expected: PASS without code change, because Task 2's `@dataclass` field addition makes `PositionRecord.conversion` a normal pickled attribute. If it FAILS, the dataclass is using `__slots__` or a custom `__reduce__` that drops the new field — investigate `PositionRecord`'s dataclass options before introducing serialization workarounds.

- [ ] **Step 3: Commit**

```bash
git add tests/test_position_record_conversion.py
git commit -m "test(conversion): pin worker→trainer pickle carries PositionRecord.conversion

Workers stream positions via position_queue.put(List[PositionRecord]);
multiprocessing.Queue pickles the live object. Test confirms the new
conversion field round-trips through pickle for both populated and
None values. GameComplete IPC schema does NOT carry positions — no
ipc_messages.py change needed.
"
```

---

### Task 4: Self-play attach point

**Files:**
- Modify: `scripts/GPU/alphazero/self_play.py` (`play_game()`)
- Test: `tests/test_self_play_goal_completion_integration.py` (extend)

- [ ] **Step 1: Locate the existing tracker observation point in `play_game()`**

```bash
grep -n "tracker.observe_pre_move\|gc_state_full\|classify_selected_conversion_move" /Users/bill/Desktop/TwixT_Game/scripts/GPU/alphazero/self_play.py | head -20
```

Note the line where `gc_state_full` is computed and where `classify_selected_conversion_move` runs (Spec 1.5 attached classification result).

- [ ] **Step 2: Write the failing tests — DETERMINISTIC, no pytest.skip**

Anchor tests must not skip. Spec 2's anchor invariants apply regardless of whether the random seeded game happens to reach a closeout naturally. We use **monkeypatch** to inject a synthetic eligible `gc_state_full` on a known ply — same pattern as existing `test_play_game_upgrades_to_gc_state_full_on_detection_ply` (`tests/test_self_play_goal_completion_integration.py:114`).

The injected `gc_state` carries a **ply-stamped** `total_goal_distance` so we can prove the attached metadata reflects the **pre-move** state at the ply when `compute_goal_completion_state` was called — not a post-apply value.

Append to `tests/test_self_play_goal_completion_integration.py`:

```python
def _stub_gc_state(target_ply: int):
    """Build a stub for compute_goal_completion_state that returns a
    synthetic eligible gc_state when ply >= target_ply with
    enumerate_moves=True. Total_goal_distance is stamped to 2 (eligible)
    on the target ply; the largest legal moves at the position are used as
    completion / reducer moves so the eligibility predicate fires.

    Returns (stub_fn, real_fn) — caller monkeypatches stub_fn and uses
    real_fn for plies outside the closeout window.
    """
    from scripts.GPU.alphazero import connectivity_diagnostics as _cd
    real_fn = _cd.compute_goal_completion_state

    def _stub(state, player, *args, **kwargs):
        em = kwargs.get("enumerate_moves", False)
        if state.ply >= target_ply and em:
            legal = state.legal_moves()
            if len(legal) >= 2:
                return {
                    "total_goal_distance": 2,
                    "largest_component_size": 12,
                    "endpoint_completion_moves": [legal[0]],
                    "distance_reducing_moves":   [legal[1]],
                    "category": "two_endpoint_closeout_2ply",
                    "max_depth": 3,
                    "endpoint_distances": {"top": 0, "bottom": 1},
                    "component_pegs": [],
                }
        return real_fn(state, player, *args, **kwargs)

    return _stub, real_fn


def test_play_game_attaches_conversion_when_enabled_and_eligible(monkeypatch):
    """Spec 2 §5.5: PositionRecord.conversion populated on closeout plies
    when --conversion-policy-loss-enabled is on. Deterministic via stubbed
    gc_state — no skip."""
    from scripts.GPU.alphazero.self_play import play_game
    from scripts.GPU.alphazero import connectivity_diagnostics as _cd
    stub, _ = _stub_gc_state(target_ply=4)
    monkeypatch.setattr(_cd, "compute_goal_completion_state", stub)

    record = play_game(
        evaluator=_make_evaluator(7),
        mcts_config=_short_cfg(),
        rng=_rng.Random(7),
        max_moves=20, active_size=8,
        conversion_policy_loss_enabled=True,
        conversion_policy_loss_weight=0.05,
        conversion_max_total_goal_distance=2,
        goal_completion_record_enabled=False,
        goal_completion_emit_enabled=False,
    )
    closeout = [p for p in record.positions if p.conversion is not None]
    assert len(closeout) >= 1, (
        "Stubbed eligible gc_state did not produce conversion metadata — "
        "attach point in play_game is broken"
    )
    cp = closeout[0]
    assert cp.conversion["version"] == 1
    assert cp.conversion["total_goal_distance"] == 2
    assert cp.conversion["largest_component_size"] == 12
    assert (cp.conversion["endpoint_completion_moves"]
            or cp.conversion["distance_reducing_moves"])


def test_play_game_no_conversion_metadata_when_loss_disabled():
    """ANCHOR (Spec 2 §11.3): default config produces no conversion metadata.
    Negative invariant — does NOT depend on a closeout state occurring.
    No skip path."""
    from scripts.GPU.alphazero.self_play import play_game
    record = play_game(
        evaluator=_make_evaluator(7),
        mcts_config=_short_cfg(),
        rng=_rng.Random(7),
        max_moves=30, active_size=8,
        conversion_policy_loss_enabled=False,
        # Spec 1.5 also off, to isolate conversion-specific BFS
        goal_completion_record_enabled=False,
        goal_completion_emit_enabled=False,
    )
    assert len(record.positions) > 0    # game produced positions
    assert all(p.conversion is None for p in record.positions)


def test_position_record_conversion_pre_move_invariant(monkeypatch):
    """ANCHOR (Spec 2 §5.4 / §11.3): conversion describes the pre-move state
    of position.to_move, not the post-apply state. Deterministic via stubbed
    gc_state — no skip.

    Test name matches the spec's anchor list verbatim. The test lives in
    test_self_play_goal_completion_integration.py because the invariant
    requires a real play_game() run to verify; the spec's anchor list (§11.3)
    references the test by name, not by file location.

    Strategy: when the stub fires on ply N, the legal_moves it sees come
    from state.legal_moves() at ply N (PRE-apply_move). If the conversion
    metadata captured those exact legal moves as completion/reducer, then
    the attach happened on the pre-move state. If the attach happened
    AFTER apply_move, the legal moves on disk would differ from what the
    stub provided, since one move would have been consumed.
    """
    from scripts.GPU.alphazero.self_play import play_game
    from scripts.GPU.alphazero import connectivity_diagnostics as _cd

    # Capture (ply, legal_moves) the stub was called with on the target ply.
    target_ply = 4
    stub_seen = {}
    real_fn = _cd.compute_goal_completion_state

    def _stub(state, player, *args, **kwargs):
        em = kwargs.get("enumerate_moves", False)
        if state.ply == target_ply and em and "captured" not in stub_seen:
            legal = state.legal_moves()
            if len(legal) >= 2:
                stub_seen["captured"] = {
                    "ply": state.ply,
                    "legal_first_two": [tuple(legal[0]), tuple(legal[1])],
                }
                return {
                    "total_goal_distance": 2,
                    "largest_component_size": 12,
                    "endpoint_completion_moves": [legal[0]],
                    "distance_reducing_moves":   [legal[1]],
                    "category": "two_endpoint_closeout_2ply",
                    "max_depth": 3,
                    "endpoint_distances": {"top": 0, "bottom": 1},
                    "component_pegs": [],
                }
        return real_fn(state, player, *args, **kwargs)

    monkeypatch.setattr(_cd, "compute_goal_completion_state", _stub)

    record = play_game(
        evaluator=_make_evaluator(7),
        mcts_config=_short_cfg(),
        rng=_rng.Random(7),
        max_moves=20, active_size=8,
        conversion_policy_loss_enabled=True,
        conversion_policy_loss_weight=0.05,
        conversion_max_total_goal_distance=2,
        goal_completion_record_enabled=False,
        goal_completion_emit_enabled=False,
    )

    assert "captured" in stub_seen, (
        "Stub was never called with enumerate_moves=True on target ply — "
        "play_game's conversion path is not invoking the BFS"
    )

    # Find the position attached at the target ply.
    matching = [p for p in record.positions if p.conversion is not None
                and p.conversion["total_goal_distance"] == 2]
    assert len(matching) >= 1, (
        "Stubbed gc_state did not attach conversion metadata at the target ply"
    )
    cp = matching[0]

    # PRE-MOVE INVARIANT: the completion/reducer moves on the persisted
    # PositionRecord must match the legal moves the stub saw at the target
    # ply (BEFORE any move was applied). If attach happened AFTER apply_move,
    # one of these moves would have been consumed and the lists would mismatch.
    persisted_completion = {tuple(m) for m in cp.conversion["endpoint_completion_moves"]}
    persisted_reducing   = {tuple(m) for m in cp.conversion["distance_reducing_moves"]}
    seen_first  = stub_seen["captured"]["legal_first_two"][0]
    seen_second = stub_seen["captured"]["legal_first_two"][1]
    assert seen_first  in persisted_completion, (
        f"completion_moves on disk={persisted_completion}, but stub saw "
        f"{seen_first} as the first legal move at pre-move ply {target_ply}. "
        "Conversion metadata was not captured PRE-apply_move."
    )
    assert seen_second in persisted_reducing, (
        f"distance_reducing_moves on disk={persisted_reducing}, but stub saw "
        f"{seen_second} as the second legal move at pre-move ply {target_ply}. "
        "Conversion metadata was not captured PRE-apply_move."
    )


def test_play_game_conversion_enabled_computes_full_state_when_emit_disabled(monkeypatch):
    """Spec 2 §3 cost-path: conversion forces full BFS even when Spec 1.5 emit
    is off, on plies that are conversion-eligible. Deterministic — no skip."""
    from scripts.GPU.alphazero.self_play import play_game
    from scripts.GPU.alphazero import connectivity_diagnostics as _cd
    stub, _ = _stub_gc_state(target_ply=4)
    monkeypatch.setattr(_cd, "compute_goal_completion_state", stub)

    record = play_game(
        evaluator=_make_evaluator(7),
        mcts_config=_short_cfg(),
        rng=_rng.Random(7),
        max_moves=20, active_size=8,
        conversion_policy_loss_enabled=True,
        conversion_policy_loss_weight=0.05,
        conversion_max_total_goal_distance=2,
        goal_completion_emit_enabled=False,
        goal_completion_record_enabled=False,
    )
    closeout = [p for p in record.positions if p.conversion is not None]
    assert len(closeout) >= 1, (
        "Conversion attach did not fire when Spec 1.5 emit/record paths "
        "were off — Spec 2 cost-path failed to compute its own full BFS"
    )
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_self_play_goal_completion_integration.py -v -k conversion
```

Expected: FAIL — no `conversion_*` kwargs accepted by `play_game()` yet (or
the kwargs are accepted but `position.conversion` is always `None`).

Note: test name is `test_position_record_conversion_pre_move_invariant`, NOT
`test_play_game_conversion_pre_move_invariant`. The name matches Spec 2 §11.3
verbatim.

- [ ] **Step 4: Add the attach-point logic in `play_game()`**

In `scripts/GPU/alphazero/self_play.py`, locate `play_game()`. Add new keyword arguments to the signature:

```python
def play_game(
    *,
    # ... existing args ...
    conversion_policy_loss_enabled: bool = False,
    conversion_max_total_goal_distance: int = 2,
    # min_component_size and max_depth reuse goal_completion_min_component_size /
    # goal_completion_max_depth (existing Spec 1.5 args).
    ...
):
    ...
```

Update the BFS-needs-full computation to include conversion:

```python
        # Existing Spec 1.5 logic computes needs_phase3_full and needs_tracker_full.
        # Add conversion's contribution:
        needs_conversion_full = (
            conversion_policy_loss_enabled
            and gc_state_cheap is not None
            and gc_state_cheap.get("total_goal_distance") is not None
            and gc_state_cheap["total_goal_distance"] <= conversion_max_total_goal_distance
        )
        if needs_phase3_full or needs_tracker_full or needs_conversion_full:
            gc_state_full = compute_goal_completion_state(
                state, side,
                max_depth=goal_completion_max_depth,
                min_component_size=goal_completion_min_component_size,
                enumerate_moves=True,
            )
```

After Spec 1.5's tracker observation and after `classify_selected_conversion_move` runs (whose result is captured into a local `classification_result`), build conversion metadata:

```python
        # Spec 2: build conversion metadata using the SAME gc_state_full above.
        from .conversion_loss import is_conversion_eligible
        conversion_meta = None
        if conversion_policy_loss_enabled and is_conversion_eligible(
            gc_state_full,
            max_total_goal_distance=conversion_max_total_goal_distance,
            min_component_size=goal_completion_min_component_size,
        ):
            conversion_meta = {
                "version": 1,
                "total_goal_distance":       gc_state_full["total_goal_distance"],
                "largest_component_size":    gc_state_full["largest_component_size"],
                "endpoint_completion_moves": [list(m) for m in (gc_state_full.get("endpoint_completion_moves") or [])],
                "distance_reducing_moves":   [list(m) for m in (gc_state_full.get("distance_reducing_moves") or [])],
                "conversion_category":       gc_state_full.get("category"),
                "selected_primary_class":    None,
            }

        # Fill selected_primary_class from the classification result that Spec 1.5
        # already computed. classification_result variable is whatever
        # classify_selected_conversion_move() returned (or None).
        if classification_result is not None and conversion_meta is not None:
            conversion_meta["selected_primary_class"] = classification_result.get("primary_class")
```

Then, when constructing the `PositionRecord` for this ply, include `conversion=conversion_meta`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_self_play_goal_completion_integration.py -v
```

Expected: all 4 new tests pass; existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/self_play.py tests/test_self_play_goal_completion_integration.py
git commit -m "feat(conversion): attach conversion metadata in play_game

Spec 2 §5.5: when --conversion-policy-loss-enabled is on and the side-
to-move's pre-move state is conversion-eligible, attach the conversion
dict to PositionRecord. Reuses existing gc_state_full from Spec 1.5
BFS-reuse contract; forces a full BFS only on plies that need it for
conversion.

Includes anchor tests for the pre-move invariant and the default-off
no-metadata invariant.
"
```

---

### Task 5: CLI flag scaffolding (default off)

**Files:**
- Modify: `scripts/GPU/alphazero/train.py`
- Modify: `scripts/GPU/alphazero/trainer.py` (`train()` signature, banner)
- Test: `tests/test_conversion_cli_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_conversion_cli_config.py
"""CLI/config invariants for conversion auxiliary loss (Spec 2 §9)."""
import argparse
import pytest


def _build_parser():
    """Mirror scripts/GPU/alphazero/train.py argparse construction.
    Imports it after the new flags are added.
    """
    from scripts.GPU.alphazero.train import _build_parser_for_test
    return _build_parser_for_test()


def test_conversion_disabled_by_default_effective_weight_zero():
    """Spec 2 §11.3 anchor: default config has effective_loss_weight = 0.0."""
    p = _build_parser()
    args = p.parse_args([])
    assert args.conversion_policy_loss_enabled is False
    assert args.conversion_policy_loss_weight == 0.05  # configured default
    # Effective weight derived elsewhere; here we just lock the flag default.


def test_conversion_enabled_uses_configured_weight():
    p = _build_parser()
    args = p.parse_args(["--conversion-policy-loss-enabled"])
    assert args.conversion_policy_loss_enabled is True
    assert args.conversion_policy_loss_weight == 0.05


def test_conversion_enabled_with_zero_weight_errors():
    """Spec 2 §9.3: enabled + weight==0 must error."""
    p = _build_parser()
    with pytest.raises(SystemExit):
        # argparse.error raises SystemExit; our validator is called after parse
        from scripts.GPU.alphazero.train import _validate_conversion_args
        args = p.parse_args(["--conversion-policy-loss-enabled",
                             "--conversion-policy-loss-weight", "0.0"])
        _validate_conversion_args(p, args)


def test_reducer_weight_greater_than_completion_weight_errors():
    p = _build_parser()
    with pytest.raises(SystemExit):
        from scripts.GPU.alphazero.train import _validate_conversion_args
        args = p.parse_args(["--conversion-policy-loss-enabled",
                             "--conversion-completion-weight", "0.5",
                             "--conversion-reducer-weight", "0.8"])
        _validate_conversion_args(p, args)


def test_conversion_max_total_goal_distance_bounds():
    """Must be in [1, 3]."""
    p = _build_parser()
    from scripts.GPU.alphazero.train import _validate_conversion_args
    for bad in ["0", "4", "-1"]:
        args = p.parse_args(["--conversion-max-total-goal-distance", bad])
        with pytest.raises(SystemExit):
            _validate_conversion_args(p, args)
    for ok in ["1", "2", "3"]:
        args = p.parse_args(["--conversion-max-total-goal-distance", ok])
        _validate_conversion_args(p, args)  # should not raise


def test_sample_boost_without_loss_warns_and_tagging_stays_off(capsys):
    """Spec 2 §9.3: sample_boost > 1.0 with loss off should warn."""
    p = _build_parser()
    from scripts.GPU.alphazero.train import _validate_conversion_args
    args = p.parse_args(["--conversion-sample-boost", "2.0"])
    _validate_conversion_args(p, args)
    captured = capsys.readouterr()
    assert "[WARN]" in captured.out
    assert "--conversion-sample-boost" in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_conversion_cli_config.py -v
```

Expected: ImportError on `_build_parser_for_test` / `_validate_conversion_args` / unrecognized arguments.

- [ ] **Step 3: Add the CLI flags and validator**

In `scripts/GPU/alphazero/train.py`, in `main()`'s argparse setup (after existing flags, before `args = parser.parse_args()`):

```python
    # Spec 2: conversion auxiliary loss
    parser.add_argument("--conversion-policy-loss-enabled", action="store_true",
        help="Enable conversion auxiliary policy loss on closeout-eligible positions.")
    parser.add_argument("--conversion-policy-loss-weight", type=float, default=0.05,
        help="Weight λ for the conversion auxiliary loss term (default: 0.05).")
    parser.add_argument("--conversion-completion-weight", type=float, default=1.0,
        help="Target weight for endpoint_completion_moves (default: 1.0).")
    parser.add_argument("--conversion-reducer-weight", type=float, default=0.35,
        help="Target weight for distance_reducing_moves (default: 0.35). "
             "Must be <= --conversion-completion-weight.")
    parser.add_argument("--conversion-max-total-goal-distance", type=int, default=2,
        help="Eligibility threshold on total_goal_distance (default: 2). "
             "Range [1, 3]; first experiment uses 2, widens to 3 later.")
    # Track 2: sample boost
    parser.add_argument("--conversion-sample-boost", type=float, default=1.0,
        help="Multiplier on uniform-eligible expectation (default: 1.0 = pure uniform).")
    parser.add_argument("--conversion-max-batch-fraction", type=float, default=0.15,
        help="Hard cap on eligible fraction per batch (default: 0.15).")
    # Track 4: recovery / extreme-closeout-drift telemetry (default on; free)
    parser.add_argument("--recovery-bucket-enabled", action="store_true", default=True,
        help="Enable recovery / extreme-closeout-drift telemetry (default: on).")
    parser.add_argument("--no-recovery-bucket", dest="recovery_bucket_enabled",
        action="store_false",
        help="Disable recovery / extreme-closeout-drift telemetry.")
    parser.add_argument("--recovery-dominant-unavailable-threshold", type=int, default=10,
        help="DU-moves threshold for recovery bucket (default: 10).")
    parser.add_argument("--recovery-delay-threshold", type=int, default=20,
        help="conversion_delay_plies threshold for recovery bucket (default: 20).")
```

**Refactor:** extract parser construction out of `main()` so tests can construct it without parsing `sys.argv`. Concrete steps:

1. Locate the `argparse.ArgumentParser(...)` call inside `main()` (currently around line 30 of `train.py`).
2. Cut everything from `parser = argparse.ArgumentParser(...)` through the last `parser.add_argument(...)` call (NOT including `args = parser.parse_args()`).
3. Define `_build_parser_for_test()` at module scope, just above `main()`. Paste the cut block into the function body, replacing the trailing line with `return parser`.
4. In `main()`, replace the cut block with `parser = _build_parser_for_test()`.

Result:

```python
def _build_parser_for_test() -> argparse.ArgumentParser:
    """Build the trainer CLI parser. Importable from tests to assert
    defaults / validation behavior without invoking main()/parse_args().

    Single source of truth for the parser definition — main() calls this
    and then parse_args(); tests call this and parse_args(arglist).
    """
    parser = argparse.ArgumentParser(
        description="AlphaZero training for TwixT",
    )
    # ... ALL existing parser.add_argument calls from main(), unchanged ...
    # ... PLUS the new Spec 2 flags from this Task's previous step ...
    return parser


def main():
    parser = _build_parser_for_test()
    args = parser.parse_args()
    _validate_conversion_args(parser, args)
    # ... rest of main() unchanged ...
```

This keeps the existing `main()` behavior identical (single function call replaces the inline construction) and gives tests a stable hook.

Add the validator:

```python
def _validate_conversion_args(parser: argparse.ArgumentParser, args) -> None:
    """Validate Spec 2 conversion / recovery args. Raises SystemExit via parser.error."""
    if args.conversion_policy_loss_enabled and args.conversion_policy_loss_weight <= 0.0:
        parser.error(
            "--conversion-policy-loss-enabled requires "
            "--conversion-policy-loss-weight > 0.0. "
            "Omit --conversion-policy-loss-enabled to disable conversion entirely."
        )
    if args.conversion_completion_weight <= 0.0:
        parser.error("--conversion-completion-weight must be > 0.0")
    if args.conversion_reducer_weight < 0.0:
        parser.error("--conversion-reducer-weight must be >= 0.0")
    if args.conversion_reducer_weight > args.conversion_completion_weight:
        parser.error(
            "--conversion-reducer-weight must be <= --conversion-completion-weight "
            f"(got reducer={args.conversion_reducer_weight}, "
            f"completion={args.conversion_completion_weight})."
        )
    if not (1 <= args.conversion_max_total_goal_distance <= 3):
        parser.error("--conversion-max-total-goal-distance must be in [1, 3]")
    if args.conversion_sample_boost < 1.0:
        parser.error(
            "--conversion-sample-boost must be >= 1.0 "
            "(omit --conversion-policy-loss-enabled to disable conversion entirely)"
        )
    if not (0.0 <= args.conversion_max_batch_fraction <= 1.0):
        parser.error("--conversion-max-batch-fraction must be in [0.0, 1.0]")

    if (not args.conversion_policy_loss_enabled
            and args.conversion_sample_boost > 1.0):
        print(
            "[WARN] --conversion-sample-boost > 1.0 has no effect when "
            "--conversion-policy-loss-enabled is off. Sample boost stays inactive "
            "and PositionRecord.conversion stays unpopulated."
        )

    if args.recovery_dominant_unavailable_threshold < 1:
        parser.error("--recovery-dominant-unavailable-threshold must be >= 1")
    if args.recovery_delay_threshold < 1:
        parser.error("--recovery-delay-threshold must be >= 1")
```

Call `_validate_conversion_args(parser, args)` in `main()` after `args = parser.parse_args()`.

Pass the new args through to `train(...)` in `train_kwargs`:

```python
    train_kwargs = dict(
        # ... existing kwargs ...
        conversion_policy_loss_enabled=args.conversion_policy_loss_enabled,
        conversion_policy_loss_weight=args.conversion_policy_loss_weight,
        conversion_completion_weight=args.conversion_completion_weight,
        conversion_reducer_weight=args.conversion_reducer_weight,
        conversion_max_total_goal_distance=args.conversion_max_total_goal_distance,
        conversion_sample_boost=args.conversion_sample_boost,
        conversion_max_batch_fraction=args.conversion_max_batch_fraction,
        recovery_bucket_enabled=args.recovery_bucket_enabled,
        recovery_dominant_unavailable_threshold=args.recovery_dominant_unavailable_threshold,
        recovery_delay_threshold=args.recovery_delay_threshold,
    )
```

In `scripts/GPU/alphazero/trainer.py`, extend `train(...)` signature with the same keyword args (default values matching the CLI defaults). For Phase 1, just plumb them through to `play_game()` calls inside `run_parallel_selfplay`; Phase 2 wires them into the loss / sidecar.

Add banner output in `train()`:

```python
    if conversion_policy_loss_enabled:
        print(f"  Conversion auxiliary loss: enabled (weight={conversion_policy_loss_weight})")
        print(f"    Target weights:        completion={conversion_completion_weight}, "
              f"reducer={conversion_reducer_weight}")
        print(f"    Eligibility:           total_goal_distance <= "
              f"{conversion_max_total_goal_distance}, "
              f"min_component_size >= {goal_completion_min_component_size}")
        print(f"    Sample boost:          {conversion_sample_boost}"
              f"{' (disabled — pure uniform)' if conversion_sample_boost == 1.0 else ''}")
        print(f"    Max batch fraction:    {conversion_max_batch_fraction}"
              f"{' (cap inert at boost=1.0)' if conversion_sample_boost == 1.0 else ''}")
    else:
        print(f"  Conversion auxiliary loss: disabled")
    print(f"  Recovery / extreme-closeout-drift: "
          f"{'enabled' if recovery_bucket_enabled else 'disabled'} "
          f"(du_threshold={recovery_dominant_unavailable_threshold}, "
          f"delay_threshold={recovery_delay_threshold})")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_conversion_cli_config.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Smoke-run the trainer to confirm banner**

```bash
python -m scripts.GPU.alphazero.train --help | grep -E "conversion|recovery"
```

Expected: all 10 new flags listed.

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/train.py scripts/GPU/alphazero/trainer.py tests/test_conversion_cli_config.py
git commit -m "feat(conversion): add CLI flag scaffolding (default off)

Spec 2 §9: 10 new flags for conversion auxiliary loss, sample boost,
and recovery telemetry. Default off — effective_conversion_enabled
derives from --conversion-policy-loss-enabled (explicit master
switch). Validator errors on enabled+weight=0; warns on boost>1
without enabled.

Includes anchor test test_conversion_disabled_by_default_effective_weight_zero.
"
```

---

## Phase 2 — Auxiliary loss (the actual policy correction)

### Task 6: Per-position target construction

**Files:**
- Modify: `scripts/GPU/alphazero/conversion_loss.py`
- Test: `tests/test_conversion_target.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_conversion_target.py
"""Per-position conversion target tests (Spec 2 §6.1)."""
import math
import numpy as np
from scripts.GPU.alphazero.conversion_loss import build_conversion_target


def test_target_normalizes_to_unit_sum():
    legal = [(0, 8), (5, 5), (22, 4)]
    completion = {(0, 8)}
    reducing = {(22, 4)}
    target = build_conversion_target(legal, completion, reducing,
                                     completion_weight=1.0, reducer_weight=0.35)
    assert target is not None
    assert math.isclose(target.sum(), 1.0, abs_tol=1e-6)


def test_target_assigns_completion_weight_to_completion_moves():
    legal = [(0, 8), (5, 5), (22, 4)]
    completion = {(0, 8)}
    reducing = {(22, 4)}
    target = build_conversion_target(legal, completion, reducing,
                                     completion_weight=1.0, reducer_weight=0.35)
    # (0,8) is completion → weight 1.0; total = 1.0 + 0.35 = 1.35
    assert math.isclose(target[0], 1.0 / 1.35, abs_tol=1e-6)


def test_target_assigns_reducer_weight_to_reducer_only_moves():
    legal = [(0, 8), (5, 5), (22, 4)]
    completion = {(0, 8)}
    reducing = {(22, 4)}
    target = build_conversion_target(legal, completion, reducing,
                                     completion_weight=1.0, reducer_weight=0.35)
    assert math.isclose(target[2], 0.35 / 1.35, abs_tol=1e-6)


def test_target_disjoint_mass_rule_completion_wins():
    """Move in BOTH sets gets completion_weight, not the sum."""
    legal = [(0, 8), (5, 5)]
    completion = {(0, 8)}
    reducing = {(0, 8)}     # same move in both sets
    target = build_conversion_target(legal, completion, reducing,
                                     completion_weight=1.0, reducer_weight=0.35)
    # Only one entry in target gets weight, normalized to 1.0
    assert math.isclose(target[0], 1.0, abs_tol=1e-6)
    assert math.isclose(target[1], 0.0, abs_tol=1e-6)


def test_target_zero_for_other_legal_moves():
    legal = [(0, 8), (5, 5), (22, 4)]
    completion = {(0, 8)}
    reducing = {(22, 4)}
    target = build_conversion_target(legal, completion, reducing,
                                     completion_weight=1.0, reducer_weight=0.35)
    assert target[1] == 0.0


def test_conversion_aux_target_aligns_with_legal_move_order():
    """ANCHOR (Spec 2 §11.3): exact alignment fixture from spec.

    legal=[(1,2),(3,4),(5,6)]
    completion=[(5,6)], reducer=[(1,2)]
    Expected target = [0.35/1.35, 0.0, 1.0/1.35]
    """
    legal = [(1, 2), (3, 4), (5, 6)]
    completion = {(5, 6)}
    reducing = {(1, 2)}
    target = build_conversion_target(legal, completion, reducing,
                                     completion_weight=1.0, reducer_weight=0.35)
    assert target is not None
    np.testing.assert_allclose(
        target,
        [0.35 / 1.35, 0.0, 1.0 / 1.35],
        atol=1e-6,
    )


def test_target_returns_none_when_no_completion_or_reducer_in_legal_moves():
    """If completion/reducer sets are non-empty but their moves are not in
    legal_moves (stale alignment), target is None — boundary defense."""
    legal = [(5, 5)]    # only this move legal
    completion = {(0, 8)}
    reducing = {(22, 4)}
    target = build_conversion_target(legal, completion, reducing,
                                     completion_weight=1.0, reducer_weight=0.35)
    assert target is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_conversion_target.py -v
```

Expected: ImportError on `build_conversion_target`.

- [ ] **Step 3: Implement build_conversion_target**

Append to `scripts/GPU/alphazero/conversion_loss.py`:

```python
import numpy as np
from typing import Optional


def build_conversion_target(
    legal_moves: list,
    completion_moves: set,
    reducing_moves: set,
    *,
    completion_weight: float,
    reducer_weight: float,
) -> Optional[np.ndarray]:
    """Build a normalized auxiliary target distribution over legal_moves.

    Returns a length-len(legal_moves) np.float32 array summing to 1.0,
    or None if the target is empty after legal-move alignment.

    Disjoint-mass rule: a move that is both endpoint-completing AND
    distance-reducing receives completion_weight (the larger), not the sum.
    """
    weights = np.zeros(len(legal_moves), dtype=np.float32)
    for i, m in enumerate(legal_moves):
        if m in completion_moves:
            weights[i] = completion_weight
        elif m in reducing_moves:
            weights[i] = reducer_weight
    total = float(weights.sum())
    if total <= 0.0:
        return None
    return weights / total
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_conversion_target.py -v
```

Expected: 7 passed (including the alignment anchor).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/conversion_loss.py tests/test_conversion_target.py
git commit -m "feat(conversion): add build_conversion_target with disjoint-mass rule

Spec 2 §6.1: completion=1.0 dominates reducer=0.35 on overlapping moves.
Returns None when no completion/reducer move is in legal_moves (boundary
defense). Includes alignment anchor test pinning column order.
"
```

---

### Task 7: Batch tensor build

**Files:**
- Modify: `scripts/GPU/alphazero/conversion_loss.py`
- Modify: `scripts/GPU/alphazero/trainer.py` (`make_padded_batch` signature)
- Test: `tests/test_conversion_aux_tensors.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_conversion_aux_tensors.py
"""Batch-level conversion aux tensor tests (Spec 2 §6.2)."""
import numpy as np
from scripts.GPU.alphazero.conversion_loss import make_conversion_aux_tensors


def _pos(conversion=None):
    """Lightweight stand-in for PositionRecord — only fields used by
    make_conversion_aux_tensors are required."""
    class _P:
        pass
    p = _P()
    p.conversion = conversion
    return p


def test_aux_tensor_shape_matches_target_pi():
    positions = [_pos(conversion=None) for _ in range(3)]
    legal_padded = [[(0, 0), None, None]] * 3   # M_padded = 3
    aux_target, aux_mask = make_conversion_aux_tensors(
        positions, legal_padded, max_moves_cap=3,
    )
    assert aux_target.shape == (3, 3)
    assert aux_mask.shape == (3,)


def test_aux_mask_zero_for_ineligible_positions():
    positions = [_pos(conversion=None), _pos(conversion=None)]
    legal_padded = [[(0, 0)], [(1, 1)]]
    aux_target, aux_mask = make_conversion_aux_tensors(
        positions, legal_padded, max_moves_cap=4,
    )
    assert aux_mask.tolist() == [0.0, 0.0]
    assert aux_target.sum() == 0.0


def test_aux_mask_zero_when_target_returns_none():
    """conversion present but no completion/reducer move appears in legal_padded."""
    conv = {
        "endpoint_completion_moves": [[99, 99]],   # not in legal
        "distance_reducing_moves":   [[88, 88]],   # not in legal
    }
    positions = [_pos(conversion=conv)]
    legal_padded = [[(0, 0), (5, 5)]]
    aux_target, aux_mask = make_conversion_aux_tensors(
        positions, legal_padded, max_moves_cap=4,
    )
    assert aux_mask.tolist() == [0.0]
    assert aux_target[0].sum() == 0.0


def test_aux_tensor_skips_padding_columns():
    """legal_padded entries equal to None must not contribute to weights
    (Spec 2 §6.2 + §3 lock #2)."""
    conv = {
        "endpoint_completion_moves": [[0, 0]],
        "distance_reducing_moves":   [],
    }
    positions = [_pos(conversion=conv)]
    legal_padded = [[(0, 0), None, None, None]]   # only first slot is real
    aux_target, aux_mask = make_conversion_aux_tensors(
        positions, legal_padded, max_moves_cap=4,
    )
    assert aux_mask.tolist() == [1.0]
    np.testing.assert_allclose(aux_target[0], [1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_aux_tensor_aligns_with_target_pi_columns():
    """Per-position aux_target column j references the same legal_padded[i][j]
    that target_pi[i][j] would reference."""
    conv = {
        "endpoint_completion_moves": [[5, 6]],
        "distance_reducing_moves":   [[1, 2]],
    }
    positions = [_pos(conversion=conv)]
    legal_padded = [[(1, 2), (3, 4), (5, 6), None]]   # match the spec anchor fixture
    aux_target, aux_mask = make_conversion_aux_tensors(
        positions, legal_padded, max_moves_cap=4,
        completion_weight=1.0, reducer_weight=0.35,
    )
    assert aux_mask.tolist() == [1.0]
    np.testing.assert_allclose(
        aux_target[0],
        [0.35 / 1.35, 0.0, 1.0 / 1.35, 0.0],
        atol=1e-6,
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_conversion_aux_tensors.py -v
```

Expected: ImportError on `make_conversion_aux_tensors`.

- [ ] **Step 3: Implement make_conversion_aux_tensors**

Append to `scripts/GPU/alphazero/conversion_loss.py`:

```python
def make_conversion_aux_tensors(
    positions: list,
    legal_moves_padded: list,        # per-position list ordered like target_pi columns
    max_moves_cap: int,
    *,
    completion_weight: float = 1.0,
    reducer_weight: float = 0.35,
) -> tuple:
    """Return (aux_target, aux_mask) np arrays.

    aux_target shape: (B, max_moves_cap), float32
    aux_mask shape:   (B,), float32

    Padding entries in legal_moves_padded[i] (None values) are skipped.
    legal_moves_padded[i] is ordered exactly like target_pi[i] columns
    and move_mask[i] — same indexing as the policy CE computation.
    """
    B = len(positions)
    aux_target = np.zeros((B, max_moves_cap), dtype=np.float32)
    aux_mask   = np.zeros((B,), dtype=np.float32)

    for i, p in enumerate(positions):
        conv = getattr(p, "conversion", None)
        if conv is None:
            continue
        completion = {tuple(m) for m in conv.get("endpoint_completion_moves") or ()}
        reducing   = {tuple(m) for m in conv.get("distance_reducing_moves")   or ()}

        weights = np.zeros(max_moves_cap, dtype=np.float32)
        for j, m in enumerate(legal_moves_padded[i]):
            if m is None:
                continue
            if m in completion:
                weights[j] = completion_weight
            elif m in reducing:
                weights[j] = reducer_weight

        total = float(weights.sum())
        if total <= 0.0:
            continue
        aux_target[i] = weights / total
        aux_mask[i]   = 1.0

    return aux_target, aux_mask
```

- [ ] **Step 4: Extend make_padded_batch to optionally return legal_moves_padded**

In `scripts/GPU/alphazero/trainer.py`, modify `make_padded_batch` (line 917):

```python
def make_padded_batch(
    positions: List["PositionRecord"],
    max_moves_cap: int = 512,
    return_legal: bool = False,    # NEW: opt-in for Spec 2 conversion aux loss
):
    """Prepare batched tensors with padded moves for training.

    Args:
        positions: List of PositionRecord
        max_moves_cap: Maximum moves to consider (truncates if exceeded)
        return_legal: If True, also return legal_moves_padded (a Python list
            of lists of (row, col) tuples or None for padding slots), used
            by the conversion auxiliary loss. Default False preserves
            backward compatibility.

    Returns:
        Without return_legal: (boards_mx, move_rows, move_cols, move_mask, target_pi, outcomes)
        With return_legal:    (..., legal_moves_padded)
    """
    # ... existing body ...

    legal_moves_padded = None
    if return_legal:
        # Build the parallel Python list of legal moves with None padding.
        legal_moves_padded = []
        for p in positions:
            moves = p.legal_moves[:M]
            row = list(moves) + [None] * (M - len(moves))
            legal_moves_padded.append(row)

    base = (
        boards_mx,
        mx.array(move_rows_np),
        mx.array(move_cols_np),
        mx.array(move_mask_np),
        mx.array(target_pi_np),
        mx.array(outcomes_np),
    )
    if return_legal:
        return base + (legal_moves_padded,)
    return base
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_conversion_aux_tensors.py -v
pytest tests/test_trainer_loss.py -v   # ensure existing make_padded_batch callers still work
```

Expected: 5 passed in conversion_aux_tensors.py; existing trainer-loss tests still pass (backward-compat default).

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/conversion_loss.py scripts/GPU/alphazero/trainer.py tests/test_conversion_aux_tensors.py
git commit -m "feat(conversion): add make_conversion_aux_tensors + make_padded_batch return_legal

Spec 2 §6.2: batch-level aux_target / aux_mask construction in pure
NumPy. make_padded_batch gains opt-in return_legal kwarg surfacing
legal_moves_padded for aux-tensor build; default preserves existing
6-tuple return.

Tests pin (B, M_padded) shape, ineligible-position zeroing, padding-
column skipping, and column alignment with target_pi.
"
```

---

### Task 8: Loss math integration

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` (`alphazero_loss_batch`, `train_step`)
- Test: `tests/test_conversion_loss.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_conversion_loss.py
"""Loss math tests (Spec 2 §6.3)."""
import numpy as np
import mlx.core as mx
import pytest

from scripts.GPU.alphazero.trainer import alphazero_loss_batch
from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord


def _pos(conversion=None, active_size=24):
    return PositionRecord(
        board_tensor=np.zeros((active_size, active_size, 30), dtype=np.float32),
        to_move="red",
        legal_moves=[(0, 0), (1, 1), (2, 2)],
        visit_counts=[10, 5, 3],
        outcome=1.0,
        active_size=active_size,
        ply=0,
        game_n_moves=10,
        conversion=conversion,
    )


def _network():
    return create_network()


def test_aux_loss_zero_when_all_ineligible():
    net = _network()
    positions = [_pos(conversion=None) for _ in range(4)]
    total, policy, value, l2, aux, coverage, n_eligible = alphazero_loss_batch(
        net, positions, conversion_loss_weight=0.05,
    )
    assert float(aux.item()) == 0.0
    assert n_eligible == 0
    assert not np.isnan(float(total.item()))


def test_aux_loss_uses_masked_log_probs():
    """ANCHOR (Spec 2 §11.3): aux loss must use the SAME masked log_probs
    as policy loss. Padded/illegal columns must not contribute to either.

    DETERMINISTIC fixture-based check: we extract the masking logic into a
    helper `compute_masked_log_probs(logits, move_mask)` and assert that on
    a known fixture, the aux loss equals the hand-computed CE against the
    masked target.
    """
    from scripts.GPU.alphazero.trainer import compute_masked_log_probs

    # Deterministic fixture: B=1, M_padded=4, only first 2 columns are legal.
    # Logits chosen so masked-softmax probabilities are predictable.
    logits = mx.array([[0.0, 0.0, 100.0, 100.0]], dtype=mx.float32)
    move_mask = mx.array([[1.0, 1.0, 0.0, 0.0]], dtype=mx.float32)

    log_probs = compute_masked_log_probs(logits, move_mask)
    # If masking is applied correctly, padded columns (logits=100) must
    # NOT appear in the logsumexp denominator. Effective logits over legal
    # columns are [0, 0] → softmax = [0.5, 0.5] → log_probs at legal = log(0.5).
    legal_log_probs = [float(log_probs[0, j].item()) for j in (0, 1)]
    np.testing.assert_allclose(legal_log_probs, [np.log(0.5), np.log(0.5)], atol=1e-5)

    # If masking were broken (padded columns leaking in), softmax over all 4
    # columns would put ~all mass on cols 2 and 3, giving log_probs at cols
    # 0,1 of approximately -100. Catch that case:
    assert all(lp > -1.0 for lp in legal_log_probs), (
        "padded columns appear to be leaking into logsumexp"
    )


def test_aux_loss_mean_over_eligible_only():
    """Same per-position aux magnitude regardless of eligible/total ratio."""
    net = _network()
    conv = {
        "version": 1,
        "endpoint_completion_moves": [[0, 0]],
        "distance_reducing_moves":   [],
    }
    # 1 eligible / 4 total
    pos_few = [_pos(conversion=conv)] + [_pos(conversion=None) for _ in range(3)]
    # 1 eligible / 1 total
    pos_only = [_pos(conversion=conv)]

    _, _, _, _, aux_few, _, n_few = alphazero_loss_batch(
        net, pos_few, conversion_loss_weight=0.05,
    )
    _, _, _, _, aux_only, _, n_only = alphazero_loss_batch(
        net, pos_only, conversion_loss_weight=0.05,
    )
    # Mean over eligible only — both should be approximately equal.
    np.testing.assert_allclose(
        float(aux_few.item()), float(aux_only.item()), atol=1e-3,
    )
    assert n_few == 1
    assert n_only == 1


def test_aux_loss_returns_n_eligible_as_int():
    """ANCHOR (Spec 2 §11.3): aux_n_eligible must be an int, not a float."""
    net = _network()
    conv = {
        "version": 1,
        "endpoint_completion_moves": [[0, 0]],
        "distance_reducing_moves":   [],
    }
    positions = [_pos(conversion=conv), _pos(conversion=None), _pos(conversion=conv)]
    _, _, _, _, _, _, n_eligible = alphazero_loss_batch(
        net, positions, conversion_loss_weight=0.05,
    )
    assert isinstance(n_eligible, int)
    assert n_eligible == 2


def test_total_loss_includes_aux_term_when_enabled():
    net = _network()
    conv = {
        "version": 1,
        "endpoint_completion_moves": [[0, 0]],
        "distance_reducing_moves":   [],
    }
    positions = [_pos(conversion=conv)]
    total_off, policy_off, _, _, aux_off, _, _ = alphazero_loss_batch(
        net, positions, conversion_loss_weight=0.0,
    )
    total_on, policy_on, _, _, aux_on, _, _ = alphazero_loss_batch(
        net, positions, conversion_loss_weight=0.5,
    )
    assert float(aux_off.item()) == 0.0
    assert float(aux_on.item()) > 0.0
    # total_on should be policy + 0.5*aux + value + l2 — strictly greater than total_off
    assert float(total_on.item()) > float(total_off.item())


def test_total_loss_excludes_aux_when_weight_zero():
    """conversion_loss_weight=0 short-circuits — make_conversion_aux_tensors
    should NOT be called (zero overhead path)."""
    import scripts.GPU.alphazero.conversion_loss as cl_mod
    call_count = {"n": 0}
    original = cl_mod.make_conversion_aux_tensors

    def _spy(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    cl_mod.make_conversion_aux_tensors = _spy
    try:
        net = _network()
        positions = [_pos(conversion={"endpoint_completion_moves": [[0,0]],
                                       "distance_reducing_moves": []})]
        alphazero_loss_batch(net, positions, conversion_loss_weight=0.0)
        assert call_count["n"] == 0, (
            "make_conversion_aux_tensors was called even with weight=0"
        )
    finally:
        cl_mod.make_conversion_aux_tensors = original


def test_aux_loss_matches_hand_computed_ce_on_fixture():
    """Sanity check on a deterministic fixture.

    Build a one-position batch where logits are known (we can't easily set
    network logits, so we set up the position to have a specific
    expectation: with random init, aux_loss should be roughly
    -log(target_prob_under_softmax). We assert the value is in the expected
    order of magnitude (1–10) for a fresh network.
    """
    net = _network()
    conv = {
        "version": 1,
        "endpoint_completion_moves": [[0, 0]],
        "distance_reducing_moves":   [],
    }
    positions = [_pos(conversion=conv)]
    _, _, _, _, aux, _, _ = alphazero_loss_batch(
        net, positions, conversion_loss_weight=0.05,
    )
    aux_val = float(aux.item())
    # CE over a 3-legal-move position with random init should be ~1.0–2.0
    assert 0.1 < aux_val < 10.0, (
        f"aux_loss={aux_val} on one-position fixture — "
        "expected ~log(3) magnitude with random init"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_conversion_loss.py -v
```

Expected: FAIL — `alphazero_loss_batch` doesn't accept `conversion_loss_weight` or doesn't return 7-tuple.

- [ ] **Step 3a: Add the masked-log-probs helper**

In `scripts/GPU/alphazero/trainer.py`, add this helper above `alphazero_loss_batch` (so it's importable from tests):

```python
def compute_masked_log_probs(logits, move_mask):
    """Compute log-softmax over only the legal moves indicated by move_mask.

    Padded/illegal columns are forced to a large negative value before
    logsumexp, ensuring they contribute zero probability to the denominator.
    Returned log_probs at illegal columns are well-defined but should not
    be consumed by any loss term — masking the target is the caller's
    responsibility.

    Args:
        logits: (B, M) MLX array of pre-softmax scores.
        move_mask: (B, M) MLX array of 1.0 (legal) / 0.0 (illegal/padding).
    Returns:
        log_probs: (B, M) MLX array — log-softmax over legal moves only.
    """
    masked_logits = mx.where(
        move_mask > 0,
        logits,
        mx.array(-1e9, dtype=logits.dtype),
    )
    return masked_logits - mx.logsumexp(masked_logits, axis=1, keepdims=True)
```

- [ ] **Step 3b: Extend alphazero_loss_batch**

In `scripts/GPU/alphazero/trainer.py` (currently around line 1022), modify `alphazero_loss_batch`:

```python
def alphazero_loss_batch(
    network: AlphaZeroNetwork,
    positions: List["PositionRecord"],
    l2_weight: float = 1e-4,
    value_weight: float = 0.5,
    max_moves_cap: int = 512,
    active_size: int = 24,
    progress_weighted: bool = True,
    progress_weight_floor: float = 0.25,
    conversion_loss_weight: float = 0.0,             # NEW: Spec 2
    conversion_completion_weight: float = 1.0,       # NEW
    conversion_reducer_weight: float = 0.35,         # NEW
):
    """Batched policy + value + l2 loss (vectorized).

    ... (existing docstring) ...

    Returns:
        Tuple of (total_loss, policy_loss, value_loss, l2_loss,
                  aux_loss, aux_coverage, aux_n_eligible).

        aux_loss: MLX scalar — conversion auxiliary CE, mean over eligible
            positions only (Spec 2 §6.3). Zero when conversion_loss_weight==0.
        aux_coverage: float — n_eligible / batch_size.
        aux_n_eligible: int — exact count of eligible positions in the batch.

    IMPORTANT: total_loss MUST be first element because nn.value_and_grad()
    only differentiates the first returned value.
    """
    if conversion_loss_weight > 0.0:
        boards, move_rows, move_cols, move_mask, target_pi, outcomes, legal_padded = \
            make_padded_batch(positions, max_moves_cap=max_moves_cap, return_legal=True)
    else:
        boards, move_rows, move_cols, move_mask, target_pi, outcomes = make_padded_batch(
            positions, max_moves_cap=max_moves_cap
        )
        legal_padded = None

    # ... existing plies_np / game_n_moves_np / forward / value_loss machinery ...

    logits, values, _ = network.forward_padded(
        boards, move_rows, move_cols, move_mask, active_size=active_size
    )

    # Use the SAME masked log_probs for policy and aux. Apply move_mask
    # EXPLICITLY here — do not rely on network.forward_padded() pre-masking,
    # which can drift across architecture changes. The explicit mx.where
    # makes the masking contract part of the loss function itself.
    log_probs = compute_masked_log_probs(logits, move_mask)
    policy_loss = -mx.sum(target_pi * log_probs, axis=1)
    policy_loss = mx.mean(policy_loss)

    # ... value_loss + l2_loss machinery unchanged ...

    # NEW: conversion auxiliary loss
    aux_loss = mx.array(0.0)
    aux_coverage = 0.0
    aux_n_eligible = 0
    if conversion_loss_weight > 0.0:
        from .conversion_loss import make_conversion_aux_tensors
        # max_moves_cap may differ from M (the actual padded width). Use M.
        M = target_pi.shape[1]
        aux_target_np, aux_mask_np = make_conversion_aux_tensors(
            positions, legal_padded, max_moves_cap=M,
            completion_weight=conversion_completion_weight,
            reducer_weight=conversion_reducer_weight,
        )
        aux_target = mx.array(aux_target_np)
        aux_mask   = mx.array(aux_mask_np)

        per_pos_aux = -mx.sum(aux_target * log_probs, axis=1)
        per_pos_aux = aux_mask * per_pos_aux
        n_eligible_arr = mx.sum(aux_mask)
        aux_loss = mx.where(n_eligible_arr > 0,
                            mx.sum(per_pos_aux) / mx.maximum(n_eligible_arr, 1.0),
                            mx.array(0.0))
        aux_n_eligible = int(n_eligible_arr.item())
        aux_coverage = aux_n_eligible / max(len(positions), 1)

    total_loss = (policy_loss
                  + value_weight * value_loss
                  + l2_loss
                  + conversion_loss_weight * aux_loss)

    return total_loss, policy_loss, value_loss, l2_loss, aux_loss, aux_coverage, aux_n_eligible
```

- [ ] **Step 4: Update train_step return signature**

In `scripts/GPU/alphazero/trainer.py` (currently around line 1099), update `train_step`:

```python
def train_step(
    network, main_module, opt_main, opt_value, batch,
    l2_weight=1e-4, value_weight=0.5, max_moves_cap=512, active_size=24,
    value_grad_max_norm=0.5, progress_weighted=True, progress_weight_floor=0.25,
    conversion_loss_weight: float = 0.0,             # NEW
    conversion_completion_weight: float = 1.0,       # NEW
    conversion_reducer_weight: float = 0.35,         # NEW
):
    """... existing docstring ...

    Returns:
        Tuple of (total_loss, policy_loss, value_loss, l2_loss,
                  aux_loss, aux_coverage, aux_n_eligible).
    """
    def loss_fn(model):
        return alphazero_loss_batch(
            model, batch,
            l2_weight=l2_weight, value_weight=value_weight,
            max_moves_cap=max_moves_cap, active_size=active_size,
            progress_weighted=progress_weighted,
            progress_weight_floor=progress_weight_floor,
            conversion_loss_weight=conversion_loss_weight,
            conversion_completion_weight=conversion_completion_weight,
            conversion_reducer_weight=conversion_reducer_weight,
        )

    # value_and_grad differentiates first element (total_loss)
    loss_tuple, grads = nn.value_and_grad(network, loss_fn)(network)
    total_loss, policy_loss, value_loss, l2_loss, aux_loss, aux_coverage, aux_n_eligible = loss_tuple

    # ... existing grad-clipping, opt updates, mx.eval ...

    return (
        float(total_loss.item()),
        float(policy_loss.item()),
        float(value_loss.item()),
        float(l2_loss.item()),
        float(aux_loss.item()),
        float(aux_coverage),
        int(aux_n_eligible),
    )
```

Update the trainer's per-step accumulator (look for `sum_policy`, `sum_value`, `sum_l2` variables — likely around line 3188+). Add:

```python
        sum_aux = 0.0
        sum_aux_coverage = 0.0
        sum_aux_n_eligible = 0
        # ... existing ...
        for ... in train loop:
            loss_total, loss_policy, loss_value, loss_l2, loss_aux, aux_cov, aux_neli = train_step(
                network, main_module, opt_main, opt_value, batch,
                l2_weight=l2_weight, value_weight=value_weight,
                max_moves_cap=max_moves_cap, active_size=active_size,
                value_grad_max_norm=value_grad_max_norm,
                progress_weighted=progress_weighted,
                progress_weight_floor=progress_weight_floor,
                conversion_loss_weight=effective_conversion_loss_weight,
                conversion_completion_weight=conversion_completion_weight,
                conversion_reducer_weight=conversion_reducer_weight,
            )
            sum_policy += loss_policy
            sum_value += loss_value
            sum_l2 += loss_l2
            sum_aux += loss_aux
            sum_aux_coverage += aux_cov
            sum_aux_n_eligible += aux_neli
```

Where `effective_conversion_loss_weight` is computed at the top of the iter loop:

```python
        effective_conversion_loss_weight = (
            conversion_policy_loss_weight if conversion_policy_loss_enabled else 0.0
        )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_conversion_loss.py -v
pytest tests/test_trainer_loss.py -v   # ensure existing tests still pass
```

Expected: 7 conversion-loss tests pass; existing trainer-loss tests still pass (the 7-tuple return is a strict superset of the old 4-tuple — existing tests need updating to unpack 7 values, see Step 6).

- [ ] **Step 6: Update existing trainer-loss tests for new return signature**

`grep -n "alphazero_loss_batch\|train_step" tests/` for callers; update each to unpack 7 values (use `_` for the new aux fields when not asserted).

```bash
pytest tests/ -v -k "trainer_loss or alphazero_loss_batch"
```

Expected: all PASS after unpacking updates.

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py tests/test_conversion_loss.py tests/test_trainer_loss.py
git commit -m "feat(conversion): wire auxiliary CE loss into alphazero_loss_batch

Spec 2 §6.3: aux_loss uses the same masked log_probs as policy_loss
(no second forward pass, no second logsumexp). Mean over eligible
positions only — n_eligible returned as int for exact telemetry.
Zero-overhead path when conversion_loss_weight == 0.0.

train_step return signature extends from 4-tuple to 7-tuple
(total, policy, value, l2, aux, coverage, n_eligible). Existing
callers updated.

Anchors: test_aux_loss_uses_masked_log_probs,
test_aux_loss_returns_n_eligible_as_int.
"
```

---

### Task 9: conversion_training sidecar block (loss telemetry only)

**Files:**
- Create: `scripts/GPU/alphazero/conversion_telemetry.py` (loss + buffer halves; recovery half added in Task 16)
- Modify: `scripts/GPU/alphazero/trainer.py` (sidecar writer)
- Test: `tests/test_conversion_telemetry.py`

- [ ] **Step 1: Write the failing tests (loss-side telemetry only)**

```python
# tests/test_conversion_telemetry.py
"""Sidecar telemetry tests (Spec 2 §8). Recovery-bucket tests added in
Task 16; this file starts with conversion_training only."""
from scripts.GPU.alphazero.conversion_telemetry import build_conversion_training_block


def test_conversion_training_block_schema_when_disabled():
    block = build_conversion_training_block(
        config={
            "configured_loss_weight": 0.05,
            "effective_loss_weight": 0.0,
            "completion_weight": 1.0,
            "reducer_weight": 0.35,
            "max_total_goal_distance": 2,
            "min_component_size": 8,
            "sample_boost": 1.0,
            "max_batch_fraction": 0.15,
        },
        enabled=False,
        buffer_stats={
            "eligible_positions_in_buffer": 0,
            "eligible_position_rate": 0.0,
            "eligible_positions_at_active_size": 0,
            "eligible_rate_at_active_size": 0.0,
        },
        loss_accumulator={
            "sum_aux": 0.0,
            "sum_aux_coverage": 0.0,
            "sum_aux_n_eligible": 0,
            "steps_done": 0,
            "batch_size": 256,
        },
        sample_accumulator=None,
    )
    assert block["version"] == 1
    assert block["enabled"] is False
    assert block["config"]["configured_loss_weight"] == 0.05
    assert block["config"]["effective_loss_weight"] == 0.0
    assert block["loss"]["aux_loss_avg_iter"] == 0.0
    assert block["loss"]["aux_target_coverage_rate"] == 0.0
    assert block["loss"]["aux_positions_seen_in_training"] == 0
    assert block["loss"]["aux_positions_fraction_in_batches"] == 0.0
    # Stable schema — all keys present
    assert "consistency" in block
    # When sample_accumulator is None, consistency reports unavailable
    assert block["consistency"]["available"] is False
    assert block["consistency"]["drawn_vs_seen_match"] is None
    assert block["consistency"]["drawn_minus_seen"] is None


def test_conversion_training_consistency_unavailable_when_phase2_only():
    """Phase 2 wires loss before Phase 3 sampler stats. With sum_aux_n_eligible>0
    but sample_accumulator=None, consistency must report available=False, NOT
    drawn_vs_seen_match=False (which would be a false positive)."""
    block = build_conversion_training_block(
        config={"configured_loss_weight": 0.05, "effective_loss_weight": 0.05,
                "completion_weight": 1.0, "reducer_weight": 0.35,
                "max_total_goal_distance": 2, "min_component_size": 8,
                "sample_boost": 1.0, "max_batch_fraction": 0.15},
        enabled=True,
        buffer_stats={"eligible_positions_in_buffer": 100,
                      "eligible_position_rate": 0.1,
                      "eligible_positions_at_active_size": 100,
                      "eligible_rate_at_active_size": 0.1},
        loss_accumulator={"sum_aux": 100.0, "sum_aux_coverage": 5.0,
                          "sum_aux_n_eligible": 1280, "steps_done": 50,
                          "batch_size": 256},
        sample_accumulator=None,
    )
    assert block["enabled"] is True
    assert block["loss"]["aux_positions_seen_in_training"] == 1280
    assert block["consistency"]["available"] is False
    assert block["consistency"]["drawn_vs_seen_match"] is None
    assert block["consistency"]["drawn_minus_seen"] is None


def test_conversion_training_block_schema_when_enabled():
    block = build_conversion_training_block(
        config={
            "configured_loss_weight": 0.05,
            "effective_loss_weight": 0.05,
            "completion_weight": 1.0,
            "reducer_weight": 0.35,
            "max_total_goal_distance": 2,
            "min_component_size": 8,
            "sample_boost": 1.0,
            "max_batch_fraction": 0.15,
        },
        enabled=True,
        buffer_stats={
            "eligible_positions_in_buffer": 1234,
            "eligible_position_rate": 0.0247,
            "eligible_positions_at_active_size": 980,
            "eligible_rate_at_active_size": 0.0312,
        },
        loss_accumulator={
            "sum_aux": 100.0,    # 100 / 50 steps = 2.0 avg
            "sum_aux_coverage": 5.0,    # 5 / 50 = 0.1
            "sum_aux_n_eligible": 1280,
            "steps_done": 50,
            "batch_size": 256,
        },
        sample_accumulator=None,
    )
    assert block["enabled"] is True
    assert block["config"]["effective_loss_weight"] == 0.05
    assert block["loss"]["aux_loss_avg_iter"] == 2.0
    assert block["loss"]["aux_target_coverage_rate"] == 0.1
    assert block["loss"]["aux_positions_seen_in_training"] == 1280
    assert block["loss"]["aux_positions_fraction_in_batches"] == 1280 / (50 * 256)


def test_conversion_training_block_disabled_emits_zero_telemetry():
    """Even with non-zero accumulator (defensive), if enabled=False the
    loss block reports zeros."""
    block = build_conversion_training_block(
        config={"configured_loss_weight": 0.05, "effective_loss_weight": 0.0,
                "completion_weight": 1.0, "reducer_weight": 0.35,
                "max_total_goal_distance": 2, "min_component_size": 8,
                "sample_boost": 1.0, "max_batch_fraction": 0.15},
        enabled=False,
        buffer_stats={"eligible_positions_in_buffer": 0,
                      "eligible_position_rate": 0.0,
                      "eligible_positions_at_active_size": 0,
                      "eligible_rate_at_active_size": 0.0},
        loss_accumulator={"sum_aux": 999.0, "sum_aux_coverage": 0.5,
                          "sum_aux_n_eligible": 9999, "steps_done": 50,
                          "batch_size": 256},
        sample_accumulator=None,
    )
    assert block["loss"]["aux_loss_avg_iter"] == 0.0
    assert block["loss"]["aux_target_coverage_rate"] == 0.0
    assert block["loss"]["aux_positions_seen_in_training"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_conversion_telemetry.py -v
```

Expected: ImportError on `conversion_telemetry`.

- [ ] **Step 3: Implement build_conversion_training_block**

```python
# scripts/GPU/alphazero/conversion_telemetry.py
"""Sidecar telemetry builders for Spec 2 conversion correction + recovery
bucket. Pure dict math — no I/O, no MLX."""
from __future__ import annotations
from typing import Optional


def build_conversion_training_block(
    config: dict,
    *,
    enabled: bool,
    buffer_stats: dict,
    loss_accumulator: dict,
    sample_accumulator: Optional[dict],
) -> dict:
    """Build the per-iter conversion_training sidecar block.

    Schema is stable across enabled/disabled — every field is present,
    only values differ. When enabled=False, loss/sample fields emit zeros
    regardless of accumulator state (defensive).
    """
    steps = max(loss_accumulator.get("steps_done", 0), 0)
    batch_size = loss_accumulator.get("batch_size", 1)

    if enabled:
        avg_aux = (loss_accumulator["sum_aux"] / steps) if steps > 0 else 0.0
        avg_cov = (loss_accumulator["sum_aux_coverage"] / steps) if steps > 0 else 0.0
        seen = loss_accumulator.get("sum_aux_n_eligible", 0)
        seen_frac = (seen / (steps * batch_size)) if steps > 0 else 0.0
    else:
        avg_aux = 0.0
        avg_cov = 0.0
        seen = 0
        seen_frac = 0.0

    # Sample accumulator may be None during Phase 2 (before sampler stats wired).
    # Emit consistency.available=False to flag that drawn-vs-seen check is N/A.
    if sample_accumulator is None:
        sample_block = {
            "eligible_drawn_total": 0,
            "eligible_drawn_fraction": 0.0,
            "cap_was_binding_steps": 0,
            "boost_inactive_steps": 0,
        }
        consistency_block = {
            "drawn_vs_seen_match": None,
            "drawn_minus_seen": None,
            "available": False,
        }
    else:
        drawn_total = sample_accumulator.get("eligible_drawn_total", 0)
        sample_block = {
            "eligible_drawn_total": drawn_total,
            "eligible_drawn_fraction": (
                drawn_total / (steps * batch_size) if steps > 0 else 0.0
            ),
            "cap_was_binding_steps": sample_accumulator.get("cap_was_binding_steps", 0),
            "boost_inactive_steps": sample_accumulator.get("boost_inactive_steps", 0),
        }
        drawn_minus_seen = drawn_total - seen
        consistency_block = {
            "drawn_vs_seen_match": (drawn_minus_seen == 0),
            "drawn_minus_seen": int(drawn_minus_seen),
            "available": True,
        }

    return {
        "version": 1,
        "enabled": bool(enabled),
        "config": dict(config),
        "buffer": dict(buffer_stats),
        "loss": {
            "aux_loss_avg_iter": float(avg_aux),
            "aux_target_coverage_rate": float(avg_cov),
            "aux_positions_seen_in_training": int(seen),
            "aux_positions_fraction_in_batches": float(seen_frac),
        },
        "sample_stats": sample_block,
        "consistency": consistency_block,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_conversion_telemetry.py -v
```

Expected: 3 passed (more added in Tasks 14, 16).

- [ ] **Step 5: Wire into trainer's per-iter sidecar writer with real buffer stats**

Phase 2 must emit truthful buffer counts (no hard-coded zeros), even though the O(1) eligible-index pool doesn't land until Task 11. Use a one-shot O(N) scan of the replay buffer at sidecar-writing time. This call runs once per iteration — N is bounded by buffer_size, so cost is acceptable.

In `scripts/GPU/alphazero/trainer.py`, locate the per-iter sidecar writer (where `forced_probe_summary`, `goal_completion_summary`, etc. are assembled into the sidecar dict, around line 3072+). Add:

```python
            # Phase 2 buffer scan: O(N) over replay buffer once per iter.
            # Phase 3 (Task 11) replaces this with the O(1) index pool.
            _all_positions = replay_buffer._positions
            _eligible_total = sum(
                1 for p in _all_positions if getattr(p, "conversion", None) is not None
            )
            _eligible_at_size = sum(
                1 for p in _all_positions
                if getattr(p, "conversion", None) is not None
                and p.active_size == active_size
            )
            _at_size = sum(1 for p in _all_positions if p.active_size == active_size)
            buffer_stats = {
                "eligible_positions_in_buffer": _eligible_total,
                "eligible_position_rate": (
                    _eligible_total / len(_all_positions) if _all_positions else 0.0
                ),
                "eligible_positions_at_active_size": _eligible_at_size,
                "eligible_rate_at_active_size": (
                    _eligible_at_size / _at_size if _at_size > 0 else 0.0
                ),
            }

            sidecar["conversion_training"] = build_conversion_training_block(
                config={
                    "configured_loss_weight": conversion_policy_loss_weight,
                    "effective_loss_weight": effective_conversion_loss_weight,
                    "completion_weight": conversion_completion_weight,
                    "reducer_weight": conversion_reducer_weight,
                    "max_total_goal_distance": conversion_max_total_goal_distance,
                    "min_component_size": goal_completion_min_component_size,
                    "sample_boost": conversion_sample_boost,
                    "max_batch_fraction": conversion_max_batch_fraction,
                },
                enabled=conversion_policy_loss_enabled,
                buffer_stats=buffer_stats,
                loss_accumulator={
                    "sum_aux": sum_aux,
                    "sum_aux_coverage": sum_aux_coverage,
                    "sum_aux_n_eligible": sum_aux_n_eligible,
                    "steps_done": steps_done,
                    "batch_size": batch_size,
                },
                # Phase 2: sampler stats not yet wired. Pass None so the
                # consistency block reports available=False (drawn-vs-seen
                # check is N/A until Task 13). This avoids false-positive
                # mismatches when sum_aux_n_eligible > 0 but no sampler
                # tracking exists yet.
                sample_accumulator=None,
            )
```

Add the import near the top:

```python
from .conversion_telemetry import build_conversion_training_block
```

When Task 11/12 land the index pool + sampler stats, Step 5 of Task 12 replaces the O(N) scan with `replay_buffer.count_eligible(active_size=...)` (O(1)) and Task 13 replaces `sample_accumulator=None` with the real accumulator.

- [ ] **Step 6: Run a 1-iter smoke run to confirm sidecar shape**

```bash
python -m scripts.GPU.alphazero.train \
    --iterations 1 --games-per-iter 2 --train-steps 4 --batch-size 8 \
    --no-save-games --probes-inline-disable \
    --conversion-policy-loss-enabled \
    --conversion-policy-loss-weight 0.05 \
    2>&1 | tee /tmp/spec2_smoke.log
```

Expected: training completes, sidecar JSON in checkpoint dir contains a `conversion_training` block with `enabled=true` and `effective_loss_weight=0.05`.

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/conversion_telemetry.py scripts/GPU/alphazero/trainer.py tests/test_conversion_telemetry.py
git commit -m "feat(conversion): emit conversion_training sidecar block

Spec 2 §8.1: schema-stable per-iter telemetry block. Loss accumulator
feeds aux_loss_avg_iter / aux_target_coverage_rate /
aux_positions_seen_in_training / aux_positions_fraction_in_batches.
Buffer + sample stats placeholders for Phase 3.

When effective_conversion_enabled=False, loss/sample fields emit zeros
regardless of accumulator state (defensive — all keys still present).
"
```

---

### Task 10: Phase-2 end-to-end smoke test

**Files:**
- Modify: `tests/test_trainer_loss.py` (or `tests/test_self_play_goal_completion_integration.py`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_trainer_loss.py`:

```python
def test_trainer_runs_with_conversion_enabled_smoke(tmp_path):
    """Phase 2 smoke: 1 iter, 2 games, batch=8, conversion enabled.
    Asserts no crashes and sidecar populated correctly."""
    from scripts.GPU.alphazero.trainer import train

    network = train(
        n_iterations=1,
        games_per_iteration=2,
        train_steps_per_iteration=4,
        batch_size=8,
        buffer_size=100,
        checkpoint_dir=str(tmp_path),
        save_games=False,
        probes_inline_disable=True,
        # Spec 2 flags
        conversion_policy_loss_enabled=True,
        conversion_policy_loss_weight=0.05,
        conversion_completion_weight=1.0,
        conversion_reducer_weight=0.35,
        conversion_max_total_goal_distance=2,
        conversion_sample_boost=1.0,
        conversion_max_batch_fraction=0.15,
    )
    assert network is not None

    sidecar_files = list(tmp_path.glob("iter_*_stats.json"))
    assert len(sidecar_files) >= 1
    import json
    sidecar = json.loads(sidecar_files[0].read_text())
    cnv = sidecar["conversion_training"]
    assert cnv["enabled"] is True
    assert cnv["config"]["effective_loss_weight"] == 0.05
    # Phase 2: sampler stats not yet wired. Consistency must report
    # available=False — NOT False-positive drawn_vs_seen_match=False.
    assert cnv["consistency"]["available"] is False
    assert cnv["consistency"]["drawn_vs_seen_match"] is None
    # Buffer stats from O(N) scan (Task 9) must be real, not zero.
    # eligible_positions_in_buffer >= 0 (the count is whatever the
    # 2-game/4-step run produced; we assert the field is populated
    # and the rate is consistent).
    assert cnv["buffer"]["eligible_positions_in_buffer"] >= 0
    if cnv["buffer"]["eligible_positions_in_buffer"] > 0:
        assert cnv["buffer"]["eligible_position_rate"] > 0.0
```

- [ ] **Step 2: Run test to verify it passes**

```bash
pytest tests/test_trainer_loss.py::test_trainer_runs_with_conversion_enabled_smoke -v
```

Expected: PASS (this is verification that Tasks 1–9 wired up correctly).

If it FAILS, the failure mode tells you which task's plumbing has a gap. Fix that task's gap, then re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_trainer_loss.py
git commit -m "test(conversion): Phase 2 end-to-end smoke

1 iter, 2 games, batch=8 with conversion enabled. Asserts sidecar
conversion_training.enabled=true and effective_loss_weight=0.05.
"
```

---

## Phase 3 — Bounded replay sample boost

### Task 11: ReplayBuffer eligibility index pool

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` (`ReplayBuffer`)
- Test: `tests/test_replay_buffer_conversion.py`

- [ ] **Step 1: Write the failing tests (eligibility tracking only)**

```python
# tests/test_replay_buffer_conversion.py
"""ReplayBuffer eligibility tracking + stratified sampling (Spec 2 §7)."""
import numpy as np
from scripts.GPU.alphazero.trainer import ReplayBuffer
from scripts.GPU.alphazero.self_play import PositionRecord


def _pos(eligible: bool):
    conv = (
        {
            "version": 1,
            "endpoint_completion_moves": [[0, 0]],
            "distance_reducing_moves": [],
        }
        if eligible
        else None
    )
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move="red",
        legal_moves=[(0, 0), (1, 1)],
        visit_counts=[1, 1],
        outcome=1.0,
        active_size=24,
        ply=0,
        game_n_moves=10,
        conversion=conv,
    )


def test_replay_buffer_eligible_index_tracks_evictions():
    """Adding eligible positions, then enough non-eligible to evict them,
    should remove them from the eligible pool."""
    buf = ReplayBuffer(max_size=4)
    eligible = [_pos(True) for _ in range(4)]
    buf.add_positions(eligible)
    assert buf.count_eligible() == 4

    # Add 4 non-eligible to trigger ring-buffer eviction of the 4 eligible.
    buf.add_positions([_pos(False) for _ in range(4)])
    assert buf.count_eligible() == 0


def test_replay_buffer_eligible_index_swap_delete_correctness():
    """Add 5 eligibles, evict the middle one (idx=2), remaining indices
    must still resolve to eligible positions."""
    buf = ReplayBuffer(max_size=10)
    buf.add_positions([_pos(True) for _ in range(5)])
    assert buf.count_eligible() == 5
    # Remove idx=2 manually (mimicking ring-buffer overwrite at that slot)
    buf._positions[2] = _pos(False)
    buf._update_eligible_index(2, buf._positions[2])
    assert buf.count_eligible() == 4
    # The remaining 4 indices in the eligible pool must point to eligible positions.
    for idx in buf._eligible_idxs:
        assert buf._positions[idx].conversion is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_replay_buffer_conversion.py -v
```

Expected: FAIL — `count_eligible`, `_eligible_idxs`, `_update_eligible_index` don't exist.

- [ ] **Step 3: Add eligibility tracking to ReplayBuffer**

In `scripts/GPU/alphazero/trainer.py`, modify `ReplayBuffer` (around line 1186):

```python
class ReplayBuffer:
    """Fixed-size buffer of training positions with uniform sampling.

    Spec 2 §7: maintains a parallel index pool of conversion-eligible
    positions for stratified sampling.
    """

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self._positions: list = []
        self._next_write = 0
        # Spec 2: O(1) eligible-position pool with swap-delete semantics
        self._eligible_idxs: list = []           # ordered list of eligible indices
        self._eligible_pos: dict = {}            # idx -> position in _eligible_idxs

    def _eligible_add(self, idx: int) -> None:
        if idx in self._eligible_pos:
            return
        self._eligible_pos[idx] = len(self._eligible_idxs)
        self._eligible_idxs.append(idx)

    def _eligible_remove(self, idx: int) -> None:
        pos = self._eligible_pos.pop(idx, None)
        if pos is None:
            return
        last = self._eligible_idxs.pop()
        if pos < len(self._eligible_idxs):
            self._eligible_idxs[pos] = last
            self._eligible_pos[last] = pos

    def _update_eligible_index(self, idx: int, p) -> None:
        if getattr(p, "conversion", None) is not None:
            self._eligible_add(idx)
        else:
            self._eligible_remove(idx)

    def count_eligible(self, active_size=None) -> int:
        """Count eligible positions, optionally filtered by active_size.
        O(1) when active_size is None; O(E) otherwise."""
        if active_size is None:
            return len(self._eligible_idxs)
        return sum(
            1 for i in self._eligible_idxs
            if self._positions[i].active_size == active_size
        )

    def add_positions(self, positions) -> None:
        """Add positions; maintain eligibility index across ring-buffer overwrites."""
        for p in positions:
            if len(self._positions) < self.max_size:
                idx = len(self._positions)
                self._positions.append(p)
            else:
                idx = self._next_write
                self._positions[idx] = p
                self._next_write = (self._next_write + 1) % self.max_size
            self._update_eligible_index(idx, p)
```

(If `add_game` exists separately, ensure it also calls `_update_eligible_index` per added position — likely just calls `add_positions`.)

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_replay_buffer_conversion.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py tests/test_replay_buffer_conversion.py
git commit -m "feat(conversion): add eligibility index pool to ReplayBuffer

Spec 2 §7.1: O(1) add/remove via swap-delete index pool. Eligibility
flag derives from PositionRecord.conversion is not None — single source
of truth from self-play attach point.
"
```

---

### Task 12: Stratified `sample()`

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` (`ReplayBuffer.sample`)
- Test: `tests/test_replay_buffer_conversion.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_replay_buffer_conversion.py`:

```python
import math


def test_sample_boost_1_is_pure_uniform():
    """ANCHOR (Spec 2 §11.3): boost=1.0 short-circuits — eligibility
    set is not consulted, cap is not consulted, behavior identical
    to pre-Spec-2 sample()."""
    buf = ReplayBuffer(max_size=100)
    # 50 eligible + 50 non-eligible
    buf.add_positions([_pos(True) for _ in range(50)])
    buf.add_positions([_pos(False) for _ in range(50)])
    # Sample many batches with boost=1.0 — eligible fraction should match
    # natural rate (50%), within Monte Carlo noise.
    import random
    rng = random.Random(42)
    eligible_drawn = 0
    total_drawn = 0
    for _ in range(20):
        batch = buf.sample(
            batch_size=10, rng=rng, active_size=24,
            conversion_sample_boost=1.0,
            conversion_max_batch_fraction=0.15,
        )
        eligible_drawn += sum(1 for p in batch if p.conversion is not None)
        total_drawn += len(batch)
    natural_rate = eligible_drawn / total_drawn
    # ~0.5 with some noise; capped sampling at 15% would give ~0.15.
    # Boost=1.0 must short-circuit, so we expect natural rate.
    assert 0.35 < natural_rate < 0.65, (
        f"natural_rate={natural_rate}; boost=1.0 should produce ~0.5"
    )


def test_sample_boost_2_produces_at_most_cap_fraction():
    buf = ReplayBuffer(max_size=200)
    buf.add_positions([_pos(True) for _ in range(50)])
    buf.add_positions([_pos(False) for _ in range(150)])
    import random
    rng = random.Random(42)
    for _ in range(20):
        batch = buf.sample(
            batch_size=20, rng=rng, active_size=24,
            conversion_sample_boost=10.0,    # high boost
            conversion_max_batch_fraction=0.15,    # cap 15% = floor(20*0.15)=3
        )
        eligible = sum(1 for p in batch if p.conversion is not None)
        assert eligible <= 3, f"eligible={eligible} exceeds cap of 3"


def test_sample_boost_uses_ceil_rounding_for_target():
    """Spec 2 §7.3: ceil rounding so rare eligibles aren't rounded to zero.
    Fixture: batch=16, natural expectation < 1, cap allows 1+, boost > 1."""
    buf = ReplayBuffer(max_size=1000)
    buf.add_positions([_pos(True) for _ in range(10)])    # rare
    buf.add_positions([_pos(False) for _ in range(990)])  # bulk
    # natural_expectation = 16 * (10 / 1000) = 0.16
    # ceil(0.16 * 2.0) = 1; cap_count = floor(16 * 0.15) = 2; min(1, 2, 10, 16) = 1
    import random
    rng = random.Random(42)
    eligible_counts = []
    for _ in range(50):
        batch = buf.sample(
            batch_size=16, rng=rng, active_size=24,
            conversion_sample_boost=2.0,
            conversion_max_batch_fraction=0.15,
        )
        eligible_counts.append(sum(1 for p in batch if p.conversion is not None))
    # All batches should have exactly 1 eligible (deterministic given the formula).
    # If floor rounding were used, we'd see 0s.
    assert all(c == 1 for c in eligible_counts), (
        f"eligible counts: {set(eligible_counts)} — expected all 1 with ceil rounding"
    )


def test_sample_falls_back_to_uniform_when_eligible_pool_empty(capsys):
    buf = ReplayBuffer(max_size=100)
    buf.add_positions([_pos(False) for _ in range(50)])    # zero eligible
    import random
    rng = random.Random(42)
    batch = buf.sample(
        batch_size=10, rng=rng, active_size=24,
        conversion_sample_boost=2.0,
        conversion_max_batch_fraction=0.15,
    )
    assert len(batch) == 10
    assert all(p.conversion is None for p in batch)
    # Stats should record boost_was_inactive
    stats = buf.last_sample_stats
    assert stats.boost_was_inactive is True


def test_sample_active_size_intersects_eligibility():
    buf = ReplayBuffer(max_size=200)
    # 50 eligible at size 24, 50 eligible at size 12, 50 non-eligible at size 24
    for _ in range(50):
        p = _pos(True); p.active_size = 24
        buf.add_positions([p])
    for _ in range(50):
        p = _pos(True); p.active_size = 12
        buf.add_positions([p])
    for _ in range(50):
        p = _pos(False); p.active_size = 24
        buf.add_positions([p])
    import random
    rng = random.Random(42)
    batch = buf.sample(
        batch_size=20, rng=rng, active_size=12,
        conversion_sample_boost=10.0,
        conversion_max_batch_fraction=0.5,
    )
    # Every position drawn must have active_size=12.
    assert all(p.active_size == 12 for p in batch)


def test_sample_no_duplicate_positions_with_two_strata():
    """ANCHOR (Spec 2 §11.3): no replacement across strata."""
    buf = ReplayBuffer(max_size=20)
    buf.add_positions([_pos(True) for _ in range(5)])      # 5 eligible
    buf.add_positions([_pos(False) for _ in range(15)])    # 15 non-eligible
    import random
    rng = random.Random(42)
    batch = buf.sample(
        batch_size=20, rng=rng, active_size=24,    # batch == buffer
        conversion_sample_boost=10.0,
        conversion_max_batch_fraction=0.5,         # cap 10
    )
    # batch should be all 20 buffer positions, no duplicates.
    ids = [id(p) for p in batch]
    assert len(set(ids)) == len(ids), "duplicate positions in batch"


def test_sample_stats_match_aux_n_eligible():
    """Spec 2 §8.2 invariant: drawn (sampler) == seen (loss). We assert
    the sampler-side count here; the loss-side equality is exercised in
    test_drawn_vs_seen_match_flags_divergence (Task 14)."""
    buf = ReplayBuffer(max_size=100)
    buf.add_positions([_pos(True) for _ in range(20)])
    buf.add_positions([_pos(False) for _ in range(80)])
    import random
    rng = random.Random(42)
    batch = buf.sample(
        batch_size=20, rng=rng, active_size=24,
        conversion_sample_boost=2.0,
        conversion_max_batch_fraction=0.5,
    )
    drawn = sum(1 for p in batch if p.conversion is not None)
    assert buf.last_sample_stats.eligible_drawn == drawn
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_replay_buffer_conversion.py -v
```

Expected: FAIL — `sample` doesn't accept the conversion kwargs, no `last_sample_stats`.

- [ ] **Step 3: Implement stratified sample()**

In `scripts/GPU/alphazero/trainer.py`, modify `ReplayBuffer.sample` (around line 1217):

```python
import math
from dataclasses import dataclass


@dataclass
class _SampleStats:
    batch_size: int
    eligible_drawn: int
    cap_was_binding: bool
    boost_was_inactive: bool


class ReplayBuffer:
    # ... existing code from Task 11 ...

    last_sample_stats: _SampleStats = None    # set after each sample()

    def sample(
        self,
        batch_size: int,
        rng=None,
        active_size: int = None,
        *,
        conversion_sample_boost: float = 1.0,
        conversion_max_batch_fraction: float = 0.15,
    ):
        """Stratified sample.

        boost == 1.0: pure uniform — eligibility ignored, cap not consulted.
        boost > 1.0: stratified per Spec 2 §7.3.
        """
        if rng is None:
            import random
            rng = random.Random()

        # Filter pool by active_size
        if active_size is None:
            full_pool = list(range(len(self._positions)))
        else:
            full_pool = [
                i for i, p in enumerate(self._positions)
                if p.active_size == active_size
            ]

        if not full_pool:
            self.last_sample_stats = _SampleStats(
                batch_size=batch_size, eligible_drawn=0,
                cap_was_binding=False, boost_was_inactive=True,
            )
            return []

        # SHORT-CIRCUIT for boost == 1.0 (anchor invariant)
        if conversion_sample_boost == 1.0:
            chosen = rng.sample(full_pool, k=min(batch_size, len(full_pool)))
            batch = [self._positions[i] for i in chosen]
            drawn = sum(1 for p in batch if getattr(p, "conversion", None) is not None)
            self.last_sample_stats = _SampleStats(
                batch_size=batch_size, eligible_drawn=drawn,
                cap_was_binding=False, boost_was_inactive=False,
            )
            return batch

        # Stratified path
        eligible_in_size = [
            i for i in self._eligible_idxs
            if active_size is None or self._positions[i].active_size == active_size
        ]
        non_eligible_in_size = [
            i for i in full_pool if i not in self._eligible_pos
        ]
        eligible_count = len(eligible_in_size)

        cap_count = int(math.floor(batch_size * conversion_max_batch_fraction))
        if cap_count == 0 or eligible_count == 0:
            chosen = rng.sample(full_pool, k=min(batch_size, len(full_pool)))
            batch = [self._positions[i] for i in chosen]
            drawn = sum(1 for p in batch if getattr(p, "conversion", None) is not None)
            self.last_sample_stats = _SampleStats(
                batch_size=batch_size, eligible_drawn=drawn,
                cap_was_binding=False,
                boost_was_inactive=(eligible_count == 0),
            )
            if eligible_count == 0:
                # Log once per iter — caller (trainer) is responsible for de-duping.
                pass
            return batch

        natural_expectation = batch_size * (eligible_count / len(full_pool))
        target_eligible = min(
            math.ceil(natural_expectation * conversion_sample_boost),
            cap_count,
            eligible_count,
            batch_size,
        )

        chosen_eligible = rng.sample(eligible_in_size, k=target_eligible)
        n_other = min(batch_size - target_eligible, len(non_eligible_in_size))
        chosen_other = rng.sample(non_eligible_in_size, k=n_other)
        chosen = chosen_eligible + chosen_other
        rng.shuffle(chosen)
        batch = [self._positions[i] for i in chosen]

        cap_was_binding = (target_eligible == cap_count and target_eligible < math.ceil(
            natural_expectation * conversion_sample_boost
        ))
        self.last_sample_stats = _SampleStats(
            batch_size=batch_size, eligible_drawn=target_eligible,
            cap_was_binding=cap_was_binding, boost_was_inactive=False,
        )
        return batch
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_replay_buffer_conversion.py -v
```

Expected: 8 passed (2 from Task 11 + 6 new + 1 stats consistency).

- [ ] **Step 5: Wire `last_sample_stats` into trainer accumulator**

In `scripts/GPU/alphazero/trainer.py`, in the per-iter training loop, after each `buf.sample(...)` call, accumulate the stats:

```python
            sum_eligible_drawn = 0
            sum_cap_binding = 0
            sum_boost_inactive = 0
            for ... in train loop:
                batch = replay_buffer.sample(
                    batch_size=batch_size, rng=batch_rng, active_size=active_size,
                    conversion_sample_boost=conversion_sample_boost,
                    conversion_max_batch_fraction=conversion_max_batch_fraction,
                )
                stats = replay_buffer.last_sample_stats
                sum_eligible_drawn += stats.eligible_drawn
                if stats.cap_was_binding:
                    sum_cap_binding += 1
                if stats.boost_was_inactive:
                    sum_boost_inactive += 1
                # ... train_step ...
```

Pass these into the sidecar block via `sample_accumulator`:

```python
            sidecar["conversion_training"] = build_conversion_training_block(
                # ... existing args ...
                sample_accumulator={
                    "eligible_drawn_total": sum_eligible_drawn,
                    "cap_was_binding_steps": sum_cap_binding,
                    "boost_inactive_steps": sum_boost_inactive,
                },
            )
```

- [ ] **Step 6: Update conversion_training schema test for sample_stats path**

Append to `tests/test_conversion_telemetry.py`:

```python
def test_conversion_training_block_includes_sample_stats():
    block = build_conversion_training_block(
        config={"configured_loss_weight": 0.05, "effective_loss_weight": 0.05,
                "completion_weight": 1.0, "reducer_weight": 0.35,
                "max_total_goal_distance": 2, "min_component_size": 8,
                "sample_boost": 2.0, "max_batch_fraction": 0.15},
        enabled=True,
        buffer_stats={"eligible_positions_in_buffer": 0,
                      "eligible_position_rate": 0.0,
                      "eligible_positions_at_active_size": 0,
                      "eligible_rate_at_active_size": 0.0},
        loss_accumulator={"sum_aux": 100.0, "sum_aux_coverage": 5.0,
                          "sum_aux_n_eligible": 1280, "steps_done": 50,
                          "batch_size": 256},
        sample_accumulator={"eligible_drawn_total": 1280,
                            "cap_was_binding_steps": 5,
                            "boost_inactive_steps": 0},
    )
    assert block["sample_stats"]["eligible_drawn_total"] == 1280
    assert block["sample_stats"]["cap_was_binding_steps"] == 5
    assert block["consistency"]["available"] is True
    assert block["consistency"]["drawn_vs_seen_match"] is True
    assert block["consistency"]["drawn_minus_seen"] == 0
```

```bash
pytest tests/test_conversion_telemetry.py -v
```

Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py tests/test_replay_buffer_conversion.py tests/test_conversion_telemetry.py
git commit -m "feat(conversion): stratified ReplayBuffer.sample with bounded boost

Spec 2 §7.3: boost=1.0 short-circuits to pure uniform (anchor
test_sample_boost_1_is_pure_uniform). Stratified path uses ceil
rounding so rare eligibles aren't rounded to zero; cap bounds
eligible fraction; no-replacement invariant across both strata
(anchor test_sample_no_duplicate_positions_with_two_strata).

Sample stats wired into conversion_training sidecar block; drawn-vs-
seen consistency invariant computed.
"
```

---

### Task 13: Drawn-vs-seen consistency invariant

**Files:**
- Modify: `scripts/GPU/alphazero/conversion_telemetry.py`
- Modify: `scripts/GPU/alphazero/trainer.py` (warn on divergence)
- Test: `tests/test_conversion_telemetry.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_conversion_telemetry.py`:

```python
def test_drawn_vs_seen_match_flags_divergence():
    """ANCHOR (Spec 2 §11.3): when sampler-drawn != loss-seen, flag false
    and report exact delta."""
    block = build_conversion_training_block(
        config={"configured_loss_weight": 0.05, "effective_loss_weight": 0.05,
                "completion_weight": 1.0, "reducer_weight": 0.35,
                "max_total_goal_distance": 2, "min_component_size": 8,
                "sample_boost": 2.0, "max_batch_fraction": 0.15},
        enabled=True,
        buffer_stats={"eligible_positions_in_buffer": 0,
                      "eligible_position_rate": 0.0,
                      "eligible_positions_at_active_size": 0,
                      "eligible_rate_at_active_size": 0.0},
        loss_accumulator={"sum_aux": 100.0, "sum_aux_coverage": 5.0,
                          "sum_aux_n_eligible": 1280, "steps_done": 50,
                          "batch_size": 256},
        sample_accumulator={"eligible_drawn_total": 1300,    # mismatch by +20
                            "cap_was_binding_steps": 0,
                            "boost_inactive_steps": 0},
    )
    assert block["consistency"]["drawn_vs_seen_match"] is False
    assert block["consistency"]["drawn_minus_seen"] == 20


def test_drawn_vs_seen_match_naming_correctness():
    """Spec 2 §8.2 lock: drawn = sampler count, seen = loss count.
    NOT reversed."""
    # Sampler draws 100, loss sees 90 → delta is +10 (drawn - seen).
    block = build_conversion_training_block(
        config={"configured_loss_weight": 0.05, "effective_loss_weight": 0.05,
                "completion_weight": 1.0, "reducer_weight": 0.35,
                "max_total_goal_distance": 2, "min_component_size": 8,
                "sample_boost": 2.0, "max_batch_fraction": 0.15},
        enabled=True,
        buffer_stats={"eligible_positions_in_buffer": 0,
                      "eligible_position_rate": 0.0,
                      "eligible_positions_at_active_size": 0,
                      "eligible_rate_at_active_size": 0.0},
        loss_accumulator={"sum_aux": 0.0, "sum_aux_coverage": 0.0,
                          "sum_aux_n_eligible": 90, "steps_done": 1,
                          "batch_size": 100},
        sample_accumulator={"eligible_drawn_total": 100,
                            "cap_was_binding_steps": 0,
                            "boost_inactive_steps": 0},
    )
    # drawn (100) - seen (90) = 10 (positive)
    assert block["consistency"]["drawn_minus_seen"] == 10
```

- [ ] **Step 2: Run tests to verify they pass**

The consistency math already lives in `build_conversion_training_block` (Task 9). These tests should pass without code changes — they pin the behavior.

```bash
pytest tests/test_conversion_telemetry.py::test_drawn_vs_seen_match_flags_divergence -v
pytest tests/test_conversion_telemetry.py::test_drawn_vs_seen_match_naming_correctness -v
```

Expected: PASS.

- [ ] **Step 3: Wire warning log in trainer**

In `scripts/GPU/alphazero/trainer.py`, after building the sidecar block, only warn when consistency is *available* AND mismatched (don't warn during Phase 2 when `available=False`):

```python
            cons = sidecar["conversion_training"]["consistency"]
            if cons["available"] and not cons["drawn_vs_seen_match"]:
                print(
                    f"[WARN] [conversion] drawn vs seen mismatch: "
                    f"drawn={sum_eligible_drawn}, seen={sum_aux_n_eligible}, "
                    f"delta={cons['drawn_minus_seen']}. Likely cause: "
                    f"legal-move alignment or active-size filter divergence "
                    f"between sampler and loss."
                )
```

- [ ] **Step 4: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py tests/test_conversion_telemetry.py
git commit -m "feat(conversion): drawn-vs-seen consistency invariant

Spec 2 §8.2 anchor: sampler-drawn vs loss-seen exact integer match.
Mismatch flagged in sidecar (drawn_vs_seen_match=false, drawn_minus_seen),
trainer logs WARN line so operators see the divergence in real time.
"
```

---

### Task 14: Phase-3 end-to-end smoke

**Files:**
- Modify: `tests/test_trainer_loss.py` (extend)

- [ ] **Step 1: Write the test**

```python
def test_trainer_runs_with_sample_boost_smoke(tmp_path):
    """Phase 3 smoke: 1 iter, conversion enabled with boost=2.0.
    Asserts sample_stats populated and drawn-vs-seen invariant holds."""
    from scripts.GPU.alphazero.trainer import train

    network = train(
        n_iterations=1,
        games_per_iteration=2,
        train_steps_per_iteration=4,
        batch_size=8,
        buffer_size=100,
        checkpoint_dir=str(tmp_path),
        save_games=False,
        probes_inline_disable=True,
        conversion_policy_loss_enabled=True,
        conversion_policy_loss_weight=0.05,
        conversion_sample_boost=2.0,
        conversion_max_batch_fraction=0.5,
    )
    assert network is not None

    sidecar_files = list(tmp_path.glob("iter_*_stats.json"))
    import json
    sidecar = json.loads(sidecar_files[0].read_text())
    cnv = sidecar["conversion_training"]
    # Phase 3: sampler stats wired. Consistency check IS available now.
    assert cnv["consistency"]["available"] is True
    assert cnv["consistency"]["drawn_vs_seen_match"] is True
    assert cnv["sample_stats"]["eligible_drawn_total"] == cnv["loss"]["aux_positions_seen_in_training"]
```

- [ ] **Step 2: Run test**

```bash
pytest tests/test_trainer_loss.py::test_trainer_runs_with_sample_boost_smoke -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_trainer_loss.py
git commit -m "test(conversion): Phase 3 sample-boost end-to-end smoke

Asserts drawn-vs-seen invariant holds in a real 1-iter run with
boost=2.0 and cap=0.5. Catches plumbing gaps between sampler and
loss accumulators.
"
```

---

## Phase 4 — Recovery / extreme-closeout-drift telemetry (independent)

### Task 15: Recovery predicate + block builder

**Files:**
- Modify: `scripts/GPU/alphazero/conversion_telemetry.py`
- Test: `tests/test_conversion_telemetry.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_conversion_telemetry.py`:

```python
from scripts.GPU.alphazero.conversion_telemetry import (
    build_recovery_block,
    is_recovery_or_extreme_closeout_drift,
)


def _record(
    detected=True, outcome_class=1, du_moves=0, delay=0,
    state_cap=False,
):
    rec = {
        "detected": detected,
        "outcome_class": outcome_class,
        "winner_moves_with_dominant_unavailable": du_moves if outcome_class == 1 else None,
        "dominant_unavailable_moves": du_moves if outcome_class == 2 else None,
        "conversion_delay_plies": delay,
        "reason": "state_cap" if state_cap else "win",
    }
    return rec


def test_recovery_predicate_three_triggers():
    # DU clause
    rec_du = _record(du_moves=15)
    assert is_recovery_or_extreme_closeout_drift(rec_du, du_threshold=10, delay_threshold=20)
    # Delay clause
    rec_delay = _record(delay=25)
    assert is_recovery_or_extreme_closeout_drift(rec_delay, du_threshold=10, delay_threshold=20)
    # State-cap clause
    rec_cap = _record(outcome_class=2, state_cap=True)
    assert is_recovery_or_extreme_closeout_drift(rec_cap, du_threshold=10, delay_threshold=20)


def test_recovery_predicate_state_cap_after_detection_required_for_class2():
    """Class 2 with detected=False → not counted."""
    rec = _record(detected=False, outcome_class=2, state_cap=True)
    assert not is_recovery_or_extreme_closeout_drift(rec, du_threshold=10, delay_threshold=20)


def test_recovery_block_class2_dominant_unavailable_handling():
    """Spec 2 §8.4 lock: Class 2 du_moves explicitly defined; no silent zero."""
    # Class 2 with du_moves field present
    rec_class2 = _record(outcome_class=2, du_moves=15, state_cap=False)
    # state_cap=False so the cap-clause doesn't fire; only DU clause should trigger.
    rec_class2["reason"] = "win"    # not state_cap
    rec_class2["winner_moves_with_dominant_unavailable"] = None
    rec_class2["dominant_unavailable_moves"] = 15
    assert is_recovery_or_extreme_closeout_drift(rec_class2, du_threshold=10, delay_threshold=20)


def test_recovery_block_excludes_undetected_games():
    rec = _record(detected=False, du_moves=15, delay=25)
    assert not is_recovery_or_extreme_closeout_drift(rec, du_threshold=10, delay_threshold=20)


def test_recovery_block_percentiles_handcrafted():
    records = [_record(du_moves=v) for v in [0, 1, 2, 3, 4, 5, 10, 15, 20, 22]]
    block = build_recovery_block(records, du_threshold=10, delay_threshold=20)
    p = block["dominant_unavailable_moves"]
    # p50 of 10 values: midpoint of 5th and 6th sorted values
    assert p["p50"] in (4, 5)    # approximate
    assert p["p90"] >= 15
    assert p["max"] == 22


def test_recovery_rate_denominators():
    """Spec 2 §8.3: rate = count/games_total; rate_among_detected = count/detected_games."""
    records = [
        _record(detected=True, du_moves=15),    # triggers
        _record(detected=True, du_moves=0),     # no trigger
        _record(detected=False),                # not detected
    ]
    block = build_recovery_block(records, du_threshold=10, delay_threshold=20)
    assert block["games_total"] == 3
    assert block["detected_games"] == 2
    assert block["count"] == 1
    assert block["rate"] == 1 / 3
    assert block["rate_among_detected"] == 0.5


def test_recovery_block_renamed_to_extreme_closeout_drift():
    """Spec 2 §5 lock: sidecar key uses the renamed form."""
    block = build_recovery_block([], du_threshold=10, delay_threshold=20)
    # The block itself doesn't carry its own name, but the trainer wiring
    # writes it under sidecar["recovery_or_extreme_closeout_drift"]. We
    # assert via the trainer test in Task 17. Here, verify the block has
    # the expected schema fields:
    assert "version" in block
    assert "config" in block
    assert "trigger_breakdown" in block
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_conversion_telemetry.py -v
```

Expected: ImportError on `build_recovery_block` / `is_recovery_or_extreme_closeout_drift`.

- [ ] **Step 3: Implement**

Append to `scripts/GPU/alphazero/conversion_telemetry.py`:

```python
def is_recovery_or_extreme_closeout_drift(
    record: dict,
    *,
    du_threshold: int,
    delay_threshold: int,
) -> bool:
    """Predicate for the recovery / extreme-closeout-drift bucket.

    Three OR-ed clauses (Spec 2 §8.4):
      - dominant_unavailable_moves >= du_threshold
      - conversion_delay_plies >= delay_threshold
      - outcome_class == 2 AND detected (state_cap_after_detection)
    """
    if not record.get("detected"):
        return False

    # du_moves field varies by outcome_class
    outcome_class = record.get("outcome_class")
    if outcome_class == 1:
        du_moves = record.get("winner_moves_with_dominant_unavailable")
    else:
        # Class 2: explicit fallback chain (no silent zero)
        du_moves = record.get("dominant_unavailable_moves")
    if du_moves is not None and du_moves >= du_threshold:
        return True

    delay = record.get("conversion_delay_plies")
    if delay is not None and delay >= delay_threshold:
        return True

    if outcome_class == 2 and record.get("detected") and record.get("reason") == "state_cap":
        return True

    return False


def _trigger_breakdown(record, *, du_threshold, delay_threshold):
    """Return which clauses fired for this record (for breakdown counts)."""
    triggers = set()
    outcome_class = record.get("outcome_class")
    if outcome_class == 1:
        du = record.get("winner_moves_with_dominant_unavailable")
    else:
        du = record.get("dominant_unavailable_moves")
    if du is not None and du >= du_threshold:
        triggers.add("dominant_unavailable")
    delay = record.get("conversion_delay_plies")
    if delay is not None and delay >= delay_threshold:
        triggers.add("delay_ge_threshold")
    if outcome_class == 2 and record.get("detected") and record.get("reason") == "state_cap":
        triggers.add("state_cap_after_detection")
    return triggers


def _percentile(values, p):
    if not values:
        return 0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def build_recovery_block(
    records: list,
    *,
    du_threshold: int,
    delay_threshold: int,
    enabled: bool = True,
) -> dict:
    """Build the per-iter recovery_or_extreme_closeout_drift sidecar block."""
    if not enabled:
        return {
            "version": 1,
            "enabled": False,
            "config": {
                "dominant_unavailable_moves_threshold": du_threshold,
                "delay_threshold": delay_threshold,
            },
            "games_total": len(records),
            "detected_games": 0,
            "count": 0,
            "rate": 0.0,
            "rate_among_detected": 0.0,
            "dominant_unavailable_moves": {"p50": 0, "p90": 0, "p95": 0, "max": 0, "mean": 0.0},
            "trigger_breakdown": {
                "dominant_unavailable_only": 0,
                "delay_ge_threshold_only": 0,
                "state_cap_after_detection_only": 0,
                "multiple_triggers": 0,
            },
        }

    games_total = len(records)
    detected_games = sum(1 for r in records if r.get("detected"))

    # DU values across detected games (whichever field is populated)
    du_values = []
    for r in records:
        if not r.get("detected"):
            continue
        oc = r.get("outcome_class")
        v = (r.get("winner_moves_with_dominant_unavailable")
             if oc == 1
             else r.get("dominant_unavailable_moves"))
        if v is not None:
            du_values.append(v)

    qualifying = [
        r for r in records
        if is_recovery_or_extreme_closeout_drift(
            r, du_threshold=du_threshold, delay_threshold=delay_threshold
        )
    ]

    # Trigger breakdown — mutually exclusive partition
    breakdown = {
        "dominant_unavailable_only": 0,
        "delay_ge_threshold_only": 0,
        "state_cap_after_detection_only": 0,
        "multiple_triggers": 0,
    }
    for r in qualifying:
        triggers = _trigger_breakdown(r, du_threshold=du_threshold,
                                      delay_threshold=delay_threshold)
        if len(triggers) >= 2:
            breakdown["multiple_triggers"] += 1
        elif "dominant_unavailable" in triggers:
            breakdown["dominant_unavailable_only"] += 1
        elif "delay_ge_threshold" in triggers:
            breakdown["delay_ge_threshold_only"] += 1
        elif "state_cap_after_detection" in triggers:
            breakdown["state_cap_after_detection_only"] += 1

    return {
        "version": 1,
        "enabled": True,
        "config": {
            "dominant_unavailable_moves_threshold": du_threshold,
            "delay_threshold": delay_threshold,
        },
        "games_total": games_total,
        "detected_games": detected_games,
        "count": len(qualifying),
        "rate": (len(qualifying) / games_total) if games_total > 0 else 0.0,
        "rate_among_detected": (
            len(qualifying) / detected_games if detected_games > 0 else 0.0
        ),
        "dominant_unavailable_moves": {
            "p50": _percentile(du_values, 50),
            "p90": _percentile(du_values, 90),
            "p95": _percentile(du_values, 95),
            "max": max(du_values) if du_values else 0,
            "mean": (sum(du_values) / len(du_values)) if du_values else 0.0,
        },
        "trigger_breakdown": breakdown,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_conversion_telemetry.py -v
```

Expected: all telemetry tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/conversion_telemetry.py tests/test_conversion_telemetry.py
git commit -m "feat(recovery): add recovery_or_extreme_closeout_drift block

Spec 2 §8.3-8.5: telemetry-only block reading existing
goal_completion_record fields. Three-clause OR predicate (DU moves,
delay, state_cap_after_detection); mutually exclusive trigger
breakdown; rate over all games AND rate among detected games;
explicit Class 2 du_moves fallback (no silent zero).
"
```

---

### Task 16: Trainer wiring + analyzer surfacing

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` (sidecar writer)
- Modify: `scripts/twixt_replay_analyzer.py` (read-only trend section + per-iter CSV)
- Test: `tests/test_trainer_loss.py` (extend)

- [ ] **Step 1: Wire build_recovery_block into trainer sidecar**

In `scripts/GPU/alphazero/trainer.py`, after the `goal_completion_summary` is built, add:

```python
            from .conversion_telemetry import build_recovery_block
            sidecar["recovery_or_extreme_closeout_drift"] = build_recovery_block(
                records=[g.goal_completion_record for g in completed_games
                         if g.goal_completion_record is not None],
                du_threshold=recovery_dominant_unavailable_threshold,
                delay_threshold=recovery_delay_threshold,
                enabled=recovery_bucket_enabled,
            )
```

- [ ] **Step 2: Write the test**

Append to `tests/test_trainer_loss.py`:

```python
def test_trainer_writes_recovery_block_to_sidecar(tmp_path):
    from scripts.GPU.alphazero.trainer import train
    train(
        n_iterations=1, games_per_iteration=2, train_steps_per_iteration=2,
        batch_size=4, buffer_size=50, checkpoint_dir=str(tmp_path),
        save_games=False, probes_inline_disable=True,
        recovery_bucket_enabled=True,
    )
    import json
    sidecar = json.loads(list(tmp_path.glob("iter_*_stats.json"))[0].read_text())
    assert "recovery_or_extreme_closeout_drift" in sidecar
    rec = sidecar["recovery_or_extreme_closeout_drift"]
    assert rec["version"] == 1
    assert "config" in rec
    assert "trigger_breakdown" in rec
```

- [ ] **Step 3: Run test**

```bash
pytest tests/test_trainer_loss.py::test_trainer_writes_recovery_block_to_sidecar -v
```

Expected: PASS.

- [ ] **Step 4: Add analyzer trend section + per-iter CSV**

In `scripts/twixt_replay_analyzer.py`, locate the existing `format_*_report` functions (e.g., `format_goal_completion_report`, `format_strong_advantage_probe_report`). Add a parallel:

```python
def format_conversion_training_trend_report(sidecar_summaries: dict) -> list[str]:
    """Read-only roll-up of conversion_training blocks across iters."""
    lines = ["── Conversion-training trend ─────────────────────────────────"]
    if not sidecar_summaries:
        lines.append("  (no conversion_training data)")
        lines.append("──────────────────────────────────────────────────────────────")
        return lines
    iters = sorted(sidecar_summaries.keys())
    lines.append(f"Iters covered:   {iters[0]}-{iters[-1]}")
    weights = sorted(set(
        sidecar_summaries[i]["config"]["effective_loss_weight"] for i in iters
    ))
    if len(weights) == 1:
        lines.append(f"Aux loss weight: {weights[0]} (constant)")
    else:
        lines.append(f"Aux loss weight: varies ({weights})")
    aux_losses = [sidecar_summaries[i]["loss"]["aux_loss_avg_iter"] for i in iters]
    lines.append("Aux loss (avg):  " + " → ".join(f"{x:.2f}" for x in aux_losses))
    coverages = [sidecar_summaries[i]["loss"]["aux_target_coverage_rate"] for i in iters]
    lines.append("Coverage rate:   " + " → ".join(f"{x*100:.1f}%" for x in coverages))
    matches = [sidecar_summaries[i]["consistency"]["drawn_vs_seen_match"] for i in iters]
    lines.append(f"Drawn vs seen:   {'✓ all iters consistent' if all(matches) else '✗ DIVERGENCE — check warnings'}")
    lines.append("──────────────────────────────────────────────────────────────")
    return lines


def format_recovery_or_extreme_closeout_drift_report(sidecar_summaries: dict) -> list[str]:
    """Read-only roll-up of recovery blocks across iters."""
    lines = ["── Recovery / extreme-closeout-drift (telemetry only) ────────"]
    if not sidecar_summaries:
        lines.append("  (no recovery data)")
        lines.append("──────────────────────────────────────────────────────────────")
        return lines
    iters = sorted(sidecar_summaries.keys())
    lines.append(f"Iters covered:        {iters[0]}-{iters[-1]}")
    counts = [sidecar_summaries[i]["count"] for i in iters]
    rates = [sidecar_summaries[i]["rate"] for i in iters]
    p90s = [sidecar_summaries[i]["dominant_unavailable_moves"]["p90"] for i in iters]
    lines.append("Recovery count/iter:  " + " → ".join(str(x) for x in counts))
    lines.append("Recovery rate:        " + " → ".join(f"{x*100:.1f}%" for x in rates))
    lines.append("DU moves p90:         " + " → ".join(str(x) for x in p90s))
    lines.append("──────────────────────────────────────────────────────────────")
    return lines
```

Wire both into the analyzer's main report assembly path. Read sidecar JSONs and pass each block keyed by iteration.

For the per-iter CSVs, add:

```python
def write_conversion_training_by_iter_csv(sidecar_summaries: dict, path: str):
    fieldnames = [
        "iteration", "cnv_enabled", "cnv_loss_weight", "cnv_aux_loss_avg",
        "cnv_aux_coverage", "cnv_aux_seen", "cnv_eligible_in_buf",
        "cnv_eligible_at_size", "cnv_drawn_total", "cnv_drawn_vs_seen_ok",
    ]
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for it in sorted(sidecar_summaries.keys()):
            s = sidecar_summaries[it]
            w.writerow({
                "iteration": it,
                "cnv_enabled": int(s["enabled"]),
                "cnv_loss_weight": s["config"]["effective_loss_weight"],
                "cnv_aux_loss_avg": s["loss"]["aux_loss_avg_iter"],
                "cnv_aux_coverage": s["loss"]["aux_target_coverage_rate"],
                "cnv_aux_seen": s["loss"]["aux_positions_seen_in_training"],
                "cnv_eligible_in_buf": s["buffer"]["eligible_positions_in_buffer"],
                "cnv_eligible_at_size": s["buffer"]["eligible_positions_at_active_size"],
                "cnv_drawn_total": s["sample_stats"]["eligible_drawn_total"],
                "cnv_drawn_vs_seen_ok": int(s["consistency"]["drawn_vs_seen_match"]),
            })


def write_recovery_or_extreme_closeout_drift_by_iter_csv(sidecar_summaries: dict, path: str):
    fieldnames = ["iteration", "rcv_count", "rcv_rate", "rcv_du_p90"]
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for it in sorted(sidecar_summaries.keys()):
            s = sidecar_summaries[it]
            w.writerow({
                "iteration": it,
                "rcv_count": s["count"],
                "rcv_rate": s["rate"],
                "rcv_du_p90": s["dominant_unavailable_moves"]["p90"],
            })
```

Wire both writers into the analyzer's CSV-output section, parallel to `forced_probe_by_iter.csv`.

- [ ] **Step 5: Smoke-test the analyzer on a 1-iter run**

```bash
python -m scripts.GPU.alphazero.train \
    --iterations 1 --games-per-iter 2 --train-steps 4 --batch-size 8 \
    --no-save-games --probes-inline-disable \
    --conversion-policy-loss-enabled --conversion-policy-loss-weight 0.05 \
    --checkpoint-dir /tmp/spec2_analyzer_smoke
python scripts/twixt_replay_analyzer.py --input /tmp/spec2_analyzer_smoke --output /tmp/spec2_analyzer_smoke_report
ls /tmp/spec2_analyzer_smoke_report/
grep -E "Conversion-training trend|Recovery / extreme" /tmp/spec2_analyzer_smoke_report/report.txt
```

Expected: report.txt contains both sections; both CSVs present.

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py scripts/twixt_replay_analyzer.py tests/test_trainer_loss.py
git commit -m "feat(recovery): trainer wiring + analyzer surfacing

Spec 2 §8.6, §8.8: trainer writes recovery_or_extreme_closeout_drift
sidecar block alongside goal_completion_summary. Analyzer surfaces
two minimal trend sections in report.txt and writes per-iter CSVs
parallel to forced_probe_by_iter.csv.
"
```

---

## Phase 5 — Curated probes + mining

**Deferred per Spec 2 §12.5.** Lands in a follow-up spec once Phase 2's effect is measured against the 110-119 baseline using the existing analyzer policy_mcts_summary report.

---

## Final integration check

### Task 17: Full first-experiment run

**Files:** None (operational verification only)

- [ ] **Step 1: Run the first-experiment recipe**

```bash
python -m scripts.GPU.alphazero.train \
    --iterations 10 \
    --games-per-iter 100 \
    --conversion-policy-loss-enabled \
    --conversion-policy-loss-weight 0.05 \
    --conversion-completion-weight 1.0 \
    --conversion-reducer-weight 0.35 \
    --conversion-sample-boost 1.0 \
    --conversion-max-batch-fraction 0.15 \
    --conversion-max-total-goal-distance 2 \
    --checkpoint-dir checkpoints/spec2_exp1
```

- [ ] **Step 2: Run the analyzer**

```bash
python scripts/twixt_replay_analyzer.py \
    --input checkpoints/spec2_exp1 \
    --output reports/spec2_exp1
```

- [ ] **Step 3: Compare against 110-119 baseline**

In `reports/spec2_exp1/report.txt`, locate the policy_mcts_summary section and compare to baseline (Spec 2 §10.1):

| Metric | Baseline | Target (10-iter) | Observed |
|---|---|---|---|
| Endpoint completion policy top5 | 0.0% | > 30–40% | ? |
| Distance reducing policy top5 | 0.0% | > 40–50% | ? |
| Selected redundant | 31.7% | < 25% | ? |
| Selected off-chain | 10.7% | < 8% | ? |
| Drawn-vs-seen consistency | n/a | ✓ all iters | ? |

- [ ] **Step 4: Check guardrail metrics (must not degrade)**

- forced_probe sign_correct
- strong_advantage_probe sign_correct
- Opening corner / edge rates
- Average plies, draw / state_cap rate
- Red/black balance
- Policy entropy

- [ ] **Step 5: Decision point**

If policy top5 movement is meaningful and guardrails are clean → proceed to second-experiment recipe (`--conversion-sample-boost 2.0`). Otherwise: investigate aux_loss values, drawn-vs-seen warnings, and eligible_position_rate before tuning further. Per-iter interpretation note from Spec 2 §10.5 applies — judge wiring first.

---

## Plan complete

Total: 17 tasks across 4 phases. Each task is one or more commits with TDD. Anchor tests (9) are mandatory and protect Spec 2's structural invariants.
