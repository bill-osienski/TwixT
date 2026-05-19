# Marathon Termination Tuning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship four marathon-termination diagnostics (no-progress windows, adjudication coverage, resign-gate breakdown, value-uncertain guard predicate) as a pure-analyzer feature so the bucket-count-driven decision rule from spec §5 can be applied to existing training data.

**Architecture:** All four diagnostics are pure functions over per-game `goal_completion_record` + `goal_completion_diagnostics` + `meta` data already on disk. A new module `scripts/GPU/alphazero/marathon_termination_diagnostics.py` holds them; the analyzer aggregates per-iter + range-level and emits one new CSV + one new report section. The opponent-block helper in §3.1 must share an implementation with the existing recovery-retargeting defense classifier so the two diagnostics cannot diverge on what counts as a block.

**Tech Stack:** Python 3.14, pytest, pure-stdlib (csv module). No self-play change required (Task 0 pre-check resolved as Outcome A: `meta.adjudication_block_reason` already persisted with values matching the spec taxonomy).

**Spec:** [`docs/superpowers/specs/2026-05-19-marathon-termination-tuning-design.md`](../specs/2026-05-19-marathon-termination-tuning-design.md)

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `scripts/GPU/alphazero/marathon_termination_diagnostics.py` | Create | Five pure-function diagnostics: `detect_no_progress_windows`, `classify_adjudication_coverage`, `compute_resign_gate_breakdown`, `value_uncertain_guard`, `aggregate_marathon_termination`. Plus stable taxonomy/constant exports. |
| `scripts/twixt_replay_analyzer.py` | Modify | Add `write_marathon_termination_csv` + `format_marathon_termination_report` + main-path wiring (collect per-game pairs, call aggregator, write CSV, append report section). |
| `tests/test_marathon_termination_diagnostics.py` | Create | All 22 unit tests from spec §7. |
| `tests/test_analyzer_marathon_termination.py` | Create | 2 analyzer-integration tests (CSV emission + report section presence). |

The opponent-block helper in `marathon_termination_diagnostics.py` MUST import / share a helper with `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py` so the same defensive-move rule is applied in both diagnostics.

---

## Conventions

- Run tests via project venv: `.venv/bin/python -m pytest <path> -v`.
- All public functions use keyword-only arguments after the first positional.
- Adjudication taxonomy stable values: `not_attempted`, `value_below_threshold`, `min_top1_share`, `min_visits`, `missing_signal`, `would_have_passed`.
- Game-length partition: `short` (n_moves ≤ 100), `mid` (100 < n_moves ≤ 200), `long` (n_moves > 200).
- Window size for no-progress detector: N=15 own-moves; the detector counts **non-overlapping** 15-window chunks inside any stagnant run (a 30-move run → 2 windows; a 45-move run → 3). This matches the spec's K=2-consecutive-windows termination logic; an "episode count" (1 per stale run) would not.

### Range / iteration naming convention

Four overlapping range concepts exist; pin them up-front so CSV suffixes and report content don't drift:

| Concept | Where it lives | Example |
|---|---|---|
| **User-facing block name** | filename suffix, Replay staging dir, memory notes, this plan's prose | `220-229` |
| **Analyzer `iteration_min..iteration_max`** | first/last `meta.iteration` value seen in the staged JSONs; appears in the report's "Iteration range:" line | `219..228` (one-off because staging includes `iter_0219_*` from the previous block's tail) |
| **Checkpoint resume label** | `--resume model_iter_0XXX.safetensors` | `model_iter_0219` |
| **Per-game JSON `meta.iteration`** | actual training iteration produced by self-play | `220, 221, ..., 228` (iter 229 is training-only tail; no game files) |

**Rule:** range labels in filenames + memory notes follow the **user-facing block name** (e.g., `220-229`) even though the analyzer's iteration_min/iteration_max may read `219..228`. The aggregator MUST use the actual `meta.iteration` values for per-iter rows; the file suffix may use the requested block label. Tests that fixture per-iter rows should match the iteration values that real meta would produce (e.g., 220-228 for the 220-229 block, not 219-228).

- Conventional commit prefixes: `feat(diagnostics):` / `feat(analyzer):` / `docs(memory):` / `chore(memory):`.

---

## Task 0: Pre-check — adjudication telemetry already present (RESOLVED)

**Files:** none (verification-only).

Task 0 from spec §6 has been resolved in advance via inspection:

- `scripts/GPU/alphazero/self_play.py:1213-1222` computes `blocked_by` per attempt with values `"ply"`, `"threshold"`, `"visits"`, `"top1"`, or `None`.
- `scripts/GPU/alphazero/self_play.py:1248-1249` stores it as `adj_blocked_by`.
- `scripts/GPU/alphazero/self_play_worker.py:262` forwards via IPC.
- `scripts/GPU/alphazero/game_saver.py:33,146` persists as `meta["adjudication_block_reason"]` on the per-game JSON.
- Verified empirically on `scripts/GPU/logs/games/iter_0220_game_097.json`: `meta.adjudication_block_reason == "top1"` (state_cap game blocked by top1 gate).

**Outcome: A.** No self-play hook needed. Spec §3.2 analyzer-side implementation can proceed on existing 190-219 / 220-229 data.

Taxonomy mapping (self-play field → spec bucket). **None handling matches spec §3.2 verbatim**: a state_cap game with no block reason is a bug indicator (`would_have_passed`), NOT silently collapsed to `not_attempted`. Do not weaken this — collapsing None → `not_attempted` would hide the bug class the taxonomy was designed to surface.

| `meta.adjudication_block_reason` | Spec bucket | Reasoning |
|---|---|---|
| `"threshold"` | `value_below_threshold` | direct mapping |
| `"top1"` | `min_top1_share` | direct mapping |
| `"visits"` | `min_visits` | direct mapping |
| `"ply"` | `not_attempted` | ply gate fired before `adjudicate_min_ply` was reached — adjudication couldn't run |
| `None` AND `meta.reason == "state_cap"` | **`would_have_passed`** | bug indicator: game state-capped but no blocking gate recorded. Adjudication-enabled was set in the 190-219 + 220-229 launches, so a None block_reason on a state_cap game means an attempt should have happened. If adjudication was actually disabled (a separate launch config), the report MUST note this caveat alongside the count. |
| `None` AND `meta.reason == "adjudicated"` | (not a state_cap game; excluded from §3.2 scope) | classifier returns None |
| `meta.adjudication_block_reason` key absent entirely from old-format `meta` | `missing_signal` | per-game JSON predates the field; observability gap rather than a bug |

- [ ] **Step 1: Confirm Outcome A via one verification command**

Run:
```bash
.venv/bin/python -c "
import json, glob
counts = {'threshold': 0, 'top1': 0, 'visits': 0, 'ply': 0, None: 0}
for p in sorted(glob.glob('scripts/GPU/logs/games/iter_022?_game_*.json')):
    g = json.load(open(p))
    meta = g.get('meta') or {}
    if meta.get('reason') == 'state_cap':
        counts[meta.get('adjudication_block_reason')] = counts.get(meta.get('adjudication_block_reason'), 0) + 1
print('state_cap games adjudication_block_reason distribution (iters 220-228):')
for k, v in sorted(counts.items(), key=lambda kv: (kv[0] is None, str(kv[0]))):
    print(f'  {str(k):>12s}: {v}')
"
```

Expected: at least `threshold` and/or `top1` populated with non-zero counts, confirming per-game persistence is live on production data.

- [ ] **Step 2: No commit (verification-only).**

---

## Task 1: Rollback record (two surfaces)

**Files:**
- Already-created (this revision): `docs/superpowers/decisions/2026-05-19-reverted-closeout-experiments.md` (git-tracked decision record).
- Update: `~/.claude/projects/-Users-bill-projects-TwixT-Game/memory/spec4_recovery_retargeting_diagnostic.md` (Claude-session memory).

Both surfaces are needed:
- The **git-tracked decision record** is the canonical record. It survives across sessions, is visible in code review / PR history, and cannot be "forgotten" by a memory reset or context truncation.
- The **Claude-session memory** is for fast retrieval in subsequent sessions so a future suggestion can be vetoed without rereading the decision record from scratch.

- [ ] **Step 1: Verify the git-tracked decision record is committed**

Run:
```bash
git log --oneline --diff-filter=A -- docs/superpowers/decisions/2026-05-19-reverted-closeout-experiments.md
```

Expected: a commit listing the file's creation. If absent, the file is in the working tree but not yet committed — commit it before Task 2 starts.

- [ ] **Step 2: Append a short pointer to Claude-session memory**

Append the following to `~/.claude/projects/-Users-bill-projects-TwixT-Game/memory/spec4_recovery_retargeting_diagnostic.md` (or create a new closeout-experiments memory file if preferred). The pointer keeps the canonical detail in the decision record; the memory entry only needs to surface the takeaway and the reference.

```markdown
**Reverted closeout-side experiments (canonical record: `docs/superpowers/decisions/2026-05-19-reverted-closeout-experiments.md`):**

- `--closeout-selection-tiebreak-min-value 0.90` was tried and reverted (worsened tail). Stable value: 0.95.
- `--conversion-policy-loss-weight 0.075` was tried and reverted (worsened state-cap pressure). Stable value: 0.05.

Do NOT re-suggest either knob change without new evidence per the canonical record's "When to revisit" section.
```

- [ ] **Step 3: No additional git commit needed**

The decision record is the git-tracked surface; auto-memory is not committed. The pointer in memory is one-line and additive.

---

## Task 2: No-progress window detector (§3.1)

**Files:**
- Create: `scripts/GPU/alphazero/marathon_termination_diagnostics.py`
- Test: `tests/test_marathon_termination_diagnostics.py`

- [ ] **Step 1: Create the empty test file with imports**

Write `tests/test_marathon_termination_diagnostics.py`:

```python
"""Tests for the marathon-termination diagnostics module.

Spec: docs/superpowers/specs/2026-05-19-marathon-termination-tuning-design.md
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.marathon_termination_diagnostics import (
    detect_no_progress_windows,
    NO_PROGRESS_WINDOW_SIZE,
)


def _ply_entry(*, primary_class: str,
               own_td_before: int = 5, own_td_after: int = 5,
               opp_td_before: int = 5, opp_td_after: int = 5):
    """A goal_completion_diagnostics entry, minimally shaped for the detector."""
    return {
        "ply": 50, "side_to_move": "red",
        "goal_completion": {
            "total_goal_distance_before": own_td_before,
            "category": "one_endpoint_distance_2",
            "_own_td_after": own_td_after,
            "_opp_td_before": opp_td_before,
            "_opp_td_after": opp_td_after,
        },
        "selected_move": [10, 10],
        "selected_move_classification": {
            "primary_class": primary_class,
            "total_goal_distance_before": own_td_before,
            "total_goal_distance_after": own_td_after,
        },
    }
```

- [ ] **Step 2: Write the five window-detector tests**

Append to `tests/test_marathon_termination_diagnostics.py`:

```python
def test_no_progress_window_detects_pure_structural_run():
    """Spec §7 test 1. 15 consecutive redundant_reinforcement moves
    with no goal-distance progress → 1 window detected."""
    entries = [
        _ply_entry(primary_class="redundant_reinforcement",
                   own_td_before=5, own_td_after=5,
                   opp_td_before=5, opp_td_after=5)
        for _ in range(15)
    ]
    assert detect_no_progress_windows(entries, side="red") == 1


def test_no_progress_window_breaks_on_distance_reduction():
    """Spec §7 test 2. 14 redundant + 1 reduces_total_goal_distance → 0 windows."""
    entries = [
        _ply_entry(primary_class="redundant_reinforcement",
                   own_td_before=5, own_td_after=5)
        for _ in range(14)
    ]
    entries.append(_ply_entry(
        primary_class="reduces_total_goal_distance",
        own_td_before=5, own_td_after=4,
    ))
    assert detect_no_progress_windows(entries, side="red") == 0


def test_no_progress_window_breaks_on_endpoint_completion():
    """Spec §7 test 3. 14 redundant + 1 completes_endpoint → 0 windows."""
    entries = [
        _ply_entry(primary_class="redundant_reinforcement")
        for _ in range(14)
    ]
    entries.append(_ply_entry(primary_class="completes_endpoint"))
    assert detect_no_progress_windows(entries, side="red") == 0


def test_no_progress_window_breaks_on_opponent_block():
    """Spec §7 test 4. 14 redundant + 1 blocks_opponent_closeout → 0 windows."""
    entries = [
        _ply_entry(primary_class="redundant_reinforcement")
        for _ in range(14)
    ]
    entries.append(_ply_entry(primary_class="blocks_opponent_closeout"))
    assert detect_no_progress_windows(entries, side="red") == 0


def test_no_progress_window_window_size_15():
    """Spec §7 test 5. Exactly 14 redundant → 0 windows; 15 → 1."""
    e14 = [_ply_entry(primary_class="redundant_reinforcement") for _ in range(14)]
    e15 = [_ply_entry(primary_class="redundant_reinforcement") for _ in range(15)]
    assert detect_no_progress_windows(e14, side="red") == 0
    assert detect_no_progress_windows(e15, side="red") == 1
    # Sanity-check the exported constant.
    assert NO_PROGRESS_WINDOW_SIZE == 15
```

- [ ] **Step 3: Run tests to verify they fail with ImportError**

Run: `.venv/bin/python -m pytest tests/test_marathon_termination_diagnostics.py -v`

Expected: `ImportError: cannot import name 'detect_no_progress_windows'`.

- [ ] **Step 4: Create the module with the detector**

Write `scripts/GPU/alphazero/marathon_termination_diagnostics.py`:

```python
"""Marathon-termination diagnostics. Pure-function analyzer surface
that computes no-progress windows, adjudication-coverage gate-block
distributions, resign-gate breakdown by game-length partition, and the
value-uncertain guard predicate used by termination knobs.

Spec: docs/superpowers/specs/2026-05-19-marathon-termination-tuning-design.md

All inputs are per-game data already on disk:
  - per-game `goal_completion_record` (dict)
  - per-game `goal_completion_diagnostics` (list of per-ply entries)
  - per-game `meta` (dict; carries adjudication_block_reason from
    self_play.py:1248-1249 via game_saver.py:146)

No self-play change required (Task 0 outcome A).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Optional, Tuple


NO_PROGRESS_WINDOW_SIZE = 15

# §3.1 — local-structural primary_class vocabulary for the no-progress
# window detector. Matches Spec 4's primary_class names + the
# goal_completion classifier vocabulary (completes_endpoint /
# reduces_total_goal_distance / redundant_reinforcement / off_chain
# from self_play._classify_argmax_against_gc, and the Spec 4 broader
# vocabulary including blocks_opponent_closeout).
_LOCAL_STRUCTURAL_CLASSES = frozenset({
    "redundant_reinforcement",
    "off_chain",
    "connects_to_existing_component",
    "improves_own_largest_component",
})


def _is_local_structural(entry: dict) -> bool:
    cls = (entry.get("selected_move_classification") or {}).get("primary_class")
    return cls in _LOCAL_STRUCTURAL_CLASSES


def _own_distance_reduced(entry: dict) -> bool:
    cls_info = entry.get("selected_move_classification") or {}
    before = cls_info.get("total_goal_distance_before")
    after = cls_info.get("total_goal_distance_after")
    if before is None or after is None:
        return False
    return int(after) < int(before)


def _completed_endpoint(entry: dict) -> bool:
    cls = (entry.get("selected_move_classification") or {}).get("primary_class")
    return cls == "completes_endpoint"


def _blocked_opponent(entry: dict) -> bool:
    """Opponent block (Spec §3.1, definition pinned).

    Primary signal: `selected_move_classification.primary_class ==
    "blocks_opponent_closeout"` (Spec 4 defense-classifier output).

    This is the SHARED definition with the recovery-retargeting
    diagnostic — both consult the same classifier output via
    `primary_class`, so the two diagnostics cannot diverge on what
    counts as a block.
    """
    cls = (entry.get("selected_move_classification") or {}).get("primary_class")
    return cls == "blocks_opponent_closeout"


def detect_no_progress_windows(diagnostics: list, *, side: str) -> int:
    """Count distinct sliding-window no-progress runs for `side`.

    A no-progress window of length NO_PROGRESS_WINDOW_SIZE (=15) ends at
    own-move ply t iff the trailing 15 own-moves for this side all satisfy:
      - moves are local-structural (primary_class in the structural set)
      - no own-move reduced own total_goal_distance
      - no own-move completed an endpoint
      - no own-move was an opponent block

    Overlapping windows anchored at distinct end-plies count separately
    only if there's at least 1 own-move gap between their end-plies;
    otherwise they collapse into the longest contiguous run.

    Concretely: count maximal-length runs of consecutive own-moves
    satisfying the four conditions, then sum floor(run_length / 15) for
    each run.
    """
    own_entries = [e for e in (diagnostics or []) if e.get("side_to_move") == side]
    if len(own_entries) < NO_PROGRESS_WINDOW_SIZE:
        return 0

    def is_no_progress(e: dict) -> bool:
        return (
            _is_local_structural(e)
            and not _own_distance_reduced(e)
            and not _completed_endpoint(e)
            and not _blocked_opponent(e)
        )

    windows = 0
    run_len = 0
    for e in own_entries:
        if is_no_progress(e):
            run_len += 1
        else:
            windows += run_len // NO_PROGRESS_WINDOW_SIZE
            run_len = 0
    windows += run_len // NO_PROGRESS_WINDOW_SIZE
    return windows
```

- [ ] **Step 5: Run the five detector tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_marathon_termination_diagnostics.py -v`

Expected: 5 passed.

- [ ] **Step 6: Add the shared-helper invariant test**

Append to `tests/test_marathon_termination_diagnostics.py`:

```python
def test_no_progress_window_opponent_block_uses_primary_class_only():
    """Spec §7 test 6. The opponent-block test uses the
    primary_class == 'blocks_opponent_closeout' marker (Spec 4 vocabulary)
    — confirms we are NOT applying a stricter local recomputation. If
    Spec 4's defense classifier flagged the move, we trust it."""
    # A move classified as blocks_opponent_closeout (per Spec 4) — even if
    # the inline distance fields look ambiguous — must count as a block.
    entries = [
        _ply_entry(primary_class="redundant_reinforcement")
        for _ in range(14)
    ]
    entries.append({
        "ply": 50, "side_to_move": "red",
        "goal_completion": {"total_goal_distance_before": 5, "category": "x"},
        "selected_move": [10, 10],
        "selected_move_classification": {
            "primary_class": "blocks_opponent_closeout",
        },
    })
    # Run of 14 followed by a block → 0 no-progress windows.
    assert detect_no_progress_windows(entries, side="red") == 0
```

- [ ] **Step 7: Run all detector tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_marathon_termination_diagnostics.py -v`

Expected: 6 passed.

- [ ] **Step 8: Commit**

```bash
git add scripts/GPU/alphazero/marathon_termination_diagnostics.py tests/test_marathon_termination_diagnostics.py
git commit -m "$(cat <<'EOF'
feat(diagnostics): no-progress window detector (Spec marathon-termination §3.1)

Sliding-window detector: counts runs of >= 15 consecutive own-moves
that are all local-structural (primary_class in {redundant_reinforcement,
off_chain, connects_to_existing_component, improves_own_largest_component})
AND do not reduce own total_goal_distance AND do not complete an endpoint
AND are not opponent-blocks (Spec 4 defense-classifier vocabulary).

Opponent-block definition uses primary_class == 'blocks_opponent_closeout'
— shared with Spec 4 defense classifier so the two diagnostics cannot
diverge on what counts as a block.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Adjudication coverage classifier (§3.2)

**Files:**
- Modify: `scripts/GPU/alphazero/marathon_termination_diagnostics.py` (append `classify_adjudication_coverage` + constants).
- Modify: `tests/test_marathon_termination_diagnostics.py` (append 7 adjudication tests).

- [ ] **Step 1: Extend the test-file import**

```python
from scripts.GPU.alphazero.marathon_termination_diagnostics import (
    detect_no_progress_windows,
    classify_adjudication_coverage,
    ADJUDICATION_GATE_BUCKETS,
    NO_PROGRESS_WINDOW_SIZE,
)
```

- [ ] **Step 2: Add a per-game fixture for adjudication coverage**

Append to `tests/test_marathon_termination_diagnostics.py`:

```python
def _gc_record_state_cap(**meta_overrides):
    """A per-game (record, meta, diagnostics) triple for a 280-ply state_cap game."""
    record = {
        "iteration": 220, "game_idx": 0,
        "winner": None, "reason": "state_cap", "n_moves": 280,
        "first_total_goal_distance": 2,
        "winner_moves_with_dominant_unavailable": 0,
        "conversion_delay_plies": 0,
    }
    meta = {"reason": "state_cap", "n_moves": 280}
    meta.update(meta_overrides)
    return record, meta, []
```

- [ ] **Step 3: Write the 7 adjudication-coverage tests**

```python
def test_adjudication_coverage_blocked_by_min_top1():
    """Spec §7 test 7. self-play 'top1' → bucket 'min_top1_share'."""
    rec, meta, diag = _gc_record_state_cap(adjudication_block_reason="top1")
    assert classify_adjudication_coverage(rec, meta, diag) == "min_top1_share"


def test_adjudication_coverage_blocked_by_value_below_threshold():
    """Spec §7 test 8. self-play 'threshold' → bucket 'value_below_threshold'."""
    rec, meta, diag = _gc_record_state_cap(adjudication_block_reason="threshold")
    assert classify_adjudication_coverage(rec, meta, diag) == "value_below_threshold"


def test_adjudication_coverage_blocked_by_min_visits():
    """Spec §7 test 9. self-play 'visits' → bucket 'min_visits'."""
    rec, meta, diag = _gc_record_state_cap(adjudication_block_reason="visits")
    assert classify_adjudication_coverage(rec, meta, diag) == "min_visits"


def test_adjudication_coverage_not_attempted_when_ply_blocked():
    """Spec §7 test 10. self-play 'ply' → 'not_attempted' (adjudication
    couldn't run because the ply gate fired)."""
    rec, meta, diag = _gc_record_state_cap(adjudication_block_reason="ply")
    assert classify_adjudication_coverage(rec, meta, diag) == "not_attempted"


def test_adjudication_coverage_would_have_passed_when_none_on_state_cap():
    """Spec §7 test 11. state_cap game with adjudication_block_reason
    PRESENT as None (key exists, value is None) → 'would_have_passed'
    (bug indicator: game state-capped, key was set but no blocking gate
    recorded → an attempt should have happened). MUST NOT silently
    collapse to 'not_attempted'."""
    rec, meta, diag = _gc_record_state_cap(adjudication_block_reason=None)
    # Explicit assertion: the key IS in meta, just None-valued.
    assert "adjudication_block_reason" in meta
    assert meta["adjudication_block_reason"] is None
    assert classify_adjudication_coverage(rec, meta, diag) == "would_have_passed"


def test_adjudication_coverage_missing_signal_when_key_absent():
    """Spec §7 test 12. Old-format per-game JSON where the key isn't
    present at all → 'missing_signal' (observability gap, not a bug)."""
    rec, meta, diag = _gc_record_state_cap()
    assert "adjudication_block_reason" not in meta
    assert classify_adjudication_coverage(rec, meta, diag) == "missing_signal"


def test_adjudication_coverage_skipped_for_non_state_cap_games():
    """Spec §7 test 14. Game ending in win → returns None
    (excluded from §3.2 scope)."""
    record = {
        "iteration": 220, "game_idx": 0,
        "winner": "red", "reason": "win", "n_moves": 80,
    }
    meta = {"reason": "win", "adjudication_block_reason": None}
    assert classify_adjudication_coverage(record, meta, []) is None


def test_adjudication_gate_buckets_export_matches_spec_taxonomy():
    """Spec §3.2 enumeration. The exported tuple lists exactly the six
    bucket names so the analyzer and tests share one source of truth."""
    assert set(ADJUDICATION_GATE_BUCKETS) == {
        "not_attempted",
        "value_below_threshold",
        "min_top1_share",
        "min_visits",
        "missing_signal",
        "would_have_passed",
    }
```

- [ ] **Step 4: Run the new tests to verify they fail with ImportError**

Run: `.venv/bin/python -m pytest tests/test_marathon_termination_diagnostics.py -v -k "adjudication or ADJUDICATION"`

Expected: ImportError on `classify_adjudication_coverage`.

- [ ] **Step 5: Implement the classifier + taxonomy constants**

Append to `scripts/GPU/alphazero/marathon_termination_diagnostics.py`:

```python
ADJUDICATION_GATE_BUCKETS = (
    "not_attempted",
    "value_below_threshold",
    "min_top1_share",
    "min_visits",
    "missing_signal",
    "would_have_passed",
)


# Mapping: self_play.py's adj_blocked_by value -> spec §3.2 bucket name.
# Source: self_play.py:1213-1222 (deterministic first-failure label).
_BLOCKED_BY_TO_BUCKET = {
    "ply":       "not_attempted",
    "threshold": "value_below_threshold",
    "top1":      "min_top1_share",
    "visits":    "min_visits",
}


def classify_adjudication_coverage(
    record: dict, meta: dict, diagnostics: list,
) -> Optional[str]:
    """Classify which gate blocked adjudication for a state_cap game.

    Returns one of ADJUDICATION_GATE_BUCKETS, or None if the game does
    not qualify for §3.2 (not a state_cap game).

    Inputs (all already on disk; Task 0 outcome A):
      record: per-game goal_completion_record
      meta:   per-game `meta` dict (contains adjudication_block_reason
              from game_saver.py:146, populated by self_play.py:1248-1249)
      diagnostics: per-game goal_completion_diagnostics

    None handling matches spec §3.2 strictly:
      - key absent from meta      -> 'missing_signal' (observability gap)
      - key present, value is None -> 'would_have_passed' (bug indicator)
      - known value                -> direct mapping via _BLOCKED_BY_TO_BUCKET
    """
    meta = meta or {}
    if meta.get("reason") != "state_cap" and record.get("reason") != "state_cap":
        return None  # not in scope

    if "adjudication_block_reason" not in meta:
        # Old-format JSON written before the field was persisted.
        return "missing_signal"

    reason = meta["adjudication_block_reason"]
    if reason in _BLOCKED_BY_TO_BUCKET:
        return _BLOCKED_BY_TO_BUCKET[reason]
    if reason is None:
        # Key present but null: an attempt should have happened (adjudication
        # was on for our 220-229 launches). Treat as a bug indicator. If
        # adjudication was actually disabled in a different launch config,
        # the report MUST surface this as a caveat alongside the count.
        return "would_have_passed"
    # Unknown string value — defensive fallback.
    return "missing_signal"
```

- [ ] **Step 6: Run the adjudication tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_marathon_termination_diagnostics.py -v -k "adjudication or ADJUDICATION"`

Expected: 7 passed.

- [ ] **Step 7: Run all marathon-termination tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_marathon_termination_diagnostics.py -v`

Expected: 14 passed (6 from Task 2 + 8 from Task 3 — the None-handling split added one test).

- [ ] **Step 8: Commit**

```bash
git add scripts/GPU/alphazero/marathon_termination_diagnostics.py tests/test_marathon_termination_diagnostics.py
git commit -m "$(cat <<'EOF'
feat(diagnostics): adjudication-coverage classifier (Spec marathon-termination §3.2)

Maps self_play.py's deterministic adj_blocked_by value
(ply/threshold/visits/top1/None) to the spec's six-bucket taxonomy
(not_attempted/value_below_threshold/min_top1_share/min_visits/
missing_signal/would_have_passed). Reads meta.adjudication_block_reason
already persisted per-game by game_saver.py:146 — Task 0 outcome A,
no self-play change required.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Resign-gate breakdown by game-length partition (§3.3)

**Files:**
- Modify: `scripts/GPU/alphazero/marathon_termination_diagnostics.py` (append).
- Modify: `tests/test_marathon_termination_diagnostics.py` (append).

- [ ] **Step 1: Extend the test-file import**

```python
from scripts.GPU.alphazero.marathon_termination_diagnostics import (
    detect_no_progress_windows,
    classify_adjudication_coverage,
    compute_resign_gate_breakdown,
    game_length_bucket,
    ADJUDICATION_GATE_BUCKETS,
    GAME_LENGTH_BUCKETS,
    NO_PROGRESS_WINDOW_SIZE,
)
```

- [ ] **Step 2: Write the resign-gate tests**

Append to `tests/test_marathon_termination_diagnostics.py`:

```python
def _losing_side_ply(*, search_score: float, top1_share: float,
                     visit_count: int = 250, ply: int = 200,
                     side_to_move: str = "black"):
    """A goal_completion_diagnostics entry suitable for resign-gate scoring.
    'side_to_move' here is the LOSING side's own-move; the helper uses
    the convention that root_summary.q_value is the score from the
    side-to-move's perspective."""
    return {
        "ply": ply, "side_to_move": side_to_move,
        "root_summary": {
            "q_value": search_score,
            "visit_count": visit_count,
        },
        "root_top1_share": top1_share,
    }


def _resign_thresholds():
    """Production defaults from the 220-229 launch command (memory entry)."""
    return dict(
        resign_threshold=-0.945,
        resign_min_ply=80,
        resign_min_visits=200,
        resign_min_top1_share=0.102,
    )


def test_resign_gate_breakdown_separates_value_hits_from_eligible_hits():
    """Spec §7 test 15 + 16. Game where value crosses threshold often
    but visits/top1 sometimes fail: value_hits >= eligible_hits >= blocked_by_top1."""
    diag = [
        # Hit 1: value crosses, all gates pass except top1 (eligible, blocked by top1).
        _losing_side_ply(search_score=-0.95, top1_share=0.05, visit_count=300, ply=200),
        # Hit 2: value crosses, all gates pass (eligible, NOT blocked).
        _losing_side_ply(search_score=-0.97, top1_share=0.20, visit_count=300, ply=210),
        # Hit 3: value crosses but visits fail (value_hit yes, eligible no).
        _losing_side_ply(search_score=-0.96, top1_share=0.20, visit_count=100, ply=220),
        # Non-hit: value below threshold doesn't qualify (no value_hit).
        _losing_side_ply(search_score=-0.50, top1_share=0.20, visit_count=300, ply=230),
    ]
    record = {"winner": "red", "n_moves": 250}  # losing side: black
    out = compute_resign_gate_breakdown(record, diag, losing_side="black", **_resign_thresholds())
    assert out["value_hits"] == 3
    assert out["eligible_hits"] == 2          # hits 1 and 2 (hit 3 fails visits)
    assert out["blocked_by_top1"] == 1        # hit 1 only
    # over_value_hits = 1/3, over_eligible_hits = 1/2.
    assert abs(out["top1_block_rate_over_value_hits"] - 1/3) < 1e-9
    assert abs(out["top1_block_rate_over_eligible_hits"] - 1/2) < 1e-9


def test_resign_gate_breakdown_empty_when_no_value_hits():
    """Game where the loser never crossed resign_threshold → all counts 0,
    rates 0.0 (not NaN)."""
    diag = [
        _losing_side_ply(search_score=-0.50, top1_share=0.20, ply=200),
    ]
    record = {"winner": "red", "n_moves": 250}
    out = compute_resign_gate_breakdown(record, diag, losing_side="black", **_resign_thresholds())
    assert out["value_hits"] == 0
    assert out["eligible_hits"] == 0
    assert out["blocked_by_top1"] == 0
    assert out["top1_block_rate_over_value_hits"] == 0.0
    assert out["top1_block_rate_over_eligible_hits"] == 0.0


def test_game_length_bucket_partitions():
    """Spec §3.3 game-length partition: short / mid / long."""
    assert game_length_bucket(50) == "short"
    assert game_length_bucket(100) == "short"
    assert game_length_bucket(101) == "mid"
    assert game_length_bucket(200) == "mid"
    assert game_length_bucket(201) == "long"
    assert game_length_bucket(280) == "long"
    assert set(GAME_LENGTH_BUCKETS) == {"short", "mid", "long"}


def test_resign_separates_no_value_signal_from_blocked_by_top1():
    """Spec §7 test 16. A game with no value hits is distinguishable from
    a game with high blocked_by_top1 rate."""
    record = {"winner": "red", "n_moves": 250}
    diag_no_value = [_losing_side_ply(search_score=-0.50, top1_share=0.20)]
    diag_high_block = [
        _losing_side_ply(search_score=-0.97, top1_share=0.05, visit_count=300, ply=200),
        _losing_side_ply(search_score=-0.97, top1_share=0.05, visit_count=300, ply=210),
    ]
    out_no_value = compute_resign_gate_breakdown(record, diag_no_value, losing_side="black", **_resign_thresholds())
    out_high_block = compute_resign_gate_breakdown(record, diag_high_block, losing_side="black", **_resign_thresholds())
    # Distinguishable: no-value has zero value_hits; high-block has positive value_hits + positive blocked_by_top1.
    assert out_no_value["value_hits"] == 0
    assert out_no_value["blocked_by_top1"] == 0
    assert out_high_block["value_hits"] == 2
    assert out_high_block["blocked_by_top1"] == 2
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_marathon_termination_diagnostics.py -v -k "resign or game_length"`

Expected: ImportError on `compute_resign_gate_breakdown` / `game_length_bucket` / `GAME_LENGTH_BUCKETS`.

- [ ] **Step 4: Implement the resign-gate breakdown + game-length partition**

Append to `scripts/GPU/alphazero/marathon_termination_diagnostics.py`:

```python
GAME_LENGTH_BUCKETS = ("short", "mid", "long")


def game_length_bucket(n_moves: int) -> str:
    """§3.3 partition: short (<=100), mid (101-200), long (>200)."""
    n = int(n_moves or 0)
    if n <= 100:
        return "short"
    if n <= 200:
        return "mid"
    return "long"


def compute_resign_gate_breakdown(
    record: dict,
    diagnostics: list,
    *,
    losing_side: str,
    resign_threshold: float,
    resign_min_ply: int,
    resign_min_visits: int,
    resign_min_top1_share: float,
) -> dict:
    """Spec §3.3 resign-gate breakdown for the losing side, looking at
    its last 40 plies (plies where side_to_move == losing_side).

    Returns:
      value_hits           : count of own-plies where q_value < resign_threshold
      eligible_hits        : count of value_hits also satisfying
                             ply >= resign_min_ply AND visits >= resign_min_visits
      blocked_by_top1      : count of eligible_hits where top1_share <
                             resign_min_top1_share
      final_eval_below_thr : at the last own-ply for the loser,
                             q_value < resign_threshold
      top1_block_rate_over_value_hits     : blocked_by_top1 / max(value_hits, 1)
                                            (returns 0.0 when value_hits == 0)
      top1_block_rate_over_eligible_hits  : blocked_by_top1 / max(eligible_hits, 1)
                                            (returns 0.0 when eligible_hits == 0)
    """
    own = [e for e in (diagnostics or []) if e.get("side_to_move") == losing_side]
    own = own[-40:]  # last 40 own-plies

    value_hits = 0
    eligible_hits = 0
    blocked_by_top1 = 0
    for e in own:
        rs = e.get("root_summary") or {}
        q = rs.get("q_value")
        visits = int(rs.get("visit_count") or 0)
        top1 = e.get("root_top1_share")
        ply = int(e.get("ply") or 0)
        if q is None or q >= resign_threshold:
            continue
        value_hits += 1
        if ply >= resign_min_ply and visits >= resign_min_visits:
            eligible_hits += 1
            if top1 is not None and float(top1) < resign_min_top1_share:
                blocked_by_top1 += 1

    final_eval_below_thr = bool(
        own and (own[-1].get("root_summary") or {}).get("q_value") is not None
        and (own[-1]["root_summary"]["q_value"] < resign_threshold)
    )

    def _rate(num, denom):
        return float(num) / float(denom) if denom > 0 else 0.0

    return {
        "value_hits": value_hits,
        "eligible_hits": eligible_hits,
        "blocked_by_top1": blocked_by_top1,
        "final_eval_below_thr": final_eval_below_thr,
        "top1_block_rate_over_value_hits": _rate(blocked_by_top1, value_hits),
        "top1_block_rate_over_eligible_hits": _rate(blocked_by_top1, eligible_hits),
    }
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_marathon_termination_diagnostics.py -v -k "resign or game_length"`

Expected: 4 passed.

- [ ] **Step 6: Run all marathon-termination tests**

Run: `.venv/bin/python -m pytest tests/test_marathon_termination_diagnostics.py -v`

Expected: 18 passed (6 + 8 + 4).

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/marathon_termination_diagnostics.py tests/test_marathon_termination_diagnostics.py
git commit -m "$(cat <<'EOF'
feat(diagnostics): resign-gate breakdown by game-length partition (Spec marathon-termination §3.3)

Three-count split (value_hits >= eligible_hits >= blocked_by_top1) on
the losing side's last 40 own-plies. Two derived rates: over_value_hits
(was the value signal present but blocked) and over_eligible_hits (of
resigns that ONLY needed top1 to pass, what fraction were blocked).

Partition: short (<=100) / mid (<=200) / long (>200). The two-rate
split distinguishes 'no losing value signal' (low value_hits) from
'top1 gate prevented resignation' (high over_eligible_hits).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Value-uncertain guard predicate (§5.1)

**Files:**
- Modify: `scripts/GPU/alphazero/marathon_termination_diagnostics.py` (append).
- Modify: `tests/test_marathon_termination_diagnostics.py` (append).

- [ ] **Step 1: Extend the test-file import**

```python
from scripts.GPU.alphazero.marathon_termination_diagnostics import (
    detect_no_progress_windows,
    classify_adjudication_coverage,
    compute_resign_gate_breakdown,
    value_uncertain_guard,
    game_length_bucket,
    ADJUDICATION_GATE_BUCKETS,
    GAME_LENGTH_BUCKETS,
    NO_PROGRESS_WINDOW_SIZE,
)
```

- [ ] **Step 2: Write the three guard tests**

```python
def test_value_uncertain_guard_blocks_termination_when_neutral():
    """Spec §7 test 17. Last 10 own-plies for both sides with |score|<0.30
    → guard returns True (do not terminate)."""
    diagnostics = []
    for ply in range(220, 240):
        side = "red" if ply % 2 == 0 else "black"
        diagnostics.append({
            "ply": ply, "side_to_move": side,
            "root_summary": {"q_value": 0.05},  # near neutral
        })
    assert value_uncertain_guard(diagnostics) is True


def test_value_uncertain_guard_blocks_termination_when_oscillatory():
    """Spec §7 test 18. Last 10 own-plies with >=3 sign-flips → guard True."""
    diagnostics = []
    # Build a clearly oscillatory sequence for both sides.
    for i, score in enumerate([0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5,
                                0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5]):
        diagnostics.append({
            "ply": 220 + i, "side_to_move": "red" if i % 2 == 0 else "black",
            "root_summary": {"q_value": score},
        })
    assert value_uncertain_guard(diagnostics) is True


def test_value_uncertain_guard_allows_termination_when_stable_losing():
    """Spec §7 test 19. Last 10 own-plies stably below -0.30 for the loser
    AND above 0.30 for the winner → guard returns False (terminate is safe)."""
    diagnostics = []
    for i in range(20):
        side = "red" if i % 2 == 0 else "black"
        score = 0.85 if side == "red" else -0.85
        diagnostics.append({
            "ply": 220 + i, "side_to_move": side,
            "root_summary": {"q_value": score},
        })
    assert value_uncertain_guard(diagnostics) is False
```

- [ ] **Step 3: Run the guard tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_marathon_termination_diagnostics.py -v -k "value_uncertain"`

Expected: ImportError on `value_uncertain_guard`.

- [ ] **Step 4: Implement the guard predicate**

Append to `scripts/GPU/alphazero/marathon_termination_diagnostics.py`:

```python
def value_uncertain_guard(
    diagnostics: list,
    *,
    window_per_side: int = 10,
    neutral_abs_threshold: float = 0.30,
    sign_flip_min: int = 3,
) -> bool:
    """Spec §5.1 value-uncertain guard.

    Returns True (DO NOT terminate) when EITHER:
      - both sides' last `window_per_side` own-plies have
        |q_value| < neutral_abs_threshold for ALL plies in the window
      - the combined last 2*window_per_side plies have >= sign_flip_min
        sign-flips in q_value

    Returns False (termination is safe per this guard) when both sides
    show stable, non-neutral assessments.

    Implementation note: callers in self-play / training enforce this at
    the termination call-site, not the diagnostic call-site. This
    predicate is pure and side-effect-free.
    """
    own_red = [
        (e["root_summary"] or {}).get("q_value")
        for e in (diagnostics or [])
        if e.get("side_to_move") == "red" and (e.get("root_summary") or {}).get("q_value") is not None
    ]
    own_black = [
        (e["root_summary"] or {}).get("q_value")
        for e in (diagnostics or [])
        if e.get("side_to_move") == "black" and (e.get("root_summary") or {}).get("q_value") is not None
    ]
    last_red = own_red[-window_per_side:]
    last_black = own_black[-window_per_side:]

    # Neutral both-sides condition.
    both_neutral = (
        len(last_red) >= 1
        and len(last_black) >= 1
        and all(abs(q) < neutral_abs_threshold for q in last_red)
        and all(abs(q) < neutral_abs_threshold for q in last_black)
    )
    if both_neutral:
        return True

    # Oscillatory: count sign flips across the interleaved last 2*window plies.
    combined = []
    for e in (diagnostics or []):
        rs = e.get("root_summary") or {}
        q = rs.get("q_value")
        if q is None:
            continue
        combined.append(float(q))
    tail = combined[-2 * window_per_side:]
    sign_flips = sum(
        1 for a, b in zip(tail, tail[1:])
        if (a > 0 and b < 0) or (a < 0 and b > 0)
    )
    return sign_flips >= sign_flip_min
```

- [ ] **Step 5: Run guard tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_marathon_termination_diagnostics.py -v -k "value_uncertain"`

Expected: 3 passed.

- [ ] **Step 6: Run all marathon-termination tests**

Run: `.venv/bin/python -m pytest tests/test_marathon_termination_diagnostics.py -v`

Expected: 21 passed (6 + 8 + 4 + 3).

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/marathon_termination_diagnostics.py tests/test_marathon_termination_diagnostics.py
git commit -m "$(cat <<'EOF'
feat(diagnostics): value-uncertain guard predicate (Spec marathon-termination §5.1)

Pure predicate to be consulted by any termination knob before acting:
returns True (do not terminate) when both sides' last 10 own-plies are
near-neutral (|q|<0.30) OR the combined last 20 plies show >=3 sign-flips.

Enforced at the termination call-site, not the diagnostic call-site.
Pure / side-effect-free.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Aggregator + analyzer wiring (CSV + report)

**Files:**
- Modify: `scripts/GPU/alphazero/marathon_termination_diagnostics.py` (append `aggregate_marathon_termination`).
- Modify: `scripts/twixt_replay_analyzer.py` (add CSV writer + report formatter + main-path wiring).
- Create: `tests/test_analyzer_marathon_termination.py`.
- Modify: `tests/test_marathon_termination_diagnostics.py` (append aggregator tests).

- [ ] **Step 1: Add aggregator test fixture and aggregator tests**

Append to `tests/test_marathon_termination_diagnostics.py`:

```python
from scripts.GPU.alphazero.marathon_termination_diagnostics import (
    aggregate_marathon_termination,
)


def _per_game(iteration, game_idx, *, reason="win", n_moves=80, winner="red",
              adj_block=None, diagnostics=None):
    record = {
        "iteration": iteration, "game_idx": game_idx,
        "winner": winner, "reason": reason, "n_moves": n_moves,
        "first_total_goal_distance": 2,
        "winner_moves_with_dominant_unavailable": 0,
        "conversion_delay_plies": 0,
    }
    meta = {"reason": reason, "n_moves": n_moves}
    if adj_block is not None:
        meta["adjudication_block_reason"] = adj_block
    return record, meta, (diagnostics or [])


def test_aggregate_marathon_termination_per_iter_and_range_totals():
    """Spec §7 test 20. 3 games across 2 iters → correct per-iter rows + range-total row."""
    games = [
        _per_game(220, 0, reason="state_cap", n_moves=280, winner=None, adj_block="top1"),
        _per_game(220, 1, reason="state_cap", n_moves=280, winner=None, adj_block="threshold"),
        _per_game(221, 0, reason="state_cap", n_moves=280, winner=None, adj_block="top1"),
    ]
    resign_cfg = dict(
        resign_threshold=-0.945, resign_min_ply=80,
        resign_min_visits=200, resign_min_top1_share=0.102,
    )
    out = aggregate_marathon_termination(games, **resign_cfg)
    assert out["per_iter"][220]["state_cap_280_games"] == 2
    assert out["per_iter"][221]["state_cap_280_games"] == 1
    assert out["per_iter"][220]["adjudication_gate_counts"]["min_top1_share"] == 1
    assert out["per_iter"][220]["adjudication_gate_counts"]["value_below_threshold"] == 1
    assert out["per_iter"][221]["adjudication_gate_counts"]["min_top1_share"] == 1
    assert out["range_total"]["state_cap_280_games"] == 3
    assert out["range_total"]["adjudication_gate_counts"]["min_top1_share"] == 2
    assert out["range_total"]["adjudication_gate_counts"]["value_below_threshold"] == 1


def test_aggregate_marathon_termination_no_progress_window_mean():
    """Spec §3.4. Per-iter mean of detected no-progress windows across games."""
    games = [
        _per_game(220, 0, reason="win", n_moves=80, winner="red",
                  diagnostics=[]),
        _per_game(220, 1, reason="win", n_moves=80, winner="red",
                  diagnostics=[
                      {"ply": p, "side_to_move": "red" if i % 2 == 0 else "black",
                       "selected_move_classification": {"primary_class": "redundant_reinforcement"}}
                      for i, p in enumerate(range(50, 80))
                  ]),
    ]
    resign_cfg = dict(
        resign_threshold=-0.945, resign_min_ply=80,
        resign_min_visits=200, resign_min_top1_share=0.102,
    )
    out = aggregate_marathon_termination(games, **resign_cfg)
    # Iter 220 has two games; one has 0 windows, the other has at least one
    # 15-window run on one side. Mean across games > 0.
    assert out["per_iter"][220]["mean_no_progress_windows_per_game"] > 0.0
```

- [ ] **Step 2: Run aggregator tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_marathon_termination_diagnostics.py -v -k "aggregate"`

Expected: ImportError on `aggregate_marathon_termination`.

- [ ] **Step 3: Implement the aggregator**

Append to `scripts/GPU/alphazero/marathon_termination_diagnostics.py`:

```python
def _losing_side(record: dict) -> Optional[str]:
    winner = record.get("winner")
    if winner == "red":
        return "black"
    if winner == "black":
        return "red"
    return None


def aggregate_marathon_termination(
    games: Iterable[Tuple[dict, dict, list]],
    *,
    resign_threshold: float,
    resign_min_ply: int,
    resign_min_visits: int,
    resign_min_top1_share: float,
) -> dict:
    """Aggregate per-game diagnostics into per-iter and range tables.

    games: iterable of (record, meta, diagnostics) triples.

    Returns the structure consumed by write_marathon_termination_csv +
    format_marathon_termination_report (see spec §4.2/§4.3).
    """
    def _empty_iter_row() -> dict:
        return {
            "games_total": 0,
            "state_cap_280_games": 0,
            "no_progress_window_counts": [],  # per-game ints
            "adjudication_gate_counts": {b: 0 for b in ADJUDICATION_GATE_BUCKETS},
            "resign_top1_block_over_value_hits": {b: [] for b in GAME_LENGTH_BUCKETS},
            "resign_top1_block_over_eligible_hits": {b: [] for b in GAME_LENGTH_BUCKETS},
            # Observability counters (spec §3.1 follow-up — surface coverage
            # gaps so a zero no-progress rate is not confused with a
            # no-data situation).
            "diagnostics_entries_red": 0,
            "diagnostics_entries_black": 0,
            "no_progress_observable_games_red": 0,    # games with >=15 red own-entries
            "no_progress_observable_games_black": 0,  # games with >=15 black own-entries
        }

    per_iter: dict = defaultdict(_empty_iter_row)

    for record, meta, diagnostics in games:
        it = record.get("iteration")
        row = per_iter[it]
        row["games_total"] += 1

        # Observability: count diagnostics entries by side per game.
        red_entries = sum(1 for e in (diagnostics or []) if e.get("side_to_move") == "red")
        black_entries = sum(1 for e in (diagnostics or []) if e.get("side_to_move") == "black")
        row["diagnostics_entries_red"] += red_entries
        row["diagnostics_entries_black"] += black_entries
        if red_entries >= NO_PROGRESS_WINDOW_SIZE:
            row["no_progress_observable_games_red"] += 1
        if black_entries >= NO_PROGRESS_WINDOW_SIZE:
            row["no_progress_observable_games_black"] += 1

        # §3.1 — no-progress window count summed across both sides.
        npw = (
            detect_no_progress_windows(diagnostics, side="red")
            + detect_no_progress_windows(diagnostics, side="black")
        )
        row["no_progress_window_counts"].append(npw)

        # §3.2 — only for state_cap 280-ply games.
        n_moves = int(record.get("n_moves") or 0)
        if record.get("reason") == "state_cap" and n_moves == 280:
            row["state_cap_280_games"] += 1
            bucket = classify_adjudication_coverage(record, meta, diagnostics)
            if bucket:
                row["adjudication_gate_counts"][bucket] += 1

        # §3.3 — resign-gate breakdown (only when winner is known).
        loser = _losing_side(record)
        if loser is not None:
            br = compute_resign_gate_breakdown(
                record, diagnostics,
                losing_side=loser,
                resign_threshold=resign_threshold,
                resign_min_ply=resign_min_ply,
                resign_min_visits=resign_min_visits,
                resign_min_top1_share=resign_min_top1_share,
            )
            len_bucket = game_length_bucket(record.get("n_moves") or 0)
            row["resign_top1_block_over_value_hits"][len_bucket].append(
                br["top1_block_rate_over_value_hits"]
            )
            row["resign_top1_block_over_eligible_hits"][len_bucket].append(
                br["top1_block_rate_over_eligible_hits"]
            )

    # Compute derived per-iter values.
    def _finalize(row: dict) -> dict:
        npw_list = row["no_progress_window_counts"]
        mean_npw = sum(npw_list) / len(npw_list) if npw_list else 0.0
        finalized = {
            "games_total": row["games_total"],
            "state_cap_280_games": row["state_cap_280_games"],
            "mean_no_progress_windows_per_game": round(mean_npw, 3),
            "adjudication_gate_counts": dict(row["adjudication_gate_counts"]),
            "mean_resign_top1_block_rate_over_value_hits": {
                b: round(sum(vs) / len(vs), 3) if vs else 0.0
                for b, vs in row["resign_top1_block_over_value_hits"].items()
            },
            "mean_resign_top1_block_rate_over_eligible_hits": {
                b: round(sum(vs) / len(vs), 3) if vs else 0.0
                for b, vs in row["resign_top1_block_over_eligible_hits"].items()
            },
            "observability": {
                "diagnostics_entries_red":   row["diagnostics_entries_red"],
                "diagnostics_entries_black": row["diagnostics_entries_black"],
                "no_progress_observable_games_red":   row["no_progress_observable_games_red"],
                "no_progress_observable_games_black": row["no_progress_observable_games_black"],
            },
        }
        return finalized

    per_iter_final = {it: _finalize(row) for it, row in per_iter.items()}

    # Range totals.
    range_row = _empty_iter_row()
    for it, row in per_iter.items():
        range_row["games_total"] += row["games_total"]
        range_row["state_cap_280_games"] += row["state_cap_280_games"]
        range_row["no_progress_window_counts"].extend(row["no_progress_window_counts"])
        for b, c in row["adjudication_gate_counts"].items():
            range_row["adjudication_gate_counts"][b] += c
        for b in GAME_LENGTH_BUCKETS:
            range_row["resign_top1_block_over_value_hits"][b].extend(
                row["resign_top1_block_over_value_hits"][b]
            )
            range_row["resign_top1_block_over_eligible_hits"][b].extend(
                row["resign_top1_block_over_eligible_hits"][b]
            )
        range_row["diagnostics_entries_red"]   += row["diagnostics_entries_red"]
        range_row["diagnostics_entries_black"] += row["diagnostics_entries_black"]
        range_row["no_progress_observable_games_red"]   += row["no_progress_observable_games_red"]
        range_row["no_progress_observable_games_black"] += row["no_progress_observable_games_black"]
    range_final = _finalize(range_row)

    return {
        "per_iter": per_iter_final,
        "range_total": range_final,
    }
```

- [ ] **Step 4: Run aggregator tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_marathon_termination_diagnostics.py -v`

Expected: 23 passed (21 from prior + 2 new).

- [ ] **Step 5: Add CSV writer + report formatter to analyzer**

Find the long-tail-buckets CSV writer + report formatter in `scripts/twixt_replay_analyzer.py` (added in commit `e6a0246`, search for `write_goal_completion_long_tail_buckets_csv` or `format_long_tail_bucket_report`). Add the marathon-termination siblings immediately after.

Locate the long-tail report formatter (around line 3686 — `def format_long_tail_bucket_report`). Append after it:

```python
def write_marathon_termination_csv(out_path: str, agg: dict) -> str:
    """Spec marathon-termination §4.2. Long format; one row per iter +
    one range-total row (iteration=-1)."""
    from scripts.GPU.alphazero.marathon_termination_diagnostics import (
        ADJUDICATION_GATE_BUCKETS, GAME_LENGTH_BUCKETS,
    )
    fields = [
        "iteration", "games_total", "state_cap_280_games",
        "mean_no_progress_windows_per_game",
    ] + [f"adjudication_gate_{b}" for b in ADJUDICATION_GATE_BUCKETS] + [
        f"mean_resign_top1_block_rate_over_value_hits_{b}" for b in GAME_LENGTH_BUCKETS
    ] + [
        f"mean_resign_top1_block_rate_over_eligible_hits_{b}" for b in GAME_LENGTH_BUCKETS
    ] + [
        # Observability counters: surface coverage gaps so zero no-progress
        # rates can be distinguished from missing data.
        "diagnostics_entries_red",
        "diagnostics_entries_black",
        "no_progress_observable_games_red",
        "no_progress_observable_games_black",
    ]

    def _row_dict(iteration, row):
        d = {
            "iteration": iteration,
            "games_total": row["games_total"],
            "state_cap_280_games": row["state_cap_280_games"],
            "mean_no_progress_windows_per_game": row["mean_no_progress_windows_per_game"],
        }
        for b in ADJUDICATION_GATE_BUCKETS:
            d[f"adjudication_gate_{b}"] = row["adjudication_gate_counts"][b]
        for b in GAME_LENGTH_BUCKETS:
            d[f"mean_resign_top1_block_rate_over_value_hits_{b}"] = \
                row["mean_resign_top1_block_rate_over_value_hits"][b]
            d[f"mean_resign_top1_block_rate_over_eligible_hits_{b}"] = \
                row["mean_resign_top1_block_rate_over_eligible_hits"][b]
        obs = row["observability"]
        d["diagnostics_entries_red"]   = obs["diagnostics_entries_red"]
        d["diagnostics_entries_black"] = obs["diagnostics_entries_black"]
        d["no_progress_observable_games_red"]   = obs["no_progress_observable_games_red"]
        d["no_progress_observable_games_black"] = obs["no_progress_observable_games_black"]
        return d

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for it in sorted(agg["per_iter"].keys()):
            w.writerow(_row_dict(it, agg["per_iter"][it]))
        w.writerow(_row_dict(-1, agg["range_total"]))
    return out_path


def format_marathon_termination_report(agg: dict, range_label: str = "") -> list:
    """Spec marathon-termination §4.3. Report section + decision-rule
    suggestion line."""
    from scripts.GPU.alphazero.marathon_termination_diagnostics import (
        ADJUDICATION_GATE_BUCKETS, GAME_LENGTH_BUCKETS,
    )
    rt = agg["range_total"]
    suffix = f" ({range_label})" if range_label else ""
    lines = []
    lines.append(f"Marathon termination diagnostics{suffix}")
    lines.append("=" * (32 + len(suffix)))
    lines.append(f"state_cap 280-ply games: {rt['state_cap_280_games']}")
    lines.append("  adjudication gate blocked by:")
    for b in ADJUDICATION_GATE_BUCKETS:
        lines.append(f"    {b+':':28s} {rt['adjudication_gate_counts'][b]}")
    lines.append("")
    lines.append(f"No-progress windows (length {15} own-moves, structural-only):")
    lines.append(f"  mean per game: {rt['mean_no_progress_windows_per_game']:.2f}")
    obs = rt["observability"]
    games_total = rt["games_total"] or 1
    red_obs_pct = obs["no_progress_observable_games_red"] / games_total * 100
    black_obs_pct = obs["no_progress_observable_games_black"] / games_total * 100
    lines.append(
        f"  observability: red entries={obs['diagnostics_entries_red']}, "
        f"black entries={obs['diagnostics_entries_black']}, "
        f"games with >=15 own-entries: red {red_obs_pct:.1f}% / black {black_obs_pct:.1f}%"
    )
    if red_obs_pct < 50 or black_obs_pct < 50:
        lines.append(
            "  WARNING: observability < 50% for at least one side — "
            "no-progress rates may be undercounted (diagnostics list may be "
            "capped or biased toward winner-side / closeout-scoped plies)."
        )
    lines.append("")
    lines.append("Resign top1-gate block rate (losing-side, last 40 plies):")
    lines.append("  over value_hits:")
    for b in GAME_LENGTH_BUCKETS:
        v = rt["mean_resign_top1_block_rate_over_value_hits"][b]
        lines.append(f"    {b+' games:':14s} {v*100:>5.1f}%")
    lines.append("  over eligible_hits:")
    for b in GAME_LENGTH_BUCKETS:
        v = rt["mean_resign_top1_block_rate_over_eligible_hits"][b]
        lines.append(f"    {b+' games:':14s} {v*100:>5.1f}%")
    lines.append("")
    lines.append(f"Suggested termination action: {_marathon_termination_decision(rt)}")
    return lines


def _marathon_termination_decision(rt: dict) -> str:
    """Spec §5 decision rule applied to the range-total row.
    Returns a single short suggestion string."""
    gate = rt["adjudication_gate_counts"]
    total_state_cap = rt["state_cap_280_games"]
    long_eligible = rt["mean_resign_top1_block_rate_over_eligible_hits"]["long"]
    short_eligible = rt["mean_resign_top1_block_rate_over_eligible_hits"]["short"]

    if total_state_cap > 0:
        share_value = gate["value_below_threshold"] / total_state_cap
        share_top1 = gate["min_top1_share"] / total_state_cap
        share_visits = gate["min_visits"] / total_state_cap
        if share_value > 0.50:
            return "do not terminate; consider lowering adjudicate_threshold for very late plies only"
        if share_top1 > 0.50:
            return "relax --adjudicate-min-top1-share in late-game tier (>=260 ply)"
        if share_visits > 0.50:
            return "investigate MCTS budget (visits below min at relevant plies) — not adjudication"

    if short_eligible > 0 and long_eligible > 2 * short_eligible:
        return "relax --resign-min-top1-share late-game (>=200 ply) — top1 gating too strict on hopeless positions"

    if rt["mean_no_progress_windows_per_game"] > 2.0:
        return "candidate: early state-cap on K=2 consecutive no-progress windows (default-off; treatment requires explicit user approval per spec §5)"

    return "no dominant remedy — hand-review 5-10 representative cases before another knob change"
```

- [ ] **Step 6: Wire the aggregator into the analyzer's main path**

Find the long-tail wiring block (search for `from scripts.GPU.alphazero.long_tail_bucket_classifier import` — at around line 5060 in the analyzer). Immediately after that block, add the marathon-termination wiring:

```python
    # Spec marathon-termination §4 — diagnostics CSV + report section.
    from scripts.GPU.alphazero.marathon_termination_diagnostics import (
        aggregate_marathon_termination,
    )
    marathon_triples = [
        (r.get("goal_completion_record"),
         r.get("meta") or {},
         r.get("goal_completion_diagnostics") or [])
        for r in (replays or [])
        if isinstance(r, dict) and r.get("goal_completion_record")
    ]
    # Resign thresholds for the breakdown come from launch-args; falls back
    # to the production defaults from the 220-229 launch command.
    marathon_agg = aggregate_marathon_termination(
        marathon_triples,
        resign_threshold=getattr(args, "resign_threshold", -0.945) if args else -0.945,
        resign_min_ply=getattr(args, "resign_min_ply", 80) if args else 80,
        resign_min_visits=getattr(args, "resign_min_visits", 200) if args else 200,
        resign_min_top1_share=getattr(args, "resign_min_top1_share", 0.102) if args else 0.102,
    )
    write_marathon_termination_csv(
        os.path.join(out_dir, _suffixed("marathon_termination_by_iter", "csv", suffix)),
        marathon_agg,
    )
    summary["marathon_termination"] = marathon_agg
    _marathon_range_label = suffix.lstrip("_") if suffix else ""
```

Then find the long-tail bucket report-render call (`format_long_tail_bucket_report(long_tail_agg, range_label=_long_tail_range_label)`) and add the marathon-termination render immediately after:

```python
    marathon_lines = format_marathon_termination_report(marathon_agg, range_label=_marathon_range_label)
    if marathon_lines:
        lines.append("")
        lines.extend(marathon_lines)
```

- [ ] **Step 7: Add analyzer-integration tests**

Create `tests/test_analyzer_marathon_termination.py`:

```python
"""Analyzer-side integration tests for marathon-termination diagnostics."""
import sys, csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.twixt_replay_analyzer import (
    write_marathon_termination_csv,
    format_marathon_termination_report,
)
from scripts.GPU.alphazero.marathon_termination_diagnostics import (
    aggregate_marathon_termination,
)


def _per_game(iteration, *, reason="win", n_moves=80, winner="red",
              adj_block=None, diagnostics=None):
    record = {
        "iteration": iteration, "game_idx": 0,
        "winner": winner, "reason": reason, "n_moves": n_moves,
        "first_total_goal_distance": 2,
        "winner_moves_with_dominant_unavailable": 0,
        "conversion_delay_plies": 0,
    }
    meta = {"reason": reason, "n_moves": n_moves}
    if adj_block is not None:
        meta["adjudication_block_reason"] = adj_block
    return record, meta, (diagnostics or [])


def _cfg():
    return dict(resign_threshold=-0.945, resign_min_ply=80,
                resign_min_visits=200, resign_min_top1_share=0.102)


def test_analyzer_writes_marathon_termination_csv(tmp_path):
    """Spec §7 test. CSV emits per-iter rows + a range-total row at iteration=-1."""
    games = [
        _per_game(220, reason="state_cap", n_moves=280, winner=None, adj_block="top1"),
        _per_game(221, reason="state_cap", n_moves=280, winner=None, adj_block="threshold"),
    ]
    agg = aggregate_marathon_termination(games, **_cfg())
    out = tmp_path / "marathon.csv"
    write_marathon_termination_csv(str(out), agg)
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 3  # 220, 221, -1
    iters = sorted(int(r["iteration"]) for r in rows)
    assert iters == [-1, 220, 221]
    # Range-total row's adjudication_gate_min_top1_share == 1, value_below_threshold == 1.
    rt = next(r for r in rows if int(r["iteration"]) == -1)
    assert int(rt["adjudication_gate_min_top1_share"]) == 1
    assert int(rt["adjudication_gate_value_below_threshold"]) == 1


def test_analyzer_report_includes_marathon_section_with_decision_suggestion():
    """Spec §7 test. Report section header + decision-rule line are rendered."""
    games = [
        _per_game(220, reason="state_cap", n_moves=280, winner=None, adj_block="top1")
        for _ in range(10)
    ]
    agg = aggregate_marathon_termination(games, **_cfg())
    lines = format_marathon_termination_report(agg, range_label="220-229")
    text = "\n".join(lines)
    assert "Marathon termination diagnostics (220-229)" in text
    assert "state_cap 280-ply games: 10" in text
    assert "adjudication gate blocked by:" in text
    assert "Suggested termination action:" in text
    # 10/10 blocked by min_top1_share → suggestion mentions adjudicate-min-top1-share.
    assert "adjudicate-min-top1-share" in text


def test_format_marathon_termination_report_neutral_when_no_signal_dominates():
    """When no remedy clearly dominates, suggestion line says 'no dominant remedy'."""
    # Single non-state-cap game with no resign-gate-block signal.
    games = [_per_game(220, reason="win", n_moves=80, winner="red")]
    agg = aggregate_marathon_termination(games, **_cfg())
    lines = format_marathon_termination_report(agg, range_label="220-229")
    text = "\n".join(lines)
    assert "no dominant remedy" in text


def test_report_emits_observability_warning_when_diagnostics_sparse(tmp_path):
    """Spec §3.1 observability follow-up. If less than 50% of games have
    >=15 own-entries on either side, the report surfaces a WARNING line
    so a zero no-progress rate isn't confused with a no-data state."""
    # Diagnostics are short (5 entries / game) — won't reach 15 own-entries.
    short_diag = [
        {"ply": p, "side_to_move": "red" if i % 2 == 0 else "black",
         "selected_move_classification": {"primary_class": "redundant_reinforcement"}}
        for i, p in enumerate(range(50, 55))
    ]
    games = [_per_game(220, reason="win", n_moves=80, winner="red", diagnostics=short_diag)
             for _ in range(10)]
    agg = aggregate_marathon_termination(games, **_cfg())
    lines = format_marathon_termination_report(agg, range_label="220-229")
    text = "\n".join(lines)
    assert "WARNING" in text
    assert "observability" in text


def test_csv_includes_observability_columns(tmp_path):
    """Observability counters present per row + range-total row."""
    games = [_per_game(220, reason="win", n_moves=80, winner="red")]
    agg = aggregate_marathon_termination(games, **_cfg())
    out = tmp_path / "m.csv"
    write_marathon_termination_csv(str(out), agg)
    rows = list(csv.DictReader(open(out)))
    for r in rows:
        assert "diagnostics_entries_red" in r
        assert "diagnostics_entries_black" in r
        assert "no_progress_observable_games_red" in r
        assert "no_progress_observable_games_black" in r
```

- [ ] **Step 8: Run analyzer tests + full marathon-termination suite**

Run: `.venv/bin/python -m pytest tests/test_marathon_termination_diagnostics.py tests/test_analyzer_marathon_termination.py -v`

Expected: 28 passed (23 module + 5 analyzer-integration).

- [ ] **Step 9: Commit**

```bash
git add scripts/GPU/alphazero/marathon_termination_diagnostics.py \
        scripts/twixt_replay_analyzer.py \
        tests/test_marathon_termination_diagnostics.py \
        tests/test_analyzer_marathon_termination.py
git commit -m "$(cat <<'EOF'
feat(analyzer): marathon-termination aggregator + CSV + report (Spec §4-5)

Aggregates per-game diagnostics into per-iter rows + range-total row.
New CSV: marathon_termination_by_iter_<range>.csv with the spec §4.2
column set (six adjudication gate columns matching §3.2 taxonomy, both
resign rates per game-length partition).

New report section "Marathon termination diagnostics" with a
"Suggested termination action:" line that applies the spec §5 decision
rule to the range-total row: at most one knob suggestion per run, with
explicit reminders that early state-cap is default-off and requires
explicit user approval.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Smoke on existing 190-219 + 220-229 data

**Files:** none (verification-only).

The diagnostic is analyzer-only and works on existing per-game JSON files. Tasks 2-6 ran TDD on fixtures; Task 7 confirms the production code path works end-to-end.

- [ ] **Step 1: Confirm all four ranges have staging directories**

Run:
```bash
for r in 190-199 200-209 210-219 220-229; do
  [ -d "Replays/$r" ] && echo "$r: present ($(ls Replays/$r | wc -l) files)" || echo "$r: MISSING — re-stage needed"
done
```

If any range is missing, re-stage with the pattern from prior sessions:
```bash
for range in "189 19" "199 20" "209 21" "219 22"; do
  read prev_iter range_prefix <<< "$range"
  rstart=$(( range_prefix * 10 ))
  rend=$(( rstart + 9 ))
  rdir="Replays/${rstart}-${rend}"
  mkdir -p "$rdir"
  for f in scripts/GPU/logs/games/iter_0$(printf '%03d' $prev_iter)_game_*.json \
           scripts/GPU/logs/games/iter_0${range_prefix}?_game_*.json; do
    [ -f "$f" ] && ln -sf "../../$f" "$rdir/$(basename $f)" 2>/dev/null
  done
  for f in scripts/GPU/logs/games/iter_0$(printf '%03d' $prev_iter)_stats.json \
           scripts/GPU/logs/games/iter_0${range_prefix}?_stats.json; do
    [ -f "$f" ] && ln -sf "../../$f" "$rdir/$(basename $f)" 2>/dev/null
  done
done
```

- [ ] **Step 2: Re-run the analyzer on all four ranges**

Run:
```bash
for r in 190-199 200-209 210-219 220-229; do
  echo "=== $r ==="
  .venv/bin/python ./scripts/twixt_replay_analyzer.py --input Replays/$r --out Replays/${r}_Replay 2>&1 | tail -1
done
```

Expected: each completes with `[OK] heatmaps saved in: ...` and no error.

- [ ] **Step 3: Inspect the marathon-termination report sections**

Run:
```bash
for r in 190-199 200-209 210-219 220-229; do
  echo "=== $r ==="
  awk '/Marathon termination diagnostics/,0' Replays/${r}_Replay/report_${r}.txt | head -30
  echo
done
```

Confirm each range emits:
- the section header with the range label
- a "state_cap 280-ply games: N" line
- a non-empty adjudication-gate distribution
- a "Suggested termination action:" line

- [ ] **Step 4: Inspect the new CSV's range-total row**

Run:
```bash
for r in 190-199 200-209 210-219 220-229; do
  echo "=== $r range-total ==="
  awk -F, 'NR==1 {for (i=1;i<=NF;i++) h[i]=$i; next} $1=="-1" {for (i=1;i<=NF;i++) print "  " h[i] " = " $i}' \
    Replays/${r}_Replay/marathon_termination_by_iter_${r}.csv
  echo
done
```

Expected: a clean key=value listing per range with non-zero values in the relevant gate columns.

- [ ] **Step 5: No commit (verification-only).**

---

## Task 8: Apply the §5 decision rule to the diagnostic output

**Files:** none (decision + memory update).

- [ ] **Step 1: Examine the "Suggested termination action" line from each range**

For each of 190-219 (3 ranges) and 220-229, note the suggestion from §5. Tabulate:

```
range      | state_cap | dominant gate          | resign long/short | npw mean | suggestion
190-199    |        N  | ...                    | ...               | ...      | ...
200-209    |        N  | ...                    | ...               | ...      | ...
210-219    |        N  | ...                    | ...               | ...      | ...
220-229    |        N  | ...                    | ...               | ...      | ...
```

- [ ] **Step 2: If a single termination knob is dominantly suggested**

Update memory (`spec4_recovery_retargeting_diagnostic.md` or a new memory entry) with:
- which knob the diagnostic recommends
- the supporting numbers from the report
- the value-uncertain-guard wording from §5.1 (any treatment must enforce it at the termination call-site)
- explicit deferral: actual training-knob change is gated on user approval

- [ ] **Step 3: If no single knob dominates**

Update memory with:
- the diagnostic confirms the marathon shape is heterogeneous
- a small set (5-10) of representative 280-ply games to hand-review
- defer the next training-knob change until after hand-review

- [ ] **Step 4: No code commit. Memory commit only if memory was updated.**

```bash
# Memory lives outside the project repo (~/.claude/projects/...).
# No git action needed.
```

---

## Self-Review

**Spec coverage** — every spec section has a task:
- §1 (goal/scope) → covered by overall plan
- §2 (rollback) → Task 1
- §3.1 (no-progress window) → Task 2
- §3.2 (adjudication coverage) → Task 3 (+ Task 0 pre-check resolved outcome A)
- §3.3 (resign-gate breakdown) → Task 4
- §3.4 (stagnation rate per-iter) → Task 6 (`mean_no_progress_windows_per_game` per-iter row)
- §4.1 (per-game record extension) → SKIPPED: spec §4.1 contemplates adding fields to `goal_completion_record`, but the aggregator reads from the existing per-game JSON's record + meta + diagnostics without mutating them, which preserves the existing record schema. If §4.1 fields are strictly required, that's a separate task; the spec's intent (downstream tooling can access these values) is met via the CSV + summary JSON in §4.2/§4.3.
- §4.2 (new CSV) → Task 6
- §4.3 (report section) → Task 6
- §5 (decision rule) → Task 6 (`_marathon_termination_decision`) + Task 8 (application)
- §5.1 (value-uncertain guard) → Task 5
- §6 (implementation order) → Tasks 0-8 follow it
- §7 (tests 1-22) → Tasks 2-6 (each test from §7 has a corresponding step)

**Type/name consistency:**
- `ADJUDICATION_GATE_BUCKETS` / `GAME_LENGTH_BUCKETS` / `NO_PROGRESS_WINDOW_SIZE` — consistent module-level exports across Tasks 2-6.
- `detect_no_progress_windows(diagnostics, *, side)` → consistent in Tasks 2 + 6.
- `classify_adjudication_coverage(record, meta, diagnostics) -> Optional[str]` → consistent in Tasks 3 + 6.
- `compute_resign_gate_breakdown(record, diagnostics, *, losing_side, **resign_cfg)` → consistent in Tasks 4 + 6.
- `value_uncertain_guard(diagnostics, *, ...)` → Task 5; not consumed by the analyzer in this plan (intended for future termination call-sites).
- `aggregate_marathon_termination(games, *, **resign_cfg)` → Task 6.
- `write_marathon_termination_csv(out_path, agg)` / `format_marathon_termination_report(agg, range_label)` → Task 6.

**Placeholder scan:** None. Every step has either complete code or an exact verification command.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-19-marathon-termination-tuning.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best for this plan because Task 0 is verification-only (already resolved), and Tasks 2-6 are well-isolated TDD units. Subagent isolation prevents cross-task context bleed.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Faster turnaround if you want to interject between tasks.

**Which approach?**
