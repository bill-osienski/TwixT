# Strong-Advantage Diversity Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the simple `admitted = admitted[: args.max_probes]` slice in the `strong_advantage` probe generator with a post-Phase-2 diversity-aware selector that prevents same-game clustering, fixes the audit's `reason="admitted"` double-counting, and regenerates the committed suite under the new selector.

**Architecture:** Three-rule round-robin selector (near-duplicate → ply-too-close → per-game cap) walking a fixed canonical 4-tuple of categories. All logic lives in `scripts/build_probe_suite.py`; one supporting line removal in `scripts/GPU/alphazero/probe_eval.py` makes the audit's admitted-count honest. Phase 2 stops writing `admitted` audit rows; the selector becomes the single writer of `reason="admitted"` and `reason="diversity_*"` rows.

**Tech Stack:** Python 3.14, pytest, existing `scripts/build_probe_suite.py` and `scripts/GPU/alphazero/probe_eval.py` infrastructure.

**Spec:** [`docs/superpowers/specs/2026-04-28-strong-advantage-diversity-selector-design.md`](../specs/2026-04-28-strong-advantage-diversity-selector-design.md).

**Files touched:**
- Modify: `scripts/GPU/alphazero/probe_eval.py:855` (one-line removal)
- Modify: `scripts/build_probe_suite.py` (~100 LOC: constants, helpers, selector, CLI flag, call-site, meta enrichment)
- Create: `tests/test_strong_advantage_diversity_selector.py` (~280 LOC, 14 tests)
- Modify: `docs/probe-suite-generation.md` (~30 LOC: new flag, audit reasons, regeneration note)
- Regenerate (operator step): `tests/probes/strong_advantage_probes.json` and `tests/probes/candidates_strong_advantage.json`

**Conventions:**
- Tests use the existing `tests/test_strong_advantage_probe_suite.py` style: no class wrappers, function-level imports of the modules under test, `from __future__ import annotations` at the top.
- Commit messages follow the repo's `prefix(scope): summary` style (see `git log --oneline -10`).
- Each task is one focused commit. If pre-commit ESLint warnings appear, they are pre-existing in `node_modules` and unrelated; the commit proceeds.

---

## Task 1 — Audit cleanup: remove Phase-1 `admitted` row

The Phase-1 extractor currently writes `audit.append({**base, "reason": "admitted"})` for every Phase-1 survivor. Phase 2 also writes an audit row per labeled candidate (with `reason="admitted"` if it passes), so each Phase-2-admitted candidate appears twice in the audit. Removing the Phase-1 row is a precondition for the selector's audit semantics ("`reason="admitted"` ⇔ probe is in the final suite, exactly once").

**Files:**
- Modify: `scripts/GPU/alphazero/probe_eval.py:855`
- Test: `tests/test_strong_advantage_probe_suite.py` (extend; don't create a new file — this is testing the existing extractor)

- [ ] **Step 1: Write the failing test**

Add this test at the bottom of `tests/test_strong_advantage_probe_suite.py`:

```python
def test_extract_strong_advantage_writes_no_admitted_audit_rows_in_phase1():
    """Phase 1 must NOT write admitted audit rows. The Phase-2 audit row
    (written by build_probe_suite.py's _run_strong_advantage loop, then
    superseded by the diversity selector) is the single canonical
    post-labeling record. See spec §7.1.

    This test loads a real committed game file and runs the extractor on
    it. The strong assertion: no audit row carries reason="admitted",
    regardless of how many Phase-1 candidates the game produces. This
    test passes trivially if the game produces zero candidates, but the
    end-to-end integration test in tests/test_strong_advantage_diversity_selector.py
    (added in Task 8) provides the load-bearing check that admitted rows
    appear exactly once per kept probe in the final selector output.
    """
    import json
    from pathlib import Path

    from scripts.GPU.alphazero.probe_eval import extract_strong_advantage_candidates

    project_root = Path(__file__).resolve().parent.parent
    games_dir = project_root / "scripts" / "GPU" / "logs" / "games"

    # Pick any one decisive game with iteration in the committed range.
    games = []
    for fp in sorted(games_dir.glob("iter_0057_game_*.json"))[:5]:
        with open(fp) as f:
            try:
                g = json.load(f)
            except json.JSONDecodeError:
                continue
        meta = g.get("meta") or {}
        if (meta.get("reason") or g.get("winner_reason")) == "win":
            g["source_game"] = fp.stem
            games.append(g)
        if len(games) >= 1:
            break

    assert games, "no decisive game files found in iter_0057 range — fixture missing"

    candidates, audit = extract_strong_advantage_candidates(games)

    admitted_audit_rows = [r for r in audit if r["reason"] == "admitted"]
    assert admitted_audit_rows == [], (
        f"Phase 1 wrote {len(admitted_audit_rows)} admitted audit row(s); "
        f"after the cleanup, Phase 1 should write only rejection rows. "
        f"Sample: {admitted_audit_rows[:2]}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_strong_advantage_probe_suite.py::test_extract_strong_advantage_writes_no_admitted_audit_rows_in_phase1 -v
```

Expected: FAIL with `AssertionError: Phase 1 wrote N admitted audit row(s); ...`

- [ ] **Step 3: Implement the change**

In `scripts/GPU/alphazero/probe_eval.py`, find this block in `extract_strong_advantage_candidates` (around line 844-855):

```python
            cand = {
                "move_history": [(m["row"], m["col"]) for m in moves_list[:target_ply]],
                "ply": target_ply,
                "winner": winner,
                "category": category,
                "phase1_features": feats,
                "source_game": source_game,
                "source_ply": target_ply,
                "starting_player": starting_player,
            }
            candidates.append(cand)
            audit.append({**base_audit, "reason": "admitted"})
```

Remove the last line (`audit.append({**base_audit, "reason": "admitted"})`). The block becomes:

```python
            cand = {
                "move_history": [(m["row"], m["col"]) for m in moves_list[:target_ply]],
                "ply": target_ply,
                "winner": winner,
                "category": category,
                "phase1_features": feats,
                "source_game": source_game,
                "source_ply": target_ply,
                "starting_player": starting_player,
            }
            candidates.append(cand)
```

- [ ] **Step 4: Run the new test (expect PASS) and the full extractor test file (expect no regressions)**

```
pytest tests/test_strong_advantage_probe_suite.py -v
```

Expected: all tests PASS, including the new one. If any pre-existing test breaks because it asserted on Phase-1 admitted audit rows, fix that test by removing the assertion (the spec deliberately changes this contract).

- [ ] **Step 5: Commit**

```
git add scripts/GPU/alphazero/probe_eval.py tests/test_strong_advantage_probe_suite.py
git commit -m "$(cat <<'EOF'
fix(probes): remove Phase-1 admitted audit double-counting

Phase 1's audit.append for surviving candidates duplicated the Phase-2
admitted row, making reason="admitted" counts unreliable (2x for
Phase-2 admits, 1x for Phase-2 rejects). Phase-2 audit row is the
single canonical post-labeling record. Spec §7.1.
EOF
)"
```

---

## Task 2 — Add module constants and quality sort key

Adds the building blocks the selector needs: the per-game ply-separation constant, the canonical category iteration order, and the Stage-2 sort key. Pure functions, fully unit-testable.

**Files:**
- Modify: `scripts/build_probe_suite.py` (add at module top, after imports)
- Create: `tests/test_strong_advantage_diversity_selector.py`

- [ ] **Step 1: Create the test file with the failing test**

Create `tests/test_strong_advantage_diversity_selector.py`:

```python
"""Tests for the diversity-aware selector in scripts/build_probe_suite.py.

The selector replaces the simple `admitted = admitted[: args.max_probes]`
slice with category round-robin + near-duplicate / ply-separation /
per-game cap rules. See spec
docs/superpowers/specs/2026-04-28-strong-advantage-diversity-selector-design.md.

Each test constructs a synthetic admitted list and asserts on the
selector's output and audit deltas. No live MCTS labeling.
"""
from __future__ import annotations

from copy import deepcopy

import pytest


def _make_candidate(
    *,
    source_game: str,
    source_ply: int,
    category: str,
    cc_size: int = 18,
    axis_span_margin: float = 0.30,
    cc_axis_span: float = 0.74,
    min_top1_share: float = 0.25,
    value_stability: float = 0.05,
    mean_root_value: float = 0.62,
):
    """Construct a synthetic admitted candidate matching the fields the
    selector reads. Defaults pass every Phase-1 and Phase-2 gate so the
    only thing exercised is selection logic."""
    return {
        "source_game": source_game,
        "source_ply": source_ply,
        "category": category,
        "winner": "red" if category.endswith("_red") else "black",
        "ply": source_ply,
        "starting_player": "red",
        "move_history": [],  # selector doesn't touch this
        "phase1_features": {
            "cc_size": cc_size,
            "cc_axis_span": cc_axis_span,
            "cc_touches_own_goal": True,
            "axis_span_margin": axis_span_margin,
            "centroid_chebyshev_from_center": 4 if "central" in category else 10,
            "forced_within_2": False,
        },
        "phase2_label": {
            "mean_root_value": mean_root_value,
            "value_per_run": [mean_root_value, mean_root_value],
            "value_stability": value_stability,
            "min_top1_share": min_top1_share,
            "label_mcts_sims": 2000,
            "label_mcts_repeats": 2,
            "rng_seed_base": 1,
            "label_checkpoint": "test_ckpt.safetensors",
        },
    }


def test_diversity_sort_key_orders_by_cc_size_desc_first():
    """Stage-2 sort: larger cc_size sorts before smaller, all else equal."""
    from scripts.build_probe_suite import _diversity_sort_key

    a = _make_candidate(source_game="iter_0001_game_001", source_ply=10,
                        category="chain_advantage_central_red", cc_size=20)
    b = _make_candidate(source_game="iter_0001_game_001", source_ply=10,
                        category="chain_advantage_central_red", cc_size=15)

    # Lower sort-key tuple sorts first, so larger cc_size (negated) wins.
    assert _diversity_sort_key(a) < _diversity_sort_key(b)
```

- [ ] **Step 2: Run the test to verify it fails**

```
pytest tests/test_strong_advantage_diversity_selector.py::test_diversity_sort_key_orders_by_cc_size_desc_first -v
```

Expected: FAIL with `ImportError` or `AttributeError` — `_diversity_sort_key` does not exist yet.

- [ ] **Step 3: Add module constants and the sort-key function**

In `scripts/build_probe_suite.py`, find the imports block (lines 16-22) and add right after it (so it sits at module scope, before `main()`):

```python
# --- Diversity selector constants and helpers ---

MIN_PLY_SEPARATION_SAME_GAME = 3
"""Same-game probes must be at least this many plies apart. Tied to the
current K-range [3, 8]: with span 5, separation 3 admits at most 2 plies
per game, matching the default --max-probes-per-game cap."""

CATEGORY_ITERATION_ORDER = (
    "chain_advantage_central_red",
    "chain_advantage_central_black",
    "chain_advantage_edge_red",
    "chain_advantage_edge_black",
)
"""Fixed canonical order for round-robin category fill. Empty buckets
are skipped at iteration time. See spec §5.4."""


def _diversity_sort_key(cand: dict) -> tuple:
    """Stage-2 rank key: structural-first, Phase-2 secondary, source order
    as final determinism guarantee. Lower tuple sorts first. See spec §4.2."""
    p1 = cand["phase1_features"]
    p2 = cand["phase2_label"]
    try:
        iter_num = int(cand["source_game"].split("_")[1])
    except (IndexError, ValueError):
        iter_num = 0
    return (
        -p1["cc_size"],
        -p1["axis_span_margin"],
        -p1["cc_axis_span"],
        -p2["min_top1_share"],
        p2["value_stability"],
        -iter_num,
        -cand["source_ply"],
        cand["source_game"],
    )
```

- [ ] **Step 4: Run the test (expect PASS)**

```
pytest tests/test_strong_advantage_diversity_selector.py::test_diversity_sort_key_orders_by_cc_size_desc_first -v
```

Expected: PASS.

- [ ] **Step 5: Add three more sort-key tests covering the full ordering chain**

Append to `tests/test_strong_advantage_diversity_selector.py`:

```python
def test_diversity_sort_key_axis_span_margin_breaks_cc_size_tie():
    """When cc_size matches, larger axis_span_margin wins."""
    from scripts.build_probe_suite import _diversity_sort_key

    a = _make_candidate(source_game="iter_0001_game_001", source_ply=10,
                        category="chain_advantage_central_red",
                        cc_size=20, axis_span_margin=0.40)
    b = _make_candidate(source_game="iter_0001_game_001", source_ply=10,
                        category="chain_advantage_central_red",
                        cc_size=20, axis_span_margin=0.20)

    assert _diversity_sort_key(a) < _diversity_sort_key(b)


def test_diversity_sort_key_min_top1_share_breaks_structural_ties():
    """When all structural fields tie, higher min_top1_share wins."""
    from scripts.build_probe_suite import _diversity_sort_key

    a = _make_candidate(source_game="iter_0001_game_001", source_ply=10,
                        category="chain_advantage_central_red",
                        min_top1_share=0.40)
    b = _make_candidate(source_game="iter_0001_game_001", source_ply=10,
                        category="chain_advantage_central_red",
                        min_top1_share=0.20)

    assert _diversity_sort_key(a) < _diversity_sort_key(b)


def test_diversity_sort_key_total_order_via_source_tiebreak():
    """Every field equal except source — final _sort_key tiebreak applies."""
    from scripts.build_probe_suite import _diversity_sort_key

    a = _make_candidate(source_game="iter_0099_game_001", source_ply=50,
                        category="chain_advantage_central_red")
    b = _make_candidate(source_game="iter_0050_game_001", source_ply=50,
                        category="chain_advantage_central_red")

    # Higher iter (-iter is smaller) wins → a sorts before b.
    assert _diversity_sort_key(a) < _diversity_sort_key(b)
```

- [ ] **Step 6: Run the four sort-key tests (expect PASS)**

```
pytest tests/test_strong_advantage_diversity_selector.py -v
```

Expected: 4 passed.

- [ ] **Step 7: Commit**

```
git add scripts/build_probe_suite.py tests/test_strong_advantage_diversity_selector.py
git commit -m "$(cat <<'EOF'
feat(probes): add diversity selector module constants and sort key

Adds MIN_PLY_SEPARATION_SAME_GAME, CATEGORY_ITERATION_ORDER, and
_diversity_sort_key (structural-first Stage-2 rank). Building blocks
for the diversity selector. Spec §4.2.
EOF
)"
```

---

## Task 3 — Rule A helper: near-duplicate detection

Implements `_find_near_duplicate_keeper(cand, kept)`. Returns the matching keeper or None. Same source_game AND same category AND `|Δcc_size| < 2 AND |Δaxis_span_margin| < 0.05`. Multiple matches: smallest source_ply (deterministic).

**Files:**
- Modify: `scripts/build_probe_suite.py` (add helper after `_diversity_sort_key`)
- Modify: `tests/test_strong_advantage_diversity_selector.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_strong_advantage_diversity_selector.py`:

```python
def test_rule_a_near_duplicate_matches_same_game_same_category_close_features():
    from scripts.build_probe_suite import _find_near_duplicate_keeper

    keeper = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                             category="chain_advantage_central_red",
                             cc_size=20, axis_span_margin=0.30)
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=51,
                           category="chain_advantage_central_red",
                           cc_size=21, axis_span_margin=0.31)  # Δcc=1, Δasm=0.01

    assert _find_near_duplicate_keeper(cand, [keeper]) is keeper


def test_rule_a_near_duplicate_skips_different_game():
    from scripts.build_probe_suite import _find_near_duplicate_keeper

    keeper = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                             category="chain_advantage_central_red",
                             cc_size=20, axis_span_margin=0.30)
    cand = _make_candidate(source_game="iter_0058_game_999", source_ply=51,
                           category="chain_advantage_central_red",
                           cc_size=21, axis_span_margin=0.31)

    assert _find_near_duplicate_keeper(cand, [keeper]) is None


def test_rule_a_near_duplicate_skips_different_category():
    """Cross-category same-game pair is NOT a near-duplicate, even when
    structural deltas are below thresholds. Spec §5.6."""
    from scripts.build_probe_suite import _find_near_duplicate_keeper

    keeper = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                             category="chain_advantage_central_red",
                             cc_size=20, axis_span_margin=0.30)
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=51,
                           category="chain_advantage_edge_red",
                           cc_size=21, axis_span_margin=0.31)

    assert _find_near_duplicate_keeper(cand, [keeper]) is None


def test_rule_a_near_duplicate_skips_when_cc_size_delta_at_threshold():
    """|Δcc_size| < 2 is strict: delta == 2 is NOT a duplicate."""
    from scripts.build_probe_suite import _find_near_duplicate_keeper

    keeper = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                             category="chain_advantage_central_red",
                             cc_size=20, axis_span_margin=0.30)
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=51,
                           category="chain_advantage_central_red",
                           cc_size=22, axis_span_margin=0.30)  # Δcc=2

    assert _find_near_duplicate_keeper(cand, [keeper]) is None


def test_rule_a_near_duplicate_returns_smallest_source_ply_when_multiple_match():
    """Tie-break: when multiple kept candidates match, return the one
    with the smallest source_ply."""
    from scripts.build_probe_suite import _find_near_duplicate_keeper

    keeper_low = _make_candidate(source_game="iter_0058_game_040", source_ply=48,
                                 category="chain_advantage_central_red",
                                 cc_size=20, axis_span_margin=0.30)
    keeper_high = _make_candidate(source_game="iter_0058_game_040", source_ply=52,
                                  category="chain_advantage_central_red",
                                  cc_size=21, axis_span_margin=0.31)
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                           category="chain_advantage_central_red",
                           cc_size=20, axis_span_margin=0.30)

    # Both keepers match cand; smallest source_ply wins.
    assert _find_near_duplicate_keeper(cand, [keeper_high, keeper_low]) is keeper_low
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_strong_advantage_diversity_selector.py -v -k rule_a
```

Expected: 5 FAILed with `ImportError` (the helper doesn't exist).

- [ ] **Step 3: Implement the helper**

In `scripts/build_probe_suite.py`, after `_diversity_sort_key`:

```python
def _find_near_duplicate_keeper(cand: dict, kept: list) -> dict | None:
    """Rule A — Near-duplicate. Returns the matching kept candidate or None.

    Same source_game AND same category AND |Δcc_size| < 2 AND
    |Δaxis_span_margin| < 0.05. Multiple matches: smallest source_ply
    (deterministic). See spec §4.2.
    """
    cand_p1 = cand["phase1_features"]
    matches = [
        k for k in kept
        if k["source_game"] == cand["source_game"]
        and k["category"] == cand["category"]
        and abs(k["phase1_features"]["cc_size"] - cand_p1["cc_size"]) < 2
        and abs(k["phase1_features"]["axis_span_margin"] - cand_p1["axis_span_margin"]) < 0.05
    ]
    if not matches:
        return None
    return min(matches, key=lambda k: k["source_ply"])
```

- [ ] **Step 4: Run the tests (expect PASS)**

```
pytest tests/test_strong_advantage_diversity_selector.py -v -k rule_a
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```
git add scripts/build_probe_suite.py tests/test_strong_advantage_diversity_selector.py
git commit -m "$(cat <<'EOF'
feat(probes): add Rule A (near-duplicate) helper for diversity selector

Same source_game AND same category AND |Δcc_size| < 2 AND
|Δaxis_span_margin| < 0.05. Spec §4.2 / §5.6.
EOF
)"
```

---

## Task 4 — Rule B helper: ply-too-close detection with tiered tie-break

Implements `_find_ply_too_close_keeper(cand, kept, rank_index)`. Returns the blocking keeper or None. Same source_game AND `|Δsource_ply| < MIN_PLY_SEPARATION_SAME_GAME`. Tie-break: closest sibling → better Stage-2 rank → smallest source_ply.

The `rank_index` parameter maps `id(cand)` to its position in its category's Stage-2 sort. The selector builds it once after Stage 2 and passes it to this helper. Identity-based keys (`id()`) are used because candidate dicts are mutable and can't be hashed.

**Files:**
- Modify: `scripts/build_probe_suite.py`
- Modify: `tests/test_strong_advantage_diversity_selector.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_strong_advantage_diversity_selector.py`:

```python
def test_rule_b_ply_too_close_matches_same_game_within_separation():
    from scripts.build_probe_suite import _find_ply_too_close_keeper

    keeper = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                             category="chain_advantage_central_red")
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=51,
                           category="chain_advantage_central_red")
    rank_index = {id(keeper): 0, id(cand): 1}

    assert _find_ply_too_close_keeper(cand, [keeper], rank_index) is keeper


def test_rule_b_ply_too_close_admits_at_separation_boundary():
    """|Δsource_ply| < MIN_PLY_SEPARATION_SAME_GAME=3 is strict: Δ=3
    is admissible, not too close."""
    from scripts.build_probe_suite import _find_ply_too_close_keeper

    keeper = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                             category="chain_advantage_central_red")
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=53,  # Δ=3
                           category="chain_advantage_central_red")
    rank_index = {id(keeper): 0, id(cand): 1}

    assert _find_ply_too_close_keeper(cand, [keeper], rank_index) is None


def test_rule_b_ply_too_close_skips_different_game():
    from scripts.build_probe_suite import _find_ply_too_close_keeper

    keeper = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                             category="chain_advantage_central_red")
    cand = _make_candidate(source_game="iter_0058_game_999", source_ply=51,
                           category="chain_advantage_central_red")
    rank_index = {id(keeper): 0, id(cand): 1}

    assert _find_ply_too_close_keeper(cand, [keeper], rank_index) is None


def test_rule_b_ply_too_close_ignores_category_only_game_matters():
    """Rule B is category-agnostic: same-game cross-category pair within
    separation still triggers Rule B."""
    from scripts.build_probe_suite import _find_ply_too_close_keeper

    keeper = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                             category="chain_advantage_central_red")
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=51,
                           category="chain_advantage_edge_red")
    rank_index = {id(keeper): 0, id(cand): 1}

    assert _find_ply_too_close_keeper(cand, [keeper], rank_index) is keeper


def test_rule_b_tie_break_prefers_closest_keeper():
    """Two keepers, candidate at ply 51: keeper at 50 (Δ=1) wins over
    keeper at 49 (Δ=2)."""
    from scripts.build_probe_suite import _find_ply_too_close_keeper

    closest = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                              category="chain_advantage_central_red")
    farther = _make_candidate(source_game="iter_0058_game_040", source_ply=49,
                              category="chain_advantage_central_red")
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=51,
                           category="chain_advantage_central_red")
    rank_index = {id(closest): 0, id(farther): 1, id(cand): 2}

    assert _find_ply_too_close_keeper(cand, [farther, closest], rank_index) is closest


def test_rule_b_tie_break_uses_better_rank_when_equidistant():
    """Two keepers equidistant from candidate (both Δ=1): the one with
    better Stage-2 rank (lower rank_index value) wins."""
    from scripts.build_probe_suite import _find_ply_too_close_keeper

    higher_rank = _make_candidate(source_game="iter_0058_game_040", source_ply=49,
                                  category="chain_advantage_central_red")
    lower_rank = _make_candidate(source_game="iter_0058_game_040", source_ply=51,
                                 category="chain_advantage_central_red")
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                           category="chain_advantage_central_red")
    # higher_rank has rank 0 (better); lower_rank has rank 5 (worse).
    rank_index = {id(higher_rank): 0, id(lower_rank): 5, id(cand): 99}

    assert _find_ply_too_close_keeper(cand, [lower_rank, higher_rank], rank_index) is higher_rank


def test_rule_b_tie_break_falls_back_to_smallest_source_ply():
    """When equidistant AND same rank_index value (synthetic edge case
    only achievable via test setup), smallest source_ply wins."""
    from scripts.build_probe_suite import _find_ply_too_close_keeper

    later_ply = _make_candidate(source_game="iter_0058_game_040", source_ply=51,
                                category="chain_advantage_central_red")
    earlier_ply = _make_candidate(source_game="iter_0058_game_040", source_ply=49,
                                  category="chain_advantage_central_red")
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                           category="chain_advantage_central_red")
    # Force same rank_index for both keepers to exercise the final tie-break.
    rank_index = {id(later_ply): 0, id(earlier_ply): 0, id(cand): 99}

    assert _find_ply_too_close_keeper(cand, [later_ply, earlier_ply], rank_index) is earlier_ply
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_strong_advantage_diversity_selector.py -v -k rule_b
```

Expected: 7 FAILed with `ImportError`.

- [ ] **Step 3: Implement the helper**

In `scripts/build_probe_suite.py`, after `_find_near_duplicate_keeper`:

```python
def _find_ply_too_close_keeper(cand: dict, kept: list, rank_index: dict) -> dict | None:
    """Rule B — Ply-too-close. Returns the blocking kept candidate or None.

    Same source_game AND |Δsource_ply| < MIN_PLY_SEPARATION_SAME_GAME (any
    category — Rule B is category-agnostic).

    Tiered tie-break:
      1. Closest kept sibling (smallest |Δsource_ply|).
      2. Better Stage-2 rank (smaller rank_index value).
      3. Smallest source_ply.

    rank_index: maps id(cand) to its position in its category's Stage-2
    sort order. The selector builds this once after Stage 2.
    See spec §4.2.
    """
    matches = [
        k for k in kept
        if k["source_game"] == cand["source_game"]
        and abs(k["source_ply"] - cand["source_ply"]) < MIN_PLY_SEPARATION_SAME_GAME
    ]
    if not matches:
        return None
    return min(
        matches,
        key=lambda k: (
            abs(k["source_ply"] - cand["source_ply"]),
            rank_index[id(k)],
            k["source_ply"],
        ),
    )
```

- [ ] **Step 4: Run the tests (expect PASS)**

```
pytest tests/test_strong_advantage_diversity_selector.py -v -k rule_b
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```
git add scripts/build_probe_suite.py tests/test_strong_advantage_diversity_selector.py
git commit -m "$(cat <<'EOF'
feat(probes): add Rule B (ply-too-close) helper with tiered tie-break

Same source_game AND |Δsource_ply| < MIN_PLY_SEPARATION_SAME_GAME (any
category). Tie-break: closest, then better Stage-2 rank, then smallest
source_ply — points the audit at the actually-blocking keeper. Spec §4.2.
EOF
)"
```

---

## Task 5 — Rule C helper: per-game cap detection

Implements `_find_per_game_cap_keeper(cand, kept, cap)`. Returns the smallest-source_ply keeper from the same game (deterministic) when the cap is exceeded, else None. Counted total across all categories.

**Files:**
- Modify: `scripts/build_probe_suite.py`
- Modify: `tests/test_strong_advantage_diversity_selector.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_strong_advantage_diversity_selector.py`:

```python
def test_rule_c_per_game_cap_returns_none_when_under_cap():
    from scripts.build_probe_suite import _find_per_game_cap_keeper

    keeper = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                             category="chain_advantage_central_red")
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=53,
                           category="chain_advantage_central_red")

    # 1 keeper, cap=2 → not exceeded.
    assert _find_per_game_cap_keeper(cand, [keeper], cap=2) is None


def test_rule_c_per_game_cap_fires_when_at_cap():
    from scripts.build_probe_suite import _find_per_game_cap_keeper

    keeper_a = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                               category="chain_advantage_central_red")
    keeper_b = _make_candidate(source_game="iter_0058_game_040", source_ply=53,
                               category="chain_advantage_central_red")
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=56,
                           category="chain_advantage_central_red")

    # 2 keepers, cap=2 → exceeded for the next candidate.
    keeper = _find_per_game_cap_keeper(cand, [keeper_a, keeper_b], cap=2)
    assert keeper is keeper_a  # smallest source_ply


def test_rule_c_per_game_cap_counts_across_categories():
    """Cap is total per game, not per (game, category). One central +
    one edge from the same game already fills cap=2."""
    from scripts.build_probe_suite import _find_per_game_cap_keeper

    central = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                              category="chain_advantage_central_red")
    edge = _make_candidate(source_game="iter_0058_game_040", source_ply=53,
                           category="chain_advantage_edge_red")
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=56,
                           category="chain_advantage_central_black")

    keeper = _find_per_game_cap_keeper(cand, [central, edge], cap=2)
    assert keeper is central


def test_rule_c_per_game_cap_ignores_other_games():
    from scripts.build_probe_suite import _find_per_game_cap_keeper

    keeper_other = _make_candidate(source_game="iter_0058_game_999", source_ply=50,
                                   category="chain_advantage_central_red")
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=53,
                           category="chain_advantage_central_red")

    assert _find_per_game_cap_keeper(cand, [keeper_other], cap=1) is None
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_strong_advantage_diversity_selector.py -v -k rule_c
```

Expected: 4 FAILed with `ImportError`.

- [ ] **Step 3: Implement the helper**

In `scripts/build_probe_suite.py`, after `_find_ply_too_close_keeper`:

```python
def _find_per_game_cap_keeper(cand: dict, kept: list, cap: int) -> dict | None:
    """Rule C — Per-game cap. Returns the smallest-source_ply keeper from
    the same game when the cap is exceeded, else None. Counted total
    across all categories. See spec §4.2 / §5.2.
    """
    from_game = [k for k in kept if k["source_game"] == cand["source_game"]]
    if len(from_game) < cap:
        return None
    return min(from_game, key=lambda k: k["source_ply"])
```

- [ ] **Step 4: Run the tests (expect PASS)**

```
pytest tests/test_strong_advantage_diversity_selector.py -v -k rule_c
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add scripts/build_probe_suite.py tests/test_strong_advantage_diversity_selector.py
git commit -m "$(cat <<'EOF'
feat(probes): add Rule C (per-game cap) helper for diversity selector

Counts kept candidates per source_game total across all 4 categories.
Returns smallest-source_ply keeper when cap exceeded. Spec §4.2 / §5.2.
EOF
)"
```

---

## Task 6 — Round-robin orchestrator: `_select_diverse_admitted_candidates`

Wires the three rules into a category round-robin walk. Mutates the audit list in-place (appends one row per drop AND one row per admit), returns the kept list.

**Files:**
- Modify: `scripts/build_probe_suite.py`
- Modify: `tests/test_strong_advantage_diversity_selector.py`

- [ ] **Step 1: Write the failing test (basic behavior)**

Append to `tests/test_strong_advantage_diversity_selector.py`:

```python
def test_selector_returns_all_when_under_max_probes_and_no_rules_fire():
    """Three structurally distinct candidates from three different games
    in three different categories. cap=2, max_probes=10. All three kept;
    audit gets 3 admitted rows; no diversity drops."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    cands = [
        _make_candidate(source_game=f"iter_0001_game_{i:03d}", source_ply=50,
                        category=cat, cc_size=20 + i)
        for i, cat in enumerate([
            "chain_advantage_central_red",
            "chain_advantage_central_black",
            "chain_advantage_edge_red",
        ])
    ]
    audit = []

    kept = _select_diverse_admitted_candidates(
        cands, audit, max_probes=10, max_probes_per_game=2,
    )

    assert len(kept) == 3
    admitted_rows = [r for r in audit if r["reason"] == "admitted"]
    diversity_rows = [r for r in audit if r["reason"].startswith("diversity_")]
    assert len(admitted_rows) == 3
    assert len(diversity_rows) == 0
```

- [ ] **Step 2: Run the test to verify it fails**

```
pytest tests/test_strong_advantage_diversity_selector.py::test_selector_returns_all_when_under_max_probes_and_no_rules_fire -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the orchestrator**

In `scripts/build_probe_suite.py`, after `_find_per_game_cap_keeper`:

```python
def _select_diverse_admitted_candidates(
    admitted: list,
    audit: list,
    *,
    max_probes: int,
    max_probes_per_game: int,
) -> list:
    """Post-Phase-2 diversity-aware selector. Replaces the simple
    `admitted[: max_probes]` slice with a category round-robin walk
    applying near-duplicate, ply-separation, and per-game cap rules.

    Mutates `audit` in place: appends one row per CONSIDERED candidate
    (reason="admitted" if kept, reason="diversity_*" if dropped). Returns
    the kept list in selection order.

    Audit-coverage policy (Option A — by design): once `max_probes` is
    reached, the round-robin terminates and any remaining post-Phase-2
    candidates are NOT visited and therefore get NO audit row. The audit
    is exhaustive over CONSIDERED candidates, not over ALL Phase-2
    survivors. Rationale: an unvisited candidate is not "evicted by a
    rule" — it simply lost the race for a finite suite slot. Tagging
    every Phase-2 survivor with a "would have been considered next"
    pseudo-reason adds noise without diagnostic value. Operators who
    need a total Phase-2-admit count should derive it externally
    (e.g., len(admitted) before this function is called), not by
    counting audit rows.

    See spec §4.2 for the algorithm and §7 for audit semantics.
    """
    # Stage 1: bucket by category.
    buckets = {cat: [] for cat in CATEGORY_ITERATION_ORDER}
    for cand in admitted:
        cat = cand["category"]
        if cat in buckets:
            buckets[cat].append(cand)

    # Stage 2: rank within each category.
    for cat in buckets:
        buckets[cat].sort(key=_diversity_sort_key)

    # Build rank_index: id(cand) → rank position in its category.
    # Used by Rule B's tie-break (better Stage-2 rank wins).
    rank_index = {}
    for cands in buckets.values():
        for i, c in enumerate(cands):
            rank_index[id(c)] = i

    # Stage 3: round-robin walk with suppression rules.
    kept = []
    cursors = {cat: 0 for cat in CATEGORY_ITERATION_ORDER}

    while len(kept) < max_probes:
        progressed = False
        for cat in CATEGORY_ITERATION_ORDER:
            if len(kept) >= max_probes:
                break
            if cursors[cat] >= len(buckets[cat]):
                continue
            progressed = True
            cand = buckets[cat][cursors[cat]]
            cursors[cat] += 1

            audit_base = {
                "source_game": cand["source_game"],
                "source_ply": cand["source_ply"],
                "phase1_features": cand["phase1_features"],
                "phase2_label": cand["phase2_label"],
            }

            # Rule A: near-duplicate.
            keeper = _find_near_duplicate_keeper(cand, kept)
            if keeper is not None:
                audit.append({
                    **audit_base,
                    "reason": "diversity_near_duplicate",
                    "kept_instead_source_ply": keeper["source_ply"],
                })
                continue

            # Rule B: ply-too-close.
            keeper = _find_ply_too_close_keeper(cand, kept, rank_index)
            if keeper is not None:
                audit.append({
                    **audit_base,
                    "reason": "diversity_ply_too_close",
                    "kept_instead_source_ply": keeper["source_ply"],
                })
                continue

            # Rule C: per-game cap.
            keeper = _find_per_game_cap_keeper(cand, kept, max_probes_per_game)
            if keeper is not None:
                audit.append({
                    **audit_base,
                    "reason": "diversity_per_game_cap",
                    "kept_instead_source_ply": keeper["source_ply"],
                })
                continue

            # Admit.
            kept.append(cand)
            audit.append({**audit_base, "reason": "admitted"})

        if not progressed:
            break

    return kept
```

- [ ] **Step 4: Run the test (expect PASS)**

```
pytest tests/test_strong_advantage_diversity_selector.py::test_selector_returns_all_when_under_max_probes_and_no_rules_fire -v
```

Expected: PASS.

- [ ] **Step 5: Add focused tests for each rule firing through the orchestrator**

Append to `tests/test_strong_advantage_diversity_selector.py`:

```python
def test_selector_per_game_cap_test():
    """5 probes from one game (no near-dupes, well-separated plies) plus
    1 from another. cap=2 → 2 from clustered game survive; 3 dropped
    with diversity_per_game_cap; 1 from the other game survives."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    # Same game, 5 plies far enough apart to clear ply-separation,
    # cc_size descending so they don't trip near-duplicate.
    clustered = [
        _make_candidate(source_game="iter_0058_game_040", source_ply=ply,
                        category="chain_advantage_central_red", cc_size=cs)
        for ply, cs in [(40, 28), (44, 24), (48, 20), (52, 16), (56, 12)]
    ]
    other = _make_candidate(source_game="iter_0058_game_041", source_ply=50,
                            category="chain_advantage_central_red", cc_size=22)
    audit = []

    kept = _select_diverse_admitted_candidates(
        clustered + [other], audit, max_probes=10, max_probes_per_game=2,
    )

    kept_from_clustered = [k for k in kept if k["source_game"] == "iter_0058_game_040"]
    assert len(kept_from_clustered) == 2
    assert other in kept

    cap_drops = [r for r in audit if r["reason"] == "diversity_per_game_cap"]
    assert len(cap_drops) == 3
    for row in cap_drops:
        assert row["source_game"] == "iter_0058_game_040"
        assert row["kept_instead_source_ply"] == 40  # smallest kept ply


def test_selector_near_duplicate_suppression():
    """3 same-game same-category probes with cc_size=(20,21,25) and
    axis_span_margin=(0.20,0.21,0.40). Probes with cc_size=20 and 21 are
    duplicates; rank-2 of those is dropped with diversity_near_duplicate;
    cc_size=25 is kept as structurally distinct."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    cands = [
        _make_candidate(source_game="iter_0058_game_040", source_ply=40,
                        category="chain_advantage_central_red",
                        cc_size=20, axis_span_margin=0.20),
        _make_candidate(source_game="iter_0058_game_040", source_ply=44,
                        category="chain_advantage_central_red",
                        cc_size=21, axis_span_margin=0.21),
        _make_candidate(source_game="iter_0058_game_040", source_ply=48,
                        category="chain_advantage_central_red",
                        cc_size=25, axis_span_margin=0.40),
    ]
    audit = []

    kept = _select_diverse_admitted_candidates(
        cands, audit, max_probes=10, max_probes_per_game=3,
    )

    # cc_size=25 is rank-1 (sorts first), kept. Then cc_size=21 and 20
    # are siblings of 25 — but Δcc_size from 25 is 4 and 5 (≥ 2), so
    # they're NOT duplicates of 25. Walking in rank order: 25 is kept;
    # then 21 (rank 2) is checked — Δcc from 25 is 4, not a duplicate;
    # 21 is kept. Then 20 (rank 3) is checked — Δcc from 21 is 1 AND
    # Δasm = 0.01 → IS a duplicate → dropped.
    kept_cc_sizes = sorted(k["phase1_features"]["cc_size"] for k in kept)
    assert kept_cc_sizes == [21, 25]

    dup_drops = [r for r in audit if r["reason"] == "diversity_near_duplicate"]
    assert len(dup_drops) == 1
    assert dup_drops[0]["source_ply"] == 40  # cc_size=20 was the dropped one
    assert dup_drops[0]["kept_instead_source_ply"] == 44  # ply of cc_size=21 keeper


def test_selector_ply_separation():
    """3 same-game probes, structurally distinct (no near-dupes),
    source_ply ∈ {50, 51, 54}. cap=3 (so cap doesn't bind). Plies 50
    and 54 kept; 51 dropped with diversity_ply_too_close."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    # Stage-2 sort wants larger cc_size first. Order them so that 50
    # has the largest cc, then 54, then 51 — exercising the sort.
    cands = [
        _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                        category="chain_advantage_central_red", cc_size=28),
        _make_candidate(source_game="iter_0058_game_040", source_ply=54,
                        category="chain_advantage_central_red", cc_size=22),
        _make_candidate(source_game="iter_0058_game_040", source_ply=51,
                        category="chain_advantage_central_red", cc_size=18),
    ]
    audit = []

    kept = _select_diverse_admitted_candidates(
        cands, audit, max_probes=10, max_probes_per_game=3,
    )

    kept_plies = sorted(k["source_ply"] for k in kept)
    assert kept_plies == [50, 54]

    too_close = [r for r in audit if r["reason"] == "diversity_ply_too_close"]
    assert len(too_close) == 1
    assert too_close[0]["source_ply"] == 51
    assert too_close[0]["kept_instead_source_ply"] == 50  # closest keeper


def test_selector_drop_reason_precedence():
    """Synthetic candidate that triggers BOTH diversity_near_duplicate
    AND diversity_per_game_cap: audit reason must be near_duplicate
    (more specific wins)."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    # Two keepers from same game at cap=2, then a third candidate that
    # is also a near-duplicate of one of them.
    a = _make_candidate(source_game="iter_0058_game_040", source_ply=40,
                        category="chain_advantage_central_red",
                        cc_size=28, axis_span_margin=0.40)
    b = _make_candidate(source_game="iter_0058_game_040", source_ply=44,
                        category="chain_advantage_central_red",
                        cc_size=22, axis_span_margin=0.30)
    # c is a near-duplicate of b (Δcc=1, Δasm=0.01) AND would exceed
    # cap=2 if kept. Rule A fires first.
    c = _make_candidate(source_game="iter_0058_game_040", source_ply=48,
                        category="chain_advantage_central_red",
                        cc_size=21, axis_span_margin=0.31)
    audit = []

    kept = _select_diverse_admitted_candidates(
        [a, b, c], audit, max_probes=10, max_probes_per_game=2,
    )

    drop_rows = [r for r in audit if r["source_ply"] == 48]
    assert len(drop_rows) == 1
    assert drop_rows[0]["reason"] == "diversity_near_duplicate"
```

- [ ] **Step 6: Run the new tests (expect PASS)**

```
pytest tests/test_strong_advantage_diversity_selector.py -v -k selector_
```

Expected: 4 selector_ tests passed.

- [ ] **Step 7: Commit**

```
git add scripts/build_probe_suite.py tests/test_strong_advantage_diversity_selector.py
git commit -m "$(cat <<'EOF'
feat(probes): add diversity-selector orchestrator with round-robin walk

_select_diverse_admitted_candidates wires Rules A/B/C into a category
round-robin over the canonical 4-tuple. Mutates audit in place: writes
one row per kept (reason="admitted") or dropped (reason="diversity_*")
candidate. Spec §4.2.
EOF
)"
```

---

## Task 7 — CLI flag and meta enrichment

Adds `--max-probes-per-game` to argparse and writes the new selection-rules keys into the output payload's meta block. No selector wire-in yet (Task 8).

**Files:**
- Modify: `scripts/build_probe_suite.py`
- Modify: `tests/test_strong_advantage_diversity_selector.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_strong_advantage_diversity_selector.py`:

```python
def test_cli_accepts_max_probes_per_game_flag():
    """argparse accepts --max-probes-per-game with int, default 2.

    Inspects --help output via subprocess rather than importing the
    parser directly, since the parser is constructed inside main()
    and isn't exposed as a module-level object.
    """
    import subprocess
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "build_probe_suite.py"),
         "--tier", "strong_advantage", "--help"],
        capture_output=True, text=True, cwd=project_root,
    )
    assert "--max-probes-per-game" in result.stdout, (
        f"--max-probes-per-game flag not found in --help output:\n{result.stdout}"
    )
    assert "default: 2" in result.stdout.lower() or "default 2" in result.stdout.lower() \
        or "(default: 2)" in result.stdout, (
        f"Expected default of 2 documented in --help:\n{result.stdout}"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

```
pytest tests/test_strong_advantage_diversity_selector.py::test_cli_accepts_max_probes_per_game_flag -v
```

Expected: FAIL — `--max-probes-per-game` not in help output.

- [ ] **Step 3: Add the argparse flag**

In `scripts/build_probe_suite.py`, find the argparse block in `main()` (around line 36-51). Add this line right after `--max-probes`:

```python
    ap.add_argument("--max-probes-per-game", type=int, default=2,
                    help="Maximum number of admitted probes from any single "
                         "source game. Counts total across all 4 categories. "
                         "Default 2. Strong-advantage tier only.")
```

The existing `--max-probes` line provides the format reference; place the new line directly after it.

- [ ] **Step 4: Run the test (expect PASS)**

```
pytest tests/test_strong_advantage_diversity_selector.py::test_cli_accepts_max_probes_per_game_flag -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```
git add scripts/build_probe_suite.py tests/test_strong_advantage_diversity_selector.py
git commit -m "$(cat <<'EOF'
feat(probes): add --max-probes-per-game CLI flag (default 2)

Strong-advantage tier only. Counts total across all 4 categories.
Spec §6.
EOF
)"
```

---

## Task 8 — Wire selector into `_run_strong_advantage` + restructure Phase-2 audit writer

The big integration step. Three coordinated changes:

1. Phase-2 audit writer stops writing `reason="admitted"` rows; only writes for rejections and errors. The selector becomes the single writer of `reason="admitted"` rows (and `reason="diversity_*"` rows for evictions).
2. Replace `admitted = admitted[: args.max_probes]` with a call to `_select_diverse_admitted_candidates`.
3. Enrich `meta.selection_rules` in the output payload with the new diversity-related keys.

**Note on the integration test's monkeypatch pattern (Step 1):** The test patches `scripts.GPU.alphazero.probe_eval.label_candidate_with_mcts`, `apply_admission_filter`, `load_network_for_scoring`, and `_set_default_labeler_network` BEFORE invoking `main_with_args`. This relies on the fact that `_run_strong_advantage` does its `from scripts.GPU.alphazero.probe_eval import (...)` lazily, *inside* the function body (build_probe_suite.py:215-221), so the import resolves the patched module attributes at call time. **If those imports are ever hoisted to module scope, this test will silently start using the real implementations and must be updated** (e.g., switch to patching `scripts.build_probe_suite.label_candidate_with_mcts` etc. instead). Worth a comment in the test if you make the change later.

**Files:**
- Modify: `scripts/build_probe_suite.py` (in `_run_strong_advantage`)
- Modify: `tests/test_strong_advantage_diversity_selector.py`

- [ ] **Step 1: Write the failing integration test**

This test runs the full generator end-to-end with mocked Phase-2 labeling, asserts the selector ran and the audit/meta are consistent. Append to `tests/test_strong_advantage_diversity_selector.py`:

```python
def test_end_to_end_strong_advantage_runs_selector_and_writes_meta(tmp_path, monkeypatch):
    """Full _run_strong_advantage with stubbed labeler, network loader,
    and admission filter. Asserts:
    - draft file is written
    - audit file is written
    - admitted audit rows count == probes in draft (no double-counting)
    - meta.selection_rules contains the new diversity keys
    - per-game cap is upheld

    Stubs are applied to the module BEFORE main_with_args runs, so the
    function-local `from scripts.GPU.alphazero.probe_eval import ...`
    inside _run_strong_advantage picks up the stubbed callables.
    """
    import json
    from pathlib import Path
    from unittest.mock import MagicMock

    import scripts.GPU.alphazero.probe_eval as pe
    from scripts.build_probe_suite import main_with_args

    project_root = Path(__file__).resolve().parent.parent

    # Stub the labeler to return a sign-positive passing label. The
    # generator post-normalizes to red-perspective per STM, so a positive
    # raw value is fine for whichever side moved last.
    def stub_label(state, sims, repeats, rng_seed_base, labeler=None):
        return {
            "mean_root_value": 0.95,
            "value_per_run": [0.95, 0.95],
            "value_stability": 0.0,
            "min_top1_share": 0.30,
            "label_mcts_sims": sims,
            "label_mcts_repeats": repeats,
            "rng_seed_base": rng_seed_base,
        }
    monkeypatch.setattr(pe, "label_candidate_with_mcts", stub_label)

    # Stub admission filter to always pass. This decouples the
    # integration test from the (real) sign-checking logic — the
    # selector's behavior is the focus of this test, and the admission
    # filter is independently covered in test_strong_advantage_probe_suite.py.
    monkeypatch.setattr(pe, "apply_admission_filter",
                        lambda cand, **kwargs: (True, "admitted"))

    # Stub network loader. MagicMock so the subsequent .eval() call works.
    mock_network = MagicMock()
    monkeypatch.setattr(pe, "load_network_for_scoring",
                        lambda path: (mock_network, 24, 128, 6))
    monkeypatch.setattr(pe, "_set_default_labeler_network", lambda net: None)

    # Fake checkpoint file: must exist (existence check), never read.
    fake_ckpt = tmp_path / "fake.safetensors"
    fake_ckpt.write_bytes(b"fake")

    # Run against the same source range the committed file uses.
    out_path = tmp_path / "strong_advantage_probes.json"
    rc = main_with_args([
        "--tier", "strong_advantage",
        "--input", str(project_root / "scripts" / "GPU" / "logs" / "games"),
        "--source-iter-range", "57", "58",
        "--label-checkpoint", str(fake_ckpt),
        "--out", str(out_path),
        "--max-probes", "30",
        "--max-probes-per-game", "2",
        "--label-mcts-sims", "100",
        "--label-mcts-repeats", "1",
        "--force",
    ])
    assert rc == 0, f"generator exited {rc}"

    draft_path = out_path.with_suffix(".draft.json")
    audit_path = out_path.parent / "candidates_strong_advantage.json"
    assert draft_path.exists(), f"draft file missing: {draft_path}"
    assert audit_path.exists(), f"audit file missing: {audit_path}"

    draft = json.loads(draft_path.read_text())
    audit = json.loads(audit_path.read_text())["audit"]

    # No audit double-counting: admitted rows == probes in draft.
    admitted_rows = [r for r in audit if r["reason"] == "admitted"]
    assert len(admitted_rows) == len(draft["probes"]), (
        f"audit admitted count ({len(admitted_rows)}) != probes "
        f"({len(draft['probes'])})"
    )

    # Per-game cap upheld.
    from collections import Counter
    per_game = Counter(p["source_game"] for p in draft["probes"])
    assert all(n <= 2 for n in per_game.values()), (
        f"per-game cap of 2 violated: {per_game.most_common(5)}"
    )

    # meta.selection_rules has the new keys.
    rules = draft["meta"]["selection_rules"]
    assert rules["max_probes_per_game"] == 2
    assert rules["min_ply_separation_same_game"] == 3
    assert rules["category_iteration_order"] == [
        "chain_advantage_central_red",
        "chain_advantage_central_black",
        "chain_advantage_edge_red",
        "chain_advantage_edge_black",
    ]
    assert "diversity_quality_key_order" in rules
    assert isinstance(rules["diversity_quality_key_order"], list)
    assert len(rules["diversity_quality_key_order"]) >= 6
```

- [ ] **Step 2: Run the test to verify it fails**

```
pytest tests/test_strong_advantage_diversity_selector.py::test_end_to_end_strong_advantage_runs_selector_and_writes_meta -v
```

Expected: FAIL. Likely failure: the selector isn't called (so per-game cap is violated), or `meta.selection_rules` lacks the new keys, or admitted rows are double-counted.

- [ ] **Step 3: Restructure the Phase-2 audit writer**

In `scripts/build_probe_suite.py`, find the Phase-2 loop block (around lines 337-353). Locate this section:

```python
        cand["phase2_label"] = label
        ok, reason = apply_admission_filter(
            cand,
            magnitude_threshold=args.magnitude_threshold,
            top1_share_floor=args.top1_share_floor,
            stability_cap=args.stability_cap,
        )
        cand["phase2_label"]["label_checkpoint"] = label_ckpt.name
        audit.append({
            "source_game": cand["source_game"],
            "source_ply": cand["source_ply"],
            "phase1_features": cand["phase1_features"],
            "phase2_label": cand["phase2_label"],
            "reason": reason,
        })
        if ok:
            admitted.append(cand)
```

Change it to write the audit row only when the candidate is rejected:

```python
        cand["phase2_label"] = label
        ok, reason = apply_admission_filter(
            cand,
            magnitude_threshold=args.magnitude_threshold,
            top1_share_floor=args.top1_share_floor,
            stability_cap=args.stability_cap,
        )
        cand["phase2_label"]["label_checkpoint"] = label_ckpt.name
        if ok:
            # Admitted candidates' audit rows are written by the selector
            # (Task 8), so they reflect the FINAL outcome (reason="admitted"
            # for survivors, reason="diversity_*" for evictions). Spec §7.1.
            admitted.append(cand)
        else:
            audit.append({
                "source_game": cand["source_game"],
                "source_ply": cand["source_ply"],
                "phase1_features": cand["phase1_features"],
                "phase2_label": cand["phase2_label"],
                "reason": reason,
            })
```

- [ ] **Step 4: Replace the slice with the selector call**

In `scripts/build_probe_suite.py`, find this line (around line 380):

```python
    admitted = admitted[: args.max_probes]
```

Replace it with:

```python
    admitted = _select_diverse_admitted_candidates(
        admitted,
        audit,
        max_probes=args.max_probes,
        max_probes_per_game=args.max_probes_per_game,
    )
```

- [ ] **Step 5: Enrich `meta.selection_rules`**

Find the `selection_rules` dict in the payload construction (around lines 412-436). After the existing keys (`board_size`, `winner_reasons`, `k_plies_from_terminal_range`, `phase1_thresholds`, `phase2_thresholds`, `label_checkpoint`, `label_checkpoint_sha256`, `source_iter_range`, `dedup`, `category_min_count`), add:

```python
                "max_probes_per_game": args.max_probes_per_game,
                "min_ply_separation_same_game": MIN_PLY_SEPARATION_SAME_GAME,
                "category_iteration_order": list(CATEGORY_ITERATION_ORDER),
                "diversity_quality_key_order": [
                    "phase1_features.cc_size desc",
                    "phase1_features.axis_span_margin desc",
                    "phase1_features.cc_axis_span desc",
                    "phase2_label.min_top1_share desc",
                    "phase2_label.value_stability asc",
                    "default_sort_key (-iter, -source_ply, source_game)",
                ],
```

- [ ] **Step 6: Run the integration test (expect PASS)**

```
pytest tests/test_strong_advantage_diversity_selector.py::test_end_to_end_strong_advantage_runs_selector_and_writes_meta -v
```

Expected: PASS.

- [ ] **Step 7: Run the full new test file (expect no regressions)**

```
pytest tests/test_strong_advantage_diversity_selector.py -v
```

Expected: all selector tests PASS.

- [ ] **Step 8: Run the existing strong_advantage tests (expect no regressions)**

```
pytest tests/test_strong_advantage_probe_suite.py -v
```

Expected: all PASS.

- [ ] **Step 9: Commit**

```
git add scripts/build_probe_suite.py tests/test_strong_advantage_diversity_selector.py
git commit -m "$(cat <<'EOF'
feat(probes): wire diversity selector into _run_strong_advantage

Phase 2 stops writing reason="admitted" audit rows; the selector
becomes the single writer of admitted/diversity_* rows. Replaces the
admitted[: args.max_probes] slice with a category round-robin walk.
Enriches meta.selection_rules with diversity keys. Spec §4 / §6 / §7.
EOF
)"
```

---

## Task 9 — Determinism and empty-skip tests

Pinning down the determinism promise: same input → byte-identical output regardless of input order, and the canonical category order is preserved when middle buckets are empty.

**Files:**
- Modify: `tests/test_strong_advantage_diversity_selector.py`

- [ ] **Step 1: Write the determinism test**

Append to `tests/test_strong_advantage_diversity_selector.py`:

```python
def test_selector_determinism_under_input_shuffle():
    """Same admitted list in different orders must produce byte-identical
    selector output AND identical audit deltas."""
    from copy import deepcopy
    import random

    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    base_cands = [
        _make_candidate(source_game=f"iter_{50 + g:04d}_game_{g:03d}",
                        source_ply=40 + g * 3,
                        category=cat, cc_size=20 + g)
        for g in range(8)
        for cat in ["chain_advantage_central_red", "chain_advantage_central_black"]
    ]

    rng = random.Random(42)
    shuffled = deepcopy(base_cands)
    rng.shuffle(shuffled)

    audit_a = []
    kept_a = _select_diverse_admitted_candidates(
        deepcopy(base_cands), audit_a, max_probes=10, max_probes_per_game=2,
    )
    audit_b = []
    kept_b = _select_diverse_admitted_candidates(
        shuffled, audit_b, max_probes=10, max_probes_per_game=2,
    )

    # Compare by stable identity (source_game, source_ply, category).
    def _identity(c):
        return (c["source_game"], c["source_ply"], c["category"])

    assert [_identity(k) for k in kept_a] == [_identity(k) for k in kept_b], (
        "kept order differs between original and shuffled input"
    )

    audit_keys_a = sorted(
        (r["source_game"], r["source_ply"], r["reason"]) for r in audit_a
    )
    audit_keys_b = sorted(
        (r["source_game"], r["source_ply"], r["reason"]) for r in audit_b
    )
    assert audit_keys_a == audit_keys_b, "audit deltas differ between runs"


def test_round_robin_skips_empty_without_reordering_nonempty():
    """Only central_black (position 2) and edge_red (position 3) populated;
    central_red (1) and edge_black (4) empty. Selector walks the canonical
    4-tuple, skips empties, and the relative order between the two
    non-empty categories matches [central_black, edge_red] repeating —
    NOT alphabetical or yield-based. Spec §5.4."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    # 4 candidates each in central_black and edge_red, all from different
    # games so per-game cap doesn't bind.
    cands = []
    for i in range(4):
        cands.append(_make_candidate(
            source_game=f"iter_{60 + i:04d}_game_001",
            source_ply=40, category="chain_advantage_central_black",
            cc_size=30 - i,  # decreasing so insertion-order != rank-order
        ))
        cands.append(_make_candidate(
            source_game=f"iter_{70 + i:04d}_game_001",
            source_ply=40, category="chain_advantage_edge_red",
            cc_size=30 - i,
        ))

    audit = []
    kept = _select_diverse_admitted_candidates(
        cands, audit, max_probes=8, max_probes_per_game=2,
    )

    # First 4 picks should alternate central_black, edge_red, central_black,
    # edge_red because the round-robin walks canonical order (1=skip empty,
    # 2=central_black, 3=edge_red, 4=skip empty), and after one pass takes
    # one from each non-empty bucket.
    cats = [k["category"] for k in kept]
    expected_pattern = ["chain_advantage_central_black", "chain_advantage_edge_red"] * 4
    assert cats == expected_pattern, (
        f"Expected canonical-order alternation, got: {cats}"
    )
```

- [ ] **Step 2: Run the tests (expect PASS — selector already implemented)**

```
pytest tests/test_strong_advantage_diversity_selector.py -v -k "determinism or skips_empty"
```

Expected: 2 passed.

- [ ] **Step 3: Add the cross-category-not-deduped, sparse-backfill, and edge-empty tests**

Append to `tests/test_strong_advantage_diversity_selector.py`:

```python
def test_cross_category_same_game_not_deduped():
    """Two same-game probes with structural deltas BELOW thresholds but in
    DIFFERENT categories. Rule A's same-category requirement prevents
    dedupe; both kept (under cap). Spec §5.6."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    a = _make_candidate(source_game="iter_0058_game_040", source_ply=40,
                        category="chain_advantage_central_red",
                        cc_size=20, axis_span_margin=0.30)
    b = _make_candidate(source_game="iter_0058_game_040", source_ply=44,  # > sep
                        category="chain_advantage_edge_red",
                        cc_size=21, axis_span_margin=0.31)  # Δcc=1, Δasm=0.01
    audit = []

    kept = _select_diverse_admitted_candidates(
        [a, b], audit, max_probes=10, max_probes_per_game=2,
    )

    assert len(kept) == 2
    assert {k["category"] for k in kept} == {
        "chain_advantage_central_red", "chain_advantage_edge_red"
    }
    near_dup = [r for r in audit if r["reason"] == "diversity_near_duplicate"]
    assert near_dup == []


def test_sparse_category_backfill():
    """Only central_red populated (10 candidates from 5 games, 2 per game).
    max_probes=10, cap=2. All 10 kept; no errors; round-robin gracefully
    skips empties."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    cands = []
    for game_idx in range(5):
        for ply_offset in [0, 4]:  # > separation
            cands.append(_make_candidate(
                source_game=f"iter_0099_game_{game_idx:03d}",
                source_ply=40 + ply_offset,
                category="chain_advantage_central_red",
                cc_size=28 - game_idx - ply_offset,  # all distinct
                axis_span_margin=0.40 - game_idx * 0.06 - ply_offset * 0.01,
            ))
    audit = []

    kept = _select_diverse_admitted_candidates(
        cands, audit, max_probes=10, max_probes_per_game=2,
    )

    assert len(kept) == 10
    assert all(k["category"] == "chain_advantage_central_red" for k in kept)


def test_edge_categories_empty_alternates_two_centrals():
    """central_red and central_black populated, edge_* empty. Round-robin
    alternates central_red ↔ central_black; output respects max_probes."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    cands = []
    for game_idx in range(6):
        for cat in ["chain_advantage_central_red", "chain_advantage_central_black"]:
            cands.append(_make_candidate(
                source_game=f"iter_0099_game_{game_idx:03d}_{cat[-3:]}",
                source_ply=40,
                category=cat,
                cc_size=28 - game_idx,
            ))
    audit = []

    kept = _select_diverse_admitted_candidates(
        cands, audit, max_probes=6, max_probes_per_game=2,
    )

    assert len(kept) == 6
    cats = [k["category"] for k in kept]
    # First 6 picks alternate central_red, central_black, central_red, ...
    expected = ["chain_advantage_central_red", "chain_advantage_central_black"] * 3
    assert cats == expected
```

- [ ] **Step 4: Run all selector tests (expect PASS)**

```
pytest tests/test_strong_advantage_diversity_selector.py -v
```

Expected: all PASS (now including determinism, empty-skip, cross-category, sparse, edge-empty).

- [ ] **Step 5: Add the kept_instead and audit-canonical tests**

Append to `tests/test_strong_advantage_diversity_selector.py`:

```python
def test_audit_kept_instead_field_present_on_diversity_drops_only():
    """diversity_* rows carry kept_instead_source_ply pointing at a real
    keeper. admitted rows do NOT have this field."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    # Setup: clustered game forces per-game cap drops.
    cands = [
        _make_candidate(source_game="iter_0058_game_040", source_ply=ply,
                        category="chain_advantage_central_red", cc_size=cs)
        for ply, cs in [(40, 28), (44, 24), (48, 20)]
    ]
    audit = []
    kept = _select_diverse_admitted_candidates(
        cands, audit, max_probes=10, max_probes_per_game=2,
    )
    kept_plies = {k["source_ply"] for k in kept}

    for row in audit:
        if row["reason"].startswith("diversity_"):
            assert "kept_instead_source_ply" in row
            assert row["kept_instead_source_ply"] in kept_plies, (
                f"kept_instead_source_ply={row['kept_instead_source_ply']} "
                f"not in actually-kept plies {kept_plies}"
            )
        else:
            assert row["reason"] == "admitted"
            assert "kept_instead_source_ply" not in row


def test_audit_admitted_count_equals_kept_count():
    """Selector writes exactly one reason='admitted' audit row per kept
    candidate. No double-counting."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    cands = [
        _make_candidate(source_game=f"iter_0099_game_{i:03d}",
                        source_ply=40,
                        category="chain_advantage_central_red",
                        cc_size=28 - i)
        for i in range(5)
    ]
    audit = []
    kept = _select_diverse_admitted_candidates(
        cands, audit, max_probes=3, max_probes_per_game=1,
    )

    admitted_rows = [r for r in audit if r["reason"] == "admitted"]
    assert len(admitted_rows) == len(kept) == 3


def test_quality_key_structural_priority():
    """Two same-game same-category candidates: one with higher cc_size
    but lower min_top1_share, the other with lower cc_size but higher
    min_top1_share, structurally far enough apart that the near-duplicate
    rule does not fire (cc_size = (15, 25)). With max_probes_per_game=1,
    the higher cc_size wins (structural beats Phase-2 fields)."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    high_struct = _make_candidate(
        source_game="iter_0058_game_040", source_ply=40,
        category="chain_advantage_central_red",
        cc_size=25, min_top1_share=0.20,
    )
    high_phase2 = _make_candidate(
        source_game="iter_0058_game_040", source_ply=44,
        category="chain_advantage_central_red",
        cc_size=15, min_top1_share=0.50,
    )
    audit = []

    kept = _select_diverse_admitted_candidates(
        [high_phase2, high_struct], audit,  # input order shouldn't matter
        max_probes=10, max_probes_per_game=1,
    )

    assert len(kept) == 1
    assert kept[0] is high_struct
```

- [ ] **Step 6: Run all selector tests (expect PASS)**

```
pytest tests/test_strong_advantage_diversity_selector.py -v
```

Expected: all 14+ tests pass (the basic selector tests from Task 6, the per-rule tests from Tasks 3-5, the integration from Task 8, the determinism/empty/cross-category/sparse/edge from this task, plus the kept_instead and quality_key_structural tests).

- [ ] **Step 7: Commit**

```
git add tests/test_strong_advantage_diversity_selector.py
git commit -m "$(cat <<'EOF'
test(probes): add determinism, empty-skip, and cross-cutting selector tests

Covers: input-shuffle determinism, canonical-order skip-empty without
reordering non-empty buckets, cross-category same-game non-dedupe,
sparse-category backfill, edge-categories-empty alternation,
kept_instead_source_ply field semantics, audit admitted count, and
structural-vs-phase2 quality key priority. Spec §9 tests 1-14.
EOF
)"
```

---

## Task 10 — Update operator doc

Document the new flag, the diversity rules, the new audit reasons, and the regeneration that occurred.

**Files:**
- Modify: `docs/probe-suite-generation.md`

- [ ] **Step 1: Read the existing doc to find the right insertion points**

```
cat docs/probe-suite-generation.md | head -120
```

Identify:
- The CLI flags / argparse documentation section (where `--max-probes` is documented).
- The audit-reasons section (where Phase-1 and Phase-2 reasons are listed).
- The "what the file contains" section (where to mention `meta.selection_rules` additions).

- [ ] **Step 2: Add the `--max-probes-per-game` documentation**

In the section that documents CLI flags for `--tier strong_advantage`, add (after the `--max-probes` documentation):

```markdown
- `--max-probes-per-game N` (default 2)

  Maximum number of admitted probes from any single source game,
  counted total across all 4 categories. Combined with the internal
  `MIN_PLY_SEPARATION_SAME_GAME=3` constant, this ensures that no
  single game contributes more than 2 probes and that those 2 (when
  the game contributes 2) are at least 3 plies apart in the source
  trajectory. Default 2 is conservative; increase only if the suite
  is consistently undersized.
```

- [ ] **Step 3: Add the diversity rules and audit-reasons documentation**

In the audit-reasons section (or create one if absent), add:

```markdown
### Diversity selector audit reasons (post-Phase-2)

The selector applies three suppression rules in precedence order. Each
drop produces one audit row with `phase2_label` and a
`kept_instead_source_ply` field pointing at the keeper that triggered
the drop:

- `diversity_near_duplicate` — same source_game AND same category AND
  `|Δcc_size| < 2 AND |Δaxis_span_margin| < 0.05`. The candidate is a
  structural near-duplicate of an already-kept sibling.

- `diversity_ply_too_close` — same source_game AND `|Δsource_ply| < 3`
  (any category — the rule is category-agnostic). The candidate sits
  too close in the source trajectory to an already-kept sibling.

- `diversity_per_game_cap` — the source_game already has
  `--max-probes-per-game` keepers (default 2, total across categories).
  The candidate is dropped to keep no single game over-represented.

`reason="admitted"` audit rows correspond exactly 1:1 with the probes
in the committed suite. Phase 1 and Phase 2 admit rows were collapsed
into a single canonical post-selection row in the audit-cleanup change.
```

- [ ] **Step 4: Add the meta.selection_rules note**

In the "what the file contains" section, add:

```markdown
The strong_advantage tier writes the diversity-selector configuration
into `meta.selection_rules`:

- `max_probes_per_game` — the value of the CLI flag at generation time.
- `min_ply_separation_same_game` — fixed at 3 in the current
  implementation (tied to the K-range [3, 8]).
- `category_iteration_order` — the canonical 4-tuple the round-robin
  walks: `[central_red, central_black, edge_red, edge_black]`.
- `diversity_quality_key_order` — the Stage-2 sort precedence used to
  rank candidates within each category before the round-robin walk.
```

- [ ] **Step 5: Commit**

```
git add docs/probe-suite-generation.md
git commit -m "$(cat <<'EOF'
docs(probes): document diversity selector flag, rules, and audit reasons

Adds operator documentation for --max-probes-per-game, the three
suppression rules with their precedence, the new diversity_*
audit reasons, and the meta.selection_rules diversity keys.
EOF
)"
```

---

## Task 11 — Operator regeneration of the committed suite

This task is performed BY THE OPERATOR (the user), not by the implementing engineer. The plan documents it explicitly so it doesn't get forgotten and the historical record is honest.

The current committed `tests/probes/strong_advantage_probes.json` is a 22-probe hand-thinned snapshot. After the selector lands, the suite should be regenerated under the new selector with the same source range and label checkpoint, and the new file committed in place of the hand-thinned one.

**Files:**
- Operator-modified: `tests/probes/strong_advantage_probes.json`
- Operator-modified: `tests/probes/candidates_strong_advantage.json`

- [ ] **Step 1: Operator confirms the implementation is complete**

```
pytest tests/test_strong_advantage_probe_suite.py tests/test_strong_advantage_diversity_selector.py -v
```

All tests must pass before regeneration.

- [ ] **Step 2: Operator records the committed file's original generator invocation**

```
grep -A 3 "label_checkpoint\":" tests/probes/strong_advantage_probes.json | head -5
grep -A 3 "source_iter_range\":" tests/probes/strong_advantage_probes.json | head -5
```

Note the values — these will be reused for the regeneration so source range and labeling checkpoint are unchanged.

- [ ] **Step 3: Operator regenerates the draft**

Replace `<min>`, `<max>`, and `<path>` with the values from Step 2:

```
python3 scripts/build_probe_suite.py --tier strong_advantage \
    --source-iter-range <min> <max> \
    --label-checkpoint <path> \
    --force
```

Inspect the new draft:

```
python3 -c "
import json
from collections import Counter
d = json.load(open('tests/probes/strong_advantage_probes.draft.json'))
print('n probes:', len(d['probes']))
print('by category:', Counter(p['category'] for p in d['probes']).most_common())
per_game = Counter(p['source_game'] for p in d['probes'])
print('per-game distribution:', Counter(per_game.values()).most_common())
print('top games:', per_game.most_common(5))
a = json.load(open('tests/probes/candidates_strong_advantage.json'))['audit']
from collections import Counter as C
print('audit reasons:', C(r['reason'] for r in a).most_common())
"
```

Verify:
- Per-game distribution shows max 2.
- `diversity_*` reasons appear in the audit (proves the selector ran).
- `admitted` count in the audit equals the probe count in the draft.

- [ ] **Step 4: Operator promotes the regenerated suite**

```
python3 scripts/build_probe_suite.py --tier strong_advantage --promote --reviewer "<name>" --force
```

- [ ] **Step 5: Operator commits the regenerated files**

```
git add tests/probes/strong_advantage_probes.json tests/probes/candidates_strong_advantage.json
git commit -m "$(cat <<'EOF'
data(probes): regenerate strong_advantage suite under diversity-aware selector

The previous committed file was a manually-thinned 22-probe snapshot
of the unbalanced --max-probes=30 draft from the original run. The
new file is reproducible from:
  scripts/build_probe_suite.py --tier strong_advantage \
      --source-iter-range <min> <max> \
      --label-checkpoint <path> --max-probes-per-game 2

Source range and label checkpoint unchanged. The hand-thinned file is
preserved in git history. Light-reviewed (reviewer in meta.reviewer).
EOF
)"
```

- [ ] **Step 6: Operator runs the full test suite**

```
pytest tests/ -v -k "probe"
```

Expected: all pass against the regenerated artifact.

---

## Self-review

After writing all tasks above, verify:

**1. Spec coverage:**

- §2 Goals — diverse across games (Tasks 5, 8 selector + Task 11 regeneration), diverse across categories (Tasks 6, 9), deterministic (Task 9), audit-visible (Tasks 1, 8, 9), doesn't sacrifice yield from sparse categories (Task 9), honest admitted count (Tasks 1, 8, 9). ✓
- §3 Non-goals — no Phase-1 changes (correctly avoided), no multi-mode CLI (only `--max-probes-per-game`), no tunable quality key (hard-coded in Task 2). ✓
- §4.2 Pipeline (Stages 1-3, Rules A-C) — Stage 1+2 in Task 6 (bucket+rank in selector); Rules A/B/C in Tasks 3/4/5 with their helpers; Stage 3 walk in Task 6. ✓
- §5 Resolved questions — ply-separation as constant (Task 2), category iteration order (Task 2), structural-first quality key (Task 2), per-game cap total across categories (Task 5), same-category clause (Task 3 + test in Task 9). ✓
- §6 CLI surface — `--max-probes-per-game` (Task 7), meta.selection_rules enrichment (Task 8). ✓
- §7 Audit changes — Phase-1 admitted row removal (Task 1), Phase-2 admitted row removal (Task 8 step 3), diversity reasons + kept_instead_source_ply (Tasks 3-6, 9). ✓
- §9 Test plan — all 14 tests covered: per_game_cap (Task 6), near_duplicate (Task 6), ply_separation (Task 6), category_round_robin canonical order (Task 9 via empty-skip test which exercises the canonical traversal), sparse_category_backfill (Task 9), edge_categories_empty (Task 9), drop_reason_precedence (Task 6), determinism_under_input_shuffle (Task 9), audit_admitted_count_canonical (Task 9), audit_kept_instead_field (Task 9), quality_key_structural_priority (Task 9), cross_category_same_game_not_deduped (Task 9), meta_selection_rules_recorded (Task 8 integration test asserts these), round_robin_skips_empty_without_reordering_nonempty (Task 9). ✓
- §10 Regeneration — Task 11. ✓

**2. Placeholder scan:** No "TBD"/"TODO"/"implement later" found. Each step has either complete code, a complete command with expected output, or an explicit operator-action. ✓

**3. Type/name consistency:**
- `_diversity_sort_key` — defined Task 2, used Task 6 (orchestrator passes it to `bucket.sort(key=_diversity_sort_key)`). ✓
- `_find_near_duplicate_keeper(cand, kept) -> dict | None` — defined Task 3, called Task 6. ✓
- `_find_ply_too_close_keeper(cand, kept, rank_index) -> dict | None` — defined Task 4, called Task 6 with the rank_index built in Task 6. ✓
- `_find_per_game_cap_keeper(cand, kept, cap) -> dict | None` — defined Task 5, called Task 6 (note: signature uses `cap` not `max_probes_per_game`; the caller passes `max_probes_per_game` as positional). ✓
- `_select_diverse_admitted_candidates(admitted, audit, *, max_probes, max_probes_per_game) -> list` — defined Task 6, called Task 8. ✓
- `MIN_PLY_SEPARATION_SAME_GAME` constant — defined Task 2, used by `_find_ply_too_close_keeper` (Task 4) and the meta enrichment (Task 8). ✓
- `CATEGORY_ITERATION_ORDER` — defined Task 2, used by `_select_diverse_admitted_candidates` (Task 6) and meta enrichment (Task 8). ✓

All names match across tasks.
