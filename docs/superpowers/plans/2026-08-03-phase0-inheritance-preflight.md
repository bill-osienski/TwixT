# Phase 0 Inheritance Preflight — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how much of its search tree a shipped self-play search inherits through tree reuse, and emit a frozen binary verdict deciding whether the convergence atlas must use a warm-start root regime.

**Architecture:** One new module `inheritance_probe.py` holding the tracker (producer), the summary, and the frozen verdict evaluator (consumer). It is wired into `self_play.play_game` through one optional parameter defaulting to `None`, mirroring the existing `RecoveryRetargetingConfig` / `RecoveryRetargetingTracker` pattern already in that function. A thin CLI runs exactly one game and evaluates the verdict. `mcts.py` is not modified.

**Tech Stack:** Python 3, stdlib only (`statistics`, `dataclasses`, `json`, `argparse`). No scipy — it is not in `.venv`. Tests are pytest, run as `.venv/bin/python -m pytest -p no:cacheprovider`.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-08-03-convergence-atlas-design.md` §2. §2 is FROZEN; do not alter the protocol or the decision rule.
- **No `mcts.py` change.** Not one line. `forced_count` is obtained caller-side as a delta of the existing monotonic counter `MCTS._closeout_td1_forced_sims_total`.
- **Byte-identical off.** `inheritance_probe_config=None` must leave `play_game` behaviour bit-for-bit unchanged. The entire pre-existing test suite must pass unchanged.
- **An undefined row-level or summary statistic is `None`, never `0.0`,** and never a reason to drop a row or abort a run. This is a durable v18 lesson and is load-bearing here: a phase with no searches has an undefined median, not a zero one.
- **Fail loud on impossible input.** A negative forced-sim delta means the telemetry counter was reset mid-search; raise, never clamp. Out-of-order lifecycle calls raise too.
- **Phase 0 is a technical preflight, not evidence.** Its rows are serially correlated single-trajectory observations. No later work may cite it as an inheritance distribution.
- **Frozen decision rule (design §2, as amended 2026-08-03):** median `inherited_fraction_320 >= 0.10` in any post-opening phase **or** overall p75 `>= 0.20` → `WARM_START_REQUIRED`, and this stands even under partial coverage. Nothing crossed **and** every post-opening phase observed → `FRESH_ROOT_ACCEPTABLE`. Nothing crossed **and** coverage incomplete → `PREFLIGHT_INCOMPLETE`, which resolves nothing and must **not** trigger another game.
- **Frozen phase bounds:** opening `0–30`, early-mid `31–60`, midgame `61–90`, late `91+`. Post-opening = early-mid, midgame, late.
- **Shipped batching is `(14, 48, 8)`** = `eval_batch_size=14`, `stall_flush_sims=48`, `pending_virtual_visits=8`. Note `MCTSConfig.stall_flush_sims` defaults to `16`, so **48 must be set explicitly** wherever shipped batching is required.
- Commit after every task. Do not run the preflight itself — this plan builds and qualifies the instrument only.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/GPU/alphazero/inheritance_probe.py` (create) | Phase classification, row model, tracker, summary, frozen verdict evaluator. Single module: the v18 lesson is to drive real producers into real consumers, and keeping them adjacent makes the integration test trivial. |
| `scripts/GPU/alphazero/self_play.py` (modify) | One import, one `GameRecord` field, one optional parameter, one tracker construction, three observation call sites, one finalize. Default `None` = no-op. |
| `scripts/GPU/alphazero/run_inheritance_preflight.py` (create) | CLI: play exactly one game, write the artifact, print the verdict. |
| `tests/test_inheritance_probe.py` (create) | Unit tests for classification, rows, tracker, summary, verdict, and threshold pinning. |
| `tests/test_inheritance_probe_identity.py` (create) | Batched observer-on/off identity qualification, and the real-producer-to-real-consumer integration test. |

---

### Task 1: Phase classification and the inherited-fraction row

**Files:**
- Create: `scripts/GPU/alphazero/inheritance_probe.py`
- Test: `tests/test_inheritance_probe.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `phase_for_ply(ply: int) -> str`; `inherited_fraction_320(starting_visits: int) -> float`; `@dataclass SearchRow` with fields `ply: int`, `phase: str`, `starting_visits: int`, `starting_visited_children: int`, `forced_count: int`, `played_child_visits: Optional[int]`, `inherited_fraction_320: float`; module constants `DECISION_POINT_SIMS`, `POST_OPENING_MEDIAN_LIMIT`, `OVERALL_P75_LIMIT`, `POST_OPENING_PHASES`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inheritance_probe.py
import pytest

from scripts.GPU.alphazero.inheritance_probe import (
    DECISION_POINT_SIMS,
    SearchRow,
    inherited_fraction_320,
    phase_for_ply,
)


@pytest.mark.parametrize(
    "ply,expected",
    [
        (0, "opening"), (30, "opening"),
        (31, "early_mid"), (60, "early_mid"),
        (61, "midgame"), (90, "midgame"),
        (91, "late"), (279, "late"),
    ],
)
def test_phase_boundaries_are_exact(ply, expected):
    assert phase_for_ply(ply) == expected


def test_negative_ply_is_rejected():
    with pytest.raises(ValueError):
        phase_for_ply(-1)


def test_inherited_fraction_uses_the_320_decision_point():
    assert DECISION_POINT_SIMS == 320
    # A fresh root inherits nothing -- genuinely zero, not undefined.
    assert inherited_fraction_320(0) == 0.0
    # 166 inherited visits is the v16a-implied magnitude.
    assert inherited_fraction_320(166) == pytest.approx(166 / 486)
    assert inherited_fraction_320(320) == pytest.approx(0.5)


def test_negative_starting_visits_is_rejected():
    with pytest.raises(ValueError):
        inherited_fraction_320(-1)


def test_search_row_computes_its_own_fraction():
    row = SearchRow.build(
        ply=95, starting_visits=166, starting_visited_children=41, forced_count=0
    )
    assert row.phase == "late"
    assert row.inherited_fraction_320 == pytest.approx(166 / 486)
    # Not yet observed -- None, never 0.
    assert row.played_child_visits is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_inheritance_probe.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.GPU.alphazero.inheritance_probe'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/GPU/alphazero/inheritance_probe.py
"""Phase 0 inheritance preflight -- convergence atlas design, section 2.

Read-only characterization of how much of its search tree a shipped self-play
search inherits through `MCTS.advance_root` tree reuse.

This is a TECHNICAL PREFLIGHT, NOT EVIDENCE. It observes a single game
trajectory, so its rows are serially correlated and one game's inheritance may
be atypical. Adequate for the frozen binary regime decision; not adequate to
characterize the inheritance distribution.

CPU-SAFE: stdlib only, no MLX, no scipy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

# --- Frozen decision rule (design section 2). ---------------------------
# Changing any of these changes the preflight verdict.
# `test_frozen_thresholds_are_pinned` fails if they move.
DECISION_POINT_SIMS: int = 320
POST_OPENING_MEDIAN_LIMIT: float = 0.10
OVERALL_P75_LIMIT: float = 0.20

# Frozen phase bounds: (name, first_ply, last_ply_inclusive_or_None).
PHASE_BOUNDS: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("opening", 0, 30),
    ("early_mid", 31, 60),
    ("midgame", 61, 90),
    ("late", 91, None),
)
POST_OPENING_PHASES: Tuple[str, ...] = ("early_mid", "midgame", "late")


def phase_for_ply(ply: int) -> str:
    """Frozen phase bucket for a ply index."""
    if ply < 0:
        raise ValueError(f"ply must be non-negative, got {ply}")
    for name, lo, hi in PHASE_BOUNDS:
        if ply >= lo and (hi is None or ply <= hi):
            return name
    raise AssertionError(f"unreachable: no phase for ply {ply}")


def inherited_fraction_320(starting_visits: int) -> float:
    """Design section 2: starting_visits / (starting_visits + 320).

    A fresh root gives exactly 0.0. That is a genuine zero, not an undefined
    value -- do not convert it to None.
    """
    if starting_visits < 0:
        raise ValueError(f"starting_visits must be non-negative, got {starting_visits}")
    return starting_visits / (starting_visits + DECISION_POINT_SIMS)


@dataclass
class SearchRow:
    """One shipped search's start-of-search telemetry."""

    ply: int
    phase: str
    starting_visits: int
    starting_visited_children: int
    forced_count: int
    inherited_fraction_320: float
    # Filled in immediately before `advance_root`. None until observed --
    # never 0, which would be indistinguishable from a genuinely unvisited
    # played child.
    played_child_visits: Optional[int] = None

    @classmethod
    def build(
        cls,
        ply: int,
        starting_visits: int,
        starting_visited_children: int,
        forced_count: int,
    ) -> "SearchRow":
        if starting_visited_children < 0:
            raise ValueError("starting_visited_children must be non-negative")
        if forced_count < 0:
            # The tracker always builds with 0 and fills this in at
            # observe_search_end; this guard covers direct construction.
            raise ValueError(f"forced_count must be non-negative, got {forced_count}")
        return cls(
            ply=ply,
            phase=phase_for_ply(ply),
            starting_visits=starting_visits,
            starting_visited_children=starting_visited_children,
            forced_count=forced_count,
            inherited_fraction_320=inherited_fraction_320(starting_visits),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_inheritance_probe.py -v -p no:cacheprovider`
Expected: PASS — 12 passed (test_phase_boundaries_are_exact is parametrized over 8 cases).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/inheritance_probe.py tests/test_inheritance_probe.py
git commit -m "feat(phase0): phase classification and inherited-fraction row model"
```

---

### Task 2: The tracker

**Files:**
- Modify: `scripts/GPU/alphazero/inheritance_probe.py`
- Test: `tests/test_inheritance_probe.py`

**Interfaces:**
- Consumes: `SearchRow.build` from Task 1.
- Produces: `@dataclass InheritanceProbeConfig(enabled: bool = True)`; `class InheritanceProbeTracker` with `observe_search_start(ply: int, root, forced_sims_total: int) -> None`, `observe_search_end(forced_sims_total: int) -> None`, `observe_played_child(visits: Optional[int]) -> None`, `finalize_game() -> dict`. The tracker holds `rows: List[SearchRow]`.

**Three-call lifecycle per ply, strictly ordered:**

```text
observe_search_start(counter_before)   # immediately BEFORE search_from_root
        mcts.search_from_root(...)
observe_search_end(counter_after)      # immediately AFTER  search_from_root
observe_played_child(visits)           # immediately BEFORE advance_root
```

`forced_count` is `counter_after - counter_before` **on the same row**. A two-call
lifecycle that sampled the counter only at search start would attribute each ply's
forcing to the *following* ply and would report zero on row 0 unconditionally — the
counter is incremented by the very search the row describes.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_inheritance_probe.py
from scripts.GPU.alphazero.inheritance_probe import (
    InheritanceProbeConfig,
    InheritanceProbeTracker,
)


class _FakeChild:
    def __init__(self, visit_count):
        self.visit_count = visit_count


class _FakeRoot:
    """Minimal stand-in exposing only what the tracker reads."""

    def __init__(self, visit_count, child_visits):
        self.visit_count = visit_count
        self.children = {i: _FakeChild(v) for i, v in enumerate(child_visits)}


def _ply(tracker, ply, root, before, after, played):
    """One complete three-call lifecycle."""
    tracker.observe_search_start(ply=ply, root=root, forced_sims_total=before)
    tracker.observe_search_end(forced_sims_total=after)
    tracker.observe_played_child(visits=played)


def test_tracker_records_one_row_per_search():
    t = InheritanceProbeTracker(InheritanceProbeConfig())
    _ply(t, 0, _FakeRoot(0, []), 0, 0, 140)
    _ply(t, 1, _FakeRoot(140, [3, 0, 9]), 0, 0, None)
    assert len(t.rows) == 2
    assert t.rows[0].starting_visits == 0
    assert t.rows[0].played_child_visits == 140
    # Only children with a COMPLETED visit count as visited.
    assert t.rows[1].starting_visited_children == 2
    assert t.rows[1].played_child_visits is None


def test_forced_count_belongs_to_the_search_that_produced_it():
    """The counter is incremented BY the search the row describes, so the delta
    must be taken ACROSS that search. Sampling only at the next search start
    shifts every value one ply late and forces row 0 to zero."""
    t = InheritanceProbeTracker(InheritanceProbeConfig())
    _ply(t, 0, _FakeRoot(0, []), 0, 7, 1)     # ply 0's search forced 7
    _ply(t, 1, _FakeRoot(1, []), 7, 7, 1)     # ply 1 forced none
    _ply(t, 2, _FakeRoot(1, []), 7, 10, 1)    # ply 2 forced 3
    assert [r.forced_count for r in t.rows] == [7, 0, 3]


def test_counter_going_backwards_fails_loud():
    t = InheritanceProbeTracker(InheritanceProbeConfig())
    t.observe_search_start(ply=0, root=_FakeRoot(0, []), forced_sims_total=9)
    with pytest.raises(ValueError, match="went backwards"):
        t.observe_search_end(forced_sims_total=2)


def test_out_of_order_calls_fail_loud():
    t = InheritanceProbeTracker(InheritanceProbeConfig())
    with pytest.raises(RuntimeError):
        t.observe_search_end(forced_sims_total=0)   # no open row
    with pytest.raises(RuntimeError):
        t.observe_played_child(visits=1)            # no open row
    t.observe_search_start(ply=0, root=_FakeRoot(0, []), forced_sims_total=0)
    with pytest.raises(RuntimeError):
        t.observe_search_start(ply=1, root=_FakeRoot(0, []), forced_sims_total=0)
    with pytest.raises(RuntimeError):
        t.observe_played_child(visits=1)            # search_end not called yet


def test_disabled_tracker_records_nothing():
    t = InheritanceProbeTracker(InheritanceProbeConfig(enabled=False))
    _ply(t, 0, _FakeRoot(0, []), 0, 5, 5)
    assert t.rows == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_inheritance_probe.py -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'InheritanceProbeConfig'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/GPU/alphazero/inheritance_probe.py
from typing import Any, Dict, List


@dataclass
class InheritanceProbeConfig:
    """Opt-in switch, mirroring RecoveryRetargetingConfig in self_play.py."""

    enabled: bool = True


class InheritanceProbeTracker:
    """Records start-of-search telemetry for one game.

    Call contract, three calls per ply in strict order:
      1. `observe_search_start` immediately BEFORE `mcts.search_from_root`
      2. `observe_search_end`   immediately AFTER  `mcts.search_from_root`
      3. `observe_played_child` immediately BEFORE `mcts.advance_root`

    `forced_sims_total` is `MCTS._closeout_td1_forced_sims_total`, a monotonic
    per-instance counter. This class takes its delta ACROSS the search rather
    than requiring any `mcts.py` change: `forced_count` is a local inside
    `search_from_root`, resolved there after the forcing check, so it is
    start-of-search telemetry that cannot be read before the call returns.

    The delta must span the search. Sampling the counter only at the next
    search start attributes each ply's forcing to the following ply and makes
    row 0 unconditionally zero, because the counter is incremented by the very
    search the row describes.
    """

    def __init__(self, config: InheritanceProbeConfig) -> None:
        self.config = config
        self.rows: List[SearchRow] = []
        self._open_row: Optional[SearchRow] = None
        self._open_forced_before: Optional[int] = None
        self._awaiting_search_end: bool = False

    def observe_search_start(self, ply: int, root: Any, forced_sims_total: int) -> None:
        if not self.config.enabled:
            return
        if self._open_row is not None:
            raise RuntimeError(
                f"observe_search_start at ply {ply} with an unclosed row from "
                f"ply {self._open_row.ply}; observe_played_child was not called"
            )
        visited = sum(
            1 for c in root.children.values() if getattr(c, "visit_count", 0) > 0
        )
        row = SearchRow.build(
            ply=ply,
            starting_visits=root.visit_count,
            starting_visited_children=visited,
            forced_count=0,          # provisional; set in observe_search_end
        )
        self.rows.append(row)
        self._open_row = row
        self._open_forced_before = forced_sims_total
        self._awaiting_search_end = True

    def observe_search_end(self, forced_sims_total: int) -> None:
        if not self.config.enabled:
            return
        if self._open_row is None or not self._awaiting_search_end:
            raise RuntimeError("observe_search_end called with no open search row")
        delta = forced_sims_total - self._open_forced_before
        if delta < 0:
            raise ValueError(
                f"td1 forced-sims counter went backwards during ply "
                f"{self._open_row.ply} ({self._open_forced_before} -> "
                f"{forced_sims_total}); the telemetry counter was reset mid-search"
            )
        self._open_row.forced_count = delta
        self._awaiting_search_end = False

    def observe_played_child(self, visits: Optional[int]) -> None:
        """`visits` is the played child's visit_count read BEFORE advance_root,
        or None when the played move has no child node at all."""
        if not self.config.enabled:
            return
        if self._open_row is None:
            raise RuntimeError("observe_played_child called with no open search row")
        if self._awaiting_search_end:
            raise RuntimeError(
                f"observe_played_child at ply {self._open_row.ply} before "
                "observe_search_end; forced_count would be left provisional"
            )
        self._open_row.played_child_visits = visits
        self._open_row = None
        self._open_forced_before = None

    def finalize_game(self) -> Dict[str, Any]:
        if self._open_row is not None:
            raise RuntimeError(
                f"finalize_game with an unclosed row at ply {self._open_row.ply}"
            )
        return {"rows": [row.__dict__.copy() for row in self.rows]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_inheritance_probe.py -v -p no:cacheprovider`
Expected: PASS — 17 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/inheritance_probe.py tests/test_inheritance_probe.py
git commit -m "feat(phase0): inheritance tracker with fail-loud call-order and counter guards"
```

---

### Task 3: Summary and the frozen verdict evaluator

**Files:**
- Modify: `scripts/GPU/alphazero/inheritance_probe.py`
- Test: `tests/test_inheritance_probe.py`

**Interfaces:**
- Consumes: `SearchRow`, `InheritanceProbeTracker.finalize_game` from Tasks 1–2.
- Produces: `summarize(rows: List[SearchRow]) -> dict` and `evaluate_verdict(summary: dict) -> dict`. The verdict dict has keys `verdict` (`"WARM_START_REQUIRED"` or `"FRESH_ROOT_ACCEPTABLE"`), `reasons: List[str]`, `coverage_complete: bool`, `unobserved_post_opening_phases: List[str]`.

`statistics.quantiles` needs at least two data points, so p75 is `None` for fewer than two observations. A phase with no rows has median `None`. Neither is `0.0`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_inheritance_probe.py
from scripts.GPU.alphazero.inheritance_probe import (
    OVERALL_P75_LIMIT,
    POST_OPENING_MEDIAN_LIMIT,
    evaluate_verdict,
    summarize,
)


def _rows(*specs):
    return [
        SearchRow.build(ply=p, starting_visits=v, starting_visited_children=0, forced_count=0)
        for p, v in specs
    ]


def test_frozen_thresholds_are_pinned():
    # These encode the design's frozen decision rule. If this test fails,
    # the rule was changed -- that requires a spec amendment, not a test edit.
    assert POST_OPENING_MEDIAN_LIMIT == 0.10
    assert OVERALL_P75_LIMIT == 0.20
    assert DECISION_POINT_SIMS == 320


def test_absent_phase_median_is_none_not_zero():
    s = summarize(_rows((0, 0), (5, 0)))
    assert s["by_phase"]["opening"]["median"] == 0.0     # observed, genuinely zero
    assert s["by_phase"]["late"]["median"] is None       # unobserved
    assert s["by_phase"]["late"]["n"] == 0


def test_p75_is_none_with_fewer_than_two_observations():
    assert summarize(_rows((0, 100)))["overall"]["p75"] is None
    assert summarize(_rows((0, 100), (1, 100)))["overall"]["p75"] is not None


def test_warm_start_stands_on_partial_coverage():
    """Amendment 1: a crossing is valid evidence even when coverage is partial
    -- a phase that fired had enough evidence to fire.
    40 inherited visits -> 40/360 = 0.111 >= 0.10, in midgame."""
    v = evaluate_verdict(summarize(_rows((61, 40), (62, 40), (63, 40))))
    assert v["verdict"] == "WARM_START_REQUIRED"
    assert v["coverage_complete"] is False
    assert any("midgame" in r for r in v["reasons"])


def test_fresh_root_requires_complete_coverage():
    """The negative conclusion needs every post-opening phase observed."""
    v = evaluate_verdict(summarize(_rows((0, 0), (31, 0), (61, 0), (91, 0))))
    assert v["verdict"] == "FRESH_ROOT_ACCEPTABLE"
    assert v["coverage_complete"] is True
    assert v["unobserved_post_opening_phases"] == []


def test_no_crossing_with_partial_coverage_is_incomplete():
    """Opening is excluded from the median branch, and p75 of a constant 0.111
    series is 0.111 < 0.20 -- so nothing fires. With every post-opening phase
    unobserved this must NOT read as fresh-root acceptability."""
    v = evaluate_verdict(summarize(_rows((0, 40), (1, 40), (2, 40))))
    assert v["verdict"] == "PREFLIGHT_INCOMPLETE"
    assert v["coverage_complete"] is False
    assert set(v["unobserved_post_opening_phases"]) == {"early_mid", "midgame", "late"}


def test_overall_p75_branch_fires_independently():
    # All opening, so the median branch cannot fire; p75 must carry it.
    # 160 -> 0.333, well above 0.20.
    v = evaluate_verdict(summarize(_rows((0, 0), (1, 160), (2, 160), (3, 160))))
    assert v["verdict"] == "WARM_START_REQUIRED"
    assert any("p75" in r for r in v["reasons"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_inheritance_probe.py -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'summarize'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/GPU/alphazero/inheritance_probe.py
import statistics


def _median_or_none(values: List[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _p75_or_none(values: List[float]) -> Optional[float]:
    # statistics.quantiles requires n >= 2. Fewer observations means the
    # statistic is undefined -- None, never 0.0.
    if len(values) < 2:
        return None
    return statistics.quantiles(values, n=4, method="inclusive")[2]


def summarize(rows: List[SearchRow]) -> Dict[str, Any]:
    """Per-phase and overall inherited-fraction summary."""
    all_vals = [r.inherited_fraction_320 for r in rows]
    by_phase: Dict[str, Any] = {}
    for name, _lo, _hi in PHASE_BOUNDS:
        vals = [r.inherited_fraction_320 for r in rows if r.phase == name]
        by_phase[name] = {
            "n": len(vals),
            "median": _median_or_none(vals),
            "p75": _p75_or_none(vals),
        }
    return {
        "n_searches": len(rows),
        "overall": {
            "n": len(all_vals),
            "median": _median_or_none(all_vals),
            "p75": _p75_or_none(all_vals),
        },
        "by_phase": by_phase,
        "forced_sims_total": sum(r.forced_count for r in rows),
    }


def evaluate_verdict(summary: Dict[str, Any]) -> Dict[str, Any]:
    """The FROZEN design section 2 decision rule, as amended 2026-08-03.

    - Median inherited_fraction_320 >= 0.10 in ANY post-opening phase, OR
      overall p75 >= 0.20                       -> WARM_START_REQUIRED
    - Nothing crossed AND coverage complete     -> FRESH_ROOT_ACCEPTABLE
    - Nothing crossed AND coverage incomplete   -> PREFLIGHT_INCOMPLETE

    The asymmetry is deliberate. A crossing is valid evidence on partial
    coverage: the phase that fired had enough evidence to fire. The negative
    conclusion is not: concluding fresh-root probing is safe requires having
    looked at every post-opening phase.

    PREFLIGHT_INCOMPLETE resolves nothing and MUST NOT trigger another game --
    that is the top-up pattern the protocol forbids. It requires a written
    protocol revision.
    """
    reasons: List[str] = []
    unobserved: List[str] = []

    for phase in POST_OPENING_PHASES:
        median = summary["by_phase"][phase]["median"]
        if median is None:
            unobserved.append(phase)
            continue
        if median >= POST_OPENING_MEDIAN_LIMIT:
            reasons.append(
                f"post-opening phase {phase} median {median:.6f} "
                f">= {POST_OPENING_MEDIAN_LIMIT}"
            )

    overall_p75 = summary["overall"]["p75"]
    if overall_p75 is not None and overall_p75 >= OVERALL_P75_LIMIT:
        reasons.append(f"overall p75 {overall_p75:.6f} >= {OVERALL_P75_LIMIT}")

    if reasons:
        verdict = "WARM_START_REQUIRED"
    elif unobserved:
        verdict = "PREFLIGHT_INCOMPLETE"
    else:
        verdict = "FRESH_ROOT_ACCEPTABLE"

    return {
        "verdict": verdict,
        "reasons": reasons,
        "coverage_complete": not unobserved,
        "unobserved_post_opening_phases": unobserved,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_inheritance_probe.py -v -p no:cacheprovider`
Expected: PASS — 24 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/inheritance_probe.py tests/test_inheritance_probe.py
git commit -m "feat(phase0): summary plus the frozen verdict evaluator, thresholds pinned"
```

---

### Task 4: Wire the probe into `play_game`

**Files:**
- Modify: `scripts/GPU/alphazero/self_play.py` (eight edits: the import at the top, the `GameRecord` field at line 449, and six inside `play_game`, which begins at line 579)
- Test: `tests/test_inheritance_probe_identity.py`

**Interfaces:**
- Consumes: `InheritanceProbeConfig`, `InheritanceProbeTracker` from Task 2.
- Produces: `play_game(..., inheritance_probe_config: Optional[InheritanceProbeConfig] = None)`, and `GameRecord` carrying the probe payload under the existing record mechanism used by `recovery_retargeting`.

Follow the `RecoveryRetargetingConfig` pattern already present: parameter at ~line 635, tracker construction at ~695, observation calls in the loop, finalize at ~1342.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inheritance_probe_identity.py
"""Probe-off must be byte-identical; probe-on must produce real rows."""
import random

import pytest

from scripts.GPU.alphazero.inheritance_probe import InheritanceProbeConfig
from scripts.GPU.alphazero.mcts import MCTSConfig
from scripts.GPU.alphazero.self_play import play_game

# Existing GPU-free deterministic fake: uniform priors, fixed value, no
# checkpoint required. Do NOT write a new evaluator helper.
from tests.eval_fakes import FakeEvaluator


def _shipped_batching_config(n_simulations: int = 64) -> MCTSConfig:
    # Shipped batching is (14, 48, 8). stall_flush_sims defaults to 16,
    # so 48 MUST be set explicitly or the batched path is not the shipped one.
    return MCTSConfig(
        n_simulations=n_simulations,
        eval_batch_size=14,
        stall_flush_sims=48,
        pending_virtual_visits=8,
    )


def _play(probe_config, seed: int = 20260803):
    return play_game(
        evaluator=FakeEvaluator(value=0.0),
        mcts_config=_shipped_batching_config(),
        rng=random.Random(seed),
        max_moves=8,
        add_noise=False,
        active_size=24,
        game_id=1,
        inheritance_probe_config=probe_config,
    )


def test_probe_on_records_one_row_per_ply():
    rec = _play(InheritanceProbeConfig())
    rows = rec.inheritance_probe_record["rows"]
    assert len(rows) >= 2
    assert [r["ply"] for r in rows] == list(range(len(rows)))
    # First search of a game starts from a fresh root.
    assert rows[0]["starting_visits"] == 0
    assert rows[0]["inherited_fraction_320"] == 0.0
    # Every closed row observed its played child.
    assert all("played_child_visits" in r for r in rows)


def test_probe_off_leaves_the_record_untouched():
    assert _play(None).inheritance_probe_record is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_inheritance_probe_identity.py -v -p no:cacheprovider`
Expected: FAIL — `TypeError: play_game() got an unexpected keyword argument 'inheritance_probe_config'`

- [ ] **Step 3: Write minimal implementation**

Edit 1 — import, beside the other alphazero imports at the top of `self_play.py`:

```python
from .inheritance_probe import InheritanceProbeConfig, InheritanceProbeTracker
```

Edit 2 — parameter, immediately after `recovery_retargeting_config` (~line 635):

```python
    # Phase 0 inheritance preflight (convergence atlas design section 2).
    # None (default) = probe absent, play_game behaviour byte-identical.
    inheritance_probe_config: Optional[InheritanceProbeConfig] = None,
```

Edit 3 — tracker construction, beside the `recovery_tracker` block (~line 695):

```python
    inheritance_tracker = None
    if inheritance_probe_config is not None and inheritance_probe_config.enabled:
        inheritance_tracker = InheritanceProbeTracker(inheritance_probe_config)
```

Edit 4 — observation, immediately before the `mcts.search_from_root` call (~line 854), which currently reads `# Run MCTS search from current root (reuses subtree)`:

```python
        if inheritance_tracker is not None:
            inheritance_tracker.observe_search_start(
                ply=ply,
                root=root,
                forced_sims_total=mcts._closeout_td1_forced_sims_total,
            )
```

Edit 4b — observation, immediately **after** the `mcts.search_from_root(...)` call
returns (~line 856, before the `# Build opening diagnostic record` block). The counter
is incremented by this search, so the delta must span it:

```python
        if inheritance_tracker is not None:
            inheritance_tracker.observe_search_end(
                forced_sims_total=mcts._closeout_td1_forced_sims_total,
            )
```

Edit 5 — observation, immediately before `root = mcts.advance_root(root, move)` (~line 1146, under the `# TREE REUSE:` comment). `encode_move` is already imported at `self_play.py:24`; add no import:

```python
        if inheritance_tracker is not None:
            _played_child = root.children.get(encode_move(move[0], move[1]))
            inheritance_tracker.observe_played_child(
                visits=None if _played_child is None else _played_child.visit_count
            )
```

Edit 6 — add the field to the `GameRecord` dataclass immediately after
`recovery_retargeting_record: Optional[dict] = None` (line 449):

```python
    inheritance_probe_record: Optional[dict] = None
```

Edit 7 — finalize, beside the `recovery_retargeting_record = (` assignment (line 1341),
and pass it into the `GameRecord(...)` construction exactly as
`recovery_retargeting_record` is passed:

```python
    inheritance_probe_record = (
        inheritance_tracker.finalize_game()
        if inheritance_tracker is not None else None
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_inheritance_probe_identity.py -v -p no:cacheprovider`
Expected: PASS — 2 passed.

- [ ] **Step 5: Prove byte-identical off — the whole pre-existing suite**

Run: `.venv/bin/python -m pytest -p no:cacheprovider -q`
Expected: every pre-existing test still passes, zero failures. A single pre-existing failure means the default path was changed and must be fixed before proceeding — do not continue to Task 5.

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/self_play.py tests/test_inheritance_probe_identity.py
git commit -m "feat(phase0): opt-in inheritance probe in play_game, default None is a no-op"
```

---

### Task 5: Batched observer-on/off identity qualification

**Files:**
- Modify: `tests/test_inheritance_probe_identity.py`

**Interfaces:**
- Consumes: `play_game(..., inheritance_probe_config=...)` from Task 4.
- Produces: nothing consumed downstream; this is the qualification gate.

Design §9 requires exact equality on the **batched** path with `(14, 48, 8)` and `add_noise=false`. Synchronous CPU tests cannot substitute, because they never exercise pending leaves or virtual visits.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_inheritance_probe_identity.py


def _identity_fields(rec):
    """Everything design section 9 requires to match, probe on vs off.

    Uses only verified GameRecord/PositionRecord fields. Note PositionRecord
    has NO `search_score` attribute -- root value is passed to trackers, not
    stored on the record -- and `visit_counts` is a List[int] parallel to
    `legal_moves`, not a dict. Per-ply visit counts ARE the search output, so
    equality across every ply is a strong search-identity proof.
    """
    return {
        "move_history": list(rec.move_history),
        "winner": rec.winner,
        "n_moves": rec.n_moves,
        "draw_reason": rec.draw_reason,
        # Per-move root summaries -- these ARE the search's value output, and
        # they already exist on GameRecord, so no new instrumentation is needed
        # to meet the design's root-value identity requirement.
        "move_root_values": list(rec.move_root_values),
        "move_top1_shares": list(rec.move_top1_shares),
        "final_root_value": rec.final_root_value,
        "final_top1_share": rec.final_top1_share,
        "positions": [
            (p.ply, p.to_move, list(p.legal_moves), list(p.visit_counts))
            for p in rec.positions
        ],
    }


def test_probe_does_not_perturb_batched_search():
    off = _play(None)
    on = _play(InheritanceProbeConfig())
    assert _identity_fields(on) == _identity_fields(off)


def test_identity_check_is_not_vacuous():
    """A different seed must produce a different game, or the comparison above
    proves nothing."""
    assert _identity_fields(_play(None)) != _identity_fields(_play(None, seed=999999))


def test_batched_path_was_actually_exercised():
    """Configuration alone proves nothing -- the run must actually reach a
    full batch flush, in BOTH observer states, or the identity comparison
    says nothing about the batched path."""
    off = _play(None)
    on = _play(InheritanceProbeConfig())
    assert off.flush_full > 0, (
        "no batch-full flush occurred; the fixture never exercised the batched "
        "path. Strengthen the fixture (more sims or more plies) -- do NOT accept "
        "this qualification."
    )
    assert on.flush_full > 0
    assert (on.flush_full, on.flush_stall, on.flush_tail) == (
        off.flush_full, off.flush_stall, off.flush_tail
    )
```

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_inheritance_probe_identity.py -v -p no:cacheprovider`
Expected: PASS — 5 passed. All field names above are verified against the real
dataclasses, so these should pass on the Task 4 implementation without change.

If `test_probe_does_not_perturb_batched_search` fails, the probe is perturbing search
— almost certainly by mutating a node or advancing an RNG. Fix the probe. **Never**
weaken `_identity_fields` to make it pass.

If `test_identity_check_is_not_vacuous` fails, the fixture is too weak to distinguish
games — `FakeEvaluator` gives uniform priors, so with only 8 plies two seeds may
coincide. Raise `max_moves` until the two games differ, then re-run the whole file.

- [ ] **Step 3: Commit**

```bash
git add tests/test_inheritance_probe_identity.py
git commit -m "test(phase0): batched probe-on/off identity qualification, non-vacuous"
```

---

### Task 6: CLI runner and the producer-to-consumer integration test

**Files:**
- Create: `scripts/GPU/alphazero/run_inheritance_preflight.py`
- Modify: `tests/test_inheritance_probe_identity.py`

**Interfaces:**
- Consumes: `play_game(..., inheritance_probe_config=...)`, `summarize`, `evaluate_verdict`.
- Produces: CLI writing a JSON artifact `{"rows": [...], "summary": {...}, "verdict": {...}, "provenance": {...}}`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_inheritance_probe_identity.py
from scripts.GPU.alphazero.inheritance_probe import evaluate_verdict, summarize
from scripts.GPU.alphazero.inheritance_probe import SearchRow


def test_real_producer_feeds_the_real_consumer():
    """The v18 lesson: drive the real producer into the real consumer at least
    once. Hand-written surrogates hid four contract defects."""
    rec = _play(InheritanceProbeConfig())
    rows = [SearchRow(**r) for r in rec.inheritance_probe_record["rows"]]
    summary = summarize(rows)
    verdict = evaluate_verdict(summary)

    assert summary["n_searches"] == len(rows)
    assert verdict["verdict"] in {"WARM_START_REQUIRED", "FRESH_ROOT_ACCEPTABLE"}
    # An 8-ply game observes opening only, so post-opening coverage is absent
    # and must be reported rather than silently read as zero.
    assert verdict["coverage_complete"] is False
    assert summary["by_phase"]["late"]["median"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_inheritance_probe_identity.py::test_real_producer_feeds_the_real_consumer -v -p no:cacheprovider`
Expected: FAIL — `SearchRow(**r)` raises `TypeError` if `finalize_game`'s dict keys do not exactly match the dataclass fields. That mismatch is the contract defect this test exists to catch; fix the producer, not the test.

- [ ] **Step 3: Write the CLI**

```python
# scripts/GPU/alphazero/run_inheritance_preflight.py
"""Phase 0 inheritance preflight runner -- convergence atlas design section 2.

Plays exactly ONE unchanged shipped self-play game with tree reuse, records
start-of-search telemetry, and evaluates the frozen decision rule.

TECHNICAL PREFLIGHT, NOT EVIDENCE. The observed game must be excluded from the
atlas corpus and must never be used to tune labels, features or gates.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

from .inheritance_probe import (
    InheritanceProbeConfig,
    SearchRow,
    evaluate_verdict,
    summarize,
)
from .mcts import MCTSConfig
from .self_play import play_game


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _worktree_clean() -> bool:
    return subprocess.check_output(["git", "status", "--porcelain"], text=True) == ""


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 0 inheritance preflight (one game)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--max-moves", type=int, default=280)
    p.add_argument("--active-size", type=int, default=24)
    p.add_argument("--out", required=True, help="output JSON path (required)")
    args = p.parse_args()

    # Established checkpoint-loading path: auto-detects 24/30-channel, wraps in
    # LocalGPUEvaluator with the project's eval compile setting. `local_evaluator`
    # exports LocalGPUEvaluator(net, compile=...) -- it takes a loaded network,
    # NOT a path -- so do not construct it directly. Imported lazily: this keeps
    # MLX out of module scope so the tests stay GPU-free.
    from .eval_runner import _default_evaluator_factory

    cfg = MCTSConfig(
        n_simulations=args.sims,
        eval_batch_size=14,
        stall_flush_sims=48,
        pending_virtual_visits=8,
    )
    record = play_game(
        evaluator=_default_evaluator_factory(args.checkpoint),
        mcts_config=cfg,
        rng=random.Random(args.seed),
        max_moves=args.max_moves,
        add_noise=False,
        active_size=args.active_size,
        game_id=args.seed,
        inheritance_probe_config=InheritanceProbeConfig(),
    )

    raw = record.inheritance_probe_record["rows"]
    rows = [SearchRow(**r) for r in raw]
    summary = summarize(rows)
    verdict = evaluate_verdict(summary)

    artifact = {
        "rows": raw,
        "summary": summary,
        "verdict": verdict,
        "provenance": {
            "git_head": _git_head(),
            "worktree_clean": _worktree_clean(),
            "checkpoint": args.checkpoint,
            "seed": args.seed,
            "n_simulations": args.sims,
            "batching": [cfg.eval_batch_size, cfg.stall_flush_sims,
                         cfg.pending_virtual_visits],
            "add_noise": False,
            "note": "TECHNICAL PREFLIGHT, NOT EVIDENCE. Exclude this game from "
                    "the atlas corpus.",
        },
    }
    Path(args.out).write_text(json.dumps(artifact, indent=2, sort_keys=True))

    print(f"verdict: {verdict['verdict']}")
    for reason in verdict["reasons"]:
        print(f"  reason: {reason}")
    if not verdict["coverage_complete"]:
        print(
            "  COVERAGE INCOMPLETE -- unobserved post-opening phases: "
            + ", ".join(verdict["unobserved_post_opening_phases"])
            + " (an unobserved phase supplies no evidence either way)"
        )
    print(f"artifact: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`--out` is deliberately **required with no default**. The v18 postmortem records a script whose default output path was a frozen evidence artifact, so the documented bare command would have destroyed it.

> **Flag before the operator run — unresolved contradiction about `compile=True`.**
> `_default_evaluator_factory` constructs `LocalGPUEvaluator(net, compile=True)`, and
> its docstring says compile is there *to prevent* Metal resource exhaustion during
> long sequential eval runs. A recorded project gotcha says the opposite: *"MLX
> compile=True breaks sequential eval."* Phase 0 is a single in-process sequential
> game, which is exactly the contested regime.
>
> Use the factory as written — it is the established path and the reviewer directed
> it. But if the run hangs, produces garbage values, or exhausts Metal resources, the
> first thing to try is a `compile=False` evaluator, and the contradiction should be
> resolved and recorded rather than worked around silently. Do not resolve it by
> editing `eval_runner`.

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -p no:cacheprovider -q`
Expected: all tests pass, including the pre-existing suite unchanged.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/run_inheritance_preflight.py tests/test_inheritance_probe_identity.py
git commit -m "feat(phase0): preflight CLI plus real-producer-to-real-consumer integration test"
```

---

## The run command (operator, NOT authorized by this plan)

This plan builds and qualifies the instrument. Running it is a separate authorization.

```bash
nohup .venv/bin/python -m scripts.GPU.alphazero.run_inheritance_preflight \
  --checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --seed <frozen> \
  --sims 400 \
  --out logs/eval/phase0_inheritance/preflight.json \
  > logs/eval/phase0_inheritance/run.log 2>&1 &
disown
```

Operator rules that apply:

- **Launch and wait in separate tool calls.** `nohup` + `disown` is not sufficient when the launch and the wait share one call — a tool timeout SIGTERMs the whole process group. `setsid` does not exist on macOS.
- Do not commit between generation and any later qualification step.
- One game only. Do not add games to improve coverage after seeing the verdict.

## Out of scope

No atlas producer, corpus, ladder, detector, gate calibration, widening tracer, `mcts.py` change, or strength work. Those wait on this verdict and on the four remaining pre-freeze open items in the spec's freeze table.
