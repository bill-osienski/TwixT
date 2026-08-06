# Amendment 5 — Trajectory-Relative Phases: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the corpus phase definition with `phase_index = min(3, (4*ply)//n_moves)`
and thread `n_moves` to every consumer. **Nothing else changes.**

**Governing amendment:** `2026-08-03-convergence-atlas-design.md` §3, Amendment 5, written
2026-08-05 **before** this plan. No threshold, pilot size, sizing formula, gate or
read-out is touched.

## Global Constraints

- **No new threshold, predicate or protocol rule.** The formula is the amendment's,
  verbatim.
- **No reservoir generation, no checkpoint loading, no MLX, no measurement run.** Every
  test is synthetic.
- **No `mcts.py` change.**
- **`inheritance_probe.phase_for_ply` is NOT touched** — Phase 0 ran under absolute
  bounds and its medians are absolute-phase facts. A test pins that it stays absolute.
- **Baseline against a MEASURED collect** before starting, never a quoted number.
- Commit after every task.

## The change, in one line

```python
# was:  phase_for_ply(ply)                -> absolute bounds 0-30 / 31-60 / 61-90 / 91+
# now:  phase_for_ply(ply, n_moves)       -> PHASE_NAMES[min(3, (4*ply)//n_moves)]
```

`n_moves` is a required positional argument, **not** an optional one with a default. A
default would let a stale caller silently keep absolute-like behaviour; the whole point
is that every call site must now supply the trajectory.

### Call sites — all of them

| site | has `n_moves`? | change |
|---|---|---|
| `corpus_geometry.eligible_plies(meta, …)` | yes, `meta.n_moves` | pass it |
| `corpus_geometry.eligible_cells(meta)` | yes | pass it |
| `atlas_row_facts.derive_row_facts(...)` | **no** | gains an `n_moves` parameter |
| `atlas_run.run_row` | yes, `meta.n_moves` | pass it through |
| `inheritance_probe.phase_for_ply` | — | **untouched, deliberately** |
| `tests/test_corpus_geometry.py`, `tests/test_atlas_run.py` | — | updated |

---

### Task 1: The phase function and the corpus geometry

**Files:**
- Modify: `scripts/GPU/alphazero/corpus_geometry.py`
- Test: `tests/test_corpus_geometry.py`

**Interfaces:**
- Produces: `PHASE_NAMES`; `phase_for_ply(ply, n_moves) -> str`. `_PHASE_BOUNDS` is
  **deleted** — leaving it would be a second, contradictory definition of the same word.
- `eligible_plies` / `eligible_cells` keep their signatures; only their internals change.

- [ ] **Step 1: Write the failing test**

```python
# replaces the absolute-bounds cases in tests/test_corpus_geometry.py
import pytest
from scripts.GPU.alphazero.corpus_geometry import (
    PHASES, GameMeta, eligible_cells, eligible_plies, phase_for_ply,
)


@pytest.mark.parametrize("ply,expected", [
    (0, "opening"), (9, "opening"),
    (10, "early_mid"), (19, "early_mid"),
    (20, "midgame"), (29, "midgame"),
    (30, "late"), (39, "late"),
])
def test_quarters_of_a_40_move_game(ply, expected):
    """(4p)//40 == p//10, so the quarters are exactly 10 plies each."""
    assert phase_for_ply(ply, 40) == expected


def test_the_SAME_ply_lands_in_different_phases_in_different_games():
    """The whole point of the amendment: phase is trajectory-relative, so an
    absolute ply carries no phase on its own."""
    assert phase_for_ply(30, 40) == "late"
    assert phase_for_ply(30, 120) == "opening"


def test_the_final_ply_is_always_late_and_the_first_always_opening():
    for n in (8, 39, 57, 76, 280):
        assert phase_for_ply(0, n) == "opening"
        assert phase_for_ply(n - 1, n) == "late"


def test_the_min_clamp_holds_at_and_past_the_end():
    """`min(3, ...)` is the amendment's own clamp: a ply at or past n_moves is
    in the final quarter, not an index error."""
    assert phase_for_ply(40, 40) == "late"
    assert phase_for_ply(99, 40) == "late"


def test_n_moves_is_REQUIRED_not_defaulted():
    """A default would let a stale call site silently keep the old behaviour,
    which is exactly what this amendment must not permit."""
    with pytest.raises(TypeError):
        phase_for_ply(5)


def test_guards_are_kept():
    with pytest.raises(ValueError, match="non-negative"):
        phase_for_ply(-1, 40)
    with pytest.raises(ValueError, match="n_moves"):
        phase_for_ply(0, 0)


def test_every_game_of_at_least_eight_moves_serves_ALL_EIGHT_cells():
    """The geometry failure this amendment exists to fix. Each quarter of a
    game with n_moves >= 8 holds at least two plies, hence both sides."""
    for n in (8, 39, 57, 76):
        meta = GameMeta(game_id=0, seed=1, n_moves=n, start_player="red")
        assert len(eligible_cells(meta)) == 8


def test_the_retired_pilots_lengths_would_now_all_serve_late():
    """The 24 retired games ran 39-76 plies and produced ZERO late capacity
    under absolute bounds. Retained as a regression fixture ONLY -- these are
    lengths, not positions, and no retired position enters any corpus."""
    for n in (39, 46, 48, 55, 57, 62, 70, 76):
        meta = GameMeta(game_id=0, seed=1, n_moves=n, start_player="red")
        assert eligible_plies(meta, "late", "red")
        assert eligible_plies(meta, "late", "black")


def test_quarters_partition_the_whole_trajectory():
    """No ply is unassigned and none is double-assigned."""
    for n in (8, 39, 57, 76):
        meta = GameMeta(game_id=0, seed=1, n_moves=n, start_player="red")
        seen = [p for ph in PHASES for s in ("red", "black")
                for p in eligible_plies(meta, ph, s)]
        assert sorted(seen) == list(range(n))
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_corpus_geometry.py -v -p no:cacheprovider`
Expected: FAIL — the old absolute-bounds parametrization, and `TypeError` on the new
two-argument calls.

- [ ] **Step 3: Implement**

```python
# Amendment 5 (2026-08-05): phase is the quarter of the game's REALIZED
# trajectory, not an absolute ply band. Absolute bounds made the late cells
# unfillable -- zero of 24 pilot games reached ply 91 -- and a lower fitted
# cutoff would tune to those very lengths.
PHASE_NAMES: Tuple[str, ...] = ("opening", "early_mid", "midgame", "late")


def phase_for_ply(ply: int, n_moves: int) -> str:
    """`min(3, (4 * ply) // n_moves)` -- amendment 5, verbatim.

    `n_moves` is REQUIRED. A default would let a stale call site keep
    absolute-like behaviour silently, which is the one failure this change
    cannot tolerate.

    A ply at or past `n_moves` is in the final quarter by the amendment's own
    `min(3, ...)` clamp, not an error.
    """
    if ply < 0:
        raise ValueError(f"ply must be non-negative, got {ply}")
    if n_moves <= 0:
        raise ValueError(f"n_moves must be positive, got {n_moves}")
    return PHASE_NAMES[min(3, (4 * ply) // n_moves)]
```

`_PHASE_BOUNDS` is deleted. `eligible_plies` and `eligible_cells` pass `meta.n_moves`.
`PHASES` keeps its existing value and order; `PHASE_NAMES` is the index-ordered tuple the
formula selects from, and a test asserts the two agree.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_corpus_geometry.py tests/test_inheritance_probe.py -v -p no:cacheprovider`
Expected: PASS — and Phase 0's absolute probe still green, untouched.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/corpus_geometry.py tests/test_corpus_geometry.py
git commit -m "feat(atlas-a5): trajectory-relative phases in the corpus geometry"
```

---

### Task 2: Thread `n_moves` to the row facts

**Files:**
- Modify: `scripts/GPU/alphazero/atlas_row_facts.py`, `scripts/GPU/alphazero/atlas_run.py`
- Test: `tests/test_atlas_row_facts.py`, `tests/test_atlas_run.py`

**Interfaces:**
- `derive_row_facts(legs, snapshots, target_ply, n_moves, start_player, assigned_phase=None, assigned_side=None)`.
  `n_moves` sits immediately after `target_ply`, because the two are meaningless apart.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atlas_row_facts.py
def test_phase_is_trajectory_relative_and_cross_checks_against_the_assignment():
    """Ply 30 is `late` in a 40-move game and `opening` in a 120-move one, so
    the cross-check now has to agree with the game, not just the ply."""
    f = derive_row_facts(_legs(), _snaps(), 30, 40, "red")
    assert f["phase"] == "late"
    f = derive_row_facts(_legs(), _snaps(), 30, 120, "red")
    assert f["phase"] == "opening"
    with pytest.raises(ValueError, match="phase"):
        derive_row_facts(_legs(), _snaps(), 30, 40, "red",
                         assigned_phase="opening")


def test_n_moves_is_required_here_too():
    with pytest.raises(TypeError):
        derive_row_facts(_legs(), _snaps(), 30, start_player="red")
```

`tests/test_atlas_run.py`'s `_assigned(...)` helper gains `n_moves` and passes it to
`phase_for_ply`; `verify_pilot`'s re-derivation already reads `pilot_games`, so it looks
up `by_id[row["game_idx"]].n_moves` alongside `.start_player`.

- [ ] **Step 2: Run to verify it fails** — `TypeError` on the new argument.

- [ ] **Step 3: Implement.** `run_row` passes `meta.n_moves`; `verify_pilot` passes the
      block game's `n_moves`. No other logic changes.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_row_facts.py tests/test_atlas_run.py -v -p no:cacheprovider`

- [ ] **Step 5: Commit**

---

### Task 3: Pin the Phase 0 exclusion, then qualify

**Files:**
- Test: `tests/test_inheritance_probe.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_phase0_keeps_ABSOLUTE_bounds_and_is_not_amended():
    """Phase 0 ran under absolute bounds and returned WARM_START_REQUIRED; its
    recorded medians (opening 0.160105 n=31, early_mid 0.254947 n=20) are facts
    about ABSOLUTE phases. Re-labelling them under a definition adopted
    afterwards would rewrite a completed, frozen measurement.

    The two functions are intentionally different. This test is the seam that
    says so out loud.
    """
    import inspect
    from scripts.GPU.alphazero import corpus_geometry, inheritance_probe
    assert phase_for_ply(95) == "late"          # absolute: one argument
    assert phase_for_ply(30) == "opening"
    # ...while the corpus function requires the trajectory and disagrees.
    assert corpus_geometry.phase_for_ply(30, 40) == "late"
    assert len(inspect.signature(inheritance_probe.phase_for_ply).parameters) == 1
    assert len(inspect.signature(corpus_geometry.phase_for_ply).parameters) == 2
```

- [ ] **Step 2–4: Run, implement (nothing to implement), then the full suite**

```bash
.venv/bin/python -m pytest -p no:cacheprovider -q > /tmp/a5.out 2>&1; echo "REAL_EXIT=$?" >> /tmp/a5.out; tail -3 /tmp/a5.out
```

- [ ] **Step 5: Commit**

---

## Completion criteria

- [ ] `phase_for_ply(ply, n_moves)` implements `min(3, (4*ply)//n_moves)` verbatim, with
      `n_moves` **required**; `_PHASE_BOUNDS` is deleted from `corpus_geometry`.
- [ ] Every game with `n_moves ≥ 8` serves all eight phase×side cells, and the quarters
      partition the trajectory exactly — no ply unassigned, none double-assigned.
- [ ] The retired pilot's **lengths** (39–76) all yield late capacity on both sides,
      retained as a regression fixture only. **No retired position enters any corpus.**
- [ ] `derive_row_facts` takes `n_moves`; `run_row` and `verify_pilot` supply it.
- [ ] **`inheritance_probe.phase_for_ply` is unchanged and absolute**, pinned by a test
      that asserts the two functions differ in both arity and answer.
- [ ] No threshold, pilot size, sizing formula, gate or read-out changed. `mcts.py` diff
      empty. No MLX, no reservoir, no checkpoint.
- [ ] **Recount** `def test_` on disk and baseline against a **measured** collect. The
      delta must equal the recount; anything else means a pre-existing test changed
      behaviour and must be explained. Read the exit code from the process, never a pipe.

## Out of scope

No new pilot, no generation, no authorization. Amendment 5 names a proposed successor
range — `[20321000, 20321480)`, sampling seed `20260806` — but **freezing it requires a
new written authorization** issued after this plan is implemented and qualified. The
existing pilot authorization is spent: its block is retired and its geometry gate
returned a no-go.
