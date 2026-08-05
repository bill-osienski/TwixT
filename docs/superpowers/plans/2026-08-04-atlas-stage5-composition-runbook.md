# Atlas Stage 5 — Row Facts, Composition and the Operator Runbook

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the last unqualified seam (`phase` / `flat_policy` / `near_even` are
supplied facts), compose the delivered parts into one end-to-end run, and hand the
operator a single launchable protocol with explicit stop conditions and exit-status
sidecars. **Stage 5 finishes the tooling. It does not launch it.**

**Architecture:** Three files over Stage 4's interfaces. `atlas_row_facts.py` derives the
three row facts from frozen measured fields and cross-checks them against the assignment.
`atlas_run.py` is pure orchestration over an **injected** evaluator: assigned row → warm
replay → additive ladder → features → `build_row` → all three read-outs → one run
document. `run_atlas.py` is the operator CLI: a zero-GPU `preflight`, an `emit-runbook`,
and the single `run` entry point Stage 5 writes and never executes.

**Tech Stack:** Python 3, stdlib only. Tests: `.venv/bin/python -m pytest -p no:cacheprovider`.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-08-03-convergence-atlas-design.md` §2b, §3, §4,
  §5, §6, §6a, §7, §8, §9. **§3–§12 are EXECUTION-FROZEN.**
- **No new threshold, protocol change or predicate.** Every number Stage 5 uses is already
  frozen and is cited to its section. Stage 5 introduces **zero** new constants.
- **No reservoir generation, no checkpoint loading, no MLX execution, no pilot, no
  measurement run.** Every test uses synthetic input or `FakeEvaluator` at
  `active_size=6`.
- **No `mcts.py` change.** Stage 1's scoped exception already delivered every hook.
- **Undefined statistics are `None`, never `0`, never `false`** — including the row facts.
- **Baseline against a MEASURED collect.** Stage 4 predicted 2554 and measured 2556; the
  gap was a stale baseline in a document, not a defect. Collect before you start.
- Commit after every task.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/GPU/alphazero/atlas_row_facts.py` (create) | Derive `phase` / `side` / `flat_policy` / `near_even` from frozen measured fields; cross-check against the assignment. |
| `scripts/GPU/alphazero/atlas_run.py` (create) | `run_row`, `run_corpus` — the composition, over an injected evaluator. Imports no MLX. |
| `scripts/GPU/alphazero/run_atlas.py` (create) | Operator CLI: `preflight`, `emit-runbook`, `run`. The only place a real evaluator is ever constructed. |
| `scripts/GPU/alphazero/atlas_readout_c.py` (modify) | `_is_flat` → `is_flat`, so the root-flat derivation reuses the one frozen predicate instead of copying it. |
| `tests/test_atlas_row_facts.py`, `..._atlas_run.py`, `..._run_atlas_cli.py` (create) | One suite per file, synthetic or `FakeEvaluator` only. |

---

### Task 0: Row facts, derived and cross-checked

**Files:**
- Modify: `scripts/GPU/alphazero/atlas_readout_c.py`
- Create: `scripts/GPU/alphazero/atlas_row_facts.py`
- Test: `tests/test_atlas_row_facts.py`

**Interfaces:**
- Consumes: `corpus_geometry.phase_for_ply` / `side_for_ply`; `atlas_readout_c.is_flat`;
  Stage 4's `snapshots["reference_lines"]["merged"]` and the `LegResult` list.
- Produces: `NEAR_EVEN_ABS_VALUE`; `derive_row_facts(legs, snapshots, target_ply, start_player) -> dict`.

> **This is the seam Stage 4 could not qualify.** `flat_policy` and `near_even` were
> caller-supplied booleans and every Stage 4 test hardcoded them, so Read-out B's
> `flat_policy` / `near_even` strata and Read-out C's `root_flat` / `near_even` strata
> were only as good as a producer that did not exist.

**Where each fact comes from, and why nothing new is introduced:**

| fact | frozen definition | source |
|---|---|---|
| `phase` | ply bounds 0–30 / 31–60 / 61–90 / 91+ (§3) | `phase_for_ply(target_ply)` |
| `side` | side-to-move at that ply (§3) | `side_for_ply(target_ply, start_player)` |
| `flat_policy` | normalized policy entropy ≥ `0.90` **and** top prior ≤ `0.025` (§8, "use the existing flat-policy definition ... rather than inventing strata after measurement") | `is_flat` applied to the merged line's **root edge** `parent_priors` |
| `near_even` | `\|V_stm\| ≤ 0.30` (§8) | `abs(legs[B=400].root_value)` — §5: "All values are side-to-move perspective" |

**Three decisions, each avoiding an invention:**

**a. `phase` and `side` are DERIVED and then ASSERTED against the assignment.**
`assign_corpus` already emits both, so re-deriving them is a cross-check, not a
computation: if `phase_for_ply(row["ply"])` disagrees with `row["phase"]`, the assignment
and the ply have drifted apart and the row must fail. That is a stronger use of the same
function than deriving a value nothing can contradict.

**b. Root flatness reads the reference line's root priors, not a new capture field.**
`capture_tree_state` records `policy_entropy` and `n_legal` but **not the top prior**, so
the frozen two-part predicate is not computable from a capture alone. The merged deep
line's root edge (`parent_path == ()`) carries `parent_priors`, and
`merge_reference_lines` already **asserts** those priors are identical at both deep rungs
— which under the frozen `add_noise=False` ladder is exactly the guarantee that makes
them the root's priors rather than one rung's. Widening the capture schema would be a
producer change for a value already in hand.

**c. An undefined fact is `None`, and the conflation is reported rather than fixed.**
A row whose merged line has no root edge has no root priors, so `flat_policy` is `None`.
`classify_strata` treats a falsy value as "not in the stratum", so a `None` and a `False`
both yield no `root_flat` membership. **Distinguishing them would require a new stratum,
which is a protocol change and is out of scope.** Instead `derive_row_facts` reports
`undefined`, `run_corpus` sums it, and the run document carries
`row_facts_undefined` so a reader can see how many rows the strata silently exclude.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atlas_row_facts.py
import pytest

from scripts.GPU.alphazero.atlas_row_facts import (
    NEAR_EVEN_ABS_VALUE, derive_row_facts,
)
from scripts.GPU.alphazero.atlas_readout_c import classify_edge_strata, is_flat
from scripts.GPU.alphazero.warm_prefix_replay import LegResult

FLAT = {i: 1.0 / 500 for i in range(500)}
SHARP = {0: 0.9, 1: 0.05, 2: 0.05}


def _legs(v400=0.10):
    return [LegResult(nominal_B=b, inherited_I=10, effective=10 + b,
                      root_value=(v400 if b == 400 else 0.05),
                      selected_move=3, selected_move_prior_rank=1,
                      top_share=0.5, top_two_margin=0.2,
                      effective_children=12.0, n_visited_children=20,
                      visit_counts={3: 100})
            for b in (400, 1600, 3200, 6400)]


def _snaps(root_priors=FLAT, with_root_edge=True):
    edges = ([{"parent_path": (), "move": 0, "depth": 0,
               "parent_priors": root_priors, "sources": (3200, 6400)}]
             if with_root_edge else [])
    return {"reference_lines": {"merged": {"required_edges": edges,
                                           "agreement": {}}}}


def test_the_frozen_near_even_bound_is_pinned_and_not_a_new_number():
    assert NEAR_EVEN_ABS_VALUE == 0.30          # design section 8, verbatim


def test_phase_and_side_are_derived_from_the_frozen_ply_bounds():
    f = derive_row_facts(_legs(), _snaps(), target_ply=95, start_player="red")
    assert f["phase"] == "late"                 # 91+
    assert f["side"] == "black"                 # odd ply, red started
    f = derive_row_facts(_legs(), _snaps(), target_ply=12, start_player="red")
    assert f["phase"] == "opening" and f["side"] == "red"


def test_near_even_uses_the_B400_root_value_in_stm_perspective():
    assert derive_row_facts(_legs(0.10), _snaps(), 12, "red")["near_even"] is True
    assert derive_row_facts(_legs(-0.29), _snaps(), 12, "red")["near_even"] is True
    assert derive_row_facts(_legs(0.31), _snaps(), 12, "red")["near_even"] is False
    # The bound is inclusive, exactly as section 8 states it.
    assert derive_row_facts(_legs(0.30), _snaps(), 12, "red")["near_even"] is True


def test_near_even_is_None_when_the_400_rung_is_absent():
    legs = [l for l in _legs() if l.nominal_B != 400]
    f = derive_row_facts(legs, _snaps(), 12, "red")
    assert f["near_even"] is None               # None, never False
    assert "near_even" in f["undefined"]


def test_flat_policy_applies_the_frozen_predicate_to_the_ROOT_EDGE_priors():
    assert derive_row_facts(_legs(), _snaps(FLAT), 12, "red")["flat_policy"] is True
    assert derive_row_facts(_legs(), _snaps(SHARP), 12, "red")["flat_policy"] is False


def test_flat_policy_is_None_when_the_merged_line_has_no_root_edge():
    """Undefined, never False. The row is KEPT and the gap is reported."""
    f = derive_row_facts(_legs(), _snaps(with_root_edge=False), 12, "red")
    assert f["flat_policy"] is None
    assert "flat_policy" in f["undefined"]


def test_root_and_edge_flatness_use_THE_SAME_predicate():
    """One frozen definition, one implementation. A second copy is how the
    root stratum and the local strata drift apart."""
    assert is_flat(FLAT) is True and is_flat(SHARP) is False
    # The edge-level classifier is built on the same function.
    assert classify_edge_strata({"depth": 1, "parent_priors": FLAT}) == {
        "locally_flat_depth1"}
    assert classify_edge_strata({"depth": 1, "parent_priors": SHARP}) == set()


def test_a_derived_phase_that_contradicts_the_assignment_FAILS_the_row():
    """assign_corpus already emits phase and side, so re-deriving them is a
    CROSS-CHECK. Disagreement means the assignment and the ply have drifted."""
    with pytest.raises(ValueError, match="phase"):
        derive_row_facts(_legs(), _snaps(), 95, "red", assigned_phase="opening")
    with pytest.raises(ValueError, match="side"):
        derive_row_facts(_legs(), _snaps(), 95, "red", assigned_side="red")
    # Agreement passes silently.
    derive_row_facts(_legs(), _snaps(), 95, "red",
                     assigned_phase="late", assigned_side="black")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_atlas_row_facts.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named '...atlas_row_facts'`

- [ ] **Step 3: Implement**

First, in `atlas_readout_c.py`, rename `_is_flat` → `is_flat` and update its two call
sites. It is the single frozen flat-policy predicate and now has a second legitimate
consumer, so it stops being private. No behaviour changes.

```python
# scripts/GPU/alphazero/atlas_row_facts.py
"""Derive the atlas row facts from FROZEN measured fields -- design sections 3
and 8. Introduces no new constant and no new predicate.

Stage 4 accepted `phase`, `flat_policy` and `near_even` as caller-supplied
booleans, so the one seam it could not qualify was the one that computes them.
This module is that producer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .atlas_readout_c import is_flat
from .corpus_geometry import phase_for_ply, side_for_ply

# Section 8's existing near-even definition, verbatim. NOT a new threshold.
NEAR_EVEN_ABS_VALUE = 0.30


def _root_edge_priors(snapshots: Dict[str, Any]) -> Optional[Dict[int, float]]:
    """The merged deep line's ROOT edge priors, or None.

    `merge_reference_lines` asserts a parent's priors are identical at both deep
    rungs, and the ladder runs add_noise=False, so these ARE the root's priors
    rather than one rung's reading of them.
    """
    merged = ((snapshots or {}).get("reference_lines") or {}).get("merged") or {}
    for edge in merged.get("required_edges", ()):
        if edge.get("depth") == 0:
            return edge.get("parent_priors")
    return None


def derive_row_facts(legs: Sequence[Any], snapshots: Dict[str, Any],
                     target_ply: int, start_player: str,
                     assigned_phase: Optional[str] = None,
                     assigned_side: Optional[str] = None) -> Dict[str, Any]:
    """The three row facts, plus the two the assignment already knows.

    `assigned_phase` / `assigned_side` turn the derivation into a CROSS-CHECK:
    disagreement means the assignment and the ply have drifted, which fails the
    row rather than being silently overwritten by either side.
    """
    phase = phase_for_ply(target_ply)
    side = side_for_ply(target_ply, start_player)
    if assigned_phase is not None and assigned_phase != phase:
        raise ValueError(
            f"ply {target_ply} derives phase {phase!r} but the assignment says "
            f"{assigned_phase!r}; the assignment and the ply have drifted")
    if assigned_side is not None and assigned_side != side:
        raise ValueError(
            f"ply {target_ply} with start_player {start_player!r} derives side "
            f"{side!r} but the assignment says {assigned_side!r}")

    undefined: List[str] = []

    priors = _root_edge_priors(snapshots)
    if priors:
        flat_policy: Optional[bool] = is_flat(priors)
    else:
        # None, never False: no root edge means no root priors, which is not
        # the same fact as a concentrated policy.
        flat_policy = None
        undefined.append("flat_policy")

    by_b = {l.nominal_B: l for l in legs}
    if 400 in by_b:
        near_even: Optional[bool] = abs(by_b[400].root_value) <= NEAR_EVEN_ABS_VALUE
    else:
        near_even = None
        undefined.append("near_even")

    return {"phase": phase, "side": side, "flat_policy": flat_policy,
            "near_even": near_even, "undefined": undefined}
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_row_facts.py tests/test_atlas_readout_c.py -v -p no:cacheprovider`
Expected: PASS — 8 new, plus Read-out C's 30 still green after the rename.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/atlas_row_facts.py scripts/GPU/alphazero/atlas_readout_c.py tests/test_atlas_row_facts.py
git commit -m "feat(atlas-s5): derive phase, side, flat_policy and near_even from frozen fields"
```

---

### Task 1: The composition — assigned row to authenticated artifact

**Files:**
- Create: `scripts/GPU/alphazero/atlas_run.py`
- Test: `tests/test_atlas_run.py`

**Interfaces:**
- Consumes: `GameMeta`; an assigned row `{"game_id", "seed", "split", "phase", "side", "ply"}` exactly as `corpus_geometry.assign_corpus` emits it; `replay_seed_for`, `replay_prefix`, `BatchSafeBoundaryObserver`, `run_additive_ladder`, `SelectionTracer`; `collect_features`; `derive_row_facts`; `classify_row`; `build_row`; all three read-outs; `emit`.
- Produces: `LADDER_BATCHING`; `ladder_config(n_simulations) -> MCTSConfig`; `RowOutcome`; `run_row(evaluator, meta, assigned, *, move_history, base_seed, ...) -> RowOutcome`; `run_corpus(evaluator, metas, assigned_rows, *, base_seed, move_histories, provenance, ...) -> dict`.
- **Imports no MLX and never constructs an evaluator.**

> **`ladder_config` is new here, and deliberately so.** `tests/atlas_stage1_fixtures.py`
> has a `shipped_config` helper, but it is a **test fixture** — production code must not
> import from `tests/`, and the frozen `(14, 48, 8)` batching and `add_noise=False` of
> §2b belong in the module that runs the ladder. It introduces no value that is not
> already frozen; it only stops the tuple being written out at each call site.

> **§2b's implementation consequence, enforced in code.** One evaluator for the whole
> run, one `MCTS` per row carrying its own `random.Random(replay_seed)`, continued across
> the prefix and all four legs and **never reseeded**. Rebuilding a compiled evaluator per
> unit of work is the documented MLX trap. `run_corpus` takes the evaluator as a
> parameter and has no factory argument at all, so the trap is unreachable rather than
> merely discouraged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atlas_run.py
"""Stage 5 composition -- CPU only, FakeEvaluator, no reservoir or checkpoint."""
import json
import random

import pytest

from scripts.GPU.alphazero.atlas_run import RowOutcome, run_corpus, run_row
from scripts.GPU.alphazero.corpus_geometry import GameMeta

from tests.eval_fakes import FakeEvaluator

BASE = 20500000
SIZE = 6


def _history(n, size=SIZE):
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    s = TwixtState(active_size=size, to_move="red")
    out = []
    for _ in range(n):
        lm = s.legal_moves()
        if not lm:
            break
        out.append(lm[0])
        s = s.apply_move(lm[0])
    return out


def _meta(game_id=0, n_moves=4):
    return GameMeta(game_id=game_id, seed=BASE + game_id, n_moves=n_moves,
                    start_player="red")


def _assigned(game_id=0, split="discovery", ply=2):
    from scripts.GPU.alphazero.corpus_geometry import phase_for_ply, side_for_ply
    return {"game_id": game_id, "seed": BASE + game_id, "split": split,
            "phase": phase_for_ply(ply), "side": side_for_ply(ply, "red"),
            "ply": ply}


@pytest.fixture(scope="module")
def one_row():
    hist = _history(4)
    ev = FakeEvaluator(value=0.0)
    out = run_row(ev, _meta(), _assigned(), move_history=hist, base_seed=BASE,
                  active_size=SIZE, increments=(80, 80, 80, 80), threshold=40,
                  leg_B=80)
    return out


def test_a_row_produces_a_complete_artifact_row(one_row):
    assert isinstance(one_row, RowOutcome)
    assert one_row.ok is True and one_row.failure is None
    row = one_row.row
    # The artifact row is simultaneously an A row, a B row and a C row.
    assert {"snapshots", "label", "phase", "flat_policy", "near_even"} <= set(row)
    assert {"features_at_boundary", "features_at_400"} <= set(row)
    assert len(row["legs"]) == 4


def test_the_row_facts_are_DERIVED_not_supplied(one_row):
    """The Stage 4 gap: these were hardcoded booleans everywhere."""
    row = one_row.row
    assert row["phase"] == "opening" and row["side"] == "red"
    assert row["flat_policy"] in (True, False, None)
    assert row["near_even"] in (True, False, None)
    assert "row_facts_undefined" in row


def test_ONE_evaluator_is_shared_across_every_row():
    """Section 2b: construct the evaluator once for the whole run. Rebuilding a
    compiled evaluator per unit of work is the documented MLX trap."""
    hist = _history(4)
    ev = FakeEvaluator(value=0.0)
    seen = []
    metas = [_meta(0), _meta(1)]
    rows = [_assigned(0), _assigned(1)]
    doc = run_corpus(ev, metas, rows, base_seed=BASE,
                     move_histories={0: hist, 1: hist},
                     provenance=_PROV, active_size=SIZE,
                     increments=(80, 80, 80, 80), threshold=40, leg_B=80,
                     _on_row=lambda mcts: seen.append(mcts.evaluator))
    assert len(seen) == 2
    assert seen[0] is seen[1] is ev          # identity, not equality


def test_each_row_gets_its_OWN_mcts_seeded_from_the_verified_replay_seed():
    hist = _history(4)
    ev = FakeEvaluator(value=0.0)
    seen = []
    run_corpus(ev, [_meta(0), _meta(1)], [_assigned(0), _assigned(1)],
               base_seed=BASE, move_histories={0: hist, 1: hist},
               provenance=_PROV, active_size=SIZE,
               increments=(80, 80, 80, 80), threshold=40, leg_B=80,
               _on_row=lambda mcts: seen.append(mcts))
    assert seen[0] is not seen[1]            # a fresh MCTS per row
    # Each carries the row's own frozen stream, verified against the sidecar.
    assert seen[0].rng is not seen[1].rng


def test_a_sidecar_seed_mismatch_fails_the_row_rather_than_being_assumed():
    """replay_seed_for verifies base_seed + game_idx against the sidecar."""
    bad = GameMeta(game_id=0, seed=BASE + 99, n_moves=4, start_player="red")
    out = run_row(FakeEvaluator(value=0.0), bad, _assigned(),
                  move_history=_history(4), base_seed=BASE, active_size=SIZE,
                  increments=(80, 80, 80, 80), threshold=40, leg_B=80)
    assert out.ok is False and "seed" in out.failure


def test_a_backup_invariant_violation_FAILS_the_row_and_is_recorded():
    """Section 6a: a violation means the accounting is wrong and the row must
    fail, not be recorded."""
    out = run_row(FakeEvaluator(value=0.0), _meta(), _assigned(),
                  move_history=_history(4), base_seed=BASE, active_size=SIZE,
                  increments=(80, 80, 80, 80), threshold=40, leg_B=80,
                  _corrupt_d3=True)
    assert out.ok is False and "backup accounting" in out.failure
    assert out.row is None                   # not a half-recorded row


def test_a_row_whose_boundary_never_fired_is_FAILED_not_defaulted():
    """A missing boundary is a missing measurement, not an N_actual of zero."""
    out = run_row(FakeEvaluator(value=0.0), _meta(), _assigned(),
                  move_history=_history(4), base_seed=BASE, active_size=SIZE,
                  # threshold above the leg: no flush can reach it
                  increments=(80, 80, 80, 80), threshold=10_000, leg_B=80)
    assert out.ok is False and "boundary" in out.failure


def test_inheritance_resets_KEEP_the_row():
    """Section 2b: never top up, drop, substitute or resample a row. Every row
    stays in the primary analysis and the reset rate is REPORTED."""
    hist = _history(4)
    out = run_row(FakeEvaluator(value=0.0), _meta(), _assigned(),
                  move_history=hist, base_seed=BASE, active_size=SIZE,
                  increments=(80, 80, 80, 80), threshold=40, leg_B=80)
    assert out.ok is True
    assert "reset_count" in out.row and "reset_rate" in out.row


_PROV = {"git_head": "a" * 40, "worktree_clean": True,
         "checkpoint_sha1": "0" * 40}


def test_run_corpus_composes_all_three_readouts_into_one_document():
    hist = _history(4)
    metas = [_meta(i) for i in range(4)]
    rows = [_assigned(0, "discovery"), _assigned(1, "discovery"),
            _assigned(2, "validation"), _assigned(3, "validation")]
    doc = run_corpus(FakeEvaluator(value=0.0), metas, rows, base_seed=BASE,
                     move_histories={i: hist for i in range(4)},
                     provenance=_PROV, active_size=SIZE,
                     increments=(80, 80, 80, 80), threshold=40, leg_B=80)
    assert set(doc) >= {"rows", "provenance", "readout_a", "readout_b",
                        "readout_c", "class_counts", "capacity",
                        "failed_rows", "row_facts_undefined"}
    # Splits are routed, not pooled: discovery fits, validation judges.
    assert doc["splits"] == {"discovery": 2, "validation": 2}
    assert doc["readout_a"]["authoritative"] == "features_at_boundary"
    assert doc["readout_c"]["selected_on"] == "discovery"


def test_failed_rows_are_reported_and_never_silently_dropped():
    hist = _history(4)
    metas = [_meta(0), GameMeta(game_id=1, seed=BASE + 99, n_moves=4,
                                start_player="red")]
    doc = run_corpus(FakeEvaluator(value=0.0), metas,
                     [_assigned(0), _assigned(1)], base_seed=BASE,
                     move_histories={0: hist, 1: hist}, provenance=_PROV,
                     active_size=SIZE, increments=(80, 80, 80, 80),
                     threshold=40, leg_B=80)
    assert len(doc["failed_rows"]) == 1
    assert doc["failed_rows"][0]["game_id"] == 1
    assert "seed" in doc["failed_rows"][0]["failure"]
    # Reported, NEVER gated: no invented "too many failures" threshold. The
    # frozen capacity gates are the only things that can stop the atlas.
    assert "max_failures" not in doc


def test_the_run_document_survives_emission_with_valid_provenance():
    hist = _history(4)
    doc = run_corpus(FakeEvaluator(value=0.0), [_meta()], [_assigned()],
                     base_seed=BASE, move_histories={0: hist},
                     provenance=_PROV, active_size=SIZE,
                     increments=(80, 80, 80, 80), threshold=40, leg_B=80)
    from scripts.GPU.alphazero.atlas_artifact import emit
    back = json.loads(emit(doc))
    assert back["provenance"]["worktree_clean"] is True
    assert "" in back["rows"][0]["snapshots"]["parent_visits"]["at_400"]


def test_run_corpus_cannot_build_an_evaluator():
    """The MLX trap is unreachable by construction: there is no factory
    parameter to pass, only an already-built evaluator."""
    import inspect
    params = set(inspect.signature(run_corpus).parameters)
    assert "evaluator" in params
    for forbidden in ("evaluator_factory", "checkpoint", "checkpoint_path"):
        assert forbidden not in params
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_atlas_run.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`atlas_run.py`, in outline — the full body follows the Stage 4 modules' style:

```python
# Section 2b's frozen ladder regime, named once instead of at each call site.
LADDER_BATCHING = (14, 48, 8)          # eval_batch_size, stall_flush, virtual


def ladder_config(n_simulations: int) -> MCTSConfig:
    eb, stall, virt = LADDER_BATCHING
    return MCTSConfig(n_simulations=n_simulations, eval_batch_size=eb,
                      stall_flush_sims=stall, pending_virtual_visits=virt)


@dataclass
class RowOutcome:
    ok: bool
    row: Optional[Dict[str, Any]]
    failure: Optional[str]
    game_id: int


def run_row(evaluator, meta, assigned, *, move_history, base_seed,
            active_size=24, increments=LEG_INCREMENTS,
            threshold=BOUNDARY_THRESHOLD, leg_B=400,
            _on_row=None, _corrupt_d3=False) -> RowOutcome:
    """ONE row: verified seed -> one MCTS -> prefix -> ladder -> facts -> row.

    Every failure path returns a RowOutcome carrying the reason. Nothing here
    raises past the caller, because one bad row must not abort a run that has
    already paid for the rows before it -- and nothing here defaults a missing
    measurement either.
    """
    try:
        seed = replay_seed_for(meta, base_seed)          # verifies the sidecar
    except ValueError as e:
        return RowOutcome(False, None, str(e), meta.game_id)

    # ONE MCTS per row, carrying ITS OWN frozen stream, continued across the
    # prefix and all four legs and never reseeded (section 2b). The evaluator is
    # the caller's and is never rebuilt.
    mcts = MCTS(evaluator, ladder_config(increments[0]), random.Random(seed))
    if _on_row is not None:
        _on_row(mcts)
    ...
    # prefix -> attach a FRESH tracer -> boundary observer -> ladder
    # then: boundary missing -> fail; check_backup_invariant -> fail;
    #       collect_features at both instants; derive_row_facts (cross-checked
    #       against assigned["phase"] / assigned["side"]); classify_row;
    #       build_row(..., snapshots=snaps)
```

```python
def run_corpus(evaluator, metas: Sequence[GameMeta],
               assigned_rows: Sequence[Dict[str, Any]], *, base_seed: int,
               move_histories: Dict[int, Sequence[Tuple[int, int]]],
               provenance: Dict[str, Any], active_size: int = 24,
               increments: Sequence[int] = LEG_INCREMENTS,
               threshold: int = BOUNDARY_THRESHOLD, leg_B: int = 400,
               _on_row=None) -> Dict[str, Any]:
    """Every assigned row through the full chain, then all three read-outs.

    `evaluator` is built ONCE by the caller and shared by every row. There is
    deliberately no factory or checkpoint parameter: section 2b requires one
    long-lived evaluator, and rebuilding a compiled one per unit of work is the
    documented MLX trap, so the trap is unreachable rather than discouraged.

    A failed row is recorded with its reason and the run continues. There is no
    aggregate failure threshold -- inventing one would be a new gate, and the
    frozen capacity gates are the only things that may stop the atlas.
    """
```

It loops `run_row`, partitions on `assigned["split"]`, then composes:

```python
    a = evaluate_detector_both(discovery_rows, validation_rows)
    b = {g: by_stratum_summary(all_rows, g) for g in GATE_NAMES}
    natural = natural_convergence_report(all_rows)
    c = select_on_discovery_validate_on_selected(discovery_rows, validation_rows)
    counts = class_counts([r["legs"] for r in all_rows])
    capacity = final_capacity_gate(class_counts([r["legs"] for r in validation_rows]))
    remaining = deployability([r["boundary"]["remaining"] for r in all_rows
                               if r["boundary"] is not None])
```

The read-out rows and the artifact rows are **the same objects** — Stage 4's `build_row`
result is already a valid A, B and C row, so no adapter exists anywhere in this module.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_run.py -v -p no:cacheprovider`
Expected: PASS — 12 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/atlas_run.py tests/test_atlas_run.py
git commit -m "feat(atlas-s5): compose assigned row through the read-outs into one run document"
```

---

### Task 2: The operator CLI, stop conditions and exit-status sidecars

**Files:**
- Create: `scripts/GPU/alphazero/run_atlas.py`
- Test: `tests/test_run_atlas_cli.py`

**Interfaces:**
- Produces: `EXIT_OK=0`, `EXIT_USAGE=2`, `EXIT_PROVENANCE=3`, `EXIT_ABORTED=5`; `STOP_CONDITIONS`; `write_status_sidecar(path, verdict, exit_code, **extra)`; `main()` with `preflight`, `emit-runbook`, `run`.
- **`run` is the only place `_default_evaluator_factory` is called, and it is imported lazily inside that branch.** Stage 5 writes it and never executes it.

> **Verdicts are results; exit codes are process outcomes.** A `CAPACITY_FAILURE`, a
> `NO_SHAPE_PASSES` or a `NOT_DEPLOYABLE` is a **finding**, written into the artifact and
> exiting **0** — the run did what it was asked. Only a usage error, a provenance failure
> or an abort is nonzero. Conflating the two would make an operational no-go look like a
> crash, which is exactly the framing the v18 closeout was careful to keep apart.

**The exit-status sidecar exists because a detached run's `$?` is unrecoverable.**
Phase 0 lost its exit code to `nohup`+`disown` with no sidecar, and completion had to be
qualified from the log. `run` writes `status.json` **last**, after the artifact, so its
presence is itself evidence the run reached the end; and the runbook additionally
captures the shell's `$?`, because a process killed before writing cannot write.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_atlas_cli.py
import json
import subprocess
import sys

import pytest

from scripts.GPU.alphazero.run_atlas import (
    EXIT_ABORTED, EXIT_OK, EXIT_PROVENANCE, EXIT_USAGE, STOP_CONDITIONS,
    write_status_sidecar,
)


def _cli(*args):
    return subprocess.run([sys.executable, "-m",
                           "scripts.GPU.alphazero.run_atlas", *args],
                          capture_output=True, text=True)


def test_exit_codes_follow_the_established_convention():
    assert (EXIT_OK, EXIT_USAGE, EXIT_PROVENANCE, EXIT_ABORTED) == (0, 2, 3, 5)


def test_every_frozen_stop_condition_is_listed_with_its_owner():
    """The runbook is the operator's only document. A stop condition that is
    not in it does not exist as far as the run is concerned."""
    names = {s["verdict"] for s in STOP_CONDITIONS}
    assert names >= {"PHASE_GEOMETRY_NO_GO", "ASSIGNMENT_SHORTFALL",
                     "PROJECTED_CAPACITY_NO_GO", "CAPACITY_FAILURE",
                     "INSUFFICIENT_CLASSES", "NOT_DEPLOYABLE",
                     "NO_SHAPE_PASSES", "PROVENANCE_FAILURE"}
    for s in STOP_CONDITIONS:
        assert s["owner"] and s["action"]      # who raises it, what to do


def test_a_read_out_verdict_is_a_RESULT_not_a_nonzero_exit():
    """CAPACITY_FAILURE and NO_SHAPE_PASSES are findings the run was asked to
    produce. Only process failures are nonzero."""
    for s in STOP_CONDITIONS:
        if s["verdict"] in ("CAPACITY_FAILURE", "NO_SHAPE_PASSES",
                            "NOT_DEPLOYABLE", "INSUFFICIENT_CLASSES"):
            assert s["exit_code"] == EXIT_OK


def test_emit_runbook_is_zero_gpu_and_prints_the_operator_stop():
    r = _cli("emit-runbook")
    assert r.returncode == EXIT_OK
    assert "OPERATOR STOP" in r.stdout
    # The recorded operator rules, verbatim in the runbook.
    for rule in ("nohup", "disown", "REAL_EXIT", "status.json",
                 "separate tool calls", "setsid"):
        assert rule in r.stdout


def test_the_runbook_command_captures_the_shell_exit_code():
    """A process killed before it can write its sidecar writes nothing, so the
    shell's $? must be captured too."""
    r = _cli("emit-runbook")
    assert 'echo "REAL_EXIT=$?"' in r.stdout


def test_preflight_fails_closed_on_provenance(tmp_path):
    art = tmp_path / "corpus.json"
    art.write_text(json.dumps({"verdict": "OK", "rows": []}))
    r = _cli("preflight", "--corpus-artifact", str(art),
             "--git-head", "z" * 40, "--checkpoint-sha1", "0" * 40,
             "--worktree-clean", "false")
    assert r.returncode == EXIT_PROVENANCE
    assert "PROVENANCE_FAILURE" in r.stdout


def test_preflight_rejects_an_unreadable_corpus_artifact(tmp_path):
    r = _cli("preflight", "--corpus-artifact", str(tmp_path / "nope.json"),
             "--git-head", "a" * 40, "--checkpoint-sha1", "0" * 40,
             "--worktree-clean", "true")
    assert r.returncode == EXIT_USAGE


def test_the_status_sidecar_records_the_verdict_and_the_exit_code(tmp_path):
    p = tmp_path / "status.json"
    write_status_sidecar(p, verdict="OK", exit_code=0, rows=240)
    d = json.loads(p.read_text())
    assert d["verdict"] == "OK" and d["exit_code"] == 0 and d["rows"] == 240


def test_no_subcommand_loads_a_checkpoint_or_mlx(tmp_path):
    """preflight and emit-runbook are zero-GPU. Only `run` builds an evaluator,
    and it imports the factory lazily inside that branch."""
    import scripts.GPU.alphazero.run_atlas as mod
    src = open(mod.__file__).read()
    assert "_default_evaluator_factory" in src
    assert src.index("def _cmd_run") < src.index("_default_evaluator_factory")
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_run_atlas_cli.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# scripts/GPU/alphazero/run_atlas.py
"""Atlas operator CLI -- design sections 3, 4, 9.

`preflight` and `emit-runbook` are ZERO-GPU. `run` is the single launchable
entry point; it is the only place an evaluator is ever constructed, and the
factory is imported lazily inside that branch so the other subcommands -- and
every test in this stage -- never touch MLX.
"""
EXIT_OK, EXIT_USAGE, EXIT_PROVENANCE, EXIT_ABORTED = 0, 2, 3, 5

# Verdict, the module that raises it, the operator action, and the exit code.
# A read-out verdict is a RESULT and exits 0; only process failures are nonzero.
STOP_CONDITIONS = ( ... )        # the table below, as dicts with those keys


def write_status_sidecar(path, *, verdict: str, exit_code: int, **extra) -> None:
    """Written LAST, after the artifact, so its presence is itself evidence the
    run reached the end.

    Phase 0 lost its exit code to nohup+disown with no sidecar and had to
    qualify completion from the log. This does not replace capturing the
    shell's $?: a process killed before it can write writes nothing.
    """


def _cmd_preflight(args) -> int: ...
def _cmd_emit_runbook(args) -> int: ...
def _cmd_run(args) -> int:
    """The launchable one. Stage 5 writes it and never executes it."""
    from .eval_runner import _default_evaluator_factory      # lazy: MLX
    evaluator = _default_evaluator_factory(args.checkpoint)  # ONCE per run
    ...
```

`STOP_CONDITIONS` is a literal table — verdict, the module that raises it, the operator
action, and the exit code:

| verdict | owner | exit | action |
|---|---|---|---|
| `PHASE_GEOMETRY_NO_GO` | `build_atlas_corpus pilot-gate` | 3 | Stop before the pilot ladder. No replacement games, no reassignment. |
| `ASSIGNMENT_SHORTFALL` | `build_atlas_corpus assign` | 4 | Stop. Do not top up, rebalance cells, move pilot rows, or relax one-position-per-game. |
| `PROJECTED_CAPACITY_NO_GO` | `atlas_labelling.size_from_pilot` | 0 | Stop with a projected capacity no-go rather than spending the full run. |
| `CAPACITY_FAILURE` | `atlas_labelling.final_capacity_gate` | 0 | Operational capacity failure. Do not weaken labels, move ambiguous rows, or add games. |
| `INSUFFICIENT_CLASSES` / `INSUFFICIENT_DISCOVERY_CLASSES` | `evaluate_detector` | 0 | Absence of evidence. Report as itself; never read as lateness. |
| `NOT_DEPLOYABLE` | `deployability` | 0 | Median `remaining` is zero: Read-out A cannot authorize the bounded 320+80 prototype. Separation is still reported. |
| `NO_SHAPE_PASSES` | `select_shape` | 0 | No widening shape clears the floors. Do not invent a third shape. |
| `PROVENANCE_FAILURE` | `atlas_artifact.emit` / `preflight` | 3 | The run is not reconstructible. Fix the tree or the checkpoint digest and start over. |
| row `failure` | `atlas_run.run_row` | 0 | Reported per row. **No aggregate failure threshold exists** — the frozen capacity gates are the only stop. |

`emit-runbook` prints the operator sequence, with the recorded §9 rules inline:

```text
OPERATOR STOP -- this tool has not been authorized to run the atlas.

 1. Confirm a clean tree and record HEAD.       git status --porcelain; git rev-parse HEAD
 2. Zero-GPU preflight (no checkpoint, no MLX):
      run_atlas preflight --corpus-artifact <assign.json> \
        --git-head <sha> --checkpoint-sha1 <sha> --worktree-clean true
    exit 0 continue | 2 usage | 3 PROVENANCE_FAILURE -- stop.
 3. Project runtime from the corpus's OBSERVED per-phase ply supply:
      run_atlas_ladder project-runtime --rows <N> --mean-prefix-plies <measured>
    Never scale a smoke to estimate runtime.
 4. LAUNCH and WAIT in SEPARATE shell invocations. A tool timeout SIGTERMs the
    whole process group when the launch and the wait share one call, and
    `setsid` does not exist on macOS:
      nohup .venv/bin/python -m scripts.GPU.alphazero.run_atlas run \
        --corpus-artifact <assign.json> --block-dir <dir> --base-seed <n> \
        --checkpoint <path> --out-dir <dir> > <dir>/run.log 2>&1 &
      disown
 5. Wait in a LATER call, then read the exit status from the sidecar the run
    writes LAST, and from the shell:
      cat <dir>/status.json            # verdict + exit_code, written after the artifact
      wait $!; echo "REAL_EXIT=$?"     # only in the launching shell
    `cmd | tail` reports the PIPE's exit code -- redirect to a file instead.
 6. No source edit after preflight, and no commit between generation and
    qualification.
```

- [ ] **Step 4: Run to verify it passes, then the full suite**

```bash
.venv/bin/python -m pytest tests/test_run_atlas_cli.py -v -p no:cacheprovider
.venv/bin/python -m pytest -p no:cacheprovider -q > /tmp/s5.out 2>&1; echo "REAL_EXIT=$?" >> /tmp/s5.out; tail -3 /tmp/s5.out
```

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/run_atlas.py tests/test_run_atlas_cli.py
git commit -m "feat(atlas-s5): operator CLI, stop conditions and exit-status sidecars"
```

---

## Stage 5 completion criteria

- [ ] `phase` and `side` are **derived and cross-checked** against the assignment; a
      disagreement fails the row rather than being silently overwritten.
- [ ] `flat_policy` uses the **one** frozen predicate — `is_flat`, shared with
      `classify_edge_strata`, not a second copy — applied to the merged line's root-edge
      priors; `near_even` is `|V_stm| ≤ 0.30` at nominal `B = 400`.
- [ ] An undefined row fact is **`None`**, the row is **kept**, and
      `row_facts_undefined` reports how many rows the strata therefore exclude.
- [ ] **One evaluator for the whole run, one seeded `MCTS` per row**, continued across
      the prefix and all four legs and never reseeded. `run_corpus` has **no** factory or
      checkpoint parameter, so rebuilding a compiled evaluator is unreachable.
- [ ] A verified-seed mismatch, a missing boundary, or a backup-invariant violation
      **fails that row with a recorded reason and no partial row**; every other row still
      runs; inheritance resets **keep** the row.
- [ ] Failed rows are **reported, never gated**. Stage 5 adds **no** aggregate failure
      threshold — the frozen capacity gates remain the only stops.
- [ ] The run document carries rows, provenance, all three read-outs, class counts, the
      capacity gate, deployability, splits, failures, and emits through the fail-closed
      `emit`.
- [ ] `STOP_CONDITIONS` lists every frozen verdict with its owner, action and exit code;
      **read-out verdicts exit 0** because they are findings, not process failures.
- [ ] The runbook prints the launch/wait separation, `nohup`+`disown`, the `setsid`
      absence, `REAL_EXIT=$?`, the `status.json` sidecar written **last**, and the
      no-edit-after-preflight and no-commit-between rules.
- [ ] `preflight` and `emit-runbook` are zero-GPU; `run` is the only branch that
      constructs an evaluator, and it imports the factory lazily.
- [ ] **Test counting.** Planned: row-facts 8, run 12, CLI 9 = **29**. Recount from
      `def test_` on disk at qualification and **baseline against a measured collect**,
      not against any number in a document — Stage 4's predicted total was short by
      exactly two tests a post-qualification commit had added. The full-suite delta must
      equal the recount; anything else means a pre-existing test changed behaviour and
      must be explained. Read the exit code from the process, never from a pipe.

## Out of scope

No reservoir generation, no checkpoint loading, no MLX execution, no pilot, no
measurement run, and no new threshold, protocol change or predicate. The three
distribution gaps — real-scale throughput, the `remaining` distribution, and the
inheritance-reset rate — remain **operator/pilot measurements**; Stage 5 builds the
instrument that will report them.

**What Stage 5 hands over:** one launchable protocol. Authorizing the pilot is a separate
written decision, taken after Stage 5 qualifies, and nothing in this plan grants it.
