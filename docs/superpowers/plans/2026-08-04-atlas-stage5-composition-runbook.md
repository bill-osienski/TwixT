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
document, in the two modes §3's chronology requires. `run_atlas.py` is the operator CLI:
a zero-GPU `preflight` that **measures** provenance, an `emit-runbook`, and the two
launchable entry points Stage 5 writes, qualifies against a patched factory, and never
executes for real.

**Tech Stack:** Python 3, stdlib only. Tests: `.venv/bin/python -m pytest -p no:cacheprovider`.

## Revision 2 — 2026-08-05, four operator-path repairs

Revision 1's row-fact design stands and no-go verdicts still exit 0. Four defects in the
operator path did not, and one qualification gap was missing.

| # | Defect | Repair |
|---|---|---|
| 1 | **The frozen pilot chronology was absent.** The runner consumed only `assign_corpus` rows — but that artifact exists only *after* the 24-row pilot ladder has determined `N`, and it contains only the `N − 24` continuation rows. There was no way to run the pilot at all, and no run could ever produce `N` positions. | Two modes. `run-pilot` executes the fixed 24-row pilot assignment, calls `size_from_pilot`, runs §8's early static widening check, and emits an authenticated pilot artifact. `run-final` combines those 24 discovery rows with the continuation assignment and **asserts the total is exactly `N`**. |
| 2 | **Preflight validated typed claims.** `--git-head`, `--checkpoint-sha1` and `--worktree-clean` were operator-supplied strings, so a wrong claim passed. | Preflight **measures**: `preflight_source_provenance` runs `git status --porcelain`, `git rev-parse HEAD` and hashes the actual `--checkpoint`, all **before** evaluator construction, then compares the measured digest against the block manifests and the pilot artifact. Negative cases are **constructed** through the pure `validate_source_provenance`, never observed from ambient dirtiness. |
| 3 | **A surviving subset was aggregated as an authoritative result.** Continuing past a failed row preserves already-paid diagnostics, but the frozen corpus is exactly `N` assigned positions, so read-outs over `N − k` are not the atlas. | Any seed mismatch, absent boundary or accounting failure makes the run `ABORTED`, **exit 5**. Failures and partial rows are retained for diagnosis and the read-outs still run, but the document carries `authoritative: false`. **A completeness condition, not a statistical threshold.** |
| 4 | **`wait $!` cannot recover a disowned PID.** A later shell has neither the job table nor a useful `$!` — the exact Phase 0 defect already on record, reproduced in the runbook that was supposed to prevent it. | Launch through a **detached shell wrapper** that runs Python and writes `REAL_EXIT` to a `shell_status` sidecar. The operator later reads that sidecar and `status.json`; no `wait` anywhere. |
| 5 | **No test invoked the launchable command.** The only entry point the operator can run was an unqualified producer/consumer seam. | `run-pilot` and `run-final` are each driven end to end with a patched `FakeEvaluator` factory over real temporary directories, assignments, artifacts and both sidecars. |

**No frozen parameter, threshold or predicate changed.** The completeness condition
counts assigned positions against measured ones; `PREFIX_SIMS = 400` is §2b's frozen
"400-sim searches" named once; `PILOT_GAMES` and `PILOT_PER_CELL` come from
`corpus_geometry`. Stage 5 still introduces no number of its own.

## Revision 3 — 2026-08-05, the pilot-to-final boundary

The two-stage structure survives review; seven defects at its seam did not. Most share a
cause: **revision 2 treated the pilot artifact as if it were still live memory.**

| # | Defect | Repair |
|---|---|---|
| 1 | **The pilot artifact is not reloadable.** `emit` runs everything through `_jsonable` and `json.dumps`: `LegResult`/`BoundaryRecord` become dicts, tuple paths become `"7\|3"` strings, `()` becomes `""`, and every integer map key becomes a string. Read-outs A and B address legs by *attribute*; C indexes `parent_visits` by *tuple* and priors by *int*. `run-final` would have fed all three a representation none of them can read. | New **Task 1**: `atlas_artifact.load_run` — the authenticated inverse of `emit`, rehydrating every type — qualified by an **emit → disk → load → all three read-outs** round trip. |
| 2 | **The pilot fixture cannot produce its own history.** It asked an `active_size=6` game for 94 moves (a 6×6 fixture game terminates around 29) and then declared `n_moves=95`, which `replay_prefix` rejects outright: it requires `meta.n_moves == len(move_history)`. | A real `active_size=24` late-position fixture with `n_moves` **derived from the history actually produced**, kept CPU-only by `FakeEvaluator` and an internal reduced prefix budget. |
| 3 | **`run-final` accepted an invented `N`.** The test passed `n_target=26` — outside the frozen `ALLOWED_N`, and produced by nobody. | `N` comes **exclusively** from `pilot_doc["sizing"]["N"]`, after validating an authoritative, successful pilot artifact holding exactly 24 fixed discovery rows. The `n_target` parameter is gone from the production path, and the continuation count must equal `N − 24`. |
| 4 | **Undefined widening evidence was scored as failure.** `_early_widening_check` treated a `None` retention rate as a failing shape, so a sparse pilot could close progressive widening on an *absence* of evidence. | Each shape gets `validation_verdict`'s frozen `FAIL > INCONCLUSIVE > PASS`, and `both_fail` fires only when **both** are genuine `FAIL`. |
| 5 | **Frozen parameters leaked onto the CLI.** `--active-size`, `--prefix-sims` and `--tiny-legs` would have let an operator change the board, the replay budget and the ladder. | The production parser exposes **none** of them, with a test proving it. Reduced budgets stay **internal test injection** at the module seam. |
| 6 | **The launch wrapper lost the exit code.** Emitted shell where the outer redirection races the sidecar write, and — worse — a substring assertion can never catch a redirection-order bug. | `launch_wrapper(...)` becomes a **function**, structured `rc=$?; echo "REAL_EXIT=$rc" > sidecar; exit $rc`, and the test **executes a harmless emitted wrapper and reads the file**. |
| 7 | *(corrections, below)* | |

**Corrections carried in the same pass**

- An **aborted pilot must not size**: `size_from_pilot` is not called, `sizing` is
  `UNAVAILABLE` with `N: None`, `early_widening_check` is non-authoritative, partial
  read-outs are preserved, exit 5.
- **Provenance binds `git_head`**, not just the checkpoint digest — see the asymmetry
  note in Task 3, which is a judgement call worth confirming.
- `_fake_block` writes the **full production manifest**: board 24, 400 simulations, 280
  max moves, batching `(14, 48, 8)`, noise **on**, clean provenance, exact filenames and
  seeds. Its `active_size=6` manifest could never have passed the real `load_block`.
- The stop-condition table said a row failure exits 0; it says **`ABORTED`, exit 5**.
- Task 2's expected result said 12 tests while the task contained 22.

Tasks renumber: **1** reloadable artifact, **2** composition, **3** operator CLI. Row
facts stay Task 0.

## Revision 4 — 2026-08-05, two calls and the last four seam defects

| # | Defect | Repair |
|---|---|---|
| 1 | **HEAD identity was asymmetric.** Revision 3 argued generation predates this tooling — **wrong: no reservoir exists yet.** Generation happens *after* Stage 5 qualifies, so the whole chain comes from one frozen commit and there is no mismatch to accommodate. | `measure_provenance` **requires** the measured `git_head` to equal both block manifests and the pilot artifact. A mismatch means regeneration or requalification. |
| 2 | **The assignment artifact was trusted.** `run-final` consumed hand-written continuation rows without re-deriving a *deterministic* assignment, and the fixture set the continuation block and the selected rows both to 176 — so an assignment that selected everything would have passed. | `verify_assignment` takes `sampling_seed` from the pilot artifact, loads the **complete** `G_total − 24` block, re-runs `size_continuation` + `assign_corpus`, and requires exact equality. Fixtures now keep **216 games** distinct from **176 rows**. |
| 3 | **The successful `run-final` was left unqualified**, on the argument that 200 real ladders cannot run on CPU. The ladders and the composition are **separable**. | New pure `combine_final_runs`, plus a CLI success-path test at the real frozen `N = 200` with `run_corpus` patched to a schema-valid complete 176-row document. No ladder, no budget override, no CLI flag. **The first production run stays evidence.** |
| 4 | **The loader restored only merged edges.** `at_3200` / `at_6400` carry `edges`, not `required_edges`, so both deep lines came back with list paths and string-keyed priors — an inverse in name only. It also accepted a truncated document as an empty run. | Both keys rehydrated and both pinned in the round trip; a missing or non-list `rows` field is refused. |

## Revision 5 — 2026-08-05, the verified-inputs contract

The contract, stated once and implemented in this order:

> **Load pilot games from the verified pilot block, recompute and cross-check the pilot
> assignment and sizing, derive selected continuation metadata from the verified complete
> continuation block, then measure exactly those assigned rows.**

| # | Defect | Repair |
|---|---|---|
| 1 | **The real pilot artifact lacked the assignment inputs.** `run_pilot` never recorded `sampling_seed`, `pilot_games` or `pilot_assignment`, yet `verify_assignment` required all three — and `pilot_games` could not be stored anyway, since `emit` flattens `GameMeta` to dicts and `load_run` does not rehydrate them. | The artifact stores **`sampling_seed` and the measured rows, nothing else**. New `verify_pilot` re-derives the gate, the assignment and the rows from the **verified pilot block** plus that seed, and returns the assignment for the continuation step. |
| 2 | **Pilot sizing was never revalidated.** The synthetic pilot held 24 stable-negatives yet claimed `N = 200`; the frozen rule returns `PROJECTED_CAPACITY_NO_GO` at `p_m = 0`. | `verify_pilot` recomputes `class_counts` + `size_from_pilot` from the carried rows and requires exact agreement. The fixture is now the formula's own worked case — **8 misleading, 9 stable-negative, 7 ambiguous**, which genuinely yields 200 — with a test proving it. |
| 3 | **The success stub could return an unrelated corpus.** The patched `run_corpus` invented ids `1000+` with an arbitrary split, so the seam would have passed even if measurement had substituted a different corpus. | The stub is built **from the `assigned_rows` it receives**, carrying every game id, split, phase, side and ply. The real split at `N = 200` — **96 discovery + 80 validation** — is pinned. |
| 4 | **The fixture artifact failed provenance.** It wrote `checkpoint_sha1 = "0"*40` while the manifests and the measurement carried the real digest, so symmetric validation rejected both CLI paths before `run-final` began. | One `_fixture_prov(ck)` object, used by the manifests, the artifact **and** the patched measurement. |

**Mechanical, same pass:** `run_final`'s stale `continuation_rows` becomes
`assignment_rows` and its gate tests take the new `pilot_games` /
`continuation_games`; the checkpoint-mismatch test patches
`preflight_source_provenance` too, since a TDD run's own uncommitted code would
otherwise raise "dirty worktree" first; `size_continuation["verdict"]` is checked before
`G_total` is read; and the stale task expectations become artifact **+5**, run **34** and CLI **18**.

Planned tests **61 → 65**, then **65 → 68** with revision 6's repairs; expected suite
**2624**.

## Revision 6 — 2026-08-05, two hard failures and one silent-consistency gap

| # | Defect | Repair |
|---|---|---|
| 1 | **`verify_pilot` read assignment field names off an artifact row.** `build_row` stores `game_idx` / `replay_seed` / `target_ply`; `assign_corpus` emits `game_id` / `seed` / `ply`. Every *honest* call raised `KeyError` — the happy path was the broken one. | An explicit `_AS_ASSIGNED` mapping, with **`seed` included in the comparison** so replay provenance is verified rather than merely carried. |
| 2 | **`_fake_block` spread `**prov` without assigning it.** Every CLI fixture would have died with `NameError` before loading a block. | `prov = _fixture_prov(ck)` immediately after `ck` is resolved; the stale `import hashlib` goes with it. |
| 3 | **The carried row's `label` was still trusted**, while sizing was re-derived. Read-out A takes its classes and Read-out C its intervention denominators from that stored label, and both strata sets from `flat_policy` / `near_even` — so editing only the label moves a row between classes while every leg-derived check still passes. | `verify_pilot` requires `label == classify_row(legs)` and re-derives `phase` / `side` / `flat_policy` / `near_even` through `derive_row_facts`. The fixture derives them too, since a fixture that cannot survive the validation it tests qualifies nothing — which surfaced that its own prior map is **flat**, so the previously hardcoded `flat_policy=False` was wrong. |

**Revision 4's mechanical pass:** `run_corpus` gains `prefix_sims`; deployability reads
`boundary.remaining` by attribute, not by key; the two bare `_pilot_metas()` calls get
their required length; `_authoritative_pilot_artifact` is built from **schema-valid**
rows (four rungs, a `BoundaryRecord`, populated snapshots) since recomposition consumes
them; and the CLI tests patch **only** `preflight_source_provenance`, because a TDD run
necessarily sees its own uncommitted implementation as a dirty tree and would compare the
real HEAD against the fixtures' `"a"*40`.

Planned tests move **45 → 55** (revision 3), then **55 → 61** with revision 4's repairs.

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
| `scripts/GPU/alphazero/atlas_run.py` (create) | `run_row`, `run_corpus`, and §3's two modes `run_pilot` / `run_final`. Over an injected evaluator; imports no MLX. |
| `scripts/GPU/alphazero/run_atlas.py` (create) | Operator CLI: `preflight`, `emit-runbook`, `run-pilot`, `run-final`. The only place a real evaluator is ever constructed. |
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

### Task 1: The pilot artifact must be reloadable

**Files:**
- Modify: `scripts/GPU/alphazero/atlas_artifact.py`
- Test: `tests/test_atlas_artifact.py` (append)

**Interfaces:**
- Produces: `load_run(path_or_text) -> dict` — the **authenticated inverse of `emit`**.

> **Why this task exists, and why it is before the composition.** `run-final` consumes
> the pilot artifact. `emit` writes it through `_jsonable` and `json.dumps`, which is
> lossy for exactly the types the read-outs need:
>
> | written as | comes back as | who breaks |
> |---|---|---|
> | `LegResult` dataclass | `dict` | Read-out B and `atlas_labelling` read `l.nominal_B` by **attribute** |
> | `BoundaryRecord` | `dict` | `deployability` reads `.remaining` |
> | `parent_visits` key `()` | `""` | `edge_retention` looks up **tuples** |
> | `parent_visits` key `(7, 3)` | `"7\|3"` | same |
> | `parent_priors` key `7` (int) | `"7"` | `static_retention` ranks **int** move ids |
> | `visit_counts` key `7` | `"7"` | same |
> | `parent_path` / `sources` tuple | `list` | edge keys and dedup identity |
>
> Every one of these is silently *wrong* rather than loudly broken: a string-keyed prior
> map still sorts, still has a length, and still produces a rank — just not the right
> one. Round-tripping is therefore not a nicety, it is the only thing that makes the
> two-stage protocol possible at all.

Rehydration is exactly invertible because move ids are integers, so `"7|3" → (7, 3)` and
`"" → ()` are unambiguous. `load_run` also **authenticates**: it checks
`schema_version`, and re-runs `validate_provenance` on the loaded provenance so a
hand-edited or truncated artifact cannot be consumed.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_atlas_artifact.py
from scripts.GPU.alphazero.atlas_artifact import load_run


def test_load_run_is_the_inverse_of_emit_for_every_lossy_type():
    leg = LegResult(nominal_B=400, inherited_I=137, effective=537,
                    root_value=0.25, selected_move=7,
                    selected_move_prior_rank=1, top_share=0.5,
                    top_two_margin=0.2, effective_children=12.0,
                    n_visited_children=20, visit_counts={7: 100})
    deep_edge = {"parent_path": (7,), "move": 3, "depth": 1,
                 "parent_priors": {3: 0.7, 4: 0.3}}
    snaps = _snapshots(
        parent_visits={"at_boundary": {(): 463, (7, 3): 12},
                       "at_400": {(): 537}},
        reference_lines={
            "at_3200": {"edges": [dict(deep_edge)], "moves": [3]},
            "at_6400": {"edges": [dict(deep_edge)], "moves": [3]},
            "merged": {"required_edges": [
                {"parent_path": (), "move": 7, "depth": 0,
                 "parent_priors": {7: 0.6, 8: 0.4},
                 "sources": (3200, 6400)}], "agreement": {}}})
    row = build_row(**_kw(legs=[leg], boundary=None, snapshots=snaps))
    back = load_run(emit({"rows": [row], "provenance": PROV}))["rows"][0]

    # Dataclasses, by ATTRIBUTE -- Read-out B and atlas_labelling need this.
    assert back["legs"][0].nominal_B == 400
    assert back["legs"][0].visit_counts == {7: 100}        # int keys, not "7"
    # Tuple paths, including the empty root path.
    pv = back["snapshots"]["parent_visits"]["at_boundary"]
    assert pv[()] == 463 and pv[(7, 3)] == 12
    # Edge identity: tuple path, int-keyed priors, tuple sources.
    lines = back["snapshots"]["reference_lines"]
    edge = lines["merged"]["required_edges"][0]
    assert edge["parent_path"] == () and edge["sources"] == (3200, 6400)
    assert edge["parent_priors"] == {7: 0.6, 8: 0.4}
    # BOTH DEEP LINES too: `merged` uses `required_edges`, the deep lines use
    # `edges`, and rehydrating only one leaves the other list-pathed and
    # string-keyed -- an inverse in name only.
    for rung in ("at_3200", "at_6400"):
        e = lines[rung]["edges"][0]
        assert e["parent_path"] == (7,)
        assert e["parent_priors"] == {3: 0.7, 4: 0.3}


def test_load_run_refuses_a_truncated_document():
    """A file that lost its rows must not read as a valid zero-row run."""
    doc = json.loads(emit({"rows": [], "provenance": PROV}))
    del doc["rows"]
    with pytest.raises(ValueError, match="rows"):
        load_run(json.dumps(doc))
    with pytest.raises(ValueError, match="rows"):
        load_run(json.dumps({**doc, "rows": {"0": {}}}))   # not a list


def test_a_boundary_record_rehydrates_or_stays_None():
    row = build_row(**_kw(boundary=BoundaryRecord(N_actual=326, overshoot=6,
                                                  remaining=74,
                                                  flush_type="full")))
    back = load_run(emit({"rows": [row], "provenance": PROV}))["rows"][0]
    assert back["boundary"].remaining == 74
    none_row = build_row(**_kw(boundary=None))
    back = load_run(emit({"rows": [none_row], "provenance": PROV}))["rows"][0]
    assert back["boundary"] is None and back["boundary_missing"] is True


def test_load_run_AUTHENTICATES_rather_than_merely_parsing():
    """A hand-edited or truncated artifact must not be consumable."""
    good = emit({"rows": [], "provenance": PROV})
    load_run(good)                                   # baseline: accepted
    doc = json.loads(good)
    doc["provenance"]["worktree_clean"] = False
    with pytest.raises(ValueError, match="provenance"):
        load_run(json.dumps(doc))
    doc = json.loads(good)
    doc["rows"] = [dict(build_row(**_kw()), schema_version=999)]
    with pytest.raises(ValueError, match="schema_version"):
        load_run(json.dumps(doc))


def test_the_ROUND_TRIP_feeds_all_three_readouts(tmp_path):
    """emit -> DISK -> load -> A, B and C. The two-stage protocol is exactly
    this path, so it is qualified as one."""
    from scripts.GPU.alphazero.atlas_readout_a import evaluate_detector_both
    from scripts.GPU.alphazero.atlas_readout_b import calibrate_gate
    from scripts.GPU.alphazero.atlas_readout_c import aggregate_shape

    def _four_rungs():
        """All four frozen rungs -- labelling and Read-out B index every one."""
        return [LegResult(nominal_B=b, inherited_I=10, effective=10 + b,
                          root_value=0.05, selected_move=3,
                          selected_move_prior_rank=1, top_share=0.5,
                          top_two_margin=0.2, effective_children=12.0,
                          n_visited_children=20, visit_counts={3: 100})
                for b in (400, 1600, 3200, 6400)]

    rows = [build_row(**_kw(legs=_four_rungs(), label=lbl))
            for lbl in ("misleading", "stable_negative")]
    p = tmp_path / "pilot_artifact.json"
    p.write_text(emit({"rows": rows, "provenance": PROV}))

    back = load_run(p)["rows"]
    # B reads legs by attribute; a dict would raise AttributeError here.
    assert calibrate_gate(back, "top_share_increase")["verdict"] in {
        "needs review", "no finding"}
    # C indexes parent_visits by tuple and ranks int-keyed priors.
    agg = aggregate_shape(back, ("c4a05", 4.0, 0.5))
    assert agg["gated_on"] == "at_400"
    # A reads the two feature dicts off the row.
    r = evaluate_detector_both(back, back, replicates=8)
    assert r["authoritative"] == "features_at_boundary"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_atlas_artifact.py -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'load_run'`

- [ ] **Step 3: Implement**

```python
def _unpath(key: str) -> Tuple[int, ...]:
    """"" -> (), "7|3" -> (7, 3). Unambiguous because move ids are integers."""
    return tuple(int(p) for p in key.split("|")) if key else ()


def load_run(source) -> Dict[str, Any]:
    """The AUTHENTICATED inverse of `emit`.

    `emit` is lossy for exactly the types the read-outs need -- dataclasses,
    tuple keys, integer keys -- and every loss is silently wrong rather than
    loudly broken: a string-keyed prior map still sorts and still yields a rank,
    just not the right one. Nothing may consume an artifact except through here.
    """
    doc = json.loads(source.read_text() if hasattr(source, "read_text")
                     else source)
    checked = validate_provenance(doc.get("provenance"))
    if checked["verdict"] != "OK":
        raise ValueError(f"refusing to load: provenance does not validate "
                         f"({', '.join(checked['problems'])})")
    # A truncated document must not read as an empty run: `.get("rows", ())`
    # would turn a file that lost its rows into a valid zero-row artifact.
    if not isinstance(doc.get("rows"), list):
        raise ValueError("refusing to load: `rows` is missing or not a list; "
                         "the artifact is truncated or was not written by emit")
    for row in doc["rows"]:
        if row.get("schema_version") != ROW_SCHEMA_VERSION:
            raise ValueError(
                f"row schema_version {row.get('schema_version')!r} != "
                f"{ROW_SCHEMA_VERSION}; this artifact was not written by this code")
        row["legs"] = [LegResult(**{**l, "visit_counts": {
            int(k): v for k, v in l["visit_counts"].items()}})
            for l in row["legs"]]
        row["boundary"] = (BoundaryRecord(**row["boundary"])
                           if row["boundary"] is not None else None)
        snaps = row["snapshots"]
        snaps["parent_visits"] = {
            inst: ({_unpath(k): v for k, v in (m or {}).items()}
                   if m is not None else None)
            for inst, m in snaps["parent_visits"].items()}
        # BOTH edge lists. The deep lines carry `edges`; only `merged` carries
        # `required_edges`, so rehydrating one key leaves at_3200 / at_6400
        # holding list paths and string-keyed priors -- which is not an
        # inverse of emit, and is silently wrong rather than broken.
        for line in snaps["reference_lines"].values():
            if not line:
                continue
            for key in ("edges", "required_edges"):
                for edge in line.get(key, ()):
                    edge["parent_path"] = tuple(edge["parent_path"])
                    if "sources" in edge:          # merged edges only
                        edge["sources"] = tuple(edge["sources"])
                    edge["parent_priors"] = {int(k): v for k, v
                                             in edge["parent_priors"].items()}
    return doc
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_artifact.py -v -p no:cacheprovider`
Expected: PASS — Stage 4's 14 plus 5 new.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/atlas_artifact.py tests/test_atlas_artifact.py
git commit -m "feat(atlas-s5): authenticated load_run, the reloadable inverse of emit"
```

---

### Task 2: The composition — assigned row to authenticated artifact

**Files:**
- Create: `scripts/GPU/alphazero/atlas_run.py`
- Test: `tests/test_atlas_run.py`

**Interfaces:**
- Consumes: `GameMeta`; an assigned row `{"game_id", "seed", "split", "phase", "side", "ply"}` exactly as `corpus_geometry.assign_corpus` emits it; `replay_seed_for`, `replay_prefix`, `BatchSafeBoundaryObserver`, `run_additive_ladder`, `SelectionTracer`; `collect_features`; `derive_row_facts`; `classify_row`; `build_row`; all three read-outs; `emit`.
- Produces: `LADDER_BATCHING`; `PREFIX_SIMS`; `ladder_config(n_simulations) -> MCTSConfig`; `RowOutcome`; `run_row(...) -> RowOutcome`; `run_corpus(...) -> dict`; `pilot_rows(pilot_games, sampling_seed) -> list[dict]`; `_early_widening_check(rows) -> dict`; `run_pilot(...) -> dict`; **`verify_pilot(pilot_doc, pilot_games) -> assignment`**; **`verify_assignment(pilot_games, pilot_assignment, sampling_seed, n_target, continuation_games, assignment_rows)`**; **`combine_final_runs(pilot_doc, continuation_doc, *, provenance) -> dict`**; `run_final(evaluator, *, pilot_doc, pilot_games, continuation_games, assignment_rows, base_seed, move_histories, provenance) -> dict`.
- **Imports no MLX and never constructs an evaluator.**

#### §3's chronology, which the runner must obey

`assign_corpus` cannot be the runner's only input, because it does not exist yet when the
first ladder runs and it never contains the pilot:

```text
pilot_geometry_gate(24 games)      -> assignment, 3 per phase x side cell
   |                                  STOP here on PHASE_GEOMETRY_NO_GO
   v
run-pilot: 24 rows on the full ladder, ALL discovery, never validation
   |
   v
size_from_pilot(pilot class counts) -> N          (or PROJECTED_CAPACITY_NO_GO)
   |
   v
generate the continuation block, then assign_corpus(...) -> exactly N - 24 rows
   |                                  (its pool EXCLUDES the pilot game ids)
   v
run-final: pilot's 24 discovery rows + continuation's N - 24  ==  exactly N
```

`assign_corpus` demands `d_c = N/8 − 3` per cell precisely because the pilot already
supplied 3, so the pilot rows are not re-derived at the final stage — they are **read back
from the pilot artifact**, which is what makes "pilot assignments are fixed as discovery
rows and never reconsidered" (§3) true in code rather than in intent. The 60/40 split
holds only when the pilot counts as discovery: `8 × (3N/40 − 3) + 24 = 3N/5`.

#### The assignment artifact is RECOMPUTED, never trusted

The continuation rows arrive as a JSON file. Consuming them as given would let a
hand-edited, stale or mis-seeded assignment become the corpus — and assignment is
**deterministic**, so there is no reason to trust rather than verify. `run-final`:

1. reads `sampling_seed` from the **pilot artifact**, so one seed governs the whole
   corpus and cannot be re-supplied at the final stage;
2. `load_block`s the complete authorized continuation block — **all `G_total − 24`
   games**, not just the selected ones;
3. re-runs `size_continuation` and `assign_corpus` against the pilot's `N`;
4. requires **exact equality** with the assignment artifact's rows.

**`G_total − 24` and `N − 24` are different numbers, and the fixture must reflect it.**
At `N = 200` the frozen sizing gives `g_cont = 185`, `G_total = 240`, so the continuation
*block* holds **216 games** while the assignment selects **176 rows** from it. Revision 3
made both 176, which would have hidden any defect in the selection step: an assignment
that simply took every game would have passed.

#### The completeness condition

The frozen corpus is exactly `N` assigned positions. A run that measured `N − k` of them
is not the atlas, so **any row failure makes the run `ABORTED`**, whatever the read-outs
say. The read-outs still run and are still written — a half-measured corpus is worth
diagnosing — but the document carries `authoritative: false` and the CLI exits **5**.

This is a **completeness condition, not a statistical threshold**: it counts assigned
positions against measured ones and introduces no number. It is also why there is still
no "maximum tolerable failures" knob — one failure is already disqualifying.

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


# -- the completeness condition ----------------------------------------------

def test_ANY_row_failure_makes_the_whole_run_ABORTED_and_non_authoritative():
    """The frozen corpus is exactly N assigned positions. A run that measured
    N-k of them is not the atlas, however good the surviving rows look.

    A COMPLETENESS condition, not a statistical threshold: it compares assigned
    against measured and introduces no number. One failure is disqualifying, so
    no "maximum tolerable failures" knob exists or can exist.
    """
    hist = _history(4)
    metas = [_meta(0), GameMeta(game_id=1, seed=BASE + 99, n_moves=4,
                                start_player="red")]
    doc = run_corpus(FakeEvaluator(value=0.0), metas,
                     [_assigned(0), _assigned(1)], base_seed=BASE,
                     move_histories={0: hist, 1: hist}, provenance=_PROV,
                     active_size=SIZE, increments=(80, 80, 80, 80),
                     threshold=40, leg_B=80)
    assert doc["verdict"] == "ABORTED"
    assert doc["authoritative"] is False
    assert doc["assigned"] == 2 and doc["measured"] == 1


def test_an_aborted_run_still_retains_its_diagnostics():
    """Already-paid rows and the failure reasons are kept -- the run is not
    authoritative, but it is worth diagnosing."""
    hist = _history(4)
    metas = [_meta(0), GameMeta(game_id=1, seed=BASE + 99, n_moves=4,
                                start_player="red")]
    doc = run_corpus(FakeEvaluator(value=0.0), metas,
                     [_assigned(0), _assigned(1)], base_seed=BASE,
                     move_histories={0: hist, 1: hist}, provenance=_PROV,
                     active_size=SIZE, increments=(80, 80, 80, 80),
                     threshold=40, leg_B=80)
    assert len(doc["rows"]) == 1                    # the row that succeeded
    assert len(doc["failed_rows"]) == 1
    assert doc["failed_rows"][0]["game_id"] == 1
    # The read-outs still ran, and say so plainly.
    assert doc["readout_a"] is not None
    assert doc["readout_a_authoritative"] is False


def test_a_complete_run_is_OK_and_authoritative():
    hist = _history(4)
    doc = run_corpus(FakeEvaluator(value=0.0), [_meta(0), _meta(1)],
                     [_assigned(0), _assigned(1)], base_seed=BASE,
                     move_histories={0: hist, 1: hist}, provenance=_PROV,
                     active_size=SIZE, increments=(80, 80, 80, 80),
                     threshold=40, leg_B=80)
    assert doc["verdict"] == "OK" and doc["authoritative"] is True
    assert doc["assigned"] == doc["measured"] == 2


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


def test_no_maximum_failure_threshold_is_invented():
    """Completeness is binary. A tolerance would be a new number, and there is
    nothing to tune: one unmeasured assigned position already disqualifies."""
    import inspect
    src = inspect.getsource(run_corpus)
    for invented in ("max_failures", "failure_rate", "tolerance"):
        assert invented not in src


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


# -- section 3's chronology: pilot first, then final --------------------------

from scripts.GPU.alphazero.atlas_run import pilot_rows, run_final, run_pilot
from scripts.GPU.alphazero.atlas_labelling import ALLOWED_N
from scripts.GPU.alphazero.corpus_geometry import PILOT_GAMES, PILOT_PER_CELL

SAMPLING_SEED = 20260805


def _late_history(min_plies=92):
    """A REAL board-24 history long enough to serve a late cell.

    Three things this cannot be faked around. A 6x6 fixture game terminates
    after roughly 29 legal moves, so `active_size=6` can never reach ply 91+.
    `replay_prefix` asserts `meta.n_moves == len(move_history)`, so the length
    must be DERIVED, never declared. And a history that walks into a win ends
    the game early, so the walk stops at terminal and the caller checks what it
    actually got.
    """
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    s, out = TwixtState(active_size=24, to_move="red"), []
    while len(out) < min_plies + 8:
        if s.is_terminal():
            break
        lm = s.legal_moves()
        if not lm:
            break
        # Spread the moves out rather than always taking legal_moves()[0],
        # which walks a single file and can connect early.
        mv = lm[(len(out) * 37) % len(lm)]
        out.append(mv)
        s = s.apply_move(mv)
    assert len(out) >= min_plies, (
        f"fixture produced only {len(out)} plies; a late cell needs ply 91+")
    return out


def _pilot_metas(n_moves):
    """24 games, all of length `n_moves` -- DERIVED from the history actually
    produced, never declared. Every game can then serve every cell, so the
    matching is feasible and the gate exercises its PASS path."""
    return [GameMeta(game_id=i, seed=BASE + i, n_moves=n_moves,
                     start_player="red") for i in range(PILOT_GAMES)]


_PILOT_CACHE = {}


def _tiny_pilot():
    """One cached 24-row pilot at board 24 with `prefix_sims=2`.

    Prefix replay costs `target_ply x prefix_sims` per row and the late cells
    sit at ply 91+, so the frozen 400 would be ~36,000 simulations per late row
    -- fine on a GPU, absurd in a unit test. `prefix_sims` defaults to the
    frozen 400 and is lowered only here, exactly as `replicates` is in
    Read-out A. The BOARD is not reduced: it is a frozen production setting and
    the block manifest would reject anything else.
    """
    if "doc" not in _PILOT_CACHE:
        hist = _late_history()
        metas = _pilot_metas(len(hist))
        _PILOT_CACHE["doc"] = run_pilot(
            FakeEvaluator(value=0.0), metas, sampling_seed=SAMPLING_SEED,
            base_seed=BASE,
            move_histories={m.game_id: hist for m in metas},
            provenance=_PROV, prefix_sims=2,
            increments=(80, 80, 80, 80), threshold=40, leg_B=80)
    return _PILOT_CACHE["doc"]


def test_the_pilot_fixture_can_actually_serve_a_late_cell():
    """The fixture is load-bearing: if it cannot reach ply 91+, the geometry
    gate fails and every pilot test below is vacuous."""
    hist = _late_history()
    assert len(hist) >= 92
    rows = pilot_rows(_pilot_metas(len(hist)), SAMPLING_SEED)
    assert max(r["ply"] for r in rows) >= 91
    assert {r["phase"] for r in rows} == {"opening", "early_mid", "midgame",
                                          "late"}


def test_pilot_rows_are_the_24_fixed_discovery_rows():
    rows = pilot_rows(_pilot_metas(len(_late_history())), SAMPLING_SEED)
    assert len(rows) == PILOT_GAMES
    # Three per phase x side cell, exactly as the gate demanded.
    cells = {}
    for r in rows:
        cells[(r["phase"], r["side"])] = cells.get((r["phase"], r["side"]), 0) + 1
    assert set(cells.values()) == {PILOT_PER_CELL}
    assert len(cells) == 8
    # One position per game, and every row carries a real ply.
    assert len({r["game_id"] for r in rows}) == PILOT_GAMES
    assert all(isinstance(r["ply"], int) for r in rows)


def test_pilot_rows_are_discovery_only_and_never_validation():
    """Section 3: included in DISCOVERY only and never eligible for
    validation."""
    metas = _pilot_metas(len(_late_history()))
    assert {r["split"] for r in pilot_rows(metas, SAMPLING_SEED)} == {
        "discovery"}


def test_pilot_rows_fail_closed_when_the_geometry_gate_did_not_pass():
    """A no-go here costs nothing but the pilot block, which is the point. It
    must not be silently downgraded into a smaller pilot."""
    short = [GameMeta(game_id=i, seed=BASE + i, n_moves=8, start_player="red")
             for i in range(PILOT_GAMES)]          # no game reaches ply 91
    with pytest.raises(ValueError, match="PHASE_GEOMETRY_NO_GO"):
        pilot_rows(short, SAMPLING_SEED)


def test_run_pilot_sizes_from_ITS_OWN_class_counts():
    """N is not an input to the pilot -- it is the pilot's output."""
    doc = _tiny_pilot()
    assert doc["sizing"]["verdict"] in {"OK", "PROJECTED_CAPACITY_NO_GO"}
    if doc["sizing"]["verdict"] == "OK":
        assert doc["sizing"]["N"] in ALLOWED_N
    else:
        assert doc["sizing"]["N"] is None          # None, never a default
    assert doc["splits"] == {"discovery": len(doc["rows"])}


def test_an_ABORTED_pilot_does_not_size_and_does_not_close_widening():
    """Sizing an incomplete pilot would set N from a class frequency measured
    over fewer than 24 rows, and closing progressive widening on it would end a
    read-out on evidence that was never gathered."""
    hist = _late_history()
    metas = _pilot_metas(len(hist))
    metas[0] = GameMeta(game_id=0, seed=BASE + 999, n_moves=len(hist),
                        start_player="red")          # seed will not verify
    doc = run_pilot(FakeEvaluator(value=0.0), metas,
                    sampling_seed=SAMPLING_SEED, base_seed=BASE,
                    move_histories={m.game_id: hist for m in metas},
                    provenance=_PROV, prefix_sims=2,
                    increments=(80, 80, 80, 80), threshold=40, leg_B=80)
    assert doc["verdict"] == "ABORTED" and doc["authoritative"] is False
    assert doc["sizing"]["verdict"] == "UNAVAILABLE"
    assert doc["sizing"]["N"] is None
    assert doc["early_widening_check_authoritative"] is False
    # Partial diagnostics are preserved, not discarded.
    assert doc["rows"] and doc["failed_rows"]


def test_run_pilot_reports_the_early_static_widening_check():
    """Section 8: if BOTH shapes clearly fail retention on the pilot, close
    progressive widening without inventing another shape."""
    doc = _tiny_pilot()
    ew = doc["early_widening_check"]
    assert set(ew) == {"c4a05", "c13a03", "both_fail"}
    for shape in ("c4a05", "c13a03"):
        assert ew[shape]["verdict"] in {"PASS", "FAIL", "INCONCLUSIVE"}
    assert doc["early_widening_check_authoritative"] is True


def _row_with_no_reference_edges():
    """A row whose merged line reached nothing, so every retention denominator
    is empty and every rate is None."""
    return {"snapshots": {"at_boundary": None, "at_400": None,
                          "parent_visits": {"at_boundary": {}, "at_400": {}},
                          "reference_lines": {"merged": {
                              "required_edges": [],
                              "agreement": {d: {"state": "absent_both"}
                                            for d in ("root", "reply",
                                                      "two_ply")}}}},
            "label": "ambiguous", "phase": "late",
            "flat_policy": None, "near_even": None}


def test_INCONCLUSIVE_widening_evidence_is_not_a_failure():
    """A None retention rate means the rate is UNDEFINED, not that the shape
    failed. Closing progressive widening on an absence of evidence is exactly
    the mistake the verdict precedence exists to prevent."""
    from scripts.GPU.alphazero.atlas_run import _early_widening_check
    # No required edges anywhere -> every retention rate is None.
    rows = [_row_with_no_reference_edges() for _ in range(3)]
    ew = _early_widening_check(rows)
    assert ew["c4a05"]["verdict"] == "INCONCLUSIVE"
    assert ew["c13a03"]["verdict"] == "INCONCLUSIVE"
    assert ew["both_fail"] is False               # NOT closed


# -- N comes from the pilot, and from nowhere else ---------------------------

def _pilot_stub(n=200, rows=24, verdict="OK", authoritative=True,
                sizing_verdict="OK"):
    """A pilot DOCUMENT, not a pilot run: these gates must reject before any
    measurement is paid for, so they are tested without one."""
    return {"mode": "pilot", "verdict": verdict, "authoritative": authoritative,
            "rows": [{"split": "discovery"} for _ in range(rows)],
            "sizing": {"verdict": sizing_verdict,
                       "N": n if sizing_verdict == "OK" else None}}


def test_run_final_takes_N_ONLY_from_the_pilot_sizing():
    """There is no n_target parameter to supply an invented value through."""
    import inspect
    assert "n_target" not in inspect.signature(run_final).parameters


def _final_kw(**over):
    """The gate arguments, defaulted so each test overrides only its own."""
    base = dict(pilot_games=_pilot_metas(len(_late_history())),
                continuation_games=[], assignment_rows=[], base_seed=BASE,
                move_histories={}, provenance=_PROV)
    base.update(over)
    return base


def test_run_final_requires_the_continuation_to_be_exactly_N_minus_24():
    with pytest.raises(ValueError, match="exactly"):
        run_final(FakeEvaluator(value=0.0), pilot_doc=_pilot_stub(n=200),
                  **_final_kw(assignment_rows=[_assigned(100)] * 3))


def test_run_final_refuses_a_non_authoritative_or_unsized_pilot():
    for stub in (_pilot_stub(verdict="ABORTED", authoritative=False),
                 _pilot_stub(sizing_verdict="PROJECTED_CAPACITY_NO_GO"),
                 _pilot_stub(sizing_verdict="UNAVAILABLE"),
                 _pilot_stub(rows=23)):            # not the fixed 24
        with pytest.raises(ValueError):
            run_final(FakeEvaluator(value=0.0), pilot_doc=stub, **_final_kw())


def test_run_final_revalidates_the_pilots_OWN_sizing():
    """A stored N that the carried rows do not produce is the difference
    between "N=200" and "N=200 because 8 of 24 were misleading"."""
    pilot = _complete_pilot_doc(n=200)
    lied = dict(pilot, sizing={**pilot["sizing"], "N": 400})
    with pytest.raises(ValueError, match="sizing"):
        verify_pilot(lied, _pilot_metas(len(_late_history())))
    # The honest one passes, and returns the recomputed assignment.
    assert verify_pilot(pilot, _pilot_metas(len(_late_history())))


def test_verify_pilot_rejects_rows_that_are_not_the_recomputed_assignment():
    pilot = _complete_pilot_doc(n=200)
    tampered = dict(pilot, rows=[dict(pilot["rows"][0], target_ply=7)]
                    + pilot["rows"][1:])
    with pytest.raises(ValueError, match="recomputed"):
        verify_pilot(tampered, _pilot_metas(len(_late_history())))


def test_verify_pilot_compares_the_ARTIFACT_field_names_not_the_assignment_ones():
    """`build_row` stores game_idx / replay_seed / target_ply; `assign_corpus`
    emits game_id / seed / ply. A comparison that reads the assignment names
    off an artifact row raises KeyError on every honest call -- so the happy
    path is the test that catches it."""
    pilot = _complete_pilot_doc(n=200)
    assert "game_id" not in pilot["rows"][0]          # the trap, made explicit
    assert verify_pilot(pilot, _pilot_metas(len(_late_history())))
    # `seed` is part of the comparison, so replay provenance is verified.
    bad_seed = dict(pilot, rows=[dict(pilot["rows"][0], replay_seed=BASE + 999)]
                    + pilot["rows"][1:])
    with pytest.raises(ValueError, match="recomputed"):
        verify_pilot(bad_seed, _pilot_metas(len(_late_history())))


def test_verify_pilot_refuses_a_row_whose_STORED_LABEL_was_edited():
    """Sizing reads the legs, but Read-out A takes its classes and Read-out C
    its intervention denominators from the stored `label`. Editing only the
    label moves a row between classes while every leg-derived check passes.

    The replacement is derived from the row's OWN label rather than written in:
    `_PILOT_MIX` begins with eight misleading rows, so a hardcoded
    `label="misleading"` on row 0 is not a tamper at all -- `verify_pilot`
    would rightly accept it and the `pytest.raises` would fail.
    """
    pilot = _complete_pilot_doc(n=200)
    row = pilot["rows"][0]
    other = "stable_negative" if row["label"] != "stable_negative" else "misleading"
    relabelled = dict(row, label=other)
    with pytest.raises(ValueError, match="label"):
        verify_pilot(dict(pilot, rows=[relabelled] + pilot["rows"][1:]),
                     _pilot_metas(len(_late_history())))


def test_verify_pilot_rederives_the_stratum_facts():
    """`flat_policy` and `near_even` drive both strata sets, so a stored value
    that the row's own measurements do not produce is a silently wrong
    stratum.

    Each wrong value is the NEGATION of what the row actually carries. Writing
    literals here would assert nothing the moment the fixture derives those
    same values -- which is exactly what happened once already.
    """
    pilot = _complete_pilot_doc(n=200)
    row = pilot["rows"][0]
    for field in ("flat_policy", "near_even"):
        assert row[field] in (True, False)          # a negation must be real
        tampered = dict(row, **{field: not row[field]})
        with pytest.raises(ValueError, match=field):
            verify_pilot(dict(pilot, rows=[tampered] + pilot["rows"][1:]),
                         _pilot_metas(len(_late_history())))


def _four_rung_legs(label_as="stable_negative"):
    """All four frozen rungs, shaped to CLASSIFY as `label_as`.

    `class_counts` re-derives labels from the LEGS, so a fixture cannot simply
    claim a label: a pilot of 24 stable-negatives has p_m = 0 and its own
    sizing rule returns PROJECTED_CAPACITY_NO_GO, whatever the artifact says.
    """
    v400 = {"misleading": 0.90,        # |0.90 - 0.05| = 0.85 >= 0.25
            "ambiguous": 0.20,        # 0.15 sits in the kept band
            "stable_negative": 0.06}[label_as]
    values = {400: v400, 1600: 0.10, 3200: 0.05, 6400: 0.05}
    return [LegResult(nominal_B=b, inherited_I=137, effective=137 + b,
                      root_value=values[b], selected_move=3,
                      selected_move_prior_rank=1, top_share=0.5,
                      top_two_margin=0.2, effective_children=12.0,
                      n_visited_children=20, visit_counts={3: 100})
            for b in (400, 1600, 3200, 6400)]


# 8 misleading + 9 stable-negative of 24 is the frozen formula's own worked
# case: max(60/(8/24), 75/(9/24)) = max(180, 200) = 200, already a multiple of
# 40. The remaining 7 are ambiguous, which section 5 keeps and counts.
_PILOT_MIX = (["misleading"] * 8 + ["stable_negative"] * 9 + ["ambiguous"] * 7)


def test_the_pilot_mix_really_produces_N_200():
    """If the fixture's own class counts do not yield 200, every run-final test
    built on it is asserting against an impossible artifact."""
    from scripts.GPU.alphazero.atlas_labelling import class_counts, size_from_pilot
    counts = class_counts([_four_rung_legs(m) for m in _PILOT_MIX])
    assert counts["misleading"] == 8 and counts["stable_negative"] == 9
    assert size_from_pilot(counts) == {"p_m": 8 / 24, "p_s": 9 / 24,
                                       "verdict": "OK", "N": 200,
                                       "required": 200}


def _populated_snapshots():
    """Snapshots Read-out C can aggregate: a merged line with a root edge, and
    parent-visit maps at both instants."""
    priors = {i: (1.0 if i == 3 else 0.5 - i * 1e-4) for i in range(500)}
    edge = {"parent_path": (), "move": 3, "depth": 0,
            "parent_priors": priors, "sources": (3200, 6400)}
    agree = {d: {"in_3200": True, "in_6400": True, "state": "agree"}
             for d in ("root", "reply", "two_ply")}
    return {"at_boundary": None, "at_400": None,
            "captures": {"at_start": {}, "at_boundary": {}, "at_400": {}},
            "parent_visits": {"at_boundary": {(): 463}, "at_400": {(): 537}},
            "reference_lines": {"at_3200": {"edges": [dict(edge)]},
                                "at_6400": {"edges": [dict(edge)]},
                                "merged": {"required_edges": [edge],
                                           "agreement": agree}}}


def _pilot_assignment():
    """`pilot_geometry_gate`'s assignment for the standard pilot fixture."""
    from scripts.GPU.alphazero.corpus_geometry import pilot_geometry_gate
    gate = pilot_geometry_gate(_pilot_metas(len(_late_history())),
                               SAMPLING_SEED)
    assert gate["verdict"] == "PASS"
    return gate["assignment"]


def _row_for(assigned, label_as="stable_negative", start_player="red"):
    """A schema-valid row FOR A SPECIFIC ASSIGNED ROW.

    Every identifying field is carried through -- game id, split, phase, side,
    ply -- so a stub built from these cannot silently substitute a different
    corpus for the one the assignment selected.

    The row facts and the label are DERIVED here exactly as the production path
    derives them, because `verify_pilot` re-derives both and would reject a
    fixture that asserted otherwise. A fixture that cannot survive the
    validation it is used to test qualifies nothing.
    """
    from scripts.GPU.alphazero.atlas_artifact import build_row
    from scripts.GPU.alphazero.atlas_labelling import classify_row
    from scripts.GPU.alphazero.atlas_row_facts import derive_row_facts
    from scripts.GPU.alphazero.warm_prefix_replay import BoundaryRecord

    legs, snaps = _four_rung_legs(label_as), _populated_snapshots()
    facts = derive_row_facts(legs, snaps, assigned["ply"], start_player,
                             assigned_phase=assigned["phase"],
                             assigned_side=assigned["side"])
    assert classify_row(legs) == label_as, (
        f"the fixture's legs classify as {classify_row(legs)}, not {label_as}")
    return build_row(
        game_idx=assigned["game_id"], replay_seed=assigned["seed"],
        target_ply=assigned["ply"], phase=facts["phase"],
        side=facts["side"], split=assigned["split"], inherited_I=137,
        reset_count=0, reset_rate=0.0, last_reset_ply=None,
        boundary=BoundaryRecord(N_actual=326, overshoot=6, remaining=74,
                                flush_type="full"),
        legs=legs, label=label_as,
        features_at_boundary={k: 0.5 for k in FEATURE_NAMES},
        features_at_400={k: 0.5 for k in FEATURE_NAMES},
        snapshots=snaps,
        flat_policy=facts["flat_policy"], near_even=facts["near_even"])


def _measured_pilot_rows():
    """The 24 pilot rows as a COMPLETE pilot run would have measured them:
    real assigned rows, and the class mix that genuinely yields N = 200."""
    assigned = pilot_rows(_pilot_metas(len(_late_history())), SAMPLING_SEED)
    return [_row_for(a, m) for a, m in zip(assigned, _PILOT_MIX)]


def _complete_pilot_doc(n=200):
    """The in-memory equivalent of `_authoritative_pilot_artifact`.

    It stores `sampling_seed` and the measured rows and NOTHING ELSE about the
    assignment -- exactly what `run_pilot` writes, because GameMeta objects
    cannot survive emit/load and must be re-derived from the pilot block.
    """
    return {"mode": "pilot", "verdict": "OK", "authoritative": True,
            "rows": _measured_pilot_rows(),
            "failed_rows": [], "assigned": 24, "measured": 24,
            "sampling_seed": SAMPLING_SEED,
            "sizing": {"p_m": 8 / 24, "p_s": 9 / 24, "verdict": "OK",
                       "N": n, "required": n}}


def _continuation_metas(n_games, start_index=24):
    """G_total - 24 games -- the COMPLETE authorized block, which is larger
    than the N - 24 rows the assignment selects from it."""
    hist = _late_history()
    return [GameMeta(game_id=start_index + i, seed=BASE + start_index + i,
                     n_moves=len(hist), start_player="red")
            for i in range(n_games)]


def _complete_continuation_doc(assigned_rows):
    """A schema-valid COMPLETE continuation result, built FROM the assigned
    rows the caller actually received.

    Not from an arbitrary range: a stub that invents its own game ids would let
    the success seam pass even if measurement had substituted a different
    corpus for the one the assignment selected. Every id, split, phase, side
    and ply is carried through.
    """
    rows = [_row_for(a) for a in assigned_rows]
    return {"verdict": "OK", "authoritative": True, "rows": rows,
            "failed_rows": [], "assigned": len(rows), "measured": len(rows),
            "splits": {"discovery": sum(1 for r in rows
                                        if r["split"] == "discovery"),
                       "validation": sum(1 for r in rows
                                         if r["split"] == "validation")}}


def _stub_measurement(monkeypatch, *, complete=True):
    """Patch the expensive half only, deriving its result from the REAL
    assigned rows so the substitution the seam must reject is impossible."""
    import scripts.GPU.alphazero.atlas_run as ar

    def _fake(evaluator, metas, assigned_rows, **kw):
        doc = _complete_continuation_doc(assigned_rows)
        if complete:
            return doc
        return dict(doc, verdict="ABORTED", authoritative=False,
                    rows=doc["rows"][:-1], measured=len(doc["rows"]) - 1,
                    failed_rows=[{"game_id": doc["rows"][-1]["game_idx"],
                                  "failure": "seed mismatch"}])

    monkeypatch.setattr(ar, "run_corpus", _fake)


def _assigned_176():
    """The REAL 176 continuation rows: 96 discovery + 80 validation at N=200."""
    games = _continuation_metas(216)                 # G_total - 24
    return assign_corpus(_pilot_assignment(), games, 200, SAMPLING_SEED)["rows"]


def test_the_continuation_split_is_96_discovery_and_80_validation():
    """8 x (3N/40 - 3) = 96 and 8 x N/20 = 80 at N = 200. With the pilot's 24
    counted as discovery the corpus is 120/80 -- the frozen 60/40."""
    rows = _assigned_176()
    assert len(rows) == 176
    assert sum(1 for r in rows if r["split"] == "discovery") == 96
    assert sum(1 for r in rows if r["split"] == "validation") == 80


def test_combine_final_runs_is_PURE_and_recomposes_over_the_whole_corpus():
    """The SUCCESSFUL final composition, qualified on CPU at the real frozen
    N -- no ladders, no budget override, no CLI flag. Carry, recomposition and
    the completeness inheritance are all exercised."""
    pilot = _complete_pilot_doc(n=200)               # 24 schema-valid rows
    cont = _complete_continuation_doc(_assigned_176())
    doc = combine_final_runs(pilot, cont, provenance=_PROV)
    assert doc["verdict"] == "OK" and doc["authoritative"] is True
    assert len(doc["rows"]) == 200
    assert doc["pilot_rows_carried"] == 24
    # The pilot's rows come FIRST and keep their discovery split.
    assert all(r["split"] == "discovery" for r in doc["rows"][:24])
    # All three read-outs ran over the COMBINED corpus, not one half.
    for key in ("readout_a", "readout_b", "readout_c"):
        assert doc[key] is not None
    assert doc["readout_a_authoritative"] is True


def test_combine_inherits_incompleteness_from_EITHER_half():
    pilot = _complete_pilot_doc(n=200)
    rows = _assigned_176()
    for bad in (dict(pilot, authoritative=False),):
        doc = combine_final_runs(bad, _complete_continuation_doc(rows),
                                 provenance=_PROV)
        assert doc["verdict"] == "ABORTED" and doc["authoritative"] is False
    cont = dict(_complete_continuation_doc(rows[:-1]), authoritative=False,
                measured=175, failed_rows=[{"game_id": 9, "failure": "seed"}])
    doc = combine_final_runs(pilot, cont, provenance=_PROV)
    assert doc["verdict"] == "ABORTED" and doc["authoritative"] is False


def test_run_final_recomputes_the_assignment_rather_than_trusting_it():
    """Assignment is deterministic, so a hand-edited or stale artifact must not
    become the corpus."""
    pilot_games = _pilot_metas(len(_late_history()))
    games = _continuation_metas(216)                 # G_total - 24, not N - 24
    good = _assigned_176()
    args = (pilot_games, _pilot_assignment(), SAMPLING_SEED, 200, games)
    verify_assignment(*args, good)                   # baseline: accepted
    tampered = [dict(good[0], ply=good[0]["ply"] + 2)] + good[1:]
    with pytest.raises(ValueError, match="recomputed"):
        verify_assignment(*args, tampered)


def test_the_continuation_BLOCK_is_larger_than_the_selected_rows():
    """G_total - 24 = 216 games supply N - 24 = 176 rows at N = 200. A fixture
    that made them equal would pass an assignment that selected everything."""
    pilot_games = _pilot_metas(len(_late_history()))
    games, rows = _continuation_metas(216), _assigned_176()
    assert len(games) == 216 and len(rows) == 176
    with pytest.raises(ValueError, match="frozen sizing"):
        verify_assignment(pilot_games, _pilot_assignment(), SAMPLING_SEED, 200,
                          _continuation_metas(176), rows)
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


PREFIX_SIMS = 400          # section 2b: "frozen 400-sim searches". NOT new.


def run_row(evaluator, meta, assigned, *, move_history, base_seed,
            active_size=24, prefix_sims=PREFIX_SIMS,
            increments=LEG_INCREMENTS,
            threshold=BOUNDARY_THRESHOLD, leg_B=400,
            _on_row=None, _corrupt_d3=False) -> RowOutcome:
    """ONE row: verified seed -> one MCTS -> prefix -> ladder -> facts -> row.

    Every failure path returns a RowOutcome carrying the reason rather than
    raising, so one bad row does not discard the diagnostics already paid for by
    the rows before it. It does NOT make the run survivable: `run_corpus` turns
    any failure into an ABORTED, non-authoritative run.

    `prefix_sims` defaults to section 2b's frozen 400 and is lowered only by CPU
    tests, where a 91-ply late prefix would otherwise be ~36,000 simulations.
    """
    try:
        seed = replay_seed_for(meta, base_seed)          # verifies the sidecar
    except ValueError as e:
        return RowOutcome(False, None, str(e), meta.game_id)

    # ONE MCTS per row, carrying ITS OWN frozen stream, continued across the
    # prefix and all four legs and never reseeded (section 2b). The evaluator is
    # the caller's and is never rebuilt. The prefix runs at prefix_sims; the
    # ladder overrides n_simulations per leg and restores it in a finally.
    mcts = MCTS(evaluator, ladder_config(prefix_sims), random.Random(seed))
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
               prefix_sims: int = PREFIX_SIMS,
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

It loops `run_row`, partitions on `assigned["split"]`, applies the completeness
condition, then composes:

```python
    measured, failures = [], []
    ...
    complete = not failures
    verdict = "OK" if complete else "ABORTED"
    # A run that measured N-k of N assigned positions is not the atlas. The
    # read-outs still run -- a half-measured corpus is worth diagnosing -- but
    # nothing computed from a partial corpus may be called authoritative.
    doc = {"verdict": verdict, "authoritative": complete,
           "assigned": len(assigned_rows), "measured": len(measured),
           "readout_a_authoritative": complete,
           "readout_b_authoritative": complete,
           "readout_c_authoritative": complete, ...}
```

```python
    a = evaluate_detector_both(discovery_rows, validation_rows)
    b = {g: by_stratum_summary(all_rows, g) for g in GATE_NAMES}
    natural = natural_convergence_report(all_rows)
    c = select_on_discovery_validate_on_selected(discovery_rows, validation_rows)
    counts = class_counts([r["legs"] for r in all_rows])
    capacity = final_capacity_gate(class_counts([r["legs"] for r in validation_rows]))
    # `boundary` is a BoundaryRecord dataclass, not a dict -- by ATTRIBUTE.
    remaining = deployability([r["boundary"].remaining for r in all_rows
                               if r["boundary"] is not None])
```

The read-out rows and the artifact rows are **the same objects** — Stage 4's `build_row`
result is already a valid A, B and C row, so no adapter exists anywhere in this module.

§3's two modes sit on top of that engine:

```python
def pilot_rows(pilot_games: Sequence[GameMeta], sampling_seed: int
               ) -> List[Dict[str, Any]]:
    """The 24 FIXED pilot rows, in `assign_corpus`'s row shape.

    Fails closed on PHASE_GEOMETRY_NO_GO rather than quietly running a smaller
    pilot: the gate exists to stop before the pilot ladder is paid for, and a
    downgraded pilot would silently change the sizing denominator.
    """
    gate = pilot_geometry_gate(pilot_games, sampling_seed)
    if gate["verdict"] != "PASS":
        raise ValueError(f"PHASE_GEOMETRY_NO_GO: {gate['unmet']}")
    by_id = {g.game_id: g for g in pilot_games}
    rows = []
    for gid in sorted(gate["assignment"]):
        split, phase, side = gate["assignment"][gid]      # split is discovery
        rows.append({"game_id": gid, "seed": by_id[gid].seed, "split": split,
                     "phase": phase, "side": side,
                     "ply": select_ply(by_id[gid], split, phase, side,
                                       sampling_seed)})
    return rows


def _early_widening_check(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Section 8's early static check, on the pilot only.

    Each shape gets the FROZEN three-way precedence, because a `None` retention
    rate means the rate is UNDEFINED -- not that the shape failed. Scoring an
    undefined rate as a failure would let a sparse pilot close progressive
    widening on an ABSENCE of evidence, which is the exact mistake
    `validation_verdict`'s precedence exists to prevent.

    `both_fail` therefore fires only on two genuine FAILs. Reported to the
    operator; never acted on automatically.
    """
    per = {s[0]: validation_verdict(aggregate_shape(rows, s))
           for s in WIDENING_SHAPES}
    return {**per,
            "both_fail": all(v["verdict"] == "FAIL" for v in per.values())}


def run_pilot(evaluator, pilot_games, *, sampling_seed, base_seed,
              move_histories, provenance, **kw) -> dict:
    """Section 3's pilot: 24 rows, ALL discovery, on the full ladder.

    N is this function's OUTPUT, not its input.
    """
    doc = run_corpus(evaluator, pilot_games, pilot_rows(pilot_games, sampling_seed),
                     base_seed=base_seed, move_histories=move_histories,
                     provenance=provenance, **kw)
    doc["mode"] = "pilot"
    # The ONLY assignment input the artifact carries. `pilot_games` and the
    # gate's assignment are deliberately NOT stored: `emit` would flatten
    # GameMeta objects to dicts and `load_run` does not rehydrate them, so
    # run-final re-derives both from the verified pilot block plus this seed.
    doc["sampling_seed"] = sampling_seed
    if not doc["authoritative"]:
        # An incomplete pilot must not size: N would come from a class
        # frequency measured over fewer than 24 rows, and closing progressive
        # widening on it would end a read-out on evidence never gathered.
        # Partial diagnostics are preserved; the conclusions are not offered.
        doc["sizing"] = {"verdict": "UNAVAILABLE", "N": None,
                         "reason": "the pilot did not measure all 24 assigned "
                                   "positions; sizing needs a complete pilot"}
        doc["early_widening_check"] = _early_widening_check(doc["rows"])
        doc["early_widening_check_authoritative"] = False
        return doc
    doc["sizing"] = size_from_pilot(doc["class_counts"])
    doc["early_widening_check"] = _early_widening_check(doc["rows"])
    doc["early_widening_check_authoritative"] = True
    return doc


def verify_pilot(pilot_doc, pilot_games) -> Dict[int, Tuple[str, str, str]]:
    """Recompute and cross-check everything the pilot claims.

    `pilot_games` comes from the VERIFIED pilot block, not from the artifact:
    `emit` would have flattened `GameMeta` objects to dicts and `load_run` does
    not rehydrate them, so the artifact stores only `sampling_seed` and the
    measured rows. The geometry gate, the assignment and the sizing are all
    re-derived here from the block plus that one seed.

    Returns the recomputed pilot assignment, which the continuation assignment
    needs as its input.
    """
    seed = pilot_doc["sampling_seed"]
    gate = pilot_geometry_gate(pilot_games, seed)
    if gate["verdict"] != "PASS":
        raise ValueError(f"PHASE_GEOMETRY_NO_GO on the pilot block: "
                         f"{gate['unmet']}")
    # An ARTIFACT row and an ASSIGNMENT row name the same facts differently:
    # build_row stores game_idx / replay_seed / target_ply, while assign_corpus
    # emits game_id / seed / ply. Mapping them explicitly is the difference
    # between a comparison and a KeyError. `seed` is included so replay
    # provenance is actually verified, not merely carried.
    _AS_ASSIGNED = {"game_idx": "game_id", "replay_seed": "seed",
                    "target_ply": "ply", "split": "split", "phase": "phase",
                    "side": "side"}
    expected = pilot_rows(pilot_games, seed)
    measured = [{dst: r[src] for src, dst in _AS_ASSIGNED.items()}
                for r in pilot_doc["rows"]]
    if measured != [{k: e[k] for k in _AS_ASSIGNED.values()}
                    for e in expected]:
        raise ValueError("the pilot artifact's rows do not match a recomputed "
                         "pilot assignment; it is stale, edited or mis-seeded")

    # Sizing is re-derived from the CARRIED rows, not trusted. A stored N that
    # its own class counts do not produce is the difference between "N=200" and
    # "N=200 because 8 of 24 were misleading".
    counts = class_counts([r["legs"] for r in pilot_doc["rows"]])
    if size_from_pilot(counts) != pilot_doc["sizing"]:
        raise ValueError(
            f"stored sizing {pilot_doc['sizing']} is not what the carried "
            f"pilot rows produce ({size_from_pilot(counts)})")

    # The STORED label and row facts are re-derived too. Sizing reads the legs,
    # but Read-out A takes its classes and Read-out C its intervention
    # denominators from `label`, and both strata sets from `flat_policy` /
    # `near_even` -- so an edited label silently moves a row between classes
    # while every leg-derived check still passes.
    by_id = {g.game_id: g for g in pilot_games}
    for row in pilot_doc["rows"]:
        if row["label"] != classify_row(row["legs"]):
            raise ValueError(
                f"row {row['game_idx']}: stored label {row['label']!r} is not "
                f"what its own legs classify as "
                f"({classify_row(row['legs'])!r})")
        facts = derive_row_facts(row["legs"], row["snapshots"],
                                 row["target_ply"],
                                 by_id[row["game_idx"]].start_player)
        for field in ("phase", "side", "flat_policy", "near_even"):
            if row[field] != facts[field]:
                raise ValueError(
                    f"row {row['game_idx']}: stored {field} {row[field]!r} != "
                    f"re-derived {facts[field]!r}")
    return gate["assignment"]


def verify_assignment(pilot_games, pilot_assignment, sampling_seed, n_target,
                      continuation_games, assignment_rows) -> None:
    """Assignment is DETERMINISTIC, so recompute it rather than trusting the
    artifact. A hand-edited, stale or mis-seeded file would otherwise become
    the corpus.

    `continuation_games` is the COMPLETE authorized block -- `G_total - 24`,
    which is larger than the `N - 24` rows selected from it. Passing the
    selected rows here instead would make any selection look correct.
    """
    sizing = size_continuation(pilot_games, n_target)
    if sizing.get("verdict") != "OK":
        raise ValueError(f"continuation sizing is {sizing.get('verdict')!r}, "
                         f"so there is no G_total to check against")
    if len(continuation_games) != sizing["G_total"] - PILOT_GAMES:
        raise ValueError(
            f"continuation block holds {len(continuation_games)} games but the "
            f"frozen sizing requires {sizing['G_total'] - PILOT_GAMES}")
    recomputed = assign_corpus(pilot_assignment, continuation_games,
                               n_target, sampling_seed)
    if recomputed["verdict"] != "OK" or recomputed["rows"] != assignment_rows:
        raise ValueError("the assignment artifact does not match a recomputed "
                         "assignment; it is stale, edited, or mis-seeded")


def run_final(evaluator, *, pilot_doc, pilot_games, continuation_games,
              assignment_rows, base_seed, move_histories, provenance,
              **kw) -> dict:
    """The pilot's 24 discovery rows PLUS the continuation's N-24.

    The contract, in order:

        1. load pilot games from the VERIFIED pilot block
        2. recompute and cross-check the pilot assignment AND its sizing
        3. derive the selected continuation rows from the VERIFIED complete
           continuation block, by recomputing the assignment
        4. measure exactly those assigned rows

    N comes from `pilot_doc["sizing"]` and from NOWHERE else -- there is no
    `n_target` parameter, so an invented or out-of-set value cannot be
    supplied. The pilot rows are CARRIED, never re-measured: section 3 fixes
    them as discovery rows that are never reconsidered, and `assign_corpus`
    already excluded their game ids from its pool.

    `continuation_games` is the COMPLETE authorized block (G_total - 24), a
    different and larger number than the N-24 selected rows; the selection step
    is what `verify_assignment` re-derives.
    """
    if pilot_doc.get("verdict") != "OK" or not pilot_doc.get("authoritative"):
        raise ValueError("refusing to run: the pilot artifact is not an "
                         "authoritative, complete pilot")
    sizing = pilot_doc.get("sizing") or {}
    if sizing.get("verdict") != "OK" or sizing.get("N") is None:
        raise ValueError(f"refusing to run: pilot sizing is "
                         f"{sizing.get('verdict')!r}, so there is no N")
    carried = pilot_doc["rows"]
    if len(carried) != PILOT_GAMES:
        raise ValueError(f"pilot artifact holds {len(carried)} rows, not the "
                         f"fixed {PILOT_GAMES}")
    n_target = sizing["N"]
    if len(carried) + len(assignment_rows) != n_target:
        raise ValueError(
            f"corpus must contain exactly N={n_target} positions: "
            f"{len(carried)} pilot + {len(assignment_rows)} continuation")
    pilot_assignment = verify_pilot(pilot_doc, pilot_games)
    verify_assignment(pilot_games, pilot_assignment,
                      pilot_doc["sampling_seed"], n_target,
                      continuation_games, assignment_rows)
    # Measure exactly the assigned rows, whose metadata comes from the VERIFIED
    # continuation block rather than from the assignment file.
    by_id = {g.game_id: g for g in continuation_games}
    metas = [by_id[r["game_id"]] for r in assignment_rows]
    cont = run_corpus(evaluator, metas, assignment_rows, base_seed=base_seed,
                      move_histories=move_histories, provenance=provenance, **kw)
    return combine_final_runs(pilot_doc, cont, provenance=provenance)


def combine_final_runs(pilot_doc, continuation_doc, *, provenance) -> dict:
    """PURE. Carry the pilot rows, recompose all three read-outs over the
    combined corpus, and inherit the completeness condition from both halves.

    Separated from `run_final` on purpose: the expensive part is the ladders,
    and the part most likely to be wrong is this one. Splitting them lets the
    SUCCESSFUL final composition be qualified on CPU with a synthetic complete
    continuation result -- at the real frozen N, with no budget override and no
    ladder -- so the first production run stays evidence rather than becoming a
    disposable qualification run.
    """
    rows = list(pilot_doc["rows"]) + list(continuation_doc["rows"])
    complete = pilot_doc["authoritative"] and continuation_doc["authoritative"]
    ...
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_run.py -v -p no:cacheprovider`
Expected: PASS — 37 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/atlas_run.py tests/test_atlas_run.py
git commit -m "feat(atlas-s5): compose assigned row through the read-outs into one run document"
```

---

### Task 3: The operator CLI, stop conditions and exit-status sidecars

**Files:**
- Create: `scripts/GPU/alphazero/run_atlas.py`
- Test: `tests/test_run_atlas_cli.py`

**Interfaces:**
- Produces: `EXIT_OK=0`, `EXIT_USAGE=2`, `EXIT_PROVENANCE=3`, `EXIT_ABORTED=5`; `STOP_CONDITIONS`; `measure_provenance(checkpoint, *, pilot_dir=None, continuation_dir=None, pilot_artifact=None) -> dict`; **`launch_wrapper(argv, *, out_dir) -> str`**; `write_status_sidecar(path, *, verdict, exit_code, **extra)`; **`main(argv=None) -> int`** with `preflight`, `emit-runbook`, `run-pilot`, `run-final`. `argv` is a parameter so the launchable commands can be driven in-process by tests, rather than only through a subprocess that could never patch the evaluator factory.
- **`run-pilot` and `run-final` are the only places `_default_evaluator_factory` is called, and it is imported lazily inside those branches.** Stage 5 writes them, drives them once each against a patched factory, and never executes them for real.

#### Provenance is MEASURED, not claimed

Accepting `--git-head`, `--checkpoint-sha1` and `--worktree-clean` as arguments validates
a *typed claim*: an operator who mistypes a digest, or who passes `--worktree-clean true`
on a dirty tree, passes the gate. `measure_provenance` instead:

1. calls `generate_atlas_reservoir.preflight_source_provenance(checkpoint)`, which runs
   `git status --porcelain`, `git rev-parse HEAD`, and hashes the actual checkpoint file
   — **before any evaluator is constructed**, because checking afterwards means a dirty
   tree can consume an entire GPU run before anything rejects it;
2. compares the **measured** digest against `load_manifest(pilot_dir)` and
   `load_manifest(continuation_dir)`, and calls `assert_blocks_agree` on the pair, so the
   run cannot use a different network from the one that generated its games;
3. **requires the measured `git_head` to equal both block manifests and the pilot
   artifact** — symmetric, not merely the checkpoint digest.

**The HEAD requirement is symmetric.** Revision 3 argued for an asymmetry on the grounds
that generation predates this tooling; that is **wrong — no reservoir exists yet.**
Generation happens *after* Stage 5 is implemented and qualified, so the whole chain —
pilot block, continuation block, pilot run, final run — is produced at one frozen,
qualified commit. There is no unavoidable mismatch to accommodate, and accepting one
would discard the strongest provenance guarantee available for free.

**A HEAD mismatch therefore means regeneration or requalification, not a recorded note.**
Blocks generated at an older commit are not this protocol's blocks.

**Negative cases are CONSTRUCTED, never observed.** `validate_source_provenance` is pure
and takes `porcelain` as a string, so the dirty-tree test passes `" M foo.py"` directly.
A test that dirtied the ambient worktree would pass only while the tree was dirty and
fail at the clean HEAD the protocol requires — the v18 defect that survived three
revisions.

> **Verdicts are results; exit codes are process outcomes.** A `CAPACITY_FAILURE`, a
> `NO_SHAPE_PASSES` or a `NOT_DEPLOYABLE` is a **finding**, written into the artifact and
> exiting **0** — the run did what it was asked. Only a usage error, a provenance failure
> or an abort is nonzero. Conflating the two would make an operational no-go look like a
> crash, which is exactly the framing the v18 closeout was careful to keep apart.

#### Two sidecars, because `wait $!` cannot work here

Phase 0 lost its exit code to `nohup`+`disown` with no sidecar. Revision 1 of this plan
reproduced the same defect in the fix: **after `disown`, a later shell has neither the
job table nor a usable `$!`, so `wait $!` cannot recover anything.** The launch must
therefore record the exit code itself, in the shell that owns the process:

```text
shell_status   written by a DETACHED SHELL WRAPPER around python -- `REAL_EXIT=<n>`
status.json    written by the run itself, LAST, after the artifact
```

Together they distinguish three outcomes that a single sidecar cannot:

| `shell_status` | `status.json` | meaning |
|---|---|---|
| present | present | the run finished and reported; trust `verdict` |
| present | **absent** | python died before writing — read `REAL_EXIT` and `run.log` |
| **absent** | absent | the wrapper never ran, or the machine died; nothing was measured |

`status.json` is written last precisely so its presence is itself evidence the run
reached the end, and the wrapper exists because a process killed before writing cannot
write.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_atlas_cli.py
import json
import subprocess
import sys

import pytest

from scripts.GPU.alphazero.run_atlas import (
    EXIT_ABORTED, EXIT_OK, EXIT_PROVENANCE, EXIT_USAGE, STOP_CONDITIONS,
    main as run_atlas_main, measure_provenance, write_status_sidecar,
)

from tests.eval_fakes import FakeEvaluator

BASE = 20500000
PROV = {"git_head": "a" * 40, "worktree_clean": True,
        "checkpoint_sha1": "0" * 40}


def _cli(*args):
    return subprocess.run([sys.executable, "-m",
                           "scripts.GPU.alphazero.run_atlas", *args],
                          capture_output=True, text=True)


def _fake_ck(tmp_path):
    """A real FILE, so preflight has something to hash. Never loaded: the
    factory is patched, and only its digest is used."""
    ck = tmp_path / "net.safetensors"
    ck.write_bytes(b"atlas-stage5-fake-checkpoint")
    return ck


def _fixture_prov(ck):
    """ONE provenance object for the whole fixture chain.

    Symmetric validation compares the measured digest and HEAD against the
    manifests AND the pilot artifact, so a fixture that writes "0"*40 in the
    artifact while the manifests carry the real digest is rejected before
    run-final starts -- and the test would be asserting on a provenance
    failure it did not mean to create.
    """
    import hashlib
    return {"git_head": "a" * 40, "worktree_clean": True,
            "checkpoint_path": str(ck),
            "checkpoint_sha1": hashlib.sha1(ck.read_bytes()).hexdigest()}


def _authoritative_pilot_artifact(ck, n=200):
    """A complete, sized pilot artifact as `emit` would have written one.

    Its rows are SCHEMA-VALID -- four rungs of real LegResults, a
    BoundaryRecord, and populated snapshots -- because `run-final` recomposes
    all three read-outs over them. A row with no legs and empty snapshots is
    not an authoritative pilot row; recomposition would raise on it, and the
    test would be asserting against a document the production path can never
    produce.
    """
    from scripts.GPU.alphazero.atlas_artifact import emit
    return emit({"rows": _measured_pilot_rows(),
                 "provenance": _fixture_prov(ck),      # the SAME object
                 "mode": "pilot", "verdict": "OK", "authoritative": True,
                 "sampling_seed": SAMPLING_SEED,
                 "sizing": {"p_m": 8 / 24, "p_s": 9 / 24, "verdict": "OK",
                            "N": n, "required": n}})


def _assign_artifact(tmp_path, cont_block):
    """`assign_corpus`'s real output over the real block -- not hand-written.

    `run-final` recomputes this, so a hand-written file would only ever test
    that the recomputation rejects it.
    """
    from scripts.GPU.alphazero.corpus_geometry import assign_corpus
    from scripts.GPU.alphazero.generate_atlas_reservoir import load_block
    games = load_block(cont_block, BASE, 24, 216)
    rows = assign_corpus(_pilot_assignment(), games, 200, SAMPLING_SEED)["rows"]
    p = tmp_path / "assign.json"
    p.write_text(json.dumps({"verdict": "OK", "rows": rows}, sort_keys=True))
    return p


def _patch_measured_provenance(monkeypatch, ck):
    """Patch ONLY the measurement, so every comparison still runs for real.

    Without this, a TDD run necessarily observes its own uncommitted
    implementation as a DIRTY TREE, and symmetric HEAD validation compares the
    machine's real HEAD against the fixture's "a"*40. Patching the measuring
    function -- not `measure_provenance` itself -- leaves the manifest,
    artifact and HEAD equality checks under test.
    """
    import scripts.GPU.alphazero.run_atlas as ra
    monkeypatch.setattr(ra, "preflight_source_provenance",
                        lambda path: _fixture_prov(ck))


def _fake_block(tmp_path, name="pilot", n_games=24, start_index=0,
                checkpoint=None, history=None):
    """A block directory carrying the FULL PRODUCTION manifest.

    `load_block` verifies PRODUCTION_SETTINGS -- board 24, 400 simulations, 280
    max moves, batching (14, 48, 8), noise ON -- plus clean provenance, exact
    filenames, exact index coverage and `seed == base_seed + game_idx`. A
    manifest claiming active_size=6 could never pass it, and a fixture that
    cannot pass the real loader qualifies nothing.

    Constructed, not generated: Stage 5 generates no reservoir.
    """
    from scripts.GPU.alphazero.generate_atlas_reservoir import (
        MANIFEST, seed_for_index,
    )
    ck = checkpoint or _fake_ck(tmp_path)
    prov = _fixture_prov(ck)          # the ONE fixture provenance object
    hist = history if history is not None else _late_history()
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / MANIFEST).write_text(json.dumps({
        "base_seed": BASE, "start_index": start_index, "n_games": n_games,
        "seed_range": [seed_for_index(BASE, start_index),
                       seed_for_index(BASE, start_index + n_games)],
        # The frozen production settings, verbatim.
        "n_simulations": 400, "max_moves": 280, "active_size": 24,
        "batching": [14, 48, 8], "add_noise": True,
        # Clean provenance, CONSTRUCTED -- never read from the ambient tree,
        # and the SAME object the artifact and the patched measurement use.
        **prov,
    }, indent=2, sort_keys=True))
    for i in range(start_index, start_index + n_games):
        (d / f"game_{i:06d}.json").write_text(json.dumps({
            "game_idx": i, "seed": seed_for_index(BASE, i),
            "start_player": "red", "n_moves": len(hist),
            "winner": None, "draw_reason": "state_cap",
            "move_history": [list(m) for m in hist]}))
    return d


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
                 "shell_status", "separate", "setsid"):
        assert rule in r.stdout


def test_the_runbook_launches_through_a_DETACHED_SHELL_WRAPPER():
    """Python cannot record its own exit code if it is killed, so the wrapper
    shell records it. `sh -c` runs python and writes REAL_EXIT itself."""
    out = _cli("emit-runbook").stdout
    assert "sh -c" in out
    assert 'REAL_EXIT=$?' in out and "shell_status" in out


def test_the_runbook_NEVER_tells_the_operator_to_wait_on_a_disowned_pid():
    """The Phase 0 defect, and revision 1 of this plan reproduced it: after
    `disown` a later shell has neither the job table nor a usable $!, so
    `wait $!` recovers nothing. The sidecars are read instead."""
    out = _cli("emit-runbook").stdout
    assert "wait $!" not in out
    assert "cat" in out and "status.json" in out


def test_preflight_MEASURES_provenance_instead_of_accepting_claims():
    """No --git-head / --checkpoint-sha1 / --worktree-clean arguments exist: a
    typed claim is not evidence, and an operator who mistypes a digest or
    asserts a clean tree on a dirty one would pass the gate."""
    out = _cli("preflight", "--help").stdout
    for claim in ("--git-head", "--checkpoint-sha1", "--worktree-clean"):
        assert claim not in out
    assert "--checkpoint" in out          # the FILE, which preflight hashes


def test_the_dirty_tree_case_is_CONSTRUCTED_not_observed():
    """validate_source_provenance is pure and takes porcelain as a string, so
    the negative is built rather than made by dirtying the ambient worktree --
    which would pass only while the tree was dirty and fail at the clean HEAD
    the protocol requires."""
    from scripts.GPU.alphazero.generate_atlas_reservoir import (
        validate_source_provenance,
    )
    with pytest.raises(RuntimeError, match="dirty"):
        validate_source_provenance(porcelain=" M scripts/GPU/alphazero/mcts.py",
                                   git_head="a" * 40, checkpoint_path="ck",
                                   checkpoint_sha1="0" * 40)
    # ...and the clean case passes on exactly the same code path.
    ok = validate_source_provenance(porcelain="", git_head="a" * 40,
                                    checkpoint_path="ck",
                                    checkpoint_sha1="0" * 40)
    assert ok["worktree_clean"] is True


def test_preflight_rejects_a_checkpoint_that_disagrees_with_the_manifest(
        tmp_path, monkeypatch):
    """The run must not use a different network from the one that generated its
    games. Constructed: two files with different digests."""
    import scripts.GPU.alphazero.run_atlas as ra
    ck = _fake_ck(tmp_path)
    # Patch the MEASUREMENT only: a TDD run has uncommitted code, so ambient
    # provenance would raise "dirty worktree" long before the comparison runs.
    monkeypatch.setattr(ra, "preflight_source_provenance",
                        lambda path: _fixture_prov(ck))
    block = _fake_block(tmp_path, checkpoint=ck)
    manifest = json.loads((block / "block_manifest.json").read_text())
    (block / "block_manifest.json").write_text(
        json.dumps({**manifest, "checkpoint_sha1": "f" * 40}))
    with pytest.raises(ValueError, match="checkpoint_sha1"):
        ra.measure_provenance(str(ck), pilot_dir=str(block))


def test_preflight_rejects_an_unreadable_corpus_artifact(tmp_path):
    r = _cli("preflight", "--checkpoint", str(tmp_path / "no.safetensors"),
             "--corpus-artifact", str(tmp_path / "nope.json"))
    assert r.returncode == EXIT_USAGE


def test_the_status_sidecar_records_the_verdict_and_the_exit_code(tmp_path):
    p = tmp_path / "status.json"
    write_status_sidecar(p, verdict="OK", exit_code=0, rows=240)
    d = json.loads(p.read_text())
    assert d["verdict"] == "OK" and d["exit_code"] == 0 and d["rows"] == 240


def test_no_zero_gpu_subcommand_constructs_an_evaluator():
    """preflight and emit-runbook are zero-GPU. Only the two run modes build an
    evaluator, and they import the factory lazily inside those branches."""
    import scripts.GPU.alphazero.run_atlas as mod
    src = open(mod.__file__).read()
    assert "_default_evaluator_factory" in src
    assert src.index("def _cmd_run_pilot") < src.index("_default_evaluator_factory")


# -- the launchable seam, driven once per mode --------------------------------

def _patch_factory(monkeypatch):
    """Patch the ONE factory the run modes import lazily. Everything else in
    the path is real: real assignment, real ladder, real artifact, real
    sidecars."""
    import scripts.GPU.alphazero.eval_runner as er
    monkeypatch.setattr(er, "_default_evaluator_factory",
                        lambda _p: FakeEvaluator(value=0.0), raising=True)


def _cheap_budgets(monkeypatch):
    """Reduced budgets are INTERNAL TEST INJECTION at the module seam, never
    CLI flags: the board, the replay budget and the ladder are frozen, and an
    operator must not be able to change them from the command line."""
    import scripts.GPU.alphazero.atlas_run as ar
    monkeypatch.setattr(ar, "PREFIX_SIMS", 2)
    monkeypatch.setattr(ar, "LEG_INCREMENTS_DEFAULT", (80, 80, 80, 80))
    monkeypatch.setattr(ar, "BOUNDARY_THRESHOLD_DEFAULT", 40)
    monkeypatch.setattr(ar, "LEG_B_DEFAULT", 80)


def test_the_production_parser_exposes_NO_frozen_parameter():
    """The board, the replay budget and the ladder are frozen. A flag that can
    change them is a protocol change with a command-line interface."""
    for sub in ("run-pilot", "run-final"):
        out = _cli(sub, "--help").stdout
        for leaked in ("--active-size", "--prefix-sims", "--tiny-legs",
                       "--increments", "--threshold", "--leg-b",
                       "--n-target", "--n-simulations"):
            assert leaked not in out, f"{sub} exposes {leaked}"


def test_run_pilot_end_to_end_with_a_patched_factory(tmp_path, monkeypatch):
    """The only entry point the operator can launch, actually launched --
    against real blocks, a real artifact and both real sidecars."""
    _patch_factory(monkeypatch)
    _cheap_budgets(monkeypatch)
    ck = _fake_ck(tmp_path)
    _patch_measured_provenance(monkeypatch, ck)
    out = tmp_path / "out"
    rc = run_atlas_main(["run-pilot",
                         "--pilot-dir", str(_fake_block(tmp_path, checkpoint=ck)),
                         "--base-seed", str(BASE),
                         "--sampling-seed", "20260805",
                         "--checkpoint", str(ck), "--out-dir", str(out)])
    assert rc == EXIT_OK
    doc = json.loads((out / "pilot_artifact.json").read_text())
    assert doc["mode"] == "pilot" and doc["verdict"] == "OK"
    assert len(doc["rows"]) == 24
    assert doc["sizing"]["verdict"] in {"OK", "PROJECTED_CAPACITY_NO_GO"}
    status = json.loads((out / "status.json").read_text())
    assert status["verdict"] == "OK" and status["exit_code"] == EXIT_OK
    # ...and the artifact it wrote is RELOADABLE, which is what run-final needs.
    from scripts.GPU.alphazero.atlas_artifact import load_run
    assert len(load_run(out / "pilot_artifact.json")["rows"]) == 24


def _final_argv(tmp_path, ck, out, pilot):
    """The frozen production argument set -- no budget flags exist to pass."""
    cont_block = _fake_block(tmp_path, name="cont", n_games=216,
                             start_index=24, checkpoint=ck)   # G_total - 24
    return ["run-final", "--pilot-artifact", str(pilot),
            "--corpus-artifact", str(_assign_artifact(tmp_path, cont_block)),
            "--pilot-dir", str(_fake_block(tmp_path, checkpoint=ck)),
            "--continuation-dir", str(cont_block),
            "--base-seed", str(BASE), "--checkpoint", str(ck),
            "--out-dir", str(out)]


def test_run_final_SUCCESS_path_end_to_end(tmp_path, monkeypatch):
    """The successful final run, qualified without a single ladder.

    `run_corpus` -- the expensive half -- is patched to return a schema-valid
    COMPLETE 176-row document at the pilot-produced N=200. Everything else is
    real: the pilot artifact is loaded and authenticated, the assignment is
    recomputed, the carry and recomposition run, the artifact is emitted, and
    the success sidecar is written. No frozen parameter is relaxed and no CLI
    budget flag exists, so the first production run stays evidence.
    """
    _patch_factory(monkeypatch)
    _patch_measured_provenance(monkeypatch, _fake_ck(tmp_path))
    _stub_measurement(monkeypatch, complete=True)
    ck, out = _fake_ck(tmp_path), tmp_path / "out"
    pilot = tmp_path / "pilot_artifact.json"
    pilot.write_text(_authoritative_pilot_artifact(ck, n=200))

    rc = run_atlas_main(_final_argv(tmp_path, ck, out, pilot))
    assert rc == EXIT_OK
    doc = json.loads((out / "atlas_artifact.json").read_text())
    assert doc["n_target"] == 200 and doc["measured"] == 200
    assert doc["pilot_rows_carried"] == 24
    assert doc["authoritative"] is True
    status = json.loads((out / "status.json").read_text())
    assert status["verdict"] == "OK" and status["exit_code"] == EXIT_OK


def test_run_final_ABORTED_path_end_to_end(tmp_path, monkeypatch):
    """Same path, one unmeasured position: exit 5 and non-authoritative."""
    _patch_factory(monkeypatch)
    _patch_measured_provenance(monkeypatch, _fake_ck(tmp_path))
    _stub_measurement(monkeypatch, complete=False)
    ck, out = _fake_ck(tmp_path), tmp_path / "out"
    pilot = tmp_path / "pilot_artifact.json"
    pilot.write_text(_authoritative_pilot_artifact(ck, n=200))

    rc = run_atlas_main(_final_argv(tmp_path, ck, out, pilot))
    assert rc == EXIT_ABORTED
    doc = json.loads((out / "atlas_artifact.json").read_text())
    assert doc["authoritative"] is False and doc["failed_rows"]
    status = json.loads((out / "status.json").read_text())
    assert status["verdict"] == "ABORTED" and status["exit_code"] == EXIT_ABORTED


def test_a_HEAD_mismatch_is_refused_symmetrically(tmp_path, monkeypatch):
    """The chain is produced at ONE qualified commit, so a HEAD that differs
    from a manifest is regeneration or requalification -- not a note."""
    import hashlib
    import scripts.GPU.alphazero.run_atlas as ra
    ck = _fake_ck(tmp_path)
    monkeypatch.setattr(ra, "preflight_source_provenance", lambda path: {
        "git_head": "b" * 40, "worktree_clean": True,      # fixtures say "a"*40
        "checkpoint_path": str(path),
        "checkpoint_sha1": hashlib.sha1(ck.read_bytes()).hexdigest()})
    with pytest.raises(ValueError, match="git_head"):
        ra.measure_provenance(str(ck),
                              pilot_dir=str(_fake_block(tmp_path, checkpoint=ck)))


def test_the_emitted_launch_wrapper_ACTUALLY_records_the_exit_code(tmp_path):
    """Executed, not asserted about.

    A substring check cannot catch a redirection-order defect -- the previous
    revision's wrapper read plausibly and left the sidecar empty. So the
    wrapper is built for a harmless command that exits 3, run, and the file is
    read back.
    """
    from scripts.GPU.alphazero.run_atlas import launch_wrapper
    out = tmp_path / "out"; out.mkdir()
    script = launch_wrapper([sys.executable, "-c", "raise SystemExit(3)"],
                            out_dir=out)
    rc = subprocess.run(["sh", "-c", script]).returncode
    assert (out / "shell_status").read_text().strip() == "REAL_EXIT=3"
    assert rc == 3                      # the wrapper exits with the real code
    assert (out / "run.log").exists()   # ...and the log was still captured
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
import shlex

EXIT_OK, EXIT_USAGE, EXIT_PROVENANCE, EXIT_ABORTED = 0, 2, 3, 5

# Verdict, the module that raises it, the operator action, and the exit code.
# A read-out verdict is a RESULT and exits 0; only process failures are nonzero.
STOP_CONDITIONS = ( ... )        # the table below, as dicts with those keys


def measure_provenance(checkpoint: str, *, pilot_dir=None,
                       continuation_dir=None, pilot_artifact=None) -> dict:
    """MEASURE the tree, HEAD and the checkpoint -- never accept them as args.

    Runs before any evaluator exists, because checking afterwards means a dirty
    tree can consume an entire GPU run before anything rejects it.
    """
    def _sources(pilot_dir, continuation_dir, pilot_artifact):
        """Everything the run must agree with, named for the error message."""
        for label, d in (("pilot block", pilot_dir),
                         ("continuation block", continuation_dir)):
            if d:
                yield label, load_manifest(d)
        if pilot_artifact:
            yield "pilot artifact", pilot_artifact["provenance"]

    prov = preflight_source_provenance(checkpoint)       # git + sha1, measured
    # SYMMETRIC: the whole chain is produced at one frozen qualified commit, so
    # both the digest and the HEAD must match everywhere. A mismatch means
    # regeneration or requalification, not a recorded note.
    for name, recorded in _sources(pilot_dir, continuation_dir, pilot_artifact):
        for field in ("checkpoint_sha1", "git_head"):
            if recorded.get(field) != prov[field]:
                raise ValueError(
                    f"{field} mismatch against {name}: measured "
                    f"{prov[field]} != recorded {recorded.get(field)}. The "
                    f"chain must be produced at ONE qualified commit; "
                    f"regenerate or requalify rather than proceeding.")
    if pilot_dir and continuation_dir:
        assert_blocks_agree(load_manifest(pilot_dir),
                            load_manifest(continuation_dir))
    return prov


def launch_wrapper(argv: Sequence[str], *, out_dir) -> str:
    """The detached wrapper body, as a string a shell can run.

    A FUNCTION, not runbook prose, so a test can build a harmless one, EXECUTE
    it, and read the sidecar back. A substring assertion cannot catch a
    redirection-order defect: the previous revision's wrapper read plausibly
    and left the file empty.

    `rc` is captured FIRST, then written, then re-raised as the wrapper's own
    exit status. Nothing about the ordering is left to redirection precedence.
    """
    q = " ".join(shlex.quote(a) for a in argv)
    out = shlex.quote(str(out_dir))
    return (f"{q} > {out}/run.log 2>&1\n"
            f"rc=$?\n"
            f"echo \"REAL_EXIT=$rc\" > {out}/shell_status\n"
            f"exit $rc\n")


def write_status_sidecar(path, *, verdict: str, exit_code: int, **extra) -> None:
    """Written LAST, after the artifact, so its presence is itself evidence the
    run reached the end. It does NOT replace the wrapper shell's REAL_EXIT: a
    process killed before it can write writes nothing.
    """


def _cmd_preflight(args) -> int: ...
def _cmd_emit_runbook(args) -> int: ...


def _cmd_run_pilot(args) -> int:
    """Launchable. Stage 5 writes it, drives it once against a patched factory,
    and never executes it for real."""
    prov = measure_provenance(args.checkpoint, pilot_dir=args.pilot_dir)
    from .eval_runner import _default_evaluator_factory      # lazy: MLX
    evaluator = _default_evaluator_factory(args.checkpoint)  # ONCE per run
    ...


def _cmd_run_final(args) -> int:
    """Same shape, plus the pilot artifact as an INPUT."""
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
| row `failure` → `ABORTED` | `atlas_run.run_corpus` | **5** | The corpus is exactly `N` assigned positions, so **one** unmeasured position disqualifies the run. Failures and partial rows are retained and the read-outs are marked non-authoritative. **No failure-tolerance number exists** — completeness is binary. |
| `UNAVAILABLE` sizing | `atlas_run.run_pilot` | **5** | An aborted pilot does not size and does not close widening. Re-run the pilot; do not carry a partial `N`. |

`emit-runbook` prints the operator sequence, with the recorded §9 rules inline:

```text
OPERATOR STOP -- this tool has not been authorized to run the atlas.

 A. PILOT  (N is the pilot's OUTPUT, not an input)
 1. Clean tree. Preflight MEASURES it -- do not assert it:
      run_atlas preflight --checkpoint <net> --pilot-dir <pilot_block>
    exit 0 continue | 2 usage | 3 PROVENANCE_FAILURE -- stop, fix, restart.
 2. Geometry gate already passed at build_atlas_corpus pilot-gate; a
    PHASE_GEOMETRY_NO_GO here means stop, NOT a smaller pilot.
 3. Launch the 24-row pilot (see LAUNCH below), then read its artifact:
      .sizing.verdict     OK -> N ; PROJECTED_CAPACITY_NO_GO -> stop
      .early_widening_check.both_fail  true -> close progressive widening
                                       WITHOUT inventing another shape
 B. CONTINUATION
 4. Generate exactly G_total-24 games, then assign:
      build_atlas_corpus assign ... --n-target <N>
    exit 3 PHASE_GEOMETRY_NO_GO | 4 ASSIGNMENT_SHORTFALL -- stop. No top-up,
    no cell rebalance, no moving pilot rows.
 5. Project runtime from the corpus's OBSERVED per-phase ply supply:
      run_atlas_ladder project-runtime --rows <N> --mean-prefix-plies <measured>
    Never scale a smoke to estimate runtime.
 6. Launch the final run (LAUNCH below) with --pilot-artifact: the pilot's 24
    discovery rows are CARRIED, never re-measured, and 24 + (N-24) must be N.

 LAUNCH -- always in a shell invocation of its own
 7. `setsid` does not exist on macOS, and a tool timeout SIGTERMs the whole
    process group when the launch and the wait share one call. Launch through a
    DETACHED SHELL WRAPPER so the shell -- not python -- records the exit code:

    `run_atlas emit-runbook --out-dir <dir> ...` prints the exact wrapper for
    your arguments; it is generated by `launch_wrapper`, which is the same code
    the tests execute. Its shape:

      OUT=<out_dir>; mkdir -p "$OUT"
      cat > "$OUT/launch.sh" <<'EOF'
      .venv/bin/python -m scripts.GPU.alphazero.run_atlas run-final \
        --pilot-artifact <pilot.json> --corpus-artifact <assign.json> \
        --pilot-dir <pilot_block> --continuation-dir <cont_block> \
        --base-seed <n> --checkpoint <net> --out-dir <dir> \
        > <dir>/run.log 2>&1
      rc=$?
      echo "REAL_EXIT=$rc" > <dir>/shell_status
      exit $rc
      EOF
      nohup sh "$OUT/launch.sh" > /dev/null 2>&1 &
      disown

    `rc` is captured BEFORE anything else writes, so no redirection can eat it.

 8. In a LATER call, read the two sidecars. Do NOT use `wait $!`: after
    `disown` a later shell has neither the job table nor a usable $!, which is
    exactly how Phase 0 lost its exit code.
      cat "$OUT/shell_status"     # REAL_EXIT=<n>, written by the wrapper shell
      cat "$OUT/status.json"      # verdict + exit_code, written LAST by the run

      both present            -> trust .verdict
      shell_status only       -> python died before reporting; read run.log
      neither                 -> the wrapper never ran; nothing was measured

    `cmd | tail` reports the PIPE's exit code -- redirect to a file instead.
 9. Exit 5 / verdict ABORTED means the corpus was not measured completely. The
    read-outs in the artifact are marked non-authoritative and may be used for
    diagnosis only.
10. No source edit after preflight, and no commit between generation and
    qualification.
```

- [ ] **Step 4: Run to verify it passes, then the full suite**

Expected: PASS — 18 passed.

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
- [ ] **§3's chronology is executable.** `run-pilot` runs the fixed 24 discovery rows,
      `size_from_pilot` produces `N`, and §8's early static widening check is reported;
      `run-final` **carries** the pilot's 24 rows from the pilot artifact rather than
      re-deriving them and asserts `24 + (N − 24) == N` exactly.
- [ ] **One evaluator for the whole run, one seeded `MCTS` per row**, continued across
      the prefix and all four legs and never reseeded. `run_corpus` has **no** factory or
      checkpoint parameter, so rebuilding a compiled evaluator is unreachable.
- [ ] A verified-seed mismatch, a missing boundary, or a backup-invariant violation
      **fails that row with a recorded reason and no partial row**; every other row still
      runs; inheritance resets **keep** the row.
- [ ] **Any row failure makes the run `ABORTED` and non-authoritative, exit 5.** The
      corpus is exactly `N` assigned positions, so a surviving subset is not the atlas.
      Failures and partial rows are retained, the read-outs still run, and the document
      says plainly that they are not authoritative. **No failure-tolerance number exists**
      — completeness is binary, so there is nothing to tune.
- [ ] **Provenance is MEASURED before the evaluator exists**: the actual worktree, the
      actual HEAD, the actual checkpoint hashed, then compared against both block
      manifests and the pilot artifact. No `--git-head` / `--checkpoint-sha1` /
      `--worktree-clean` argument exists. The dirty-tree negative is **constructed**
      through the pure `validate_source_provenance`, never observed from ambient state.
- [ ] **The runbook launches through a detached shell wrapper** that writes
      `shell_status`, and tells the operator to read the two sidecars rather than
      `wait $!` — which cannot work on a disowned PID and is the defect Phase 0 already
      paid for.
- [ ] **The pilot artifact is RELOADABLE.** `load_run` is the authenticated inverse of
      `emit`, rehydrating dataclasses, tuple paths and integer keys, and is qualified by
      an **emit → disk → load → all three read-outs** round trip. Nothing consumes an
      artifact except through it.
- [ ] **An aborted pilot does not size and does not close widening**: `sizing` is
      `UNAVAILABLE` with `N: None`, `early_widening_check_authoritative` is false,
      partial read-outs are preserved, exit 5.
- [ ] **The early widening check uses the frozen three-way precedence.** `both_fail`
      fires only on two genuine `FAIL`s, so a sparse pilot cannot close progressive
      widening on an absence of evidence.
- [ ] **The production parser exposes no frozen parameter** — no `--active-size`,
      `--prefix-sims`, `--tiny-legs`, `--increments` or `--n-target` — with a test
      proving it. Reduced budgets are internal test injection at the module seam.
- [ ] **The launch wrapper is a tested function.** `launch_wrapper` captures `rc` before
      writing and re-raises it, and the test **executes** a harmless emitted wrapper and
      reads `shell_status` back. Substring assertions are explicitly insufficient here.
- [ ] **Both launchable commands are driven end to end** against a patched
      `FakeEvaluator` factory, over real production-schema block directories,
      assignments, artifacts and both sidecars — including the `ABORTED` path.

- [ ] **The assignment artifact is RECOMPUTED, not trusted**: `sampling_seed` comes from
      the pilot artifact, the **complete** `G_total − 24` continuation block is loaded,
      `size_continuation` and `assign_corpus` are re-derived, and exact equality with the
      artifact's rows is required. The fixtures keep `G_total − 24 = 216` distinct from
      `N − 24 = 176`, so an assignment that selected everything would fail.
- [ ] **The successful `run-final` is qualified without GPU work.** `combine_final_runs`
      is pure and separately tested, and the CLI success path runs at the real frozen
      `N = 200` with `run_corpus` patched to a schema-valid complete 176-row document —
      no ladder, no budget override, no CLI flag. **The first production run therefore
      remains evidence** rather than becoming a disposable qualification run.
- [ ] **HEAD identity is symmetric**: the measured `git_head` must equal both block
      manifests and the pilot artifact. A mismatch means regeneration or requalification.
- [ ] **CLI tests patch the measured-provenance boundary only.** Otherwise a TDD run
      observes its own uncommitted implementation as a dirty tree and compares the real
      HEAD against the fixtures' `"a"*40`. Every comparison the gate performs stays under
      test; only the measurement is stubbed.
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
- [ ] **Test counting.** Planned: row-facts 8, artifact +5, run 37, CLI 18 = **68**
      (29 → 45 → 55 → 61 → 65 → 68; each superseded, never adjusted). Stage 4 measured
      **2556**, so the expected total is **2624**. Recount from `def test_` on disk at qualification
      and **baseline against a measured collect**,
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
