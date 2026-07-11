# Context-Relative FPU (Policy-Mass) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the *tooling* for the context-relative FPU successor — the opt-in parent-relative + explored-policy-mass rule, an optional read-only search-trace observer, a geometry development-corpus builder, and a mode-enforced discovery diagnostic — all verified, with the shipped absolute-FPU path proven byte-identical. The heavy MCTS phases stay operator-run.

**Architecture:** One small `mcts.py` change (a new opt-in `MCTSConfig` field + `_select_child` branch + a guarded `_backup` observer hook + a pure helper), plus two new operator-run scripts (corpus builder, discovery diagnostic) and a shared complete-state hash. Everything MCTS-heavy is behind opt-in flags and is exercised only in operator phases; unit tests use pure functions, fakes, and synthetic trees — no MCTS in tests.

**Tech Stack:** Python 3 stdlib (dataclasses, math, hashlib, json, csv, argparse, random). No numpy in new pure code. Tests via `.venv/bin/python -m pytest -p no:cacheprovider`. Full spec: `docs/superpowers/specs/2026-07-10-context-relative-fpu-policy-mass-design.md` (frozen protocol; this plan implements §A rule, §B builder, §C diagnostic, §4 observer, §6 gates).

## Global Constraints

- **Byte-identical off:** `fpu_policy_mass_reduction=None` **and** observer `None` reproduce the pre-branch code exactly. The full existing suite passes unchanged; a synthetic `old==new` selection-trace proof is included (Task 8).
- **`0.0` ≠ `None`:** `fpu_policy_mass_reduction=0.0` is an *enabled* mode (`FPU = Q_parent`); `None` keeps the absolute `fpu_value` path. Tested explicitly.
- **Mutual exclusion:** raise if `fpu_policy_mass_reduction is not None` and `fpu_value != 0.0`; raise if the reduction is non-finite or `< 0`.
- **Completed visits only:** `P_explored` counts a child only when `child.visit_count > 0` is a completed backed-up visit (never virtual/pending). A test asserts a pending/zero-visit child contributes 0.
- **Observer isolation:** read-only; no effect on selection/eval/RNG/batching/backup; exceptions abort the diagnostic; reuse the MCTS leader comparator (`_best_child`) — no duplicated tie-break.
- **Do NOT modify** `self_play.py`, `SIMS_TABLE`, trainer, network, promotion, calibration manifests, value-adapter/projection. No self-play adoption.
- **Do NOT run** the operator phases: seed20116 geometry scan, corpus generation, coefficient sweep, selection, frozen-check, held-out validation. Ship tooling only.
- **Tests:** pure/fakes/synthetic; run from repo root with `.venv/bin/python -m pytest`.
- **All numeric gate thresholds are the frozen §6 values.** The only control-run-substituted values are `r0_target_lockin_count` / `absoff_target_lockin_count` into the `baseline + 2` caps.

## File structure

- **Modify** `scripts/GPU/alphazero/mcts.py` — `MCTSConfig.fpu_policy_mass_reduction` + `__post_init__` guard; module-level `policy_mass_fpu` helper; `_select_child` one-pass `P_explored` + branch; `MCTS.__init__(observer=None)` + guarded `_backup` call.
- **Create** `scripts/GPU/alphazero/fpu_state_hash.py` — `canonical_state_key`/`canonical_state_sha1` (complete-state hash).
- **Create** `scripts/GPU/alphazero/build_fpu_dev_corpus.py` — pure sampler + two-stage confirm-under-anchor shell + disjointness (operator-run).
- **Create** `scripts/GPU/alphazero/diagnose_fpu_policy_mass.py` — `FpuTraceObserver`, mode-enforced discovery diagnostic, dual-reference metrics + §6 gates (operator-run).
- **Create tests** `tests/test_fpu_policy_mass_rule.py`, `tests/test_fpu_state_hash.py`, `tests/test_fpu_dev_corpus.py`, `tests/test_fpu_trace_observer.py`, `tests/test_fpu_diagnostic_modes.py`.

---

## Task 1: Rule helper + config field + guard

**Files:** Modify `scripts/GPU/alphazero/mcts.py`; Test `tests/test_fpu_policy_mass_rule.py` (create).

**Interfaces (produces):** `policy_mass_fpu(parent_q, explored_mass, r) -> float`; `MCTSConfig.fpu_policy_mass_reduction: float | None = None` with `__post_init__` validation.

- [ ] **Step 1: Write the failing tests** — create `tests/test_fpu_policy_mass_rule.py`:

```python
import math
import dataclasses
import pytest
from scripts.GPU.alphazero.mcts import MCTSConfig, policy_mass_fpu


def test_policy_mass_fpu_formula_and_clamp():
    assert policy_mass_fpu(0.0, 0.0, 0.20) == 0.0                     # no mass -> parent value
    assert abs(policy_mass_fpu(0.0, 1.0, 0.20) - (-0.20)) < 1e-12     # full mass -> full reduction
    assert abs(policy_mass_fpu(0.3, 0.25, 0.20) - (0.3 - 0.20*0.5)) < 1e-12
    assert policy_mass_fpu(0.1, 5.0, 0.20) == policy_mass_fpu(0.1, 1.0, 0.20)   # clamp high
    assert policy_mass_fpu(0.1, -3.0, 0.20) == 0.1                    # clamp low -> sqrt(0)


def test_policy_mass_fpu_rejects_nonfinite():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            policy_mass_fpu(bad, 0.5, 0.2)
        with pytest.raises(ValueError):
            policy_mass_fpu(0.0, bad, 0.2)
        with pytest.raises(ValueError):
            policy_mass_fpu(0.0, 0.5, bad)


def test_config_default_is_none_absolute_path():
    assert MCTSConfig().fpu_policy_mass_reduction is None
    assert MCTSConfig().fpu_value == 0.0


def test_config_zero_is_enabled_mode_not_none():
    c = MCTSConfig(fpu_policy_mass_reduction=0.0)
    assert c.fpu_policy_mass_reduction == 0.0 and c.fpu_policy_mass_reduction is not None


def test_config_mutual_exclusion_and_bounds():
    with pytest.raises(ValueError):
        MCTSConfig(fpu_value=-0.2, fpu_policy_mass_reduction=0.10)     # absolute + relative
    with pytest.raises(ValueError):
        MCTSConfig(fpu_policy_mass_reduction=-0.1)                     # negative
    with pytest.raises(ValueError):
        MCTSConfig(fpu_policy_mass_reduction=float("inf"))             # nonfinite
    MCTSConfig(fpu_value=0.0, fpu_policy_mass_reduction=0.20)          # ok (fpu_value at default)
    dataclasses.replace(MCTSConfig(), fpu_policy_mass_reduction=0.35)  # replace path ok
```

- [ ] **Step 2: Run to verify fail** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_policy_mass_rule.py -q`. Expected: FAIL (`cannot import name 'policy_mass_fpu'`).

- [ ] **Step 3: Add the helper** — in `mcts.py` (module level, near the top after imports; `math` is already imported):

```python
def policy_mass_fpu(parent_q: float, explored_mass: float, r: float) -> float:
    """Parent-relative FPU with explored-policy-mass scaling (design §A).
    FPU = parent_q - r*sqrt(clamp(explored_mass, 0, 1)). Rejects non-finite
    inputs (a NaN mass passes both clamp comparisons and would propagate)."""
    if not (math.isfinite(parent_q) and math.isfinite(explored_mass) and math.isfinite(r)):
        raise ValueError("policy_mass_fpu requires finite inputs")
    m = 0.0 if explored_mass < 0.0 else (1.0 if explored_mass > 1.0 else explored_mass)
    return parent_q - r * math.sqrt(m)
```

- [ ] **Step 4: Add the field + guard** — in `MCTSConfig` (after `fpu_value`, line ~100):

```python
    fpu_policy_mass_reduction: float | None = None  # None => absolute fpu_value path
                                                    # (byte-identical). Not None =>
                                                    # FPU = Q_parent - r*sqrt(P_explored);
                                                    # 0.0 is an ENABLED mode (FPU=Q_parent).
```

Extend `__post_init__` (line ~147):

```python
        if self.fpu_policy_mass_reduction is not None:
            if self.fpu_value != 0.0:
                raise ValueError(
                    "fpu_policy_mass_reduction and a nonzero absolute fpu_value are "
                    "mutually exclusive")
            r = self.fpu_policy_mass_reduction
            if not math.isfinite(r) or r < 0:
                raise ValueError("fpu_policy_mass_reduction must be finite and >= 0")
```

- [ ] **Step 5: Run tests** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_policy_mass_rule.py -q`. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/mcts.py tests/test_fpu_policy_mass_rule.py
git commit -m "feat(fpu): policy-mass FPU helper + opt-in config field + mutual-exclusion guard"
```

---

## Task 2: Wire the rule into `_select_child` (byte-identical off)

**Files:** Modify `scripts/GPU/alphazero/mcts.py`; Test `tests/test_fpu_policy_mass_rule.py`.

**Interfaces:** `_select_child` computes `P_explored` once (completed-visit children only) and, when the mode is enabled, uses `policy_mass_fpu(node.q_value, P_explored, r)` as the unvisited-child assumed value; `None` keeps `q = self.config.fpu_value` unchanged.

- [ ] **Step 1: Write the failing tests** — append:

```python
from scripts.GPU.alphazero.mcts import MCTS, MCTSNode, encode_move


def _node_with_children(priors, visited):
    """priors: {(r,c): P}; visited: set of (r,c) with a completed visit."""
    n = MCTSNode(state=None, visit_count=3, value_sum=0.9)   # q_value = 0.3
    n.priors = {encode_move(*rc): p for rc, p in priors.items()}
    for rc, p in priors.items():
        mid = encode_move(*rc)
        vc = 5 if rc in visited else 0
        n.children[mid] = MCTSNode(state=None, parent=n, move=mid,
                                   visit_count=vc, value_sum=0.0)
    return n


def _explored_mass_via_selectchild(cfg, node):
    """Drive the private P_explored computation through a tiny MCTS with a fake
    evaluator; we assert on the assumed value it produces for an unvisited child."""
    # Rather than reach into privates, we recompute what _select_child must use:
    from scripts.GPU.alphazero.mcts import policy_mass_fpu
    mass = sum(p for mid, p in node.priors.items()
               if node.children[mid].visit_count > 0)
    return mass


def test_p_explored_counts_completed_visits_only():
    node = _node_with_children({(1, 1): 0.5, (2, 2): 0.3, (3, 3): 0.2}, visited={(1, 1)})
    # pending/virtual must NOT count: give (2,2) a pending marker but visit_count 0
    assert _explored_mass_via_selectchild(MCTSConfig(fpu_policy_mass_reduction=0.2), node) == 0.5


def test_selectchild_absolute_path_byte_identical_when_none():
    # With None, the unvisited assumed value is exactly config.fpu_value (0.0),
    # independent of parent q or explored mass.
    cfg_none = MCTSConfig(fpu_value=0.0)                       # None default
    assert cfg_none.fpu_policy_mass_reduction is None
    # (Full byte-identical selection-trace proof is Task 8; here we assert the
    # branch condition: None => fpu_value path.)
    assert cfg_none.fpu_value == 0.0
```

(Note: `_select_child` is exercised end-to-end in Task 8's synthetic-tree trace; Task 2 asserts the `P_explored` definition and the branch condition directly.)

- [ ] **Step 2: Run to verify fail** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_policy_mass_rule.py -q -k "p_explored or byte_identical_when_none"`. Expected: the `p_explored` test drives the same summation the implementation will use; it passes as written (it recomputes locally) — so instead assert against the real code path by importing a small extracted helper. Replace `_explored_mass_via_selectchild` to call a new `mcts.explored_policy_mass(node)` (created in Step 3); rerun → FAIL until Step 3 adds it.

- [ ] **Step 3: Implement** — add a tiny pure helper + wire `_select_child`. Add near `policy_mass_fpu`:

```python
def explored_policy_mass(node) -> float:
    """Σ prior over children with a COMPLETED (backed-up) visit. Virtual/pending
    visits do not affect child.visit_count, so they are excluded (design §A)."""
    total = 0.0
    for move_id, prior in node.priors.items():
        child = node.children.get(move_id)
        if child is not None and child.visit_count > 0:
            total += prior
    return total
```

In `_select_child`, immediately after `sqrt_parent = math.sqrt(node.visit_count + 1)` (line ~949), add:

```python
        _pm = self.config.fpu_policy_mass_reduction
        if _pm is not None:
            _fpu_pm = policy_mass_fpu(node.q_value, explored_policy_mass(node), _pm)
```

Change the unvisited branch (currently `q = self.config.fpu_value`) to:

```python
            else:
                q = _fpu_pm if _pm is not None else self.config.fpu_value
                child_visits = 0
```

(When `_pm is None`, the expression is exactly `self.config.fpu_value` — no new work, byte-identical. Update the test helper to call `mcts.explored_policy_mass`.)

- [ ] **Step 4: Run tests** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_policy_mass_rule.py -q`. Expected: PASS.

- [ ] **Step 5: Existing MCTS tests still pass (off-path)** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_value.py tests/test_fpu_sweep.py -q`. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/mcts.py tests/test_fpu_policy_mass_rule.py
git commit -m "feat(fpu): _select_child policy-mass branch + completed-visit P_explored (None=byte-identical)"
```

---

## Task 3: Optional read-only trace observer hook

**Files:** Modify `scripts/GPU/alphazero/mcts.py`; Test `tests/test_fpu_trace_observer.py` (create).

**Interfaces:** `MCTS.__init__(..., observer=None)` stores `self._observer`; a monotonic completed-sim counter; a single guarded call at the end of `_backup`: `self._observer.on_root_simulation(count, root, updated_root_move)` where `root = search_path[0]`, `updated_root_move = search_path[1].move if len(search_path) >= 2 else None`. Observer `None` → no call → byte-identical.

- [ ] **Step 1: Write the failing test** — create `tests/test_fpu_trace_observer.py`:

```python
from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig, MCTSNode, encode_move


class _SpyObserver:
    def __init__(self):
        self.calls = []
    def on_root_simulation(self, count, root, updated_root_move):
        self.calls.append((count, id(root), updated_root_move))


def _mcts_with_spy(spy):
    m = MCTS.__new__(MCTS)          # bypass heavy __init__ deps for a unit probe
    m.config = MCTSConfig()
    m._observer = spy
    m._backup_sim_counter = 0
    m._total_backups = 0
    return m


def test_backup_fires_one_event_per_completed_sim_with_root_move():
    spy = _SpyObserver()
    m = _mcts_with_spy(spy)
    root = MCTSNode(state=None, visit_count=0)
    child = MCTSNode(state=None, parent=root, move=encode_move(3, 4), visit_count=0)
    # depth-2 path: root -> child -> leaf
    leaf = MCTSNode(state=None, parent=child, move=encode_move(5, 6), visit_count=0)
    m._backup([root, child, leaf], 0.5)
    assert spy.calls == [(1, id(root), encode_move(3, 4))]


def test_backup_root_only_path_passes_none_root_move_and_advances_counter():
    spy = _SpyObserver()
    m = _mcts_with_spy(spy)
    root = MCTSNode(state=None, visit_count=0)
    m._backup([root], 0.5)                        # root-only (first sim edge case)
    assert spy.calls == [(1, id(root), None)]


def test_observer_none_no_effect():
    m = _mcts_with_spy(None)
    root = MCTSNode(state=None, visit_count=0)
    child = MCTSNode(state=None, parent=root, move=encode_move(1, 1), visit_count=0)
    m._backup([root, child], 0.5)                 # must not raise; no observer
    assert child.visit_count == 1 and root.visit_count == 1
```

- [ ] **Step 2: Run to verify fail** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_trace_observer.py -q`. Expected: FAIL (`_backup` doesn't call observer / attrs missing).

- [ ] **Step 3: Implement** — in `MCTS.__init__`, add param `observer=None`, store `self._observer = observer` and `self._backup_sim_counter = 0` (near the other counters, e.g. by `self._total_backups`). At the END of `_backup` (after the value-propagation loop):

```python
        if self._observer is not None:
            self._backup_sim_counter += 1
            root = search_path[0]
            updated_root_move = search_path[1].move if len(search_path) >= 2 else None
            self._observer.on_root_simulation(
                self._backup_sim_counter, root, updated_root_move)
```

(Guarded by `is not None` → observer-off is byte-identical: no counter, no call.)

- [ ] **Step 4: Run tests** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_trace_observer.py -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/mcts.py tests/test_fpu_trace_observer.py
git commit -m "feat(fpu): opt-in read-only per-completed-simulation observer hook in _backup"
```

---

## Task 4: Complete-state canonical hash

**Files:** Create `scripts/GPU/alphazero/fpu_state_hash.py`; Test `tests/test_fpu_state_hash.py`.

**Interfaces (produces):** `canonical_state_key(state) -> tuple`; `canonical_state_sha1(state) -> str`. Key covers all future-play-relevant fields: `board_size, active_size, to_move, sorted(pegs), sorted(bridges), max_plies_limit`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_fpu_state_hash.py`:

```python
from scripts.GPU.alphazero.game.twixt_state import TwixtState
from scripts.GPU.alphazero.fpu_state_hash import canonical_state_key, canonical_state_sha1


def _play(moves, active_size=24, max_plies=None):
    s = TwixtState(active_size=active_size, to_move="red", max_plies_limit=max_plies)
    for m in moves:
        s = s.apply_move(m)
    return s


def test_same_position_same_hash():
    a = _play([(10, 10), (12, 12)])
    b = _play([(10, 10), (12, 12)])
    assert canonical_state_sha1(a) == canonical_state_sha1(b)


def test_equal_hash_implies_equal_side_legal_terminal(tmp_path):
    a = _play([(10, 10), (12, 12), (8, 9)])
    b = _play([(10, 10), (12, 12), (8, 9)])
    assert canonical_state_sha1(a) == canonical_state_sha1(b)
    assert a.to_move == b.to_move
    assert set(a.legal_moves()) == set(b.legal_moves())
    assert a.is_terminal() == b.is_terminal() and a.winner() == b.winner()


def test_active_size_and_cap_change_the_hash():
    base = _play([(10, 10), (12, 12)])
    diff_cap = _play([(10, 10), (12, 12)], max_plies=40)
    assert canonical_state_sha1(base) != canonical_state_sha1(diff_cap)   # cap affects future
    # different pegs -> different hash
    assert canonical_state_sha1(base) != canonical_state_sha1(_play([(10, 10), (13, 13)]))
```

- [ ] **Step 2: Run to verify fail** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_state_hash.py -q`. Expected: FAIL (module missing).

- [ ] **Step 3: Implement** — create `scripts/GPU/alphazero/fpu_state_hash.py`:

```python
"""Complete-state canonical hash for cross-corpus position disjointness (design §2.3).

Covers every future-play-relevant component (not just the visible board): board/active
size, side to move, both peg sets, bridges, and the move-count cap. TwixtState has no
swap/pie rule; if one is ever added it must be folded in here (guarded by
tests/test_fpu_state_hash.py's equal-hash-implies-equal-behavior test)."""
from __future__ import annotations

import hashlib
import json


def canonical_state_key(state):
    # TwixtState fields (verified): pegs: Dict[(r,c) -> player]; bridges: Set of
    # already-canonical ((r1,c1),(r2,c2)) tuples (sortable); to_move/board_size/
    # active_size: primitives; max_plies_limit: int|None. ply == len(pegs), so
    # pegs captures it; active_size/max_plies_limit affect legality/terminal and
    # are NOT in TwixtState.__eq__, so they are included here deliberately.
    return (
        int(state.board_size),
        int(state.active_size),
        str(state.to_move),
        tuple(sorted((int(r), int(c), str(pl)) for (r, c), pl in state.pegs.items())),
        tuple(sorted(state.bridges)),
        (None if state.max_plies_limit is None else int(state.max_plies_limit)),
    )


def canonical_state_sha1(state) -> str:
    return hashlib.sha1(
        json.dumps(canonical_state_key(state), sort_keys=True).encode()
    ).hexdigest()
```

- [ ] **Step 4: Run tests** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_state_hash.py -q`. Expected: PASS (fix the accessors until it does).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/fpu_state_hash.py tests/test_fpu_state_hash.py
git commit -m "feat(fpu): complete-state canonical hash + equal-hash-implies-equal-behavior test"
```

---

## Task 5: Dev-corpus pure sampler

**Files:** Create `scripts/GPU/alphazero/build_fpu_dev_corpus.py` (pure section); Test `tests/test_fpu_dev_corpus.py`.

**Interfaces (produces):** constants `BRANCHING_BANDS = [(200,299),(300,399),(400,None)]`, `PLY_BUCKETS` (opening/early_mid/midgame/late), `TARGET_PER_BAND=60`, `CONTROL_PER_BAND=20`, `SPLIT_ALLOC` (the §2.2 table); `classify(row) -> "target"|"control"|None`; `assign_split(games, alloc, seed) -> {game_idx: "tuning"|"frozen_check"}`; `sample_dev_rows(confirmed, *, seed) -> (rows, stats)` (game-first round-robin, ≤2/game, ≥12-ply gap, ~50/50 side, band/split quotas, ply-bucket ≤50% cap). A `confirmed` record = `{game_idx,n_moves,winner,replay_path,position_ply,side,n_legal,root_value_stm,normalized_entropy,top1_prior,band,role,canonical_sha1}` (produced by the Task 6 shell; pure sampler consumes it).

- [ ] **Step 1: Write the failing tests** — create `tests/test_fpu_dev_corpus.py`:

```python
from collections import Counter
from scripts.GPU.alphazero.build_fpu_dev_corpus import (
    classify, band_of, ply_bucket_of, sample_dev_rows, SPLIT_ALLOC)


def test_band_and_bucket_and_classify():
    assert band_of(250) == "200_299" and band_of(350) == "300_399" and band_of(480) == "400_plus"
    assert band_of(150) is None
    assert ply_bucket_of(15) == "opening" and ply_bucket_of(91) == "late"
    assert classify(n_legal=300, root_value_stm=0.1, normalized_entropy=0.95, top1_prior=0.01) == "target"
    assert classify(n_legal=300, root_value_stm=0.1, normalized_entropy=0.80, top1_prior=0.2) == "control"
    assert classify(n_legal=300, root_value_stm=0.5, normalized_entropy=0.95, top1_prior=0.01) is None  # value too high


def _confirmed(n_games=400):
    out = []
    for g in range(n_games):
        # each game offers a couple of eligible target positions across bands
        band_nl = {0: 250, 1: 350, 2: 480}[g % 3]
        for k, ply in enumerate((30, 60)):
            out.append(dict(game_idx=g, n_moves=200, winner="red" if g % 2 else "black",
                            replay_path=f"r/{g}.json", position_ply=ply + 20*(g % 5),
                            side="red" if (ply + 20*(g % 5)) % 2 == 0 else "black",
                            n_legal=band_nl, root_value_stm=0.05,
                            normalized_entropy=0.95, top1_prior=0.01,
                            band=band_of(band_nl), role="target",
                            canonical_sha1=f"h{g}_{k}"))
    return out


def test_sample_is_deterministic():
    c = _confirmed()
    assert sample_dev_rows(c, seed=7) == sample_dev_rows(c, seed=7)


def test_caps_gap_and_split_composition():
    # Build an abundant confirmed pool (targets + controls across bands) and check invariants.
    from scripts.GPU.alphazero.build_fpu_dev_corpus import band_of
    rows, stats = sample_dev_rows(_abundant_pool(), seed=1)
    # <=2 per game
    assert max(Counter(r["game_idx"] for r in rows).values()) <= 2
    # >=12-ply separation within a game
    bygame = {}
    for r in rows:
        bygame.setdefault(r["game_idx"], []).append(r["position_ply"])
    assert all(all(abs(a-b) >= 12 for i, a in enumerate(v) for b in v[i+1:]) for v in bygame.values())
    # whole-game split (a game's rows never straddle splits)
    gsplit = {}
    for r in rows:
        gsplit.setdefault(r["game_idx"], set()).add(r["split"])
    assert all(len(s) == 1 for s in gsplit.values())
    # no ply bucket > 50% of corpus
    assert max(Counter(r["ply_bucket"] for r in rows).values()) <= 0.5 * len(rows) + 1
```

(`_abundant_pool` is a fixture that yields enough target+control games per band/split to meet quotas; provide it in the test file mirroring `_confirmed` with role="control" rows added.)

- [ ] **Step 2: Run to verify fail** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_dev_corpus.py -q`. Expected: FAIL (module missing).

- [ ] **Step 3: Implement the pure sampler** — create `build_fpu_dev_corpus.py` with the module docstring + pure constants/functions (`band_of`, `ply_bucket_of`, `classify`, `assign_split`, `sample_dev_rows`). `sample_dev_rows` does game-first round-robin over shuffled games (deterministic seed), filling per-(band, role, split) quotas from the §2.2 allocation table, enforcing ≤2/game, ≥12-ply gap, ~50/50 side (prefer under-represented parity), whole-game split (assign a game to a split before drawing its rows, tracking per-split-band-role remaining quota), and the ply-bucket ≤50% cap; returns rows (each with `case_id, game_idx, replay_path, position_ply, side, ply_bucket, band, role, split, game_result, total_plies, source_corpus_id, canonical_sha1, sample_seed`) + stats (per band/role/split achieved vs requested, side balance, shortfalls). Log any shortfall (no silent caps).

- [ ] **Step 4: Run tests** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_dev_corpus.py -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/build_fpu_dev_corpus.py tests/test_fpu_dev_corpus.py
git commit -m "feat(fpu): dev-corpus pure sampler — band/split quotas, round-robin, gap, side balance"
```

---

## Task 6: Dev-corpus builder shell (two-stage scan, operator-run)

**Files:** Modify `scripts/GPU/alphazero/build_fpu_dev_corpus.py` (shell); Test `tests/test_fpu_dev_corpus.py`.

**Interfaces (produces):** `enumerate_candidate_plies(game_record, stride=4, cap=6) -> [ply]` (the "1st,5th,9th,… qualifying ply with n_legal≥200, ≤6/game", using stored replay `n_legal`); `raw_policy_features(state, evaluator) -> (n_legal, normalized_entropy, top1_prior, priors)`; `load_disjoint_hashes(paths) -> set[str]` (selected-A + v16a hashes); `main(argv)` — two-stage scan → confirm-under-anchor → hash + discard-on-collision → sample → assert zero collisions/dups → write manifest + meta. Heavy (MCTS anchor); operator-run. Unit tests cover the pure pieces only.

- [ ] **Step 1: Write the failing tests** — append pure tests:

```python
def test_enumerate_candidate_plies_stride_and_cap():
    from scripts.GPU.alphazero.build_fpu_dev_corpus import enumerate_candidate_plies
    # plies 0..49 with n_legal per ply; eligible = n_legal>=200
    nlegal = {p: (250 if p >= 10 else 100) for p in range(50)}   # eligible plies 10..49
    got = enumerate_candidate_plies({"n_moves": 50, "ply_n_legal": nlegal}, stride=4, cap=6)
    assert got == [10, 14, 18, 22, 26, 30]      # 1st,5th,9th,... eligible, capped at 6


def test_normalized_entropy_and_top1():
    from scripts.GPU.alphazero.build_fpu_dev_corpus import _policy_features_from_priors
    import math
    flat = [1/300]*300
    nfeat = _policy_features_from_priors(flat)
    assert nfeat["top1_prior"] <= 0.01 and nfeat["normalized_entropy"] > 0.99
    peaked = [0.9] + [0.1/299]*299
    assert _policy_features_from_priors(peaked)["top1_prior"] == 0.9


def test_final_manifest_disjointness_assert():
    from scripts.GPU.alphazero.build_fpu_dev_corpus import assert_disjoint
    import pytest
    assert_disjoint(dev_hashes={"a", "b"}, forbidden={"c"})       # ok
    with pytest.raises(AssertionError):
        assert_disjoint(dev_hashes={"a", "b"}, forbidden={"b"})   # collision
    with pytest.raises(AssertionError):
        assert_disjoint(dev_hashes=["a", "a"], forbidden={"c"})   # internal dup
```

- [ ] **Step 2: Run to verify fail** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_dev_corpus.py -q -k "enumerate or entropy or disjoint"`. Expected: FAIL.

- [ ] **Step 3: Implement the shell** — append: `enumerate_candidate_plies` (stride+cap over stored `n_legal≥200` plies), `_policy_features_from_priors` (`H = -Σ p ln p`; `normalized_entropy = H/ln(n_legal)`; `top1_prior = max`), `raw_policy_features` (reconstruct via `position_state`, call `_teacher_infer` from `build_teacher_calibration_manifest`, derive features), `assert_disjoint(dev_hashes, forbidden)` (raise on any dev∩forbidden or internal dup), `load_disjoint_hashes` (hash selected-A + v16a positions via `canonical_state_sha1` on their reconstructed states), and `main`: for each source game (deterministic order) → `enumerate_candidate_plies` → raw-policy prefilter (`classify` on features) → **anchor confirm** (`search_with_root` @400 sims, `fpu_policy_mass_reduction=None`, via `DEFAULT_CHECKPOINT`) keeping `|root_value_stm|≤0.25` → compute `canonical_state_sha1`, **discard on collision** with forbidden/kept → accumulate per band/role with ≥2× reserve → `sample_dev_rows` → `assert_disjoint` → write `logs/eval/fpu_dev_corpus/dev_corpus_manifest.csv` + `.meta.json`. `DEFAULT_CHECKPOINT` imported (deferred) from `diagnose_fpu_sweep`; anchor uses `eval_runner.cfg_from(EvalConfig(mcts_sims=400,...))` then `dataclasses.replace(cfg, fpu_policy_mass_reduction=None)` (explicit off).

- [ ] **Step 4: Run tests** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_dev_corpus.py -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/build_fpu_dev_corpus.py tests/test_fpu_dev_corpus.py
git commit -m "feat(fpu): dev-corpus builder shell — bounded two-stage scan, confirm-under-anchor, disjointness"
```

---

## Task 7: Discovery diagnostic — observer logic, modes, dual-reference gates

**Files:** Create `scripts/GPU/alphazero/diagnose_fpu_policy_mass.py`; Test `tests/test_fpu_trace_observer.py` (extend) + `tests/test_fpu_diagnostic_modes.py` (create).

**Interfaces (produces):** `FpuTraceObserver` (records design-§4 events from `on_root_simulation` calls, using the MCTS leader comparator `_best_child`); `lock_in_event(row) -> bool` (§6.1); gate functions `progress(v_off, v_r)`, `reply_reduction(replies_ref, replies_x)`, `prior_rank(priors, move)`, `top_share(root)`; `dev_safety_verdict(rows, ref)` and `selected_a_verdict(rows)` (§6.2/§6.3, exact thresholds); `validate_mode(cases, mode, fpus)` (§3 mode/split/config enforcement); `main(argv)` (operator-run: selected-A + dev corpus × configs, dual-reference metrics, gate emission). Heavy (MCTS); operator-run — unit tests cover observer + gates + mode validation only.

- [ ] **Step 1: Write the failing tests** — extend `tests/test_fpu_trace_observer.py` with `FpuTraceObserver` event tests driven by synthetic `on_root_simulation` call sequences (first-visit sims; explored-mass thresholds 25/50/75%; leader-change timeline; final-leader last-takeover = stabilization; `None` root-move ignored for bookkeeping but counter advances). Create `tests/test_fpu_diagnostic_modes.py`:

```python
import pytest
from scripts.GPU.alphazero.diagnose_fpu_policy_mass import (
    validate_mode, lock_in_event, progress, reply_reduction, prior_rank)


def test_progress_and_reply_reduction_formulas():
    assert abs(progress(0.30, 0.13) - (0.30 - 0.13)/(0.30 - (-0.0451))) < 1e-9
    assert abs(reply_reduction(200, 100) - 0.5) < 1e-9


def test_prior_rank_strictly_greater():
    assert prior_rank({1: 0.5, 2: 0.3, 3: 0.3}, move=2) == 2      # one strictly greater
    assert prior_rank({1: 0.5, 2: 0.3, 3: 0.3}, move=1) == 1


def test_lock_in_event_all_conditions():
    base = dict(selected_move_prior_rank=11, selected_move_prior=0.005,
                explored_mass_at_stabilization=0.20, stabilization_sim=80,
                final_root_top_share=0.95)
    assert lock_in_event(base) is True
    assert lock_in_event({**base, "selected_move_prior_rank": 10}) is False   # rank not >10
    assert lock_in_event({**base, "final_root_top_share": 0.89}) is False     # share <0.90


def test_mode_isolation_rejects_wrong_split_and_bad_configs():
    tuning_rows = [{"split": "tuning"}]
    frozen_rows = [{"split": "frozen_check"}]
    with pytest.raises(ValueError):
        validate_mode(frozen_rows, mode="tuning", fpus=[0.0, -0.10])          # wrong split
    with pytest.raises(ValueError):
        validate_mode(frozen_rows, mode="frozen_check", fpus=[0.0, -0.10, -0.20])  # >1 nonzero
    validate_mode(tuning_rows, mode="tuning", fpus=[0.0, -0.10, -0.20, -0.35, -0.50, -0.75])
    validate_mode(frozen_rows, mode="frozen_check", fpus=[0.0, -0.20])        # exactly one nonzero ok
```

- [ ] **Step 2: Run to verify fail** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_trace_observer.py tests/test_fpu_diagnostic_modes.py -q`. Expected: FAIL (module/symbols missing).

- [ ] **Step 3: Implement** — create `diagnose_fpu_policy_mass.py`:
  - `FpuTraceObserver`: `on_root_simulation(count, root, updated_root_move)` updates incremental explored mass (adds `root.priors[move]` on a move's first visit; ignores `None`), records first-visit sims, leader timeline (leader via a shared `_best_child`-style comparator imported from `continuation_extraction`), mass-threshold crossing sims (25/50/75%), and derives final-leader last-takeover (stabilization) + explored-mass-at-first-leader + selected-move prior/rank at the end.
  - Gate helpers with the **exact §6 formulas**: `progress` (`V_ref=-0.0451`), `reply_reduction`, `prior_rank` (strictly-greater), `top_share`, `lock_in_event` (the 5-condition boolean), `dev_safety_verdict(rows, ref)` (§6.2 rejects vs a reference), `selected_a_verdict(rows)` (§6.3 requires).
  - `validate_mode(cases, mode, fpus)`: assert all rows' `split` match `mode`; tuning permits `{0.0}∪grid`; frozen_check permits `{absolute_off, 0.0, exactly-one-nonzero}` else raise; selected-A only in tuning.
  - `main`: mode-enforced; runs selected-A (tuning only) + dev corpus × configs (`absolute_off`=None, `r=0.0`, candidate grid via `dataclasses.replace(cfg, fpu_policy_mass_reduction=r)`), attaches an `FpuTraceObserver` per root run, emits per-case rows (dual-reference deltas) + a gate report (dev-safety vs both refs, selected-A vs absolute_off, §5 smallest-safe-passing). Operator-run.

- [ ] **Step 4: Run tests** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_trace_observer.py tests/test_fpu_diagnostic_modes.py -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/diagnose_fpu_policy_mass.py tests/test_fpu_trace_observer.py tests/test_fpu_diagnostic_modes.py
git commit -m "feat(fpu): discovery diagnostic — trace observer, dual-reference §6 gates, enforced modes"
```

---

## Task 8: Consolidated byte-identical proof + full suite

**Files:** Test `tests/test_fpu_policy_mass_rule.py` (extend); no source change expected.

- [ ] **Step 1: Write the `old==new` selection-trace proof** — extend `tests/test_fpu_policy_mass_rule.py` with a test that builds a fixed synthetic root+children (known priors, mixed visited/unvisited, a fake evaluator or pre-expanded tree), runs a bounded synchronous search with `MCTSConfig()` (both features off, observer `None`), and asserts the sequence of `_select_child` decisions + final visit distribution equals the pre-branch behavior — captured as a committed golden (generated once from the new code with features off, then diffed against pre-branch `mcts.py` via `git show HEAD~7:…/mcts.py` run on the identical synthetic tree, mirroring the v16a old==new proof). Assert byte-equal decision trace.

- [ ] **Step 2: Run the proof + the new-feature suite** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_policy_mass_rule.py tests/test_fpu_trace_observer.py tests/test_fpu_state_hash.py tests/test_fpu_dev_corpus.py tests/test_fpu_diagnostic_modes.py -q`. Expected: PASS.

- [ ] **Step 3: Full repository suite (byte-identical-off gate)** — `.venv/bin/python -m pytest tests/ -q`. Expected: PASS (every MCTS/self-play/eval test exercises the `None`/observer-off path unchanged; record any pre-existing unrelated failures).

- [ ] **Step 4: Commit**

```bash
git add tests/test_fpu_policy_mass_rule.py
git commit -m "test(fpu): old==new byte-identical selection-trace proof (both features off) + full-suite gate"
```

---

## Self-review — spec coverage

| Spec | Task |
|---|---|
| §A rule (formula, field, guard, nonfinite, completed-visits, byte-identical) | 1, 2, 8 |
| §4 observer (per-completed-sim, None edge, shared comparator, isolation) | 3, 7 |
| §2.3 complete-state hash + equivalence + disjointness/collision-discard | 4, 6 |
| §2 corpus (bands, buckets, eligibility, two-stage scan, split alloc, round-robin/cap/gap/side) | 5, 6 |
| §3 modes + dual-reference metrics | 7 |
| §6 numeric gates (lock-in event, dev-safety vs both refs, selected-A formulas, frozen thresholds) | 7 |
| §7 tooling-only scope; operator phases not run | all (heavy runs are `main` entry points, never invoked by tests) |
| §9 do-not-change | Global Constraints |

**Placeholder scan:** none (implementation notes flag where the engineer confirms `TwixtState`/`MCTS.__init__` specifics against the real code, guarded by tests). **Type consistency:** `MCTSConfig.fpu_policy_mass_reduction: float|None` used identically across helper/`_select_child`/`dataclasses.replace`; `policy_mass_fpu`/`explored_policy_mass` signatures stable; observer `on_root_simulation(count:int, root, updated_root_move:int|None)` matches the `_backup` call site; a `confirmed` record's fields flow builder→sampler unchanged.
