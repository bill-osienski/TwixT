# Context-Relative FPU (Policy-Mass) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the *tooling* for the context-relative FPU successor — the opt-in parent-relative + explored-policy-mass rule, a read-only search-trace observer, a geometry development-corpus builder, and a staged, mode-enforced discovery diagnostic — all verified, with the shipped absolute-FPU path proven byte-identical against a **pre-branch** golden. Heavy MCTS phases stay operator-run.

**Architecture:** One small `mcts.py` change (opt-in `MCTSConfig` field + `_select_child` branch + a guarded `_backup` observer hook that passes the canonical visit-leader + a pure helper), two operator-run scripts (corpus builder, discovery diagnostic), and a shared complete-state hash. Everything MCTS-heavy is behind opt-in flags, exercised only in operator phases; unit/integration tests use pure functions, a CPU fake evaluator, and synthetic trees — no GPU/MLX.

**Tech Stack:** Python 3 stdlib (dataclasses, math, hashlib, json, csv, argparse, random) + numpy (already a dep) for the fake evaluator/tensors. Tests via `.venv/bin/python -m pytest -p no:cacheprovider`. Frozen spec: `docs/superpowers/specs/2026-07-10-context-relative-fpu-policy-mass-design.md`.

## Global Constraints

- **Byte-identical off** is proven against a **pre-branch golden captured before any source edit** (Task 0): `MCTSConfig()` default (`fpu_policy_mass_reduction=None`, observer `None`) reproduces the pre-branch deterministic search output exactly; full existing suite passes unchanged.
- **`0.0` ≠ `None`:** `0.0` is an enabled mode (`FPU = Q_parent`); `None` keeps the absolute `fpu_value` path. Represented as distinct typed configs, never the same float.
- **Mutual exclusion / bounds:** raise if `fpu_policy_mass_reduction is not None` and `fpu_value != 0.0`; raise if non-finite or `< 0`. The coefficient is **nonnegative** (grid `0.10, 0.20, 0.35, 0.50, 0.75`).
- **Completed visits only** for `P_explored` (never virtual/pending).
- **Observer:** read-only; observer-local completed counter (no mutation when off); one callback per completed simulation carrying the **canonical MCTS visit-leader** (from a `mcts.py` helper, not a diagnostic module); exceptions abort the diagnostic.
- **Do NOT modify** `self_play.py`, `SIMS_TABLE`, trainer, network, promotion, calibration manifests, value-adapter/projection. No self-play adoption.
- **Do NOT run** operator phases (geometry scan, corpus generation, coefficient sweep, selection, frozen-check, held-out validation). Ship tooling only.
- **Tests** run from repo root: `.venv/bin/python -m pytest -p no:cacheprovider …` (always with `-p no:cacheprovider`).
- **All §6 numeric gate thresholds are frozen;** only `r0_target_lockin_count`/`absoff_target_lockin_count` are substituted from the control stage into the frozen `baseline + 2` caps.

## File structure

- **Modify** `scripts/GPU/alphazero/mcts.py` — `MCTSConfig.fpu_policy_mass_reduction` + `__post_init__` guard; module helpers `policy_mass_fpu`, `explored_policy_mass`, `visit_leader_move`; `_select_child` branch; `MCTS.__init__(observer=None)` + guarded `_backup` callback.
- **Create** `scripts/GPU/alphazero/fpu_state_hash.py` — `canonical_state_key` / `canonical_state_sha1`.
- **Create** `scripts/GPU/alphazero/build_fpu_dev_corpus.py` — pure sampler + two-stage confirm-under-anchor shell + disjointness (operator-run).
- **Create** `scripts/GPU/alphazero/diagnose_fpu_policy_mass.py` — `FpuRunConfig`, `FpuTraceObserver`, staged mode-enforced diagnostic, dual-reference §6 gates (operator-run).
- **Create** `tests/fpu_search_fixture.py` — CPU fake evaluator + deterministic `search_with_root` harness (shared by Tasks 0/3/8).
- **Create tests** `tests/test_fpu_policy_mass_rule.py`, `tests/test_fpu_state_hash.py`, `tests/test_fpu_dev_corpus.py`, `tests/test_fpu_trace_observer.py`, `tests/test_fpu_diagnostic_modes.py`; golden `tests/golden/fpu_prebranch_search.json`.

---

## Task 0: Pre-branch golden + CPU search fixture (SETUP — before any source edit)

Captures the byte-identical reference from the **unmodified** `mcts.py`, so the Task 8 proof is independent (fix 12). Also provides the CPU fake evaluator + harness reused by the observer integration test (fix 2) and Task 8.

**Files:** Create `tests/fpu_search_fixture.py`, `tests/golden/fpu_prebranch_search.json`.

- [ ] **Step 1: Write the fixture** — `tests/fpu_search_fixture.py`:

```python
"""CPU fake evaluator + deterministic search harness (no GPU/MLX). The fake
returns uniform priors over legal moves and value 0.0; with a fixed rng seed the
real search is fully deterministic, so its output is a stable byte-identical
fingerprint."""
import numpy as np
from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
from scripts.GPU.alphazero.game.twixt_state import TwixtState


class FakeEvaluator:
    network = None  # _expand_batch never reads this on the fake path
    def build_input_tensor(self, state):
        a = state.active_size
        return np.zeros((1, a, a), dtype=np.float32)          # infer ignores boards
    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        B, M = move_rows.shape
        s = move_mask.sum(axis=1, keepdims=True)
        priors = np.divide(move_mask, s, out=np.zeros_like(move_mask), where=s > 0)
        return priors.astype(np.float32), np.zeros(B, dtype=np.float32)


def run_search(config=None, *, seed=1234, active_size=6, moves=((2, 2), (3, 3)),
               n_sims=200, observer=None):
    import random
    st = TwixtState(active_size=active_size, to_move="red", max_plies_limit=None)
    for m in moves:
        st = st.apply_move(m)
    mcts = MCTS(FakeEvaluator(), config or MCTSConfig(), random.Random(seed),
                **({"observer": observer} if observer is not None else {}))
    visit_counts, root_value, root = mcts.search_with_root(st, add_noise=False)
    return {"visit_counts": {f"{r},{c}": v for (r, c), v in sorted(visit_counts.items())},
            "root_value": round(float(root_value), 12)}, root, mcts
```

(If `MCTS.__init__` does not yet accept `observer`, the `**({...})` passes it only when set; before Task 3 the harness is called without an observer.)

- [ ] **Step 2: Capture the golden from the UNMODIFIED mcts.py**

```bash
mkdir -p tests/golden
.venv/bin/python - <<'PY'
import json
from tests.fpu_search_fixture import run_search
out, _root, _m = run_search()          # MCTSConfig() default; current (pre-branch) code
json.dump(out, open("tests/golden/fpu_prebranch_search.json", "w"), indent=2, sort_keys=True)
print("captured", out)
PY
cat tests/golden/fpu_prebranch_search.json
```
Expected: a deterministic `visit_counts` (summing to `n_sims`) + `root_value`. Eyeball that visits sum to 200.

- [ ] **Step 3: Commit (pre-branch reference)**

```bash
git add tests/fpu_search_fixture.py tests/golden/fpu_prebranch_search.json
git commit -m "test(fpu): pre-branch search golden + CPU fake-evaluator fixture (byte-identical reference)"
```

---

## Task 1: Rule helper + config field + guard

**Files:** Modify `mcts.py`; Test `tests/test_fpu_policy_mass_rule.py` (create).

**Interfaces:** `policy_mass_fpu(parent_q, explored_mass, r) -> float`; `MCTSConfig.fpu_policy_mass_reduction: float | None = None` + `__post_init__` validation.

- [ ] **Step 1: Failing tests** — create `tests/test_fpu_policy_mass_rule.py`:

```python
import math, dataclasses, pytest
from scripts.GPU.alphazero.mcts import MCTSConfig, policy_mass_fpu


def test_formula_and_clamp():
    assert policy_mass_fpu(0.0, 0.0, 0.20) == 0.0
    assert abs(policy_mass_fpu(0.0, 1.0, 0.20) - (-0.20)) < 1e-12
    assert abs(policy_mass_fpu(0.3, 0.25, 0.20) - (0.3 - 0.20*0.5)) < 1e-12
    assert policy_mass_fpu(0.1, 5.0, 0.20) == policy_mass_fpu(0.1, 1.0, 0.20)
    assert policy_mass_fpu(0.1, -3.0, 0.20) == 0.1


def test_rejects_nonfinite():
    for bad in (float("nan"), float("inf")):
        for args in ((bad, 0.5, 0.2), (0.0, bad, 0.2), (0.0, 0.5, bad)):
            with pytest.raises(ValueError):
                policy_mass_fpu(*args)


def test_config_none_default_and_zero_is_enabled():
    assert MCTSConfig().fpu_policy_mass_reduction is None
    c = MCTSConfig(fpu_policy_mass_reduction=0.0)
    assert c.fpu_policy_mass_reduction == 0.0 and c.fpu_policy_mass_reduction is not None


def test_config_guard_and_bounds():
    with pytest.raises(ValueError): MCTSConfig(fpu_value=-0.2, fpu_policy_mass_reduction=0.10)
    with pytest.raises(ValueError): MCTSConfig(fpu_policy_mass_reduction=-0.1)
    with pytest.raises(ValueError): MCTSConfig(fpu_policy_mass_reduction=float("inf"))
    MCTSConfig(fpu_policy_mass_reduction=0.20)                       # fpu_value default 0.0 -> ok
    dataclasses.replace(MCTSConfig(), fpu_policy_mass_reduction=0.35)
```

- [ ] **Step 2: Run → fail** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_policy_mass_rule.py -q`.

- [ ] **Step 3: Implement** — add the helper (module level, `math` already imported):

```python
def policy_mass_fpu(parent_q: float, explored_mass: float, r: float) -> float:
    """Parent-relative FPU with explored-policy-mass scaling (design §A).
    Rejects non-finite inputs (a NaN mass passes both clamp comparisons)."""
    if not (math.isfinite(parent_q) and math.isfinite(explored_mass) and math.isfinite(r)):
        raise ValueError("policy_mass_fpu requires finite inputs")
    m = 0.0 if explored_mass < 0.0 else (1.0 if explored_mass > 1.0 else explored_mass)
    return parent_q - r * math.sqrt(m)
```

Add the field after `fpu_value` (~:100):
```python
    fpu_policy_mass_reduction: float | None = None  # None => absolute fpu_value path
                                                    # (byte-identical). Not None => FPU =
                                                    # Q_parent - r*sqrt(P_explored); r>=0.
                                                    # 0.0 is an ENABLED mode (FPU=Q_parent).
```
Extend `__post_init__` (~:147):
```python
        if self.fpu_policy_mass_reduction is not None:
            if self.fpu_value != 0.0:
                raise ValueError("fpu_policy_mass_reduction and a nonzero absolute "
                                 "fpu_value are mutually exclusive")
            r = self.fpu_policy_mass_reduction
            if not math.isfinite(r) or r < 0:
                raise ValueError("fpu_policy_mass_reduction must be finite and >= 0")
```

- [ ] **Step 4: Run → pass**; **Step 5: Commit**
```bash
git add scripts/GPU/alphazero/mcts.py tests/test_fpu_policy_mass_rule.py
git commit -m "feat(fpu): policy-mass helper + opt-in nonnegative field + mutual-exclusion guard"
```

---

## Task 2: `_select_child` branch + completed-visit `P_explored`

**Files:** Modify `mcts.py`; Test `tests/test_fpu_policy_mass_rule.py`.

**Interfaces:** `explored_policy_mass(node) -> float` (Σ prior over children with completed visits); `_select_child` uses `policy_mass_fpu(node.q_value, explored_policy_mass(node), r)` for the unvisited assumed value when enabled, else `self.config.fpu_value` (unchanged).

- [ ] **Step 1: Failing tests** — append:
```python
from scripts.GPU.alphazero.mcts import MCTSNode, encode_move, explored_policy_mass


def _node(priors, visited):
    n = MCTSNode(state=None, visit_count=3, value_sum=0.9)     # q_value 0.3
    n.priors = {encode_move(*rc): p for rc, p in priors.items()}
    for rc, p in priors.items():
        mid = encode_move(*rc)
        n.children[mid] = MCTSNode(state=None, parent=n, move=mid,
                                   visit_count=(5 if rc in visited else 0), value_sum=0.0)
    return n


def test_explored_mass_completed_visits_only():
    n = _node({(1, 1): 0.5, (2, 2): 0.3, (3, 3): 0.2}, visited={(1, 1)})
    assert explored_policy_mass(n) == 0.5           # zero-visit children excluded
```

- [ ] **Step 2: Run → fail** (`cannot import name 'explored_policy_mass'`).

- [ ] **Step 3: Implement** — add near `policy_mass_fpu`:
```python
def explored_policy_mass(node) -> float:
    """Σ prior over children with a COMPLETED (backed-up) visit. Virtual/pending
    visits do not touch child.visit_count, so they are excluded (design §A)."""
    total = 0.0
    for move_id, prior in node.priors.items():
        child = node.children.get(move_id)
        if child is not None and child.visit_count > 0:
            total += prior
    return total
```
In `_select_child`, after `sqrt_parent = math.sqrt(node.visit_count + 1)` (~:949):
```python
        _pm = self.config.fpu_policy_mass_reduction
        _fpu_pm = (policy_mass_fpu(node.q_value, explored_policy_mass(node), _pm)
                   if _pm is not None else None)
```
Change the unvisited branch (`q = self.config.fpu_value`) to:
```python
            else:
                q = _fpu_pm if _pm is not None else self.config.fpu_value
                child_visits = 0
```
(When `_pm is None`: no mass pass, `q = self.config.fpu_value` exactly — byte-identical.)

- [ ] **Step 4: Run → pass**; existing MCTS tests: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_value.py tests/test_fpu_sweep.py -q`.
- [ ] **Step 5: Commit**
```bash
git add scripts/GPU/alphazero/mcts.py tests/test_fpu_policy_mass_rule.py
git commit -m "feat(fpu): _select_child policy-mass branch + completed-visit P_explored"
```

---

## Task 3: Observer hook (per-completed-sim, canonical leader) + integration proof

**Files:** Modify `mcts.py`; Test `tests/test_fpu_trace_observer.py` (create).

**Interfaces:** `visit_leader_move(node) -> int | None` (max `visit_count`, tie-break lowest encoded move id — the canonical selection leader, in `mcts.py`); `MCTS.__init__(..., observer=None)` → `self._observer`, `self._observer_completed_count = 0`; guarded `_backup` tail firing `on_root_simulation(count, root, updated_root_move, current_root_leader_move)`. Observer-off mutates nothing (fix 1).

- [ ] **Step 1: Failing tests** — create `tests/test_fpu_trace_observer.py`: a direct-`_backup` unit test (spy: one call per `_backup`, `updated_root_move`=`search_path[1].move` or `None`, counter advances only when observer set; `None`-observer path mutates nothing) **and** the integration test (fix 2):

```python
from tests.fpu_search_fixture import run_search


class _Spy:
    def __init__(self): self.calls = []
    def on_root_simulation(self, count, root, updated_root_move, leader):
        self.calls.append((count, updated_root_move, leader))


def test_integration_one_callback_per_completed_sim():
    spy = _Spy()
    _out, _root, _m = run_search(n_sims=200, observer=spy)
    assert [c for c, _, _ in spy.calls] == list(range(1, 201))       # 1..200, no gaps/dups
    assert all(m is not None for _, m, _ in spy.calls)              # root pre-expanded => has a root move
    assert all(ldr is not None for _, _, ldr in spy.calls)


def test_observer_off_output_matches_prebranch_golden():
    import json
    out, _root, _m = run_search()                                   # observer None, default config
    assert out == json.load(open("tests/golden/fpu_prebranch_search.json"))
```

(Also unit-test `visit_leader_move`: max-visit, lowest-id tie-break, `None` on no visited child.)

- [ ] **Step 2: Run → fail**.

- [ ] **Step 3: Implement** — add `visit_leader_move` (near the other helpers):
```python
def visit_leader_move(node) -> "int | None":
    """Canonical MCTS visit leader: max visit_count, ties -> lowest encoded move id.
    Matches final-move selection semantics; used by the trace observer so it never
    diverges from search (design §4)."""
    best_id, best_vc = None, -1
    for move_id, child in node.children.items():
        vc = child.visit_count
        if vc > 0 and (vc > best_vc or (vc == best_vc and move_id < best_id)):
            best_id, best_vc = move_id, vc
    return best_id
```
In `MCTS.__init__` (:211) add `observer=None`; store `self._observer = observer` and `self._observer_completed_count = 0`. At the END of `_backup`:
```python
        if self._observer is not None:
            self._observer_completed_count += 1
            root = search_path[0]
            move = search_path[1].move if len(search_path) >= 2 else None
            self._observer.on_root_simulation(
                self._observer_completed_count, root, move, visit_leader_move(root))
```
(Guarded → observer-off: no counter, no leader computation, byte-identical.)

- [ ] **Step 4: Run → pass** (`tests/test_fpu_trace_observer.py`); **Step 5: Commit**
```bash
git add scripts/GPU/alphazero/mcts.py tests/test_fpu_trace_observer.py
git commit -m "feat(fpu): per-completed-sim observer hook w/ canonical visit-leader; integration + golden proof"
```

---

## Task 4: Complete-state canonical hash

**Files:** Create `scripts/GPU/alphazero/fpu_state_hash.py`; Test `tests/test_fpu_state_hash.py`.

**Interfaces:** `canonical_state_key(state) -> tuple`; `canonical_state_sha1(state) -> str`.

- [ ] **Step 1: Failing tests** — create `tests/test_fpu_state_hash.py` covering the field inventory + independent-field mutation + NN-input equality (`state.to_tensor()`) + transposition (fix 10):

```python
import numpy as np
from scripts.GPU.alphazero.game.twixt_state import TwixtState
from scripts.GPU.alphazero.fpu_state_hash import canonical_state_sha1


def _play(moves, active_size=10, max_plies=None):
    s = TwixtState(active_size=active_size, to_move="red", max_plies_limit=max_plies)
    for m in moves: s = s.apply_move(m)
    return s


def test_equal_hash_implies_equal_side_legal_terminal_and_nninput():
    a, b = _play([(3, 3), (5, 5), (4, 6)]), _play([(3, 3), (5, 5), (4, 6)])
    assert canonical_state_sha1(a) == canonical_state_sha1(b)
    assert a.to_move == b.to_move
    assert set(a.legal_moves()) == set(b.legal_moves())
    assert a.is_terminal() == b.is_terminal() and a.winner() == b.winner()
    assert np.array_equal(a.to_tensor(), b.to_tensor())              # NN input identical


def test_transposition_same_state_same_hash():
    # two independent-legal moves in either order reach the same peg set
    ab = _play([(3, 3), (5, 5)])
    ba = _play([(5, 5), (3, 3)])
    assert canonical_state_sha1(ab) == canonical_state_sha1(ba)


def test_each_future_relevant_field_changes_hash():
    base = _play([(3, 3), (5, 5)])
    assert canonical_state_sha1(base) != canonical_state_sha1(_play([(3, 3), (6, 6)]))     # pegs
    assert canonical_state_sha1(base) != canonical_state_sha1(_play([(3, 3), (5, 5)], active_size=12))  # active_size
    assert canonical_state_sha1(base) != canonical_state_sha1(_play([(3, 3), (5, 5)], max_plies=40))    # cap
    # side-to-move: same pegs, different to_move
    import dataclasses
    flipped = dataclasses.replace(base, to_move=("black" if base.to_move == "red" else "red"))
    assert canonical_state_sha1(base) != canonical_state_sha1(flipped)
```

- [ ] **Step 2: Run → fail** (module missing).

- [ ] **Step 3: Implement** — create `fpu_state_hash.py`:
```python
"""Complete-state canonical hash for cross-corpus disjointness (design §2.3).
Covers all future-play-relevant fields (TwixtState has no swap/pie rule; add it
here if ever introduced — guarded by the equal-hash-implies-equal-behavior test)."""
from __future__ import annotations
import hashlib, json


def canonical_state_key(state):
    # Verified TwixtState fields: pegs Dict[(r,c)->player]; bridges Set of canonical
    # ((r1,c1),(r2,c2)) tuples (sortable); to_move/board_size/active_size primitives;
    # max_plies_limit int|None. ply == len(pegs) (captured by pegs). active_size and
    # max_plies_limit affect legality/terminal and are absent from TwixtState.__eq__,
    # so they are included here deliberately. _adj is a derived cache (excluded).
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
        json.dumps(canonical_state_key(state), sort_keys=True).encode()).hexdigest()
```

- [ ] **Step 4: Run → pass**; **Step 5: Commit**
```bash
git add scripts/GPU/alphazero/fpu_state_hash.py tests/test_fpu_state_hash.py
git commit -m "feat(fpu): complete-state canonical hash + field-inventory/NN-input/transposition tests"
```

---

## Task 5: Dev-corpus pure sampler (split classification order + contribution-aware split)

**Files:** Create `scripts/GPU/alphazero/build_fpu_dev_corpus.py` (pure section); Test `tests/test_fpu_dev_corpus.py`.

**Interfaces (produces):** `band_of(n_legal)`, `ply_bucket_of(ply)`; **two-stage classification (fix 6):** `raw_policy_role(normalized_entropy, top1_prior) -> "target"|"control"|None` (no value), `anchor_eligible(root_value_stm) -> bool`; `assign_split(games_profile, alloc, seed) -> {game_idx: split}` (contribution-aware, feasibility-checked, deterministic retry — fix 9); `sample_dev_rows(confirmed, *, seed) -> (rows, stats)`. Constants: `BANDS`, `TARGET_PER_BAND=60`, `CONTROL_PER_BAND=20`, `SPLIT_ALLOC` (§2.2 table), `MIN_PLY_GAP=12`, `MAX_PER_GAME=2`.

- [ ] **Step 1: Failing tests** — create `tests/test_fpu_dev_corpus.py`:
```python
from collections import Counter
from scripts.GPU.alphazero.build_fpu_dev_corpus import (
    band_of, ply_bucket_of, raw_policy_role, anchor_eligible, sample_dev_rows)


def test_band_bucket_and_two_stage_classification():
    assert band_of(250) == "b200_299" and band_of(480) == "b400_plus" and band_of(150) is None
    assert ply_bucket_of(15) == "opening" and ply_bucket_of(91) == "late"
    assert raw_policy_role(0.95, 0.01) == "target"
    assert raw_policy_role(0.80, 0.20) == "control"
    assert raw_policy_role(0.88, 0.03) is None            # neither target nor control band
    assert anchor_eligible(0.20) is True and anchor_eligible(0.30) is False


def test_sampler_deterministic_and_invariants():
    rows_a, _ = sample_dev_rows(_pool(), seed=7)
    rows_b, _ = sample_dev_rows(_pool(), seed=7)
    assert rows_a == rows_b
    assert max(Counter(r["game_idx"] for r in rows_a).values()) <= 2
    bygame = {}
    for r in rows_a: bygame.setdefault(r["game_idx"], []).append(r["position_ply"])
    assert all(all(abs(a-b) >= 12 for i,a in enumerate(v) for b in v[i+1:]) for v in bygame.values())
    gsplit = {}
    for r in rows_a: gsplit.setdefault(r["game_idx"], set()).add(r["split"])
    assert all(len(s) == 1 for s in gsplit.values())      # whole-game split
```
(`_pool()` yields an abundant `confirmed` list — target+control across all three bands and enough distinct games to satisfy the split allocation; each record already has `role` from `raw_policy_role` and passed `anchor_eligible`.)

- [ ] **Step 2: Run → fail**.

- [ ] **Step 3: Implement** the pure section. `raw_policy_role`: `target` iff `normalized_entropy ≥ 0.90 and top1_prior ≤ 0.025`; `control` iff `normalized_entropy < 0.85 or top1_prior ≥ 0.05`; else `None`. `anchor_eligible`: `abs(root_value_stm) ≤ 0.25`. `assign_split`: build each game's `(role, band)` contribution profile, greedily assign whole games to `tuning`/`frozen_check` to satisfy the per-`(role, band, split)` `SPLIT_ALLOC` quotas (deterministic order by game_idx under the seed), run a feasibility check, and on shortfall retry with a deterministic secondary ordering; log any residual shortfall. `sample_dev_rows`: game-first round-robin within each assigned split, ≤2/game, ≥12-ply gap, ~50/50 side (prefer under-represented parity), band/role/split quotas from `SPLIT_ALLOC`, ply-bucket ≤50% cap; emit rows (`case_id, game_idx, replay_path, position_ply, side, ply_bucket, band, role, split, game_result, total_plies, source_corpus_id, canonical_sha1, sample_seed`) + stats (achieved/requested per cell, side balance, shortfalls).

- [ ] **Step 4: Run → pass**; **Step 5: Commit**
```bash
git add scripts/GPU/alphazero/build_fpu_dev_corpus.py tests/test_fpu_dev_corpus.py
git commit -m "feat(fpu): dev-corpus sampler — two-stage role classify + contribution-aware whole-game split"
```

---

## Task 6: Builder shell — two-stage scan (operator-run)

**Files:** Modify `build_fpu_dev_corpus.py` (shell); Test `tests/test_fpu_dev_corpus.py`.

**Interfaces:** `per_ply_n_legal(replay) -> {ply: n_legal}` (from the **per-game JSON** `moves[i]["n_legal"]` — verified present in seed20116; **fallback (fix 7):** if absent, reconstruct every 4th ply via `position_state` and compute `len(legal_moves())`); `enumerate_candidate_plies(replay, stride=4, cap=6)` (1st,5th,9th… ply with `n_legal ≥ 200`, ≤6/game); `_policy_features_from_priors(priors)` (`normalized_entropy=H/ln(n_legal)`, `top1_prior=max`); `load_forbidden_hashes(paths)` (selected-A ∪ v16a canonical hashes); `assert_disjoint(dev_hashes, forbidden)`; reserve targets `RESERVE = {target: 120/band, control: 40/band}` (2× final; fix 8); `main` (operator-run).

- [ ] **Step 1: Failing tests** — append pure tests: `enumerate_candidate_plies` stride+cap over stored `n_legal≥200`; `_policy_features_from_priors` (flat→high entropy/low top1; peaked→top1 0.9); `assert_disjoint` (collision + internal-dup raise).

- [ ] **Step 2: Run → fail**.

- [ ] **Step 3: Implement the shell.** `per_ply_n_legal` reads `moves[i]["n_legal"]` from the per-game JSON, with the reconstruct-fallback. `main` (operator-run): for each source game (ascending) → `enumerate_candidate_plies` → reconstruct via `position_state` + raw-policy forward (`_teacher_infer`) → `raw_policy_role` prefilter (drop `None`) → **anchor confirm** (`search_with_root` @400 sims, config `MCTSConfig(fpu_policy_mass_reduction=None)`, checkpoint `DEFAULT_CHECKPOINT` deferred-imported from `diagnose_fpu_sweep`) → keep `anchor_eligible(root_value_stm)` → compute `canonical_state_sha1`, **discard on collision** with `forbidden ∪ kept` (deterministic continue) → accumulate per `(role, band)` until `RESERVE` reached → `assign_split` + `sample_dev_rows` → if sampling reports a shortfall, **continue scanning deterministically** and re-sample; stop when final quotas filled or corpus exhausted (log shortfall) → `assert_disjoint` (final zero cross-collision + zero internal dup) → write `logs/eval/fpu_dev_corpus/dev_corpus_manifest.csv` + `.meta.json`.

- [ ] **Step 4: Run → pass**; **Step 5: Commit**
```bash
git add scripts/GPU/alphazero/build_fpu_dev_corpus.py tests/test_fpu_dev_corpus.py
git commit -m "feat(fpu): dev-corpus builder shell — two-stage scan, per-ply n_legal(+fallback), reserve loop, disjointness"
```

---

## Task 7: Discovery diagnostic — typed configs, staged r0-qualification, dual-reference gates

**Files:** Create `scripts/GPU/alphazero/diagnose_fpu_policy_mass.py`; Test `tests/test_fpu_diagnostic_modes.py` (create) + extend `tests/test_fpu_trace_observer.py`.

**Interfaces:** `@dataclass(frozen=True) FpuRunConfig(label: str, reduction: float | None)` (fix 3/4: `absolute_off`=`(…, None)`, `r0`=`("r0", 0.0)`, candidates `("r0.10", 0.10)…("r0.75", 0.75)`); `FpuTraceObserver` (records §4 events from `on_root_simulation`, using the **passed** `current_root_leader_move` — fix 11); gate fns `progress(v_off, v_r)` (`V_ref=-0.0451`), `reply_reduction(replies_ref, replies_x)`, `prior_rank(priors, move)` (strictly-greater), `top_share(root)`, `lock_in_event(row)`; `dev_safety_verdict(rows, ref)`, `selected_a_verdict(rows)`; `validate_stage_mode(cases, mode, run_configs)`; `main(argv)` with `--mode {tuning,frozen_check}` and `--stage {controls,candidates}` (fix 5). Operator-run.

- [ ] **Step 1: Failing tests** — create `tests/test_fpu_diagnostic_modes.py`:
```python
import pytest
from scripts.GPU.alphazero.diagnose_fpu_policy_mass import (
    FpuRunConfig, validate_stage_mode, lock_in_event, progress, reply_reduction, prior_rank, GRID)

ABS = FpuRunConfig("absolute_off", None); R0 = FpuRunConfig("r0", 0.0)


def test_typed_configs_distinguish_absolute_off_from_r0():
    assert ABS.reduction is None and R0.reduction == 0.0 and ABS != R0
    assert [c.reduction for c in GRID] == [0.10, 0.20, 0.35, 0.50, 0.75]     # positive, no negatives


def test_progress_reply_rank_lockin():
    assert abs(progress(0.30, 0.13) - (0.30-0.13)/(0.30-(-0.0451))) < 1e-9
    assert abs(reply_reduction(200, 100) - 0.5) < 1e-9
    assert prior_rank({1: 0.5, 2: 0.3, 3: 0.3}, 2) == 2
    base = dict(selected_move_prior_rank=11, selected_move_prior=0.005,
                explored_mass_at_stabilization=0.20, stabilization_sim=80, final_root_top_share=0.95)
    assert lock_in_event(base) and not lock_in_event({**base, "final_root_top_share": 0.89})


def test_stage_mode_enforcement():
    tuning = [{"split": "tuning"}]; frozen = [{"split": "frozen_check"}]
    with pytest.raises(ValueError):                                   # wrong split
        validate_stage_mode(frozen, mode="tuning", run_configs=[ABS, R0] + GRID)
    with pytest.raises(ValueError):                                   # frozen: >1 nonzero
        validate_stage_mode(frozen, mode="frozen_check",
                            run_configs=[ABS, R0, FpuRunConfig("r0.20", 0.20), FpuRunConfig("r0.35", 0.35)])
    validate_stage_mode(tuning, mode="tuning", run_configs=[ABS, R0] + GRID)
    validate_stage_mode(frozen, mode="frozen_check", run_configs=[ABS, R0, FpuRunConfig("r0.20", 0.20)])
```
Extend `tests/test_fpu_trace_observer.py` with `FpuTraceObserver` event tests driven by synthetic `on_root_simulation(count, root, move, leader)` sequences (first-visit sims; 25/50/75% mass-crossing; leader timeline; final-leader last-takeover = stabilization; `None` move ignored but counter advances; leader taken from the passed argument, never recomputed).

- [ ] **Step 2: Run → fail**.

- [ ] **Step 3: Implement** `diagnose_fpu_policy_mass.py`:
  - `FpuRunConfig` + `GRID = [FpuRunConfig(f"r{r}", r) for r in (0.10,0.20,0.35,0.50,0.75)]`, `ABSOLUTE_OFF`, `R0`.
  - `FpuTraceObserver`: incremental explored-mass (add `root.priors[move]` on a move's first visit; ignore `None`), first-visit sims, leader timeline from the **passed** `current_root_leader_move`, 25/50/75% crossing sims, explored-mass-at-first-leader, final-leader last-takeover (stabilization), end-state selected-move prior/rank.
  - Gate fns with exact §6 formulas (`V_ref=-0.0451`); `lock_in_event` (5-condition boolean); `dev_safety_verdict(rows, ref)` (§6.2 rejects) + `selected_a_verdict(rows)` (§6.3 requires).
  - `validate_stage_mode(cases, mode, run_configs)`: all rows' `split` == mode else raise; `tuning` permits `{ABSOLUTE_OFF, R0} ∪ GRID`; `frozen_check` permits `{ABSOLUTE_OFF, R0, exactly-one-nonzero}` else raise; selected-A only in tuning.
  - `main`: `--stage controls` runs only `ABSOLUTE_OFF` + `R0`, evaluates `R0` vs `ABSOLUTE_OFF` on the full §6.2 table, writes a control-result file with `r0_qualified` + `r0_target_lockin_count` + `absoff_target_lockin_count`; `--stage candidates` **requires** that file with `r0_qualified=true` (else refuse), substitutes the frozen lock-in caps, runs the candidates, applies dev-safety vs **both** references + selected-A vs `absolute_off`, and reports the smallest-safe-passing `r`. Each root run attaches an `FpuTraceObserver`; configs built via `dataclasses.replace(cfg, fpu_policy_mass_reduction=c.reduction)` (and the `None` config for `absolute_off`).

- [ ] **Step 4: Run → pass** (`tests/test_fpu_diagnostic_modes.py tests/test_fpu_trace_observer.py`); **Step 5: Commit**
```bash
git add scripts/GPU/alphazero/diagnose_fpu_policy_mass.py tests/test_fpu_diagnostic_modes.py tests/test_fpu_trace_observer.py
git commit -m "feat(fpu): staged diagnostic — typed configs, r0-qualification stage, dual-reference §6 gates, observer events"
```

---

## Task 8: Byte-identical proof vs pre-branch golden + full suite

**Files:** none (proof already lands in Task 3's `test_observer_off_output_matches_prebranch_golden`; this task is the consolidated gate).

- [ ] **Step 1: Confirm the pre-branch golden reproduces** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_trace_observer.py::test_observer_off_output_matches_prebranch_golden -q`. Expected: PASS (new code, default config + observer off, equals the Task 0 golden captured from unmodified `mcts.py`).
- [ ] **Step 2: All new suites** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_policy_mass_rule.py tests/test_fpu_state_hash.py tests/test_fpu_dev_corpus.py tests/test_fpu_trace_observer.py tests/test_fpu_diagnostic_modes.py -q`. Expected: PASS.
- [ ] **Step 3: Full repository suite** — `.venv/bin/python -m pytest -p no:cacheprovider tests/ -q`. Expected: PASS (every MCTS/self-play/eval test exercises the `None`/observer-off path unchanged; record any pre-existing unrelated failures).
- [ ] **Step 4: Commit (if any golden regeneration/notes needed; else no-op)** — typically nothing to commit; if the golden needed a note, `git commit`.

---

## Self-review — spec + review-fix coverage

| Item | Task |
|---|---|
| §A rule (helper, field, guard, nonfinite, completed-visits) | 1, 2 |
| Byte-identical off vs **pre-branch** golden (fix 12) | 0, 3, 8 |
| Observer: observer-local counter (fix 1), one-per-completed-sim integration proof (fix 2), canonical leader passed from MCTS (fix 11) | 0, 3 |
| Positive-r typed configs, `absolute_off`≠`r0` (fix 3, 4) | 7 |
| Staged r0-qualification (fix 5) | 7 |
| Two-stage role classify order (fix 6) | 5, 6 |
| Per-ply `n_legal` source + fallback (fix 7) | 6 |
| Reserve exact quotas + scan-until-quota loop (fix 8) | 6 |
| Contribution-aware whole-game split (fix 9) | 5 |
| State-hash frozen fields + strong tests (fix 10) | 4 |
| `-p no:cacheprovider` everywhere (fix 13) | all |
| §6 numeric gates (frozen; lock-in; dual-reference; A formulas) | 7 |
| §7 tooling-only; operator phases not run | all (heavy runs are `main` entry points, never invoked by tests) |

**Placeholder scan:** none — hash fields and evaluator/`__init__` sites are verified against the real code; the only run-time-substituted values are the two lock-in baselines (fed from the controls stage). **Type consistency:** `FpuRunConfig(label, reduction: float|None)` distinguishes `absolute_off`/`r0`/candidates everywhere; observer signature `on_root_simulation(count:int, root, updated_root_move:int|None, current_root_leader_move:int|None)` matches the `_backup` call site and the `FpuTraceObserver`/tests; `raw_policy_role`→`anchor_eligible`→`role` ordering is consistent builder↔sampler.
