# Context-Relative FPU (Policy-Mass) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the *tooling* for the context-relative FPU successor — the opt-in parent-relative + explored-policy-mass rule, a read-only search-trace observer, a geometry development-corpus builder, and a staged, mode-enforced discovery diagnostic — all verified, with the shipped absolute-FPU path proven bit-identical against a **pre-branch** golden. Heavy MCTS phases stay operator-run.

**Architecture:** One small `mcts.py` change (opt-in `MCTSConfig` field + `_select_child` branch + a guarded `_backup` observer hook passing the canonical visit-leader + pure helpers), two operator-run scripts (corpus builder, staged diagnostic), and a shared complete-state hash. All MCTS-heavy work is behind opt-in flags, exercised only in operator phases; tests use pure functions, a CPU fake evaluator, and synthetic trees — no GPU/MLX.

**Tech Stack:** Python 3 stdlib (dataclasses, math, hashlib, json, csv, argparse, random, struct) + numpy (existing dep). Tests via `.venv/bin/python -m pytest -p no:cacheprovider`. Frozen spec: `docs/superpowers/specs/2026-07-10-context-relative-fpu-policy-mass-design.md`.

## Global Constraints

- **Bit-identical off** proven against a **pre-branch golden captured before any source edit** (Task 0), using exact float bits (`.hex()`), `root.visit_count`, and the ordered visit distribution — not rounded values.
- **`0.0` ≠ `None`:** distinct typed configs; `0.0` enabled (`FPU=Q_parent`), `None` keeps the absolute `fpu_value` path.
- **Mutual exclusion / bounds:** raise if `fpu_policy_mass_reduction is not None` and `fpu_value != 0.0`; raise if non-finite or `< 0`. Coefficient nonnegative; grid `0.10,0.20,0.35,0.50,0.75`.
- **Completed visits only** for `P_explored`.
- **Observer:** read-only; the completed-count attr is set **only when an observer is attached** (observer-off mutates nothing); one callback per completed simulation carrying the canonical MCTS visit-leader (from `mcts.py`, not a diagnostic module); exceptions abort the diagnostic.
- **Exact protocol sets:** stage+mode enforcement uses **exact** config sets (not subset/superset); all §6 gate thresholds are frozen; the controls stage persists **full joinable per-position rows** + a fingerprint; the candidate stage refuses stale/mismatched controls.
- **Do NOT modify** `self_play.py`, `SIMS_TABLE`, trainer, network, promotion, calibration manifests, value-adapter/projection. No self-play adoption.
- **Do NOT run** operator phases (geometry scan, corpus generation, coefficient sweep, selection, frozen-check, held-out validation).
- **Tests** always run with `-p no:cacheprovider`, from repo root.

## File structure

- **Modify** `scripts/GPU/alphazero/mcts.py` — field + `__post_init__` guard; helpers `policy_mass_fpu`, `explored_policy_mass`, `visit_leader_move`; `_select_child` branch; `MCTS.__init__(observer=None)` + guarded `_backup` callback.
- **Create** `scripts/GPU/alphazero/fpu_state_hash.py`.
- **Create** `scripts/GPU/alphazero/build_fpu_dev_corpus.py` (operator-run).
- **Create** `scripts/GPU/alphazero/diagnose_fpu_policy_mass.py` (operator-run).
- **Create** `tests/fpu_search_fixture.py`; golden `tests/golden/fpu_prebranch_search.json`.
- **Create tests** `tests/test_fpu_policy_mass_rule.py`, `tests/test_fpu_state_hash.py`, `tests/test_fpu_dev_corpus.py`, `tests/test_fpu_trace_observer.py`, `tests/test_fpu_diagnostic_modes.py`.

---

## Task 0: Pre-branch golden + CPU search fixture (SETUP — before any source edit)

**Files:** Create `tests/fpu_search_fixture.py`, `tests/golden/fpu_prebranch_search.json`.

- [ ] **Step 1: Write the fixture** (fix 1: apply `n_sims`; fix 4: bit-exact fingerprint) — `tests/fpu_search_fixture.py`:

```python
"""CPU fake evaluator + deterministic search harness (no GPU/MLX). Uniform priors +
value 0.0; with a fixed rng seed the real search is fully deterministic, so its output
is a bit-exact fingerprint."""
import dataclasses, random
import numpy as np
from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
from scripts.GPU.alphazero.game.twixt_state import TwixtState


class FakeEvaluator:
    network = None
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
    st = TwixtState(active_size=active_size, to_move="red", max_plies_limit=None)
    for m in moves:
        st = st.apply_move(m)
    cfg = dataclasses.replace(config or MCTSConfig(), n_simulations=n_sims)   # FIX: apply n_sims
    mcts = MCTS(FakeEvaluator(), cfg, random.Random(seed),
                **({"observer": observer} if observer is not None else {}))
    visit_counts, root_value, root = mcts.search_with_root(st, add_noise=False)
    assert root.visit_count == n_sims, (root.visit_count, n_sims)
    assert sum(visit_counts.values()) == n_sims               # each sim descends one root child
    fp = {
        "n_sims": n_sims,
        "root_visit_count": int(root.visit_count),
        "root_value_hex": float(root_value).hex(),            # bit-exact, not rounded (fix 4)
        "visits": [[f"{r},{c}", int(v)] for (r, c), v in sorted(visit_counts.items())],
    }
    return fp, root, mcts
```

(The `**({"observer": …})` passes `observer` only when set, so this fixture works both before and after Task 3 adds the param.)

- [ ] **Step 2: Capture the golden from UNMODIFIED mcts.py**

```bash
mkdir -p tests/golden
.venv/bin/python - <<'PY'
import json
from tests.fpu_search_fixture import run_search
out, _root, _m = run_search()
json.dump(out, open("tests/golden/fpu_prebranch_search.json", "w"), indent=2, sort_keys=True)
print(out)
PY
```
Expected: `root_visit_count == 200`, `visits` sum to 200, a `root_value_hex` string. Eyeball.

- [ ] **Step 3: Commit** — `git add tests/fpu_search_fixture.py tests/golden/fpu_prebranch_search.json && git commit -m "test(fpu): pre-branch bit-exact search golden + CPU fake-evaluator fixture"`

---

## Task 1: Rule helper + config field + guard

(unchanged from prior plan — implementation is stable.)

**Files:** Modify `mcts.py`; Test `tests/test_fpu_policy_mass_rule.py` (create).

- [ ] **Step 1: Failing tests** — `tests/test_fpu_policy_mass_rule.py`: `policy_mass_fpu` formula/clamp; nonfinite reject (all three args × NaN/inf); `MCTSConfig()` default `None`; `0.0` enabled ≠ `None`; guard (nonzero `fpu_value` + reduction raises; negative/inf raises; `replace` path ok). *(Same test bodies as the prior plan revision.)*
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** — module `policy_mass_fpu` (finite-guard + clamp + `parent_q - r*sqrt(m)`); `MCTSConfig.fpu_policy_mass_reduction: float | None = None` after `fpu_value`; `__post_init__` adds the mutual-exclusion + finite/`>=0` checks.
- [ ] **Step 4: Run → pass.** **Step 5: Commit** — `feat(fpu): policy-mass helper + opt-in nonnegative field + mutual-exclusion guard`.

---

## Task 2: `_select_child` branch + completed-visit `P_explored`

(unchanged.)

**Files:** Modify `mcts.py`; Test `tests/test_fpu_policy_mass_rule.py`.

- [ ] **Step 1: Failing test** — `explored_policy_mass(node)` counts only completed-visit children (zero-visit excluded).
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** — `explored_policy_mass` helper; in `_select_child` after `sqrt_parent`, compute `_pm = self.config.fpu_policy_mass_reduction` and `_fpu_pm = policy_mass_fpu(node.q_value, explored_policy_mass(node), _pm) if _pm is not None else None`; unvisited branch `q = _fpu_pm if _pm is not None else self.config.fpu_value`.
- [ ] **Step 4: Run → pass**; `tests/test_fpu_value.py tests/test_fpu_sweep.py` pass. **Step 5: Commit** — `feat(fpu): _select_child policy-mass branch + completed-visit P_explored`.

---

## Task 3: Observer hook (per-completed-sim, canonical leader) + integration proof

**Files:** Modify `mcts.py`; Test `tests/test_fpu_trace_observer.py` (create).

**Interfaces:** `visit_leader_move(node)`; `MCTS.__init__(..., observer=None)` (fix 5b: completed-count attr set only when observer attached); guarded `_backup` tail firing `on_root_simulation(count, root, updated_root_move, current_root_leader_move)`.

- [ ] **Step 1: Failing tests** — `tests/test_fpu_trace_observer.py`:
  - Direct `_backup` unit tests: one call per `_backup`; `updated_root_move = search_path[1].move` (or `None` for a length-1 path); observer-off (`None`) path mutates nothing and does not raise.
  - `visit_leader_move`: max-visit; lowest-id tie-break; `None` when no visited child.
  - **Root pre-expansion invariant (fix 5a):**
    ```python
    from tests.fpu_search_fixture import run_search
    class _Spy:
        def __init__(self): self.calls = []
        def on_root_simulation(self, count, root, move, leader): self.calls.append((count, move, leader))

    def test_search_with_root_pre_expands_root_so_first_sim_has_a_root_move():
        spy = _Spy(); _out, root, _m = run_search(n_sims=1, observer=spy)
        assert len(spy.calls) == 1 and spy.calls[0][0] == 1
        assert spy.calls[0][1] is not None            # proves root was expanded before sim 1
    ```
  - **Integration (fix 2/5a):**
    ```python
    from scripts.GPU.alphazero.mcts import visit_leader_move
    def test_one_callback_per_completed_sim():
        spy = _Spy(); _out, root, _m = run_search(n_sims=200, observer=spy)
        assert [c for c, _, _ in spy.calls] == list(range(1, 201))     # exactly 1..n, no gaps/dups
        legal = set(root.priors.keys())
        assert all((m is None) or (m in legal) for _, m, _ in spy.calls)   # None allowed by contract
        assert spy.calls[-1][2] == visit_leader_move(root)            # final leader == final visit leader
    ```
  - **Observer-off reproduces the pre-branch golden:**
    ```python
    import json
    def test_observer_off_matches_prebranch_golden():
        out, _r, _m = run_search()                    # default config, observer None
        assert out == json.load(open("tests/golden/fpu_prebranch_search.json"))
    ```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement** — `visit_leader_move` (max `visit_count`, ties→lowest move id, `None` if none visited). In `MCTS.__init__` (:211) add `observer=None`:
```python
        self._observer = observer
        if observer is not None:
            self._observer_completed_count = 0      # observer-off mutates nothing (fix 5b)
```
`_backup` tail:
```python
        if self._observer is not None:
            self._observer_completed_count += 1
            root = search_path[0]
            move = search_path[1].move if len(search_path) >= 2 else None
            self._observer.on_root_simulation(
                self._observer_completed_count, root, move, visit_leader_move(root))
```
(search_with_root calls `_expand(root)` before the sim loop, so on that path every descent has a root move — proven by the `n_sims=1` test; the `None` branch remains correct for any path that backs up a bare root.)

- [ ] **Step 4: Run → pass.** **Step 5: Commit** — `feat(fpu): per-completed-sim observer hook + canonical visit-leader; pre-expansion + integration + golden proof`.

---

## Task 4: Complete-state canonical hash

**Files:** Create `scripts/GPU/alphazero/fpu_state_hash.py`; Test `tests/test_fpu_state_hash.py`.

- [ ] **Step 1: Failing tests** — `tests/test_fpu_state_hash.py`:
```python
import dataclasses
import numpy as np
from scripts.GPU.alphazero.game.twixt_state import TwixtState
from scripts.GPU.alphazero.fpu_state_hash import canonical_state_sha1

def _play(moves, active_size=10, max_plies=None):
    s = TwixtState(active_size=active_size, to_move="red", max_plies_limit=max_plies)
    for m in moves: s = s.apply_move(m)
    return s

def test_equal_hash_implies_equal_behavior_and_nninput():
    a, b = _play([(3, 3), (5, 5), (4, 6)]), _play([(3, 3), (5, 5), (4, 6)])
    assert canonical_state_sha1(a) == canonical_state_sha1(b)
    assert a.to_move == b.to_move and set(a.legal_moves()) == set(b.legal_moves())
    assert a.is_terminal() == b.is_terminal() and a.winner() == b.winner()
    assert np.array_equal(a.to_tensor(), b.to_tensor())

def test_transposition_same_state_same_hash():
    # FIX 6: reorder ONLY within each player's turns (alternation preserves ownership).
    # red gets {(2,2),(2,7)}, black gets {(7,7),(7,2)} in BOTH orders; interior, no bridges.
    a = _play([(2, 2), (7, 7), (2, 7), (7, 2)])       # red A, black B, red C, black D
    b = _play([(2, 7), (7, 2), (2, 2), (7, 7)])       # red C, black D, red A, black B
    assert canonical_state_sha1(a) == canonical_state_sha1(b)
    assert a.pegs == b.pegs and a.bridges == b.bridges
    assert a.to_move == b.to_move and set(a.legal_moves()) == set(b.legal_moves())
    assert a.is_terminal() == b.is_terminal() and np.array_equal(a.to_tensor(), b.to_tensor())

def test_each_future_relevant_field_changes_hash():
    base = _play([(3, 3), (5, 5)])
    assert canonical_state_sha1(base) != canonical_state_sha1(_play([(3, 3), (6, 6)]))            # pegs
    assert canonical_state_sha1(base) != canonical_state_sha1(_play([(3, 3), (5, 5)], active_size=12))
    assert canonical_state_sha1(base) != canonical_state_sha1(_play([(3, 3), (5, 5)], max_plies=40))
    flip = dataclasses.replace(base, to_move=("black" if base.to_move == "red" else "red"))
    assert canonical_state_sha1(base) != canonical_state_sha1(flip)                               # side
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** `fpu_state_hash.py` — `canonical_state_key(state)` = `(board_size, active_size, to_move, sorted((r,c,player)), sorted(bridges), max_plies_limit)`; `canonical_state_sha1` = sha1 of `json.dumps(key, sort_keys=True)`. *(Fields verified against `TwixtState`; `_adj` is a derived cache, excluded; `ply == len(pegs)` captured by pegs.)*
- [ ] **Step 4: Run → pass.** **Step 5: Commit** — `feat(fpu): complete-state canonical hash + field-inventory/NN-input/transposition tests`.

---

## Task 5: Dev-corpus pure sampler (two-stage classify + contribution-aware split + exact composition)

**Files:** Create `scripts/GPU/alphazero/build_fpu_dev_corpus.py` (pure section); Test `tests/test_fpu_dev_corpus.py`.

**Interfaces:** `band_of`, `ply_bucket_of`; `raw_policy_role(normalized_entropy, top1_prior)`; `anchor_eligible(root_value_stm)`; `assign_split(games_profile, seed)`; `sample_dev_rows(confirmed, *, seed) -> (rows, stats)`; constants `BANDS`, `TARGET_PER_BAND=60`, `CONTROL_PER_BAND=20`, `MIN_PLY_GAP=12`, `MAX_PER_GAME=2`, `SIDE_TOL`, and the **frozen** `SPLIT_ALLOC` (fix 7):

```python
SPLIT_ALLOC = {   # (role, band) -> {"tuning": n, "frozen_check": n}
    ("target",  "b200_299"): {"tuning": 40, "frozen_check": 20},
    ("target",  "b300_399"): {"tuning": 40, "frozen_check": 20},
    ("target",  "b400_plus"): {"tuning": 40, "frozen_check": 20},
    ("control", "b200_299"): {"tuning": 13, "frozen_check": 7},
    ("control", "b300_399"): {"tuning": 13, "frozen_check": 7},
    ("control", "b400_plus"): {"tuning": 14, "frozen_check": 6},
}
```

- [ ] **Step 1: Failing tests** — `tests/test_fpu_dev_corpus.py`:
  - two-stage classify: `raw_policy_role(0.95,0.01)=="target"`, `raw_policy_role(0.80,0.20)=="control"`, `raw_policy_role(0.88,0.03) is None`; `anchor_eligible(0.20) is True`, `anchor_eligible(0.30) is False`; band/bucket boundaries.
  - determinism; ≤2/game; ≥12-ply gap; whole-game split isolation.
  - **exact composition (fix 7):**
    ```python
    from collections import Counter
    def test_exact_split_composition_and_totals():
        rows, stats = sample_dev_rows(_abundant_pool(), seed=1)
        cell = Counter((r["role"], r["band"], r["split"]) for r in rows)
        for (role, band), alloc in SPLIT_ALLOC.items():
            for split, n in alloc.items():
                assert cell[(role, band, split)] == n            # every cell EXACTLY full
        assert sum(1 for r in rows if r["split"] == "tuning") == 160
        assert sum(1 for r in rows if r["split"] == "frozen_check") == 80
        assert sum(1 for r in rows if r["role"] == "target") == 180
        assert sum(1 for r in rows if r["role"] == "control") == 60
        assert len({r["canonical_sha1"] for r in rows}) == len(rows)   # no dup hash
        assert max(Counter(r["ply_bucket"] for r in rows).values()) <= 0.5 * len(rows)
        for split in ("tuning", "frozen_check"):
            sc = Counter(r["side"] for r in rows if r["split"] == split)
            assert abs(sc["red"] - sc["black"]) <= SIDE_TOL

    def test_shortfall_on_final_manifest_is_an_error():
        import pytest
        with pytest.raises(ValueError):
            sample_dev_rows(_insufficient_pool(), seed=1)          # cannot fill a cell -> raise
    ```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** the pure section. `raw_policy_role`: target iff `entropy≥0.90 and top1≤0.025`; control iff `entropy<0.85 or top1≥0.05`; else `None`. `anchor_eligible`: `abs(root_value_stm)≤0.25`. `assign_split`: build each game's `(role,band)` contribution profile; greedily assign whole games to satisfy per-`(role,band,split)` `SPLIT_ALLOC` quotas (deterministic by game_idx under seed); feasibility check + deterministic secondary-ordering retry. `sample_dev_rows`: round-robin within assigned split, ≤2/game, ≥12-gap, side-balance, ply-bucket ≤50% cap; **every `SPLIT_ALLOC` cell must be filled exactly or raise `ValueError`** (final manifest shortfall is an error); return rows + stats.
- [ ] **Step 4: Run → pass.** **Step 5: Commit** — `feat(fpu): dev-corpus sampler — two-stage classify, contribution-aware split, exact frozen composition`.

---

## Task 6: Builder shell — two-stage scan (operator-run)

**Files:** Modify `build_fpu_dev_corpus.py`; Test `tests/test_fpu_dev_corpus.py`.

**Interfaces:** `per_ply_n_legal(replay)` (from per-game JSON `moves[i]["n_legal"]`, verified present in seed20116; **fallback (fix 7 prior):** reconstruct every 4th ply and compute `len(legal_moves())`); `enumerate_candidate_plies(replay, stride=4, cap=6)`; `_policy_features_from_priors(priors)`; `load_forbidden_hashes(paths)` (selected-A ∪ v16a); `assert_disjoint(dev_hashes, forbidden)`; `RESERVE={"target":120,"control":40}` per band; `main` (operator-run).

- [ ] **Step 1: Failing tests** — pure: `enumerate_candidate_plies` (1st/5th/9th… ply with `n_legal≥200`, cap 6); `_policy_features_from_priors` (flat→high entropy/low top1; peaked→top1 0.9); `assert_disjoint` (collision + internal-dup raise).
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement the shell** — `per_ply_n_legal` (+fallback); `main`: per source game → `enumerate_candidate_plies` → reconstruct + raw-policy (`_teacher_infer`) → `raw_policy_role` prefilter → **anchor confirm** (`search_with_root` @400, `MCTSConfig(fpu_policy_mass_reduction=None)`, `DEFAULT_CHECKPOINT`) → keep `anchor_eligible` → `canonical_state_sha1`, **discard on collision** with `forbidden ∪ kept` → accumulate per `(role,band)` until `RESERVE` → `assign_split`+`sample_dev_rows`; if sampling raises shortfall, continue scanning deterministically and re-sample; stop when filled or corpus exhausted → `assert_disjoint` → write `logs/eval/fpu_dev_corpus/dev_corpus_manifest.csv` + `.meta.json`.
- [ ] **Step 4: Run → pass.** **Step 5: Commit** — `feat(fpu): dev-corpus builder shell — two-stage scan, per-ply n_legal(+fallback), reserve loop, disjointness`.

---

## Task 7: Staged diagnostic — typed configs, exact stage+mode sets, joinable controls, dual-reference gate boundaries

**Files:** Create `scripts/GPU/alphazero/diagnose_fpu_policy_mass.py`; Test `tests/test_fpu_diagnostic_modes.py` (create) + extend `tests/test_fpu_trace_observer.py`.

**Interfaces:**
- `@dataclass(frozen=True) FpuRunConfig(label: str, reduction: float | None)` with **explicit labels (fix 9):**
  ```python
  ABSOLUTE_OFF = FpuRunConfig("absolute_off", None)
  R0 = FpuRunConfig("r0", 0.0)
  GRID = (FpuRunConfig("r0.10", 0.10), FpuRunConfig("r0.20", 0.20),
          FpuRunConfig("r0.35", 0.35), FpuRunConfig("r0.50", 0.50), FpuRunConfig("r0.75", 0.75))
  ```
- `FpuTraceObserver` (records §4 events; leader taken from the **passed** `current_root_leader_move`).
- Gate fns (exact §6): `progress(v_off, v_r)` (`V_REF=-0.0451`), `reply_reduction(replies_ref, replies_x)`, `prior_rank(priors, move)` (strictly-greater), `top_share(root)`, `lock_in_event(row)`, `dev_safety_verdict(rows, ref, r0_lockin, absoff_lockin)`, `selected_a_verdict(rows)`.
- `validate_stage_mode(cases, *, mode, stage, run_configs)` (fix 3): **exact** sets —
  ```
  (tuning, controls):       {ABSOLUTE_OFF, R0}                 exactly
  (tuning, candidates):     set(GRID)                          exactly (+ valid controls artifact)
  (frozen_check, controls): {ABSOLUTE_OFF, R0}                 exactly
  (frozen_check, candidates): {exactly one nonzero r}          (+ matching frozen-split controls)
  ```
  and every row's `split == mode` else raise.
- Controls artifact (fix 2): `controls_cases.csv` (full per-position rows for `absolute_off` AND `r0`, keyed by `canonical_sha1`), `controls_summary.csv`, `controls_gate.json` (`r0_qualified`, `r0_target_lockin_count`, `absoff_target_lockin_count`, + fingerprint: dev-manifest sha1, selected-A manifest sha1, checkpoint identity, `mcts_sims=400` + search cfg, seeds/batching, git commit, observer/schema version).

- [ ] **Step 1: Failing tests** — `tests/test_fpu_diagnostic_modes.py`:
```python
import pytest
from scripts.GPU.alphazero.diagnose_fpu_policy_mass import (
    FpuRunConfig, ABSOLUTE_OFF, R0, GRID, validate_stage_mode,
    lock_in_event, progress, reply_reduction, prior_rank,
    dev_safety_verdict, selected_a_verdict)

def test_labels_and_grid_are_explicit_and_positive():
    assert [c.label for c in GRID] == ["r0.10", "r0.20", "r0.35", "r0.50", "r0.75"]
    assert [c.reduction for c in GRID] == [0.10, 0.20, 0.35, 0.50, 0.75]
    assert ABSOLUTE_OFF.reduction is None and R0.reduction == 0.0 and ABSOLUTE_OFF != R0

def test_stage_mode_exact_sets():
    tun = [{"split": "tuning"}]; frz = [{"split": "frozen_check"}]
    validate_stage_mode(tun, mode="tuning", stage="controls", run_configs=[ABSOLUTE_OFF, R0])
    validate_stage_mode(tun, mode="tuning", stage="candidates", run_configs=list(GRID))
    validate_stage_mode(frz, mode="frozen_check", stage="candidates",
                        run_configs=[FpuRunConfig("r0.20", 0.20)])
    for bad in (
        dict(cases=frz, mode="tuning", stage="controls", run_configs=[ABSOLUTE_OFF, R0]),   # wrong split
        dict(cases=tun, mode="tuning", stage="controls", run_configs=[ABSOLUTE_OFF]),        # not exact set
        dict(cases=tun, mode="tuning", stage="candidates", run_configs=[ABSOLUTE_OFF]+list(GRID)),  # superset
        dict(cases=frz, mode="frozen_check", stage="candidates",
             run_configs=[FpuRunConfig("r0.20",0.20), FpuRunConfig("r0.35",0.35)]),          # >1 nonzero
    ):
        with pytest.raises(ValueError):
            validate_stage_mode(**bad)

def test_formula_exactness():
    assert abs(progress(0.30, 0.13) - (0.30-0.13)/(0.30-(-0.0451))) < 1e-9
    assert abs(reply_reduction(200, 100) - 0.5) < 1e-9
    assert prior_rank({1: 0.5, 2: 0.3, 3: 0.3}, 2) == 2

def test_gate_boundaries_table_driven():          # fix 8 — the executable preregistration
    assert lock_in_event(dict(selected_move_prior_rank=11, selected_move_prior=0.005,
        explored_mass_at_stabilization=0.20, stabilization_sim=80, final_root_top_share=0.95))
    assert not lock_in_event(dict(selected_move_prior_rank=10, selected_move_prior=0.005,
        explored_mass_at_stabilization=0.20, stabilization_sim=80, final_root_top_share=0.95))
    # dev-safety: exact reject boundaries (helpers below build minimal row sets)
    assert _dev_rejects(new_collapse_rate=0.05)      and not _dev_rejects(new_collapse_rate=0.0499)
    assert _dev_rejects(band_new_collapse_rate=0.10) and not _dev_rejects(band_new_collapse_rate=0.0999)
    assert _dev_rejects(lockin_count=lambda base: base+3) and not _dev_rejects(lockin_count=lambda base: base+2)
    assert _dev_rejects(p95_mover_delta=0.35)        and not _dev_rejects(p95_mover_delta=0.349)
    assert _dev_rejects(eff_reduction=0.50, top_share_inc=0.15)          # compound: both
    assert not _dev_rejects(eff_reduction=0.50, top_share_inc=0.14)
    assert _dev_rejects(control_lowprior_flip_rate=0.10) and not _dev_rejects(control_lowprior_flip_rate=0.099)
    # selected-A: exact pass boundaries
    assert _a_passes(reply_reduction=0.50, progress=0.50, a_new_collapse=2, a_top_share_inc=0.15)
    assert not _a_passes(reply_reduction=0.49, progress=0.50, a_new_collapse=2, a_top_share_inc=0.15)
    assert not _a_passes(reply_reduction=0.50, progress=0.49, a_new_collapse=2, a_top_share_inc=0.15)
    assert not _a_passes(reply_reduction=0.50, progress=0.50, a_new_collapse=3, a_top_share_inc=0.15)
    assert not _a_passes(reply_reduction=0.50, progress=0.50, a_new_collapse=2, a_top_share_inc=0.16)
```
(`_dev_rejects`/`_a_passes` are thin test builders constructing minimal `rows` that isolate one metric and calling `dev_safety_verdict`/`selected_a_verdict`; the candidate-stage-refuses-stale-controls and `r0`-fail-blocks-candidates behaviors get their own tests using a fabricated controls_gate.json fingerprint.) Extend `tests/test_fpu_trace_observer.py` with `FpuTraceObserver` event tests (first-visit sims; 25/50/75% crossings; leader timeline; final-leader last-takeover=stabilization; `None` ignored; leader from the passed arg).

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** `diagnose_fpu_policy_mass.py`:
  - Typed configs (explicit labels above).
  - `FpuTraceObserver`: incremental explored-mass; first-visit sims; leader timeline from the **passed** leader; 25/50/75% crossings; explored-mass-at-first-leader; final-leader last-takeover; end-state selected prior/rank.
  - Gate fns with exact frozen boundaries: new-collapse reject `>= 0.05` (target) / `>= 0.10` (band); lock-in reject `count > baseline + 2`; p95 mover-delta reject `>= 0.35`; compound reject `eff_reduction >= 0.50 AND top_share_inc >= 0.15`; control-flip reject `>= 0.10`; selected-A require `reply_reduction >= 0.50 AND progress >= 0.50 AND a_new_collapse <= 2 AND a_top_share_inc <= 0.15`. `dev_safety_verdict` runs vs **both** references.
  - `validate_stage_mode(cases, *, mode, stage, run_configs)`: exact-set enforcement per the table; `split == mode` for all rows.
  - `main(--mode, --stage)`: `controls` runs `{ABSOLUTE_OFF, R0}`, writes `controls_cases.csv`/`controls_summary.csv`/`controls_gate.json` (full joinable rows + fingerprint), sets `r0_qualified` via `dev_safety_verdict(r0_rows, ref=absolute_off,…)`; `candidates` loads the controls artifact, **validates the fingerprint** (refuse stale/mismatch), **requires `r0_qualified=true`** (else refuse), joins control rows by `canonical_sha1`, runs the grid (tuning) / one r (frozen), computes dual-reference deltas, applies gates, reports the smallest-safe-passing `r`. Each root run attaches an `FpuTraceObserver`; configs via `dataclasses.replace(cfg, fpu_policy_mass_reduction=c.reduction)` (and `None` for `absolute_off`).
- [ ] **Step 4: Run → pass** (`tests/test_fpu_diagnostic_modes.py tests/test_fpu_trace_observer.py`). **Step 5: Commit** — `feat(fpu): staged diagnostic — typed configs, exact stage+mode sets, joinable fingerprinted controls, gate-boundary tests`.

---

## Task 8: Bit-identical proof vs pre-branch golden + full suite

**Files:** none (the golden proof lands in Task 3).

- [ ] **Step 1** — `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_trace_observer.py::test_observer_off_matches_prebranch_golden -q`. Expected: PASS.
- [ ] **Step 2** — new suites: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_policy_mass_rule.py tests/test_fpu_state_hash.py tests/test_fpu_dev_corpus.py tests/test_fpu_trace_observer.py tests/test_fpu_diagnostic_modes.py -q`. Expected: PASS.
- [ ] **Step 3** — full suite: `.venv/bin/python -m pytest -p no:cacheprovider tests/ -q`. Expected: PASS (record any pre-existing unrelated failures).
- [ ] **Step 4** — nothing to commit unless notes are added.

---

## Self-review — review-fix coverage

| Fix | Task |
|---|---|
| 1 `run_search` applies `n_sims` + visit-count assert | 0 |
| 2 controls stage persists full joinable rows + fingerprint; candidate joins/validates | 7 |
| 3 `validate_stage_mode(*, mode, stage, run_configs)` exact sets | 7 |
| 4 golden uses exact float bits (`.hex()`) + visit_count + ordered visits | 0, 3 |
| 5a root-only observer lifecycle: permissive assert + pre-expansion proof (`n_sims=1`) | 3 |
| 5b observer-off stores no completed-count attr | 3 |
| 6 transposition test reorders within each player's turns (4 moves) | 4 |
| 7 exact frozen `SPLIT_ALLOC` + composition tests + shortfall-is-error | 5 |
| 8 table-driven gate-boundary tests (inclusive/exclusive) | 7 |
| 9 explicit `FpuRunConfig` labels (`r0.10`…) | 7 |
| `-p no:cacheprovider` everywhere | all |

**Placeholder scan:** none. **Type consistency:** `FpuRunConfig(label, reduction: float|None)` distinguishes all three config kinds; observer `on_root_simulation(count, root, updated_root_move:int|None, current_root_leader_move:int|None)` matches `_backup` and `FpuTraceObserver`; `raw_policy_role`→`anchor_eligible`→`role`→`SPLIT_ALLOC` cells are consistent builder↔sampler↔diagnostic; `canonical_sha1` is the join key across corpus, controls artifact, and disjointness.
