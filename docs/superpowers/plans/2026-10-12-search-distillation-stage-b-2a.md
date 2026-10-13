# Search Distillation Stage B — Checkpoint 2A Plan: label adapter, soft temperature fit, stable KL

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three pieces to the Stage B module, each checked on CPU:

1. A label adapter that runs the **real** sequential search with a deterministic fake evaluator.
2. A soft-target temperature fit.
3. KL computed stably from logits. This fixes the `log(0)` failure in checkpoint 1's `game_measures`.

**Architecture:** All new functions go into the existing `scripts/GPU/alphazero/search_distillation_stage_b.py`. New tests are appended to `tests/test_search_distillation_stage_b.py`. The rest of the code is reused unchanged:

- `MCTS.search_with_root`
- `eval_runner.cfg_from`, `EvalConfig`
- `external_learning_stage3.PROTOCOL`
- `external_learning.BETA_*` constants
- the committed CPU `FakeEvaluator` in `tests/fpu_search_fixture.py`

MCTS itself is not modified. There is no settings grid.

**Tech Stack:** Python stdlib, numpy (fake evaluator only), pytest. No MLX: importing `mcts` and `eval_runner` was checked to load no MLX.

**Spec:** `docs/superpowers/2026-10-12-search-distillation-stage-b-card.md` §5, §7, §8 (committed `a9e16fc9`). It builds on checkpoint 1 (`fd435150`).

## Global Constraints

- **Approved:** this plan only. Its implementation and execution need review first.
- **Not approved:**
  - phase scripts
  - model loading
  - seed accounting
  - corpus generation
  - labelling
  - training
- **CPU only.** No checkpoint, no real evaluator, no file outside `tmp_path`, no MCTS edit.
- **Fixed simulation ceiling: 2,800 simulations in the whole test file.** These are exactly four real searches, all in one module-scoped fixture (`runs`):

  | Search | Simulations |
  |---|---|
  | 1,600, instrumented | 1,600 |
  | 400 | 400 |
  | 400, same seed, **through `label(...)`** | 400 |
  | 400, another seed (descriptive only) | 400 |

  No other test calls the search.
- **Bounded runs.** One red run and one green run, each under `timeout 120`, with both exit statuses captured. An unexpected failure or timeout stops for review, with no extra runs.
- **Test command:**
  `timeout 120 env PYTHONPATH=. .venv/bin/python -B -m pytest tests/test_search_distillation_stage_b.py -q -p no:cacheprovider`
- **No seed-band literal.** Fixtures use the seed `1000`. Afterwards, scan both files with the pinned scanner.
- **Repository hygiene.** Keep the capture work out. Commit only when approved, and never push.

## Review Focus

- **The 400-simulation prefix.** It holds only if nothing in the search depends on `n_simulations` before the last simulation. The test detects any such dependence; it does not assume its absence. If the prefix fails, that is a design finding for the card (§5 common random numbers), not something to patch around.
- **Visit sums.** Root children must sum to exactly `sims`. If they do not, every label would be refused by `sparse_label`, so this must be seen on the real search, not assumed.
- **Another seed is descriptive, not a requirement** (review correction). Identical counts across seeds would not invalidate same-seed repeatability. The fourth search is kept; the repeat test prints whether its counts differ and asserts nothing about it. No other seed or fixture is tried.
- **The adapter itself is exercised** (review correction). The same-seed search runs through `label(...)`. A test-side wrapper around `label_mcts` captures that search's full visit dict, so the reconstruction test compares `label`'s own sparse output against the search it came from.
- **An upper-bound derivative that rounds to 0 refuses the fit.** `g(100) <= 0` is a refusal, not a finite optimum.
- **Underflow.** Temperature-scaled logits such as `[-1000, 0]` underflow `softmax` to 0. KL must come from a stable log-softmax: about 1000, with no exception.
- **β at an endpoint.** A uniform target is matched exactly at β = 0, which must be accepted. A one-hot target on the top logit never stops decreasing, and the fit must refuse it rather than return 100.

---

## Task 1: Write all checkpoint 2A tests (appended)

**Files:** Modify `tests/test_search_distillation_stage_b.py`. Append the following, and add `import numpy as np` and the fixture import to the top-of-file imports.

```python
import numpy as np

from tests.fpu_search_fixture import FakeEvaluator


# ─── checkpoint 2A: the label adapter on the REAL search (CPU fake evaluator), stable KL, soft beta ───

class CentralFakeEvaluator(FakeEvaluator):
    """Deterministic CPU fake: priors favour the centre and are mirror-symmetric, so exact ties occur; value 0."""

    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        w = move_mask * np.exp(-((move_rows - 11.5) ** 2 + (move_cols - 11.5) ** 2) / 32.0)
        s = w.sum(axis=1, keepdims=True)
        priors = np.divide(w, s, out=np.zeros_like(w), where=s > 0)
        return priors.astype(np.float32), np.zeros(len(move_rows), dtype=np.float32)


SEED = 1000
EV = CentralFakeEvaluator()
STATE = B.replay_state(MOVES[:2])          # 24x24, ply cap 280, red to move


@pytest.fixture(scope="module")
def runs():
    """THE ONLY SEARCHES IN THIS FILE: 1,600 + 400 + 400 + 400 = 2,800 simulations (the plan's ceiling)."""
    m = B.label_mcts(EV, 1600, SEED)
    orig, calls, snap = m._run_single_simulation, [0], {}

    def counted(root, root_move_override=None):         # test-side instrumentation; MCTS is not modified
        orig(root, root_move_override=root_move_override)
        calls[0] += 1
        if calls[0] == 400:
            snap.update({mid: ch.visit_count for mid, ch in root.children.items() if ch.visit_count})

    m._run_single_simulation = counted
    v1600, _, _ = m.search_with_root(STATE, add_noise=False)
    v400, _, root400 = B.label_mcts(EV, 400, SEED).search_with_root(STATE, add_noise=False)

    real, seen = B.label_mcts, {}

    def capturing(evaluator, sims, seed):               # test-side: record the full output of label's own search
        mc = real(evaluator, sims, seed)
        search = mc.search_with_root

        def recorded(state, add_noise=True):
            out = search(state, add_noise=add_noise)
            seen["visits"] = out[0]
            return out

        mc.search_with_root = recorded
        return mc

    B.label_mcts = capturing
    try:
        sparse = B.label(EV, STATE, 400, SEED)
    finally:
        B.label_mcts = real
    other, _, _ = B.label_mcts(EV, 400, SEED + 1).search_with_root(STATE, add_noise=False)
    return {"v1600": v1600, "calls1600": calls[0], "snap400": snap, "v400": v400, "again": seen["visits"],
            "label": sparse, "other": other,
            "root400": {mid: ch.visit_count for mid, ch in root400.children.items() if ch.visit_count}}


def test_label_construction_is_the_frozen_call():
    m = B.label_mcts(EV, 400, SEED)
    assert (m.config.n_simulations, m.config.eval_batch_size, m.config.stall_flush_sims) == (400, 14, 48)
    assert m.rng.getstate() == random.Random(SEED ^ 0xA5A5A5).getstate()


def test_completed_visits_sum_to_sims_at_400_and_1600(runs):
    assert sum(runs["v400"].values()) == 400 and runs["calls1600"] == 1600 and sum(runs["v1600"].values()) == 1600
    assert set(runs["v400"]) == set(STATE.legal_moves())                     # every legal move, zeros included
    assert 0 in runs["v400"].values()


def test_same_seed_repeats_exactly_through_the_adapter(runs, capsys):
    differ = sum(runs["other"][m] != runs["v400"][m] for m in runs["v400"])
    with capsys.disabled():                                                  # descriptive only: no assertion
        print(f"\nDESCRIPTIVE another-seed 400: counts differ = {differ > 0} ({differ} moves differ)")
    assert runs["again"] == runs["v400"]


def test_400_search_is_the_first_400_simulations_of_the_1600_search(runs):
    assert runs["snap400"] == runs["root400"] and sum(runs["snap400"].values()) == 400


def test_label_output_rebuilds_its_own_search_exactly(runs):
    legal = [tuple(m) for m in STATE.legal_moves()]
    got_legal, counts = B.full_counts(STATE, runs["label"], 400, B.legal_digest(legal))
    assert got_legal == legal and counts == [runs["again"][m] for m in legal]


def test_kl_from_logits_is_stable_where_softmax_underflows():
    assert B.kl_logits([1.0, 0.0], [-1000.0, 0.0]) == pytest.approx(1000.0)


def test_kl_from_logits_matches_the_probability_form_on_benign_logits():
    t, z = B.probs([3, 1, 0]), [0.3, -0.2, 1.1]
    assert math.isclose(B.kl_logits(t, z), B.kl(t, B.softmax(z)), rel_tol=1e-12)


def test_game_measures_survives_underflowing_temperature_logits():
    pos = [{"target": [1, 0], "logits": {a: [-1000.0, 0.0] for a in B.ARMS}}]
    m = B.game_measures(pos)
    assert all(m["kl"][a] == pytest.approx(1000.0) for a in B.ARMS)


def test_soft_beta_hand_solution_two_moves():
    # logits [0, 1], target [0.25, 0.75]: softmax(beta z) matches the target at sigma(beta) = 0.75, beta = ln 3;
    # derivative g(beta) = sigma(beta) - 0.75 (= sum p z - sum T z), so g(0) = -0.25 exactly.
    logits, targets = [[0.0, 1.0]], [[0.25, 0.75]]
    assert B.soft_beta_derivative(logits, targets, 0.0) == -0.25
    fit = B.fit_beta_soft(logits, targets)
    assert abs(fit["beta"] - math.log(3)) < 1e-6 and abs(fit["derivative"]) < 1e-6
    assert B.soft_beta_derivative(logits, targets, math.log(3) - 0.1) < 0 < B.soft_beta_derivative(logits, targets, math.log(3) + 0.1)


def test_soft_beta_accepts_the_zero_endpoint_for_a_uniform_target():
    fit = B.fit_beta_soft([[0.0, 1.0]], [[0.5, 0.5]])
    assert fit["beta"] == 0.0 and fit["bracket"] == [0.0, 0.0]


def test_soft_beta_refuses_a_target_it_can_never_reach():
    with pytest.raises(ValueError):
        B.fit_beta_soft([[0.0, 1.0]], [[0.0, 1.0]])     # CE still decreasing at beta = 100: not a minimum
```

- [ ] **Step 1:** Append the block above and add the two imports at the top.
- [ ] **Step 2: Red run.** Run the test command once and capture the exit status.

  Expected: exit status **1**.
  - The 39 checkpoint-1 tests pass.
  - The 11 new tests fail or error:
    - `AttributeError` on `label_mcts`, `kl_logits`, `soft_beta_derivative` or `fit_beta_soft`;
    - `test_game_measures_survives_underflowing_temperature_logits` fails with `ValueError: math domain error`. That is the bug itself.

  Any other outcome stops for review.

## Task 2: Implement (module only)

**Files:** Modify `scripts/GPU/alphazero/search_distillation_stage_b.py`.

- [ ] **Step 1: Imports and header.** Add these after the existing imports:

```python
from .eval_runner import EvalConfig, cfg_from
from .external_learning import BETA_BOUNDS, BETA_ITERATIONS, BETA_TOLERANCE
from .mcts import MCTS
```

In the module docstring, replace "no model, no search, no file I/O" with "no model and no file I/O; `label` runs the real search with whatever evaluator it is given".

- [ ] **Step 2: The label adapter.** Append:

```python
def label_mcts(evaluator: Any, sims: int, seed: int) -> MCTS:
    """Card §5: the evaluation harness's red-search construction on the frozen protocol, at `sims`."""
    return MCTS(evaluator, cfg_from(EvalConfig(**PROTOCOL), sims), random.Random(seed ^ 0xA5A5A5))


def label(evaluator: Any, state: TwixtState, sims: int, seed: int) -> List[List[int]]:
    """One label: a fresh search_with_root without noise; the sparse visits, refused unless they sum to sims."""
    visits, _value, _root = label_mcts(evaluator, sims, seed).search_with_root(state, add_noise=False)
    return sparse_label(visits, sims)
```

- [ ] **Step 3: Stable KL.** Append the following, then replace the KL line in `game_measures`.

```python
def log_softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits)
    lse = m + math.log(sum(math.exp(z - m) for z in logits))
    return [z - lse for z in logits]


def kl_logits(t: Sequence[float], logits: Sequence[float]) -> float:
    """KL(t || softmax(logits)) through a stable log-softmax: finite where softmax underflows to 0."""
    if len(t) != len(logits):
        raise ValueError("distributions over different supports")
    return sum(a * (math.log(a) - l) for a, l in zip(t, log_softmax(logits)) if a > 0)
```

In `game_measures`, change

```python
    return {"kl": {a: sum(kl(probs(p["target"]), softmax(p["logits"][a])) for p in positions) / n for a in ARMS},
```

to

```python
    return {"kl": {a: sum(kl_logits(probs(p["target"]), p["logits"][a]) for p in positions) / n for a in ARMS},
```

- [ ] **Step 4: The soft-target β fit.** Append the following. It is #53's `fit_beta` bracketing with a soft target. It uses pure Python and the same stable log-softmax as the evaluation, so the fit and the KL it is compared with share one probability function.

```python
def soft_beta_derivative(logits_rows: Sequence[Sequence[float]], target_rows: Sequence[Sequence[float]],
                         beta: float) -> float:
    """d/dbeta of the mean soft CE: mean over rows of sum p(beta) z - sum T z, over each row's legal moves."""
    total = 0.0
    for z, t in zip(logits_rows, target_rows):
        p = softmax([beta * x for x in z])
        total += sum(a * x for a, x in zip(p, z)) - sum(a * x for a, x in zip(t, z))
    v = total / len(logits_rows)
    if not math.isfinite(v):
        raise ValueError(f"nonfinite derivative at beta = {beta}")
    return v


def fit_beta_soft(logits_rows: Sequence[Sequence[float]], target_rows: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Card §7: one inverse temperature on the TRAINING positions' T1600 targets, minimizing the mean soft CE.
    Convex in beta: #53's bracketing -- bounds [0, 100], start 1, bisection to 1e-6 in at most 64 steps; an
    endpoint only when its one-sided derivative establishes the constrained minimum. Refusals are ValueError."""
    if not logits_rows or len(logits_rows) != len(target_rows) or any(len(z) != len(t) for z, t in zip(logits_rows, target_rows)):
        raise ValueError("logits and targets are not row-aligned over the same supports")

    def g(b: float) -> float:
        return soft_beta_derivative(logits_rows, target_rows, b)

    lo_b, hi_b = BETA_BOUNDS
    iterations = 0
    if g(lo_b) >= 0:
        beta, bracket = lo_b, (lo_b, lo_b)
    else:
        if g(hi_b) <= 0:
            raise ValueError(f"soft CE still decreasing at beta = {hi_b}: the fit is not bracketed")
        lo, hi = (lo_b, 1.0) if g(1.0) >= 0 else (1.0, hi_b)
        while hi - lo > BETA_TOLERANCE and iterations < BETA_ITERATIONS:
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
            iterations += 1
        if hi - lo > BETA_TOLERANCE:
            raise ValueError("bisection did not reach the bracket tolerance")
        beta, bracket = (lo + hi) / 2, (lo, hi)
    ce = sum(-sum(a * l for a, l in zip(t, log_softmax([beta * x for x in z])) if a > 0)
             for z, t in zip(logits_rows, target_rows)) / len(logits_rows)
    if not math.isfinite(ce):
        raise ValueError("nonfinite soft CE at the fitted beta")
    return {"beta": beta, "ce": ce, "derivative": g(beta), "iterations": iterations, "bracket": list(bracket),
            "bounds": list(BETA_BOUNDS)}
```

- [ ] **Step 5: Green run.** Run the test command once and capture the exit status.

  Expected: exit status **0**, **50 passed** (39 + 11), plus one `DESCRIPTIVE another-seed 400` line whose value is reported as observed, whichever it is. An unexpected failure or a timeout stops for review, with no extra runs.

- [ ] **Step 6: Scan.** Run the pinned scan of the two files; each must print `{}`. Use the same snippet as plan `a7a838fa`, Task 5, Step 5.

- [ ] **Step 7: Report and STOP.** Report both exit statuses, the counts and the duration; prefix equality, visit sums, same-seed repeatability, adapter reconstruction, and the descriptive another-seed line. Leave the module and tests **uncommitted** unless separately approved.

## After 2A (not planned here)

Phase scripts, evaluator and checkpoint loading, seed accounting, corpus generation, labelling and training each still need their own plan and approval.
