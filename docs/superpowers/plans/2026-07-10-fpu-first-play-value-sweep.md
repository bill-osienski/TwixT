# FPU First-Play-Value Knob + Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add one opt-in `MCTSConfig.fpu_value` field (byte-identical at its `0.0` default) and a read-only diagnostic that sweeps it over the 30 A probe roots, to test whether making an unvisited child's assumed value pessimistic for the mover pulls the 400-sim A metric toward the 6400-sim reference.

**Architecture:** Task 1 is the core-search change — one dataclass field plus one line in `_select_child`, with unit tests pinning both the default (byte-identical) and the negative-FPU selection semantics. Task 2 is the sweep diagnostic, structurally identical to the merged `diagnose_cpuct_sweep.py`. Task 3 (controller) proves byte-identity via the full suite, runs the integrity-checked sweep, and reports against the decision rule.

**Tech Stack:** Python 3.14 / MLX, pytest.

**Spec:** `docs/superpowers/specs/2026-07-10-fpu-first-play-value-sweep-design.md` (APPROVED).

## Global Constraints

- Python: always `.venv/bin/python`. Authoritative full-suite baseline on merged main = **1443 passed**. After this branch: 1443 + new unit tests, **0 failures** — the unchanged 1443 IS the byte-identical proof for the default path.
- **The ONLY behavioral edit is in `mcts.py`:** add `fpu_value: float = 0.0` to `MCTSConfig`, and change the unvisited-child `q = 0.0` in `_select_child` to `q = self.config.fpu_value`. Nothing else in `mcts.py` changes. **`fpu_value=0.0` must reproduce current behavior exactly** — same float, one read site.
- Do NOT modify `trainer.py`, `network.py`, `self_play.py`, `eval_runner.py`, `calibration_pool.py`, or any manifest/builder. No value rows, no prior pruning, no top-k, no promotion change, no self-play adoption. `self_play.py:881`'s `c_puct=cfg.c_puct` is a telemetry helper — leave it.
- **Gate-faithful search / knob isolation:** `dataclasses.replace(cfg_from(EvalConfig(mcts_sims=400, mcts_eval_batch_size=14, mcts_stall_flush_sims=48)), fpu_value=x)`. Never hand-build `MCTSConfig`. One evaluator built once, reused across all values, via an explicit `_make_search_fn(evaluator, cfg)` factory (NOT a default-argument closure — a factory makes late binding structurally impossible).
- **Mandatory integrity check** at `fpu_value == 0.0`: each case's `root_mcts_black_value` matches `logs/eval/v15prep_a_continuation_concentration.csv` within `1e-6`, else `SystemExit`. `--fpu-values` must include `0.0` or abort.
- **Runtime budget guard:** `root.visit_count != 400` → `SystemExit` (catches a wrong `n_simulations`; the root is pre-expanded outside the sim loop so this holds exactly).
- Gate thresholds `>= 0.25` / `>= 0.50`, imported from `position_probe_cases` (`OVERVALUE_THRESHOLD`, `SEVERE_OVERVALUE_THRESHOLD`). Never `> 0`; `positive_pct_gt_0` is its own column.
- Perspective: `root_mcts_black_value = to_black(root_value_stm, side)`; `top_child_q_black = to_black(child.q_value, child.state.to_move)` — the CHILD's to-move.
- `_best_child` imported from `continuation_extraction` (side-effect-free import). `search_for_row` imported from `diagnose_v15_a_continuation_concentration`.
- NEVER `sys.modules.pop("mlx")` in tests. Diagnostic tests are pure (synthetic `MCTSNode`); the mcts.py unit tests use a real `MCTS` with a deterministic stub value function (mirror `tests/test_mcts_forced_root_visit_equivalence.py`).
- Worktree `feature/fpu-first-play-value`; symlink `.venv`; FF-merge (no `--no-ff`, never force-push); code-commit trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; file-scoped `git add`; **locate code by content, not line numbers**. Fresh worktree lacks gitignored data + checkpoints → whole-repo suite there = exactly 14 failed + 6 errors; judge on file-scoped runs; authoritative suite on merged main.

## Interfaces (verified — use exactly)

- `MCTSConfig` is a `@dataclass` in `mcts.py`; `c_puct: float = 1.5` is its first hyperparameter field. `cfg_from` (`eval_runner.py`) sets six fields and never touches `c_puct`/`fpu_value`, so both take their dataclass defaults — `dataclasses.replace` is exact.
- `_select_child` (`mcts.py`, one definition, both callers route through it): the unvisited branch is
  ```python
  else:
      q = 0.0
      child_visits = 0
  ```
  and the score is `score = q + u` where `u = c * prior * sqrt_parent / (1 + child_visits)`.
- `search_with_root(state, add_noise=False) -> (visit_counts, root_value_stm, root)`; `MCTSNode.q_value` returns `0.0` when `visit_count == 0`.
- `from .continuation_extraction import _best_child`; `from .diagnose_v15_a_continuation_concentration import search_for_row`; `from .eval_raw_nn_position_rows import to_black`; `from .position_probe_cases import OVERVALUE_THRESHOLD, SEVERE_OVERVALUE_THRESHOLD, load_csv_manifest`; `from .eval_runner import EvalConfig, cfg_from, _default_evaluator_factory`; `from .mcts import MCTS, decode_move`.
- Phase-0 CSV: `root_case_id`, `root_mcts_black_value` (repeated per child row; take first per case). A manifest: 30 cases, columns `case_id, game_idx, replay_path, position_ply, side_to_move`.

## File Structure

| File | Role |
|---|---|
| `scripts/GPU/alphazero/mcts.py` (modify) | add `fpu_value` field + read it in `_select_child` |
| `tests/test_fpu_value.py` (create) | unit tests: default, byte-identical selection at 0.0, negative-FPU semantics |
| `scripts/GPU/alphazero/diagnose_fpu_sweep.py` (create) | read-only sweep diagnostic |
| `tests/test_fpu_sweep.py` (create) | pure tests for the diagnostic's aggregation/counters |

---

### Task 1: The `fpu_value` search knob

**Files:** Modify `scripts/GPU/alphazero/mcts.py`; create `tests/test_fpu_value.py`.

**Interfaces:**
- Produces: `MCTSConfig.fpu_value` (float, default 0.0), consumed by Task 2's `dataclasses.replace(base, fpu_value=x)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_fpu_value.py`. Fully controlled synthetic roots built by hand (no `_expand`, no real board): `_select_child` reads only `node.priors`, `node.children`, `node.visit_count` and each child's `visit_count`/`value_sum`, and `is_expanded` is just `priors is not None`. `_select_child(node, pending_node_ids=None)` — with no pending set, the virtual-visit branch never fires. The stub value fn is never called by `_select_child`.

```python
import random

from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig, MCTSNode, encode_move


def _stub_value_fn():
    def f(state):
        return {}, 0.0
    return f


def _synthetic_root(fpu):
    """Root with two candidate moves, each prior 0.01:
      A = a decent VISITED reply for the mover -- child q in the child's own
          perspective = -0.1, so the mover (parent) sees -(-0.1) = +0.1 --
          visited 100 times;
      B = UNVISITED (no child node).
    Arithmetic (c_puct=1.5, sqrt_parent = sqrt(101) = 10.0499):
      score_A = 0.1 + 1.5*0.01*10.0499/(1+100) = 0.1 + 0.00149 = 0.10149
      score_B = fpu + 1.5*0.01*10.0499/(1+0)   = fpu + 0.15075
    So at fpu=0.0, B (0.15075) outranks A (0.10149) -- the legacy pathology,
    an unexplored move beating a decent visited reply. At fpu=-0.5, B is
    -0.34925 and A wins. The two scores are far apart => no rng tie-break."""
    cfg = MCTSConfig(n_simulations=1, c_puct=1.5, fpu_value=fpu)
    m = MCTS(_stub_value_fn(), cfg, random.Random(0))
    A, B = encode_move(0, 0), encode_move(1, 1)
    root = MCTSNode(state=None, visit_count=100)
    root.priors = {A: 0.01, B: 0.01}
    root.children[A] = MCTSNode(state=None, parent=root, move=A,
                                visit_count=100, value_sum=-10.0)  # q_value=-0.1
    return m, root, A, B


def test_fpu_value_default_is_zero():
    assert MCTSConfig().fpu_value == 0.0


def test_fpu_zero_reproduces_legacy_unvisited_wins():
    # fpu=0.0 IS the old hardcoded q=0.0: the unvisited move B wins.
    m, root, A, B = _synthetic_root(fpu=0.0)
    assert m._select_child(root)[0] == B


def test_negative_fpu_makes_the_mover_keep_the_good_visited_child():
    # same root, fpu=-0.5 lowers B's assumed value below A's real value.
    # (If _select_child ignored fpu_value, B would still win -- this discriminates.)
    m, root, A, B = _synthetic_root(fpu=-0.5)
    assert m._select_child(root)[0] == A
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_fpu_value.py -v` → FAIL (`fpu_value` not a field / `TypeError` on the `fpu_value=` kwarg).

- [ ] **Step 3: Add the field.** In `mcts.py`, in the `MCTSConfig` dataclass, immediately after the `c_puct: float = 1.5` line, add:

```python
    fpu_value: float = 0.0   # First-Play Urgency: assumed Q for an unvisited
                             # child, in the MOVER's perspective. 0.0 reproduces
                             # the prior hardcoded value exactly. Negative =>
                             # pessimistic => the mover revisits known-good
                             # children before scanning unexplored ones.
```

- [ ] **Step 4: Read the field in `_select_child`.** Change the unvisited branch (locate by content — the `else:` with `q = 0.0` and `child_visits = 0` inside `_select_child`) from:

```python
            else:
                q = 0.0
                child_visits = 0
```

to:

```python
            else:
                q = self.config.fpu_value
                child_visits = 0
```

Change nothing else in `_select_child` — not the `q = -child.q_value` visited branch, not the pending/virtual-visit block, not the `u`/`score` computation.

- [ ] **Step 5: Run the unit tests** → `.venv/bin/python -m pytest tests/test_fpu_value.py -v` → PASS.

- [ ] **Step 6: Byte-identical spot check** — run the existing MCTS test suites, which exercise the default path end to end:

Run: `.venv/bin/python -m pytest tests/test_mcts.py tests/test_mcts_forced_root_visit_equivalence.py -q`
Expected: PASS, same counts as before the edit. (The authoritative full-suite proof is Task 3.)

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/mcts.py tests/test_fpu_value.py
git commit -m "feat(mcts): opt-in fpu_value first-play-urgency knob (byte-identical at 0.0)

MCTSConfig.fpu_value (default 0.0) replaces the hardcoded q=0.0 for unvisited
children in _select_child. 0.0 reproduces prior behavior exactly (one read
site, same float, both selection callers route through _select_child). Negative
values make an unvisited child pessimistic for the mover, so the mover revisits
known-good children instead of scanning unexplored replies -- the lever the
c_puct falsification test isolated. Diagnostic-only; nothing sets it yet.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: The FPU sweep diagnostic

**Files:** Create `scripts/GPU/alphazero/diagnose_fpu_sweep.py`, `tests/test_fpu_sweep.py`.

This is structurally identical to the merged `diagnose_cpuct_sweep.py`; the only differences are the swept field (`fpu_value` not `c_puct`), the default values, the output paths, and the column/summary names. Read `scripts/GPU/alphazero/diagnose_cpuct_sweep.py` first and mirror it — including `_make_search_fn`, the integrity check, the budget guard, the `gate_flags`/`n_visited_children`/`summarize` pure helpers, and the `--<field>-values must include the baseline` guard.

**Interfaces:**
- Produces: `gate_flags(value)`, `n_visited_children(node)`, `summarize(rows)`, `main(argv)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_fpu_sweep.py` (mirror `tests/test_cpuct_sweep.py`, which is the reviewed template; `summarize` here also carries `top_child_visit_share_mean`):

```python
from scripts.GPU.alphazero.mcts import MCTSNode, encode_move
from scripts.GPU.alphazero.diagnose_fpu_sweep import (
    gate_flags, n_visited_children, summarize)


def _child(parent, rc, visits):
    n = MCTSNode(state=None, parent=parent, move=encode_move(*rc),
                 visit_count=visits, value_sum=0.0)
    parent.children[n.move] = n
    return n


def test_gate_flags_use_the_gate_thresholds_not_zero():
    assert gate_flags(0.10) == (False, False)
    assert gate_flags(0.25) == (True, False)
    assert gate_flags(0.50) == (True, True)
    assert gate_flags(-0.30) == (False, False)


def test_n_visited_children_counts_only_visited():
    root = MCTSNode(state=None, visit_count=10)
    _child(root, (1, 1), 7)
    _child(root, (2, 2), 3)
    _child(root, (3, 3), 0)
    assert n_visited_children(root) == 2


def test_summarize_uses_gate_thresholds_and_reports_tree_shape():
    rows = [
        {"root_mcts_black_value": 0.60, "root_n_visited_children": 4,
         "top_child_n_visited_children": 300, "top_child_visit_share": 0.8},
        {"root_mcts_black_value": 0.30, "root_n_visited_children": 6,
         "top_child_n_visited_children": 200, "top_child_visit_share": 0.6},
        {"root_mcts_black_value": 0.10, "root_n_visited_children": 8,
         "top_child_n_visited_children": 100, "top_child_visit_share": 0.4},
        {"root_mcts_black_value": -0.40, "root_n_visited_children": 2,
         "top_child_n_visited_children": 400, "top_child_visit_share": 0.2},
    ]
    s = summarize(rows)
    assert s["n"] == 4
    assert abs(s["mean_black_value"] - 0.15) < 1e-9
    assert abs(s["over_pct_ge_0_25"] - 50.0) < 1e-9
    assert abs(s["severe_pct_ge_0_50"] - 25.0) < 1e-9
    assert abs(s["positive_pct_gt_0"] - 75.0) < 1e-9
    assert abs(s["root_children_mean"] - 5.0) < 1e-9
    assert abs(s["top_child_children_mean"] - 250.0) < 1e-9
    assert abs(s["top_child_visit_share_mean"] - 0.5) < 1e-9
    assert abs(s["min"] - (-0.40)) < 1e-9
    assert abs(s["max"] - 0.60) < 1e-9


def test_summarize_boundary_values_are_inclusive():
    rows = [{"root_mcts_black_value": 0.25, "root_n_visited_children": 1,
             "top_child_n_visited_children": 1, "top_child_visit_share": 0.5},
            {"root_mcts_black_value": 0.50, "root_n_visited_children": 1,
             "top_child_n_visited_children": 1, "top_child_visit_share": 0.5}]
    s = summarize(rows)
    assert s["over_pct_ge_0_25"] == 100.0
    assert s["severe_pct_ge_0_50"] == 50.0
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_fpu_sweep.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `diagnose_fpu_sweep.py`** by mirroring `diagnose_cpuct_sweep.py`. Header docstring must state the claim under test (first-touch optimism) and that this is diagnostic-only, opt-in, byte-identical off. Concrete differences from the c_puct script:
  - `DEFAULT_FPUS = "0.0,-0.05,-0.10,-0.20,-0.35,-0.50"`; `BASELINE_FPU = 0.0`.
  - `DEFAULT_OUT = "logs/eval/fpu_check/a_predrop_fpu_sweep_cases.csv"`, `DEFAULT_SUMMARY_OUT = "logs/eval/fpu_check/a_predrop_fpu_sweep_summary.csv"`.
  - `cfg = dataclasses.replace(base, fpu_value=x)`.
  - Case CSV column 1 is `fpu_value`; the integrity check and abort-guard key on `BASELINE_FPU` in the parsed `--fpu-values`.
  - `summarize` returns, in addition to the c_puct set: `top_child_visit_share_mean` (the c_puct script omitted it; here it is required by the decision rule). `SUMMARY_FIELDNAMES` = `["fpu_value", "n", "mean_black_value", "over_pct_ge_0_25", "severe_pct_ge_0_50", "positive_pct_gt_0", "root_children_mean", "top_child_children_mean", "top_child_visit_share_mean", "min", "max"]`. Accumulate `top_child_visit_share` per case row (blank-guard if `top` is None: skip Nones in the mean, or treat an empty sweep row's share as 0.0 — but every A root has visited children, so `top` is never None; still, guard defensively).
  - `FIELDNAMES` = `["fpu_value", "case_id", "root_mcts_black_value", "gate_over_ge_0_25", "gate_severe_ge_0_50", "root_n_visited_children", "top_child_move", "top_child_visit_share", "top_child_q_black", "top_child_n_visited_children"]`.

- [ ] **Step 4: Run the pure tests** → `.venv/bin/python -m pytest tests/test_fpu_sweep.py -v` → PASS.

- [ ] **Step 5: Verify import + `--help`** — `.venv/bin/python -c "from scripts.GPU.alphazero.diagnose_fpu_sweep import main; print('ok')"` → `ok`; `--help` renders, exit 0.

- [ ] **Step 6: Prove the abort-guard fires without a checkpoint** — `.venv/bin/python -c "from scripts.GPU.alphazero.diagnose_fpu_sweep import main; main(['--fpu-values','-0.1,-0.2'])"` must exit with the "must include the baseline 0.0" `SystemExit` before loading a checkpoint. Report the message.

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/diagnose_fpu_sweep.py tests/test_fpu_sweep.py
git commit -m "feat(diagnostic): FPU first-play-value sweep (read-only)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Suite, byte-identical proof, merge, run (controller-run)

- [ ] **Step 1:** Worktree full suite → exactly the known 14 failed + 6 errors, plus the new unit + pure tests passing.
- [ ] **Step 2: FF-merge to main; authoritative suite.** Expected: **1443 + (new tests) passed, 0 failed.** The unchanged 1443 pre-existing tests are the byte-identical proof for the default path — if any MCTS/self-play/eval test changed behavior, it fails here. Do not push until green.
- [ ] **Step 3: Run the sweep** — `.venv/bin/python -u -m scripts.GPU.alphazero.diagnose_fpu_sweep` (6 values × 30 roots × 400 sims = 180 searches; ~30–45 min; background). The `fpu_value=0.0` integrity check runs first — if it fails, **stop and report; do not interpret the sweep** (a failure would mean the field edit was not byte-identical). Then push.
- [ ] **Step 4: Report against the decision rule (spec §4)** — the summary table, and explicitly whether `mean`, `over`, `severe`, and `top_child_n_visited_children` all move materially toward the 6400 reference (mean −0.045 / over 10.0% / severe 3.3%) without `top_child_visit_share` collapsing to a degenerate root choice. STOP — do not proceed to the unbiased-sample / B-C-D / strength-eval validation, and do not touch self-play; those are the operator's call.
