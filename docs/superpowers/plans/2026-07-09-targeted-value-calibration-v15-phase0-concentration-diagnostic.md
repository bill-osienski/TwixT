# Targeted Value Calibration v15 — Phase 0: A-Continuation Concentration Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A read-only diagnostic that, for each A "black pre-drop" root, runs BASE 400-sim (gate-faithful) MCTS and measures how concentrated the value backup is — so we can decide whether v15's A-continuation correction can be a few-row branch (Phase 1) or must be tree/path-level.

**Architecture:** One new read-only script `diagnose_v15_a_continuation_concentration.py`. It reconstructs each A root, runs the same MCTS the gates use, and for each root child computes `visit_share`, `child_contribution_share` (these sum to the root Q), depth, and the child's raw value under BASE and under v14b. It writes one CSV row per (root, child) plus a per-root classification (concentrated / semi / broad by the top-1–3 contribution share). **No manifest, no child replay JSONs, no training, no loader/trainer/network change.**

**Tech Stack:** Python 3.14 / MLX, pytest.

**Spec:** `docs/superpowers/specs/2026-07-09-targeted-value-calibration-v15-a-searched-continuation-correction-design.md` (APPROVED — Phase 0 = concentration CSV only).

## Global Constraints

- Python: always `.venv/bin/python`; tests `.venv/bin/python -m pytest <file> -v`; full-suite baseline on merged main = **1411 passed**.
- **READ-ONLY / diagnostic:** the script only READS checkpoints + the A probe manifest and WRITES the one diagnostic CSV. It must NOT modify any shared module. Do NOT change `mcts.py`, `continuation_extraction.py`, `calibration_pool.py`, `eval_runner.py`, `probe_eval.py`, `trainer.py`, `network.py`, or any manifest/builder.
- NEVER `sys.modules.pop("mlx")` in tests.
- **Gate-faithful MCTS** — must match the A/B/C/D gate exactly: `EvalConfig(mcts_sims=400, mcts_eval_batch_size=14, mcts_stall_flush_sims=48)`, `_default_evaluator_factory` (train-mode BN, `compile=True`), `MCTS(...).search_with_root(state, add_noise=False)`, seeded via `row_seed(...)`. (Same setup `build_searched_continuation_retention_manifest._real_search_fn` uses.)
- **Raw child values** — computed IN-PROCESS on the live `TwixtState` via `_teacher_infer(state, evaluator)` (eval-mode BN evaluators for BASE and v14b). Do NOT route child boards through `eval_raw_nn_position_rows` (it ignores `extra_moves_json` and would score the parent).
- Worktree `feature/tvc-v15-phase0-diagnostic`; symlink `.venv`; FF-merge (no `--no-ff`); code-commit trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; file-scoped `git add`; **locate code by content, not line numbers**. Fresh worktree → whole-repo suite there = 14 failed + 6 errors; judge tasks file-scoped; authoritative suite on merged main.

## Interfaces (verified — use exactly)

- `from scripts.GPU.alphazero.probe_eval import load_network_for_scoring` → `net, *_ = load_network_for_scoring(path)`; `net.eval()`; `from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator` → `LocalGPUEvaluator(net)` for the raw (eval-mode) evaluators (BASE + v14b).
- Gate-faithful search: `from scripts.GPU.alphazero.eval_runner import EvalConfig, cfg_from, _default_evaluator_factory`; `from scripts.GPU.alphazero.mcts import MCTS, decode_move`; `evaluator = _default_evaluator_factory(base_ckpt)`; `cfg = cfg_from(EvalConfig(mcts_sims=400, mcts_eval_batch_size=14, mcts_stall_flush_sims=48))`; `counts, root_value_stm, root = MCTS(evaluator, cfg, random.Random(seed)).search_with_root(state, add_noise=False)`.
- Row reconstruction/seed: **copy the exact import lines for `position_state` and `row_seed` from `build_searched_continuation_retention_manifest.py`** (grep its top-of-file imports — do not guess the module). Usage: `state = position_state(replay, position_ply, side)`; `seed = row_seed(tag, game_idx, ply, pos_base_seed=20260616, goal_base_seed=20260614)` — same seed args the v6 builder passes (see its `build_rows_v6`). Read `game_idx`/`position_ply`/`side_to_move`/`tag`/`replay_path` from each A manifest row (columns confirmed present in the A probe manifest).
- Raw forward: `from scripts.GPU.alphazero.build_teacher_calibration_manifest import _teacher_infer` → `legal, priors, value_stm = _teacher_infer(state, evaluator)`; convert to black perspective with the existing helper (`to_black` in `eval_raw_nn_position_rows` / `target_in_to_move` in `calibration_pool` — import one, do not re-implement).
- **MCTS node fields** (`MCTSNode`): `root.children` is `Dict[move_id, MCTSNode]`; each child has `.visit_count`, `.value_sum`, `.q_value` (== `value_sum/visit_count`), `.state` (live `TwixtState`), `.move` (encoded; `decode_move(move_id) -> (r,c)`). `root.visit_count == sum(child.visit_count)`. Continuation helpers to COPY (do not import — `extract_continuations` raises for the A tag): `path_moves_of(node)` from `continuation_extraction` gives the root-relative move path (for depth/PV).
- **The metric math (derived + must be unit-tested):** for each root child,
  - `visit_share = child.visit_count / root.visit_count`
  - `child_contribution_share = visit_share * (-child.q_value)`  (root perspective)
  - `sum(child_contribution_share for all children) == root.q_value`  (invariant to assert in tests)
- A probe manifest (roots): `logs/eval/loss_analysis_v2_calib020_0001_vs_0379_black/0001_black_post_opening_top30_predrop_probe_manifest.csv`. BASE = `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`; v14b = `checkpoints/alphazero-v14b-value-adapter-projection-from-calib020-0001/model_iter_0001.safetensors`.

## File Structure

| File | Role |
|---|---|
| `scripts/GPU/alphazero/diagnose_v15_a_continuation_concentration.py` (create) | read-only diagnostic: per-child metrics + raw values + per-root classification → concentration CSV |
| `tests/test_v15_concentration_diagnostic.py` (create) | unit-test the pure metric + classification logic on synthetic `MCTSNode` trees (no checkpoint/MCTS) |

---

### Task 1: The concentration diagnostic (pure logic + script)

**Files:** Create `scripts/GPU/alphazero/diagnose_v15_a_continuation_concentration.py`, `tests/test_v15_concentration_diagnostic.py`.

**Interfaces:**
- Produces: `per_child_metrics(root) -> list[dict]` (one dict per root child: `move`, `visit_share`, `child_contribution_share`, `q_value`, `visit_count`); `classify_concentration(metrics, top_n=3) -> tuple[str, float]` returning `("concentrated"|"semi"|"broad", top_n_positive_contribution_share)`; `main(argv) -> int`.

- [ ] **Step 1: Write the failing tests** — `tests/test_v15_concentration_diagnostic.py`, using real `MCTSNode` (mirror `tests/test_continuation_extraction.py:12-18`), NOT mocks:

```python
import random
from scripts.GPU.alphazero.mcts import MCTSNode, encode_move
from scripts.GPU.alphazero.diagnose_v15_a_continuation_concentration import (
    per_child_metrics, classify_concentration)


def _root_with_children(specs):
    # specs: list of (move_rc, visit_count, q_value). Builds a root whose
    # invariants (visit_count sum, value_sum = q*visits) match the real backup.
    root = MCTSNode(state=None)
    for (rc, vc, q) in specs:
        ch = MCTSNode(state=None, parent=root, move=encode_move(*rc),
                      visit_count=vc, value_sum=q * vc)
        root.children[ch.move] = ch
    root.visit_count = sum(vc for _, vc, _ in specs)
    root.value_sum = sum(-q * vc for _, _, q in specs)   # single sign flip child->root
    return root


def test_contributions_sum_to_root_q():
    root = _root_with_children([((1, 1), 300, -0.9), ((2, 2), 60, 0.2), ((3, 3), 40, 0.1)])
    m = per_child_metrics(root)
    assert abs(sum(c["child_contribution_share"] for c in m) - root.q_value) < 1e-9
    assert abs(sum(c["visit_share"] for c in m) - 1.0) < 1e-9


def test_visit_share_and_contribution_values():
    # child (1,1): 300/400 visits, q=-0.9 -> contribution = 0.75 * 0.9 = +0.675 (root perspective)
    root = _root_with_children([((1, 1), 300, -0.9), ((2, 2), 100, 0.0)])
    m = {tuple(c["move"]): c for c in per_child_metrics(root)}
    assert abs(m[(1, 1)]["visit_share"] - 0.75) < 1e-9
    assert abs(m[(1, 1)]["child_contribution_share"] - 0.675) < 1e-9


def test_classify_concentrated():
    # top child carries ~96% of the positive backup mass
    root = _root_with_children([((1, 1), 380, -0.9), ((2, 2), 20, -0.1)])
    label, share = classify_concentration(per_child_metrics(root))
    assert label == "concentrated" and share >= 0.70


def test_classify_broad():
    # positive backup spread across many similar children
    specs = [((r, r), 40, -0.2) for r in range(1, 11)]     # 10 children, equal
    label, share = classify_concentration(per_child_metrics(_root_with_children(specs)))
    assert label == "broad" and share < 0.40
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_v15_concentration_diagnostic.py -v` → FAIL (`ModuleNotFoundError`/`ImportError`).

- [ ] **Step 3: Implement the pure logic** in `diagnose_v15_a_continuation_concentration.py`:

```python
def per_child_metrics(root) -> list[dict]:
    """Per root-child visit_share + contribution (root perspective). Contributions
    sum to root.q_value; visit_shares sum to 1. Children with 0 visits are dropped."""
    from scripts.GPU.alphazero.mcts import decode_move
    total = root.visit_count or sum(c.visit_count for c in root.children.values())
    out = []
    for move_id, ch in root.children.items():
        if ch.visit_count <= 0:
            continue
        vs = ch.visit_count / total
        out.append({
            "move": list(decode_move(move_id)),
            "visit_count": ch.visit_count,
            "visit_share": vs,
            "q_value": ch.q_value,
            "child_contribution_share": vs * (-ch.q_value),   # root perspective
        })
    return out


def classify_concentration(metrics: list[dict], top_n: int = 3) -> tuple[str, float]:
    """Share of the POSITIVE backup mass explained by the top_n highest-contribution
    children. >=0.70 concentrated / 0.40-0.70 semi / <0.40 broad (spec §0)."""
    pos = [m["child_contribution_share"] for m in metrics if m["child_contribution_share"] > 0]
    total_pos = sum(pos)
    if total_pos <= 0:
        return "broad", 0.0
    top = sum(sorted(pos, reverse=True)[:top_n])
    share = top / total_pos
    label = "concentrated" if share >= 0.70 else ("semi" if share >= 0.40 else "broad")
    return label, share
```

- [ ] **Step 4: Run the logic tests** → `.venv/bin/python -m pytest tests/test_v15_concentration_diagnostic.py -v` → all PASS.

- [ ] **Step 5: Implement `main(argv)`** — the read-only driver (reconstruct roots, run gate-faithful MCTS, compute per-child metrics + raw values under BASE & v14b, write the CSV, print per-root + aggregate classification). Use the Interfaces block above verbatim. Per (root, child) CSV columns: `root_case_id, child_move, depth, visit_count, visit_share, child_contribution_share, child_q_value, child_raw_black_BASE, child_raw_black_v14b, root_mcts_black_value, root_case_classification, root_top3_positive_share`. Compute `child_raw_black_*` via `_teacher_infer(child.state, evaluator)` on the two eval-mode evaluators; `depth` via `len(path_moves_of(child))`. `--out` default `logs/eval/v15prep_a_continuation_concentration.csv`. Guard: assert `abs(sum(child_contribution_share) - root_mcts_black_value) < 1e-6` per root (the backup invariant) and fail loud if violated.

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/diagnose_v15_a_continuation_concentration.py tests/test_v15_concentration_diagnostic.py
git commit -m "feat(diagnostic): v15 Phase-0 A-continuation concentration diagnostic (read-only)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Run the diagnostic + classify (controller-run)

- [ ] **Step 1: Full suite (worktree)** — `.venv/bin/python -m pytest tests/ -q` → baseline + the new logic tests, with EXACTLY the known 14 failed + 6 errors. Then FF-merge to main, authoritative suite (1411 + new, 0 failures), push.
- [ ] **Step 2: Run the diagnostic (gate-faithful MCTS over the 30 A roots)** — `.venv/bin/python -m scripts.GPU.alphazero.diagnose_v15_a_continuation_concentration --out logs/eval/v15prep_a_continuation_concentration.csv`. (~30 × 400-sim searches; if too heavy for this environment, hand the exact command to the operator.) Then classify: aggregate the per-root labels (how many concentrated / semi / broad) → the spec §0 decision. **STOP — the concentration read is the USER's; do not design or build Phase 1 until reviewed.**

---

## After Phase 0 (USER's decision, per spec §0/§5)

- **Concentrated (top 1–3 children ≥ 70%)** → Phase 1 = few-row v15 manifest (high-contribution children, target −0.35, Option 1 child replay JSONs).
- **Semi (40–70%)** → Phase 1 = top children + PV depth-1–2, tiered −0.35/−0.20.
- **Broad (< 40%)** → do NOT build few-row v15; write a separate tree/path-level design.
