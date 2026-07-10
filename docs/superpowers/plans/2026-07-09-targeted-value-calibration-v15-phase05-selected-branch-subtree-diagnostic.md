# Targeted Value Calibration v15 — Phase 0.5: Selected-Branch Subtree Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A read-only diagnostic that walks the full expanded subtree beneath each selected positive branch of the 17 overvaluing A roots, recording raw and searched values per node with PV annotation, so the operator can decide whether v15 Phase 1 should be a PV/path correction, a frontier/tree-level correction, or should not be built at all.

**Architecture:** One behavior-preserving extract-function patch to the merged Phase-0 script (exposing `search_for_row` and `raw_black_value`), plus one new sibling script that imports those helpers, re-runs the deterministic BASE search on the 17 roots, walks every descendant with `visit_count >= 1` under each selected branch, and writes a per-node CSV plus a by-depth summary CSV (emitted twice: full subtree, and PV nodes only). No manifest, no replay JSONs, no training, no loader/trainer/network/MCTS change.

**Tech Stack:** Python 3.14 / MLX, pytest.

**Spec:** `docs/superpowers/specs/2026-07-09-targeted-value-calibration-v15-phase05-selected-branch-subtree-diagnostic-design.md` (APPROVED).

## Global Constraints

- Python: always `.venv/bin/python`; tests `.venv/bin/python -m pytest <file> -v`. Authoritative full-suite baseline on merged main = **1416 passed**.
- **READ-ONLY / diagnostic.** The scripts only READ checkpoints, the A probe manifest, and the Phase-0 CSV, and WRITE two diagnostic CSVs. Do NOT change `mcts.py`, `continuation_extraction.py`, `calibration_pool.py`, `eval_runner.py`, `probe_eval.py`, `trainer.py`, `network.py`, or any manifest/builder. No new loss mode, no loader change, no verifier change, no projection code.
- **The only edit to an existing file** is the extract-function patch in Task 1, to `scripts/GPU/alphazero/diagnose_v15_a_continuation_concentration.py`. It must be behavior-preserving: no CSV schema change, no flag change, no output change.
- **Gate-faithful MCTS** — must match the A/B/C/D gate exactly, and must reproduce Phase 0's trees: `EvalConfig(mcts_sims=400, mcts_eval_batch_size=14, mcts_stall_flush_sims=48)`, `_default_evaluator_factory` (train-mode BN, `compile=True`), `MCTS(...).search_with_root(state, add_noise=False)`, seeded via `row_seed(CORRECTION_TAG, game_idx, position_ply, pos_base_seed=20260616, goal_base_seed=20260614)`.
- **Raw values** are computed IN-PROCESS on the live `TwixtState` via eval-mode evaluators (`load_network_for_scoring` + `net.eval()`), for BASE and v14b. Never route boards through `eval_raw_nn_position_rows`.
- **Perspective.** Every A root is black-to-move; assert it. `q_value_node_perspective = node.q_value` as stored by MCTS. `q_value_root_perspective = to_black(node.q_value, node.state.to_move)` — **the analysis column**. Same conversion for raw values.
- **Three fail-loud integrity checks** (Task 2, Step 7): fresh `root_mcts_black_value` matches the Phase-0 CSV within `1e-6`; the contribution invariant `sum(child_contribution_share) == root.q_value` holds within `1e-6` on every root; every depth-1 node's `q_value_root_perspective` equals `-child_q_value` from the Phase-0 CSV within `1e-6`.
- **No depth cap.** Each of the 400 sims expands exactly one leaf, so the whole tree is bounded at ~401 expanded nodes per root.
- **`pct_visit_mass_raw_positive` is the decision metric**, not `pct_raw_positive`.
- NEVER `sys.modules.pop("mlx")` in tests. Tests are pure: synthetic `MCTSNode` trees, no checkpoints, no MCTS.
- Worktree `feature/tvc-v15-phase05-subtree-diagnostic`; symlink `.venv`; FF-merge (no `--no-ff`, never force-push); code-commit trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; file-scoped `git add`; **locate code by content, not line numbers**. A fresh worktree lacks gitignored game-log data and checkpoints → whole-repo suite there = exactly 14 failed + 6 errors; judge tasks on file-scoped runs; run the authoritative suite on merged main.

## Interfaces (verified — use exactly)

- Phase-0 module `scripts.GPU.alphazero.diagnose_v15_a_continuation_concentration` already exports: `per_child_metrics(root) -> list[dict]` (keys `move`, `visit_count`, `visit_share`, `q_value`, `child_contribution_share`, `positive_contribution_share`), `classify_concentration(metrics, top_n=3)`, `path_moves_of(node) -> tuple[(r,c), ...]`, `_real_search_fn(base_checkpoint, sims, eval_batch_size, stall_flush_sims) -> search_fn(state, seed)`, `_build_raw_evaluator(checkpoint_path)`, `FIELDNAMES`, `DEFAULT_BASE_CHECKPOINT`, `DEFAULT_V14B_CHECKPOINT`, `DEFAULT_A_MANIFEST`.
- Task 1 adds to it: `search_for_row(row, search_fn, *, pos_base_seed, goal_base_seed) -> (state, side, root_value_stm, root)` and `raw_black_value(state, evaluator) -> float`.
- `MCTSNode` is a dataclass: `state`, `parent`, `move` (encoded int), `visit_count`, `value_sum`, `children: Dict[int, MCTSNode]`, property `q_value == value_sum / visit_count`, property `is_expanded`.
- `from .mcts import decode_move, encode_move`; `from .eval_raw_nn_position_rows import to_black`; `from .build_mcts_root_retention_manifest import CORRECTION_TAG, row_seed`; `from .position_probe_cases import load_csv_manifest`.
- Phase-0 CSV columns (read with `csv.DictReader`, all values are strings): `root_case_id, child_move, depth, visit_count, visit_share, child_contribution_share, positive_contribution_share, child_q_value, child_raw_black_BASE, child_raw_black_v14b, root_mcts_black_value, root_case_classification, root_top3_positive_share`. `child_move` is formatted `"r:c"`.
- `load_csv_manifest(path)["cases"]` returns the A probe manifest rows with `game_idx` coerced to `int`; each row has `case_id`, `replay_path`, `position_ply`, `side_to_move`.
- Terminal nodes have no legal moves and were never expanded (`_run_single_simulation` backs up the terminal outcome without calling `_expand`). They must not be passed to `_teacher_infer`.

## File Structure

| File | Role |
|---|---|
| `scripts/GPU/alphazero/diagnose_v15_a_continuation_concentration.py` (modify) | Task 1: extract `search_for_row` + `raw_black_value` out of `main()`; behavior-preserving |
| `tests/test_v15_concentration_diagnostic.py` (modify) | Task 1: add two tests pinning the extracted helpers |
| `scripts/GPU/alphazero/diagnose_v15_a_selected_branch_subtrees.py` (create) | Task 2: Phase 0.5 — selection, subtree walk, PV annotation, per-node CSV, by-depth summary |
| `tests/test_v15_selected_branch_subtrees.py` (create) | Task 2: pure logic on synthetic `MCTSNode` trees |

---

### Task 1: Extract `search_for_row` and `raw_black_value` from the Phase-0 script

**Files:**
- Modify: `scripts/GPU/alphazero/diagnose_v15_a_continuation_concentration.py`
- Test: `tests/test_v15_concentration_diagnostic.py` (append two tests; do not alter the existing five)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `search_for_row(row, search_fn, *, pos_base_seed, goal_base_seed) -> (state, side, root_value_stm, root)` and `raw_black_value(state, evaluator) -> float`, both imported by Task 2.

This is a pure refactor. The Phase-0 CSV at `logs/eval/v15prep_a_continuation_concentration.csv` is the operator's evidence of record; this task must not change a single value in it.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_v15_concentration_diagnostic.py`. Keep the existing five tests and the existing imports; add these imports and tests:

```python
import json

from scripts.GPU.alphazero.diagnose_v15_a_continuation_concentration import (
    raw_black_value, search_for_row)
from tests.goal_line_probe_fixtures import legal_replay


def test_raw_black_value_converts_to_black_perspective():
    # _teacher_infer returns the value in the state's OWN to-move perspective.
    # raw_black_value must flip it when the state is red-to-move, and not
    # otherwise. A fake evaluator returns a fixed value for any board.
    class _FakeEvaluator:
        def build_input_tensor(self, state):
            import numpy as np
            return np.zeros((3, state.active_size, state.active_size), dtype=np.float32)

        def infer(self, board, rows, cols, mask, active_size):
            import numpy as np
            return np.zeros((1, rows.shape[1]), dtype=np.float32), np.array([0.7], dtype=np.float32)

    replay = legal_replay(4)
    from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
    black_state = position_state(replay, 1, "black")   # ply 1 -> black to move
    red_state = position_state(replay, 2, "red")       # ply 2 -> red to move
    assert abs(raw_black_value(black_state, _FakeEvaluator()) - 0.7) < 1e-9
    assert abs(raw_black_value(red_state, _FakeEvaluator()) + 0.7) < 1e-9


def test_search_for_row_reconstructs_state_and_seeds_deterministically(tmp_path):
    # search_for_row must reconstruct the root from replay_path/position_ply,
    # pass the row_seed-derived seed to search_fn, and return the search's
    # (state, side, root_value_stm, root) without touching them.
    replay = legal_replay(6)
    replay_path = tmp_path / "game_000007.json"
    replay_path.write_text(json.dumps(replay))
    # legal_replay starts with red to move, so an ODD position_ply is
    # black-to-move; position_state raises if this disagrees with side_to_move.
    row = {"case_id": "c", "game_idx": 7, "position_ply": 1,
           "side_to_move": "black", "replay_path": str(replay_path)}

    seen = {}
    sentinel_root = object()

    def fake_search_fn(state, seed):
        seen["state"], seen["seed"] = state, seed
        return {"counts": 1}, 0.25, sentinel_root

    state, side, root_value_stm, root = search_for_row(
        row, fake_search_fn, pos_base_seed=20260616, goal_base_seed=20260614)

    assert seen["seed"] == 20260616 ^ 7 ^ 1      # row_seed's position-probe branch
    assert side == "black" and state.to_move == "black"
    assert root_value_stm == 0.25 and root is sentinel_root
    assert state is seen["state"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_v15_concentration_diagnostic.py -v`
Expected: FAIL — `ImportError: cannot import name 'raw_black_value'`.

- [ ] **Step 3: Add the two helpers** to `diagnose_v15_a_continuation_concentration.py`. Place them immediately after `_build_raw_evaluator` and before `_parse_args`:

```python
def search_for_row(row, search_fn, *, pos_base_seed, goal_base_seed):
    """Reconstruct an A probe row's root state and run the seeded gate-faithful
    search. Returns (state, side, root_value_stm, root). The seed is row_seed's
    position-probe branch (CORRECTION_TAG is never the goal-line tag), which
    reproduces the A gate's per-case rng exactly."""
    replay = json.loads(Path(row["replay_path"]).read_text())
    ply = int(float(row["position_ply"]))
    side = row["side_to_move"]
    state = position_state(replay, ply, side)
    seed = row_seed(CORRECTION_TAG, row["game_idx"], ply,
                    pos_base_seed=pos_base_seed, goal_base_seed=goal_base_seed)
    _counts, root_value_stm, root = search_fn(state, seed)
    return state, side, root_value_stm, root


def raw_black_value(state, evaluator) -> float:
    """Raw (non-MCTS) value at `state`, converted to BLACK's perspective.
    _teacher_infer returns the value in the state's OWN to-move perspective, so
    the conversion must use state.to_move -- not the root's side, which differs
    at every odd depth."""
    _legal, _priors, value_stm = _teacher_infer(state, evaluator)
    return to_black(value_stm, state.to_move)
```

- [ ] **Step 4: Rewrite `main()`'s loop head to call them.** Replace exactly this block inside `main()`:

```python
    for i, row in enumerate(rows):
        case_id = row["case_id"]
        replay = json.loads(Path(row["replay_path"]).read_text())
        ply = int(float(row["position_ply"]))
        side = row["side_to_move"]
        state = position_state(replay, ply, side)
        seed = row_seed(CORRECTION_TAG, row["game_idx"], ply,
                        pos_base_seed=args.position_probe_base_seed,
                        goal_base_seed=args.goal_line_base_seed)
        counts, root_value_stm, root = search_fn(state, seed)
```

with:

```python
    for i, row in enumerate(rows):
        case_id = row["case_id"]
        state, side, root_value_stm, root = search_for_row(
            row, search_fn,
            pos_base_seed=args.position_probe_base_seed,
            goal_base_seed=args.goal_line_base_seed)
```

and replace exactly this block in the per-child loop:

```python
            _, _, child_raw_stm_base = _teacher_infer(child.state, base_raw_evaluator)
            _, _, child_raw_stm_v14b = _teacher_infer(child.state, v14b_raw_evaluator)
```

with nothing, changing the two CSV cells that consumed them:

```python
                "child_raw_black_BASE": raw_black_value(child.state, base_raw_evaluator),
                "child_raw_black_v14b": raw_black_value(child.state, v14b_raw_evaluator),
```

Leave every other line of `main()` untouched. `state` is now unused inside the loop body but is still returned; that is intentional — Task 2 needs it. Do not remove the `_teacher_infer`, `position_state`, `json`, or `Path` imports: `raw_black_value` and `search_for_row` still use them.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_v15_concentration_diagnostic.py -v`
Expected: PASS, 7 tests (the original 5 plus the 2 new).

- [ ] **Step 6: Prove the module still imports and `--help` renders**

Run: `.venv/bin/python -c "from scripts.GPU.alphazero.diagnose_v15_a_continuation_concentration import main, search_for_row, raw_black_value; print('ok')"`
Expected: `ok`

Run: `.venv/bin/python -m scripts.GPU.alphazero.diagnose_v15_a_continuation_concentration --help`
Expected: usage text renders, exit 0.

(The byte-for-byte output regression — a `--limit-cases 1` re-run compared against the committed Phase-0 CSV — needs the real checkpoints, which a fresh worktree lacks. The controller runs it on merged main in Task 3.)

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/diagnose_v15_a_continuation_concentration.py tests/test_v15_concentration_diagnostic.py
git commit -m "refactor(diagnostic): extract search_for_row + raw_black_value from v15 Phase-0

Behavior-preserving. Phase 0.5 imports both rather than copying the
gate-faithful search setup and the to_black(state.to_move) conversion,
so the sign convention cannot drift between the two diagnostics.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: The selected-branch subtree diagnostic

**Files:**
- Create: `scripts/GPU/alphazero/diagnose_v15_a_selected_branch_subtrees.py`
- Create: `tests/test_v15_selected_branch_subtrees.py`

**Interfaces:**
- Consumes: `search_for_row`, `raw_black_value`, `per_child_metrics`, `path_moves_of`, `_real_search_fn`, `_build_raw_evaluator` from the Phase-0 module (Task 1).
- Produces: `load_phase0_rows(path) -> list[dict]`; `group_phase0_by_root(rows) -> OrderedDict[str, list[dict]]`; `select_positive_branches(groups, *, cum_threshold=0.90, max_children=3) -> list[tuple[str, list[dict]]]`; `walk_subtree(branch_root) -> list[MCTSNode]`; `pv_chain(branch_root) -> dict[int, int]`; `node_metrics(node, root, branch_root, pv_index) -> dict`; `aggregate_by_depth(rows, scope) -> list[dict]`; `main(argv) -> int`.

Note: `classify_concentration` is deliberately **not** imported — the label is carried through from the Phase-0 CSV, so importing it would be an unused import.

- [ ] **Step 1: Write the failing tests** — `tests/test_v15_selected_branch_subtrees.py`, using real `MCTSNode` instances (mirroring `tests/test_continuation_extraction.py`), with a `SimpleNamespace` standing in for the game state where only `to_move`/`is_terminal` are read:

```python
import types

from scripts.GPU.alphazero.mcts import MCTSNode, encode_move
from scripts.GPU.alphazero.diagnose_v15_a_selected_branch_subtrees import (
    aggregate_by_depth, group_phase0_by_root, node_metrics, pv_chain,
    select_positive_branches, walk_subtree)


def _state(to_move, terminal=False):
    return types.SimpleNamespace(to_move=to_move,
                                 is_terminal=lambda: terminal)


def _node(parent, move_rc, visits, q, to_move="red", terminal=False):
    """Child of `parent` with the real backup relationship value_sum = q*visits."""
    n = MCTSNode(state=_state(to_move, terminal), parent=parent,
                 move=encode_move(*move_rc), visit_count=visits,
                 value_sum=q * visits)
    if parent is not None:
        parent.children[n.move] = n
    return n


def _phase0_row(root_case_id, child_move, contrib, pos_share, root_black,
                child_q=-0.5):
    return {"root_case_id": root_case_id, "child_move": child_move,
            "child_contribution_share": str(contrib),
            "positive_contribution_share": str(pos_share),
            "root_mcts_black_value": str(root_black),
            "child_q_value": str(child_q),
            "root_case_classification": "concentrated"}


# ---------- selection ----------

def test_select_positive_branches_skips_nonpositive_roots():
    rows = [_phase0_row("neg", "1:1", 0.3, 1.0, -0.19),
            _phase0_row("pos", "2:2", 0.4, 1.0, +0.42)]
    picked = select_positive_branches(group_phase0_by_root(rows))
    assert [cid for cid, _ in picked] == ["pos"]


def test_select_positive_branches_stops_at_cumulative_threshold():
    rows = [_phase0_row("r", "1:1", 0.50, 0.60, +0.5),
            _phase0_row("r", "2:2", 0.30, 0.35, +0.5),   # cum 0.95 >= 0.90 -> stop
            _phase0_row("r", "3:3", 0.04, 0.05, +0.5)]
    (_cid, picked), = select_positive_branches(group_phase0_by_root(rows))
    assert [p["child_move"] for p in picked] == ["1:1", "2:2"]


def test_select_positive_branches_caps_at_max_children():
    rows = [_phase0_row("r", f"{i}:{i}", 0.2, 0.2, +0.5) for i in range(1, 6)]
    (_cid, picked), = select_positive_branches(group_phase0_by_root(rows))
    assert len(picked) == 3            # cum never reaches 0.90; cap wins


def test_select_positive_branches_skips_root_with_zero_positive_mass():
    rows = [_phase0_row("r", "1:1", -0.1, 0.0, +0.01),
            _phase0_row("r", "2:2", -0.2, 0.0, +0.01)]
    assert select_positive_branches(group_phase0_by_root(rows)) == []


# ---------- walk ----------

def test_walk_subtree_visits_only_nodes_with_visits():
    root = MCTSNode(state=_state("black"), visit_count=100, value_sum=50.0)
    branch = _node(root, (1, 1), 90, -0.5)
    kept = _node(branch, (2, 2), 40, 0.4, to_move="black")
    _dropped = _node(branch, (3, 3), 0, 0.0, to_move="black")   # never visited
    deep = _node(kept, (4, 4), 10, -0.2)
    walked = walk_subtree(branch)
    assert set(id(n) for n in walked) == {id(branch), id(kept), id(deep)}


def test_walk_subtree_includes_the_branch_root():
    root = MCTSNode(state=_state("black"), visit_count=10, value_sum=1.0)
    branch = _node(root, (1, 1), 10, -0.3)
    assert walk_subtree(branch) == [branch]


# ---------- PV ----------

def test_pv_chain_marks_only_the_best_child_chain():
    root = MCTSNode(state=_state("black"), visit_count=100, value_sum=10.0)
    branch = _node(root, (1, 1), 90, -0.5)
    best = _node(branch, (2, 2), 70, 0.4, to_move="black")
    other = _node(branch, (3, 3), 20, 0.1, to_move="black")
    deep = _node(best, (4, 4), 60, -0.3)
    chain = pv_chain(branch)
    assert chain == {id(branch): 0, id(best): 1, id(deep): 2}
    assert id(other) not in chain


# ---------- node metrics ----------

def test_node_metrics_depths_shares_and_perspective():
    root = MCTSNode(state=_state("black"), visit_count=100, value_sum=10.0)
    branch = _node(root, (1, 1), 80, -0.5)               # red to move
    deep = _node(branch, (2, 2), 40, 0.25, to_move="black")

    m = node_metrics(deep, root, branch, pv_index=1)
    assert m["depth_from_root"] == 2
    assert m["depth_from_selected_child"] == 1
    assert m["move_from_parent"] == "2:2"
    assert m["path_moves"] == "1:1 2:2"
    assert abs(m["visit_share_from_parent"] - 40 / 80) < 1e-9
    assert abs(m["visit_share_from_root"] - 40 / 100) < 1e-9
    assert abs(m["q_value_node_perspective"] - 0.25) < 1e-9
    assert abs(m["q_value_root_perspective"] - 0.25) < 1e-9   # black to move: unchanged
    assert m["is_pv_path"] is True and m["pv_depth_index"] == 1

    mb = node_metrics(branch, root, branch, pv_index=None)
    assert mb["depth_from_selected_child"] == 0
    assert abs(mb["q_value_node_perspective"] + 0.5) < 1e-9
    assert abs(mb["q_value_root_perspective"] - 0.5) < 1e-9   # red to move: flipped
    assert mb["is_pv_path"] is False and mb["pv_depth_index"] == ""


def test_node_metrics_counts_unvisited_children_and_terminal():
    root = MCTSNode(state=_state("black"), visit_count=10, value_sum=1.0)
    branch = _node(root, (1, 1), 10, -0.3)
    _node(branch, (2, 2), 5, 0.1, to_move="black")
    _node(branch, (3, 3), 0, 0.0, to_move="black")
    m = node_metrics(branch, root, branch, pv_index=0)
    assert m["num_children"] == 2 and m["unvisited_children_count"] == 1
    assert m["is_terminal"] is False

    term = _node(branch, (4, 4), 3, 1.0, to_move="black", terminal=True)
    assert node_metrics(term, root, branch, pv_index=None)["is_terminal"] is True


# ---------- aggregate ----------

def _agg_row(depth, visit_share, raw, terminal=False):
    return {"depth_from_root": depth, "visit_share_from_root": visit_share,
            "raw_black_BASE": "" if terminal else raw,
            "raw_black_v14b": "" if terminal else raw,
            "unvisited_children_count": 0}


def test_aggregate_visit_mass_beats_node_count():
    # 1 raw-positive node holding 60% of the visit mass, 3 raw-negative nodes
    # holding 40% between them: pct_raw_positive=0.25 but the DECISION metric
    # pct_visit_mass_raw_positive=0.60. This is exactly why the two differ.
    rows = [_agg_row(3, 0.60, +0.4)] + [_agg_row(3, 0.40 / 3, -0.2)] * 3
    (rec,) = aggregate_by_depth(rows, "full_subtree")
    assert rec["scope"] == "full_subtree" and rec["depth_from_root"] == 3
    assert rec["nodes_count"] == 4 and rec["raw_scored_nodes_count"] == 4
    assert abs(rec["pct_raw_positive_BASE"] - 0.25) < 1e-9
    assert abs(rec["pct_visit_mass_raw_positive_BASE"] - 0.60) < 1e-9
    assert abs(rec["max_raw_black_BASE"] - 0.4) < 1e-9
    assert abs(rec["weighted_mean_raw_black_BASE"] - (0.6 * 0.4 + 0.4 * -0.2)) < 1e-9


def test_aggregate_excludes_terminal_nodes_from_raw_stats_but_keeps_visit_mass():
    rows = [_agg_row(2, 0.5, +0.3), _agg_row(2, 0.5, None, terminal=True)]
    (rec,) = aggregate_by_depth(rows, "pv_only")
    assert rec["nodes_count"] == 2 and rec["raw_scored_nodes_count"] == 1
    assert abs(rec["total_visit_share_from_root"] - 1.0) < 1e-9
    assert abs(rec["mean_raw_black_BASE"] - 0.3) < 1e-9          # over scored only
    assert abs(rec["pct_raw_positive_BASE"] - 1.0) < 1e-9        # over scored only
    # visit mass denominator is the SCORED mass, so the terminal node cannot
    # silently dilute the decision metric
    assert abs(rec["pct_visit_mass_raw_positive_BASE"] - 1.0) < 1e-9


def test_aggregate_handles_a_depth_with_no_scored_nodes():
    rows = [_agg_row(5, 0.2, None, terminal=True)]
    (rec,) = aggregate_by_depth(rows, "full_subtree")
    assert rec["raw_scored_nodes_count"] == 0
    assert rec["mean_raw_black_BASE"] == ""
    assert rec["pct_visit_mass_raw_positive_BASE"] == ""
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_v15_selected_branch_subtrees.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named '...diagnose_v15_a_selected_branch_subtrees'`.

- [ ] **Step 3: Implement the pure logic** in `scripts/GPU/alphazero/diagnose_v15_a_selected_branch_subtrees.py`:

```python
"""v15 Phase-0.5 (READ-ONLY) diagnostic: selected-branch subtree walk.

Phase 0 established that the optimistic MCTS backup at the A roots is
concentrated: all 17 roots with root_mcts_black_value > 0 classify as
"concentrated", and the top-3 children carry 98.1% of the positive backup
mass. It also found the reason not to build depth-1 correction rows: at the
top child of each overvaluing root the RAW black value is already -0.087
under BASE while its SEARCHED value is +0.619. A node's own NN evaluation
enters its q exactly once, so a child with ~370 visits moves its searched q
by only ~delta/370 -- correcting a depth-1 child from -0.087 to -0.35 shifts
its backed-up value by ~0.0007. Depth-1 correction can therefore only work
through generalization to the deep leaves, the same assumption that failed
for root correction across v2-v14.

This script tests that assumption. It re-runs the deterministic BASE search
on the overvaluing roots, walks EVERY expanded descendant (visit_count >= 1)
beneath each selected positive branch, and records each node's raw value
(BASE and v14b) alongside its searched q, with the principal variation
annotated as a column. The by-depth summary is emitted twice -- over the full
subtree and over PV nodes only -- and the comparison answers: is the raw
optimism path-concentrated, spread across the frontier, or absent entirely?

pct_visit_mass_raw_positive is the decision metric, NOT pct_raw_positive: a
thousand single-visit frontier leaves must not outvote one 300-visit node.

READ-ONLY: reads the BASE + v14b checkpoints, the A probe manifest, and the
Phase-0 CSV; writes two diagnostic CSVs. No manifest, no child replay JSONs,
no training, and no change to mcts.py, calibration_pool.py, trainer.py,
network.py, probe_eval.py, eval_runner.py, or continuation_extraction.py.
"""
from __future__ import annotations

import argparse
import csv
from collections import OrderedDict, defaultdict
from pathlib import Path

from .diagnose_v15_a_continuation_concentration import (
    DEFAULT_A_MANIFEST, DEFAULT_BASE_CHECKPOINT, DEFAULT_V14B_CHECKPOINT,
    _build_raw_evaluator, _real_search_fn, path_moves_of, per_child_metrics,
    raw_black_value, search_for_row)
from .eval_raw_nn_position_rows import to_black
from .mcts import decode_move, encode_move
from .position_probe_cases import load_csv_manifest

DEFAULT_PHASE0_CSV = "logs/eval/v15prep_a_continuation_concentration.csv"
DEFAULT_OUT = "logs/eval/v15prep_a_selected_branch_subtrees.csv"
DEFAULT_SUMMARY_OUT = (
    "logs/eval/v15prep_a_selected_branch_subtrees_by_depth_summary.csv")

FIELDNAMES = [
    "root_case_id", "root_mcts_black_value", "root_case_classification",
    "branch_rank", "root_child_move", "root_child_positive_contribution_share",
    "depth_from_root", "depth_from_selected_child", "path_moves",
    "move_from_parent", "visit_count", "visit_share_from_parent",
    "visit_share_from_root", "q_value_node_perspective",
    "q_value_root_perspective", "raw_black_BASE", "raw_black_v14b",
    "raw_delta_v14b_minus_BASE", "raw_positive_BASE", "raw_positive_v14b",
    "is_pv_path", "pv_depth_index", "num_children", "unvisited_children_count",
    "is_terminal",
]

SUMMARY_FIELDNAMES = (
    ["scope", "depth_from_root", "nodes_count", "raw_scored_nodes_count",
     "unvisited_children_count", "total_visit_share_from_root"]
    + [f"{stat}_{tag}"
       for tag in ("BASE", "v14b")
       for stat in ("mean_raw_black", "weighted_mean_raw_black",
                    "pct_raw_positive", "pct_visit_mass_raw_positive",
                    "max_raw_black")])

TOLERANCE = 1e-6


def _best_child(node):
    """Max-visit child (ties: lowest encoded move id); None if no visited child.

    COPIED from continuation_extraction._best_child: extract_continuations
    raises for the A tag, so that module cannot be used here, and Phase 0 set
    the precedent of copying its generic helpers rather than modifying it.
    """
    visited = [c for c in node.children.values() if c.visit_count > 0]
    if not visited:
        return None
    return min(visited, key=lambda c: (-c.visit_count, c.move))


def load_phase0_rows(csv_path) -> list[dict]:
    """Rows of the Phase-0 concentration CSV, values as strings."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def group_phase0_by_root(rows) -> "OrderedDict[str, list[dict]]":
    """Phase-0 rows grouped by root_case_id, preserving file order."""
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for r in rows:
        groups.setdefault(r["root_case_id"], []).append(r)
    return groups


def select_positive_branches(groups, *, cum_threshold: float = 0.90,
                             max_children: int = 3) -> list:
    """[(root_case_id, [phase-0 child rows in rank order])] for the roots that
    actually overvalue (root_mcts_black_value > 0). Within a root, children
    with positive contribution are ranked by positive_contribution_share
    descending and taken until the cumulative share reaches cum_threshold or
    max_children have been taken. Roots with no positive backup mass are
    skipped -- Phase 0 labels those "broad" with share 0.0, which conflates
    "not overvalued at all" with "overvalued broadly"."""
    out = []
    for cid, rows in groups.items():
        if float(rows[0]["root_mcts_black_value"]) <= 0:
            continue
        positive = [r for r in rows
                    if float(r["child_contribution_share"]) > 0]
        if not positive:
            continue
        positive.sort(key=lambda r: -float(r["positive_contribution_share"]))
        picked, cum = [], 0.0
        for r in positive:
            picked.append(r)
            cum += float(r["positive_contribution_share"])
            if cum >= cum_threshold or len(picked) >= max_children:
                break
        out.append((cid, picked))
    return out


def walk_subtree(branch_root) -> list:
    """Every descendant of branch_root with visit_count >= 1, including
    branch_root itself. Iterative (no recursion depth limit). Unvisited
    children are excluded and never descended into -- MCTS creates them at
    parent expansion but never evaluates them, so they carry no NN value."""
    out, stack = [], [branch_root]
    while stack:
        node = stack.pop()
        if node.visit_count < 1:
            continue
        out.append(node)
        stack.extend(node.children.values())
    return out


def pv_chain(branch_root) -> dict:
    """{id(node): pv_depth_index} along the best-child chain rooted at
    branch_root (branch_root itself is index 0)."""
    chain, node, i = {}, branch_root, 0
    while node is not None:
        chain[id(node)] = i
        node = _best_child(node)
        i += 1
    return chain


def node_metrics(node, root, branch_root, pv_index) -> dict:
    """Tree-derived per-node fields (no NN evaluation). q_value_root_perspective
    is the analysis column: node.q_value is stored in the node's OWN to-move
    perspective (mcts._backup flips the sign once per level, and apply_move
    flips to_move every ply), so converting with node.state.to_move lands every
    node in black's perspective -- which is the root's, since every A root is
    black to move."""
    # node_metrics is only called for selected branch descendants, never for the
    # true MCTS root, so node.parent is guaranteed non-None below.
    path = path_moves_of(node)
    branch_depth = len(path_moves_of(branch_root))
    return {
        "depth_from_root": len(path),
        "depth_from_selected_child": len(path) - branch_depth,
        "path_moves": " ".join(f"{r}:{c}" for r, c in path),
        "move_from_parent": "{}:{}".format(*decode_move(node.move)),
        "visit_count": node.visit_count,
        "visit_share_from_parent": node.visit_count / node.parent.visit_count,
        "visit_share_from_root": node.visit_count / root.visit_count,
        "q_value_node_perspective": node.q_value,
        "q_value_root_perspective": to_black(node.q_value, node.state.to_move),
        "is_pv_path": pv_index is not None,
        "pv_depth_index": "" if pv_index is None else pv_index,
        "num_children": len(node.children),
        "unvisited_children_count": sum(
            1 for c in node.children.values() if c.visit_count < 1),
        "is_terminal": node.state.is_terminal(),
    }


def aggregate_by_depth(rows, scope: str) -> list[dict]:
    """Per-depth summary over `rows` (already filtered to `scope`). Terminal
    nodes carry no raw value (blank cells) and are excluded from every raw_*
    statistic, but still counted in nodes_count and total_visit_share_from_root
    so visit mass is never silently dropped. pct_visit_mass_raw_positive is
    normalized by the SCORED visit mass, so a terminal node cannot dilute the
    decision metric."""
    by_depth = defaultdict(list)
    for r in rows:
        by_depth[int(r["depth_from_root"])].append(r)

    out = []
    for depth in sorted(by_depth):
        group = by_depth[depth]
        rec = {
            "scope": scope,
            "depth_from_root": depth,
            "nodes_count": len(group),
            "unvisited_children_count": sum(
                int(r["unvisited_children_count"]) for r in group),
            "total_visit_share_from_root": sum(
                float(r["visit_share_from_root"]) for r in group),
        }
        scored = [r for r in group if r["raw_black_BASE"] != ""]
        rec["raw_scored_nodes_count"] = len(scored)
        scored_mass = sum(float(r["visit_share_from_root"]) for r in scored)

        for tag in ("BASE", "v14b"):
            if not scored:
                for stat in ("mean_raw_black", "weighted_mean_raw_black",
                             "pct_raw_positive", "pct_visit_mass_raw_positive",
                             "max_raw_black"):
                    rec[f"{stat}_{tag}"] = ""
                continue
            vals = [float(r[f"raw_black_{tag}"]) for r in scored]
            masses = [float(r["visit_share_from_root"]) for r in scored]
            weighted = sum(m * v for m, v in zip(masses, vals))
            pos_mass = sum(m for m, v in zip(masses, vals) if v > 0)
            rec[f"mean_raw_black_{tag}"] = sum(vals) / len(vals)
            rec[f"weighted_mean_raw_black_{tag}"] = (
                weighted / scored_mass if scored_mass > 0 else 0.0)
            rec[f"pct_raw_positive_{tag}"] = (
                sum(1 for v in vals if v > 0) / len(vals))
            rec[f"pct_visit_mass_raw_positive_{tag}"] = (
                pos_mass / scored_mass if scored_mass > 0 else 0.0)
            rec[f"max_raw_black_{tag}"] = max(vals)
        out.append(rec)
    return out
```

- [ ] **Step 4: Run the pure tests**

Run: `.venv/bin/python -m pytest tests/test_v15_selected_branch_subtrees.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 5: Implement `_parse_args` and `main(argv)`** — append to the same file:

```python
def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description="v15 Phase-0.5 (READ-ONLY) diagnostic: walk the full "
                    "expanded subtree under each selected positive branch of "
                    "the overvaluing A roots, recording raw (BASE + v14b) and "
                    "searched values per node with PV annotation, to decide "
                    "whether Phase-1 correction should be PV/path-level, "
                    "frontier/tree-level, or should not be built. Reads the "
                    "checkpoints, the A probe manifest, and the Phase-0 CSV; "
                    "writes two diagnostic CSVs. No manifest, no replay JSONs, "
                    "no training.")
    ap.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT,
                    help="searched (gate-faithful 400-sim MCTS) AND raw-scored "
                         "(eval-mode) checkpoint.")
    ap.add_argument("--v14b-checkpoint", default=DEFAULT_V14B_CHECKPOINT,
                    help="raw-scored (eval-mode) only checkpoint; never searched.")
    ap.add_argument("--a-manifest", default=DEFAULT_A_MANIFEST)
    ap.add_argument("--phase0-csv", default=DEFAULT_PHASE0_CSV,
                    help="the Phase-0 concentration CSV; supplies the roots, "
                         "the selected children, and the cross-check values.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    ap.add_argument("--cum-threshold", type=float, default=0.90)
    ap.add_argument("--max-children", type=int, default=3)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--eval-batch-size", type=int, default=14)
    ap.add_argument("--stall-flush-sims", type=int, default=48)
    ap.add_argument("--position-probe-base-seed", type=int, default=20260616)
    ap.add_argument("--goal-line-base-seed", type=int, default=20260614)
    ap.add_argument("--limit-roots", type=int, default=None,
                    help="process only the first N selected roots (smoke testing).")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    groups = group_phase0_by_root(load_phase0_rows(args.phase0_csv))
    branches = select_positive_branches(groups,
                                        cum_threshold=args.cum_threshold,
                                        max_children=args.max_children)
    if args.limit_roots is not None:
        branches = branches[:args.limit_roots]
    manifest = {r["case_id"]: r
                for r in load_csv_manifest(args.a_manifest)["cases"]}

    base_ev = _build_raw_evaluator(args.base_checkpoint)
    v14b_ev = _build_raw_evaluator(args.v14b_checkpoint)
    search_fn = _real_search_fn(args.base_checkpoint, args.sims,
                                args.eval_batch_size, args.stall_flush_sims)

    out_rows = []
    for cid, picked in branches:
        state, side, root_value_stm, root = search_for_row(
            manifest[cid], search_fn,
            pos_base_seed=args.position_probe_base_seed,
            goal_base_seed=args.goal_line_base_seed)

        # Every A root is black to move; this is what makes "root perspective"
        # and "black perspective" the same thing in every column below.
        if side != "black":
            raise SystemExit(f"{cid}: expected side_to_move 'black', got {side!r}")

        # CHECK 1 -- tree reproduction: the fresh search must reproduce Phase 0's.
        fresh_black = to_black(root_value_stm, side)
        csv_black = float(picked[0]["root_mcts_black_value"])
        if abs(fresh_black - csv_black) > TOLERANCE:
            raise SystemExit(
                f"{cid}: TREE NOT REPRODUCED: fresh root_mcts_black_value="
                f"{fresh_black:+.6f} != Phase-0 CSV {csv_black:+.6f}; the "
                f"search config or seed drifted -- DO NOT trust this run")

        # CHECK 2 -- Phase 0's contribution invariant, on every root.
        metrics = per_child_metrics(root)
        _sum = sum(m["child_contribution_share"] for m in metrics)
        if abs(_sum - root.q_value) > TOLERANCE:
            raise SystemExit(
                f"{cid}: contribution invariant broken: sum={_sum:+.6f} != "
                f"root.q_value={root.q_value:+.6f} (check the (-child.q_value) sign)")

        for rank, prow in enumerate(picked, start=1):
            move_rc = tuple(int(x) for x in prow["child_move"].split(":"))
            branch_root = root.children[encode_move(*move_rc)]

            # CHECK 3 -- cross-CSV perspective tie: the depth-1 node's
            # root-perspective q must equal -child_q_value from Phase 0.
            qrp = to_black(branch_root.q_value, branch_root.state.to_move)
            expected = -float(prow["child_q_value"])
            if abs(qrp - expected) > TOLERANCE:
                raise SystemExit(
                    f"{cid} child {prow['child_move']}: PERSPECTIVE MISMATCH: "
                    f"q_value_root_perspective={qrp:+.6f} != -child_q_value="
                    f"{expected:+.6f} -- the to_black conversion drifted between "
                    f"Phase 0 and Phase 0.5; DO NOT trust this run")

            chain = pv_chain(branch_root)
            for node in walk_subtree(branch_root):
                m = node_metrics(node, root, branch_root, chain.get(id(node)))
                if m["is_terminal"]:
                    raw_base = raw_v14b = delta = ""
                    pos_base = pos_v14b = ""
                else:
                    raw_base = raw_black_value(node.state, base_ev)
                    raw_v14b = raw_black_value(node.state, v14b_ev)
                    delta = raw_v14b - raw_base
                    pos_base, pos_v14b = raw_base > 0, raw_v14b > 0
                out_rows.append({
                    "root_case_id": cid,
                    "root_mcts_black_value": fresh_black,
                    "root_case_classification": prow["root_case_classification"],
                    "branch_rank": rank,
                    "root_child_move": prow["child_move"],
                    "root_child_positive_contribution_share":
                        float(prow["positive_contribution_share"]),
                    "raw_black_BASE": raw_base,
                    "raw_black_v14b": raw_v14b,
                    "raw_delta_v14b_minus_BASE": delta,
                    "raw_positive_BASE": pos_base,
                    "raw_positive_v14b": pos_v14b,
                    **m,
                })
        print(f"[v15 phase0.5] {cid}: {len(picked)} branch(es), "
              f"root_mcts_black_value={fresh_black:+.4f}, checks OK")

    out_rows.sort(key=lambda r: (r["root_case_id"], r["branch_rank"],
                                 r["depth_from_root"], -r["visit_count"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(out_rows)

    summary = (aggregate_by_depth(out_rows, "full_subtree")
               + aggregate_by_depth([r for r in out_rows if r["is_pv_path"]],
                                    "pv_only"))
    with open(args.summary_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        w.writeheader()
        w.writerows(summary)

    print(f"\n[v15 phase0.5] by-depth summary "
          f"(pct_visit_mass_raw_positive is the decision metric):")
    for rec in summary:
        print(f"  {rec['scope']:<13} d={rec['depth_from_root']:<3} "
              f"n={rec['nodes_count']:<5} scored={rec['raw_scored_nodes_count']:<5} "
              f"vmass={rec['total_visit_share_from_root']:.3f} "
              f"wmean_raw_BASE={rec['weighted_mean_raw_black_BASE']} "
              f"pct_vmass_raw_pos_BASE={rec['pct_visit_mass_raw_positive_BASE']}")
    print(f"\nwrote {len(out_rows)} node rows -> {args.out}")
    print(f"wrote {len(summary)} summary rows -> {args.summary_out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

- [ ] **Step 6: Verify the module imports and `--help` renders**

Run: `.venv/bin/python -c "from scripts.GPU.alphazero.diagnose_v15_a_selected_branch_subtrees import main; print('ok')"`
Expected: `ok`

Run: `.venv/bin/python -m scripts.GPU.alphazero.diagnose_v15_a_selected_branch_subtrees --help`
Expected: usage text renders, exit 0.

- [ ] **Step 7: Re-run both test files**

Run: `.venv/bin/python -m pytest tests/test_v15_selected_branch_subtrees.py tests/test_v15_concentration_diagnostic.py -v`
Expected: PASS, 19 tests (12 + 7).

- [ ] **Step 8: Commit**

```bash
git add scripts/GPU/alphazero/diagnose_v15_a_selected_branch_subtrees.py tests/test_v15_selected_branch_subtrees.py
git commit -m "feat(diagnostic): v15 Phase-0.5 selected-branch subtree diagnostic (read-only)

Walks every expanded descendant (visit_count >= 1) under each selected
positive branch of the 17 overvaluing A roots, recording raw (BASE + v14b)
and searched values per node with PV annotation, plus a by-depth summary
emitted over the full subtree and over PV nodes only. Three fail-loud
integrity checks: tree reproduction against the Phase-0 CSV, the
contribution invariant, and the depth-1 cross-CSV perspective tie.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Regression-check Phase 0, merge, and run the diagnostic (controller-run)

- [ ] **Step 1: Full suite in the worktree** — `.venv/bin/python -m pytest tests/ -q`. Expected: the new tests pass, and EXACTLY the known 14 failed + 6 errors from gitignored-data fixtures. No other failures.

- [ ] **Step 2: FF-merge to main and run the authoritative suite** — `git merge --ff-only feature/tvc-v15-phase05-subtree-diagnostic`, then `.venv/bin/python -m pytest tests/ -q`. Expected: **1416 + 14 = 1430 passed, 0 failed, 0 errors** (2 new tests in Task 1, 12 in Task 2). Do not push yet.

- [ ] **Step 3: Prove the Task-1 refactor did not change Phase-0's output.** On merged main (checkpoints present), re-run Phase 0 for one root into a scratch path and diff its row against the committed CSV:

```bash
.venv/bin/python -m scripts.GPU.alphazero.diagnose_v15_a_continuation_concentration \
  --limit-cases 1 --out /tmp/phase0_regression.csv
head -1 logs/eval/v15prep_a_continuation_concentration.csv > /tmp/phase0_expected.csv
grep '^black_loss_game_000347_predrop_ply_73_drop_75,' \
  logs/eval/v15prep_a_continuation_concentration.csv >> /tmp/phase0_expected.csv
diff /tmp/phase0_regression.csv /tmp/phase0_expected.csv && echo "PHASE-0 OUTPUT UNCHANGED"
```

Expected: `PHASE-0 OUTPUT UNCHANGED`, exit 0. If the diff is non-empty, the refactor changed behavior — **stop, revert the merge, and report**; do not push and do not run Phase 0.5.

- [ ] **Step 4: Push and clean up** — `git push origin main`; remove the `.venv` symlink, remove the worktree, prune, delete the branch.

- [ ] **Step 5: Run the diagnostic** (~17 seeded 400-sim searches plus ~13.6k raw forwards; expect 8–15 minutes):

```bash
.venv/bin/python -m scripts.GPU.alphazero.diagnose_v15_a_selected_branch_subtrees \
  --out logs/eval/v15prep_a_selected_branch_subtrees.csv \
  --summary-out logs/eval/v15prep_a_selected_branch_subtrees_by_depth_summary.csv
```

Expected: all three integrity checks pass on every root (`checks OK` per root); two CSVs written. If any check raises, the read is untrustworthy — report the failure verbatim and stop. If the run is too heavy for the environment, hand the operator the exact command instead of forcing it.

---

### Task 4: Stop and report (controller-run)

- [ ] **Step 1: Report to the operator** — the by-depth summary table (full_subtree versus pv_only), specifically: the depth at which `weighted_mean_raw_black_BASE` first turns positive, `pct_visit_mass_raw_positive_BASE` by depth in both scopes, the terminal-node count per depth (`nodes_count - raw_scored_nodes_count`), and the same for v14b.

- [ ] **Step 2: STOP.** Do NOT design or build Phase 1. The read belongs to the operator, per the spec's decision table:

| Read | Next |
|---|---|
| Raw rises positive with depth **and** most raw-positive visit mass sits on or near the PV | Phase 1 = PV/path correction rows |
| Raw rises positive **broadly** across many sibling/frontier nodes | Few-row Phase 1 is too narrow — design a tree/frontier-level correction, or stop value calibration |
| Raw stays ≤ 0 across the visited subtree while root backup stays positive | Value calibration is the wrong lever — redirect to search behavior or training data; close this branch |

---

## Notes on two spec addenda made while writing this plan

Both are recorded in the spec (`§6`, `§7`) and are not implementer discretion:

1. **`is_terminal` and `unvisited_children_count` are per-node CSV columns.** A subtree walk reaches terminal nodes, which Phase 0 never could (depth-1 children of a midgame root are never terminal). Terminal nodes are visited but never expanded, have no legal moves, and would make `_teacher_infer` build zero-width arrays. They are emitted with blank raw values and excluded from every `raw_*` statistic, while still counting toward `nodes_count` and `total_visit_share_from_root`. `unvisited_children_count` is per-node because the by-depth summary aggregates it.

2. **`pct_visit_mass_raw_positive` is normalized by the *scored* visit mass**, not the total. Otherwise a terminal node carrying visit mass would silently dilute the decision metric toward "not optimistic".
