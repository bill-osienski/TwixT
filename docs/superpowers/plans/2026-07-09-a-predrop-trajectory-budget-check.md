# A Pre-Drop Trajectory Budget Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A read-only diagnostic that re-searches a 6-ply window around the drop for 5 representative A loss games at 400 and 6400 sims, so the operator can decide whether the "post-opening sharp value drop" survives adequate search or is a shallow-search artifact.

**Architecture:** One new script reusing the already-reviewed `search_for_row` from the Phase-0 diagnostic (gate-faithful seed + reconstruction) and `to_black` for perspective. One evaluator is built and reused across both budgets; only `MCTSConfig` differs. Writes one per-position CSV and one per-(case,budget) summary CSV. No manifest, no training, no change to `mcts.py`, `eval_runner.py`, `trainer.py`, `network.py`, `calibration_pool.py`, or any builder.

**Tech Stack:** Python 3.14 / MLX, pytest.

**Spec:** `docs/superpowers/specs/2026-07-09-a-predrop-trajectory-budget-check-design.md` (APPROVED).

## Global Constraints

- Python: always `.venv/bin/python`; tests `.venv/bin/python -m pytest <file> -v`. Authoritative full-suite baseline on merged main = **1430 passed**.
- **READ-ONLY.** The script reads one checkpoint, the A probe manifest, and the replay JSONs; it writes exactly two CSVs. It must NOT modify `mcts.py`, `eval_runner.py`, `probe_eval.py`, `trainer.py`, `network.py`, `calibration_pool.py`, `continuation_extraction.py`, or any manifest/builder. **No `MCTSConfig` change — no FPU, no `prior_top_k`, no `prior_min_mass`, no `c_puct` knob.** Those belong to v16, which is not this branch.
- **Gate-faithful search:** `EvalConfig(mcts_sims=<budget>, mcts_eval_batch_size=14, mcts_stall_flush_sims=48)`, `_default_evaluator_factory` (train-mode BN, `compile=True`), `MCTS(...).search_with_root(state, add_noise=False)`, seeded via `row_seed(CORRECTION_TAG, game_idx, ply, pos_base_seed=20260616, goal_base_seed=20260614)`.
- **Build the evaluator ONCE and reuse it across budgets**; construct a fresh `MCTSConfig` per budget. Do not call `_real_search_fn` per budget (it builds a new evaluator each time, and the MLX `compile=True` sequential-eval gotcha is a known hazard).
- **Perspective:** `root_black_value = to_black(root_value_stm, side)`; `top_child_q_black = to_black(child.q_value, child.state.to_move)` — the CHILD's to-move, never the root's side. `side` is `"red"` if `ply % 2 == 0` else `"black"`; `position_state` independently raises if that disagrees with the board.
- **`drop_ply` is a manifest column.** Do not parse it from `case_id`.
- Reuse `search_for_row` from `diagnose_v15_a_continuation_concentration` (already reviewed; it owns the seed + reconstruction convention). Copy `_best_child` for the PV chain, as Phase 0/0.5 did — `continuation_extraction.extract_continuations` raises for the A tag.
- NEVER `sys.modules.pop("mlx")` in tests. Tests are pure: no checkpoints, no MCTS.
- Worktree `feature/a-predrop-trajectory-budget-check`; symlink `.venv`; FF-merge (no `--no-ff`, never force-push); code-commit trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; file-scoped `git add`; **locate code by content, not line numbers**. A fresh worktree lacks gitignored game-log data and checkpoints → whole-repo suite there = exactly 14 failed + 6 errors; judge on file-scoped runs; authoritative suite on merged main.

## Interfaces (verified — use exactly)

- `from .diagnose_v15_a_continuation_concentration import search_for_row` → `(state, side, root_value_stm, root) = search_for_row(row, search_fn, pos_base_seed=..., goal_base_seed=...)`. `row` needs keys `replay_path`, `position_ply`, `side_to_move`, `game_idx`.
- `from .eval_raw_nn_position_rows import to_black`; `from .mcts import decode_move`; `from .position_probe_cases import load_csv_manifest` (`["cases"]`, `game_idx` coerced to `int`).
- A manifest columns present: `case_id, game_idx, replay_path, position_ply, drop_ply, side_to_move`.
- Replay JSON: `{"board_size", "n_moves", "moves": [{"ply","player","row","col","root_value", ...}]}`. `moves[ply]["root_value"]` is the original 400-sim value in **side-to-move** perspective.
- `MCTSNode`: `.children` (`Dict[move_id, MCTSNode]`), `.visit_count`, `.q_value`, `.state`, `.move`. `root.visit_count == sum(child.visit_count)`.
- Verified inputs: all 5 case-ids present; `drop_ply == predrop_ply + 2` for all; every window ply black-to-move; only game 347 clips (ply 79 of `n_moves=79`). 29 positions total.

## File Structure

| File | Role |
|---|---|
| `scripts/GPU/alphazero/diagnose_a_predrop_trajectory_budget.py` (create) | the read-only trajectory diagnostic |
| `tests/test_a_predrop_trajectory_budget.py` (create) | pure tests: window construction/clipping, ply parity, summary aggregation |

---

### Task 1: The trajectory diagnostic

**Files:** Create `scripts/GPU/alphazero/diagnose_a_predrop_trajectory_budget.py`, `tests/test_a_predrop_trajectory_budget.py`.

**Interfaces:**
- Produces: `side_for_ply(ply) -> str`; `ply_window(predrop_ply, drop_ply, n_moves) -> list[int]`; `summarize_case(rows, drop_ply) -> dict`; `main(argv) -> int`.

- [ ] **Step 1: Write the failing tests** — `tests/test_a_predrop_trajectory_budget.py`:

```python
from scripts.GPU.alphazero.diagnose_a_predrop_trajectory_budget import (
    ply_window, side_for_ply, summarize_case)


def test_side_for_ply_parity():
    # red moves first (ply 0), so even plies are red-to-move
    assert side_for_ply(0) == "red"
    assert side_for_ply(19) == "black"
    assert side_for_ply(20) == "red"


def test_ply_window_spans_predrop_and_drop():
    # predrop=19, drop=21 -> {15,17,19} u {21,23,25}
    assert ply_window(19, 21, n_moves=49) == [15, 17, 19, 21, 23, 25]


def test_ply_window_clips_to_replay_length():
    # game 347: predrop=73, drop=75, n_moves=79 -> ply 79 is out of range
    assert ply_window(73, 75, n_moves=79) == [69, 71, 73, 75, 77]


def test_ply_window_clips_negative_plies_and_dedupes():
    # predrop=2, drop=4 -> {-2,0,2} u {4,6,8}; -2 dropped, nothing duplicated
    assert ply_window(2, 4, n_moves=7) == [0, 2, 4, 6]


def test_ply_window_dedupes_when_drop_overlaps_predrop_offsets():
    # a hypothetical drop only 2 plies later still yields 6 distinct plies,
    # but a larger predrop offset overlap must not double-count
    assert ply_window(10, 12, n_moves=100) == [6, 8, 10, 12, 14, 16]
    assert len(ply_window(10, 12, n_moves=100)) == 6


def test_summarize_case_splits_at_drop_ply():
    rows = [
        {"ply": 15, "root_black_value": 0.4},
        {"ply": 17, "root_black_value": 0.6},
        {"ply": 19, "root_black_value": 0.8},   # predrop, still pre
        {"ply": 21, "root_black_value": 0.0},   # drop_ply -> post
        {"ply": 23, "root_black_value": -0.3},
    ]
    s = summarize_case(rows, drop_ply=21)
    assert s["n_pre"] == 3 and s["n_post"] == 2
    assert abs(s["pre_drop_mean"] - 0.6) < 1e-9
    assert abs(s["post_drop_mean"] - (-0.15)) < 1e-9
    assert abs(s["drop_delta"] - (-0.75)) < 1e-9
    assert abs(s["max_pre_drop_value"] - 0.8) < 1e-9
    assert s["ply_of_max_pre_drop"] == 19


def test_summarize_case_handles_empty_post_side():
    rows = [{"ply": 5, "root_black_value": 0.2}]
    s = summarize_case(rows, drop_ply=7)
    assert s["n_post"] == 0
    assert s["post_drop_mean"] == "" and s["drop_delta"] == ""
    assert abs(s["pre_drop_mean"] - 0.2) < 1e-9
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_a_predrop_trajectory_budget.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the pure logic:**

```python
"""READ-ONLY diagnostic: does the A 'post-opening sharp value drop' survive
deeper search, or is it shallow-search optimism disappearing?

The Targeted Value Calibration line selected its A cases from positions where a
400-sim replay root_value was high and then collapsed. But root.q_value is the
unweighted mean of the raw leaf evaluations backed up through it, and at 400
sims against ~500 legal moves the expansion frontier is 89% single-visit
opponent blunders after which black really is winning. The budget sweep on the
30 A roots showed the metric decaying +0.2570 (400) -> +0.0626 (1600) ->
-0.0451 (6400). root.q_value trains nothing (the value target is z, the game
outcome); it is only the gate metric and the replay `root_value` from which the
A cases were selected.

So this script re-searches a 6-ply window spanning predrop_ply -> drop_ply for
5 representative A loss games, at 400 and 6400 sims, and records the trajectory.
If the 6400 curve is already flat or negative before the drop, the phenomenon is
an artifact and the value-calibration line closes.

READ-ONLY: reads one checkpoint, the A probe manifest, and the replay JSONs;
writes two CSVs. No MCTSConfig change (FPU / prior_top_k / c_puct belong to v16),
no manifest, no training.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from .diagnose_v15_a_continuation_concentration import search_for_row
from .eval_raw_nn_position_rows import to_black
from .mcts import decode_move
from .position_probe_cases import load_csv_manifest

DEFAULT_A_MANIFEST = (
    "logs/eval/loss_analysis_v2_calib020_0001_vs_0379_black/"
    "0001_black_post_opening_top30_predrop_probe_manifest.csv")
DEFAULT_CHECKPOINT = (
    "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors")
DEFAULT_OUT = "logs/eval/v16_a_predrop_trajectory_budget.csv"
DEFAULT_SUMMARY_OUT = "logs/eval/v16_a_predrop_trajectory_budget_summary.csv"
DEFAULT_CASE_IDS = (
    "black_loss_game_000281_predrop_ply_19_drop_21",
    "black_loss_game_000259_predrop_ply_35_drop_37",
    "black_loss_game_000127_predrop_ply_33_drop_35",
    "black_loss_game_000611_predrop_ply_19_drop_21",
    "black_loss_game_000347_predrop_ply_73_drop_75",
)
PREDROP_OFFSETS = (-4, -2, 0)
DROP_OFFSETS = (0, 2, 4)

FIELDNAMES = [
    "case_id", "replay_path", "budget_sims", "ply", "predrop_ply", "drop_ply",
    "relative_to_predrop", "relative_to_drop", "side_to_move",
    "root_black_value", "replay_stored_root_black_value",
    "top_move", "top_visit_share", "top_child_q_black", "pv_path",
]
SUMMARY_FIELDNAMES = [
    "case_id", "budget_sims", "n_pre", "n_post", "pre_drop_mean",
    "post_drop_mean", "drop_delta", "max_pre_drop_value", "ply_of_max_pre_drop",
]


def side_for_ply(ply: int) -> str:
    """Red moves first (ply 0), so even plies are red-to-move. position_state
    independently raises if this disagrees with the reconstructed board."""
    return "red" if ply % 2 == 0 else "black"


def ply_window(predrop_ply: int, drop_ply: int, n_moves: int) -> list[int]:
    """{predrop-4, predrop-2, predrop} u {drop, drop+2, drop+4}, sorted,
    deduplicated, clipped to a ply the replay can reconstruct (position_state
    requires 0 <= ply < len(moves))."""
    plies = ({predrop_ply + o for o in PREDROP_OFFSETS}
             | {drop_ply + o for o in DROP_OFFSETS})
    return [p for p in sorted(plies) if 0 <= p < n_moves]


def summarize_case(rows, drop_ply: int) -> dict:
    """Pre/post split at drop_ply (the drop ply itself is POST). Blank cells
    rather than 0.0 when a side is empty -- 0.0 would read as 'no drop'."""
    pre = [r for r in rows if r["ply"] < drop_ply]
    post = [r for r in rows if r["ply"] >= drop_ply]
    out = {"n_pre": len(pre), "n_post": len(post)}
    out["pre_drop_mean"] = (
        sum(r["root_black_value"] for r in pre) / len(pre) if pre else "")
    out["post_drop_mean"] = (
        sum(r["root_black_value"] for r in post) / len(post) if post else "")
    out["drop_delta"] = (
        out["post_drop_mean"] - out["pre_drop_mean"] if pre and post else "")
    if pre:
        best = max(pre, key=lambda r: r["root_black_value"])
        out["max_pre_drop_value"] = best["root_black_value"]
        out["ply_of_max_pre_drop"] = best["ply"]
    else:
        out["max_pre_drop_value"] = ""
        out["ply_of_max_pre_drop"] = ""
    return out
```

- [ ] **Step 4: Run the logic tests** — `.venv/bin/python -m pytest tests/test_a_predrop_trajectory_budget.py -v` → 7 PASS.

- [ ] **Step 5: Implement the search helpers, `_parse_args`, and `main(argv)`** — append to the same file:

```python
def _best_child(node):
    """Max-visit child (ties: lowest encoded move id); None if no visited child.

    COPIED from continuation_extraction._best_child: extract_continuations
    raises for the A tag, so that module cannot be used here.
    """
    visited = [c for c in node.children.values() if c.visit_count > 0]
    if not visited:
        return None
    return min(visited, key=lambda c: (-c.visit_count, c.move))


def pv_path_of(root, max_depth: int = 8) -> str:
    """Best-child chain from the root, as 'r:c r:c ...'."""
    moves, node = [], _best_child(root)
    while node is not None and len(moves) < max_depth:
        moves.append("{}:{}".format(*decode_move(node.move)))
        node = _best_child(node)
    return " ".join(moves)


def _search_fns(checkpoint: str, budgets, eval_batch_size: int,
                stall_flush_sims: int) -> dict:
    """One evaluator, reused across budgets; only MCTSConfig differs. Rebuilding
    an evaluator per budget would be slower and trips the known MLX compile=True
    sequential-eval gotcha."""
    from .eval_runner import EvalConfig, cfg_from, _default_evaluator_factory
    from .mcts import MCTS
    evaluator = _default_evaluator_factory(checkpoint)
    fns = {}
    for sims in budgets:
        cfg = cfg_from(EvalConfig(mcts_sims=sims,
                                  mcts_eval_batch_size=eval_batch_size,
                                  mcts_stall_flush_sims=stall_flush_sims))

        def fn(state, seed, cfg=cfg):        # default arg: bind cfg per budget
            return MCTS(evaluator, cfg, random.Random(seed)).search_with_root(
                state, add_noise=False)

        fns[sims] = fn
    return fns


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description="READ-ONLY: re-search the plies around each A case's value "
                    "drop at several sim budgets, to test whether the "
                    "'post-opening sharp value drop' survives deeper search or "
                    "is shallow-search optimism disappearing. No MCTSConfig "
                    "change, no manifest, no training.")
    ap.add_argument("--a-manifest", default=DEFAULT_A_MANIFEST)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--case-ids", nargs="*", default=list(DEFAULT_CASE_IDS))
    ap.add_argument("--budgets", default="400,6400",
                    help="comma-separated sim budgets, e.g. '400,1600,6400'")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    ap.add_argument("--eval-batch-size", type=int, default=14)
    ap.add_argument("--stall-flush-sims", type=int, default=48)
    ap.add_argument("--position-probe-base-seed", type=int, default=20260616)
    ap.add_argument("--goal-line-base-seed", type=int, default=20260614)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]

    by_case = {r["case_id"]: r
               for r in load_csv_manifest(args.a_manifest)["cases"]}
    missing = [c for c in args.case_ids if c not in by_case]
    if missing:
        raise SystemExit(f"case_ids not in {args.a_manifest}: {missing}")

    search_fns = _search_fns(args.checkpoint, budgets, args.eval_batch_size,
                             args.stall_flush_sims)

    out_rows, summary_rows, stored_diffs = [], [], []
    for cid in args.case_ids:
        case = by_case[cid]
        predrop = int(float(case["position_ply"]))
        drop = int(float(case["drop_ply"]))
        replay = json.loads(Path(case["replay_path"]).read_text())
        window = ply_window(predrop, drop, len(replay["moves"]))
        skipped = sorted(({predrop + o for o in PREDROP_OFFSETS}
                          | {drop + o for o in DROP_OFFSETS}) - set(window))
        if skipped:
            print(f"[trajectory] {cid}: skipped out-of-range plies {skipped} "
                  f"(n_moves={len(replay['moves'])})")

        per_budget = {b: [] for b in budgets}
        for sims in budgets:
            for ply in window:
                side = side_for_ply(ply)
                row = {"replay_path": case["replay_path"], "position_ply": ply,
                       "side_to_move": side, "game_idx": case["game_idx"]}
                _state, side_out, root_value_stm, root = search_for_row(
                    row, search_fns[sims],
                    pos_base_seed=args.position_probe_base_seed,
                    goal_base_seed=args.goal_line_base_seed)
                root_black = to_black(root_value_stm, side_out)

                stored_black = to_black(
                    float(replay["moves"][ply]["root_value"]), side_out)
                if sims == 400:
                    stored_diffs.append(abs(root_black - stored_black))

                top = _best_child(root)
                out_rows.append({
                    "case_id": cid,
                    "replay_path": case["replay_path"],
                    "budget_sims": sims,
                    "ply": ply,
                    "predrop_ply": predrop,
                    "drop_ply": drop,
                    "relative_to_predrop": ply - predrop,
                    "relative_to_drop": ply - drop,
                    "side_to_move": side_out,
                    "root_black_value": root_black,
                    "replay_stored_root_black_value": stored_black,
                    "top_move": "" if top is None else
                                "{}:{}".format(*decode_move(top.move)),
                    "top_visit_share": "" if top is None else
                                       top.visit_count / root.visit_count,
                    "top_child_q_black": "" if top is None else
                                         to_black(top.q_value,
                                                  top.state.to_move),
                    "pv_path": pv_path_of(root),
                })
                per_budget[sims].append({"ply": ply,
                                         "root_black_value": root_black})
            s = summarize_case(per_budget[sims], drop)
            s.update({"case_id": cid, "budget_sims": sims})
            summary_rows.append(s)
            print(f"[trajectory] {cid} @ {sims} sims: pre={s['pre_drop_mean']} "
                  f"post={s['post_drop_mean']} delta={s['drop_delta']} "
                  f"max_pre={s['max_pre_drop_value']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(out_rows)
    with open(args.summary_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        w.writeheader()
        w.writerows(summary_rows)

    if stored_diffs:
        print(f"\n[trajectory] 400-sim re-run vs the replay's stored "
              f"root_value: max |diff| = {max(stored_diffs):.4f} over "
              f"{len(stored_diffs)} plies (small => seed + perspective agree "
              f"with the search that originally defined the drop)")
    print(f"wrote {len(out_rows)} rows -> {args.out}")
    print(f"wrote {len(summary_rows)} summary rows -> {args.summary_out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

- [ ] **Step 6: Verify import and `--help`**

Run: `.venv/bin/python -c "from scripts.GPU.alphazero.diagnose_a_predrop_trajectory_budget import main; print('ok')"` → `ok`
Run: `.venv/bin/python -m scripts.GPU.alphazero.diagnose_a_predrop_trajectory_budget --help` → usage renders, exit 0.

- [ ] **Step 7: Re-run the tests** — `.venv/bin/python -m pytest tests/test_a_predrop_trajectory_budget.py -v` → 7 PASS.

- [ ] **Step 8: Commit**

```bash
git add scripts/GPU/alphazero/diagnose_a_predrop_trajectory_budget.py tests/test_a_predrop_trajectory_budget.py
git commit -m "feat(diagnostic): A pre-drop trajectory budget check (read-only)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Suite, merge, run (controller-run)

- [ ] **Step 1:** Worktree full suite → exactly the known 14 failed + 6 errors, plus the 7 new tests passing.
- [ ] **Step 2:** FF-merge to main; authoritative suite → **1430 + 7 = 1437 passed, 0 failed**; push.
- [ ] **Step 3:** Run: `.venv/bin/python -m scripts.GPU.alphazero.diagnose_a_predrop_trajectory_budget --out logs/eval/v16_a_predrop_trajectory_budget.csv --summary-out logs/eval/v16_a_predrop_trajectory_budget_summary.csv` (58 searches, ~197k sims; expect roughly 1–1.5 h — run in background). **Check the reported max |diff| between the 400-sim re-run and the replay's stored `root_value` first.** If it is large, the seed or perspective disagrees with the search that defined the drop and the 6400 curve cannot be compared — stop and report.
- [ ] **Step 4: STOP and report** the two trajectories per case. Do not start v16 or any model work; the decision table in spec §5 is the operator's.
