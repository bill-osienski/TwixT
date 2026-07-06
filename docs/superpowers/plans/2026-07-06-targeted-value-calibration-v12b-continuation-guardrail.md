# Targeted Value Calibration v12b — Continuation Guardrail Rows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the v12 guardrail hinge also cover searched-continuation states — one loader gate so a guardrail row carrying `extra_moves_json` reconstructs the continuation state, plus a new manifest builder that clones B/C/D root and C/D continuation rows into value-only guardrail rows.

**Architecture:** v12b reuses the v12 objective unchanged (`asymmetric_guardrail_retention`, black-perspective sign, one-sided hinge, no policy CE). The only code is (1) one condition in `build_calibration_position` so a guardrail row with a non-empty `extra_moves_json` walks the existing `_apply_extra_moves` continuation path, and (2) a new builder that emits root guardrail clones (sign from `side_to_move`, continuation fields blanked) and continuation guardrail clones (sign from `continuation_side_to_move`, continuation reconstruction fields preserved). No trainer.py change, no new loss mode, no new CLI flag, no gate change, no gradient projection.

**Tech Stack:** Python 3.14 / MLX (Apple Metal), pytest.

**Spec:** `docs/superpowers/specs/2026-07-06-targeted-value-calibration-v12b-continuation-guardrail-design.md` (APPROVED — do not redesign).

## Global Constraints

- Python: always `.venv/bin/python`; tests `.venv/bin/python -m pytest <file> -v`; full-suite baseline on main after v12: **1346 passed**.
- NEVER `sys.modules.pop("mlx")` in tests.
- **The only loader change** is the `build_calibration_position` extra-moves gate (Task 1). The `build_calibration_sample` GUARDRAIL validation branch, `from_manifest`, `_ALLOWED_RETENTION_MODE_SETS`, and trainer.py are UNCHANGED.
- **Byte-identical v12 root behavior:** a guardrail row with blank `extra_moves_json` still reconstructs the root exactly as v12 (the gate only fires when `extra_moves_json` is non-empty).
- **Sign rule (critical correctness):** root guardrail clone `sign = +1 if side_to_move=="black" else -1`; continuation guardrail clone `sign = +1 if continuation_side_to_move=="black" else -1`. `target_black_value = float(teacher_value) * sign`. Using the root side for a continuation clone would flip the target on odd-depth continuations.
- **Value-only:** guardrail rows carry NO policy CE; every clone blanks `teacher_policy_json`/`root_visits_json` (+ root/search-metadata scalars). `teacher_value` is kept as provenance only.
- **Continuation clones PRESERVE** the reconstruction/identity columns `extra_moves_json`, `continuation_side_to_move`, `continuation_legal_moves_sha1`, `continuation_depth`, `continuation_parent_case_id`, `continuation_source`, `continuation_path_moves`. **Root clones BLANK** all of those.
- **Drops:** `goal_line_continuation_retention` (B stays root-only) and `red_predrop_root_value_retention` (D root already covered) are NOT cloned; the source root/continuation rows themselves are not kept.
- Margin stays 0.10; training surface stays `--freeze-batchnorm-stats --train-value-head-and-final-block`.
- Worktree `feature/tvc-v12b-continuation-guardrail`; a fresh worktree lacks gitignored game-log data → known 14F+6E in the whole-repo suite there; judge tasks on file-scoped runs, authoritative suite on merged main. Per-task commits, FF-merge (no `--no-ff`, never force-push). Trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. File-scoped `git add`. Locate code by content, not line numbers.

## File Structure

| File | Role |
|---|---|
| `scripts/GPU/alphazero/calibration_pool.py` (modify) | one gate in `build_calibration_position` (guardrail rows with `extra_moves_json` walk the continuation path) |
| `scripts/GPU/alphazero/build_v12b_continuation_guardrail_manifest.py` (create) | root + C/D continuation guardrail clones from the v7 manifest |
| `scripts/GPU/alphazero/smoke_v12b_continuation_guardrail.py` (create) | gate-0 smoke |
| `tests/test_asymmetric_guardrail_continuation_loader.py` (create) | loader-reconstruction tests (uses committed fixture) |
| `tests/test_build_v12b_continuation_guardrail_manifest.py` (create) | builder clone + routing/drop tests |

**Task → work-item map:** T1 = loader gate + tests; T2 = builder + tests; T3 = smoke + full suite + merge (controller-run).

---

### Task 1: Loader gate — guardrail continuation reconstruction

**Files:**
- Modify: `scripts/GPU/alphazero/calibration_pool.py` (`build_calibration_position`)
- Test: `tests/test_asymmetric_guardrail_continuation_loader.py` (create)

**Interfaces:**
- Consumes: existing `build_calibration_sample`, `build_calibration_position`, `_apply_extra_moves`, `GUARDRAIL_LOSS_MODE`, `CONTINUATION_LOSS_MODE`, `split_samples_with_guardrail`, `target_in_to_move`, `legal_moves_sha1`, `position_state`.
- Produces: a guardrail row (`loss_mode == GUARDRAIL_LOSS_MODE`) with a non-empty `extra_moves_json` reconstructs the continuation state — `record.ply == position_ply + n_extra`, `record.to_move == continuation_side_to_move`, `record.outcome == target_in_to_move(continuation_side, target_black_value)`. Task 2's continuation clones rely on this.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_asymmetric_guardrail_continuation_loader.py`:

```python
"""v12b: a guardrail row carrying extra_moves_json reconstructs the CONTINUATION
state (not the root), with the target/sign in continuation-side perspective.
Root guardrail rows (blank extra_moves_json) still reconstruct the root."""
import json

import pytest

from scripts.GPU.alphazero.calibration_pool import (
    GUARDRAIL_LOSS_MODE, build_calibration_sample, split_samples_with_guardrail,
    target_in_to_move, legal_moves_sha1)
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
from tests.goal_line_probe_fixtures import legal_replay


def _root_state(replay):
    return position_state(replay, 5, "black")   # plies 0-4 applied, black to move


def _apply_n(replay, n):
    """Apply n legal moves from the ply-5 root; return (extra_moves, final_state)."""
    state = _root_state(replay)
    moves = []
    for _ in range(n):
        m = state.legal_moves()[0]
        moves.append({"row": m[0], "col": m[1]})
        state = state.apply_move(m)
    return moves, state


def _cont_guardrail_case(tmp_path, n_moves, target_black):
    rp = tmp_path / "game_000001.json"
    replay = legal_replay(9, game_idx=1)
    rp.write_text(json.dumps(replay))
    moves, final = _apply_n(replay, n_moves)
    case = {
        "game_idx": "1", "case_id": "game_000001_ply_005__cont__guardrail",
        "replay_path": str(rp), "position_ply": "5", "side_to_move": "black",
        "tag": "old_post_opening_continuation_guardrail_retention",
        "loss_mode": GUARDRAIL_LOSS_MODE,
        "target_black_value": repr(target_black),
        "teacher_value": repr(target_black),      # provenance only
        "extra_moves_json": json.dumps(moves),
        "continuation_side_to_move": final.to_move,
        "continuation_legal_moves_sha1": legal_moves_sha1(final.legal_moves()),
        "continuation_depth": str(n_moves), "continuation_source": "pv",
        "teacher_policy_json": "", "root_visits_json": "",
    }
    return case, final


def test_continuation_guardrail_reconstructs_even_depth(tmp_path):
    # 2 moves from black root -> black to move (even depth)
    case, final = _cont_guardrail_case(tmp_path, 2, 0.30)
    s = build_calibration_sample(case, calibration_target=-0.35)
    assert s.loss_mode == GUARDRAIL_LOSS_MODE
    assert s.has_policy_target is False
    rec = s.record
    assert rec.ply == 7                                       # 5 + 2
    assert rec.to_move == final.to_move                       # continuation side
    assert rec.legal_moves == final.legal_moves()
    assert rec.outcome == pytest.approx(target_in_to_move(final.to_move, 0.30))
    _r, _w, sign = split_samples_with_guardrail([s], False)
    assert list(sign) == [1.0 if final.to_move == "black" else -1.0]


def test_continuation_guardrail_reconstructs_odd_depth(tmp_path):
    # 1 move from black root -> red to move (odd depth): sign flips to -1
    case, final = _cont_guardrail_case(tmp_path, 1, 0.30)
    s = build_calibration_sample(case, calibration_target=-0.35)
    rec = s.record
    assert rec.ply == 6                                       # 5 + 1
    assert rec.to_move == final.to_move                       # red
    assert rec.outcome == pytest.approx(target_in_to_move(final.to_move, 0.30))
    _r, _w, sign = split_samples_with_guardrail([s], False)
    assert list(sign) == [-1.0]                               # red-to-move continuation


def test_root_guardrail_still_reconstructs_root(tmp_path):
    # blank extra_moves_json -> falls through to the root branch (v12 behavior)
    rp = tmp_path / "game_000001.json"
    replay = legal_replay(9, game_idx=1)
    rp.write_text(json.dumps(replay))
    root = _root_state(replay)
    case = {
        "game_idx": "1", "case_id": "game_000001_ply_005__guardrail",
        "replay_path": str(rp), "position_ply": "5", "side_to_move": "black",
        "tag": "goal_line_guardrail_retention", "loss_mode": GUARDRAIL_LOSS_MODE,
        "target_black_value": "0.20", "teacher_value": "0.20",
        "teacher_policy_json": "", "root_visits_json": "", "extra_moves_json": "",
    }
    s = build_calibration_sample(case, calibration_target=-0.35)
    rec = s.record
    assert rec.ply == 5                                       # no extra moves
    assert rec.to_move == root.to_move                        # black (root side)
    assert rec.outcome == pytest.approx(target_in_to_move("black", 0.20))   # +0.20


def test_continuation_guardrail_bad_sha1_fails_loud(tmp_path):
    case, _ = _cont_guardrail_case(tmp_path, 2, 0.30)
    case["continuation_legal_moves_sha1"] = "deadbeef"
    with pytest.raises(ValueError, match="sha1"):
        build_calibration_sample(case, calibration_target=-0.35)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_asymmetric_guardrail_continuation_loader.py -v`
Expected: the two continuation-reconstruction tests FAIL (current loader ignores `extra_moves_json` for a guardrail row → `rec.ply == 5`, not 6/7) and `test_continuation_guardrail_bad_sha1_fails_loud` FAILS (DID NOT RAISE — `_apply_extra_moves` is never called). `test_root_guardrail_still_reconstructs_root` already passes (v12 behavior).

- [ ] **Step 3: Implement the gate**

In `build_calibration_position`, locate the existing extra-moves block:

```python
    record_ply = position_ply
    if loss_mode == CONTINUATION_LOSS_MODE:
        state, n_extra = _apply_extra_moves(state, case)
        record_ply = position_ply + n_extra
```

Replace it with (add the guardrail-continuation condition):

```python
    record_ply = position_ply
    is_guardrail_continuation = (
        loss_mode == GUARDRAIL_LOSS_MODE
        and case.get("extra_moves_json") not in (None, ""))
    if loss_mode == CONTINUATION_LOSS_MODE or is_guardrail_continuation:
        state, n_extra = _apply_extra_moves(state, case)
        record_ply = position_ply + n_extra
```

(No other change. A guardrail row with blank `extra_moves_json` keeps `is_guardrail_continuation == False` and falls through to the existing root/default branch — byte-identical v12. `_apply_extra_moves` already fail-loud-verifies `continuation_side_to_move` + `continuation_legal_moves_sha1`. The guardrail row then reaches the default return branch, whose `outcome = target_in_to_move(state.to_move, _resolve_target_black(case, ...))` now uses the *continuation* `state.to_move`.)

- [ ] **Step 4: Run the new tests + loader regression**

Run: `.venv/bin/python -m pytest tests/test_asymmetric_guardrail_continuation_loader.py tests/test_asymmetric_guardrail_pool.py tests/test_calibration_pool_continuation.py tests/test_calibration_pool.py -v`
Expected: ALL PASS (the new tests green; v12 pool tests and continuation tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/calibration_pool.py tests/test_asymmetric_guardrail_continuation_loader.py
git commit -m "feat(calibration): v12b loader gate — guardrail rows with extra_moves_json reconstruct the continuation state

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: v12b continuation guardrail manifest builder

**Files:**
- Create: `scripts/GPU/alphazero/build_v12b_continuation_guardrail_manifest.py`
- Test: `tests/test_build_v12b_continuation_guardrail_manifest.py` (create)

**Interfaces:**
- Consumes: the v7 manifest columns; the Task 1 loader (continuation guardrail rows must reconstruct through it).
- Produces: `make_root_guardrail_clone(parent) -> dict`, `make_continuation_guardrail_clone(parent) -> dict`, `ROOT_TO_GUARDRAIL_TAG`, `CONTINUATION_TO_GUARDRAIL_TAG`, and `main()` writing `logs/eval/targeted_calibration_v12b_continuation_guardrail_from_calib020_0001.csv`. Task 3's smoke loads that manifest.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_v12b_continuation_guardrail_manifest.py`:

```python
import csv

from scripts.GPU.alphazero.build_v12b_continuation_guardrail_manifest import (
    make_root_guardrail_clone, make_continuation_guardrail_clone,
    ROOT_TO_GUARDRAIL_TAG, CONTINUATION_TO_GUARDRAIL_TAG)


def _root_parent(side, tv):
    return {"case_id": "game_1_ply_9", "tag": "goal_line_retention",
            "loss_mode": "mcts_root_retention", "side_to_move": side,
            "teacher_value": tv, "target_black_value": "",
            "root_visits_json": "[0.5,0.5]", "root_legal_moves_sha1": "abc",
            "root_black_value": "0.83", "teacher_policy_json": "",
            "extra_moves_json": "", "continuation_side_to_move": ""}


def _cont_parent(cont_side, tv, depth="2"):
    return {"case_id": "game_1_ply_9__cont_pv2",
            "tag": "old_post_opening_continuation_retention",
            "loss_mode": "searched_continuation_retention", "side_to_move": "black",
            "teacher_value": tv, "target_black_value": "",
            "extra_moves_json": '[{"row":3,"col":4},{"row":5,"col":6}]',
            "continuation_side_to_move": cont_side,
            "continuation_legal_moves_sha1": "abc123", "continuation_depth": depth,
            "continuation_parent_case_id": "game_1_ply_9", "continuation_source": "pv",
            "continuation_path_moves": "d4 f6", "continuation_tree_visits": "400",
            "continuation_tree_nn_value": "0.31",
            "teacher_policy_json": "", "root_visits_json": ""}


def test_root_clone_uses_root_side_and_blanks_continuation():
    b = make_root_guardrail_clone(_root_parent("black", "0.30"))
    assert b["loss_mode"] == "asymmetric_guardrail_retention"
    assert b["tag"] == "goal_line_guardrail_retention"
    assert b["case_id"] == "game_1_ply_9__guardrail"
    assert float(b["target_black_value"]) == 0.30              # black root: +tv
    r = make_root_guardrail_clone(_root_parent("red", "-0.97"))
    assert float(r["target_black_value"]) == 0.97             # red root: -tv
    for row in (b, r):
        assert row["extra_moves_json"] == ""
        assert row["continuation_side_to_move"] == ""
        assert row["root_black_value"] == ""
        assert row["teacher_policy_json"] == ""


def test_continuation_clone_uses_continuation_side_and_preserves_reconstruction():
    # continuation side red (odd depth) -> sign -1 -> target_black = tv * -1
    c = make_continuation_guardrail_clone(_cont_parent("red", "-0.40", depth="1"))
    assert c["loss_mode"] == "asymmetric_guardrail_retention"
    assert c["tag"] == "old_post_opening_continuation_guardrail_retention"
    assert c["case_id"] == "game_1_ply_9__cont_pv2__guardrail"
    assert float(c["target_black_value"]) == 0.40            # -0.40 * -1
    # continuation side black (even depth) -> sign +1
    c2 = make_continuation_guardrail_clone(_cont_parent("black", "0.22", depth="2"))
    assert float(c2["target_black_value"]) == 0.22           # 0.22 * +1
    for row in (c, c2):
        # reconstruction/identity fields PRESERVED
        assert row["extra_moves_json"] == '[{"row":3,"col":4},{"row":5,"col":6}]'
        assert row["continuation_legal_moves_sha1"] == "abc123"
        assert row["continuation_source"] == "pv"
        assert row["continuation_depth"] in ("1", "2")
        # policy/root/search-scalar fields BLANKED
        assert row["teacher_policy_json"] == ""
        assert row["root_visits_json"] == ""
        assert row["continuation_tree_visits"] == ""
        assert row["continuation_tree_nn_value"] == ""
        # teacher_value kept as provenance
        assert row["teacher_value"] in ("-0.40", "0.22")


def test_tag_maps():
    assert ROOT_TO_GUARDRAIL_TAG == {
        "goal_line_retention": "goal_line_guardrail_retention",
        "old_post_opening_retention": "old_post_opening_guardrail_retention",
        "red_predrop_retention": "red_predrop_guardrail_retention"}
    assert CONTINUATION_TO_GUARDRAIL_TAG == {
        "old_post_opening_continuation_retention":
            "old_post_opening_continuation_guardrail_retention",
        "red_predrop_continuation_retention":
            "red_predrop_continuation_guardrail_retention"}


def _write_csv(tmp_path, rows):
    fields = sorted({k for r in rows for k in r})
    p = tmp_path / "v7.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return p


def test_main_routes_and_drops(tmp_path, monkeypatch):
    import scripts.GPU.alphazero.build_v12b_continuation_guardrail_manifest as m
    rows = [
        {"case_id": "a", "tag": "black_predrop_correction", "loss_mode": "hard_value",
         "target_black_value": "-0.35", "teacher_value": "", "side_to_move": "black"},
        {"case_id": "sev", "tag": "red_predrop_severe_root_correction",
         "loss_mode": "hard_value", "target_black_value": "-0.35",
         "teacher_value": "", "side_to_move": "red"},
        {"case_id": "broot", "tag": "goal_line_retention",
         "loss_mode": "mcts_root_retention", "teacher_value": "0.2",
         "side_to_move": "black"},
        {"case_id": "ccont", "tag": "old_post_opening_continuation_retention",
         "loss_mode": "searched_continuation_retention", "teacher_value": "0.1",
         "side_to_move": "black", "continuation_side_to_move": "red",
         "extra_moves_json": '[{"row":1,"col":1}]',
         "continuation_legal_moves_sha1": "x"},
        {"case_id": "glcont", "tag": "goal_line_continuation_retention",   # DROPPED
         "loss_mode": "searched_continuation_retention", "teacher_value": "0.1",
         "side_to_move": "black", "continuation_side_to_move": "red"},
        {"case_id": "drv", "tag": "red_predrop_root_value_retention",      # DROPPED
         "loss_mode": "searched_continuation_retention", "teacher_value": "0.4",
         "side_to_move": "black", "continuation_side_to_move": "black",
         "extra_moves_json": ""},
    ]
    inp = _write_csv(tmp_path, rows)
    outp = tmp_path / "v12b.csv"
    monkeypatch.setattr("sys.argv", ["prog", "--input", str(inp), "--output", str(outp)])
    m.main()
    with outp.open(newline="") as f:
        out = list(csv.DictReader(f))
    assert sorted(r["tag"] for r in out) == sorted([
        "black_predrop_correction", "red_predrop_severe_root_correction",
        "goal_line_guardrail_retention",
        "old_post_opening_continuation_guardrail_retention"])
    cc = [r for r in out
          if r["tag"] == "old_post_opening_continuation_guardrail_retention"][0]
    assert cc["extra_moves_json"] == '[{"row":1,"col":1}]'
    assert cc["loss_mode"] == "asymmetric_guardrail_retention"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_build_v12b_continuation_guardrail_manifest.py -v`
Expected: `ModuleNotFoundError` (builder not created yet).

- [ ] **Step 3: Implement `scripts/GPU/alphazero/build_v12b_continuation_guardrail_manifest.py`**

```python
#!/usr/bin/env python3
"""v12b: build the continuation guardrail manifest.

Output = every hard_value row from the v7 manifest (A correction + severe-D,
kept) + one value-only guardrail clone per B/C/D ROOT retention row (sign from
side_to_move, continuation fields blanked) + one value-only guardrail clone per
C/D CONTINUATION retention row (sign from continuation_side_to_move, continuation
reconstruction fields PRESERVED so the loader rebuilds the searched state). Pure
copy + arithmetic — the BASE anchor already lives in each parent's teacher_value.

Dropped (not cloned, not kept): the source root/continuation rows themselves,
goal_line_continuation_retention (B stays root-only), and
red_predrop_root_value_retention (D root already covered by the red_predrop root
guardrail)."""
import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT_TO_GUARDRAIL_TAG = {
    "goal_line_retention": "goal_line_guardrail_retention",
    "old_post_opening_retention": "old_post_opening_guardrail_retention",
    "red_predrop_retention": "red_predrop_guardrail_retention",
}
CONTINUATION_TO_GUARDRAIL_TAG = {
    "old_post_opening_continuation_retention":
        "old_post_opening_continuation_guardrail_retention",
    "red_predrop_continuation_retention":
        "red_predrop_continuation_guardrail_retention",
}

# policy/root/search-metadata columns blanked on EVERY guardrail clone (value-only)
_POLICY_ROOT_BLANK = (
    "teacher_policy_json", "teacher_legal_moves_sha1",
    "root_visits_json", "root_legal_moves_sha1",
    "root_value_stm", "root_black_value", "root_sims", "root_base_checkpoint",
    "root_seed", "root_mcts_eval_batch_size", "root_mcts_stall_flush_sims",
    "continuation_tree_visits", "continuation_tree_nn_value",
)
# continuation reconstruction/identity columns — blanked on ROOT clones only
_CONTINUATION_COLS = (
    "extra_moves_json", "continuation_side_to_move", "continuation_legal_moves_sha1",
    "continuation_depth", "continuation_parent_case_id", "continuation_source",
    "continuation_path_moves",
)


def _blank(row: dict, cols) -> None:
    for col in cols:
        if col in row:
            row[col] = ""


def _clone_base(parent: dict, new_tag: str, sign: float) -> dict:
    if parent.get("teacher_value") in (None, ""):
        raise ValueError(f"{parent.get('case_id')}: parent lacks teacher_value")
    row = dict(parent)
    row["case_id"] = f"{parent['case_id']}__guardrail"
    row["tag"] = new_tag
    row["loss_mode"] = "asymmetric_guardrail_retention"
    row["target_black_value"] = repr(float(parent["teacher_value"]) * sign)
    _blank(row, _POLICY_ROOT_BLANK)
    return row


def make_root_guardrail_clone(parent: dict) -> dict:
    tag = parent["tag"]
    if tag not in ROOT_TO_GUARDRAIL_TAG:
        raise ValueError(f"{parent.get('case_id')}: not a B/C/D root tag: {tag}")
    sign = 1.0 if parent["side_to_move"] == "black" else -1.0
    row = _clone_base(parent, ROOT_TO_GUARDRAIL_TAG[tag], sign)
    _blank(row, _CONTINUATION_COLS)          # root clones carry no continuation state
    return row


def make_continuation_guardrail_clone(parent: dict) -> dict:
    tag = parent["tag"]
    if tag not in CONTINUATION_TO_GUARDRAIL_TAG:
        raise ValueError(f"{parent.get('case_id')}: not a C/D continuation tag: {tag}")
    side = parent.get("continuation_side_to_move")
    if side in (None, ""):
        raise ValueError(
            f"{parent.get('case_id')}: continuation row lacks continuation_side_to_move")
    sign = 1.0 if side == "black" else -1.0    # CONTINUATION side, not root side
    # _clone_base blanks policy/root cols but leaves _CONTINUATION_COLS intact
    return _clone_base(parent, CONTINUATION_TO_GUARDRAIL_TAG[tag], sign)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    default="logs/eval/targeted_calibration_v7_severe_d_root_correction_from_calib020_0001.csv")
    ap.add_argument("--output",
                    default="logs/eval/targeted_calibration_v12b_continuation_guardrail_from_calib020_0001.csv")
    args = ap.parse_args()
    with Path(args.input).open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    kept = [r for r in rows if (r.get("loss_mode") or "hard_value") == "hard_value"]
    root_clones, cont_clones = [], []
    for r in rows:
        mode = r.get("loss_mode") or "hard_value"
        tag = r.get("tag")
        if mode == "mcts_root_retention" and tag in ROOT_TO_GUARDRAIL_TAG:
            root_clones.append(make_root_guardrail_clone(r))
        elif (mode == "searched_continuation_retention"
              and tag in CONTINUATION_TO_GUARDRAIL_TAG):
            cont_clones.append(make_continuation_guardrail_clone(r))
        # else dropped: source root/cont rows, goal_line_continuation, red_predrop_root_value
    out_rows = kept + root_clones + cont_clones
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    counts = Counter(r["tag"] for r in out_rows)
    print(f"Wrote {args.output}: {len(out_rows)} rows "
          f"({len(kept)} hard_value kept, {len(root_clones)} root guardrail, "
          f"{len(cont_clones)} continuation guardrail)")
    for t in sorted(list(ROOT_TO_GUARDRAIL_TAG.values())
                    + list(CONTINUATION_TO_GUARDRAIL_TAG.values())):
        print(f"  {t}: {counts.get(t, 0)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests + build if data present**

Run: `.venv/bin/python -m pytest tests/test_build_v12b_continuation_guardrail_manifest.py -v` (ALL PASS).
Then, if the v7 manifest is present locally: `.venv/bin/python scripts/GPU/alphazero/build_v12b_continuation_guardrail_manifest.py` and confirm root-guardrail counts 18/30/30 + nonzero continuation-guardrail counts, then that it loads: `.venv/bin/python -c "from scripts.GPU.alphazero.calibration_pool import CalibrationPool; p=CalibrationPool.from_manifest('logs/eval/targeted_calibration_v12b_continuation_guardrail_from_calib020_0001.csv', -0.35); print('schema', p.schema, 'tags', sorted(p.tag_counts()))"`. If the v7 CSV is absent (fresh worktree), NOTE it — the committed unit tests are the gate; the real build/load defers to the operator box. Do NOT commit any CSV.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/build_v12b_continuation_guardrail_manifest.py tests/test_build_v12b_continuation_guardrail_manifest.py
git commit -m "feat(calibration): v12b continuation guardrail manifest builder — root + C/D continuation clones

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: gate-0 smoke + full-suite verification + merge handoff (controller-run)

**Files:**
- Create: `scripts/GPU/alphazero/smoke_v12b_continuation_guardrail.py`

- [ ] **Step 1: Write the smoke** (mirrors `smoke_asymmetric_guardrail_v12.py`; adds a continuation-row-drawn assertion)

```python
#!/usr/bin/env python3
"""Gate-0 smoke: load the v12b continuation-guardrail manifest, draw the v12b
schedule, run a train_step with the guardrail sign, and assert the guardrail
telemetry engaged and at least one continuation-guardrail row was drawn."""
import sys
import numpy as np
import mlx.optimizers as optim

from scripts.GPU.alphazero.calibration_pool import (
    CalibrationPool, split_samples_with_guardrail, GUARDRAIL_LOSS_MODE)
from scripts.GPU.alphazero.trainer import (
    MainModule, freeze_batchnorm_running_stats, train_step)
from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord

MANIFEST = "logs/eval/targeted_calibration_v12b_continuation_guardrail_from_calib020_0001.csv"
SCHEDULE = {"black_predrop_correction": 2, "goal_line_guardrail_retention": 1,
            "old_post_opening_guardrail_retention": 1,
            "old_post_opening_continuation_guardrail_retention": 2,
            "red_predrop_guardrail_retention": 1,
            "red_predrop_continuation_guardrail_retention": 2}
CONTINUATION_TAGS = {"old_post_opening_continuation_guardrail_retention",
                     "red_predrop_continuation_guardrail_retention"}


def main() -> int:
    pool = CalibrationPool.from_manifest(MANIFEST, calibration_target=-0.35)
    assert pool.schema == GUARDRAIL_LOSS_MODE, pool.schema
    import random
    rng = random.Random(0)
    samples = pool.sample_by_tag(SCHEDULE, rng)
    assert any(s.tag in CONTINUATION_TAGS for s in samples), "no continuation guardrail row drawn"
    records, weights, sign = split_samples_with_guardrail(samples, pool.has_weight_scale)
    assert set(np.unique(sign)) <= {-1.0, 0.0, 1.0}
    assert (np.abs(sign) > 0).sum() > 0, "no guardrail rows drawn"
    net = create_network(hidden=64, n_blocks=2)
    freeze_batchnorm_running_stats(net)
    mm = MainModule(net.encoder, net.policy_head)
    om, ov = optim.Adam(learning_rate=1e-3), optim.Adam(learning_rate=1e-3)
    def _p():
        return PositionRecord(board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
                              to_move="red", legal_moves=[(0, 0), (1, 1)],
                              visit_counts=[1, 1], outcome=1.0, active_size=24,
                              ply=0, game_n_moves=10)
    out = train_step(network=net, main_module=mm, opt_main=om, opt_value=ov,
                     batch=[_p() for _ in range(4)], calibration_positions=records,
                     calibration_weights=weights, calibration_loss_weight=0.01,
                     calibration_guardrail_sign=sign, guardrail_margin=0.10,
                     train_value_head_and_final_block=True)
    assert len(out) == 13, len(out)
    print(f"SMOKE PASS: guardrail_hinge_loss={out[10]:.4g} active_frac={out[11]:.3g} "
          f"guardrail_n={out[12]} (schema={pool.schema})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Verify it imports (`.venv/bin/python -c "import scripts.GPU.alphazero.smoke_v12b_continuation_guardrail"`); RUN it only if the v12b manifest + local replays are present, else note the deferral to the operator box. Commit:
```bash
git add scripts/GPU/alphazero/smoke_v12b_continuation_guardrail.py
git commit -m "test(calibration): v12b gate-0 continuation guardrail smoke

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 2: Full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected in the worktree: baseline-passed + the new v12b tests, with EXACTLY the known 14 failed + 6 errors (missing gitignored data). Authoritative check (1346 + new v12b tests, 0 failures) on merged main before push.

- [ ] **Step 3: Hand off to merge**

FF-merge to main, authoritative suite on merged main, push (superpowers:finishing-a-development-branch). STOP after push — the operator run (build the v12b manifest, train with the v12b schedule + `--guardrail-margin 0.10` + `--freeze-batchnorm-stats --train-value-head-and-final-block`, verifier exit 0, gates A/B/C/D) is the USER's; the exact command block is in the spec's Operator-run section.

---

## Operator run (USER's, after merge) — from the spec

1. Build: `.venv/bin/python scripts/GPU/alphazero/build_v12b_continuation_guardrail_manifest.py` (root counts 18/30/30 + the two continuation-guardrail counts reported).
2. Train: the v9/v12 command + `--post-opening-calibration-manifest logs/eval/targeted_calibration_v12b_continuation_guardrail_from_calib020_0001.csv`, schedule `black_predrop_correction=2,goal_line_guardrail_retention=1,old_post_opening_guardrail_retention=1,old_post_opening_continuation_guardrail_retention=2,red_predrop_guardrail_retention=1,red_predrop_continuation_guardrail_retention=2`, `--guardrail-margin 0.10`, `--freeze-batchnorm-stats --train-value-head-and-final-block`, new checkpoint dir.
3. Telemetry: `guardrail_hinge_loss`/`guardrail_active_frac` present, `guardrail_margin=0.10`, `guardrail_n_drawn_by_tag` nonzero for all five scheduled guardrail tags plus `black_predrop_correction` drawn separately, `n_teacher_retention_drawn=0`, `calib_policy_ce=0`.
4. `verify_value_head_and_final_block_checkpoint.py` exit 0.
5. Gates A/B/C/D vs `calib020_0001`, `OUT=logs/eval/v12b_continuation_guardrail_from_calib020_0001_gates_400s`. No promotion unless all four pass.
```
