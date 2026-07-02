# Raw-NN Position-Row Diagnostic (`eval_raw_nn_position_rows.py`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a raw-NN-only (no-MCTS) diagnostic CLI that scores fixed calibration/probe positions across checkpoints and quantifies per-position **value drift from the teacher (BASE)** plus the top-1 policy move, so we can tell whether v4's raw value head still matched the teacher on the shared severe C/D gate rows before MCTS ran.

**Architecture:** One new, self-contained script `scripts/GPU/alphazero/eval_raw_nn_position_rows.py` that **only imports** existing, verified helpers (board reconstruction, raw NN forward, checkpoint loading, thresholds). It touches no manifest, checkpoint, or training path. Core = a pure per-row scorer (`score_row`) + a two-pass teacher/delta resolver (`resolve_deltas`) + a manifest union/filter loader + a thin argparse `main` with an injectable evaluator factory (mirrors `eval_position_probe.main`). Output is one CSV row per `(checkpoint, case)`.

**Tech Stack:** Python 3, MLX (only inside the checkpoint-loading factory), numpy, `csv`/`json`/`argparse`. Tests use `pytest` with a `_FakeEval` stand-in + the `legal_replay` fixture; a single test exercises a real `create_network(hidden=64, n_blocks=2)` forward.

## Global Constraints

- **Diagnostic only.** Do **not** modify any existing module, manifest, checkpoint, gate script, or training behavior. The new script may only `import` existing code. (Acceptance criterion #5.)
- **No MCTS, no search.** Raw NN forward only, via the shared `_teacher_infer`. A guard test asserts the module never references MCTS.
- **Eval-mode BatchNorm.** Load every checkpoint and call `network.eval()` before wrapping in `LocalGPUEvaluator` — running stats, batch-independent, matches the v4 teacher cache and the calibration training forward (mirror of `build_teacher_calibration_manifest.py:79-86`).
- **Perspective contract (get exactly right):**
  - `raw_value_stm`, `teacher_value`, `value_delta_vs_teacher` = **side-to-move** perspective. Delta is computed in stm-space: `raw_value_stm − teacher_value`, **no flip**.
  - `raw_black_value`, `overvalue`, `severe_overvalue`, `target_black_value` = **black** perspective. Flip: `black = v if side=='black' else -v` (mirror of `eval_position_probe.py:86-91` and the involution `calibration_pool.target_in_to_move`).
  - Thresholds (import, do not redefine): `OVERVALUE_THRESHOLD = 0.25`, `SEVERE_OVERVALUE_THRESHOLD = 0.50` from `position_probe_cases`.
- **No `--batch-size` for v1.** Per-position scoring; eval-mode is batch-independent so per-position is correct. Total load is tiny (3 checkpoints × 5 cases = 15 forwards) — far below the Metal-handle limit that forces batching in `probe_eval`.
- Tests run with `.venv/bin/python -m pytest`.

---

## Grounding findings & key decisions (READ FIRST — verified against current code 2026-07-01)

These deviate from the task's literal draft. They are the result of verifying every reuse target and every named artifact against the live repo. **Flag for reviewer:** decisions **D1** and **D2** change the CLI inputs and add two columns relative to the original sketch.

### Verified reuse targets (exact, current)

| Symbol | Location | Contract as used here |
|---|---|---|
| `load_csv_manifest(path)` | `position_probe_cases.py:24` | Returns `{"cases": [...]}`. Each case = `dict(row)` (all CSV columns preserved as strings) **plus** `game_idx:int`, `position_ply:int`, defaulted `case_id`. **Requires** non-empty `game_idx, case_id, replay_path, position_ply, side_to_move` (`REQUIRED_CASE_KEYS`), else raises. |
| `OVERVALUE_THRESHOLD` / `SEVERE_OVERVALUE_THRESHOLD` | `position_probe_cases.py:18-19` | `0.25` / `0.50`. Import; do not redefine. |
| `position_state(replay, position_ply, side_to_move)` | `goal_line_trigger_probe_cases.py:73` | Applies `moves[0:position_ply]` to a fresh `TwixtState(active_size=replay["board_size"], to_move="red", max_plies_limit=replay["n_moves"])`. **Raises `ValueError`** on out-of-range ply or when reconstructed `to_move != side_to_move`. (This is the canonical reconstructor — NOT `probe_eval._replay_probe`, which is the move_history/forced-probe format.) |
| `_teacher_infer(state, evaluator)` | `build_teacher_calibration_manifest.py:28` | Single-position RAW forward, **no MCTS**. Returns `(legal, policy_list, value_stm)`: `legal`=list of `(r,c)`, `policy_list`=floats aligned to `legal`, `value_stm`=`float`. Uses `evaluator.build_input_tensor` + `evaluator.infer`. |
| `load_network_for_scoring(path)` | `probe_eval.py:99` | Returns `(network, in_channels, hidden, n_blocks)`; auto-detects 24/30-channel checkpoints. Then call `network.eval()`. |
| `LocalGPUEvaluator(network)` | `local_evaluator.py:35` | `compile=False` default (correct here). `build_input_tensor(state)`→`(C,H,W)`; `infer(...)`→`(priors (B,M), values (B,))`, values in **stm** perspective. |
| `create_network(hidden=128, n_blocks=6, in_channels=None)` | `network.py:635` | Test uses `create_network(hidden=64, n_blocks=2)` (default 30-channel). |
| `target_in_to_move(side, v)` | `calibration_pool.py:32` | Confirms the flip is an involution: black as-is, red negated. |
| `legal_replay(n, *, board_size=24, game_idx=0, ...)` | `tests/goal_line_probe_fixtures.py:10` | Deterministic legal replay dict (`moves`/`board_size`/`n_moves`). Red starts ⇒ after N moves: N even → red to move, N odd → black. So ply 5 → black, ply 4 → red. |

### D1 — The five focus rows are **gate** positions, not training-manifest rows (INPUT CORRECTION)

The task's first command points `--manifest` at the v4 **training** manifest `logs/eval/targeted_calibration_v4_teacher_from_calib020_0001.csv` and filters the 5 focus `case_id`s. **Verified: that manifest (128 rows) contains only 1 of the 5** (`red_loss_game_000362_predrop_ply_52_drop_54`, a `red_predrop_retention` row). The other four are **not** in it.

The 5 focus rows are the **shared severe C/D gate cases** from the overlap doc (`docs/2026-06-26-...-v3f-v4-overlap-updated.md` §"v4/v3-frozenBN severe-overlap follow-up"). Their real, on-disk homes (both **gitignored** local artifacts — invisible to a default `rg`; use `--no-ignore`):

| case_id | home manifest | ply | side | replay exists |
|---|---|---|---|---|
| `game_000065_ply_021` | `logs/eval/tvc_v3_gate_C_old_post_opening_manifest.csv` | 21 | black | ✅ |
| `game_000369_ply_051` | `logs/eval/tvc_v3_gate_C_old_post_opening_manifest.csv` | 51 | black | ✅ |
| `game_000619_ply_061` | `logs/eval/tvc_v3_gate_C_old_post_opening_manifest.csv` | 61 | black | ✅ |
| `game_000505_ply_037` | `logs/eval/tvc_v3_gate_C_old_post_opening_manifest.csv` | 37 | black | ✅ |
| `red_loss_game_000362_predrop_ply_52_drop_54` | `logs/eval/tvc_v2_gate_D_red_predrop_manifest.csv` | 52 | red | ✅ |

Both gate manifests share this schema (superset of `REQUIRED_CASE_KEYS`, **no `teacher_value` column**):
`case_rank, tag, source, source_rank, target_black_value, weight_scale, game_idx, case_id, replay_path, position_ply, side_to_move, anchor_checkpoint, drop_ply, largest_drop_phase, collapse_type`

**Decision:** `--manifest` is **repeatable** (like `--checkpoint`); the CLI scores the **union** of rows across manifests (dedup by `(case_id, replay_path, position_ply, side_to_move)`), then `--case-id` filters. The corrected run points at both gate manifests. This also means the CLI works unchanged if later pointed at the training manifest.

### D2 — Teacher reference = the **BASE** checkpoint's own raw value (not a manifest column)

The task sketch computed `value_delta_vs_teacher = raw_value_stm − float(case["teacher_value"])`. But the gate manifests have **no `teacher_value` column**, so that would be blank for exactly the rows we care about. The v4 **teacher checkpoint is the anchor `calib020_0001` = BASE** (`build_teacher_calibration_manifest --teacher-checkpoint`; overlap doc line 128; memory anchor). So the teacher raw value on any position is simply **BASE's own raw NN value**, which the CLI computes anyway when it scores BASE.

**Decision:** resolve the teacher per row as: **manifest `teacher_value` if present and non-empty, else the BASE checkpoint's `raw_value_stm` for that same case.** Emit a `teacher_value_source ∈ {manifest, base_checkpoint}` column for transparency, plus the passthrough black-perspective `target_black_value`. `value_delta_vs_teacher = raw_value_stm − teacher_value` (stm, no flip). For the focus rows this yields `value_delta_vs_teacher = candidate_raw − BASE_raw` = **drift-from-teacher**, which is exactly what the decision hinges on. `--base-checkpoint` selects the reference (defaults to the first `--checkpoint`).

This gives a **built-in acceptance anchor**: BASE's `raw_value_stm` must reproduce the doc's recorded teacher raw values (below), since BASE *is* the teacher.

### D3 — Is a new CLI even needed? (task question #1)

**For pure severe-case _overlap_ (which cases fail in both v4 and v3F): No new code needed.** `eval_position_probe` already writes per-case `position_probe_cases.csv` with `case_id + probe_black_root_value + severe_black_overvalue` per checkpoint; the overlap is a diff of the v4 and v3F gate CSVs, and it is **already computed** — the artifact `logs/eval/v3f_v4_severe_overlap_review.csv` and the doc's overlap section list the shared rows (4 C, 1 D). Those CSVs carry only the **MCTS** root value.

**What the raw-NN CLI adds (why it is justified):**
1. **Raw NN value with NO search** — isolates value-head drift from MCTS-root drift. The gate CSVs cannot tell you whether v4's *raw* value head still matched the teacher; that is the exact fork in the overlap doc (§"Current next hypothesis").
2. **`value_delta_vs_teacher`** (candidate raw − BASE/teacher raw, stm) — a per-position, quantified drift-from-teacher number.
3. **`top1_move` + `top1_prob`** — a value-drift-vs-policy-drift lens the MCTS gate CSVs don't provide.

So: overlap → already answered from gate CSVs; **raw-vs-teacher + no-search value + top-1 → this CLI.** Scope the CLI to exactly that added value.

### Decision interpretation (what the output feeds, from overlap doc §139-142)

Recorded teacher (=BASE) raw stm values to reproduce as validation anchors:

| case_id | side | doc teacher raw (stm) | ⇒ raw_black | base MCTS (black) | v4 MCTS (black) |
|---|---|---|---|---|---|
| `game_000065_ply_021` | black | **+0.1105** | +0.1105 | +0.480 | +0.758 |
| `game_000369_ply_051` | black | **−0.1389** | −0.1389 | +0.334 | +0.765 |
| `game_000505_ply_037` | black | **+0.9455** | +0.9455 | +0.856 | — (diagnostic only; already pro-black) |
| `red_loss_game_000362_...` | red | **−0.9379** | +0.9379 | +0.198 | +0.582 (v3F +0.677) |
| `game_000619_ply_061` | black | (not recorded) | — | — | — |

- If v4 `raw_value_stm` ≈ BASE (small `|value_delta_vs_teacher|`) on the C rows but the gate MCTS drifted severe → **raw retention held, MCTS-root drifted** → next branch = MCTS-root / root-behavior retention (not another raw-teacher weight/schedule tweak).
- If v4 `raw_value_stm` drifted from BASE (large `|delta|`) → raw teacher-retention did not hold the model → inspect loss weighting / sampling / gradient before a new objective.
- `game_000505_ply_037` is diagnostic-only (BASE already +0.9455 / +0.856) — not a clean retention failure.

---

## File Structure

- **Create:** `scripts/GPU/alphazero/eval_raw_nn_position_rows.py` — the entire diagnostic. One responsibility: raw-NN scoring of fixed rows across checkpoints + teacher-delta resolution + CSV emit. Imports only; mutates nothing.
- **Create:** `tests/test_eval_raw_nn_position_rows.py` — unit + end-to-end tests (flat `tests/` dir, matching every sibling test).
- **Modify:** none.

Module surface (names are fixed here and referenced identically across all tasks):

```
OVERVALUE_THRESHOLD, SEVERE_OVERVALUE_THRESHOLD   # imported from position_probe_cases
PASSTHROUGH_COLUMNS: tuple[str, ...]
OUTPUT_COLUMNS: tuple[str, ...]

to_black(value_stm: float, side_to_move: str) -> float
_format_move(move: tuple[int, int]) -> str                     # "r:c"
score_row(evaluator, case: dict) -> dict                       # raw_* + top1_* + over/severe
score_all(cases, checkpoints, evaluator_factory) -> list[dict] # + checkpoint / checkpoint_label + passthrough
resolve_deltas(rows: list[dict], base_checkpoint: str) -> None # fills teacher_value / source / delta (in place)
load_and_filter_cases(manifest_paths, case_ids=None, tags=None, limit=None) -> list[dict]
checkpoint_label(path: str) -> str                             # Path(path).parent.name (display only)
build_evaluator(checkpoint_path: str)                          # real factory: load -> eval() -> LocalGPUEvaluator
write_rows(out_path: str, rows: list[dict]) -> None
parse_args(argv=None)
main(argv=None, evaluator_factory=None) -> int
```

`OUTPUT_COLUMNS` (exact order the CSV is written in):

```
checkpoint, checkpoint_label, case_id, tag, source, source_rank,
side_to_move, position_ply, replay_path, loss_mode, target_black_value,
teacher_value, teacher_value_source, raw_value_stm, raw_black_value,
value_delta_vs_teacher, abs_value_delta_vs_teacher,
top1_move, top1_prob, overvalue, severe_overvalue, teacher_legal_moves_sha1
```

`PASSTHROUGH_COLUMNS` (copied verbatim from each case; missing → `""`):

```
case_id, tag, source, source_rank, side_to_move, position_ply,
replay_path, loss_mode, target_black_value, teacher_value, teacher_legal_moves_sha1
```

> **Design note — why delta keys on the checkpoint _path_, not the label:** `checkpoint_label` is `Path(ckpt).parent.name` (nice distinct display labels for the three real checkpoints). It can collide when checkpoints share a parent dir (e.g. tmp in tests). `resolve_deltas` therefore matches BASE by the exact `checkpoint` **path**, never the label — collision-free.

---

### Task 1: Perspective helper + single-row raw scorer

**Files:**
- Create: `scripts/GPU/alphazero/eval_raw_nn_position_rows.py`
- Test: `tests/test_eval_raw_nn_position_rows.py`

**Interfaces:**
- Consumes: `position_state` (`goal_line_trigger_probe_cases.py:73`), `_teacher_infer` (`build_teacher_calibration_manifest.py:28`), `OVERVALUE_THRESHOLD`/`SEVERE_OVERVALUE_THRESHOLD` (`position_probe_cases.py:18-19`).
- Produces: `to_black(value_stm, side_to_move) -> float`; `_format_move((r,c)) -> "r:c"`; `score_row(evaluator, case) -> dict` with keys `raw_value_stm, raw_black_value, top1_move, top1_prob, overvalue, severe_overvalue`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_raw_nn_position_rows.py`:

```python
import csv
import json

import numpy as np
import pytest

from scripts.GPU.alphazero import eval_raw_nn_position_rows as R
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
from tests.goal_line_probe_fixtures import legal_replay


class _FakeEval:
    """Deterministic stand-in for LocalGPUEvaluator: uniform priors + fixed value. No MCTS."""

    def __init__(self, value=0.2):
        self._value = value

    def build_input_tensor(self, state):
        return state.to_tensor()

    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        b, m = move_mask.shape
        priors = move_mask / np.maximum(move_mask.sum(axis=1, keepdims=True), 1.0)
        values = np.full((b,), self._value, dtype=np.float32)
        return priors.astype(np.float32), values


def _replay_file(tmp_path, n=9, game_idx=1):
    rp = tmp_path / f"game_{game_idx:06d}.json"
    rp.write_text(json.dumps(legal_replay(n, game_idx=game_idx)))
    return rp


def _case(rp, case_id, ply, side, **extra):
    base = {
        "game_idx": "1", "case_id": case_id, "replay_path": str(rp),
        "position_ply": str(ply), "side_to_move": side,
    }
    base.update(extra)
    return base


def test_to_black_flips_red_to_move():
    assert R.to_black(0.7, "black") == pytest.approx(0.7)
    assert R.to_black(0.7, "red") == pytest.approx(-0.7)
    with pytest.raises(ValueError):
        R.to_black(0.1, "green")


def test_score_row_red_to_move_flips_black_value(tmp_path):
    rp = _replay_file(tmp_path)
    row = R.score_row(_FakeEval(value=0.2), _case(rp, "red1", 4, "red"))  # 4 moves -> red to move
    assert row["raw_value_stm"] == pytest.approx(0.2)
    assert row["raw_black_value"] == pytest.approx(-0.2)     # red-to-move: black = -stm
    assert row["overvalue"] is False and row["severe_overvalue"] is False


def test_score_row_black_overvalue_flags_and_top1(tmp_path):
    rp = _replay_file(tmp_path)
    row = R.score_row(_FakeEval(value=0.6), _case(rp, "b1", 5, "black"))  # 5 moves -> black
    assert row["raw_black_value"] == pytest.approx(0.6)
    assert row["overvalue"] is True and row["severe_overvalue"] is True   # 0.6 >= 0.50
    assert ":" in row["top1_move"] and 0.0 < row["top1_prob"] <= 1.0


def test_raw_nn_rows_scores_with_local_evaluator(tmp_path):
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator

    rp = _replay_file(tmp_path)
    net = create_network(hidden=64, n_blocks=2)
    net.eval()
    ev = LocalGPUEvaluator(net)
    row = R.score_row(ev, _case(rp, "ret1", 5, "black"))
    # score_row must apply NO transform to the stm value beyond the shared infer wrapper.
    state = position_state(legal_replay(9, game_idx=1), 5, "black")
    _, _, value = R._teacher_infer(state, ev)   # _teacher_infer wraps evaluator.infer (the "direct infer")
    assert row["raw_value_stm"] == pytest.approx(value, abs=1e-6)
    assert row["raw_black_value"] == pytest.approx(value, abs=1e-6)   # black-to-move: no flip


def test_score_row_side_to_move_mismatch_raises(tmp_path):
    rp = _replay_file(tmp_path)
    with pytest.raises(ValueError, match="side_to_move"):
        R.score_row(_FakeEval(), _case(rp, "bad", 4, "black"))   # ply 4 -> red; claims black
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_raw_nn_position_rows.py -q`
Expected: FAIL — `ModuleNotFoundError` / `AttributeError: module ... has no attribute 'to_black'`.

- [ ] **Step 3: Write the minimal implementation**

Create `scripts/GPU/alphazero/eval_raw_nn_position_rows.py`:

```python
"""Raw NN-only diagnostic scorer for fixed calibration/probe positions.

DIAGNOSTIC ONLY. Imports existing helpers; changes no manifest, checkpoint, or
training path. NO MCTS: it runs the shared single-position raw forward
(_teacher_infer) in eval-mode BatchNorm across one or more checkpoints and
reports per-position value drift from the teacher (the BASE checkpoint) plus the
top-1 policy move.

Answers: on the shared severe C/D gate rows, did the candidate raw network still
match the teacher, or did it drift before MCTS? See
docs/superpowers/plans/2026-07-01-eval-raw-nn-position-rows-diagnostic.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .position_probe_cases import (
    OVERVALUE_THRESHOLD,
    SEVERE_OVERVALUE_THRESHOLD,
    load_csv_manifest,
)
from .goal_line_trigger_probe_cases import position_state
from .build_teacher_calibration_manifest import _teacher_infer


def to_black(value_stm: float, side_to_move: str) -> float:
    """Express a side-to-move value in the black perspective.

    Involution matching calibration_pool.target_in_to_move and
    eval_position_probe.py:86-91: black as-is, red negated.
    """
    if side_to_move == "black":
        return float(value_stm)
    if side_to_move == "red":
        return float(-value_stm)
    raise ValueError(f"unexpected side_to_move {side_to_move!r}")


def _format_move(move) -> str:
    r, c = move
    return f"{r}:{c}"


def score_row(evaluator, case: dict) -> dict:
    """Raw NN score of one reconstructed position. NN-only (no MCTS)."""
    replay = json.loads(Path(case["replay_path"]).read_text())
    state = position_state(replay, int(float(case["position_ply"])), case["side_to_move"])
    legal, policy, value_stm = _teacher_infer(state, evaluator)
    raw_black = to_black(value_stm, case["side_to_move"])
    if legal:
        i = max(range(len(policy)), key=lambda j: policy[j])
        top1_move = _format_move(legal[i])
        top1_prob = policy[i]
    else:                                     # non-terminal in practice; guard against empty
        top1_move, top1_prob = "", ""
    return {
        "raw_value_stm": value_stm,
        "raw_black_value": raw_black,
        "top1_move": top1_move,
        "top1_prob": top1_prob,
        "overvalue": raw_black >= OVERVALUE_THRESHOLD,
        "severe_overvalue": raw_black >= SEVERE_OVERVALUE_THRESHOLD,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_raw_nn_position_rows.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_raw_nn_position_rows.py tests/test_eval_raw_nn_position_rows.py
git commit -m "feat(diagnostic): raw-NN per-position scorer (to_black + score_row)"
```

---

### Task 2: Teacher/base resolution + deltas

**Files:**
- Modify: `scripts/GPU/alphazero/eval_raw_nn_position_rows.py`
- Test: `tests/test_eval_raw_nn_position_rows.py`

**Interfaces:**
- Consumes: scored rows (each has `checkpoint`, `case_id`, `raw_value_stm`, and a raw passthrough `teacher_value` string that may be `""`).
- Produces: `resolve_deltas(rows, base_checkpoint) -> None` — mutates each row in place to set `teacher_value` (resolved float, or `""`), `teacher_value_source ∈ {"manifest","base_checkpoint"}`, `value_delta_vs_teacher` (float or `""`), `abs_value_delta_vs_teacher` (float or `""`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_raw_nn_position_rows.py`:

```python
def test_delta_vs_teacher_uses_manifest_value_stm_perspective():
    rows = [
        {"checkpoint": "/ck/base.st", "case_id": "x", "raw_value_stm": 0.10, "teacher_value": "-0.50"},
        {"checkpoint": "/ck/cand.st", "case_id": "x", "raw_value_stm": 0.30, "teacher_value": "-0.50"},
    ]
    R.resolve_deltas(rows, base_checkpoint="/ck/base.st")
    cand = next(r for r in rows if r["checkpoint"] == "/ck/cand.st")
    assert cand["teacher_value_source"] == "manifest"
    assert cand["teacher_value"] == pytest.approx(-0.50)
    assert cand["value_delta_vs_teacher"] == pytest.approx(0.30 - (-0.50))  # stm - stm, NO flip
    assert cand["abs_value_delta_vs_teacher"] == pytest.approx(0.80)


def test_delta_vs_teacher_falls_back_to_base_when_no_manifest_value():
    rows = [
        {"checkpoint": "/ck/base.st", "case_id": "y", "raw_value_stm": 0.11, "teacher_value": ""},
        {"checkpoint": "/ck/v4.st",   "case_id": "y", "raw_value_stm": 0.42, "teacher_value": ""},
    ]
    R.resolve_deltas(rows, base_checkpoint="/ck/base.st")
    base = next(r for r in rows if r["checkpoint"] == "/ck/base.st")
    v4 = next(r for r in rows if r["checkpoint"] == "/ck/v4.st")
    assert v4["teacher_value_source"] == "base_checkpoint"
    assert v4["teacher_value"] == pytest.approx(0.11)             # BASE raw is the teacher
    assert v4["value_delta_vs_teacher"] == pytest.approx(0.31)    # 0.42 - 0.11
    assert base["value_delta_vs_teacher"] == pytest.approx(0.0)   # base vs itself
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_raw_nn_position_rows.py -q -k "delta_vs_teacher"`
Expected: FAIL — `AttributeError: module ... has no attribute 'resolve_deltas'`.

- [ ] **Step 3: Write the minimal implementation**

Add to `scripts/GPU/alphazero/eval_raw_nn_position_rows.py`:

```python
def resolve_deltas(rows: list, base_checkpoint: str) -> None:
    """Second pass: resolve each row's teacher_value and delta, in place.

    teacher_value = manifest teacher_value if present/non-empty, else the BASE
    checkpoint's raw_value_stm for the same case (BASE = the v4 teacher/anchor).
    Delta is computed in side-to-move space (raw_value_stm - teacher_value); NO flip.
    """
    base_raw = {
        r["case_id"]: r["raw_value_stm"]
        for r in rows
        if r["checkpoint"] == base_checkpoint
    }
    for r in rows:
        manifest_tv = r.get("teacher_value", "")
        if manifest_tv not in (None, ""):
            tv, source = float(manifest_tv), "manifest"
        else:
            tv, source = base_raw.get(r["case_id"]), "base_checkpoint"
        r["teacher_value"] = "" if tv is None else tv
        r["teacher_value_source"] = source
        if tv is None:
            r["value_delta_vs_teacher"] = ""
            r["abs_value_delta_vs_teacher"] = ""
        else:
            delta = r["raw_value_stm"] - tv
            r["value_delta_vs_teacher"] = delta
            r["abs_value_delta_vs_teacher"] = abs(delta)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_raw_nn_position_rows.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_raw_nn_position_rows.py tests/test_eval_raw_nn_position_rows.py
git commit -m "feat(diagnostic): resolve teacher_value (manifest-or-base) + stm delta"
```

---

### Task 3: Manifest union + filter loader

**Files:**
- Modify: `scripts/GPU/alphazero/eval_raw_nn_position_rows.py`
- Test: `tests/test_eval_raw_nn_position_rows.py`

**Interfaces:**
- Consumes: `load_csv_manifest` (`position_probe_cases.py:24`).
- Produces: `load_and_filter_cases(manifest_paths, case_ids=None, tags=None, limit=None) -> list[dict]` — union across manifests, dedup by `(case_id, replay_path, position_ply, side_to_move)`, filter by `case_ids`/`tags`, cap at `limit`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_raw_nn_position_rows.py`:

```python
def _write_manifest(path, cases):
    cols = [
        "game_idx", "case_id", "replay_path", "position_ply", "side_to_move",
        "tag", "target_black_value", "teacher_value", "source", "source_rank",
        "loss_mode", "teacher_legal_moves_sha1",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for c in cases:
            w.writerow(c)


def test_case_id_filter(tmp_path):
    rp = _replay_file(tmp_path)
    man = tmp_path / "m.csv"
    _write_manifest(man, [_case(rp, "keep", 5, "black"), _case(rp, "drop", 5, "black")])
    cases = R.load_and_filter_cases([str(man)], case_ids={"keep"})
    assert [c["case_id"] for c in cases] == ["keep"]


def test_union_across_manifests_and_dedup(tmp_path):
    rp = _replay_file(tmp_path)
    m1, m2 = tmp_path / "m1.csv", tmp_path / "m2.csv"
    _write_manifest(m1, [_case(rp, "a", 5, "black")])
    _write_manifest(m2, [_case(rp, "a", 5, "black"), _case(rp, "b", 4, "red")])  # 'a' duplicated
    cases = R.load_and_filter_cases([str(m1), str(m2)])
    assert sorted(c["case_id"] for c in cases) == ["a", "b"]   # dedup 'a'


def test_tag_and_limit_filters(tmp_path):
    rp = _replay_file(tmp_path)
    man = tmp_path / "m.csv"
    _write_manifest(man, [
        _case(rp, "c1", 5, "black", tag="old_post_opening_retention"),
        _case(rp, "c2", 5, "black", tag="goal_line_retention"),
    ])
    assert [c["case_id"] for c in R.load_and_filter_cases([str(man)], tags={"goal_line_retention"})] == ["c2"]
    assert len(R.load_and_filter_cases([str(man)], limit=1)) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_raw_nn_position_rows.py -q -k "filter or union"`
Expected: FAIL — `AttributeError: ... 'load_and_filter_cases'`.

- [ ] **Step 3: Write the minimal implementation**

Add to `scripts/GPU/alphazero/eval_raw_nn_position_rows.py`:

```python
def load_and_filter_cases(manifest_paths, case_ids=None, tags=None, limit=None) -> list:
    """Union rows across manifests; dedup; filter by case_id/tag; cap at limit."""
    seen, cases = set(), []
    for path in manifest_paths:
        for case in load_csv_manifest(path)["cases"]:
            cid = case["case_id"]
            if case_ids and cid not in case_ids:
                continue
            if tags and case.get("tag", "") not in tags:
                continue
            key = (cid, case.get("replay_path"), case["position_ply"], case["side_to_move"])
            if key in seen:
                continue
            seen.add(key)
            cases.append(case)
            if limit and len(cases) >= limit:
                return cases
    return cases
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_raw_nn_position_rows.py -q`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_raw_nn_position_rows.py tests/test_eval_raw_nn_position_rows.py
git commit -m "feat(diagnostic): manifest union + case-id/tag/limit filter loader"
```

---

### Task 4: CLI (`main`) + evaluator factory + CSV writer + no-MCTS guard

**Files:**
- Modify: `scripts/GPU/alphazero/eval_raw_nn_position_rows.py`
- Test: `tests/test_eval_raw_nn_position_rows.py`

**Interfaces:**
- Consumes: everything above, plus (lazily, inside `build_evaluator` only) `load_network_for_scoring` (`probe_eval.py:99`) and `LocalGPUEvaluator` (`local_evaluator.py:35`).
- Produces: `checkpoint_label(path)`, `score_all(cases, checkpoints, evaluator_factory)`, `build_evaluator(path)`, `write_rows(out_path, rows)`, `parse_args(argv)`, `main(argv=None, evaluator_factory=None) -> int`. `main` mirrors `eval_position_probe.main`'s injectable-factory pattern so tests never load a real checkpoint.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_raw_nn_position_rows.py`:

```python
def test_main_end_to_end_with_fake_factory(tmp_path):
    rp = _replay_file(tmp_path)
    man = tmp_path / "m.csv"
    _write_manifest(man, [
        _case(rp, "b1", 5, "black", teacher_value="-0.50"),  # manifest teacher present
        _case(rp, "r1", 4, "red"),                           # no teacher -> base fallback
    ])
    base = tmp_path / "base.safetensors"; base.write_text("x")
    cand = tmp_path / "cand.safetensors"; cand.write_text("x")
    out = tmp_path / "out.csv"

    rc = R.main(
        ["--manifest", str(man),
         "--checkpoint", str(base), "--checkpoint", str(cand),
         "--out", str(out)],
        evaluator_factory=lambda ckpt: _FakeEval(value=0.2 if "base" in ckpt else 0.4),
    )
    assert rc == 0

    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4                                     # 2 checkpoints x 2 cases
    for col in ["raw_value_stm", "raw_black_value", "value_delta_vs_teacher",
                "teacher_value_source", "top1_move", "top1_prob",
                "overvalue", "severe_overvalue"]:
        assert col in rows[0]

    b1_cand = next(r for r in rows if r["case_id"] == "b1" and r["checkpoint"] == str(cand))
    assert b1_cand["teacher_value_source"] == "manifest"
    assert float(b1_cand["value_delta_vs_teacher"]) == pytest.approx(0.4 - (-0.5))  # 0.9

    r1_cand = next(r for r in rows if r["case_id"] == "r1" and r["checkpoint"] == str(cand))
    assert r1_cand["teacher_value_source"] == "base_checkpoint"
    assert float(r1_cand["value_delta_vs_teacher"]) == pytest.approx(0.4 - 0.2)     # stm-space, 0.2
    assert float(r1_cand["raw_black_value"]) == pytest.approx(-0.4)                 # red -> flip


def test_case_id_filter_reflected_in_output(tmp_path):
    rp = _replay_file(tmp_path)
    man = tmp_path / "m.csv"
    _write_manifest(man, [_case(rp, "keep", 5, "black"), _case(rp, "drop", 5, "black")])
    base = tmp_path / "base.safetensors"; base.write_text("x")
    out = tmp_path / "out.csv"
    rc = R.main(
        ["--manifest", str(man), "--checkpoint", str(base),
         "--case-id", "keep", "--out", str(out)],
        evaluator_factory=lambda ckpt: _FakeEval(),
    )
    assert rc == 0
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert {r["case_id"] for r in rows} == {"keep"}


def test_main_side_to_move_mismatch_raises(tmp_path):
    rp = _replay_file(tmp_path)
    man = tmp_path / "m.csv"
    _write_manifest(man, [_case(rp, "bad", 4, "black")])      # ply 4 -> red; claims black
    base = tmp_path / "base.safetensors"; base.write_text("x")
    out = tmp_path / "o.csv"
    with pytest.raises(ValueError, match="side_to_move"):
        R.main(["--manifest", str(man), "--checkpoint", str(base), "--out", str(out)],
               evaluator_factory=lambda ckpt: _FakeEval())


def test_module_does_not_import_mcts():
    import importlib
    mod = importlib.import_module("scripts.GPU.alphazero.eval_raw_nn_position_rows")
    src = open(mod.__file__).read()
    assert "from .mcts" not in src and "MCTS(" not in src
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_raw_nn_position_rows.py -q -k "main or import_mcts"`
Expected: FAIL — `AttributeError: ... 'main'`.

- [ ] **Step 3: Write the minimal implementation**

Add to `scripts/GPU/alphazero/eval_raw_nn_position_rows.py`:

```python
PASSTHROUGH_COLUMNS = (
    "case_id", "tag", "source", "source_rank", "side_to_move", "position_ply",
    "replay_path", "loss_mode", "target_black_value", "teacher_value",
    "teacher_legal_moves_sha1",
)

OUTPUT_COLUMNS = (
    "checkpoint", "checkpoint_label", "case_id", "tag", "source", "source_rank",
    "side_to_move", "position_ply", "replay_path", "loss_mode", "target_black_value",
    "teacher_value", "teacher_value_source", "raw_value_stm", "raw_black_value",
    "value_delta_vs_teacher", "abs_value_delta_vs_teacher",
    "top1_move", "top1_prob", "overvalue", "severe_overvalue", "teacher_legal_moves_sha1",
)


def checkpoint_label(path: str) -> str:
    """Display label (parent dir name). Cosmetic only — deltas key on the path."""
    return Path(path).parent.name


def score_all(cases, checkpoints, evaluator_factory) -> list:
    """Score every (checkpoint, case): passthrough columns + raw NN scores."""
    rows = []
    for ckpt in checkpoints:
        label = checkpoint_label(ckpt)
        evaluator = evaluator_factory(ckpt)
        for case in cases:
            row = {k: case.get(k, "") for k in PASSTHROUGH_COLUMNS}
            row["checkpoint"] = ckpt
            row["checkpoint_label"] = label
            row.update(score_row(evaluator, case))
            rows.append(row)
    return rows


def build_evaluator(checkpoint_path: str):
    """Real factory: load checkpoint, eval-mode BatchNorm, wrap in LocalGPUEvaluator."""
    from .local_evaluator import LocalGPUEvaluator
    from .probe_eval import load_network_for_scoring

    network, *_ = load_network_for_scoring(checkpoint_path)
    network.eval()                              # running stats, batch-independent (matches teacher cache)
    return LocalGPUEvaluator(network)


def write_rows(out_path: str, rows: list) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(OUTPUT_COLUMNS), extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Raw NN-only diagnostic scorer for fixed calibration/probe positions (no MCTS)."
    )
    p.add_argument("--manifest", action="append", default=[], required=True, dest="manifests",
                   metavar="PATH", help="position manifest CSV (repeatable; rows unioned).")
    p.add_argument("--checkpoint", action="append", default=[], required=True, dest="checkpoints",
                   metavar="PATH")
    p.add_argument("--base-checkpoint", default=None, metavar="PATH",
                   help="teacher reference (default: first --checkpoint). Rows without a manifest "
                        "teacher_value use this checkpoint's raw value as the teacher.")
    p.add_argument("--case-id", action="append", default=[], dest="case_ids", metavar="CASE_ID")
    p.add_argument("--tag", action="append", default=[], dest="tags", metavar="TAG")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", required=True, metavar="PATH")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None, evaluator_factory=None) -> int:
    args = parse_args(argv)
    factory = evaluator_factory or build_evaluator

    for ckpt in args.checkpoints:
        if not Path(ckpt).exists():
            print(f"error: checkpoint not found: {ckpt}", file=sys.stderr)
            return 2
    for man in args.manifests:
        if not Path(man).exists():
            print(f"error: manifest not found: {man}", file=sys.stderr)
            return 2

    base_ckpt = args.base_checkpoint or args.checkpoints[0]
    checkpoints = list(args.checkpoints)
    if base_ckpt not in checkpoints:            # ensure base is scored for the teacher fallback
        checkpoints = [base_ckpt] + checkpoints

    cases = load_and_filter_cases(
        args.manifests,
        set(args.case_ids) or None,
        set(args.tags) or None,
        args.limit,
    )
    if not cases:
        print("error: no cases matched filters", file=sys.stderr)
        return 2

    rows = score_all(cases, checkpoints, factory)
    resolve_deltas(rows, base_ckpt)
    write_rows(args.out, rows)

    print(f"wrote {len(rows)} rows ({len(checkpoints)} checkpoints x {len(cases)} cases) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the full test file to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_raw_nn_position_rows.py -q`
Expected: PASS (15 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_raw_nn_position_rows.py tests/test_eval_raw_nn_position_rows.py
git commit -m "feat(diagnostic): eval_raw_nn_position_rows CLI (union manifests, base-teacher deltas, CSV)"
```

---

## Post-implementation validation (operator run — not a code task)

**1. Confirm no existing behavior changed** (acceptance #5):

```bash
git diff --stat HEAD~4 -- scripts/GPU/alphazero/ | grep -v eval_raw_nn_position_rows || echo "only the new file touched"
.venv/bin/python -m pytest tests/test_eval_raw_nn_position_rows.py -q
```

**2. Corrected first command** (points at the two **gate** manifests that actually hold the focus rows — see decision D1; base defaults to the first checkpoint, `--base-checkpoint` shown for clarity):

```bash
.venv/bin/python -m scripts.GPU.alphazero.eval_raw_nn_position_rows \
  --manifest logs/eval/tvc_v3_gate_C_old_post_opening_manifest.csv \
  --manifest logs/eval/tvc_v2_gate_D_red_predrop_manifest.csv \
  --checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --checkpoint checkpoints/alphazero-v4-teacher-from-calib020-0001/model_iter_0001.safetensors \
  --checkpoint checkpoints/alphazero-v3-frozenBN-control-from-calib020-0001/model_iter_0001.safetensors \
  --base-checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --case-id game_000065_ply_021 \
  --case-id game_000369_ply_051 \
  --case-id game_000619_ply_061 \
  --case-id game_000505_ply_037 \
  --case-id red_loss_game_000362_predrop_ply_52_drop_54 \
  --out logs/eval/v3f_v4_raw_nn_focus_rows.csv
```

Expected: `wrote 15 rows (3 checkpoints x 5 cases) -> logs/eval/v3f_v4_raw_nn_focus_rows.csv`.

**3. Summarize** (adjusted to the real columns):

```bash
.venv/bin/python - <<'PY'
import csv
from pathlib import Path
rows = list(csv.DictReader(Path("logs/eval/v3f_v4_raw_nn_focus_rows.csv").open(newline="")))
cols = ["checkpoint_label", "case_id", "tag", "side_to_move",
        "teacher_value", "teacher_value_source", "raw_value_stm",
        "value_delta_vs_teacher", "raw_black_value", "overvalue",
        "severe_overvalue", "top1_move", "top1_prob"]
for r in rows:
    print("\n" + "=" * 120)
    for c in cols:
        if c in r:
            print(f"{c}: {r[c]}")
PY
```

**4. Validation anchors — approximate sanity checks, NOT exact-equality gates.** BASE = teacher, so BASE `raw_value_stm` should land *near* the doc's recorded teacher raw values. Treat these as ballpark checks only — they depend on the exact checkpoint path, eval-mode reconstruction, and small numeric differences in the raw forward, so do **not** assert exact equality or a fixed tolerance. What matters is agreement in sign and rough magnitude (e.g. the D row clearly negative-stm / strongly black; the C rows near zero-to-mild), not the third decimal.

| case_id | BASE `raw_value_stm` ≈ | ⇒ `raw_black_value` ≈ |
|---|---|---|
| `game_000065_ply_021` | +0.1105 | +0.1105 |
| `game_000369_ply_051` | −0.1389 | −0.1389 |
| `game_000505_ply_037` | +0.9455 | +0.9455 |
| `red_loss_game_000362_...` | −0.9379 | +0.9379 |

If a BASE row is *materially* off an anchor (wrong sign, or off by ≫0.1), stop and reconcile (wrong base checkpoint, wrong reconstruction, or BN mode) before trusting the V4/V3F deltas. A small numeric offset is expected and fine.

---

## Acceptance criteria

1. `.venv/bin/python -m pytest tests/test_eval_raw_nn_position_rows.py` passes (15 tests).
2. The corrected first command runs and emits **15 rows** (BASE, V4, V3F × 5 focus cases) to `logs/eval/v3f_v4_raw_nn_focus_rows.csv`.
3. Output includes `raw_value_stm`, `raw_black_value`, `teacher_value` (+ `teacher_value_source`), `value_delta_vs_teacher` (+ `abs_`), `top1_move`, `top1_prob`, `overvalue`, `severe_overvalue`.
4. BASE `raw_value_stm` lands *near* the four recorded teacher anchors (approximate sanity check — sign + rough magnitude, not exact equality); the V4/V3F `value_delta_vs_teacher` on the C rows then reads as "raw retention held vs raw drifted" (feeds the doc's next-branch decision).
5. No existing module, manifest, checkpoint, or training path is modified (only the new script + test file); the module never references MCTS (guarded by `test_module_does_not_import_mcts`).

---

## Self-Review

**Spec coverage:** CLI + inputs (Task 4 / D1 repeatable `--manifest`), reuse of `load_csv_manifest` (T3), `position_state` (T1), `_teacher_infer` (T1), `load_network_for_scoring` (T4 `build_evaluator`), `LocalGPUEvaluator` (T4). Perspective flip + delta (T1/T2). Output columns (T4 `OUTPUT_COLUMNS`). "New CLI needed?" (D3). Acceptance criteria + TDD test list (all five requested tests present: `test_raw_nn_rows_scores_with_local_evaluator`, `test_to_black_flips_red_to_move`, `test_delta_vs_teacher_*`, `test_case_id_filter`, mismatch-raises — plus dedup/tag/limit/end-to-end/no-MCTS guards). Task question #1 answered in D3.

**Placeholder scan:** none — every step carries the real test or implementation code and an exact command with expected output.

**Type/name consistency:** `resolve_deltas` keys on `checkpoint` (path) in both T2 and T4; `checkpoint_label` is display-only. `teacher_value` is a passthrough string in T1/T4 that `resolve_deltas` overwrites with a resolved float in T2. `PASSTHROUGH_COLUMNS` ⊆ `OUTPUT_COLUMNS`; `score_row` keys ⊆ `OUTPUT_COLUMNS`; `resolve_deltas` outputs (`teacher_value_source`, `value_delta_vs_teacher`, `abs_value_delta_vs_teacher`) ⊆ `OUTPUT_COLUMNS`. `_FakeEval.infer` signature matches `LocalGPUEvaluator.infer`. Focus case `side_to_move` (4×black, 1×red) matches the reconstructed plies verified on disk.

**Reviewer decision — RESOLVED (approved 2026-07-01):** D1 (repeatable `--manifest` at the two **gate** manifests) and D2 (teacher-reference = the **BASE checkpoint**, with manifest-`teacher_value` passthrough as a fallback source) are approved as written. Amendment applied: the BASE-raw validation anchors are **approximate** sanity checks (sign + rough magnitude), not exact-equality/fixed-tolerance gates.

---

## Kickoff prompt for a fresh session

Paste this into a new session to execute this plan:

> Execute the approved, pre-reviewed implementation plan at `docs/superpowers/plans/2026-07-01-eval-raw-nn-position-rows-diagnostic.md`. It is fully specified and approved — do **not** re-plan or re-brainstorm; implement it task-by-task.
>
> Use **superpowers:subagent-driven-development** (fresh subagent per task, two-stage review between tasks). *(If inline is preferred: superpowers:executing-plans.)*
>
> It builds a NEW, read-only diagnostic CLI `scripts/GPU/alphazero/eval_raw_nn_position_rows.py` + `tests/test_eval_raw_nn_position_rows.py` that scores raw NN (no-MCTS) value + top-1 policy across checkpoints for fixed calibration/probe rows, to measure per-position drift-from-teacher on the shared severe C/D gate rows.
>
> Non-negotiable constraints (from the plan — enforce in review):
> - **Diagnostic only:** import existing helpers; modify NO existing module, manifest, checkpoint, or training path.
> - **No MCTS** (a guard test asserts it). Reuse `load_csv_manifest`, `position_state`, `_teacher_infer`, `load_network_for_scoring`, `LocalGPUEvaluator`; eval-mode BatchNorm (`network.eval()`).
> - **Perspective:** `value_delta_vs_teacher` in side-to-move space (`raw_stm − teacher_stm`, NO flip); `overvalue`/`severe`/`raw_black_value` in black perspective (`black = v if black else −v`).
> - **D1:** `--manifest` is repeatable. **D2:** teacher ref = the BASE checkpoint's raw value when a manifest `teacher_value` is absent; `value_delta_vs_teacher = candidate_raw_stm − BASE_raw_stm`.
> - **TDD** per task: write failing test → confirm it fails → implement → confirm it passes → commit. Tests: `.venv/bin/python -m pytest tests/test_eval_raw_nn_position_rows.py`.
> - **Git:** FF-merge to main, no force-push (linear-history preference).
>
> **STOP** after all 4 tasks pass and the full test file is green. Do **not** run the real 3-checkpoint GPU diagnostic — that is an operator step (the "Post-implementation validation" section), and its BASE-raw anchors are approximate sanity checks, not equality gates.
>
> For *why* this diagnostic exists (is v4's failure raw value-head drift-from-teacher, or MCTS-root drift?), consult memory `targeted-value-calibration-experiment-ledger`.
