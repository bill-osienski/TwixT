# Post-Opening Sharp-Drop Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional, value-only calibration mechanism that nudges `model_iter_0409` toward danger-recognition on the post-opening sharp-drop position class, training from ~134 held-out replay positions while gating on the frozen 30-case probe.

**Architecture:** A separate calibration pool of fixed external positions (Mechanism B), fed through a standalone value-only MSE auxiliary term folded into `total_loss` inside `alphazero_loss_batch`/`train_step`. Disabled by default ⇒ the main self-play loss path is byte-identical. A deterministic CSV selector builds the training manifest from the analyzer's review queue, excluding the frozen 30 probe games.

**Tech Stack:** Python 3.14, MLX (Apple), pytest. Package path `scripts.GPU.alphazero.*` (tests rely on `conftest.py` adding `scripts/GPU` to `sys.path`).

**Design doc:** `docs/superpowers/specs/2026-06-16-post-opening-sharp-drop-calibration-design.md` (read it first).

## Global Constraints

These apply to every task; copied verbatim from the spec:

- **Value-only.** Calibration positions supervise the value head only — no policy target, no policy loss contribution.
- **Soft target.** Default `calibration_target = -0.50` (black perspective). Never hard −1.
- **Absolute weight.** Default `calibration_weight = 0.02`. The calibration term is added to `total_loss` standalone — **NOT** multiplied by `value_weight`.
- **Batch fraction.** Default `calibration_batch_fraction = 0.10`. Calibration mini-batch size `k = max(1, round(batch_size * batch_fraction))`.
- **Disabled by default, truly byte-identical.** When not enabled: no pool load, no calibration forward pass, the existing 7-tuple loss structure is unchanged. The extended (10-tuple) return appears ONLY when enabled.
- **`total_loss` MUST be the first returned value** of `alphazero_loss_batch` — `nn.value_and_grad()` differentiates only the first element.
- **Frozen-30 are sacred.** The 30 probe `game_idx` values are never added to a training manifest.
- **Deterministic manifest.** Same source queue + same excluded game_idx list ⇒ byte-identical output; stable ordering; stable output path.
- **`calib_mean_value_pred` logged every training iteration** (cheap, no MCTS) — the headline signal that the mechanism is working.
- **Perspective:** the value head outputs side-to-move perspective. Target stored in side-to-move perspective: black ⇒ `target`, red ⇒ `-target`. Current pool is all black-to-move.
- **Encoder:** `state.to_tensor()` returns `(30, 24, 24)` CHW float32; transpose `np.transpose(chw, (1,2,0))` ⇒ `(24, 24, 30)` NHWC for `PositionRecord.board_tensor`.

---

## File Structure

**New files:**
- `scripts/GPU/alphazero/build_calibration_manifest.py` — deterministic train-manifest selector (Task 1).
- `scripts/GPU/alphazero/calibration_pool.py` — `target_in_to_move`, `build_calibration_position`, `CalibrationPool`, `build_post_opening_calibration_block` (Tasks 2 & 4).
- `tests/test_build_calibration_manifest.py` (Task 1)
- `tests/test_calibration_pool.py` (Task 2)
- `tests/test_calibration_loss.py` (Task 3)
- `tests/test_calibration_cli_flags.py` (Task 5)

**Modified files:**
- `scripts/GPU/alphazero/trainer.py` — `alphazero_loss_batch` + `train_step` calibration term (Task 3); `train()` params, pool build, loop wiring, sidecar (Task 4).
- `scripts/GPU/alphazero/train.py` — `--post-opening-calibration-*` flags + threading (Task 5).

Test invocation throughout: `.venv/bin/python -m pytest tests/<file> -v`

---

### Task 1: Train-manifest selector

Pure Python, no MLX. Reads the analyzer's `manual_review_queue.csv`, excludes the frozen probe `game_idx` set, derives the probe-manifest schema (`position_ply = drop_ply - 2`, `case_id = game_{idx:06d}_ply_{ply:03d}`), and writes a deterministic train manifest.

> **Run this only against a WIDE queue** regenerated with `--review-queue 200` (Task 6, Step 2). The existing 50-row queue is truncated and would yield only ~20 train cases after excluding the frozen 30 — far short of the ~134 target.

**Files:**
- Create: `scripts/GPU/alphazero/build_calibration_manifest.py`
- Test: `tests/test_build_calibration_manifest.py`

**Interfaces:**
- Produces:
  - `derive_case(row: dict, case_rank: int) -> dict`
  - `select_calibration_cases(queue_rows: list[dict], holdout_game_idxs: set[int]) -> list[dict]`
  - `load_holdout_game_idxs(frozen_manifest_path) -> set[int]`
  - `write_manifest(cases: list[dict], out_path) -> None`
  - `main(argv=None) -> int`
- Output CSV columns (frozen-probe schema **plus** a `source_rank` traceability column = original review-queue rank): `case_rank,source_rank,game_idx,case_id,replay_path,position_ply,drop_ply,side_to_move,a_color,winner,n_moves,initial_a_value,final_a_value,largest_a_value_drop,largest_drop_phase,collapse_type`. (`case_rank` = position in the holdout-excluded train manifest; `source_rank` = original analyzer rank. Extra columns are preserved-and-ignored by `load_csv_manifest`.)

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_calibration_manifest.py`:

```python
import csv
from scripts.GPU.alphazero.build_calibration_manifest import (
    derive_case, select_calibration_cases, load_holdout_game_idxs, main,
)

QUEUE_COLS = [
    "rank", "game_idx", "task_id", "replay_path", "a_color", "winner",
    "n_moves", "collapse_type", "initial_a_value", "final_a_value",
    "largest_a_value_drop", "largest_drop_ply", "largest_drop_fraction",
    "largest_drop_phase", "first_a_value_below_lost_ply",
    "first_a_value_below_lost_fraction", "mean_top1_share_post",
    "median_selected_visit_rank_post", "opening_key",
]


def _qrow(game_idx, drop_ply=41, collapse="sharp_value_drop",
          phase="post_opening", a_color="black", winner="red", rank=1):
    return {
        "rank": rank, "game_idx": game_idx, "task_id": game_idx,
        "replay_path": f"logs/eval/replays/game_{game_idx:06d}.json",
        "a_color": a_color, "winner": winner, "n_moves": 51,
        "collapse_type": collapse, "initial_a_value": 0.07,
        "final_a_value": -0.95, "largest_a_value_drop": -1.78,
        "largest_drop_ply": drop_ply, "largest_drop_fraction": 0.82,
        "largest_drop_phase": phase, "first_a_value_below_lost_ply": drop_ply,
        "first_a_value_below_lost_fraction": 0.82, "mean_top1_share_post": 0.45,
        "median_selected_visit_rank_post": 1, "opening_key": "r11c9",
    }


def test_derive_case_computes_position_ply_and_case_id():
    case = derive_case(_qrow(637, drop_ply=41), case_rank=1)
    assert case["game_idx"] == 637
    assert case["position_ply"] == 39          # drop_ply - 2
    assert case["drop_ply"] == 41
    assert case["side_to_move"] == "black"
    assert case["case_id"] == "game_000637_ply_039"
    assert case["case_rank"] == 1
    assert case["source_rank"] == 1            # original review-queue rank


def test_select_excludes_holdout_and_nonmatching():
    rows = [
        _qrow(637, rank=1),                                   # keep
        _qrow(277, rank=2),                                   # holdout -> drop
        _qrow(100, rank=3, collapse="gradual_decay"),         # wrong collapse -> drop
        _qrow(101, rank=4, phase="opening"),                  # wrong phase -> drop
        _qrow(102, rank=5, a_color="red"),                    # wrong color -> drop
        _qrow(103, rank=6, winner="black"),                   # wrong winner -> drop
        _qrow(200, rank=7),                                   # keep
    ]
    cases = select_calibration_cases(rows, holdout_game_idxs={277})
    assert [c["game_idx"] for c in cases] == [637, 200]
    assert [c["case_rank"] for c in cases] == [1, 2]          # re-ranked 1..N
    assert [c["source_rank"] for c in cases] == [1, 7]        # original queue ranks


def test_main_writes_manifest_excluding_holdout(tmp_path):
    queue = tmp_path / "queue.csv"
    with queue.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=QUEUE_COLS)
        w.writeheader()
        w.writerow(_qrow(637, rank=1))
        w.writerow(_qrow(277, rank=2))
        w.writerow(_qrow(200, rank=3))
    holdout = tmp_path / "frozen.csv"
    with holdout.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["game_idx"])
        w.writeheader()
        w.writerow({"game_idx": 277})
    out = tmp_path / "train.csv"
    rc = main(["--queue", str(queue), "--holdout-manifest", str(holdout),
               "--out", str(out)])
    assert rc == 0
    rows = list(csv.DictReader(out.open()))
    assert [int(r["game_idx"]) for r in rows] == [637, 200]
    assert load_holdout_game_idxs(holdout) == {277}
    # determinism: re-run yields identical bytes
    first = out.read_bytes()
    main(["--queue", str(queue), "--holdout-manifest", str(holdout),
          "--out", str(out)])
    assert out.read_bytes() == first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_build_calibration_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.GPU.alphazero.build_calibration_manifest'`

- [ ] **Step 3: Write the implementation**

Create `scripts/GPU/alphazero/build_calibration_manifest.py`:

```python
"""Deterministic builder for the post-opening sharp-drop calibration TRAIN
manifest.

Reads the loss-replay analyzer's review-queue CSV, keeps only black-loss
post-opening sharp-drop rows, excludes the frozen probe game_idx set, and
writes a manifest in the same schema the probe loader expects
(position_probe_cases.load_csv_manifest). The frozen 30 probe games are the
EVAL set and must never appear here (see design §4 invariant).

position_ply = drop_ply - 2  (black's decision point, two plies before the
collapse) — matches how the frozen probe manifest was derived.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

OUTPUT_COLUMNS = [
    "case_rank", "source_rank", "game_idx", "case_id", "replay_path",
    "position_ply", "drop_ply", "side_to_move", "a_color", "winner", "n_moves",
    "initial_a_value", "final_a_value", "largest_a_value_drop",
    "largest_drop_phase", "collapse_type",
]


def derive_case(row: dict, case_rank: int) -> dict:
    """Map one review-queue row to a probe-manifest case dict."""
    game_idx = int(row["game_idx"])
    drop_ply = int(float(row["largest_drop_ply"]))
    position_ply = drop_ply - 2
    return {
        "case_rank": case_rank,
        "source_rank": int(row["rank"]),
        "game_idx": game_idx,
        "case_id": f"game_{game_idx:06d}_ply_{position_ply:03d}",
        "replay_path": row["replay_path"],
        "position_ply": position_ply,
        "drop_ply": drop_ply,
        "side_to_move": row["a_color"],
        "a_color": row["a_color"],
        "winner": row["winner"],
        "n_moves": int(float(row["n_moves"])),
        "initial_a_value": row["initial_a_value"],
        "final_a_value": row["final_a_value"],
        "largest_a_value_drop": row["largest_a_value_drop"],
        "largest_drop_phase": row["largest_drop_phase"],
        "collapse_type": row["collapse_type"],
    }


def select_calibration_cases(queue_rows: list, holdout_game_idxs: set) -> list:
    """Filter to black-loss post-opening sharp-drop rows, drop the holdout,
    preserve the analyzer's (rank) order, and re-rank 1..N.

    Rows whose decision point would be negative (drop_ply < 2) are skipped.
    """
    kept = []
    for row in queue_rows:
        if row["collapse_type"] != "sharp_value_drop":
            continue
        if row["largest_drop_phase"] != "post_opening":
            continue
        if row["a_color"] != "black":
            continue
        if row["winner"] != "red":
            continue
        if int(row["game_idx"]) in holdout_game_idxs:
            continue
        if int(float(row["largest_drop_ply"])) - 2 < 0:
            continue
        kept.append(row)
    return [derive_case(r, i + 1) for i, r in enumerate(kept)]


def load_holdout_game_idxs(frozen_manifest_path) -> set:
    """Read the frozen probe manifest CSV; return its set of game_idx ints."""
    with Path(frozen_manifest_path).open(newline="") as f:
        return {int(r["game_idx"]) for r in csv.DictReader(f)}


def write_manifest(cases: list, out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        w.writeheader()
        for case in cases:
            w.writerow({k: case[k] for k in OUTPUT_COLUMNS})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Build the post-opening sharp-drop calibration train manifest."
    )
    p.add_argument("--queue", required=True,
                   help="analyzer manual_review_queue.csv (regenerate wide).")
    p.add_argument("--holdout-manifest", required=True,
                   help="frozen probe manifest CSV whose game_idx column is excluded.")
    p.add_argument("--out", required=True, help="output train manifest CSV path.")
    args = p.parse_args(argv)

    for path in (args.queue, args.holdout_manifest):
        if not Path(path).exists():
            print(f"error: not found: {path}", file=sys.stderr)
            return 2

    with Path(args.queue).open(newline="") as f:
        queue_rows = list(csv.DictReader(f))

    holdout = load_holdout_game_idxs(args.holdout_manifest)
    cases = select_calibration_cases(queue_rows, holdout)
    write_manifest(cases, args.out)
    print(f"wrote {len(cases)} calibration train cases -> {args.out} "
          f"(excluded {len(holdout)} holdout games)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_build_calibration_manifest.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/build_calibration_manifest.py tests/test_build_calibration_manifest.py
git commit -m "feat(calibration): deterministic train-manifest selector"
```

---

### Task 2: CalibrationPool

Reconstructs each manifest case to a board, encodes it, and builds a value-only `PositionRecord` whose `outcome` carries the soft target (in side-to-move perspective).

**Files:**
- Create: `scripts/GPU/alphazero/calibration_pool.py`
- Test: `tests/test_calibration_pool.py`

**Interfaces:**
- Consumes: `position_probe_cases.load_csv_manifest`, `goal_line_trigger_probe_cases.position_state`, `self_play.PositionRecord`, `TwixtState.to_tensor()/legal_moves()`.
- Produces:
  - `target_in_to_move(side_to_move: str, calibration_target: float) -> float`
  - `build_calibration_position(case: dict, calibration_target: float) -> PositionRecord`
  - `class CalibrationPool`: `__init__(records)`, `__len__()`, `sample(k: int, rng) -> list[PositionRecord]`, classmethod `from_manifest(manifest_path, calibration_target) -> CalibrationPool`

- [ ] **Step 1: Write the failing test**

Create `tests/test_calibration_pool.py`:

```python
import csv
import json
import random

import numpy as np
import pytest

from scripts.GPU.alphazero.calibration_pool import (
    target_in_to_move, build_calibration_position, CalibrationPool,
)
from scripts.GPU.alphazero.self_play import PositionRecord
from goal_line_probe_fixtures import legal_replay


def test_target_in_to_move_perspective():
    assert target_in_to_move("black", -0.5) == -0.5
    assert target_in_to_move("red", -0.5) == 0.5
    with pytest.raises(ValueError):
        target_in_to_move("green", -0.5)


def _write_case(tmp_path, game_idx=0, position_ply=5):
    # legal_replay alternates from red; odd ply => black to move.
    assert position_ply % 2 == 1
    replay = legal_replay(position_ply + 3, game_idx=game_idx)
    rpath = tmp_path / f"game_{game_idx:06d}.json"
    rpath.write_text(json.dumps(replay))
    return {
        "game_idx": game_idx,
        "case_id": f"game_{game_idx:06d}_ply_{position_ply:03d}",
        "replay_path": str(rpath),
        "position_ply": position_ply,
        "side_to_move": "black",
    }


def test_build_calibration_position_black(tmp_path):
    case = _write_case(tmp_path, game_idx=1, position_ply=5)
    rec = build_calibration_position(case, calibration_target=-0.5)
    assert isinstance(rec, PositionRecord)
    assert rec.to_move == "black"
    assert rec.outcome == -0.5
    assert rec.active_size == 24
    assert rec.board_tensor.shape == (24, 24, 30)
    assert rec.board_tensor.dtype == np.float32
    assert len(rec.legal_moves) > 0
    assert len(rec.visit_counts) == len(rec.legal_moves)


def test_from_manifest_loads_all_cases(tmp_path):
    manifest = tmp_path / "train.csv"
    cases = [_write_case(tmp_path, game_idx=i, position_ply=5) for i in (1, 2, 3)]
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["game_idx", "case_id", "replay_path",
                           "position_ply", "side_to_move"])
        w.writeheader()
        w.writerows(cases)
    pool = CalibrationPool.from_manifest(str(manifest), calibration_target=-0.5)
    assert len(pool) == 3
    drawn = pool.sample(7, random.Random(0))
    assert len(drawn) == 7
    assert all(r.outcome == -0.5 for r in drawn)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.GPU.alphazero.calibration_pool'`

- [ ] **Step 3: Write the implementation**

Create `scripts/GPU/alphazero/calibration_pool.py`:

```python
"""Post-opening sharp-drop calibration pool (design Mechanism B).

A fixed set of external replay positions where the checkpoint (as black)
overvalued a losing position. Each becomes a value-only training sample whose
target is a soft negative (black perspective). The pool is sampled each train
step; the value-only MSE term is added to total_loss in alphazero_loss_batch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .goal_line_trigger_probe_cases import position_state
from .position_probe_cases import load_csv_manifest
from .self_play import PositionRecord


def target_in_to_move(side_to_move: str, calibration_target: float) -> float:
    """Express the black-perspective target in the side-to-move perspective.

    The value head outputs side-to-move perspective. For black-to-move the
    target is used as-is; for red-to-move it is negated.
    """
    if side_to_move == "black":
        return float(calibration_target)
    if side_to_move == "red":
        return float(-calibration_target)
    raise ValueError(f"unexpected side_to_move {side_to_move!r}")


def build_calibration_position(case: dict, calibration_target: float) -> PositionRecord:
    """Reconstruct a case to a board and build a value-only PositionRecord.

    visit_counts is a zero vector (policy is never supervised here); outcome
    carries the soft target in side-to-move perspective.
    """
    replay_path = Path(case["replay_path"])
    if not replay_path.exists():
        raise FileNotFoundError(
            f"{case.get('case_id')}: replay not found: {replay_path}")
    replay = json.loads(replay_path.read_text())
    position_ply = int(case["position_ply"])
    side = case["side_to_move"]
    state = position_state(replay, position_ply, side)

    board_chw = state.to_tensor()                       # (30, 24, 24) CHW
    board_hwc = np.transpose(board_chw, (1, 2, 0)).astype(np.float32)  # (24,24,30)
    legal = state.legal_moves()

    return PositionRecord(
        board_tensor=board_hwc,
        to_move=state.to_move,
        legal_moves=legal,
        visit_counts=[0] * len(legal),
        outcome=target_in_to_move(state.to_move, calibration_target),
        active_size=state.active_size,
        ply=position_ply,
        game_n_moves=None,
    )


class CalibrationPool:
    """Fixed pool of calibration PositionRecords; sampled with replacement."""

    def __init__(self, records):
        if not records:
            raise ValueError("CalibrationPool requires at least one record")
        self._records = list(records)

    def __len__(self):
        return len(self._records)

    def sample(self, k: int, rng):
        if k <= 0:
            return []
        return [rng.choice(self._records) for _ in range(k)]

    @classmethod
    def from_manifest(cls, manifest_path, calibration_target: float):
        manifest = load_csv_manifest(manifest_path)
        records = [build_calibration_position(c, calibration_target)
                   for c in manifest["cases"]]
        return cls(records)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/calibration_pool.py tests/test_calibration_pool.py
git commit -m "feat(calibration): CalibrationPool + value-only position builder"
```

---

### Task 3: Calibration aux value loss in `alphazero_loss_batch` + `train_step`

Add the optional value-only calibration term to the core loss and the step wrapper. Disabled ⇒ unchanged 7-tuple. Enabled ⇒ 10-tuple appending `(calib_loss, calib_value_mean, calib_n)`.

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` (`alphazero_loss_batch` ends at line 1180; `train_step` lines 1183-1284)
- Test: `tests/test_calibration_loss.py`

**Interfaces:**
- Consumes: `CalibrationPool` records (`PositionRecord` list); `make_padded_batch`; `network.forward_padded`.
- Produces (extended signatures):
  - `alphazero_loss_batch(..., calibration_positions=None, calibration_loss_weight=0.0)` → 7-tuple (disabled) or 10-tuple `(total, policy, value, l2, aux, aux_cov, aux_neli, calib_loss, calib_value_mean, calib_n)` (enabled).
  - `train_step(..., calibration_positions=None, calibration_loss_weight=0.0)` → 7-tuple (disabled) or 10-tuple of floats/int (enabled).

- [ ] **Step 1: Write the failing test**

Create `tests/test_calibration_loss.py`:

```python
import numpy as np
import mlx.core as mx
import mlx.optimizers as optim

from scripts.GPU.alphazero.trainer import (
    alphazero_loss_batch, train_step, MainModule, flatten_params,
)
from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord


def _main_pos():
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move="red", legal_moves=[(0, 0), (1, 1), (2, 2)],
        visit_counts=[10, 5, 3], outcome=1.0, active_size=24,
        ply=0, game_n_moves=10,
    )


def _calib_pos(target=-0.5):
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move="black", legal_moves=[(0, 0), (1, 1)],
        visit_counts=[0, 0], outcome=target, active_size=24,
        ply=20, game_n_moves=None,
    )


def test_disabled_returns_seven_tuple():
    net = create_network(hidden=64, n_blocks=2)
    out = alphazero_loss_batch(net, [_main_pos() for _ in range(3)])
    assert len(out) == 7


def test_zero_weight_is_inert_seven_tuple():
    net = create_network(hidden=64, n_blocks=2)
    pos = [_main_pos() for _ in range(3)]
    base = alphazero_loss_batch(net, pos)
    out = alphazero_loss_batch(net, pos, calibration_positions=[_calib_pos()],
                               calibration_loss_weight=0.0)
    assert len(out) == 7
    np.testing.assert_allclose(float(out[0].item()), float(base[0].item()), atol=1e-6)


def test_enabled_returns_ten_tuple_and_adds_weighted_mse():
    net = create_network(hidden=64, n_blocks=2)
    pos = [_main_pos() for _ in range(3)]
    calib = [_calib_pos(-0.5), _calib_pos(-0.5)]
    base_total = float(alphazero_loss_batch(net, pos)[0].item())
    out = alphazero_loss_batch(net, pos, calibration_positions=calib,
                               calibration_loss_weight=0.02)
    assert len(out) == 10
    total, calib_loss, calib_n = float(out[0].item()), float(out[7].item()), out[9]
    assert calib_n == 2
    np.testing.assert_allclose(total, base_total + 0.02 * calib_loss, atol=1e-5)


def _vh_gnorm(grads):
    return sum(float(mx.sum(mx.abs(p)).item())
               for _, p in flatten_params(grads["value_head"]))


def test_calibration_gradient_reaches_value_head():
    net = create_network(hidden=64, n_blocks=2)
    pos = [_main_pos() for _ in range(3)]
    calib = [_calib_pos(-0.5)]

    def off(m):
        return alphazero_loss_batch(m, pos, value_weight=0.0, l2_weight=0.0)

    def on(m):
        return alphazero_loss_batch(m, pos, value_weight=0.0, l2_weight=0.0,
                                    calibration_positions=calib,
                                    calibration_loss_weight=0.02)

    _, g_off = nn_value_and_grad(net, off)
    _, g_on = nn_value_and_grad(net, on)
    # Essential claim: calibration drives a value-head gradient the disabled
    # path does not. Assert the margin (robust to MLX init/path noise) rather
    # than relying solely on exact-zero for the disabled case.
    assert _vh_gnorm(g_on) > 1e-6
    assert _vh_gnorm(g_on) > _vh_gnorm(g_off)


def nn_value_and_grad(net, fn):
    import mlx.nn as nn
    return nn.value_and_grad(net, fn)(net)


def test_train_step_arity_disabled_and_enabled():
    net = create_network(hidden=64, n_blocks=2)
    mm = MainModule(net.encoder, net.policy_head)
    opt_main = optim.Adam(learning_rate=1e-3)
    opt_value = optim.Adam(learning_rate=1e-4)
    pos = [_main_pos() for _ in range(3)]

    off = train_step(network=net, main_module=mm, opt_main=opt_main,
                     opt_value=opt_value, batch=pos)
    assert len(off) == 7

    on = train_step(network=net, main_module=mm, opt_main=opt_main,
                    opt_value=opt_value, batch=pos,
                    calibration_positions=[_calib_pos()],
                    calibration_loss_weight=0.02)
    assert len(on) == 10
    assert all(np.isfinite(x) for x in on[:9])
    assert on[9] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_loss.py -v`
Expected: FAIL — `alphazero_loss_batch() got an unexpected keyword argument 'calibration_positions'`

- [ ] **Step 3: Modify `alphazero_loss_batch`**

In `scripts/GPU/alphazero/trainer.py`, add two params to the signature (after `conversion_reducer_weight: float = 0.35,` at line 1085):

```python
    conversion_reducer_weight: float = 0.35,         # NEW
    calibration_positions=None,                       # NEW: design Mechanism B
    calibration_loss_weight: float = 0.0,             # NEW
```

Then replace the final loss-assembly block (lines 1174-1180):

```python
    total_loss = (policy_loss
                  + value_weight * value_loss
                  + l2_loss
                  + conversion_loss_weight * aux_loss)

    # Post-opening calibration (value-only, standalone weight, NOT * value_weight)
    calib_active = (
        calibration_loss_weight > 0.0
        and calibration_positions is not None
        and len(calibration_positions) > 0
    )
    if calib_active:
        cb_boards, cb_rows, cb_cols, cb_mask, _cb_pi, cb_targets = make_padded_batch(
            calibration_positions, max_moves_cap=max_moves_cap
        )
        _, cb_values, _ = network.forward_padded(
            cb_boards, cb_rows, cb_cols, cb_mask,
            active_size=calibration_positions[0].active_size,
        )  # value head ignores move arrays; values are side-to-move perspective
        calib_loss = mx.mean((cb_values - cb_targets) ** 2)
        # Per-STEP mean prediction. The sidecar averages these across steps;
        # since k is constant, step-weighted == sample-weighted. Do NOT change
        # the sidecar to divide a raw sum by n_drawn — this is already a mean.
        calib_value_mean = mx.mean(cb_values)
        total_loss = total_loss + calibration_loss_weight * calib_loss
        # CRITICAL: total_loss must be first for nn.value_and_grad()
        return (total_loss, policy_loss, value_loss, l2_loss,
                aux_loss, aux_coverage, aux_n_eligible,
                calib_loss, calib_value_mean, len(calibration_positions))

    # CRITICAL: total_loss must be first for nn.value_and_grad()
    return total_loss, policy_loss, value_loss, l2_loss, aux_loss, aux_coverage, aux_n_eligible
```

- [ ] **Step 4: Modify `train_step`**

In `scripts/GPU/alphazero/trainer.py`, add two params to the `train_step` signature (after `conversion_reducer_weight: float = 0.35,` at line 1198):

```python
    conversion_reducer_weight: float = 0.35,         # NEW
    calibration_positions=None,                       # NEW
    calibration_loss_weight: float = 0.0,             # NEW
```

Replace the `loss_fn` + unpack block (lines 1231-1250) with:

```python
    calib_active = (
        calibration_loss_weight > 0.0
        and calibration_positions is not None
        and len(calibration_positions) > 0
    )

    def loss_fn(model):
        return alphazero_loss_batch(
            model, batch,
            l2_weight=l2_weight,
            value_weight=value_weight,
            max_moves_cap=max_moves_cap,
            active_size=active_size,
            progress_weighted=progress_weighted,
            progress_weight_floor=progress_weight_floor,
            conversion_loss_weight=conversion_loss_weight,
            conversion_completion_weight=conversion_completion_weight,
            conversion_reducer_weight=conversion_reducer_weight,
            calibration_positions=calibration_positions,
            calibration_loss_weight=calibration_loss_weight,
        )

    # value_and_grad differentiates first element (total_loss)
    loss_tuple, grads = nn.value_and_grad(network, loss_fn)(network)

    # Unpack losses (7-tuple, or 10-tuple when calibration active)
    if calib_active:
        (total_loss, policy_loss, value_loss, l2_loss, aux_loss, aux_coverage,
         aux_n_eligible, calib_loss, calib_value_mean, calib_n) = loss_tuple
    else:
        total_loss, policy_loss, value_loss, l2_loss, aux_loss, aux_coverage, aux_n_eligible = loss_tuple
```

Then replace the final `return` block (lines 1276-1284) with:

```python
    if calib_active:
        return (
            float(total_loss.item()),
            float(policy_loss.item()),
            float(value_loss.item()),
            float(l2_loss.item()),
            float(aux_loss.item()),
            float(aux_coverage),
            int(aux_n_eligible),
            float(calib_loss.item()),
            float(calib_value_mean.item()),
            int(calib_n),
        )
    return (
        float(total_loss.item()),
        float(policy_loss.item()),
        float(value_loss.item()),
        float(l2_loss.item()),
        float(aux_loss.item()),
        float(aux_coverage),
        int(aux_n_eligible),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_loss.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Run the existing trainer/loss suite (regression — disabled path unchanged)**

Run: `.venv/bin/python -m pytest tests/test_conversion_loss.py tests/test_replay_buffer_conversion.py -v`
Expected: PASS (no regressions)

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py tests/test_calibration_loss.py
git commit -m "feat(calibration): value-only aux loss in alphazero_loss_batch + train_step"
```

---

### Task 4: Wire calibration into `train()` + diagnostics sidecar

Add `train()` params, build the pool, gate the effective weight, sample + pass a calibration mini-batch each step (disabled branch byte-identical), accumulate stats, and emit a `post_opening_calibration` sidecar block logging `calib_mean_value_pred` every iteration.

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` — `train()` signature (after line 2282); effective-weight gating + pool build (after line 2678); pre-loop accumulator init (near line 2678); per-iteration reset (near line 3650); the step loop (lines 3654-3676); accumulation (near line 3692); sidecar (after line 3786).
- Modify/extend: `scripts/GPU/alphazero/calibration_pool.py` — add `build_post_opening_calibration_block`.
- Test: append to `tests/test_calibration_pool.py`.

**Interfaces:**
- Consumes: `CalibrationPool.from_manifest`, `train_step(..., calibration_positions=, calibration_loss_weight=)`.
- Produces:
  - `train(..., post_opening_calibration_enabled=False, post_opening_calibration_manifest=None, post_opening_calibration_target=-0.50, post_opening_calibration_weight=0.02, post_opening_calibration_batch_fraction=0.10)`
  - `build_post_opening_calibration_block(config: dict, enabled: bool, loss_accumulator: dict) -> dict`

- [ ] **Step 1: Write the failing test for the sidecar block**

Append to `tests/test_calibration_pool.py`:

```python
def test_build_post_opening_calibration_block():
    from scripts.GPU.alphazero.calibration_pool import (
        build_post_opening_calibration_block,
    )
    block = build_post_opening_calibration_block(
        config={"enabled": True, "target": -0.5, "effective_weight": 0.02,
                "pool_size": 134},
        enabled=True,
        loss_accumulator={"sum_calib_loss": 4.0, "sum_calib_n_drawn": 60,
                          "sum_calib_value_pred": 3.0, "steps_done": 10},
    )
    assert block["enabled"] is True
    assert block["version"] == 1
    assert block["config"]["pool_size"] == 134
    np.testing.assert_allclose(block["loss"]["calib_loss_avg_iter"], 0.4)
    np.testing.assert_allclose(block["loss"]["calib_mean_value_pred"], 0.3)
    assert block["loss"]["calib_n_drawn_total"] == 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py::test_build_post_opening_calibration_block -v`
Expected: FAIL — `cannot import name 'build_post_opening_calibration_block'`

- [ ] **Step 3: Add `build_post_opening_calibration_block` to `calibration_pool.py`**

Append to `scripts/GPU/alphazero/calibration_pool.py`:

```python
def build_post_opening_calibration_block(config: dict, enabled: bool,
                                         loss_accumulator: dict) -> dict:
    """Per-iteration calibration telemetry for the training stats sidecar.

    calib_mean_value_pred is the headline signal: it should drift from ~+0.6
    toward the target (~-0.5) over the run.
    """
    steps = max(int(loss_accumulator.get("steps_done", 0)), 1)
    n_drawn = int(loss_accumulator.get("sum_calib_n_drawn", 0))
    return {
        "version": 1,
        "enabled": bool(enabled),
        "config": dict(config),
        "loss": {
            "calib_loss_avg_iter":
                float(loss_accumulator.get("sum_calib_loss", 0.0)) / steps,
            "calib_mean_value_pred":
                float(loss_accumulator.get("sum_calib_value_pred", 0.0)) / steps,
            "calib_n_drawn_total": n_drawn,
            "calib_n_drawn_per_step": n_drawn / steps,
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py::test_build_post_opening_calibration_block -v`
Expected: PASS

- [ ] **Step 5: Add `train()` params**

In `scripts/GPU/alphazero/trainer.py`, add to the `train()` signature, immediately after the recovery_retargeting params (after line 2282, before the closing `) -> AlphaZeroNetwork:`):

```python
    # Post-opening sharp-drop calibration (design 2026-06-16)
    post_opening_calibration_enabled: bool = False,
    post_opening_calibration_manifest: Optional[str] = None,
    post_opening_calibration_target: float = -0.50,
    post_opening_calibration_weight: float = 0.02,
    post_opening_calibration_batch_fraction: float = 0.10,
```

- [ ] **Step 6: Gate the effective weight and build the pool**

In `scripts/GPU/alphazero/trainer.py`, immediately after the conversion effective-weight gating (after line 2678, where `effective_conversion_loss_weight` is set):

```python
    # Post-opening calibration: gate weight + build the fixed pool once.
    effective_post_opening_calibration_weight: float = (
        post_opening_calibration_weight if post_opening_calibration_enabled else 0.0
    )
    _calib_pool = None
    if effective_post_opening_calibration_weight > 0.0:
        if not post_opening_calibration_manifest:
            raise ValueError(
                "post_opening_calibration_enabled requires "
                "post_opening_calibration_manifest")
        from .calibration_pool import CalibrationPool
        _calib_pool = CalibrationPool.from_manifest(
            post_opening_calibration_manifest, post_opening_calibration_target)
        print(f"Post-opening calibration: {len(_calib_pool)} positions, "
              f"weight={effective_post_opening_calibration_weight}, "
              f"target={post_opening_calibration_target}, "
              f"batch_fraction={post_opening_calibration_batch_fraction}")

    # Iteration-scope calibration accumulators (mirror sum_aux* hoist).
    sum_calib_loss: float = 0.0
    sum_calib_n_drawn: int = 0
    sum_calib_value_pred: float = 0.0
```

- [ ] **Step 7: Reset calibration accumulators per training run**

In the per-iteration accumulator reset block (after line 3650, alongside `sum_boost_inactive = 0`):

```python
                sum_calib_loss = 0.0
                sum_calib_n_drawn = 0
                sum_calib_value_pred = 0.0
```

- [ ] **Step 8: Branch the step loop on calibration**

In `scripts/GPU/alphazero/trainer.py`, replace the `train_step(...)` call inside the step loop (lines 3661-3676) with this branch. The `else` branch is the original call verbatim — the disabled path stays byte-identical.

> **Do NOT simplify this into a single `train_step(..., calibration_positions=None)` call.** A unified call would route the disabled path through the calibration-aware return handling and defeats the byte-identical guarantee (Global Constraint). Keep the two explicit branches: when `_calib_pool is None`, the original 7-return call runs unchanged; the 10-return form appears only in the calibrated branch.

```python
                        if _calib_pool is not None:
                            _k = max(1, round(batch_size * post_opening_calibration_batch_fraction))
                            _calib_batch = _calib_pool.sample(_k, train_rng)
                            (loss_total, loss_policy, loss_value, loss_l2, loss_aux, aux_cov, aux_neli,
                             _calib_loss, _calib_value_pred, _calib_n) = train_step(
                                network=network,
                                main_module=main_module,
                                opt_main=opt_main,
                                opt_value=opt_value,
                                batch=batch,
                                l2_weight=l2_weight,
                                value_weight=curr_value_weight,
                                active_size=active_size,
                                value_grad_max_norm=value_grad_max_norm,
                                progress_weighted=progress_weighted,
                                progress_weight_floor=progress_weight_floor,
                                conversion_loss_weight=effective_conversion_loss_weight,
                                conversion_completion_weight=conversion_completion_weight,
                                conversion_reducer_weight=conversion_reducer_weight,
                                calibration_positions=_calib_batch,
                                calibration_loss_weight=effective_post_opening_calibration_weight,
                            )
                            sum_calib_loss += _calib_loss
                            sum_calib_n_drawn += _calib_n
                            sum_calib_value_pred += _calib_value_pred
                        else:
                            loss_total, loss_policy, loss_value, loss_l2, loss_aux, aux_cov, aux_neli = train_step(
                                network=network,
                                main_module=main_module,
                                opt_main=opt_main,
                                opt_value=opt_value,
                                batch=batch,
                                l2_weight=l2_weight,
                                value_weight=curr_value_weight,
                                active_size=active_size,
                                value_grad_max_norm=value_grad_max_norm,
                                progress_weighted=progress_weighted,
                                progress_weight_floor=progress_weight_floor,
                                conversion_loss_weight=effective_conversion_loss_weight,
                                conversion_completion_weight=conversion_completion_weight,
                                conversion_reducer_weight=conversion_reducer_weight,
                            )
```

- [ ] **Step 9: Emit the calibration sidecar block**

In `scripts/GPU/alphazero/trainer.py`, immediately after the `conversion_training` block is added to `_sidecar` (after line 3786):

```python
        if post_opening_calibration_enabled or _calib_pool is not None:
            from .calibration_pool import build_post_opening_calibration_block
            _sidecar["post_opening_calibration"] = build_post_opening_calibration_block(
                config={
                    "enabled": post_opening_calibration_enabled,
                    "manifest": post_opening_calibration_manifest,
                    "target": post_opening_calibration_target,
                    "configured_weight": post_opening_calibration_weight,
                    "effective_weight": effective_post_opening_calibration_weight,
                    "batch_fraction": post_opening_calibration_batch_fraction,
                    "pool_size": len(_calib_pool) if _calib_pool is not None else 0,
                },
                enabled=post_opening_calibration_enabled,
                loss_accumulator={
                    "sum_calib_loss": sum_calib_loss,
                    "sum_calib_n_drawn": sum_calib_n_drawn,
                    "sum_calib_value_pred": sum_calib_value_pred,
                    "steps_done": steps_done,
                },
            )
```

- [ ] **Step 10: Run the calibration + regression suites**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py tests/test_calibration_loss.py tests/test_conversion_loss.py -v`
Expected: PASS (all)

- [ ] **Step 11: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py scripts/GPU/alphazero/calibration_pool.py tests/test_calibration_pool.py
git commit -m "feat(calibration): wire pool into train() loop + per-iter sidecar"
```

---

### Task 5: CLI flags in `train.py`

Expose the calibration knobs and thread them into the `train()` call.

**Files:**
- Modify: `scripts/GPU/alphazero/train.py` — argparse block (after line 373); `train_kwargs.update(...)` (lines 756-779).
- Test: `tests/test_calibration_cli_flags.py`

**Interfaces:**
- Consumes: `train()` params from Task 4.
- Produces: argparse attrs `post_opening_calibration_enabled/_manifest/_target/_weight/_batch_fraction` threaded into `train_kwargs`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_calibration_cli_flags.py`:

```python
from scripts.GPU.alphazero.train import build_arg_parser


def test_calibration_flag_defaults():
    args = build_arg_parser().parse_args([])
    assert args.post_opening_calibration_enabled is False
    assert args.post_opening_calibration_manifest is None
    assert args.post_opening_calibration_target == -0.50
    assert args.post_opening_calibration_weight == 0.02
    assert args.post_opening_calibration_batch_fraction == 0.10


def test_calibration_flags_set():
    args = build_arg_parser().parse_args([
        "--post-opening-calibration-enabled",
        "--post-opening-calibration-manifest", "train.csv",
        "--post-opening-calibration-weight", "0.05",
        "--post-opening-calibration-target", "-0.35",
        "--post-opening-calibration-batch-fraction", "0.15",
    ])
    assert args.post_opening_calibration_enabled is True
    assert args.post_opening_calibration_manifest == "train.csv"
    assert args.post_opening_calibration_weight == 0.05
    assert args.post_opening_calibration_target == -0.35
    assert args.post_opening_calibration_batch_fraction == 0.15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_cli_flags.py -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'post_opening_calibration_enabled'`

- [ ] **Step 3: Add the argparse flags**

In `scripts/GPU/alphazero/train.py`, immediately after the conversion flags (after line 373):

```python
    # Post-opening sharp-drop calibration (design 2026-06-16)
    parser.add_argument("--post-opening-calibration-enabled", action="store_true",
        help="Enable the post-opening sharp-drop value calibration aux loss.")
    parser.add_argument("--post-opening-calibration-manifest", type=str, default=None,
        help="Path to the calibration TRAIN manifest CSV (required when enabled).")
    parser.add_argument("--post-opening-calibration-target", type=float, default=-0.50,
        help="Soft value target (black perspective) for calibration positions "
             "(default: -0.50).")
    parser.add_argument("--post-opening-calibration-weight", type=float, default=0.02,
        help="Absolute coefficient on the calibration value-loss term "
             "(default: 0.02; NOT multiplied by value_weight).")
    parser.add_argument("--post-opening-calibration-batch-fraction", type=float, default=0.10,
        help="Calibration mini-batch size as a fraction of batch_size (default: 0.10).")
```

- [ ] **Step 4: Thread into `train_kwargs`**

In `scripts/GPU/alphazero/train.py`, inside the `train_kwargs.update(dict(...))` block (lines 756-779), add:

```python
        post_opening_calibration_enabled=args.post_opening_calibration_enabled,
        post_opening_calibration_manifest=args.post_opening_calibration_manifest,
        post_opening_calibration_target=args.post_opening_calibration_target,
        post_opening_calibration_weight=args.post_opening_calibration_weight,
        post_opening_calibration_batch_fraction=args.post_opening_calibration_batch_fraction,
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibration_cli_flags.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Smoke-check the flags reach the CLI**

Run: `.venv/bin/python -m scripts.GPU.alphazero.train --help | grep -i "post-opening-calibration"`
Expected: all five `--post-opening-calibration-*` flags are listed (`enabled`, `manifest`, `target`, `weight`, `batch-fraction`).

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/train.py tests/test_calibration_cli_flags.py
git commit -m "feat(calibration): --post-opening-calibration-* CLI flags"
```

---

### Task 6: Integration verification + data pipeline

Prove the whole pipeline on real data (no training run yet) and run the full suite.

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite (regression gate)**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS (all tests; new calibration tests + no regressions).

- [ ] **Step 2: Regenerate the review queue wide enough to cover the full cohort**

The full post-opening sharp-drop black-loss cohort is ~164 games; the default `--review-queue 50` truncates. Re-run the analyzer with a larger cap (200 covers it):

```bash
.venv/bin/python -m scripts.GPU.alphazero.eval_loss_replay_analyzer \
  --games-jsonl logs/eval/lr0003_0409_vs_0379_800g_w4_seed40937_replay_games.jsonl \
  --a-color black --review-queue 200 \
  --output-dir logs/eval/loss_analysis_v2_lr0003_0409_wide
```
Expected: writes `logs/eval/loss_analysis_v2_lr0003_0409_wide/lr0003_0409_vs_0379_800g_w4_seed40937_replay_manual_review_queue.csv`.

- [ ] **Step 3: Build the calibration train manifest (holdout-excluded)**

```bash
.venv/bin/python -m scripts.GPU.alphazero.build_calibration_manifest \
  --queue logs/eval/loss_analysis_v2_lr0003_0409_wide/lr0003_0409_vs_0379_800g_w4_seed40937_replay_manual_review_queue.csv \
  --holdout-manifest logs/eval/loss_analysis_v2_lr0003_0409/post_opening_sharp_drop_probe_manifest.csv \
  --out logs/eval/loss_analysis_v2_lr0003_0409/post_opening_sharp_drop_train_manifest.csv
```
Expected: prints `wrote ~134 calibration train cases ... (excluded 30 holdout games)`.

- [ ] **Step 4: Verify zero overlap with the frozen 30**

Run:
```bash
.venv/bin/python -c "
import csv
def gidx(p): return {int(r['game_idx']) for r in csv.DictReader(open(p))}
frozen = gidx('logs/eval/loss_analysis_v2_lr0003_0409/post_opening_sharp_drop_probe_manifest.csv')
train = gidx('logs/eval/loss_analysis_v2_lr0003_0409/post_opening_sharp_drop_train_manifest.csv')
print('frozen', len(frozen), 'train', len(train), 'overlap', len(frozen & train))
assert frozen & train == set(), 'LEAKAGE: train manifest contains frozen probe games'
print('OK: disjoint')
"
```
Expected: `overlap 0` and `OK: disjoint`.

- [ ] **Step 5: Verify the pool loads end-to-end (encodes real positions)**

Run:
```bash
.venv/bin/python -c "
from scripts.GPU.alphazero.calibration_pool import CalibrationPool
pool = CalibrationPool.from_manifest(
    'logs/eval/loss_analysis_v2_lr0003_0409/post_opening_sharp_drop_train_manifest.csv',
    -0.50)
print('pool size', len(pool))
r = pool._records[0]
print('to_move', r.to_move, 'outcome', r.outcome, 'board', r.board_tensor.shape)
assert all(x.to_move == 'black' and x.outcome == -0.5 for x in pool._records)
print('OK')
"
```
Expected: prints pool size ~134, `to_move black outcome -0.5 board (24, 24, 30)`, `OK`.

- [ ] **Step 6: Commit any manifest artifacts intended for version control**

```bash
git add logs/eval/loss_analysis_v2_lr0003_0409/post_opening_sharp_drop_train_manifest.csv
git commit -m "data(calibration): post-opening sharp-drop train manifest (ranks 31-164)"
```
(Skip if the project does not version-control generated manifests — confirm with the repo's `.gitignore` conventions first.)

---

## Experiment Runbook (post-implementation — operator-run, references spec §8–§9)

This is the actual experiment; it is long-running and not part of the TDD tasks.

**Preflight — confirm exact resume/checkpoint flag names** (avoids a mis-typed long run):
```bash
.venv/bin/python -m scripts.GPU.alphazero.train --help | grep -iE "weights|resume|checkpoint-dir|iterations|--lr"
```
Use the weights-only load flag (loads 0409 weights with a fresh optimizer/iteration counter), a NEW `--checkpoint-dir`, and the calibration flags:

```bash
.venv/bin/python -m scripts.GPU.alphazero.train \
  --load-weights-from checkpoints/alphazero-v2-lr0003-eps035-from0379/model_iter_0409.safetensors \
  --checkpoint-dir checkpoints/alphazero-v2-calib-from0409 \
  --iterations 15 --lr 0.0003 \
  --post-opening-calibration-enabled \
  --post-opening-calibration-manifest logs/eval/loss_analysis_v2_lr0003_0409/post_opening_sharp_drop_train_manifest.csv \
  --post-opening-calibration-weight 0.02 \
  --post-opening-calibration-target -0.50 \
  --post-opening-calibration-batch-fraction 0.10
```

> **Checkpoint numbering — confirm before probing.** `--load-weights-from` starts a fresh run counter, so 15 iterations produce `model_iter_0015.safetensors` in the new `--checkpoint-dir`. If you instead use `--resume` from 0409 (which *continues* the counter, as the from-0379 runs did), 15 iterations produce `model_iter_0424.safetensors`. After the run, `ls checkpoints/alphazero-v2-calib-from0409/` and substitute the **actual** produced filename into the probe commands below — the `model_iter_0015.safetensors` paths assume the fresh-counter case.

**Watch the headline signal** in each `iter_XXXX_stats.json` sidecar:
`post_opening_calibration.loss.calib_mean_value_pred` should drift from ~+0.6 toward −0.5. If it does not move, raise weight to 0.05.

**Gates (spec §8), in order — do NOT run the match until both probes pass:**

1. Goal-line probe (must not regress vs 0409):
```bash
.venv/bin/python -m scripts.GPU.alphazero.eval_goal_line_trigger_probe \
  --manifest <goal_line_trigger_probe_manifest.json> \
  --checkpoint checkpoints/alphazero-v2-calib-from0409/model_iter_0015.safetensors \
  --checkpoint checkpoints/alphazero-v2-lr0003-eps035-from0379/model_iter_0409.safetensors \
  --checkpoint checkpoints/alphazero-v2-staged/model_iter_0379.safetensors \
  --output-dir logs/eval/calib_goal_line
```
Pass: candidate severe ≤ 5.6%, overvalue ≤ 11.1%, no new per-case severe spike.

2. Post-opening probe (must improve materially vs 0409 — overvalue 93.3% / severe 76.7% / mean +0.604):
```bash
.venv/bin/python -m scripts.GPU.alphazero.eval_position_probe \
  --manifest logs/eval/loss_analysis_v2_lr0003_0409/post_opening_sharp_drop_probe_manifest.csv \
  --checkpoint checkpoints/alphazero-v2-calib-from0409/model_iter_0015.safetensors \
  --checkpoint checkpoints/alphazero-v2-lr0003-eps035-from0379/model_iter_0409.safetensors \
  --checkpoint checkpoints/alphazero-v2-staged/model_iter_0379.safetensors \
  --output-dir logs/eval/calib_post_opening
```
First-experiment pass: severe ≤ 60% AND mean_black_value ≤ +0.35.

3. Only if both pass: 800-game match candidate vs 0379; require non-inferiority (parity or better).

---

## Self-Review

**1. Spec coverage** (design §-by-§ → task):
- §4 train/eval split + §4.1 deterministic builder → Task 1 (selector, holdout-exclusion, determinism test) + Task 6 (regen + overlap=0 check).
- §5.1 CalibrationPool (reconstruct/encode) → Task 2.
- §5.1 aux value term in total_loss + §5.3 disabled byte-identity → Task 3 (loss/train_step, 7- vs 10-tuple, inert-when-disabled, gradient-to-value-head).
- §5.2 perspective helper → Task 2 (`target_in_to_move`, tested black/red/invalid).
- §6 CLI flags/defaults → Task 5.
- §7 diagnostics, calib_mean_value_pred every iteration → Task 4 (sidecar block + loop accumulation).
- §8 gates / §9 run config → Experiment Runbook.
- §10 tests → Tasks 1-5 each ship tests; §11/§12 (deferred distillation, rollback) → no code (disabled-by-default covered in Task 3/4).

**2. Placeholder scan:** No TBD/TODO. Every code step shows complete code. The Runbook's goal-line `<manifest.json>` is an operator input (the existing goal-line manifest path), and the weights-load flag has an explicit preflight `--help` confirm step — these are operator-runbook items, not engineer code placeholders.

**3. Type consistency:** `target_in_to_move`, `build_calibration_position`, `CalibrationPool.sample/from_manifest`, the `(calibration_positions, calibration_loss_weight)` param pair, the 10-tuple order `(total, policy, value, l2, aux, aux_cov, aux_neli, calib_loss, calib_value_mean, calib_n)`, and `build_post_opening_calibration_block(config, enabled, loss_accumulator)` are used identically across Tasks 2-5. `train()` param names match the Task 5 CLI threading exactly (`post_opening_calibration_enabled/_manifest/_target/_weight/_batch_fraction`).
