# Targeted Value Calibration v12 — Asymmetric Guardrail Hinge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-sided, black-perspective guardrail hinge loss (`loss_mode=asymmetric_guardrail_retention`) that penalizes only pro-black overvalue drift from BASE on B/C/D rows, while A keeps its symmetric hard correction — a new objective, tested under the v9 final-block training surface.

**Architecture:** A new value-only loss_mode whose rows carry a black-perspective BASE target in `target_black_value`. The trainer computes, in its single backward pass, a per-row hinge `relu(sign*(cb_values − cb_targets) − margin)²` (sign = +1 black-to-move / −1 red-to-move) over guardrail rows and the existing symmetric MSE over the remaining rows. New plumbing: a per-row guardrail **sign vector** emitted by the pool and threaded to the loss, plus a `--guardrail-margin` scalar. Additive and byte-identical when no guardrail rows are drawn.

**Tech Stack:** Python 3.14 / MLX (Apple Metal — no `requires_grad`; nested-dict params; `nn.value_and_grad` single backward pass), pytest.

**Spec:** `docs/superpowers/specs/2026-07-05-targeted-value-calibration-v12-asymmetric-guardrail-hinge-design.md` (APPROVED — do not redesign).

## Global Constraints

- Python: always `.venv/bin/python`; tests `.venv/bin/python -m pytest <file> -v`; full-suite baseline on main: **1331 passed**.
- NEVER `sys.modules.pop("mlx")` in tests.
- **The correctness triple** (must be pinned by tests) for an `asymmetric_guardrail_retention` row: `cb_targets` = `target_black_value` converted to side-to-move (via `target_in_to_move`); `guardrail_sign` = +1 if `to_move=="black"` else −1; policy mask / `has_policy_target` = 0/False.
- **Target semantics:** `target_black_value` is the loss target (BASE raw black value); `teacher_value` is provenance only for guardrail rows.
- **Byte-identical when unused:** `calibration_guardrail_sign=None` → `alphazero_loss_batch`/`train_step` return the pre-v12 7/10/14-tuples unchanged; all pre-existing calibration tests pass UNMODIFIED. No change to symmetric/teacher/continuation loss paths.
- Guardrail rows are value-only: NOT in `RETENTION_POLICY_LOSS_MODES` or `TEACHER_MODE_LOSS_MODES`; no policy CE; `calib_policy_ce`/`kl_est`/`n_teacher_retention` stay 0.
- Hinge uses eval-mode forward (like teacher_mode) so the candidate value is comparable to the eval-mode BASE target; pair with `--freeze-batchnorm-stats`.
- Training surface unchanged: `--freeze-batchnorm-stats --train-value-head-and-final-block` (v9). A stays `hard_value` −0.35. Margin default 0.10.
- Do NOT touch: mcts.py, continuation_extraction.py, the v8/v9 verifiers, docs/post-game-analysis.md, existing builders.
- Worktree `feature/tvc-v12-guardrail-hinge`; fresh worktree lacks gitignored game-log data → known 14F+6E in the whole-repo suite there; judge tasks file-scoped, authoritative suite on merged main. Per-task commits, FF-merge (no `--no-ff`, never force-push). Trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. File-scoped `git add`. Locate code by content, not line numbers.

## File Structure

| File | Role |
|---|---|
| `scripts/GPU/alphazero/calibration_pool.py` (modify) | new loss_mode, validation, schema, `split_samples_with_guardrail` |
| `scripts/GPU/alphazero/build_v12_guardrail_manifest.py` (create) | clone B/C/D root rows → guardrail rows |
| `scripts/GPU/alphazero/trainer.py` (modify) | `alphazero_loss_batch` hinge + `train_step` (Task 3); train-loop gate + telemetry (Task 4) |
| `scripts/GPU/alphazero/train.py` (modify) | `--guardrail-margin` (Task 4) |
| `scripts/GPU/alphazero/smoke_asymmetric_guardrail_v12.py` (create) | gate-0 smoke (Task 5) |
| `tests/test_asymmetric_guardrail_pool.py` / `_loss.py` / `_wiring.py` (create) | per-task tests |

**Task → user work-item map:** T1=pool plumbing, T2=manifest builder, T3+T4=trainer hinge+telemetry (split loss-core / loop-wiring), T5=smoke+suite+operator.

---

### Task 1: calibration_pool — new loss_mode, validation, schema, sign emission

**Files:**
- Modify: `scripts/GPU/alphazero/calibration_pool.py`
- Test: `tests/test_asymmetric_guardrail_pool.py` (create)

**Interfaces:**
- Consumes: existing `build_calibration_sample`, `CalibrationSample`, `target_in_to_move`, `PositionRecord`, `VALID_LOSS_MODES`, `_ALLOWED_RETENTION_MODE_SETS`, `from_manifest`, `split_samples`.
- Produces: `GUARDRAIL_LOSS_MODE = "asymmetric_guardrail_retention"`; `split_samples_with_guardrail(samples, has_weight_scale) -> (records, weights, guardrail_sign: np.float32 (N,))` where `guardrail_sign[i]` = +1/−1 (by `record.to_move`) for guardrail rows, else 0.0; `from_manifest` sets `schema == GUARDRAIL_LOSS_MODE` for guardrail manifests. Task 3/4 consume these.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_asymmetric_guardrail_pool.py`:

```python
"""v12 guardrail loss_mode: value-only rows whose target_black_value is the
BASE black-perspective value, emitting a per-row black-perspective sign."""
import numpy as np
import pytest

from scripts.GPU.alphazero import calibration_pool as cp
from scripts.GPU.alphazero.calibration_pool import (
    GUARDRAIL_LOSS_MODE, build_calibration_sample, split_samples_with_guardrail,
    target_in_to_move, VALID_LOSS_MODES, RETENTION_POLICY_LOSS_MODES,
    TEACHER_MODE_LOSS_MODES)


def test_guardrail_mode_registered_value_only():
    assert GUARDRAIL_LOSS_MODE == "asymmetric_guardrail_retention"
    assert GUARDRAIL_LOSS_MODE in VALID_LOSS_MODES
    assert GUARDRAIL_LOSS_MODE not in RETENTION_POLICY_LOSS_MODES
    assert GUARDRAIL_LOSS_MODE not in TEACHER_MODE_LOSS_MODES


def _case(**over):
    c = dict(
        case_id="c1", tag="goal_line_guardrail_retention",
        loss_mode=GUARDRAIL_LOSS_MODE,
        replay_path="MISSING.json", position_ply="0", side_to_move="black",
        target_black_value="0.20", teacher_value="0.20",
        teacher_policy_json="", root_visits_json="", extra_moves_json="",
        continuation_source="", continuation_depth="")
    c.update(over)
    return c


def test_guardrail_row_is_value_only(monkeypatch):
    # Stub position building so we test metadata, not board reconstruction.
    import scripts.GPU.alphazero.calibration_pool as m
    from scripts.GPU.alphazero.self_play import PositionRecord
    def fake_pos(case, target):
        # mirror the default (hard_value) branch: outcome = target in stm
        return PositionRecord(
            board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
            to_move=case["side_to_move"], legal_moves=[(0, 0)],
            visit_counts=[0], outcome=target_in_to_move(
                case["side_to_move"], float(case["target_black_value"])),
            active_size=24, ply=0, game_n_moves=None)
    monkeypatch.setattr(m, "build_calibration_position", fake_pos)
    s = build_calibration_sample(_case(), calibration_target=-0.35)
    assert s.loss_mode == GUARDRAIL_LOSS_MODE
    assert s.has_policy_target is False
    assert s.target_black_value == pytest.approx(0.20)


def test_guardrail_validation_rejects_policy_and_root(monkeypatch):
    import scripts.GPU.alphazero.calibration_pool as m
    monkeypatch.setattr(m, "build_calibration_position", lambda c, t: None)
    with pytest.raises(ValueError, match="teacher_policy_json"):
        build_calibration_sample(_case(teacher_policy_json="[0.5,0.5]"), -0.35)
    with pytest.raises(ValueError, match="root_visits_json"):
        build_calibration_sample(_case(root_visits_json="[0.5,0.5]"), -0.35)
    with pytest.raises(ValueError, match="target_black_value"):
        build_calibration_sample(_case(target_black_value=""), -0.35)


def test_split_emits_black_perspective_sign():
    from scripts.GPU.alphazero.self_play import PositionRecord
    from scripts.GPU.alphazero.calibration_pool import CalibrationSample
    def rec(side):
        return PositionRecord(
            board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
            to_move=side, legal_moves=[(0, 0)], visit_counts=[0],
            outcome=0.0, active_size=24, ply=0, game_n_moves=None)
    samples = [
        CalibrationSample(record=rec("black"), loss_mode=GUARDRAIL_LOSS_MODE,
                          tag="g", target_black_value=0.2),
        CalibrationSample(record=rec("red"), loss_mode=GUARDRAIL_LOSS_MODE,
                          tag="g", target_black_value=0.2),
        CalibrationSample(record=rec("black"), loss_mode="hard_value",
                          tag="a", target_black_value=-0.35),
    ]
    _records, _weights, sign = split_samples_with_guardrail(samples, False)
    assert list(sign) == [1.0, -1.0, 0.0]     # black=+1, red=-1, non-guardrail=0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_asymmetric_guardrail_pool.py -v`
Expected: `ImportError` on `GUARDRAIL_LOSS_MODE` / `split_samples_with_guardrail`.

- [ ] **Step 3: Implement**

**(a)** Locate `VALID_LOSS_MODES = frozenset({"hard_value"}) | TEACHER_MODE_LOSS_MODES` and replace with:

```python
GUARDRAIL_LOSS_MODE = "asymmetric_guardrail_retention"  # v12: value-only one-sided hinge
VALID_LOSS_MODES = frozenset({"hard_value", GUARDRAIL_LOSS_MODE}) | TEACHER_MODE_LOSS_MODES
```

**(b)** Locate the `_ALLOWED_RETENTION_MODE_SETS = (` tuple and add one member (guardrail may only coexist with `hard_value`, which is stripped before the check):

```python
    frozenset({"mcts_root_retention", CONTINUATION_LOSS_MODE}),
    frozenset({GUARDRAIL_LOSS_MODE}),                    # v12: guardrail-only B/C/D
)
```

**(c)** In `build_calibration_sample`, locate the `elif loss_mode == CONTINUATION_LOSS_MODE:` validation branch (the one checking `root_visits_json` blank) and add a new branch directly after it:

```python
    elif loss_mode == GUARDRAIL_LOSS_MODE:
        if case.get("target_black_value") in (None, ""):
            raise ValueError(
                f"{case.get('case_id')}: asymmetric_guardrail_retention row must "
                f"populate target_black_value (BASE black-perspective value)")
        if case.get("teacher_policy_json") not in (None, ""):
            raise ValueError(
                f"{case.get('case_id')}: asymmetric_guardrail_retention row must "
                f"leave teacher_policy_json blank (value-only, no policy CE)")
        if case.get("root_visits_json") not in (None, ""):
            raise ValueError(
                f"{case.get('case_id')}: asymmetric_guardrail_retention row must "
                f"leave root_visits_json blank (not a policy target)")
```

(No change needed in `build_calibration_position`: `GUARDRAIL_LOSS_MODE` is not `teacher_retention`/`mcts_root_retention`/`CONTINUATION_LOSS_MODE`, so it falls through the existing default branch → `outcome = target_in_to_move(state.to_move, _resolve_target_black(case, calibration_target))`, zero visit_counts. `has_policy_target` is False because `GUARDRAIL_LOSS_MODE` is not in `RETENTION_POLICY_LOSS_MODES` and `teacher_policy_json` is blank.)

**(d)** In `from_manifest`, locate `elif "teacher_retention" in modes:` (the schema selection chain) and add a branch directly below the `teacher_retention` case (before the `elif any(... target_black_value ...)` case):

```python
        elif "teacher_retention" in modes:
            schema = "teacher_retention"
        elif GUARDRAIL_LOSS_MODE in modes:
            schema = GUARDRAIL_LOSS_MODE
```

**(e)** Add `split_samples_with_guardrail` directly after `split_samples_with_modes`:

```python
def split_samples_with_guardrail(samples, has_weight_scale: bool):
    """Like split_samples, plus a guardrail_sign (float32 (N,)): +1.0 for a
    black-to-move guardrail row, -1.0 for a red-to-move guardrail row, 0.0 for
    non-guardrail rows. The v12 hinge uses this to convert the candidate value
    to black perspective per row (relu(sign*(v - target) - margin)**2)."""
    records, weights = split_samples(samples, has_weight_scale)
    sign = np.asarray(
        [(1.0 if s.record.to_move == "black" else -1.0)
         if s.loss_mode == GUARDRAIL_LOSS_MODE else 0.0
         for s in samples],
        dtype=np.float32)
    return records, weights, sign
```

- [ ] **Step 4: Run tests + pool regression**

Run: `.venv/bin/python -m pytest tests/test_asymmetric_guardrail_pool.py tests/test_calibration_pool.py tests/test_value_calibration_sampling.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/calibration_pool.py tests/test_asymmetric_guardrail_pool.py
git commit -m "feat(calibration): asymmetric_guardrail_retention loss_mode + black-perspective sign emission (v12)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: v12 guardrail manifest builder

**Files:**
- Create: `scripts/GPU/alphazero/build_v12_guardrail_manifest.py`
- Test: `tests/test_build_v12_guardrail_manifest.py` (create)

**Interfaces:**
- Consumes: the v7 manifest columns; `GUARDRAIL_LOSS_MODE` (Task 1) — built rows must load through `build_calibration_sample`.
- Produces: `logs/eval/targeted_calibration_v12_guardrail_from_calib020_0001.csv` = all `hard_value` rows from v7 + one guardrail clone per B/C/D root-retention row.

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_v12_guardrail_manifest.py`:

```python
import csv
from scripts.GPU.alphazero.build_v12_guardrail_manifest import (
    make_guardrail_clone, SOURCE_TO_GUARDRAIL_TAG)


def _parent(side, tv):
    return {"case_id": "game_1_ply_9", "tag": "goal_line_retention",
            "loss_mode": "mcts_root_retention", "side_to_move": side,
            "teacher_value": tv, "target_black_value": "",
            "root_visits_json": "[0.5,0.5]", "root_legal_moves_sha1": "abc",
            "teacher_policy_json": "", "extra_moves_json": ""}


def test_clone_converts_stm_teacher_value_to_black_target():
    # black to move: black target == stm teacher value
    b = make_guardrail_clone(_parent("black", "0.30"))
    assert b["loss_mode"] == "asymmetric_guardrail_retention"
    assert b["tag"] == "goal_line_guardrail_retention"
    assert b["case_id"] == "game_1_ply_9__guardrail"
    assert float(b["target_black_value"]) == 0.30
    # red to move: black target == -stm teacher value
    r = make_guardrail_clone(_parent("red", "-0.97"))
    assert float(r["target_black_value"]) == 0.97
    # value-only: policy/root blanked
    for row in (b, r):
        assert row["teacher_policy_json"] == ""
        assert row["root_visits_json"] == ""


def test_tag_map_covers_bcd_roots():
    assert SOURCE_TO_GUARDRAIL_TAG == {
        "goal_line_retention": "goal_line_guardrail_retention",
        "old_post_opening_retention": "old_post_opening_guardrail_retention",
        "red_predrop_retention": "red_predrop_guardrail_retention"}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_build_v12_guardrail_manifest.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `scripts/GPU/alphazero/build_v12_guardrail_manifest.py`**

```python
#!/usr/bin/env python3
"""v12: build the guardrail manifest.

Output = every hard_value row from the v7 manifest (A correction + severe-D,
kept for provenance though only A is scheduled) + one value-only guardrail
clone per B/C/D root-retention (`mcts_root_retention`) row. Each clone's
target_black_value is the parent's stm teacher_value converted to black
perspective (× sign). Pure copy + arithmetic — no reconstruction/MCTS; the
BASE anchor already lives in the parent's teacher_value. The old
retention/continuation/root_value rows are dropped (guardrail replaces
symmetric retention for B/C/D)."""
import argparse
import csv
from collections import Counter
from pathlib import Path

SOURCE_TO_GUARDRAIL_TAG = {
    "goal_line_retention": "goal_line_guardrail_retention",
    "old_post_opening_retention": "old_post_opening_guardrail_retention",
    "red_predrop_retention": "red_predrop_guardrail_retention",
}


def make_guardrail_clone(parent: dict) -> dict:
    tag = parent["tag"]
    if tag not in SOURCE_TO_GUARDRAIL_TAG:
        raise ValueError(f"{parent.get('case_id')}: not a B/C/D root tag: {tag}")
    tv = parent.get("teacher_value")
    if tv in (None, ""):
        raise ValueError(f"{parent.get('case_id')}: parent lacks teacher_value")
    sign = 1.0 if parent["side_to_move"] == "black" else -1.0
    target_black = float(tv) * sign
    row = dict(parent)
    row["case_id"] = f"{parent['case_id']}__guardrail"
    row["tag"] = SOURCE_TO_GUARDRAIL_TAG[tag]
    row["loss_mode"] = "asymmetric_guardrail_retention"
    row["target_black_value"] = repr(target_black)
    # value-only: blank every policy/root/continuation field
    for col in ("teacher_policy_json", "teacher_legal_moves_sha1",
                "root_visits_json", "root_legal_moves_sha1", "extra_moves_json",
                "continuation_side_to_move", "continuation_legal_moves_sha1",
                "continuation_depth", "continuation_parent_case_id",
                "continuation_source", "continuation_path_moves",
                "continuation_tree_visits", "continuation_tree_nn_value"):
        if col in row:
            row[col] = ""
    # teacher_value kept as provenance (per spec)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    default="logs/eval/targeted_calibration_v7_severe_d_root_correction_from_calib020_0001.csv")
    ap.add_argument("--output",
                    default="logs/eval/targeted_calibration_v12_guardrail_from_calib020_0001.csv")
    args = ap.parse_args()
    with Path(args.input).open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    kept = [r for r in rows if (r.get("loss_mode") or "hard_value") == "hard_value"]
    clones = [make_guardrail_clone(r) for r in rows
              if r.get("tag") in SOURCE_TO_GUARDRAIL_TAG]
    out_rows = kept + clones
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    counts = Counter(r["tag"] for r in out_rows)
    print(f"Wrote {args.output}: {len(out_rows)} rows "
          f"({len(kept)} hard_value kept, {len(clones)} guardrail clones)")
    for t in sorted(SOURCE_TO_GUARDRAIL_TAG.values()):
        print(f"  {t}: {counts.get(t, 0)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test + build + validate through the pool**

Run: `.venv/bin/python -m pytest tests/test_build_v12_guardrail_manifest.py -v` (PASS), then:
`.venv/bin/python scripts/GPU/alphazero/build_v12_guardrail_manifest.py`
Expected: guardrail counts `18 / 30 / 30`. Then confirm it loads (if the local game-log replays are present):
`.venv/bin/python -c "from scripts.GPU.alphazero.calibration_pool import CalibrationPool; p=CalibrationPool.from_manifest('logs/eval/targeted_calibration_v12_guardrail_from_calib020_0001.csv', -0.35); print('schema', p.schema, 'tags', sorted(p.tag_counts()))"`
Expected: `schema asymmetric_guardrail_retention` and the guardrail tags present. (If replays are absent locally, note it — the schema/validation still loads; full reconstruction is exercised in the smoke on the operator box.)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/build_v12_guardrail_manifest.py tests/test_build_v12_guardrail_manifest.py
git commit -m "feat(calibration): v12 guardrail manifest builder — B/C/D root rows -> black-perspective guardrail rows

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `alphazero_loss_batch` hinge + `train_step` (loss core, behavior-tested)

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` (`alphazero_loss_batch` signature + calib section + forward gate; `train_step` signature + loss_fn pass + unpack + return)
- Test: `tests/test_asymmetric_guardrail_loss.py` (create)

**Interfaces:**
- Consumes: `make_padded_batch` (order-preserving), the existing calib section, `train_step`/`alphazero_loss_batch`.
- Produces: `alphazero_loss_batch(..., calibration_guardrail_sign=None, guardrail_margin: float = 0.1)` and `train_step(..., calibration_guardrail_sign=None, guardrail_margin: float = 0.1)`; when the sign vector is present, a **13-tuple** = the 10-tuple + `(guardrail_hinge_loss, guardrail_active_frac, guardrail_n)`. Task 4 consumes these.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_asymmetric_guardrail_loss.py`:

```python
"""v12 hinge: relu(sign*(v - target) - margin)^2, value-only, byte-identical
when the guardrail sign vector is absent."""
import numpy as np
import pytest
import mlx.optimizers as optim

from scripts.GPU.alphazero.trainer import (
    MainModule, freeze_batchnorm_running_stats, train_step)
from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord


def _pos(to_move="red", outcome=1.0):
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move=to_move, legal_moves=[(0, 0), (1, 1), (2, 2)],
        visit_counts=[10, 5, 3], outcome=outcome, active_size=24,
        ply=0, game_n_moves=10)


def _setup():
    net = create_network(hidden=64, n_blocks=2)
    freeze_batchnorm_running_stats(net)
    mm = MainModule(net.encoder, net.policy_head)
    return net, mm, optim.Adam(learning_rate=1e-3), optim.Adam(learning_rate=1e-3)


def _guardrail_calib(to_move, target_black):
    # a single guardrail calibration row; outcome carries target in stm
    from scripts.GPU.alphazero.calibration_pool import target_in_to_move
    return [PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move=to_move, legal_moves=[(0, 0), (1, 1)], visit_counts=[0, 0],
        outcome=target_in_to_move(to_move, target_black), active_size=24,
        ply=20, game_n_moves=None)]


def _run(calib, sign, margin=0.10):
    net, mm, om, ov = _setup()
    out = train_step(network=net, main_module=mm, opt_main=om, opt_value=ov,
                     batch=[_pos() for _ in range(3)],
                     calibration_positions=calib, calibration_loss_weight=0.01,
                     calibration_guardrail_sign=np.asarray(sign, dtype=np.float32),
                     guardrail_margin=margin,
                     train_value_head_and_final_block=True)
    return out


def _expected_hinge(v_stm, to_move, target_black, sign, margin=0.10):
    from scripts.GPU.alphazero.calibration_pool import target_in_to_move
    target_stm = target_in_to_move(to_move, target_black)
    over = sign * (v_stm - target_stm) - margin
    return max(0.0, over) ** 2


def test_guardrail_tuple_arity_is_13():
    out = _run(_guardrail_calib("black", 0.2), [1.0])
    assert len(out) == 13


def test_hinge_matches_formula_black():
    # Derive the expected hinge from the ACTUAL predicted value (out[8] =
    # calib_value_mean = this single row's stm value) so the test does not
    # depend on random init. Covers below/within/above the margin generically.
    for target_black in (0.9, 0.0, -0.9):
        out = _run(_guardrail_calib("black", target_black), [1.0])
        v_stm = out[8]                                    # calib_value_mean
        exp = _expected_hinge(v_stm, "black", target_black, sign=1.0)
        assert out[10] == pytest.approx(exp, abs=1e-5), (target_black, v_stm)
        assert out[11] == pytest.approx(1.0 if exp > 0 else 0.0)  # active_frac


def test_below_target_zero_hinge_black():
    # target_black=0.9 → threshold v>1.0, impossible (tanh<1) → hinge always 0.
    out = _run(_guardrail_calib("black", 0.9), [1.0])
    assert out[10] == 0.0


def test_red_to_move_sign_matches_formula():
    # Red-to-move: hinge fires on cand_black > target_black, i.e. cand_stm BELOW
    # target_stm. Pin the exact formula with sign=-1 against the actual value.
    out = _run(_guardrail_calib("red", -0.9), [-1.0])
    v_stm = out[8]
    exp = _expected_hinge(v_stm, "red", -0.9, sign=-1.0)
    assert out[10] == pytest.approx(exp, abs=1e-5), v_stm
    # sanity: a +1 sign on the same values would give a different result
    wrong = _expected_hinge(v_stm, "red", -0.9, sign=1.0)
    assert not (exp == pytest.approx(wrong)) or exp == 0.0 == wrong


def test_byte_identical_when_sign_absent():
    # Same calibration row, no guardrail sign -> pre-v12 10-tuple, symmetric MSE.
    net, mm, om, ov = _setup()
    out = train_step(network=net, main_module=mm, opt_main=om, opt_value=ov,
                     batch=[_pos() for _ in range(3)],
                     calibration_positions=_guardrail_calib("black", 0.2),
                     calibration_loss_weight=0.01,
                     train_value_head_and_final_block=True)
    assert len(out) == 10                 # unchanged calib value-only path
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_asymmetric_guardrail_loss.py -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'calibration_guardrail_sign'`.

- [ ] **Step 3: Implement**

**(a) `alphazero_loss_batch` signature.** Locate `teacher_policy_kl_weight: float = 0.25,                # v4 (CE gradient term)` in the `def alphazero_loss_batch(` params and add below it:

```python
    teacher_policy_kl_weight: float = 0.25,                # v4 (CE gradient term)
    calibration_guardrail_sign=None,                  # v12: (N,) +1/-1 guardrail / 0 else
    guardrail_margin: float = 0.1,                    # v12: hinge tolerance band
```

**(b) forward gate.** Locate `teacher_mode = calibration_teacher_policy_mask is not None` inside the `if calib_active:` block and the following `if teacher_mode:` eval-mode forward. Replace the `if teacher_mode:` line with a combined gate so guardrail rows also use the eval-mode forward:

```python
        teacher_mode = calibration_teacher_policy_mask is not None
        guardrail_mode = calibration_guardrail_sign is not None
        if teacher_mode or guardrail_mode:
```

**(c) hinge branch.** Locate `calib_value_mean = mx.mean(cb_values)` and, directly after it, insert the guardrail branch (before the existing `if not teacher_mode:`):

```python
        if guardrail_mode:
            # v12: one-sided black-perspective hinge on guardrail rows; symmetric
            # MSE on the remaining (A hard_value) rows. Value-only, no policy CE.
            sign = mx.reshape(mx.array(calibration_guardrail_sign), per_value.shape)
            gmask = mx.abs(sign)                              # 1.0 guardrail / 0.0 else
            base_w = _w if _w is not None else mx.ones(per_value.shape)
            ng_w = base_w * (1.0 - gmask)
            value_term = mx.sum(ng_w * per_value) / mx.maximum(mx.sum(ng_w), 1e-8)
            signed_over = sign * (cb_values - cb_targets) - guardrail_margin
            hinge = mx.maximum(signed_over, 0.0) ** 2
            g_w = base_w * gmask
            denom_g = mx.maximum(mx.sum(g_w), 1e-8)
            guardrail_hinge_loss = mx.sum(g_w * hinge) / denom_g
            active = (signed_over > 0.0).astype(cb_values.dtype)
            guardrail_active_frac = mx.sum(g_w * active) / denom_g
            calib_loss = value_term + guardrail_hinge_loss
            total_loss = total_loss + calibration_loss_weight * calib_loss
            # CRITICAL: total_loss must be first for nn.value_and_grad()
            return (total_loss, policy_loss, value_loss, l2_loss,
                    aux_loss, aux_coverage, aux_n_eligible,
                    calib_loss, calib_value_mean, len(calibration_positions),
                    guardrail_hinge_loss, guardrail_active_frac,
                    int(mx.sum(gmask).item()))
```

**(d) `train_step` signature.** Locate `train_value_head_and_final_block: bool = False,              # v8: skip opt_main.update`… actually locate `train_value_head_and_final_block: bool = False,   # v9: only value head + final block` in the `def train_step(` params and add below it:

```python
    train_value_head_and_final_block: bool = False,   # v9: only value head + final block
    calibration_guardrail_sign=None,                  # v12
    guardrail_margin: float = 0.1,                    # v12
```

**(e) `train_step` loss_fn pass.** In `def loss_fn(model):` locate `teacher_policy_kl_weight=teacher_policy_kl_weight,` (the last kwarg passed to `alphazero_loss_batch`) and add below it:

```python
            teacher_policy_kl_weight=teacher_policy_kl_weight,
            calibration_guardrail_sign=calibration_guardrail_sign,
            guardrail_margin=guardrail_margin,
```

**(f) `train_step` unpack.** Locate the unpack chain `if calib_active and teacher_mode:` (with the 14-var unpack). Add `guardrail_mode = calibration_guardrail_sign is not None` directly above it, and add an `elif` for the 13-tuple between the teacher and the plain-calib cases:

```python
    teacher_mode = calibration_teacher_policy_mask is not None
    guardrail_mode = calibration_guardrail_sign is not None
    if calib_active and teacher_mode:
        (total_loss, policy_loss, value_loss, l2_loss, aux_loss, aux_coverage,
         aux_n_eligible, calib_loss, calib_value_mean, calib_n,
         calib_value_term, calib_policy_ce, calib_policy_kl_est, calib_n_retention) = loss_tuple
    elif calib_active and guardrail_mode:
        (total_loss, policy_loss, value_loss, l2_loss, aux_loss, aux_coverage,
         aux_n_eligible, calib_loss, calib_value_mean, calib_n,
         guardrail_hinge_loss, guardrail_active_frac, guardrail_n) = loss_tuple
    elif calib_active:
```

**(g) `train_step` return.** Locate `if calib_active and teacher_mode:` in the RETURN section (the `return (float(total_loss.item()), ...)` 14-tuple). Add a guardrail return branch directly after that teacher return block (before `if calib_active:`):

```python
    if calib_active and guardrail_mode:
        return (
            float(total_loss.item()), float(policy_loss.item()),
            float(value_loss.item()), float(l2_loss.item()),
            float(aux_loss.item()), float(aux_coverage), int(aux_n_eligible),
            float(calib_loss.item()), float(calib_value_mean.item()), int(calib_n),
            float(guardrail_hinge_loss.item()), float(guardrail_active_frac.item()),
            int(guardrail_n),
        )
```

- [ ] **Step 4: Run the new tests + trainer regression**

Run: `.venv/bin/python -m pytest tests/test_asymmetric_guardrail_loss.py tests/test_calibration_loss.py tests/test_trainer_teacher_mode_gate.py tests/test_train_value_head_and_final_block.py tests/test_training.py -v`
Expected: ALL PASS (pre-existing files unmodified; byte-identical when sign absent).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py tests/test_asymmetric_guardrail_loss.py
git commit -m "feat(training): v12 guardrail hinge in alphazero_loss_batch + train_step (13-tuple; byte-identical when unused)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: train-loop gate + telemetry + `--guardrail-margin` (wiring)

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` (train-loop gate/call/accumulators; `train()` signature; `build_post_opening_calibration_block`), `scripts/GPU/alphazero/train.py` (CLI + plumb)
- Test: `tests/test_asymmetric_guardrail_wiring.py` (create)

**Interfaces:**
- Consumes: `split_samples_with_guardrail` (T1), `GUARDRAIL_LOSS_MODE` (T1), the 13-tuple `train_step` (T3).
- Produces: `--guardrail-margin` CLI (default 0.10) → `post_opening_guardrail_margin` → `train_step`; JSON telemetry `guardrail_hinge_loss`, `guardrail_active_frac`, `guardrail_margin`.

- [ ] **Step 1: Write the failing (source-level) tests**

Create `tests/test_asymmetric_guardrail_wiring.py`:

```python
"""Wiring pins for the guardrail path inside the 4000-line train loop
(precedent: tests/test_train_value_head_and_final_block.py)."""
from scripts.GPU.alphazero import trainer as trainer_mod
from scripts.GPU.alphazero import train as train_mod


def test_train_loop_selects_guardrail_split_and_forwards_sign():
    src = open(trainer_mod.__file__).read()
    assert "split_samples_with_guardrail" in src
    assert "GUARDRAIL_LOSS_MODE" in src
    assert "calibration_guardrail_sign=_calib_guard_sign," in src
    assert "guardrail_margin=post_opening_guardrail_margin," in src
    # telemetry accumulation + JSON
    assert "sum_guardrail_hinge_loss" in src
    assert '"guardrail_hinge_loss"' in src
    assert '"guardrail_active_frac"' in src
    assert '"guardrail_margin"' in src


def test_cli_guardrail_margin_flag_and_plumb():
    src = open(train_mod.__file__).read()
    assert '"--guardrail-margin"' in src
    assert "post_opening_guardrail_margin=args.guardrail_margin," in src
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_asymmetric_guardrail_wiring.py -v`
Expected: FAIL on the missing strings.

- [ ] **Step 3: Implement**

**(a) train-loop import + gate.** Locate the calibration import + gate block (the `from .calibration_pool import (split_samples, split_samples_with_modes, TEACHER_MODE_LOSS_MODES)` and the `if _calib_pool.schema in TEACHER_MODE_LOSS_MODES:` chain). Extend the import and add a guardrail branch:

```python
                            from .calibration_pool import (
                                split_samples, split_samples_with_modes,
                                split_samples_with_guardrail,
                                TEACHER_MODE_LOSS_MODES, GUARDRAIL_LOSS_MODE)
```
and replace the `if/else` schema branch with:

```python
                            _calib_guard_sign = None
                            if _calib_pool.schema in TEACHER_MODE_LOSS_MODES:
                                _calib_batch, _calib_weights, _calib_tp_mask = (
                                    split_samples_with_modes(_calib_samples,
                                                             _calib_pool.has_weight_scale))
                            elif _calib_pool.schema == GUARDRAIL_LOSS_MODE:
                                _calib_batch, _calib_weights, _calib_guard_sign = (
                                    split_samples_with_guardrail(
                                        _calib_samples, _calib_pool.has_weight_scale))
                                _calib_tp_mask = None
                            else:
                                _calib_batch, _calib_weights = split_samples(
                                    _calib_samples, _calib_pool.has_weight_scale)
                                _calib_tp_mask = None
```

**(b) forward the sign + margin at the `train_step(...)` call.** Locate `train_value_head_and_final_block=train_value_head_and_final_block,` in the calibration-branch `train_step(` call (the one near `calibration_teacher_policy_mask=_calib_tp_mask,`) and add below it:

```python
                                train_value_head_and_final_block=train_value_head_and_final_block,
                                calibration_guardrail_sign=_calib_guard_sign,
                                guardrail_margin=post_opening_guardrail_margin,
```

**(c) accumulators.** Locate `sum_n_teacher_retention = 0` (the calibration accumulator init) and add below it:

```python
                sum_n_teacher_retention = 0
                sum_guardrail_hinge_loss = 0.0
                sum_guardrail_active_frac = 0.0
```

Then locate the teacher-telemetry accumulation `if _calib_tp_mask is not None and len(_ret) == 14:` block and add a sibling directly after it:

```python
                            if _calib_guard_sign is not None and len(_ret) == 13:
                                sum_guardrail_hinge_loss += _ret[10]
                                sum_guardrail_active_frac += _ret[11]
```

**(d) `train()` signature.** Locate `train_value_head_and_final_block: bool = False,` in the `def train(` parameter list (the one whose neighbors are the calibration params, near `freeze_batchnorm_stats`) and add below it:

```python
    train_value_head_and_final_block: bool = False,
    post_opening_guardrail_margin: float = 0.1,
```

**(e) JSON telemetry.** In `build_post_opening_calibration_block` (calibration_pool.py), locate `"n_teacher_retention_drawn":` in the `"loss"` dict and add three keys after that entry:

```python
            "n_teacher_retention_drawn":
                int(loss_accumulator.get("sum_n_teacher_retention", 0)),
            "guardrail_hinge_loss":
                float(loss_accumulator.get("sum_guardrail_hinge_loss", 0.0)) / steps,
            "guardrail_active_frac":
                float(loss_accumulator.get("sum_guardrail_active_frac", 0.0)) / steps,
            "guardrail_margin":
                float(loss_accumulator.get("guardrail_margin", 0.0)),
```

Then, where `train()` builds the `loss_accumulator` dict passed to `build_post_opening_calibration_block` (locate `"sum_calib_n_drawn_by_tag": sum_calib_n_drawn_by_tag,`), add:

```python
                        "sum_calib_n_drawn_by_tag": sum_calib_n_drawn_by_tag,
                        "sum_guardrail_hinge_loss": sum_guardrail_hinge_loss,
                        "sum_guardrail_active_frac": sum_guardrail_active_frac,
                        "guardrail_margin": post_opening_guardrail_margin,
```

**(f) `train.py` CLI flag.** Locate `parser.add_argument("--train-value-head-and-final-block", action="store_true",` and add a sibling argument after its full definition:

```python
    parser.add_argument("--guardrail-margin", type=float, default=0.10,
        help="v12: tolerance band (black-value units) for the asymmetric "
             "guardrail hinge; penalize pro-black drift above BASE by more "
             "than this. Default 0.10.")
```

**(g) `train.py` plumb.** Locate `train_value_head_and_final_block=args.train_value_head_and_final_block,` in the `train(...)` call and add below it:

```python
        post_opening_guardrail_margin=args.guardrail_margin,
```

- [ ] **Step 4: Run wiring tests + regression**

Run: `.venv/bin/python -m pytest tests/test_asymmetric_guardrail_wiring.py tests/test_calibration_cli_flags.py tests/test_value_calibration_sampling.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py scripts/GPU/alphazero/train.py scripts/GPU/alphazero/calibration_pool.py tests/test_asymmetric_guardrail_wiring.py
git commit -m "feat(training): v12 train-loop guardrail gate + telemetry + --guardrail-margin

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: gate-0 smoke + full-suite verification + merge handoff

**Files:**
- Create: `scripts/GPU/alphazero/smoke_asymmetric_guardrail_v12.py`
- Test: none new (the smoke is a script)

- [ ] **Step 1: Write the smoke** (mirrors `smoke_searched_continuation_retention_v6.py`)

```python
#!/usr/bin/env python3
"""Gate-0 smoke: load the v12 guardrail manifest, draw the v12 schedule, run a
handful of train_steps with the guardrail sign, and assert the guardrail
telemetry engaged (hinge present, policy CE zero)."""
import sys
import numpy as np
import mlx.optimizers as optim

from scripts.GPU.alphazero.calibration_pool import (
    CalibrationPool, split_samples_with_guardrail, GUARDRAIL_LOSS_MODE)
from scripts.GPU.alphazero.trainer import (
    MainModule, freeze_batchnorm_running_stats, train_step)
from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord

MANIFEST = "logs/eval/targeted_calibration_v12_guardrail_from_calib020_0001.csv"
SCHEDULE = {"black_predrop_correction": 2, "goal_line_guardrail_retention": 1,
            "old_post_opening_guardrail_retention": 2, "red_predrop_guardrail_retention": 2}


def main() -> int:
    pool = CalibrationPool.from_manifest(MANIFEST, calibration_target=-0.35)
    assert pool.schema == GUARDRAIL_LOSS_MODE, pool.schema
    import random
    rng = random.Random(0)
    samples = pool.sample_by_tag(SCHEDULE, rng)
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

Run: `.venv/bin/python scripts/GPU/alphazero/smoke_asymmetric_guardrail_v12.py` (requires the local replays; if absent, note it and defer the smoke to the operator box). Commit the smoke:
```bash
git add scripts/GPU/alphazero/smoke_asymmetric_guardrail_v12.py
git commit -m "test(calibration): v12 gate-0 guardrail smoke

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 2: Full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected in the worktree: baseline-passed + new tests with EXACTLY the known 14 failed + 6 errors. Authoritative check (1331 + new, 0 failures) on merged main before push.

- [ ] **Step 3: Hand off to merge**

FF-merge to main, authoritative suite on merged main, push (superpowers:finishing-a-development-branch). STOP after push — the operator run (build v12 manifest, train with the v12 schedule + `--guardrail-margin 0.10` + `--freeze-batchnorm-stats --train-value-head-and-final-block`, verifier exit 0, gates A/B/C/D) is the USER's; the exact command block is in the spec's Operator-run section.

---

## Operator run (USER's, after merge) — from the spec

Build the v12 manifest, then run the v9 train command with `--post-opening-calibration-manifest logs/eval/targeted_calibration_v12_guardrail_from_calib020_0001.csv`, schedule `black_predrop_correction=2,goal_line_guardrail_retention=1,old_post_opening_guardrail_retention=2,red_predrop_guardrail_retention=2`, `--guardrail-margin 0.10`, `--freeze-batchnorm-stats --train-value-head-and-final-block`. Telemetry: `guardrail_hinge_loss`/`guardrail_active_frac` present, `guardrail_margin=0.10`, `n_teacher_retention_drawn=0`, `calib_policy_ce_avg_iter=0`. Then `verify_value_head_and_final_block_checkpoint` exit 0, then gates A/B/C/D vs `calib020_0001`. No promotion unless all four pass.
