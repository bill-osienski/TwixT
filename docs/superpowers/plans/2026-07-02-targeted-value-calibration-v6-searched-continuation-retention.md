# Targeted Value Calibration v6 — Searched-Continuation Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add searched-continuation/PV retention to the calibration line: extract child/PV states from BASE's gate-faithful 400-sim MCTS trees under the fragile B/C/D roots, anchor each with a raw eval-mode teacher value, and train them through the existing masked teacher-retention loss path.

**Architecture:** Four new pieces ride the existing machinery: (1) a `searched_continuation_retention` loss mode + `extra_moves_json` reconstruction in `calibration_pool.py`; (2) a `search_with_root()` refactor of `MCTS.search()` so the builder can walk the gate-faithful tree; (3) a pure tree-walk extraction module + a manifest builder mirroring the v5 builder; (4) a gate-0 smoke. No new loss math — the 14-tuple masked value/policy path in `trainer.py` is reused, with a per-row policy mask replacing the per-mode mask.

**Tech Stack:** Python 3.14 / MLX (Apple Metal), pytest, CSV manifests, existing AlphaZero training stack under `scripts/GPU/alphazero/`.

**Spec:** `docs/superpowers/specs/2026-07-02-targeted-value-calibration-v6-searched-continuation-retention-design.md` (APPROVED). Ledger: `docs/2026-06-26-targeted-value-calibration-experiment-ledger-v3f-v4-overlap-updated-v6-prep.md`.

## Global Constraints

- Python: always `.venv/bin/python` from the repo root; tests: `.venv/bin/python -m pytest <file> -v`; full suite `.venv/bin/python -m pytest tests/ -q` (baseline: 1253 passed).
- NEVER `sys.modules.pop("mlx")` (or any mlx submodule) in tests — a later fresh `import mlx.core` re-inits the native Metal module and SIGABRTs the suite.
- Existing v2/v3/v4/v5 manifests must load **byte-identically**: blank/absent new columns → exactly today's behavior. `RETENTION_POLICY_LOSS_MODES` keeps its current two members.
- Gate-faithful search config is sacred: 400 sims, `mcts_eval_batch_size=14`, `mcts_stall_flush_sims=48`, `add_noise=False`, seeds `20260616 ^ game_idx ^ position_ply` (position-probe families) / `20260614 ^ game_idx` (goal-line), evaluator via `_default_evaluator_factory` (train-mode BN). `search_from_root` is FORBIDDEN for target generation (different leaf-eval path).
- Continuation `teacher_value` = fresh eval-mode `_teacher_infer` forward (separate `load_network_for_scoring(...).eval()` network). Tree `nn_value`s are train-mode-BN — provenance only, never a target.
- Do not change loss math in `trainer.py` (`alphazero_loss_batch`); only the two schema-gate call sites listed in Task 2.
- Builders never silently trim: every cap/threshold exclusion is logged; cap overflow is a hard `ValueError`.
- One feature branch (`feature/tvc-v6-searched-continuation-retention` via superpowers:using-git-worktrees), per-task commits, FF-merge to main when done (no `--no-ff`, never force-push).
- Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## File Structure

| File | Role |
|---|---|
| `scripts/GPU/alphazero/calibration_pool.py` (modify) | new loss mode, extra-moves reconstruction, per-row policy mask, schema rules |
| `scripts/GPU/alphazero/trainer.py` (modify, 2 call sites) | teacher-mode gate recognizes the new schema |
| `scripts/GPU/alphazero/mcts.py` (modify) | `search_with_root()` refactor of `search()` |
| `scripts/GPU/alphazero/continuation_extraction.py` (create) | pure tree-walk extraction (no MLX imports) |
| `scripts/GPU/alphazero/build_searched_continuation_retention_manifest.py` (create) | v6 builder CLI |
| `scripts/GPU/alphazero/smoke_searched_continuation_retention_v6.py` (create) | gate-0 smoke CLI |
| `tests/test_calibration_pool_continuation.py` (create) | Task 1 tests |
| `tests/test_calibration_loss.py` (modify) | Task 2 zero-mask loss tests |
| `tests/test_trainer_teacher_mode_gate.py` (create) | Task 2 wiring tests |
| `tests/test_mcts_search_with_root.py` (create) | Task 3 tests |
| `tests/test_continuation_extraction.py` (create) | Task 4 tests |
| `tests/test_build_searched_continuation_retention_manifest.py` (create) | Task 5 tests |
| `tests/test_smoke_searched_continuation_retention_v6.py` (create) | Task 6 tests |

---

### Task 1: Loader — `searched_continuation_retention` mode + `extra_moves_json` reconstruction

**Files:**
- Modify: `scripts/GPU/alphazero/calibration_pool.py`
- Test: `tests/test_calibration_pool_continuation.py` (create)

**Interfaces:**
- Consumes: `position_state(replay, ply, side)` (`goal_line_trigger_probe_cases.py:73`), `TwixtState.apply_move((r, c)) -> TwixtState` (`scripts/GPU/alphazero/game/twixt_state.py:322`), existing helpers in `calibration_pool.py`.
- Produces (later tasks rely on these exact names):
  - `CONTINUATION_LOSS_MODE = "searched_continuation_retention"` (module constant)
  - `TEACHER_MODE_LOSS_MODES: frozenset` = `RETENTION_POLICY_LOSS_MODES | {CONTINUATION_LOSS_MODE}`
  - `CalibrationSample.has_policy_target: bool` (new field, default `False`)
  - `split_samples_with_modes(...)` mask now `1.0 iff sample.has_policy_target`
  - `CalibrationPool.from_manifest` accepts manifests mixing `mcts_root_retention` + `searched_continuation_retention`; `pool.schema == "searched_continuation_retention"` when the new mode is present
  - Continuation-row required columns: `teacher_value`, `extra_moves_json` (non-empty JSON list of `{"row": int, "col": int}`), `continuation_side_to_move`, `continuation_legal_moves_sha1`; forbidden non-blank: `root_visits_json`; optional: `teacher_policy_json` + `teacher_legal_moves_sha1` (both continuation-state-aligned)
  - Continuation `PositionRecord`: `outcome = teacher_value` (stm), `visit_counts = teacher_policy` if policy present else `[0]*len(legal)`, `ply = position_ply + len(extra_moves)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_calibration_pool_continuation.py`:

```python
import json

import pytest

from scripts.GPU.alphazero.calibration_pool import (
    CONTINUATION_LOSS_MODE, RETENTION_POLICY_LOSS_MODES, TEACHER_MODE_LOSS_MODES,
    VALID_LOSS_MODES, CalibrationPool, build_calibration_sample,
    legal_moves_sha1, split_samples_with_modes)
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
from tests.goal_line_probe_fixtures import legal_replay


def _root_state(replay):
    return position_state(replay, 5, "black")   # plies 0-4 applied, black to move


def _continuation_fields(replay):
    """Two legal extra moves from ply 5; returns (fields, final_state)."""
    state = _root_state(replay)
    m1 = state.legal_moves()[0]
    s1 = state.apply_move(m1)
    m2 = s1.legal_moves()[0]
    s2 = s1.apply_move(m2)
    fields = {
        "extra_moves_json": json.dumps(
            [{"row": m1[0], "col": m1[1]}, {"row": m2[0], "col": m2[1]}]),
        "continuation_side_to_move": s2.to_move,
        "continuation_legal_moves_sha1": legal_moves_sha1(s2.legal_moves()),
    }
    return fields, s2


def _case(tmp_path, **overrides):
    rp = tmp_path / "game_000001.json"
    replay = legal_replay(9, game_idx=1)
    rp.write_text(json.dumps(replay))
    fields, final_state = _continuation_fields(replay)
    case = {
        "game_idx": "1",
        "case_id": "game_000001_ply_005__cont_pv2_x",
        "replay_path": str(rp), "position_ply": "5", "side_to_move": "black",
        "tag": "old_post_opening_continuation_retention",
        "loss_mode": CONTINUATION_LOSS_MODE,
        "teacher_value": "-0.25", "weight_scale": "1.0",
        **fields,
    }
    case.update(overrides)
    return case, final_state


def test_mode_sets():
    assert CONTINUATION_LOSS_MODE == "searched_continuation_retention"
    assert CONTINUATION_LOSS_MODE in VALID_LOSS_MODES
    assert CONTINUATION_LOSS_MODE in TEACHER_MODE_LOSS_MODES
    # backward-compat: the always-policy set is unchanged
    assert RETENTION_POLICY_LOSS_MODES == frozenset(
        {"teacher_retention", "mcts_root_retention"})
    assert CONTINUATION_LOSS_MODE not in RETENTION_POLICY_LOSS_MODES


def test_continuation_row_reconstructs_and_is_value_only(tmp_path):
    case, final_state = _case(tmp_path)
    sample = build_calibration_sample(case, calibration_target=-0.35)
    assert sample.loss_mode == CONTINUATION_LOSS_MODE
    assert sample.has_policy_target is False
    rec = sample.record
    assert rec.outcome == pytest.approx(-0.25)          # teacher_value, stm, direct
    assert rec.to_move == final_state.to_move           # side flipped twice from black
    assert rec.legal_moves == final_state.legal_moves() # continuation legal set
    assert rec.visit_counts == [0] * len(rec.legal_moves)
    assert rec.ply == 7                                  # position_ply 5 + 2 extra


def test_continuation_row_with_policy_sets_mask(tmp_path):
    case, final_state = _case(tmp_path)
    legal = final_state.legal_moves()
    policy = [1.0 / len(legal)] * len(legal)
    case["teacher_policy_json"] = json.dumps(policy)
    case["teacher_legal_moves_sha1"] = legal_moves_sha1(legal)
    sample = build_calibration_sample(case, calibration_target=-0.35)
    assert sample.has_policy_target is True
    assert sample.record.visit_counts == pytest.approx(policy)
    _, _, mask = split_samples_with_modes([sample], has_weight_scale=False)
    assert mask.tolist() == [1.0]


def test_value_only_mask_is_zero_and_v5_mask_unchanged(tmp_path):
    case, _ = _case(tmp_path)
    cont = build_calibration_sample(case, calibration_target=-0.35)
    _, _, mask = split_samples_with_modes([cont], has_weight_scale=False)
    assert mask.tolist() == [0.0]


@pytest.mark.parametrize("break_field,break_value,match", [
    ("extra_moves_json", "", "extra_moves_json"),
    ("extra_moves_json", "[]", "extra_moves_json"),
    ("extra_moves_json", "not json", "extra_moves_json"),
    # (99,99) is off the active board -> never in legal_moves()
    ("extra_moves_json", json.dumps([{"row": 99, "col": 99}]), "illegal"),
    ("continuation_side_to_move", "", "continuation_side_to_move"),
    ("continuation_legal_moves_sha1", "deadbeef", "sha1"),
    ("teacher_value", "", "teacher_value"),
])
def test_continuation_row_fails_loud(tmp_path, break_field, break_value, match):
    case, _ = _case(tmp_path, **{break_field: break_value})
    with pytest.raises(ValueError, match=match):
        build_calibration_sample(case, calibration_target=-0.35)


def test_wrong_continuation_side_fails(tmp_path):
    case, final_state = _case(tmp_path)
    wrong = "red" if final_state.to_move == "black" else "black"
    case["continuation_side_to_move"] = wrong
    with pytest.raises(ValueError, match="continuation_side_to_move"):
        build_calibration_sample(case, calibration_target=-0.35)


def test_continuation_row_rejects_root_visits_json(tmp_path):
    case, _ = _case(tmp_path, root_visits_json=json.dumps([1.0]))
    with pytest.raises(ValueError, match="root_visits_json"):
        build_calibration_sample(case, calibration_target=-0.35)


def test_hard_value_row_rejects_continuation_columns(tmp_path):
    case, _ = _case(tmp_path)
    case["loss_mode"] = "hard_value"
    case["teacher_value"] = ""
    case["target_black_value"] = "-0.35"
    with pytest.raises(ValueError, match="hard_value"):
        build_calibration_sample(case, calibration_target=-0.35)


def _write_manifest(tmp_path, rows):
    import csv
    fields = sorted({k for r in rows for k in r})
    p = tmp_path / "manifest.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return p


def test_from_manifest_allows_root_plus_continuation_mix(tmp_path):
    cont_case, _ = _case(tmp_path)
    replay = legal_replay(9, game_idx=1)
    state = _root_state(replay)
    legal = state.legal_moves()
    dense = [0.0] * len(legal); dense[0] = 1.0
    root_case = {
        "game_idx": "1", "case_id": "game_000001_ply_005",
        "replay_path": cont_case["replay_path"],
        "position_ply": "5", "side_to_move": "black",
        "tag": "old_post_opening_retention",
        "loss_mode": "mcts_root_retention", "teacher_value": "-0.11",
        "root_visits_json": json.dumps(dense),
        "root_legal_moves_sha1": legal_moves_sha1(legal),
    }
    p = _write_manifest(tmp_path, [root_case, cont_case])
    pool = CalibrationPool.from_manifest(p, calibration_target=-0.35)
    assert pool.schema == "searched_continuation_retention"
    assert len(pool) == 2


def test_from_manifest_still_rejects_teacher_plus_root_mix(tmp_path):
    cont_case, _ = _case(tmp_path)
    teach = dict(cont_case)
    teach["case_id"] = "t1"
    teach["loss_mode"] = "teacher_retention"
    root = dict(cont_case)
    root["case_id"] = "r1"
    root["loss_mode"] = "mcts_root_retention"
    p = _write_manifest(tmp_path, [teach, root])
    with pytest.raises(ValueError, match="retention loss_modes"):
        CalibrationPool.from_manifest(p, calibration_target=-0.35)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool_continuation.py -v`
Expected: FAIL — `ImportError: cannot import name 'CONTINUATION_LOSS_MODE'`

- [ ] **Step 3: Implement in `calibration_pool.py`**

Replace lines 32-33 (the mode-set constants) with:

```python
RETENTION_POLICY_LOSS_MODES = frozenset({"teacher_retention", "mcts_root_retention"})
CONTINUATION_LOSS_MODE = "searched_continuation_retention"
# Modes whose pools use the teacher-mode (masked 14-tuple) loss path. The
# continuation mode is NOT in RETENTION_POLICY_LOSS_MODES: its rows carry a
# policy target only per-row (has_policy_target), not by mode.
TEACHER_MODE_LOSS_MODES = RETENTION_POLICY_LOSS_MODES | {CONTINUATION_LOSS_MODE}
VALID_LOSS_MODES = frozenset({"hard_value"}) | TEACHER_MODE_LOSS_MODES
# A manifest may mix at most these retention-mode combinations (v6 keeps the
# inert v5 root rows alongside the new continuation rows):
_ALLOWED_RETENTION_MODE_SETS = (
    frozenset(), frozenset({"teacher_retention"}), frozenset({"mcts_root_retention"}),
    frozenset({CONTINUATION_LOSS_MODE}),
    frozenset({"mcts_root_retention", CONTINUATION_LOSS_MODE}),
)
```

Add field to `CalibrationSample` (after `teacher_policy_len`):

```python
    has_policy_target: bool = False            # per-row policy-CE mask input
```

Add after `_parse_teacher_value` (module level):

```python
def _parse_extra_moves(case: dict) -> list[tuple[int, int]]:
    """Required non-empty extra_moves_json for continuation rows: JSON list of
    {"row": int, "col": int} applied after the position_ply reconstruction."""
    cid = case.get("case_id")
    raw = case.get("extra_moves_json")
    if raw in (None, ""):
        raise ValueError(f"{cid}: continuation row needs extra_moves_json")
    try:
        moves = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{cid}: extra_moves_json invalid JSON: {e}") from e
    if not isinstance(moves, list) or not moves:
        raise ValueError(f"{cid}: extra_moves_json must be a non-empty list")
    out = []
    for m in moves:
        if not isinstance(m, dict) or "row" not in m or "col" not in m:
            raise ValueError(f"{cid}: extra_moves_json entries need row/col: {m!r}")
        out.append((int(m["row"]), int(m["col"])))
    return out


def _apply_extra_moves(state, case: dict):
    """Apply extra_moves_json, then verify continuation_side_to_move and
    continuation_legal_moves_sha1 against the reconstructed state. Fail loud."""
    cid = case.get("case_id")
    extra = _parse_extra_moves(case)
    for (r, c) in extra:
        if (r, c) not in set(state.legal_moves()):
            raise ValueError(
                f"{cid}: extra move ({r},{c}) illegal at reconstructed state")
        state = state.apply_move((r, c))
    expected_side = case.get("continuation_side_to_move")
    if expected_side in (None, ""):
        raise ValueError(f"{cid}: continuation row needs continuation_side_to_move")
    if state.to_move != expected_side:
        raise ValueError(
            f"{cid}: continuation_side_to_move {expected_side!r} != reconstructed "
            f"{state.to_move!r}")
    stored = case.get("continuation_legal_moves_sha1") or ""
    recomputed = legal_moves_sha1(state.legal_moves())
    if stored != recomputed:
        raise ValueError(
            f"{cid}: continuation_legal_moves_sha1 mismatch; stored {stored!r} "
            f"!= recomputed {recomputed!r}")
    return state, len(extra)
```

In `build_calibration_position` (lines 132-196): move the loss-mode validation
BEFORE the tensor build and apply extra moves for continuation rows. Replace the
body between `state = position_state(...)` and the first `if loss_mode ==` branch:

```python
    state = position_state(replay, position_ply, side)

    loss_mode = case.get("loss_mode") or "hard_value"
    if loss_mode not in VALID_LOSS_MODES:
        raise ValueError(
            f"{case.get('case_id')}: unknown loss_mode {loss_mode!r} "
            f"(valid: {sorted(VALID_LOSS_MODES)})")
    record_ply = position_ply
    if loss_mode == CONTINUATION_LOSS_MODE:
        state, n_extra = _apply_extra_moves(state, case)
        record_ply = position_ply + n_extra

    board_chw = state.to_tensor()                       # (30, 24, 24) CHW
    board_hwc = np.transpose(board_chw, (1, 2, 0)).astype(np.float32)  # (24,24,30)
    legal = state.legal_moves()
```

(Delete the now-duplicate `loss_mode` parse that previously sat after `legal = ...`.)
Add the continuation branch before the final `hard_value` return, and use
`record_ply` instead of `position_ply` in ALL `PositionRecord(...)` constructions
in this function (it equals `position_ply` for non-continuation modes):

```python
    if loss_mode == CONTINUATION_LOSS_MODE:
        teacher_value = _parse_teacher_value(case)
        if case.get("teacher_policy_json") not in (None, ""):
            visit_counts = _parse_teacher_policy(case, legal)
        else:
            visit_counts = [0] * len(legal)
        return PositionRecord(
            board_tensor=board_hwc,
            to_move=state.to_move,
            legal_moves=legal,
            visit_counts=visit_counts,       # dense teacher policy or zeros (mask 0)
            outcome=teacher_value,           # raw eval-mode value anchor, stm, DIRECT
            active_size=state.active_size,
            ply=record_ply,
            game_n_moves=None,
        )
```

In `build_calibration_sample` (lines 199-227): extend the guards and metadata.
Replace the guard block and the two metadata lines:

```python
    loss_mode = case.get("loss_mode") or "hard_value"
    if loss_mode == "hard_value":
        populated = [k for k in ("teacher_value", "teacher_policy_json",
                                 "teacher_legal_moves_sha1",
                                 "root_visits_json", "root_legal_moves_sha1",
                                 "extra_moves_json", "continuation_side_to_move",
                                 "continuation_legal_moves_sha1")
                     if case.get(k) not in (None, "")]
        if populated:
            raise ValueError(
                f"{case.get('case_id')}: hard_value row must leave retention columns "
                f"blank; found {populated}")
    elif loss_mode == "mcts_root_retention":
        if case.get("teacher_policy_json") not in (None, ""):
            raise ValueError(
                f"{case.get('case_id')}: mcts_root_retention row must leave "
                f"teacher_policy_json blank (root_visits_json is the policy target)")
    elif loss_mode == CONTINUATION_LOSS_MODE:
        if case.get("root_visits_json") not in (None, ""):
            raise ValueError(
                f"{case.get('case_id')}: continuation row must leave "
                f"root_visits_json blank (it is not a root-policy target)")
```

and

```python
    teacher_value = (float(case["teacher_value"])
                     if loss_mode in TEACHER_MODE_LOSS_MODES else None)
    teacher_policy_len = (len(record.visit_counts)
                          if loss_mode in TEACHER_MODE_LOSS_MODES else None)
    has_policy_target = (
        loss_mode in RETENTION_POLICY_LOSS_MODES
        or (loss_mode == CONTINUATION_LOSS_MODE
            and case.get("teacher_policy_json") not in (None, "")))
    return CalibrationSample(record=record, weight_scale=weight_scale,
                             tag=tag, target_black_value=target_black,
                             loss_mode=loss_mode, teacher_value=teacher_value,
                             teacher_policy_len=teacher_policy_len,
                             has_policy_target=has_policy_target)
```

In `from_manifest` (lines 290-309): replace the mixing guard and schema pick:

```python
        cases = load_csv_manifest(manifest_path)["cases"]
        modes = {(c.get("loss_mode") or "hard_value") for c in cases}
        retention_modes = frozenset(modes - {"hard_value"})
        if retention_modes not in _ALLOWED_RETENTION_MODE_SETS:
            raise ValueError(
                f"manifest mixes retention loss_modes {sorted(retention_modes)}; "
                f"allowed combinations: "
                f"{sorted(sorted(s) for s in _ALLOWED_RETENTION_MODE_SETS)}")
        samples = [build_calibration_sample(c, calibration_target) for c in cases]
        has_weight_scale = any(c.get("weight_scale") not in (None, "") for c in cases)
        if CONTINUATION_LOSS_MODE in modes:
            schema = CONTINUATION_LOSS_MODE
        elif "mcts_root_retention" in modes:
            schema = "mcts_root_retention"
        elif "teacher_retention" in modes:
            schema = "teacher_retention"
        elif any(c.get("target_black_value") not in (None, "") for c in cases):
            schema = "per_row_target"
        else:
            schema = "global_target"
        return cls(samples, has_weight_scale=has_weight_scale, schema=schema)
```

In `split_samples_with_modes` (lines 322-330): replace the mask line:

```python
    mask = np.asarray(
        [1.0 if s.has_policy_target else 0.0 for s in samples],
        dtype=np.float32)
```

(`has_policy_target` is `True` for every `teacher_retention`/`mcts_root_retention`
row, so v4/v5 mask values are unchanged — the regression tests in
`tests/test_calibration_pool.py` and `tests/test_smoke_mcts_root_retention_v5.py`
must still pass untouched.)

- [ ] **Step 4: Run the new tests and the existing loader/smoke regression tests**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool_continuation.py tests/test_calibration_pool.py tests/test_smoke_mcts_root_retention_v5.py tests/test_build_mcts_root_retention_manifest.py -v`
Expected: ALL PASS (existing files unmodified)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/calibration_pool.py tests/test_calibration_pool_continuation.py
git commit -m "feat(calibration): searched_continuation_retention loss mode + extra_moves_json reconstruction + per-row policy mask"
```

---

### Task 2: Trainer wiring — teacher-mode gate recognizes the continuation schema

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py:2882-2896` and `trainer.py:3909-3930`
- Modify: `tests/test_calibration_loss.py` (append tests)
- Test: `tests/test_trainer_teacher_mode_gate.py` (create)

**Interfaces:**
- Consumes: `TEACHER_MODE_LOSS_MODES` from Task 1.
- Produces: a pool with `schema == "searched_continuation_retention"` takes the masked 14-tuple path in the train loop (mask passed via `split_samples_with_modes`); `alphazero_loss_batch` with an all-zero mask returns `policy_ce == 0.0` and `n_retention == 0`, finite everywhere.

Background (verified): `trainer.py:2889` and `trainer.py:3923` gate on
`_calib_pool.schema in RETENTION_POLICY_LOSS_MODES`. Without this task a v6 pool
silently falls back to the value-only 10-tuple path and the value anchor would
train through TRAIN-mode BN (batch-dependent) — the exact v2/v3 hazard.
`denom_p = mx.maximum(mx.sum(wm), 1e-8)` (`trainer.py:1277`) makes the all-zero
mask safe: `policy_ce == 0.0` exactly.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trainer_teacher_mode_gate.py`:

```python
"""The train-loop schema gate must route continuation pools down the masked
teacher-mode path. Source-level check (the gate is inline in a 4000-line
function; mirrors test_builder_module_defers_heavy_imports's source-inspection
style) plus a set-membership check."""
import re

from scripts.GPU.alphazero import trainer as trainer_mod
from scripts.GPU.alphazero.calibration_pool import (
    CONTINUATION_LOSS_MODE, TEACHER_MODE_LOSS_MODES)


def test_continuation_mode_is_teacher_mode():
    assert CONTINUATION_LOSS_MODE in TEACHER_MODE_LOSS_MODES


def test_trainer_gates_on_teacher_mode_loss_modes():
    src = open(trainer_mod.__file__).read()
    assert len(re.findall(r"schema in TEACHER_MODE_LOSS_MODES", src)) == 2, (
        "both schema gates (setup print + train-step mask split) must use "
        "TEACHER_MODE_LOSS_MODES")
    assert "schema in RETENTION_POLICY_LOSS_MODES" not in src
```

Append to `tests/test_calibration_loss.py`:

```python
def test_zero_mask_teacher_mode_policy_ce_is_zero():
    """v6 value-only continuation batch: mask all zeros -> 14-tuple,
    policy_ce exactly 0, n_retention 0, all terms finite."""
    import math
    net = create_network(hidden=64, n_blocks=2)
    pos = [_main_pos() for _ in range(3)]
    calib = [_calib_pos(-0.25), _calib_pos(-0.25)]
    mask = np.zeros((2,), dtype=np.float32)
    out = alphazero_loss_batch(
        net, pos,
        calibration_positions=calib,
        calibration_loss_weight=0.01,
        calibration_teacher_policy_mask=mask,
        teacher_value_weight=1.0, teacher_policy_kl_weight=0.25,
    )
    assert len(out) == 14
    value_term = float(out[CALIB_VALUE_TERM_IDX])
    policy_ce = float(out[11])       # CALIB_POLICY_CE_IDX
    n_retention = int(out[13])       # CALIB_N_RETENTION_IDX
    assert math.isfinite(value_term)
    assert policy_ce == 0.0
    assert n_retention == 0
```

- [ ] **Step 2: Run tests to verify the gate test fails**

Run: `.venv/bin/python -m pytest tests/test_trainer_teacher_mode_gate.py tests/test_calibration_loss.py -v`
Expected: `test_trainer_gates_on_teacher_mode_loss_modes` FAILS (trainer still uses `RETENTION_POLICY_LOSS_MODES` at both sites); the zero-mask loss test should PASS already (denominator is clamped) — if it fails, STOP and investigate before touching the trainer.

- [ ] **Step 3: Edit the two trainer call sites**

At `trainer.py:2882` change the import and both uses:

```python
        from .calibration_pool import CalibrationPool, TEACHER_MODE_LOSS_MODES
```

At `trainer.py:2889-2891`:

```python
        if _calib_pool.schema in TEACHER_MODE_LOSS_MODES:
            _n_retention = sum(1 for _s in _calib_pool._samples
                               if _s.loss_mode in TEACHER_MODE_LOSS_MODES)
```

At `trainer.py:3909-3912` extend the local import:

```python
                            from .calibration_pool import (
                                split_samples, split_samples_with_modes,
                                TEACHER_MODE_LOSS_MODES)
```

At `trainer.py:3923`:

```python
                            if _calib_pool.schema in TEACHER_MODE_LOSS_MODES:
```

(The counting line 2891 now counts continuation rows as retention rows in the
startup print — intended. `RETENTION_POLICY_LOSS_MODES` is no longer referenced
in either block after these edits, so the imports above intentionally list only
`CalibrationPool, TEACHER_MODE_LOSS_MODES` and
`split_samples, split_samples_with_modes, TEACHER_MODE_LOSS_MODES`.)

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_trainer_teacher_mode_gate.py tests/test_calibration_loss.py tests/test_value_calibration_sampling.py tests/test_calibration_cli_flags.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py tests/test_trainer_teacher_mode_gate.py tests/test_calibration_loss.py
git commit -m "feat(calibration): route searched_continuation_retention pools down the masked teacher-mode train path"
```

---

### Task 3: `MCTS.search_with_root()` — gate-faithful search that returns the tree

**Files:**
- Modify: `scripts/GPU/alphazero/mcts.py:410-455`
- Test: `tests/test_mcts_search_with_root.py` (create)

**Interfaces:**
- Consumes: existing `MCTS.search` body, `MCTSNode`.
- Produces: `MCTS.search_with_root(root_state, add_noise=True) -> Tuple[Dict[Tuple[int,int],int], float, MCTSNode]` — identical semantics to `search()` (same per-sim `_run_single_simulation` loop, same visit-count assembly, same `_capture_final_root_stats`) plus the root node. `search()` delegates to it and returns the first two elements — behaviorally byte-identical.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mcts_search_with_root.py`:

```python
"""search_with_root must be the SAME search as search() (the gate-faithful
synchronous path), returning the root node as a third element. NOT
search_from_root (different, batched leaf-eval path — forbidden for target
generation; see the v5 path diagnostic)."""
import random

import numpy as np


def _mcts(sims=50, seed=42):
    import mlx.core as mx
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    np.random.seed(7)
    mx.random.seed(7)
    net = create_network(hidden=64, n_blocks=2)
    evaluator = LocalGPUEvaluator(net)
    return (MCTS(evaluator, MCTSConfig(n_simulations=sims), rng=random.Random(seed)),
            MCTS(evaluator, MCTSConfig(n_simulations=sims), rng=random.Random(seed)))


def test_search_with_root_matches_search_and_exposes_tree():
    from scripts.GPU.alphazero.game import TwixtState
    from scripts.GPU.alphazero.mcts import MCTSNode, decode_move
    m1, m2 = _mcts()
    state = TwixtState()
    counts_a, value_a = m1.search(state, add_noise=False)
    counts_b, value_b, root = m2.search_with_root(state, add_noise=False)
    assert counts_a == counts_b
    assert value_a == value_b
    assert isinstance(root, MCTSNode)
    # the returned tree IS the searched tree: child visits match the counts dict
    for move_id, child in root.children.items():
        assert counts_b[decode_move(move_id)] == child.visit_count
    # walkable: some expanded child carries state + nn_value
    visited = [c for c in root.children.values() if c.visit_count > 0]
    assert visited, "no visited children after 50 sims"
    top = max(visited, key=lambda c: c.visit_count)
    assert top.is_expanded and top.nn_value is not None
    assert top.state.to_move != state.to_move
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mcts_search_with_root.py -v`
Expected: FAIL — `AttributeError: 'MCTS' object has no attribute 'search_with_root'`

- [ ] **Step 3: Refactor `mcts.py`**

Rename the existing `search` (lines 410-455) to `search_with_root`, change its
return annotation to `Tuple[Dict[Tuple[int, int], int], float, "MCTSNode"]`, and
change its final line to:

```python
        return visit_counts, root.q_value, root
```

Extend its docstring first line with: `Gate-faithful search returning the root
node for tree extraction (v6 builder). Same synchronous per-sim path as
search(); NOT search_from_root's batched waiter path.`

Then add a new `search` delegating wrapper in its old position:

```python
    def search(
        self,
        root_state: TwixtState,
        add_noise: bool = True,
    ) -> Tuple[Dict[Tuple[int, int], int], float]:
        """Run MCTS from given state.

        Args:
            root_state: Current game state
            add_noise: Whether to add Dirichlet noise at root (for training)

        Returns:
            visit_counts: Dict mapping (row, col) tuple -> visit count (decoded for callers)
            root_value: Estimated value of position for current player
        """
        visit_counts, root_value, _root = self.search_with_root(root_state, add_noise)
        return visit_counts, root_value
```

- [ ] **Step 4: Run the new test plus the whole MCTS test surface**

Run: `.venv/bin/python -m pytest tests/test_mcts_search_with_root.py tests/test_mcts.py tests/test_mcts_force_root_visits.py -v`
Expected: ALL PASS (delegation is behaviorally identical)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/mcts.py tests/test_mcts_search_with_root.py
git commit -m "feat(mcts): search_with_root() — gate-faithful search returning the root node (search() delegates)"
```

---

### Task 4: Continuation extraction — pure tree walk

**Files:**
- Create: `scripts/GPU/alphazero/continuation_extraction.py`
- Test: `tests/test_continuation_extraction.py` (create)

**Interfaces:**
- Consumes: `MCTSNode` (`.children: Dict[int, MCTSNode]`, `.visit_count`, `.state`, `.nn_value`, `.move`, `.parent`, `.is_expanded`), `decode_move(move_id) -> (r, c)` (`mcts.py:74`). NO MLX import — this module must import cleanly without Metal (builder tests run with fakes).
- Produces (Task 5 relies on these exact names):

```python
FAMILY_BY_SOURCE_TAG = {"goal_line_retention": "B",
                        "old_post_opening_retention": "C",
                        "red_predrop_retention": "D"}
CONTINUATION_TAG_BY_SOURCE_TAG = {
    "goal_line_retention": "goal_line_continuation_retention",
    "old_post_opening_retention": "old_post_opening_continuation_retention",
    "red_predrop_retention": "red_predrop_continuation_retention"}

@dataclass(frozen=True)
class ContinuationSpec:
    path_moves: tuple      # ((r, c), ...) root -> continuation, len == depth
    source: str            # "pv" | "top_child" | "child_pv"
    depth: int
    tree_visits: int       # node.visit_count in the root search
    tree_nn_value: float | None   # node.nn_value (train-mode BN; provenance only)
    state: object          # TwixtState at the continuation

def extract_continuations(root, source_tag, *, b_pv_depth=2, c_pv_depth=3,
                          d_top_k=3, d_child_pv_depth=1,
                          d_child_pv_min_visits=40,
                          max_per_root=6) -> list[ContinuationSpec]
def path_moves_of(node) -> tuple            # walk .parent/.move, decode_move
def format_path_moves(path_moves) -> str    # "19:9>18:11"
def case_path_token(path_moves) -> str      # "19-9_18-11"
def continuation_case_id(parent_case_id, spec) -> str
    # f"{parent}__cont_{spec.source}{spec.depth}_{case_path_token(...)}"
def root_max_visit_share(root) -> float     # top child visits / total child visits
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_continuation_extraction.py`:

```python
import pytest

from scripts.GPU.alphazero.continuation_extraction import (
    ContinuationSpec, case_path_token, continuation_case_id,
    extract_continuations, format_path_moves, path_moves_of,
    root_max_visit_share)
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
from scripts.GPU.alphazero.mcts import MCTSNode, encode_move
from tests.goal_line_probe_fixtures import legal_replay


def _child(parent, move_rc, visits, nn_value=0.1, expanded=True):
    node = MCTSNode(state=parent.state.apply_move(move_rc), parent=parent,
                    move=encode_move(*move_rc), visit_count=visits,
                    nn_value=nn_value if expanded else None,
                    priors={} if expanded else None)
    parent.children[node.move] = node
    return node


def _root():
    state = position_state(legal_replay(9, game_idx=1), 5, "black")
    return MCTSNode(state=state, visit_count=400, priors={})


def _tree_sharp():
    """PV chain a > b > c plus a small sibling."""
    root = _root()
    legal = root.state.legal_moves()
    a = _child(root, legal[0], 300, nn_value=-0.4)
    _child(root, legal[1], 100, nn_value=0.2)
    b = _child(a, a.state.legal_moves()[0], 200, nn_value=0.3)
    c = _child(b, b.state.legal_moves()[0], 120, nn_value=-0.2)
    _child(c, c.state.legal_moves()[0], 60, nn_value=0.0)
    return root, legal


def test_c_family_pv_depth_3():
    root, legal = _tree_sharp()
    specs = extract_continuations(root, "old_post_opening_retention")
    assert [s.source for s in specs] == ["pv", "pv", "pv"]
    assert [s.depth for s in specs] == [1, 2, 3]
    assert specs[0].path_moves == (legal[0],)
    assert specs[0].tree_visits == 300
    assert specs[0].tree_nn_value == pytest.approx(-0.4)
    assert len(specs[1].path_moves) == 2 and len(specs[2].path_moves) == 3
    # states are the node states (side alternates from black root)
    assert specs[0].state.to_move == "red"
    assert specs[1].state.to_move == "black"


def test_b_family_pv_depth_2():
    root, _ = _tree_sharp()
    specs = extract_continuations(root, "goal_line_retention")
    assert [s.depth for s in specs] == [1, 2]


def test_d_family_top_k_and_gated_child_pv():
    root = _root()
    legal = root.state.legal_moves()
    c1 = _child(root, legal[0], 150, nn_value=0.1)   # >= 40 -> child_pv allowed
    c2 = _child(root, legal[1], 30, nn_value=0.2)    # < 40  -> no child_pv
    c3 = _child(root, legal[2], 20, nn_value=0.3)
    _child(root, legal[3], 5, nn_value=0.4)          # rank 4 -> not in top-3
    g = _child(c1, c1.state.legal_moves()[0], 90, nn_value=-0.6)
    _child(c2, c2.state.legal_moves()[0], 25, nn_value=0.0)
    specs = extract_continuations(root, "red_predrop_retention")
    by_source = {}
    for s in specs:
        by_source.setdefault(s.source, []).append(s)
    assert len(by_source["top_child"]) == 3
    assert [s.tree_visits for s in by_source["top_child"]] == [150, 30, 20]
    assert len(by_source["child_pv"]) == 1            # only under the 150-visit child
    assert by_source["child_pv"][0].tree_visits == 90
    assert by_source["child_pv"][0].depth == 2


def test_unexpanded_and_terminal_nodes_are_skipped():
    root, legal = _tree_sharp()
    # deepest child unexpanded -> PV stops at depth reached so far
    deep = root.children[encode_move(*legal[0])]
    for _ in range(2):
        deep = max(deep.children.values(), key=lambda n: n.visit_count)
    deep.children.clear()
    deep.priors = None      # unexpanded
    deep.nn_value = None
    specs = extract_continuations(root, "old_post_opening_retention")
    assert [s.depth for s in specs] == [1, 2]         # depth-3 skipped


def test_max_per_root_hard_fails():
    root = _root()
    legal = root.state.legal_moves()
    for i in range(4):
        c = _child(root, legal[i], 100 - i, nn_value=0.0)
        _child(c, c.state.legal_moves()[0], 50, nn_value=0.0)
    with pytest.raises(ValueError, match="max_per_root"):
        extract_continuations(root, "red_predrop_retention",
                              d_top_k=4, max_per_root=6)


def test_path_helpers_and_case_id():
    root, legal = _tree_sharp()
    a = root.children[encode_move(*legal[0])]
    b = max(a.children.values(), key=lambda n: n.visit_count)
    path = path_moves_of(b)
    assert path[0] == legal[0] and len(path) == 2
    (r1, c1), (r2, c2) = path
    assert format_path_moves(path) == f"{r1}:{c1}>{r2}:{c2}"
    assert case_path_token(path) == f"{r1}-{c1}_{r2}-{c2}"
    spec = ContinuationSpec(path_moves=path, source="pv", depth=2,
                            tree_visits=b.visit_count, tree_nn_value=b.nn_value,
                            state=b.state)
    assert continuation_case_id("game_000433_ply_029", spec) == (
        f"game_000433_ply_029__cont_pv2_{case_path_token(path)}")


def test_root_max_visit_share():
    root, _ = _tree_sharp()
    assert root_max_visit_share(root) == pytest.approx(300 / 400)


def test_unknown_tag_raises():
    root, _ = _tree_sharp()
    with pytest.raises(ValueError, match="tag"):
        extract_continuations(root, "black_predrop_correction")


def test_module_has_no_mlx_import():
    import scripts.GPU.alphazero.continuation_extraction as m
    src = open(m.__file__).read()
    assert "import mlx" not in src
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_continuation_extraction.py -v`
Expected: FAIL — `ModuleNotFoundError: ...continuation_extraction`

- [ ] **Step 3: Implement `scripts/GPU/alphazero/continuation_extraction.py`**

```python
"""Pure tree-walk extraction of searched-continuation states (v6).

Consumes a root MCTSNode from MCTS.search_with_root (gate-faithful search) and
returns ContinuationSpecs per the spec's tag-based rules:
  B goal_line_retention          -> PV depth 1-2
  C old_post_opening_retention   -> PV depth 1-3
  D red_predrop_retention        -> top-k root children (+1 PV step below a
                                    child only when its subtree visits pass
                                    d_child_pv_min_visits)
Eligibility: a node is extractable only if it was expanded during search
(is_expanded) and is non-terminal (has legal moves). tree_nn_value comes from
the TRAIN-mode search evaluator — provenance only, never a training target.
NO MLX imports here: the builder's tests run with fakes.
"""
from __future__ import annotations

from dataclasses import dataclass

from .mcts import decode_move

FAMILY_BY_SOURCE_TAG = {
    "goal_line_retention": "B",
    "old_post_opening_retention": "C",
    "red_predrop_retention": "D",
}
CONTINUATION_TAG_BY_SOURCE_TAG = {
    "goal_line_retention": "goal_line_continuation_retention",
    "old_post_opening_retention": "old_post_opening_continuation_retention",
    "red_predrop_retention": "red_predrop_continuation_retention",
}


@dataclass(frozen=True)
class ContinuationSpec:
    path_moves: tuple            # ((r, c), ...) root -> continuation
    source: str                  # "pv" | "top_child" | "child_pv"
    depth: int                   # == len(path_moves)
    tree_visits: int
    tree_nn_value: float | None  # train-mode BN; provenance ONLY
    state: object                # TwixtState at the continuation


def _eligible(node) -> bool:
    """Expanded during search and non-terminal."""
    return node.is_expanded and len(node.state.legal_moves()) > 0


def _best_child(node):
    """Max-visit child (ties: lowest encoded move id); None if no visited child."""
    visited = [c for c in node.children.values() if c.visit_count > 0]
    if not visited:
        return None
    return min(visited, key=lambda c: (-c.visit_count, c.move))


def _top_children(node, k: int) -> list:
    visited = [c for c in node.children.values() if c.visit_count > 0]
    return sorted(visited, key=lambda c: (-c.visit_count, c.move))[:k]


def path_moves_of(node) -> tuple:
    """(r, c) moves from the root to this node, via parent links."""
    moves = []
    while node.parent is not None:
        moves.append(decode_move(node.move))
        node = node.parent
    return tuple(reversed(moves))


def format_path_moves(path_moves) -> str:
    return ">".join(f"{r}:{c}" for r, c in path_moves)


def case_path_token(path_moves) -> str:
    return "_".join(f"{r}-{c}" for r, c in path_moves)


def continuation_case_id(parent_case_id: str, spec: ContinuationSpec) -> str:
    return (f"{parent_case_id}__cont_{spec.source}{spec.depth}_"
            f"{case_path_token(spec.path_moves)}")


def root_max_visit_share(root) -> float:
    total = sum(c.visit_count for c in root.children.values())
    if total <= 0:
        return 0.0
    return max(c.visit_count for c in root.children.values()) / total


def _spec_for(node, source: str) -> ContinuationSpec:
    path = path_moves_of(node)
    return ContinuationSpec(path_moves=path, source=source, depth=len(path),
                            tree_visits=node.visit_count,
                            tree_nn_value=node.nn_value, state=node.state)


def _pv_specs(root, max_depth: int) -> list:
    specs, node = [], root
    for _ in range(max_depth):
        child = _best_child(node)
        if child is None or not _eligible(child):
            break
        specs.append(_spec_for(child, "pv"))
        node = child
    return specs


def extract_continuations(root, source_tag: str, *, b_pv_depth: int = 2,
                          c_pv_depth: int = 3, d_top_k: int = 3,
                          d_child_pv_depth: int = 1,
                          d_child_pv_min_visits: int = 40,
                          max_per_root: int = 6) -> list:
    family = FAMILY_BY_SOURCE_TAG.get(source_tag)
    if family is None:
        raise ValueError(f"not an extraction-source tag: {source_tag!r}")
    if family == "B":
        specs = _pv_specs(root, b_pv_depth)
    elif family == "C":
        specs = _pv_specs(root, c_pv_depth)
    else:                                   # D
        specs = []
        for child in _top_children(root, d_top_k):
            if not _eligible(child):
                continue
            specs.append(_spec_for(child, "top_child"))
            if child.visit_count < d_child_pv_min_visits:
                continue
            node = child
            for _ in range(d_child_pv_depth):
                grand = _best_child(node)
                if grand is None or not _eligible(grand):
                    break
                specs.append(_spec_for(grand, "child_pv"))
                node = grand
    if len(specs) > max_per_root:
        raise ValueError(
            f"{source_tag}: {len(specs)} continuations exceed max_per_root "
            f"{max_per_root}")
    return specs
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_continuation_extraction.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/continuation_extraction.py tests/test_continuation_extraction.py
git commit -m "feat(calibration): continuation_extraction — pure tree-walk PV/top-k extraction for v6"
```

---

### Task 5: v6 manifest builder

**Files:**
- Create: `scripts/GPU/alphazero/build_searched_continuation_retention_manifest.py`
- Test: `tests/test_build_searched_continuation_retention_manifest.py` (create)

**Interfaces:**
- Consumes: Task 4's extraction API; from the v5 builder module (DRY imports): `row_seed`, `_to_black`, `cross_check_gate_values`, `output_fieldnames`; `_teacher_infer(state, evaluator) -> (legal, dense_priors, value)` (`build_teacher_calibration_manifest.py:28`); `legal_moves_sha1`, `CONTINUATION_LOSS_MODE` (Task 1); `MCTS.search_with_root` (Task 3).
- Produces: CLI `python -m scripts.GPU.alphazero.build_searched_continuation_retention_manifest`; testable core `build_rows_v6(rows, raw_evaluator, search_fn, *, params) -> (out_rows, stats)` where `search_fn(state, seed) -> (counts, value_stm, root_node)`.

Row classification (spec §6): `hard_value` + tag `black_predrop_correction` → pass through; `mcts_root_retention` + tag in `FAMILY_BY_SOURCE_TAG` → pass through AND extract; `searched_continuation_retention` + tag in `CONTINUATION_TAG_BY_SOURCE_TAG.values()` → pass through (rerun safety); anything else → `ValueError`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_searched_continuation_retention_manifest.py`:

```python
import json

import numpy as np
import pytest

from scripts.GPU.alphazero.build_searched_continuation_retention_manifest import (
    NEW_COLUMNS_V6, build_rows_v6, classify_row)
from scripts.GPU.alphazero.calibration_pool import (
    CONTINUATION_LOSS_MODE, build_calibration_sample, legal_moves_sha1)
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
from scripts.GPU.alphazero.mcts import MCTSNode, encode_move
from tests.goal_line_probe_fixtures import legal_replay


class _FakeRawEval:
    def build_input_tensor(self, state):
        return state.to_tensor()
    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        b, m = move_mask.shape
        priors = move_mask / np.maximum(move_mask.sum(axis=1, keepdims=True), 1.0)
        return priors.astype(np.float32), np.full((b,), -0.25, dtype=np.float32)


def _child(parent, move_rc, visits, nn_value=0.1):
    node = MCTSNode(state=parent.state.apply_move(move_rc), parent=parent,
                    move=encode_move(*move_rc), visit_count=visits,
                    nn_value=nn_value, priors={})
    parent.children[node.move] = node
    return node


def _fake_search(state, seed):
    """Deterministic tree: PV chain depth 3 + one sibling. Root value -0.1389 stm."""
    root = MCTSNode(state=state, visit_count=400, priors={})
    legal = state.legal_moves()
    a = _child(root, legal[0], 300, nn_value=-0.4)
    _child(root, legal[1], 99, nn_value=0.2)
    b = _child(a, a.state.legal_moves()[0], 200, nn_value=0.3)
    _child(b, b.state.legal_moves()[0], 120, nn_value=-0.2)
    counts = {m: 0 for m in legal}
    counts[legal[0]] = 300
    counts[legal[1]] = 99
    return counts, -0.1389, root


def _rows(tmp_path):
    rp = tmp_path / "game_000001.json"
    replay = legal_replay(9, game_idx=1)
    rp.write_text(json.dumps(replay))
    state = position_state(replay, 5, "black")
    legal = state.legal_moves()
    dense = [0.0] * len(legal); dense[0] = 300 / 399; dense[1] = 99 / 399
    common = {"replay_path": str(rp), "position_ply": "5", "side_to_move": "black",
              "weight_scale": "1.0"}
    return [
        {"game_idx": "1", "case_id": "corr1",
         "tag": "black_predrop_correction", "loss_mode": "hard_value",
         "target_black_value": "-0.35", **common},
        {"game_idx": "1", "case_id": "game_000001_ply_005",
         "tag": "old_post_opening_retention", "loss_mode": "mcts_root_retention",
         "teacher_value": "-0.11", "target_black_value": "",
         "root_visits_json": json.dumps(dense),
         "root_legal_moves_sha1": legal_moves_sha1(legal),
         "root_value_stm": "-0.1389", "root_black_value": "-0.1389",
         "root_sims": "400", "root_seed": str(20260616 ^ 1 ^ 5),
         "root_base_checkpoint": "ckpt/base.safetensors",
         "root_mcts_eval_batch_size": "14", "root_mcts_stall_flush_sims": "48",
         **common},
    ]


def _build(tmp_path, **kw):
    params = dict(pos_base_seed=20260616, goal_base_seed=20260614,
                  b_pv_depth=2, c_pv_depth=3, d_top_k=3, d_child_pv_depth=1,
                  d_child_pv_min_visits=40, max_per_root=6, max_total=250,
                  emit_policy=False, source_root_tolerance=1e-3,
                  limit_cases=None, only_case_ids=None)
    params.update(kw)
    return build_rows_v6(_rows(tmp_path), _FakeRawEval(), _fake_search, **params)


def test_classify_row():
    assert classify_row({"loss_mode": "hard_value",
                         "tag": "black_predrop_correction"}) == "passthrough"
    assert classify_row({"loss_mode": "mcts_root_retention",
                         "tag": "old_post_opening_retention"}) == "extract"
    assert classify_row({"loss_mode": CONTINUATION_LOSS_MODE,
                         "tag": "old_post_opening_continuation_retention"}) == "passthrough"
    with pytest.raises(ValueError, match="unknown"):
        classify_row({"loss_mode": "mcts_root_retention", "tag": "mystery"})


def test_passthrough_and_continuation_rows(tmp_path):
    out, stats = _build(tmp_path)
    # source rows unchanged and first, C continuations appended after their parent
    assert out[0]["case_id"] == "corr1" and out[0]["loss_mode"] == "hard_value"
    assert out[1]["case_id"] == "game_000001_ply_005"
    assert out[1]["root_visits_json"] != ""            # untouched passthrough
    conts = out[2:]
    assert len(conts) == 3                             # C family: PV depth 1-3
    for depth, row in enumerate(conts, start=1):
        assert row["loss_mode"] == CONTINUATION_LOSS_MODE
        assert row["tag"] == "old_post_opening_continuation_retention"
        assert row["continuation_parent_case_id"] == "game_000001_ply_005"
        assert row["continuation_source"] == "pv"
        assert int(row["continuation_depth"]) == depth
        assert row["teacher_value_source"] == "base_raw_continuation"
        assert abs(float(row["teacher_value"]) - (-0.25)) < 1e-6   # fresh eval fwd
        assert row["target_black_value"] == "" and row["root_visits_json"] == ""
        assert row["teacher_policy_json"] == ""
        assert row["weight_scale"] == "1.0"
        moves = json.loads(row["extra_moves_json"])
        assert len(moves) == depth
    # tree provenance recorded
    assert int(conts[0]["continuation_tree_visits"]) == 300
    assert abs(float(conts[0]["continuation_tree_nn_value"]) - (-0.4)) < 1e-9
    assert stats["n_continuation"] == 3
    assert stats["by_tag"]["old_post_opening_continuation_retention"] == 3


def test_continuation_rows_load_through_the_pool(tmp_path):
    out, _ = _build(tmp_path)
    for row in out[2:]:
        sample = build_calibration_sample(row, calibration_target=-0.35)
        assert sample.loss_mode == CONTINUATION_LOSS_MODE
        assert sample.has_policy_target is False
        assert sample.record.outcome == pytest.approx(-0.25)


def test_case_ids_unique_and_deterministic(tmp_path):
    out1, _ = _build(tmp_path)
    out2, _ = _build(tmp_path)
    assert out1 == out2                                # byte-identical rebuild
    ids = [r["case_id"] for r in out1]
    assert len(ids) == len(set(ids))


def test_source_root_value_mismatch_fails(tmp_path):
    rows = _rows(tmp_path)
    rows[1]["root_black_value"] = "0.9"                # stored v5 value disagrees
    with pytest.raises(ValueError, match="source root value"):
        build_rows_v6(rows, _FakeRawEval(), _fake_search,
                      pos_base_seed=20260616, goal_base_seed=20260614,
                      b_pv_depth=2, c_pv_depth=3,
                      d_top_k=3, d_child_pv_depth=1, d_child_pv_min_visits=40,
                      max_per_root=6, max_total=250, emit_policy=False,
                      source_root_tolerance=1e-3, limit_cases=None,
                      only_case_ids=None)


def test_total_cap_hard_fails(tmp_path):
    with pytest.raises(ValueError, match="max_total"):
        _build(tmp_path, max_total=2)


def test_emit_policy_writes_normalized_teacher_policy(tmp_path):
    out, _ = _build(tmp_path, emit_policy=True)
    row = out[2]
    policy = json.loads(row["teacher_policy_json"])
    assert abs(sum(policy) - 1.0) < 1e-6
    sample = build_calibration_sample(row, calibration_target=-0.35)
    assert sample.has_policy_target is True


def test_limit_and_only_case_id_filters(tmp_path):
    out, stats = _build(tmp_path, limit_cases=0)
    assert stats["n_continuation"] == 0 and len(out) == 2   # passthrough only
    out, stats = _build(tmp_path, only_case_ids={"game_000001_ply_005"})
    assert stats["n_continuation"] == 3
    out, stats = _build(tmp_path, only_case_ids={"nope"})
    assert stats["n_continuation"] == 0


def test_module_defers_heavy_imports():
    import importlib
    # NOTE: never pop mlx from sys.modules — native re-import SIGABRTs.
    m = importlib.import_module(
        "scripts.GPU.alphazero.build_searched_continuation_retention_manifest")
    head = open(m.__file__).read().split("def ", 1)[0]
    assert "eval_runner" not in head and "local_evaluator" not in head
    assert "probe_eval" not in head and "import mlx" not in head
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_build_searched_continuation_retention_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the builder**

Create `scripts/GPU/alphazero/build_searched_continuation_retention_manifest.py`:

```python
"""Deterministic v6 searched-continuation retention manifest builder.

Source = the v5 manifest (targeted_calibration_v5_mcts_root_from_calib020_0001.csv).
Every source row passes through UNCHANGED. Additionally, each B/C/D
mcts_root_retention row is re-searched with the gate-faithful config
(search_with_root: same synchronous path as the gates; NEVER search_from_root)
and continuation rows are extracted per the spec's tag-based rules and appended
immediately after their parent row.

Each continuation row anchors a fresh EVAL-mode raw teacher value at the
continuation state (_teacher_infer on a separate eval() network — the tree's
train-mode nn_value is provenance only). Policy columns stay blank unless
--emit-continuation-policy.

Cross-checks (all hard failures):
  - fresh root_black_value vs the SOURCE row's stored root_black_value
    (--source-root-tolerance) — proves the v6 rebuild reproduces v5's search;
  - optional --gate-cases-csv cross-check (reuses the v5 builder's
    cross_check_gate_values with the fresh values);
  - per-root and total continuation caps; case_id uniqueness.

See docs/superpowers/specs/2026-07-02-targeted-value-calibration-v6-searched-
continuation-retention-design.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from .position_probe_cases import load_csv_manifest
from .goal_line_trigger_probe_cases import position_state
from .build_teacher_calibration_manifest import _teacher_infer
from .build_mcts_root_retention_manifest import (
    cross_check_gate_values, output_fieldnames, row_seed, _to_black)
from .calibration_pool import CONTINUATION_LOSS_MODE, legal_moves_sha1
from .continuation_extraction import (
    CONTINUATION_TAG_BY_SOURCE_TAG, FAMILY_BY_SOURCE_TAG, continuation_case_id,
    extract_continuations, format_path_moves, root_max_visit_share)

CORRECTION_TAG = "black_predrop_correction"
NEW_COLUMNS_V6 = [
    "extra_moves_json", "continuation_side_to_move",
    "continuation_legal_moves_sha1", "continuation_depth",
    "continuation_parent_case_id", "continuation_source",
    "continuation_path_moves", "continuation_tree_visits",
    "continuation_tree_nn_value", "teacher_value_source",
]
# Expected family shapes for the D2 telemetry warnings (not row gates):
_SHARP_WARN = {"old_post_opening_retention": 0.65}   # C expected sharp (>=)
_DIFFUSE_WARN = {"red_predrop_retention": 0.65}      # D expected diffuse (<)


def classify_row(r: dict) -> str:
    tag = r.get("tag", "")
    mode = r.get("loss_mode") or "hard_value"
    if mode == "hard_value" and tag == CORRECTION_TAG:
        return "passthrough"
    if mode == "mcts_root_retention" and tag in FAMILY_BY_SOURCE_TAG:
        return "extract"
    if (mode == CONTINUATION_LOSS_MODE
            and tag in CONTINUATION_TAG_BY_SOURCE_TAG.values()):
        return "passthrough"                        # rerun on a v6 output
    raise ValueError(f"{r.get('case_id')}: unknown loss_mode/tag combination "
                     f"({mode!r}, {tag!r})")


def _continuation_row(parent: dict, spec, raw_evaluator, emit_policy: bool) -> dict:
    row = dict(parent)                              # inherit ALL parent columns
    for c in NEW_COLUMNS_V6:
        row.setdefault(c, "")
    legal_c, priors_c, raw_value = _teacher_infer(spec.state, raw_evaluator)
    row["case_id"] = continuation_case_id(parent["case_id"], spec)
    row["tag"] = CONTINUATION_TAG_BY_SOURCE_TAG[parent["tag"]]
    row["loss_mode"] = CONTINUATION_LOSS_MODE
    row["teacher_value"] = repr(float(raw_value))
    row["teacher_value_source"] = "base_raw_continuation"
    row["extra_moves_json"] = json.dumps(
        [{"row": r, "col": c} for (r, c) in spec.path_moves])
    row["continuation_side_to_move"] = spec.state.to_move
    row["continuation_legal_moves_sha1"] = legal_moves_sha1(legal_c)
    row["continuation_depth"] = str(spec.depth)
    row["continuation_parent_case_id"] = parent["case_id"]
    row["continuation_source"] = spec.source
    row["continuation_path_moves"] = format_path_moves(spec.path_moves)
    row["continuation_tree_visits"] = str(spec.tree_visits)
    row["continuation_tree_nn_value"] = (
        "" if spec.tree_nn_value is None else repr(float(spec.tree_nn_value)))
    row["target_black_value"] = ""                  # never a hard target
    row["root_visits_json"] = ""                    # not a root-policy row
    if emit_policy:
        total = sum(priors_c) or 1.0
        row["teacher_policy_json"] = json.dumps([p / total for p in priors_c])
        row["teacher_legal_moves_sha1"] = legal_moves_sha1(legal_c)
    else:
        row["teacher_policy_json"] = ""
        row["teacher_legal_moves_sha1"] = ""
    return row


def build_rows_v6(rows, raw_evaluator, search_fn, *, pos_base_seed,
                  goal_base_seed, b_pv_depth, c_pv_depth, d_top_k,
                  d_child_pv_depth, d_child_pv_min_visits, max_per_root,
                  max_total, emit_policy, source_root_tolerance,
                  limit_cases, only_case_ids):
    # NOTE: continuation rows inherit their parent's root_* provenance stamps
    # (sims/seed/checkpoint/batch/stall) via dict(parent) — the search config
    # itself is proven equivalent by the source-root cross-check, so this
    # function does not take sims/base_checkpoint/batch/stall parameters.
    out, fresh_root_black = [], {}
    stats = {"n_continuation": 0, "by_tag": {}, "excluded": []}
    n_extracted_roots = 0
    for r in rows:
        row = dict(r)
        for c in NEW_COLUMNS_V6:
            row.setdefault(c, "")
        out.append(row)
        if classify_row(r) != "extract":
            continue
        cid = r["case_id"]
        if only_case_ids is not None and cid not in only_case_ids:
            stats["excluded"].append(f"{cid}: not in --only-case-id")
            continue
        if limit_cases is not None and n_extracted_roots >= limit_cases:
            stats["excluded"].append(f"{cid}: past --limit-cases {limit_cases}")
            continue
        n_extracted_roots += 1
        replay = json.loads(Path(r["replay_path"]).read_text())
        ply = int(float(r["position_ply"]))
        side = r["side_to_move"]
        state = position_state(replay, ply, side)
        seed = row_seed(r.get("tag", ""), r["game_idx"], ply,
                        pos_base_seed, goal_base_seed)
        counts, root_value_stm, root = search_fn(state, seed)
        fresh_black = _to_black(root_value_stm, side)
        fresh_root_black[cid] = fresh_black
        stored = r.get("root_black_value")
        if stored not in (None, "") and (
                abs(fresh_black - float(stored)) > source_root_tolerance):
            raise ValueError(
                f"{cid}: source root value mismatch — recomputed "
                f"{fresh_black:+.4f} vs stored {float(stored):+.4f} "
                f"(wrong seeds / BN mode / config?)")
        share = root_max_visit_share(root)
        tag = r["tag"]
        if tag in _SHARP_WARN and share < _SHARP_WARN[tag]:
            print(f"WARNING: {cid}: C root diffuse (max share {share:.3f})")
        if tag in _DIFFUSE_WARN and share >= _DIFFUSE_WARN[tag]:
            print(f"WARNING: {cid}: D root sharp (max share {share:.3f})")
        print(f"{cid}: root max-visit-share {share:.3f}")
        specs = extract_continuations(
            root, tag, b_pv_depth=b_pv_depth, c_pv_depth=c_pv_depth,
            d_top_k=d_top_k, d_child_pv_depth=d_child_pv_depth,
            d_child_pv_min_visits=d_child_pv_min_visits,
            max_per_root=max_per_root)
        for spec in specs:
            crow = _continuation_row(r, spec, raw_evaluator, emit_policy)
            out.append(crow)
            stats["n_continuation"] += 1
            ctag = crow["tag"]
            stats["by_tag"][ctag] = stats["by_tag"].get(ctag, 0) + 1
        if stats["n_continuation"] > max_total:
            raise ValueError(
                f"continuation rows {stats['n_continuation']} exceed max_total "
                f"{max_total} — raise the cap or tighten thresholds "
                f"(operator tuning point, see spec §1 D4)")
    ids = [r["case_id"] for r in out]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate case_ids in output: {dupes}")
    stats["fresh_root_black"] = fresh_root_black
    return out, stats


def _real_search_fn(base_checkpoint: str, sims: int,
                    eval_batch_size: int, stall_flush_sims: int):
    """Gate-faithful root-returning search. Heavy imports deferred (fakes in
    tests). Same evaluator/config as the v5 builder and the gate probes."""
    from .eval_runner import EvalConfig, cfg_from, _default_evaluator_factory
    from .mcts import MCTS
    evaluator = _default_evaluator_factory(base_checkpoint)
    cfg = cfg_from(EvalConfig(mcts_sims=sims,
                              mcts_eval_batch_size=eval_batch_size,
                              mcts_stall_flush_sims=stall_flush_sims))

    def search_fn(state, seed):
        return MCTS(evaluator, cfg, random.Random(seed)).search_with_root(
            state, add_noise=False)

    return search_fn


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build the v6 searched-continuation retention manifest.")
    ap.add_argument("--source", required=True, help="the v5 manifest CSV")
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--position-probe-base-seed", type=int, default=20260616)
    ap.add_argument("--goal-line-base-seed", type=int, default=20260614)
    ap.add_argument("--eval-batch-size", type=int, default=14)
    ap.add_argument("--stall-flush-sims", type=int, default=48)
    ap.add_argument("--b-pv-depth", type=int, default=2)
    ap.add_argument("--c-pv-depth", type=int, default=3)
    ap.add_argument("--d-top-k", type=int, default=3)
    ap.add_argument("--d-child-pv-depth", type=int, default=1)
    ap.add_argument("--d-child-pv-min-visits", type=int, default=40)
    ap.add_argument("--max-continuations-per-root", type=int, default=6)
    ap.add_argument("--max-total-continuation-rows", type=int, default=250)
    ap.add_argument("--emit-continuation-policy", action="store_true",
                    help="also write dense eval-mode teacher policy on "
                         "continuation rows (v6b variant; default OFF = value-only)")
    ap.add_argument("--source-root-tolerance", type=float, default=1e-3)
    ap.add_argument("--gate-cases-csv", action="append", default=[])
    ap.add_argument("--gate-tolerance", type=float, default=1e-3)
    ap.add_argument("--gate-checkpoint-label", default=None)
    ap.add_argument("--limit-cases", type=int, default=None,
                    help="extract from only the first N eligible roots")
    ap.add_argument("--only-case-id", action="append", default=None,
                    help="extract only from these root case_ids (repeatable)")
    args = ap.parse_args(argv)

    from .local_evaluator import LocalGPUEvaluator
    from .probe_eval import load_network_for_scoring
    rows = load_csv_manifest(args.source)["cases"]
    network, *_ = load_network_for_scoring(args.base_checkpoint)
    network.eval()                       # raw anchors: EVAL-mode BN
    raw_evaluator = LocalGPUEvaluator(network)
    search_fn = _real_search_fn(args.base_checkpoint, args.sims,
                                args.eval_batch_size, args.stall_flush_sims)

    out_rows, stats = build_rows_v6(
        rows, raw_evaluator, search_fn,
        pos_base_seed=args.position_probe_base_seed,
        goal_base_seed=args.goal_line_base_seed,
        b_pv_depth=args.b_pv_depth, c_pv_depth=args.c_pv_depth,
        d_top_k=args.d_top_k, d_child_pv_depth=args.d_child_pv_depth,
        d_child_pv_min_visits=args.d_child_pv_min_visits,
        max_per_root=args.max_continuations_per_root,
        max_total=args.max_total_continuation_rows,
        emit_policy=args.emit_continuation_policy,
        source_root_tolerance=args.source_root_tolerance,
        limit_cases=args.limit_cases,
        only_case_ids=set(args.only_case_id) if args.only_case_id else None)

    if args.gate_cases_csv:
        check_rows = [{"loss_mode": "mcts_root_retention", "case_id": cid,
                       "root_black_value": repr(v)}
                      for cid, v in stats["fresh_root_black"].items()]
        gs = cross_check_gate_values(check_rows, args.gate_cases_csv,
                                     args.gate_tolerance,
                                     checkpoint_label=args.gate_checkpoint_label)
        print(f"gate cross-check PASS: {gs['checked']} matched, "
              f"{gs['unmatched']} roots without a gate row")
    else:
        print("WARNING: no --gate-cases-csv given; fresh root values checked "
              "only against the source manifest")

    for line in stats["excluded"]:
        print(f"excluded: {line}")
    base_columns = list(rows[0].keys()) if rows else []
    fieldnames = output_fieldnames(base_columns, out_rows)
    for c in NEW_COLUMNS_V6:
        if c not in fieldnames:
            fieldnames.append(c)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows ({stats['n_continuation']} continuation: "
          f"{stats['by_tag']}) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_build_searched_continuation_retention_manifest.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/build_searched_continuation_retention_manifest.py tests/test_build_searched_continuation_retention_manifest.py
git commit -m "feat(calibration): v6 searched-continuation retention manifest builder (gate-faithful search_with_root, hard caps, source/gate cross-checks)"
```

---

### Task 6: v6 gate-0 smoke

**Files:**
- Create: `scripts/GPU/alphazero/smoke_searched_continuation_retention_v6.py`
- Test: `tests/test_smoke_searched_continuation_retention_v6.py` (create)

**Interfaces:**
- Consumes: `CalibrationPool.from_manifest`, `split_samples_with_modes`, `CONTINUATION_LOSS_MODE` (Task 1); `alphazero_loss_batch` + `CALIB_*_IDX` constants (`trainer.py:54-58`).
- Produces: CLI `python -m scripts.GPU.alphazero.smoke_searched_continuation_retention_v6 --manifest ... --base-checkpoint ...`; testable core `assert_continuation_retention_mechanics(network, manifest_path, value_tol=1e-4) -> dict`.

V6 schedule constant (used by the smoke's hard draw assertion):

```python
V6_TAG_SCHEDULE = {
    "black_predrop_correction": 2,
    "goal_line_continuation_retention": 1,
    "old_post_opening_continuation_retention": 2,
    "red_predrop_continuation_retention": 2,
}
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_smoke_searched_continuation_retention_v6.py` (fixture builds a
tiny v6-shaped manifest CSV using the same helpers as Task 1's tests; network =
`create_network(hidden=64, n_blocks=2)` — teacher values are then WRONG for this
random network, so the value-anchor assertion is tested via a monkeypatched
teacher, mirroring `tests/test_smoke_mcts_root_retention_v5.py`'s approach):

```python
import csv
import json

import numpy as np
import pytest

from scripts.GPU.alphazero.calibration_pool import (
    CONTINUATION_LOSS_MODE, legal_moves_sha1)
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
from scripts.GPU.alphazero.smoke_searched_continuation_retention_v6 import (
    V6_TAG_SCHEDULE, assert_continuation_retention_mechanics)
from tests.goal_line_probe_fixtures import legal_replay


def _manifest(tmp_path, teacher_value):
    """1 hard_value row + 1 inert root row + 3 continuation rows (one per
    continuation tag), all on the same tiny replay."""
    rp = tmp_path / "game_000001.json"
    replay = legal_replay(9, game_idx=1)
    rp.write_text(json.dumps(replay))
    state = position_state(replay, 5, "black")
    legal = state.legal_moves()
    m1 = legal[0]
    s1 = state.apply_move(m1)
    dense = [0.0] * len(legal); dense[0] = 1.0
    common = {"game_idx": "1", "replay_path": str(rp), "position_ply": "5",
              "side_to_move": "black", "weight_scale": "1.0"}
    cont_common = {
        "loss_mode": CONTINUATION_LOSS_MODE,
        "teacher_value": repr(teacher_value),
        "extra_moves_json": json.dumps([{"row": m1[0], "col": m1[1]}]),
        "continuation_side_to_move": s1.to_move,
        "continuation_legal_moves_sha1": legal_moves_sha1(s1.legal_moves()),
        **common}
    rows = [
        {"case_id": "corr1", "tag": "black_predrop_correction",
         "loss_mode": "hard_value", "target_black_value": "-0.35", **common},
        {"case_id": "root1", "tag": "old_post_opening_retention",
         "loss_mode": "mcts_root_retention", "teacher_value": repr(teacher_value),
         "root_visits_json": json.dumps(dense),
         "root_legal_moves_sha1": legal_moves_sha1(legal), **common},
        {"case_id": "b1", "tag": "goal_line_continuation_retention", **cont_common},
        {"case_id": "c1", "tag": "old_post_opening_continuation_retention", **cont_common},
        {"case_id": "d1", "tag": "red_predrop_continuation_retention", **cont_common},
    ]
    fields = sorted({k for r in rows for k in r})
    p = tmp_path / "v6_manifest.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return p


def _network_value(net):
    """The network's actual eval-mode stm value at the continuation state —
    write it back as teacher_value so the anchor reproduces exactly."""
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.build_teacher_calibration_manifest import _teacher_infer
    replay = legal_replay(9, game_idx=1)
    state = position_state(replay, 5, "black")
    s1 = state.apply_move(state.legal_moves()[0])
    prev = net.training
    net.eval()
    try:
        _, _, v = _teacher_infer(s1, LocalGPUEvaluator(net))
    finally:
        net.train(prev)
    return float(v)


def test_smoke_passes_on_reproducing_anchor(tmp_path):
    from scripts.GPU.alphazero.network import create_network
    net = create_network(hidden=64, n_blocks=2)
    v = _network_value(net)
    p = _manifest(tmp_path, teacher_value=v)
    report = assert_continuation_retention_mechanics(net, str(p))
    assert report["n_continuation"] == 3
    assert report["policy_ce"] == 0.0                  # value-only: mask all zero
    assert report["n_policy_rows"] == 0
    assert abs(report["value_mse"]) < 1e-4
    assert report["draws_by_tag"] == V6_TAG_SCHEDULE   # hard schedule assertion


def test_smoke_fails_on_drifted_anchor(tmp_path):
    from scripts.GPU.alphazero.network import create_network
    net = create_network(hidden=64, n_blocks=2)
    v = _network_value(net)
    drifted = max(-1.0, min(1.0, v - 0.5))
    p = _manifest(tmp_path, teacher_value=drifted)
    with pytest.raises(AssertionError, match="value"):
        assert_continuation_retention_mechanics(net, str(p))


def test_smoke_fails_on_wrong_schema(tmp_path):
    """A v5-only manifest (no continuation rows) must be rejected."""
    from scripts.GPU.alphazero.network import create_network
    rp = tmp_path / "game_000001.json"
    replay = legal_replay(9, game_idx=1)
    rp.write_text(json.dumps(replay))
    state = position_state(replay, 5, "black")
    legal = state.legal_moves()
    dense = [0.0] * len(legal); dense[0] = 1.0
    rows = [{"game_idx": "1", "case_id": "root1", "replay_path": str(rp),
             "position_ply": "5", "side_to_move": "black",
             "tag": "old_post_opening_retention",
             "loss_mode": "mcts_root_retention", "teacher_value": "0.0",
             "root_visits_json": json.dumps(dense),
             "root_legal_moves_sha1": legal_moves_sha1(legal)}]
    fields = sorted({k for r in rows for k in r})
    p = tmp_path / "v5_only.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    net = create_network(hidden=64, n_blocks=2)
    with pytest.raises(AssertionError, match="schema"):
        assert_continuation_retention_mechanics(net, str(p))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_smoke_searched_continuation_retention_v6.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the smoke**

Create `scripts/GPU/alphazero/smoke_searched_continuation_retention_v6.py`:

```python
"""v6 gate-0 mechanics smoke (pattern: smoke_mcts_root_retention_v5).

Asserts, at the BASE checkpoint, BEFORE training:
  1. manifest schema == searched_continuation_retention (loading already
     re-applies extra_moves_json and verifies continuation side/sha1 per row);
  2. continuation value anchors reproduce under the eval-mode calibration
     forward: value MSE ~ 0;
  3. per-row policy mask: policy CE == 0.0 exactly on a value-only manifest,
     finite when --emit-continuation-policy rows exist; no NaN anywhere;
  4. HARD schedule assertion: one sample_by_tag round with the locked v6
     schedule draws exactly the scheduled counts per tag (acceptance
     criterion 1 reads calib_n_drawn_by_tag, NOT n_teacher_retention_drawn,
     which is policy-mask-derived and stays 0 on a value-only run).
"""
from __future__ import annotations

import argparse
import math
import random
import sys

from .calibration_pool import (
    CONTINUATION_LOSS_MODE, CalibrationPool, split_samples_with_modes)
from .trainer import (
    alphazero_loss_batch, CALIB_VALUE_TERM_IDX, CALIB_POLICY_CE_IDX,
    CALIB_POLICY_KL_EST_IDX, CALIB_N_RETENTION_IDX)

V6_TAG_SCHEDULE = {
    "black_predrop_correction": 2,
    "goal_line_continuation_retention": 1,
    "old_post_opening_continuation_retention": 2,
    "red_predrop_continuation_retention": 2,
}


def assert_continuation_retention_mechanics(network, manifest_path: str,
                                            value_tol: float = 1e-4) -> dict:
    pool = CalibrationPool.from_manifest(manifest_path, calibration_target=-0.35)
    if pool.schema != CONTINUATION_LOSS_MODE:
        raise AssertionError(
            f"manifest schema is {pool.schema!r}, expected {CONTINUATION_LOSS_MODE}")
    continuation = [s for s in pool._samples
                    if s.loss_mode == CONTINUATION_LOSS_MODE]
    if not continuation:
        raise AssertionError("no searched_continuation_retention rows in manifest")
    records, weights, mask = split_samples_with_modes(
        continuation, pool.has_weight_scale)
    n_policy_rows = int(sum(1 for s in continuation if s.has_policy_target))
    prev_training = network.training
    network.eval()
    try:
        out = alphazero_loss_batch(
            network, records,
            calibration_positions=records,
            calibration_weights=weights,
            calibration_loss_weight=1.0,
            calibration_teacher_policy_mask=mask,
            teacher_value_weight=1.0, teacher_policy_kl_weight=1.0,
        )
        value_mse = float(out[CALIB_VALUE_TERM_IDX])
        policy_ce = float(out[CALIB_POLICY_CE_IDX])
        kl_est = float(out[CALIB_POLICY_KL_EST_IDX])
        n_retention = int(out[CALIB_N_RETENTION_IDX])
    finally:
        network.train(prev_training)
    if not (math.isfinite(value_mse) and math.isfinite(policy_ce)
            and math.isfinite(kl_est)):
        raise AssertionError(
            f"non-finite terms: value_mse={value_mse}, ce={policy_ce}, kl={kl_est}")
    if abs(value_mse) > value_tol:
        raise AssertionError(
            f"continuation value anchor FAILED to reproduce: "
            f"value_mse={value_mse:.3e} (tol={value_tol}). Check eval-mode "
            f"forward / checkpoint / perspective / extra-move reconstruction.")
    if n_policy_rows == 0 and policy_ce != 0.0:
        raise AssertionError(
            f"value-only manifest but policy_ce={policy_ce} != 0 (mask leak?)")
    if n_retention != n_policy_rows:
        raise AssertionError(
            f"mask count {n_retention} != policy-carrying rows {n_policy_rows}")
    # HARD schedule assertion (acceptance criterion 1 source of truth).
    draws = pool.sample_by_tag(V6_TAG_SCHEDULE, random.Random(0))
    draws_by_tag: dict = {}
    for s in draws:
        draws_by_tag[s.tag] = draws_by_tag.get(s.tag, 0) + 1
    if draws_by_tag != V6_TAG_SCHEDULE:
        raise AssertionError(
            f"schedule draw mismatch: {draws_by_tag} != {V6_TAG_SCHEDULE}")
    return {"n_continuation": len(continuation), "n_policy_rows": n_policy_rows,
            "value_mse": value_mse, "policy_ce": policy_ce, "kl_est": kl_est,
            "draws_by_tag": draws_by_tag,
            "tag_counts": pool.tag_counts()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="v6 continuation-retention gate-0 smoke.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--value-tol", type=float, default=1e-4)
    args = ap.parse_args(argv)
    from .probe_eval import load_network_for_scoring
    network, *_ = load_network_for_scoring(args.base_checkpoint)
    report = assert_continuation_retention_mechanics(
        network, args.manifest, value_tol=args.value_tol)
    print(f"PASS v6 continuation retention mechanics: "
          f"n_continuation={report['n_continuation']}, "
          f"value_mse={report['value_mse']:.3e}, "
          f"policy_ce={report['policy_ce']:.4f} "
          f"({report['n_policy_rows']} policy rows), "
          f"draws_by_tag={report['draws_by_tag']}, "
          f"pool_tags={report['tag_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_smoke_searched_continuation_retention_v6.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/smoke_searched_continuation_retention_v6.py tests/test_smoke_searched_continuation_retention_v6.py
git commit -m "feat(calibration): v6 gate-0 smoke — schema, value-anchor reproduction, zero-mask policy CE, hard schedule assertion"
```

---

### Task 7: Full-suite verification

**Files:** none new.

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: baseline 1253 + all new tests pass, 0 failures. If anything fails, fix before proceeding — do NOT merge red.

- [ ] **Step 2: Verify no accidental behavior change for v5 manifests**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py tests/test_smoke_mcts_root_retention_v5.py tests/test_build_mcts_root_retention_manifest.py tests/test_calibration_loss.py -v`
Expected: ALL PASS with those files' pre-existing assertions untouched (only the Task 2 additions to test_calibration_loss.py are new).

- [ ] **Step 3: Commit anything outstanding, then hand off**

Merge/review handled by superpowers:finishing-a-development-branch (FF-merge to main, push).

---

## Operator run (post-merge; NOT part of implementation)

1. **Diagnostic build** on the six v5 path-diagnostic rows first (spec §10.5), inspect by hand:

```bash
.venv/bin/python -m scripts.GPU.alphazero.build_searched_continuation_retention_manifest \
  --source logs/eval/targeted_calibration_v5_mcts_root_from_calib020_0001.csv \
  --base-checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --out logs/eval/targeted_calibration_v6_diag6.csv \
  --only-case-id game_000433_ply_029 --only-case-id game_000065_ply_021 \
  --only-case-id game_000565_ply_033 \
  --only-case-id red_loss_game_000780_predrop_ply_56_drop_58 \
  --only-case-id red_loss_game_000362_predrop_ply_52_drop_54 \
  --only-case-id red_loss_game_000176_predrop_ply_42_drop_44 \
  --gate-cases-csv <BASE position_probe_cases.csv> \
  --gate-cases-csv <BASE goal_line_trigger_probe_cases.csv> \
  --gate-checkpoint-label 0001
```

2. **Full build** (same command, drop `--only-case-id`, `--out logs/eval/targeted_calibration_v6_continuation_from_calib020_0001.csv`).
3. **Smoke**: `.venv/bin/python -m scripts.GPU.alphazero.smoke_searched_continuation_retention_v6 --manifest logs/eval/targeted_calibration_v6_continuation_from_calib020_0001.csv --base-checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`
4. **Train** — the v5 command verbatim with three deltas (spec §8): v6 manifest path, `--checkpoint-dir checkpoints/alphazero-v6-continuation-from-calib020-0001`, `--post-opening-calibration-tag-schedule black_predrop_correction=2,goal_line_continuation_retention=1,old_post_opening_continuation_retention=2,red_predrop_continuation_retention=2`.
5. **Telemetry check**: the training stats sidecar's `draws_by_tag` must show all three `*_continuation_retention` tags in 1:2:2 ratio (`n_teacher_retention_drawn` will be 0 — expected on a value-only run).
6. **Gates A/B/C/D** vs `calib020_0001`, `OUT=logs/eval/v6_continuation_from_calib020_0001_gates_400s`. Acceptance thresholds in spec §8. No promotion unless all four pass.
7. **Ledger update** (result row + do-not-repeat if rejected; spec §9 failure fork).
