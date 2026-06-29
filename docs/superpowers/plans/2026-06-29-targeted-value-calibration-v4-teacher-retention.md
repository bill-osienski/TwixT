# Targeted Value Calibration v4 — Teacher-Retention Anchors — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-06-29-targeted-value-calibration-v4-teacher-retention-design.md` (read it — it carries the rationale, gate table, and math; this plan carries the code).

**Goal:** Add teacher-retention distillation to the post-opening calibration loss — correction rows keep a hard value target; retention rows match a frozen teacher's raw-NN value (MSE) and policy distribution (cross-entropy) — to fix gate A without breaking guardrails B/C/D.

**Architecture:** Option A (spec §3): the dense teacher policy rides in `PositionRecord.visit_counts` (reusing `make_padded_batch` → `target_pi` and `compute_masked_log_probs`), and the teacher value rides in `PositionRecord.outcome`. A new manifest builder caches the teacher's raw `infer` outputs; the calibration loss gains a masked policy-CE term alongside the existing value-MSE term; a gate-0 smoke proves self-distillation (teacher == base ⇒ losses ≈ 0) before any training run.

**Tech Stack:** Python 3.14, MLX (`mlx.core as mx`), NumPy, pytest. Run tests with `.venv/bin/python -m pytest`.

## Global Constraints

Copied verbatim from the spec; every task implicitly includes these.

- **Raw-NN only — no MCTS in the builder.** The builder calls `LocalGPUEvaluator.infer` / `forward_padded` exclusively and MUST NOT import, instantiate, or call `MCTS` (spec §5).
- **`PositionRecord` is unchanged.** Teacher targets ride in existing fields: `outcome` (side-to-move value), `visit_counts` (dense teacher policy aligned to `legal_moves`) (spec §4.1).
- **Schema-gated mask.** `calibration_teacher_policy_mask` is built **only when `pool.schema == "teacher_retention"`**; for `global_target` / `per_row_target` pools it is `None`, which keeps the v2/v3 loss path **byte-identical** (spec §7/§8). Gate on `mask is None`, never on per-row `loss_mode`.
- **CE is the gradient term; telemetry distinguishes CE from KL.** Policy loss is teacher cross-entropy `−Σ teacher_pi·log cand_pi`. Telemetry reports `calib_policy_ce` (headline) and `calib_policy_kl_est = CE − mean teacher entropy` (≈ 0 at teacher-match) (spec §7 clarification 1, §9).
- **Per-row `weight_scale`, separate denominators.** Value weighted mean uses `Σ w` over all calibration rows; policy weighted mean uses `Σ (w·m)` over retention rows only (spec §7 clarification 5).
- **`teacher_value` perspective.** Stored side-to-move and assigned directly to `record.outcome`; never routed through `target_in_to_move`. `target_black_value` on retention rows is debug-only and the builder blanks it (spec §4.1, §5).
- **`hard_value` rows blank.** The three teacher-data columns (`teacher_value`, `teacher_policy_json`, `teacher_legal_moves_sha1`) must be blank on correction rows; a populated column is a validation error (spec §6).
- **Legal-move order is an invariant pinned by SHA-1** over the canonical `";".join(f"{r},{c}" for r,c in legal)` string, shared by builder and loader (spec §4.3, §6).
- TDD, one logical change per commit, exact paths always.

## File Structure

| File | Responsibility |
|---|---|
| `scripts/GPU/alphazero/calibration_pool.py` | (modify) SHA-1 helper; `CalibrationSample` fields; `loss_mode` branch in `build_calibration_position`/`build_calibration_sample`; `schema="teacher_retention"`; validation; `split_samples_with_modes`; telemetry block fields. |
| `scripts/GPU/alphazero/trainer.py` | (modify) `alphazero_loss_batch` value + masked-policy-CE term + telemetry + extended return tuple; `train_step` unpack; `train()` schema-gated mask + weight threading + accumulators + sidecar. |
| `scripts/GPU/alphazero/train.py` | (modify) two CLI flags + thread into `train()`. |
| `scripts/GPU/alphazero/build_teacher_calibration_manifest.py` | (new) deterministic teacher-cache builder (raw `infer`, no MCTS). |
| `scripts/GPU/alphazero/smoke_teacher_calibration_v4.py` | (new) gate-0 pre-flight self-distillation check. |
| `tests/test_calibration_pool.py` | (modify) parser/validation/split tests. |
| `tests/test_calibration_loss.py` | (modify) loss term + return-shape + regression tests. |
| `tests/test_calibration_cli_flags.py` | (modify) flag default/set tests. |
| `tests/test_training.py` | (modify) train() schema-gate + sidecar integration. |
| `tests/test_build_teacher_calibration_manifest.py` | (new) builder round-trip / no-MCTS / blanking tests. |
| `docs/post-game-analysis.md` | (modify) catalog the new builder + smoke. |

Dependency order: Task 1 → 2 → 3 → 4 (calibration_pool), 5 → 6 → 7 → 8 (loss/wiring/CLI/telemetry), 9 (builder), 10 (smoke capstone).

---

### Task 1: Shared canonical legal-move SHA-1 helper

**Files:**
- Modify: `scripts/GPU/alphazero/calibration_pool.py` (add `legal_moves_sha1`)
- Test: `tests/test_calibration_pool.py`

**Interfaces:**
- Produces: `legal_moves_sha1(legal: list[tuple[int, int]]) -> str` — 40-char hex SHA-1 over `";".join(f"{r},{c}" for r,c in legal)`. Order-sensitive. Imported by the loader (Task 3) and builder (Task 9).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_calibration_pool.py`:

```python
def test_legal_moves_sha1_stable_and_order_sensitive():
    from scripts.GPU.alphazero.calibration_pool import legal_moves_sha1
    a = legal_moves_sha1([(0, 0), (1, 2), (3, 4)])
    b = legal_moves_sha1([(0, 0), (1, 2), (3, 4)])
    c = legal_moves_sha1([(1, 2), (0, 0), (3, 4)])  # same length, reordered
    assert a == b                       # deterministic
    assert a != c                       # catches a same-length reorder
    assert len(a) == 40
    assert all(ch in "0123456789abcdef" for ch in a)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py::test_legal_moves_sha1_stable_and_order_sensitive -v`
Expected: FAIL with `ImportError: cannot import name 'legal_moves_sha1'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/GPU/alphazero/calibration_pool.py`, add `import hashlib` to the imports, then add near the top (after the existing imports / before `target_in_to_move`):

```python
def legal_moves_sha1(legal) -> str:
    """SHA-1 over the canonical legal-move ordering. Order-sensitive: pins the
    alignment between teacher_policy_json and legal_moves between build time and
    train time (legal_moves() is sorted/deterministic, so the same reconstructed
    position yields the same hash)."""
    canonical = ";".join(f"{r},{c}" for r, c in legal)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py::test_legal_moves_sha1_stable_and_order_sensitive -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/calibration_pool.py tests/test_calibration_pool.py
git commit -m "feat(calibration): add legal_moves_sha1 alignment fingerprint for v4"
```

---

### Task 2: `CalibrationSample` teacher fields + `loss_mode` parsing + schema

**Files:**
- Modify: `scripts/GPU/alphazero/calibration_pool.py` (`CalibrationSample`, `build_calibration_position`, `build_calibration_sample`, `CalibrationPool.from_manifest`)
- Test: `tests/test_calibration_pool.py`

**Interfaces:**
- Consumes: `legal_moves_sha1` (Task 1).
- Produces:
  - `CalibrationSample` gains `loss_mode: str = "hard_value"`, `teacher_value: float | None = None`, `teacher_policy_len: int | None = None`.
  - `build_calibration_position(case, calibration_target)` branches on `case["loss_mode"]`: `teacher_retention` → `outcome = teacher_value` (side-to-move, direct), `visit_counts = teacher_policy` (list[float]); else unchanged.
  - `_parse_teacher_policy(case, legal) -> list[float]` (validation added in Task 3; here returns the parsed list).
  - `CalibrationPool.from_manifest` sets `schema = "teacher_retention"` when any row has `loss_mode == "teacher_retention"`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_calibration_pool.py` (helpers `_write_case_side`, `legal_replay` already exist):

```python
def _teacher_case(tmp_path, position_ply=5, game_idx=7):
    """A teacher_retention row: black to move (odd ply), with a teacher policy
    aligned to the reconstructed legal_moves order."""
    from scripts.GPU.alphazero.calibration_pool import legal_moves_sha1
    from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
    import json as _json
    case = _write_case_side(tmp_path, "black", position_ply, game_idx=game_idx)
    replay = _json.loads((tmp_path / f"game_{game_idx:06d}.json").read_text())
    state = position_state(replay, position_ply, "black")
    legal = state.legal_moves()
    n = len(legal)
    policy = [1.0 / n] * n                       # uniform teacher policy
    case.update({
        "loss_mode": "teacher_retention",
        "teacher_value": "0.20",                 # side-to-move
        "teacher_policy_json": _json.dumps(policy),
        "teacher_legal_moves_sha1": legal_moves_sha1(legal),
    })
    return case, n


def test_teacher_retention_row_uses_teacher_value_and_policy(tmp_path):
    from scripts.GPU.alphazero.calibration_pool import build_calibration_position
    case, n = _teacher_case(tmp_path)
    rec = build_calibration_position(case, calibration_target=-0.5)
    assert rec.outcome == 0.20                   # teacher_value, NOT through target_in_to_move
    assert len(rec.visit_counts) == n
    assert abs(sum(rec.visit_counts) - 1.0) < 1e-6
    assert rec.to_move == "black"


def test_from_manifest_detects_teacher_schema(tmp_path):
    import csv as _csv
    from scripts.GPU.alphazero.calibration_pool import CalibrationPool
    case, _ = _teacher_case(tmp_path)
    manifest = tmp_path / "v4.csv"
    with manifest.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(case.keys()))
        w.writeheader()
        w.writerow(case)
    pool = CalibrationPool.from_manifest(str(manifest), calibration_target=-0.35)
    assert pool.schema == "teacher_retention"
    assert pool._samples[0].loss_mode == "teacher_retention"
    assert pool._samples[0].teacher_value == 0.20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py -k "teacher_retention_row or teacher_schema" -v`
Expected: FAIL (`build_calibration_position` ignores `loss_mode`; `schema` is `per_row_target`/`global_target`).

- [ ] **Step 3: Write minimal implementation**

In `scripts/GPU/alphazero/calibration_pool.py`:

(a) Extend the dataclass:

```python
@dataclass(frozen=True)
class CalibrationSample:
    record: PositionRecord
    weight_scale: float = 1.0
    tag: str = ""
    target_black_value: float | None = None
    loss_mode: str = "hard_value"            # "hard_value" | "teacher_retention"
    teacher_value: float | None = None        # side-to-move; telemetry/validation
    teacher_policy_len: int | None = None      # == len(legal_moves); validation
```

(b) Add a teacher-policy parser (validation hardened in Task 3):

```python
def _parse_teacher_policy(case: dict, legal) -> list[float]:
    """Parse teacher_policy_json into a dense list aligned to legal_moves."""
    raw = case.get("teacher_policy_json")
    if raw in (None, ""):
        raise ValueError(f"{case.get('case_id')}: teacher_retention row needs teacher_policy_json")
    policy = [float(x) for x in json.loads(raw)]
    return policy
```

(c) Branch `build_calibration_position` on `loss_mode` (keep the existing reconstruct block that produces `state`, `board_hwc`, `legal`; replace only the `return PositionRecord(...)`):

```python
    loss_mode = case.get("loss_mode") or "hard_value"
    if loss_mode == "teacher_retention":
        teacher_value = float(case["teacher_value"])     # side-to-move; direct
        teacher_policy = _parse_teacher_policy(case, legal)
        return PositionRecord(
            board_tensor=board_hwc,
            to_move=state.to_move,
            legal_moves=legal,
            visit_counts=teacher_policy,                 # float "counts"; make_padded_batch normalizes
            outcome=teacher_value,
            active_size=state.active_size,
            ply=position_ply,
            game_n_moves=None,
        )
    return PositionRecord(
        board_tensor=board_hwc,
        to_move=state.to_move,
        legal_moves=legal,
        visit_counts=[0] * len(legal),
        outcome=target_in_to_move(state.to_move, _resolve_target_black(case, calibration_target)),
        active_size=state.active_size,
        ply=position_ply,
        game_n_moves=None,
    )
```

(d) Populate metadata in `build_calibration_sample` (add after the existing `tag`/`target_black` lines, before the `return`):

```python
    loss_mode = case.get("loss_mode") or "hard_value"
    teacher_value = (float(case["teacher_value"])
                     if loss_mode == "teacher_retention" else None)
    teacher_policy_len = (len(record.visit_counts)
                          if loss_mode == "teacher_retention" else None)
    return CalibrationSample(record=record, weight_scale=weight_scale,
                             tag=tag, target_black_value=target_black,
                             loss_mode=loss_mode, teacher_value=teacher_value,
                             teacher_policy_len=teacher_policy_len)
```

(e) Detect schema in `from_manifest` (replace the existing `schema = (...)` expression):

```python
        if any((c.get("loss_mode") or "hard_value") == "teacher_retention" for c in cases):
            schema = "teacher_retention"
        elif any(c.get("target_black_value") not in (None, "") for c in cases):
            schema = "per_row_target"
        else:
            schema = "global_target"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py -v`
Expected: PASS (new tests pass; all pre-existing tests still pass — `hard_value` is the default so v2/v3 rows are unaffected).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/calibration_pool.py tests/test_calibration_pool.py
git commit -m "feat(calibration): parse teacher_retention rows + schema in calibration pool"
```

---

### Task 3: Load-time validation guards

**Files:**
- Modify: `scripts/GPU/alphazero/calibration_pool.py` (`_parse_teacher_policy`, plus a `_validate_hard_value_blank` check in `build_calibration_sample`)
- Test: `tests/test_calibration_pool.py`

**Interfaces:**
- Consumes: `legal_moves_sha1` (Task 1), the `loss_mode` branch (Task 2).
- Produces: fail-fast `ValueError` at load for: teacher_value out of `[−1,1]`; policy length ≠ `len(legal)`; policy entry < 0; policy sum ∉ `1 ± 1e-3`; `teacher_legal_moves_sha1` mismatch; a `hard_value` row with any populated teacher column.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_calibration_pool.py`:

```python
import copy

def test_teacher_policy_length_mismatch_rejected(tmp_path):
    from scripts.GPU.alphazero.calibration_pool import build_calibration_position
    case, n = _teacher_case(tmp_path)
    case["teacher_policy_json"] = json.dumps([1.0 / (n + 1)] * (n + 1))  # wrong length
    with pytest.raises(ValueError, match="length"):
        build_calibration_position(case, calibration_target=-0.5)


def test_teacher_policy_sha1_reorder_rejected(tmp_path):
    from scripts.GPU.alphazero.calibration_pool import build_calibration_position
    case, n = _teacher_case(tmp_path)
    case["teacher_legal_moves_sha1"] = "0" * 40            # same length, wrong hash
    with pytest.raises(ValueError, match="sha1|alignment"):
        build_calibration_position(case, calibration_target=-0.5)


def test_teacher_policy_not_normalized_rejected(tmp_path):
    from scripts.GPU.alphazero.calibration_pool import build_calibration_position
    case, n = _teacher_case(tmp_path)
    case["teacher_policy_json"] = json.dumps([2.0 / n] * n)  # sums to 2.0
    with pytest.raises(ValueError, match="sum|normal"):
        build_calibration_position(case, calibration_target=-0.5)


def test_teacher_value_out_of_range_rejected(tmp_path):
    from scripts.GPU.alphazero.calibration_pool import build_calibration_position
    case, n = _teacher_case(tmp_path)
    case["teacher_value"] = "1.5"
    with pytest.raises(ValueError, match="teacher_value"):
        build_calibration_position(case, calibration_target=-0.5)


def test_hard_value_row_with_teacher_column_rejected(tmp_path):
    from scripts.GPU.alphazero.calibration_pool import build_calibration_sample
    case = _write_case_side(tmp_path, "black", 5, game_idx=9)
    case["loss_mode"] = "hard_value"
    case["teacher_value"] = "0.1"                          # must be blank
    with pytest.raises(ValueError, match="hard_value|blank"):
        build_calibration_sample(case, calibration_target=-0.35)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py -k "rejected" -v`
Expected: FAIL (no validation yet — rows are accepted).

- [ ] **Step 3: Write minimal implementation**

In `scripts/GPU/alphazero/calibration_pool.py`, replace `_parse_teacher_policy` with the validating version (it needs the case's `replay`-reconstructed `legal`, already passed in):

```python
def _parse_teacher_policy(case: dict, legal) -> list[float]:
    cid = case.get("case_id")
    raw = case.get("teacher_policy_json")
    if raw in (None, ""):
        raise ValueError(f"{cid}: teacher_retention row needs teacher_policy_json")
    policy = [float(x) for x in json.loads(raw)]
    if len(policy) != len(legal):
        raise ValueError(
            f"{cid}: teacher_policy length {len(policy)} != legal_moves {len(legal)}")
    if any(p < 0.0 or not math.isfinite(p) for p in policy):
        raise ValueError(f"{cid}: teacher_policy has negative/non-finite entries")
    if abs(sum(policy) - 1.0) > 1e-3:
        raise ValueError(f"{cid}: teacher_policy not normalized (sum={sum(policy)})")
    stored = case.get("teacher_legal_moves_sha1") or ""
    expected = legal_moves_sha1(legal)
    if stored != expected:
        raise ValueError(
            f"{cid}: teacher_legal_moves_sha1 mismatch (alignment); "
            f"stored {stored!r} != recomputed {expected!r}")
    return policy
```

Add the teacher_value range check inside the `teacher_retention` branch of `build_calibration_position` (replace the `teacher_value = float(case["teacher_value"])` line):

```python
        teacher_value = float(case["teacher_value"])
        if not math.isfinite(teacher_value) or not (-1.0 <= teacher_value <= 1.0):
            raise ValueError(
                f"{case.get('case_id')}: teacher_value {teacher_value!r} must be finite in [-1,1]")
```

Add the strict-blank guard at the top of `build_calibration_sample` (before `build_calibration_position`):

```python
    if (case.get("loss_mode") or "hard_value") == "hard_value":
        populated = [k for k in ("teacher_value", "teacher_policy_json",
                                 "teacher_legal_moves_sha1")
                     if case.get(k) not in (None, "")]
        if populated:
            raise ValueError(
                f"{case.get('case_id')}: hard_value row must leave teacher columns "
                f"blank; found {populated}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py -v`
Expected: PASS (all reject tests pass; Task 2 happy-path tests still pass).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/calibration_pool.py tests/test_calibration_pool.py
git commit -m "feat(calibration): fail-fast validation for teacher_retention rows"
```

---

### Task 4: `split_samples_with_modes` (mask producer)

**Files:**
- Modify: `scripts/GPU/alphazero/calibration_pool.py`
- Test: `tests/test_calibration_pool.py`

**Interfaces:**
- Consumes: `CalibrationSample.loss_mode`, `split_samples` (existing).
- Produces: `split_samples_with_modes(samples, has_weight_scale) -> tuple[list[PositionRecord], np.ndarray | None, np.ndarray]` — returns `(records, weights, teacher_policy_mask)` where `teacher_policy_mask` is a float32 `(len(samples),)` array, `1.0` for `teacher_retention` rows, `0.0` otherwise.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_calibration_pool.py`:

```python
def test_split_samples_with_modes_builds_mask(tmp_path):
    from scripts.GPU.alphazero.calibration_pool import (
        build_calibration_sample, split_samples_with_modes)
    hard = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=1), calibration_target=-0.35)
    tcase, _ = _teacher_case(tmp_path, position_ply=5, game_idx=2)
    teach = build_calibration_sample(tcase, calibration_target=-0.35)
    records, weights, mask = split_samples_with_modes([hard, teach, hard],
                                                      has_weight_scale=False)
    assert len(records) == 3
    assert mask.tolist() == [0.0, 1.0, 0.0]
    assert mask.dtype == np.float32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py::test_split_samples_with_modes_builds_mask -v`
Expected: FAIL with `ImportError: cannot import name 'split_samples_with_modes'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/GPU/alphazero/calibration_pool.py`, after `split_samples`:

```python
def split_samples_with_modes(samples, has_weight_scale: bool):
    """Like split_samples, plus a teacher_policy_mask (float32 (N,), 1.0 for
    teacher_retention rows, 0.0 otherwise). Used by the v4 calibration loss to
    gate the policy-CE term to retention rows only."""
    records, weights = split_samples(samples, has_weight_scale)
    mask = np.asarray(
        [1.0 if s.loss_mode == "teacher_retention" else 0.0 for s in samples],
        dtype=np.float32)
    return records, weights, mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py::test_split_samples_with_modes_builds_mask -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/calibration_pool.py tests/test_calibration_pool.py
git commit -m "feat(calibration): split_samples_with_modes returns teacher_policy_mask"
```

---

### Task 5: Calibration loss — value + masked policy-CE + telemetry + extended return tuple

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` (`alphazero_loss_batch`, signature ~1078–1093 and calib block ~1189–1222)
- Test: `tests/test_calibration_loss.py`

**Interfaces:**
- Consumes: `compute_masked_log_probs` (trainer.py:1055), `make_padded_batch`.
- Produces: `alphazero_loss_batch(..., calibration_teacher_policy_mask=None, teacher_value_weight=1.0, teacher_policy_kl_weight=0.25)`. Return shapes: **7-tuple** (no calib), **10-tuple** (calib active, `mask is None` — byte-identical to today), **14-tuple** (calib active, `mask` present) appending `(calib_value_term, calib_policy_ce_term, calib_policy_kl_est_term, n_teacher_retention)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_calibration_loss.py` (`_main_pos`, `_calib_pos`, `create_network` already imported):

```python
def _teacher_calib_pos(value=0.2):
    # 2 legal moves, uniform teacher policy in visit_counts.
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move="black", legal_moves=[(0, 0), (1, 1)],
        visit_counts=[0.5, 0.5], outcome=value, active_size=24,
        ply=20, game_n_moves=None,
    )


def test_mask_none_stays_ten_tuple_regression():
    net = create_network(hidden=64, n_blocks=2)
    out = alphazero_loss_batch(
        net, [_main_pos() for _ in range(3)],
        calibration_positions=[_calib_pos(-0.5)],
        calibration_loss_weight=0.02,
        calibration_teacher_policy_mask=None,     # v2/v3 path
    )
    assert len(out) == 10                          # unchanged shape


def test_teacher_mode_returns_fourteen_tuple():
    net = create_network(hidden=64, n_blocks=2)
    calib = [_calib_pos(-0.5), _teacher_calib_pos(0.2)]   # 1 correction, 1 retention
    out = alphazero_loss_batch(
        net, [_main_pos() for _ in range(3)],
        calibration_positions=calib,
        calibration_weights=np.array([1.0, 1.0], dtype=np.float32),
        calibration_loss_weight=0.01,
        calibration_teacher_policy_mask=np.array([0.0, 1.0], dtype=np.float32),
        teacher_value_weight=1.0, teacher_policy_kl_weight=0.25,
    )
    assert len(out) == 14
    (_, _, _, _, _, _, _, _, _, _,
     value_term, policy_ce, policy_kl_est, n_ret) = out
    assert n_ret == 1                              # one retention row
    assert float(policy_ce) >= float(policy_kl_est) - 1e-5   # CE >= CE - H  (H >= 0)
    assert float(policy_kl_est) >= -1e-4           # KL is non-negative


def test_policy_ce_zero_when_no_retention_rows():
    net = create_network(hidden=64, n_blocks=2)
    out = alphazero_loss_batch(
        net, [_main_pos() for _ in range(3)],
        calibration_positions=[_calib_pos(-0.5)],
        calibration_weights=np.array([1.0], dtype=np.float32),
        calibration_loss_weight=0.01,
        calibration_teacher_policy_mask=np.array([0.0], dtype=np.float32),  # no retention
        teacher_value_weight=1.0, teacher_policy_kl_weight=0.25,
    )
    assert len(out) == 14
    assert abs(float(out[11])) < 1e-6              # policy_ce == 0 (guarded denominator)


def test_make_padded_batch_correction_vs_retention_target_pi():
    # spec §10: bridge between parsing and loss — correction rows produce a
    # zero target_pi, retention rows a normalized one, padded/masked columns no mass.
    from scripts.GPU.alphazero.trainer import make_padded_batch
    corr = _calib_pos(-0.5)                        # 2 legal moves, visit_counts [0, 0]
    ret3 = PositionRecord(                          # 3 legal moves, uniform teacher policy
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move="black", legal_moves=[(0, 0), (1, 1), (2, 2)],
        visit_counts=[1 / 3, 1 / 3, 1 / 3], outcome=0.2, active_size=24,
        ply=20, game_n_moves=None)
    _, _, _, mask, target_pi, _ = make_padded_batch([corr, ret3])
    tp = np.array(target_pi.tolist())
    msk = np.array(mask.tolist())
    assert tp.shape[1] == msk.shape[1] == 3        # target_pi width == padded legal dim
    assert np.allclose(tp[0], 0.0)                 # correction row: all-zero target
    np.testing.assert_allclose(tp[1].sum(), 1.0, atol=1e-6)  # retention row sums to 1
    assert tp[0, 2] == 0.0 and msk[0, 2] == 0.0    # corr's padded slot: masked, no mass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_loss.py -k "fourteen or mask_none or policy_ce_zero" -v`
Expected: the three new-behavior tests FAIL with `TypeError: alphazero_loss_batch() got an unexpected keyword argument 'calibration_teacher_policy_mask'`. (`test_make_padded_batch_correction_vs_retention_target_pi` is a characterization guard of existing `make_padded_batch` behavior — it passes immediately and is exercised with the full file in Step 4.)

- [ ] **Step 3: Write minimal implementation**

(a) Add the three params to the `alphazero_loss_batch` signature (after `calibration_loss_weight: float = 0.0,`):

```python
    calibration_teacher_policy_mask=None,             # v4: (N,) 1.0 retention / 0.0 correction
    teacher_value_weight: float = 1.0,                # v4
    teacher_policy_kl_weight: float = 0.25,           # v4 (CE gradient term)
```

(b) Replace the calib block (current lines ~1189–1219, from `if calib_active:` through its `return`) with:

```python
    if calib_active:
        cb_boards, cb_rows, cb_cols, cb_mask, cb_pi, cb_targets = make_padded_batch(
            calibration_positions, max_moves_cap=max_moves_cap
        )
        teacher_mode = calibration_teacher_policy_mask is not None
        if teacher_mode:
            cb_logits, cb_values, _ = network.forward_padded(
                cb_boards, cb_rows, cb_cols, cb_mask,
                active_size=calibration_positions[0].active_size)
        else:
            _, cb_values, _ = network.forward_padded(
                cb_boards, cb_rows, cb_cols, cb_mask,
                active_size=calibration_positions[0].active_size)

        # Value term (ALL calibration rows), optional per-row weight_scale.
        per_value = (cb_values - cb_targets) ** 2
        if calibration_weights is not None:
            _w = mx.reshape(mx.array(calibration_weights), per_value.shape)
            value_term = mx.sum(_w * per_value) / mx.maximum(mx.sum(_w), 1e-8)
        else:
            _w = None
            value_term = mx.mean(per_value)
        calib_value_mean = mx.mean(cb_values)

        if not teacher_mode:
            # v2/v3 byte-identical path: value-only, 10-tuple.
            calib_loss = value_term
            total_loss = total_loss + calibration_loss_weight * calib_loss
            return (total_loss, policy_loss, value_loss, l2_loss,
                    aux_loss, aux_coverage, aux_n_eligible,
                    calib_loss, calib_value_mean, len(calibration_positions))

        # v4 teacher-retention path: value + masked policy CE; 14-tuple.
        m = mx.reshape(mx.array(calibration_teacher_policy_mask), per_value.shape)
        w = _w if _w is not None else mx.ones(per_value.shape)
        wm = w * m
        denom_p = mx.maximum(mx.sum(wm), 1e-8)
        cb_log_probs = compute_masked_log_probs(cb_logits, cb_mask)
        per_ce = -mx.sum(cb_pi * cb_log_probs, axis=1)          # (B,) cross-entropy
        policy_ce = mx.sum(wm * per_ce) / denom_p
        # Telemetry-only KL estimate: CE - teacher entropy (NOT in the gradient path).
        safe_pi = mx.where(cb_pi > 0, cb_pi, mx.array(1.0, dtype=cb_pi.dtype))
        per_H = -mx.sum(cb_pi * mx.log(safe_pi), axis=1)        # (B,) teacher entropy
        teacher_H = mx.sum(wm * per_H) / denom_p
        policy_kl_est = policy_ce - teacher_H

        calib_loss = (teacher_value_weight * value_term
                      + teacher_policy_kl_weight * policy_ce)
        total_loss = total_loss + calibration_loss_weight * calib_loss
        n_retention = int(mx.sum(m).item())
        return (total_loss, policy_loss, value_loss, l2_loss,
                aux_loss, aux_coverage, aux_n_eligible,
                calib_loss, calib_value_mean, len(calibration_positions),
                value_term, policy_ce, policy_kl_est, n_retention)
```

Note: `cb_pi` (was `_cb_pi`) is now used (the teacher policy `target_pi`). Correction rows have all-zero `cb_pi` and `m=0`, so they contribute nothing to the policy term — both by mask and by zero target.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_loss.py -v`
Expected: PASS (new tests pass; pre-existing 7-tuple/10-tuple tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py tests/test_calibration_loss.py
git commit -m "feat(calibration): v4 teacher value+policy-CE calibration loss term"
```

---

### Task 6: `train_step` unpack + `train()` schema-gated mask + weight threading

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` (`train_step` signature ~1225–1244 and unpack ~1303–1310; `train()` signature ~2357–2362, sampling block ~3781–3817)
- Test: `tests/test_calibration_loss.py`, `tests/test_training.py`

**Interfaces:**
- Consumes: `alphazero_loss_batch` v4 params (Task 5), `split_samples_with_modes` (Task 4).
- Produces:
  - `train_step(..., calibration_teacher_policy_mask=None, teacher_value_weight=1.0, teacher_policy_kl_weight=0.25)` returning 7/10/14-tuple of floats matching the loss.
  - `train(..., post_opening_calibration_teacher_value_weight: float = 1.0, post_opening_calibration_teacher_policy_kl_weight: float = 0.25)`; per-step it builds the mask via `split_samples_with_modes` **only when `_calib_pool.schema == "teacher_retention"`** (else passes `None`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_calibration_loss.py`:

```python
def test_train_step_teacher_mode_returns_fourteen_floats():
    net = create_network(hidden=64, n_blocks=2)
    main = MainModule(net.encoder, net.policy_head)
    opt_main = optim.Adam(learning_rate=1e-3)
    opt_value = optim.Adam(learning_rate=1e-3)
    out = train_step(
        network=net, main_module=main, opt_main=opt_main, opt_value=opt_value,
        batch=[_main_pos() for _ in range(3)],
        calibration_positions=[_calib_pos(-0.5), _teacher_calib_pos(0.2)],
        calibration_weights=np.array([1.0, 1.0], dtype=np.float32),
        calibration_loss_weight=0.01,
        calibration_teacher_policy_mask=np.array([0.0, 1.0], dtype=np.float32),
        teacher_value_weight=1.0, teacher_policy_kl_weight=0.25,
    )
    assert len(out) == 14
    assert all(isinstance(float(x), float) for x in out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_loss.py::test_train_step_teacher_mode_returns_fourteen_floats -v`
Expected: FAIL with `TypeError: train_step() got an unexpected keyword argument 'calibration_teacher_policy_mask'`.

- [ ] **Step 3: Write minimal implementation**

(a) `train_step` signature — add after `calibration_loss_weight: float = 0.0,`:

```python
    calibration_teacher_policy_mask=None,             # v4
    teacher_value_weight: float = 1.0,                # v4
    teacher_policy_kl_weight: float = 0.25,           # v4
```

(b) In `train_step`'s `loss_fn`, pass them to `alphazero_loss_batch` (add after `calibration_loss_weight=calibration_loss_weight,`):

```python
            calibration_teacher_policy_mask=calibration_teacher_policy_mask,
            teacher_value_weight=teacher_value_weight,
            teacher_policy_kl_weight=teacher_policy_kl_weight,
```

(c) Update the `calib_active` unpack to handle the 14-tuple. Replace the unpack block (currently the `if calib_active:` / `else:` around lines 1306–1310) with:

```python
    teacher_mode = calibration_teacher_policy_mask is not None
    if calib_active and teacher_mode:
        (total_loss, policy_loss, value_loss, l2_loss, aux_loss, aux_coverage,
         aux_n_eligible, calib_loss, calib_value_mean, calib_n,
         calib_value_term, calib_policy_ce, calib_policy_kl_est, calib_n_retention) = loss_tuple
    elif calib_active:
        (total_loss, policy_loss, value_loss, l2_loss, aux_loss, aux_coverage,
         aux_n_eligible, calib_loss, calib_value_mean, calib_n) = loss_tuple
    else:
        total_loss, policy_loss, value_loss, l2_loss, aux_loss, aux_coverage, aux_n_eligible = loss_tuple
```

(d) Add the 14-float teacher return. Insert it immediately **before** the existing `if calib_active:` return (trainer.py:1336), matching the existing `.item()` style (`calib_value_term` / `calib_policy_ce` / `calib_policy_kl_est` are mx scalars from the loss tuple; `calib_n_retention` is already an int):

```python
    if calib_active and teacher_mode:
        return (
            float(total_loss.item()), float(policy_loss.item()),
            float(value_loss.item()), float(l2_loss.item()),
            float(aux_loss.item()), float(aux_coverage), int(aux_n_eligible),
            float(calib_loss.item()), float(calib_value_mean.item()), int(calib_n),
            float(calib_value_term.item()), float(calib_policy_ce.item()),
            float(calib_policy_kl_est.item()), int(calib_n_retention),
        )
```

(e) `train()` signature — add after `post_opening_calibration_tag_schedule: Optional[dict] = None,`:

```python
    post_opening_calibration_teacher_value_weight: float = 1.0,
    post_opening_calibration_teacher_policy_kl_weight: float = 0.25,
```

(f) In the per-step sampling block (~3781–3794), build the mask schema-gated and thread the weights. Replace the `split_samples(...)` call and the `train_step(...)` call in the `_calib_pool is not None` branch:

```python
                            if _calib_pool.schema == "teacher_retention":
                                from .calibration_pool import split_samples_with_modes
                                _calib_batch, _calib_weights, _calib_tp_mask = (
                                    split_samples_with_modes(_calib_samples,
                                                             _calib_pool.has_weight_scale))
                            else:
                                _calib_batch, _calib_weights = split_samples(
                                    _calib_samples, _calib_pool.has_weight_scale)
                                _calib_tp_mask = None
```

Then in the `train_step(...)` call add (after `calibration_loss_weight=effective_post_opening_calibration_weight,`):

```python
                                calibration_teacher_policy_mask=_calib_tp_mask,
                                teacher_value_weight=post_opening_calibration_teacher_value_weight,
                                teacher_policy_kl_weight=post_opening_calibration_teacher_policy_kl_weight,
```

The unpack of the `train_step` return on the teacher path is handled in Task 8 (telemetry accumulators); for now keep the existing 10-value unpack working by leaving the teacher-path accumulation as a follow-up — to avoid a broken intermediate, **guard the unpack**: change the existing `(loss_total, ..., _calib_n) = train_step(...)` to slice the first 10 returns:

```python
                            _ret = train_step(... )
                            (loss_total, loss_policy, loss_value, loss_l2, loss_aux, aux_cov,
                             aux_neli, _calib_loss, _calib_value_pred, _calib_n) = _ret[:10]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_loss.py tests/test_training.py -v`
Expected: PASS (train_step teacher test passes; existing training tests unaffected — schema is not `teacher_retention` for v2/v3 manifests, so mask stays `None`).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py tests/test_calibration_loss.py
git commit -m "feat(calibration): thread teacher mask + weights through train_step/train"
```

---

### Task 7: CLI flags in `train.py`

**Files:**
- Modify: `scripts/GPU/alphazero/train.py` (argparse ~388; `train_kwargs` thread ~837)
- Test: `tests/test_calibration_cli_flags.py`

**Interfaces:**
- Consumes: `train()` params from Task 6.
- Produces: `--post-opening-calibration-teacher-value-weight` (default `1.0`), `--post-opening-calibration-teacher-policy-kl-weight` (default `0.25`), threaded into `train_kwargs`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_calibration_cli_flags.py`:

```python
def test_calibration_teacher_weight_flag_defaults():
    args = build_arg_parser().parse_args([])
    assert args.post_opening_calibration_teacher_value_weight == 1.0
    assert args.post_opening_calibration_teacher_policy_kl_weight == 0.25


def test_calibration_teacher_weight_flags_set():
    args = build_arg_parser().parse_args([
        "--post-opening-calibration-teacher-value-weight", "0.5",
        "--post-opening-calibration-teacher-policy-kl-weight", "0.0",
    ])
    assert args.post_opening_calibration_teacher_value_weight == 0.5
    assert args.post_opening_calibration_teacher_policy_kl_weight == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_cli_flags.py -k teacher_weight -v`
Expected: FAIL (`AttributeError: 'Namespace' object has no attribute 'post_opening_calibration_teacher_value_weight'`).

- [ ] **Step 3: Write minimal implementation**

In `scripts/GPU/alphazero/train.py`, after the `--post-opening-calibration-tag-schedule` argument (~line 388–392):

```python
    parser.add_argument("--post-opening-calibration-teacher-value-weight", type=float, default=1.0,
        help="v4: weight on the calibration value-MSE term (correction + teacher "
             "retention rows). Default 1.0.")
    parser.add_argument("--post-opening-calibration-teacher-policy-kl-weight", type=float, default=0.25,
        help="v4: weight on the teacher policy cross-entropy (KL) term on "
             "teacher_retention rows only. Default 0.25; 0.0 = value-only ablation.")
```

Thread into `train_kwargs` (after the `post_opening_calibration_tag_schedule=...` lines ~837–838):

```python
        post_opening_calibration_teacher_value_weight=args.post_opening_calibration_teacher_value_weight,
        post_opening_calibration_teacher_policy_kl_weight=args.post_opening_calibration_teacher_policy_kl_weight,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibration_cli_flags.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/train.py tests/test_calibration_cli_flags.py
git commit -m "feat(calibration): v4 teacher value/policy-kl weight CLI flags"
```

---

### Task 8: Telemetry — accumulators + sidecar fields

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` (accumulator hoist ~2790–2794, reset ~3756, sampling-block accumulation ~3815–3817, sidecar `loss_accumulator` ~3963–3967) and `scripts/GPU/alphazero/calibration_pool.py` (`build_post_opening_calibration_block`)
- Test: `tests/test_calibration_pool.py`, `tests/test_training.py`

**Interfaces:**
- Consumes: the 14-tuple `train_step` return (Task 6).
- Produces: sidecar `post_opening_calibration.loss` gains `calib_value_term_avg_iter`, `calib_policy_ce_avg_iter`, `calib_policy_kl_est_avg_iter`, and `n_teacher_retention_drawn`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_calibration_pool.py`:

```python
def test_calibration_block_includes_teacher_telemetry():
    from scripts.GPU.alphazero.calibration_pool import build_post_opening_calibration_block
    block = build_post_opening_calibration_block(
        config={"enabled": True, "schema": "teacher_retention"},
        enabled=True,
        loss_accumulator={
            "sum_calib_loss": 4.0, "sum_calib_n_drawn": 60,
            "sum_calib_value_pred": 3.0, "steps_done": 10,
            "sum_calib_value_term": 2.0, "sum_calib_policy_ce": 5.0,
            "sum_calib_policy_kl_est": 0.1, "sum_n_teacher_retention": 20,
        },
    )
    np.testing.assert_allclose(block["loss"]["calib_value_term_avg_iter"], 0.2)
    np.testing.assert_allclose(block["loss"]["calib_policy_ce_avg_iter"], 0.5)
    np.testing.assert_allclose(block["loss"]["calib_policy_kl_est_avg_iter"], 0.01)
    assert block["loss"]["n_teacher_retention_drawn"] == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py::test_calibration_block_includes_teacher_telemetry -v`
Expected: FAIL (`KeyError: 'calib_value_term_avg_iter'`).

- [ ] **Step 3: Write minimal implementation**

(a) In `calibration_pool.py`, extend `build_post_opening_calibration_block`'s `"loss"` dict (after the existing `calib_n_drawn_per_step` entry, before the closing brace):

```python
            "calib_value_term_avg_iter":
                float(loss_accumulator.get("sum_calib_value_term", 0.0)) / steps,
            "calib_policy_ce_avg_iter":
                float(loss_accumulator.get("sum_calib_policy_ce", 0.0)) / steps,
            "calib_policy_kl_est_avg_iter":
                float(loss_accumulator.get("sum_calib_policy_kl_est", 0.0)) / steps,
            "n_teacher_retention_drawn":
                int(loss_accumulator.get("sum_n_teacher_retention", 0)),
```

(b) In `trainer.py`, hoist the new accumulators next to the existing ones (~2791–2794):

```python
    sum_calib_value_term: float = 0.0
    sum_calib_policy_ce: float = 0.0
    sum_calib_policy_kl_est: float = 0.0
    sum_n_teacher_retention: int = 0
```

(c) Accumulate on the teacher path. In the sampling block, after the `_ret = train_step(...)` call from Task 6, add:

```python
                            if _calib_tp_mask is not None and len(_ret) == 14:
                                sum_calib_value_term += _ret[10]
                                sum_calib_policy_ce += _ret[11]
                                sum_calib_policy_kl_est += _ret[12]
                                sum_n_teacher_retention += int(_ret[13])
```

(d) Add the four sums to the sidecar `loss_accumulator` dict (~3963–3967, after `"sum_calib_n_drawn_by_tag": sum_calib_n_drawn_by_tag,`):

```python
                        "sum_calib_value_term": sum_calib_value_term,
                        "sum_calib_policy_ce": sum_calib_policy_ce,
                        "sum_calib_policy_kl_est": sum_calib_policy_kl_est,
                        "sum_n_teacher_retention": sum_n_teacher_retention,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py tests/test_training.py -v`
Expected: PASS (unit test passes; existing training telemetry tests unaffected — sums default to 0 on non-teacher runs).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py scripts/GPU/alphazero/calibration_pool.py tests/test_calibration_pool.py
git commit -m "feat(calibration): v4 CE/KL-est/n_retention telemetry in sidecar"
```

---

### Task 9: Teacher manifest builder + operator-guide catalog

**Files:**
- Create: `scripts/GPU/alphazero/build_teacher_calibration_manifest.py`
- Modify: `docs/post-game-analysis.md`
- Test: `tests/test_build_teacher_calibration_manifest.py` (new)

**Interfaces:**
- Consumes: `load_csv_manifest`, `position_state`, `LocalGPUEvaluator.infer`, `legal_moves_sha1` (Task 1), the v4 parser/validation (Tasks 2–3).
- Produces: a CLI `build_teacher_calibration_manifest` writing the v3 source CSV's rows plus `loss_mode` / `teacher_value` / `teacher_policy_json` / `teacher_legal_moves_sha1`; retention rows get raw `infer` value + priors and a **blanked** inherited `target_black_value`; correction rows pass through with blank teacher columns. Module-level `build_rows(rows, evaluator) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_teacher_calibration_manifest.py`:

```python
import importlib
import json
import numpy as np

from scripts.GPU.alphazero.build_teacher_calibration_manifest import build_rows
from scripts.GPU.alphazero.calibration_pool import (
    build_calibration_sample, legal_moves_sha1)
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
from tests.goal_line_probe_fixtures import legal_replay


class _FakeEval:
    """Stand-in for LocalGPUEvaluator: deterministic uniform priors + fixed value.
    Records that infer() was called (no MCTS)."""
    def build_input_tensor(self, state):
        return state.to_tensor()
    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        b, m = move_mask.shape
        priors = move_mask / np.maximum(move_mask.sum(axis=1, keepdims=True), 1.0)
        values = np.full((b,), 0.2, dtype=np.float32)
        return priors.astype(np.float32), values


def _rows(tmp_path):
    rp = tmp_path / "game_000001.json"
    rp.write_text(json.dumps(legal_replay(9, game_idx=1)))
    return [
        {"game_idx": "1", "case_id": "corr1", "replay_path": str(rp),
         "position_ply": "5", "side_to_move": "black",
         "tag": "black_predrop_correction", "target_black_value": "-0.35",
         "weight_scale": "1.0"},
        {"game_idx": "1", "case_id": "ret1", "replay_path": str(rp),
         "position_ply": "5", "side_to_move": "black",
         "tag": "old_post_opening_retention", "target_black_value": "-0.11",  # stale MCTS scalar
         "weight_scale": "1.0"},
    ]


def test_builder_blanks_correction_and_fills_retention(tmp_path):
    out = build_rows(_rows(tmp_path), _FakeEval())
    corr = next(r for r in out if r["case_id"] == "corr1")
    ret = next(r for r in out if r["case_id"] == "ret1")
    assert corr["loss_mode"] == "hard_value"
    assert corr["teacher_value"] == "" and corr["teacher_policy_json"] == ""
    assert ret["loss_mode"] == "teacher_retention"
    assert float(ret["teacher_value"]) == 0.2
    assert ret["target_black_value"] == ""          # stale MCTS scalar blanked
    policy = json.loads(ret["teacher_policy_json"])
    assert abs(sum(policy) - 1.0) < 1e-6


def test_builder_output_passes_parser(tmp_path):
    out = build_rows(_rows(tmp_path), _FakeEval())
    ret = next(r for r in out if r["case_id"] == "ret1")
    # round-trip: the built row must satisfy the v4 loader/validation.
    sample = build_calibration_sample(ret, calibration_target=-0.35)
    assert sample.loss_mode == "teacher_retention"
    assert sample.teacher_value == 0.2


def test_builder_module_does_not_import_mcts():
    mod = importlib.import_module(
        "scripts.GPU.alphazero.build_teacher_calibration_manifest")
    src = open(mod.__file__).read()
    assert "import mcts" not in src.lower()
    assert "from .mcts" not in src and "MCTS(" not in src
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_build_teacher_calibration_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: ... build_teacher_calibration_manifest`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/GPU/alphazero/build_teacher_calibration_manifest.py`:

```python
"""Deterministic v4 teacher-cache builder: read the v3 stratified manifest, run
the teacher checkpoint's RAW forward (LocalGPUEvaluator.infer — NO MCTS) over each
retention row, and append teacher_value / teacher_policy_json /
teacher_legal_moves_sha1 + loss_mode. Correction rows pass through with blank
teacher columns. See docs/superpowers/specs/2026-06-29-...-v4-teacher-retention-design.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .position_probe_cases import load_csv_manifest
from .goal_line_trigger_probe_cases import position_state
from .calibration_pool import legal_moves_sha1

CORRECTION_TAG = "black_predrop_correction"
OUT_COLUMNS = [
    "case_rank", "tag", "source", "source_rank", "target_black_value", "weight_scale",
    "game_idx", "case_id", "replay_path", "position_ply", "side_to_move",
    "anchor_checkpoint", "drop_ply", "largest_drop_phase", "collapse_type",
    "loss_mode", "teacher_value", "teacher_policy_json", "teacher_legal_moves_sha1",
]


def _teacher_infer(state, evaluator):
    """Single-position RAW forward → (priors over legal_moves, value). No MCTS."""
    legal = state.legal_moves()
    board_chw = evaluator.build_input_tensor(state)
    board_hwc = np.transpose(board_chw, (1, 2, 0)).astype(np.float32)[None, ...]
    n = len(legal)
    rows = np.zeros((1, n), dtype=np.int32)
    cols = np.zeros((1, n), dtype=np.int32)
    mask = np.ones((1, n), dtype=np.float32)
    for j, (r, c) in enumerate(legal):
        rows[0, j], cols[0, j] = r, c
    priors, values = evaluator.infer(board_hwc, rows, cols, mask, state.active_size)
    return legal, priors[0][:n].astype(float).tolist(), float(values[0])


def build_rows(rows: list, evaluator) -> list:
    out = []
    for r in rows:
        row = {c: r.get(c, "") for c in OUT_COLUMNS}
        is_correction = (r.get("tag") == CORRECTION_TAG)
        if is_correction:
            row["loss_mode"] = "hard_value"
            row["teacher_value"] = ""
            row["teacher_policy_json"] = ""
            row["teacher_legal_moves_sha1"] = ""
            out.append(row)
            continue
        replay = json.loads(Path(r["replay_path"]).read_text())
        state = position_state(replay, int(float(r["position_ply"])), r["side_to_move"])
        legal, policy, value = _teacher_infer(state, evaluator)
        row["loss_mode"] = "teacher_retention"
        row["teacher_value"] = repr(value)
        row["teacher_policy_json"] = json.dumps(policy)
        row["teacher_legal_moves_sha1"] = legal_moves_sha1(legal)
        row["target_black_value"] = ""          # blank the stale v3 MCTS-root scalar
        out.append(row)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the v4 teacher calibration manifest.")
    ap.add_argument("--source", required=True, help="v3 stratified manifest CSV")
    ap.add_argument("--teacher-checkpoint", required=True, help=".safetensors teacher")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args(argv)

    from .local_evaluator import LocalGPUEvaluator
    from .probe_eval import load_network_for_scoring
    rows = load_csv_manifest(args.source)["cases"]
    network = load_network_for_scoring(args.teacher_checkpoint)
    evaluator = LocalGPUEvaluator(network)
    out_rows = build_rows(rows, evaluator)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        w.writeheader()
        w.writerows(out_rows)
    n_ret = sum(1 for r in out_rows if r["loss_mode"] == "teacher_retention")
    print(f"wrote {len(out_rows)} rows ({n_ret} teacher_retention) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note: `load_network_for_scoring` (in `probe_eval.py`) does `create_network()` + `net.load_weights(path)` — the canonical "load a checkpoint for inference" helper used by the probes. `build_rows` is evaluator-injected, so the unit tests drive it with a fake evaluator and never touch `main()`.

- [ ] **Step 4: Run tests to verify they pass, then add docs**

Run: `.venv/bin/python -m pytest tests/test_build_teacher_calibration_manifest.py -v`
Expected: PASS

Then add a catalog entry to `docs/post-game-analysis.md` (after the §6 `build_targeted_calibration_manifest` section), mirroring its style:

```markdown
## 7. `build_teacher_calibration_manifest` — v4 teacher-retention manifest

**Purpose:** Read the v3 stratified manifest and cache the teacher checkpoint's
RAW forward (`infer`, no MCTS) over each retention row, writing `loss_mode`,
`teacher_value` (side-to-move), `teacher_policy_json` (dense, aligned to
legal_moves), and `teacher_legal_moves_sha1`. Correction rows pass through with
blank teacher columns.

```bash
.venv/bin/python -m scripts.GPU.alphazero.build_teacher_calibration_manifest \
  --source logs/eval/targeted_calibration_v3_strat_from_calib020_0001.csv \
  --teacher-checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --out logs/eval/targeted_calibration_v4_teacher_from_calib020_0001.csv
```

**Gate 0:** run `smoke_teacher_calibration_v4.py` after building — must pass
(`value_mse ≈ 0`, `kl_est ≈ 0`) before any training run.
```

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/build_teacher_calibration_manifest.py tests/test_build_teacher_calibration_manifest.py docs/post-game-analysis.md
git commit -m "feat(calibration): v4 teacher manifest builder (raw infer, no MCTS) + docs"
```

---

### Task 10: Gate-0 pre-flight self-distillation smoke

**Files:**
- Create: `scripts/GPU/alphazero/smoke_teacher_calibration_v4.py`
- Test: `tests/test_build_teacher_calibration_manifest.py` (extend) — exercise `assert_self_distillation` with a fake teacher whose outputs match the manifest.

**Interfaces:**
- Consumes: `CalibrationPool.from_manifest` (teacher schema), `split_samples_with_modes`, `alphazero_loss_batch` (Task 5).
- Produces: `assert_self_distillation(network, manifest_path, tol=1e-4) -> dict` returning `{"value_mse": float, "kl_est": float}` and raising `AssertionError` if either exceeds `tol`; a `main()` CLI that loads the teacher and runs it.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_build_teacher_calibration_manifest.py`:

```python
def test_self_distillation_holds_for_matching_teacher(tmp_path):
    import csv as _csv
    from scripts.GPU.alphazero.smoke_teacher_calibration_v4 import assert_self_distillation
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator

    # Build a 1-retention-row manifest whose teacher == THIS network's own outputs.
    rp = tmp_path / "game_000001.json"
    rp.write_text(json.dumps(legal_replay(9, game_idx=1)))
    net = create_network(hidden=64, n_blocks=2)
    rows = [{"game_idx": "1", "case_id": "ret1", "replay_path": str(rp),
             "position_ply": "5", "side_to_move": "black",
             "tag": "old_post_opening_retention", "weight_scale": "1.0"}]
    built = build_rows(rows, LocalGPUEvaluator(net))     # teacher = net itself
    manifest = tmp_path / "v4.csv"
    with manifest.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(built[0].keys()))
        w.writeheader(); w.writerows(built)

    stats = assert_self_distillation(net, str(manifest), tol=1e-3)
    assert abs(stats["value_mse"]) < 1e-3
    assert abs(stats["kl_est"]) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_build_teacher_calibration_manifest.py::test_self_distillation_holds_for_matching_teacher -v`
Expected: FAIL with `ModuleNotFoundError: ... smoke_teacher_calibration_v4`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/GPU/alphazero/smoke_teacher_calibration_v4.py`:

```python
"""Gate-0 pre-flight self-distillation check (spec §5.1). With base == teacher,
the v4 calibration forward must reproduce the teacher's stored value and policy
on retention rows: value_mse ≈ 0 and kl_est ≈ 0 (CE ≈ teacher entropy). Run after
building the manifest, before any training run.
"""
from __future__ import annotations

import argparse
import sys

from .calibration_pool import CalibrationPool, split_samples_with_modes
from .trainer import alphazero_loss_batch


def assert_self_distillation(network, manifest_path: str, tol: float = 1e-4) -> dict:
    pool = CalibrationPool.from_manifest(manifest_path, calibration_target=-0.35)
    if pool.schema != "teacher_retention":
        raise AssertionError(f"manifest schema is {pool.schema!r}, expected teacher_retention")
    retention = [s for s in pool._samples if s.loss_mode == "teacher_retention"]
    if not retention:
        raise AssertionError("no teacher_retention rows in manifest")
    records, weights, mask = split_samples_with_modes(retention, pool.has_weight_scale)
    out = alphazero_loss_batch(
        network, records,                       # dummy main batch == calib batch is fine
        calibration_positions=records,
        calibration_weights=weights,
        calibration_loss_weight=1.0,
        calibration_teacher_policy_mask=mask,
        teacher_value_weight=1.0, teacher_policy_kl_weight=1.0,
    )
    value_mse = float(out[10])                  # calib_value_term
    kl_est = float(out[12])                     # calib_policy_kl_est_term
    if abs(value_mse) > tol or abs(kl_est) > tol:
        raise AssertionError(
            f"self-distillation FAILED: value_mse={value_mse:.3e}, kl_est={kl_est:.3e} "
            f"(tol={tol}). Check checkpoint / canonicalization / perspective / policy "
            f"alignment / accidental MCTS targets.")
    return {"value_mse": value_mse, "kl_est": kl_est}


def main(argv=None):
    ap = argparse.ArgumentParser(description="v4 gate-0 self-distillation smoke")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--teacher-checkpoint", required=True)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args(argv)
    from .probe_eval import load_network_for_scoring
    network = load_network_for_scoring(args.teacher_checkpoint)
    stats = assert_self_distillation(network, args.manifest, tol=args.tol)
    print(f"PASS gate-0 self-distillation: value_mse={stats['value_mse']:.3e}, "
          f"kl_est={stats['kl_est']:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note: the test drives `assert_self_distillation` directly with an in-memory network; only `main()` loads from disk, via `load_network_for_scoring` (`probe_eval.py`).

- [ ] **Step 4: Run the full suite to verify green**

Run: `.venv/bin/python -m pytest tests/test_build_teacher_calibration_manifest.py tests/test_calibration_pool.py tests/test_calibration_loss.py tests/test_calibration_cli_flags.py tests/test_training.py -v`
Expected: PASS (all v4 tests green; all pre-existing calibration/training tests still green).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/smoke_teacher_calibration_v4.py tests/test_build_teacher_calibration_manifest.py
git commit -m "feat(calibration): v4 gate-0 self-distillation pre-flight smoke"
```

---

## Operator runbook (after implementation, not a code task)

1. **Build the manifest** (Task 9 CLI) → `logs/eval/targeted_calibration_v4_teacher_from_calib020_0001.csv`.
2. **Gate 0** (Task 10 CLI): `smoke_teacher_calibration_v4.py --manifest <v4.csv> --teacher-checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors` — must print PASS.
3. **Run** the v4 training command (spec §11 — the v3 command with the three deltas; `--iterations 1`).
4. **Gates A–D**: 400-sim probes vs `calib020_0001` (spec §2). No promotion match unless all four pass.
5. **Record**: append the v4 row to `docs/2026-06-26-...experiment-ledger.md` (template in spec §11) and update do-not-repeat / severe-overlap.

## Whole-suite gate before integration

Before FF-merging `calibration-v4-teacher-retention` into `main`:

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: the full suite passes (matches the v2/v3 merge discipline in the experiment ledger).
