# Targeted Value Calibration v5 — MCTS-Root-Visit Policy Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement v5 calibration = A hard-value correction rows unchanged + B/C/D retention rows anchored by **raw teacher value (stm)** for the value term and **BASE's 400-sim MCTS root visit distribution** for the policy-CE term — the first untested combination in the calibration lineage (v3 = root-value-only → failed; v4 = raw value + raw priors → raw held, MCTS root failed).

**Architecture:** One new offline builder (`build_mcts_root_retention_manifest.py`) writes a v5 manifest whose retention rows carry `teacher_value` (raw, eval-mode BN) plus dense normalized root visits + provenance columns. `calibration_pool` gains a `RETENTION_POLICY_LOSS_MODES` registry, a parse branch for `loss_mode=mcts_root_retention`, and a generalized mask predicate. The trainer needs only two string-gate touches (sampling gate + startup print). **No new loss math**: root rows ride the existing v4 14-tuple masked value + policy-CE path unchanged. A v5 smoke validates mechanics with the corrected expectations (value term ≈ 0, policy CE finite and positive — NOT ≈ 0).

**Tech Stack:** Python 3 / MLX (lazily, in builder `main()` + smoke only), numpy, csv/json/argparse, pytest.

**Spec:** the locked v5 section of `docs/2026-06-26-targeted-value-calibration-experiment-ledger-v3f-v4-overlap-updated.md` (§"Current next hypothesis — v5 MCTS-root-visit policy retention (LOCKED 2026-07-01)").

## Global Constraints

- **Do NOT modify:** `mcts.py`, `self_play.py`, any gate probe (`eval_position_probe.py`, `eval_goal_line_trigger_probe.py`, `eval_runner.py`), `network.py`, `local_evaluator.py`, `build_teacher_calibration_manifest.py`, `smoke_teacher_calibration_v4.py`, or any manifest/checkpoint on disk.
- **No new loss math path.** `alphazero_loss_batch` (`trainer.py:1104-1298`) is untouched. Root rows must flow the existing 14-tuple masked path (`teacher_mode = calibration_teacher_policy_mask is not None`).
- **THE MAKE-OR-BREAK HAZARD:** `mcts_root_retention` rows must get **mask = 1.0** in `split_samples_with_modes`. If the mask predicate is not generalized, v5 silently trains value-only = a v3 rerun (ledger do-not-repeat #9). There are exactly FOUR `"teacher_retention"` string-literal gates to audit: `calibration_pool.py:277` (mask predicate), `calibration_pool.py:252` (schema detection), `trainer.py:3921` (sampling gate `if _calib_pool.schema == "teacher_retention":`), `trainer.py:2889-2891` (startup print + count). Tasks 1–3 cover all four; the Task 3 tiny-train test proves the mask end-to-end.
- **Byte-identical when unused:** v2/v3 (`per_row_target`/`global_target`), v4 (`teacher_retention`), and hard-value paths must behave identically to today. All existing calibration tests must stay green unmodified (error-message wording may only change in ways that keep existing `pytest.raises(..., match=...)` patterns passing).
- **Two-evaluator split (verified 2026-07-01, do not "unify"):** the gate probes load via `_default_evaluator_factory` (`eval_runner.py:196-209`) which **never calls `net.eval()`** (`load_network_for_scoring` → `_load_network` = create + load_weights only) — gate MCTS leaf inference runs **train-mode BatchNorm, batch=1** (sync `search()` = one leaf per sim). The builder must therefore use: (a) `_default_evaluator_factory` **verbatim** for the root search (reproduces gate numbers by construction), and (b) a **separate eval-mode** evaluator (`load_network_for_scoring` → `net.eval()` → `LocalGPUEvaluator`) for the raw teacher value (matches the training-path eval-mode calibration forward `trainer.py:1232-1248`, so the value term starts ≈ 0).
- **Perspective:** `teacher_value` and `root_value_stm` are **side-to-move**; `root_black_value` is black-perspective (`black = v if side=='black' else -v`). The root search's `root.q_value` is stm.
- **Policy target = dense normalized root visits** aligned to `state.legal_moves()` order, pinned by `legal_moves_sha1` (`calibration_pool.py:23-29`), sum = 1.0. Not top-k. Zero total visits on a retention row = builder ValueError (loud), NOT a uniform fallback.
- Tests: `.venv/bin/python -m pytest <file> -q` from repo root. Extend existing test files (`tests/test_calibration_pool.py`, `tests/test_calibration_loss.py`, `tests/test_training.py`); the only new test files are `tests/test_build_mcts_root_retention_manifest.py` and `tests/test_smoke_mcts_root_retention_v5.py`.
- TDD per task (write failing test → confirm RED → implement → confirm GREEN → commit); file-scoped `git add`; commit trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Locate code by **function/content, not line number** (line anchors below were verified 2026-07-01 at main=7064621 but may drift).

---

## Grounding facts (verified against live code 2026-07-01, main @ 7064621)

| Symbol | Location | Contract |
|---|---|---|
| `MCTS(evaluator, cfg, rng).search(state, add_noise=False)` | `mcts.py:410-455` | Returns `(visit_counts, root.q_value)`. `visit_counts` = `{(r,c): int}` over **ALL** `root.state.legal_moves()` (unvisited → 0). `q_value` is stm. Sync path: one leaf per sim (batch=1 inference). |
| `cfg_from(EvalConfig(...))` | `eval_runner.py:79-95` | Gate probes use `EvalConfig(mcts_sims=400, mcts_eval_batch_size=14, mcts_stall_flush_sims=48)` → `MCTSConfig`. |
| `_default_evaluator_factory(path)` | `eval_runner.py:196-209` | `load_network_for_scoring` (NO eval()) → `LocalGPUEvaluator(net, compile=True)`. This is the gate probes' loader. |
| Gate seeds | `eval_position_probe.py:76` / `eval_goal_line_trigger_probe.py:69` | position probe: `random.Random(base_seed ^ int(game_idx) ^ int(position_ply))`, default base_seed **20260616**; goal-line probe: `random.Random(base_seed ^ game_idx)`, default **20260614**. |
| `_teacher_infer(state, evaluator)` | `build_teacher_calibration_manifest.py:28-40` | Raw single-position forward → `(legal, policy_list, value_stm)`. Reused for the v5 raw value anchor (already imported cross-module by `eval_raw_nn_position_rows.py` — precedent). |
| `legal_moves_sha1(legal)` | `calibration_pool.py:23-29` | Canonical `";".join(f"{r},{c}")` SHA-1. |
| `_parse_teacher_policy(case, legal)` | `calibration_pool.py:85-110` | Validates length / ≥0 finite / sum 1±1e-3 / sha1-vs-recomputed. Task 2 generalizes it. |
| `split_samples_with_modes(samples, has_ws)` | `calibration_pool.py:271-279` | Mask predicate `s.loss_mode == "teacher_retention"` — **Task 1 generalizes this**. |
| `from_manifest` schema detection | `calibration_pool.py:247-258` | teacher_retention checked first, then per_row_target, else global_target — **Task 2 extends**. |
| Trainer sampling gate | `trainer.py:3909-3928` | `if _calib_pool.schema == "teacher_retention": split_samples_with_modes ... else: split_samples + _calib_tp_mask=None` — **Task 3 generalizes**. |
| Trainer startup print | `trainer.py:2882-2908` | `if _calib_pool.schema == "teacher_retention": ... mode=teacher_retention (N teacher / M hard-value)` — **Task 3 generalizes** (this is the stale-label-bug site fixed once already @ 0f6a18b — do not reintroduce). |
| 14-tuple loss path | `trainer.py:1221-1295` | Gates on `calibration_teacher_policy_mask is not None`; forwards calibration batch in EVAL mode; value = weighted MSE vs `record.outcome`; policy CE vs normalized `record.visit_counts`. Mode-agnostic — **no changes**. |
| `PositionRecord.visit_counts` | `self_play.py` | Raw float "counts"; `make_padded_batch` normalizes (`calibration_pool.py:144` comment). Normalized visits are valid input. |
| Tiny-train test pattern | `tests/test_training.py:454-511` | Real `train()` 1-iter run with a tiny manifest, asserts model_iter JSON telemetry (`n_teacher_retention_drawn > 0`). Task 3 clones this for root mode. |

**Correction-row tag:** `black_predrop_correction` (constant `CORRECTION_TAG` in the v4 builder). **Goal-line retention tag:** `goal_line_retention` (uses the goal-line seed scheme); all other retention tags use the position-probe seed scheme.

---

## File Structure

- **Create:** `scripts/GPU/alphazero/build_mcts_root_retention_manifest.py` — offline v5 manifest builder (two evaluators: gate-mode search + eval-mode raw anchor; gate-CSV cross-check).
- **Create:** `scripts/GPU/alphazero/smoke_mcts_root_retention_v5.py` — gate-0 mechanics smoke (value ≈ 0, policy CE finite/positive, mask aligned).
- **Modify:** `scripts/GPU/alphazero/calibration_pool.py` — `RETENTION_POLICY_LOSS_MODES` + `VALID_LOSS_MODES` registries, unknown-mode guard, generalized mask predicate, `_parse_policy_json` generalization, `mcts_root_retention` parse branch, blank-guards, schema detection.
- **Modify:** `scripts/GPU/alphazero/trainer.py` — two string-gate generalizations only (sampling gate + startup print). No loss changes.
- **Modify (docs):** `docs/post-game-analysis.md` — new §8 operator block (mirrors §7 for v4).
- **Test (extend):** `tests/test_calibration_pool.py`, `tests/test_calibration_loss.py`, `tests/test_training.py`.
- **Test (create):** `tests/test_build_mcts_root_retention_manifest.py`, `tests/test_smoke_mcts_root_retention_v5.py`.

Module surface fixed across tasks:

```
calibration_pool.RETENTION_POLICY_LOSS_MODES: frozenset  # {"teacher_retention", "mcts_root_retention"}
calibration_pool.VALID_LOSS_MODES: frozenset             # {"hard_value"} | RETENTION_POLICY_LOSS_MODES
calibration_pool._parse_policy_json(case, legal, policy_col, sha1_col) -> list[float]
build_mcts_root_retention_manifest.CORRECTION_TAG = "black_predrop_correction"
build_mcts_root_retention_manifest.GOAL_LINE_TAG = "goal_line_retention"
build_mcts_root_retention_manifest.NEW_COLUMNS  # see Task 4
build_mcts_root_retention_manifest.row_seed(tag, game_idx, position_ply, pos_base, goal_base) -> int
build_mcts_root_retention_manifest.dense_normalized_visits(counts, legal, case_id) -> list[float]
build_mcts_root_retention_manifest.build_rows(rows, raw_evaluator, search_fn, *, sims, base_checkpoint,
                                              pos_base_seed, goal_base_seed,
                                              eval_batch_size, stall_flush_sims) -> list[dict]
build_mcts_root_retention_manifest.cross_check_gate_values(out_rows, gate_csv_paths, tol) -> dict
smoke_mcts_root_retention_v5.assert_root_retention_mechanics(network, manifest_path, value_tol=1e-4) -> dict
```

**How this plan satisfies the user-specified critical tests** (each folded into its owning TDD task): root mask=1.0 → Task 1; teacher still 1.0 / hard 0.0 → Task 1; schema detection → Task 2; correction rows reject populated root columns → Task 2; root_visits_json validates length/sum/nonnegative/sha1 → Task 2; root mode flows the 14-tuple path → Task 3 (loss test + tiny-train telemetry test); builder correctness → Task 4; smoke expectations → Task 5.

---

### Task 1: Loss-mode registry + generalized mask predicate + unknown-mode guard

**Files:**
- Modify: `scripts/GPU/alphazero/calibration_pool.py`
- Test: `tests/test_calibration_pool.py`

**Interfaces:**
- Consumes: existing `CalibrationSample`, `split_samples_with_modes`, `build_calibration_position`.
- Produces: module constants `RETENTION_POLICY_LOSS_MODES = frozenset({"teacher_retention", "mcts_root_retention"})` and `VALID_LOSS_MODES = frozenset({"hard_value"}) | RETENTION_POLICY_LOSS_MODES`; `split_samples_with_modes` masks 1.0 for ANY retention mode; `build_calibration_position` raises `ValueError` on an unknown `loss_mode` (today an unknown mode silently falls into the hard-value branch — that silent fallthrough is exactly how root rows could become a v3 rerun).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_calibration_pool.py` (reuse the file's existing helpers/imports; `np` and `split_samples_with_modes` import patterns already appear around the v4 tests):

```python
def test_retention_policy_loss_modes_registry():
    from scripts.GPU.alphazero.calibration_pool import (
        RETENTION_POLICY_LOSS_MODES, VALID_LOSS_MODES)
    assert RETENTION_POLICY_LOSS_MODES == frozenset({"teacher_retention", "mcts_root_retention"})
    assert VALID_LOSS_MODES == frozenset({"hard_value", "teacher_retention", "mcts_root_retention"})


def _sample_with_mode(loss_mode):
    """Direct CalibrationSample construction (no manifest parse needed) to test
    the mask predicate in isolation."""
    from scripts.GPU.alphazero.calibration_pool import CalibrationSample
    from scripts.GPU.alphazero.self_play import PositionRecord
    import numpy as _np
    rec = PositionRecord(
        board_tensor=_np.zeros((24, 24, 30), dtype=_np.float32), to_move="black",
        legal_moves=[(0, 0), (1, 1)], visit_counts=[0.5, 0.5], outcome=0.1,
        active_size=24, ply=5, game_n_moves=None)
    return CalibrationSample(record=rec, loss_mode=loss_mode)


def test_split_samples_with_modes_masks_all_retention_modes():
    from scripts.GPU.alphazero.calibration_pool import split_samples_with_modes
    samples = [_sample_with_mode("hard_value"),
               _sample_with_mode("teacher_retention"),
               _sample_with_mode("mcts_root_retention")]
    _, _, mask = split_samples_with_modes(samples, has_weight_scale=False)
    assert mask.tolist() == [0.0, 1.0, 1.0]   # root rows MUST be 1.0 (the v5 make-or-break)
    assert mask.dtype.name == "float32"


def test_unknown_loss_mode_rejected(tmp_path):
    import pytest as _pytest
    case = _write_case_side(tmp_path, "black")          # existing helper in this file
    case["loss_mode"] = "typo_mode"
    from scripts.GPU.alphazero.calibration_pool import build_calibration_position
    with _pytest.raises(ValueError, match="loss_mode"):
        build_calibration_position(case, calibration_target=-0.35)
```

> Note: `_write_case_side` is the existing helper at `tests/test_calibration_pool.py:40` producing a loadable case dict with a real replay on disk. If its exact signature differs, adapt the call — the requirement is a valid hard-value case dict whose `loss_mode` is then set to a garbage string.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py -q -k "retention_policy or masks_all or unknown_loss"`
Expected: FAIL — `ImportError: cannot import name 'RETENTION_POLICY_LOSS_MODES'` (and the mask test fails since `mcts_root_retention` currently gets 0.0; the unknown-mode test fails because no ValueError is raised).

- [ ] **Step 3: Write the minimal implementation**

In `scripts/GPU/alphazero/calibration_pool.py`:

(a) After `legal_moves_sha1` / near the top-of-module helpers, add:

```python
RETENTION_POLICY_LOSS_MODES = frozenset({"teacher_retention", "mcts_root_retention"})
VALID_LOSS_MODES = frozenset({"hard_value"}) | RETENTION_POLICY_LOSS_MODES
```

(b) In `build_calibration_position` (currently `loss_mode = case.get("loss_mode") or "hard_value"` then `if loss_mode == "teacher_retention":`), insert the guard immediately after loss_mode is resolved:

```python
    loss_mode = case.get("loss_mode") or "hard_value"
    if loss_mode not in VALID_LOSS_MODES:
        raise ValueError(
            f"{case.get('case_id')}: unknown loss_mode {loss_mode!r} "
            f"(valid: {sorted(VALID_LOSS_MODES)})")
```

Note: after this task, `mcts_root_retention` is VALID but has no parse branch yet, so it would fall to the hard-value path — that transient is closed by Task 2 (sequential, same branch); no test exercises it in between.

(c) In `split_samples_with_modes`, change the mask predicate:

```python
    mask = np.asarray(
        [1.0 if s.loss_mode in RETENTION_POLICY_LOSS_MODES else 0.0 for s in samples],
        dtype=np.float32)
```

(d) Update the `CalibrationSample.loss_mode` field comment (currently `# "hard_value" | "teacher_retention"`):

```python
    loss_mode: str = "hard_value"            # one of VALID_LOSS_MODES
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py -q`
Expected: PASS — all existing pool tests (34+ from v4) plus the 3 new ones. Zero existing failures (the v4 mask behavior is a superset-preserving change).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/calibration_pool.py tests/test_calibration_pool.py
git commit -m "feat(calibration): loss-mode registry + retention-mode mask + unknown-mode guard

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Parse `mcts_root_retention` rows + validation + schema detection

**Files:**
- Modify: `scripts/GPU/alphazero/calibration_pool.py`
- Test: `tests/test_calibration_pool.py`

**Interfaces:**
- Consumes: Task 1's registries; existing `_parse_teacher_policy`, `build_calibration_position`, `build_calibration_sample`, `CalibrationPool.from_manifest`.
- Produces: `_parse_policy_json(case, legal, policy_col, sha1_col) -> list[float]` (generalized validator; `_parse_teacher_policy` becomes a thin delegate so v4 error-message `match=` patterns keep passing — keep the words "length", "alignment", "normal" in messages); `build_calibration_position` branch for `mcts_root_retention` (`outcome = float(case["teacher_value"])` raw stm anchor, `visit_counts = _parse_policy_json(case, legal, "root_visits_json", "root_legal_moves_sha1")`); blank-guards (hard rows reject populated root columns; root rows reject populated `teacher_policy_json`); `from_manifest` schema `"mcts_root_retention"` (checked FIRST), mixed-retention-modes manifests rejected.

Root-row manifest columns consumed here: `teacher_value` (required, finite, [-1,1], stm), `root_visits_json` (required), `root_legal_moves_sha1` (required). Other `root_*` columns are provenance-only (not parsed).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_calibration_pool.py`:

```python
def _root_case(tmp_path, **overrides):
    """A valid mcts_root_retention case dict with matching sha1/policy computed
    from the actually reconstructed position."""
    import json as _json
    from scripts.GPU.alphazero.calibration_pool import legal_moves_sha1
    from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
    from tests.goal_line_probe_fixtures import legal_replay
    replay = legal_replay(9, game_idx=1)
    rp = tmp_path / "game_000001.json"
    rp.write_text(_json.dumps(replay))
    state = position_state(replay, 5, "black")
    legal = state.legal_moves()
    n = len(legal)
    case = {
        "game_idx": "1", "case_id": "root1", "replay_path": str(rp),
        "position_ply": "5", "side_to_move": "black",
        "tag": "old_post_opening_retention", "weight_scale": "1.0",
        "loss_mode": "mcts_root_retention",
        "teacher_value": "0.2",
        "root_visits_json": _json.dumps([1.0 / n] * n),
        "root_legal_moves_sha1": legal_moves_sha1(legal),
    }
    case.update(overrides)
    return case


def test_root_retention_row_parses(tmp_path):
    from scripts.GPU.alphazero.calibration_pool import build_calibration_sample
    sample = build_calibration_sample(_root_case(tmp_path), calibration_target=-0.35)
    assert sample.loss_mode == "mcts_root_retention"
    rec = sample.record
    assert abs(rec.outcome - 0.2) < 1e-9                       # raw teacher anchor, stm, DIRECT
    assert len(rec.visit_counts) == len(rec.legal_moves)       # dense root policy
    assert abs(sum(rec.visit_counts) - 1.0) < 1e-6
    assert abs(sample.teacher_value - 0.2) < 1e-9              # metadata reused


def test_root_retention_requires_teacher_value(tmp_path):
    import pytest as _pytest
    from scripts.GPU.alphazero.calibration_pool import build_calibration_sample
    with _pytest.raises(ValueError, match="teacher_value"):
        build_calibration_sample(_root_case(tmp_path, teacher_value=""),
                                 calibration_target=-0.35)


def test_root_retention_rejects_bad_policy(tmp_path):
    import json as _json
    import pytest as _pytest
    from scripts.GPU.alphazero.calibration_pool import build_calibration_sample
    # wrong length
    with _pytest.raises(ValueError, match="length"):
        build_calibration_sample(_root_case(tmp_path, root_visits_json=_json.dumps([1.0])),
                                 calibration_target=-0.35)
    # bad sum
    base = _root_case(tmp_path)
    n = len(_json.loads(base["root_visits_json"]))
    with _pytest.raises(ValueError, match="normal"):
        build_calibration_sample(_root_case(tmp_path, root_visits_json=_json.dumps([2.0 / n] * n)),
                                 calibration_target=-0.35)
    # sha1 mismatch (alignment)
    with _pytest.raises(ValueError, match="alignment|sha1"):
        build_calibration_sample(_root_case(tmp_path, root_legal_moves_sha1="0" * 40),
                                 calibration_target=-0.35)


def test_hard_value_rejects_populated_root_columns(tmp_path):
    import pytest as _pytest
    from scripts.GPU.alphazero.calibration_pool import build_calibration_sample
    case = _root_case(tmp_path, loss_mode="hard_value", target_black_value="-0.35",
                      teacher_value="")
    # root_visits_json / root_legal_moves_sha1 still populated -> must fail loudly
    with _pytest.raises(ValueError, match="blank"):
        build_calibration_sample(case, calibration_target=-0.35)


def test_root_retention_rejects_populated_teacher_policy(tmp_path):
    import json as _json
    import pytest as _pytest
    from scripts.GPU.alphazero.calibration_pool import build_calibration_sample
    case = _root_case(tmp_path, teacher_policy_json=_json.dumps([0.5, 0.5]))
    with _pytest.raises(ValueError, match="teacher_policy_json"):
        build_calibration_sample(case, calibration_target=-0.35)


def test_from_manifest_detects_root_schema_and_rejects_mixed(tmp_path):
    import csv as _csv
    import pytest as _pytest
    from scripts.GPU.alphazero.calibration_pool import CalibrationPool
    root = _root_case(tmp_path)
    hard = dict(_root_case(tmp_path), case_id="corr1", loss_mode="hard_value",
                teacher_value="", root_visits_json="", root_legal_moves_sha1="",
                target_black_value="-0.35")
    cols = sorted(set(root) | set(hard))
    man = tmp_path / "v5.csv"
    with man.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader(); w.writerows([root, hard])
    pool = CalibrationPool.from_manifest(str(man), calibration_target=-0.35)
    assert pool.schema == "mcts_root_retention"

    # mixed retention modes in one manifest -> loud error
    teacher = dict(_root_case(tmp_path), case_id="t1", loss_mode="teacher_retention")
    with (tmp_path / "mixed.csv").open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=sorted(set(root) | set(teacher)), restval="")
        w.writeheader(); w.writerows([root, teacher])
    with _pytest.raises(ValueError, match="mixes"):
        CalibrationPool.from_manifest(str(tmp_path / "mixed.csv"), calibration_target=-0.35)
```

> The mixed-manifest teacher row is intentionally NOT a fully valid teacher row — the mixed-modes check must fire in `from_manifest` **before** per-row parsing, otherwise the test would fail for the wrong reason. Implementation below detects modes from the raw cases first.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py -q -k "root_retention or root_schema or populated_root"`
Expected: FAIL — root rows currently fall into the hard-value branch (Task 1 guard admits the mode but no branch parses it), so `rec.outcome` is the hard target, not 0.2; the reject tests raise nothing or the wrong error.

- [ ] **Step 3: Write the minimal implementation**

In `scripts/GPU/alphazero/calibration_pool.py`:

(a) Generalize the policy validator. Replace `_parse_teacher_policy(case, legal)` with:

```python
def _parse_policy_json(case: dict, legal, policy_col: str, sha1_col: str) -> list[float]:
    """Parse and validate a dense policy JSON column against the reconstructed
    legal_moves. Checks: non-empty, length == len(legal), all entries >= 0 and
    finite, sum in 1 ± 1e-3, and the stored sha1 matches the recomputed hash
    over legal (catches a same-length reorder / stale alignment)."""
    cid = case.get("case_id")
    raw = case.get(policy_col)
    if raw in (None, ""):
        raise ValueError(f"{cid}: retention row needs {policy_col}")
    policy = [float(x) for x in json.loads(raw)]
    if len(policy) != len(legal):
        raise ValueError(
            f"{cid}: {policy_col} length {len(policy)} != legal_moves {len(legal)}")
    if any(p < 0.0 or not math.isfinite(p) for p in policy):
        raise ValueError(f"{cid}: {policy_col} has negative/non-finite entries")
    if abs(sum(policy) - 1.0) > 1e-3:
        raise ValueError(f"{cid}: {policy_col} not normalized (sum={sum(policy)})")
    stored = case.get(sha1_col) or ""
    expected = legal_moves_sha1(legal)
    if stored != expected:
        raise ValueError(
            f"{cid}: {sha1_col} mismatch (alignment); "
            f"stored {stored!r} != recomputed {expected!r}")
    return policy


def _parse_teacher_policy(case: dict, legal) -> list[float]:
    return _parse_policy_json(case, legal, "teacher_policy_json", "teacher_legal_moves_sha1")
```

(Error text keeps "length" / "alignment" / "normal" / the column name, so every existing v4 `match=` regex still passes — verify in Step 4.)

(b) Add a shared teacher-value validator and the root branch in `build_calibration_position` (insert the root branch after the existing `teacher_retention` branch, before the hard-value return):

```python
def _parse_teacher_value(case: dict) -> float:
    """Required raw eval-mode value anchor (side-to-move), finite in [-1, 1]."""
    raw = case.get("teacher_value")
    if raw in (None, ""):
        raise ValueError(
            f"{case.get('case_id')}: retention row needs teacher_value (raw stm anchor)")
    v = float(raw)
    if not math.isfinite(v) or not (-1.0 <= v <= 1.0):
        raise ValueError(
            f"{case.get('case_id')}: teacher_value {v!r} must be finite in [-1,1]")
    return v
```

Refactor the existing `teacher_retention` branch to use `_parse_teacher_value(case)` (byte-identical behavior — it performs the same checks; the "needs teacher_value" message is new-but-additive since v4 rows always populate it), then add:

```python
    if loss_mode == "mcts_root_retention":
        teacher_value = _parse_teacher_value(case)
        root_policy = _parse_policy_json(
            case, legal, "root_visits_json", "root_legal_moves_sha1")
        return PositionRecord(
            board_tensor=board_hwc,
            to_move=state.to_move,
            legal_moves=legal,
            visit_counts=root_policy,        # BASE MCTS root visit distribution (normalized)
            outcome=teacher_value,           # raw eval-mode value anchor, stm, DIRECT
            active_size=state.active_size,
            ply=position_ply,
            game_n_moves=None,
        )
```

(c) Extend the blank-guards in `build_calibration_sample` (currently only guards the 3 teacher columns on hard rows):

```python
    loss_mode = case.get("loss_mode") or "hard_value"
    if loss_mode == "hard_value":
        populated = [k for k in ("teacher_value", "teacher_policy_json",
                                 "teacher_legal_moves_sha1",
                                 "root_visits_json", "root_legal_moves_sha1")
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
```

Then set the metadata fields for root rows exactly as for teacher rows (reuse fields; update the trailing metadata assignments so `loss_mode in RETENTION_POLICY_LOSS_MODES` populates `teacher_value` / `teacher_policy_len`):

```python
    teacher_value = (float(case["teacher_value"])
                     if loss_mode in RETENTION_POLICY_LOSS_MODES else None)
    teacher_policy_len = (len(record.visit_counts)
                          if loss_mode in RETENTION_POLICY_LOSS_MODES else None)
```

(d) Extend `from_manifest` schema detection — detect modes from raw cases BEFORE building samples so the mixed check fires first:

```python
    @classmethod
    def from_manifest(cls, manifest_path, calibration_target: float):
        cases = load_csv_manifest(manifest_path)["cases"]
        modes = {(c.get("loss_mode") or "hard_value") for c in cases}
        retention_modes = sorted(modes & RETENTION_POLICY_LOSS_MODES)
        if len(retention_modes) > 1:
            raise ValueError(
                f"manifest mixes retention loss_modes {retention_modes}; "
                f"one retention mode per manifest")
        samples = [build_calibration_sample(c, calibration_target) for c in cases]
        has_weight_scale = any(c.get("weight_scale") not in (None, "") for c in cases)
        if "mcts_root_retention" in modes:
            schema = "mcts_root_retention"
        elif "teacher_retention" in modes:
            schema = "teacher_retention"
        elif any(c.get("target_black_value") not in (None, "") for c in cases):
            schema = "per_row_target"
        else:
            schema = "global_target"
        return cls(samples, has_weight_scale=has_weight_scale, schema=schema)
```

(e) Update the `build_calibration_position` docstring to name all three modes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py tests/test_build_teacher_calibration_manifest.py -q`
Expected: PASS — all new tests AND every pre-existing v4 pool/builder test (the `match=` regressions and byte-identical v4 parse are the point of running both files).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/calibration_pool.py tests/test_calibration_pool.py
git commit -m "feat(calibration): parse mcts_root_retention rows (raw-value anchor + root-visit policy)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Trainer string-gates + end-to-end mask proof

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` (two sites only)
- Test: `tests/test_calibration_loss.py`, `tests/test_training.py`

**Interfaces:**
- Consumes: Task 1's `RETENTION_POLICY_LOSS_MODES`; Task 2's parse branch; existing `alphazero_loss_batch` 14-tuple path (unchanged).
- Produces: the sampling gate routes `schema == "mcts_root_retention"` pools through `split_samples_with_modes` (mask present → 14-tuple); the startup print reports `mode=mcts_root_retention (N retention / M hard-value rows)` — no stale label.

- [ ] **Step 1: Write the failing tests**

(a) Append to `tests/test_calibration_loss.py` (this file already imports `create_network`, `alphazero_loss_batch`, `PositionRecord`, and the `_main_pos` helper — reuse them):

```python
def test_root_retention_flows_masked_policy_ce_path(tmp_path):
    """mcts_root_retention samples must produce mask=1.0 and drive the 14-tuple
    masked policy-CE path with a finite, positive CE (root visits != raw priors
    for a fresh network, so CE > 0 is expected — NOT ~0 like v4 self-distillation)."""
    import json as _json
    import math as _math
    from scripts.GPU.alphazero.calibration_pool import (
        build_calibration_sample, legal_moves_sha1, split_samples_with_modes)
    from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
    from tests.goal_line_probe_fixtures import legal_replay

    replay = legal_replay(9, game_idx=1)
    rp = tmp_path / "game_000001.json"
    rp.write_text(_json.dumps(replay))
    state = position_state(replay, 5, "black")
    legal = state.legal_moves()
    n = len(legal)
    # A sharp (non-uniform) root-visit target so CE > H(target) is comfortably > 0.
    visits = [0.0] * n
    visits[0] = 0.9
    if n > 1:
        visits[1] = 0.1
    case = {"game_idx": "1", "case_id": "root1", "replay_path": str(rp),
            "position_ply": "5", "side_to_move": "black",
            "tag": "old_post_opening_retention", "weight_scale": "1.0",
            "loss_mode": "mcts_root_retention", "teacher_value": "0.2",
            "root_visits_json": _json.dumps(visits),
            "root_legal_moves_sha1": legal_moves_sha1(legal)}
    sample = build_calibration_sample(case, calibration_target=-0.35)
    records, weights, mask = split_samples_with_modes([sample], has_weight_scale=True)
    assert mask.tolist() == [1.0]

    net = create_network(hidden=64, n_blocks=2)
    out = alphazero_loss_batch(
        net, records,
        calibration_positions=records,
        calibration_weights=weights,
        calibration_loss_weight=1.0,
        calibration_teacher_policy_mask=mask,
        teacher_value_weight=1.0, teacher_policy_kl_weight=0.25,
    )
    assert len(out) == 14                                  # 14-tuple teacher path
    ce = float(out[11])                                    # CALIB_POLICY_CE_IDX
    assert _math.isfinite(ce) and ce > 0.0
    assert int(out[13]) == 1                               # n_retention counts the root row
```

(b) Append to `tests/test_training.py` — the end-to-end make-or-break guard, cloned from `test_teacher_calibration_scalars_and_freeze_flag_in_model_iter_json` (`tests/test_training.py:454-511`) with a hand-written root manifest (no builder dependency — Task 4 comes later):

```python
def test_root_retention_run_persists_retention_telemetry():
    """v5 make-or-break: a mcts_root_retention manifest must route through the
    schema gate -> split_samples_with_modes -> mask=1.0 -> 14-tuple ->
    n_teacher_retention_drawn > 0 in model_iter JSON. If the mask predicate or
    the trainer schema gate misses the new mode, this reads 0 (silent v3 rerun)."""
    import csv as _csv
    import json as _json
    import math as _math
    from scripts.GPU.alphazero.trainer import train
    from scripts.GPU.alphazero.calibration_pool import legal_moves_sha1
    from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
    from tests.goal_line_probe_fixtures import legal_replay

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        replay = legal_replay(8, game_idx=1)
        rpath = tmp / "game_000001.json"
        rpath.write_text(_json.dumps(replay))
        state = position_state(replay, 5, "black")
        legal = state.legal_moves()
        n = len(legal)
        row = {"game_idx": "1", "case_id": "root1", "replay_path": str(rpath),
               "position_ply": "5", "side_to_move": "black",
               "tag": "old_post_opening_retention", "weight_scale": "1.0",
               "loss_mode": "mcts_root_retention", "teacher_value": "0.2",
               "root_visits_json": _json.dumps([1.0 / n] * n),
               "root_legal_moves_sha1": legal_moves_sha1(legal)}
        manifest = tmp / "v5_root.csv"
        with manifest.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader(); w.writerows([row])

        ckpt_dir = tmp / "ckpt"
        train(
            n_iterations=1, games_per_iteration=1, train_steps_per_iteration=5,
            batch_size=4, buffer_size=1000, checkpoint_dir=str(ckpt_dir),
            mcts_simulations=10, learning_rate=1e-3, hidden=64, n_blocks=2,
            max_moves=10, seed=42,
            post_opening_calibration_enabled=True,
            post_opening_calibration_manifest=str(manifest),
            post_opening_calibration_weight=0.02,
            post_opening_calibration_batch_fraction=0.10,
            freeze_batchnorm_stats=True,
            games_dir_override=str(tmp / "games"),
        )

        state_json = _json.loads(sorted(ckpt_dir.glob("model_iter_*.json"))[-1].read_text())
        assert state_json.get("n_teacher_retention_drawn", 0) > 0, \
            "root rows fell through mask=0.0 (v3-rerun hazard) or schema gate missed mcts_root_retention"
        assert _math.isfinite(float(state_json["calib_policy_ce_avg_iter"]))
        assert float(state_json["calib_policy_ce_avg_iter"]) > 0.0

    print("PASS: v5 root-retention telemetry persisted (mask flowed end-to-end)")
```

> `tempfile` and `Path` are already imported at the top of `test_training.py` (the v4 test uses them bare).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_loss.py -q -k root_retention && .venv/bin/python -m pytest tests/test_training.py -q -k root_retention`
Expected: the loss test PASSES already (mask + parse landed in Tasks 1-2 and the loss path is mode-agnostic — it is a characterization guard; the plan expects immediate green here, mirroring the v4 Task 5 characterization pattern). The **train test FAILS**: the trainer schema gate is still `== "teacher_retention"`, so the root pool takes the `split_samples` branch, `_calib_tp_mask=None`, 10-tuple, and `n_teacher_retention_drawn` stays 0 → the assertion message fires. Confirm THIS failure mode specifically — it is the exact silent-v3-rerun the task exists to prevent.

- [ ] **Step 3: Write the minimal implementation**

In `scripts/GPU/alphazero/trainer.py` — exactly two sites, locate by content:

(a) **Startup print block** (in `train()`, the `if effective_post_opening_calibration_weight > 0.0:` setup — currently `from .calibration_pool import CalibrationPool` then `if _calib_pool.schema == "teacher_retention":` printing `mode=teacher_retention (N teacher / M hard-value rows)`). Change the import line and the first branch:

```python
        from .calibration_pool import CalibrationPool, RETENTION_POLICY_LOSS_MODES
        _calib_pool = CalibrationPool.from_manifest(
            post_opening_calibration_manifest, post_opening_calibration_target)
        _sampling_desc = (
            f"tag_schedule={post_opening_calibration_tag_schedule}"
            if post_opening_calibration_tag_schedule
            else f"batch_fraction={post_opening_calibration_batch_fraction}")
        if _calib_pool.schema in RETENTION_POLICY_LOSS_MODES:
            _n_retention = sum(1 for _s in _calib_pool._samples
                               if _s.loss_mode in RETENTION_POLICY_LOSS_MODES)
            print(f"Post-opening calibration: {len(_calib_pool)} positions, "
                  f"mode={_calib_pool.schema} ({_n_retention} retention / "
                  f"{len(_calib_pool) - _n_retention} hard-value rows), "
                  f"weight={effective_post_opening_calibration_weight}, "
                  f"{_sampling_desc}")
        elif _calib_pool.schema == "per_row_target":
```

(`mode=` now prints the actual schema — `teacher_retention` runs keep `mode=teacher_retention`, root runs print `mode=mcts_root_retention`; the parenthetical wording changes from "N teacher / M hard-value" to "N retention / M hard-value" for both, which no test asserts. This is the stale-label-bug site — the fix is printing the schema variable, not a literal.)

(b) **Sampling gate** (in the train-steps loop — currently `from .calibration_pool import split_samples, split_samples_with_modes` then `if _calib_pool.schema == "teacher_retention":`):

```python
                            from .calibration_pool import (
                                split_samples, split_samples_with_modes,
                                RETENTION_POLICY_LOSS_MODES)
                            ...
                            if _calib_pool.schema in RETENTION_POLICY_LOSS_MODES:
                                _calib_batch, _calib_weights, _calib_tp_mask = (
                                    split_samples_with_modes(_calib_samples,
                                                             _calib_pool.has_weight_scale))
                            else:
                                _calib_batch, _calib_weights = split_samples(
                                    _calib_samples, _calib_pool.has_weight_scale)
                                _calib_tp_mask = None
```

(Only the import line and the `if` condition change; everything between stays byte-identical.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_loss.py tests/test_training.py -q`
Expected: PASS — the new train test now reads `n_teacher_retention_drawn > 0`; ALL pre-existing training tests green (v4 teacher run test at `test_training.py:454` proves the teacher path still routes identically).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py tests/test_calibration_loss.py tests/test_training.py
git commit -m "feat(trainer): route mcts_root_retention schema through the masked retention path

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Offline v5 manifest builder

**Files:**
- Create: `scripts/GPU/alphazero/build_mcts_root_retention_manifest.py`
- Test: `tests/test_build_mcts_root_retention_manifest.py`

**Interfaces:**
- Consumes: `load_csv_manifest` (`position_probe_cases.py:24`), `position_state` (`goal_line_trigger_probe_cases.py:73`), `_teacher_infer` (`build_teacher_calibration_manifest.py:28` — cross-module import precedent: `eval_raw_nn_position_rows.py`), `legal_moves_sha1` (`calibration_pool.py:23`). Real path only (inside `main()` / the real search factory): `MCTS`/`MCTSConfig` via `cfg_from(EvalConfig(...))` (`eval_runner.py:55-95`), `_default_evaluator_factory` (`eval_runner.py:196` — gate BN mode by construction), `load_network_for_scoring` + `LocalGPUEvaluator` (eval-mode raw anchor).
- Produces: `build_rows(rows, raw_evaluator, search_fn, *, sims, base_checkpoint, pos_base_seed, goal_base_seed, eval_batch_size, stall_flush_sims) -> list[dict]` where `search_fn(state, seed) -> (counts_dict, root_value_stm)` is injectable for tests; `row_seed(...)`, `dense_normalized_visits(...)`, `cross_check_gate_values(...)`; a CLI `main()`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_mcts_root_retention_manifest.py`:

```python
import json

import numpy as np
import pytest

from scripts.GPU.alphazero.build_mcts_root_retention_manifest import (
    build_rows, cross_check_gate_values, dense_normalized_visits, row_seed)
from scripts.GPU.alphazero.calibration_pool import (
    build_calibration_sample, legal_moves_sha1)
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
from tests.goal_line_probe_fixtures import legal_replay


class _FakeRawEval:
    """Eval-mode raw-anchor stand-in (LocalGPUEvaluator API)."""
    def build_input_tensor(self, state):
        return state.to_tensor()
    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        b, m = move_mask.shape
        priors = move_mask / np.maximum(move_mask.sum(axis=1, keepdims=True), 1.0)
        return priors.astype(np.float32), np.full((b,), 0.2, dtype=np.float32)


def _fake_search(state, seed):
    """Deterministic fake gate search: all visits on the first legal move except one."""
    legal = state.legal_moves()
    counts = {m: 0 for m in legal}
    counts[legal[0]] = 399
    counts[legal[-1]] = 1
    return counts, -0.1389        # root value, stm


def _rows(tmp_path):
    rp = tmp_path / "game_000001.json"
    rp.write_text(json.dumps(legal_replay(9, game_idx=1)))
    return [
        {"game_idx": "1", "case_id": "corr1", "replay_path": str(rp),
         "position_ply": "5", "side_to_move": "black",
         "tag": "black_predrop_correction", "target_black_value": "-0.35",
         "weight_scale": "1.0"},
        {"game_idx": "1", "case_id": "game_000001_ply_005", "replay_path": str(rp),
         "position_ply": "5", "side_to_move": "black",
         "tag": "old_post_opening_retention", "target_black_value": "-0.11",
         "weight_scale": "1.0"},
    ]


def _build(tmp_path):
    return build_rows(_rows(tmp_path), _FakeRawEval(), _fake_search,
                      sims=400, base_checkpoint="ckpt/base.safetensors",
                      pos_base_seed=20260616, goal_base_seed=20260614,
                      eval_batch_size=14, stall_flush_sims=48)


def test_row_seed_matches_gate_probe_schemes():
    # position-probe families: base ^ game ^ ply (eval_position_probe.py:76)
    assert row_seed("old_post_opening_retention", 7, 51, 20260616, 20260614) == (20260616 ^ 7 ^ 51)
    assert row_seed("red_predrop_retention", 7, 51, 20260616, 20260614) == (20260616 ^ 7 ^ 51)
    # goal-line family: base ^ game only (eval_goal_line_trigger_probe.py:69)
    assert row_seed("goal_line_retention", 7, 51, 20260616, 20260614) == (20260614 ^ 7)


def test_dense_normalized_visits_aligned_and_zero_total_rejected(tmp_path):
    rp = tmp_path / "game_000001.json"
    rp.write_text(json.dumps(legal_replay(9, game_idx=1)))
    state = position_state(legal_replay(9, game_idx=1), 5, "black")
    legal = state.legal_moves()
    counts = {m: 0 for m in legal}
    counts[legal[0]] = 3
    counts[legal[1]] = 1
    dense = dense_normalized_visits(counts, legal, "c1")
    assert len(dense) == len(legal)
    assert dense[0] == pytest.approx(0.75) and dense[1] == pytest.approx(0.25)
    assert sum(dense) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="zero"):
        dense_normalized_visits({m: 0 for m in legal}, legal, "c1")


def test_builder_blanks_correction_and_fills_root_retention(tmp_path):
    out = _build(tmp_path)
    corr = next(r for r in out if r["case_id"] == "corr1")
    ret = next(r for r in out if r["case_id"] == "game_000001_ply_005")
    assert corr["loss_mode"] == "hard_value"
    assert corr["teacher_value"] == "" and corr["root_visits_json"] == ""
    assert corr["root_legal_moves_sha1"] == "" and corr["root_value_stm"] == ""
    assert corr["target_black_value"] == "-0.35"          # A hard target PRESERVED

    assert ret["loss_mode"] == "mcts_root_retention"
    assert abs(float(ret["teacher_value"]) - 0.2) < 1e-6  # raw eval-mode anchor
    assert abs(float(ret["root_value_stm"]) - (-0.1389)) < 1e-9
    assert abs(float(ret["root_black_value"]) - (-0.1389)) < 1e-9  # black to move: no flip
    assert ret["target_black_value"] == ""                # stale v3 MCTS scalar blanked
    policy = json.loads(ret["root_visits_json"])
    assert abs(sum(policy) - 1.0) < 1e-6
    assert max(policy) == pytest.approx(399 / 400)
    # provenance stamps
    assert ret["root_sims"] == "400"
    assert ret["root_base_checkpoint"] == "ckpt/base.safetensors"
    assert ret["root_seed"] == str(20260616 ^ 1 ^ 5)
    assert ret["root_mcts_eval_batch_size"] == "14"
    assert ret["root_mcts_stall_flush_sims"] == "48"
    # sha1 matches the actually reconstructed legal order
    state = position_state(legal_replay(9, game_idx=1), 5, "black")
    assert ret["root_legal_moves_sha1"] == legal_moves_sha1(state.legal_moves())


def test_builder_output_passes_v5_parser(tmp_path):
    out = _build(tmp_path)
    ret = next(r for r in out if r["loss_mode"] == "mcts_root_retention")
    sample = build_calibration_sample(ret, calibration_target=-0.35)
    assert sample.loss_mode == "mcts_root_retention"
    assert abs(sample.record.outcome - 0.2) < 1e-6        # value = RAW anchor, not root value
    assert abs(sum(sample.record.visit_counts) - 1.0) < 1e-6


def test_cross_check_gate_values(tmp_path):
    out = _build(tmp_path)
    gate_csv = tmp_path / "position_probe_cases.csv"
    gate_csv.write_text(
        "checkpoint,case_id,probe_black_root_value\n"
        "calib020_0001,game_000001_ply_005,-0.1389\n")
    stats = cross_check_gate_values(out, [str(gate_csv)], tol=1e-3)
    assert stats["checked"] == 1 and stats["unmatched"] == 0

    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text(
        "checkpoint,case_id,probe_black_root_value\n"
        "calib020_0001,game_000001_ply_005,0.9\n")
    with pytest.raises(ValueError, match="cross-check"):
        cross_check_gate_values(out, [str(bad_csv)], tol=1e-3)


def test_builder_module_defers_heavy_imports():
    """MLX/MCTS must not load at import time (tests run with fakes)."""
    import importlib, sys
    for mod in ("mlx", "mlx.core"):
        sys.modules.pop(mod, None)
    m = importlib.import_module(
        "scripts.GPU.alphazero.build_mcts_root_retention_manifest")
    src = open(m.__file__).read()
    head = src.split("def ", 1)[0]                 # module-level import block
    assert "eval_runner" not in head and "local_evaluator" not in head
    assert "probe_eval" not in head
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_build_mcts_root_retention_manifest.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.GPU.alphazero.build_mcts_root_retention_manifest'`.

- [ ] **Step 3: Write the minimal implementation**

Create `scripts/GPU/alphazero/build_mcts_root_retention_manifest.py`:

```python
"""Deterministic v5 root-retention manifest builder.

For each retention row of a v3-style stratified source manifest, computes TWO
targets against the BASE checkpoint:
  1. teacher_value  — raw single-position forward in EVAL-mode BatchNorm
     (matches the training-path eval-mode calibration forward, so the value
     term starts ~0 at gate-0), via the shared _teacher_infer.
  2. root_visits_json / root_value_stm — a 400-sim MCTS search using the GATE
     probes' exact loader (_default_evaluator_factory: NO eval(), train-mode
     BatchNorm, batch=1 sync search) and per-family gate seeds, so BASE root
     values reproduce the gate CSVs by construction.

Correction rows (tag == black_predrop_correction) pass through with all new
columns blank and target_black_value PRESERVED. Retention rows blank
target_black_value (stale v3 MCTS-root scalar) and teacher_policy_json.

See the v5 section of docs/2026-06-26-targeted-value-calibration-experiment-
ledger-v3f-v4-overlap-updated.md and the plan doc for the two-evaluator split.
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
from .calibration_pool import legal_moves_sha1

CORRECTION_TAG = "black_predrop_correction"
GOAL_LINE_TAG = "goal_line_retention"
NEW_COLUMNS = [
    "loss_mode", "teacher_value",
    "root_value_stm", "root_black_value", "root_visits_json",
    "root_legal_moves_sha1", "root_sims", "root_base_checkpoint", "root_seed",
    "root_mcts_eval_batch_size", "root_mcts_stall_flush_sims",
    # blanked-on-purpose teacher columns (kept so a v4-source manifest can't
    # leak stale raw-prior targets through the v5 contamination guard):
    "teacher_policy_json", "teacher_legal_moves_sha1",
]


def row_seed(tag: str, game_idx: int, position_ply: int,
             pos_base_seed: int, goal_base_seed: int) -> int:
    """Replicate the gate probes' per-case rng seeds exactly.

    goal_line_retention rows came from eval_goal_line_trigger_probe
    (seed = base ^ game_idx); every other retention family came from
    eval_position_probe (seed = base ^ game_idx ^ position_ply).
    """
    if tag == GOAL_LINE_TAG:
        return goal_base_seed ^ int(game_idx)
    return pos_base_seed ^ int(game_idx) ^ int(position_ply)


def dense_normalized_visits(counts: dict, legal, case_id: str) -> list[float]:
    """Dense visit vector aligned to legal order, normalized to sum 1.0.

    INTENTIONALLY hand-normalized instead of reusing MCTS.get_policy_target:
    that helper (mcts.py:1229-1245) returns a DICT {move: prob}, not a dense
    vector aligned to the manifest legal order that root_legal_moves_sha1
    pins, and it silently falls back to UNIFORM when total visits == 0. Here
    zero total visits is a build FAILURE (loud), never a uniform fallback —
    a uniform target would silently anchor garbage.
    """
    total = float(sum(counts.get(m, 0) for m in legal))
    if total <= 0:
        raise ValueError(f"{case_id}: root search returned zero total visits")
    return [counts.get(m, 0) / total for m in legal]


def _to_black(value_stm: float, side_to_move: str) -> float:
    if side_to_move == "black":
        return float(value_stm)
    if side_to_move == "red":
        return float(-value_stm)
    raise ValueError(f"unexpected side_to_move {side_to_move!r}")


def build_rows(rows: list, raw_evaluator, search_fn, *, sims: int,
               base_checkpoint: str, pos_base_seed: int, goal_base_seed: int,
               eval_batch_size: int, stall_flush_sims: int) -> list[dict]:
    out = []
    for r in rows:
        row = dict(r)                            # preserve ALL source columns
        for c in NEW_COLUMNS:
            row[c] = ""
        if r.get("tag") == CORRECTION_TAG:
            # A-correction: hard target stays; every retention column blank.
            row["loss_mode"] = "hard_value"
            out.append(row)
            continue
        cid = r.get("case_id")
        replay = json.loads(Path(r["replay_path"]).read_text())
        ply = int(float(r["position_ply"]))
        side = r["side_to_move"]
        state = position_state(replay, ply, side)
        legal = state.legal_moves()

        # (1) raw eval-mode value anchor (matches training-path eval forward).
        _, _, raw_value = _teacher_infer(state, raw_evaluator)

        # (2) gate-faithful BASE root search.
        seed = row_seed(r.get("tag", ""), r["game_idx"], ply,
                        pos_base_seed, goal_base_seed)
        counts, root_value_stm = search_fn(state, seed)
        dense = dense_normalized_visits(counts, legal, cid)

        row["loss_mode"] = "mcts_root_retention"
        row["teacher_value"] = repr(float(raw_value))
        row["root_value_stm"] = repr(float(root_value_stm))
        row["root_black_value"] = repr(_to_black(root_value_stm, side))
        row["root_visits_json"] = json.dumps(dense)
        row["root_legal_moves_sha1"] = legal_moves_sha1(legal)
        row["root_sims"] = str(sims)
        row["root_base_checkpoint"] = base_checkpoint
        row["root_seed"] = str(seed)
        row["root_mcts_eval_batch_size"] = str(eval_batch_size)
        row["root_mcts_stall_flush_sims"] = str(stall_flush_sims)
        row["target_black_value"] = ""      # blank stale v3 MCTS-root scalar
        out.append(row)
    return out


def cross_check_gate_values(out_rows: list, gate_csv_paths: list, tol: float) -> dict:
    """Builder sanity gate: for every retention row whose case_id appears in a
    gate cases CSV, the recomputed root_black_value must match the gate's
    probe_black_root_value within tol. Proves the search config/seeds/BN mode
    reproduce the gate setup. Raises on any mismatch."""
    gate = {}
    for path in gate_csv_paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                gate[r["case_id"]] = float(r["probe_black_root_value"])
    checked, unmatched, errors = 0, 0, []
    for row in out_rows:
        if row.get("loss_mode") != "mcts_root_retention":
            continue
        cid = row["case_id"]
        if cid not in gate:
            unmatched += 1
            continue
        checked += 1
        got = float(row["root_black_value"])
        want = gate[cid]
        if abs(got - want) > tol:
            errors.append(f"{cid}: recomputed {got:+.4f} vs gate {want:+.4f}")
    if errors:
        raise ValueError(
            "gate cross-check FAILED (wrong seeds / BN mode / config?): "
            + "; ".join(errors))
    return {"checked": checked, "unmatched": unmatched}


def _real_search_fn(base_checkpoint: str, sims: int,
                    eval_batch_size: int, stall_flush_sims: int):
    """Gate-faithful search factory. Heavy imports deferred: MLX loads here,
    NOT at module import (tests run with fakes). Uses the gate probes' exact
    loader (_default_evaluator_factory: no eval(), compile=True)."""
    from .eval_runner import EvalConfig, cfg_from, _default_evaluator_factory
    from .mcts import MCTS
    evaluator = _default_evaluator_factory(base_checkpoint)
    cfg = cfg_from(EvalConfig(mcts_sims=sims,
                              mcts_eval_batch_size=eval_batch_size,
                              mcts_stall_flush_sims=stall_flush_sims))

    def search_fn(state, seed):
        return MCTS(evaluator, cfg, random.Random(seed)).search(state, add_noise=False)

    return search_fn


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the v5 mcts-root-retention manifest.")
    ap.add_argument("--source", required=True, help="v3-style stratified manifest CSV")
    ap.add_argument("--base-checkpoint", required=True, help=".safetensors BASE (= teacher)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--position-probe-base-seed", type=int, default=20260616)
    ap.add_argument("--goal-line-base-seed", type=int, default=20260614)
    ap.add_argument("--eval-batch-size", type=int, default=14)
    ap.add_argument("--stall-flush-sims", type=int, default=48)
    ap.add_argument("--gate-cases-csv", action="append", default=[],
                    help="gate cases CSV (repeatable) for the root-value cross-check; "
                         "STRONGLY recommended")
    ap.add_argument("--gate-tolerance", type=float, default=1e-3)
    args = ap.parse_args(argv)

    from .local_evaluator import LocalGPUEvaluator
    from .probe_eval import load_network_for_scoring
    rows = load_csv_manifest(args.source)["cases"]
    # Raw anchor evaluator: EVAL-mode BN (running stats) — matches the
    # training-path eval-mode calibration forward, so gate-0 value term ~ 0.
    network, *_ = load_network_for_scoring(args.base_checkpoint)
    network.eval()
    raw_evaluator = LocalGPUEvaluator(network)
    # Root search evaluator: the GATE loader, by construction (separate load;
    # do NOT share the eval()'d network above).
    search_fn = _real_search_fn(args.base_checkpoint, args.sims,
                                args.eval_batch_size, args.stall_flush_sims)

    out_rows = build_rows(rows, raw_evaluator, search_fn, sims=args.sims,
                          base_checkpoint=args.base_checkpoint,
                          pos_base_seed=args.position_probe_base_seed,
                          goal_base_seed=args.goal_line_base_seed,
                          eval_batch_size=args.eval_batch_size,
                          stall_flush_sims=args.stall_flush_sims)
    if args.gate_cases_csv:
        stats = cross_check_gate_values(out_rows, args.gate_cases_csv,
                                        args.gate_tolerance)
        print(f"gate cross-check PASS: {stats['checked']} matched, "
              f"{stats['unmatched']} retention rows without a gate row")
    else:
        print("WARNING: no --gate-cases-csv given; root targets NOT cross-checked "
              "against the gate CSVs")

    base_columns = list(rows[0].keys()) if rows else []
    fieldnames = base_columns + [c for c in NEW_COLUMNS if c not in base_columns]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    n_ret = sum(1 for r in out_rows if r["loss_mode"] == "mcts_root_retention")
    print(f"wrote {len(out_rows)} rows ({n_ret} mcts_root_retention) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_build_mcts_root_retention_manifest.py tests/test_calibration_pool.py -q`
Expected: PASS (7 new builder tests; pool tests untouched-green).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/build_mcts_root_retention_manifest.py tests/test_build_mcts_root_retention_manifest.py
git commit -m "feat(calibration): v5 root-retention manifest builder (gate-faithful search + raw anchor)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: v5 mechanics smoke + docs

**Files:**
- Create: `scripts/GPU/alphazero/smoke_mcts_root_retention_v5.py`
- Test: `tests/test_smoke_mcts_root_retention_v5.py`
- Modify: `docs/post-game-analysis.md` (append §8 after the v4 §7 block)

**Interfaces:**
- Consumes: Tasks 1-4 (pool parse, mask, builder); `alphazero_loss_batch` + `CALIB_VALUE_TERM_IDX`/`CALIB_POLICY_CE_IDX`/`CALIB_POLICY_KL_EST_IDX` (`trainer.py:55-58`); mirrors `smoke_teacher_calibration_v4.assert_self_distillation` structure.
- Produces: `assert_root_retention_mechanics(network, manifest_path, value_tol=1e-4) -> dict` with keys `value_mse`, `policy_ce`, `kl_est`, `n_retention`.

**Semantics (the v4↔v5 difference, stated for the implementer):** at gate-0 (candidate == BASE), the **value term must be ≈ 0** (raw eval-mode anchor reproduces under the eval-mode calibration forward) but the **policy CE must NOT be ≈ 0** — root visits are search-improved, so `kl_est = CE − H(target) > 0` is EXPECTED. The smoke asserts value ≤ tol, CE finite and > 0, kl_est finite and ≥ −1e-6 (numerical floor), and REPORTS (never gates on) the kl_est magnitude.

- [ ] **Step 1: Write the failing test**

Create `tests/test_smoke_mcts_root_retention_v5.py`:

```python
import csv
import json

import numpy as np
import pytest

from scripts.GPU.alphazero.build_mcts_root_retention_manifest import build_rows
from tests.goal_line_probe_fixtures import legal_replay


def _sharp_search(state, seed):
    legal = state.legal_moves()
    counts = {m: 0 for m in legal}
    counts[legal[0]] = 400                     # sharp target != raw priors
    return counts, 0.3


def _manifest_from_net(tmp_path, net):
    """1 retention row whose raw anchor comes from THIS network (gate-0: base==candidate)."""
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    rp = tmp_path / "game_000001.json"
    rp.write_text(json.dumps(legal_replay(9, game_idx=1)))
    net.eval()                                  # raw anchor cached in eval mode (mirrors builder main())
    rows = build_rows(
        [{"game_idx": "1", "case_id": "r1", "replay_path": str(rp),
          "position_ply": "5", "side_to_move": "black",
          "tag": "old_post_opening_retention", "weight_scale": "1.0"}],
        LocalGPUEvaluator(net), _sharp_search,
        sims=400, base_checkpoint="in-memory",
        pos_base_seed=20260616, goal_base_seed=20260614,
        eval_batch_size=14, stall_flush_sims=48)
    man = tmp_path / "v5.csv"
    with man.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return man


def test_root_retention_mechanics_gate0(tmp_path):
    from scripts.GPU.alphazero.smoke_mcts_root_retention_v5 import (
        assert_root_retention_mechanics)
    from scripts.GPU.alphazero.network import create_network

    net = create_network(hidden=64, n_blocks=2)
    man = _manifest_from_net(tmp_path, net)
    net.train()                                 # production-faithful: net in TRAIN at smoke time
    stats = assert_root_retention_mechanics(net, str(man), value_tol=1e-4)
    assert abs(stats["value_mse"]) < 1e-4       # raw anchor reproduces (eval-mode forward)
    assert np.isfinite(stats["policy_ce"]) and stats["policy_ce"] > 0.0
    assert stats["kl_est"] > 1e-3               # sharp target vs raw priors: genuinely > 0
    assert stats["n_retention"] == 1


def test_smoke_rejects_wrong_schema(tmp_path):
    from scripts.GPU.alphazero.smoke_mcts_root_retention_v5 import (
        assert_root_retention_mechanics)
    from scripts.GPU.alphazero.network import create_network
    man = tmp_path / "hard.csv"
    rp = tmp_path / "game_000001.json"
    rp.write_text(json.dumps(legal_replay(9, game_idx=1)))
    with man.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "game_idx", "case_id", "replay_path", "position_ply",
            "side_to_move", "target_black_value"])
        w.writeheader()
        w.writerow({"game_idx": "1", "case_id": "h1", "replay_path": str(rp),
                    "position_ply": "5", "side_to_move": "black",
                    "target_black_value": "-0.35"})
    with pytest.raises(AssertionError, match="schema"):
        assert_root_retention_mechanics(create_network(hidden=64, n_blocks=2), str(man))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_smoke_mcts_root_retention_v5.py -q`
Expected: FAIL — `ModuleNotFoundError: ... smoke_mcts_root_retention_v5`.

- [ ] **Step 3: Write the minimal implementation**

Create `scripts/GPU/alphazero/smoke_mcts_root_retention_v5.py`:

```python
"""v5 gate-0 mechanics smoke. UNLIKE the v4 self-distillation smoke, policy
CE ~ 0 is NOT expected here: root visits are search-improved, so at gate-0
(candidate == BASE) kl_est = CE - H(target) > 0 is the healthy state. The
smoke asserts only what must hold: value term ~ 0 (raw eval-mode anchor
reproduces under the eval-mode calibration forward), policy CE finite and
positive, mask aligned, no NaN. Run after building the v5 manifest, before
training. Pair with the builder's --gate-cases-csv cross-check (root values
vs gate CSVs), which validates the search targets themselves.
"""
from __future__ import annotations

import argparse
import math
import sys

from .calibration_pool import (
    CalibrationPool, split_samples_with_modes, RETENTION_POLICY_LOSS_MODES)
from .trainer import (
    alphazero_loss_batch, CALIB_VALUE_TERM_IDX, CALIB_POLICY_CE_IDX,
    CALIB_POLICY_KL_EST_IDX, CALIB_N_RETENTION_IDX)


def assert_root_retention_mechanics(network, manifest_path: str,
                                    value_tol: float = 1e-4) -> dict:
    pool = CalibrationPool.from_manifest(manifest_path, calibration_target=-0.35)
    if pool.schema != "mcts_root_retention":
        raise AssertionError(
            f"manifest schema is {pool.schema!r}, expected mcts_root_retention")
    retention = [s for s in pool._samples
                 if s.loss_mode in RETENTION_POLICY_LOSS_MODES]
    if not retention:
        raise AssertionError("no mcts_root_retention rows in manifest")
    records, weights, mask = split_samples_with_modes(retention, pool.has_weight_scale)
    if not all(m == 1.0 for m in mask.tolist()):
        raise AssertionError("retention rows produced mask != 1.0 (v3-rerun hazard)")
    prev_training = network.training
    network.eval()      # batch-independent forward; loss path re-wraps in eval anyway
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
            f"raw value anchor FAILED to reproduce: value_mse={value_mse:.3e} "
            f"(tol={value_tol}). Check eval-mode caching / checkpoint / perspective.")
    if policy_ce <= 0.0:
        raise AssertionError(f"policy CE not positive: {policy_ce!r}")
    if kl_est < -1e-6:
        raise AssertionError(f"kl_est negative beyond numerical floor: {kl_est!r}")
    return {"value_mse": value_mse, "policy_ce": policy_ce,
            "kl_est": kl_est, "n_retention": n_retention}


def main(argv=None):
    ap = argparse.ArgumentParser(description="v5 gate-0 root-retention mechanics smoke")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--value-tol", type=float, default=1e-4)
    args = ap.parse_args(argv)
    from .probe_eval import load_network_for_scoring
    network, *_ = load_network_for_scoring(args.base_checkpoint)
    stats = assert_root_retention_mechanics(network, args.manifest,
                                            value_tol=args.value_tol)
    print(f"PASS v5 gate-0 mechanics: value_mse={stats['value_mse']:.3e}, "
          f"policy_ce={stats['policy_ce']:.4f}, kl_est={stats['kl_est']:.4f} "
          f"(EXPECTED > 0 — root visits are search-improved), "
          f"n_retention={stats['n_retention']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Then append to `docs/post-game-analysis.md`, directly after the v4 §7 block, a new section:

```markdown
## 8. Targeted Value Calibration v5 — MCTS-root-visit policy retention

v5 keeps the v4 raw-teacher VALUE anchor on retention rows but replaces the
policy target with BASE's 400-sim MCTS root visit distribution (dense,
normalized, sha1-pinned). Rationale + full experiment record: the v5 section
of `docs/2026-06-26-targeted-value-calibration-experiment-ledger-v3f-v4-overlap-updated.md`
(root-value-only = v3, rejected; raw-priors policy = v4, rejected).

Build (offline, once, frozen):

    .venv/bin/python -m scripts.GPU.alphazero.build_mcts_root_retention_manifest \
      --source logs/eval/targeted_calibration_v3_strat_from_calib020_0001.csv \
      --base-checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
      --gate-cases-csv <BASE position_probe_cases.csv> \
      --gate-cases-csv <BASE goal_line_trigger_probe_cases.csv> \
      --out logs/eval/targeted_calibration_v5_root_from_calib020_0001.csv

Gate-0 smoke (value ~0 REQUIRED; policy CE > 0 EXPECTED — do not "fix" it):

    .venv/bin/python -m scripts.GPU.alphazero.smoke_mcts_root_retention_v5 \
      --manifest logs/eval/targeted_calibration_v5_root_from_calib020_0001.csv \
      --base-checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors

Training reuses the v4 command verbatim with two deltas: the v5 manifest path
and a fresh checkpoint dir. Same flags: weight 0.01, teacher-value-weight 1.0,
teacher-policy-kl-weight 0.25, tag schedule 2:1:2:1, --freeze-batchnorm-stats.
Gates A–D vs calib020_0001; no promotion unless all four pass.

Known limitation (recorded in the ledger): root-visit anchors constrain the
candidate's policy AT the anchored root positions only. If gate drift comes
from value/prior changes deeper in the tree, v5 can still fail — in that case
the next hypothesis is tree/path-level retention, not more rows or weights.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_smoke_mcts_root_retention_v5.py -q`
Expected: PASS (2 tests).

Then the full v5 surface: `.venv/bin/python -m pytest tests/test_calibration_pool.py tests/test_calibration_loss.py tests/test_build_mcts_root_retention_manifest.py tests/test_build_teacher_calibration_manifest.py tests/test_smoke_mcts_root_retention_v5.py tests/test_training.py -q`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/smoke_mcts_root_retention_v5.py tests/test_smoke_mcts_root_retention_v5.py docs/post-game-analysis.md
git commit -m "feat(calibration): v5 gate-0 mechanics smoke + operator docs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Post-implementation validation (operator run — NOT a code task)

Executed manually by the operator after merge; the implementing session STOPS after Task 5 + final review.

1. **Confirm untouched surfaces:** `git diff --stat <merge-base>..HEAD` shows ONLY: the 2 new scripts, 2 new test files, `calibration_pool.py`, `trainer.py`, the 3 extended test files, `docs/post-game-analysis.md`, and the plan doc. `mcts.py`, `self_play.py`, `eval_runner.py`, both gate probes, `build_teacher_calibration_manifest.py`, `smoke_teacher_calibration_v4.py` ABSENT from the diff.
2. **Build the v5 manifest** with the §8 command. The `--gate-cases-csv` paths are the BASE (`calib020_0001`) gate runs' per-case CSVs (`position_probe_cases.csv` from the C/D gate output dirs, `goal_line_trigger_probe_cases.csv` from the B gate output dir — operator locates the exact `logs/eval/...` dirs from the gate-run logs). Expect `gate cross-check PASS`. If the cross-check fails: wrong seeds / BN mode / sims — reconcile BEFORE training; do not loosen the tolerance to pass.
3. **Run the gate-0 smoke.** Expect `value_mse ≈ 0` (≤1e-4) and a **positive** `policy_ce`/`kl_est` — that is the designed state, not a failure.
4. **Training run:** v4 command verbatim with the v5 manifest + fresh checkpoint dir (`checkpoints/alphazero-v5-root-from-calib020-0001`), `--iterations 1`, weights 0.01 / 1.0 / 0.25, schedule 2:1:2:1, `--freeze-batchnorm-stats`. Verify the startup line prints `mode=mcts_root_retention (...retention / ...hard-value rows)` and the iter sidecar shows `n_teacher_retention_drawn > 0` with finite `calib_policy_ce_avg_iter`.
5. **Gates A–D** vs `calib020_0001` (400 sims, standard commands). **No promotion match unless all four pass.**
6. **Ledger:** append the v5 row + result to the experiment ledger; if v5 fails with raw value AND root policy held at the anchors, the recorded next hypothesis is tree/path-level retention (do NOT sweep v5 weights).

## Acceptance criteria

1. `.venv/bin/python -m pytest tests/ -q` fully green (baseline 1231 + new tests; zero pre-existing failures).
2. `mcts_root_retention` rows provably get mask = 1.0 (Task 1 unit test) AND drive `n_teacher_retention_drawn > 0` through a real `train()` run (Task 3 test) — the anti-v3-rerun guarantee.
3. v2/v3/v4 paths byte-identical: all pre-existing calibration/training tests pass unmodified.
4. Builder output round-trips through `CalibrationPool.from_manifest` (schema `mcts_root_retention`) and the gate cross-check machinery works (match + mismatch tested).
5. The smoke enforces value ≈ 0 / CE > 0 semantics and rejects wrong-schema manifests.
6. No modification to MCTS, self-play, gate probes, eval_runner, or the v4 builder/smoke.

## Self-Review

**Spec coverage:** locked-design points 1–6 → Task 4 (builder, two evaluators, provenance, gate cross-check), Tasks 1–3 (mode handling: all FOUR `"teacher_retention"` literals addressed — pool mask :277 T1, pool schema :252 T2, trainer gate :3921 T3, trainer print :2889 T3), Task 2 (parse + blank-guards), Task 5 (smoke semantics + docs + limitation). User's five do-not rules → Global Constraints lines 1, 2, 3 (mask hazard), and the two-evaluator split. User's critical-test list → mapped in the File Structure section.

**Placeholder scan:** none — every step carries complete code and exact commands. The two operator placeholders (`<BASE ... cases.csv>`) are in the NON-task operator section by design (paths live in the operator's gate-run logs, not in the repo).

**Type consistency:** `RETENTION_POLICY_LOSS_MODES` spelled identically in Tasks 1/2/3/5; `search_fn(state, seed) -> (counts_dict, root_value_stm)` consistent between Task 4 impl and both fake searchers; builder column names (`root_visits_json`, `root_legal_moves_sha1`, `teacher_value`) match Task 2's parser exactly; `CALIB_*_IDX` constants exist at `trainer.py:55-58` (verified, v4 Task 5); `build_rows` keyword signature identical at all three call sites (Task 4 tests, Task 5 test, Task 4 `main()`).
