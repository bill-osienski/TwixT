# Targeted Value Calibration v6c — D Root-Value-Only Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add value-only D root-value retention rows to the v6 line: for each `red_predrop_retention` source row, the builder emits a depth-0 clone tagged `red_predrop_root_value_retention` that anchors the raw eval-mode `teacher_value` at the ROOT state with zero policy signal.

**Architecture:** Three small deltas on the merged v6 machinery (origin/main @ 9c2e693): (1) the loader accepts empty `extra_moves_json` for depth-0 rows gated on `continuation_source == "root_value"` (all other validation — side, sha1, mask=0 — unchanged); (2) the v6 builder gains a disabled-by-default `--emit-d-root-value-rows` flag that appends the clone after each D parent; (3) the gate-0 smoke gains a `V6C_TAG_SCHEDULE` and a `--schedule {v6,v6c}` flag. No trainer/loss/MCTS changes at all.

**Tech Stack:** Python 3.14 / MLX (Apple Metal), pytest, CSV manifests, existing v6 calibration stack under `scripts/GPU/alphazero/`.

**Requirements source:** user's v6c message of 2026-07-02 (design decisions LOCKED: reuse `searched_continuation_retention` with depth 0 rather than a new loss mode; new tag `red_predrop_root_value_retention`; schedule `2:1:2:1:2`; weight 0.01; new manifest file, never overwrite v6). Rationale: v6 showed value-only continuation doesn't fix the D root raw failure; v6b showed D root rows help D but root policy CE/KL breaks B/C; v6c keeps the D root VALUE anchor and drops the policy/root-visit part.

## Global Constraints

- Python: always `.venv/bin/python` from the repo root; tests: `.venv/bin/python -m pytest <file> -v`; full suite `.venv/bin/python -m pytest tests/ -q` (main-checkout baseline: 1294 passed).
- NEVER `sys.modules.pop("mlx")` (or any mlx submodule) in tests — a later fresh `import mlx.core` re-inits the native Metal module and SIGABRTs the suite.
- **do-not-repeat #9 (experiment ledger):** the clone's `teacher_value` is the source row's raw eval-mode teacher anchor, inherited verbatim — it must NEVER be taken from `root_black_value`/`root_value_stm` (BASE's MCTS root scalar). Any code path reading those columns as a target is a defect.
- Existing v6 behavior byte-identical when the new flag/columns are absent: default builder output unchanged; existing v6 manifests load exactly as today; existing smoke tests pass unmodified; `VALID_LOSS_MODES` / `TEACHER_MODE_LOSS_MODES` / `RETENTION_POLICY_LOSS_MODES` membership unchanged (NO new loss mode).
- Value-only rule: `red_predrop_root_value_retention` rows must get policy mask 0 (`has_policy_target False`) — they contribute only to the value term, never policy CE/KL.
- Builders never silently trim: exclusions logged, hard `ValueError` on caps/duplicates/missing teacher_value.
- Do NOT touch: trainer.py, mcts.py, continuation_extraction.py, any v2-v5 builder/smoke, docs/post-game-analysis.md, any manifest/checkpoint.
- One feature branch `feature/tvc-v6c-d-root-value-retention` (worktree; note: a fresh worktree lacks gitignored local game-log data → 14F+6E pre-existing in the whole-repo suite there; judge tasks on file-scoped runs, authoritative suite on merged main). Per-task commits, FF-merge to main (no `--no-ff`, never force-push).
- Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. File-scoped `git add`. Locate code by content, not line numbers.

## File Structure

| File | Role |
|---|---|
| `scripts/GPU/alphazero/calibration_pool.py` (modify, `_parse_extra_moves` only) | accept explicit-empty extra moves for root_value rows |
| `scripts/GPU/alphazero/build_searched_continuation_retention_manifest.py` (modify) | `D_ROOT_VALUE_TAG`, `_root_value_row`, `--emit-d-root-value-rows`, classify_row rerun tag |
| `scripts/GPU/alphazero/smoke_searched_continuation_retention_v6.py` (modify) | `V6C_TAG_SCHEDULE`, schedule param + `--schedule` flag |
| `tests/test_calibration_pool_continuation.py` (append) | Task 1 tests |
| `tests/test_build_searched_continuation_retention_manifest.py` (append) | Task 2 tests |
| `tests/test_smoke_searched_continuation_retention_v6.py` (append) | Task 3 tests |

---

### Task 1: Loader — depth-0 root-value rows (`continuation_source == "root_value"`)

**Files:**
- Modify: `scripts/GPU/alphazero/calibration_pool.py` (ONLY `_parse_extra_moves`)
- Modify: `tests/test_calibration_pool_continuation.py` (append tests)

**Interfaces:**
- Consumes: existing `_parse_extra_moves` / `_apply_extra_moves` / `build_calibration_sample` from v6 Task 1.
- Produces: a row with `loss_mode == "searched_continuation_retention"`, `continuation_source == "root_value"`, `extra_moves_json == "[]"` loads with the continuation state == ROOT state (`ply == position_ply`, side/sha1 verified against the root), `outcome == teacher_value`, `visit_counts == [0]*len(legal)`, `has_policy_target is False`, mask 0. A root_value row with NON-empty moves fails loud; empty moves WITHOUT the root_value marker still fail exactly as today.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_calibration_pool_continuation.py`:

```python
def _root_value_case(tmp_path, **overrides):
    """Depth-0 D root-value clone: continuation state IS the root state."""
    rp = tmp_path / "game_000002.json"
    replay = legal_replay(9, game_idx=2)
    rp.write_text(json.dumps(replay))
    state = _root_state(replay)
    case = {
        "game_idx": "2",
        "case_id": "game_000002_ply_005__root_value",
        "replay_path": str(rp), "position_ply": "5", "side_to_move": "black",
        "tag": "red_predrop_root_value_retention",
        "loss_mode": CONTINUATION_LOSS_MODE,
        "teacher_value": "-0.4173", "weight_scale": "1.0",
        "extra_moves_json": "[]",
        "continuation_source": "root_value",
        "continuation_depth": "0",
        "continuation_side_to_move": state.to_move,
        "continuation_legal_moves_sha1": legal_moves_sha1(state.legal_moves()),
    }
    case.update(overrides)
    return case, state


def test_root_value_row_loads_at_root_state(tmp_path):
    case, root = _root_value_case(tmp_path)
    sample = build_calibration_sample(case, calibration_target=-0.35)
    assert sample.loss_mode == CONTINUATION_LOSS_MODE
    assert sample.tag == "red_predrop_root_value_retention"
    assert sample.has_policy_target is False
    rec = sample.record
    assert rec.outcome == pytest.approx(-0.4173)         # teacher_value, stm, direct
    assert rec.to_move == root.to_move                   # root side, no moves applied
    assert rec.legal_moves == root.legal_moves()         # root legal set
    assert rec.visit_counts == [0] * len(rec.legal_moves)
    assert rec.ply == 5                                  # position_ply + 0
    _, _, mask = split_samples_with_modes([sample], has_weight_scale=False)
    assert mask.tolist() == [0.0]                        # value-only: never policy


def test_root_value_row_blank_extra_moves_also_accepted(tmp_path):
    case, root = _root_value_case(tmp_path, extra_moves_json="")
    sample = build_calibration_sample(case, calibration_target=-0.35)
    assert sample.record.legal_moves == root.legal_moves()


def test_root_value_row_rejects_nonempty_extra_moves(tmp_path):
    case, root = _root_value_case(tmp_path)
    m = root.legal_moves()[0]
    case["extra_moves_json"] = json.dumps([{"row": m[0], "col": m[1]}])
    with pytest.raises(ValueError, match="root_value"):
        build_calibration_sample(case, calibration_target=-0.35)


def test_root_value_row_still_verifies_sha1(tmp_path):
    case, _ = _root_value_case(tmp_path, continuation_legal_moves_sha1="deadbeef")
    with pytest.raises(ValueError, match="sha1"):
        build_calibration_sample(case, calibration_target=-0.35)


def test_empty_extra_moves_without_root_value_marker_still_fails(tmp_path):
    # non-root_value continuation rows keep today's fail-loud behavior
    case, _ = _case(tmp_path, extra_moves_json="[]")
    with pytest.raises(ValueError, match="extra_moves_json"):
        build_calibration_sample(case, calibration_target=-0.35)
    case2, _ = _case(tmp_path, extra_moves_json="")
    with pytest.raises(ValueError, match="extra_moves_json"):
        build_calibration_sample(case2, calibration_target=-0.35)


def test_root_value_row_rejects_root_visits_json(tmp_path):
    case, _ = _root_value_case(tmp_path, root_visits_json=json.dumps([1.0]))
    with pytest.raises(ValueError, match="root_visits_json"):
        build_calibration_sample(case, calibration_target=-0.35)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool_continuation.py -v`
Expected: the 4 new happy-path/root-state tests FAIL with `ValueError: ... continuation row needs extra_moves_json` / `must be a non-empty list`; `test_root_value_row_rejects_nonempty_extra_moves` FAILS (no "root_value" in message); the two negative regression tests (`..._without_root_value_marker_...`, `..._rejects_root_visits_json`) PASS already. All 16 pre-existing tests in the file PASS.

- [ ] **Step 3: Implement — replace `_parse_extra_moves` in `calibration_pool.py`**

Replace the whole function body (locate `def _parse_extra_moves` by content):

```python
def _parse_extra_moves(case: dict) -> list[tuple[int, int]]:
    """Required non-empty extra_moves_json for continuation rows: JSON list of
    {"row": int, "col": int} applied after the position_ply reconstruction.

    Depth-0 exception (v6c): rows with continuation_source == "root_value"
    anchor the ROOT state itself — they carry an explicit empty list (or
    blank) and MUST NOT list any moves. Side/sha1 verification still runs
    against the root state in _apply_extra_moves."""
    cid = case.get("case_id")
    raw = case.get("extra_moves_json")
    is_root_value = case.get("continuation_source") == "root_value"
    if raw in (None, ""):
        if is_root_value:
            return []
        raise ValueError(f"{cid}: continuation row needs extra_moves_json")
    try:
        moves = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{cid}: extra_moves_json invalid JSON: {e}") from e
    if not isinstance(moves, list):
        raise ValueError(f"{cid}: extra_moves_json must be a non-empty list")
    if not moves:
        if is_root_value:
            return []
        raise ValueError(f"{cid}: extra_moves_json must be a non-empty list")
    if is_root_value:
        raise ValueError(
            f"{cid}: root_value row must have empty extra_moves_json; got {raw!r}")
    out = []
    for m in moves:
        if not isinstance(m, dict) or "row" not in m or "col" not in m:
            raise ValueError(f"{cid}: extra_moves_json entries need row/col: {m!r}")
        out.append((int(m["row"]), int(m["col"])))
    return out
```

No other loader change: `_apply_extra_moves` already handles an empty move list (the loop no-ops, then side + sha1 are verified against the unmodified root state), `record_ply = position_ply + 0`, and the value-only mask comes free from `has_policy_target` (no `teacher_policy_json` on these rows).

- [ ] **Step 4: Run the file plus the v6/v5 loader regression surface**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool_continuation.py tests/test_calibration_pool.py tests/test_smoke_mcts_root_retention_v5.py tests/test_smoke_searched_continuation_retention_v6.py -v`
Expected: ALL PASS (pre-existing files unmodified except the appends).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/calibration_pool.py tests/test_calibration_pool_continuation.py
git commit -m "feat(calibration): accept depth-0 root_value continuation rows (empty extra_moves_json gated on continuation_source)"
```

---

### Task 2: Builder — `--emit-d-root-value-rows`

**Files:**
- Modify: `scripts/GPU/alphazero/build_searched_continuation_retention_manifest.py`
- Modify: `tests/test_build_searched_continuation_retention_manifest.py` (append tests)

**Interfaces:**
- Consumes: Task 1's loader acceptance; existing `NEW_COLUMNS_V6`, `classify_row`, `build_rows_v6`, `_continuation_row`, `CONTINUATION_LOSS_MODE`, `legal_moves_sha1`.
- Produces: `D_ROOT_VALUE_TAG = "red_predrop_root_value_retention"`; `_root_value_row(parent, root_state) -> dict`; `build_rows_v6(..., emit_d_root_value=False)` keyword WITH default `False` (so every pre-existing call site — including the direct call in `test_source_root_value_mismatch_fails` — passes unmodified; disabled-by-default = byte-identical v6 output); CLI flag `--emit-d-root-value-rows` (store_true, default OFF); `classify_row` passes through `(CONTINUATION_LOSS_MODE, D_ROOT_VALUE_TAG)` rows for rerun safety; `stats["n_root_value"]` count.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_build_searched_continuation_retention_manifest.py`:

```python
def _rows_with_d(tmp_path):
    """The standard fixture rows plus one D-family root row on the same replay."""
    rows = _rows(tmp_path)
    state = position_state(json.loads(open(rows[0]["replay_path"]).read()), 5, "black")
    legal = state.legal_moves()
    dense = [0.0] * len(legal); dense[0] = 300 / 399; dense[1] = 99 / 399
    d_row = dict(rows[1])
    d_row["case_id"] = "red_loss_game_000001_predrop_ply_5_drop_7"
    d_row["tag"] = "red_predrop_retention"
    d_row["teacher_value"] = "-0.4173"
    d_row["root_visits_json"] = json.dumps(dense)
    return rows + [d_row], state


def _build_d(tmp_path, emit_d_root_value, **kw):
    rows, _ = _rows_with_d(tmp_path)
    params = dict(pos_base_seed=20260616, goal_base_seed=20260614,
                  b_pv_depth=2, c_pv_depth=3, d_top_k=3, d_child_pv_depth=1,
                  d_child_pv_min_visits=40, max_per_root=6, max_total=250,
                  emit_policy=False, source_root_tolerance=1e-3,
                  limit_cases=None, only_case_ids=None,
                  emit_d_root_value=emit_d_root_value)
    params.update(kw)
    return build_rows_v6(rows, _FakeRawEval(), _fake_search, **params)


def test_flag_off_emits_no_root_value_rows(tmp_path):
    out, stats = _build_d(tmp_path, emit_d_root_value=False)
    assert stats["n_root_value"] == 0
    assert not [r for r in out if r["tag"] == "red_predrop_root_value_retention"]


def test_root_value_clone_fields(tmp_path):
    from scripts.GPU.alphazero.build_searched_continuation_retention_manifest import (
        D_ROOT_VALUE_TAG)
    out, stats = _build_d(tmp_path, emit_d_root_value=True)
    assert stats["n_root_value"] == 1
    parent_idx = next(i for i, r in enumerate(out)
                      if r["case_id"] == "red_loss_game_000001_predrop_ply_5_drop_7")
    clone = out[parent_idx + 1]                      # appended right after parent
    assert clone["case_id"] == out[parent_idx]["case_id"] + "__root_value"
    assert clone["tag"] == D_ROOT_VALUE_TAG
    assert clone["loss_mode"] == CONTINUATION_LOSS_MODE
    assert clone["teacher_value"] == "-0.4173"       # COPIED raw anchor, verbatim
    assert clone["teacher_value_source"] == "base_raw_root_clone"
    assert clone["extra_moves_json"] == "[]"
    assert clone["continuation_source"] == "root_value"
    assert clone["continuation_depth"] == "0"
    assert clone["continuation_parent_case_id"] == out[parent_idx]["case_id"]
    assert clone["continuation_side_to_move"] == "black"
    assert clone["root_visits_json"] == ""           # NO policy signal of any kind
    assert clone["teacher_policy_json"] == ""
    assert clone["target_black_value"] == ""
    # C parent (non-D) must NOT get a clone
    assert not [r for r in out
                if r["case_id"] == "game_000001_ply_005__root_value"]


def test_root_value_clone_loads_value_only(tmp_path):
    from scripts.GPU.alphazero.calibration_pool import split_samples_with_modes
    out, _ = _build_d(tmp_path, emit_d_root_value=True)
    clone = next(r for r in out if r["tag"] == "red_predrop_root_value_retention")
    sample = build_calibration_sample(clone, calibration_target=-0.35)
    assert sample.has_policy_target is False
    assert sample.record.outcome == pytest.approx(-0.4173)
    assert sample.record.ply == 5
    _, _, mask = split_samples_with_modes([sample], has_weight_scale=False)
    assert mask.tolist() == [0.0]


def test_root_value_clone_deterministic_and_unique(tmp_path):
    out1, _ = _build_d(tmp_path, emit_d_root_value=True)
    out2, _ = _build_d(tmp_path, emit_d_root_value=True)
    assert out1 == out2
    ids = [r["case_id"] for r in out1]
    assert len(ids) == len(set(ids))


def test_classify_row_accepts_root_value_rerun():
    from scripts.GPU.alphazero.build_searched_continuation_retention_manifest import (
        D_ROOT_VALUE_TAG)
    assert classify_row({"loss_mode": CONTINUATION_LOSS_MODE,
                         "tag": D_ROOT_VALUE_TAG}) == "passthrough"


def test_missing_teacher_value_on_d_parent_fails(tmp_path):
    rows, _ = _rows_with_d(tmp_path)
    rows[2]["teacher_value"] = ""
    with pytest.raises(ValueError, match="teacher_value"):
        build_rows_v6(rows, _FakeRawEval(), _fake_search,
                      pos_base_seed=20260616, goal_base_seed=20260614,
                      b_pv_depth=2, c_pv_depth=3, d_top_k=3, d_child_pv_depth=1,
                      d_child_pv_min_visits=40, max_per_root=6, max_total=250,
                      emit_policy=False, source_root_tolerance=1e-3,
                      limit_cases=None, only_case_ids=None,
                      emit_d_root_value=True)
```

Do NOT modify any pre-existing test in the file — the signature default keeps them green.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_build_searched_continuation_retention_manifest.py -v`
Expected: new tests FAIL (`ImportError: cannot import name 'D_ROOT_VALUE_TAG'` for the tests importing it; `TypeError: build_rows_v6() got an unexpected keyword argument 'emit_d_root_value'` for the `_build_d`-based ones); ALL 9 pre-existing tests PASS.

- [ ] **Step 3: Implement in `build_searched_continuation_retention_manifest.py`**

After `CORRECTION_TAG = ...` add:

```python
D_ROOT_VALUE_TAG = "red_predrop_root_value_retention"
D_ROOT_VALUE_SOURCE_TAG = "red_predrop_retention"
```

In `classify_row`, add before the final `raise`:

```python
    if mode == CONTINUATION_LOSS_MODE and tag == D_ROOT_VALUE_TAG:
        return "passthrough"                        # rerun on a v6c output
```

After `_continuation_row` add:

```python
def _root_value_row(parent: dict, root_state) -> dict:
    """Depth-0 value-only clone of a D root row (v6c): anchors the raw
    eval-mode teacher_value at the ROOT state with no policy signal.
    teacher_value is INHERITED from the source row (raw eval anchor) —
    NEVER root_black_value/root_value_stm (the MCTS root scalar; experiment
    ledger do-not-repeat #9)."""
    if (parent.get("teacher_value") or "") == "":
        raise ValueError(
            f"{parent.get('case_id')}: D root row lacks teacher_value; cannot "
            f"emit a root-value clone")
    row = dict(parent)                              # inherit ALL parent columns
    for c in NEW_COLUMNS_V6:
        row.setdefault(c, "")
    legal = root_state.legal_moves()
    row["case_id"] = f"{parent['case_id']}__root_value"
    row["tag"] = D_ROOT_VALUE_TAG
    row["loss_mode"] = CONTINUATION_LOSS_MODE
    row["teacher_value_source"] = "base_raw_root_clone"
    row["extra_moves_json"] = "[]"
    row["continuation_side_to_move"] = root_state.to_move
    row["continuation_legal_moves_sha1"] = legal_moves_sha1(legal)
    row["continuation_depth"] = "0"
    row["continuation_parent_case_id"] = parent["case_id"]
    row["continuation_source"] = "root_value"
    row["continuation_path_moves"] = ""
    row["continuation_tree_visits"] = ""
    row["continuation_tree_nn_value"] = ""
    row["target_black_value"] = ""                  # never a hard target
    row["root_visits_json"] = ""                    # NO policy signal
    row["teacher_policy_json"] = ""
    row["teacher_legal_moves_sha1"] = ""
    return row
```

In `build_rows_v6`: add `emit_d_root_value=False` to the keyword-only parameter list (after `only_case_ids`; the ONLY defaulted parameter — deliberate, it is the disabled-by-default v6c switch), add `"n_root_value": 0` to the `stats` dict literal, and insert the clone emission immediately after `state = position_state(replay, ply, side)`:

```python
        state = position_state(replay, ply, side)
        if emit_d_root_value and r["tag"] == D_ROOT_VALUE_SOURCE_TAG:
            rv_row = _root_value_row(r, state)
            out.append(rv_row)
            stats["n_root_value"] += 1
            stats["by_tag"][D_ROOT_VALUE_TAG] = (
                stats["by_tag"].get(D_ROOT_VALUE_TAG, 0) + 1)
```

(Clones are bounded at one per D source row — they are counted in `stats["n_root_value"]`/`by_tag`, NOT against `max_total`, which remains the continuation-extraction cap. The existing duplicate-case_id guard covers clone collisions.)

In `main()`: add the flag and pass it through:

```python
    ap.add_argument("--emit-d-root-value-rows", action="store_true",
                    help="v6c: for each red_predrop_retention source row, also "
                         "emit a depth-0 value-only red_predrop_root_value_retention "
                         "clone (default OFF = byte-identical v6 output)")
```

and in the `build_rows_v6(...)` call: `emit_d_root_value=args.emit_d_root_value_rows,`
and extend the final summary print to include `stats['n_root_value']`:

```python
    print(f"wrote {len(out_rows)} rows ({stats['n_continuation']} continuation, "
          f"{stats['n_root_value']} root-value: {stats['by_tag']}) -> {args.out}")
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_build_searched_continuation_retention_manifest.py tests/test_calibration_pool_continuation.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/build_searched_continuation_retention_manifest.py tests/test_build_searched_continuation_retention_manifest.py
git commit -m "feat(calibration): v6c --emit-d-root-value-rows — depth-0 value-only red_predrop_root_value_retention clones"
```

---

### Task 3: Smoke — `V6C_TAG_SCHEDULE` + `--schedule` flag

**Files:**
- Modify: `scripts/GPU/alphazero/smoke_searched_continuation_retention_v6.py`
- Modify: `tests/test_smoke_searched_continuation_retention_v6.py` (append tests)

**Interfaces:**
- Consumes: Task 1's loader acceptance (the test fixture gains a root_value row); existing `assert_continuation_retention_mechanics`, `V6_TAG_SCHEDULE`.
- Produces: `V6C_TAG_SCHEDULE` (exact dict below), `SCHEDULES = {"v6": V6_TAG_SCHEDULE, "v6c": V6C_TAG_SCHEDULE}`, `assert_continuation_retention_mechanics(network, manifest_path, value_tol=1e-4, schedule=None)` (None → `V6_TAG_SCHEDULE`, back-compat), CLI `--schedule {v6,v6c}` default `v6c`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_smoke_searched_continuation_retention_v6.py`:

```python
def _manifest_v6c(tmp_path, cont_teacher_value, root_teacher_value):
    """The v6 fixture manifest plus one depth-0 D root-value row. Continuation
    rows anchor the CONTINUATION-state value; the root-value row anchors the
    ROOT-state value — pass each state's actual network value so both anchors
    reproduce simultaneously."""
    base = _manifest(tmp_path, cont_teacher_value)
    replay = legal_replay(9, game_idx=1)
    state = position_state(replay, 5, "black")
    root_value_row = {
        "game_idx": "1", "replay_path": str(tmp_path / "game_000001.json"),
        "position_ply": "5", "side_to_move": "black", "weight_scale": "1.0",
        "case_id": "rv1", "tag": "red_predrop_root_value_retention",
        "loss_mode": CONTINUATION_LOSS_MODE,
        "teacher_value": repr(root_teacher_value),
        "extra_moves_json": "[]",
        "continuation_source": "root_value",
        "continuation_depth": "0",
        "continuation_side_to_move": state.to_move,
        "continuation_legal_moves_sha1": legal_moves_sha1(state.legal_moves()),
    }
    rows = list(csv.DictReader(open(base)))
    fields = sorted(set(rows[0].keys()) | set(root_value_row.keys()))
    p = tmp_path / "v6c_manifest.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
        w.writerow({k: root_value_row.get(k, "") for k in fields})
    return p


def _root_network_value(net):
    """Network's eval-mode stm value at the ROOT state (ply 5, black)."""
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.build_teacher_calibration_manifest import _teacher_infer
    replay = legal_replay(9, game_idx=1)
    state = position_state(replay, 5, "black")
    prev = net.training
    net.eval()
    try:
        _, _, v = _teacher_infer(state, LocalGPUEvaluator(net))
    finally:
        net.train(prev)
    return float(v)


def test_v6c_schedule_passes_with_root_value_rows(tmp_path):
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.smoke_searched_continuation_retention_v6 import (
        V6C_TAG_SCHEDULE)
    net = create_network(hidden=64, n_blocks=2)
    p = _manifest_v6c(tmp_path, cont_teacher_value=_network_value(net),
                      root_teacher_value=_root_network_value(net))
    report = assert_continuation_retention_mechanics(
        net, str(p), schedule=V6C_TAG_SCHEDULE)
    assert report["n_continuation"] == 4          # 3 continuation + 1 root_value
    assert report["policy_ce"] == 0.0
    assert report["n_policy_rows"] == 0
    assert abs(report["value_mse"]) < 1e-4
    assert report["draws_by_tag"] == V6C_TAG_SCHEDULE


def test_default_schedule_unchanged_for_v6_manifests(tmp_path):
    from scripts.GPU.alphazero.network import create_network
    net = create_network(hidden=64, n_blocks=2)
    v = _network_value(net)
    p = _manifest(tmp_path, teacher_value=v)
    report = assert_continuation_retention_mechanics(net, str(p))
    assert report["draws_by_tag"] == V6_TAG_SCHEDULE


def test_v6c_schedule_on_v6_manifest_fails_loud(tmp_path):
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.smoke_searched_continuation_retention_v6 import (
        V6C_TAG_SCHEDULE)
    net = create_network(hidden=64, n_blocks=2)
    v = _network_value(net)
    p = _manifest(tmp_path, teacher_value=v)      # no root_value rows
    with pytest.raises(ValueError, match="missing tags"):
        assert_continuation_retention_mechanics(
            net, str(p), schedule=V6C_TAG_SCHEDULE)
```

Also add `V6_TAG_SCHEDULE` to the existing import from the smoke module at the top of the test file (it currently imports `V6_TAG_SCHEDULE, assert_continuation_retention_mechanics` — if `V6_TAG_SCHEDULE` is already imported, no change needed).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_smoke_searched_continuation_retention_v6.py -v`
Expected: new tests FAIL (`ImportError: cannot import name 'V6C_TAG_SCHEDULE'` / `TypeError: ... unexpected keyword argument 'schedule'`); the 3 pre-existing tests PASS.

- [ ] **Step 3: Implement in `smoke_searched_continuation_retention_v6.py`**

After the `V6_TAG_SCHEDULE` dict add:

```python
V6C_TAG_SCHEDULE = {
    "black_predrop_correction": 2,
    "goal_line_continuation_retention": 1,
    "old_post_opening_continuation_retention": 2,
    "red_predrop_root_value_retention": 1,
    "red_predrop_continuation_retention": 2,
}
SCHEDULES = {"v6": V6_TAG_SCHEDULE, "v6c": V6C_TAG_SCHEDULE}
```

Change the function signature and the two schedule uses (locate by content):

```python
def assert_continuation_retention_mechanics(network, manifest_path: str,
                                            value_tol: float = 1e-4,
                                            schedule: dict | None = None) -> dict:
```

```python
    schedule = V6_TAG_SCHEDULE if schedule is None else schedule
    draws = pool.sample_by_tag(schedule, random.Random(0))
```

```python
    if draws_by_tag != schedule:
        raise AssertionError(
            f"schedule draw mismatch: {draws_by_tag} != {schedule}")
```

In `main()`:

```python
    ap.add_argument("--schedule", choices=sorted(SCHEDULES), default="v6c",
                    help="which locked tag schedule to hard-assert (default v6c)")
```

and pass it: `report = assert_continuation_retention_mechanics(network, args.manifest, value_tol=args.value_tol, schedule=SCHEDULES[args.schedule])`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_smoke_searched_continuation_retention_v6.py tests/test_calibration_pool_continuation.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/smoke_searched_continuation_retention_v6.py tests/test_smoke_searched_continuation_retention_v6.py
git commit -m "feat(calibration): v6c smoke schedule — V6C_TAG_SCHEDULE + --schedule flag (default v6c, v6 behavior preserved)"
```

---

### Task 4: Full-suite verification

**Files:** none new.

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected in the worktree: 1266 + ~14 new = ~1280 passed with EXACTLY the known 14 failed + 6 errors (missing gitignored local game-log data). Authoritative check (1294 + new, 0 failures) happens on merged main before push. Do NOT merge red.

- [ ] **Step 2: v6/v5 no-behavior-change spot check**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py tests/test_smoke_mcts_root_retention_v5.py tests/test_build_mcts_root_retention_manifest.py tests/test_calibration_loss.py -v`
Expected: ALL PASS with pre-existing assertions untouched.

- [ ] **Step 3: Hand off to merge**

FF-merge to main (no `--no-ff`), authoritative full suite on merged main, push. Handled by superpowers:finishing-a-development-branch.

---

## Controller-run after merge (per user scope choice): build + smoke

1. **Build v6c manifest** from the **v5 source manifest** (NOT the v6 output — rerunning extraction on a v6/v6c output duplicates continuation case_ids and hard-fails, by design):

```bash
.venv/bin/python -m scripts.GPU.alphazero.build_searched_continuation_retention_manifest \
  --source logs/eval/targeted_calibration_v5_mcts_root_from_calib020_0001.csv \
  --base-checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --out logs/eval/targeted_calibration_v6c_d_root_value_only_from_calib020_0001.csv \
  --emit-d-root-value-rows \
  --max-total-continuation-rows 306 \
  [same --gate-cases-csv/--gate-checkpoint-label args as the v6 build if locatable]
```

Expected shape: **411 rows** = 50 `black_predrop_correction` + 78 inert `mcts_root_retention` (30 D + 30 C + 18 B) + 253 continuation (127 D + 90 C + 36 B) + **30 `red_predrop_root_value_retention`**. Continuation rows byte-identical to the v6 build (same seeds/config); verify by tag-count comparison against `targeted_calibration_v6_continuation_from_calib020_0001.csv`.

2. **Smoke**:

```bash
.venv/bin/python -m scripts.GPU.alphazero.smoke_searched_continuation_retention_v6 \
  --manifest logs/eval/targeted_calibration_v6c_d_root_value_only_from_calib020_0001.csv \
  --base-checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --schedule v6c
```

Expected: PASS with `policy_ce=0.0000`, `draws_by_tag == V6C_TAG_SCHEDULE`.

## Operator run (USER's, after controller hands back)

3. **Train from BASE** — v6 command with three deltas: v6c manifest path, `--checkpoint-dir checkpoints/alphazero-v6c-d-root-value-only-from-calib020-0001`, schedule `black_predrop_correction=2,goal_line_continuation_retention=1,old_post_opening_continuation_retention=2,red_predrop_root_value_retention=1,red_predrop_continuation_retention=2`. Same weight 0.01. (`n_teacher_retention_drawn` stays 0 — expected on a value-only run; draw telemetry = `calib_n_drawn_by_tag`, now FOUR retention tags 1:2:1:2 plus corrections 2.)
4. **Gates A/B/C/D** vs `calib020_0001`, `OUT=logs/eval/v6c_d_root_value_only_from_calib020_0001_gates_400s`. Promotion ONLY if: A passes; B severe = 0.0 and over ≤ 11.1%; C severe ≤ 13.3%, over ≤ 33.3%, mean ≤ +0.099; D severe = 0.0 and mean ≤ 0.0.
5. **Ledger update** (v6c row + do-not-repeat if rejected).
