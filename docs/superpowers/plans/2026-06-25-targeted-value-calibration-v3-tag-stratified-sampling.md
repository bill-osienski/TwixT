# Targeted Value Calibration v3 — Tag-Stratified Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tag-stratified calibration sampling to the trainer so correction draws can be over-weighted relative to retention draws per training step, with per-tag draw-count telemetry — without touching the loss math, `train_step` tuple, CSV schema, or manifest builder.

**Architecture:** A new `CalibrationPool.sample_by_tag(schedule, rng)` draws an explicit per-tag count (with replacement) instead of uniform-with-replacement. The trainer selects it when a `--post-opening-calibration-tag-schedule` is provided; otherwise the existing uniform `batch_fraction` path is byte-identical to today. Per-tag draw counts accumulate into a dict surfaced in the `iter_<N>_stats.json` sidecar (key `draws_by_tag`) and the `model_iter_<N>.json` state (key `calib_n_drawn_by_tag`) — never in `metrics.csv`, which holds flat scalars only.

**Tech Stack:** Python 3.14.6 (`.venv/bin/python`), MLX, pytest. Tests live in repo-root `tests/`.

## Global Constraints

- Python 3.14.6 — run everything with `.venv/bin/python`.
- **Option A only.** Implement tag-stratified *draw counts* + `calib_n_drawn_by_tag` JSON telemetry. Do **not** add per-tag loss/value means (deferred "B" — needs widening the train_step tuple or N× forward passes; out of scope).
- **Do NOT modify:** `train_step` signature or its 10-tuple return; `alphazero_loss_batch` and its calibration MSE math; `split_samples` (stays tag-agnostic); `CSV_FIELDNAMES` / `metrics.csv` (flat scalars only — `DictWriter` uses `extrasaction='raise'`); `build_targeted_calibration_manifest.py`.
- `calib_n_drawn_by_tag` is a **dict** → it goes in `state` (model_iter JSON) and the sidecar (`draws_by_tag`), and is **NEVER** added to `iteration_metrics` or `CSV_FIELDNAMES`.
- **Extend** existing test files in repo-root `tests/`; never create parallel test files.
- The no-schedule path (schedule is `None`) MUST remain **sampling/loss/optimization-identical** to today's uniform `batch_fraction` path. JSON telemetry MAY gain the new by-tag draw-count fields (`calib_n_drawn_by_tag` in state, `draws_by_tag` in the sidecar) on **both** paths — they are additive and harmless (on the uniform path they simply report the natural tag distribution). They are still never added to `metrics.csv`.
- One commit per task, in the repo's `type(scope): subject` style.

---

## File Structure

| File | Change |
|------|--------|
| `scripts/GPU/alphazero/calibration_pool.py` | Add `self._by_tag` index in `CalibrationPool.__init__`; add `sample_by_tag()`; extend `build_post_opening_calibration_block()` to surface `draws_by_tag`. |
| `scripts/GPU/alphazero/train.py` | Add `--post-opening-calibration-tag-schedule` flag; add module-level `parse_calibration_tag_schedule()`; thread parsed dict into `train_kwargs` in `main()`. |
| `scripts/GPU/alphazero/trainer.py` | Add `post_opening_calibration_tag_schedule` param to `train()`; switch sampling on the schedule; accumulate a per-tag `Counter`; wire it into the sidecar + `state`; update the startup print. |
| `tests/test_calibration_pool.py` | `sample_by_tag` unit tests; `draws_by_tag` sidecar test. |
| `tests/test_calibration_cli_flags.py` | Flag default + `parse_calibration_tag_schedule` unit tests. |
| `tests/test_training.py` | Integration test: tag schedule → `calib_n_drawn_by_tag` in model_iter JSON, absent from CSV. |

Task order: **1 (pool sampling) → 2 (CLI) → 3 (sidecar block) → 4 (trainer wiring)**. Tasks 1–3 are independently testable and have no dependency on each other; Task 4 consumes all three.

---

### Task 1: `CalibrationPool` — `_by_tag` index + `sample_by_tag`

**Files:**
- Modify: `scripts/GPU/alphazero/calibration_pool.py:116-134` (`__init__` + after `sample`)
- Test: `tests/test_calibration_pool.py`

**Interfaces:**
- Consumes: `CalibrationSample` (has `.tag: str`), `CalibrationPool.__init__(samples, ...)`, existing `sample(k, rng)` (uniform, with replacement; uses `rng.choice`).
- Produces:
  - `CalibrationPool.validate_tag_schedule(schedule: dict[str, int]) -> None` — raises `ValueError` listing any positively-scheduled tag absent from the pool (zero-count tags ignored). Called once at setup to fail before self-play.
  - `CalibrationPool.sample_by_tag(schedule: dict[str, int], rng) -> list[CalibrationSample]` — calls `validate_tag_schedule` first, then draws `schedule[tag]` samples per tag (with replacement), preserving schedule (dict-insertion) order and skipping tags whose count `<= 0`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_calibration_pool.py` (the file already imports `random`, `pytest`, `CalibrationPool`, `CalibrationSample`, `build_calibration_sample`, and defines `_write_case_side(tmp_path, side, position_ply, game_idx=1, **extra)`):

```python
def test_sample_by_tag_draws_requested_counts(tmp_path):
    s_corr_a = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=1, tag="correction"), -0.5)
    s_corr_b = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=2, tag="correction"), -0.5)
    s_ret = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=3, tag="retention"), -0.5)
    pool = CalibrationPool([s_corr_a, s_corr_b, s_ret])
    drawn = pool.sample_by_tag({"correction": 2, "retention": 1}, random.Random(0))
    tags = [s.tag for s in drawn]
    assert len(drawn) == 3
    assert tags.count("correction") == 2
    assert tags.count("retention") == 1


def test_sample_by_tag_samples_with_replacement(tmp_path):
    s = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=1, tag="correction"), -0.5)
    pool = CalibrationPool([s])  # single-member bucket
    drawn = pool.sample_by_tag({"correction": 4}, random.Random(0))
    assert len(drawn) == 4
    assert all(d.tag == "correction" for d in drawn)


def test_sample_by_tag_zero_count_skips(tmp_path):
    s_corr = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=1, tag="correction"), -0.5)
    s_ret = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=2, tag="retention"), -0.5)
    pool = CalibrationPool([s_corr, s_ret])
    drawn = pool.sample_by_tag({"correction": 2, "retention": 0}, random.Random(0))
    assert len(drawn) == 2
    assert all(d.tag == "correction" for d in drawn)


def test_sample_by_tag_unknown_tag_raises(tmp_path):
    s = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=1, tag="correction"), -0.5)
    pool = CalibrationPool([s])
    with pytest.raises(ValueError):
        pool.sample_by_tag({"nonexistent": 1}, random.Random(0))


def test_validate_tag_schedule_passes_for_known_tags(tmp_path):
    s = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=1, tag="correction"), -0.5)
    pool = CalibrationPool([s])
    pool.validate_tag_schedule({"correction": 2})  # no raise


def test_validate_tag_schedule_raises_for_missing_tag(tmp_path):
    s = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=1, tag="correction"), -0.5)
    pool = CalibrationPool([s])
    with pytest.raises(ValueError):
        pool.validate_tag_schedule({"correction": 1, "typo_tag": 1})


def test_validate_tag_schedule_ignores_zero_count_missing_tag(tmp_path):
    s = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=1, tag="correction"), -0.5)
    pool = CalibrationPool([s])
    pool.validate_tag_schedule({"correction": 1, "absent": 0})  # 0-count tag skipped
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py -k "sample_by_tag or validate_tag_schedule" -v`
Expected: FAIL — `AttributeError: 'CalibrationPool' object has no attribute 'sample_by_tag'` (and `validate_tag_schedule`).

- [ ] **Step 3: Build the `_by_tag` index in `__init__`**

In `scripts/GPU/alphazero/calibration_pool.py`, in `CalibrationPool.__init__`, after the existing `self.schema = schema` line (currently line 126), add:

```python
        self.schema = schema
        self._by_tag: dict[str, list] = {}
        for s in self._samples:
            self._by_tag.setdefault(s.tag, []).append(s)
```

(Building it here covers `from_manifest` too, which calls `cls(samples, ...)` → `__init__`.)

- [ ] **Step 4: Add `validate_tag_schedule` + `sample_by_tag`**

Immediately after the existing `sample(self, k, rng)` method (currently ends line 134), add both methods. `sample_by_tag` validates first (DRY — the same check the trainer calls at setup), so an absent tag fails before any draw:

```python
    def validate_tag_schedule(self, schedule: dict) -> None:
        """Raise ValueError if any positively-scheduled tag is absent from the pool.

        Call once at setup (before self-play) so a typo'd tag fails fast rather
        than after a wasted self-play iteration. Zero-count tags are ignored.
        """
        missing = sorted(tag for tag, n in schedule.items()
                         if n > 0 and tag not in self._by_tag)
        if missing:
            raise ValueError(
                f"calibration tag schedule requested missing tags {missing}; "
                f"pool has tags {sorted(self._by_tag)}")

    def sample_by_tag(self, schedule: dict, rng):
        """Draw per-tag counts (with replacement) per an explicit tag->count schedule.

        Validates the schedule first (raises on an absent positively-scheduled
        tag), then preserves schedule (dict-insertion) order and skips any tag
        whose count is <= 0.
        """
        self.validate_tag_schedule(schedule)
        out = []
        for tag, n in schedule.items():
            if n <= 0:
                continue
            out.extend(rng.choice(self._by_tag[tag]) for _ in range(n))
        return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py -k "sample_by_tag or validate_tag_schedule" -v`
Expected: PASS (7 tests).

- [ ] **Step 6: Run the full pool suite (no regressions)**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py -v`
Expected: PASS (all existing + 7 new).

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/calibration_pool.py tests/test_calibration_pool.py
git commit -m "feat(calibration): tag-stratified sample_by_tag + validate_tag_schedule + by-tag index"
```

---

### Task 2: CLI flag + `parse_calibration_tag_schedule` + `main()` threading

**Files:**
- Modify: `scripts/GPU/alphazero/train.py:387` (new flag), new module-level function before `def main()` (line 505), and `train.py:799` (thread into `train_kwargs`)
- Test: `tests/test_calibration_cli_flags.py`

**Interfaces:**
- Consumes: `build_arg_parser()` (returns the configured `argparse.ArgumentParser`); the existing `train_kwargs.update(dict(...))` block in `main()` that forwards `post_opening_calibration_*` args to `train()`.
- Produces: `parse_calibration_tag_schedule(raw: str | None) -> dict[str, int] | None` — parses `"tag=count,tag=count"` into an ordered dict; `None`/`""` → `None`. New CLI arg `--post-opening-calibration-tag-schedule` (dest `post_opening_calibration_tag_schedule`, default `None`).

- [ ] **Step 1: Write the failing tests**

Replace the existing import line at the top of `tests/test_calibration_cli_flags.py`:

```python
from scripts.GPU.alphazero.train import build_arg_parser
```

with:

```python
import pytest

from scripts.GPU.alphazero.train import (
    build_arg_parser, parse_calibration_tag_schedule,
)
```

Add `assert args.post_opening_calibration_tag_schedule is None` to the existing `test_calibration_flag_defaults` (after its last assert), then append:

```python
def test_calibration_tag_schedule_flag_parsed_raw():
    args = build_arg_parser().parse_args([
        "--post-opening-calibration-tag-schedule",
        "black_predrop_correction=2,goal_line_retention=1",
    ])
    assert (args.post_opening_calibration_tag_schedule
            == "black_predrop_correction=2,goal_line_retention=1")


def test_parse_calibration_tag_schedule_none():
    assert parse_calibration_tag_schedule(None) is None
    assert parse_calibration_tag_schedule("") is None


def test_parse_calibration_tag_schedule_valid_ordered():
    out = parse_calibration_tag_schedule(
        "black_predrop_correction=2,goal_line_retention=1,"
        "old_post_opening_retention=2,red_predrop_retention=1")
    assert out == {"black_predrop_correction": 2, "goal_line_retention": 1,
                   "old_post_opening_retention": 2, "red_predrop_retention": 1}
    assert list(out) == ["black_predrop_correction", "goal_line_retention",
                         "old_post_opening_retention", "red_predrop_retention"]


def test_parse_calibration_tag_schedule_missing_equals_raises():
    with pytest.raises(ValueError):
        parse_calibration_tag_schedule("black_predrop_correction")


def test_parse_calibration_tag_schedule_negative_raises():
    with pytest.raises(ValueError):
        parse_calibration_tag_schedule("black_predrop_correction=-1")


def test_parse_calibration_tag_schedule_duplicate_raises():
    with pytest.raises(ValueError):
        parse_calibration_tag_schedule(
            "black_predrop_correction=2,black_predrop_correction=1")


def test_parse_calibration_tag_schedule_zero_total_raises():
    with pytest.raises(ValueError):
        parse_calibration_tag_schedule("black_predrop_correction=0")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_cli_flags.py -v`
Expected: FAIL — `ImportError: cannot import name 'parse_calibration_tag_schedule'`.

- [ ] **Step 3: Add the CLI flag**

In `scripts/GPU/alphazero/train.py`, in `build_arg_parser()`, immediately after the `--post-opening-calibration-batch-fraction` argument (currently ends line 387), add:

```python
    parser.add_argument("--post-opening-calibration-tag-schedule", type=str, default=None,
        help="Tag-stratified calibration sampling schedule, e.g. "
             "'black_predrop_correction=2,goal_line_retention=1,"
             "old_post_opening_retention=2,red_predrop_retention=1'. When set, "
             "replaces uniform batch-fraction sampling (batch-fraction is ignored).")
```

- [ ] **Step 4: Add the parse helper**

In `scripts/GPU/alphazero/train.py`, add this module-level function immediately before `def main():` (currently line 505):

```python
def parse_calibration_tag_schedule(raw):
    """Parse 'tag=count,tag=count' into an ordered dict[str, int], or None.

    None/'' -> None (uniform batch-fraction sampling). Each count must be a
    non-negative int. Rejects entries missing '=', empty tags, duplicate tags,
    and an all-zero total.
    """
    if raw in (None, ""):
        return None
    out: dict = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"invalid calibration tag schedule entry {part!r}")
        tag, value = part.split("=", 1)
        tag = tag.strip()
        if not tag:
            raise ValueError("calibration tag schedule contains an empty tag")
        n = int(value)
        if n < 0:
            raise ValueError(
                f"calibration tag schedule count must be >= 0 for {tag!r}")
        if tag in out:
            raise ValueError(f"duplicate calibration tag schedule entry {tag!r}")
        out[tag] = n
    if not out or sum(out.values()) <= 0:
        raise ValueError("calibration tag schedule must draw at least one sample")
    return out
```

- [ ] **Step 5: Thread the parsed dict into `train()`**

In `main()`, find the `post_opening_calibration_batch_fraction=...` line in the `train_kwargs.update(dict(...))` block (currently line 799). Change:

```python
        post_opening_calibration_batch_fraction=args.post_opening_calibration_batch_fraction,
    ))
```

to:

```python
        post_opening_calibration_batch_fraction=args.post_opening_calibration_batch_fraction,
        post_opening_calibration_tag_schedule=parse_calibration_tag_schedule(
            args.post_opening_calibration_tag_schedule),
    ))
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_cli_flags.py -v`
Expected: PASS (defaults + flag + 6 parse-helper tests).

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/train.py tests/test_calibration_cli_flags.py
git commit -m "feat(calibration): --post-opening-calibration-tag-schedule CLI flag + parser"
```

---

### Task 3: `build_post_opening_calibration_block` surfaces `draws_by_tag`

**Files:**
- Modify: `scripts/GPU/alphazero/calibration_pool.py:163-184` (`build_post_opening_calibration_block`)
- Test: `tests/test_calibration_pool.py`

**Interfaces:**
- Consumes: `loss_accumulator: dict` (already carries `sum_calib_loss`, `sum_calib_n_drawn`, `sum_calib_value_pred`, `steps_done`).
- Produces: the returned block gains a top-level key `"draws_by_tag": dict` read from `loss_accumulator["sum_calib_n_drawn_by_tag"]` (defaults to `{}` when absent — keeps v1/v2 callers unchanged).

**Block `version` stays `1`.** `draws_by_tag` is purely additive, and `tests/test_calibration_pool.py::test_build_post_opening_calibration_block` pins `block["version"] == 1`; bumping to `2` would break that assertion for no functional gain. (Reviewer's optional bump declined on this basis — left as-is per the stated "if tests expect version 1, leave it" condition.)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_calibration_pool.py`:

```python
def test_sidecar_block_surfaces_draws_by_tag():
    from scripts.GPU.alphazero.calibration_pool import (
        build_post_opening_calibration_block,
    )
    block = build_post_opening_calibration_block(
        config={"enabled": True},
        enabled=True,
        loss_accumulator={"sum_calib_loss": 4.0, "sum_calib_n_drawn": 60,
                          "sum_calib_value_pred": 3.0, "steps_done": 10,
                          "sum_calib_n_drawn_by_tag": {"correction": 40,
                                                       "retention": 20}})
    assert block["draws_by_tag"] == {"correction": 40, "retention": 20}
```

Also add one line to the existing `test_build_post_opening_calibration_block` (which passes no by-tag key) to pin the default, after its last assert:

```python
    assert block["draws_by_tag"] == {}  # absent in accumulator -> empty dict
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py -k "draws_by_tag or build_post_opening" -v`
Expected: FAIL — `KeyError: 'draws_by_tag'`.

- [ ] **Step 3: Add `draws_by_tag` to the returned block**

In `scripts/GPU/alphazero/calibration_pool.py`, in `build_post_opening_calibration_block`, change the returned dict (currently ends with the `"loss": {...}` block at line 183) to add a sibling key:

```python
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
        "draws_by_tag": dict(loss_accumulator.get("sum_calib_n_drawn_by_tag", {})),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_pool.py -k "draws_by_tag or build_post_opening" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/calibration_pool.py tests/test_calibration_pool.py
git commit -m "feat(calibration): surface draws_by_tag in post-opening calibration sidecar block"
```

---

### Task 4: Trainer wiring — schedule param, sampling switch, per-tag counter, telemetry

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` — `train()` signature (~2361), hoist init (~2786), per-iteration reset (~3760), sampling block (~3772-3777), startup print (~2772-2781), sidecar `loss_accumulator` (~3946-3951), `state` dict (~4336)
- Test: `tests/test_training.py`

**Interfaces:**
- Consumes: `CalibrationPool.validate_tag_schedule(schedule)` + `sample_by_tag(schedule, rng)` (Task 1); `build_post_opening_calibration_block` reading `sum_calib_n_drawn_by_tag` (Task 3); existing `_calib_pool`, `_calib_active`, `train_rng` (a `random.Random`), `sum_calib_*` accumulators, `split_samples`.
- Produces: `train()` gains kwarg `post_opening_calibration_tag_schedule: Optional[dict] = None`. When set, per-step calibration sampling uses `sample_by_tag`. `model_iter_<N>.json` `state` gains `"calib_n_drawn_by_tag": dict`; `iter_<N>_stats.json` sidecar gains `post_opening_calibration.draws_by_tag`. Neither appears in `metrics.csv`.

- [ ] **Step 1: Write the failing integration test**

Append to `tests/test_training.py` (the file already imports `tempfile`, `Path`, and uses `legal_replay` from `tests.goal_line_probe_fixtures`):

```python
def test_calibration_tag_schedule_draw_counts_persisted():
    """v3: a tag schedule draws per-tag counts each step and persists
    calib_n_drawn_by_tag (a dict) into model_iter_*.json state -- in the
    scheduled ratio -- and never into metrics.csv (flat scalars only)."""
    import csv as _csv
    import json as _json
    from scripts.GPU.alphazero.trainer import train
    from tests.goal_line_probe_fixtures import legal_replay

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Two tagged black-to-move rows (ply 5 odd => black to move).
        rows = []
        for gi, tag in ((1, "correction"), (2, "retention")):
            replay = legal_replay(8, game_idx=gi)
            rpath = tmp / f"game_{gi:06d}.json"
            rpath.write_text(_json.dumps(replay))
            rows.append({"game_idx": gi, "case_id": f"game_{gi:06d}_ply_005",
                         "replay_path": str(rpath), "position_ply": 5,
                         "side_to_move": "black", "tag": tag})
        manifest = tmp / "calib_tagged.csv"
        with manifest.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["game_idx", "case_id", "replay_path",
                                               "position_ply", "side_to_move", "tag"])
            w.writeheader()
            w.writerows(rows)

        ckpt_dir = tmp / "ckpt"
        train(
            n_iterations=1, games_per_iteration=1, train_steps_per_iteration=5,
            batch_size=4, buffer_size=1000, checkpoint_dir=str(ckpt_dir),
            mcts_simulations=10, learning_rate=1e-3, hidden=64, n_blocks=2,
            max_moves=10, seed=42,
            post_opening_calibration_enabled=True,
            post_opening_calibration_manifest=str(manifest),
            post_opening_calibration_weight=0.02,
            post_opening_calibration_tag_schedule={"correction": 2, "retention": 1},
            # Isolate self-play output from the shared analyzer fixtures.
            games_dir_override=str(tmp / "games"),
        )

        # metrics.csv must NOT carry the dict (flat scalars only).
        header = next(_csv.reader((ckpt_dir / "metrics.csv").open()))
        assert "calib_n_drawn_by_tag" not in header

        # model_iter_*.json state carries the dict, in the scheduled 2:1 ratio.
        model_jsons = sorted(ckpt_dir.glob("model_iter_*.json"))
        assert model_jsons, "no model_iter_*.json written"
        state = _json.loads(model_jsons[-1].read_text())
        by_tag = state["calib_n_drawn_by_tag"]
        assert set(by_tag) == {"correction", "retention"}
        assert by_tag["correction"] > 0 and by_tag["retention"] > 0
        assert by_tag["correction"] == 2 * by_tag["retention"]  # each step: 2 corr : 1 ret

    print("PASS: tag-stratified calibration draw counts persisted to model_iter JSON")


def test_calibration_tag_schedule_unknown_tag_fails_before_selfplay():
    """A schedule naming a tag absent from the manifest must raise at trainer
    setup (before self-play), not after a wasted iteration. Validation runs
    right after the pool is built, so train() returns the error fast."""
    import csv as _csv
    import json as _json
    from scripts.GPU.alphazero.trainer import train
    from tests.goal_line_probe_fixtures import legal_replay

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        replay = legal_replay(8, game_idx=1)
        rpath = tmp / "game_000001.json"
        rpath.write_text(_json.dumps(replay))
        manifest = tmp / "calib_tagged.csv"
        with manifest.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["game_idx", "case_id", "replay_path",
                                               "position_ply", "side_to_move", "tag"])
            w.writeheader()
            w.writerow({"game_idx": 1, "case_id": "game_000001_ply_005",
                        "replay_path": str(rpath), "position_ply": 5,
                        "side_to_move": "black", "tag": "correction"})

        raised = False
        try:
            train(
                n_iterations=1, games_per_iteration=1, train_steps_per_iteration=5,
                batch_size=4, buffer_size=1000, checkpoint_dir=str(tmp / "ckpt"),
                mcts_simulations=10, learning_rate=1e-3, hidden=64, n_blocks=2,
                max_moves=10, seed=42,
                post_opening_calibration_enabled=True,
                post_opening_calibration_manifest=str(manifest),
                post_opening_calibration_weight=0.02,
                post_opening_calibration_tag_schedule={"correction": 1, "typo_tag": 1},
                games_dir_override=str(tmp / "games"),
            )
        except ValueError:
            raised = True
        assert raised, "expected ValueError for a schedule tag absent from the manifest"
    print("PASS: unknown scheduled tag fails before self-play")
```

Then register both new tests in this file's manual runner: in `main()`, add `test_calibration_tag_schedule_draw_counts_persisted,` and `test_calibration_tag_schedule_unknown_tag_fails_before_selfplay,` to the `tests = [...]` list immediately after `test_calibration_telemetry_persisted_to_metrics_and_model_iter_json,`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_training.py -k tag_schedule -v`
Expected: both FAIL — `TypeError: train() got an unexpected keyword argument 'post_opening_calibration_tag_schedule'`.

- [ ] **Step 3a: Add the `train()` parameter**

In `scripts/GPU/alphazero/trainer.py`, in the `train(` signature, immediately after `post_opening_calibration_batch_fraction: float = 0.10,` (currently line 2361), add:

```python
    post_opening_calibration_tag_schedule: Optional[dict] = None,
```

(`Optional` is already imported — see `Optional[str]` on the adjacent `post_opening_calibration_manifest` param.)

- [ ] **Step 3b: Update the startup print + fail fast on an unknown scheduled tag**

Replace the schema-branch print block (currently lines 2772-2781). Because `_calib_pool` and the schedule are both in scope here — before the iteration loop, so before any self-play — also append a fail-fast `validate_tag_schedule` call so a typo'd tag (e.g. `old_postopening_retention`) raises *now* instead of after iteration 0's self-play. Replace:

```python
        if _calib_pool.schema == "per_row_target":
            print(f"Post-opening calibration: {len(_calib_pool)} positions, "
                  f"mode=per_row_target, "
                  f"weight={effective_post_opening_calibration_weight}, "
                  f"batch_fraction={post_opening_calibration_batch_fraction}")
        else:
            print(f"Post-opening calibration: {len(_calib_pool)} positions, "
                  f"mode=global_target, target={post_opening_calibration_target}, "
                  f"weight={effective_post_opening_calibration_weight}, "
                  f"batch_fraction={post_opening_calibration_batch_fraction}")
```

with (a schedule overrides batch_fraction in the sampling path, so report whichever is active):

```python
        _sampling_desc = (
            f"tag_schedule={post_opening_calibration_tag_schedule}"
            if post_opening_calibration_tag_schedule
            else f"batch_fraction={post_opening_calibration_batch_fraction}")
        if _calib_pool.schema == "per_row_target":
            print(f"Post-opening calibration: {len(_calib_pool)} positions, "
                  f"mode=per_row_target, "
                  f"weight={effective_post_opening_calibration_weight}, "
                  f"{_sampling_desc}")
        else:
            print(f"Post-opening calibration: {len(_calib_pool)} positions, "
                  f"mode=global_target, target={post_opening_calibration_target}, "
                  f"weight={effective_post_opening_calibration_weight}, "
                  f"{_sampling_desc}")
        if post_opening_calibration_tag_schedule:
            _calib_pool.validate_tag_schedule(post_opening_calibration_tag_schedule)
```

- [ ] **Step 3c: Hoist the per-tag accumulator**

After `sum_calib_value_pred: float = 0.0` (currently line 2786), add:

```python
    sum_calib_n_drawn_by_tag: dict = {}
```

- [ ] **Step 3d: Reset it per iteration**

After the per-iteration `sum_calib_value_pred = 0.0` reset (currently line 3760), add:

```python
                sum_calib_n_drawn_by_tag = {}
```

- [ ] **Step 3e: Switch sampling on the schedule + count by tag**

Replace the calibration sampling block (currently lines 3772-3777):

```python
                        if _calib_pool is not None:
                            _k = max(1, round(batch_size * post_opening_calibration_batch_fraction))
                            from .calibration_pool import split_samples
                            _calib_samples = _calib_pool.sample(_k, train_rng)
                            _calib_batch, _calib_weights = split_samples(
                                _calib_samples, _calib_pool.has_weight_scale)
```

with:

```python
                        if _calib_pool is not None:
                            from .calibration_pool import split_samples
                            if post_opening_calibration_tag_schedule:
                                _calib_samples = _calib_pool.sample_by_tag(
                                    post_opening_calibration_tag_schedule, train_rng)
                            else:
                                _k = max(1, round(
                                    batch_size * post_opening_calibration_batch_fraction))
                                _calib_samples = _calib_pool.sample(_k, train_rng)
                            for _s in _calib_samples:
                                sum_calib_n_drawn_by_tag[_s.tag] = (
                                    sum_calib_n_drawn_by_tag.get(_s.tag, 0) + 1)
                            _calib_batch, _calib_weights = split_samples(
                                _calib_samples, _calib_pool.has_weight_scale)
```

- [ ] **Step 3f: Pass the counter into the sidecar block**

In the `loss_accumulator={...}` dict passed to `build_post_opening_calibration_block` (currently lines 3946-3951), add the by-tag key:

```python
                    loss_accumulator={
                        "sum_calib_loss": sum_calib_loss,
                        "sum_calib_n_drawn": sum_calib_n_drawn,
                        "sum_calib_value_pred": sum_calib_value_pred,
                        "sum_calib_n_drawn_by_tag": sum_calib_n_drawn_by_tag,
                        "steps_done": steps_done,
                    },
```

- [ ] **Step 3g: Add the dict to `state` (model_iter JSON), NOT to `iteration_metrics`/CSV**

In the `state = {**iteration_metrics, ...}` dict, after `"selfplay_progress": selfplay_progress,` (currently line 4336), add:

```python
            # Tag-stratified calibration draw counts (dict; JSON sibling only,
            # never iteration_metrics/CSV). Empty when calibration is disabled.
            "calib_n_drawn_by_tag": sum_calib_n_drawn_by_tag if _calib_active else {},
```

- [ ] **Step 4: Run the integration test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_training.py -k tag_schedule -v`
Expected: PASS (both: draw-counts persisted + unknown-tag fails fast before self-play).

- [ ] **Step 5: Run the calibration regression + telemetry tests (no regressions on the uniform path)**

Run: `.venv/bin/python -m pytest tests/test_training.py -k "calib" tests/test_calibration_pool.py tests/test_calibration_cli_flags.py -v`
Expected: PASS — including `test_calibration_telemetry_persisted_to_metrics_and_model_iter_json` (the no-schedule uniform path is unchanged).

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py tests/test_training.py
git commit -m "feat(calibration): tag-stratified sampling + calib_n_drawn_by_tag telemetry in trainer"
```

---

## Final verification (after all four tasks)

- [ ] **Full repo suite green**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (the v2 baseline was 1172 passing; v3 adds tests, removes none).

- [ ] **Diff sanity — no forbidden surfaces touched**

Run: `git diff main --stat` and confirm only `calibration_pool.py`, `train.py`, `trainer.py`, and the three test files changed. Confirm `git diff main -- scripts/GPU/alphazero/build_targeted_calibration_manifest.py` is empty, and that `CSV_FIELDNAMES` / `train_step` / `alphazero_loss_batch` are untouched (`git diff main -- scripts/GPU/alphazero/trainer.py | grep -nE "CSV_FIELDNAMES|def train_step|def alphazero_loss_batch"` returns nothing).

---

## Operator runbook (post-merge; manual, no commits)

Locked parameters for the v3 experiment (decided 2026-06-25): calib **weight 0.01** (isolate the tag-schedule variable from v2), **retention_weight 1.0** (uniform per-row weight → tag schedule alone controls the draw ratio), correction target **−0.35**, schedule **2:1:2:1**. Gates A–D are unchanged from the v2 spec §10.

**1. Build the v3 manifest** (gitignored — regenerable, do not commit). The builder emits exactly these four tag strings: `black_predrop_correction`, `red_predrop_retention`, `old_post_opening_retention`, `goal_line_retention` (verified against `build_targeted_calibration_manifest.py`).

```bash
.venv/bin/python -m scripts.GPU.alphazero.build_targeted_calibration_manifest \
  --correction-manifest logs/eval/loss_analysis_v2_calib020_0001_vs_0379_black/0001_black_post_opening_predrop_train_manifest.csv \
  --correction-holdout-manifest logs/eval/loss_analysis_v2_calib020_0001_vs_0379_black/0001_black_post_opening_top30_predrop_probe_manifest.csv \
  --red-predrop-cases logs/eval/calib020_0001_red_loss_post_opening_predrop_probe/position_probe_cases.csv \
  --old-post-opening-cases logs/eval/black_predrop_calib010_checkpoint_sweep_old_post_opening/position_probe_cases.csv \
  --old-post-opening-anchor-label "alphazero-v2-calib020-from0409:0001" \
  --goal-line-cases logs/eval/calib020_goal_line_sweep/goal_line_trigger_probe_cases.csv \
  --goal-line-candidates logs/eval/loss_analysis_v2_1/goal_line_trigger_probe_candidates.csv \
  --correction-target -0.35 \
  --retention-weight 1.0 \
  --out logs/eval/targeted_calibration_v3_strat_from_calib020_0001.csv
```

Sanity (from the builder's per-tag stats print, §5.5 / §6 baselines): expect **128 rows** — `black_predrop_correction` n=50, `red_predrop_retention` n=30, `old_post_opening_retention` n=30, `goal_line_retention` n=18; all `weight_scale=1.0`.

**2. Run the 1-iteration v3 experiment.** The schedule controls the calibration draw count, so `--post-opening-calibration-batch-fraction` is intentionally omitted (it is ignored when a schedule is set).

```bash
.venv/bin/python -m scripts.GPU.alphazero.train \
  --load-weights checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --checkpoint-dir checkpoints/alphazero-v3-strat-from-calib020-0001 \
  --iterations 1 --lr 0.0003 --curriculum-sizes 24 \
  --games-per-iter 100 --simulations 400 --max-moves 280 --batch-size 64 \
  --mcts-eval-batch-size 14 --mcts-pending-virtual-visits 8 --mcts-stall-flush-sims 48 \
  --n-workers 10 \
  --opening-noise-ply 10 --opening-dirichlet-alpha 0.7 --opening-dirichlet-eps 0.35 \
  --resign-enabled --resign-min-ply 80 --resign-threshold -0.945 --resign-window 12 \
  --resign-k 4 --resign-min-visits 200 \
  --adjudicate-enabled --adjudicate-min-ply 240 --max-positions-per-game 280 \
  --post-opening-calibration-enabled \
  --post-opening-calibration-manifest logs/eval/targeted_calibration_v3_strat_from_calib020_0001.csv \
  --post-opening-calibration-weight 0.01 \
  --post-opening-calibration-target -0.35 \
  --post-opening-calibration-tag-schedule black_predrop_correction=2,goal_line_retention=1,old_post_opening_retention=2,red_predrop_retention=1
```

After the run, confirm `model_iter_0001.json` (in the checkpoint dir) carries `calib_n_drawn_by_tag` with all four tags in the 2:1:2:1 ratio, and that the sidecar `post_opening_calibration.draws_by_tag` matches. (A tag typo now fails fast: `train()` validates the schedule against the pool immediately after loading the manifest — before any self-play — and raises `ValueError` naming the missing tags, so no iteration is wasted.)

**Index/location gotcha:** the checkpoint is **1-based** (`model_iter_{iteration+1:04d}` → first iteration = `model_iter_0001`), but the sidecar stats file is **0-based** and lives in the **games dir** (`iter_{iteration:04d}_stats.json` → first iteration = `iter_0000_stats.json`, default dir `scripts/GPU/logs/games/` unless `--games-dir` was passed). Post-run sanity check:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

ckpt = Path("checkpoints/alphazero-v3-strat-from-calib020-0001")
state = json.loads((ckpt / "model_iter_0001.json").read_text())
print("model_iter calib_n_drawn_by_tag:", state.get("calib_n_drawn_by_tag"))

# Sidecar is 0-indexed by loop iteration and lives in the games dir, so the
# FIRST iteration's sidecar is iter_0000_stats.json (its checkpoint is
# model_iter_0001). Default games dir below; change if you passed --games-dir.
stats = Path("scripts/GPU/logs/games/iter_0000_stats.json")
if stats.exists():
    sidecar = json.loads(stats.read_text())
    print("sidecar draws_by_tag:",
          sidecar.get("post_opening_calibration", {}).get("draws_by_tag"))
else:
    print(f"sidecar not found at {stats} (check --games-dir)")
PY
```

**3. Gates A–D (400-sim probes; promote only if all four pass)** — verbatim from spec §10:
- **Gate A — black pre-drop** (frozen-30, held out). Baseline over 50.0% / severe 43.3% / mean +0.257. **Pass:** mean ≤ 0.0 **and** severe materially below 43.3%.
- **Gate B — goal-line.** Baseline over 5.6% / severe 0.0%. **Pass:** severe 0.0% **and** over ≤ 11.1%.
- **Gate C — old broad post-opening.** Baseline over 33.3% / severe 13.3% / mean +0.099. **Pass:** severe ≤ 13.3% **and** over ≤ 33.3% **and** mean_black_value ≤ +0.099.
- **Gate D — red pre-drop.** Baseline over 13.3% / severe 0.0% / mean −0.188. **Pass:** severe = 0.0% **and** mean_black_value ≤ 0.0.

**4. Promotion match** — only if all four gates pass: run **vs current best `calib020_0001`** (`…/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`), not 0379.

---

## Self-Review (completed)

- **Scope coverage** — all five locked v3-A surfaces map to tasks: pool `sample_by_tag` (T1), CLI flag + parser (T2), sidecar `draws_by_tag` (T3), trainer threading + `calib_n_drawn_by_tag` state telemetry + startup print (T4), 1-iteration experiment (runbook). Deferred "B" (per-tag loss/value) explicitly out of scope.
- **Stable-surface guard** — `train_step`/10-tuple, `alphazero_loss_batch`, `split_samples`, `CSV_FIELDNAMES`, and the manifest builder are untouched; final verification step diffs to prove it.
- **Type/name consistency** — schedule is `dict[str, int]` end to end (CLI parser → `train()` kwarg → `sample_by_tag`); telemetry keys are `calib_n_drawn_by_tag` (state) and `draws_by_tag` (sidecar), each consistent across the producing task (T3/T4) and its test.
- **No placeholders** — every code step shows complete before/after content; every test step shows the asserting test body; every run step gives the command + expected result.

**Review revisions (2026-06-25):**
- **Fail-fast tag validation** — added `CalibrationPool.validate_tag_schedule` (T1, unit-tested) called by both `sample_by_tag` (DRY) and the trainer right after pool build, before self-play (T4). A trainer-level test (T4) proves `train()` raises at setup, not after a wasted iteration.
- **"Byte-identical" wording corrected** — the no-schedule path is *sampling/loss/optimization-identical*; JSON telemetry gains the additive by-tag fields on both paths (per reviewer preference: keep telemetry, fix wording). Verified no existing telemetry test asserts dict equality.
- **Sidecar block `version` kept at `1`** — `draws_by_tag` is additive and `test_build_post_opening_calibration_block` pins `version == 1`; bump declined per the reviewer's stated condition.
