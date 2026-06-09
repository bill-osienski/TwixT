# Eval Loss Analyzer (V1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only postprocessor over checkpoint-eval `*_games.jsonl` files that explains how checkpoint A is losing to checkpoint B — by color, game length, termination, and across training branches.

**Architecture:** Pure, IO-free analysis module (`eval_loss_analysis.py`) that takes row dicts and returns dicts/lists, reusing `eval_elo` for all stats; plus a thin CLI (`eval_loss_analyzer.py`) that does file IO, writes JSON/CSV outputs, and prints a console summary. Mirrors the existing `eval_elo` (pure) / `eval_runner` (IO) split.

**Tech Stack:** Python 3.14, stdlib only (`json`, `csv`, `argparse`, `glob`, `statistics`), pytest. Reuses `scripts.GPU.alphazero.eval_elo` and `eval_runner.short_id`.

**Spec:** `docs/superpowers/specs/2026-06-09-eval-loss-analyzer-design.md`

**Run tests with:** `.venv/bin/python -m pytest <path> -v` (the repo's `pytest.ini` sets `testpaths = tests`).

**Conventions confirmed:** tests live flat in `tests/`; imports use `from scripts.GPU.alphazero.<mod> import ...`; `short_id("model_iter_0399.safetensors") -> "0399"` and `short_id("0379") -> "0379"`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `scripts/GPU/alphazero/eval_loss_analysis.py` | Pure logic: validation, scoring, color/length/overall summaries, worst-loss sampler, branch combiner. No IO. |
| `scripts/GPU/alphazero/eval_loss_analyzer.py` | Thin CLI: arg parsing, file load/glob, sidecar read, JSON/CSV writes, console summary. |
| `tests/test_eval_loss_analysis.py` | Unit tests for the pure module. |
| `tests/test_eval_loss_analyzer_cli.py` | CLI test over tmp fixtures (no MLX, fast). |

---

## Task 1: Validation + per-row scoring

**Files:**
- Create: `scripts/GPU/alphazero/eval_loss_analysis.py`
- Test: `tests/test_eval_loss_analysis.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_loss_analysis.py`:

```python
import pytest

from scripts.GPU.alphazero.eval_loss_analysis import (
    score_for_checkpoint, a_color, validate_rows,
)

A = "ckpts/model_iter_0399.safetensors"
B = "ckpts/model_iter_0379.safetensors"


def _row(game_idx, red, black, winner, reason="win", n=50, task_id=0,
         pairing_id="0399_vs_0379"):
    if winner == "red":
        rs, bs, wc = 1.0, 0.0, red
    elif winner == "black":
        rs, bs, wc = 0.0, 1.0, black
    else:
        rs, bs, wc = 0.5, 0.5, None
    return {
        "task_id": task_id, "pairing_id": pairing_id, "game_idx": game_idx,
        "red_checkpoint": red, "black_checkpoint": black,
        "winner": winner, "winner_checkpoint": wc, "reason": reason,
        "n_moves": n, "red_score": rs, "black_score": bs,
    }


def test_score_for_checkpoint_win_red_a():
    r = _row(0, A, B, "red")
    assert score_for_checkpoint(r, A) == 1.0
    assert score_for_checkpoint(r, B) == 0.0


def test_score_for_checkpoint_win_black_a():
    # A is seated as black this game and wins.
    r = _row(1, B, A, "black")
    assert score_for_checkpoint(r, A) == 1.0
    assert score_for_checkpoint(r, B) == 0.0


def test_score_for_checkpoint_draw_state_cap():
    r = _row(2, A, B, None, reason="state_cap", n=280)
    assert score_for_checkpoint(r, A) == 0.5
    assert score_for_checkpoint(r, B) == 0.5


def test_a_color_tracks_seat():
    assert a_color(_row(0, A, B, "red"), A) == "red"
    assert a_color(_row(1, B, A, "black"), A) == "black"


def test_validation_rejects_winner_checkpoint_mismatch():
    bad = _row(0, A, B, "red")
    bad["winner_checkpoint"] = B  # winner says red(A) but ckpt points at B
    with pytest.raises(ValueError, match="winner_checkpoint"):
        validate_rows([bad])


def test_validation_rejects_inconsistent_draw_scores():
    bad = _row(0, A, B, None, reason="state_cap", n=280)
    bad["red_score"] = 1.0  # draw must be 0.5/0.5
    with pytest.raises(ValueError, match="draw"):
        validate_rows([bad])


def test_validation_rejects_unknown_error():
    bad = _row(0, A, B, None, reason="unknown_error")
    bad["winner_checkpoint"] = None
    with pytest.raises(ValueError, match="unknown_error"):
        validate_rows([bad])


def test_validation_rejects_mixed_jsonl():
    rows = [_row(0, A, B, "red"), _row(1, A, "ckpts/model_iter_0123.safetensors", "red")]
    with pytest.raises(ValueError, match="mixed"):
        validate_rows(rows, A, B)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_analysis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.GPU.alphazero.eval_loss_analysis'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/GPU/alphazero/eval_loss_analysis.py`:

```python
"""Pure loss-shape analysis over checkpoint-eval *_games.jsonl rows.

No IO, no MLX: rows in (plain dicts), dicts/lists out (ready to serialize).
Scoring matches eval_summary: A/B keyed off winner_checkpoint, color off
red/black_checkpoint, draws (state_cap/board_full) = 0.5 for both sides.
"""
from __future__ import annotations

from statistics import mean

from scripts.GPU.alphazero.eval_elo import (
    score_rate, elo_diff, score_ci_trinomial, elo_ci, verdict,
)
from scripts.GPU.alphazero.eval_runner import short_id

LENGTH_BUCKETS_DEFAULT = (40, 60, 80, 120, 279, 280)

REQUIRED_KEYS = {
    "task_id", "pairing_id", "game_idx", "red_checkpoint", "black_checkpoint",
    "winner", "winner_checkpoint", "reason", "n_moves", "red_score", "black_score",
}
DRAW_REASONS = {"state_cap", "board_full"}
VALID_REASONS = {"win", "state_cap", "board_full", "unknown_error"}


def _require(i, cond, msg):
    if not cond:
        raise ValueError(f"row {i}: {msg}")


def validate_rows(rows, a_ckpt=None, b_ckpt=None):
    """Fail loud on any row that breaks the eval scoring invariants.

    When a_ckpt and b_ckpt are both given, also require every row to be
    between exactly those two checkpoints (catches a mixed/concatenated
    JSONL of more than one pairing).
    """
    if not rows:
        raise ValueError("no rows to analyze")
    ab = {a_ckpt, b_ckpt} if (a_ckpt is not None and b_ckpt is not None) else None
    for i, r in enumerate(rows):
        missing = REQUIRED_KEYS - r.keys()
        _require(i, not missing, f"missing keys {sorted(missing)}")
        reason = r["reason"]
        _require(i, reason in VALID_REASONS, f"bad reason {reason!r}")
        _require(i, reason != "unknown_error",
                 "reason 'unknown_error' not handled in V1 (none expected in current data)")
        winner = r["winner"]
        _require(i, winner in ("red", "black", None), f"bad winner {winner!r}")
        red, black = r["red_checkpoint"], r["black_checkpoint"]
        if winner == "red":
            _require(i, r["winner_checkpoint"] == red, "winner_checkpoint != red_checkpoint")
            _require(i, r["red_score"] == 1.0 and r["black_score"] == 0.0,
                     "red-win scores not 1.0/0.0")
        elif winner == "black":
            _require(i, r["winner_checkpoint"] == black, "winner_checkpoint != black_checkpoint")
            _require(i, r["red_score"] == 0.0 and r["black_score"] == 1.0,
                     "black-win scores not 0.0/1.0")
        else:  # draw
            _require(i, r["winner_checkpoint"] is None, "draw winner_checkpoint not None")
            _require(i, r["red_score"] == 0.5 and r["black_score"] == 0.5,
                     "draw scores not 0.5/0.5")
            _require(i, reason in DRAW_REASONS, f"draw reason {reason!r} not a draw reason")
        if ab is not None and {red, black} != ab:
            _require(i, False,
                     f"checkpoints {{{short_id(red)}, {short_id(black)}}} != resolved A/B "
                     "— mixed JSONL?")


def score_for_checkpoint(row, ckpt):
    """1.0 if ckpt won, 0.5 on a draw (no winner), else 0.0. Keyed off
    winner_checkpoint — never off color."""
    if row["winner_checkpoint"] == ckpt:
        return 1.0
    if row["winner_checkpoint"] is None:
        return 0.5
    return 0.0


def a_color(row, a_ckpt):
    """Which seat A played this game: 'red' or 'black'."""
    if row["red_checkpoint"] == a_ckpt:
        return "red"
    if row["black_checkpoint"] == a_ckpt:
        return "black"
    raise ValueError(f"A checkpoint {short_id(a_ckpt)} not in row {row['game_idx']}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_analysis.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_loss_analysis.py tests/test_eval_loss_analysis.py
git commit -m "feat(eval): loss-analysis validation + per-row scoring"
```

---

## Task 2: A/B checkpoint resolution

**Files:**
- Modify: `scripts/GPU/alphazero/eval_loss_analysis.py`
- Test: `tests/test_eval_loss_analysis.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_loss_analysis.py`:

```python
from scripts.GPU.alphazero.eval_loss_analysis import resolve_checkpoints


def _rows_ab():
    # balanced colors: A red on even idx, A black on odd idx
    return [_row(0, A, B, "red"), _row(1, B, A, "black")]


def test_resolve_from_override():
    a, b = resolve_checkpoints(_rows_ab(), a_override=A, b_override=B)
    assert (a, b) == (A, B)


def test_resolve_from_sidecar():
    summary = {"checkpoint_a": A, "checkpoint_b": B}
    a, b = resolve_checkpoints(_rows_ab(), summary=summary)
    assert (a, b) == (A, B)


def test_resolve_from_pairing_fallback():
    # no override, no sidecar -> infer from pairing_id "0399_vs_0379" via short_id
    a, b = resolve_checkpoints(_rows_ab(), pairing_id="0399_vs_0379")
    assert (a, b) == (A, B)


def test_resolve_rejects_absent_checkpoint():
    with pytest.raises(ValueError, match="not present"):
        resolve_checkpoints(_rows_ab(), a_override="ckpts/model_iter_9999.safetensors",
                            b_override=B)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_analysis.py -k resolve -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_checkpoints'`

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/GPU/alphazero/eval_loss_analysis.py`:

```python
def _match_short(ckpts, sid):
    hits = [c for c in ckpts if short_id(c) == sid]
    if len(hits) != 1:
        raise ValueError(f"pairing side {sid!r} matched {len(hits)} checkpoints, expected 1")
    return hits[0]


def _infer_from_pairing(rows, pairing_id):
    pid = pairing_id or rows[0]["pairing_id"]
    if "_vs_" not in pid:
        raise ValueError(f"cannot infer A/B: pairing_id {pid!r} has no '_vs_'")
    a_id, b_id = pid.split("_vs_", 1)
    ckpts = ({r["red_checkpoint"] for r in rows}
             | {r["black_checkpoint"] for r in rows})
    return _match_short(ckpts, a_id), _match_short(ckpts, b_id)


def resolve_checkpoints(rows, pairing_id=None, a_override=None,
                        b_override=None, summary=None):
    """Resolve (A, B) checkpoint paths.

    Precedence: explicit overrides -> sidecar summary checkpoint_a/checkpoint_b
    -> infer from pairing_id + short_id of the row checkpoints. Both resolved
    paths must actually appear across the rows.
    """
    if a_override and b_override:
        a, b = a_override, b_override
    elif summary and summary.get("checkpoint_a") and summary.get("checkpoint_b"):
        a, b = summary["checkpoint_a"], summary["checkpoint_b"]
    else:
        a, b = _infer_from_pairing(rows, pairing_id)
    present = ({r["red_checkpoint"] for r in rows}
               | {r["black_checkpoint"] for r in rows})
    for label, ckpt in (("A", a), ("B", b)):
        if ckpt not in present:
            raise ValueError(f"resolved {label} checkpoint {ckpt!r} not present in rows")
    return a, b
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_analysis.py -k resolve -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_loss_analysis.py tests/test_eval_loss_analysis.py
git commit -m "feat(eval): A/B checkpoint resolution (override/sidecar/pairing)"
```

---

## Task 3: By-color and by-length summaries

**Files:**
- Modify: `scripts/GPU/alphazero/eval_loss_analysis.py`
- Test: `tests/test_eval_loss_analysis.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_loss_analysis.py`:

```python
from scripts.GPU.alphazero.eval_loss_analysis import (
    summarize_by_color, summarize_by_length,
)


def test_by_color_uses_checkpoint_assignment_not_winner_color():
    # A red and wins (red win) ; A black and wins (black win).
    # Both are A-wins. By-color must file each under A's SEAT, not winner color.
    rows = [_row(0, A, B, "red"), _row(1, B, A, "black")]
    by_color = {c["a_color"]: c for c in summarize_by_color(rows, A, B)}
    assert by_color["red"]["games"] == 1
    assert by_color["red"]["a_wins"] == 1   # A won its red game
    assert by_color["black"]["games"] == 1
    assert by_color["black"]["a_wins"] == 1  # A won its black game
    assert by_color["red"]["a_score_rate"] == 1.0
    assert by_color["black"]["a_score_rate"] == 1.0


def test_by_length_buckets_280_state_cap():
    rows = [
        _row(0, A, B, "red", n=30),                       # <=40
        _row(1, A, B, "black", n=50),                     # 41-60
        _row(2, A, B, "red", n=279),                      # 121-279
        _row(3, A, B, None, reason="state_cap", n=280),   # 280
    ]
    by_len = {b["length_bucket"]: b for b in summarize_by_length(rows, A, B)}
    assert by_len["<=40"]["games"] == 1
    assert by_len["41-60"]["games"] == 1
    assert by_len["121-279"]["games"] == 1
    assert by_len["280"]["games"] == 1
    assert by_len["280"]["draws"] == 1
    # empty buckets (61-80, 81-120) are omitted, not zero-filled
    assert "61-80" not in by_len
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_analysis.py -k "by_color or by_length" -v`
Expected: FAIL with `ImportError: cannot import name 'summarize_by_color'`

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/GPU/alphazero/eval_loss_analysis.py`:

```python
def _ab_stats(rows, a_ckpt):
    """Games / A-wins / B-wins / draws / A-score-rate / avg moves for a subset."""
    n = len(rows)
    a_wins = sum(1 for r in rows if r["winner_checkpoint"] == a_ckpt)
    draws = sum(1 for r in rows if r["winner"] is None)
    b_wins = n - a_wins - draws
    return {
        "games": n,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_score_rate": (score_rate(a_wins, draws, n) if n else None),
        "avg_moves": (round(mean(r["n_moves"] for r in rows), 2) if n else None),
    }


def _bucket_name(edge, buckets):
    idx = buckets.index(edge)
    lo = (buckets[idx - 1] + 1) if idx > 0 else None
    if lo is None:
        return f"<={edge}"
    if lo == edge:
        return f"{edge}"
    return f"{lo}-{edge}"


def _length_bucket_label(n_moves, buckets):
    for edge in buckets:
        if n_moves <= edge:
            return _bucket_name(edge, buckets)
    return f">{buckets[-1]}"


def summarize_by_color(rows, a_ckpt, b_ckpt):
    out = []
    for color in ("red", "black"):
        sub = [r for r in rows if a_color(r, a_ckpt) == color]
        out.append({"a_color": color, **_ab_stats(sub, a_ckpt)})
    return out


def summarize_by_length(rows, a_ckpt, b_ckpt, buckets=LENGTH_BUCKETS_DEFAULT):
    ordered_labels = [_bucket_name(e, buckets) for e in buckets]
    groups = {}
    for r in rows:
        groups.setdefault(_length_bucket_label(r["n_moves"], buckets), []).append(r)
    # bucket order first, then any overflow label (e.g. ">280") last
    labels = ordered_labels + [k for k in groups if k not in ordered_labels]
    out = []
    for lbl in labels:
        sub = groups.get(lbl)
        if sub:  # omit empty buckets
            out.append({"length_bucket": lbl, **_ab_stats(sub, a_ckpt)})
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_analysis.py -k "by_color or by_length" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_loss_analysis.py tests/test_eval_loss_analysis.py
git commit -m "feat(eval): by-color and by-length loss summaries"
```

---

## Task 4: Overall summary (Elo/CI/verdict reuse + termination block)

**Files:**
- Modify: `scripts/GPU/alphazero/eval_loss_analysis.py`
- Test: `tests/test_eval_loss_analysis.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eval_loss_analysis.py`:

```python
from scripts.GPU.alphazero.eval_loss_analysis import summarize_overall


def test_overall_summary_scoring_and_termination():
    # 6 games: A wins 2, B wins 3, 1 state-cap draw.
    rows = [
        _row(0, A, B, "red"),     # A win
        _row(1, B, A, "black"),   # A win
        _row(2, A, B, "black"),   # B win
        _row(3, B, A, "red"),     # B win
        _row(4, A, B, "black"),   # B win
        _row(5, A, B, None, reason="state_cap", n=280),  # draw
    ]
    s = summarize_overall(rows, A, B)
    assert s["games"] == 6
    assert s["a_wins"] == 2
    assert s["b_wins"] == 3
    assert s["draws"] == 1
    assert s["a_score"] == 2.5            # 2 wins + 0.5 draw
    assert s["a_score_rate"] == 2.5 / 6
    assert s["verdict"] == "worse"        # rate < 0.48
    assert s["elo"] < 0                   # A scoring below 0.5 -> negative Elo
    assert len(s["elo_ci95"]) == 2 and s["elo_ci95"][0] < s["elo_ci95"][1]
    assert s["termination"] == {
        "win": 5, "state_cap": 1, "board_full": 0, "unknown_error": 0,
        "draws": 1, "state_cap_rate": 1 / 6, "board_full_rate": 0.0,
    }
    assert s["color_gap"] is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_analysis.py -k overall_summary -v`
Expected: FAIL with `ImportError: cannot import name 'summarize_overall'`

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/GPU/alphazero/eval_loss_analysis.py`:

```python
def summarize_overall(rows, a_ckpt, b_ckpt):
    n = len(rows)
    a_wins = sum(1 for r in rows if r["winner_checkpoint"] == a_ckpt)
    b_wins = sum(1 for r in rows if r["winner_checkpoint"] == b_ckpt)
    wins = sum(1 for r in rows if r["reason"] == "win")
    state_caps = sum(1 for r in rows if r["reason"] == "state_cap")
    board_full = sum(1 for r in rows if r["reason"] == "board_full")
    draws = state_caps + board_full
    rate = score_rate(a_wins, draws, n)
    s_lo, s_hi = score_ci_trinomial(a_wins, draws, b_wins)
    e_lo, e_hi = elo_ci(a_wins, draws, b_wins)

    by_color = summarize_by_color(rows, a_ckpt, b_ckpt)
    rates = {c["a_color"]: c["a_score_rate"] for c in by_color}
    red_rate, black_rate = rates.get("red"), rates.get("black")
    color_gap = (red_rate - black_rate
                 if red_rate is not None and black_rate is not None else None)

    return {
        "games": n,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_score": a_wins + 0.5 * draws,
        "a_score_rate": rate,
        "elo": elo_diff(rate, n),
        "elo_ci95": [e_lo, e_hi],
        "score_rate_ci95": [s_lo, s_hi],
        "verdict": verdict(rate),
        "color_gap": color_gap,
        "termination": {
            "win": wins,
            "state_cap": state_caps,
            "board_full": board_full,
            "unknown_error": 0,
            "draws": draws,
            "state_cap_rate": state_caps / n,
            "board_full_rate": board_full / n,
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_analysis.py -k overall_summary -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_loss_analysis.py tests/test_eval_loss_analysis.py
git commit -m "feat(eval): overall loss summary with elo/CI/verdict + termination"
```

---

## Task 5: Worst-loss sampler

**Files:**
- Modify: `scripts/GPU/alphazero/eval_loss_analysis.py`
- Test: `tests/test_eval_loss_analysis.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eval_loss_analysis.py`:

```python
from scripts.GPU.alphazero.eval_loss_analysis import sample_worst_losses


def test_sample_worst_losses_buckets_and_order():
    rows = [
        _row(0, A, B, "black", n=35),                     # A loss, short
        _row(1, A, B, "black", n=120),                    # A loss, long
        _row(2, A, B, "red"),                             # A win (excluded from losses)
        _row(3, A, B, None, reason="state_cap", n=280),   # draw cap
    ]
    out = sample_worst_losses(rows, A, B, limit=10)
    by_bucket = {}
    for w in out:
        by_bucket.setdefault(w["loss_bucket"], []).append(w)

    # short_loss sorted shortest-first; long_loss sorted longest-first
    assert [w["n_moves"] for w in by_bucket["short_loss"]] == [35, 120]
    assert [w["n_moves"] for w in by_bucket["long_loss"]] == [120, 35]
    assert [w["game_idx"] for w in by_bucket["draw_cap"]] == [3]
    # row shape carries the inspection handles
    sample = by_bucket["short_loss"][0]
    assert sample["a_color"] == "red"          # A was seated red in game 0
    assert sample["a_score"] == 0.0
    assert sample["game_idx"] == 0 and sample["task_id"] == 0
    assert sample["reason"] == "win"           # decisive B win
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_analysis.py -k worst_losses -v`
Expected: FAIL with `ImportError: cannot import name 'sample_worst_losses'`

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/GPU/alphazero/eval_loss_analysis.py`:

```python
def _worst_row(r, bucket, a_ckpt):
    return {
        "loss_bucket": bucket,
        "game_idx": r["game_idx"],
        "task_id": r["task_id"],
        "a_color": a_color(r, a_ckpt),
        "winner": r["winner"],
        "reason": r["reason"],
        "n_moves": r["n_moves"],
        "a_score": score_for_checkpoint(r, a_ckpt),
        "red_checkpoint": r["red_checkpoint"],
        "black_checkpoint": r["black_checkpoint"],
    }


def sample_worst_losses(rows, a_ckpt, b_ckpt, limit=50):
    """Up to `limit` rows per bucket: A's shortest decisive losses
    (short_loss), A's longest decisive losses (long_loss), and the
    non-decisive cap/board-full games (draw_cap). short_loss and long_loss
    draw from the same A-loss pool, so they overlap when losses are few."""
    a_losses = [r for r in rows if score_for_checkpoint(r, a_ckpt) == 0.0]
    caps = [r for r in rows if r["winner"] is None]
    short = sorted(a_losses, key=lambda r: r["n_moves"])[:limit]
    long = sorted(a_losses, key=lambda r: -r["n_moves"])[:limit]
    out = []
    for bucket, group in (("short_loss", short), ("long_loss", long),
                          ("draw_cap", caps[:limit])):
        out.extend(_worst_row(r, bucket, a_ckpt) for r in group)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_analysis.py -k worst_losses -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_loss_analysis.py tests/test_eval_loss_analysis.py
git commit -m "feat(eval): worst-loss sampler (short/long/draw_cap buckets)"
```

---

## Task 6: Branch combiner + match orchestrator

**Files:**
- Modify: `scripts/GPU/alphazero/eval_loss_analysis.py`
- Test: `tests/test_eval_loss_analysis.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_loss_analysis.py`:

```python
from scripts.GPU.alphazero.eval_loss_analysis import (
    analyze_match, combine_branch_summaries,
)


def test_analyze_match_shape():
    rows = [_row(0, A, B, "red"), _row(1, B, A, "black"),
            _row(2, A, B, "black"), _row(3, B, A, "red")]
    s = analyze_match(rows, A, B, match="demo", pairing_id="0399_vs_0379")
    assert s["match"] == "demo"
    assert s["pairing_id"] == "0399_vs_0379"
    assert s["a_checkpoint"] == A and s["b_checkpoint"] == B
    assert s["games"] == 4
    assert isinstance(s["by_color"], list) and isinstance(s["by_length"], list)
    assert "termination" in s


def test_combined_branch_comparison_orders_by_a_score_rate():
    weak = {"match": "weak", "pairing_id": "p", "a_checkpoint": A,
            "b_checkpoint": B, "games": 10, "a_score_rate": 0.30,
            "a_wins": 3, "b_wins": 7, "draws": 0, "elo": -147.0, "verdict": "worse"}
    strong = {**weak, "match": "strong", "a_score_rate": 0.55,
              "a_wins": 6, "b_wins": 4, "elo": 35.0, "verdict": "stronger"}
    combined = combine_branch_summaries([weak, strong])
    assert [r["match"] for r in combined] == ["strong", "weak"]  # descending rate
    assert list(combined[0].keys()) == [
        "match", "pairing_id", "a_checkpoint", "b_checkpoint", "games",
        "a_score_rate", "a_wins", "b_wins", "draws", "elo", "verdict",
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_analysis.py -k "analyze_match or combined" -v`
Expected: FAIL with `ImportError: cannot import name 'analyze_match'`

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/GPU/alphazero/eval_loss_analysis.py`:

```python
def analyze_match(rows, a_ckpt, b_ckpt, *, match=None, pairing_id=None,
                  length_buckets=LENGTH_BUCKETS_DEFAULT):
    """Full per-match summary dict (the loss_summary.json payload). The
    worst-loss CSV is produced separately via sample_worst_losses()."""
    overall = summarize_overall(rows, a_ckpt, b_ckpt)
    return {
        "match": match,
        "pairing_id": pairing_id or rows[0]["pairing_id"],
        "a_checkpoint": a_ckpt,
        "b_checkpoint": b_ckpt,
        **overall,
        "by_color": summarize_by_color(rows, a_ckpt, b_ckpt),
        "by_length": summarize_by_length(rows, a_ckpt, b_ckpt, length_buckets),
    }


def combine_branch_summaries(match_summaries):
    """One row per match, sorted descending by a_score_rate (strongest
    branch-vs-anchor first)."""
    rows = [
        {
            "match": s["match"],
            "pairing_id": s["pairing_id"],
            "a_checkpoint": s["a_checkpoint"],
            "b_checkpoint": s["b_checkpoint"],
            "games": s["games"],
            "a_score_rate": s["a_score_rate"],
            "a_wins": s["a_wins"],
            "b_wins": s["b_wins"],
            "draws": s["draws"],
            "elo": s["elo"],
            "verdict": s["verdict"],
        }
        for s in match_summaries
    ]
    rows.sort(key=lambda r: r["a_score_rate"], reverse=True)
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_analysis.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_loss_analysis.py tests/test_eval_loss_analysis.py
git commit -m "feat(eval): match orchestrator + cross-branch combiner"
```

---

## Task 7: Thin CLI + console summary

**Files:**
- Create: `scripts/GPU/alphazero/eval_loss_analyzer.py`
- Test: `tests/test_eval_loss_analyzer_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_loss_analyzer_cli.py`:

```python
import json
from pathlib import Path

from scripts.GPU.alphazero.eval_loss_analyzer import main

A = "ckpts/model_iter_0399.safetensors"
B = "ckpts/model_iter_0379.safetensors"


def _row(game_idx, red, black, winner, reason="win", n=50, task_id=0,
         pairing_id="0399_vs_0379"):
    if winner == "red":
        rs, bs, wc = 1.0, 0.0, red
    elif winner == "black":
        rs, bs, wc = 0.0, 1.0, black
    else:
        rs, bs, wc = 0.5, 0.5, None
    return {
        "task_id": task_id, "pairing_id": pairing_id, "game_idx": game_idx,
        "red_checkpoint": red, "black_checkpoint": black,
        "winner": winner, "winner_checkpoint": wc, "reason": reason,
        "n_moves": n, "red_score": rs, "black_score": bs,
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_cli_writes_outputs_and_combined(tmp_path, capsys):
    # match 1: A loses badly (B wins all 4)
    m1 = tmp_path / "weak_0399_vs_0379_games.jsonl"
    _write_jsonl(m1, [_row(i, A, B, "black", n=30 + i) for i in range(4)])
    # match 2: A wins all 4
    m2 = tmp_path / "strong_0399_vs_0379_games.jsonl"
    _write_jsonl(m2, [_row(i, A, B, "red", n=30 + i) for i in range(4)])
    out = tmp_path / "loss_analysis"

    rc = main(["--games-jsonl", str(m1), "--games-jsonl", str(m2),
               "--output-dir", str(out)])
    assert rc == 0

    for stem in ("weak_0399_vs_0379", "strong_0399_vs_0379"):
        assert (out / f"{stem}_loss_summary.json").exists()
        assert (out / f"{stem}_by_color.csv").exists()
        assert (out / f"{stem}_by_length.csv").exists()
        assert (out / f"{stem}_worst_losses.csv").exists()

    combined = (out / "combined_branch_comparison.csv").read_text().splitlines()
    # header + 2 rows, strong first (higher a_score_rate)
    assert combined[1].startswith("strong_0399_vs_0379")
    assert combined[2].startswith("weak_0399_vs_0379")

    summary = json.loads((out / "weak_0399_vs_0379_loss_summary.json").read_text())
    assert summary["verdict"] == "worse"
    assert "LOSS ANALYSIS" in capsys.readouterr().out


def test_cli_skips_self_match(tmp_path, capsys):
    m = tmp_path / "0419_vs_0419_sanity_games.jsonl"
    _write_jsonl(m, [_row(i, A, A, "red", pairing_id="0399_vs_0399") for i in range(2)])
    rc = main(["--games-jsonl", str(m), "--output-dir", str(tmp_path / "out")])
    assert rc == 0
    assert "self-match" in capsys.readouterr().out
    assert not (tmp_path / "out" / "combined_branch_comparison.csv").exists()
```

Note: in `test_cli_skips_self_match` both seats are A and `pairing_id` is `0399_vs_0399`, so resolution infers A==B and the CLI skips before validating.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_analyzer_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.GPU.alphazero.eval_loss_analyzer'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/GPU/alphazero/eval_loss_analyzer.py`:

```python
"""CLI for the eval loss analyzer.

Reads one or more *_games.jsonl files, writes a per-match loss summary
(JSON) + by-color / by-length / worst-loss CSVs, a cross-branch comparison
CSV, and prints a console summary. All analysis lives in eval_loss_analysis;
this module is only IO + formatting.
"""
from __future__ import annotations

import argparse
import csv
import glob as globmod
import json
import sys
from pathlib import Path

from scripts.GPU.alphazero.eval_loss_analysis import (
    LENGTH_BUCKETS_DEFAULT, analyze_match, combine_branch_summaries,
    resolve_checkpoints, sample_worst_losses, validate_rows,
)
from scripts.GPU.alphazero.eval_runner import short_id


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Explain how checkpoint A loses to B from *_games.jsonl.")
    p.add_argument("--games-jsonl", action="append", default=[], metavar="PATH",
                   help="input games jsonl (repeatable)")
    p.add_argument("--glob", default=None, metavar="PATTERN",
                   help="glob for input games jsonl files")
    p.add_argument("--output-dir", default=Path("logs/eval/loss_analysis"), type=Path)
    p.add_argument("--a-checkpoint", default=None)
    p.add_argument("--b-checkpoint", default=None)
    p.add_argument("--length-buckets", default=None,
                   help="comma-separated upper-inclusive edges, e.g. 40,60,80,120,279,280")
    p.add_argument("--worst-losses", type=int, default=50)
    return p.parse_args(argv)


def load_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_sibling_summary(games_path):
    sib = Path(str(games_path).replace("_games.jsonl", ".json"))
    if sib != Path(games_path) and sib.exists():
        try:
            return json.loads(sib.read_text())
        except (OSError, json.JSONDecodeError):
            return None
    return None


def stem_of(games_path):
    name = Path(games_path).name
    if name.endswith("_games.jsonl"):
        return name[:-len("_games.jsonl")]
    return Path(games_path).stem


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def resolve_inputs(args):
    paths = list(args.games_jsonl)
    if args.glob:
        paths += sorted(globmod.glob(args.glob))
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _weighted_rate(by_len, labels):
    subs = [b for b in by_len if b["length_bucket"] in labels]
    g = sum(b["games"] for b in subs)
    if not g:
        return None
    return sum(b["a_score_rate"] * b["games"] for b in subs) / g


def loss_shape(s):
    overall = s["a_score_rate"]
    by_len = s["by_length"]
    signals = []
    short = _weighted_rate(by_len, {"<=40", "41-60"})
    if short is not None and short < overall - 0.03:
        signals.append("short/opening games")
    long = _weighted_rate(by_len, {"81-120", "121-279"})
    if long is not None and long < overall - 0.03:
        signals.append("long/endgame games")
    if s["color_gap"] is not None and abs(s["color_gap"]) >= 0.05:
        weaker = "red" if s["color_gap"] < 0 else "black"
        signals.append(f"as {weaker}")
    if s["termination"]["state_cap_rate"] >= 0.05:
        signals.append("state-cap tail")
    if not signals:
        return "A is losing broadly, no strong length/color/termination concentration."
    return "A is losing primarily in " + ", ".join(signals) + "."


def print_console_summary(s):
    print("=" * 60)
    print(f"LOSS ANALYSIS: {s['match']}")
    print("=" * 60)
    print(f"Games: {s['games']}")
    print(f"A score: {s['a_score_rate']:.4f}")
    print(f"Elo: {s['elo']:.1f} [{s['elo_ci95'][0]:.1f}, {s['elo_ci95'][1]:.1f}]")
    print(f"Verdict: {s['verdict']}")
    print("By A color:")
    for c in s["by_color"]:
        rate = c["a_score_rate"]
        rate_s = "n/a" if rate is None else f"{rate:.4f}"
        print(f"  A as {c['a_color']:<5}: {rate_s} over {c['games']} games")
    if s["color_gap"] is not None:
        print(f"  Gap: {s['color_gap']:+.4f}")
    print("By length:")
    for b in s["by_length"]:
        print(f"  {b['length_bucket']:<9}: {b['a_score_rate']:.4f} over {b['games']} games")
    t = s["termination"]
    print(f"State caps: {t['state_cap']} / {s['games']} ({t['state_cap_rate']:.1%})")
    print(f"Board full: {t['board_full']} / {s['games']} ({t['board_full_rate']:.1%})")
    print("Likely loss shape:")
    print(f"  {loss_shape(s)}")


def main(argv=None):
    args = parse_args(argv)
    inputs = resolve_inputs(args)
    if not inputs:
        print("error: no input files (use --games-jsonl and/or --glob)", file=sys.stderr)
        return 2
    buckets = (tuple(int(x) for x in args.length_buckets.split(","))
               if args.length_buckets else LENGTH_BUCKETS_DEFAULT)
    out_dir = args.output_dir
    summaries = []
    for path in inputs:
        rows = load_jsonl(path)
        stem = stem_of(path)
        if not rows:
            print(f"skip {stem}: empty file")
            continue
        sidecar = load_sibling_summary(path)
        a, b = resolve_checkpoints(rows, rows[0]["pairing_id"],
                                   args.a_checkpoint, args.b_checkpoint, sidecar)
        if a == b:
            print(f"skip {stem}: self-match ({short_id(a)})")
            continue
        validate_rows(rows, a, b)
        summary = analyze_match(rows, a, b, match=stem,
                                pairing_id=rows[0]["pairing_id"],
                                length_buckets=buckets)
        worst = [{"match": stem, **w}
                 for w in sample_worst_losses(rows, a, b, args.worst_losses)]
        write_json(out_dir / f"{stem}_loss_summary.json", summary)
        write_csv(out_dir / f"{stem}_by_color.csv",
                  [{"match": stem, **c} for c in summary["by_color"]])
        write_csv(out_dir / f"{stem}_by_length.csv",
                  [{"match": stem, **c} for c in summary["by_length"]])
        write_csv(out_dir / f"{stem}_worst_losses.csv", worst)
        print_console_summary(summary)
        summaries.append(summary)
    if summaries:
        write_csv(out_dir / "combined_branch_comparison.csv",
                  combine_branch_summaries(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_analyzer_cli.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_loss_analyzer.py tests/test_eval_loss_analyzer_cli.py
git commit -m "feat(eval): loss-analyzer CLI + console summary"
```

---

## Task 8: Full-suite check + real-data verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full default test suite**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_analysis.py tests/test_eval_loss_analyzer_cli.py -v`
Expected: PASS (all tests). Then a broader check to confirm nothing regressed:
Run: `.venv/bin/python -m pytest -q`
Expected: no NEW failures versus the pre-existing baseline (the repo has a known set of pre-existing default-suite items; compare against `git stash`-clean baseline only if in doubt).

- [ ] **Step 2: Run against the three real branches vs 0379**

Run:
```bash
.venv/bin/python -m scripts.GPU.alphazero.eval_loss_analyzer \
  --glob "logs/eval/*0399_vs_0379*_games.jsonl" \
  --output-dir logs/eval/loss_analysis
```
Expected: console summaries for `control0399_vs_0379_800g_w4`,
`eps035_0399_vs_0379_800g_w4`, `lr0003_eps035_0399_vs_0379_800g_w4`; files written
under `logs/eval/loss_analysis/`.

- [ ] **Step 3: Sanity-check the numbers against the existing sidecar summary**

Run:
```bash
.venv/bin/python -c "
import json
s = json.load(open('logs/eval/loss_analysis/lr0003_eps035_0399_vs_0379_800g_w4_loss_summary.json'))
ref = json.load(open('logs/eval/lr0003_eps035_0399_vs_0379_800g_w4.json'))
print('a_score_rate', s['a_score_rate'], 'vs ref', ref['a_score_rate'])
print('elo', round(s['elo'],2), 'vs ref', round(ref['elo_estimate'],2))
print('a_wins/b_wins', s['a_wins'], s['b_wins'], 'vs ref', ref['a_wins'], ref['b_wins'])
print('state_caps', s['termination']['state_cap'], 'vs ref', ref['state_caps'])
assert abs(s['a_score_rate'] - ref['a_score_rate']) < 1e-9
assert s['a_wins'] == ref['a_wins'] and s['b_wins'] == ref['b_wins']
assert s['termination']['state_cap'] == ref['state_caps']
print('OK: matches eval_summary')
"
```
Expected: `OK: matches eval_summary` — the analyzer's overall numbers must equal the
already-computed `eval_summary` output for the same file (same scoring policy).

- [ ] **Step 4: Inspect the combined comparison**

Run: `column -t -s, logs/eval/loss_analysis/combined_branch_comparison.csv`
Expected: three branches ranked descending by `a_score_rate`, answering which of
control / eps035 / lr0003 is least-bad versus anchor 0379.

- [ ] **Step 5: Commit any output worth keeping (optional)**

The `logs/eval/loss_analysis/` outputs are analysis artifacts. Commit them only if the
project tracks eval artifacts in git; otherwise leave them untracked. No code change in
this task.

---

## Self-Review

**Spec coverage:**
- Pure module + thin CLI split → Tasks 1–6 (module), 7 (CLI). ✓
- A/B keyed off `winner_checkpoint`, color off `red/black_checkpoint` → Task 1 (`score_for_checkpoint`, `a_color`), tested in Task 3. ✓
- Draw policy 0.5/0.5 → enforced in `validate_rows` (Task 1), used in `_ab_stats`/`summarize_overall`. ✓
- A/B resolution order (override → sidecar `checkpoint_a/b` → pairing-id `short_id`) → Task 2. ✓
- Length buckets `(40,60,80,120,279,280)`, 279/280 split isolates caps → Task 3. ✓
- Self-match by resolved `a==b` (not pairing_id string) → Task 7 CLI + `test_cli_skips_self_match`. ✓
- Fail-loud on `unknown_error` and mixed JSONL → Task 1 tests. ✓
- `loss_summary.json` with `match`/`pairing_id` + enriched `termination` (counts + rates + draws) → Task 4. ✓
- `worst_losses.csv` with `loss_bucket` (short/long/draw_cap) → Task 5. ✓
- `combined_branch_comparison.csv` descending by `a_score_rate` → Task 6. ✓
- Lean output set (no by_reason.csv) → Task 7 writes exactly the four per-match files + combined. ✓
- Console "Likely loss shape" heuristics → Task 7 (`loss_shape`). ✓
- Reuse `eval_elo` stats verbatim → Task 4 imports + Task 8 Step 3 cross-checks against `eval_summary`. ✓

**Placeholder scan:** none — every step has complete code/commands.

**Type/name consistency:** `_ab_stats`, `_bucket_name`, `_length_bucket_label`,
`score_for_checkpoint`, `a_color`, `summarize_by_color/by_length/overall`,
`sample_worst_losses`, `analyze_match`, `combine_branch_summaries`,
`resolve_checkpoints` used consistently across module and CLI. CLI `loss_shape`
reads `by_length` entries by the same `length_bucket` labels the module emits.

**Heuristic note:** `loss_shape` long-game signal uses `81-120` + `121-279` (decisive
long games) and deliberately excludes the `280` cap bucket, which is covered separately
by the `state_cap_rate` signal — consistent with the spec's draw/cap handling.
