# Eval Replay Loss Analyzer (V2 Phase B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the replay-aware loss analyzer that explains *why* checkpoint A (eps035 0399) loses to B (staged 0379) as black in 41–80-move decisive games — value drop vs visit diffusion vs low-visit-rank moves, and when.

**Architecture:** Pure analysis module (`eval_loss_replay_analysis.py`: game rows + replay sidecar dicts in, feature dicts / table rows / verdict out; no IO, no MLX) + thin CLI (`eval_loss_replay_analyzer.py`: input resolution, sidecar loading, six artifacts, console verdict). Mirrors the V1 pattern and reuses V1 helpers by import; V1 files are not modified.

**Tech Stack:** Python 3.14 stdlib only (`statistics`, `dataclasses`, `json`, `argparse`, `csv` via V1's writers). Tests: pytest (repo defaults from `pytest.ini`). Run everything with `.venv/bin/python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-06-12-eval-replay-analyzer-design.md` — read it first; it locks thresholds, rule precedence, artifact shapes, and the opening-temperature exclusion rule.

**Two deliberate naming notes (vs the spec's sketch):**
- The spec's `a_ply_series(replay, color)` is implemented as `side_plies(replay, color)` — it serves both A and B sides.
- Test files follow the repo convention: `tests/test_eval_loss_replay_analysis.py` (pure) and `tests/test_eval_loss_replay_analyzer_cli.py` (CLI; matches existing `test_eval_loss_analyzer_cli.py`).

**Key domain facts the engineer must know:**
- `root_value` in a replay ply is negamax — always from the perspective of the player about to move. A's trajectory uses A's own plies; B's series is reported in B's own perspective, never merged/sign-flipped.
- Eval games temperature-sample the first 20 plies (`opening_temp_plies`). A `selected_visit_rank` of 4 at ply 2 is sampling, not low confidence. Confidence/diffusion features use post-opening plies only; value features use all A plies.
- `replay_path` in a games.jsonl row is relative to the process CWD (repo root) for real data; tests use absolute tmp paths — both work because the CLI opens the string as-is.

---

## File map

- Create: `scripts/GPU/alphazero/eval_loss_replay_analysis.py` — pure analysis (Tasks 1–12)
- Create: `scripts/GPU/alphazero/eval_loss_replay_analyzer.py` — CLI (Tasks 13–14)
- Create: `tests/eval_replay_fixtures.py` — shared synthetic row+replay builders (Task 1)
- Create: `tests/test_eval_loss_replay_analysis.py` — pure tests (Tasks 1–12)
- Create: `tests/test_eval_loss_replay_analyzer_cli.py` — CLI tests (Tasks 13–14)
- Read-only dependencies: `scripts/GPU/alphazero/eval_loss_analysis.py` (V1 pure: `a_color`, `score_for_checkpoint`, `validate_rows`, `resolve_checkpoints`), `scripts/GPU/alphazero/eval_loss_analyzer.py` (V1 CLI: `load_jsonl`, `load_sibling_summary`, `stem_of`, `resolve_inputs`, `write_json`, `write_csv`), `scripts/GPU/alphazero/eval_elo.py` (`score_rate`), `scripts/GPU/alphazero/eval_runner.py` (`short_id`).

---

### Task 1: Module skeleton, Thresholds, side_plies + shared test fixtures

**Files:**
- Create: `scripts/GPU/alphazero/eval_loss_replay_analysis.py`
- Create: `tests/eval_replay_fixtures.py`
- Create: `tests/test_eval_loss_replay_analysis.py`

- [ ] **Step 1: Write the shared fixtures module**

`tests/eval_replay_fixtures.py`:

```python
"""Shared synthetic Phase A capture builders for V2 Phase B tests. No MLX.

make_game builds a matched (games.jsonl row, replay sidecar dict) pair with
correct red/black alternation, A seated by color, and consistent identity
fields, so validate_rows (V1) and validate_replay (V2) both accept it.
"""

A = "ckpts/model_iter_0399.safetensors"
B = "ckpts/model_iter_0379.safetensors"
PAIRING = "0399_vs_0379"


def make_ply(ply, player, root_value, *, row=None, col=None, top1=0.5, rank=1,
             visits=200, total=400, n_legal=100):
    return {
        "ply": ply, "player": player,
        "row": ply if row is None else row, "col": ply if col is None else col,
        "root_value": root_value, "root_top1_share": top1,
        "selected_visit_rank": rank, "selected_visit_count": visits,
        "root_total_visits": total, "n_legal": n_legal,
    }


def make_game(game_idx, *, a_is_black=True, a_wins=False, n_moves=50,
              a_values=None, b_values=None, a_top1=0.5, a_rank=1,
              reason="win", task_id=None, replay_dir="replays"):
    """Build a (row, replay) pair.

    a_values / b_values: per-side root_value sequences (must match that
    side's ply count: n_moves // 2 plus the odd ply for red); default flat
    0.0. a_top1 / a_rank: scalar applied to every A ply, or a per-A-ply list.
    reason="state_cap" builds a draw (winner None, 0.5/0.5 scores).
    """
    red_ck, black_ck = (B, A) if a_is_black else (A, B)
    a_clr = "black" if a_is_black else "red"
    if reason == "win":
        winner = a_clr if a_wins else ("red" if a_is_black else "black")
        winner_ck = A if a_wins else B
        rs, bs = (1.0, 0.0) if winner == "red" else (0.0, 1.0)
    else:
        winner, winner_ck, rs, bs = None, None, 0.5, 0.5
    moves, ai, bi = [], 0, 0
    for ply in range(n_moves):
        player = "red" if ply % 2 == 0 else "black"
        if player == a_clr:
            v = a_values[ai] if a_values is not None else 0.0
            t1 = a_top1[ai] if isinstance(a_top1, (list, tuple)) else a_top1
            rk = a_rank[ai] if isinstance(a_rank, (list, tuple)) else a_rank
            moves.append(make_ply(ply, player, v, top1=t1, rank=rk))
            ai += 1
        else:
            v = b_values[bi] if b_values is not None else 0.0
            moves.append(make_ply(ply, player, v))
            bi += 1
    row = {
        "task_id": game_idx if task_id is None else task_id,
        "pairing_id": PAIRING, "game_idx": game_idx,
        "red_checkpoint": red_ck, "black_checkpoint": black_ck,
        "winner": winner, "winner_checkpoint": winner_ck, "reason": reason,
        "n_moves": n_moves, "red_score": rs, "black_score": bs,
        "replay_path": f"{replay_dir}/game_{game_idx:06d}.json",
    }
    replay = {
        "schema_version": 1, "pairing_id": PAIRING, "game_idx": game_idx,
        "task_id": row["task_id"], "seed": 1000 + game_idx, "board_size": 24,
        "red_checkpoint": red_ck, "black_checkpoint": black_ck,
        "winner": winner, "winner_checkpoint": winner_ck, "reason": reason,
        "n_moves": n_moves, "moves": moves,
    }
    return row, replay
```

- [ ] **Step 2: Write the failing tests**

`tests/test_eval_loss_replay_analysis.py`:

```python
import pytest

from scripts.GPU.alphazero.eval_loss_replay_analysis import (
    Thresholds, side_plies,
)
from tests.eval_replay_fixtures import A, B, make_game


def test_thresholds_defaults_match_spec():
    th = Thresholds()
    assert th.bad_value == -0.25
    assert th.lost_value == -0.50
    assert th.sharp_drop == 0.40
    assert th.low_top1_share == 0.10
    assert th.low_visit_rank == 5
    assert th.opening_plies == 20


def test_side_plies_filters_one_side_in_order():
    _row, replay = make_game(0, a_is_black=True, n_moves=6)
    black = side_plies(replay, "black")
    red = side_plies(replay, "red")
    assert [m["ply"] for m in black] == [1, 3, 5]
    assert [m["ply"] for m in red] == [0, 2, 4]
    assert all(m["player"] == "black" for m in black)


def test_fixture_seats_a_by_color():
    row_b, _ = make_game(0, a_is_black=True)
    assert row_b["black_checkpoint"] == A and row_b["red_checkpoint"] == B
    row_r, _ = make_game(1, a_is_black=False, a_wins=True)
    assert row_r["red_checkpoint"] == A and row_r["winner"] == "red"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py`
Expected: FAIL/ERROR — `ModuleNotFoundError: ... eval_loss_replay_analysis`

- [ ] **Step 4: Write the module skeleton**

`scripts/GPU/alphazero/eval_loss_replay_analysis.py`:

```python
"""Pure replay-aware loss analysis (V2 Phase B) over Phase A capture data.

No IO, no MLX: game rows + replay sidecar dicts in, feature dicts / table
rows / verdict out. The V1 game-level analyzer is untouched; game-row
semantics (scoring, color, validation) live in eval_loss_analysis.

Value-sign convention (confirmed against mcts.py): root_value is negamax,
always from the perspective of the player about to move. A's trajectory uses
A's own plies; B's series is reported in B's own perspective, never merged.

Opening-temperature rule: eval games temperature-sample the first
opening_plies plies, so selected_visit_rank / root_top1_share there reflect
sampling, not confidence. Confidence/diffusion features and rules use
post-opening plies only; value features use all A plies.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean, median, variance

from .eval_elo import score_rate

REPLAY_SCHEMA_VERSION = 1

# Classification constants (spec-locked). The five operator-facing thresholds
# live in Thresholds; these pin down the spec's "many"/"multiple" words.
HEALTHY_START = -0.10        # gradual_decay: the game started healthy
DECAYED_FINAL = -0.40        # gradual_decay: the game ended decayed
DIFFUSE_MEAN_TOP1 = 0.15     # search_diffusion: mean post-opening top1 share
DIFFUSE_PLY_FRACTION = 0.25  # search_diffusion: share of diffuse post plies
LOW_RANK_MEDIAN = 3          # low_visit_selection: median post rank
LOW_RANK_PLY_COUNT = 3       # low_visit_selection: count of low-rank plies
B_ONSET_LOW = 0.25           # B win-onset crossings (B's own perspective)
B_ONSET_HIGH = 0.50
PRIMARY_SHARE = 0.35         # verdict: bar for a primary failure mode
SECONDARY_SHARE = 0.20       # verdict: bar for a secondary signal
MIN_WIN_COHORT = 5           # below this, effect sizes -> insufficient_contrast

COLLAPSE_PRECEDENCE = (
    ("already_bad", "flag_already_bad"),
    ("sharp_value_drop", "flag_sharp"),
    ("gradual_decay", "flag_gradual"),
    ("search_diffusion", "flag_diffusion"),
    ("low_visit_selection", "flag_low_visit"),
)

FAILURE_MODE_GROUPS = {
    "value-drop": ("sharp_value_drop", "gradual_decay"),
    "already-losing": ("already_bad",),
    "diffusion": ("search_diffusion",),
    "low-visit-selection": ("low_visit_selection",),
}

PHASES = ("opening", "early_midgame", "midgame", "late_midgame", "pre_terminal")
MIDGAME_PHASES = PHASES[1:]

CROSSING_KEYS = ("first_a_value_below_0", "first_a_value_below_bad",
                 "first_a_value_below_lost")

REQUIRED_PLY_KEYS = {
    "ply", "player", "row", "col", "root_value", "root_top1_share",
    "selected_visit_rank", "selected_visit_count", "root_total_visits",
    "n_legal",
}

EFFECT_METRICS = ("final_a_value", "largest_a_value_drop", "initial_a_value",
                  "mean_top1_share_post", "median_selected_visit_rank_post")
EFFECT_FORMULA = (
    "cohens_d = (loss_mean - win_mean) / pooled_std(ddof=1); negative d on "
    "value metrics = lower in losses; positive d on visit rank = higher rank "
    "(less confident) in losses")

OPENING_SAMPLING_NOTE = (
    "Plies before opening_plies are temperature-sampled: selected_visit_rank "
    "and root_top1_share there reflect sampling, not confidence. Confidence/"
    "diffusion features and rules use post-opening plies only.")


@dataclass(frozen=True)
class Thresholds:
    bad_value: float = -0.25
    lost_value: float = -0.50
    sharp_drop: float = 0.40
    low_top1_share: float = 0.10
    low_visit_rank: int = 5
    opening_plies: int = 20


def _mean(vals):
    return mean(vals) if vals else None


def _median(vals):
    return median(vals) if vals else None


def side_plies(replay, color):
    """Per-ply records for one side, in game order (spec: a_ply_series)."""
    return [m for m in replay["moves"] if m["player"] == color]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/eval_loss_replay_analysis.py tests/eval_replay_fixtures.py tests/test_eval_loss_replay_analysis.py
git commit -m "feat(eval): V2 Phase B skeleton — thresholds, side_plies, replay fixtures"
```

---

### Task 2: validate_replay

**Files:**
- Modify: `scripts/GPU/alphazero/eval_loss_replay_analysis.py` (append after `side_plies`)
- Modify: `tests/test_eval_loss_replay_analysis.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_loss_replay_analysis.py` (extend the existing import from `eval_loss_replay_analysis` with `validate_replay`):

```python
def test_validate_replay_accepts_consistent_pair():
    row, replay = make_game(3, n_moves=8)
    validate_replay(row, replay)  # no raise


def test_validate_replay_rejects_wrong_schema_version():
    row, replay = make_game(0)
    replay["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        validate_replay(row, replay)


def test_validate_replay_rejects_identity_mismatch():
    row, replay = make_game(0)
    replay["winner"] = "red" if replay["winner"] == "black" else "black"
    with pytest.raises(ValueError, match="winner"):
        validate_replay(row, replay)


def test_validate_replay_rejects_move_count_mismatch():
    row, replay = make_game(0, n_moves=10)
    replay["moves"] = replay["moves"][:-1]
    replay["n_moves"] = 10  # identity still matches the row
    with pytest.raises(ValueError, match="move records"):
        validate_replay(row, replay)


def test_validate_replay_rejects_broken_alternation():
    row, replay = make_game(0, n_moves=6)
    replay["moves"][2]["player"] = "black"  # ply 2 must be red
    with pytest.raises(ValueError, match="player"):
        validate_replay(row, replay)


def test_validate_replay_rejects_bad_ply_field():
    row, replay = make_game(0, n_moves=6)
    replay["moves"][4]["ply"] = 99
    with pytest.raises(ValueError, match="ply field"):
        validate_replay(row, replay)


def test_validate_replay_rejects_missing_ply_key():
    row, replay = make_game(0, n_moves=6)
    del replay["moves"][1]["root_value"]
    with pytest.raises(ValueError, match="missing keys"):
        validate_replay(row, replay)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py -k validate_replay`
Expected: ERROR — ImportError (`validate_replay` not defined)

- [ ] **Step 3: Implement**

Append to `eval_loss_replay_analysis.py`:

```python
def validate_replay(row, replay):
    """Fail loud if a sidecar contradicts its games.jsonl row."""
    gi = row["game_idx"]
    if replay.get("schema_version") != REPLAY_SCHEMA_VERSION:
        raise ValueError(
            f"game {gi}: schema_version {replay.get('schema_version')!r} "
            f"!= {REPLAY_SCHEMA_VERSION}")
    for key in ("game_idx", "task_id", "pairing_id", "winner", "reason",
                "n_moves", "red_checkpoint", "black_checkpoint"):
        if replay.get(key) != row[key]:
            raise ValueError(
                f"game {gi}: replay {key}={replay.get(key)!r} != row {row[key]!r}")
    moves = replay["moves"]
    if len(moves) != row["n_moves"]:
        raise ValueError(
            f"game {gi}: {len(moves)} move records != n_moves {row['n_moves']}")
    for i, m in enumerate(moves):
        missing = REQUIRED_PLY_KEYS - m.keys()
        if missing:
            raise ValueError(f"game {gi} ply {i}: missing keys {sorted(missing)}")
        if m["ply"] != i:
            raise ValueError(f"game {gi} ply {i}: ply field is {m['ply']}")
        expect = "red" if i % 2 == 0 else "black"
        if m["player"] != expect:
            raise ValueError(
                f"game {gi} ply {i}: player {m['player']!r}, expected {expect!r}")
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_loss_replay_analysis.py tests/test_eval_loss_replay_analysis.py
git commit -m "feat(eval): Phase B validate_replay — fail-loud sidecar/row cross-checks"
```

---

### Task 3: Value-trajectory features

**Files:**
- Modify: `scripts/GPU/alphazero/eval_loss_replay_analysis.py`
- Modify: `tests/test_eval_loss_replay_analysis.py`

All test values are binary-exact floats (x/2^k) so equality assertions are safe; fractions use `pytest.approx`.

- [ ] **Step 1: Write the failing tests**

Append (extend imports with `value_features`):

```python
# A-as-black, n_moves=12 -> A plies at global plies 1,3,5,7,9,11 (6 A plies).
TRAJ = [0.5, 0.125, -0.125, -0.375, -0.625, -1.0]
# deltas: -0.375, -0.25, -0.25, -0.25, -0.375 (all binary-exact)


def _a_plies(values, n_moves=12):
    _row, replay = make_game(0, a_is_black=True, n_moves=n_moves, a_values=values)
    return side_plies(replay, "black")


def test_value_features_medians_mean_min():
    f = value_features(_a_plies(TRAJ), 12, Thresholds())
    assert f["initial_a_value"] == 0.125          # median(0.5, 0.125, -0.125)
    assert f["final_a_value"] == -0.625           # median(-0.375, -0.625, -1.0)
    assert f["mean_a_value"] == -0.25             # sum = -1.5 over 6
    assert f["min_a_value"] == -1.0


def test_value_features_largest_drop_with_tie_takes_earliest():
    f = value_features(_a_plies(TRAJ), 12, Thresholds())
    # ties at -0.375 (a_ply 1 and 5): earliest wins
    assert f["largest_a_value_drop"] == -0.375
    assert f["largest_drop_a_ply"] == 1
    assert f["largest_drop_ply"] == 3
    assert f["largest_drop_fraction"] == pytest.approx(3 / 11)


def test_value_features_first_crossings():
    f = value_features(_a_plies(TRAJ), 12, Thresholds())
    assert (f["first_a_value_below_0_ply"], f["first_a_value_below_0_a_ply"]) == (5, 2)
    assert f["first_a_value_below_0_fraction"] == pytest.approx(5 / 11)
    assert (f["first_a_value_below_bad_ply"], f["first_a_value_below_bad_a_ply"]) == (7, 3)
    assert (f["first_a_value_below_lost_ply"], f["first_a_value_below_lost_a_ply"]) == (9, 4)
    assert f["first_a_value_below_lost_fraction"] == pytest.approx(9 / 11)


def test_value_features_never_crossed_is_none():
    f = value_features(_a_plies([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), 12, Thresholds())
    assert f["first_a_value_below_0_ply"] is None
    assert f["first_a_value_below_lost_fraction"] is None


def test_value_features_single_ply_has_null_drop():
    _row, replay = make_game(0, a_is_black=True, n_moves=2, a_values=[-0.5])
    f = value_features(side_plies(replay, "black"), 2, Thresholds())
    assert f["largest_a_value_drop"] is None
    assert f["largest_drop_ply"] is None
    assert f["initial_a_value"] == -0.5           # median of the single value
    assert f["first_a_value_below_lost_ply"] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py -k value_features`
Expected: ImportError (`value_features` not defined)

- [ ] **Step 3: Implement**

Append to `eval_loss_replay_analysis.py`:

```python
def _crossing(plies, n_moves, pred):
    """First ply where pred(root_value) -> {ply, a_ply, fraction} or None."""
    for i, m in enumerate(plies):
        if pred(m["root_value"]):
            frac = m["ply"] / (n_moves - 1) if n_moves > 1 else 0.0
            return {"ply": m["ply"], "a_ply": i, "fraction": frac}
    return None


def value_features(a_plies, n_moves, th):
    """Value-trajectory features over ALL of A's plies (see module docstring:
    value readings are not temperature-distorted, so the opening is included
    — that is what lets initial_a_value detect already_bad games)."""
    vals = [m["root_value"] for m in a_plies]
    feats = {
        "initial_a_value": _median(vals[:3]),
        "final_a_value": _median(vals[-3:]),
        "mean_a_value": _mean(vals),
        "min_a_value": min(vals) if vals else None,
        "largest_a_value_drop": None,
        "largest_drop_ply": None,
        "largest_drop_a_ply": None,
        "largest_drop_fraction": None,
    }
    if len(vals) >= 2:
        # (delta, index) tuple-min: ties on delta resolve to the earliest ply.
        d, i = min((vals[i] - vals[i - 1], i) for i in range(1, len(vals)))
        ply = a_plies[i]["ply"]
        feats.update(
            largest_a_value_drop=d, largest_drop_ply=ply, largest_drop_a_ply=i,
            largest_drop_fraction=ply / (n_moves - 1) if n_moves > 1 else 0.0)
    for name, thresh in (("first_a_value_below_0", 0.0),
                         ("first_a_value_below_bad", th.bad_value),
                         ("first_a_value_below_lost", th.lost_value)):
        c = _crossing(a_plies, n_moves, lambda v, t=thresh: v <= t)
        feats[f"{name}_ply"] = c["ply"] if c else None
        feats[f"{name}_a_ply"] = c["a_ply"] if c else None
        feats[f"{name}_fraction"] = c["fraction"] if c else None
    return feats
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_loss_replay_analysis.py tests/test_eval_loss_replay_analysis.py
git commit -m "feat(eval): Phase B value-trajectory features (medians, signed drop, crossings)"
```

---

### Task 4: Confidence features (opening exclusion) + opening_key

**Files:**
- Modify: `scripts/GPU/alphazero/eval_loss_replay_analysis.py`
- Modify: `tests/test_eval_loss_replay_analysis.py`

- [ ] **Step 1: Write the failing tests**

Append (extend imports with `confidence_features`, `opening_key`):

```python
def test_confidence_features_post_opening_only():
    # opening_plies=4 -> A (black) post plies are global 5,7,9,11 (a_ply 2..5)
    _row, replay = make_game(
        0, a_is_black=True, n_moves=12, a_values=TRAJ,
        a_top1=[0.5, 0.5, 0.08, 0.12, 0.3, 0.05],
        a_rank=[4, 1, 6, 2, 1, 7])
    th = Thresholds(opening_plies=4)
    f = confidence_features(side_plies(replay, "black"), th)
    assert f["n_a_plies"] == 6
    assert f["n_a_plies_post"] == 4
    assert f["mean_top1_share_post"] == pytest.approx((0.08 + 0.12 + 0.3 + 0.05) / 4)
    assert f["min_top1_share_post"] == 0.05
    assert f["median_selected_visit_rank_post"] == 4.0   # median(6, 2, 1, 7)
    assert f["max_selected_visit_rank_post"] == 7
    assert f["low_confidence_ply_count"] == 2            # ranks 6 and 7 >= 5
    assert f["diffuse_ply_fraction"] == 0.5              # 0.08, 0.05 <= 0.10
    assert f["mean_selected_visit_share_post"] == 0.5    # 200/400 everywhere
    assert f["mean_n_legal"] == 100


def test_confidence_features_all_opening_yields_nulls():
    _row, replay = make_game(0, a_is_black=True, n_moves=6)
    th = Thresholds(opening_plies=20)  # whole game inside the opening window
    f = confidence_features(side_plies(replay, "black"), th)
    assert f["n_a_plies_post"] == 0
    assert f["mean_top1_share_post"] is None
    assert f["median_selected_visit_rank_post"] is None
    assert f["low_confidence_ply_count"] is None
    assert f["diffuse_ply_fraction"] is None
    assert f["mean_n_legal"] == 100                      # all-plies metric survives


def test_opening_key_first_k_plies():
    _row, replay = make_game(0, n_moves=6)
    # fixture rows/cols default to the ply number
    assert opening_key(replay, 4) == "r0c0|r1c1|r2c2|r3c3"
    assert opening_key(replay, 2) == "r0c0|r1c1"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py -k "confidence or opening_key"`
Expected: ImportError

- [ ] **Step 3: Implement**

Append:

```python
def confidence_features(a_plies, th):
    """Confidence/diffusion features over POST-OPENING A plies only (the
    opening is temperature-sampled — see OPENING_SAMPLING_NOTE). All-null
    when the game has no post-opening A plies."""
    post = [m for m in a_plies if m["ply"] >= th.opening_plies]
    feats = {
        "n_a_plies": len(a_plies),
        "n_a_plies_post": len(post),
        "mean_n_legal": _mean([m["n_legal"] for m in a_plies]),
        "mean_top1_share_post": None,
        "min_top1_share_post": None,
        "median_selected_visit_rank_post": None,
        "max_selected_visit_rank_post": None,
        "mean_selected_visit_share_post": None,
        "low_confidence_ply_count": None,
        "diffuse_ply_fraction": None,
    }
    if post:
        shares = [m["root_top1_share"] for m in post]
        ranks = [m["selected_visit_rank"] for m in post]
        feats.update(
            mean_top1_share_post=mean(shares),
            min_top1_share_post=min(shares),
            median_selected_visit_rank_post=median(ranks),
            max_selected_visit_rank_post=max(ranks),
            mean_selected_visit_share_post=mean(
                [m["selected_visit_count"] / m["root_total_visits"] for m in post]),
            low_confidence_ply_count=sum(r >= th.low_visit_rank for r in ranks),
            diffuse_ply_fraction=(
                sum(s <= th.low_top1_share for s in shares) / len(post)),
        )
    return feats


def opening_key(replay, key_plies):
    """First key_plies moves (both players) as a compact cluster key."""
    return "|".join(f"r{m['row']}c{m['col']}"
                    for m in replay["moves"][:key_plies])
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_loss_replay_analysis.py tests/test_eval_loss_replay_analysis.py
git commit -m "feat(eval): Phase B confidence features (post-opening only) + opening_key"
```

---

### Task 5: classify_collapse — rules, boundaries, precedence

**Files:**
- Modify: `scripts/GPU/alphazero/eval_loss_replay_analysis.py`
- Modify: `tests/test_eval_loss_replay_analysis.py`

`classify_collapse` consumes a feature dict directly, so tests construct minimal dicts — no replays needed.

- [ ] **Step 1: Write the failing tests**

Append (extend imports with `classify_collapse`):

```python
def _feats(**over):
    base = {
        "initial_a_value": 0.0, "final_a_value": 0.0,
        "largest_a_value_drop": -0.1,
        "mean_top1_share_post": 0.5, "diffuse_ply_fraction": 0.0,
        "median_selected_visit_rank_post": 1, "low_confidence_ply_count": 0,
    }
    base.update(over)
    return base


def test_classify_already_bad_at_boundary():
    label, flags = classify_collapse(_feats(initial_a_value=-0.25), Thresholds())
    assert label == "already_bad" and flags["flag_already_bad"]


def test_classify_sharp_drop_at_boundary():
    label, _ = classify_collapse(_feats(largest_a_value_drop=-0.40), Thresholds())
    assert label == "sharp_value_drop"


def test_classify_gradual_decay_requires_healthy_start_and_no_cliff():
    label, _ = classify_collapse(
        _feats(initial_a_value=0.0, final_a_value=-0.40,
               largest_a_value_drop=-0.39), Thresholds())
    assert label == "gradual_decay"


def test_gradual_flag_suppressed_by_sharp():
    label, flags = classify_collapse(
        _feats(initial_a_value=0.0, final_a_value=-0.5,
               largest_a_value_drop=-0.45), Thresholds())
    assert label == "sharp_value_drop"
    assert flags["flag_gradual"] is False        # spec: "and not sharp"


def test_classify_diffusion_mean_or_fraction():
    label, _ = classify_collapse(_feats(mean_top1_share_post=0.15), Thresholds())
    assert label == "search_diffusion"
    label, _ = classify_collapse(_feats(diffuse_ply_fraction=0.25), Thresholds())
    assert label == "search_diffusion"


def test_classify_low_visit_median_or_count():
    label, _ = classify_collapse(
        _feats(median_selected_visit_rank_post=3), Thresholds())
    assert label == "low_visit_selection"
    label, _ = classify_collapse(
        _feats(low_confidence_ply_count=3), Thresholds())
    assert label == "low_visit_selection"


def test_classify_precedence_already_bad_beats_sharp_but_keeps_flag():
    label, flags = classify_collapse(
        _feats(initial_a_value=-0.3, largest_a_value_drop=-0.5), Thresholds())
    assert label == "already_bad"
    assert flags["flag_sharp"] is True           # multi-signal stays visible


def test_classify_no_clear_signal():
    label, flags = classify_collapse(_feats(), Thresholds())
    assert label == "no_clear_signal"
    assert not any(flags.values())


def test_classify_null_post_features_disable_those_rules():
    label, flags = classify_collapse(
        _feats(mean_top1_share_post=None, diffuse_ply_fraction=None,
               median_selected_visit_rank_post=None,
               low_confidence_ply_count=None), Thresholds())
    assert label == "no_clear_signal"
    assert flags["flag_diffusion"] is False and flags["flag_low_visit"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py -k classify`
Expected: ImportError

- [ ] **Step 3: Implement**

Append:

```python
def classify_collapse(f, th):
    """(label, flags) for one game's features. One label via the documented
    precedence; every rule's flag is returned so multi-signal games stay
    visible in the CSVs. Rules with null inputs do not fire."""
    init, fin = f["initial_a_value"], f["final_a_value"]
    drop = f["largest_a_value_drop"]
    sharp = drop is not None and drop <= -th.sharp_drop
    flags = {
        "flag_already_bad": init is not None and init <= th.bad_value,
        "flag_sharp": sharp,
        "flag_gradual": (init is not None and fin is not None
                         and init > HEALTHY_START and fin <= DECAYED_FINAL
                         and not sharp),
        "flag_diffusion": (
            f["mean_top1_share_post"] is not None
            and (f["mean_top1_share_post"] <= DIFFUSE_MEAN_TOP1
                 or f["diffuse_ply_fraction"] >= DIFFUSE_PLY_FRACTION)),
        "flag_low_visit": (
            f["median_selected_visit_rank_post"] is not None
            and (f["median_selected_visit_rank_post"] >= LOW_RANK_MEDIAN
                 or f["low_confidence_ply_count"] >= LOW_RANK_PLY_COUNT)),
    }
    label = next((lab for lab, flag in COLLAPSE_PRECEDENCE if flags[flag]),
                 "no_clear_signal")
    return label, flags
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_loss_replay_analysis.py tests/test_eval_loss_replay_analysis.py
git commit -m "feat(eval): Phase B collapse classification — rules, boundaries, precedence"
```

---

### Task 6: game_features + b_side_features

**Files:**
- Modify: `scripts/GPU/alphazero/eval_loss_replay_analysis.py`
- Modify: `tests/test_eval_loss_replay_analysis.py`

- [ ] **Step 1: Write the failing tests**

Append (extend imports with `game_features`, `b_side_features`):

```python
def test_game_features_merges_identity_value_confidence():
    row, replay = make_game(7, a_is_black=True, n_moves=12, a_values=TRAJ)
    f = game_features(row, replay, "black", Thresholds(opening_plies=4),
                      key_plies=2)
    assert f["game_idx"] == 7 and f["a_color"] == "black"
    assert f["replay_path"] == row["replay_path"]
    assert f["opening_key"] == "r0c0|r1c1"
    assert f["initial_a_value"] == 0.125          # value features present
    assert f["n_a_plies_post"] == 4               # confidence features present


def test_b_side_features_onsets_and_saw_it_first():
    # B is red: B plies at global 0,2,4,6,8,10. B's values rise to a win.
    _row, replay = make_game(
        0, a_is_black=True, n_moves=12,
        b_values=[0.0, 0.125, 0.25, 0.5, 0.75, 1.0])
    th = Thresholds(opening_plies=4)
    f = b_side_features(replay, "red", th, a_first_below_lost_fraction=9 / 11)
    assert f["b_first_value_above_025_ply"] == 4         # 0.25 >= 0.25
    assert f["b_first_value_above_050_ply"] == 6         # 0.5 >= 0.50
    assert f["b_first_value_above_050_fraction"] == pytest.approx(6 / 11)
    assert f["b_saw_it_first"] is True                   # 6/11 < 9/11
    assert f["b_mean_value"] == pytest.approx((0.0 + 0.125 + 0.25 + 0.5 + 0.75 + 1.0) / 6)
    assert f["b_mean_top1_share_post"] == 0.5            # fixture default
    assert f["b_median_visit_rank_post"] == 1


def test_b_saw_it_first_false_when_either_onset_missing():
    _row, replay = make_game(0, a_is_black=True, n_moves=12)  # flat 0.0: no onset
    f = b_side_features(replay, "red", Thresholds(), a_first_below_lost_fraction=0.5)
    assert f["b_first_value_above_050_ply"] is None
    assert f["b_saw_it_first"] is False
    f2 = b_side_features(replay, "red", Thresholds(), a_first_below_lost_fraction=None)
    assert f2["b_saw_it_first"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py -k "game_features or b_side or saw_it"`
Expected: ImportError

- [ ] **Step 3: Implement**

Append:

```python
def game_features(row, replay, a_clr, th, key_plies=4):
    """One flat per-game feature dict: identity + value + confidence."""
    a_plies = side_plies(replay, a_clr)
    feats = {
        "game_idx": row["game_idx"], "task_id": row["task_id"],
        "replay_path": row.get("replay_path"), "a_color": a_clr,
        "winner": row["winner"], "n_moves": row["n_moves"],
        "opening_key": opening_key(replay, key_plies),
    }
    feats.update(value_features(a_plies, row["n_moves"], th))
    feats.update(confidence_features(a_plies, th))
    return feats


def b_side_features(replay, b_clr, th, a_first_below_lost_fraction):
    """B's series inside one (lost) game, in B's OWN perspective — kept
    separate from A's series, never sign-flipped or merged."""
    n_moves = replay["n_moves"]
    b_plies = side_plies(replay, b_clr)
    post = [m for m in b_plies if m["ply"] >= th.opening_plies]
    feats = {
        "b_mean_value": _mean([m["root_value"] for m in b_plies]),
        "b_mean_top1_share_post": _mean([m["root_top1_share"] for m in post]),
        "b_median_visit_rank_post": _median(
            [m["selected_visit_rank"] for m in post]),
    }
    for name, t in (("b_first_value_above_025", B_ONSET_LOW),
                    ("b_first_value_above_050", B_ONSET_HIGH)):
        c = _crossing(b_plies, n_moves, lambda v, t=t: v >= t)
        feats[f"{name}_ply"] = c["ply"] if c else None
        feats[f"{name}_fraction"] = c["fraction"] if c else None
    bf = feats["b_first_value_above_050_fraction"]
    feats["b_saw_it_first"] = (bf is not None
                               and a_first_below_lost_fraction is not None
                               and bf < a_first_below_lost_fraction)
    return feats
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_loss_replay_analysis.py tests/test_eval_loss_replay_analysis.py
git commit -m "feat(eval): Phase B game_features + B-side win-onset features"
```

---

### Task 7: cohort_comparison_row + phase buckets

**Files:**
- Modify: `scripts/GPU/alphazero/eval_loss_replay_analysis.py`
- Modify: `tests/test_eval_loss_replay_analysis.py`

- [ ] **Step 1: Write the failing tests**

Append (extend imports with `cohort_comparison_row`, `phase_of`, `phase_bucket_rows`):

```python
def test_cohort_comparison_row_pools_plies_across_games():
    th = Thresholds(opening_plies=4)
    games = []
    for i, vals in enumerate(([0.5, 0.25, -0.25, -0.5, -0.75, -1.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
        _row, replay = make_game(i, a_is_black=True, n_moves=12, a_values=vals)
        games.append(side_plies(replay, "black"))
    r = cohort_comparison_row("loss", games, th.opening_plies)
    assert r["cohort"] == "loss" and r["games"] == 2 and r["plies"] == 12
    assert r["mean_root_value"] == pytest.approx(-1.75 / 12)
    assert r["mean_n_legal"] == 100
    # post pool: 4 post plies per game = 8
    assert r["mean_top1_share_post"] == 0.5


def test_phase_of_boundaries():
    # opening_plies=20, n_moves=80: post-opening span is plies 20..79
    assert phase_of(19, 80, 20) == "opening"
    assert phase_of(20, 80, 20) == "early_midgame"
    assert phase_of(34, 80, 20) == "early_midgame"   # f = 14/60 < 0.25
    assert phase_of(35, 80, 20) == "midgame"         # f = 15/60 = 0.25
    assert phase_of(79, 80, 20) == "pre_terminal"
    assert phase_of(40, 41, 20) == "pre_terminal"    # short game, last ply


def test_phase_bucket_rows_labels_opening_as_temperature():
    _row, replay = make_game(0, a_is_black=True, n_moves=12, a_values=TRAJ)
    rows = phase_bucket_rows("loss", [(side_plies(replay, "black"), 12)], 4)
    by_phase = {r["phase"]: r for r in rows}
    assert by_phase["opening"]["sampling"] == "temperature"
    assert all(r["sampling"] == "argmax" for p, r in by_phase.items()
               if p != "opening")
    assert by_phase["opening"]["plies"] == 2          # A plies 1, 3
    assert sum(r["plies"] for r in rows) == 6
    assert all(r["games"] == 1 for r in rows)
    assert "mean_root_value" in rows[0] and "median_selected_visit_rank" in rows[0]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py -k "cohort_comparison or phase"`
Expected: ImportError

- [ ] **Step 3: Implement**

Append:

```python
def cohort_comparison_row(cohort, a_plies_per_game, opening_plies):
    """Ply-pooled aggregates for one cohort (the cohort_comparison.csv row)."""
    plies = [m for g in a_plies_per_game for m in g]
    post = [m for m in plies if m["ply"] >= opening_plies]
    return {
        "cohort": cohort,
        "games": len(a_plies_per_game),
        "plies": len(plies),
        "mean_root_value": _mean([m["root_value"] for m in plies]),
        "median_root_value": _median([m["root_value"] for m in plies]),
        "mean_top1_share_post": _mean([m["root_top1_share"] for m in post]),
        "median_top1_share_post": _median([m["root_top1_share"] for m in post]),
        "mean_selected_visit_rank_post": _mean(
            [m["selected_visit_rank"] for m in post]),
        "median_selected_visit_rank_post": _median(
            [m["selected_visit_rank"] for m in post]),
        "mean_selected_visit_share_post": _mean(
            [m["selected_visit_count"] / m["root_total_visits"] for m in post]),
        "mean_n_legal": _mean([m["n_legal"] for m in plies]),
    }


def phase_of(ply, n_moves, opening_plies):
    """opening = absolute temp-sampled window; the rest splits into four
    equal game-fraction bands."""
    if ply < opening_plies:
        return "opening"
    f = (ply - opening_plies) / (n_moves - opening_plies)
    return MIDGAME_PHASES[min(3, int(f * 4))]


def phase_bucket_rows(cohort, games, opening_plies):
    """games: list of (a_plies, n_moves). Empty phases are omitted."""
    plies_by = {p: [] for p in PHASES}
    games_by = {p: set() for p in PHASES}
    for gi, (a_plies, n_moves) in enumerate(games):
        for m in a_plies:
            p = phase_of(m["ply"], n_moves, opening_plies)
            plies_by[p].append(m)
            games_by[p].add(gi)
    rows = []
    for p in PHASES:
        ms = plies_by[p]
        if not ms:
            continue
        rows.append({
            "cohort": cohort, "phase": p,
            "sampling": "temperature" if p == "opening" else "argmax",
            "games": len(games_by[p]), "plies": len(ms),
            "mean_root_value": _mean([m["root_value"] for m in ms]),
            "median_root_value": _median([m["root_value"] for m in ms]),
            "mean_top1_share": _mean([m["root_top1_share"] for m in ms]),
            "median_top1_share": _median([m["root_top1_share"] for m in ms]),
            "mean_selected_visit_rank": _mean(
                [m["selected_visit_rank"] for m in ms]),
            "median_selected_visit_rank": _median(
                [m["selected_visit_rank"] for m in ms]),
        })
    return rows
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_loss_replay_analysis.py tests/test_eval_loss_replay_analysis.py
git commit -m "feat(eval): Phase B cohort comparison + phase buckets (temperature-labeled opening)"
```

---

### Task 8: cohens_d + effect_sizes

**Files:**
- Modify: `scripts/GPU/alphazero/eval_loss_replay_analysis.py`
- Modify: `tests/test_eval_loss_replay_analysis.py`

- [ ] **Step 1: Write the failing tests**

Append (extend imports with `cohens_d`, `effect_sizes`):

```python
def test_cohens_d_hand_computed():
    # means 2 vs 4, each var 1 (ddof=1), pooled sd 1 -> d = -2.0
    assert cohens_d([1, 2, 3], [3, 4, 5]) == pytest.approx(-2.0)


def test_cohens_d_degenerate_and_short():
    assert cohens_d([1.0, 1.0], [1.0, 1.0]) is None   # zero pooled variance
    assert cohens_d([1.0], [1.0, 2.0]) is None        # too few samples


def test_effect_sizes_sign_convention_and_nulls():
    loss = [{"final_a_value": -0.9, "largest_a_value_drop": -0.5,
             "initial_a_value": 0.0, "mean_top1_share_post": 0.2,
             "median_selected_visit_rank_post": 3},
            {"final_a_value": -0.7, "largest_a_value_drop": -0.4,
             "initial_a_value": 0.1, "mean_top1_share_post": 0.3,
             "median_selected_visit_rank_post": 2}]
    win = [{"final_a_value": 0.8, "largest_a_value_drop": -0.1,
            "initial_a_value": 0.1, "mean_top1_share_post": 0.5,
            "median_selected_visit_rank_post": 1},
           {"final_a_value": 0.6, "largest_a_value_drop": -0.2,
            "initial_a_value": 0.2, "mean_top1_share_post": 0.4,
            "median_selected_visit_rank_post": 1}]
    out = effect_sizes(loss, win)
    m = out["metrics"]
    assert "cohens_d" in out["formula"]
    assert m["final_a_value"]["d"] < 0                # lower in losses
    assert m["final_a_value"]["delta"] == pytest.approx(-0.8 - 0.7)
    assert m["median_selected_visit_rank_post"]["d"] > 0   # higher rank in losses
    # a metric that is all-None in one cohort yields nulls, not a crash
    for f in win:
        f["mean_top1_share_post"] = None
    out2 = effect_sizes(loss, win)
    assert out2["metrics"]["mean_top1_share_post"]["d"] is None
    assert out2["metrics"]["mean_top1_share_post"]["win_mean"] is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py -k "cohens or effect"`
Expected: ImportError

- [ ] **Step 3: Implement**

Append:

```python
def cohens_d(xs, ys):
    """Cohen's d with pooled sample std (ddof=1); None when either side has
    < 2 samples or the pooled variance is zero (degenerate)."""
    if len(xs) < 2 or len(ys) < 2:
        return None
    pooled = sqrt(((len(xs) - 1) * variance(xs) + (len(ys) - 1) * variance(ys))
                  / (len(xs) + len(ys) - 2))
    if pooled == 0:
        return None
    return (mean(xs) - mean(ys)) / pooled


def effect_sizes(loss_feats, win_feats):
    """Loss-vs-win effect sizes per EFFECT_METRICS. Sign convention is fixed
    by EFFECT_FORMULA: d = (loss - win) / pooled_std."""
    metrics = {}
    for name in EFFECT_METRICS:
        xs = [f[name] for f in loss_feats if f[name] is not None]
        ys = [f[name] for f in win_feats if f[name] is not None]
        lm, wm = _mean(xs), _mean(ys)
        metrics[name] = {
            "loss_mean": lm, "win_mean": wm,
            "delta": (lm - wm) if lm is not None and wm is not None else None,
            "d": cohens_d(xs, ys),
        }
    return {"formula": EFFECT_FORMULA, "metrics": metrics}
```

Note: the `delta` assertion in the test is `pytest.approx(-0.8 - 0.7)` because loss_mean = (−0.9 + −0.7)/2 = −0.8 and win_mean = (0.8 + 0.6)/2 = 0.7, so delta = −0.8 − 0.7 = −1.5.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_loss_replay_analysis.py tests/test_eval_loss_replay_analysis.py
git commit -m "feat(eval): Phase B effect sizes (Cohen's d, explicit sign convention)"
```

---

### Task 9: collapse_distribution + timing_distribution + secondary_contrast_summary

**Files:**
- Modify: `scripts/GPU/alphazero/eval_loss_replay_analysis.py`
- Modify: `tests/test_eval_loss_replay_analysis.py`

- [ ] **Step 1: Write the failing tests**

Append (extend imports with `collapse_distribution`, `timing_distribution`, `secondary_contrast_summary`):

```python
def test_collapse_distribution_groups_failure_modes():
    labels = (["sharp_value_drop"] * 5 + ["gradual_decay"] * 2
              + ["search_diffusion"] * 2 + ["no_clear_signal"])
    d = collapse_distribution(labels)
    assert d["n"] == 10
    assert d["counts"]["sharp_value_drop"] == 5
    assert d["mode_shares"]["value-drop"] == pytest.approx(0.7)
    assert d["mode_shares"]["diffusion"] == pytest.approx(0.2)
    assert d["mode_shares"]["unexplained"] == pytest.approx(0.1)
    assert d["mode_shares"]["already-losing"] == 0.0


def test_timing_distribution_percentiles_and_never():
    feats = [{"first_a_value_below_0_fraction": x,
              "first_a_value_below_bad_fraction": None,
              "first_a_value_below_lost_fraction": None,
              "largest_drop_fraction": x}
             for x in (0.2, 0.4, 0.6)]
    feats.append({"first_a_value_below_0_fraction": None,
                  "first_a_value_below_bad_fraction": None,
                  "first_a_value_below_lost_fraction": None,
                  "largest_drop_fraction": None})
    t = timing_distribution(feats)
    assert t["first_a_value_below_0"]["p50"] == pytest.approx(0.4)
    assert t["first_a_value_below_0"]["p25"] == pytest.approx(0.3)
    assert t["first_a_value_below_0"]["never"] == 1
    assert t["first_a_value_below_lost"]["p50"] is None
    assert t["first_a_value_below_lost"]["never"] == 4
    assert t["largest_drop"]["p75"] == pytest.approx(0.5)


def test_secondary_contrast_summary_gap_and_share():
    f1 = {"mean_a_value": -0.5, "b_mean_value": 0.5,
          "mean_top1_share_post": 0.2, "b_mean_top1_share_post": 0.6,
          "median_selected_visit_rank_post": 3, "b_median_visit_rank_post": 1,
          "first_a_value_below_lost_fraction": 0.5,
          "b_first_value_above_050_fraction": 0.3, "b_saw_it_first": True}
    f2 = {"mean_a_value": -0.25, "b_mean_value": 0.25,
          "mean_top1_share_post": 0.4, "b_mean_top1_share_post": 0.5,
          "median_selected_visit_rank_post": 1, "b_median_visit_rank_post": 1,
          "first_a_value_below_lost_fraction": 0.6,
          "b_first_value_above_050_fraction": None, "b_saw_it_first": False}
    s = secondary_contrast_summary([f1, f2])
    assert s["games"] == 2
    assert s["b_saw_it_first_share"] == 0.5
    assert s["onset_gap_games"] == 1
    assert s["median_onset_gap_fraction"] == pytest.approx(0.2)   # 0.5 - 0.3
    assert s["a_mean_value"] == pytest.approx(-0.375)
    assert s["b_mean_value"] == pytest.approx(0.375)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py -k "distribution or secondary"`
Expected: ImportError

- [ ] **Step 3: Implement**

Append:

```python
def collapse_distribution(labels):
    """Counts per collapse label + shares per failure-mode group."""
    n = len(labels)
    counts = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    mode_shares = {mode: sum(counts.get(l, 0) for l in group) / n
                   for mode, group in FAILURE_MODE_GROUPS.items()}
    mode_shares["unexplained"] = counts.get("no_clear_signal", 0) / n
    return {"n": n, "counts": counts, "mode_shares": mode_shares}


def _pct(vals, q):
    """Linear-interpolated percentile; None on empty input."""
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def timing_distribution(loss_feats):
    """When the loss cohort crosses each value threshold (game fractions)."""
    out = {}
    keys = [(k, f"{k}_fraction") for k in CROSSING_KEYS]
    keys.append(("largest_drop", "largest_drop_fraction"))
    for name, field in keys:
        fracs = [f[field] for f in loss_feats if f[field] is not None]
        out[name] = {"p25": _pct(fracs, 0.25), "p50": _pct(fracs, 0.50),
                     "p75": _pct(fracs, 0.75),
                     "never": len(loss_feats) - len(fracs)}
    return out


def secondary_contrast_summary(loss_feats):
    """A vs B inside the loss cohort. B metrics are in B's own perspective;
    the onset gap asks: did B see the win (>= B_ONSET_HIGH) before A admitted
    the loss (<= lost_value)?"""
    def col(key):
        return [f[key] for f in loss_feats if f.get(key) is not None]

    both = [f for f in loss_feats
            if f.get("b_first_value_above_050_fraction") is not None
            and f.get("first_a_value_below_lost_fraction") is not None]
    gaps = [f["first_a_value_below_lost_fraction"]
            - f["b_first_value_above_050_fraction"] for f in both]
    return {
        "games": len(loss_feats),
        "a_mean_value": _mean(col("mean_a_value")),
        "b_mean_value": _mean(col("b_mean_value")),
        "a_mean_top1_share_post": _mean(col("mean_top1_share_post")),
        "b_mean_top1_share_post": _mean(col("b_mean_top1_share_post")),
        "a_median_visit_rank_post": _median(col("median_selected_visit_rank_post")),
        "b_median_visit_rank_post": _median(col("b_median_visit_rank_post")),
        "b_saw_it_first_share": (
            sum(1 for f in loss_feats if f.get("b_saw_it_first")) / len(loss_feats)
            if loss_feats else None),
        "median_onset_gap_fraction": _median(gaps),
        "onset_gap_games": len(both),
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_loss_replay_analysis.py tests/test_eval_loss_replay_analysis.py
git commit -m "feat(eval): Phase B distributions + secondary (A-vs-B) contrast summary"
```

---

### Task 10: make_verdict

**Files:**
- Modify: `scripts/GPU/alphazero/eval_loss_replay_analysis.py`
- Modify: `tests/test_eval_loss_replay_analysis.py`

- [ ] **Step 1: Write the failing tests**

Append (extend imports with `make_verdict`):

```python
def test_verdict_primary_and_secondary():
    labels = (["sharp_value_drop"] * 5 + ["gradual_decay"] * 2
              + ["search_diffusion"] * 2 + ["no_clear_signal"])
    v = make_verdict(labels, "A-as-black 41-80")
    assert v["primary"] == "value-drop"
    assert v["primary_share"] == pytest.approx(0.7)
    assert v["secondary"] == "diffusion"
    assert v["secondary_share"] == pytest.approx(0.2)
    assert "value-drop" in v["narrative"] and "A-as-black 41-80" in v["narrative"]


def test_verdict_mixed_when_no_mode_reaches_bar():
    labels = (["sharp_value_drop"] * 3 + ["search_diffusion"] * 3
              + ["low_visit_selection"] * 2 + ["already_bad"] * 2)
    v = make_verdict(labels, "X")
    assert v["primary"] == "mixed / no strong single signal"
    assert v["secondary"] is None


def test_verdict_mixed_when_unexplained_dominates():
    labels = ["no_clear_signal"] * 6 + ["sharp_value_drop"] * 4
    v = make_verdict(labels, "X")   # value-drop 0.4 >= bar, but unexplained 0.6 wins
    assert v["primary"] == "mixed / no strong single signal"


def test_verdict_no_secondary_below_bar():
    labels = ["sharp_value_drop"] * 8 + ["search_diffusion"] * 1 + ["no_clear_signal"]
    v = make_verdict(labels, "X")
    assert v["primary"] == "value-drop" and v["secondary"] is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py -k verdict`
Expected: ImportError

- [ ] **Step 3: Implement**

Append:

```python
def make_verdict(labels, cohort_desc):
    """Deterministic verdict from the loss-cohort collapse labels.

    Primary = the failure-mode group with the largest share, if it reaches
    PRIMARY_SHARE and is not beaten by the unexplained share (a tie goes to
    the explained mode). Secondary = next group at SECONDARY_SHARE+.
    """
    dist = collapse_distribution(labels)
    shares = dist["mode_shares"]
    modes = [(m, s) for m, s in shares.items() if m != "unexplained"]
    modes.sort(key=lambda kv: -kv[1])   # FAILURE_MODE_GROUPS order breaks ties
    top_mode, top_share = modes[0]
    unexplained = shares["unexplained"]
    base = {"mode_shares": shares, "primary_share": top_share}
    if top_share < PRIMARY_SHARE or unexplained > top_share:
        return {**base, "primary": "mixed / no strong single signal",
                "secondary": None, "secondary_share": None,
                "narrative": (
                    f"{cohort_desc} losses show no dominant failure mode "
                    f"(top: {top_mode} {top_share:.0%}, "
                    f"unexplained {unexplained:.0%}).")}
    sec, sec_share = next(((m, s) for m, s in modes[1:] if s >= SECONDARY_SHARE),
                          (None, None))
    tail = (f"; secondary signal: {sec} {sec_share:.0%})." if sec else ").")
    return {**base, "primary": top_mode, "secondary": sec,
            "secondary_share": sec_share,
            "narrative": (f"{cohort_desc} losses are best explained by "
                          f"{top_mode} ({top_share:.0%} of losses{tail}")}
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_loss_replay_analysis.py tests/test_eval_loss_replay_analysis.py
git commit -m "feat(eval): Phase B deterministic verdict from collapse distribution"
```

---

### Task 11: review_queue_rows + opening_cluster_rows

**Files:**
- Modify: `scripts/GPU/alphazero/eval_loss_replay_analysis.py`
- Modify: `tests/test_eval_loss_replay_analysis.py`

- [ ] **Step 1: Write the failing tests**

Append (extend imports with `review_queue_rows`, `opening_cluster_rows`):

```python
def _queue_feat(idx, drop, final, top1=0.5, rank=1):
    return {"game_idx": idx, "task_id": idx, "replay_path": f"r/{idx}.json",
            "a_color": "black", "winner": "red", "n_moves": 50,
            "collapse_type": "sharp_value_drop",
            "initial_a_value": 0.1, "final_a_value": final,
            "largest_a_value_drop": drop, "largest_drop_ply": 30,
            "largest_drop_fraction": 0.6,
            "first_a_value_below_lost_ply": 35,
            "first_a_value_below_lost_fraction": 0.7,
            "mean_top1_share_post": top1,
            "median_selected_visit_rank_post": rank, "opening_key": "k"}


def test_review_queue_composite_sort_and_limit():
    feats = [
        _queue_feat(1, -0.5, -0.2),          # mid drop, better final
        _queue_feat(2, -0.8, -0.9),          # sharpest drop -> rank 1
        _queue_feat(3, -0.5, -0.9),          # tie on drop -> worse final first
        _queue_feat(4, -0.1, -0.1),
    ]
    rows = review_queue_rows(feats, limit=3)
    assert [r["game_idx"] for r in rows] == [2, 3, 1]
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert rows[0]["initial_a_value"] == 0.1            # spec: queue carries both
    assert rows[0]["final_a_value"] == -0.9
    assert "flag_sharp" not in rows[0]                  # queue is the curated view


def test_review_queue_null_drop_sorts_last():
    feats = [_queue_feat(1, None, -0.9), _queue_feat(2, -0.3, -0.1)]
    rows = review_queue_rows(feats, limit=10)
    assert [r["game_idx"] for r in rows] == [2, 1]


def test_opening_cluster_rows_grouping_and_sort():
    g0_row, g0 = make_game(0, a_is_black=True, a_wins=False, n_moves=12)
    g1_row, g1 = make_game(1, a_is_black=True, a_wins=True, n_moves=12)
    g2_row, g2 = make_game(2, a_is_black=True, a_wins=False, n_moves=14)
    for m in g2["moves"][:2]:
        m["row"], m["col"] = 9, 9           # distinct opening key
    rows = opening_cluster_rows(
        [(g0, "black", False), (g1, "black", True), (g2, "black", False)],
        key_plies=2, cohort_label="A_black_41_80_decisive", opening_plies=4)
    assert rows[0]["games"] == 2            # the shared key sorts first
    assert rows[0]["wins"] == 1 and rows[0]["losses"] == 1
    assert rows[0]["a_score_rate"] == 0.5
    assert rows[0]["cohort"] == "A_black_41_80_decisive"
    assert rows[0]["opening_plies"] == 2
    assert rows[1]["games"] == 1 and rows[1]["opening_key"] == "r9c9|r9c9"
    assert rows[1]["avg_moves"] == 14
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py -k "queue or cluster"`
Expected: ImportError

- [ ] **Step 3: Implement**

Append:

```python
QUEUE_COLUMNS = (
    "game_idx", "task_id", "replay_path", "a_color", "winner", "n_moves",
    "collapse_type", "initial_a_value", "final_a_value",
    "largest_a_value_drop", "largest_drop_ply", "largest_drop_fraction",
    "first_a_value_below_lost_ply", "first_a_value_below_lost_fraction",
    "mean_top1_share_post", "median_selected_visit_rank_post", "opening_key",
)


def review_queue_rows(loss_feats, limit):
    """Top-N loss games by composite priority: sharpest drop, then lowest
    final value, then lowest post-opening top1 share, then highest rank.
    Null sort keys go last (None drop -> +inf etc.)."""
    def sort_key(f):
        return (
            f["largest_a_value_drop"] if f["largest_a_value_drop"] is not None
            else float("inf"),
            f["final_a_value"] if f["final_a_value"] is not None
            else float("inf"),
            f["mean_top1_share_post"] if f["mean_top1_share_post"] is not None
            else float("inf"),
            -(f["median_selected_visit_rank_post"]
              if f["median_selected_visit_rank_post"] is not None else 0),
        )
    ranked = sorted(loss_feats, key=sort_key)[:limit]
    return [{"rank": i + 1, **{c: f.get(c) for c in QUEUE_COLUMNS}}
            for i, f in enumerate(ranked)]


def opening_cluster_rows(games, key_plies, cohort_label, opening_plies):
    """Context table over focus-window decisive games; NOT the diagnostic.
    games: list of (replay, a_color, a_won). One row per opening key."""
    groups = {}
    for replay, a_clr, won in games:
        groups.setdefault(opening_key(replay, key_plies), []).append(
            (replay, a_clr, won))
    rows = []
    for key, items in sorted(groups.items()):
        n = len(items)
        wins = sum(1 for _r, _c, w in items if w)
        early_vals, early_shares, moves = [], [], []
        for replay, a_clr, _w in items:
            moves.append(replay["n_moves"])
            for m in side_plies(replay, a_clr):
                if m["ply"] < opening_plies:
                    early_vals.append(m["root_value"])
                    early_shares.append(m["root_top1_share"])
        rows.append({
            "opening_plies": key_plies, "opening_key": key,
            "cohort": cohort_label, "games": n,
            "losses": n - wins, "wins": wins,
            "a_score_rate": score_rate(wins, 0, n),
            "mean_root_value_early": _mean(early_vals),
            "mean_top1_share_early": _mean(early_shares),
            "avg_moves": _mean(moves),
        })
    rows.sort(key=lambda r: (-r["games"], r["a_score_rate"]))
    return rows
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_loss_replay_analysis.py tests/test_eval_loss_replay_analysis.py
git commit -m "feat(eval): Phase B manual review queue + opening cluster context table"
```

---

### Task 12: build_replay_summary

**Files:**
- Modify: `scripts/GPU/alphazero/eval_loss_replay_analysis.py`
- Modify: `tests/test_eval_loss_replay_analysis.py`

- [ ] **Step 1: Write the failing tests**

Append (extend imports with `build_replay_summary`, `MIN_WIN_COHORT`, `OPENING_SAMPLING_NOTE`):

```python
def _summary_inputs(n_wins):
    # Values vary per game: identical dicts would give zero variance and a
    # (correctly) null Cohen's d, which is not what this test exercises.
    loss = [{"collapse_type": "sharp_value_drop",
             "final_a_value": -0.9 + 0.05 * i,
             "largest_a_value_drop": -0.6 - 0.01 * i,
             "initial_a_value": 0.0 + 0.01 * i,
             "mean_top1_share_post": 0.3 + 0.01 * i,
             "median_selected_visit_rank_post": 2 + (i % 2),
             "first_a_value_below_0_fraction": 0.4,
             "first_a_value_below_bad_fraction": 0.5,
             "first_a_value_below_lost_fraction": 0.6,
             "largest_drop_fraction": 0.55, "mean_a_value": -0.4,
             "b_mean_value": 0.4, "b_mean_top1_share_post": 0.5,
             "b_median_visit_rank_post": 1,
             "b_first_value_above_050_fraction": 0.5,
             "b_saw_it_first": True} for i in range(6)]
    win = [{"collapse_type": "no_clear_signal",
            "final_a_value": 0.8 - 0.05 * i,
            "largest_a_value_drop": -0.1 - 0.01 * i,
            "initial_a_value": 0.1 + 0.01 * i,
            "mean_top1_share_post": 0.5 - 0.01 * i,
            "median_selected_visit_rank_post": 1,
            "first_a_value_below_0_fraction": None,
            "first_a_value_below_bad_fraction": None,
            "first_a_value_below_lost_fraction": None,
            "largest_drop_fraction": 0.3, "mean_a_value": 0.5}
           for i in range(n_wins)]
    return loss, win


def _build(loss, win):
    return build_replay_summary(
        match="m", pairing_id="0399_vs_0379", a_ckpt=A, b_ckpt=B,
        filters={"a_color": "black"}, counts={"loss": len(loss), "win": len(win)},
        loss_feats=loss, win_feats=win,
        verdict=make_verdict([f["collapse_type"] for f in loss], "A-as-black"),
        cohort_rows=[{"cohort": "loss"}, {"cohort": "win"}],
        secondary=secondary_contrast_summary(loss))


def test_build_replay_summary_full_shape():
    loss, win = _summary_inputs(n_wins=6)
    s = _build(loss, win)
    assert s["match"] == "m" and s["a_checkpoint"] == A
    assert OPENING_SAMPLING_NOTE in s["notes"]
    assert s["primary_contrast"]["effect_sizes"]["metrics"]["final_a_value"]["d"] is not None
    assert s["primary_contrast"]["note"] is None
    assert s["collapse_type_distribution"]["mode_shares"]["value-drop"] == 1.0
    assert s["timing_distribution"]["first_a_value_below_lost"]["p50"] == pytest.approx(0.6)
    assert s["verdict"]["primary"] == "value-drop"
    assert s["secondary_contrast"]["b_saw_it_first_share"] == 1.0


def test_build_replay_summary_insufficient_contrast():
    loss, win = _summary_inputs(n_wins=MIN_WIN_COHORT - 1)
    s = _build(loss, win)
    assert s["primary_contrast"]["effect_sizes"] is None
    assert s["primary_contrast"]["note"] == "insufficient_contrast"
    assert s["verdict"]["primary"] == "value-drop"   # verdict still computed
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py -k build_replay_summary`
Expected: ImportError

- [ ] **Step 3: Implement**

Append:

```python
def build_replay_summary(*, match, pairing_id, a_ckpt, b_ckpt, filters,
                         counts, loss_feats, win_feats, verdict,
                         cohort_rows, secondary):
    """Assemble the replay_summary.json payload (pure; CLI writes it)."""
    insufficient = len(win_feats) < MIN_WIN_COHORT
    return {
        "match": match,
        "pairing_id": pairing_id,
        "a_checkpoint": a_ckpt,
        "b_checkpoint": b_ckpt,
        "filters": filters,
        "cohorts": counts,
        "notes": [OPENING_SAMPLING_NOTE],
        "primary_contrast": {
            "cohort_comparison": cohort_rows,
            "effect_sizes": (None if insufficient
                             else effect_sizes(loss_feats, win_feats)),
            "note": "insufficient_contrast" if insufficient else None,
        },
        "secondary_contrast": secondary,
        "collapse_type_distribution": collapse_distribution(
            [f["collapse_type"] for f in loss_feats]),
        "timing_distribution": timing_distribution(loss_feats),
        "verdict": verdict,
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_loss_replay_analysis.py tests/test_eval_loss_replay_analysis.py
git commit -m "feat(eval): Phase B replay summary assembly (insufficient-contrast aware)"
```

---

### Task 13: CLI parse_args + thresholds

**Files:**
- Create: `scripts/GPU/alphazero/eval_loss_replay_analyzer.py`
- Create: `tests/test_eval_loss_replay_analyzer_cli.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_eval_loss_replay_analyzer_cli.py`:

```python
import json

import pytest

from scripts.GPU.alphazero.eval_loss_replay_analyzer import (
    parse_args, thresholds_from_args,
)
from tests.eval_replay_fixtures import A, B, make_game


def test_parse_args_defaults():
    args = parse_args(["--games-jsonl", "x_games.jsonl"])
    assert args.a_color == "black"
    assert (args.min_moves, args.max_moves) == (41, 80)
    assert args.opening_plies == 20 and args.opening_key_plies == 4
    assert args.review_queue == 50
    th = thresholds_from_args(args)
    assert th.bad_value == -0.25 and th.lost_value == -0.50
    assert th.sharp_drop == 0.40 and th.low_top1_share == 0.10
    assert th.low_visit_rank == 5 and th.opening_plies == 20


def test_parse_args_rejects_bad_value_not_above_lost_value():
    with pytest.raises(SystemExit) as e:
        parse_args(["--games-jsonl", "x", "--bad-value", "-0.6"])
    assert e.value.code == 2


def test_parse_args_rejects_nonpositive_sharp_drop():
    with pytest.raises(SystemExit) as e:
        parse_args(["--games-jsonl", "x", "--sharp-drop", "0"])
    assert e.value.code == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analyzer_cli.py`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement the CLI skeleton**

`scripts/GPU/alphazero/eval_loss_replay_analyzer.py`:

```python
"""CLI for the V2 Phase B replay-aware loss analyzer.

Reads Phase A capture data (*_games.jsonl rows carrying replay_path + per-game
replay sidecars), explains WHY checkpoint A loses in the focus window, and
writes six artifacts per match to --output-dir. All analysis lives in
eval_loss_replay_analysis; this module is IO + composition + formatting.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .eval_loss_analysis import (
    a_color, resolve_checkpoints, score_for_checkpoint, validate_rows,
)
from .eval_loss_analyzer import (
    load_jsonl, load_sibling_summary, resolve_inputs, stem_of, write_csv,
    write_json,
)
from .eval_loss_replay_analysis import (
    MIN_WIN_COHORT, Thresholds, b_side_features, build_replay_summary,
    classify_collapse, cohort_comparison_row, game_features, make_verdict,
    opening_cluster_rows, phase_bucket_rows, review_queue_rows,
    secondary_contrast_summary, side_plies, validate_replay,
)
from .eval_runner import short_id


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Explain WHY checkpoint A loses, from Phase A replay data.")
    p.add_argument("--games-jsonl", action="append", default=[], metavar="PATH",
                   help="input games jsonl with replay_path rows (repeatable)")
    p.add_argument("--glob", default=None, metavar="PATTERN",
                   help="glob for input games jsonl files")
    p.add_argument("--output-dir", default=Path("logs/eval/loss_analysis_v2"),
                   type=Path)
    p.add_argument("--a-checkpoint", default=None)
    p.add_argument("--b-checkpoint", default=None)
    p.add_argument("--a-color", choices=("red", "black"), default="black")
    p.add_argument("--min-moves", type=int, default=41)
    p.add_argument("--max-moves", type=int, default=80)
    p.add_argument("--opening-plies", type=int, default=20,
                   help="temperature-sampled opening window; confidence/"
                        "diffusion features use plies >= this only")
    p.add_argument("--opening-key-plies", type=int, default=4)
    p.add_argument("--bad-value", type=float, default=-0.25)
    p.add_argument("--lost-value", type=float, default=-0.50)
    p.add_argument("--sharp-drop", type=float, default=0.40)
    p.add_argument("--low-top1-share", type=float, default=0.10)
    p.add_argument("--low-visit-rank", type=int, default=5)
    p.add_argument("--review-queue", type=int, default=50)
    args = p.parse_args(argv)
    if args.bad_value <= args.lost_value:
        p.error("--bad-value must be greater than --lost-value")
    if args.sharp_drop <= 0:
        p.error("--sharp-drop must be > 0")
    return args


def thresholds_from_args(args):
    return Thresholds(
        bad_value=args.bad_value, lost_value=args.lost_value,
        sharp_drop=args.sharp_drop, low_top1_share=args.low_top1_share,
        low_visit_rank=args.low_visit_rank, opening_plies=args.opening_plies)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analyzer_cli.py`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_loss_replay_analyzer.py tests/test_eval_loss_replay_analyzer_cli.py
git commit -m "feat(eval): Phase B CLI argparse + threshold sanity"
```

---

### Task 14: CLI pipeline — analyze, write artifacts, console verdict

**Files:**
- Modify: `scripts/GPU/alphazero/eval_loss_replay_analyzer.py`
- Modify: `tests/test_eval_loss_replay_analyzer_cli.py`

- [ ] **Step 1: Write the failing integration tests**

Append to `tests/test_eval_loss_replay_analyzer_cli.py` (extend imports with `main` from the CLI module):

```python
def _write_capture(tmp_path, games):
    """Write a games.jsonl + sidecars for (row, replay) pairs; returns jsonl path."""
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir(exist_ok=True)
    jsonl = tmp_path / "synth_games.jsonl"
    with jsonl.open("w") as fh:
        for row, replay in games:
            row = dict(row)
            path = replay_dir / f"game_{row['game_idx']:06d}.json"
            path.write_text(json.dumps(replay))
            row["replay_path"] = str(path)
            fh.write(json.dumps(row) + "\n")
    return jsonl


def _synth_games():
    """6 A-black losses + 6 A-black wins in a 41-80 window, 1 draw, 1 short."""
    games = []
    losing = [0.25] * 10 + [-0.125, -0.375, -0.625, -0.75, -0.875,
                            -0.875, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    winning = [0.25] * 10 + [0.375, 0.5, 0.5, 0.625, 0.625,
                             0.75, 0.75, 0.875, 0.875, 1.0, 1.0, 1.0]
    rising = [0.0] * 10 + [0.125, 0.25, 0.375, 0.5, 0.625, 0.75,
                           0.875, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(6):
        games.append(make_game(i, a_is_black=True, a_wins=False, n_moves=44,
                               a_values=losing, b_values=rising))
    for i in range(6, 12):
        games.append(make_game(i, a_is_black=True, a_wins=True, n_moves=44,
                               a_values=winning))
    games.append(make_game(12, a_is_black=True, reason="state_cap", n_moves=44))
    games.append(make_game(13, a_is_black=True, a_wins=False, n_moves=30,
                           a_values=[0.0] * 15))
    return games


def test_cli_end_to_end_writes_all_artifacts(tmp_path, capsys):
    jsonl = _write_capture(tmp_path, _synth_games())
    out = tmp_path / "out"
    rc = main(["--games-jsonl", str(jsonl), "--output-dir", str(out)])
    assert rc == 0
    stem = "synth"
    for suffix in ("replay_summary.json", "cohort_comparison.csv",
                   "phase_buckets.csv", "collapse_timing.csv",
                   "manual_review_queue.csv", "opening_clusters.csv"):
        assert (out / f"{stem}_{suffix}").exists(), suffix
    s = json.loads((out / f"{stem}_replay_summary.json").read_text())
    assert s["cohorts"] == {"focus_window_games": 13, "excluded_draws": 1,
                            "loss": 6, "win": 6}
    assert s["primary_contrast"]["effect_sizes"] is not None   # 6 wins >= 5
    assert s["verdict"]["primary"] == "value-drop"
    timing = (out / f"{stem}_collapse_timing.csv").read_text().splitlines()
    assert len(timing) == 13                                   # header + 12 games
    header = timing[0].split(",")
    assert "collapse_type" in header and "flag_sharp" in header
    assert "b_saw_it_first" in header and "cohort" in header
    console = capsys.readouterr().out
    assert "Phase B verdict:" in console
    assert "manual_review_queue.csv" in console


def test_cli_skips_v1_era_file_without_replay_path(tmp_path, capsys):
    jsonl = tmp_path / "old_games.jsonl"
    with jsonl.open("w") as fh:
        for row, _replay in _synth_games():
            row = dict(row)
            del row["replay_path"]
            fh.write(json.dumps(row) + "\n")
    rc = main(["--games-jsonl", str(jsonl), "--output-dir", str(tmp_path / "o")])
    assert rc == 0
    assert "no replay capture" in capsys.readouterr().out
    assert not (tmp_path / "o").exists()


def test_cli_null_replay_path_in_focus_window_raises(tmp_path):
    games = _synth_games()
    jsonl = _write_capture(tmp_path, games)
    rows = [json.loads(l) for l in jsonl.read_text().splitlines()]
    rows[0]["replay_path"] = None
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError, match="replay_path"):
        main(["--games-jsonl", str(jsonl), "--output-dir", str(tmp_path / "o")])


def test_cli_empty_loss_cohort_raises(tmp_path):
    games = [g for g in _synth_games() if g[0]["winner_checkpoint"] == A
             or g[0]["winner"] is None]
    jsonl = _write_capture(tmp_path, games)
    with pytest.raises(ValueError, match="nothing to explain"):
        main(["--games-jsonl", str(jsonl), "--output-dir", str(tmp_path / "o")])
```

Note on `_synth_games` lengths: a 44-move game seats black on 22 plies, so each `a_values`/`b_values` list has 22 entries (10 opening + 12). The 30-move game has 15 black plies. All values are binary-exact.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analyzer_cli.py`
Expected: ImportError (`main` not defined)

- [ ] **Step 3: Implement the pipeline**

Append to `eval_loss_replay_analyzer.py`:

```python
def load_replay(row):
    path = row.get("replay_path")
    if path is None:
        raise ValueError(
            f"game {row['game_idx']}: focus-window row has no replay_path "
            "(partially captured file)")
    with open(path) as fh:
        replay = json.load(fh)
    validate_replay(row, replay)
    return replay


def analyze_input(path, args, th):
    """Full analysis for one games.jsonl; returns the artifact bundle or
    None when the file is skippable (empty / no capture / self-match)."""
    rows = load_jsonl(path)
    stem = stem_of(path)
    if not rows:
        print(f"skip {stem}: empty file")
        return None
    if not any(r.get("replay_path") for r in rows):
        print(f"skip {stem}: no replay capture (no replay_path in rows)")
        return None
    sidecar = load_sibling_summary(path)
    a, b = resolve_checkpoints(rows, rows[0]["pairing_id"],
                               args.a_checkpoint, args.b_checkpoint, sidecar)
    if a == b:
        print(f"skip {stem}: self-match ({short_id(a)})")
        return None
    validate_rows(rows, a, b)
    a_clr = args.a_color
    b_clr = "red" if a_clr == "black" else "black"
    window = [r for r in rows if a_color(r, a) == a_clr
              and args.min_moves <= r["n_moves"] <= args.max_moves]
    decisive = [r for r in window if r["reason"] == "win"]
    loss_rows = [r for r in decisive if score_for_checkpoint(r, a) == 0.0]
    win_rows = [r for r in decisive if score_for_checkpoint(r, a) == 1.0]
    if not loss_rows:
        raise ValueError(
            f"{stem}: no decisive A losses in the focus window (a_color="
            f"{a_clr}, moves {args.min_moves}-{args.max_moves}) — "
            "nothing to explain")

    feats = {"loss": [], "win": []}
    plies_games = {"loss": [], "win": []}
    cluster_games = []
    for cohort, cohort_rows_in in (("loss", loss_rows), ("win", win_rows)):
        for r in cohort_rows_in:
            replay = load_replay(r)
            f = game_features(r, replay, a_clr, th, args.opening_key_plies)
            f["cohort"] = cohort
            label, flags = classify_collapse(f, th)
            f["collapse_type"] = label
            f.update(flags)
            if cohort == "loss":
                f.update(b_side_features(
                    replay, b_clr, th, f["first_a_value_below_lost_fraction"]))
            feats[cohort].append(f)
            plies_games[cohort].append((side_plies(replay, a_clr), r["n_moves"]))
            cluster_games.append((replay, a_clr, cohort == "win"))

    cohort_rows = [
        cohort_comparison_row(c, [g for g, _n in plies_games[c]],
                              th.opening_plies)
        for c in ("loss", "win") if plies_games[c]]
    phase_rows = [row for c in ("loss", "win") if plies_games[c]
                  for row in phase_bucket_rows(c, plies_games[c],
                                               th.opening_plies)]
    cohort_desc = f"A-as-{a_clr} {args.min_moves}-{args.max_moves}"
    verdict = make_verdict([f["collapse_type"] for f in feats["loss"]],
                           cohort_desc)
    summary = build_replay_summary(
        match=stem, pairing_id=rows[0]["pairing_id"], a_ckpt=a, b_ckpt=b,
        filters={"a_color": a_clr, "min_moves": args.min_moves,
                 "max_moves": args.max_moves,
                 "opening_key_plies": args.opening_key_plies, **asdict(th)},
        counts={"focus_window_games": len(window),
                "excluded_draws": len(window) - len(decisive),
                "loss": len(loss_rows), "win": len(win_rows)},
        loss_feats=feats["loss"], win_feats=feats["win"], verdict=verdict,
        cohort_rows=cohort_rows,
        secondary=secondary_contrast_summary(feats["loss"]))
    return {
        "stem": stem, "summary": summary, "feats": feats,
        "cohort_rows": cohort_rows, "phase_rows": phase_rows,
        "queue": review_queue_rows(feats["loss"], args.review_queue),
        "clusters": opening_cluster_rows(
            cluster_games, args.opening_key_plies,
            f"A_{a_clr}_{args.min_moves}_{args.max_moves}_decisive",
            th.opening_plies),
    }


def timing_csv_rows(feats):
    """One row per focus game, loss rows first so the CSV header carries the
    B-side columns; win rows get blanks for those."""
    rows = feats["loss"] + feats["win"]
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    return [{k: r.get(k) for k in keys} for r in rows]


def write_outputs(out_dir, res):
    stem = res["stem"]
    write_json(out_dir / f"{stem}_replay_summary.json", res["summary"])
    write_csv(out_dir / f"{stem}_cohort_comparison.csv", res["cohort_rows"])
    write_csv(out_dir / f"{stem}_phase_buckets.csv", res["phase_rows"])
    write_csv(out_dir / f"{stem}_collapse_timing.csv",
              timing_csv_rows(res["feats"]))
    write_csv(out_dir / f"{stem}_manual_review_queue.csv", res["queue"])
    write_csv(out_dir / f"{stem}_opening_clusters.csv", res["clusters"])


def print_console_summary(res, out_dir):
    s = res["summary"]
    print("=" * 60)
    print(f"REPLAY LOSS ANALYSIS (V2): {s['match']}")
    print("=" * 60)
    f, c = s["filters"], s["cohorts"]
    print(f"Focus window: A as {f['a_color']}, {f['min_moves']}-"
          f"{f['max_moves']} moves -> {c['loss']} losses, {c['win']} wins "
          f"({c['excluded_draws']} draws excluded)")
    dist = s["collapse_type_distribution"]
    print("Collapse types (losses):")
    for lab, cnt in sorted(dist["counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {lab:<20} {cnt:>4}  ({cnt / dist['n']:.0%})")
    pc = s["primary_contrast"]
    if pc["effect_sizes"] is None:
        print(f"Effect sizes: {pc['note']} (win cohort < {MIN_WIN_COHORT})")
    else:
        print("Effect sizes (loss vs win, Cohen's d):")
        for name, e in pc["effect_sizes"]["metrics"].items():
            d = e["d"]
            d_s = "n/a" if d is None else f"{d:+.2f}"
            print(f"  {name:<34} d={d_s}")
    sec = s["secondary_contrast"]
    if sec["b_saw_it_first_share"] is not None:
        print(f"B saw the win first in {sec['b_saw_it_first_share']:.0%} of "
              f"losses (onset-gap games: {sec['onset_gap_games']})")
    print(f"Phase B verdict: {s['verdict']['narrative']}")
    print(f"Manual review queue: "
          f"{out_dir / (s['match'] + '_manual_review_queue.csv')}")


def main(argv=None):
    args = parse_args(argv)
    inputs = resolve_inputs(args)
    if not inputs:
        print("error: no input files (use --games-jsonl and/or --glob)",
              file=sys.stderr)
        return 2
    th = thresholds_from_args(args)
    for path in inputs:
        res = analyze_input(path, args, th)
        if res is None:
            continue
        write_outputs(args.output_dir, res)
        print_console_summary(res, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analyzer_cli.py`
Expected: all passed (7 tests)

- [ ] **Step 5: Run the full new-test surface**

Run: `.venv/bin/python -m pytest tests/test_eval_loss_replay_analysis.py tests/test_eval_loss_replay_analyzer_cli.py`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/eval_loss_replay_analyzer.py tests/test_eval_loss_replay_analyzer_cli.py
git commit -m "feat(eval): Phase B CLI pipeline — six artifacts + console verdict"
```

---

### Task 15: Full-suite regression + real-data acceptance run

**Files:** none created; verification only.

- [ ] **Step 1: Full test suite**

Run: `.venv/bin/python -m pytest`
Expected: no NEW failures vs main. (Known pre-existing failures on main: missing Replays/ data, JS-parity, Node-oracle groups — about 12, documented in the checkpoint-tournament memory note. Compare against a `git stash`-free baseline if unsure.)

- [ ] **Step 2: Acceptance run on the real capture**

Run:
```bash
.venv/bin/python -m scripts.GPU.alphazero.eval_loss_replay_analyzer \
  --games-jsonl logs/eval/eps035_0399_vs_0379_800g_w4_seed35791_replay_games.jsonl
```
Expected:
- Console reports the focus cohort ≈ "172 losses" (A-as-black 41–80 decisive; wins ≈ 100–130) and ends with `Phase B verdict: ...` + the queue path.
- Six artifacts in `logs/eval/loss_analysis_v2/` with stem `eps035_0399_vs_0379_800g_w4_seed35791_replay`.

- [ ] **Step 3: Sanity-check the verdict against raw sidecars**

Open the top 3 rows of `..._manual_review_queue.csv`, then `python3 -c` print those games' A-ply `root_value` sequences from their `replay_path` files; confirm the assigned `collapse_type` visually matches each trajectory.

- [ ] **Step 4: Commit any acceptance artifacts you want tracked (optional) and stop**

Analysis outputs under `logs/` stay untracked (matches V1 convention). Nothing to commit in this step unless a bug was found and fixed.

---

## Self-review notes (already applied)

- Spec coverage: every spec section maps to a task — schema/conventions (T1–2), value features (T3), opening exclusion (T4), classification (T5), per-game + B-side (T6), cohort/phase tables (T7), effect sizes (T8), distributions/secondary (T9), verdict (T10), queue/clusters (T11), summary JSON (T12), CLI flags/sanity (T13), pipeline/artifacts/console + error paths (T14), acceptance (T15).
- The spec's `a_ply_series` → `side_plies` rename and the `*_analyzer_cli.py` test naming are intentional and documented at the top.
- Type consistency: feature keys used by `classify_collapse`, `EFFECT_METRICS`, `QUEUE_COLUMNS`, `timing_distribution`, and `secondary_contrast_summary` all match the keys produced in `value_features` / `confidence_features` / `game_features` / `b_side_features` (`b_first_value_above_050_fraction` etc.).
