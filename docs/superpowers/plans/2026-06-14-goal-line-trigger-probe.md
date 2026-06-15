# Goal-Line Trigger Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fast 18-case checkpoint calibration probe that re-evaluates fixed "goal-line trigger" positions (black to move, one ply before red's goal-line move) and reports whether each checkpoint overvalues black — lower black `root_value` = better calibrated.

**Architecture:** Three new modules in `scripts/GPU/alphazero/`: a pure helper module (selection, board reconstruction, summary stats — no MLX), a Mode-A manifest generator (candidates CSV → manifest), and the probe evaluator CLI (MCTS over each case × checkpoint). All heavy machinery (`TwixtState`, `MCTS`, evaluator loading) is reused from the existing eval modules; the probe takes an injectable `evaluator_factory` so integration tests run on `FakeEvaluator` with no MLX.

**Tech Stack:** Python 3.14 stdlib (`statistics`, `argparse`, `csv`, `json`, `random`). Tests: pytest (repo defaults). Run with `.venv/bin/python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-06-14-goal-line-trigger-probe-design.md` — read it first. Branch: `feature/goal-line-trigger-probe` (already checked out; spec committed at `12a8073`).

**Key domain facts:**
- A case's `position_ply` is **black's** decision point; `trigger_red_ply = position_ply+1`, `drop_black_ply = position_ply+2`. The probe evaluates the board after `moves[0:position_ply]` (black to move) and records black's `root_value` (already black's perspective — no sign flip).
- `post_opening_only` keys on the drop's `largest_drop_phase`, **not** `position_ply` — so `position_ply` may be < `opening_plies` (canonical game 15: `position_ply=19`). No `position_ply >= opening_plies` filter exists.
- `TwixtState.apply_move((row,col))` **validates legality**, so any synthetic replay used in tests must be a legal game prefix — build it by walking `legal_moves()`.

**Canonical data (untracked, under `logs/eval/loss_analysis_v2_1/`):** real-data tests `skipif` these are absent; synthetic tests always run.

---

## File map

- Create: `scripts/GPU/alphazero/goal_line_trigger_probe_cases.py` — pure: `select_cases`, `case_id`, `position_state`, `summarize` (Tasks 1–3)
- Create: `scripts/GPU/alphazero/generate_goal_line_trigger_probe_manifest.py` — Mode-A CLI (Task 4)
- Create: `scripts/GPU/alphazero/eval_goal_line_trigger_probe.py` — probe CLI (Task 5)
- Create: `tests/goal_line_probe_fixtures.py` — `legal_replay` helper (Task 2)
- Create: `tests/test_goal_line_trigger_probe_cases.py` — pure tests (Tasks 1–3)
- Create: `tests/test_goal_line_trigger_probe_cli.py` — generator + probe integration (Tasks 4–5)
- Read-only deps: `scripts/GPU/alphazero/eval_runner.py` (`EvalConfig`, `cfg_from`, `short_id`, `_default_evaluator_factory`), `mcts.py` (`MCTS`), `game/twixt_state.py` (`TwixtState`), `tests/eval_fakes.py` (`FakeEvaluator`).

Commit footer for every commit: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. `git add` only the files each task touches (the tree has unrelated untracked files). The pre-commit hook prints unrelated ESLint noise; ignore it.

---

### Task 1: `select_cases` + candidate→case mapping + `case_id`

**Files:**
- Create: `scripts/GPU/alphazero/goal_line_trigger_probe_cases.py`
- Create: `tests/test_goal_line_trigger_probe_cases.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_goal_line_trigger_probe_cases.py`:

```python
import csv
import json
from pathlib import Path

import pytest

from scripts.GPU.alphazero.goal_line_trigger_probe_cases import (
    DEFAULT_SELECTION, EXPECTED_PROBLEM, case_id, select_cases,
)

CANON_DIR = Path("logs/eval/loss_analysis_v2_1")
CANON_CANDIDATES = CANON_DIR / "goal_line_trigger_probe_candidates.csv"
CANON_MANIFEST = CANON_DIR / "goal_line_trigger_probe_manifest.json"


def _cand(**over):
    base = {
        "game_idx": "769", "rank": "4", "n_moves": "45",
        "collapse_type": "sharp_value_drop", "largest_drop_phase": "post_opening",
        "trigger_zone": "red_goal_band_3", "prev_black_ply": "39",
        "prev_black_row": "21", "prev_black_col": "21", "prev_black_value": "0.88",
        "prev_black_top1": "0.885", "trigger_red_ply": "40", "trigger_red_row": "22",
        "trigger_red_col": "22", "trigger_red_value": "0.65", "trigger_red_top1": "0.955",
        "drop_black_ply": "41", "drop_black_row": "18", "drop_black_col": "6",
        "drop_black_value": "-0.46", "drop_black_top1": "0.08", "drop_amount": "-1.34",
        "replay_path": "logs/eval/x_replays/game_000769.json",
    }
    base.update(over)
    return base


def test_candidate_to_case_field_mapping():
    case = select_cases([_cand()], DEFAULT_SELECTION)[0]
    assert case["game_idx"] == 769 and case["rank"] == 4
    assert case["position_ply"] == 39 and case["side_to_move"] == "black"
    assert case["expected_problem"] == EXPECTED_PROBLEM
    assert case["trigger_red_ply"] == 40
    assert case["trigger_red_move"] == {"row": 22, "col": 22}
    assert case["trigger_zone"] == "red_goal_band_3"
    assert case["baseline_black_prev_value"] == 0.88
    assert case["baseline_black_prev_top1"] == 0.885
    assert case["drop_black_ply"] == 41 and case["drop_amount"] == -1.34
    assert case["replay_path"] == "logs/eval/x_replays/game_000769.json"


def test_select_filters_each_knob_at_boundary():
    sel = DEFAULT_SELECTION
    # value below 0.25 -> dropped; exactly 0.25 -> kept
    assert select_cases([_cand(prev_black_value="0.24")], sel) == []
    assert len(select_cases([_cand(prev_black_value="0.25")], sel)) == 1
    # top1 below 0.5 -> dropped; exactly 0.5 -> kept
    assert select_cases([_cand(prev_black_top1="0.49")], sel) == []
    assert len(select_cases([_cand(prev_black_top1="0.5")], sel)) == 1
    # not post_opening -> dropped (when post_opening_only)
    assert select_cases([_cand(largest_drop_phase="opening")], sel) == []
    # zone not red_goal* -> dropped
    assert select_cases([_cand(trigger_zone="center")], sel) == []


def test_select_post_opening_only_can_be_disabled():
    sel = {**DEFAULT_SELECTION, "post_opening_only": False}
    assert len(select_cases([_cand(largest_drop_phase="opening")], sel)) == 1


def test_select_preserves_input_order():
    rows = [_cand(game_idx="3", rank="1"), _cand(game_idx="1", rank="2")]
    assert [c["game_idx"] for c in select_cases(rows, DEFAULT_SELECTION)] == [3, 1]


def test_case_id_format():
    case = select_cases([_cand(game_idx="15", prev_black_ply="19")], DEFAULT_SELECTION)[0]
    assert case_id(case) == "game_000015_ply_19"


@pytest.mark.skipif(not CANON_CANDIDATES.exists() or not CANON_MANIFEST.exists(),
                    reason="canonical loss_analysis_v2_1 artifacts not present")
def test_real_candidates_reproduce_canonical_18():
    with CANON_CANDIDATES.open() as f:
        rows = list(csv.DictReader(f))
    manifest = json.loads(CANON_MANIFEST.read_text())
    got = select_cases(rows, manifest["selection"])
    got_keys = [(c["game_idx"], c["position_ply"]) for c in got]
    want_keys = [(c["game_idx"], c["position_ply"]) for c in manifest["cases"]]
    assert got_keys == want_keys          # exact set AND order
    assert len(got_keys) == manifest["num_cases"] == 18
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_goal_line_trigger_probe_cases.py`
Expected: ERROR — `ModuleNotFoundError: ... goal_line_trigger_probe_cases`

- [ ] **Step 3: Implement**

`scripts/GPU/alphazero/goal_line_trigger_probe_cases.py`:

```python
"""Pure helpers for the goal-line trigger probe.

No MLX. Selection of fixed trigger cases from the candidates CSV, board
reconstruction at a case's position_ply, and per-checkpoint summary stats.
A case's position_ply is BLACK's decision point, one ply before red's goal-line
trigger move (trigger_red_ply = position_ply + 1). post_opening_only keys on the
drop's largest_drop_phase, NOT on position_ply (which may sit inside the opening
window — e.g. game 15, position_ply=19).
"""
from __future__ import annotations

from statistics import mean, median

from .game.twixt_state import TwixtState

EXPECTED_PROBLEM = "black_overvalues_red_goal_trigger"
OVERVALUE_THRESHOLD = 0.25
SEVERE_OVERVALUE_THRESHOLD = 0.50

DEFAULT_SELECTION = {
    "min_prev_black_value": 0.25,
    "min_prev_black_top1": 0.5,
    "post_opening_only": True,
    "trigger_zone_prefix": "red_goal",
}


def _candidate_to_case(r):
    """Map one candidates.csv row (string values) to a manifest case dict."""
    return {
        "game_idx": int(r["game_idx"]),
        "rank": int(r["rank"]),
        "replay_path": r["replay_path"],
        "position_ply": int(r["prev_black_ply"]),
        "side_to_move": "black",
        "expected_problem": EXPECTED_PROBLEM,
        "trigger_red_ply": int(r["trigger_red_ply"]),
        "trigger_red_move": {"row": int(r["trigger_red_row"]),
                             "col": int(r["trigger_red_col"])},
        "trigger_zone": r["trigger_zone"],
        "baseline_black_prev_value": float(r["prev_black_value"]),
        "baseline_black_prev_top1": float(r["prev_black_top1"]),
        "drop_black_ply": int(r["drop_black_ply"]),
        "drop_black_value": float(r["drop_black_value"]),
        "drop_amount": float(r["drop_amount"]),
    }


def select_cases(candidate_rows, selection):
    """Filter candidate rows by the selection criteria; map survivors to cases.

    Order is preserved (the candidates CSV is rank-sorted). The post_opening_only
    test reads largest_drop_phase, never position_ply.
    """
    out = []
    for r in candidate_rows:
        if float(r["prev_black_value"]) < selection["min_prev_black_value"]:
            continue
        if float(r["prev_black_top1"]) < selection["min_prev_black_top1"]:
            continue
        if selection["post_opening_only"] and r["largest_drop_phase"] != "post_opening":
            continue
        if not r["trigger_zone"].startswith(selection["trigger_zone_prefix"]):
            continue
        out.append(_candidate_to_case(r))
    return out


def case_id(case):
    return f"game_{case['game_idx']:06d}_ply_{case['position_ply']}"
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_goal_line_trigger_probe_cases.py`
Expected: all passed (the canonical test runs if the data is present, else skips)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/goal_line_trigger_probe_cases.py tests/test_goal_line_trigger_probe_cases.py
git commit -m "feat(probe): goal-line trigger case selection + candidate->case mapping"
```

---

### Task 2: `position_state` (board reconstruction) + legal-replay fixture

**Files:**
- Modify: `scripts/GPU/alphazero/goal_line_trigger_probe_cases.py` (append `position_state`)
- Create: `tests/goal_line_probe_fixtures.py`
- Modify: `tests/test_goal_line_trigger_probe_cases.py` (append)

- [ ] **Step 1: Write the legal-replay fixture**

`tests/goal_line_probe_fixtures.py`:

```python
"""Shared test fixtures for the goal-line trigger probe. No MLX.

legal_replay builds a synthetic replay whose moves are a legal TwixtState game
prefix, so position_state (which applies moves to a real TwixtState and validates
legality) can replay any prefix. Deterministic: always takes the first legal move.
"""
from scripts.GPU.alphazero.game.twixt_state import TwixtState


def legal_replay(n_plies, *, board_size=24, game_idx=0, winner="red", reason="win"):
    state = TwixtState(active_size=board_size, to_move="red",
                       max_plies_limit=board_size * board_size)
    moves = []
    for ply in range(n_plies):
        if state.winner() is not None:
            break
        legal = state.legal_moves()
        if not legal:
            break
        r, c = legal[0]
        moves.append({
            "ply": ply, "player": state.to_move, "row": r, "col": c,
            "root_value": 0.0, "root_top1_share": 0.5,
            "selected_visit_rank": 1, "selected_visit_count": 100,
            "root_total_visits": 100, "n_legal": len(legal),
        })
        state = state.apply_move((r, c))
    return {
        "schema_version": 1, "game_idx": game_idx, "board_size": board_size,
        "n_moves": len(moves), "winner": winner, "reason": reason, "moves": moves,
    }
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_goal_line_trigger_probe_cases.py` (add `position_state` to the module import; add `from tests.goal_line_probe_fixtures import legal_replay`):

```python
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
from tests.goal_line_probe_fixtures import legal_replay


def test_position_state_reconstructs_black_to_move():
    replay = legal_replay(8)                      # plies 0..7; ply 5 is black's turn
    state = position_state(replay, 5, "black")    # apply moves[0:5] -> black to move
    assert state.to_move == "black"


def test_position_state_position_ply_19_inside_opening_window():
    # Boundary: game-15-style case. Drop is post-opening but the black decision
    # ply is 19 (< opening_plies). position_state must reconstruct it normally.
    replay = legal_replay(22)
    assert replay["n_moves"] >= 20
    state = position_state(replay, 19, "black")   # 19 moves applied -> black to move
    assert state.to_move == "black"


def test_position_state_raises_on_out_of_range_ply():
    replay = legal_replay(8)
    with pytest.raises(ValueError, match="out of range"):
        position_state(replay, 99, "black")


def test_position_state_raises_on_side_to_move_mismatch():
    replay = legal_replay(8)
    # apply moves[0:4] -> red to move; claiming black must fail loud
    with pytest.raises(ValueError, match="side_to_move"):
        position_state(replay, 4, "black")
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_goal_line_trigger_probe_cases.py -k position_state`
Expected: ImportError (`position_state` not defined)

- [ ] **Step 4: Implement**

Append to `goal_line_trigger_probe_cases.py`:

```python
def position_state(replay, position_ply, side_to_move):
    """Board at the side-to-move's decision point: apply moves[0:position_ply]
    to a fresh TwixtState. Fail loud if the ply is out of range or the
    reconstructed side to move disagrees with the manifest."""
    moves = replay["moves"]
    if not (0 <= position_ply < len(moves)):
        raise ValueError(
            f"position_ply {position_ply} out of range [0, {len(moves)})")
    state = TwixtState(active_size=replay["board_size"], to_move="red",
                       max_plies_limit=replay["n_moves"])
    for m in moves[:position_ply]:
        state = state.apply_move((m["row"], m["col"]))
    if state.to_move != side_to_move:
        raise ValueError(
            f"reconstructed to_move {state.to_move!r} != side_to_move "
            f"{side_to_move!r} at position_ply {position_ply}")
    return state
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_goal_line_trigger_probe_cases.py`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/goal_line_trigger_probe_cases.py tests/goal_line_probe_fixtures.py tests/test_goal_line_trigger_probe_cases.py
git commit -m "feat(probe): position_state board reconstruction + legal-replay fixture"
```

---

### Task 3: `summarize` (per-checkpoint metrics)

**Files:**
- Modify: `scripts/GPU/alphazero/goal_line_trigger_probe_cases.py` (append)
- Modify: `tests/test_goal_line_trigger_probe_cases.py` (append)

- [ ] **Step 1: Write the failing tests**

Append (add `summarize` to the module import):

```python
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import summarize


def test_summarize_metrics_hand_computed():
    # values: two >= 0.5, one in [0.25,0.5), one below 0.25
    values = [0.8, 0.5, 0.3, -0.4]
    shares = [0.9, 0.8, 0.5, 0.2]
    s = summarize(values, shares)
    assert s["num_cases"] == 4
    assert s["mean_black_root_value"] == pytest.approx((0.8 + 0.5 + 0.3 - 0.4) / 4)
    assert s["median_black_root_value"] == pytest.approx(0.4)   # median(0.8,0.5,0.3,-0.4)
    assert s["black_overvalue_rate"] == 0.75                    # 3 of 4 >= 0.25
    assert s["severe_black_overvalue_rate"] == 0.5             # 2 of 4 >= 0.50
    assert s["mean_top1_share"] == pytest.approx((0.9 + 0.8 + 0.5 + 0.2) / 4)
    assert s["median_top1_share"] == pytest.approx(0.65)


def test_summarize_threshold_boundaries_inclusive():
    s = summarize([0.25, 0.50], [0.5, 0.5])
    assert s["black_overvalue_rate"] == 1.0      # 0.25 counts (>=)
    assert s["severe_black_overvalue_rate"] == 0.5  # only 0.50 counts


def test_summarize_empty_raises():
    with pytest.raises(ValueError, match="no cases"):
        summarize([], [])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_goal_line_trigger_probe_cases.py -k summarize`
Expected: ImportError

- [ ] **Step 3: Implement**

Append:

```python
def summarize(black_values, top1_shares):
    """Per-checkpoint metrics over parallel lists of black root_value + top1
    share. Overvalue thresholds are inclusive (>=)."""
    n = len(black_values)
    if n == 0:
        raise ValueError("no cases to summarize")
    over = sum(1 for v in black_values if v >= OVERVALUE_THRESHOLD)
    severe = sum(1 for v in black_values if v >= SEVERE_OVERVALUE_THRESHOLD)
    return {
        "num_cases": n,
        "mean_black_root_value": mean(black_values),
        "median_black_root_value": median(black_values),
        "black_overvalue_rate": over / n,
        "severe_black_overvalue_rate": severe / n,
        "mean_top1_share": mean(top1_shares),
        "median_top1_share": median(top1_shares),
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_goal_line_trigger_probe_cases.py`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/goal_line_trigger_probe_cases.py tests/test_goal_line_trigger_probe_cases.py
git commit -m "feat(probe): summarize per-checkpoint calibration metrics"
```

---

### Task 4: Mode-A manifest generator

**Files:**
- Create: `scripts/GPU/alphazero/generate_goal_line_trigger_probe_manifest.py`
- Create: `tests/test_goal_line_trigger_probe_cli.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_goal_line_trigger_probe_cli.py`:

```python
import csv
import json
from pathlib import Path

import pytest

from scripts.GPU.alphazero.generate_goal_line_trigger_probe_manifest import (
    build_manifest, main as gen_main,
)
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import DEFAULT_SELECTION

CANON_DIR = Path("logs/eval/loss_analysis_v2_1")
CANON_CANDIDATES = CANON_DIR / "goal_line_trigger_probe_candidates.csv"
CANON_MANIFEST = CANON_DIR / "goal_line_trigger_probe_manifest.json"

_CAND_HEADER = [
    "game_idx", "rank", "n_moves", "collapse_type", "largest_drop_phase",
    "trigger_zone", "prev_black_ply", "prev_black_row", "prev_black_col",
    "prev_black_value", "prev_black_top1", "trigger_red_ply", "trigger_red_row",
    "trigger_red_col", "trigger_red_value", "trigger_red_top1", "drop_black_ply",
    "drop_black_row", "drop_black_col", "drop_black_value", "drop_black_top1",
    "drop_amount", "replay_path",
]


def _write_candidates(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CAND_HEADER)
        w.writeheader()
        w.writerows(rows)


def _row(**over):
    base = dict(zip(_CAND_HEADER, [
        "769", "4", "45", "sharp_value_drop", "post_opening", "red_goal_band_3",
        "39", "21", "21", "0.88", "0.885", "40", "22", "22", "0.65", "0.955",
        "41", "18", "6", "-0.46", "0.08", "-1.34",
        "logs/eval/x_replays/game_000769.json"]))
    base.update(over)
    return base


def test_build_manifest_shape_and_selection_echo():
    rows = [_row(), _row(game_idx="1", prev_black_value="0.10")]  # 2nd filtered out
    m = build_manifest(rows, DEFAULT_SELECTION, "src.csv")
    assert m["schema_version"] == 1
    assert m["name"] == "goal_line_trigger_black_defense_probe"
    assert m["source"] == "src.csv"
    assert m["selection"] == DEFAULT_SELECTION
    assert m["num_cases"] == 1 and len(m["cases"]) == 1
    assert m["cases"][0]["game_idx"] == 769


def test_generator_cli_writes_manifest(tmp_path):
    csv_path = tmp_path / "cand.csv"
    _write_candidates(csv_path, [_row(), _row(game_idx="2")])
    out = tmp_path / "manifest.json"
    rc = gen_main(["--from-candidates-csv", str(csv_path), "--output", str(out)])
    assert rc == 0
    m = json.loads(out.read_text())
    assert m["num_cases"] == 2 and {c["game_idx"] for c in m["cases"]} == {769, 2}


@pytest.mark.skipif(not CANON_CANDIDATES.exists() or not CANON_MANIFEST.exists(),
                    reason="canonical loss_analysis_v2_1 artifacts not present")
def test_generator_reproduces_canonical_manifest(tmp_path):
    out = tmp_path / "regenerated.json"
    rc = gen_main(["--from-candidates-csv", str(CANON_CANDIDATES), "--output", str(out)])
    assert rc == 0
    got = json.loads(out.read_text())
    want = json.loads(CANON_MANIFEST.read_text())
    got_keys = [(c["game_idx"], c["position_ply"]) for c in got["cases"]]
    want_keys = [(c["game_idx"], c["position_ply"]) for c in want["cases"]]
    assert got_keys == want_keys and got["num_cases"] == want["num_cases"] == 18
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_goal_line_trigger_probe_cli.py`
Expected: ModuleNotFoundError (`generate_goal_line_trigger_probe_manifest`)

- [ ] **Step 3: Implement**

`scripts/GPU/alphazero/generate_goal_line_trigger_probe_manifest.py`:

```python
"""Mode A generator: goal-line trigger candidates CSV -> probe manifest JSON.

Reads the checked-in candidates CSV, applies the selection filter, and writes the
fixed probe manifest. Reproduces the canonical 18-case manifest from the
canonical candidates CSV.

Mode B (DEFERRED): a future generator may RE-DERIVE the candidates CSV by scanning
V2.1 collapse/replay outputs (collapse_timing.csv, drop_windows.csv, replays) and
classifying trigger zones, so candidates can be regenerated from any new capture.
Mode A intentionally consumes the checked/known candidates CSV so the probe target
stays fixed and reproducible.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .goal_line_trigger_probe_cases import DEFAULT_SELECTION, select_cases

MANIFEST_NAME = "goal_line_trigger_black_defense_probe"
MANIFEST_DESCRIPTION = (
    "Positions where a checkpoint as black confidently overvalued the position "
    "immediately before a red goal-line or near-goal-line trigger move.")


def build_manifest(candidate_rows, selection, source):
    cases = select_cases(candidate_rows, selection)
    return {
        "schema_version": 1,
        "name": MANIFEST_NAME,
        "source": source,
        "description": MANIFEST_DESCRIPTION,
        "selection": selection,
        "num_cases": len(cases),
        "cases": cases,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Mode A: goal-line trigger candidates CSV -> probe manifest.")
    p.add_argument("--from-candidates-csv", required=True, metavar="PATH")
    p.add_argument("--output", required=True, metavar="PATH")
    p.add_argument("--min-prev-black-value", type=float,
                   default=DEFAULT_SELECTION["min_prev_black_value"])
    p.add_argument("--min-prev-black-top1", type=float,
                   default=DEFAULT_SELECTION["min_prev_black_top1"])
    p.add_argument("--post-opening-only", action="store_true", default=True)
    p.add_argument("--no-post-opening-only", action="store_false",
                   dest="post_opening_only")
    p.add_argument("--trigger-zone-prefix",
                   default=DEFAULT_SELECTION["trigger_zone_prefix"])
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.from_candidates_csv) as f:
        rows = list(csv.DictReader(f))
    selection = {
        "min_prev_black_value": args.min_prev_black_value,
        "min_prev_black_top1": args.min_prev_black_top1,
        "post_opening_only": args.post_opening_only,
        "trigger_zone_prefix": args.trigger_zone_prefix,
    }
    manifest = build_manifest(rows, selection, args.from_candidates_csv)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest['num_cases']} cases -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_goal_line_trigger_probe_cli.py`
Expected: all passed (3 generator tests; the canonical one runs if data present)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/generate_goal_line_trigger_probe_manifest.py tests/test_goal_line_trigger_probe_cli.py
git commit -m "feat(probe): Mode-A manifest generator (candidates CSV -> manifest)"
```

---

### Task 5: probe evaluator CLI + FakeEvaluator integration

**Files:**
- Create: `scripts/GPU/alphazero/eval_goal_line_trigger_probe.py`
- Modify: `tests/test_goal_line_trigger_probe_cli.py` (append)

- [ ] **Step 1: Write the failing integration tests**

Append to `tests/test_goal_line_trigger_probe_cli.py` (add imports):

```python
from scripts.GPU.alphazero.eval_goal_line_trigger_probe import main as probe_main
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import EXPECTED_PROBLEM
from tests.eval_fakes import FakeEvaluator
from tests.goal_line_probe_fixtures import legal_replay


def _fake_factory(path):
    # Two distinct constant evaluators so the probe must produce different
    # per-checkpoint readouts. NOTE: FakeEvaluator's constant negates/clamps
    # through the negamax backup (+0.9 leaf -> root -0.9; <=0 leaf -> root ~0.0),
    # so these do NOT model real over/under-valuation. The real 0399-vs-0379
    # direction is the operator acceptance run (Task 6), not a fake unit test.
    return FakeEvaluator(value=0.9 if "0399" in path else 0.0)


def _write_probe_inputs(tmp_path, position_plies=(5, 7)):
    """Write sidecars + a manifest + two dummy checkpoint files; return paths."""
    rdir = tmp_path / "replays"
    rdir.mkdir()
    cases = []
    for i, pp in enumerate(position_plies):
        replay = legal_replay(pp + 3, game_idx=i)     # ensure n_moves > position_ply
        assert replay["moves"][pp]["player"] == "black"  # pp must be black's turn
        rpath = rdir / f"game_{i:06d}.json"
        rpath.write_text(json.dumps(replay))
        cases.append({
            "game_idx": i, "rank": i + 1, "replay_path": str(rpath),
            "position_ply": pp, "side_to_move": "black",
            "expected_problem": EXPECTED_PROBLEM, "trigger_red_ply": pp + 1,
            "trigger_red_move": {"row": 0, "col": 1}, "trigger_zone": "red_goal_row_exact",
            "baseline_black_prev_value": 0.7, "baseline_black_prev_top1": 0.9,
            "drop_black_ply": pp + 2, "drop_black_value": -0.5, "drop_amount": -1.2,
        })
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "name": "goal_line_trigger_black_defense_probe",
        "selection": DEFAULT_SELECTION, "num_cases": len(cases), "cases": cases}))
    ck = tmp_path / "ckpts"
    ck.mkdir()
    a = ck / "model_iter_0379.safetensors"; a.write_text("x")
    b = ck / "model_iter_0399.safetensors"; b.write_text("x")
    return manifest, a, b


def _run(tmp_path, manifest, a, b, outdir):
    return probe_main(
        ["--manifest", str(manifest), "--checkpoint", str(a), "--checkpoint", str(b),
         "--output-dir", str(outdir), "--mcts-sims", "12"],
        evaluator_factory=_fake_factory)


def test_probe_writes_summary_and_cases(tmp_path, capsys):
    manifest, a, b = _write_probe_inputs(tmp_path)
    out = tmp_path / "out"
    assert _run(tmp_path, manifest, a, b, out) == 0
    summary = json.loads((out / "goal_line_trigger_probe_summary.json").read_text())
    assert summary["num_cases"] == 2 and summary["mcts_sims"] == 12
    assert set(summary["checkpoints"]) == {"0379", "0399"}
    rows = list(csv.DictReader((out / "goal_line_trigger_probe_cases.csv").open()))
    assert len(rows) == 4                                   # 2 checkpoints x 2 cases
    assert {"checkpoint", "case_id", "probe_black_root_value", "probe_top1_share",
            "black_overvalue", "baseline_black_prev_value"} <= set(rows[0].keys())


def test_probe_distinguishes_checkpoints(tmp_path):
    manifest, a, b = _write_probe_inputs(tmp_path)
    out = tmp_path / "out"
    _run(tmp_path, manifest, a, b, out)
    s = json.loads((out / "goal_line_trigger_probe_summary.json").read_text())["checkpoints"]
    # Different evaluators -> different per-checkpoint readouts (the comparison
    # machinery works). Exact root values are an MCTS detail; the directional
    # 0399-overvalues-more-than-0379 readout is operator acceptance (Task 6),
    # not a constant-fake unit test.
    assert s["0399"]["mean_black_root_value"] != s["0379"]["mean_black_root_value"]


def test_probe_is_deterministic(tmp_path):
    manifest, a, b = _write_probe_inputs(tmp_path)
    o1, o2 = tmp_path / "o1", tmp_path / "o2"
    _run(tmp_path, manifest, a, b, o1)
    _run(tmp_path, manifest, a, b, o2)
    assert (o1 / "goal_line_trigger_probe_cases.csv").read_text() == \
           (o2 / "goal_line_trigger_probe_cases.csv").read_text()


def test_probe_missing_checkpoint_returns_2(tmp_path):
    manifest, a, _b = _write_probe_inputs(tmp_path)
    rc = probe_main(["--manifest", str(manifest), "--checkpoint", str(a),
                     "--checkpoint", str(tmp_path / "nope.safetensors"),
                     "--output-dir", str(tmp_path / "o")], evaluator_factory=_fake_factory)
    assert rc == 2
    assert not (tmp_path / "o").exists()


def test_probe_out_of_range_position_ply_raises(tmp_path):
    manifest, a, b = _write_probe_inputs(tmp_path)
    data = json.loads(manifest.read_text())
    data["cases"][0]["position_ply"] = 999
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="out of range"):
        _run(tmp_path, manifest, a, b, tmp_path / "o")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_goal_line_trigger_probe_cli.py -k probe`
Expected: ImportError (`eval_goal_line_trigger_probe`)

- [ ] **Step 3: Implement**

`scripts/GPU/alphazero/eval_goal_line_trigger_probe.py`:

```python
"""Goal-line trigger probe: re-evaluate fixed trigger positions across
checkpoints, measuring whether each overvalues black before red's goal-line
trigger move. Lower black root_value = better calibrated.

Run:
  .venv/bin/python -m scripts.GPU.alphazero.eval_goal_line_trigger_probe \
    --manifest logs/eval/loss_analysis_v2_1/goal_line_trigger_probe_manifest.json \
    --checkpoint checkpoints/.../model_iter_0379.safetensors \
    --checkpoint checkpoints/.../model_iter_0399.safetensors \
    --output-dir logs/eval/goal_line_trigger_probe --mcts-sims 400
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .eval_runner import (
    EvalConfig, cfg_from, short_id, _default_evaluator_factory,
)
from .mcts import MCTS
from .goal_line_trigger_probe_cases import (
    OVERVALUE_THRESHOLD, SEVERE_OVERVALUE_THRESHOLD, case_id, position_state,
    summarize,
)

REQUIRED_CASE_KEYS = ("game_idx", "replay_path", "position_ply", "side_to_move",
                      "trigger_zone", "baseline_black_prev_value",
                      "baseline_black_prev_top1")
CASE_CSV_COLUMNS = (
    "checkpoint", "game_idx", "case_id", "rank", "position_ply", "trigger_zone",
    "side_to_move", "baseline_black_prev_value", "baseline_black_prev_top1",
    "probe_black_root_value", "probe_top1_share", "black_overvalue",
    "severe_black_overvalue")


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"manifest schema_version != 1: {manifest.get('schema_version')}")
    cases = manifest.get("cases") or []
    if not cases:
        raise ValueError(f"manifest has no cases: {path}")
    for i, c in enumerate(cases):
        missing = [k for k in REQUIRED_CASE_KEYS if k not in c]
        if missing:
            raise ValueError(f"case {i}: missing keys {missing}")
    return manifest


def evaluate_case(evaluator, case, config, base_seed):
    """Reconstruct the case position and search it -> (black_value, top1_share)."""
    replay = json.loads(Path(case["replay_path"]).read_text())
    state = position_state(replay, case["position_ply"], case["side_to_move"])
    rng = random.Random(base_seed ^ case["game_idx"])
    counts, root_value = MCTS(evaluator, cfg_from(config), rng).search(
        state, add_noise=False)
    total = sum(counts.values())
    if total <= 0:
        raise ValueError(f"{case_id(case)}: empty search counts")
    return root_value, max(counts.values()) / total


def run_probe(manifest, checkpoints, config, base_seed, evaluator_factory):
    """Evaluate every case with every checkpoint -> (summary, case_rows)."""
    cases = manifest["cases"]
    per_ckpt, case_rows = {}, []
    for ckpt in checkpoints:
        evaluator = evaluator_factory(ckpt)        # one load, reused across cases
        sid = short_id(ckpt)
        values, shares = [], []
        for case in cases:
            v, t1 = evaluate_case(evaluator, case, config, base_seed)
            values.append(v)
            shares.append(t1)
            case_rows.append({
                "checkpoint": sid, "game_idx": case["game_idx"],
                "case_id": case_id(case), "rank": case.get("rank"),
                "position_ply": case["position_ply"],
                "trigger_zone": case["trigger_zone"],
                "side_to_move": case["side_to_move"],
                "baseline_black_prev_value": case["baseline_black_prev_value"],
                "baseline_black_prev_top1": case["baseline_black_prev_top1"],
                "probe_black_root_value": v, "probe_top1_share": t1,
                "black_overvalue": v >= OVERVALUE_THRESHOLD,
                "severe_black_overvalue": v >= SEVERE_OVERVALUE_THRESHOLD,
            })
        per_ckpt[sid] = summarize(values, shares)
    summary = {
        "manifest": manifest.get("name"),
        "num_cases": len(cases),
        "mcts_sims": config.mcts_sims,
        "base_seed": base_seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "checkpoints": per_ckpt,
    }
    return summary, case_rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Goal-line trigger calibration probe.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", action="append", default=[], required=True,
                   dest="checkpoints", metavar="PATH")
    p.add_argument("--output-dir", type=Path,
                   default=Path("logs/eval/goal_line_trigger_probe"))
    p.add_argument("--mcts-sims", type=int, default=400)
    p.add_argument("--mcts-eval-batch-size", type=int, default=14)
    p.add_argument("--mcts-stall-flush-sims", type=int, default=48)
    p.add_argument("--base-seed", type=int, default=20260614)
    return p.parse_args(argv)


def main(argv=None, evaluator_factory=None):
    args = parse_args(argv)
    factory = evaluator_factory or _default_evaluator_factory
    for ckpt in args.checkpoints:               # fail before the long MLX load
        if not Path(ckpt).exists():
            print(f"error: checkpoint not found: {ckpt}", file=sys.stderr)
            return 2
    manifest = load_manifest(args.manifest)
    config = EvalConfig(mcts_sims=args.mcts_sims,
                        mcts_eval_batch_size=args.mcts_eval_batch_size,
                        mcts_stall_flush_sims=args.mcts_stall_flush_sims)
    summary, case_rows = run_probe(manifest, args.checkpoints, config,
                                   args.base_seed, factory)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "goal_line_trigger_probe_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    with (out / "goal_line_trigger_probe_cases.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CASE_CSV_COLUMNS))
        w.writeheader()
        w.writerows(case_rows)
    for sid, m in summary["checkpoints"].items():
        print(f"{sid}: overvalue_rate={m['black_overvalue_rate']:.1%} "
              f"mean_black_value={m['mean_black_root_value']:+.3f} "
              f"(severe {m['severe_black_overvalue_rate']:.1%})")
    print(f"summary -> {out / 'goal_line_trigger_probe_summary.json'}")
    print(f"cases   -> {out / 'goal_line_trigger_probe_cases.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_goal_line_trigger_probe_cli.py`
Expected: all passed (generator + probe integration)

- [ ] **Step 5: Run the full new-test surface**

Run: `.venv/bin/python -m pytest tests/test_goal_line_trigger_probe_cases.py tests/test_goal_line_trigger_probe_cli.py`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/eval_goal_line_trigger_probe.py tests/test_goal_line_trigger_probe_cli.py
git commit -m "feat(probe): goal-line trigger probe evaluator CLI (+ FakeEvaluator tests)"
```

---

### Task 6: Regression + real-data operator acceptance

**Files:** none created; verification only.

- [ ] **Step 1: Full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: no NEW failures vs the branch point. (Known pre-existing failures predate this work — missing Replays/ data, JS-parity, Node-oracle groups; compare against the merge base if unsure.)

- [ ] **Step 2: Regenerate the manifest from the canonical candidates (reproducibility check)**

Run:
```bash
.venv/bin/python -m scripts.GPU.alphazero.generate_goal_line_trigger_probe_manifest \
  --from-candidates-csv logs/eval/loss_analysis_v2_1/goal_line_trigger_probe_candidates.csv \
  --output /tmp/regen_manifest.json
.venv/bin/python -c "import json; a=json.load(open('/tmp/regen_manifest.json')); b=json.load(open('logs/eval/loss_analysis_v2_1/goal_line_trigger_probe_manifest.json')); ak=[(c['game_idx'],c['position_ply']) for c in a['cases']]; bk=[(c['game_idx'],c['position_ply']) for c in b['cases']]; print('cases match canonical:', ak==bk, '| n =', a['num_cases'])"
```
Expected: `cases match canonical: True | n = 18`

- [ ] **Step 3: Operator acceptance — the real probe run (real MLX checkpoints)**

Run exactly the spec's command:
```bash
.venv/bin/python -m scripts.GPU.alphazero.eval_goal_line_trigger_probe \
  --manifest logs/eval/loss_analysis_v2_1/goal_line_trigger_probe_manifest.json \
  --checkpoint checkpoints/alphazero-v2-staged/model_iter_0379.safetensors \
  --checkpoint checkpoints/alphazero-v2-eps035-from0379/model_iter_0399.safetensors \
  --output-dir logs/eval/goal_line_trigger_probe \
  --mcts-sims 400
```
Expected:
- Two console lines (`0379:` and `0399:`) + the two output paths.
- `logs/eval/goal_line_trigger_probe/goal_line_trigger_probe_summary.json` and `..._cases.csv` written; summary has 18 `num_cases` and both checkpoint blocks.
- **The readout:** `0399` shows a higher `black_overvalue_rate` / `mean_black_root_value` than `0379` (the calibration gap the probe exists to detect). Sanity: `0399`'s `probe_black_root_value` per case should track the `baseline_black_prev_value` column (same net, same positions, modulo MCTS seeding).

- [ ] **Step 4: Stop — do not commit logs/ outputs**

Probe outputs under `logs/` stay untracked (matches the eval convention). Nothing to commit in this step unless a bug was found and fixed.

---

## Self-review notes (already applied)

- **Spec coverage:** signature/position semantics (Tasks 2,5), `select_cases` filter incl. `post_opening_only`-on-drop and the game-15 `position_ply<opening_plies` boundary (Tasks 1,2), candidate→case mapping (Task 1), `summarize` metric list (Task 3), Mode-A generator reproducing the 18 (Task 4), probe MCTS evaluation + outputs (Task 5), fail-loud paths (Tasks 2,5), determinism/seeding (Task 5), operator acceptance (Task 6). Mode B is explicitly out of scope (documented in the generator docstring).
- **Type consistency:** `select_cases`/`position_state`/`summarize`/`case_id` signatures and the manifest case keys (`position_ply`, `baseline_black_prev_value`, `trigger_red_move`, …) match across the generator and probe; `CASE_CSV_COLUMNS` and `REQUIRED_CASE_KEYS` reference only keys produced by `select_cases`.
- **Determinism / fake semantics:** `legal_replay` always takes the first legal move; the probe seeds `base_seed ^ game_idx` with `add_noise=False`; FakeEvaluator is constant-valued. **Verified:** a constant FakeEvaluator value negates/clamps through the negamax backup (`+0.9` leaf → root `−0.9`; `≤0` leaf → root `~0.0`), so the fake tests assert only structure, determinism, and that two evaluators **differ** — never a value direction. The real `0399`-overvalues-more-than-`0379` readout is operator acceptance (Task 6). `short_id` on the test paths is verified to yield `"0379"`/`"0399"`.
