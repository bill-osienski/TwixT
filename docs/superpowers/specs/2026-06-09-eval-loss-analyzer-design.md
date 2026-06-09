# Eval Loss Analyzer (V1) — Design

**Date:** 2026-06-09
**Status:** Approved, pending implementation plan
**Author:** bill + Claude

## Purpose

A read-only postprocessor over checkpoint-eval `*_games.jsonl` files that explains
**how checkpoint A is losing to checkpoint B** — by color, by game length, by
termination, and across training branches. It answers the immediate question for the
three failed branches (`control`, `eps035`, `lr0003_eps035`) versus anchor `0379`:

- Is A worse overall?
- Is A worse as red or as black?
- Is A worse in short, mid, long, or cap-length games?
- Which concrete games should I inspect?
- How do the branches rank against each other?

### Explicit non-goals (V1)

The current `*_games.jsonl` row schema carries only game-level metadata (no move
history, no per-move search stats). V1 therefore **cannot** analyze opening clusters,
first moves, root values, or collapse timing, and must not pretend to. Move-level
analysis is deferred to a future V2 that adds replay capture to the eval runner.

## Input schema (ground truth)

One JSON object per line in `*_games.jsonl`. Confirmed fields:

```json
{
  "task_id": 0,
  "pairing_id": "0399_vs_0379",
  "game_idx": 0,
  "red_checkpoint": ".../model_iter_0399.safetensors",
  "black_checkpoint": ".../model_iter_0379.safetensors",
  "winner": "red",                       // "red" | "black" | null
  "winner_checkpoint": ".../model_iter_0399.safetensors",   // null on draw
  "reason": "win",                       // "win" | "state_cap" | "board_full" | "unknown_error"
  "n_moves": 75,
  "red_score": 1.0,
  "black_score": 0.0
}
```

Observed invariants (verified against `lr0003_eps035_0399_vs_0379_800g_w4_games.jsonl`,
800 rows): `state_cap` games have `n_moves == 280` exactly (`== max_moves`);
`board_full == 0` in current data; draws (`winner is null`) score `0.5/0.5`.

### Optional sidecar summary

A sibling `<stem>.json` (written by `eval_summary.summarize_match`) may exist. It
exposes the A/B identity under **`checkpoint_a` / `checkpoint_b`** (NOT
`a_checkpoint`/`b_checkpoint`; there is no `a_label`/`b_label`). When present it is the
preferred source of A/B identity.

## Scoring policy (must match `eval_summary.py`)

- A/B score is keyed off **`winner_checkpoint`**, never off color:
  ```python
  def score_for_checkpoint(row, ckpt):
      if row["winner_checkpoint"] == ckpt: return 1.0
      if row["winner_checkpoint"] is None: return 0.5   # state_cap / board_full
      return 0.0
  ```
- Color stats key off **`red_checkpoint` / `black_checkpoint`** + `winner`.
- `state_cap` and `board_full` both count as `0.5` for both sides
  (`DRAW_SCORE_POLICY = "state_cap_and_board_full_score_0.5"`).

All Elo/CI/verdict math is **reused** from `eval_elo.py` — no reimplementation:
`score_rate(wins, draws_plus_caps, total)`, `elo_diff(p, n)`,
`score_ci_trinomial(w, d, l)`, `elo_ci(w, d, l)`, `verdict(rate)`. Note `verdict` has
four levels: `stronger` (≥0.55), `weak_signal` (≥0.52), `tied` (≥0.48), `worse` (else).

## Architecture

Pure module + thin CLI, mirroring the existing `eval_elo` (pure) / `eval_runner` (IO)
split. The biggest risk in this analyzer is subtly conflating
`winner == "red"` vs `winner_checkpoint == A` vs `red_checkpoint == A`; a pure,
IO-free module lets us unit-test those cases directly.

```
scripts/GPU/alphazero/eval_loss_analysis.py   # pure logic: dicts in, dicts/lists out
scripts/GPU/alphazero/eval_loss_analyzer.py   # thin CLI: argparse + file IO + console
tests/test_eval_loss_analysis.py              # unit tests (flat tests/ dir, per convention)
```

Import convention: `from scripts.GPU.alphazero.eval_loss_analysis import ...`.

### Pure module — `eval_loss_analysis.py`

IO-free. Accepts rows as plain dicts; returns plain dicts/lists ready to serialize.

```python
def validate_rows(rows: list[dict]) -> None: ...
def resolve_checkpoints(rows, pairing_id=None, a_override=None,
                        b_override=None, summary=None) -> tuple[str, str]: ...
def score_for_checkpoint(row: dict, ckpt: str) -> float: ...
def a_color(row: dict, a_ckpt: str) -> str: ...                  # "red" | "black"
def summarize_overall(rows, a_ckpt, b_ckpt) -> dict: ...
def summarize_by_color(rows, a_ckpt, b_ckpt) -> list[dict]: ...
def summarize_by_length(rows, a_ckpt, b_ckpt, buckets) -> list[dict]: ...
def sample_worst_losses(rows, a_ckpt, b_ckpt, limit) -> list[dict]: ...
def analyze_match(rows, a_ckpt, b_ckpt, *,
                  length_buckets=(40, 60, 80, 120, 279, 280),
                  worst_losses=50) -> dict: ...
def combine_branch_summaries(match_summaries: list[dict]) -> list[dict]: ...
```

### Thin CLI — `eval_loss_analyzer.py`

```python
def parse_args() -> argparse.Namespace: ...
def load_jsonl(path: Path) -> list[dict]: ...
def load_sibling_summary(path: Path) -> dict | None: ...   # <stem>.json if present
def write_json(path: Path, obj: dict) -> None: ...
def write_csv(path: Path, rows: list[dict]) -> None: ...
def main() -> int: ...
```

CLI does only: resolve input files → load rows → load sibling summary → call
`analyze_match()` → write JSON/CSV → print console summary → emit combined CSV.

## Data flow

```
*_games.jsonl ─┐
               ├─ load_jsonl ─ validate_rows ─ resolve_checkpoints ─ analyze_match ─┬─ write per-match outputs
<stem>.json  ──┘                                                                    └─ console summary
                                                                          (collect match summaries)
                                                                                    └─ combine_branch_summaries ─ combined CSV
```

## CLI

```bash
# explicit files
.venv/bin/python -m scripts.GPU.alphazero.eval_loss_analyzer \
  --games-jsonl logs/eval/eps035_0399_vs_0379_800g_w4_games.jsonl \
  --games-jsonl logs/eval/lr0003_eps035_0399_vs_0379_800g_w4_games.jsonl \
  --output-dir logs/eval/loss_analysis

# glob
.venv/bin/python -m scripts.GPU.alphazero.eval_loss_analyzer \
  --glob "logs/eval/*0399_vs_0379*_games.jsonl" \
  --output-dir logs/eval/loss_analysis
```

Arguments:
- `--games-jsonl PATH` (repeatable) and/or `--glob PATTERN` — inputs
- `--output-dir PATH` — default `logs/eval/loss_analysis/`
- `--a-checkpoint PATH`, `--b-checkpoint PATH` — optional A/B override
- `--length-buckets 40,60,80,120,279,280` — upper-inclusive edges
- `--worst-losses 50` — sampler size

## Outputs (Lean + combined)

Per match, into `--output-dir`, named by stem (`*_games.jsonl` → `<stem>`):

| File | Content |
|------|---------|
| `<stem>_loss_summary.json` | overall: a_wins/b_wins/draws, a_score_rate, elo + CI, verdict, color gap, **`termination` counts block**, nested by-color + by-length |
| `<stem>_by_color.csv` | A-as-red / A-as-black rows |
| `<stem>_by_length.csv` | per length bucket |
| `<stem>_worst_losses.csv` | worst-loss sampler (game_idx for manual inspection) |

Cross-branch: `combined_branch_comparison.csv` — one row per match, **sorted
descending by `a_score_rate`** (strongest-vs-anchor branch first).

Console: human summary with overall score/elo/verdict, by-color gap, by-length table,
cap/board-full rates, and a "Likely loss shape" line from simple heuristics.

**No standalone `by_reason.csv`.** Termination counts (`win` / `state_cap` /
`board_full` / `unknown_error`) live inside `<stem>_loss_summary.json` under
`"termination"` and in the console. A by-reason table carries little diagnostic signal
because `state_cap`/`board_full` are always `0.5/0.5` by definition and `win` just
mirrors the decisive A/B result.

### Table shapes

`by_color.csv`: `match,a_color,games,a_score_rate,a_wins,b_wins,draws,avg_moves`
`by_length.csv`: `match,length_bucket,games,a_score_rate,a_wins,b_wins,draws,avg_moves`
`worst_losses.csv`: `match,game_idx,task_id,a_color,winner,reason,n_moves,a_score,red_checkpoint,black_checkpoint`
`combined_branch_comparison.csv`: `match,pairing_id,a_checkpoint,b_checkpoint,games,a_score_rate,a_wins,b_wins,draws,elo,verdict`

### Console "Likely loss shape" heuristics

```
short_loss_signal = score_rate(<=60)  < overall_rate - 0.03
long_loss_signal  = score_rate(81+)   < overall_rate - 0.03
color_signal      = abs(a_red_rate - a_black_rate) >= 0.05
state_cap_signal  = state_cap_rate >= 0.05
```

## Edge-case rulings (the parts the original sketch left open)

1. **A/B resolution order** (first that succeeds): `--a/--b-checkpoint` flags →
   sidecar `<stem>.json` `checkpoint_a`/`checkpoint_b` → fallback infer from
   `pairing_id` split on `_vs_` matched against `short_id(red/black_checkpoint)`.
   After resolving, assert both A and B actually appear across rows.

2. **Length buckets** are upper-inclusive edges. Default `(40,60,80,120,279,280)` →
   labels `<=40, 41-60, 61-80, 81-120, 121-279, 280`. The `279`/`280` split
   deliberately isolates state-caps (`n_moves == 280`) from long decisive games.

3. **Self-match** (`a == b`, e.g. `0419_vs_0419` sanity files): no "A loses to B"
   meaning. Skip with a printed note; exclude from `combined_branch_comparison.csv`.

4. **`unknown_error` rows**: none exist in current data. V1 **fails loud** if any
   appear rather than guess a score. Handling is added when one is actually observed.

5. **`combined_branch_comparison.csv` ordering**: descending by `a_score_rate`.

## Error handling

`validate_rows` fails loud (raises) on:
- missing required keys
- `winner ∉ {"red", "black", None}`
- `winner == "red"` but `winner_checkpoint != red_checkpoint` or scores ≠ `1.0/0.0`
- `winner == "black"` but `winner_checkpoint != black_checkpoint` or scores ≠ `0.0/1.0`
- `winner is None` but `winner_checkpoint is not None`, scores ≠ `0.5/0.5`, or
  `reason ∉ {"state_cap", "board_full"}`
- `reason == "unknown_error"` (see ruling 4)

## Testing (TDD — written red first)

Pure-module unit tests in `tests/test_eval_loss_analysis.py`:
- `test_score_for_checkpoint_win_red_a`
- `test_score_for_checkpoint_win_black_a`
- `test_score_for_checkpoint_draw_state_cap`
- `test_by_color_uses_checkpoint_assignment_not_winner_color`
- `test_by_length_buckets_280_state_cap`
- `test_combined_branch_comparison_orders_by_a_score_rate`
- `test_validation_rejects_inconsistent_draw_scores`
- `test_validation_rejects_winner_checkpoint_mismatch`
- A/B resolution: flag override / sidecar / pairing-id fallback
- self-match skip

## Future (V2 — out of scope)

Move-level analysis requires replay capture in the eval runner: a `--save-eval-games`
flag writing per-game replay sidecars (`<match>_replays/game_NNNNNN.json`) plus a
`replay_path` field in each `*_games.jsonl` row. That unlocks opening-cluster analysis,
first-N-ply score tables, loss-by-opening-coordinate, and collapse-from-value timing.
Build only after V1's loss-shape answers say it's warranted.
