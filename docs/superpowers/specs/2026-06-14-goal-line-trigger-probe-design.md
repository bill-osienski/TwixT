# Goal-Line Trigger Probe — Design

**Date:** 2026-06-14
**Status:** Approved, pending implementation plan
**Author:** bill + Claude
**Related:** V2.1 replay analyzer (`docs/superpowers/specs/2026-06-12-eval-replay-analyzer-design.md`),
operator guide (`docs/post-game-analysis.md`)

## Purpose

A **fast checkpoint-level calibration probe**: take a fixed set of "goal-line
trigger" positions — where a net, playing black, confidently overvalued the
position one ply *before* red's goal-line-completing move — and re-evaluate each
position with one or more checkpoints to measure whether the checkpoint
**overvalues black before red's trigger**. Lower black value = better
calibrated. The bad signature is **black `root_value` strongly positive at the
trigger position**.

This is a ~18-position diagnostic that runs in seconds-to-minutes per checkpoint,
giving a targeted read on the value-head failure the V2.1 analysis surfaced
(0399's post-opening value cliffs) *without* a fresh 800-game eval. Expected: the
eps035 `0399` checkpoint looks bad here; `0379` looks more cautious if it truly
sees the goal-line danger earlier.

## The signature (what a "case" is)

V2.1 found that 0399-as-black loses via sharp post-opening value drops. Each
selected case is the three-ply window around one such drop:

| ply | who | meaning |
|---|---|---|
| `position_ply` (odd) | **black** | black's decision point; source net valued black at `baseline_black_prev_value` (≥ +0.25, confident). **This is the position the probe evaluates.** |
| `trigger_red_ply` = `position_ply + 1` | red | red's goal-line / near-goal trigger move (`trigger_zone` ∈ red goal band). |
| `drop_black_ply` = `position_ply + 2` | black | black's value craters (`drop_black_value`); `drop_amount` is the cliff. |

The probe reconstructs the board at `position_ply` (black to move) and asks each
checkpoint: *how do you value black here?* A well-calibrated net should already
see red's goal-line threat and **not** sit at a high positive value.

## Scope

- **In (this task):**
  - `eval_goal_line_trigger_probe.py` — the probe evaluator (the operator's run target).
  - `generate_goal_line_trigger_probe_manifest.py` — **Mode A** generator: the
    existing `goal_line_trigger_probe_candidates.csv` → manifest (reproduces the
    canonical 18 cases).
  - `goal_line_trigger_probe_cases.py` — pure helpers shared by both.
  - Tests for all three; operator acceptance run on `0379` + `0399`.
- **Out (deferred — Mode B):** a `generate_goal_line_trigger_candidates.py` that
  *re-derives* the candidates CSV by scanning V2.1 collapse/replay outputs
  (`collapse_timing.csv`, `drop_windows.csv`, replays) and classifying trigger
  zones, so candidates can be regenerated from any new capture. Mode A
  intentionally consumes the checked-in candidates CSV so the probe target stays
  fixed and reproducible. The generator and spec carry a `Mode B deferred` note.

## Canonical artifacts

The probe inputs live under `logs/eval/loss_analysis_v2_1/`:

- `goal_line_trigger_probe_candidates.csv` — selection source (26 rows).
- `goal_line_trigger_probe_manifest.json` — the fixed 18-case probe (schema below).

**Root-level copies are not canonical and must not be committed** (the stale root
duplicates were removed as part of this work). Probe/generator outputs default to
`logs/eval/goal_line_trigger_probe/` per the run command.

## Manifest schema (the contract — already exists, v1)

```json
{
  "schema_version": 1,
  "name": "goal_line_trigger_black_defense_probe",
  "source": "logs/eval/loss_analysis_v2_1/goal_line_trigger_probe_candidates.csv",
  "description": "Positions where ... black confidently overvalued ... before a red goal-line ... trigger move.",
  "selection": {
    "min_prev_black_value": 0.25,
    "min_prev_black_top1": 0.5,
    "post_opening_only": true,
    "trigger_zone_prefix": "red_goal"
  },
  "num_cases": 18,
  "cases": [
    {
      "game_idx": 769, "rank": 4,
      "replay_path": "logs/eval/eps035_..._replays/game_000769.json",
      "position_ply": 39, "side_to_move": "black",
      "expected_problem": "black_overvalues_red_goal_trigger",
      "trigger_red_ply": 40, "trigger_red_move": {"row": 22, "col": 22},
      "trigger_zone": "red_goal_band_3",
      "baseline_black_prev_value": 0.8797, "baseline_black_prev_top1": 0.885,
      "drop_black_ply": 41, "drop_black_value": -0.4644, "drop_amount": -1.3441
    }
  ]
}
```

The probe treats this as a read-only contract. New/unknown keys are ignored; the
keys it *requires* per case are `game_idx`, `replay_path`, `position_ply`,
`side_to_move`, `trigger_zone`, `baseline_black_prev_value`,
`baseline_black_prev_top1` (the rest are carried through to the cases CSV for
context).

## Candidates CSV schema (Mode A input)

Columns (26 rows): `game_idx`, `rank`, `n_moves`, `collapse_type`,
`largest_drop_phase`, `trigger_zone`, `prev_black_ply`, `prev_black_row`,
`prev_black_col`, `prev_black_value`, `prev_black_top1`, `trigger_red_ply`,
`trigger_red_row`, `trigger_red_col`, `trigger_red_value`, `trigger_red_top1`,
`drop_black_ply`, `drop_black_row`, `drop_black_col`, `drop_black_value`,
`drop_black_top1`, `drop_amount`, `replay_path`.

## Architecture

### `goal_line_trigger_probe_cases.py` (pure — no MLX, no IO)

- `select_cases(candidate_rows, selection) -> list[case]` — apply the filter
  (`prev_black_value >= min_prev_black_value`, `prev_black_top1 >=
  min_prev_black_top1`, `largest_drop_phase == "post_opening"` when
  `post_opening_only`, `trigger_zone.startswith(trigger_zone_prefix)`) and map
  each surviving candidate row → the manifest `case` dict (field mapping:
  `prev_black_ply → position_ply`, `{trigger_red_row, trigger_red_col} →
  trigger_red_move`, `prev_black_value → baseline_black_prev_value`, etc.;
  `side_to_move = "black"`, `expected_problem =
  "black_overvalues_red_goal_trigger"`). Order preserved from the CSV (already
  rank-sorted).
- `case_id(case) -> str` — `f"game_{game_idx:06d}_ply_{position_ply}"`.
- `position_state(replay, position_ply, side_to_move) -> TwixtState` — build
  `TwixtState(active_size=replay["board_size"], to_move="red",
  max_plies_limit=replay["n_moves"])`, apply `moves[0:position_ply]` as
  `(row, col)`, and **fail loud** unless the resulting `to_move == side_to_move`
  and `0 <= position_ply < len(moves)`.
- `summarize(black_values, top1_shares) -> dict` — the per-checkpoint metrics:
  `num_cases`, `mean_black_root_value`, `median_black_root_value`,
  `black_overvalue_rate` (fraction with `value >= 0.25`),
  `severe_black_overvalue_rate` (fraction with `value >= 0.50`),
  `mean_top1_share`, `median_top1_share`.

`TwixtState` is pure Python, so `position_state` is unit-testable without MLX.

### `generate_goal_line_trigger_probe_manifest.py` (CLI)

Mode A: `--from-candidates-csv <csv> --output <manifest.json>` plus the four
selection knobs (`--min-prev-black-value 0.25`, `--min-prev-black-top1 0.5`,
`--post-opening-only` default true, `--trigger-zone-prefix red_goal`). Reads the
candidates CSV, runs `select_cases`, writes the manifest dict (with `source`,
`selection`, `num_cases`, `cases`). A module docstring carries the **Mode B
deferred** note. Acceptance: regenerating from the canonical candidates CSV
reproduces the canonical 18-case manifest (same `game_idx`/`position_ply` set and
order).

### `eval_goal_line_trigger_probe.py` (CLI — the run target)

```
.venv/bin/python -m scripts.GPU.alphazero.eval_goal_line_trigger_probe \
  --manifest logs/eval/loss_analysis_v2_1/goal_line_trigger_probe_manifest.json \
  --checkpoint checkpoints/alphazero-v2-staged/model_iter_0379.safetensors \
  --checkpoint checkpoints/alphazero-v2-eps035-from0379/model_iter_0399.safetensors \
  --output-dir logs/eval/goal_line_trigger_probe \
  --mcts-sims 400
```

Flow: load manifest → for each `--checkpoint` (repeatable): build one evaluator
via `eval_runner._default_evaluator_factory(path)` (the `compile=True` loader,
reused once per checkpoint); for each case: `position_state(...)`, then
`MCTS(evaluator, cfg, random.Random(base_seed ^ game_idx)).search(state,
add_noise=False)` → `(counts, root_value)`; record `probe_black_root_value =
root_value` (black to move, so it is already black's perspective — no sign flip)
and `probe_top1_share = max(counts.values()) / sum(counts.values())`.

The search config is built the same way eval games build it — reuse
`eval_runner.EvalConfig` + `cfg_from(...)` (or an equivalent `MCTSConfig`),
parameterized by `--mcts-sims` (default 400) with the eval defaults for
`mcts_eval_batch_size` / `mcts_stall_flush_sims`. `--base-seed` (default fixed)
makes per-case search reproducible.

**Other args:** `--manifest` (required), `--checkpoint` (repeatable, required),
`--output-dir` (default `logs/eval/goal_line_trigger_probe`), `--mcts-sims`
(400), `--base-seed`.

**Reuse, not reinvention:** `_default_evaluator_factory`, `EvalConfig`/`cfg_from`,
`MCTS`/`MCTSConfig`, `TwixtState` are all imported from the existing eval modules.
A `run_probe(manifest, checkpoints, config, evaluator_factory=...)` core takes an
injectable `evaluator_factory` (default the real loader) so tests pass a
`FakeEvaluator` factory — mirroring `eval_runner.run_game_tasks`.

## Outputs

Written to `--output-dir`:

- **`goal_line_trigger_probe_summary.json`** — `{manifest, num_cases, mcts_sims,
  base_seed, generated_at, git_commit, checkpoints: {<short_id>: {num_cases,
  mean_black_root_value, median_black_root_value, black_overvalue_rate,
  severe_black_overvalue_rate, mean_top1_share, median_top1_share}}}`.
- **`goal_line_trigger_probe_cases.csv`** — one row per (checkpoint, case):
  `checkpoint` (short id), `game_idx`, `case_id`, `rank`, `position_ply`,
  `trigger_zone`, `side_to_move`, `baseline_black_prev_value`,
  `baseline_black_prev_top1`, `probe_black_root_value`, `probe_top1_share`,
  `black_overvalue` (`probe value >= 0.25`), `severe_black_overvalue`
  (`>= 0.50`). The `baseline_*` columns let you sanity-check that the
  source-checkpoint (0399) probe value ≈ its in-game value, and that the
  baseline-checkpoint (0379) reads lower.

Console: a short per-checkpoint summary line (`overvalue_rate`,
`mean_black_root_value`) and the two output paths.

## Validation & error handling (fail-loud, eval-module style)

- Manifest missing / unparseable / `schema_version != 1` / empty `cases` → raise.
- Per case: `replay_path` missing or unreadable → raise naming the case;
  `position_ply` out of `[0, n_moves)` → raise; reconstructed `to_move !=
  side_to_move` → raise (catches a drifted manifest or wrong ply).
- A `--checkpoint` path that does not exist → raise *before* the run starts
  (check all paths up front; a long MLX load failing mid-run is worse).
- Empty `counts` after search → raise (should be impossible).

## Determinism / performance

Per-case search is seeded (`base_seed ^ game_idx`) and uses `add_noise=False`, so
re-runs match. One evaluator is built per checkpoint and reused across all 18
cases (avoids repeated MLX/Metal load). 18 cases × N checkpoints × `mcts_sims`
searches is small — the probe is deliberately cheap.

## Testing (TDD)

Pure (`tests/test_goal_line_trigger_probe_cases.py`, no MLX):

- `select_cases` on the **real canonical** candidates CSV + the canonical
  `selection` reproduces exactly the canonical manifest's
  `(game_idx, position_ply)` set and order (the Mode-A acceptance gate, no MLX).
- `select_cases` filter boundaries on synthetic rows (each of the four knobs;
  `>=` boundaries at 0.25 / 0.5; `post_opening_only`; zone prefix).
- candidate→case field mapping (every manifest key).
- `position_state`: synthetic replay → applies `moves[0:position_ply]`, asserts
  `to_move == "black"`; raises on out-of-range ply and on a parity/`side_to_move`
  mismatch.
- `summarize`: hand-computed mean/median + the `>= 0.25` / `>= 0.50` rate
  boundaries; empty input guarded.

Integration (`tests/test_goal_line_trigger_probe_cli.py`, `FakeEvaluator`, no MLX):

- Generator Mode A: a small synthetic candidates CSV → manifest with the right
  schema, `selection` echoed, `num_cases` correct.
- Probe: synthetic manifest + sidecars + a `FakeEvaluator` factory → both output
  files with the documented columns/shape; per-checkpoint metrics computed;
  identical across two runs (seeded determinism).
- Fail-loud paths: missing replay, out-of-range `position_ply`, `to_move`
  mismatch, nonexistent checkpoint path.

Operator acceptance (real MLX, manual): run on `0379` + `0399`, `--mcts-sims 400`.
Confirm 18 cases, that `0399` shows a high `black_overvalue_rate` /
`mean_black_root_value` (and its probe values ≈ the `baseline_*` columns), and
that `0379` reads lower — the calibration gap the probe exists to detect.
