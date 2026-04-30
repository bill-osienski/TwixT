# Replay Analyzer Per-Game Stats Surfacing

**Date:** 2026-04-29
**Author:** brainstormed with Bill
**Status:** Design approved, ready for implementation
**Touches:** `scripts/twixt_replay_analyzer.py`, `tests/test_analyzer_per_game_stats.py` (new)
**Predecessor:** `docs/superpowers/specs/2026-04-29-per-game-stats-persistence-design.md` — that work persisted the data; this work surfaces it for human consumption.

## 1. Problem

The per-game stats persistence work (shipped earlier today) writes eight new fields per game into `scripts/GPU/logs/games/iter_NNNN_game_NNN.json`:

- Five flat diagnostic fields: `worker_id`, `wall_time_s`, `adjudication_block_reason`, `final_root_value`, `final_top1_share`
- A nested `compute` block: `leaf_evals`, `backups`, `nn_batches`

The data is on disk and well-formed but invisible to `scripts/twixt_replay_analyzer.py`, the tool the operator uses to review training runs (`summary.json` and `report.txt`). The original persistence spec was scoped narrowly to disk format and explicitly listed "no new aggregations" as a non-goal; analyzer integration was an implicit follow-up.

The operator's stated workflow is: read `summary.json` and `report.txt` after a run. If the new per-game data isn't surfaced there, it is effectively unused.

## 2. Goals

1. Make the five new diagnostic fields and the per-game compute counters visible in `summary.json` as a new `per_game_stats` block.
2. Render a human-readable summary of the same in `report.txt`, placed near the existing `Compute:` line so totals (sidecar) and distributions (per-game) sit together.
3. Surface coverage (`n_games_with_stats`) so a partial-coverage run (mixed old + new replays) is self-evident in the output.
4. Support worker imbalance diagnostics (per-worker totals + max/min wall-time ratio) so stragglers in multi-worker self-play are visible at a glance.
5. Confine all changes to `scripts/twixt_replay_analyzer.py`. No new analyzer modules.

## 3. Non-goals

- **No changes to the persistence layer.** All upstream work (MCTS, GameRecord, GameComplete, game_saver, trainer routing) stays exactly as shipped.
- **No new sidecar aggregations.** The sidecar `compute` block already carries totals; the new block is per-game distributions, derived from replays.
- **No `adjudication_block_reason` per-game histogram.** The sidecar already aggregates this as `adjudication.blocks.{ply,threshold,visits,top1}` and `report.txt` already prints it. Per-game version would be redundant.
- **No new heatmap figures or matplotlib plots.** Text-first; figures are a future enhancement if needed.
- **No CSV emitters.** Existing CSV outputs (`replay_cap_by_iter.csv` etc.) are unaffected. Adding per-game-stats CSVs is a separate decision.
- **No backfill of old games.** Old replays without the new fields are silently excluded from per-stat aggregations and reflected in `n_games_with_stats`.

## 4. JSON schema (`summary.json` `per_game_stats` block)

A single new top-level key in `summary.json`, sibling of the existing `compute` block:

```jsonc
"per_game_stats": {
  "n_games_total": 1234,                  // total replays loaded
  "n_games_with_stats": 1234,             // games with at least one of the new fields populated
  "wall_time_s": {                        // null when n_games_with_stats == 0 or no game has wall_time_s
    "mean": 14.27,
    "p50":  12.10,
    "p90":  22.83,
    "p99":  38.20,
    "max":  51.04,
    "min":  3.21,
    "total": 17852.4                      // sum of wall_time_s across all games
  },
  "worker_balance": {
    "by_worker": {
      "0": {"games": 154, "wall_time_total_s": 2230.4, "wall_time_mean_s": 14.48},
      "1": {"games": 156, "wall_time_total_s": 2204.1, "wall_time_mean_s": 14.13}
      // ... one entry per distinct worker_id
    },
    "in_process_count": 0,                // games with worker_id == null
    "max_min_wall_time_ratio": 1.42       // null if < 2 workers carrying wall_time_s
  },
  "final_root_value": {                   // null if no game has the field
    "mean":     0.18,
    "p10":     -0.62,
    "p50":      0.21,
    "p90":      0.93,
    "abs_mean": 0.51                      // mean(|root_value|) — endgame decisiveness proxy
  },
  "final_top1_share": {                   // null if no game has the field
    "mean": 0.41,
    "p10":  0.12,
    "p50":  0.38,
    "p90":  0.79,
    "min":  0.04
  },
  "compute_per_game": {                   // null if no game has meta.compute populated
    "leaf_evals": {"mean": 17400.5, "p50": 17400, "p90": 22100, "max": 28900},
    "backups":    {"mean": 17400.5, "p50": 17400, "p90": 22100, "max": 28900},
    "nn_batches": {"mean":   850.3, "p50":   850, "p90":   980, "max":  1240}
  }
}
```

### 4.1 Field contracts

| Field | Type | None / empty semantics |
|---|---|---|
| `n_games_total` | int | always set; 0 if no replays loaded |
| `n_games_with_stats` | int | always set; 0 if every replay is from before the persistence change |
| `wall_time_s` | object \| null | null when no game has a non-null `wall_time_s` |
| `worker_balance.by_worker` | object | empty `{}` if no worker_id is non-null in any replay |
| `worker_balance.in_process_count` | int | always set |
| `worker_balance.max_min_wall_time_ratio` | float \| null | null when < 2 workers carry wall_time_s |
| `final_root_value` | object \| null | null when no replay has the field |
| `final_top1_share` | object \| null | null when no replay has the field |
| `compute_per_game` | object \| null | null when no replay has `meta.compute` |

Percentile calculation uses `numpy.percentile` with the library default (linear interpolation) — same convention as elsewhere in the analyzer.

### 4.2 Schema rationale

- **`per_game_stats` as a top-level block** mirrors the existing convention (`compute`, `adjudication`, `resign`, `opening`, etc. are all top-level). Lets downstream tools select just the per-game distributions without parsing the whole summary.
- **Distributions, not totals.** The sidecar `compute` block already provides per-iteration totals. The per-game block is intentionally complementary: percentiles + per-worker breakdown + decisiveness proxies.
- **Coverage field (`n_games_with_stats`)** is essential during the rollout window where some replays are old-schema. Without it, a partial coverage looks indistinguishable from a fully-covered low-volume run.
- **`abs_mean` for `final_root_value`** captures endgame decisiveness without conflating the sign. High abs_mean + low entropy on `final_top1_share` = "decisive endings"; low abs_mean + diffuse top1 = "fuzzy endings".

## 5. `report.txt` rendering

A new section placed **immediately after** the existing `Compute:` line (currently at `scripts/twixt_replay_analyzer.py:2179-2180`). Format:

```
Per-game stats (n=1234 / 1234 games carry new fields):
  Wall time:    mean=14.3s p50=12.1s p90=22.8s p99=38.2s max=51.0s (total=4h57m)
  Workers:      8 active; max/min wall-time ratio=1.42 (in-process: 0)
  Final root:   mean=0.18 p50=0.21 p10=-0.62 p90=0.93 (|abs| mean=0.51)
  Final top1:   mean=0.41 p50=0.38 p10=0.12 p90=0.79 min=0.04
  Compute/game: leaf_evals p50=17400 p90=22100 | backups p50=17400 | nn_batches p50=850
```

### 5.1 Empty / partial-coverage rendering

- **Zero games with new fields:** print one line and stop the section: `Per-game stats: no games carry new fields (all replays predate persistence change).`
- **Partial coverage** (n_games_with_stats < n_games_total): the header reflects the ratio (e.g. `n=812 / 1234`) and individual stat lines are computed only over games that have the field.
- **Single worker:** `Workers: 1 active; max/min wall-time ratio=n/a (in-process: M)`.
- **In-process only (worker_id all null):** `Workers: 0 active; in-process: 1234`.

### 5.2 Time formatting

`total` wall time is rendered human-readable: seconds when < 60s, `Xm Ys` when < 1h, `XhYm` otherwise. Per-game means/p-values stay in seconds with one decimal.

## 6. Architecture & data flow

### 6.1 Aggregation function

New top-level function in `scripts/twixt_replay_analyzer.py`, placed near `aggregate_sidecars` (~line 340):

```python
def aggregate_per_game_stats(replays: List[dict]) -> dict:
    """Aggregate per-game stats from loaded replay records.

    Reads meta.{worker_id, wall_time_s, adjudication_block_reason,
    final_root_value, final_top1_share, compute.{leaf_evals, backups,
    nn_batches}} per replay. Old replays lacking these fields are
    silently excluded from per-stat aggregates; n_games_with_stats
    reports coverage.

    Pure function: takes the list, returns the per_game_stats dict.
    Does not mutate replays.
    """
```

Behavior:
- One pass over the replay list. Per replay, extract any populated new fields into per-statistic accumulator lists.
- After the pass: compute mean / percentiles / max / min / total per accumulator using `numpy`. Use `np.percentile(arr, [10, 50, 90, 99])` etc.
- Group worker totals into `by_worker`; compute `max_min_wall_time_ratio` only when ≥ 2 worker IDs carry wall_time_s.
- Return the dict in the shape of §4.

### 6.2 Format function

New function near other `format_*_report` functions (~line 948):

```python
def format_per_game_stats_report(per_game_stats: dict) -> List[str]:
    """Render the per-game stats block as report.txt lines."""
```

Pure function returning a list of strings (final blank line included). Handles all empty/partial cases per §5.1. Uses `_format_duration_human(total_seconds)` helper for the `total=` rendering (§5.2).

### 6.3 Integration points

1. **Summary builder** (~line 1864): after the line `compute_val = ...`, add:
   ```python
   per_game_stats_val = aggregate_per_game_stats(replays)
   ```
   and one new entry inside the `summary` dict literal:
   ```python
   "per_game_stats": per_game_stats_val,
   ```

2. **Report builder** (~line 2180): after the `lines.append(f"Compute: ...")` call, append:
   ```python
   lines.extend(format_per_game_stats_report(summary["per_game_stats"]))
   ```

That's it — three call-site additions and two new functions, all in one file.

## 7. Edge cases & invariants

| Case | Behavior |
|---|---|
| No replays loaded (`replays == []`) | `per_game_stats = {"n_games_total": 0, "n_games_with_stats": 0, "wall_time_s": null, "worker_balance": {"by_worker": {}, "in_process_count": 0, "max_min_wall_time_ratio": null}, "final_root_value": null, "final_top1_share": null, "compute_per_game": null}` |
| All replays old-schema (no new fields) | `n_games_with_stats == 0`; all distribution blocks `null`; `worker_balance.by_worker == {}`. Report renders the single-line "no games carry new fields" message. |
| Mixed coverage (some old, some new) | Aggregates over only the games carrying each field. Header in report shows ratio. |
| `final_top1_share` outside (0, 1] in some game (shouldn't happen given upstream invariants, but defensive) | Included in stats as-is. No clamping. Out-of-range values surface naturally in min/max if they occur. |
| Negative `wall_time_s` (clock skew, defensive) | Included as-is. |
| Single distinct worker carrying wall_time_s | `max_min_wall_time_ratio == null` (ratio undefined for n=1). |
| `worker_id == 0` legitimately | Treated as a worker (key `"0"` in `by_worker`); not conflated with `None`. |
| `meta.compute` present but missing one of three keys (defensive) | Treat missing key as 0 for that game's contribution; computed stats reflect that. |
| `meta` missing entirely (extremely defensive — analyzer already uses `.get()` for this) | Treated as old-schema replay, excluded from new-stat aggregates. |

## 8. Backward compatibility

- **Existing summary.json consumers**: the new `per_game_stats` key is purely additive at the top level. Anything reading `summary["compute"]` or `summary["adjudication"]` is unaffected. The original `report.txt` sections appear in the same order with the same content; the new section is added between the existing `Compute:` line and whatever follows it.
- **Old replays on disk** (saved before the persistence change): silently handled — they show up in `n_games_total` but not in `n_games_with_stats`.
- **No analyzer-level CLI change**: no new flags. The new aggregation always runs; cost is one extra linear pass over the replay list (negligible vs the existing per-replay loop).

## 9. Test plan

New file `tests/test_analyzer_per_game_stats.py`. Pattern mirrors `tests/test_analyzer_phase2_sidecar_fields.py`. Synthetic replay records constructed inline; no MLX or trainer dependency.

1. **`test_aggregate_returns_zero_coverage_for_empty_replays`** — `aggregate_per_game_stats([])` returns the zero-coverage shape per §7.
2. **`test_aggregate_returns_zero_coverage_for_old_schema_only`** — list of replays without any new fields → `n_games_with_stats == 0`, all distribution blocks `null`.
3. **`test_aggregate_full_coverage_populates_all_blocks`** — 5 replays, every new field populated → all blocks non-null, `n_games_with_stats == 5`, percentiles correct (assert specific values).
4. **`test_aggregate_worker_balance_groups_by_worker_id`** — 4 replays from 2 workers (different wall_time_s) → `by_worker` has both keys, `max_min_wall_time_ratio` correct.
5. **`test_aggregate_in_process_games_counted_separately`** — mix of `worker_id=0`, `worker_id=1`, `worker_id=null` → `in_process_count` reflects only the nulls; `by_worker` only contains the integer keys.
6. **`test_aggregate_partial_coverage_excludes_missing_fields_per_stat`** — some games have `wall_time_s` but not `final_top1_share` → wall_time_s block populated, top1 block null (since 0/N games have it... or computed only over those that do — assert chosen behavior).
7. **`test_format_renders_zero_coverage_short_message`** — empty `per_game_stats` → single-line message.
8. **`test_format_renders_full_block`** — fully-populated `per_game_stats` → expected lines present, expected number formatting.
9. **`test_format_handles_single_worker_n_a_ratio`** — one worker → "max/min wall-time ratio=n/a".
10. **`test_format_handles_in_process_only`** — all worker_id null → "Workers: 0 active; in-process: N".
11. **`test_aggregate_compute_per_game_handles_missing_subkeys`** — replays where `meta.compute` is present but missing `nn_batches` → that key contributes 0; other keys aggregated normally.

### 9.1 End-to-end smoke test

12. **`test_analyzer_run_produces_per_game_stats_in_summary_json`** — fixture with 3 synthetic replay JSONs written to `tmp_path`, invoke the analyzer's main flow programmatically (or through `subprocess.run`), assert `summary.json` has the `per_game_stats` block and `report.txt` contains the "Per-game stats" section header.

If invoking the analyzer's CLI proves heavy, this single test can be skipped and the surface-level integration is implicit from tests 1–11 plus the existing analyzer regression suite.

### 9.2 Existing analyzer test regression

Run before commit, must stay green:
```bash
.venv/bin/python -m pytest tests/test_analyzer_phase2_sidecar_fields.py tests/test_analyzer_phase2_smoke.py tests/test_analyzer_replay_probe_scoring_end_to_end.py tests/test_strong_advantage_analyzer_aggregation.py -v
```

## 10. Implementation sequencing

Two commits.

1. **`feat(analyzer): aggregate per-game stats from replays`** — `scripts/twixt_replay_analyzer.py` changes (new `aggregate_per_game_stats`, new `format_per_game_stats_report`, two integration call sites) + new `tests/test_analyzer_per_game_stats.py` (tests 1–11). Run `.venv/bin/python -m pytest tests/test_analyzer_per_game_stats.py -v` and the four-file analyzer regression. Both green → commit.

2. **`test(analyzer): end-to-end smoke test for per-game stats in summary.json`** — only if test 12 in §9.1 is included (decide during implementation based on cost). Skip this commit if the test proves too heavy; the unit-level coverage is already comprehensive.

## 11. Verification

```bash
# 1. New + existing analyzer tests
.venv/bin/python -m pytest tests/test_analyzer_per_game_stats.py tests/test_analyzer_phase2_sidecar_fields.py tests/test_analyzer_phase2_smoke.py tests/test_analyzer_replay_probe_scoring_end_to_end.py tests/test_strong_advantage_analyzer_aggregation.py -v

# 2. Manual smoke: run the analyzer over real saved games
.venv/bin/python scripts/twixt_replay_analyzer.py \
    scripts/GPU/logs/games \
    --out /tmp/analyzer_smoke_out
cat /tmp/analyzer_smoke_out/summary.json | python -m json.tool | grep -A 30 per_game_stats
grep -A 6 "Per-game stats" /tmp/analyzer_smoke_out/report.txt
```

Expected: `per_game_stats` block present in `summary.json` with non-zero `n_games_with_stats` (assuming any post-persistence games are on disk), and the new section visible in `report.txt`.

## 12. Out of scope

- Sidecar additions for any of these stats.
- New CSV emitters for per-game stats.
- New heatmap / matplotlib figures.
- Backfill of old game JSONs to the new schema.
- A separate analyzer module for per-game stats.
- Per-iteration breakdown of per-game stats (e.g., wall-time-by-iteration trend). Possible future enhancement; the data supports it but not in this scope.
- Replay.html or other UI consumers.
