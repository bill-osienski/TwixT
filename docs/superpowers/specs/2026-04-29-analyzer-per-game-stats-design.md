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
3. Surface **per-field coverage** (`coverage` map plus `n_games_with_any_stats`) so a partial-coverage run (mixed old + new replays) is unambiguous — the operator can tell which specific stats are trustworthy and which were sparsely populated.
4. Support **three complementary worker imbalance diagnostics** (per-worker totals + `max_min_wall_time_ratio` + `wall_time_cv` + `max_min_games_ratio`) so both runaway-worker cases and throughput-imbalance cases (slow worker finishing fewer games) are visible at a glance.
5. Surface **game-length distribution** (from pre-existing `meta.n_moves`) and **outcome breakdown** (from pre-existing `meta.reason`) inside the same `per_game_stats` section, even though those data are aggregated elsewhere — a single eye-pass over the section is the operator's run-health triage.
6. Treat **missing compute subkeys as missing, not zero** so per-game compute averages aren't silently depressed by old-schema replays.
7. Confine all changes to `scripts/twixt_replay_analyzer.py`. No new analyzer modules.

## 3. Non-goals

- **No changes to the persistence layer.** All upstream work (MCTS, GameRecord, GameComplete, game_saver, trainer routing) stays exactly as shipped.
- **No new sidecar aggregations.** The sidecar `compute` block already carries totals; the new block is per-game distributions, derived from replays.
- **No `adjudication_block_reason` per-game histogram.** The sidecar already aggregates this as `adjudication.blocks.{ply,threshold,visits,top1}` and `report.txt` already prints it. Per-game version would be redundant.
- **No new heatmap figures or matplotlib plots.** Text-first; figures are a future enhancement if needed.
- **No CSV emitters.** Existing CSV outputs (`replay_cap_by_iter.csv` etc.) are unaffected. Adding per-game-stats CSVs is a separate decision.
- **No backfill of old games.** Old replays without the new fields are silently excluded from per-stat aggregations and reflected in `n_games_with_any_stats`.

## 4. JSON schema (`summary.json` `per_game_stats` block)

A single new top-level key in `summary.json`, sibling of the existing `compute` block:

```jsonc
"per_game_stats": {
  "n_games_total": 1234,                  // total replays loaded
  "n_games_with_any_stats": 812,          // games carrying at least one of the new persistence fields
  "coverage": {                           // per-field replay counts; "812 / 1234" → field reliability
    "wall_time_s":          812,
    "worker_id":            812,
    "final_root_value":     790,
    "final_top1_share":     790,
    "compute.leaf_evals":   812,
    "compute.backups":      812,
    "compute.nn_batches":   805,
    "n_moves":             1234,          // pre-existing field; 100% coverage on a well-formed run
    "reason":              1234           // pre-existing field; 100% coverage on a well-formed run
  },
  "game_length": {                        // from meta.n_moves; null only on n_games_total == 0
    "mean": 126.4, "p50": 118, "p90": 211, "p95": 244, "p99": 286, "max": 312, "min": 12
  },
  "outcomes": {                           // counts derived from meta.reason; small redundant view
    "decisive":    1144,                  // reason == "win"
    "resign":        42,
    "adjudicated":   31,
    "timeout":       17,                  // "timeout" or "timeout_selfplay"
    "draw_other":     0                   // "board_full" | "state_cap" | "unknown" | other draws
  },
  "wall_time_s": {                        // null when no game has wall_time_s populated
    "mean": 14.27,
    "p50":  12.10,
    "p90":  22.83,
    "p95":  31.40,
    "p99":  38.20,
    "max":  51.04,
    "min":   3.21,
    "total": 17852.4                      // sum across all games carrying the field
  },
  "worker_balance": {
    "by_worker": {                        // empty {} if no replay carries a non-null worker_id
      "0": {"games": 154, "n_moves_total": 18972, "wall_time_total_s": 2230.4, "wall_time_mean_s": 14.48},
      "1": {"games": 156, "n_moves_total": 19120, "wall_time_total_s": 2204.1, "wall_time_mean_s": 14.13}
      // ... one entry per distinct integer worker_id
    },
    "in_process_count": 0,                // games with worker_id == null
    "max_min_wall_time_ratio": 1.42,      // null if < 2 workers have wall_time_s
    "max_min_games_ratio":     1.01,      // null if < 2 workers; ratio of busiest to quietest by games
    "wall_time_cv":            0.18       // coefficient of variation of per-worker wall_time_total_s; null if < 2 workers
  },
  "final_root_value": {                   // null when no game has the field
    "mean":     0.18,
    "p10":     -0.62,
    "p50":      0.21,
    "p90":      0.93,
    "abs_mean": 0.51                      // mean(|root_value|) — endgame decisiveness proxy
  },
  "final_top1_share": {                   // null when no game has the field
    "mean": 0.41,
    "p10":  0.12,
    "p50":  0.38,
    "p90":  0.79,
    "min":  0.04
  },
  "compute_per_game": {                   // each subkey null when zero games carry that subkey
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
| `n_games_with_any_stats` | int | always set; 0 if every replay is from before the persistence change |
| `coverage` | object | always set; each entry is an int count of replays where the field is non-null. Subkeys for `compute.*` use dotted notation (`"compute.leaf_evals"`). Includes `n_moves` and `reason` so malformed-replay rates are auditable. |
| `game_length` | object \| null | null when `coverage.n_moves == 0` (no replay carries `meta.n_moves`). On well-formed runs `coverage.n_moves == n_games_total`. |
| `outcomes` | object | always set when `n_games_total > 0`; integer counts. Categories are mutually exclusive: `decisive`, `resign`, `adjudicated`, `timeout`, `draw_other`. `meta.reason` values not in those categories — including replays where `reason` is missing/null — fall into `draw_other`. The `coverage.reason` count + the `outcomes.draw_other` count let you tell "missing reason" from "actual rare draw." |
| `wall_time_s` | object \| null | null when `coverage.wall_time_s == 0` |
| `worker_balance.by_worker` | object | empty `{}` if no replay carries a non-null integer `worker_id` |
| `worker_balance.in_process_count` | int | always set |
| `worker_balance.max_min_wall_time_ratio` | float \| null | null when < 2 workers carry wall_time_s |
| `worker_balance.max_min_games_ratio` | float \| null | null when < 2 workers exist (regardless of wall_time_s coverage) |
| `worker_balance.wall_time_cv` | float \| null | null when < 2 workers carry wall_time_s. Coefficient of variation = stddev / mean of per-worker `wall_time_total_s`. |
| `final_root_value` | object \| null | null when `coverage.final_root_value == 0` |
| `final_top1_share` | object \| null | null when `coverage.final_top1_share == 0` |
| `compute_per_game` | object \| null | null when ALL three `coverage.compute.*` are 0 |
| `compute_per_game.{leaf_evals,backups,nn_batches}` | object \| null | each subkey null when its corresponding `coverage.compute.*` is 0 |

Percentile calculation uses `numpy.percentile` with the library default (linear interpolation) — same convention as elsewhere in the analyzer.

### 4.2 Schema rationale

- **`per_game_stats` as a top-level block** mirrors the existing convention (`compute`, `adjudication`, `resign`, `opening`, etc. are all top-level). Lets downstream tools select just the per-game distributions without parsing the whole summary.
- **Distributions, not totals.** The sidecar `compute` block already provides per-iteration totals. The per-game block is intentionally complementary: percentiles + per-worker breakdown + decisiveness proxies.
- **Per-field coverage (`coverage`) is the authoritative answer to "is this field trustworthy?"** `n_games_with_any_stats` is a quick at-a-glance number, but a healthy 812/1234 there could still hide a `final_top1_share` field that landed on only 5/812 of those games. The `coverage` map makes per-field rollout state unambiguous.
- **Three worker imbalance metrics, not one.** `max_min_wall_time_ratio` catches one runaway worker. `wall_time_cv` is robust against single-outlier denominators and reflects overall imbalance. `max_min_games_ratio` catches the case where a slow worker finishes far fewer games (the imbalance is in throughput, not just per-game time). The trio together is much more diagnostic than any one alone.
- **`game_length` and `outcomes` near the per-game-stats section** intentionally duplicate data already aggregated elsewhere in the report. Per the operator workflow ("I only review summary.json and report.txt"), having long-game and timeout signals adjacent to wall-time and compute distributions makes the run-health scan a single eye-pass. The cost is small (one int counter set + one numpy percentile call); the win is that "longer games + more timeouts + worker imbalance" all sit together when you're triaging.
- **Missing compute subkey ≠ zero.** A missing field means "this replay does not carry it" (old schema). A zero means "we measured it and it was zero" (new schema). Conflating them silently depresses averages. We use coverage to distinguish.
- **`abs_mean` for `final_root_value`** captures endgame decisiveness without conflating the sign. High abs_mean + low entropy on `final_top1_share` = "decisive endings"; low abs_mean + diffuse top1 = "fuzzy endings".

## 5. `report.txt` rendering

A new section placed **immediately after** the existing `Compute:` line (currently at `scripts/twixt_replay_analyzer.py:2179-2180`). Format:

```
Per-game stats (n=812 / 1234 games carry new fields):
  Game length:  mean=126.4 p50=118 p90=211 p95=244 max=312
  Outcomes:     decisive=1144 resign=42 adjudicated=31 timeout=17 draw_other=0
  Wall time:    mean=14.3s p50=12.1s p90=22.8s p95=31.4s max=51.0s (total=4h57m)
  Workers:      10 active; games min/max=3/12 ratio=4.00; wall-time ratio=7.55; cv=0.84 (in-process: 0)
  Final root:   mean=0.18 p50=0.21 p10=-0.62 p90=0.93 (|abs| mean=0.51)
  Final top1:   mean=0.41 p50=0.38 p10=0.12 p90=0.79 min=0.04
  Compute/game: leaf_evals p50=17400 p90=22100 max=28900 | backups p50=17400 p90=22100 max=28900 | nn_batches p50=850 p90=980 max=1240
  Coverage:     wall_time_s=812 worker_id=812 final_root_value=790 final_top1_share=790 compute={leaf_evals=812, backups=812, nn_batches=805}
```

The `Game length` and `Outcomes` lines always render (driven by pre-existing fields). The other six lines render conditionally per §5.1. The `Coverage` line is suppressed when coverage is uniform (every persistence-era field has the same count) — printed only when there is per-field divergence to surface.

### 5.1 Empty / partial-coverage rendering

- **Zero games with new persistence fields** (`n_games_with_any_stats == 0`): print only the `Game length` and `Outcomes` lines (which use pre-existing fields), then a one-liner: `Per-game stats: no games carry new persistence fields (all replays predate persistence change).`
- **Partial coverage** (`n_games_with_any_stats < n_games_total`): the header shows the ratio (`n=812 / 1234`); individual stat lines are computed only over games that have each respective field; the per-field `Coverage:` line is printed.
- **Uniform coverage** (every persistence-era field has the same count): suppress the `Coverage:` line — the header ratio carries the info.
- **Single worker:** `Workers: 1 active; games=N; wall-time mean=Xs (in-process: M)` — no ratios (undefined for n=1).
- **In-process only** (`by_worker == {}`): `Workers: 0 active; in-process: 1234`.
- **Field with zero coverage** (e.g. nobody populated `final_top1_share`): omit that line entirely rather than print "Final top1: n/a".

### 5.2 Number formatting

- `total` wall time: seconds when < 60s, `Xm Ys` when < 1h, `XhYm` otherwise. Per-game means/p-values stay in seconds with one decimal.
- `n_moves` percentiles render as integers (no decimal). The `mean` for game_length renders with one decimal.
- Compute counters render as integers (no thousands separator — keeps the line compact for grep'ability).
- Ratios render with two decimals (`1.42`, `7.55`).
- `wall_time_cv` renders with two decimals.

## 6. Architecture & data flow

### 6.1 Aggregation function

New top-level function in `scripts/twixt_replay_analyzer.py`, placed near `aggregate_sidecars` (~line 340):

```python
def aggregate_per_game_stats(replays: List[dict]) -> dict:
    """Aggregate per-game stats from loaded replay records.

    Reads:
      - meta.n_moves (pre-existing) → game_length
      - meta.reason (pre-existing) → outcomes
      - meta.{worker_id, wall_time_s, final_root_value, final_top1_share,
        compute.{leaf_evals, backups, nn_batches}} (new in 2026-04-29
        persistence change) → distribution + worker_balance + compute_per_game

    Old replays lacking persistence fields are silently excluded from
    those per-stat aggregates. Per-field coverage is recorded in
    coverage.{...} so consumers can tell exactly which stats are reliable.
    A missing meta.compute subkey is treated as MISSING (excluded from
    that subkey's stats), not as zero.

    Pure function: takes the replay list, returns the per_game_stats dict.
    Does not mutate replays.
    """
```

Behavior:
- **Worker identity comes solely from `meta.worker_id`.** Never infer it from filename, sidecar, or any other source. The aggregator treats this as the single source of truth.
- One pass over the replay list. For each replay:
  - Extract `meta.n_moves` if present and non-null (treat missing as MISSING, same convention as other fields). Increment `coverage["n_moves"]` only when present.
  - Extract `meta.reason` if present and non-null. Increment `coverage["reason"]` only when present. For outcome categorization: present+recognized values bin into the four real categories (`decisive`, `resign`, `adjudicated`, `timeout`); present-but-unrecognized OR missing values bin into `draw_other`.
  - For each persistence-era field, extract only if non-null. Append to the field's accumulator AND increment the field's coverage counter.
  - For `meta.compute.{leaf_evals,backups,nn_batches}`, treat each subkey independently: if `meta.compute` is present and the subkey is present and non-null, append to that subkey's accumulator and increment `coverage["compute.<subkey>"]`. Missing subkey = excluded.
  - For `worker_id`: if integer, accumulate per-worker totals (games count + n_moves total + wall_time list) and increment `coverage["worker_id"]`; if explicitly `null`, increment `in_process_count` and increment `coverage["worker_id"]` (the field IS present and explicitly null = persistence-era in-process game); if absent (key not in `meta`), do not touch `coverage["worker_id"]` and do not bucket the game.
- After the pass:
  - Compute means/percentiles/max/min/total per accumulator using `numpy.percentile` (default linear interpolation).
  - For `worker_balance`: compute `max_min_wall_time_ratio` (max/min of per-worker `wall_time_total_s`, only when ≥ 2 workers have wall_time_s); `max_min_games_ratio` (max/min of per-worker `games`, only when ≥ 2 workers); `wall_time_cv` (stddev/mean of per-worker `wall_time_total_s`, only when ≥ 2 workers carry wall_time_s; uses ddof=0 to match `numpy.std` default).
  - `n_games_with_any_stats` = count of replays where at least one persistence-era field is non-null.
- Return the dict in the shape of §4. Each top-level distribution block is `null` when its corresponding coverage count is 0 (per §4.1).

### 6.2 Format function

New function near other `format_*_report` functions (~line 948):

```python
def format_per_game_stats_report(per_game_stats: dict) -> List[str]:
    """Render the per-game stats block as report.txt lines.

    Suppresses lines for fields with zero coverage (per §5.1), suppresses
    the per-field Coverage: line when coverage is uniform across all
    persistence-era fields, and falls back to a single short message when
    n_games_with_any_stats == 0.
    """
```

Pure function returning a list of strings (final blank line included). Handles all empty/partial cases per §5.1. Uses a small `_format_duration_human(total_seconds)` helper for the `total=` rendering (§5.2).

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
| No replays loaded (`replays == []`) | `per_game_stats = {"n_games_total": 0, "n_games_with_any_stats": 0, "coverage": {<all keys>: 0, including n_moves and reason}, "game_length": null, "outcomes": {<all categories>: 0}, "wall_time_s": null, "worker_balance": {"by_worker": {}, "in_process_count": 0, "max_min_wall_time_ratio": null, "max_min_games_ratio": null, "wall_time_cv": null}, "final_root_value": null, "final_top1_share": null, "compute_per_game": null}` |
| All replays old-schema (no new persistence fields) | `n_games_with_any_stats == 0`; all persistence-era distribution blocks `null`; `coverage.<persistence-era keys>` all 0; `worker_balance.by_worker == {}`. `game_length` and `outcomes` still populated from pre-existing fields. Report renders the short "no games carry new persistence fields" message after the game_length / outcomes lines. |
| Mixed coverage (some old, some new) | Aggregates over only the games carrying each field. Per-field counts in `coverage`. Report header shows ratio; `Coverage:` line printed when coverage is non-uniform. |
| `final_top1_share` outside (0, 1] in some game (shouldn't happen given upstream invariants, but defensive) | Included in stats as-is. No clamping. Out-of-range values surface naturally in min/max if they occur. |
| Negative `wall_time_s` (clock skew, defensive) | Included as-is. |
| Single distinct worker carrying wall_time_s | `max_min_wall_time_ratio == null`, `wall_time_cv == null` (ratio + cv undefined for n=1); `max_min_games_ratio == null` if only 1 distinct worker. |
| `worker_id == 0` legitimately | Treated as a worker (key `"0"` in `by_worker`); not conflated with `None`. |
| `meta.compute` present but missing one of three subkeys | The missing subkey is treated as MISSING (not 0). That game does not contribute to that subkey's stats. `coverage["compute.<subkey>"]` reflects this. |
| `meta.compute` is `null` or absent | Game does not contribute to any `compute_per_game` subkey. All three coverage counters unchanged for this game. |
| `meta.reason` value not in any of the five outcome categories | Counted in `outcomes.draw_other` (catch-all bucket — keeps the five categories mutually exclusive and total == n_games_total). |
| `meta` missing entirely (extremely defensive — analyzer already uses `.get()` for this) | Treated as old-schema replay; excluded from new-stat aggregates; `n_moves` and `reason` access fail safely → not counted in `coverage`; counted in `outcomes.draw_other` (since reason is effectively absent). |
| `meta.n_moves` missing or null | Excluded from `game_length` stats; `coverage["n_moves"]` not incremented; `n_games_total` still increments (the replay was loaded). |
| `meta.reason` missing or null | Counted in `outcomes.draw_other`; `coverage["reason"]` not incremented. The two together — low `coverage.reason` + non-zero `draw_other` — diagnose corrupted replay metadata. |
| Worker identity inferred from filename or sidecar | **Explicitly forbidden.** Only `meta.worker_id` (when present and integer) determines per-worker bucketing. |
| Per-worker `wall_time_total_s == 0` (would div-by-zero in `max_min_wall_time_ratio`) | Exclude that worker from the ratio computation; if fewer than 2 workers remain, ratio is `null`. |
| All per-worker `wall_time_total_s` equal | `max_min_wall_time_ratio == 1.0`, `wall_time_cv == 0.0`. |

## 8. Backward compatibility

- **Existing summary.json consumers**: the new `per_game_stats` key is purely additive at the top level. Anything reading `summary["compute"]` or `summary["adjudication"]` is unaffected. The original `report.txt` sections appear in the same order with the same content; the new section is added between the existing `Compute:` line and whatever follows it.
- **Old replays on disk** (saved before the persistence change): silently handled — they show up in `n_games_total` but not in `n_games_with_any_stats`.
- **No analyzer-level CLI change**: no new flags. The new aggregation always runs; cost is one extra linear pass over the replay list (negligible vs the existing per-replay loop).

## 9. Test plan

New file `tests/test_analyzer_per_game_stats.py`. Pattern mirrors `tests/test_analyzer_phase2_sidecar_fields.py`. Synthetic replay records constructed inline; no MLX or trainer dependency.

1. **`test_aggregate_returns_zero_coverage_for_empty_replays`** — `aggregate_per_game_stats([])` returns the zero-coverage shape per §7.
2. **`test_aggregate_returns_zero_coverage_for_old_schema_only`** — list of replays without any persistence-era fields → `n_games_with_any_stats == 0`; all persistence-era distribution blocks `null`; `coverage["wall_time_s"] == 0` etc.; `game_length` and `outcomes` still populated from `meta.n_moves` and `meta.reason`.
3. **`test_aggregate_full_coverage_populates_all_blocks`** — 5 replays, every persistence-era field populated → all blocks non-null, `n_games_with_any_stats == 5`, every `coverage` entry == 5, percentiles correct (assert specific values).
4. **`test_aggregate_per_field_coverage_counts_independently`** — 10 replays where 8 have `wall_time_s`, 5 have `final_top1_share`, 7 have `compute.nn_batches` → coverage exactly `{wall_time_s: 8, final_top1_share: 5, "compute.nn_batches": 7, ...}`. Each distribution block computed over only its covering games.
5. **`test_aggregate_worker_balance_groups_by_worker_id`** — 4 replays from 2 workers (different wall_time_s) → `by_worker` has both keys, `max_min_wall_time_ratio`, `max_min_games_ratio`, `wall_time_cv` all populated and numerically correct.
6. **`test_aggregate_worker_balance_includes_n_moves_per_worker`** — replays with `n_moves` and `worker_id` populated → `by_worker[w]["n_moves_total"]` is the sum of `n_moves` across that worker's games.
7. **`test_aggregate_in_process_games_counted_separately`** — mix of `worker_id=0`, `worker_id=1`, `worker_id=null` → `in_process_count` reflects only the nulls; `by_worker` only contains the integer keys; `worker_id=0` appears as `"0"` (not conflated with null).
8. **`test_aggregate_compute_subkey_missing_is_excluded_not_zero`** — replays where `meta.compute = {leaf_evals: 100, backups: 200}` (no nn_batches) → `coverage["compute.nn_batches"] == 0`, `compute_per_game["nn_batches"]` is `null`; `leaf_evals` and `backups` stats reflect actual values (not depressed by phantom zeros).
9. **`test_aggregate_outcomes_categorizes_meta_reason`** — replays with `meta.reason` ∈ {`win`, `resign`, `adjudicated`, `timeout`, `timeout_selfplay`, `board_full`, `state_cap`, `unknown`} → `outcomes` counts categorize correctly; `timeout` and `timeout_selfplay` both go into `outcomes.timeout`; non-listed reasons go into `draw_other`; counts sum to `n_games_total`.
10. **`test_aggregate_game_length_uses_meta_n_moves`** — replays with various `meta.n_moves` → `game_length.{mean,p50,p90,p95,max,min}` correct.
11. **`test_aggregate_single_worker_yields_null_ratios`** — one distinct worker → `max_min_wall_time_ratio is None`, `max_min_games_ratio is None`, `wall_time_cv is None`.
12. **`test_aggregate_workers_with_zero_wall_time_excluded_from_ratio`** — one worker has wall_time_total_s=0 → that worker excluded from `max_min_wall_time_ratio` computation; if fewer than 2 workers remain, ratio is None.
13. **`test_aggregate_uniform_per_worker_wall_time_yields_unity_ratio_zero_cv`** — all per-worker wall_time_total_s equal → `max_min_wall_time_ratio == 1.0`, `wall_time_cv == 0.0`.
14. **`test_format_renders_zero_coverage_short_message`** — `n_games_with_any_stats == 0` → game_length + outcomes lines render, then the short "no games carry new persistence fields" line.
15. **`test_format_renders_full_block`** — fully-populated `per_game_stats` → expected lines present in expected order; uniform coverage → no `Coverage:` line printed.
16. **`test_format_renders_coverage_line_on_partial_coverage`** — non-uniform coverage → `Coverage:` line is printed with the per-field counts.
17. **`test_format_omits_lines_for_zero_coverage_fields`** — `final_top1_share` block is `None` → "Final top1:" line absent (not "n/a"), other lines still present.
18. **`test_format_handles_single_worker`** — one worker → worker line shows `1 active; games=N; wall-time mean=Xs (in-process: M)`, no ratios.
19. **`test_format_handles_in_process_only`** — all worker_id null → `Workers: 0 active; in-process: N`.
20. **`test_format_human_readable_duration`** — `total = 17852.4` → `total=4h57m`; `total = 145.0` → `total=2m 25s`; `total = 30.0` → `total=30.0s`. Edge: `total = 0` → `total=0.0s`.

### 9.1 End-to-end smoke test

**21. (optional) `test_analyzer_run_produces_per_game_stats_in_summary_json`** — fixture with 3 synthetic replay JSONs written to `tmp_path`, invoke the analyzer's main flow programmatically (or through `subprocess.run`), assert `summary.json` has the `per_game_stats` block and `report.txt` contains the "Per-game stats" section header.

If invoking the analyzer's CLI proves heavy, this single test can be skipped and the surface-level integration is implicit from tests 1–20 plus the existing analyzer regression suite.

### 9.2 Existing analyzer test regression

Run before commit, must stay green:
```bash
.venv/bin/python -m pytest tests/test_analyzer_phase2_sidecar_fields.py tests/test_analyzer_phase2_smoke.py tests/test_analyzer_replay_probe_scoring_end_to_end.py tests/test_strong_advantage_analyzer_aggregation.py -v
```

## 10. Implementation sequencing

Two commits.

1. **`feat(analyzer): aggregate per-game stats from replays`** — `scripts/twixt_replay_analyzer.py` changes (new `aggregate_per_game_stats`, new `format_per_game_stats_report`, two integration call sites) + new `tests/test_analyzer_per_game_stats.py` (tests 1–20). Run `.venv/bin/python -m pytest tests/test_analyzer_per_game_stats.py -v` and the four-file analyzer regression. Both green → commit. (The implementation plan that follows this spec may decompose this single commit into multiple, more bite-sized commits — that is preferred for review purposes.)

2. **`test(analyzer): end-to-end smoke test for per-game stats in summary.json`** — only if optional test 21 in §9.1 is included (decide during implementation based on cost). Skip this commit if the test proves too heavy; the unit-level coverage from tests 1–20 is already comprehensive.

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

Expected: `per_game_stats` block present in `summary.json` with non-zero `n_games_with_any_stats` (assuming any post-persistence games are on disk), and the new section visible in `report.txt`.

## 12. Out of scope

- Sidecar additions for any of these stats.
- New CSV emitters for per-game stats.
- New heatmap / matplotlib figures.
- Backfill of old game JSONs to the new schema.
- A separate analyzer module for per-game stats.
- Per-iteration breakdown of per-game stats (e.g., wall-time-by-iteration trend). Possible future enhancement; the data supports it but not in this scope.
- Replay.html or other UI consumers.
