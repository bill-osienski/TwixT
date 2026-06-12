# Eval Replay Loss Analyzer (V2 Phase B) — Design

**Date:** 2026-06-12
**Status:** Approved, pending implementation plan
**Author:** bill + Claude
**Predecessors:** V1 game-level analyzer
(`docs/superpowers/specs/2026-06-09-eval-loss-analyzer-design.md`), Phase A
replay capture (`docs/superpowers/specs/2026-06-09-eval-replay-capture-design.md`)

## Purpose

Explain **why** checkpoint A (eps035 `0399`) loses to B (staged `0379`) as black
in midgame, using the per-ply replay data captured in Phase A. The focus
question: among A-as-black, decisive A losses of 41–80 moves, does 0399 collapse
via **value drop**, **visit diffusion**, or **low-confidence (low-visit-rank)
moves** — and **when**?

New V2 files (V1 is not modified):

- `scripts/GPU/alphazero/eval_loss_replay_analysis.py` — pure analysis (no IO,
  no MLX): game rows + replay dicts in, feature dicts / table rows / verdict out.
- `scripts/GPU/alphazero/eval_loss_replay_analyzer.py` — thin CLI: input
  resolution, sidecar loading, artifact writing, console summary.
- Tests in `tests/test_eval_loss_replay_analysis.py` (pure) and
  `tests/test_eval_loss_replay_analyzer.py` (CLI integration).

### Scope

- **In:** the replay-aware analyzer over one or more `*_games.jsonl` files whose
  rows carry `replay_path` (Phase A capture format), keyed on the focus signal
  but with generic filters.
- **Out (deferred):** per-ply long-format dump (`--write-ply-dump`); changepoint
  detection (CUSUM/piecewise) for collapse timing; cross-match combined CSV;
  tournament capture (Phase C); `selected_policy_rank` (not captured in Phase A).

## Dataset

Primary input: `logs/eval/eps035_0399_vs_0379_800g_w4_seed35791_replay_games.jsonl`
plus `..._seed35791_replays/game_*.json` sidecars (800 games; run summary
0.4688 / elo −21.7, consistent with the V1 finding on a fresh seed-35791 sample —
**not** game-for-game identical to the V1 seed-24681 sample). Focus cohort in
this capture: 400 A-as-black games → 239 decisive A losses → **172 in the 41–80
move window**.

Per-ply record fields (Phase A locked): `ply`, `player`, `row`, `col`,
`root_value`, `root_top1_share`, `selected_visit_rank`, `selected_visit_count`,
`root_total_visits`, `n_legal`.

### Value-sign convention (confirmed)

`mcts.py` uses negamax backup (terminal values and `_backup` are from the
perspective of the side to move; child q is negated at selection). So
`root_value` is **always from the perspective of the player about to move**.
Consequences:

- A's value trajectory uses **A's own plies only** — the primary signal.
- B's series is reported **in B's own perspective, kept separate**; no merged
  sign-flipped series is produced.

### Opening-temperature exclusion (key rule)

Eval games select moves by **temperature sampling for the first
`opening_temp_plies = 20` plies** (then near-argmax). A `selected_visit_rank`
of 4 at ply 0 is sampling, not low confidence; in 41–80-move games the opening
is 25–50% of all plies. Therefore:

- **Confidence/diffusion features** (`selected_visit_rank`, `root_top1_share`
  rules, `low_confidence_ply_count`, `selected_visit_share`) are computed on
  **post-opening plies only** (`ply >= --opening-plies`, default 20).
- **Value-trajectory features** use all A plies (the value reading itself is
  not temperature-distorted; early values legitimately detect `already_bad`).
- Opening plies still feed the opening-cluster context table and the `opening`
  phase bucket, which exists precisely to keep that regime separate.

## Cohorts & filters

CLI filters (generic, defaulted to the focus question): `--a-color`
(`red`/`black`, default `black`), `--min-moves 41`, `--max-moves 80`. Within
the window, **decisive games only** (`reason == "win"`); draws are counted and
reported as excluded. Two cohorts are always built:

- **loss** — A's focus-window decisive losses (the cohort to explain). Empty →
  hard error.
- **win** — A's focus-window decisive wins (the primary contrast baseline).
  If `< 5` games, effect sizes are emitted as `null` with an explicit
  `"insufficient_contrast"` note instead of failing.

Contrasts:

1. **Primary (drives the verdict):** loss vs win cohorts, A's own plies only —
   what differs when 0399 loses vs wins in the same role/length band?
2. **Secondary (supporting evidence):** A's plies vs B's plies inside the loss
   cohort — did 0379 see/control the win earlier?

## Per-game features

For every focus-window decisive game (one feature dict per game, `cohort` ∈
{loss, win}):

Value trajectory (all A plies; values are A's own `root_value`):

- `initial_a_value` / `final_a_value` — median of the first / last 3 A plies
  (or fewer, if the game has fewer A plies).
- `mean_a_value`, `min_a_value`.
- `largest_a_value_drop` — most negative delta between consecutive A plies
  (signed; more negative = sharper cliff), with `largest_drop_ply` (global ply
  index of the later ply) and `largest_drop_fraction` (`ply / (n_moves - 1)`);
  `null` when the game has fewer than 2 A plies.
- First crossings: `first_a_value_below_0`, `first_a_value_below_bad`
  (−0.25), `first_a_value_below_lost` (−0.50) — each as global ply index and
  game fraction; `null` if never crossed. Plain first crossings (decisive
  losses rarely recover; cliffs are caught by `largest_a_value_drop`).

Confidence/diffusion (post-opening A plies only; `null` if none exist):

- `mean_top1_share_post`, `min_top1_share_post`.
- `median_selected_visit_rank_post`, `max_selected_visit_rank_post`.
- `mean_selected_visit_share_post` (`selected_visit_count / root_total_visits`).
- `low_confidence_ply_count` — post-opening A plies with
  `selected_visit_rank >= --low-visit-rank`.
- `diffuse_ply_fraction` — fraction of post-opening A plies with
  `root_top1_share <= --low-top1-share`.

Context: `mean_n_legal`, `n_a_plies`, `n_a_plies_post`, `opening_key`
(see opening clusters).

B-side (loss-cohort games only; B's plies, B's perspective):

- `b_mean_value`, `b_mean_top1_share_post`, `b_median_visit_rank_post`.
- Win-onset crossings: `b_first_value_above_0_25`, `b_first_value_above_0_50`
  (ply + fraction; `null` if never).
- `b_saw_it_first` — true when `b_first_value_above_0_50` precedes
  `first_a_value_below_lost` (both non-null, compared by fraction).

## Collapse classification

Per loss-cohort game, evaluated against CLI-tunable thresholds
(`--bad-value −0.25`, `--lost-value −0.50`, `--sharp-drop 0.40`,
`--low-top1-share 0.10`, `--low-visit-rank 5`) plus named module constants
(`HEALTHY_START = −0.10`, `DECAYED_FINAL = −0.40`, `DIFFUSE_MEAN_TOP1 = 0.15`,
`DIFFUSE_PLY_FRACTION = 0.25`, `LOW_RANK_MEDIAN = 3`, `LOW_RANK_PLY_COUNT = 3`):

- `already_bad` — `initial_a_value <= bad_value`.
- `sharp_value_drop` — `largest_a_value_drop <= -sharp_drop`.
- `gradual_decay` — `initial_a_value > HEALTHY_START` and
  `final_a_value <= DECAYED_FINAL` and not sharp.
- `search_diffusion` — `mean_top1_share_post <= DIFFUSE_MEAN_TOP1` or
  `diffuse_ply_fraction >= DIFFUSE_PLY_FRACTION`.
- `low_visit_selection` — `median_selected_visit_rank_post >= LOW_RANK_MEDIAN`
  or `low_confidence_ply_count >= LOW_RANK_PLY_COUNT`.
- `no_clear_signal` — none of the above.

**One label per game**, assigned in the documented precedence order above
(`already_bad` → `sharp_value_drop` → `gradual_decay` → `search_diffusion` →
`low_visit_selection` → `no_clear_signal`). Every rule's boolean flag is also
stored per game (`flag_already_bad`, `flag_sharp`, …) so multi-signal games are
visible in the CSVs. Rules whose inputs are `null` (no post-opening plies) do
not fire.

## Phase buckets

A-ply phase assignment: `opening` = `ply < opening_plies` (absolute — matches
the sampling regime); remaining A plies split into four equal **game-fraction**
bands over `[opening_plies, n_moves)`: `early_midgame`, `midgame`,
`late_midgame`, `pre_terminal`. Per cohort × phase: games, plies,
mean/median `root_value`, mean/median `root_top1_share`, mean/median
`selected_visit_rank`. Interpretation note in the artifact: the `opening` row
is temperature-sampled (rank/share there are not confidence evidence).

## Verdict synthesis

Two evidence layers, both reported:

1. **Collapse-type distribution** over the loss cohort, grouped into failure
   modes: `value-drop` = {sharp_value_drop, gradual_decay}; `already-losing` =
   {already_bad}; `diffusion` = {search_diffusion}; `low-visit-selection` =
   {low_visit_selection}; unexplained = {no_clear_signal}.
2. **Effect sizes** (Cohen's d, pooled sample std, ddof=1; `null` on degenerate
   variance) for loss vs win on: `final_a_value`, `largest_a_value_drop`,
   `initial_a_value`, `mean_top1_share_post`,
   `median_selected_visit_rank_post`.

Deterministic rule (module constants `PRIMARY_SHARE = 0.35`,
`SECONDARY_SHARE = 0.20`): primary failure mode = largest failure-mode share if
`>= PRIMARY_SHARE` (if the largest share is `unexplained`, or no mode reaches
the bar, the verdict is `"mixed / no strong single signal"`); secondary signal =
next-largest mode share if `>= SECONDARY_SHARE`. Effect sizes are printed as
confirming/contradicting evidence alongside, never override the distribution.
With an insufficient win cohort, the verdict is still computed from the
distribution; the effect-size layer says `"insufficient_contrast"`.

## Artifacts

Output dir: `logs/eval/loss_analysis_v2/` (CLI `--output-dir`). Per input
match, `<stem>` from `<stem>_games.jsonl`:

- **`<stem>_replay_summary.json`** — match metadata (A/B checkpoints,
  pairing_id, filters + thresholds used), cohort definition + counts (incl.
  excluded draws), primary contrast (per-metric loss/win aggregates + effect
  sizes), secondary contrast (A vs B aggregates in losses, B win-onset timing,
  `b_saw_it_first` share + median fraction gap), collapse-type distribution,
  timing distribution (loss cohort: p25/p50/p75 of crossing fractions +
  never-crossed counts), verdict {primary, primary_share, secondary,
  narrative}.
- **`<stem>_cohort_comparison.csv`** — one row per cohort: games, plies,
  mean/median `root_value`, mean/median `top1_share` (post-opening),
  mean/median `selected_visit_rank` (post-opening),
  mean `selected_visit_share` (post-opening), mean `n_legal`.
- **`<stem>_phase_buckets.csv`** — cohort × phase rows (shape above).
- **`<stem>_collapse_timing.csv`** — one row per focus-window decisive game
  (both cohorts, `cohort` column): identity (`game_idx`, `task_id`,
  `replay_path`, `a_color`, `winner`, `n_moves`), all per-game features above,
  `collapse_type` + per-rule flags, B-side columns (loss rows; empty for wins).
- **`<stem>_manual_review_queue.csv`** — top `--review-queue 50` loss games
  sorted by composite priority: `largest_a_value_drop` asc (most negative
  first), then `final_a_value` asc, then `mean_top1_share_post` asc, then
  `median_selected_visit_rank_post` desc. Columns: rank, identity +
  `replay_path`, `collapse_type`, key timing/confidence features,
  `opening_key` (the same key the clusters use; default 4 plies — the
  "opening_4" column from the brainstorm, named generically since
  `--opening-key-plies` is tunable).
- **`<stem>_opening_clusters.csv`** — context only, not the diagnostic. Key =
  first `--opening-key-plies 4` plies (both players) rendered
  `r{row}c{col}|r{row}c{col}|…`. One row per key over focus-window decisive
  games: `opening_plies`, `opening_key`, `cohort` (constant focus label, e.g.
  `A_black_41_80_decisive`), `games`, `losses`, `wins`, `a_score_rate`,
  `mean_root_value_early` (A's opening-ply values), `mean_top1_share_early`,
  `avg_moves`; sorted by games desc then a_score_rate asc. With temperature
  openings most keys are singletons — expected and fine; the table gains value
  as more captures accumulate.

Console output mirrors V1 (cohort counts, collapse-type shares, effect-size
table, secondary-contrast line) and ends with:

```
Phase B verdict: A-as-black 41–80 losses are best explained by <mode> (...).
Manual review queue: logs/eval/loss_analysis_v2/<stem>_manual_review_queue.csv
```

## CLI

```
python -m scripts.GPU.alphazero.eval_loss_replay_analyzer \
  --games-jsonl logs/eval/eps035_0399_vs_0379_800g_w4_seed35791_replay_games.jsonl \
  [--glob PATTERN] [--output-dir logs/eval/loss_analysis_v2] \
  [--a-checkpoint ... --b-checkpoint ...] \
  [--a-color black] [--min-moves 41] [--max-moves 80] \
  [--opening-plies 20] [--opening-key-plies 4] \
  [--bad-value -0.25] [--lost-value -0.50] [--sharp-drop 0.40] \
  [--low-top1-share 0.10] [--low-visit-rank 5] \
  [--review-queue 50]
```

Multiple inputs are processed independently (per-match artifacts; no combined
CSV in this phase). Reuse from V1: `resolve_checkpoints`, `validate_rows`,
`a_color`, `score_for_checkpoint` (from `eval_loss_analysis`); `load_jsonl`,
`load_sibling_summary`, `stem_of`, `resolve_inputs`, `write_json`, `write_csv`
(from `eval_loss_analyzer`); stats from `eval_elo` (`score_rate` for cluster
rows). Inputs whose rows lack `replay_path` entirely (V1-era files) are skipped
with a console note, so a broad `--glob` stays safe.

## Architecture

`eval_loss_replay_analysis.py` (pure):

```python
validate_replay(row, replay)                    # fail-loud cross-checks
a_ply_series(replay, color)                     # the per-ply dicts for one side
game_features(row, replay, a_color, thresholds) # flat per-game feature dict
classify_collapse(features, thresholds)         # (label, flags) via precedence
cohort_aggregates(features_list)                # cohort_comparison rows
phase_bucket_rows(games, opening_plies)         # cohort × phase rows
effect_sizes(loss_feats, win_feats)             # Cohen's d per metric (or null)
make_verdict(distribution, effects, counts)     # deterministic verdict dict
review_queue_rows(loss_feats, limit)            # sorted queue rows
opening_cluster_rows(games, key_plies, label)   # context table rows
build_replay_summary(...)                       # assembles the JSON payload
```

`thresholds` is a small frozen dataclass carrying the five CLI thresholds +
`opening_plies`; the named module constants stay in the pure module.

`eval_loss_replay_analyzer.py` (CLI): parse args → resolve inputs → per input:
load rows, resolve A/B, `validate_rows` (V1), filter to the focus window, split
cohorts, load each cohort game's sidecar (`json.load(replay_path)` resolved
relative to CWD, per the Phase A contract), `validate_replay`, compute
features, write the six artifacts, print the console summary.

## Validation & error handling (fail-loud, V1 style)

- Focus-window rows with `replay_path is None` → hard error naming the game
  (a partially-captured file is corrupt, not skippable).
- Sidecar checks per game: file exists and parses; `schema_version == 1`;
  identity fields match the row (`game_idx`, `task_id`, `pairing_id`, `winner`,
  `reason`, `n_moves`, both checkpoints); `len(moves) == n_moves`; plies are
  `0..n_moves-1` alternating `red`/`black` starting `red`; required per-ply
  keys present. Any violation raises with the game index and path.
- Empty loss cohort → hard error (nothing to explain).
- Win cohort `< 5` → proceed; effect sizes `null` + `insufficient_contrast`.
- Threshold sanity: `--bad-value > --lost-value` required (e.g. −0.25 > −0.50);
  `--sharp-drop > 0`.
- Exit-code convention: argument-level violations (threshold sanity, no
  inputs) use argparse-style errors / return code 2; data-level violations
  (missing sidecar, identity mismatch, null `replay_path` in the focus window,
  empty loss cohort) raise `ValueError` from the pure/loader layer — a dev
  tool traceback is acceptable, the message names the game and file.

## Testing (TDD, no MLX, no real checkpoints)

Pure (`tests/test_eval_loss_replay_analysis.py`) with hand-built tiny replays:

- `a_ply_series` filters the right plies; series order preserved.
- Each value feature: medians-of-3, signed largest drop + ply + fraction,
  each first-crossing (incl. never-crossed → `null`), fraction math.
- Opening exclusion: rank/share features ignore plies `< opening_plies`; a game
  with no post-opening A plies yields `null` confidence features and the
  dependent rules don't fire.
- Each classification rule at and across its threshold boundary; the precedence
  order (a game matching sharp+diffusion gets `sharp_value_drop` but both
  flags); `no_clear_signal` fallback.
- Effect sizes: known hand-computed d; degenerate variance → `null`.
- Verdict: primary/secondary share thresholds, unexplained-dominant → mixed,
  insufficient-contrast path.
- Review queue: composite sort order on constructed ties; limit.
- Opening clusters: key construction, grouping, score rates.
- `validate_replay`: every violation raises (mismatched identity, bad ply
  alternation, wrong move count, missing keys, wrong schema_version).

CLI integration (`tests/test_eval_loss_replay_analyzer.py`) in `tmp_path`:

- Synthetic `_games.jsonl` + sidecars → all six artifacts written, expected
  columns/shapes, summary JSON carries verdict + filters; console ends with the
  verdict + queue lines.
- A V1-era file without `replay_path` keys → skipped with a note, exit 0.
- A focus-window row with `replay_path: null` → non-zero exit, clear error.
- Threshold-sanity violations → exit 2 with message.

## Acceptance (manual, after implementation)

Run the CLI on the real seed-35791 capture; confirm cohort counts match the
dataset (172 losses), artifacts land in `logs/eval/loss_analysis_v2/`, and the
verdict line directly answers the focus question. Read the top of the manual
review queue against a few replay sidecars to sanity-check the collapse labels.
