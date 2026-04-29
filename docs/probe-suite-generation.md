# Probe Suite Generation

Operator-facing reference for `scripts/build_probe_suite.py` — the offline tool that produces the curated probe suites used as yardsticks for value-head quality across training iterations.

**This is not training.** The labeling network's weights are frozen. The output is a measurement artifact (a JSON file of positions + ground-truth labels) that the trainer then scores every candidate checkpoint against.

## What it produces

| Tier | File | Use |
|---|---|---|
| `forced` | `tests/probes/twixt_probes.json` | Near-terminal positions (1–2 plies from a winning move). Sanity floor; gate target ≥95% sign-correct |
| `strong_advantage` | `tests/probes/strong_advantage_probes.json` | Strong but not-yet-forced chain advantages (3–8 plies from terminal). Detects value-head bias on dominant-but-not-trivial positions; gate target ≥80% sign-correct, **not yet active as a hard gate** |

Both tiers share the same JSON shape (probe entries with `id`, `category`, `confidence`, `side_to_move`, `expected_value_sign`, `move_history`, etc.). The strong-advantage tier adds two per-probe blocks (`phase1_features`, `phase2_label`) recording how each probe was selected and labeled. See [`tests/probes/README.md`](../tests/probes/README.md) for the full schema.

## How it helps training

The trainer reads these files at startup. On every iteration, after producing a candidate checkpoint, it:

1. Loads each probe's `move_history`, replays it to a `TwixtState`.
2. Runs a single forward pass through the candidate network.
3. Compares the network's value sign against the probe's `expected_value_sign`.
4. Logs `sign_correct/n`, `median |v|`, deltas, and a rolling-5 average.

Output lands in:
- `checkpoints/<dir>/iteration_<N>.json` (sidecar) — `forced_probe_summary` and `probe_summary.<tier>` blocks
- `report.txt` (analyzer output) — per-tier sections with trend tables
- `<tier>_probe_by_iter.csv` (analyzer output) — full per-iter time series, easy to plot

This gives a stable signal that's independent of self-play noise. Training-loop metrics like `loss` and `winrate` move with self-play randomness; probe scores only move when the value head genuinely changes.

## Generation pipeline

`build_probe_suite.py --tier <forced|strong_advantage>` runs three phases:

- **Phase 1 — Mine** decisive games for candidates. Forced tier picks positions 1-2 plies from terminal; strong-advantage tier picks 3-8 plies from terminal and applies structural-dominance heuristics (chain size, axis span, goal-touch, span margin, centroid).
- **Phase 2 — Label** (strong-advantage only). Runs deep MCTS at `--label-mcts-sims` × `--label-mcts-repeats` per candidate. Admits only candidates where the search sign matches the source-game winner AND magnitude/concentration/stability all clear thresholds.
- **Phase 3 — Review.** Operator opens the `*.draft.json`, eyeballs the list (~10–20 minutes for 30 probes), removes anything that looks like a tactical trap, then runs `--promote --reviewer NAME` to copy the draft to the committed file with timestamp and reviewer name stamped into the meta.

The forced tier has no Phase 2 (positions 1–2 plies from terminal don't need MCTS confirmation; the source game is the ground truth). The strong-advantage tier needs Phase 2 because "is red genuinely winning here?" is the kind of question that requires search to answer.

## Knobs

| Flag | Default | Meaning |
|---|---|---|
| `--tier` | required | `forced` or `strong_advantage` |
| `--input` | `scripts/GPU/logs/games` | Directory of `iter_NNNN_game_MMM.json` source replays |
| `--source-iter-range MIN MAX` | required | Inclusive iter range to mine from. Most recent iters give labels best aligned with current play |
| `--label-checkpoint` | required for strong_advantage | `.safetensors` weights for the Phase 2 labeler. Currently must use the `create_network` default architecture (hidden=128, n_blocks=6) |
| `--label-mcts-sims` | 10000 | Sims per MCTS run during Phase 2 labeling. Higher = more trustworthy labels, more wallclock |
| `--label-mcts-repeats` | 3 | Number of repeated MCTS runs per candidate (with different seeds). Higher = stricter stability check |
| `--magnitude-threshold` | 0.45 | Phase 2 admission: `abs(mean_root_value) >= this`. Lower = admits more borderline positions |
| `--top1-share-floor` | 0.15 | Phase 2 admission: most-visited move must hold this fraction of total visits. Higher = rejects positions where MCTS sprays visits across many continuations |
| `--stability-cap` | 0.15 | Phase 2 admission: `max(value_per_run) - min(value_per_run) <= this`. Higher = tolerates more noise across repeated MCTS runs |
| `--max-probes` | 30 | Final cap on admitted probes (post-filter) |
| `--max-probes-per-game` | 2 | Maximum probes from any single source game, total across all 4 categories. Combined with the internal `MIN_PLY_SEPARATION_SAME_GAME=3` constant, no single game contributes more than 2 probes and any 2 it contributes are at least 3 plies apart in the source trajectory. Strong-advantage tier only. Default 2 is conservative; raise only if the suite is consistently undersized. |
| `--label-worker-mode` | process | Phase 2 execution mode. Default `process` runs a `ProcessPoolExecutor` of `--label-workers` workers. Pass `serial` for the single-process byte-reference path (slower; useful for debugging or strict reproducibility). |
| `--label-workers` | 10 (process) / 1 (serial) | Worker count under `--label-worker-mode=process`. Default 10 (M3 Pro optimum). Use `scripts/probes/benchmark_phase2_knobs.py` to tune for your machine. Ignored under `serial`. |
| `--mcts-eval-batch-size` | 14 | NN batch size for the labeler's MCTS. Capped at 14 because larger batches have caused Metal hangs; pass `--allow-unsafe-eval-batch` to exceed. |
| `--mcts-stall-flush-sims` | 16 | MCTS stall-flush threshold (see `MCTSConfig`). 0 disables. |
| `--allow-unsafe-eval-batch` | flag | Required to set `--mcts-eval-batch-size > 14`. Intended only for local benchmarking. |
| `--admission-borderline-epsilon` | 0.01 | In process mode, candidates whose phase-2 label is within ε of any admission threshold are re-labeled in the main process to use the serial reference label. 0 disables. |
| `--no-borderline-rerun` | flag | Disable borderline rerun even when ε > 0. Used for benchmarking the raw process-pool path. |
| `--out` | per-tier default | Output JSON path |
| `--promote` | flag | Copy `*.draft.json` to committed file with reviewer + timestamp stamped into meta |
| `--reviewer NAME` | required with `--promote` | Reviewer attribution recorded in `meta.reviewer` |
| `--force` | flag | Overwrite existing draft (during Phase 1+2) or committed file (during `--promote`). Default refuses |

## When to regenerate

The suites are designed to be **stable yardsticks** — frequent regeneration defeats the purpose because metrics across iterations stop being comparable. Refresh only when:

1. **Structural change to game encoding.** Channel-count bumps, new connectivity channels, etc. The previous suite's labels were against a different feature representation; refresh against the new one. The committed `meta.label_checkpoint_sha256` records labeler identity so any drift is visible.
2. **Materially better labeler available.** The current strong-advantage suite is labeled by `model_iter_0059`. If a checkpoint many iterations later has clearly better positional understanding, its MCTS-derived labels would be more trustworthy. Refresh against the newer checkpoint and compare scores against both old and new suites for one or two iterations to confirm the new suite is actually stricter.
3. **Persistent stagnation pattern.** If `probe_summary.strong_advantage.sign_correct_pct` plateaus below ~60% for 20+ iterations with no other quality regression visible (training loss looks fine, win rate against heuristic baseline is improving), suspect the suite — the original labeling network may have shared the bias the current network is also exhibiting. Regenerate with a stronger checkpoint as the labeler.
4. **Self-play distribution shift.** Curriculum bumps, dirichlet alpha changes, large opening-exploration changes — these change the distribution of strong-advantage positions in self-play. Regenerate so the suite samples from the new distribution.
5. **Forced parity test fails because the source replay corpus changed.** If `tests/probes/twixt_probes.json` regenerates to byte-different output (e.g., training was rerun and overwrote `scripts/GPU/logs/games/iter_25-30/`), regenerate as a deliberate `data(probes): …` commit. Recovery procedure documented in [`tests/probes/README.md`](../tests/probes/README.md).

In normal operation, regeneration is a **once-every-50-to-100-iterations** activity at most. Day-to-day training runs the existing suites untouched.

## What signals tell you a regen is needed

Watch the per-iter sidecar fields and the analyzer's CSV outputs:

| Signal | Look at | Action |
|---|---|---|
| `forced_probe_summary.sign_correct_pct` plateaus below 90% for 20+ iters | `forced_probe_by_iter.csv` | Forced tier is supposed to be a high floor. If checkpoint can't get sign-correct on near-terminal positions, the value head is broken — fix the network, not the probes |
| `probe_summary.strong_advantage.sign_correct_pct` drops sharply (>10pp in a single iter) | `strong_advantage_probe_by_iter.csv` | Likely a real value-head regression. Investigate training-loop changes, not the suite |
| `strong_advantage` plateaus 50-65% for many iters with no other regression | `strong_advantage_probe_by_iter.csv` + win-rate vs heuristic | Suspect a shared bias between the labeling checkpoint and the current network. Regenerate with a more recent checkpoint as the labeler |
| `forced_probe_by_iter.csv` median \|v\| trends toward 0 | sidecar trend | Network is becoming uncertain on positions it should be confident about. Could be LR collapse or overregularization. Investigate training, not the suite |
| Both tiers' `sign_correct_pct` improve in lockstep with win rate | both CSVs | Healthy. Don't regenerate. The suites are doing their job |

## First-time setup workflow

```bash
# 1. Generate forced tier (fast, no Phase 2). Auto-runs as part of training
#    if absent — also runnable on demand:
.venv/bin/python scripts/build_bootstrap_probe_suite.py \
    --source-iter-range MIN MAX \
    --out tests/probes/twixt_probes.json

# 2. Generate strong-advantage tier (slow — minutes to hours depending on
#    --source-iter-range and --label-mcts-sims). For a first pass with a
#    cheap label budget:
.venv/bin/python scripts/build_probe_suite.py \
    --tier strong_advantage \
    --input scripts/GPU/logs/games \
    --source-iter-range 50 58 \
    --label-checkpoint checkpoints/alphazero-v2-staged/model_iter_0059.safetensors \
    --label-mcts-sims 2000 \
    --label-mcts-repeats 2 \
    --max-probes 30 \
    --out tests/probes/strong_advantage_probes.json
    # For strict byte-reproducibility, add: --label-worker-mode serial
# Phase 2 prints per-candidate progress (every ~5%, with ETA).

# 3. Eyeball the draft.
$EDITOR tests/probes/strong_advantage_probes.draft.json

# 4. Promote.
.venv/bin/python scripts/build_probe_suite.py \
    --tier strong_advantage --promote \
    --reviewer "$(git config user.name)" \
    --out tests/probes/strong_advantage_probes.json

# 5. Commit the committed file (the .draft.json and audit are gitignored).
git add tests/probes/strong_advantage_probes.json
git commit -m "data(probes): seed committed strong_advantage suite"
```

## Cost guide (rough)

Phase 2 throughput is dominated by MCTS, which is dominated by NN forward passes. On Apple M-series GPUs:

- ~250 MCTS sims/sec total throughput (single-process, no batching benefits at this granularity).
- A first-pass run with 200-400 candidates × 2k sims × 2 repeats ≈ 1-2 hours.
- A high-confidence run with 200-400 candidates × 10k sims × 3 repeats ≈ 6-12 hours.
- A whole-corpus run (1500+ candidates × 10k × 3) is ~50+ hours and probably unnecessary.

For a first iteration: use `--label-mcts-sims 2000 --label-mcts-repeats 2` and a narrow source-iter range (e.g., 5-9 iters). If the admitted set looks healthy, optionally do a tighter run with the full budget — but the lower-budget labels are usually good enough.

### Parallel labeling

Phase 2 runs in a `ProcessPoolExecutor` by default (`--label-worker-mode process` with `--label-workers 10`), one MLX network per worker. The default 10 workers is the M3 Pro optimum validated by `scripts/probes/benchmark_phase2_knobs.py`. For strict byte-reproducibility (e.g., generating golden test fixtures), pass `--label-worker-mode serial` to use the single-process reference path.

To tune for a different machine, run `scripts/probes/benchmark_phase2_knobs.py` to sweep worker counts and pick the best for your hardware. Avoid increasing `--mcts-eval-batch-size` above 14 unless intentionally benchmarking with `--allow-unsafe-eval-batch`. Higher worker counts are not always faster: each process loads its own MLX network and can contend for the Metal scheduler.

`--label-worker-mode process --label-workers 1` is valid (useful for testing worker initialization and deterministic reassembly), but it is not expected to speed up the run.

**Reproducibility under parallel mode.** Serial mode is the strict reference path for mocked/deterministic labelers and the supported strict reproducibility mode for generated artifacts. For real MLX runs, the supported target is identical admitted probe IDs, identical final committed probe IDs, and identical rejection reasons for non-borderline candidates under normal deterministic MLX behavior. Borderline rerun (`--admission-borderline-epsilon`, default 0.01) re-labels threshold-sensitive candidates in the main process so admission decisions match serial. Byte-identical numeric labels are not promised across machines, worker counts, or MLX versions for real runs.

`meta.phase2_run_stats` in the draft JSON records the mode, worker counts (requested and effective), MCTSConfig values, per-status counters, and borderline rerun counters for postmortem reproduction.

## Diversity selector (strong_advantage tier)

After Phase 2 admits candidates, a category-aware round-robin selector walks the four `chain_advantage_*` buckets in fixed canonical order — `central_red, central_black, edge_red, edge_black` — and applies three suppression rules in precedence order before admitting each candidate:

- **Rule A — `diversity_near_duplicate`.** Same `source_game` AND same `category` AND `|Δcc_size| < 2` AND `|Δaxis_span_margin| < 0.05`. The candidate is a structural near-duplicate of an already-kept sibling. Cross-category same-game pairs are NOT deduped (the category boundary is structurally meaningful).
- **Rule B — `diversity_ply_too_close`.** Same `source_game` AND `|Δsource_ply| < 3` (any category — Rule B is category-agnostic). The candidate sits too close in the source trajectory to an already-kept sibling. The closer keeper wins; if equidistant, better Stage-2 rank wins; if still tied, smallest `source_ply` wins.
- **Rule C — `diversity_per_game_cap`.** The source game already has `--max-probes-per-game` keepers (default 2, total across categories). The candidate is dropped to keep no single game over-represented.

Each diversity-driven drop produces one audit row with the corresponding `reason`, the full `phase2_label`, and a `kept_instead_source_ply` field pointing at the keeper that triggered the drop. `reason="admitted"` rows correspond exactly 1:1 with the probes in the committed suite — Phase 1 and Phase 2 no longer write `admitted` rows; only the selector does.

Within each category, candidates are ranked by a structural-first key:
`cc_size desc → axis_span_margin desc → cc_axis_span desc → min_top1_share desc → value_stability asc → (-iter, -source_ply, source_game)`.
Structural fields dominate so the rank is invariant to labeling-checkpoint noise; the existing `_sort_key` provides total-order determinism on full ties.

The selector's configuration is recorded in `meta.selection_rules` so a reviewer can verify which policy produced the suite:

- `max_probes_per_game` — the value of the CLI flag at generation time.
- `min_ply_separation_same_game` — fixed at 3 in the current implementation (tied to the K-range `[3, 8]`).
- `category_iteration_order` — the canonical 4-tuple the round-robin walks.
- `diversity_quality_key_order` — the Stage-2 sort precedence above.

**Audit-coverage policy:** the selector writes one audit row per candidate it considers (admitted or diversity-dropped). Once `--max-probes` is reached, remaining candidates are not visited and produce no audit row — the audit is exhaustive over considered candidates only. For a total Phase-2-admit count, see the `Phase 2 complete: ... (N admitted ...)` line printed by the generator before selection runs.

## Determinism

Generator output is byte-reproducible given identical inputs:
- Probe IDs derive from `source_game` + `source_ply` + `category` (deterministic).
- Per-candidate MCTS RNG seeds are `int.from_bytes(sha256(probe_id)[:4], "big") ^ repeat_idx` — stable across processes (Python's built-in `hash()` is process-randomized and can't be used for this).
- Audit files (`candidates_strong_advantage.json`) and draft files re-emit identically given the same source corpus and same labeler checkpoint.

The committed `meta.selection_rules.label_checkpoint_sha256` records the labeler identity so that if anyone questions a probe later, they can verify the labels came from the expected weights.

## Architectural limitation (current)

The strong-advantage generator only supports labeling checkpoints built with `create_network` defaults (`hidden=128`, `n_blocks=6`). `load_network_for_scoring` auto-detects input channels (24 vs 30) but does NOT auto-detect `hidden` or `n_blocks`. To label against a different architecture, the generator must first be extended with `--hidden` / `--blocks` flags — out of scope for the current iteration.

## Tests

| Test file | What it covers |
|---|---|
| `tests/test_probe_suite_forced_parity.py` | CI-enforced byte-identical regeneration of `tests/probes/twixt_probes.json` from `meta.selection_rules`. Protects against silent drift during refactors |
| `tests/test_strong_advantage_probe_suite.py` | Schema, ID determinism, admission filter (clause-by-clause), promotion workflow, perspective contract (red-perspective vs STM-perspective), canonical-schema regression |
| `tests/test_strong_advantage_smoke_live.py` | Marker-gated (`-m slow_live`) end-to-end pipeline test. Runs the full generator at sims=200 repeats=1 against a real checkpoint. Plumbing-only; not a label-quality test |
| `tests/test_strong_advantage_analyzer_aggregation.py` | Tier-parameterized analyzer (`_read_tier_summary`, `format_tier_probe_report`) plus an end-to-end test that runs the analyzer against synthetic sidecars |
| `tests/test_trainer_probe_summary_emission.py` | Verifies the trainer's per-iter sidecar emits `probe_summary.{forced,strong_advantage}` alongside the legacy `forced_probe_summary` field |
