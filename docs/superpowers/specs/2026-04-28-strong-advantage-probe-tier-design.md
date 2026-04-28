# Strong-Advantage Probe Tier — Design

- **Date:** 2026-04-28
- **Status:** Approved (brainstorming complete; ready for implementation planning)
- **Related:** [2026-04-21-probes-and-calibration-closure-design.md](2026-04-21-probes-and-calibration-closure-design.md), [tests/probes/README.md](../../../tests/probes/README.md)

## Context

The current bootstrap probe suite (`tests/probes/twixt_probes.json`) populates only the `confidence='forced'` tier — 30 probes, all `near_win_*`. The schema reserves a second tier, `confidence='strong_advantage'` (80% gate target, per the closure spec), but it is unpopulated.

In recent live games against `model_iter_0059`, the value head systematically mis-evaluates positions where the eventual winner has built a strong but not-yet-forced chain — confidently saying the eventual winner is losing through plies 17–49 of a game they win at ply 51. Because the existing probe suite only exercises near-terminal positions, this failure mode is invisible to the current per-iter telemetry.

Adding a `strong_advantage` tier closes the observability gap: it produces a metric that tells us specifically whether the network's value head is improving on dominant-but-not-forced positions, separately from the forced tier (which may stay at ≥95% even while `strong_advantage` scoring is poor).

## Goals

- Populate the `strong_advantage` tier with 20–30 bootstrap-quality probes covering both colors and both central / edge geometries.
- Plumb per-iter telemetry for the new tier through the trainer → analyzer → `summary.json` / `report.txt` / CSV pipeline so its evolution across training iterations is trackable.
- Refactor the probe generator into a tier-parameterized form without disturbing the existing forced-tier output.
- Provide a reproducible operator workflow (Phase 1 mining → Phase 2 deep-MCTS labeling → Phase 3 light review → promote).

## Non-goals

- Not a formal §7 gate suite. This is bootstrap-quality with light operator review; the full reviewer-curated gate suite is a separate (future) workstream.
- Not promoting `strong_advantage` to a hard CI gate threshold yet. We measure first, then choose a threshold.
- Not regenerating historical replay CSVs (`Replays/*/forced_probe_by_iter_*.csv`); the new tier appears starting next replay run.
- Not retraining the model. This is a measurement add-on; whether to retrain is a separate decision informed by what this tier reveals.

## Approach overview

A 3-phase generation pipeline (Approach 2 from brainstorming):

- **Phase 1 — Candidate mining (heuristic):** scan decisive games, sample positions 3–8 plies before terminal, apply structural-dominance heuristics to extract candidates.
- **Phase 2 — Deep-MCTS labeling (label authority):** run 10,000-sim MCTS × 3 repeats per candidate against `model_iter_0059`; admit only candidates that pass the strict admission filter (sign-agreement with source winner + magnitude + concentration + stability + not-already-forced).
- **Phase 3 — Light review:** generator emits a draft file; operator eyeballs (~10–20 min) and runs `--promote` to commit.

The generator is shared between tiers (tier-parameterized CLI: `--tier {forced,strong_advantage}`); per-tier output files keep the bootstrap-vs-formal-gate distinction intact; trainer telemetry is keyed by tier; analyzer aggregation is parameterized over tier names rather than duplicated.

## File layout

### New files

- `scripts/build_probe_suite.py` — tier-parameterized generator (replaces `build_bootstrap_probe_suite.py` as the real implementation).
- `tests/probes/strong_advantage_probes.json` — bootstrap strong-advantage tier output (deep-MCTS labeled, light-reviewed). Separate from any future formal gate suite.
- `tests/test_strong_advantage_probe_suite.py` — schema + ID determinism + admission-filter rules + promotion workflow.
- `tests/test_probe_suite_forced_parity.py` — critical parity gate: regenerating `--tier forced` produces byte-identical `tests/probes/twixt_probes.json` to what's committed.
- `tests/test_strong_advantage_analyzer_aggregation.py` — verify analyzer correctly populates new aggregates from synthetic sidecar fixtures.

### Modified files

- `scripts/build_bootstrap_probe_suite.py` — 5-line shim that calls `build_probe_suite.py --tier forced` (preserves operator muscle memory and existing CI/cron commands).
- `scripts/GPU/alphazero/probe_eval.py` — extract `extract_forced_probes_from_games` into a tier-parameterized internal API; add `extract_strong_advantage_candidates` (Phase 1) and the deep-MCTS labeling step (Phase 2).
- `scripts/GPU/alphazero/trainer.py` — emit a tier-keyed `probe_summary` dict per-iter sidecar; keep `forced_probe_summary` populated for one release cycle (deprecation window).
- `scripts/twixt_replay_analyzer.py` — add tier-parameterized aggregation (`<tier>_probe_by_iter` / `<tier>_probe_latest`); add `<tier>_probe` block to `summary.json`; add `<tier>_probe_by_iter.csv` companion CSV; add `format_<tier>_probe_report` section in `report.txt`. One helper, looped over tier names — not a copy-paste.
- `tests/probes/README.md` — document the new file, generator CLI, admission filter, telemetry path, and the deprecation note for `forced_probe_summary`.

### Untouched

- Existing `tests/probes/twixt_probes.json` content (parity test guarantees).
- Historical CSVs in `Replays/*` (the new tier appears starting next replay run).
- Existing `forced_probe` block in `summary.json` and existing `forced_probe_by_iter.csv` (parity-tested as-is).
- The bootstrap-vs-formal-gate distinction documented in the README.

## Telemetry data flow

### Per-iteration trainer output

The trainer continues to write the existing `forced_probe_summary` block to per-iter sidecar JSON (during the one-release deprecation window). It also writes a sibling tier-keyed block:

```json
{
  "forced_probe_summary": { ... existing, unchanged ... },
  "probe_summary": {
    "forced": { ... same payload as forced_probe_summary ... },
    "strong_advantage": {
      "n": 28, "n_skipped_size": 0,
      "sign_correct": 19, "sign_correct_pct": 67.9,
      "median_abs_v": 0.41,
      "delta_sign_correct_pct": 3.6,
      "delta_median_abs_v": 0.05,
      "rolling5_sign_correct_pct": 64.3,
      "rolling5_median_abs_v": 0.38
    }
  }
}
```

`probe_summary.<tier>` is the forward path; `forced_probe_summary` is legacy compatibility for one release cycle.

### Analyzer aggregation

`scripts/twixt_replay_analyzer.py` reads `sc.get("probe_summary", {}).get("<tier>")` for each known tier name in a list, populating `agg["<tier>_probe_by_iter"]` and `agg["<tier>_probe_latest"]`. The aggregation helper is parameterized over tier name (one loop, not duplicated code per tier). Falls through silently if the field is absent (older sidecars without `probe_summary` still aggregate the legacy `forced_probe_summary` field).

### `summary.json` shape

```json
{
  "forced_probe": { "by_iter": [...], "latest": {...} },
  "strong_advantage_probe": { "by_iter": [...], "latest": {...} }
}
```

Per-tier human-friendly keys are preserved during the transition (no nested generic `probes` object).

### `report.txt` section (parallel to existing forced section)

```
Strong-advantage probes (bootstrap, deep-MCTS labeled):
  Latest iter: 0072 - n=28, sign_correct=19/28 (67.9%), median |v|=0.41
  Delta vs prev: +3.6 pp sign-correct, +0.05 median |v|
  Rolling-5: 64.3% sign-correct, median |v|=0.38
  ... full per-iter table: strong_advantage_probe_by_iter.csv
```

### New CSV: `strong_advantage_probe_by_iter.csv`

Columns: `iteration, n, n_skipped_size, sign_correct, sign_correct_pct, median_abs_v, delta_sign_correct_pct, delta_median_abs_v, rolling5_sign_correct_pct, rolling5_median_abs_v`. Same shape as the existing forced CSV so analysis tooling can ingest both with one helper.

## Generator internals

### CLI surface

```
build_probe_suite.py --tier strong_advantage \
    --input scripts/GPU/logs/games \
    --source-iter-range 50 80 \
    --label-checkpoint checkpoints/alphazero-v2-staged/model_iter_0059.safetensors \
    --label-mcts-sims 10000 \
    --label-mcts-repeats 3 \
    --magnitude-threshold 0.45 \
    --top1-share-floor 0.15 \
    --stability-cap 0.15 \
    --max-probes 30 \
    --out tests/probes/strong_advantage_probes.json
```

`--out` defaults to `tests/probes/<tier>_probes.json` (where `<tier>` is `twixt` for `--tier forced` to preserve the existing committed filename, else `strong_advantage`). Tier-specific flags (`--label-mcts-sims`, etc.) are accepted but only meaningful when `--tier strong_advantage`. `--dry-run` produces draft + audit but does not suggest promotion. `--seed` optional; if omitted, RNG is derived deterministically from probe IDs.

### Phase 1 — Candidate mining

For each replay in `--source-iter-range`:
1. Skip non-decisive games (`winner_reason != "win"`, draws, timeouts).
2. Walk the game; extract candidate positions at `terminal_ply - K` for K in `[3, 8]`.
3. Compute structural features for the eventual winner:
   - `cc_size` — size of largest connected peg component.
   - `cc_axis_span` — fraction of goal axis the component spans.
   - `cc_touches_own_goal` — boolean: largest CC touches at least one of the player's two goal edges (red: row 0 or row 23; black: col 0 or col 23).
   - `axis_span_margin` — `winner_cc_axis_span - loser_cc_axis_span`. Relative-strength feature so we don't admit positions where the "winner" is dominant in absolute but the loser is just as advanced.
   - `centroid_chebyshev_from_center` — used for central / edge category assignment.
   - `forced_within_2` — boolean from the existing forced detector. Must be `False` (the whole point is strong-but-not-forced).
4. Phase-1 admission gate (heuristic; not the final label gate):
   - `cc_size >= 10`
   - `cc_axis_span >= 0.55`
   - `axis_span_margin >= 0.10`
   - `cc_touches_own_goal == True`
   - `forced_within_2 == False`
5. Dedup by exact + 4-form mirror canonicalization (same as the existing forced builder).

Typical yield: 100–300 raw candidates per source-iter range.

### Phase 2 — Deep-MCTS labeling (label authority)

For each candidate:
1. Build the `TwixtState` from `move_history`.
2. Run MCTS at `--label-mcts-sims` sims × `--label-mcts-repeats` repeats, with **different RNG seeds per repeat** (seeded as `hash(probe_id) ^ repeat_index`, stored as a resolved integer in the per-probe meta).
3. Aggregate across repeats: `mean_root_value`, `value_per_run`, `value_stability = max(root_value) - min(root_value)`, `min_top1_share`.
4. **Admission filter** — admit only if **all** are true:
   - `sign(mean_root_value) == sign_of(source_winner)` (the cross-check that protects against same-model self-labeling bias).
   - `abs(mean_root_value) >= --magnitude-threshold` (default 0.45).
   - `min_top1_share >= --top1-share-floor` (default 0.15).
   - `value_stability <= --stability-cap` (default 0.15).
   - `forced_within_2 == False` (re-verified after labeling — defensive).
5. Per-candidate audit row written to `candidates_strong_advantage.json` (gitignored, parallel to the existing `candidates.json`). Each row records the candidate's source_game/source_ply, the computed Phase-1 features and Phase-2 label values, and — if the candidate was dropped — the failure stage and reason. Phase-2 audit reasons:
   - `sign_mismatch`, `magnitude_below_threshold`, `low_top1_share`, `unstable_value`, `position_already_forced`.
   - Phase-1 drops are similarly recorded with reasons matching the failed clause (`phase1_cc_size`, `phase1_axis_span`, `phase1_axis_span_margin`, `phase1_no_goal_touch`, `phase1_already_forced`, `category_midband`).

Typical cull rate: 60–80% of Phase-1 candidates dropped at this stage.

### Phase 3 — Light review

Generator stops at Phase 2 and emits `tests/probes/strong_advantage_probes.draft.json` plus a one-page summary printed to stdout (per-probe row with `id`, `category`, `cc_size`, `cc_axis_span`, `mean_root_value`, `min_top1_share`, source iter/game, ASCII rendering of the position). Operator eyeballs the list (~10–20 min for 30 probes), removes any that look like tactical traps, then runs:

```
build_probe_suite.py --promote --tier strong_advantage --reviewer "<name>"
```

`--promote` copies `*.draft.json` → `tests/probes/strong_advantage_probes.json`, populating `meta.review_mode = "light_review"`, `meta.reviewer`, `meta.reviewed_at_utc` so we can tell apart probes that have been eyeballed from raw generator output.

### Categories assigned during Phase 1

- `chain_advantage_central_red` / `chain_advantage_central_black` — centroid Chebyshev distance ≤6 from board center.
- `chain_advantage_edge_red` / `chain_advantage_edge_black` — centroid in outer band (Chebyshev ≥9 from center).
- **Mid-band candidates** (centroid Chebyshev distance 7 or 8) are dropped at Phase 1 to keep the central/edge separation crisp. Audit row records `category_midband` as the drop reason. This prevents weak examples of either category from diluting the signal.

Target distribution: ~7–8 probes per category, ~30 total. **Falling-short rule:** if any category has fewer than 5 surviving probes after Phase 2, the generator emits a warning so the operator can broaden the source iter range or relax the magnitude threshold.

### Determinism

Identical inputs (source iter range, checkpoint, sim budget, RNG seeds derived from probe IDs) produce byte-identical output. RNG seed for each MCTS repeat is a resolved integer (`hash(probe_id) ^ repeat_index`), recorded into per-probe meta so a re-run can be byte-reproduced.

## Schema

### Probe entry

```json
{
  "id": "iter_0072_game_014_ply038_chain_advantage_central_red",
  "category": "chain_advantage_central_red",
  "confidence": "strong_advantage",
  "side_to_move": "red",
  "expected_value_sign": 1,
  "active_size": 24,
  "ply": 38,
  "move_history": [[r,c], [r,c], ...],
  "source_game": "iter_0072/game_014.json",
  "source_ply": 38,
  "starting_player": "red",
  "phase1_features": {
    "cc_size": 14,
    "cc_axis_span": 0.74,
    "cc_touches_own_goal": true,
    "axis_span_margin": 0.18,
    "centroid_chebyshev_from_center": 4
  },
  "phase2_label": {
    "mean_root_value": 0.62,
    "value_per_run": [0.58, 0.65, 0.63],
    "value_stability": 0.07,
    "min_top1_share": 0.22,
    "label_checkpoint": "model_iter_0059.safetensors",
    "label_mcts_sims": 10000,
    "label_mcts_repeats": 3,
    "rng_seed_base": 1493847291
  }
}
```

`phase1_features` and `phase2_label` blocks make every admission decision auditable from the file alone — no need to re-run anything to see why a probe was admitted. `rng_seed_base` is the resolved integer (not a string description) so a re-run is byte-reproducible.

### Meta block

```json
{
  "type": "bootstrap_rule_selected",
  "tier": "strong_advantage",
  "not_gate_suite": true,
  "review_mode": "light_review",
  "reviewer": "bill",
  "reviewed_at_utc": "2026-04-28T03:45:12Z",
  "generator": "scripts/build_probe_suite.py",
  "generator_version": 1,
  "selection_rules": {
    "board_size": 24,
    "winner_reasons": ["win"],
    "k_plies_from_terminal_range": [3, 8],
    "phase1_thresholds": {
      "min_cc_size": 10,
      "min_cc_axis_span": 0.55,
      "min_axis_span_margin": 0.10,
      "require_cc_touches_own_goal": true,
      "exclude_forced_within_2": true
    },
    "phase2_thresholds": {
      "label_mcts_sims": 10000,
      "label_mcts_repeats": 3,
      "min_magnitude": 0.45,
      "min_top1_share": 0.15,
      "max_value_stability": 0.15,
      "require_sign_match_source_winner": true
    },
    "label_checkpoint": "checkpoints/alphazero-v2-staged/model_iter_0059.safetensors",
    "label_checkpoint_sha256": "<64-char hex>",
    "source_iter_range": [50, 80],
    "dedup": "exact + 4-form-mirror-canonical",
    "category_min_count": 5
  }
}
```

`label_checkpoint_sha256` is hashed at generation time and recorded in meta so a reviewer can confirm labels were produced against a known weights blob — even if the file is later overwritten.

## Edge cases

| Case | Behavior |
|---|---|
| `--label-checkpoint` file missing | Hard error before Phase 1 starts. |
| Checkpoint architecture (hidden / blocks) differs from `create_network` defaults (128 / 6) | Weight-loading fails with a tensor-shape mismatch from `network.load_weights(...)`. `load_network_for_scoring` only auto-detects input channels (24 vs 30); `hidden` and `n_blocks` use defaults. Operator must use a checkpoint matching the default architecture, or extend the generator with `--hidden` / `--blocks` flags in a follow-up. |
| Phase-1 yields zero candidates in a category | Warn, continue. Per-category falling-short warning fires if final count < 5. |
| Phase-2 yields zero admitted probes overall | Hard error, with per-category drop reasons aggregated so operator sees *why*. No `*.draft.json` written. |
| `*.draft.json` already exists | Refuse to overwrite without `--force`. |
| `--promote` invoked but no `*.draft.json` | Error: nothing to promote. |
| `--promote` with committed file already present | Refuse without `--force`. Promotion is one-way; overwrites must be deliberate. |
| MCTS errors mid-labeling (NaN, abort) | Skip candidate, log to stderr with source_game/source_ply, continue. |
| Repeat runs return inconsistent winners | Counted as `unstable_value` rejection; per-run sign disagreement recorded in audit. |
| Source-game `winner` field missing | Skip with one-line warning; don't crash. |

## Tests

### `tests/test_probe_suite_forced_parity.py` — critical safety gate

Reads the committed `tests/probes/twixt_probes.json`, pulls `meta.selection_rules` and `meta.generator` out of it (so the test isn't pinned to a specific source-iter-range — it follows whatever the committed file used), invokes the new generator entrypoint with those same args targeting a tmp file, then `assert tmp.read_bytes() == committed.read_bytes()`. **CI-enforced**: required check on probe-related changes; runtime ~10–60 seconds.

**Assumed stable inputs:** the source replay JSONs that the committed forced suite was generated from must remain on disk and unchanged. Concretely, this means `scripts/GPU/logs/games/iter_NNNN_game_MMM.json` for the iter range recorded in `meta.selection_rules.source_iter_range` must be present and byte-identical to when the committed suite was produced. If those files are ever moved, deleted, or rewritten, the parity test will fail spuriously and the committed suite must be regenerated against the new replay set (with a deliberate commit). This assumption is documented in `tests/probes/README.md`.

If this test ever turns red after a refactor, the refactor is wrong, full stop.

### `tests/test_strong_advantage_probe_suite.py` — schema, IDs, admission, promotion

1. **Schema validation** — every probe has the required top-level fields, `phase1_features` (5 keys), `phase2_label` (8 keys), valid `confidence == "strong_advantage"`, valid category from the 4-element enum, valid `expected_value_sign` and `side_to_move`.
2. **Meta validation** — `tier == "strong_advantage"`, `review_mode == "light_review"`, `reviewer` non-empty, `label_checkpoint_sha256` is a 64-char hex, all `phase2_thresholds` numeric ranges sane.
3. **ID determinism** — small synthetic fixture (3 candidates), run canonicalizer twice, assert IDs stable across runs and across input reordering.
4. **Admission filter unit tests** — for each filter clause, synthesize a candidate that passes everything else and fails just that one clause, assert it's rejected with the correct audit reason. Clauses tested independently:
   - `sign_mismatch`, `magnitude_below_threshold`, `low_top1_share`, `unstable_value`, `position_already_forced`.
5. **Phase-1 edge case** — candidate where the LOSER has a longer span than the winner: must be rejected via `axis_span_margin < 0.10` even if winner alone looks dominant.
6. **Category assignment** — three positions (centroid at center, at edge, mid-band) → assigned to the right `chain_advantage_*` category for the right color.
7. **Promotion workflow** — `--promote` with no draft errors; `--promote` with draft writes file with `reviewer`/`reviewed_at_utc` populated; second `--promote` without `--force` errors; with `--force` overwrites and updates the timestamp.

Labeling is mocked: tests inject a stub MCTS labeler returning predetermined `(value, top1_share)` per candidate.

### `tests/test_strong_advantage_analyzer_aggregation.py`

Synthetic sidecar fixtures: tmp directory with three fake `iter_NNNN.json` files containing minimal structure plus a `probe_summary.strong_advantage` block with canned numbers.

Asserts:
1. `agg["strong_advantage_probe_by_iter"]` has 3 rows in iter order.
2. `agg["strong_advantage_probe_latest"]` matches latest iter's payload.
3. `summary.json` has both `forced_probe` and `strong_advantage_probe` blocks.
4. `strong_advantage_probe_by_iter.csv` written with right columns and values.
5. `report.txt` includes the strong-advantage section with latest pct and delta-vs-prev.
6. **Backward-compat:** sidecar with only legacy `forced_probe_summary` (no `probe_summary`) aggregates correctly into `forced_probe_by_iter`.
7. **Forward-compat:** sidecar with `probe_summary.strong_advantage` but no `forced_probe_summary` aggregates strong_advantage correctly.

### `tests/test_strong_advantage_smoke_live.py` — opt-in live plumbing test

Marker-gated (`@pytest.mark.slow_live` or similar). Tiny fixture (2–3 candidates), `--label-mcts-sims=200`, `--label-mcts-repeats=1`. Goal: confirm the labeling code path runs end-to-end without crashing — checkpoint load, candidate replay, MCTS label call, admission filter wiring, draft output. **Not** a label-correctness test. Not required in default CI; runs on demand or on probe-related branches.

## Migration / rollout

1. Land the generator refactor + parity test (no behavior change on the forced tier).
2. Land the strong_advantage generator + admission filter (operator runs it once to produce the first `strong_advantage_probes.json`; commit to git).
3. Land the trainer telemetry change (writes `probe_summary.<tier>` alongside legacy `forced_probe_summary`).
4. Land the analyzer aggregation change (reads new tiered structure when present, falls through to legacy field otherwise).
5. After one release cycle of dual-emit: delete `forced_probe_summary` from trainer output and remove the legacy-field fallback from analyzer.

The changes are independently shippable — the parity test gates step 1, no schema break gates the rest.

## Open questions

None. All design decisions resolved during brainstorming.
