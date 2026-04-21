# Probes & Calibration Closure — Design

**Date:** 2026-04-21
**Status:** Approved for implementation planning
**Spec owner:** bill-osienski

## 1. Problem

Two major features shipped with the connectivity-retrain work (docs/superpowers/specs/2026-04-19-connectivity-retrain-design.md) are currently producing empty outputs:

1. **`forced_probe_summary` in trainer sidecars** — the trainer loads `tests/probes/twixt_probes.json` at startup and per-iter writes a `forced_probe_summary` dict into each iteration's stats sidecar. The code path is fully wired, but the probes file has never been committed — only `tests/probes/README.md` and `tests/probes/baselines/` exist. So the loader prints *"probes file not found at tests/probes/twixt_probes.json (Phase 0 not yet committed; per-iter Probe block will be skipped)"* and `forced_probe_summary` is `None` in every sidecar produced to date.

2. **`value_calibration` in analyzer `summary_*.json`** — the analyzer writes `{"status": "not_implemented", "note": "Phase 1 scaffold — full scoring loop to be added in a follow-up"}` when invoked with `--calibrate --calibrate-weights <path>`. Helpers exist in `scripts/GPU/alphazero/value_calibration.py` (`classify_position`, `compute_calibration_bins`, `aggregate_calibration`) but nothing assembles the `[{bucket, nn_value, outcome}]` samples they consume. The checkpoint-scoring loop was never implemented.

Both gaps leave the two formal gate criteria in the connectivity-retrain spec §7 unevaluable:

- Forced-probe ≥95% sign-correct on `confidence=='forced'` tier
- Value calibration by position type (winning_structure vs. phase buckets)

Additionally, these gaps block the user's post-hoc workflow: the user pulls game replays from `scripts/GPU/logs/games/` into `Replays/<range>/` and runs `scripts/twixt_replay_analyzer.py` to produce a per-chunk diagnostic summary. Today that summary silently omits the two retrain-specific signals.

## 2. Approach

Close both gaps in one implementation round, split along signal semantics:

- **Trainer-side inline probes = stable / regression-suite / curated.** Use a committed `tests/probes/twixt_probes.json`. Fixed across a run. Per-iteration scoring gives live drift detection. No training-behavior changes, only observability changes conditioned on the probes file being present.
- **Analyzer-side post-hoc probes = dynamic / snapshot / replay-derived.** Extract probes from the games in `--input` at analysis time. Score once against the checkpoint associated with the end of the replay range. End-of-chunk health check, appropriate for dynamic samples.

The two paths share probe-extraction and network-loading primitives but remain semantically distinct in output fields (`forced_probe_summary` vs. `replay_probe_scoring`) so their different properties are never conflated in reports or gate decisions.

For the committed file, human review (spec §7 two-reviewer curation) is deferred — the committed artifact is a **bootstrap rule-selected forced probe suite**, explicitly labeled as such. It is suitable for trainer-side inline telemetry and practical regression monitoring, not for the spec-formal gate evaluation. A future effort can produce the reviewed gate suite and drop it in as a replacement without code changes.

## 3. Architecture

Two signals, two cadences, shared primitives.

```
Shared primitives (new):
  probe_eval.py::extract_forced_probes_from_games()
  probe_eval.py::load_network_for_scoring()       (public wrapper for _load_network)
  value_calibration.py::score_samples_against_checkpoint()

Bootstrap suite generation (one-shot but reusable):
  scripts/build_bootstrap_probe_suite.py
    → writes tests/probes/twixt_probes.json

Live signal (trainer, per-iter):
  trainer.py reads tests/probes/twixt_probes.json (existing code path, UNCHANGED)
    → sidecar.forced_probe_summary populated each iter
  [VERIFIED by integration test in §5.2]

Snapshot signal (analyzer, once per run):
  twixt_replay_analyzer.py (modified)
    → auto-discovers checkpoint from max(game.meta.iteration)
    → extract_forced_probes_from_games(replays) via shared helper
    → scores probes once → summary.replay_probe_scoring
    → score_samples_against_checkpoint(replays, stratified) → summary.value_calibration
```

### 3.1 Key invariants

- **Trainer code: no training-behavior changes.** Only observability changes, conditioned on the probes file being present. Self-play logic, gradients, curriculum, resign/adjudication — all unchanged.
- **Analyzer user workflow: unchanged.** The command `python scripts/twixt_replay_analyzer.py --input Replays/21-30 --out Replays/21-30_Replay` continues to produce all existing outputs plus the two new sections. No required new flags.
- **Distinct top-level keys in `summary_*.json`:**
  - `forced_probe_summary` — per-iter values aggregated from sidecars (stable-suite trainer-emitted signal). Unchanged semantics; will populate once the probes file is committed.
  - `replay_probe_scoring` — end-of-chunk single-snapshot from analyzer (replay-derived signal). New.
  - `value_calibration` — end-of-chunk stratified calibration from analyzer. New (replaces current stub).
- **Graceful degradation:** all new analyzer outputs are opt-out-only. If checkpoint auto-discovery fails, the relevant sections are skipped with a warning; the rest of the analyzer output is unaffected.

## 4. Shared primitives

Three new functions, all pure / synchronous / independently testable.

### 4.1 `probe_eval.py::extract_forced_probes_from_games`

```python
def extract_forced_probes_from_games(
    games: list[dict],                                 # parsed game JSONs
    active_size: int = 24,                             # filter games to this size
    k_plies: int = 2,                                  # positions at n_moves-1 and n_moves-2
    winner_reasons: frozenset = frozenset({"win"}),    # natural wins only
    dedupe_exact: bool = True,
    dedupe_mirror: bool = True,
    max_probes: int | None = None,
) -> list[dict]:
    """Extract near-terminal forced probes from a list of parsed game JSONs.

    Returns probe dicts with schema matching tests/probes/twixt_probes.json:
      {id, category, confidence='forced', side_to_move, expected_value_sign,
       active_size, ply, move_history, source_game, source_ply}

    Category is based on the eventual winner only:
      winner == 'red'   → 'near_win_red'
      winner == 'black' → 'near_win_black'

    Side-to-move and expected_value_sign are computed from the replayed state
    independently of category.

    Filtering:
      - Only games where meta.board_size == active_size
      - Only games where meta.reason in winner_reasons (defaults: 'win' only, so
        resigns / adjudications / draws / timeouts are excluded)
      - For each qualifying game, extract positions at plies n_moves-1 and
        n_moves-2 when k_plies >= 2, or only n_moves-1 when k_plies == 1

    Deduplication:
      - dedupe_exact: drop probes whose move_history is byte-exact-equal to an
        already-kept probe's move_history
      - dedupe_mirror: drop probes whose move_history is mirror-equivalent to
        an already-kept probe's move_history. Mirror equivalence is defined by
        the three color-preserving board symmetries:
          - horizontal reflection: (r, c) → (r, N-1-c)
          - vertical reflection:   (r, c) → (N-1-r, c)
          - 180° rotation:         (r, c) → (N-1-r, N-1-c)
        Transpose is NOT included (it would swap red/black goal orientations
        and conflate near_win_red with near_win_black). Implementation:
        compute all 4 canonical forms (identity + 3 symmetries) of each probe's
        move_history, keep the lex-smallest as the dedupe key; collision on
        dedupe key → drop.

    Probe ID format (deterministic, no hash required):
      {source_game_basename}_ply{source_ply:03d}_{winner}
      Example: 'iter_0029_game_042_ply041_red'

    Sort order (always applied, for determinism across reruns):
      1. source iteration descending (most recent training regime first;
         iteration pulled from each game's meta.iteration field)
      2. source_ply descending (later-ply probe from the same game)
      3. source game basename ascending (stable tiebreaker)

    Truncation:
      If max_probes is set, truncate to max_probes after sorting.
      If max_probes is None, return the full sorted list (order still
      deterministic for reproducible downstream consumers like per-probe CSV).

    Invalid winner handling:
      Games with winner not in {'red', 'black'} (e.g., draws, missing field)
      are skipped regardless of winner_reasons filter, because no valid
      expected_value_sign can be assigned.

    Pure function — no I/O, no network load, no global state.
    """
```

### 4.2 `probe_eval.py::load_network_for_scoring`

```python
def load_network_for_scoring(weights_path: str, verbose: bool = False):
    """Public wrapper over the existing private _load_network helper.

    Renames _load_network to a stable public symbol so trainer, analyzer, and
    the bootstrap generator can import it without reaching through the
    underscore-prefix convention. Zero behavior change — just re-exports the
    existing function with its current auto-detection of 24ch vs 30ch
    checkpoints.

    Returns: (network, in_channels, hidden, n_blocks)
    """
```

### 4.3 `value_calibration.py::score_samples_against_checkpoint`

```python
def score_samples_against_checkpoint(
    replays: list[dict],                    # parsed game JSONs
    network,                                # pre-loaded via load_network_for_scoring
    samples_per_bucket: int = 200,
    max_total: int = 2000,                  # safety cap
    min_size: int = 8,                      # winning-structure threshold
) -> dict:
    """Phase-stratified calibration scoring.

    Two-pass sampling:
      1. Pre-pass: enumerate every (game, ply) position in replays, classify
         via classify_position (state-based — works identically for 24ch and
         30ch checkpoints), count per bucket.
      2. Sample pass: for each bucket in stable alphabetical order, sample
         min(samples_per_bucket, natural_count) positions without replacement;
         halt if cumulative samples reach max_total (buckets processed after
         that point are reported with sampled=0). No redistribution across
         buckets — stratification is preserved over total count.
      3. Score pass: replay each sampled game's move_history up to the target
         ply to reconstruct TwixtState, build input tensor via the network's
         evaluator (auto-adapts to the network's channel count), NN forward,
         collect {bucket, nn_value (red-perspective), outcome (red-perspective)}.
      4. Feed the assembled samples list to existing aggregate_calibration().

    Returns:
      {
        "samples_per_bucket_target": int,
        "max_total": int,
        "natural_distribution": {bucket: count_in_full_pool},
        "sampled_distribution":  {bucket: count_actually_sampled},
        "stratified": True,
        "overall_note": "stratified aggregate, not population-weighted",
        "aggregate": <result of existing aggregate_calibration()>,
      }
    """
```

## 5. Bootstrap probe suite generation

New one-shot-but-reusable script `scripts/build_bootstrap_probe_suite.py`, separate from the existing `scripts/build_probe_candidates.py` (which remains the candidate sampler targeted at curated-gate-suite production — different purpose, different filters).

### 5.1 CLI

```
--input <dir>              default: scripts/GPU/logs/games
--source-iter-range MIN MAX   required (e.g., --source-iter-range 25 30)
--out <path>               default: tests/probes/twixt_probes.json
--samples-per-bucket N     default: 12 (per winner class, before dedup)
--max-probes N             default: 30 (final cap)
```

### 5.2 Logic

1. Scan `--input` for `iter_NNNN_game_MMM.json` files with `meta.iteration` in `[MIN, MAX]`.
2. Filter: `meta.board_size == 24` AND `meta.reason == 'win'` (natural wins only — no resign, no adjudicate, no timeout, no draw).
3. For each qualifying game, call `extract_forced_probes_from_games` (the shared helper from §4.1) with `k_plies=2`, `winner_reasons={'win'}`, `dedupe_exact=True`, `dedupe_mirror=True`.
4. Balance: enforce that the ratio of majority-class to minority-class count is ≤ 2:1. Concretely, if `majority_count > 2 * minority_count`, truncate majority to `2 * minority_count` using the §4.1 sort order. If the ratio is already ≤ 2:1, no truncation in this step. This keeps all minority-class probes available while preventing extreme skew.
5. Truncate to `max_probes` using the sort order from §4.1: source iteration desc, source_ply desc, source game basename asc.
6. Serialize to `--out` in the committed-file structure below.

### 5.3 Committed file structure

```json
{
  "meta": {
    "type": "bootstrap_rule_selected",
    "not_gate_suite": true,
    "note": "Rule-selected bootstrap suite for trainer-side inline telemetry and practical regression monitoring. NOT the spec §7 review-curated gate suite — see tests/probes/README.md for the distinction.",
    "generator": "scripts/build_bootstrap_probe_suite.py",
    "generator_version": 1,
    "selection_rules": {
      "board_size": 24,
      "winner_reasons": ["win"],
      "k_plies_from_terminal": 2,
      "dedup": "exact + 4-form-mirror-canonical",
      "source_iter_range": [25, 30]
    }
  },
  "probes": [ ... 20-30 dicts per §4.1 schema ... ]
}
```

**No wall-clock fields** (`generated_at`, `timestamp`, `created_at` etc.) are emitted. Git commit history captures when the file was regenerated.

### 5.4 Reusability

The script is committed (not thrown away). Re-running it with identical `--source-iter-range` produces **byte-identical output** because:
- Probe IDs are deterministic (§4.1 ID format).
- Dedupe canonicalization is deterministic (§4.1 lex-smallest of 4 symmetries).
- Sort keys are deterministic (§4.1 sort order, all tiebreakers specified).
- No wall-clock fields in output.

Enables future refreshes (e.g., "regenerate bootstrap after another 50 iters of training") by re-running with an updated `--source-iter-range`.

### 5.5 Trainer compatibility

The trainer's existing loader at `trainer.py:1904-1917` reads:
```python
_probes_data = json.load(_pf)
_all_probes = _probes_data.get("probes") or _probes_data.get("candidates") or []
forced_probes = [p for p in _all_probes if p.get("confidence") == "forced"]
```

The `"meta"` key in the committed file is ignored by the loader. No trainer code change required.

## 6. Analyzer changes

All additions are opt-out-via-flag, enabled by default when the prerequisites (checkpoint discoverable) are met.

### 6.1 New CLI flags (all optional)

```
--checkpoint-dir <path>              # default: auto-discover (§6.2)
--weights <path>                     # explicit checkpoint override; skips auto-discovery
--probe-scoring-disable              # opt out of replay_probe_scoring
--calibration-disable                # opt out of value_calibration
--calibration-samples-per-bucket N   # default: 200
--calibration-max-total N            # default: 2000
```

Existing `--calibrate` and `--calibrate-weights` flags are kept for backwards compatibility with any existing invocations. Help text is updated to note they're superseded by `--weights` + auto-discovery. When both `--weights` and `--calibrate-weights` are passed, `--weights` takes precedence.

### 6.2 Checkpoint auto-discovery

Resolution order:
1. If `--weights <path>` given: use it, skip all auto-discovery.
2. If `--calibrate-weights <path>` given (legacy): use it as fallback.
3. Compute `max_iter = max(game["meta"]["iteration"] for game in replays)`.
4. Target filename: `model_iter_{max_iter + 1:04d}.safetensors` — off-by-one verified in existing code: sidecar `meta.iteration=29` maps to checkpoint written as `model_iter_0030.safetensors` (the trainer writes `ckpt_base = f"model_iter_{iteration+1:04d}"` when an iteration completes).
5. Search locations in order:
   1. `--checkpoint-dir <path>/` if provided
   2. `checkpoints/<single-subdir>/` if `checkpoints/` contains exactly one subdirectory
   3. Current working directory
6. First match wins.
7. No match → print one-line warning, set resolved path to `None`, skip both `replay_probe_scoring` and `value_calibration` sections, continue with rest of analyzer output.

The resolved path is established once early in `main()` and shared across both downstream consumers (probe scoring + calibration), so the network is loaded exactly once per run.

### 6.3 Replay-derived probe scoring (`replay_probe_scoring`)

Runs when `resolved_weights is not None and not args.probe_scoring_disable`.

```python
probes = extract_forced_probes_from_games(
    replays,
    active_size=24,
    k_plies=2,
    winner_reasons=frozenset({"win"}),
    dedupe_exact=True,
    dedupe_mirror=True,
    max_probes=None,   # no cap — analyzer wants all applicable probes
)

result = run_forced_probes_inline(network, probes, active_size=24)
# Existing function at probe_eval.py:201 — unchanged. Returns
# {n, n_skipped_size, sign_correct, sign_correct_pct, median_abs_v, ...}
```

Emitted under top-level key `replay_probe_scoring`:

```json
"replay_probe_scoring": {
  "source": "replay_derived",
  "weights": "<abs path>",
  "checkpoint_in_channels": 30,
  "selection_rules": {
    "k_plies": 2,
    "winner_reasons": ["win"],
    "dedup": "exact + 4-form-mirror-canonical"
  },
  "probe_count": 1973,
  "n": 1973,
  "sign_correct": 1847,
  "sign_correct_pct": 0.9361,
  "median_abs_v": 0.847,
  "by_category": {
    "near_win_red":   {"n": 982, "sign_correct_pct": 0.941, "median_abs_v": 0.861},
    "near_win_black": {"n": 991, "sign_correct_pct": 0.931, "median_abs_v": 0.833}
  }
}
```

If zero probes are extractable (e.g., all games in `--input` are draws), the block emits `{"probe_count": 0, "skipped_reason": "no_natural_wins"}` and the other scoring fields are omitted.

### 6.4 Value calibration (`value_calibration`)

Runs when `resolved_weights is not None and not args.calibration_disable`.

Replaces the current stub at `scripts/twixt_replay_analyzer.py:1422-1427`. The revised call site is:

```python
if _HAS_PHASE1_DIAG and resolved_weights is not None and not args.calibration_disable:
    value_calibration_summary = score_samples_against_checkpoint(
        replays,
        network=shared_loaded_network,
        samples_per_bucket=args.calibration_samples_per_bucket,
        max_total=args.calibration_max_total,
        min_size=args.winning_structure_min_size,
    )
```

Emits the full dict from §4.3 under top-level key `value_calibration`:

```json
"value_calibration": {
  "weights": "<abs path>",
  "samples_per_bucket_target": 200,
  "max_total": 2000,
  "natural_distribution": {
    "red_winning_structure": 47, "black_winning_structure": 52,
    "balanced_no_winning_structure": 203,
    "early_game": 8012, "mid_game": 3841, "late_game": 892
  },
  "sampled_distribution": {
    "red_winning_structure": 47, "black_winning_structure": 52,
    "balanced_no_winning_structure": 200,
    "early_game": 200, "mid_game": 200, "late_game": 200
  },
  "stratified": true,
  "overall_note": "stratified aggregate, not population-weighted",
  "aggregate": { ... per existing aggregate_calibration schema ... }
}
```

### 6.5 New CSV outputs in `--out` directory

- `replay_probe_per_probe_<suffix>.csv` — one row per probe: `id, category, source_game, source_ply, expected_value_sign, nn_value, sign_correct_nn, nn_magnitude`
- `value_calibration_by_bucket_<suffix>.csv` — one row per bucket: `bucket, natural_count, sampled_count, sign_agree, mse, pred_mean, outcome_mean`

Output naming follows the existing `<name>_<suffix>.csv` pattern (e.g., `sanity_by_connectivity_by_iter_21-30.csv`).

### 6.6 New report sections in `report_<suffix>.txt`

Replace the two current `(not available)` placeholders.

- `format_replay_probe_scoring_report(summary)` — shows weights path, probe count, overall sign-correct percentage, per-category breakdown
- Updated `format_value_calibration_report(summary)` — shows:
  - Header flagging stratified origin: *"Per-bucket calibration is phase-stratified (target N=200/bucket). Overall row is a stratified aggregate, not population-weighted."*
  - Natural vs. sampled distribution table
  - Per-bucket stats rows

### 6.7 Graceful degradation matrix

| Condition | `replay_probe_scoring` | `value_calibration` | Rest of analyzer |
|---|---|---|---|
| Checkpoint found, probes extractable | full output | full output | unchanged |
| Checkpoint found, 0 extractable probes (e.g., all draws) | `{"probe_count": 0, "skipped_reason": "no_natural_wins"}` | full output | unchanged |
| Checkpoint not found | skipped with warning | skipped with warning | unchanged |
| `--probe-scoring-disable` | omitted from summary | runs if possible | unchanged |
| `--calibration-disable` | runs if possible | omitted from summary | unchanged |
| 24-channel checkpoint (pre-retrain) | runs normally; `nn_value` reflects pre-retrain quality (expect lower `sign_correct_pct`) | runs normally; structural buckets are identical to 30ch case (classification uses `classify_position` on the replayed `TwixtState`, not on the tensor); per-bucket `nn_value` / `mse` / `sign_agree` reflect pre-retrain quality | unchanged |

The 24-channel row is intentionally useful: pointing `--weights` at the old iter-0999 checkpoint lets you produce an apples-to-apples before/after calibration comparison against the current retrain.

## 7. Verification strategy

### 7.1 Unit tests (extensions)

Add to existing `tests/test_inline_probe_observability.py`:
- `test_extract_forced_probes_from_games_category_winner_based` — asserts category is derived from game's winner, independent of side-to-move
- `test_extract_forced_probes_deterministic_ids` — asserts rerunning on identical input produces identical `id` fields
- `test_extract_forced_probes_mirror_dedup` — constructs two games whose move_history is a horizontal-mirror pair; asserts only one probe survives dedup
- `test_extract_forced_probes_natural_wins_only` — a resign/adjudicated/draw/timeout game produces zero probes

Add new `tests/test_value_calibration_sampling.py`:
- `test_score_samples_stratified_budget` — synthetic replays with known bucket distribution; assert `sampled_distribution` matches per-bucket caps and stable alphabetical ordering when `max_total` binds
- `test_score_samples_natural_distribution_reported` — asserts `natural_distribution` counts every position in the full pool, not just sampled ones
- `test_score_samples_24ch_checkpoint` — smoke test against a 24-channel tiny-network weights fixture; asserts no crashes and bucket stats populate with real values (not `'unknown'`)

Add new `tests/test_bootstrap_probe_suite.py`:
- `test_build_bootstrap_deterministic_rerun` — run the generator twice against identical input; assert byte-identical output
- `test_build_bootstrap_schema` — validate output conforms to `tests/probes/README.md` schema
- `test_build_bootstrap_only_natural_wins` — construct synthetic input with mixed termination reasons; assert no resign/adjudicated/draw/timeout probes survive
- `test_build_bootstrap_no_wall_clock_fields` — belt-and-suspenders negative assertion: no `generated_at`, `timestamp`, `created_at`, or similar wall-clock keys in `meta`. Prevents regression if a future developer reintroduces such a field.

### 7.2 Required integration test

New `tests/test_trainer_forced_probe_live.py`, marked `@pytest.mark.integration`:

```python
@pytest.mark.integration
def test_trainer_writes_forced_probe_summary_to_sidecar(tmp_path):
    """Runs 2 training iterations with minimal config against the committed
    bootstrap probes file; asserts forced_probe_summary lands in the sidecar
    with n > 0 and rolling-window math populates correctly on iter 2."""
    # Uses:
    #   - existing latest checkpoint as --resume source
    #   - committed tests/probes/twixt_probes.json
    #   - tmp_path for checkpoint + games output
    #   - games_per_iter=2, n_workers=1, iterations=<target+2>, simulations=20
    # Runs ~30-90s on the test machine.
    # Asserts:
    #   - sidecar['forced_probe_summary']['n'] > 0 on both iters
    #   - sidecar['forced_probe_summary']['rolling5_sign_correct_pct'] is None on iter 1, float on iter 2
    #   - sidecar['forced_probe_summary']['delta_sign_correct_pct'] is None on iter 1, float on iter 2
    #   - sidecar['sanity_by_connectivity'] is dict with winning/no_winning keys
```

**CI-vs-local policy (explicit):**
- **Locally:** opt-in only. Default `pytest` skips tests marked `@pytest.mark.integration`. Developers run `pytest -m integration` (or repo's equivalent) when they want to exercise it.
- **CI:** enabled as a required merge gate. CI configuration explicitly opts into the `integration` marker so the test must pass before merge.

If the repo doesn't already have a marker-opt-in mechanism in `pytest.ini` / `conftest.py`, the implementation adds a minimal one consistent with existing pytest patterns in the project.

### 7.3 Required end-to-end analyzer test

New `tests/test_analyzer_replay_probe_scoring_end_to_end.py`:

```python
def test_analyzer_against_committed_replays(tmp_path):
    """Invokes the analyzer main() on a small subset of committed replay JSONs
    (2-3 iterations × 5 games each) with the committed bootstrap probes.
    Asserts summary.json contains real values for replay_probe_scoring and
    value_calibration (not stubs), and new CSVs/report sections render."""
    # Uses a handful of committed sample game JSONs copied into tmp_path,
    # plus the committed tests/probes/twixt_probes.json, plus an existing
    # checkpoint. Asserts:
    #   - summary['replay_probe_scoring']['n'] > 0
    #   - summary['value_calibration']['stratified'] is True
    #   - summary['value_calibration']['aggregate']['overall']['n'] > 0
    #   - summary['value_calibration']['natural_distribution'] contains all 6 expected buckets
    #   - report text contains 'Value Head Calibration' section with populated stats
    #   - replay_probe_per_probe_*.csv exists with non-empty rows
    #   - value_calibration_by_bucket_*.csv exists with non-empty rows
```

Runs under default `pytest` (not integration-marked). Exercises the full analyzer pipeline from parsing input game JSONs to emitting CSVs, report text, and summary JSON.

### 7.4 Post-merge operational smoke (required operational check, not a merge gate)

Documented in the PR description; not a pytest:

1. Pull `main` after merge.
2. Run one real iteration: resume from the most-recent non-`_partial` checkpoint and set `--iterations <current_iter + 1>`. The trainer uses **absolute-target** semantics — resuming from iter `N` with `--iterations N+1` produces exactly one additional iteration. Example: if `model_iter_0030.safetensors` is the latest, invoke `--resume checkpoints/alphazero-v2-staged/model_iter_0030.safetensors --iterations 31 ...` with your standard full config.
3. Inspect the new `iter_0030_stats.json` (or whichever absolute index fires). Confirm `forced_probe_summary.n` > 0 and matches the bootstrap probe count filtered to `active_size == 24`.
4. Run the analyzer against the resulting `scripts/GPU/logs/games/` directory. Confirm `summary.json` contains real `replay_probe_scoring` and `value_calibration` data (not empty dicts or stubs).

### 7.5 CI gate

§§7.1, 7.2, 7.3 all run under pytest. Failing tests block merge. §7.2 (integration) is CI-opt-in per §7.2 policy. §7.4 is operational confidence and is documented-but-not-enforced.

## 8. Rollout

### 8.1 Backwards compatibility

- **Trainer:** zero behavior change for users who don't have the probes file yet. The existing `"probes file not found ... per-iter Probe block will be skipped"` path remains functional. The day the file lands, inline probes fire automatically with no other user action.
- **Analyzer legacy flags:** `--calibrate` and `--calibrate-weights` continue to work. Help text gains a line noting they're superseded by `--weights` + auto-discovery. Existing CI or scripts that pass these flags still produce identical-or-better output (identical when the user also didn't want probe scoring; better when they did).
- **Existing per-iter sidecars in `scripts/GPU/logs/games/`:** no schema change. New analyzer output keys are purely additive in `summary_*.json`.
- **No migration script required.** All changes are additive.

### 8.2 Documentation updates

- `tests/probes/README.md` — add a section explicitly distinguishing the committed bootstrap suite (rule-selected, not gate-suite) from the eventual spec §7 curated gate suite. Explicit warning: a 94% pass rate on the bootstrap suite is **not** equivalent to passing the formal spec §7 ≥95% gate on a curated suite. The two use different probe-selection methodologies and measure related-but-distinct quantities.
- `scripts/twixt_replay_analyzer.py` module docstring — document the new flags and auto-discovery behavior.
- `scripts/build_bootstrap_probe_suite.py` — module docstring explaining purpose, selection rules, how to refresh with a newer `--source-iter-range`.
- This spec: `docs/superpowers/specs/2026-04-21-probes-and-calibration-closure-design.md`.
- `docs/superpowers/specs/2026-04-19-connectivity-retrain-design.md` — no changes; remains historical.

### 8.3 Commit strategy

Four logical commits, each independently reviewable and revertable:

1. **Shared primitives.** `scripts/GPU/alphazero/probe_eval.py` (add `extract_forced_probes_from_games`, `load_network_for_scoring`) + `scripts/GPU/alphazero/value_calibration.py` (add `score_samples_against_checkpoint`) + unit tests for both (§7.1 additions to `test_inline_probe_observability.py` and new `test_value_calibration_sampling.py`). Touches no existing callers — safe to merge independently.

2. **Bootstrap generator + committed suite.** `scripts/build_bootstrap_probe_suite.py` + generated `tests/probes/twixt_probes.json` + new `tests/test_bootstrap_probe_suite.py` + `tests/probes/README.md` update. The trainer's inline probe path lights up as a side effect of this commit — no trainer code change needed. The committed JSON excludes wall-clock fields; the deterministic-rerun test and the negative wall-clock-fields test (both in §7.1) assert byte-identity invariants.

3. **Analyzer wiring.** `scripts/twixt_replay_analyzer.py` modifications (new flags, checkpoint auto-discovery, replay probe scoring integration, calibration integration replacing the stub, CSV writers, report formatters) + new `tests/test_analyzer_replay_probe_scoring_end_to_end.py`.

4. **Integration test.** `tests/test_trainer_forced_probe_live.py` marked `@pytest.mark.integration`. Per the CI-vs-local policy in §7.2: the integration test is opt-in locally but **enabled as a required CI merge gate** via CI configuration.

Each commit passes `pytest` (integration test in commit 4 only runs under its opt-in marker; all other commits' tests are default-included).

**Commit ordering is strict.** Commits #2, #3, and #4 all depend on symbols introduced in #1 (`extract_forced_probes_from_games`, `load_network_for_scoring`, `score_samples_against_checkpoint`). Commit #4 depends on the committed file from #2. Commit order must be preserved (#1 → #2 → #3 → #4) for each commit to pass CI standalone. Commits may be co-reviewed but must land in order.

### 8.4 Rollback plan

- Each commit reverts independently. Commits are ordered so later ones depend on earlier ones, but reverting only commit #4 (integration test) or only commit #3 (analyzer) cleanly reverts that functionality while leaving the rest intact.
- Commit #2 (probes file) can be reverted by deleting the JSON. Trainer inline probes silently skip again, no other breakage.

### 8.5 Completion criteria ("done" means)

- All four commits merged to `main`.
- CI green, including the integration test (CI opt-in policy per §7.2).
- Post-merge operational smoke (§7.4) completed successfully. Manual walkthrough documented in the PR.
- An analyzer run against the current replay set (`Replays/21-30/` or later) produces a `summary_*.json` containing populated `replay_probe_scoring` and `value_calibration` objects — not empty dicts or `{"status": "not_implemented"}` stubs.

## 9. What remains out of scope

- **Spec §7 formal curated gate suite.** The committed `tests/probes/twixt_probes.json` is a rule-selected bootstrap suite. Achieving the spec §7 gate criteria (≥95% sign-correct on a curated, human-reviewed `confidence='forced'` tier) requires the deferred two-reviewer curation effort. That is a separate future track; nothing in this design precludes it, and the bootstrap file can be replaced in place.
- **Trainer-side auto-generated probes.** Explicitly rejected during design discussion: regression suites need stability across iterations and runs. Per-iter auto-generation would destroy that property and silently break cross-iter comparability. The trainer stays on the stable committed file.
- **Phase-stratified calibration as the only mode.** A future follow-up could add a natural-distribution (unstratified) calibration mode as a secondary metric for when readers need population-weighted aggregates. This design emits the natural-distribution counts alongside the stratified stats, so adding unstratified later is a small additive change.
- **Changes to the per-game `iter_NNNN_game_NNN.json` sidecars.** Those are owned by the self-play worker. All new data emitted by this design lives in the analyzer output directory or in the per-iteration stats sidecar (the latter unchanged in schema; the `forced_probe_summary` field merely starts populating with real values).
