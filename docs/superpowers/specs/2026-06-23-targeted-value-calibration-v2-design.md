# Targeted Value Calibration v2 — Design

- **Date:** 2026-06-23
- **Status:** Approved (ready for implementation plan)
- **Owner:** bill-osienski
- **Supersedes / extends:** [Post-opening sharp-drop calibration (v1)](2026-06-16-post-opening-sharp-drop-calibration-design.md) (Mechanism B)
- **Base checkpoint:** `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors` ("calib020_0001", the current promoted best)

---

## 1. Context & motivation

v1 added a value-only auxiliary loss (Mechanism B): a fixed pool of external replay
positions, each pulled toward a single global soft target. A single-target
black-pre-drop calibration branch
(`checkpoints/alphazero-v2-black-predrop-calib010-from-calib020-0001`) **improved the
target probe but damaged match strength badly** — fixing the overvalue on one family of
positions silently regressed the model's behavior on other, fragile guardrail families.

The lesson: **guardrails must be part of the training objective, not just post-run
checks.** v2 makes the calibration manifest *mixed* — every row can carry its own
`target_black_value`, `weight_scale`, and `tag`. Correction rows pull the known black
pre-drop overvalue down; **retention rows pin the fragile guardrails to calib020_0001's
own current predictions** (self-distillation), so the optimizer is explicitly told *not
to move them* while it fixes the target.

The v2 question is therefore clean:

> Can we improve black pre-drop value calibration **while explicitly preventing
> movement** on the known-fragile guardrails (goal-line, old broad post-opening, red
> pre-drop)?

## 2. Goals / Non-goals

**Goals**

1. Mixed manifest with per-row `target_black_value`, `weight_scale`, `tag`,
   backward-compatible with v1 manifests (no new columns ⇒ identical behavior).
2. Per-row **target** support (rides on `PositionRecord.outcome`; no loss-path change).
3. Per-row **weight** support — the one genuinely new loss-path plumbing — as a relative
   per-sample weight in the calibration MSE.
4. A deterministic builder that assembles the mixed manifest from the verified correction
   + retention sources, anchoring retention targets to calib020_0001.
5. Sidecar telemetry that makes the run's mode and tag composition obvious.

**Non-goals (v2)**

- Disjoint train/eval retention sets (retention is `train == eval` this pass; §10).
- Per-tag *value-mean* sidecar metrics (only counts/mass this pass; §4.3).
- 0379 distillation (retention anchors to calib020_0001, not 0379).
- Tag-stratified sampling (uniform-with-replacement + relative weights this pass).

## 3. High-level design & data flow

```
manifest CSV ──load_csv_manifest──▶ CalibrationPool[ CalibrationSample ]
  (per-row target/weight/tag)            │  CalibrationSample(record, weight_scale, tag, target_black_value)
                                         │  • per-row target already baked into record.outcome
                       pool.sample(k) ───┤    via target_in_to_move(side, target_black)
                                         ▼
   trainer boundary (split — no polymorphism downstream):
        records  = [s.record       for s in samples]
        weights  = np.asarray([s.weight_scale for s in samples], np.float32)
                                         ▼
   train_step(calibration_positions=records, calibration_weights=weights, ...)
                                         ▼
   alphazero_loss_batch:
        per_sample = (cb_values - cb_targets) ** 2
        calib_loss = sum(w * per_sample) / max(sum(w), 1e-8)   if weights is not None
                   = mean(per_sample)                          otherwise
        total_loss += calibration_loss_weight * calib_loss      # standalone weight
```

**Key principle (your Q3 decision):** `CalibrationSample` bundles domain metadata *inside
the pool*, but the hot path (`train_step`, `alphazero_loss_batch`) stays **homogeneous and
typed** — `calibration_positions: List[PositionRecord]` + `calibration_weights:
Optional[np.ndarray]`. No `isinstance` sniffing in the loss. The pool→records/weights
split happens at the trainer boundary.

## 4. Component changes

### 4.1 `scripts/GPU/alphazero/calibration_pool.py`

Add the wrapper and per-row parsing; keep the existing public API.

```python
@dataclass(frozen=True)
class CalibrationSample:
    record: PositionRecord
    weight_scale: float = 1.0
    tag: str = ""
    target_black_value: float | None = None   # metadata-only (loss reads record.outcome)
```

- **Per-row target** — `build_calibration_position(case, calibration_target)` resolves the
  target with a fallback rule, then keeps the existing sign-safe conversion:
  ```python
  raw = case.get("target_black_value")
  target_black = float(raw) if raw not in (None, "") else calibration_target
  outcome = target_in_to_move(state.to_move, target_black)   # unchanged sign handling
  ```
- **Validation (loud, at parse time):**
  - `target_black_value`: must be finite and in `[-1.0, +1.0]` ⇒ else `ValueError`.
  - `weight_scale`: optional, default `1.0`; must be finite and `>= 0` ⇒ else `ValueError`.
  - `tag`: optional, default `""`.
- `CalibrationPool` stores `List[CalibrationSample]`; `sample(k, rng)` returns
  `List[CalibrationSample]`. `from_manifest(manifest_path, calibration_target)` keeps its
  signature and now builds samples with per-row weight/tag/target.
- Pool exposes introspection for the sidecar: `schema` (`per_row_target` if **any** row
  has a non-empty `target_black_value`, else `global_target`), `has_weight_scale` (any row
  had an explicit `weight_scale`), and `tag_counts()` → `{tag: count}`.

`load_csv_manifest` (in `position_probe_cases.py`) needs **no change**: it already does
`case = dict(r)`, so `target_black_value`/`weight_scale`/`tag` survive as string keys when
present.

### 4.2 `scripts/GPU/alphazero/trainer.py`

- **Train loop** — after `_calib_pool.sample(...)`, split:
  ```python
  _calib_samples = _calib_pool.sample(_k, train_rng)
  _calib_batch   = [s.record for s in _calib_samples]
  # Pass None when the manifest specified no explicit weights → loss uses mx.mean,
  # byte-identical to v1. Only materialize a weights array when weighting is in play.
  _calib_weights = (np.asarray([s.weight_scale for s in _calib_samples], dtype=np.float32)
                    if _calib_pool.has_weight_scale else None)
  train_step(..., calibration_positions=_calib_batch,
                  calibration_weights=_calib_weights,
                  calibration_loss_weight=effective_post_opening_calibration_weight)
  ```
- **`train_step`** — add a 1-line passthrough arg `calibration_weights=None` forwarded to
  `alphazero_loss_batch`. No optimizer/gradient changes.
- **`alphazero_loss_batch`** — add `calibration_weights=None`; replace the unweighted mean:
  ```python
  per_sample = (cb_values - cb_targets) ** 2
  if calibration_weights is not None:
      w = mx.array(calibration_weights)
      calib_loss = mx.sum(w * per_sample) / mx.maximum(mx.sum(w), 1e-8)
  else:
      calib_loss = mx.mean(per_sample)
  ```
  The return tuple is unchanged (10-tuple when calibration active: `…, calib_loss,
  calib_value_mean, calib_n`). `calibration_loss_weight` remains **standalone** (NOT
  multiplied by `value_weight`).

**Weight semantics (must be documented):** because `calib_loss = Σ(w·loss)/Σ(w)`,
`weight_scale` is **relative within the calibration batch, not absolute** — scaling all
weights by a constant is a no-op. Absolute calibration force is
`--post-opening-calibration-weight`. So correction `1.0` vs retention `0.5` means
correction rows carry **2× the per-sample influence** of retention rows. A tag's *total*
gradient share ≈ (its share of the pool) × (its weight_scale). For the v2 pool
(50 correction + 78 retention) that is **≈56% correction / 44% retention** of the
calibration gradient — correction-leaning, as intended.

### 4.3 Sidecar telemetry

Keep the v1 fields (`calib_loss_avg_iter`, `calib_mean_value_pred`,
`calib_n_drawn_total`, `calib_n_drawn_per_step`). Add to the config block:

```json
{ "schema": "per_row_target", "has_weight_scale": true,
  "tags": { "black_predrop_correction": 50, "red_predrop_retention": 30,
            "goal_line_retention": 18, "old_post_opening_retention": 30 } }
```

**Caveat (document inline):** in v2 `calib_mean_value_pred` is a **blended** number
(correction pulling toward −0.35, retention holding near anchors) and is **not** the clean
"drift to target" signal it was in v1. The authoritative correction signal is the
post-training **Gate A** probe. Per-tag *value-means* are deferred to v2.1 if the blended
metric proves too coarse.

### 4.4 CLI (`train.py`) — unchanged flags

No new required flags. `--post-opening-calibration-target -0.35` becomes a *fallback*
(every v2 row overrides it). Add a mode-aware startup line:

```
Post-opening calibration: 128 positions, mode=per_row_target, weight=0.01, batch_fraction=0.10
# old manifests:
Post-opening calibration: 50 positions, mode=global_target, target=-0.5, weight=0.01, batch_fraction=0.10
```

## 5. Mixed manifest builder — `scripts/GPU/alphazero/build_targeted_calibration_manifest.py`

A new deterministic builder. Unified output schema — `case_rank` is a global `1..N` rank
across the mixed manifest (kept for the existing CSV/inspection tools that expect a
rank-like column); the loader requires `game_idx, case_id, replay_path, position_ply,
side_to_move`; the pool reads `target_black_value, weight_scale, tag`; the rest are
best-effort metadata, blank where a source lacks them:

```
case_rank, tag, source, source_rank, target_black_value, weight_scale,
game_idx, case_id, replay_path, position_ply, side_to_move,
anchor_checkpoint, drop_ply, largest_drop_phase, collapse_type
```

### 5.1 Per-source adapters

| Family (Gate) | tag | weight | target | Source | Adapter notes |
|---|---|---|---|---|---|
| **Correction (A)** | `black_predrop_correction` | `1.0` | **−0.35** (hard) | `…/loss_analysis_v2_calib020_0001_vs_0379_black/0001_black_post_opening_predrop_train_manifest.csv` (50 rows) | holdout-verify vs the frozen-30 eval (§5.3) |
| **Retention D (red pre-drop)** | `red_predrop_retention` | `0.5` | `probe_black_root_value` @ anchor | `…/calib020_0001_red_loss_post_opening_predrop_probe/position_probe_cases.csv` | self-sufficient (has `replay_path`); anchor label `0001` |
| **Retention C (old broad post-opening)** | `old_post_opening_retention` | `0.5` | `probe_black_root_value` @ anchor | `…/black_predrop_calib010_checkpoint_sweep_old_post_opening/position_probe_cases.csv` | self-sufficient; anchor label **`alphazero-v2-calib020-from0409:0001`** (exact — two `:0001` labels exist) |
| **Retention B (goal-line)** | `goal_line_retention` | `0.5` | `probe_black_root_value` @ anchor | values: `…/calib020_goal_line_sweep/goal_line_trigger_probe_cases.csv` (anchor `0001`) **JOIN** `replay_path`: `…/loss_analysis_v2_1/goal_line_trigger_probe_candidates.csv` | goal-line cases CSV has **no `replay_path`** → must join (§5.4) |

> **Gate C correction (your change #1):** the "old broad post-opening" set is the
> **original 0409 broad post-opening frozen-30** (manifest
> `…/loss_analysis_v2_lr0003_0409/post_opening_sharp_drop_probe_manifest.csv`, 30 rows),
> probed with calib020_0001 included. It is **NOT** the black-loss drop-ply probe
> (`calib020_0001_black_loss_post_opening_probe`) — that is a different family. The chosen
> source CSV's calib020_0001 row reproduces the Gate C baseline exactly (mean +0.099,
> over 33.3%, severe 13.3%, n=30), confirming the family.

CLI flags (each retention source takes a path + an anchor label defaulting to `0001`):

```
--correction-manifest PATH                 --correction-holdout-manifest PATH
--red-predrop-cases PATH                    [--red-predrop-anchor-label 0001]
--old-post-opening-cases PATH              --old-post-opening-anchor-label alphazero-v2-calib020-from0409:0001
--goal-line-cases PATH --goal-line-candidates PATH   [--goal-line-anchor-label 0001]
--correction-target -0.35  --retention-weight 0.5  --out PATH
```

### 5.2 Strict, ambiguity-safe anchor resolution (your change #3)

For each retention source, resolve the anchor rows by checkpoint label:

```
resolve(rows, anchor_label):
    exact  = rows where checkpoint == anchor_label
    if distinct_checkpoints(exact) == 1:  return exact
    suffix = rows where checkpoint.endswith(":" + anchor_label)
    d = distinct_checkpoints(suffix)
    if d == 1:  return suffix
    if d  > 1:  ERROR  ambiguous: list candidate labels; require exact --…-anchor-label
    ERROR  no checkpoint matches anchor_label in <source>
```

This is mandatory for Gate C: the source has both
`alphazero-v2-calib020-from0409:0001` (wanted) and
`alphazero-v2-black-predrop-calib010-from-calib020-0001:0001` (the failed branch). Bare
`0001` is absent and `:0001` is ambiguous → the exact label is required. After resolution,
require **exactly one anchor row per `case_id`**.

### 5.3 Correction holdout verification (your change #2)

The builder takes both the train manifest and the frozen eval manifest explicitly and
**asserts no leak**:

```
train   = load(--correction-manifest)            # 0001_black_post_opening_predrop_train_manifest.csv (50)
holdout = load(--correction-holdout-manifest)    # 0001_black_post_opening_top30_predrop_probe_manifest.csv (30)
overlap = {(replay_path, position_ply) of train} ∩ {(replay_path, position_ply) of holdout}
if overlap:  ERROR  "correction train leaks N frozen-eval positions: <list>"
```

### 5.4 Goal-line `replay_path` join (your change #4) — no best-effort

The goal-line cases CSV carries the probe value but not `replay_path`; the candidates CSV
carries `replay_path` (col 23), `game_idx`, and `prev_black_ply` (→ `position_ply`).

```
key  = (game_idx, position_ply)            # case_id "game_000769_ply_39" is a cross-check
join goal_line_cases[anchor]  ⋈  candidates  on key
for each joined row, REQUIRE:
    exactly one candidate match for the key      (else ERROR)
    Path(replay_path).exists()                   (else ERROR)
    candidate position_ply == case position_ply  (else ERROR)
    candidate side_to_move == case side_to_move  (== "black")  (else ERROR)
```

Any mismatch fails loudly. No silent drops.

### 5.5 Builder sanity output (your change #6)

After writing the manifest, print per-tag diagnostics so a wrong anchor direction is
caught immediately:

```
black_predrop_correction:   n=50, weight_mass=50.0, target mean=-0.350 min=-0.350 max=-0.350
red_predrop_retention:      n=30, weight_mass=15.0, target mean=-0.188 min=... max=...
goal_line_retention:        n=18, weight_mass= 9.0, target mean=-0.244 min=... max=...
old_post_opening_retention: n=30, weight_mass=15.0, target mean=+0.099 min=... max=...
```

(`weight_mass = Σ weight_scale` per tag.) Determinism: re-running the builder on the same
inputs yields byte-identical output.

## 6. Verified source inventory

All anchor sources already exist on disk (no new probe runs needed). Cross-checks against
the cited gate baselines (computed from the `0001`/exact-anchor rows):

| Family | Baseline (your numbers) | Verified @ anchor | Source has `replay_path` |
|---|---|---|---|
| Retention D (red pre-drop) | over 13.3%, severe 0.0%, mean −0.188 | mean=−0.188, n=30, over=13.3%, severe=0.0% ✓ | yes |
| Retention C (old broad PO) | over 33.3%, severe 13.3%, mean +0.099 | mean=+0.099, n=30, over=33.3%, severe=13.3% ✓ | yes (col 11) |
| Retention B (goal-line) | over 5.6%, severe 0.0% | mean=−0.244, n=18, over=5.6%, severe=0.0% ✓ | via join |
| Correction train / holdout | — | 50 train rows / 30 frozen rows ✓ | yes |

## 7. Backward compatibility

- v1 manifests (no `target_black_value`/`weight_scale`/`tag`): `schema=global_target`,
  `has_weight_scale=False` ⇒ the trainer passes `calibration_weights=None` ⇒ the loss uses
  `mx.mean` exactly as v1 ⇒ **byte-identical** training behavior.
- Existing `test_calibration_loss.py` passes raw `List[PositionRecord]` with
  `calibration_weights=None` ⇒ plain mean ⇒ unchanged 7-/10-tuple arity.

## 8. Testing

- **`tests/test_calibration_pool.py`** (extend): per-row target overrides global; red
  side-to-move sign flip into `outcome`; `weight_scale` missing → 1.0 / parsed value
  carried on `CalibrationSample`; invalid target (out of `[-1,1]`/non-finite) and invalid
  weight (negative/non-finite) raise `ValueError`; `from_manifest` returns
  `CalibrationSample`s; old manifest (no v2 columns) uses global target exactly as before;
  **mixed-weights manifest** (some rows specify `weight_scale`, others omit) → omitted rows
  default to `1.0`, `pool.has_weight_scale` is `True`, and the trainer materializes the full
  weights array (vs the all-omitted case → `has_weight_scale=False` → `calibration_weights=None`).
- **`tests/test_calibration_loss.py`** (extend): **weighted-loss identity** — two
  calibration samples (w=1.0, w=3.0), assert
  `calib_loss == Σ(wᵢ·mseᵢ)/Σ(wᵢ)`; keep disabled→7-tuple, zero-weight→7-tuple,
  `weights=None`→plain-mean, enabled→10-tuple, gradient-reaches-value-head. The existing
  identity test keeps its `total = policy + value_weight·value + l2 + calib_w·calib_loss`
  form (calib_loss now the weighted mean when weights are present).
- **`tests/test_build_targeted_calibration_manifest.py`** (new): correction rows get
  −0.35 + tag `black_predrop_correction`; retention rows get `target_black_value` from the
  anchor's `probe_black_root_value`; strict anchor resolution (exact vs ambiguous `:0001`);
  goal-line join recovers + validates `replay_path`/ply/side; correction holdout overlap
  raises; output carries the three v2 columns; determinism (re-run byte-identical).

## 9. First experiment

- **Base:** `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`
- **Manifest:** `logs/eval/targeted_calibration_v2_from_calib020_0001.csv`
- **Iterations: 3.** Do **not** run 5 or 15 for the first v2 pass — the failed branch
  proved a single iteration can move things; start short and measure (your change #8).
- calib weight `0.01`, batch-fraction `0.10`, correction target `−0.35`, retention weight
  `0.5`.

```bash
.venv/bin/python -m scripts.GPU.alphazero.train \
  --load-weights checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --checkpoint-dir checkpoints/alphazero-v2-targeted-calib-v2-from-calib020-0001 \
  --iterations 3 --lr 0.0003 --curriculum-sizes 24 \
  --games-per-iter 100 --simulations 400 --max-moves 280 --batch-size 64 \
  --mcts-eval-batch-size 14 --mcts-pending-virtual-visits 8 --mcts-stall-flush-sims 48 \
  --n-workers 10 \
  --opening-noise-ply 10 --opening-dirichlet-alpha 0.7 --opening-dirichlet-eps 0.35 \
  --resign-enabled --resign-min-ply 80 --resign-threshold -0.945 --resign-window 12 \
  --resign-k 4 --resign-min-visits 200 \
  --adjudicate-enabled --adjudicate-min-ply 240 --max-positions-per-game 280 \
  --post-opening-calibration-enabled \
  --post-opening-calibration-manifest logs/eval/targeted_calibration_v2_from_calib020_0001.csv \
  --post-opening-calibration-weight 0.01 \
  --post-opening-calibration-target -0.35 \
  --post-opening-calibration-batch-fraction 0.10
```

## 10. Gates & promotion

Run all four probes on the candidate; **promote only if all four pass.**

- **Gate A — black pre-drop target** (frozen-30 eval, held out from correction train).
  Baseline: over 50.0%, severe 43.3%, mean +0.257. **Pass:** mean ≤ 0.0 **and** severe
  materially below 43.3%.
- **Gate B — goal-line.** Baseline: over 5.6%, severe 0.0%. **Pass:** severe 0.0% **and**
  over ≤ 11.1%.
- **Gate C — old broad post-opening.** Baseline: over 33.3%, severe 13.3%, mean +0.099.
  **Pass:** `severe ≤ 13.3%` **and** `over ≤ 33.3%` **and** `mean_black_value ≤ +0.099`
  (no worse than baseline on all three; ideally better).
- **Gate D — red pre-drop guardrail.** Baseline: over 13.3%, severe 0.0%, mean −0.188.
  **Pass:** `severe = 0.0%` **and** `mean_black_value ≤ 0.0` (do not move red pre-drop back
  into positive black-value territory).

**Promotion match (your change #7):** if all four gates pass, run the promotion match
**vs current best `calib020_0001`** (`…/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`)
— *not* 0379. 0379 is no longer the relevant challenger; promotion is against the current
best. A vs-0379 run is an optional sanity comparison only **after** beating current best.

Retention is `train == eval` this pass — Gates B/C/D therefore test "did the (relative
0.5-weight) retention anchoring counteract the correction's collateral damage on these
exact positions." Broad collateral damage on **unpinned** guardrails is caught by the
promotion match. If v2 passes the exact guardrails but still loses the match, the next
version adds disjoint/broader retention families.

## 11. Risks & mitigations

- **Wrong anchor direction / wrong checkpoint** → builder sanity output (§5.5) prints
  per-tag target means; cross-checked against §6 baselines.
- **Ambiguous `:0001` label (Gate C)** → strict resolver errors and requires the exact
  label (§5.2).
- **Goal-line join mismatch** → loud-fail join with replay_path/ply/side validation
  (§5.4).
- **Frozen-eval leak into correction** → explicit overlap assertion (§5.3).
- **Blended `calib_mean_value_pred` misread** → documented caveat; Gate A probe is
  authoritative (§4.3).

## 12. Out of scope / deferred (v2.1+)

Disjoint/broader retention families; per-tag value-mean sidecar metrics; 0379
distillation; tag-stratified calibration sampling.

## 13. Prerequisites / inputs (resolved) — exact source paths

All inputs verified present on disk (§6). Copy/paste-exact paths for the builder, its test
fixtures, and the first-run command (the only non-default anchor label is Gate C's
`alphazero-v2-calib020-from0409:0001`):

```
# correction
--correction-manifest          logs/eval/loss_analysis_v2_calib020_0001_vs_0379_black/0001_black_post_opening_predrop_train_manifest.csv
--correction-holdout-manifest  logs/eval/loss_analysis_v2_calib020_0001_vs_0379_black/0001_black_post_opening_top30_predrop_probe_manifest.csv
# retention D (red pre-drop)        anchor label: 0001
--red-predrop-cases            logs/eval/calib020_0001_red_loss_post_opening_predrop_probe/position_probe_cases.csv
# retention C (old broad post-opening)  anchor label: alphazero-v2-calib020-from0409:0001  (exact; two ':0001' labels exist)
--old-post-opening-cases       logs/eval/black_predrop_calib010_checkpoint_sweep_old_post_opening/position_probe_cases.csv
# retention B (goal-line)           anchor label: 0001  (join cases ⋈ candidates for replay_path)
--goal-line-cases              logs/eval/calib020_goal_line_sweep/goal_line_trigger_probe_cases.csv
--goal-line-candidates         logs/eval/loss_analysis_v2_1/goal_line_trigger_probe_candidates.csv
--out                          logs/eval/targeted_calibration_v2_from_calib020_0001.csv

# reference only (Gate C canonical eval manifest / case universe, 30 rows):
#   logs/eval/loss_analysis_v2_lr0003_0409/post_opening_sharp_drop_probe_manifest.csv
```
