# Post-Game Analysis — Checkpoint Eval & Loss Analyzers

Operator guide for the scripts that run **after** training has produced
checkpoints: play checkpoints against each other, then analyze *who* wins, *how*
they lose, and *why* — and (§6) turn those findings into the value-calibration
manifest that feeds the **next** training run. All scripts live in
`scripts/GPU/alphazero/` and run as `python -m scripts.GPU.alphazero.<name>` from
the repo root (use `.venv/bin/python`).

This is a how/when-to-run guide. For the *design* of each tool see the specs and
plans under `docs/superpowers/`; for *metric definitions* see
[`analysis-metrics-guide.md`](analysis-metrics-guide.md).

---

## When to run what (decision guide)

| Your question | Run | Needs replay capture? |
|---|---|---|
| Did checkpoint **A** beat **B**, and by how much (score / Elo)? | `eval_checkpoint_match` | no |
| Which checkpoint is strongest across **many** pairings? | `eval_checkpoint_tournament` | no |
| **How** does A lose — as which color, short vs long games, worst losses? | `eval_loss_analyzer` (V1) | no — reads existing `*_games.jsonl` |
| **Why** does A lose — value collapse vs search diffusion vs low-confidence, and **when** in the game? | `eval_loss_replay_analyzer` (V2) | **yes** — the match must have run with `--save-eval-replays` |
| **Is a checkpoint's value head blind to red's goal-line trigger?** (fast targeted screen across checkpoints) | `eval_goal_line_trigger_probe` | n/a — re-evaluates a fixed manifest of captured positions |
| Found value-head errors — **how do I turn them into a calibration training signal** that corrects them *without* moving the fragile guardrails? | `build_targeted_calibration_manifest` (then `train.py --post-opening-calibration-*`) | n/a — consumes the probe / loss-analysis CSVs |
| Is a checkpoint **over-valuing a fixed set of positions** (broad post-opening / pre-drop families — the A/C/D gate screens)? | `eval_position_probe` (§9) | n/a — re-evaluates a fixed CSV manifest of positions |
| Did a candidate's **raw value head drift from its teacher** on specific rows, or does the drift only appear at the **MCTS root**? | `eval_raw_nn_position_rows` (§10) | n/a — raw NN forward only, no search |

**The one thing to decide up front:** per-ply replay data is captured *only if
you ask for it at match time* (`--save-eval-replays`), and it cannot be
reconstructed after the fact. If there's any chance you'll want the "why"
analysis, run the match with capture on — it's ~free at capture time (it only
records search outputs the eval loop already computes).

```
                                   ┌── eval_loss_analyzer (V1) ──► loss SHAPE  (always cheap)
match / tournament  ──►  *_games.jsonl
   (+ --save-eval-replays)  └─ + _replays/  ──► eval_loss_replay_analyzer (V2) ──► loss CAUSE
```

---

## 1. `eval_checkpoint_match` — play one A-vs-B match

**Purpose:** Play `--games` games between two checkpoints with colors balanced by
game index, and write a strength summary (`a_score_rate`, Elo, color balance,
termination breakdown) plus a per-game `*_games.jsonl`.

**When:** You want a head-to-head strength number for a specific pair, or you
want to *produce the data* the analyzers consume.

```bash
.venv/bin/python -m scripts.GPU.alphazero.eval_checkpoint_match \
  --checkpoint-a checkpoints/alphazero-v2-eps035-from0379/model_iter_0399.safetensors \
  --checkpoint-b checkpoints/alphazero-v2-staged/model_iter_0379.safetensors \
  --games 800 --workers 4 --base-seed 35791 \
  --save-eval-replays \
  --replay-dir logs/eval/eps035_0399_vs_0379_800g_w4_seed35791_replays \
  --output logs/eval/eps035_0399_vs_0379_800g_w4_seed35791_replay.json
```

**Key arguments**

| Flag | Default | Notes |
|---|---|---|
| `--checkpoint-a`, `--checkpoint-b` | (required) | `.safetensors` paths. A is the "subject", B the baseline. |
| `--games` | 400 | Must be even (colors are balanced by game-index parity). |
| `--save-eval-replays` | off | Write a per-ply replay sidecar per game + add `replay_path` to each jsonl row. Required for V2 analysis. |
| `--replay-dir` | `<output-stem>_replays` | Where sidecars go. Set explicitly if your stem differs from the dir you want. |
| `--workers` | 1 | Parallel games. **Hardware-dependent** (MLX/Metal); `>1` is best-effort and falls back to a `--workers 1` hint. See [`mlx-memory-management.md`](mlx-memory-management.md). |
| `--base-seed` | 12345 | Same seed + same code → identical games. Reuse a prior run's seed to reproduce its exact games (now with replays). |
| `--mcts-sims` | 400 | Search budget per move. |
| `--selection-mode` / `--opening-temp-plies` / `--temp-high` / `--temp-low` | `opening_temperature` / 20 / 1.0 / 0.1 | Temperature-sampled opening so games diverge, then near-argmax. **The first `--opening-temp-plies` plies are sampled** — the V2 analyzer excludes them from confidence signals. |
| `--max-moves` | 280 | State cap. Capped games score 0.5 / 0.5 (no resign/adjudication in v1). |
| `--output` | (required) | Summary JSON path. The `*_games.jsonl` is written next to it. |

**Outputs:** `<output>.json` (match summary) and `<stem>_games.jsonl` (one row
per game; carries `replay_path` when capture is on), plus
`<replay-dir>/game_NNNNNN.json` sidecars when `--save-eval-replays` is set.

---

## 2. `eval_checkpoint_tournament` — many pairings at once

**Purpose:** Run several pairings (explicit list, or round-robin over a checkpoint
set) through one shared worker pool, writing a match summary + `*_games.jsonl`
per pairing into `--output-dir`.

**When:** You're comparing more than two checkpoints (e.g. "is training still
gaining past iter 0419, or plateauing around 0379–0419?").

```bash
.venv/bin/python -m scripts.GPU.alphazero.eval_checkpoint_tournament \
  --checkpoints-dir checkpoints/alphazero-v2-staged \
  --pairings 0419:0379,0419:0339,0379:0339 \
  --games 400 --workers 4 \
  --output-dir logs/eval/tournament_0379_anchor_800g_w4
```

**Key arguments:** `--checkpoints-dir`, `--pairings` (or `--checkpoints` +
`--round-robin`), `--games`, `--workers`, `--base-seed`, and the same
`--mcts-*` / `--selection-mode` / `--max-moves` knobs as the match CLI.
`--output-dir` is required.

**Note:** The tournament CLI does **not** support `--save-eval-replays` yet
(per-pairing replay capture is the deferred "Phase C"). To get replays for a
specific pairing today, run that pairing through `eval_checkpoint_match` with
capture on.

---

## 3. `eval_loss_analyzer` — V1, game-level loss *shape*

**Purpose:** Read one or more `*_games.jsonl` and explain *how* checkpoint A
loses to B at the **game** level: score/Elo (matching the match summary exactly),
by-color and by-length breakdowns, a worst-losses sample, and a cross-branch
comparison. Read-only; no MLX.

**When:** After *any* match/tournament — it's cheap and needs no replay capture.
This is your first look: "A is weaker as black, concentrated in 61–120-move
decisive games," etc.

```bash
.venv/bin/python -m scripts.GPU.alphazero.eval_loss_analyzer \
  --glob "logs/eval/*_games.jsonl" \
  --output-dir logs/eval/loss_analysis
```

**Key arguments:** `--games-jsonl PATH` (repeatable) and/or `--glob PATTERN`;
`--output-dir` (default `logs/eval/loss_analysis`); `--a-checkpoint` /
`--b-checkpoint` to override A/B resolution; `--length-buckets`
(e.g. `40,60,80,120,279,280`); `--worst-losses N` (default 50).

**Outputs (per match `<stem>`):** `<stem>_loss_summary.json`,
`<stem>_by_color.csv`, `<stem>_by_length.csv`, `<stem>_worst_losses.csv`, and a
combined `combined_branch_comparison.csv` (descending by `a_score_rate`).

---

## 4. `eval_loss_replay_analyzer` — V2, per-ply loss *cause*

**Purpose:** Explain *why* and *when* checkpoint A collapses, using the per-ply
replay sidecars. Classifies each loss (value drop vs search diffusion vs
low-confidence moves vs already-lost), reports loss-vs-win effect sizes, an
A-vs-B "who saw it first" contrast, and produces a manual-review queue and
per-ply **drop windows** around each collapse.

**Prerequisite:** the match must have been run with `--save-eval-replays`
(rows need `replay_path` + the `_replays/` sidecars). Files without capture are
skipped with a note.

**When:** After V1 tells you *that* A loses in some regime and you want the
mechanism — e.g. "of A's black midgame losses, is it a value cliff or search
falling apart, and at what point in the game?"

```bash
.venv/bin/python -m scripts.GPU.alphazero.eval_loss_replay_analyzer \
  --games-jsonl logs/eval/eps035_0399_vs_0379_800g_w4_seed35791_replay_games.jsonl
```

**Key arguments**

| Flag | Default | Notes |
|---|---|---|
| `--games-jsonl` / `--glob` | — | Input(s); rows must carry `replay_path`. |
| `--output-dir` | `logs/eval/loss_analysis_v2` | |
| `--a-color` | `black` | Which seat to study (A is usually weaker as black). |
| `--min-moves` / `--max-moves` | 41 / 80 | Focus window (decisive games only; draws reported as excluded). |
| `--opening-plies` | 20 | **Confidence/diffusion features use plies ≥ this only** (the opening is temperature-sampled). Value-trajectory features use all plies. Keep aligned with the match's `--opening-temp-plies`. |
| `--bad-value` / `--lost-value` / `--sharp-drop` | −0.25 / −0.50 / 0.40 | Collapse-classification thresholds. |
| `--low-top1-share` / `--low-visit-rank` | 0.10 / 5 | Diffusion / low-confidence thresholds. |
| `--opening-key-plies` | 4 | Opening-cluster key length. |
| `--review-queue` | 50 | Rows in the manual-review queue. |

**Outputs (per match `<stem>`, 7 files):** `<stem>_replay_summary.json` (verdict
+ contrasts + distributions), `_cohort_comparison.csv`, `_phase_buckets.csv`,
`_collapse_timing.csv` (one row per focus game; carries `collapse_type`,
per-rule flags, `largest_drop_phase`, B-side columns), `_manual_review_queue.csv`
(top games by collapse sharpness, each with its `replay_path`),
`_opening_clusters.csv`, and `_drop_windows.csv` (per-ply window `[drop_ply ± 3]`
around each **post-opening** collapse). The console ends with a one-line verdict
and the queue path.

**Reading the output:** start from the console verdict, then prioritize review by
**post-opening** drops (filter the queue or `_drop_windows.csv` on
`largest_drop_phase == post_opening`) — those are the structural midgame
collapses, not the temperature-sampled opening. Open a game's `replay_path` or
its drop-window rows to see the value cliff ply-by-ply.

---

## 5. `eval_goal_line_trigger_probe` — checkpoint value-head calibration screen

**Purpose:** Re-evaluate a *fixed* set of "goal-line trigger" positions (black to
move, one ply before red's goal-line-completing move) with one or more
checkpoints, and report whether each **overvalues black** there. Lower black
`root_value` = better calibrated. It is a seconds-to-minutes diagnostic for one
specific value-head failure mode, not a full strength eval.

**When:** You suspect (e.g. from a V2 replay analysis) that a checkpoint's value
head is blind to red's goal-line conversion, and you want a fast targeted screen
across checkpoints *before* committing to an 800-game match.

**⚠️ Run from the repo root.** The probe manifest's cases carry `replay_path`
entries that are **relative to the repo root** (e.g.
`logs/eval/…_replays/game_000769.json`), and the probe resolves them against the
current working directory. Run every command below from the repo root, exactly as
the other eval CLIs expect — otherwise the replay reads fail with
`FileNotFoundError`. Path resolution is intentionally left CWD-relative to match
the rest of the eval tooling; this is a usage convention, not a bug.

**Two steps — build the fixed manifest (once), then probe:**

```bash
# 1. Generate the manifest from the curated candidates CSV (Mode A; reproducible).
.venv/bin/python -m scripts.GPU.alphazero.generate_goal_line_trigger_probe_manifest \
  --from-candidates-csv logs/eval/loss_analysis_v2_1/goal_line_trigger_probe_candidates.csv \
  --output logs/eval/loss_analysis_v2_1/goal_line_trigger_probe_manifest.json

# 2. Probe each checkpoint against the fixed positions.
.venv/bin/python -m scripts.GPU.alphazero.eval_goal_line_trigger_probe \
  --manifest logs/eval/loss_analysis_v2_1/goal_line_trigger_probe_manifest.json \
  --checkpoint checkpoints/alphazero-v2-staged/model_iter_0379.safetensors \
  --checkpoint checkpoints/alphazero-v2-eps035-from0379/model_iter_0399.safetensors \
  --output-dir logs/eval/goal_line_trigger_probe \
  --mcts-sims 400
```

**Key arguments (probe):** `--manifest` (required); `--checkpoint` (repeatable,
required); `--output-dir` (default `logs/eval/goal_line_trigger_probe`);
`--mcts-sims` (400); `--base-seed` (per-case search is seeded for
reproducibility). Checkpoints that share an iter number across different run dirs
are disambiguated by parent-dir name, so they never collide in the output.

**Key arguments (generator):** `--from-candidates-csv` and `--output` (required),
plus the selection knobs `--min-prev-black-value` (0.25), `--min-prev-black-top1`
(0.5), `--post-opening-only` / `--no-post-opening-only`, `--trigger-zone-prefix`
(`red_goal`). The defaults reproduce the canonical 18-case manifest. Re-deriving
the candidates CSV from a fresh capture ("Mode B") is deferred; the generator
consumes the curated candidates so the probe target stays fixed and reproducible.

**Outputs (in `--output-dir`):** `goal_line_trigger_probe_summary.json` (per
checkpoint: `num_cases`, mean/median `black_root_value`, `black_overvalue_rate`
≥+0.25, `severe_black_overvalue_rate` ≥+0.50, mean/median `top1_share`) and
`goal_line_trigger_probe_cases.csv` (one row per checkpoint×case, carrying the
in-game `baseline_black_prev_value` next to the probe's `probe_black_root_value`
so you can confirm a source checkpoint reproduces its own in-game evaluation).

**Reading the output:** a well-calibrated checkpoint reads **low/negative** black
value on these positions (it already sees red's goal-line threat). A high
`black_overvalue_rate` / positive `mean_black_root_value` is the failure mode —
the value head thinks black is winning right before red closes the goal line.
(Example readout: eps035 `0399` overvalued black on 94% of the cases, vs staged
`0379`'s 11%.)

The fixed positions come from the V2.1 loss analysis (sharp post-opening value
drops where black was confidently positive just before a red goal-band move); the
canonical 18-case manifest + candidates CSV live under
`logs/eval/loss_analysis_v2_1/`.

---

## 6. `build_targeted_calibration_manifest` — turn analysis findings into a calibration training manifest

**Purpose:** Assemble the **Targeted Value Calibration v2** *mixed* manifest — one
CSV where every row carries its own `target_black_value`, `weight_scale`, and
`tag`. **Correction** rows pull a known value-head error toward a hard target
(black pre-drop overvalue → −0.35); **retention** rows pin the fragile guardrail
families (red pre-drop, old broad post-opening, goal-line) to a checkpoint's
*own* `probe_black_root_value` (self-distillation), so the follow-on calibration
run fixes the target *without* moving the guardrails. Deterministic
(byte-identical re-run) and fails **loud** on any anchor ambiguity, frozen-eval
leak, or goal-line join mismatch — no silent drops.

**When:** *Before* a calibration training run, after the analysis CLIs above
(§3–§5) have produced the per-family probe / loss-analysis CSVs. This is the
bridge from "the analyzers told me which positions A misvalues" to "here is the
per-row training signal that corrects them while holding the guardrails." It
consumes the loss-analysis predrop manifests and the probe
`position_probe_cases.csv` / goal-line `*_cases.csv` + `*_candidates.csv`, and
produces the manifest that `train.py --post-opening-calibration-manifest` loads.

```bash
.venv/bin/python -m scripts.GPU.alphazero.build_targeted_calibration_manifest \
  --correction-manifest          logs/eval/loss_analysis_v2_calib020_0001_vs_0379_black/0001_black_post_opening_predrop_train_manifest.csv \
  --correction-holdout-manifest  logs/eval/loss_analysis_v2_calib020_0001_vs_0379_black/0001_black_post_opening_top30_predrop_probe_manifest.csv \
  --red-predrop-cases            logs/eval/calib020_0001_red_loss_post_opening_predrop_probe/position_probe_cases.csv \
  --old-post-opening-cases       logs/eval/black_predrop_calib010_checkpoint_sweep_old_post_opening/position_probe_cases.csv \
  --old-post-opening-anchor-label "alphazero-v2-calib020-from0409:0001" \
  --goal-line-cases              logs/eval/calib020_goal_line_sweep/goal_line_trigger_probe_cases.csv \
  --goal-line-candidates         logs/eval/loss_analysis_v2_1/goal_line_trigger_probe_candidates.csv \
  --out                          logs/eval/targeted_calibration_v2_from_calib020_0001.csv
```

**Key arguments**

| Flag | Default | Notes |
|---|---|---|
| `--correction-manifest` | (required) | Correction train rows (hard target) — the black pre-drop predrop train manifest from the V2 loss analysis. |
| `--correction-holdout-manifest` | (required) | The frozen eval holdout. The builder **asserts no `(replay_path, position_ply)` overlap** with the correction train set and fails loud if any frozen-eval position leaks in. |
| `--red-predrop-cases` | (required) | Retention D source: red pre-drop `position_probe_cases.csv` (self-sufficient — carries `replay_path`). |
| `--red-predrop-anchor-label` | `0001` | Checkpoint label whose `probe_black_root_value` becomes each red row's retention target. |
| `--old-post-opening-cases` | (required) | Retention C source: old broad post-opening `position_probe_cases.csv`. |
| `--old-post-opening-anchor-label` | `0001` | **Usually must be set explicitly** — e.g. `alphazero-v2-calib020-from0409:0001`. A bare `0001` is rejected as ambiguous when two `:0001` checkpoint labels exist in the source (strict resolver, below). |
| `--goal-line-cases` | (required) | Retention B *values*: goal-line `goal_line_trigger_probe_cases.csv` (carries the probe value but **no** `replay_path`). |
| `--goal-line-candidates` | (required) | Retention B *`replay_path`*: the goal-line `*_candidates.csv`, joined to the cases on `(game_idx, prev_black_ply)`. The builder requires exactly one candidate per case **and** that the replay file exists on disk. |
| `--goal-line-anchor-label` | `0001` | Anchor for the goal-line retention targets. |
| `--correction-target` | `-0.35` | Hard black-perspective target for every correction row. |
| `--retention-weight` | `0.5` | Per-sample `weight_scale` for *all* retention rows (correction rows are fixed at `1.0`). It is **relative within the calibration batch** — at the default a correction row carries 2× a retention row's per-sample influence; absolute force is `train.py --post-opening-calibration-weight`. |
| `--out` | (required) | Output CSV path (parent dirs are created). |

**Strict anchor resolution:** for each retention source the builder takes rows
whose `checkpoint` equals the anchor label exactly; failing that, the *unique*
set whose `checkpoint` ends with `":" + label`. More than one distinct `:label`
→ it errors and asks for the exact label; no match → it errors. This is why
old-post-opening usually needs the full `alphazero-v2-calib020-from0409:0001`
(its source also contains the failed `…calib010…:0001` branch).

**Outputs:** one unified CSV (`--out`) with 15 columns — `case_rank, tag, source,
source_rank, target_black_value, weight_scale, game_idx, case_id, replay_path,
position_ply, side_to_move, anchor_checkpoint, drop_ply, largest_drop_phase,
collapse_type` — plus a per-tag sanity summary to stdout:

```
black_predrop_correction:   n=50, weight_mass=50.0, target mean=-0.350 ...
goal_line_retention:        n=18, weight_mass= 9.0, target mean=-0.244 ...
old_post_opening_retention: n=30, weight_mass=15.0, target mean=+0.099 ...
red_predrop_retention:      n=30, weight_mass=15.0, target mean=-0.188 ...
wrote 128 rows -> logs/eval/targeted_calibration_v2_from_calib020_0001.csv
```

**Reading the output:** check the per-tag `target mean` against the source
baselines — a wrong anchor direction shows up immediately as a flipped sign. The
output CSV is a regenerable artifact (git-ignored under `logs/eval/`);
re-running on the same inputs is byte-identical.

**Feeds the next training run:** point the trainer's calibration flags at the
output — `train.py --post-opening-calibration-enabled
--post-opening-calibration-manifest <out.csv> --post-opening-calibration-weight
0.01 --post-opening-calibration-target -0.35
--post-opening-calibration-batch-fraction 0.10`. The pool auto-detects the
per-row schema (`schema=per_row_target`, `has_weight_scale=True`); a manifest
with no v2 columns loads byte-identically to v1 (`schema=global_target`,
plain-mean calibration loss). After training, re-run the §3–§5 probes as the
acceptance gates and (if they pass) a promotion match (§1) vs the current best.

**Smoke check (optional, before a GPU run):** `smoke_targeted_calibration_v2`
loads the real manifest, draws + splits a weighted batch, runs it through
`alphazero_loss_batch`, and asserts a finite weighted calibration term — a fast
end-to-end check that the manifest + replays + loss path are wired correctly.

```bash
.venv/bin/python -m scripts.GPU.alphazero.smoke_targeted_calibration_v2 \
  logs/eval/targeted_calibration_v2_from_calib020_0001.csv
# → "pool: 128 schema per_row_target ..." then "OK calib_loss=... calib_value_mean=..." (exit 0)
```

**Tag-stratified sampling (v3 — `--post-opening-calibration-tag-schedule`):** v2
draws the calibration pool *uniformly* (with replacement) each training step, so
the per-step correction:retention mix is whatever the manifest's natural tag
counts happen to be. v3 adds an explicit **tag-stratified draw schedule** so you
can over-weight correction draws relative to retention *per step* — without
touching weights, targets, or the loss math. When a schedule is set it
**replaces** uniform `--post-opening-calibration-batch-fraction` sampling (the
batch-fraction is then ignored). Pair it with a manifest built at
`--retention-weight 1.0` (uniform per-row `weight_scale`, so the *schedule alone*
controls the draw ratio).

```bash
.venv/bin/python -m scripts.GPU.alphazero.train \
  --post-opening-calibration-enabled \
  --post-opening-calibration-manifest logs/eval/targeted_calibration_v3_strat_from_calib020_0001.csv \
  --post-opening-calibration-weight 0.01 \
  --post-opening-calibration-target -0.35 \
  --post-opening-calibration-tag-schedule black_predrop_correction=2,goal_line_retention=1,old_post_opening_retention=2,red_predrop_retention=1
```

The schedule is `tag=count,tag=count` (each count a non-negative int; at least one
positive; duplicate/empty tags and missing `=` are rejected). Each training step
draws exactly `count` samples per tag with replacement, so the example draws
**2 correction : 1 goal-line : 2 old-post-opening : 1 red** every step. A tag named
in the schedule but **absent from the manifest fails fast — a `ValueError` at
trainer setup, before any self-play** — so a typo'd tag costs no iteration.

**Reading the per-tag draw telemetry (v3, Option A):** the run records how many
samples it drew per tag — a **dict** — in two JSON places, and **never** in
`metrics.csv` (which stays flat scalars only):
- `model_iter_<N>.json` → `state.calib_n_drawn_by_tag`
- the sidecar `iter_<N>_stats.json` → `post_opening_calibration.draws_by_tag`

Confirm both carry every scheduled tag in the intended ratio. **Index/location
gotcha:** the checkpoint is **1-based** (`model_iter_0001` = first iteration) but
the sidecar is **0-based** and lives in the games dir (default
`scripts/GPU/logs/games/iter_0000_stats.json` unless `--games-dir` was passed).
Per-tag *loss/value* means are deliberately **deferred** — v3 surfaces draw counts
only; the global `calib_loss_avg_iter` / `calib_mean_value_pred` telemetry is
unchanged. Omit the flag and the **sampling, loss, and optimization are identical**
to the v2 uniform path (the by-tag draw-count telemetry is still emitted there,
additively — it simply reports the manifest's natural tag distribution).

---

## 7. `build_teacher_calibration_manifest` — v4 teacher-retention manifest

**Purpose:** Read the v3 stratified manifest and cache the teacher checkpoint's
RAW forward (`infer`, no MCTS) over each retention row, writing `loss_mode`,
`teacher_value` (side-to-move), `teacher_policy_json` (dense, aligned to
legal_moves), and `teacher_legal_moves_sha1`. Correction rows pass through with
blank teacher columns.

```bash
.venv/bin/python -m scripts.GPU.alphazero.build_teacher_calibration_manifest \
  --source logs/eval/targeted_calibration_v3_strat_from_calib020_0001.csv \
  --teacher-checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --out logs/eval/targeted_calibration_v4_teacher_from_calib020_0001.csv
```

**Gate 0:** run `smoke_teacher_calibration_v4.py` after building — must pass
(`value_mse ≈ 0`, `kl_est ≈ 0`) before any training run:

```bash
.venv/bin/python -m scripts.GPU.alphazero.smoke_teacher_calibration_v4 \
  --manifest logs/eval/targeted_calibration_v4_teacher_from_calib020_0001.csv \
  --teacher-checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors
```

**Telemetry (v4+; v5 reuses the same keys):** a teacher/root-retention run persists,
per iteration, into `model_iter_<N>.json` state — `n_teacher_retention_drawn`,
`calib_policy_ce_avg_iter`, `calib_policy_kl_est_avg_iter`,
`calib_value_term_avg_iter`, `freeze_batchnorm_stats` — and the full block into
the sidecar `iter_<N>_stats.json` under `post_opening_calibration.loss`
(alongside `draws_by_tag`). Sanity-check after any run: `n_teacher_retention_drawn > 0`
and a finite `calib_policy_ce_avg_iter` prove retention rows actually took the
masked policy path (0 would mean value-only — a silent v3 rerun). The startup
log must print `mode=teacher_retention` (v4) / `mode=mcts_root_retention` (v5).

---

## 8. Targeted Value Calibration v5 — MCTS-root-visit policy retention

v5 keeps the v4 raw-teacher VALUE anchor on retention rows but replaces the
policy target with BASE's 400-sim MCTS root visit distribution (dense,
normalized, sha1-pinned). Rationale + full experiment record: the v5 section
of `docs/2026-06-26-targeted-value-calibration-experiment-ledger-v3f-v4-overlap-updated.md`
(root-value-only = v3, rejected; raw-priors policy = v4, rejected).

Build (offline, once, frozen):

    .venv/bin/python -m scripts.GPU.alphazero.build_mcts_root_retention_manifest \
      --source logs/eval/targeted_calibration_v3_strat_from_calib020_0001.csv \
      --base-checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
      --gate-cases-csv <BASE position_probe_cases.csv> \
      --gate-cases-csv <BASE goal_line_trigger_probe_cases.csv> \
      --gate-checkpoint-label 0001 \
      --out logs/eval/targeted_calibration_v5_mcts_root_from_calib020_0001.csv

The label selects BASE's rows inside gate cases CSVs that mix multiple
checkpoints (one row per checkpoint x case_id).

Gate-0 smoke (value ~0 REQUIRED; policy CE > 0 EXPECTED — do not "fix" it):

    .venv/bin/python -m scripts.GPU.alphazero.smoke_mcts_root_retention_v5 \
      --manifest logs/eval/targeted_calibration_v5_mcts_root_from_calib020_0001.csv \
      --base-checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors

Training reuses the v4 command verbatim with two deltas: the v5 manifest path
and a fresh checkpoint dir. Same flags: weight 0.01, teacher-value-weight 1.0,
teacher-policy-kl-weight 0.25, tag schedule 2:1:2:1, --freeze-batchnorm-stats.
Gates A–D vs calib020_0001; no promotion unless all four pass.

Known limitation (recorded in the ledger): root-visit anchors constrain the
candidate's policy AT the anchored root positions only. If gate drift comes
from value/prior changes deeper in the tree, v5 can still fail — in that case
the next hypothesis is tree/path-level retention, not more rows or weights.

---

## 9. `eval_position_probe` — generic fixed-position value probe (gates A/C/D)

**Purpose:** Re-evaluate a fixed CSV manifest of replay positions across
checkpoints with a full 400-sim MCTS search, reporting per-case black-perspective
root value + top-1 visit share and the over/severe flags. This is the runner
behind the **A (black pre-drop), C (old broad post-opening), and D (red pre-drop)
gates**; the goal-line probe (§5) is its B-gate sibling with a JSON manifest.

```bash
.venv/bin/python -m scripts.GPU.alphazero.eval_position_probe \
  --manifest logs/eval/tvc_v3_gate_C_old_post_opening_manifest.csv \
  --checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --checkpoint checkpoints/<candidate>/model_iter_0001.safetensors \
  --output-dir logs/eval/<run-name>
```

- Manifest needs at least `game_idx, case_id, replay_path, position_ply,
  side_to_move` (extra columns pass through to the output).
- Defaults: `--mcts-sims 400 --mcts-eval-batch-size 14 --mcts-stall-flush-sims 48
  --base-seed 20260616`; per-case rng seed = `base_seed ^ game_idx ^ position_ply`
  (the goal-line probe uses `base_seed ^ game_idx`, default 20260614).
- Outputs: `position_probe_summary.json` (per-checkpoint mean/median black root
  value, overvalue ≥ **0.25** and severe ≥ **0.50** rates) and
  `position_probe_cases.csv` (per-case `probe_black_root_value`,
  `probe_top1_share`, flags).
- **Gotcha:** the cases CSV is **one row per (checkpoint × case_id)**; checkpoint
  labels are the iter short-id (`0001`) or `parent-dir:short-id` when two runs
  share an iter number. Any case-keyed downstream lookup must filter by
  checkpoint first (see `--gate-checkpoint-label` in §8).

---

## 10. `eval_raw_nn_position_rows` — raw-NN (no-MCTS) drift-from-teacher diagnostic

**Purpose:** Score fixed manifest rows across checkpoints with the RAW network
forward only (eval-mode BatchNorm, no search) and quantify per-position
`value_delta_vs_teacher`. Separates **value-head drift** from **MCTS-root
drift**: when a candidate's raw values match the teacher but its §9 gate values
went severe, the failure is root/search behavior, not the raw net. (This is the
diagnostic that redirected the calibration line from raw-teacher sweeps to v5
root retention — 2026-07-01, see the experiment ledger.)

```bash
.venv/bin/python -m scripts.GPU.alphazero.eval_raw_nn_position_rows \
  --manifest logs/eval/tvc_v3_gate_C_old_post_opening_manifest.csv \
  --manifest logs/eval/tvc_v2_gate_D_red_predrop_manifest.csv \
  --checkpoint <BASE.safetensors> --checkpoint <candidate.safetensors> \
  --base-checkpoint <BASE.safetensors> \
  --case-id game_000369_ply_051 \
  --out logs/eval/raw_nn_rows.csv
```

- `--manifest` is repeatable (rows unioned + deduped); `--case-id` / `--tag` /
  `--limit` filter; `--base-checkpoint` defaults to the first `--checkpoint`.
- Teacher reference per row: the manifest's `teacher_value` if present, else the
  **BASE checkpoint's own raw value** for that case (`teacher_value_source`
  column says which). `value_delta_vs_teacher` is side-to-move space (no flip);
  `raw_black_value` / `overvalue` / `severe_overvalue` are black-perspective
  (same 0.25/0.50 thresholds as §9).
- Also emits `top1_move` / `top1_prob` (raw policy argmax) for a value-vs-policy
  drift lens. Booleans serialize as the strings `"True"`/`"False"` — string-compare,
  don't truthy-test.

---

## Internal libraries (not run directly)

- `eval_runner` — the game-playing task queue / worker pool used by the match and
  tournament CLIs.
- `eval_summary` — aggregates match results into the summary JSON sidecar.
- `eval_elo` — pure stats (score rate, Elo, draw-aware trinomial CI).
- `eval_replay` — the replay sidecar schema + writer (capture path).
- `eval_loss_analysis`, `eval_loss_replay_analysis` — the pure analysis modules
  behind the V1 and V2 CLIs (importable, fully unit-tested).
- `goal_line_trigger_probe_cases` — pure selection / board-reconstruction /
  summary helpers behind the goal-line trigger probe (no MLX, unit-tested);
  its `position_state` is the canonical replay→board reconstructor shared by
  every probe, diagnostic, and calibration builder.
- `position_probe_cases` — CSV-manifest loader (`load_csv_manifest`) + the
  shared `OVERVALUE_THRESHOLD`/`SEVERE_OVERVALUE_THRESHOLD` (0.25/0.50) and
  summary helpers behind the generic position probe (§9) and the raw-NN
  diagnostic (§10).
- `calibration_pool` — training-side loader/validator for every calibration
  manifest (§6/§7/§8): `loss_mode` registry (`hard_value` / `teacher_retention` /
  `mcts_root_retention`), per-row parsing + sha1 alignment checks, sampling,
  and the retention mask consumed by the trainer's masked loss path.
- `build_teacher_calibration_manifest._teacher_infer` — the shared
  single-position raw forward (no MCTS) reused by §10 and the §8 builder.

## Typical end-to-end workflow

1. **Play with capture:** `eval_checkpoint_match … --save-eval-replays` →
   `…_games.jsonl` + `…_replays/`.
2. **Shape (always):** `eval_loss_analyzer --glob "logs/eval/*_games.jsonl"` →
   which color / length band A is weak in.
3. **Cause (when needed):** `eval_loss_replay_analyzer --games-jsonl …_games.jsonl`
   → the collapse mechanism, timing, and a review queue.
4. **Inspect:** open the top post-opening drop windows against their replay
   sidecars.
5. **Targeted screen:** when V2 shows goal-line-trigger / defender-side value cliffs,
   run `eval_goal_line_trigger_probe` on candidate checkpoints before spending time
   on another full match.

## See also

- Designs/plans: `docs/superpowers/specs/` and `docs/superpowers/plans/` —
  `*-checkpoint-tournament-*`, `*-eval-loss-analyzer-*`, `*-eval-replay-capture-*`,
  `2026-06-12-eval-replay-analyzer-*`, `2026-06-14-goal-line-trigger-probe-*`,
  and the calibration builder's `2026-06-23-targeted-value-calibration-v2-design.md`
  + `2026-06-24-targeted-value-calibration-v2.md` (per-tag baselines, gates A–D,
  promotion). The v1 single-target predecessor is
  `2026-06-16-post-opening-sharp-drop-calibration-*`. For §10 and §8:
  `2026-07-01-eval-raw-nn-position-rows-diagnostic.md` and
  `2026-07-01-targeted-value-calibration-v5-mcts-root-retention.md`.
- Experiment record (which calibration branches were tried, why they were
  rejected, do-not-repeat list):
  `docs/2026-06-26-targeted-value-calibration-experiment-ledger-v3f-v4-overlap-updated.md`.
- Metric definitions: [`analysis-metrics-guide.md`](analysis-metrics-guide.md).
- MLX/Metal eval performance and the `--workers` gotcha:
  [`mlx-memory-management.md`](mlx-memory-management.md).
