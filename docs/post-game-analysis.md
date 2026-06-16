# Post-Game Analysis — Checkpoint Eval & Loss Analyzers

Operator guide for the scripts that run **after** training has produced
checkpoints: play checkpoints against each other, then analyze *who* wins, *how*
they lose, and *why*. All scripts live in `scripts/GPU/alphazero/` and run as
`python -m scripts.GPU.alphazero.<name>` from the repo root (use `.venv/bin/python`).

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

## Internal libraries (not run directly)

- `eval_runner` — the game-playing task queue / worker pool used by the match and
  tournament CLIs.
- `eval_summary` — aggregates match results into the summary JSON sidecar.
- `eval_elo` — pure stats (score rate, Elo, draw-aware trinomial CI).
- `eval_replay` — the replay sidecar schema + writer (capture path).
- `eval_loss_analysis`, `eval_loss_replay_analysis` — the pure analysis modules
  behind the V1 and V2 CLIs (importable, fully unit-tested).
- `goal_line_trigger_probe_cases` — pure selection / board-reconstruction /
  summary helpers behind the goal-line trigger probe (no MLX, unit-tested).

## Typical end-to-end workflow

1. **Play with capture:** `eval_checkpoint_match … --save-eval-replays` →
   `…_games.jsonl` + `…_replays/`.
2. **Shape (always):** `eval_loss_analyzer --glob "logs/eval/*_games.jsonl"` →
   which color / length band A is weak in.
3. **Cause (when needed):** `eval_loss_replay_analyzer --games-jsonl …_games.jsonl`
   → the collapse mechanism, timing, and a review queue.
4. **Inspect:** open the top post-opening drop windows against their replay
   sidecars.

## See also

- Designs/plans: `docs/superpowers/specs/` and `docs/superpowers/plans/` —
  `*-checkpoint-tournament-*`, `*-eval-loss-analyzer-*`, `*-eval-replay-capture-*`,
  `2026-06-12-eval-replay-analyzer-*`, `2026-06-14-goal-line-trigger-probe-*`.
- Metric definitions: [`analysis-metrics-guide.md`](analysis-metrics-guide.md).
- MLX/Metal eval performance and the `--workers` gotcha:
  [`mlx-memory-management.md`](mlx-memory-management.md).
