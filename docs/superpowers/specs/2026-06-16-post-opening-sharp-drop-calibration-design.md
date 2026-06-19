# Post-Opening Sharp-Drop Calibration — Design

**Date:** 2026-06-16
**Status:** Approved design, pre-implementation
**Author:** bill-osienski (with Claude Code)

## 1. Motivation

`model_iter_0379` remains the promoted-best checkpoint. `model_iter_0409`
(`checkpoints/alphazero-v2-lr0003-eps035-from0379/`) is a **parity candidate**:
it ties 0379 in match play and *repaired* the narrow 18-case goal-line trigger
probe. But it has one sharp, well-characterized weakness that blocks promotion.

On the 30-case **post-opening sharp-drop probe** (black-to-move positions where
0409-as-black, having just lost to 0379-as-red, held a *confidently positive*
value one to two plies before a sharp value collapse):

| Checkpoint | overvalue rate (`v ≥ 0.25`) | severe rate (`v ≥ 0.50`) | mean black root value |
|---|---|---|---|
| **0379** (target behavior) | 3.3% | 3.3% | −0.469 |
| **0409** (current) | **93.3%** | **76.7%** | **+0.604** |

0409 is *confidently wrong* on exactly the class of positions where 0379 already
recognizes danger. `0414` continuations (training further from 0409) moved some
broad cases but **regressed goal-line gates** — blind continuation is not the answer.

**Goal:** teach 0409 the danger-recognition it lacks on this position class, with
a small, controlled nudge that does **not** sacrifice the goal-line repair or
general match strength. The output is a new candidate that can be gated and, if it
passes, matched against 0379 for promotion.

## 2. Decision summary

- **Approach:** outcome-grounded, **value-only**, **soft-target** calibration, added
  as a **separate auxiliary value-loss term** (Mechanism B). *Not* teacher
  distillation from 0379 (deferred — see §11), *not* a hard −1 outcome label.
- **Why value-only:** the eval-game replay JSONs store only the *played* move plus
  `root_value`/`top1_share` per ply — there is **no full visit distribution**, so a
  policy target literally is not available without re-running search. The replay
  analyzer also reports selected-move rank is usually 1 in these games; the failure
  is **value recognition**, not move selection. Value-only is the correct first cut.
- **Why soft −0.50, not −1.0:** these positions sit ~1–2 plies before the cliff and
  some retain defensive resources. Hard −1 risks teaching "already lost." The lesson
  we want is "black should *not* believe this is +0.60 to +0.90." Target = **−0.50**
  (black perspective). A more conservative −0.35 is a one-flag change if needed.
- **Why Mechanism B (separate pool + aux term), not main-buffer injection:** these are
  diagnostic replay positions with value supervision only. Forcing them through the
  main `ReplayBuffer` would require policy-masking, mixed value semantics,
  progress-weighting ambiguity, and interaction with the existing conversion
  aux-loss sampler — all risk to the normal self-play path. A separate pool + a
  cleanly-added value MSE term keeps the main loss path byte-identical when disabled.
- **Base:** train from `model_iter_0409.safetensors`.
- **Disabled by default:** with no manifest / not enabled, behavior is exactly today's.

## 3. Scope / non-goals

**In scope:** a separate calibration pool of fixed external positions; a value-only
auxiliary MSE term folded into the existing `train_step`; CLI flags; per-iteration
diagnostics; a held-out gating protocol; unit tests; a train-manifest builder that
excludes the frozen probe games.

**Non-goals (YAGNI for this iteration):**
- No policy target / no policy distillation on calibration positions.
- No teacher (0379) inference pass. (Deferred; §11.)
- No `sample_boost` knob — with a dedicated pool there is no "natural rate" to boost
  against; `batch_fraction` directly sets the mini-batch size.
- No new self-play matches — the training pool already exists (§4).
- No red-to-move calibration cases yet — the current pool is all black-to-move. The
  perspective helper is written generically so red-to-move is a data change, not a
  code change.

## 4. Data: train / eval split

The replay analyzer cohort (`logs/eval/loss_analysis_v2_lr0003_0409/…replay_summary.json`)
defines the eligible failure pool precisely:

| Set | Count | Definition |
|---|---|---|
| Eligible failure pool (`loss` cohort: a_color=black, post-opening, sharp-drop filters) | **164** | 0409-as-black lost to 0379-as-red with a qualifying post-opening sharp value drop |
| **Frozen eval / gate** (`post_opening_sharp_drop_probe_manifest.csv`) | **30** | the *most severe* drops, top-ranked; already written and version-controlled |
| **Training pool** | **~134** | ranks 31–164, **disjoint by `game_idx`** from the frozen 30 |

Split discipline:
- Train and eval are **disjoint by `game_idx`** — zero leakage.
- The eval set is the **hardest tail**; training is the **milder body**. This is a
  genuine generalization test (generalize from milder examples to the hardest tail),
  not contamination. A consequence: initial improvement on the 30 may be **modest** —
  that is acceptable. First target is reducing catastrophic overconfidence, not
  solving all 30.
- **Invariant — the frozen 30 are sacred.** Those 30 `game_idx` values are never added
  to any calibration training manifest. If a future experiment needs them for training,
  a *new, larger* holdout must be minted first; the current frozen 30 are not repurposed.

### 4.1 Train-manifest builder

Both manifests share one schema and one loader (`position_probe_cases.load_csv_manifest`,
required columns `game_idx, case_id, replay_path, position_ply, side_to_move`). The
training manifest is produced by:

1. Taking the analyzer's full ranked post-opening sharp-drop output (the same source
   the frozen 30 were selected from — the `manual_review_queue` / drop-window ranking
   over the 164-game cohort).
2. **Excluding the 30 frozen `game_idx` values** (`1 65 73 103 113 191 277 309 319 349
   369 403 425 433 463 499 505 511 565 593 595 603 619 637 639 681 695 705 735 793`).
3. Writing the survivors (ranks 31–164, all `side_to_move=black`,
   `largest_drop_phase=post_opening`, `collapse_type=sharp_value_drop`) to a train
   manifest CSV in the shared schema.

This is a small selection step over existing data — **no new matches or self-play.**
The exact emit point (a flag on the analyzer vs. a tiny standalone selector) is an
implementation-plan detail; the contract is: *same schema, holdout excluded,* and
**fully deterministic** — same source queue, same excluded `game_idx` list, stable
tie-broken ordering (e.g. by `case_rank` then `game_idx`), and a stable output path, so
re-running produces a byte-identical manifest.

## 5. Mechanism (B): separate calibration pool + auxiliary value term

### 5.1 Components

**`CalibrationPool`** (new, pure / framework-light where possible):
- Loads the train manifest via the shared CSV loader.
- For each case: reconstruct the board with the existing
  `position_state(replay, position_ply, side_to_move)`
  (`goal_line_trigger_probe_cases.py:73`) — load the replay JSON, apply
  `moves[0:position_ply]` to a fresh `TwixtState`, assert reconstructed `to_move`
  matches `side_to_move` (fail loud otherwise).
- **Eagerly** encode each state to the 30-channel board tensor (same encoder the
  trainer uses to build `PositionRecord.board_tensor`) and cache, with its
  `legal_moves`, `side_to_move`, and per-case target value (default −0.50, black
  perspective). ~134 positions — cheap to hold in memory.
- `sample(k, rng)` → k records, with replacement (k is small, ~6).

**Loss integration** — extend `alphazero_loss_batch` / `train_step` with optional
`calibration_batch`, `calibration_weight`, `calibration_target`:
- Build network inputs for the calibration mini-batch the same way
  `make_padded_batch` does (board tensors + padded legal-move row/col/mask arrays),
  at **`active_size=24`** (these are full-board positions; this is independent of the
  curriculum stage of the main batch — moot in practice since fine-tuning from 0409
  runs at full 24 throughout).
- Forward pass; use the **value head output only** (ignore policy logits).
- Compute `calib_value_loss = mean( (value_in_to_move − target_in_to_move)^2 )`.
- `total_loss += calibration_weight * calib_value_loss`. The term is **standalone** —
  it is *not* additionally multiplied by `value_weight`; `calibration_weight` (0.02)
  is the absolute coefficient.
- Because the term is added to `total_loss` inside the existing `train_step`, the
  current dual-optimizer setup (`opt_main` = encoder+policy, `opt_value` = value head)
  and gradient clipping apply unchanged: the calibration gradient flows into the
  encoder via `opt_main` and the value head via `opt_value`, exactly as the normal
  value loss does.

### 5.2 Perspective helper (mirror the probe)

The network value head outputs value in **side-to-move** perspective (post
canonicalization). The probe normalizes to black perspective with
(`eval_position_probe.py:85-91`): black-to-move ⇒ value as-is; red-to-move ⇒ negate.
Training mirrors this on the **target** side instead:

```
target_in_to_move = calibration_target           if side_to_move == "black"
                  = -calibration_target           if side_to_move == "red"
```

For the current all-black pool this is simply −0.50, no inversion. The helper is
written generically so a future red-to-move case is a data change only.

### 5.3 Data flow (per training step)

```
buffer.sample(batch_size, …)         → main self-play batch          (unchanged)
calib_pool.sample(k, rng)            → k = round(batch_size × batch_fraction)
train_step(… main batch …,
           calibration_batch=k recs,
           calibration_weight=w,
           calibration_target=t):
    total = policy + value_weight*value + l2 + conv_weight*aux        (unchanged)
          + w * mean( (value_pred_to_move − target_in_to_move)^2 )    (NEW)
    backprop total via opt_main + opt_value (existing clip/step)
```

Disabled (not enabled / no manifest) ⇒ **truly byte-identical to today**: the pool is
not loaded, no calibration forward pass runs, the NEW line is absent, and the returned
loss-tuple structure is unchanged. The extended tuple / calib stats appear *only* when
calibration is enabled.

## 6. CLI flags / defaults

Mirrors the existing `--conversion-*` naming convention in `train.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--post-opening-calibration-enabled` | `False` | master switch |
| `--post-opening-calibration-manifest <path>` | `None` | train manifest CSV; required when enabled |
| `--post-opening-calibration-target` | `-0.50` | soft value target, black perspective |
| `--post-opening-calibration-weight` | `0.02` | absolute coefficient on the calib term |
| `--post-opening-calibration-batch-fraction` | `0.10` | k = round(batch_size × fraction) |
| `--post-opening-calibration-max-cases` | `None` (later) | cap pool size for ablation |
| `--post-opening-calibration-seed` | derived (later) | reproducible pool sampling |

Disabled-by-default semantics: enabling without a manifest is an error; not enabling
loads no pool and adds no loss term (zero behavioral change).

## 7. Diagnostics / logging

Extend the per-iteration loss-split sidecar (alongside the existing `sum_aux*`
accumulators in the training loop). Emit **every iteration** (not gated behind a
sampling interval):
- `calib_loss` (mean over steps),
- `calib_n_drawn` (mini-batch size actually used),
- **`calib_mean_value_pred`** — the network's mean value on calibration positions,
  logged **every iteration** as the headline learning signal: we expect it to drift
  from ~**+0.60 toward −0.50** over the run. If it does not move, the weight is too
  small (or the nudge is being overwhelmed by self-play). This is the fastest signal
  that the mechanism is doing anything at all.

## 8. Gates & success criteria

Evaluate in this order; **do not** run the match until both probes pass.

**Gate 1 — goal-line trigger probe (18 cases): must not regress vs 0409.**
Re-measure 0409's current goal-line numbers as the baseline, then require the
candidate to satisfy:
- severe overvalue rate **≤ 5.6%** (≤ 1 / 18),
- overvalue rate **≤ 11.1%** (≤ 2 / 18),
- **no per-case new severe spike** — no individual case that 0409 held under threshold
  may newly cross into severe overvalue (watch the cases that spiked in prior
  continuations, e.g. 315 / 535 / 537).

**Gate 2 — post-opening sharp-drop probe (frozen 30): must improve materially vs 0409**
(baseline overvalue 93.3% / severe 76.7% / mean +0.604; target behavior is 0379's
overvalue ~3.3% / mean ~ −0.469). First-experiment acceptance bar:
- severe overvalue rate **≤ 60%**, **and**
- mean black root value **≤ +0.35** (moved downward from +0.604).

Clearing this bar makes the candidate "worth a match or another short branch." The
stretch direction is materially toward 0379, but solving all 30 is **not** required
on the first pass.

**Gate 3 — 800-game match vs 0379:** only if Gates 1 and 2 pass. The candidate must be
**non-inferior** to 0379 (parity or better) to justify promotion.

## 9. Run configuration

- Base: `checkpoints/alphazero-v2-lr0003-eps035-from0379/model_iter_0409.safetensors`.
- Calibration **enabled** with MVP knobs: target −0.50, weight **0.02**,
  batch-fraction 0.10, train manifest = ranks 31–164.
- **Short** run (e.g. 10–20 iterations) — this is a nudge, not a retrain. Normal
  self-play continues so general strength is maintained; the small weight is what
  keeps self-play dominant.
- `calib_mean_value_pred` is logged every training iteration (cheap, no MCTS). Run the
  two probe **suites** (MCTS-based) at the end, and optionally at a checkpoint interval,
  to watch the gate metrics trend.
- If Gate 2 is missed but trending, the conservative next branches are weight → 0.05,
  or target → tighter — re-checking Gate 1 each time.

## 10. Testing

Unit tests (fakes / no real checkpoint where possible, following the existing
`FakeEvaluator` / fixture pattern):
- **Disabled path is inert:** with calibration not enabled, the total loss equals the
  no-calibration path exactly, the pool is never loaded, no calibration forward pass
  runs, and the returned loss-tuple structure is unchanged. (Enabled-with-weight-0 is a
  separate degenerate case: the pool loads and the forward runs, but the term is 0.)
- **Pool load + reconstruction:** `CalibrationPool` loads a manifest, reconstructs
  states via `position_state`, asserts `to_move == side_to_move`, and fails loud on a
  bad `position_ply` or a perspective mismatch.
- **Perspective helper:** black-to-move ⇒ target −0.50; red-to-move ⇒ +0.50.
- **Loss math:** the calibration term equals `weight × MSE(value, target_in_to_move)`;
  on a toy network, repeated steps reduce `|value − target|` on the pool positions.
- **Mini-batch sizing:** `k = round(batch_size × batch_fraction)`; small-pool sampling
  with replacement is stable.
- **Manifest builder:** the emitted train pool excludes all 30 frozen `game_idx`
  values (no leakage) and conforms to the shared schema.

## 11. Risks, mitigations, and future work

- **Small weight may not move the hard tail.** Accepted: the first goal is reducing
  catastrophic overconfidence, gated by §8. Escalation path: weight 0.05, then the
  deferred options below.
- **Overcorrection harming goal-line.** Gated explicitly (Gate 1, including per-case).
- **Catastrophic forgetting / match regression.** Gated by the 800-game match (Gate 3),
  run only after the probes pass.
- **Train/eval severity skew.** Intended (generalization test); makes the gate honest.
- **Deferred — teacher distillation from 0379 (Option 2/3).** If value-only soft
  targeting under-delivers, distill 0379's MCTS value (and optionally its visit
  policy, which would also supply the missing policy target) at these positions,
  possibly clipped to the known loss sign (hybrid). More machinery and risks copying
  0379's broader biases — revisit only with evidence the simple nudge is insufficient.

## 12. Rollback / ablation

Disabled by default; a single weight knob; the pool and loss term are isolated from
the main self-play path. Reverting to current behavior is "don't pass the flags."
