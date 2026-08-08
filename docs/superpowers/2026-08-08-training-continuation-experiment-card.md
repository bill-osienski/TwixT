# Training Continuation — Experiment Card

**Date:** 2026-08-08 · **Status:** DRAFT, not countersigned · **Scope: one five-iteration
training run and one 400-game evaluation. Nothing else.**

Successor to the closed competitive-readout line
(`docs/superpowers/2026-08-08-competitive-readout-closeout.md`). Tests the
**network/training axis** directly. Revives no calibration row engineering, distillation,
search knob, readout formula, or the unvalidated A/B/C/D proxy pipeline.

## Hypothesis

A short, ordinary self-play continuation from the accepted best `calib020_0001` — using
game outcomes rather than calibration targets or deeper-search teachers — produces a
large, directly measurable strength gain.

## Prediction, on record before the run

**Null or mildly negative.** Continuations after `0379` mostly plateaued or regressed
(staged `0399` −85; eps035 `0399` −22, unresolved; lr0003 `0409` ≈ equal; staged `0419`
−33). The test is worth running because it is direct, cheap, and can detect a large gain
if one exists — not because a gain is expected. `calib020_0001` (+80.0, CI95
[+55.9, +104.8], confirmed in both directions) shows the network had not hit a hard
capacity ceiling.

Historical figures: `logs/eval/current_best_and_candidates.md` and the experiment ledger.
Cite those, not recollection.

## Parent

```
checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors
sha1  209cf2d4fd24a48553d259dd71b4954867b9473e
```

## Dose — state it precisely

> Five iterations comprising **500 self-play games and 800 training steps**
> (160/iteration at board size 24), starting from an **empty replay buffer**.

For reference, `train_overnight.sh` is a *different, smaller* configuration (20
games/iter) that did **not** produce these checkpoints. `games_per_iter: 100` is what both
the `staged 0379` and `calib020_0001` sidecars record.

The replay buffer is constructed fresh (`trainer.py:2896`) with **no save/load anywhere**,
so all five iterations sit in the cold-start regime; the historical 50-iteration runs
washed that out and this one cannot.

## Training command — every parameter pinned

Provenance per parameter: **[S]** recorded in the checkpoint sidecars · **[R]** the
`f132452` runbook · **[T]** the code's scaling table · **[D]** CLI default, pinned
explicitly · **[O]** operator choice, not reconstructed.

```bash
nohup bash -c '.venv/bin/python -m scripts.GPU.alphazero.train \
  --load-weights checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --checkpoint-dir checkpoints/alphazero-v2-cont5-from-calib020 \
  --games-dir logs/selfplay/cont5_from_calib020 \
  --iterations 5 \
  --games-per-iter 100 \
  --train-steps 160 \
  --simulations 400 \
  --lr 0.0003 \
  --l2 0.0001 \
  --batch-size 64 \
  --buffer-size 100000 \
  --curriculum-sizes 24 \
  --hidden 128 \
  --blocks 6 \
  --seed 20260808 \
  --n-workers 10 \
  --mcts-eval-batch-size 14 \
  --mcts-stall-flush-sims 48 \
  --mcts-pending-virtual-visits 8 \
  --max-moves 280 \
  --max-positions-per-game 280 \
  --endgame-keep-positions 16 \
  --mirror-prob 0.5 \
  --dirichlet-alpha 0.3 \
  --dirichlet-eps 0.25 \
  --opening-noise-ply 0 \
  --temp-high 1.0 \
  --temp-low 0.1 \
  --temp-threshold-ply 20 \
  --value-weight 0.5 \
  --value-lr-scale 0.1 \
  --value-grad-max-norm 0.5 \
  --progress-weighted-value-loss \
  --progress-weight-floor 0.25
rc=$?
printf "%s\n" "$rc" > logs/train/cont5_from_calib020.exit
exit "$rc"' \
  > logs/train/cont5_from_calib020.stdout 2>&1 &
disown
```

| parameter | value | src |
|---|---|---|
| `--games-per-iter` | 100 | **[S]** |
| `--train-steps` | 160 | **[T]** `TRAIN_STEPS_TABLE[24]`; sidecars record `None`=auto, which resolves to 160 |
| `--simulations` | 400 | **[S]** |
| `--lr` | 0.0003 | **[R]** |
| `--l2` | 0.0001 | **[D]** |
| `--batch-size` / `--buffer-size` | 64 / 100000 | **[S]** |
| `--curriculum-sizes` | 24 | **[S]** calib020 records `sizes:[24], idx 0` — **not** the `8,10,12,16,20,24` default |
| `--mcts-eval-batch-size` / `--max-moves` | 14 / 280 | **[S]** |
| `--hidden` `--blocks` `--mirror-prob` `--value-*` `--progress-*` `--dirichlet-*` `--temp-*` `--opening-noise-ply` `--mcts-pending-virtual-visits` `--mcts-stall-flush-sims` `--max-positions-per-game` `--endgame-keep-positions` | as written | **[D]** pinned explicitly |
| `--seed` | 20260808 | **[O]** training seed, distinct from the evaluation interval |
| `--n-workers` | 10 | **[O]** top of the CLI's "max recommended: 10" band |

**`--lr 0.0003` is documented for the run that created the parent** (`f132452`), and that
run carried `--post-opening-calibration-*` flags. This experiment **inherits the general
training configuration and deliberately drops the calibration objective** — it does not
reproduce the parent's complete objective.

**Disabled by omission, asserted positively:** resign and adjudication
(`--resign-enabled` / `--adjudicate-enabled` are `action="store_true"`, default false);
root edge/corner penalties; every `--post-opening-calibration-*` flag; all other
experimental/auxiliary objectives. Absence here is a decision, not an oversight.

**`--load-weights`, not `--resume`.** `--load-weights` starts a fresh iteration counter,
so five iterations produce `model_iter_0005.safetensors` in the new checkpoint dir.
`--resume` would continue the parent's counter and produce a different filename.

## Frozen endpoint

**Only `checkpoints/alphazero-v2-cont5-from-calib020/model_iter_0005.safetensors` is
evaluated.** Iterations 1–4 are **not** evaluated, probed, or selected from, under any
result. That is what prevents checkpoint shopping.

## Evaluation — 400 games, candidate vs parent

```bash
nohup bash -c '.venv/bin/python -m scripts.GPU.alphazero.eval_checkpoint_match \
  --checkpoint-a checkpoints/alphazero-v2-cont5-from-calib020/model_iter_0005.safetensors \
  --checkpoint-b checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --games 400 \
  --base-seed 202608988 \
  --board-size 24 \
  --mcts-sims 400 \
  --mcts-eval-batch-size 14 \
  --mcts-stall-flush-sims 48 \
  --selection-mode opening_temperature \
  --opening-temp-plies 20 \
  --temp-high 1.0 \
  --temp-low 0.1 \
  --max-moves 280 \
  --workers 4 \
  --output logs/eval/cont5_vs_calib020.json
rc=$?
printf "%s\n" "$rc" > logs/eval/cont5_vs_calib020.exit
exit "$rc"' \
  > logs/eval/cont5_vs_calib020.stdout 2>&1 &
disown
```

Seed interval **`[202608988, 202609388)`**, half-open, 400 wide. Priors, per the seed
ledger's closing note: `[202608060, 202608124)`, `[202608124, 202608188)`,
`[202608188, 202608988)`.

**Seed disjointness is NOT code-enforced on this path.** `eval_checkpoint_match` has no
`--prior-seed-interval` and never calls `validate_seed_intervals` — verified. Disjointness
here is a manual precondition and a ledger entry, nothing more.

`--workers 4` is an **[O]** operator choice per `docs/post-game-analysis.md:56`.

## Preconditions

1. **New output paths, refuse if any exists** — `checkpoints/alphazero-v2-cont5-from-calib020`,
   `logs/selfplay/cont5_from_calib020`, `logs/train/cont5_from_calib020.{stdout,exit}`,
   `logs/eval/cont5_vs_calib020.{json,stdout,exit}`, `logs/eval/cont5_vs_calib020_games.jsonl`.
2. **Clean worktree**; HEAD recorded with both results.
3. **Suite passes**, measured immediately before the training run.
4. **Parent path and sha1 confirmed** against the block above.

## Gate between the two runs

Evaluation starts **only** when both hold:

- `logs/train/cont5_from_calib020.exit` contains **`0`**; and
- `checkpoints/alphazero-v2-cont5-from-calib020/model_iter_0005.safetensors` **exists**.

A non-zero training exit, a missing endpoint, or a different filename means **stop** —
not evaluate what was produced.

## Decision rule

**Success:** the candidate's overall **95% lower bound above 50%** — at 400 games,
an observed score rate around `0.549`, i.e. a conspicuous gain of roughly **+35 Elo or
more**. Also reject on convincing one-colour harm (a colour's own 95% upper bound below
50%).

**Anything else:** no credible large gain. **Stop this continuation recipe.**

Do **not**, after seeing the result: extend to iteration 10, change the learning rate,
select an earlier iteration, add games, or rerun the match.

## On success — replicate the whole training run

A win requires **rerunning the entire five-iteration training experiment from the original
parent `calib020_0001`** — not continuing from the first candidate, and not rematching it.
Rematching would test match variance; the question is recipe repeatability.

The replication uses the same base checkpoint and frozen recipe, a **different explicit
`--seed`**, a **different 400-game evaluation interval**, and its own frozen
`model_iter_0005` endpoint.

## Deliberate omissions, and what they cost

- **No 64-game screen.** The last one returned an unresolved eligible-but-meaningless
  result in front of a cheap decisive test.
- **No A/B/C/D gating before strength.** **Cost, stated: a strength-neutral result with a
  behavioural regression will not be visible here.** The probes still exist and run in
  seconds — they precede any promotion, they do not gate this test.
- **No 0379 rematch.** Parent-versus-child answers the incremental question at half the
  cost.
- **No external strength anchor.** Worthwhile only for an ongoing program; unnecessary for
  one causal parent-versus-child test measured inside a single match.
- **No match against `calib020_0015`** — known goal-line regression, so even a win would
  not be cleanly actionable.
- **No new runner, corpus, telemetry analyzer, replay analysis, probe, or code.**

## A positive result is research evidence only

It does not promote a checkpoint. Safety probes precede promotion, and adoption is a
separate decision with its own record.

## No worker fallback, no partial continuation

Changing `--n-workers`, `--workers`, or any pinned parameter **invalidates the run** — it
does not trigger an automatic fallback. A retry, a partial continuation, or a resumption
after failure requires **fresh authorization** and new paths.

---

## Countersignature

Execution is authorized only when this block is filled in, committed, and pushed.
**Unsigned, this card authorizes nothing.**

```
authorizer          : ____________________
timestamp (UTC)     : ____________________
authorization basis : ____________________   # the reviewed commit this signature approves
execution commit    : the commit containing this completed countersignature block
approved scope      : the exact two commands above, unmodified — every flag as
                      written, none added, none omitted
```

**Conditions:**

- Both runs execute from the **execution commit** with a clean worktree.
- Approval covers **one training run and one evaluation**. It does not extend to a
  re-run, a parameter change, an extra iteration, or a retry after failure.
- On signature, record the evaluation interval `[202608988, 202609388)` in the seed ledger
  as `RESERVED`, in the same commit.
- Changing any frozen parameter voids this signature. Amend and re-sign **before**
  running, never after seeing a result.
