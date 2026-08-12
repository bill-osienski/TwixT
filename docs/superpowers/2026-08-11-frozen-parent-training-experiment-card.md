# Frozen-Parent Training — Experiment Card

**Date:** 2026-08-11 · **Status: COMPLETE / BAR NOT MET / AUTHORIZATION EXHAUSTED.**
**Verdict: BAR NOT MET — PARITY NOT RESOLVED — FROZEN-PARENT LINE CLOSED.**

> ## Closeout (2026-08-12)
>
> Executed once from execution commit **`13dd72f6261f60e5256f25af5ce1c851dbd821cf`**, clean
> worktree. Both exit files `0`; the match JSON records the same `git_commit`.
>
> ### Aggregate result — the decision
>
> | | |
> |---|---:|
> | record | 184–211, 5 state caps ⇒ 186.5 / 400 |
> | **score rate** | **`0.46625`**, CI95 **`[0.4177, 0.5148]`** |
> | primary decision | lower bound `0.4177` is not above `0.50` — **BAR NOT MET** |
> | Elo | **`−23.5`**, CI95 `[−57.7, +10.3]` |
>
> **The candidate was NOT shown stronger. It was also NOT shown weaker at this dose:** both the
> score interval and the Elo interval **include parity**. The candidate is **not statistically
> distinguishable from the parent** — which is not the same as equal, and must not be written
> that way.
>
> Unlike `cont5` and `warm5`, this is **not a decisive rejection**. It is a non-pass with the
> question left open.
>
> ### Prediction, assessed
>
> The preregistered forecast was an aggregate around **`0.47–0.51`**, roughly equal to or
> slightly weaker than the parent, with about a **10%** chance of clearing the bar. Observed
> `0.46625` sits **slightly below the band's `0.47` lower edge**, so the prediction matched
> **approximately in direction and magnitude — not precisely.**
>
> **A single non-pass does not validate the 10% probability.** One draw cannot confirm or refute
> a small stated probability; the figure remains unexamined as a calibration claim.
>
> ### Where it sits in the line — descriptive only
>
> | run | mechanism | score rate | Elo |
> |---|---|---:|---:|
> | `cont5` | cold-buffer continuation | 0.31375 | −136.0 |
> | `warm5` | parent-bootstrap continuation | 0.4325 | −47.2 |
> | `fp6` | **frozen parent** | **0.46625** | **−23.5** |
>
> **This progression is DESCRIPTIVE across three separate runs** — different mechanisms,
> different seeds, different evaluation intervals — **not a paired causal estimate.** No
> confidence interval on any difference between them exists or was computed.
>
> ### Per-colour — reported, decides nothing
>
> | | record | rate | CI95 |
> |---|---|---:|---|
> | as red | 102–94, caps 4 | 0.5200 | `[0.4515, 0.5885]` |
> | as black | 82–117, caps 1 | 0.4125 | `[0.3444, 0.4806]` |
>
> `red_win_rate_decisive` **0.5544**. Per the 2026-08-10 erratum the absolute per-colour veto is
> retired: with red winning 55.4% of decisive games, this split is confounded with the board's
> colour advantage and the same-match null cannot separate it from candidate strength.
>
> ### Dose, actually measured
>
> Warmup **46,523 positions in 5,481.6 s**. Learner rows per iteration: **9,611 · 9,085 · 9,382
> · 9,858 · 9,701** — confirming the assumption that pinned 200 mixed-agent games to reproduce
> `warm5`'s ~9,000 learner rows from 100 self-play games. Final buffer 94,160 of 100,000, no
> eviction. Secondary: 5/400 state caps (1.25%), 0 board-full, 395 decisive, avg 61.78 plies.
>
> ### Timings, and one anomaly
>
> Training **5 h 59 m** (estimate 8 h 21 m); evaluation **3 h 08 m**. Iteration timings
> **8,114.9 · 1,703.6 · 1,788.0 · 2,324.4 · 2,142.5 s**.
>
> **Iteration 1 took roughly 4× the others and is unexplained. It is recorded as an
> INFRASTRUCTURE OBSERVATION, not scientific evidence** — it says nothing about the checkpoint.
>
> ### Provenance and artifacts
>
> Candidate `checkpoints/alphazero-v2-fp6-from-calib020/model_iter_0005.safetensors`; parent
> `209cf2d4fd24a48553d259dd71b4954867b9473e`; `train_head` and eval `git_commit` both
> `13dd72f6261f60e5256f25af5ce1c851dbd821cf`. Iterations 1–4 were never evaluated, probed or
> inspected.
>
> ```
> candidate            22f8d2196140aff5b04fac0b68e1e5fa955d5ad4
> fp6_vs_calib020.json 544d6e335a773a3ac0e410d3fda7950e04c545dc
> games.jsonl          c6c4b76ccbd3005072c5fd512578fec939ac22ae
> eval stdout          6dd40a97c312d224b829e5ea33369b6c65deea20
> eval exit            09d2af8dd22201dd8d48e5dcfcaed281ff9422c7
> provenance           f58b49db67d0bcfa079bc433298a6ce99b6e7c23
> ```
>
> ### Consequence
>
> Per the frozen disposition, bar-not-met **closes frozen-opponent training**: no dose change, no
> warmup change, no second opponent, no opponent pool, no extension, no iteration 1–4 inspection,
> no larger match, and **no 0379 generalization match** — that was conditional on a bar-met
> result. **Keep `calib020_0001`.** See do-not-repeat **#51**.
>
> **`#50` is unchanged and vindicated at scale:** the single arbiter served ~1,500 games at 400
> simulations without a driver abort. The two-server prohibition stands.

**Scope as authorized (historical): one five-iteration
training run and one 400-game evaluation. Nothing else.**

**Prior:** `cf76982f` — the closed acceptance card, which accepted the frozen-opponent
implementation at `95987ffa` as the enabled baseline and explicitly left its **use**
unauthorized. This card authorizes that use, once.

**Where this sits, precisely.** `fp6` is the **fourth training execution in the broader
continuation line** and the **second execution attempt of the frozen-parent mechanism**:

| run | mechanism | outcome |
|---|---|---|
| `cont5` | ordinary continuation, cold buffer | **rejected** on strength (`0.31375`, `−136` Elo) |
| `warm5` | ordinary continuation, 500-game parent bootstrap | **rejected** on strength (`0.4325`, `−47.2` Elo) |
| `fp5` | **frozen parent** | **infrastructure abort** before iteration 1 — no scientific result |
| `fp6` | **frozen parent** | pending |

The first two tested **ordinary continuation**, a different mechanism, and are closed as a
family (`#49`). Only `fp5` attempted the frozen-parent hypothesis, and it never generated a
training game. **That hypothesis remains untested.**

## Hypothesis

Short continuations degrade because the learner trains on its own drifting play: as it weakens,
so does its opposition, and errors compound. Pinning the opponent to the **frozen best parent**
holds opposition quality fixed, so the training signal stays anchored to a strong reference.

## Prediction, on record — PRESERVED, not restated

**Carried verbatim from the aborted card, where it was recorded before any implementation
existed and has never been examined:**

> **Central forecast: material recovery relative to `warm5`, but no promotion.** Expect the
> frozen endpoint to finish roughly equal to or slightly weaker than the parent, with an
> aggregate point score around **`0.47–0.51`**. Estimated chance of clearing the promotion bar
> is **about 10%**.
>
> Holding opposition strength fixed directly addresses self-play co-drift, so improvement over
> the descriptive `warm5` result of `0.4325` is plausible. However, the mechanism does not
> change the terminal outcome targets, policy-dominated optimization, short five-iteration
> horizon, or risk of specializing against one opponent. Therefore parity is more likely than a
> statistically significant gain.

**It must not be softened, re-derived or re-anchored after any result is seen.** Its value is
that it predates the implementation entirely.

## The frozen budget — unchanged from the aborted card

| choice | value |
|---|---|
| parent warmup | **500 games**, ordinary single-network path |
| games per iteration | **200** (learner-only filtering halves usable rows) |
| colour split | exactly **100 learner-as-red, 100 learner-as-black**, by game id |
| iterations | **5** |
| optimizer | **160 steps**/iteration, batch **64**, buffer **100,000** |
| simulations | **400**, both agents |
| game count | **fixed** — no adaptive target, no top-up |
| per-iteration reporting | actual learner positions added |

**Nothing about the dose changes.** Only the paths, the seed and the interval differ from the
aborted attempt — because those are spent, not because the recipe is being retuned. Retuning
after an infrastructure abort would silently convert a repeat into a new experiment.

## New paths and seeds — the old ones are spent

| item | value |
|---|---|
| checkpoint dir | `checkpoints/alphazero-v2-fp6-from-calib020` |
| games dir | `logs/selfplay/fp6_from_calib020` |
| training logs | `logs/eval/fp6_train.{stdout,exit}` |
| provenance | `logs/eval/fp6_candidate_provenance.txt` |
| evaluation | `logs/eval/fp6_vs_calib020.{json,stdout,exit}` + `_games.jsonl` |
| training seed | **`20260813`** — zero pre-existing occurrences (see below) |
| evaluation interval | **`[202609788, 202610188)`** — interval 6, **RELEASED UNUSED**, to be **explicitly re-reserved** at countersignature |

**Why not `20260812`.** It was the composition smoke's `MASTER_SEED`, passed as `master_rng`
to `run_parallel_selfplay` — the same role a training seed plays — and that smoke generated four
**real** frozen-opponent games. Reusing it could overlap worker and per-game RNG streams. It is
spent.

**`20260813` had ZERO pre-existing exact-token occurrences** across `scripts/`, `docs/` and
`tests/` **before this card was written**. Its **only current occurrences are inside this
unsigned card** — four of them, all in this file, none anywhere else in the repository. Stated
this way because the earlier phrasing ("zero occurrences repo-wide") was falsified the moment
the card containing the seed was committed; the evidence that matters is that nothing
*executable* and no prior authorization ever used it. Re-verifiable later with
`grep -rEow 20260813 scripts/ docs/ tests/`, which should show occurrences confined to this
card and its successors in the record.

Interval 6 drew **zero** seeds: the aborted run died before the evaluation started, and
`eval_checkpoint_match` is its only consumer. Re-reserving it is legitimate, and the ledger row
is **replaced in full** — label, date, commit and status — per the exact row in the
countersignature conditions. A status-only flip would leave the stale `fp5` provenance in place.

## Parent

```
checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors
sha1  209cf2d4fd24a48553d259dd71b4954867b9473e
```

Learner start, frozen opponent, and evaluation baseline are all this checkpoint.

## Frozen endpoint

**Only `checkpoints/alphazero-v2-fp6-from-calib020/model_iter_0005.safetensors` is evaluated.**
Iterations 1–4 are never evaluated, probed or selected from, under any result.

## Decision rule — aggregate only

**Success:** the candidate's overall **95% lower bound above 50%** over 400 games against
`calib020_0001`.

**No per-colour veto.** The 2026-08-10 erratum retired the absolute `0.50` per-colour rule:
equal agents need not score `0.50` in each colour, and a colour null estimated from the same
match is confounded with candidate strength. Per-colour figures are **reported and descriptive**.

## Frozen disposition

| outcome | disposition |
|---|---|
| **Bar not met** | Close frozen-opponent training. No dose change, no warmup-size change, no second opponent, no opponent pool, no extension. |
| **Bar met** | **Do not promote.** Run the **0379 generalization match** under its own authorization: beating the parent while failing 0379 means the learner exploited one opponent rather than becoming broadly stronger — the central risk of frozen-opponent training. |
| **Clear loss** | Close immediately. |

## Infrastructure risk, stated honestly

The aborted attempt died at the first line of iteration 1. The arbiter and composition smokes
(`9c5847a0`, `d312caa2`) now cover that failure mode, but **at doses orders of magnitude below
this run**: 402 synthetic calls and 4 games at 32 simulations, versus ~1,500 games at 400
simulations here. Sustained contention, long-run memory behaviour and throughput cost are
**unprobed**.

**A second infrastructure abort is a real possibility and is not a scientific result.** If it
happens, it is recorded as such — paths and training seed spent, interval consumed only if the
evaluation actually began — and the frozen-parent hypothesis remains untested after **two**
execution attempts.

## Cost

Warmup ≈ **1 h 39 m** — **measured on the aborted `fp5` run's identical 500-game warmup at
5,963.5 s**, not the older `warm5` figure of 1 h 12 m. Training self-play ≈ **3 h 35 m**
interpolated from measured like-for-like work. Evaluation ≈ **3 h 07 m** measured.

**Unadjusted total ≈ 8 h 21 m, before unknown arbiter overhead** — per-model batching may reduce
effective batch size under balanced demand, and that cost has never been measured. A planning
estimate, **not a timeout**: exceeding it triggers no retry, no parameter change and no abort.

## Command block 1 — training `[GPU, state-changing]`

```bash
nohup bash -c '.venv/bin/python -m scripts.GPU.alphazero.train \
  --load-weights checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --frozen-opponent-checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --checkpoint-dir checkpoints/alphazero-v2-fp6-from-calib020 \
  --games-dir logs/selfplay/fp6_from_calib020 \
  --replay-warmup-games 500 \
  --iterations 5 \
  --games-per-iter 200 \
  --train-steps 160 \
  --simulations 400 \
  --lr 0.0003 \
  --l2 0.0001 \
  --batch-size 64 \
  --buffer-size 100000 \
  --curriculum-sizes 24 \
  --hidden 128 \
  --blocks 6 \
  --seed 20260813 \
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
  --progress-weight-floor 0.25 \
  --probes-inline-disable
rc=$?
printf "%s\n" "$rc" > logs/eval/fp6_train.exit
exit "$rc"' \
  > logs/eval/fp6_train.stdout 2>&1 &
disown
```

Differences from `warm5`, and only these: `--frozen-opponent-checkpoint` (new, the same
path as `--load-weights`), `--games-per-iter 200` (was 100), `--seed 20260813`, and the new
output paths. Resign and adjudication remain absent, which the trainer now enforces at
startup rather than trusting.

## Command block 2 — provenance gate `[no GPU; writes ONE artifact]`

```bash
bash -c 'CK=checkpoints/alphazero-v2-fp6-from-calib020/model_iter_0005.safetensors
OUT=logs/eval/fp6_candidate_provenance.txt

[ -e "$OUT" ] && { echo "REFUSE: $OUT already exists"; exit 1; }
[ -f "$CK" ] || { echo "REFUSE: endpoint missing: $CK"; exit 1; }
[ "$(cat logs/eval/fp6_train.exit 2>/dev/null)" = "0" ] || { echo "REFUSE: training exit not 0"; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "REFUSE: worktree dirty"; exit 1; }

HEAD=$(git rev-parse HEAD) || { echo "REFUSE: git rev-parse failed"; exit 1; }
SHA=$(shasum -a 1 "$CK" | cut -d" " -f1) || { echo "REFUSE: hashing failed"; exit 1; }
[ ${#HEAD} -eq 40 ] || { echo "REFUSE: HEAD not 40 chars: [$HEAD]"; exit 1; }
[ ${#SHA} -eq 40 ] || { echo "REFUSE: sha1 not 40 chars: [$SHA]"; exit 1; }

printf "candidate: %s\nsha1: %s\ntrain_head: %s\nworktree_clean: true\n" \
  "$CK" "$SHA" "$HEAD" > "$OUT" || { echo "REFUSE: write failed"; exit 1; }
cat "$OUT"'
```

## Command block 3 — evaluation `[GPU, state-changing]`

```bash
nohup bash -c '.venv/bin/python -m scripts.GPU.alphazero.eval_checkpoint_match \
  --checkpoint-a checkpoints/alphazero-v2-fp6-from-calib020/model_iter_0005.safetensors \
  --checkpoint-b checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --games 400 \
  --base-seed 202609788 \
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
  --output logs/eval/fp6_vs_calib020.json
rc=$?
printf "%s\n" "$rc" > logs/eval/fp6_vs_calib020.exit
exit "$rc"' \
  > logs/eval/fp6_vs_calib020.stdout 2>&1 &
disown
```

The evaluation is an **ordinary two-checkpoint match**; the frozen-opponent mechanism exists
only in training and has no part in it.

## Command block 4 — per-colour reporting `[read-only, DESCRIPTIVE ONLY]`

```bash
.venv/bin/python -c "
import json
from scripts.GPU.alphazero.eval_elo import score_ci_trinomial
s = json.load(open('logs/eval/fp6_vs_calib020.json'))
print(f\"red_win_rate_decisive: {s['color_bias']['red_win_rate_decisive']:.4f}\")
for k in ('a_as_red','a_as_black'):
    d = s[k]
    lo, hi = score_ci_trinomial(d['wins'], d['caps'], d['losses'])
    print(f\"{k}: {d['wins']}-{d['losses']} caps={d['caps']} rate={d['score_rate']:.4f} \"
          f\"CI95 [{lo:.4f}, {hi:.4f}]\")
"
```

**No `UPPER<0.50` column and no veto.** Per the 2026-08-10 erratum the absolute per-colour
rule is retired; these numbers are reported for the record and decide nothing. The decisive
red win rate is printed alongside precisely so the split is not read as candidate-specific.

## Pre-GPU gate — every item measured and recorded before block 1

1. **Full suite green**, measured immediately before the run, from the execution commit.
2. **Parent path and SHA-1 confirmed** — `209cf2d4fd24a48553d259dd71b4954867b9473e`.
3. **All nine `fp6` output paths absent.**
4. **Clean worktree**, HEAD equal to the execution commit and to upstream.
5. **Accepted implementation present** — the flag enabled at `95987ffa`, acceptance closed at
   `cf76982f`, and no startup refusal in `train()`.
6. **Ledger row 6 replaced in full and reading `RESERVED`** for `fp6`, in the countersigning
   commit.

Any item failing is a stop. Block 1 does not launch until all six are recorded.

## Deliberate omissions

- **No 64-game screen**, no A/B/C/D, no iterations 1–4, no checkpoint sweep, no analyzer.
- **No external strength anchor** — still conditional on this becoming an ongoing programme.

## A positive result is research evidence only

It promotes no checkpoint. The 0379 generalization match precedes any promotion discussion, and
adoption is a separate decision with its own record.

---

## Countersignature

**Unsigned, this card authorizes nothing.**

```
authorizer          : bill-osienski
timestamp (UTC)     : 2026-08-11T20:30:53Z
authorization basis : a70ce7ff586b458ba20647817e97734d8b72fa7f   # the reviewed commit this signature approves
execution commit    : the commit containing this completed countersignature block
approved scope      : the exact FOUR command blocks (to be written into this card
                      before signature), unmodified:
                        1. training run              [GPU, state-changing]
                        2. provenance gate           [no GPU; writes ONE artifact]
                        3. evaluation match          [GPU, state-changing]
                        4. per-colour reporting      [read-only, descriptive]
```

**Conditions:**

- Both runs execute from the **execution commit** with a clean worktree.
- On signature, replace ledger row 6 **entirely** — label, date, commit and status, not just the
  status cell — **in the same commit**, using this exact row:

  ```
  | 6 | `[202609788, 202610188)` | 400 | Frozen-parent opponent — fp6 vs calib020 | 2026-08-11 | the commit containing this reservation | **RESERVED** |
  ```

  The stale `fp5` label, its `2026-08-10` date and its `fbe37f3` commit must not survive the
  re-reservation. **The historical release note below the table is retained**, so the record
  still shows the interval was reserved for `fp5`, released unused after the abort, and
  re-reserved here.
- Approval covers **one** training run and **one** evaluation. No re-run, parameter change,
  extra iteration, or retry after failure — including after an infrastructure abort.
- The prediction above is frozen. Amend and re-sign **before** running, never after a result.
