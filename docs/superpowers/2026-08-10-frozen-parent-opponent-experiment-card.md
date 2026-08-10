# Frozen-Parent Opponent — Experiment Card

**Date:** 2026-08-10 · **Status:** DRAFT, not countersigned · **Scope: one training-data
mechanism, one training run, one 400-game evaluation. Nothing else.**

Successor to the closed ordinary-continuation family
(`docs/superpowers/2026-08-08-parent-replay-bootstrap-experiment-card.md`, rejected at
`−47.2` Elo; do-not-repeat **#49**). Changes **who the learner plays**, not how much it
trains.

## Hypothesis

Short continuations from a converged checkpoint degrade because the learner trains on its
own drifting play: as it weakens, the opposition weakens with it, and the errors compound.
Pinning the opponent to the **frozen best parent** holds opposition quality fixed, so the
learner's training signal stays anchored to a strong reference.

## Why this is a new mechanism, not a dose rescue

#49 closed ordinary continuation in any dose. This is not a dose change: it replaces the
**data-generating process**. Every prior run in the family — cold and warm — generated games
by self-play against the evolving learner. Here the opponent is a fixed checkpoint that
never updates, and only learner-to-move positions are trained on. That is the
"genuinely different mechanism" #49 requires, and it names a specific cause for the ~47 Elo
shortfall that warm-starting did not close.

**#49 does not prohibit this budget.** 200 games/iteration was selected to *preserve learner
exposure* under learner-only filtering — not to rescue ordinary continuation through dose.
The opponent mechanism is different; the budget follows from the filtering, and holds the
quantity that matters constant.

## Prediction, on record before the run

**Central forecast: material recovery relative to `warm5`, but no promotion.** Expect the
frozen endpoint to finish roughly equal to or slightly weaker than the parent, with an
aggregate point score around **`0.47–0.51`**. Estimated chance of clearing the promotion bar
is **about 10%**.

Holding opposition strength fixed directly addresses self-play co-drift, so improvement over
the descriptive `warm5` result of `0.4325` is plausible. However, the mechanism does not
change the terminal outcome targets, policy-dominated optimization, short five-iteration
horizon, or risk of specializing against one opponent. Therefore parity is more likely than a
statistically significant gain.

**This forecast is not a gate.** The aggregate decision rule and the frozen disposition remain
authoritative. **No per-colour direction is predicted.**

## The frozen budget — every choice pinned

| choice | value | why |
|---|---|---|
| parent warmup | **500 games, unchanged** | identical to `warm5`; the buffer starts warm |
| games per iteration | **200** | learner-only filtering halves usable rows; 200 mixed-agent games ≈ the ~9,000 learner rows `warm5` got from 100 self-play games |
| colour split | **exactly 100 learner-as-red, 100 learner-as-black** | fixed, not sampled |
| iterations | **5** | unchanged from the family |
| optimizer steps | **160/iteration**, batch **64**, buffer **100,000** | unchanged |
| simulations | **400 for both agents** | the opponent searches at full strength |
| game count | **fixed — no adaptive position target, no top-up** | an adaptive target would make the dose result-dependent |
| per-iteration reporting | **actual learner positions added** | the halving is an assumption; it must be measured, not asserted |

**Self-play cost roughly doubles.** 1,000 games instead of 500, at the same per-ply cost
(one 400-simulation search per ply either way). Estimated ≈ **3 h 35 m** training self-play
plus ≈ **1 h 12 m** warmup (both interpolated from measured `cont5`/`warm5` work), plus
≈ **3 h 07 m** evaluation ⇒ **≈ 8 h** end to end.

**That eight hours is a planning estimate, not a timeout.** Two inference servers may fragment
batches or contend for the GPU enough to make the real run slower than the single-network
measurements it is interpolated from. Exceeding the estimate is **not** a failure condition and
triggers **no automatic retry, no parameter change, and no abort** — only the frozen exit
conditions govern.

## Implementation — the dual-root seam

`play_game` currently threads **one** tree through the whole game: `root = MCTSNode(state)`
(`self_play.py:733`) is passed into `mcts.search_from_root` (`:869`), reassigned from its
return, consumed by `mcts.select_move` (`:1096`), and advanced by `mcts.advance_root`
(`:1176`). Two networks sharing that tree would blend priors produced by one with visits and
values produced by both. **A single search dispatch is insufficient.**

Required:

1. **Separate learner and parent roots.** Two trees, never shared.
2. **Route `search_from_root` *and* `select_move` through the active agent**, with
   `select_move` using that MCTS instance's own RNG.
3. **Advance both trees after every played move.** `mcts.advance_root` already creates a
   fresh node when the move was never explored (`mcts.py`), so the inactive tree needs no new
   logic — only a second call.
4. **Assert both roots stay synchronised to the same board state**, every ply.
5. **Game telemetry, resolved here rather than left to implementation:**
   - **Learner-only** for search-derived diagnostics and replay positions.
   - **Skip opening/root diagnostics on parent-controlled plies** — never read an unsearched
     or stale learner root there. The diagnostics at `:899`/`:933` read
     `root.priors_raw`/`root.priors`, which are network-specific.
   - **Sum learner and parent operational counters** — evaluations, backups, batches — and
     label them **combined**.
   - **Preserve active-agent move values** only where a complete game record requires them.
     Two consequences, decided and approved: the **closeout selection tie-break follows the
     active agent**, because it rewrites `visit_counts` before `select_move` and so changes
     the move played — learner-only routing would alter the parent's search behaviour; and
     **`gc_tracker` / `recovery_tracker` stay all-ply**, because they model whole-game
     trajectories and skipping alternating plies would corrupt their records. Their scores are
     therefore **active-agent / mixed-network telemetry** and must be read as such.
6. **Learner-only positions enter training**, filtered on `PositionRecord.to_move`
   (`:1037`, already explicit).

**Resign and adjudication are pinned OFF, and the mode refuses them.** Both are hard-wired to
the learner and are wrong on a parent ply: the resign check reads the learner root's
`visit_count` (`self_play.py` ~`:993`), which was never searched that ply, and timeout
adjudication searches with the learner MCTS (~`:1277`) whoever is to move. Frozen-opponent
mode therefore **raises** if either is enabled rather than silently mis-attributing.
Supporting them needs its own design and is out of scope. Both are `store_true` and absent
from the training command, so this costs the experiment nothing.

**Explicitly not built:** opponent pool, league, ratings, persistence, adaptive matchmaking,
checkpoint rotation, or any opponent-selection policy. One frozen opponent, fixed for the run.

**Feasibility: FEASIBLE, provisionally.** Estimated **≈190–290 production lines** across
`self_play.py`, `trainer.py`, `self_play_worker.py`, `train.py`; `inference_server.py` and
`remote_evaluator.py` need no changes, since a second *instance* of each serves the frozen
network. The estimate is tight against the ~300-line ceiling and the swing factor is the
per-ply telemetry binding, not the plumbing. **Stop and re-scope if implementation exceeds
roughly two working days or ~300 production lines.**

**`self_play.py` and `mcts.py` have been byte-identical to `d5326a0`** through the entire
competitive-readout line — verified. This change breaks that invariant for the first time.
The readout line is closed so nothing depends on it going forward, but the default-off path
must be provably inert and every future citation of that invariant needs a before/after
boundary.

## Pre-GPU gate — all must pass before any training run

1. **Default-off identity — deterministic behavioural equivalence**, not literal byte identity
   of the file, which editing necessarily breaks. With the flag absent or off, the run must
   produce the same games and the same training data as today, and must **not** construct a
   second root, evaluator or server, nor consume any extra RNG state.
2. **Dual-root seam tests**, covering at minimum: the two roots never alias; each agent's
   search and `select_move` use its own instance and RNG; both roots advance on every played
   move; the inactive tree creates a fresh child for an unexplored move; the roots' board
   states remain synchronised; and priors/visits from one network never appear in the other's
   tree.
3. **Learner-only filtering** verified — parent-to-move positions never enter the buffer.
4. **Colour split** verified — exactly 100/100 per iteration.
5. **Full suite green**, measured immediately before the run, with the new tests in it.
6. **New output paths absent**; **clean worktree**; **parent path and SHA-1 confirmed**.

## Parent and opponent

```
checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors
sha1  209cf2d4fd24a48553d259dd71b4954867b9473e
```

The same checkpoint is the **warmup generator**, the **frozen opponent**, and the
**evaluation baseline**. It never updates during the run.

## Command block 1 — training `[GPU, state-changing]`

```bash
nohup bash -c '.venv/bin/python -m scripts.GPU.alphazero.train \
  --load-weights checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --frozen-opponent-checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --checkpoint-dir checkpoints/alphazero-v2-fp5-from-calib020 \
  --games-dir logs/selfplay/fp5_from_calib020 \
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
  --seed 20260810 \
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
printf "%s\n" "$rc" > logs/eval/fp5_train.exit
exit "$rc"' \
  > logs/eval/fp5_train.stdout 2>&1 &
disown
```

Differences from `warm5`, and only these: `--frozen-opponent-checkpoint` (new, the same
path as `--load-weights`), `--games-per-iter 200` (was 100), `--seed 20260810`, and the new
output paths. Resign and adjudication remain absent, which the trainer now enforces at
startup rather than trusting.

## Command block 2 — provenance gate `[no GPU; writes ONE artifact]`

```bash
bash -c 'CK=checkpoints/alphazero-v2-fp5-from-calib020/model_iter_0005.safetensors
OUT=logs/eval/fp5_candidate_provenance.txt

[ -e "$OUT" ] && { echo "REFUSE: $OUT already exists"; exit 1; }
[ -f "$CK" ] || { echo "REFUSE: endpoint missing: $CK"; exit 1; }
[ "$(cat logs/eval/fp5_train.exit 2>/dev/null)" = "0" ] || { echo "REFUSE: training exit not 0"; exit 1; }
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
  --checkpoint-a checkpoints/alphazero-v2-fp5-from-calib020/model_iter_0005.safetensors \
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
  --output logs/eval/fp5_vs_calib020.json
rc=$?
printf "%s\n" "$rc" > logs/eval/fp5_vs_calib020.exit
exit "$rc"' \
  > logs/eval/fp5_vs_calib020.stdout 2>&1 &
disown
```

The evaluation is an **ordinary two-checkpoint match**; the frozen-opponent mechanism exists
only in training and has no part in it.

## Command block 4 — per-colour reporting `[read-only, DESCRIPTIVE ONLY]`

```bash
.venv/bin/python -c "
import json
from scripts.GPU.alphazero.eval_elo import score_ci_trinomial
s = json.load(open('logs/eval/fp5_vs_calib020.json'))
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

## Frozen endpoint

**Only `model_iter_0005.safetensors` is evaluated.** Iterations 1–4 are not evaluated,
probed, or selected from, under any result.

## Decision rule — aggregate only

**Success:** the candidate's overall **95% lower bound above 50%** in the 400-game match
against `calib020_0001`.

**There is no absolute per-colour veto.** The 2026-08-10 colour-rule erratum retired it: an
equal agent need not score `0.50` in each colour, and a colour null estimated from the same
match is confounded with candidate strength. Per-colour results are **reported and
descriptive**, never a gate, unless a future card preregisters an independently measured
colour baseline.

## Frozen disposition — decided before the result

| outcome | disposition |
|---|---|
| **Bar not met** | Close frozen-parent opposition. No opponent-pool successor, no league, no dose or warmup follow-up, no second opponent. |
| **Bar met** | **Do not promote yet.** Run the **0379 generalization match** (already supported by the existing harness) as a separate short authorization with its own seed interval. Beating the parent while failing 0379 means the learner exploited one opponent rather than becoming broadly stronger — the central risk of frozen-opponent training. |
| **Clear loss** | Close immediately. |

## Seeds

Evaluation interval **`[202609788, 202610188)`**, half-open, 400 wide — *proposed, reserved
only at countersignature*. Priors, all consumed: `[202608060, 202608124)`,
`[202608124, 202608188)`, `[202608188, 202608988)`, `[202608988, 202609388)`,
`[202609388, 202609788)`. **Disjointness is not code-enforced on this path** —
`eval_checkpoint_match` has no `--prior-seed-interval`. Training seed `20260810`, distinct
from the evaluation interval.

## Deliberate omissions

- **No external strength anchor.** Deferred by decision: it is worthwhile *if* this becomes an
  ongoing programme, not before another bounded mechanism test. #49's requirement is
  conditional, not a precondition.
- **No 64-game screen, no A/B/C/D, no iterations 1–4, no analyzer or telemetry framework, no
  learning-rate or dose grid.**
- **The product-model question is a separate workstream** — comparing the current best
  checkpoint against the served model or its confirmed source. The provenance audit excludes
  `calib020_0001` but does **not** establish that the served model is `model_iter_0193`, and
  its own frozen conclusion forbids replacing the model on that audit alone.

## A positive result is research evidence only

It does not promote a checkpoint. The 0379 generalization match precedes any promotion
discussion, and adoption is a separate decision with its own record.

---

## Countersignature

Execution is authorized only when this block is filled in, committed, and pushed.
**Unsigned, this card authorizes nothing.** The mechanism may be implemented before
signature; **no training or evaluation may run before it**, and the pre-GPU gate must pass.

```
authorizer          : ____________________
timestamp (UTC)     : ____________________
authorization basis : ____________________   # the reviewed commit this signature approves
execution commit    : the commit containing this completed countersignature block
approved scope      : the exact FOUR command blocks above, unmodified — every
                      flag as written, none added, none omitted:
                        1. training run              [GPU, state-changing]
                        2. provenance gate           [no GPU; writes ONE artifact]
                        3. evaluation match          [GPU, state-changing]
                        4. per-colour reporting      [read-only, descriptive only]
```

**Conditions:**

- Both runs execute from the **execution commit** with a clean worktree, and that commit
  contains the implemented mechanism with the full pre-GPU gate passing.
- Approval covers **one training run and one evaluation**. It does not extend to a re-run, a
  parameter change, an extra iteration, a different opponent, or a retry after failure.
- The **prediction section must be filled in before signature**, not after seeing anything.
- On signature, record `[202609788, 202610188)` as `RESERVED` in the same commit, in
  `docs/superpowers/2026-08-06-competitive-readout-seed-ledger.md`:

  ```
  | 6 | `[202609788, 202610188)` | 400 | Frozen-parent opponent — fp5 vs calib020 | 2026-08-10 | the commit containing this reservation | **RESERVED** |
  ```

- Changing any frozen parameter voids this signature. Amend and re-sign **before** running.
