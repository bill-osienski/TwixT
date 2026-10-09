# External-engine learning — bounded viability proposal

**2026-10-08 · PROPOSED, NOT AUTHORIZED.** Preparation at `cb2ae987`; this pass read
source/evidence and public repository metadata only. No runtime was installed, model
loaded, engine queried, seed registered/drawn, game generated, gate opened or code changed.

**Question:** can fresh twixtbot choices on our positions support supervised move
learning and one candidate worth further investigation? Confidence is low. This is a
new data/objective feasibility experiment, not a promotion or an ongoing programme.
`calib020_0001` stays incumbent; T1j remains evaluation-only. H4/H5's advantage under
their tested configuration is preserved. Tactical-pilot outcomes remain unread.
Every position uses physical and active board size **24**, automatic
links, no removal, forbidden own/opponent link crossings, and swap off.

## Viability findings

| Question | Evidence and decision |
|---|---|
| Available and permitted? | The [public UI repository](https://github.com/stevens68/twixtbot-ui) lists its bundled engine/model. Use historical commit `83749f230a0bae1766b46a05bfde0ed87f0a9a0a` and model hashes in `twixtbot_g3_schedule.ANCHOR`. Its recorded checkout/venv paths are gone. The [upstream MIT license](https://github.com/BonyJordan/twixtbot/blob/master/LICENSE) permits software use with notice retention; coverage of the exact redistributed model bundle is **not reverified** (pinned UI license fetches failed). Reacquire pinned bytes, inspect bundle-specific terms and retain notices before inference. Unresolved/restrictive permission stops this source. |
| Exact positions/rules? | Adapter: ordered reconstruction, red→WHITE, `(row,col)`→`(x=col,y=row)`, `Game(allow_scl=False)`. August bridge/state/cap qualification passed; G3 completed 128 games without abort. Recheck the installation and search-child rule propagation, especially own-link crossings. Historical runtime evidence does not certify learned rules. |
| Obtainable data? | Chosen moves, histories, authoritative winner/reason; at `trials>0`, visit arrays with `moves_order`. Zero trials returns raw policy, **no visits**. MVP trains on choices only; outcomes are provenance. No value estimates or visit/policy arrays enter training. |
| Smallest objective? | Legal-move cross-entropy against a one-hot **chosen-move label**, updating only the existing policy head. This label is not a measured search distribution. A fitted scalar-temperature control tests whether any held-out CE gain exceeds simple recalibration. |
| Budget? | One fixed teacher configuration; 12 data games / 60 minutes, one 128-step training run / 30 minutes, then 32 evaluation games / 120 minutes. These are enforced ceilings, not measured runtime promises. Separate acquisition/setup ceiling: 30 minutes. Total maximum: four hours, sequential execution. No extensions or parameter grid. |

The teacher network was trained with own-link crossings allowed. Generating fresh
labels on verified states avoids importing unreachable foreign-rule positions; it
does **not** prove the network understands our rules or gives correct moves. The
smoke can qualify runtime compatibility and test imitation usefulness, not remove
this limitation. It must be explicitly accepted for this narrow trial; otherwise
the concrete blocker is the absence of a rules-matched teacher.

R0 rejected independent data within its scope; R1 specifically left supervised move
prediction needing its own card. This proposal supplies that scope. Ledger #48/#49
ordinary continuation, #51 frozen-parent opposition, and calibration/retention
rescues stay closed. The incumbent supplies opponent moves only, never labels.
#52's saturated-anchor pilot is not rerun or retuned. #50's two independent Metal
servers remain prohibited.

## Three stages and stopping rules

**1. Teacher/data smoke — the only first trial proposed for approval.** Reacquire
the exact pinned engine/model and an isolated environment, using the recorded G3
dependency versions where available. Verify hashes, module origins, notices and
runtime before inference; no engine patches or substituted weights. Use one worker,
sequential teacher/reference calls, TensorFlow on CPU and one MLX Metal owner.
Fix `trials=100`, temperature/noise zero, rotation off, swap off, `allow_scl=False`.
This uses an already exercised search path; it is not a strength-setting search.

First check 12 fixed 24×24 reconstruction fixtures: edges, both colours' automatic
links and wins, own/opponent blocked crossings, both knight-link slopes, shared
endpoints, and external cap boundaries. Reuse the qualified fixtures and independent
intersection oracle; add the missing own-crossing controls. Check that search Game
copies preserve the rule flag. Deliberate bridge/side/ply corruption must be detected.
Query three nonterminal fixtures twice through fresh Players/Games in two fresh
processes: **six queries total**, identical chosen moves required. No full G2 rerun.

Generate six fresh four-ply opening families using registered opening RNG seeds,
then play each against unchanged incumbent research settings in both colour arms:
**12 games**, cap 280. Freeze family assignment **before generation**: first four
families train (8 games), last two held out (4). Both colour arms and all derivatives
stay in their family. No old G3/T1j/tactical games enter the corpus. Opening and
incumbent moves are never supervised targets. Check state equivalence after **each
opening ply**, each game ply, and reconstruction of every exported teacher row.

Export create-only JSONL: schema/target type, game/family/split, ordered prefix,
ply/side, rule/config and runtime hashes, full ordered legal moves plus digest,
chosen move/index, normalized peg/bridge state digest, terminal reason/outcome.
Reload in a separate process; require identical states, legal ordering and one-hot
labels. Keep `draw_ply_cap` distinct from a natural result. Require at least **96
training and 32 held-out teacher rows**, with no exact canonical position shared
across splits. Archive all games, including partial failures, before compiling labels.
G3's exact `trials=100`/incumbent subset contains 251 teacher rows in 16 games:
12–20 per game (mean 15.6875), games 27–44 plies. Eight training games therefore
project to about 126 rows; 96 is the historical minimum-per-game projection, **not
a guaranteed yield** on fresh openings. Keep the 12-game budget unchanged.

Before requesting stage 2, reuse the already loaded incumbent to cache full legal
raw logits on **training rows only**. Persist them as a create-only, SHA-256-pinned
stage-1 artifact bound to the training-corpus digest, row IDs, ordered legal-move
digests, teacher indices and verified incumbent identity; compute `f` from its
reloaded contents. Stage 2 consumes that artifact, verifies those bindings against
the frozen training rows/incumbent and refuses any mismatch; it does not recompute
or silently replace the cache. Report teacher/raw-top-1 match and disagreement
counts/rates overall, per game and by colour, using the same canonicalization and
legal masking as training. Let `f` be the fraction of training rows whose teacher
move differs from incumbent raw top-1; exact ties choose the first maximum in the
recorded legal-move order, and a teacher move at another tied index counts as a
disagreement. **If `f=0`, stop before training**: this corpus
offers no demonstrated opportunity to change chosen moves toward the teacher.
A positive `f` establishes target disagreement, not that this head can fit it;
stage 3 must show improved held-out choices. Do not inspect held-out logits here.
This check fits inside stage 1's 60-minute ceiling and needs no extra model load or
historical-G3 inference authorization. G3 remains out of every learning split.

**Stop** before queries on unresolved permission, missing pinned bytes/dependencies
or failed fixtures. Stop the whole stage on nondeterministic choices, illegal output,
swap/resign, engine exception, any state/search-rule divergence, timeout, export/load
failure, split leakage or insufficient rows. No dropping games, replacement openings,
resampling or retry. A pass means usable examples under the recorded limitations,
not teacher optimality or stronger play.

**2. Training smoke — separate approval after stage 1.** Load a verified copy of
incumbent SHA-1 `209cf2d4fd24a48553d259dd71b4954867b9473e`. One fresh training RNG,
**128 steps, batch 32, Adam LR 0.001, gradient clipping 1.0, no L2/weight decay**.
Uniform sampling with replacement from the frozen training split; no augmentation,
self-play, warmup, calibration, retention or auxiliary losses. Save only the final
candidate for evaluation; optimizer/log diagnostics may be retained for failure
analysis. Deadline 30 minutes including the scalar control and save/reload; no
intermediate selection.

Update only `policy_head.{conv,bn,fc,out}.{weight,bias}`. Encoder, value head/adapter
and every BN running statistic stay byte-identical. Compute gradients in eval mode
using incumbent BN statistics. Raw values on identical positions stay unchanged;
searched values can change.

At hidden width 128 this is **519 trainable parameters**: conv 258, BN affine 4,
fc 192, out 65, behind a two-channel per-cell bottleneck. This is deliberately a
limited feasibility surface; a null does not reject external imitation in general.

Fit a **single inverse-temperature scalar** `beta` to the cached training logits:
scale first, then call `compute_masked_log_probs(beta * logits, move_mask)` to
re-mask padding **after scaling**, including at `beta=0`; minimize mean chosen-move CE. Use the
convex scalar objective, initialized at `beta=1`, bounded to `[0,100]`, with at most
64 derivative-bisection iterations and bracket tolerance `1e-6`. An endpoint is
accepted only when its one-sided derivative establishes the constrained minimum;
if CE is still decreasing at 100, stop because the control is inadequately bounded.
`beta=0` is the uniform-policy limit. No grid, teacher queries, held-out fitting or
extra checkpoint is involved. Freeze the fitted scalar before stage 3; retain
its training objective, derivative and convergence record. Failed/nonfinite fitting
stops before candidate optimization. It is a CE comparator, not a playing engine.

Reuse `PositionRecord`, `make_padded_batch(max_moves_cap=576)`, canonicalized
`network.forward_padded(active_size=24)` and `compute_masked_log_probs` from the
existing trainer/network. Map the selected coordinate into **our full legal order**;
put 1 there and 0 elsewhere in the legacy `visit_counts` field only in the in-memory
adapter, explicitly tagged `teacher_chosen_move`. The normalized tensor becomes the
existing CE target. Ignore the batching helper's outcome tensor (its required zero
placeholder is not a value label), discard the value output, and differentiate only
mean `-sum(target * masked_log_probs)`. An isolated offline step updates the live
policy head only. The ordinary `train()` loop and its two-optimizer `train_step()`
are unsuitable here: zero value weight alone does not skip the value optimizer.
This is a small wrapper around existing primitives, not a trainer rewrite.

**Stop** on nonfinite tensors/gradients, any forbidden parameter/statistic change,
missing chosen target, partial step budget, save/reload failure, or no measurable
policy change. Loss reduction is a mechanics observation, never strength evidence.

**3. Preliminary evaluation — separate approval after stage 2.** First evaluate
incumbent, its frozen scalar-temperature control, and final candidate once on the
sealed held-out games. Report per-game mean chosen-move CE (all three), raw top-1
agreement and teacher rank (incumbent/candidate), and move disagreements, with
colour summaries descriptive. Report game-level paired differences; rows within
a game are not independent samples. Require candidate game-mean CE strictly below
**both** incumbent and scalar control, and candidate game-mean top-1 agreement
strictly above incumbent. Otherwise stop **before spending evaluation-game compute**.
Thus flattening alone cannot pass. Four held-out games still cannot establish broad
generalization, and beating this control excludes only a global temperature explanation.

Run **32 fresh candidate-vs-incumbent games**, 16 balanced colour pairs with distinct
audited game seeds, from the existing evaluation runner's empty-board/stochastic
opening path. Both use unchanged research settings: 400 simulations, batch 14,
stall flush 48, opening temperature 20 plies at 1.0 then 0.1, cap 280, no search
noise/resign/adjudication, one worker. No teacher or T1j evaluation is needed.
Use sequential dual-checkpoint evaluation; no independent inference-server threads.

Report W/L/draw/cap counts and the mean pair score, with conservative 95% Hoeffding
interval treating 16 independent pair means in [0,1] as units (unclipped half-width
**0.3395**). Independence is a protocol assumption, not proved by fresh seeds.
No Elo or absolute per-colour 0.50 veto. This is a **disaster screen only**:
the wholly-below-parity veto fires only at mean pair score below about **0.1605**.
Passing excludes little; the full 120-minute ceiling buys no modest-gain resolution.

**Stop** on integrity failures/deadline; incomplete cohorts have no strength rate.
If more than half the games are caps, strength is uninterpretable. A completed run
is worth further consideration only if the held-out criteria above pass and the
aggregate score interval does not lie wholly below 0.50. Passing is **provisional
viability**, not promotion, noninferiority, or proof of improved strength. Every other result ends
this trial; no rescue dose, checkpoint shopping or enlarged match.

## Smallest implementation and approval boundary

1. After approval, add one prototype data module/CLI and a dedicated stage gate,
   created closed and included in the source-derived gate inventory. Adding it makes
   28 gates: update `EXPECTED_GATES`, the pinned count, the word scanner's
   `"twenty-eight": 28` entry and its non-vacuity controls together. This proposal
   changes none of those source files. Reuse adapter construction/comparison,
   harness move capture/play, pinned runtime verifier,
   reconstruction and seeded reference agent. Add only per-opening-ply checks and
   export/load glue locally; preserve historical runners/cards/evidence.
2. Focus tests on illegal/crossing-state rejection, export/reload, split enforcement,
   black canonicalization and a chosen move beyond slot 512. Add focused checks that
   `f=0` refuses training and that pure temperature rescaling cannot pass the held-out
   comparison; verify the scalar fit's derivative/convergence on known optima.
   The padded known-optima fixture must assert legal probability `1/n_legal` and
   zero padding mass at `beta=0`, plus no padding leakage at a small positive beta;
   verify cache/row binding rejects a changed logit artifact or reordered legal moves.
   Negative controls must fail. Before later training, test actual frozen-surface
   updates/save/reload; before evaluation, verify incumbent/config identity and
   fresh-seed binding. Run affected
   regression and gate-inventory checks with gates closed; no exhaustive mutation work.
3. Use a fresh create-only run directory under `runs/external-engine-learning/` and
   a separate evidence directory. Flush partial records, preserve stdout/stderr,
   environment, exit status, manifests/hashes and failed checkpoints. Never overwrite
   incumbent checkpoints or historic artifacts; nothing is pushed.

**Minimum approval now:** authorize preparation and **stage 1 only**, including the
pinned runtime/model/dependency acquisition (30-minute setup ceiling), acceptance of
the documented learned-rules limitation conditional on exact runtime checks, six
teacher queries, 12 games and the training-row raw-logit disagreement check
(60-minute execution ceiling), and seed preparation.
Freeze the implementation/schedule at a reviewed commit before execution. Seed
preparation uses the existing union survey/collision-proof method over scripts,
tests and `docs/superpowers` (including compressed/failed/untracked evidence), all
registry/spent/reserved ranges and derived streams. Stage 1 needs **18 seed slots**:
six opening draws and twelve game seeds. The six deterministic teacher queries
have fixture/query IDs, input hashes and process records, **no seed slots**. Neither
they nor the deterministic raw-logit check draw experimental RNG streams.
Exact integers are not chosen, reserved or registered in this proposal. A dirty
survey stops; do not hunt another range. Survey and registration/opening occur in
the repository's reviewed order, with pre/post proofs and actual exposure retirement.

Open only the new stage-1 gate for its approved run, restore it on every exit, and
verify all unrelated gates remain closed. Approval does not reopen historical gates,
authorize training/evaluation, or waive promotion rules. The later comparison needs
explicit approval as this separately scoped preliminary screen; it is not an earned
historical promotion match. Stages 2/3 require their own fixed manifests and fresh
seed accounting before approval. No work beyond the proposed first trial follows
automatically from a pass.

Evidence read: R0/R1 memos; ledger closure, four-gate promotion rule and relevant
do-not-repeat entries; twixtbot pilot card; adapter qualification **attempt 3**;
G3 environment/dependency record; current adapter/harness/runtime verifier,
trainer/network, reconstruction, evaluation, seed and gate-inventory sources.
