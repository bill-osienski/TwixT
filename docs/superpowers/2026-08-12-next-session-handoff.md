# Next-Session Handoff — Research Closed, Product Alignment Open

**Recorded:** 2026-08-12
**Repository:** `/Users/bill/projects/TwixT_Game`
**Branch:** `codex/competitive-readout`
**Durable research closeout at handoff:** `b931a476817f090a4a94f71d2e426e990f4b0742`
**Status:** no experiment is running or authorized; all experimental lines are closed.

This document is the compact durable memory for the next planning session. It does not
replace the experiment ledger. If a statement here and the ledger ever differ, the ledger
and the countersigned experiment cards control.

## Overall goal

Improve the TwixT AI's **actual playing strength and user-facing quality**, with direct,
reproducible evidence. Promote a model or policy only when it clears a preregistered strength
bar without breaking established safety/guardrail evidence. Prefer small, decisive tests over
long research branches, and stop cleanly when a mechanism fails.

The decision standard is intentionally skeptical:

- distinguish evidence from interpretation;
- distinguish a genuinely new mechanism from a parameter rescue;
- distinguish scientific learning from an actual strength gain;
- do not increase sample size or alter a dose after seeing an unfavorable result;
- do not agree with a proposal merely because it is plausible or already written down.

## Current best and current decision

Keep `calib020_0001` as the best-supported checkpoint:

- path: `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`
- SHA-1: `209cf2d4fd24a48553d259dd71b4954867b9473e`
- durable strength context: it beat staged `0379` by roughly `+80 Elo` in both recorded
  match directions.

No tested successor met its promotion bar. The calibration, search-reliability, competitive
readout, ordinary-continuation and frozen-parent research lines are closed. This does **not**
prove that `calib020_0001` is globally optimal or that a fundamentally new architecture,
objective or data source could never improve it. It does mean the tested families and their
nearby rescues are exhausted under the recorded evidence.

## Authoritative records to read first

1. `docs/alphazero-value-search-experiment-ledger.md` — authoritative programme history,
   final conclusions and do-not-repeat entries `#1–#51`.
2. `docs/superpowers/2026-08-06-competitive-readout-seed-ledger.md` — every reserved and
   consumed evaluation interval; intervals 1–6 are spent.
3. `docs/superpowers/2026-08-11-frozen-parent-training-experiment-card.md` — latest training
   hypothesis, frozen dose, authorization, execution and closeout.
4. `docs/superpowers/2026-08-11-frozen-opponent-acceptance-card.md` — accepted two-network
   infrastructure and its evidence chain.
5. `docs/superpowers/2026-08-06-model-path-provenance-audit.md` — product ONNX mismatch.
6. `logs/eval/current_best_and_candidates.md` — checkpoint/result digest when available.

## What the completed programme tried

The connected programme tested these hypothesis families:

- value-head calibration and A/B/C/D guardrails;
- partial-network updates, adapters, row/schedule changes and gradient projection;
- searched-continuation and depth-2 observables;
- c_puct, absolute FPU and two policy-mass FPU formulations;
- convergence/progressive-widening diagnostics;
- competitive final-move readout;
- ordinary self-play continuation from cold and parent-warmed replay;
- frozen-parent opposition, where the learner plays a fixed parent and trains only from
  learner-to-move positions.

The major durable findings are:

- The original A black-pre-drop issue was largely a `400`-simulation search artifact, not a
  stable value-head defect.
- Calibration candidates that improved A repeatedly harmed B/C/D or failed strength/safety
  gates. Nearby adapter, projection, schedule, margin and row-coverage rescues are closed.
- Absolute negative FPU reached the intended reply-scanning mechanism but caused unacceptable
  late-play collapse and policy disruption.
- Both policy-mass FPU families failed on independent evidence because control cost remained
  high; shrinking the coefficient did not make the mechanism cheap.
- The depth-2/provisional-backup observable had reach but not sufficient selectivity or a
  feasible fresh corpus.
- Progressive widening failed retention in the authoritative pilot; the broader tree-local
  heuristic line closed.
- Competitive readout produced a statistically unresolved result, not a promotable gain.
- Ordinary continuation clearly regressed; parent replay mitigated part of the observed gap
  but did not reach parity.
- Frozen-parent opposition improved the point estimate again, but still did not meet the bar
  and left parity unresolved.

Do not revive those branches with a renamed formula, relaxed gate, larger corpus, longer run,
extra match games, intermediate-checkpoint shopping, nearby coefficient grid or post-hoc replay
mining. Read do-not-repeat `#1–#51` before describing any future research idea as new.

## Strength results that anchor the conclusion

| line | result | durable interpretation |
|---|---|---|
| Competitive readout Candidate 2 | score `0.5125`, CI95 `[0.4779, 0.5471]`; `+8.7 Elo`, CI95 `[-15.3, +32.8]` | Promotion bar not met; neither shown stronger nor shown weaker. Readout line closed. |
| Cold ordinary continuation (`cont5`) | score `0.31375`, CI95 `[0.2687, 0.3588]`; `-136.0 Elo` | Clear regression; exact recipe rejected. |
| Parent-replay continuation (`warm5`) | score `0.4325`, CI95 `[0.3843, 0.4807]`; `-47.2 Elo` | Substantial descriptive recovery, still a clear aggregate loss. Ordinary continuation closed. |
| Frozen-parent opposition (`fp6`) | score `0.46625`, CI95 `[0.4177, 0.5148]`; `-23.5 Elo`, CI95 `[-57.7, +10.3]` | Bar not met and parity not resolved. Candidate was not shown stronger and was not shown weaker at this dose. Frozen-parent line closed. |

The progression `cont5 → warm5 → fp6` is descriptive across separate runs, not a paired causal
estimate. Do not describe the fp6 candidate as "equal" to its parent merely because the
confidence interval includes parity.

The historical absolute per-colour `0.50` veto was mis-specified because TwixT has a board-colour
advantage. Historical firings remain protocol facts, but do not establish candidate-specific
colour harm. Aggregate verdicts remain unchanged.

## The previous plan: frozen-parent opposition

### Hypothesis

Ordinary self-play continuation may let the learner and its data distribution co-drift. Playing
against a frozen `calib020_0001` parent, while retaining only learner-to-move positions, might
provide a stable target and preserve parent strength while allowing improvement.

The prediction was recorded before implementation: aggregate score `0.47–0.51`, roughly equal
to or slightly weaker than the parent, with about a `10%` chance of clearing the promotion bar.
It was preserved unchanged across the infrastructure work.

### First attempt (`fp5`)

The 500-game parent warmup completed, but iteration 1 aborted immediately with exit `134` and a
Metal assertion: two independent inference-server threads submitted concurrent GPU work to the
same command buffer. No training game was produced, so this was an infrastructure result and no
scientific result. The training seed `20260810` and all `fp5` paths are spent.

### Infrastructure repair

The two-server design was removed and replaced by one Metal-owning inference arbiter serving
both model identities. The work then passed:

- a one-thread/two-network feasibility probe;
- a real-device routing smoke using two different checkpoints and an independent oracle;
- a real-MCTS composition smoke with ragged batches and `38` mixed-model flushes;
- the full fp6 training workload without the original driver abort.

The two-server prohibition in do-not-repeat `#50` remains permanent. The repair satisfies it;
it does not repeal it.

### Second attempt (`fp6`)

The scientific dose was kept unchanged so the original prediction still applied:

- 500 parent warmup games;
- 5 iterations;
- 200 mixed-agent games per iteration, exactly 100 learner-as-red and 100 learner-as-black;
- 160 training steps per iteration, batch size 64, replay buffer 100,000;
- 400 MCTS simulations for learner and parent;
- learner-only training rows;
- training seed `20260813`;
- fixed endpoint `model_iter_0005`, with iterations 1–4 never inspected.

Training completed in `5 h 59 m`. Warmup produced `46,523` positions. Learner rows by iteration
were `9,611`, `9,085`, `9,382`, `9,858`, `9,701`; the final buffer held `94,160` rows with no
eviction. Iteration 1 took an anomalous `8,114.9 s`, recorded as an infrastructure observation,
not a parameter to tune. The endpoint SHA-1 was
`22f8d2196140aff5b04fac0b68e1e5fa955d5ad4`.

The authorized 400-game match used interval `[202609788, 202610188)` and returned:

- 184 wins, 211 losses and 5 state caps (`186.5/400`);
- score `0.46625`, CI95 `[0.4177, 0.5148]`;
- Elo `-23.5`, CI95 `[-57.7, +10.3]`;
- 395 decisive games and 1.25% state caps.

The promotion rule required the aggregate 95% lower bound to exceed `0.50`; it did not. The
prediction matched approximately in direction and magnitude—the point estimate fell just below
the forecast band—but one non-promotion does not validate the stated `10%` probability.

Frozen disposition: no larger match, dose change, second opponent, opponent pool, extension,
match against `0379` or inspection of iterations 1–4. The line is closed under do-not-repeat
`#51`.

## Infrastructure and process improvements that survived

The negative science did leave reusable, shipped capability:

- one Metal-owning inference arbiter serving two resident networks;
- model-addressed requests and distinct response queues keyed by `(worker_id, model_id)`;
- dual-root self-play with separate MCTS instances and synchronized trees;
- learner-only replay filtering and deterministic colour assignment by claimed game id;
- worker/server fail-loud behavior and pre-device/pre-filesystem startup validation;
- suite status at research acceptance: `2,881 passed / 4 skipped / 53 deselected / 0 failed`.

Durable review lessons:

- an oracle must not traverse the route it is meant to verify;
- patch the module that owns the symbol actually resolved at runtime;
- test safety properties behaviorally and construct negative cases;
- a stub transport test is not evidence about a device-level contract;
- search exact tokens before declaring a seed fresh;
- phrase provenance claims so the document introducing an identifier does not falsify them;
- bind authorizations, gates and artifacts to exact commits and hashes.

## Seed and authorization state

- Evaluation intervals 1–6 in the shared seed ledger are **CONSUMED**.
- Interval 6's full history is reserved for fp5, released unused because fp5 evaluation never
  started, re-reserved for fp6, then consumed by the fp6 match.
- Training seeds `20260810` and `20260813`, and all fp5/fp6 artifact paths, are spent.
- No experiment is running or authorized.
- Any future experiment needs a genuinely new hypothesis, fresh paths and seeds, its own card,
  countersignature and commit-bound gate. This handoff authorizes none of that.

## Only open workstream: product-model alignment

This is a product/deployment workstream, not another AlphaZero research line.

The read-only provenance audit established conclusively that `server/model.onnx` did **not** come
from `calib020_0001`: the ONNX file was written 2026-05-15, five weeks before the checkpoint
existed. The probable source is staged `model_iter_0193`, based on timing and the auto-export
selection mechanism, but that positive identification is **circumstantial, not proven**.

Known current artifact facts from the audit:

- product path: `server/model.onnx`;
- SHA-256: `f1b4411a9d46cc767aa31a3f6885c307704897f21c327a3210da5d5c810a6ae5`;
- no ONNX metadata or source-checkpoint identity;
- the file is gitignored;
- `scripts/startServer.js` auto-selects the latest checkpoint from
  `checkpoints/alphazero-v2-staged`, not the calibration line;
- documentation and the auto-export code name different source checkpoints/directories.

The next planning session should decide the smallest safe path to align the product with the
best-supported checkpoint. A credible plan should address, in order:

1. Re-verify the live product selection chain and current artifact identity without changing it.
2. Make future exports self-identifying with source path, source SHA-1, exporter/config identity
   and artifact hash.
3. Export `calib020_0001` to a **new staging path**, never over `server/model.onnx` during
   discovery.
4. Establish numerical parity between native MLX and staged ONNX on a deterministic, diverse
   board corpus, including shapes, channel/layout contract, policy ordering/masking and value
   perspective.
5. Verify the Node/ONNX runtime and product MCTS consume the staged artifact correctly.
6. Decide whether additional product-facing strength evidence is needed. Do not infer that
   "newer" or "best-supported native checkpoint" automatically guarantees a stronger shipped
   experience under a different runtime/search stack.
7. Design deployment as a separate reviewed action with immutable before/after hashes, backup,
   rollback, startup validation and a clear acceptance gate.

Start read-only. Do not overwrite `server/model.onnx`, deploy, reserve research seeds, run a new
training experiment or reopen a closed hypothesis merely because this handoff names the product
gap.

## Recommended first deliverable in the new session

Produce a concise decision memo, not implementation:

- restate the product gap and what is proven versus circumstantial;
- inspect the current export, verification and server-loading paths;
- identify the smallest staged provenance/parity workflow;
- name hidden risks and any evidence still required before replacement;
- propose explicit review gates, artifact paths and rollback;
- give an honest recommendation: proceed, revise the workstream, or stop if the expected
  user-facing value cannot be established economically.
