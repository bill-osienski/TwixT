# Product-Model Alignment — Decision Memo

**Date:** 2026-08-13 · **Branch:** `codex/competitive-readout`
**Status: DECISION MEMO. IT AUTHORIZES NOTHING.**
**Repository state at writing: unmodified. This file is the only change.**

This is the deliverable requested by `docs/superpowers/2026-08-12-next-session-handoff.md`
("produce a concise decision memo, not implementation"). It records the converged position of
a planning session held 2026-08-12/13, in which two independent reviews were reconciled.

**It does not authorize Phase 1 implementation, any export, any model change, any deployment,
or any training run.** Each phase below needs its own separate go-ahead. Nothing here reopens
the closed research programme, and nothing here is a countersigned experiment card.

---

## 1. Standing position on research — unchanged

`docs/alphazero-value-search-experiment-ledger.md` carries `Status: CLOSED 2026-08-12` and
do-not-repeat `#1–#51`. `calib020_0001` remains the best-supported checkpoint. This memo does
not alter, soften or reinterpret that closure.

The planning session asked whether any axis is genuinely untested. Three candidates survived
the "not a relabelled rescue" test:

| axis | verdict |
|---|---|
| **capacity / architecture** — every checkpoint here is `hidden=128, n_blocks=6`, all 7,524,333 bytes | genuinely untested; **fails the cost bar** — no cheap falsification, weeks of wall clock, and it can still only be measured against siblings |
| **non-self-play data** — no human, external-engine or book data has ever entered training | genuinely untested; the data does not exist in this repository |
| **a non-sibling measurement of the incumbent** | not a successor experiment at all; see §7 |

**Conclusion: none worth the cost. The research stays closed.**

### Boundary clarification, agreed by both reviews

The programme closed **the tested short-continuation mechanisms**, not training in general.
`cont5`, `warm5` and `fp6` were each five iterations; the run that produced this model was 399.

**This is a boundary statement, not a licence.** It exists so a future reader does not read
"training is closed" as "the model is optimal". Any future training proposal still needs its
own scope, named mechanism, preregistered prediction, cheap falsification gate, external
strength anchor and countersigned authorization. **Nothing in Phases 1–3 supplies any of that.**

---

## 2. The product artifact, stated exactly

The served artifact is a **pair**. `server/model.onnx` is the graph only; the weights live in a
separate sidecar referenced by relative filename from inside the graph.

| | graph | external data |
|---|---|---|
| path | `server/model.onnx` | `server/model.onnx.data` |
| size | 82,855 bytes | 7,493,120 bytes |
| SHA-256 | `f1b4411a9d46cc767aa31a3f6885c307704897f21c327a3210da5d5c810a6ae5` | `111546445ea4db8eb775adb7ca611539ac60c63780e200fb9a8ec861ab3b0937` |
| mtime | 2026-05-15 20:57:44 | 2026-05-15 20:57 |
| tracked | no — `.gitignore:53` | no — `.gitignore:54` |

**Provenance verdict is unchanged: `MISMATCH`.** Both files predate `calib020_0001`
(2026-06-20 22:14:54) by five weeks, so neither can have come from it. The probable source
`checkpoints/alphazero-v2-staged/model_iter_0193.safetensors` remains **circumstantial**, and
positive identification is **not pursued** — see §6.

**Artifact identity and provenance are distinct.** A two-file content hash identifies the exact
executable artifact even when its training source is unknown. The baseline manifest will record
`source_checkpoint: unknown` honestly rather than assert a probable one.

---

## 3. Verified findings that shaped the plan

Each was checked directly during the planning session; the check is named so it can be repeated.

1. **The product has no pinned model.** `ensureOnnxModel` (`scripts/startServer.js:45-99`)
   re-exports whenever the newest checkpoint in the hardcoded `checkpoints/alphazero-v2-staged`
   is newer than the ONNX, and `findLatestCheckpoint` (`:31-40`) takes the **lexicographic
   max**. `model_iter_0399.safetensors` has mtime 2026-06-04 05:02, newer than the ONNX. **The
   next successful `npm start` would replace the pair with staged `0399`** — not `0193`, not
   `calib020_0001`. If the export subprocess fails it logs and returns false, so the outcome is
   either replacement or a possibly-partial overwrite. This is the primary defect: selection,
   not the current model.

2. **There is a second, separate unpinned path.** `server/index.js:573` reads
   `process.env.MODEL_PATH || './model.onnx'` — a **relative** path resolved against cwd.
   `scripts/startServer.js:114` sets `MODEL_PATH`; `npm run server` (`package.json:8`) sets
   nothing and falls through to a path that does not exist at the repository root. A third
   reference, `scripts/train_overnight.sh:81`, prints that same broken pattern as user guidance.

3. **BatchNorm is folded into the convolutions.** The graph contains **0**
   `BatchNormalization` ops against 14 `Conv`, 4 `Gemm`, 6 residual `Add` and 2 `Tanh`.
   `export_onnx.py` calls `.eval()` and leaves `do_constant_folding` at its default. The graph
   holds **33** external initializers referencing `model.onnx.data`, which reconciles exactly:
   14 convs × (W, b) + 4 Gemms × (W, b) = 36, less the three sub-threshold biases stored inline.

4. **The served graph is post-`cc1b3fa` (2026-04-28).** Move tensors are **576**, not 512 —
   confirmed by execution: a 576-length input runs, a 512-length input is rejected with
   `Expected: 576`. Canonicalization is embedded in the graph. Board input is `[1, 30, 24, 24]`.
   The structural-staleness risk this could have carried does **not** apply.

5. **The exporter will work unchanged on `calib020_0001`.** All checkpoints are 7,524,333 bytes
   — one architecture — and `export_onnx.py` defaults (`hidden=128`, `blocks=6`,
   `in_channels=NUM_CHANNELS`) already match. `--weights` / `--output` are parameterized.

6. **Real-checkpoint parity has never been tested.** `tests/test_onnx_export.py` already tracks
   30 channels and 576 moves and covers masking and move-order invariance, but
   `test_parity_simple` and `test_parity_multiple_boards` run on randomly-initialized
   `hidden=64, n_blocks=2` networks. `server/test_parity.js` uses `server/test_model.onnx`.
   The missing evidence is real-checkpoint parity and Node-runtime parity, not a new framework.

7. **A product-stack match would be degenerate without design work.** `server/mcts.js:6` states
   there is no Dirichlet noise, and the only `Math.random()` is at `:289` inside the stochastic
   branch of `selectMove`. So `hard` (`nSims: 800, moveTemp: 0`) is **fully deterministic** —
   one game per opening per colour assignment, repeated. `medium` (the default;
   `nSims: 400, moveTemp: 0.5`) is stochastic but **unseeded**, so it is not reproducible today.

---

## 4. The plan

### Gate 0 — artifact boundary

Establish the complete artifact boundary before pinning anything: graph, external-data sidecar,
both hashes, the external references, the ONNX schema, and the Node/ONNX Runtime contract.
A content hash of the graph alone is not the artifact.

### Phase 1 — code and artifact contract only. No export, no model change.

The current pair remains served throughout.

1. **Define the manifest.** Graph filename, size, SHA-256; external-data filename, size,
   SHA-256; logical model ID; source checkpoint path and SHA-1; export commit, exporter
   configuration and runtime versions; supported Node/ONNX Runtime contract.

   **Every provenance field that is unavailable for the baseline is recorded as `unknown`** —
   not only `source_checkpoint`, but export commit, exporter configuration and runtime versions
   as well. **None of them may be reconstructed, inferred or guessed.** The hash and size fields
   are always populated, because they are measured from the bytes on disk rather than recovered
   from history. Future candidate manifests, produced by a known export under Phase 2, must
   populate every field completely.
2. **Commit the current pair** under a content-addressed or otherwise immutable directory,
   with its manifest. **Directly in Git, not LFS** — see §5. The directory **must retain the
   relative `model.onnx.data` basename** the graph expects.
3. **Add `.gitignore` exceptions** for the committed models directory and correct the now-false
   comment at `.gitignore:52` ("use LFS or upload separately if needed").
4. **Make the manifest the single loading path** for both `npm start` (`package.json:7`) and
   `npm run server` (`package.json:8`).
5. **Resolve the default manifest relative to the application, never to cwd.**
6. **Fail loudly** on a missing manifest, missing graph or sidecar, either hash mismatch,
   external-reference mismatch, or schema mismatch. **No silent fallback and no startup export.**
7. **Staging override is manifest-shaped only** — e.g. `MODEL_MANIFEST`, subject to identical
   validation. A raw `MODEL_PATH` will not remain an accepted serving override; that seam is
   how the present situation began.
8. **Update `scripts/train_overnight.sh:81` and `docs/alphazero-twixt.md:2041`.** Leave the
   append-only `docs/superpowers/` history untouched.
9. **Append the erratum** to `docs/superpowers/2026-08-06-model-path-provenance-audit.md` —
   see §8.
10. **Negative tests for each failure case**, including "started via `npm run server` with no
    environment override".

**The manifest asserts identity, not validity.** Phase 1 commits the current pair before Phase 2
has checked anything, so the manifest must claim only *these bytes, this identity,
`source_checkpoint: unknown`* — no parity claim and no strength claim.

### Phase 2 — staged candidate and real parity

**Phase 2 stages the existing `calib020_0001` checkpoint as the sole candidate for the
preregistered product-alignment comparison. It does not extend training, alter the dose, retune
after a non-pass, or authorize any new training run.**

Export to a **new** content-addressed staging directory — never over the served pair — then
compare across:

- native MLX;
- Python ONNX Runtime;
- **Node ONNX Runtime** (separating export error from the JS tensor-encoding path in
  `server/inference.js`);
- JavaScript versus Python full 30-channel state encoding;
- legal-move order and masks;
- raw-logit error **plus argmax and top-k stability** — an ordering flip between near-tied
  moves changes play; max-abs-diff alone does not detect it;
- current-player and red-perspective values.

Expect agreement near 1e-4, not bitwise: BatchNorm running statistics, fp32 accumulation order,
and MLX NHWC versus ONNX NCHW all contribute.

**Phase 2 earns its keep under either Phase 3 branch.** Real-checkpoint parity has never been
run (§3.6), and it cannot be obtained from the current pair, whose source is unknown by
construction. Staging `calib020_0001` is the only way to validate the export pipeline — the
same pipeline that produced whatever is served today.

### Phase 3 — product decision

The branch is **chosen: stronger shipped play**. See §9.

- ~~**Repository correctness only:** keep the exact current pair pinned, `source_checkpoint:
  unknown`, and make **no strength claim**.~~ *(Not selected. Recorded for the reasoning trail.)*
- **Stronger shipped play — SELECTED:** run a reproducible product-stack comparison **before**
  changing the default. This requires preregistered, paired diverse openings (hard mode is
  deterministic) and an injected seeded RNG in the evaluation harness (medium is unseeded).
- A human hard-mode playtest may follow as qualitative external evidence. It **cannot** produce
  Elo, establish that the incumbent "holds up", or support promotion. If its outcome will
  influence a decision, its conditions must be recorded in advance.

---

## 5. Storage decision — settled

**Commit the baseline graph and data pair directly to Git. Do not use LFS.**

The repository has a GitHub remote and the project prioritizes durable handoffs. A local
manifest detects mutation but cannot recover the artifact after a disk failure or support a
clean clone; both the ONNX pair and all checkpoints are currently gitignored, so a manifest
naming a checkpoint SHA-1 would point at something no clone has. ~7.2 MB per pinned model is a
reasonable cost. Regenerating deployment bytes from an ignored checkpoint is not relied upon —
exporter and runtime drift make byte reproduction unguaranteed.

When the Phase 2 candidate becomes part of a formal head-to-head decision, its graph, sidecar
and manifest are committed the same way.

---

## 6. Where a strength claim is and is not licensed

**Not licensed:** "`calib020_0001` is +80 Elo over `0379`, therefore the product gains +80."
That measurement was made under the Python/MLX search stack; the product runs a different MCTS
implementation in Node (`server/mcts.js`, `cPuct` 1.5) over ONNX. The served checkpoint is
probably `0193`, so the actual delta is `0193 → calib020_0001`, **never measured**. Research
promotion evidence says nothing about a different runtime and a different readout.

**Licensed only by:** a reproducible head-to-head in the product's own stack — Node MCTS, ONNX,
fixed readout — under Phase 3's second branch.

**Also relevant to expectations:** `readout_policy.js` sets `DEFAULT_DIFFICULTY = 'medium'` with
`moveTemp: 0.5` at every ply, and the comment states outright that easy and medium sacrifice
strength by design. Any claim about the *default* user experience must be tested at medium, not
only at hard.

**Positive identification of the outgoing artifact is not pursued.** It has no decision value,
and with BatchNorm folded (§3.3) a weight comparison would require reproducing the fusion as
well as the layout conversion. Do it only if it falls out of the Phase 1 validation tooling.

---

## 7. Interpretations examined and NOT adopted

Recorded so they are not re-proposed as findings.

| interpretation | status |
|---|---|
| `cont5 0.31375 → warm5 0.4325 → fp6 0.46625` shows progressively less damage from progressively less disruptive intervention | **Not adopted.** Unpaired across three runs with different mechanisms, seeds and evaluation intervals. Plausible, not established. |
| "fp6 landed at parity" | **Wrong phrasing, withdrawn.** fp6's interval *includes* parity; it was not shown stronger and not shown weaker. Parity is **unresolved**. Never write "equal". |
| Compute/horizon was the binding constraint on the training family | **Not established.** The evidence cannot separate horizon from optimization, data, objective or capacity. Only the design arithmetic (5 iterations vs 399) is factual. |
| Candidate 1's +293 Elo shows checkpoint choice is second-order for the default product | **Overstated, withdrawn.** Its control was `T=1.0` to ply 19 then `T=0.1`, under native MLX search — not the product's `T=0.5`-throughout under ONNX/Node. What survives: readout temperature is *capable* of effects larger than any measured checkpoint delta here, so neither ordering may be assumed. |
| The server has not been used since June, per the ONNX mtime | **Wrong, withdrawn.** `npm run server`, `node server/index.js` and an explicit override all bypass `ensureOnnxModel`. The mtime shows only that no successful automatic re-export has occurred — i.e. the defect in §3.1 is armed and has not yet fired. |
| Positive source provenance is required before the current pair can be pinned | **Wrong, withdrawn.** A two-file content hash is the identity; provenance is a separate question and may remain `unknown`. |

---

## 8. Correction owed to an existing record

`docs/superpowers/2026-08-06-model-path-provenance-audit.md` describes the product artifact as a
single 82,855-byte file with one SHA-256. That is **incomplete**: the artifact is the pair, and
the graph holds 33 external initializers pointing at a 7,493,120-byte sidecar.

An erratum should be appended in Phase 1 adding the sidecar size and hash and stating that the
artifact is the pair. **The `MISMATCH` verdict is unaffected** — both files predate
`calib020_0001`. The audit is item 5 on the handoff's read-first list, so the correction matters
for every future reader.

---

## 9. Phase 3 branch — CHOSEN: stronger shipped play

**Decision (2026-08-13): the Phase 3 branch is stronger shipped play.**

Reason: correctness-only is valuable, but it cannot complete the handoff's stated goal of
improving actual user-facing strength, and it cannot justify replacing the incumbent.

**This choice sets the evidence standard. It does not authorize a match or a deployment.** It
fixes what evidence would be required *if* a replacement is ever proposed — a reproducible
product-stack comparison under the conditions in §4 Phase 3 — and nothing more.

Consequences:

- **If Phase 2 passes:** the next deliverable is a **separate preregistered product-stack
  comparison specification**, written before any games are played. Not this memo, and not an
  extension of it.
- **If Phase 2 fails:** **stop.** A parity failure means the export or serving path is not
  faithful, and no strength comparison built on it would mean anything.

Phases 1 and 2 remain worth doing on their own merits and are unchanged by this choice.

---

## 10. What this memo does not authorize

- Phase 1 implementation, or any code change whatsoever.
- Any ONNX export, including of `calib020_0001`.
- Overwriting, replacing or deploying `server/model.onnx` or its sidecar.
- Any product-stack strength match, or any strength claim about the shipped experience.
- Any training run, of any dose, horizon or mechanism.
- Any reservation or consumption of research seeds.
- Any reopening of a line closed under do-not-repeat `#1–#51`.

The recommended single next action, when authorized, is **Phase 1 only**: pin selection, define
and validate the artifact contract, fail loud, commit the current pair, correct the
documentation — **without exporting or changing the served model.**
