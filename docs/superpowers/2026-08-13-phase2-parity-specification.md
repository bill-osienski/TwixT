# Phase 2 Parity Specification — PREREGISTERED

**Date:** 2026-08-13 · **Branch:** `codex/product-model-alignment-phase2`
**Basis:** `b5b4320` (merged Phase 1 head)
**Status: PREREGISTERED. Committed before any parity result was generated or examined.**

Authorized by `docs/superpowers/2026-08-13-product-model-alignment-decision-memo.md` §4 Phase 2
and the explicit Phase 2 go-ahead of 2026-08-13.

This document fixes the corpus, the metrics, the tolerances and the pass/fail rule **in
advance**. Nothing below may be revised after a measurement is taken. If a tolerance turns out
to be wrong, that is a finding to report, not a number to adjust.

> ## Pre-measurement correction — 2026-08-13
>
> This specification was corrected in review **before any corpus was generated, any model was
> exported, or any measurement was taken**. Preregistration is intact: no result existed that
> could have motivated a change. The first version is preserved in git history at `1df9c6d`.
>
> | # | change | reason |
> |---|---|---|
> | 1 | §6.2 tolerances tightened by roughly an order of magnitude | `2e-3` conflicted with the repository's own established export target — `tests/test_onnx_export.py:235,236,317,318` already assert `< 1e-4` for MLX↔ONNX policy and value. A parity gate looser than the existing unit tests would pass artifacts those tests reject. |
> | 2 | §6.3 near-tie band doubled to `2 ×` tolerance; reference side named; cap counted per surface pair | **Arithmetic error.** Two logits may move in opposite directions, so ordering is ambiguous when the reference gap is `≤ 2 × max-abs tolerance`, not `≤ 1 ×`. The original also failed to define which side is the reference, and left the cap ambiguous between all near-ties and actual disagreements. |
> | 3 | §6 edge cases scoped explicitly; `top-k` and Kendall τ made total | Percentage gates over a 126-item set including 6 deliberately extreme positions would have been diluted by them. `top-5` is undefined when `n_legal < 5`, and Kendall τ is undefined for fewer than two items. |
> | 4 | §8 export now content-addressed via scratch, and Node parity loads through `MODEL_MANIFEST` | `models/staging-calib020-0001/` is a *name*, not a content address, which contradicts the Phase 1 design it sits beside. Routing Node parity through the manifest also makes S3/S4 exercise the real validated loader rather than a bare path. |
> | 5 | §3 environment completed; §8 records the exact export command; §6.1 S1 compares the real feed | MLX version was recorded as unknown because `mlx.__version__` is not exposed; package metadata reports `0.30.3`. S1 must compare the actual float32 NCHW buffer `inference.js` builds, not an HWC intermediate, since that conversion is itself the thing under test. |

---

## 1. What is being tested, and what is not

**Tested:** that the ONNX export pipeline and the runtimes that consume its output agree with
the native MLX network, on a fixed corpus, to a preregistered tolerance — and that the
JavaScript and Python state encodings feeding them are identical.

**Not tested, and not claimed:** playing strength, product-stack behaviour, or that
`calib020_0001` is better than the pinned baseline. Parity is a **correctness** result. A pass
licenses no strength claim and no deployment.

**Not permitted regardless of outcome:** changing `DEFAULT_MODEL_ID`, overwriting the pinned
baseline `models/1d64027db521a50f/`, deployment, product-stack games, promotion, training, or
research-seed use.

---

## 2. The single candidate

| | |
|---|---|
| checkpoint | `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors` |
| SHA-1 | `209cf2d4fd24a48553d259dd71b4954867b9473e` (verified before export) |
| architecture | `hidden=128`, `n_blocks=6`, `NUM_CHANNELS=30` |
| staging location | scratch first, then `models/<combined-id>/` — see §8. **Never** the pinned path. |
| exporter | `scripts/GPU/alphazero/export_onnx.py`, unmodified |

Exactly one checkpoint is exported. No alternate export, no exporter change, no re-export with
different arguments — if the export is defective, that is the finding.

## 3. Environment of record

Recorded now so a later run can be compared against it, and written into the candidate
manifest's provenance in full.

| component | version | source |
|---|---|---|
| Python | 3.14.6 | `.venv/bin/python` |
| mlx | 0.30.3 | package metadata |
| mlx-metal | 0.30.3 | package metadata |
| PyTorch | 2.10.0 | `torch.__version__` |
| onnx | 1.20.1 | `onnx.__version__` |
| onnxruntime (Python) | 1.24.4 | `onnxruntime.__version__` |
| numpy | 2.4.1 | `numpy.__version__` |
| Node.js | 26.7.0 | `node -v` |
| onnxruntime-node | 1.23.2 | package metadata |

`mlx.__version__` is not exposed; both MLX versions come from `importlib.metadata`.

The two ONNX Runtime versions **differ by design** — they are what the two surfaces actually
use. Their disagreement is a measurement, not a defect to be configured away.

## 4. Corpus

**Construction.** The corpus is a committed JSON file of **move sequences**, not encoded
states. Each entry is an ordered list of `[row, col]` moves. Both languages replay the same
list into their own state representation, so the encoding comparison tests the real path rather
than a shared intermediate.

Sequences are generated once by seeded pseudo-random legal play (seed recorded in the file),
then **committed**. Regeneration is not part of the procedure; the committed file is the corpus.

**Primary strata — 120 positions.** These and only these carry the aggregate percentage gates.

| stratum | plies | count |
|---|---:|---:|
| opening | 2–19 | 30 |
| early-mid | 20–49 | 30 |
| midgame | 50–99 | 30 |
| late | ≥100 | 30 |

**Edge cases — 6 positions**, reported separately:

1. empty board (ply 0, maximum legal moves)
2. one move played
3. a position with a single legal move remaining, if constructible; otherwise the
   fewest-legal-moves position in the corpus
4. a position whose legal-move count exceeds 512 (the pre-`cc1b3fa` cap, to confirm the 576
   contract holds where the old one would have overflowed)
5. a position with black to move
6. a position with red to move

**How the two sets are treated.** All **126** positions receive every exact check (§6.1) and
every numerical check (§6.2); a violation anywhere is a failure. The **aggregate percentage and
median gates** in §6.3 — near-tie cap, top-k set agreement, Kendall τ median — are computed over
the **primary 120 only**, so that six deliberately extreme positions cannot dilute or dominate a
rate. Edge-case ordering results are reported individually.

**Balance.** Red-to-move and black-to-move counts must each be ≥ 40% of the primary 120. A
corpus failing this is regenerated **before any model is run against it**, and the regeneration
is recorded.

**Terminal positions are excluded.** The network is not consulted at terminal states in play, so
including them would test an unused path.

## 5. Surfaces

| id | surface | reference side | what it isolates |
|---|---|---|---|
| **S1** | JavaScript vs Python state encoding | Python | the input tensor and legal-move list, before any model runs |
| **S2** | native MLX vs Python ONNX Runtime | **MLX** | the export itself: layout conversion, BatchNorm folding, canonicalization |
| **S3** | Python ONNX Runtime vs Node ONNX Runtime | **Python ORT** | runtime and version differences on identical bytes |
| **S4** | native MLX vs Node ONNX Runtime | **MLX** | the end-to-end product path, closing the triangle |

S4 is not redundant. S2 and S3 could each pass within tolerance while their errors compose to
exceed the bound, and S4 is the pair the product actually depends on.

## 6. Metrics and tolerances

All comparisons are over **valid (unmasked) entries only**, in the legal-move order fed in.

### 6.1 Exact-equality metrics — tolerance zero

| metric | surface | rule |
|---|---|---|
| board tensor | S1 | max absolute difference **exactly 0.0** between the float32 **NCHW** buffer `inference.js` builds after its HWC→NCHW conversion and Python's float32 CHW tensor, over all `30 × 24 × 24` elements |
| legal-move list | S1 | identical length, identical order, identical coordinates |
| mask positions | S2, S3, S4 | every index ≥ `n_legal` is exactly `-1e9` |
| move-order equivariance | S2, S3, S4 | permuting the input move order permutes output logits identically (tested on 10 primary positions with a fixed permutation seed) |

The S1 comparison deliberately targets the **real feed**, not an HWC intermediate:
`inference.js` documents itself as "the ONLY place Node.js does layout conversion", so that
conversion is part of what is under test.

These metrics are integer-valued or structural. A non-zero difference is a defect, not noise.

### 6.2 Numerical metrics — preregistered tolerances

| metric | S2 (MLX↔ORT-py) | S3 (ORT-py↔ORT-node) | S4 (MLX↔ORT-node) |
|---|---:|---:|---:|
| max abs logit difference | **≤ 1e-4** | **≤ 1e-5** | **≤ 1.1e-4** |
| mean abs logit difference | **≤ 1e-5** | **≤ 1e-6** | **≤ 1.1e-5** |
| max abs value difference (STM) | **≤ 1e-4** | **≤ 1e-5** | **≤ 1.1e-4** |

**Rationale, recorded in advance.** S2 is held to the repository's own established export
target: `tests/test_onnx_export.py` already asserts `< 1e-4` for MLX↔ONNX policy and value
(lines 235, 236, 317, 318), and a parity gate looser than the existing unit tests would admit
artifacts those tests reject. S3 compares the same graph under the same implementation family,
so only version-level kernel differences apply and the bound is an order of magnitude tighter.
S4 is the **explicit triangle bound** `S2 + S3` — it grants no independent slack, and a S4
failure with S2 and S3 both passing would mean the two error sources compose adversarially,
which is exactly what S4 exists to detect.

### 6.3 Ordering stability — the decision-relevant metric

Numerical agreement is not the thing that matters; **which move gets played** is.

For each surface pair the **reference side** is the one named in §5. Ranking is by logit,
descending, over the legal moves in the order fed in.

| metric | rule |
|---|---|
| top-1 agreement | must agree, **except** where the reference side's top-1/top-2 logit gap is `≤ 2 × the surface's max-abs-logit tolerance` |
| near-tie exemptions | at most **6** *actual top-1 disagreements* may be exempted, counted **independently for each surface pair**, over the primary 120 |
| top-k set agreement | with `k = min(5, n_legal)`: ≥ **95%** of the primary 120 must have identical top-k sets (order within the set not required) |
| full-ranking Kendall τ | median over the primary 120 ≥ **0.99**; τ is defined as **1** where `n_legal < 2` |

**Why `2 ×`.** A flip requires the two logits to cross. In the worst case the top-1 logit falls
by the tolerance while the top-2 rises by it, so the reference gap that can be closed is twice
the per-value bound. Using `1 ×` would have counted genuine floating-point flips as defects.

**What the cap counts.** Only positions where the top-1 move actually differs *and* the gap
falls inside the ambiguous band. Near-ties that do not flip are not exemptions and are not
counted. A top-1 disagreement outside the band is a **failure**, not a note. If flips inside the
band exceed six on any single surface pair, the comparison is uninformative at this tolerance
and that is itself a failure.

### 6.4 Value perspective

| metric | rule |
|---|---|
| side-to-move value | agrees across surfaces within §6.2 |
| red-perspective derivation | the JavaScript conversion to red perspective and an independently written Python conversion agree to **≤ 1e-6** |
| range | every value in `[-1, 1]` on every surface, before any clamping |
| sign convention | for each position, the sign of the STM value agrees across all surfaces except where `|value| ≤ 1e-3` |

The range check runs on **raw** outputs. `AlphaZeroInference` clamps and re-`tanh`es out-of-range
values; a defect masked by that repair would otherwise be invisible.

## 7. Pass/fail rule

**PARITY PASSES only if every metric in §6 passes on every surface.**

There is no partial pass, no weighted score, and no per-metric waiver. A single failing metric
is a parity FAIL.

**On PASS:**
- commit the candidate pair and its manifest under `models/<combined-id>/` as computed in §8;
- populate **every** provenance field completely — source path, source SHA-1, export commit,
  the exact export command, exporter configuration, and every version in §3 — since unlike the
  baseline these are all known;
- report the measured numbers, including the margin against each tolerance;
- **stop.** The next action is a separately reviewed product-stack comparison specification.
  A parity pass is not a promotion, not a deployment, and not a strength claim.

**On FAIL:**
- **stop and report the defect**, with the failing metric, the surface, and the positions;
- no retuning, no alternate export, no exporter modification, no tolerance revision, no
  promotion, and no strength games;
- the candidate directory is **not** committed.

## 8. Procedure order

Deviating from this order invalidates the preregistration.

1. Commit this specification. *(No model has been exported at this point.)*
2. Generate and commit the corpus file. Verify the §4 balance constraint.
3. Verify the checkpoint SHA-1 equals `209cf2d4…`.
4. **Export to scratch**, outside the repository, with the unmodified exporter and exactly:

   ```
   .venv/bin/python -m scripts.GPU.alphazero.export_onnx \
     --weights checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
     --output <scratch>/model.onnx
   ```

   `--hidden` and `--blocks` are left at their defaults of `128` and `6`. The exact command,
   including the resolved scratch path, is recorded in the result files.
5. **Content-address before measuring.** Compute
   `combined-id = computeModelId(sha256(graph), sha256(external_data))` — the same derivation
   `server/model_manifest.js` uses — then place the pair at `models/<combined-id>/` and write
   its complete manifest there. `models/staging-calib020-0001/` is **not** used; a name is not
   a content address.
6. Run S1. Then S2, S3, S4, each writing a machine-readable result file. **Node-side parity
   loads the candidate through `MODEL_MANIFEST`**, so S3 and S4 exercise the real validated
   loading path — hashes, external-data binding, and application contract — rather than a bare
   file path.
7. Apply §7 to the results **as written**.
8. Report.

The corpus is committed at step 2, before the export at step 4, so it cannot be selected to
suit a model.

## 9. Threats to validity, acknowledged in advance

- **Tolerances are judgement, not measurement.** They are anchored to the repository's existing
  `< 1e-4` export assertions rather than to a pilot run on this candidate. If a metric fails
  narrowly, that is still a fail.
- **A pass is not a guarantee for unseen positions.** 126 positions is a sample. It is chosen to
  span game phase, both colours and the move-count extremes, but parity outside it is inferred,
  not measured.
- **CPU-only ONNX Runtime.** Both ORT surfaces run on CPU. A future GPU execution provider is a
  different surface and is not covered.
- **The pinned baseline is not the subject.** This measures the candidate export. It says
  nothing about the artifact currently served, whose source remains unknown.
