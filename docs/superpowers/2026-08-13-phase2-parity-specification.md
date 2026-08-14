# Phase 2 Parity Specification — PREREGISTERED

**Date:** 2026-08-13 · **Branch:** `codex/product-model-alignment-phase2`
**Basis:** `b5b4320` (merged Phase 1 head)
**Status: PREREGISTERED. Committed before any parity result was generated or examined.**

Authorized by `docs/superpowers/2026-08-13-product-model-alignment-decision-memo.md` §4 Phase 2
and the explicit Phase 2 go-ahead of 2026-08-13.

This document fixes the corpus, the metrics, the tolerances and the pass/fail rule **in
advance**. Nothing below may be revised after a measurement is taken. If a tolerance turns out
to be wrong, that is a finding to report, not a number to adjust.

---

## 1. What is being tested, and what is not

**Tested:** that the ONNX export pipeline and the three runtimes that consume its output agree
with the native MLX network, on a fixed corpus, to a preregistered tolerance — and that the
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
| staging directory | `models/staging-calib020-0001/` — **new**, never the pinned path |
| exporter | `scripts/GPU/alphazero/export_onnx.py`, unmodified |

Exactly one checkpoint is exported. No alternate export, no exporter change, no re-export with
different arguments — if the export is defective, that is the finding.

## 3. Environment of record

Recorded now so a later run can be compared against it, and written into the candidate
manifest's provenance.

| component | version |
|---|---|
| Python | 3.14.6 (`.venv/bin/python`) |
| MLX | present (`mlx.core`) |
| PyTorch | 2.10.0 |
| onnx | 1.20.1 |
| onnxruntime (Python) | 1.24.4 |
| Node.js | 26.7.0 |
| onnxruntime-node | 1.23.2 |

The two ONNX Runtime versions **differ by design** — they are what the two surfaces actually
use. Their disagreement is a measurement, not a defect to be configured away.

## 4. Corpus

**Construction.** The corpus is a committed JSON file of **move sequences**, not encoded
states. Each entry is an ordered list of `[row, col]` moves. Both languages replay the same
list into their own state representation, so the encoding comparison tests the real path rather
than a shared intermediate.

Sequences are generated once by seeded pseudo-random legal play (seed recorded in the file),
then **committed**. Regeneration is not part of the procedure; the committed file is the corpus.

**Strata.** 120 positions total:

| stratum | plies | count |
|---|---:|---:|
| opening | 2–19 | 30 |
| early-mid | 20–49 | 30 |
| midgame | 50–99 | 30 |
| late | ≥100 | 30 |

Plus 6 fixed edge cases, evaluated under every metric and reported separately:

1. empty board (ply 0, maximum legal moves)
2. one move played
3. a position with a single legal move remaining, if constructible; otherwise the
   fewest-legal-moves position in the corpus
4. a position whose legal-move count exceeds 512 (the pre-`cc1b3fa` cap, to confirm the 576
   contract holds where the old one would have overflowed)
5. a position with black to move
6. a position with red to move

**Balance.** Red-to-move and black-to-move counts must each be ≥ 40% of the 120. A corpus
failing this is regenerated **before any model is run against it**, and the regeneration is
recorded.

**Terminal positions are excluded.** The network is not consulted at terminal states in play, so
including them would test an unused path.

## 5. Surfaces

| id | surface | what it isolates |
|---|---|---|
| **S1** | JavaScript vs Python state encoding | the input tensor and legal-move list, before any model runs |
| **S2** | native MLX vs Python ONNX Runtime | the export itself: layout conversion, BatchNorm folding, canonicalization |
| **S3** | Python ONNX Runtime vs Node ONNX Runtime | runtime and version differences on identical bytes |
| **S4** | Node ONNX Runtime vs native MLX | the end-to-end product path, closing the triangle |

S4 is not redundant. S2 and S3 could each pass within tolerance while their errors compose to
exceed it, and S4 is the pair the product actually depends on.

## 6. Metrics and tolerances

All comparisons are over **valid (unmasked) entries only**, in the legal-move order fed in.

### 6.1 Exact-equality metrics — tolerance zero

| metric | surface | rule |
|---|---|---|
| board tensor | S1 | max absolute difference **exactly 0.0** over all `30 × 24 × 24` elements |
| legal-move list | S1 | identical length, identical order, identical coordinates |
| mask positions | S2, S3, S4 | every index ≥ `n_legal` is exactly `-1e9` |
| move-order equivariance | S2, S3, S4 | permuting the input move order permutes output logits identically (tested on 10 positions with a fixed permutation seed) |

These are integer-valued or structural. A non-zero difference is a defect, not noise.

### 6.2 Numerical metrics — preregistered tolerances

| metric | S2 (MLX↔ORT-py) | S3 (ORT-py↔ORT-node) | S4 (MLX↔ORT-node) |
|---|---:|---:|---:|
| max abs logit difference | **≤ 2e-3** | **≤ 1e-4** | **≤ 2e-3** |
| mean abs logit difference | **≤ 2e-4** | **≤ 1e-5** | **≤ 2e-4** |
| max abs value difference (STM) | **≤ 1e-4** | **≤ 1e-5** | **≤ 1e-4** |

**Rationale, recorded in advance.** S2 and S4 cross framework and device: MLX on Metal versus
ONNX Runtime on CPU, with different fp32 accumulation orders and a BatchNorm folding step that
algebraically rearranges the convolution. `2e-3` on a logit is far below the scale at which
policy ordering changes for anything but a genuine near-tie, and the value head passes through
a double `tanh` that compresses residual error. S3 compares the same graph under the same
implementation family, so only version-level kernel differences apply and the bound is an order
of magnitude tighter.

### 6.3 Ordering stability — the decision-relevant metric

Numerical agreement is not the thing that matters; **which move gets played** is. For every
position and every surface pair:

| metric | rule |
|---|---|
| top-1 agreement | must agree, **except** where the top-1/top-2 logit gap on the reference side is ≤ the surface's max-abs-logit tolerance |
| near-tie exemptions | ≤ **5%** of positions (6 of 120) may be exempted this way |
| top-5 set agreement | ≥ **95%** of positions must have identical top-5 sets (order within the set not required) |
| full-ranking Kendall τ | median across positions ≥ **0.99** |

A top-1 disagreement outside a near-tie is a **failure**, not a note. The near-tie exemption
exists because a flip between two moves whose logits differ by less than the numerical tolerance
carries no information — but if such flips are pervasive (>5%), the comparison is uninformative
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
- commit the exact candidate pair and its manifest under `models/staging-calib020-0001/`;
- populate **every** provenance field completely — source path, source SHA-1, export commit,
  exporter configuration, runtime versions — since unlike the baseline these are all known;
- report the measured numbers, including margins against each tolerance;
- **stop.** The next action is a separately reviewed product-stack comparison specification.
  A parity pass is not a promotion, not a deployment, and not a strength claim.

**On FAIL:**
- **stop and report the defect**, with the failing metric, the surface, and the positions;
- no retuning, no alternate export, no exporter modification, no tolerance revision, no
  promotion, and no strength games;
- the candidate pair is **not** committed.

## 8. Procedure order

Deviating from this order invalidates the preregistration.

1. Commit this specification. *(No model has been exported at this point.)*
2. Generate and commit the corpus file. Verify the §4 balance constraint.
3. Verify the checkpoint SHA-1 equals `209cf2d4…`.
4. Export to `models/staging-calib020-0001/` with the unmodified exporter.
5. Run S1. Then S2, S3, S4, each writing a machine-readable result file.
6. Apply §7 to the results **as written**.
7. Report.

The corpus is committed at step 2, before the export at step 4, so it cannot be selected to
suit a model.

## 9. Threats to validity, acknowledged in advance

- **Tolerances are judgement, not measurement.** They were chosen from the arithmetic of fp32
  across backends, not from a pilot run on this candidate. If the true agreement is far tighter,
  the bounds were loose; if a metric fails narrowly, that is still a fail.
- **A pass is not a guarantee for unseen positions.** 120 positions plus 6 edge cases is a
  sample. It is chosen to span game phase, both colours and the move-count extremes, but parity
  outside it is inferred, not measured.
- **CPU-only ONNX Runtime.** Both ORT surfaces run on CPU. A future GPU execution provider is a
  different surface and is not covered.
- **The pinned baseline is not the subject.** This measures the candidate export. It says
  nothing about the artifact currently served, whose source remains unknown.
