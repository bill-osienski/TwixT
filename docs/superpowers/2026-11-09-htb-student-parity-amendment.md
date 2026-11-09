# H-TB student export parity — integration-specific amendment of the Python ONNX Runtime surface (2026-11-09)

**Status:** DRAFT, for review. Nothing in §5 has run.

**Applies to:** one artifact only, the ONNX export of the validated H-TB student:
- pair id `78c16a31c3dba37f`;
- graph sha256 `bc68a2742803a846…`, 82,855 bytes;
- data sha256 `317d7a7c0f704c5a…`, 7,493,120 bytes;
- exported once, unmodified exporter, from the student sha256 `0e1cb621…` (ledger, Post-closure entry 55).

**Unchanged and not superseded:**
- `docs/superpowers/2026-08-13-phase2-parity-specification.md` (August's specification). It still governs
  calib020_0001's record.
- **The original V2 FAIL** (entry 55, `docs/superpowers/evidence/htb-integration-g2/`). It stands as the record of
  this artifact on the Python onnxruntime 1.24.4 surface.

This amendment defines a separate, new qualification run, **V2-R**.

## 1. Why an amendment

The V2 failure was diagnosed in entry 56 (`…/htb-integration-g2/diagnostic-1/`):
- **The defect:** Python onnxruntime 1.24.4 computes same-shape Conv nodes whose filter 0 is all zero with one set
  of parameters, within a session and across sessions in a process.
- **Why it hits this artifact:** the student has four such 128×128 convolutions, so the S2 and S3 reference runtime
  was not computing this model.
- **The product's runtime is unaffected:** onnxruntime-node 1.23.2 computes the same graphs correctly, and V2's S4
  passed with wide margins.

A correctness canary was then built and run (`…/htb-integration-g2/canary-1/`). It uses synthetic weights only, and
compares every output with its own float64 reference, within a session and across sessions.

| runtime | environment | canary |
|---|---|---|
| Python onnxruntime **1.24.4** | project `.venv` (Python 3.14.10) | **FAIL**: zero-filter convs off by 5.0–8.3; the dense control passes (≤ 5.1e-6) |
| Python onnxruntime **1.23.2** | new isolated venv (Python 3.13.15) | **PASS**: every output ≤ 2.3e-6; a repeat run agrees |
| onnxruntime-node **1.23.2** (product) | Node v24.22.0 | **PASS**. Its JS-built graphs also reproduce the 1.24.4 failure exactly under Python 1.24.4, so the test is sensitive to the defect |

A canary pass does not qualify the artifact. It qualifies a runtime to serve as the reference surface.

## 2. The one change

In S2 (MLX ↔ ORT-py) and S3 (ORT-py ↔ ORT-node), **"ORT-py" is Python onnxruntime 1.23.2**:
- wheel `onnxruntime-1.23.2-cp313-cp313-macosx_13_0_arm64.whl`, sha256 `2ff531ad8496281b…`, equal to PyPI's published
  digest;
- the same version as the product's onnxruntime-node.

**Everything else is August's specification as written:**
- the corpus (`tests/parity/corpus.json`, 126 positions);
- the surfaces S1–S4;
- every metric and tolerance in §6;
- the near-tie rule, top-k and Kendall gates;
- the value-perspective checks;
- the §7 pass/fail rule;
- the measurement tools, unmodified: `tests/parity/run_python_side.py`, `run_node_side.mjs` and `compare.mjs`;
- **the same exported pair**: no re-export, no exporter change, no change to its bytes.

## 3. Environment of record for V2-R

- **Python side:** the isolated venv `~/twixt-cache/ort-canary-1.23.2/venv`.
  - It was built from an existing standalone interpreter, `~/twixt-cache/htb-teacher/python/bin/python3.13`
    (3.13.15, sha256 `c0f2d302…`), with bytecode writing off. Nothing under `htb-teacher` changed.
  - It now holds onnxruntime 1.23.2, onnx 1.20.1, numpy 2.4.1 and their dependencies, installed offline from
    hash-recorded wheels (`canary-1/wheel_sha256.txt`, `pip_freeze.txt`).
  - **For V2-R it needs, with approval:** `mlx==0.30.3`, `mlx-metal==0.30.3` and `torch==2.10.0`, so the unmodified
    `run_python_side.py` runs in one environment. MLX is the native reference; torch is only imported for version
    reporting. They would be installed the same way: download, record sha256, install with `--no-index`.
  - Run with `MLX_ENABLE_TF32=0`, as V2 and the H-TB runs were.
- **Node side:** onnxruntime-node **1.23.2**, now pinned exactly in `package.json` (it was `^1.20.0`; the lockfile
  already held 1.23.2). Node v24.22.0.
- **Known differences from August's environment of record:**
  - Python 3.13.15 (August 3.14.6);
  - ORT-py 1.23.2 (August 1.24.4);
  - Node v24.22.0 (August v26.7.0);
  - protobuf 7.36.2 in the isolated venv.

## 4. Preconditions, each recorded; any failure stops before measurement

1. **The student and the export:**
   - the student's sha256 is `0e1cb621…`;
   - the pair hashes are as above;
   - the pair is placed at `models/78c16a31c3dba37f/`, uncommitted, with the manifest preserved in
     `…/candidate_manifest_NOT_COMMITTED_AS_MODEL.json`;
   - it validates through `resolveHardModel` with `HARD_MODEL_MANIFEST`.
2. **Canaries:** both pass immediately before the measurement, in the exact environments used:
   - `tests/parity/ort_conv_canary.py` in the V2-R Python environment, after MLX and torch are installed;
   - `server/test_ort_conv_canary.js` on the Node side.
3. **MLX drift check:** this preregisters a stop condition, not a parity gate. V2-R's native MLX logits and values
   are compared with V2's recorded MLX outputs (`python_side.json.gz`, Python 3.14.10). If any position differs by
   more than **1e-4** (S2's max-abs tolerance), stop and report environment drift before applying §7. A different
   Python build must not silently move the reference side.

## 5. Procedure (after this amendment is reviewed and approved)

1. Commit the reviewed amendment.
2. Install the three packages of §3 into the isolated venv, recording their hashes.
3. Run the preconditions of §4.
4. Run, once each:
   - `MLX_ENABLE_TF32=0 <isolated python> tests/parity/run_python_side.py models/78c16a31c3dba37f <out>`;
   - `MODEL_MANIFEST=models/78c16a31c3dba37f/manifest.json node tests/parity/run_node_side.mjs …`;
   - `node tests/parity/compare.mjs …`.
5. Apply August's §7 **as written**.
6. **On PASS:**
   - commit the pair and its manifest under `models/78c16a31c3dba37f/`;
   - pin `HARD_MODEL_ID`;
   - commit the gzipped halves and the verdict as V2-R evidence;
   - G2's remaining approved steps then follow (V3, the deferred suites and golden comparison, the live requests,
     R1–R3).
7. **On FAIL:** stop and report. Nothing is committed under `models/`. There is no retry, no further runtime swap
   and no tolerance change.

## 6. What a PASS would and would not mean

- **It would mean:** the export agrees with the native network on the product path (S4), and on a Python reference
  runtime that passes the canary (S2, S3), to August's tolerances on this corpus.
- **It would not mean:**
  - that V2's FAIL was wrong. It correctly recorded a defective reference surface.
  - that onnxruntime 1.23.2 is free of other defects. The canary covers one known class. S4's agreement between MLX
    and the product's runtime is the independent check on the product path.
  - anything about strength, deployment or the G3 match.

## 7. Product guard (applied in the same change set as this draft)

- `package.json`: `"onnxruntime-node": "1.23.2"`, exact; the lockfile's root entry matches. `npm ls` reports
  1.23.2 with no mismatch, and nothing was reinstalled.
- `server/test_ort_conv_canary.js` is added to `npm run test:server`. Any onnxruntime-node that mixes up these Convs
  fails the server suite. The script now also lists the G1 test files it had omitted.
