# G2 V2 — export parity: FAIL on the Python ONNX Runtime surface (2026-11-09)

## The gate, as run

- **Export:** once, to scratch, with `scripts/GPU/alphazero/export_onnx.py` unmodified (byte-identical to
  August's export commit `32751284`):
  `MLX_ENABLE_TF32=0 .venv/bin/python -m scripts.GPU.alphazero.export_onnx --weights /Users/bill/twixt-cache/htb-pilot/student.safetensors --output <scratch>/model.onnx`
  - It ran at `cef270de`, with no modified tracked files (`export.log`).
  - The student's sha256 was `0e1cb621…` before and after; its SHA-1 is `a35e3f4c…`.
  - The pair:
    - graph `bc68a274…`, 82,855 bytes;
    - data `317d7a7c…`, 7,493,120 bytes;
    - **id `78c16a31c3dba37f`**.
  - It was validated through `resolveHardModel` with `HARD_MODEL_MANIFEST`.
- **Measurement:** the Phase 2 tooling, unmodified.
  - `MLX_ENABLE_TF32=0 .venv/bin/python tests/parity/run_python_side.py models/78c16a31c3dba37f …`
  - `MODEL_MANIFEST=models/78c16a31c3dba37f/manifest.json node tests/parity/run_node_side.mjs …`
  - `node tests/parity/compare.mjs …`
  - The verdict recomputes from the committed gz halves, apart from the `audit` field.
- **Verdict: PARITY FAIL** (`parity_result.json`, `compare.log`), with 16 failures, all on S2 and S3.

| surface | max abs logit | mean abs logit | max abs value | top-1 out of band | top-5 sets | τ median |
|---|---:|---:|---:|---:|---:|---:|
| S1 encoding, legal moves | exact | | | | | |
| S2 MLX ↔ ORT-py | 12.46 / 1e-4 | 1.748 / 1e-5 | 1.949 / 1e-4 | 109 | 0% | 0.271 |
| S3 ORT-py ↔ ORT-node | 12.46 / 1e-5 | 1.748 / 1e-6 | 1.949 / 1e-5 | 109 | 0% | 0.271 |
| **S4 MLX ↔ ORT-node** | **2.38e-5 / 1.1e-4** | **1.44e-6 / 1.1e-5** | **1.94e-6 / 1.1e-4** | **0** | **100%** | **1.000** |

- Masks and move-order equivariance pass on all three runtimes.
- The red-perspective check passes, and no value falls outside [−1, 1].
- **The product path (S4: the native network against the product's own runtime) agrees.** Every failure involves
  Python ONNX Runtime 1.24.4, the one runtime the product does not use.

## Diagnostics after the verdict

These do not change the verdict. They were run once each, in fresh Python processes, on corpus position
`opening_00` unless stated. They wrote nothing and produced no alternate artifact.

1. **Python ORT still reproduces August on c34b7ff.** Its logits differ from August's recorded `ort_py` by
   1.24e-5, and from August's MLX by 2.38e-5. The runtime is not broken in general. (It is also not bit-identical
   to August's record, at the same ORT version; Python was 3.14.6 then and is 3.14.10 now.)
2. **On the student, Python ORT is deterministic but wrong.**
   - Three fresh file-loaded sessions agree exactly with each other and with the tool's output: 9.89 from MLX.
   - Two runs in one session are identical.
   - **Loading the same bytes in memory** (`onnx.load(load_external_data=True)`, then `SerializeToString()`),
     bypassing ORT's external-data file loader, gives the same 9.89.
   - **Disabling all graph optimizations** (`ORT_DISABLE_ALL`) gives the same 9.89.
3. **Process state affects it.** In one process where a c34b7ff session was created first, the student session's
   logits differed both from MLX (11.42) and from the fresh-process result (4.14). This was observed once and not
   repeated.
4. **Structure.**
   - The student graph is node-for-node identical to c34b7ff's: 76 nodes, the same ops, inputs, outputs and
     attributes.
   - Both graphs have the same initializer names, shapes and dtypes.
   - No initializer is non-finite.
   - The student's weights contain **373,207 subnormal float32 values**, against **3,044** in c34b7ff.
5. **Subnormals ruled out as the cause.** Flushing every subnormal initializer to zero, in memory, left Python ORT's
   output unchanged: 9.89 on `opening_00`; 9.32, 9.60, 6.60, 2.35 and 8.96 on five more positions.

**What this establishes:** the failure is located in the **Python ONNX Runtime surface on this artifact**. Its output
is deterministic in a fresh process, independent of the external-data loader and of graph optimization, and
disagrees with two independent computations of the same weights. Native MLX and Node ONNX Runtime 1.23.2 agree with
each other to 2.4e-5 on all 126 positions.

**What it does not establish:** the cause. Python ORT 1.24.4 could have a defect on this graph's values; a
value-dependent difference in some kernel is the other open possibility. Nothing here shows the Node result is wrong,
but Node agreeing with MLX is the only evidence for the product path.

## Disposition (Phase 2 §7, "On FAIL")

- Stop and report.
- No retuning, alternate export, exporter change or tolerance revision.
- **The candidate directory is not committed.** The placed copy was moved to scratch; its manifest is preserved
  here as `candidate_manifest_NOT_COMMITTED_AS_MODEL.json`.
- `HARD_MODEL_ID` stays `null`.
- G2's remaining steps (V3, the deferred suites, the golden comparison, the live requests, R1–R3) were **not
  run**. They depend on a pinned student.
