# DRAFT — MLX bug report (not submitted)

**Status:** draft only. Submitting it, and where, is a separate decision for the user.
- **Its evidence today:** ledger entry 27 (fixture operands).
- **Not yet included:** the synthetic probe's results (card `2026-10-23-search-distillation-d2-synthetic-rows-probe-card.md`). That probe has not been run under its approval (ledger entry 28).
- **Snippet below:** not run on its own. It mirrors the probe's code for one case.

---

**Title:** Metal: float32 matmul with a very narrow operand returns wrong leading rows when the row count exceeds 32,768 (wrong first N − 32,736 rows)

**Environment** (from the measured run, ledger entry 27):
- MLX 0.30.3 (pip), NumPy 2.4.1, Python 3.14.9;
- macOS 27.0.1, Apple M5 Max (`applegpu_g17s`), default GPU device.

**Summary.**
- On the GPU, `X @ W.T + b` with X float32 of shape (N, 2) and W of shape (64, 2) returns grossly wrong values in exactly rows 0 … N − 32,737 when N = 33,600 or N = 33,728.
  - Those rows equal `b` alone, as if the X·Wᵀ term were missing.
  - All other rows are correct to within about 1e-3 of scale.
- The CPU device gives correct results everywhere, to about 1e-7.
- The same leading rows are wrong, at those two sizes, in:
  - `C @ W` with C (N, 64) and W (64, 2);
  - `C @ W` with C (N, 1) and W (1, 64).
- **What stays correct:**
  - reductions over the rows (`C.T @ X` and `C.sum(0)`);
  - the (N, 64) @ (64, 1) product.
- **Where nothing was wrong:** with N = 512, 2,048, 8,192, 16,384 and 32,768.
- **What does not change it:**
  - `nn.Linear` or plain `@`;
  - autodiff (`mx.vjp`) or explicit products;
  - X contiguous or a transposed view;
  - outputs evaluated jointly or one at a time.
- **Repeatability:** for the explicit products, the wrong rows were identical across three runs.

**Minimal reproduction** (synthetic; mirrors case `fc_33600` of the probe, not run on its own):

```python
import numpy as np, mlx.core as mx

N = 33_600
rng = np.random.default_rng([20261023, 0, N])
X = rng.standard_normal((N, 2)).astype(np.float32)
W = (rng.standard_normal((64, 2)) / np.sqrt(2)).astype(np.float32)
b = rng.standard_normal(64).astype(np.float32)

expected = X.astype(np.float64) @ W.astype(np.float64).T + b
for dev in (mx.gpu, mx.cpu):
    mx.set_default_device(dev)
    y = mx.array(X) @ mx.array(W).T + mx.array(b)
    mx.eval(y)
    err = np.abs(np.array(y) - expected).max(axis=1)
    bad = np.flatnonzero(err > 1e-2 * np.abs(expected).max())
    print(dev, "bad rows:", (bad.min(), bad.max(), bad.size) if bad.size else None)
# Observed with the fixture operands (entry 27): GPU bad rows 0..863 (864 = N - 32,736); CPU none.
```

**Results with synthetic operands:** *pending the approved probe run.*
- The probe predicts: no wrong rows at N ≤ 32,768, and exactly [0, N − 32,736) at N = 32,769, 32,800, 33,600 and 33,728.
- It also covers a 64→64 control.

**Not claimed:**
- the kernel or dispatch path responsible;
- a general rule for other shapes, devices or MLX versions;
- a safe row count.

**Attachments to consider:**
- the standalone script `mlx_matmul_rows_probe.py`;
- the probe's `environment.json` and `cases.jsonl` (once run);
- one compact `.npz` with operands and the expected rows.
