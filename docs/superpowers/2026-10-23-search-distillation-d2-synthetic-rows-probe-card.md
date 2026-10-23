# Search distillation D2 — synthetic row-count probe card

**Date:** 2026-10-23

**Status:** FROZEN before execution.
- **Approved by the user** after the linear-layer reproducer (ledger entry 27, `e1823eed`, addendum `1b10ba29`): one synthetic-only probe, under a single 10-minute ceiling, plus a bug-report draft. Submitting the report is a separate decision.
- **Not approved:** training, any workaround or fix, package changes, model or fixture loading, and pushing. Every other gate stays closed.
- The read-only historical exposure audit approved at the same time is a separate document. It runs nothing.

## 1. Purpose

Entry 27 reproduced the GPU failure with one linear operation on fixture operands. At N = 33,600 and 33,728 rows, the first N − 32,736 rows of fc's Y and dX and of out's dX were grossly wrong. Nothing was gross at N ≤ 32,768, but only five sizes up to 32,768 were measured.

This probe asks three questions with synthetic operands only:
- Does the failure depend on the fixtures, or on the product shape and row count?
- Where does it start, measured at a small predeclared set of row counts?
- Does a wider control layer fail the same way?

It is also meant as a **portable check for a repair**: one file that needs only NumPy and MLX and carries its own expected results.

## 2. The generator (standalone)

`scripts/GPU/alphazero/mlx_matmul_rows_probe.py`, SHA-256 `1217bc8e48589b077bbe0962564ffb5c0298326f96de6658ae39a83cf739bb57`.
- It imports only the standard library, NumPy and MLX. A test checks the imports by AST, and a copy runs with `python -I` outside the repository.
- It is not gated itself, so that it stays portable. In this repository it runs only through the gated entry `search_distillation_d2_mlxrows.py`. That entry refuses unless the gate is open, this card is the pinned bytes and the generator is the pinned bytes above.

## 3. Operands, products and expected result

- **Operands:** float32, from `numpy.random.default_rng([20261023, layer index, N])`.
  - X: standard normal (N × in).
  - W: standard normal / √in (out × in).
  - b: standard normal (out).
  - C: standard normal (N × out), each column centred in float64 before the cast. This makes db = ΣC nearly cancel, as a softmax cotangent does.
- **Operations, on the current device, evaluated jointly with `mx.eval` before being copied to NumPy:**
  - `Y = X @ W.T + b`
  - `dX = C @ W`
  - `dW = C.T @ X`
  - `db = sum(C, axis=0)`
- **Expected:** NumPy accumulates the same float32 operands in float64. MLX is not used.

| Layer (in → out) | Y product | dX product | Role |
|---|---|---|---|
| fc (2 → 64) | (N×2)·(2×64) | (N×64)·(64×2) | failing shapes from entry 27 (Y, dX) |
| out (64 → 1) | (N×64)·(64×1) | (N×1)·(1×64) | failing shape from entry 27 (dX); Y did not fail |
| wide (64 → 64) | (N×64)·(64×64) | (N×64)·(64×64) | the one wider control |

## 4. The case list (frozen)

- **Row counts N:** 32,704; 32,736; 32,737; 32,767; 32,768; 32,769; 32,800; 33,600; 33,728.
- **Cases:** every layer at every row count, 3 × 9 = **27 cases**, in the order fc, out, wide, with N ascending within each layer.
- **Repeats:** each case runs on the **GPU three times** and on the **CPU once**. The CPU is a comparison backend, not an oracle.
- There is no adaptive search and no layout or context grid.

## 5. Judgement (frozen)

- **Scale, per output:**
  - Y, dX and dW use max|expected|, as entry 27 did.
  - db uses an **absolute** scale: max over columns of Σ_rows |C|, the magnitude of the terms being summed. A near-zero expected sum therefore cannot make a small error look gross. This replaces entry 27's rule, under which cancellation alone flagged out's db as gross.
- **Outside the tolerance:** |got − expected| > 1e-4 × scale + 1e-6.
- **Gross:** |got − expected| > 1e-2 × scale.
- **For Y and dX**, which have one row per row of X, the report gives:
  - the rows holding a gross element, as [start, stop) ranges (the first 64 ranges, plus the total count);
  - the largest absolute error inside and outside those rows;
  - how many gross rows are near zero, and for Y how many are near the bias alone (within 1e-2 × scale).
- **Every output:** the largest absolute error, the counts, and the first 20 failing elements.
- **Repeatability:** the GPU repeats are compared bitwise with the first, and the largest difference is reported.

## 6. Evidence

- **In the repository**, under `docs/superpowers/evidence/search-distillation-d2-mlxrows/`, all create-only:
  - `environment.json`, written first: Python, platform and macOS version, NumPy, MLX, Metal device information and the generator's SHA-256;
  - `progress.jsonl`;
  - `cases.jsonl`, one line per case, flushed and fsynced as each case finishes;
  - `artifacts.jsonl`;
  - `record.json`, written on every exit the script survives.
- **Outside the repository:** `~/twixt-cache/search-distillation-d2/mlxrows-<UTC stamp>/compact_example.npz`, read-only, with its SHA-256 in `artifacts.jsonl`. It is **one** compact failing example, chosen by a fixed rule:
  - the first case, in case order, whose GPU fc forward Y has gross rows;
  - failing that, the first case with gross rows in any GPU Y or dX.

  It holds that product's operands (in full when they fit in 4 MiB, otherwise only the rows shown, since the seed regenerates the rest), the rows [0, last gross row + 8) capped at 4,096, the expected rows, the observed rows from GPU repeat 0, and the seed entropy.
- **Nothing else is saved:** no full outputs. Entry 27's artifacts are not touched.

## 7. Stops and outcome

- **A difference is an observation and never stops the run.**
- **COMPLETED (exit 0):** all 27 cases ran.
- **VOID (exit 3):**
  - a non-finite operand or result (the case is written as void first);
  - any exception;
  - SIGTERM.
- **REFUSED (exit 4):**
  - the gate is closed;
  - the card or generator is not the pinned bytes;
  - an output exists.
- **USAGE (exit 2).**
- **No retry.**

## 8. Predictions on record (stated before the run, derived from entry 27 only)

1. **fc Y, fc dX and out dX on the GPU:**
   - no gross rows at N ≤ 32,768;
   - at N = 32,769, 32,800, 33,600 and 33,728, the gross rows are exactly [0, N − 32,736): 33, 64, 864 and 992 rows.

   The sizes 32,769 and 32,800 extrapolate a formula seen at two sizes only, now on synthetic operands.
2. **out Y, every dW and every db:** no gross element at any N on either device. For db, this is under the new absolute scale.
3. **CPU:** nothing gross anywhere.
4. **GPU repeats:** bitwise identical, as in entry 27.
5. **wide control:** **None stated in advance.**

## 9. What this cannot establish

- **The mechanism:** the probe locates rows; it does not explain them.
- **A universal boundary or a safe chunk size:** the probe covers one MLX version, one machine, three layer shapes and nine row counts.
- **Whether any historical training was corrupted:** that is the exposure audit's question, and records decide it.
- **Anything about measured playing strength.**

## 10. Ceiling, cleanup and gate

- **Ceiling:** 10 minutes in all. The wrapper sends TERM to the run's process group at 590 s and KILL 10 s later.
- **Gate and cleanup:** the wrapper restores the gate on every exit, reads it back against the pre-open commit and reports surviving processes.
- **Gate:** `SEARCH_DISTILLATION_D2_MLXROWS_AUTHORIZED` in `scripts/GPU/alphazero/search_distillation_d2_mlxrows_authorization.py`. A one-line commit opens it.
- **Command:** run from a clean linked worktree whose HEAD is that gate-opening commit:

  `bash docs/superpowers/evidence/search-distillation-d2-mlxrows-run/01_wrapper.sh`

  The wrapper runs `env PYTHONPATH=. <main>/.venv/bin/python -B -m scripts.GPU.alphazero.search_distillation_d2_mlxrows run`. That entry calls the generator with `--out <evidence dir> --example-dir ~/twixt-cache/search-distillation-d2/mlxrows-<UTC stamp>`.
- **Before the run:** the machine is checked for idle, as before every GPU run.
