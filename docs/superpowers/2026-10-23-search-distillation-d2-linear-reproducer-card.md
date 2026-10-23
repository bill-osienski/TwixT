# Search distillation D2 — linear-layer reproducer card

**Date:** 2026-10-23

**Status:** FROZEN before execution.
- **Approved by the user** after the gradient correctness check (ledger entry 26, `2d067917`): preparation and ONE invocation, capped at 10 minutes.
- **Not approved:** training, model replay, held-out data, checkpoint creation, package changes, production fixes and pushing. The encoder is outside this probe. Every other gate stays closed.

## 1. Purpose

Entry 26 localized the GPU discrepancy, given identical stage inputs, to two parts of the policy head:
- the **fc** layer (Linear 2→64): its forward pass and its input gradient;
- the **out** layer (Linear 64→1): its input gradient.

This probe asks whether ONE linear operation, alone, reproduces that, against an independently computed expected result. The intended deliverable is a compact reproducer with that expected result. It would support a precise backend fix or bug report, and give a focused check for validating a repair.

## 2. Operands (float32, from the saved fixtures only)

All fixtures are in `~/twixt-cache/search-distillation-d2/gradcheck-fixtures-20261023T034401Z/`. Each is checked against the SHA-256 recorded in entry 26's `fixtures.jsonl`, which is pinned by its own SHA-256 `5a5e16e8…`. No network is loaded and no model is replayed.

- **out**, states S0 and S21: the `stage_<S>_out` fixtures' own operands.
  - the input X, of shape 33,600 × 64 (S0) or 33,728 × 64 (S21);
  - the weight W (1 × 64) and bias b (1);
  - the cotangent C, reshaped to N × 1.
- **fc**, states S0 and S21: **no fixture holds fc's input**, so its operands are **derived** from the `reference_real_<S>` fixtures by entry 26's NumPy reference.
  - X is the gather output (N × 2), computed in float64 and cast to float32.
  - C is the analytic cotangent of fc's output (N × 64), computed in float64 and cast to float32.
  - W (64 × 2) and b (64) are the saved policy-head tensors, which are exact float32 values.
  - The derived operands are saved, with their hashes, before any case runs.
- **Rows:** each row of X is one move slot. A row is **legal** or **padded** according to the fixture's mask.

## 3. Expected result (independent of MLX)

NumPy accumulates the same float32 operands in float64:
- `Y = X Wᵀ + b`
- `dX = C W`
- `dW = Cᵀ X`
- `db = Σ_rows C`

## 4. Implementations, devices and repeats

Three implementations:
- **linear_autodiff:** `nn.Linear`, with the gradients from `mx.vjp`;
- **explicit_autodiff:** `x @ w.T + b`, with the gradients from `mx.vjp`;
- **explicit_products:** the forward pass and the three gradient formulas, written as direct MLX products with no autodiff.

Each implementation runs on the **GPU three times** and on the CPU once. The CPU result is a comparison only.

Every output is evaluated with `mx.eval` before it is copied to NumPy. The exact shapes, dtypes and layouts are recorded for every case.

## 5. Judgement (frozen)

Per output, the scale is `max|expected|`.
- **Outside the tolerance:** an element with `|got − expected| > 1e-4 × scale + 1e-6`.
- **Gross:** an element with `|got − expected| > 1e-2 × scale`.
- For Y and dX, which have one row per X row, both counts are reported separately for **legal and padded rows**, along with the rows holding a gross element.
- The first 20 failing elements are listed with their indices and both values.
- **Repeats** are compared bitwise.

**The failure REPRODUCES** for a layer and state if any GPU implementation has a gross element or a repeat that is not bitwise identical. An excess at tolerance level is recorded, but it does not count as a reproduction.

## 6. The complete case list (frozen)

1. **Original shapes:** `fc_S0`, `fc_S21`, `out_S0` and `out_S21`, with row-major layout and joint evaluation (all four outputs in one `mx.eval`).
2. **Targeted cases**, run **only** for a layer and state whose original shape reproduces:
   - five predeclared row counts, using the first n rows of X and C: **512, 2,048, 8,192, 16,384 and 32,768**;
   - **one alternative layout** at the original size: X supplied as the transposed view of a row-major (in × N) array;
   - **one alternative evaluation context** at the original size: each output evaluated by its own `mx.eval`, with `mx.synchronize` between them.

   The cases change one factor at a time. There is no product of factors and no adaptive search: 4 + 4 × 7 = **32 cases** at most.
3. **If no original shape reproduces,** that is recorded and nothing more runs: the failure needs the surrounding context of entry 26. The investigation is not expanded during the invocation.

## 7. Evidence, written as it exists

- **In the repository** (`docs/superpowers/evidence/search-distillation-d2-linrepro/`): `progress.jsonl`, `cases.jsonl` (one line per case, skipped cases included), `artifacts.jsonl` and `record.json`.
- **Outside the repository** (`~/twixt-cache/search-distillation-d2/linrepro-<UTC stamp>/`, create-only and read-only):
  - the derived fc operands;
  - for each case with a failing output, that case's operands, the expected values and every failing output of every repeat.

  SHA-256s are in `artifacts.jsonl`.

## 8. Stops and outcome

- **A difference is an observation and never stops the run.**
- **COMPLETED (exit 0):** the case list ran to its end.
- **VOID (exit 3):**
  - a fixture or index hash mismatch;
  - operand shapes that do not fit a linear layer;
  - a non-finite operand or result;
  - an existing output directory;
  - any exception;
  - SIGTERM.
- **REFUSED (exit 4)** and **USAGE (exit 2)**, as before.
- **No retry.**

## 9. Predictions on record

**None stated in advance.**

## 10. Ceiling, cleanup and gate

- **Ceiling:** 10 minutes in all. The wrapper sends TERM to the run's process group at 590 s and KILL 10 s later.
- **Gate and cleanup:** the wrapper restores the gate on every exit, reads it back against the pre-open commit and reports surviving processes.
- **Gate:** `SEARCH_DISTILLATION_D2_LINREPRO_AUTHORIZED` in `scripts/GPU/alphazero/search_distillation_d2_linrepro_authorization.py`. A one-line commit opens it.
- **Command:** run from a clean linked worktree whose HEAD is that gate-opening commit:

  `bash docs/superpowers/evidence/search-distillation-d2-linrepro-run/01_wrapper.sh`

  The wrapper runs `env PYTHONPATH=. <main>/.venv/bin/python -B -m scripts.GPU.alphazero.search_distillation_d2_linrepro run`.
