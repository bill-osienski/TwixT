# Targeted Value Calibration v13c — Projection-Strength Scalar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--post-opening-calibration-projection-strength` scalar (default 1.0) that scales the v13 gradient-conflict correction, by folding it into the effective projection weight — strengthening the correction only where the A and guardrail gradients already conflict, without changing the margin or any guardrail row.

**Architecture:** The v13 correction is applied by the pure primitive `project_conflicting_gradient(..., weight=calibration_loss_weight)` at trainer.py:1595. v13c leaves that primitive unchanged and passes `weight = projection_strength * calibration_loss_weight`. One behavioral line + a CLI arg + plumbing + a telemetry echo in both JSON sites. Numerically identical projection update at strength=1.0.

**Tech Stack:** Python 3.14 / MLX, pytest.

**Spec:** `docs/superpowers/specs/2026-07-07-targeted-value-calibration-v13c-projection-strength-design.md` (APPROVED — do not redesign).

## Global Constraints

- Python: always `.venv/bin/python`; tests `.venv/bin/python -m pytest <file> -v`; full-suite baseline on main after v13 + telemetry fix: **1367 passed**.
- NEVER `sys.modules.pop("mlx")` in tests.
- **The one behavioral change:** at the projection call site (trainer.py:1595) pass `weight = post_opening_calibration_projection_strength * calibration_loss_weight`. `project_conflicting_gradient`'s signature, the conflict-only gate, `c = dot/(normsq+1e-12)`, and the projected surface (`value_head.*` + `encoder.blocks[last].*`) are UNCHANGED.
- **`calib_projection_strength` MUST be persisted in BOTH telemetry sites:** the sidecar `post_opening_calibration.loss` block (`build_post_opening_calibration_block` in calibration_pool.py) AND the flattened `model_iter_*.json` row (the `_teacher_calib_scalars` mirror in trainer.py). Adding to only the sidecar silently drops it from the per-iteration row (the v13 bug fixed in `89e2965`).
- **Numerically identical projection update at strength=1.0** (`effective_weight = calibration_loss_weight`): this is a per-update numerical-identity claim, NOT a whole-run checkpoint byte-identity claim across a stochastic training run. The new `calib_projection_strength` telemetry key is additive (defaults 1.0).
- Do NOT change: `project_conflicting_gradient`, the component-loss helper, `alphazero_loss_batch`, the v12b manifest/builder, the guardrail margin (stays 0.10), the v12b schedule.
- Worktree `feature/tvc-v13c-projection-strength`; fresh worktree lacks gitignored data → known 14F+6E in the whole-repo suite there; judge tasks file-scoped, authoritative suite on merged main. Per-task commits, FF-merge (no `--no-ff`, never force-push). Trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. File-scoped `git add`. Locate code by content, not line numbers.

## File Structure

| File | Role |
|---|---|
| `scripts/GPU/alphazero/train.py` (modify) | `--post-opening-calibration-projection-strength` arg + plumb |
| `scripts/GPU/alphazero/trainer.py` (modify) | `train()` + `train_step` param; the effective-weight multiply; `proj_strength` accumulator + `calib_projection_strength` mirror key |
| `scripts/GPU/alphazero/calibration_pool.py` (modify) | `calib_projection_strength` in `build_post_opening_calibration_block` |
| `tests/test_v13c_projection_strength.py` (create) | weight-linearity + self-consistency + wiring + telemetry-both-sites tests |

**Task → work-item map:** T1 = the full change + tests; T2 = full suite + merge (controller-run).

---

### Task 1: projection-strength scalar (arg + fold-into-weight + telemetry)

**Files:**
- Modify: `scripts/GPU/alphazero/train.py`, `scripts/GPU/alphazero/trainer.py`, `scripts/GPU/alphazero/calibration_pool.py`
- Test: `tests/test_v13c_projection_strength.py` (create)

**Interfaces:**
- Consumes: v13's `project_conflicting_gradient(surf_total, surf_A, surf_G, weight, eps=1e-8) -> (surf_final, telem)` (telem keys `evaluated, conflict, skip_reason, dot, cos, c, removed_norm, norm_G, norm_A`); the v13 `train_step` projection block + 14-tuple; the `_teacher_calib_scalars` flattening mirror; `build_post_opening_calibration_block`.
- Produces: `--post-opening-calibration-projection-strength` → `train(post_opening_calibration_projection_strength=...)` → `train_step(post_opening_calibration_projection_strength=...)` → `weight = strength * calibration_loss_weight`; the `calib_projection_strength` JSON key in both sites.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_v13c_projection_strength.py`:

```python
"""v13c: projection_strength folds into the effective projection weight
(strength * calibration_loss_weight), scaling the conflict correction without
touching the geometry (c/cos/dot) or any guardrail row."""
import numpy as np
import pytest
import mlx.core as mx
import mlx.optimizers as optim

from scripts.GPU.alphazero.trainer import (
    project_conflicting_gradient, MainModule, freeze_batchnorm_running_stats,
    train_step)
from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord
from scripts.GPU.alphazero.calibration_pool import target_in_to_move


def _surf(vh, blk):
    return {"value_head": {"w": mx.array(vh, dtype=mx.float32)},
            "block": {"w": mx.array(blk, dtype=mx.float32)}}


def test_weight_scaling_is_linear_same_geometry():
    # folding strength into weight: 2x weight -> 2x correction + removed_norm,
    # identical c/cos/dot (geometry is weight-independent).
    st, a, g = _surf([1.0, 0.0], [0.0]), _surf([1.0, 0.0], [0.0]), _surf([-1.0, 0.0], [0.0])
    out1, t1 = project_conflicting_gradient(st, a, g, weight=0.01)          # strength 1.0
    out2, t2 = project_conflicting_gradient(st, a, g, weight=0.02)          # strength 2.0
    assert t1["conflict"] is True and t2["conflict"] is True
    assert t1["c"] == t2["c"] and t1["dot"] == t2["dot"] and t1["cos"] == t2["cos"]
    dev1 = 1.0 - float(out1["value_head"]["w"][0].item())
    dev2 = 1.0 - float(out2["value_head"]["w"][0].item())
    assert dev2 == pytest.approx(2.0 * dev1, abs=1e-6)
    assert t2["removed_norm"] == pytest.approx(2.0 * t1["removed_norm"], abs=1e-6)


def _pos():
    return PositionRecord(board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
                          to_move="red", legal_moves=[(0, 0), (1, 1), (2, 2)],
                          visit_counts=[10, 5, 3], outcome=1.0, active_size=24,
                          ply=0, game_n_moves=10)


def _row(to_move, target_black):
    return PositionRecord(board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
                          to_move=to_move, legal_moves=[(0, 0), (1, 1)],
                          visit_counts=[0, 0], outcome=target_in_to_move(to_move, target_black),
                          active_size=24, ply=20, game_n_moves=None)


_CALIB = [_row("black", -0.35), _row("red", -0.9)]
_SIGN = np.array([0.0, 1.0], dtype=np.float32)


def _run(strength):
    net = create_network(hidden=64, n_blocks=2)
    freeze_batchnorm_running_stats(net)
    mm = MainModule(net.encoder, net.policy_head)
    om, ov = optim.Adam(learning_rate=1e-3), optim.Adam(learning_rate=1e-3)
    return train_step(network=net, main_module=mm, opt_main=om, opt_value=ov,
                      batch=[_pos() for _ in range(3)], calibration_positions=_CALIB,
                      calibration_loss_weight=0.01, calibration_guardrail_sign=_SIGN,
                      guardrail_margin=0.10, train_value_head_and_final_block=True,
                      post_opening_calibration_gradient_projection=True,
                      post_opening_calibration_projection_strength=strength)


def test_train_step_folds_strength_into_weight():
    # removed_norm must equal strength * calib_weight * |c| * norm_G. Holds always
    # (c=0 on no-conflict), and on a conflict step this catches a missing multiply.
    for strength in (1.0, 2.0):
        proj = _run(strength)[13]
        expected = strength * 0.01 * abs(proj["c"]) * proj["norm_G"]
        assert proj["removed_norm"] == pytest.approx(expected, rel=1e-5, abs=1e-9), (strength, proj)


def test_projection_strength_default_is_one():
    # strength omitted -> effective weight == calibration_loss_weight (numerically
    # identical projection update to v13); removed_norm uses 1.0.
    net = create_network(hidden=64, n_blocks=2)
    freeze_batchnorm_running_stats(net)
    mm = MainModule(net.encoder, net.policy_head)
    om, ov = optim.Adam(learning_rate=1e-3), optim.Adam(learning_rate=1e-3)
    proj = train_step(network=net, main_module=mm, opt_main=om, opt_value=ov,
                      batch=[_pos() for _ in range(3)], calibration_positions=_CALIB,
                      calibration_loss_weight=0.01, calibration_guardrail_sign=_SIGN,
                      guardrail_margin=0.10, train_value_head_and_final_block=True,
                      post_opening_calibration_gradient_projection=True)[13]
    assert proj["removed_norm"] == pytest.approx(1.0 * 0.01 * abs(proj["c"]) * proj["norm_G"],
                                                 rel=1e-5, abs=1e-9)


def test_strength_does_not_resurrect_no_op():
    # no conflict (dot>=0): the correction is 0 regardless of weight/strength.
    st, a, g = _surf([1.0, 2.0], [3.0]), _surf([1.0, 0.0], [0.0]), _surf([1.0, 0.0], [0.0])
    out2, t2 = project_conflicting_gradient(st, a, g, weight=0.02)          # strength 2.0
    assert t2["conflict"] is False and t2["c"] == 0.0 and t2["removed_norm"] == 0.0
    assert out2 is st                                                       # unchanged


def test_cli_and_telemetry_wiring():
    from scripts.GPU.alphazero import train as train_mod
    from scripts.GPU.alphazero import trainer as trainer_mod
    from scripts.GPU.alphazero import calibration_pool as cp_mod
    tsrc = open(train_mod.__file__).read()
    assert '"--post-opening-calibration-projection-strength"' in tsrc
    assert ("post_opening_calibration_projection_strength="
            "args.post_opening_calibration_projection_strength,") in tsrc
    rsrc = open(trainer_mod.__file__).read()
    # the fold-into-weight multiply + the plumb + both accumulator/mirror keys
    assert ("post_opening_calibration_projection_strength * calibration_loss_weight") in rsrc
    assert ("post_opening_calibration_projection_strength="
            "post_opening_calibration_projection_strength,") in rsrc
    assert '"proj_strength"' in rsrc
    assert '"calib_projection_strength"' in rsrc          # flattening mirror tuple
    csrc = open(cp_mod.__file__).read()
    assert '"calib_projection_strength"' in csrc          # sidecar loss block
```

Note on `test_projection_strength_default_is_one`: use a plain net constructed inline (the walrus keeps `net` in scope for `MainModule`); it asserts the default path uses strength 1.0.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_v13c_projection_strength.py -v`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'post_opening_calibration_projection_strength'` (and the source-string asserts fail).

- [ ] **Step 3: Implement**

**(a) `train_step` signature.** Locate `post_opening_calibration_gradient_projection: bool = False,   # v13` in the `def train_step(` params (there is exactly one in `train_step`, near line 1453) and add below it:

```python
    post_opening_calibration_projection_strength: float = 1.0,   # v13c
```

**(b) the fold-into-weight multiply.** Locate the projection call site:

```python
                _surf_final, _proj_telem = project_conflicting_gradient(
                    _surf_total, _surf_a, _surf_g, weight=calibration_loss_weight)
```

Replace it with (compute the effective weight, pass it as `weight=`):

```python
                _effective_projection_weight = (
                    post_opening_calibration_projection_strength * calibration_loss_weight)
                _surf_final, _proj_telem = project_conflicting_gradient(
                    _surf_total, _surf_a, _surf_g, weight=_effective_projection_weight)
```

**(c) `train()` signature.** Locate `post_opening_calibration_gradient_projection: bool = False,   # v13` in the `def train(` params (near line 2691) and add below it:

```python
    post_opening_calibration_projection_strength: float = 1.0,   # v13c
```

**(d) `train()` forward to `train_step`.** Locate `post_opening_calibration_gradient_projection=post_opening_calibration_gradient_projection,` in the CALIBRATION-branch `train_step(...)` call and add below it:

```python
                                post_opening_calibration_projection_strength=post_opening_calibration_projection_strength,
```

**(e) `loss_accumulator` key.** Locate `"proj_enabled": post_opening_calibration_gradient_projection,` in the `loss_accumulator` dict and add below it:

```python
                        "proj_strength": post_opening_calibration_projection_strength,
```

**(f) flattening mirror.** Locate `"calib_projection_no_conflict_steps")` (the last entry of the `_teacher_calib_scalars` mirror tuple) and change it to add the strength key inside the tuple:

```python
                    "calib_projection_no_conflict_steps", "calib_projection_strength")
```

**(g) `train.py` CLI arg.** Locate `parser.add_argument("--post-opening-calibration-gradient-projection",` and add a sibling after its full definition:

```python
    parser.add_argument("--post-opening-calibration-projection-strength", type=float,
        default=1.0,
        help="v13c: scale the gradient-conflict correction by folding this into "
             "the effective projection weight (strength * calibration weight). "
             "Only affects conflicting steps; 1.0 = v13 behavior.")
```

**(h) `train.py` plumb.** Locate `post_opening_calibration_gradient_projection=args.post_opening_calibration_gradient_projection,` in the `train(...)` call (or the `train_kwargs.update(dict(...))` block that carries it) and add alongside it:

```python
        post_opening_calibration_projection_strength=args.post_opening_calibration_projection_strength,
```

**(i) sidecar JSON key.** In `build_post_opening_calibration_block` (calibration_pool.py), locate `"calib_projection_scope": "value_head_and_final_block",` and add after it:

```python
            "calib_projection_strength":
                float(loss_accumulator.get("proj_strength", 1.0)),
```

- [ ] **Step 4: Run the new tests + regression**

Run: `.venv/bin/python -m pytest tests/test_v13c_projection_strength.py tests/test_v13_projection_wiring.py tests/test_gradient_projection.py tests/test_calibration_cli_flags.py tests/test_calibration_pool.py -v`
Expected: ALL PASS (v13c new tests green; v13 projection/wiring + CLI + pool unchanged — byte-identical at the default strength 1.0).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/train.py scripts/GPU/alphazero/trainer.py scripts/GPU/alphazero/calibration_pool.py tests/test_v13c_projection_strength.py
git commit -m "feat(training): v13c --post-opening-calibration-projection-strength (fold into effective projection weight)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: full-suite verification + merge handoff (controller-run)

- [ ] **Step 1: Full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected in the worktree: baseline-passed + the new v13c tests, with EXACTLY the known 14 failed + 6 errors (missing gitignored data). Authoritative check (1367 + new v13c tests, 0 failures) on merged main before push.

- [ ] **Step 2: Hand off to merge**

FF-merge to main, authoritative suite on merged main, push (superpowers:finishing-a-development-branch). STOP after push — the operator run (canonical v13 command + `--post-opening-calibration-projection-strength 2.0` + new checkpoint dir, confirm `calib_projection_strength=2.0` in `model_iter_*.json`, verifier exit 0, gates A/B/C/D) is the USER's; the exact command block is in the spec's Operator-run section.

---

## Operator run (USER's, after merge) — from the spec

The same canonical v13 command (full harness + v12b manifest + schedule + `--guardrail-margin 0.10 --freeze-batchnorm-stats --train-value-head-and-final-block --post-opening-calibration-gradient-projection`) **plus `--post-opening-calibration-projection-strength 2.0`**, new checkpoint dir `checkpoints/alphazero-v13c-projection-strength-from-calib020-0001`. Confirm `calib_projection_strength=2.0` in `model_iter_*.json`; nonzero `calib_projection_conflict_steps`; `verify_value_head_and_final_block_checkpoint` exit 0; gates A/B/C/D vs `calib020_0001`. v13c is the final projection branch: no change / `conflict_rate ≈ 0` ⇒ stop the projection line (default next = v14 adapter/gated value correction).
```
