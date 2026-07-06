# Targeted Value Calibration v13 — Asymmetric Gradient-Conflict Projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an asymmetric, conflict-only projection of the A-correction gradient away from the guardrail-hinge gradient on the shared final-block surface, behind a flag, so A moves only in directions that do not fight the guardrail — keeping the v12b objective/manifest/schedule unchanged.

**Architecture:** The existing single backward pass gives `g_total = g_S + w·(g_A + g_G)`. v13 adds two cheap calibration-only backward passes for the unweighted A and guardrail component losses (`g_A`, `g_G`), and — over the applied surface leaves only (`value_head.*` + `encoder.blocks[last].*`) — corrects `g_total := g_total − w·c·g_G` when `dot(g_A,g_G) < 0` (`c = dot/(‖g_G‖²+1e-12)`), else leaves `g_total` untouched. The correction happens before the grad split/clip/apply. Off by default; byte-identical to v12b when off.

**Tech Stack:** Python 3.14 / MLX (Apple Metal — nested-dict grad pytrees; `nn.value_and_grad` differentiates one scalar), pytest.

**Spec:** `docs/superpowers/specs/2026-07-06-targeted-value-calibration-v13-gradient-projection-design.md` (APPROVED — do not redesign).

## Global Constraints

- Python: always `.venv/bin/python`; tests `.venv/bin/python -m pytest <file> -v`; full-suite baseline on main after v12b+hardening: **1355 passed**.
- NEVER `sys.modules.pop("mlx")` in tests.
- **The correction is `g_final = g_total − calibration_loss_weight · c · g_G` on the surface leaves only**, with `c = dot/(normsq+1e-12)` iff `dot < 0` AND `norm_G > eps` (`eps = 1e-8`), else `c = 0` and `g_final = g_total` (same object, no leaf changed).
- **`g_A` and `g_G` are gradients of the UNWEIGHTED component losses** (`value_term` for A, `guardrail_hinge_loss` for G). The single factor of `calibration_loss_weight` appears once, in the correction term — do NOT weight the component losses (double-weight bug).
- **Component selection is by the guardrail-sign MASK, never by tag:** A = `base_w·(1−|sign|)` (guardrail rows contribute 0), guardrail = `base_w·|sign|` (A rows contribute 0). These reproduce the exact `value_term`/`guardrail_hinge_loss` from `alphazero_loss_batch`'s guardrail branch (trainer.py:1271-1285).
- **Projection scope = `value_head.*` + `encoder.blocks[last].*` only** (`last = len(network.encoder.blocks) − 1`). Never stem, blocks 0..last-1, policy_head, BN running stats, or frozen tensors.
- **Order in `train_step`:** compute `g_total` → (if flag) compute `g_A`/`g_G`, correct `g_total` surface leaves → split main/value → clip → v9-guarded apply. The correction precedes clipping.
- **Flag:** `--post-opening-calibration-gradient-projection` (store_true, default off). REQUIRES `--train-value-head-and-final-block`; enabling with `--train-value-head-only` raises `ValueError`. No-op (telemetry-visible, not an error) on: no A rows, no guardrail rows, `norm_G ≤ eps`, or `dot ≥ 0`.
- **Byte-identical when the flag is OFF** (7/10/13-tuple arities + values unchanged; pre-existing calibration tests pass unmodified). When ON with no conflict, the corrected gradient is elementwise equal to `g_total` (no whole-run byte-identity claim — two extra passes touch MLX state).
- Do NOT touch: the v12b loss/hinge math, `alphazero_loss_batch`'s bundled path, the v12b manifest/builder, mcts.py, continuation_extraction.py, the v8/v9 verifiers, docs/post-game-analysis.md.
- Worktree `feature/tvc-v13-gradient-projection`; fresh worktree lacks gitignored data → known 14F+6E in the whole-repo suite there; judge tasks file-scoped, authoritative suite on merged main. Per-task commits, FF-merge (no `--no-ff`, never force-push). Trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. File-scoped `git add`. Locate code by content, not line numbers.

## File Structure

| File | Role |
|---|---|
| `scripts/GPU/alphazero/trainer.py` (modify) | `_calibration_component_loss` + `project_conflicting_gradient` + `_tree_dot`/`_tree_axpy` (Task 1); `train_step` projection block + 14-tuple + guard (Task 2); train-loop telemetry (Task 4) |
| `scripts/GPU/alphazero/train.py` (modify) | `--post-opening-calibration-gradient-projection` (Task 3) |
| `scripts/GPU/alphazero/smoke_v13_gradient_projection.py` (create) | gate-0 smoke (Task 5) |
| `tests/test_gradient_projection.py` (create) | projection-math + component-loss unit tests (Task 1) |
| `tests/test_v13_projection_wiring.py` (create) | train_step wiring + byte-identical + guard + no-conflict-telemetry tests (Task 2); CLI/telemetry source pins (Tasks 3-4) |

**Task → work-item map:** T1 helpers, T2 train_step wiring, T3 CLI, T4 telemetry, T5 smoke+suite+merge.

---

### Task 1: Component-loss + projection helpers

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` (add three module-level helpers near `clip_grad_norm`)
- Test: `tests/test_gradient_projection.py` (create)

**Interfaces:**
- Consumes: `make_padded_batch`, `network.forward_padded`, existing `create_network`/`MainModule`/`freeze_batchnorm_running_stats`, `target_in_to_move`.
- Produces:
  - `_calibration_component_loss(model, calibration_positions, calibration_weights, calibration_guardrail_sign, guardrail_margin, component: str, max_moves_cap: int = 512) -> mx.array` — eval-mode calib-only scalar; `component in {"a_correction","guardrail_hinge"}`.
  - `project_conflicting_gradient(surf_total, surf_A, surf_G, weight: float, eps: float = 1e-8) -> (surf_final, telem: dict)` — pure pytree projection over the surface views.
  Task 2 consumes both.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gradient_projection.py`:

```python
"""v13 gradient-conflict projection: a pure surface-pytree projection helper and
an eval-mode calibration-only component-loss helper."""
import numpy as np
import mlx.core as mx
import pytest

from scripts.GPU.alphazero.trainer import (
    project_conflicting_gradient, _calibration_component_loss,
    freeze_batchnorm_running_stats)
from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord
from scripts.GPU.alphazero.calibration_pool import target_in_to_move


def _surf(vh, blk):
    return {"value_head": {"w": mx.array(vh, dtype=mx.float32)},
            "block": {"w": mx.array(blk, dtype=mx.float32)}}


def test_projection_conflict_subtracts():
    # g_A and g_G anti-parallel on value_head -> dot<0 -> correct g_total.
    surf_total = _surf([1.0, 0.0], [0.0])
    surf_A = _surf([1.0, 0.0], [0.0])
    surf_G = _surf([-1.0, 0.0], [0.0])
    out, telem = project_conflicting_gradient(surf_total, surf_A, surf_G, weight=0.01)
    assert telem["conflict"] is True and telem["skip_reason"] is None
    assert telem["c"] == pytest.approx(-1.0, abs=1e-6)         # dot/(normsq+eps)
    # g_final = total - weight*c*G = [1,0] - 0.01*(-1)*[-1,0] = [0.99, 0]
    assert float(out["value_head"]["w"][0].item()) == pytest.approx(0.99, abs=1e-5)
    assert telem["removed_norm"] == pytest.approx(0.01, abs=1e-6)   # |w*c|*norm_G


def test_projection_no_conflict_unchanged_and_telemetry():
    # dot>=0 -> no correction, g_final IS g_total, telemetry counts non-conflict.
    surf_total = _surf([1.0, 2.0], [3.0])
    surf_A = _surf([1.0, 0.0], [0.0])
    surf_G = _surf([1.0, 0.0], [0.0])                          # dot = +1
    out, telem = project_conflicting_gradient(surf_total, surf_A, surf_G, weight=0.01)
    assert telem["evaluated"] is True and telem["conflict"] is False
    assert telem["skip_reason"] is None                       # a genuine no-conflict
    assert telem["c"] == 0.0 and telem["removed_norm"] == 0.0
    assert out is surf_total                                   # unchanged object


def test_projection_tiny_guardrail_skipped():
    # anti-parallel but ||g_G|| below eps -> skip (tiny_guardrail), unchanged.
    surf_total = _surf([1.0], [0.0])
    surf_A = _surf([1.0], [0.0])
    surf_G = _surf([-1e-12], [0.0])                            # dot<0 but norm_G<eps
    out, telem = project_conflicting_gradient(surf_total, surf_A, surf_G, weight=0.01)
    assert telem["conflict"] is False and telem["skip_reason"] == "tiny_guardrail"
    assert telem["c"] == 0.0 and out is surf_total


def _row(to_move, target_black):
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move=to_move, legal_moves=[(0, 0), (1, 1)], visit_counts=[0, 0],
        outcome=target_in_to_move(to_move, target_black), active_size=24,
        ply=20, game_n_moves=None)


def test_component_loss_masks_by_sign():
    # a_correction ignores guardrail rows; guardrail_hinge ignores A rows.
    # Frozen/eval BN makes the per-row value batch-independent, so the mixed-batch
    # component equals the solo-batch component exactly.
    net = create_network(hidden=64, n_blocks=2)
    freeze_batchnorm_running_stats(net)
    A = _row("black", -0.35)                                   # sign 0 (hard_value)
    G = _row("red", -0.9)                                      # sign +... guardrail
    sign_ag = np.array([0.0, 1.0], dtype=np.float32)
    la_mixed = _calibration_component_loss(net, [A, G], None, sign_ag, 0.10, "a_correction")
    la_solo = _calibration_component_loss(net, [A], None, np.array([0.0], np.float32), 0.10, "a_correction")
    assert float(la_mixed.item()) == pytest.approx(float(la_solo.item()), abs=1e-5)
    lg_mixed = _calibration_component_loss(net, [A, G], None, sign_ag, 0.10, "guardrail_hinge")
    lg_solo = _calibration_component_loss(net, [G], None, np.array([1.0], np.float32), 0.10, "guardrail_hinge")
    assert float(lg_mixed.item()) == pytest.approx(float(lg_solo.item()), abs=1e-5)


def test_component_loss_rejects_unknown_component():
    net = create_network(hidden=64, n_blocks=2)
    freeze_batchnorm_running_stats(net)
    with pytest.raises(ValueError, match="component"):
        _calibration_component_loss(net, [_row("black", -0.35)], None,
                                    np.array([0.0], np.float32), 0.10, "bogus")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_gradient_projection.py -v`
Expected: `ImportError` on `project_conflicting_gradient` / `_calibration_component_loss`.

- [ ] **Step 3: Implement the three helpers**

In `scripts/GPU/alphazero/trainer.py`, locate `def clip_grad_norm(` and add ABOVE it (they are used by `train_step` below `clip_grad_norm`):

```python
def _tree_dot(a, b):
    """Sum of elementwise products over matching leaves of two aligned pytrees."""
    if isinstance(a, mx.array):
        return mx.sum(a.astype(mx.float32) * b.astype(mx.float32))
    if isinstance(a, dict):
        acc = mx.array(0.0, dtype=mx.float32)
        for k in a:
            acc = acc + _tree_dot(a[k], b[k])
        return acc
    if isinstance(a, (list, tuple)):
        acc = mx.array(0.0, dtype=mx.float32)
        for x, y in zip(a, b):
            acc = acc + _tree_dot(x, y)
        return acc
    return mx.array(0.0, dtype=mx.float32)          # None / non-array leaf


def _tree_axpy(total, coef: float, g):
    """Leaf-wise total + coef*g over aligned pytrees (coef a Python float)."""
    if isinstance(total, mx.array):
        return total + coef * g
    if isinstance(total, dict):
        return {k: _tree_axpy(total[k], coef, g[k]) for k in total}
    if isinstance(total, (list, tuple)):
        return type(total)(_tree_axpy(x, coef, y) for x, y in zip(total, g))
    return total                                    # None / non-array leaf


def project_conflicting_gradient(surf_total, surf_A, surf_G, weight: float,
                                 eps: float = 1e-8):
    """Asymmetric, conflict-only projection of A away from G on the applied
    surface. surf_* are aligned pytrees (dict/list/mx.array) over the trainable
    surface (value_head + final block). Returns (surf_final, telem).

    surf_final = surf_total - weight * c * surf_G   when dot(A,G) < 0 and
    ||G|| > eps, with c = dot/(||G||**2 + 1e-12); otherwise surf_final IS
    surf_total (same object, elementwise unchanged) and c = 0.
    """
    dot = float(_tree_dot(surf_A, surf_G).item())
    normsq = float(_tree_dot(surf_G, surf_G).item())
    norm_G = normsq ** 0.5
    norm_A = float(_tree_dot(surf_A, surf_A).item()) ** 0.5
    if norm_G <= eps:
        skip_reason = "tiny_guardrail"
        conflict = False
    elif dot >= 0.0:
        skip_reason = None
        conflict = False
    else:
        skip_reason = None
        conflict = True
    c = dot / (normsq + 1e-12) if conflict else 0.0
    surf_final = _tree_axpy(surf_total, -weight * c, surf_G) if conflict else surf_total
    telem = {
        "evaluated": True, "conflict": conflict, "skip_reason": skip_reason,
        "dot": dot, "cos": dot / (norm_A * norm_G + 1e-12),
        "c": c, "removed_norm": abs(weight * c) * norm_G,
        "norm_G": norm_G, "norm_A": norm_A,
    }
    return surf_final, telem


def _calibration_component_loss(model, calibration_positions, calibration_weights,
                                calibration_guardrail_sign, guardrail_margin,
                                component: str, max_moves_cap: int = 512):
    """Eval-mode calibration-ONLY component scalar (no self-play, no policy CE).
    Reproduces alphazero_loss_batch's guardrail-branch value_term / hinge, split
    by the guardrail-sign mask. component in {"a_correction","guardrail_hinge"}."""
    cb_boards, cb_rows, cb_cols, cb_mask, _cb_pi, cb_targets = make_padded_batch(
        calibration_positions, max_moves_cap=max_moves_cap)
    _prev = model.training
    model.eval()
    try:
        _, cb_values, _ = model.forward_padded(
            cb_boards, cb_rows, cb_cols, cb_mask,
            active_size=calibration_positions[0].active_size)
    finally:
        model.train(_prev)
    per_value = (cb_values - cb_targets) ** 2
    base_w = (mx.reshape(mx.array(calibration_weights), per_value.shape)
              if calibration_weights is not None else mx.ones(per_value.shape))
    gmask = mx.abs(mx.reshape(mx.array(calibration_guardrail_sign), per_value.shape))
    if component == "a_correction":
        ng_w = base_w * (1.0 - gmask)
        return mx.sum(ng_w * per_value) / mx.maximum(mx.sum(ng_w), 1e-8)
    if component == "guardrail_hinge":
        sign = mx.reshape(mx.array(calibration_guardrail_sign), per_value.shape)
        signed_over = sign * (cb_values - cb_targets) - guardrail_margin
        hinge = mx.maximum(signed_over, 0.0) ** 2
        g_w = base_w * gmask
        return mx.sum(g_w * hinge) / mx.maximum(mx.sum(g_w), 1e-8)
    raise ValueError(f"unknown component {component!r} "
                     "(expected 'a_correction' or 'guardrail_hinge')")
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_gradient_projection.py -v`
Expected: ALL PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py tests/test_gradient_projection.py
git commit -m "feat(training): v13 gradient-projection + calibration-component-loss helpers

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `train_step` projection wiring + guard + 14-tuple

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` (`train_step` signature + projection block + guardrail return)
- Test: `tests/test_v13_projection_wiring.py` (create)

**Interfaces:**
- Consumes: `project_conflicting_gradient`, `_calibration_component_loss` (Task 1); existing `train_step`.
- Produces: `train_step(..., post_opening_calibration_gradient_projection: bool = False)`; when projection is active (flag on AND `calib_active` AND `guardrail_mode`), the guardrail return is a **14-tuple** = the 13-tuple + a projection-telemetry dict; otherwise the 13-tuple is unchanged. Task 4 consumes the dict.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_v13_projection_wiring.py`:

```python
"""v13 train_step wiring: projection off is byte-identical (13-tuple); on with a
mixed A+guardrail batch produces a 14-tuple with projection telemetry; the
value-head-only surface is rejected."""
import numpy as np
import pytest
import mlx.optimizers as optim

from scripts.GPU.alphazero.trainer import (
    MainModule, freeze_batchnorm_running_stats, train_step)
from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord
from scripts.GPU.alphazero.calibration_pool import target_in_to_move


def _pos(to_move="red", outcome=1.0):
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move=to_move, legal_moves=[(0, 0), (1, 1), (2, 2)],
        visit_counts=[10, 5, 3], outcome=outcome, active_size=24,
        ply=0, game_n_moves=10)


def _row(to_move, target_black):
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move=to_move, legal_moves=[(0, 0), (1, 1)], visit_counts=[0, 0],
        outcome=target_in_to_move(to_move, target_black), active_size=24,
        ply=20, game_n_moves=None)


def _setup():
    net = create_network(hidden=64, n_blocks=2)
    freeze_batchnorm_running_stats(net)
    mm = MainModule(net.encoder, net.policy_head)
    return net, mm, optim.Adam(learning_rate=1e-3), optim.Adam(learning_rate=1e-3)


# A row (sign 0) + guardrail row (sign +1)
_CALIB = [_row("black", -0.35), _row("red", -0.9)]
_SIGN = np.array([0.0, 1.0], dtype=np.float32)


def _run(projection, **kw):
    net, mm, om, ov = _setup()
    return train_step(network=net, main_module=mm, opt_main=om, opt_value=ov,
                      batch=[_pos() for _ in range(3)],
                      calibration_positions=_CALIB, calibration_loss_weight=0.01,
                      calibration_guardrail_sign=_SIGN, guardrail_margin=0.10,
                      train_value_head_and_final_block=True,
                      post_opening_calibration_gradient_projection=projection, **kw)


def test_projection_off_is_13_tuple():
    out = _run(projection=False)
    assert len(out) == 13                     # byte-identical guardrail path


def test_projection_on_appends_telemetry_dict():
    out = _run(projection=True)
    assert len(out) == 14
    telem = out[13]
    assert isinstance(telem, dict)
    assert set(telem) >= {"evaluated", "conflict", "skip_reason", "dot", "cos",
                          "c", "removed_norm", "norm_G", "norm_A"}


def test_projection_requires_final_block_surface():
    net, mm, om, ov = _setup()
    with pytest.raises(ValueError, match="train-value-head-and-final-block"):
        train_step(network=net, main_module=mm, opt_main=om, opt_value=ov,
                   batch=[_pos() for _ in range(3)],
                   calibration_positions=_CALIB, calibration_loss_weight=0.01,
                   calibration_guardrail_sign=_SIGN, guardrail_margin=0.10,
                   train_value_head_only=True,
                   post_opening_calibration_gradient_projection=True)


def test_projection_no_guardrail_rows_skips():
    # all-A batch (sign all 0) -> no guardrail rows -> skip_reason no_guardrail
    net, mm, om, ov = _setup()
    out = train_step(network=net, main_module=mm, opt_main=om, opt_value=ov,
                     batch=[_pos() for _ in range(3)],
                     calibration_positions=[_row("black", -0.35)],
                     calibration_loss_weight=0.01,
                     calibration_guardrail_sign=np.array([0.0], np.float32),
                     guardrail_margin=0.10, train_value_head_and_final_block=True,
                     post_opening_calibration_gradient_projection=True)
    assert len(out) == 14
    assert out[13]["evaluated"] is False and out[13]["skip_reason"] == "no_guardrail"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_v13_projection_wiring.py -v`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'post_opening_calibration_gradient_projection'`.

- [ ] **Step 3: Implement**

**(a) signature.** Locate `train_value_head_and_final_block: bool = False,   # v9: only value head + final block` in the `def train_step(` params and, near it (after the v12 guardrail params `calibration_guardrail_sign` / `guardrail_margin`), add:

```python
    post_opening_calibration_gradient_projection: bool = False,   # v13
```

**(b) projection block.** Locate the unpack chain ending near `else:\n        total_loss, policy_loss, value_loss, l2_loss, aux_loss, aux_coverage, aux_n_eligible = loss_tuple` (just above `# Slice GRADS only`). Insert, directly after that unpack chain and BEFORE `main_grads = {`:

```python
    # v13: asymmetric A-yields-to-guardrail projection on the applied surface.
    _proj_telem = None
    if post_opening_calibration_gradient_projection:
        if train_value_head_only:
            raise ValueError(
                "post_opening_calibration_gradient_projection requires "
                "--train-value-head-and-final-block (the value-head-only surface "
                "does not define the A-vs-guardrail final-block conflict)")
        if calib_active and calibration_guardrail_sign is not None:
            _sgn = np.asarray(calibration_guardrail_sign)
            _has_a = bool((np.abs(_sgn) < 0.5).any())
            _has_g = bool((np.abs(_sgn) > 0.5).any())
            if not _has_a:
                _proj_telem = {"evaluated": False, "conflict": False,
                               "skip_reason": "no_a", "dot": 0.0, "cos": 0.0,
                               "c": 0.0, "removed_norm": 0.0, "norm_G": 0.0,
                               "norm_A": 0.0}
            elif not _has_g:
                _proj_telem = {"evaluated": False, "conflict": False,
                               "skip_reason": "no_guardrail", "dot": 0.0,
                               "cos": 0.0, "c": 0.0, "removed_norm": 0.0,
                               "norm_G": 0.0, "norm_A": 0.0}
            else:
                _last = len(network.encoder.blocks) - 1

                def _a_fn(m):
                    return _calibration_component_loss(
                        m, calibration_positions, calibration_weights,
                        calibration_guardrail_sign, guardrail_margin,
                        "a_correction", max_moves_cap)

                def _g_fn(m):
                    return _calibration_component_loss(
                        m, calibration_positions, calibration_weights,
                        calibration_guardrail_sign, guardrail_margin,
                        "guardrail_hinge", max_moves_cap)

                _, _g_a = nn.value_and_grad(network, _a_fn)(network)
                _, _g_g = nn.value_and_grad(network, _g_fn)(network)
                _surf_total = {"value_head": grads["value_head"],
                               "block": grads["encoder"]["blocks"][_last]}
                _surf_a = {"value_head": _g_a["value_head"],
                           "block": _g_a["encoder"]["blocks"][_last]}
                _surf_g = {"value_head": _g_g["value_head"],
                           "block": _g_g["encoder"]["blocks"][_last]}
                _surf_final, _proj_telem = project_conflicting_gradient(
                    _surf_total, _surf_a, _surf_g, weight=calibration_loss_weight)
                grads["value_head"] = _surf_final["value_head"]
                grads["encoder"]["blocks"][_last] = _surf_final["block"]
```

(The subsequent `main_grads = {...}` / `value_grads = grads["value_head"]` read the corrected `grads`, so clipping + the v9-guarded apply operate on the projected gradient. When `_proj_telem` conflict is False, `grads` is unchanged.)

**(c) 14-tuple return.** Locate the guardrail return branch `if calib_active and guardrail_mode:` in the RETURN section (the `return (float(total_loss.item()), ..., int(guardrail_n),)` 13-tuple). Bind it to a name and append the telemetry when present:

```python
    if calib_active and guardrail_mode:
        _guard_ret = (
            float(total_loss.item()), float(policy_loss.item()),
            float(value_loss.item()), float(l2_loss.item()),
            float(aux_loss.item()), float(aux_coverage), int(aux_n_eligible),
            float(calib_loss.item()), float(calib_value_mean.item()), int(calib_n),
            float(guardrail_hinge_loss.item()), float(guardrail_active_frac.item()),
            int(guardrail_n),
        )
        if _proj_telem is not None:
            return _guard_ret + (_proj_telem,)
        return _guard_ret
```

- [ ] **Step 4: Run wiring tests + trainer regression**

Run: `.venv/bin/python -m pytest tests/test_v13_projection_wiring.py tests/test_asymmetric_guardrail_loss.py tests/test_calibration_loss.py tests/test_train_value_head_and_final_block.py tests/test_gradient_projection.py -v`
Expected: ALL PASS (projection off → 13-tuple byte-identical; v12 guardrail + v9 tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py tests/test_v13_projection_wiring.py
git commit -m "feat(training): v13 train_step projection block + 14-tuple + value-head-only guard

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: CLI flag `--post-opening-calibration-gradient-projection`

**Files:**
- Modify: `scripts/GPU/alphazero/train.py` (arg + plumb), `scripts/GPU/alphazero/trainer.py` (`train()` signature + `train_step(...)` call sites)
- Test: append to `tests/test_v13_projection_wiring.py`

**Interfaces:**
- Consumes: the `train_step` param (Task 2).
- Produces: `--post-opening-calibration-gradient-projection` → `train(post_opening_calibration_gradient_projection=...)` → both `train_step(...)` call sites.

- [ ] **Step 1: Write the failing (source-level) test**

Append to `tests/test_v13_projection_wiring.py`:

```python
def test_cli_projection_flag_and_plumb():
    from scripts.GPU.alphazero import train as train_mod
    from scripts.GPU.alphazero import trainer as trainer_mod
    tsrc = open(train_mod.__file__).read()
    assert '"--post-opening-calibration-gradient-projection"' in tsrc
    assert ("post_opening_calibration_gradient_projection="
            "args.post_opening_calibration_gradient_projection,") in tsrc
    rsrc = open(trainer_mod.__file__).read()
    # forwarded to train_step at the calibration call site(s)
    assert ("post_opening_calibration_gradient_projection="
            "post_opening_calibration_gradient_projection,") in rsrc
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_v13_projection_wiring.py::test_cli_projection_flag_and_plumb -v`
Expected: FAIL on the missing strings.

- [ ] **Step 3: Implement**

**(a) `train.py` arg.** Locate `parser.add_argument("--guardrail-margin",` and add a sibling after its full definition:

```python
    parser.add_argument("--post-opening-calibration-gradient-projection",
        action="store_true",
        help="v13: project the A-correction gradient away from the guardrail "
             "hinge gradient on the value_head + final-block surface when they "
             "conflict (dot<0). Requires --train-value-head-and-final-block. "
             "Off by default; byte-identical to v12b when off.")
```

**(b) `train.py` plumb.** Locate `post_opening_guardrail_margin=args.guardrail_margin,` in the `train(...)` call and add below it:

```python
        post_opening_calibration_gradient_projection=args.post_opening_calibration_gradient_projection,
```

**(c) `trainer.py` `train()` signature.** Locate `post_opening_guardrail_margin: float = 0.1,` in the `def train(` params and add below it:

```python
    post_opening_calibration_gradient_projection: bool = False,   # v13
```

**(d) `trainer.py` call sites.** There are TWO `train_step(...)` calls in `train()` (the calibration branch and the non-calibration branch — grep `main_module=main_module,`). In the CALIBRATION-branch call (the one that already passes `calibration_guardrail_sign=_calib_guard_sign,` and `guardrail_margin=post_opening_guardrail_margin,`), add below `guardrail_margin=post_opening_guardrail_margin,`:

```python
                                post_opening_calibration_gradient_projection=post_opening_calibration_gradient_projection,
```

- [ ] **Step 4: Run test + CLI regression**

Run: `.venv/bin/python -m pytest tests/test_v13_projection_wiring.py tests/test_calibration_cli_flags.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/train.py scripts/GPU/alphazero/trainer.py tests/test_v13_projection_wiring.py
git commit -m "feat(training): v13 --post-opening-calibration-gradient-projection CLI + plumb

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Projection telemetry accumulation + JSON

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` (train-loop accumulators + telemetry accumulation), `scripts/GPU/alphazero/calibration_pool.py` (`build_post_opening_calibration_block` JSON keys)
- Test: append to `tests/test_v13_projection_wiring.py`

**Interfaces:**
- Consumes: the 14-tuple `train_step` return (Task 2); `post_opening_calibration_gradient_projection` (Task 3).
- Produces: JSON telemetry keys `calib_projection_enabled`, `calib_projection_scope`, `calib_projection_conflict_steps`, `calib_projection_conflict_rate`, `calib_projection_dot_avg`, `calib_projection_cos_avg`, `calib_projection_c_avg`, `calib_projection_removed_norm_avg`, `calib_projection_guardrail_grad_norm_avg`, `calib_projection_a_grad_norm_avg`, and the skip counters `calib_projection_no_a_steps`, `calib_projection_no_guardrail_steps`, `calib_projection_tiny_guardrail_steps`, `calib_projection_no_conflict_steps`.

- [ ] **Step 1: Write the failing (source-level) test**

Append to `tests/test_v13_projection_wiring.py`:

```python
def test_projection_telemetry_accumulation_and_json():
    from scripts.GPU.alphazero import trainer as trainer_mod
    from scripts.GPU.alphazero import calibration_pool as cp_mod
    rsrc = open(trainer_mod.__file__).read()
    # accumulators + the 14-tuple telemetry read
    assert "sum_proj_dot" in rsrc
    assert "proj_conflict_steps" in rsrc
    assert "len(_ret) == 14" in rsrc
    assert 'proj["skip_reason"]' in rsrc
    csrc = open(cp_mod.__file__).read()
    for k in ('"calib_projection_enabled"', '"calib_projection_conflict_steps"',
              '"calib_projection_conflict_rate"', '"calib_projection_dot_avg"',
              '"calib_projection_cos_avg"', '"calib_projection_c_avg"',
              '"calib_projection_removed_norm_avg"',
              '"calib_projection_guardrail_grad_norm_avg"',
              '"calib_projection_a_grad_norm_avg"',
              '"calib_projection_no_a_steps"', '"calib_projection_no_guardrail_steps"',
              '"calib_projection_tiny_guardrail_steps"',
              '"calib_projection_no_conflict_steps"', '"calib_projection_scope"'):
        assert k in csrc, k
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_v13_projection_wiring.py::test_projection_telemetry_accumulation_and_json -v`
Expected: FAIL on the missing strings.

- [ ] **Step 3: Implement**

**(a) accumulator init.** Locate `sum_guardrail_active_frac = 0.0` (the v12 calibration accumulator init) and add below it:

```python
                proj_conflict_steps = 0
                proj_no_a_steps = 0
                proj_no_guardrail_steps = 0
                proj_tiny_guardrail_steps = 0
                proj_no_conflict_steps = 0
                sum_proj_dot = 0.0
                sum_proj_cos = 0.0
                sum_proj_c = 0.0
                sum_proj_removed_norm = 0.0
                sum_proj_norm_g = 0.0
                sum_proj_norm_a = 0.0
```

**(b) accumulation.** Locate the v12 guardrail telemetry accumulation `if _calib_guard_sign is not None and len(_ret) == 13:` block and add a sibling directly after it (a projection step returns a 14-tuple):

```python
                            if (_calib_guard_sign is not None
                                    and len(_ret) == 14):
                                sum_guardrail_hinge_loss += _ret[10]
                                sum_guardrail_active_frac += _ret[11]
                                proj = _ret[13]
                                if proj["skip_reason"] == "no_a":
                                    proj_no_a_steps += 1
                                elif proj["skip_reason"] == "no_guardrail":
                                    proj_no_guardrail_steps += 1
                                elif proj["skip_reason"] == "tiny_guardrail":
                                    proj_tiny_guardrail_steps += 1
                                elif proj["conflict"]:
                                    proj_conflict_steps += 1
                                    sum_proj_dot += proj["dot"]
                                    sum_proj_cos += proj["cos"]
                                    sum_proj_c += proj["c"]
                                    sum_proj_removed_norm += proj["removed_norm"]
                                    sum_proj_norm_g += proj["norm_G"]
                                    sum_proj_norm_a += proj["norm_A"]
                                else:
                                    proj_no_conflict_steps += 1
                                    sum_proj_dot += proj["dot"]
                                    sum_proj_cos += proj["cos"]
                                    sum_proj_norm_g += proj["norm_G"]
                                    sum_proj_norm_a += proj["norm_A"]
```

(Note: the guardrail-hinge/active-frac accumulation is duplicated into this 14-tuple sibling so a projection step still contributes its guardrail telemetry. `dot`/`cos`/`norm` averages accumulate over *evaluatable* steps = conflict + no_conflict; `c`/`removed_norm` accumulate over conflict steps only.)

**(c) `loss_accumulator` dict.** Locate where `train()` builds the `loss_accumulator` dict passed to `build_post_opening_calibration_block` (the `"sum_guardrail_active_frac": sum_guardrail_active_frac,` entry) and add below it:

```python
                        "proj_enabled": post_opening_calibration_gradient_projection,
                        "proj_conflict_steps": proj_conflict_steps,
                        "proj_no_a_steps": proj_no_a_steps,
                        "proj_no_guardrail_steps": proj_no_guardrail_steps,
                        "proj_tiny_guardrail_steps": proj_tiny_guardrail_steps,
                        "proj_no_conflict_steps": proj_no_conflict_steps,
                        "sum_proj_dot": sum_proj_dot,
                        "sum_proj_cos": sum_proj_cos,
                        "sum_proj_c": sum_proj_c,
                        "sum_proj_removed_norm": sum_proj_removed_norm,
                        "sum_proj_norm_g": sum_proj_norm_g,
                        "sum_proj_norm_a": sum_proj_norm_a,
```

**(d) JSON block.** In `build_post_opening_calibration_block` (calibration_pool.py), locate `"guardrail_margin":` in the `"loss"` dict and add after that entry:

```python
            "calib_projection_enabled":
                bool(loss_accumulator.get("proj_enabled", False)),
            "calib_projection_scope": "value_head_and_final_block",
            "calib_projection_conflict_steps":
                int(loss_accumulator.get("proj_conflict_steps", 0)),
            "calib_projection_no_a_steps":
                int(loss_accumulator.get("proj_no_a_steps", 0)),
            "calib_projection_no_guardrail_steps":
                int(loss_accumulator.get("proj_no_guardrail_steps", 0)),
            "calib_projection_tiny_guardrail_steps":
                int(loss_accumulator.get("proj_tiny_guardrail_steps", 0)),
            "calib_projection_no_conflict_steps":
                int(loss_accumulator.get("proj_no_conflict_steps", 0)),
            "calib_projection_conflict_rate": (
                int(loss_accumulator.get("proj_conflict_steps", 0))
                / max(int(loss_accumulator.get("proj_conflict_steps", 0))
                      + int(loss_accumulator.get("proj_no_conflict_steps", 0)), 1)),
            "calib_projection_dot_avg": (
                float(loss_accumulator.get("sum_proj_dot", 0.0))
                / max(int(loss_accumulator.get("proj_conflict_steps", 0))
                      + int(loss_accumulator.get("proj_no_conflict_steps", 0)), 1)),
            "calib_projection_cos_avg": (
                float(loss_accumulator.get("sum_proj_cos", 0.0))
                / max(int(loss_accumulator.get("proj_conflict_steps", 0))
                      + int(loss_accumulator.get("proj_no_conflict_steps", 0)), 1)),
            "calib_projection_c_avg": (
                float(loss_accumulator.get("sum_proj_c", 0.0))
                / max(int(loss_accumulator.get("proj_conflict_steps", 0)), 1)),
            "calib_projection_removed_norm_avg": (
                float(loss_accumulator.get("sum_proj_removed_norm", 0.0))
                / max(int(loss_accumulator.get("proj_conflict_steps", 0)), 1)),
            "calib_projection_guardrail_grad_norm_avg": (
                float(loss_accumulator.get("sum_proj_norm_g", 0.0))
                / max(int(loss_accumulator.get("proj_conflict_steps", 0))
                      + int(loss_accumulator.get("proj_no_conflict_steps", 0)), 1)),
            "calib_projection_a_grad_norm_avg": (
                float(loss_accumulator.get("sum_proj_norm_a", 0.0))
                / max(int(loss_accumulator.get("proj_conflict_steps", 0))
                      + int(loss_accumulator.get("proj_no_conflict_steps", 0)), 1)),
```

- [ ] **Step 4: Run test + telemetry regression**

Run: `.venv/bin/python -m pytest tests/test_v13_projection_wiring.py tests/test_value_calibration_sampling.py tests/test_calibration_pool.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py scripts/GPU/alphazero/calibration_pool.py tests/test_v13_projection_wiring.py
git commit -m "feat(training): v13 projection telemetry (core averages + skip counters) + JSON

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: gate-0 smoke + full-suite verification + merge handoff (controller-run)

**Files:**
- Create: `scripts/GPU/alphazero/smoke_v13_gradient_projection.py`

- [ ] **Step 1: Write the smoke** (mirrors `smoke_v12b_continuation_guardrail.py`; runs a projection step and asserts the telemetry dict)

```python
#!/usr/bin/env python3
"""Gate-0 smoke: draw the v12b schedule, run a train_step with the v13 gradient
projection on the v9 surface, and assert the projection telemetry dict engaged.

Run as a module: .venv/bin/python -m scripts.GPU.alphazero.smoke_v13_gradient_projection
"""
import sys
import numpy as np
import mlx.optimizers as optim

from scripts.GPU.alphazero.calibration_pool import (
    CalibrationPool, split_samples_with_guardrail, GUARDRAIL_LOSS_MODE)
from scripts.GPU.alphazero.trainer import (
    MainModule, freeze_batchnorm_running_stats, train_step)
from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord

MANIFEST = "logs/eval/targeted_calibration_v12b_continuation_guardrail_from_calib020_0001.csv"
SCHEDULE = {"black_predrop_correction": 2, "goal_line_guardrail_retention": 1,
            "old_post_opening_guardrail_retention": 1,
            "old_post_opening_continuation_guardrail_retention": 2,
            "red_predrop_guardrail_retention": 1,
            "red_predrop_continuation_guardrail_retention": 2}


def main() -> int:
    pool = CalibrationPool.from_manifest(MANIFEST, calibration_target=-0.35)
    assert pool.schema == GUARDRAIL_LOSS_MODE, pool.schema
    import random
    rng = random.Random(0)
    samples = pool.sample_by_tag(SCHEDULE, rng)
    records, weights, sign = split_samples_with_guardrail(samples, pool.has_weight_scale)
    assert (np.abs(sign) < 0.5).any() and (np.abs(sign) > 0.5).any(), "need A + guardrail rows"
    net = create_network(hidden=64, n_blocks=2)
    freeze_batchnorm_running_stats(net)
    mm = MainModule(net.encoder, net.policy_head)
    om, ov = optim.Adam(learning_rate=1e-3), optim.Adam(learning_rate=1e-3)
    def _p():
        return PositionRecord(board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
                              to_move="red", legal_moves=[(0, 0), (1, 1)],
                              visit_counts=[1, 1], outcome=1.0, active_size=24,
                              ply=0, game_n_moves=10)
    out = train_step(network=net, main_module=mm, opt_main=om, opt_value=ov,
                     batch=[_p() for _ in range(4)], calibration_positions=records,
                     calibration_weights=weights, calibration_loss_weight=0.01,
                     calibration_guardrail_sign=sign, guardrail_margin=0.10,
                     train_value_head_and_final_block=True,
                     post_opening_calibration_gradient_projection=True)
    assert len(out) == 14, len(out)
    proj = out[13]
    assert set(proj) >= {
        "evaluated", "conflict", "skip_reason", "dot", "cos",
        "c", "removed_norm", "norm_G", "norm_A",
    }
    # tiny_guardrail is a legitimate no-op; no_a/no_guardrail shouldn't happen
    # here (both sign classes were asserted present) but are allowed for debugging.
    assert proj["skip_reason"] in (None, "tiny_guardrail", "no_a", "no_guardrail")
    print(
        f"SMOKE PASS: evaluated={proj['evaluated']} conflict={proj['conflict']} "
        f"skip={proj['skip_reason']} dot={proj['dot']:.4g} "
        f"cos={proj['cos']:.3g} c={proj['c']:.4g} "
        f"removed_norm={proj['removed_norm']:.4g}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Verify it imports (`.venv/bin/python -c "import scripts.GPU.alphazero.smoke_v13_gradient_projection"`); RUN it as a module only if the v12b manifest + local replays are present, else note the deferral to the operator box. Commit:
```bash
git add scripts/GPU/alphazero/smoke_v13_gradient_projection.py
git commit -m "test(training): v13 gate-0 gradient-projection smoke

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 2: Full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected in the worktree: baseline-passed + the new v13 tests, with EXACTLY the known 14 failed + 6 errors (missing gitignored data). Authoritative check (1355 + new v13 tests, 0 failures) on merged main before push.

- [ ] **Step 3: Hand off to merge**

FF-merge to main, authoritative suite on merged main, push (superpowers:finishing-a-development-branch). STOP after push — the operator run is the USER's; the exact command block is in the spec's Operator-run section.

---

## Operator run (USER's, after merge) — from the spec

The same canonical v12b command (full harness + v12b manifest + schedule + `--guardrail-margin 0.10 --freeze-batchnorm-stats --train-value-head-and-final-block`) **plus `--post-opening-calibration-gradient-projection`**, new checkpoint dir. Confirm the projection telemetry engaged (`calib_projection_scope=value_head_and_final_block`; nonzero conflict/no-conflict steps); `verify_value_head_and_final_block_checkpoint` exit 0; gates A/B/C/D vs `calib020_0001`. No promotion unless all four pass. Read `conflict_rate ≈ 0` as "the bind is not a per-step surface gradient conflict — projection is the wrong lever."
