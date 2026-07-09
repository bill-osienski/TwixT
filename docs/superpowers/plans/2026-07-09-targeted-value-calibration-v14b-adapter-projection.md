# Targeted Value Calibration v14b — Gradient Projection over the Value-Adapter Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the existing v13 A-yields-to-guardrail gradient projection run on v14's value-only adapter surface `{value_head, value_adapter}`, so the projection flag composes with `--train-value-head-and-value-adapter` instead of being rejected.

**Architecture:** Three surgical `train_step` edits (relax the projection guard for the adapter surface; select the projected surface by mode; fold the post-projection `value_adapter_grad_norm` into the projection telemetry dict) plus one `train()` accumulator edit (branch slot `[13]` by `isinstance(dict)`, not by the v14 flag). `project_conflicting_gradient` stays surface-agnostic and untouched; the optimizer routing, the verifier, and `train.py` are unchanged. Slot `[13]` becomes self-describing: dict ⇒ projection (v13/v14b), float ⇒ v14 grad-norm only, absent ⇒ plain guardrail.

**Tech Stack:** Python 3.14 / MLX, pytest.

**Spec:** `docs/superpowers/specs/2026-07-09-targeted-value-calibration-v14b-adapter-projection-design.md` (APPROVED — do not redesign).

## Global Constraints

- Python: always `.venv/bin/python`; tests `.venv/bin/python -m pytest <file> -v`; full-suite baseline on merged main = **1394 passed** (post v14 + probe_eval fix).
- NEVER `sys.modules.pop("mlx")` in tests.
- **No new flag.** v14b = `--value-adapter --train-value-head-and-value-adapter --post-opening-calibration-gradient-projection` (+ optional `--post-opening-calibration-projection-strength N`), with `--guardrail-margin 0.10 --freeze-batchnorm-stats`. `train.py` is UNCHANGED (no guard ties projection to the final block; the three-surface mutual exclusion stays).
- **`project_conflicting_gradient` is surface-agnostic and MUST NOT be modified.** `train_step` chooses which trainable surface to present to it. v13/v9 surface = `{value_head, encoder.blocks[last]}`; v14b surface = `{value_head, value_adapter}`. Projection math (conflict-only gate, `c = dot/(normsq+1e-12)`, `removed_norm`, geometry) is unchanged; margin stays 0.10.
- **Slot `[13]` self-describing:** dict ⇒ projection telemetry (v13; and v14b, where the dict also carries `value_adapter_grad_norm`); float ⇒ adapter grad-norm only (v14); absent ⇒ plain guardrail. Return arity is UNCHANGED (still a 14-tuple under projection). `train()` branches by `isinstance(extra, dict)`, NOT by the v14 flag.
- **`value_adapter_grad_norm` under v14b is the POST-PROJECTION / applied adapter grad norm.** Do NOT add a pre-projection diagnostic (`value_adapter_grad_norm_pre_projection`) in this branch.
- **Two-site telemetry is INHERITED from v14/v13 — no new plumbing.** `value_adapter_grad_norm` reaches both JSON sites (sidecar `build_post_opening_calibration_block` + the flattened `_teacher_calib_scalars` mirror) via `sum_value_adapter_grad_norm` (now sourced from the dict in Step 6); `value_adapter_gate` via the existing direct `network.value_adapter.gate` read; the projection `calib_projection_*` metrics via the existing v13 accumulation. v14b adds no telemetry keys beyond folding `value_adapter_grad_norm` into the projection dict.
- **Byte-identical** when projection is off (v14 path intact) and when the adapter flags are off (v13/v9/v8 paths intact).
- **Do NOT change:** `project_conflicting_gradient`, `_calibration_component_loss`, `alphazero_loss_batch`, the v12b manifest/builder, the guardrail margin, the v12b schedule, `MainModule`, `ValueModule`, `verify_value_head_and_adapter_checkpoint.py` (a v14b checkpoint trains the same surface as v14 — the verifier already covers it), the v8/v9 verifiers, `mcts.py`, `continuation_extraction.py`, `train.py`.
- Worktree `feature/tvc-v14b-adapter-projection`; symlink `.venv`; FF-merge (no `--no-ff`, never force-push); code-commit trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; file-scoped `git add`; **locate code by content, not line numbers**. Fresh worktree lacks gitignored data → whole-repo suite there = 14 failed + 6 errors (analyzer/probe/strong_advantage fixtures); judge tasks file-scoped; authoritative suite on merged main.

## File Structure

| File | Role |
|---|---|
| `scripts/GPU/alphazero/trainer.py` (modify) | `train_step`: guard relax + surface selection + grad-norm fold; `train()`: type-based slot-`[13]` accumulator |
| `tests/test_v14b_adapter_projection.py` (create) | projection accepted on the adapter surface; self-describing slot + folded grad-norm; forced-conflict projection over `{value_head, value_adapter}`; byte-identical v14/v13 slot types |

**Task → work-item map:** T1 = the four trainer.py edits + tests. T2 = full suite + merge (controller-run).

---

### Task 1: Projection over the value-adapter surface (trainer.py)

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py`
- Test: `tests/test_v14b_adapter_projection.py` (create)

**Interfaces:**
- Consumes: existing `train_step(..., post_opening_calibration_gradient_projection=False, post_opening_calibration_projection_strength=1.0, train_value_head_and_value_adapter=False, value_module=None)`; `project_conflicting_gradient(surf_total, surf_A, surf_G, weight)` (unchanged); `ValueModule`, `MainModule`, `freeze_batchnorm_running_stats`, `create_network(value_adapter=True)`, `target_in_to_move`, `PositionRecord`.
- Produces: v14b behavior — projection permitted when `train_value_head_and_value_adapter` is set, operating on `{value_head, value_adapter}`; the guardrail 14-tuple slot `[13]` is a projection dict carrying `value_adapter_grad_norm` under v14b.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_v14b_adapter_projection.py`:

```python
"""v14b: gradient projection over the value-only adapter surface
{value_head, value_adapter}. Combines v13's A-yields-to-guardrail projection with
v14's isolated adapter surface. No new flag; slot [13] is self-describing."""
import numpy as np
import pytest
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord
from scripts.GPU.alphazero.calibration_pool import target_in_to_move
from scripts.GPU.alphazero.trainer import (
    MainModule, ValueModule, train_step, freeze_batchnorm_running_stats)


def _adapter_net():
    net = create_network(hidden=64, n_blocks=2, value_adapter=True)
    freeze_batchnorm_running_stats(net)
    return net


def _pos():
    return PositionRecord(board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
                          to_move="red", legal_moves=[(0, 0), (1, 1), (2, 2)],
                          visit_counts=[10, 5, 3], outcome=1.0, active_size=24,
                          ply=0, game_n_moves=10)


def _row(target_black):
    # Shared zero board so an A row and a guardrail row have collinear-up-to-sign
    # value gradients over ANY surface (the v13c forced-conflict construction).
    return PositionRecord(board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
                          to_move="black", legal_moves=[(0, 0), (1, 1)],
                          visit_counts=[0, 0], outcome=target_in_to_move("black", target_black),
                          active_size=24, ply=20, game_n_moves=None)


def _v14b_call(net, projection=True, strength=1.0):
    mm = MainModule(net.encoder, net.policy_head)
    vm = ValueModule(net.value_head, net.value_adapter)
    om, ov = optim.Adam(learning_rate=1e-3), optim.Adam(learning_rate=1e-3)
    # Forced A-vs-guardrail conflict: A row (sign 0) target +0.9 pulls value UP;
    # guardrail row (sign +1) target -0.9 -> hinge relu(v+0.8) active for v>-0.8
    # pushes DOWN -> dot < 0 on the shared zero board.
    calib = [_row(0.9), _row(-0.9)]
    sign = np.array([0.0, 1.0], dtype=np.float32)   # 0 = A row, 1 = guardrail row
    return train_step(
        network=net, main_module=mm, opt_main=om, opt_value=ov,
        batch=[_pos() for _ in range(3)], calibration_positions=calib,
        calibration_loss_weight=0.01, calibration_guardrail_sign=sign,
        guardrail_margin=0.10, train_value_head_and_value_adapter=True,
        value_module=vm,
        post_opening_calibration_gradient_projection=projection,
        post_opening_calibration_projection_strength=strength)


def test_projection_accepted_on_adapter_surface():
    # v14: this combo raised ValueError; v14b: it must run and return a 14-tuple.
    ret = _v14b_call(_adapter_net(), projection=True)
    assert isinstance(ret, tuple) and len(ret) == 14


def test_projection_still_rejected_on_value_head_only():
    net = _adapter_net()
    mm = MainModule(net.encoder, net.policy_head)
    with pytest.raises(ValueError, match="multi-component trainable surface"):
        train_step(network=net, main_module=mm,
                   opt_main=optim.Adam(learning_rate=1e-3),
                   opt_value=optim.Adam(learning_rate=1e-3),
                   batch=[_pos() for _ in range(3)],
                   calibration_positions=[_row(-0.9)],
                   calibration_loss_weight=0.01,
                   calibration_guardrail_sign=np.array([1.0], dtype=np.float32),
                   guardrail_margin=0.10,
                   train_value_head_only=True,
                   post_opening_calibration_gradient_projection=True)


def test_slot13_is_projection_dict_with_folded_grad_norm():
    mx.random.seed(0)
    extra = _v14b_call(_adapter_net(), projection=True)[13]
    assert isinstance(extra, dict)                              # self-describing: dict
    assert "conflict" in extra and "removed_norm" in extra     # projection telemetry
    assert "value_adapter_grad_norm" in extra                  # folded (post-projection)
    assert isinstance(extra["value_adapter_grad_norm"], float)
    assert extra["value_adapter_grad_norm"] >= 0.0


def test_forced_conflict_projects_on_adapter_surface():
    mx.random.seed(0)
    proj = _v14b_call(_adapter_net(), projection=True)[13]
    assert proj["conflict"] is True
    assert proj["c"] != 0.0
    assert proj["removed_norm"] > 0.0


def test_v14_float_slot_unchanged_when_projection_off():
    mx.random.seed(0)
    assert isinstance(_v14b_call(_adapter_net(), projection=False)[13], float)


def test_projection_changes_applied_value_side_update():
    # On the forced conflict, projection removes part of the A push -> the applied
    # value-side update differs from projection-off (same seed, same batch).
    mx.random.seed(0); net_on = _adapter_net()
    before = {k: np.array(v) for k, v in tree_flatten(net_on.parameters())}
    _v14b_call(net_on, projection=True)
    after_on = {k: np.array(v) for k, v in tree_flatten(net_on.parameters())}
    mx.random.seed(0); net_off = _adapter_net()
    _v14b_call(net_off, projection=False)
    after_off = {k: np.array(v) for k, v in tree_flatten(net_off.parameters())}
    vs = [k for k in after_on if k.startswith("value_head.") or k.startswith("value_adapter.")]
    assert any(not np.array_equal(before[k], after_on[k]) for k in vs)      # value side trained
    assert any(not np.array_equal(after_on[k], after_off[k]) for k in vs)   # projection changed it


def test_accumulator_branches_by_type_not_flag():
    # Pin the train() slot-[13] disambiguation: type-based (isinstance dict) with a
    # graceful .get so v13 (dict, no key) and v14b (dict + key) both work.
    from scripts.GPU.alphazero import trainer as trainer_mod
    src = open(trainer_mod.__file__).read()
    assert "isinstance(_extra, dict)" in src
    assert 'get("value_adapter_grad_norm", 0.0)' in src
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_v14b_adapter_projection.py -v`
Expected: FAIL — `test_projection_accepted_on_adapter_surface`, `test_slot13_*`, `test_forced_conflict_*`, and `test_projection_changes_*` raise `ValueError: post_opening_calibration_gradient_projection requires --train-value-head-and-final-block ...` (the current v14 guard rejects projection on the adapter surface); `test_accumulator_branches_by_type_not_flag` fails on the missing `isinstance(_extra, dict)` string.

- [ ] **Step 3: Relax the projection guard (§B)**

In `train_step`, locate:

```python
    if post_opening_calibration_gradient_projection:
        if train_value_head_only or train_value_head_and_value_adapter:
            raise ValueError(
                "post_opening_calibration_gradient_projection requires "
                "--train-value-head-and-final-block (the value-head-only and "
                "value-adapter surfaces do not define the A-vs-guardrail "
                "final-block conflict; adapter-surface projection is v14b)")
```

Replace with:

```python
    if post_opening_calibration_gradient_projection:
        if train_value_head_only:
            raise ValueError(
                "post_opening_calibration_gradient_projection requires a "
                "multi-component trainable surface (value-head-only offers only "
                "value_head, with no A-vs-guardrail conflict to project); use "
                "--train-value-head-and-final-block (v13) or "
                "--train-value-head-and-value-adapter (v14b)")
```

- [ ] **Step 4: Select the projected surface by mode (§C)**

Still in `train_step`, locate the `else:` block that assembles the surfaces (it currently starts with `_last = len(network.encoder.blocks) - 1` and ends by writing `grads["encoder"]["blocks"][_last] = _surf_final["block"]`):

```python
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
                _effective_projection_weight = (
                    post_opening_calibration_projection_strength * calibration_loss_weight)
                _surf_final, _proj_telem = project_conflicting_gradient(
                    _surf_total, _surf_a, _surf_g, weight=_effective_projection_weight)
                grads["value_head"] = _surf_final["value_head"]
                grads["encoder"]["blocks"][_last] = _surf_final["block"]
```

Replace with (surface chosen by mode; `project_conflicting_gradient` call is byte-identical — only the surface dict changes):

```python
            else:
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
                if train_value_head_and_value_adapter:
                    # v14b: project over the value-only adapter surface. The A/G
                    # passes include value_adapter grads (the adapter is in
                    # forward_padded).
                    _surf_total = {"value_head": grads["value_head"],
                                   "value_adapter": grads["value_adapter"]}
                    _surf_a = {"value_head": _g_a["value_head"],
                               "value_adapter": _g_a["value_adapter"]}
                    _surf_g = {"value_head": _g_g["value_head"],
                               "value_adapter": _g_g["value_adapter"]}
                else:
                    # v13/v9: project over value_head + the final residual block.
                    _last = len(network.encoder.blocks) - 1
                    _surf_total = {"value_head": grads["value_head"],
                                   "block": grads["encoder"]["blocks"][_last]}
                    _surf_a = {"value_head": _g_a["value_head"],
                               "block": _g_a["encoder"]["blocks"][_last]}
                    _surf_g = {"value_head": _g_g["value_head"],
                               "block": _g_g["encoder"]["blocks"][_last]}
                _effective_projection_weight = (
                    post_opening_calibration_projection_strength * calibration_loss_weight)
                _surf_final, _proj_telem = project_conflicting_gradient(
                    _surf_total, _surf_a, _surf_g, weight=_effective_projection_weight)
                grads["value_head"] = _surf_final["value_head"]
                if train_value_head_and_value_adapter:
                    grads["value_adapter"] = _surf_final["value_adapter"]
                else:
                    grads["encoder"]["blocks"][_last] = _surf_final["block"]
```

- [ ] **Step 5: Fold the post-projection adapter grad norm into the dict (§D)**

Still in `train_step`, locate the value-side grad block (the projected `grads["value_adapter"]` is read here, so `_adapter_grad_norm` is already post-projection):

```python
    if train_value_head_and_value_adapter:
        value_grads = {"value_head": grads["value_head"],
                       "value_adapter": grads["value_adapter"]}
        _, _agn = clip_grad_norm(grads["value_adapter"], max_norm=1e9)  # raw norm (no real clip)
        _adapter_grad_norm = float(_agn.item())
    else:
        value_grads = grads["value_head"]
        _adapter_grad_norm = 0.0
```

Add immediately below that block:

```python
    # v14b: fold the (post-projection) adapter grad norm into the projection dict
    # so slot [13] stays one self-describing object (dict = projection, float =
    # v14 grad-norm only). Covers the no-A / no-guardrail / no-conflict skip dicts
    # too, so train()'s accumulator always finds the key.
    if _proj_telem is not None and train_value_head_and_value_adapter:
        _proj_telem["value_adapter_grad_norm"] = _adapter_grad_norm
```

(No change to the return path: under v14b `_proj_telem` is a dict, so the existing `if _proj_telem is not None: return _guard_ret + (_proj_telem,)` already returns the dict at `[13]`, and the `train_value_head_and_value_adapter` float branch only fires when `_proj_telem is None` — i.e. v14 with projection off.)

- [ ] **Step 6: Branch the train() accumulator by type, not by the v14 flag (§E)**

In `train()`, inside the `if (_calib_guard_sign is not None and len(_ret) == 14):` block, locate:

```python
                                if train_value_head_and_value_adapter:
                                    # v14: the 14th slot is the adapter grad norm
                                    # (float), NOT a projection dict.
                                    sum_value_adapter_grad_norm += float(_ret[13])
                                    proj = None
                                else:
                                    proj = _ret[13]
```

Replace with:

```python
                                _extra = _ret[13]
                                if isinstance(_extra, dict):
                                    # v13 / v14b: slot [13] is the projection dict.
                                    # v14b folds the (post-projection) adapter grad
                                    # norm into it; v13 has no such key (-> 0.0).
                                    proj = _extra
                                    sum_value_adapter_grad_norm += float(
                                        _extra.get("value_adapter_grad_norm", 0.0))
                                else:
                                    # v14 (projection off): slot [13] is the adapter
                                    # grad norm float.
                                    proj = None
                                    sum_value_adapter_grad_norm += float(_extra)
```

(The following `if proj is not None and proj["skip_reason"] == "no_a":` chain is unchanged.)

- [ ] **Step 7: Run the new tests + regression**

Run: `.venv/bin/python -m pytest tests/test_v14b_adapter_projection.py tests/test_v14_value_adapter.py tests/test_gradient_projection.py tests/test_v13_projection_wiring.py tests/test_training.py tests/test_calibration_pool.py -v`
Expected: ALL PASS. v14b green; v14 (float slot, projection off) unchanged; v13 projection/wiring unchanged (the `isinstance(dict)` branch is byte-identical for v13 — the dict has no `value_adapter_grad_norm` key so the accumulator adds `0.0`); training + calibration unchanged.

- [ ] **Step 8: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py tests/test_v14b_adapter_projection.py
git commit -m "feat(training): v14b gradient projection over the value-adapter surface

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Full suite + merge (controller-run)

**Files:** none (verification + integration only).

- [ ] **Step 1: Worktree full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected in the worktree: baseline-passed + the new v14b tests, with EXACTLY the known 14 failed + 6 errors (missing gitignored data).

- [ ] **Step 2: FF-merge + authoritative suite + push**

FF-merge `feature/tvc-v14b-adapter-projection` to main (no `--no-ff`, never force-push), leaving the pre-existing main-checkout churn untouched. Run the authoritative suite on merged main (`.venv/bin/python -m pytest tests/ -q`) — expect **1394 + the new v14b tests, 0 failures**. Push (superpowers:finishing-a-development-branch). The FF-merge also pushes the previously-unpushed v14b spec (`e223fa5`) + this plan doc. STOP after push — the operator run is the USER's.

---

## Operator run (USER's, after merge) — from the spec §J

The v14 command PLUS `--post-opening-calibration-gradient-projection` (+ optional `--post-opening-calibration-projection-strength N`), new checkpoint dir `checkpoints/alphazero-v14b-adapter-projection-from-calib020-0001`. Confirm `train_value_head_and_value_adapter=true`, projection active (`calib_projection_conflict_steps > 0`), `value_adapter_gate` off 0, `value_adapter_grad_norm` present (post-projection), `verify_value_head_and_adapter_checkpoint` exit 0; then gates A/B/C/D vs `calib020_0001` (no promotion unless all four pass). Interpretation: A holds + B/C/D recover → promotion match; A regresses → lower `--…-projection-strength`; B/C/D still drift → not gradient-conflict on this surface → per-channel gate / richer adapter (later design).
