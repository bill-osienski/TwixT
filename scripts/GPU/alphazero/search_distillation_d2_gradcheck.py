"""SEARCH DISTILLATION D2 -- GRADIENT CORRECTNESS CHECK: CPU/GPU comparison, an independent NumPy reference for the
policy head, and stage-by-stage localization, at two fixed states. BUILT; run only through its wrapper.

Card: `docs/superpowers/2026-10-22-search-distillation-d2-gradient-check-card.md`, pinned below by sha256. Nothing
runs on import. `run` refuses unless
`search_distillation_d2_gradcheck_authorization.SEARCH_DISTILLATION_D2_GRADCHECK_AUTHORIZED` is open, read at call
time.

A DIAGNOSTIC. The reproducibility diagnostic (ledger entry 25) measured gradients that differ from identical state and
inputs and an update whose computed gradient fails the linearity of the stated loss; it did not establish WHICH
computation is wrong. This run, with no optimizer update outside one prefix:
  1. STATES. S0 = the unchanged incumbent. S21 = a NEWLY RECONSTRUCTED pre-update-21 state, from ONE GPU prefix of
     Stage 3's updates 0-20 (its digest is compared with the digests entry 25 recorded; it is not those states).
     S0 is evaluated on update 0's batch, S21 on update 21's.
  2. CPU/GPU. At each state, identical state and batch arrays are copied to each device (digests checked) and the
     training loss's forward outputs and every gradient tensor are evaluated three times per device; plus a weighted
     -record additivity check. CPU MLX is a COMPARISON backend, not an oracle: it shares the autodiff implementation.
  3. AN INDEPENDENT REFERENCE: NumPy float64 forward and hand-derived analytic derivatives of the policy head (1x1
     conv, BatchNorm in evaluation mode, ReLU, the gather at legal moves, the two linear layers, the masks and the
     masked cross-entropy), checked by fixed directional finite differences where the perturbation crosses no ReLU
     boundary -- on synthetic cases (repeated gather indices, different batch members, padding, weighted-record
     additivity) and on the real batch at each state.
  4. LOCALIZATION: the forward is split into stages; each stage's output and its vector-Jacobian product are
     computed on each device, three times, from IDENTICAL stage inputs and cotangents (the CPU chain's), so the
     first disagreeing stage is located, not just a difference between two complete traversals.
  5. Every result is written as soon as it exists; the minimal fixture reproducing any component that disagrees is
     saved (outside the repository; hashes and the declared tolerances in the evidence). No full model state and no
     playable checkpoint is saved; encoder weights are never saved.

A DIFFERENCE is an observation. Invalid inputs, a non-finite result, an exception or SIGTERM (the wrapper's 10-minute
ceiling) are a VOID. No retry. The record holds measurements only: it names no cause.

    python -m scripts.GPU.alphazero.search_distillation_d2_gradcheck run --checkpoint <absolute path to calib020_0001>
"""
from __future__ import annotations

import contextlib
import dataclasses
import datetime
import hashlib
import json
import math
import os
import pathlib
import platform
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from . import search_distillation_d2_gradcheck_authorization as GATE
from . import search_distillation_d2_gradrepeat as G
from . import search_distillation_d2_gradtrace as R
from . import search_distillation_d2_measure as M
from . import search_distillation_d2_train as T

ROOT = M.ROOT
RECORD_KIND = "search-distillation-d2-gradcheck-record/1"
CARD = "docs/superpowers/2026-10-22-search-distillation-d2-gradient-check-card.md"
CARD_SHA256 = "0aba00a5a6ce57e238888ac552ea7bac080afefdd5a5e3f0ff56db4212916958"
EVIDENCE_DIR = "docs/superpowers/evidence/search-distillation-d2-gradcheck"
FIXTURE_PARENT = pathlib.Path.home() / "twixt-cache" / "search-distillation-d2"      # outside the repository
REFERENCE = R.REFERENCE
REPEAT_RECORD = "docs/superpowers/evidence/search-distillation-d2-gradrepeat/record.json"
REPEAT_DIGESTS = "docs/superpowers/evidence/search-distillation-d2-gradrepeat/digests.jsonl"
CORPUS_JSON = R.CORPUS_JSON
ADAM_DEFAULTS = T.ADAM_DEFAULTS
FROZEN = {**{k: R.FROZEN[k] for k in ("corpus_sha256", "teacher_digest", "reference_sha256",
                                      "reference_student_digest", "training_ids_sha256", "training_games",
                                      "held_out_games", "training_records", "passes", "steps", "batch_size",
                                      "learning_rate", "anchor_weight", "l2_weight", "clip_norm", "seed")},
          "repeat_record_sha256": "62953e5b201c94b2664236051959f979cb5a49981922c695820876b8ea1ea842",
          "repeat_digests_sha256": "ea4fe6316bf1dd29d9c4ecfe5467f4149745648b1e65cc8c6e5ea54c03498b7b",
          "states": [{"name": "S0", "prefix_updates": 0, "batch_update": 0},
                     {"name": "S21", "prefix_updates": 21, "batch_update": 21}],
          "devices": ["cpu", "gpu"], "repeats": 3,
          "cross_rel_tol": 1e-3, "cross_abs_tol": 1e-6, "fd_eps": 1e-5, "fd_rel_tol": 1e-6, "fd_mlx_rel_tol": 1e-3,
          "fd_seed": 0, "synthetic_seed": 0}
EXIT_COMPLETED, EXIT_VOID, EXIT_REFUSED, EXIT_USAGE = 0, 3, 4, 2
Void, Terminated = M.Void, M.Terminated


def _mx_devices():
    import mlx.core as mx
    return {"cpu": mx.cpu, "gpu": mx.gpu}


DEVICES: Dict[str, Any] = {}                      # name -> MLX device; filled at first use (tests may remap)
CURRENT = {"device": None}                        # the device label a computation is running under


class Refused(Exception):
    """The run never started: gate closed, card not the pinned bytes, an output exists, or a bad argument."""


def _require_gate() -> None:
    if GATE.SEARCH_DISTILLATION_D2_GRADCHECK_AUTHORIZED is not True:
        raise Refused("SEARCH_DISTILLATION_D2_GRADCHECK_AUTHORIZED is closed: the check is not authorized")


@dataclasses.dataclass
class Seams:
    """What the run reads from outside this module. TEST seams; `run` passes `_real_seams` only."""
    read_corpus_json: Callable[[], bytes]
    read_reference: Callable[[], bytes]
    read_repeat_record: Callable[[], bytes]
    read_repeat_digests: Callable[[], bytes]
    checkpoint_sha1: Callable[[], str]
    load_network: Callable[[], Any]                    # a FRESH load of the incumbent on every call
    loaded_modules: Callable[[], Dict[str, str]]


@contextlib.contextmanager
def on(device: str):
    """Run MLX operations on `device` (the default device for the duration)."""
    import mlx.core as mx
    if not DEVICES:
        DEVICES.update(_mx_devices())
    previous = mx.default_device()
    mx.set_default_device(DEVICES[device])
    CURRENT["device"] = device
    try:
        yield
    finally:
        mx.set_default_device(previous)
        CURRENT["device"] = None


def _digest(arrays: Mapping[str, np.ndarray]) -> str:
    from . import training_replay_capture as C
    return C.model_state_digest(dict(arrays))


def _finite(name: str, arrays) -> None:
    for a in arrays:
        if not np.all(np.isfinite(a)):
            raise Void("non_finite", f"{name}: a non-finite result")


def agree(a: np.ndarray, b: np.ndarray, spec: Mapping[str, Any]) -> Dict[str, Any]:
    """`a` against the comparison `b`: bitwise equality, and the declared tolerance
    max|a - b| <= cross_rel_tol * max|b| + cross_abs_tol."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return {"bitwise": False, "max_abs": math.inf, "scale": None, "within_tol": False, "shape_mismatch": True}
    d = float(np.abs(a - b).max()) if a.size else 0.0
    scale = float(np.abs(b).max()) if b.size else 0.0
    return {"bitwise": bool(np.array_equal(a, b)), "max_abs": d, "scale": scale,
            "within_tol": d <= spec["cross_rel_tol"] * scale + spec["cross_abs_tol"]}


# ─────────────────────────────── the NumPy reference (float64, analytic) ───────────────────────────────

PH_KEYS = ("conv.weight", "conv.bias", "bn.weight", "bn.bias", "fc.weight", "fc.bias", "out.weight", "out.bias")


def policy_head_params(ph) -> Dict[str, np.ndarray]:
    """The policy head's trainable tensors and its BatchNorm statistics and eps, as float64 NumPy."""
    p = {k: np.array(v).astype(np.float64) for k, v in
         (("conv.weight", ph.conv.weight), ("conv.bias", ph.conv.bias), ("bn.weight", ph.bn.weight),
          ("bn.bias", ph.bn.bias), ("fc.weight", ph.fc.weight), ("fc.bias", ph.fc.bias),
          ("out.weight", ph.out.weight), ("out.bias", ph.out.bias))}
    p["bn.running_mean"] = np.array(ph.bn.running_mean).astype(np.float64)
    p["bn.running_var"] = np.array(ph.bn.running_var).astype(np.float64)
    p["bn.eps"] = np.array(ph.bn.eps, dtype=np.float64)
    return p


def np_policy_head(p: Mapping[str, np.ndarray], F, rows, cols, mask, target, w, grads: bool = True):
    """Loss L = sum_i w_i * CE_i / B through the policy head, and (if `grads`) its analytic gradient with respect to
    every trainable tensor and to the input features F. Every step as PolicyHead.forward_padded and
    trainer.compute_masked_log_probs compute it, in float64."""
    F = np.asarray(F, dtype=np.float64)
    B, H, W, Cin = F.shape
    Wc = p["conv.weight"].reshape(2, Cin)
    pre = F @ Wc.T + p["conv.bias"]                                         # (B, H, W, 2)
    inv = 1.0 / np.sqrt(p["bn.running_var"] + p["bn.eps"])
    xhat = (pre - p["bn.running_mean"]) * inv
    y = xhat * p["bn.weight"] + p["bn.bias"]
    x = np.maximum(y, 0.0)
    r, c = np.asarray(rows), np.asarray(cols)
    bidx = np.arange(B)[:, None]
    g = x[bidx, r, c, :]                                                    # (B, M, 2)
    a1 = g @ p["fc.weight"].T + p["fc.bias"]                                # (B, M, 64)
    h = np.maximum(a1, 0.0)
    z = (h @ p["out.weight"].T + p["out.bias"])[..., 0]                     # (B, M)
    m = np.asarray(mask) > 0.5
    zm = np.where(m, z, -1e9)
    zmax = zm.max(axis=1, keepdims=True)
    lse = zmax + np.log(np.exp(zm - zmax).sum(axis=1, keepdims=True))
    lp = zm - lse
    t, wv = np.asarray(target, dtype=np.float64), np.asarray(w, dtype=np.float64)
    ce = -(t * lp).sum(axis=1)
    L = float((wv * ce).sum() / B)
    inter = {"pre": pre, "y": y, "x": x, "gathered": g, "a1": a1, "h": h, "z": z, "lp": lp, "ce": ce}
    if not grads:
        return L, None, inter
    dlp = -(wv[:, None] * t) / B
    soft = np.exp(lp)
    dzm = dlp - soft * dlp.sum(axis=1, keepdims=True)
    dz = np.where(m, dzm, 0.0)
    d = {"out.weight": (dz[..., None] * h).sum(axis=(0, 1))[None, :], "out.bias": np.array([dz.sum()])}
    dh = dz[..., None] * p["out.weight"][0]
    da1 = dh * (a1 > 0)
    d["fc.weight"] = np.einsum("bmk,bmj->kj", da1, g)
    d["fc.bias"] = da1.sum(axis=(0, 1))
    dg = da1 @ p["fc.weight"]                                               # (B, M, 2)
    dx = np.zeros_like(x)
    np.add.at(dx, (np.broadcast_to(bidx, r.shape), r, c), dg)               # repeated indices accumulate
    dy = dx * (y > 0)
    d["bn.weight"] = (dy * xhat).sum(axis=(0, 1, 2))
    d["bn.bias"] = dy.sum(axis=(0, 1, 2))
    dpre = dy * p["bn.weight"] * inv
    d["conv.weight"] = np.einsum("bhwo,bhwi->oi", dpre, F).reshape(p["conv.weight"].shape)
    d["conv.bias"] = dpre.sum(axis=(0, 1, 2))
    d["F"] = dpre @ Wc
    return L, d, inter


def mlx_policy_head(ph, F, rows, cols, mask, target, w):
    """The same loss through the MLX policy head on the current device: (L, gradients, logits) as NumPy."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    from .trainer import compute_masked_log_probs

    params = ph.trainable_parameters()
    keys = [k for k, _ in tree_flatten(params)]
    Fm, rm, cm, mm, tm, wm = (mx.array(a) for a in (F, rows, cols, mask, target, w))
    B = Fm.shape[0]

    def loss(flat, f):
        ph.update(tree_unflatten(list(zip(keys, flat))))
        logits = ph.forward_padded(f, rm, cm, mm)
        return mx.sum(wm * -mx.sum(tm * compute_masked_log_probs(logits, mm), axis=1)) / B, logits

    flat = [mx.array(np.array(v)) for _, v in tree_flatten(params)]       # fresh copies on this device
    try:
        (L, logits), (gp, gF) = mx.value_and_grad(lambda fl, f: loss(fl, f), argnums=(0, 1))(flat, Fm)
        mx.eval(L, logits, gp, gF)
    finally:
        ph.update(params)
    out = {k: np.array(v) for k, v in zip(keys, gp)}
    out["F"] = np.array(gF)
    return float(L), out, np.array(logits)


def directional_check(p, F, rows, cols, mask, target, w, analytic, mlx_grads, spec, seed: int) -> Dict[str, Any]:
    """d . grad against (L(theta + eps d) - L(theta - eps d)) / (2 eps), for one fixed random direction d over every
    trainable tensor and F -- only if neither ReLU's pattern changes between the two points; eps is halved up to
    three times to find such a step, else the check is SKIPPED (recorded)."""
    rng = np.random.default_rng(seed)
    keys = list(PH_KEYS) + ["F"]
    base = {**{k: p[k] for k in PH_KEYS}, "F": np.asarray(F, dtype=np.float64)}
    dirs = {k: rng.standard_normal(base[k].shape) for k in keys}
    eps = spec["fd_eps"]
    for _ in range(4):
        pts = []
        for sgn in (1.0, -1.0):
            q = {**p, **{k: base[k] + sgn * eps * dirs[k] for k in PH_KEYS}}
            pts.append(np_policy_head(q, base["F"] + sgn * eps * dirs["F"], rows, cols, mask, target, w,
                                      grads=False))
        m = np.asarray(mask) > 0.5
        same = all(np.array_equal(pts[0][2][k] > 0, pts[1][2][k] > 0) for k in ("y",)) and \
            np.array_equal((pts[0][2]["a1"] > 0)[m], (pts[1][2]["a1"] > 0)[m])
        if same:
            fd = (pts[0][0] - pts[1][0]) / (2 * eps)
            an = float(sum((analytic[k] * dirs[k]).sum() for k in keys))
            out = {"eps": eps, "skipped": False, "finite_difference": fd, "analytic": an,
                   "analytic_ok": abs(fd - an) <= spec["fd_rel_tol"] * (abs(an) + 1.0)}
            for dev, g in mlx_grads.items():
                v = float(sum((np.asarray(g[k], dtype=np.float64) * dirs[k]).sum() for k in keys))
                out[f"mlx_{dev}"] = v
                out[f"mlx_{dev}_ok"] = abs(fd - v) <= spec["fd_mlx_rel_tol"] * (abs(fd) + 1e-6)
            return out
        eps /= 2
    return {"eps": eps, "skipped": True, "reason": "every step tried crossed a ReLU boundary"}


def reference_case(name: str, ph, F, rows, cols, mask, target, w, spec, sink) -> Dict[str, Any]:
    """The policy head on one case: NumPy analytic against MLX on each device, three times, and the directional
    check. Weighted-record additivity: MLX's gradient against the sum of its single-record gradients."""
    p = policy_head_params(ph)
    L_np, g_np, _ = np_policy_head(p, F, rows, cols, mask, target, w)
    out: Dict[str, Any] = {"case": name, "numpy_loss": L_np, "devices": {}}
    first: Dict[str, Dict[str, np.ndarray]] = {}
    failed = False
    for dev in spec["devices"]:
        evals = []
        with on(dev):
            for _ in range(spec["repeats"]):
                L, g, _logits = mlx_policy_head(ph, F, rows, cols, mask, target, w)
                _finite(f"{name} on {dev}", [np.array(L), *g.values()])
                evals.append((L, g))
            singles = None
            if np.count_nonzero(w) > 1:
                singles = {}
                for j in range(len(w)):
                    if w[j] == 0:
                        continue
                    e = np.zeros_like(w)
                    e[j] = w[j]
                    gj = mlx_policy_head(ph, F, rows, cols, mask, target, e)[1]
                    singles = {k: singles.get(k, 0) + v.astype(np.float64) for k, v in gj.items()}
        L0, g0 = evals[0]
        first[dev] = g0
        d = {"loss": [L for L, _ in evals],
             "vs_numpy": {k: agree(g0[k], g_np[k], spec) for k in g_np},
             "repeats": {k: [agree(g[k], g0[k], spec)["bitwise"] for _, g in evals[1:]] for k in g0},
             "loss_vs_numpy": agree(np.array(L0), np.array(L_np), spec)}
        if singles is not None:
            d["additivity_vs_full"] = {k: agree(singles[k], g0[k].astype(np.float64), spec) for k in g0}
        bad = (not all(v["within_tol"] for v in d["vs_numpy"].values()) or not d["loss_vs_numpy"]["within_tol"]
               or not all(all(v) for v in d["repeats"].values())
               or ("additivity_vs_full" in d and not all(v["within_tol"] for v in d["additivity_vs_full"].values())))
        d["disagrees"] = bad
        failed = failed or bad
        out["devices"][dev] = d
    out["directional"] = directional_check(p, F, rows, cols, mask, target, w, g_np, first, spec, spec["fd_seed"])
    if failed or (not out["directional"].get("skipped") and not all(
            v for k, v in out["directional"].items() if k.endswith("_ok"))):
        out["fixture"] = sink.fixture(f"reference_{name}", {"F": np.asarray(F), "rows": np.asarray(rows),
                                                            "cols": np.asarray(cols), "mask": np.asarray(mask),
                                                            "target": np.asarray(target), "w": np.asarray(w),
                                                            **{f"param.{k}": v for k, v in p.items()}},
                                      {"component": "policy head", "case": name})
    sink.write(f"reference/{name}.json", out)
    return out


def synthetic_cases(seed: int):
    """Small policy heads (8 input channels) on hand-built cases: (name, head, F, rows, cols, mask, target, w)."""
    import mlx.core as mx

    from .network import PolicyHead

    rng = np.random.default_rng(seed)

    def head():
        mx.random.seed(int(rng.integers(1 << 30)))
        ph = PolicyHead(in_channels=8, hidden=64)
        ph.bn.running_mean = mx.array(rng.normal(0, 0.3, 2).astype(np.float32))
        ph.bn.running_var = mx.array(rng.uniform(0.5, 2.0, 2).astype(np.float32))
        ph.bn.weight = mx.array(rng.uniform(0.5, 1.5, 2).astype(np.float32))
        ph.bn.bias = mx.array(rng.uniform(0.2, 1.0, 2).astype(np.float32))
        ph.eval()
        return ph

    def feats(B):
        return rng.normal(0, 1, (B, 24, 24, 8)).astype(np.float32)

    def targets(mask):
        t = rng.uniform(0.1, 1.0, mask.shape).astype(np.float32) * mask
        return (t / t.sum(axis=1, keepdims=True)).astype(np.float32)

    cases = []
    rows = np.array([[3, 3, 3, 0, 7], [3, 5, 9, 11, 3]], dtype=np.int32)          # (3, 4) three times in row 0
    cols = np.array([[4, 4, 4, 0, 2], [4, 6, 1, 8, 4]], dtype=np.int32)          # and twice in row 1
    mask = np.ones((2, 5), dtype=np.float32)
    cases.append(("repeated_indices", head(), feats(2), rows, cols, mask, targets(mask), np.ones(2, np.float32)))
    rows = rng.integers(0, 24, (4, 6)).astype(np.int32)
    cols = rng.integers(0, 24, (4, 6)).astype(np.int32)
    mask = np.ones((4, 6), dtype=np.float32)
    cases.append(("batch_members", head(), feats(4), rows, cols, mask, targets(mask), np.ones(4, np.float32)))
    rows = np.array([[1, 2, 3, 4, 5], [0, 6, 0, 0, 0], [8, 0, 0, 0, 0]], dtype=np.int32)    # padding at (0, 0),
    cols = np.array([[1, 2, 3, 4, 5], [0, 6, 0, 0, 0], [9, 0, 0, 0, 0]], dtype=np.int32)    # which row 1 also plays
    mask = np.array([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=np.float32)
    cases.append(("padding", head(), feats(3), rows, cols, mask, targets(mask), np.ones(3, np.float32)))
    rows = rng.integers(0, 24, (4, 5)).astype(np.int32)
    cols = rng.integers(0, 24, (4, 5)).astype(np.int32)
    mask = np.ones((4, 5), dtype=np.float32)
    cases.append(("weighted_additivity", head(), feats(4), rows, cols, mask, targets(mask),
                  np.array([0.3, 1.7, 0.0, 1.0], dtype=np.float32)))
    return cases


# ─────────────────────────────── stages: identical inputs, each device ───────────────────────────────

def _gather(x, rows, cols):
    """The policy head's gather at the legal moves, as PolicyHead.forward_padded performs it (B > 1)."""
    import mlx.core as mx
    return x[mx.arange(rows.shape[0], dtype=rows.dtype)[:, None], rows, cols, :]


def stage_list(net, ex) -> List[Tuple[str, Any, Callable]]:
    """The training loss's forward, split into stages: (name, module whose trainable tensors the stage reads, fn).
    The policy branch runs in order; the value branch reads the encoder's output."""
    import mlx.core as mx
    import mlx.nn as nn

    from .search_distillation_d2 import BOARD
    from .trainer import compute_masked_log_probs

    ph, enc = net.policy_head, net.encoder
    B, Mm = ex["rows_c"].shape
    return [("encoder", enc, lambda a: enc(a)),
            ("policy_conv", ph.conv, lambda a: ph.conv(a)),
            ("policy_bn", ph.bn, lambda a: ph.bn(a)),
            ("policy_relu", None, nn.relu),
            ("gather", None, lambda a: _gather(a, ex["rows_c"], ex["cols_c"])),
            ("fc", ph.fc, lambda a: ph.fc(a.reshape(B * Mm, 2))),
            ("fc_relu", None, nn.relu),
            ("out", ph.out, lambda a: ph.out(a).reshape(B, Mm)),
            ("head_mask", None, lambda a: mx.where(ex["mask"] > 0.5, a, ph._neg_inf_f32)),
            ("log_softmax", None, lambda a: compute_masked_log_probs(a, ex["mask"])),
            ("policy_ce", None, lambda a: mx.mean(-mx.sum(ex["target"] * a, axis=1))),
            ("value_head", None, lambda a: net.value_head(net._value_features(a), BOARD)),
            ("anchor", None, lambda a: mx.mean((a - ex["tv"]) ** 2))]


def _vjp(module, fn, x, ct):
    """fn's output at x and its vector-Jacobian product with cotangent ct: (out, d/dx, {tensor: d/dtensor})."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    if module is None:
        outs, (gx,) = mx.vjp(fn, [x], [ct])
        mx.eval(outs, gx)
        return outs[0], gx, {}
    params = module.trainable_parameters()
    flat = tree_flatten(params)
    keys = [k for k, _ in flat]

    def f(*arrs):
        module.update(tree_unflatten(list(zip(keys, arrs[:-1]))))
        return fn(arrs[-1])

    try:
        outs, grads = mx.vjp(f, [*[v for _, v in flat], x], [ct])
        mx.eval(outs, grads)
    finally:
        module.update(params)
    return outs[0] if isinstance(outs, (list, tuple)) else outs, grads[-1], dict(zip(keys, grads[:-1]))


def _ex(batch_np, tv_np, canon=None):
    import mlx.core as mx
    ex = {k: mx.array(v) for k, v in zip(("boards", "rows", "cols", "mask", "target"), batch_np)}
    ex["tv"] = mx.array(tv_np)
    if canon is not None:
        ex["rows_c"], ex["cols_c"] = mx.array(canon["rows_c"]), mx.array(canon["cols_c"])
    return ex


def localize(state_name, nets: Mapping[str, Any], batch_np, tv_np, spec, sink) -> Dict[str, Any]:
    """Each stage's output and VJP on each device, three times, from the CPU chain's stage inputs and cotangents."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from .network import canonicalize_batch
    from .search_distillation_d2 import BOARD

    ref_dev = spec["devices"][0]
    with on(ref_dev):
        ex = _ex(batch_np, tv_np)
        bc, rc, cc, _ = canonicalize_batch(ex["boards"], ex["rows"], ex["cols"], ex["mask"], BOARD)
        mx.eval(bc, rc, cc)
    canon = {"board_c": np.array(bc), "rows_c": np.array(rc), "cols_c": np.array(cc)}
    out: Dict[str, Any] = {"state": state_name, "canonicalize": {}, "forward": {}, "backward": {}}
    for dev in spec["devices"]:                             # the canonicalization itself, on each device
        with on(dev):
            res = []
            for _ in range(spec["repeats"]):
                e = _ex(batch_np, tv_np)
                b2, r2, c2, _ = canonicalize_batch(e["boards"], e["rows"], e["cols"], e["mask"], BOARD)
                mx.eval(b2, r2, c2)
                res.append({"board_c": np.array(b2), "rows_c": np.array(r2), "cols_c": np.array(c2)})
        out["canonicalize"][dev] = {k: [agree(r[k], canon[k], spec)["bitwise"] for r in res] for k in canon}

    # the reference chain: forward stage by stage, then cotangents backward
    with on(ref_dev):
        ex = _ex(batch_np, tv_np, canon)
        st = stage_list(nets[ref_dev], ex)
        names = [s[0] for s in st]
        ins, outs = {}, {}
        a = mx.array(canon["board_c"])
        for name, _m, fn in st[:11]:
            ins[name] = np.array(a)
            a = fn(a)
            mx.eval(a)
            outs[name] = np.array(a)
        ins["value_head"] = outs["encoder"]
        v = st[11][2](mx.array(outs["encoder"]))
        mx.eval(v)
        outs["value_head"], ins["anchor"] = np.array(v), np.array(v)
        an = st[12][2](v)
        mx.eval(an)
        outs["anchor"] = np.array(an)
        cts = {"policy_ce": np.array(1.0, dtype=np.float32),
               "anchor": np.array(spec["anchor_weight"], dtype=np.float32)}
        grads_ref = {}
        for name in ["anchor", "value_head"] + names[10::-1]:
            if name == "encoder":
                cts["encoder"] = (grads_ref["policy_conv"][0] + grads_ref["value_head"][0]).astype(np.float32)
            module, fn = next((m, f) for n, m, f in st if n == name)
            _, gx, gp = _vjp(module, fn, mx.array(ins[name]), mx.array(cts[name]))
            grads_ref[name] = (np.array(gx), {k: np.array(v) for k, v in gp.items()})
            follow = {"anchor": "value_head", "policy_ce": "log_softmax", "log_softmax": "head_mask",
                      "head_mask": "out", "out": "fc_relu", "fc_relu": "fc", "fc": "gather", "gather": "policy_relu",
                      "policy_relu": "policy_bn", "policy_bn": "policy_conv"}.get(name)
            if follow:
                cts[follow] = grads_ref[name][0].reshape(outs[follow].shape).astype(np.float32)
        # the decomposition is faithful: the chain's tensor gradients, composed, against autodiff of the same loss
        # (policy CE + anchor_weight x anchor, no L2) through the module's own forward on the same device
        import mlx.nn as nn

        from .trainer import MainModule, compute_masked_log_probs
        net = nets[ref_dev]
        surface = MainModule(net.encoder, net.policy_head)

        def whole(_s):
            logits, values, _ = net.forward_padded(ex["boards"], ex["rows"], ex["cols"], ex["mask"],
                                                   active_size=BOARD)
            return (mx.mean(-mx.sum(ex["target"] * compute_masked_log_probs(logits, ex["mask"]), axis=1)) +
                    spec["anchor_weight"] * mx.mean((values - ex["tv"]) ** 2))

        _, g = nn.value_and_grad(surface, whole)(surface)
        mx.eval(g)
        auto = {k: np.array(v) for k, v in tree_flatten(g)}
    composed = {f"encoder.{k}": v for k, v in grads_ref["encoder"][1].items()}
    for st_name, prefix in (("policy_conv", "policy_head.conv."), ("policy_bn", "policy_head.bn."),
                            ("fc", "policy_head.fc."), ("out", "policy_head.out.")):
        composed.update({prefix + k: v for k, v in grads_ref[st_name][1].items()})
    out["chain_vs_autodiff"] = {k: agree(composed[k], auto[k], spec) if k in composed else {"missing": True}
                                for k in auto}
    _finite(f"{state_name} reference chain", [*outs.values(), *[g for g, _ in grads_ref.values()]])

    for dev in spec["devices"]:
        out["forward"][dev], out["backward"][dev] = {}, {}
        with on(dev):
            ex = _ex(batch_np, tv_np, canon)
            st = stage_list(nets[dev], ex)
            for name, module, fn in st:
                fw, bw = [], []
                for _ in range(spec["repeats"]):
                    o, gx, gp = _vjp(module, fn, mx.array(ins[name]), mx.array(cts[name]))
                    fw.append(np.array(o))
                    bw.append((np.array(gx), {k: np.array(v) for k, v in gp.items()}))
                _finite(f"{state_name} {name} on {dev}", [*fw, *[g for g, _ in bw]])
                out["forward"][dev][name] = {
                    "vs_reference": [agree(f, outs[name], spec) for f in fw],
                    "repeats_bitwise": [bool(np.array_equal(f, fw[0])) for f in fw[1:]]}
                ref_gx, ref_gp = grads_ref[name]
                out["backward"][dev][name] = {
                    "input": {"vs_reference": [agree(g, ref_gx, spec) for g, _ in bw],
                              "repeats_bitwise": [bool(np.array_equal(g, bw[0][0])) for g, _ in bw[1:]]},
                    "tensors": {k: {"vs_reference": [agree(gp[k], ref_gp[k], spec) for _, gp in bw],
                                    "repeats_bitwise": [bool(np.array_equal(gp[k], bw[0][1][k])) for _, gp in bw[1:]]}
                                for k in ref_gp}}
    order_fw = names
    order_bw = ["anchor", "value_head"] + names[10::-1]

    def strict(entry):
        items = [entry] if "vs_reference" in entry else [entry["input"], *entry["tensors"].values()]
        return any(not all(v["within_tol"] for v in it["vs_reference"]) or not all(it["repeats_bitwise"])
                   for it in items)

    out["first_disagreement"] = {
        dev: {"forward": next((n for n in order_fw if strict(out["forward"][dev][n])), None),
              "backward": next((n for n in order_bw if strict(out["backward"][dev][n])), None)}
        for dev in spec["devices"]}
    for dev in spec["devices"]:
        for kind in ("forward", "backward"):
            name = out["first_disagreement"][dev][kind]
            if name is None or (state_name, name) in sink.saved_stages:
                continue
            sink.saved_stages.add((state_name, name))
            module = next(m for n, m, _f in stage_list(nets[dev], _ex(batch_np, tv_np, canon)) if n == name)
            arrays = {"input": ins[name], "cotangent": cts[name], "reference_output": outs[name],
                      "rows_c": canon["rows_c"], "cols_c": canon["cols_c"], "mask": batch_np[3],
                      "target": batch_np[4], "teacher_values": tv_np}
            meta = {"component": f"stage {name}", "state": state_name, "found_on": dev, "kind": kind}
            if module is not None and name != "encoder":
                arrays.update({f"param.{k}": np.array(v) for k, v in tree_flatten(module.trainable_parameters())})
            elif name == "encoder":
                meta["weights"] = "not saved (encoder weights are never saved); the state digest identifies them"
            out.setdefault("fixtures", []).append(sink.fixture(f"stage_{state_name}_{name}", arrays, meta))
    sink.write(f"stages/{state_name}.json", out)
    return out


# ─────────────────────────────── the full training loss on each device ───────────────────────────────

def full_model(state_name, state_np, state_digest, make_net, batch_np, tv_np, recipe, spec, sink) -> Dict[str, Any]:
    """The training loss (search_distillation_d2.train_student's) on each device, three times, each from a FRESH
    network set to the state and fresh batch arrays (both digests checked): forward outputs and every gradient
    tensor. No optimizer update. Plus weighted-record additivity: g(first half) + g(second half) = g(all)."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten

    from . import search_distillation_d2 as D
    from .trainer import MainModule, compute_masked_log_probs

    names_in = ("boards", "rows", "cols", "mask", "target")
    in_digest = _digest({**dict(zip(names_in, batch_np)), "teacher_values": tv_np})
    evals: Dict[str, list] = {}
    additivity = {}
    for dev in spec["devices"]:
        evals[dev] = []
        with on(dev):
            for r in range(spec["repeats"]):
                net = make_net()
                net.eval()
                net.update(tree_unflatten([(k, mx.array(v)) for k, v in state_np.items()]))
                if D.state_digest(net) != state_digest:
                    raise Void("state", f"{state_name} on {dev}: the network is not the state")
                ex = _ex(batch_np, tv_np)
                arrays = {n: np.array(ex[n]) for n in names_in}
                if _digest({**arrays, "teacher_values": np.array(ex["tv"])}) != in_digest:
                    raise Void("inputs", f"{state_name} on {dev}: the batch arrays are not the batch")
                surface = MainModule(net.encoder, net.policy_head)
                B = batch_np[0].shape[0]

                def loss_fn(_s, w=None, l2=True):
                    logits, values, _ = net.forward_padded(ex["boards"], ex["rows"], ex["cols"], ex["mask"],
                                                           active_size=D.BOARD)
                    ce = -mx.sum(ex["target"] * compute_masked_log_probs(logits, ex["mask"]), axis=1)
                    sq = (values - ex["tv"]) ** 2
                    if w is None:
                        policy, anchor = mx.mean(ce), mx.mean(sq)
                        total = policy + recipe.anchor_weight * anchor + recipe.l2_weight * sum(
                            mx.sum(p ** 2) for _, p in tree_flatten(_s.trainable_parameters()))
                    else:
                        policy, anchor = mx.sum(w * ce) / B, mx.sum(w * sq) / B
                        total = policy + recipe.anchor_weight * anchor
                    return total, policy, anchor, logits, values

                (loss, policy, anchor, logits, values), g = nn.value_and_grad(surface, loss_fn)(surface)
                mx.eval(loss, policy, anchor, logits, values, g)
                gnp = {k: np.array(v) for k, v in tree_flatten(g)}
                _finite(f"{state_name} full model on {dev}", [np.array(loss), np.array(logits), *gnp.values()])
                evals[dev].append({"loss": float(loss), "policy": float(policy), "anchor": float(anchor),
                                   "logits": np.array(logits), "values": np.array(values), "grads": gnp})
                sink.line("full.jsonl", {"state": state_name, "device": dev, "repeat": r, "loss": float(loss),
                                         "policy": float(policy), "anchor": float(anchor),
                                         "grad_norm": math.sqrt(sum(float((v.astype(np.float64) ** 2).sum())
                                                                    for v in gnp.values())),
                                         "grad_digest": _digest(gnp),
                                         "logits_digest": _digest({"logits": np.array(logits)})})
                if r == 0:
                    halves = []
                    for part in (np.arange(B) < B // 2, np.arange(B) >= B // 2, np.ones(B, bool)):
                        w = mx.array(part.astype(np.float32))
                        _, gw = nn.value_and_grad(surface, lambda s, w=w: loss_fn(s, w=w)[0])(surface)
                        mx.eval(gw)
                        halves.append({k: np.array(v).astype(np.float64) for k, v in tree_flatten(gw)})
                    additivity[dev] = {k: agree(halves[0][k] + halves[1][k], halves[2][k], spec) for k in halves[2]}
                del net, surface
    ref = evals[spec["devices"][0]][0]
    out = {"state": state_name, "state_digest": state_digest, "inputs_digest": in_digest, "devices": {}}
    for dev in spec["devices"]:
        e = evals[dev]
        out["devices"][dev] = {
            "loss_terms": [{k: x[k] for k in ("loss", "policy", "anchor")} for x in e],
            "grad_norms": [math.sqrt(sum(float((v.astype(np.float64) ** 2).sum()) for v in x["grads"].values()))
                           for x in e],
            "logits_vs_reference": [agree(x["logits"], ref["logits"], spec) for x in e],
            "values_vs_reference": [agree(x["values"], ref["values"], spec) for x in e],
            "grads_vs_reference": [{k: agree(x["grads"][k], ref["grads"][k], spec) for k in ref["grads"]} for x in e],
            "repeats_bitwise": [{k: bool(np.array_equal(x["grads"][k], e[0]["grads"][k])) for k in ref["grads"]}
                                for x in e[1:]],
            "additivity": additivity[dev]}
    sink.write(f"full/{state_name}.json", out)
    return out


# ─────────────────────────────── the reconstructed state ───────────────────────────────

def draw_batches(n_records: int, recipe, upto: int) -> List[List[int]]:
    rng, order, out = np.random.default_rng(recipe.seed), [], []
    for _ in range(upto):
        if len(order) < recipe.batch_size:
            order += rng.permutation(n_records).tolist()
        out.append(order[:recipe.batch_size])
        order = order[recipe.batch_size:]
    return out


def reconstruct(student, records, recipe, teacher_values, n_updates: int, spec, sink, reference) -> Dict[str, Any]:
    """ONE prefix of Stage 3's loop (search_distillation_d2.train_student) for updates 0..n_updates-1, nothing
    inside it; returns the state after the last (numpy, every parameter and BatchNorm statistic)."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    from . import search_distillation_d2 as D
    from .trainer import MainModule, clip_grad_norm, compute_masked_log_probs

    M._require_eval(student, "the reconstruction student")
    if D.state_digest(student) != spec["teacher_digest"]:
        raise Void("networks", "the reconstruction does not start from the pinned incumbent")
    teacher_values = np.asarray(teacher_values, dtype=np.float32)
    surface = MainModule(student.encoder, student.policy_head)
    opt = optim.Adam(learning_rate=recipe.learning_rate)

    def loss_fn(_surface, boards, rows, cols, mask, target, tv):
        logits, values, _ = student.forward_padded(boards, rows, cols, mask, active_size=D.BOARD)
        policy = mx.mean(-mx.sum(target * compute_masked_log_probs(logits, mask), axis=1))
        anchor = mx.mean((values - tv) ** 2)
        l2 = sum(mx.sum(p ** 2) for _, p in tree_flatten(_surface.trainable_parameters()))
        return policy + recipe.anchor_weight * anchor + recipe.l2_weight * l2, policy, anchor

    grad_fn = nn.value_and_grad(surface, loss_fn)
    rng = np.random.default_rng(recipe.seed)
    order: List[int] = []
    for step in range(n_updates):
        if len(order) < recipe.batch_size:
            order += rng.permutation(len(records)).tolist()
        idx, order = order[:recipe.batch_size], order[recipe.batch_size:]
        boards, rows, cols, mask, target = D._batch([records[i] for i in idx])
        (loss, policy, anchor), grads = grad_fn(surface, boards, rows, cols, mask, target,
                                                mx.array(teacher_values[idx]))
        grads, norm = clip_grad_norm(grads, recipe.clip_norm)
        opt.update(surface, grads)
        mx.eval(surface.parameters(), opt.state, loss, policy, anchor, norm)
        values = {"loss": float(loss), "policy": float(policy), "anchor": float(anchor), "grad_norm": float(norm)}
        sink.line("reconstruction.jsonl", {"update": step, "batch": idx, **values,
                                           "stage3": {f: reference[step][f] for f in R.TRACE_FIELDS}})
        if not all(math.isfinite(v) for v in values.values()):
            raise Void("non_finite", f"reconstruction update {step}: {values}")
    return D.snapshot(student)


# ─────────────────────────────── the run ───────────────────────────────

class _Sink:
    """Every result to disk as it exists: lines fsynced, each result file and fixture create-only."""

    def __init__(self, out_dir: pathlib.Path, fixture_dir: pathlib.Path, spec):
        self.out_dir, self.fixture_dir, self.spec = out_dir, fixture_dir, spec
        for d in ("full", "stages", "reference"):
            (out_dir / d).mkdir()
        self.files: List[str] = []
        self.fixtures: List[Dict[str, Any]] = []
        self.saved_stages: set = set()
        self.handles: Dict[str, Any] = {}

    def line(self, rel, obj):
        fh = self.handles.get(rel) or self.handles.setdefault(rel, open(self.out_dir / rel, "x"))
        fh.write(json.dumps(obj) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    def write(self, rel, obj):
        M._write_json_new(self.out_dir / rel, obj)
        self.files.append(rel)

    def fixture(self, name, arrays: Mapping[str, np.ndarray], meta) -> Dict[str, Any]:
        """A create-only .npz outside the repository; its hash, shapes and the declared tolerances in
        fixtures.jsonl inside the evidence."""
        if not self.fixture_dir.exists():
            self.fixture_dir.mkdir(parents=True)
        path = self.fixture_dir / f"{name}.npz"
        tmp = self.fixture_dir / f".{name}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in arrays.items()})
        try:
            os.link(tmp, path)
        finally:
            os.unlink(tmp)
        entry = {"name": name, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "arrays": {k: {"shape": list(np.shape(v)), "dtype": str(np.asarray(v).dtype)}
                            for k, v in arrays.items()},
                 "tolerances": {k: self.spec[k] for k in ("cross_rel_tol", "cross_abs_tol", "fd_eps", "fd_rel_tol",
                                                          "fd_mlx_rel_tol")}, **meta}
        self.fixtures.append(entry)
        self.line("fixtures.jsonl", entry)
        return {"name": name, "sha256": entry["sha256"]}

    def close(self):
        for fh in self.handles.values():
            if not fh.closed:
                fh.close()


def _execute(spec: Mapping[str, Any], seams: Seams, out_dir: pathlib.Path, fixture_dir: pathlib.Path,
             rec: Dict[str, Any], log) -> None:
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_unflatten

    from . import search_distillation_d2 as D
    from .network import canonicalize_batch

    recipe = T.recipe_for(spec)
    expected_adam = D.adam_settings(optim.Adam(learning_rate=recipe.learning_rate))
    if {k: expected_adam[k] for k in ADAM_DEFAULTS} != ADAM_DEFAULTS:
        raise Void("optimizer", f"MLX's Adam resolves to {expected_adam}, not the card's {ADAM_DEFAULTS}")
    if [s["name"] for s in spec["states"]] != ["S0", "S21"] or spec["states"][0]["prefix_updates"] != 0 or \
            spec["states"][1]["prefix_updates"] != spec["states"][1]["batch_update"] or \
            spec["devices"][0] != "cpu" or spec["repeats"] < 2:
        raise Void("plan", "the states, devices or repeats are not the card's shape")
    rec["recipe"], rec["optimizer"] = dataclasses.asdict(recipe), expected_adam
    rec["mlx"] = getattr(mx, "__version__", None)

    reference = R.read_reference(spec, seams.read_reference(), recipe)
    raw_rec, raw_dig = seams.read_repeat_record(), seams.read_repeat_digests()
    if M._sha(raw_rec) != spec["repeat_record_sha256"] or M._sha(raw_dig) != spec["repeat_digests_sha256"]:
        raise Void("repeat", "the reproducibility diagnostic's record or digests are not the pinned bytes")
    repeat_rec = json.loads(raw_rec)
    recorded = {(d["pass"], d["point"]): d for d in map(json.loads, raw_dig.decode().splitlines())}
    training, _ids, rec["corpus"] = R.read_training(spec, seams.read_corpus_json())
    log("inputs_verified", training_records=len(training))

    sha1 = seams.checkpoint_sha1()
    if sha1 != M.CHECKPOINT_SHA1:
        raise Void("checkpoint", f"SHA-1 {sha1} is not calib020_0001's")
    rec["checkpoint"] = {"path": M.CHECKPOINT, "sha1": sha1}
    teacher = seams.load_network()
    teacher.eval()
    if D.state_digest(teacher) != spec["teacher_digest"]:
        raise Void("networks", "the loaded incumbent is not the pinned state")
    rec["modules"] = seams.loaded_modules()
    sink = _Sink(out_dir, fixture_dir, spec)
    try:
        dev_train = spec["devices"][-1]
        with on(dev_train):                                  # as Stage 3 computed the anchor targets
            _, tv = D.forward(teacher, training)
        tv = np.asarray(tv, dtype=np.float32)
        rec["teacher_values"] = {"digest": _digest({"teacher_values": tv}),
                                 "equals_reproducibility_diagnostic_s": _digest({"teacher_values": tv}) ==
                                 repeat_rec.get("teacher_values_digest")}
        log("teacher_values", **rec["teacher_values"])

        states = {"S0": D.snapshot(teacher)}
        student = seams.load_network()
        student.eval()
        with on(dev_train):
            states["S21"] = reconstruct(student, training, recipe, tv, spec["states"][1]["prefix_updates"], spec,
                                        sink, reference)
        del student
        dg = {n: _digest(s) for n, s in states.items()}               # named-tensors/1, as D.state_digest
        rec["states"] = {
            "S0": {"digest": dg["S0"], "is_the_incumbent": dg["S0"] == spec["teacher_digest"]},
            "S21": {"digest": dg["S21"], "reconstructed": True,
                    "equals_pass_A_recorded_digest": dg["S21"] == recorded[("A", "before_update")]["model"],
                    "equals_passes_B_C_recorded_digest": dg["S21"] == recorded[("B", "before_update")]["model"]}}
        sink.write("states.json", rec["states"])
        log("states", **{k: v["digest"] for k, v in rec["states"].items()})

        batches = draw_batches(len(training), recipe, max(s["batch_update"] for s in spec["states"]) + 1)
        rec["results"] = {}
        for st in spec["states"]:
            name = st["name"]
            idx = batches[st["batch_update"]]
            batch_np = tuple(np.array(a) for a in D._batch([training[i] for i in idx]))
            tv_b = tv[idx]
            full = full_model(name, states[name], dg[name], seams.load_network, batch_np, tv_b, recipe, spec, sink)
            log("full_model", state=name)
            nets = {}
            for dev in spec["devices"]:
                with on(dev):
                    net = seams.load_network()
                    net.eval()
                    net.update(tree_unflatten([(k, mx.array(v)) for k, v in states[name].items()]))
                    if D.state_digest(net) != dg[name]:
                        raise Void("state", f"{name} on {dev}: the stage network is not the state")
                    nets[dev] = net
            loc = localize(name, nets, batch_np, tv_b, spec, sink)
            log("localized", state=name, first=loc["first_disagreement"])
            with on(spec["devices"][0]):                    # the real policy-head input: the CPU chain's features
                ex = _ex(batch_np, tv_b)
                bc, rc, cc, _ = canonicalize_batch(ex["boards"], ex["rows"], ex["cols"], ex["mask"], D.BOARD)
                F = nets[spec["devices"][0]].encoder(bc)
                mx.eval(F, rc, cc)
                F, rc, cc = np.array(F), np.array(rc), np.array(cc)
            refc = reference_case(f"real_{name}", nets[spec["devices"][0]].policy_head, F, rc, cc, batch_np[3],
                                  batch_np[4], np.ones(len(idx), np.float32), spec, sink)
            log("reference", case=f"real_{name}")
            rec["results"][name] = {"batch_update": st["batch_update"], "batch": idx,
                                    "full_model": {dev: {"grad_norms": full["devices"][dev]["grad_norms"],
                                                         "loss_terms": full["devices"][dev]["loss_terms"]}
                                                   for dev in spec["devices"]},
                                    "first_disagreement": loc["first_disagreement"],
                                    "reference_disagrees": {d: v["disagrees"] for d, v in refc["devices"].items()}}
            del nets
        rec["synthetic"] = {}
        for case in synthetic_cases(spec["synthetic_seed"]):
            r = reference_case(case[0], *case[1:], spec, sink)
            rec["synthetic"][case[0]] = {"disagrees": {d: v["disagrees"] for d, v in r["devices"].items()},
                                         "directional": r["directional"]}
            log("reference", case=case[0])
        rec["teacher_digest_after"] = D.state_digest(teacher)
    finally:
        sink.close()
        rec["files"] = {"written": list(sink.files), "fixtures": [{k: f[k] for k in ("name", "path", "sha256")}
                                                                  for f in sink.fixtures]}


def conclude(spec: Mapping[str, Any], seams: Seams, out_dir: pathlib.Path, fixture_dir: pathlib.Path,
             head: str | None = None) -> Tuple[int, Dict[str, Any]]:
    """THE RECORD BOUNDARY (as Stage 3's): record.json on every exit; SIGTERM -> VOID at step "terminated"."""
    if os.path.lexists(fixture_dir):
        raise Void("fixtures", f"{fixture_dir} exists")
    out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "card": CARD, "card_sha256": CARD_SHA256, "spec": dict(spec),
                           "head": head, "status": None, "written_by": "script", "diagnostic": True,
                           "fixture_dir": str(fixture_dir), "full_state_saved": False,
                           "held_out_records_read": 0, "causal_claims": None, "strength_verdict": None,
                           "environment": {"python": sys.version, "platform": platform.platform(),
                                           "executable": sys.executable},
                           "loadavg_before": os.getloadavg()}
    log_fh = open(out_dir / "progress.jsonl", "x")

    def log(event, **kw):
        log_fh.write(json.dumps({"event": event, "t": round(time.monotonic() - t0, 3), **kw}) + "\n")
        log_fh.flush()
        os.fsync(log_fh.fileno())

    def on_term(signum, _frame):
        raise Terminated(signal.Signals(signum).name)

    code = EXIT_VOID
    main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, on_term) if main_thread else None
    try:
        log("start")
        _execute(spec, seams, out_dir, fixture_dir, rec, log)
        rec["status"] = "COMPLETED"
        code = EXIT_COMPLETED
    except Terminated as t:
        rec.update(status="VOID", void={"step": "terminated", "detail": f"{t} after {time.monotonic() - t0:.1f} s"})
    except Void as v:
        rec.update(status="VOID", void={"step": v.step, "detail": v.detail})
    except Exception as e:                       # any exception is a VOID, and it is recorded
        rec.update(status="VOID", void={"step": "exception", "detail": f"{type(e).__name__}: {e}"})
    except BaseException as e:
        rec.update(status="INTERRUPTED", void={"step": "interrupted", "detail": type(e).__name__})
        raise
    finally:
        if main_thread:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            rec["total_seconds"] = round(time.monotonic() - t0, 3)
            rec["loadavg_after"] = os.getloadavg()
            log("end", status=rec["status"])
            log_fh.close()
            M._write_json_new(out_dir / "record.json", rec)
        finally:
            if main_thread:
                signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)
    return code, rec


def _real_seams(checkpoint: pathlib.Path) -> Seams:
    base = M._real_seams(checkpoint)
    return Seams(read_corpus_json=lambda: (ROOT / CORPUS_JSON).read_bytes(),
                 read_reference=lambda: (ROOT / REFERENCE).read_bytes(),
                 read_repeat_record=lambda: (ROOT / REPEAT_RECORD).read_bytes(),
                 read_repeat_digests=lambda: (ROOT / REPEAT_DIGESTS).read_bytes(),
                 checkpoint_sha1=base.checkpoint_sha1, load_network=base.load_network,
                 loaded_modules=base.loaded_modules)


def run(checkpoint: str) -> int:
    """THE PUBLIC ENTRY. No seam: the frozen spec, the real sources, the pinned card."""
    _require_gate()
    if M._sha((ROOT / CARD).read_bytes()) != CARD_SHA256:
        raise Refused(f"{CARD} is not the pinned card")
    ckpt = pathlib.Path(checkpoint)
    if not (ckpt.is_absolute() and ckpt.is_file()):
        raise Refused("--checkpoint must name an existing file by absolute path")
    out = ROOT / EVIDENCE_DIR
    if os.path.lexists(out):
        raise Refused(f"{out} exists: the run is one-shot")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fixtures = FIXTURE_PARENT / f"gradcheck-fixtures-{stamp}"
    if not FIXTURE_PARENT.is_dir() or os.path.lexists(fixtures):
        raise Refused(f"{FIXTURE_PARENT} is not a directory, or {fixtures} exists")
    code, rec = conclude(FROZEN, _real_seams(ckpt), out, fixtures, head=M._git("rev-parse", "HEAD"))
    print(json.dumps({"status": rec["status"], "void": rec.get("void"), "head": rec["head"],
                      "first_disagreement": {k: v.get("first_disagreement") for k, v in rec.get("results", {}).items()},
                      "fixtures": len(rec.get("files", {}).get("fixtures", []))}))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3 or argv[0] != "run" or argv[1] != "--checkpoint":
        print("usage: python -m scripts.GPU.alphazero.search_distillation_d2_gradcheck run --checkpoint <abs path>",
              file=sys.stderr)
        return EXIT_USAGE
    try:
        return run(argv[2])
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
