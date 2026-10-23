"""SEARCH DISTILLATION D2 -- FULL-FLOAT32 SETTING (MLX_ENABLE_TF32=0): ENCODER NONLINEAR-STAGE (BRANCH) TRACE for entry
33's INHERITED encoder gradients, in ONE fresh process launched with the setting. BUILT; run only through its wrapper.

Card: `docs/superpowers/2026-10-23-search-distillation-d2-branch-trace-card.md`, pinned below by sha256, as are the
reused code and entry 33's evidence. Nothing runs on import, and importing this module imports no MLX: `run` refuses
unless MLX_ENABLE_TF32 is exactly "0" and no `mlx` module is loaded when it starts, then unless the gate
`search_distillation_d2_branchtrace_authorization.SEARCH_DISTILLATION_D2_BRANCHTRACE_AUTHORIZED` is open (read at call
time), every pin matches and no output exists. At the incumbent on Stage 3's update-0 batch, with no optimizer update:
  STEP 1, reproduce: on each device the uninstrumented loss and gradient; digests must equal entry 33's, else
    NOT_REPRODUCED and stop.
  STEP 2, trace: the encoder and its blocks swapped, for the duration, to subclasses doing the same operations in the
    same order with a zero probe at every traced point (conv outputs, BN outputs = ReLU inputs, ReLU outputs, the
    residual sum); the 60 gradients and outputs must equal step 1's bitwise, else TRACE_NOT_FAITHFUL and stop.
  STEP 3, analysis in NumPy float64: every forward and backward local step checked from each device's own operands,
    the ReLU masks and their flips (quantified), the focus element, the backward walk, and the accounting (a linear
    float64 propagation of the flips alone to the two target weight gradients). Outcome: LOCAL_DERIVATIVE_ERROR,
    LOCAL_FORWARD_ERROR, BRANCH_CHANGE or NOT_EXPLAINED (card §7, in that order).
A difference is an observation. An input, pin, identity or plan mismatch, the adjoint self-check failing, a
non-finite result, an exception or SIGTERM (the wrapper's 10-minute ceiling) are a VOID. No retry and no recursive
expansion. Entry 32's verdict (FAIL at phase 2) is unchanged by any outcome.

    MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.search_distillation_d2_branchtrace run --checkpoint <calib020_0001>
"""
from __future__ import annotations

import contextlib
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import platform
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from . import search_distillation_d2_branchtrace_authorization as GATE
from . import search_distillation_d2_convref as C
from . import search_distillation_d2_gradcheck as K
from . import search_distillation_d2_gradtrace as R
from . import search_distillation_d2_measure as M
from . import search_distillation_d2_tf32path as E
from . import search_distillation_d2_train as T

ROOT = M.ROOT
RECORD_KIND = "search-distillation-d2-branchtrace-record/1"
CARD = "docs/superpowers/2026-10-23-search-distillation-d2-branch-trace-card.md"
CARD_SHA256 = "b2166516916326aacb8e97d24692cae0b6c656596d78de1c2597880aed4c25a3"
EVIDENCE_DIR = "docs/superpowers/evidence/search-distillation-d2-branchtrace"
FIXTURE_PARENT = pathlib.Path.home() / "twixt-cache" / "search-distillation-d2"      # outside the repository
ENTRY32 = C.ENTRY32
ENTRY33 = "docs/superpowers/evidence/search-distillation-d2-convref"
CODE = {"convref": "scripts/GPU/alphazero/search_distillation_d2_convref.py", **C.CODE}
PINNED = {CODE["convref"]: "1a399d8a9d972a0aafdd51849709462cb118e0797bb03b14742695de2a2621b6",
          **{CODE[k]: C.PINNED[C.CODE[k]] for k in C.CODE},
          f"{ENTRY32}/record.json": C.PINNED[f"{ENTRY32}/record.json"],
          f"{ENTRY33}/record.json": "c34d4f733516343ab95157d3ca32a294ecc8dadb065daf5ffc1800a463049ccf",
          f"{ENTRY33}/full.jsonl": "b2c4ba83793c9a425232ffbc311ffa765ed61816342aef156e7b396ca8b79050",
          f"{ENTRY33}/capture/cpu.json": "b46ecc6e2afd9c76498e445e9197487c553cb551743bc2386953ce0d8f2174ce",
          f"{ENTRY33}/capture/gpu.json": "643acf8dd4db6c9882656bef794fb3684df20002303a5f4bdb31521a1b4307b6"}
SETTING, SETTING_VALUE = E.SETTING, E.SETTING_VALUE
FROZEN = {**{k: C.FROZEN[k] for k in ("corpus_sha256", "teacher_digest", "training_ids_sha256", "training_games",
                                      "held_out_games", "training_records", "passes", "steps", "batch_size",
                                      "learning_rate", "anchor_weight", "l2_weight", "clip_norm", "seed", "devices",
                                      "cross_rel_tol", "cross_abs_tol", "adjoint_rel_tol", "direction_seed",
                                      "targets", "state", "batch")},
          "focus": [1, 13, 4, 42], "fixture_record": 1, "flips_recorded": 64, "elements_recorded": 64}
BLOCK_POINTS = ("z1", "a1", "h", "z2", "s", "out")                       # forward order within a block
STEM_POINTS = ("z0", "a0", "x0")
OUTCOMES = ("LOCAL_DERIVATIVE_ERROR", "LOCAL_FORWARD_ERROR", "BRANCH_CHANGE", "NOT_EXPLAINED", "NOT_REPRODUCED",
            "TRACE_NOT_FAITHFUL")
EXIT_COMPLETED, EXIT_VOID, EXIT_REFUSED, EXIT_USAGE = 0, 3, 4, 2
Void, Terminated = M.Void, M.Terminated
on, compare, Seams = K.on, E.compare, C.Seams


class Refused(Exception):
    """The run never started: the setting, MLX already loaded, gate closed, a pin, a bad argument or an output
    exists."""


def _mlx_loaded() -> bool:
    return any(m == "mlx" or m.startswith("mlx.") for m in sys.modules)


# ─────────────────────────────── the NumPy float64 reference ───────────────────────────────

def np_conv_input_grad(G, Wt) -> np.ndarray:
    """g_X[n, i, j, c] = sum_{a, b, o} G[n, i + 1 - a, j + 1 - b, o] * W[o, a, b, c] (G zero outside), in float64: the
    input gradient of C.np_conv_forward's stride-1, 'same'-padded convolution."""
    G, Wt = np.asarray(G, dtype=np.float64), np.asarray(Wt, dtype=np.float64)
    O, kh, kw, I = Wt.shape
    B, H, W, _ = G.shape
    ph, pw = kh // 2, kw // 2
    Gp = np.pad(G, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    out = np.zeros((B * H * W, I), dtype=np.float64)
    for a in range(kh):
        for b in range(kw):
            out += Gp[:, 2 * ph - a:2 * ph - a + H, 2 * pw - b:2 * pw - b + W, :].reshape(-1, O) @ Wt[:, a, b, :]
    return out.reshape(B, H, W, I)


def bn_scale(c: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.asarray(c["weight"], np.float64) / np.sqrt(np.asarray(c["var"], np.float64) + c["eps"])


def np_bn(z, c: Mapping[str, np.ndarray]) -> np.ndarray:
    """BatchNorm in evaluation mode, in float64: (z - mean) / sqrt(var + eps) * weight + bias."""
    return ((np.asarray(z, np.float64) - np.asarray(c["mean"], np.float64)) * bn_scale(c)
            + np.asarray(c["bias"], np.float64))


def adjoint_input_check(X, G, Wt, spec) -> Dict[str, Any]:
    """<convT(G, W), X> against <G, conv(X, W)>: the transposed convolution's self-check."""
    a = float((np_conv_input_grad(G, Wt) * np.asarray(X, np.float64)).sum())
    b = float((np.asarray(G, np.float64) * C.np_conv_forward(X, Wt)).sum())
    return {"lhs": a, "rhs": b, "abs_diff": abs(a - b),
            "ok": abs(a - b) <= spec["adjoint_rel_tol"] * (abs(a) + abs(b))}


# ─────────────────────────────── the instrumentation ───────────────────────────────

def _traced_block_call(self, x):
    """ResBlock.__call__'s operations in its order (conv1, bn1, relu, conv2, bn2, + residual, relu), with a zero
    probe added at each traced point and the values recorded."""
    import mlx.nn as nn

    p, v = self._trace["probes"], self._trace["values"]
    v["x_in"] = x
    z1 = self.conv1(x) + p["z1"]
    a1 = self.bn1(z1) + p["a1"]
    h = nn.relu(a1) + p["h"]
    z2 = self.conv2(h) + p["z2"]
    s = self.bn2(z2) + x + p["s"]
    v.update(z1=z1, a1=a1, z2=z2, s=s)
    return nn.relu(s) + p["out"]


def _traced_encoder_call(self, x):
    """BoardEncoder.__call__'s operations in its order (conv1, bn1, relu, the blocks), probes at the stem."""
    import mlx.nn as nn

    p, v = self._trace["probes"], self._trace["values"]
    v["input"] = x
    z0 = self.conv1(x) + p["z0"]
    a0 = self.bn1(z0) + p["a0"]
    v.update(z0=z0, a0=a0)
    x = nn.relu(a0) + p["x0"]
    for block in self.blocks:
        x = block(x)
    return x


_TRACED: Dict[Tuple[type, str], type] = {}


@contextlib.contextmanager
def instrumented(encoder):
    """The encoder and each block swapped, for the duration, to a subclass of ITS OWN class with the traced call; each
    gets a plain-attribute `_trace` dict (never a parameter or child). Yields [stem, block 0, ..., block n-1] traces."""
    mods = [encoder, *encoder.blocks]
    old = [type(m) for m in mods]
    traces = [{"probes": {}, "values": {}} for _ in mods]
    for i, (m, cls) in enumerate(zip(mods, old)):
        key = (cls, "encoder" if i == 0 else "block")
        if key not in _TRACED:
            _TRACED[key] = type(f"Traced{cls.__name__}", (cls,),
                                {"__call__": _traced_encoder_call if i == 0 else _traced_block_call})
        object.__setattr__(m, "_trace", traces[i])
        m.__class__ = _TRACED[key]
    try:
        yield traces
    finally:
        for m, cls in zip(mods, old):
            m.__class__ = cls
            object.__delattr__(m, "_trace")


def _constants(net) -> Dict[str, Any]:
    """The encoder's conv weights and biases and BN evaluation constants, as NumPy (the state's own values)."""
    def bn(m):
        return {"mean": np.array(m.running_mean), "var": np.array(m.running_var), "weight": np.array(m.weight),
                "bias": np.array(m.bias), "eps": float(m.eps)}

    def conv(m):
        return {"W": np.array(m.weight), "b": np.array(m.bias), "stride": list(m.stride), "padding": list(m.padding),
                "dilation": [int(x) for x in np.atleast_1d(m.dilation)], "groups": int(m.groups)}

    enc = net.encoder
    return {"stem": {"conv": conv(enc.conv1), "bn": bn(enc.bn1)},
            "blocks": [{"conv1": conv(b.conv1), "bn1": bn(b.bn1), "conv2": conv(b.conv2), "bn2": bn(b.bn2)}
                       for b in enc.blocks]}


def evaluate(dev, state_np, state_digest, make_net, batch_np, tv_np, recipe, spec, trace: bool) -> Dict[str, Any]:
    """One evaluation on `dev` of the training loss and its gradient (as entries 32 and 33 compute it), from a fresh
    network and fresh batch arrays; traced (probes and recorded values) if `trace`."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from . import search_distillation_d2 as D
    from .trainer import MainModule, compute_masked_log_probs

    names_in = ("boards", "rows", "cols", "mask", "target")
    in_digest = K._digest({**dict(zip(names_in, batch_np)), "teacher_values": tv_np})
    with on(dev):
        net = C._set_state(make_net, state_np, state_digest, f"{'trace' if trace else 'evaluation'} on {dev}")
        ex = K._ex(batch_np, tv_np)
        if K._digest({**{n: np.array(ex[n]) for n in names_in}, "teacher_values": np.array(ex["tv"])}) != in_digest:
            raise Void("inputs", f"{dev}: the batch arrays are not the batch")
        surface = MainModule(net.encoder, net.policy_head)

        def loss_fn(_s):                                  # full_block's training loss (unweighted), unchanged
            logits, values, _ = net.forward_padded(ex["boards"], ex["rows"], ex["cols"], ex["mask"],
                                                   active_size=D.BOARD)
            ce = -mx.sum(ex["target"] * compute_masked_log_probs(logits, ex["mask"]), axis=1)
            sq = (values - ex["tv"]) ** 2
            policy, anchor = mx.mean(ce), mx.mean(sq)
            total = policy + recipe.anchor_weight * anchor + recipe.l2_weight * sum(
                mx.sum(p ** 2) for _, p in tree_flatten(_s.trainable_parameters()))
            return total, policy, anchor, logits, values

        out: Dict[str, Any] = {"device": dev}
        if not trace:
            (loss, policy, anchor, logits, values), g = nn.value_and_grad(surface, loss_fn)(surface)
            mx.eval(loss, policy, anchor, logits, values, g)
            out["traced"] = None
        else:
            B, H, W = batch_np[0].shape[:3]
            hidden = net.encoder.conv1.weight.shape[0]
            names = [STEM_POINTS] + [BLOCK_POINTS] * len(net.encoder.blocks)
            out["constants"] = _constants(net)
            with instrumented(net.encoder) as traces:
                def inner(params, probes):
                    surface.update(params)
                    for t, ps in zip(traces, probes):
                        t["probes"] = ps
                    res = loss_fn(surface)
                    return (*res, [dict(t["values"]) for t in traces])

                probes = [{k: mx.zeros((B, H, W, hidden), dtype=mx.float32) for k in ks} for ks in names]
                outs, (g, gprobe) = mx.value_and_grad(inner, argnums=(0, 1))(surface.trainable_parameters(), probes)
                mx.eval(outs, g, gprobe)
            loss, policy, anchor, logits, values, vals = outs
            out["traced"] = {"values": [{k: np.array(v) for k, v in t.items()} for t in vals],
                             "grads": [{k: np.array(v) for k, v in t.items()} for t in gprobe]}
            K._finite(f"trace on {dev}", [a for part in ("values", "grads") for t in out["traced"][part]
                                          for a in t.values()])
        out.update(terms=[float(loss), float(policy), float(anchor)], logits=np.array(logits),
                   values=np.array(values), grads={k: np.array(v) for k, v in tree_flatten(g)})
        K._finite(f"evaluation on {dev}", [out["logits"], out["values"], *out["grads"].values()])
        del net, surface
    out["grad_digest"] = K._digest(out["grads"])
    out["logits_digest"] = K._digest({"logits": out["logits"]})
    out["values_digest"] = K._digest({"values": out["values"]})
    return out


def fidelity(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    """The traced evaluation `b` against the uninstrumented `a`: everything bitwise."""
    r = {"terms_equal": b["terms"] == a["terms"],
         "logits_equal": bool(np.array_equal(b["logits"], a["logits"])),
         "values_equal": bool(np.array_equal(b["values"], a["values"])),
         "grads_equal": set(b["grads"]) == set(a["grads"]) and all(np.array_equal(b["grads"][k], a["grads"][k])
                                                                   for k in a["grads"]),
         "grads_compared": len(a["grads"])}
    r["ok"] = all(r[k] for k in ("terms_equal", "logits_equal", "values_equal", "grads_equal"))
    return r


# ─────────────────────────────── step 3: the analysis ───────────────────────────────

def _bwd(t, b, k):
    return t["grads"][b + 1][k] if b >= 0 else t["grads"][0][k]


def _fwd(t, b, k):
    return t["values"][b + 1][k] if b >= 0 else t["values"][0][k]


def local_checks(t: Mapping[str, Any], cons: Mapping[str, Any], spec) -> Dict[str, Any]:
    """Every local forward and backward step of one device's trace, against float64 from its own operands (card §5)."""
    fwd: Dict[str, Any] = {}
    bwd: Dict[str, Any] = {}
    st = cons["stem"]
    fwd["stem.z0"] = compare(_fwd(t, -1, "z0"), C.np_conv_forward(_fwd(t, -1, "input"), st["conv"]["W"],
                                                                    st["conv"]["b"]), spec)
    fwd["stem.a0"] = compare(_fwd(t, -1, "a0"), np_bn(_fwd(t, -1, "z0"), st["bn"]), spec)
    a0, gx0 = _fwd(t, -1, "a0"), _bwd(t, -1, "x0")
    bwd["stem.relu"] = {"bitwise": bool(np.array_equal(_bwd(t, -1, "a0"), np.where(a0 > 0, gx0, 0)))}
    bwd["stem.bn"] = compare(_bwd(t, -1, "z0"), _bwd(t, -1, "a0").astype(np.float64) * bn_scale(st["bn"]), spec)
    for b, c in enumerate(cons["blocks"]):
        n = f"blocks.{b}"
        x_in, z1, a1, z2, s = (_fwd(t, b, k) for k in ("x_in", "z1", "a1", "z2", "s"))
        fwd[f"{n}.z1"] = compare(z1, C.np_conv_forward(x_in, c["conv1"]["W"], c["conv1"]["b"]), spec)
        fwd[f"{n}.a1"] = compare(a1, np_bn(z1, c["bn1"]), spec)
        fwd[f"{n}.z2"] = compare(z2, C.np_conv_forward(np.maximum(a1, 0), c["conv2"]["W"], c["conv2"]["b"]), spec)
        fwd[f"{n}.s"] = compare(s, np_bn(z2, c["bn2"]) + x_in, spec)
        g = {k: _bwd(t, b, k) for k in BLOCK_POINTS}
        bwd[f"{n}.relu_inner"] = {"bitwise": bool(np.array_equal(g["a1"], np.where(a1 > 0, g["h"], 0)))}
        bwd[f"{n}.relu_outer"] = {"bitwise": bool(np.array_equal(g["s"], np.where(s > 0, g["out"], 0)))}
        bwd[f"{n}.bn1"] = compare(g["z1"], g["a1"].astype(np.float64) * bn_scale(c["bn1"]), spec)
        bwd[f"{n}.bn2"] = compare(g["z2"], g["s"].astype(np.float64) * bn_scale(c["bn2"]), spec)
        bwd[f"{n}.conv2_input"] = compare(g["h"], np_conv_input_grad(g["z2"], c["conv2"]["W"]), spec)
        prev = _bwd(t, b - 1, "out") if b > 0 else _bwd(t, -1, "x0")
        bwd[f"{n}.block_input"] = compare(prev, g["s"].astype(np.float64) +
                                          np_conv_input_grad(g["z1"], c["conv1"]["W"]), spec)
    fails_f = [k for k, v in fwd.items() if not v["within_tol"]]
    fails_b = [k for k, v in bwd.items() if not v.get("within_tol", v.get("bitwise"))]
    return {"forward": fwd, "backward": bwd, "forward_failures": fails_f, "backward_failures": fails_b}


def relus(n_blocks: int) -> List[Tuple[str, int, str, str]]:
    """(name, block (-1 the stem), pre point, post point), in forward order."""
    out = [("stem.relu", -1, "a0", "x0")]
    for b in range(n_blocks):
        out += [(f"blocks.{b}.relu_inner", b, "a1", "h"), (f"blocks.{b}.relu_outer", b, "s", "out")]
    return out


def _pre64(t, cons, b, pre, idx) -> float:
    """The pre-activation at one element, recomputed in float64 from the device's own operands."""
    if b < 0:
        return float(np_bn(_fwd(t, -1, "z0")[idx], _chan(cons["stem"]["bn"], idx[-1])))
    c = cons["blocks"][b]
    if pre == "a1":
        return float(np_bn(_fwd(t, b, "z1")[idx], _chan(c["bn1"], idx[-1])))
    return float(np_bn(_fwd(t, b, "z2")[idx], _chan(c["bn2"], idx[-1])) + float(_fwd(t, b, "x_in")[idx]))


def _same(a, b) -> bool:
    """Nested constants equal, arrays bitwise."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    return a == b


def _chan(bn, ch):
    return {"mean": bn["mean"][ch], "var": bn["var"][ch], "weight": bn["weight"][ch], "bias": bn["bias"][ch],
            "eps": bn["eps"]}


def flips_at(name, b, pre, post, tr, cons, spec) -> Dict[str, Any]:
    """The flips of one ReLU (masks differing between the devices), counted, and up to `flips_recorded` quantified,
    largest injected gradient first."""
    cpu, gpu = spec["devices"][0], spec["devices"][-1]
    P = {d: _fwd(tr[d], b, pre) for d in (cpu, gpu)}
    m = {d: P[d] > 0 for d in (cpu, gpu)}
    gpost = {d: _bwd(tr[d], b, post) for d in (cpu, gpu)}
    gpre = {d: _bwd(tr[d], b, pre) for d in (cpu, gpu)}
    idx = np.argwhere(m[cpu] != m[gpu])
    noise = float(np.abs(P[gpu].astype(np.float64) - P[cpu]).max())
    inj = np.abs(gpost[cpu][tuple(idx.T)].astype(np.float64)) if len(idx) else np.zeros(0)
    order = np.argsort(-inj, kind="stable")[:spec["flips_recorded"]]
    recs = []
    for i in order:
        e = tuple(int(x) for x in idx[i])
        r = {"index": list(e), "noise_max_abs_pre_diff": noise}
        bn = cons["stem"]["bn"] if b < 0 else cons["blocks"][b]["bn1" if pre == "a1" else "bn2"]
        r["bn_channel"] = {k: float(v) for k, v in _chan(bn, e[-1]).items()}
        for d in (cpu, gpu):
            p64 = _pre64(tr[d], cons, b, pre, e)
            zname = "z0" if b < 0 else ("z1" if pre == "a1" else "z2")
            ops = {"bn_input": float(_fwd(tr[d], b, zname)[e])}
            if pre == "s":
                ops["residual"] = float(_fwd(tr[d], b, "x_in")[e])
            mag = max(abs(np.float32(v)) for v in ops.values())
            r[d] = {"pre": float(P[d][e]), "pre_float64_own_operands": p64, "mask": bool(m[d][e]),
                    "float64_sign_agrees": bool((p64 > 0) == m[d][e]), "operands": ops,
                    "pre_over_operand_ulp": float(abs(P[d][e]) / np.spacing(np.float32(mag))) if mag else None,
                    "grad_post": float(gpost[d][e]), "grad_pre": float(gpre[d][e])}
        r["pre_gpu_minus_cpu"] = float(P[gpu][e]) - float(P[cpu][e])
        r["abs_pre_over_noise"] = (max(abs(float(P[cpu][e])), abs(float(P[gpu][e]))) / noise) if noise else None
        recs.append(r)
    return {"relu": name, "block": b, "count": int(len(idx)), "recorded": recs, "noise_max_abs_pre_diff": noise,
            "pre_compare": compare(P[gpu], P[cpu], spec),
            "focus_flip": bool(m[cpu][tuple(spec["focus"])] != m[gpu][tuple(spec["focus"])])}


def walk(tr, n_blocks, spec) -> Dict[str, Any]:
    """The traced gradients in backward order, GPU against CPU; the first outside the tolerance is the entry point."""
    cpu, gpu = spec["devices"][0], spec["devices"][-1]
    order = [(f"blocks.{b}.{k}", b, k) for b in reversed(range(n_blocks))
             for k in ("out", "s", "z2", "h", "a1", "z1")] + [(f"stem.{k}", -1, k) for k in ("x0", "a0", "z0")]
    pts, entry = [], None
    for name, b, k in order:
        g, c = _bwd(tr[gpu], b, k), _bwd(tr[cpu], b, k)
        r = compare(g, c, spec)
        pts.append({"point": name, **{x: r[x] for x in ("within_tol", "bitwise", "max_abs", "scale")},
                    "n_outside": r.get("n_outside", 0)})
        if entry is None and not r["within_tol"]:
            d = np.abs(g.astype(np.float64) - c)
            out = np.argwhere(d > spec["cross_rel_tol"] * r["scale"] + spec["cross_abs_tol"])
            entry = {"point": name, "outside": [[int(x) for x in e] for e in out[:spec["elements_recorded"]]],
                     "n_outside": int(len(out))}
    return {"points": pts, "entry": entry}


def focus(tr, n_blocks, spec) -> Dict[str, Any]:
    """Every traced value, mask and gradient at the focus element, on both devices."""
    e = tuple(spec["focus"])
    res: Dict[str, Any] = {"element": spec["focus"]}
    for d, t in tr.items():
        res[d] = {"stem": {k: float(_fwd(t, -1, k)[e]) for k in ("z0", "a0")} |
                  {f"grad_{k}": float(_bwd(t, -1, k)[e]) for k in STEM_POINTS},
                  "blocks": [{**{k: float(_fwd(t, b, k)[e]) for k in ("x_in", "z1", "a1", "z2", "s")},
                              "mask_inner": bool(_fwd(t, b, "a1")[e] > 0), "mask_outer": bool(_fwd(t, b, "s")[e] > 0),
                              **{f"grad_{k}": float(_bwd(t, b, k)[e]) for k in BLOCK_POINTS}}
                             for b in range(n_blocks)]}
    return res


def accounting(tr, cons, spec, originals) -> Dict[str, Any]:
    """The flips' effect alone, propagated linearly in float64 from the encoder's output (delta 0) down to the lowest
    target block, with the GPU's masks: delta_pre = delta_post * m_gpu + g_post_cpu * (m_gpu - m_cpu). For each target,
    dW(X_gpu, G_cpu + dG) + L2 against the GPU's original gradient, with and without dG."""
    cpu, gpu = spec["devices"][0], spec["devices"][-1]
    blocks = {int(t.split(".")[2]): t for t in spec["targets"]}
    n = len(cons["blocks"])
    dG: Dict[int, np.ndarray] = {}
    delta = np.zeros_like(_bwd(tr[cpu], n - 1, "out"), dtype=np.float64)
    for b in reversed(range(min(blocks), n)):
        c = cons["blocks"][b]
        m_c, m_g = _fwd(tr[cpu], b, "s") > 0, _fwd(tr[gpu], b, "s") > 0
        ds = delta * m_g + _bwd(tr[cpu], b, "out").astype(np.float64) * (m_g.astype(np.float64) - m_c)
        dz2 = ds * bn_scale(c["bn2"])
        dG[b] = dz2
        dh = np_conv_input_grad(dz2, c["conv2"]["W"])
        m_c, m_g = _fwd(tr[cpu], b, "a1") > 0, _fwd(tr[gpu], b, "a1") > 0
        da1 = dh * m_g + _bwd(tr[cpu], b, "h").astype(np.float64) * (m_g.astype(np.float64) - m_c)
        delta = np_conv_input_grad(da1 * bn_scale(c["bn1"]), c["conv1"]["W"]) + ds
    res: Dict[str, Any] = {"targets": {}}
    for b, t in sorted(blocks.items()):
        c = cons["blocks"][b]
        W = c["conv2"]["W"]
        X = np.maximum(_fwd(tr[gpu], b, "a1"), 0)
        Gc = _bwd(tr[cpu], b, "z2").astype(np.float64)
        l2 = 2.0 * spec["l2_weight"] * W.astype(np.float64)
        with_ = C.np_conv_weight_grad(X, Gc + dG[b], W.shape) + l2
        without = C.np_conv_weight_grad(X, Gc, W.shape) + l2
        actual_dG = _bwd(tr[gpu], b, "z2").astype(np.float64) - Gc
        r = {"with_flips": compare(with_, originals[gpu][t], spec),
             "without_flips": compare(without, originals[gpu][t], spec),
             "cpu_original_vs_gpu_original": compare(originals[cpu][t], originals[gpu][t], spec),
             "predicted_dG_vs_actual_dG": compare(dG[b], actual_dG, spec),
             "predicted_dG_nonzero": int(np.count_nonzero(dG[b]))}
        r["explained"] = bool(r["with_flips"]["within_tol"] and not r["without_flips"]["within_tol"])
        res["targets"][t] = r
    res["both_explained"] = all(r["explained"] for r in res["targets"].values())
    return res


def outcome(checks, flips, acc, spec) -> str:
    """Card §7, in order."""
    if any(c["backward_failures"] for c in checks.values()):
        return "LOCAL_DERIVATIVE_ERROR"
    if any(c["forward_failures"] for c in checks.values()):
        return "LOCAL_FORWARD_ERROR"
    lo = min(int(t.split(".")[2]) for t in spec["targets"])
    relevant = sum(f["count"] for f in flips.values() if f["block"] >= lo)
    if relevant and acc["both_explained"]:
        return "BRANCH_CHANGE"
    return "NOT_EXPLAINED"


# ─────────────────────────────── the run ───────────────────────────────

class _Sink(C._Sink):
    """Entry 33's sink (lines fsynced, each result file and fixture create-only), with this trace's directories."""

    def __init__(self, out_dir: pathlib.Path, fixture_dir: pathlib.Path, spec):
        self.out_dir, self.fixture_dir = out_dir, fixture_dir
        self.spec = {**spec, "recon_rel_tol": None}                     # the fixture's recorded tolerances
        for d in ("trace", "stages"):
            (out_dir / d).mkdir()
        self.files: List[str] = []
        self.fixtures: List[Dict[str, Any]] = []
        self.handles: Dict[str, Any] = {}


def _execute(spec: Mapping[str, Any], seams: Seams, out_dir: pathlib.Path, fixture_dir: pathlib.Path,
             rec: Dict[str, Any], log) -> None:
    import mlx.core as mx

    from . import search_distillation_d2 as D

    if pathlib.Path(D.__file__).resolve() != ROOT / CODE["d2"]:
        raise Void("code", f"{D.__file__} is not {CODE['d2']}")
    recipe = T.recipe_for(spec)
    if spec["devices"] != ["cpu", "gpu"] or spec["state"] != "S0" or spec["batch"] != 0 or not spec["targets"] or \
            not all(C._TARGET.match(t) for t in spec["targets"]) or len(spec["focus"]) != 4:
        raise Void("plan", "the devices, state, batch, targets or focus are not the card's shape")
    rec["recipe"], rec["mlx"] = dataclasses.asdict(recipe), getattr(mx, "__version__", None)

    for rel in [r for r in PINNED if r.startswith((ENTRY32, ENTRY33))]:
        if M._sha(seams.read_repo(rel)) != PINNED[rel]:
            raise Void("evidence", f"{rel} is not the pinned bytes")
    entry32 = json.loads(seams.read_repo(f"{ENTRY32}/record.json"))
    entry33 = json.loads(seams.read_repo(f"{ENTRY33}/record.json"))
    lines33 = [json.loads(x) for x in seams.read_repo(f"{ENTRY33}/full.jsonl").decode().splitlines()]
    expect = {}
    for dev in spec["devices"]:
        cap = json.loads(seams.read_repo(f"{ENTRY33}/capture/{dev}.json"))["uninstrumented"]
        line = next(x for x in lines33 if x["device"] == dev and x["repeat"] == 0)
        expect[dev] = {"grad_digest": cap["grad_digest"], "terms": cap["terms"],
                       "logits_digest": line["logits_digest"], "values_digest": line["values_digest"]}
    training, _ids, rec["corpus"] = R.read_training(spec, seams.read_corpus_json())
    sha1 = seams.checkpoint_sha1()
    if sha1 != M.CHECKPOINT_SHA1:
        raise Void("checkpoint", f"SHA-1 {sha1} is not calib020_0001's")
    rec["checkpoint"] = {"path": M.CHECKPOINT, "sha1": sha1}
    draw = K.draw_batches(len(training), recipe, spec["batch"] + 1)[spec["batch"]]
    if draw != entry32["batches"][str(spec["batch"])]:
        raise Void("batch", "the update-0 draw is not the one entry 32 recorded")
    rec["batch_draw"] = draw
    log("inputs_verified", training_records=len(training))

    sink = _Sink(out_dir, fixture_dir, spec)
    try:
        teacher = seams.load_network()
        teacher.eval()
        if D.state_digest(teacher) != spec["teacher_digest"]:
            raise Void("networks", "the loaded incumbent is not the pinned state")
        rec["modules"] = seams.loaded_modules()
        with on(spec["devices"][-1]):                            # as entries 32 and 33 computed the anchor targets
            _, tv = D.forward(teacher, training)
        tv = np.asarray(tv, dtype=np.float32)
        K._finite("teacher values", [tv])
        rec["teacher_values"] = {"digest": K._digest({"teacher_values": tv}),
                                 "entry33_digest": entry33["teacher_values"]["digest"]}
        rec["teacher_values"]["equals_entry33"] = rec["teacher_values"]["digest"] == entry33["teacher_values"]["digest"]
        log("teacher_values", **rec["teacher_values"])
        batch = tuple(np.array(a) for a in D._batch([training[i] for i in draw]))
        s0 = D.snapshot(teacher)
        dg = K._digest(s0)
        del teacher

        plain, traced = {}, {}
        rec["step1"], rec["step2"] = {}, {}
        for dev in spec["devices"]:
            plain[dev] = evaluate(dev, s0, dg, seams.load_network, batch, tv[draw], recipe, spec, trace=False)
            got = {"grad_digest": plain[dev]["grad_digest"], "terms": plain[dev]["terms"],
                   "logits_digest": plain[dev]["logits_digest"], "values_digest": plain[dev]["values_digest"]}
            rec["step1"][dev] = {"got": got, "expected": expect[dev], "ok": got == expect[dev]}
        cpu, gpu = spec["devices"][0], spec["devices"][-1]
        rec["step1"]["targets_gpu_vs_cpu"] = {t: compare(plain[gpu]["grads"][t], plain[cpu]["grads"][t], spec)
                                              for t in spec["targets"]}
        rec["step1"]["ok"] = all(rec["step1"][d]["ok"] for d in spec["devices"])
        log("step1", ok=rec["step1"]["ok"])
        if not rec["step1"]["ok"]:
            rec["outcome"] = "NOT_REPRODUCED"
            return

        for dev in spec["devices"]:
            traced[dev] = evaluate(dev, s0, dg, seams.load_network, batch, tv[draw], recipe, spec, trace=True)
            fid = fidelity(plain[dev], traced[dev])
            r = {"device": dev, "step1": rec["step1"][dev], "fidelity": fid, "grad_digest": traced[dev]["grad_digest"]}
            fr = spec["fixture_record"]
            arrays = {"input": traced[dev]["traced"]["values"][0]["input"][fr]}
            for i, part in enumerate(traced[dev]["traced"]["values"]):
                arrays.update({f"{'stem' if i == 0 else f'blocks.{i - 1}'}.{k}": v[fr]
                               for k, v in part.items() if k != "input"})
            for i, part in enumerate(traced[dev]["traced"]["grads"]):
                arrays.update({f"grad.{'stem' if i == 0 else f'blocks.{i - 1}'}.{k}": v[fr] for k, v in part.items()})
            r["fixture"] = sink.fixture(f"{dev}_record{fr}", arrays, {"device": dev, "record": fr,
                                                                      "state": spec["state"], "batch": spec["batch"]})
            sink.write(f"trace/{dev}.json", r)
            rec["step2"][dev] = {"faithful": fid["ok"]}
            log("trace", device=dev, faithful=fid["ok"])
        if not all(v["faithful"] for v in rec["step2"].values()):
            rec["outcome"] = "TRACE_NOT_FAITHFUL"
            return

        cons = {d: traced[d]["constants"] for d in spec["devices"]}
        if not _same(cons[cpu], cons[gpu]):
            raise Void("state", "the encoder constants differ between devices")
        tr = {d: traced[d]["traced"] for d in spec["devices"]}
        del traced
        c0 = cons[cpu]
        n_blocks = len(c0["blocks"])
        for b in range(n_blocks):
            for k in ("conv1", "conv2"):
                cc = c0["blocks"][b][k]
                if not C._conv_shape_ok(cc):
                    raise Void("plan", f"blocks.{b}.{k} is not a stride-1, padding-1 convolution")
        tb = min(n_blocks - 1, 2)
        adj = adjoint_input_check(_fwd(tr[cpu], tb, "x_in"), _bwd(tr[cpu], tb, "z1"), c0["blocks"][tb]["conv1"]["W"],
                                  spec)
        rec["self_check"] = {"adjoint_input_grad": adj, "block": tb}
        if not adj["ok"]:
            raise Void("reference", f"the float64 transposed convolution failed its adjoint self-check ({adj})")

        checks = {d: local_checks(tr[d], c0, spec) for d in spec["devices"]}
        flips = {}
        for name, b, pre, post in relus(n_blocks):
            f = flips_at(name, b, pre, post, tr, c0, spec)
            flips[name] = f
            stage = "stem" if b < 0 else f"blocks.{b}"
            sink.write(f"stages/{name}.json", {**f, "local_checks": {
                d: {"forward": {k: v for k, v in checks[d]["forward"].items() if k.startswith(stage + ".")},
                    "backward": {k: v for k, v in checks[d]["backward"].items() if k.startswith(stage + ".")}}
                for d in spec["devices"]}})
        log("stages", flips={k: v["count"] for k, v in flips.items()})
        rec["local_checks"] = {d: {"forward_failures": checks[d]["forward_failures"],
                                   "backward_failures": checks[d]["backward_failures"]} for d in spec["devices"]}
        rec["flips"] = {k: {"count": v["count"], "focus_flip": v["focus_flip"]} for k, v in flips.items()}
        w = walk(tr, n_blocks, spec)
        sink.write("walk.json", w)
        rec["walk_entry"] = w["entry"]
        sink.write("focus.json", focus(tr, n_blocks, spec))
        acc = accounting(tr, c0, spec, {d: plain[d]["grads"] for d in spec["devices"]})
        sink.write("accounting.json", acc)
        rec["accounting"] = {t: {"explained": v["explained"], "with_flips": v["with_flips"],
                                 "without_flips": v["without_flips"]} for t, v in acc["targets"].items()}
        log("accounting", both_explained=acc["both_explained"])
        rec["outcome"] = outcome(checks, flips, acc, spec)
    finally:
        sink.close()
        rec["files"] = list(sink.files)
        rec["fixtures"] = [{k: f[k] for k in ("name", "path", "sha256", "bytes")} for f in sink.fixtures]


def conclude(spec: Mapping[str, Any], seams: Seams, out_dir: pathlib.Path, fixture_dir: pathlib.Path,
             head: str | None = None) -> Tuple[int, Dict[str, Any]]:
    """THE RECORD BOUNDARY: record.json on every exit this survives; SIGTERM -> VOID at step "terminated"."""
    out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "card": CARD, "card_sha256": CARD_SHA256, "spec": dict(spec),
                           "head": head, "status": None, "outcome": None, "written_by": "script", "diagnostic": True,
                           "setting": {SETTING: os.environ.get(SETTING)}, "fixture_dir": str(fixture_dir),
                           "entry32_verdict": "FAIL at phase 2 (unchanged)", "optimizer_updates": 0,
                           "model_state_saved": False, "held_out_records_read": 0, "causal_claims": None,
                           "strength_verdict": None, "elapsed_is_a_benchmark": False,
                           "environment": {"python": sys.version, "platform": platform.platform(),
                                           "executable": sys.executable, "numpy": np.__version__},
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
        log("start", setting=rec["setting"])
        _execute(spec, seams, out_dir, fixture_dir, rec, log)
        if rec["outcome"] not in OUTCOMES:
            raise Void("plan", f"no outcome ({rec['outcome']!r})")
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
            if rec["status"] != "COMPLETED":
                rec["outcome"] = None
            rec["setting_at_end"] = {SETTING: os.environ.get(SETTING)}
            rec["total_seconds"] = round(time.monotonic() - t0, 3)
            rec["loadavg_after"] = os.getloadavg()
            log("end", status=rec["status"], outcome=rec["outcome"])
            log_fh.close()
            M._write_json_new(out_dir / "record.json", rec)
        finally:
            if main_thread:
                signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)
    return code, rec


def run(checkpoint: str) -> int:
    """THE PUBLIC ENTRY. No seam: the setting, the frozen spec, the pinned card, code and evidence, the real
    sources."""
    if os.environ.get(SETTING) != SETTING_VALUE:
        raise Refused(f"{SETTING} is {os.environ.get(SETTING)!r}, not {SETTING_VALUE!r}: launch a fresh process with it")
    if _mlx_loaded():
        raise Refused("MLX was imported before the setting was checked: launch a fresh process")
    if GATE.SEARCH_DISTILLATION_D2_BRANCHTRACE_AUTHORIZED is not True:
        raise Refused("SEARCH_DISTILLATION_D2_BRANCHTRACE_AUTHORIZED is closed: the trace is not authorized")
    if M._sha((ROOT / CARD).read_bytes()) != CARD_SHA256:
        raise Refused(f"{CARD} is not the pinned card")
    for rel, sha in PINNED.items():
        if M._sha((ROOT / rel).read_bytes()) != sha:
            raise Refused(f"{rel} is not the pinned bytes")
    for mod, rel in ((C, CODE["convref"]), (E, CODE["tf32path"]), (K, CODE["gradcheck"]), (R, CODE["gradtrace"])):
        if pathlib.Path(mod.__file__).resolve() != ROOT / rel:
            raise Refused(f"{mod.__file__} is not the pinned {rel}")
    ckpt = pathlib.Path(checkpoint)
    if not (ckpt.is_absolute() and ckpt.is_file()):
        raise Refused("--checkpoint must name an existing file by absolute path")
    out = ROOT / EVIDENCE_DIR
    if os.path.lexists(out):
        raise Refused(f"{out} exists: the run is one-shot")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fixture_dir = FIXTURE_PARENT / f"branchtrace-fixtures-{stamp}"
    if os.path.lexists(fixture_dir):
        raise Refused(f"{fixture_dir} exists")
    code, rec = conclude(FROZEN, C._real_seams(ckpt), out, fixture_dir, head=M._git("rev-parse", "HEAD"))
    print(json.dumps({"status": rec["status"], "void": rec.get("void"), "outcome": rec.get("outcome"),
                      "step1_ok": rec.get("step1", {}).get("ok"), "step2": rec.get("step2"),
                      "flips": rec.get("flips"), "walk_entry": rec.get("walk_entry"),
                      "accounting": {t: v["explained"] for t, v in rec.get("accounting", {}).items()},
                      "fixture_dir": str(fixture_dir), "head": rec["head"]}))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3 or argv[0] != "run" or argv[1] != "--checkpoint":
        print("usage: MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.search_distillation_d2_branchtrace run "
              "--checkpoint <abs path>", file=sys.stderr)
        return EXIT_USAGE
    try:
        return run(argv[2])
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
