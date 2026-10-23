"""SEARCH DISTILLATION D2 -- FULL-FLOAT32 SETTING (MLX_ENABLE_TF32=0): CONVOLUTION-REFERENCE DIAGNOSTIC for entry 32's
two failing encoder gradients, in ONE fresh process launched with the setting. BUILT; run only through its wrapper.

Card: `docs/superpowers/2026-10-23-search-distillation-d2-conv-reference-card.md`, pinned below by sha256, as are the
reused code and entry 32's evidence. Nothing runs on import, and importing this module imports no MLX: `run` refuses
unless MLX_ENABLE_TF32 is exactly "0" and no `mlx` module is loaded when it starts, then unless the gate
`search_distillation_d2_convref_authorization.SEARCH_DISTILLATION_D2_CONVREF_AUTHORIZED` is open (read at call time),
every pin matches and no output exists. At the incumbent on Stage 3's update-0 batch, with no optimizer update:
  STEP 1, reproduce: entry 32's `full_block`, unchanged; the GPU's failures must be exactly entry 32's six (the two
    targets in each of three evaluations), else NOT_REPRODUCED and stop.
  STEP 2, capture, on each device: the uninstrumented gradient (its digest must be step 1's); the same loss with a
    zero probe added to each target convolution's output, so the probe's gradient is the upstream gradient G and the
    convolution's input X is returned beside the loss (all 60 gradients bitwise equal to the uninstrumented ones); and
    the local reconstruction, the stage VJP from X and G plus the separately calculated L2 contribution, against the
    original gradient. Any miss: CAPTURE_NOT_FAITHFUL and stop. X, G and the local gradient go to create-only
    fixtures outside the repository as soon as they exist.
  STEP 3, reference: NumPy float64 weight gradients from the captured operands (validated first by a layout check
    against MLX's forward convolution and an adjoint identity, else VOID); each device's local gradient on IDENTICAL
    operands (each device's set, copied to both), then the native operands separately; each target classified by
    three flags (local_gpu, local_cpu, inherited).
A difference is an observation. An input, pin, identity or plan mismatch, a reference failing its self-checks, a
non-finite result, an exception or SIGTERM (the wrapper's 10-minute ceiling) are a VOID. No retry. Entry 32's verdict
(FAIL at phase 2) is unchanged by any outcome.

    MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.search_distillation_d2_convref run --checkpoint <calib020_0001>
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
import re
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from . import search_distillation_d2_convref_authorization as GATE
from . import search_distillation_d2_gradcheck as K
from . import search_distillation_d2_gradtrace as R
from . import search_distillation_d2_measure as M
from . import search_distillation_d2_tf32path as E
from . import search_distillation_d2_train as T

ROOT = M.ROOT
RECORD_KIND = "search-distillation-d2-convref-record/1"
CARD = "docs/superpowers/2026-10-23-search-distillation-d2-conv-reference-card.md"
CARD_SHA256 = "155011acc1db1dd4b86cb65d12e7da078d24c7b1c1544e7a6a9921f37eb127da"
EVIDENCE_DIR = "docs/superpowers/evidence/search-distillation-d2-convref"
FIXTURE_PARENT = pathlib.Path.home() / "twixt-cache" / "search-distillation-d2"      # outside the repository
ENTRY32 = "docs/superpowers/evidence/search-distillation-d2-tf32path"
CODE = {"tf32path": "scripts/GPU/alphazero/search_distillation_d2_tf32path.py",
        "gradcheck": "scripts/GPU/alphazero/search_distillation_d2_gradcheck.py",
        "gradtrace": "scripts/GPU/alphazero/search_distillation_d2_gradtrace.py",
        "d2": "scripts/GPU/alphazero/search_distillation_d2.py"}
PINNED = {CODE["tf32path"]: "8afaf9d097e1beb14cc2fa93c6c2c42325dde8dfa4eb6da8e9d701a866371762",
          CODE["gradcheck"]: E.PINNED[E.CODE["gradcheck"]],
          CODE["gradtrace"]: E.PINNED[E.CODE["gradtrace"]],
          CODE["d2"]: E.PINNED[E.CODE["d2"]],
          f"{ENTRY32}/record.json": "0a8dd3ddfcb6d4cbd5ff7073e4f2883b7d341612d604dd20514a380e1d4905c8",
          f"{ENTRY32}/full.jsonl": "47599a5fd886418c038fa3167f8a4872bc05d744ea96695f433a0a5fb29d1b4b",
          f"{ENTRY32}/full/S0_b0.json": "123c538c8580b2768c65d20aefd8d2022b3820fbac8e305a6a4e868d1421de80"}
SETTING, SETTING_VALUE = E.SETTING, E.SETTING_VALUE
TARGETS = ["encoder.blocks.1.conv2.weight", "encoder.blocks.2.conv2.weight"]
FROZEN = {**{k: E.FROZEN[k] for k in ("corpus_sha256", "teacher_digest", "training_ids_sha256", "training_games",
                                      "held_out_games", "training_records", "passes", "steps", "batch_size",
                                      "learning_rate", "anchor_weight", "l2_weight", "clip_norm", "seed", "devices",
                                      "repeats", "cross_rel_tol", "cross_abs_tol")},
          "state": "S0", "batch": 0, "targets": TARGETS, "reproduce_failing": TARGETS,
          "recon_rel_tol": 1e-6, "adjoint_rel_tol": 1e-9, "direction_seed": 0, "elements_recorded": 64}
OUTCOMES = ("CLASSIFIED", "NOT_REPRODUCED", "CAPTURE_NOT_FAITHFUL")
CLASSES = {(True, False, False): "LOCAL_GPU", (False, True, False): "LOCAL_CPU", (False, False, True): "INHERITED",
           (True, False, True): "LOCAL_GPU_AND_INHERITED", (False, True, True): "LOCAL_CPU_AND_INHERITED",
           (False, False, False): "NEITHER_ALONE"}
EXIT_COMPLETED, EXIT_VOID, EXIT_REFUSED, EXIT_USAGE = 0, 3, 4, 2
Void, Terminated = M.Void, M.Terminated
on, compare = K.on, E.compare
_TARGET = re.compile(r"^encoder\.blocks\.(\d+)\.conv2\.weight$")


class Refused(Exception):
    """The run never started: the setting, MLX already loaded, gate closed, a pin, a bad argument or an output
    exists."""


def _mlx_loaded() -> bool:
    return any(m == "mlx" or m.startswith("mlx.") for m in sys.modules)


@dataclasses.dataclass
class Seams:
    """What the run reads from outside this module. TEST seams; `run` passes `_real_seams` only."""
    read_corpus_json: Callable[[], bytes]
    checkpoint_sha1: Callable[[], str]
    load_network: Callable[[], Any]                    # a FRESH load of the incumbent on every call
    loaded_modules: Callable[[], Dict[str, str]]
    read_repo: Callable[[str], bytes]                  # a pinned repository file, by its relative path


# ─────────────────────────────── the NumPy float64 reference ───────────────────────────────

def np_conv_weight_grad(X, G, kshape) -> np.ndarray:
    """dW[o, a, b, c] = sum_{n, y, x} G[n, y, x, o] * Xpad[n, y + a, x + b, c], in float64: a stride-1, 'same'-padded
    convolution in MLX's layouts (X and G NHWC, the weight (out, kh, kw, in)). One matrix product per kernel offset."""
    X, G = np.asarray(X, dtype=np.float64), np.asarray(G, dtype=np.float64)
    O, kh, kw, I = kshape
    H, W = G.shape[1:3]
    Xp = np.pad(X, ((0, 0), (kh // 2, kh // 2), (kw // 2, kw // 2), (0, 0)))
    G2 = G.reshape(-1, O).T
    dW = np.empty(kshape, dtype=np.float64)
    for a in range(kh):
        for b in range(kw):
            dW[:, a, b, :] = G2 @ Xp[:, a:a + H, b:b + W, :].reshape(-1, I)
    return dW


def np_conv_forward(X, Wt, bias=None) -> np.ndarray:
    """out[n, y, x, o] = sum_{a, b, c} Xpad[n, y + a, x + b, c] * W[o, a, b, c] (+ bias[o]), in float64."""
    X, Wt = np.asarray(X, dtype=np.float64), np.asarray(Wt, dtype=np.float64)
    O, kh, kw, I = Wt.shape
    B, H, W, _ = X.shape
    Xp = np.pad(X, ((0, 0), (kh // 2, kh // 2), (kw // 2, kw // 2), (0, 0)))
    out = np.zeros((B * H * W, O), dtype=np.float64)
    for a in range(kh):
        for b in range(kw):
            out += Xp[:, a:a + H, b:b + W, :].reshape(-1, I) @ Wt[:, a, b, :].T
    out = out.reshape(B, H, W, O)
    return out if bias is None else out + np.asarray(bias, dtype=np.float64)


def adjoint_check(X, G, dW, spec) -> Dict[str, Any]:
    """<dW, V> against <G, conv(X, V)> for one fixed random direction V: the reference's self-check."""
    V = np.random.default_rng(spec["direction_seed"]).standard_normal(dW.shape)
    a = float((dW * V).sum())
    b = float((np.asarray(G, dtype=np.float64) * np_conv_forward(X, V)).sum())
    return {"lhs": a, "rhs": b, "abs_diff": abs(a - b),
            "ok": abs(a - b) <= spec["adjoint_rel_tol"] * (abs(a) + abs(b))}


# ─────────────────────────────── step 2: the capture ───────────────────────────────

def target_conv(net, name: str):
    m = _TARGET.match(name)
    if m is None:
        raise Void("plan", f"{name} is not an encoder block's conv2 weight")
    return net.encoder.blocks[int(m.group(1))].conv2


_PROBED: Dict[type, type] = {}


@contextlib.contextmanager
def instrumented(convs):
    """Each convolution's class swapped, for the duration, for a subclass of ITS OWN class that records its input and
    adds a probe to its output (after the bias): `slots[i]` holds the probe in and the input out."""
    slots = [{} for _ in convs]
    old = [type(c) for c in convs]
    for c, s, cls in zip(convs, slots, old):
        if cls not in _PROBED:
            def call(self, x, _cls=cls):
                self._slot["x"] = x
                return _cls.__call__(self, x) + self._slot["probe"]
            _PROBED[cls] = type(f"Probed{cls.__name__}", (cls,), {"__call__": call})
        object.__setattr__(c, "_slot", s)             # a plain attribute: never a parameter or child
        c.__class__ = _PROBED[cls]
    try:
        yield slots
    finally:
        for c, cls in zip(convs, old):
            c.__class__ = cls
            object.__delattr__(c, "_slot")


def _set_state(make_net, state_np, state_digest, where):
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    from . import search_distillation_d2 as D

    net = make_net()
    net.eval()
    net.update(tree_unflatten([(k, mx.array(v)) for k, v in state_np.items()]))
    if D.state_digest(net) != state_digest:
        raise Void("state", f"{where}: the network is not the state")
    return net


def _local(conv, X, G):
    """The convolution's local weight gradient from its input X and upstream gradient G (the stage VJP helper)."""
    _, _, grads = K._vjp(conv, lambda a: conv(a), X, G)
    return grads["weight"]


def _l2_term(w, recipe):
    import mlx.core as mx
    return mx.grad(lambda v: recipe.l2_weight * mx.sum(v ** 2))(w)


def capture(dev, state_np, state_digest, make_net, batch_np, tv_np, recipe, spec, step1, sink, log) -> Dict[str, Any]:
    """Step 2 on one device: (a) the uninstrumented loss and gradient, as `full_block` computes them; (b) the same with
    the targets instrumented; (c) the local reconstruction. Returns the operands (in memory) and the result."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from . import search_distillation_d2 as D
    from .trainer import MainModule, compute_masked_log_probs

    names_in = ("boards", "rows", "cols", "mask", "target")
    in_digest = K._digest({**dict(zip(names_in, batch_np)), "teacher_values": tv_np})
    res: Dict[str, Any] = {"device": dev, "targets": {}}
    with on(dev):
        net = _set_state(make_net, state_np, state_digest, f"capture on {dev}")
        ex = K._ex(batch_np, tv_np)
        if K._digest({**{n: np.array(ex[n]) for n in names_in}, "teacher_values": np.array(ex["tv"])}) != in_digest:
            raise Void("inputs", f"capture on {dev}: the batch arrays are not the batch")
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

        (loss, policy, anchor, logits, values), g = nn.value_and_grad(surface, loss_fn)(surface)
        mx.eval(loss, policy, anchor, logits, values, g)
        a = {"terms": [float(loss), float(policy), float(anchor)], "logits": np.array(logits),
             "values": np.array(values), "grads": {k: np.array(v) for k, v in tree_flatten(g)}}
        K._finite(f"capture (a) on {dev}", [a["logits"], a["values"], *a["grads"].values()])
        res["uninstrumented"] = {"grad_digest": K._digest(a["grads"]), "step1_grad_digest": step1["grad_digest"],
                                 "terms": a["terms"], "step1_terms": [step1[t] for t in ("loss", "policy", "anchor")]}
        res["uninstrumented"]["ok"] = (res["uninstrumented"]["grad_digest"] == step1["grad_digest"] and
                                       res["uninstrumented"]["terms"] == res["uninstrumented"]["step1_terms"])

        convs = [target_conv(net, t) for t in spec["targets"]]
        B, H, W = batch_np[0].shape[:3]
        with instrumented(convs) as slots:
            def inner(params, probes):
                surface.update(params)
                for s, p in zip(slots, probes):
                    s["probe"] = p
                return (*loss_fn(surface), [s["x"] for s in slots])

            probes = [mx.zeros((B, H, W, c.weight.shape[0]), dtype=mx.float32) for c in convs]
            outs, (gp, gprobe) = mx.value_and_grad(inner, argnums=(0, 1))(surface.trainable_parameters(), probes)
            mx.eval(outs, gp, gprobe)
        b = {"terms": [float(x) for x in outs[:3]], "logits": np.array(outs[3]), "values": np.array(outs[4]),
             "grads": {k: np.array(v) for k, v in tree_flatten(gp)}}
        res["instrumented"] = {
            "terms_equal": b["terms"] == a["terms"],
            "logits_equal": bool(np.array_equal(b["logits"], a["logits"])),
            "values_equal": bool(np.array_equal(b["values"], a["values"])),
            "grads_equal": set(b["grads"]) == set(a["grads"]) and all(np.array_equal(b["grads"][k], a["grads"][k])
                                                                      for k in a["grads"]),
            "grads_compared": len(a["grads"])}
        res["instrumented"]["ok"] = all(res["instrumented"][k] for k in ("terms_equal", "logits_equal",
                                                                         "values_equal", "grads_equal"))
        operands: Dict[str, Dict[str, np.ndarray]] = {}
        for t, conv, xin, gin in zip(spec["targets"], convs, outs[5], gprobe):
            local, l2 = _local(conv, xin, gin), _l2_term(conv.weight, recipe)
            recon = local + l2
            mx.eval(local, l2, recon)
            X, G, loc, l2n, rec = (np.array(v) for v in (xin, gin, local, l2, recon))
            K._finite(f"capture (c) {t} on {dev}", [X, G, loc, l2n, rec])
            orig = a["grads"][t]
            d = float(np.abs(rec.astype(np.float64) - orig).max())
            scale = float(np.abs(orig).max())
            r = {"recon_bitwise": bool(np.array_equal(rec, orig)), "recon_max_abs": d, "orig_scale": scale,
                 "recon_ok": d <= spec["recon_rel_tol"] * scale,
                 "l2_max_abs": float(np.abs(l2n).max()),
                 "local_alone_max_abs": float(np.abs(loc.astype(np.float64) - orig).max()),
                 "weight_digest": K._digest({"w": np.array(conv.weight)}),
                 "orig_digest": K._digest({"g": orig}), "l2_digest": K._digest({"l2": l2n}),
                 "X": {"shape": list(X.shape), "digest": K._digest({"X": X})},
                 "G": {"shape": list(G.shape), "digest": K._digest({"G": G})},
                 "conv": {"stride": list(conv.stride), "padding": list(conv.padding),
                          "dilation": [int(x) for x in np.atleast_1d(conv.dilation)], "groups": int(conv.groups)}}
            r["fixture"] = sink.fixture(f"{dev}_{t}", {"X": X, "G": G, "local": loc},
                                        {"device": dev, "target": t, "state": spec["state"], "batch": spec["batch"]})
            res["targets"][t] = r
            operands[t] = {"X": X, "G": G, "local": loc, "l2": l2n, "orig": orig, "W": np.array(conv.weight),
                           "bias": np.array(conv.bias)}
        del net, surface
    res["faithful"] = bool(res["uninstrumented"]["ok"] and res["instrumented"]["ok"] and
                           all(r["recon_ok"] for r in res["targets"].values()))
    sink.write(f"capture/{dev}.json", res)
    log("capture", device=dev, faithful=res["faithful"])
    return {"result": res, "operands": operands}


# ─────────────────────────────── step 3: the reference and the comparisons ───────────────────────────────

def _conv_shape_ok(c) -> bool:
    return (c["stride"] == [1, 1] and c["padding"] == [1, 1] and all(x == 1 for x in c["dilation"])
            and c["groups"] == 1)


def flags_of(r: Mapping[str, Any]) -> Dict[str, bool]:
    """The three flags of one target's step-3 result (card §7)."""
    ident = r["identical_operands"]
    return {"local_gpu": any(not v["within_tol"] for s in ident.values() for v in s["gpu_vs_reference"]),
            "local_cpu": any(not v["within_tol"] for s in ident.values() for v in s["cpu_vs_reference"]),
            "inherited": not r["native"]["reference_gpu_ops_vs_reference_cpu_ops"]["within_tol"]}


def classify(flags: Mapping[str, bool]) -> str:
    if flags["local_gpu"] and flags["local_cpu"]:
        return "BOTH_KERNELS"
    return CLASSES[(flags["local_gpu"], flags["local_cpu"], flags["inherited"])]


def overall(classes: Mapping[str, str]) -> str:
    vals = set(classes.values())
    return vals.pop() if len(vals) == 1 else "MIXED"


def reference_step(spec, caps, make_net, state_np, state_digest, sink, log) -> Dict[str, Any]:
    """Step 3, per target: the float64 reference (self-checked first), identical operands, native operands, and the
    elements that failed."""
    import mlx.core as mx

    devs = spec["devices"]
    convs: Dict[str, Dict[str, Any]] = {}
    nets = {}
    for dev in devs:                                     # one fresh network per device, for its convolution modules
        with on(dev):
            nets[dev] = _set_state(make_net, state_np, state_digest, f"reference on {dev}")
            convs[dev] = {t: target_conv(nets[dev], t) for t in spec["targets"]}
    out: Dict[str, Any] = {"targets": {}}
    for t in spec["targets"]:
        ops = {dev: caps[dev]["operands"][t] for dev in devs}
        for dev in devs:
            if not _conv_shape_ok(caps[dev]["result"]["targets"][t]["conv"]):
                raise Void("plan", f"{t}: not a stride-1, padding-1, undilated, ungrouped convolution")
        W = ops[devs[0]]["W"]
        if not all(np.array_equal(ops[d]["W"], W) for d in devs):
            raise Void("state", f"{t}: the weight differs between devices")
        kshape = W.shape
        ref = {s: np_conv_weight_grad(ops[s]["X"], ops[s]["G"], kshape) for s in devs}
        l2_64 = 2.0 * spec["l2_weight"] * W.astype(np.float64)
        K._finite(f"reference {t}", [*ref.values()])

        with on(devs[0]):                                # self-check 1: the layout, against MLX's forward on the CPU
            y = convs[devs[0]][t](mx.array(ops[devs[0]]["X"]))
            mx.eval(y)
        layout = compare(np_conv_forward(ops[devs[0]]["X"], W, ops[devs[0]]["bias"]), np.array(y), spec)
        adjoint = {s: adjoint_check(ops[s]["X"], ops[s]["G"], ref[s], spec) for s in devs}
        if not layout["within_tol"] or not all(a["ok"] for a in adjoint.values()):
            raise Void("reference", f"{t}: the float64 reference failed its self-check (layout {layout}, "
                                    f"adjoint {adjoint})")

        ident: Dict[str, Any] = {}
        for s in devs:                                   # identical operands: each set, copied to each device
            locs: Dict[str, List[np.ndarray]] = {}
            for k in devs:
                locs[k] = []
                with on(k):
                    for _ in range(spec["repeats"]):
                        g = _local(convs[k][t], mx.array(ops[s]["X"]), mx.array(ops[s]["G"]))
                        mx.eval(g)
                        locs[k].append(np.array(g))
                K._finite(f"{t} local on {k} from {s}'s operands", locs[k])
            ident[s] = {f"{k}_vs_reference": [compare(x, ref[s], spec) for x in locs[k]] for k in devs}
            ident[s].update({f"{k}_repeats_bitwise": [bool(np.array_equal(x, locs[k][0])) for x in locs[k][1:]]
                             for k in devs})
            ident[s].update({f"{k}_equals_capture": bool(np.array_equal(locs[k][0], ops[s]["local"]))
                             for k in devs if k == s})
            ident[s]["gpu_vs_cpu"] = [compare(x, y, spec) for x, y in zip(locs[devs[-1]], locs[devs[0]])]
        cpu, gpu = devs[0], devs[-1]
        native = {f"{d}_original_vs_reference_own_ops": compare(ops[d]["orig"], ref[d] + l2_64, spec) for d in devs}
        native.update({"X_gpu_vs_cpu": compare(ops[gpu]["X"], ops[cpu]["X"], spec),
                       "G_gpu_vs_cpu": compare(ops[gpu]["G"], ops[cpu]["G"], spec),
                       "reference_gpu_ops_vs_reference_cpu_ops": compare(ref[gpu], ref[cpu], spec),
                       "original_gpu_vs_cpu": compare(ops[gpu]["orig"], ops[cpu]["orig"], spec),
                       "l2_gpu_vs_cpu": compare(ops[gpu]["l2"], ops[cpu]["l2"], spec),
                       "l2_vs_float64": {d: compare(ops[d]["l2"], l2_64, spec) for d in devs}})

        diff = np.abs(ops[gpu]["orig"].astype(np.float64) - ops[cpu]["orig"])
        bad = np.argwhere(diff > spec["cross_rel_tol"] * float(np.abs(ops[cpu]["orig"]).max()) + spec["cross_abs_tol"])
        order = np.argsort(-diff[tuple(bad.T)])[:spec["elements_recorded"]] if len(bad) else []
        abs_terms = np_conv_weight_grad(np.abs(ops[cpu]["X"]), np.abs(ops[cpu]["G"]), kshape) if len(bad) else None
        elements = [{"index": [int(x) for x in bad[i]],
                     **{f"original_{d}": float(ops[d]["orig"][tuple(bad[i])]) for d in devs},
                     **{f"reference_{d}_ops_with_l2": float((ref[d] + l2_64)[tuple(bad[i])]) for d in devs},
                     "abs_terms_cpu_ops": float(abs_terms[tuple(bad[i])])} for i in order]

        r = {"target": t, "kshape": list(kshape), "self_check": {"layout": layout, "adjoint": adjoint},
             "reference_scale": {d: float(np.abs(ref[d]).max()) for d in devs},
             "identical_operands": ident, "native": native, "failed_elements": {"count": int(len(bad)),
                                                                                "recorded": elements}}
        r["flags"] = flags_of(r)
        r["class"] = classify(r["flags"])
        sink.write(f"reference/{t}.json", r)
        log("reference", target=t, cls=r["class"], **r["flags"])
        out["targets"][t] = {"flags": r["flags"], "class": r["class"]}
    del nets, convs
    out["overall"] = overall({t: v["class"] for t, v in out["targets"].items()})
    return out


# ─────────────────────────────── entry 32, for the reproduction and the predictions ───────────────────────────────

def reproduced(blk: Mapping[str, Any], spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Step 1's rule: the GPU's failures are exactly the expected tensors in every evaluation; the CPU has none of its
    own (repeats, additivity)."""
    gpu, cpu = blk["devices"][spec["devices"][-1]], blk["devices"][spec["devices"][0]]
    expected = sorted(f"evaluation {i}: {t}" for i in range(spec["repeats"]) for t in spec["reproduce_failing"])
    cpu_own = [x for x in cpu["failures"] if x.startswith(("repeat", "additivity"))]
    return {"gpu_failures": gpu["failures"], "expected": expected, "cpu_own_failures": cpu_own,
            "ok": sorted(gpu["failures"]) == expected and not cpu_own}


def entry32_compare(seams: Seams, out_dir: pathlib.Path) -> Dict[str, Any]:
    """Prediction 1: step 1's lines and block against entry 32's S0_b0."""
    old = [json.loads(x) for x in seams.read_repo(f"{ENTRY32}/full.jsonl").decode().splitlines()]
    new = [json.loads(x) for x in (out_dir / "full.jsonl").read_text().splitlines()]
    keys = ("loss", "policy", "anchor", "grad_digest", "logits_digest", "values_digest")
    res: Dict[str, Any] = {}
    for dev in ("cpu", "gpu"):
        o = [x for x in old if x["block"] == "S0_b0" and x["device"] == dev]
        n = [x for x in new if x["block"] == "S0_b0" and x["device"] == dev]
        same = len(o) == len(n) > 0
        res[dev] = {"compared": min(len(o), len(n)),
                    "keys_equal": {k: same and all(a[k] == b[k] for a, b in zip(n, o)) for k in keys}}
        res[dev]["lines_equal"] = all(res[dev]["keys_equal"].values())
    old_blk = json.loads(seams.read_repo(f"{ENTRY32}/full/S0_b0.json"))
    res["block_equal"] = json.loads((out_dir / "full" / "S0_b0.json").read_text()) == old_blk
    return res


# ─────────────────────────────── the run ───────────────────────────────

class _Sink(E._Sink):
    """Entry 32's sink (lines fsynced, each result file create-only), with step 2's operand fixtures: create-only
    compressed .npz files outside the repository, each recorded in fixtures.jsonl as it is written."""

    def __init__(self, out_dir: pathlib.Path, fixture_dir: pathlib.Path, spec):
        self.out_dir, self.fixture_dir, self.spec = out_dir, fixture_dir, spec
        for d in ("full", "capture", "reference"):
            (out_dir / d).mkdir()
        self.files: List[str] = []
        self.fixtures: List[Dict[str, Any]] = []
        self.handles: Dict[str, Any] = {}

    def fixture(self, name, arrays: Mapping[str, np.ndarray], meta) -> Dict[str, Any]:
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
                 "bytes": path.stat().st_size,
                 "arrays": {k: {"shape": list(np.shape(v)), "dtype": str(np.asarray(v).dtype),
                                "digest": K._digest({k: np.asarray(v)})} for k, v in arrays.items()},
                 "tolerances": {k: self.spec[k] for k in ("cross_rel_tol", "cross_abs_tol", "recon_rel_tol",
                                                          "adjoint_rel_tol")}, **meta}
        self.fixtures.append(entry)
        self.line("fixtures.jsonl", entry)
        return {"name": name, "sha256": entry["sha256"]}


def _execute(spec: Mapping[str, Any], seams: Seams, out_dir: pathlib.Path, fixture_dir: pathlib.Path,
             rec: Dict[str, Any], log) -> None:
    import mlx.core as mx

    from . import search_distillation_d2 as D

    if pathlib.Path(D.__file__).resolve() != ROOT / CODE["d2"]:
        raise Void("code", f"{D.__file__} is not {CODE['d2']}")
    recipe = T.recipe_for(spec)
    if spec["devices"] != ["cpu", "gpu"] or spec["repeats"] < 2 or spec["state"] != "S0" or spec["batch"] != 0 \
            or not spec["targets"] or not all(_TARGET.match(t) for t in spec["targets"]):
        raise Void("plan", "the devices, repeats, state, batch or targets are not the card's shape")
    rec["recipe"], rec["mlx"] = dataclasses.asdict(recipe), getattr(mx, "__version__", None)

    for rel in [r for r in PINNED if r.startswith(ENTRY32)]:
        if M._sha(seams.read_repo(rel)) != PINNED[rel]:
            raise Void("entry32", f"{rel} is not the pinned bytes")
    entry32 = json.loads(seams.read_repo(f"{ENTRY32}/record.json"))
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
        with on(spec["devices"][-1]):                            # as entry 32 computed the anchor targets
            _, tv = D.forward(teacher, training)
        tv = np.asarray(tv, dtype=np.float32)
        K._finite("teacher values", [tv])
        rec["teacher_values"] = {"digest": K._digest({"teacher_values": tv}),
                                 "entry32_digest": entry32["teacher_values"]["digest"]}
        rec["teacher_values"]["equals_entry32"] = rec["teacher_values"]["digest"] == entry32["teacher_values"]["digest"]
        log("teacher_values", **rec["teacher_values"])
        batch = tuple(np.array(a) for a in D._batch([training[i] for i in draw]))
        s0 = D.snapshot(teacher)
        dg = K._digest(s0)
        del teacher

        blk = E.full_block(spec["state"], spec["batch"], s0, dg, seams.load_network, batch, tv[draw], recipe, spec,
                           sink)
        rec["step1"] = reproduced(blk, spec)
        rec["entry32"] = entry32_compare(seams, out_dir)
        log("step1", ok=rec["step1"]["ok"], gpu_failures=len(rec["step1"]["gpu_failures"]))
        if not rec["step1"]["ok"]:
            rec["outcome"] = "NOT_REPRODUCED"
            return

        lines = [json.loads(x) for x in (out_dir / "full.jsonl").read_text().splitlines()]
        step1 = {dev: next(x for x in lines if x["device"] == dev and x["repeat"] == 0) for dev in spec["devices"]}
        caps = {dev: capture(dev, s0, dg, seams.load_network, batch, tv[draw], recipe, spec, step1[dev], sink, log)
                for dev in spec["devices"]}
        rec["step2"] = {dev: {"faithful": c["result"]["faithful"]} for dev, c in caps.items()}
        if not all(c["result"]["faithful"] for c in caps.values()):
            rec["outcome"] = "CAPTURE_NOT_FAITHFUL"
            return

        rec["step3"] = reference_step(spec, caps, seams.load_network, s0, dg, sink, log)
        rec["outcome"] = "CLASSIFIED"
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
                           "strength_verdict": None,
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


def _real_seams(checkpoint: pathlib.Path) -> Seams:
    base = M._real_seams(checkpoint)
    return Seams(read_corpus_json=lambda: (ROOT / R.CORPUS_JSON).read_bytes(), checkpoint_sha1=base.checkpoint_sha1,
                 load_network=base.load_network, loaded_modules=base.loaded_modules,
                 read_repo=lambda rel: (ROOT / rel).read_bytes())


def run(checkpoint: str) -> int:
    """THE PUBLIC ENTRY. No seam: the setting, the frozen spec, the pinned card, code and evidence, the real
    sources."""
    if os.environ.get(SETTING) != SETTING_VALUE:
        raise Refused(f"{SETTING} is {os.environ.get(SETTING)!r}, not {SETTING_VALUE!r}: launch a fresh process with it")
    if _mlx_loaded():
        raise Refused("MLX was imported before the setting was checked: launch a fresh process")
    if GATE.SEARCH_DISTILLATION_D2_CONVREF_AUTHORIZED is not True:
        raise Refused("SEARCH_DISTILLATION_D2_CONVREF_AUTHORIZED is closed: the diagnostic is not authorized")
    if M._sha((ROOT / CARD).read_bytes()) != CARD_SHA256:
        raise Refused(f"{CARD} is not the pinned card")
    for rel, sha in PINNED.items():
        if M._sha((ROOT / rel).read_bytes()) != sha:
            raise Refused(f"{rel} is not the pinned bytes")
    for mod, rel in ((E, CODE["tf32path"]), (K, CODE["gradcheck"]), (R, CODE["gradtrace"])):
        if pathlib.Path(mod.__file__).resolve() != ROOT / rel:
            raise Refused(f"{mod.__file__} is not the pinned {rel}")
    ckpt = pathlib.Path(checkpoint)
    if not (ckpt.is_absolute() and ckpt.is_file()):
        raise Refused("--checkpoint must name an existing file by absolute path")
    out = ROOT / EVIDENCE_DIR
    if os.path.lexists(out):
        raise Refused(f"{out} exists: the run is one-shot")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fixture_dir = FIXTURE_PARENT / f"convref-fixtures-{stamp}"
    if os.path.lexists(fixture_dir):
        raise Refused(f"{fixture_dir} exists")
    code, rec = conclude(FROZEN, _real_seams(ckpt), out, fixture_dir, head=M._git("rev-parse", "HEAD"))
    print(json.dumps({"status": rec["status"], "void": rec.get("void"), "outcome": rec.get("outcome"),
                      "step1_ok": rec.get("step1", {}).get("ok"), "step2": rec.get("step2"),
                      "classes": {t: v["class"] for t, v in rec.get("step3", {}).get("targets", {}).items()},
                      "overall": rec.get("step3", {}).get("overall"), "fixture_dir": str(fixture_dir),
                      "head": rec["head"]}))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3 or argv[0] != "run" or argv[1] != "--checkpoint":
        print("usage: MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.search_distillation_d2_convref run "
              "--checkpoint <abs path>", file=sys.stderr)
        return EXIT_USAGE
    try:
        return run(argv[2])
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
