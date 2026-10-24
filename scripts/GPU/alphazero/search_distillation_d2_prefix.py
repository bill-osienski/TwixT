"""SEARCH DISTILLATION D2 -- FULL-FLOAT32 SETTING (MLX_ENABLE_TF32=0): THE 21-UPDATE DIAGNOSTIC PREFIX, judged under a
PROSPECTIVE criterion, in ONE fresh process launched with the setting. BUILT; run only through its wrapper.

Card: `docs/superpowers/2026-10-23-search-distillation-d2-prefix-qualification-card.md`, pinned below by sha256, as
are the reused code and entry 32's evidence. Nothing runs on import, and importing this module imports no MLX: `run`
refuses unless MLX_ENABLE_TF32 is exactly "0" and no `mlx` module is loaded when it starts, then unless the gate
`search_distillation_d2_prefix_authorization.SEARCH_DISTILLATION_D2_PREFIX_AUTHORIZED` is open (read at call time),
every pin matches and no output exists.
  STEP 1, reproduce at the incumbent: the teacher values and, on each device, one uninstrumented evaluation on Stage
    3's update-0 and update-21 batches must equal entry 32's, else NOT_REPRODUCED and stop (no optimizer update).
  STEP 2, the prefix twice (P1, P2): each from a fresh incumbent and a fresh Adam, Stage 3's updates 0-20 on the GPU
    with identical batches and teacher targets; after every update the scalars, the model digest and the Adam-state
    digest, compared update by update.
  STEP 3, at U21 (P1's state, in memory only) on both batches: full_block's native evaluations; the pre-tanh capture;
    the instrumented trace; float64 checks of every local step, all 52 encoder parameter gradients and the policy
    head's 8 (per device from its own encoder output, and entry 26's common-input check); the flips; the accounting
    for any failing native gradient.
Judged in card §6's order: QUALIFIED_AT_U21, or NOT_QUALIFIED with the first failing criterion, and the run stops
there. An input, pin, identity, state, batch, plan or optimizer-default mismatch, the adjoint self-check failing, an
exception or SIGTERM (the wrapper's 10-minute ceiling) are a VOID; a non-finite result is NOT_QUALIFIED. No retry and
no follow-up. Entry 32's verdict (FAIL at phase 2) is unchanged by any outcome.

    MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.search_distillation_d2_prefix run --checkpoint <calib020_0001>
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib
import platform
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from . import search_distillation_d2_branchtrace as B
from . import search_distillation_d2_convref as C
from . import search_distillation_d2_gradcheck as K
from . import search_distillation_d2_gradtrace as R
from . import search_distillation_d2_measure as M
from . import search_distillation_d2_prefix_authorization as GATE
from . import search_distillation_d2_tf32path as E
from . import search_distillation_d2_train as T

ROOT = M.ROOT
RECORD_KIND = "search-distillation-d2-prefix-record/1"
CARD = "docs/superpowers/2026-10-23-search-distillation-d2-prefix-qualification-card.md"
CARD_SHA256 = "cdd900c20a6db9aeb2d88fd3a32a77da2f9330271711c0fbfea0c2e72fb73f17"
EVIDENCE_DIR = "docs/superpowers/evidence/search-distillation-d2-prefix"
ENTRY32 = C.ENTRY32
CODE = {"branchtrace": "scripts/GPU/alphazero/search_distillation_d2_branchtrace.py", **B.CODE}
PINNED = {CODE["branchtrace"]: "bddd0e5f28a3e10ee1ee2920ce3965f0c280969151fb2c56d5a9a6883210ccb5",
          **{CODE[k]: B.PINNED[B.CODE[k]] for k in B.CODE},
          f"{ENTRY32}/record.json": C.PINNED[f"{ENTRY32}/record.json"],
          f"{ENTRY32}/full.jsonl": C.PINNED[f"{ENTRY32}/full.jsonl"]}
SETTING, SETTING_VALUE = E.SETTING, E.SETTING_VALUE
FROZEN = {**{k: E.FROZEN[k] for k in ("corpus_sha256", "teacher_digest", "reference_sha256", "reference_student_digest",
                                      "training_ids_sha256", "training_games", "held_out_games", "training_records",
                                      "passes", "steps", "batch_size", "learning_rate", "anchor_weight", "l2_weight",
                                      "clip_norm", "seed", "devices", "repeats", "cross_rel_tol", "cross_abs_tol",
                                      "fd_eps", "fd_rel_tol", "fd_mlx_rel_tol", "fd_seed")},
          "batches": [0, 21], "prefix_updates": 21, "prefixes": 2, "adjoint_rel_tol": 1e-9,
          "focus": B.FROZEN["focus"], "flips_recorded": 64, "elements_recorded": 64}
OUTCOMES = ("QUALIFIED_AT_U21", "NOT_QUALIFIED", "NOT_REPRODUCED")
REASONS = ("non_finite", "not_repeatable", "not_additive", "forward_disagreement", "trace_not_faithful",
           "local_derivative_error", "local_forward_error", "unexplained_gradient")
EXIT_COMPLETED, EXIT_VOID, EXIT_REFUSED, EXIT_USAGE = 0, 3, 4, 2
Void, Terminated = M.Void, M.Terminated
on, compare, Seams = K.on, E.compare, C.Seams
_bwd, _fwd = B._bwd, B._fwd


class Refused(Exception):
    """The run never started: the setting, MLX already loaded, gate closed, a pin, a bad argument or an output
    exists."""


class NotQualified(Exception):
    """A criterion of card §6 failed: the reason, and every failure behind it."""

    def __init__(self, reason: str, failures: List[str]):
        super().__init__(reason)
        self.reason, self.failures = reason, failures


def _mlx_loaded() -> bool:
    return any(m == "mlx" or m.startswith("mlx.") for m in sys.modules)


# ─────────────────────────────── step 2: the prefix ───────────────────────────────

def _adam(recipe):
    import mlx.optimizers as optim
    return optim.Adam(learning_rate=recipe.learning_rate)


def prefix(student, records, recipe, teacher_values, n_updates: int, spec, sink, reference, label: str,
           counter: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """gradcheck.reconstruct's loop (search_distillation_d2.train_student's updates 0..n_updates-1, nothing inside it),
    unchanged, with the model digest (every parameter and BatchNorm statistic) and the Adam-state digest (step,
    learning rate, every moment) recorded after each update. Returns the final state and the per-update rows."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from . import search_distillation_d2 as D
    from .trainer import MainModule, clip_grad_norm, compute_masked_log_probs

    M._require_eval(student, f"prefix {label}'s student")
    if D.state_digest(student) != spec["teacher_digest"]:
        raise Void("networks", f"prefix {label} does not start from the pinned incumbent")
    teacher_values = np.asarray(teacher_values, dtype=np.float32)
    surface = MainModule(student.encoder, student.policy_head)
    opt = _adam(recipe)

    def loss_fn(_surface, boards, rows, cols, mask, target, tv):
        logits, values, _ = student.forward_padded(boards, rows, cols, mask, active_size=D.BOARD)
        policy = mx.mean(-mx.sum(target * compute_masked_log_probs(logits, mask), axis=1))
        anchor = mx.mean((values - tv) ** 2)
        l2 = sum(mx.sum(p ** 2) for _, p in tree_flatten(_surface.trainable_parameters()))
        return policy + recipe.anchor_weight * anchor + recipe.l2_weight * l2, policy, anchor

    grad_fn = nn.value_and_grad(surface, loss_fn)
    rng = np.random.default_rng(recipe.seed)
    order: List[int] = []
    rows_out: List[Dict[str, Any]] = []
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
        counter["updates"] += 1
        values = {"loss": float(loss), "policy": float(policy), "anchor": float(anchor), "grad_norm": float(norm)}
        row = {"prefix": label, "update": step, "batch": idx, **values,
               "model_digest": K._digest(D.snapshot(student)),
               "adam_digest": K._digest({k: np.array(v) for k, v in tree_flatten(opt.state)}),
               "stage3": {f: reference[step][f] for f in R.TRACE_FIELDS}}
        sink.line("prefix.jsonl", row)
        rows_out.append(row)
        if not all(math.isfinite(v) for v in values.values()):
            raise Void("non_finite", f"prefix {label} update {step}: {values}")
    return D.snapshot(student), rows_out


PREFIX_FIELDS = ("batch", *R.TRACE_FIELDS, "model_digest", "adam_digest")


def prefixes_equal(p1: Sequence[Mapping[str, Any]], p2: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """P1 against P2, update by update: the batch, the scalars, the model digest and the Adam digest."""
    diffs = [{"update": a["update"], "fields": [f for f in PREFIX_FIELDS if a[f] != b[f]]}
             for a, b in zip(p1, p2) if any(a[f] != b[f] for f in PREFIX_FIELDS)]
    return {"updates_compared": min(len(p1), len(p2)), "same_length": len(p1) == len(p2), "n_differing": len(diffs),
            "first_difference": diffs[0] if diffs else None, "differing": diffs[:64],
            "equal": len(p1) == len(p2) and not diffs}


# ─────────────────────────────── step 3: the evaluations ───────────────────────────────

def pretanh_eval(dev, state_np, state_digest, make_net, batch_np, tv_np, recipe) -> Dict[str, Any]:
    """branchtrace.evaluate's uninstrumented evaluation, with the value head's pre-tanh output read through the
    network's own `return_value_pretanh=True` and returned beside the loss (supplementary; it must leave the loss terms,
    gradients, logits and values bitwise unchanged)."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from . import search_distillation_d2 as D
    from .trainer import MainModule, compute_masked_log_probs

    names_in = ("boards", "rows", "cols", "mask", "target")
    in_digest = K._digest({**dict(zip(names_in, batch_np)), "teacher_values": tv_np})
    with on(dev):
        net = C._set_state(make_net, state_np, state_digest, f"pre-tanh capture on {dev}")
        ex = K._ex(batch_np, tv_np)
        if K._digest({**{n: np.array(ex[n]) for n in names_in}, "teacher_values": np.array(ex["tv"])}) != in_digest:
            raise Void("inputs", f"{dev}: the batch arrays are not the batch")
        surface = MainModule(net.encoder, net.policy_head)

        def loss_fn(_s):
            logits, values, pre = net.forward_padded(ex["boards"], ex["rows"], ex["cols"], ex["mask"],
                                                     active_size=D.BOARD, return_value_pretanh=True)
            ce = -mx.sum(ex["target"] * compute_masked_log_probs(logits, ex["mask"]), axis=1)
            sq = (values - ex["tv"]) ** 2
            policy, anchor = mx.mean(ce), mx.mean(sq)
            total = policy + recipe.anchor_weight * anchor + recipe.l2_weight * sum(
                mx.sum(p ** 2) for _, p in tree_flatten(_s.trainable_parameters()))
            return total, policy, anchor, logits, values, pre

        (loss, policy, anchor, logits, values, pre), g = nn.value_and_grad(surface, loss_fn)(surface)
        mx.eval(loss, policy, anchor, logits, values, pre, g)
        out = {"device": dev, "terms": [float(loss), float(policy), float(anchor)], "logits": np.array(logits),
               "values": np.array(values), "pretanh": np.array(pre), "grads": {k: np.array(v) for k, v in
                                                                             tree_flatten(g)}}
        K._finite(f"pre-tanh capture on {dev}", [out["logits"], out["values"], out["pretanh"], *out["grads"].values()])
        del net, surface
    out["grad_digest"] = K._digest(out["grads"])
    out["logits_digest"] = K._digest({"logits": out["logits"]})
    out["values_digest"] = K._digest({"values": out["values"]})
    return out


def matches_first(e: Mapping[str, Any], line: Mapping[str, Any]) -> Dict[str, Any]:
    """An evaluation against full_block's evaluation 1 (its full.jsonl line): loss terms and digests, bitwise."""
    r = {"terms_equal": e["terms"] == [line["loss"], line["policy"], line["anchor"]],
         **{f"{k}_equal": e[f"{k}_digest"] == line[f"{k}_digest"] for k in ("grad", "logits", "values")}}
    r["ok"] = all(r.values())
    return r


# ─────────────────────────────── step 3: the float64 parameter references ───────────────────────────────

def param_table(cons: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Every encoder parameter tensor: its kind (conv | bn), the traced point whose gradient feeds it, its input
    (conv: the convolution's input; bn: the BatchNorm's input), its constants, and the ReLUs after it in the forward
    pass (a flip there reaches it)."""
    n = len(cons["blocks"])
    every = [r[0] for r in B.relus(n)]

    def later(b):
        return [f"blocks.{j}.relu_{k}" for j in range(b + 1, n) for k in ("inner", "outer")]

    out: Dict[str, Dict[str, Any]] = {}

    def add(name, kind, block, point, inp, const, path):
        for which in ("weight", "bias"):
            out[f"{name}.{which}"] = {"kind": kind, "which": which, "block": block, "point": point, "input": inp,
                                      "const": const, "path": path}

    st = cons["stem"]
    add("encoder.conv1", "conv", -1, "z0", "input", st["conv"], every)
    add("encoder.bn1", "bn", -1, "a0", "z0", st["bn"], every)
    for b, c in enumerate(cons["blocks"]):
        n_ = f"encoder.blocks.{b}"
        inner = [f"blocks.{b}.relu_inner", f"blocks.{b}.relu_outer", *later(b)]
        outer = [f"blocks.{b}.relu_outer", *later(b)]
        add(f"{n_}.conv1", "conv", b, "z1", "x_in", c["conv1"], inner)
        add(f"{n_}.bn1", "bn", b, "a1", "z1", c["bn1"], inner)
        add(f"{n_}.conv2", "conv", b, "z2", "h", c["conv2"], outer)
        add(f"{n_}.bn2", "bn", b, "s", "z2", c["bn2"], outer)
    return out


def _input(t, e) -> np.ndarray:
    b = e["block"]
    if e["input"] == "h":                                  # conv2's input: the inner ReLU's output
        return np.maximum(_fwd(t, b, "a1"), 0)
    return _fwd(t, b, e["input"])


def param_grad(e: Mapping[str, Any], X, G, l2_weight: float) -> np.ndarray:
    """One encoder parameter's gradient in float64 from its input X and the gradient G at its point, plus L2."""
    G = np.asarray(G, dtype=np.float64)
    c = e["const"]
    if e["kind"] == "conv":
        theta = c["W"] if e["which"] == "weight" else c["b"]
        g = C.np_conv_weight_grad(X, G, c["W"].shape) if e["which"] == "weight" else G.sum(axis=(0, 1, 2))
    else:
        theta = c[e["which"]]
        if e["which"] == "weight":
            xhat = (np.asarray(X, np.float64) - np.asarray(c["mean"], np.float64)) / np.sqrt(
                np.asarray(c["var"], np.float64) + c["eps"])
            g = (G * xhat).sum(axis=(0, 1, 2))
        else:
            g = G.sum(axis=(0, 1, 2))
    return g + 2.0 * l2_weight * np.asarray(theta, np.float64)


def parameter_checks(t, cons, grads, spec) -> Dict[str, Any]:
    """All encoder parameter gradients from the device's own operands (card §5), against its native gradients."""
    table = param_table(cons)
    names = sorted(k for k in grads if k.startswith("encoder."))
    if sorted(table) != names:
        raise Void("plan", "the encoder's parameter tensors are not the ones the references cover")
    res = {k: compare(grads[k], param_grad(e, _input(t, e), _bwd(t, e["block"], e["point"]), spec["l2_weight"]),
                      spec) for k, e in table.items()}
    return {"tensors": res, "failures": [k for k, v in res.items() if not v["within_tol"]]}


def policy_head_native(t, n_blocks, ph_params, rows_c, cols_c, mask, target, grads, terms, spec) -> Dict[str, Any]:
    """The policy head's 8 native gradients of the full loss on one device, against np_policy_head on THAT device's
    encoder output (relu of the last block's s), every record weighted 1, plus L2."""
    F = np.maximum(_fwd(t, n_blocks - 1, "s"), 0)
    L, g, _ = K.np_policy_head(ph_params, F, rows_c, cols_c, mask, target, np.ones(F.shape[0]))
    res = {k: compare(grads[f"policy_head.{k}"], g[k] + 2.0 * spec["l2_weight"] * ph_params[k], spec)
           for k in K.PH_KEYS}
    return {"tensors": res, "failures": [f"policy_head.{k}" for k, v in res.items() if not v["within_tol"]],
            "numpy_policy_term": L, "native_policy_term": terms[1],
            "policy_term_vs_numpy": compare(np.array(terms[1]), np.array(L), spec)}


# ─────────────────────────────── step 3: the accounting ───────────────────────────────

def deltas(tr, cons, spec) -> Dict[int, Dict[str, np.ndarray]]:
    """Entry 34's linear float64 propagation of the flips alone, Delta = 0 at the encoder's output, through every block
    and the stem, with the GPU's masks: delta_pre = delta_post * m_gpu + g_post_cpu * (m_gpu - m_cpu). The gradient's
    predicted change at each parameter's point."""
    cpu, gpu = spec["devices"][0], spec["devices"][-1]
    n = len(cons["blocks"])
    out: Dict[int, Dict[str, np.ndarray]] = {}
    delta = np.zeros(_bwd(tr[cpu], n - 1, "out").shape, dtype=np.float64)

    def through(d_post, b, pre, post):
        m_c, m_g = _fwd(tr[cpu], b, pre) > 0, _fwd(tr[gpu], b, pre) > 0
        return d_post * m_g + _bwd(tr[cpu], b, post).astype(np.float64) * (m_g.astype(np.float64) - m_c)

    for b in reversed(range(n)):
        c = cons["blocks"][b]
        ds = through(delta, b, "s", "out")
        dz2 = ds * B.bn_scale(c["bn2"])
        da1 = through(B.np_conv_input_grad(dz2, c["conv2"]["W"]), b, "a1", "h")
        dz1 = da1 * B.bn_scale(c["bn1"])
        out[b] = {"s": ds, "z2": dz2, "a1": da1, "z1": dz1}
        delta = B.np_conv_input_grad(dz1, c["conv1"]["W"]) + ds
    da0 = through(delta, -1, "a0", "x0")
    out[-1] = {"a0": da0, "z0": da0 * B.bn_scale(cons["stem"]["bn"])}
    return out


def explained(is_encoder: bool, encoder_output_ok: bool, flip_on_path: bool, with_ok: bool, without_ok: bool) -> bool:
    """Card §6, criterion 7: every condition, and no other way."""
    return bool(is_encoder and encoder_output_ok and flip_on_path and with_ok and not without_ok)


def accounting(tr, cons, spec, native_gpu, failing: Sequence[str], flips: Mapping[str, Any],
               encoder_output: Mapping[str, Any]) -> Dict[str, Any]:
    """For each failing tensor: its float64 formula from the GPU's operands with the CPU's upstream gradient plus the
    flips' Delta (and without it), plus L2, against the GPU's native gradient."""
    cpu, gpu = spec["devices"][0], spec["devices"][-1]
    table = param_table(cons)
    dl = deltas(tr, cons, spec)
    res: Dict[str, Any] = {"encoder_output_gradient": encoder_output, "tensors": {}}
    for k in failing:
        e = table.get(k)
        if e is None:
            res["tensors"][k] = {"explained": False, "why": "no accounting is defined for this tensor"}
            continue
        X, Gc = _input(tr[gpu], e), _bwd(tr[cpu], e["block"], e["point"]).astype(np.float64)
        with_ = compare(param_grad(e, X, Gc + dl[e["block"]][e["point"]], spec["l2_weight"]), native_gpu[k], spec)
        without = compare(param_grad(e, X, Gc, spec["l2_weight"]), native_gpu[k], spec)
        on_path = [r for r in e["path"] if flips[r]["count"]]
        r = {"with_flips": with_, "without_flips": without, "flips_on_path": {x: flips[x]["count"] for x in on_path}}
        r["explained"] = explained(True, encoder_output["within_tol"], bool(on_path), with_["within_tol"],
                                   without["within_tol"])
        res["tensors"][k] = r
    res["unexplained"] = [k for k, v in res["tensors"].items() if not v["explained"]]
    return res


# ─────────────────────────────── the run ───────────────────────────────

class _Sink(E._Sink):
    """Entry 32's sink (lines fsynced, each result file create-only, no array ever saved), with this run's directories;
    the lines are also kept in memory."""

    def __init__(self, out_dir: pathlib.Path):
        super().__init__(out_dir)
        for d in ("checks", "stages", "accounting"):
            (out_dir / d).mkdir()
        self.kept: Dict[str, List[Dict[str, Any]]] = {}

    def line(self, rel, obj):
        super().line(rel, obj)
        self.kept.setdefault(rel, []).append(obj)


def _native_phase(spec, seams, recipe, u21, dg, batches, tv, draws, sink, rec, log) -> Dict[int, Dict[str, Any]]:
    """full_block at U21 on both batches; criteria 2-4 judged over both (card §6)."""
    cpu, gpu = spec["devices"][0], spec["devices"][-1]
    blocks = {}
    for b in spec["batches"]:
        blocks[b] = E.full_block("U21", b, u21, dg, seams.load_network, batches[b], tv[draws[b]], recipe, spec, sink)
        log("native", block=f"U21_b{b}", failures=len(blocks[b]["devices"][gpu]["failures"]))
    rep, add, fwd, native = [], [], [], {}
    for b, blk in blocks.items():
        for dev in spec["devices"]:
            d = blk["devices"][dev]
            for i, rp in enumerate(d["repeats_bitwise"], start=1):
                rep += [f"U21_b{b} {dev} evaluation {i + 1}: {k}" for k in ("loss_terms", "logits", "values")
                        if not rp[k]]
                rep += [f"U21_b{b} {dev} evaluation {i + 1}: {k}" for k, ok in rp["grads"].items() if not ok]
            add += [f"U21_b{b} {dev}: {k}" for k, v in d["additivity"].items() if not v["within_tol"]]
        g = blk["devices"][gpu]
        for i, terms in enumerate(g["loss_terms_vs_reference"]):
            fwd += [f"U21_b{b} evaluation {i + 1}: {t}" for t, v in terms.items() if not v["within_tol"]]
        fwd += [f"U21_b{b} evaluation {i + 1}: legal logits" for i, v in enumerate(g["logits_legal_vs_reference"])
                if not v["within_tol"]]
        fwd += [f"U21_b{b} evaluation {i + 1}: padded logits" for i, ok in enumerate(g["logits_padding_equal"])
                if not ok]
        fwd += [f"U21_b{b} evaluation {i + 1}: values" for i, v in enumerate(g["values_vs_reference"])
                if not v["within_tol"]]
        native[b] = sorted({k for ev in g["grads_vs_reference"] for k, v in ev.items() if not v["within_tol"]})
    rec["native"] = {str(b): {"gpu_gradient_failures": native[b]} for b in blocks}
    for reason, fails in (("not_repeatable", rep), ("not_additive", add), ("forward_disagreement", fwd)):
        if fails:
            raise NotQualified(reason, fails)
    return {b: {"block": blocks[b], "failing": native[b]} for b in blocks}


def _analyse(b, spec, seams, recipe, u21, dg, batch, tvb, failing, sink, rec, log) -> None:
    """Criteria 5-7 on one batch (card §6)."""
    import mlx.core as mx

    from . import search_distillation_d2 as D
    from .network import canonicalize_batch

    cpu, gpu = spec["devices"][0], spec["devices"][-1]
    first = {ln["device"]: ln for ln in sink.kept["full.jsonl"]
             if ln["block"] == f"U21_b{b}" and ln["repeat"] == 0}
    traced, pre, fid = {}, {}, {}
    for dev in spec["devices"]:
        traced[dev] = B.evaluate(dev, u21, dg, seams.load_network, batch, tvb, recipe, spec, trace=True)
        p = pretanh_eval(dev, u21, dg, seams.load_network, batch, tvb, recipe)
        pre[dev] = p["pretanh"]
        fid[dev] = {"trace": matches_first(traced[dev], first[dev]), "pretanh_capture": matches_first(p, first[dev])}
    pre_cmp = compare(pre[gpu], pre[cpu], spec)
    rec["batches"][str(b)] = {"fidelity": {d: {k: v["ok"] for k, v in f.items()} for d, f in fid.items()},
                              "pretanh_gpu_vs_cpu": pre_cmp}
    log("fidelity", batch=b, **rec["batches"][str(b)]["fidelity"])
    unfaithful = [f"U21_b{b} {d}: {k}" for d, f in fid.items() for k, v in f.items() if not v["ok"]]
    if unfaithful:
        for dev in spec["devices"]:
            sink.write(f"checks/b{b}_{dev}.json", {"fidelity": fid[dev]})
        raise NotQualified("trace_not_faithful", unfaithful)

    cons = {d: traced[d]["constants"] for d in spec["devices"]}
    if not B._same(cons[cpu], cons[gpu]):
        raise Void("state", "the encoder constants differ between devices")
    c0 = cons[cpu]
    n = len(c0["blocks"])
    for name, cc in [("encoder.conv1", c0["stem"]["conv"])] + [
            (f"encoder.blocks.{i}.{k}", c0["blocks"][i][k]) for i in range(n) for k in ("conv1", "conv2")]:
        if not C._conv_shape_ok(cc):
            raise Void("plan", f"{name} is not a stride-1, padding-1, undilated, ungrouped convolution")
    tr = {d: traced[d]["traced"] for d in spec["devices"]}
    tb = min(n - 1, 2)
    adj = B.adjoint_input_check(_fwd(tr[cpu], tb, "x_in"), _bwd(tr[cpu], tb, "z1"), c0["blocks"][tb]["conv1"]["W"],
                                spec)
    rec["batches"][str(b)]["self_check"] = adj
    if not adj["ok"]:
        raise Void("reference", f"the float64 transposed convolution failed its adjoint self-check ({adj})")

    with on(cpu):
        bc, rc, cc_, _ = canonicalize_batch(*(mx.array(a) for a in batch[:4]), D.BOARD)
        mx.eval(bc, rc, cc_)
        bc, rc, cc_ = np.array(bc), np.array(rc), np.array(cc_)
    if not all(np.array_equal(_fwd(tr[d], -1, "input"), bc) for d in spec["devices"]):
        raise Void("inputs", "a device's encoder input is not the canonicalized batch")
    ph_net = C._set_state(seams.load_network, u21, dg, "the policy-head reference's network")
    ph_params = K.policy_head_params(ph_net.policy_head)

    derivative, forward = [], []
    for dev in spec["devices"]:
        lc = B.local_checks(tr[dev], c0, spec)
        pc = parameter_checks(tr[dev], c0, traced[dev]["grads"], spec)
        ph = policy_head_native(tr[dev], n, ph_params, rc, cc_, batch[3], batch[4], traced[dev]["grads"],
                                traced[dev]["terms"], spec)
        sink.write(f"checks/b{b}_{dev}.json", {"fidelity": fid[dev], "local_checks": lc, "parameters": pc,
                                               "policy_head_native": ph})
        derivative += [f"U21_b{b} {dev}: {x}" for x in lc["backward_failures"] + pc["failures"] + ph["failures"]]
        forward += [f"U21_b{b} {dev}: {x}" for x in lc["forward_failures"]]
        rec["batches"][str(b)][dev] = {"backward_failures": lc["backward_failures"],
                                       "forward_failures": lc["forward_failures"],
                                       "parameter_failures": pc["failures"], "policy_head_failures": ph["failures"],
                                       "parameters_checked": len(pc["tensors"])}
    F_cpu = np.maximum(_fwd(tr[cpu], n - 1, "s"), 0)
    common = K.reference_case(f"U21_b{b}", ph_net.policy_head, F_cpu, rc, cc_, batch[3], batch[4],
                              np.ones(F_cpu.shape[0], np.float32), spec, sink)
    common_fail = [f"U21_b{b} common input {dev}: {x}" for dev in spec["devices"]
                   for x in E.judge_reference(common, dev)]
    rec["batches"][str(b)]["common_input_failures"] = common_fail
    rec["batches"][str(b)]["common_input_directional_skipped"] = bool(common["directional"].get("skipped"))
    del ph_net
    log("local_checks", batch=b, derivative=len(derivative) + len(common_fail), forward=len(forward))
    if derivative or common_fail:
        raise NotQualified("local_derivative_error", derivative + common_fail)
    if forward:
        raise NotQualified("local_forward_error", forward)

    flips = {}
    for name, blk, p_, q_ in B.relus(n):
        flips[name] = B.flips_at(name, blk, p_, q_, tr, c0, spec)
        sink.write(f"stages/b{b}_{name}.json", flips[name])
    enc_out = compare(_bwd(tr[gpu], n - 1, "out"), _bwd(tr[cpu], n - 1, "out"), spec)
    rec["batches"][str(b)].update(flips={k: v["count"] for k, v in flips.items()}, encoder_output_gradient=enc_out,
                                  native_failing=list(failing))
    log("flips", batch=b, flips=rec["batches"][str(b)]["flips"], encoder_output_within_tol=enc_out["within_tol"])
    if not failing:
        return
    acc = accounting(tr, c0, spec, traced[gpu]["grads"], failing, flips, enc_out)
    sink.write(f"accounting/b{b}.json", acc)
    rec["batches"][str(b)]["explained"] = [k for k, v in acc["tensors"].items() if v["explained"]]
    rec["batches"][str(b)]["unexplained"] = acc["unexplained"]
    log("accounting", batch=b, unexplained=acc["unexplained"])
    if acc["unexplained"]:
        raise NotQualified("unexplained_gradient", [f"U21_b{b}: {k}" for k in acc["unexplained"]])


def _qualify(spec, seams, recipe, training, tv, draws, batches, reference, sink, rec, log) -> None:
    gpu = spec["devices"][-1]
    runs = []
    counter = {"updates": 0}
    try:
        for i in range(spec["prefixes"]):
            student = seams.load_network()
            student.eval()
            label = f"P{i + 1}"
            with on(gpu):
                state, rows = prefix(student, training, recipe, tv, spec["prefix_updates"], spec, sink, reference,
                                     label, counter)
            del student
            runs.append({"label": label, "state": state, "rows": rows, "digest": K._digest(state)})
            log("prefix", label=label, digest=runs[-1]["digest"], updates=len(rows))
    finally:
        rec["optimizer_updates"] = counter["updates"]
    cmp = prefixes_equal(runs[0]["rows"], runs[1]["rows"])
    cmp["final_digests_equal"] = runs[0]["digest"] == runs[1]["digest"]
    rec["prefix"] = {"digests": {r["label"]: r["digest"] for r in runs}, "comparison": cmp, "u21": "P1",
                     "saved": False}
    if not (cmp["equal"] and cmp["final_digests_equal"]):
        fails = [f"update {d['update']}: {', '.join(d['fields'])}" for d in cmp["differing"]]
        raise NotQualified("not_repeatable", fails or ["the final U21 digests differ"])
    u21, dg = runs[0]["state"], runs[0]["digest"]
    del runs
    native = _native_phase(spec, seams, recipe, u21, dg, batches, tv, draws, sink, rec, log)
    rec["batches"] = {}
    for b in spec["batches"]:
        _analyse(b, spec, seams, recipe, u21, dg, batches[b], tv[draws[b]], native[b]["failing"], sink, rec, log)


def _execute(spec: Mapping[str, Any], seams: Seams, out_dir: pathlib.Path, rec: Dict[str, Any], log) -> None:
    import mlx.core as mx

    from . import search_distillation_d2 as D

    if pathlib.Path(D.__file__).resolve() != ROOT / CODE["d2"]:
        raise Void("code", f"{D.__file__} is not {CODE['d2']}")
    recipe = T.recipe_for(spec)
    adam = D.adam_settings(_adam(recipe))
    if {k: adam[k] for k in T.ADAM_DEFAULTS} != T.ADAM_DEFAULTS:
        raise Void("optimizer", f"MLX's Adam resolves to {adam}, not the card's {T.ADAM_DEFAULTS}")
    if spec["devices"] != ["cpu", "gpu"] or spec["repeats"] < 2 or spec["prefixes"] != 2 or \
            spec["prefix_updates"] < 1 or spec["batches"] != [0, spec["prefix_updates"]]:
        raise Void("plan", "the devices, repeats, prefixes or batches are not the card's shape")
    rec["recipe"], rec["optimizer"], rec["mlx"] = dataclasses.asdict(recipe), adam, getattr(mx, "__version__", None)

    for rel in [r for r in PINNED if r.startswith(ENTRY32)]:
        if M._sha(seams.read_repo(rel)) != PINNED[rel]:
            raise Void("evidence", f"{rel} is not the pinned bytes")
    entry32 = json.loads(seams.read_repo(f"{ENTRY32}/record.json"))
    lines32 = [json.loads(x) for x in seams.read_repo(f"{ENTRY32}/full.jsonl").decode().splitlines()]
    expect = {}
    for b in spec["batches"]:
        for dev in spec["devices"]:
            ln = [x for x in lines32 if x["block"] == f"S0_b{b}" and x["device"] == dev and x["repeat"] == 0]
            if len(ln) != 1:
                raise Void("evidence", f"entry 32 has no single first evaluation of S0_b{b} on {dev}")
            expect[(b, dev)] = {"terms": [ln[0]["loss"], ln[0]["policy"], ln[0]["anchor"]],
                                **{f"{k}_digest": ln[0][f"{k}_digest"] for k in ("grad", "logits", "values")}}
    training, _ids, rec["corpus"] = R.read_training(spec, seams.read_corpus_json())
    reference = R.read_reference(spec, seams.read_repo(R.REFERENCE), recipe)
    if len(reference) < spec["prefix_updates"]:
        raise Void("reference", "Stage 3's log is shorter than the prefix")
    sha1 = seams.checkpoint_sha1()
    if sha1 != M.CHECKPOINT_SHA1:
        raise Void("checkpoint", f"SHA-1 {sha1} is not calib020_0001's")
    rec["checkpoint"] = {"path": M.CHECKPOINT, "sha1": sha1}
    draws = K.draw_batches(len(training), recipe, max(spec["batches"]) + 1)
    if any(draws[b] != entry32["batches"].get(str(b)) for b in spec["batches"]):
        raise Void("batch", "a draw is not the one entry 32 recorded")
    rec["batch_draws"] = {str(b): draws[b] for b in spec["batches"]}
    log("inputs_verified", training_records=len(training))

    sink = _Sink(out_dir)
    try:
        teacher = seams.load_network()
        teacher.eval()
        if D.state_digest(teacher) != spec["teacher_digest"]:
            raise Void("networks", "the loaded incumbent is not the pinned state")
        rec["modules"] = seams.loaded_modules()
        with on(spec["devices"][-1]):                            # as entries 32-34 computed the anchor targets
            _, tv = D.forward(teacher, training)
        tv = np.asarray(tv, dtype=np.float32)
        if not np.all(np.isfinite(tv)):
            raise Void("teacher_values", "a non-finite teacher value")
        rec["teacher_values"] = {"digest": K._digest({"teacher_values": tv}),
                                 "entry32_digest": entry32["teacher_values"]["digest"]}
        rec["teacher_values"]["equals_entry32"] = rec["teacher_values"]["digest"] == entry32["teacher_values"]["digest"]
        log("teacher_values", **rec["teacher_values"])
        batches = {b: tuple(np.array(a) for a in D._batch([training[i] for i in draws[b]])) for b in spec["batches"]}
        s0 = D.snapshot(teacher)
        dg0 = K._digest(s0)
        del teacher

        rep: Dict[str, Any] = {"teacher_values_equal": rec["teacher_values"]["equals_entry32"], "blocks": {}}
        for b in spec["batches"]:
            for dev in spec["devices"]:
                try:
                    e = B.evaluate(dev, s0, dg0, seams.load_network, batches[b], tv[draws[b]], recipe, spec,
                                   trace=False)
                    got = {"terms": e["terms"], **{f"{k}_digest": e[f"{k}_digest"] for k in ("grad", "logits",
                                                                                            "values")}}
                except Void as v:                        # entry 32's were finite: a non-finite one did not repeat
                    if v.step != "non_finite":
                        raise
                    got = {"non_finite": v.detail}
                rep["blocks"].setdefault(f"S0_b{b}", {})[dev] = {"got": got, "expected": expect[(b, dev)],
                                                                 "ok": got == expect[(b, dev)]}
        rep["ok"] = rep["teacher_values_equal"] and all(v["ok"] for blk in rep["blocks"].values()
                                                        for v in blk.values())
        sink.write("reproduce.json", rep)
        rec["reproduce"] = {"ok": rep["ok"], "teacher_values_equal": rep["teacher_values_equal"],
                            **{n: {d: v["ok"] for d, v in blk.items()} for n, blk in rep["blocks"].items()}}
        log("reproduce", ok=rep["ok"])
        if not rep["ok"]:
            rec["outcome"] = "NOT_REPRODUCED"
            return
        del s0

        try:
            _qualify(spec, seams, recipe, training, tv, draws, batches, reference, sink, rec, log)
            rec["outcome"] = "QUALIFIED_AT_U21"
        except NotQualified as q:
            rec.update(outcome="NOT_QUALIFIED", reason=q.reason, failures=q.failures[:256],
                       n_failures=len(q.failures))
        except Void as v:
            if v.step != "non_finite":
                raise
            rec.update(outcome="NOT_QUALIFIED", reason="non_finite", failures=[v.detail], n_failures=1)
    finally:
        sink.close()
        rec["files"] = list(sink.files)


def conclude(spec: Mapping[str, Any], seams: Seams, out_dir: pathlib.Path,
             head: str | None = None) -> Tuple[int, Dict[str, Any]]:
    """THE RECORD BOUNDARY: record.json on every exit this survives; SIGTERM -> VOID at step "terminated"."""
    out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "card": CARD, "card_sha256": CARD_SHA256, "spec": dict(spec),
                           "head": head, "status": None, "outcome": None, "reason": None, "written_by": "script",
                           "diagnostic": True, "setting": {SETTING: os.environ.get(SETTING)},
                           "entry32_verdict": "FAIL at phase 2 (unchanged)", "optimizer_updates": 0,
                           "model_state_saved": False, "candidate_checkpoint": None, "training_qualified": False,
                           "held_out_records_read": 0, "causal_claims": None, "strength_verdict": None,
                           "elapsed_is_a_benchmark": False,
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
        _execute(spec, seams, out_dir, rec, log)
        if rec["outcome"] not in OUTCOMES or (rec["outcome"] == "NOT_QUALIFIED") != (rec["reason"] in REASONS):
            raise Void("plan", f"no outcome ({rec['outcome']!r}, {rec['reason']!r})")
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
                rec["outcome"], rec["reason"] = None, None
            rec["setting_at_end"] = {SETTING: os.environ.get(SETTING)}
            rec["total_seconds"] = round(time.monotonic() - t0, 3)
            rec["loadavg_after"] = os.getloadavg()
            log("end", status=rec["status"], outcome=rec["outcome"], reason=rec["reason"])
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
    if GATE.SEARCH_DISTILLATION_D2_PREFIX_AUTHORIZED is not True:
        raise Refused("SEARCH_DISTILLATION_D2_PREFIX_AUTHORIZED is closed: the prefix is not authorized")
    if M._sha((ROOT / CARD).read_bytes()) != CARD_SHA256:
        raise Refused(f"{CARD} is not the pinned card")
    for rel, sha in PINNED.items():
        if M._sha((ROOT / rel).read_bytes()) != sha:
            raise Refused(f"{rel} is not the pinned bytes")
    for mod, rel in ((B, CODE["branchtrace"]), (C, CODE["convref"]), (E, CODE["tf32path"]), (K, CODE["gradcheck"]),
                     (R, CODE["gradtrace"])):
        if pathlib.Path(mod.__file__).resolve() != ROOT / rel:
            raise Refused(f"{mod.__file__} is not the pinned {rel}")
    ckpt = pathlib.Path(checkpoint)
    if not (ckpt.is_absolute() and ckpt.is_file()):
        raise Refused("--checkpoint must name an existing file by absolute path")
    out = ROOT / EVIDENCE_DIR
    if os.path.lexists(out):
        raise Refused(f"{out} exists: the run is one-shot")
    code, rec = conclude(FROZEN, C._real_seams(ckpt), out, head=M._git("rev-parse", "HEAD"))
    print(json.dumps({"status": rec["status"], "void": rec.get("void"), "outcome": rec.get("outcome"),
                      "reason": rec.get("reason"), "reproduce_ok": rec.get("reproduce", {}).get("ok"),
                      "optimizer_updates": rec.get("optimizer_updates"),
                      "prefix_equal": rec.get("prefix", {}).get("comparison", {}).get("equal"),
                      "batches": {b: {k: v.get(k) for k in ("flips", "explained", "unexplained")}
                                  for b, v in rec.get("batches", {}).items()},
                      "head": rec["head"]}))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3 or argv[0] != "run" or argv[1] != "--checkpoint":
        print("usage: MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.search_distillation_d2_prefix run "
              "--checkpoint <abs path>", file=sys.stderr)
        return EXIT_USAGE
    try:
        return run(argv[2])
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
