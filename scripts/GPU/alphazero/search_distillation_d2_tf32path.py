"""SEARCH DISTILLATION D2 -- FULL-FLOAT32 SETTING (MLX_ENABLE_TF32=0) THROUGH THE POLICY HEAD AND THE FULL GRADIENT
PATH: three sequential checks in ONE fresh process launched with the setting. BUILT; run only through its wrapper.

Card: `docs/superpowers/2026-10-23-search-distillation-d2-tf32off-path-validation-card.md`, pinned below by sha256, as
are the reused code and entry 26's evidence. Nothing runs on import, and importing this module imports no MLX: the
setting is read by MLX once per process, so `run` refuses unless MLX_ENABLE_TF32 is exactly "0" and no `mlx` module is
loaded when it starts. It then refuses unless the gate
`search_distillation_d2_tf32path_authorization.SEARCH_DISTILLATION_D2_TF32PATH_AUTHORIZED` is open (read at call
time), every pin matches and no output exists.
  PHASE 1, policy head: entry 26's six saved reference cases (four synthetic, the real batch at S0 and at S21), each
    checked against entry 26's fixtures.jsonl; entry 26's case runner, unchanged: NumPy float64 analytic gradients
    against MLX on each device three times, single-record additivity and the directional central difference (required
    to run on the four synthetic cases; on the real batches it was skipped in entry 26 and a skip is recorded).
  PHASE 2, full model at the incumbent (only if phase 1 passes): Stage 3's update-0 and update-21 batches; on each
    device three times from a fresh network and fresh batch arrays (digests checked), the training loss's terms,
    logits (legal moves; the padding bitwise), values and all 60 gradient tensors; half + half additivity.
  PHASE 3, updated state (only if phases 1 and 2 pass): ONE prefix of Stage 3's updates 0-20 on the GPU (21
    in-memory diagnostic optimizer updates, nothing saved), then phase 2's checks at that state, U21.
  JUDGED on the GPU only, with entry 26's frozen tolerances; the CPU's own results are reported beside. The first
    failing phase is named and the later phases never run.
A difference is an observation. An input, pin, identity or plan mismatch, a reference failing its own check, a
non-finite result, an exception or SIGTERM (the wrapper's 10-minute ceiling) are a VOID. No retry. No array, model
state or checkpoint is saved; the training-only reader opens no held-out game file.

    MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.search_distillation_d2_tf32path run --checkpoint <calib020_0001>
"""
from __future__ import annotations

import dataclasses
import io
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

from . import search_distillation_d2_gradcheck as K
from . import search_distillation_d2_gradtrace as R
from . import search_distillation_d2_measure as M
from . import search_distillation_d2_tf32path_authorization as GATE
from . import search_distillation_d2_train as T

ROOT = M.ROOT
RECORD_KIND = "search-distillation-d2-tf32path-record/1"
CARD = "docs/superpowers/2026-10-23-search-distillation-d2-tf32off-path-validation-card.md"
CARD_SHA256 = "95bc073a293b145fb4e29d17b35df4c6b76ebea83fa8dfe7ffccdd164f186f37"
EVIDENCE_DIR = "docs/superpowers/evidence/search-distillation-d2-tf32path"
FIXTURE_DIR = pathlib.Path.home() / "twixt-cache" / "search-distillation-d2" / "gradcheck-fixtures-20261023T034401Z"
ENTRY26 = "docs/superpowers/evidence/search-distillation-d2-gradcheck"
CODE = {"gradcheck": "scripts/GPU/alphazero/search_distillation_d2_gradcheck.py",
        "gradtrace": "scripts/GPU/alphazero/search_distillation_d2_gradtrace.py",
        "d2": "scripts/GPU/alphazero/search_distillation_d2.py"}
PINNED = {CODE["gradcheck"]: "4ba735994e4e559bd1be655b0c6800f18a7ff7ac8ecad29b7a5119c79063d14e",
          CODE["gradtrace"]: "85b84b074b3359be938fe3e40cc2849ce2b1f7e6db88b909a784b625e3e35c97",
          CODE["d2"]: "bc8fd0c80fc533f50273ebda60c81375c5131d068a70344e07afef9a73617be9",
          f"{ENTRY26}/fixtures.jsonl": "5a5e16e84a4c4f709bce65f26746481504a4127dbe9909fe63311ac226aeb055",
          f"{ENTRY26}/full.jsonl": "efe5145dfaba14bc48c1f7a89b95ab1d94c10ca4658216cfa9fd2800984ce2f0",
          f"{ENTRY26}/record.json": "bb276d78b6ce474f9323e5e41b69836e0265f68924dfded3339c3e652869b953",
          f"{ENTRY26}/reference/repeated_indices.json":
              "8322e0cd31e48787ce07bcc7481b240ffd5b8883d8d00937eaeb89ee4dddcc92",
          f"{ENTRY26}/reference/batch_members.json": "5a8a60c692361f0b5f3ce6163912589a08d41b2ffb1754b285b52113e5f1709a",
          f"{ENTRY26}/reference/padding.json": "eb0512ce22253076bffbc3f89ce2eae60e081ba7d2b7cb2cdcb09653bac63ff5",
          f"{ENTRY26}/reference/weighted_additivity.json":
              "5cbd3b0bda05078f62dd790f12a290e45e81bf3c9b49638ef22f96c8a3fee435",
          f"{ENTRY26}/reference/real_S0.json": "faef0033544b0fa95200dfd6a975b1de898527a8f3c3280a5c6252a087ec150e",
          f"{ENTRY26}/reference/real_S21.json": "a8746c77cc51059a429de41b2c14336e224fd2efb25a204c55db4e0b87251e47"}
SETTING, SETTING_VALUE = "MLX_ENABLE_TF32", "0"
FROZEN = {**{k: K.FROZEN[k] for k in ("corpus_sha256", "teacher_digest", "reference_sha256",
                                      "reference_student_digest", "training_ids_sha256", "training_games",
                                      "held_out_games", "training_records", "passes", "steps", "batch_size",
                                      "learning_rate", "anchor_weight", "l2_weight", "clip_norm", "seed", "devices",
                                      "repeats", "cross_rel_tol", "cross_abs_tol", "fd_eps", "fd_rel_tol",
                                      "fd_mlx_rel_tol", "fd_seed")},
          "policy_head_cases": ["repeated_indices", "batch_members", "padding", "weighted_additivity", "real_S0",
                                "real_S21"],
          "directional_required": ["repeated_indices", "batch_members", "padding", "weighted_additivity"],
          "batches": [0, 21], "prefix_updates": 21}
PHASES = ("policy_head", "full_model_incumbent", "full_model_updated")
EXIT_COMPLETED, EXIT_VOID, EXIT_REFUSED, EXIT_USAGE = 0, 3, 4, 2
Void, Terminated = M.Void, M.Terminated
on = K.on


class Refused(Exception):
    """The run never started: the setting, MLX already loaded, gate closed, a pin, a bad argument or an output
    exists."""


def _mlx_loaded() -> bool:
    return any(m == "mlx" or m.startswith("mlx.") for m in sys.modules)


@dataclasses.dataclass
class Seams:
    """What the run reads from outside this module. TEST seams; `run` passes `_real_seams` only."""
    read_corpus_json: Callable[[], bytes]
    read_reference: Callable[[], bytes]
    checkpoint_sha1: Callable[[], str]
    load_network: Callable[[], Any]                    # a FRESH load of the incumbent on every call
    loaded_modules: Callable[[], Dict[str, str]]
    read_repo: Callable[[str], bytes]                  # a pinned repository file, by its relative path
    read_fixture: Callable[[str], bytes]               # an entry-26 fixture, by its recorded absolute path


def compare(a, b, spec: Mapping[str, Any]) -> Dict[str, Any]:
    """K.agree (bitwise; max|a - b| <= cross_rel_tol * max|b| + cross_abs_tol), plus compact failure evidence when it
    is outside the tolerance: the count of elements outside it, and the worst element's index and both values."""
    out = K.agree(a, b, spec)
    if not out["within_tol"] and not out.get("shape_mismatch"):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        d = np.abs(a - b)
        i = int(np.argmax(d))
        out.update(n_outside=int((d > spec["cross_rel_tol"] * out["scale"] + spec["cross_abs_tol"]).sum()),
                   worst={"index": [int(x) for x in np.unravel_index(i, d.shape)] if d.ndim else [],
                          "got": float(a.flat[i]), "reference": float(b.flat[i])})
    return out


# ─────────────────────────────── phase 1: the policy head against NumPy ───────────────────────────────

def fixture_entries(seams: Seams) -> Dict[str, Dict[str, Any]]:
    raw = seams.read_repo(f"{ENTRY26}/fixtures.jsonl")
    if M._sha(raw) != PINNED[f"{ENTRY26}/fixtures.jsonl"]:
        raise Void("fixtures", f"{ENTRY26}/fixtures.jsonl is not the pinned bytes")
    return {e["name"]: e for e in map(json.loads, raw.decode().splitlines())}


def fixture_case(case: str, entries, seams: Seams) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """The saved arrays of `reference_<case>.npz`; VOID unless its bytes are the ones entry 26 recorded, at the
    recorded path."""
    name = f"reference_{case}"
    entry = entries.get(name)
    if entry is None or entry.get("path") != str(FIXTURE_DIR / f"{name}.npz"):
        raise Void("fixtures", f"{name}: not recorded at {FIXTURE_DIR} in {ENTRY26}/fixtures.jsonl")
    raw = seams.read_fixture(entry["path"])
    if M._sha(raw) != entry["sha256"]:
        raise Void("fixtures", f"{name}.npz is not the recorded file")
    with np.load(io.BytesIO(raw)) as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, {"path": entry["path"], "sha256": entry["sha256"]}


def rebuild_head(arrays: Mapping[str, np.ndarray], case: str):
    """A policy head set from the saved float64 parameters and BatchNorm statistics, in evaluation mode; VOID unless
    reading it back gives the saved values exactly, eps included."""
    import mlx.core as mx

    from .network import PolicyHead

    p = {k[len("param."):]: v for k, v in arrays.items() if k.startswith("param.")}
    ph = PolicyHead(in_channels=int(p["conv.weight"].shape[-1]), hidden=int(p["fc.weight"].shape[0]))
    for k in K.PH_KEYS:
        module, attr = k.split(".")
        setattr(getattr(ph, module), attr, mx.array(p[k].astype(np.float32)))
    ph.bn.running_mean = mx.array(p["bn.running_mean"].astype(np.float32))
    ph.bn.running_var = mx.array(p["bn.running_var"].astype(np.float32))
    ph.eval()
    back = K.policy_head_params(ph)
    if set(back) != set(p) or not all(np.array_equal(back[k], p[k]) for k in p):
        raise Void("fixtures", f"{case}: the rebuilt policy head is not the saved one")
    return ph


def judge_reference(out: Mapping[str, Any], dev: str) -> List[str]:
    """Every way one device misses phase 1 on one case (empty: it passes)."""
    d, bad = out["devices"][dev], []
    bad += [f"{k} vs NumPy" for k, v in d["vs_numpy"].items() if not v["within_tol"]]
    if not d["loss_vs_numpy"]["within_tol"]:
        bad.append("loss vs NumPy")
    bad += [f"{k} repeat" for k, v in d["repeats"].items() if not all(v)]
    if any(x != d["loss"][0] for x in d["loss"][1:]):
        bad.append("loss repeat")
    bad += [f"{k} additivity" for k, v in d.get("additivity_vs_full", {}).items() if not v["within_tol"]]
    dirc = out["directional"]
    if not dirc.get("skipped") and not dirc.get(f"mlx_{dev}_ok"):
        bad.append("directional vs central difference")
    return [f"{out['case']}: {x}" for x in bad]


def policy_head_phase(spec, cases, sink, log) -> Dict[str, Any]:
    """Entry 26's case runner on each saved case; the reference must validate itself (else VOID)."""
    res: Dict[str, Any] = {"cases": {}}
    for case in spec["policy_head_cases"]:
        a, source = cases[case]
        ph = rebuild_head(a, case)
        out = K.reference_case(case, ph, a["F"], a["rows"], a["cols"], a["mask"], a["target"], a["w"], spec, sink)
        dirc = out["directional"]
        if (dirc.get("skipped") and case in spec["directional_required"]) or \
                (not dirc.get("skipped") and not dirc.get("analytic_ok")):
            raise Void("reference", f"{case}: the NumPy reference failed its own central-difference check ({dirc})")
        res["cases"][case] = {"source": source, "directional_skipped": bool(dirc.get("skipped")),
                              "gpu_failures": judge_reference(out, spec["devices"][-1]),
                              "cpu_failures": judge_reference(out, spec["devices"][0])}
        log("policy_head_case", case=case, gpu_failures=len(res["cases"][case]["gpu_failures"]))
    res["gpu_failures"] = [x for c in res["cases"].values() for x in c["gpu_failures"]]
    res["cpu_own_failures"] = [x for c in res["cases"].values() for x in c["cpu_failures"]]
    res["status"] = "FAIL" if res["gpu_failures"] else "PASS"
    return res


# ─────────────────────────────── phases 2 and 3: the full model ───────────────────────────────

def full_block(state_name, b, state_np, state_digest, make_net, batch_np, tv_np, recipe, spec, sink) -> Dict[str, Any]:
    """The training loss (search_distillation_d2.train_student's) at one state on one batch, on each device three
    times, each from a FRESH network set to the state and fresh batch arrays (both digests checked): loss terms,
    logits, values and every gradient tensor, each evaluation against the CPU's first. No optimizer update. Plus
    weighted-record additivity: g(first half) + g(second half) = g(all), without the L2 term."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten

    from . import search_distillation_d2 as D
    from .trainer import MainModule, compute_masked_log_probs

    label = f"{state_name}_b{b}"
    names_in = ("boards", "rows", "cols", "mask", "target")
    in_digest = K._digest({**dict(zip(names_in, batch_np)), "teacher_values": tv_np})
    legal = np.asarray(batch_np[3]) > 0.5
    evals: Dict[str, list] = {}
    additivity: Dict[str, Dict[str, Any]] = {}
    for dev in spec["devices"]:
        evals[dev] = []
        with on(dev):
            for r in range(spec["repeats"]):
                net = make_net()
                net.eval()
                net.update(tree_unflatten([(k, mx.array(v)) for k, v in state_np.items()]))
                if D.state_digest(net) != state_digest:
                    raise Void("state", f"{label} on {dev}: the network is not the state")
                ex = K._ex(batch_np, tv_np)
                arrays = {n: np.array(ex[n]) for n in names_in}
                if K._digest({**arrays, "teacher_values": np.array(ex["tv"])}) != in_digest:
                    raise Void("inputs", f"{label} on {dev}: the batch arrays are not the batch")
                surface = MainModule(net.encoder, net.policy_head)
                B = batch_np[0].shape[0]

                def loss_fn(_s, w=None):
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
                lg, vl = np.array(logits), np.array(values)
                K._finite(f"{label} full model on {dev}", [np.array(loss), lg, vl, *gnp.values()])
                e = {"loss": float(loss), "policy": float(policy), "anchor": float(anchor), "logits": lg,
                     "values": vl, "grads": gnp}
                evals[dev].append(e)
                sink.line("full.jsonl", {"block": label, "state": state_name, "batch_update": b, "device": dev,
                                         "repeat": r, **{t: e[t] for t in ("loss", "policy", "anchor")},
                                         "grad_norm": math.sqrt(sum(float((v.astype(np.float64) ** 2).sum())
                                                                    for v in gnp.values())),
                                         "grad_digest": K._digest(gnp), "logits_digest": K._digest({"logits": lg}),
                                         "values_digest": K._digest({"values": vl})})
                if r == 0:
                    halves = []
                    for part in (np.arange(B) < B // 2, np.arange(B) >= B // 2, np.ones(B, bool)):
                        w = mx.array(part.astype(np.float32))
                        _, gw = nn.value_and_grad(surface, lambda s, w=w: loss_fn(s, w=w)[0])(surface)
                        mx.eval(gw)
                        halves.append({k: np.array(v).astype(np.float64) for k, v in tree_flatten(gw)})
                    K._finite(f"{label} additivity on {dev}", [v for h in halves for v in h.values()])
                    additivity[dev] = {k: compare(halves[0][k] + halves[1][k], halves[2][k], spec) for k in halves[2]}
                del net, surface
    ref = evals[spec["devices"][0]][0]
    out: Dict[str, Any] = {"block": label, "state": state_name, "batch_update": b, "state_digest": state_digest,
                           "inputs_digest": in_digest, "legal_slots": int(legal.sum()), "devices": {}}
    for dev in spec["devices"]:
        e = evals[dev]
        d = {"loss_terms": [{t: x[t] for t in ("loss", "policy", "anchor")} for x in e],
             "loss_terms_vs_reference": [{t: compare(np.array(x[t]), np.array(ref[t]), spec)
                                          for t in ("loss", "policy", "anchor")} for x in e],
             "logits_legal_vs_reference": [compare(x["logits"][legal], ref["logits"][legal], spec) for x in e],
             "logits_padding_equal": [bool(np.array_equal(x["logits"][~legal], ref["logits"][~legal])) for x in e],
             "values_vs_reference": [compare(x["values"], ref["values"], spec) for x in e],
             "grads_vs_reference": [{k: compare(x["grads"][k], ref["grads"][k], spec) for k in ref["grads"]}
                                    for x in e],
             "repeats_bitwise": [{"loss_terms": all(x[t] == e[0][t] for t in ("loss", "policy", "anchor")),
                                  "logits": bool(np.array_equal(x["logits"], e[0]["logits"])),
                                  "values": bool(np.array_equal(x["values"], e[0]["values"])),
                                  "grads": {k: bool(np.array_equal(x["grads"][k], e[0]["grads"][k]))
                                            for k in ref["grads"]}} for x in e[1:]],
             "additivity": additivity[dev]}
        d["failures"] = judge_block(d)
        out["devices"][dev] = d
    sink.write(f"full/{label}.json", out)
    return out


def judge_block(d: Mapping[str, Any]) -> List[str]:
    """Every way one device misses the full-model rule on one block (empty: it passes)."""
    bad = []
    for i, terms in enumerate(d["loss_terms_vs_reference"]):
        bad += [f"evaluation {i}: {t}" for t, v in terms.items() if not v["within_tol"]]
    for i, v in enumerate(d["logits_legal_vs_reference"]):
        if not v["within_tol"]:
            bad.append(f"evaluation {i}: legal logits")
    bad += [f"evaluation {i}: padded logits" for i, ok in enumerate(d["logits_padding_equal"]) if not ok]
    for i, v in enumerate(d["values_vs_reference"]):
        if not v["within_tol"]:
            bad.append(f"evaluation {i}: values")
    for i, g in enumerate(d["grads_vs_reference"]):
        bad += [f"evaluation {i}: {k}" for k, v in g.items() if not v["within_tol"]]
    for i, rp in enumerate(d["repeats_bitwise"], start=1):
        bad += [f"repeat {i}: {k}" for k in ("loss_terms", "logits", "values") if not rp[k]]
        bad += [f"repeat {i}: {k}" for k, ok in rp["grads"].items() if not ok]
    bad += [f"additivity: {k}" for k, v in d["additivity"].items() if not v["within_tol"]]
    return bad


def full_phase(state_name, state_np, state_digest, make_net, batches, tv, recipe, spec, sink, log) -> Dict[str, Any]:
    res: Dict[str, Any] = {"state": state_name, "state_digest": state_digest, "blocks": {}}
    gpu, cpu = spec["devices"][-1], spec["devices"][0]
    for b in spec["batches"]:
        blk = full_block(state_name, b, state_np, state_digest, make_net, batches[b][0], tv[batches[b][1]], recipe,
                         spec, sink)
        res["blocks"][blk["block"]] = {"gpu_failures": blk["devices"][gpu]["failures"],
                                       "cpu_own_failures": [x for x in blk["devices"][cpu]["failures"]
                                                            if x.startswith(("repeat", "additivity"))]}
        log("full_block", block=blk["block"], gpu_failures=len(blk["devices"][gpu]["failures"]))
    res["gpu_failures"] = [f"{n}: {x}" for n, v in res["blocks"].items() for x in v["gpu_failures"]]
    res["cpu_own_failures"] = [f"{n}: {x}" for n, v in res["blocks"].items() for x in v["cpu_own_failures"]]
    res["status"] = "FAIL" if res["gpu_failures"] else "PASS"
    return res


# ─────────────────────────────── entry 26, for the predictions ───────────────────────────────

def entry26_policy_head(spec, seams: Seams, out_dir: pathlib.Path) -> Dict[str, Any]:
    """Prediction 1: the CPU's losses and largest differences from NumPy, and NumPy's loss, against entry 26's;
    the GPU's beside it (an observation)."""
    res: Dict[str, Any] = {}
    for case in spec["policy_head_cases"]:
        old = json.loads(seams.read_repo(f"{ENTRY26}/reference/{case}.json"))
        new = json.loads((out_dir / "reference" / f"{case}.json").read_text())
        r = {"numpy_loss_equal": new["numpy_loss"] == old["numpy_loss"]}
        for dev in spec["devices"]:
            o, n = old["devices"][dev], new["devices"][dev]
            diff = [k for k in n["vs_numpy"] if n["vs_numpy"][k]["max_abs"] != o["vs_numpy"][k]["max_abs"]]
            r[dev] = {"losses_equal": all(x == y for x, y in zip(n["loss"], o["loss"])),
                      "losses_compared": min(len(n["loss"]), len(o["loss"])), "tensors": len(n["vs_numpy"]),
                      "max_abs_different": diff}
        res[case] = r
    res["cpu_all_equal"] = all(r["numpy_loss_equal"] and r["cpu"]["losses_equal"] and not r["cpu"]["max_abs_different"]
                               for r in res.values())
    return res


def entry26_full(seams: Seams, out_dir: pathlib.Path) -> Dict[str, Any]:
    """Prediction 2: at S0 on the update-0 batch, the CPU's logits digest and policy term against entry 26's."""
    old = [json.loads(x) for x in seams.read_repo(f"{ENTRY26}/full.jsonl").decode().splitlines()]
    new = [json.loads(x) for x in (out_dir / "full.jsonl").read_text().splitlines()]
    res = {}
    for dev in ("cpu", "gpu"):
        o = [x for x in old if x["state"] == "S0" and x["device"] == dev]
        n = [x for x in new if x["block"] == "S0_b0" and x["device"] == dev]
        res[dev] = [{f: a[f] == b[f] for f in ("logits_digest", "policy", "anchor", "grad_digest")}
                    for a, b in zip(n, o)]
    res["cpu_prediction_holds"] = all(x["logits_digest"] and x["policy"] for x in res["cpu"]) and len(res["cpu"]) > 0
    return res


# ─────────────────────────────── the run ───────────────────────────────

class _Sink:
    """Every result to disk as it exists: lines fsynced, each result file create-only. No array is ever saved: a
    failing policy-head case's inputs are entry 26's fixtures already."""

    def __init__(self, out_dir: pathlib.Path):
        self.out_dir = out_dir
        for d in ("full", "reference"):
            (out_dir / d).mkdir()
        self.files: List[str] = []
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

    def fixture(self, name, arrays, meta) -> Dict[str, Any]:
        return {"name": name, "saved": False, "reason": "no arrays are saved; the inputs are entry 26's fixtures"}

    def close(self):
        for fh in self.handles.values():
            if not fh.closed:
                fh.close()


def _execute(spec: Mapping[str, Any], seams: Seams, out_dir: pathlib.Path, rec: Dict[str, Any], log) -> None:
    import mlx.core as mx
    import mlx.optimizers as optim

    from . import search_distillation_d2 as D

    if pathlib.Path(D.__file__).resolve() != ROOT / CODE["d2"]:
        raise Void("code", f"{D.__file__} is not {CODE['d2']}")
    recipe = T.recipe_for(spec)
    adam = D.adam_settings(optim.Adam(learning_rate=recipe.learning_rate))
    if {k: adam[k] for k in T.ADAM_DEFAULTS} != T.ADAM_DEFAULTS:
        raise Void("optimizer", f"MLX's Adam resolves to {adam}, not the card's {T.ADAM_DEFAULTS}")
    if spec["devices"] != ["cpu", "gpu"] or spec["repeats"] < 2 or \
            spec["batches"] != [0, spec["prefix_updates"]] or spec["prefix_updates"] < 1:
        raise Void("plan", "the devices, repeats, batches or prefix are not the card's shape")
    rec["recipe"], rec["optimizer"], rec["mlx"] = dataclasses.asdict(recipe), adam, getattr(mx, "__version__", None)

    for rel in [r for r in PINNED if r.startswith(ENTRY26)]:
        if M._sha(seams.read_repo(rel)) != PINNED[rel]:
            raise Void("entry26", f"{rel} is not the pinned bytes")
    entry26 = json.loads(seams.read_repo(f"{ENTRY26}/record.json"))
    entries = fixture_entries(seams)
    cases = {c: fixture_case(c, entries, seams) for c in spec["policy_head_cases"]}
    reference = R.read_reference(spec, seams.read_reference(), recipe)
    training, _ids, rec["corpus"] = R.read_training(spec, seams.read_corpus_json())
    sha1 = seams.checkpoint_sha1()
    if sha1 != M.CHECKPOINT_SHA1:
        raise Void("checkpoint", f"SHA-1 {sha1} is not calib020_0001's")
    rec["checkpoint"] = {"path": M.CHECKPOINT, "sha1": sha1}
    log("inputs_verified", training_records=len(training), fixtures=sorted(cases))

    sink = _Sink(out_dir)
    try:
        ph = policy_head_phase(spec, cases, sink, log)
        rec["phases"]["policy_head"] = ph
        rec["entry26"]["policy_head"] = entry26_policy_head(spec, seams, out_dir)
        log("phase", phase="policy_head", status=ph["status"])
        if ph["status"] != "PASS":
            return

        teacher = seams.load_network()
        teacher.eval()
        if D.state_digest(teacher) != spec["teacher_digest"]:
            raise Void("networks", "the loaded incumbent is not the pinned state")
        rec["modules"] = seams.loaded_modules()
        with on(spec["devices"][-1]):                            # as Stage 3 computed the anchor targets
            _, tv = D.forward(teacher, training)
        tv = np.asarray(tv, dtype=np.float32)
        K._finite("teacher values", [tv])
        rec["teacher_values"] = {"digest": K._digest({"teacher_values": tv}),
                                 "entry26_digest": entry26["teacher_values"]["digest"]}
        rec["teacher_values"]["equals_entry26"] = rec["teacher_values"]["digest"] == entry26["teacher_values"]["digest"]
        log("teacher_values", **rec["teacher_values"])
        draws = K.draw_batches(len(training), recipe, max(spec["batches"]) + 1)
        batches = {b: (tuple(np.array(a) for a in D._batch([training[i] for i in draws[b]])), draws[b])
                   for b in spec["batches"]}
        rec["batches"] = {str(b): draws[b] for b in spec["batches"]}

        s0 = D.snapshot(teacher)
        full = full_phase("S0", s0, K._digest(s0), seams.load_network, batches, tv, recipe, spec, sink, log)
        rec["phases"]["full_model_incumbent"] = full
        rec["entry26"]["full_model"] = entry26_full(seams, out_dir)
        log("phase", phase="full_model_incumbent", status=full["status"])
        if full["status"] != "PASS":
            return

        student = seams.load_network()
        student.eval()
        with on(spec["devices"][-1]):
            u21 = K.reconstruct(student, training, recipe, tv, spec["prefix_updates"], spec, sink, reference)
        del student
        dg = K._digest(u21)
        rec["updated_state"] = {"name": "U21", "updates": spec["prefix_updates"], "digest": dg, "saved": False,
                                "entry26_s21_digest": entry26["states"]["S21"]["digest"],
                                "equals_entry26_s21": dg == entry26["states"]["S21"]["digest"]}
        log("prefix", **rec["updated_state"])
        upd = full_phase("U21", u21, dg, seams.load_network, batches, tv, recipe, spec, sink, log)
        rec["phases"]["full_model_updated"] = upd
        log("phase", phase="full_model_updated", status=upd["status"])
        rec["teacher_digest_after"] = D.state_digest(teacher)
    finally:
        sink.close()
        rec["files"] = list(sink.files)


def verdict(phases: Mapping[str, Any]) -> Dict[str, Any]:
    """PASS only if all three phases pass; otherwise the first phase that does not, which must have FAILED (the
    later ones never ran)."""
    for name in PHASES:
        p = phases.get(name)
        if p is None or p["status"] != "PASS":
            return {"verdict": "FAIL", "failed_phase": name,
                    "not_run": [n for n in PHASES[PHASES.index(name) + 1:]]}
    return {"verdict": "PASS", "failed_phase": None, "not_run": []}


def conclude(spec: Mapping[str, Any], seams: Seams, out_dir: pathlib.Path,
             head: str | None = None) -> Tuple[int, Dict[str, Any]]:
    """THE RECORD BOUNDARY: record.json on every exit this survives; SIGTERM -> VOID at step "terminated"."""
    out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "card": CARD, "card_sha256": CARD_SHA256, "spec": dict(spec),
                           "head": head, "status": None, "written_by": "script", "diagnostic": True,
                           "setting": {SETTING: os.environ.get(SETTING)}, "phases": {}, "entry26": {},
                           "model_state_saved": False, "arrays_saved": False, "held_out_records_read": 0,
                           "causal_claims": None, "strength_verdict": None,
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
        log("start", setting=rec["setting"])
        _execute(spec, seams, out_dir, rec, log)
        rec.update(verdict(rec["phases"]))
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
            rec["setting_at_end"] = {SETTING: os.environ.get(SETTING)}
            rec["total_seconds"] = round(time.monotonic() - t0, 3)
            rec["loadavg_after"] = os.getloadavg()
            log("end", status=rec["status"], verdict=rec.get("verdict"))
            log_fh.close()
            M._write_json_new(out_dir / "record.json", rec)
        finally:
            if main_thread:
                signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)
    return code, rec


def _real_seams(checkpoint: pathlib.Path) -> Seams:
    base = M._real_seams(checkpoint)
    return Seams(read_corpus_json=lambda: (ROOT / R.CORPUS_JSON).read_bytes(),
                 read_reference=lambda: (ROOT / R.REFERENCE).read_bytes(),
                 checkpoint_sha1=base.checkpoint_sha1, load_network=base.load_network,
                 loaded_modules=base.loaded_modules, read_repo=lambda rel: (ROOT / rel).read_bytes(),
                 read_fixture=lambda path: pathlib.Path(path).read_bytes())


def run(checkpoint: str) -> int:
    """THE PUBLIC ENTRY. No seam: the setting, the frozen spec, the pinned card, code and evidence, the real
    sources."""
    if os.environ.get(SETTING) != SETTING_VALUE:
        raise Refused(f"{SETTING} is {os.environ.get(SETTING)!r}, not {SETTING_VALUE!r}: launch a fresh process with it")
    if _mlx_loaded():
        raise Refused("MLX was imported before the setting was checked: launch a fresh process")
    if GATE.SEARCH_DISTILLATION_D2_TF32PATH_AUTHORIZED is not True:
        raise Refused("SEARCH_DISTILLATION_D2_TF32PATH_AUTHORIZED is closed: the validation is not authorized")
    if M._sha((ROOT / CARD).read_bytes()) != CARD_SHA256:
        raise Refused(f"{CARD} is not the pinned card")
    for rel, sha in PINNED.items():
        if M._sha((ROOT / rel).read_bytes()) != sha:
            raise Refused(f"{rel} is not the pinned bytes")
    for mod, rel in ((K, CODE["gradcheck"]), (R, CODE["gradtrace"])):
        if pathlib.Path(mod.__file__).resolve() != ROOT / rel:
            raise Refused(f"{mod.__file__} is not the pinned {rel}")
    ckpt = pathlib.Path(checkpoint)
    if not (ckpt.is_absolute() and ckpt.is_file()):
        raise Refused("--checkpoint must name an existing file by absolute path")
    if not FIXTURE_DIR.is_dir():
        raise Refused(f"{FIXTURE_DIR} is not a directory")
    out = ROOT / EVIDENCE_DIR
    if os.path.lexists(out):
        raise Refused(f"{out} exists: the run is one-shot")
    code, rec = conclude(FROZEN, _real_seams(ckpt), out, head=M._git("rev-parse", "HEAD"))
    print(json.dumps({"status": rec["status"], "void": rec.get("void"), "verdict": rec.get("verdict"),
                      "failed_phase": rec.get("failed_phase"),
                      "phases": {k: v.get("status") for k, v in rec.get("phases", {}).items()},
                      "gpu_failures": {k: len(v.get("gpu_failures", [])) for k, v in rec.get("phases", {}).items()},
                      "head": rec["head"]}))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3 or argv[0] != "run" or argv[1] != "--checkpoint":
        print("usage: MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.search_distillation_d2_tf32path run "
              "--checkpoint <abs path>", file=sys.stderr)
        return EXIT_USAGE
    try:
        return run(argv[2])
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
