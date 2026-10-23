"""SEARCH DISTILLATION D2 -- GRADIENT DIAGNOSTIC: one instrumented replay of Stage 3's 401 updates. BUILT; run only
through its wrapper.

Card: `docs/superpowers/2026-10-22-search-distillation-d2-gradient-diagnostic-card.md`, pinned below by sha256.
Nothing runs on import. `run` refuses unless
`search_distillation_d2_gradtrace_authorization.SEARCH_DISTILLATION_D2_GRADTRACE_AUTHORIZED` is open, read at call time.

A DIAGNOSTIC REPLAY, not a learning experiment: the student is trained again from the unchanged incumbent at Stage 3's
frozen recipe and draw order, on the 108 TRAINING games only, to measure what Stage 3's log could not show. It
judges nothing about the student; the student is never saved.

WHAT THE RUN DOES: verifies the Stage 2 corpus against the pinned `corpus.json` hash and reopens the training games
only -- each phase's context, receipts and completion marker are checked against the manifest, and only the training
games' files are opened, each by its receipt's hash. No held-out game file is opened. It reads Stage 3's record (the
REFERENCE trace) by its pinned hash, loads calib020_0001 twice (teacher and student, evaluation mode, the pinned
digest), and replays the 401 updates with the loop of search_distillation_d2.train_student, recording per update:
  - the batch's training-record indices;
  - the loss, policy and anchor terms and the global pre-clip gradient norm, each compared with the reference;
  - every surface tensor's pre-clip gradient norm, the clip scale and the clipped global norm;
  - every surface tensor's applied update (L2 norm and largest absolute element).
At steps 20, 21 and 361 only, BEFORE that step's update, it also records detailed attribution: the gradient of each
loss component; each record's own policy and anchor gradients; and each record's normalized BatchNorm inputs in every
BatchNorm layer, split into channels whose running variance is below the card's threshold and the rest.

OUTCOME: REPRODUCED if every step's four logged values are within the frozen tolerance of the reference. The first
step outside it stops the replay: MISMATCH, with the step, field and both values. A corpus, reference, identity or
optimizer mismatch, an exception or SIGTERM (the wrapper's 10-minute ceiling) is a VOID. No retry. The record holds
measurements only: it names no cause.

    python -m scripts.GPU.alphazero.search_distillation_d2_gradtrace run --checkpoint <absolute path to calib020_0001>
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
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from . import search_distillation_d2_gradtrace_authorization as GATE
from . import search_distillation_d2_measure as M
from . import search_distillation_d2_train as T

ROOT = M.ROOT
RECORD_KIND = "search-distillation-d2-gradtrace-record/1"
CARD = "docs/superpowers/2026-10-22-search-distillation-d2-gradient-diagnostic-card.md"
CARD_SHA256 = "bb1da144f287a77c7ea4a78c904eb259ecec421d78a80210f8df94100b61ff8f"
EVIDENCE_DIR = "docs/superpowers/evidence/search-distillation-d2-gradtrace"
REFERENCE = "docs/superpowers/evidence/search-distillation-d2-train/record.json"
CORPUS_JSON = T.CORPUS_JSON
ADAM_DEFAULTS = T.ADAM_DEFAULTS
TRACE_FIELDS = ("loss", "policy", "anchor", "grad_norm")
FROZEN = {"corpus_sha256": "a9a425f28c296e36ba334e55bcd26c56164e982145f11a09427436f5351edeb0",
          "teacher_digest": "ae0e71fc395ce6b95d24f9e696f02870c7b3bfc6f8ea83f9d5d0a85c4979816b",
          "reference_sha256": "30b1d3ad6adb4dc575c5c33c4d5ae590324a5e9f1ade4eb0bb89f77ecd667a94",
          "reference_student_digest": "9c73feb215961b074cdf775b485f7c910152b5a4c20ba28bd2875a73b833ffdc",
          "training_ids_sha256": "e90df4d31214eb95d496eb1f30a3945f1aa38b938e44ae0e139600d859cfe07d",
          "training_games": 108, "held_out_games": 12, "training_records": 6403,
          "passes": 4, "steps": 401, "batch_size": 64, "learning_rate": 1e-4, "anchor_weight": 1.0,
          "l2_weight": 1e-4, "clip_norm": 1.0, "seed": 0,
          "rel_tol": 1e-5, "abs_tol": 1e-8, "attribution_steps": [20, 21, 361],
          "low_variance_max": 1e-6, "spike_norm": 1e4}
STEP_LOG_EVERY = 25
EXIT_REPRODUCED, EXIT_MISMATCH, EXIT_VOID, EXIT_REFUSED, EXIT_USAGE = 0, 6, 3, 4, 2
Void, Terminated = M.Void, M.Terminated


class Refused(Exception):
    """The run never started: gate closed, card not the pinned bytes, an output exists, or a bad argument."""


class Mismatch(Exception):
    """A replayed step is outside the frozen tolerance of the reference trace."""

    def __init__(self, detail: Dict[str, Any]):
        super().__init__(json.dumps(detail))
        self.detail = detail


def _require_gate() -> None:
    if GATE.SEARCH_DISTILLATION_D2_GRADTRACE_AUTHORIZED is not True:
        raise Refused("SEARCH_DISTILLATION_D2_GRADTRACE_AUTHORIZED is closed: the diagnostic is not authorized")


@dataclasses.dataclass
class Seams:
    """What the run reads from outside this module. TEST seams; `run` passes `_real_seams` only."""
    read_corpus_json: Callable[[], bytes]
    read_reference: Callable[[], bytes]
    checkpoint_sha1: Callable[[], str]
    load_network: Callable[[], Any]                    # a FRESH load of the incumbent on every call
    loaded_modules: Callable[[], Dict[str, str]]


def close(replayed: float, reference: float, spec: Mapping[str, Any]) -> bool:
    """The frozen tolerance: |replayed - reference| <= abs_tol + rel_tol * |reference|. NaN is never close."""
    return abs(replayed - reference) <= spec["abs_tol"] + spec["rel_tol"] * abs(reference)


# ─────────────────────────────── reading: training games only ───────────────────────────────

def read_training(spec: Mapping[str, Any], raw: bytes) -> Tuple[list, list, Dict[str, Any]]:
    """(training records, their identities, summary) from the pinned manifest. Phase files (context, receipts,
    completion marker) are checked against the manifest; ONLY training games' files are opened, each verified against
    its receipt. Every opened path is listed in the summary."""
    from . import search_distillation_d2 as D
    from . import search_distillation_d2_generate as G
    from . import training_replay_capture as C

    sha = M._sha(raw)
    if sha != spec["corpus_sha256"]:
        raise Void("corpus", f"corpus.json is {sha}, not the pinned {spec['corpus_sha256']}")
    corpus = json.loads(raw)
    if corpus.get("kind") != G.CORPUS_KIND or corpus.get("eligible_for_training") is not True:
        raise Void("corpus", "the manifest is not an eligible D2 corpus")
    if corpus.get("model_digest") != spec["teacher_digest"]:
        raise Void("corpus", "the corpus was not generated by the pinned teacher")
    games = [g for ph in corpus["phases"] for g in ph["games"]]
    train_ids, held_ids = corpus["training_games"], corpus["held_out_games"]
    if (sorted(games) != list(range(len(games))) or sorted(train_ids + held_ids) != sorted(games)
            or held_ids != [g for g in games if G.held_out(g)]):
        raise Void("split", "the manifest's split is not the corpus's slot % 10 == 9 split over all its games")
    if (len(train_ids), len(held_ids)) != (spec["training_games"], spec["held_out_games"]):
        raise Void("split", f"{len(train_ids)} training / {len(held_ids)} held-out games, not the card's")
    held = set(held_ids)
    cdir = pathlib.Path(corpus["corpus_dir"])
    opened: List[str] = []

    def read(d: pathlib.Path, rel: str, limit: int) -> bytes:
        opened.append(f"{d.name}/{rel}")
        return C._read(str(d), rel, limit)

    records, identities, phases = [], [], []
    for ph in corpus["phases"]:
        d = cdir / ph["dir"]
        marker_bytes = read(d, "finalized.json", C.CONTEXT_MAX_BYTES)
        ctx_bytes = read(d, "context.json", C.CONTEXT_MAX_BYTES)
        for f, data, key in (("context.json", ctx_bytes, "context_sha256"),
                             ("finalized.json", marker_bytes, "finalized_sha256")):
            if M._sha(data) != ph[key]:
                raise Void("corpus", f"{ph['dir']}/{f} differs from the manifest")
        marker = json.loads(marker_bytes, parse_constant=C._no_constants)
        if marker.get("schema") != C.SCHEMA or C.sha256_hex(ctx_bytes) != marker.get("context_sha256"):
            raise Void("corpus", f"{ph['dir']}: the context differs from the completion marker")
        ctx = json.loads(ctx_bytes, parse_constant=C._no_constants)
        C.validate_context(ctx)
        if ctx["models"][M.MODEL_ID]["state_digest"] != spec["teacher_digest"]:
            raise Void("corpus", f"{ph['dir']}: the capture context names another model state")
        ids = ctx["scheduled_game_ids"]
        if ids != ph["games"]:
            raise Void("corpus", f"{ph['dir']}: the scheduled games are not the manifest's")
        rec_bytes = read(d, "receipts.jsonl", len(ids) * 1024)
        if C.sha256_hex(rec_bytes) != marker.get("receipts_sha256"):
            raise Void("corpus", f"{ph['dir']}: the receipts differ from the completion marker")
        receipts = {r["game_id"]: r for r in map(json.loads, rec_bytes.decode("utf-8").splitlines())}
        if sorted(receipts) != sorted(ids) or marker.get("games") != len(ids):
            raise Void("corpus", f"{ph['dir']}: the receipts are not exactly one per scheduled game")
        n = 0
        for g in ids:
            if g in held:
                continue                                         # a held-out game: its file is never opened
            r = receipts[g]
            if r.get("path") != C._game_path(g):
                raise Void("corpus", f"{ph['dir']}: receipt {g} has an unexpected path")
            data = read(d, r["path"], ctx["limits"]["game_bytes"])
            if len(data) != r.get("bytes") or C.sha256_hex(data) != r.get("sha256"):
                raise Void("corpus", f"{ph['dir']}/{r['path']}: bytes differ from the receipt")
            env = C.validate_envelope(data, ctx, marker["context_sha256"])
            if env["game_id"] != g or len(env["rows"]) != r.get("rows"):
                raise Void("corpus", f"{ph['dir']}/{r['path']}: identity or rows differ from the receipt")
            recs, row_ids = D.records_from_envelopes([env])
            records += recs
            identities += [{"game": g, "ply": rr.ply, "row_id": i} for rr, i in zip(recs, row_ids)]
            n += len(recs)
        phases.append({"dir": ph["dir"], "training_games": sum(g not in held for g in ids), "training_records": n})
    if len(records) != spec["training_records"]:
        raise Void("split", f"{len(records)} training records, not the card's {spec['training_records']}")
    ids_sha = M._sha(json.dumps([i["row_id"] for i in identities]).encode())
    if ids_sha != spec["training_ids_sha256"]:
        raise Void("corpus", f"the training row identities hash to {ids_sha}, not the pinned "
                             f"{spec['training_ids_sha256']}")
    held_paths = {f"{ph['dir']}/{C._game_path(g)}" for ph in corpus["phases"] for g in ph["games"] if g in held}
    if held_paths & set(opened):                                 # by construction; checked, not assumed
        raise Void("held_out", "a held-out game file was opened")
    return records, identities, {"corpus_json_sha256": sha, "corpus_dir": str(cdir), "phases": phases,
                                 "training_games": train_ids, "held_out_games_never_opened": held_ids,
                                 "training_records": len(records), "training_ids_sha256": ids_sha,
                                 "files_opened": opened}


def read_reference(spec: Mapping[str, Any], raw: bytes, recipe) -> List[Dict[str, float]]:
    """Stage 3's per-update log, from its record by the pinned hash."""
    sha = M._sha(raw)
    if sha != spec["reference_sha256"]:
        raise Void("reference", f"the reference record is {sha}, not the pinned {spec['reference_sha256']}")
    ref = json.loads(raw)
    if ref.get("status") != "COMPLETED" or ref.get("recipe") != dataclasses.asdict(recipe):
        raise Void("reference", "the reference is not a COMPLETED run of this recipe")
    if ref.get("teacher_digest_before") != spec["teacher_digest"] or \
            ref.get("training", {}).get("student_digest") != spec["reference_student_digest"]:
        raise Void("reference", "the reference's teacher or student digest is not the pinned one")
    log = ref.get("training_log") or []
    if [e.get("step") for e in log] != list(range(spec["steps"])):
        raise Void("reference", f"the reference log does not hold steps 0..{spec['steps'] - 1}")
    return log


# ─────────────────────────────── the instrumented replay ───────────────────────────────

def _norms(tree_items) -> list:
    import mlx.core as mx
    return [mx.sqrt(mx.sum(g.astype(mx.float32) ** 2)) for _, g in tree_items]


def _floats(arrays) -> List[float]:
    return [float(a) for a in arrays]


def bn_taps(network, boards, rows, cols, mask, low_variance_max: float) -> Dict[str, Any]:
    """Each record's normalized BatchNorm inputs (x - running_mean) / sqrt(running_var + eps), largest absolute value
    per layer, in channels with running_var < low_variance_max and in the others. The encoder is re-run layer by layer
    as BoardEncoder runs it; the largest difference from the module's own output is recorded (it should be 0)."""
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    from .network import canonicalize_batch
    from .search_distillation_d2 import BOARD

    board, _, _, _ = canonicalize_batch(boards, rows, cols, mask, BOARD)
    enc, taps = network.encoder, []

    def tap(name, bn, x):
        taps.append((name, bn, x))
        return bn(x)

    x = nn.relu(tap("encoder.bn1", enc.bn1, enc.conv1(board)))
    for i, b in enumerate(enc.blocks):
        h = nn.relu(tap(f"encoder.blocks.{i}.bn1", b.bn1, b.conv1(x)))
        x = nn.relu(tap(f"encoder.blocks.{i}.bn2", b.bn2, b.conv2(h)) + x)
    tap("policy_head.bn", network.policy_head.bn, network.policy_head.conv(x))
    reference = enc(board)
    mx.eval(x, reference, *[t[2] for t in taps])
    layers = {}
    for name, bn, pre in taps:
        var = np.array(bn.running_var)
        xh = np.abs((np.array(pre) - np.array(bn.running_mean)) / np.sqrt(var + bn.eps))    # (B, H, W, C)
        low = var < low_variance_max
        per = xh.max(axis=(1, 2))                                                           # (B, C)
        layers[name] = {
            "low_variance_channels": int(low.sum()),
            "max_abs_low": [float(v) for v in (per[:, low].max(axis=1) if low.any() else np.zeros(len(per)))],
            "argmax_channel_low": [int(np.flatnonzero(low)[j]) for j in per[:, low].argmax(axis=1)]
            if low.any() else None,
            "max_abs_other": [float(v) for v in (per[:, ~low].max(axis=1) if (~low).any() else np.zeros(len(per)))]}
    return {"manual_encoder_max_abs_diff": float(np.abs(np.array(x) - np.array(reference)).max()),
            "layers": layers}


def attribute(student, surface, recipe, batch, tv, names, low_variance_max: float) -> Dict[str, Any]:
    """At the CURRENT parameters (before this step's update), read-only: the batch's loss terms and each component's
    gradient; each record's own policy and anchor gradients; the BatchNorm taps. Nothing is updated."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from .search_distillation_d2 import BOARD
    from .trainer import compute_masked_log_probs

    boards, rows, cols, mask, target = batch
    B = boards.shape[0]

    def terms(_surface, w):
        logits, values, _ = student.forward_padded(boards, rows, cols, mask, active_size=BOARD)
        ce = -mx.sum(target * compute_masked_log_probs(logits, mask), axis=1)
        sq = (values - tv) ** 2
        l2 = sum(mx.sum(p ** 2) for _, p in tree_flatten(_surface.trainable_parameters()))
        return ce, sq, l2, values

    def component(which):
        def f(_surface, w):
            ce, sq, l2, _ = terms(_surface, w)
            if which == "policy":
                return mx.sum(w * ce) / B
            if which == "anchor":
                return recipe.anchor_weight * mx.sum(w * sq) / B
            if which == "l2":
                return recipe.l2_weight * l2
            return mx.sum(w * ce) / B + recipe.anchor_weight * mx.sum(w * sq) / B + recipe.l2_weight * l2
        return nn.value_and_grad(surface, f)

    ones = mx.ones((B,))
    ce, sq, _, values = terms(surface, ones)
    mx.eval(ce, sq, values)
    out: Dict[str, Any] = {"per_record": {"policy_ce": _floats(ce), "anchor_sq": _floats(sq),
                                          "value_abs_diff": [math.sqrt(v) for v in _floats(sq)],
                                          "n_legal": [int(n) for n in mx.sum(mask, axis=1).tolist()]}}
    grads = {}
    for which in ("policy", "anchor", "l2", "total"):
        _, g = component(which)(surface, ones)
        flat = tree_flatten(g)
        if [k for k, _ in flat] != names:
            raise RuntimeError("the gradient's tensor names are not the surface's")
        tensor = _norms(flat)
        mx.eval(*tensor)
        grads[which] = flat
        out[f"{which}_grad"] = {"global_norm": math.sqrt(sum(float(t) ** 2 for t in tensor)),
                                "per_tensor": _floats(tensor)}

    def gap(a_flat, b_flat):
        diff = [mx.sum((a - b).astype(mx.float32) ** 2) for (_, a), (_, b) in zip(a_flat, b_flat)]
        ref = [mx.sum(b.astype(mx.float32) ** 2) for _, b in b_flat]
        mx.eval(*diff, *ref)
        return math.sqrt(sum(map(float, diff))) / max(math.sqrt(sum(map(float, ref))), 1e-30)

    summed = [(k, a + b + c) for (k, a), (_, b), (_, c) in zip(grads["policy"], grads["anchor"], grads["l2"])]
    out["components_sum_vs_total_relative_gap"] = gap(summed, grads["total"])
    for which in ("policy", "anchor"):
        f = component(which)
        acc, norms, per_tensor = None, [], []
        for j in range(B):
            _, g = f(surface, mx.array([1.0 if i == j else 0.0 for i in range(B)]))
            flat = tree_flatten(g)
            tensor = _norms(flat)
            acc = flat if acc is None else [(k, a + b) for (k, a), (_, b) in zip(acc, flat)]
            mx.eval(*tensor, *[a for _, a in acc])
            per_tensor.append(_floats(tensor))
            norms.append(math.sqrt(sum(t ** 2 for t in per_tensor[-1])))
        out["per_record"][f"{which}_grad_norm"] = norms
        out["per_record"][f"{which}_grad_per_tensor"] = per_tensor
        out[f"{which}_records_sum_vs_batch_relative_gap"] = gap(acc, grads[which])
    out["bn"] = bn_taps(student, boards, rows, cols, mask, low_variance_max)
    return out


def replay(student, teacher, records, recipe, teacher_values, reference, spec: Mapping[str, Any], trace_fh,
           on_step=None) -> Dict[str, Any]:
    """Stage 3's loop (search_distillation_d2.train_student), step for step, with read-only instrumentation. Stops
    with Mismatch at the first step outside the tolerance. Nothing is saved."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten

    from . import search_distillation_d2 as D
    from .trainer import MainModule, clip_grad_norm, compute_masked_log_probs

    if student is teacher:
        raise D.DistillError("the student and the teacher are the same object")
    for net, who in ((student, "the student"), (teacher, "the teacher")):
        D._require_eval(net, who)
    teacher_digest = D.state_digest(teacher)
    if D.state_digest(student) != teacher_digest:
        raise D.DistillError("the student does not start as an exact copy of the teacher")
    teacher_values = np.asarray(teacher_values, dtype=np.float32)
    if teacher_values.shape != (len(records),):
        raise D.DistillError("teacher values do not match the records")

    before = D.snapshot(student)
    surface = MainModule(student.encoder, student.policy_head)
    names = [k for k, _ in tree_flatten(surface.trainable_parameters())]       # the order every gradient comes in
    if sorted(names) != D.surface_names(student):
        raise D.DistillError("the surface's tensors are not search_distillation_d2's update surface")
    opt = optim.Adam(learning_rate=recipe.learning_rate)
    optimizer = D.adam_settings(opt)

    def loss_fn(_surface, boards, rows, cols, mask, target, tv):
        logits, values, _ = student.forward_padded(boards, rows, cols, mask, active_size=D.BOARD)
        policy = mx.mean(-mx.sum(target * compute_masked_log_probs(logits, mask), axis=1))
        anchor = mx.mean((values - tv) ** 2)
        l2 = sum(mx.sum(p ** 2) for _, p in tree_flatten(_surface.trainable_parameters()))
        return policy + recipe.anchor_weight * anchor + recipe.l2_weight * l2, policy, anchor

    grad_fn = nn.value_and_grad(surface, loss_fn)
    rng = np.random.default_rng(recipe.seed)
    order: List[int] = []
    attribution: Dict[str, Any] = {}
    steps = 0
    for step in range(recipe.steps):
        if len(order) < recipe.batch_size:
            order += rng.permutation(len(records)).tolist()
        idx, order = order[:recipe.batch_size], order[recipe.batch_size:]
        boards, rows, cols, mask, target = D._batch([records[i] for i in idx])
        tv = mx.array(teacher_values[idx])
        if step in spec["attribution_steps"]:
            attribution[str(step)] = {"batch": idx, **attribute(student, surface, recipe,
                                                                (boards, rows, cols, mask, target), tv, names,
                                                                spec["low_variance_max"])}
        params_before = tree_flatten(surface.trainable_parameters())
        (loss, policy, anchor), raw = grad_fn(surface, boards, rows, cols, mask, target, tv)
        grads, norm = clip_grad_norm(raw, recipe.clip_norm)
        opt.update(surface, grads)
        mx.eval(surface.parameters(), opt.state, loss, policy, anchor, norm)
        values = {"loss": float(loss), "policy": float(policy), "anchor": float(anchor), "grad_norm": float(norm)}

        raw_flat, clipped_flat = tree_flatten(raw), tree_flatten(grads)
        params_after = tree_flatten(surface.trainable_parameters())
        if not ([k for k, _ in raw_flat] == [k for k, _ in params_after] == [k for k, _ in params_before] == names):
            raise RuntimeError("the gradient or parameter names are not the surface's")
        raw_n, clipped_n = _norms(raw_flat), _norms(clipped_flat)
        delta = [(k, a - b) for (k, a), (_, b) in zip(params_after, params_before)]
        upd_n = _norms(delta)
        upd_max = [mx.max(mx.abs(d)) for _, d in delta]
        mx.eval(*raw_n, *clipped_n, *upd_n, *upd_max)
        ref = reference[step]
        line = {"step": step, "batch": idx, **values, "reference": {f: ref[f] for f in TRACE_FIELDS},
                "clip_scale": float(np.minimum(np.float32(1.0), np.float32(recipe.clip_norm) /
                                               (np.float32(values["grad_norm"]) + np.float32(1e-8)))),
                "clipped_global_norm": math.sqrt(sum(v ** 2 for v in _floats(clipped_n))),
                "raw_per_tensor": _floats(raw_n), "update_l2_per_tensor": _floats(upd_n),
                "update_max_abs_per_tensor": _floats(upd_max)}
        trace_fh.write(json.dumps(line) + "\n")
        trace_fh.flush()
        steps = step + 1
        bad = [f for f in TRACE_FIELDS if not close(values[f], ref[f], spec)]
        if bad:
            raise Mismatch({"step": step, "fields": bad, "replayed": {f: values[f] for f in bad},
                            "reference": {f: ref[f] for f in bad}, "steps_replayed": steps})
        if str(step) in attribution:
            attribution[str(step)]["step_grad_norm"] = values["grad_norm"]
            attribution[str(step)]["total_grad_matches_step"] = close(
                attribution[str(step)]["total_grad"]["global_norm"], values["grad_norm"], spec)
        if on_step is not None:
            on_step(line)

    D._require_eval(student, "the student after the replay")
    if D.state_digest(teacher) != teacher_digest:
        raise D.DistillError("the teacher changed during the replay")
    checks = D.check_after(before, D.snapshot(student), D.surface_names(student))
    return {"optimizer": optimizer, "steps_replayed": steps, "checks": checks, "surface_names": names,
            "teacher_digest": teacher_digest, "student_digest": D.state_digest(student), "attribution": attribution}


def summarize(trace: Sequence[Mapping[str, Any]], names: Sequence[str], spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Measured, per update: which surface tensor carries the largest share of the squared pre-clip norm, split by
    whether the update is a spike (the reference norm above the card's spike threshold)."""
    out = {}
    for label, rows in (("spike", [t for t in trace if t["reference"]["grad_norm"] > spec["spike_norm"]]),
                        ("other", [t for t in trace if t["reference"]["grad_norm"] <= spec["spike_norm"]])):
        counts, shares = {}, []
        for t in rows:
            sq = [v * v for v in t["raw_per_tensor"]]
            k = max(range(len(sq)), key=sq.__getitem__)
            counts[names[k]] = counts.get(names[k], 0) + 1
            shares.append(sq[k] / sum(sq))
        out[label] = {"updates": len(rows), "largest_tensor_counts": dict(sorted(counts.items())),
                      "largest_tensor_share_min": min(shares) if shares else None,
                      "largest_tensor_share_median": sorted(shares)[len(shares) // 2] if shares else None}
    return out


def _execute(spec: Mapping[str, Any], seams: Seams, out_dir: pathlib.Path, rec: Dict[str, Any], log) -> None:
    import mlx.core as mx
    import mlx.optimizers as optim

    from . import search_distillation_d2 as D

    recipe = T.recipe_for(spec)
    expected_adam = D.adam_settings(optim.Adam(learning_rate=recipe.learning_rate))
    if {k: expected_adam[k] for k in ADAM_DEFAULTS} != ADAM_DEFAULTS:
        raise Void("optimizer", f"MLX's Adam resolves to {expected_adam}, not the card's {ADAM_DEFAULTS}")
    rec["recipe"], rec["optimizer"] = dataclasses.asdict(recipe), expected_adam
    rec["device"], rec["mlx"] = str(mx.default_device()), getattr(mx, "__version__", None)

    reference = read_reference(spec, seams.read_reference(), recipe)
    training, identities, rec["corpus"] = read_training(spec, seams.read_corpus_json())
    rec["training_identities"] = [[i["game"], i["ply"]] for i in identities]
    log("inputs_verified", training_records=len(training), reference_steps=len(reference))

    sha1 = seams.checkpoint_sha1()
    if sha1 != M.CHECKPOINT_SHA1:
        raise Void("checkpoint", f"SHA-1 {sha1} is not calib020_0001's")
    rec["checkpoint"] = {"path": M.CHECKPOINT, "sha1": sha1}
    teacher, student = seams.load_network(), seams.load_network()
    teacher.eval()
    student.eval()
    if student is teacher:
        raise Void("networks", "the student and the teacher are one object")
    rec["modes_at_load"] = {"teacher": M._require_eval(teacher, "the teacher after load"),
                            "student": M._require_eval(student, "the student after load")}
    teacher_digest = D.state_digest(teacher)
    if teacher_digest != spec["teacher_digest"] or D.state_digest(student) != teacher_digest:
        raise Void("networks", f"the loaded state {teacher_digest} is not the pinned teacher, or the copies differ")
    rec["teacher_digest_before"] = teacher_digest
    rec["modules"] = seams.loaded_modules()
    log("loaded")

    _, train_values = D.forward(teacher, training)              # as Stage 3 computed the anchor targets
    log("teacher_values")

    def on_step(line):
        if line["step"] % STEP_LOG_EVERY == 0 or line["step"] == recipe.steps - 1:
            log("step", step=line["step"], grad_norm=line["grad_norm"])

    t0 = time.monotonic()
    trace_fh = open(out_dir / "trace.jsonl", "x")
    try:
        try:
            out = replay(student, teacher, training, recipe, train_values, reference, spec, trace_fh, on_step)
        finally:
            rec["replay_seconds"] = round(time.monotonic() - t0, 3)
            trace_fh.flush()
            os.fsync(trace_fh.fileno())
            trace_fh.close()
    except Mismatch as m:
        rec["mismatch"] = m.detail
        rec["trace"] = "MISMATCH"
        log("mismatch", **m.detail)
        return
    if out["optimizer"] != expected_adam:
        raise Void("optimizer", f"the replay resolved Adam to {out['optimizer']}, not {expected_adam}")
    rec["trace"] = "REPRODUCED"
    rec["replay"] = {k: out[k] for k in ("steps_replayed", "checks", "surface_names", "teacher_digest",
                                         "student_digest")}
    rec["final_state_bitwise_equal_to_reference"] = out["student_digest"] == spec["reference_student_digest"]
    rec["attribution"] = out["attribution"]
    trace = [json.loads(line) for line in (out_dir / "trace.jsonl").read_text().splitlines()]
    rec["largest_tensor"] = summarize(trace, out["surface_names"], spec)
    rec["teacher_digest_after"] = D.state_digest(teacher)
    if rec["teacher_digest_after"] != teacher_digest:
        raise Void("teacher", "the teacher changed")
    log("replayed", trace=rec["trace"], bitwise=rec["final_state_bitwise_equal_to_reference"])


def conclude(spec: Mapping[str, Any], seams: Seams, out_dir: pathlib.Path,
             head: str | None = None) -> Tuple[int, Dict[str, Any]]:
    """THE RECORD BOUNDARY (as Stage 3's): record.json on every exit; SIGTERM -> VOID at step "terminated"."""
    out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "card": CARD, "card_sha256": CARD_SHA256, "spec": dict(spec),
                           "head": head, "status": None, "trace": None, "written_by": "script",
                           "diagnostic_replay": True, "student_saved": False, "held_out_records_read": 0,
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
        log("start")
        _execute(spec, seams, out_dir, rec, log)
        rec["status"] = "COMPLETED"
        code = EXIT_REPRODUCED if rec["trace"] == "REPRODUCED" else EXIT_MISMATCH
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
                rec["trace"] = None
            rec["total_seconds"] = round(time.monotonic() - t0, 3)
            rec["loadavg_after"] = os.getloadavg()
            log("end", status=rec["status"], trace=rec["trace"])
            log_fh.close()
            M._write_json_new(out_dir / "record.json", rec)
        finally:
            if main_thread:
                signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)
    return code, rec


def _real_seams(checkpoint: pathlib.Path) -> Seams:
    base = M._real_seams(checkpoint)
    return Seams(read_corpus_json=lambda: (ROOT / CORPUS_JSON).read_bytes(),
                 read_reference=lambda: (ROOT / REFERENCE).read_bytes(), checkpoint_sha1=base.checkpoint_sha1,
                 load_network=base.load_network, loaded_modules=base.loaded_modules)


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
    code, rec = conclude(FROZEN, _real_seams(ckpt), out, head=M._git("rev-parse", "HEAD"))
    print(json.dumps({"status": rec["status"], "trace": rec.get("trace"), "void": rec.get("void"),
                      "mismatch": rec.get("mismatch"), "head": rec["head"],
                      "bitwise": rec.get("final_state_bitwise_equal_to_reference")}))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3 or argv[0] != "run" or argv[1] != "--checkpoint":
        print("usage: python -m scripts.GPU.alphazero.search_distillation_d2_gradtrace run --checkpoint <abs path>",
              file=sys.stderr)
        return EXIT_USAGE
    try:
        return run(argv[2])
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
