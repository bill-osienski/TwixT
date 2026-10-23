"""SEARCH DISTILLATION D2 -- REPRODUCIBILITY DIAGNOSTIC: three replays of Stage 3's updates 0-21 in one process.
BUILT; run only through its wrapper.

Card: `docs/superpowers/2026-10-22-search-distillation-d2-reproducibility-card.md`, pinned below by sha256. Nothing
runs on import. `run` refuses unless
`search_distillation_d2_gradrepeat_authorization.SEARCH_DISTILLATION_D2_GRADREPEAT_AUTHORIZED` is open, read at call
time.

A DIAGNOSTIC REPLAY. The gradient diagnostic (ledger entry 24) reproduced Stage 3 bit for bit through update 20 and
diverged at update 21; it could not say whether the update path repeats. This run replays updates 0-21 three times,
each from a FRESH load of the unchanged incumbent with a fresh optimizer, in the frozen order:
  A -- no attribution;   B -- attribution before updates 20 and 21 (as the gradient diagnostic did);   C -- none.
Every pass uses the same training records (training games only, read as the gradient diagnostic reads them: no
held-out game file is opened), the same draw order, the same teacher values (computed once) and Stage 3's recipe.

RECORDED, each written and flushed to disk as soon as it exists:
  - every update's loss, policy and anchor terms and pre-clip norm, beside Stage 3's and the gradient diagnostic's;
  - at update 21, in every pass: digests of the model state (every parameter and BatchNorm statistic), of the Adam
    state, and of the update's input and target arrays -- before the attribution point, again just before the
    update, and after it;
  - pass B's attribution at updates 20 and 21 (search_distillation_d2_gradtrace.attribute), each in its own file,
    BEFORE that update is computed;
  - AFTER each pass's 22 updates: at the pass's own pre-update-21 state (reloaded into a fresh network, its digest
    checked), the numerical scale of update 21's batch -- policy-head inputs (the gathered features), the hidden
    layer, the legal-move logits' magnitude and spread, the BatchNorm taps -- and a recomputation of update 21's
    loss and gradient. Nothing is measured inside A's or C's 22 updates beyond the digests at update 21.

A DIFFERENCE between passes, or from Stage 3, is an observation: it never stops the run. Invalid inputs, a
non-finite update, an exception or SIGTERM (the wrapper's 10-minute ceiling) are a VOID. No retry. The record holds
measurements only: it names no cause.

    python -m scripts.GPU.alphazero.search_distillation_d2_gradrepeat run --checkpoint <absolute path to calib020_0001>
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

from . import search_distillation_d2_gradrepeat_authorization as GATE
from . import search_distillation_d2_gradtrace as R
from . import search_distillation_d2_measure as M
from . import search_distillation_d2_train as T

ROOT = M.ROOT
RECORD_KIND = "search-distillation-d2-gradrepeat-record/1"
CARD = "docs/superpowers/2026-10-22-search-distillation-d2-reproducibility-card.md"
CARD_SHA256 = "cb51275974f4d7b5069358adf8a2984fe47daf2cb41354dece1696f70324dcb8"
EVIDENCE_DIR = "docs/superpowers/evidence/search-distillation-d2-gradrepeat"
REFERENCE = R.REFERENCE
REPLAY1_TRACE = "docs/superpowers/evidence/search-distillation-d2-gradtrace/trace.jsonl"
CORPUS_JSON = R.CORPUS_JSON
ADAM_DEFAULTS = T.ADAM_DEFAULTS
TRACE_FIELDS = R.TRACE_FIELDS
PLAN = ({"name": "A", "attribution": []}, {"name": "B", "attribution": [20, 21]}, {"name": "C", "attribution": []})
FROZEN = {**{k: R.FROZEN[k] for k in ("corpus_sha256", "teacher_digest", "reference_sha256",
                                      "reference_student_digest", "training_ids_sha256", "training_games",
                                      "held_out_games", "training_records", "passes", "steps", "batch_size",
                                      "learning_rate", "anchor_weight", "l2_weight", "clip_norm", "seed",
                                      "rel_tol", "abs_tol", "low_variance_max")},
          "replay1_trace_sha256": "04a69366864e1fa9c2d7a2d6ad126297e04c424be069500c832c4c47927b316a",
          "prefix_updates": 22, "snapshot_update": 21, "plan": [dict(p) for p in PLAN]}
STEP_LOG_EVERY = 25
EXIT_COMPLETED, EXIT_VOID, EXIT_REFUSED, EXIT_USAGE = 0, 3, 4, 2
Void, Terminated = M.Void, M.Terminated


class Refused(Exception):
    """The run never started: gate closed, card not the pinned bytes, an output exists, or a bad argument."""


def _require_gate() -> None:
    if GATE.SEARCH_DISTILLATION_D2_GRADREPEAT_AUTHORIZED is not True:
        raise Refused("SEARCH_DISTILLATION_D2_GRADREPEAT_AUTHORIZED is closed: the diagnostic is not authorized")


@dataclasses.dataclass
class Seams:
    """What the run reads from outside this module. TEST seams; `run` passes `_real_seams` only."""
    read_corpus_json: Callable[[], bytes]
    read_reference: Callable[[], bytes]
    read_replay1_trace: Callable[[], bytes]
    checkpoint_sha1: Callable[[], str]
    load_network: Callable[[], Any]                    # a FRESH load of the incumbent on every call
    loaded_modules: Callable[[], Dict[str, str]]


def read_replay1(spec: Mapping[str, Any], raw: bytes) -> List[Dict[str, Any]]:
    """The gradient diagnostic's trace (updates 0-21), by the pinned hash."""
    sha = M._sha(raw)
    if sha != spec["replay1_trace_sha256"]:
        raise Void("replay1", f"the gradient diagnostic's trace is {sha}, not the pinned "
                              f"{spec['replay1_trace_sha256']}")
    lines = [json.loads(line) for line in raw.decode("utf-8").splitlines()][:spec["prefix_updates"]]
    if [t.get("step") for t in lines] != list(range(spec["prefix_updates"])):
        raise Void("replay1", "the gradient diagnostic's trace does not hold updates "
                              f"0..{spec['prefix_updates'] - 1}")
    return lines


# ─────────────────────────────── digests ───────────────────────────────

def _np(a):
    import numpy as np
    return np.array(a)


def input_digests(batch, tv, idx) -> Dict[str, Any]:
    from . import training_replay_capture as C
    names = ("boards", "move_rows", "move_cols", "move_mask", "target")
    arrays = {**{n: _np(a) for n, a in zip(names, batch)}, "teacher_values": _np(tv)}
    return {"batch": list(idx), "inputs": C.model_state_digest(arrays),
            "per_array": {n: C.model_state_digest({n: a}) for n, a in arrays.items()}}


def state_digests(student, opt) -> Dict[str, str]:
    from mlx.utils import tree_flatten

    from . import search_distillation_d2 as D
    from . import training_replay_capture as C
    return {"model": D.state_digest(student),
            "adam": C.model_state_digest({k: _np(v) for k, v in tree_flatten(opt.state)})}


# ─────────────────────────────── one pass ───────────────────────────────

def prefix(student, teacher, records, recipe, teacher_values, plan: Mapping[str, Any], spec: Mapping[str, Any],
           sink) -> Dict[str, Any]:
    """Stage 3's loop (search_distillation_d2.train_student) for updates 0..prefix_updates-1. Inside the loop only
    the digests at the snapshot update and the plan's attribution run; each reaches `sink` before the update that
    follows it is computed. Returns the snapshot state (every parameter, numpy) taken just before that update."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten

    from . import search_distillation_d2 as D
    from .trainer import MainModule, clip_grad_norm, compute_masked_log_probs

    if student is teacher:
        raise Void("networks", "the student and the teacher are one object")
    for net, who in ((student, "the student"), (teacher, "the teacher")):
        M._require_eval(net, who)
    if D.state_digest(student) != spec["teacher_digest"]:
        raise Void("networks", f"pass {plan['name']}: the student does not start as the pinned incumbent")
    teacher_values = np.asarray(teacher_values, dtype=np.float32)

    surface = MainModule(student.encoder, student.policy_head)
    names = [k for k, _ in tree_flatten(surface.trainable_parameters())]
    opt = optim.Adam(learning_rate=recipe.learning_rate)
    if D.adam_settings(opt) != sink.expected_adam:
        raise Void("optimizer", f"pass {plan['name']}: Adam resolves to {D.adam_settings(opt)}")

    def loss_fn(_surface, boards, rows, cols, mask, target, tv):
        logits, values, _ = student.forward_padded(boards, rows, cols, mask, active_size=D.BOARD)
        policy = mx.mean(-mx.sum(target * compute_masked_log_probs(logits, mask), axis=1))
        anchor = mx.mean((values - tv) ** 2)
        l2 = sum(mx.sum(p ** 2) for _, p in tree_flatten(_surface.trainable_parameters()))
        return policy + recipe.anchor_weight * anchor + recipe.l2_weight * l2, policy, anchor

    grad_fn = nn.value_and_grad(surface, loss_fn)
    rng = np.random.default_rng(recipe.seed)
    order: List[int] = []
    snap, snapshot = spec["snapshot_update"], None
    for step in range(spec["prefix_updates"]):
        if len(order) < recipe.batch_size:
            order += rng.permutation(len(records)).tolist()
        idx, order = order[:recipe.batch_size], order[recipe.batch_size:]
        boards, rows, cols, mask, target = D._batch([records[i] for i in idx])
        tv = mx.array(teacher_values[idx])
        if step == snap:
            sink.digests({"pass": plan["name"], "point": "before_attribution", "update": step,
                          **state_digests(student, opt), **input_digests((boards, rows, cols, mask, target), tv, idx)})
        if step in plan["attribution"]:
            sink.attribution(plan["name"], step, {"batch": idx, **R.attribute(
                student, surface, recipe, (boards, rows, cols, mask, target), tv, names, spec["low_variance_max"])})
        if step == snap:
            snapshot = D.snapshot(student)
            sink.digests({"pass": plan["name"], "point": "before_update", "update": step,
                          **state_digests(student, opt), **input_digests((boards, rows, cols, mask, target), tv, idx)})
        (loss, policy, anchor), grads = grad_fn(surface, boards, rows, cols, mask, target, tv)
        grads, norm = clip_grad_norm(grads, recipe.clip_norm)
        opt.update(surface, grads)
        mx.eval(surface.parameters(), opt.state, loss, policy, anchor, norm)
        values = {"loss": float(loss), "policy": float(policy), "anchor": float(anchor), "grad_norm": float(norm)}
        sink.update(plan["name"], step, idx, values)
        if not all(math.isfinite(v) for v in values.values()):
            raise Void("non_finite", f"pass {plan['name']} update {step}: {values}")
    sink.digests({"pass": plan["name"], "point": "after_update", "update": snap, **state_digests(student, opt)})
    if D.state_digest(teacher) != spec["teacher_digest"]:
        raise Void("teacher", f"the teacher changed during pass {plan['name']}")
    return {"snapshot": snapshot, "names": names}


# ─────────────────────────────── after each pass: numerical scale at update 21 ───────────────────────────────

def scale(net, batch, tv, recipe, low_variance_max: float) -> Dict[str, Any]:
    """Update 21's batch at `net`'s state: the policy head re-run step by step as PolicyHead.forward_padded runs it
    (the largest difference from the module's own logits is recorded), per record."""
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten

    from . import search_distillation_d2 as D
    from .network import canonicalize_batch
    from .trainer import MainModule, compute_masked_log_probs

    boards, rows, cols, mask, target = batch
    board, rows_c, cols_c, _ = canonicalize_batch(boards, rows, cols, mask, D.BOARD)
    features = net.encoder(board)
    ph = net.policy_head
    x = nn.relu(ph.bn(ph.conv(features)))
    B, Mm = rows_c.shape
    gathered = x[mx.arange(B, dtype=rows_c.dtype)[:, None], rows_c, cols_c, :]
    h = nn.relu(ph.fc(gathered.reshape(B * Mm, 2)))
    logits = ph.out(h).reshape(B, Mm)
    ref_logits, values, _ = net.forward_padded(boards, rows, cols, mask, active_size=D.BOARD)
    lp = compute_masked_log_probs(ref_logits, mask)
    mx.eval(features, gathered, h, logits, ref_logits, values, lp)
    g, hh, lg, rl = np.array(gathered), np.array(h).reshape(B, Mm, -1), np.array(logits), np.array(ref_logits)
    m, t, lpn, f = np.array(mask) > 0.5, np.array(target), np.array(lp), np.abs(np.array(features))
    per: Dict[str, list] = {k: [] for k in ("n_legal", "features_abs_max", "gathered_abs_max", "gathered_mean",
                                            "hidden_abs_max", "logit_max", "logit_min", "logit_mean", "logit_std",
                                            "logit_spread", "logit_abs_max", "softmax_max", "softmax_entropy",
                                            "policy_ce", "target_weighted_logit")}
    for i in range(B):
        L = m[i]
        z, p = lg[i][L].astype(np.float64), np.exp(lpn[i][L].astype(np.float64))
        per["n_legal"].append(int(L.sum()))
        per["features_abs_max"].append(float(f[i].max()))
        per["gathered_abs_max"].append([float(v) for v in np.abs(g[i][L]).max(axis=0)])
        per["gathered_mean"].append([float(v) for v in g[i][L].mean(axis=0)])
        per["hidden_abs_max"].append(float(np.abs(hh[i][L]).max()))
        per["logit_max"].append(float(z.max()))
        per["logit_min"].append(float(z.min()))
        per["logit_mean"].append(float(z.mean()))
        per["logit_std"].append(float(z.std()))
        per["logit_spread"].append(float(z.max() - z.min()))
        per["logit_abs_max"].append(float(np.abs(z).max()))
        per["softmax_max"].append(float(p.max()))
        per["softmax_entropy"].append(float(-(p[p > 0] * np.log(p[p > 0])).sum()))
        per["policy_ce"].append(float(-(t[i][L] * lpn[i][L]).sum()))
        per["target_weighted_logit"].append(float((t[i][L] * z).sum()))
    out = {"manual_logits_max_abs_diff": float(np.abs(np.where(m, lg - rl, 0.0)).max()),
           "values": [float(v) for v in np.array(values)], "per_record": per,
           "bn": R.bn_taps(net, boards, rows, cols, mask, low_variance_max)}

    surface = MainModule(net.encoder, net.policy_head)

    def loss_fn(_surface):
        lo, v, _ = net.forward_padded(boards, rows, cols, mask, active_size=D.BOARD)
        policy = mx.mean(-mx.sum(target * compute_masked_log_probs(lo, mask), axis=1))
        anchor = mx.mean((v - tv) ** 2)
        l2 = sum(mx.sum(q ** 2) for _, q in tree_flatten(_surface.trainable_parameters()))
        return policy + recipe.anchor_weight * anchor + recipe.l2_weight * l2, policy, anchor

    (loss, policy, anchor), grads = nn.value_and_grad(surface, loss_fn)(surface)
    flat = tree_flatten(grads)
    norms = R._norms(flat)
    mx.eval(loss, policy, anchor, *norms)
    per_tensor = R._floats(norms)
    out["recomputed"] = {"loss": float(loss), "policy": float(policy), "anchor": float(anchor),
                         "grad_norm": math.sqrt(sum(v * v for v in per_tensor)),
                         "names": [k for k, _ in flat], "per_tensor": per_tensor}
    return out


def measure(seams: Seams, snapshot, digests_before: Mapping[str, Any], records, teacher_values, idx, recipe,
            spec: Mapping[str, Any]) -> Dict[str, Any]:
    """A fresh network set to the pass's pre-update-21 state (its digest must be the recorded one), on update 21's
    batch rebuilt from the same records (its input digest must be the recorded one)."""
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_unflatten

    from . import search_distillation_d2 as D

    net = seams.load_network()
    net.eval()
    net.update(tree_unflatten([(k, mx.array(v)) for k, v in snapshot.items()]))
    if D.state_digest(net) != digests_before["model"]:
        raise Void("measurement", "the measurement network is not the pass's pre-update state")
    M._require_eval(net, "the measurement network")
    batch = D._batch([records[i] for i in idx])
    tv = mx.array(np.asarray(teacher_values, dtype=np.float32)[idx])
    if input_digests(batch, tv, idx)["inputs"] != digests_before["inputs"]:
        raise Void("measurement", "update 21's rebuilt batch is not the pass's")
    return scale(net, batch, tv, recipe, spec["low_variance_max"])


# ─────────────────────────────── comparison ───────────────────────────────

def compare(updates: Mapping[str, List[Dict[str, Any]]], reference, replay1, digests, measurements,
            spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Measured equality, bit for bit (and, beside it, the gradient diagnostic's tolerance)."""
    def differing(a, b):
        return [i for i, (x, y) in enumerate(zip(a, b)) if any(x[f] != y[f] for f in TRACE_FIELDS)]

    def outside(a, b):
        return [i for i, (x, y) in enumerate(zip(a, b)) if not all(R.close(x[f], y[f], spec) for f in TRACE_FIELDS)]

    names = list(updates)
    out: Dict[str, Any] = {"vs_stage3": {}, "vs_gradient_diagnostic": {}, "between_passes": {}}
    for n in names:
        for key, other in (("vs_stage3", reference), ("vs_gradient_diagnostic", replay1)):
            other = other[:len(updates[n])]
            out[key][n] = {"updates_differing_bitwise": differing(updates[n], other),
                           "updates_outside_tolerance": outside(updates[n], other)}
    by = {(d["pass"], d["point"]): d for d in digests}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pair = {"updates_differing_bitwise": differing(updates[a], updates[b])}
            for point in ("before_attribution", "before_update", "after_update"):
                da, db = by.get((a, point), {}), by.get((b, point), {})
                pair[point] = {k: da.get(k) == db.get(k) for k in ("model", "adam", "inputs") if k in da or k in db}
            pair["measurements_equal"] = (measurements.get(a) == measurements.get(b)) if a in measurements and \
                b in measurements else None
            out["between_passes"][f"{a}-{b}"] = pair
    out["attribution_left_state_unchanged"] = {
        n: by[(n, "before_attribution")]["model"] == by[(n, "before_update")]["model"] and
        by[(n, "before_attribution")]["adam"] == by[(n, "before_update")]["adam"]
        for n in names if (n, "before_attribution") in by and (n, "before_update") in by}
    return out


# ─────────────────────────────── the run ───────────────────────────────

class _Sink:
    """Every observation to disk as it exists: lines flushed, each attribution and measurement its own fsynced
    create-only file -- so a later stop cannot discard an earlier observation."""

    def __init__(self, out_dir: pathlib.Path, reference, replay1, expected_adam):
        self.out_dir, self.reference, self.replay1, self.expected_adam = out_dir, reference, replay1, expected_adam
        (out_dir / "attribution").mkdir()
        (out_dir / "measurements").mkdir()
        self.updates_fh = open(out_dir / "updates.jsonl", "x")
        self.digests_fh = open(out_dir / "digests.jsonl", "x")
        self.updates: Dict[str, List[Dict[str, Any]]] = {}
        self.digest_list: List[Dict[str, Any]] = []
        self.files: List[str] = []

    @staticmethod
    def _line(fh, obj):
        fh.write(json.dumps(obj) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    def update(self, name, step, idx, values):
        self.updates.setdefault(name, []).append(values)
        self._line(self.updates_fh, {"pass": name, "update": step, "batch": idx, **values,
                                     "stage3": {f: self.reference[step][f] for f in TRACE_FIELDS},
                                     "gradient_diagnostic": {f: self.replay1[step][f] for f in TRACE_FIELDS}})

    def digests(self, entry):
        self.digest_list.append(entry)
        self._line(self.digests_fh, entry)

    def _file(self, rel, obj):
        M._write_json_new(self.out_dir / rel, obj)
        self.files.append(rel)

    def attribution(self, name, step, data):
        self._file(f"attribution/pass_{name}_update_{step}.json", {"pass": name, "update": step, **data})

    def measurement(self, name, data):
        self._file(f"measurements/pass_{name}.json", {"pass": name, **data})

    def close(self):
        for fh in (self.updates_fh, self.digests_fh):
            if not fh.closed:
                fh.close()


def _execute(spec: Mapping[str, Any], seams: Seams, out_dir: pathlib.Path, rec: Dict[str, Any], log) -> None:
    import mlx.core as mx
    import mlx.optimizers as optim

    from . import search_distillation_d2 as D

    recipe = T.recipe_for(spec)
    expected_adam = D.adam_settings(optim.Adam(learning_rate=recipe.learning_rate))
    if {k: expected_adam[k] for k in ADAM_DEFAULTS} != ADAM_DEFAULTS:
        raise Void("optimizer", f"MLX's Adam resolves to {expected_adam}, not the card's {ADAM_DEFAULTS}")
    if not 0 <= spec["snapshot_update"] < spec["prefix_updates"] <= spec["steps"] or \
            [p["name"] for p in spec["plan"]] != ["A", "B", "C"] or \
            any(s >= spec["prefix_updates"] for p in spec["plan"] for s in p["attribution"]):
        raise Void("plan", "the pass plan or the prefix is not the card's shape")
    rec["recipe"], rec["optimizer"] = dataclasses.asdict(recipe), expected_adam
    rec["device"], rec["mlx"] = str(mx.default_device()), getattr(mx, "__version__", None)

    reference = R.read_reference(spec, seams.read_reference(), recipe)
    replay1 = read_replay1(spec, seams.read_replay1_trace())
    training, identities, rec["corpus"] = R.read_training(spec, seams.read_corpus_json())
    log("inputs_verified", training_records=len(training))

    sha1 = seams.checkpoint_sha1()
    if sha1 != M.CHECKPOINT_SHA1:
        raise Void("checkpoint", f"SHA-1 {sha1} is not calib020_0001's")
    rec["checkpoint"] = {"path": M.CHECKPOINT, "sha1": sha1}
    teacher = seams.load_network()
    teacher.eval()
    rec["modes_at_load"] = {"teacher": M._require_eval(teacher, "the teacher after load")}
    if D.state_digest(teacher) != spec["teacher_digest"]:
        raise Void("networks", "the loaded teacher is not the pinned incumbent")
    rec["modules"] = seams.loaded_modules()
    _, train_values = D.forward(teacher, training)              # once, as Stage 3 computed the anchor targets
    from . import training_replay_capture as C
    rec["teacher_values_digest"] = C.model_state_digest({"teacher_values": _np(train_values.astype("float32"))})
    log("teacher_values")

    sink = _Sink(out_dir, reference, replay1, expected_adam)
    rec["passes"] = {}
    try:
        for plan in spec["plan"]:
            t0 = time.monotonic()
            student = seams.load_network()
            student.eval()
            if student is teacher:
                raise Void("networks", "the student and the teacher are one object")
            log("pass_start", name=plan["name"], attribution=plan["attribution"])
            out = prefix(student, teacher, training, recipe, train_values, plan, spec, sink)
            before = next(d for d in sink.digest_list if d["pass"] == plan["name"] and d["point"] == "before_update")
            sink.measurement(plan["name"], measure(seams, out["snapshot"], before, training, train_values,
                                                   before["batch"], recipe, spec))
            rec["passes"][plan["name"]] = {"seconds": round(time.monotonic() - t0, 3),
                                           "update_21": sink.updates[plan["name"]][spec["snapshot_update"]]}
            log("pass_end", name=plan["name"], grad_norm_21=rec["passes"][plan["name"]]["update_21"]["grad_norm"])
            del student, out
    finally:
        sink.close()
        rec["files"] = {"updates": "updates.jsonl", "digests": "digests.jsonl", "written": list(sink.files)}
    measurements = {}
    for plan in spec["plan"]:
        m = json.loads((out_dir / "measurements" / f"pass_{plan['name']}.json").read_text())
        m.pop("pass")
        measurements[plan["name"]] = m
    rec["comparison"] = compare(sink.updates, reference, replay1, sink.digest_list, measurements, spec)
    rec["teacher_digest_after"] = D.state_digest(teacher)
    log("compared")


def conclude(spec: Mapping[str, Any], seams: Seams, out_dir: pathlib.Path,
             head: str | None = None) -> Tuple[int, Dict[str, Any]]:
    """THE RECORD BOUNDARY (as Stage 3's): record.json on every exit; SIGTERM -> VOID at step "terminated"."""
    out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "card": CARD, "card_sha256": CARD_SHA256, "spec": dict(spec),
                           "head": head, "status": None, "written_by": "script", "diagnostic_replay": True,
                           "student_saved": False, "held_out_records_read": 0, "causal_claims": None,
                           "strength_verdict": None,
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
                 read_replay1_trace=lambda: (ROOT / REPLAY1_TRACE).read_bytes(),
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
    code, rec = conclude(FROZEN, _real_seams(ckpt), out, head=M._git("rev-parse", "HEAD"))
    print(json.dumps({"status": rec["status"], "void": rec.get("void"), "head": rec["head"],
                      "update_21": {k: v["update_21"] for k, v in rec.get("passes", {}).items()}}))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3 or argv[0] != "run" or argv[1] != "--checkpoint":
        print("usage: python -m scripts.GPU.alphazero.search_distillation_d2_gradrepeat run --checkpoint <abs path>",
              file=sys.stderr)
        return EXIT_USAGE
    try:
        return run(argv[2])
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
