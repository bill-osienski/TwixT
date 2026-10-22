"""SEARCH DISTILLATION D2 -- the offline whole-policy trainer (Stage 0). Nothing runs on import; no gate.

Mechanism: a STUDENT copy of the incumbent is trained to match the incumbent's own captured search targets
(each row's `child_visits_after`, the very counts play_game put in its learner PositionRecord). The update surface
is the encoder and the policy head -- trainer.MainModule, the same wrapper the ordinary trainer optimizes. The value
head, the value adapter (when present) and every BatchNorm running statistic are never updated; training runs in
evaluation mode throughout, so normalization uses the frozen running statistics exactly as inference does.

Because the value head reads the encoder, changing the encoder can move the value output. The loss therefore adds a
value ANCHOR: the mean squared difference between the student's value and the TEACHER's (the unchanged incumbent's)
value on the same positions. The anchor is a safeguard, not a guarantee; `evaluate` reports the ABSOLUTE value
differences (mean, 95th percentile, maximum) so that cancellation cannot hide drift.

    loss = mean policy cross-entropy to the normalized counts (trainer.make_padded_batch + compute_masked_log_probs)
         + anchor_weight * mean((v_student - v_teacher)^2)
         + l2_weight * sum of squares over the update surface's trainable parameters

Reading: `load_records` reopens a FINALIZED phase through training_replay_capture.load_phase (strict) and rebuilds
each buffer member's board tensor by replaying the moves through the rules engine. No search, no model inference.
The recipe (steps, batch, learning rate, weights, clip, seed) is the caller's: the plan freezes it before any run.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from . import training_replay_capture as C
from .game.twixt_state import TwixtState
from .self_play import PositionRecord
from .trainer import MainModule, clip_grad_norm, compute_masked_log_probs, make_padded_batch

BOARD, CAP_PLIES, MAX_MOVES = C.BOARD, C.CAP_PLIES, 576
SURFACE_PREFIXES = ("encoder.", "policy_head.")
FROZEN_SUFFIXES = ("running_mean", "running_var")


class DistillError(RuntimeError):
    """The inputs, the networks or a check after training are not what the mechanism requires."""


@dataclasses.dataclass(frozen=True)
class Recipe:
    steps: int
    batch_size: int
    learning_rate: float
    anchor_weight: float
    l2_weight: float
    clip_norm: float
    seed: int

    def __post_init__(self):
        if self.steps < 1 or self.batch_size < 1 or not self.learning_rate > 0 or self.clip_norm <= 0:
            raise DistillError(f"recipe out of range: {self}")
        if self.anchor_weight < 0 or self.l2_weight < 0:
            raise DistillError(f"recipe weights must be non-negative: {self}")


# ─────────────────────────────── reading ───────────────────────────────

def records_from_envelopes(envelopes: Sequence[Mapping[str, Any]]) -> Tuple[List[PositionRecord], List[str]]:
    """Every buffer member of every envelope as a PositionRecord (board rebuilt by replay; target = the row's
    child_visits_after; outcome = z), with its row id. Game order, then ply order."""
    records, ids = [], []
    for env in envelopes:
        members = {m["source_row_id"] for m in env["buffer_membership"]}
        s = TwixtState(active_size=BOARD, to_move=env["start_player"], max_plies_limit=CAP_PLIES)
        for i, (row, mv) in enumerate(zip(env["rows"], env["move_history"])):
            legal = [tuple(m) for m in s.legal_moves()]
            if [tuple(m) for m in row["legal_moves"]] != legal or row["ply"] != i or row["to_move"] != s.to_move:
                raise DistillError(f"row {row.get('row_id')}: the replay does not reproduce its position")
            if row["row_id"] in members:
                records.append(PositionRecord(
                    board_tensor=np.transpose(s.to_tensor(), (1, 2, 0)), to_move=s.to_move, legal_moves=legal,
                    visit_counts=list(row["child_visits_after"]), outcome=float(row["z"]), active_size=BOARD,
                    ply=i, game_n_moves=len(env["rows"])))
                ids.append(row["row_id"])
            s = s.apply_move(tuple(mv))
    if not records:
        raise DistillError("no buffer member to train on")
    return records, ids


def load_records(capture_dir: str) -> Tuple[dict, List[PositionRecord], List[str]]:
    ctx, envelopes = C.load_phase(capture_dir)
    records, ids = records_from_envelopes(envelopes)
    return ctx, records, ids


# ─────────────────────────────── networks ───────────────────────────────

def state_digest(network) -> str:
    """named-tensors/1 over parameters() -- every weight and every BatchNorm running statistic."""
    return C.model_state_digest(snapshot(network))


def snapshot(network) -> Dict[str, np.ndarray]:
    return {k: np.array(v) for k, v in tree_flatten(network.parameters())}


def surface_names(network) -> List[str]:
    """The trainable update surface: encoder and policy-head parameters, BatchNorm running statistics excluded."""
    return sorted(k for k, _ in tree_flatten(MainModule(network.encoder, network.policy_head).trainable_parameters()))


def _require_eval(network, who: str) -> None:
    training = [n or "<root>" for n, m in network.named_modules() if m.training]
    if training:
        raise DistillError(f"{who} has modules in training mode: {training[:5]}")


def check_after(before: Mapping[str, np.ndarray], after: Mapping[str, np.ndarray], surface: Sequence[str]) -> Dict:
    """Off the surface, every tensor bitwise unchanged; on it, at least one tensor changed."""
    if set(before) != set(after):
        raise DistillError("the parameter names changed")
    surf = {f"{p}" for p in surface}
    if any(n.endswith(FROZEN_SUFFIXES) for n in surf) or any(not n.startswith(SURFACE_PREFIXES) for n in surf):
        raise DistillError("the surface includes a running statistic or a tensor outside encoder/policy_head")
    moved_off = sorted(n for n in before if n not in surf and not np.array_equal(before[n], after[n]))
    if moved_off:
        raise DistillError(f"tensors off the update surface changed: {moved_off[:5]}")
    moved_on = sorted(n for n in surf if not np.array_equal(before[n], after[n]))
    if not moved_on:
        raise DistillError("no tensor on the update surface changed")
    return {"surface_tensors": len(surf), "surface_tensors_changed": len(moved_on),
            "frozen_tensors_unchanged": len(before) - len(surf)}


# ─────────────────────────────── batches and losses ───────────────────────────────

def _batch(records: Sequence[PositionRecord]):
    boards, rows, cols, mask, target, _ = make_padded_batch(list(records), max_moves_cap=MAX_MOVES)
    if mask.shape[1] < max(len(r.legal_moves) for r in records):
        raise DistillError("the padded batch truncated a legal-move list")
    return boards, rows, cols, mask, target


def forward(network, records: Sequence[PositionRecord], chunk: int = 64) -> Tuple[List[np.ndarray], np.ndarray]:
    """Per-record legal-move log-probabilities and values (evaluation mode, no gradient)."""
    logps, values = [], []
    for i in range(0, len(records), chunk):
        part = records[i:i + chunk]
        boards, rows, cols, mask, _ = _batch(part)
        logits, v, _ = network.forward_padded(boards, rows, cols, mask, active_size=BOARD)
        lp = compute_masked_log_probs(logits, mask)
        mx.eval(lp, v)
        lp = np.array(lp)
        logps += [lp[j, :len(r.legal_moves)].astype(np.float64) for j, r in enumerate(part)]
        values.append(np.array(v, dtype=np.float64))
    return logps, np.concatenate(values)


def targets_of(records: Sequence[PositionRecord]) -> List[np.ndarray]:
    out = []
    for r in records:
        c = np.asarray(r.visit_counts, dtype=np.float64)
        if c.sum() <= 0:
            raise DistillError("a target with zero mass")
        out.append(c / c.sum())
    return out


def evaluate(network, records: Sequence[PositionRecord], teacher_values: np.ndarray) -> Dict[str, float]:
    """Policy cross-entropy and KL to the targets, and ABSOLUTE value differences from the teacher."""
    logps, values = forward(network, records)
    tg = targets_of(records)
    ce = [-float(np.dot(t, lp)) for t, lp in zip(tg, logps)]
    ent = [-float(np.dot(t[t > 0], np.log(t[t > 0]))) for t in tg]
    d = np.abs(values - np.asarray(teacher_values, dtype=np.float64))
    return {"n": len(records), "policy_ce": float(np.mean(ce)), "policy_kl": float(np.mean(np.subtract(ce, ent))),
            "value_abs_mean": float(d.mean()), "value_abs_p95": float(np.quantile(d, 0.95)),
            "value_abs_max": float(d.max())}


def train_student(student, teacher, records: Sequence[PositionRecord], recipe: Recipe,
                  teacher_values: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Train `student` in place on `records`. The teacher is read, never updated; both stay in evaluation mode."""
    if student is teacher:
        raise DistillError("the student and the teacher are the same object")
    for net, who in ((student, "the student"), (teacher, "the teacher")):
        _require_eval(net, who)
    if not records:
        raise DistillError("no records")
    teacher_digest = state_digest(teacher)
    if state_digest(student) != teacher_digest:
        raise DistillError("the student does not start as an exact copy of the teacher")
    if teacher_values is None:
        teacher_values = forward(teacher, records)[1]
    teacher_values = np.asarray(teacher_values, dtype=np.float32)
    if teacher_values.shape != (len(records),):
        raise DistillError("teacher values do not match the records")

    before = snapshot(student)
    surface = MainModule(student.encoder, student.policy_head)
    names = surface_names(student)
    opt = optim.Adam(learning_rate=recipe.learning_rate)

    def loss_fn(_surface, boards, rows, cols, mask, target, tv):
        logits, values, _ = student.forward_padded(boards, rows, cols, mask, active_size=BOARD)
        policy = mx.mean(-mx.sum(target * compute_masked_log_probs(logits, mask), axis=1))
        anchor = mx.mean((values - tv) ** 2)
        l2 = sum(mx.sum(p ** 2) for _, p in tree_flatten(_surface.trainable_parameters()))
        return policy + recipe.anchor_weight * anchor + recipe.l2_weight * l2, policy, anchor

    grad_fn = nn.value_and_grad(surface, loss_fn)
    rng = np.random.default_rng(recipe.seed)
    order: List[int] = []
    log = []
    for step in range(recipe.steps):
        if len(order) < recipe.batch_size:
            order += rng.permutation(len(records)).tolist()
        idx, order = order[:recipe.batch_size], order[recipe.batch_size:]
        boards, rows, cols, mask, target = _batch([records[i] for i in idx])
        (loss, policy, anchor), grads = grad_fn(surface, boards, rows, cols, mask, target,
                                                   mx.array(teacher_values[idx]))
        grads, norm = clip_grad_norm(grads, recipe.clip_norm)
        opt.update(surface, grads)
        mx.eval(surface.parameters(), opt.state, loss, policy, anchor, norm)
        values = (float(loss), float(policy), float(anchor), float(norm))
        if not all(np.isfinite(values)):
            raise DistillError(f"step {step}: a non-finite loss or gradient norm {values}")
        log.append({"step": step, "loss": values[0], "policy": values[1], "anchor": values[2], "grad_norm": values[3]})

    _require_eval(student, "the student after training")
    if state_digest(teacher) != teacher_digest:
        raise DistillError("the teacher changed during training")
    checks = check_after(before, snapshot(student), names)
    return {"recipe": dataclasses.asdict(recipe), "steps": log, "checks": checks,
            "teacher_digest": teacher_digest, "student_digest": state_digest(student)}


# ─────────────────────────────── save and reload ───────────────────────────────

def save_student(network, path: str) -> Dict[str, str]:
    """Create-only safetensors of every parameter, BatchNorm running statistics included."""
    p = pathlib.Path(path)
    if p.suffix != ".safetensors" or os.path.lexists(p):
        raise DistillError(f"{p} exists or is not a .safetensors path")
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp.safetensors")
    network.save_weights(str(tmp))
    try:
        os.link(tmp, p)                                     # no-replace publication
    finally:
        os.unlink(tmp)
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "state_digest": state_digest(network)}


def reload(path: str, make_network) -> Any:
    """A fresh network from `make_network()`, loaded strictly from `path`, in evaluation mode."""
    net = make_network()
    net.load_weights(str(path), strict=True)
    net.eval()
    return net
