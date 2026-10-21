"""POLICY-HEAD CAPACITY PILOT -- PLANTED-TARGET QUALIFICATION of the fitting recipe. BUILT; NOT RUN.

Card: `docs/superpowers/2026-10-21-policy-head-capacity-planted-qualification-card.md` (rev 2, 28abdbfe), pinned
below by sha256. Nothing runs on import. `run` refuses unless
`policy_head_capacity_planted_authorization.POLICY_HEAD_CAPACITY_PLANTED_AUTHORIZED` is open, read at call time.

WHAT THE RUN DOES, in the card's order (every §8 failure is a VOID; nothing is retried, selected or extended):
  §3 replays the two closed screens' committed games (VOID on any n_legal, legality or ply-order mismatch), splits
     by game (game_idx % 5 == 4 is evaluation), dedupes by Stage B's state digest, drops evaluation digests that
     occur in training and draws 1,600 + 400 positions with seed 11;
  §4 verifies calib020_0001 by SHA-1, loads it in eval mode (0 modules training), computes encoder features on the
     CPU, caches them OUTSIDE the repository (only the digest is recorded) and plants TC (seed 12) and TR (seed 13)
     at a 1.0-nat within-position scale measured on the training features;
  §5 fits arms C and R to each target -- 400 Adam steps, batch 32, lr 1e-3, global-norm clip 1.0 as
     external-learning stage 2, draw order seed 14 -- and a temperature on TR, on training positions only;
  §6 scores every fit on the evaluation positions only and applies F1, F2, S1, S2, S3;
  §7 records the descriptive numbers. Every outcome is OPERATIONAL QUALIFICATION ONLY.
The verdict is computed inside the handler that writes record.json, so every exit leaves a record.

    python -m scripts.GPU.alphazero.policy_head_capacity_planted run --checkpoint <absolute path to calib020_0001>
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import math
import os
import pathlib
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from . import policy_head_capacity as PHC
from . import policy_head_capacity_planted_authorization as GATE
from .game.twixt_state import TwixtState
from .network import PolicyHead
from .search_distillation_stage_b import state_digest     # card §3: Stage B's frozen digest, the object itself

ROOT = pathlib.Path(__file__).resolve().parents[3]
RECORD_KIND = "policy-head-capacity-planted-record/1"
CARD = "docs/superpowers/2026-10-21-policy-head-capacity-planted-qualification-card.md"
CARD_SHA256 = "41554c31fa278da4a16ca25bb873f81910b3f3cff095cab2d42641217e9d37c0"
CHECKPOINT = "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"
CHECKPOINT_SHA1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"
SOURCES = ("inference-mode-match", "mirror-averaging-match")         # card §3, in this order
SOURCE_ROOT = "docs/superpowers/evidence"
EVIDENCE_DIR = "docs/superpowers/evidence/policy-head-capacity-planted"
CACHE_DIR = pathlib.Path.home() / "twixt-cache" / "policy-head-capacity-planted"   # card §4: outside the repo

BOARD, PLY_CAP, M_PAD, HIDDEN, TR_HIDDEN = 24, 280, 528, 128, 64
FROZEN = {"n_train": 1600, "n_eval": 400, "steps": 400, "batch": 32,                 # §2, §5
          "games": 64, "moves": 3293, "eval_games": 12, "eval_moves": 611}          # §3, counted from the records
SEEDS = {"selection": 11, "tc": 12, "tr": 13, "order": 14}                          # §2: small, outside the registry
LEARNING_RATE, CLIP_NORM, WEIGHT_DECAY = 1e-3, 1.0, 0.0                              # as external_learning_stage2
ADAM = {"betas": [0.9, 0.999], "eps": 1e-8, "bias_correction": False}
SCALE, SCALE_TOL = 1.0, 1e-3
BISECT_BOUNDS, BISECT_ITERS = (0.0, 10.0), 60
GOLDEN_BOUNDS, GOLDEN_ITERS = (-3.0, 3.0), 60
DELTA, FIT_FRACTION = 0.05, 0.25
CAP_S, MARGIN_S = 900.0, 30.0              # the wrapper's alarm is the cap; the script stops 30 s short to record
TC_ARRAYS = tuple(n.split(".", 1)[1] for n in PHC.BASE)       # the existing head's eight arrays, sorted
FITS = (("C", "TC"), ("R", "TC"), ("C", "TR"), ("R", "TR"))
EXIT_DONE, EXIT_VOID, EXIT_REFUSED, EXIT_USAGE = 0, 3, 4, 2


class Void(Exception):
    """A card §8 integrity failure. Names the step; never retried."""

    def __init__(self, step: str, detail: str):
        super().__init__(f"[{step}] {detail}")
        self.step, self.detail = step, detail


class Refused(Exception):
    """The run never started: the gate is closed, the card is not the pinned bytes, the evidence exists or the
    checkpoint argument is not an existing absolute path."""


def _require_gate() -> None:
    if GATE.POLICY_HEAD_CAPACITY_PLANTED_AUTHORIZED is not True:
        raise Refused("POLICY_HEAD_CAPACITY_PLANTED_AUTHORIZED is closed: the qualification is not authorized")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _jsha(obj: Any) -> str:
    return _sha(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def _sha_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json_new(path: pathlib.Path, obj: Any) -> str:
    """Create-only, flushed and fsynced."""
    data = (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()
    with open(path, "xb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    return _sha(data)


# ─────────────────────────────── §3 positions ───────────────────────────────

def is_eval_game(game_idx: int) -> bool:
    return game_idx % 5 == 4


def replay(game: Mapping[str, Any]) -> List[Tuple[int, TwixtState]]:
    """The states BEFORE each recorded move. Each must be 24x24, reproduce the record's n_legal, and allow the
    recorded move; the moves must be in ply order."""
    if game.get("board_size") != BOARD:
        raise Void("replay", f"game {game.get('game_idx')} has board_size {game.get('board_size')}, not {BOARD}")
    s = TwixtState(active_size=BOARD, to_move="red", max_plies_limit=PLY_CAP)
    out = []
    for i, m in enumerate(game["moves"]):
        where = f"game {game['game_idx']} ply {i}"
        if m["ply"] != i:
            raise Void("replay", f"{where}: the record says ply {m['ply']}; moves are not in ply order")
        if (s.board_size, s.active_size) != (BOARD, BOARD):
            raise Void("replay", f"{where}: the state is not {BOARD}x{BOARD}")
        legal = s.legal_moves()
        if len(legal) != m["n_legal"]:
            raise Void("replay", f"{where}: {len(legal)} legal moves, the record says {m['n_legal']}")
        move = (int(m["row"]), int(m["col"]))
        if move not in set(legal):
            raise Void("replay", f"{where}: the recorded move {move} is not legal")
        out.append((i, s))
        s = s.apply_move(move)
    return out


def check_counts(games: Sequence[Tuple[str, Mapping[str, Any]]], spec: Mapping[str, int]) -> Dict[str, int]:
    held = [g for _, g in games if is_eval_game(g["game_idx"])]
    got = {"games": len(games), "moves": sum(len(g["moves"]) for _, g in games),
           "eval_games": len(held), "eval_moves": sum(len(g["moves"]) for g in held)}
    want = {k: spec[k] for k in got}
    if got != want:
        raise Void("counts", f"the sources hold {got}, not the frozen {want}")
    return got


def pools(games: Sequence[Tuple[str, Mapping[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Unique positions per split, first occurrence in (screen, game_idx, ply) order -- the order `games` is in.
    An evaluation digest that also occurs in training leaves the evaluation pool and stays in training."""
    seen: Dict[bool, Dict[str, Dict[str, Any]]] = {False: {}, True: {}}
    for screen, g in games:
        held = is_eval_game(g["game_idx"])
        for ply, s in replay(g):
            d = state_digest(s)
            if d not in seen[held]:
                seen[held][d] = {"digest": d, "screen": screen, "game_idx": g["game_idx"], "ply": ply,
                                 "to_move": s.to_move, "n_legal": len(s.legal_moves()), "state": s}
    train = list(seen[False].values())
    held_out = [p for d, p in seen[True].items() if d not in seen[False]]
    return train, held_out, len(seen[True]) - len(held_out)


def select(train_pool: Sequence[Dict[str, Any]], eval_pool: Sequence[Dict[str, Any]], spec: Mapping[str, int],
           seed: int = SEEDS["selection"]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if len(train_pool) < spec["n_train"] or len(eval_pool) < spec["n_eval"]:
        raise Void("selection", f"{len(train_pool)} training / {len(eval_pool)} evaluation unique positions, "
                                f"fewer than {spec['n_train']} / {spec['n_eval']}")
    rng = np.random.default_rng(seed)
    ti = rng.choice(len(train_pool), size=spec["n_train"], replace=False)
    ei = rng.choice(len(eval_pool), size=spec["n_eval"], replace=False)
    return [train_pool[int(i)] for i in ti], [eval_pool[int(i)] for i in ei]


def draw_order(n: int, steps: int, batch: int, seed: int = SEEDS["order"]) -> np.ndarray:
    """Card §5: whole permutations of the training indices, each cut into batches, in order."""
    if n % batch or (steps * batch) % n:
        raise ValueError(f"{steps} x {batch} is not a whole number of passes over {n}")
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(n) for _ in range(steps * batch // n)]).reshape(steps, batch)


def padded(states: Sequence[TwixtState]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols, mask = (np.zeros((len(states), M_PAD), np.int32) for _ in range(3))
    for i, s in enumerate(states):
        moves = s.legal_moves()
        if len(moves) > M_PAD:
            raise Void("positions", f"{len(moves)} legal moves exceed the {M_PAD} padded slots")
        rows[i, :len(moves)] = [m[0] for m in moves]
        cols[i, :len(moves)] = [m[1] for m in moves]
        mask[i, :len(moves)] = 1
    return rows, cols, mask.astype(np.float32)


def boards(states: Sequence[TwixtState]) -> np.ndarray:
    return np.stack([s.to_tensor().transpose(1, 2, 0) for s in states]).astype(np.float32)


# ─────────────────────────────── §4 logits, scale and the planted teachers ───────────────────────────────

def head_logits(head, x: Mapping[str, np.ndarray], chunk: int = 100) -> np.ndarray:
    """The head's padded logits (float32) on cached features."""
    out = []
    for i in range(0, len(x["feats"]), chunk):
        z = head.forward_padded(*(mx.array(x[k][i:i + chunk]) for k in ("feats", "rows", "cols", "mask")))
        mx.eval(z)
        out.append(np.asarray(z))
    return np.concatenate(out)


def centred_scale(added: np.ndarray, mask: np.ndarray) -> float:
    """Card §4: subtract each position's mean added logit over its legal moves, then the SD over ALL legal moves."""
    legal = mask > 0.5
    a = np.asarray(added, np.float64)
    mean = np.where(legal, a, 0.0).sum(1) / legal.sum(1)
    c = (a - mean[:, None])[legal]
    return float(np.sqrt(np.mean(c * c)))


def tc_noise(head, seed: int = SEEDS["tc"]) -> Dict[str, np.ndarray]:
    params = dict(tree_flatten(head.parameters()))
    rng = np.random.default_rng(seed)
    return {k: rng.standard_normal(params[k].shape) for k in TC_ARRAYS}


def perturbed_head(head, noise: Mapping[str, np.ndarray], alpha: float) -> PolicyHead:
    """TC: w_k + alpha * sigma_k * eps_k on the eight arrays (sigma_k the population SD of incumbent array k);
    BatchNorm running statistics copied unchanged."""
    params = dict(tree_flatten(head.parameters()))
    new = PolicyHead()
    new.update(head.parameters())
    upd = []
    for k in TC_ARRAYS:
        w = np.asarray(params[k], np.float64)
        upd.append((k, mx.array((w + alpha * w.std() * noise[k]).astype(np.float32))))
    new.update(tree_unflatten(upd))
    new.eval()
    return new


def bisect_alpha(scale_of: Callable[[float], float]) -> Tuple[float, float]:
    lo, hi = BISECT_BOUNDS
    if not scale_of(hi) >= SCALE:
        raise Void("tc", f"the within-position scale at alpha = {hi} does not reach {SCALE}")
    for _ in range(BISECT_ITERS):
        mid = (lo + hi) / 2
        if scale_of(mid) < SCALE:
            lo = mid
        else:
            hi = mid
    alpha = (lo + hi) / 2
    got = scale_of(alpha)
    if not abs(got - SCALE) <= SCALE_TOL:
        raise Void("tc", f"bisection ended at scale {got}, not {SCALE} +/- {SCALE_TOL}")
    return alpha, got


def tr_teacher(seed: int = SEEDS["tr"]) -> Tuple[np.ndarray, np.ndarray]:
    """g's weights: He-normal, zero biases, rounded to float32 so that arm R can hold them exactly."""
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((TR_HIDDEN, HIDDEN)) * math.sqrt(2.0 / HIDDEN)
    w2 = rng.standard_normal((1, TR_HIDDEN)) * math.sqrt(2.0 / TR_HIDDEN)
    return w1.astype(np.float32).astype(np.float64), w2.astype(np.float32).astype(np.float64)


def g_values(x: Mapping[str, np.ndarray], w1: np.ndarray, w2: np.ndarray, chunk: int = 100) -> np.ndarray:
    """g(features at the move), float64, 0 on padding."""
    out = []
    for i in range(0, len(x["feats"]), chunk):
        f, r, c = x["feats"][i:i + chunk], x["rows"][i:i + chunk], x["cols"][i:i + chunk]
        cells = f[np.arange(len(f))[:, None], r, c, :].astype(np.float64)
        out.append((np.maximum(cells @ w1.T, 0.0) @ w2.T)[..., 0])
    return np.where(x["mask"] > 0.5, np.concatenate(out), 0.0)


def targets(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Softmax over legal moves in float64, stored as float32; 0 on padding."""
    legal = mask > 0.5
    z = np.where(legal, np.asarray(logits, np.float64), -np.inf)
    p = np.where(legal, np.exp(z - z.max(1, keepdims=True)), 0.0)
    return (p / p.sum(1, keepdims=True)).astype(np.float32)


def _log_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    legal = mask > 0.5
    z = np.where(legal, np.asarray(logits, np.float64), -np.inf)
    m = z.max(1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(1, keepdims=True))


def mean_kl(p: np.ndarray, logits: np.ndarray, mask: np.ndarray) -> float:
    """Mean over positions of KL(p || softmax(logits)) over legal moves, nats, float64."""
    p64, lq = np.asarray(p, np.float64), _log_softmax(logits, mask)
    pos = p64 > 0
    terms = np.where(pos, p64 * (np.log(np.where(pos, p64, 1.0)) - np.where(pos, lq, 0.0)), 0.0)
    return float(terms.sum(1).mean())


def mean_ce(p: np.ndarray, logits: np.ndarray, mask: np.ndarray) -> float:
    p64, lq = np.asarray(p, np.float64), _log_softmax(logits, mask)
    return float(-np.where(p64 > 0, p64 * np.where(p64 > 0, lq, 0.0), 0.0).sum(1).mean())


def mean_entropy(p: np.ndarray) -> float:
    p64 = np.asarray(p, np.float64)
    return float(-np.where(p64 > 0, p64 * np.log(np.where(p64 > 0, p64, 1.0)), 0.0).sum(1).mean())


# ─────────────────────────────── §5 the fitting recipe ───────────────────────────────

def soft_ce(model, feats, rows, cols, mask, target):
    logits = model.policy_head.forward_padded(feats, rows, cols, mask)
    return -mx.sum(target * (logits - mx.logsumexp(logits, axis=-1, keepdims=True))) / logits.shape[0]


def snapshot(network) -> Dict[str, np.ndarray]:
    return {k: np.array(v) for k, v in tree_flatten(network.parameters())}


def check_after(before: Mapping[str, np.ndarray], after: Mapping[str, np.ndarray], surface: Sequence[str]) -> None:
    """Card §8: after a fit only the arm's surface may differ; no BatchNorm running statistic may."""
    if set(before) != set(after):
        raise Void("fit", "the network's parameter set changed")
    changed = {k for k in before if not np.array_equal(before[k], after[k])}
    if changed & {k for k in before if k.endswith(("running_mean", "running_var"))}:
        raise Void("fit", "a BatchNorm running statistic changed")
    if not changed <= set(surface):
        raise Void("fit", f"parameters outside the arm's surface changed: {sorted(changed - set(surface))}")


def _eval_mode(network, where: str) -> None:
    if network.training or network.policy_head.training:
        raise Void("fit", f"{where}: the network left eval mode")


def fit(make_network: Callable[[], Any], arm: str, train: Mapping[str, np.ndarray], held: Mapping[str, np.ndarray],
        order: np.ndarray, i_eval: np.ndarray) -> Dict[str, Any]:
    """One fit, card §5: a fresh network, arm installed, 400 clipped Adam steps in the frozen order."""
    from .trainer import clip_grad_norm          # trainer imports self_play: imported here, as stage 2 does

    network = make_network()
    surface = PHC.install(network, arm)
    if not np.array_equal(head_logits(network.policy_head, held), i_eval):
        raise Void("fit", f"arm {arm} does not start exactly at the incumbent head")
    before = snapshot(network)
    opt = optim.Adam(learning_rate=LEARNING_RATE, betas=ADAM["betas"], eps=ADAM["eps"],
                     bias_correction=ADAM["bias_correction"])
    step_fn = nn.value_and_grad(network, soft_ce)
    steps = []
    for step, idx in enumerate(order):
        _eval_mode(network, f"arm {arm} step {step}")
        loss, grads = step_fn(network, *(mx.array(train[k][idx]) for k in ("feats", "rows", "cols", "mask", "target")))
        if sorted(k for k, _ in tree_flatten(grads)) != sorted(surface):
            raise Void("fit", f"arm {arm} step {step}: a gradient outside the arm's surface")
        head_grads, gnorm = clip_grad_norm(grads["policy_head"], max_norm=CLIP_NORM)
        mx.eval(loss, head_grads, gnorm)
        loss_f, gnorm_f = float(loss.item()), float(gnorm.item())
        if not (math.isfinite(loss_f) and math.isfinite(gnorm_f)):
            raise Void("fit", f"arm {arm} step {step}: nonfinite loss or gradient norm")
        opt.update(network.policy_head, head_grads)
        mx.eval(network.policy_head.parameters(), opt.state)
        steps.append({"loss": loss_f, "grad_norm": gnorm_f})
    _eval_mode(network, f"arm {arm} after training")
    check_after(before, snapshot(network), surface)
    return {"surface": surface, "eval_logits": head_logits(network.policy_head, held),
            "train_ce": mean_ce(train["target"], head_logits(network.policy_head, train), train["mask"]),
            "steps": steps}


def fit_temperature(i_train: np.ndarray, p_train: np.ndarray, mask: np.ndarray) -> float:
    """Card §5: tau = e^u, golden-section on u over the TRAINING positions only; no gradient steps."""
    i64 = np.asarray(i_train, np.float64)
    ce = lambda u: mean_ce(p_train, i64 / math.exp(u), mask)      # noqa: E731
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = GOLDEN_BOUNDS
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = ce(c), ce(d)
    for _ in range(GOLDEN_ITERS):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = ce(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = ce(d)
    return math.exp((a + b) / 2.0)


# ─────────────────────────────── §6 the verdict, §7 descriptive ───────────────────────────────

def verdict(kl: Mapping[str, float], kl0: Mapping[str, float]) -> Tuple[Dict[str, bool], str]:
    if not all(math.isfinite(v) for v in [*kl.values(), *kl0.values()]):
        raise Void("verdict", "a nonfinite KL")
    rules = {"F1": kl["C/TC"] <= FIT_FRACTION * kl0["TC"] and kl["R/TC"] <= FIT_FRACTION * kl0["TC"],
             "F2": kl["R/TR"] <= FIT_FRACTION * kl0["TR"],
             "S1": kl["C/TR"] - kl["R/TR"] >= DELTA,
             "S2": kl["T/TR"] - kl["R/TR"] >= DELTA,
             "S3": kl["C/TC"] - kl["R/TC"] < DELTA}
    return rules, ("QUALIFIED" if all(rules.values()) else "NOT_QUALIFIED")


def top2_fraction(x: Mapping[str, np.ndarray], chunk: int = 100) -> float:
    """Share of legal-cell feature variance in the top two principal components (float64 covariance)."""
    n, s, ss = 0, np.zeros(HIDDEN), np.zeros((HIDDEN, HIDDEN))
    for i in range(0, len(x["feats"]), chunk):
        f, r, c, m = (x[k][i:i + chunk] for k in ("feats", "rows", "cols", "mask"))
        cells = f[np.arange(len(f))[:, None], r, c, :][m > 0.5].astype(np.float64)
        n, s, ss = n + len(cells), s + cells.sum(0), ss + cells.T @ cells
    mean = s / n
    ev = np.sort(np.linalg.eigvalsh(ss / n - np.outer(mean, mean)))[::-1]
    return float((ev[0] + ev[1]) / ev.sum())


# ─────────────────────────────── the cache, the seams and the record boundary ───────────────────────────────

def write_cache(cache_dir: pathlib.Path, run_id: str, arrays: Mapping[str, np.ndarray]) -> Tuple[pathlib.Path, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"features-{run_id}.npz"
    with open(path, "xb") as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    return path, _sha_file(path)


def verify_cache(path: pathlib.Path, digest: str) -> None:
    if _sha_file(path) != digest:
        raise Void("cache", f"{path} no longer has its recorded digest")


@dataclasses.dataclass(frozen=True)
class Seams:
    """What the run reads from outside this module. TEST seams; `run` passes `_real_seams` only."""
    load_games: Callable[[], List[Tuple[str, Dict[str, Any]]]]
    checkpoint_sha1: Callable[[], str]
    load_network: Callable[[], Tuple[Any, int]]
    features: Callable[[Any, Sequence[TwixtState]], np.ndarray]


def _split(chosen: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    states = [p["state"] for p in chosen]
    rows, cols, mask = padded(states)
    return {"states": states, "rows": rows, "cols": cols, "mask": mask}


def _position(p: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: p[k] for k in ("digest", "screen", "game_idx", "ply", "to_move", "n_legal")}


def _execute(spec: Mapping[str, int], seams: Seams, out_dir: pathlib.Path, cache_dir: pathlib.Path,
             clock: Callable[[], float], rec: Dict[str, Any]) -> None:
    t0 = clock()
    phases = rec["phases_s"] = {}

    def phase(name: str) -> None:
        elapsed = clock() - t0
        if elapsed > CAP_S - MARGIN_S:
            raise Void("cap", f"{elapsed:.1f} s elapsed before {name}: the 15-minute cap")
        phases[name] = round(elapsed, 3)

    mx.set_default_device(mx.cpu)
    if mx.default_device().type != mx.cpu:
        raise Void("device", f"the default device is {mx.default_device()}, not the CPU")

    phase("positions")
    games = seams.load_games()
    rec["counts"] = check_counts(games, spec)
    train_pool, eval_pool, leaked = pools(games)
    rec["pools"] = {"train": len(train_pool), "eval": len(eval_pool), "eval_leaked_into_train": leaked}
    chosen_t, chosen_e = select(train_pool, eval_pool, spec)
    rec["positions_sha256"] = _write_json_new(out_dir / "positions.json",
                                              {"train": [_position(p) for p in chosen_t],
                                               "eval": [_position(p) for p in chosen_e]})

    phase("network")
    sha1 = seams.checkpoint_sha1()
    if sha1 != CHECKPOINT_SHA1:
        raise Void("checkpoint", f"SHA-1 {sha1} is not calib020_0001's {CHECKPOINT_SHA1}")
    network, in_channels = seams.load_network()
    network.eval()
    training = sum(bool(m.training) for m in network.modules())
    channels = TwixtState(active_size=BOARD, to_move="red", max_plies_limit=PLY_CAP).to_tensor().shape[0]
    rec["network"] = {"sha1": sha1, "in_channels": in_channels, "tensor_channels": channels,
                      "modules_training": training}
    if training:
        raise Void("network", f"{training} modules are in training mode after eval()")
    if in_channels != channels:
        raise Void("network", f"the loader's {in_channels} input channels are not the tensor's {channels}")

    phase("features")
    T, E = _split(chosen_t), _split(chosen_e)
    for x in (T, E):
        x["feats"] = np.asarray(seams.features(network, x["states"]))
        if x["feats"].shape != (len(x["states"]), BOARD, BOARD, HIDDEN) or x["feats"].dtype != np.float32:
            raise Void("features", f"features {x['feats'].shape} {x['feats'].dtype} are not "
                                   f"({len(x['states'])}, {BOARD}, {BOARD}, {HIDDEN}) float32")
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cache, digest = write_cache(cache_dir, run_id, {f"{s}_{k}": x[k] for s, x in (("train", T), ("eval", E))
                                                    for k in ("feats", "rows", "cols", "mask")})
    rec["cache"] = {"path": str(cache), "sha256": digest}

    phase("targets")
    head = network.policy_head
    i_t, i_e = head_logits(head, T), head_logits(head, E)
    noise = tc_noise(head)
    alpha, tc_scale = bisect_alpha(lambda a: centred_scale(
        head_logits(perturbed_head(head, noise, a), T).astype(np.float64) - i_t, T["mask"]))
    tc_head = perturbed_head(head, noise, alpha)
    T["TC"], E["TC"] = targets(head_logits(tc_head, T), T["mask"]), targets(head_logits(tc_head, E), E["mask"])
    w1, w2 = tr_teacher()
    g_t, g_e = g_values(T, w1, w2), g_values(E, w1, w2)
    g_scale = centred_scale(g_t, T["mask"])
    if not (math.isfinite(g_scale) and g_scale > 0):
        raise Void("tr", f"g's within-position scale is {g_scale}")
    s = SCALE / g_scale
    T["TR"] = targets(i_t.astype(np.float64) + s * g_t, T["mask"])
    E["TR"] = targets(i_e.astype(np.float64) + s * g_e, E["mask"])
    rec["teachers"] = {"tc_alpha": alpha, "tc_scale": tc_scale, "tr_s": s,
                       "tr_scale": centred_scale(s * g_t, T["mask"])}

    order = draw_order(spec["n_train"], spec["steps"], spec["batch"])
    rec["draw_order_sha256"] = _jsha(order.tolist())
    kl, fits = {}, {}
    for arm, target in FITS:
        key = f"{arm}/{target}"
        phase(f"fit {key}")
        out = fit(lambda: seams.load_network()[0], arm, {**T, "target": T[target]}, {**E, "target": E[target]},
                  order, i_e)
        kl[key] = mean_kl(E[target], out["eval_logits"], E["mask"])
        norms = [st["grad_norm"] for st in out["steps"]]
        fits[key] = {"surface": out["surface"], "final_train_ce": out["train_ce"], "grad_norms": norms,
                     "clipped_fraction": sum(g > CLIP_NORM for g in norms) / len(norms),
                     "losses": [st["loss"] for st in out["steps"]]}
    phase("temperature")
    tau = fit_temperature(i_t, T["TR"], T["mask"])
    kl["T/TR"] = mean_kl(E["TR"], i_e.astype(np.float64) / tau, E["mask"])
    kl0 = {t: mean_kl(E[t], i_e, E["mask"]) for t in ("TC", "TR")}
    for key in fits:
        fits[key]["starting_kl"], fits[key]["final_kl"] = kl0[key.split("/")[1]], kl[key]

    rules, outcome = verdict(kl, kl0)
    rec.update(kl=kl, kl0=kl0, rules=rules, verdict=outcome, fits=fits, temperature=tau)
    phase("descriptive")
    rec["descriptive"] = {"chance_baseline": float(np.mean(np.log(E["mask"].sum(1)))),
                          "target_entropy": {t: mean_entropy(E[t]) for t in ("TC", "TR")},
                          "top2_variance_fraction": top2_fraction(T)}
    verify_cache(cache, digest)
    phases["end"] = round(clock() - t0, 3)


def conclude(spec: Mapping[str, int], seams: Seams, out_dir: pathlib.Path, cache_dir: pathlib.Path,
             clock: Callable[[], float] = time.monotonic) -> Tuple[int, Dict[str, Any]]:
    """THE RECORD BOUNDARY: the verdict is computed inside it, and record.json is written on every exit."""
    out_dir.mkdir(parents=True, exist_ok=False)
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "card": CARD, "card_sha256": CARD_SHA256, "spec": dict(spec),
                           "seeds": dict(SEEDS), "status": None}
    code = EXIT_VOID
    try:
        _execute(spec, seams, out_dir, cache_dir, clock, rec)
        rec["status"], code = "COMPLETED", EXIT_DONE
    except Void as v:
        rec.update(status="VOID", void={"step": v.step, "detail": v.detail})
    except Exception as e:                       # card §8: any exception is a VOID, and it is recorded
        rec.update(status="VOID", void={"step": "exception", "detail": f"{type(e).__name__}: {e}"})
    except BaseException as e:
        rec.update(status="INTERRUPTED", void={"step": "interrupted", "detail": type(e).__name__})
        raise
    finally:
        _write_json_new(out_dir / "record.json", rec)
    return code, rec


def _real_seams(checkpoint: pathlib.Path) -> Seams:
    def load_games():
        out = []
        for screen in SOURCES:
            recs = [json.loads(p.read_text()) for p in (ROOT / SOURCE_ROOT / screen / "games").glob("game_*.json")]
            out += [(screen, g) for g in sorted(recs, key=lambda g: g["game_idx"])]
        return out

    def checkpoint_sha1():
        return hashlib.sha1(checkpoint.read_bytes()).hexdigest()

    def load_network():
        from .probe_eval import load_network_for_scoring
        network, in_channels, _hidden, _blocks = load_network_for_scoring(str(checkpoint), verbose=False)
        network.eval()
        return network, in_channels

    def features(network, states, chunk: int = 50):
        out = []
        for i in range(0, len(states), chunk):
            f = network.encoder(mx.array(boards(states[i:i + chunk])))
            mx.eval(f)
            out.append(np.asarray(f))
        return np.concatenate(out)

    return Seams(load_games, checkpoint_sha1, load_network, features)


def run(checkpoint: str) -> int:
    """THE PUBLIC ENTRY. No seam: the frozen spec, the real sources, the pinned card."""
    _require_gate()
    if _sha((ROOT / CARD).read_bytes()) != CARD_SHA256:
        raise Refused(f"{CARD} is not the pinned rev-2 card")
    ckpt = pathlib.Path(checkpoint)
    if not (ckpt.is_absolute() and ckpt.is_file()):
        raise Refused("--checkpoint must name an existing file by absolute path")
    out = ROOT / EVIDENCE_DIR
    if os.path.lexists(out):
        raise Refused(f"{out} exists: the run is one-shot")
    code, rec = conclude(FROZEN, _real_seams(ckpt), out, CACHE_DIR)
    print(json.dumps({"status": rec["status"], "verdict": rec.get("verdict"), "void": rec.get("void")}))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3 or argv[0] != "run" or argv[1] != "--checkpoint":
        print("usage: python -m scripts.GPU.alphazero.policy_head_capacity_planted run --checkpoint <abs path>",
              file=sys.stderr)
        return EXIT_USAGE
    try:
        return run(argv[2])
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
