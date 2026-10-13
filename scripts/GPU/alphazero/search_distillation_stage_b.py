"""SEARCH DISTILLATION STAGE B -- the pilot's pure pieces: split, position selection, state digest, leakage,
full legal support, TV/KL, gate Q, the L1-L3 verdict and seed exposure. CPU-only: no model, no search, no
file I/O, no seed literal. Card: docs/superpowers/2026-10-12-search-distillation-stage-b-card.md (a9e16fc9).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .external_learning_stage3 import PROTOCOL
from .game.twixt_state import TwixtState

N_GAMES, PER_GAME, N_HELDOUT, SIGN_K, Q2_MAX = 200, 4, 40, 26, 0.5
GAME, LABEL, REPEAT, TRAIN, N_SLOTS = 0, 200, 400, 440, 441
STATE_FIELDS = ("active_size", "board_size", "bridges", "max_plies_limit", "pegs", "ply", "to_move")
ARMS = ("student", "control", "temperature", "incumbent")


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def is_heldout(g: int) -> bool:
    return g % 5 == 4


def repeat_slot(g: int) -> int:
    if not is_heldout(g):
        raise ValueError(f"game {g} is not held out: it has no repeat seed")
    return REPEAT + g // 5


def replay_state(moves: Sequence[Sequence[int]]) -> TwixtState:
    """The state a corpus game reached after `moves`, built as play_eval_game builds it -- with its ply cap."""
    s = TwixtState(active_size=PROTOCOL["board_size"], to_move="red", max_plies_limit=PROTOCOL["max_moves"])
    for r, c in moves:
        s = s.apply_move((int(r), int(c)))
    return s


def state_digest(s: TwixtState) -> str:
    """Card §4: every constructor field. Refuses if TwixtState's field set has changed."""
    names = tuple(sorted(f.name for f in dataclasses.fields(TwixtState) if f.init))
    if names != STATE_FIELDS:
        raise RuntimeError(f"TwixtState fields {names} are not the frozen {STATE_FIELDS}: the digest would be incomplete")
    return _sha({"board_size": s.board_size, "active_size": s.active_size, "to_move": s.to_move, "ply": s.ply,
                 "max_plies_limit": s.max_plies_limit,
                 "pegs": sorted([r, c, colour] for (r, c), colour in s.pegs.items()),
                 "bridges": sorted([r1, c1, r2, c2] for (r1, c1), (r2, c2) in s.bridges)})


def select_plies(n_moves: int, label_seed: int) -> List[int]:
    """Card §4: up to PER_GAME plies from 1..n-1 (never the empty board, never the final state), sorted.
    The caller logs `select_start` for this game's label slot BEFORE calling (the draw uses L_g)."""
    if n_moves < 2:
        return []
    return sorted(random.Random(label_seed).sample(range(1, n_moves), min(PER_GAME, n_moves - 1)))


def select_game(moves: Sequence[Sequence[int]], label_seed: int) -> List[Dict[str, Any]]:
    return [{"ply": p, "digest": state_digest(replay_state(moves[:p]))} for p in select_plies(len(moves), label_seed)]


def leakage_filter(selected: Mapping[int, Sequence[Mapping[str, Any]]]) -> Tuple[Dict[int, List[Mapping[str, Any]]], int]:
    """Held-out positions whose digest occurs among TRAINING positions are dropped (they stay in training)."""
    if sorted(selected) != list(range(N_GAMES)):
        raise ValueError("the selection must cover all 200 games before leakage can be filtered")
    train = {p["digest"] for g, ps in selected.items() if not is_heldout(g) for p in ps}
    kept = {g: [p for p in ps if p["digest"] not in train] for g, ps in selected.items() if is_heldout(g)}
    return kept, sum(len(selected[g]) - len(kept[g]) for g in kept)


def legal_digest(legal: Sequence[Sequence[int]]) -> str:
    return _sha([[int(r), int(c)] for r, c in legal])


def sparse_label(visits: Mapping[Tuple[int, int], int], sims: int) -> List[List[int]]:
    """Card §5 storage: the nonzero (row, col, visits); the visits must sum to sims."""
    if sum(visits.values()) != sims:
        raise ValueError(f"visits sum to {sum(visits.values())}, not {sims}")
    return sorted([int(r), int(c), int(v)] for (r, c), v in visits.items() if v > 0)


def full_counts(state: TwixtState, sparse: Sequence[Sequence[int]], sims: int, legal_sha: str
                ) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Card §5: restore the FULL legal support (zero-visit moves included) from a sparse label."""
    legal = [tuple(m) for m in state.legal_moves()]
    if legal_digest(legal) != legal_sha:
        raise ValueError("the rebuilt legal list is not the one the label was made on")
    index = {m: i for i, m in enumerate(legal)}
    counts = [0] * len(legal)
    for r, c, v in sparse:
        i = index.get((r, c))
        if i is None or v <= 0 or counts[i]:
            raise ValueError(f"label entry {(r, c, v)} is not a single positive count on a legal move")
        counts[i] = v
    if sum(counts) != sims:
        raise ValueError(f"label visits sum to {sum(counts)}, not {sims}")
    return legal, counts


def probs(counts: Sequence[float]) -> List[float]:
    s = sum(counts)
    return [c / s for c in counts]


def softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits)
    e = [math.exp(z - m) for z in logits]
    s = sum(e)
    return [x / s for x in e]


def tv(p: Sequence[float], q: Sequence[float]) -> float:
    if len(p) != len(q):
        raise ValueError("distributions over different supports")
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def kl(t: Sequence[float], p: Sequence[float]) -> float:
    if len(t) != len(p):
        raise ValueError("distributions over different supports")
    return sum(a * (math.log(a) - math.log(b)) for a, b in zip(t, p) if a > 0)


def top1(logits: Sequence[float]) -> int:
    """Argmax; ties to the lowest index = the lowest encoded move id (legal order is row-major)."""
    return max(range(len(logits)), key=lambda i: (logits[i], -i))


def argmax_set(counts: Sequence[float]) -> set:
    m = max(counts)
    return {i for i, c in enumerate(counts) if c == m}


def q_game(positions: Sequence[Mapping[str, Sequence[int]]]) -> Optional[Tuple[float, float]]:
    """Card §6 per held-out game: (mean D, mean R); None for an empty game."""
    if not positions:
        return None
    d = [tv(probs(p["t1600"]), probs(p["t400"])) for p in positions]
    r = [tv(probs(p["t1600"]), probs(p["repeat"])) for p in positions]
    return sum(d) / len(d), sum(r) / len(r)


def game_measures(positions: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Dict[str, float]]]:
    """Card §8 per held-out game: each arm's mean KL(T1600 || softmax) and top-1 share; None for an empty game.
    Logits are over the full legal support; the temperature arm's logits arrive already scaled by beta."""
    if not positions:
        return None
    n = len(positions)
    return {"kl": {a: sum(kl(probs(p["target"]), softmax(p["logits"][a])) for p in positions) / n for a in ARMS},
            "top1": {a: sum(top1(p["logits"][a]) in argmax_set(p["target"]) for p in positions) / n for a in ARMS}}


def _forty(games: Sequence[Any]) -> None:
    if len(games) != N_HELDOUT:
        raise ValueError(f"{len(games)} held-out games, not {N_HELDOUT}")


def gate_q(games: Sequence[Optional[Tuple[float, float]]]) -> Dict[str, Any]:
    """Card §6. Q1: R < D in >= 26 of 40 (empty games and ties fail). Q2: sum R / sum D <= 0.5 over non-empty
    games; a zero (or absent) sum D fails Q2."""
    _forty(games)
    present = [g for g in games if g is not None]
    wins = sum(1 for d, r in present if r < d)
    sum_d, sum_r = sum(d for d, _ in present), sum(r for _, r in present)
    ratio = sum_r / sum_d if sum_d > 0 else None
    failed = [name for name, ok in (("Q1", wins >= SIGN_K), ("Q2", ratio is not None and ratio <= Q2_MAX)) if not ok]
    return {"q1_wins": wins, "sum_d": sum_d, "sum_r": sum_r, "ratio": ratio, "failed": failed,
            "verdict": "QUALIFIED" if not failed else "QUALIFICATION NON-PASS"}


def learning_verdict(games: Sequence[Optional[Mapping[str, Mapping[str, float]]]]) -> Dict[str, Any]:
    """Card §8. L1: KL_s < KL_control; L2: KL_s < min(KL_incumbent, KL_temperature); each in >= 26 of 40, ties and
    empty games are losses. L3: student top-1 agreement (mean over all 40; empty = 0 for every arm) strictly
    above both the control's and the incumbent's."""
    _forty(games)
    present = [g["kl"] for g in games if g is not None]
    l1 = sum(1 for k in present if k["student"] < k["control"])
    l2 = sum(1 for k in present if k["student"] < min(k["incumbent"], k["temperature"]))
    top = {a: sum(g["top1"][a] for g in games if g is not None) / N_HELDOUT for a in ARMS}
    l3 = top["student"] > top["control"] and top["student"] > top["incumbent"]
    failed = [name for name, ok in (("L1", l1 >= SIGN_K), ("L2", l2 >= SIGN_K), ("L3", l3)) if not ok]
    return {"l1_wins": l1, "l2_wins": l2, "top1": top, "failed": failed,
            "verdict": "PASS" if not failed else "NON-PASS"}


def _game(e: Mapping[str, Any]) -> int:
    g = e["game"]
    if not (isinstance(g, int) and 0 <= g < N_GAMES):
        raise ValueError(f"event {e} names no scheduled game")
    return g


_SLOT_OF = {"game_start": lambda e: GAME + _game(e), "select_start": lambda e: LABEL + _game(e),
            "label_start": lambda e: LABEL + _game(e), "repeat_start": lambda e: repeat_slot(_game(e)),
            "train_start": lambda e: TRAIN}


def used_slots(events: Sequence[Mapping[str, Any]]) -> List[int]:
    """Card §9: the union of slots any phase logged as used; other events are ignored."""
    return sorted({_SLOT_OF[e["event"]](e) for e in events if e["event"] in _SLOT_OF})


def closing_intervals(base: int, events: Sequence[Mapping[str, Any]]) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
    """EXPOSED = the used slots as merged half-open intervals; RETIRED = the whole block, whatever happened."""
    runs: List[List[int]] = []
    for s in used_slots(events):
        if runs and runs[-1][1] == base + s:
            runs[-1][1] += 1
        else:
            runs.append([base + s, base + s + 1])
    return [tuple(r) for r in runs], (base, base + N_SLOTS)
