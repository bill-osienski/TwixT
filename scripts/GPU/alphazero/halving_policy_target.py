"""THE COMBINATION POLICY PILOT'S TARGET -- card `docs/superpowers/2026-10-28-combination-policy-pilot-card.md` §4.
Pure numpy over one completed halving search; nothing runs on import and importing this module loads no MLX.

From one halving search (ply >= 20, `last_choice` set): C = the first round's candidates (`last_rounds[0]`), p = the
root's priors over its legal moves (the mirror-averaged prior the search used; no noise), q(a) = candidate a's mean
value from the root player's side (-child.q_value, as the halving rank reads it), N_max = the largest child visit
count when the search ends. Then

    for a in C:     pi(a) = P_C * softmax_C(score(p(a), q(a), N_max)),    P_C = sum over b in C of p(b)
    for a not in C: pi(a) = p(a)

with `root_sequential_halving.score` imported, never modified, and the sum computed in float64. Moves the search never
examined keep exactly their prior mass; within C the mass moves as the search judged. Halving's visit counts are a
schedule (3 / 6 / 12 / 28 per round at 200 simulations), not a preference, and are never the target.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import numpy as np

from . import root_sequential_halving as RSH
from .mcts import encode_move


def halving_target(prior: Sequence[float], cand: Sequence[int], q: Sequence[float], n_max: int) -> np.ndarray:
    """pi over the legal moves (float64), from p over the same moves, the candidates' indices, their q and N_max."""
    prior = np.asarray(prior, dtype=np.float64)
    cand = [int(i) for i in cand]
    if not cand or len(set(cand)) != len(cand) or len(q) != len(cand):
        raise ValueError(f"candidates {cand} with {len(q)} values are not a non-empty set with one value each")
    if not all(0 <= i < prior.size for i in cand):
        raise ValueError(f"a candidate index is outside the {prior.size} legal moves")
    s = np.array([RSH.score(float(prior[i]), float(v), int(n_max)) for i, v in zip(cand, q)], dtype=np.float64)
    if not np.all(np.isfinite(s)):
        raise ValueError(f"non-finite halving scores {s}")
    w = np.exp(s - s.max())
    pi = prior.copy()
    pi[cand] = math.fsum(prior[cand]) * (w / math.fsum(w))
    return pi


def search_row(mcts: Any, root: Any) -> Dict[str, Any]:
    """The row for one completed halving search: what the target needs, read from the root that chose the move."""
    if mcts.last_choice is None or not mcts.last_rounds:
        raise ValueError("not a halving search: last_choice / last_rounds are unset")
    legal = [tuple(m) for m in root.state.legal_moves()]
    index = {m: i for i, m in enumerate(legal)}
    prior = np.array([root.priors[encode_move(r, c)] for r, c in legal], dtype=np.float64)
    cand = [index[tuple(m)] for m in mcts.last_rounds[0]]
    kids = [root.children.get(encode_move(*legal[i])) for i in cand]
    if any(k is None or k.visit_count < 1 for k in kids):
        raise ValueError("a first-round candidate was never visited")
    q = [-k.q_value for k in kids]
    visits = [int(k.visit_count) for k in kids]
    n_max = max(int(c.visit_count) for c in root.children.values())
    entered = {tuple(m): k + 1 for k, survivors in enumerate(mcts.last_rounds) for m in survivors}
    return {"ply": int(root.state.ply), "to_move": root.state.to_move, "n_legal": len(legal),
            "prior": prior.astype(np.float32), "cand": cand, "q": q, "visits": visits, "n_max": n_max,
            "rounds": [entered[legal[i]] for i in cand], "n_rounds": len(mcts.last_rounds),
            "choice": index[tuple(mcts.last_choice)], "target": halving_target(prior, cand, q, n_max)}
