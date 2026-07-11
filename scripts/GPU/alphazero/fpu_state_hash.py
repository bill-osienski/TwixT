"""Complete-state canonical hash for `TwixtState`.

A pure, standalone module (no MCTS, no GPU/MLX) providing a canonical SHA1
digest over the COMPLETE future-play-relevant `TwixtState`. Used by later
FPU (policy-mass) tooling tasks (dev-corpus disjointness, controls join key)
to dedupe/join positions.

Contract: equal hash <=> identical future play from this state, i.e. same
side-to-move, legal-move set, terminal result, and NN input tensor
(`to_tensor`). Transpositions that reach the same board state hash equal.
Any change to a future-relevant field changes the hash.

Frozen design ref: docs/superpowers/specs/2026-07-10-context-relative-fpu-policy-mass-design.md
Plan Task 4.

Field inventory against `TwixtState` (scripts/GPU/alphazero/game/twixt_state.py):
    board_size        -- included (physical tensor dimension; feeds to_tensor)
    active_size       -- included (curriculum playable region; feeds legality
                          and to_tensor)
    to_move           -- included (side to move)
    pegs              -- included, as sorted (row, col, player) triples
    bridges           -- included, as sorted nested-tuple pairs
    max_plies_limit   -- included (changes is_terminal()/game_result()
                          without changing pegs/bridges) -- MUST serialize
                          None as JSON null, not be silently dropped
    ply               -- EXCLUDED. Every `apply_move` call adds exactly one
                          peg and increments ply by exactly one, and no
                          production path mutates pegs without going through
                          apply_move, so ply == len(pegs) always holds for
                          reachable states; pegs already determines it. The
                          equal-hash-implies-equal-to_tensor test is the
                          completeness guard: to_tensor encodes ply in its
                          MOVE_NUMBER channel (ply / MAX_PLIES), so if ply
                          could ever diverge from len(pegs) that test would
                          catch it.
    _adj              -- EXCLUDED. A derived, lazily-built adjacency cache
                          (peg -> neighbor pegs) backing connected-component
                          queries; fully determined by pegs/bridges and
                          already excluded from TwixtState's own
                          __eq__/__hash__ (`compare=False`).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Tuple

from scripts.GPU.alphazero.game.twixt_state import TwixtState


def canonical_state_key(state: TwixtState) -> Tuple[Any, ...]:
    """Return the canonical, JSON-serializable key tuple for `state`.

    Key = (board_size, active_size, to_move, sorted peg triples,
    sorted bridges, max_plies_limit).

    All fields future play could depend on are present; nothing derived
    (e.g. `_adj`) or redundant (e.g. `ply`, captured by `pegs`) is included.
    Tuples become JSON arrays under `json.dumps` (nested bridge tuples
    included), so every element here is JSON-serializable as-is.
    """
    peg_triples = sorted((r, c, player) for (r, c), player in state.pegs.items())
    sorted_bridges = sorted(state.bridges)
    return (
        state.board_size,
        state.active_size,
        state.to_move,
        peg_triples,
        sorted_bridges,
        state.max_plies_limit,
    )


def canonical_state_sha1(state: TwixtState) -> str:
    """Return the hex SHA1 digest of `state`'s canonical key.

    Digest = sha1(json.dumps(canonical_state_key(state), sort_keys=True)).
    `sort_keys=True` is belt-and-suspenders here: the key is a tuple (JSON
    array), not a dict, so determinism actually comes from the explicit
    `sorted(...)` calls in `canonical_state_key`.
    """
    key = canonical_state_key(state)
    payload = json.dumps(key, sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()
