"""H3 — THE IMMUTABLE GENERATION PROTOCOL. Pinned by content in every artifact.

🔴 WHY THIS MODULE EXISTS, AND IT IS NOT TIDINESS.

The frozen artifact records a sha256 of every source that determines the walk, so
a population cannot be played under code that would produce a different one.
Those sources used to include `h3_study_rules.py` — **which also holds
`OPENING_SET_DIGEST`, the constant the freeze sequence must edit next.**

    freeze  ->  artifact pins h3_study_rules.py with OPENING_SET_DIGEST = None
    pin     ->  record the digest in h3_study_rules.py
    play    ->  load_opening_set REFUSES: source_pins drift

The population invalidated itself on the one edit its own procedure required, and
nothing could ever be played. A pin that covers the file it will be written into
is not a pin, it is a trap.

🔑 THE RULE THIS MODULE ENFORCES BY EXISTING: **an OUTPUT may never live inside a
PINNED INPUT.** Everything here determines what the walk produces. Nothing here
records what it produced. `OPENING_SET_DIGEST`, the retired seed ranges, the
destination paths and every other thing that moves after a run stay in
`h3_study_rules` and `h3_study_generator`, which are NOT pinned.

WHAT IS PINNED, AND WHY EACH ONE

    h3_generation_protocol.py   this file: the walk, the filters, the allocation
    game/twixt_state.py         `legal_moves()` ordering and `apply_move`
    d1_selection.py             `canonical_digest`, the symmetry reduction

  plus, BY VALUE rather than by source: the resolved exclusion set (28 digests).
  Pinning `h3_pilot_rules.py` would pin the CODE that computes those digests; the
  population depends on the digests themselves, and a refactor that leaves them
  identical must not invalidate a frozen artifact.

WHAT IS DELIBERATELY *NOT* HERE

  * the spent/retired generation ranges — retiring a range refuses a base, it
    never changes what a live base produces. Keeping them here would make every
    retirement invalidate every frozen population, which is the same bug in a
    different costume;
  * `OPENING_SET_DIGEST` and the output paths — outputs.

The policy guard is INJECTED (`guard=`) so it can refuse a base without its
implementation being part of the pinned surface.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class H3ProtocolError(RuntimeError):
    """A refusal from the generation protocol. Never a population."""


# ═══════════════════════ the constants the walk obeys ══════════════════════
BOARD_SIZE = 24
OPENING_PLIES = 6

#: 296 pairs / 592 games, from the precision target alone (card §3).
N_PAIRS = 296

#: per-opening attempt ceiling. Exhausting it ABORTS: never a short population,
#: never a relaxed filter. Also the term that sizes the seed range --
#: N_PAIRS x MAX_ATTEMPTS is the number of CANDIDATES.
MAX_ATTEMPTS = 400

#: the live generation base. One contiguous range of 118,400 covering every
#: candidate, proved separate directly and through the derived streams by
#: collision proof v13.
GEN_SEED_UNIFORM = 20_261_600_000


def generation_seed_range(base: int) -> Tuple[int, int]:
    """The whole half-open range generation can touch: EVERY candidate."""
    return (base, base + N_PAIRS * MAX_ATTEMPTS)


def seed_for(base: int, index: int, attempt: int) -> int:
    """THE SEED ALLOCATION. Attempt `attempt` of opening `index`.

    🔑 DISJOINT AND PRE-COMPUTABLE. A rejected candidate is never regenerated
    from the seed that produced it, so the loop cannot spin; and the whole range
    can be checked against the registries before a single seed is used.

    Pure arithmetic and type checks. The SPENT-RANGE refusal is policy and lives
    in `h3_study_rules.attempt_seed`, which delegates here -- so retiring a range
    changes that module's hash and not this one's.
    """
    for name, v in (("base", base), ("index", index), ("attempt", attempt)):
        if type(v) is not int or isinstance(v, bool) or v < 0:
            raise H3ProtocolError(f"{name} must be a non-negative int, got {v!r}")
    if attempt >= MAX_ATTEMPTS:
        raise H3ProtocolError(
            f"attempt {attempt} is beyond MAX_ATTEMPTS={MAX_ATTEMPTS}; exhausting "
            f"the attempts ABORTS generation and never relaxes a filter")
    if index >= N_PAIRS:
        raise H3ProtocolError(f"index {index} is beyond N_PAIRS={N_PAIRS}")
    return base + index * MAX_ATTEMPTS + attempt


# ═══════════════════════ the board, the walk, the filters ══════════════════
def fresh_state():
    """The ENGINE's state — `TwixtState`, the one `canonical_digest` keys on."""
    from .game.twixt_state import TwixtState
    st = TwixtState()
    if st.board_size != BOARD_SIZE:
        raise H3ProtocolError(f"engine board is {st.board_size}, not {BOARD_SIZE}")
    return st


def replay(moves: Sequence[Tuple[int, int]]):
    """Replay `moves` on a fresh board. Raises if any is illegal."""
    st = fresh_state()
    for m in moves:
        st = st.apply_move(tuple(m))
    return st


def admissible(st, moves: Sequence[Tuple[int, int]]) -> bool:
    """The card's §1.3 structural filters. ENGINE-NEUTRAL BY CONSTRUCTION: no
    evaluator is consulted, because "the incumbent thinks this is decided" is the
    incumbent's judgement and would re-import the bias a uniform population
    exists to avoid.
    """
    if len(moves) != OPENING_PLIES:
        return False
    if st.is_terminal():
        return False
    if not st.legal_moves():
        return False
    return True


def walk(seed: int):
    """THE CANDIDATE WALK: six uniformly random legal plies from ONE seed.

    Deterministic in `seed` alone — `PCG64(seed)`, the engine's own
    `legal_moves()` ordering (built by nested ranges into a list, so it does not
    depend on the interpreter's hashing), and nothing else. That determinism is
    what lets a frozen artifact be re-derived from its seeds, and what makes the
    digest a pin rather than a snapshot.

    🔑 ONE IMPLEMENTATION, CALLED BY BOTH GENERATION AND VERIFICATION. A verifier
    with its own copy of the walk checks that the copy agrees with itself.
    """
    import numpy as np
    if type(seed) is not int or isinstance(seed, bool):
        raise H3ProtocolError(f"seed must be an int, got {seed!r}")
    rng = np.random.Generator(np.random.PCG64(seed))
    st = fresh_state()
    moves: List[Tuple[int, int]] = []
    for _ in range(OPENING_PLIES):
        legal = st.legal_moves()
        if not legal or st.is_terminal():
            break
        r, c = legal[int(rng.integers(len(legal)))]
        moves.append((int(r), int(c)))
        st = st.apply_move((int(r), int(c)))
    return moves, st


def canonical_digest(st) -> str:
    """The symmetry-reduced digest the population is keyed on."""
    from . import d1_selection as SEL
    return SEL.canonical_digest(st)


def verify_candidate(base: int, index: int, attempts: int
                     ) -> Tuple[List[Tuple[int, int]], str]:
    """Re-derive opening `index` from its DECLARED provenance: the moves and the
    canonical digest the generator must have produced.

    🔑 THIS IS THE BINDING. An artifact row asserts "these moves, this digest,
    this seed, this many attempts". Only re-running the walk from
    `seed_for(base, index, attempts - 1)` can show those four agree; comparing a
    row's digest to its own digest shows nothing at all.
    """
    for name, v in (("index", index), ("attempts", attempts)):
        if type(v) is not int or isinstance(v, bool):
            raise H3ProtocolError(f"{name} must be an int, got {v!r}")
    if attempts < 1:
        raise H3ProtocolError(f"attempts must be >= 1, got {attempts}")
    seed = seed_for(base, index, attempts - 1)
    moves, st = walk(seed)
    if not admissible(st, moves):
        raise H3ProtocolError(
            f"opening {index}: the candidate re-derived from seed {seed} is NOT "
            f"admissible, so the generator cannot have accepted it there")
    return moves, canonical_digest(st)


# ═══════════════════════ the acceptance loop ═══════════════════════════════
def generate(base: int, n: int, *, excluded: Iterable[str],
             guard=None, check_deadline=None) -> List[Dict[str, Any]]:
    """`n` admissible, distinct positions from uniformly random legal play.

    WHOLE-POSITION REJECTION: a candidate failing any filter is discarded entire
    and the next ATTEMPT SEED is used. Never per-move resampling, which would
    distort the distribution it claims to draw from.

    `excluded` is the resolved set of digests this population may not contain —
    a VALUE, so the code that computes it is not part of the pinned surface.
    `guard` is called once with `base` before anything is drawn, so a policy
    refusal (a spent range) can bind without living here.

    🔴 `check_deadline` IS CALLED INSIDE THE CANDIDATE LOOP, not around it. A
    guard that only runs between openings cannot stop a walk that hangs, and one
    evaluated before the loop starts cannot stop anything at all. It is called
    once per ATTEMPT and may raise; nothing here catches it.
    """
    if guard is not None:
        guard(base)
    excluded = frozenset(excluded)
    out: List[Dict[str, Any]] = []
    seen = set()
    for index in range(n):
        for attempt in range(MAX_ATTEMPTS):
            if check_deadline is not None:
                check_deadline(index, attempt, len(out))
            s = seed_for(base, index, attempt)
            moves, st = walk(s)
            if not admissible(st, moves):
                continue
            digest = canonical_digest(st)
            if digest in seen or digest in excluded:
                continue
            seen.add(digest)
            out.append({"index": index, "stratum": "uniform",
                        "moves": moves, "digest": digest, "state": st,
                        "seed": s, "attempts": attempt + 1})
            break
        else:
            raise H3ProtocolError(
                f"opening {index} exhausted MAX_ATTEMPTS={MAX_ATTEMPTS} attempts. "
                f"Generation ABORTS rather than delivering {len(out)} of {n} or "
                f"relaxing a filter.")
    return out
