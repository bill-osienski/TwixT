"""H3 PILOT — the frozen constants and the opening generator.

Everything here is fixed by `docs/superpowers/2026-09-13-t1j-h3-pilot-card.md`
BEFORE any game is played, because fixing it afterwards is how a pilot becomes a
result hunt.

🔴 NOTHING HERE PLAYS, LOADS OR DRAWS. Generation is not play: `OPENING_SEED` is
a declared constant, is NOT drawn from any match seed block, and a test asserts it
sits in no registry. No seed block is reserved by this module; reserving one is a
separate authorization.

THE PILOT CANNOT SUPPLY A STRENGTH VERDICT. There is deliberately no rate, no
interval and no comparison in this module.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import d1_selection as SEL

# ═══════════════════════════ the frozen design (card §2, §3) ═══════════════
BOARD_SIZE = 24
N_OPENINGS = 20
N_ARMS = 2                                   # our incumbent as red, then as black
N_GAMES = N_OPENINGS * N_ARMS                # 40
#: 6 -- the `opening_bound` every H1/H2 game used, so the pilot's positions sit at
#: the depth the programme has already played from.
OPENING_PLIES = 6

#: 🔑 A DECLARED CONSTANT, NOT A MATCH SEED. Generation is reproducible and must
#: not consume a drawable seed; `test_the_generation_seed_is_NOT_a_match_seed`
#: asserts it is absent from every registry.
OPENING_SEED = 20260913

# ═══════════════════════════ bounds (card §4) ══════════════════════════════
#: CHOSEN, not derived, and THE PILOT MAY TIME OUT. Attempt 3's records carry no
#: timestamps, so no per-game average exists to bound anything with.
RUN_DEADLINE_S = 7200
PER_CALL_TIMEOUT_S = 120

# ═══════════════════════════ stop rules (card §6) ══════════════════════════
#: Each is a CEILING: the rule fires when the count EXCEEDS it.
MAX_DUPLICATE_PAIRS = 2                      # S1
MAX_CAPPED_GAMES = 8                         # S2
MAX_WITHIN_PAIR_IDENTICAL = 2                # S3
MAX_TOTAL_ELAPSED_S = 3600                   # S4a
MAX_P90_OVER_MEDIAN = 4.0                    # S4b -- a SHAPE ratio
#: Below this many completed pairs the run withholds INTERPRETATION -- never the
#: monotone stop rules, which may still fire (card §4.4).
REPORT_FLOOR_PAIRS = 10

#: The eight openings H1 and H2 played. The pilot's positions must duplicate none
#: of them, so its evidence is independent of theirs.
PRIOR_OPENING_NAMES = ("o1_center", "o2_offcenter", "o3_low", "o4_high",
                       "o5_wide_left", "o6_wide_right", "o7_diagonal",
                       "o8_contact")


def _fresh_state():
    """The ENGINE's state -- `TwixtState`, the one `canonical_digest` keys on.

    🔑 NOT `game.state.GameState`. The two are different classes; the digest reads
    `active_size`, which only the engine state carries, so digesting the wrong one
    raises rather than silently hashing a different key.
    """
    from .game.twixt_state import TwixtState
    st = TwixtState()
    if st.board_size != BOARD_SIZE:
        raise RuntimeError(f"engine board is {st.board_size}, not {BOARD_SIZE}")
    return st


def _replay(moves: Sequence[Tuple[int, int]]):
    """An empty board advanced through the ENGINE's own rules. Never hand-built."""
    st = _fresh_state()
    for m in moves:
        st = st.apply_move(tuple(m))
    return st


def prior_opening_digests() -> frozenset:
    """Canonical digests of H1/H2's eight openings, resolved from the frozen H2
    source plan rather than retyped."""
    from . import h2_match_plan as PLAN
    openings = PLAN.load_source_plan()["openings"]
    if set(openings) != set(PRIOR_OPENING_NAMES):
        raise RuntimeError(
            f"the H2 source plan names {sorted(openings)}, not the eight this "
            f"module excludes: {sorted(PRIOR_OPENING_NAMES)}")
    out = set()
    for name in PRIOR_OPENING_NAMES:
        out.add(SEL.canonical_digest(_replay([tuple(m) for m in openings[name]])))
    return frozenset(out)


def generate_openings(seed: int = OPENING_SEED,
                      n: int = N_OPENINGS) -> List[Dict[str, Any]]:
    """`n` distinct legal non-terminal positions at `OPENING_PLIES`, card §2.

    UNIFORMLY RANDOM LEGAL PLAY, which is **engine-independent** -- neither side's
    preferences touch the selection, the property H1/H2's hand-chosen openings
    lacked -- and **not realistic**. The pilot therefore answers "do varied
    positions give varied games", never "is the incumbent stronger in realistic
    play". That trade-off is stated in the card and is not hidden here.

    Rejection is WHOLE-POSITION: a candidate that ends terminal, duplicates an
    accepted digest, or duplicates one of H1/H2's openings is discarded entirely
    and resampled. Rejecting move-by-move would bias the walk.
    """
    import numpy as np

    rng = np.random.Generator(np.random.PCG64(seed))
    prior = prior_opening_digests()
    out: List[Dict[str, Any]] = []
    seen = set()
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > 1000 * n:                       # a walk that cannot finish
            raise RuntimeError(
                f"only {len(out)} of {n} openings after {attempts} attempts")
        st = _fresh_state()
        moves: List[Tuple[int, int]] = []
        for _ in range(OPENING_PLIES):
            legal = st.legal_moves()
            if not legal or st.is_terminal():
                break
            r, c = legal[int(rng.integers(len(legal)))]
            moves.append((int(r), int(c)))
            st = st.apply_move((int(r), int(c)))
        if len(moves) != OPENING_PLIES or st.is_terminal():
            continue
        digest = SEL.canonical_digest(st)
        if digest in seen or digest in prior:
            continue
        seen.add(digest)
        out.append({"index": len(out), "moves": moves, "digest": digest,
                    "state": st})
    return out


def opening_set_digest(openings: Sequence[Dict[str, Any]]) -> str:
    """sha256 over the openings' canonical digests, in order. Pins the SET."""
    return hashlib.sha256(
        "\n".join(o["digest"] for o in openings).encode()).hexdigest()


#: Recomputed from the generated set by a test -- a pin never checked against the
#: artifact it pins is decoration.
OPENING_SET_DIGEST = "4027efe39c78ceca8e2b6ea808940e940ca795c269e5d0c8a807cfc974f01cc6"


def opening_name(index: int) -> str:
    """The opening's NAME, which the shared state factory keys on."""
    return f"p{int(index):02d}"


def openings_mapping(openings: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """`{name: moves}`, the shape `e4_screen_integration.make_state_factory` wants."""
    return {opening_name(op["index"]): [tuple(m) for m in op["moves"]]
            for op in openings}


class H3PilotError(RuntimeError):
    """A refusal from the pilot's own rules. Never a verdict."""


def build_tasks(openings: Sequence[Dict[str, Any]],
                seed_interval: Optional[Tuple[int, int]] = None
                ) -> List[Dict[str, Any]]:
    """The 40 tasks: each opening played BOTH WAYS, the pair kept adjacent.

    SEEDS ARE NOT ASSIGNED BY DEFAULT, for D1's reason: the card reserves no
    block, and a plan that hands out seeds before one is authorized has spent it
    on paper. Supply `seed_interval` only once a block has been reserved, proved
    disjoint and registered; it is size-checked and refuses any interval that
    overlaps a spent one.
    """
    from . import e4_screen_reference as REF
    from . import h2_match_rules as H2R

    if len(openings) != N_OPENINGS:
        raise H3PilotError(f"{len(openings)} openings, expected {N_OPENINGS}")

    seeds: List[Optional[int]] = [None] * N_GAMES
    if seed_interval is not None:
        # 🔴 TYPE-STRICT, not coerced. `int()` accepted "777000000" and
        # 777000000.0 silently; the programme refuses those everywhere else,
        # because a seed that arrives as a float or a string has been through a
        # conversion an int would not have survived. `True` is not 1 either.
        lo, hi = seed_interval
        for name, v in (("lo", lo), ("hi", hi)):
            if type(v) is not int:
                raise H3PilotError(
                    f"seed interval {name} is {v!r} ({type(v).__name__}); an int "
                    f"is required and no value is coerced into one")
        if hi - lo != N_GAMES:
            raise H3PilotError(
                f"the supplied interval holds {hi - lo} seeds for {N_GAMES} games")
        # 🔑 REFUSE A SPENT BLOCK BY STATUS, not by naming known intervals: a list
        # of names goes stale, a status query cannot.
        for s_ in range(lo, hi):
            st = REF.seed_status(s_)
            if st["exposed"] or st["retired"] or st["test_only"]:
                raise H3PilotError(
                    f"seed {s_} is spent or reserved for tests "
                    f"(exposed={st['exposed']} RETIRED={st['retired']} "
                    f"test_only={st['test_only']}); a spent block may not be revived")
        seeds = list(range(lo, hi))

    out: List[Dict[str, Any]] = []
    for op in openings:
        for colour in ("red", "black"):
            i = len(out)
            out.append({
                "task_id": f"h3pilot-{i:03d}-p{op['index']:02d}-inc_{colour}",
                "index": i,
                "pair_id": op["index"],
                "incumbent_colour": colour,
                # the shared machinery keys on these two: `make_state_factory`
                # looks the opening up by NAME, and `reference_colour` derives OUR
                # colour as the opposite of the anchor's -- so the anchor is T1j's.
                "opening": opening_name(op["index"]),
                "anchor_colour": "black" if colour == "red" else "red",
                "opening_digest": op["digest"],
                "opening_moves": [list(m) for m in op["moves"]],
                "opening_plies": OPENING_PLIES,
                "seed": seeds[i],
                # H2's frozen configuration, unchanged and READ, never retyped
                "selection_mode": H2R.SELECTION_MODE,
                "mcts_sims": H2R.MCTS_SIMS,
                "t1j_mdPly": H2R.T1J_MDPLY,
                "t1j_mdFixedPly": True,
                "ply_cap": H2R.PLY_CAP,
            })
    if len(out) != N_GAMES:
        raise H3PilotError(f"built {len(out)} tasks, expected {N_GAMES}")
    return out


def task_digest(tasks: Sequence[Dict[str, Any]]) -> str:
    """sha256 over EVERY field of every task, in order. Nothing projected away --
    H2's full-field digest lesson, where a design-dimension digest left a forged
    `rng_streams` unchanged."""
    import json
    return hashlib.sha256(json.dumps([dict(t) for t in tasks], sort_keys=True,
                                     separators=(",", ":"),
                                     default=str).encode()).hexdigest()


#: Recomputed from the BUILT schedule by a test.
TASK_DIGEST = "17516342892f0be35d9c282c4cbea2bf792449a02084d35a2904c0256110dd43"
