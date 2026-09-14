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
    from . import l0_match_plan as L0PLAN

    if len(openings) != N_OPENINGS:
        raise H3PilotError(f"{len(openings)} openings, expected {N_OPENINGS}")

    # 🔑 THE IDENTITY THE QUALIFIED CONSTRUCTION READS OFF A TASK, from the
    # sha256-VERIFIED source plan L0, H1 and H2 all use. Not retyped, and
    # deliberately NOT taken from `REFERENCE_CHECKPOINTS` -- that registry is what
    # `validate_task_structure` compares the task against, so sourcing it there
    # would leave the check comparing a value with itself.
    ref = L0PLAN.load_source_plan()["reference"]

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
            t = {
                "task_id": f"h3pilot-{i:03d}-p{op['index']:02d}-inc_{colour}",
                "index": i,
                "pair_id": op["index"],
                "incumbent_colour": colour,
                # the shared machinery keys on these two: `make_state_factory`
                # looks the opening up by NAME, and `reference_colour` derives OUR
                # colour as the opposite of the anchor's -- so the anchor is T1j's.
                "opening": opening_name(op["index"]),
                "anchor_colour": "black" if colour == "red" else "red",
                "reference": ref["name"],
                "reference_sha1": ref["sha1"],
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
            }
            # 🔴 READ, NOT DERIVED, by the two callers that matter:
            # `make_agent_factory` routes on `task["reference_colour"]` by
            # subscript, and `_enforce_evaluator` refuses a task that names none.
            # Its absence killed every game at ply 6 -- which no seam test could
            # see, because each supplied its own agent factory.
            t["reference_colour"] = REF.reference_colour(t)
            out.append(t)
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


#: THE UNSEEDED DESIGN IDENTITY, recomputed from the BUILT schedule by a test.
#: This is the schedule as the card fixes it, with `seed: None` on every task.
TASK_DIGEST = "b972ce46beb637d45a56fe3b002eb88932652fc5134b5db3dbbf6702de192337"

#: 🔴 THE SEEDED PIN. `task_digest` covers EVERY field, so assigning 40 seeds
#: changes it: the unseeded pin above does NOT describe the schedule that would
#: actually run, and comparing a seeded schedule against it would refuse the real
#: thing at execution time.
#:
#: PINNED 2026-09-14 at seed registration on block [202624000, 202624040), then
#: RE-PINNED THE SAME DAY -- both digests above -- when the tasks gained the
#: `reference`, `reference_sha1` and `reference_colour` fields the qualified
#: construction path reads. A full-field digest moves whenever the task shape
#: does, and a pin carried over from a schedule with a different shape would name
#: something that no longer exists. Recomputed from the 40 SEEDED tasks, never
#: derived from the unseeded pin. A test rebuilds the seeded schedule and
#: recomputes both, so a pin that drifts from the schedule it names fails loudly.
SEEDED_TASK_DIGEST: Optional[str] = (
    "aa527cc9a1a7b1e657911171c63f19fc006909dd64518bd96de3ce4ddfabfba9")


def expected_task_digest(tasks: Sequence[Dict[str, Any]]) -> str:
    """Which pin this schedule must match -- the seeded one if it carries seeds.

    Refuses rather than choosing a pin that does not exist, and refuses a schedule
    that carries SOME seeds: a half-seeded schedule is neither artifact.
    """
    seeded = [t for t in tasks if t.get("seed") is not None]
    if not seeded:
        return TASK_DIGEST
    if len(seeded) != len(tasks):
        raise H3PilotError(
            f"{len(seeded)} of {len(tasks)} tasks carry a seed; a half-seeded "
            f"schedule matches neither pin")
    if SEEDED_TASK_DIGEST is None:
        raise H3PilotError(
            "this schedule carries seeds, but SEEDED_TASK_DIGEST is None. The "
            "full-field digest changes when seeds are assigned, so the pin must be "
            "RECOMPUTED FROM THE SEEDED SCHEDULE and set at seed registration "
            "before any run. Nothing may execute against the unseeded pin.")
    return SEEDED_TASK_DIGEST
