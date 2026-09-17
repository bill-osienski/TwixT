"""H3 FULL STUDY — the frozen constants, the strata, the segments, the schedule.

Everything here is fixed by
`docs/superpowers/2026-09-15-t1j-h3-full-study-design.md` (AMENDED TWICE) BEFORE
any study game exists, because fixing it afterwards is how a study becomes a
result hunt.

🔴 NOTHING HERE PLAYS, LOADS A MODEL, STARTS A JVM OR DRAWS A MATCH SEED.
Stratum A is generated here — uniform random legal play needs no engine.
Stratum B is NOT: generating it is a RUN, and it lives behind its own gate in
`h3_study_generator`.

🔴 NO SEED BLOCK IS RESERVED BY THIS MODULE. `build_tasks` refuses to hand out
seeds unless an interval is supplied, and refuses any interval whose seeds are
spent.

THIS MODULE CANNOT PRODUCE A STRENGTH VERDICT. There is no rate, no interval and
no comparison in it; §3's `half_width` is a precision function of N alone and
never sees an outcome.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import d1_selection as SEL


class H3StudyError(RuntimeError):
    """A refusal from the study's own rules. Never a verdict."""


# ═══════════════════════ the frozen design (card §1, §3, §5) ═══════════════
BOARD_SIZE = 24
OPENING_PLIES = 6

#: 🔴 296, NOT 288. `ln(40)/(2·0.08²) = 288.194` needs 289, and 289 is PRIME --
#: it divides by neither stratum nor segment. 296 meets the target AND leaves no
#: remainder anywhere. The first draft rounded 288.194 DOWN and reported
#: h = 0.08003, above the target the arithmetic existed to enforce.
N_PAIRS = 296
N_ARMS = 2
N_GAMES = N_PAIRS * N_ARMS                      # 592

N_SEGMENTS = 4
PAIRS_PER_SEGMENT = N_PAIRS // N_SEGMENTS       # 74
GAMES_PER_SEGMENT = PAIRS_PER_SEGMENT * N_ARMS  # 148

STRATUM_UNIFORM = "uniform"
STRATUM_CO_PRODUCED = "co_produced"
STRATA = (STRATUM_UNIFORM, STRATUM_CO_PRODUCED)
PAIRS_PER_STRATUM = N_PAIRS // len(STRATA)                    # 148
STRATUM_PAIRS_PER_SEGMENT = PAIRS_PER_SEGMENT // len(STRATA)  # 37

#: 🔑 37 PER SEGMENT IS ODD, so the alternating orders cannot split evenly and the
#: allocation is DECLARED rather than derived at run time (card §1.7.6).
INCUMBENT_FIRST_PER_SEGMENT = (19, 18, 19, 18)
T1J_FIRST_PER_SEGMENT = (18, 19, 18, 19)

ORDER_INCUMBENT_FIRST = "incumbent_first"
ORDER_T1J_FIRST = "t1j_first"

# ═══════════════════════ precision (card §3) ═══════════════════════════════
ALPHA = 0.05
#: DECLARED on decision-relevance and on H1 having already failed at ~0.09.
#: Never derived from an observation.
PRECISION_TARGET = 0.08

# ═══════════════════════ generation (card §1.7) ════════════════════════════
#: 🔑 DECLARED CONSTANTS, NOT MATCH SEEDS. Generation must never consume a
#: drawable seed. Attempt j of opening i uses `base + i*MAX_ATTEMPTS + j`, so the
#: ranges below are reserved whole and a test asserts they sit in no registry.
#:
#: 🔴 THE SPACING HAS BEEN WRONG TWICE, each time caught by a different check.
#:   1. The first pair were 500 apart while each range is 148 x 400 = 59,200
#:      wide, so stratum B would have drawn from INSIDE stratum A's range. A test
#:      comparing the RANGES rather than the bases caught it.
#:   2. The second pair were 40,800 apart -- disjoint, but INSIDE THE GAP FLOOR,
#:      which this programme sets at the candidate's OWN SIZE. Collision proof
#:      v11 rejected them: disjoint is not the same as separated, and the floor
#:      exists so that extending either range later cannot silently collide.
#: They are now 140,800 apart, and both sit far above every registry's extent
#: (max 202,624,040), checked across their whole span against all four registries
#: and against each other, directly and through the derived streams.
MAX_ATTEMPTS = 400
GEN_SEED_UNIFORM = 20_261_000_000

#: 🔴 ATTEMPT 1's CO-PRODUCED RANGE, SPENT WHOLE ON 2026-09-16.
#: The single authorized generation VOIDed at opening 0 on `KeyError: None` --
#: a bare `IntegrationContext` whose `bump` had no bucket -- and produced NOTHING.
#: It is retired all the same, and the reason is not ceremony: the first attempt
#: USED its seed. It built the incumbent's agent on 20261200000 and put a query
#: to T1j from it. A drawn seed is drawn whether or not an opening survived.
SPENT_GENERATION_RANGES = (
    (20_261_200_000, 20_261_259_200),        # attempt 1, VOID at opening 0
)

#: ATTEMPT 2's range. 140,800 clear of the uniform range and 140,800 clear of
#: attempt 1's, both above the gap floor of 59,200 (a candidate's own size), and
#: proved directly and through the derived streams by collision proof v12.
GEN_SEED_CO_PRODUCED = 20_261_400_000

# ═══════════════════════ bounds (card §5) ══════════════════════════════════
#: CHOSEN, from the pilot's measured mean of 41.22 s/game: 148 games is ~1.69 h,
#: so this is 1.77x. It MAY still time out, and that is a reportable outcome.
SEGMENT_DEADLINE_S = 10800
PER_CALL_TIMEOUT_S = 120

#: 🔴 THE GENERATION RUN'S OWN DEADLINE, PREREGISTERED SEPARATELY. It happens to
#: be the same number as `SEGMENT_DEADLINE_S`, and that is a coincidence of
#: rounding, not a shared derivation: the segment cap is for 148 GAMES at the
#: pilot's measured 41.22 s/game, and says nothing about 148 six-ply GENERATION
#: walks. Reusing it would have been a bound borrowed from a different quantity.
#:
#: ITS OWN RATIONALE, from the pilot's measured per-ply cost of 41.22/51 = 0.81 s
#: (both engines, one ply):
#:   * an opening is 6 plies  ->  ~4.85 s per ATTEMPT
#:   * 148 openings at one attempt each  ->  ~718 s
#:   * 10,800 s therefore allows ~15 ATTEMPTS PER OPENING on average, against a
#:     MAX_ATTEMPTS ceiling of 400.
#:
#: 🔑 THE REJECTION RATE IS NOT KNOWN AND IS NOT ESTIMATED HERE. The uniform
#: stratum needs 1 attempt per opening, but it is engine-free and says nothing
#: about a co-produced walk. So this is a CHOSEN limit, exactly as the pilot's
#: was, and IT MAY EXPIRE.
#:
#: 🔴 AND EXPIRY RETIRES THE WHOLE GENERATION RANGE. A timeout does not mean "try
#: again with more time": the attempts made drew from their seeds, and every
#: terminal record says so. A second generation needs a FRESH range with its own
#: collision re-proof. That cost is accepted in advance, here, rather than argued
#: about afterwards.
GENERATION_DEADLINE_S = 10800

# ═══════════════════════ degeneracy gates (card §4.6) ══════════════════════
#: 🔴 > 0, and a HARNESS FAULT rather than a population property: `opening_digest`
#: is part of the pair identity and the openings are distinct by construction, so
#: two distinct pairs CANNOT share one. Any occurrence means the same opening was
#: scheduled or recorded twice.
MAX_DUPLICATE_PAIRS = 0
MAX_WITHIN_PAIR_IDENTICAL = 7                    # 2.5% of 296
MAX_CAPPED_GAMES = 118                           # 20% of 592
REPORT_FLOOR_PAIRS = N_PAIRS // 2                # 148
#: 🔑 SHARED CONTINUATIONS HAVE NO CEILING AND NEVER WILL. Two games from
#: different openings are distinct observations even when their move sequences
#: match -- the opening changes the board those moves are played on. Amendment 1
#: put a ceiling here; Amendment 2 removed it, because a gate with no
#: preregistered reason to fire is a gate that suppresses sound results.


def reference_identity() -> Dict[str, str]:
    """The incumbent's name and sha1, from the sha256-VERIFIED source plan.

    NOT from `REFERENCE_CHECKPOINTS`, which is what `validate_task_structure`
    compares a task against — sourcing it there would leave that check comparing a
    value with itself.
    """
    from . import l0_match_plan as L0PLAN
    ref = L0PLAN.load_source_plan()["reference"]
    return {"name": ref["name"], "sha1": ref["sha1"]}


def t1j_ply_cap() -> int:
    """The ply cap, READ from H2's frozen rules, never retyped."""
    from . import h2_match_rules as H2R
    return H2R.PLY_CAP


def t1j_depth() -> int:
    """T1j's fixed search depth, READ from H2's frozen rules, never retyped."""
    from . import h2_match_rules as H2R
    return H2R.T1J_MDPLY


def half_width(n: int) -> float:
    """The two-sided Hoeffding half-width at `n` pair scores in [0, 1].

    🔑 DISTRIBUTION-FREE: it takes n and nothing else. No variance, no effect
    size, no observation -- which is precisely why the pilot's outcome
    distribution has no route into the study's size (card §3.1).
    """
    if type(n) is not int or isinstance(n, bool) or n <= 0:
        raise H3StudyError(f"n must be a positive int, got {n!r}")
    return math.sqrt(math.log(2 / ALPHA) / (2 * n))


# ═══════════════════════ generation seeds ══════════════════════════════════
def attempt_seed(base: int, index: int, attempt: int) -> int:
    """The seed for attempt `attempt` of opening `index`.

    🔑 DISJOINT AND PRE-COMPUTABLE. A rejected candidate is never regenerated
    from the seed that produced it, so the loop cannot spin; and the ranges can be
    checked against the registries before a single one is used.
    """
    if is_spent_generation_range(base):
        raise H3StudyError(
            f"the generation range at {base} is SPENT: it was attempted and its "
            f"seeds were drawn. A further attempt needs a FRESH range with its "
            f"own collision re-proof.")
    for name, v in (("index", index), ("attempt", attempt)):
        if type(v) is not int or isinstance(v, bool) or v < 0:
            raise H3StudyError(f"{name} must be a non-negative int, got {v!r}")
    if attempt >= MAX_ATTEMPTS:
        raise H3StudyError(
            f"attempt {attempt} is beyond MAX_ATTEMPTS={MAX_ATTEMPTS}; exhausting "
            f"the attempts ABORTS generation and never relaxes a filter")
    return base + index * MAX_ATTEMPTS + attempt


def is_spent_generation_range(base: int) -> bool:
    """Has a generation range already been attempted? A spent range may never be
    used again: its seeds were drawn the moment an agent was built on one."""
    lo, hi = base, base + PAIRS_PER_STRATUM * MAX_ATTEMPTS
    return any(not (hi <= s_lo or s_hi <= lo)
               for s_lo, s_hi in SPENT_GENERATION_RANGES)


def generation_seed_range(base: int) -> Tuple[int, int]:
    """The whole half-open range a stratum's generation can touch.

    🔑 THE COLLISION RE-PROOF MUST COVER THIS, not just the match block:
    generation seeds derive search and readout streams by the same masks, so a
    proof over the match seeds alone would miss a generation/match collision.
    """
    return (base, base + PAIRS_PER_STRATUM * MAX_ATTEMPTS)


def generation_config():
    """The configuration the incumbent GENERATES under — and it is NOT the one it
    PLAYS under.

    🔴 THE ENTROPY FINDING (card §1.7.2). Under the match's `argmax` the
    incumbent's move is a deterministic function of the position, and E3a proved
    T1j deterministic at fixed depth. An alternating generator built from those
    two players has NO ENTROPY ANYWHERE: every incumbent-first opening would be
    the same opening and every T1j-first one the same, so stratum B would hold two
    positions repeated 74 times each.

    The FROZEN RESEARCH configuration samples instead: `temp_high` applies for the
    first `opening_temp_plies` plies and generation is only six, so every
    generated ply is drawn from the visit distribution. It is the programme's own
    qualified mechanism for opening diversity, used unmodified and NOT retyped.
    """
    from . import twixtbot_g3_reference as G3
    cfg = G3.eval_config()
    if cfg.selection_mode == "argmax":
        raise H3StudyError(
            "the frozen research configuration is already argmax, so generation "
            "would have no entropy and stratum B would be two positions")
    if cfg.opening_temp_plies < OPENING_PLIES:
        raise H3StudyError(
            f"opening_temp_plies={cfg.opening_temp_plies} does not cover all "
            f"{OPENING_PLIES} generated plies, so some would be deterministic")
    return cfg


# ═══════════════════════ stratum A: uniform, engine-free ═══════════════════
def _fresh_state():
    """The ENGINE's state — `TwixtState`, the one `canonical_digest` keys on."""
    from .game.twixt_state import TwixtState
    st = TwixtState()
    if st.board_size != BOARD_SIZE:
        raise H3StudyError(f"engine board is {st.board_size}, not {BOARD_SIZE}")
    return st


def _replay(moves: Sequence[Tuple[int, int]]):
    st = _fresh_state()
    for m in moves:
        st = st.apply_move(tuple(m))
    return st


def excluded_digests() -> frozenset:
    """Canonical digests the study's openings may NOT duplicate: the PILOT's 20
    and H1/H2's 8, both resolved from their own frozen artifacts, never retyped.

    Without this the pilot's games would leak into the study.
    """
    from . import h3_pilot_rules as PILOT
    out = set(PILOT.prior_opening_digests())                 # H1/H2's eight
    out |= {o["digest"] for o in PILOT.generate_openings()}  # the pilot's twenty
    return frozenset(out)


def _admissible(st, moves: Sequence[Tuple[int, int]]) -> bool:
    """The card's §1.3 structural filters. ENGINE-NEUTRAL BY CONSTRUCTION: no
    evaluator is consulted, because "the incumbent thinks this is decided" is the
    incumbent's judgement and would re-import the bias the strata exist to limit.
    """
    if len(moves) != OPENING_PLIES:
        return False
    if st.is_terminal():
        return False
    if not st.legal_moves():
        return False
    return True


def generate_uniform_openings(seed: int = GEN_SEED_UNIFORM,
                              n: int = PAIRS_PER_STRATUM) -> List[Dict[str, Any]]:
    """Stratum A: `n` admissible positions from UNIFORMLY RANDOM legal play.

    Engine-independent by construction and unrealistic — which is exactly why the
    design keeps it as the comparator beside the co-produced stratum rather than
    betting on either alone.

    WHOLE-POSITION REJECTION: a candidate failing any filter is discarded entire
    and the next ATTEMPT SEED is used. Never per-move resampling, which would
    distort the distribution it claims to draw from. Exhausting `MAX_ATTEMPTS`
    ABORTS; it never returns a short set and never relaxes a filter.
    """
    import numpy as np

    excluded = excluded_digests()
    out: List[Dict[str, Any]] = []
    seen = set()
    for index in range(n):
        for attempt in range(MAX_ATTEMPTS):
            s = attempt_seed(seed, index, attempt)
            rng = np.random.Generator(np.random.PCG64(s))
            st = _fresh_state()
            moves: List[Tuple[int, int]] = []
            for _ in range(OPENING_PLIES):
                legal = st.legal_moves()
                if not legal or st.is_terminal():
                    break
                r, c = legal[int(rng.integers(len(legal)))]
                moves.append((int(r), int(c)))
                st = st.apply_move((int(r), int(c)))
            if not _admissible(st, moves):
                continue
            digest = SEL.canonical_digest(st)
            if digest in seen or digest in excluded:
                continue
            seen.add(digest)
            out.append({"index": index, "stratum": STRATUM_UNIFORM, "order": None,
                        "moves": moves, "digest": digest, "state": st,
                        "seed": s, "attempts": attempt + 1})
            break
        else:
            raise H3StudyError(
                f"opening {index} exhausted MAX_ATTEMPTS={MAX_ATTEMPTS} attempts. "
                f"Generation ABORTS rather than delivering {len(out)} of {n} or "
                f"relaxing a filter.")
    return out


def opening_set_digest(openings: Sequence[Dict[str, Any]]) -> str:
    """sha256 over the openings' canonical digests, in order. Pins the SET."""
    return hashlib.sha256(
        "\n".join(o["digest"] for o in openings).encode()).hexdigest()


#: 🔴 UNSET, AND IT MUST STAY UNSET UNTIL THE AUTHORIZED GENERATION PRODUCES IT.
#: A pin invented before the artifact exists pins nothing: it would either be a
#: guess the real set has to match, or -- worse -- a value the generator is
#: tempted to reproduce. The co-produced stratum comes from a run that has not
#: happened, so there is nothing to pin yet and this says so.
OPENING_SET_DIGEST: Optional[str] = None


def expected_opening_set_digest() -> str:
    """The pin, or a refusal. Never a default and never a computed stand-in."""
    if OPENING_SET_DIGEST is None:
        raise H3StudyError(
            "OPENING_SET_DIGEST is None. The study's population does not exist "
            "yet: the co-produced stratum is produced by a separate AUTHORIZED "
            "GENERATION RUN, and the pin is recorded FROM that run's artifact "
            "afterwards. Nothing may execute against an unpinned population.")
    return OPENING_SET_DIGEST


# ═══════════════════════ the study order and the segments ══════════════════
def segment_of(index: int) -> int:
    """Which segment a pair index belongs to.

    THE STUDY ORDER IS SEGMENT-MAJOR: pairs 0..73 are segment 0, 74..147 segment
    1, and so on. Because each pair contributes two adjacent tasks, a positional
    seed assignment then hands every segment a CONTIGUOUS quarter of the block
    with no further arithmetic — which is what lets one segment retire its own
    seeds and leave the others usable.
    """
    if type(index) is not int or isinstance(index, bool) or not 0 <= index < N_PAIRS:
        raise H3StudyError(f"index {index!r} is outside 0..{N_PAIRS - 1}")
    return index // PAIRS_PER_SEGMENT


def assemble_opening_set(uniform: Sequence[Dict[str, Any]],
                         co_produced: Sequence[Dict[str, Any]]
                         ) -> List[Dict[str, Any]]:
    """The 296 openings in STUDY ORDER, composition enforced segment by segment.

    🔑 THE COMPOSITION IS BUILT IN, NOT CHECKED AFTERWARDS. Each segment takes 37
    from each stratum, and the co-produced 37 split by the DECLARED
    19/18/19/18 allocation — because 37 is odd and an allocation left to emerge at
    run time is an allocation nobody fixed.
    """
    for name, got, want in (("uniform", len(uniform), PAIRS_PER_STRATUM),
                            ("co_produced", len(co_produced), PAIRS_PER_STRATUM)):
        if got != want:
            raise H3StudyError(f"stratum {name} has {got} openings, expected {want}")
    inc = [o for o in co_produced if o.get("order") == ORDER_INCUMBENT_FIRST]
    t1j = [o for o in co_produced if o.get("order") == ORDER_T1J_FIRST]
    if (len(inc), len(t1j)) != (sum(INCUMBENT_FIRST_PER_SEGMENT),
                                sum(T1J_FIRST_PER_SEGMENT)):
        raise H3StudyError(
            f"the co-produced stratum has {len(inc)} incumbent-first and "
            f"{len(t1j)} t1j-first, expected {sum(INCUMBENT_FIRST_PER_SEGMENT)} "
            f"and {sum(T1J_FIRST_PER_SEGMENT)}")

    out: List[Dict[str, Any]] = []
    ui = ii = ti = 0
    for k in range(N_SEGMENTS):
        take = [uniform[ui + j] for j in range(STRATUM_PAIRS_PER_SEGMENT)]
        ui += STRATUM_PAIRS_PER_SEGMENT
        take += [inc[ii + j] for j in range(INCUMBENT_FIRST_PER_SEGMENT[k])]
        ii += INCUMBENT_FIRST_PER_SEGMENT[k]
        take += [t1j[ti + j] for j in range(T1J_FIRST_PER_SEGMENT[k])]
        ti += T1J_FIRST_PER_SEGMENT[k]
        if len(take) != PAIRS_PER_SEGMENT:
            raise H3StudyError(f"segment {k} assembled {len(take)} pairs")
        for op in take:
            out.append({**op, "index": len(out), "segment": k})
    if len(out) != N_PAIRS:
        raise H3StudyError(f"assembled {len(out)} openings, expected {N_PAIRS}")
    return out


def stub_opening_set() -> List[Dict[str, Any]]:
    """The study order with a STUB co-produced stratum, for building and testing
    the schedule BEFORE the generation run is authorized.

    🔴 EVERY STUB OPENING IS MARKED `stub: True`, and `check_opening_set` REFUSES
    a set containing one. A stub that could pass for the artifact is how a study
    ends up run against placeholder positions.
    """
    stubs = []
    for i in range(PAIRS_PER_STRATUM):
        order = (ORDER_INCUMBENT_FIRST if i < sum(INCUMBENT_FIRST_PER_SEGMENT)
                 else ORDER_T1J_FIRST)
        stubs.append({
            "stratum": STRATUM_CO_PRODUCED, "order": order, "stub": True,
            "moves": [], "state": None, "seed": None, "attempts": 0,
            "digest": hashlib.sha256(
                f"H3-STUB-co_produced-{i}".encode()).hexdigest(),
        })
    uniform = [{**o, "stub": False} for o in generate_uniform_openings()]
    return assemble_opening_set(uniform, stubs)


def check_opening_set(openings: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The set must BE the study's population: count, composition, orders,
    distinctness, and no placeholder anywhere."""
    if len(openings) != N_PAIRS:
        raise H3StudyError(f"{len(openings)} openings, expected {N_PAIRS}")
    stubs = [o["index"] for o in openings if o.get("stub")]
    if stubs:
        raise H3StudyError(
            f"{len(stubs)} openings are STUBS (first index {stubs[0]}). A stub "
            f"stands in for the generation run's artifact and may never be played "
            f"against, pinned, or reported.")
    from collections import Counter
    strata = Counter(o["stratum"] for o in openings)
    if strata != {STRATUM_UNIFORM: PAIRS_PER_STRATUM,
                  STRATUM_CO_PRODUCED: PAIRS_PER_STRATUM}:
        raise H3StudyError(f"stratum composition is {dict(strata)}")
    for k in range(N_SEGMENTS):
        seg = [o for o in openings if segment_of(o["index"]) == k]
        orders = Counter(o["order"] for o in seg
                         if o["stratum"] == STRATUM_CO_PRODUCED)
        if (orders.get(ORDER_INCUMBENT_FIRST, 0) != INCUMBENT_FIRST_PER_SEGMENT[k]
                or orders.get(ORDER_T1J_FIRST, 0) != T1J_FIRST_PER_SEGMENT[k]):
            raise H3StudyError(f"segment {k} order split is {dict(orders)}")
    digests = [o["digest"] for o in openings]
    if len(set(digests)) != len(digests):
        raise H3StudyError("two openings share a canonical digest")
    clash = set(digests) & excluded_digests()
    if clash:
        raise H3StudyError(
            f"{len(clash)} openings duplicate a PILOT or H1/H2 position; the "
            f"study's evidence must be independent of theirs")
    return {"n": len(openings), "set_digest": opening_set_digest(openings)}


# ═══════════════════════ the schedule ══════════════════════════════════════
def opening_name(index: int) -> str:
    """The opening's NAME, which the shared state factory keys on."""
    return f"s{int(index):03d}"


def openings_mapping(openings: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {opening_name(op["index"]): [tuple(m) for m in op["moves"]]
            for op in openings}


def build_tasks(openings: Sequence[Dict[str, Any]],
                seed_interval: Optional[Tuple[int, int]] = None
                ) -> List[Dict[str, Any]]:
    """The 592 tasks: each opening played BOTH WAYS, the pair kept adjacent.

    SEEDS ARE NOT ASSIGNED BY DEFAULT. A plan that hands out seeds before a block
    is authorized has spent it on paper.
    """
    from . import e4_screen_reference as REF
    from . import h2_match_rules as H2R
    from . import l0_match_plan as L0PLAN

    if len(openings) != N_PAIRS:
        raise H3StudyError(f"{len(openings)} openings, expected {N_PAIRS}")

    # the identity the qualified construction READS off a task, from the
    # sha256-verified source plan -- not from REFERENCE_CHECKPOINTS, which is what
    # validates it
    ref = L0PLAN.load_source_plan()["reference"]

    seeds: List[Optional[int]] = [None] * N_GAMES
    if seed_interval is not None:
        lo, hi = seed_interval
        for name, v in (("lo", lo), ("hi", hi)):
            if type(v) is not int or isinstance(v, bool):
                raise H3StudyError(
                    f"seed interval {name} is {v!r} ({type(v).__name__}); an int "
                    f"is required and no value is coerced into one")
        if hi - lo != N_GAMES:
            raise H3StudyError(
                f"the supplied interval holds {hi - lo} seeds for {N_GAMES} games")
        for s_ in range(lo, hi):
            st = REF.seed_status(s_)
            if st["exposed"] or st["retired"] or st["test_only"]:
                raise H3StudyError(
                    f"seed {s_} is spent or reserved for tests "
                    f"(exposed={st['exposed']} RETIRED={st['retired']} "
                    f"test_only={st['test_only']}); a spent block may not be revived")
        seeds = list(range(lo, hi))

    out: List[Dict[str, Any]] = []
    for op in openings:
        for colour in ("red", "black"):
            i = len(out)
            t = {
                "task_id": f"h3study-{i:03d}-p{op['index']:03d}-inc_{colour}",
                "index": i,
                "pair_id": op["index"],
                "segment": segment_of(op["index"]),
                "stratum": op["stratum"],
                "order": op["order"],
                "incumbent_colour": colour,
                "opening": opening_name(op["index"]),
                "anchor_colour": "black" if colour == "red" else "red",
                "opening_digest": op["digest"],
                "opening_moves": [list(m) for m in op["moves"]],
                "opening_plies": OPENING_PLIES,
                "seed": seeds[i],
                "reference": ref["name"],
                "reference_sha1": ref["sha1"],
                # H2's frozen configuration, unchanged and READ, never retyped
                "selection_mode": H2R.SELECTION_MODE,
                "mcts_sims": H2R.MCTS_SIMS,
                "t1j_mdPly": H2R.T1J_MDPLY,
                "t1j_mdFixedPly": True,
                "ply_cap": H2R.PLY_CAP,
            }
            t["reference_colour"] = REF.reference_colour(t)
            out.append(t)
    if len(out) != N_GAMES:
        raise H3StudyError(f"built {len(out)} tasks, expected {N_GAMES}")
    return out


def task_digest(tasks: Sequence[Dict[str, Any]]) -> str:
    """sha256 over EVERY field of every task, in order. Nothing projected away."""
    import json
    return hashlib.sha256(json.dumps([dict(t) for t in tasks], sort_keys=True,
                                     separators=(",", ":"),
                                     default=str).encode()).hexdigest()
