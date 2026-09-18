"""H3 FULL STUDY — the UNIFORM population writer. ENGINE-FREE.

🔴 AMENDMENT 3. The co-produced stratum is CLOSED and this module no longer
generates it. Two authorized attempts VOIDed at opening 0 with zero openings
accepted; the second established that T1j cannot move at the plies the protocol
required. The retired engine path is preserved, inert, in
`h3_coproduced_generator_retired` — no active study path may reach it.

🔴 GENERATION IS NO LONGER A RUN, AND THIS MODULE HAS NO EXECUTION GATE.
Producing the population is a PRNG and four structural filters: no checkpoint, no
JVM, no engine, no move requested, no process to contain, nothing to clean up.
The gate, supervisor, containment boundary and launch receipt that guarded the
co-produced generation guarded an ENGINE, and there is no engine here. Keeping
them would be ceremony that looks like a control — the exact shape this programme
keeps finding — so they are gone rather than kept as decoration.

🔑 WHAT REPLACES THEM IS NARROWER AND REAL. Building the population in memory is
free and unrestricted: `build_population()` is a pure function and the test suite
calls it. What is NOT free is declaring one population THE population:

    build_population()   ungated, deterministic, in-memory, no side effect
    write_artifact()     create-only, destination always EXPLICIT
    freeze_population()  the OFFICIAL artifact -> BARRIER + create-only + trace

🔴 AND THE BARRIER EXISTS *BECAUSE* GENERATION IS CHEAP. When recomputing the
population costs 0.3 s, "just regenerate it" becomes an easy way to slide a
different population under a study that has already started. The digest pin is
what makes the population a fact instead of a recipe, so fixing it is a separate
reviewed act — not a side effect of running the generator.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import h3_study_rules as RULES


class H3GenerationError(RuntimeError):
    """A refusal from the population writer. Never a population."""


# ═══════════════════ THE BARRIER: freezing, not generating ═════════════════
#: 🔴 THIS IS NOT AN EXECUTION GATE AND MUST NOT BE DESCRIBED AS ONE. Nothing
#: here executes. It guards ONE act: writing the OFFICIAL artifact whose digest
#: becomes `RULES.OPENING_SET_DIGEST` and therefore DEFINES the study's
#: population. Generating is free (`build_population`); a temporary destination
#: is free (`write_artifact`); only the official write is barred.
H3_POPULATION_FREEZE_AUTHORIZED = False

EXIT_UNAUTHORIZED = 5

#: 🔴 THE OFFICIAL DESTINATION, AND IT MUST BE ABSENT UNTIL THE FREEZE STEP RUNS.
#: Fresh, outside every spent directory. The two co-produced destinations
#: (`...-study-openings` and `...-study-openings-attempt2`) hold VOID records and
#: are spent whole; nothing may write into either again.
OUT_DIR = "docs/superpowers/evidence/2026-09-17-t1j-h3-study-openings-uniform"
DEFAULT_OUT = f"{OUT_DIR}/01_opening_set.json"
DEFAULT_TRACE = f"{OUT_DIR}/02_generation_trace.jsonl"

#: 🔴 THE ARTIFACT'S SCHEMA, frozen before anything writes it. A record whose
#: shape is decided while writing it is a record nobody can check.
#: `selection_mode` and `config_pins` are GONE: no engine generates, so there is
#: no generating configuration to pin, and pinning the MATCH configuration here
#: would state something the artifact cannot know.
ARTIFACT_KEYS = ("design", "stratum", "n", "generation_note", "generator",
                 "claim", "openings", "opening_set_digest")

#: 🔑 `order` IS GONE AND `attempts` IS MANDATORY. Alternating order went with the
#: co-produced stratum. The attempt count stays required because the rejection
#: rate CONDITIONS the population: it is evidence about the population, not a
#: private detail of the loop.
OPENING_KEYS = ("index", "segment", "stratum", "moves", "digest", "seed",
                "attempts")

#: 🔴 THE NARROWED CLAIM, CARRIED IN THE ARTIFACT ITSELF, so a reader who never
#: opens the card cannot mistake what this population supports. Every final
#: report must reproduce it verbatim; `h3_study_analysis` refuses a report that
#: weakens or omits it.
CLAIM = (
    "Over legal six-ply TwixT positions drawn UNIFORMLY AT RANDOM and filtered "
    "only by the structural admissibility rules of card section 1.3, played in "
    "the frozen deterministic argmax configuration. This population says "
    "NOTHING about realistic play, about positions either engine would actually "
    "reach, or about any engine-produced population. A uniformly random six-ply "
    "position is not a position anyone plays."
)

GENERATION_NOTE = (
    "ENGINE-FREE. Uniform random legal play to six plies, whole-position "
    "rejection against the structural filters, one contiguous declared seed "
    "range covering every candidate. No model was loaded, no JVM started, no "
    "engine consulted and no move requested."
)


#: 🔴 THE SOURCES THE WALK DEPENDS ON — AND *ONLY* THOSE.
#:
#: This used to pin `h3_study_rules.py`, which also holds `OPENING_SET_DIGEST`.
#: The freeze sequence is: write the artifact (digest None), then RECORD the
#: digest in that same file. Its hash moved, and `load_opening_set` then refused
#: the population it had just frozen, for source drift caused by its own
#: procedure. **A pin that covers the file it will be written into is a trap, not
#: a pin.**
#:
#: The rule now: AN OUTPUT MAY NEVER LIVE INSIDE A PINNED INPUT. Everything that
#: determines what the walk produces is in `h3_generation_protocol`; everything
#: that records what it produced -- the digest, the retired ranges, the
#: destinations -- is in modules that are NOT pinned.
_IDENTITY_SOURCES = ("h3_generation_protocol.py", "game/twixt_state.py",
                     "d1_selection.py")


def _source_pins() -> Dict[str, str]:
    """sha256 of each source the walk depends on. Read, never retyped."""
    import hashlib
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for rel in _IDENTITY_SOURCES:
        with open(os.path.join(here, rel), "rb") as fh:
            out[rel] = hashlib.sha256(fh.read()).hexdigest()
    return out


def _commit() -> Optional[str]:
    """The commit the population was produced at, or None. Never fabricated."""
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           cwd=os.path.dirname(os.path.abspath(__file__)),
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None if r.returncode == 0 else None


def generator_identity() -> Dict[str, Any]:
    """WHAT produced the population — and it is not an engine.

    The co-produced artifact pinned two engine configurations and a toolchain.
    Neither exists here, and recording them anyway would be provenance for a
    process that did not happen.

    🔴 BUT "NO ENGINE" IS NOT "NO TOOLCHAIN", AND THE FIRST VERSION CONFUSED THE
    TWO. It recorded the seed base, the attempt ceiling and the filter names --
    the DESIGN -- and nothing that would let anyone reproduce the walk. The same
    seed gives a different opening under a different NumPy bit generator, a
    different `legal_moves()` ordering, or a different canonical digest, so the
    artifact now pins the interpreter, NumPy, the bit generator by NAME, the
    three sources the walk depends on by CONTENT, and the commit.
    """
    import sys
    import numpy as np
    return {
        "kind": "uniform_random_legal_play",
        "engine_free": True,
        "gen_seed_base": RULES.GEN_SEED_UNIFORM,
        "seed_range": list(RULES.generation_seed_range(RULES.GEN_SEED_UNIFORM)),
        "max_attempts": RULES.MAX_ATTEMPTS,
        "opening_plies": RULES.OPENING_PLIES,
        "board_size": RULES.BOARD_SIZE,
        "n_pairs": RULES.N_PAIRS,
        "filters": ["legal_and_not_won", "all_pegs_placed",
                    "no_depth1_forced_win", "distinct_up_to_symmetry",
                    "not_a_pilot_or_h1_h2_opening"],
        # ── the toolchain the walk is deterministic UNDER ──
        "python": "%d.%d.%d" % sys.version_info[:3],
        "numpy": np.__version__,
        "bit_generator": "PCG64",
        "source_pins": _source_pins(),
        # 🔑 THE EXCLUSION SET BY VALUE, NOT BY SOURCE. The population depends on
        # WHICH 28 digests are excluded. Pinning `h3_pilot_rules.py` would pin the
        # code that computes them, so a refactor leaving them identical would
        # invalidate a frozen population and a change that moved one might not be
        # noticed at all. The resolved set is the thing that matters.
        "excluded_digest_set": RULES.excluded_digest_set_pin(),
        "commit": _commit(),
    }


#: 🔴 ENFORCED: the artifact is REFUSED if any of these disagrees with the
#: running code, because each one changes what a seed produces.
#:
#: `numpy` IS ENFORCED, deliberately. NumPy guarantees stream compatibility for
#: the legacy `RandomState` and explicitly does NOT for `Generator`/`PCG64`, so a
#: NumPy upgrade may silently change every opening. Refusing on a version bump is
#: fail-closed and forces a human to check; accepting would let the study play a
#: population it can no longer re-derive.
IDENTITY_MUST_MATCH = ("kind", "engine_free", "gen_seed_base", "seed_range",
                       "max_attempts", "opening_plies", "board_size", "n_pairs",
                       "filters", "bit_generator", "source_pins",
                       "excluded_digest_set", "numpy")

#: 🔑 RECORDED ONLY, and named EXPLICITLY so the split is a stated policy rather
#: than an omission. Every identity field is in exactly one of these two sets --
#: a test asserts that, so a new field cannot arrive unclassified.
#:
#: `python`  -- the walk does not depend on it. `legal_moves()` builds its list
#:              with nested `range` loops, so its order does not vary with the
#:              interpreter's hashing, and `PCG64` is NumPy's. Enforcing it would
#:              refuse a population on a patch bump that cannot change a move.
#: `commit`  -- a frozen population stays valid across later commits that do not
#:              touch the walk. `source_pins` refuses exactly when it changed,
#:              which is the honest version of the same check.
IDENTITY_RECORDED_ONLY = ("python", "commit")


def _typed_moves(o) -> list:
    """`moves` as a list of six (int, int) pairs, TYPE-STRICTLY.

    🔴 `True == 1` AND `6 == 6.0` IN PYTHON, so a bool or a float in a
    coordinate would replay, compare equal, and produce the declared digest --
    while the artifact says something the study cannot reproduce from JSON. Type
    is checked before value, everywhere.
    """
    mv = o.get("moves")
    if not isinstance(mv, list) or len(mv) != RULES.OPENING_PLIES:
        raise H3GenerationError(
            f"opening {o.get('index')}: moves must be a list of "
            f"{RULES.OPENING_PLIES}, got {mv!r}")
    out = []
    for m in mv:
        if not isinstance(m, (list, tuple)) or len(m) != 2:
            raise H3GenerationError(
                f"opening {o.get('index')}: move {m!r} is not a pair")
        r, c = m
        for v in (r, c):
            if type(v) is not int or isinstance(v, bool):
                raise H3GenerationError(
                    f"opening {o.get('index')}: coordinate {v!r} is "
                    f"{type(v).__name__}, not int -- `True == 1` and `6 == 6.0`, "
                    f"so an equal value is not the same value")
            if not 0 <= v < RULES.BOARD_SIZE:
                raise H3GenerationError(
                    f"opening {o.get('index')}: coordinate {v} is off a "
                    f"{RULES.BOARD_SIZE}-point board")
        out.append((r, c))
    return out


def validate_artifact(doc: Mapping[str, Any], *,
                      bind_provenance: bool = True) -> Dict[str, Any]:
    """The artifact must BE the population it claims to be.

    🔴 THE FIRST VERSION TRUSTED EVERY ROW'S OWN `digest`. It recomputed
    `opening_set_digest` -- a hash OF THE DECLARED DIGESTS -- and never replayed a
    single move. An opening's `moves` could therefore be rewritten while its
    `digest` and the set digest stayed untouched, and the runner would accept and
    play a different position. A digest that is never recomputed from the thing
    it digests is a label, not a checksum.

    So validation now BINDS, in this order:

      schema      exact artifact keys and exact opening keys, nothing extra
      position    index and segment are the study's, not the artifact's opinion
      replay      six moves, type-strict, replayed through the REAL engine
      digest      recomputed canonical digest, compared to the declared one
      provenance  seed == attempt_seed(base, index, attempts - 1)
      derivation  the PRNG walk from that seed REPRODUCES these exact moves
      identity    claim, generation note and generator identity, exactly

    `bind_provenance=False` drops only the last two rows of that list, for the
    one case where they cannot hold: a test constructing a deliberately broken
    artifact. It never relaxes schema, replay or digest.
    """
    missing = [k for k in ARTIFACT_KEYS if k not in doc]
    if missing:
        raise H3GenerationError(f"the artifact is missing {missing}")
    extra = [k for k in doc if k not in ARTIFACT_KEYS]
    if extra:
        raise H3GenerationError(
            f"the artifact carries unknown keys {extra}; the schema is exact so "
            f"a field nothing validates cannot ride along")
    if doc["stratum"] != RULES.STRATUM_UNIFORM:
        raise H3GenerationError(
            f"stratum {doc['stratum']!r}: Amendment 3 closed every stratum but "
            f"{RULES.STRATUM_UNIFORM!r}, and an artifact naming another is not "
            f"this study's population")
    if type(doc["n"]) is not int or doc["n"] != RULES.N_PAIRS:
        raise H3GenerationError(
            f"the artifact holds n={doc['n']!r}; the study is defined over "
            f"{RULES.N_PAIRS} openings and a partial population is not a "
            f"population")
    openings = doc["openings"]
    if not isinstance(openings, list) or len(openings) != RULES.N_PAIRS:
        raise H3GenerationError(
            f"{len(openings) if isinstance(openings, list) else openings!r} "
            f"openings recorded against a claimed n={doc['n']}")
    if doc.get("claim") != CLAIM:
        raise H3GenerationError(
            "the artifact's claim is missing or altered; the narrowed "
            "uniform-position claim travels WITH the population, so a reader "
            "who never opens the card still cannot overstate it")
    if doc.get("generation_note") != GENERATION_NOTE:
        raise H3GenerationError(
            "the artifact's generation note is missing or altered")

    ident = doc.get("generator")
    if not isinstance(ident, Mapping):
        raise H3GenerationError("the artifact carries no generator identity")
    live = generator_identity()
    drift = [k for k in IDENTITY_MUST_MATCH if ident.get(k) != live[k]]
    if drift:
        raise H3GenerationError(
            f"the generator identity disagrees with the running code on {drift}. "
            f"The same seed gives a DIFFERENT opening under a different walk, "
            f"bit generator or engine, so this population cannot be reproduced "
            f"here and may not be played.")
    base = ident["gen_seed_base"]

    for pos, o in enumerate(openings):
        if not isinstance(o, Mapping):
            raise H3GenerationError(f"opening at position {pos} is not a record")
        gaps = [k for k in OPENING_KEYS if k not in o]
        if gaps:
            raise H3GenerationError(f"opening {o.get('index')} is missing {gaps}")
        spare = [k for k in o if k not in OPENING_KEYS]
        if spare:
            raise H3GenerationError(
                f"opening {o.get('index')} carries unknown keys {spare}; "
                f"`order` and `stub` are gone, and a field nothing validates is "
                f"a field that can say anything")
        if o.get("stratum") != RULES.STRATUM_UNIFORM:
            raise H3GenerationError(
                f"opening {o.get('index')} is stratum {o.get('stratum')!r}")
        # ── position: the study's, not the artifact's opinion ──
        if type(o["index"]) is not int or isinstance(o["index"], bool) \
                or o["index"] != pos:
            raise H3GenerationError(
                f"opening at position {pos} declares index {o['index']!r}; the "
                f"study order is positional and a reordered set is a different "
                f"population")
        if o["segment"] != RULES.segment_of(pos):
            raise H3GenerationError(
                f"opening {pos} declares segment {o['segment']!r}, but the "
                f"frozen plan puts it in segment {RULES.segment_of(pos)}")
        if type(o["attempts"]) is not int or isinstance(o["attempts"], bool) \
                or o["attempts"] < 1:
            raise H3GenerationError(
                f"opening {pos} records attempts={o['attempts']!r}; every "
                f"opening cost at least one attempt and the count is evidence "
                f"about the population's conditioning, not an optional detail")
        if type(o["seed"]) is not int or isinstance(o["seed"], bool):
            raise H3GenerationError(
                f"opening {pos} records seed={o['seed']!r}, which is not an int")

        # ── replay: through the REAL engine, and recompute the digest ──
        moves = _typed_moves(o)
        try:
            st = RULES._replay(moves)
        except Exception as e:                            # noqa: BLE001
            raise H3GenerationError(
                f"opening {pos}: its moves are NOT LEGAL on a real board "
                f"({type(e).__name__}: {e})") from None
        from . import d1_selection as SEL
        recomputed = SEL.canonical_digest(st)
        if recomputed != o["digest"]:
            raise H3GenerationError(
                f"opening {pos}: replaying its moves gives canonical digest "
                f"{recomputed} but the record claims {o['digest']}. The moves "
                f"have been changed and the digest left behind -- which is "
                f"invisible to any check that only re-hashes the declared "
                f"digests.")

        if bind_provenance:
            # ── provenance: the seed is the one its own attempt count implies ──
            want_seed = RULES.attempt_seed(base, pos, o["attempts"] - 1)
            if o["seed"] != want_seed:
                raise H3GenerationError(
                    f"opening {pos} records seed {o['seed']} but "
                    f"attempt_seed({base}, {pos}, {o['attempts'] - 1}) is "
                    f"{want_seed}; the seed and the attempt count disagree")
            # ── derivation: the walk from that seed REPRODUCES these moves ──
            derived_moves, derived_digest = RULES.verify_candidate(
                base, pos, o["attempts"])
            if [tuple(m) for m in derived_moves] != moves:
                raise H3GenerationError(
                    f"opening {pos}: re-running the PRNG walk from seed "
                    f"{want_seed} does NOT produce the recorded moves. The "
                    f"population cannot be re-derived from its own provenance.")
            if derived_digest != o["digest"]:
                raise H3GenerationError(
                    f"opening {pos}: the re-derived digest is {derived_digest}, "
                    f"not {o['digest']}")

    # ── and only now the set digest, over digests every one of which was
    # ── recomputed from replayed moves above
    got = RULES.opening_set_digest(openings)
    if got != doc["opening_set_digest"]:
        raise H3GenerationError(
            f"the artifact's openings give {got} but it claims "
            f"{doc['opening_set_digest']}; it has been edited")
    if len({o["digest"] for o in openings}) != len(openings):
        raise H3GenerationError("two openings share a canonical digest")
    clash = {o["digest"] for o in openings} & RULES.excluded_digests()
    if clash:
        raise H3GenerationError(
            f"{len(clash)} openings duplicate a PILOT or H1/H2 position")
    return {"n": len(openings), "opening_set_digest": got,
            "bound": bool(bind_provenance)}


def check_freeze_barrier() -> None:
    """Read the barrier, FIRST, before anything durable."""
    if H3_POPULATION_FREEZE_AUTHORIZED is not True:
        raise H3GenerationError(
            "freezing the H3 population is NOT AUTHORIZED "
            "(H3_POPULATION_FREEZE_AUTHORIZED is False). Generating openings in "
            "memory is free and needs no permission; declaring one set THE "
            "population -- writing the official artifact and fixing "
            "OPENING_SET_DIGEST -- is a separate reviewed step, because a "
            "population that can be recomputed on demand can also be swapped.")


# ═══════════════════════ the population itself ═════════════════════════════
def build_population(check_deadline=None, *, n: Optional[int] = None,
                     seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """The study's openings, in STUDY ORDER, with segments stamped.

    UNGATED AND SIDE-EFFECT-FREE. It draws from a PRNG, applies the structural
    filters and returns dictionaries. It writes nothing, loads nothing and starts
    nothing, so the suite calls it freely.

    🔴 `check_deadline` IS POSITIONAL AND FIRST, so `build=build_population`
    threads it with no lambda in between. A lambda that took the hook and dropped
    it -- `lambda cd: build_population()` -- is how the guard ends up checked only
    after the walk it was meant to bound, which is the shape this repair exists
    to remove. It is passed straight through to the candidate loop.
    """
    n = RULES.N_PAIRS if n is None else n
    base = RULES.GEN_SEED_UNIFORM if seed is None else seed
    return RULES.assemble_opening_set(
        RULES.generate_uniform_openings(seed=base, n=n,
                                        check_deadline=check_deadline))


def artifact_document(openings: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The artifact, assembled and VALIDATED — but written nowhere."""
    doc = {
        "design": "H3_FULL_STUDY_OPENINGS",
        "stratum": RULES.STRATUM_UNIFORM,
        "n": len(openings),
        "generation_note": GENERATION_NOTE,
        "generator": generator_identity(),
        "claim": CLAIM,
        "openings": [{k: v for k, v in o.items() if k != "state"}
                     for o in openings],
        "opening_set_digest": RULES.opening_set_digest(openings),
    }
    validate_artifact(doc)
    return doc


def _create_only(path: str) -> int:
    """`O_EXCL` plus an explicit `lexists`, so a DANGLING SYMLINK cannot be
    written through. `O_EXCL` alone fails on one, but with an error that cannot
    be told from a real collision; checking first names the cause."""
    if os.path.lexists(path):
        raise H3GenerationError(
            f"{path} already exists (or is a symlink, possibly dangling). "
            f"Outputs are CREATE-ONLY: an existing path is never overwritten, "
            f"never appended to and never followed.")
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)


class H3GenerationDeadline(H3GenerationError):
    """The runaway guard fired. 🔑 ITS OWN TYPE, so a timeout cannot be reported
    as an ordinary refusal and the trace cannot call it VOID."""


def write_artifact(*, out_path: str, trace_path: str,
                   build=None, openings: Optional[Sequence[Dict[str, Any]]] = None,
                   deadline_s: Optional[float] = None) -> Dict[str, Any]:
    """Build the population and write it, with the trace and the clock around
    BOTH. The destination is always EXPLICIT.

    🔴 GENERATION USED TO HAPPEN OUTSIDE THIS FUNCTION ENTIRELY. The caller wrote
    `write_artifact(openings=build_population(), ...)`, and Python evaluates the
    argument first -- so the whole walk ran BEFORE the trace was opened and before
    the clock started. A hang or a crash inside generation produced NO terminal
    record and could not trip the runaway guard the card promises. The guard was
    real, and it guarded only the part that never takes any time.

    So `build` is a CALLABLE taking `check_deadline`, invoked here, after the
    trace exists and the clock is running. The deadline is checked INSIDE the
    candidate loop, not between openings.

    `openings=` remains for the one case that has no walk to time: a test writing
    a set it constructed itself. It is mutually exclusive with `build`.

    🔴 A TERMINAL RECORD IS WRITTEN ON EVERY PATH. A trace that simply stops
    cannot be told from one that never started, so success, refusal, timeout and
    interrupt each leave a durable `generation_end` naming the verdict.
    """
    if (build is None) == (openings is None):
        raise H3GenerationError("pass exactly one of `build` or `openings`")
    deadline_s = RULES.GENERATION_DEADLINE_S if deadline_s is None else deadline_s
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 🔑 MONOTONIC. `time.time()` can step backwards over an NTP correction or a
    # DST change, which would silently extend or collapse the window a runaway
    # guard exists to bound.
    started = time.monotonic()
    trace = os.fdopen(_create_only(trace_path), "w")

    def emit(obj):
        trace.write(json.dumps(obj, sort_keys=True, default=str) + "\n")
        trace.flush()
        os.fsync(trace.fileno())

    def check_deadline(index, attempt, accepted):
        spent = time.monotonic() - started
        if spent > deadline_s:
            raise H3GenerationDeadline(
                f"the {deadline_s}s runaway guard expired after {spent:.1f}s at "
                f"opening {index}, attempt {attempt} ({accepted} accepted) of an "
                f"ENGINE-FREE generation. This is a DEFECT REPORT, not a "
                f"capacity result: the whole population builds in under a "
                f"second.")

    verdict, failure, doc, built = "VOID", None, None, None
    try:
        try:
            emit({"event": "generation_start", "n": RULES.N_PAIRS,
                  "engine_free": True, "deadline_s": deadline_s,
                  "clock": "monotonic",
                  "generates_here": build is not None,
                  "generator": generator_identity()})
            built = list(openings) if openings is not None else build(check_deadline)
            emit({"event": "population_built", "accepted": len(built),
                  "elapsed_s": round(time.monotonic() - started, 3)})
            doc = artifact_document(built)
            check_deadline(RULES.N_PAIRS, 0, len(built))     # and after, too
            fd = _create_only(out_path)
            with os.fdopen(fd, "w") as fh:
                json.dump(doc, fh, indent=1, sort_keys=True, default=str)
                fh.flush()
                # 🔴 FSYNC BEFORE THE VERDICT. `OK` in the trace asserts the
                # artifact is ON DISK. Without this the process could report OK
                # and lose the file to a crash, leaving a trace that swears to a
                # population nobody has.
                os.fsync(fh.fileno())
            verdict = "OK"
        except BaseException as e:               # noqa: BLE001 -- interrupts too
            if isinstance(e, KeyboardInterrupt):
                verdict = "INTERRUPTED"
            elif isinstance(e, H3GenerationDeadline):
                verdict = "TIMEOUT"
            else:
                verdict = "VOID"
            failure = f"{type(e).__name__}: {e}"
            raise
        finally:
            emit({"event": "generation_end", "verdict": verdict,
                  "accepted": len(built) if built else 0,
                  "expected": RULES.N_PAIRS,
                  "elapsed_s": round(time.monotonic() - started, 3),
                  "failure": failure,
                  "opening_set_digest": (doc or {}).get("opening_set_digest"),
                  "attempts_total": sum(o.get("attempts", 0) for o in (built or [])),
                  # 🔴 NO RANGE IS RETIRED BY A WRITE, and this is the one place
                  # the uniform path differs from the co-produced one, so it is
                  # said out loud. A co-produced attempt retired its WHOLE range
                  # because it built an agent and queried an engine from a drawn
                  # seed -- irreversible. Nothing here draws in that sense: the
                  # same seeds give the same positions every time, which is
                  # exactly what makes the digest a pin rather than a snapshot.
                  "retires": None,
                  "retirement_rule":
                      "none: engine-free generation is deterministic and "
                      "repeatable. The range is committed by the FREEZE, not by "
                      "a write.",
                  })
            trace.close()
    except H3GenerationError:
        raise
    if verdict != "OK":
        raise H3GenerationError(failure or verdict)
    return doc


def freeze_population() -> Dict[str, Any]:
    """Write the OFFICIAL artifact. NO ARGUMENTS, and that is the repair.

    🔴 IT USED TO ACCEPT `out_path` AND `trace_path`. With the barrier open, a
    caller could therefore write any number of different "official" populations
    to any number of destinations -- and the barrier, never restored, stayed open
    for all of them. An entry that takes a destination is an entry whose
    authorization does not name what it authorizes.

    Every input is resolved here: the destination is `DEFAULT_OUT`, the
    population is `build_population`, and the deadline is the frozen constant.
    Temporary writing lives in `write_artifact`, which is not this.

    🔑 THE BARRIER IS NOT RESTORED HERE. This function must not close the thing
    that let it run -- a body that reopens and re-closes its own authorization is
    a body that can be re-entered. `h3_freeze_command` opens nothing, calls this
    once, and restores and VERIFIES the barrier in a `finally`.
    """
    check_freeze_barrier()
    return write_artifact(out_path=DEFAULT_OUT, trace_path=DEFAULT_TRACE,
                          build=build_population)
