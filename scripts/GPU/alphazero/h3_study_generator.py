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


def generator_identity() -> Dict[str, Any]:
    """WHAT produced the population — and it is not an engine.

    The co-produced artifact pinned two engine configurations and a toolchain.
    None of that exists here, and recording it anyway would be provenance for a
    process that did not happen. What DOES determine this population is the seed
    base, the attempt ceiling, the ply count and the filter set, so those are
    what the artifact names.
    """
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
    }


def validate_artifact(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """The artifact must carry its whole frozen schema, name ONE population, and
    its digest must be the one its own openings give."""
    missing = [k for k in ARTIFACT_KEYS if k not in doc]
    if missing:
        raise H3GenerationError(f"the artifact is missing {missing}")
    if doc["stratum"] != RULES.STRATUM_UNIFORM:
        raise H3GenerationError(
            f"stratum {doc['stratum']!r}: Amendment 3 closed every stratum but "
            f"{RULES.STRATUM_UNIFORM!r}, and an artifact naming another is not "
            f"this study's population")
    if doc["n"] != RULES.N_PAIRS:
        raise H3GenerationError(
            f"the artifact holds n={doc['n']}; the study is defined over "
            f"{RULES.N_PAIRS} openings and a partial population is not a "
            f"population")
    if len(doc["openings"]) != RULES.N_PAIRS:
        raise H3GenerationError(
            f"{len(doc['openings'])} openings recorded against a claimed "
            f"n={doc['n']}")
    if doc.get("claim") != CLAIM:
        raise H3GenerationError(
            "the artifact's claim is missing or altered; the narrowed "
            "uniform-position claim travels WITH the population, so a reader "
            "who never opens the card still cannot overstate it")
    for o in doc["openings"]:
        gaps = [k for k in OPENING_KEYS if k not in o]
        if gaps:
            raise H3GenerationError(f"opening {o.get('index')} is missing {gaps}")
        if o.get("stub"):
            raise H3GenerationError(
                f"opening {o.get('index')} is a STUB; a placeholder may never be "
                f"pinned or played against")
        if o.get("order") is not None:
            raise H3GenerationError(
                f"opening {o.get('index')} carries an `order`; alternating order "
                f"was removed with the co-produced stratum and a stale field is "
                f"one nothing validates")
        if o.get("stratum") != RULES.STRATUM_UNIFORM:
            raise H3GenerationError(
                f"opening {o.get('index')} is stratum {o.get('stratum')!r}")
        if type(o.get("attempts")) is not int or o["attempts"] < 1:
            raise H3GenerationError(
                f"opening {o.get('index')} records attempts="
                f"{o.get('attempts')!r}; every opening cost at least one "
                f"attempt and the count is evidence about the population's "
                f"conditioning, not an optional detail")
    got = RULES.opening_set_digest(doc["openings"])
    if got != doc["opening_set_digest"]:
        raise H3GenerationError(
            f"the artifact's openings give {got} but it claims "
            f"{doc['opening_set_digest']}; it has been edited")
    return {"n": len(doc["openings"]), "opening_set_digest": got}


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
def build_population(*, n: Optional[int] = None,
                     seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """The study's openings, in STUDY ORDER, with segments stamped.

    UNGATED AND SIDE-EFFECT-FREE. It draws from a PRNG, applies the structural
    filters and returns dictionaries. It writes nothing, loads nothing and starts
    nothing, so the suite calls it freely.
    """
    n = RULES.N_PAIRS if n is None else n
    base = RULES.GEN_SEED_UNIFORM if seed is None else seed
    return RULES.assemble_opening_set(
        RULES.generate_uniform_openings(seed=base, n=n))


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


def write_artifact(*, openings: Sequence[Dict[str, Any]], out_path: str,
                   trace_path: str,
                   deadline_s: Optional[float] = None) -> Dict[str, Any]:
    """Write the artifact and a terminal trace to an EXPLICIT destination.

    🔑 THE DESTINATION IS ALWAYS EXPLICIT — there is no default here. Only
    `freeze_population` knows the official path and it is the only caller that
    passes it. A helper that defaulted to the official destination would leave
    every test one missing argument away from freezing the population.

    🔴 A TERMINAL RECORD IS WRITTEN ON EVERY PATH. A trace that simply stops
    cannot be told from one that never started, so success, refusal, timeout and
    interrupt each leave a durable `generation_end` naming the verdict.
    """
    deadline_s = RULES.GENERATION_DEADLINE_S if deadline_s is None else deadline_s
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    started = time.time()
    trace = os.fdopen(_create_only(trace_path), "w")

    def emit(obj):
        trace.write(json.dumps(obj, sort_keys=True, default=str) + "\n")
        trace.flush()
        os.fsync(trace.fileno())

    verdict, failure, doc = "VOID", None, None
    try:
        try:
            emit({"event": "generation_start", "n": len(openings),
                  "engine_free": True, "deadline_s": deadline_s,
                  "generator": generator_identity()})
            doc = artifact_document(openings)
            elapsed = time.time() - started
            if elapsed > deadline_s:
                # 🔑 EXPECTED TO BE UNREACHABLE BY THREE ORDERS OF MAGNITUDE.
                # If this fires it is a DEFECT REPORT, not a capacity result.
                verdict = "TIMEOUT"
                failure = (f"the {deadline_s}s runaway guard expired after "
                           f"{elapsed:.1f}s writing an ENGINE-FREE population; "
                           f"this is a defect report, not a capacity result")
                raise H3GenerationError(failure)
            with os.fdopen(_create_only(out_path), "w") as fh:
                json.dump(doc, fh, indent=1, sort_keys=True, default=str)
            verdict = "OK"
        except BaseException as e:               # noqa: BLE001 -- interrupts too
            if isinstance(e, KeyboardInterrupt):
                verdict = "INTERRUPTED"
            elif verdict != "TIMEOUT":
                verdict = "VOID"
            failure = failure or f"{type(e).__name__}: {e}"
            raise
        finally:
            emit({"event": "generation_end", "verdict": verdict,
                  "accepted": len(doc["openings"]) if doc else 0,
                  "expected": RULES.N_PAIRS,
                  "elapsed_s": round(time.time() - started, 3),
                  "failure": failure,
                  "opening_set_digest": (doc or {}).get("opening_set_digest"),
                  "attempts_total": sum(o.get("attempts", 0) for o in openings),
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


def freeze_population(*, out_path: Optional[str] = None,
                      trace_path: Optional[str] = None) -> Dict[str, Any]:
    """Write the OFFICIAL artifact. BARRIER FIRST, before anything durable.

    🔴 THE BARRIER IS READ BEFORE THE DESTINATION IS TOUCHED, so a refusal leaves
    no directory, no trace and no partial file behind.
    """
    check_freeze_barrier()
    return write_artifact(openings=build_population(),
                          out_path=DEFAULT_OUT if out_path is None else out_path,
                          trace_path=(DEFAULT_TRACE if trace_path is None
                                      else trace_path))
