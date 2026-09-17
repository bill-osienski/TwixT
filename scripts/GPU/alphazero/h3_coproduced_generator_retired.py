"""🔴 RETIRED 2026-09-17. THE CO-PRODUCED OPENING GENERATOR. NEVER REOPENABLE.

**This module is history, not machinery.** It is preserved so the record of what
was built, why it failed and what it cost stays in the tree rather than only in
git. NO ACTIVE STUDY PATH MAY IMPORT IT, and
`tests/test_h3_uniform_population.py` asserts that none does.

WHAT IT WAS. The generator for stratum B: the incumbent and T1j ALTERNATING for
six plies, both orders half and half, behind its own gate, supervisor and
containment boundary.

WHY IT IS RETIRED — and it is not because the design was disliked:

  attempt 1, 2026-09-16, `[20261200000, 20261259200)`
      VOID at opening 0. `KeyError: None` -- a bare `IntegrationContext` whose
      `bump` had no bucket. 0 openings accepted. Repaired by `ctx.reset(...)`,
      and the repair WORKED.
  attempt 2, 2026-09-16, `[20261400000, 20261459200)`
      VOID at opening 0. T1j `exit 3` at board-ply 1:
      `FAIL q1: requested depth 6 completed ... failures=1`. 0 accepted.

🔴 ATTEMPT 2 IS THE ONE THAT MATTERS, AND IT WAS NOT A BUG. It is the 2026-08-31
low-ply qualification, this programme's own recorded evidence: **T1j never enters
alpha-beta at plies 1 and 3** (12/12 fail at each; ply 5 completes 0/12). The
protocol below asks T1j to move at 1/3/5 stones under `incumbent_first` and at
0/2/4 under `t1j_first` -- precisely where it was already MEASURED unable to
search. The design was impossible as specified, and that was discoverable without
running anything.

⚠ THE LESSON, KEPT WITH THE CODE. Freezing this protocol DID catch a real defect
by inspection -- the entropy finding (argmax + deterministic T1j = one opening per
order). That success is what hid the second miss: the freeze checked the
generator for ENTROPY and never checked it against the QUALIFICATION RECORD of
the engine it scheduled. Frozen and well-defined is not feasible. Cost: two seed
ranges and two destinations, all spent whole.

Delaying T1j's first move to ply 5+ would not have rescued it -- that defines a
DIFFERENT opening population, needing its own pilot. The stratum is CLOSED.

The live population writer is `h3_study_generator`, and it is engine-free.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import h3_study_rules as RULES


class H3GenerationError(RuntimeError):
    """A refusal from the generator. Never a population."""


class H3GenerationContainmentError(H3GenerationError):
    """The generation seam was reached from a test process."""


class H3GenerationCleanupError(H3GenerationError):
    """Teardown failed. 🔑 ITS OWN TERMINAL OUTCOME, never a footnote to the
    body's: a run that produced a population and then left a JVM alive has not
    succeeded, and the wrapper gives this priority over the worker's result."""


# ═══════════════════════ BARRIER 1: its OWN gate ═══════════════════════════
#: 🔴 THE RETIRED STRATUM'S OWN CONSTANTS, MOVED HERE FROM `h3_study_rules`.
#: They describe a population that can no longer be produced, so leaving them in
#: the live rules module would be dead design constants sitting beside live ones
#: -- and the next reader could not tell which was which.
STRATUM_CO_PRODUCED = "co_produced"
PAIRS_PER_STRATUM = 148
STRATUM_PAIRS_PER_SEGMENT = 37
INCUMBENT_FIRST_PER_SEGMENT = (19, 18, 19, 18)
T1J_FIRST_PER_SEGMENT = (18, 19, 18, 19)
ORDER_INCUMBENT_FIRST = "incumbent_first"
ORDER_T1J_FIRST = "t1j_first"

#: both SPENT WHOLE 2026-09-16, both VOID at opening 0, 0 openings accepted
GEN_SEED_CO_PRODUCED = 20_261_400_000
GEN_SEED_CO_PRODUCED_ATTEMPT1 = 20_261_200_000


def generation_seed_range(base: int):
    """The retired stratum's range: 148 openings x 400 attempts."""
    return (base, base + PAIRS_PER_STRATUM * RULES.MAX_ATTEMPTS)


def generation_config():
    """The FROZEN RESEARCH configuration the retired generator sampled under.

    🔑 KEPT BECAUSE THE ENTROPY FINDING IS WORTH KEEPING. Under the match's
    argmax the incumbent's move is a deterministic function of the position and
    E3a proved T1j deterministic, so an alternating generator built from the two
    PLAYERS had no entropy anywhere in it: one opening per order, not 74.
    """
    from . import twixtbot_g3_reference as G3
    cfg = G3.eval_config()
    if cfg.selection_mode == "argmax":
        raise H3GenerationError(
            "the frozen research configuration is already argmax, so generation "
            "would have no entropy and the stratum would be two positions")
    if cfg.opening_temp_plies < RULES.OPENING_PLIES:
        raise H3GenerationError(
            f"opening_temp_plies={cfg.opening_temp_plies} does not cover all "
            f"{RULES.OPENING_PLIES} generated plies")
    return cfg


#: 🔴 THERE IS NO GATE HERE ANY MORE, AND ITS ABSENCE IS THE POINT.
#: `H3_GENERATION_AUTHORIZED = False` used to sit here. `check_gate` stopped
#: reading it when the stratum was retired, which left a constant that LOOKED
#: like a control and bound nothing -- this programme's recurring defect wearing
#: the costume of a safety feature. It is GONE rather than left False: a `False`
#: constant invites a reviewer to set it True, and no review can make T1j search
#: at ply 1. The refusal below is unconditional and there is nothing to flip.

EXIT_UNAUTHORIZED = 5

#: Create-only, and never inside a spent run's directory.
#: 🔴 ATTEMPT 2. Attempt 1's directory
#: (`2026-09-15-t1j-h3-study-openings`) holds a VOID's records -- a launch
#: receipt, a terminal trace and a compiled helper -- and is now in
#: SPENT_OUT_DIRS. Nothing may write into it again, and this run must not try.
OUT_DIR = "docs/superpowers/evidence/2026-09-16-t1j-h3-study-openings-attempt2"
DEFAULT_OUT = f"{OUT_DIR}/01_opening_set.json"
DEFAULT_TRACE = f"{OUT_DIR}/02_generation_trace.jsonl"


#: 🔴 THE ARTIFACT'S SCHEMA, frozen before the run that writes it. A record whose
#: shape is decided while writing it is a record nobody can check.
ARTIFACT_KEYS = ("design", "stratum", "n", "selection_mode", "generation_note",
                 "config_pins", "toolchain", "openings", "opening_set_digest")
OPENING_KEYS = ("index", "stratum", "order", "stub", "moves", "digest", "seed",
                "attempts")

#: The configuration fields the artifact must pin, so a reader can tell WHICH
#: player produced the population without re-deriving anything.
CONFIG_PIN_FIELDS = ("board_size", "mcts_sims", "mcts_eval_batch_size",
                     "mcts_stall_flush_sims", "selection_mode",
                     "opening_temp_plies", "temp_high", "temp_low", "max_moves")


def config_pins(config) -> Dict[str, Any]:
    """The generating configuration, read off the OBJECT that will generate."""
    return {f: getattr(config, f) for f in CONFIG_PIN_FIELDS}


def toolchain_identity(paths: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The jar and JDK the generation used, by their VERIFIED pins.

    `t1j_toolchain.verified_paths` hashes both and refuses a mismatch; it starts
    no process. Recording its answer means the artifact names the T1j that helped
    produce the population, not merely 'T1j'.
    """
    from . import t1j_toolchain as TC
    p = dict(paths if paths is not None else TC.verified_paths())
    return {"jar": p.get("jar"), "jdk_home": p.get("jdk_home"),
            "root": p.get("root"), "verified": p.get("verified"),
            "t1j_mdPly": RULES.t1j_depth(),
            "reference": RULES.reference_identity()}


def validate_artifact(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """The artifact must carry its whole frozen schema, and its digest must be
    the one its own openings give."""
    missing = [k for k in ARTIFACT_KEYS if k not in doc]
    if missing:
        raise H3GenerationError(f"the artifact is missing {missing}")
    if doc["stratum"] != STRATUM_CO_PRODUCED:
        raise H3GenerationError(f"stratum {doc['stratum']!r}")
    if doc["selection_mode"] == "argmax":
        raise H3GenerationError(
            "the artifact says it was generated under argmax, which has no "
            "entropy: that population would be one position per order")
    for o in doc["openings"]:
        gaps = [k for k in OPENING_KEYS if k not in o]
        if gaps:
            raise H3GenerationError(f"opening {o.get('index')} is missing {gaps}")
        if o.get("stub"):
            raise H3GenerationError(
                f"opening {o.get('index')} is a STUB; a placeholder may never be "
                f"pinned or played against")
    got = RULES.opening_set_digest(doc["openings"])
    if got != doc["opening_set_digest"]:
        raise H3GenerationError(
            f"the artifact's openings give {got} but it claims "
            f"{doc['opening_set_digest']}; it has been edited")
    return {"n": len(doc["openings"]), "opening_set_digest": got}


def check_gate() -> None:
    """🔴 REFUSES UNCONDITIONALLY, AND DOES NOT READ THE GATE.

    A gate is a switch someone may flip after review. This is not one: no review
    can make T1j search at ply 1. Reading `H3_GENERATION_AUTHORIZED` here would
    leave a reopenable path, so the refusal does not depend on it -- flipping the
    constant changes nothing, which a control proves.
    """
    raise H3GenerationError(
        "the co-produced opening generation is RETIRED, not merely unauthorized. "
        "Two attempts VOIDed at opening 0 and the second showed why: T1j does "
        "not enter alpha-beta at plies 1 and 3 (low-ply qualification, "
        "2026-08-31), and this protocol requires it to move exactly there. Both "
        "seed ranges and both destinations are spent whole. The study's "
        "population is UNIFORM and is produced by `h3_study_generator`, which is "
        "engine-free.")


# ═══════════════════════ the alternating protocol ══════════════════════════
def mover_at_ply(order: str, ply: int) -> str:
    """Which ENGINE moves at a 1-based ply, under `order` (card §1.7.6).

    The incumbent takes the odd plies under `incumbent_first` and the even ones
    under `t1j_first`; running both orders half and half is what balances each
    engine's ROLE across the stratum.
    """
    if order not in (ORDER_INCUMBENT_FIRST, ORDER_T1J_FIRST):
        raise H3GenerationError(f"order {order!r}")
    if type(ply) is not int or isinstance(ply, bool) or not 1 <= ply <= RULES.OPENING_PLIES:
        raise H3GenerationError(f"ply {ply!r} is outside 1..{RULES.OPENING_PLIES}")
    incumbent_moves_first = (order == ORDER_INCUMBENT_FIRST)
    odd = (ply % 2 == 1)
    return "incumbent" if (odd == incumbent_moves_first) else "t1j"


def order_for_index(index: int) -> str:
    """The DECLARED allocation: which order opening `index` of the stratum uses.

    The first `sum(INCUMBENT_FIRST_PER_SEGMENT)` are incumbent-first. The
    per-segment split is applied by `assemble_opening_set`, which is where the
    odd 37 is resolved; this only has to produce the right TOTALS.
    """
    if type(index) is not int or isinstance(index, bool) or not (
            0 <= index < PAIRS_PER_STRATUM):
        raise H3GenerationError(f"index {index!r}")
    return (ORDER_INCUMBENT_FIRST
            if index < sum(INCUMBENT_FIRST_PER_SEGMENT)
            else ORDER_T1J_FIRST)


def generate_co_produced(*, out_path: str = DEFAULT_OUT,
                         trace_path: str = DEFAULT_TRACE) -> Dict[str, Any]:
    """THE PUBLIC ENTRY. Produces the 148 co-produced openings and pins them.

    🔴 EVERY INPUT IS RESOLVED HERE. The seeds are the declared generation
    constants, the configurations come from `h3_study_rules`, and the count and
    order allocation are the card's — so an opened gate cannot authorize a
    caller-supplied population.
    """
    check_gate()
    from . import d1_probe as D1
    from . import e4_screen_command as SCREEN_CMD
    from . import t1j_toolchain as TC
    from . import twixtbot_g3_reference as G3

    # 🔴 THE BOUNDARY IS INLINE, NOT BEHIND A HELPER. It was one call away in
    # `_assert_inert()`, and `test_NO_CONTROL_DELETES_AN_AUTHORIZATION_CHECK`
    # could not SEE it: that test finds the innermost function calling
    # `verified_paths` and looks for the boundary IN IT, structurally, precisely
    # so a boundary cannot be taken on trust. An indirection the checker cannot
    # follow is, to the checker, no boundary at all.
    try:
        SCREEN_CMD.assert_production_acts_are_inert(
            "H3 study's OPENING GENERATION", (
                (TC, "verified_paths"),
                (D1, "_default_compile"),
                (SCREEN_CMD, "_default_load_evaluator"),
                (G3, "build_reference_agent")))
    except SCREEN_CMD.ContainmentError as e:
        raise H3GenerationContainmentError(str(e)) from None

    if os.path.lexists(out_path) or os.path.lexists(trace_path):
        raise H3GenerationError(
            f"an output already exists ({out_path}, {trace_path}); the opening "
            f"set is create-only and a second generation would silently replace "
            f"the population the study is defined over")
    cfg = generation_config()          # REFUSES to be argmax
    tc = TC.verified_paths()
    java = os.path.join(tc["jdk_home"], "bin", "java")
    classes = out_path + ".t1j_classes"
    from . import e4_screen_integration as INT
    from . import h2_match_rules as H2R
    deadline = D1.Deadline(RULES.GENERATION_DEADLINE_S)
    deadline.start()
    D1._default_compile(deadline, paths=D1.T1jPaths(
        java=java, jar=tc["jar"], classes=classes, ply_cap=H2R.PLY_CAP))
    evaluator = SCREEN_CMD._default_load_evaluator(".")
    runtime = INT.T1jRuntime(java=java, jar=tc["jar"], classes=classes,
                             ply_cap=H2R.PLY_CAP,
                             timeout_s=RULES.PER_CALL_TIMEOUT_S)
    movers = production_movers(evaluator=evaluator, runtime=runtime, config=cfg)
    return _generate_unguarded(movers=movers, out_path=out_path,
                               trace_path=trace_path,
                               cleanup=SCREEN_CMD._default_cleanup,
                               deadline=deadline,
                               deadline_s=RULES.GENERATION_DEADLINE_S)


def incumbent_colour(order: str) -> str:
    """The colour the INCUMBENT plays for a whole opening.

    Plies 1/3/5 are red and 2/4/6 black, so the alternating order fixes each
    engine's colour for the entire opening — which is what lets ONE agent serve
    all three of its moves, streams advancing, as the card requires.
    """
    if order not in (ORDER_INCUMBENT_FIRST, ORDER_T1J_FIRST):
        raise H3GenerationError(f"order {order!r}")
    return "red" if order == ORDER_INCUMBENT_FIRST else "black"


def production_movers(*, evaluator, runtime, config) -> Dict[str, Any]:
    """FACTORIES for the two real agents, plus the move-log context.

    🔴 FACTORIES, NOT PER-MOVE FUNCTIONS, and the difference is not stylistic.
    The card gives each opening ONE agent whose streams advance across its plies
    (§1.7.4). My first version built a FRESH agent every ply and, to stop the
    three moves being identical, offset the seed by `ply * 7919` — which pushed
    the seeds 47,514 BEYOND the declared generation range. That overrun is larger
    than the 40,800-seed gap between the two strata's ranges, so a collision proof
    over the DECLARED ranges would have passed while the two strata quietly shared
    seeds. Preparing the proof is what found it.

    SEPARATE FROM THE WALK so the walk can be driven with inert factories.
    """
    from . import e4_screen_integration as INT
    from . import twixtbot_g3_reference as G3
    ref = RULES.reference_identity()
    depth = RULES.t1j_depth()

    def incumbent_agent(*, seed: int, colour: str):
        """ONE agent for the whole opening, seeded with the ATTEMPT SEED EXACTLY
        — no offset, so `generation_seed_range` describes what is touched."""
        return G3.build_reference_agent(
            task={"seed": seed, "anchor_colour": ("black" if colour == "red"
                                                  else "red"),
                  "reference": ref["name"], "reference_sha1": ref["sha1"]},
            evaluator=evaluator, colour=colour, config=config, capture=False)

    def t1j_agent(*, colour: str, ctx):
        return INT.T1jAgent(runtime=runtime, ctx=ctx, depth=depth, colour=colour,
                            timeout_s=RULES.PER_CALL_TIMEOUT_S)

    return {"incumbent_agent": incumbent_agent, "t1j_agent": t1j_agent,
            "new_context": INT.IntegrationContext, "config": config,
            "toolchain": toolchain_identity()}


def generate_one(*, index: int, order: str, movers: Dict[str, Any]) -> Dict[str, Any]:
    """ONE co-produced opening: six plies, the engines alternating.

    🔑 A FRESH AGENT PER OPENING AND NO TREE CARRIED ACROSS OPENINGS (card
    §1.7.4). Reusing a search tree would make opening i+1 depend on opening i and
    the population would become order-dependent — a coupling worse than any the
    interval already declares.

    WHOLE-POSITION REJECTION: a candidate failing a filter is discarded entire and
    the NEXT attempt seed is used, so a rejection never re-draws the seed that
    caused it. Exhausting MAX_ATTEMPTS ABORTS.
    """
    from . import d1_selection as SEL
    excluded = RULES.excluded_digests()
    inc_colour = incumbent_colour(order)
    t1j_colour = "black" if inc_colour == "red" else "red"
    for attempt in range(RULES.MAX_ATTEMPTS):
        seed = RULES.attempt_seed(GEN_SEED_CO_PRODUCED, index, attempt)
        st = RULES._fresh_state()
        # 🔑 ONE AGENT EACH, BUILT ONCE FOR THIS OPENING (card §1.7.4). A fresh
        # agent per ply would carry no stream across the opening, and a fresh one
        # per ATTEMPT is right: a rejected candidate must leave nothing behind.
        ctx = movers["new_context"]()
        # 🔴 RESET BEFORE EITHER AGENT CAN MOVE. A bare IntegrationContext has
        # `task_id = None` and `stats = {}`, and `T1jAgent.__call__` ends with
        # `ctx.bump("t1j_queries")`, which is `self.stats[self.task_id][key] += 1`
        # -- so the FIRST T1j move raised `KeyError: None` and the one authorized
        # generation attempt VOIDed at opening 0 having produced nothing.
        #
        # In the MATCH nothing had to do this: `make_state_factory` resets the
        # context per task. The generator builds its own positions, so it has no
        # state factory, and nothing reset it.
        #
        # 🔑 A DISTINCT IDENTITY PER ATTEMPT, never a shared constant. `stats` is
        # keyed by task_id and is "PER TASK and NEVER cleared" by its own
        # contract; one shared id would accumulate every attempt's counters into
        # a single bucket, which is the defect that comment already records.
        # The opening prefix is EMPTY: the walk starts from an empty board.
        ctx.reset(f"h3gen-{index:03d}-a{attempt:03d}-s{seed}", [])
        inc = movers["incumbent_agent"](seed=seed, colour=inc_colour)
        t1j = movers["t1j_agent"](colour=t1j_colour, ctx=ctx)
        moves = []
        for ply in range(1, RULES.OPENING_PLIES + 1):
            if st.is_terminal() or not st.legal_moves():
                break
            who = mover_at_ply(order, ply)
            mv = (inc if who == "incumbent" else t1j)(st)
            mv = (int(mv[0]), int(mv[1]))
            if mv not in [tuple(m) for m in st.legal_moves()]:
                raise H3GenerationError(
                    f"opening {index} ply {ply}: {who} returned {mv}, illegal")
            moves.append(mv)
            ctx.moves.append(mv)     # the log T1jAgent checks against the ply
            st = st.apply_move(mv)
        if len(moves) != RULES.OPENING_PLIES or st.is_terminal():
            continue
        digest = SEL.canonical_digest(st)
        if digest in excluded:
            continue
        return {"index": index, "stratum": STRATUM_CO_PRODUCED,
                "order": order, "stub": False, "moves": moves, "digest": digest,
                "state": st, "seed": seed, "attempts": attempt + 1}
    raise H3GenerationError(
        f"opening {index} exhausted MAX_ATTEMPTS={RULES.MAX_ATTEMPTS}. "
        f"Generation ABORTS rather than delivering fewer or relaxing a filter.")


def _generate_unguarded(*, movers: Dict[str, Any], out_path: str,
                        trace_path: str, cleanup=None, deadline=None,
                        deadline_s: Optional[float] = None,
                        n: Optional[int] = None) -> Dict[str, Any]:
    """The walk, the rejection loop and the pinned artifact. PRIVATE, and never a
    way around the gate — it takes prepared movers and builds no engine.

    🔴 CLEANUP IS UNCONDITIONAL AND FINAL. It used to run only after an opening
    was ACCEPTED, so an exception anywhere — agent construction, a move, the
    rejection loop, artifact validation — or an operator interrupt skipped it and
    could leave production collaborators alive. It now runs in a `finally`,
    whatever happened, and ITS OWN FAILURE IS A TERMINAL OUTCOME rather than a
    footnote to someone else's.

    🔴 THE DEADLINE CAPS THE WHOLE LOOP. The only deadline used to be the
    compile's, which said nothing about a generation that walks for ever.

    🔴 A TERMINAL RECORD IS WRITTEN ON EVERY PATH. Success, timeout, void and
    cleanup failure each leave a durable `generation_end` naming the verdict; a
    run whose trace simply stops cannot be told from one that never started.
    """
    import json
    n = PAIRS_PER_STRATUM if n is None else n
    deadline_s = RULES.GENERATION_DEADLINE_S if deadline_s is None else deadline_s
    if deadline is not None and not deadline.started:
        deadline.start()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tfd = os.open(trace_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    trace = os.fdopen(tfd, "w")

    def emit(obj):
        trace.write(json.dumps(obj, sort_keys=True, default=str) + "\n")
        trace.flush()
        os.fsync(trace.fileno())

    seen = set()
    openings: List[Dict[str, Any]] = []
    verdict = "VOID"
    failure: Optional[str] = None
    timed_out = False
    doc: Optional[Dict[str, Any]] = None
    try:
        try:
            emit({"event": "generation_start", "n": n,
                  "deadline_s": deadline_s,
                  "selection_mode": movers["config"].selection_mode,
                  "note": "the generating incumbent is NOT the playing one"})
            for index in range(n):
                if deadline is not None and deadline.elapsed() > deadline_s:
                    timed_out = True
                    emit({"event": "deadline", "index": index,
                          "accepted": len(openings)})
                    break
                op = generate_one(index=index, order=order_for_index(index),
                                  movers=movers)
                if op["digest"] in seen:
                    raise H3GenerationError(
                        f"opening {index} duplicates an accepted digest; the "
                        f"population must be distinct up to symmetry")
                seen.add(op["digest"])
                openings.append(op)
                emit({"event": "opening", "index": index, "order": op["order"],
                      "attempts": op["attempts"], "seed": op["seed"],
                      "digest": op["digest"]})
                # 🔑 BETWEEN OPENINGS, so no search tree is carried across one
                # (card §1.7.4). This is NOT the teardown: that is unconditional
                # and lives in the `finally`. The two exist for different reasons
                # and dropping either one is a different defect.
                if cleanup is not None:
                    cleanup()
            if timed_out:
                # 🔑 A PARTIAL POPULATION IS NOT A POPULATION. The study is
                # defined over all 148; a short set is evidence of a failed run,
                # never a set to play against, so NO ARTIFACT IS WRITTEN.
                verdict = "TIMEOUT"
                failure = (f"the generation deadline of {deadline_s}s expired "
                           f"after {len(openings)} of {n} openings; a partial "
                           f"population is not a population and none is written")
            else:
                doc = {
                    "design": "H3_FULL_STUDY_OPENINGS",
                    "stratum": STRATUM_CO_PRODUCED,
                    "n": len(openings),
                    "selection_mode": movers["config"].selection_mode,
                    "generation_note":
                        "SYMMETRICALLY CO-PRODUCED, NOT NEUTRAL. Both orders are "
                        "run half and half, which balances each engine's ROLE; it "
                        "does not make the positions engine-independent. And only "
                        "the incumbent supplies variation: T1j is deterministic at "
                        "fixed depth, so it contributes content and no entropy.",
                    "config_pins": config_pins(movers["config"]),
                    "toolchain": movers.get("toolchain"),
                    "openings": [{k: v for k, v in o.items() if k != "state"}
                                 for o in openings],
                    "opening_set_digest": RULES.opening_set_digest(openings),
                }
                validate_artifact(doc)      # the schema, before it is written
                fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                with os.fdopen(fd, "w") as fh:
                    json.dump(doc, fh, indent=1, sort_keys=True, default=str)
                verdict = "OK"
        except BaseException as e:          # noqa: BLE001 -- interrupts included
            verdict = ("INTERRUPTED" if isinstance(e, KeyboardInterrupt)
                       else "VOID")
            failure = f"{type(e).__name__}: {e}"
            raise
        finally:
            # 🔴 UNCONDITIONAL. Whatever happened above -- success, timeout,
            # refusal, crash, Ctrl-C -- the collaborators are torn down here, and
            # a teardown that ITSELF fails becomes the terminal verdict.
            cleanup_ok = True
            cleanup_error = None
            if cleanup is not None:
                try:
                    cleanup()
                except BaseException as ce:                   # noqa: BLE001
                    cleanup_ok, cleanup_error = False, f"{type(ce).__name__}: {ce}"
            if not cleanup_ok:
                verdict = "CLEANUP_FAILED"
                failure = (f"teardown failed after a {verdict!r} body: "
                           f"{cleanup_error}")
            emit({"event": "generation_end", "verdict": verdict,
                  "accepted": len(openings), "expected": n,
                  "timed_out": timed_out, "cleanup_ok": cleanup_ok,
                  "cleanup_error": cleanup_error, "failure": failure,
                  "opening_set_digest": (doc or {}).get("opening_set_digest"),
                  # 🔴 ANY ATTEMPT RETIRES THE WHOLE RANGE. Attempts consume
                  # generation seeds whether or not the opening was accepted, and
                  # a rejected candidate drew from its seed exactly as an accepted
                  # one did. There is no partial retirement to argue about.
                  "retires": list(generation_seed_range(GEN_SEED_CO_PRODUCED)),
                  "retirement_rule": "WHOLE RANGE, on ANY attempted generation"})
            trace.close()
            if not cleanup_ok:
                raise H3GenerationCleanupError(failure)
    except H3GenerationCleanupError:
        raise
    if verdict != "OK":
        raise H3GenerationError(failure or verdict)
    return doc
