"""H4 §4A -- RAW CAPABILITY CHARACTERIZATION. THE GATED RUNNER. NOT AUTHORIZED.

Frozen by `docs/superpowers/2026-09-21-t1j-h4-replacement-card.md` §4A. This
module CHARACTERIZES what the existing helper emits. It changes no adapter
acceptance behaviour, reserves and draws no seed, loads no model, plays no game,
and aggregates no strength. It is not §4B and must never grow into it.

🔑 WHAT §4A EXISTS TO ANSWER, and why guessing was not allowed. The H4 design
needs two facts that this programme's records do not contain:

  1. **Is a searched-position dump emitted when T1j never searches?** The
     2026-08-31 low-ply run measured that T1j does not enter alpha-beta at
     board-plies 1 and 3 -- but it recorded no PROC lines and its query path is
     not the same surface question. If no dump is emitted, `len(dumps) == 1`
     can never hold for a fallback move, and §4B's board-coherence requirement
     is UNREACHABLE rather than merely unimplemented.
  2. **Does the helper accept a ZERO-LENGTH position?** Both entry points build
     `args = [java, ...] + mode + [f"{x},{y}" for (x, y) in xy]`, so at board-ply
     0 the jvm is invoked with an EMPTY position argument tail -- on the query
     path (`E4Preflight`) and, separately, on the replay path (`E3bDump`). Those
     are two different Java mains and neither answers for the other.

🔑 A SHORTFALL BY T1J IS A RESULT, NOT AN ABORT. This is the lesson D1 paid for:
a reply that did not complete its requested depth aborted the whole run, so the
observation could not be recorded. Here every reply is recorded and the run
continues. VOID is reserved for THE INSTRUMENT -- a per-call timeout, a deadline
breach, output that cannot be parsed at all, or a toolchain that does not verify.

🔴 THE FOUR BRANCHES ARE FROZEN IN THE CARD, BEFORE ANY OUTPUT WAS SEEN, and
this module may not add a fifth. `classify()` evaluates them by a documented
precedence AND records every branch that fired, so the precedence can never hide
a finding. A stop is a RESULT and exits 0; only the instrument failing exits 3.

🔴 WHAT THIS MODULE MAY NOT DO, restated because the gate alone is not the
boundary: it never relaxes `completed`, never touches `T1jAgent`, never writes a
move into a game, never reads another experiment's gate, and never computes a
score. A test asserts each of those against this file.

SCOPE OF THE FROZEN MATRIX, stated here so no reader infers more from the record
than it holds: board-plies 0, 1, 3 and 5 only. Because TwixT's first player is
red, ply 0 is the ONLY position in this matrix where T1j is asked to move as
RED; plies 1, 3 and 5 are all black-to-move. So the red role is observed at
exactly one position, and nothing here speaks for plies 2 or 4.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import e4_screen_integration as INT
from . import l0_match_rules as L0R
from . import t1j_adapter as A
from .d1_probe import D1Error, D1VoidError
from .d1_probe import Deadline, QueryBudget, T1jPaths, _supervisor
from .d1_probe import _default_compile as _compile_helper_verified
from .t1j_toolchain import ToolchainError

Pos = Tuple[int, int]

#: §4A IS NOT AUTHORIZED. Changing this is a reviewed one-line code change. It is
#: read directly at BOTH public entry points -- `run_characterization` and `main`
#: -- because gating only the CLI protects nothing: a direct Python caller
#: reaches the runner without passing any gate. No supported override exists:
#: not argv, not the environment, not a configuration file, not an import hook.
#:
#: This is §4A's OWN gate. It reads no other experiment's, and a test asserts no
#: other gate name appears anywhere in this file: one gate must never be
#: openable by opening another.
H4_4A_CHARACTERIZATION_AUTHORIZED = True

#: H4's frozen T1j search depth (`T1J_MDPLY`, with `mdFixedPly=True`). §4A
#: characterizes the surface H4 will actually use, so it asks at H4's depth and
#: at no other. A second depth would be a different question.
DEPTH = 6

#: The frozen matrix. Ply 0 is the empty board -- the position that does not
#: exist in any prior record and the whole reason §4A runs.
MATRIX_PLIES = (0, 1, 3, 5)

PER_CALL_TIMEOUT_S = 120
RUN_DEADLINE_S = 900

#: READ, never retyped. The card's §1.6 affirms the established 280-TOTAL-ply
#: boundary, and `l0_match_rules` is where that number lives. `T1jPaths` has no
#: default for `ply_cap` on purpose -- "a cap that must be named cannot be
#: forgotten quietly" -- and the first version of this CLI handed it `None`
#: from an argparse default, putting the silence straight back.
PLY_CAP = L0R.PLY_CAP

#: 10 positions (1 empty board + 9 frozen prefixes) x (1 query + 1 replay).
#: The cap is DERIVED below from the matrix actually built, never from this
#: number -- a hand-typed count is the thing that goes stale.
N_POSITIONS_EXPECTED = 10
SUBPROCESS_CAP = N_POSITIONS_EXPECTED * 2

#: The non-empty positions, pinned BY CONTENT to the same file the 2026-08-31
#: low-ply qualification was frozen against. Reusing that exact input is what
#: makes §4A's observations comparable with the existing record instead of being
#: a fresh, incomparable sample -- and it needs no seed, because it draws none.
FROZEN_PREFIXES_REL = ("docs/superpowers/evidence/2026-08-28-t1j-lowply-qualification/"
                       "01_frozen_prefixes.json")
FROZEN_PREFIXES_SHA256 = "a9054cb2d56cf75f554292e47ac7e5b86a0c74f54b255ee9147fb65f92207acf"

#: The four frozen branches. §4A REPORTS one of these; it decides nothing else.
BRANCH_PROCEED = "PROCEED_TO_4B"
BRANCH_ZERO_LENGTH_REFUSED = "STOP_ZERO_LENGTH_REFUSED"
BRANCH_NO_SAME_PROCESS_DUMP = "STOP_NO_SAME_PROCESS_DUMP"
BRANCH_INSTRUMENT = "STOP_INSTRUMENT_FAILURE"

#: Evaluated in THIS order, and every branch that fired is also recorded under
#: `branches_fired`, so choosing a headline can never suppress a finding.
#:
#: Zero-length refusal outranks the instrument branch deliberately: a refusal at
#: ply 0 will often produce absent or unclean postcondition output, and calling
#: that "the instrument failed" would relabel the exact finding §4A exists to
#: make. The instrument branch is for a position that was SUPPOSED to work.
BRANCH_PRECEDENCE = (BRANCH_ZERO_LENGTH_REFUSED, BRANCH_INSTRUMENT,
                     BRANCH_NO_SAME_PROCESS_DUMP)


class H4A4Error(Exception):
    """A refusal by the harness. Never a statement about T1j."""


class H4A4VoidError(H4A4Error):
    """The run is VOID: the instrument failed, so nothing was measured."""


def digest_of(state) -> str:
    """The canonical position digest, recomputed -- never read from a field."""
    from .d1_selection import canonical_digest
    return canonical_digest(state)


def load_frozen_prefixes(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """The frozen non-empty positions, VERIFIED BY HASH before they are returned.

    An explicit path RELOCATES the file; it does not choose a different one. The
    pin is what makes this the same input the low-ply record was made against.
    """
    p = FROZEN_PREFIXES_REL if path is None else path
    try:
        raw = open(p, "rb").read()
    except OSError as e:
        raise H4A4Error(f"cannot read the frozen prefixes: {e}") from None
    got = hashlib.sha256(raw).hexdigest()
    if got != FROZEN_PREFIXES_SHA256:
        raise H4A4Error(
            f"{p}: sha256 {got} != the pinned {FROZEN_PREFIXES_SHA256}. §4A is bound "
            f"to that exact content, so a different file answers a different question "
            f"while looking identical.")
    return json.loads(raw)["prefixes"]


def _state_for(prefix: Sequence[Pos], *, where: str):
    """Replay a prefix in OUR engine. An illegal prefix is an INSTRUMENT fault."""
    from .game.twixt_state import TwixtState
    state = TwixtState(active_size=A.BOARD_N, to_move="red")
    for i, move in enumerate(prefix):
        if tuple(move) not in set(state.legal_moves()):
            raise H4A4VoidError(
                f"{where}: prefix move {i} {tuple(move)} is illegal at ply {state.ply}. "
                f"The frozen input is wrong, which is an instrument failure: VOID.")
        state = state.apply_move(tuple(move))
    return state


def _matrix_from(prefixes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Shape a prefix list into the matrix. NO pin check -- `build_matrix` does that."""
    out: List[Dict[str, Any]] = [{
        "task_id": "empty_board", "ply": 0, "prefix": [], "digest": None,
        "opening": None, "colour_arm": None, "source": "constructed",
    }]
    for p in prefixes:
        if int(p["ply"]) not in MATRIX_PLIES:
            continue
        out.append({"task_id": p.get("task_id"), "ply": int(p["ply"]),
                    "prefix": [list(m) for m in p["prefix"]],
                    "digest": p.get("digest"), "opening": p.get("opening"),
                    "colour_arm": p.get("colour_arm"), "source": "frozen_prefixes"})
    return out


def build_matrix(prefixes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The frozen matrix: the empty board, then the pinned prefixes.

    🔴 THE WHOLE CANONICAL MATRIX IS COMPARED AGAINST THE PINNED FILE, not just
    the set of ply numbers it covers. Checking only `{0,1,3,5}` was a FAIL-OPEN:
    a caller could drop six of the nine pinned prefixes, duplicate a row, swap
    one for another position, or add rows, and still present plies 1, 3 and 5 --
    and the public runner would execute that different matrix and report on it
    as though it were the frozen one.

    An input that is not the pinned input answers a different question while
    looking identical, which is the same shape as a gate that does not gate.

    The empty board carries no pinned digest because there is exactly one of it;
    its digest is RECOMPUTED in `_stages` and recorded like any other.
    """
    got = _matrix_from(prefixes)
    want = _matrix_from(load_frozen_prefixes())
    if got != want:
        only_got = [p for p in got if p not in want]
        only_want = [p for p in want if p not in got]
        detail = (f"\n  only in the input: {[(p['task_id'], p['ply']) for p in only_got]}"
                  f"\n  missing from it  : "
                  f"{[(p['task_id'], p['ply']) for p in only_want]}")
        if not only_got and not only_want:
            # ORDER IS PART OF THE PIN, and the record proves why: every
            # observation carries a monotonic `ordinal`, so a reordered input
            # produces a differently-numbered record of the same positions. The
            # only supported input is `load_frozen_prefixes()`, whose order is
            # deterministic, so nothing legitimate reorders.
            detail = ("\n  the same positions in a DIFFERENT ORDER. Order is part "
                      "of the pin: each observation carries a monotonic `ordinal`.")
        raise H4A4Error(
            f"the matrix is not the pinned one ({len(got)} positions vs "
            f"{len(want)}).{detail}\n"
            f"A matrix that is not the frozen one answers a different question.")
    plies = sorted({m["ply"] for m in got})
    if plies != sorted(MATRIX_PLIES):
        raise H4A4Error(
            f"the pinned matrix covers plies {plies}, not the frozen "
            f"{sorted(MATRIX_PLIES)}. The pin and this module disagree.")
    return got


def _procs(out: str, *, where: str) -> List[Dict[str, Any]]:
    """The helper's OWN process identity, parsed -- the count IS the jvm count.

    `parse_procs` has existed and been tested since the adapter was written and
    has never had a production caller; §4A is the first. A MALFORMED PROC line
    means the instrument cannot be read, which is VOID.
    """
    try:
        recs = A.parse_procs(out)
    except (ValueError, KeyError, A.HelperOutputError) as e:
        # `parse_procs` raises ValueError and `parse_postconds` raises
        # HelperOutputError for the same KIND of fault. Catching only one leaks
        # the other past `_run_unguarded` into `main`'s catch-all, where a fully
        # understood refusal reports as UNEXPECTED (exit 4) instead of VOID
        # (exit 3) -- the exit code and the verdict must say ONE thing.
        raise H4A4VoidError(
            f"{where}: the helper's PROC output could not be parsed: {e}. "
            f"T1j reported: {A.helper_failure_excerpt(out)}. VOID.") from None
    return [{"pid": r.pid, "java_version": r.java_version, "vm": r.vm,
             "headless": r.headless, "prefs_factory": r.prefs_factory} for r in recs]


def _postconds(out: str, *, expected_refl: int, where: str) -> Dict[str, Any]:
    """The postcondition surface, READ. Unparseable means VOID.

    🔴 THE SAFETY SURFACE AND THE FAILURES COUNTER ARE DIFFERENT THINGS, and
    conflating them would make §4A classify its own subject matter as broken.

    `PostCond.clean` is False whenever `failures > 0` -- and E4Preflight sets
    that counter precisely when the requested depth did not complete, which is
    the condition §4A exists to characterize. The 2026-08-31 low-ply record
    shows it: every non-completing invocation carries
    `PostCond(..., prefs_ok=True, refl_ok=True, refl_n=3, failures=1)`.

    So `failures` is RECORDED and never judged here. What makes the INSTRUMENT
    unreadable is the SAFETY surface: a throw, a window or frame, a headless or
    preferences violation, or the wrong reflective-access count. That is the
    same distinction D1 got wrong when it turned a measurement into an abort.
    """
    try:
        posts = A.parse_postconds(out)
    except (ValueError, KeyError, A.HelperOutputError) as e:
        raise H4A4VoidError(
            f"{where}: the helper's POSTCOND output could not be parsed: {e}. "
            f"T1j reported: {A.helper_failure_excerpt(out)}. VOID.") from None
    rows = [{"no_throw": p.no_throw, "windows": p.windows, "frames": p.frames,
             "headless": p.headless, "prefs_ok": p.prefs_ok, "refl_ok": p.refl_ok,
             "refl_n": p.refl_n, "failures": p.failures, "clean": p.clean}
            for p in posts]
    notes: List[str] = []
    safety_clean = False
    if len(rows) != 1:
        notes.append(f"{len(rows)} POSTCOND lines, expected exactly 1")
    else:
        r = rows[0]
        for ok, msg in ((r["no_throw"], "the helper threw"),
                        (r["windows"] == 0, f"{r['windows']} windows opened"),
                        (r["frames"] == 0, f"{r['frames']} frames opened"),
                        (r["headless"], "not headless"),
                        (r["prefs_ok"], "the preferences surface was disturbed"),
                        (r["refl_ok"], "the reflective-access check failed"),
                        (r["refl_n"] == expected_refl,
                         f"{r['refl_n']} reflective accesses, expected exactly "
                         f"{expected_refl}")):
            if not ok:
                notes.append(f"safety surface not clean: {msg}")
        safety_clean = not notes
    return {"rows": rows, "n": len(rows), "expected_refl": expected_refl,
            "notes": notes, "safety_clean": safety_clean,
            # RECORDED, NOT JUDGED: for a non-searching query this is expected
            # to be non-zero, and that is the measurement, not a fault.
            "failures_counter": rows[0]["failures"] if len(rows) == 1 else None,
            "clean_single": len(rows) == 1 and rows[0]["clean"]}


def _dump_summary(d) -> Dict[str, Any]:
    return {"ply": d.ply, "next_player": d.next_player, "term_y": d.term_y,
            "term_x": d.term_x, "n_pegs": len(d.pegs), "n_bridges": len(d.bridges),
            "n_legal": len(d.legal), "history_len": len(d.history)}


def observe_query(*, position: Dict[str, Any], state, paths: T1jPaths,
                  budget: QueryBudget, ordinal: int) -> Dict[str, Any]:
    """ONE query subprocess, described in full. Records; does not judge T1j."""
    where = f"{position['task_id']}@ply{position['ply']} query"
    moves = [tuple(m) for m in position["prefix"]]
    budget.spend(1)
    try:
        recs, dumps, rc, out = A.query(
            moves, depth=DEPTH, java=paths.java, jar=paths.jar,
            classes=paths.classes, repeats=1, timeout_s=PER_CALL_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        raise H4A4VoidError(
            f"{where}: T1j did not answer within {PER_CALL_TIMEOUT_S}s ({e}). "
            f"A hung process is an instrument failure: VOID.") from None
    except A.HelperOutputError as e:
        # The raw text is UNPARSEABLE, which is different from T1j refusing: we
        # cannot read the instrument, so nothing was measured here.
        raise H4A4VoidError(
            f"{where}: the helper's QUERY output could not be parsed ({e}). "
            f"T1j reported: {A.helper_failure_excerpt(e.stdout)}. VOID.") from None

    procs = _procs(out, where=where)
    post = _postconds(out, expected_refl=INT.QUERY_REFL_N, where=where)

    # 🔴 A NON-ZERO EXIT IS EXPECTED AT LOW PLY AND IS NOT A VOID. E4Preflight
    # exits 3 when its own `failures` counter is set, which is the very
    # condition under characterization. What makes a reply ABSENT is having no
    # parsed record at all -- and that, at ply 0, is the refusal §4A measures.
    rec = recs[0] if len(recs) == 1 else None
    coherence: Optional[List[str]] = None
    if len(dumps) == 1:
        coherence = list(INT.compare_state(state, dumps[0], moves))

    obs = {
        "role": "query", "ordinal": ordinal, "where": where,
        "task_id": position["task_id"], "ply": position["ply"],
        "prefix": [list(m) for m in moves],
        "our_to_move": state.to_move,
        "t1j_colour_role": state.to_move,
        "requested_depth": DEPTH,
        "return_code": rc,
        "n_query_records": len(recs),
        "n_dumps": len(dumps),
        "procs": procs, "n_procs": len(procs),
        "postcond": post,
        "dumps": [_dump_summary(d) for d in dumps],
        "dump_coherence_divergences": coherence,
        # 🔑 SAME-PROCESS BY CONSTRUCTION, and recorded as such rather than
        # assumed: these dumps were parsed from THIS subprocess's own stdout,
        # which also carries the PROC line(s) above. The binder's replay jvm is
        # a DIFFERENT process and can never establish this property.
        "same_process_dump": len(dumps) == 1 and len(procs) == 1,
        "dump_source_pids": [p["pid"] for p in procs],
        "record": None,
        "stdout": out,
    }
    if rec is not None:
        obs["record"] = {
            "q": rec.q, "requested_depth": rec.requested_depth,
            "move": list(rec.move) if rec.move else None, "to_move": rec.to_move,
            "usealphabeta": rec.usealphabeta, "current_max_ply": rec.current_max_ply,
            "completed_depth": rec.completed_depth, "completed": rec.completed,
            "legal": rec.legal, "null_sentinel": rec.null_sentinel,
            # `move_nr` is T1j's OWN move counter and was omitted from the first
            # version. §4B's coherence work needs the engine's count, not only
            # ours, and a field absent from the record cannot be recovered later.
            "move_nr": rec.move_nr,
            "eval_regime": rec.eval_regime, "elapsed_us": rec.elapsed_us,
        }
        obs["move_legal_in_our_engine"] = (
            rec.move is not None and rec.move in set(state.legal_moves()))
        # 🔴 NEUTRAL BY DESIGN: `incomplete`, NEVER `native_low_ply_fallback`.
        # `completed == false` does NOT establish the qualified fallback
        # signature -- distinguishing a legitimate native fallback from a broken
        # search is the job §4B was reserved for, and §4A naming it here would
        # hand §4B a conclusion it is supposed to reach.
        obs["path"] = "searched" if rec.completed else "incomplete"
    else:
        obs["move_legal_in_our_engine"] = False
        obs["path"] = "no_record"
    return obs


def observe_replay(*, position: Dict[str, Any], state, paths: T1jPaths,
                   budget: QueryBudget, ordinal: int) -> Dict[str, Any]:
    """ONE replay subprocess, described in full.

    Deliberately calls `A.replay` DIRECTLY rather than through `make_binder`:
    the binder validates and then discards the helper's output, and §4A exists
    to keep it.
    """
    where = f"{position['task_id']}@ply{position['ply']} replay"
    moves = [tuple(m) for m in position["prefix"]]
    budget.spend(1)
    try:
        plies, rc, out = A.replay(
            moves, ply_cap=paths.ply_cap, java=paths.java, jar=paths.jar,
            classes=paths.classes, timeout_s=PER_CALL_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        raise H4A4VoidError(
            f"{where}: the replay did not answer within {PER_CALL_TIMEOUT_S}s ({e}). "
            f"A hung process is an instrument failure: VOID.") from None
    except A.HelperOutputError as e:
        raise H4A4VoidError(
            f"{where}: the replay output could not be parsed ({e}). "
            f"T1j reported: {A.helper_failure_excerpt(e.stdout)}. VOID.") from None

    procs = _procs(out, where=where)
    post = _postconds(out, expected_refl=INT.REPLAY_REFL_N, where=where)
    # The binder's own arity check, RECORDED rather than raised: one block per
    # ply plus the starting position.
    expected_blocks = state.ply + 1
    divergences: Optional[List[str]] = None
    if plies:
        divergences = list(INT.compare_state(state, plies[-1], moves))
    return {
        "role": "replay", "ordinal": ordinal, "where": where,
        "task_id": position["task_id"], "ply": position["ply"],
        "prefix": [list(m) for m in moves],
        "our_to_move": state.to_move,
        "ply_cap": paths.ply_cap,
        "return_code": rc,
        "n_plies": len(plies),
        "expected_plies": expected_blocks,
        "ply_count_matches": len(plies) == expected_blocks,
        "procs": procs, "n_procs": len(procs),
        "postcond": post,
        "plies": [_dump_summary(d) for d in plies],
        "final_state_divergences": divergences,
        "stdout": out,
    }


def _zero_length_refused(obs: Dict[str, Any]) -> bool:
    """Did the ply-0 call fail to yield the thing that call exists to yield?

    Query: NO parsed query record. Replay: NO parsed ply block. Both are read
    from the PARSE, never from the exit status -- E4Preflight exits 3 on a
    condition §4A is measuring, so an exit code cannot carry this meaning.

    🔑 ZERO, not "other than one". More than one record is the helper
    misbehaving, which is the INSTRUMENT branch; it is not a refusal, and
    calling it one would put the wrong headline on the record.

    🔑 A NULL OR ABSENT MOVE AT PLY 0 IS ALSO A REFUSAL, and this mirrors the
    empty replay exactly. The helper accepted the empty-board grammar but did
    not produce the acceptable move Arm B requires, and that is the finding --
    not a broken instrument. Classifying it as an instrument failure would
    relabel the answer §4A exists to get.

    At a NON-ZERO ply the same condition is an instrument failure, because a
    reply was expected there.
    """
    if obs["ply"] != 0:
        return False
    if obs["role"] == "query":
        if obs["n_query_records"] == 0:
            return True
        rec = obs.get("record")
        return bool(rec is not None
                    and (rec["null_sentinel"] or rec["move"] is None))
    return obs["n_plies"] == 0


def classify(observations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The FOUR FROZEN BRANCHES, and nothing else may be added here.

    Every branch that fired is recorded; the precedence only chooses a headline.
    """
    fired: Dict[str, List[str]] = {}

    def fire(branch: str, why: str) -> None:
        fired.setdefault(branch, []).append(why)

    for o in observations:
        refused = _zero_length_refused(o)
        if refused:
            fire(BRANCH_ZERO_LENGTH_REFUSED,
                 f"{o['where']}: the zero-length position yielded no "
                 f"{'query record' if o['role'] == 'query' else 'ply block'}")

        # ── the INSTRUMENT branch: a call that was SUPPOSED to work, did not ──
        for note in o["postcond"]["notes"]:
            fire(BRANCH_INSTRUMENT, f"{o['where']}: {note}")
        if o["n_procs"] != 1:
            fire(BRANCH_INSTRUMENT,
                 f"{o['where']}: {o['n_procs']} PROC lines, expected exactly 1 "
                 f"per jvm under the frozen fresh-process lifecycle")

        # 🔴 THE `failures` EXEMPTION IS CONTEXT-SENSITIVE. It exists because
        # E4Preflight sets the counter when the requested depth did not
        # complete. Applying it universally was a FAIL-OPEN: a COMPLETED query,
        # or any replay, carrying failures > 0 is the instrument misbehaving and
        # must not ride on an exemption earned by a different condition.
        exempt = o["role"] == "query" and o.get("path") == "incomplete"
        if o["postcond"]["failures_counter"] and not exempt:
            fire(BRANCH_INSTRUMENT,
                 f"{o['where']}: failures={o['postcond']['failures_counter']} on a "
                 f"{'completed query' if o['role'] == 'query' else 'replay'}, where "
                 f"the incomplete-search exemption does not apply")

        if o["role"] == "query":
            if o["n_query_records"] > 1:
                fire(BRANCH_INSTRUMENT,
                     f"{o['where']}: {o['n_query_records']} query records, expected 1")
            # 🔴 ABSENT AT A NON-ZERO PLY. Only ply 0 may be a refusal; anywhere
            # else, no record at all is the instrument failing. This fired
            # NOTHING before, so a query that simply vanished reported PROCEED.
            elif o["n_query_records"] == 0 and not refused:
                fire(BRANCH_INSTRUMENT,
                     f"{o['where']}: no query record at ply {o['ply']}, where a "
                     f"reply was expected")
            elif o["record"] is not None:
                # 🔴 MOVE VALIDATION APPLIES TO EVERY QUERY RECORD, whatever its
                # path. It was nested under `path == "searched"`, so an
                # INCOMPLETE query returning a null sentinel, a move T1j itself
                # calls illegal, or a move illegal in our engine reported
                # PROCEED_TO_4B. An unusable reply is unusable whether or not a
                # search ran, and §4B cannot build a coherence contract on one.
                rec = o["record"]
                if rec["null_sentinel"] or rec["move"] is None:
                    # 🔴 AT PLY 0 THIS IS THE REFUSAL, ALREADY FIRED ABOVE, and
                    # it must NOT also be logged as an instrument failure:
                    # `branches_fired` carries true findings, not the validation
                    # steps taken along the way.
                    #
                    # The ONE exception is a reply that claims it COMPLETED a
                    # search and still returns no move. That is a contradiction
                    # in the reply itself, so it is both a refusal and an
                    # instrument fault, and legitimately fires both.
                    if not refused:
                        fire(BRANCH_INSTRUMENT,
                             f"{o['where']}: the reply is the null sentinel, not "
                             f"a move")
                    elif o["path"] == "searched":
                        fire(BRANCH_INSTRUMENT,
                             f"{o['where']}: the reply claims completed={rec['completed']} "
                             f"yet returns no move -- the record contradicts itself")
                else:
                    if not rec["legal"]:
                        fire(BRANCH_INSTRUMENT,
                             f"{o['where']}: T1j reports its own move {rec['move']} "
                             f"illegal")
                    if not o["move_legal_in_our_engine"]:
                        fire(BRANCH_INSTRUMENT,
                             f"{o['where']}: the returned move {rec['move']} is not "
                             f"legal in our engine")

                # A SEARCHED query's dump is an INSTRUMENT matter; the
                # incomplete query's dump is the separate question below.
                if o["path"] == "searched":
                    if o["n_dumps"] != 1:
                        fire(BRANCH_INSTRUMENT,
                             f"{o['where']}: a completed search emitted "
                             f"{o['n_dumps']} searched-position dumps, expected 1")
                    elif o["dump_coherence_divergences"]:
                        fire(BRANCH_INSTRUMENT,
                             f"{o['where']}: the searched position diverges from "
                             f"ours: " + "; ".join(o["dump_coherence_divergences"]))

        # 🔴 REPLAY STRUCTURE AND COHERENCE WERE RECORDED AND NEVER CLASSIFIED.
        # A truncated, padded or incoherent replay left `fired` empty.
        if o["role"] == "replay" and not refused:
            if not o["ply_count_matches"]:
                fire(BRANCH_INSTRUMENT,
                     f"{o['where']}: {o['n_plies']} ply blocks, expected "
                     f"{o['expected_plies']} (one per ply plus the start)")
            elif o["final_state_divergences"]:
                fire(BRANCH_INSTRUMENT,
                     f"{o['where']}: the replayed final state diverges from ours: "
                     + "; ".join(o["final_state_divergences"]))

        # ── the DUMP branch, asked ONLY of the incomplete queries it is about ──
        if o["role"] == "query" and o.get("path") == "incomplete":
            if o["n_dumps"] != 1:
                fire(BRANCH_NO_SAME_PROCESS_DUMP,
                     f"{o['where']}: a non-searching query emitted {o['n_dumps']} "
                     f"searched-position dumps, expected exactly 1")
            elif o["dump_coherence_divergences"]:
                fire(BRANCH_NO_SAME_PROCESS_DUMP,
                     f"{o['where']}: the incomplete query's dump is not coherent "
                     f"with our position: "
                     + "; ".join(o["dump_coherence_divergences"]))
            elif not o["same_process_dump"]:
                fire(BRANCH_NO_SAME_PROCESS_DUMP,
                     f"{o['where']}: the dump could not be tied to this process")

    verdict = BRANCH_PROCEED
    for branch in BRANCH_PRECEDENCE:
        if branch in fired:
            verdict = branch
            break
    return {"verdict": verdict,
            "branches_fired": sorted(fired),
            "branch_reasons": {k: fired[k] for k in sorted(fired)},
            "precedence": list(BRANCH_PRECEDENCE)}


def run_characterization(*, prefixes: Sequence[Dict[str, Any]], paths: T1jPaths,
                         out_path: str, deadline: Optional[Deadline] = None,
                         budget: Optional[QueryBudget] = None,
                         _compile: Optional[Callable] = None) -> Dict[str, Any]:
    """PUBLIC ENTRY. Refuses while the gate is shut, before anything happens."""
    if not H4_4A_CHARACTERIZATION_AUTHORIZED:
        raise H4A4Error(
            "H4 §4A is UNAUTHORIZED. Gating only the CLI would protect nothing: a "
            "direct Python caller reaches this runner without passing it. Nothing "
            "has been compiled, queried or written.")
    return _run_unguarded(prefixes=prefixes, paths=paths, out_path=out_path,
                          deadline=deadline, budget=budget, _compile=_compile)


def _run_unguarded(*, prefixes, paths, out_path, deadline=None, budget=None,
                   _compile=None):
    """Everything below the gate. PRIVATE, and never a way around the gate.

    It exists so the machinery can be tested WITHOUT lifting the gate in a
    fixture -- a fixture that flips an execution gate IS the gate failing.
    """
    matrix = build_matrix(prefixes)
    deadline = deadline or Deadline(limit_s=RUN_DEADLINE_S)
    # DERIVED from the matrix actually built, never from the hand-typed constant.
    budget = budget or QueryBudget(cap=len(matrix) * 2)
    compile_fn = (_compile if _compile is not None
                  else functools.partial(_compile_helper_verified, paths=paths))
    deadline.start()
    try:
        with _supervisor(deadline):
            return _stages(matrix, paths, out_path, deadline, budget, compile_fn)
    except H4A4Error:
        raise
    except D1VoidError as e:
        # REUSING A CHECK MEANS ADOPTING ITS FAILURES. Untranslated, D1's
        # exceptions escape `main`'s handlers and a fully understood refusal
        # reports as UNEXPECTED (exit 4) instead of VOID (exit 3).
        raise H4A4VoidError(str(e)) from None
    except D1Error as e:
        raise H4A4Error(str(e)) from None


def _stages(matrix, paths, out_path, deadline, budget, compile_fn):
    try:
        artifacts = compile_fn(deadline)
    except (ToolchainError, D1Error) as e:
        raise H4A4VoidError(f"toolchain or compilation failed: {e}. VOID.") from None
    deadline.check("after helper compilation")

    observations: List[Dict[str, Any]] = []
    ordinal = 0
    for position in matrix:
        label = f"{position['task_id']}@ply{position['ply']}"
        deadline.check(f"before {label}")
        moves = [tuple(m) for m in position["prefix"]]
        state = _state_for(moves, where=label)
        if state.ply != position["ply"]:
            raise H4A4VoidError(
                f"{label}: the prefix replays to ply {state.ply}, not the recorded "
                f"{position['ply']}. The input describes a different position: VOID.")
        got = digest_of(state)
        if position["digest"] is not None and got != position["digest"]:
            raise H4A4VoidError(
                f"{label}: the frozen prefix replays to digest {got}, not the "
                f"recorded {position['digest']!r}: VOID.")
        position["digest_recomputed"] = got

        observations.append(observe_query(position=position, state=state, paths=paths,
                                          budget=budget, ordinal=ordinal))
        ordinal += 1
        deadline.check(f"after {label} query")
        observations.append(observe_replay(position=position, state=state, paths=paths,
                                           budget=budget, ordinal=ordinal))
        ordinal += 1
        deadline.check(f"after {label} replay")

    deadline.check("before writing the record")
    result = classify(observations)
    report = {
        "stage": "h4_4A_raw_capability_characterization",
        "verdict": result["verdict"],
        "branches_fired": result["branches_fired"],
        "branch_reasons": result["branch_reasons"],
        "branch_precedence": result["precedence"],
        "matrix_plies": list(MATRIX_PLIES),
        "depth": DEPTH,
        "n_positions": len(matrix),
        "n_observations": len(observations),
        "subprocesses_spent": budget.spent,
        "subprocess_cap": budget.cap,
        "per_call_timeout_s": PER_CALL_TIMEOUT_S,
        "run_deadline_s": deadline.limit_s,
        "elapsed_s": deadline.elapsed(),
        "frozen_prefixes_sha256": FROZEN_PREFIXES_SHA256,
        "toolchain_identity": artifacts,
        "positions": matrix,
        "observations": observations,
        # 🔴 The scope, carried IN the record, because a stored verdict is the
        # easiest thing here to quote past its evidence.
        "scope": (
            "Board-plies 0, 1, 3 and 5 only, at depth 6, on the frozen prefix set. "
            "Ply 0 is the ONLY position where T1j is asked to move as RED; plies 1, "
            "3 and 5 are black-to-move. Nothing here speaks for plies 2 or 4, for "
            "any other depth, or for T1j's strength. No game was played and no "
            "score exists."),
    }
    fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    return report


def resolve_paths(classes: str) -> T1jPaths:
    """Resolve the VERIFIED toolchain and freeze the cap. Nothing is defaulted.

    🔴 WHY THIS EXISTS. The first version of the CLI offered `--java`, `--jar`
    and `--ply-cap`, each defaulting to `None`, so an omitted argument reached
    execution as `None` -- and `T1jPaths` carries no default for `ply_cap`
    precisely because "a cap that must be named cannot be forgotten quietly".
    An argparse default put the silence straight back, which is this
    programme's recurring shape: a default is a switch-off.

    So the toolchain is RESOLVED rather than accepted. `verified_paths` hashes
    the jar and every pinned JDK component before returning a path, and refuses
    a root under /tmp whatever supplied it. `_default_compile` then re-checks
    that what we hand it IS what was verified, so an unverifiable path cannot
    reach a jvm.

    `classes` stays the caller's, because it is an OUTPUT: create-only, so a
    stale `.class` from another build can never decide what ran.
    """
    from . import t1j_toolchain as TC
    tc = TC.verified_paths()
    return T1jPaths(java=os.path.join(tc["jdk_home"], "bin", "java"),
                    jar=tc["jar"], classes=classes, ply_cap=PLY_CAP)


EXIT_OK = 0
EXIT_VOID = 3
EXIT_UNEXPECTED = 4
EXIT_UNAUTHORIZED = 5


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI. Refuses while the gate is shut, BEFORE touching anything.

    This is the SECOND of the two guard reads; `run_characterization` carries
    the other. Nothing reads an environment variable, a flag or a configuration
    file to reach the gate.

    NOTE ON THE EXIT STATUS: every one of the four frozen branches exits 0,
    including the three STOPs. A stop is a RESULT and the record carries it.
    Only the instrument failing to be readable at all is a non-zero abort.
    """
    import argparse
    ap = argparse.ArgumentParser(
        prog="h4_4a_characterization",
        description="H4 §4A raw capability characterization. IT IS NOT AUTHORIZED.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefixes")
    #: The one path the caller must supply, because it is an OUTPUT: the
    #: class directory is create-only, so it cannot be resolved for them.
    ap.add_argument("--classes", required=True)
    a = ap.parse_args(argv)

    if not H4_4A_CHARACTERIZATION_AUTHORIZED:
        print("H4 §4A is UNAUTHORIZED. No JVM was started, no position queried, "
              "and no file was written.", file=sys.stderr)
        return EXIT_UNAUTHORIZED

    try:                                                      # pragma: no cover
        run_characterization(
            prefixes=load_frozen_prefixes(a.prefixes),
            paths=resolve_paths(a.classes),
            out_path=a.out)
    except H4A4VoidError as e:                                # pragma: no cover
        print(f"VOID: {e}", file=sys.stderr)
        return EXIT_VOID
    except H4A4Error as e:                                    # pragma: no cover
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_UNEXPECTED
    except Exception as e:                                    # noqa: BLE001
        # A CATCH-ALL, because without one anything unnamed escapes as a
        # traceback with no verdict at all -- worse than an UNEXPECTED exit,
        # which at least reports.
        print(f"UNEXPECTED {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_UNEXPECTED
    return EXIT_OK                                            # pragma: no cover


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
