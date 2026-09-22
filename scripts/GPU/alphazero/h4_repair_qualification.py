"""H4 REPAIR QUALIFICATION -- THE GATED RUNNER. IT IS NOT AUTHORIZED.

Frozen by `docs/superpowers/2026-09-21-t1j-h4-repair-qualification-card.md`.
🔴 **THE CARD IS THE AUTHORITY.** Every constant below is transcribed FROM it;
this module does not get to settle a question by being written. If the helper or
this runner disagrees with the card, the card wins and the disagreement is a
design amendment, not a code fix.

SEPARATELY NAMED, NOT A §4A RETRY. §4A closed at `STOP_ZERO_LENGTH_REFUSED` on
the helper as it was. The helper's behaviour has changed, so this qualification
has its own name, its own gate, its own create-only destination and its own
frozen stop conditions, and its results MAY NOT be pooled with §4A's.

WHAT IT ESTABLISHES: that the repaired helper returns a legal, board-coherent
move from every position H4 will actually query; that WHICH ROUTINE ANSWERED is
identifiable; and that the process and reflection contracts hold. It plays no
game, draws no seed, computes no score, and is not strength evidence.

🔴 EXIT SEMANTICS ARE THE DANGEROUS PART, and the card freezes them in BOTH
directions. The repair does NOT change the helper's completion requirement, so a
LEGITIMATE native-initial reply still looks like a failure at the process
boundary: `exit 3`, `failures 1`, `completed false`. That is the exact shape D1
mistook for an abort. A native-initial reply is permitted ONLY at exit 3 with
failures 1; a completed search is required to be exit 0 with failures 0; and
anything else STOPS. Read "clean" as "exit 0 everywhere" and every legitimate
native reply is rejected; ignore the exit code and a genuinely broken one is
accepted.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import subprocess
import sys
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import e4_screen_integration as INT
from . import h4_4a_characterization as H4A
from . import l0_match_rules as L0R
from . import t1j_adapter as A
from .d1_probe import D1Error, D1VoidError
from .d1_probe import Deadline, QueryBudget, T1jPaths, _supervisor
from .d1_probe import _default_compile as _compile_helper_verified
from .t1j_toolchain import ToolchainError

Pos = Tuple[int, int]

#: THE H4 REPAIR QUALIFICATION IS NOT AUTHORIZED. Changing this is a reviewed
#: one-line code change. Read directly at BOTH public entry points --
#: `run_qualification` and `main` -- because gating only the CLI protects
#: nothing: a direct Python caller reaches the runner without passing any gate.
#: No supported override exists: not argv, not the environment, not a
#: configuration file, not an import hook.
#:
#: This is the qualification's OWN gate. It reads no other experiment's, and a
#: test asserts no other gate name appears in this file.
H4_REPAIR_QUALIFICATION_AUTHORIZED = False

#: Transcribed from the card. H4's frozen T1j search depth.
DEPTH = 6
PLY_CAP = L0R.PLY_CAP
PER_CALL_TIMEOUT_S = 120
RUN_DEADLINE_S = 1800

#: The card's §2 matrix: the empty board plus plies 1-5 from each family, in
#: THIS family order.
FAMILIES = ("o1_center", "o3_low", "o4_high")
MATRIX_PLIES = (1, 2, 3, 4, 5)
N_POSITIONS = 1 + len(FAMILIES) * len(MATRIX_PLIES)          # 16

#: §3. Plies 0-3 dispatch to a routine that constructs a fresh unseeded Random;
#: plies 4-5 dispatch to `fifthOrMoreMove()`, which contains none.
RANDOMIZED_PLIES = (0, 1, 2, 3)
REPETITIONS_RANDOMIZED = 5
REPETITIONS_DETERMINISTIC = 1

#: 🔴 THE DERIVED MATRIX IS PINNED IN ITS OWN RIGHT. The source file's hash
#: authenticates the source SEQUENCES; it says nothing about the 16 rows
#: TRUNCATED from them. Truncation code could drop, duplicate, reorder or
#: mis-length a row while the source hash kept passing -- the §4A fail-open one
#: level up.
#:
#: The canonical form is the CARD'S, reproduced here, never defined here:
#: UTF-8 compact JSON array, keys ordered family/ply/prefix, `empty_board`
#: first, families in FAMILIES order, plies ascending 1-5, no trailing newline.
MATRIX_SHA256 = "3cc14ca99935af511614742feafe6b3966ce8da83dc2789a103f98d3d22f27cd"

#: §5. The opt-in query path performs a FOURTH reflective access
#: (`Match.matchData(write)`); the default path performs three and must not
#: drift. Re-derived from the repaired source, not assumed.
QUERY_REFL_N_OPTIN = 4
QUERY_REFL_N_DEFAULT = INT.QUERY_REFL_N                      # 3
REPLAY_REFL_N = INT.REPLAY_REFL_N                            # 1

#: §4 classifications.
NATIVE_FIRST = "native_initial_first"
NATIVE_SECOND_TO_FOURTH = "native_initial_second_to_fourth"
NATIVE_FIFTH_OR_MORE = "native_initial_fifth_or_more"
SEARCHED = "searched"

#: §4.1 exit semantics, frozen in both directions.
NATIVE_EXIT = 3
NATIVE_FAILURES = 1
SEARCHED_EXIT = 0
SEARCHED_FAILURES = 0

#: §4: a completed depth-6 search reports currentMaxPly = depth + 1.
SEARCHED_CURRENT_MAX_PLY = DEPTH + 1


class H4RQError(Exception):
    """A refusal by the harness. Never a statement about T1j."""


class H4RQVoidError(H4RQError):
    """The run is VOID: the instrument failed, so nothing was measured."""


class H4RQStop(H4RQError):
    """A frozen STOP condition fired. A RESULT, and it ends the qualification.

    It CARRIES ITS CONTEXT because a STOP must leave a durable record: retries
    are forbidden, so the one observation of the failure is the only one there
    will ever be.
    """

    def __init__(self, message: str, *, where: Optional[str] = None,
                 row: Optional[Dict[str, Any]] = None,
                 stdout: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.where = where
        self.row = row
        self.stdout = stdout


# ─────────────────────────── the pinned derived matrix ───────────────────────

def _canonical(matrix: Sequence[Dict[str, Any]]) -> str:
    """The card's canonical serialization. REPRODUCED, never invented here."""
    rows = [OrderedDict([("family", r["family"]), ("ply", r["ply"]),
                         ("prefix", [list(m) for m in r["prefix"]])])
            for r in matrix]
    return json.dumps(rows, separators=(",", ":"))


def derive_matrix(prefixes: Optional[Sequence[Dict[str, Any]]] = None
                  ) -> List[Dict[str, Any]]:
    """The 16 rows, derived by TRUNCATING each family's ply-5 sequence.

    Nesting is VERIFIED, not assumed: each family's ply-1 prefix must be a
    prefix of its ply-3, which must be a prefix of its ply-5. If that fails the
    derivation is invalid and the qualification does not start.
    """
    rows = H4A.load_frozen_prefixes() if prefixes is None else prefixes
    byfam: Dict[str, Dict[int, List[Pos]]] = {}
    for r in rows:
        byfam.setdefault(str(r["opening"]), {})[int(r["ply"])] = [
            tuple(m) for m in r["prefix"]]

    out: List[Dict[str, Any]] = [
        {"family": "empty_board", "ply": 0, "prefix": [], "source": "constructed"}]
    for fam in FAMILIES:
        if fam not in byfam:
            raise H4RQError(f"the frozen input has no family {fam!r}; it holds "
                            f"{sorted(byfam)}")
        plies = byfam[fam]
        for need in (1, 3, 5):
            if need not in plies:
                raise H4RQError(f"{fam}: the frozen input has no ply {need}")
        five = plies[5]
        if plies[1] != five[:1] or plies[3] != five[:3]:
            raise H4RQError(
                f"{fam}: the frozen prefixes are NOT NESTED -- ply1={plies[1]}, "
                f"ply3={plies[3]}, ply5={five}. Plies 2 and 4 are derived by "
                f"TRUNCATING ply 5, so without nesting the derivation describes "
                f"positions the pinned input never contained.")
        for ply in MATRIX_PLIES:
            out.append({"family": fam, "ply": ply, "prefix": list(five[:ply]),
                        "source": "truncated_from_ply5"})
    return out


def verify_matrix(matrix: Sequence[Dict[str, Any]]) -> str:
    """Reproduce the canonical form and CHECK IT AGAINST THE CARD'S PIN."""
    if len(matrix) != N_POSITIONS:
        raise H4RQError(f"the matrix holds {len(matrix)} rows, not {N_POSITIONS}")
    canon = _canonical(matrix)
    got = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    if got != MATRIX_SHA256:
        raise H4RQError(
            f"the derived matrix hashes {got}, not the card's pinned "
            f"{MATRIX_SHA256}. Content, ORDER, row count and length are all "
            f"pinned: a matrix that is not the frozen one answers a different "
            f"question while the source file's hash keeps passing.")
    return got


def repetitions_for(ply: int) -> int:
    """§3: five fresh JVMs where a routine draws randomness, one where none does."""
    return (REPETITIONS_RANDOMIZED if ply in RANDOMIZED_PLIES
            else REPETITIONS_DETERMINISTIC)


def derived_caps(matrix: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """DERIVED from the matrix actually built -- never a hand-typed total."""
    q = sum(repetitions_for(int(r["ply"])) for r in matrix)
    return {"queries": q, "replays": len(matrix), "total": q + len(matrix)}


# ───────────────────────────── the frozen classifier ─────────────────────────

def classify_reply(*, ply: int, rec, exit_status: int, failures: Optional[int]
                   ) -> str:
    """§4 + §4.1. Returns a classification or raises H4RQStop. No fifth outcome.

    🔴 Keyed on the PLY DISPATCH plus the telemetry, never on a narrative read
    off an exit code -- and the exit code and failure count are checked in BOTH
    directions, because "clean" is otherwise ambiguous.
    """
    native_telemetry = (rec.usealphabeta is False and rec.current_max_ply == 0
                        and rec.completed is False)
    searched_telemetry = (rec.usealphabeta is True
                          and rec.current_max_ply == SEARCHED_CURRENT_MAX_PLY
                          and rec.completed is True
                          and rec.completed_depth == DEPTH)

    if native_telemetry:
        if exit_status != NATIVE_EXIT or failures != NATIVE_FAILURES:
            raise H4RQStop(
                f"ply {ply}: native-initial telemetry with exit={exit_status} "
                f"failures={failures}; the frozen signature permits ONLY "
                f"exit={NATIVE_EXIT} with failures={NATIVE_FAILURES}. The "
                f"completion requirement is unchanged, so a legitimate native "
                f"reply is exactly that pair and nothing else.")
        if ply == 0:
            return NATIVE_FIRST
        if 1 <= ply <= 3:
            return NATIVE_SECOND_TO_FOURTH
        if 4 <= ply <= 5:
            return NATIVE_FIFTH_OR_MORE
        raise H4RQStop(
            f"ply {ply}: a native-initial reply is impossible here -- "
            f"InitialMoves.initialMove() returns null for moveNr >= 6 by "
            f"dispatch, so no routine can have answered.")
    if searched_telemetry:
        # 🔴 SEARCH IS IMPOSSIBLE AT PLY 0 under the frozen 24x24 no-pie
        # configuration, and this is DERIVED, not assumed: `firstMove()`
        # contains no `aconst_null`, and its no-pie branch always constructs
        # `new Move(x, y)` with x = Xsize/2 + nextInt(Xsize/4) - (Xsize/4)/2,
        # which on a 24-wide board is 9..14 -- always >= 0, so
        # `initialMove()`'s `getX() >= 0` gate always passes it through. A
        # search at ply 0 therefore means the injection did not take.
        #
        # ⚠ PLY 1 IS NOT INCLUDED HERE. Review argued it is equally impossible,
        # and it may well be, but I could not derive it: `secondToFourthMove()`
        # has a single `areturn` and a dispatch that reads as "moveNr 2 or 3
        # -> compute, else -> new Move(-1,-1)", which would make ply 1 ALWAYS
        # search -- flatly contradicting §4A, which observed native replies at
        # ply 1 in all three families. That contradiction means the reading is
        # wrong, so the rule is NOT encoded on it. Left permissive pending a
        # derivation that holds.
        if ply == 0:
            raise H4RQStop(
                f"ply 0: a completed search is impossible here -- under the "
                f"frozen 24x24 no-pie configuration InitialMoves.firstMove() "
                f"always returns a central move (x in 9..14), so a search at "
                f"ply 0 means the MatchData injection did not take effect.")
        if exit_status != SEARCHED_EXIT or failures != SEARCHED_FAILURES:
            raise H4RQStop(
                f"ply {ply}: completed-search telemetry with exit={exit_status} "
                f"failures={failures}; a completed search is required to be "
                f"exit={SEARCHED_EXIT} with failures={SEARCHED_FAILURES}.")
        return SEARCHED
    raise H4RQStop(
        f"ply {ply}: telemetry outside the frozen table -- usealphabeta="
        f"{rec.usealphabeta} currentMaxPly={rec.current_max_ply} "
        f"completed={rec.completed} completed_depth={rec.completed_depth} "
        f"exit={exit_status} failures={failures}.")


# ─────────────────────────────── the observations ────────────────────────────

def _with_context(stop: H4RQStop, *, where: str, row: Dict[str, Any],
                  stdout: Optional[str]) -> H4RQStop:
    """Attach the failing context ONCE, at the boundary.

    Enriching here rather than at every `raise` means a future STOP condition
    gets a durable record for free. A dozen raisers each remembering to pass
    three arguments is a dozen chances to forget, and the one that forgets is
    the one whose failure goes unrecorded.
    """
    if stop.where is None:
        stop.where = where
    if stop.row is None:
        stop.row = row
    if stop.stdout is None:
        stop.stdout = stdout
    return stop

def _postcond_one(out: str, *, expected_refl: int, where: str) -> Dict[str, Any]:
    """The safety surface, read. Unparseable is VOID; unclean is a STOP."""
    try:
        posts = A.parse_postconds(out)
    except (ValueError, KeyError, A.HelperOutputError) as e:
        raise H4RQVoidError(
            f"{where}: the POSTCOND output could not be parsed: {e}. "
            f"T1j reported: {A.helper_failure_excerpt(out)}. VOID.") from None
    if len(posts) != 1:
        raise H4RQStop(f"{where}: {len(posts)} POSTCOND lines, expected exactly 1")
    p = posts[0]
    for ok, msg in ((p.no_throw, "the helper threw"),
                    (p.windows == 0, f"{p.windows} windows opened"),
                    (p.frames == 0, f"{p.frames} frames opened"),
                    (p.headless, "not headless"),
                    (p.prefs_ok, "the preferences surface was disturbed"),
                    (p.refl_ok, "the reflective-access check failed"),
                    (p.refl_n == expected_refl,
                     f"refl_n={p.refl_n}, expected exactly {expected_refl}")):
        if not ok:
            raise H4RQStop(f"{where}: safety surface not clean: {msg}")
    return {"no_throw": p.no_throw, "windows": p.windows, "frames": p.frames,
            "headless": p.headless, "prefs_ok": p.prefs_ok, "refl_ok": p.refl_ok,
            "refl_n": p.refl_n, "failures": p.failures}


def _procs_one(out: str, *, where: str) -> Dict[str, Any]:
    """Exactly one PROC per jvm. The count IS the process count."""
    try:
        recs = A.parse_procs(out)
    except (ValueError, KeyError, A.HelperOutputError) as e:
        raise H4RQVoidError(
            f"{where}: the PROC output could not be parsed: {e}. VOID.") from None
    if len(recs) != 1:
        raise H4RQStop(
            f"{where}: {len(recs)} PROC lines, expected exactly 1 per jvm under "
            f"the frozen fresh-process lifecycle")
    r = recs[0]
    return {"pid": r.pid, "java_version": r.java_version, "vm": r.vm,
            "headless": r.headless, "prefs_factory": r.prefs_factory}


def observe_query(*, row: Dict[str, Any], state, paths: T1jPaths,
                  budget: QueryBudget, ordinal: int, rep: int) -> Dict[str, Any]:
    """ONE opt-in query subprocess. Records, and STOPS on a frozen condition."""
    where = f"{row['family']}@ply{row['ply']} query rep{rep}"
    moves = [tuple(m) for m in row["prefix"]]
    budget.spend(1)
    try:
        recs, dumps, rc, out = A.query(
            moves, depth=DEPTH, java=paths.java, jar=paths.jar,
            classes=paths.classes, repeats=1, timeout_s=PER_CALL_TIMEOUT_S,
            inject_matchdata=True)
    except subprocess.TimeoutExpired as e:
        raise H4RQVoidError(
            f"{where}: no answer within {PER_CALL_TIMEOUT_S}s ({e}). VOID.") from None
    except A.HelperOutputError as e:
        raise H4RQVoidError(
            f"{where}: the QUERY output could not be parsed ({e}). "
            f"T1j reported: {A.helper_failure_excerpt(e.stdout)}. VOID.") from None

    try:
        return _observe_query_checks(row=row, state=state, where=where,
                                     moves=moves, recs=recs, dumps=dumps,
                                     rc=rc, out=out, ordinal=ordinal, rep=rep)
    except H4RQStop as stop:
        raise _with_context(stop, where=where, row=row, stdout=out) from None


def _observe_query_checks(*, row, state, where, moves, recs, dumps, rc, out,
                          ordinal, rep) -> Dict[str, Any]:
    proc = _procs_one(out, where=where)
    post = _postcond_one(out, expected_refl=QUERY_REFL_N_OPTIN, where=where)

    try:
        mds = A.parse_matchdata(out)
    except (ValueError, KeyError) as e:
        raise H4RQVoidError(f"{where}: MATCHDATA unparseable: {e}. VOID.") from None
    if len(mds) != 1:
        raise H4RQStop(f"{where}: {len(mds)} MATCHDATA lines, expected exactly 1 "
                       f"on the opt-in path")
    md = mds[0]
    if md.pie_rule or md.xsize != A.BOARD_N or md.ysize != A.BOARD_N \
            or not md.ystarts:
        raise H4RQStop(
            f"{where}: the injected MatchData did not read back as frozen -- "
            f"pieRule={md.pie_rule} xsize={md.xsize} ysize={md.ysize} "
            f"ystarts={md.ystarts}. mdPieRule must be false: H4 has no swap rule.",
            where=where, row=row, stdout=out)
    if not md.identity:
        # The helper emits this field SPECIFICALLY to prove `getMatchData()`
        # returned the object we injected. Values matching is not the same as
        # the engine holding OUR object.
        raise H4RQStop(
            f"{where}: MATCHDATA identity=false -- getMatchData() did not "
            f"return the injected object, so the values read back describe "
            f"something else that happens to agree",
            where=where, row=row, stdout=out)

    if len(recs) != 1:
        raise H4RQStop(f"{where}: {len(recs)} query records, expected exactly 1")
    rec = recs[0]

    if rec.null_sentinel or rec.move is None:
        raise H4RQStop(f"{where}: the reply is the null sentinel, not a move")
    if not rec.legal:
        raise H4RQStop(f"{where}: T1j reports its own move {rec.move} illegal")
    if rec.move not in set(state.legal_moves()):
        raise H4RQStop(f"{where}: {rec.move} is illegal in OUR engine")
    if rec.requested_depth != DEPTH:
        raise H4RQStop(f"{where}: requested depth {rec.requested_depth}, not {DEPTH}",
                       where=where, row=row, stdout=out)

    # 🔴 THE QUERY LINE MUST DESCRIBE THE POSITION WE SENT. The dump is checked
    # for coherence separately, but the QUERY record is a DIFFERENT line and was
    # unchecked: it could report another mover or move number while
    # `classify_reply` keys on the outer row's ply, so a reply about a different
    # position would be classified as though it were about this one.
    if rec.q != 1:
        raise H4RQStop(f"{where}: query index q={rec.q}, expected 1",
                       where=where, row=row, stdout=out)
    if rec.move_nr != int(row["ply"]):
        raise H4RQStop(
            f"{where}: the reply reports moveNr={rec.move_nr} but this position "
            f"is at ply {row['ply']}; classification keys on the ply, so a "
            f"record about another move number cannot be classified here",
            where=where, row=row, stdout=out)
    ours = A.PLAYER_TO_T1J[state.to_move]
    if rec.to_move != ours:
        raise H4RQStop(
            f"{where}: the reply reports to_move={rec.to_move!r} but our side to "
            f"move is {state.to_move!r} ({ours!r})",
            where=where, row=row, stdout=out)

    if len(dumps) != 1:
        raise H4RQStop(f"{where}: {len(dumps)} searched-position dumps, expected 1")
    div = list(INT.compare_state(state, dumps[0], moves))
    if div:
        raise H4RQStop(f"{where}: the dump diverges from our position: "
                       + "; ".join(div))

    source = classify_reply(ply=int(row["ply"]), rec=rec, exit_status=rc,
                            failures=post["failures"])
    return {
        "role": "query", "ordinal": ordinal, "rep": rep, "where": where,
        "family": row["family"], "ply": row["ply"],
        "prefix": [list(m) for m in moves], "our_to_move": state.to_move,
        "return_code": rc, "source": source,
        "move": list(rec.move), "move_nr": rec.move_nr,
        "usealphabeta": rec.usealphabeta, "current_max_ply": rec.current_max_ply,
        "completed": rec.completed, "completed_depth": rec.completed_depth,
        "eval_regime": rec.eval_regime, "elapsed_us": rec.elapsed_us,
        "matchdata": {"pie_rule": md.pie_rule, "xsize": md.xsize,
                      "ysize": md.ysize, "ystarts": md.ystarts,
                      "identity": md.identity},
        "proc": proc, "postcond": post, "stdout": out,
    }


def observe_replay(*, row: Dict[str, Any], state, paths: T1jPaths,
                   budget: QueryBudget, ordinal: int) -> Dict[str, Any]:
    """ONE replay subprocess. The default path: refl_n stays 1."""
    where = f"{row['family']}@ply{row['ply']} replay"
    moves = [tuple(m) for m in row["prefix"]]
    budget.spend(1)
    try:
        plies, rc, out = A.replay(
            moves, ply_cap=paths.ply_cap, java=paths.java, jar=paths.jar,
            classes=paths.classes, timeout_s=PER_CALL_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        raise H4RQVoidError(
            f"{where}: no answer within {PER_CALL_TIMEOUT_S}s ({e}). VOID.") from None
    except A.HelperOutputError as e:
        raise H4RQVoidError(
            f"{where}: the replay output could not be parsed ({e}). VOID.") from None

    try:
        return _observe_replay_checks(row=row, state=state, where=where,
                                      moves=moves, plies=plies, rc=rc, out=out,
                                      ordinal=ordinal, paths=paths)
    except H4RQStop as stop:
        raise _with_context(stop, where=where, row=row, stdout=out) from None


def _observe_replay_checks(*, row, state, where, moves, plies, rc, out, ordinal,
                           paths) -> Dict[str, Any]:
    proc = _procs_one(out, where=where)
    post = _postcond_one(out, expected_refl=REPLAY_REFL_N, where=where)
    if rc != 0:
        raise H4RQStop(f"{where}: replay exit {rc}. "
                       f"T1j reported: {A.helper_failure_excerpt(out)}")
    if post["failures"] != 0:
        raise H4RQStop(f"{where}: failures={post['failures']} on a replay")
    if len(plies) != state.ply + 1:
        raise H4RQStop(f"{where}: {len(plies)} ply blocks, expected {state.ply + 1}")
    div = list(INT.compare_state(state, plies[-1], moves))
    if div:
        raise H4RQStop(f"{where}: the replayed final state diverges: "
                       + "; ".join(div))
    return {"role": "replay", "ordinal": ordinal, "where": where,
            "family": row["family"], "ply": row["ply"],
            "prefix": [list(m) for m in moves], "our_to_move": state.to_move,
            "return_code": rc, "n_plies": len(plies), "ply_cap": paths.ply_cap,
            "proc": proc, "postcond": post, "stdout": out}


# ─────────────────────────────────── the run ─────────────────────────────────

def run_qualification(*, paths: T1jPaths, out_path: str,
                      prefixes: Optional[Sequence[Dict[str, Any]]] = None,
                      deadline: Optional[Deadline] = None,
                      budget: Optional[QueryBudget] = None,
                      _compile: Optional[Callable] = None) -> Dict[str, Any]:
    """PUBLIC ENTRY. Refuses while the gate is shut, before anything happens."""
    if not H4_REPAIR_QUALIFICATION_AUTHORIZED:
        raise H4RQError(
            "the H4 repair qualification is UNAUTHORIZED. Gating only the CLI "
            "would protect nothing: a direct Python caller reaches this runner "
            "without passing it. Nothing has been compiled, queried or written.")
    return _run_unguarded(paths=paths, out_path=out_path, prefixes=prefixes,
                          deadline=deadline, budget=budget, _compile=_compile)


def _claim(out_path: str) -> None:
    """Claim the create-only destination ATOMICALLY, before anything runs.

    🔴 IT USED TO BE CLAIMED AFTER ALL 72 SUBPROCESSES. Two failures in one:
    an occupied destination was discovered only after the whole run had been
    spent, and a STOP -- which is a RESULT, and may not be retried -- produced
    NO DURABLE RECORD AT ALL. The single observation of the failure was lost.
    """
    fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    os.close(fd)


def _write_record(out_path: str, report: Dict[str, Any]) -> None:
    """Write into the ALREADY-CLAIMED path. Exclusivity was taken by `_claim`."""
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())


def _run_unguarded(*, paths, out_path, prefixes=None, deadline=None, budget=None,
                   _compile=None):
    """Everything below the gate. PRIVATE, and never a way around the gate."""
    matrix = derive_matrix(prefixes)
    digest = verify_matrix(matrix)
    caps = derived_caps(matrix)
    deadline = deadline or Deadline(limit_s=RUN_DEADLINE_S)
    budget = budget or QueryBudget(cap=caps["total"])
    compile_fn = (_compile if _compile is not None
                  else functools.partial(_compile_helper_verified, paths=paths))
    # 🔴 BEFORE COMPILATION, BEFORE ANY JVM. An occupied destination must be
    # discovered now, not after 72 subprocesses have been spent.
    _claim(out_path)
    deadline.start()
    try:
        with _supervisor(deadline):
            return _stages(matrix, digest, caps, paths, out_path, deadline,
                           budget, compile_fn)
    except H4RQError:
        raise
    except D1VoidError as e:
        raise H4RQVoidError(str(e)) from None
    except D1Error as e:
        raise H4RQError(str(e)) from None


def _stages(matrix, digest, caps, paths, out_path, deadline, budget, compile_fn):
    """The run. A STOP is CAUGHT here so it leaves a durable record.

    🔴 RETRIES ARE FORBIDDEN, so the one observation of a failure is the only
    one there will ever be. Losing it to an exception that wrote nothing would
    mean the qualification stopped and said nothing about why -- exactly the
    gap D1's VOID left, which is why the low-ply qualification had to exist.
    """
    observations: List[Dict[str, Any]] = []
    try:
        return _run_matrix(matrix, digest, caps, paths, out_path, deadline,
                           budget, compile_fn, observations)
    except H4RQStop as stop:
        _write_record(out_path, {
            "stage": "h4_repair_qualification",
            "verdict": "STOP",
            "stop_reason": stop.message,
            "stop_where": stop.where,
            "stop_position": stop.row,
            "stop_stdout": stop.stdout,
            "matrix_sha256": digest,
            "n_positions": len(matrix),
            "observations_completed": len(observations),
            "subprocesses_spent": budget.spent,
            "derived_caps": caps,
            "elapsed_s": deadline.elapsed(),
            "observations": observations,
            "scope": (
                "A STOP is a RESULT and it ENDS the qualification: no repair, "
                "no retry, no improvisation. The observations below are the "
                "ones COMPLETED BEFORE the stop and are not a partial "
                "qualification."),
        })
        raise


def _run_matrix(matrix, digest, caps, paths, out_path, deadline, budget,
                compile_fn, observations):
    try:
        artifacts = compile_fn(deadline)
    except (ToolchainError, D1Error) as e:
        raise H4RQVoidError(f"toolchain or compilation failed: {e}. VOID.") from None
    deadline.check("after helper compilation")

    ordinal = 0
    for row in matrix:
        label = f"{row['family']}@ply{row['ply']}"
        deadline.check(f"before {label}")
        moves = [tuple(m) for m in row["prefix"]]
        state = H4A._state_for(moves, where=label)
        if state.ply != int(row["ply"]):
            raise H4RQVoidError(
                f"{label}: replays to ply {state.ply}, not {row['ply']}. VOID.")
        row["digest_recomputed"] = H4A.digest_of(state)

        for rep in range(repetitions_for(int(row["ply"]))):
            observations.append(observe_query(row=row, state=state, paths=paths,
                                              budget=budget, ordinal=ordinal,
                                              rep=rep))
            ordinal += 1
            deadline.check(f"after {label} query rep{rep}")
        observations.append(observe_replay(row=row, state=state, paths=paths,
                                           budget=budget, ordinal=ordinal))
        ordinal += 1
        deadline.check(f"after {label} replay")

    if budget.spent != caps["total"]:
        raise H4RQStop(
            f"spent {budget.spent} subprocesses, expected the derived "
            f"{caps['total']}")

    deadline.check("before writing the record")
    by_source: Dict[str, int] = {}
    for o in observations:
        if o["role"] == "query":
            by_source[o["source"]] = by_source.get(o["source"], 0) + 1
    report = {
        "stage": "h4_repair_qualification",
        "verdict": "CLEAN",
        "matrix_sha256": digest,
        "matrix_plies": list(MATRIX_PLIES),
        "families": list(FAMILIES),
        "n_positions": len(matrix),
        "n_observations": len(observations),
        "subprocesses_spent": budget.spent,
        "derived_caps": caps,
        "depth": DEPTH,
        "refl_n_optin": QUERY_REFL_N_OPTIN,
        "refl_n_default": QUERY_REFL_N_DEFAULT,
        "refl_n_replay": REPLAY_REFL_N,
        "per_call_timeout_s": PER_CALL_TIMEOUT_S,
        "run_deadline_s": deadline.limit_s,
        "elapsed_s": deadline.elapsed(),
        "source_counts": by_source,
        "toolchain_identity": artifacts,
        "positions": matrix,
        "observations": observations,
        "scope": (
            "Board-plies 0-5 on the frozen prefix set, at depth 6, through the "
            "opt-in h4query mode. Establishes legality, board coherence, which "
            "routine answered, and the process and reflection contracts. It is "
            "NOT a probability estimate: realized moves are reported and the "
            "number of distinct ones is never a pass condition. No game was "
            "played, no seed drawn, no score computed."),
    }
    _write_record(out_path, report)
    return report


def resolve_paths(classes: str) -> T1jPaths:
    """Resolve the VERIFIED toolchain and freeze the cap. Nothing is defaulted."""
    from . import t1j_toolchain as TC
    tc = TC.verified_paths()
    return T1jPaths(java=os.path.join(tc["jdk_home"], "bin", "java"),
                    jar=tc["jar"], classes=classes, ply_cap=PLY_CAP)


EXIT_OK = 0
EXIT_STOP = 2
EXIT_VOID = 3
EXIT_UNEXPECTED = 4
EXIT_UNAUTHORIZED = 5


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI. Refuses while the gate is shut, BEFORE touching anything.

    A STOP exits 2, distinctly from VOID's 3: a stop is a RESULT about the
    repaired helper, and a void is the instrument failing to be readable.
    """
    import argparse
    ap = argparse.ArgumentParser(
        prog="h4_repair_qualification",
        description="H4 repair qualification. IT IS NOT AUTHORIZED.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", required=True)
    a = ap.parse_args(argv)

    if not H4_REPAIR_QUALIFICATION_AUTHORIZED:
        print("the H4 repair qualification is UNAUTHORIZED. No JVM was started, "
              "no position queried, and no file was written.", file=sys.stderr)
        return EXIT_UNAUTHORIZED

    try:                                                      # pragma: no cover
        run_qualification(paths=resolve_paths(a.classes), out_path=a.out)
    except H4RQStop as e:                                     # pragma: no cover
        print(f"STOP: {e}", file=sys.stderr)
        return EXIT_STOP
    except H4RQVoidError as e:                                # pragma: no cover
        print(f"VOID: {e}", file=sys.stderr)
        return EXIT_VOID
    except H4RQError as e:                                    # pragma: no cover
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_UNEXPECTED
    except Exception as e:                                    # noqa: BLE001
        print(f"UNEXPECTED {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_UNEXPECTED
    return EXIT_OK                                            # pragma: no cover


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
