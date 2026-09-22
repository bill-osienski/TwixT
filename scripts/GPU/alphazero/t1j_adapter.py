"""Adapter for the external T1j engine, qualified by E3b.

This is the reusable product of E3b: the coordinate/player mapping, the process
driver, the external ply-cap semantics, and -- under ``t1j_java/`` -- the Java
helper sources themselves. E4 and anything after it should import this rather
than rebuild it. Given a JDK and the T1j jar, ``compile_helper()`` builds the
helper from the committed sources, so nothing outside the repository is needed.

WHAT THIS IS NOT
----------------
It does not reimplement any T1j rule. Every fact about T1j's state is read back
from T1j's own public API through the scratch-compiled helper. It also cannot
convert a bare board position into T1j state: T1j is advanced by replaying an
ordered move sequence through its own ``Match.setlastMove``, which is the only
path E3b qualified.

THE MAPPING IS NOT UNIQUE
-------------------------
E3b derived the mapping from observed legality maps and then narrowed it by full
lockstep comparison. Four candidates survive -- ``identity`` plus three flips,
all with ``Y=red`` -- and they form a symmetry orbit of the board. TwixT is
invariant under them, so they are indistinguishable by state comparison *by
construction*. ``CANONICAL`` below is one representative; the others are equally
correct. Do not describe it as the unique mapping.

THE PLY CAP IS EXTERNAL
-----------------------
``ply_cap`` is a required keyword-only argument with no default everywhere it
appears. It is a harness limit applied identically to both engines; nothing about
it is read from or attributed to T1j. Each engine's ply is derived from its own
accessor and compared, never copied across.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

BOARD_N = 24
LEGAL_BITS = BOARD_N * BOARD_N   # the serialized legal-cell map is exactly this wide

# The Java helper is COMMITTED, so this module is runnable from a checkout plus a
# JDK and the T1j jar -- nothing has to be reconstructed. compile_helper() builds
# from exactly these paths.
JAVA_SRC_ROOT = Path(__file__).resolve().parent / "t1j_java"
JAVA_SOURCES = (
    JAVA_SRC_ROOT / "e2probe" / "ScratchPrefs.java",
    JAVA_SRC_ROOT / "e2probe" / "ScratchPrefsFactory.java",
    JAVA_SRC_ROOT / "net" / "schwagereit" / "t1j" / "E3bDump.java",
)
HELPER_MAIN = "net.schwagereit.t1j.E3bDump"
PREFS_FACTORY = "e2probe.ScratchPrefsFactory"

# The E4 preflight's GENERIC fixed-position query path: position and depth both
# come from argv. Additive -- the E3b surface above is unchanged.
PREFLIGHT_SOURCES = JAVA_SOURCES + (
    JAVA_SRC_ROOT / "net" / "schwagereit" / "t1j" / "E4Preflight.java",
)
PREFLIGHT_MAIN = "net.schwagereit.t1j.E4Preflight"

Pos = Tuple[int, int]      # ours: (row, col)
T1jXY = Tuple[int, int]    # T1j:  (x, y)

# The four mappings that survived E3b's lockstep comparison. They are a symmetry
# orbit; CANONICAL is a representative, not the unique answer.
SURVIVING_TRANSFORMS = ("identity", "flip_x", "flip_y", "flip_both")
CANONICAL = "identity"

# T1j player constants, read from Board in the qualified run.
T1J_YPLAYER = -1
T1J_XPLAYER = 1
PLAYER_TO_T1J: Dict[str, str] = {"red": "Y", "black": "X"}
T1J_TO_PLAYER: Dict[str, str] = {"Y": "red", "X": "black"}


def to_t1j(row: int, col: int, *, transform: str = CANONICAL) -> T1jXY:
    """Our (row, col) -> T1j (x, y)."""
    if transform == "identity":
        return (col, row)
    if transform == "flip_x":
        return (BOARD_N - 1 - col, row)
    if transform == "flip_y":
        return (col, BOARD_N - 1 - row)
    if transform == "flip_both":
        return (BOARD_N - 1 - col, BOARD_N - 1 - row)
    raise ValueError(f"unknown transform {transform!r}")


def to_ours(x: int, y: int, *, transform: str = CANONICAL) -> Pos:
    """T1j (x, y) -> our (row, col). Inverse of :func:`to_t1j`."""
    if transform == "identity":
        return (y, x)
    if transform == "flip_x":
        return (y, BOARD_N - 1 - x)
    if transform == "flip_y":
        return (BOARD_N - 1 - y, x)
    if transform == "flip_both":
        return (BOARD_N - 1 - y, BOARD_N - 1 - x)
    raise ValueError(f"unknown transform {transform!r}")


def terminal_with_cap(ply: int, natural_terminal: bool, *, ply_cap: int) -> bool:
    """Cap-aware terminality for ONE engine, from that engine's OWN ply.

    ``natural_terminal`` must come from the engine being asked -- T1j's
    ``Board.checkGameOver`` or our ``TwixtState.winner`` -- never copied from the
    other side. The cap is external and applies identically to both.
    """
    if ply_cap is None:
        raise TypeError("ply_cap is required")
    if ply_cap < 0:
        raise ValueError("ply_cap must be >= 0")
    return bool(natural_terminal) or ply >= ply_cap


@dataclass(frozen=True)
class PlyState:
    """One ply of T1j state, entirely read back from T1j's own accessors."""
    ply: int                 # Match.getMoveNr()
    next_player: str         # "Y" or "X", from Match.getNextPlayer()
    term_y: bool             # boardY.checkGameOver()
    term_x: bool             # boardX.checkGameOver()
    pegs: Set[str]           # "x,y,owner"
    bridges: Set[str]        # "x1,y1|x2,y2|owner"
    history: Tuple[T1jXY, ...]   # Match.getMoveX/getMoveY, 1-based, in order
    legal: Set[T1jXY]        # Board.pinAllowed for the player to move

    @property
    def winner(self) -> Optional[str]:
        if self.term_y:
            return "Y"
        if self.term_x:
            return "X"
        return None


_PLY_RE = re.compile(r"^PLY (\d+) ")


def parse_legal_bits(bits: str) -> Set[T1jXY]:
    """Decode the serialized legal-cell map, REQUIRING the full board width.

    Without the width check a truncated serialization decodes to the identical
    set whenever the dropped tail is all zeroes -- and on this board the tail can
    be up to a full column of 24 -- so a silently short dump would qualify. The
    width is therefore part of what is compared, not an incidental detail.
    """
    if len(bits) != LEGAL_BITS:
        raise ValueError(
            f"legal map is {len(bits)} bits, expected exactly {LEGAL_BITS}"
        )
    bad = set(bits) - {"0", "1"}
    if bad:
        raise ValueError(f"legal map has non-binary characters {sorted(bad)}")
    return {(i // BOARD_N, i % BOARD_N) for i, b in enumerate(bits) if b == "1"}


def parse_dump(text: str) -> List[PlyState]:
    """Parse the helper's per-ply dump. Unknown lines are ignored."""
    out: List[PlyState] = []
    hdr: Dict[str, str] = {}
    pegs: Set[str] = set()
    bridges: Set[str] = set()
    hist: Tuple[T1jXY, ...] = ()
    legal: Set[T1jXY] = set()
    pending = False

    def flush() -> None:
        nonlocal pending, pegs, bridges, hist, legal
        if not pending:
            return
        out.append(PlyState(
            ply=int(hdr["moveNr"]),
            next_player=hdr["next"],
            term_y=hdr["termY"] == "true",
            term_x=hdr["termX"] == "true",
            pegs=set(pegs), bridges=set(bridges), history=hist, legal=set(legal),
        ))
        pegs, bridges, hist, legal, pending = set(), set(), (), set(), False

    for line in text.splitlines():
        if _PLY_RE.match(line):
            flush()
            hdr = dict(kv.split("=", 1) for kv in line.split() if "=" in kv)
            pending = True
        elif line.startswith("  PEGS "):
            pegs = set(line[7:].split())
        elif line.startswith("  BRIDGES "):
            bridges = set(line[10:].split())
        elif line.startswith("  HIST "):
            hist = tuple(
                (int(p.split(",")[0]), int(p.split(",")[1])) for p in line[7:].split()
            )
        elif line.startswith("  LEGAL "):
            legal = parse_legal_bits(line[8:].strip())
    flush()
    return out


def compile_helper(javac: str, jar: str, out_dir: str,
                   sources: Optional[Sequence[Path]] = None) -> subprocess.CompletedProcess:
    """Compile the COMMITTED helper sources into ``out_dir``.

    T1j itself is neither modified nor rebuilt -- its jar is only a classpath
    entry. Raises if any committed source is missing, so a partial checkout fails
    loudly instead of compiling something else. ``sources`` defaults to the E3b
    set (resolved at call time); pass ``PREFLIGHT_SOURCES`` to include the E4
    preflight query path.
    """
    # resolved here, not in the signature: a def-time default would silently
    # ignore a caller that reassigns JAVA_SOURCES
    sources = JAVA_SOURCES if sources is None else sources
    missing = [str(p) for p in sources if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"committed helper sources missing: {missing}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [javac, "-Xlint:-options", "-encoding", "UTF-8", "-cp", jar, "-d", out_dir]
        + [str(p) for p in sources],
        capture_output=True, text=True,
    )


def replay(
    moves: Sequence[Pos],
    *,
    ply_cap: int,
    java: str,
    jar: str,
    classes: str,
    timeout_s: Optional[float],
    transform: str = CANONICAL,
) -> Tuple[List[PlyState], int, str]:
    """Advance T1j through ``moves`` (ours, in order), one ply at a time.

    Returns (per-ply states, process exit status, raw stdout). The helper applies
    each move through T1j's own ``Match.setlastMove``.

    ``timeout_s`` is REQUIRED and may not be ``None``, exactly as ``ply_cap`` is.
    Until this parameter existed no caller could bound a replay through this API
    at all, and ``subprocess.run(timeout=None)`` waits forever -- so a hung jvm
    consumed no further query and blocked the run indefinitely. A default here
    would put the protection back in the switched-off state that made ``query``'s
    ``timeout_s: Optional[float] = None`` a protection in name only.
    """
    if ply_cap is None:
        raise TypeError("ply_cap is required")
    if timeout_s is None:
        raise TypeError(
            "timeout_s is required: subprocess.run(timeout=None) waits forever")
    xy = [to_t1j(r, c, transform=transform) for (r, c) in moves]
    args = [
        java,
        f"-Djava.util.prefs.PreferencesFactory={PREFS_FACTORY}",
        "-Djava.awt.headless=true",
        "-cp", f"{jar}:{classes}",
        HELPER_MAIN, "replay", str(ply_cap),
    ] + [f"{x},{y}" for (x, y) in xy]
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)
    try:
        plies = parse_dump(p.stdout)
    except (ValueError, KeyError) as e:
        raise HelperOutputError(
            f"the helper's replay output could not be parsed: {e}", p.stdout) from None
    return plies, p.returncode, p.stdout


def our_snapshot(state, *, transform: str = CANONICAL):
    """Render one of our TwixtState objects in the helper's dump vocabulary."""
    pegs = set()
    for (r, c), owner in state.pegs.items():
        x, y = to_t1j(r, c, transform=transform)
        pegs.add(f"{x},{y},{PLAYER_TO_T1J[owner]}")
    bridges = set()
    for (a, b) in state.bridges:
        owner = state.pegs[a]
        sa = "%d,%d" % to_t1j(*a, transform=transform)
        sb = "%d,%d" % to_t1j(*b, transform=transform)
        lo, hi = (sa, sb) if sa <= sb else (sb, sa)
        bridges.add(f"{lo}|{hi}|{PLAYER_TO_T1J[owner]}")
    return pegs, bridges


# --------------------------------------------------------------- E4 preflight

@dataclass(frozen=True)
class QueryRecord:
    """One fixed-position query: what T1j returned, and whether the depth ran."""
    q: int
    requested_depth: int
    move: Optional[Pos]          # ours (row, col); None only for the null sentinel
    to_move: str                 # "Y" or "X"
    usealphabeta: bool
    current_max_ply: int
    completed_depth: int
    completed: bool
    legal: bool
    null_sentinel: bool
    move_nr: int
    eval_regime: str
    elapsed_us: int


_KV_RE = re.compile(r"(\w+)=(\S+)")


class HelperOutputError(ValueError):
    """The helper's output could not be parsed. CARRIES THE TRANSCRIPT.

    The parsers raise bare `ValueError`/`KeyError`, which discards the stdout
    that would explain the failure -- and stdout is only in scope inside `query`
    and `replay`, so no caller can recover it afterwards. This is raised there
    instead, with the output attached, so a caller can refuse WITH the evidence.

    Subclasses `ValueError` so nothing that already handled a parse failure
    changes behaviour.
    """

    def __init__(self, message: str, stdout: str):
        super().__init__(message)
        self.stdout = stdout


def parse_queries(text: str, *, transform: str = CANONICAL) -> List[QueryRecord]:
    """Parse QUERY lines. Unknown lines are ignored; malformed ones raise."""
    out: List[QueryRecord] = []
    for line in text.splitlines():
        if not line.startswith("QUERY "):
            continue
        kv = dict(_KV_RE.findall(line))
        missing = {"q", "requested_depth", "move_x", "move_y", "to_move", "usealphabeta",
                   "currentMaxPly", "completed_depth", "completed", "legal",
                   "null_sentinel", "moveNr", "eval_regime", "elapsed_us"} - set(kv)
        if missing:
            raise ValueError(f"QUERY line missing fields {sorted(missing)}: {line!r}")
        x, y = int(kv["move_x"]), int(kv["move_y"])
        sentinel = kv["null_sentinel"] == "true"
        out.append(QueryRecord(
            q=int(kv["q"]),
            requested_depth=int(kv["requested_depth"]),
            move=None if sentinel else to_ours(x, y, transform=transform),
            to_move=kv["to_move"],
            usealphabeta=kv["usealphabeta"] == "true",
            current_max_ply=int(kv["currentMaxPly"]),
            completed_depth=int(kv["completed_depth"]),
            completed=kv["completed"] == "true",
            legal=kv["legal"] == "true",
            null_sentinel=sentinel,
            move_nr=int(kv["moveNr"]),
            eval_regime=kv["eval_regime"],
            elapsed_us=int(kv["elapsed_us"]),
        ))
    return out


def query(
    moves: Sequence[Pos],
    *,
    depth: int,
    java: str,
    jar: str,
    classes: str,
    repeats: int = 1,
    timeout_s: Optional[float] = None,
    transform: str = CANONICAL,
    inject_matchdata: bool = False,
) -> Tuple[List[QueryRecord], List[PlyState], int, str]:
    """Ask T1j for a move at ``depth`` from the frozen position ``moves``.

    ``repeats`` > 1 rebuilds the position from scratch that many times inside ONE
    jvm -- the E3a structure. Returns (query records, the position T1j searched
    as parsed by :func:`parse_dump`, exit status, raw stdout).

    ``depth`` is the requested FIXED ply. Wall-clock mode is not reachable from
    here: the helper forces ``mdFixedPly = true``.

    ``inject_matchdata`` selects the opt-in H4 mode (``h4query``). DEFAULT OFF:
    every existing caller keeps the exact behaviour it was qualified with, and
    a test asserts a default call's argv still says ``query``.
    """
    if depth < 3:
        raise ValueError(
            "T1j's deepening loop starts at currentMaxPly=3, so a requested depth "
            f"below 3 executes no search at all; got {depth}")
    xy = [to_t1j(r, c, transform=transform) for (r, c) in moves]
    if inject_matchdata and repeats != 1:
        raise ValueError(
            "inject_matchdata is only available on the single-query mode; the "
            "determinism mode reuses one jvm's Zobrist salt and answers a "
            "different question")
    if repeats != 1:
        mode = ["determinism", str(repeats), str(depth)]
    elif inject_matchdata:
        # 🔴 THE OPT-IN H4 MODE, DEFAULT OFF. A distinct mode name rather than a
        # flag on `query`, so a default caller's argv is byte-identical to what
        # it was qualified with. It is BEHAVIOURAL: the helper injects MatchData
        # so board-ply 0 does not throw in InitialMoves.firstMove(), and it
        # performs a FOURTH reflective access, so the opt-in path reports
        # refl_n=4 while every default caller still reports 3.
        mode = ["h4query", str(depth)]
    else:
        mode = ["query", str(depth)]
    args = [
        java,
        f"-Djava.util.prefs.PreferencesFactory={PREFS_FACTORY}",
        "-Djava.awt.headless=true",
        "-cp", f"{jar}:{classes}",
        PREFLIGHT_MAIN,
    ] + mode + [f"{x},{y}" for (x, y) in xy]
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)
    try:
        recs = parse_queries(p.stdout, transform=transform)
        dumps = parse_dump(p.stdout)
    except (ValueError, KeyError) as e:
        raise HelperOutputError(
            f"the helper's query output could not be parsed: {e}", p.stdout) from None
    return recs, dumps, p.returncode, p.stdout


@dataclass(frozen=True)
class ProcRecord:
    """The helper's PROC line -- process identity for one jvm."""
    pid: int
    java_version: str
    vm: str
    headless: str
    prefs_factory: str


@dataclass(frozen=True)
class MatchDataRecord:
    """The helper's MATCHDATA line: what `Match.getMatchData()` actually holds.

    READ BACK THROUGH THE PUBLIC GETTER by the helper, not hashed from our own
    input object -- a digest of what we just built proves only that we can hash
    it. Emitted only by the opt-in `h4query` mode.
    """
    pie_rule: bool
    xsize: int
    ysize: int
    ystarts: bool
    identity: bool


def parse_matchdata(text: str) -> List[MatchDataRecord]:
    """Parse MATCHDATA lines. Absent on every default path, by design."""
    out: List[MatchDataRecord] = []
    for line in text.splitlines():
        if not line.startswith("MATCHDATA "):
            continue
        kv = dict(_KV_RE.findall(line))
        missing = {"pieRule", "xsize", "ysize", "ystarts", "identity"} - set(kv)
        if missing:
            raise ValueError(f"MATCHDATA line missing fields {sorted(missing)}: {line!r}")
        out.append(MatchDataRecord(pie_rule=kv["pieRule"] == "true",
                                   xsize=int(kv["xsize"]), ysize=int(kv["ysize"]),
                                   ystarts=kv["ystarts"] == "true",
                                   identity=kv["identity"] == "true"))
    return out


def parse_procs(text: str) -> List[ProcRecord]:
    """Parse PROC lines. One per jvm, so the count IS the process count."""
    out: List[ProcRecord] = []
    for line in text.splitlines():
        if not line.startswith("PROC "):
            continue
        kv = dict(_KV_RE.findall(line))
        missing = {"pid", "java_version", "vm", "headless", "prefs_factory"} - set(kv)
        if missing:
            raise ValueError(f"PROC line missing fields {sorted(missing)}: {line!r}")
        out.append(ProcRecord(pid=int(kv["pid"]), java_version=kv["java_version"],
                              vm=kv["vm"], headless=kv["headless"],
                              prefs_factory=kv["prefs_factory"]))
    return out


#: How much of the helper's own transcript a refusal may carry. BOUNDED on
#: purpose: each dump holds a 576-character legal-cell map, and an unbounded
#: excerpt would bury the FAIL lines it exists to surface.
FAILURE_EXCERPT_LINES = 12
FAILURE_EXCERPT_CHARS = 800

#: Lines that carry the helper's own verdict. `FAIL <what>` is printed by its
#: `req()` for each failed check, `THREW` by its catch-all, and POSTCOND carries
#: the failures counter that produced the exit status.
_VERDICT_PREFIXES = ("FAIL ", "THREW", "POSTCOND ")


def helper_failure_excerpt(out: str) -> str:
    """The helper's OWN explanation of a non-zero exit, bounded.

    E4Preflight exits 3 when its `failures` counter is non-zero, and it prints a
    `FAIL <what>` line for every check that failed. This module used to raise
    "exit {rc} with {n} query records" and DISCARD `out` -- so the one thing that
    said WHY went in the bin.

    That is not hypothetical. The single authorized D1 run aborted with exactly
    that message on 2026-08-28, and the reason it failed is still unknown
    because the transcript was dropped.

    Falls back to the tail rather than to silence, skipping dump body lines,
    because a refusal that reports nothing is worse than one that reports
    roughly the right neighbourhood.
    """
    lines = [l.strip() for l in out.splitlines() if l.startswith(_VERDICT_PREFIXES)]
    if not lines:
        lines = [l.strip() for l in out.splitlines()
                 if l.strip() and not l.startswith(("  ", "PLY "))][-FAILURE_EXCERPT_LINES:]
    text = " | ".join(lines[:FAILURE_EXCERPT_LINES])
    if not text:
        return "(the helper produced no readable output)"
    return text[:FAILURE_EXCERPT_CHARS] + ("..." if len(text) > FAILURE_EXCERPT_CHARS else "")


@dataclass(frozen=True)
class PostCond:
    """The helper's POSTCOND line: the safety surface, read rather than assumed."""
    no_throw: bool
    windows: int
    frames: int
    headless: bool
    prefs_ok: bool
    refl_ok: bool
    refl_n: int
    failures: int
    #: The helper's OWN preference observations -- the two values `prefs_ok`
    #: compared, before and after, as it saw them. ADDITIVE (2026-09-05): None
    #: when the helper ran the earlier E4Preflight source, or is E3bDump. They
    #: are not part of `clean`; `prefs_ok` is still the verdict and the abort.
    prefs_before: Optional[str] = None
    prefs_after: Optional[str] = None
    count_before: Optional[int] = None
    count_after: Optional[int] = None

    @property
    def clean(self) -> bool:
        return (self.no_throw and self.windows == 0 and self.frames == 0 and self.headless
                and self.prefs_ok and self.refl_ok and self.failures == 0)


_PREFS_OBS_FIELDS = ("prefs_before", "prefs_after", "count_before", "count_after")
_POSTCOND_BASE_FIELDS = frozenset({"no_throw", "windows", "frames", "headless", "prefs_ok",
                                   "refl_ok", "refl_n", "failures"})


def _prefs_observation(kv: Dict[str, str]) -> Dict[str, Any]:
    """The four observation fields from one POSTCOND key/value map, typed.

    🔴 NONE OR ALL FOUR. Review reproduced `prefs_ok=true prefs_before=ABSENT
    prefs_after=ERROR` with count_after MISSING parsing as CLEAN: the
    self-agreement check required all four and a partial line simply skipped
    it. A partial observation is an unreadable instrument -> ValueError, which
    `parse_postconds` wraps with the transcript. Four Nones = the helper emitted
    no observation at all (the earlier E4Preflight source, or E3bDump).
    """
    present = [k for k in _PREFS_OBS_FIELDS if k in kv]
    if present and len(present) != len(_PREFS_OBS_FIELDS):
        raise ValueError(
            f"partial preference observation: {present} present, "
            f"{[k for k in _PREFS_OBS_FIELDS if k not in kv]} missing")
    if not present:
        return {k: None for k in _PREFS_OBS_FIELDS}
    return {"prefs_before": kv["prefs_before"], "prefs_after": kv["prefs_after"],
            "count_before": int(kv["count_before"]),
            "count_after": int(kv["count_after"])}


def postcond_prefs_observation(text: str) -> Optional[Dict[str, Any]]:
    """The helper's prefs observation, plus `prefs_ok`, from ANY text holding a
    POSTCOND segment -- a full transcript or a bounded excerpt, where POSTCOND
    does not start a line. Reads the LAST "POSTCOND " onwards (the helper prints
    it last).

    🔴 UNKNOWN IS None, NOT FOUR Nones. A segment cut short by the excerpt bound,
    or a partial observation, used to read as four Nones -- indistinguishable
    from "the helper ran the earlier source". Missing fields cannot identify an
    older helper. So: None when there is no POSTCOND segment, when the segment
    ends in the excerpt's truncation marker, when any BASE field is missing, or
    when the observation is partial or unparseable. Four Nones ONLY for a
    complete segment that carries no observation field at all.
    """
    at = text.rfind("POSTCOND ")
    if at < 0:
        return None
    segment = text[at:].split(" | ", 1)[0].strip()
    if segment.endswith("..."):
        return None
    kv = dict(_KV_RE.findall(segment))
    if _POSTCOND_BASE_FIELDS - set(kv):
        return None
    try:
        obs = _prefs_observation(kv)
    except ValueError:
        return None
    return {"prefs_ok": kv["prefs_ok"] == "true", **obs}


def parse_postconds(text: str) -> List[PostCond]:
    """Parse POSTCOND lines. One per jvm run."""
    out: List[PostCond] = []
    for line in text.splitlines():
        if not line.startswith("POSTCOND "):
            continue
        kv = dict(_KV_RE.findall(line))
        missing = {"no_throw", "windows", "frames", "headless", "prefs_ok", "refl_ok",
                   "refl_n", "failures"} - set(kv)
        if missing:
            # CARRIES THE TRANSCRIPT, like the query and replay parsers. This is
            # the only scope that still has it: `check_postcond` consumes the
            # text inside the binder, so a bare ValueError propagating out of
            # `make_binder` leaves every caller unable to say what was
            # unreadable -- and unable to tell an unreadable instrument from a
            # T1j observation, which is the distinction the low-ply card turns on.
            raise HelperOutputError(
                f"the helper's POSTCOND output could not be parsed: line missing "
                f"fields {sorted(missing)}: {line!r}", text)
        try:
            obs = _prefs_observation(kv)
            post = PostCond(
                no_throw=kv["no_throw"] == "true", windows=int(kv["windows"]),
                frames=int(kv["frames"]), headless=kv["headless"] == "true",
                prefs_ok=kv["prefs_ok"] == "true", refl_ok=kv["refl_ok"] == "true",
                refl_n=int(kv["refl_n"]), failures=int(kv["failures"]), **obs)
        except (ValueError, KeyError) as e:
            # A COMPLETE line whose numeric fields will not parse. The
            # missing-field branch above was already wrapped; this one was not,
            # so `int("bad")` escaped as a bare ValueError with no transcript --
            # and on the replay path, where `check_postcond` has already consumed
            # the text, that left every caller unable to say what was unreadable.
            raise HelperOutputError(
                f"the helper's POSTCOND output could not be parsed: {e} in "
                f"{line!r}", text) from None
        # 🔑 TWO CHANNELS ON ONE LINE MUST SAY ONE THING. `prefs_ok` is the
        # verdict the helper computed from exactly these four values. When the
        # observation is present and contradicts the verdict, the instrument is
        # unreadable -- refused, with the transcript, not recorded as either.
        if all(v is not None for v in obs.values()):
            unchanged = (obs["prefs_before"] == obs["prefs_after"]
                         and obs["count_before"] == obs["count_after"])
            if unchanged != post.prefs_ok:
                raise HelperOutputError(
                    f"the helper's POSTCOND line disagrees with itself: prefs_ok="
                    f"{str(post.prefs_ok).lower()} but the observation it was "
                    f"computed from {'did not change' if unchanged else 'changed'}: "
                    f"{line!r}", text)
        out.append(post)
    return out
