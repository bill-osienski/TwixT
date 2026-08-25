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
from typing import Dict, List, Optional, Sequence, Set, Tuple

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


def compile_helper(javac: str, jar: str, out_dir: str) -> subprocess.CompletedProcess:
    """Compile the COMMITTED helper sources into ``out_dir``.

    T1j itself is neither modified nor rebuilt -- its jar is only a classpath
    entry. Raises if any committed source is missing, so a partial checkout fails
    loudly instead of compiling something else.
    """
    missing = [str(p) for p in JAVA_SOURCES if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"committed helper sources missing: {missing}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [javac, "-Xlint:-options", "-encoding", "UTF-8", "-cp", jar, "-d", out_dir]
        + [str(p) for p in JAVA_SOURCES],
        capture_output=True, text=True,
    )


def replay(
    moves: Sequence[Pos],
    *,
    ply_cap: int,
    java: str,
    jar: str,
    classes: str,
    transform: str = CANONICAL,
) -> Tuple[List[PlyState], int, str]:
    """Advance T1j through ``moves`` (ours, in order), one ply at a time.

    Returns (per-ply states, process exit status, raw stdout). The helper applies
    each move through T1j's own ``Match.setlastMove``.
    """
    if ply_cap is None:
        raise TypeError("ply_cap is required")
    xy = [to_t1j(r, c, transform=transform) for (r, c) in moves]
    args = [
        java,
        f"-Djava.util.prefs.PreferencesFactory={PREFS_FACTORY}",
        "-Djava.awt.headless=true",
        "-cp", f"{jar}:{classes}",
        HELPER_MAIN, "replay", str(ply_cap),
    ] + [f"{x},{y}" for (x, y) in xy]
    p = subprocess.run(args, capture_output=True, text=True)
    return parse_dump(p.stdout), p.returncode, p.stdout


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
