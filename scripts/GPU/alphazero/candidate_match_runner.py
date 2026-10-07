"""THE TWO-AGENT MATCH RUNNER (R1): the tactical candidate against the unchanged incumbent.

`docs/superpowers/2026-10-05-tactical-candidate-evaluation-card.md` §3 R1, §7.1, §11 A1.2-A1.3,
§12 A2.4. Adjacent colour-reversed pairs from the empty official board: in Arm A the candidate
plays red, in Arm B black. Each game builds ONE FRESH agent per colour, and the SEED FOLLOWS THE
COLOUR -- the pair's bundle {red, black} -- never the role. 24x24, cap 280, the winner first.

EXACT REPLAY (§7.1). Until its first override each game IS the incumbent-only game G0, move for
move. So the two games of a pair must agree up to the earlier first override; there, the
overriding game's BASELINE must be the other game's move; and a pair in which nothing fired must
be move-identical. Any breach STOPS the segment: the determinism premise has failed.

BLIND. The record holds every move, every candidate witness and the terminal type, so outcomes
are derivable from the evidence alone -- by design. Nothing outcome-bearing reaches the console
or the journal: no winner, no score, no ply count, no terminal type. Nothing here scores a game.

RUN CONTRACT (the H5 studies'): pure refusals; the parent's create-only CLAIM with a nonce; the
child in its own session under the PARENT's watchdog, which SIGKILLs the process group at the
segment limit (4 h, setup included) AND when one game has been open 15 min (§13 A3.1) -- the
child's own SIGALRM fires 30 s earlier for a clean recorded STOP; every completed decision and
game durable before the next (§13 A3.2); an fsynced blind journal; record.json on every exit,
the completion check inside the recorded boundary; watchdog.json; no automatic retry. A study
segment starts only after every earlier one VALIDATES (§13 A3.3). Gate:
`candidate_match_authorization`, created closed.
"""
from __future__ import annotations

from time import monotonic as _monotonic      # A8.5 item 3: m0 is read before every other import
_T_START = _monotonic()

import collections
import contextlib
import hashlib
import json
import math
import os
import pathlib
import re
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import candidate_match_authorization as AUTH
from . import h4_pilot_feasibility as F
from . import h4_runner as R4
from . import h5_cap_completion as CC
from . import h5_incumbent_defence as D
from . import h5_incumbent_tactics as T
from . import tactical_candidate as TC
from . import twixtbot_g3_reference as G3

MODULE = "scripts.GPU.alphazero.candidate_match_runner"
BOARD, PLY_CAP = 24, 280                   # tests pin both to the tactics card's
COLOURS = ("red", "black")
OTHER = {"red": "black", "black": "red"}
ARMS = {"A": "red", "B": "black"}          # the CANDIDATE's colour in each arm
MODES = ("fixture", "pilot", "study")      # "fixture": synthetic plans, never a public entry
GAME_DEADLINE_S = 900.0                    # §12 A2.4 / §13 A3.1: 15 min per game, the PARENT kills
CHILD_GRACE_S = 30.0                       # §13 A3.1: the child's own alarm fires this much earlier
POLL_S = 0.5                               # §13 A3.1: the parent's polling interval
GAMES_DIR = "games"                        # §13 A3.2: durable per-game evidence
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")   # a pair id names files: no path in it
SEGMENT_DEADLINE_S = 14_400.0              # §12 A2.4: 4 h per segment, setup included
STUDY_CEILING_S = 80 * 3600.0              # §12 A2.2: across all study segments
MARGIN_S = 60.0
PLAN_DIR = "tactical-candidate-plans"
CLAIM_FILE, WATCHDOG_FILE, CLAIM_ENV = "claim.json", "watchdog.json", "CANDIDATE_MATCH_CLAIM"
CHILD_STREAMS = {"stdout": "child.stdout.txt", "stderr": "child.stderr.txt"}
EXIT_OK, EXIT_REFUSED, EXIT_UNAUTHORIZED = 0, 4, 5
CARD = "docs/superpowers/2026-10-05-tactical-candidate-evaluation-card.md"
#: A8.4: the accepted qualification run, pinned byte for byte as committed at this commit
QUALIFICATION_COMMIT = "d815461eb8daa7f462dc8c492316717a0b3045b4"
QUALIFICATION_FILES = {
    "claim.json": "bbf45b8e0572113dfed097fce279175466eb37f4197d572ee8de68142dece49a",
    "child.stderr.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "child.stdout.txt": "c0b2ec1ed5ff6f9e565fb365f891d2a55e91d7dee5327ba8ec4693d4c2aa7dba",
    "log.jsonl": "8a0ed8763616abf1db18b9ab76bba8fce79c017ab8dba9f8722e6a814da9904a",
    "observations.jsonl": "6551e6d3efcdb49fc2f2417ce9a7ab68096ca8ee1f38ee58823fe9cfe2e8dbbd",
    "record.json": "698338279b8d6402b2a1aaf446fdebbd69ab03b833ad0a7937f18f50d65c5ace",
    "watchdog.json": "c33068294a19c06bc9a18cfd80c34c8119e8094bada218e2bd36ba3dd614ffa1"}
QUALIFICATION_MODULE_SHA256 = "bb073d9a5eb9b047667f68976466ac89c24262bbbfd10654a779a1eddfedd805"
QUALIFICATION_CALLS = 109


class MatchRefused(Exception):
    """An input, binding or claim failed: nothing further runs."""


class MatchUnauthorized(Exception):
    """The gate is closed. Nothing was opened, read, built or written."""


class MatchStop(Exception):
    """A check failed during the run: recorded, never retried."""


class MatchTimeout(MatchStop):
    """A hard limit fired."""


def _t(m) -> Tuple[int, int]:
    return (int(m[0]), int(m[1]))


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ─────────────────────────────── the plan ───────────────────────────────

def check_plan(plan: Mapping[str, Any], *, mode: str) -> None:
    """Whole pairs, a seed per colour, every id and seed unique; synthetic seeds negative,
    research seeds positive AND accounted in the shared registry, none spent."""
    if mode not in MODES:
        raise MatchRefused(f"unknown mode {mode!r}; one of {list(MODES)}")
    if not isinstance(plan, Mapping) or plan.get("mode") != mode:
        raise MatchRefused(f"the plan is not a {mode} plan")
    pairs = plan.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise MatchRefused("a plan is one or more whole pairs; none given")
    ids, seeds = [], []
    for p in pairs:
        s = p.get("seeds") if isinstance(p, Mapping) else None
        if not isinstance(s, Mapping) or set(s) != set(COLOURS):
            raise MatchRefused(f"pair {p!r}: seeds must be exactly one per colour")
        ids.append(p.get("pair_id"))
        seeds += [s["red"], s["black"]]
    if any(not isinstance(i, str) or not SAFE_ID.fullmatch(i) for i in ids) or len(set(ids)) != len(ids):
        raise MatchRefused("pair ids must be unique and SAFE: 1-64 of A-Z a-z 0-9 _ -, starting "
                           "alphanumeric (a pair id names its game files)")
    if any(type(x) is not int for x in seeds):
        raise MatchRefused("every seed must be an int")
    if len(set(seeds)) != len(seeds):
        raise MatchRefused("a seed repeats: that is an integrity fault, not a replicate")
    if mode == "fixture":
        if any(x >= 0 for x in seeds):
            raise MatchRefused("a fixture plan uses NEGATIVE seeds only; research seeds are positive")
        return
    if any(x < 0 for x in seeds):
        raise MatchRefused(f"a {mode} plan refuses a negative (synthetic) seed")
    faults = R4.seed_registry_faults([{"seed": x} for x in seeds])
    if faults:
        raise MatchRefused(f"seeds not registered for use, or spent: {faults[:5]}")


# ─────────────────────────────── agents and one game ───────────────────────────────

def production_builder(config, evaluator, identity: Mapping[str, Any]) -> Callable:
    """The ONE construction path: G3's seeded reference agent, its seed bound to the COLOUR; the
    candidate is that agent behind the enabled wrapper (§1), the incumbent the bare agent."""
    def make(colour: str, seed: int, role: str):
        task = {"seed": seed, "reference": identity["reference"],
                "reference_sha1": identity["reference_sha1"], "anchor_colour": OTHER[colour]}
        agent = G3.build_reference_agent(task=task, evaluator=evaluator, colour=colour,
                                         config=config, capture=role == "candidate")
        return TC.TacticalCandidate(agent, enabled=True) if role == "candidate" else agent
    return make


@contextlib.contextmanager
def _alarm(seconds: float, what: str):
    """A HARD limit: SIGALRM interrupts the blocking call itself (as d1_probe's supervisor).
    Fails closed: refuses where it cannot arm, or with nothing left (setitimer(0) disarms)."""
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        raise MatchRefused("a hard limit cannot be armed here (needs SIGALRM on the main thread)")
    if seconds <= 0:
        raise MatchTimeout(f"{what}: no time remained to arm the hard limit")

    def _fire(_signum, _frame):
        raise MatchTimeout(f"{what}: the hard limit of {seconds:.1f} s was exceeded")
    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _compact(w: Mapping[str, Any]) -> Dict[str, Any]:
    """One candidate decision (card §3 R1): the full class only where the rule was active."""
    c = w["class"]
    cls = c if w["rule"] != "inactive" else (c and {"class": c["class"], "T_void": c["T_void"]})
    stat = lambda s: [*s["move"], s["visits"], s["q_root_mover"]]   # noqa: E731
    return {"ply": w["ply"], "rule": w["rule"], "class": cls, "reason": w["reason"],
            "override": w["override"], "unsearched": w["unsearched"], "baseline": stat(w["baseline"]),
            "final": stat(w["final"]), "scan_s": w["scan_s"], "search_s": w["search_s"]}


def game_digest(moves: Sequence, terminal: Mapping[str, Any]) -> str:
    return _sha({"moves": [list(m) for m in moves], "terminal": dict(terminal)})


def play_game(make: Callable, pair: Mapping[str, Any], arm: str, *, limit_s: float,
              clock: Callable[[], float] = time.monotonic,
              on_decision: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    """One game from the empty official board under a HARD limit. Fresh agents; the seed
    follows the colour. The winner is checked before the cap; no move is asked for after."""
    cand = ARMS[arm]
    roles = {c: "candidate" if c == cand else "incumbent" for c in COLOURS}
    t0 = clock()
    with _alarm(limit_s, f"game {pair['pair_id']}-{arm}"):
        agents = {c: make(c, pair["seeds"][c], roles[c]) for c in COLOURS}
        state, moves = T.replay([]), []
        while state.winner() is None and state.ply < PLY_CAP:
            if clock() - t0 > limit_s:
                raise MatchTimeout(f"game {pair['pair_id']}-{arm}: {limit_s:.1f} s passed at ply "
                                   f"{state.ply}")
            move = _t(agents[state.to_move](state))
            if not state.is_valid_placement(*move):
                raise MatchStop(f"game {pair['pair_id']}-{arm}: the illegal move {list(move)} "
                                f"at ply {state.ply}")
            moves.append(list(move))
            if on_decision is not None:               # §13 A3.2: durable before the next decision
                mover = state.to_move
                on_decision({"ply": state.ply, "move": list(move),
                             "witness": _compact(agents[cand].witnesses[-1]) if mover == cand else None})
            state = state.apply_move(move)
    witnesses = agents[cand].witnesses
    own = [i for i in range(len(moves)) if i % 2 == (0 if cand == "red" else 1)]
    if [w["ply"] for w in witnesses] != own or \
            any(w["final"]["move"] != moves[w["ply"]] for w in witnesses):
        raise MatchStop(f"game {pair['pair_id']}-{arm}: the candidate's witnesses are not its "
                        "decisions in this game")
    terminal = {"reason": "win" if state.winner() is not None else "cap", "ply": state.ply}
    return {"pair_id": pair["pair_id"], "arm": arm, "candidate": cand, "roles": roles,
            "seeds": dict(pair["seeds"]), "moves": moves, "terminal": terminal,
            "witnesses": [_compact(w) for w in witnesses], "digest": game_digest(moves, terminal),
            "seconds": clock() - t0}


def first_override(game: Mapping[str, Any]) -> Optional[int]:
    return min((w["ply"] for w in game["witnesses"] if w["override"]), default=None)


def check_pair(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    """§7.1, checked: both games are G0 until the earlier first override."""
    pid = a["pair_id"]
    if (a["arm"], b["arm"], b["pair_id"], a["seeds"]) != ("A", "B", pid, b["seeds"]):
        raise MatchStop(f"pair {pid}: not one A-then-B pair on one seed bundle")
    fa, fb = first_override(a), first_override(b)
    out = {"pair_id": pid, "fired": fa is not None or fb is not None,
           "first_override": {"A": fa, "B": fb}}
    if not out["fired"]:
        if a["moves"] != b["moves"] or a["terminal"] != b["terminal"]:
            raise MatchStop(f"pair {pid}: nothing fired, yet the two games differ -- the "
                            "exact-replay premise (§7.1) failed")
        return {**out, "agree_through": len(a["moves"])}
    k = min(x for x in (fa, fb) if x is not None)
    if a["moves"][:k] != b["moves"][:k]:
        raise MatchStop(f"pair {pid}: the games differ before the first override (ply {k})")
    g, other = (a, b) if fa == k else (b, a)
    base = next(w for w in g["witnesses"] if w["ply"] == k)["baseline"][:2]
    if k >= len(other["moves"]) or other["moves"][k] != base:
        raise MatchStop(f"pair {pid}: at ply {k} the overriding game's baseline {base} is not "
                        "the other game's move")
    return {**out, "agree_through": k}


# ─────────────────────────────── the child's run ───────────────────────────────

def _failure(d: pathlib.Path, step: str, e: BaseException) -> Dict[str, Any]:
    return {"failing_step": step, "exception_type": type(e).__name__, "reason": str(e),
            **CC._store(d, "failure", CC.carried(e))}


def _file_sha(path: str) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _fsync_dir(path: pathlib.Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class _Partial:
    """§13 A3.2: one game's completed decisions, append-only and fsynced line by line --
    outcome-bearing evidence, kept apart from the blind journal."""

    def __init__(self, path: pathlib.Path):
        self._f = os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), "w",
                            encoding="utf-8")
        _fsync_dir(path.parent)

    def __call__(self, decision: Mapping[str, Any]) -> None:
        self._f.write(json.dumps(decision, sort_keys=True) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self) -> None:
        self._f.close()


def _game_file(gdir: pathlib.Path, pair_id: Any, arm: Any, suffix: str) -> pathlib.Path:
    """A game file DIRECTLY inside games/ -- refused otherwise, whatever the plan said."""
    name = f"{pair_id}-{arm}{suffix}"
    path = gdir / name
    if not isinstance(pair_id, str) or not SAFE_ID.fullmatch(pair_id) or arm not in ARMS or \
            path.parent != gdir or path.name != name:
        raise MatchStop(f"the game file {name!r} would not lie directly inside {gdir}")
    return path


_write = os.write                                # the one raw write path (a test seam)


def _write_all(fd: int, raw: bytes) -> None:
    """Every byte, or a recorded failure: a short write continues, zero progress STOPS, and the
    file's size is checked against the bytes meant."""
    view, done = memoryview(raw), 0
    while done < len(raw):
        n = _write(fd, view[done:])
        if n <= 0:
            raise MatchStop(f"a write made no progress at byte {done} of {len(raw)}")
        done += n
    if os.fstat(fd).st_size != len(raw):
        raise MatchStop(f"the file holds {os.fstat(fd).st_size} bytes, not the {len(raw)} written")


def _save_game(gdir: pathlib.Path, g: Mapping[str, Any]) -> Dict[str, Any]:
    """§13 A3.2: the completed game, create-only and fsynced (file and directory)."""
    path = _game_file(gdir, g["pair_id"], g["arm"], ".json")
    name = path.name
    raw = (json.dumps(g, indent=1, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        _write_all(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(gdir)
    return {"pair_id": g["pair_id"], "arm": g["arm"], "digest": g["digest"], "file": name,
            "sha256": hashlib.sha256(raw).hexdigest()}


def _run(plan: Mapping[str, Any], config, identity: Mapping[str, Any], loader: Callable,
         d: pathlib.Path, *, mode: str, builder: Optional[Callable] = None,
         clock: Callable[[], float] = time.monotonic, wall_s: float = SEGMENT_DEADLINE_S - MARGIN_S,
         game_s: float = GAME_DEADLINE_S - CHILD_GRACE_S) -> Dict[str, Any]:
    """The segment. `builder(evaluator) -> make` is a TEST seam; no public entry passes one."""
    check_plan(plan, mode=mode)
    if not _unconsumed(d):
        raise MatchRefused(f"{d} does not hold exactly an unconsumed claim")
    t0 = clock()
    left = lambda: wall_s - (clock() - t0)    # noqa: E731
    log, step, games, pairs, partial, cleanup = None, "claim", [], [], None, []
    status, failure, interrupt = "COMPLETE", None, None
    try:
        log = CC._Log(d / "log.jsonl")
        gdir = d / GAMES_DIR
        os.mkdir(gdir)
        step = "model_load"
        with _alarm(left(), "the segment (model load)"):
            evaluator = loader()
        make = builder(evaluator) if builder is not None else \
            production_builder(config, evaluator, identity)
        log.done("model_load", {"incumbent_identity": dict(identity)})
        for p in plan["pairs"]:
            got = []
            for arm in ARMS:
                step = f"game:{p['pair_id']}-{arm}"
                if left() <= 0:
                    raise MatchTimeout(f"the segment's {wall_s:.0f} s passed before {step}")
                log.done("game_start", {"pair_id": p["pair_id"], "arm": arm})
                partial = _Partial(_game_file(gdir, p["pair_id"], arm, ".partial.jsonl"))
                g = play_game(make, p, arm, limit_s=min(game_s, left()), clock=clock,
                              on_decision=partial)
                h, partial = partial, None
                h.close()
                games.append(_save_game(gdir, g))
                got.append(g)
                log.done("game", {"pair_id": g["pair_id"], "arm": arm, "digest": g["digest"],
                                  "overrides": sum(w["override"] for w in g["witnesses"]),
                                  "seconds": round(g["seconds"], 3)})
            step = f"replay:{p['pair_id']}"
            pc = check_pair(*got)
            pairs.append(pc)
            log.done("pair", {"pair_id": pc["pair_id"], "fired": pc["fired"],
                              "first_override": pc["first_override"]})
        step = "close:journal"                   # A8.5: closing is inside the recorded boundary too
        h, log = log, None
        h.close()
    except (KeyboardInterrupt, SystemExit) as e:
        status, interrupt, failure = "INTERRUPTED", e, _failure(d, step, e)
    except Exception as e:                       # noqa: BLE001 -- a recorded STOP
        status, failure = "STOP", _failure(d, step, e)
    for name, h in (("partial", partial), ("journal", log)):      # after a failure: every handle
        if h is None:                                             # closed, every error recorded
            continue
        try:
            h.close()
        except BaseException as e:               # noqa: BLE001 -- recorded; an interruption re-raised below
            cleanup.append({"handle": name, "exception_type": type(e).__name__, "reason": str(e)})
            if interrupt is None and isinstance(e, (KeyboardInterrupt, SystemExit)):
                status, interrupt = "INTERRUPTED", e
    if status == "COMPLETE":                     # inside the recorded boundary
        step = "completion_check"
        try:
            want = [(p["pair_id"], arm) for p in plan["pairs"] for arm in ARMS]
            got_ = [(g["pair_id"], g["arm"]) for g in games]
            if got_ != want or [x["pair_id"] for x in pairs] != [p["pair_id"] for p in plan["pairs"]]:
                status, failure = "INCOMPLETE", {
                    "failing_step": "completion_check", "exception_type": None,
                    "reason": "the games are not the plan",
                    "missing": [list(k) for k in want if k not in got_],
                    "duplicated": [list(k) for k, n in collections.Counter(got_).items() if n > 1],
                    "extra": [list(k) for k in got_ if k not in want]}
        except (KeyboardInterrupt, SystemExit) as e:
            status, interrupt, failure = "INTERRUPTED", e, _failure(d, step, e)
        except Exception as e:                   # noqa: BLE001 -- a recorded STOP
            status, failure = "STOP", _failure(d, step, e)
    record = {"card": CARD, "mode": mode, "status": status, "failure": failure,
              "plan_sha256": _sha(plan), "plan": plan, "incumbent_identity": dict(identity),
              "wrapper_sha256": _file_sha(TC.__file__), "m1_sha256": _file_sha(T.__file__),
              "board": BOARD, "ply_cap": PLY_CAP, "game_deadline_s": game_s, "deadline_s": wall_s,
              "elapsed_s": clock() - t0, "games": games, "pairs": pairs, "cleanup_failures": cleanup,
              "note": "nothing here scores a game; outcomes are derivable from the game files, by design"}
    with open(d / "record.json", "x", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
    if interrupt is not None:
        raise interrupt
    return record


# ─────────────────────────────── the parent ───────────────────────────────

def _unconsumed(d: pathlib.Path) -> bool:
    if not d.is_dir():
        return False
    names = set(os.listdir(d))
    return CLAIM_FILE in names and names <= {CLAIM_FILE, *CHILD_STREAMS.values()}


def _create(d: pathlib.Path) -> None:
    """A8.5: the one-shot destination, created by THIS invocation; an occupied one is refused and
    never written to."""
    try:
        os.mkdir(d)
    except FileExistsError:
        raise MatchRefused(f"{d} is occupied: each segment is one-shot") from None


@contextlib.contextmanager
def _sigint_held():
    """A8.5: a SIGINT while the destination is created and marked as ours is HELD -- Python runs
    this handler in the main thread whichever thread the kernel delivers to -- and re-raised to the
    restored handler afterwards: inside the caller's protected region, never between the two."""
    held: List[int] = []
    previous = signal.signal(signal.SIGINT, lambda signum, _frame: held.append(signum))
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
        if held:
            signal.raise_signal(signal.SIGINT)


def _claim(d: pathlib.Path, extra: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    """A8.5: claim.json, create-only and fsynced, in a destination this invocation created."""
    import secrets
    nonce = secrets.token_hex(16)
    claim = {"claimed_monotonic": time.monotonic(),
             "claimed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "parent_pid": os.getpid(),
             "nonce_sha256": hashlib.sha256(nonce.encode()).hexdigest(), **extra}
    fd = os.open(d / CLAIM_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        _write_all(fd, json.dumps(claim, indent=1, sort_keys=True).encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(d)
    return claim, nonce


def _record_parent_failure(d: pathlib.Path, e: BaseException, stage: str) -> None:
    """A8.5: a claim, watch or settlement that failed or was interrupted still leaves watchdog.json,
    in a destination this invocation created."""
    if (d / WATCHDOG_FILE).exists():
        return
    try:
        with open(d / WATCHDOG_FILE, "x", encoding="utf-8") as fh:
            json.dump({"card": CARD, "stage": stage,
                       "status": "INTERRUPTED" if isinstance(e, (KeyboardInterrupt, SystemExit))
                       else f"{stage.upper()}_FAILED",
                       "failure": {"exception_type": type(e).__name__, "reason": str(e)},
                       "claim_present": (d / CLAIM_FILE).exists(),
                       "record_present": (d / "record.json").exists()}, fh, indent=1, sort_keys=True)
    except Exception:                            # noqa: BLE001 -- nothing more can be recorded
        pass


def _timing_path(d: pathlib.Path) -> pathlib.Path:
    return d / "timing.json"


def _write_timing(d: pathlib.Path, marks: Sequence[float]) -> None:
    """A8.5 item 3: four intervals between five readings of ONE monotonic clock -- they tile
    [m0, m4] without overlap. Written create-only and fsynced, after watchdog.json. A failure here
    leaves the segment without a valid timing.json, which every later use refuses."""
    m0, m1, m2, m3, m4 = marks
    t = {"startup_s": m1 - m0, "pre_launch_s": m2 - m1, "child_s": m3 - m2, "settlement_s": m4 - m3,
         "clock": "time.monotonic, in the parent",
         "note": "start-up counts from R1's first statement (meaningful when the process was started "
                 "for this run); the interpreter's own start before it is not measured"}
    fd = os.open(_timing_path(d), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        _write_all(fd, json.dumps(t, indent=1, sort_keys=True).encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(d)


def _accept_claim(d: pathlib.Path) -> Dict[str, Any]:
    if not _unconsumed(d):
        raise MatchRefused(f"{d} does not hold exactly an unconsumed claim")
    claim = json.loads((d / CLAIM_FILE).read_text(encoding="utf-8"))
    nonce = os.environ.get(CLAIM_ENV) or ""
    if hashlib.sha256(nonce.encode()).hexdigest() != claim.get("nonce_sha256"):
        raise MatchRefused("the claim's nonce does not match: this process was not started by "
                           "the parent that claimed")
    return claim


def _open_game(d: pathlib.Path) -> Optional[str]:
    """The game the journal shows started and not finished -- read for (pair, arm) only."""
    j = d / "log.jsonl"
    if not j.exists():
        return None
    current = None
    for line in j.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except ValueError:                       # a line still being written
            continue
        if e.get("step") == "game_start":
            current = f"{e['observation']['pair_id']}-{e['observation']['arm']}"
        elif e.get("step") == "game":
            current = None
    return current


def _watch_games(argv: Sequence[str], *, limit_s: float, game_s: float, d: pathlib.Path,
                 env: Mapping[str, str], cwd: str, stdout_path: pathlib.Path,
                 stderr_path: pathlib.Path, poll_s: float = POLL_S) -> Dict[str, Any]:
    """§13 A3.1: the child in its OWN session, streams to create-only files; the PARENT kills its
    whole process group at the segment limit, when one game has been open `game_s` on the
    parent's clock (from first seeing its `game_start`), or when this parent is interrupted.
    Needs nothing from the child but its blind journal. (CC._watch, with the game limit added.)"""
    import subprocess
    t0 = time.monotonic()
    out: Dict[str, Any] = {"child_pid": None, "limit_s": limit_s, "game_limit_s": game_s,
                           "terminated": False, "cause": None, "signal": None, "rc": None,
                           "survivors_killed": False, "game_breach": None}
    try:
        with open(stdout_path, "xb") as so, open(stderr_path, "xb") as se:
            p = subprocess.Popen(list(argv), start_new_session=True, env=dict(env), cwd=cwd,
                                 stdout=so, stderr=se)
    except OSError as e:
        return {**out, "cause": f"the child could not be started ({type(e).__name__}: {e})",
                "elapsed_s": time.monotonic() - t0}
    out["child_pid"] = p.pid
    seen: Dict[str, float] = {}
    try:
        while True:
            try:
                p.wait(timeout=poll_s)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            if now - t0 > limit_s:
                out.update(terminated=True, cause=f"the segment limit of {limit_s:.1f} s from the "
                                                  "claim, measured by the parent")
                break
            g = _open_game(d)
            if g is not None and now - seen.setdefault(g, now) > game_s:
                out.update(terminated=True, game_breach=g,
                           cause=f"game {g} open longer than {game_s:.1f} s on the parent's clock")
                break
    except BaseException as e:                   # noqa: BLE001 -- killed, then re-raised
        out.update(terminated=True, cause=f"the parent was interrupted ({type(e).__name__})",
                   parent_exception=e)
    try:
        os.killpg(p.pid, signal.SIGKILL)         # the child's session IS its group
        killed = True
    except ProcessLookupError:
        killed = False
    out.update(rc=p.wait(), elapsed_s=time.monotonic() - t0,
               signal="SIGKILL" if out["terminated"] else None,
               survivors_killed=killed and not out["terminated"])
    return out


def _inventory(d: pathlib.Path) -> List[Dict[str, Any]]:
    """§13 A3.2: every file under games/ by name, size and sha256 -- no content is read."""
    gdir = d / GAMES_DIR
    if not gdir.is_dir():
        return []
    out = []
    for f in sorted(gdir.iterdir()):
        raw = f.read_bytes()
        out.append({"file": f.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return out


def _settle(d: pathlib.Path, watch: Mapping[str, Any]) -> Dict[str, Any]:
    """watchdog.json: statuses, streams, the journal, the record and the game files by hash --
    never an outcome."""
    rec, present = None, (d / "record.json").exists()
    if present:
        try:
            rec = json.loads((d / "record.json").read_text(encoding="utf-8"))
        except ValueError:
            rec = None
    status = ("TERMINATED" if watch["terminated"] else
              "EXITED_WITHOUT_RECORD" if rec is None else
              "EXITED_ABNORMALLY" if watch["rc"] != 0 and rec.get("status") != "INTERRUPTED"
              else rec.get("status"))
    rc, streams = watch["rc"], {}
    for s, name in sorted(CHILD_STREAMS.items()):
        raw = (d / name).read_bytes() if (d / name).exists() else None
        streams[s] = None if raw is None else {
            "file": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    summary = {"card": CARD, "status": status, "cause": watch["cause"],
               "limit_s": watch["limit_s"], "elapsed_s": watch["elapsed_s"],
               "child_pid": watch["child_pid"], "child_rc": rc,
               "child_exit": "never started" if watch["child_pid"] is None else
               f"killed by signal {CC._signal_name(-rc)}" if rc < 0 else f"exit status {rc}",
               "child_streams": streams, "signal": watch["signal"],
               "survivors_killed": watch["survivors_killed"], "journal": CC._journal(d),
               "game_limit_s": watch.get("game_limit_s"), "game_breach": watch.get("game_breach"),
               "record_sha256": _file_sha(str(d / "record.json")) if present else None,
               "games": _inventory(d),
               "record_present": present, "record_readable": rec is not None,
               "record_status": None if rec is None else rec.get("status")}
    with open(d / WATCHDOG_FILE, "x", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    return summary


# ─────────────────────────────── the public entries ───────────────────────────────

def _gate() -> None:
    if not AUTH.CANDIDATE_MATCH_AUTHORIZED:
        raise MatchUnauthorized("the candidate match is UNAUTHORIZED. Nothing was opened, read, "
                                "built, loaded, played or written.")


def _role(role: str, segment: int) -> None:
    if role not in ("pilot", "study") or type(segment) is not int or segment < 0 or \
            (role == "pilot" and segment != 0):
        raise MatchRefused(f"no {role} segment {segment}: the pilot is segment 0; study segments "
                           "count from 0")


def out_dir(role: str, segment: int) -> pathlib.Path:
    name = "tactical-candidate-pilot" if role == "pilot" else f"tactical-candidate-study-s{segment}"
    return pathlib.Path(R4.EVIDENCE_ROOT) / name


def plan_path(role: str, segment: int) -> pathlib.Path:
    return pathlib.Path(R4.EVIDENCE_ROOT) / PLAN_DIR / f"{role}-s{segment}.json"


def _committed_plan(role: str, segment: int) -> Tuple[Dict[str, Any], str]:
    path = plan_path(role, segment)
    try:
        F.check_durable([str(path)], _fixture=False)
    except F.H4AnalysisRefused as e:
        raise MatchRefused(f"the plan must be committed before any model load: {e}") from None
    raw = path.read_bytes()
    plan = json.loads(raw)
    check_plan(plan, mode=role)
    return plan, hashlib.sha256(raw).hexdigest()


def qualification_binding() -> Dict[str, Any]:
    """A8.4: the accepted qualification run -- exactly its seven committed, unmodified, pinned
    files, COMPLETE and QUALIFIED, on the corpus and the exact 109-call list recomputed NOW. Reads
    only; refuses anything else. Returns the binding a plan and a claim must carry."""
    from . import tactical_corpus as TQ                   # both import this module
    from . import tactical_qualification as Q

    def bad(why: str) -> MatchRefused:
        return MatchRefused(f"the qualification does not bind: {why} (A8.4)")
    if _file_sha(Q.__file__) != QUALIFICATION_MODULE_SHA256:
        raise bad("tactical_qualification.py is not the qualified call-list definition")
    d = Q.out_dir()
    names = sorted(os.listdir(d)) if d.is_dir() else []
    if names != sorted(QUALIFICATION_FILES):
        raise bad(f"{d} does not hold exactly the seven evidence files")
    try:
        F.check_durable([str(d / n) for n in names], _fixture=False)
    except F.H4AnalysisRefused as e:
        raise bad(f"its evidence is not committed and unmodified ({e})") from None
    got = {n: _file_sha(str(d / n)) for n in names}
    for n, want in sorted(QUALIFICATION_FILES.items()):
        if got[n] != want:
            raise bad(f"{n} is not the pinned file")
    try:
        w, rec, claim = (json.loads((d / f).read_text(encoding="utf-8"))
                         for f in (WATCHDOG_FILE, "record.json", CLAIM_FILE))
        obs = [json.loads(x) for x in (d / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
        if (w.get("status"), w.get("verdict"), w.get("child_rc"), w.get("signal")) != \
                ("COMPLETE", "QUALIFIED", 0, None):
            raise bad("its watchdog is not COMPLETE and QUALIFIED with a clean exit")
        if (w.get("record_sha256"), w.get("observations_sha256"), (w.get("journal") or {}).get("sha256")) != \
                (got["record.json"], got["observations.jsonl"], got["log.jsonl"]):
            raise bad("its watchdog hashed other files")
        if (rec.get("status"), rec.get("verdict"), rec.get("failure"), rec.get("cleanup_failures")) != \
                ("COMPLETE", "QUALIFIED", None, []):
            raise bad("its record is not COMPLETE and QUALIFIED without a failure")
        if not rec.get("calls") == rec.get("observed") == len(obs) == QUALIFICATION_CALLS:
            raise bad(f"its record does not hold all {QUALIFICATION_CALLS} calls")
        try:
            corpus, corpus_sha = TQ.committed_corpus()
        except TQ.CorpusRefused as e:
            raise bad(f"the corpus is not accepted now ({e})") from None
        if (claim.get("corpus_sha256"), rec.get("corpus_sha256")) != (corpus_sha, corpus_sha):
            raise bad("it ran on another corpus")
        calls = Q.plan_of(corpus)
        calls_sha = Q._sha(calls)
        if (claim.get("plan_sha256"), rec.get("plan_sha256")) != (calls_sha, calls_sha):
            raise bad("it ran another call list")
        if [o.get("id") for o in obs] != [c["id"] for c in calls]:
            raise bad("its observations are not the call list, in order")
    except MatchRefused:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        raise bad(f"malformed ({type(e).__name__}: {e})") from None
    return {"commit": QUALIFICATION_COMMIT, "files": dict(QUALIFICATION_FILES),
            "corpus_sha256": corpus_sha, "call_list_sha256": calls_sha}


def _bound_plan(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """A8.3-A8.4: the plan carries exactly the binding the guard produces now."""
    binding = qualification_binding()
    if plan.get("qualification") != binding:
        raise MatchRefused("the plan is not bound to the accepted qualification (A8.3, A8.4)")
    return binding


def verification_path(role: str, k: int) -> pathlib.Path:
    """A8.7: the post-segment verification record of one segment."""
    return pathlib.Path(R4.EVIDENCE_ROOT) / "tactical-candidate-verification" / f"{role}-s{k}.json"


def _nonneg(v: Any) -> bool:
    return type(v) in (int, float) and math.isfinite(v) and v >= 0


def _read_timing(d: pathlib.Path, bad: Callable[[str], MatchRefused]) -> Tuple[float, str]:
    """A8.5 item 4: a segment's timing.json -- committed, unmodified, four finite, nonnegative
    intervals. Returns their sum (m4 - m0) and the file's sha256."""
    path = _timing_path(d)
    try:
        F.check_durable([str(path)], _fixture=False)
    except F.H4AnalysisRefused as e:
        raise bad(f"its timing.json is not committed ({e})") from None
    raw = path.read_bytes()
    t = json.loads(raw)
    parts = []
    for key in ("startup_s", "pre_launch_s", "child_s", "settlement_s"):
        v = t.get(key) if isinstance(t, dict) else None
        if not _nonneg(v):
            raise bad(f"its timing {key} {v!r} is not a finite, nonnegative number")
        parts.append(float(v))
    return sum(parts), hashlib.sha256(raw).hexdigest()


def _read_verification(role: str, k: int, files: Mapping[str, str], games: int,
                       bad: Callable[[str], MatchRefused]) -> float:
    """A8.5 item 4: a segment's verification record -- committed, unmodified, for THIS segment,
    bound to the watchdog, record and timing it verified, VERIFIED. Returns its measured time."""
    path = verification_path(role, k)
    try:
        F.check_durable([str(path)], _fixture=False)
    except F.H4AnalysisRefused as e:
        raise bad(f"its verification record is not committed ({e})") from None
    v = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(v, dict) or (v.get("role"), v.get("segment")) != (role, k):
        raise bad("its verification record is for another segment")
    if {f"{n}_sha256": v.get(f"{n}_sha256") for n in files} != {f"{n}_sha256": h for n, h in files.items()}:
        raise bad("its verification record verified other files")
    if v.get("verdict") != "VERIFIED":
        raise bad(f"its verification record says {v.get('verdict')!r}")
    if type(v.get("game_files")) is not int or v["game_files"] != games:
        raise bad(f"its verification record counts {v.get('game_files')!r} game files, not {games}")
    for key in ("t_fixed_s", "t_games_s"):
        if not _nonneg(v.get(key)):
            raise bad(f"its verification {key} {v.get(key)!r} is not a finite, nonnegative number")
    return float(v["t_fixed_s"]) + float(v["t_games_s"])


def _validate_segment(role: str, k: int) -> float:
    """§13 A3.3: an earlier segment, VALIDATED outcome-blind -- durability, the claim, the
    watchdog's hashes of journal and record, the record's plan binding and game sequence, every
    game file's hash; and (A8.5 item 4) its timing.json and its verification record. No move,
    terminal or witness is parsed. Returns its FULL measured time: timing.json's m4 - m0 plus its
    verification run -- never the child time alone. The predecessor checks it made itself lie
    inside its own pre-launch interval: counted there, once."""
    def bad(why: str) -> MatchRefused:
        return MatchRefused(f"{role} segment {k} does not validate: {why}")
    d = out_dir(role, k)
    plan, plan_file_sha = _committed_plan(role, k)
    core = [d / CLAIM_FILE, d / "log.jsonl", d / "record.json", d / WATCHDOG_FILE]
    try:
        F.check_durable([str(x) for x in core], _fixture=False)
    except F.H4AnalysisRefused as e:
        raise bad(f"not committed ({e})") from None
    try:
        claim = json.loads((d / CLAIM_FILE).read_text(encoding="utf-8"))
        w = json.loads((d / WATCHDOG_FILE).read_text(encoding="utf-8"))
        raw = (d / "record.json").read_bytes()
        rec = json.loads(raw)
        if (claim.get("plan_sha256"), claim.get("role"), claim.get("segment")) != (plan_file_sha, role, k):
            raise bad("its claim is for another plan, role or segment")
        if (w.get("status"), w.get("record_status")) != ("COMPLETE", "COMPLETE"):
            raise bad(f"its watchdog says {w.get('status')}/{w.get('record_status')}")
        if (w.get("journal") or {}).get("sha256") != _file_sha(str(d / "log.jsonl")):
            raise bad("the journal is not the one its watchdog hashed")
        if w.get("record_sha256") != hashlib.sha256(raw).hexdigest():
            raise bad("the record is not the one its watchdog hashed")
        if (rec.get("card"), rec.get("mode"), rec.get("status"), rec.get("plan_sha256")) != \
                (CARD, role, "COMPLETE", _sha(plan)) or rec.get("plan") != plan:
            raise bad("the record is not a COMPLETE record of the committed plan")
        want = [(p["pair_id"], arm) for p in plan["pairs"] for arm in ARMS]
        games = rec.get("games")
        if not isinstance(games, list) or [(g.get("pair_id"), g.get("arm")) for g in games] != want:
            raise bad("its games are not the plan's pairs and arms, in order")
        if [x.get("pair_id") for x in rec.get("pairs") or []] != [p["pair_id"] for p in plan["pairs"]]:
            raise bad("its replay checks do not name every planned pair")
        files = []
        for g in games:
            if g.get("file") != f"{g['pair_id']}-{g['arm']}.json":
                raise bad(f"game {g['pair_id']}-{g['arm']} names the file {g.get('file')!r}")
            try:
                files.append(_game_file(d / GAMES_DIR, g["pair_id"], g["arm"], ".json"))
            except MatchStop as e:
                raise bad(str(e)) from None
        try:
            F.check_durable([str(x) for x in files], _fixture=False)
        except F.H4AnalysisRefused as e:
            raise bad(f"a game file is not committed ({e})") from None
        for g, f in zip(games, files):
            if _file_sha(str(f)) != g.get("sha256"):
                raise bad(f"{g['file']} is not the file its record hashed")
        e = w.get("elapsed_s")
        if type(e) not in (int, float) or not math.isfinite(e) or e < 0:
            raise bad(f"its elapsed time {e!r} is not a finite, nonnegative number")
        timing_s, timing_sha = _read_timing(d, bad)
        verified_s = _read_verification(role, k, {"watchdog": _file_sha(str(d / WATCHDOG_FILE)),
                                                  "record": hashlib.sha256(raw).hexdigest(),
                                                  "timing": timing_sha}, len(games), bad)
        return timing_s + verified_s
    except MatchRefused:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        raise bad(f"malformed ({type(e).__name__}: {e})") from None


def _segment_limit(role: str, segment: int, plan: Mapping[str, Any]) -> float:
    """§12 A2.4: 4 h per segment. A study segment needs every earlier segment VALIDATED (§13
    A3.3), and (A8.5 item 4) starts only if the FULL measured time already used -- every
    predecessor's timing.json and verification run -- plus this process's time so far, this
    segment's own 4 h and its plan's verification reserve fit the 80 h ceiling.
    NOT a hard end-to-end 80-hour guarantee: the reserve is a planning estimate (A8.7), and this
    segment's settlement after its limit is not reserved."""
    if role == "pilot":
        return SEGMENT_DEADLINE_S
    reserve = plan.get("verification_reserve_s")
    if not _nonneg(reserve):
        raise MatchRefused(f"the plan's verification_reserve_s {reserve!r} is not a finite, "
                           "nonnegative number (A8.5 item 4)")
    used = sum(_validate_segment("study", k) for k in range(segment))
    now = _monotonic() - _T_START
    if used + now + SEGMENT_DEADLINE_S + reserve > STUDY_CEILING_S:
        raise MatchRefused(f"{used / 3600:.2f} h used and {now:.0f} s in this process: another "
                           f"{SEGMENT_DEADLINE_S / 3600:.0f} h segment and its {reserve:.0f} s "
                           f"verification reserve could pass the {STUDY_CEILING_S / 3600:.0f} h ceiling")
    return SEGMENT_DEADLINE_S


def _child_argv(role: str, segment: int) -> List[str]:
    return [sys.executable, "-m", MODULE, "run-claimed", "--role", role, "--segment", str(segment)]


def run(role: str, segment: int) -> Dict[str, Any]:
    """PUBLIC ENTRY -- the PARENT. Pure refusals (the qualification guard among them, A8.4), THE
    CLAIM, the child under the hard watchdog; from the destination's creation on, `watchdog.json`
    on every exit (A8.5); then `timing.json` (A8.5 item 3). Returns statuses only."""
    m1 = _monotonic()
    _gate()
    _role(role, segment)
    out = out_dir(role, segment)
    if os.path.lexists(out):
        raise MatchRefused(f"{out} is occupied: each segment is one-shot")
    plan, sha = _committed_plan(role, segment)
    binding = _bound_plan(plan)
    limit = _segment_limit(role, segment, plan)
    D._identity(R4.frozen_argmax_config())
    ours, stage = False, "claim"
    try:                                         # from the directory's creation on, every exit is recorded
        with _sigint_held():
            _create(out)
            ours = True
        claim, nonce = _claim(out, {"plan_sha256": sha, "role": role, "segment": segment,
                                    "qualification": binding})
        stage = "settle"
        m2 = _monotonic()
        watch = _watch_games(_child_argv(role, segment),
                             limit_s=limit - (time.monotonic() - claim["claimed_monotonic"]),
                             game_s=GAME_DEADLINE_S, d=out,
                             env={**os.environ, CLAIM_ENV: nonce, "PYTHONUNBUFFERED": "1"},
                             cwd=str(R4.REPO_ROOT), stdout_path=out / CHILD_STREAMS["stdout"],
                             stderr_path=out / CHILD_STREAMS["stderr"])
        m3 = _monotonic()
        summary = _settle(out, watch)
        m4 = _monotonic()
        _write_timing(out, (_T_START, m1, m2, m3, m4))
    except BaseException as e:
        if ours:
            _record_parent_failure(out, e, stage)
        raise
    if "parent_exception" in watch:
        raise watch["parent_exception"]
    return summary


def run_claimed(role: str, segment: int) -> Dict[str, Any]:
    """The CHILD: the gate, the committed plan (durable, the claim's sha, this role and segment),
    the qualification guard (A8.4), the identity, the claim -- its plan, role, segment and
    qualification binding; then the model load and the games."""
    _gate()
    _role(role, segment)
    out = out_dir(role, segment)
    plan, sha = _committed_plan(role, segment)
    binding = _bound_plan(plan)                  # A8.4: in the child too, before the load
    config = R4.frozen_argmax_config()
    identity = D._identity(config)
    claim = _accept_claim(out)
    if (claim.get("plan_sha256"), claim.get("role"), claim.get("segment")) != (sha, role, segment):
        raise MatchRefused("the claim is for another plan, role or segment")
    if claim.get("qualification") != binding:
        raise MatchRefused("the claim is bound to another qualification (A8.4)")
    since = time.monotonic() - claim["claimed_monotonic"]
    return _run(plan, config, identity, lambda: R4.load_production_evaluator(identity), out,
                mode=role, wall_s=SEGMENT_DEADLINE_S - max(since, 0.0) - MARGIN_S)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """PUBLIC ENTRY. The gate is read before the arguments are acted on. Statuses only."""
    import argparse
    ap = argparse.ArgumentParser(prog="candidate_match_runner",
                                 description="Tactical candidate vs incumbent. NOT AUTHORIZED.")
    ap.add_argument("step", choices=("run", "run-claimed"))
    ap.add_argument("--role", choices=("pilot", "study"), required=True)
    ap.add_argument("--segment", type=int, default=0)
    a = ap.parse_args(argv)
    if not AUTH.CANDIDATE_MATCH_AUTHORIZED:
        print("the candidate match is UNAUTHORIZED. Nothing was opened, read, built, loaded, "
              "played or written.", file=sys.stderr)
        return EXIT_UNAUTHORIZED
    try:
        if a.step == "run":
            res = run(a.role, a.segment)
            print(json.dumps({k: res[k] for k in ("status", "record_status", "child_rc")}))
        else:
            print(json.dumps({"status": run_claimed(a.role, a.segment)["status"]}))
    except (MatchRefused, MatchUnauthorized) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
