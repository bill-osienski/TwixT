"""H5 CAP-COMPLETION DIAGNOSTIC -- EXPLORATORY.

Card: `docs/superpowers/2026-10-02-t1j-h5-cap-completion-diagnostic-card.md`, with §6 A1,
§7 A2 (Stage-1 recording) and §8 A3 (Stage-2 recording). 🔴 THE CARD IS THE AUTHORITY.
No game, no strength estimate, no rescoring, no H4-H5 pooling; nothing is inferred about
any search mechanism from a chosen move.

Three public steps, each reading the gate FIRST, each its own authorization:
  freeze  -- the frozen set re-derived from the committed H5 records -> 00_frozen_set.json
  stage1  -- every frozen witness replayed in BOTH rules engines (no agent is asked)
  stage2  -- the unchanged agents at the frozen positions; its permission is DERIVED from
             the committed, integrity-bound Stage-1 record (§6 A1.2), never from a flag
Three episode INPUTS (not 13 situations); all 13 provenance links and their frequencies
travel with every record (§6).
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import d0_postmortem as D0
from . import e4_screen_integration as INT
from . import h4_pilot_feasibility as F
from . import h4_runner as R4
from . import h5_cap_completion_authorization as AUTH
from . import h5_runner as H5
from . import h5_weakness_review as WR
from . import t1j_adapter as A
from .game.twixt_state import TwixtState

LABEL = "EXPLORATORY"
CARD = "docs/superpowers/2026-10-02-t1j-h5-cap-completion-diagnostic-card.md"
MANIFEST = "docs/superpowers/evidence/t1j-h5-manifests/study_manifest.json"
MANIFEST_SHA256 = "e08c5f352eaabc9fac2fa73c48a432abbbdc23caf8ec5c4c52735de8bb619d32"
FROZEN_FILE, STAGE1_DIR, STAGE2_DIR = "00_frozen_set.json", "01_stage1", "02_stage2"
K_T1J, N_INCUMBENT, STAGE2_WALL_S = 3, 2, 3600.0
SEED = -1                       # the qualifications' synthetic seed; no research seed
OUTCOMES = ("AGREE", "RECONSTRUCTION_DISAGREEMENT", "TERMINAL_DISAGREEMENT", "UNAVAILABLE")
EXIT_OK, EXIT_REFUSED, EXIT_UNAUTHORIZED = 0, 4, 5

#: Card §1.1, frozen: three episode INPUTS with their 13 provenance links (Arm B).
EPISODES: Dict[str, Dict[str, Any]] = {
    "E1": {"members": ("h5s0-21-B", "h5s1-14-B", "h5s1-27-B", "h5s1-32-B", "h5s1-54-B",
                       "h5s1-65-B", "h5s2-19-B"), "t1j_ply": 39,
           "t1j_digest": "66ef7606efa05e032b0ceba0f04cb2720147b416cdae1710461baf291217b61b",
           "inc_digest": "26d7d7ad2abc9be3246a15eff538bb652ab9d945a268911cb849667e09cc16ce",
           "W": ((17, 9),), "t1j_recorded": (16, 4), "inc_recorded": (1, 2),
           "saving": ((16, 9), (17, 9))},
    "E2": {"members": ("h5s0-69-B", "h5s2-23-B", "h5s2-35-B", "h5s3-36-B", "h5s3-45-B"),
           "t1j_ply": 57,
           "t1j_digest": "3644a590483f504b8cf6041284540adfbc402eb03a93295d51832d0c06e20873",
           "inc_digest": "466a8ecead28b58f01a4ee2b66045f3e779c1e6e3b39c37a7510613227ee3c06",
           "W": ((12, 5),), "t1j_recorded": (11, 7), "inc_recorded": (1, 4),
           "saving": ((12, 5),)},
    "E3": {"members": ("h5s3-32-B",), "t1j_ply": 81,
           "t1j_digest": "ce4196689c58b9586e31349053e06ff2fe05e7cd0e186d75e490e1b123aad98d",
           "inc_digest": "98a84694d4c9218aaad41e9ac5c63115ff75c0f6255765f0cf717438606929aa",
           "W": ((17, 7),), "t1j_recorded": (14, 8), "inc_recorded": (11, 10),
           "saving": ((16, 6), (17, 7))},
}
#: Card §1.2 with §6 A1.1: each control's threat and acceptable-reply rule pinned.
CONTROLS: Dict[str, Dict[str, Any]] = {
    "CT1": {"game": "h5s0-14-B", "n": 30, "agent": "t1j", "role": "complete", "threat": None,
            "recorded": (2, 15),
            "digest": "0a0f0e6db54de4233c3e00037263241d58457bd4dd42d9752a50a1ed8cc5434f"},
    "CT2": {"game": "h5s0-09-B", "n": 58, "agent": "t1j", "role": "complete", "threat": None,
            "recorded": (23, 2),
            "digest": "7afd27fb538832f6a4ff720e0699a943bb12e171e5fbb088028268347c97bdf7"},
    "CT3": {"game": "h5s0-09-A", "n": 39, "agent": "t1j", "role": "complete", "threat": None,
            "recorded": (9, 0),
            "digest": "5ae39981eadd70d243855a81a9a16c467820412d4c27fd776452c247b59382d3"},
    "CI1": {"game": "h5s2-05-B", "n": 41, "agent": "incumbent", "role": "complete",
            "threat": None, "recorded": (13, 10),
            "digest": "b8f2977e636bc7f46d3b8c79157c58fc6e19359f21359fd469aab03f6c9cd339"},
    "CI2": {"game": "h5s0-15-A", "n": 64, "agent": "incumbent", "role": "complete",
            "threat": None, "recorded": (15, 16),
            "digest": "5aaef1686e29e9c906a090fe77a15b61c85aec836b7b240083033a8dedbd209a"},
    "CD1": {"game": "h5s0-21-B", "n": 41, "agent": "incumbent", "role": "defend",
            "threat": ((17, 9),), "recorded": (16, 9),
            "digest": "ab4b4552f037c1388a9dcb7f80caebdc8f79ab5950736633294ea96eb80bbbe6"},
    "CD2": {"game": "h5s0-69-B", "n": 127, "agent": "incumbent", "role": "defend",
            "threat": ((13, 5),), "recorded": (12, 5),
            "digest": "1ce728b6ef11e1ac6601a6906a942b53e4dc695b65505c308ed601f280784800"},
}


class CapRefused(Exception):
    """An input, binding or permission failed: nothing further runs."""


class CapUnauthorized(Exception):
    """The gate is closed. Nothing was opened, read, built or written."""


class CapStop(Exception):
    """A recorded STOP raised by this module (the Stage-2 deadline, a call returning no
    move). A §4B refusal is NOT converted: it is recorded under its own type (§8 A3.4)."""


# ─────────────────────────────── board facts (ours) ───────────────────────────────

def _t(m) -> Tuple[int, int]:
    return (int(m[0]), int(m[1]))


def prefix_digest(moves: Sequence) -> str:
    """Card §1: sha256 of the canonical JSON list [[r,c],...]."""
    return hashlib.sha256(json.dumps([list(_t(m)) for m in moves],
                                     separators=(",", ":")).encode()).hexdigest()


def state_of(moves: Sequence) -> TwixtState:
    s = TwixtState(active_size=A.BOARD_N, to_move="red")
    for m in moves:
        s = s.apply_move(_t(m))
    return s


def immediate_wins(state: TwixtState) -> List[Tuple[int, int]]:
    """The mover's immediate winning moves (D0's full scan)."""
    return sorted(_t(m) for m in D0.one_ply_threats(state))


def threats_against(state: TwixtState) -> List[Tuple[int, int]]:
    """The OPPONENT's immediate wins were it to move (board threats)."""
    return sorted(_t(m) for m in D0.one_ply_threats(D0._flip(state)))


def acceptable(state: TwixtState, move, role: str) -> bool:
    """§6 A1.1. Any immediate winning move is acceptable (a win counts as saving); for a
    `defend` role so is any move after which the opponent has NO immediate win. Never
    the historical move as such."""
    if role not in ("complete", "defend"):
        raise CapRefused(f"unknown role {role!r}")
    move = _t(move)
    if not state.is_valid_placement(*move):
        return False
    if move in set(immediate_wins(state)):
        return True
    return role == "defend" and not D0.one_ply_threats(state.apply_move(move))


def has_winning_path(pegs: Mapping[Tuple[int, int], str],
                     bridges: Iterable[Tuple[Tuple[int, int], Tuple[int, int]]],
                     colour: str, size: int = A.BOARD_N) -> bool:
    """§6 A1.3: does `colour` hold a bridge-connected chain touching both of its goal
    edges IN THIS GRAPH? Applied to each engine's own reported graph; the path itself is
    not compared, so different valid paths never disagree."""
    adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = collections.defaultdict(list)
    for a, b in bridges:
        if pegs.get(a) == colour and pegs.get(b) == colour:
            adj[a].append(b)
            adj[b].append(a)
    axis = 0 if colour == "red" else 1
    start = [p for p, c in pegs.items() if c == colour and p[axis] == 0]
    seen, stack = set(start), list(start)
    while stack:
        p = stack.pop()
        if p[axis] == size - 1:
            return True
        for q in adj[p]:
            if q not in seen:
                seen.add(q)
                stack.append(q)
    return False


def ours_graph(state: TwixtState):
    return dict(state.pegs), [(a, b) for a, b in state.bridges]


def t1j_graph(tp) -> Tuple[Dict[Tuple[int, int], str], List]:
    """T1j's REPORTED pegs and bridges, in our coordinates -- for OUR independent
    validation of T1j's board; T1j reports no path."""
    pegs = {}
    for s in tp.pegs:
        x, y, p = s.split(",")
        pegs[A.to_ours(int(x), int(y))] = A.T1J_TO_PLAYER[p]
    bridges = []
    for s in tp.bridges:
        a, b, _p = s.split("|")
        bridges.append(tuple(A.to_ours(*map(int, e.split(","))) for e in (a, b)))
    return pegs, bridges


# ─────────────────────────────── the frozen set (§1, §6 A1.1) ───────────────────────────────

def build_frozen_set(games: Sequence[Mapping[str, Any]],
                     episodes: Mapping[str, Mapping[str, Any]] = EPISODES,
                     controls: Mapping[str, Mapping[str, Any]] = CONTROLS) -> Dict[str, Any]:
    """Re-derive and VERIFY every pinned fact from the source games; refuse on any
    mismatch -- a digest, a member's prefix, a threat coordinate, a witness, a rule."""
    by_id = {g["task_id"]: g for g in games}
    frozen: Dict[str, Any] = {"label": LABEL, "card": CARD, "episodes": {}, "controls": {}}

    def need(game_id):
        if game_id not in by_id:
            raise CapRefused(f"source game {game_id} is not among the committed records")
        return by_id[game_id]

    for e, spec in episodes.items():
        p = spec["t1j_ply"]
        for gid in spec["members"]:
            g = need(gid)
            if prefix_digest(g["moves"][:p - 1]) != spec["t1j_digest"] or \
                    prefix_digest(g["moves"][:p - 2]) != spec["inc_digest"]:
                raise CapRefused(f"{e}: member {gid}'s prefix does not hash to the pin")
            if g["actors"][p - 1] != "t1j" or g["actors"][p - 2] != "incumbent":
                raise CapRefused(f"{e}: member {gid} does not have T1j at ply {p} and the "
                                 f"incumbent at ply {p - 1}")
            if (_t(g["moves"][p - 1]), _t(g["moves"][p - 2])) != \
                    (_t(spec["t1j_recorded"]), _t(spec["inc_recorded"])):
                raise CapRefused(f"{e}: member {gid}'s recorded moves are not the pinned ones")
        moves = [_t(m) for m in need(spec["members"][0])["moves"]]
        t1j_pos, inc_pos = moves[:p - 1], moves[:p - 2]
        st, si = state_of(t1j_pos), state_of(inc_pos)
        W = [_t(w) for w in spec["W"]]
        if immediate_wins(st) != sorted(W):
            raise CapRefused(f"{e}: T1j's immediate wins are {immediate_wins(st)}, not the "
                             f"pinned witness {W}")
        if acceptable(st, spec["t1j_recorded"], "complete"):
            raise CapRefused(f"{e}: T1j's recorded move completes -- not a missed win")
        if threats_against(si) != sorted(W):
            raise CapRefused(f"{e}: the incumbent's threat is {threats_against(si)}, not the "
                             f"pinned {W}")
        #: every saving move: the mover's immediate wins (a win counts as saving) plus
        #: every move leaving no immediate threat -- by the weakness review's EXACT
        #: reduction (the opponent's wins after any mover move are a subset of T),
        #: which its tests verify against the full scan.
        saving = sorted(set(immediate_wins(si)) |
                        {_t(m) for m in WR.saving_moves_of(si, threats_against(si))})
        if saving != sorted(_t(s) for s in spec["saving"]):
            raise CapRefused(f"{e}: the saving moves are {saving}, not the pinned "
                             f"{sorted(_t(s) for s in spec['saving'])}")
        if acceptable(si, spec["inc_recorded"], "defend"):
            raise CapRefused(f"{e}: the incumbent's recorded move defends -- not a miss")
        frozen["episodes"][e] = {
            "members": list(spec["members"]), "frequency": len(spec["members"]),
            "t1j_position": {"moves": [list(m) for m in t1j_pos], "digest": spec["t1j_digest"],
                             "mover": st.to_move},
            "incumbent_position": {"moves": [list(m) for m in inc_pos],
                                   "digest": spec["inc_digest"], "mover": si.to_move},
            "W": [list(w) for w in W], "t1j_recorded": list(_t(spec["t1j_recorded"])),
            "incumbent_recorded": list(_t(spec["inc_recorded"])),
            "saving": [list(s) for s in saving]}
    for c, spec in controls.items():
        g = need(spec["game"])
        n = spec["n"]
        pos = [_t(m) for m in g["moves"][:n]]
        if prefix_digest(pos) != spec["digest"]:
            raise CapRefused(f"{c}: the position does not hash to the pin")
        if g["actors"][n] != spec["agent"] or _t(g["moves"][n]) != _t(spec["recorded"]):
            raise CapRefused(f"{c}: the recorded reply is not {spec['agent']}'s pinned move")
        s = state_of(pos)
        threat = threats_against(s)
        if spec["role"] == "defend" and threat != sorted(_t(t) for t in spec["threat"]):
            raise CapRefused(f"{c}: the opponent's immediate wins are {threat}, not the "
                             f"pinned threat {sorted(_t(t) for t in spec['threat'])}")
        if not acceptable(s, spec["recorded"], spec["role"]):
            raise CapRefused(f"{c}: the recorded move does not satisfy its {spec['role']} rule")
        frozen["controls"][c] = {
            "game": spec["game"], "agent": spec["agent"], "role": spec["role"],
            "position": {"moves": [list(m) for m in pos], "digest": spec["digest"],
                         "mover": s.to_move},
            "threat": [list(t) for t in threat] if spec["role"] == "defend" else None,
            "recorded": list(_t(spec["recorded"]))}
    return frozen


def position_digests(frozen: Mapping[str, Any]) -> Dict[str, str]:
    out = {}
    for e, ep in frozen["episodes"].items():
        out[f"{e}/t1j"] = ep["t1j_position"]["digest"]
        out[f"{e}/incumbent"] = ep["incumbent_position"]["digest"]
    for c, ct in frozen["controls"].items():
        out[c] = ct["position"]["digest"]
    return out


def frozen_bytes(frozen: Mapping[str, Any]) -> bytes:
    return (json.dumps(frozen, indent=1, sort_keys=True) + "\n").encode()


# ─────────────────────────────── Stage 1 (§2, §6 A1.3) ───────────────────────────────

def stage1_items(frozen: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Card §2: the EXACT item set -- 18 for the frozen card set."""
    items = []
    for e, ep in frozen["episodes"].items():
        tpos, ipos = ep["t1j_position"]["moves"], ep["incumbent_position"]["moves"]
        for w in ep["W"]:
            items.append({"id": f"{e}/t1j+W{w[0]},{w[1]}", "base": tpos, "apply": [w],
                          "expect": "win"})
        items.append({"id": f"{e}/t1j+recorded", "base": tpos,
                      "apply": [ep["t1j_recorded"]], "expect": "not_terminal"})
        for s in ep["saving"]:
            for w in ep["W"]:
                if s == w:
                    items.append({"id": f"{e}/incumbent+{s[0]},{s[1]}=W", "base": ipos,
                                  "apply": [s], "expect": "occupied", "hole": w})
                else:
                    items.append({"id": f"{e}/incumbent+{s[0]},{s[1]}+W{w[0]},{w[1]}",
                                  "base": ipos, "apply": [s, w], "expect": "not_terminal"})
    for c, ct in frozen["controls"].items():
        pos = ct["position"]["moves"]
        if ct["role"] == "complete":
            items.append({"id": f"{c}/witness", "base": pos, "apply": [ct["recorded"]],
                          "expect": "win"})
        else:
            for t in ct["threat"]:
                if t == ct["recorded"]:
                    items.append({"id": f"{c}/block=threat", "base": pos,
                                  "apply": [ct["recorded"]], "expect": "occupied", "hole": t})
                else:
                    items.append({"id": f"{c}/block+threat{t[0]},{t[1]}", "base": pos,
                                  "apply": [ct["recorded"], t], "expect": "not_terminal"})
    ids = [i["id"] for i in items]
    if len(set(ids)) != len(ids):
        raise CapRefused(f"duplicate Stage-1 item ids: {ids}")
    return items


def ours_check(item: Mapping[str, Any]) -> Tuple[List[TwixtState], List[Tuple[int, int]], List[str]]:
    """Our engine on one item: every state, the move list, and any unmet expectation."""
    moves = [_t(m) for m in item["base"]]
    states = [state_of([])]
    for m in moves:
        states.append(states[-1].apply_move(m))
    problems = []
    for m in item["apply"]:
        if not states[-1].is_valid_placement(*_t(m)):
            problems.append(f"ours: {list(m)} is illegal")
            return states, moves, problems
        moves.append(_t(m))
        states.append(states[-1].apply_move(_t(m)))
    final, w = states[-1], states[-1].winner()
    mover = states[-2].to_move
    if item["expect"] == "win":
        if w != mover or not has_winning_path(*ours_graph(final), mover):
            problems.append(f"ours: no win for {mover} (winner {w})")
    elif item["expect"] == "not_terminal":
        if w is not None:
            problems.append(f"ours: terminal, winner {w}")
    elif item["expect"] == "occupied":
        if final.is_valid_placement(*_t(item["hole"])):
            problems.append(f"ours: the hole {item['hole']} is still legal")
    return states, moves, problems


def classify(item: Mapping[str, Any], states: Sequence[TwixtState], moves: Sequence,
             result: Mapping[str, Any]) -> Dict[str, Any]:
    """One item's outcome from T1j's replay `result` = {plies, rc, out, error}."""
    if result.get("error") is not None:
        return {"outcome": "UNAVAILABLE", "reason": str(result["error"])}
    if result.get("rc") != 0:
        return {"outcome": "UNAVAILABLE", "reason": f"T1j replay exit {result.get('rc')}"}
    plies = result.get("plies") or []
    if len(plies) != len(moves) + 1:
        return {"outcome": "UNAVAILABLE",
                "reason": f"T1j reported {len(plies)} plies, expected {len(moves) + 1}"}
    terminal_div, other_div = [], []
    for i, tp in enumerate(plies):
        for d in INT.compare_state(states[i], tp, moves[:i]):
            (terminal_div if d.startswith("terminal") else other_div).append(f"ply {i}: {d}")
    if other_div:
        return {"outcome": "RECONSTRUCTION_DISAGREEMENT", "reason": "; ".join(other_div[:4])}
    if terminal_div:
        return {"outcome": "TERMINAL_DISAGREEMENT", "reason": "; ".join(terminal_div[:4])}
    tg = t1j_graph(plies[-1])
    win_colours = [c for c in ("red", "black") if has_winning_path(*tg, c)]
    final_winner = states[-1].winner()
    if item["expect"] == "win":
        if win_colours != [final_winner]:
            return {"outcome": "TERMINAL_DISAGREEMENT",
                    "reason": f"T1j's reported graph holds a winning path for {win_colours}, "
                              f"terminal flags say {final_winner}"}
    elif win_colours:
        return {"outcome": "TERMINAL_DISAGREEMENT",
                "reason": f"T1j's reported graph holds a winning path for {win_colours} "
                          f"on a non-terminal position"}
    if item["expect"] == "occupied" and A.to_t1j(*_t(item["hole"])) in plies[-1].legal:
        return {"outcome": "RECONSTRUCTION_DISAGREEMENT",
                "reason": f"the hole {item['hole']} is legal for T1j"}
    return {"outcome": "AGREE", "reason": None}


NO_STDERR = "unavailable: t1j_adapter.replay does not return stderr"


class _Log:
    """§7 A2.1: append-only `log.jsonl`, flushed and fsynced after each completed step,
    so completed observations survive even a kill."""

    def __init__(self, path: pathlib.Path):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        self._f = os.fdopen(fd, "w", encoding="utf-8")

    def done(self, step: str, observation: Any) -> None:
        self._f.write(json.dumps({"step": step, "observation": observation},
                                 sort_keys=True) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self) -> None:
        self._f.close()


def _text(v) -> Optional[str]:
    if v is None:
        return None
    return v.decode("utf-8", "replace") if isinstance(v, (bytes, bytearray)) else str(v)


def carried(exc: BaseException) -> Dict[str, Any]:
    """§7 A2.2: what a failure CARRIES -- stdout, stderr, return code -- each present or
    explicitly unavailable, with why. Never an empty string standing in for 'unknown'."""
    import subprocess
    if isinstance(exc, subprocess.TimeoutExpired):
        out = _text(exc.stdout if exc.stdout is not None else exc.output)
        err = _text(exc.stderr)
        return {"stdout": out, "stdout_note": None if out is not None else
                "unavailable: the timeout carried no stdout",
                "stderr": err, "stderr_note": None if err is not None else
                "unavailable: the timeout carried no stderr",
                "rc": None, "rc_note": f"unavailable: killed at the {exc.timeout} s timeout"}
    if isinstance(exc, A.HelperOutputError):
        out = _text(getattr(exc, "stdout", None))
        return {"stdout": out, "stdout_note": None if out is not None else
                "unavailable: the exception carried no stdout",
                "stderr": None, "stderr_note": NO_STDERR,
                "rc": None, "rc_note": "unavailable: raised before the exit status was returned"}
    out, err = _text(getattr(exc, "stdout", None)), _text(getattr(exc, "stderr", None))
    rc = getattr(exc, "returncode", None)
    return {"stdout": out, "stdout_note": None if out is not None else
            f"unavailable: {type(exc).__name__} carried no stdout",
            "stderr": err, "stderr_note": None if err is not None else
            f"unavailable: {type(exc).__name__} carried no stderr",
            "rc": rc, "rc_note": None if rc is not None else
            f"unavailable: {type(exc).__name__} carried no return code"}


def _store(d: pathlib.Path, stem: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    """Each present stream written create-only beside the record, its sha256 recorded;
    an absent stream keeps its explicit note."""
    streams = {}
    os.makedirs(d / "streams", exist_ok=True)
    for s in ("stdout", "stderr"):
        if obs.get(s) is None:
            streams[s] = None
            continue
        body = obs[s].encode()
        name = f"streams/{stem}.{s}.txt"
        with open(d / name, "xb") as fh:
            fh.write(body)
        streams[s] = {"file": name, "sha256": hashlib.sha256(body).hexdigest()}
    return {"streams": streams, "stdout_note": obs.get("stdout_note"),
            "stderr_note": obs.get("stderr_note"), "rc": obs.get("rc"),
            "rc_note": obs.get("rc_note")}


def _observe(it, states, moves, replay, d: pathlib.Path) -> Dict[str, Any]:
    """One item: T1j's replay, kept whole whether it returned or raised."""
    try:
        res = dict(replay(list(moves)))
        obs = {"stdout": res.get("out"), "stdout_note": None if res.get("out") is not None
               else "unavailable: the replay returned no stdout",
               "stderr": res.get("stderr"),
               "stderr_note": None if res.get("stderr") is not None else NO_STDERR,
               "rc": res.get("rc"), "rc_note": None if res.get("rc") is not None
               else "unavailable: the replay returned no exit status"}
        exc_type, verdict = None, classify(it, states, moves, res)
    except Exception as e:                       # noqa: BLE001 -- recorded, never hidden
        obs, exc_type = carried(e), type(e).__name__
        verdict = {"outcome": "UNAVAILABLE", "reason": f"{exc_type}: {e}"}
    return {"id": it["id"], "expect": it["expect"], "moves_applied": it["apply"],
            "exception_type": exc_type, **verdict,
            **_store(d, it["id"].replace("/", "__"), obs)}


def _stage1(frozen: Mapping[str, Any], frozen_sha: str,
            compile_fn: Callable[[], Tuple[Any, Dict[str, Any]]],
            replay_factory: Callable[[Any], Callable], out_dir: pathlib.Path) -> Dict[str, Any]:
    """§7 A2.1. PURE refusals first (our engine on every item); then the CLAIM, before
    the compile; then everything inside a recorded handler -- a record on EVERY exit."""
    checked = []
    for it in stage1_items(frozen):
        states, moves, problems = ours_check(it)
        if problems:
            raise CapRefused(f"{it['id']}: our own engine does not meet the frozen "
                             f"expectation: {problems}")
        checked.append((it, states, moves))
    d = pathlib.Path(out_dir) / STAGE1_DIR
    try:
        os.mkdir(d)                              # THE CLAIM: create-only, before compiling
    except FileExistsError:
        raise CapRefused(f"{STAGE1_DIR} is occupied: Stage 1 is one-shot") from None
    log, stage, rows, binding = None, "claim", [], None
    status, failure, interrupt = "COMPLETE", None, None
    try:
        os.mkdir(d / "streams")
        log = _Log(d / "log.jsonl")
        stage = "compile"
        paths, content = compile_fn()
        log.done("compile", {"content": content})
        stage = "binding"
        binding = {"frozen_set_sha256": frozen_sha, "position_digests": position_digests(frozen),
                   "toolchain": content}
        log.done("binding", binding)
        replay = replay_factory(paths)
        for it, states, moves in checked:
            stage = f"replay:{it['id']}"
            row = _observe(it, states, moves, replay, d)
            rows.append(row)
            log.done("item", row)
    except (KeyboardInterrupt, SystemExit) as e:
        status, interrupt = "INTERRUPTED", e
        failure = {"failing_stage": stage, "exception_type": type(e).__name__,
                   "reason": str(e), **_store(d, "failure", carried(e))}
    except Exception as e:                       # noqa: BLE001 -- a recorded STOP
        status = "STOP"
        failure = {"failing_stage": stage, "exception_type": type(e).__name__,
                   "reason": str(e), **_store(d, "failure", carried(e))}
    finally:
        if log is not None:
            log.close()
    record = {"label": LABEL, "card": CARD, "status": status, "failure": failure,
              "binding": binding, "binding_note": None if binding is not None else
              "unavailable: the run stopped before the binding was computed",
              "items": rows,
              "all_agree": status == "COMPLETE" and bool(rows)
              and all(r["outcome"] == "AGREE" for r in rows),
              "provenance": {e: {"members": ep["members"], "frequency": ep["frequency"]}
                             for e, ep in frozen["episodes"].items()}}
    with open(d / "record.json", "x", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
    if interrupt is not None:
        raise interrupt
    return record


# ─────────────────────────────── Stage-2 permission (§6 A1.2) ───────────────────────────────

def stage2_permission(stage1_dir: pathlib.Path, frozen: Mapping[str, Any],
                      binding: Mapping[str, Any], *, _fixture: bool) -> Dict[str, Any]:
    """Refuses unless the Stage-1 record is committed and unmodified, its item ids are
    EXACTLY the frozen set (no omission, duplicate or extra), every item AGREE, its
    binding equal to the re-derived one, and every transcript committed and hashing to
    its recorded sha256. `_fixture` asks the file's own temporary repository and is
    never reachable from a public entry."""
    rec_path = pathlib.Path(stage1_dir) / "record.json"
    try:
        F.check_durable([str(rec_path)], _fixture=_fixture)
    except F.H4AnalysisRefused as e:
        raise CapRefused(f"the Stage-1 record is not durable: {e}") from None
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    if rec.get("status") != "COMPLETE":
        raise CapRefused(f"the Stage-1 record's status is {rec.get('status')!r}, not "
                         f"COMPLETE (failure: {rec.get('failure')})")
    want = collections.Counter(i["id"] for i in stage1_items(frozen))
    got = collections.Counter(r.get("id") for r in rec.get("items", []))
    if got != want:
        raise CapRefused(f"the Stage-1 item set is not exact: missing "
                         f"{sorted(set(want) - set(got))}, extra {sorted(set(got) - set(want))}, "
                         f"duplicated {sorted(k for k, n in got.items() if n > 1)}")
    bad = [r["id"] for r in rec["items"] if r.get("outcome") != "AGREE"]
    if bad:
        raise CapRefused(f"Stage-1 items not AGREE: {bad}")
    if rec.get("binding") != dict(binding):
        diff = sorted(k for k in set(binding) | set(rec.get("binding") or {})
                      if (rec.get("binding") or {}).get(k) != binding.get(k))
        raise CapRefused(f"the Stage-1 binding differs in {diff}")
    root = pathlib.Path(stage1_dir).resolve()
    for r in rec["items"]:
        if not (r.get("streams") or {}).get("stdout"):
            raise CapRefused(f"{r['id']}: an AGREE item with no stored stdout")
        for s, stored in sorted((r.get("streams") or {}).items()):
            if stored is None:
                continue
            t = pathlib.Path(stage1_dir) / stored["file"]
            if not str(t.resolve()).startswith(str(root)):
                raise CapRefused(f"{r['id']}: its {s} lies outside the Stage-1 directory")
            try:
                F.check_durable([str(t)], _fixture=_fixture)
            except F.H4AnalysisRefused as e:
                raise CapRefused(f"{r['id']}: {s} transcript not durable: {e}") from None
            if hashlib.sha256(t.read_bytes()).hexdigest() != stored["sha256"]:
                raise CapRefused(f"{r['id']}: its {s} does not hash to its record")
    return rec


# ─────────────────────────────── Stage 2 (§3, §6 A1.1, §8 A3) ───────────────────────────────

def _positions(frozen: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for e, ep in frozen["episodes"].items():
        out.append({"id": f"{e}/t1j", "agent": "t1j", "role": "complete", "control": False,
                    "moves": ep["t1j_position"]["moves"], "recorded": ep["t1j_recorded"]})
        out.append({"id": f"{e}/incumbent", "agent": "incumbent", "role": "defend",
                    "control": False, "moves": ep["incumbent_position"]["moves"],
                    "recorded": ep["incumbent_recorded"]})
    for c, ct in frozen["controls"].items():
        out.append({"id": c, "agent": ct["agent"], "role": ct["role"], "control": True,
                    "moves": ct["position"]["moves"], "recorded": ct["recorded"]})
    return out


def stage2_expected(frozen: Mapping[str, Any]) -> List[Tuple[str, str, int]]:
    """§8 A3.5: the EXACT call set, derived from the frozen set APART from the run loop --
    18 T1j queries and 14 incumbent decisions for the card's set."""
    n = {"t1j": K_T1J, "incumbent": N_INCUMBENT}
    who = [(f"{e}/{a}", a) for e in frozen["episodes"] for a in ("t1j", "incumbent")]
    who += [(c, ct["agent"]) for c, ct in frozen["controls"].items()]
    return sorted((p, a, k) for p, a in who for k in range(n[a]))


NO_QUERY_STDERR = "unavailable: t1j_adapter.query does not return stderr"
IN_PROCESS = "not applicable: the incumbent decides in-process; no process stream exists"


def recording_query(ev: Dict[str, Any]) -> Callable:
    """§8 A3.3: THE diagnostic boundary around `t1j_adapter.query` -- the same arguments
    in, THE object out, THE exception re-raised. What the query returned or carried is
    appended to `ev["queries"]`; nothing else in the agent changes."""
    def keep(*args, **kwargs):
        try:
            result = A.query(*args, **kwargs)
        except BaseException as e:
            ev.setdefault("queries", []).append(
                {**carried(e), "exception_type": type(e).__name__, "reason": str(e)})
            raise
        out = result[3]
        ev.setdefault("queries", []).append(
            {"stdout": out, "stdout_note": None if out is not None else
             "unavailable: the query returned no stdout", "stderr": None,
             "stderr_note": NO_QUERY_STDERR, "rc": result[2], "rc_note": None,
             "exception_type": None, "reason": None})
        return result
    return keep


def _call_evidence(d: pathlib.Path, stem: str, agent: str, ev: Mapping[str, Any]
                   ) -> Dict[str, Any]:
    """One call's evidence, its streams written create-only NOW, returned or raised."""
    if agent != "t1j":
        return {"processes": None, "processes_note": IN_PROCESS, "observations": None}
    procs = [{"exception_type": q["exception_type"], "reason": q["reason"],
              **_store(d, f"{stem}.q{j}", q)} for j, q in enumerate(ev.get("queries", []))]
    return {"processes": procs, "processes_note": None if procs else
            "unavailable: no query process was started", "observations": ev.get("observations")}


def _readings(frozen: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
              controls_ok: bool) -> Dict[str, Any]:
    readings = {}
    for e in frozen["episodes"]:
        t = [r for r in rows if r["position"] == f"{e}/t1j"]
        i = [r for r in rows if r["position"] == f"{e}/incumbent"]
        k = sum(r["acceptable"] for r in t)
        reading = ("uninterpretable" if not controls_ok else
                   "AGENT_CHOICE (reproduced)" if t and k == 0 else
                   "not reproduced" if t and k == len(t) else "mixed")
        readings[e] = {"t1j_completes": f"{k}/{len(t)}",
                       "incumbent_blocks": f"{sum(r['acceptable'] for r in i)}/{len(i)}",
                       "t1j_equals_recorded": sum(r["equals_recorded"] for r in t),
                       "incumbent_equals_recorded": sum(r["equals_recorded"] for r in i),
                       "members": frozen["episodes"][e]["members"],
                       "frequency": frozen["episodes"][e]["frequency"], "reading": reading}
    return readings


def _stage2(frozen: Mapping[str, Any], permission: Mapping[str, Any],
            stage1: Mapping[str, Any], compile_fn: Callable, incumbent_factory: Callable,
            t1j_factory: Callable, out_dir: pathlib.Path, *,
            clock: Callable[[], float] = time.monotonic, wall_s: float = STAGE2_WALL_S
            ) -> Dict[str, Any]:
    """§8 A3. Reached only after the pure refusals, the Stage-1 permission included. The
    CLAIM comes before the compile and the model load; ONE deadline, started at the claim,
    covers both and every call; each row is persisted before the next call; a record on
    EVERY exit. Judged by §6 A1.1's rules, never by the historical move (reported apart)."""
    from . import d1_probe as D1
    if permission.get("status") != "COMPLETE" or not permission.get("all_agree"):
        raise CapRefused("Stage 2 needs a COMPLETE, all-AGREE Stage-1 record")
    d = pathlib.Path(out_dir) / STAGE2_DIR
    try:
        os.mkdir(d)                              # THE CLAIM: before compiling or loading
    except FileExistsError:
        raise CapRefused(f"{STAGE2_DIR} is occupied: Stage 2 is one-shot") from None
    deadline = D1.Deadline(limit_s=wall_s, clock=clock).start()
    log, step, rows, partial, compiled, identity = None, "claim", [], None, None, None
    status, failure, interrupt = "COMPLETE", None, None

    def due(next_step):
        if deadline.elapsed() > wall_s:
            raise CapStop(f"the {wall_s:.0f} s Stage-2 deadline passed before {next_step}")

    try:
        os.mkdir(d / "streams")
        log = _Log(d / "log.jsonl")
        step = "compile"
        with D1._supervisor(deadline):           # enforced MID-step too (SIGALRM)
            paths, compiled = compile_fn(deadline)
            log.done("compile", {"content": compiled})
            step = "model_load"
            due(step)
            incumbent_choose, identity = incumbent_factory()
            log.done("model_load", {"incumbent_identity": identity})
            t1j_choose = t1j_factory(paths)
            for pos in _positions(frozen):
                state, t1j = state_of(pos["moves"]), pos["agent"] == "t1j"
                for k in range(K_T1J if t1j else N_INCUMBENT):
                    tag = f"{pos['id']}#{k}"
                    step = f"{'query' if t1j else 'decision'}:{tag}"
                    due(step)
                    ev: Dict[str, Any] = {}
                    try:
                        got = dict((t1j_choose if t1j else incumbent_choose)(
                            pos["moves"], tag, ev))
                    finally:
                        evidence = _call_evidence(d, f"{pos['id'].replace('/', '__')}.k{k}",
                                                  pos["agent"], ev)
                        partial = {"step": step, **evidence}
                    move = got.pop("move", None)
                    if move is None:
                        raise CapStop(f"{step}: no move was returned")
                    row = {"position": pos["id"], "agent": pos["agent"], "query": k,
                           "role": pos["role"], "control": pos["control"],
                           "move": list(_t(move)),
                           "acceptable": acceptable(state, move, pos["role"]),
                           "equals_recorded": list(_t(move)) == list(pos["recorded"]),
                           **evidence, "raw": got}
                    rows.append(row)
                    log.done("call", row)
                    partial = None
    except (KeyboardInterrupt, SystemExit) as e:
        status, interrupt = "INTERRUPTED", e
        failure = {"failing_step": step, "exception_type": type(e).__name__,
                   "reason": str(e), **_store(d, "failure", carried(e)), "partial": partial}
    except Exception as e:                       # noqa: BLE001 -- a recorded STOP
        status = "STOP"
        failure = {"failing_step": step, "exception_type": type(e).__name__,
                   "reason": str(e), **_store(d, "failure", carried(e)), "partial": partial}
    finally:
        if log is not None:
            log.close()
    want = stage2_expected(frozen)
    done = sorted((r["position"], r["agent"], r["query"]) for r in rows)
    if status == "COMPLETE" and done != want:
        status = "INCOMPLETE"
    controls_ok = episodes = None
    if status == "COMPLETE":
        controls_ok = all(r["acceptable"] for r in rows if r["control"])
        episodes = _readings(frozen, rows, controls_ok)
    record = {"label": LABEL, "card": CARD, "status": status, "failure": failure,
              "stage1": dict(stage1), "compile": compiled, "compile_note": None
              if compiled is not None else "unavailable: the run stopped before compiling",
              "incumbent_identity": identity, "incumbent_identity_note": None
              if identity is not None else "unavailable: the run stopped before the model load",
              "calls": {a: f"{sum(c[1] == a for c in done)}/{sum(c[1] == a for c in want)}"
                        for a in ("t1j", "incumbent")},
              "rows": rows, "controls_ok": controls_ok, "episodes": episodes,
              "episodes_note": None if episodes is not None else
              f"not interpreted: the run is {status}",
              "elapsed_s": deadline.elapsed(),
              "provenance": {e: {"members": ep["members"], "frequency": ep["frequency"]}
                             for e, ep in frozen["episodes"].items()},
              "note": "what was chosen, never why; no strength estimate, no rescoring"}
    with open(d / "record.json", "x", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
    if interrupt is not None:
        raise interrupt
    return record


# ─────────────────────────────── production wiring ───────────────────────────────

def _gate() -> None:
    if not AUTH.H5_CAP_COMPLETION_AUTHORIZED:
        raise CapUnauthorized("the H5 cap-completion diagnostic is UNAUTHORIZED. Nothing "
                              "was opened, read, built, compiled, queried or written.")


def _evidence_dir(out_dir: str) -> pathlib.Path:
    p = pathlib.Path(out_dir).resolve()
    if "EXPLORATORY" not in p.name or not R4._inside(p, pathlib.Path(R4.EVIDENCE_ROOT)):
        raise CapRefused(f"{out_dir}: the evidence directory must be under the evidence "
                         f"tree and named EXPLORATORY")
    return p


def _frozen_from_records() -> Dict[str, Any]:
    try:
        games, _ = WR.load_and_verify()
    except WR.ReviewRefused as e:
        raise CapRefused(f"the committed H5 records: {e}") from None
    return build_frozen_set(games)


def _committed_frozen(out: pathlib.Path) -> Tuple[Dict[str, Any], str]:
    path = out / FROZEN_FILE
    try:
        F.check_durable([str(path)], _fixture=False)
    except F.H4AnalysisRefused as e:
        raise CapRefused(f"{FROZEN_FILE} must be committed before any JVM: {e}") from None
    raw = path.read_bytes()
    frozen = _frozen_from_records()
    if raw != frozen_bytes(frozen):
        raise CapRefused(f"{FROZEN_FILE} is not the set re-derived from the committed records")
    return frozen, hashlib.sha256(raw).hexdigest()


def _bound_toolchain() -> Dict[str, Any]:
    raw = (pathlib.Path(R4.REPO_ROOT) / MANIFEST).read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise CapRefused("the H5 study manifest does not hash to its pin")
    F.check_durable([str(pathlib.Path(R4.REPO_ROOT) / MANIFEST)], _fixture=False)
    return json.loads(raw)["segments"][0]["t1j_runtime"]["toolchain"]


def _check_classes(classes: str) -> None:
    """A PURE refusal (§7 A2.1): the class directory is outside the repository and free."""
    if R4._inside(pathlib.Path(classes).resolve(), pathlib.Path(R4.REPO_ROOT)):
        raise CapRefused(f"the class directory must be OUTSIDE the repository: {classes}")
    if os.path.lexists(classes):
        raise CapRefused(f"{classes} is occupied: compiles are create-only")


def _compile(classes: str, deadline):
    """H5's BOUND toolchain, compiled fresh outside the repository under `deadline`;
    content must equal the H5 study manifest's, or nothing runs."""
    from . import d1_probe as D1
    paths = R4._verified_t1j_paths(classes)
    content, _local = R4.split_toolchain(D1._default_compile(deadline, paths=paths))
    if content != _bound_toolchain():
        raise CapRefused("the compiled toolchain is not H5's bound toolchain")
    return paths, content


def _verified_paths(classes: str):
    """Stage 1's compile, under its own compile deadline and supervisor."""
    from . import d1_probe as D1
    from . import h4_production_qualification as Q4
    deadline = D1.Deadline(limit_s=Q4.COMPILE_DEADLINE_S)
    deadline.start()
    with D1._supervisor(deadline):
        return _compile(classes, deadline)


def _t1j_replay(paths) -> Callable:
    def replay(moves):
        plies, rc, out = A.replay(moves, ply_cap=R4.PLY_CAP, java=paths.java, jar=paths.jar,
                                  classes=paths.classes, timeout_s=R4.REPLAY_TIMEOUT_S)
        error = None
        if rc == 0:
            try:
                INT.check_postcond(out, expected_refl=INT.REPLAY_REFL_N,
                                   where="cap-completion replay", phase=INT.PHASE_BIND)
            except INT.AbortError as e:
                error = f"postcond: {e}"
        return {"plies": plies, "rc": rc, "out": out, "error": error}
    return replay


def _t1j_choose(paths) -> Callable:
    runtime = INT.T1jRuntime(java=paths.java, jar=paths.jar, classes=paths.classes,
                             ply_cap=R4.PLY_CAP, timeout_s=R4.REPLAY_TIMEOUT_S,
                             h4_acceptance=True)

    def choose(moves, tag, ev):
        """The unchanged §4B path; only `_query` is the recording boundary (§8 A3.3).
        Any exception propagates UNCONVERTED -- the stage records it as a STOP."""
        ctx = INT.IntegrationContext()
        ctx.bind_runtime(runtime)
        ctx.reset(tag, [_t(m) for m in moves])
        ev["observations"] = ctx.processes[tag]  # THE bucket, filled as the agent goes
        state = state_of(moves)
        agent = INT.T1jAgent(runtime=runtime, ctx=ctx, depth=R4.DEPTH, colour=state.to_move,
                             timeout_s=R4.QUERY_TIMEOUT_S, _query=recording_query(ev))
        return {"move": list(_t(agent(state)))}
    return choose


def _incumbent_choose() -> Tuple[Callable, Dict[str, Any]]:
    """The model load (§8 A3.2: after the claim, inside the deadline)."""
    config = R4.frozen_argmax_config()
    identity = R4.frozen_incumbent_identity(config, design=H5.STUDY_DESIGN)
    R4.check_incumbent_identity(identity, config, design=H5.STUDY_DESIGN)
    bound = json.loads((pathlib.Path(R4.REPO_ROOT) / MANIFEST).read_text())
    if identity != bound["segments"][0]["incumbent_identity"]:
        raise CapRefused("the incumbent identity is not H5's bound identity")
    evaluator = R4.load_production_evaluator(identity)
    build = R4.production_incumbent_build(config)
    tasks = {"red": R4.make_schedule([("h5-cap-completion-red", SEED, SEED - 2)],
                                     mode="fixture", reference=identity)[0],
             "black": R4.make_schedule([("h5-cap-completion-black", SEED - 2, SEED)],
                                       mode="fixture", reference=identity)[1]}

    def choose(moves, tag, ev):
        state = state_of(moves)
        task = tasks[state.to_move]
        t0 = time.monotonic()
        move = build(task, evaluator)(state)
        return {"move": None if move is None else list(_t(move)),
                "seconds": time.monotonic() - t0, "seed": task["seed"]}
    return choose, identity


# ─────────────────────────────── the public entries ───────────────────────────────

def freeze(out_dir: str) -> str:
    """PUBLIC ENTRY. Writes 00_frozen_set.json -- to be COMMITTED before any JVM."""
    _gate()
    out = _evidence_dir(out_dir)
    if os.path.lexists(out):
        raise CapRefused(f"{out_dir} is occupied: the evidence directory is create-only")
    frozen = _frozen_from_records()
    os.makedirs(out)                             # create-only
    with open(out / FROZEN_FILE, "xb") as fh:
        fh.write(frozen_bytes(frozen))
    return str(out / FROZEN_FILE)


def run_stage1(out_dir: str, classes: str) -> Dict[str, Any]:
    """PUBLIC ENTRY. Both rules engines, no agent. No seam is accepted. Pure refusals
    first; the claim before the compile; a record on every exit (§7 A2)."""
    _gate()
    out = _evidence_dir(out_dir)
    frozen, frozen_sha = _committed_frozen(out)
    _check_classes(classes)
    return _stage1(frozen, frozen_sha, lambda: _verified_paths(classes), _t1j_replay, out)


def run_stage2(out_dir: str, classes: str) -> Dict[str, Any]:
    """PUBLIC ENTRY. The unchanged agents. §8 A3.1: PURE refusals first -- the Stage-1
    permission among them, against a binding re-derived WITHOUT compiling -- then the
    claim, then the compile and the model load inside one deadline. No seam is accepted."""
    _gate()
    out = _evidence_dir(out_dir)
    frozen, frozen_sha = _committed_frozen(out)
    if os.path.lexists(out / STAGE2_DIR):
        raise CapRefused(f"{STAGE2_DIR} is occupied: Stage 2 is one-shot")
    _check_classes(classes)
    binding = {"frozen_set_sha256": frozen_sha, "position_digests": position_digests(frozen),
               "toolchain": _bound_toolchain()}
    permission = stage2_permission(out / STAGE1_DIR, frozen, binding, _fixture=False)
    stage1 = {"binding": binding, "record_sha256": hashlib.sha256(
        (out / STAGE1_DIR / "record.json").read_bytes()).hexdigest()}
    return _stage2(frozen, permission, stage1, lambda dl: _compile(classes, dl),
                   _incumbent_choose, _t1j_choose, out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """PUBLIC ENTRY. The gate is read before the arguments are acted on."""
    import argparse
    ap = argparse.ArgumentParser(prog="h5_cap_completion",
                                 description="H5 cap-completion diagnostic -- EXPLORATORY. "
                                             "NOT AUTHORIZED.")
    ap.add_argument("step", choices=("freeze", "stage1", "stage2"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--classes")
    a = ap.parse_args(argv)
    if not AUTH.H5_CAP_COMPLETION_AUTHORIZED:
        print("the H5 cap-completion diagnostic is UNAUTHORIZED. Nothing was opened, read, "
              "built, compiled, queried or written.", file=sys.stderr)
        return EXIT_UNAUTHORIZED
    try:
        if a.step == "freeze":
            print(freeze(a.out_dir))
        elif not a.classes:
            raise CapRefused("--classes is required for a stage")
        elif a.step == "stage1":
            print(json.dumps({"all_agree": run_stage1(a.out_dir, a.classes)["all_agree"]}))
        else:
            rec = run_stage2(a.out_dir, a.classes)
            print(json.dumps({k: rec[k] for k in ("status", "calls", "controls_ok")}))
    except (CapRefused, CapUnauthorized) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    return EXIT_OK


if __name__ == "__main__":                                                        # pragma: no cover
    raise SystemExit(main())
