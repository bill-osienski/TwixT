"""EXTERNAL-ENGINE LEARNING, STAGE 1 -- the teacher/data smoke. PROTOTYPE; PREPARATION ONLY.

Card: `docs/superpowers/2026-10-08-external-engine-learning-viability.md`, design baseline ff1422fb.
Nothing here runs on import. The run entry refuses unless the 28th gate,
`external_learning_authorization.EXTERNAL_LEARNING_STAGE1_AUTHORIZED`, is open -- read at call time,
in the parent AND in every child process that touches an engine.

WHAT STAGE 1 DOES, in the card's order (each failure is a STOP; nothing is dropped, replaced, resampled
or retried):
  1. the committed plan's 18 seed slots must be the frozen layout, ACCOUNTED and unspent;
  2. the pinned twixtbot checkout, model bytes and module origins (G3's verifier, reused), its license/
     notice files hashed, TensorFlow restricted to the CPU;
  3. twelve 24x24 reconstruction fixtures in lockstep with BOTH engines -- edges, both colours' links
     and wins, opponent AND own blocked crossings, both knight slopes, shared endpoints, the external
     cap boundary -- plus deliberate corruption that must be detected, and the rule flag on copies;
  4. three nonterminal fixtures queried in each of two fresh processes: six queries, identical moves;
  5. twelve games, six fresh four-ply openings x both colour arms, the incumbent at unchanged research
     settings; state equivalence after EVERY opening ply and every game ply; every game archived;
  6. teacher rows exported create-only, rebuilt in both engines, reloaded in a separate process;
     family splits, exact-position leakage and the 96 / 32 row floors enforced;
  7. raw incumbent logits on TRAINING rows only, persisted as a hashed create-only artifact; f computed
     from its reloaded contents. f = 0 stops before training.
Stage 2 and 3 helpers below (the scalar-temperature fit, the held-out comparison, the artifact binding)
exist only so the card's checks can be tested now; nothing here trains or plays an evaluation game.

    python -m scripts.GPU.alphazero.external_learning run --clone-root <pinned twixtbot-ui checkout>
"""
from __future__ import annotations

import copy as _copy
import hashlib
import json
import math
import os
import pathlib
import random
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import external_learning_authorization as GATE
from . import twixtbot_adapter as A

ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE = "scripts.GPU.alphazero.external_learning"
CARD = "docs/superpowers/2026-10-08-external-engine-learning-viability.md"
BASELINE = "ff1422fb"

BOARD, PLY_CAP, TRIALS, OPENING_PLIES = 24, 280, 100, 4
FAMILIES = 6
TRAIN_FAMILIES, HELDOUT_FAMILIES = (0, 1, 2, 3), (4, 5)
ARMS = (("teacher_red", "red"), ("teacher_black", "black"))
N_SLOTS = FAMILIES + FAMILIES * len(ARMS)            # 18: six opening draws, twelve game seeds
MIN_TRAIN_ROWS, MIN_HELDOUT_ROWS = 96, 32
MAX_MOVES_CAP = BOARD * BOARD                        # 576: never truncate a legal move
REFERENCE = "calib020_0001"
REFERENCE_SHA1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"
TARGET_TYPE = "teacher_chosen_move"
ROW_SCHEMA = "external-learning-teacher-row/1"
PLAN_KIND = "external-learning-stage1-plan/1"
LOGITS_KIND = "external-learning-training-logits/1"
RECORD_KIND = "external-learning-stage1-record/1"
PLAN = "docs/superpowers/evidence/external-engine-learning-plans/stage1.json"
RUN_DIR = "runs/external-engine-learning/stage1"
EVIDENCE_DIR = "docs/superpowers/evidence/external-engine-learning-stage1"
CLEANUP_FAILED = RUN_DIR + ".cleanup-failed.json"    # parent-owned, OUTSIDE the run directory
EXECUTION_S = 3600                                   # the card's 60-minute execution ceiling
QUERY_S = 600
GROUP_GRACE_S = 5.0                                  # SIGTERM -> SIGKILL for the run's process group
BETA_BOUNDS, BETA_ITERATIONS, BETA_TOLERANCE = (0.0, 100.0), 64, 1e-6

RULES = {"board": BOARD, "active_size": BOARD, "automatic_links": True, "link_removal": False,
         "own_crossings": "forbidden", "opponent_crossings": "forbidden", "swap": False,
         "allow_scl": False, "ply_cap": PLY_CAP}

EXIT_PASS, EXIT_STOP_F_ZERO, EXIT_STOP, EXIT_REFUSED, EXIT_TIMEOUT = 0, 1, 2, 3, 4
#: The process group's end could NOT BE VERIFIED -- not reaped, or the check itself failed (e.g. EPERM from
#: killpg). A descendant may still be writing, so nothing is finalized. Its own status, shared with no
#: verdict, and it overrides every other answer (stages 2 and 3 use it too).
EXIT_CLEANUP_UNVERIFIED = 5


class Stage1Stop(Exception):
    """A card stop condition. Names the step; never retried."""

    def __init__(self, step: str, detail: str):
        super().__init__(f"[{step}] {detail}")
        self.step, self.detail = step, detail


class Refused(Exception):
    """The run never started: the gate is closed, the destination exists or the tree is dirty."""


def _require_gate() -> None:
    if GATE.EXTERNAL_LEARNING_STAGE1_AUTHORIZED is not True:
        raise Refused("EXTERNAL_LEARNING_STAGE1_AUTHORIZED is closed: stage 1 is not authorized")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _jsha(obj: Any) -> str:
    return _sha(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def _write_new(path: pathlib.Path, data: bytes) -> str:
    """Create-only, flushed and fsynced. Returns the sha256 of what was written."""
    with open(path, "xb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    return _sha(data)


def _write_json(path: pathlib.Path, obj: Any) -> str:
    return _write_new(path, (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode())


# ─────────────────────────────── the plan: 18 slots ───────────────────────────────

def split_of(family: int) -> str:
    """Frozen before generation: the first four families train, the last two are held out."""
    if family in TRAIN_FAMILIES:
        return "train"
    if family in HELDOUT_FAMILIES:
        return "heldout"
    raise Stage1Stop("split", f"family {family!r} is not one of the {FAMILIES}")


def slot_layout(block: Sequence[int]) -> Dict[str, list]:
    """The frozen slot rule: family j opens with BASE + j; its game in arm c (teacher red 0, black 1)
    is seeded BASE + 6 + 2j + c. Both arms of a family replay the same opening."""
    lo, hi = (int(x) for x in block)
    if hi - lo != N_SLOTS:
        raise Stage1Stop("plan", f"a stage-1 block is {N_SLOTS} slots, got [{lo}, {hi})")
    openings = [{"family": j, "split": split_of(j), "opening_seed": lo + j} for j in range(FAMILIES)]
    games = [{"game_id": f"f{j}-{arm}", "family": j, "split": split_of(j), "arm": arm, "teacher_colour": colour,
              "seed": lo + FAMILIES + 2 * j + c}
             for j in range(FAMILIES) for c, (arm, colour) in enumerate(ARMS)]
    return {"openings": openings, "games": games}


def build_plan(block: Sequence[int], survey_record_sha256: str) -> Dict[str, Any]:
    """Serialized as base + slot count, never the exclusive endpoint: that endpoint is outside the
    block, so the seed scanner would count it and the plan would lose its exemption."""
    return {"kind": PLAN_KIND, "card": CARD, "baseline": BASELINE, "base": int(block[0]), "n_slots": N_SLOTS,
            "survey_record_sha256": survey_record_sha256, **slot_layout(block)}


def plan_block(plan: Mapping[str, Any]) -> Tuple[int, int]:
    """The half-open block, derived from base + slot count."""
    return plan["base"], plan["base"] + plan["n_slots"]


def plan_seeds(plan: Mapping[str, Any]) -> List[int]:
    return [o["opening_seed"] for o in plan["openings"]] + [g["seed"] for g in plan["games"]]


def check_plan(plan: Any) -> Dict[str, Any]:
    """The plan must BE the frozen layout of its block, every seed an int, unique, ACCOUNTED in the
    shared registry, neither exposed nor retired, and not one G3 consumed."""
    from . import e4_screen_reference as REG
    from .twixtbot_g3_schedule import CONSUMED_SEEDS

    if not isinstance(plan, dict) or plan.get("kind") != PLAN_KIND:
        raise Stage1Stop("plan", "not a stage-1 plan")
    if type(plan.get("base")) is not int or plan.get("n_slots") != N_SLOTS or type(plan["n_slots"]) is not int:
        raise Stage1Stop("plan", f"the plan's base is not an int or its slot count is not {N_SLOTS}")
    block = plan_block(plan)
    sha = plan.get("survey_record_sha256")
    if not (isinstance(sha, str) and len(sha) == 64):
        raise Stage1Stop("plan", "the plan names no survey record")
    if plan != build_plan(block, sha):
        raise Stage1Stop("plan", "the plan is not the frozen slot layout of its block")
    seeds = plan_seeds(plan)
    if sorted(seeds) != list(range(*block)):
        raise Stage1Stop("plan", "the slots are not exactly the block")
    for s in seeds:
        if not REG.seed_is_accounted(s):
            raise Stage1Stop("plan", f"seed {s} is not ACCOUNTED in the shared registry")
        if REG.seed_is_unavailable(s) or s in CONSUMED_SEEDS:
            raise Stage1Stop("plan", f"seed {s} is spent (exposed, retired or consumed)")
    return plan


# ─────────────────────────────── openings and digests ───────────────────────────────

def generate_opening(seed: int) -> List[Tuple[int, int]]:
    """Four plies, engine-free: `random.Random(seed).choice` over OUR legal moves at each ply."""
    from .game.twixt_state import TwixtState

    rng = random.Random(int(seed))
    state, moves = TwixtState(max_plies_limit=PLY_CAP), []
    for _ in range(OPENING_PLIES):
        move = rng.choice(state.legal_moves())
        moves.append(move)
        state = state.apply_move(move)
    if state.winner() is not None:
        raise Stage1Stop("opening", f"seed {seed}: a four-ply opening cannot be won")
    return moves


def state_digest(state) -> str:
    """Normalized peg/bridge state: side to move, pegs and bridges -- not the path that led there."""
    return _jsha({"to_move": state.to_move,
                  "pegs": sorted([r, c, p] for (r, c), p in state.pegs.items()),
                  "bridges": sorted([list(a), list(b)] for a, b in state.bridges)})


def legal_digest(legal: Sequence[Sequence[int]]) -> str:
    return _jsha([[int(r), int(c)] for r, c in legal])


def rules_config() -> Dict[str, Any]:
    from .twixtbot_g3_schedule import ANCHOR_SETTINGS, OUR_SETTINGS

    return {"rules": RULES, "teacher": {**ANCHOR_SETTINGS, "trials": TRIALS},
            "incumbent": {"reference": REFERENCE, "sha1": REFERENCE_SHA1, **OUR_SETTINGS}}


# ─────────────────────────────── the twelve fixtures ───────────────────────────────

def _alternate(red: Sequence, black: Sequence) -> List[Tuple[int, int]]:
    out = []
    for i, m in enumerate(red):
        out.append(tuple(m))
        if i < len(black):
            out.append(tuple(black[i]))
    return out


_FAR_BLACK = [(r, 20) for r in range(2, 10)]          # vertical neighbours: never a knight apart
_FAR_RED = [(20, c) for c in range(2, 20)]
_RED_LADDER = [(2 * i, 10 if i % 2 == 0 else 11) for i in range(12)] + [(23, 13)]
_BLACK_LADDER = [(10 if i % 2 == 0 else 11, 2 * i) for i in range(12)] + [(13, 23)]
_RED_LINKS = _alternate([(10, 10), (12, 11), (11, 13)], _FAR_BLACK)

#: Each fixture: moves from the empty board, the external cap, the EXACT bridge sets and winner both
#: engines must hold at the end, and (cap fixtures) the capped terminality both must report.
FIXTURES: Tuple[Dict[str, Any], ...] = (
    {"name": "F01_edges", "moves": [(0, 5), (5, 0), (23, 6), (6, 23), (1, 6), (7, 1)], "cap": None,
     "red": [], "black": [((5, 0), (7, 1))], "winner": None},
    {"name": "F02_red_links", "moves": _RED_LINKS, "cap": None,
     "red": [((10, 10), (12, 11)), ((11, 13), (12, 11))], "black": [], "winner": None},
    {"name": "F03_black_links", "moves": _alternate(_FAR_RED[:3], [(10, 10), (11, 12), (13, 11)]), "cap": None,
     "red": [], "black": [((10, 10), (11, 12)), ((11, 12), (13, 11))], "winner": None},
    {"name": "F04_red_win", "moves": _alternate(_RED_LADDER, [(r, 2) for r in range(1, 13)]), "cap": None,
     "red": [tuple(sorted(p)) for p in zip(_RED_LADDER, _RED_LADDER[1:])], "black": [], "winner": "red"},
    {"name": "F05_black_win", "moves": _alternate([(2, c) for c in range(1, 14)], _BLACK_LADDER), "cap": None,
     "red": [], "black": [tuple(sorted(p)) for p in zip(_BLACK_LADDER, _BLACK_LADDER[1:])], "winner": "black"},
    {"name": "F06_opponent_cross", "moves": [(5, 5), (5, 6), (7, 6), (7, 5)], "cap": None,
     "red": [((5, 5), (7, 6))], "black": [], "winner": None},
    {"name": "F07_own_cross_red", "moves": _alternate([(5, 5), (7, 6), (5, 6), (7, 5)], _FAR_BLACK), "cap": None,
     "red": [((5, 5), (7, 6))], "black": [], "winner": None},
    {"name": "F08_own_cross_black", "moves": _alternate(_FAR_RED[:4], [(5, 5), (6, 7), (6, 5), (5, 7)]),
     "cap": None, "red": [], "black": [((5, 5), (6, 7))], "winner": None},
    {"name": "F09_both_slopes", "moves": _alternate([(12, 12), (14, 13), (13, 14), (10, 13), (11, 14)], _FAR_BLACK),
     "cap": None, "red": [((12, 12), (14, 13)), ((12, 12), (13, 14)), ((10, 13), (12, 12)), ((11, 14), (12, 12))],
     "black": [], "winner": None},
    {"name": "F10_shared_endpoint", "moves": _alternate([(5, 5), (7, 6), (5, 7)], _FAR_BLACK), "cap": None,
     "red": [((5, 5), (7, 6)), ((5, 7), (7, 6))], "black": [], "winner": None},
    {"name": "F11_cap_reached", "moves": _RED_LINKS, "cap": len(_RED_LINKS), "terminal": True,
     "red": [((10, 10), (12, 11)), ((11, 13), (12, 11))], "black": [], "winner": None},
    {"name": "F12_cap_one_below", "moves": _RED_LINKS, "cap": len(_RED_LINKS) + 1, "terminal": False,
     "red": [((10, 10), (12, 11)), ((11, 13), (12, 11))], "black": [], "winner": None},
)
QUERY_FIXTURES = ("F02_red_links", "F06_opponent_cross", "F09_both_slopes")


def _bind(our, tb, twixt, *, ply_cap, label: str, step: str) -> None:
    d = A.state_divergences(our, tb, twixt, ply_cap=ply_cap)
    if d:
        raise Stage1Stop(step, f"{label}: {d}")


def check_fixture(fx: Mapping[str, Any], twixt, Point, TwixtState) -> str:
    """Lockstep replay; the two states compared before the first move and after every ply; then the
    exact bridge sets, the winner and (cap fixtures) the capped terminality, in BOTH engines."""
    name, step = fx["name"], "fixtures"
    our, tb = TwixtState(), twixt.Game(allow_scl=False)
    _bind(our, tb, twixt, ply_cap=fx["cap"], label=f"{name} before any move", step=step)
    for i, (r, c) in enumerate(fx["moves"]):
        if (r, c) not in our.legal_moves():
            raise Stage1Stop(step, f"{name} ply {i}: {(r, c)} illegal in our engine")
        our = our.apply_move((r, c))
        tb.play(Point(*A.rc_to_xy(r, c)))
        _bind(our, tb, twixt, ply_cap=fx["cap"], label=f"{name} after ply {i}", step=step)
    ours = {A.RED: set(), A.BLACK: set()}
    for p1, p2 in our.bridges:
        ours[our.pegs[p1]].add(tuple(sorted((p1, p2))))
    theirs = A.tb_bridges(tb, twixt)
    for colour in (A.RED, A.BLACK):
        want = {tuple(sorted(b)) for b in fx[colour]}
        if ours[colour] != want or theirs[colour] != want:
            raise Stage1Stop(step, f"{name} {colour} bridges: want {sorted(want)}, ours {sorted(ours[colour])}, "
                                   f"theirs {sorted(theirs[colour])}")
    if our.winner() != fx["winner"] or A.tb_winner(tb) != fx["winner"]:
        raise Stage1Stop(step, f"{name} winner: want {fx['winner']}, ours {our.winner()}, theirs {A.tb_winner(tb)}")
    if "terminal" in fx and A.terminal_with_cap(our, tb, fx["cap"]) != (fx["terminal"], fx["terminal"]):
        raise Stage1Stop(step, f"{name}: capped terminality is not {fx['terminal']} in both engines")
    return f"{name}: {len(fx['moves'])} plies equal at every ply"


def corruption_controls(twixt, Point, TwixtState) -> List[str]:
    """Deliberate peg / bridge / side / ply corruption of OUR state must each be detected by the
    comparison, under the field it corrupts."""
    our, tb = A.build_pair(FIXTURES[1]["moves"], twixt, Point, TwixtState)
    _bind(our, tb, twixt, ply_cap=PLY_CAP, label="corruption premise", step="corruption")
    cases = []
    bad = our.copy()
    bad.pegs[(3, 3)] = "red"
    cases.append(("peg", bad, "pegs["))
    bad = our.copy()
    bad.bridges.discard(sorted(bad.bridges)[0])
    cases.append(("bridge", bad, "bridges["))
    bad = our.copy()
    bad.to_move = "red" if bad.to_move == "black" else "black"
    cases.append(("side", bad, "side to move"))
    bad = our.copy()
    bad.ply += 1
    cases.append(("ply", bad, "ply:"))
    out = []
    for kind, state, field in cases:
        found = A.state_divergences(state, tb, twixt, ply_cap=PLY_CAP)
        if not any(field in d for d in found):
            raise Stage1Stop("corruption", f"{kind} corruption NOT detected ({found})")
        out.append(kind)
    return out


def rule_flag_copies(twixt, Point) -> List[str]:
    """`allow_scl` must stay False on every copy a Game can make of itself: copy, deepcopy and any
    public copy/clone method it exposes. Which of these the engine's search uses is read from the
    pinned source at setup; this proves none of them drops the flag."""
    g = twixt.Game(allow_scl=False)
    for r, c in FIXTURES[5]["moves"]:
        g.play(Point(*A.rc_to_xy(r, c)))
    if getattr(g, "allow_scl", None) is not False:
        raise Stage1Stop("rule_flag", "a Game built with allow_scl=False does not report it")
    import inspect

    makers = {"copy.copy": lambda: _copy.copy(g), "copy.deepcopy": lambda: _copy.deepcopy(g)}
    for name in dir(g):                    # filter by NAME first: other attributes are never touched
        if name.startswith("_") or not ("copy" in name.lower() or "clone" in name.lower()):
            continue
        attr = getattr(g, name)
        if not callable(attr):
            continue
        try:
            required = [q for q in inspect.signature(attr).parameters.values()
                        if q.default is q.empty and q.kind in (q.POSITIONAL_ONLY, q.POSITIONAL_OR_KEYWORD)]
        except (TypeError, ValueError):
            required = [None]
        if required:
            raise Stage1Stop("rule_flag", f"Game.{name} needs arguments; read its use in the pinned source first")
        makers[f"Game.{name}"] = attr
    for label, make in makers.items():
        dup = make()
        if getattr(dup, "allow_scl", None) is not False:
            raise Stage1Stop("rule_flag", f"{label} does not preserve allow_scl=False")
    return sorted(makers)


# ─────────────────────────────── the teacher ───────────────────────────────

def teacher_kwargs(ct) -> Dict[str, Any]:
    from . import twixtbot_g3_harness as H
    from . import twixtbot_g3_runner as R

    kw = H.anchor_player_kwargs(TRIALS, ct)
    R.assert_anchor_settings(kw, {"trials": TRIALS}, ct)
    return kw


def query_fixtures(rt: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Three nonterminal fixtures, each through a FRESH Game and a FRESH Player."""
    from . import twixtbot_g3_harness as H

    out = []
    by_name = {f["name"]: f for f in FIXTURES}
    for name in QUERY_FIXTURES:
        moves = by_name[name]["moves"]
        our, tb = A.build_pair(moves, rt["twixt"], rt["Point"], rt["TwixtState"])
        _bind(our, tb, rt["twixt"], ply_cap=PLY_CAP, label=f"query {name}", step="queries")
        if our.winner() is not None or not our.legal_moves():
            raise Stage1Stop("queries", f"{name} is terminal")
        player = rt["nnmplayer"].Player(**teacher_kwargs(rt["ct"]))
        try:
            rc, _rec = H.anchor_move(player, tb, our, our.ply, TRIALS)
        except H.HarnessAbort as e:
            raise Stage1Stop("queries", f"{name}: {e}") from None
        out.append({"fixture": name, "input_sha256": _jsha([list(m) for m in moves]), "move": list(rc)})
    return out


def compare_queries(first: Any, second: Any) -> List[Dict[str, Any]]:
    if not (isinstance(first, list) and first == second and [q.get("fixture") for q in first] == list(QUERY_FIXTURES)):
        raise Stage1Stop("queries", f"the two processes disagree or are incomplete: {first} vs {second}")
    return first


# ─────────────────────────────── the games ───────────────────────────────

def play_teacher_game(*, game: Mapping[str, Any], opening: Sequence[Sequence[int]], rt: Mapping[str, Any],
                      player, reference_agent, log=None) -> Dict[str, Any]:
    """One game: the opening replayed with BOTH engines compared after every opening ply, then teacher
    vs incumbent with the comparison after every ply. Each teacher decision is captured as a row.
    G3's harness supplies the move capture; aborts are recorded, never retried. `log` receives every
    move as it is played, so a killed process still leaves the partial game."""
    from . import twixtbot_g3_harness as H

    twixt, Point, TwixtState = rt["twixt"], rt["Point"], rt["TwixtState"]
    teacher = game["teacher_colour"]
    our, tb = TwixtState(max_plies_limit=PLY_CAP), twixt.Game(allow_scl=False)
    record = {"game_id": game["game_id"], "family": game["family"], "split": game["split"], "arm": game["arm"],
              "teacher_colour": teacher, "seed": game["seed"], "opening": [list(m) for m in opening],
              "ply_cap": PLY_CAP, "aborted": False, "abort_reason": None, "abort_detail": None}
    moves: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    def played(mrec):
        moves.append(mrec)
        if log is not None:
            log({"event": "move", **mrec})

    def bind(ply, label):
        d = A.state_divergences(our, tb, twixt, ply_cap=PLY_CAP)
        if d:
            raise H.HarnessAbort("state_divergence", f"{label}: {d}", ply)

    try:
        if getattr(reference_agent, "seed", None) != game["seed"]:
            raise H.HarnessAbort("malformed_output", "the incumbent agent is not bound to this game's seed", 0)
        bind(0, "before the opening")
        for i, (r, c) in enumerate(opening):
            if (r, c) not in our.legal_moves():
                raise H.HarnessAbort("illegal_move", f"opening move {(r, c)} illegal", i)
            our = our.apply_move((r, c))
            tb.play(Point(*A.rc_to_xy(r, c)))
            played({"ply": i, "mover": "opening", "move": [r, c]})
            bind(i + 1, f"after opening ply {i}")
        while not our.is_terminal():
            ply = our.ply
            if our.to_move == teacher:
                legal = our.legal_moves()
                before = {"ply": ply, "side": our.to_move, "prefix": [m["move"] for m in moves],
                          "legal_moves": [list(m) for m in legal], "state_digest": state_digest(our)}
                rc, mrec = H.anchor_move(player, tb, our, ply, TRIALS)
                mrec["mover"] = "teacher"
                rows.append({**before, "legal_digest": legal_digest(legal), "chosen": list(rc),
                             "chosen_index": legal.index(rc)})
            else:
                rc, mrec = H.reference_move(reference_agent, our, ply)
                mrec["mover"] = "incumbent"
            our = our.apply_move(rc)
            tb.play(Point(*A.rc_to_xy(*rc)))
            played(mrec)
            bind(our.ply, f"after ply {ply} ({mrec['mover']} {rc})")
        winner = our.winner()
        record.update(winner=winner, plies=our.ply,
                      result=("draw_ply_cap" if winner is None and our.ply >= PLY_CAP else
                              "draw" if winner is None else "teacher" if winner == teacher else "incumbent"))
    except H.HarnessAbort as e:
        record.update(aborted=True, abort_reason=e.reason, abort_detail=e.detail, plies=our.ply,
                      result=None, winner=None)
    except Exception as e:                                       # noqa: BLE001 -- recorded, then the stage stops
        record.update(aborted=True, abort_reason="engine_exception", abort_detail=f"{type(e).__name__}: {e}",
                      plies=our.ply, result=None, winner=None)
    record["moves"], record["rows"] = moves, rows
    return record


# ─────────────────────────────── export, reconstruction, reload ───────────────────────────────

def rows_from_games(games: Sequence[Mapping[str, Any]], *, config_sha256: str,
                    runtime_sha256: str) -> List[Dict[str, Any]]:
    """Every teacher decision of every (finished) game; opening and incumbent moves never become rows."""
    out = []
    for g in games:
        if g.get("aborted"):
            raise Stage1Stop("export", f"{g['game_id']} aborted; no labels are compiled from a stopped stage")
        for r in g["rows"]:
            out.append({"schema": ROW_SCHEMA, "target_type": TARGET_TYPE,
                        "row_id": f"{g['game_id']}-p{r['ply']:03d}", "game_id": g["game_id"],
                        "family": g["family"], "split": g["split"], "teacher_colour": g["teacher_colour"],
                        "prefix": r["prefix"], "ply": r["ply"], "side": r["side"],
                        "rules": RULES, "config_sha256": config_sha256, "runtime_sha256": runtime_sha256,
                        "legal_moves": r["legal_moves"], "legal_digest": r["legal_digest"],
                        "chosen": r["chosen"], "chosen_index": r["chosen_index"],
                        "state_digest": r["state_digest"], "terminal_reason": g["result"],
                        "outcome": g["winner"]})
    return out


def rows_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode()


def load_rows(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    for i, line in enumerate(pathlib.Path(path).read_text(encoding="utf-8").splitlines()):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise Stage1Stop("reload", f"line {i}: {e}") from None
    return rows


def verify_row(row: Mapping[str, Any]) -> None:
    """Rebuild the row from its prefix in OUR engine: the same state, the same legal moves in the same
    order, a one-hot label on the recorded index, the right side and split."""
    from .game.twixt_state import TwixtState

    rid = row.get("row_id")
    if row.get("schema") != ROW_SCHEMA or row.get("target_type") != TARGET_TYPE:
        raise Stage1Stop("rows", f"{rid}: not a {TARGET_TYPE} row")
    if row.get("split") != split_of(row.get("family")):
        raise Stage1Stop("rows", f"{rid}: split {row.get('split')} is not its family's")
    state = TwixtState(max_plies_limit=PLY_CAP)
    for m in row["prefix"]:
        if tuple(m) not in state.legal_moves():
            raise Stage1Stop("rows", f"{rid}: prefix move {m} is illegal")
        state = state.apply_move(tuple(m))
    legal = [list(m) for m in state.legal_moves()]
    i = row.get("chosen_index")
    if state.ply != row["ply"] or state.to_move != row["side"] or row["side"] != row["teacher_colour"]:
        raise Stage1Stop("rows", f"{rid}: ply or side does not match its prefix")
    if state_digest(state) != row["state_digest"]:
        raise Stage1Stop("rows", f"{rid}: the rebuilt state is not the recorded one")
    if legal != row["legal_moves"] or legal_digest(legal) != row["legal_digest"]:
        raise Stage1Stop("rows", f"{rid}: legal moves or their order differ")
    if type(i) is not int or not 0 <= i < len(legal) or legal[i] != row["chosen"]:
        raise Stage1Stop("rows", f"{rid}: the chosen move is not at its recorded index")


def reconstruct_in_both(rows: Sequence[Mapping[str, Any]], rt: Mapping[str, Any]) -> int:
    for row in rows:
        our, tb = A.build_pair([tuple(m) for m in row["prefix"]], rt["twixt"], rt["Point"], rt["TwixtState"])
        _bind(our, tb, rt["twixt"], ply_cap=PLY_CAP, label=f"rebuilding {row['row_id']}", step="reconstruction")
        if state_digest(our) != row["state_digest"]:
            raise Stage1Stop("reconstruction", f"{row['row_id']}: rebuilt state differs")
    return len(rows)


def reload_summary(path: pathlib.Path) -> Dict[str, Any]:
    """What a SEPARATE process sees: every row re-verified, and the digest of what it re-serializes."""
    rows = load_rows(path)
    for r in rows:
        verify_row(r)
    return {"n": len(rows), "rows_sha256": _sha(rows_bytes(rows)), "file_sha256": _sha(pathlib.Path(path).read_bytes()),
            "splits": {s: sum(r["split"] == s for r in rows) for s in ("train", "heldout")}}


def split_checks(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Families never cross splits; no exact position in both; the row floors."""
    train = [r for r in rows if r["split"] == "train"]
    held = [r for r in rows if r["split"] == "heldout"]
    if len(train) + len(held) != len(rows) or any(r["split"] != split_of(r["family"]) for r in rows):
        raise Stage1Stop("splits", "a row is outside its family's frozen split")
    leak = {r["state_digest"] for r in train} & {r["state_digest"] for r in held}
    if leak:
        raise Stage1Stop("splits", f"{len(leak)} exact positions occur in both splits")
    if len(train) < MIN_TRAIN_ROWS or len(held) < MIN_HELDOUT_ROWS:
        raise Stage1Stop("rows", f"insufficient rows: train {len(train)} (floor {MIN_TRAIN_ROWS}), "
                                 f"held-out {len(held)} (floor {MIN_HELDOUT_ROWS})")
    return {"train": len(train), "heldout": len(held),
            "games": {s: sorted({r["game_id"] for r in rows if r["split"] == s}) for s in ("train", "heldout")}}


# ─────────────────────────────── the in-memory training adapter ───────────────────────────────

def position_record(row: Mapping[str, Any]):
    """A `PositionRecord` whose legacy `visit_counts` holds the ONE-HOT teacher choice in our full legal
    order, tagged `teacher_chosen_move`. In memory only; its outcome is a required zero placeholder."""
    import numpy as np

    from .game.twixt_state import TwixtState
    from .self_play import PositionRecord

    state = TwixtState.from_moves([tuple(m) for m in row["prefix"]])
    legal = [tuple(m) for m in state.legal_moves()]
    if [list(m) for m in legal] != row["legal_moves"]:
        raise Stage1Stop("adapter", f"{row['row_id']}: legal order differs from the row's")
    counts = [0] * len(legal)
    counts[row["chosen_index"]] = 1
    rec = PositionRecord(board_tensor=np.transpose(state.to_tensor(), (1, 2, 0)), to_move=state.to_move,
                         legal_moves=legal, visit_counts=counts, outcome=0.0, active_size=BOARD, ply=row["ply"])
    rec.target_type = TARGET_TYPE
    return rec


def target_batch(rows: Sequence[Mapping[str, Any]]):
    """`make_padded_batch(max_moves_cap=576)`; every target row must carry its whole unit of mass --
    at the default cap of 512 a chosen move beyond slot 511 is silently dropped."""
    import numpy as np

    from .trainer import make_padded_batch

    batch = make_padded_batch([position_record(r) for r in rows], max_moves_cap=MAX_MOVES_CAP)
    mass = np.asarray(batch[4]).sum(axis=1)
    if not np.allclose(mass, 1.0):
        raise Stage1Stop("adapter", "a teacher target lost its mass in batching")
    return batch


def raw_logits(network, rows: Sequence[Mapping[str, Any]], chunk: int = 32) -> List[List[float]]:
    """The incumbent's raw logits over each row's FULL legal order, through the canonicalized training
    forward pass (`forward_padded(active_size=24)`); padding discarded."""
    import mlx.core as mx

    out = []
    for i in range(0, len(rows), chunk):
        part = rows[i:i + chunk]
        boards, mr, mc, mask, _target, _outcome = target_batch(part)
        logits, _value, _ = network.forward_padded(boards, mr, mc, mask, active_size=BOARD)
        mx.eval(logits)
        values = logits.tolist()
        for r, z in zip(part, values):
            out.append([float(x) for x in z[:len(r["legal_moves"])]])
    return out


# ─────────────────────────────── the logit artifact and f ───────────────────────────────

def _bindings(rows: Sequence[Mapping[str, Any]], incumbent_sha1: str) -> Dict[str, Any]:
    return {"incumbent": {"reference": REFERENCE, "sha1": incumbent_sha1},
            "corpus_sha256": _sha(rows_bytes(rows)),
            "rows": [{"row_id": r["row_id"], "legal_digest": r["legal_digest"],
                      "chosen_index": r["chosen_index"]} for r in rows]}


def write_logit_artifact(path: pathlib.Path, rows: Sequence[Mapping[str, Any]], logits: Sequence[Sequence[float]],
                         incumbent_sha1: str) -> str:
    if any(r["split"] != "train" for r in rows):
        raise Stage1Stop("logits", "held-out logits are never cached in stage 1")
    if len(rows) != len(logits) or any(len(z) != len(r["legal_moves"]) for r, z in zip(rows, logits)):
        raise Stage1Stop("logits", "a logit row does not cover its legal moves")
    if not all(math.isfinite(x) for z in logits for x in z):
        raise Stage1Stop("logits", "a nonfinite logit")
    return _write_json(path, {"kind": LOGITS_KIND, **_bindings(rows, incumbent_sha1),
                              "logits": [list(map(float, z)) for z in logits]})


def load_logit_artifact(path: pathlib.Path, *, sha256: str, rows: Sequence[Mapping[str, Any]],
                        incumbent_sha1: str) -> List[List[float]]:
    """Stage 2's only way in: the pinned bytes, bound to these frozen training rows and this incumbent.
    Any mismatch refuses; nothing is recomputed or silently replaced."""
    raw = pathlib.Path(path).read_bytes()
    if _sha(raw) != sha256:
        raise Stage1Stop("logits", "the logit artifact is not the pinned bytes")
    art = json.loads(raw)
    want = {"kind": LOGITS_KIND, **_bindings(rows, incumbent_sha1)}
    if {k: art.get(k) for k in want} != want:
        raise Stage1Stop("logits", "the logit artifact is not bound to these rows and this incumbent")
    logits = art.get("logits")
    if not isinstance(logits, list) or len(logits) != len(rows) or \
            any(len(z) != len(r["legal_moves"]) for z, r in zip(logits, rows)):
        raise Stage1Stop("logits", "the logit artifact does not cover the rows' legal moves")
    return logits


def top1(z: Sequence[float]) -> int:
    """First maximum in the recorded legal order."""
    best = 0
    for i, x in enumerate(z):
        if x > z[best]:
            best = i
    return best


def disagreement(rows: Sequence[Mapping[str, Any]], logits: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """f = the fraction of training rows whose teacher move is not the incumbent's raw top-1; a teacher
    move at another TIED index counts as a disagreement."""
    by_game, by_colour = {}, {}
    n_dis = 0
    for r, z in zip(rows, logits):
        d = top1(z) != r["chosen_index"]
        n_dis += d
        for key, table in ((r["game_id"], by_game), (r["teacher_colour"], by_colour)):
            n, k = table.get(key, (0, 0))
            table[key] = (n + 1, k + d)
    fmt = lambda t: {k: {"rows": n, "disagree": k_, "rate": k_ / n} for k, (n, k_) in sorted(t.items())}
    return {"rows": len(rows), "disagree": n_dis, "match": len(rows) - n_dis,
            "f": n_dis / len(rows) if rows else 0.0, "per_game": fmt(by_game), "per_colour": fmt(by_colour)}


def require_training_opportunity(report: Mapping[str, Any]) -> None:
    """f = 0 stops before training: the corpus shows no chance to move a choice toward the teacher."""
    if not report.get("rows") or report.get("disagree", 0) == 0:
        raise Stage1Stop("f", "f = 0: no training row disagrees with the incumbent's raw top-1")


# ─────────────────────────────── stage-2 control and stage-3 comparison ───────────────────────────────

def _padded(logits: Sequence[Sequence[float]]):
    """Legal logits padded with the network's own finite NEG_INF, and the legal mask."""
    import numpy as np

    from .network import NEG_INF

    width = max(len(z) for z in logits)
    Z = np.full((len(logits), width), NEG_INF, dtype=np.float32)
    mask = np.zeros((len(logits), width), dtype=np.float32)
    for i, z in enumerate(logits):
        Z[i, :len(z)] = z
        mask[i, :len(z)] = 1.0
    return Z, mask


def scaled_probs(Z, mask, beta: float):
    """softmax(beta * logits) over legal moves: SCALE FIRST, then `compute_masked_log_probs` re-masks the
    padding -- at beta = 0, beta * NEG_INF is 0 and would otherwise rank padding with the legal moves."""
    import mlx.core as mx
    import numpy as np

    from .trainer import compute_masked_log_probs

    lp = compute_masked_log_probs(mx.array(np.float32(beta) * Z), mx.array(mask))
    mx.eval(lp)
    return np.exp(np.asarray(lp, dtype=np.float64))


def fit_beta(logits: Sequence[Sequence[float]], chosen: Sequence[int]) -> Dict[str, Any]:
    """One inverse temperature on the cached TRAINING logits, minimizing mean chosen-move CE. The
    objective is convex in beta, so the sign of its derivative brackets the minimum: started at
    beta = 1, bounded to [0, 100], at most 64 bisections to a 1e-6 bracket. An endpoint is accepted only
    when its one-sided derivative establishes the constrained minimum."""
    import numpy as np

    Z, mask = _padded(logits)
    y = np.asarray(chosen)
    zm = np.where(mask > 0, Z, 0.0).astype(np.float64)
    rows = np.arange(len(y))

    def ce(b):
        p = scaled_probs(Z, mask, b)
        return float(-np.mean(np.log(p[rows, y])))

    def g(b):
        p = scaled_probs(Z, mask, b) * mask
        v = float(np.mean((p * zm).sum(axis=1) - zm[rows, y]))
        if not math.isfinite(v):
            raise Stage1Stop("beta", f"nonfinite derivative at beta = {b}")
        return v

    lo_b, hi_b = BETA_BOUNDS
    g_init, iterations = g(1.0), 0
    if g(lo_b) >= 0:
        beta, bracket = lo_b, (lo_b, lo_b)
    else:
        if g(hi_b) <= 0:          # an underflowed zero is not an established minimum either
            raise Stage1Stop("beta", f"CE still decreasing (not increasing) at beta = {hi_b}: the control is "
                                     f"inadequately bounded")
        lo, hi = (lo_b, 1.0) if g_init >= 0 else (1.0, hi_b)
        while hi - lo > BETA_TOLERANCE and iterations < BETA_ITERATIONS:
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
            iterations += 1
        if hi - lo > BETA_TOLERANCE:
            raise Stage1Stop("beta", "bisection did not reach the bracket tolerance")
        beta, bracket = (lo + hi) / 2, (lo, hi)
    value = ce(beta)
    if not math.isfinite(value):
        raise Stage1Stop("beta", "nonfinite CE at the fitted beta")
    return {"beta": beta, "ce": value, "derivative": g(beta), "derivative_at_init": g_init,
            "iterations": iterations, "bracket": list(bracket), "bounds": list(BETA_BOUNDS)}


def game_metrics(rows: Sequence[Mapping[str, Any]], logits: Sequence[Sequence[float]]) -> Dict[str, Dict[str, float]]:
    """Per game: mean chosen-move CE and top-1 agreement (first maximum) over its rows."""
    out: Dict[str, List[Tuple[float, int]]] = {}
    for r, z in zip(rows, logits):
        m = max(z)
        lse = m + math.log(sum(math.exp(x - m) for x in z))
        out.setdefault(r["game_id"], []).append((lse - z[r["chosen_index"]], int(top1(z) == r["chosen_index"])))
    return {gid: {"ce": sum(c for c, _ in v) / len(v), "top1": sum(t for _, t in v) / len(v)}
            for gid, v in sorted(out.items())}


def heldout_passes(candidate: Mapping[str, Mapping[str, float]], incumbent: Mapping[str, Mapping[str, float]],
                   control: Mapping[str, Mapping[str, float]]) -> bool:
    """Candidate game-mean CE strictly below BOTH the incumbent and the scalar control, and candidate
    game-mean top-1 strictly above the incumbent. A pure temperature rescaling keeps every top-1."""
    if not candidate or set(candidate) != set(incumbent) or set(candidate) != set(control):
        raise Stage1Stop("heldout", "the three metric sets are not over the same held-out games")
    mean = lambda t, k: sum(v[k] for v in t.values()) / len(t)
    return (mean(candidate, "ce") < mean(incumbent, "ce") and mean(candidate, "ce") < mean(control, "ce")
            and mean(candidate, "top1") > mean(incumbent, "top1"))


# ─────────────────────────────── the runtime ───────────────────────────────

def license_notices(clone_root: str) -> List[Dict[str, str]]:
    """The checkout's license/notice files, hashed (kept with the evidence). The permission judgment itself
    is made by a person at setup; no notice at all stops the stage."""
    root = pathlib.Path(clone_root)
    found = sorted(p for p in root.iterdir() if p.is_file() and p.name.upper().startswith(("LICENSE", "NOTICE", "COPYING")))
    if not found:
        raise Stage1Stop("permission", "the pinned checkout carries no license or notice file")
    return [{"path": p.name, "sha256": _sha(p.read_bytes())} for p in found]


def tensorflow_cpu_only() -> str:
    """Before any model loads: TensorFlow sees no GPU, so MLX is the only Metal owner."""
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    if tf.config.get_visible_devices("GPU"):
        raise Stage1Stop("runtime", "TensorFlow still sees a GPU")
    return tf.__version__


def load_runtime(clone_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """G3's pinned-runtime verifier and module-origin import, reused unchanged."""
    from . import twixtbot_g3_runner as R

    identity = R.verify_anchor_runtime(clone_root)
    tf_version = tensorflow_cpu_only()
    rt = R._import_anchor_runtime(clone_root, str(ROOT))
    identity = {k: v for k, v in identity.items() if k != "clone_root"}
    return rt, {**identity, "tensorflow": tf_version, "runtime_sha256": _jsha(identity)}


# ─────────────────────────────── the stage, in a child process ───────────────────────────────

def _committed(rel: str) -> bool:
    r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", rel], capture_output=True)
    d = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", rel])
    return r.returncode == 0 and d.returncode == 0


def _subprocess_json(args: List[str], out: pathlib.Path, timeout: int, step: str) -> Any:
    """Its output goes straight to create-only files, so a timeout keeps what it had written."""
    with open(out.with_suffix(".stdout.txt"), "xb") as so, open(out.with_suffix(".stderr.txt"), "xb") as se:
        try:
            proc = subprocess.run([sys.executable, "-m", MODULE, *args, "--out", str(out)], cwd=str(ROOT),
                                  stdout=so, stderr=se, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise Stage1Stop(step, f"subprocess timed out after {timeout} s; partial output in "
                                   f"{out.with_suffix('.stderr.txt').name}") from None
    if proc.returncode != 0 or not out.exists():
        raise Stage1Stop(step, f"subprocess exit {proc.returncode}; see {out.with_suffix('.stderr.txt').name}")
    return json.loads(out.read_text(encoding="utf-8"))


def progress(run_dir: pathlib.Path, event: Mapping[str, Any]) -> None:
    """One JSON line appended and flushed per event: survives a stop, a crash and the ceiling's kill."""
    with open(run_dir / "progress.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": round(time.time(), 3), **event}, sort_keys=True) + "\n")
        fh.flush()


class _Steps(dict):
    """The record's steps: every completed check is also written to progress.jsonl when it is set."""
    def __init__(self, run_dir: pathlib.Path):
        super().__init__()
        self.run_dir = run_dir

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        progress(self.run_dir, {"event": "step", "step": key, "value": value})


def stage1(run_dir: pathlib.Path, clone_root: str, rec: Dict[str, Any]) -> None:
    """The whole stage. Writes into `rec` as it goes, so a stop keeps every completed step."""
    from . import twixtbot_g3_reference as RF

    steps = rec["steps"] = _Steps(run_dir)
    if not _committed(PLAN):
        raise Stage1Stop("plan", f"{PLAN} is absent or not committed")
    plan = check_plan(json.loads((ROOT / PLAN).read_text(encoding="utf-8")))
    steps["plan"] = {"sha256": _sha((ROOT / PLAN).read_bytes()), "base": plan["base"], "n_slots": plan["n_slots"]}
    steps["permission"] = license_notices(clone_root)
    rt, identity = load_runtime(clone_root)
    steps["runtime"] = identity
    config_sha = _jsha(rules_config())
    steps["config_sha256"] = config_sha

    steps["fixtures"] = [check_fixture(fx, rt["twixt"], rt["Point"], rt["TwixtState"]) for fx in FIXTURES]
    steps["corruption_detected"] = corruption_controls(rt["twixt"], rt["Point"], rt["TwixtState"])
    steps["rule_flag_copies"] = rule_flag_copies(rt["twixt"], rt["Point"])

    q = [_subprocess_json(["_query", "--clone-root", clone_root], run_dir / f"queries-{k}.json", QUERY_S, "queries")
         for k in ("a", "b")]
    steps["queries"] = compare_queries(q[0], q[1])

    ev = RF.load_reference_evaluator(REFERENCE, str(ROOT))
    if getattr(ev, "_g3_sha1", None) != REFERENCE_SHA1:
        raise Stage1Stop("incumbent", "the loaded incumbent is not the pinned checkpoint")
    teacher_eval = rt["nneval"].NNEvaluater("model/pb")
    kwargs = teacher_kwargs(rt["ct"])
    openings = {o["family"]: generate_opening(o["opening_seed"]) for o in plan["openings"]}
    steps["openings"] = {str(j): [list(m) for m in v] for j, v in openings.items()}
    (run_dir / "games").mkdir()
    games = []
    for g in plan["games"]:
        agent = RF.build_reference_agent(
            task={"seed": g["seed"], "reference": REFERENCE, "reference_sha1": REFERENCE_SHA1,
                  "anchor_colour": g["teacher_colour"]},
            evaluator=ev, colour="black" if g["teacher_colour"] == "red" else "red")
        player = rt["nnmplayer"].Player(evaluator=teacher_eval, **kwargs)
        t0 = time.monotonic()
        progress(run_dir, {"event": "game_start", **g, "opening": [list(m) for m in openings[g["family"]]]})
        record = play_teacher_game(game=g, opening=openings[g["family"]], rt=rt, player=player,
                                   reference_agent=agent,
                                   log=lambda e, gid=g["game_id"]: progress(run_dir, {"game_id": gid, **e}))
        record["seconds"] = round(time.monotonic() - t0, 3)
        _write_json(run_dir / "games" / f"{g['game_id']}.json", record)
        games.append(record)
        if record["aborted"]:
            raise Stage1Stop("games", f"{g['game_id']} aborted: {record['abort_reason']} -- {record['abort_detail']}")
        RF.between_games_cleanup()
    steps["games"] = [{k: g[k] for k in ("game_id", "plies", "result", "seconds")} | {"rows": len(g["rows"])}
                      for g in games]

    rows = rows_from_games(games, config_sha256=config_sha, runtime_sha256=identity["runtime_sha256"])
    for r in rows:
        verify_row(r)
    steps["reconstructed_in_both"] = reconstruct_in_both(rows, rt)
    rows_path = run_dir / "teacher_rows.jsonl"
    rows_sha = _write_new(rows_path, rows_bytes(rows))
    mine = {"n": len(rows), "rows_sha256": _sha(rows_bytes(rows)), "file_sha256": rows_sha,
            "splits": {s: sum(r["split"] == s for r in rows) for s in ("train", "heldout")}}
    theirs = _subprocess_json(["_reload", "--rows", str(rows_path)], run_dir / "reload.json", QUERY_S, "reload")
    if theirs != mine:
        raise Stage1Stop("reload", f"the separate process read {theirs}, this one wrote {mine}")
    steps["rows"] = {**mine, **split_checks(rows)}

    train = [r for r in rows if r["split"] == "train"]
    art_path = run_dir / "training_logits.json"
    art_sha = write_logit_artifact(art_path, train, raw_logits(ev.network, train), REFERENCE_SHA1)
    logits = load_logit_artifact(art_path, sha256=art_sha, rows=train, incumbent_sha1=REFERENCE_SHA1)
    report = disagreement(train, logits)
    steps["logit_artifact_sha256"] = art_sha
    steps["disagreement"] = report
    require_training_opportunity(report)


def _child(run_dir: pathlib.Path, clone_root: str) -> int:
    """The record boundary covers the conclusion: the verdict is decided inside it and record.json is
    written on every exit."""
    _require_gate()
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "card": CARD, "baseline": BASELINE, "verdict": None, "steps": {}}
    code = EXIT_STOP

    def ceiling(signum, _frame):
        raise Stage1Stop("ceiling", f"signal {signum} from the parent's {EXECUTION_S} s ceiling")
    previous = signal.signal(signal.SIGTERM, ceiling)
    try:
        stage1(run_dir, clone_root, rec)
        rec["verdict"], code = "PASS", EXIT_PASS
    except Stage1Stop as e:
        rec["stop"] = {"step": e.step, "detail": e.detail}
        rec["verdict"], code = ("STOP_F_ZERO", EXIT_STOP_F_ZERO) if e.step == "f" else ("STOP", EXIT_STOP)
    except BaseException as e:                                   # noqa: BLE001 -- recorded below, then re-raised
        rec["stop"] = {"step": "unexpected", "detail": f"{type(e).__name__}: {e}"}
        rec["verdict"] = "STOP"
        _write_json(run_dir / "record.json", rec)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
    _write_json(run_dir / "record.json", rec)
    return code


# ─────────────────────────────── the parent: claim, ceiling, finalization ───────────────────────────────

def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


VERDICT_EXIT = {"PASS": EXIT_PASS, "STOP_F_ZERO": EXIT_STOP_F_ZERO, "STOP": EXIT_STOP}


def _verdict(record: pathlib.Path) -> Optional[str]:
    try:
        return json.loads(record.read_text(encoding="utf-8")).get("verdict")
    except (OSError, ValueError, AttributeError):
        return None


def end_group(proc: subprocess.Popen, grace: float = GROUP_GRACE_S) -> bool:
    """Terminate and reap the process group led by `proc` (the child and every descendant it started):
    SIGTERM, then SIGKILL. True once the direct child is reaped and the group has no member left."""
    def gone() -> bool:
        if proc.poll() is None:
            return False
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            return True
        return False
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if gone():
                return True
            time.sleep(0.05)
    proc.wait()
    return gone()


def run_group(argv: Sequence[str], *, stdout, stderr,
              timeout: float) -> Tuple[str, Optional[int], Optional[bool], Optional[str]]:
    """`argv` in a new session (its own process group) under `timeout`; the group is ended on EVERY exit,
    so no descendant outlives it. Returns (status, exit code, group reaped, cleanup error). A cleanup that
    RAISES is UNVERIFIED: reaped None with the error kept -- never read as "group gone"."""
    proc = subprocess.Popen(list(argv), cwd=str(ROOT), stdout=stdout, stderr=stderr, start_new_session=True)
    status, code = "UNKNOWN", None
    try:
        code = proc.wait(timeout=timeout)
        status = "EXITED"
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
    finally:
        try:
            reaped, error = end_group(proc, GROUP_GRACE_S), None
        except Exception as e:                                   # noqa: BLE001 -- kept as the diagnostic
            reaped, error = None, f"{type(e).__name__}: {e}"
    return status, code, reaped, error


def exit_code_for(record: pathlib.Path, status: str, child_exit: Optional[int], reaped: Optional[bool] = True) -> int:
    """ONE answer: the recorded verdict decides, and the child's exit status must agree with it. A
    timeout, a missing record, a disagreement or a process group not reaped is never a pass. A cleanup
    not verified answers first, with its own status: a descendant may still write, so read nothing."""
    if reaped is not True:
        return EXIT_CLEANUP_UNVERIFIED
    if status == "TIMEOUT":
        return EXIT_TIMEOUT
    verdict = _verdict(record)
    if verdict not in VERDICT_EXIT or child_exit != VERDICT_EXIT[verdict]:
        return EXIT_STOP
    return VERDICT_EXIT[verdict]


def run(clone_root: str) -> int:
    """THE entry point. Paths only; no seams. Refuses before claiming anything unless the gate is open,
    the tree clean and both destinations absent. The child runs under the 60-minute ceiling; the parent
    writes finalized.json LAST and copies the run directory into the evidence directory -- ONLY once the
    process group is reaped. Otherwise a descendant may still be writing: the parent records the failure
    in its own file beside the run directory, leaves the run directory as it is, hashes and copies
    nothing, and declares no finalization."""
    _require_gate()
    if _git("status", "--porcelain"):
        raise Refused("the working tree is dirty: run only from a committed state")
    run_dir, evidence = ROOT / RUN_DIR, ROOT / EVIDENCE_DIR
    if evidence.exists():
        raise Refused(f"{EVIDENCE_DIR} exists: stage 1 runs once")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir()                                          # the claim: exclusive
    except FileExistsError:
        raise Refused(f"{RUN_DIR} exists: stage 1 runs once") from None
    t0 = time.time()
    fin: Dict[str, Any] = {"head": _git("rev-parse", "HEAD"), "python": sys.version, "clone_root": clone_root,
                           "module_sha256": _sha(pathlib.Path(__file__).read_bytes()),
                           "gate_sha256": _sha(pathlib.Path(GATE.__file__).read_bytes()),
                           "started_unix": t0, "ceiling_s": EXECUTION_S}
    _write_json(run_dir / "claim.json", fin)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    _write_new(run_dir / "environment.txt", (f"python {sys.version}\nexecutable {sys.executable}\n"
                                             f"pip freeze exit {freeze.returncode}\n{freeze.stdout}").encode())
    status, child_exit, reaped, cleanup_error = "UNKNOWN", None, None, None
    try:
        with open(run_dir / "child.stdout.txt", "xb") as out, open(run_dir / "child.stderr.txt", "xb") as err:
            status, child_exit, reaped, cleanup_error = run_group(
                [sys.executable, "-m", MODULE, "_child", "--run-dir", str(run_dir), "--clone-root", clone_root],
                stdout=out, stderr=err, timeout=EXECUTION_S)
    finally:
        code = exit_code_for(run_dir / "record.json", status, child_exit, reaped)
        fin.update(status=status, child_exit=child_exit, group_reaped=reaped, cleanup_error=cleanup_error,
                   exit_code=code, elapsed_s=round(time.time() - t0, 3))
        if reaped is not True:
            _write_json(ROOT / CLEANUP_FAILED, {**fin, "finalized": False, "run_dir": RUN_DIR,
                                                "detail": "the process group's end was not verified (not reaped, "
                                                          "the check failed, or it never ran): nothing hashed, copied or "
                                                          "finalized"})
    if reaped is not True:
        return code
    fin.update(verdict=_verdict(run_dir / "record.json"),
               files={str(p.relative_to(run_dir)): _sha(p.read_bytes())
                      for p in sorted(run_dir.rglob("*")) if p.is_file()})
    _write_json(run_dir / "finalized.json", fin)
    shutil.copytree(run_dir, evidence)
    manifest = "".join(f"{_sha(p.read_bytes())}  {p.relative_to(evidence)}\n"
                       for p in sorted(evidence.rglob("*")) if p.is_file())
    _write_new(evidence / "MANIFEST.sha256", manifest.encode())
    return code


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="External-engine learning, stage 1 (gated).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--clone-root", required=True)
    p = sub.add_parser("_child")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--clone-root", required=True)
    p = sub.add_parser("_query")
    p.add_argument("--clone-root", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("_reload")
    p.add_argument("--rows", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "run":
            return run(a.clone_root)
        if a.cmd == "_child":
            return _child(pathlib.Path(a.run_dir), a.clone_root)
        if a.cmd == "_query":
            _require_gate()
            rt, _identity = load_runtime(a.clone_root)
            _write_json(pathlib.Path(a.out), query_fixtures(rt))
            return 0
        _write_json(pathlib.Path(a.out), reload_summary(pathlib.Path(a.rows)))
        return 0
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
