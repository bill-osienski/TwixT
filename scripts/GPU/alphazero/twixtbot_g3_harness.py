"""Minimum external-match harness for G3. PREPARATION ONLY.

Pilot card: docs/superpowers/2026-08-22-twixtbot-anchor-pilot-card.md.
`play_game` is defined here but is NOT called by the preflight: G3 games are not
authorized and no reserved seed may be consumed. Nothing in this module trains,
calibrates, scores a calibration, or touches the product service.

DESIGN RULES, each of which the preflight checks structurally:

* Our `TwixtState(max_plies_limit=280)` is AUTHORITATIVE. twixtbot's `Game` is a
  paired shadow, never the source of truth about legality or termination.
* Both engines are advanced in LOCKSTEP from the same move sequence, because our
  state carries no move history and cannot be translated from a bare state.
* `state_divergences(..., ply_cap=280)` runs after opening replay and after every
  single ply. The cap is passed explicitly; the signature makes omitting it a
  TypeError rather than a silent loss of normalisation.
* Every move an engine returns is validated in OUR engine before it is applied.
* The game ABORTS on divergence, illegal move, resignation/swap, engine
  exception, or malformed output. It is never resampled, repaired or retried,
  and an aborted game is recorded as aborted rather than dropped.
* The FULL twixtbot visit array is preserved per move. Visit concentration is
  reported descriptively; it is NOT a gate and does NOT alter the ladder.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import twixtbot_adapter as A
from .twixtbot_g3_schedule import PLY_CAP

#: Every way a game may stop without a natural result. All are terminal for that
#: game: none is a retry condition.
ABORT_REASONS = (
    "state_divergence",
    "illegal_move",
    "resignation_or_swap",
    "engine_exception",
    "malformed_output",
    "ply_cap_without_result",
)


class HarnessAbort(Exception):
    """A game stopped abnormally. Carries the reason and the ply it happened at."""

    def __init__(self, reason: str, detail: str, ply: int):
        if reason not in ABORT_REASONS:
            raise ValueError(f"unknown abort reason {reason!r}")
        super().__init__(f"[{reason} @ ply {ply}] {detail}")
        self.reason, self.detail, self.ply = reason, detail, ply


def anchor_player_kwargs(trials: int, ct) -> Dict:
    """The frozen anchor settings. Not parameters beyond `trials`."""
    return dict(
        model="model/pb",
        trials=trials,
        temperature=0,
        add_noise=0,
        rotation=ct.ROT_OFF,
        allow_swap=0,
    )


def anchor_move(player_factory, tb_game, our_state, ply: int) -> Tuple[Tuple[int, int], Dict]:
    """One anchor move under a FRESH Player, with its full visit array preserved.

    A fresh Player per move is required: NeuralMCTS caches `self.root`, so a
    reused player would accumulate a tree across moves.
    """
    sink = A.ProgressSink()
    try:
        resp = player_factory().pick_move(tb_game, window=sink)
    except Exception as e:                                   # noqa: BLE001
        raise HarnessAbort("engine_exception", f"{type(e).__name__}: {e}", ply) from e

    if not isinstance(resp, dict) or "moves" not in resp or "Y" not in resp:
        raise HarnessAbort("malformed_output", f"response keys {sorted(resp) if isinstance(resp, dict) else type(resp)}", ply)

    top = resp["moves"][0] if resp["moves"] else None
    if top is None:
        raise HarnessAbort("malformed_output", "empty move list", ply)
    if not hasattr(top, "x"):
        # twixt.SWAP or a resign string reaches us here.
        raise HarnessAbort("resignation_or_swap", f"non-Point move {top!r}", ply)

    rc = A.xy_to_rc(int(top.x), int(top.y))
    if rc not in our_state.legal_moves():
        raise HarnessAbort("illegal_move", f"anchor chose {rc}, illegal in our engine", ply)

    record = {
        "ply": ply,
        "mover": "anchor",
        "move": list(rc),
        # FULL array, never a head slice: a truncated transcript reads as a
        # complete one, and this is the observation G2 asked us to characterise.
        "visits": [int(v) for v in resp["Y"]],
        "moves_order": [[int(m.x), int(m.y)] for m in resp["moves"]],
        "proven": bool(resp.get("proven")),
        "progress_events": list(sink.events),
    }
    return rc, record


def reference_move(reference_fn, our_state, ply: int) -> Tuple[Tuple[int, int], Dict]:
    """One reference (our MLX agent) move, validated in our own engine."""
    try:
        rc = reference_fn(our_state)
    except Exception as e:                                   # noqa: BLE001
        raise HarnessAbort("engine_exception", f"{type(e).__name__}: {e}", ply) from e
    if not (isinstance(rc, (tuple, list)) and len(rc) == 2):
        raise HarnessAbort("malformed_output", f"reference returned {rc!r}", ply)
    rc = (int(rc[0]), int(rc[1]))
    if rc not in our_state.legal_moves():
        raise HarnessAbort("illegal_move", f"reference chose {rc}, illegal in our engine", ply)
    return rc, {"ply": ply, "mover": "reference", "move": list(rc)}


def play_game(
    *,
    task: dict,
    twixt,
    Point,
    ct,
    TwixtState,
    player_factory: Callable,
    reference_fn: Callable,
    ply_cap: int = PLY_CAP,
) -> dict:
    """Play ONE game. NOT called by the preflight; G3 games are unauthorized.

    Returns a record with the result and every move; raises nothing on a normal
    finish. An abort is caught and recorded, never retried.
    """
    anchor_colour = task["anchor_colour"]
    our = TwixtState(max_plies_limit=ply_cap)
    tb = twixt.Game(allow_scl=False)
    moves: List[Dict] = []

    record = {
        "task_index": task["task_index"], "seed": task["seed"],
        "trials": task["trials"], "reference": task["reference"],
        "opening_id": task["opening_id"], "colour_arm": task["colour_arm"],
        "anchor_colour": anchor_colour, "ply_cap": ply_cap,
        "aborted": False, "abort_reason": None, "abort_detail": None,
    }

    def bind(ply: int, label: str):
        d = A.state_divergences(our, tb, twixt, ply_cap=ply_cap)
        if d:
            raise HarnessAbort("state_divergence", f"{label}: {d}", ply)

    try:
        for i, (r, c) in enumerate(task["opening_moves"]):
            rc = (int(r), int(c))
            if rc not in our.legal_moves():
                raise HarnessAbort("illegal_move", f"opening move {rc} illegal", i)
            our = our.apply_move(rc)
            tb.play(Point(*A.rc_to_xy(*rc)))
            moves.append({"ply": i, "mover": "opening", "move": list(rc)})
        bind(len(task["opening_moves"]), "after opening replay")

        while not our.is_terminal():
            ply = our.ply
            if our.to_move == anchor_colour:
                rc, mrec = anchor_move(player_factory, tb, our, ply)
            else:
                rc, mrec = reference_move(reference_fn, our, ply)
            our = our.apply_move(rc)
            tb.play(Point(*A.rc_to_xy(*rc)))
            moves.append(mrec)
            bind(our.ply, f"after ply {ply} ({mrec['mover']} {rc})")

        record["winner"] = our.winner()
        record["plies"] = our.ply
        record["result"] = (
            "draw" if our.winner() is None else
            ("anchor" if our.winner() == anchor_colour else "reference")
        )
        if our.winner() is None and our.ply >= ply_cap:
            record["result"] = "draw_ply_cap"
    except HarnessAbort as e:
        record.update(aborted=True, abort_reason=e.reason,
                      abort_detail=e.detail, plies=our.ply, result=None, winner=None)

    record["moves"] = moves
    return record
