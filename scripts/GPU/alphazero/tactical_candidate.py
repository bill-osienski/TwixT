"""Exact one-ply tactics as a CANDIDATE: a default-off readout wrapper and an independent checker.

`docs/superpowers/2026-10-05-tactical-candidate-evaluation-card.md` §1-§2 (+ §11 A1, §12 A2).

THE WRAPPER. The incumbent's unchanged search runs ONCE. Only when ENABLED, the frozen M1
readout of the H5 tactics card -- `classify` / `m1_choice`, imported unchanged -- replaces a
baseline move that is not admissible; an admissible baseline is preserved. Protocol semantics
only, cap 280; 24x24 only. It reads: no stream is drawn from, no tree or configuration changed.

THE CHECKER. `exhaustive_scan` classifies by brute force from the rules engine ALONE -- no
helper of the optimized detector (`h5_cap_completion`, `h5_weakness_review`, `d0_postmortem`,
`h5_incumbent_tactics`): every legal move, and after every non-winning one every legal reply.
It never examines replies after a winning move or after protocol termination, and it checks
the detector's exact-reduction premise (a mover's own peg never creates a new opponent win).
Qualification only -- never in the production play loop (card §2).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from . import eval_readout as RO
from . import h5_incumbent_tactics as T
from .game.twixt_state import TwixtState

BOARD = 24                       # §11 A2.1 of the tactics card; a test pins it to T.BOARD
PLY_CAP = 280                    # the protocol cap; a test pins it to T.PLY_CAP
INACTIVE = ("UNDEFENDABLE", "QUIET")
REASON = {"OWN_WIN": "an own immediate win exists and the baseline is not one",
          "DEFEND_SINGLE": "the baseline leaves an immediate winning reply",
          "DEFEND_MULTI": "the baseline leaves an immediate winning reply"}


class CandidateRefused(Exception):
    """An input the candidate cannot act on: nothing was decided."""


def _t(m) -> Tuple[int, int]:
    return (int(m[0]), int(m[1]))


# ─────────────────────────────── the wrapper (card §1) ───────────────────────────────

def witness(state: TwixtState, counts, root, baseline) -> Dict[str, Any]:
    """The frozen M1 readout on the UNCHANGED search, with its record. Reads only."""
    t0 = time.perf_counter()
    c = T.classify(state, semantics="protocol")            # refuses a non-official board
    baseline = _t(baseline)
    if baseline != _t(RO._argmax(counts)):
        raise CandidateRefused(f"the baseline {list(baseline)} is not the visit argmax: the "
                               "readout is not the frozen argmax readout")
    if c is None or c["class"] in INACTIVE:
        rule, final = "inactive", baseline
    else:
        m1 = T.m1_choice(c, counts)
        rule, final = (("preserved", baseline) if m1["baseline_admissible"]
                       else ("override", _t(m1["choice"])))
    scan_s = time.perf_counter() - t0
    return {"ply": state.ply, "colour": state.to_move, "class": c, "rule": rule,
            "reason": REASON[c["class"]] if rule == "override" else None,
            "override": final != baseline, "unsearched": counts.get(final, 0) == 0,
            "baseline": T._value(root, baseline), "final": T._value(root, final),
            "value_note": T.VALUE_NOTE, "scan_s": scan_s}


class TacticalCandidate:
    """§1: the incumbent agent, then -- only when ENABLED -- the frozen M1 readout. Default OFF:
    disabled, it is the incumbent. The wrapped agent must expose its finished search
    (`capture=True`, which by contract changes neither the move nor either stream)."""

    def __init__(self, agent, *, enabled: bool = False):
        if not getattr(agent, "capture", False):
            raise CandidateRefused("the wrapped agent must expose its finished search "
                                   "(capture=True)")
        self.agent, self.enabled = agent, bool(enabled)
        self.witnesses: List[Dict[str, Any]] = []

    @property
    def colour(self) -> str:
        return self.agent.colour

    def __call__(self, state: TwixtState) -> Tuple[int, int]:
        T._official(state, "the candidate")              # before any search
        #: protocol-terminal: a winner, ply >= 280 or no legal move. A decision AT ply 279 --
        #: possibly a winning 280th move -- is still made. Enabled or not.
        if state.winner() is not None or state.ply >= PLY_CAP or not state.legal_moves():
            raise CandidateRefused(f"a protocol-terminal position (ply {state.ply}, winner "
                                   f"{state.winner()}): no decision is made")
        before = self.agent.last_capture
        t0 = time.perf_counter()
        baseline = _t(self.agent(state))                  # the unchanged search, ONCE
        search_s = time.perf_counter() - t0
        if not self.enabled:
            return baseline
        cap = self.agent.last_capture
        if cap is None or cap is before:
            raise CandidateRefused("the agent did not refresh its capture for this call")
        #: the search's root holds THE input state object (MCTSNode(state=root_state))
        if cap["ply"] != state.ply or _t(cap["move"]) != baseline or cap["root"].state is not state:
            raise CandidateRefused("the captured search is not the one that chose this move")
        w = {**witness(state, cap["counts"], cap["root"], baseline), "search_s": search_s}
        self.witnesses.append(w)
        return _t(w["final"]["move"])


# ─────────────────────────────── the independent checker (card §2.1) ───────────────────────────

def exhaustive_scan(state: TwixtState) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """The protocol class by brute force, in `classify`'s shape, plus diagnostics. Uses only
    TwixtState's own rules (legal_moves, apply_move, winner) and its constructor."""
    if (state.board_size, state.active_size) != (BOARD, BOARD):
        raise CandidateRefused(f"a {state.board_size}/{state.active_size} board, not the "
                               f"official {BOARD}x{BOARD}")
    diag: Dict[str, Any] = {"replies_examined": 0, "new_opponent_wins": []}
    mover = state.to_move
    opp = "black" if mover == "red" else "red"
    legal = state.legal_moves()
    if state.winner() is not None or not legal or state.ply >= PLY_CAP:
        return None, diag
    W = sorted(m for m in legal if state.apply_move(m).winner() == mover)
    if W:                                    # the game ends on a win: no reply is examined
        return {"class": "OWN_WIN", "W": [list(m) for m in W], "T": None, "D": None,
                "T_void": None}, diag
    flipped = TwixtState(board_size=state.board_size, active_size=state.active_size,
                         to_move=opp, pegs=dict(state.pegs), bridges=set(state.bridges),
                         ply=state.ply)
    threats = sorted(c for c in flipped.legal_moves() if flipped.apply_move(c).winner() == opp)
    if state.ply + 1 >= PLY_CAP:             # protocol termination after this move: no reply
        return {"class": "QUIET", "W": [], "T": [], "D": [],
                "T_void": [list(t) for t in threats] or None}, diag
    saving = []
    for m in legal:                          # EVERY move, EVERY reply: no reduction
        after = state.apply_move(m)
        wins = []
        for r in after.legal_moves():
            diag["replies_examined"] += 1
            if after.apply_move(r).winner() == opp:
                wins.append(r)
        diag["new_opponent_wins"] += [[list(m), list(r)] for r in wins if r not in threats]
        if not wins:
            saving.append(m)
    if not threats:
        return {"class": "QUIET", "W": [], "T": [], "D": [], "T_void": None}, diag
    cls = ("UNDEFENDABLE" if not saving else
           "DEFEND_SINGLE" if len(threats) == 1 else "DEFEND_MULTI")
    return {"class": cls, "W": [], "T": [list(t) for t in threats],
            "D": [list(m) for m in sorted(saving)], "T_void": None}, diag


def cross_check(state: TwixtState) -> Dict[str, Any]:
    """The optimized detector (via `classify`) against the exhaustive scan. Agreement needs the
    same class dict AND no new opponent win (the premise of the detector's reduction)."""
    fast = T.classify(state, semantics="protocol")
    slow, diag = exhaustive_scan(state)
    return {"agree": fast == slow and not diag["new_opponent_wins"], "classify": fast,
            "exhaustive": slow, **diag}
