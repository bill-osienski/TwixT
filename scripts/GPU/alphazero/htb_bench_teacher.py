"""Student-teacher benchmark: the teacher process (card docs/superpowers/2026-11-14-htb-teacher-benchmark-card.md).

Run only by tests/htb_integration/teacher_bench.mjs, inside the teacher venv, behind the stage's gate in
tests/htb_integration/teacher_bench_authorization.mjs. It speaks JSON lines on stdin/stdout:

  {"cmd": "start", "opening": [[r, c], ...]}  -> a new game, the opening applied ply by ply with every check
  {"cmd": "apply", "move": [r, c]}            -> one move applied, with every check
  {"cmd": "move"}                             -> the teacher's move for the side to move (NOT applied)
  {"cmd": "close"}                            -> exit 0

Every reply to start/apply carries the canonical state's sha256 (`canon`) and twixtbot's own early-draw flag for both
lanes (`tb_draw_ply`, `tb_draw_ply_reflected`): purely observational, never compared and never a stop. Any check
failure replies {"ok": false, "stop": ..., "detail": ...} and exits 3. Before the teacher model loads, `serve` checks
the runtime against card §4 (TEACHER_ENV) and refuses (exit 4) on any difference.

THE TEACHER (card §4, §5): the H-TB label path's wrapper (htb_pilot.TwixtbotTeacher): the raw network, rotation off,
softmax over our legal moves, averaged with the left-right reflection (reflect_rc), as in label_game. Its move is the
argmax of that average, ties to the lower (row, col). No search.

LOCKSTEP (card §7): four lanes advance one move at a time: our TwixtState and twixtbot's Game, and the same for the
reflected moves. After the opening and after every ply: twixtbot_adapter.state_divergences (ply_cap=None) on both
pairs, with its `terminal` line left out (with no cap it differs only through twixtbot's early-draw flag), the
rules-level terminal status compared directly, and the reflected state required to be the reflection.

`canon <moves.json>` prints the per-ply canonical digests of our TwixtState for a move list (no teacher, no gate):
the cross-language test compares them with Node's.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import pathlib
import re
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import htb_pilot as P
from . import twixtbot_adapter as A
from .game.twixt_state import TwixtState

ROOT = pathlib.Path(__file__).resolve().parents[3]
GATE_REL = "tests/htb_integration/teacher_bench_authorization.mjs"
GATES = {"smoke": "BENCH_SMOKE_AUTHORIZED", "match": "BENCH_MATCH_AUTHORIZED"}
#: The teacher runtime pinned in card §4, checked BEFORE the teacher model loads (package metadata only; nothing
#: imported from TensorFlow for the check).
TEACHER_ENV = {"python": "3.13.15", "tensorflow": "2.21.0", "numpy": "2.5.2"}


class Stop(RuntimeError):
    def __init__(self, step: str, detail: str):
        super().__init__(f"STOP at {step}: {detail}")
        self.step, self.detail = step, detail


def gate_open(stage: str, root: pathlib.Path = ROOT) -> bool:
    if stage not in GATES:
        return False
    text = (root / GATE_REL).read_text(encoding="utf-8")
    return re.search(rf"^export const {GATES[stage]} = true;$", text, re.M) is not None


# ─────────────────────────────── the canonical state (shared with Node, byte for byte) ───────────────────────────────

def _rc(p) -> List[int]:
    return [int(p[0]), int(p[1])]


def natural_terminal(s: TwixtState) -> bool:
    """The rules alone: a winner, or no legal move. No cap (the product's 600-ply limit is unreachable on 24x24)."""
    return s.winner() is not None or not s.legal_moves()


def canonical(s: TwixtState) -> Dict[str, Any]:
    pegs = {"red": [], "black": []}
    for p, colour in s.pegs.items():
        pegs[colour].append(_rc(p))
    bridges = {"red": [], "black": []}
    for a, b in s.bridges:
        a, b = sorted([tuple(_rc(a)), tuple(_rc(b))])
        owner = s.pegs[a]
        bridges[owner].append([list(a), list(b)])
    for d in (pegs, bridges):
        for k in d:
            d[k].sort()
    return {
        "ply": int(s.ply),
        "to_move": s.to_move,
        "pegs": pegs,
        "bridges": bridges,
        "legal": sorted(_rc(m) for m in s.legal_moves()),
        "winner": s.winner(),
        "terminal": natural_terminal(s),
    }


def canon_digest(s: TwixtState) -> str:
    return hashlib.sha256(json.dumps(canonical(s), separators=(",", ":")).encode()).hexdigest()


# ─────────────────────────────── lockstep ───────────────────────────────

def bench_divergences(our: TwixtState, tb, twixt) -> List[str]:
    """The adapter's full check without its cap-free `terminal` line, plus the rules-level terminal status."""
    out = [d for d in A.state_divergences(our, tb, twixt, ply_cap=None) if not d.startswith("terminal")]
    theirs = A.tb_winner(tb) is not None or not A.tb_legal_moves(tb)
    if natural_terminal(our) != theirs:
        out.append(f"rules-level terminal: ours={natural_terminal(our)} theirs={theirs}")
    return out


class BenchTeacher:
    """Four lanes in lockstep and the mirror-averaged argmax. `teacher` is htb_pilot.TwixtbotTeacher (or a test double
    with the same new_game/play/evaluate), `twixt` the clone's twixt module."""

    def __init__(self, teacher, twixt):
        self.teacher, self.twixt = teacher, twixt
        self.our = self.tb = self.rour = self.rtb = None
        self.tb_draw_ply: Optional[int] = None
        self.rtb_draw_ply: Optional[int] = None

    def _check(self) -> None:
        for label, s, tb in (("position", self.our, self.tb), ("reflection", self.rour, self.rtb)):
            d = bench_divergences(s, tb, self.twixt)
            if d:
                raise Stop("lockstep", f"{label} at ply {s.ply}: {d[:3]}")
        if not P.reflect_state_matches(self.our, self.rour):
            raise Stop("lockstep", f"the reflected lane is not the reflection at ply {self.our.ply}")
        # twixtbot's early-draw flag is OBSERVATIONAL (card §6): recorded per lane, never compared, never a stop.
        if self.tb.result is not None and self.tb_draw_ply is None:
            self.tb_draw_ply = int(self.our.ply)
        if self.rtb.result is not None and self.rtb_draw_ply is None:
            self.rtb_draw_ply = int(self.our.ply)

    def _reply(self) -> Dict[str, Any]:
        return {"ok": True, "ply": int(self.our.ply), "canon": canon_digest(self.our),
                "tb_draw": self.tb.result is not None, "tb_draw_ply": self.tb_draw_ply,
                "tb_draw_ply_reflected": self.rtb_draw_ply}

    def start(self, opening: Sequence[Sequence[int]]) -> Dict[str, Any]:
        self.our, self.tb = TwixtState(), self.teacher.new_game()
        self.rour, self.rtb = TwixtState(), self.teacher.new_game()
        self.tb_draw_ply = self.rtb_draw_ply = None
        self._check()
        for rc in opening:
            self._apply(rc)
        return self._reply()

    def _apply(self, rc) -> None:
        rc = (int(rc[0]), int(rc[1]))
        if rc not in self.our.legal_moves():
            raise Stop("illegal", f"{rc} is illegal in our engine at ply {self.our.ply}")
        rrc = P.reflect_rc(rc)
        self.our = self.our.apply_move(rc)
        self.teacher.play(self.tb, rc)
        self.rour = self.rour.apply_move(rrc)
        self.teacher.play(self.rtb, rrc)
        self._check()

    def apply(self, rc) -> Dict[str, Any]:
        if self.our is None:
            raise Stop("protocol", "apply before start")
        if natural_terminal(self.our):
            raise Stop("protocol", f"apply after the game ended at ply {self.our.ply}")
        self._apply(rc)
        return self._reply()

    def mirror_policy(self) -> Tuple[List[Tuple[int, int]], np.ndarray, float]:
        """htb_pilot.label_game's averaging, on the current lanes: (legal in our order, probabilities, value)."""
        colour = self.our.to_move
        legal = [tuple(m) for m in self.our.legal_moves()]
        rlegal = [tuple(m) for m in self.rour.legal_moves()]
        p, v = self.teacher.evaluate(self.tb, colour, legal)
        rp, rv = self.teacher.evaluate(self.rtb, colour, rlegal)
        at = {m: i for i, m in enumerate(rlegal)}
        rp_back = np.array([rp[at[P.reflect_rc(m)]] for m in legal], dtype=np.float64)
        prob = 0.5 * (np.asarray(p, dtype=np.float64) + rp_back)
        value = 0.5 * (float(v) + float(rv))
        if not (np.all(np.isfinite(prob)) and np.isfinite(value)):
            raise Stop("teacher", f"non-finite output at ply {self.our.ply}")
        return legal, prob, value

    def move(self) -> Dict[str, Any]:
        if self.our is None or natural_terminal(self.our):
            raise Stop("protocol", "move requested with no game in progress")
        legal, prob, value = self.mirror_policy()
        best = max(range(len(legal)), key=lambda i: (prob[i], -legal[i][0], -legal[i][1]))
        return {"ok": True, "move": list(legal[best]), "prob": float(prob[best]), "value": value,
                "n_legal": len(legal)}


def serve(bench: BenchTeacher, inp, out) -> int:
    """The JSON-lines loop. Returns the exit code: 0 on close, 3 on a stop, 2 on a malformed request."""
    for line in inp:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "start":
                rep = bench.start(req["opening"])
            elif cmd == "apply":
                rep = bench.apply(req["move"])
            elif cmd == "move":
                rep = bench.move()
            elif cmd == "close":
                out.write(json.dumps({"ok": True, "closed": True}) + "\n")
                out.flush()
                return 0
            else:
                raise Stop("protocol", f"unknown command {cmd!r}")
        except (Stop, P.Stop) as e:
            out.write(json.dumps({"ok": False, "stop": e.step, "detail": e.detail}) + "\n")
            out.flush()
            return 3
        except (KeyError, TypeError, ValueError) as e:
            out.write(json.dumps({"ok": False, "stop": "protocol", "detail": f"malformed request: {e}"}) + "\n")
            out.flush()
            return 2
        out.write(json.dumps(rep) + "\n")
        out.flush()
    return 3  # stdin closed without `close`: the driver is gone


def runtime_versions() -> Dict[str, str]:
    from importlib import metadata

    def ver(dist: str) -> str:
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            return "absent"
    return {"python": ".".join(map(str, sys.version_info[:3])), "tensorflow": ver("tensorflow"),
            "numpy": ver("numpy")}


def env_mismatches(found: Mapping[str, str]) -> List[str]:
    return [f"{k}: {found.get(k)} (card: {v})" for k, v in TEACHER_ENV.items() if found.get(k) != v]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--stage", required=True, choices=sorted(GATES))
    s.add_argument("--clone-root", required=True)
    c = sub.add_parser("canon")
    c.add_argument("moves")
    a = ap.parse_args(argv)
    if a.cmd == "canon":
        moves = json.loads(pathlib.Path(a.moves).read_text(encoding="utf-8"))
        st = TwixtState()
        digests = [canon_digest(st)]
        for m in moves:
            st = st.apply_move((int(m[0]), int(m[1])))
            digests.append(canon_digest(st))
        print(json.dumps(digests))
        return 0
    if not gate_open(a.stage):
        print(f"REFUSED: {GATES[a.stage]} is closed", file=sys.stderr)
        return 4
    env = runtime_versions()
    bad = env_mismatches(env)
    if bad:  # before any model load
        sys.stdout.write(json.dumps({"ok": False, "refused": "runtime", "detail": bad, "env": env}) + "\n")
        sys.stdout.flush()
        print(f"REFUSED: the teacher runtime is not the card's: {bad}", file=sys.stderr)
        return 4
    t0 = time.time()
    with contextlib.redirect_stdout(sys.stderr):  # stdout carries the protocol only
        teacher, identity = P._load_teacher_real(a.clone_root)
    bench = BenchTeacher(teacher, teacher.rt["twixt"])
    sys.stdout.write(json.dumps({"ok": True, "ready": True, "env": env, "identity": identity, "teacher": P.TEACHER,
                                 "load_seconds": time.time() - t0}, default=P._jsonable) + "\n")
    sys.stdout.flush()
    return serve(bench, sys.stdin, sys.stdout)


if __name__ == "__main__":
    sys.exit(main())
