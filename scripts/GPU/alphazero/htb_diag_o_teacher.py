"""Diagnostic O, stage B: the pinned twixtbot raw network on the saved opening positions.

Plan: docs/superpowers/2026-11-12-htb-diagnostic-o-plan.md. Run only by tests/htb_integration/diag_o.mjs, inside the
teacher venv, behind DIAG_O_AUTHORIZED (tests/htb_integration/diag_o_authorization.mjs).

For every input position (a red-first move prefix, Black to move) it evaluates, with the H-TB label path's own teacher
wrapper (htb_pilot.TwixtbotTeacher via htb_pilot._load_teacher_real) and its mirror-averaging (the position and its
left-right reflection replayed in lockstep, the reflection's policy mapped back, both averaged):
  - the position itself: the value from Black's side, the policy on our legal moves, the teacher's top move, and the
    probability and rank of each candidate move;
  - the position after each candidate move (Red to move there): the value, negated to Black's side.
Every replay checks per-ply state equivalence with our engine (the adapter's divergence check); any divergence stops.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from . import htb_pilot as P
from .game.twixt_state import TwixtState

ROOT = pathlib.Path(__file__).resolve().parents[3]
GATE_REL = "tests/htb_integration/diag_o_authorization.mjs"


class Stop(RuntimeError):
    pass


def gate_open(root: pathlib.Path = ROOT) -> bool:
    text = (root / GATE_REL).read_text(encoding="utf-8")
    return re.search(r"^export const DIAG_O_AUTHORIZED = true;$", text, re.M) is not None


def _key(rc: Sequence[int]) -> str:
    return f"{int(rc[0])},{int(rc[1])}"


def _decode(k: str) -> Tuple[int, int]:
    r, c = k.split(",")
    return int(r), int(c)


def evaluate_final(teacher, moves: Sequence[Sequence[int]]):
    """Replay `moves` in lockstep through our engine and the teacher's game; evaluate the final position."""
    our, tb = TwixtState(), teacher.new_game()
    for k, rc in enumerate(moves):
        rc = (int(rc[0]), int(rc[1]))
        if rc not in our.legal_moves():
            raise Stop(f"move {k} {rc} is illegal in our engine")
        our = our.apply_move(rc)
        teacher.play(tb, rc)
    d = teacher.divergences(our, tb)
    if d:
        raise Stop(f"lockstep divergence after {len(moves)} moves: {d[:3]}")
    legal = [tuple(m) for m in our.legal_moves()]
    p, v = teacher.evaluate(tb, our.to_move, legal)
    return our, legal, np.asarray(p, dtype=np.float64), float(v)


def mirror_evaluate(teacher, moves: Sequence[Sequence[int]]):
    """The label path's mirror-averaging (htb_pilot.label_game), for the final position of `moves`."""
    s, legal, p, v = evaluate_final(teacher, moves)
    rs, rlegal, rp, rv = evaluate_final(teacher, [P.reflect_rc(m) for m in moves])
    if not P.reflect_state_matches(s, rs):
        raise Stop("the reflected replay is not the reflection of the state")
    at = {tuple(m): i for i, m in enumerate(rlegal)}
    rp_back = np.array([rp[at[P.reflect_rc(m)]] for m in legal])
    prob = 0.5 * (p + rp_back)
    value = 0.5 * (v + rv)
    if not (np.all(np.isfinite(prob)) and np.isfinite(value)):
        raise Stop("non-finite teacher output")
    return s.to_move, legal, prob, value


def evaluate_position(teacher, prefix: Sequence[Sequence[int]], candidates: Sequence[str]) -> Dict[str, Any]:
    to_move, legal, prob, value = mirror_evaluate(teacher, prefix)
    if to_move != "black":
        raise Stop(f"expected Black to move after {len(prefix)} moves, got {to_move}")
    idx = {_key(m): i for i, m in enumerate(legal)}
    order = np.argsort(-prob, kind="stable")
    top = _key(legal[int(order[0])])
    out_prob, out_rank, after = {}, {}, {}
    for c in candidates:
        if c not in idx:
            raise Stop(f"candidate {c} is not legal after {len(prefix)} moves")
        pc = float(prob[idx[c]])
        out_prob[c] = pc
        out_rank[c] = 1 + int(np.sum(prob > pc))
        tm2, _, _, v2 = mirror_evaluate(teacher, list(prefix) + [list(_decode(c))])
        if tm2 != "red":
            raise Stop("expected Red to move after a Black candidate")
        after[c] = -float(v2)  # Red to move there; Black's side
    return {"value_black": float(value), "top_move": top, "top_prob": float(prob[int(order[0])]),
            "prob": out_prob, "rank": out_rank, "after_value_black": after}


def run(teacher, items: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for it in items:
        rec = evaluate_position(teacher, it["prefix"], it["candidates"])
        out.append({"pair": it["pair"], "ply": it["ply"], **rec})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--clone-root", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not gate_open():
        print("REFUSED: DIAG_O_AUTHORIZED is closed", file=sys.stderr)
        return 4
    out = pathlib.Path(a.out)
    if out.exists():
        print(f"REFUSED: {out} exists", file=sys.stderr)
        return 4
    items = json.loads(pathlib.Path(a.inp).read_text(encoding="utf-8"))
    t0 = time.time()
    teacher, identity = P._load_teacher_real(a.clone_root)
    t1 = time.time()
    try:
        records = run(teacher, items)
    except (Stop, P.Stop) as e:
        print(f"STOP: {e}", file=sys.stderr)
        return 3
    doc = {"schema": "htb-diag-o-teacher/1", "identity": identity, "teacher": P.TEACHER, "calls": teacher.calls,
           "positions": len(records), "seconds": {"load": t1 - t0, "evaluate": time.time() - t1}, "records": records}
    with open(out, "x", encoding="utf-8") as f:
        json.dump(doc, f, default=P._jsonable if hasattr(P, "_jsonable") else str)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
