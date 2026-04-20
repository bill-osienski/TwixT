from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from .format import GameRecord


def _empty_board(n: int) -> List[List[str]]:
    return [["." for _ in range(n)] for _ in range(n)]


def _apply_moves(rec: GameRecord, upto_turn: int, board_size: int = 24) -> List[List[str]]:
    b = _empty_board(board_size)
    for m in rec.moves:
        if m.turn > upto_turn:
            break
        ch = "R" if m.player.lower().startswith("r") else "B"
        if 0 <= m.row < board_size and 0 <= m.col < board_size:
            b[m.row][m.col] = ch
    return b


def _render_board(b: List[List[str]]) -> str:
    n = len(b)
    header = "    " + " ".join(f"{i:02d}" for i in range(n))
    lines = [header]
    for r in range(n):
        lines.append(f"{r:02d}: " + " ".join(b[r]))
    return "\n".join(lines)


def load_record(path: Path) -> GameRecord:
    d = json.loads(path.read_text(encoding="utf-8"))
    return GameRecord.from_dict(d)


def interactive_replay(path: Path, *, board_size: int = 24) -> None:
    rec = load_record(path)
    t = 0
    max_t = rec.moves[-1].turn if rec.moves else 0

    print(f"Game {rec.id}  winner={rec.winner}  depth={rec.depth}  seed={rec.seed}  hash={rec.config_hash}")
    print("Commands: n(ext), p(rev), j <turn>, b(oard), m(ove), q(uit)")

    while True:
        cmd = input(f"turn {t}/{max_t}> ").strip()
        if cmd in ("q", "quit", "exit"):
            return
        if cmd in ("n", "next"):
            t = min(max_t, t + 1)
        elif cmd in ("p", "prev"):
            t = max(0, t - 1)
        elif cmd.startswith("j "):
            try:
                t = max(0, min(max_t, int(cmd.split()[1])))
            except Exception:
                print("Invalid turn")
        elif cmd in ("b", "board"):
            b = _apply_moves(rec, t, board_size=board_size)
            print(_render_board(b))
        elif cmd in ("m", "move"):
            mv = next((m for m in rec.moves if m.turn == t), None)
            if mv is None:
                print("(no move)")
            else:
                print(f"turn={mv.turn} player={mv.player} row={mv.row} col={mv.col} search_score={mv.search_score}")
                if mv.heuristics:
                    keys = sorted(mv.heuristics.keys())
                    shown = ", ".join(f"{k}={mv.heuristics[k]:.3g}" for k in keys[:12])
                    more = " ..." if len(keys) > 12 else ""
                    print(f"heuristics: {shown}{more}")
        else:
            print("Unknown command")
