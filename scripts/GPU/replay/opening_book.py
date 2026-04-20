from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .format import GameRecord


def _iter_game_paths(games_dir: Path) -> Iterable[Path]:
    return games_dir.rglob("game-*.json")


def _load_record(path: Path) -> GameRecord:
    data = json.loads(path.read_text(encoding="utf-8"))
    return GameRecord.from_dict(data)


def _move_token(player: str, row: int, col: int) -> str:
    return f"{player[0].lower()}{row},{col}"


def _opening_key(
    *,
    moves: List[Tuple[str, int, int]],
    board_size: int,
    starting_player: str,
) -> str:
    parts = [f"b:{board_size}", f"s:{starting_player.lower()}"]
    parts.extend(_move_token(p, r, c) for p, r, c in moves)
    return "|".join(parts)


def _side_value(winner: str, side: str) -> float:
    if winner == "draw":
        return 0.0
    return 1.0 if winner == side else -1.0


def build_opening_book(
    *,
    games_dir: Path,
    max_plies: int,
    top_k: int,
    min_visits: int,
) -> Dict:
    position_stats: Dict[str, Dict] = {}

    for path in _iter_game_paths(games_dir):
        rec = _load_record(path)
        if not rec.moves:
            continue

        starting_player = rec.moves[0].player
        board_size = int(rec.meta.get("board_size", 24))
        winner = rec.winner

        prefix: List[Tuple[str, int, int]] = []
        for idx, mv in enumerate(rec.moves):
            if idx >= max_plies:
                break

            key = _opening_key(
                moves=prefix,
                board_size=board_size,
                starting_player=starting_player,
            )
            side_to_move = mv.player
            move_key = f"{mv.row},{mv.col}"
            value = _side_value(winner, side_to_move)

            slot = position_stats.setdefault(
                key,
                {
                    "side_to_move": side_to_move,
                    "ply": idx,
                    "total": 0,
                    "moves": defaultdict(lambda: {"count": 0, "value_sum": 0.0}),
                },
            )

            slot["total"] += 1
            slot["moves"][move_key]["count"] += 1
            slot["moves"][move_key]["value_sum"] += value

            prefix.append((mv.player, mv.row, mv.col))

    positions = {}
    for key, slot in position_stats.items():
        total = slot["total"]
        if total < min_visits:
            continue

        move_rows = []
        for move_key, stats in slot["moves"].items():
            row_s, col_s = move_key.split(",", 1)
            count = stats["count"]
            move_rows.append(
                {
                    "row": int(row_s),
                    "col": int(col_s),
                    "prior": count / total,
                    "value": stats["value_sum"] / max(1, count),
                    "visits": int(count),
                }
            )

        move_rows.sort(key=lambda m: (-m["visits"], -m["prior"]))
        positions[key] = {
            "side_to_move": slot["side_to_move"],
            "ply": slot["ply"],
            "top_k": move_rows[:top_k],
        }

    return positions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an opening book from replay games.")
    parser.add_argument("--games-dir", default="scripts/GPU/logs/games", help="Directory with replay games.")
    parser.add_argument("--output", default="assets/js/ai/opening-book.json", help="Output JSON path.")
    parser.add_argument("--max-plies", type=int, default=12, help="Max plies to include.")
    parser.add_argument("--top-k", type=int, default=8, help="Top K moves to store per position.")
    parser.add_argument("--min-visits", type=int, default=10, help="Min visits to keep a position.")
    args = parser.parse_args()

    games_dir = Path(args.games_dir).resolve()
    output_path = Path(args.output).resolve()

    positions = build_opening_book(
        games_dir=games_dir,
        max_plies=int(args.max_plies),
        top_k=int(args.top_k),
        min_visits=int(args.min_visits),
    )

    payload = {
        "version": 1,
        "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "plies": list(range(0, int(args.max_plies))),
        "board_size": 24,
        "key_format": "b:<size>|s:<start>|<p><row>,<col>|...",
        "positions": positions,
        "stats": {
            "positions": len(positions),
            "top_k": int(args.top_k),
            "min_visits": int(args.min_visits),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote opening book -> {output_path}")


if __name__ == "__main__":
    main()
