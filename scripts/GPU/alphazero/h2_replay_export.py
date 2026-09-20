"""Export the recorded H2 games to the JSON format consumed by Replay.html.

H2 persisted its opening boundary and every post-opening ply.  This module is a
read-only adapter: it validates the stored transcript against its digest, joins
the frozen six-ply opening to the recorded continuation, and writes a standalone
viewer file.  It never invokes an agent, draws a seed, or changes study evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from . import h2_match_plan as PLAN
from . import h2_match_rules as RULES


DEFAULT_RESULTS = Path(
    "docs/superpowers/evidence/2026-09-12-t1j-h2-match-attempt3/"
    "03_h2_results.jsonl"
)


class H2ReplayExportError(ValueError):
    """The recorded game cannot be exported without guessing."""


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, 1):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise H2ReplayExportError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from None
                if not isinstance(obj, dict):
                    raise H2ReplayExportError(
                        f"{path}:{line_number}: record is not an object"
                    )
                yield obj
    except FileNotFoundError:
        raise H2ReplayExportError(f"results file does not exist: {path}") from None


def load_recorded_games(path: Path = DEFAULT_RESULTS) -> List[Dict[str, Any]]:
    """Load complete H2 games in their recorded order.

    The evidence is grouped by ``task_id`` rather than assumed to be adjacent.
    A game is returned only when it has exactly one result, opening boundary and
    transcript record.  The interrupted task in the VOID H2 run therefore never
    appears as an exportable game.
    """
    header: Dict[str, Any] | None = None
    order: List[str] = []
    grouped: Dict[str, Dict[str, Any]] = {}

    for record in _read_jsonl(Path(path)):
        kind = record.get("record_type")
        if kind == "header":
            if header is not None:
                raise H2ReplayExportError("more than one H2 header")
            header = record
            continue
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise H2ReplayExportError(
                f"{kind!r} record has no non-empty task_id"
            )
        game = grouped.setdefault(task_id, {"plies": []})
        if kind == "ply":
            game["plies"].append(record)
        elif kind in {"task_result", "opening_bound", "transcript"}:
            if kind in game:
                raise H2ReplayExportError(
                    f"{task_id}: more than one {kind} record"
                )
            game[kind] = record
            if kind == "task_result":
                order.append(task_id)

    if header is None or header.get("design") != "H2":
        raise H2ReplayExportError("results do not carry exactly one H2 header")
    if len(order) != len(set(order)):
        raise H2ReplayExportError("a task_result task_id occurs more than once")

    complete: List[Dict[str, Any]] = []
    for task_id in order:
        game = grouped[task_id]
        missing = [k for k in ("task_result", "opening_bound", "transcript")
                   if k not in game]
        if missing:
            raise H2ReplayExportError(
                f"{task_id}: completed result is missing {missing}"
            )
        complete.append({"task_id": task_id, "header": header, **game})
    return complete


def _viewer_move(turn: int, player: str, move: Sequence[Any]) -> Dict[str, Any]:
    if (type(turn) is not int or turn < 1 or player not in {"red", "black"}
            or not isinstance(move, (list, tuple)) or len(move) != 2
            or type(move[0]) is not int or type(move[1]) is not int):
        raise H2ReplayExportError(
            f"invalid move at turn {turn!r}: player={player!r}, move={move!r}"
        )
    return {
        "turn": turn,
        "player": player,
        "row": move[0],
        "col": move[1],
        "bridges_created": [],
        "heuristics": {},
        "search_score": None,
        "root_top1_share": None,
    }


def viewer_record(game: Mapping[str, Any], *,
                  openings: Mapping[str, Sequence[Sequence[int]]] | None = None,
                  source: Path = DEFAULT_RESULTS) -> Dict[str, Any]:
    """Validate one recorded game and return a Replay.html-compatible object."""
    result = game["task_result"]
    bound = game["opening_bound"]
    transcript = game["transcript"]
    plies = list(game["plies"])
    task_id = game["task_id"]
    for record in (result, bound, transcript, *plies):
        if record.get("task_id") != task_id:
            raise H2ReplayExportError(f"{task_id}: mixed task ids in one game")

    opening_name = transcript.get("opening")
    opening_map = openings if openings is not None else PLAN.load_source_plan()["openings"]
    if opening_name not in opening_map:
        raise H2ReplayExportError(f"{task_id}: unknown opening {opening_name!r}")
    opening = list(opening_map[opening_name])
    opening_bound = bound.get("ply")
    if (type(opening_bound) is not int or opening_bound != len(opening)
            or transcript.get("opening_bound") != opening_bound):
        raise H2ReplayExportError(
            f"{task_id}: opening boundary does not equal the frozen opening"
        )

    frozen = RULES.transcript(plies, result, opening_bound=opening_bound)
    got_digest = RULES.transcript_digest(frozen)
    if transcript.get("transcript_digest") != got_digest:
        raise H2ReplayExportError(
            f"{task_id}: transcript digest mismatch: recorded "
            f"{transcript.get('transcript_digest')!r}, recomputed {got_digest}"
        )
    if transcript.get("n_plies") != len(plies):
        raise H2ReplayExportError(
            f"{task_id}: transcript count {transcript.get('n_plies')!r} "
            f"does not equal {len(plies)} stored plies"
        )

    moves: List[Dict[str, Any]] = []
    for turn, move in enumerate(opening, 1):
        moves.append(_viewer_move(turn, RULES.colour_at_ply(turn), move))
    for ply in plies:
        moves.append(_viewer_move(ply["ply"], ply["mover"], ply["move"]))

    declared_plies = result.get("plies")
    if type(declared_plies) is not int or declared_plies != len(moves):
        raise H2ReplayExportError(
            f"{task_id}: result declares {declared_plies!r} plies but the "
            f"frozen opening plus recorded continuation has {len(moves)}"
        )
    if [m["turn"] for m in moves] != list(range(1, len(moves) + 1)):
        raise H2ReplayExportError(f"{task_id}: full move sequence is not contiguous")

    winner = result.get("winner")
    reason = result.get("terminal_reason")
    header = game["header"]
    simulations = header.get("identity", {}).get("eval_config", {}).get("mcts_sims")
    colour_arm = transcript.get("colour_arm")
    if colour_arm == "t1j_red":
        players = {"red": "T1j (theirs)", "black": "Incumbent (ours)"}
    elif colour_arm == "t1j_black":
        players = {"red": "Incumbent (ours)", "black": "T1j (theirs)"}
    else:
        raise H2ReplayExportError(
            f"{task_id}: unknown colour arm {colour_arm!r}"
        )
    return {
        "id": task_id,
        "config_hash": "h2-recorded-evidence",
        "depth": simulations,
        "seed": result.get("seed"),
        "winner": winner if winner is not None else "draw",
        "starting_player": RULES.STARTING_COLOUR,
        "moves": moves,
        "meta": {
            "board_size": 24,
            "mode": "recorded_h2_match",
            "reason": reason,
            "n_moves": len(moves),
            "simulations": simulations,
            "starting_player": RULES.STARTING_COLOUR,
            "task_id": task_id,
            "opening": opening_name,
            "opening_bound": opening_bound,
            "colour_arm": colour_arm,
            "players": players,
            "transcript_digest": got_digest,
            "source": str(source),
            "evidence_note": (
                "Exact recorded H2 game from the partial VOID attempt; this "
                "viewer export is derived evidence and is not an H3 result."
            ),
        },
    }


def write_viewer_record(record: Mapping[str, Any], path: Path) -> Path:
    """Write one viewer file create-only so an earlier export is never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except FileExistsError:
        raise H2ReplayExportError(f"refusing to replace existing export: {path}") from None
    return path


def _summary(index: int, game: Mapping[str, Any]) -> str:
    result = game["task_result"]
    transcript = game["transcript"]
    return (f"{index:3d}  {game['task_id']}  opening={transcript['opening']}  "
            f"winner={result.get('winner') or 'draw'}  "
            f"reason={result.get('terminal_reason')}  plies={result.get('plies')}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export exact recorded H2 games for the browser replay viewer")
    parser.add_argument("--source", type=Path, default=DEFAULT_RESULTS)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--list", action="store_true",
                           help="list exportable games without writing")
    selection.add_argument("--index", type=int, help="zero-based recorded game index")
    selection.add_argument("--task-id", help="exact recorded task id")
    selection.add_argument("--all", action="store_true", help="export every completed game")
    parser.add_argument("--out", type=Path, help="output JSON for --index/--task-id")
    parser.add_argument("--out-dir", type=Path, help="output directory for --all")
    args = parser.parse_args(argv)

    games = load_recorded_games(args.source)
    if args.list:
        for index, game in enumerate(games):
            print(_summary(index, game))
        return 0
    if args.all:
        if args.out_dir is None or args.out is not None:
            parser.error("--all requires --out-dir and does not accept --out")
        for index, game in enumerate(games):
            record = viewer_record(game, source=args.source)
            write_viewer_record(record, args.out_dir / f"{index:03d}_{game['task_id']}.json")
        print(f"exported {len(games)} exact H2 games to {args.out_dir}")
        return 0
    if args.out is None or args.out_dir is not None:
        parser.error("--index/--task-id requires --out and does not accept --out-dir")

    if args.index is not None:
        if args.index < 0 or args.index >= len(games):
            raise H2ReplayExportError(
                f"index {args.index} outside 0..{len(games) - 1}")
        game = games[args.index]
    else:
        matches = [game for game in games if game["task_id"] == args.task_id]
        if len(matches) != 1:
            raise H2ReplayExportError(
                f"task id {args.task_id!r} matched {len(matches)} completed games")
        game = matches[0]
    write_viewer_record(viewer_record(game, source=args.source), args.out)
    print(f"exported {game['task_id']} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
