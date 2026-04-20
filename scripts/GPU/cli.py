from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from .config.search_config import SearchConfigIO, write_search_json
from .tuning.loop import Paths, init_logs, rank_from_sweeps, sweep_cycle, validate_queue
from .tuning.state import load_state
from .replay.viewer import interactive_replay
from .selfplay.random_policy import play_random_game, play_greedy_game


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def cmd_init(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    paths = Paths(root=root)
    init_logs(paths)
    print(f"Initialized GPU autotune logs under: {paths.logs}")


def _default_search_path(root: Path) -> Path:
    return root / "assets" / "js" / "ai" / "search.json"


def _load_base_knobs(root: Path, search_path: Optional[str]) -> Dict[str, float]:
    p = Path(search_path).resolve() if search_path else _default_search_path(root)
    knobs, _raw = SearchConfigIO(p).load()
    return knobs


def cmd_sweep(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    paths = Paths(root=root)
    init_logs(paths)

    base = _load_base_knobs(root, args.search)
    depths = _parse_int_list(args.depths)

    sweep_cycle(
        paths=paths,
        base_knobs=base,
        depths=depths,
        games=args.games,
        total=args.total,
        fixed=args.fixed,
        mutate=args.mutate,
        seed=args.seed,
        pred_gate=args.pred_gate,
        max_pred_bias=args.max_pred_bias,
        min_r2=args.min_r2,
        board=args.board,
        workers=args.workers,
    )
    print(f"Sweep complete -> {paths.sweep_results_jsonl}")


def cmd_rank(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    paths = Paths(root=root)
    init_logs(paths)
    depths = _parse_int_list(args.depths)
    queue = rank_from_sweeps(paths=paths, depths=depths, top=args.top)
    print("Top validation queue:")
    for item in queue:
        print(f"  score={item['score']:.4f}  hash={item['hash']}")
    print(f"Wrote -> {paths.pending_validation_json}")


def cmd_validate(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    paths = Paths(root=root)
    init_logs(paths)
    depths = _parse_int_list(args.depths)

    stable = validate_queue(
        paths=paths,
        depths=depths,
        games=args.games,
        pass_score=args.pass_score,
        streak_needed=args.streak_needed,
        retire_after=args.retire_after,
        board=args.board,
        workers=args.workers,
    )

    if stable:
        print(f"STABLE FOUND: {stable}")
        if args.write_search:
            st = load_state(paths.state_json)
            # Find knobs for stable from queue or last sweeps; simplest: pull from pending-validation.json
            q = json.loads(paths.pending_validation_json.read_text(encoding='utf-8'))
            item = next((x for x in q.get('queue', []) if x.get('hash') == stable), None)
            if item and item.get('knobs'):
                out_path = Path(args.write_search).resolve()
                write_search_json(out_path, dict(item['knobs']))
                print(f"Wrote winning knobs -> {out_path}")
            else:
                print("Stable hash found, but knobs were not present in pending queue. Use export --hash ...")
    else:
        print("No stable candidate yet.")


def cmd_tune(args: argparse.Namespace) -> None:
    """Full tuning loop: sweep (short games) → rank → validate (long games).

    Workflow matches JS:
    - Sweep: 10 games/depth for quick bias estimation
    - Validation: 60 games/depth with 2% threshold per depth
    - Streak: Need 4 consecutive validation passes for "stable"
    """
    import copy

    for i in range(args.cycles):
        print(f"\n{'='*60}")
        print(f"TUNING CYCLE {i+1}/{args.cycles}")
        print(f"{'='*60}")

        # Sweep phase: short games for quick filtering
        print(f"\n[SWEEP] Running {args.total} configs x {args.sweep_games} games/depth...")
        sweep_args = copy.copy(args)
        sweep_args.games = args.sweep_games  # Use sweep game count
        cmd_sweep(sweep_args)

        # Rank phase: select top candidates
        print(f"\n[RANK] Selecting top {args.top} candidates...")
        cmd_rank(args)

        # Validate phase: longer games with strict requirements
        print(f"\n[VALIDATE] Running {args.val_games} games/depth, need {args.streak_needed} consecutive passes...")
        validate_args = copy.copy(args)
        validate_args.games = args.val_games  # Use validation game count
        cmd_validate(validate_args)

        # Check if we found a stable config
        root = Path(args.root).resolve()
        st = load_state(Paths(root=root).state_json)
        if st.active_hash:
            print(f"\n*** STABLE CONFIG FOUND: {st.active_hash} ***")
            break
        else:
            print(f"\n[CYCLE {i+1}] No stable config yet, continuing...")


def cmd_replay(args: argparse.Namespace) -> None:
    interactive_replay(Path(args.path).resolve(), board_size=args.board)


def cmd_status(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    paths = Paths(root=root)
    init_logs(paths)
    st = load_state(paths.state_json)
    print(f"iteration={st.iteration} active_hash={st.active_hash} best_score={st.best_score} cycles_since_improvement={st.cycles_since_improvement}")
    # Show a few registry stats
    counts = {}
    for e in st.hash_registry.values():
        counts[e.status] = counts.get(e.status, 0) + 1
    if counts:
        print("registry:")
        for k in sorted(counts.keys()):
            print(f"  {k}: {counts[k]}")


def cmd_play(args: argparse.Namespace) -> None:
    """Interactive play mode for testing game logic."""
    from .game.state import GameState
    from .game.rules import apply_move, check_winner, generate_moves
    from .ai.search import choose_move

    def print_board(state: GameState) -> None:
        """Print ASCII board."""
        size = state.board_size
        # Header
        print("     " + " ".join(f"{c:2d}" for c in range(size)))
        print("    +" + "---" * size + "+")

        for r in range(size):
            row_str = f"{r:2d}  |"
            for c in range(size):
                pos = (r, c)
                if pos in state.pegs:
                    char = "R" if state.pegs[pos] == "red" else "B"
                elif (r == 0 or r == size - 1) and (c == 0 or c == size - 1):
                    char = "x"  # Corner
                elif r == 0 or r == size - 1:
                    char = "="  # Red goal edge
                elif c == 0 or c == size - 1:
                    char = "|"  # Black goal edge
                else:
                    char = "."
                row_str += f" {char} "
            row_str += "|"
            print(row_str)

        print("    +" + "---" * size + "+")
        print(f"    Pegs: {len(state.pegs)}  Bridges: {len(state.bridges)}  To move: {state.to_move}")

    state = GameState(board_size=args.board)
    human_player = args.human if args.human else ("black" if args.ai_vs_ai else "black")
    ai_depth = args.depth

    print("TwixT Interactive Play")
    print("=" * 40)
    print(f"Board size: {args.board}x{args.board}")
    print(f"AI depth: {ai_depth}")
    if args.ai_vs_ai:
        print("Mode: AI vs AI")
    else:
        print(f"You are: {human_player}")
        print("Enter moves as 'row,col' (e.g., '11,12')")
        print("Commands: 'q' to quit, 'u' to undo, 'm' to show legal moves")
    print()

    move_history = []

    while True:
        print_board(state)

        winner = check_winner(state)
        if winner:
            print(f"\n*** {winner.upper()} WINS! ***\n")
            break

        moves = generate_moves(state)
        if not moves:
            print("\nNo legal moves - game is a draw.\n")
            break

        current = state.to_move
        is_ai_turn = args.ai_vs_ai or current != human_player

        if is_ai_turn:
            print(f"\nAI ({current}) is thinking...", end="", flush=True)
            result = choose_move(state, depth=ai_depth)
            print(f" plays ({result.row}, {result.col})")

            move_history.append(state)
            state = apply_move(state, result.row, result.col)

            if args.ai_vs_ai:
                import time
                time.sleep(0.3)  # Pause so human can follow
        else:
            print(f"\nYour move ({current}): ", end="")
            try:
                inp = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nGame ended.")
                break

            if inp == 'q':
                print("Game ended.")
                break
            elif inp == 'u':
                if move_history:
                    state = move_history.pop()
                    if move_history:  # Undo AI move too
                        state = move_history.pop()
                    print("Move undone.")
                else:
                    print("Nothing to undo.")
                continue
            elif inp == 'm':
                print(f"Legal moves ({len(moves)}): ", end="")
                for i, (r, c) in enumerate(moves[:20]):
                    print(f"({r},{c})", end=" ")
                if len(moves) > 20:
                    print(f"... and {len(moves) - 20} more")
                else:
                    print()
                continue
            else:
                try:
                    parts = inp.replace(" ", ",").split(",")
                    row, col = int(parts[0]), int(parts[1])
                    if (row, col) not in moves:
                        print(f"Invalid move. Legal moves include: {moves[:5]}...")
                        continue
                    move_history.append(state)
                    state = apply_move(state, row, col)
                except (ValueError, IndexError):
                    print("Invalid input. Enter row,col (e.g., '11,12')")
                    continue

    print("Final position:")
    print_board(state)


def cmd_fuzz(args: argparse.Namespace) -> None:
    """Run random/greedy games to fuzz test game rules."""
    from collections import Counter
    import time

    games = args.games
    mode = args.mode
    verbose = args.verbose

    print(f"Running {games} {mode} fuzz games...")
    start = time.time()

    outcomes = Counter()
    anomalies = []
    wins = []

    for seed in range(games):
        if mode == "random":
            result = play_random_game(seed=seed, max_moves=args.max_moves, stall_limit=args.stall_limit)
        else:
            result = play_greedy_game(seed=seed, max_moves=args.max_moves, stall_limit=args.stall_limit)

        outcomes[result["reason"]] += 1

        if result["reason"] == "win":
            wins.append(result)

        # Check for anomalies (unexpected terminations)
        # Note: stall games can exceed 200 moves if progress is intermittent, which is normal
        # True anomalies would be: unexpected reasons, impossible outcomes
        if result["reason"] not in ("win", "max_moves", "stall", "no_moves"):
            anomalies.append(result)

        if verbose and (seed + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f"  {seed + 1}/{games} games ({elapsed:.1f}s)")

    elapsed = time.time() - start
    print()
    print(f"Completed in {elapsed:.1f}s ({games / elapsed:.1f} games/sec)")
    print(f"Outcomes: {dict(outcomes)}")
    print(f"Wins: {len(wins)}")
    print(f"Anomalies (unexpected terminations): {len(anomalies)}")

    if wins and verbose:
        print("\nWinning games:")
        for r in wins[:10]:
            print(f"  seed={r['seed']}: {r['winner']} in {r['total_moves']} moves")

    if anomalies:
        print("\nAnomalies:")
        for r in anomalies[:10]:
            print(f"  seed={r['seed']}: {r['reason']} in {r['total_moves']} moves")

    # Report pass/fail
    if anomalies:
        print("\nFUZZ TEST FAILED: Found anomalies")
        return 1
    else:
        print("\nFUZZ TEST PASSED: No anomalies")
        return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="twixt-gpu")
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--search", default=None)
    p.add_argument("--board", type=int, default=24)
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel workers for game execution (0=auto, based on CPU cores)")

    sp = p.add_subparsers(dest="cmd", required=True)

    p_init = sp.add_parser("init")
    p_init.set_defaults(func=cmd_init)

    p_sweep = sp.add_parser("sweep")
    p_sweep.add_argument("--depths", default="2,3")
    p_sweep.add_argument("--games", type=int, default=10)
    p_sweep.add_argument("--total", type=int, default=24)
    p_sweep.add_argument("--fixed", type=int, default=6)
    p_sweep.add_argument("--mutate", type=int, default=10)
    p_sweep.add_argument("--seed", type=int, default=None)
    p_sweep.add_argument("--pred_gate", action="store_true")
    p_sweep.add_argument("--max_pred_bias", type=float, default=0.08)
    p_sweep.add_argument("--min_r2", type=float, default=0.20)
    p_sweep.set_defaults(func=cmd_sweep)

    p_rank = sp.add_parser("rank")
    p_rank.add_argument("--depths", default="2,3")
    p_rank.add_argument("--top", type=int, default=8)
    p_rank.set_defaults(func=cmd_rank)

    p_val = sp.add_parser("validate")
    p_val.add_argument("--depths", default="2,3")
    p_val.add_argument("--games", type=int, default=60, help="Games per depth (JS uses 60)")
    p_val.add_argument("--pass_score", type=float, default=0.02, help="Max bias per depth to pass (2 percent)")
    p_val.add_argument("--streak_needed", type=int, default=4, help="Consecutive passes for stable (JS uses 4)")
    p_val.add_argument("--retire_after", type=int, default=3, help="Retire after N failed validations")
    p_val.add_argument("--write_search", default=None)
    p_val.set_defaults(func=cmd_validate)

    p_tune = sp.add_parser("tune")
    p_tune.add_argument("--cycles", type=int, default=3)
    # Sweep settings
    p_tune.add_argument("--depths", default="2,3")
    p_tune.add_argument("--sweep_games", type=int, default=10, help="Games per depth for sweeps")
    p_tune.add_argument("--total", type=int, default=24, help="Total configs per sweep cycle")
    p_tune.add_argument("--fixed", type=int, default=6)
    p_tune.add_argument("--mutate", type=int, default=10)
    p_tune.add_argument("--seed", type=int, default=None)
    p_tune.add_argument("--pred_gate", action="store_true")
    p_tune.add_argument("--max_pred_bias", type=float, default=0.08)
    p_tune.add_argument("--min_r2", type=float, default=0.20)
    # Ranking settings
    p_tune.add_argument("--top", type=int, default=8)
    # Validation settings (longer runs, stricter requirements)
    p_tune.add_argument("--val_games", type=int, default=60, help="Games per depth for validation")
    p_tune.add_argument("--pass_score", type=float, default=0.02, help="Max bias per depth (2 percent)")
    p_tune.add_argument("--streak_needed", type=int, default=4, help="Consecutive passes for stable")
    p_tune.add_argument("--retire_after", type=int, default=3)
    p_tune.add_argument("--write_search", default=None)
    p_tune.set_defaults(func=cmd_tune)

    p_rep = sp.add_parser("replay")
    p_rep.add_argument("path")
    p_rep.set_defaults(func=cmd_replay)

    p_status = sp.add_parser("status")
    p_status.set_defaults(func=cmd_status)

    p_play = sp.add_parser("play", help="Interactive play mode to test game logic")
    p_play.add_argument("--ai-vs-ai", action="store_true", help="Watch AI vs AI game")
    p_play.add_argument("--human", choices=["red", "black"], default="black", help="Your color")
    p_play.add_argument("--depth", type=int, default=2, help="AI search depth")
    p_play.set_defaults(func=cmd_play)

    p_fuzz = sp.add_parser("fuzz", help="Run random/greedy games to fuzz test game rules")
    p_fuzz.add_argument("--games", type=int, default=1000, help="Number of games to run")
    p_fuzz.add_argument("--mode", choices=["random", "greedy"], default="random", help="Policy type")
    p_fuzz.add_argument("--max_moves", type=int, default=220, help="Max moves before draw")
    p_fuzz.add_argument("--stall_limit", type=int, default=40, help="Stall detection limit")
    p_fuzz.add_argument("--verbose", action="store_true", help="Show progress")
    p_fuzz.set_defaults(func=cmd_fuzz)

    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
