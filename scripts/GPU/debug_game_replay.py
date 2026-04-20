#!/usr/bin/env python3
"""Debug script to replay a saved game and verify win detection."""
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.game.twixt_state import TwixtState


def replay_game(game_file: str, actual_start_player: str = None):
    """Replay a game and check win detection at each step."""
    with open(game_file) as f:
        data = json.load(f)

    moves = data["moves"]
    saved_winner = data["winner"]
    board_size = data["meta"].get("board_size", 24)

    print(f"Game: {data['id']}")
    print(f"Saved winner: {saved_winner}")
    print(f"Board size: {board_size}")
    print(f"Total moves: {len(moves)}")
    print()

    # If actual_start_player provided, swap colors accordingly
    if actual_start_player:
        print(f"Assuming actual start player: {actual_start_player}")
        if actual_start_player == "black":
            # Saved "red" moves were actually black, saved "black" were actually red
            color_map = {"red": "black", "black": "red"}
        else:
            color_map = {"red": "red", "black": "black"}
    else:
        color_map = {"red": "red", "black": "black"}

    # Extract just (row, col) and determine actual player
    move_sequence = []
    for m in moves:
        saved_player = m["player"]
        actual_player = color_map[saved_player]
        move_sequence.append((m["row"], m["col"], actual_player, saved_player))

    # Replay through TwixtState
    state = TwixtState(active_size=board_size, to_move=actual_start_player or "red")

    print("Replaying moves...")
    for i, (row, col, actual_player, saved_player) in enumerate(move_sequence):
        turn = i + 1

        # Check if move is legal
        if not state.is_valid_placement(row, col):
            print(f"  Turn {turn}: ILLEGAL move ({row}, {col}) for {state.to_move}")
            print(f"    Saved as: {saved_player}, Actual: {actual_player}")
            print(f"    State to_move: {state.to_move}")
            return

        # Verify state.to_move matches expected player
        if state.to_move != actual_player:
            print(f"  Turn {turn}: Player mismatch! state.to_move={state.to_move}, expected={actual_player}")

        # Apply move
        state = state.apply_move((row, col))

        # Check for win
        winner = state.winner()
        if winner:
            print(f"  Turn {turn}: {actual_player} played ({row}, {col}) -> {winner} WINS!")
            print(f"    (Saved as {saved_player})")

            # Show the connected component
            print(f"\n  Final position pegs:")
            red_pegs = [(r, c) for (r, c), p in state.pegs.items() if p == "red"]
            black_pegs = [(r, c) for (r, c), p in state.pegs.items() if p == "black"]
            print(f"    Red pegs: {len(red_pegs)}")
            print(f"    Black pegs: {len(black_pegs)}")

            # Check edges
            red_top = [p for p in red_pegs if p[0] == 0]
            red_bottom = [p for p in red_pegs if p[0] == board_size - 1]
            black_left = [p for p in black_pegs if p[1] == 0]
            black_right = [p for p in black_pegs if p[1] == board_size - 1]

            print(f"    Red on top edge (row 0): {red_top}")
            print(f"    Red on bottom edge (row {board_size-1}): {red_bottom}")
            print(f"    Black on left edge (col 0): {black_left}")
            print(f"    Black on right edge (col {board_size-1}): {black_right}")

            print(f"\n  Bridges: {len(state.bridges)}")

            # Trace the winning path for black
            if winner == "black":
                print(f"\n  Tracing black's winning path from col 0 to col {board_size-1}:")
                for start_pos in black_left:
                    component = state._get_connected_component(start_pos, "black")
                    right_edge_in_component = [p for p in component if p[1] == board_size - 1]
                    if right_edge_in_component:
                        print(f"    Starting from {start_pos}:")
                        print(f"    Component size: {len(component)}")
                        print(f"    Component pegs (sorted by col): {sorted(component, key=lambda p: (p[1], p[0]))}")
                        print(f"    Reaches right edge at: {right_edge_in_component}")
                        break

            # Trace the winning path for red
            if winner == "red":
                print(f"\n  Tracing red's winning path from row 0 to row {board_size-1}:")
                for start_pos in red_top:
                    component = state._get_connected_component(start_pos, "red")
                    bottom_edge_in_component = [p for p in component if p[0] == board_size - 1]
                    if bottom_edge_in_component:
                        print(f"    Starting from {start_pos}:")
                        print(f"    Component size: {len(component)}")
                        print(f"    Component pegs (sorted by row): {sorted(component, key=lambda p: (p[0], p[1]))}")
                        print(f"    Reaches bottom edge at: {bottom_edge_in_component}")
                        break
            return

        if state.is_terminal():
            print(f"  Turn {turn}: Game ended (no winner) - draw")
            return

    print(f"\nGame ended after {len(moves)} moves")
    print(f"Final state: terminal={state.is_terminal()}, winner={state.winner()}")

    # Show final position
    red_pegs = [(r, c) for (r, c), p in state.pegs.items() if p == "red"]
    black_pegs = [(r, c) for (r, c), p in state.pegs.items() if p == "black"]
    print(f"Red pegs ({len(red_pegs)}): on col 0: {[p for p in red_pegs if p[1] == 0]}")
    print(f"Black pegs ({len(black_pegs)}): on row 0: {[p for p in black_pegs if p[0] == 0]}")


if __name__ == "__main__":
    game_file = sys.argv[1] if len(sys.argv) > 1 else "scripts/GPU/logs/games/iter_0189_game_010.json"
    start_player = sys.argv[2] if len(sys.argv) > 2 else "black"  # Assume black started based on illegal move analysis

    replay_game(game_file, start_player)
