"""Tests for score_samples_against_checkpoint (stratified calibration sampling).

Covers:
- Stratified budget: per-bucket caps honored, stable alphabetical ordering
  when max_total binds
- natural_distribution reports counts across the full pool
- 24-channel checkpoint smoke: no crash, real buckets populated
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.GPU.alphazero.network import create_network


def _make_game_for_calibration(n_moves=40, winner="red", board_size=24):
    """Parsed-game JSON with enough moves for classify_position to produce
    a mix of buckets across plies.

    Generates a scatter pattern but skips positions that are illegal for the
    current player (corners / opponent-edge / duplicates).
    """
    from scripts.GPU.alphazero.game.twixt_state import TwixtState

    moves = []
    state = TwixtState(active_size=board_size)
    max_i = n_moves * 20  # safety: avoid infinite loop if board fills
    i = 0
    while len(moves) < n_moves and i < max_i:
        r = (i * 7) % board_size
        c = (i * 11) % board_size
        i += 1
        if not state.is_valid_placement(r, c):
            continue
        player = state.to_move
        moves.append({"player": player, "move": [r, c]})
        state = state.apply_move((r, c))
    return {
        "id": f"iter_0029_game_{n_moves:03d}",
        "meta": {"board_size": board_size, "iteration": 29, "reason": "win", "n_moves": len(moves)},
        "moves": moves,
        "winner": winner,
        "starting_player": "red",
    }


@pytest.fixture
def tiny_30ch_network(tmp_path):
    net = create_network(in_channels=30, hidden=8, n_blocks=1)
    path = tmp_path / "tiny_30ch.safetensors"
    net.save_weights(str(path))
    return net, str(path)


def test_score_samples_natural_distribution_reported(tiny_30ch_network):
    """natural_distribution counts every position in the full pool."""
    from scripts.GPU.alphazero.value_calibration import score_samples_against_checkpoint
    net, _ = tiny_30ch_network
    # 3 games × 40 moves → 120 positions total.
    replays = [_make_game_for_calibration(n_moves=40) for _ in range(3)]
    result = score_samples_against_checkpoint(
        replays, network=net, samples_per_bucket=5, max_total=2000
    )
    # Sum across buckets should equal total positions across all games.
    total_natural = sum(result["natural_distribution"].values())
    total_positions = sum(len(g["moves"]) for g in replays)
    assert total_natural == total_positions


def test_score_samples_stratified_per_bucket_caps(tiny_30ch_network):
    """For each bucket, sampled_count <= min(samples_per_bucket, natural_count)."""
    from scripts.GPU.alphazero.value_calibration import score_samples_against_checkpoint
    net, _ = tiny_30ch_network
    replays = [_make_game_for_calibration(n_moves=40) for _ in range(5)]
    result = score_samples_against_checkpoint(
        replays, network=net, samples_per_bucket=3, max_total=2000
    )
    for bucket, sampled in result["sampled_distribution"].items():
        natural = result["natural_distribution"][bucket]
        assert sampled <= min(3, natural), (
            f"bucket={bucket!r} sampled={sampled} exceeds min(cap=3, natural={natural})"
        )


def test_score_samples_stratified_flag_and_note(tiny_30ch_network):
    """Output advertises itself as stratified."""
    from scripts.GPU.alphazero.value_calibration import score_samples_against_checkpoint
    net, _ = tiny_30ch_network
    replays = [_make_game_for_calibration(n_moves=40) for _ in range(2)]
    result = score_samples_against_checkpoint(
        replays, network=net, samples_per_bucket=5, max_total=2000
    )
    assert result["stratified"] is True
    assert "stratified" in result["overall_note"].lower()
    assert "aggregate" in result  # carries the existing aggregate_calibration schema


def test_score_samples_max_total_binds_alphabetical_halt(tiny_30ch_network):
    """When max_total binds, later-alphabetical buckets get sampled=0."""
    from scripts.GPU.alphazero.value_calibration import score_samples_against_checkpoint
    net, _ = tiny_30ch_network
    # Enough games to populate multiple buckets heavily.
    replays = [_make_game_for_calibration(n_moves=40) for _ in range(20)]
    # Set max_total so it binds after ~2 buckets worth of samples.
    result = score_samples_against_checkpoint(
        replays, network=net, samples_per_bucket=200, max_total=50
    )
    total_sampled = sum(result["sampled_distribution"].values())
    assert total_sampled <= 50
    # At least one bucket should have sampled=0 given the tight cap.
    zero_buckets = [b for b, n in result["sampled_distribution"].items() if n == 0]
    # Note: if natural distribution is lopsided, zero_buckets could be empty only
    # if all 50 samples fit in the first alphabetical bucket. Assert the ordering
    # invariant: any zero-bucket must be alphabetically AFTER the last nonzero bucket.
    sampled_names_sorted = sorted(result["sampled_distribution"].keys())
    saw_nonzero = False
    saw_zero_after_nonzero = False
    for name in sampled_names_sorted:
        n = result["sampled_distribution"][name]
        if n > 0:
            if saw_zero_after_nonzero:
                pytest.fail(f"bucket ordering violated: {name!r} nonzero after a zero bucket")
            saw_nonzero = True
        elif saw_nonzero:
            saw_zero_after_nonzero = True


def test_score_samples_24ch_checkpoint_no_crash(tmp_path):
    """A 24-channel checkpoint still produces real bucket stats (structural
    classification is state-based, independent of network channel count)."""
    from scripts.GPU.alphazero.value_calibration import score_samples_against_checkpoint
    net_24 = create_network(in_channels=24, hidden=8, n_blocks=1)
    replays = [_make_game_for_calibration(n_moves=30) for _ in range(3)]
    result = score_samples_against_checkpoint(
        replays, network=net_24, samples_per_bucket=3, max_total=100
    )
    # Real buckets populate — not just 'unknown'.
    buckets = list(result["natural_distribution"].keys())
    assert any(b != "unknown" for b in buckets)


def test_score_samples_accepts_row_col_schema(tiny_30ch_network):
    """Real on-disk game JSONs use moves[i]['row'] + moves[i]['col'], not
    moves[i]['move'] = [r, c]. Both schemas must work identically — this
    regression guard catches the KeyError 'move' that surfaced in the
    live analyzer run against real replays."""
    from scripts.GPU.alphazero.value_calibration import score_samples_against_checkpoint
    from scripts.GPU.alphazero.game.twixt_state import TwixtState

    net, _ = tiny_30ch_network
    # Build a small legal-move game directly from a real TwixtState so the
    # replay is well-formed, and emit moves in the row/col canonical schema.
    state = TwixtState(active_size=24)
    row_col_moves = []
    n_moves = 20
    i = 0
    while len(row_col_moves) < n_moves and i < n_moves * 20:
        r = (i * 7) % 24
        c = (i * 11) % 24
        i += 1
        if state.is_valid_placement(r, c):
            row_col_moves.append({
                "player": state.to_move,
                "row": r, "col": c,
            })
            state = state.apply_move((r, c))

    game = {
        "id": "iter_0029_game_000",
        "meta": {"board_size": 24, "iteration": 29, "reason": "win",
                 "n_moves": len(row_col_moves)},
        "moves": row_col_moves,
        "winner": "red",
        "starting_player": "red",
    }
    result = score_samples_against_checkpoint(
        [game], network=net, samples_per_bucket=5, max_total=50
    )
    # Must succeed (no KeyError) and produce real buckets.
    assert sum(result["natural_distribution"].values()) == len(row_col_moves)
    assert any(b != "unknown" for b in result["natural_distribution"].keys())


def test_score_samples_honors_starting_player(tiny_30ch_network):
    """Black-started games reconstruct correctly (TwixtState to_move must be
    initialized from game.starting_player, matching _replay_probe)."""
    from scripts.GPU.alphazero.value_calibration import score_samples_against_checkpoint
    from scripts.GPU.alphazero.game.twixt_state import TwixtState

    net, _ = tiny_30ch_network
    # Build a black-started game. At ply 0 TwixtState must have to_move='black'
    # for the first move to be legal on a col=0 or col=23 cell (black's goals).
    state = TwixtState(active_size=24, to_move="black")
    moves = []
    # First black move: on black's goal col (0) — illegal for red-starts default.
    if state.is_valid_placement(5, 0):
        moves.append({"player": "black", "row": 5, "col": 0})
        state = state.apply_move((5, 0))
    # Fill out with arbitrary legal moves.
    i = 0
    while len(moves) < 10 and i < 200:
        r = (i * 3) % 24
        c = 5 + (i * 7) % 14
        i += 1
        if state.is_valid_placement(r, c):
            moves.append({"player": state.to_move, "row": r, "col": c})
            state = state.apply_move((r, c))

    game = {
        "id": "iter_0029_blackstart_000",
        "meta": {"board_size": 24, "iteration": 29, "reason": "win",
                 "n_moves": len(moves)},
        "moves": moves,
        "winner": "black",
        "starting_player": "black",   # <-- the field score_samples must honor
    }
    # Without the starting_player fix, this replay crashes on black's col-0 move.
    result = score_samples_against_checkpoint(
        [game], network=net, samples_per_bucket=5, max_total=50
    )
    assert sum(result["natural_distribution"].values()) == len(moves)
