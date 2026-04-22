"""Tests for the inline forced-probe + connectivity-aware sanity additions.

Covers:
- _classify_position_from_tensor bucket logic on synthetic tensors
- summarize_value_sanity returns sanity_by_connectivity dict
- run_forced_probes_inline matches the CLI output on a tiny fixture
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from scripts.GPU.alphazero.game.twixt_state import TwixtState, NUM_CHANNELS
from scripts.GPU.alphazero.trainer import (
    _classify_position_from_tensor,
    summarize_value_sanity,
)
from scripts.GPU.alphazero.probe_eval import run_forced_probes_inline
from scripts.GPU.alphazero.network import create_network


# ---------- _classify_position_from_tensor ----------

def _make_nhwc_tensor(channels: int = 30, h: int = 24, w: int = 24) -> np.ndarray:
    """Build an empty NHWC tensor of the given channel depth."""
    return np.zeros((h, w, channels), dtype=np.float32)


def test_classify_pre_phase2_24ch_returns_unknown():
    """24-channel tensors (no connectivity masks) yield 'unknown'."""
    t = _make_nhwc_tensor(channels=24)
    assert _classify_position_from_tensor(t) == "unknown"


def test_classify_empty_30ch_returns_no_winning_structure():
    """All-zero 30-channel tensor → no_winning_structure."""
    t = _make_nhwc_tensor(channels=30)
    assert _classify_position_from_tensor(t) == "no_winning_structure"


def test_classify_red_top_only_below_threshold_no_winning():
    """Red has fewer than threshold pegs touching only top → no_winning_structure."""
    t = _make_nhwc_tensor(channels=30)
    # 3 pegs touching top (below threshold 8)
    for c in range(3):
        t[0, c, 24] = 1.0
    assert _classify_position_from_tensor(t) == "no_winning_structure"


def test_classify_red_top_at_threshold_is_winning():
    """Red has 8 pegs touching top → winning_structure."""
    t = _make_nhwc_tensor(channels=30)
    for c in range(8):
        t[0, c, 24] = 1.0
    assert _classify_position_from_tensor(t) == "winning_structure"


def test_classify_red_touches_both_edges_is_winning():
    """Red has pegs touching top AND bottom (2+ goal-touching components)
    → winning_structure even if both counts are below the size threshold."""
    t = _make_nhwc_tensor(channels=30)
    t[0, 5, 24] = 1.0   # one peg touching top
    t[23, 10, 25] = 1.0 # one peg touching bottom
    assert _classify_position_from_tensor(t) == "winning_structure"


def test_classify_black_left_at_threshold_is_winning():
    """Black has 8 pegs touching left → winning_structure."""
    t = _make_nhwc_tensor(channels=30)
    for r in range(8):
        t[r, 0, 27] = 1.0
    assert _classify_position_from_tensor(t) == "winning_structure"


def test_classify_real_terminal_state_is_winning():
    """Real terminal red-win state → winning_structure (channel 26 nonzero)."""
    state = TwixtState(active_size=8, to_move="red")
    moves = [(0, 3), (4, 0), (2, 4), (4, 1), (4, 3), (4, 5),
             (6, 4), (4, 6), (7, 2)]
    for r, c in moves:
        state = state.apply_move((r, c))
    if not state.is_terminal() or state.winner() is None:
        pytest.skip("scripted sequence did not produce a winner")
    chw = state.to_tensor()
    nhwc = np.transpose(chw, (1, 2, 0))
    assert _classify_position_from_tensor(nhwc) == "winning_structure"


# ---------- summarize_value_sanity returns sanity_by_connectivity ----------

class _FakePositionRecord:
    """Minimal PositionRecord-shaped object for sanity tests.

    Avoids importing self_play to keep the test isolated (self_play imports
    are heavyweight). Mirrors only the attrs summarize_value_sanity reads.
    """

    def __init__(self, board_tensor, to_move, legal_moves, visit_counts, outcome,
                 active_size=24, ply=0, game_n_moves=10):
        self.board_tensor = board_tensor
        self.to_move = to_move
        self.legal_moves = legal_moves
        self.visit_counts = visit_counts
        self.outcome = outcome
        self.active_size = active_size
        self.ply = ply
        self.game_n_moves = game_n_moves


def test_summarize_value_sanity_includes_connectivity_dict():
    """summarize_value_sanity returns a sanity_by_connectivity sub-dict."""
    # Use a real TwixtState so the board_tensor is well-formed
    state = TwixtState(active_size=8)
    chw = state.to_tensor()
    nhwc = np.transpose(chw, (1, 2, 0))
    pos = _FakePositionRecord(
        board_tensor=nhwc,
        to_move="red",
        legal_moves=state.legal_moves(),
        visit_counts=[1] * len(state.legal_moves()),
        outcome=1.0,
        active_size=8,
    )
    network = create_network(in_channels=NUM_CHANNELS)
    out = summarize_value_sanity(network, [pos], active_size=8, sample_n=1, seed=0)
    assert "sanity_by_connectivity" in out
    sbc = out["sanity_by_connectivity"]
    assert "winning_structure" in sbc
    assert "no_winning_structure" in sbc
    # Empty board → no_winning_structure
    assert sbc["no_winning_structure"]["n"] == 1
    assert sbc["winning_structure"]["n"] == 0


# ---------- run_forced_probes_inline ----------

def test_run_forced_probes_inline_empty_returns_zero():
    """Empty probe list returns n=0 and skipped=0."""
    network = create_network(in_channels=NUM_CHANNELS)
    out = run_forced_probes_inline(network, [], active_size=24)
    assert out["n"] == 0
    assert out["sign_correct"] == 0
    assert out["sign_correct_pct"] is None
    assert out["median_abs_v"] is None


def test_run_forced_probes_inline_filters_by_active_size():
    """Probes whose active_size doesn't match are skipped, counted in n_skipped_size."""
    network = create_network(in_channels=NUM_CHANNELS)
    # Manufacture a tiny 8x8 probe; the trainer is currently on 24 → should skip
    probe = {
        "id": "synthetic-001",
        "category": "near_win_red",
        "confidence": "forced",
        "side_to_move": "red",
        "expected_value_sign": 1,
        "expected_value_min": 0.5,
        "expected_value_max": None,
        "active_size": 8,
        "ply": 0,
        "move_history": [],
    }
    out = run_forced_probes_inline(network, [probe], active_size=24)
    assert out["n"] == 0
    assert out["n_skipped_size"] == 1


def test_run_forced_probes_inline_evaluates_matching_probe():
    """A probe matching active_size is evaluated; nn_values list is populated."""
    network = create_network(in_channels=NUM_CHANNELS)
    probe = {
        "id": "synthetic-002",
        "category": "near_win_red",
        "confidence": "forced",
        "side_to_move": "red",
        "expected_value_sign": 1,
        "expected_value_min": None,
        "expected_value_max": None,
        "active_size": 8,
        "ply": 0,
        "move_history": [],  # empty board
    }
    out = run_forced_probes_inline(network, [probe], active_size=8)
    assert out["n"] == 1
    assert len(out["nn_values"]) == 1
    assert out["expected_signs"] == [1]
    # sign_correct may be 0 or 1 depending on random init; just verify shape
    assert out["sign_correct"] in (0, 1)
    assert out["sign_correct_pct"] in (0.0, 1.0)
    assert out["median_abs_v"] is not None


# ---------- load_network_for_scoring (public wrapper over _load_network) ----------

def test_load_network_for_scoring_public_symbol_exists():
    """The public wrapper is importable under the expected name."""
    from scripts.GPU.alphazero.probe_eval import load_network_for_scoring
    assert callable(load_network_for_scoring)


def test_load_network_for_scoring_matches_private_loader(tmp_path):
    """Public wrapper delegates to _load_network and returns the same shape."""
    from scripts.GPU.alphazero.probe_eval import (
        load_network_for_scoring, _load_network,
    )
    # Fixture uses create_network defaults so it matches the wrapper's
    # no-override load path. _load_network only auto-detects in_channels
    # (24 vs 30); hidden and n_blocks fall through to create_network
    # defaults when the caller passes None.
    from scripts.GPU.alphazero.network import create_network
    net = create_network(in_channels=30)
    weights_path = tmp_path / "fixture.safetensors"
    net.save_weights(str(weights_path))

    pub = load_network_for_scoring(str(weights_path), verbose=False)
    priv = _load_network(str(weights_path), verbose=False)
    # Both return 4-tuples: (net, in_channels, hidden, n_blocks)
    assert len(pub) == 4
    assert pub[1] == priv[1] == 30   # in_channels
    assert pub[2] == priv[2]         # hidden
    assert pub[3] == priv[3]         # n_blocks


# ---------- extract_forced_probes_from_games ----------

def _make_game_dict(iteration=29, game_idx=0, n_moves=40, winner="red",
                   reason="win", board_size=24, moves=None):
    """Minimal parsed-game-JSON shape matching what load_replays produces.

    Moves list: [{"player": "red"/"black", "move": [r, c]}, ...]
    """
    if moves is None:
        # Generate n_moves of alternating moves at arbitrary (but legal-looking) cells.
        moves = []
        for i in range(n_moves):
            player = "red" if i % 2 == 0 else "black"
            moves.append({"player": player, "move": [(i * 3) % board_size, (i * 5) % board_size]})
    return {
        "id": f"iter_{iteration:04d}_game_{game_idx:03d}",
        "meta": {
            "board_size": board_size,
            "iteration": iteration,
            "game_idx": game_idx,
            "reason": reason,
            "n_moves": n_moves,
            "starting_player": "red",
        },
        "moves": moves,
        "winner": winner,
        "starting_player": "red",
    }


def test_extract_forced_probes_basic_two_per_game():
    """A single natural-win game at size 24 yields 2 probes (K=2: plies n_moves-1 and n_moves-2)."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    game = _make_game_dict(iteration=29, game_idx=0, n_moves=40, winner="red")
    probes = extract_forced_probes_from_games([game], active_size=24, k_plies=2)
    assert len(probes) == 2
    plies = sorted(p["ply"] for p in probes)
    assert plies == [38, 39]  # n_moves-2 and n_moves-1


def test_extract_forced_probes_category_from_winner_not_side_to_move():
    """Category is based on eventual winner, independent of side_to_move."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    # Red wins, 40 moves. At ply 38 (even index after starting red), side-to-move
    # alternates. Category must be 'near_win_red' for BOTH probes regardless of stm.
    game = _make_game_dict(iteration=29, n_moves=40, winner="red")
    probes = extract_forced_probes_from_games([game], active_size=24)
    assert all(p["category"] == "near_win_red" for p in probes)
    # Now black wins: both probes are near_win_black.
    game_b = _make_game_dict(iteration=29, n_moves=40, winner="black")
    probes_b = extract_forced_probes_from_games([game_b], active_size=24)
    assert all(p["category"] == "near_win_black" for p in probes_b)


def test_extract_forced_probes_deterministic_ids():
    """IDs have the form {basename}_ply{ply:03d}_{winner} and are stable across reruns."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    game = _make_game_dict(iteration=29, game_idx=42, n_moves=40, winner="red")
    probes_1 = extract_forced_probes_from_games([game], active_size=24)
    probes_2 = extract_forced_probes_from_games([game], active_size=24)
    ids_1 = sorted(p["id"] for p in probes_1)
    ids_2 = sorted(p["id"] for p in probes_2)
    assert ids_1 == ids_2
    assert "iter_0029_game_042_ply038_red" in ids_1
    assert "iter_0029_game_042_ply039_red" in ids_1


def test_extract_forced_probes_confidence_field_forced():
    """Every emitted probe has confidence='forced'."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    game = _make_game_dict(iteration=29, n_moves=40, winner="red")
    probes = extract_forced_probes_from_games([game], active_size=24)
    assert all(p["confidence"] == "forced" for p in probes)


def test_extract_forced_probes_natural_wins_only():
    """Resigns, adjudicated games, draws, and timeouts produce zero probes."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    for bad_reason in ("resign", "adjudicated", "timeout", "board_full", "state_cap", "unknown"):
        game = _make_game_dict(iteration=29, n_moves=40, winner="red", reason=bad_reason)
        probes = extract_forced_probes_from_games([game], active_size=24)
        assert probes == [], f"reason={bad_reason!r} should yield zero probes"


def test_extract_forced_probes_active_size_filter():
    """Games at wrong board size produce zero probes."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    game_16 = _make_game_dict(iteration=29, n_moves=40, winner="red", board_size=16)
    assert extract_forced_probes_from_games([game_16], active_size=24) == []
    # But size 24 games still work.
    game_24 = _make_game_dict(iteration=29, n_moves=40, winner="red", board_size=24)
    assert len(extract_forced_probes_from_games([game_24], active_size=24)) == 2


def test_extract_forced_probes_invalid_winner_skipped():
    """Games with winner None, 'draw', or any unexpected value are skipped."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    for bad_winner in (None, "draw", "", "unknown"):
        game = _make_game_dict(iteration=29, n_moves=40, winner=bad_winner)
        probes = extract_forced_probes_from_games([game], active_size=24)
        assert probes == [], f"winner={bad_winner!r} should yield zero probes"


def test_extract_forced_probes_exact_dedupe():
    """Two games with byte-identical move_history yield probes dedup'd on exact match."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    # Same moves, different game id — the move_history is what gets dedup'd.
    moves = [{"player": "red" if i % 2 == 0 else "black", "move": [i % 24, (i * 3) % 24]}
             for i in range(40)]
    g1 = _make_game_dict(iteration=29, game_idx=0, n_moves=40, winner="red", moves=moves)
    g2 = _make_game_dict(iteration=29, game_idx=1, n_moves=40, winner="red", moves=moves)
    probes = extract_forced_probes_from_games([g1, g2], active_size=24,
                                              dedupe_exact=True, dedupe_mirror=False)
    # Only 2 survive (one copy of each ply), not 4.
    assert len(probes) == 2


def test_extract_forced_probes_mirror_dedupe_horizontal():
    """A game and its horizontal mirror (c → N-1-c) collapse to one copy."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    N = 24
    moves = [{"player": "red" if i % 2 == 0 else "black", "move": [i % N, (i * 3) % N]}
             for i in range(40)]
    mirrored = [{"player": m["player"], "move": [m["move"][0], N - 1 - m["move"][1]]}
                for m in moves]
    g1 = _make_game_dict(iteration=29, game_idx=0, n_moves=40, winner="red", moves=moves)
    g2 = _make_game_dict(iteration=29, game_idx=1, n_moves=40, winner="red", moves=mirrored)
    probes = extract_forced_probes_from_games([g1, g2], active_size=24,
                                              dedupe_exact=True, dedupe_mirror=True)
    assert len(probes) == 2, f"horizontal mirror should collapse, got {len(probes)} probes"


def test_extract_forced_probes_mirror_dedupe_vertical():
    """Vertical mirror (r → N-1-r) also collapses."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    N = 24
    moves = [{"player": "red" if i % 2 == 0 else "black", "move": [i % N, (i * 3) % N]}
             for i in range(40)]
    mirrored = [{"player": m["player"], "move": [N - 1 - m["move"][0], m["move"][1]]}
                for m in moves]
    g1 = _make_game_dict(iteration=29, game_idx=0, n_moves=40, winner="red", moves=moves)
    g2 = _make_game_dict(iteration=29, game_idx=1, n_moves=40, winner="red", moves=mirrored)
    probes = extract_forced_probes_from_games([g1, g2], active_size=24, dedupe_mirror=True)
    assert len(probes) == 2


def test_extract_forced_probes_mirror_dedupe_180_rotation():
    """180° rotation (r → N-1-r AND c → N-1-c) also collapses."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    N = 24
    moves = [{"player": "red" if i % 2 == 0 else "black", "move": [i % N, (i * 3) % N]}
             for i in range(40)]
    mirrored = [{"player": m["player"], "move": [N - 1 - m["move"][0], N - 1 - m["move"][1]]}
                for m in moves]
    g1 = _make_game_dict(iteration=29, game_idx=0, n_moves=40, winner="red", moves=moves)
    g2 = _make_game_dict(iteration=29, game_idx=1, n_moves=40, winner="red", moves=mirrored)
    probes = extract_forced_probes_from_games([g1, g2], active_size=24, dedupe_mirror=True)
    assert len(probes) == 2


def test_extract_forced_probes_transpose_NOT_deduped():
    """Transpose (r,c)→(c,r) swaps red/black goals and must NOT dedup."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    N = 24
    moves = [{"player": "red" if i % 2 == 0 else "black", "move": [i % N, (i * 3) % N]}
             for i in range(40)]
    transposed = [{"player": m["player"], "move": [m["move"][1], m["move"][0]]}
                  for m in moves]
    g1 = _make_game_dict(iteration=29, game_idx=0, n_moves=40, winner="red", moves=moves)
    g2 = _make_game_dict(iteration=29, game_idx=1, n_moves=40, winner="red", moves=transposed)
    probes = extract_forced_probes_from_games([g1, g2], active_size=24, dedupe_mirror=True)
    assert len(probes) == 4, "transpose must not dedupe — goals differ"


def test_extract_forced_probes_sort_order():
    """Sort: source_iteration desc, source_ply desc, source_game basename asc."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    # Mix two iterations; verify iter 30 comes first, then within iter, later ply first.
    g_old = _make_game_dict(iteration=25, game_idx=0, n_moves=40, winner="red")
    g_new = _make_game_dict(iteration=30, game_idx=0, n_moves=40, winner="red")
    probes = extract_forced_probes_from_games([g_old, g_new], active_size=24,
                                              dedupe_exact=False, dedupe_mirror=False)
    # Expected order: (iter=30, ply=39), (iter=30, ply=38), (iter=25, ply=39), (iter=25, ply=38)
    assert probes[0]["source_game"].startswith("iter_0030")
    assert probes[0]["source_ply"] == 39
    assert probes[1]["source_game"].startswith("iter_0030")
    assert probes[1]["source_ply"] == 38
    assert probes[2]["source_game"].startswith("iter_0025")
    assert probes[2]["source_ply"] == 39
    assert probes[3]["source_game"].startswith("iter_0025")
    assert probes[3]["source_ply"] == 38


def test_extract_forced_probes_max_probes_truncation():
    """max_probes=N truncates after sorting."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    g_old = _make_game_dict(iteration=25, game_idx=0, n_moves=40, winner="red")
    g_new = _make_game_dict(iteration=30, game_idx=0, n_moves=40, winner="red")
    probes = extract_forced_probes_from_games([g_old, g_new], active_size=24,
                                              dedupe_exact=False, dedupe_mirror=False,
                                              max_probes=2)
    assert len(probes) == 2
    # Should keep the two most-recent-iter probes.
    assert all(p["source_game"].startswith("iter_0030") for p in probes)


def test_extract_forced_probes_max_probes_none_returns_all_sorted():
    """max_probes=None returns all probes, still in sorted order."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    g_old = _make_game_dict(iteration=25, game_idx=0, n_moves=40, winner="red")
    g_new = _make_game_dict(iteration=30, game_idx=0, n_moves=40, winner="red")
    probes = extract_forced_probes_from_games([g_old, g_new], active_size=24,
                                              dedupe_exact=False, dedupe_mirror=False,
                                              max_probes=None)
    assert len(probes) == 4
    # Still sorted: first probe is from iter 30.
    assert probes[0]["source_game"].startswith("iter_0030")


def test_extract_forced_probes_emits_starting_player():
    """Every probe carries a starting_player field. _replay_probe needs it to
    initialize TwixtState.to_move correctly for black-started games."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games
    # A black-started game.
    moves = []
    for i in range(40):
        # Alternate turns: black, red, black, red, ...
        player = "black" if i % 2 == 0 else "red"
        moves.append({"player": player, "move": [(i * 3) % 24, (i * 5) % 24]})
    game = {
        "id": "iter_0029_game_000",
        "meta": {"board_size": 24, "iteration": 29, "game_idx": 0,
                 "reason": "win", "n_moves": 40, "starting_player": "black"},
        "moves": moves,
        "winner": "black",
        "starting_player": "black",
    }
    probes = extract_forced_probes_from_games([game], active_size=24)
    assert len(probes) == 2
    for p in probes:
        assert p["starting_player"] == "black"


def test_replay_probe_honors_starting_player():
    """_replay_probe must read starting_player and initialize state.to_move
    accordingly. Without this, a black-started probe's move_history replay
    crashes with 'Illegal move' because red-by-default can't play black's
    goal-edge cells."""
    from scripts.GPU.alphazero.probe_eval import _replay_probe
    # Construct a probe mimicking a black-started game with a move red couldn't
    # legally make (col 0 is black's goal edge — red can't play there).
    probe = {
        "id": "synthetic_blackstart_probe",
        "active_size": 24,
        "starting_player": "black",
        # ply 0: black plays col 0 (its own goal edge — legal for black,
        # illegal for red). If _replay_probe honors starting_player this
        # replays cleanly.
        "move_history": [[5, 0]],
    }
    state = _replay_probe(probe)
    # After 1 move from black-starts, it's red's turn.
    assert state.to_move == "red"


def test_replay_probe_raises_on_illegal_move_with_probe_id():
    """If replay fails, the error message includes the probe id and ply
    index — makes data-integrity bugs obvious rather than silent."""
    from scripts.GPU.alphazero.probe_eval import _replay_probe
    probe = {
        "id": "synthetic_bad_probe",
        "active_size": 24,
        "starting_player": "red",
        # Illegal for red: col 0 is black's goal edge.
        "move_history": [[5, 0]],
    }
    with pytest.raises(ValueError, match="synthetic_bad_probe.*ply 0.*starting_player='red'"):
        _replay_probe(probe)


def test_extract_forced_probes_accepts_row_col_schema():
    """Real game JSONs use the canonical 'row'/'col' keys (per
    scripts/GPU/replay/format.py). The legacy synthetic fixture schema
    'move: [r, c]' must also continue to work for tests. Both schemas
    must produce identical probe move_histories for equivalent moves."""
    from scripts.GPU.alphazero.probe_eval import extract_forced_probes_from_games

    # Build the same move sequence in both schemas.
    row_col_moves = []
    move_list_moves = []
    for i in range(40):
        r = (i * 3) % 24
        c = (i * 5) % 24
        row_col_moves.append({
            "player": "red" if i % 2 == 0 else "black",
            "row": r, "col": c,
        })
        move_list_moves.append({
            "player": "red" if i % 2 == 0 else "black",
            "move": [r, c],
        })

    base_game = {
        "meta": {"board_size": 24, "iteration": 29, "game_idx": 0,
                 "reason": "win", "n_moves": 40, "starting_player": "red"},
        "winner": "red",
        "starting_player": "red",
    }
    game_rc = {**base_game, "id": "iter_0029_game_000",  "moves": row_col_moves}
    game_ml = {**base_game, "id": "iter_0029_game_000",  "moves": move_list_moves}

    probes_rc = extract_forced_probes_from_games([game_rc], active_size=24)
    probes_ml = extract_forced_probes_from_games([game_ml], active_size=24)

    assert len(probes_rc) == 2
    assert len(probes_ml) == 2
    # Move histories must be byte-identical between schemas.
    for p_rc, p_ml in zip(probes_rc, probes_ml):
        assert p_rc["move_history"] == p_ml["move_history"], (
            f"schema divergence at ply {p_rc['ply']}: "
            f"row/col → {p_rc['move_history'][:3]}..., "
            f"move[] → {p_ml['move_history'][:3]}..."
        )
