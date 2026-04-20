#!/usr/bin/env python3
"""Generate probe-suite candidates from historical game JSONs.

Reads game JSONs under --input, applies per-category heuristic rules using
the shared connectivity_masks helper, emits candidates.json with candidates
grouped by category.

Default source filter: active_size=24, iteration>=900 (current regime).
Override with --any-size / --min-source-iter / --source-iter-range.

The output is intended for user review -> curation -> commit as
tests/probes/twixt_probes.json.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

# Add project root to path for scripts import
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.GPU.alphazero.game.twixt_state import TwixtState


CATEGORIES = (
    "near_win_red", "near_win_black",
    "blocked_or_trap", "false_positive_connectivity",
    "dense_but_disconnected",
    "central_win", "edge_corner_legitimate", "symmetric_sanity",
)

# symmetric_sanity is generated after the primary loop from mirror pairs,
# so it is excluded from the early-termination check.
PRIMARY_CATEGORIES = tuple(c for c in CATEGORIES if c != "symmetric_sanity")

PER_CATEGORY_TARGET_DEFAULT = 20  # aim for ~20 candidates per category -> ~160 total before pruning


def _iter_filter(meta: dict, min_iter: int, max_iter: int | None) -> bool:
    it = meta.get("iteration", -1)
    if it < min_iter:
        return False
    if max_iter is not None and it > max_iter:
        return False
    return True


def _size_filter(meta: dict, any_size: bool) -> bool:
    if any_size:
        return True
    return meta.get("board_size") == 24


def _replay_state(move_history: list, active_size: int, start_player: str) -> TwixtState:
    """Replay move_history from a fresh state."""
    state = TwixtState(active_size=active_size, to_move=start_player)
    for (r, c) in move_history:
        state = state.apply_move((r, c))
    return state


def _classify_candidates(game: dict, game_path: str, categories_wanted: set) -> list[dict]:
    """Extract candidate positions from a single game. Applies heuristic rules
    per category; each candidate carries {category, move_history, ply, note}."""
    meta = game.get("meta") or {}
    moves = game.get("moves") or []
    if not moves:
        return []
    active = meta.get("board_size", 24)
    start_player = meta.get("starting_player") or game.get("starting_player", "red")
    winner = game.get("winner")
    move_seq = [(int(m["row"]), int(m["col"])) for m in moves]

    candidates: list[dict] = []

    # Replay stepwise and analyze each ply
    state = TwixtState(active_size=active, to_move=start_player)
    for ply, (r, c) in enumerate(move_seq):
        state = state.apply_move((r, c))
        # Analyze AFTER this move -- state reflects position after move `ply`
        is_terminal = state.is_terminal()
        plies_remaining = len(move_seq) - (ply + 1)

        # near_win_red / near_win_black: 1-3 plies before terminal with
        # winner having a goal-touching component
        if "near_win_red" in categories_wanted or "near_win_black" in categories_wanted:
            if winner in ("red", "black") and 1 <= plies_remaining <= 3 and not is_terminal:
                m_g1, m_g2, m_both = state.connectivity_masks(winner)
                if m_g1.sum() > 0 and m_g2.sum() > 0:
                    cat = f"near_win_{winner}"
                    if cat in categories_wanted:
                        candidates.append({
                            "category": cat,
                            "ply": ply + 1,
                            "move_history": move_seq[:ply + 1],
                            "side_to_move": state.to_move,
                            "active_size": active,
                            "source_game": game_path,
                            "source_ply": ply + 1,
                            "note": f"{winner} has goal-touching components on both sides, {plies_remaining} plies to win",
                        })

        # central_win: near-win positions where chain avoids the outer 2 rings
        if "central_win" in categories_wanted:
            if winner in ("red", "black") and 1 <= plies_remaining <= 3 and not is_terminal:
                m_g1, m_g2, _ = state.connectivity_masks(winner)
                # "Central" heuristic: winning-component's pegs mostly in interior
                pegs_in_component = set()
                for rr in range(active):
                    for cc in range(active):
                        if m_g1[rr, cc] > 0 or m_g2[rr, cc] > 0:
                            pegs_in_component.add((rr, cc))
                if pegs_in_component:
                    interior = sum(1 for (rr, cc) in pegs_in_component
                                   if 2 <= rr < active - 2 and 2 <= cc < active - 2)
                    if interior / len(pegs_in_component) >= 0.7:
                        candidates.append({
                            "category": "central_win",
                            "ply": ply + 1,
                            "move_history": move_seq[:ply + 1],
                            "side_to_move": state.to_move,
                            "active_size": active,
                            "source_game": game_path,
                            "source_ply": ply + 1,
                            "note": f"{winner} near-win, chain primarily interior",
                        })

        # blocked_or_trap: loser has high peg count & bridge density but
        # no goal-touching component, near mid-game
        if "blocked_or_trap" in categories_wanted:
            if winner in ("red", "black") and 40 <= ply <= 120:
                loser = "black" if winner == "red" else "red"
                loser_pegs = sum(1 for c in state.pegs.values() if c == loser)
                l_m1, l_m2, _ = state.connectivity_masks(loser)
                if loser_pegs >= 12 and l_m1.sum() == 0 and l_m2.sum() == 0:
                    candidates.append({
                        "category": "blocked_or_trap",
                        "ply": ply + 1,
                        "move_history": move_seq[:ply + 1],
                        "side_to_move": state.to_move,
                        "active_size": active,
                        "source_game": game_path,
                        "source_ply": ply + 1,
                        "note": f"{loser} has {loser_pegs} pegs but no goal-touching component",
                    })

        # dense_but_disconnected: similar but for either color
        if "dense_but_disconnected" in categories_wanted:
            for player in ("red", "black"):
                peg_count = sum(1 for col in state.pegs.values() if col == player)
                if peg_count >= 15:
                    m1, m2, _ = state.connectivity_masks(player)
                    if m1.sum() == 0 and m2.sum() == 0:
                        candidates.append({
                            "category": "dense_but_disconnected",
                            "ply": ply + 1,
                            "move_history": move_seq[:ply + 1],
                            "side_to_move": state.to_move,
                            "active_size": active,
                            "source_game": game_path,
                            "source_ply": ply + 1,
                            "note": f"{player} has {peg_count} pegs, zero goal-touching",
                        })

        # false_positive_connectivity: winner has one goal-touching component
        # that LOOKS large but does not connect, while a smaller component elsewhere
        # eventually wins. Harder heuristic -- capture positions where winner has
        # a goal-touching component size >= 8 but connected_to_both is still empty.
        if "false_positive_connectivity" in categories_wanted:
            if winner in ("red", "black") and 60 <= ply <= 150:
                m_g1, m_g2, m_both = state.connectivity_masks(winner)
                large_on_g1 = m_g1.sum() >= 8
                large_on_g2 = m_g2.sum() >= 8
                if (large_on_g1 or large_on_g2) and m_both.sum() == 0:
                    candidates.append({
                        "category": "false_positive_connectivity",
                        "ply": ply + 1,
                        "move_history": move_seq[:ply + 1],
                        "side_to_move": state.to_move,
                        "active_size": active,
                        "source_game": game_path,
                        "source_ply": ply + 1,
                        "note": f"{winner} has large goal-touching component but not connected to both edges yet",
                    })

        # edge_corner_legitimate: winner-path positions where win happens
        # despite edge/corner placement (pegs in outermost row/col)
        if "edge_corner_legitimate" in categories_wanted:
            if winner in ("red", "black") and is_terminal:
                # Check winning-side pegs: how many are edge/corner cells
                winning_pegs = [(rr, cc) for (rr, cc), col in state.pegs.items() if col == winner]
                outer = sum(1 for (rr, cc) in winning_pegs
                            if rr == 0 or rr == active - 1 or cc == 0 or cc == active - 1)
                if winning_pegs and outer / len(winning_pegs) >= 0.3:
                    # Take a mid-game snapshot, not terminal
                    snapshot_ply = max(10, ply - 4)
                    candidates.append({
                        "category": "edge_corner_legitimate",
                        "ply": snapshot_ply,
                        "move_history": move_seq[:snapshot_ply],
                        "side_to_move": "red" if snapshot_ply % 2 == 0 else "black",  # depends on start
                        "active_size": active,
                        "source_game": game_path,
                        "source_ply": snapshot_ply,
                        "note": f"{winner} eventually won with {outer}/{len(winning_pegs)} edge/corner pegs",
                    })

        if is_terminal:
            break

    return candidates


def _symmetric_pairs(candidates: list[dict]) -> list[dict]:
    """For a few candidates, emit their left-right mirror as symmetric_sanity probes."""
    out = []
    for i, cand in enumerate(candidates[:20]):
        active = cand["active_size"]
        mirrored_moves = [[r, active - 1 - c] for [r, c] in cand["move_history"]]
        out.append({
            "category": "symmetric_sanity",
            "ply": cand["ply"],
            "move_history": mirrored_moves,
            "side_to_move": cand["side_to_move"],
            "active_size": active,
            "source_game": cand["source_game"] + "#mirror",
            "source_ply": cand["source_ply"],
            "note": f"mirror of candidate {i} for symmetry check",
            "mirror_of_index": i,
        })
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate probe candidates from historical game JSONs.")
    ap.add_argument("--input", default="scripts/GPU/logs/games",
                    help="Directory of game JSONs (default: scripts/GPU/logs/games)")
    ap.add_argument("--out", required=True, help="Output candidates JSON path")
    ap.add_argument("--min-source-iter", type=int, default=900,
                    help="Min iteration to include (default 900 = current regime)")
    ap.add_argument("--source-iter-range", nargs=2, type=int, metavar=("MIN", "MAX"),
                    help="Explicit iter range [MIN, MAX] (overrides --min-source-iter)")
    ap.add_argument("--any-size", action="store_true",
                    help="Don't filter by active_size=24")
    ap.add_argument("--per-category-target", type=int, default=PER_CATEGORY_TARGET_DEFAULT,
                    help=f"Target candidate count per category (default {PER_CATEGORY_TARGET_DEFAULT})")
    args = ap.parse_args()

    min_iter = args.min_source_iter
    max_iter = None
    if args.source_iter_range:
        min_iter, max_iter = args.source_iter_range

    game_pat = re.compile(r"iter_(\d{4,})_game_(\d+)\.json$")
    game_files = []
    for fp in sorted(glob.glob(os.path.join(args.input, "iter_*_game_*.json"))):
        m = game_pat.search(os.path.basename(fp))
        if not m:
            continue
        iter_num = int(m.group(1))
        if iter_num < min_iter:
            continue
        if max_iter is not None and iter_num > max_iter:
            continue
        game_files.append((iter_num, fp))

    if not game_files:
        print(f"[ERROR] No games matching filter (min_iter={min_iter}, max_iter={max_iter})",
              file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {len(game_files)} games...")

    by_category: dict[str, list[dict]] = defaultdict(list)
    wanted = set(CATEGORIES) - {"symmetric_sanity"}  # added last
    for iter_num, fp in game_files:
        with open(fp) as f:
            game = json.load(f)
        meta = game.get("meta") or {}
        if not _size_filter(meta, args.any_size):
            continue
        cands = _classify_candidates(game, fp, wanted)
        for c in cands:
            by_category[c["category"]].append(c)
        # Early termination: stop once every primary category has reached
        # the per-category target. symmetric_sanity is generated after this
        # loop from mirror pairs, so it is not included in the check.
        if all(len(by_category[cat]) >= args.per_category_target for cat in PRIMARY_CATEGORIES):
            print(
                f"All {len(PRIMARY_CATEGORIES)} primary categories reached "
                f"target={args.per_category_target}; stopping after iter_num={iter_num}"
            )
            break

    # Cap per category
    pruned: list[dict] = []
    for cat in CATEGORIES:
        if cat == "symmetric_sanity":
            continue
        cat_cands = by_category[cat][:args.per_category_target]
        for i, c in enumerate(cat_cands):
            c["id"] = f"{cat}-{i:03d}"
        pruned.extend(cat_cands)

    # Mirror pairs for symmetric_sanity
    mirrors = _symmetric_pairs(pruned)
    for i, m in enumerate(mirrors):
        m["id"] = f"symmetric_sanity-{i:03d}"
    pruned.extend(mirrors)

    out_data = {
        "version": 1,
        "generated_with": "scripts/build_probe_candidates.py",
        "source_filter": {
            "input_dir": args.input,
            "min_source_iter": min_iter,
            "max_source_iter": max_iter,
            "any_size": args.any_size,
        },
        "total_candidates": len(pruned),
        "candidates": pruned,
    }

    with open(args.out, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"Wrote {len(pruned)} candidates to {args.out}")
    for cat in CATEGORIES:
        count = sum(1 for c in pruned if c["category"] == cat)
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
