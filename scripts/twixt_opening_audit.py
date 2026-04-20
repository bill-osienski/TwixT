#!/usr/bin/env python3
"""
twixt_opening_audit.py

Analyze TwixT AlphaZero-style self-play game JSON logs for opening predictability,
corner/edge bias, and symmetry-canonicalized opening templates.

Input JSON format expected (matches iter_0533_game_099.json):
{
  "winner": "red"|"black"|"draw",
  "starting_player": "red"|"black",
  "moves": [{"turn": 1, "player": "red", "row": 21, "col": 19, ...}, ...]
}

Usage examples:
  # Analyze all game JSON files in a directory
  python twixt_opening_audit.py --path /path/to/games --glob "iter_*_game_*.json" --outdir out

  # Analyze a specific list of files
  python twixt_opening_audit.py --files iter_0533_game_099.json iter_0533_game_098.json --outdir out

  # Focus on last N games (sorted by filename)
  python twixt_opening_audit.py --path /path/to/games --glob "iter_*_game_*.json" --last 1500 --outdir out

Outputs:
  - out/summary.json (all aggregate stats)
  - out/top_moves_*.csv, out/top_sequences_*.csv
  - out/heat_*.png  (plies 1-4 heatmaps; canonical + raw)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# matplotlib is optional; we only import if making plots
def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        return None


Coord = Tuple[int, int]
Seq = Tuple[Coord, ...]


@dataclass(frozen=True)
class Geometry:
    n: int = 24
    near_corner_radius: int = 3   # Chebyshev radius
    edge_band_width: int = 2      # width B (outer band)


def chebyshev_dist(a: Coord, b: Coord) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def is_near_corner(rc: Coord, g: Geometry) -> bool:
    n = g.n
    corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    return any(chebyshev_dist(rc, c) <= g.near_corner_radius for c in corners)


def is_in_edge_band(rc: Coord, g: Geometry) -> bool:
    r, c = rc
    n = g.n
    b = g.edge_band_width
    return (r < b) or (c < b) or (r >= n - b) or (c >= n - b)


# --- Symmetry canonicalization (D4) ---
def _transforms(n: int):
    # Return 8 transforms (r,c)->(r',c') representing dihedral group of square
    def t0(r, c):  # identity
        return r, c
    def t1(r, c):  # reflect vertical
        return r, n - 1 - c
    def t2(r, c):  # reflect horizontal
        return n - 1 - r, c
    def t3(r, c):  # rotate 180
        return n - 1 - r, n - 1 - c
    def t4(r, c):  # transpose
        return c, r
    def t5(r, c):  # transpose + reflect vertical
        return c, n - 1 - r
    def t6(r, c):  # transpose + reflect horizontal
        return n - 1 - c, r
    def t7(r, c):  # rotate 90 (one of the transpose variants)
        return n - 1 - c, n - 1 - r

    return [t0, t1, t2, t3, t4, t5, t6, t7]


def canonicalize_sequence(seq: Seq, n: int) -> Seq:
    """Apply all symmetries and choose lexicographically smallest transformed sequence."""
    best: Optional[Seq] = None
    for tf in _transforms(n):
        t_seq = tuple(tf(r, c) for (r, c) in seq)
        if best is None or t_seq < best:
            best = t_seq
    assert best is not None
    return best


# --- Stats helpers ---
def entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for v in counts.values():
        p = v / total
        if p > 0:
            h -= p * math.log(p)
    return h


def topk_share(counts: Counter, k: int) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    top = sum(v for _, v in counts.most_common(k))
    return top / total


def share_of_most_common(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return counts.most_common(1)[0][1] / total


def write_counter_csv(path: str, header: List[str], rows: List[List]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def plot_heatmap(mat: np.ndarray, title: str, outpath: str):
    plt = _try_import_matplotlib()
    if plt is None:
        return
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure(figsize=(9, 7))
    plt.imshow(mat, origin="upper")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("col")
    plt.ylabel("row")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# --- Game parsing ---
def load_game(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def extract_opening_moves(game: dict, max_plies: int) -> List[Coord]:
    moves = game.get("moves", [])
    opening = []
    for m in moves:
        turn = int(m.get("turn", 0))
        if 1 <= turn <= max_plies:
            opening.append((int(m["row"]), int(m["col"])))
        if turn >= max_plies:
            break
    return opening


def extract_first_move_by_player(game: dict, player: str) -> Optional[Coord]:
    for m in game.get("moves", []):
        if m.get("player") == player:
            return (int(m["row"]), int(m["col"]))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", help="Directory containing game json files")
    ap.add_argument("--glob", default="*.json", help="Glob pattern under --path (default *.json)")
    ap.add_argument("--files", nargs="*", help="Explicit list of game json files")
    ap.add_argument("--last", type=int, default=0, help="If >0, only analyze last N files after sorting")
    ap.add_argument("--outdir", default="opening_audit_out", help="Output directory")
    ap.add_argument("--board-size", type=int, default=24)
    ap.add_argument("--opening-plies", type=int, default=4, help="Analyze first N plies (default 4)")
    ap.add_argument("--near-corner-radius", type=int, default=3)
    ap.add_argument("--edge-band-width", type=int, default=2)
    ap.add_argument("--no-plots", action="store_true", help="Disable heatmap plots")
    args = ap.parse_args()

    # Build file list
    files: List[str] = []
    if args.files:
        files.extend(args.files)
    if args.path:
        files.extend(glob.glob(os.path.join(args.path, args.glob)))
    files = sorted(set(files))
    if args.last and args.last > 0:
        files = files[-args.last:]

    if not files:
        raise SystemExit("No files found. Provide --path/--glob or --files.")

    g = Geometry(
        n=args.board_size,
        near_corner_radius=args.near_corner_radius,
        edge_band_width=args.edge_band_width,
    )

    NPLIES = args.opening_plies

    # Counters / accumulators
    first_move_game = Counter()            # first move of game (starting player)
    first_move_game_canon = Counter()

    first_move_red = Counter()
    first_move_black = Counter()
    first_move_red_canon = Counter()
    first_move_black_canon = Counter()

    seq2 = Counter()                       # (ply1, ply2)
    seq4 = Counter()                       # (ply1..ply4)
    seq2_canon = Counter()
    seq4_canon = Counter()

    # Per-ply heatmaps (raw and canonical)
    heat_raw = np.zeros((g.n, g.n), dtype=np.int64)
    heat_canon = np.zeros((g.n, g.n), dtype=np.int64)
    # Ply-bucket (1-4) heatmaps separated by player
    heat_raw_by_player = { "red": np.zeros((g.n, g.n), dtype=np.int64),
                           "black": np.zeros((g.n, g.n), dtype=np.int64) }
    heat_canon_by_player = { "red": np.zeros((g.n, g.n), dtype=np.int64),
                             "black": np.zeros((g.n, g.n), dtype=np.int64) }

    # Edge/corner rates by ply (1..NPLIES), overall and by color
    ply_stats = {
        "overall": {p: {"edge": 0, "corner": 0, "n": 0} for p in range(1, NPLIES + 1)},
        "red":     {p: {"edge": 0, "corner": 0, "n": 0} for p in range(1, NPLIES + 1)},
        "black":   {p: {"edge": 0, "corner": 0, "n": 0} for p in range(1, NPLIES + 1)},
    }

    # Aggregate ply-bucket (1..NPLIES) rates (like your analyzer summaries)
    bucket = {
        "overall": {"edge": 0, "corner": 0, "n": 0},
        "red":     {"edge": 0, "corner": 0, "n": 0},
        "black":   {"edge": 0, "corner": 0, "n": 0},
    }

    winners = Counter()
    starters = Counter()
    game_lengths = []

    for fp in files:
        try:
            game = load_game(fp)
        except Exception:
            continue

        winners[game.get("winner", "unknown")] += 1
        starters[game.get("starting_player", "unknown")] += 1

        moves = game.get("moves", [])
        game_lengths.append(len(moves))

        opening = extract_opening_moves(game, NPLIES)
        if not opening:
            continue

        # First move of game
        fm = opening[0]
        first_move_game[fm] += 1

        # Canonicalize by symmetry based on the whole opening sequence (up to NPLIES)
        opening_seq = tuple(opening)
        opening_canon = canonicalize_sequence(opening_seq, g.n)
        fm_canon = opening_canon[0]
        first_move_game_canon[fm_canon] += 1

        # Sequences (2-ply and 4-ply where available)
        if len(opening_seq) >= 2:
            s2 = tuple(opening_seq[:2])
            seq2[s2] += 1
            seq2_canon[tuple(opening_canon[:2])] += 1
        if len(opening_seq) >= 4:
            s4 = tuple(opening_seq[:4])
            seq4[s4] += 1
            seq4_canon[tuple(opening_canon[:4])] += 1

        # Player-specific first move
        r0 = extract_first_move_by_player(game, "red")
        b0 = extract_first_move_by_player(game, "black")
        if r0 is not None:
            first_move_red[r0] += 1
            first_move_red_canon[canonicalize_sequence((r0,), g.n)[0]] += 1
        if b0 is not None:
            first_move_black[b0] += 1
            first_move_black_canon[canonicalize_sequence((b0,), g.n)[0]] += 1

        # Heatmaps and edge/corner stats for plies 1..NPLIES
        for m in moves:
            turn = int(m.get("turn", 0))
            if not (1 <= turn <= NPLIES):
                continue
            rc = (int(m["row"]), int(m["col"]))
            pl = m.get("player", "unknown")
            r, c = rc

            heat_raw[r, c] += 1
            heat_raw_by_player.get(pl, heat_raw)[r, c] += 1

            # Canonical position (use same canonical transform chosen by opening)
            rc_c = opening_canon[turn - 1]  # aligned by ply index
            heat_canon[rc_c[0], rc_c[1]] += 1
            if pl in heat_canon_by_player:
                heat_canon_by_player[pl][rc_c[0], rc_c[1]] += 1

            # Update ply stats
            e = is_in_edge_band(rc, g)
            nc = is_near_corner(rc, g)
            ply_stats["overall"][turn]["n"] += 1
            ply_stats["overall"][turn]["edge"] += int(e)
            ply_stats["overall"][turn]["corner"] += int(nc)

            if pl in ("red", "black"):
                ply_stats[pl][turn]["n"] += 1
                ply_stats[pl][turn]["edge"] += int(e)
                ply_stats[pl][turn]["corner"] += int(nc)

            # Bucket (1..NPLIES)
            bucket["overall"]["n"] += 1
            bucket["overall"]["edge"] += int(e)
            bucket["overall"]["corner"] += int(nc)
            if pl in ("red", "black"):
                bucket[pl]["n"] += 1
                bucket[pl]["edge"] += int(e)
                bucket[pl]["corner"] += int(nc)

    # Build summary metrics
    def dist_summary(name: str, counts: Counter) -> Dict:
        h = entropy_from_counts(counts)
        return {
            "name": name,
            "n": int(sum(counts.values())),
            "unique": int(len(counts)),
            "entropy_nats": h,
            "effective_n": float(math.exp(h)) if h > 0 else 1.0,
            "top1_share": float(share_of_most_common(counts)) if counts else 0.0,
            "top5_share": float(topk_share(counts, 5)) if counts else 0.0,
            "top10_share": float(topk_share(counts, 10)) if counts else 0.0,
        }

    def seq_summary(name: str, counts: Counter) -> Dict:
        h = entropy_from_counts(counts)
        return {
            "name": name,
            "n": int(sum(counts.values())),
            "unique": int(len(counts)),
            "entropy_nats": h,
            "effective_n": float(math.exp(h)) if h > 0 else 1.0,
            "top1_share": float(share_of_most_common(counts)) if counts else 0.0,
        }

    summary = {
        "files_analyzed": len(files),
        "board_size": g.n,
        "opening_plies": NPLIES,
        "geometry": {
            "near_corner_radius": g.near_corner_radius,
            "edge_band_width": g.edge_band_width,
        },
        "winners": dict(winners),
        "starting_player": dict(starters),
        "game_length": {
            "count": len(game_lengths),
            "mean": float(np.mean(game_lengths)) if game_lengths else 0.0,
            "p50": float(np.percentile(game_lengths, 50)) if game_lengths else 0.0,
            "p90": float(np.percentile(game_lengths, 90)) if game_lengths else 0.0,
            "p95": float(np.percentile(game_lengths, 95)) if game_lengths else 0.0,
            "p99": float(np.percentile(game_lengths, 99)) if game_lengths else 0.0,
            "max": int(max(game_lengths)) if game_lengths else 0,
        },
        "distributions": {
            "first_move_game": dist_summary("first_move_game", first_move_game),
            "first_move_game_canon": dist_summary("first_move_game_canon", first_move_game_canon),
            "first_move_red": dist_summary("first_move_red", first_move_red),
            "first_move_black": dist_summary("first_move_black", first_move_black),
            "first_move_red_canon": dist_summary("first_move_red_canon", first_move_red_canon),
            "first_move_black_canon": dist_summary("first_move_black_canon", first_move_black_canon),
        },
        "sequences": {
            "seq2": seq_summary("seq2", seq2),
            "seq2_canon": seq_summary("seq2_canon", seq2_canon),
            "seq4": seq_summary("seq4", seq4),
            "seq4_canon": seq_summary("seq4_canon", seq4_canon),
        },
        "rates": {
            "ply_stats": ply_stats,
            "bucket_1_to_N": bucket,
        },
    }

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Write top moves CSV
    def coord_to_str(rc: Coord) -> str:
        return f"({rc[0]},{rc[1]})"

    def write_top_moves(name: str, counts: Counter):
        rows = []
        total = sum(counts.values()) or 1
        for rc, v in counts.most_common(50):
            rows.append([coord_to_str(rc), v, v / total])
        write_counter_csv(
            os.path.join(args.outdir, f"top_moves_{name}.csv"),
            ["move", "count", "share"],
            rows,
        )

    def seq_to_str(s: Seq) -> str:
        return "[" + ",".join(coord_to_str(rc) for rc in s) + "]"

    def write_top_seqs(name: str, counts: Counter):
        rows = []
        total = sum(counts.values()) or 1
        for s, v in counts.most_common(50):
            rows.append([seq_to_str(s), v, v / total])
        write_counter_csv(
            os.path.join(args.outdir, f"top_sequences_{name}.csv"),
            ["sequence", "count", "share"],
            rows,
        )

    write_top_moves("first_move_game", first_move_game)
    write_top_moves("first_move_game_canon", first_move_game_canon)
    write_top_moves("first_move_red", first_move_red)
    write_top_moves("first_move_black", first_move_black)
    write_top_moves("first_move_red_canon", first_move_red_canon)
    write_top_moves("first_move_black_canon", first_move_black_canon)

    write_top_seqs("seq2", seq2)
    write_top_seqs("seq2_canon", seq2_canon)
    write_top_seqs("seq4", seq4)
    write_top_seqs("seq4_canon", seq4_canon)

    # Heatmap plots
    if not args.no_plots:
        plot_heatmap(heat_raw, f"all placements (plies 1-{NPLIES})", os.path.join(args.outdir, f"heat_all_1-{NPLIES}_raw.png"))
        plot_heatmap(heat_canon, f"all placements (plies 1-{NPLIES}) [canonical]", os.path.join(args.outdir, f"heat_all_1-{NPLIES}_canon.png"))
        plot_heatmap(heat_raw_by_player["red"], f"red placements (plies 1-{NPLIES})", os.path.join(args.outdir, f"heat_red_1-{NPLIES}_raw.png"))
        plot_heatmap(heat_raw_by_player["black"], f"black placements (plies 1-{NPLIES})", os.path.join(args.outdir, f"heat_black_1-{NPLIES}_raw.png"))
        plot_heatmap(heat_canon_by_player["red"], f"red placements (plies 1-{NPLIES}) [canonical]", os.path.join(args.outdir, f"heat_red_1-{NPLIES}_canon.png"))
        plot_heatmap(heat_canon_by_player["black"], f"black placements (plies 1-{NPLIES}) [canonical]", os.path.join(args.outdir, f"heat_black_1-{NPLIES}_canon.png"))

    # Console quick report
    def pr(s: str):
        print(s)

    pr("\n=== Opening Predictability Audit ===")
    pr(f"Files analyzed: {len(files)}")
    pr(f"Geometry: near_corner_radius={g.near_corner_radius}, edge_band_width={g.edge_band_width}")
    pr(f"Opening plies analyzed: 1..{NPLIES}")
    pr("")

    for k, v in summary["distributions"].items():
        pr(f"[{k}] n={v['n']} unique={v['unique']} H={v['entropy_nats']:.3f} effN={v['effective_n']:.2f} "
           f"top1={v['top1_share']:.3f} top5={v['top5_share']:.3f} top10={v['top10_share']:.3f}")

    pr("")
    for k, v in summary["sequences"].items():
        pr(f"[{k}] n={v['n']} unique={v['unique']} H={v['entropy_nats']:.3f} effN={v['effective_n']:.2f} "
           f"top1={v['top1_share']:.3f}")

    pr("\nBucket rates (plies 1..N):")
    for who in ("overall", "red", "black"):
        n = bucket[who]["n"] or 1
        pr(f"  {who:7s}: edge_rate={bucket[who]['edge']/n:.3f} near_corner_rate={bucket[who]['corner']/n:.3f} n={n}")

    pr("\nPer-ply rates:")
    for who in ("overall", "red", "black"):
        pr(f"  {who}:")
        for p in range(1, NPLIES + 1):
            n = ply_stats[who][p]["n"] or 1
            pr(f"    ply {p}: edge_rate={ply_stats[who][p]['edge']/n:.3f} near_corner_rate={ply_stats[who][p]['corner']/n:.3f} n={n}")

    pr(f"\nWinners: {dict(winners)}")
    pr(f"Starting player: {dict(starters)}")
    pr(f"Game length mean={summary['game_length']['mean']:.1f} p99={summary['game_length']['p99']:.1f} max={summary['game_length']['max']}")
    pr(f"\nWrote outputs to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()