"""Build a deterministic, GAME-HELD-OUT, NON-A-SELECTED neutral position manifest
for the v16a FPU collateral-damage screen.

Samples ordinary positions across mixed games/plies/sides/outcomes, EXCLUDING
entire games named in the A discovery manifest, so discovery and validation share
no games. Output conforms to the canonical position_probe_cases schema and flows
through the SAME trusted path (load_csv_manifest -> search_for_row ->
position_state), unmodified. Stratified (opening/early-mid/midgame/late),
game-first round-robin, per-game capped, min-ply-gap separated, ~50/50 side
balanced per bucket. Winner-null games are kept (game_result="unknown"): in the
default corpus all such games are state_cap 280-ply marathons -- the most
search-stressed, highest-value late samples.

READ-ONLY on replays; writes one CSV + sidecar meta JSON. No MCTS/network/train.
Building/running THIS builder is in scope for v16a; running the FPU sweep is NOT.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from .goal_line_trigger_probe_cases import position_state

BUCKETS = (("opening", 1, 15), ("early_mid", 16, 40),
           ("midgame", 41, 90), ("late", 91, None))
BUCKET_ORDER = [name for name, _, _ in BUCKETS]
DEFAULT_QUOTAS = {"opening": 40, "early_mid": 100, "midgame": 100, "late": 100}
DEFAULT_PER_GAME_CAPS = {"opening": 1, "early_mid": 2, "midgame": 2, "late": 2}
DEFAULT_MIN_PLY_GAP = 8

NEUTRAL_FIELDNAMES = [
    "case_id", "game_idx", "replay_path", "position_ply", "side_to_move",
    "ply_bucket", "game_result", "total_game_plies", "source_replay", "sample_seed",
]


def side_to_move_for_ply(ply: int) -> str:
    """Ply 0 is red's opener; TwixtState alternates each ply. position_state
    asserts exactly this parity -> a mismatch fails loud at validation."""
    return "red" if ply % 2 == 0 else "black"


def bucket_for_ply(ply: int, buckets=BUCKETS):
    for name, lo, hi in buckets:
        if ply >= lo and (hi is None or ply <= hi):
            return name
    return None


def candidate_positions(game_records, buckets=BUCKETS):
    """{bucket -> {game_idx -> [valid plies asc]}}; valid iff in bucket AND
    0 <= ply < n_moves (position_state range)."""
    out = {name: {} for name, _, _ in buckets}
    for r in sorted(game_records, key=lambda x: x["game_idx"]):
        n = r["n_moves"]
        for name, lo, hi in buckets:
            hi_eff = (n - 1) if hi is None else min(hi, n - 1)
            plies = list(range(lo, hi_eff + 1))
            if plies:
                out[name][r["game_idx"]] = plies
    return out


def _pick_ply(cand, per_game_selected, min_gap, side_count, side_of):
    ok = [p for p in cand if all(abs(p - q) >= min_gap for q in per_game_selected)]
    if not ok:
        return None
    behind = min(side_count, key=lambda s: (side_count[s], s))
    for p in ok:
        if side_of(p) == behind:
            return p
    return ok[0]


def sample_bucket(pool_by_game, *, quota, cap, min_gap, seed,
                  side_of=side_to_move_for_ply, state_key_fn=None):
    """Game-first round-robin: pass 1 takes <=1 ply/game (covers every game),
    later passes add up to `cap`/game, each >= min_gap from that game's picks;
    side balanced toward 50/50; optional state_key_fn de-dupes across games.
    Deterministic. Returns (selected [(game,ply)], side_count)."""
    rng = random.Random(seed)
    games = sorted(pool_by_game)
    rng.shuffle(games)
    plies = {g: pool_by_game[g][:] for g in games}
    for g in games:
        rng.shuffle(plies[g])
    picked, selected, seen = {}, [], set()
    side_count = {"red": 0, "black": 0}
    progress = True
    while len(selected) < quota and progress:
        progress = False
        for g in games:
            if len(selected) >= quota:
                break
            if len(picked.get(g, [])) >= cap:
                continue
            cand = plies[g]
            if state_key_fn is not None:
                cand = [p for p in cand if state_key_fn(g, p) not in seen]
            chosen = _pick_ply(cand, picked.get(g, []), min_gap, side_count, side_of)
            if chosen is None:
                continue
            picked.setdefault(g, []).append(chosen)
            plies[g].remove(chosen)
            if state_key_fn is not None:
                seen.add(state_key_fn(g, chosen))
            selected.append((g, chosen))
            side_count[side_of(chosen)] += 1
            progress = True
    return selected, side_count


def sample_neutral_rows(game_records, *, base_seed, source_replay,
                        buckets=BUCKETS, quotas=DEFAULT_QUOTAS,
                        per_game_caps=DEFAULT_PER_GAME_CAPS,
                        min_ply_gap=DEFAULT_MIN_PLY_GAP,
                        state_key_fn_by_bucket=None):
    recs = {r["game_idx"]: r for r in game_records}
    pools = candidate_positions(game_records, buckets)
    key_by_bucket = state_key_fn_by_bucket or {}
    rows, stats = [], {}
    for offset, (name, _lo, _hi) in enumerate(buckets):
        sel, side_count = sample_bucket(
            pools[name], quota=quotas[name], cap=per_game_caps[name],
            min_gap=min_ply_gap, seed=base_seed + offset,
            state_key_fn=key_by_bucket.get(name))
        for game_idx, ply in sel:
            rec = recs[game_idx]
            rows.append({
                "case_id": f"neutral_game_{game_idx:06d}_ply_{ply:03d}",
                "game_idx": game_idx, "replay_path": rec["replay_path"],
                "position_ply": ply, "side_to_move": side_to_move_for_ply(ply),
                "ply_bucket": name, "game_result": rec["winner"],
                "total_game_plies": rec["n_moves"], "source_replay": source_replay,
                "sample_seed": base_seed,
            })
        stats[name] = {"requested": quotas[name], "achieved": len(sel),
                       "games_used": len({g for g, _ in sel}),
                       "eligible_games": len(pools[name]), "side_balance": side_count}
    return rows, stats
