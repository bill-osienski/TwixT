"""FPU (policy-mass) development-corpus builder.

Frozen design ref: docs/superpowers/specs/2026-07-10-context-relative-fpu-policy-mass-design.md
Plan Tasks 5-6.

=============================================================================
PURE SECTION (Task 5) -- constants + pure functions ONLY.
=============================================================================
Everything in this file is pure: plain-stdlib classification and sampling over
plain-dict "rows". NO MCTS / evaluator / GPU / MLX / heavy-numpy imports, no
I/O, no argument parsing. The operator shell (two-stage scan, per-ply n_legal,
raw-policy forward pass, 400-sim anchor confirm, canonical hashing/disjointness,
manifest writing, `main()`) is added by Task 6 BELOW this section and imports
these pure functions -- so keep this section cleanly separated and importable.

What this section does
----------------------
Sample a 240-row development corpus (180 target + 60 matched controls) from
anchor-CONFIRMED candidate positions and split it into 160 tuning / 80
frozen_check rows BY WHOLE GAME (a game's positions never straddle splits).
Membership/role was decided UPSTREAM by the fpu-off anchor + raw policy (never
by any candidate-FPU result); this section only classifies, allocates, and
samples.

A "row" is a dict carrying at least:
    game_idx, role ("target"|"control"), band, side ("red"|"black"),
    ply, ply_bucket, canonical_sha1
`sample_dev_rows` returns rows each additionally stamped with `split`.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

# Task 6 additions: both pure (no MCTS/GPU/MLX) -- `goal_line_trigger_probe_cases`
# imports only `statistics` + `TwixtState`; `fpu_state_hash` imports only
# `TwixtState` (see their own module docstrings). Importing build_fpu_dev_corpus
# for the pure-helper tests still touches no GPU/MLX/checkpoint/evaluator code.
from .fpu_state_hash import canonical_state_sha1
from .goal_line_trigger_probe_cases import position_state
# Review followup (dedup): side_to_move_for_ply is verbatim-identical to
# build_v16a_neutral_position_manifest's; imported rather than duplicated.
# That module's own top-level imports are equally pure (__future__/argparse/
# csv/json/random/pathlib/.goal_line_trigger_probe_cases) and it does not
# import this module, so this stays circular-import-free.
from .build_v16a_neutral_position_manifest import side_to_move_for_ply
# Stdlib-only provenance helpers (design §12.5) -- keeps this module GPU/MLX-free.
from . import fpu_provenance

# The corpus's effective result-determining source files, whose BYTES the
# evidence-grade manifest provenance pins (design §12.5). Referenced by PATH
# (from this package dir) so fingerprinting mcts.py never imports it. Includes
# the state-RECONSTRUCTION deps (goal_line_trigger_probe_cases.py +
# game/twixt_state.py): the builder rebuilds each corpus position via
# position_state -> TwixtState, so their bytes are equally result-determining
# (RF1; aligns with the diagnostic's RESULT_DETERMINING_SOURCES).
_MODULE_DIR = Path(__file__).resolve().parent
_CORPUS_SOURCES: Tuple[Path, ...] = (
    _MODULE_DIR / "build_fpu_dev_corpus.py",
    _MODULE_DIR / "mcts.py",
    _MODULE_DIR / "fpu_state_hash.py",
    _MODULE_DIR / "goal_line_trigger_probe_cases.py",
    _MODULE_DIR / "game" / "twixt_state.py",
)

# ---------------------------------------------------------------------------
# Frozen constants (verbatim from the design + Task-5 brief)
# ---------------------------------------------------------------------------

BANDS: Tuple[str, str, str] = ("b200_299", "b300_399", "b400_plus")

# Ply buckets (reused from v16a): opening 1-15, early_mid 16-40,
# midgame 41-90, late 91+.
PLY_BUCKETS: Tuple[str, str, str, str] = ("opening", "early_mid", "midgame", "late")

TARGET_PER_BAND = 60
CONTROL_PER_BAND = 20
MIN_PLY_GAP = 12
MAX_PER_GAME = 2

# Per-split side balance tolerance: |red - black| <= SIDE_TOL within each split.
# Every game supplies one red + one black position, so whole-game (2-per-game)
# picks are side-neutral; only the odd-quota control cells (13 / 7) leave a
# single leftover, and the sampler steers each leftover toward the deficit side.
# With two odd cells per split that can cancel, observed imbalance is 0; 2 is a
# safe cap that also tolerates any ordering/reserve variation.
SIDE_TOL = 2

SPLITS: Tuple[str, str] = ("tuning", "frozen_check")

# Frozen whole-game split allocation (Task-5 brief fix 7).
#   target cells: 40 tuning / 20 frozen each band  -> 120 / 60
#   control cells: {b200: 13/7, b300: 13/7, b400: 14/6} -> 40 / 20
#   totals: tuning 160, frozen_check 80; target 180, control 60; grand 240.
SPLIT_ALLOC: Dict[Tuple[str, str], Dict[str, int]] = {
    ("target", "b200_299"): {"tuning": 40, "frozen_check": 20},
    ("target", "b300_399"): {"tuning": 40, "frozen_check": 20},
    ("target", "b400_plus"): {"tuning": 40, "frozen_check": 20},
    ("control", "b200_299"): {"tuning": 13, "frozen_check": 7},
    ("control", "b300_399"): {"tuning": 13, "frozen_check": 7},
    ("control", "b400_plus"): {"tuning": 14, "frozen_check": 6},
}

# Deterministic (role, band) cell order = SPLIT_ALLOC insertion order.
CELL_ORDER: List[Tuple[str, str]] = list(SPLIT_ALLOC.keys())

# Total manifest size implied by the frozen allocation (240).
CORPUS_SIZE = sum(a["tuning"] + a["frozen_check"] for a in SPLIT_ALLOC.values())


# ---------------------------------------------------------------------------
# Two-stage classifiers (pure)
# ---------------------------------------------------------------------------

def band_of(n_legal: int) -> Optional[str]:
    """Branching band for a legal-move count, or None below the target floor.

    b200_299: 200-299, b300_399: 300-399, b400_plus: 400+. Below 200 is not an
    eligible dev-corpus position (returns None).
    """
    if n_legal >= 400:
        return "b400_plus"
    if n_legal >= 300:
        return "b300_399"
    if n_legal >= 200:
        return "b200_299"
    return None


def ply_bucket_of(ply: int) -> str:
    """Coarse game-phase bucket for the <=50% cap. opening 1-15, early_mid
    16-40, midgame 41-90, late 91+."""
    if ply <= 15:
        return "opening"
    if ply <= 40:
        return "early_mid"
    if ply <= 90:
        return "midgame"
    return "late"


def raw_policy_role(normalized_entropy: float, top1_prior: float) -> Optional[str]:
    """Stage-2 raw-policy role from geometry, or None for the grey zone.

    target  iff normalized_entropy >= 0.90 AND top1_prior <= 0.025 (flat, diffuse)
    control iff normalized_entropy <  0.85 OR  top1_prior >= 0.05  (concentrated)
    otherwise None (the grey band between, e.g. 0.88 / 0.03).

    The two positive conditions are mutually exclusive (a target requires
    entropy >= 0.90 and top1 <= 0.025, both of which fail the control test), so
    evaluation order is immaterial.
    """
    if normalized_entropy >= 0.90 and top1_prior <= 0.025:
        return "target"
    if normalized_entropy < 0.85 or top1_prior >= 0.05:
        return "control"
    return None


def anchor_eligible(root_value_stm: float) -> bool:
    """Near-even fpu-off anchor gate: |root_value_stm| <= 0.25 (inclusive)."""
    return abs(root_value_stm) <= 0.25


# ---------------------------------------------------------------------------
# Whole-game split assignment (contribution-aware, deterministic)
# ---------------------------------------------------------------------------

def _greedy_assign(games_profile, seed, attempt) -> Optional[Dict[Any, str]]:
    """One deterministic greedy pass. Returns {game_idx: split} if it satisfies
    every per-(role, band, split) quota (capacity), else None.

    Each WHOLE game is placed in the split whose still-unmet quotas it fills
    most; ties break toward the split with the larger total remaining need, then
    toward tuning. Games are visited in a seed-shuffled order (attempt 0) or its
    deterministic reverse (attempt 1, the secondary-ordering retry).

    A game's per-cell contribution is capped at MAX_PER_GAME because the sampler
    never draws more than MAX_PER_GAME rows from one game per cell; counting all
    of a >MAX_PER_GAME-position game's positions would over-state realizable
    capacity and let assign_split hand back an assignment the round-robin cannot
    fill (a spurious final-manifest shortfall).
    """
    rng = random.Random(seed * 1_000_003 + attempt)
    order = sorted(games_profile)
    rng.shuffle(order)
    if attempt == 1:
        order = order[::-1]

    need = {cell: dict(alloc) for cell, alloc in SPLIT_ALLOC.items()}
    assign: Dict[Any, str] = {}
    for gi in order:
        prof = games_profile[gi]
        cells = [c for c in prof if c in need]

        def useful(split, _cells=cells, _prof=prof):
            return sum(min(_prof[c], MAX_PER_GAME, need[c][split]) for c in _cells)

        u_t, u_f = useful("tuning"), useful("frozen_check")
        if u_t > u_f:
            split = "tuning"
        elif u_f > u_t:
            split = "frozen_check"
        else:
            tot_t = sum(need[c]["tuning"] for c in cells)
            tot_f = sum(need[c]["frozen_check"] for c in cells)
            split = "tuning" if tot_t >= tot_f else "frozen_check"

        assign[gi] = split
        for c in cells:
            need[c][split] = max(0, need[c][split] - min(prof[c], MAX_PER_GAME))

    if all(v == 0 for cell in need for v in need[cell].values()):
        return assign
    return None


def assign_split(games_profile: Mapping[Any, Mapping[Tuple[str, str], int]],
                 seed: int) -> Dict[Any, str]:
    """Assign each WHOLE game to "tuning" or "frozen_check" so every
    per-(role, band, split) SPLIT_ALLOC quota is satisfiable (enough capacity).

    `games_profile`: {game_idx: {(role, band): n_available_positions}}.

    Raises ValueError if any cell's total capacity is below its combined demand,
    or if neither the primary nor the deterministic secondary ordering yields a
    quota-satisfying assignment (a shortfall surfaces here).
    """
    capacity: Counter = Counter()
    for prof in games_profile.values():
        for cell, n in prof.items():
            capacity[cell] += min(n, MAX_PER_GAME)   # realizable, not raw, capacity
    for cell, alloc in SPLIT_ALLOC.items():
        demand = alloc["tuning"] + alloc["frozen_check"]
        have = capacity.get(cell, 0)
        if have < demand:
            raise ValueError(
                f"assign_split: cell {cell} capacity {have} < demand {demand}")

    for attempt in range(2):
        result = _greedy_assign(games_profile, seed, attempt)
        if result is not None:
            return result
    raise ValueError(
        "assign_split: no deterministic ordering satisfied the split quotas")


# ---------------------------------------------------------------------------
# 240-row sampler (round-robin, exact composition or raise)
# ---------------------------------------------------------------------------

def _choose_positions(positions, take_n, side_count, gap):
    """Pick up to `take_n` positions from one game's `positions` (sorted by ply).

    take_n == 1 -> steer toward the side that reduces the split's current
    |red - black| (ties break by lower ply then side, for determinism).
    take_n >= 2 -> greedily take the earliest positions that stay >= `gap` plies
    apart (a game's red+black pair is designed to clear the gap, keeping the pair
    side-neutral).
    """
    if take_n <= 0 or not positions:
        return []
    if take_n == 1:
        red, black = side_count["red"], side_count["black"]

        def imbalance_if(r):
            if r["side"] == "red":
                return abs((red + 1) - black)
            return abs(red - (black + 1))

        return [min(positions, key=lambda r: (imbalance_if(r), r["ply"], r["side"]))]

    chosen: List[dict] = []
    last_ply = None
    for r in positions:
        if last_ply is None or (r["ply"] - last_ply) >= gap:
            chosen.append(r)
            last_ply = r["ply"]
            if len(chosen) == take_n:
                break
    return chosen


def sample_dev_rows(confirmed: List[dict], *, seed: int) -> Tuple[List[dict], dict]:
    """Sample the 240-row dev corpus from anchor-CONFIRMED candidate rows.

    Steps: (1) build each game's (role, band) contribution profile; (2)
    `assign_split` places whole games into tuning / frozen_check; (3) round-robin
    within each assigned split fills every SPLIT_ALLOC cell EXACTLY, subject to
    MAX_PER_GAME (<=2/game), MIN_PLY_GAP (>=12-ply separation within a game), a
    per-split side balance |red-black| <= SIDE_TOL, a global ply-bucket <=50%
    cap, and no duplicate canonical_sha1.

    Every cell must reach its quota exactly; a final-manifest shortfall is an
    ERROR (raises ValueError), never a silent truncation. Deterministic under
    `seed`. Returns (rows, stats); each row is stamped with `split`.
    """
    games: Dict[Any, List[dict]] = defaultdict(list)
    for r in confirmed:
        games[r["game_idx"]].append(r)

    profile = {gi: Counter((r["role"], r["band"]) for r in rows_)
               for gi, rows_ in games.items()}

    split_of = assign_split(profile, seed)   # may raise ValueError (infeasible)

    bucket_cap = 0.5 * CORPUS_SIZE
    used_sha1: set = set()
    bucket_count: Counter = Counter()
    side_count = {s: {"red": 0, "black": 0} for s in SPLITS}
    selected: List[dict] = []

    for split in SPLITS:
        for cell in CELL_ORDER:
            quota = SPLIT_ALLOC[cell][split]
            picked = 0
            cand_games = sorted(
                gi for gi in games
                if split_of.get(gi) == split and cell in profile[gi])
            for gi in cand_games:
                if picked >= quota:
                    break
                positions = sorted(
                    (r for r in games[gi] if (r["role"], r["band"]) == cell),
                    key=lambda r: r["ply"])
                positions = [
                    r for r in positions
                    if r["canonical_sha1"] not in used_sha1
                    and bucket_count[r["ply_bucket"]] < bucket_cap]
                take_n = min(MAX_PER_GAME, quota - picked, len(positions))
                for r in _choose_positions(positions, take_n,
                                           side_count[split], MIN_PLY_GAP):
                    out = dict(r)
                    out["split"] = split
                    selected.append(out)
                    used_sha1.add(r["canonical_sha1"])
                    bucket_count[r["ply_bucket"]] += 1
                    side_count[split][r["side"]] += 1
                    picked += 1
            if picked != quota:
                raise ValueError(
                    f"final-manifest shortfall: cell {(cell[0], cell[1], split)} "
                    f"filled {picked} of required {quota}")

    # Count the rows actually selected per (role, band, split) so cell_counts is
    # an INDEPENDENT composition witness (not a re-emission of the SPLIT_ALLOC
    # quotas). On success these equal the quotas -- the exact-or-raise guard
    # above already fired if any cell fell short -- but computing them from the
    # selected rows makes the stats a real cross-check rather than a tautology.
    cell_counts_actual: Counter = Counter(
        (r["role"], r["band"], r["split"]) for r in selected)

    stats = {
        "n_rows": len(selected),
        "seed": seed,
        "cell_counts": {
            f"{role}|{band}|{split}": cell_counts_actual[(role, band, split)]
            for (role, band) in SPLIT_ALLOC for split in SPLITS},
        "side_count": {s: dict(side_count[s]) for s in SPLITS},
        "bucket_count": dict(bucket_count),
        "n_games_per_split": {
            s: sum(1 for gi in split_of if split_of[gi] == s) for s in SPLITS},
        "n_games_total": len(split_of),
    }
    return selected, stats


# =============================================================================
# OPERATOR SHELL (Task 6) -- appended below by a later task.
# per-ply n_legal, candidate enumeration, raw-policy forward pass, 400-sim
# anchor confirm, canonical hashing/disjointness, manifest writing, main().
# Nothing above this line imports MCTS/GPU/MLX or performs I/O.
#
# `main()` is an OPERATOR phase: it loads a real checkpoint, reconstructs real
# seed20116 replay positions, and runs 400-sim MCTS. It is NEVER invoked by
# this task's tests. Everything below down to `_build_anchor_search_fn` /
# `_scan_two_stage` / `main` is importable without GPU/MLX (only stdlib +
# the pure imports above, incl. `side_to_move_for_ply` imported from
# `.build_v16a_neutral_position_manifest` -- itself verified import-clean);
# the GPU/MLX/checkpoint/evaluator modules (`.eval_runner`,
# `.diagnose_fpu_sweep`, `.build_teacher_calibration_manifest`, `.mcts`'s
# MCTS class, and `.build_v16a_neutral_position_manifest`'s OTHER names) are
# imported LAZILY, inside the functions that actually need them, so merely
# importing this module (as the pure-helper tests do) never touches them.
# =============================================================================

# ---------------------------------------------------------------------------
# Per-ply legal-move counts (primary: stored; fallback: sparse reconstruction)
# ---------------------------------------------------------------------------


def per_ply_n_legal(replay: Mapping[str, Any]) -> List[Optional[int]]:
    """Per-ply legal-move count, index-aligned with `replay["moves"]`.

    PRIMARY: every move dict carries "n_legal" (verified present in the
    seed20116 source corpus) -> read directly, no reconstruction.

    FALLBACK (n_legal missing from at least one move -- a differently
    instrumented corpus): reconstruct only every 4th ply (0, 4, 8, ...) via
    `position_state` + `TwixtState.legal_moves()`. `position_state` replays
    the game from scratch on every call, so reconstructing EVERY ply would be
    O(n^2) in game length; the 4-ply stride bounds it to O(n) calls. The other
    ply slots are `None` (uncomputed, not zero) -- `enumerate_candidate_plies`
    treats `None` as non-qualifying, so a fallback corpus naturally yields a
    coarser (but still deterministic) candidate set rather than a wrong one.
    """
    moves = replay["moves"]
    if moves and all("n_legal" in m for m in moves):
        return [int(m["n_legal"]) for m in moves]

    out: List[Optional[int]] = [None] * len(moves)
    for ply in range(0, len(moves), 4):
        state = position_state(replay, ply, side_to_move_for_ply(ply))
        out[ply] = len(state.legal_moves())
    return out


# ---------------------------------------------------------------------------
# Stage-1 candidate-ply enumeration (cheap: stored/reconstructed n_legal only)
# ---------------------------------------------------------------------------

def enumerate_candidate_plies(replay: Mapping[str, Any], stride: int = 4,
                              cap: int = 6) -> List[int]:
    """The 1st, (1+stride)-th, (1+2*stride)-th, ... qualifying ply -- i.e.
    every `stride`-th entry of the QUALIFYING subsequence (n_legal >= 200,
    via `band_of(n) is not None`, the single canonical eligibility test) --
    in ascending ply order, capped at `cap` total. "Qualifying" is decided
    over ALL plies via `per_ply_n_legal`; a fallback `None` slot never
    qualifies (design S2 step 1: "every fourth eligible ply").
    """
    n_legal = per_ply_n_legal(replay)
    qualifying = [ply for ply, n in enumerate(n_legal)
                 if n is not None and band_of(n) is not None]
    return qualifying[::stride][:cap]


# ---------------------------------------------------------------------------
# Pure replay-geometry feasibility preflight (design AMENDMENT 2026-07-11 §11.4)
# ---------------------------------------------------------------------------
# A hard gate that runs BEFORE the evaluator loads in `main()`: it proves, from
# replay GEOMETRY ALONE (stored/reconstructed n_legal + ply + side; NO NN, NO
# MCTS, NO raw-policy), that a (source corpus, enumeration) pair can JOINTLY
# satisfy the four STRUCTURAL sampler constraints -- band quota, <=MAX_PER_GAME/
# game (+ >=MIN_PLY_GAP), per-split side balance, and the <=50% ply-bucket cap
# -- or it reports the binding constraint so `main()` stops before wasting
# evaluator setup (the first seed20116 build hard-stopped deep in the scan).
#
# SCOPE: ROLE (target vs control) is OUT of scope -- it needs the evaluator's
# raw policy, so the preflight checks only the per-band TOTAL (QUOTA_PER_BAND =
# 80 = 60 target + 20 control). It proves the geometric/structural upper bound;
# the role/eligibility fill stays an operator-runtime check under the
# stop-don't-retune rule (§11.4 pre-registration).
#
# SOUNDNESS: feasible=True is NEVER returned on necessary-check evidence alone
# (per-constraint capacity can each pass while the joint problem is infeasible
# -- §11.2.3). It is returned ONLY when a constructive WITNESS actually selects
# QUOTA_PER_BAND positions per band as whole-game red+black PAIRS (the sampler's
# own side-neutral mechanism -- see SIDE_TOL's note and `_choose_positions`),
# jointly honouring the per-game <=MAX_PER_GAME cap ACROSS bands, >=MIN_PLY_GAP
# spacing, the global ply-bucket cap, and a whole-game split into the frozen
# 160/80 budgets with |red-black| <= SIDE_TOL in each split. A successful
# witness IS a feasible selection, so the gate can never be false-feasible; it
# may be mildly conservative (false-infeasible) for an exotic corpus that could
# only balance via single-side games taken one apiece -- an accepted, documented
# limitation that never yields a silent false pass.

# Per-band TOTAL quota implied by the frozen SPLIT_ALLOC (60 target + 20 control
# = 80 for every band). DERIVED (not hard-coded) so it cannot drift from the
# real allocation; asserted uniform across bands.
_PER_BAND_TOTALS: Dict[str, int] = {}
for (_pf_role, _pf_band), _pf_alloc in SPLIT_ALLOC.items():
    _PER_BAND_TOTALS[_pf_band] = (_PER_BAND_TOTALS.get(_pf_band, 0)
                                  + _pf_alloc["tuning"] + _pf_alloc["frozen_check"])
assert set(_PER_BAND_TOTALS) == set(BANDS), _PER_BAND_TOTALS
assert len(set(_PER_BAND_TOTALS.values())) == 1, _PER_BAND_TOTALS
QUOTA_PER_BAND: int = _PER_BAND_TOTALS[BANDS[0]]   # 80

# Whole-game position budget per split (SPLIT_ALLOC totals: tuning 160 / 80).
_SPLIT_POS_BUDGET: Dict[str, int] = {
    split: sum(alloc[split] for alloc in SPLIT_ALLOC.values()) for split in SPLITS}


@dataclasses.dataclass(frozen=True)
class PreflightReport:
    """Structured result of `geometry_feasibility`.

    `feasible` is True IFF the constructive `witness` (a real QUOTA_PER_BAND-per-
    band selection satisfying all four constraints) was built. The remaining
    fields are the per-band / per-bucket DIAGNOSTICS that name which constraint
    bound when `feasible` is False (and, on success, corroborate the witness).
    """
    feasible: bool
    binding_constraint: Optional[str]
    quota_per_band: int
    bucket_cap: float
    n_games: int
    n_candidates: int
    realizable_by_band: Dict[str, int]      # positions realizable under <=2/game + gap
    pairs_by_band: Dict[str, int]           # distinct whole-game red+black pair-games
    red_by_band: Dict[str, int]             # total red candidate positions
    black_by_band: Dict[str, int]           # total black candidate positions
    forced_bucket_min: Dict[str, int]       # positions a band forces into ONE bucket
    witness: Optional[Tuple[dict, ...]]     # the selected rows (split-stamped) or None

    def format(self) -> str:
        head = ("FEASIBLE" if self.feasible
                else f"INFEASIBLE (binding: {self.binding_constraint})")
        lines = [
            f"[preflight] {head}",
            f"  games={self.n_games} candidates={self.n_candidates} "
            f"quota/band={self.quota_per_band} bucket_cap={self.bucket_cap:g}",
        ]
        for band in BANDS:
            lines.append(
                f"  {band}: realizable={self.realizable_by_band.get(band, 0)} "
                f"pairs={self.pairs_by_band.get(band, 0)} "
                f"red={self.red_by_band.get(band, 0)} "
                f"black={self.black_by_band.get(band, 0)}")
        if self.forced_bucket_min:
            forced = ", ".join(f"{b}={n}" for b, n in
                               sorted(self.forced_bucket_min.items()))
            lines.append(f"  forced-bucket-min: {forced}")
        return "\n".join(lines)

    __str__ = format


def build_candidates_by_game(replays_by_game: Mapping[Any, Mapping[str, Any]], *,
                             stride: int, cap: int) -> Dict[Any, List[dict]]:
    """Pure candidate builder -- REUSES the real scan primitives so the preflight
    cannot drift from `main()`'s enumeration. For each game, run the SAME
    `enumerate_candidate_plies(replay, stride, cap)` selection, and for each
    selected ply emit the geometry row {game_idx, ply, band, ply_bucket, side}
    via the SAME `band_of` / `ply_bucket_of` / `side_to_move_for_ply` the sampler
    uses. `band` is read from that ply's own `per_ply_n_legal` value, never
    re-derived. NO NN / MCTS / raw-policy -- pure n_legal geometry.
    """
    out: Dict[Any, List[dict]] = {}
    for game_idx, replay in replays_by_game.items():
        n_legal = per_ply_n_legal(replay)
        rows: List[dict] = []
        for ply in enumerate_candidate_plies(replay, stride=stride, cap=cap):
            band = band_of(n_legal[ply])
            if band is None:
                continue   # enumerate yields only qualifying plies; defensive
            rows.append({
                "game_idx": game_idx, "ply": ply, "band": band,
                "ply_bucket": ply_bucket_of(ply),
                "side": side_to_move_for_ply(ply),
            })
        out[game_idx] = rows
    return out


def _gap_selectable(plies_sorted: List[int], gap: int, cap: int) -> int:
    """How many of one game's `plies_sorted` are realizable under >=gap spacing,
    capped at `cap` -- the greedy earliest-first chain, matching
    `_choose_positions`' take_n>=2 branch and `assign_split`'s min(n, MAX_PER_GAME)
    capacity idiom."""
    chosen = 0
    last = None
    for p in plies_sorted:
        if last is None or (p - last) >= gap:
            chosen += 1
            last = p
            if chosen >= cap:
                break
    return chosen


def _first_gap_pair(reds: List[dict], blacks: List[dict], gap: int):
    """First (red_row, black_row) whose plies are >= gap apart (reds/blacks each
    ascending by ply), else None. Deterministic: lowest red then lowest black."""
    for r in reds:
        for b in blacks:
            if abs(r["ply"] - b["ply"]) >= gap:
                return r, b
    return None


def _fit_pair(reds: List[dict], blacks: List[dict], gap: int,
              bucket_count: "Counter", bucket_cap: float):
    """First gap-valid red+black pair whose ply_buckets BOTH stay within
    `bucket_cap` once added, else None -- so the witness never overfills a
    bucket (matches the sampler's `bucket_count < cap` filter: a bucket may
    reach, but not exceed, the cap)."""
    for r in reds:
        for b in blacks:
            if abs(r["ply"] - b["ply"]) < gap:
                continue
            add: Counter = Counter()
            add[r["ply_bucket"]] += 1
            add[b["ply_bucket"]] += 1
            if all(bucket_count[bk] + n <= bucket_cap for bk, n in add.items()):
                return r, b
    return None


def _build_witness(candidates_by_game, bands, quota, max_per_game, min_gap,
                   side_tol, bucket_cap):
    """Constructive witness. Greedily select `quota` positions per band as
    whole-game red+black PAIRS, honouring the JOINT <=max_per_game/game cap (a
    game used for a pair in ANY band is consumed -- the cross-band coupling the
    per-band necessary checks miss), >=min_gap spacing, and the global bucket
    cap; then split whole games into the frozen 160/80 budgets and verify
    per-split side balance. Returns (selected_rows, None) on success, else
    (None, binding_constraint). `quota` is even (QUOTA_PER_BAND=80), so `quota//2`
    pairs realize it exactly.
    """
    per_game_band: Dict[Any, Dict[str, List[dict]]] = {}
    for gi in sorted(candidates_by_game):
        m: Dict[str, List[dict]] = defaultdict(list)
        for c in candidates_by_game[gi]:
            if c["band"] in bands:
                m[c["band"]].append(c)
        per_game_band[gi] = m

    used_games: Set[Any] = set()
    bucket_count: Counter = Counter()
    selected: List[dict] = []
    selected_games_in_order: List[Any] = []
    pairs_per_band = quota // 2

    for band in bands:
        got = 0
        for gi in sorted(per_game_band):
            if got >= pairs_per_band:
                break
            if gi in used_games:
                continue
            rows = per_game_band[gi].get(band)
            if not rows:
                continue
            reds = sorted((r for r in rows if r["side"] == "red"),
                          key=lambda r: r["ply"])
            blacks = sorted((r for r in rows if r["side"] == "black"),
                            key=lambda r: r["ply"])
            pair = _fit_pair(reds, blacks, min_gap, bucket_count, bucket_cap)
            if pair is None:
                continue
            used_games.add(gi)
            selected_games_in_order.append(gi)
            for row in pair:
                selected.append(dict(row))
                bucket_count[row["ply_bucket"]] += 1
            got += 1
        if got < pairs_per_band:
            return None, (f"joint-band-quota:{band} (witness realized "
                          f"{got * 2} < {quota} positions under the JOINT "
                          f"<=2/game + gap + bucket-cap constraints)")

    # Whole-game split into the frozen 160/80 budgets. Every selected game
    # contributes exactly one side-neutral (red+black) pair, so any whole-game
    # partition is side-balanced; place each game in the first split with room.
    rows_by_game: Dict[Any, List[dict]] = defaultdict(list)
    for row in selected:
        rows_by_game[row["game_idx"]].append(row)
    split_of: Dict[Any, str] = {}
    filled = {s: 0 for s in SPLITS}
    for gi in selected_games_in_order:
        n = len(rows_by_game[gi])
        placed = False
        for split in SPLITS:
            if filled[split] + n <= _SPLIT_POS_BUDGET[split]:
                split_of[gi] = split
                filled[split] += n
                placed = True
                break
        if not placed:                       # no split has room (over-selection)
            return None, "split-budget:overflow (no split had room for a game)"
    for row in selected:
        row["split"] = split_of[row["game_idx"]]

    # Verify split budgets and per-split side balance (belt-and-suspenders:
    # all-pairs => imbalance 0, but verify a real witness rather than assume).
    side = {s: Counter() for s in SPLITS}
    per_split_pos: Counter = Counter()
    for row in selected:
        side[row["split"]][row["side"]] += 1
        per_split_pos[row["split"]] += 1
    for split in SPLITS:
        if per_split_pos[split] != _SPLIT_POS_BUDGET[split]:
            return None, (f"split-budget:{split} (realized {per_split_pos[split]} "
                          f"!= {_SPLIT_POS_BUDGET[split]})")
        imbalance = abs(side[split]["red"] - side[split]["black"])
        if imbalance > side_tol:
            return None, (f"side-balance:{split} (|red-black|={imbalance} "
                          f"> {side_tol})")
    return selected, None


def geometry_feasibility(
        candidates_by_game: Mapping[Any, List[Mapping[str, Any]]], *,
        quota_per_band: int = QUOTA_PER_BAND,
        max_per_game: int = MAX_PER_GAME,
        min_gap: int = MIN_PLY_GAP,
        side_tol: int = SIDE_TOL,
        bucket_cap: float = 0.5 * CORPUS_SIZE,
        bands: Tuple[str, ...] = BANDS) -> PreflightReport:
    """Pure JOINT feasibility core (design §11.4).

    `candidates_by_game`: {game_idx: [ {game_idx, ply, band, ply_bucket, side} ]}
    -- pure geometry rows (no file paths, no NN), as produced by
    `build_candidates_by_game`. Computes per-band / per-bucket capacity
    DIAGNOSTICS, short-circuits with a named binding constraint on any per-band
    NECESSARY violation, and only then attempts the constructive `_build_witness`
    that GOVERNS feasible=True. Deterministic (sorted iteration throughout).
    """
    realizable = {b: 0 for b in bands}
    pairs = {b: 0 for b in bands}
    red_tot = {b: 0 for b in bands}
    black_tot = {b: 0 for b in bands}
    band_buckets: Dict[str, Set[str]] = {b: set() for b in bands}
    n_candidates = 0

    for gi in sorted(candidates_by_game):
        by_band: Dict[str, List[dict]] = defaultdict(list)
        for c in candidates_by_game[gi]:
            n_candidates += 1
            if c["band"] in realizable:
                by_band[c["band"]].append(c)
        for band, rows in by_band.items():
            plies = sorted(r["ply"] for r in rows)
            realizable[band] += _gap_selectable(plies, min_gap, max_per_game)
            reds = sorted((r for r in rows if r["side"] == "red"),
                          key=lambda r: r["ply"])
            blacks = sorted((r for r in rows if r["side"] == "black"),
                            key=lambda r: r["ply"])
            red_tot[band] += len(reds)
            black_tot[band] += len(blacks)
            if _first_gap_pair(reds, blacks, min_gap) is not None:
                pairs[band] += 1
            for r in rows:
                band_buckets[band].add(r["ply_bucket"])

    # forced-bucket-min: a band whose candidates ALL sit in one bucket forces its
    # entire quota into that bucket (a sound lower bound on that bucket's count).
    forced_bucket: Counter = Counter()
    for band in bands:
        if len(band_buckets[band]) == 1:
            (only_bucket,) = tuple(band_buckets[band])
            forced_bucket[only_bucket] += quota_per_band
    forced_bucket_min = dict(forced_bucket)

    def _report(feasible, binding, witness):
        return PreflightReport(
            feasible=feasible, binding_constraint=binding,
            quota_per_band=quota_per_band, bucket_cap=float(bucket_cap),
            n_games=len(candidates_by_game), n_candidates=n_candidates,
            realizable_by_band=dict(realizable), pairs_by_band=dict(pairs),
            red_by_band=dict(red_tot), black_by_band=dict(black_tot),
            forced_bucket_min=forced_bucket_min, witness=witness)

    # Necessary checks (fast; name the binding constraint). NONE of these drives
    # feasible=True -- they only short-circuit clear INfeasibility with a reason.
    for band in bands:
        if realizable[band] < quota_per_band:
            return _report(False, f"band-capacity:{band} (realizable "
                           f"{realizable[band]} < quota {quota_per_band})", None)
    for band in bands:
        if pairs[band] * 2 < quota_per_band:
            return _report(False, f"side-aliasing:{band} (both-side pair-games "
                           f"{pairs[band]} < {quota_per_band // 2} needed for "
                           f"per-split side balance)", None)
    for bucket, forced in forced_bucket_min.items():
        if forced > bucket_cap:
            return _report(False, f"ply-bucket:{bucket} (forced {forced} > cap "
                           f"{bucket_cap:g})", None)

    # Constructive witness GOVERNS feasible=True (§11.2.3: necessary != sufficient).
    witness, binding = _build_witness(candidates_by_game, bands, quota_per_band,
                                      max_per_game, min_gap, side_tol, bucket_cap)
    if witness is None:
        return _report(False, binding, None)
    return _report(True, None, tuple(witness))


def preflight_source(records: List[Mapping[str, Any]], *,
                     stride: int, cap: int) -> PreflightReport:
    """I/O wrapper (the ONLY impure part -- kept thin). Read each
    `rec["replay_path"]`, build `candidates_by_game` via
    `build_candidates_by_game`, and run the pure `geometry_feasibility`. All
    feasibility logic lives in the pure core; this only does the file reads.
    """
    replays_by_game: Dict[Any, Mapping[str, Any]] = {}
    for rec in records:
        replay = json.loads(Path(rec["replay_path"]).read_text())
        replays_by_game[rec["game_idx"]] = replay
    candidates_by_game = build_candidates_by_game(
        replays_by_game, stride=stride, cap=cap)
    return geometry_feasibility(candidates_by_game)


# ---------------------------------------------------------------------------
# Stage-2 raw-policy geometry features (cheap prefilter input)
# ---------------------------------------------------------------------------

def _policy_features_from_priors(priors: List[float]) -> Dict[str, float]:
    """Policy-geometry features from a raw-policy prior distribution over the
    legal moves at one position (`priors`, aligned to `_teacher_infer`'s
    `legal` list; order-independent here). `n_legal = len(priors)`.

    normalized_entropy = H(prior) / log(n_legal): 1.0 at the flat/uniform
    prior, falling toward 0 as the distribution concentrates. `n_legal <= 1`
    is a degenerate case (never a real dev-corpus candidate: those all have
    n_legal >= 200) and returns normalized_entropy = 0.0 rather than dividing
    by log(0)/log(1).
    top1_prior / top4_mass / top8_mass: sum of the top 1/4/8 priors (top-k
    naturally capped at n_legal when n_legal < k -- `sorted(...)[:k]` on a
    shorter list just returns everything there is).
    """
    n_legal = len(priors)
    sorted_desc = sorted(priors, reverse=True)
    top1_prior = sorted_desc[0] if sorted_desc else 0.0
    top4_mass = sum(sorted_desc[:4])
    top8_mass = sum(sorted_desc[:8])
    if n_legal <= 1:
        normalized_entropy = 0.0
    else:
        h = -sum(p * math.log(p) for p in priors if p > 0)
        normalized_entropy = h / math.log(n_legal)
    return {
        "normalized_entropy": normalized_entropy,
        "top1_prior": top1_prior,
        "top4_mass": top4_mass,
        "top8_mass": top8_mass,
    }


# ---------------------------------------------------------------------------
# Forbidden-hash union + disjointness (design S2.3, edits 4/5)
# ---------------------------------------------------------------------------

def load_forbidden_hashes(paths: Iterable[str]) -> Set[str]:
    """Union of canonical position hashes across one or more manifest CSVs
    the dev corpus must stay disjoint from (selected-A union v16a, in the
    real run -- design S2.3). PRIMARY: a `canonical_position_sha1` column,
    read directly. FALLBACK (selected-A and v16a both predate Task 4's hash
    and carry neither): reconstruct from the shared probe-case schema
    (`replay_path`, `position_ply`, `side_to_move` -- the same
    REQUIRED_CASE_KEYS shape `position_probe_cases.load_csv_manifest` reads)
    via `position_state` + `canonical_state_sha1`, both pure (no MCTS/GPU/
    MLX). Read-only.
    """
    out: Set[str] = set()
    for p in paths:
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            has_hash_col = bool(reader.fieldnames) and (
                "canonical_position_sha1" in reader.fieldnames)
            for row in reader:
                if has_hash_col:
                    out.add(row["canonical_position_sha1"])
                else:
                    replay = json.loads(Path(row["replay_path"]).read_text())
                    state = position_state(replay, int(float(row["position_ply"])),
                                           row["side_to_move"])
                    out.add(canonical_state_sha1(state))
    return out


def assert_disjoint(dev_hashes: Iterable[str], forbidden: Iterable[str]) -> None:
    """Fail loud iff any `dev_hashes` entry collides with `forbidden` OR
    `dev_hashes` itself holds an internal duplicate; silent (returns `None`)
    when clean. Belt-and-suspenders check on the COMPLETED manifest (design
    S2.3): the scan loop already discards collisions/dupes as it goes, so a
    raise here means that per-candidate discard had a gap.
    """
    seen: Set[str] = set()
    dupes: Set[str] = set()
    for h in dev_hashes:
        if h in seen:
            dupes.add(h)
        seen.add(h)
    if dupes:
        raise ValueError(
            f"assert_disjoint: {len(dupes)} internal duplicate hash(es): "
            f"{sorted(dupes)[:5]}")
    collisions = seen & set(forbidden)
    if collisions:
        raise ValueError(
            f"assert_disjoint: {len(collisions)} dev hash(es) collide with "
            f"forbidden: {sorted(collisions)[:5]}")


# 2x per-band quota reserve pool (design S2 step 4): scan until every
# (role, band) cell holds >= RESERVE[role] anchor-confirmed candidates (or the
# corpus is exhausted), so assign_split/sample_dev_rows has slack for
# whole-game placement, the ply-gap/side-balance/bucket-cap filters, and any
# collision discards.
RESERVE: Dict[str, int] = {"target": 2 * TARGET_PER_BAND, "control": 2 * CONTROL_PER_BAND}


# ---------------------------------------------------------------------------
# Operator main(): load index -> two-stage scan -> anchor confirm -> hash/
# disjoint -> sample -> write. NOT run by this task; loads a real checkpoint
# and runs 400-sim MCTS.
# ---------------------------------------------------------------------------

SOURCE_CORPUS_ID = "0379_vs_calib020_0001_800g_w4_seed20116"
DEFAULT_SOURCE_JSONL = (
    "logs/eval/0379_vs_calib020_0001_800g_w4_seed20116_replay_games.jsonl")
DEFAULT_OUT = "logs/eval/fpu_dev_corpus/dev_corpus_manifest.csv"
DEFAULT_SAMPLE_SEED = 20260711
DEFAULT_STRIDE = 4
DEFAULT_CAP = 6
ANCHOR_SIMS = 400
# The design doesn't pin an anchor-search RNG scheme; this XORs a fixed base
# with (game_idx, ply) for a deterministic, reproducible per-position seed,
# mirroring the codebase's row_seed idiom (e.g.
# build_mcts_root_retention_manifest.row_seed's `base ^ game_idx ^ ply`).
ANCHOR_SEED_BASE = 20260711

# Recorded per row (design S2, exact order).
MANIFEST_FIELDNAMES = [
    "source_corpus_id", "game_idx", "position_ply", "side", "game_result",
    "total_plies", "n_legal", "root_value_stm", "normalized_entropy",
    "top1_prior", "top4_mass", "top8_mass", "canonical_position_sha1",
    "ply_bucket", "branching_band", "split", "role",
]


def load_game_index(jsonl_path: str) -> List[dict]:
    """Read the seed20116 replay-eval JSONL INDEX (one game record per line):
    game_idx, n_moves, winner, replay_path. Winner-null games are KEPT
    (game_result="unknown") -- state-cap/unknown games are valuable,
    search-stressed samples (design S2: "include state-cap/unknown games").
    Mirrors build_v16a_neutral_position_manifest.load_game_index. Returns
    records sorted by game_idx.
    """
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            g = json.loads(line)
            winner = g.get("winner")
            if winner not in ("red", "black"):
                winner = "unknown"
            records.append({
                "game_idx": int(g["game_idx"]),
                "n_moves": int(g["n_moves"]),
                "winner": winner,
                "replay_path": g["replay_path"],
            })
    records.sort(key=lambda r: r["game_idx"])
    return records


def _anchor_seed(game_idx: int, ply: int) -> int:
    return ANCHOR_SEED_BASE ^ int(game_idx) ^ int(ply)


def _build_anchor_search_fn(checkpoint: str, eval_batch_size: int,
                            stall_flush_sims: int):
    """Load ONE evaluator + build the fpu-off 400-sim anchor search_fn.
    Checkpoint/GPU/MLX work -- only ever called from `main()`. Mirrors
    diagnose_fpu_sweep._search_fns/_make_search_fn, simplified to the single
    `absolute_off` (fpu_policy_mass_reduction=None) config this task needs;
    same evaluator is reused for the raw-policy forward pass (design S2: "the
    fpu-off calib020_0001 anchor + raw policy" -- one network, both roles).
    """
    from .eval_runner import EvalConfig, cfg_from, _default_evaluator_factory
    from .mcts import MCTS
    evaluator = _default_evaluator_factory(checkpoint)
    base_cfg = cfg_from(EvalConfig(mcts_sims=ANCHOR_SIMS,
                                   mcts_eval_batch_size=eval_batch_size,
                                   mcts_stall_flush_sims=stall_flush_sims))
    # Explicit even though it's already the default -- this IS the frozen
    # fpu-off anchor config (design S2 step 3 / S5 step 0's `absolute_off`).
    cfg = dataclasses.replace(base_cfg, fpu_policy_mass_reduction=None)

    def search_fn(state, seed):
        return MCTS(evaluator, cfg, random.Random(seed)).search_with_root(
            state, add_noise=False)

    # Also return the frozen anchor cfg so main() can record its FULL effective
    # MCTS config in the manifest provenance (design §12.5) from the single
    # source of truth, rather than reconstructing it.
    return evaluator, search_fn, cfg


def _manifest_row(r: dict) -> dict:
    """Project one accumulated/sampled candidate row (sampler key names --
    `ply`, `band`, `canonical_sha1`, Task 5's `sample_dev_rows` contract) to
    the frozen MANIFEST_FIELDNAMES row (design S2 key names)."""
    return {
        "source_corpus_id": r["source_corpus_id"],
        "game_idx": r["game_idx"],
        "position_ply": r["ply"],
        "side": r["side"],
        "game_result": r["game_result"],
        "total_plies": r["total_plies"],
        "n_legal": r["n_legal"],
        "root_value_stm": r["root_value_stm"],
        "normalized_entropy": r["normalized_entropy"],
        "top1_prior": r["top1_prior"],
        "top4_mass": r["top4_mass"],
        "top8_mass": r["top8_mass"],
        "canonical_position_sha1": r["canonical_sha1"],
        "ply_bucket": r["ply_bucket"],
        "branching_band": r["band"],
        "split": r["split"],
        "role": r["role"],
    }


def _scan_two_stage(records: List[dict], *, evaluator, search_fn,
                    forbidden: Set[str], stride: int, cap: int,
                    sample_seed: int) -> Tuple[List[dict], dict]:
    """Design S2 two-stage scan + reserve accumulation + sample, with the
    "shortfall -> keep scanning -> re-sample" retry (plan Task 6 step 3).

    Per source game (ascending game_idx -- `records` is already sorted):
    stage 1 `enumerate_candidate_plies` (cheap, stored/reconstructed
    n_legal); stage 2 raw-policy prefilter (`_teacher_infer`, one forward
    pass, then `raw_policy_role`); stage 3 anchor confirm (400-sim fpu-off
    `search_fn`, `anchor_eligible`) -- run ONLY on stage-2 survivors, and
    skipped once a cell already holds its current target (saving sims).
    `canonical_state_sha1` collisions with `forbidden` or an already-kept
    hash are discarded and scanning continues (design S2.3 edit 4).

    Accumulates confirmed rows per (role, band) until every cell reaches
    RESERVE[role] or the corpus is exhausted, then calls
    `assign_split`+`sample_dev_rows`. If sampling raises a shortfall (the
    reserve pool met its raw quota-counts but the gap/side-balance/
    bucket-cap/dedup filters couldn't realize it), the per-role target widens
    by another RESERVE increment and scanning resumes from exactly where the
    shared game iterator left off (never re-scanned from the start) --
    repeat until sampling succeeds or the corpus is truly exhausted (raises).
    """
    from .build_teacher_calibration_manifest import _teacher_infer

    confirmed_by_cell: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    kept_hashes: Set[str] = set()
    game_iter = iter(records)
    exhausted = False
    target = dict(RESERVE)

    def cells_filled() -> bool:
        return all(len(confirmed_by_cell[(role, band)]) >= target[role]
                  for role in RESERVE for band in BANDS)

    def scan_more() -> None:
        nonlocal exhausted
        for rec in game_iter:
            replay = json.loads(Path(rec["replay_path"]).read_text())
            for ply in enumerate_candidate_plies(replay, stride=stride, cap=cap):
                side = side_to_move_for_ply(ply)
                state = position_state(replay, ply, side)
                legal, priors, _raw_value = _teacher_infer(state, evaluator)
                n_legal = len(legal)
                band = band_of(n_legal)
                if band is None:
                    continue
                feats = _policy_features_from_priors(priors)
                role = raw_policy_role(feats["normalized_entropy"], feats["top1_prior"])
                if role is None:
                    continue
                if len(confirmed_by_cell[(role, band)]) >= target[role]:
                    continue   # cell already at its current target -- skip the 400-sim search
                _counts, root_value_stm, root = search_fn(
                    state, _anchor_seed(rec["game_idx"], ply))
                if root.visit_count != ANCHOR_SIMS:
                    raise RuntimeError(
                        f"anchor confirm game_idx={rec['game_idx']} ply={ply}: "
                        f"{root.visit_count} sims != {ANCHOR_SIMS}")
                if not anchor_eligible(root_value_stm):
                    continue
                sha1 = canonical_state_sha1(state)
                if sha1 in forbidden or sha1 in kept_hashes:
                    continue   # collision discard (design S2.3 edit 4)
                kept_hashes.add(sha1)
                confirmed_by_cell[(role, band)].append({
                    "source_corpus_id": SOURCE_CORPUS_ID,
                    "game_idx": rec["game_idx"], "ply": ply, "side": side,
                    "game_result": rec["winner"], "total_plies": rec["n_moves"],
                    "n_legal": n_legal, "root_value_stm": root_value_stm,
                    "normalized_entropy": feats["normalized_entropy"],
                    "top1_prior": feats["top1_prior"],
                    "top4_mass": feats["top4_mass"], "top8_mass": feats["top8_mass"],
                    "canonical_sha1": sha1, "ply_bucket": ply_bucket_of(ply),
                    "band": band, "role": role,
                })
            if cells_filled():
                break
        else:
            exhausted = True

    scan_more()
    while True:
        flat = [row for rows in confirmed_by_cell.values() for row in rows]
        try:
            return sample_dev_rows(flat, seed=sample_seed)
        except ValueError:
            if exhausted:
                raise
            for role in target:
                target[role] += RESERVE[role]
            scan_more()


def write_manifest(rows: List[dict], out_csv: str) -> None:
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


def corpus_provenance(*, source_jsonl: Optional[str], checkpoint: Optional[str],
                      base_mcts_config: Optional[dict]) -> dict:
    """Evidence-grade provenance for the dev-corpus manifest (design §12.5):
    source-file BYTE hashes, the source-index sha1 + a deterministic replay-DATA
    hash (contents, not paths), the anchor's checkpoint identity + FULL MCTS
    config, the explicit `add_noise=False`, and runtime identity. Pure /
    stdlib-only (via `fpu_provenance`): it reads the source index + replays but
    NO MCTS/GPU/MLX. `replay_paths` come from the same `load_game_index` the
    scan used, so the hash covers exactly the games the corpus was built from."""
    replay_paths = ([r["replay_path"] for r in load_game_index(source_jsonl)]
                    if source_jsonl else [])
    return {
        "source_file_sha1s": fpu_provenance.source_file_sha1s(_CORPUS_SOURCES),
        "source_index_sha1": fpu_provenance.file_sha1(source_jsonl),
        "replay_data_sha1": fpu_provenance.replay_data_sha1(replay_paths),
        "checkpoint_identity": (
            f"{Path(checkpoint).name}:{fpu_provenance.file_sha1(checkpoint)}"
            if checkpoint else "none"),
        "base_mcts_config": base_mcts_config,
        "add_noise": False,
        "runtime_provenance": fpu_provenance.runtime_provenance(),
    }


def write_meta(out_csv: str, meta: dict) -> None:
    """Write `<out_csv>.meta.json`, ENRICHED with an evidence-grade `provenance`
    block (design §12.5) computed from the meta's own source_jsonl / checkpoint /
    base_mcts_config -- so the corpus manifest carries the same run-identity the
    diagnostic fingerprint's selection-context does. `provenance` is DERIVED, so
    it never clobbers a caller key."""
    enriched = dict(meta)
    enriched["provenance"] = corpus_provenance(
        source_jsonl=meta.get("source_jsonl"),
        checkpoint=meta.get("checkpoint"),
        base_mcts_config=meta.get("base_mcts_config"))
    Path(str(out_csv) + ".meta.json").write_text(json.dumps(enriched, indent=2))


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description="Build the FPU (policy-mass) 240-row development corpus: "
                    "two-stage scan (cheap n_legal enumeration -> raw-policy "
                    "prefilter) over the seed20116 source games, 400-sim "
                    "fpu-off anchor confirm on survivors, complete-state hash "
                    "+ disjointness vs selected-A/v16a, then the frozen "
                    "180-target/60-control sample. OPERATOR phase: loads a "
                    "real checkpoint and runs MCTS -- design S2.")
    ap.add_argument("--source-jsonl", default=DEFAULT_SOURCE_JSONL)
    ap.add_argument("--checkpoint", default=None,
                    help="defaults to diagnose_fpu_sweep.DEFAULT_CHECKPOINT")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    ap.add_argument("--forbidden-manifest", action="append", default=None,
                    help="CSV(s) of already-used positions to exclude "
                         "(repeatable). selected-A + v16a are added unless "
                         "--no-default-forbidden.")
    ap.add_argument("--no-default-forbidden", action="store_true")
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--eval-batch-size", type=int, default=14)
    ap.add_argument("--stall-flush-sims", type=int, default=48)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    # DEFAULT_A_MANIFEST / v16a's DEFAULT_OUT are each the single source of
    # truth for their exclusion set (mirrors build_v16a_neutral_position_
    # manifest's own DEFAULT_A_MANIFEST import); deferred (pulls MCTS/eval_
    # runner) so the pure-test import path stays MCTS/GPU/MLX-free.
    from .diagnose_fpu_sweep import DEFAULT_A_MANIFEST, DEFAULT_CHECKPOINT
    from .build_v16a_neutral_position_manifest import (
        DEFAULT_OUT as DEFAULT_V16A_MANIFEST)

    args = _parse_args(argv)
    checkpoint = args.checkpoint or DEFAULT_CHECKPOINT

    forbidden_paths = list(args.forbidden_manifest or [])
    if not args.no_default_forbidden:
        forbidden_paths += [DEFAULT_A_MANIFEST, DEFAULT_V16A_MANIFEST]
    forbidden = load_forbidden_hashes(forbidden_paths) if forbidden_paths else set()
    print(f"[fpu-dev-corpus] {len(forbidden)} forbidden hash(es) from "
          f"{len(forbidden_paths)} manifest(s)")

    records = load_game_index(args.source_jsonl)
    print(f"[fpu-dev-corpus] {len(records)} source games from {args.source_jsonl}")

    # Pure replay-geometry feasibility preflight (design §11.4). Proves from
    # n_legal/ply/side ALONE that this (source, enumeration) pair can jointly
    # satisfy the four structural constraints. Runs BEFORE the evaluator loads,
    # so an infeasible source hard-stops cheaply instead of deep in the scan.
    report = preflight_source(records, stride=args.stride, cap=args.cap)
    print(report.format())
    if not report.feasible:
        print(f"[fpu-dev-corpus] PREFLIGHT INFEASIBLE -- binding constraint: "
              f"{report.binding_constraint}. Stopping BEFORE evaluator load "
              f"(design §11.4 stop-don't-retune).")
        return 2
    print("[fpu-dev-corpus] preflight OK -- geometry can jointly satisfy the "
          "four structural constraints; loading evaluator.")

    evaluator, search_fn, anchor_cfg = _build_anchor_search_fn(
        checkpoint, args.eval_batch_size, args.stall_flush_sims)

    rows, stats = _scan_two_stage(
        records, evaluator=evaluator, search_fn=search_fn, forbidden=forbidden,
        stride=args.stride, cap=args.cap, sample_seed=args.sample_seed)

    final_hashes = [r["canonical_sha1"] for r in rows]
    assert_disjoint(final_hashes, forbidden)

    manifest_rows = [_manifest_row(r) for r in rows]
    write_manifest(manifest_rows, args.out)
    write_meta(args.out, {
        "source_jsonl": args.source_jsonl, "source_corpus_id": SOURCE_CORPUS_ID,
        "checkpoint": checkpoint, "sample_seed": args.sample_seed,
        "stride": args.stride, "cap": args.cap,
        "forbidden_manifests": forbidden_paths, "n_forbidden_hashes": len(forbidden),
        "n_rows": len(manifest_rows), "stats": stats,
        "fieldnames": MANIFEST_FIELDNAMES,
        "base_mcts_config": dataclasses.asdict(anchor_cfg),
    })
    print(f"[fpu-dev-corpus] wrote {len(manifest_rows)} rows -> {args.out} "
          f"(+ .meta.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
