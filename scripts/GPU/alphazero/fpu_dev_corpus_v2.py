"""FPU (policy-mass) v2 phase-primary development-corpus builder.

Frozen design ref: docs/superpowers/specs/2026-07-12-fpu-dev-corpus-v2-phase-design.md
v2 plan Task 1 ("v2 constants + classifiers") -- this task.

Successor to `build_fpu_dev_corpus.py` (v1), which stratified by branching
BAND with a <=50%-ply-bucket cap -- proved mathematically impossible on the
project's fixed 24x24 board (design Sec 0; encoded permanently as
`tests/test_fpu_dev_corpus_v2.py::test_v1_bands_impossible_on_24board`,
Task 0). v2 makes game PHASE the primary stratum (design Sec 1.2) and
demotes branching band to a recorded covariate plus explicit late coverage
floors (design Sec 1.3) -- band is no longer an independent quota stratum.

=============================================================================
PURE SECTION (Tasks 1-3) -- constants + pure functions ONLY.
=============================================================================
Mirrors build_fpu_dev_corpus.py's own PURE SECTION / OPERATOR SHELL split.
Everything in this file is pure so far: plain-stdlib constants, classifiers,
the proposal enumerator and the phase-stratified sampler. NO MCTS / evaluator
/ GPU / MLX / heavy-numpy imports, no I/O, no argument parsing. Later tasks
append BELOW this section, in order:
  Task 4: the v2 geometric preflight.
  Task 5: the operator `screen` stage (evaluator/MCTS; lazy heavy imports).
  Task 6: the pure `select` stage + config loader + `main`.
Keep this section cleanly separated and importable without ever triggering a
GPU/MLX import -- any future heavy import goes lazily inside the Task-5
operator functions, exactly as build_fpu_dev_corpus.py's own `main` /
`_build_anchor_search_fn` / evaluator plumbing do.

What this section does
-----------------------
Task 1: frozen phase-primary constants (design Sec 1.2 / 1.3 / 1.5) plus the
one v2 classifier, `proposal_cell_of`, which maps a (phase, n_legal) pair to
its PROPOSAL_CELLS membership -- or to a cell deliberately NOT in
PROPOSAL_CELLS (a `("late", None)` sentinel for sub-200 late positions is
intentionally ineligible; see `proposal_cell_of`'s own docstring).
Task 2: `enumerate_v2_proposals` -- the phase-aware side-opposed proposal
enumerator (no global stride).
Task 3: `sample_v2_rows` (+ `assign_split_v2`) -- the phase-stratified
whole-game sampler that realizes SPLIT_ALLOC_V2 EXACTLY under a GLOBAL
<=MAX_PER_GAME rule and the hard LATE_TARGET_FLOORS.
DRY: reuses `band_of` / `ply_bucket_of` / `side_to_move_for_ply` /
`per_ply_n_legal` / `_first_gap_pair` / `_choose_positions`, and re-exports the
shared v1 MIN_PLY_GAP / MAX_PER_GAME / SIDE_TOL / SPLITS constants, from
`build_fpu_dev_corpus` rather than restating them -- see each import's inline
comment for why.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

# Deliberately-shared v1 frozen constants (identical semantics in v2):
# MIN_PLY_GAP = the >=12-ply side-opposed-pair gap; MAX_PER_GAME = the <=2
# SELECTED rows per game cap (global, across all proposal cells, enforced
# by the v2 sampler -- Task 3); SIDE_TOL = the per-split |red-black| side
# balance tolerance; SPLITS = the ("tuning", "frozen_check") split vocabulary
# and its deterministic fill order. Imported (not restated) so a v1 drift is
# felt here too; pinned in tests/test_fpu_dev_corpus_v2.py.
#
# Task-2 additions (the enumerator below): `per_ply_n_legal` (per-ply legal
# count, incl. its reconstruction fallback), `ply_bucket_of` (phase), and
# `side_to_move_for_ply` (red on even plies) are the same pure per-ply
# primitives v1's own scan uses; `_first_gap_pair` is v1's deterministic
# earliest-satisfying side-opposed-pair search (build_fpu_dev_corpus.py:578).
#
# Task-3 addition (the sampler below): `_choose_positions`
# (build_fpu_dev_corpus.py:259) is v1's per-game pick rule -- take_n == 1
# steers toward the split's deficit side; take_n >= 2 walks the earliest
# >=gap-apart chain -- semantics v2 needs UNCHANGED, so it is imported, never
# copied. (v1's `_greedy_assign` / `assign_split` / `sample_dev_rows` are NOT
# reusable: they close over v1's SPLIT_ALLOC/CELL_ORDER/bucket-cap globals and
# apply the <=MAX_PER_GAME cap PER CELL, where v2's rule is GLOBAL -- see
# `_greedy_assign_v2` / `sample_v2_rows`.)
# All reused verbatim (DRY) -- never reimplemented here.
from .build_fpu_dev_corpus import (
    MAX_PER_GAME,
    MIN_PLY_GAP,
    SIDE_TOL,
    SPLITS,
    _choose_positions,
    _first_gap_pair,
    band_of,
    per_ply_n_legal,
    ply_bucket_of,
    side_to_move_for_ply,
)

# ---------------------------------------------------------------------------
# Frozen constants (verbatim from the design + Task-1 brief)
# ---------------------------------------------------------------------------

# The four phase strata = v1's ply buckets (design Sec 1.2), now PRIMARY
# rather than a secondary covariate under a <=50%-cap. Values intentionally
# match v1's ply_bucket_of bucket vocabulary one-for-one (pinned against the
# real ply_bucket_of in tests/test_fpu_dev_corpus_v2.py so a v1 rename can't
# silently drift here), but this constant is NOT imported/re-exported from
# build_fpu_dev_corpus: v2's interface names it PHASES (design Sec 1.2's own
# vocabulary) to signal its new role as the PRIMARY stratum.
PHASES: Tuple[str, str, str, str] = ("opening", "early_mid", "midgame", "late")

# Frozen phase-primary split allocation (design Sec 1.2), keyed (role, phase)
# like v1's (role, band) SPLIT_ALLOC. Each phase is allocated an IDENTICAL 45
# target (30 tuning / 15 frozen_check) + 15 control (10 tuning / 5
# frozen_check) -- 45 and 15 both divide evenly by 4 phases, so (unlike v1's
# odd-quota control bands: 13/7, 13/7, 14/6) there is no per-phase asymmetry.
#   totals: target 180 (45x4), control 60 (15x4);
#           tuning 160 ((30+10)x4), frozen_check 80 ((15+5)x4); grand 240.
SPLIT_ALLOC_V2: Dict[Tuple[str, str], Dict[str, int]] = {
    ("target", "opening"): {"tuning": 30, "frozen_check": 15},
    ("control", "opening"): {"tuning": 10, "frozen_check": 5},
    ("target", "early_mid"): {"tuning": 30, "frozen_check": 15},
    ("control", "early_mid"): {"tuning": 10, "frozen_check": 5},
    ("target", "midgame"): {"tuning": 30, "frozen_check": 15},
    ("control", "midgame"): {"tuning": 10, "frozen_check": 5},
    ("target", "late"): {"tuning": 30, "frozen_check": 15},
    ("control", "late"): {"tuning": 10, "frozen_check": 5},
}

# Total manifest size implied by SPLIT_ALLOC_V2 (240) -- DERIVED, mirroring
# build_fpu_dev_corpus.CORPUS_SIZE (build_fpu_dev_corpus.py:115), NOT a
# hard-coded literal.
CORPUS_SIZE = sum(a["tuning"] + a["frozen_check"] for a in SPLIT_ALLOC_V2.values())

# Late-target coverage floors (design Sec 1.3): among the 45 late TARGET
# rows, require >=12 with n_legal in 300-399 and >=12 with n_legal in
# 200-299. These are COVERAGE FLOORS, not an independent selection stratum
# and not a rate-gate denominator -- they exist purely so the v1
# late-collapse geometry doesn't vanish now that phase (not band) is
# primary.
LATE_TARGET_FLOORS: Dict[str, int] = {"b300_399": 12, "b200_299": 12}

# v2-specific: caps PROPOSALS per (game, proposal-cell) at the enumerator
# stage (Task 2). Deliberately DISTINCT from MAX_PER_GAME (imported above),
# which caps SELECTED rows per game GLOBALLY across all cells at the
# sampler stage (Task 3) -- both happen to equal 2, but they gate different
# rules at different pipeline stages, so they are kept as separate names
# rather than reusing one constant for both.
MAX_PER_CELL_PER_GAME = 2

# Deterministic proposal-cell order (design Sec 1.5): the three non-late
# phases (band None), in PHASES order, then the three late bands in
# DESCENDING branching order (b400_plus, b300_399, b200_299) -- pinned
# exactly per the v2 controller resolution. This order is load-bearing for
# later tasks: Task 2's enumerator iterates PROPOSAL_CELLS in this order.
PROPOSAL_CELLS: List[Tuple[str, Optional[str]]] = [
    (p, None) for p in PHASES if p != "late"
] + [
    ("late", "b400_plus"),
    ("late", "b300_399"),
    ("late", "b200_299"),
]


# ---------------------------------------------------------------------------
# Proposal-cell classifier (pure)
# ---------------------------------------------------------------------------

def proposal_cell_of(phase: str, n_legal: int) -> Tuple[str, Optional[str]]:
    """Map a (phase, n_legal) pair to its PROPOSAL_CELLS membership.

    Non-late phases ignore n_legal entirely: returns (phase, None)
    regardless of branching. "late" splits by the REAL `band_of(n_legal)`
    (imported from build_fpu_dev_corpus, not reimplemented): returns
    ("late", "b400_plus" | "b300_399" | "b200_299" | None).

    A late position with n_legal < 200 -- i.e. where `band_of(n_legal)` is
    None -- therefore returns ("late", None), which is INTENTIONALLY NOT a
    member of PROPOSAL_CELLS: that is how sub-200 positions become
    ineligible for late-cell proposals. The Task-2 enumerator additionally
    enforces n_legal >= 200 for every cell (belt-and-suspenders with this
    sentinel, not a replacement for it).
    """
    if phase != "late":
        return (phase, None)
    return (phase, band_of(n_legal))


# ---------------------------------------------------------------------------
# Phase-aware proposal enumerator (pure) -- Task 2
# ---------------------------------------------------------------------------

def enumerate_v2_proposals(replay: Mapping[str, Any]) -> List[dict]:
    """Phase-aware side-opposed proposal pairs for one replay (design Sec
    1.5). Proposals are CANDIDATES for the later operator `screen` stage
    (Task 5) -- not the final manifest; Task 3's sampler selects the final
    rows from screened proposals.

    Per `PROPOSAL_CELLS` cell (in that fixed, load-bearing order: the three
    non-late phases, band `None`, then the three late bands descending), a
    ply is eligible for the cell iff, reading per-ply legal counts via v1's
    own `per_ply_n_legal` (never reimplemented -- also handles its
    reconstruction fallback, where some plies are `None`):
      - its n_legal is not None and >= 200 (an independent guard, not solely
        reliant on `proposal_cell_of`'s own `("late", None)` sub-200
        sentinel -- "belt-and-suspenders", per that function's docstring);
      - `proposal_cell_of(ply_bucket_of(ply), n_legal) == cell` -- the SAME
        Task-1 classifier used to build `PROPOSAL_CELLS` itself, so phase
        and (for late cells only) band membership can never drift from it.

    From a cell's eligible plies, `_first_gap_pair` (v1's deterministic
    earliest-satisfying-pair search over ascending-ply reds/blacks,
    `build_fpu_dev_corpus.py:578` -- imported, not copied) selects ONE
    side-opposed pair (`side_to_move_for_ply`: red on even plies, black on
    odd) at least `MIN_PLY_GAP` plies apart. Because an opposed pair's ply
    difference is always odd, this floors the smallest realizable gap at
    `MIN_PLY_GAP + 1`. No valid pair means the cell contributes 0 proposals
    -- never a lone unpaired one -- so every cell yields exactly 0 or
    `MAX_PER_CELL_PER_GAME` (2) rows; one game can thus yield up to 12
    proposals across the 6 cells. (The GLOBAL <=2-per-game rule is a Task-3
    SAMPLER concern over SELECTED rows, never enforced here.)

    Each proposal dict carries exactly `game_idx, ply, side, phase, n_legal,
    band, proposal_cell`. `band` is ALWAYS the real `band_of(n_legal)` -- the
    recorded branching covariate -- so e.g. an `("opening", None)` cell's
    proposals still record their real band (typically `"b400_plus"` on this
    board), never the cell's own `None` band component. `phase` is the
    cell's phase (`cell[0]`), `== ply_bucket_of(ply)`.

    Fully deterministic: cells in `PROPOSAL_CELLS` order, and within a cell
    the pair in ascending ply -- NOT necessarily red-then-black, since
    `_first_gap_pair` can return either side first depending on which plies
    are actually eligible.
    """
    game_idx = replay["game_idx"]
    n_legal_by_ply = per_ply_n_legal(replay)

    proposals: List[dict] = []
    for cell in PROPOSAL_CELLS:
        # Ascending by construction (a single forward pass over `ply` in
        # increasing order), so reds/blacks need no re-sort before
        # `_first_gap_pair`, which requires each list pre-sorted by ply.
        reds: List[dict] = []
        blacks: List[dict] = []
        for ply, n_legal in enumerate(n_legal_by_ply):
            if n_legal is None or n_legal < 200:
                continue
            # The single canonical eligibility test (DRY): matches `cell`
            # iff ply_bucket_of(ply) is the cell's phase AND -- late cells
            # only -- band_of(n_legal) is the cell's band. Never restated.
            if proposal_cell_of(ply_bucket_of(ply), n_legal) != cell:
                continue
            side = side_to_move_for_ply(ply)
            row = {
                "game_idx": game_idx,
                "ply": ply,
                "side": side,
                "phase": cell[0],
                "n_legal": n_legal,
                "band": band_of(n_legal),
                "proposal_cell": cell,
            }
            (reds if side == "red" else blacks).append(row)

        pair = _first_gap_pair(reds, blacks, MIN_PLY_GAP)
        if pair is None:
            continue   # no valid side-opposed pair -- 0 proposals, never 1
        cell_proposals = sorted(pair, key=lambda row: row["ply"])
        assert len(cell_proposals) == MAX_PER_CELL_PER_GAME
        proposals.extend(cell_proposals)

    return proposals


# ---------------------------------------------------------------------------
# Phase-stratified sampler (pure) -- Task 3
# ---------------------------------------------------------------------------
# Selects the final 240-row manifest from the KEPT rows of the (later, operator)
# `screen` stage -- plain dicts carrying at least `game_idx, role, phase, band,
# side, ply, canonical_sha1`. Structurally mirrors v1's `assign_split` +
# `sample_dev_rows` (whole-game split assignment, then an exact-or-raise
# round-robin), with three deliberate v2 differences:
#
#   1. Cells are (role, PHASE), not (role, band) -- band is now a recorded
#      covariate, constrained only by the late coverage FLOORS.
#   2. <=MAX_PER_GAME is GLOBAL. v1 re-evaluates `take_n = min(MAX_PER_GAME,
#      ...)` per cell, so one v1 game can give 2 rows to EACH of several cells;
#      v2 allows <=2 SELECTED rows per game across ALL cells and BOTH splits
#      combined (design Sec 1.2). Consequently >=MIN_PLY_GAP is likewise global
#      here: two rows taken from one game in DIFFERENT cells must still be >=12
#      plies apart (v1 only ever compared rows within one cell).
#   3. v1's <=50% ply-bucket cap is DROPPED -- subsumed, since each phase is
#      exactly 60/240 = 25% by construction (design Sec 1.2). There is no
#      `bucket_cap` parameter and no `bucket_count` stat.
#
# The floors (design Sec 1.3) are a COVERAGE requirement over the (target, late)
# cell's 45 rows COMBINED across splits -- not a stratum, not extra quota. A
# naive earliest-first fill takes the abundant b400_plus late rows and misses
# them, so that one cell gets a floor-satisfaction pass (below), and the floors
# are re-verified on the SELECTED rows before returning.

# Deterministic (role, phase) cell order = SPLIT_ALLOC_V2 insertion order,
# mirroring v1's CELL_ORDER (build_fpu_dev_corpus.py:112).
CELL_ORDER_V2: List[Tuple[str, str]] = list(SPLIT_ALLOC_V2.keys())

# The ONE allocation cell LATE_TARGET_FLOORS constrains ("among the 45 late
# TARGET rows" -- design Sec 1.3). Asserted to be a real SPLIT_ALLOC_V2 cell, and
# the floors asserted to fit inside it, so a PHASES/role rename or a floor bump
# can never silently orphan (or over-subscribe) the floors.
LATE_TARGET_CELL: Tuple[str, str] = ("target", "late")
assert LATE_TARGET_CELL in SPLIT_ALLOC_V2, LATE_TARGET_CELL
assert sum(LATE_TARGET_FLOORS.values()) <= sum(
    SPLIT_ALLOC_V2[LATE_TARGET_CELL].values()), LATE_TARGET_FLOORS


def _greedy_assign_v2(games_profile, seed, attempt) -> Optional[Dict[Any, str]]:
    """One deterministic greedy pass (v1 `_greedy_assign`'s shape). Returns
    {game_idx: split} if it satisfies every per-(role, phase, split) quota, else
    None.

    Each WHOLE game is placed in the split whose still-unmet quotas it fills most;
    ties break toward the split with the larger total remaining need, then toward
    tuning. Games are visited in a seed-shuffled order (attempt 0) or its
    deterministic reverse (attempt 1, the secondary-ordering retry).

    A game's contribution is capped at MAX_PER_GAME in TOTAL across its cells --
    NOT per cell, as v1 does -- because that is v2's actual selection rule. Both
    the `realizable` scoring and the `need` decrement spend one shared per-game
    budget over the game's cells in CELL_ORDER_V2 order, which is exactly the
    order (and the greedy "as many as this cell still needs" rule) the round-robin
    in `sample_v2_rows` will use. Crediting a multi-cell game with MAX_PER_GAME in
    EVERY cell would over-state realizable capacity, drive `need` to zero early,
    and hand back an assignment the round-robin cannot fill -- a spurious
    final-manifest shortfall.
    """
    rng = random.Random(seed * 1_000_003 + attempt)
    order = sorted(games_profile)
    rng.shuffle(order)
    if attempt == 1:
        order = order[::-1]

    need = {cell: dict(alloc) for cell, alloc in SPLIT_ALLOC_V2.items()}
    assign: Dict[Any, str] = {}
    for gi in order:
        prof = games_profile[gi]
        cells = [c for c in CELL_ORDER_V2 if c in prof]

        def realizable(split, _cells=cells, _prof=prof):
            """Rows this WHOLE game could actually add to `split`: the greedy
            cell-by-cell spend of ONE shared MAX_PER_GAME budget."""
            budget = MAX_PER_GAME
            total = 0
            for c in _cells:
                if budget <= 0:
                    break
                n = min(_prof[c], need[c][split], budget)
                total += n
                budget -= n
            return total

        u_t, u_f = realizable("tuning"), realizable("frozen_check")
        if u_t > u_f:
            split = "tuning"
        elif u_f > u_t:
            split = "frozen_check"
        else:
            tot_t = sum(need[c]["tuning"] for c in cells)
            tot_f = sum(need[c]["frozen_check"] for c in cells)
            split = "tuning" if tot_t >= tot_f else "frozen_check"

        assign[gi] = split
        budget = MAX_PER_GAME
        for c in cells:
            if budget <= 0:
                break
            spend = min(prof[c], need[c][split], budget)
            need[c][split] -= spend
            budget -= spend

    if all(v == 0 for cell in need for v in need[cell].values()):
        return assign
    return None


def assign_split_v2(games_profile: Mapping[Any, Mapping[Tuple[str, str], int]],
                    seed: int) -> Dict[Any, str]:
    """Assign each WHOLE game to "tuning" or "frozen_check" so every
    per-(role, phase, split) SPLIT_ALLOC_V2 quota is satisfiable.

    `games_profile`: {game_idx: {(role, phase): n_available_kept_rows}}.

    The two capacity checks below are NECESSARY conditions only -- each is a true
    UPPER BOUND on what the selection can realize, so falling short of demand
    PROVES infeasibility and names it cheaply, while passing them proves nothing.
    (Once a game's rows span cells, no per-cell sum can be exact: its <=2-row
    global budget is claimable by any one of them.) The exact-or-raise round-robin
    in `sample_v2_rows` remains the authority on feasibility.

    Raises ValueError if a cell's capacity is below its demand, if the whole pool
    cannot yield CORPUS_SIZE rows under the global <=MAX_PER_GAME rule, or if
    neither the primary nor the deterministic secondary ordering yields a
    quota-satisfying assignment.
    """
    # (1) Per-cell upper bound (v1's `assign_split` check): a game can never give
    # a cell more than min(its rows there, MAX_PER_GAME). Over-states capacity for
    # a multi-cell game (whose <=2-row budget is counted once per cell), hence
    # upper bound / necessary only.
    capacity: Counter = Counter()
    for prof in games_profile.values():
        for cell, n in prof.items():
            if cell in SPLIT_ALLOC_V2:
                capacity[cell] += min(n, MAX_PER_GAME)
    for cell, alloc in SPLIT_ALLOC_V2.items():
        demand = alloc["tuning"] + alloc["frozen_check"]
        have = capacity.get(cell, 0)
        if have < demand:
            raise ValueError(
                f"assign_split_v2: cell {cell} capacity {have} < demand {demand}")

    # (2) GLOBAL upper bound -- the v2-specific one, and the check a per-cell-only
    # accounting cannot express: because <=MAX_PER_GAME is global across ALL cells
    # in v2, the whole corpus can never exceed sum_g min(rows(g), MAX_PER_GAME)
    # rows, however those rows are distributed. (Under v1's PER-cell cap a game's
    # global contribution was unbounded, so v1 had no such bound to check.)
    global_capacity = sum(
        min(sum(n for cell, n in prof.items() if cell in SPLIT_ALLOC_V2),
            MAX_PER_GAME)
        for prof in games_profile.values())
    if global_capacity < CORPUS_SIZE:
        raise ValueError(
            f"assign_split_v2: global capacity {global_capacity} < corpus size "
            f"{CORPUS_SIZE} under the global <=MAX_PER_GAME ({MAX_PER_GAME}) "
            f"per-game rule ({len(games_profile)} games)")

    for attempt in range(2):
        result = _greedy_assign_v2(games_profile, seed, attempt)
        if result is not None:
            return result
    raise ValueError(
        "assign_split_v2: no deterministic ordering satisfied the split quotas")


def _pickable(rows_of_game: List[dict], cell: Tuple[str, str],
              band: Optional[str], used_sha1: Set[str],
              chosen_plies: List[int]) -> List[dict]:
    """One game's still-pickable rows for `cell` (band-restricted when `band` is
    not None), ascending by ply -- the input `_choose_positions` expects.

    Excludes an already-claimed `canonical_sha1`, and any row within MIN_PLY_GAP
    of a row ALREADY SELECTED from that game -- globally, i.e. including rows
    taken in another cell or during the floor pass (v2's per-game rules span
    cells; `_choose_positions` then enforces the gap WITHIN the rows it returns).
    """
    out = [r for r in rows_of_game
           if (r["role"], r["phase"]) == cell
           and (band is None or r["band"] == band)
           and r["canonical_sha1"] not in used_sha1
           and all(abs(r["ply"] - p) >= MIN_PLY_GAP for p in chosen_plies)]
    out.sort(key=lambda r: r["ply"])
    return out


def sample_v2_rows(kept: List[dict], *, seed: int) -> Tuple[List[dict], dict]:
    """Sample the frozen 240-row v2 dev corpus from the screen's KEPT rows.

    Steps: (1) build each game's (role, phase) contribution profile; (2)
    `assign_split_v2` places WHOLE games into tuning / frozen_check; (3) a
    round-robin fills every SPLIT_ALLOC_V2 cell EXACTLY, subject -- jointly -- to
    the GLOBAL <=MAX_PER_GAME (<=2 selected rows per game across all cells and
    both splits), a GLOBAL >=MIN_PLY_GAP between any two rows taken from one game,
    a per-split side balance |red-black| <= SIDE_TOL, no duplicate
    canonical_sha1, and -- on the (target, late) cell -- the hard
    LATE_TARGET_FLOORS.

    The floors are met by a FLOOR-SATISFACTION PASS on that one cell: before its
    ordinary fill, rows are drawn from each floor band (in LATE_TARGET_FLOORS
    order) until that band's counter reaches its floor. The counters are GLOBAL
    across splits -- the floors are a combined requirement over the cell's 45
    rows (30 tuning + 15 frozen_check), never a per-split one -- so tuning's
    contribution carries into frozen_check's pass. Without this pass an
    earliest-game fill would take the abundant b400_plus late rows and miss the
    floors even where a satisfying selection exists. The floors are then
    RE-VERIFIED on the SELECTED rows (an independent witness, not the running
    counter) before returning.

    Every cell must reach its quota exactly and every floor must be met: a
    shortfall of either kind is an ERROR (raises ValueError), never a silent
    truncation. Deterministic under `seed`. Returns (rows, stats); each returned
    row is a COPY of its input row stamped with `split`.
    """
    games: Dict[Any, List[dict]] = defaultdict(list)
    for r in kept:
        games[r["game_idx"]].append(r)

    profile = {gi: Counter((r["role"], r["phase"]) for r in rows_)
               for gi, rows_ in games.items()}

    split_of = assign_split_v2(profile, seed)   # may raise ValueError (infeasible)

    used_sha1: Set[str] = set()
    game_used: Counter = Counter()                        # GLOBAL rows per game
    game_plies: Dict[Any, List[int]] = defaultdict(list)  # GLOBAL plies per game
    side_count = {s: {"red": 0, "black": 0} for s in SPLITS}
    floor_count: Counter = Counter()      # selected late-TARGET rows, by band,
    selected: List[dict] = []             # GLOBAL across both splits

    def take(gi, cell, split, band, limit) -> int:
        """Select up to `limit` of game `gi`'s rows for `cell` (band-restricted
        when `band` is not None), honouring every per-game/per-split constraint,
        and record them. Returns how many were ACTUALLY selected (0 is normal: the
        game may have spent its budget, or hold no row of `band`)."""
        positions = _pickable(games[gi], cell, band, used_sha1, game_plies[gi])
        take_n = min(MAX_PER_GAME - game_used[gi], limit, len(positions))
        n_taken = 0
        for r in _choose_positions(positions, take_n, side_count[split],
                                   MIN_PLY_GAP):
            # `_pickable` screened these against `used_sha1` as a BATCH, so a game
            # holding the same hash twice could otherwise smuggle both into one
            # `_choose_positions` result. Re-check per row so "no duplicate
            # canonical_sha1 in the selection" holds for ANY input, not only for
            # the (already hash-deduped) screen output. Skipping merely leaves the
            # cell one row short here; the exact-or-raise guard below still fires
            # if no other game can cover it -- never a silent truncation.
            if r["canonical_sha1"] in used_sha1:
                continue
            out = dict(r)
            out["split"] = split
            selected.append(out)
            used_sha1.add(r["canonical_sha1"])
            game_used[gi] += 1
            game_plies[gi].append(r["ply"])
            side_count[split][r["side"]] += 1
            if cell == LATE_TARGET_CELL:
                floor_count[r["band"]] += 1
            n_taken += 1
        return n_taken

    for split in SPLITS:
        for cell in CELL_ORDER_V2:
            quota = SPLIT_ALLOC_V2[cell][split]
            picked = 0
            cand_games = sorted(
                gi for gi in games
                if split_of.get(gi) == split and cell in profile[gi])

            # Floor-satisfaction pass: floor bands FIRST, only while their
            # (global, cross-split) counters are still short. A no-op once the
            # floors are already met -- e.g. in frozen_check when tuning met them.
            if cell == LATE_TARGET_CELL:
                for band, floor in LATE_TARGET_FLOORS.items():
                    for gi in cand_games:
                        if picked >= quota or floor_count[band] >= floor:
                            break
                        picked += take(gi, cell, split, band,
                                       min(floor - floor_count[band],
                                           quota - picked))

            # Ordinary fill: any band, earliest game first (v1's round-robin).
            for gi in cand_games:
                if picked >= quota:
                    break
                picked += take(gi, cell, split, None, quota - picked)

            if picked != quota:
                raise ValueError(
                    f"final-manifest shortfall: cell {(cell[0], cell[1], split)} "
                    f"filled {picked} of required {quota}")

    # Hard floor verification, counted FROM THE SELECTED ROWS -- an independent
    # witness rather than the running `floor_count`, so a bug in the pass's own
    # bookkeeping cannot certify itself. An unmet floor is an ERROR (design Sec
    # 1.3: the floors are a requirement, not a best-effort).
    late_band_counts: Counter = Counter(
        r["band"] for r in selected
        if (r["role"], r["phase"]) == LATE_TARGET_CELL)
    for band, floor in LATE_TARGET_FLOORS.items():
        if late_band_counts[band] < floor:
            raise ValueError(
                f"late-target coverage floor unmet: band {band} has "
                f"{late_band_counts[band]} of the required {floor} among the "
                f"{sum(late_band_counts.values())} selected late-target rows")

    # Counted per (role, phase, split) FROM THE SELECTED ROWS so cell_counts is an
    # INDEPENDENT composition witness, not a re-emission of the SPLIT_ALLOC_V2
    # quotas (v1's `cell_counts_actual` idiom). On success these equal the quotas
    # -- the exact-or-raise guard above already fired otherwise -- but computing
    # them from the rows makes the stats a real cross-check rather than a tautology.
    cell_counts_actual: Counter = Counter(
        (r["role"], r["phase"], r["split"]) for r in selected)

    stats = {
        "n_rows": len(selected),
        "seed": seed,
        "cell_counts": {
            f"{role}|{phase}|{split}": cell_counts_actual[(role, phase, split)]
            for (role, phase) in SPLIT_ALLOC_V2 for split in SPLITS},
        "side_count": {s: dict(side_count[s]) for s in SPLITS},
        # The floor WITNESS (v2-specific; v1's `bucket_count` is gone with the
        # bucket cap): the selected late-TARGET rows' band histogram.
        "late_target_band_count": dict(sorted(late_band_counts.items())),
        "n_games_per_split": {
            s: sum(1 for gi in split_of if split_of[gi] == s) for s in SPLITS},
        "n_games_total": len(split_of),
    }
    return selected, stats
