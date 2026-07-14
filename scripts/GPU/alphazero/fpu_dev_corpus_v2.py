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
PURE SECTION (Task 1) -- constants + pure functions ONLY.
=============================================================================
Mirrors build_fpu_dev_corpus.py's own PURE SECTION / OPERATOR SHELL split.
Everything in this file is pure at this task: plain-stdlib constants and one
pure classifier. NO MCTS / evaluator / GPU / MLX / heavy-numpy imports, no
I/O, no argument parsing. Later tasks append BELOW this section, in order:
  Task 2: `enumerate_v2_proposals` -- phase-aware side-opposed proposal
    enumerator (no global stride).
  Task 3: the phase-stratified sampler (with late floors).
  Task 4: the v2 geometric preflight.
  Task 5: the operator `screen` stage (evaluator/MCTS; lazy heavy imports).
  Task 6: the pure `select` stage + config loader + `main`.
Keep this section cleanly separated and importable without ever triggering a
GPU/MLX import -- any future heavy import goes lazily inside the Task-5
operator functions, exactly as build_fpu_dev_corpus.py's own `main` /
`_build_anchor_search_fn` / evaluator plumbing do.

What this section does
-----------------------
Frozen phase-primary constants (design Sec 1.2 / 1.3 / 1.5) plus the one v2
classifier, `proposal_cell_of`, which maps a (phase, n_legal) pair to its
PROPOSAL_CELLS membership -- or to a cell deliberately NOT in PROPOSAL_CELLS
(a `("late", None)` sentinel for sub-200 late positions is intentionally
ineligible; see `proposal_cell_of`'s own docstring). DRY: reuses `band_of`,
and re-exports the shared v1 MIN_PLY_GAP / MAX_PER_GAME / SIDE_TOL
constants, from `build_fpu_dev_corpus` rather than restating them -- see
each import's inline comment for why.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

# Deliberately-shared v1 frozen constants (identical semantics in v2):
# MIN_PLY_GAP = the >=12-ply side-opposed-pair gap; MAX_PER_GAME = the <=2
# SELECTED rows per game cap (global, across all proposal cells, enforced
# by the v2 sampler -- Task 3); SIDE_TOL = the per-split |red-black| side
# balance tolerance. Imported (not restated) so a v1 drift is felt here
# too; pinned in tests/test_fpu_dev_corpus_v2.py.
#
# Task-2 additions (the enumerator below): `per_ply_n_legal` (per-ply legal
# count, incl. its reconstruction fallback), `ply_bucket_of` (phase), and
# `side_to_move_for_ply` (red on even plies) are the same pure per-ply
# primitives v1's own scan uses; `_first_gap_pair` is v1's deterministic
# earliest-satisfying side-opposed-pair search (build_fpu_dev_corpus.py:578).
# All reused verbatim (DRY) -- never reimplemented here.
from .build_fpu_dev_corpus import (
    MAX_PER_GAME,
    MIN_PLY_GAP,
    SIDE_TOL,
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
