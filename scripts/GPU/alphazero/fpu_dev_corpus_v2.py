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
PURE SECTION (Tasks 1-4) -- constants + pure functions ONLY.
=============================================================================
Mirrors build_fpu_dev_corpus.py's own PURE SECTION / OPERATOR SHELL split.
Everything in this file is pure so far: plain-stdlib constants, classifiers,
the proposal enumerator, the phase-stratified sampler and the geometric
preflight. NO MCTS / evaluator / GPU / MLX / heavy-numpy imports, no argument
parsing. The ONE impure function is `v2_preflight_source`, the deliberately
thin file-read wrapper over the pure preflight core -- stdlib json/pathlib
only, exactly like v1's own `preflight_source` (build_fpu_dev_corpus.py:785).
Later tasks append BELOW this section, in order:
  Task 5: the operator `screen` stage (evaluator/MCTS; lazy heavy imports),
    plus the required config loader (`V2Config` / `load_v2_config`) and
    `main` itself -- `main --mode screen` already needs the config to record
    its own hash in the screen's `.meta.json`, so both live here rather than
    waiting for Task 6 (see fpu_dev_corpus_v2.py's own Task-5 section header).
  Task 6: the pure `select` stage, extending `main` with `--mode select`
    (`screen` and `select` are never the same invocation).
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
Task 4: `v2_geometry_feasibility` (+ `v2_preflight_source`) -- the ROLE-AGNOSTIC
geometric preflight: STAGE 1 of the design's two-stage feasibility split (Sec
1.7). It proves, from proposal GEOMETRY ALONE and JOINTLY via a constructive
witness, that a (source, enumeration) pair could support the corpus -- and
explicitly does NOT prove the target-ROLE floors or DISJOINTNESS, which need
the evaluator's raw policy and per-state hashes (Task 6's `select`).
DRY: reuses `band_of` / `ply_bucket_of` / `side_to_move_for_ply` /
`per_ply_n_legal` / `_first_gap_pair` / `_gap_selectable` / `_choose_positions`,
and re-exports the shared v1 MIN_PLY_GAP / MAX_PER_GAME / SIDE_TOL / SPLITS
constants, from `build_fpu_dev_corpus` rather than restating them -- see each
import's inline comment for why.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

# Deliberately-shared v1 frozen constants (identical semantics in v2):
# MIN_PLY_GAP = the >=12-ply side-opposed-pair gap; MAX_PER_GAME = the <=2
# SELECTED rows per game cap (global, across all proposal cells, enforced
# by the v2 sampler -- Task 3); SIDE_TOL = the per-split |red-black| side
# balance tolerance -- the same NUMBER as v1's but no longer met by v1's
# mechanism: v1 got side neutrality for FREE (see its own SIDE_TOL note) and
# never had to enforce it, whereas v2 must both STEER toward it
# (`_choose_positions_v2`) and ENFORCE it (`sample_v2_rows`' exact-or-raise
# check); SPLITS = the ("tuning", "frozen_check") split vocabulary
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
# >=gap-apart chain -- semantics v2 needs for every take_n EXCEPT 2, so it is
# imported (never copied) and WRAPPED by `_choose_positions_v2`, whose 2-take
# prefers a side-opposed pair (via the same `_first_gap_pair`) because v1's
# "a 2-take is side-neutral for free" premise is FALSE in v2 -- see that
# function's docstring. (v1's `_greedy_assign` / `assign_split` /
# `sample_dev_rows` are NOT reusable: they close over v1's
# SPLIT_ALLOC/CELL_ORDER/bucket-cap globals and apply the <=MAX_PER_GAME cap PER
# CELL, where v2's rule is GLOBAL -- see `_greedy_assign_v2` / `sample_v2_rows`.)
# Task-4 addition (the preflight below): `_gap_selectable`
# (build_fpu_dev_corpus.py:562) is v1's greedy earliest-first >=gap chain count,
# capped -- the SAME realizable-capacity idiom v1's own preflight uses, so v2's
# per-phase / per-late-cell capacity diagnostics cannot drift from v1's per-band
# ones. (v1's `_fit_pair` is deliberately NOT imported: it is ply-bucket-CAP
# aware, and v2 has no bucket cap -- each phase is exactly 25% by construction,
# design Sec 1.2. v1's `geometry_feasibility` / `_build_witness` / `PreflightReport`
# are likewise not reusable as-is: they close over v1's BANDS / QUOTA_PER_BAND /
# bucket-cap globals and apply the <=MAX_PER_GAME cap per BAND, where v2's witness
# must spend ONE shared per-game budget across PHASES -- see `_build_v2_witness`.)
#
# All reused verbatim (DRY) -- never reimplemented here.
#
# Task-5 additions (the operator `screen` stage at the bottom of this file):
# `anchor_eligible` / `raw_policy_role` / `_policy_features_from_priors` /
# `load_forbidden_hashes` / `load_game_index` are v1's own Stage-2/scan pure
# helpers (build_fpu_dev_corpus.py Tasks 5-6) -- reused verbatim, exactly as
# the design's Sec 2 reuse list names them, never reimplemented here.
from .build_fpu_dev_corpus import (
    MAX_PER_GAME,
    MIN_PLY_GAP,
    SIDE_TOL,
    SPLITS,
    _choose_positions,
    _first_gap_pair,
    _gap_selectable,
    _policy_features_from_priors,
    anchor_eligible,
    band_of,
    load_forbidden_hashes,
    load_game_index,
    per_ply_n_legal,
    ply_bucket_of,
    raw_policy_role,
    side_to_move_for_ply,
)
# Task-5 additions: both pure (no MCTS/GPU/MLX) -- `canonical_state_sha1`
# imports only `TwixtState` (fpu_state_hash's own module docstring);
# `position_state` imports only `statistics` + `TwixtState`
# (goal_line_trigger_probe_cases' own module docstring). DRY per the design's
# Sec 2 reuse list -- reused, never reimplemented, exactly as v1's own
# operator shell does (build_fpu_dev_corpus.py:46-47).
from .fpu_state_hash import canonical_state_sha1
from .goal_line_trigger_probe_cases import position_state
# Stdlib-only provenance helpers (design Sec 1.8) -- keeps this module
# GPU/MLX-free, exactly as v1's own reuse (build_fpu_dev_corpus.py:55).
from . import fpu_provenance

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
#   4. Per-split side balance must be actively STEERED and then ENFORCED. v1
#      enjoyed it for free -- its SIDE_TOL note says "Every game supplies one red
#      + one black position, so whole-game (2-per-game) picks are side-neutral",
#      true because a v1 (role, band) cell can receive at most ONE side-opposed
#      pair from a game. That premise is FALSE in v2 (see `_choose_positions_v2`),
#      so v2 prefers a side-opposed pair on every 2-take and then RE-VERIFIES
#      |red - black| <= SIDE_TOL on the selected rows, exact-or-raise.
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


# How many deterministic whole-game split assignments `sample_v2_rows` will try
# before giving up. Each is a distinct seeded ordering of `_greedy_assign_v2`, and
# each is judged by the REAL fill (see `sample_v2_rows`), never by the greedy's own
# optimistic accounting.
#
# Justified by the measured curve, not guessed: on 50 realistic randomized screens
# of 4,800 games, EVERY pool that the old sampler failed on was in fact satisfiable,
# and a valid manifest was found within the first few orderings (the deepest needed
# was the 5th). 8 leaves headroom over that observed maximum while staying in the
# small single digits -- the cost of an unused attempt is zero (the loop returns on
# the first success) and the cost of a used one is a single extra fill.
ASSIGN_ATTEMPTS = 8

# Rows each split needs in total (tuning 160, frozen_check 80) -- DERIVED from the
# frozen allocation, never hard-coded. `_greedy_assign_v2` scores a game by the
# FRACTION of a split's remaining need it closes, so it needs these denominators.
SPLIT_TOTALS: Dict[str, int] = {
    s: sum(alloc[s] for alloc in SPLIT_ALLOC_V2.values()) for s in SPLITS}


def _greedy_assign_v2(games_profile, seed, attempt) -> Optional[Dict[Any, str]]:
    """One deterministic greedy pass (v1 `_greedy_assign`'s shape). Returns
    {game_idx: split} if it satisfies every per-(role, phase, split) quota, else
    None.

    Games are visited in a seed-shuffled order that is unique per `attempt`
    (attempt 1 is the deterministic REVERSE of attempt 0 -- v1's secondary-ordering
    retry, kept verbatim; attempts >= 2 are fresh independent shuffles). Its caller
    `sample_v2_rows` walks up to ASSIGN_ATTEMPTS of them, because THIS function's
    verdict is only a necessary condition -- see below.

    A game's contribution is capped at MAX_PER_GAME in TOTAL across its cells --
    NOT per cell, as v1 does -- because that is v2's actual selection rule. Both
    the `realizable` scoring and the `need` decrement spend one shared per-game
    budget over the game's cells in CELL_ORDER_V2 order, which is exactly the
    order (and the greedy "as many as this cell still needs" rule) the round-robin
    in `sample_v2_rows` will use.

    PLACEMENT RULE -- by FILL FRACTION, not by raw row count. `tuning` needs 160
    rows and `frozen_check` only 80, so a raw-count comparison (v1's rule, which v2
    inherited) TIES on nearly every game -- both splits can use its 2 rows -- and
    the tie-break "prefer the split with the larger total remaining need" then sends
    it to tuning EVERY time, until tuning is full. Measured on a realistic 4,800-game
    screen: 2,227 games to tuning vs 78 to frozen_check. frozen_check is the SCARCE
    split (it must cover 80 rows across all 8 cells, and every game it uses costs a
    whole <=2-row budget), so that sliver left its cells one gap-blocked or
    side-degenerate game away from an unfillable quota -- the dominant cause of both
    the false `final-manifest shortfall` and the residual side-balance raises.
    Comparing `realizable / remaining_need` instead makes a game that closes 2 of
    frozen's remaining 80 outrank one that closes 2 of tuning's remaining 160, so
    the two candidate pools grow roughly in proportion to the need they must serve.
    The comparison is done by CROSS-MULTIPLICATION in exact integers -- never floats
    -- so it is bit-for-bit reproducible.

    Games that neither split can currently use are spread to keep BOTH candidate
    pools rich (in the 160:80 ratio of the splits themselves): the fill can only
    ever choose among a split's OWN games, so an over-stuffed tuning pool is not
    just wasted, it is starvation for frozen_check.

    NECESSARY, NOT SUFFICIENT: reaching `need == 0` here only means this optimistic
    accounting is satisfied. It ignores >=MIN_PLY_GAP, dedup and side balance, so an
    assignment it accepts can still be unfillable -- measured: on EVERY pool the old
    sampler failed, this function returned OK on attempt 0 and the FILL then failed.
    That is exactly why `sample_v2_rows` re-runs the real fill per ordering rather
    than trusting this verdict.
    """
    rng = random.Random(seed * 1_000_003 + attempt)
    order = sorted(games_profile)
    rng.shuffle(order)
    if attempt == 1:
        order = order[::-1]

    need = {cell: dict(alloc) for cell, alloc in SPLIT_ALLOC_V2.items()}
    placed: Counter = Counter()
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
        rem_t = sum(need[c]["tuning"] for c in CELL_ORDER_V2)
        rem_f = sum(need[c]["frozen_check"] for c in CELL_ORDER_V2)

        if u_t == 0 and u_f == 0:
            # Useless to both splits right now -- spread it, keeping the two
            # candidate pools in the splits' own 160:80 ratio.
            split = ("tuning"
                     if placed["tuning"] * SPLIT_TOTALS["frozen_check"]
                     <= placed["frozen_check"] * SPLIT_TOTALS["tuning"]
                     else "frozen_check")
        elif rem_t == 0:
            split = "frozen_check"
        elif rem_f == 0:
            split = "tuning"
        else:
            # u_t / rem_t  vs  u_f / rem_f, by exact-integer cross-multiplication.
            lhs, rhs = u_t * rem_f, u_f * rem_t
            if lhs > rhs:
                split = "tuning"
            elif rhs > lhs:
                split = "frozen_check"
            else:
                split = ("tuning"
                         if placed["tuning"] * SPLIT_TOTALS["frozen_check"]
                         <= placed["frozen_check"] * SPLIT_TOTALS["tuning"]
                         else "frozen_check")

        assign[gi] = split
        placed[split] += 1
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


def _capacity_precheck(
        games_profile: Mapping[Any, Mapping[Tuple[str, str], int]]) -> None:
    """The two GENUINE-infeasibility checks. Both are NECESSARY conditions only --
    each is a true UPPER BOUND on what the selection can realize, so falling short
    of demand PROVES infeasibility and names it cheaply, while passing them proves
    nothing. (Once a game's rows span cells, no per-cell sum can be exact: its
    <=2-row global budget is claimable by any one of them.)

    Both are independent of the split assignment, so `sample_v2_rows` runs them ONCE
    and never retries them: a pool that fails here cannot be rescued by any ordering.
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


def assign_split_v2(games_profile: Mapping[Any, Mapping[Tuple[str, str], int]],
                    seed: int, *, attempt: int = 0) -> Dict[Any, str]:
    """Assign each WHOLE game to "tuning" or "frozen_check" so every
    per-(role, phase, split) SPLIT_ALLOC_V2 quota is satisfiable.

    `games_profile`: {game_idx: {(role, phase): n_available_kept_rows}}.
    `attempt` selects ONE of the deterministic candidate orderings.

    This returns a CANDIDATE assignment, not a verdict: `_greedy_assign_v2`'s
    accounting is optimistic (it ignores >=MIN_PLY_GAP, dedup and side balance), so
    an assignment it accepts can still be unfillable. Only the fill can tell -- so
    `sample_v2_rows` walks up to ASSIGN_ATTEMPTS orderings and keeps the first whose
    manifest passes EVERY exact-or-raise verification. Callers wanting a single
    assignment (tests, diagnostics) can use the default `attempt=0`.

    Raises ValueError if a capacity precheck proves the pool infeasible outright, or
    if this ordering's greedy cannot satisfy the split quotas.
    """
    _capacity_precheck(games_profile)
    result = _greedy_assign_v2(games_profile, seed, attempt)
    if result is None:
        raise ValueError(
            f"assign_split_v2: ordering {attempt} did not satisfy the split quotas")
    return result


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


def _choose_positions_v2(positions: List[dict], take_n: int,
                         side_count: Mapping[str, int], gap: int) -> List[dict]:
    """v1's `_choose_positions` with ONE v2 amendment: a 2-take PREFERS a
    side-opposed pair. `positions` is one game's pickable rows, ascending by ply.

    v1 could walk the earliest >=gap-apart chain and stay side-neutral for free.
    Its own SIDE_TOL note states the premise: "Every game supplies one red + one
    black position, so whole-game (2-per-game) picks are side-neutral" -- true
    there because a v1 (role, band) cell can only ever receive ONE side-opposed
    pair from a game.

    That premise is FALSE in v2, for two independent reasons:
      * v2's (role, "late") SAMPLER cell aggregates THREE proposal cells
        (late/b400_plus, late/b300_399, late/b200_299), so ONE game can offer
        that single cell up to 3 reds + 3 blacks; and
      * `raw_policy_role` classifies each row INDEPENDENTLY, so a proposal
        pair's red can be classified `target` while its black goes `control`
        (or is dropped in the grey zone).
    Same-side-only -- or merely same-side-EARLIEST -- candidate sets are
    therefore routine in v2, not pathological, and v1's chain walk would happily
    take two REDS, silently skewing the split's side balance.

    So for `take_n == 2` we first ask `_first_gap_pair` (v1's own deterministic
    earliest-satisfying side-opposed-pair search -- imported, never copied) for a
    gap-valid red+black pair among `positions`, and use it whenever one exists.
    It yields the SAME 2 rows, so this can never introduce a shortfall: it only
    chooses a side-NEUTRAL 2 rows over a possibly same-side 2 rows. When no such
    pair exists (a genuinely same-side-only candidate set) we fall back to v1's
    `_choose_positions` unchanged -- and `sample_v2_rows`' exact-or-raise
    side-balance check is then what refuses to emit a skewed manifest.

    Every other `take_n` delegates to `_choose_positions` VERBATIM -- notably its
    take_n == 1 path, which already steers toward the split's deficit side (that
    steering is what absorbs the odd-quota leftovers; it is vacuous over a
    single-candidate list, which is precisely why the 2-take needs its own rule
    and why the exact-or-raise check has to be the backstop).

    Both `positions` and, hence, the reds/blacks split out of it are ascending by
    ply -- exactly `_first_gap_pair`'s precondition. The pair is returned in
    ascending ply order, matching `_choose_positions`' own return order.
    """
    if take_n == 2:
        pair = _first_gap_pair([r for r in positions if r["side"] == "red"],
                               [r for r in positions if r["side"] == "black"],
                               gap)
        if pair is not None:
            return sorted(pair, key=lambda r: r["ply"])
    return _choose_positions(positions, take_n, side_count, gap)


def _side_delta(rows: List[dict]) -> int:
    """Signed effect of taking `rows` on a split's (red - black) balance. A
    side-opposed pair scores exactly 0 -- which is why `sample_v2_rows`' cell fill
    can rank candidate GAMES by the balance their rows would leave behind, without
    ever needing to name "opposed pair" as a special case (see `fill`)."""
    return sum(1 if r["side"] == "red" else -1 for r in rows)


def _select_manifest(games: Mapping[Any, List[dict]],
                     profile: Mapping[Any, Mapping[Tuple[str, str], int]],
                     split_of: Mapping[Any, str]) -> Tuple[List[dict], dict]:
    """ONE selection attempt, for ONE candidate whole-game split assignment.

    Fills every SPLIT_ALLOC_V2 cell EXACTLY via the side-aware round-robin, subject
    -- jointly -- to the GLOBAL <=MAX_PER_GAME (<=2 selected rows per game across
    all cells and both splits), a GLOBAL >=MIN_PLY_GAP between any two rows taken
    from one game, a per-split side balance |red-black| <= SIDE_TOL, no duplicate
    canonical_sha1, and -- on the (target, late) cell -- the hard LATE_TARGET_FLOORS.

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

    Side balance gets the SAME exact-or-raise treatment as the floors, because v2
    -- unlike v1 -- cannot assume it (see `_choose_positions_v2`). It is steered at
    BOTH levels: WITHIN a game (a 2-take prefers a side-opposed pair; a 1-take
    prefers the split's deficit side) and, because that alone leaves the sampler
    with no cross-game choice at all, ACROSS games -- `fill` picks the candidate
    game whose rows leave the split closest to balanced. |red - black| <= SIDE_TOL
    is then RE-VERIFIED per split on the SELECTED rows.

    Raises ValueError on ANY shortfall, unmet floor or side-balance violation --
    never a silent truncation, never a silent skew. It builds ALL of its own running
    state from its arguments and mutates nothing outside itself, which is precisely
    what lets `sample_v2_rows` simply RE-RUN it on the next candidate assignment: the
    fill, not the assignment greedy's optimistic accounting, is the authority on
    whether an assignment actually works.
    """
    used_sha1: Set[str] = set()
    game_used: Counter = Counter()                        # GLOBAL rows per game
    game_plies: Dict[Any, List[int]] = defaultdict(list)  # GLOBAL plies per game
    side_count = {s: {"red": 0, "black": 0} for s in SPLITS}
    floor_count: Counter = Counter()      # selected late-TARGET rows by band --
                                          # GLOBAL across both splits (the floors
                                          # are a COMBINED requirement)
    picked_in: Counter = Counter()        # rows SELECTED per (cell, split): the
                                          # fill's authoritative budget counter,
                                          # maintained by `take` alone
    selected: List[dict] = []

    def rows_for(gi, cell, split, band, limit) -> List[dict]:
        """The rows game `gi` WOULD contribute to `cell`/`split` right now -- the
        single source of truth for a game's contribution.

        PURE with respect to the running state: it reads `used_sha1` / `game_used`
        / `game_plies` / `side_count` but mutates NOTHING, so the cell fill can
        PREVIEW a game (and its `_side_delta`) to decide the draw order before
        committing to it, and `take` can then commit exactly what was previewed.
        """
        positions = _pickable(games[gi], cell, band, used_sha1, game_plies[gi])
        take_n = min(MAX_PER_GAME - game_used[gi], limit, len(positions))
        return _choose_positions_v2(positions, take_n, side_count[split],
                                    MIN_PLY_GAP)

    def take(gi, cell, split, band, limit) -> int:
        """COMMIT up to `limit` of game `gi`'s rows for `cell` (band-restricted when
        `band` is not None), honouring every per-game/per-split constraint. Returns
        how many were ACTUALLY selected (0 is normal: the game may have spent its
        budget, or hold no row of `band`)."""
        n_taken = 0
        for r in rows_for(gi, cell, split, band, limit):
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
            picked_in[(cell, split)] += 1
            if cell == LATE_TARGET_CELL:
                floor_count[r["band"]] += 1
            n_taken += 1
        return n_taken

    def fill(cand_games, cell, split, band, budget_fn) -> None:
        """Draw rows for `cell`/`split` from `cand_games` in a SIDE-AWARE,
        deterministic order, until `budget_fn()` -- the rows this PASS still wants
        -- reaches 0 or no candidate game can yield anything. Used by BOTH the
        floor-satisfaction pass (band-restricted) and the ordinary fill.

        Why the order must be side-aware at all: the sampler has NO cross-game side
        choice otherwise. `_choose_positions_v2` steers WITHIN one game (its 2-take
        prefers a side-opposed pair; its 1-take prefers the split's deficit side),
        but a game that can ONLY offer same-side rows forces a same-side take no
        matter how skewed the split already is -- and a plain `sorted(cand_games)`
        walk cannot prefer a game that would fix the balance, because it never
        looks at a second game. On realistic screens (roles assigned per ROW, so a
        proposal pair's red can be `target` while its black is `control`) opposed
        pairs are the MINORITY and same-side-only games dominate, which made the
        exact-or-raise side-balance check fire on 34 of 35 satisfiable pools.

        The rule: at each step, preview every still-unused candidate game at every
        take size it could give (2 rows, or just 1), and take the (game, size) whose
        rows leave the split CLOSEST to side-balanced. That single score subsumes
        both halves of the intended behaviour:

          * BALANCED split -> a side-OPPOSED PAIR scores |0 + 0| = 0, beating any
            singleton (|+-1|) or same-side pair (|+-2|), so all opposed-pair games
            are drawn FIRST, in ascending game_idx (the final tie-break) -- exactly
            the historical round-robin order, at the maximum 2 rows per game.
          * SKEWED split -> the score reaches for the correction: a deficit-side
            pair (|imb| - 2) or, failing that, a deficit-side SINGLETON (|imb| - 1)
            now outranks a neutral opposed pair (|imb|), which merely preserves the
            skew. This is what makes an opposed-pair-rich cell able to REPAIR an
            imbalance it inherited from an earlier cell -- and it must, because
            `assign_split_v2` is side-BLIND: it can hand one split a side-skewed
            set of games for a cell (e.g. 2 red-only vs 6 black-only), and NO
            ordering within that cell can then balance it. Taking opposed pairs
            unconditionally would lock the skew in permanently.

        Each step re-previews, because every take moves the running balance, the
        games' own <=MAX_PER_GAME budgets and the >=MIN_PLY_GAP/dedup filters. The
        chosen game is then retired from this cell (as the historical single visit
        per game per cell did), which also guarantees termination.

        This is a GREEDY, not a global optimizer: it can still be CONSERVATIVE
        (raise on a pool a perfect selection could balance), but it is never a
        silent false PASS -- the exact-or-raise verifications below remain the
        authority.
        """
        candidates = list(cand_games)               # ascending game_idx
        while candidates:
            budget = budget_fn()
            if budget <= 0:
                return
            imbalance = side_count[split]["red"] - side_count[split]["black"]

            # ONE preview per game: its NATURAL (largest affordable) take.
            full = {}
            for gi in candidates:
                rows = rows_for(gi, cell, split, band, budget)
                if rows:
                    full[gi] = rows
                # else: nothing pickable -- and MONOTONE (the budget only shrinks;
                # used_sha1/game_plies only grow), so it can never revive. Dropped
                # from `candidates` below, for good.
            if not full:
                return
            candidates = sorted(full)
            # Rows the remaining games could still yield, from those same previews.
            capacity = sum(len(rows) for rows in full.values())

            scored = []
            for gi, rows in full.items():
                # Keys are a STRICT TOTAL order -- (game_idx, limit) is unique -- so
                # the draw is fully deterministic. Prefer, in order: the smallest
                # resulting |red - black|; then the bigger take (2 rows over 1, so a
                # cell burns as few games as it can); then the lowest game_idx.
                scored.append(((abs(imbalance + _side_delta(rows)), -len(rows), gi),
                               gi, budget))
                # The deficit-side SINGLETON alternative to a 2-take. It is what
                # lets an opposed-pair-rich cell REPAIR an inherited skew (a neutral
                # pair would merely preserve it), but it spends a whole extra game
                # on one row -- so offer it ONLY while the cell can still afford its
                # quota without this game's other row: the remaining games must
                # still cover the budget. Otherwise correcting the balance here
                # would just trade a side-balance raise for a SHORTFALL raise.
                if len(rows) > 1 and capacity - len(rows) >= budget - 1:
                    one = rows_for(gi, cell, split, band, 1)
                    if one:
                        scored.append(
                            ((abs(imbalance + _side_delta(one)), -len(one), gi),
                             gi, 1))
            scored.sort()
            _key, best_gi, best_limit = scored[0]
            candidates = [gi for gi in candidates if gi != best_gi]
            take(best_gi, cell, split, band, best_limit)

    for split in SPLITS:
        for cell in CELL_ORDER_V2:
            quota = SPLIT_ALLOC_V2[cell][split]
            cand_games = sorted(
                gi for gi in games
                if split_of.get(gi) == split and cell in profile[gi])

            # The rows this cell/split still wants. Read from `picked_in`, which
            # `take` -- the single commit point -- maintains, so no fill pass can
            # drift from the authoritative count.
            def ordinary_budget(_cell=cell, _split=split, _quota=quota):
                return _quota - picked_in[(_cell, _split)]

            # Floor-satisfaction pass: floor bands FIRST, only while their (global,
            # cross-split) counters are still short. A no-op once the floors are
            # already met -- e.g. in frozen_check when tuning met them. Side-aware
            # too: a floor band is exactly where same-side-only games cluster (a
            # game's b300_399 red can be `target` while its black is `control`).
            if cell == LATE_TARGET_CELL:
                for band, floor in LATE_TARGET_FLOORS.items():
                    def floor_budget(_band=band, _floor=floor, _ord=ordinary_budget):
                        return min(_floor - floor_count[_band], _ord())
                    fill(cand_games, cell, split, band, floor_budget)

            # Ordinary fill: any band.
            fill(cand_games, cell, split, None, ordinary_budget)

            picked = picked_in[(cell, split)]
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

    # Hard per-split side-balance verification -- likewise counted FROM THE
    # SELECTED ROWS, independently of the running `side_count` the take-time
    # steering consumed, so that bookkeeping cannot certify itself either.
    #
    # This is the constraint v1 never had to ENFORCE: there, one game gave a cell
    # exactly one red + one black, so every whole-game pick was side-neutral by
    # construction and SIDE_TOL was only ever a slack bound on the odd-quota
    # leftovers. v2 breaks that premise (`_choose_positions_v2`), so side balance
    # is a real constraint that a pool can genuinely fail -- and an out-of-tolerance
    # split is an ERROR, exactly like an unmet floor or a cell shortfall. Never
    # emit a silently side-skewed manifest.
    side_actual: Dict[str, Dict[str, int]] = {
        s: {"red": 0, "black": 0} for s in SPLITS}
    for r in selected:
        side_actual[r["split"]][r["side"]] += 1
    for split in SPLITS:
        red, black = side_actual[split]["red"], side_actual[split]["black"]
        if abs(red - black) > SIDE_TOL:
            raise ValueError(
                f"per-split side balance violated: split {split} has red {red} / "
                f"black {black} (|red - black| = {abs(red - black)} > SIDE_TOL "
                f"{SIDE_TOL}); the pool can only fill a cell with same-side rows")

    # Counted per (role, phase, split) FROM THE SELECTED ROWS so cell_counts is an
    # INDEPENDENT composition witness, not a re-emission of the SPLIT_ALLOC_V2
    # quotas (v1's `cell_counts_actual` idiom). On success these equal the quotas
    # -- the exact-or-raise guard above already fired otherwise -- but computing
    # them from the rows makes the stats a real cross-check rather than a tautology.
    cell_counts_actual: Counter = Counter(
        (r["role"], r["phase"], r["split"]) for r in selected)

    stats = {
        "n_rows": len(selected),
        "cell_counts": {
            f"{role}|{phase}|{split}": cell_counts_actual[(role, phase, split)]
            for (role, phase) in SPLIT_ALLOC_V2 for split in SPLITS},
        # The side WITNESS: recomputed from the selected rows and already VERIFIED
        # against SIDE_TOL above -- a real witness, not a report of the running
        # steering counter.
        "side_count": {s: dict(side_actual[s]) for s in SPLITS},
        # The floor WITNESS (v2-specific; v1's `bucket_count` is gone with the
        # bucket cap): the selected late-TARGET rows' band histogram.
        "late_target_band_count": dict(sorted(late_band_counts.items())),
        "n_games_per_split": {
            s: sum(1 for gi in split_of if split_of[gi] == s) for s in SPLITS},
        "n_games_total": len(split_of),
    }
    return selected, stats


def sample_v2_rows(kept: List[dict], *, seed: int) -> Tuple[List[dict], dict]:
    """Sample the frozen 240-row v2 dev corpus from the screen's KEPT rows.

    (1) Build each game's (role, phase) contribution profile. (2) Run the two
    capacity PRECHECKS once -- they are assignment-independent, so a pool that fails
    them is GENUINELY infeasible and is never retried. (3) Walk up to
    ASSIGN_ATTEMPTS deterministic whole-game split assignments, and for each, run
    the REAL selection (`_select_manifest`). The first assignment whose manifest
    passes every exact-or-raise verification -- exact composition, the hard
    LATE_TARGET_FLOORS, and per-split side balance -- wins.

    Why the retry is driven by the FILL and not by the assignment greedy: that
    greedy's accounting is OPTIMISTIC (it ignores >=MIN_PLY_GAP, dedup and side
    balance), so it happily returns an assignment the fill cannot realize. Measured
    on 50 realistic 4,800-game screens: on EVERY pool the sampler failed,
    `_greedy_assign_v2` returned OK on its FIRST ordering -- so a retry loop that
    only re-ran when the GREEDY refused (as v1's did, and as v2 inherited) never
    fired even once. Every one of those pools was in fact satisfiable. Only the fill
    can tell a workable assignment from an unworkable one, so only the fill may
    decide whether to try another.

    This makes the sampler CONSERVATIVE, never a silent false pass: it may still
    raise on a pool some perfect selection could satisfy (it is a greedy, over a
    finite set of orderings), but it will never emit a manifest that is short, that
    misses a floor, or that is side-skewed. Deterministic under `seed`: the
    orderings, the fill and the tie-breaks are all reproducible, so the same input
    and seed always yield identical rows AND stats.

    Returns (rows, stats); each returned row is a COPY of its input row stamped with
    `split`. `stats["assignment_attempt"]` records which ordering won.
    """
    games: Dict[Any, List[dict]] = defaultdict(list)
    for r in kept:
        games[r["game_idx"]].append(r)

    profile = {gi: Counter((r["role"], r["phase"]) for r in rows_)
               for gi, rows_ in games.items()}

    # GENUINE infeasibility -- assignment-independent, so raise now and never retry.
    _capacity_precheck(profile)

    last_error: Optional[ValueError] = None
    for attempt in range(ASSIGN_ATTEMPTS):
        split_of = _greedy_assign_v2(profile, seed, attempt)
        if split_of is None:
            last_error = ValueError(
                f"assign_split_v2: ordering {attempt} did not satisfy the split "
                f"quotas")
            continue
        try:
            rows, stats = _select_manifest(games, profile, split_of)
        except ValueError as exc:
            last_error = exc          # this ORDERING failed -- try the next one
            continue
        stats["seed"] = seed
        stats["assignment_attempt"] = attempt
        return rows, stats

    raise ValueError(
        f"sample_v2_rows: no valid manifest after {ASSIGN_ATTEMPTS} deterministic "
        f"whole-game split assignments (seed {seed}); last failure -- {last_error}")


# ---------------------------------------------------------------------------
# ROLE-AGNOSTIC geometric preflight (design Sec 1.7) -- Task 4
# ---------------------------------------------------------------------------
# STAGE 1 of the design's TWO-STAGE feasibility split. A hard gate that runs
# BEFORE the evaluator loads: it proves, from PROPOSAL GEOMETRY ALONE (the
# enumerator's ply / side / phase / n_legal / band / proposal_cell rows; NO NN, NO
# MCTS, NO raw-policy), that a (source corpus, enumeration) pair can JOINTLY
# support the v2 corpus -- or it names the binding constraint so an infeasible
# source hard-stops cheaply, before any evaluator setup is wasted. Extends v1's
# own preflight (build_fpu_dev_corpus.py:444-472) from BANDS to PHASES.
#
# SCOPE -- what this stage does NOT prove, and why (design Sec 1.7):
#   * ROLE (target vs control) comes from the evaluator's raw policy, so it is NOT
#     provable from geometry. This gate is therefore ROLE-AGNOSTIC: it proves each
#     phase's CANDIDATE total of QUOTA_PER_PHASE (60 = 45 target + 15 control),
#     exactly as v1's preflight proves QUOTA_PER_BAND (80 = 60 + 20) without ever
#     looking at role. It does NOT claim the >=12/>=12 late-TARGET floors -- only
#     the role-AGNOSTIC late CANDIDATE availability (enough late/b300_399 and
#     late/b200_299 PROPOSALS to POTENTIALLY meet those floors), which is NECESSARY
#     but NOT SUFFICIENT for them.
#   * DISJOINTNESS is not provable here either: a proposal carries no
#     `canonical_sha1` (the hash needs a reconstructed state, computed at the screen
#     stage), so this gate cannot even SEE a position identity.
#   Both are STAGE 2 -- the post-screen qualification in Task 6's pure `select`,
#   over the screen's `kept` rows, where role and canonical hashes are known. A
#   geometry that passes HERE can still fail THERE (e.g. ample late/b200_299
#   CANDIDATES, too few of which classify as `target`) -- by design.
#
# SOUNDNESS: feasible=True is NEVER returned on pre-screen evidence alone. It is
# returned ONLY when the constructive WITNESS actually selects QUOTA_PER_PHASE
# positions per phase as whole-game red+black PAIRS -- 30 pair-games per phase, 120
# distinct games, each used by AT MOST ONE pair, which is precisely how the global
# <=MAX_PER_GAME cap is honoured -- with >=MIN_PLY_GAP spacing, >=12 late positions
# in EACH floor CANDIDATE cell, and a whole-game split into the frozen 160/80
# budgets with |red-black| <= SIDE_TOL per split. A successful witness IS a feasible
# selection, so the gate can never be FALSE-feasible; it may be mildly conservative
# (false-INfeasible) for exotic geometry -- an accepted, documented limitation that
# never yields a silent false pass (v1 says exactly this at :465-472).
#
# The pre-screen checks come in TWO KINDS, and they carry DIFFERENT strengths. Do
# not read them as one thing:
#
#   * The two CAPACITY checks (`phase-capacity`, `late-candidate`) are TRUE UPPER
#     BOUNDS on what ANY selection could realize, so falling short of one genuinely
#     PROVES infeasibility. Passing them proves nothing: they are per-phase /
#     per-cell, so they cannot see the CROSS-PHASE coupling of the GLOBAL
#     <=MAX_PER_GAME cap (v1 §11.2.3's "necessary != sufficient", transposed from
#     bands to phases) -- which is exactly why the WITNESS, not they, governs
#     feasible=True.
#   * The two SIDE-ALIASING checks (`side-aliasing:{phase}`,
#     `side-aliasing:late/{band}`) are NOT upper bounds on feasibility. They bound
#     only the PAIR-BASED WITNESS STRATEGY: they fire precisely when the witness
#     could not have found its pair-games anyway, and so only turn a downstream
#     `joint-*` refusal into a sharper diagnostic -- they never change a verdict. A
#     geometry can fail them and still be genuinely FEASIBLE, because per-split side
#     balance is a per-SPLIT constraint, not a per-phase one: one game may supply two
#     SAME-SIDE rows from DIFFERENT phases (e.g. black@13 opening + black@29
#     early_mid, 16 plies apart), so a corpus can balance ACROSS phases with no
#     single phase holding a side-opposed pair-game at all. Refusing such a geometry
#     is a FALSE-INFEASIBLE -- the conservative direction, never a false pass -- and
#     it is pinned, with a constructive counter-selection, by
#     tests/test_fpu_dev_corpus_v2.py::
#     test_v2_side_aliasing_bounds_the_witness_strategy_not_feasibility.
#     It cannot fire on real `enumerate_v2_proposals` output at all: that enumerator
#     emits ONLY side-opposed pairs, so pair_games * PAIR_POSITIONS == realizable
#     identically and the capacity check always fires first. These are a contract
#     guard on the PURE core, which accepts ANY proposal geometry.

# Per-phase TOTAL candidate quota implied by the frozen SPLIT_ALLOC_V2 (45 target +
# 15 control = 60 for every phase). DERIVED (not hard-coded) so it cannot drift from
# the real allocation; asserted uniform across phases, and asserted to tile
# CORPUS_SIZE exactly -- which is what lets the witness's 4 x 60 selection BE the
# whole 240-row corpus.
_PER_PHASE_TOTALS: Dict[str, int] = {}
for (_pf_role, _pf_phase), _pf_alloc in SPLIT_ALLOC_V2.items():
    _PER_PHASE_TOTALS[_pf_phase] = (_PER_PHASE_TOTALS.get(_pf_phase, 0)
                                    + _pf_alloc["tuning"] + _pf_alloc["frozen_check"])
assert set(_PER_PHASE_TOTALS) == set(PHASES), _PER_PHASE_TOTALS
assert len(set(_PER_PHASE_TOTALS.values())) == 1, _PER_PHASE_TOTALS
QUOTA_PER_PHASE: int = _PER_PHASE_TOTALS[PHASES[0]]        # 60
assert QUOTA_PER_PHASE * len(PHASES) == CORPUS_SIZE, QUOTA_PER_PHASE
# The witness's whole-game split targets the SAME frozen budgets the sampler fills.
assert sum(SPLIT_TOTALS.values()) == CORPUS_SIZE, SPLIT_TOTALS

# The witness's ATOMIC UNIT: a side-opposed pair is exactly 2 positions -- one
# red-to-move ply + one black-to-move ply. Deliberately its OWN name rather than a
# third use of MAX_PER_GAME: that constant is the per-game selection CAP, and this is
# a pair's SIZE. They are numerically equal here, and the witness's whole mechanism
# depends on that (one pair per game is what makes "<=MAX_PER_GAME rows per game,
# GLOBALLY across phases" true by construction) -- so tie them with an assert instead
# of silently reusing one name for both meanings.
PAIR_POSITIONS = 2
assert PAIR_POSITIONS <= MAX_PER_GAME, (PAIR_POSITIONS, MAX_PER_GAME)
assert PAIR_POSITIONS == MAX_PER_CELL_PER_GAME, MAX_PER_CELL_PER_GAME   # 0-or-2/cell
# Even, so `quota // PAIR_POSITIONS` whole red+black pairs realize a phase EXACTLY
# (v1 relies on the same property of QUOTA_PER_BAND = 80).
assert QUOTA_PER_PHASE % PAIR_POSITIONS == 0, QUOTA_PER_PHASE

# The one phase whose PROPOSAL_CELLS split by band -- hence the only phase the late
# floors can constrain. Read from LATE_TARGET_CELL (the SAMPLER's own floor cell)
# rather than restating the "late" literal, so a PHASES/role rename cannot silently
# orphan the preflight's floor pass from the sampler's floors.
LATE_PHASE: str = LATE_TARGET_CELL[1]
assert LATE_PHASE in PHASES, LATE_PHASE

# The three late CANDIDATE cells' bands, in PROPOSAL_CELLS order (b400_plus,
# b300_399, b200_299). All three are reported as diagnostics; only the
# LATE_TARGET_FLOORS bands are CHECKED (b400_plus has no floor -- it is the abundant
# one, and the whole point of the floors is that it must not crowd the others out).
LATE_CELL_BANDS: Tuple[str, ...] = tuple(
    cell[1] for cell in PROPOSAL_CELLS if cell[0] == LATE_PHASE)
assert set(LATE_TARGET_FLOORS) <= set(LATE_CELL_BANDS), LATE_TARGET_FLOORS


def _floor_pair_games(floor: int) -> int:
    """Whole side-opposed PAIR-games the witness must reserve in a floor candidate
    cell to realize `floor` positions there. CEIL division, so an odd floor would be
    over-satisfied, never under-: the witness's unit of selection is a whole
    PAIR_POSITIONS-sized pair, never a lone row."""
    return -(-floor // PAIR_POSITIONS)


# The floor reservation must FIT inside the late phase's own pair quota (6 + 6 = 12
# pair-games out of 30) -- else the floor pass would over-subscribe the phase it is
# supposed to live inside.
assert sum(_floor_pair_games(f) for f in LATE_TARGET_FLOORS.values()) <= (
    QUOTA_PER_PHASE // PAIR_POSITIONS), LATE_TARGET_FLOORS


@dataclasses.dataclass(frozen=True)
class V2PreflightReport:
    """Structured result of `v2_geometry_feasibility` (v1's `PreflightReport`, re-cut
    for phases + late candidate cells).

    `feasible` is True IFF the constructive `witness` -- a real CORPUS_SIZE-row
    selection satisfying every constraint this gate claims -- was built. The
    remaining fields are the per-phase / per-late-cell DIAGNOSTICS that name which
    constraint bound when `feasible` is False (and, on success, corroborate the
    witness).

    ROLE-AGNOSTIC BY CONSTRUCTION: there is deliberately no role/target field and no
    hash/disjointness field, because a PROPOSAL carries neither -- so a passing
    report cannot even express (let alone claim) the target-role floors or
    disjointness. Both are Task 6's post-screen qualification. `late_candidate_floors`
    holds the SAME NUMBERS as LATE_TARGET_FLOORS but a strictly WEAKER meaning: the
    count of late CANDIDATES (any role) the geometry must supply for those
    target-role floors to remain POSSIBLE.
    """
    feasible: bool
    binding_constraint: Optional[str]
    quota_per_phase: int
    late_candidate_floors: Dict[str, int]        # role-AGNOSTIC candidate floors
    n_games: int
    n_proposals: int
    realizable_by_phase: Dict[str, int]          # positions realizable, <=2/game + gap
    pair_games_by_phase: Dict[str, int]          # distinct whole-game red+black pairs
    red_by_phase: Dict[str, int]                 # total red candidate positions
    black_by_phase: Dict[str, int]               # total black candidate positions
    realizable_by_late_cell: Dict[str, int]      # ...same two, per late CANDIDATE cell
    pair_games_by_late_cell: Dict[str, int]      #    (keyed by the cell's BAND)
    witness: Optional[Tuple[dict, ...]]          # the selected rows (split-stamped)

    def format(self) -> str:
        head = ("FEASIBLE" if self.feasible
                else f"INFEASIBLE (binding: {self.binding_constraint})")
        lines = [
            f"[v2-preflight] {head}",
            f"  games={self.n_games} proposals={self.n_proposals} "
            f"quota/phase={self.quota_per_phase}",
            "  ROLE-AGNOSTIC: proves CANDIDATE capacity/availability only -- NOT the "
            "target-role floors, NOT disjointness (both: post-screen `select`)",
        ]
        for phase in PHASES:
            lines.append(
                f"  {phase}: realizable={self.realizable_by_phase.get(phase, 0)} "
                f"pair-games={self.pair_games_by_phase.get(phase, 0)} "
                f"red={self.red_by_phase.get(phase, 0)} "
                f"black={self.black_by_phase.get(phase, 0)}")
        for band in LATE_CELL_BANDS:
            floor = self.late_candidate_floors.get(band)
            floor_s = "" if floor is None else f" candidate-floor={floor}"
            lines.append(
                f"  {LATE_PHASE}/{band}: "
                f"realizable={self.realizable_by_late_cell.get(band, 0)} "
                f"pair-games={self.pair_games_by_late_cell.get(band, 0)}{floor_s}")
        return "\n".join(lines)

    __str__ = format


def _opposed_pair(rows: List[dict], gap: int):
    """The deterministic earliest-satisfying side-opposed pair among `rows` (one
    game's proposals in one phase or one candidate cell), or None.

    Splits `rows` by side, sorts each ascending by ply -- `_first_gap_pair`'s
    precondition -- and delegates to it (v1's own search, build_fpu_dev_corpus.py:578;
    imported, never copied). NOT `_fit_pair`: that one is ply-bucket-CAP aware and v2
    has no bucket cap.
    """
    reds = sorted((r for r in rows if r["side"] == "red"), key=lambda r: r["ply"])
    blacks = sorted((r for r in rows if r["side"] == "black"), key=lambda r: r["ply"])
    return _first_gap_pair(reds, blacks, gap)


def _build_v2_witness(proposals_by_game, phases, quota_per_phase,
                      late_candidate_floors, max_per_game, min_gap, side_tol):
    """Constructive witness (v1's `_build_witness`, re-cut for phase cells).

    Greedily select `quota_per_phase` positions per phase as whole-game red+black
    PAIRS, honouring the JOINT constraints a per-phase accounting cannot see:

      * GLOBAL <=max_per_game/game -- a game consumed for a pair in ANY phase is
        SPENT (`used_games`), so it can never serve a second phase, and it yields
        exactly ONE PAIR_POSITIONS-sized pair. The cap therefore holds BY
        CONSTRUCTION (PAIR_POSITIONS <= MAX_PER_GAME, asserted at module level), which
        is why `max_per_game` appears below only in the diagnostics that EXPLAIN a
        failure, never as a running budget. This CROSS-PHASE coupling is exactly what
        the per-phase necessary checks miss, and it is why the witness -- not those
        checks -- governs feasible=True.
      * >=min_gap within a game -- free, and total: a game gives at most ONE pair, and
        `_first_gap_pair` only ever returns a pair already >= min_gap apart.
      * >=12 / >=12 late CANDIDATE coverage -- PASS 1 below.
      * the whole-game 160/80 split, with |red - black| <= side_tol per split.

    PASS 1 (late floor RESERVATION) runs BEFORE any phase's ordinary fill, not merely
    before the late phase's own. The floor cells are the SCARCEST resource on a real
    reservoir -- b300_399 needs ply >= 129 and b200_299 ply >= 229 (Task 0's
    `n_legal >= 528 - ply`), so ONLY long games carry them, while the opening /
    early_mid / midgame fills are happy with ANY game and would otherwise consume
    those long games first (they are drawn in ascending game_idx, which is arbitrary
    w.r.t. length). Reserving the 6 + 6 floor pair-games up front costs the other
    phases nothing they cannot replace and removes a whole class of false
    infeasibility. It is still exactly the brief's witness: the floor pairs are taken
    BEFORE the remainder is filled from any late cell.

    PASS 2 fills each phase to its `quota_per_phase // PAIR_POSITIONS` pair-games, in
    `phases` order, from the games PASS 1 left. The late phase's pass counts the floor
    pairs PASS 1 already claimed for it, and draws its remainder from the phase's whole
    candidate set (any late cell) -- which is exactly what the real sampler's
    (role, "late") cell does: it aggregates all three late proposal cells (see
    `_choose_positions_v2`).

    PRECONDITION: `late_candidate_floors` is the caller's EFFECTIVE floors -- already
    empty when LATE_PHASE is out of `phases`, so PASS 1 can never reserve games for a
    phase PASS 2 will not fill. `v2_geometry_feasibility` resolves that once, and is
    the only caller.

    Returns (selected_rows, None) on success, else (None, binding_constraint). This
    is a GREEDY, so it can be CONSERVATIVE (false-infeasible on exotic geometry a
    perfect selection could satisfy) -- but never a silent false PASS: what it
    returns IS a valid selection.
    """
    # Index each game's proposals by PHASE (pooled across the phase's cells -- what
    # the sampler's (role, phase) cell aggregates) and, for the late phase, by
    # CANDIDATE CELL (band-restricted -- what a floor actually counts). Only the FLOOR
    # cells are indexed: PASS 1 draws from those alone, and the late phase's remainder
    # comes from `per_game_phase` (all late cells pooled), so indexing the unfloored
    # b400_plus cell here would build lists nothing ever reads.
    per_game_phase: Dict[Any, Dict[str, List[dict]]] = {}
    per_game_late_cell: Dict[Any, Dict[str, List[dict]]] = {}
    for gi in sorted(proposals_by_game):
        by_phase: Dict[str, List[dict]] = defaultdict(list)
        by_cell: Dict[str, List[dict]] = defaultdict(list)
        for p in proposals_by_game[gi]:
            if p["phase"] in phases:
                by_phase[p["phase"]].append(p)
            cell = p["proposal_cell"]
            if cell[0] == LATE_PHASE and cell[1] in late_candidate_floors:
                by_cell[cell[1]].append(p)
        per_game_phase[gi] = by_phase
        per_game_late_cell[gi] = by_cell

    used_games: Set[Any] = set()
    selected: List[dict] = []
    selected_games_in_order: List[Any] = []
    got_pairs: Counter = Counter()                 # pair-games claimed FOR each phase
    pairs_per_phase = quota_per_phase // PAIR_POSITIONS

    def claim(gi, pair, phase) -> None:
        used_games.add(gi)                         # spent GLOBALLY, across all phases
        selected_games_in_order.append(gi)
        got_pairs[phase] += 1
        for row in sorted(pair, key=lambda r: r["ply"]):
            selected.append(dict(row))

    def draw(source, key, phase, want) -> int:
        """Claim unused pair-games from `source[gi][key]` (ascending game_idx) until
        `want` of them are held or no game can supply one. Returns how many."""
        got = 0
        for gi in sorted(source):
            if got >= want:
                break
            if gi in used_games:
                continue
            rows = source[gi].get(key)
            if not rows:
                continue
            pair = _opposed_pair(rows, min_gap)
            if pair is None:
                continue
            claim(gi, pair, phase)
            got += 1
        return got

    # --- PASS 1: reserve the late floor CANDIDATE cells' pair-games (role-agnostic).
    # No LATE_PHASE-in-`phases` guard: the caller already resolved that into the
    # floors themselves (see PRECONDITION above), so an out-of-scope late phase
    # arrives here as an empty dict and this loop is simply a no-op.
    for band, floor in late_candidate_floors.items():
        want = _floor_pair_games(floor)
        got = draw(per_game_late_cell, band, LATE_PHASE, want)
        if got < want:
            return None, (
                f"joint-late-floor:{LATE_PHASE}/{band} (witness realized "
                f"{got * PAIR_POSITIONS} < {floor} CANDIDATE positions under the "
                f"JOINT per-game cap (<={max_per_game}/game) + >={min_gap}-gap "
                f"constraints)")

    # --- PASS 2: fill each phase to its pair quota from the games PASS 1 left.
    for phase in phases:
        want = pairs_per_phase - got_pairs[phase]
        draw(per_game_phase, phase, phase, want)
        if got_pairs[phase] < pairs_per_phase:
            return None, (
                f"joint-phase-quota:{phase} (witness realized "
                f"{got_pairs[phase] * PAIR_POSITIONS} < {quota_per_phase} positions "
                f"under the JOINT per-game cap (<={max_per_game}/game) + "
                f">={min_gap}-gap constraints; "
                f"{len(proposals_by_game) - len(used_games)} of "
                f"{len(proposals_by_game)} games still unused)")

    # --- Whole-game split into the frozen 160/80 budgets. Every selected game gives
    # exactly one side-neutral (red+black) pair, so ANY whole-game partition is
    # side-balanced; place each game in the first split with room (v1's rule).
    #
    # This tail is the one part that structurally ECHOES v1's `_build_witness`, because
    # it implements the same FROZEN whole-game 160/80 rule. It cannot be shared: v1
    # budgets against its own private `_SPLIT_POS_BUDGET` (derived from v1's
    # SPLIT_ALLOC), while v2 must budget against SPLIT_TOTALS (derived from
    # SPLIT_ALLOC_V2) -- two different sources of truth that merely happen to agree
    # today -- and v1 is frozen byte-identical, so the budget cannot be parameterized
    # out of it.
    rows_by_game: Dict[Any, List[dict]] = defaultdict(list)
    for row in selected:
        rows_by_game[row["game_idx"]].append(row)
    split_of: Dict[Any, str] = {}
    filled = {s: 0 for s in SPLITS}
    for gi in selected_games_in_order:
        n = len(rows_by_game[gi])
        placed = False
        for split in SPLITS:
            if filled[split] + n <= SPLIT_TOTALS[split]:
                split_of[gi] = split
                filled[split] += n
                placed = True
                break
        if not placed:                             # no split has room (over-selection)
            return None, "split-budget:overflow (no split had room for a game)"
    for row in selected:
        row["split"] = split_of[row["game_idx"]]

    # Verify the split budgets and per-split side balance FROM THE SELECTED ROWS
    # (belt-and-suspenders: all-pairs => imbalance 0, but VERIFY a real witness rather
    # than assume it -- the same exact-or-refuse contract `_select_manifest` holds).
    side = {s: Counter() for s in SPLITS}
    per_split_pos: Counter = Counter()
    for row in selected:
        side[row["split"]][row["side"]] += 1
        per_split_pos[row["split"]] += 1
    for split in SPLITS:
        if per_split_pos[split] != SPLIT_TOTALS[split]:
            return None, (f"split-budget:{split} (realized {per_split_pos[split]} "
                          f"!= {SPLIT_TOTALS[split]})")
        imbalance = abs(side[split]["red"] - side[split]["black"])
        if imbalance > side_tol:
            return None, (f"side-balance:{split} (|red-black|={imbalance} "
                          f"> {side_tol})")
    return selected, None


def v2_geometry_feasibility(
        proposals_by_game: Mapping[Any, List[Mapping[str, Any]]], *,
        quota_per_phase: int = QUOTA_PER_PHASE,
        late_candidate_floors: Mapping[str, int] = LATE_TARGET_FLOORS,
        max_per_game: int = MAX_PER_GAME,
        min_gap: int = MIN_PLY_GAP,
        side_tol: int = SIDE_TOL,
        phases: Tuple[str, ...] = PHASES) -> V2PreflightReport:
    """Pure JOINT geometric feasibility core -- STAGE 1 of the two-stage split
    (design Sec 1.7). ROLE-AGNOSTIC; see this section's header for the full scope and
    the two explicit NON-claims (target-role floors, disjointness).

    `proposals_by_game`: {game_idx: [proposal dicts]} -- pure geometry rows in
    `enumerate_v2_proposals`' schema (game_idx, ply, side, phase, n_legal, band,
    proposal_cell). No file paths, no NN, no hashes, no roles.

    Computes the per-phase / per-late-cell capacity DIAGNOSTICS, short-circuits with a
    NAMED binding constraint on any PRE-SCREEN violation, and only then attempts the
    constructive `_build_v2_witness` that GOVERNS feasible=True. Deterministic (sorted
    iteration throughout).
    """
    # The late floors only mean anything while the late PHASE is in scope. Resolve the
    # EFFECTIVE floors ONCE, here, so the pre-screen checks below and the witness's own
    # PASS-1 reservation can never disagree about whether they apply. (`phases` is a
    # v1-parallel knob; only a caller that narrows it can make the two differ, and the
    # default always contains LATE_PHASE.) The report carries these EFFECTIVE floors --
    # i.e. the ones actually enforced -- not the requested ones.
    late_floors: Dict[str, int] = (
        dict(late_candidate_floors) if LATE_PHASE in phases else {})

    realizable = {p: 0 for p in phases}
    pair_games = {p: 0 for p in phases}
    red_tot = {p: 0 for p in phases}
    black_tot = {p: 0 for p in phases}
    late_realizable = {b: 0 for b in LATE_CELL_BANDS}
    late_pair_games = {b: 0 for b in LATE_CELL_BANDS}
    n_proposals = 0

    for gi in sorted(proposals_by_game):
        by_phase: Dict[str, List[dict]] = defaultdict(list)
        by_late_cell: Dict[str, List[dict]] = defaultdict(list)
        for p in proposals_by_game[gi]:
            n_proposals += 1
            if p["phase"] in realizable:
                by_phase[p["phase"]].append(p)
            cell = p["proposal_cell"]
            if cell[0] == LATE_PHASE and cell[1] in late_realizable:
                by_late_cell[cell[1]].append(p)

        # Per-PHASE capacity. `_gap_selectable` caps this game's contribution at
        # max_per_game PER PHASE -- which OVER-states a multi-phase game's real
        # capacity, since v2's <=max_per_game budget is GLOBAL across phases and is
        # counted once per phase here. A true UPPER BOUND, hence NECESSARY only: this
        # is precisely the cross-phase coupling only the witness can see.
        for phase, rows in by_phase.items():
            plies = sorted(r["ply"] for r in rows)
            realizable[phase] += _gap_selectable(plies, min_gap, max_per_game)
            red_tot[phase] += sum(1 for r in rows if r["side"] == "red")
            black_tot[phase] += sum(1 for r in rows if r["side"] == "black")
            if _opposed_pair(rows, min_gap) is not None:
                pair_games[phase] += 1

        # Per-late-CELL candidate availability -- the same two measures, band-
        # restricted, because a floor counts positions in ONE band, not in the phase.
        for band, rows in by_late_cell.items():
            plies = sorted(r["ply"] for r in rows)
            late_realizable[band] += _gap_selectable(plies, min_gap, max_per_game)
            if _opposed_pair(rows, min_gap) is not None:
                late_pair_games[band] += 1

    def _report(feasible, binding, witness):
        return V2PreflightReport(
            feasible=feasible, binding_constraint=binding,
            quota_per_phase=quota_per_phase,
            late_candidate_floors=dict(late_floors),
            n_games=len(proposals_by_game), n_proposals=n_proposals,
            realizable_by_phase=dict(realizable),
            pair_games_by_phase=dict(pair_games),
            red_by_phase=dict(red_tot), black_by_phase=dict(black_tot),
            realizable_by_late_cell=dict(late_realizable),
            pair_games_by_late_cell=dict(late_pair_games),
            witness=witness)

    # --- PRE-SCREEN checks (fast; each names its binding constraint). NONE of them
    # ever drives feasible=True -- they only short-circuit a refusal with a reason.
    # They are of TWO DIFFERENT STRENGTHS; see this section's header.

    # (1) CAPACITY -- TRUE UPPER BOUNDS on what any selection could realize, so
    # falling short of one genuinely PROVES infeasibility.
    for phase in phases:
        if realizable[phase] < quota_per_phase:
            return _report(False, f"phase-capacity:{phase} (realizable "
                           f"{realizable[phase]} < quota {quota_per_phase})", None)
    for band, floor in late_floors.items():
        if late_realizable.get(band, 0) < floor:
            return _report(False, f"late-candidate:{LATE_PHASE}/{band} (realizable "
                           f"{late_realizable.get(band, 0)} < candidate floor {floor}; "
                           f"ROLE-AGNOSTIC -- the target-role floor is the "
                           f"post-screen `select` stage's)", None)

    # (2) SIDE-ALIASING -- NOT upper bounds on feasibility: they bound the PAIR-BASED
    # WITNESS STRATEGY only. Each fires exactly when the witness could not have found
    # its pair-games anyway, so they never change a verdict -- they only turn the
    # downstream `joint-*` refusal into a sharper diagnostic. A geometry that fails
    # one CAN still be genuinely feasible (a corpus may balance ACROSS phases via
    # same-side rows drawn from different phases), in which case refusing it is a
    # FALSE-INFEASIBLE: conservative, never a false pass. See the header.
    for phase in phases:
        if pair_games[phase] * PAIR_POSITIONS < quota_per_phase:
            return _report(False, f"side-aliasing:{phase} (both-side pair-games "
                           f"{pair_games[phase]} < "
                           f"{quota_per_phase // PAIR_POSITIONS} the pair-based "
                           f"witness needs; conservative -- bounds the WITNESS "
                           f"STRATEGY, not feasibility)", None)
    for band, floor in late_floors.items():
        want = _floor_pair_games(floor)
        if late_pair_games.get(band, 0) < want:
            return _report(False, f"side-aliasing:{LATE_PHASE}/{band} (both-side "
                           f"pair-games {late_pair_games.get(band, 0)} < {want} the "
                           f"pair-based witness needs for the late CANDIDATE floor; "
                           f"conservative -- bounds the WITNESS STRATEGY, not "
                           f"feasibility)", None)

    # --- The constructive WITNESS GOVERNS feasible=True (pre-screen != sufficient).
    witness, binding = _build_v2_witness(
        proposals_by_game, phases, quota_per_phase, late_floors,
        max_per_game, min_gap, side_tol)
    if witness is None:
        return _report(False, binding, None)
    return _report(True, None, tuple(witness))


def v2_preflight_source(records: List[Mapping[str, Any]]) -> V2PreflightReport:
    """I/O wrapper (the ONLY impure part of this module -- kept thin, mirroring v1's
    `preflight_source`, build_fpu_dev_corpus.py:785). Read each `rec["replay_path"]`,
    run the REAL `enumerate_v2_proposals` on it -- so the preflight can never drift
    from the enumeration the screen will actually use -- and hand the result to the
    pure `v2_geometry_feasibility`. ALL feasibility logic lives in that pure core;
    this only does the file reads.

    The SOURCE INDEX record's `game_idx` is authoritative and overrides any stored in
    the replay file, exactly as v1's `build_candidates_by_game` keys by the record's
    game_idx and never the replay's.
    """
    proposals_by_game: Dict[Any, List[dict]] = {}
    for rec in records:
        replay = json.loads(Path(rec["replay_path"]).read_text())
        replay = {**replay, "game_idx": rec["game_idx"]}
        proposals_by_game[rec["game_idx"]] = enumerate_v2_proposals(replay)
    return v2_geometry_feasibility(proposals_by_game)


# =============================================================================
# OPERATOR SHELL (Task 5: the `screen` stage, its config loader, and `main`;
# Task 6 appends the pure `select` stage below this and extends `main` with
# `--mode select`).
#
# Frozen design ref: docs/superpowers/specs/2026-07-12-fpu-dev-corpus-v2-phase-design.md
#   Sec 1.6 (two-artifact `screen`/`select` workflow), Sec 1.7 (two-stage
#   feasibility -- the geometric preflight above GATES this stage), Sec 1.8
#   (the required versioned config).
#
# `run_screen()` / `main --mode screen` are an OPERATOR phase: `run_screen`
# loads a real checkpoint (`config.checkpoint`), reconstructs real reservoir
# replay positions, and runs 400-sim MCTS. NEITHER is ever invoked by this
# task's tests -- every test exercises only the pure `classify_exclusion` /
# `screen_row` / `load_v2_config`, plus STATIC verification (signature/
# source-text/`importlib.util.find_spec` inspection, never execution) that
# the operator path is wired correctly.
#
# Nothing ABOVE this banner imports MCTS/GPU/MLX or performs I/O beyond
# `v2_preflight_source`'s own thin, stdlib-only file reads. Below it,
# `SCREEN_FIELDNAMES` / `classify_exclusion` / `screen_row` / `V2Config` /
# `load_v2_config` / `write_screen_csv` / `v2_screen_provenance` /
# `write_screen_meta` / `_parse_v2_args` stay equally stdlib-only and
# importable without GPU/MLX. The GPU/MLX/checkpoint/evaluator modules
# (`.eval_runner`, `.mcts`'s MCTS class, `.build_teacher_calibration_manifest`)
# are imported LAZILY, inside the two functions that actually need each --
# `_build_v2_anchor_search_fn` and `run_screen` -- exactly as
# build_fpu_dev_corpus.py's own `main` / `_build_anchor_search_fn` /
# `_scan_two_stage` do (see that file's OWN banner at :375-392).
#
# `_build_anchor_search_fn` / `_anchor_seed` / `ANCHOR_SIMS` / `_scan_two_stage`
# are DELIBERATELY NOT imported from build_fpu_dev_corpus here, unlike the
# pure helpers imported at the top of this file: the Global Constraints
# reserve cross-module reuse to v1's PURE helpers only ("do NOT alter the v1
# build_fpu_dev_corpus.py behavior except by importing its pure helpers"),
# and those four are v1's own IMPURE operator plumbing (checkpoint/GPU/MLX
# work) -- so v2 MIRRORS their shape with its own `_build_v2_anchor_search_fn`
# / `_v2_anchor_seed` / `ANCHOR_SIMS_V2` / `run_screen` instead (design Sec 2:
# "the fpu-off anchor + raw-policy forward pass mirror the v1 shell").
# `_teacher_infer` (from `.build_teacher_calibration_manifest`, a DIFFERENT
# module than build_fpu_dev_corpus) IS reused verbatim, imported lazily
# inside `run_screen`, exactly as v1's own `_scan_two_stage` does.
# =============================================================================

# ---------------------------------------------------------------------------
# Screen row schema + the two pure per-proposal helpers (design Sec 1.6)
# ---------------------------------------------------------------------------

# The screen artifact's COMPLETE row schema, in this exact order. `phase` and
# `ply_bucket` deliberately carry the SAME value (`screen_row` always sets
# `ply_bucket = proposal["phase"]`) under two distinct column names: Task 7's
# diagnostic opts into stratifying by `ply_bucket` (design Sec 1.4), so every
# row must carry it independently of `phase`, even though the two never
# diverge today (`enumerate_v2_proposals` derives `phase` from
# `ply_bucket_of`). `band` is the recorded branching covariate (Sec 1.2/1.3);
# `proposal_cell` is the enumerator's own (phase, band-or-None) cell tuple,
# carried through for review. The four raw-policy geometry columns
# (`normalized_entropy` .. `top4_mass`/`top8_mass`) and `raw_policy_role` are
# nullable (null on `collision`, populated but possibly role=None on
# `ineligible_role`); `root_value_stm`/`anchor_eligible` are nullable
# precisely when `anchor_run` is False.
SCREEN_FIELDNAMES: List[str] = [
    "game_idx", "ply", "side", "phase", "n_legal", "band", "ply_bucket",
    "proposal_cell", "normalized_entropy", "top1_prior", "top4_mass",
    "top8_mass", "raw_policy_role", "anchor_run", "root_value_stm",
    "anchor_eligible", "canonical_sha1", "exclusion_status",
]


def classify_exclusion(*, collided: bool, role: Optional[str],
                       anchor_eligible_val: Optional[bool]) -> Tuple[str, bool]:
    """Pure per-proposal exclusion classifier (design Sec 1.6) -- the SAME
    facts `run_screen` computes while scoring one proposal, in the SAME
    cheap-filters-before-the-anchor order:

      1. `collided` (sha1 in forbidden-union-kept) -> `"collision"`,
         `anchor_run=False` -- checked FIRST and UNCONDITIONALLY: a collision
         short-circuits regardless of `role`/`anchor_eligible_val` (the
         raw-policy pass never even ran for a collided proposal, so those
         arguments are meaningless here -- see `screen_row`'s own
         `feats=None` contract for the collision path).
      2. else `role is None` (the raw-policy pass ran but landed in the grey
         zone) -> `"ineligible_role"`, `anchor_run=False` -- the anchor is
         NEVER run for either of the two cheap-filter rejections above.
      3. else (role is "target" or "control", so BOTH cheap filters passed
         and the 400-sim anchor DID run) -> `anchor_eligible_val` decides
         `"ineligible_anchor"` (False) or `"kept"` (True); `anchor_run=True`
         in both of these cases.

    `anchor_eligible_val` is named (not `anchor_eligible`, the imported
    Stage-2 predicate `run_screen` calls to PRODUCE this value) so this
    function's own parameter can never shadow that import.
    """
    if collided:
        return "collision", False
    if role is None:
        return "ineligible_role", False
    if not anchor_eligible_val:
        return "ineligible_anchor", True
    return "kept", True


def screen_row(proposal: Mapping[str, Any], *, feats: Optional[Mapping[str, float]],
               role: Optional[str], anchor_run: bool,
               root_value_stm: Optional[float], anchor_eligible: Optional[bool],
               canonical_sha1: str, exclusion_status: str) -> dict:
    """Assemble ONE complete `SCREEN_FIELDNAMES` row (design Sec 1.6) from a
    Task-2 `enumerate_v2_proposals` proposal dict (`game_idx, ply, side,
    phase, n_legal, band, proposal_cell`) plus this proposal's own screening
    outcome. Pure dict assembly -- v1's `_manifest_row` shape -- performing NO
    filtering/classification itself (`classify_exclusion` decides
    `exclusion_status`/`anchor_run`; this only projects the result into the
    frozen schema) and touching no evaluator/MCTS.

    `ply_bucket` is ALWAYS set to `proposal["phase"]` -- see `SCREEN_FIELDNAMES`'s
    own comment for why the schema carries both under separate names.

    `feats=None` (the COLLISION path -- the raw-policy forward pass never
    ran) leaves all four policy-geometry columns `None`: explicitly NULL,
    never a fabricated `0.0` (a real `0.0` would misreport a
    maximally-concentrated prior that was never observed). Any other `feats`
    (a real `_policy_features_from_priors(...)` result) populates all four,
    REGARDLESS of whether `role` itself is `None` -- `ineligible_role` rows
    still carry their (grey-zone) policy geometry; only `raw_policy_role`
    itself is `None` there.

    `anchor_run` and `root_value_stm`/`anchor_eligible` nullness are a
    two-way CONTRACT, asserted here rather than silently trusted: `run_screen`
    -- this function's one caller -- derives `anchor_run` via
    `classify_exclusion` from the SAME facts that determine whether the
    anchor search actually ran, so the two can never legitimately disagree.
    """
    if anchor_run:
        assert root_value_stm is not None and anchor_eligible is not None, (
            "screen_row: anchor_run=True requires a non-null root_value_stm "
            "and anchor_eligible (the anchor DID run)")
    else:
        assert root_value_stm is None and anchor_eligible is None, (
            "screen_row: anchor_run=False requires null root_value_stm and "
            "anchor_eligible (the anchor was never run)")

    if feats is None:
        normalized_entropy = top1_prior = top4_mass = top8_mass = None
    else:
        normalized_entropy = feats["normalized_entropy"]
        top1_prior = feats["top1_prior"]
        top4_mass = feats["top4_mass"]
        top8_mass = feats["top8_mass"]

    return {
        "game_idx": proposal["game_idx"],
        "ply": proposal["ply"],
        "side": proposal["side"],
        "phase": proposal["phase"],
        "n_legal": proposal["n_legal"],
        "band": proposal["band"],
        "ply_bucket": proposal["phase"],
        "proposal_cell": proposal["proposal_cell"],
        "normalized_entropy": normalized_entropy,
        "top1_prior": top1_prior,
        "top4_mass": top4_mass,
        "top8_mass": top8_mass,
        "raw_policy_role": role,
        "anchor_run": anchor_run,
        "root_value_stm": root_value_stm,
        "anchor_eligible": anchor_eligible,
        "canonical_sha1": canonical_sha1,
        "exclusion_status": exclusion_status,
    }


# ---------------------------------------------------------------------------
# The required v2 config (design Sec 1.8) -- "controller resolution": defined
# HERE (not Task 6) because `main --mode screen` already needs it to record
# the config's own hash in the screen's `.meta.json`. Task 6 reuses
# `load_v2_config` unchanged and adds its own tests pinning its required-key
# behavior; validating `expected_fingerprints` against a real screen's meta
# is Task 6's `validate_screen_identities`, not this loader's job.
# ---------------------------------------------------------------------------

# v1-matching CLI defaults (build_fpu_dev_corpus.py `_parse_args`) for the
# two pure evaluator-throughput knobs -- the SOLE config keys with a
# builder-side default; every other key below is REQUIRED (no default
# source, no default stride -- design Sec 1.8).
DEFAULT_EVAL_BATCH_SIZE = 14
DEFAULT_STALL_FLUSH_SIMS = 48

# Every REQUIRED top-level config key (design Sec 1.8): source reservoir
# index path + predeclared seed range, selection seed, phase allocation, late
# floors, proposal-enumerator policy params, `new_collapse_stratum`,
# checkpoint, forbidden manifests, output paths (screen + select), and the
# expected-fingerprint block.
_V2_CONFIG_REQUIRED_KEYS: Tuple[str, ...] = (
    "source_index_path",
    "seed_range",
    "selection_seed",
    "phase_allocation",
    "late_floors",
    "enumerator_params",
    "new_collapse_stratum",
    "checkpoint",
    "forbidden_manifests",
    "screen_out",
    "select_out",
    "expected_fingerprints",
)


@dataclasses.dataclass(frozen=True)
class V2Config:
    """The v2 pipeline's ONE required config (design Sec 1.8) -- `screen` and
    (Task 6's) `select` both load the SAME file, so a config-hash mismatch
    between the two stages is detectable (Task 6's
    `validate_screen_identities`). `load_v2_config` is the only constructor.

    config_path: this file's OWN path (set by `load_v2_config`, never read
      from the JSON) -- so `screen` can hash ITSELF into its `.meta.json`
      provenance (the "config hash" the brief requires alongside the Sec 1.8
      fingerprints).
    source_index_path: the reservoir's replay-eval JSONL index (v1's
      `--source-jsonl` analogue; `load_game_index`'s own input shape).
    seed_range: the reservoir's PREDECLARED (start, end) self-play seed range
      (design Sec 1.1) -- evidence the reservoir is a fixed, audited set,
      never silently topped up; carried through as data, not algorithmically
      consumed by `screen` itself.
    selection_seed: Task 6's `select_final_manifest` seed for
      `sample_v2_rows` (`sample_v2_rows(kept, seed=config.selection_seed)`).
    phase_allocation / late_floors / enumerator_params: recorded copies of
      this run's intended SPLIT_ALLOC_V2 / LATE_TARGET_FLOORS / proposal-
      enumerator parameters -- evidentiary. This module's OWN frozen
      constants are what actually governs `enumerate_v2_proposals` /
      `sample_v2_rows`; these fields are not cross-validated against them
      here (YAGNI beyond "present" -- see `load_v2_config`).
    new_collapse_stratum: Task 7's diagnostic knob (e.g. `"ply_bucket"`) --
      carried through as data; consumed only by that diagnostic.
    checkpoint: the ONE checkpoint used for BOTH the raw-policy forward pass
      and the 400-sim fpu-off anchor (design Sec 1.6: "one network, both
      roles").
    forbidden_manifests: CSV manifest path(s) whose canonical hashes
      `screen`'s collision cheap-filter excludes (`load_forbidden_hashes`).
    screen_out / select_out: output paths for the screen artifact
      (`fpu_dev_source_screen.csv`) and (Task 6's) final selected manifest.
    expected_fingerprints: the Sec 1.8 fingerprint block this config EXPECTS
      a screen to match -- carried through as opaque data; hard-matching it
      against a real screen's `.meta.json` is Task 6's
      `validate_screen_identities`.
    eval_batch_size / stall_flush_sims: pure evaluator-throughput knobs
      (never result-determining -- they change how the evaluator BATCHES,
      never WHICH position is screened or its anchor value); default to v1's
      own CLI defaults when the config omits them.
    """
    config_path: str
    source_index_path: str
    seed_range: Tuple[int, int]
    selection_seed: int
    phase_allocation: Dict[str, Any]
    late_floors: Dict[str, Any]
    enumerator_params: Dict[str, Any]
    new_collapse_stratum: str
    checkpoint: str
    forbidden_manifests: Tuple[str, ...]
    screen_out: str
    select_out: str
    expected_fingerprints: Dict[str, Any]
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE
    stall_flush_sims: int = DEFAULT_STALL_FLUSH_SIMS


def load_v2_config(path: str) -> V2Config:
    """Load + validate the v2 pipeline's REQUIRED config (design Sec 1.8).

    Raises `ValueError` naming EVERY missing required key (see
    `_V2_CONFIG_REQUIRED_KEYS`) rather than silently defaulting -- "no
    default source, no default stride." `eval_batch_size` / `stall_flush_sims`
    are the sole exception (see `V2Config`'s own docstring). Task 6 adds its
    OWN tests pinning this required-key behavior; this is the one and only
    implementation of `load_v2_config`.
    """
    raw = json.loads(Path(path).read_text())
    missing = sorted(k for k in _V2_CONFIG_REQUIRED_KEYS if k not in raw)
    if missing:
        raise ValueError(
            f"load_v2_config: {path} is missing required key(s): "
            f"{', '.join(missing)}")
    return V2Config(
        config_path=str(path),
        source_index_path=raw["source_index_path"],
        seed_range=tuple(raw["seed_range"]),
        selection_seed=int(raw["selection_seed"]),
        phase_allocation=raw["phase_allocation"],
        late_floors=raw["late_floors"],
        enumerator_params=raw["enumerator_params"],
        new_collapse_stratum=raw["new_collapse_stratum"],
        checkpoint=raw["checkpoint"],
        forbidden_manifests=tuple(raw["forbidden_manifests"]),
        screen_out=raw["screen_out"],
        select_out=raw["select_out"],
        expected_fingerprints=raw["expected_fingerprints"],
        eval_batch_size=int(raw.get("eval_batch_size", DEFAULT_EVAL_BATCH_SIZE)),
        stall_flush_sims=int(raw.get("stall_flush_sims", DEFAULT_STALL_FLUSH_SIMS)),
    )


# ---------------------------------------------------------------------------
# The fpu-off 400-sim anchor (mirrors v1's `_build_anchor_search_fn` /
# `_anchor_seed` -- re-implemented, not imported; see this banner's own note)
# ---------------------------------------------------------------------------

ANCHOR_SIMS_V2 = 400
# XORs a fixed base with (game_idx, ply) for a deterministic, reproducible
# per-position seed -- v1's own `ANCHOR_SEED_BASE` idiom (2026-07-11, the day
# v1's dev-corpus module was authored). Deliberately a DISTINCT value (the day
# the v2 design was frozen -- see this design doc's own filename) rather than
# a re-export of v1's constant, so v1's and v2's anchor-search RNG streams can
# never collide even where the two pipelines happen to touch the same
# (game_idx, ply) coordinate on overlapping source data.
ANCHOR_SEED_BASE_V2 = 20260712


def _v2_anchor_seed(game_idx: int, ply: int) -> int:
    return ANCHOR_SEED_BASE_V2 ^ int(game_idx) ^ int(ply)


def _build_v2_anchor_search_fn(checkpoint: str, eval_batch_size: int,
                               stall_flush_sims: int):
    """Load ONE evaluator + build the fpu-off 400-sim anchor search_fn.
    Checkpoint/GPU/MLX work -- only ever called from `run_screen`. Mirrors
    v1's `_build_anchor_search_fn` (build_fpu_dev_corpus.py:962) shape
    exactly, re-implemented here rather than imported (this banner's own
    note): same `cfg_from(EvalConfig(...))` base config (so v2's anchor
    shares every OTHER MCTS hyperparameter with v1's, differing only in
    `fpu_policy_mass_reduction`), same `dataclasses.replace(..., fpu_policy_
    mass_reduction=None)` frozen fpu-off override, same single evaluator
    reused for BOTH the raw-policy forward pass and the anchor search (design
    Sec 1.6: "the fpu-off ... anchor + raw policy" -- one network, both
    roles, exactly as v1's `_scan_two_stage` does).
    """
    from .eval_runner import EvalConfig, cfg_from, _default_evaluator_factory
    from .mcts import MCTS
    evaluator = _default_evaluator_factory(checkpoint)
    base_cfg = cfg_from(EvalConfig(mcts_sims=ANCHOR_SIMS_V2,
                                   mcts_eval_batch_size=eval_batch_size,
                                   mcts_stall_flush_sims=stall_flush_sims))
    # Explicit even though it's already the default -- this IS the frozen
    # fpu-off anchor config (design Sec 1.6's `MCTSConfig(fpu_policy_mass_
    # reduction=None)`).
    cfg = dataclasses.replace(base_cfg, fpu_policy_mass_reduction=None)

    def search_fn(state, seed):
        return MCTS(evaluator, cfg, random.Random(seed)).search_with_root(
            state, add_noise=False)

    # Also return the frozen anchor cfg so run_screen() can record its FULL
    # effective MCTS config in the screen's meta (design Sec 1.8), from the
    # single source of truth, rather than reconstructing it.
    return evaluator, search_fn, cfg


# ---------------------------------------------------------------------------
# `run_screen` -- the operator `screen` stage itself (design Sec 1.6)
# ---------------------------------------------------------------------------

def run_screen(config: V2Config) -> Tuple[List[dict], dict]:
    """Operator `screen` stage: the evaluator/MCTS phase. For EVERY proposal
    `enumerate_v2_proposals` yields, over EVERY game in the reservoir
    (ascending game_idx -- `load_game_index`'s own sorted order), apply the
    CHEAP filters -- collision, then raw-policy role -- BEFORE the expensive
    400-sim fpu-off anchor, and persist the outcome of EVERY proposal (kept,
    excluded, or ineligible alike) via `screen_row`. Screening NEVER stops
    early because a cell/reserve has "filled" -- unlike v1's own
    `_scan_two_stage`, v2 defers ALL selection to the separate, pure `select`
    stage (Task 6); this stage is a complete, reusable, reviewable evidence
    artifact over EVERY proposal (design Sec 1.6).

    Order per proposal (design Sec 1.6, exact):
      1. reconstruct the state (`position_state`) and its `canonical_state_sha1`;
      2. cheap filter 1 -- collision (sha1 in forbidden UNION this run's own
         kept-so-far hashes) -> `exclusion_status="collision"`, the anchor is
         NEVER run, and (per `screen_row`'s own contract) the raw-policy
         pass never ran either, so its four feature columns are null too;
      3. cheap filter 2 -- one raw-policy forward pass (`_teacher_infer` ->
         `_policy_features_from_priors`) then `raw_policy_role`; `None` ->
         `"ineligible_role"`, anchor never run (features ARE populated: the
         pass ran);
      4. ONLY survivors of BOTH cheap filters get the 400-sim fpu-off anchor
         (`search_with_root`, `MCTSConfig(fpu_policy_mass_reduction=None)`,
         `add_noise=False`) -> `anchor_run=True`, then `anchor_eligible(
         root_value_stm)` decides `"kept"` vs `"ineligible_anchor"`.
    A `"kept"` proposal's hash is added to the run's own kept-hash set
    immediately, so a LATER proposal that collides with an EARLIER kept one
    (not just with the caller's `forbidden` union) is itself excluded as a
    collision -- v1's own `_scan_two_stage` `kept_hashes` idiom.

    BEFORE any evaluator/checkpoint load, runs the PURE geometric preflight
    (`v2_preflight_source`, Task 4) over the SAME reservoir index and raises
    `ValueError` naming the binding constraint if it is infeasible (design
    Sec 1.7: "gates before any evaluator loads" -- v1's own `main()` "stop-
    don't-retune" gate, mirrored here rather than in `main` so `run_screen`
    is correct even when called directly, not only via the CLI).

    OPERATOR: loads a real checkpoint via `config.checkpoint` and runs
    400-sim MCTS. Never invoked by this task's tests -- exercised only via
    the pure `classify_exclusion` / `screen_row` / `load_v2_config` and
    static wiring checks (signature/source-text/`find_spec` inspection).

    Returns `(rows, meta)` and, as a side effect, writes `config.screen_out`
    + its `.meta.json` (`write_screen_csv` / `write_screen_meta` -- mirrors
    v1's `write_manifest` / `write_meta`).
    """
    records = load_game_index(config.source_index_path)
    report = v2_preflight_source(records)
    if not report.feasible:
        raise ValueError(
            f"run_screen: v2 geometric preflight INFEASIBLE -- binding "
            f"constraint: {report.binding_constraint}. Stopping BEFORE "
            f"evaluator load (design Sec 1.7 stop-don't-retune).\n"
            f"{report.format()}")

    from .build_teacher_calibration_manifest import _teacher_infer

    forbidden = load_forbidden_hashes(config.forbidden_manifests)
    evaluator, search_fn, anchor_cfg = _build_v2_anchor_search_fn(
        config.checkpoint, config.eval_batch_size, config.stall_flush_sims)

    kept_hashes: Set[str] = set()
    rows: List[dict] = []
    status_counts: Counter = Counter()

    for rec in records:
        replay = json.loads(Path(rec["replay_path"]).read_text())
        replay = {**replay, "game_idx": rec["game_idx"]}
        for proposal in enumerate_v2_proposals(replay):
            ply, side = proposal["ply"], proposal["side"]
            state = position_state(replay, ply, side)
            sha1 = canonical_state_sha1(state)
            collided = sha1 in forbidden or sha1 in kept_hashes

            feats = role = root_value_stm = anchor_elig = None
            if not collided:
                _legal, priors, _raw_value = _teacher_infer(state, evaluator)
                feats = _policy_features_from_priors(priors)
                role = raw_policy_role(feats["normalized_entropy"], feats["top1_prior"])
                if role is not None:
                    _counts, root_value_stm, root = search_fn(
                        state, _v2_anchor_seed(rec["game_idx"], ply))
                    if root.visit_count != ANCHOR_SIMS_V2:
                        raise RuntimeError(
                            f"v2 anchor confirm game_idx={rec['game_idx']} "
                            f"ply={ply}: {root.visit_count} sims != "
                            f"{ANCHOR_SIMS_V2}")
                    anchor_elig = anchor_eligible(root_value_stm)

            exclusion_status, anchor_run = classify_exclusion(
                collided=collided, role=role, anchor_eligible_val=anchor_elig)
            row = screen_row(
                proposal, feats=feats, role=role, anchor_run=anchor_run,
                root_value_stm=root_value_stm, anchor_eligible=anchor_elig,
                canonical_sha1=sha1, exclusion_status=exclusion_status)
            rows.append(row)
            status_counts[exclusion_status] += 1
            if exclusion_status == "kept":
                kept_hashes.add(sha1)
        # No early stop (design Sec 1.6): every game's every proposal is
        # screened and persisted, however "full" any cell already looks --
        # selection is entirely deferred to the separate `select` stage.

    write_screen_csv(rows, config.screen_out)
    meta = {
        "config_path": config.config_path,
        "source_index_path": config.source_index_path,
        "checkpoint": config.checkpoint,
        "forbidden_manifests": list(config.forbidden_manifests),
        "n_forbidden_hashes": len(forbidden),
        "n_games": len(records),
        "n_proposals": len(rows),
        "status_counts": dict(status_counts),
        "fieldnames": SCREEN_FIELDNAMES,
        "base_mcts_config": dataclasses.asdict(anchor_cfg),
    }
    write_screen_meta(config.screen_out, meta)
    return rows, meta


# ---------------------------------------------------------------------------
# Screen artifact persistence (mirrors v1's `write_manifest` / `corpus_
# provenance` / `write_meta`, build_fpu_dev_corpus.py:1111-1154)
# ---------------------------------------------------------------------------

# The v2 screen's own effective result-determining source files (design Sec
# 1.8), mirroring v1's `_CORPUS_SOURCES` (build_fpu_dev_corpus.py:64-71):
# this module itself (the enumerator/screen logic), v1's module (v2 imports
# its pure helpers from it), the teacher-inference module (source of
# `_teacher_infer`), the MCTS engine, the hash + state-reconstruction deps.
_V2_MODULE_DIR = Path(__file__).resolve().parent
_V2_CORPUS_SOURCES: Tuple[Path, ...] = (
    _V2_MODULE_DIR / "fpu_dev_corpus_v2.py",
    _V2_MODULE_DIR / "build_fpu_dev_corpus.py",
    _V2_MODULE_DIR / "build_teacher_calibration_manifest.py",
    _V2_MODULE_DIR / "mcts.py",
    _V2_MODULE_DIR / "fpu_state_hash.py",
    _V2_MODULE_DIR / "goal_line_trigger_probe_cases.py",
    _V2_MODULE_DIR / "game" / "twixt_state.py",
)


def write_screen_csv(rows: List[dict], out_csv: str) -> None:
    """Write the screen artifact CSV (mirrors v1's `write_manifest`). EVERY
    proposal's row -- kept, excluded and ineligible alike -- lands here. A
    tuple-valued `proposal_cell` and the nullable anchor/policy columns are
    written via `csv.DictWriter`'s ordinary `str()` / empty-string coercion
    (`None` -> an empty field), exactly like any other CSV field -- no special
    casing needed here.
    """
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCREEN_FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


def v2_screen_provenance(*, config_path: Optional[str], source_index_path: Optional[str],
                         checkpoint: Optional[str], forbidden_manifests: Iterable[str],
                         base_mcts_config: Optional[dict]) -> dict:
    """Evidence-grade provenance for the v2 screen artifact (design Sec 1.8):
    the config-file hash + the SAME fingerprint shape as v1's
    `corpus_provenance` (build_fpu_dev_corpus.py:1119) -- source-file BYTE
    hashes, the source-index sha1 + a deterministic replay-DATA hash, the
    checkpoint identity, the forbidden manifests' OWN hashes (so `select` can
    hard-match that they are the SAME files `screen` excluded against, not
    merely the same paths), and runtime identity. Pure / stdlib-only (via
    `fpu_provenance`): reads the source index + replays + config + checkpoint
    BYTES but touches NO MCTS/GPU/MLX. `replay_paths` come from the SAME
    `load_game_index` `run_screen` scans, so the hash covers exactly the
    games the screen was built from.
    """
    replay_paths = ([r["replay_path"] for r in load_game_index(source_index_path)]
                    if source_index_path else [])
    return {
        "config_sha1": fpu_provenance.file_sha1(config_path),
        "source_file_sha1s": fpu_provenance.source_file_sha1s(_V2_CORPUS_SOURCES),
        "source_index_sha1": fpu_provenance.file_sha1(source_index_path),
        "replay_data_sha1": fpu_provenance.replay_data_sha1(replay_paths),
        "checkpoint_identity": (
            f"{Path(checkpoint).name}:{fpu_provenance.file_sha1(checkpoint)}"
            if checkpoint else "none"),
        "forbidden_manifest_sha1s": fpu_provenance.source_file_sha1s(
            forbidden_manifests),
        "base_mcts_config": base_mcts_config,
        "add_noise": False,
        "runtime_provenance": fpu_provenance.runtime_provenance(),
    }


def write_screen_meta(out_csv: str, meta: dict) -> None:
    """Write `<out_csv>.meta.json`, ENRICHED with the evidence-grade
    `provenance` block (design Sec 1.8) computed from the meta's own
    `config_path` / `source_index_path` / `checkpoint` / `forbidden_
    manifests` / `base_mcts_config` -- mirrors v1's `write_meta`
    (build_fpu_dev_corpus.py:1143). `provenance` is DERIVED, so it never
    clobbers a caller key.
    """
    enriched = dict(meta)
    enriched["provenance"] = v2_screen_provenance(
        config_path=meta.get("config_path"),
        source_index_path=meta.get("source_index_path"),
        checkpoint=meta.get("checkpoint"),
        forbidden_manifests=meta.get("forbidden_manifests") or [],
        base_mcts_config=meta.get("base_mcts_config"))
    Path(str(out_csv) + ".meta.json").write_text(json.dumps(enriched, indent=2))


# ---------------------------------------------------------------------------
# CLI (design Sec 1.8: `--config` is required, no default; `--mode` is
# restricted to `("screen",)` -- Task 6 widens it to include `"select"`.
# `screen` and `select` are NEVER the same invocation, so there is no
# `elif args.mode == "select":` branch here yet -- shipping one now, ahead of
# Task 6's pure `select_final_manifest`, would be a broken/dead path.)
# ---------------------------------------------------------------------------

def _parse_v2_args(argv):
    ap = argparse.ArgumentParser(
        description="v2 phase-primary FPU dev-corpus pipeline (design Sec "
                    "1.6). `screen` (operator; evaluator+MCTS) screens EVERY "
                    "proposal from the reservoir against the cheap "
                    "collision/raw-policy filters before the 400-sim fpu-off "
                    "anchor, persisting every outcome -- never stopping "
                    "early. `select` (pure, no evaluator) is added by Task "
                    "6; screen and select are never the same invocation.")
    ap.add_argument("--mode", required=True, choices=("screen",),
                    help='pipeline stage to run. Only "screen" exists so '
                         'far -- Task 6 adds "select".')
    ap.add_argument("--config", required=True,
                    help="path to the required fpu_dev_corpus_v2_config.json "
                         "(design Sec 1.8) -- no default source, no default "
                         "stride.")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_v2_args(argv)
    config = load_v2_config(args.config)

    if args.mode == "screen":
        try:
            rows, meta = run_screen(config)
        except ValueError as exc:
            print(f"[fpu-dev-corpus-v2] screen FAILED: {exc}")
            return 2
        print(f"[fpu-dev-corpus-v2] screen: wrote {len(rows)} proposal "
              f"row(s) -> {config.screen_out} (+ .meta.json); "
              f"status_counts={meta['status_counts']}")
        return 0

    raise AssertionError(   # unreachable: argparse's own `choices` guards this
        f"main: unreachable --mode {args.mode!r}")


if __name__ == "__main__":
    raise SystemExit(main())
