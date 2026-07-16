"""Regression test proving the v1 branching-band + <=50%-ply-bucket-cap
corpus design is mathematically IMPOSSIBLE on the project's fixed 24x24
board (Task 0), plus tests for the v2 phase-primary constants + the
`proposal_cell_of` classifier in `fpu_dev_corpus_v2.py` (Task 1).

Frozen design ref: docs/superpowers/specs/2026-07-12-fpu-dev-corpus-v2-phase-design.md
  section 0 ("Why v2 (the impossibility that retired v1)"), section 1.2
  (SPLIT_ALLOC_V2), section 1.3 (late coverage floors), section 1.5
  (proposal enumerator / proposal cells).
v2 plan Task 0 ("Impossibility regression test (the WHY, and a guard)") and
Task 1 ("v2 constants + classifiers").

This is the WHY behind retiring `build_fpu_dev_corpus.py`'s v1 branching-band
(n_legal 200-299 / 300-399 / 400+) stratification in favor of the v2
phase-primary design, encoded as a permanent regression test so nobody
resurrects the v1 bands.

The argument (already verified against 800 real games -- asserted here as
pure arithmetic/geometry, NOT re-derived from replay data):
  1. On a 24x24 board each side's legal region is exactly 528 cells on the
     empty board, and each placed peg removes AT MOST one legal cell (for
     either side), so for every reachable position: n_legal >= 528 - ply.
  2. Therefore n_legal <= 399 implies ply >= 528 - 399 = 129.
  3. 129 >= 91, so every position in the two low bands (b200_299, b300_399)
     falls in the "late" ply_bucket_of bucket (ply_bucket_of returns "late"
     for ply >= 91) -- checked here across a range of plies, not just the
     boundary value.
  4. v1's frozen composition needs QUOTA_PER_BAND (80) rows in EACH of the
     two low bands = 160 "late" rows. The <=50% ply-bucket cap on the
     CORPUS_SIZE-row (240) corpus permits at most int(0.5 * 240) = 120.
     160 > 120 => impossible, independent of seed/length/enumeration/FPU.

Pure arithmetic/geometry: only stdlib + the pure v1 helpers/constants +
TwixtState. No evaluator, no MCTS, no GPU/MLX, no checkpoint, no I/O.

Task 1's tests below exercise `fpu_dev_corpus_v2.py`'s frozen phase-primary
constants (PHASES, SPLIT_ALLOC_V2, LATE_TARGET_FLOORS, PROPOSAL_CELLS, the
v2-specific MAX_PER_CELL_PER_GAME, the derived CORPUS_SIZE, and the shared
v1 MIN_PLY_GAP/MAX_PER_GAME/SIDE_TOL re-exports) and its one classifier,
`proposal_cell_of` -- still pure stdlib, still no evaluator/MCTS/GPU/MLX.

Task 3's tests (bottom of the file) exercise the phase-stratified sampler
`sample_v2_rows` + `assign_split_v2` over FABRICATED `kept` screen rows
(plain dicts; synthetic canonical_sha1 strings, never real hashes) -- the
same pure, evaluator-free shape as v1's tests/test_fpu_dev_corpus.py.
"""
import copy
import csv
import dataclasses
import inspect
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from scripts.GPU.alphazero.build_fpu_dev_corpus import (
    CORPUS_SIZE,
    QUOTA_PER_BAND,
    SPLIT_ALLOC,
    _CORPUS_SOURCES,
    anchor_eligible,
    band_of,
    load_forbidden_hashes,
    ply_bucket_of,
    side_to_move_for_ply,
)
from scripts.GPU.alphazero.fpu_dev_corpus_v2 import (
    ANCHOR_SEED_BASE_V2,
    ANCHOR_SIMS_V2,
    ASSIGN_ATTEMPTS,
    CELL_ORDER_V2,
    CORPUS_SIZE as CORPUS_SIZE_V2,
    LATE_PHASE,
    LATE_TARGET_CELL,
    LATE_TARGET_FLOORS,
    MANIFEST_FIELDNAMES_V2,
    MAX_PER_CELL_PER_GAME,
    MAX_PER_GAME,
    MIN_PLY_GAP,
    PAIR_POSITIONS,
    PHASES,
    PREREGISTERED_IDENTITY_KEYS,
    PROPOSAL_CELLS,
    QUOTA_PER_PHASE,
    SCREEN_ARTIFACT_IDENTITY,
    SCREEN_FIELDNAMES,
    SCREEN_IDENTITY_KEYS,
    SCREEN_ROW_KEY_FIELDS,
    SELF_REFERENTIAL_IDENTITY,
    SIDE_TOL,
    SPLIT_ALLOC_V2,
    SPLIT_TOTALS,
    SPLITS,
    UNPREREGISTERABLE_IDENTITIES,
    V2Config,
    V2PreflightInfeasible,
    V2PreflightReport,
    _V2_CONFIG_REQUIRED_KEYS,
    _V2_CORPUS_SOURCES,
    _build_v2_anchor_search_fn,
    _parse_v2_args,
    _v2_anchor_seed,
    assign_split_v2,
    classify_exclusion,
    enumerate_v2_proposals,
    kept_rows_from_screen,
    load_v2_config,
    main,
    post_screen_qualification,
    proposal_cell_of,
    read_screen_csv,
    run_screen,
    sample_v2_rows,
    screen_row,
    screen_row_counts,
    select_final_manifest,
    v2_geometry_feasibility,
    v2_preflight_source,
    v2_screen_provenance,
    validate_screen_identities,
    validate_screen_rows_against_meta,
    write_screen_csv,
    write_screen_meta,
    write_select_csv,
)
from scripts.GPU.alphazero.fpu_dev_reservoir_protocol import (
    ReservoirMeasurements,
    canonical_json_bytes,
    derive_config,
    measure_reservoir,
)
from scripts.GPU.alphazero.game.twixt_state import TwixtState
from scripts.GPU.alphazero import fpu_provenance


def test_empty_board_legal_moves_is_528_both_sides():
    """Anchor fact: the empty-board legal region is exactly 528 cells for
    BOTH sides on the project's fixed 24x24 board -- the base case of the
    `n_legal >= 528 - ply` invariant (ply == 0)."""
    red = TwixtState(board_size=24, active_size=24, to_move="red")
    black = TwixtState(board_size=24, active_size=24, to_move="black")
    assert len(red.legal_moves()) == 528
    assert len(black.legal_moves()) == 528
    assert red.ply == 0 and black.ply == 0


def test_low_n_legal_forces_late_ply_bucket():
    """n_legal <= 399 implies ply >= 129 (from the `n_legal >= 528 - ply`
    invariant), and every ply from that floor across a wide sampled range
    lands in the REAL ply_bucket_of's "late" bucket -- not merely the single
    boundary value."""
    ply_floor = 528 - 399
    assert ply_floor == 129
    assert ply_floor >= 91   # ply_bucket_of's "late" threshold

    for ply in range(ply_floor, ply_floor + 100):
        assert ply_bucket_of(ply) == "late"

    # Sanity: the bucket boundary itself is exactly where "late" begins, so
    # the low bands' floor (129) is comfortably inside it, not adjacent.
    assert ply_bucket_of(91) == "late"
    assert ply_bucket_of(90) != "late"


def test_v1_bands_impossible_on_24board():
    """The headline regression: v1's two low branching bands (b200_299,
    b300_399) together demand 160 "late" rows, but the <=50% ply-bucket cap
    on the real 240-row corpus permits at most 120 -- so the v1
    branching-band + 50%-cap design is IMPOSSIBLE on this board, regardless
    of source corpus, replay length, enumeration stride, or FPU config.

    Derived from the real v1 constants (SPLIT_ALLOC / QUOTA_PER_BAND /
    CORPUS_SIZE) rather than hard-coded, so this stays honest if those
    constants ever move -- then cross-checked against the brief's explicit
    pinned values for a readable, unambiguous assertion.
    """
    # --- Step 1: the ply-floor half of the argument (pinned form). ---
    assert 528 - 399 == 129 >= 91

    # --- Step 2: the two low bands' combined row requirement, derived
    # directly from the real frozen SPLIT_ALLOC (sum tuning + frozen_check
    # across BOTH roles for exactly the b200_299 / b300_399 cells). ---
    low_bands = {"b200_299", "b300_399"}
    low_band_rows = sum(
        alloc["tuning"] + alloc["frozen_check"]
        for (role, band), alloc in SPLIT_ALLOC.items()
        if band in low_bands
    )
    # QUOTA_PER_BAND is itself derived (uniform per-band total) from
    # SPLIT_ALLOC in build_fpu_dev_corpus.py, so this is an independent
    # cross-check, not a tautology.
    assert low_band_rows == QUOTA_PER_BAND * 2
    assert low_band_rows == 160

    # --- Step 3: the <=50% ply-bucket cap on the real corpus size. ---
    bucket_cap = int(0.5 * CORPUS_SIZE)
    assert bucket_cap == 120
    assert bucket_cap == int(0.5 * 240)

    # --- Step 4: the pinned form from the brief (explicit + readable). ---
    assert 80 + 80 > int(0.5 * 240)

    # --- The impossibility itself. ---
    assert low_band_rows > bucket_cap


# ---------------------------------------------------------------------------
# Task 1 -- v2 phase-primary constants + proposal_cell_of classifier
#
# Frozen design ref: docs/superpowers/specs/2026-07-12-fpu-dev-corpus-v2-phase-design.md
#   Sec 1.2 (phase-stratified SPLIT_ALLOC_V2), Sec 1.3 (late coverage
#   floors), Sec 1.5 (proposal enumerator / proposal cells).
# v2 plan Task 1 ("v2 constants + classifiers"). Exercises
# scripts/GPU/alphazero/fpu_dev_corpus_v2.py's pure constants + its one
# classifier -- still stdlib-only, no MCTS/GPU/MLX/evaluator/I-O.
# ---------------------------------------------------------------------------

def test_phases_matches_real_ply_bucket_of_vocabulary():
    """PHASES must be the v1 ply_bucket_of bucket vocabulary, in the same
    opening/early_mid/midgame/late order -- checked against the REAL
    imported ply_bucket_of (not a hand-typed duplicate that could silently
    drift from v1's naming) at every v1 bucket boundary (mirrors
    test_ply_bucket_boundaries in tests/test_fpu_dev_corpus.py)."""
    assert PHASES == ("opening", "early_mid", "midgame", "late")
    assert ply_bucket_of(1) == PHASES[0]
    assert ply_bucket_of(15) == PHASES[0]
    assert ply_bucket_of(16) == PHASES[1]
    assert ply_bucket_of(40) == PHASES[1]
    assert ply_bucket_of(41) == PHASES[2]
    assert ply_bucket_of(90) == PHASES[2]
    assert ply_bucket_of(91) == PHASES[3]
    assert ply_bucket_of(300) == PHASES[3]


def test_shared_v1_constants_pinned():
    """MIN_PLY_GAP / MAX_PER_GAME / SIDE_TOL are the IDENTICAL v1 frozen
    constants (build_fpu_dev_corpus.py) -- imported+re-exported by
    fpu_dev_corpus_v2 rather than restated, so pin their frozen values here
    too: a v1 drift then fails loudly in the v2 module as well."""
    assert MIN_PLY_GAP == 12
    assert MAX_PER_GAME == 2
    assert SIDE_TOL == 2


def test_max_per_cell_per_game_is_a_distinct_v2_constant():
    """MAX_PER_CELL_PER_GAME (Task-2 enumerator: caps PROPOSALS per game per
    proposal-cell) and MAX_PER_GAME (Task-3 sampler: caps SELECTED rows per
    game GLOBALLY across all cells) are separate frozen constants that
    happen to share the value 2 while gating different pipeline stages --
    both must exist independently under their own names."""
    assert MAX_PER_CELL_PER_GAME == 2
    assert MAX_PER_GAME == 2


def test_split_alloc_v2_totals_derived():
    """SPLIT_ALLOC_V2 sums to the frozen composition -- each total is
    RE-DERIVED directly from the dict (not trusting the module's own
    arithmetic), then cross-checked against the brief's pinned literals
    (240 grand / 180 target-60 control / 160 tuning-80 frozen_check)."""
    target_total = sum(
        alloc["tuning"] + alloc["frozen_check"]
        for (role, _phase), alloc in SPLIT_ALLOC_V2.items()
        if role == "target"
    )
    control_total = sum(
        alloc["tuning"] + alloc["frozen_check"]
        for (role, _phase), alloc in SPLIT_ALLOC_V2.items()
        if role == "control"
    )
    tuning_total = sum(alloc["tuning"] for alloc in SPLIT_ALLOC_V2.values())
    frozen_total = sum(alloc["frozen_check"] for alloc in SPLIT_ALLOC_V2.values())
    grand_total = sum(
        alloc["tuning"] + alloc["frozen_check"] for alloc in SPLIT_ALLOC_V2.values()
    )

    assert target_total == 180
    assert control_total == 60
    assert tuning_total == 160
    assert frozen_total == 80
    assert grand_total == 240

    # Every phase contributes IDENTICALLY (45 target / 15 control) -- 45/15
    # both divide evenly, unlike v1's odd-quota control bands (13/7 vs
    # 14/6), so there is no per-phase asymmetry to special-case.
    assert set(SPLIT_ALLOC_V2.keys()) == {
        (role, phase) for role in ("target", "control") for phase in PHASES
    }
    for phase in PHASES:
        assert SPLIT_ALLOC_V2[("target", phase)] == {"tuning": 30, "frozen_check": 15}
        assert SPLIT_ALLOC_V2[("control", phase)] == {"tuning": 10, "frozen_check": 5}


def test_corpus_size_v2_is_derived_not_hardcoded():
    """CORPUS_SIZE (v2) must be DERIVED from SPLIT_ALLOC_V2 -- mirroring how
    build_fpu_dev_corpus.py derives its own CORPUS_SIZE
    (build_fpu_dev_corpus.py:115) rather than hard-coding 240. Recomputed
    independently here (not just re-reading the module's own arithmetic)
    and cross-checked against the pinned literal."""
    recomputed = sum(
        alloc["tuning"] + alloc["frozen_check"] for alloc in SPLIT_ALLOC_V2.values()
    )
    assert CORPUS_SIZE_V2 == recomputed
    assert CORPUS_SIZE_V2 == 240


def test_late_target_floors_values():
    assert LATE_TARGET_FLOORS == {"b300_399": 12, "b200_299": 12}


def test_proposal_cells_exact_six_in_load_bearing_order():
    """PROPOSAL_CELLS order is deterministic and load-bearing for later
    tasks (Task 2's enumerator iterates cells in this order): the three
    non-late phases (band None) in PHASES order, then the three late bands
    in DESCENDING branching order (b400_plus, b300_399, b200_299) -- per
    the v2 controller resolution, NOT v1's BANDS ascending order."""
    assert PROPOSAL_CELLS == [
        ("opening", None),
        ("early_mid", None),
        ("midgame", None),
        ("late", "b400_plus"),
        ("late", "b300_399"),
        ("late", "b200_299"),
    ]
    assert len(PROPOSAL_CELLS) == 6
    assert len(set(PROPOSAL_CELLS)) == 6   # no duplicate cells


def test_proposal_cell_of_non_late_ignores_n_legal():
    """Non-late phases collapse to (phase, None) regardless of n_legal --
    the brief's exact example plus extra phases/extreme n_legal values."""
    assert proposal_cell_of("opening", 520) == ("opening", None)
    assert proposal_cell_of("early_mid", 10) == ("early_mid", None)
    assert proposal_cell_of("midgame", 528) == ("midgame", None)


def test_proposal_cell_of_late_splits_by_band():
    """The brief's exact three examples, plus the sub-200 ineligibility
    sentinel (controller resolution #1)."""
    assert proposal_cell_of("late", 520) == ("late", "b400_plus")
    assert proposal_cell_of("late", 350) == ("late", "b300_399")
    assert proposal_cell_of("late", 250) == ("late", "b200_299")
    # Sub-200 late position: intentionally NOT a PROPOSAL_CELLS member --
    # this is how sub-200 positions become ineligible (Task 2's enumerator
    # additionally enforces n_legal >= 200 for every cell; this sentinel is
    # belt-and-suspenders, not the only guard).
    assert proposal_cell_of("late", 150) == ("late", None)
    assert ("late", None) not in PROPOSAL_CELLS


def test_proposal_cell_of_late_matches_real_band_of():
    """Reuse check: proposal_cell_of's band element must equal the REAL
    imported band_of(n_legal) -- not a reimplemented/hand-typed copy of its
    boundaries -- checked across every band_of boundary, including the
    sub-200 None case."""
    for n_legal in (1000, 400, 399, 300, 299, 200, 199, 0):
        assert proposal_cell_of("late", n_legal) == ("late", band_of(n_legal))


def test_proposal_cell_of_outputs_land_in_proposal_cells():
    """Every phase/n_legal combination that SHOULD be eligible maps into an
    actual PROPOSAL_CELLS member -- ties the classifier and the cell list
    together so they can't silently drift apart."""
    for phase in ("opening", "early_mid", "midgame"):
        assert proposal_cell_of(phase, 999) in PROPOSAL_CELLS
        assert proposal_cell_of(phase, 1) in PROPOSAL_CELLS
    for n_legal, band in ((520, "b400_plus"), (350, "b300_399"), (250, "b200_299")):
        cell = proposal_cell_of("late", n_legal)
        assert cell in PROPOSAL_CELLS
        assert cell == ("late", band)


# ---------------------------------------------------------------------------
# Task 2 -- enumerate_v2_proposals (phase-aware side-opposed proposal
# enumerator, no global stride)
#
# Frozen design ref: docs/superpowers/specs/2026-07-12-fpu-dev-corpus-v2-phase-design.md
#   Sec 1.5 (proposal enumerator / proposal cells).
# v2 plan Task 2. Exercises scripts/GPU/alphazero/fpu_dev_corpus_v2.py's
# `enumerate_v2_proposals` -- still stdlib-only synthetic replays (a dict
# with `game_idx` + `moves`, each move carrying `n_legal`), no
# evaluator/MCTS/GPU/MLX/I-O; `per_ply_n_legal`'s reconstruction fallback
# (real TwixtState replay) is v1's own concern and already covered by v1's
# tests, not re-exercised here.
#
# Fixtures use the board's own physical lower bound, `n_legal >= 528 - ply`
# (Task 0's regression), so every fixture's n_legal values are REACHABLE at
# the plies they claim -- most via `_honest_replay`'s default TIGHT schedule
# `n_legal(ply) = 528 - ply`, which conveniently makes each phase/band's
# eligible-ply range exactly the contiguous span the design's own geometry
# predicts (matching the canonical example pairs already used elsewhere in
# this suite, e.g. tests/test_fpu_corpus_preflight.py's `_BUCKET_PAIR`).
# One test (`test_pair_output_order_is_ascending_ply_not_side_order`)
# overrides specific plies to a HIGHER (still honest -- the invariant is a
# LOWER bound only) band on purpose, to construct a cell whose selected pair
# is black-before-red; another (sub-200 guard) is explicitly and only
# unrealistic on purpose -- see its own docstring.
# ---------------------------------------------------------------------------

def _honest_replay(game_idx, n_moves, overrides=None):
    """Synthetic replay `{"game_idx": ..., "moves": [{"n_legal": ...}, ...]}`
    covering plies `0..n_moves-1`. Defaults every ply to the TIGHT physical
    floor `528 - ply` (Task 0's `n_legal >= 528 - ply` invariant, held as an
    equality) -- monotonically decreasing, so every phase/band region is
    reachable exactly where the design geometry predicts. `overrides` (ply ->
    n_legal) replaces individual plies; every caller-supplied override in
    this file still respects the >= 528 - ply LOWER bound (never below it),
    so all fixtures stay physically honest even when overridden.
    """
    overrides = overrides or {}
    moves = [{"n_legal": overrides.get(ply, 528 - ply)} for ply in range(n_moves)]
    return {"game_idx": game_idx, "moves": moves}


def test_cell_yields_side_opposed_pair_at_least_min_gap_apart():
    """Brief case 1: a cell yields a red+black pair >=12 apart -- the two
    sides differ and |ply gap| >= MIN_PLY_GAP. Opening (ply 0-15, all
    n_legal>=513 hence eligible): the earliest-satisfying pair is (0, 13)."""
    replay = _honest_replay(game_idx=1, n_moves=16)
    proposals = enumerate_v2_proposals(replay)
    opening_rows = [p for p in proposals if p["proposal_cell"] == ("opening", None)]
    assert len(opening_rows) == 2
    sides = {row["side"] for row in opening_rows}
    assert sides == {"red", "black"}
    plies = [row["ply"] for row in opening_rows]
    assert abs(plies[0] - plies[1]) >= MIN_PLY_GAP
    assert sorted(plies) == [0, 13]


def test_cell_caps_at_max_per_cell_per_game_despite_abundant_candidates():
    """Brief case 2: a cell with MANY eligible plies still yields <=2.
    Midgame (ply 41-90, 50 plies, all honestly eligible) has far more than 2
    valid opposed plies on each side, yet the cell still yields exactly
    MAX_PER_CELL_PER_GAME (2) proposals -- one pair, not every possible
    pair."""
    replay = _honest_replay(game_idx=2, n_moves=91)
    proposals = enumerate_v2_proposals(replay)
    midgame_rows = [p for p in proposals if p["proposal_cell"] == ("midgame", None)]
    assert len(midgame_rows) == MAX_PER_CELL_PER_GAME
    assert len(midgame_rows) <= 2


def test_cell_with_no_valid_gap_pair_yields_zero():
    """Brief case 3: a cell with no valid opposed-gap pair yields 0 for that
    cell -- never a lone unpaired proposal. A 9-ply game (ply 0-8, all
    "opening") has BOTH reds ([0,2,4,6,8]) and blacks ([1,3,5,7]) present,
    but the widest possible span is only 8 (< MIN_PLY_GAP), so no pair
    qualifies and the whole game yields nothing (every other cell has zero
    candidates at all, trivially)."""
    replay = _honest_replay(game_idx=20, n_moves=9)
    assert enumerate_v2_proposals(replay) == []


def test_late_cell_only_selects_matching_band():
    """Brief case 4: late cells only select plies whose band_of(n_legal)
    matches the cell's band. late x b300_399 is reachable only at ply>=129
    (Task 0's floor); under the tight honest schedule this cell's pair is
    (130, 143), both genuinely in the 300-399 range -- never a b400_plus or
    b200_299 ply leaking in from the same "late" phase."""
    replay = _honest_replay(game_idx=3, n_moves=230)
    proposals = enumerate_v2_proposals(replay)
    cell_rows = [p for p in proposals if p["proposal_cell"] == ("late", "b300_399")]
    assert len(cell_rows) == 2
    for row in cell_rows:
        assert row["ply"] >= 129
        assert band_of(row["n_legal"]) == "b300_399"
    assert {row["side"] for row in cell_rows} == {"red", "black"}
    plies = [row["ply"] for row in cell_rows]
    assert abs(plies[0] - plies[1]) >= MIN_PLY_GAP
    assert sorted(plies) == [130, 143]


def test_determinism_same_replay_same_proposals():
    """Brief case 5: same replay -> same proposals, called twice."""
    replay = _honest_replay(game_idx=42, n_moves=330)
    first = enumerate_v2_proposals(replay)
    second = enumerate_v2_proposals(replay)
    assert first == second


def test_full_replay_yields_up_to_twelve_proposals_in_cell_then_ply_order():
    """A single game can yield up to 12 proposals (2 x 6 cells) -- NOT
    capped by the global <=2/game rule, which is a Task-3 SAMPLER concern
    over SELECTED rows, never enforced by the enumerator (brief note). A
    330-ply honest game reaches every one of the 6 PROPOSAL_CELLS (opening,
    early_mid, midgame, late x b400_plus, late x b300_399, late x b200_299).
    Output order is deterministic: cells in PROPOSAL_CELLS order, each cell's
    pair in ascending ply."""
    replay = _honest_replay(game_idx=42, n_moves=330)
    proposals = enumerate_v2_proposals(replay)

    assert len(proposals) == 12
    got_cells_in_order = [proposals[i]["proposal_cell"] for i in range(0, 12, 2)]
    assert got_cells_in_order == PROPOSAL_CELLS
    for i in range(0, 12, 2):
        pair = proposals[i:i + 2]
        assert pair[0]["proposal_cell"] == pair[1]["proposal_cell"]
        assert pair[0]["ply"] < pair[1]["ply"]   # ascending ply within the cell

    expected_plies = [(0, 13), (16, 29), (42, 55), (92, 105), (130, 143), (230, 243)]
    got_plies = [(proposals[i]["ply"], proposals[i + 1]["ply"]) for i in range(0, 12, 2)]
    assert got_plies == expected_plies

    for row in proposals:
        assert row["game_idx"] == 42
        assert row["phase"] == row["proposal_cell"][0]
        assert row["phase"] == ply_bucket_of(row["ply"])
        assert row["side"] in ("red", "black")


def test_proposal_dict_has_exact_schema():
    """Interface: each proposal dict carries EXACTLY game_idx, ply, side,
    phase, n_legal, band, proposal_cell -- no more, no fewer."""
    replay = _honest_replay(game_idx=5, n_moves=16)
    proposals = enumerate_v2_proposals(replay)
    expected_keys = {"game_idx", "ply", "side", "phase", "n_legal", "band", "proposal_cell"}
    assert proposals   # sanity: this fixture is non-empty
    for row in proposals:
        assert set(row.keys()) == expected_keys
        assert row["game_idx"] == 5


def test_band_field_is_band_of_n_legal_not_cell_band_component():
    """Interface note: `band` is ALWAYS the real `band_of(n_legal)`, never
    the cell's own band component -- an `("opening", None)` cell's proposals
    still record their real band (b400_plus: opening n_legal>=513 on this
    board), not None."""
    replay = _honest_replay(game_idx=6, n_moves=16)
    proposals = enumerate_v2_proposals(replay)
    opening_rows = [p for p in proposals if p["proposal_cell"] == ("opening", None)]
    assert len(opening_rows) == 2
    for row in opening_rows:
        assert row["band"] == band_of(row["n_legal"])
        assert row["band"] == "b400_plus"
        assert row["proposal_cell"][1] is None   # the cell's OWN band component


def test_low_n_legal_excludes_ply_even_when_phase_matches():
    """n_legal >= 200 is an independent eligibility guard, not solely reliant
    on proposal_cell_of's own ("late", None) sub-200 sentinel
    ("belt-and-suspenders", per that function's docstring) -- checked
    directly with an otherwise phase-matching low value. NOTE: unlike this
    file's other fixtures, this one is deliberately NOT physically honest (an
    opening ply can never really have n_legal < 200 on this board -- Task
    0's floor already forces >=513 that early); it exists solely to prove
    the enumerator's OWN n_legal>=200 guard fires independently of
    proposal_cell_of, not to claim a reachable position."""
    moves = [{"n_legal": 150} for _ in range(16)]
    replay = {"game_idx": 10, "moves": moves}
    assert enumerate_v2_proposals(replay) == []


def test_pair_output_order_is_ascending_ply_not_side_order():
    """Interface: within a cell the pair is output in ASCENDING PLY, not
    "red always first". Constructed (still honestly -- the >=528-ply
    invariant is a LOWER bound, so pushing a ply's n_legal UP into a higher
    band is always honest) so late x b300_399's only eligible black is ply
    129 and its only eligible red is ply 142 (plies 130-141 pushed into
    b400_plus, hence ineligible for THIS cell) -- the pair is (black@129,
    red@142), so a naive "always emit red then black" implementation would
    emit them in the wrong order."""
    overrides = {129: 399, 142: 390}
    for ply in range(130, 142):
        overrides[ply] = 450   # honest (>= the ply's floor); b400_plus, not b300_399
    replay = _honest_replay(game_idx=7, n_moves=143, overrides=overrides)
    proposals = enumerate_v2_proposals(replay)
    cell_rows = [p for p in proposals if p["proposal_cell"] == ("late", "b300_399")]
    assert [p["ply"] for p in cell_rows] == [129, 142]
    assert [p["side"] for p in cell_rows] == ["black", "red"]


# ---------------------------------------------------------------------------
# Task 3 -- phase-stratified sampler (`sample_v2_rows`) + hard late floors
#
# Frozen design ref: docs/superpowers/specs/2026-07-12-fpu-dev-corpus-v2-phase-design.md
#   Sec 1.2 (SPLIT_ALLOC_V2), Sec 1.3 (late coverage floors).
# v2 plan Task 3. Operates on FABRICATED `kept` screen rows (the Task-5 screen
# stage produces the real ones): plain dicts carrying game_idx, role, phase,
# band, side, ply, canonical_sha1. Synthetic sha1 strings, no evaluator/MCTS/
# GPU/MLX/I-O -- exactly v1's tests/test_fpu_dev_corpus.py shape, re-cut for
# v2's (role, phase) cells, its GLOBAL <=MAX_PER_GAME rule, and the floors.
#
# Fixture geometry is PHYSICALLY HONEST on the 24x24 board (Task 0's
# `n_legal >= 528 - ply` floor + `side_to_move_for_ply`'s red-on-even parity):
#   * red rows sit on EVEN plies, black rows on ODD plies;
#   * b300_399 exists only at ply >= 528-399 = 129 and b200_299 only at
#     ply >= 528-299 = 229 -- BOTH "late", which is exactly why the floors live
#     in the late phase;
#   * every non-late position is necessarily b400_plus (ply <= 90 =>
#     n_legal >= 438), so all opening/early_mid/midgame rows carry that band as
#     their recorded covariate.
# ---------------------------------------------------------------------------

# (red_ply, black_ply) per phase: >= MIN_PLY_GAP apart, both inside the phase's
# real ply_bucket_of range, correct side parity.
_V2_PHASE_PLIES = {
    "opening": (2, 15),       # ply_bucket_of: 1-15
    "early_mid": (16, 29),    # 16-40
    "midgame": (42, 55),      # 41-90
}
# late-only, per BAND -- the floors' geometry (b400_plus anywhere late;
# b300_399 only at ply >= 129; b200_299 only at ply >= 229).
_V2_LATE_BAND_PLIES = {
    "b400_plus": (92, 105),
    "b300_399": (130, 143),
    "b200_299": (230, 243),
}


def _v2_row(game_idx, role, phase, band, side, ply):
    """One fabricated KEPT screen row (globally-unique synthetic sha1)."""
    return {
        "game_idx": game_idx, "role": role, "phase": phase, "band": band,
        "side": side, "ply": ply,
        "canonical_sha1": f"v2-{game_idx:05d}-{ply:04d}-{side}",
    }


def _v2_game(game_idx, role, phase, band):
    """One game = ONE side-opposed pair (red on an even ply + black on an odd
    ply, >= MIN_PLY_GAP apart) in ONE (role, phase) cell -- v1 `_game_rows`'
    shape, re-cut for v2's phase cells and honest band geometry."""
    p_red, p_black = (_V2_LATE_BAND_PLIES[band] if phase == "late"
                      else _V2_PHASE_PLIES[phase])
    return [
        _v2_row(game_idx, role, phase, band, "red", p_red),
        _v2_row(game_idx, role, phase, band, "black", p_black),
    ]


# Per-cell game supply. 240 rows at <=2 rows/game needs >=120 games; the
# whole-game split assignment additionally needs ~23 games per TARGET cell (15
# to fill tuning's 30 + 8 to fill frozen_check's 15) and ~8 per CONTROL cell --
# so every cell here carries a real surplus over its own minimum.
#
# The (target, late) cell's band mix is the FLOOR DISCRIMINATOR: its 40
# b400_plus games (80 rows -- more than the whole 45-row late-target quota) come
# FIRST, i.e. at the LOWEST game_idx, and the round-robin visits candidate games
# in ascending game_idx. So an earliest-game fill with NO floor-satisfaction pass
# takes all 45 late-target rows from b400_plus and misses BOTH floors, even
# though a floor-satisfying selection plainly exists. 20 games (40 rows) per
# floor band is a surplus that survives the split assignment sending up to 8
# late-target games to frozen_check (>=12 floor-band games always remain in
# tuning), so the floors are reachable under ANY seed.
#
# The (control, late) cell deliberately carries floor-BAND rows too, floor-band
# games FIRST so they are actually SELECTED: the floors count ONLY late TARGET
# rows, so a sampler that credited a late CONTROL row to a floor counter would
# under-fill the real floor and the final floor verification would fire.
_V2_POOL_SPEC = {
    ("target", "opening"): [("b400_plus", 50)],
    ("control", "opening"): [("b400_plus", 30)],
    ("target", "early_mid"): [("b400_plus", 50)],
    ("control", "early_mid"): [("b400_plus", 30)],
    ("target", "midgame"): [("b400_plus", 50)],
    ("control", "midgame"): [("b400_plus", 30)],
    ("target", "late"): [("b400_plus", 40), ("b300_399", 20), ("b200_299", 20)],
    ("control", "late"): [("b300_399", 10), ("b200_299", 10), ("b400_plus", 10)],
}


def _v2_singleton_game(game_idx, role, phase, band, side):
    """One game contributing a SINGLE row to one cell -- because the screen
    classified its side-opposed partner into the OTHER role, or grey-dropped it
    (`raw_policy_role` reads each row independently). Routine on a real screen, and
    the ONLY way a game can be forced to shift a split's side balance."""
    p_red, p_black = (_V2_LATE_BAND_PLIES[band] if phase == "late"
                      else _V2_PHASE_PLIES[phase])
    return [_v2_row(game_idx, role, phase, band, side,
                    p_red if side == "red" else p_black)]


def _v2_pool(overrides=None, start_gi=0):
    """Build a fabricated `kept` pool from a {(role, phase): [(band, n_games)]}
    spec (defaulting to `_V2_POOL_SPEC`), game_idx ascending from `start_gi` in
    CELL_ORDER_V2 order. (`start_gi` lets a fixture reserve the LOW game_idx range
    for its own probe games -- the sampler's deterministic tie-breaks favour the
    lowest game_idx, so a probe game must be able to sit below the bulk supply.)"""
    spec = dict(_V2_POOL_SPEC)
    spec.update(overrides or {})
    rows, gi = [], start_gi
    for cell in CELL_ORDER_V2:
        for band, n_games in spec[cell]:
            for _ in range(n_games):
                rows.extend(_v2_game(gi, cell[0], cell[1], band))
                gi += 1
    return rows


def _abundant_pool_v2():
    return _v2_pool()


def _insufficient_pool_v2():
    """Abundant everywhere except (target, midgame), starved to 3 games (6 rows)
    against its 45-row demand -> assign_split_v2's per-cell capacity PRECHECK
    fires before any selection."""
    return _v2_pool({("target", "midgame"): [("b400_plus", 3)]})


def _pool_late_floor_unmeetable():
    """Every phase quota is comfortably fillable -- the (target, late) cell holds
    85 games (170 rows) for its 45-row quota -- but the pool contains only FIVE
    b300_399 target games (10 rows), below the 12-row floor, so NO selection can
    satisfy LATE_TARGET_FLOORS. The composition round-robin therefore SUCCEEDS
    and only the hard floor verification can catch it."""
    return _v2_pool({("target", "late"): [("b400_plus", 60), ("b300_399", 5),
                                          ("b200_299", 20)]})


def _pool_gap_starved_cell_v2():
    """(control, opening) is supplied ONLY by games whose two rows sit
    < MIN_PLY_GAP apart (red@2 + black@9), so each yields exactly ONE pickable
    row. Both capacity PRECHECKS pass (8 games x min(2, MAX_PER_GAME) = 16 >= the
    15-row demand), yet the round-robin cannot reach the cell's quota -> the
    exact-or-raise `final-manifest shortfall`, a DIFFERENT failure from
    _insufficient_pool_v2's capacity precheck. Proves a shortfall is never
    silently truncated."""
    rows = _v2_pool({("control", "opening"): []})
    gi = 10_000
    for _ in range(8):
        rows.append(_v2_row(gi, "control", "opening", "b400_plus", "red", 2))
        rows.append(_v2_row(gi, "control", "opening", "b400_plus", "black", 9))
        gi += 1
    return rows


def _pool_global_two_per_game_starved():
    """100 games, each offering a side-opposed pair in ALL EIGHT cells (16 rows
    per game). Every PER-CELL capacity is ample (100 games x min(2, MAX_PER_GAME)
    = 200 >= every cell's demand), so a v1-style per-cell accounting sees a
    healthy pool -- but under v2's GLOBAL <=MAX_PER_GAME rule those 100 games can
    yield at most 200 < CORPUS_SIZE (240) rows, so the corpus is impossible. Only
    an accounting that caps a game's TOTAL contribution can see it."""
    rows = []
    for gi in range(100):
        for cell in CELL_ORDER_V2:
            band = "b400_plus" if cell[1] != "late" else "b400_plus"
            rows.extend(_v2_game(gi, cell[0], cell[1], band))
    return rows


def _pool_with_multicell_game_v2():
    """Abundant pool, but game 0 spans TWO cells with FOUR rows:
    (target, opening) red@2 + black@15 and (target, early_mid) black@27 + red@40.
    EVERY pair of those four plies is >= MIN_PLY_GAP apart and no hash repeats,
    so nothing but the GLOBAL <=MAX_PER_GAME rule can hold game 0 to 2 rows -- a
    v1-style PER-CELL cap would take all four (2 from each cell), because the
    round-robin reaches game 0 first (lowest game_idx) in BOTH of its cells,
    which share one split (whole-game isolation)."""
    rows = [r for r in _abundant_pool_v2() if r["game_idx"] != 0]
    rows += [
        _v2_row(0, "target", "opening", "b400_plus", "red", 2),
        _v2_row(0, "target", "opening", "b400_plus", "black", 15),
        _v2_row(0, "target", "early_mid", "b400_plus", "black", 27),
        _v2_row(0, "target", "early_mid", "b400_plus", "red", 40),
    ]
    return rows


def _pool_v2_gap_probe():
    """Two gap discriminators the abundant pool cannot reach.

    game 0 -- a CROSS-CELL gap: ONE row in (target, opening) (red@8) and TWO in
    (target, early_mid) (red@16, red@28). Once red@8 is taken, red@16 is only 8
    plies away: a sampler enforcing the gap only WITHIN a cell (v1's
    `sample_dev_rows` does exactly that -- its `positions` are pre-filtered to
    one cell) would take it. Both early_mid rows are RED, so `_choose_positions`'
    take_n==1 side-steering is a TIE and breaks to the LOWER ply -- such a mutant
    deterministically picks 16 regardless of the running side balance, while the
    correct GLOBAL gap filter leaves only 28.

    game 1 -- v1's within-cell probe, re-cut: FOUR (target, midgame) rows red@42,
    red@46, black@55, black@71. cap+gap pick exactly {42, 55}: dropping the
    gap-skip would take 46 (4 plies away), dropping the <=2 cap would add 71.

    A game can only ever contribute to TWO cells by giving ONE row to each -- and a
    one-row take is, by definition, not a side-opposed pair, so under the sampler's
    side-aware fill it is a PASS-2 (steered) draw. Pass 1 would otherwise fill both
    of game 0's cells from side-opposed-pair games alone and game 0 would simply
    never be reached, making the cross-cell probe vacuous. So this pool supplies
    (target, opening) and (target, early_mid) with SINGLETON games only (70 each,
    alternating red-only / black-only, so Pass 2's steering can still balance
    them): both cells are then genuinely filled by Pass 2, and game 0 -- holding
    the LOWEST game_idx, which is Pass 2's final tie-break -- is reliably drawn in
    each. The bulk supply is pushed to game_idx >= 100 to keep that range free.
    """
    rows = _v2_pool({("target", "opening"): [], ("target", "early_mid"): []},
                    start_gi=100)
    gi = 40_000
    for cell in (("target", "opening"), ("target", "early_mid")):
        for k in range(70):
            rows.extend(_v2_singleton_game(gi, cell[0], cell[1], "b400_plus",
                                           "red" if k % 2 == 0 else "black"))
            gi += 1
    rows += [
        _v2_row(0, "target", "opening", "b400_plus", "red", 8),
        _v2_row(0, "target", "early_mid", "b400_plus", "red", 16),
        _v2_row(0, "target", "early_mid", "b400_plus", "red", 28),
        _v2_row(1, "target", "midgame", "b400_plus", "red", 42),
        _v2_row(1, "target", "midgame", "b400_plus", "red", 46),
        _v2_row(1, "target", "midgame", "b400_plus", "black", 55),
        _v2_row(1, "target", "midgame", "b400_plus", "black", 71),
    ]
    return rows


# --- the SAME-SIDE fixtures (the side-balance regression) --------------------
# `_abundant_pool_v2` and every fixture above it is built from `_v2_game`: one
# game = ONE side-opposed pair in ONE cell. Such a pool is STRUCTURALLY INCAPABLE
# of offering a cell two same-side rows, so it can never express this regression
# -- which is exactly how v1's "a whole-game 2-take is side-neutral for free"
# premise (build_fpu_dev_corpus.SIDE_TOL's own note) survived into v2 untested.
#
# It is FALSE in v2 for two independent reasons, and each gets a fixture below:
#   * v2's (role, "late") SAMPLER cell aggregates THREE proposal cells
#     (late/b400_plus, late/b300_399, late/b200_299), so ONE game can offer that
#     single cell up to 3 reds + 3 blacks; and
#   * the screen's `raw_policy_role` classifies each row INDEPENDENTLY, so a
#     proposal pair's red can land in `target` while its black lands in `control`.
# Both stay PHYSICALLY HONEST (n_legal >= 528 - ply; red on even plies): the
# same-side rows below are all REDS on EVEN plies, b300_399 only at ply >= 129 and
# b200_299 only at ply >= 229.

# One game's (target, late) contribution = TWO REDS from two DIFFERENT floor
# bands -- red@130 (b300_399, ply >= 129 OK) and red@230 (b200_299, ply >= 229
# OK), 100 plies apart so NO gap filter binds -- while the opposed blacks of those
# very proposal pairs (143 / 243, odd plies) are classified `control`.
_V2_SAME_SIDE_LATE_PLIES = {"b300_399": (130, 143), "b200_299": (230, 243)}


def _pool_same_side_late_floor():
    """(target, late)'s ONLY floor-band supply is 20 SAME-SIDE (red) multi-band
    games, so the LATE_TARGET_FLOORS themselves FORCE the sampler to draw
    same-side rows: its 40 b400_plus late-target games (opposed pairs) cannot
    reach either floor, and every b300_399 / b200_299 target row in the pool is
    red. The floor pass then draws each floor band from a game via a SEPARATE
    band-restricted take_n == 1 call, so no 2-take rule can pair them -- 12 + 12
    = 24 RED late-target rows, and tuning ends up ~24 reds heavy.

    A side-balanced selection does exist in principle (the sampler could steer the
    other cells' leftovers black), but the greedy round-robin fills every other
    cell with side-opposed PAIRS and has no such slack -- so the correct behaviour
    is to RAISE, never to emit the skewed manifest. This is the reviewer's
    falsifying pool.
    """
    rows = _v2_pool({("target", "late"): [("b400_plus", 40)]})
    gi = 20_000
    for _ in range(20):
        for band, (p_red, p_black) in _V2_SAME_SIDE_LATE_PLIES.items():
            rows.append(_v2_row(gi, "target", "late", band, "red", p_red))
            rows.append(_v2_row(gi, "control", "late", band, "black", p_black))
        gi += 1
    return rows


# --- the WHOLE-GAME SPLIT ASSIGNMENT fixtures --------------------------------
# Sub-MIN_PLY_GAP side-opposed pairs: the two rows sit only 7 plies apart, so the
# game's PROFILE reports 2 rows for the cell but the fill can only ever pick ONE
# (`_choose_positions` cannot take a second row inside the gap). That gap between
# the assignment greedy's optimistic accounting and what the fill can realize is
# exactly what turns a starved `frozen_check` candidate pool into a shortfall.
_V2_TIGHT_PHASE_PLIES = {          # red on EVEN plies, black on ODD, 7 apart
    "opening": (2, 9),             # ply_bucket_of: 1-15
    "early_mid": (16, 23),         # 16-40
    "midgame": (42, 49),           # 41-90
    "late": (92, 99),              # 91+  (n_legal >= 528-99 = 429 => b400_plus)
}
# The four phases a single replay really does span, with the honest per-phase plies
# `_V2_PHASE_PLIES` / `_V2_LATE_BAND_PLIES` already pin.
_V2_ALL_PHASE_CELLS = ("opening", "early_mid", "midgame", "late")


def _v2_tight_game(game_idx, role, phase):
    """A GAP-CRIPPLED game: a side-opposed pair < MIN_PLY_GAP apart, so its profile
    claims 2 rows for the cell but it can only ever yield 1."""
    p_red, p_black = _V2_TIGHT_PHASE_PLIES[phase]
    return [_v2_row(game_idx, role, phase, "b400_plus", "red", p_red),
            _v2_row(game_idx, role, phase, "b400_plus", "black", p_black)]


def _v2_multicell_game(game_idx, role):
    """One game spanning ALL FOUR phases -- what a real replay does. It holds 8 rows
    but the GLOBAL <=MAX_PER_GAME budget still lets it give only 2, so its cells
    CONTEND for it. A large surplus of these is what made the old raw-count greedy
    dump nearly every game into `tuning`."""
    rows = []
    for phase in _V2_ALL_PHASE_CELLS:
        rows.extend(_v2_game(game_idx, role, phase, "b400_plus"))
    return rows


def _pool_frozen_starving_multicell():
    """The SPLIT-ASSIGNMENT regression pool -- a deterministic, fast (~400-game)
    distillation of what realistic screens actually do to the old greedy.

    Two ingredients, both real:
      * 60 multi-cell TARGET games (each spanning all four phases, 8 rows, but a
        <=2-row global budget) -- so the cells contend for every game; and
      * 240 GAP-CRIPPLED games (30 per role per phase) whose profile claims 2 rows
        but which can only ever yield 1 -- the optimistic-accounting trap.

    The OLD greedy compared RAW realizable row counts and broke ties toward "the
    split with the larger total remaining need". Because a multi-cell game keeps
    `realizable("tuning") == 2` for as long as ANY of its cells still wants a tuning
    row, and tuning's need (160) dwarfs frozen_check's (80), essentially every game
    went to tuning: measured on this pool, `frozen_check` is pinned at ~42 games no
    matter how large the pool grows (tuning 275, frozen 137 here vs 42 before). 42
    games x 2 rows = 84 for an 80-row demand -- and once a few of them are
    gap-crippled, `frozen_check` cannot reach its quota and the sampler raised a
    `final-manifest shortfall`. A valid assignment plainly exists (the pool has a
    large healthy surplus), so that was a FALSE infeasibility.

    Scoring by FILL FRACTION instead makes a game that closes 2 of frozen's
    remaining 80 outrank one that closes 2 of tuning's remaining 160, so the two
    candidate pools grow in the splits' own 160:80 ratio and frozen keeps real slack.
    """
    rows, gi = [], 0
    for _ in range(60):                                   # multi-cell TARGET games
        rows.extend(_v2_multicell_game(gi, "target"))
        gi += 1
    for phase in _V2_ALL_PHASE_CELLS:                     # the gap-crippled trap
        for role in ("target", "control"):
            for _ in range(30):
                rows.extend(_v2_tight_game(gi, role, phase))
                gi += 1
    for band in LATE_TARGET_FLOORS:                       # the late floors' supply
        for _ in range(20):
            rows.extend(_v2_game(gi, "target", "late", band))
            gi += 1
    for phase in ("opening", "early_mid", "midgame"):     # healthy CONTROL supply
        for _ in range(12):
            rows.extend(_v2_game(gi, "control", phase, "b400_plus"))
            gi += 1
    for band in ("b400_plus", "b300_399", "b200_299"):
        for _ in range(12):
            rows.extend(_v2_game(gi, "control", "late", band))
            gi += 1
    return rows


# Two RED plies per cell -- even (red) and >= MIN_PLY_GAP apart, each inside its
# phase's real ply_bucket_of range and its band's honest n_legal >= 528 - ply floor
# (b300_399 needs ply >= 129, b200_299 needs ply >= 229).
_V2_ALL_RED_PLIES = {
    ("opening", "b400_plus"): (2, 14),        # ply_bucket_of: 1-15
    ("early_mid", "b400_plus"): (16, 28),     # 16-40
    ("midgame", "b400_plus"): (42, 56),       # 41-90
    ("late", "b400_plus"): (92, 104),         # 91+
    ("late", "b300_399"): (130, 142),         # ...and ply >= 129
    ("late", "b200_299"): (230, 242),         # ...and ply >= 229
}


def _pool_all_red():
    """A pool holding NO black row anywhere: every game contributes TWO RED rows
    (even plies, >= MIN_PLY_GAP apart) to one cell. Composition is comfortably
    satisfiable and both late floors are reachable, so the sampler fills all 240
    rows and clears every floor -- and then MUST refuse, because every split is 100%
    red. GENUINELY side-infeasible: no split assignment, no draw order and no number
    of retries can conjure a black row that does not exist, so the raise is CORRECT.
    """
    rows, gi = [], 0
    for cell in CELL_ORDER_V2:
        for band, n_games in _V2_POOL_SPEC[cell]:
            for _ in range(n_games):
                p1, p2 = _V2_ALL_RED_PLIES[(cell[1], band)]
                rows.append(_v2_row(gi, cell[0], cell[1], band, "red", p1))
                rows.append(_v2_row(gi, cell[0], cell[1], band, "red", p2))
                gi += 1
    return rows


def _pool_side_sorted_by_game_idx():
    """The GAME-ORDER regression pool. (target, midgame) is supplied by THREE
    blocks, deliberately ordered so that ascending game_idx is the WORST possible
    draw order:

      gi   0- 29  30 RED-only games   (red@42 + red@56)      <- naive order hits
      gi  30- 59  30 BLACK-only games (black@43 + black@57)      these first
      gi  60-139  80 side-OPPOSED-pair games (red@42 + black@55)

    The first two blocks hold NO gap-valid side-opposed pair, so
    `_choose_positions_v2` cannot rescue them: every take from one MUST move the
    balance by +-2. The third block is side-neutral and, on its own, can fill the
    cell's whole 45-row demand in EITHER split.

    So a side-balanced 240-row manifest plainly exists -- just draw the opposed
    block. But the naive `sorted(cand_games)` walk never looks at side: it drains
    the RED block first, filling tuning's 30-row quota with 30 REDS
    (|red - black| = 30), and the exact-or-raise side check fires. A FALSE
    INFEASIBILITY, purely from the draw order.

    The opposed block is what makes this test isolate the ORDERING and nothing
    else: because each of its games is side-neutral by itself, ANY subset of them
    that `assign_split_v2` hands a split can still fill that split balanced. (That
    matters -- `assign_split_v2` is side-BLIND, so a cell supplied ONLY by
    same-side games can be dealt a side-degenerate candidate set for one split,
    which no cell-fill order can repair. See the report's residual analysis.)

    Physically honest: midgame is ply 41-90, red on EVEN plies (42, 56), black on
    ODD (43, 55, 57); every pair is >= MIN_PLY_GAP apart; all are necessarily
    b400_plus at ply <= 90.
    """
    rows = _v2_pool({("target", "midgame"): []}, start_gi=200)
    gi = 0
    for _ in range(30):                       # RED-only games FIRST (low game_idx)
        rows.append(_v2_row(gi, "target", "midgame", "b400_plus", "red", 42))
        rows.append(_v2_row(gi, "target", "midgame", "b400_plus", "red", 56))
        gi += 1
    for _ in range(30):                       # BLACK-only games NEXT
        rows.append(_v2_row(gi, "target", "midgame", "b400_plus", "black", 43))
        rows.append(_v2_row(gi, "target", "midgame", "b400_plus", "black", 57))
        gi += 1
    for _ in range(80):                       # SIDE-OPPOSED games LAST (high gi)
        rows.extend(_v2_game(gi, "target", "midgame", "b400_plus"))
        gi += 1
    return rows


def _pool_same_side_earliest_chain():
    """(target, midgame) is supplied by 50 games whose THREE rows are red@42,
    red@56, black@71 -- the EARLIEST >= MIN_PLY_GAP-apart chain (42 -> 56) is
    SAME-SIDE, while a gap-valid side-OPPOSED pair (42 + 71, 29 apart) also
    exists. v1's `_choose_positions` take_n >= 2 path walks that earliest chain
    with no side steering, so it takes two REDS from every such game; preferring
    the opposed pair takes the SAME TWO rows' worth (no shortfall) side-neutrally.

    Unlike `_pool_same_side_late_floor`, this pool is comfortably satisfiable, so
    the fixed sampler must EMIT a balanced 240-row manifest here -- not raise.
    (Physically honest: midgame is ply 41-90, so red@42/red@56 are even, black@71
    is odd, and all three are necessarily b400_plus at ply <= 90.)
    """
    rows = _v2_pool({("target", "midgame"): []})
    gi = 30_000
    for _ in range(50):
        rows.append(_v2_row(gi, "target", "midgame", "b400_plus", "red", 42))
        rows.append(_v2_row(gi, "target", "midgame", "b400_plus", "red", 56))
        rows.append(_v2_row(gi, "target", "midgame", "b400_plus", "black", 71))
        gi += 1
    return rows


_V2_DUP_SHA1 = "v2-dup-shared-red-42"


def _pool_with_duplicate_hash_v2():
    """Games 0 and 1 (both (target, midgame)) each carry THREE rows sharing ONE
    canonical_sha1 on a gap-valid red@42. Whichever game the round-robin draws
    first claims the shared hash; the other's red@42 is otherwise gap-valid AND
    earliest, so it MUST be dropped by the used_sha1 filter. Each game still
    yields MAX_PER_GAME from its two unique rows, so the dedup fires without
    starving the cell (both games stay live in the output)."""
    rows = [r for r in _abundant_pool_v2() if r["game_idx"] not in (0, 1)]
    for gi in (0, 1):
        rows.append({"game_idx": gi, "role": "target", "phase": "midgame",
                     "band": "b400_plus", "side": "red", "ply": 42,
                     "canonical_sha1": _V2_DUP_SHA1})
        rows.append(_v2_row(gi, "target", "midgame", "b400_plus", "black", 55))
        rows.append(_v2_row(gi, "target", "midgame", "b400_plus", "red", 68))
    return rows


# --- frozen v2 sampler surface ---------------------------------------------

def test_v2_cell_order_and_late_target_cell():
    """CELL_ORDER_V2 is the frozen (role, phase) cell order (SPLIT_ALLOC_V2
    insertion order, mirroring v1's CELL_ORDER), and LATE_TARGET_CELL is the ONE
    allocation cell the floors constrain -- pinned so a PHASES/role rename cannot
    silently orphan the floors."""
    assert CELL_ORDER_V2 == list(SPLIT_ALLOC_V2.keys())
    assert len(CELL_ORDER_V2) == 8
    assert set(CELL_ORDER_V2) == {(role, phase)
                                  for role in ("target", "control")
                                  for phase in PHASES}
    assert LATE_TARGET_CELL == ("target", "late")
    assert LATE_TARGET_CELL in SPLIT_ALLOC_V2
    assert SPLITS == ("tuning", "frozen_check")
    # The floors are a COMBINED requirement over that cell's 45 rows (30 tuning
    # + 15 frozen_check), and 12 + 12 = 24 <= 45 -- satisfiable in principle.
    assert sum(SPLIT_ALLOC_V2[LATE_TARGET_CELL].values()) == 45
    assert sum(LATE_TARGET_FLOORS.values()) <= 45


def test_sample_v2_rows_has_no_bucket_cap_parameter():
    """v2 DROPS v1's <=50% ply-bucket cap (design Sec 1.2: subsumed -- each phase
    is exactly 60/240 = 25%). It must not survive as a vestigial knob."""
    sig = inspect.signature(sample_v2_rows)
    assert list(sig.parameters) == ["kept", "seed"]
    assert sig.parameters["seed"].kind is inspect.Parameter.KEYWORD_ONLY
    assert not any("bucket" in name for name in sig.parameters)


# --- EXACT composition (the crux) ------------------------------------------

def test_v2_exact_split_composition_and_totals():
    rows, _stats = sample_v2_rows(_abundant_pool_v2(), seed=1)
    cell = Counter((r["role"], r["phase"], r["split"]) for r in rows)
    for (role, phase), alloc in SPLIT_ALLOC_V2.items():
        for split, n in alloc.items():
            assert cell[(role, phase, split)] == n        # every cell EXACTLY full
    assert len(rows) == CORPUS_SIZE_V2 == 240
    assert sum(1 for r in rows if r["split"] == "tuning") == 160
    assert sum(1 for r in rows if r["split"] == "frozen_check") == 80
    assert sum(1 for r in rows if r["role"] == "target") == 180
    assert sum(1 for r in rows if r["role"] == "control") == 60
    for phase in PHASES:
        assert sum(1 for r in rows if r["phase"] == phase) == 60
    assert len({r["canonical_sha1"] for r in rows}) == len(rows)     # no dup hash
    for split in SPLITS:
        sc = Counter(r["side"] for r in rows if r["split"] == split)
        assert abs(sc["red"] - sc["black"]) <= SIDE_TOL


def test_v2_every_row_carries_a_valid_split():
    rows, _ = sample_v2_rows(_abundant_pool_v2(), seed=1)
    assert all(r["split"] in SPLITS for r in rows)


def test_v2_phase_quota_subsumes_the_v1_bucket_cap():
    """Each phase is exactly 60/240 = 25% of the corpus, so v1's <=50%
    ply-bucket cap is structurally unreachable -- verified against the REAL
    ply_bucket_of of each selected row's PLY (the fixtures' plies are physically
    honest), which simultaneously pins phase == ply_bucket_of(ply)."""
    rows, _ = sample_v2_rows(_abundant_pool_v2(), seed=1)
    buckets = Counter(ply_bucket_of(r["ply"]) for r in rows)
    assert set(buckets) == set(PHASES)
    for phase in PHASES:
        assert buckets[phase] == 60 == 0.25 * CORPUS_SIZE_V2
    assert max(buckets.values()) <= 0.5 * len(rows)        # v1's cap, now free
    for r in rows:
        assert r["phase"] == ply_bucket_of(r["ply"])


# --- the late coverage floors (hard) ---------------------------------------

def test_v2_late_target_floors_are_met():
    """Among the 45 late TARGET rows -- the 30 tuning + 15 frozen_check rows
    COMBINED, not per split (design Sec 1.3) -- >= 12 b300_399 AND >= 12
    b200_299.

    Discriminating by construction: the pool's 80 b400_plus late-target rows all
    sit at LOWER game_idx than any floor-band game and alone exceed the whole
    45-row quota, so an earliest-game fill WITHOUT the floor-satisfaction pass
    takes 45 b400_plus rows and misses both floors.
    """
    pool = _abundant_pool_v2()
    late_target_pool = [r for r in pool if (r["role"], r["phase"]) == LATE_TARGET_CELL]
    b400_pool = [r for r in late_target_pool if r["band"] == "b400_plus"]
    floor_pool = [r for r in late_target_pool if r["band"] in LATE_TARGET_FLOORS]
    assert len(b400_pool) == 80 > 45          # b400_plus alone could fill the cell
    assert (max(r["game_idx"] for r in b400_pool)
            < min(r["game_idx"] for r in floor_pool))     # ...and it is visited FIRST

    rows, stats = sample_v2_rows(pool, seed=1)
    late_target = [r for r in rows if (r["role"], r["phase"]) == LATE_TARGET_CELL]
    assert len(late_target) == 45
    band_counts = Counter(r["band"] for r in late_target)
    for band, floor in LATE_TARGET_FLOORS.items():
        assert band_counts[band] >= floor                  # >= 12 / >= 12
    assert stats["late_target_band_count"] == dict(sorted(band_counts.items()))

    # The floors are a COMBINED requirement: frozen_check holds only 15 late
    # target rows, so a per-SPLIT reading of ">= 12 and >= 12" (24 rows) would be
    # arithmetically impossible -- the combined reading is the only coherent one.
    frozen_late = [r for r in late_target if r["split"] == "frozen_check"]
    assert len(frozen_late) == 15 < sum(LATE_TARGET_FLOORS.values())

    # Late CONTROL rows in the floor bands ARE selected -- yet they must NOT be
    # credited to the floors (which count late TARGET rows only).
    control_floor_rows = [r for r in rows
                          if (r["role"], r["phase"]) == ("control", "late")
                          and r["band"] in LATE_TARGET_FLOORS]
    assert control_floor_rows


def test_v2_late_floors_hold_across_seeds():
    """The floors (and the exact composition) are a property of the sampler, not
    of one lucky seed / one lucky whole-game split assignment."""
    for seed in (1, 2, 3, 7, 20260712):
        rows, _ = sample_v2_rows(_abundant_pool_v2(), seed=seed)
        assert len(rows) == 240
        cell = Counter((r["role"], r["phase"], r["split"]) for r in rows)
        for (role, phase), alloc in SPLIT_ALLOC_V2.items():
            for split, n in alloc.items():
                assert cell[(role, phase, split)] == n
        band_counts = Counter(r["band"] for r in rows
                              if (r["role"], r["phase"]) == LATE_TARGET_CELL)
        for band, floor in LATE_TARGET_FLOORS.items():
            assert band_counts[band] >= floor


def test_v2_unmeetable_late_floor_raises():
    """A pool that MEETS every phase quota but cannot reach a late floor (only 10
    b300_399 target rows exist, floor 12) must RAISE -- the floor is a hard
    requirement, never a best-effort. The failure must be the FLOOR one, not a
    composition shortfall."""
    pool = _pool_late_floor_unmeetable()
    supply = Counter(r["band"] for r in pool
                     if (r["role"], r["phase"]) == LATE_TARGET_CELL)
    assert supply["b300_399"] == 10 < LATE_TARGET_FLOORS["b300_399"]   # unmeetable
    assert sum(supply.values()) == 170 > 45          # ...yet the cell is fillable
    with pytest.raises(ValueError, match="late-target coverage floor unmet"):
        sample_v2_rows(pool, seed=1)


# --- per-game / per-split invariants ---------------------------------------

def test_v2_whole_game_split_isolation():
    rows, _ = sample_v2_rows(_pool_with_multicell_game_v2(), seed=1)
    game_splits = defaultdict(set)
    for r in rows:
        game_splits[r["game_idx"]].add(r["split"])
    assert game_splits
    assert all(len(splits) == 1 for splits in game_splits.values())


def test_v2_at_most_two_rows_per_game_globally():
    """v2's <=MAX_PER_GAME rule is GLOBAL -- across ALL cells and both splits --
    where v1 applied it PER CELL. Game 0 offers 2 mutually gap-valid rows in EACH
    of two cells, so a per-cell cap would take FOUR."""
    pool = _pool_with_multicell_game_v2()
    g0 = sorted((r for r in pool if r["game_idx"] == 0), key=lambda r: r["ply"])
    assert len(g0) == 4                                         # fixture honesty:
    assert len({(r["role"], r["phase"]) for r in g0}) == 2      # two cells,
    plies = [r["ply"] for r in g0]
    assert all(hi - lo >= MIN_PLY_GAP for lo, hi in zip(plies, plies[1:]))
    assert len({r["canonical_sha1"] for r in g0}) == 4          # no filter can bind

    rows, _ = sample_v2_rows(pool, seed=1)
    assert len(rows) == 240                       # no shortfall from the cap
    counts = Counter(r["game_idx"] for r in rows)
    assert counts and max(counts.values()) <= MAX_PER_GAME
    assert counts[0] == MAX_PER_GAME              # game 0 IS selected -- capped at 2
    assert {(r["role"], r["phase"]) for r in rows if r["game_idx"] == 0} == {
        ("target", "opening")}                    # ...all from its first cell


def test_v2_global_two_per_game_capacity_is_enforced():
    """100 games x 16 rows: every PER-CELL capacity is ample, but under the GLOBAL
    <=2/game rule the pool can yield at most 200 < 240 rows. A per-cell-only
    accounting would happily proceed; the global one must refuse."""
    pool = _pool_global_two_per_game_starved()
    per_cell = Counter((r["role"], r["phase"]) for r in pool)
    for cell, alloc in SPLIT_ALLOC_V2.items():
        assert per_cell[cell] == 200 >= alloc["tuning"] + alloc["frozen_check"]
    with pytest.raises(ValueError, match="global capacity"):
        sample_v2_rows(pool, seed=1)


def test_v2_min_ply_gap_within_game_holds_across_cells():
    """>= MIN_PLY_GAP between ANY two rows selected from one game -- including
    rows drawn from DIFFERENT cells (v1 only enforced it within a cell)."""
    rows, _ = sample_v2_rows(_pool_v2_gap_probe(), seed=1)
    assert len(rows) == 240
    by_game = defaultdict(list)
    for r in rows:
        by_game[r["game_idx"]].append(r["ply"])
    for plies in by_game.values():
        plies.sort()
        for lo, hi in zip(plies, plies[1:]):
            assert hi - lo >= MIN_PLY_GAP
    # game 0's two rows come from two DIFFERENT cells -- so the cross-cell gap
    # filter was genuinely exercised -- and red@16 (8 plies from red@8) is skipped.
    assert len({(r["role"], r["phase"]) for r in rows if r["game_idx"] == 0}) == 2
    assert sorted(by_game[0]) == [8, 28]
    # game 1: the sub-gap 46 is skipped and the <=2 cap stops before 71.
    assert sorted(by_game[1]) == [42, 55]


def test_v2_per_split_side_balance():
    """The HAPPY path only. `_abundant_pool_v2` gives every cell exactly one
    side-opposed pair per game, so it is STRUCTURALLY INCAPABLE of producing a
    same-side 2-take -- it cannot express the real side-balance regression at all.
    The two tests below carry that load."""
    rows, stats = sample_v2_rows(_abundant_pool_v2(), seed=1)
    for split in SPLITS:
        sc = Counter(r["side"] for r in rows if r["split"] == split)
        assert abs(sc["red"] - sc["black"]) <= SIDE_TOL
        assert stats["side_count"][split] == dict(sc)      # stats is a real witness


def test_v2_same_side_two_take_prefers_a_side_opposed_pair():
    """A 2-take whose EARLIEST >=gap chain is SAME-SIDE must instead take the
    gap-valid side-OPPOSED pair -- the same 2 rows' worth, so no shortfall, but
    side-neutral.

    v1's `_choose_positions` take_n >= 2 path walks the earliest chain with NO
    side steering (only its take_n == 1 branch steers), which was safe in v1 only
    because a v1 (role, band) cell could receive at most ONE side-opposed pair
    from a game. v2 breaks that premise, so the unfixed sampler takes red@42 +
    red@56 from all 50 of this pool's (target, midgame) games and emits a
    240-row manifest with tuning at red 95 / black 65 -- |diff| 30, a silent
    SIDE_TOL (2) violation.
    """
    pool = _pool_same_side_earliest_chain()
    g = sorted((r for r in pool if r["game_idx"] == 30_000), key=lambda r: r["ply"])
    assert [(r["ply"], r["side"]) for r in g] == [        # fixture honesty:
        (42, "red"), (56, "red"), (71, "black")]
    assert 56 - 42 >= MIN_PLY_GAP                         # the earliest chain is
    assert g[0]["side"] == g[1]["side"] == "red"          # ...gap-valid AND same-side
    assert 71 - 42 >= MIN_PLY_GAP                         # ...yet an OPPOSED pair
    assert len({r["canonical_sha1"] for r in g}) == 3     # ...is also gap-valid, and
    assert len({(r["role"], r["phase"]) for r in g}) == 1  # nothing else can bind

    rows, stats = sample_v2_rows(pool, seed=1)

    # No shortfall: preferring the opposed pair still yields 2 rows per game.
    assert len(rows) == CORPUS_SIZE_V2 == 240
    cell = Counter((r["role"], r["phase"], r["split"]) for r in rows)
    for (role, phase), alloc in SPLIT_ALLOC_V2.items():
        for split, n in alloc.items():
            assert cell[(role, phase, split)] == n
    for split in SPLITS:                                   # ...and it is BALANCED
        sc = Counter(r["side"] for r in rows if r["split"] == split)
        assert abs(sc["red"] - sc["black"]) <= SIDE_TOL
        assert stats["side_count"][split] == dict(sc)

    # The discriminator: every same-side game that gave 2 rows gave one red + one
    # black -- the opposed pair (42, 71) -- and red@56, the unfixed sampler's
    # second pick, is NEVER selected.
    same_side_rows = [r for r in rows if r["game_idx"] >= 30_000]
    assert same_side_rows
    assert 56 not in {r["ply"] for r in same_side_rows}
    by_game = defaultdict(list)
    for r in same_side_rows:
        by_game[r["game_idx"]].append(r)
    assert any(len(v) == MAX_PER_GAME for v in by_game.values())   # 2-takes happened
    for picked in by_game.values():
        assert len(picked) <= MAX_PER_GAME
        if len(picked) == MAX_PER_GAME:
            assert {r["side"] for r in picked} == {"red", "black"}
            assert sorted(r["ply"] for r in picked) == [42, 71]


def test_v2_cell_fill_game_order_is_side_aware():
    """The cell fill must choose WHICH GAME to draw from by side, not just by
    game_idx. Where every candidate game forces a same-side 2-take, the naive
    `sorted(cand_games)` walk drains the RED block first and skews tuning by 30 --
    a FALSE INFEASIBILITY, since alternating red/black games fills the identical
    quota from the identical pool, side-balanced. Steering at the game-selection
    level finds that selection; the row-level steering inside a single game cannot.
    """
    pool = _pool_side_sorted_by_game_idx()
    mid = [r for r in pool if (r["role"], r["phase"]) == ("target", "midgame")]
    by_game = defaultdict(list)
    for r in mid:
        by_game[r["game_idx"]].append(r)
    sides = {g: {r["side"] for r in v} for g, v in by_game.items()}
    reds = sorted(g for g, s in sides.items() if s == {"red"})
    blacks = sorted(g for g, s in sides.items() if s == {"black"})
    opposed = sorted(g for g, s in sides.items() if s == {"red", "black"})
    assert len(reds) == len(blacks) == 30       # fixture honesty: 30 + 30 same-side
    assert len(opposed) == 80                   # ...and an ample side-NEUTRAL block
    assert max(reds) < min(blacks) < min(opposed)      # but ascending game_idx hits
    for v in by_game.values():                         # every RED one FIRST.
        assert len(v) == MAX_PER_GAME                  # Each game is a real 2-take
        plies = sorted(r["ply"] for r in v)            # (both rows gap-valid), so a
        assert plies[1] - plies[0] >= MIN_PLY_GAP      # same-side game moves the
                                                       # balance by a full +-2.
    rows, stats = sample_v2_rows(pool, seed=1)

    assert len(rows) == CORPUS_SIZE_V2 == 240              # no shortfall
    cell = Counter((r["role"], r["phase"], r["split"]) for r in rows)
    for (role, phase), alloc in SPLIT_ALLOC_V2.items():
        for split, n in alloc.items():
            assert cell[(role, phase, split)] == n         # exact composition
    band_counts = Counter(r["band"] for r in rows
                          if (r["role"], r["phase"]) == LATE_TARGET_CELL)
    for band, floor in LATE_TARGET_FLOORS.items():
        assert band_counts[band] >= floor                  # floors still met
    for split in SPLITS:
        sc = Counter(r["side"] for r in rows if r["split"] == split)
        assert abs(sc["red"] - sc["black"]) <= SIDE_TOL    # ...and BALANCED
        assert stats["side_count"][split] == dict(sc)

    # The discriminator: the fill reached PAST the two same-side blocks sitting at
    # the LOWEST game_idx and drew the side-neutral block instead -- the one thing a
    # `sorted(cand_games)` walk can never do, because it never looks at side.
    # Pre-fix it takes 15 RED-only games for tuning: 30 RED rows out of the red
    # block, ZERO opposed, |red - black| = 30.
    mid_sel = [r for r in rows if (r["role"], r["phase"]) == ("target", "midgame")]
    assert len(mid_sel) == 45
    from_reds = [r for r in mid_sel if r["game_idx"] in set(reds)]
    from_opposed = [r for r in mid_sel if r["game_idx"] in set(opposed)]
    assert len(from_opposed) >= 35        # the side-NEUTRAL block did the work (0!)
    assert len(from_reds) <= 10           # the low-game_idx RED trap: avoided (30!)
    msc = Counter(r["side"] for r in mid_sel)      # and the trap cell itself came
    assert abs(msc["red"] - msc["black"]) <= 5     # out near-balanced (30/0 pre-fix)


def test_v2_side_aware_order_is_deterministic():
    """Same input + same seed => byte-identical rows AND stats. The new draw order
    must be a TOTAL, reproducible sort (no set iteration, no dict-order reliance)."""
    for pool_fn in (_pool_side_sorted_by_game_idx, _pool_same_side_earliest_chain,
                    _abundant_pool_v2):
        a, sa = sample_v2_rows(pool_fn(), seed=3)
        b, sb = sample_v2_rows(pool_fn(), seed=3)
        assert a == b
        assert sa == sb


def test_v2_same_side_only_supply_raises_rather_than_skewing():
    """When a cell can ONLY be filled with same-side rows, the sampler must RAISE
    -- never silently emit a side-skewed manifest. Side balance is a hard
    constraint under the SAME exact-or-raise contract as the floors.

    Here the LATE_TARGET_FLOORS themselves force the skew: every b300_399 /
    b200_299 TARGET row in the pool is RED (their opposed blacks were classified
    `control` -- `raw_policy_role` classifies each row independently), and one
    game offers the single (target, late) cell both of them, in two different
    bands. The floor pass draws each band via its own band-restricted take_n == 1
    call, so NO 2-take rule can pair them: the 24 floor rows are all red.

    The unfixed sampler SUCCEEDS here -- 240 rows, exact composition, floors met
    -- and hands back tuning at red 92 / black 68 (|diff| 24 vs SIDE_TOL 2). The
    raise below is therefore reached only AFTER the composition round-robin and
    the floor verification have both PASSED, which is what makes it specifically a
    side-balance failure and not a shortfall in disguise. (Task 6's
    `post_screen_qualification` is the pre-registered place to reject such a screen
    earlier, before the sampler is ever reached.)
    """
    pool = _pool_same_side_late_floor()
    late_target = [r for r in pool if (r["role"], r["phase"]) == LATE_TARGET_CELL]
    floor_rows = [r for r in late_target if r["band"] in LATE_TARGET_FLOORS]
    assert {r["side"] for r in floor_rows} == {"red"}     # fixture honesty: the ONLY
    assert len(floor_rows) == 40                          # floor-band supply is RED,
    for band, floor in LATE_TARGET_FLOORS.items():        # ...ample for the floors,
        assert sum(1 for r in floor_rows if r["band"] == band) == 20 > floor
    assert {r["side"] for r in late_target                # ...while b400_plus (which
            if r["band"] == "b400_plus"} == {"red", "black"}   # cannot reach a floor)
    by_game = defaultdict(list)                                # is side-opposed.
    for r in floor_rows:
        by_game[r["game_idx"]].append(r)
    two_band_games = [g for g, v in by_game.items()
                      if {r["band"] for r in v} == set(LATE_TARGET_FLOORS)]
    assert len(two_band_games) == 20      # ONE game, ONE cell, TWO same-side bands
    probe = sorted(by_game[two_band_games[0]], key=lambda r: r["ply"])
    assert [(r["ply"], r["side"], r["band"]) for r in probe] == [
        (130, "red", "b300_399"), (230, "red", "b200_299")]
    assert 230 - 130 >= MIN_PLY_GAP       # no gap filter binds -- only the side rule

    with pytest.raises(ValueError, match="per-split side balance violated"):
        sample_v2_rows(pool, seed=1)


def test_v2_same_side_late_floor_pool_is_never_skewed_across_seeds():
    """The CONTRACT, on the reviewer's original falsifying pool: under every seed
    the sampler either RAISES or returns a manifest that is exactly composed,
    floor-satisfying AND side-balanced. It never emits a skewed one.

    Both branches are legitimate and both occur here, which is the point. This pool
    is genuinely HARD -- its whole floor-band target supply is red -- but it is not
    infeasible: the retry over whole-game split assignments plus the fill's
    deficit-side correction does find a perfectly balanced selection under some
    seeds (e.g. seed 3, on the 3rd ordering: tuning 80/80, frozen_check 40/40).
    Under others, 8 orderings are not enough and it conservatively refuses. Pinning
    "always raises" would therefore be pinning the GREEDY's luck, not the contract.

    The unconditional exact-or-raise pin lives in
    `test_v2_all_red_pool_always_raises_on_side_balance`, on a pool that is
    GENUINELY side-infeasible -- no retry can ever rescue it.
    """
    for seed in (1, 2, 3, 7, 20260712):
        try:
            rows, stats = sample_v2_rows(_pool_same_side_late_floor(), seed=seed)
        except ValueError as exc:                       # conservative refusal: fine
            assert "side balance violated" in str(exc)
            continue
        # ...but if it DID return a manifest, that manifest must be fully valid.
        assert len(rows) == CORPUS_SIZE_V2 == 240
        cell = Counter((r["role"], r["phase"], r["split"]) for r in rows)
        for (role, phase), alloc in SPLIT_ALLOC_V2.items():
            for split, n in alloc.items():
                assert cell[(role, phase, split)] == n
        band_counts = Counter(r["band"] for r in rows
                              if (r["role"], r["phase"]) == LATE_TARGET_CELL)
        for band, floor in LATE_TARGET_FLOORS.items():
            assert band_counts[band] >= floor
        for split in SPLITS:
            sc = Counter(r["side"] for r in rows if r["split"] == split)
            assert abs(sc["red"] - sc["black"]) <= SIDE_TOL
            assert stats["side_count"][split] == dict(sc)


def test_v2_assignment_feeds_the_scarce_split_and_avoids_a_false_shortfall():
    """The whole-game split assignment must be CAPACITY-AWARE. `frozen_check` is the
    scarce split -- 80 rows, and every game it uses costs a whole <=2-row budget --
    so a greedy that compares RAW remaining need over-feeds `tuning` (160) and pins
    frozen at a starved sliver of games. With multi-cell games contending for the
    global budget and some games gap-crippled, that sliver cannot reach frozen's
    quota and the sampler raised a `final-manifest shortfall` on a pool that plainly
    has a valid assignment -- a FALSE infeasibility.
    """
    pool = _pool_frozen_starving_multicell()
    by_game = defaultdict(list)
    for r in pool:
        by_game[r["game_idx"]].append(r)

    # Fixture honesty: multi-cell games really do span 4 cells with 8 rows...
    multi = [g for g, v in by_game.items()
             if len({(r["role"], r["phase"]) for r in v}) == 4]
    assert len(multi) == 60
    assert all(len(by_game[g]) == 8 for g in multi)
    # ...and the gap-crippled games really are crippled: their profile claims 2 rows
    # for the cell, but the two are < MIN_PLY_GAP apart, so the fill can pick ONE.
    tight = [g for g, v in by_game.items()
             if len(v) == 2 and abs(v[0]["ply"] - v[1]["ply"]) < MIN_PLY_GAP]
    assert len(tight) == 240
    for g in tight:
        v = by_game[g]
        assert {r["side"] for r in v} == {"red", "black"}      # a real opposed pair
        assert len({(r["role"], r["phase"]) for r in v}) == 1  # ...in ONE cell

    rows, stats = sample_v2_rows(pool, seed=1)

    # No false shortfall: a complete, exactly-composed, floor-satisfying, balanced
    # manifest (pre-fix: `final-manifest shortfall` in a frozen_check cell).
    assert len(rows) == CORPUS_SIZE_V2 == 240
    cell = Counter((r["role"], r["phase"], r["split"]) for r in rows)
    for (role, phase), alloc in SPLIT_ALLOC_V2.items():
        for split, n in alloc.items():
            assert cell[(role, phase, split)] == n
    band_counts = Counter(r["band"] for r in rows
                          if (r["role"], r["phase"]) == LATE_TARGET_CELL)
    for band, floor in LATE_TARGET_FLOORS.items():
        assert band_counts[band] >= floor
    for split in SPLITS:
        sc = Counter(r["side"] for r in rows if r["split"] == split)
        assert abs(sc["red"] - sc["black"]) <= SIDE_TOL

    # The MECHANISM: the candidate pools now grow in the splits' own 160:80 ratio,
    # so frozen_check keeps real slack. Pre-fix it was pinned near 42 games however
    # big the pool grew -- 84 rows for an 80-row demand, and any gap-crippled game
    # in that sliver broke it.
    per_split = stats["n_games_per_split"]
    assert sum(per_split.values()) == stats["n_games_total"] == len(by_game)
    ratio = per_split["tuning"] / per_split["frozen_check"]
    want = SPLIT_TOTALS["tuning"] / SPLIT_TOTALS["frozen_check"]      # 160/80 == 2
    assert want == 2
    assert abs(ratio - want) < 0.5                    # ~2:1, NOT a starved sliver
    assert per_split["frozen_check"] > 100            # pre-fix: ~42


def test_v2_all_red_pool_always_raises_on_side_balance():
    """The UNCONDITIONAL exact-or-raise pin. `_pool_all_red` holds no BLACK row at
    all, so every split is 100% red and NO assignment, ordering or fill can ever
    balance it -- the raise is CORRECT, and no number of retries may paper over it.

    It must fail on SIDE BALANCE specifically, not on composition or a floor: the
    pool fills every cell exactly and clears both late floors, so reaching the side
    check proves the manifest was otherwise perfectly valid and was refused solely
    for being skewed. That is the guarantee the whole check exists for.
    """
    pool = _pool_all_red()
    assert {r["side"] for r in pool} == {"red"}          # fixture honesty: NO black
    for seed in (1, 2, 3, 7, 20260712):
        with pytest.raises(ValueError, match="per-split side balance violated"):
            sample_v2_rows(pool, seed=seed)


def test_v2_no_duplicate_hash():
    rows, _ = sample_v2_rows(_abundant_pool_v2(), seed=1)
    shas = [r["canonical_sha1"] for r in rows]
    assert len(shas) == len(set(shas))


def test_v2_duplicate_hash_is_excluded():
    rows, _ = sample_v2_rows(_pool_with_duplicate_hash_v2(), seed=1)
    shas = [r["canonical_sha1"] for r in rows]
    assert len(shas) == len(set(shas))            # used_sha1 filter left no dup
    assert shas.count(_V2_DUP_SHA1) == 1          # the collided hash survives once
    # both colliding games stay live, so the exclusion was FORCED (the loser lost
    # its red@42 to the filter, not by going unselected).
    assert {0, 1} <= {r["game_idx"] for r in rows}


def test_v2_duplicate_hash_within_one_game_is_excluded():
    """The dedup must hold for ANY input, not only for a (hash-deduped) real
    screen: game 0 carries the SAME canonical_sha1 on TWO gap-valid rows of its
    OWN, which `_choose_positions` would otherwise return in a single batch --
    the batch is screened against `used_sha1` BEFORE any of it is claimed.

    Deliberately NOT physically honest (two positions 26 plies apart in one game
    have different peg counts, so they can never share a canonical hash, and the
    screen stage's collision filter would drop such a row anyway) -- it exists
    solely to prove the per-row re-check fires, mirroring this file's other
    intentionally-unrealistic guard fixture."""
    rows_pool = [r for r in _abundant_pool_v2() if r["game_idx"] != 0]
    same = "v2-same-hash-in-one-game"
    for ply, side in ((42, "red"), (68, "red")):        # 26 plies apart: gap-valid
        rows_pool.append({"game_idx": 0, "role": "target", "phase": "midgame",
                          "band": "b400_plus", "side": side, "ply": ply,
                          "canonical_sha1": same})
    rows_pool.append(_v2_row(0, "target", "midgame", "b400_plus", "black", 55))

    rows, _ = sample_v2_rows(rows_pool, seed=1)
    assert len(rows) == 240
    shas = [r["canonical_sha1"] for r in rows]
    assert len(shas) == len(set(shas))            # no dup ANYWHERE in the output
    assert shas.count(same) == 1                  # claimed exactly once
    g0 = [r["ply"] for r in rows if r["game_idx"] == 0]
    assert len(g0) <= MAX_PER_GAME
    assert 42 in g0 and 68 not in g0              # the 2nd copy of the hash dropped


# --- determinism ------------------------------------------------------------

def test_v2_determinism_same_seed():
    a, sa = sample_v2_rows(_abundant_pool_v2(), seed=1)
    b, sb = sample_v2_rows(_abundant_pool_v2(), seed=1)
    assert a == b
    assert sa == sb


def test_v2_assign_split_v2_is_whole_game_and_deterministic():
    pool = _abundant_pool_v2()
    profile = {}
    for r in pool:
        profile.setdefault(r["game_idx"], Counter())[(r["role"], r["phase"])] += 1
    a = assign_split_v2(profile, seed=1)
    b = assign_split_v2(profile, seed=1)
    assert a == b                                  # deterministic under seed
    assert set(a) == set(profile)                  # every game assigned
    assert all(v in SPLITS for v in a.values())


# --- shortfalls are ERRORS, never silent truncation -------------------------

def test_v2_insufficient_pool_raises():
    with pytest.raises(ValueError, match="capacity"):
        sample_v2_rows(_insufficient_pool_v2(), seed=1)


def test_v2_final_manifest_shortfall_from_pick_filters_raises():
    """Distinct from the capacity precheck: every capacity check passes, but the
    gap-skip per-pick filter starves (control, opening), so the round-robin hits
    its exact-or-raise guard instead of returning a short manifest."""
    with pytest.raises(ValueError, match="final-manifest shortfall"):
        sample_v2_rows(_pool_gap_starved_cell_v2(), seed=1)


# --- stats are an INDEPENDENT witness --------------------------------------

def test_v2_stats_shape_and_independent_counts():
    """cell_counts / late_target_band_count must be counted FROM THE SELECTED
    ROWS (an independent composition + floor witness), not re-emitted from the
    frozen quotas. v1's `bucket_count` is GONE (no bucket cap in v2)."""
    rows, stats = sample_v2_rows(_abundant_pool_v2(), seed=1)
    assert set(stats) == {"n_rows", "seed", "assignment_attempt", "cell_counts",
                          "side_count", "late_target_band_count",
                          "n_games_per_split", "n_games_total"}
    # PROVENANCE: which of the ASSIGN_ATTEMPTS whole-game split orderings won.
    assert stats["assignment_attempt"] in range(ASSIGN_ATTEMPTS)
    assert "bucket_count" not in stats
    assert stats["n_rows"] == len(rows) == 240
    assert stats["seed"] == 1

    actual = Counter((r["role"], r["phase"], r["split"]) for r in rows)
    for (role, phase), alloc in SPLIT_ALLOC_V2.items():
        for split, quota in alloc.items():
            key = f"{role}|{phase}|{split}"
            assert stats["cell_counts"][key] == actual[(role, phase, split)]
            assert stats["cell_counts"][key] == quota
    assert sum(stats["cell_counts"].values()) == len(rows) == 240

    late_actual = Counter(r["band"] for r in rows
                          if (r["role"], r["phase"]) == LATE_TARGET_CELL)
    assert stats["late_target_band_count"] == dict(sorted(late_actual.items()))
    assert sum(stats["late_target_band_count"].values()) == 45
    for band, floor in LATE_TARGET_FLOORS.items():
        assert stats["late_target_band_count"][band] >= floor

    assert (stats["n_games_per_split"]["tuning"]
            + stats["n_games_per_split"]["frozen_check"]) == stats["n_games_total"]
    assert stats["n_games_total"] == len({r["game_idx"] for r in _abundant_pool_v2()})


# ---------------------------------------------------------------------------
# Task 4 -- the v2 GEOMETRIC preflight (role-AGNOSTIC, witness-backed)
#
# Frozen design ref: docs/superpowers/specs/2026-07-12-fpu-dev-corpus-v2-phase-design.md
#   Sec 1.7 ("Two-stage feasibility"), Sec 1.2 (SPLIT_ALLOC_V2), Sec 1.3 (the
#   late floors), Sec 1.5 (proposal cells).
# v2 plan Task 4. Exercises `v2_geometry_feasibility` (pure core) and
# `v2_preflight_source` (the thin file-read wrapper) in
# scripts/GPU/alphazero/fpu_dev_corpus_v2.py.
#
# STAGE 1 OF TWO. Role (target vs control) comes from the evaluator's raw
# policy, so it is NOT provable from geometry. This preflight therefore proves
# ONLY what geometry can prove -- per-PHASE CANDIDATE capacity (60/phase), late
# CANDIDATE availability per band cell (>=12 / >=12 IGNORING role), the GLOBAL
# <=MAX_PER_GAME cap, >=MIN_PLY_GAP, per-split side balance and the whole-game
# 160/80 split -- JOINTLY, via a constructive witness. It does NOT prove the
# target-ROLE floors and does NOT prove DISJOINTNESS; both are Task 6's
# post-screen `select` step. Two tests below assert that NON-CLAIM explicitly.
#
# Everything here is SYNTHETIC geometry, in the exact `enumerate_v2_proposals`
# schema: either the REAL enumerator run over honest synthetic replays
# (`_v2_full_games`, reusing Task 2's `_honest_replay`) or hand-built proposals
# (`_v2_proposal`) for the pathological cases the enumerator CANNOT emit. No
# evaluator, no MCTS, no GPU/MLX, no real replay files.
#
# Fixtures stay PHYSICALLY HONEST (Task 0's `n_legal >= 528 - ply` floor, held
# as the tight equality `n_legal = 528 - ply`, plus red-on-even-ply parity), so
# b300_399 appears only at ply >= 129 and b200_299 only at ply >= 229 -- both
# "late", which is exactly why the floors live in the late phase.
# ---------------------------------------------------------------------------

# The REAL enumerator's output on a 330-ply honest replay, pinned by
# `test_full_replay_yields_up_to_twelve_proposals_in_cell_then_ply_order` above:
# one side-opposed pair per PROPOSAL_CELLS cell, at exactly these (red, black)
# plies. Restated here so the hand-built fixtures below sit on the same geometry
# as the real enumerator's, and cross-checked against it in
# `test_v2_synthetic_proposals_match_the_real_enumerator`.
_V2_CELL_PLIES = {
    ("opening", None): (0, 13),
    ("early_mid", None): (16, 29),
    ("midgame", None): (42, 55),
    ("late", "b400_plus"): (92, 105),
    ("late", "b300_399"): (130, 143),
    ("late", "b200_299"): (230, 243),
}

# SAME-SIDE (both RED, i.e. both EVEN, >= MIN_PLY_GAP apart) ply pairs -- the
# side-aliasing fixtures. Still honest: midgame (ply <= 90) is necessarily
# b400_plus; b300_399 needs ply >= 129; b200_299 needs ply >= 229.
_V2_SAME_SIDE_RED_PLIES = {
    ("midgame", None): (42, 56),
    ("late", "b300_399"): (130, 142),
    ("late", "b200_299"): (230, 242),
}


def _v2_proposal(game_idx, ply, cell):
    """One hand-built proposal in the EXACT `enumerate_v2_proposals` schema, on the
    TIGHT physical floor `n_legal = 528 - ply` -- so `band_of(n_legal)` really is
    the cell's band, `side_to_move_for_ply` red-on-even parity really holds, and
    `ply_bucket_of(ply)` really is the cell's phase. Every field is DERIVED from the
    real v1/v2 primitives, never hand-typed."""
    n_legal = 528 - ply
    return {
        "game_idx": game_idx,
        "ply": ply,
        "side": side_to_move_for_ply(ply),
        "phase": ply_bucket_of(ply),
        "n_legal": n_legal,
        "band": band_of(n_legal),
        "proposal_cell": cell,
    }


def _v2_full_games(n_games, n_moves=330):
    """`n_games` games of REAL `enumerate_v2_proposals` output over Task 2's honest
    330-ply replay: 12 proposals each -- a side-opposed pair in EVERY one of the 6
    PROPOSAL_CELLS, hence a candidate pair in every one of the 4 PHASES.

    This is the shape a real reservoir game has, and it is what makes the GLOBAL
    <=MAX_PER_GAME cap the binding cross-phase coupling: a game rich enough to serve
    all four phases can still only ever give TWO selected positions, so 240 rows need
    >= 120 DISTINCT games no matter how many proposals each game offers.
    """
    return {gi: enumerate_v2_proposals(_honest_replay(gi, n_moves))
            for gi in range(n_games)}


def _replace_cell_rows(by_game, cell, plies):
    """Swap every game's `cell` proposals for hand-built ones at `plies` (used to
    force a same-side cell the real enumerator can never emit -- it only ever emits
    side-OPPOSED pairs)."""
    out = {}
    for gi, rows in by_game.items():
        kept = [p for p in rows if p["proposal_cell"] != cell]
        kept += [_v2_proposal(gi, ply, cell) for ply in plies]
        out[gi] = sorted(kept, key=lambda p: p["ply"])
    return out


def _phase_short_by_game():
    """(a) `early_mid` is supplied by only 20 games (40 candidate positions) against
    its 60-position phase quota -- every OTHER phase is amply supplied, so the
    per-phase CANDIDATE capacity wall is what binds."""
    by_game = _v2_full_games(130)
    return {gi: ([p for p in rows if p["phase"] != "early_mid"] if gi >= 20 else rows)
            for gi, rows in by_game.items()}


def _late_cell_missing_by_game():
    """(b) The late CANDIDATE cell `late/b200_299` is unreachable -- ZERO proposals
    -- so the >=12 late-b200_299 candidate floor cannot be met, IGNORING role. The
    late PHASE itself is amply supplied (b400_plus + b300_399), so it is specifically
    the candidate CELL that binds, not phase capacity."""
    return {gi: [p for p in rows if p["proposal_cell"] != ("late", "b200_299")]
            for gi, rows in _v2_full_games(130).items()}


def _side_aliased_phase_by_game():
    """(c) `midgame` is supplied ONLY by SAME-SIDE (all-RED) pairs -- red@42 + red@56,
    both even, 14 plies apart, both necessarily b400_plus at ply <= 90. Its CAPACITY
    is ample (130 games x 2 = 260 >= 60), so the PAIR-BASED WITNESS cannot realize the
    phase and the gate refuses CONSERVATIVELY.

    NOT "no side-balanced selection can exist" -- one DOES (a corpus can balance
    ACROSS phases via same-side rows drawn from different phases). See
    `test_v2_side_aliasing_bounds_the_witness_strategy_not_feasibility`, which builds
    it: this is a FALSE-INFEASIBLE, the conservative direction, never a false pass."""
    return _replace_cell_rows(_v2_full_games(130), ("midgame", None),
                              _V2_SAME_SIDE_RED_PLIES[("midgame", None)])


def _side_aliased_late_cell_by_game():
    """(c') The FLOOR cell's own side wall. `late/b200_299` is supplied ONLY by
    SAME-SIDE (all-RED) pairs -- red@230 + red@242 (honest: b200_299 needs ply >= 229)
    -- so its CANDIDATE availability is ample (260 >= 12) AND the late PHASE still
    holds ample opposed pairs (from b400_plus / b300_399), yet the pair-based witness
    can draw no PAIR-game from that floor cell (same conservative refusal as (c))."""
    return _replace_cell_rows(_v2_full_games(130), ("late", "b200_299"),
                              _V2_SAME_SIDE_RED_PLIES[("late", "b200_299")])


# --- the geometry's own honesty ---------------------------------------------

def test_v2_synthetic_proposals_match_the_real_enumerator():
    """The hand-built `_v2_proposal` geometry is the REAL `enumerate_v2_proposals`
    schema at the REAL enumerated plies -- so the pathological fixtures below differ
    from a real proposal set ONLY in the one property under test, never in shape."""
    real = enumerate_v2_proposals(_honest_replay(7, 330))
    fake = [_v2_proposal(7, ply, cell)
            for cell, plies in _V2_CELL_PLIES.items() for ply in plies]
    assert sorted(real, key=lambda p: (p["proposal_cell"], p["ply"])) == \
        sorted(fake, key=lambda p: (p["proposal_cell"], p["ply"]))

    # ...and every same-side fixture ply is PHYSICALLY HONEST + genuinely same-side.
    for cell, (p1, p2) in _V2_SAME_SIDE_RED_PLIES.items():
        for ply in (p1, p2):
            row = _v2_proposal(0, ply, cell)
            assert row["side"] == "red"                      # even ply
            assert row["n_legal"] == 528 - ply               # the tight floor
            assert proposal_cell_of(row["phase"], row["n_legal"]) == cell
        assert p2 - p1 >= MIN_PLY_GAP                        # a real gap-valid 2-take


# --- (e) the FEASIBLE case: a real constructive witness ----------------------

def test_v2_feasible_geometry_returns_true_with_a_witness():
    """(e) 120 games, each offering a side-opposed pair in all 6 proposal cells --
    the MINIMUM possible under the global <=2/game cap (120 x 2 = 240 = CORPUS_SIZE).
    The witness must realize it exactly, and satisfy EVERY constraint the preflight
    claims."""
    by_game = _v2_full_games(120)
    report = v2_geometry_feasibility(by_game)
    assert report.feasible is True
    assert report.binding_constraint is None
    assert report.n_games == 120
    assert report.quota_per_phase == QUOTA_PER_PHASE == 60

    w = report.witness
    assert w is not None
    assert len(w) == CORPUS_SIZE_V2 == 240

    # (1) per-PHASE candidate capacity: exactly 60 selected per phase (45 target +
    # 15 control, but role-AGNOSTICALLY -- only the TOTAL is provable here).
    by_phase = Counter(r["phase"] for r in w)
    for phase in PHASES:
        assert by_phase[phase] == QUOTA_PER_PHASE == 60
    assert QUOTA_PER_PHASE * len(PHASES) == CORPUS_SIZE_V2

    # (2) late CANDIDATE availability per band cell: the witness's 60 late positions
    # include >= 12 in late/b300_399 AND >= 12 in late/b200_299 -- necessary (NOT
    # sufficient) for the real >=12/>=12 late-TARGET floors, which need role.
    late_cells = Counter(r["proposal_cell"] for r in w if r["phase"] == LATE_PHASE)
    assert sum(late_cells.values()) == 60
    for band, floor in LATE_TARGET_FLOORS.items():
        assert late_cells[(LATE_PHASE, band)] >= floor == 12

    # (3) GLOBAL <=MAX_PER_GAME per game (across ALL phases/cells) + >=MIN_PLY_GAP.
    plies_by_game = defaultdict(list)
    for r in w:
        plies_by_game[r["game_idx"]].append(r["ply"])
    assert len(plies_by_game) == 120                       # 120 DISTINCT games...
    assert max(len(v) for v in plies_by_game.values()) <= MAX_PER_GAME
    assert all(len(v) == MAX_PER_GAME for v in plies_by_game.values())   # ...x 2 = 240
    for plies in plies_by_game.values():
        plies.sort()
        for lo, hi in zip(plies, plies[1:]):
            assert hi - lo >= MIN_PLY_GAP

    # (4) whole-game split into the frozen 160/80 budgets, + per-split side balance.
    split_of = defaultdict(set)
    for r in w:
        split_of[r["game_idx"]].add(r["split"])
    assert all(len(s) == 1 for s in split_of.values())     # whole-game isolation
    for split in SPLITS:
        n = sum(1 for r in w if r["split"] == split)
        assert n == SPLIT_TOTALS[split]                    # tuning 160 / frozen 80
        sc = Counter(r["side"] for r in w if r["split"] == split)
        assert abs(sc["red"] - sc["black"]) <= SIDE_TOL


def test_v2_preflight_report_is_frozen_with_a_readable_role_agnostic_str():
    report = v2_geometry_feasibility(_v2_full_games(120))
    assert isinstance(report, V2PreflightReport)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.feasible = False                            # frozen: cannot mutate
    s = str(report)
    assert "feasible" in s.lower()
    for phase in PHASES:
        assert phase in s                                  # per-phase diagnostics
    for band in LATE_TARGET_FLOORS:
        assert band in s                                   # per-late-cell diagnostics
    # The rendered gate must SAY what it does not claim, so nobody reads a pass as
    # a role/disjointness guarantee.
    assert "ROLE-AGNOSTIC" in s


# --- (a)-(c) the NECESSARY walls: infeasible, binding constraint named -------

def test_v2_wall_phase_capacity_infeasible():
    """(a) A phase short of its 60 CANDIDATE positions -> infeasible, binding names
    the phase."""
    report = v2_geometry_feasibility(_phase_short_by_game())
    assert report.feasible is False
    assert report.witness is None
    assert "phase-capacity" in report.binding_constraint
    assert "early_mid" in report.binding_constraint
    assert report.realizable_by_phase["early_mid"] == 40 < QUOTA_PER_PHASE
    # ...and it really is that ONE phase: every other is amply supplied.
    for phase in PHASES:
        if phase != "early_mid":
            assert report.realizable_by_phase[phase] >= QUOTA_PER_PHASE


def test_v2_wall_late_candidate_cell_unreachable_infeasible():
    """(b) A late CANDIDATE band cell with ZERO proposals -> infeasible, binding names
    that candidate CELL (not merely the phase: the late PHASE is amply supplied)."""
    report = v2_geometry_feasibility(_late_cell_missing_by_game())
    assert report.feasible is False
    assert report.witness is None
    assert "late-candidate" in report.binding_constraint
    assert "b200_299" in report.binding_constraint
    assert report.realizable_by_late_cell["b200_299"] == 0
    assert report.pair_games_by_late_cell["b200_299"] == 0
    # phase capacity is NOT what binds -- the late phase alone clears its quota.
    assert report.realizable_by_phase[LATE_PHASE] >= QUOTA_PER_PHASE
    assert report.realizable_by_late_cell["b300_399"] >= LATE_TARGET_FLOORS["b300_399"]


def test_v2_wall_side_aliased_phase_infeasible():
    """(c) A phase whose every candidate pair is ONE-SIDED -> the gate refuses on SIDE,
    even though its candidate capacity is ample.

    CONSERVATIVE, not proven-infeasible: unlike the two CAPACITY walls above, this
    check bounds the PAIR-BASED WITNESS STRATEGY, not feasibility itself -- a valid
    selection over this very geometry does exist (built in
    `test_v2_side_aliasing_bounds_the_witness_strategy_not_feasibility`). Refusing is
    still correct behaviour; it just must not be described as a proof.

    (Unreachable from today's `enumerate_v2_proposals`, which only ever emits
    side-OPPOSED pairs -- so pair_games * 2 == realizable identically and the capacity
    wall always fires first. This is a contract guard on the PURE core, which accepts
    ANY proposal geometry.)"""
    report = v2_geometry_feasibility(_side_aliased_phase_by_game())
    assert report.feasible is False
    assert report.witness is None
    assert "side" in report.binding_constraint
    assert "midgame" in report.binding_constraint
    # capacity itself is fine -- the wall is purely side (no both-side pair-game).
    assert report.realizable_by_phase["midgame"] >= QUOTA_PER_PHASE
    assert report.pair_games_by_phase["midgame"] == 0
    assert report.red_by_phase["midgame"] == 260
    assert report.black_by_phase["midgame"] == 0


def test_v2_wall_side_aliased_late_floor_cell_infeasible():
    """(c') A FLOOR cell whose every candidate pair is one-sided -> the gate refuses on
    SIDE, even though the cell's candidate availability AND the late phase's own side
    supply are both ample. Conservative in exactly the same way as (c): it bounds the
    pair-based witness strategy, not feasibility."""
    report = v2_geometry_feasibility(_side_aliased_late_cell_by_game())
    assert report.feasible is False
    assert report.witness is None
    assert "side" in report.binding_constraint
    assert "b200_299" in report.binding_constraint
    assert report.realizable_by_late_cell["b200_299"] >= LATE_TARGET_FLOORS["b200_299"]
    assert report.pair_games_by_late_cell["b200_299"] == 0
    assert report.pair_games_by_phase[LATE_PHASE] * 2 >= QUOTA_PER_PHASE   # phase is OK


def test_v2_side_aliasing_bounds_the_witness_strategy_not_feasibility():
    """THE CLAIM-CORRECTION PIN (review finding). The two SIDE-ALIASING checks are a
    CONSERVATIVE pre-screen -- NOT true upper bounds like the two CAPACITY checks -- so
    no docstring may say a side-aliasing refusal PROVES infeasibility.

    Fixture (c)'s geometry is refused with `side-aliasing:midgame`, yet a genuine
    240-row selection over it satisfies EVERY constraint the gate claims. Built here
    as a real COUNTER-SELECTION:

      *  30 games x 2 midgame REDS   (red@42  + red@56,  gap  14) -> 60 midgame
      *  60 games x 2 BLACKS SPANNING TWO PHASES
                                     (black@13 opening + black@29 early_mid, gap 16)
                                                                  -> 60 + 60
      *  30 games x 2 late REDS      (red@130 + red@230, gap 100) -> 60 late

    The mechanism the pair-based witness structurally cannot see: per-split side
    balance is a per-SPLIT constraint, NOT a per-phase one, so one game may supply two
    SAME-SIDE rows from DIFFERENT phases and the corpus balances ACROSS phases -- with
    no phase holding a single side-opposed pair-game.

    The refusal remains CORRECT behaviour (a false-INfeasible is the conservative
    direction, and it cannot fire on real enumerator output at all). It is only the
    CLAIM that had to be corrected. This test is what keeps it corrected.
    """
    by_game = _side_aliased_phase_by_game()
    report = v2_geometry_feasibility(by_game)
    assert report.feasible is False                       # the gate REFUSES...
    assert "side-aliasing:midgame" in report.binding_constraint
    assert "not feasibility" in report.binding_constraint   # ...and says why it may err

    # ...yet here is a fully valid selection over that same geometry.
    def row(gi, ply):
        return next(p for p in by_game[gi] if p["ply"] == ply)

    selected, split_of, red_games, black_games = [], {}, [], []
    for gi in range(0, 30):            # 2 same-side REDS, one phase (midgame)
        selected += [row(gi, 42), row(gi, 56)]
        red_games.append(gi)
    for gi in range(30, 90):           # 2 same-side BLACKS, ACROSS two phases
        selected += [row(gi, 13), row(gi, 29)]
        black_games.append(gi)
    for gi in range(90, 120):          # 2 same-side REDS, both late floor cells
        selected += [row(gi, 130), row(gi, 230)]
        red_games.append(gi)
    for games in (red_games, black_games):             # whole-game 160/80 split:
        for i, gi in enumerate(games):                 # 40+40 games -> tuning,
            split_of[gi] = "tuning" if i < 40 else "frozen_check"   # 20+20 -> frozen

    # EVERY constraint the preflight claims, checked on the counter-selection:
    assert len(selected) == CORPUS_SIZE_V2 == 240                       # size
    assert Counter(r["phase"] for r in selected) == {p: QUOTA_PER_PHASE
                                                    for p in PHASES}    # 60/phase
    plies_by_game = defaultdict(list)
    for r in selected:
        plies_by_game[r["game_idx"]].append(r["ply"])
    assert len(plies_by_game) == 120
    assert max(len(v) for v in plies_by_game.values()) <= MAX_PER_GAME  # <=2/game
    for plies in plies_by_game.values():                                # >=12 gap
        assert abs(plies[0] - plies[1]) >= MIN_PLY_GAP
    late_cells = Counter(r["proposal_cell"] for r in selected
                         if r["phase"] == LATE_PHASE)
    for band, floor in LATE_TARGET_FLOORS.items():                      # late floors
        assert late_cells[(LATE_PHASE, band)] == 30 >= floor
    for split in SPLITS:                                   # 160/80 + side balance
        rows = [r for r in selected if split_of[r["game_idx"]] == split]
        assert len(rows) == SPLIT_TOTALS[split]
        sc = Counter(r["side"] for r in rows)
        assert abs(sc["red"] - sc["black"]) == 0 <= SIDE_TOL            # PERFECT

    # ...and NO phase held a single side-opposed pair-game -- which is exactly why the
    # pair-based witness could not find this selection, and why the check is a
    # strategy bound rather than a proof of infeasibility.
    assert report.pair_games_by_phase["midgame"] == 0


# --- (d) the GLOBAL <=2/game cap -- a WITNESS failure, not a proposal count ---

def test_v2_per_game_cap_exceeded_is_infeasible_and_named():
    """(d) The <=MAX_PER_GAME cap is about the SELECTION, not the proposals: one game
    may legitimately offer up to 12 proposals across the 6 cells (and every game here
    does), but it can only ever CONTRIBUTE 2 selected rows. So 240 rows need >= 120
    DISTINCT games -- and 119 games can yield at most 238.

    GENUINELY infeasible (238 < 240), not a witness artifact -- yet EVERY per-phase /
    per-cell necessary check passes with room to spare, so only the constructive
    witness, which spends one shared per-game budget ACROSS the phases, can see it.
    The binding constraint must NAME the per-game cap.
    """
    by_game = _v2_full_games(119)
    # Fixture honesty: each game offers 12 proposals -- 2 in every proposal cell...
    assert all(len(rows) == 12 for rows in by_game.values())
    assert all(Counter(p["proposal_cell"] for p in rows) ==
               Counter({c: 2 for c in PROPOSAL_CELLS}) for rows in by_game.values())
    # ...and 119 games x 2 selectable rows CANNOT reach the 240-row corpus.
    assert 119 * MAX_PER_GAME == 238 < CORPUS_SIZE_V2 == 240

    report = v2_geometry_feasibility(by_game)
    assert report.feasible is False
    assert report.witness is None
    assert "per-game" in report.binding_constraint          # the cap is NAMED...
    assert f"<={MAX_PER_GAME}/game" in report.binding_constraint
    assert "joint" in report.binding_constraint             # ...by the WITNESS

    # Every per-constraint NECESSARY check passes -- the coupling is cross-phase.
    for phase in PHASES:
        assert report.realizable_by_phase[phase] == 238 >= QUOTA_PER_PHASE
        assert report.pair_games_by_phase[phase] * 2 >= QUOTA_PER_PHASE
    for band, floor in LATE_TARGET_FLOORS.items():
        assert report.realizable_by_late_cell[band] >= floor
        assert report.pair_games_by_late_cell[band] * 2 >= floor


def test_v2_one_more_game_flips_it_feasible():
    """The knife-edge that proves (d) is the per-game cap and nothing else: adding a
    SINGLE game (119 -> 120) -- not a single extra proposal to an existing game --
    flips the identical geometry to feasible."""
    assert v2_geometry_feasibility(_v2_full_games(119)).feasible is False
    assert v2_geometry_feasibility(_v2_full_games(120)).feasible is True


# --- (f) SOUNDNESS: necessary != sufficient (the load-bearing test) ----------

def test_v2_soundness_joint_infeasible_but_all_necessary_checks_pass():
    """(f) 60 games, each offering a side-opposed pair in ALL FOUR phases. EVERY
    per-constraint necessary check passes (60 x 2 = 120 >= 60 realizable per phase; 60
    pair-games >= the 30 needed; both late candidate cells amply supplied), yet under
    the GLOBAL <=MAX_PER_GAME cap those 60 games can yield at most 120 < 240
    positions IN TOTAL. That cross-phase coupling is invisible to any per-phase
    accounting -- only the constructive WITNESS, which consumes each game exactly
    once GLOBALLY, can catch it (v1 §11.2.3, transposed from bands to phases).
    """
    report = v2_geometry_feasibility(_v2_full_games(60))

    # Every per-phase / per-cell NECESSARY check passes individually...
    for phase in PHASES:
        assert report.realizable_by_phase[phase] >= QUOTA_PER_PHASE          # capacity
        assert report.pair_games_by_phase[phase] * 2 >= QUOTA_PER_PHASE      # side
    for band, floor in LATE_TARGET_FLOORS.items():
        assert report.realizable_by_late_cell[band] >= floor                 # candidate
        assert report.pair_games_by_late_cell[band] * 2 >= floor             # ...+ side

    # ...but the JOINT problem is infeasible, and ONLY the witness catches it.
    assert report.feasible is False
    assert report.witness is None
    assert "joint" in report.binding_constraint
    assert 60 * MAX_PER_GAME == 120 < CORPUS_SIZE_V2      # genuinely infeasible


def test_v2_feasible_true_only_ever_via_a_completed_witness():
    """SOUNDNESS, structurally + behaviourally: `feasible=True` is returned from
    exactly ONE place in the source -- AFTER `_build_v2_witness` handed back a real
    witness -- so no NECESSARY check can ever certify feasibility on its own. A
    completed witness IS a feasible selection, so the gate can never be
    FALSE-feasible; it may be mildly conservative (false-INfeasible) for exotic
    geometry, which is the accepted, documented limitation (v1 :465-472)."""
    src = inspect.getsource(v2_geometry_feasibility)
    assert src.count("_report(True") == 1                 # ONE feasible=True site...
    assert "if witness is None:" in src                   # ...unreachable without one

    for by_game in (_v2_full_games(120), _v2_full_games(119), _v2_full_games(60),
                    _phase_short_by_game(), _late_cell_missing_by_game(),
                    _side_aliased_phase_by_game(), _side_aliased_late_cell_by_game()):
        rep = v2_geometry_feasibility(by_game)
        assert (rep.witness is not None) is rep.feasible
        assert (rep.binding_constraint is None) is rep.feasible


# --- the NON-CLAIMS: role floors and disjointness are Task 6's, NOT this gate -

def test_v2_preflight_does_not_claim_target_role_floors():
    """The geometric preflight must NOT claim the >=12/>=12 late-TARGET floors -- only
    the role-AGNOSTIC late CANDIDATE availability. A geometry with ample late
    candidates whose ROLES would later fail the target floor MUST still pass here;
    rejecting it is Task 6's post-screen qualification's job (design Sec 1.7).
    """
    by_game = _v2_full_games(120)
    report = v2_geometry_feasibility(by_game)
    assert report.feasible is True

    # What it DOES prove: the CANDIDATE floors, ignoring role (necessary only).
    for band, floor in LATE_TARGET_FLOORS.items():
        assert report.realizable_by_late_cell[band] >= floor
        assert sum(1 for r in report.witness
                   if r["proposal_cell"] == (LATE_PHASE, band)) >= floor

    # ROLE is not even REPRESENTABLE here: it comes from the evaluator's raw policy
    # at the (later) screen stage, so a proposal has no role field and the report has
    # no role/target field to carry a claim in.
    for rows in by_game.values():
        for p in rows:
            assert "role" not in p
    names = {f.name for f in dataclasses.fields(V2PreflightReport)}
    assert not any(("role" in n) or ("target" in n) for n in names)

    # The CONCRETE non-claim. `raw_policy_role` classifies each row INDEPENDENTLY, so
    # this role assignment is entirely legal -- and under it ZERO of the (abundant)
    # late/b200_299 candidates are `target`, so the >=12 late-TARGET floor is
    # UNSATISFIABLE. The preflight still returns feasible=True: it never looked.
    def hostile_role(p):
        return "control" if p["proposal_cell"] == (LATE_PHASE, "b200_299") else "target"

    late_b200 = [p for rows in by_game.values() for p in rows
                 if p["proposal_cell"] == (LATE_PHASE, "b200_299")]
    assert len(late_b200) == 240                        # candidates: hugely abundant
    assert sum(1 for p in late_b200 if hostile_role(p) == "target") == 0 \
        < LATE_TARGET_FLOORS["b200_299"]                # ...targets: NONE. Floor dead.
    assert v2_geometry_feasibility(by_game).feasible is True         # UNCHANGED.


def test_v2_preflight_does_not_claim_disjointness():
    """The geometric preflight must NOT claim disjointness -- it cannot even SEE a
    position identity. Proposals carry no `canonical_sha1` (that is computed at the
    screen stage from a reconstructed state), so `assert_disjoint` over the screen's
    hashes, in Task 6's `select`, is the ONLY place disjointness is ever proven.

    Concretely: `_v2_full_games` is 120 IDENTICAL replays, so game g's ply-k position
    IS game h's ply-k position -- the witness's 240 "distinct" rows are really at most
    12 distinct canonical states, maximally NON-disjoint. The preflight passes anyway.
    """
    by_game = _v2_full_games(120)
    for rows in by_game.values():
        for p in rows:
            assert "canonical_sha1" not in p            # not even representable
    geometries = {tuple(sorted((p["ply"], p["side"]) for p in rows))
                  for rows in by_game.values()}
    assert len(geometries) == 1                        # 120 games, ONE geometry

    report = v2_geometry_feasibility(by_game)
    assert report.feasible is True                     # ...and it PASSES anyway
    w = report.witness
    assert len(w) == CORPUS_SIZE_V2 == 240             # 240 selected rows...
    assert len({(r["ply"], r["side"]) for r in w}) == 12    # ...<= 12 real positions

    names = {f.name for f in dataclasses.fields(V2PreflightReport)}
    assert not any(("sha" in n) or ("hash" in n) or ("disjoint" in n) for n in names)


# --- (g) determinism ---------------------------------------------------------

def test_v2_preflight_determinism_same_input_same_report():
    """(g) Same geometry -> byte-identical report (witness included). Sorted iteration
    throughout: no set-order or dict-order reliance."""
    feasible = _v2_full_games(120)
    assert v2_geometry_feasibility(feasible) == v2_geometry_feasibility(feasible)

    for by_game in (_v2_full_games(60), _phase_short_by_game(),
                    _late_cell_missing_by_game(), _side_aliased_late_cell_by_game()):
        assert v2_geometry_feasibility(by_game) == v2_geometry_feasibility(by_game)


def test_v2_preflight_core_does_not_mutate_its_input():
    """PURITY: the core stamps `split` onto its WITNESS rows, which must be COPIES --
    the caller's proposals are never touched (they are re-read by the screen stage)."""
    by_game = _v2_full_games(120)
    before = copy.deepcopy(by_game)
    report = v2_geometry_feasibility(by_game)
    assert by_game == before                              # input byte-for-byte intact
    assert all("split" not in p for rows in by_game.values() for p in rows)
    assert all("split" in r for r in report.witness)      # ...stamped on the COPIES


def test_v2_pair_positions_is_a_distinct_constant_from_the_per_game_cap():
    """The witness's atomic unit is a side-opposed PAIR = 2 positions. That is a pair
    SIZE, not the per-game selection CAP -- they are numerically equal, and the
    witness's whole mechanism depends on it (one pair per game is exactly what makes
    "<=MAX_PER_GAME rows/game GLOBALLY across phases" true by construction), so they
    are tied by an assert rather than sharing one name for two meanings."""
    assert PAIR_POSITIONS == 2
    assert PAIR_POSITIONS <= MAX_PER_GAME
    assert PAIR_POSITIONS == MAX_PER_CELL_PER_GAME        # the enumerator's 0-or-2/cell
    # ...and the frozen quota/floors really do divide into whole pairs.
    assert QUOTA_PER_PHASE % PAIR_POSITIONS == 0
    assert QUOTA_PER_PHASE // PAIR_POSITIONS == 30        # 30 pair-games per phase
    assert sum(-(-f // PAIR_POSITIONS) for f in LATE_TARGET_FLOORS.values()) == 12 <= 30


# --- the thin I/O wrapper ----------------------------------------------------

def test_v2_preflight_source_composes_reads_enumerator_and_pure_core(tmp_path):
    """`v2_preflight_source` is the ONLY impure part: read each `rec["replay_path"]`,
    run the REAL `enumerate_v2_proposals`, then the pure core. A 2-game toy source
    cannot reach 240 rows, so it is infeasible -- what matters is that the wrapper
    composes read -> enumerate -> geometry and returns a real report.

    It also pins that the SOURCE INDEX record's `game_idx` is authoritative (v1's
    `build_candidates_by_game` keys by the record's, never the replay's): both
    replays here carry a bogus `game_idx: 999`, so a wrapper that trusted the FILE
    would collapse them into ONE game.
    """
    records = []
    for gi in range(2):
        replay = {"game_idx": 999,                       # bogus on purpose
                  "moves": [{"n_legal": 528 - ply} for ply in range(330)]}
        p = tmp_path / f"game_{gi:06d}.json"
        p.write_text(json.dumps(replay))
        records.append({"game_idx": gi, "replay_path": str(p)})

    report = v2_preflight_source(records)
    assert isinstance(report, V2PreflightReport)
    assert report.n_games == 2                          # NOT 1 -- the record wins
    assert report.n_proposals == 24                     # 2 games x 12 proposals
    assert report.feasible is False                     # 2 games << 240 rows
    assert set(report.realizable_by_phase) == set(PHASES)
    assert set(report.realizable_by_late_cell) == {"b400_plus", "b300_399", "b200_299"}


def test_v2_preflight_source_feasible_end_to_end(tmp_path):
    """End-to-end on the smallest FEASIBLE source: 120 honest replay files -> read ->
    enumerate -> witness. Proves the gate can actually PASS from disk, not only from
    an in-memory geometry."""
    records = []
    moves = [{"n_legal": 528 - ply} for ply in range(330)]
    for gi in range(120):
        p = tmp_path / f"game_{gi:06d}.json"
        p.write_text(json.dumps({"game_idx": gi, "moves": moves}))
        records.append({"game_idx": gi, "replay_path": str(p)})

    report = v2_preflight_source(records)
    assert report.feasible is True
    assert report.binding_constraint is None
    assert len(report.witness) == CORPUS_SIZE_V2 == 240
    assert report.n_games == 120


# ---------------------------------------------------------------------------
# Task 5 -- the operator `screen` stage: row schema + the two pure per-
# proposal helpers (`classify_exclusion`, `screen_row`), the required config
# loader (`V2Config` / `load_v2_config`), and STATIC verification that
# `run_screen` / `main` / `_build_v2_anchor_search_fn` are wired correctly
# WITHOUT ever invoking them -- `run_screen` loads a real checkpoint and runs
# 400-sim MCTS, an OPERATOR phase never run by this suite.
#
# Frozen design ref: docs/superpowers/specs/2026-07-12-fpu-dev-corpus-v2-phase-design.md
#   Sec 1.6 (`screen`/`select` two-artifact workflow, screen row schema), Sec
#   1.8 (the required versioned config).
# v2 plan Task 5. `_v2_proposal` (Task 4's fixture helper, defined above) is
# reused here to build fabricated proposal dicts in the exact
# `enumerate_v2_proposals` schema -- still no evaluator/MCTS/GPU/MLX/I-O.
# ---------------------------------------------------------------------------

def test_screen_fieldnames_exact_schema_in_order():
    """Interface: SCREEN_FIELDNAMES is EXACTLY these 18 columns, in this
    order (design Sec 1.6) -- `phase` and `ply_bucket` are BOTH present
    (deliberately duplicated; Task 7's diagnostic opts into stratifying by
    ply_bucket, so every row must carry it independently of `phase`)."""
    assert SCREEN_FIELDNAMES == [
        "game_idx", "ply", "side", "phase", "n_legal", "band", "ply_bucket",
        "proposal_cell", "normalized_entropy", "top1_prior", "top4_mass",
        "top8_mass", "raw_policy_role", "anchor_run", "root_value_stm",
        "anchor_eligible", "canonical_sha1", "exclusion_status",
    ]


# --- classify_exclusion: the brief's four cases -----------------------------

def test_classify_exclusion_collision_short_circuits_role_and_anchor():
    """Brief case 1: collided=True -> ("collision", anchor_run=False) --
    checked with role/anchor_eligible_val SET (not None) too, proving
    collision is checked FIRST and unconditionally overrides them (a collided
    proposal never even reaches the raw-policy pass)."""
    assert classify_exclusion(collided=True, role=None,
                              anchor_eligible_val=None) == ("collision", False)
    assert classify_exclusion(collided=True, role="target",
                              anchor_eligible_val=True) == ("collision", False)


def test_classify_exclusion_ineligible_role():
    """Brief case 2: collided=False, role=None -> ("ineligible_role", False)."""
    assert classify_exclusion(
        collided=False, role=None,
        anchor_eligible_val=None) == ("ineligible_role", False)


def test_classify_exclusion_ineligible_anchor():
    """Brief case 3: role set, anchor_eligible_val=False ->
    ("ineligible_anchor", True) -- checked for BOTH roles, since any non-None
    role reaches the anchor branch."""
    assert classify_exclusion(
        collided=False, role="target",
        anchor_eligible_val=False) == ("ineligible_anchor", True)
    assert classify_exclusion(
        collided=False, role="control",
        anchor_eligible_val=False) == ("ineligible_anchor", True)


def test_classify_exclusion_kept():
    """Brief case 4: role set, anchor_eligible_val=True -> ("kept", True)."""
    assert classify_exclusion(
        collided=False, role="target",
        anchor_eligible_val=True) == ("kept", True)
    assert classify_exclusion(
        collided=False, role="control",
        anchor_eligible_val=True) == ("kept", True)


# --- screen_row: full schema + the nullable-field rules ---------------------

_SCREEN_FEATS = {"normalized_entropy": 0.95, "top1_prior": 0.01,
                 "top4_mass": 0.03, "top8_mass": 0.05}


def test_screen_row_full_schema_kept_case():
    """`screen_row` yields the full SCREEN_FIELDNAMES schema -- exact key
    set, BOTH `band` and `ply_bucket` carried (and equal to `phase`), every
    field correctly placed from `proposal` / `feats` / the explicit kwargs."""
    proposal = _v2_proposal(7, 42, ("midgame", None))
    row = screen_row(proposal, feats=_SCREEN_FEATS, role="target",
                     anchor_run=True, root_value_stm=0.125, anchor_eligible=True,
                     canonical_sha1="deadbeef", exclusion_status="kept")

    assert set(row.keys()) == set(SCREEN_FIELDNAMES)
    assert row["game_idx"] == 7
    assert row["ply"] == 42
    assert row["side"] == proposal["side"]
    assert row["phase"] == "midgame"
    assert row["n_legal"] == proposal["n_legal"]
    assert row["band"] == proposal["band"]
    assert row["ply_bucket"] == row["phase"] == "midgame"    # deliberately equal
    assert row["proposal_cell"] == ("midgame", None)
    assert row["normalized_entropy"] == 0.95
    assert row["top1_prior"] == 0.01
    assert row["top4_mass"] == 0.03
    assert row["top8_mass"] == 0.05
    assert row["raw_policy_role"] == "target"
    assert row["anchor_run"] is True
    assert row["root_value_stm"] == 0.125
    assert row["anchor_eligible"] is True
    assert row["canonical_sha1"] == "deadbeef"
    assert row["exclusion_status"] == "kept"


def test_screen_row_collision_nulls_policy_and_anchor_fields():
    """Collision: the raw-policy pass NEVER ran, so feats=None -> the four
    policy columns are null, NOT a fabricated 0.0 -- and anchor_run=False ->
    root_value_stm/anchor_eligible null too. band/ply_bucket stay populated
    (from the proposal, unaffected by the collision)."""
    proposal = _v2_proposal(3, 130, ("late", "b300_399"))
    row = screen_row(proposal, feats=None, role=None, anchor_run=False,
                     root_value_stm=None, anchor_eligible=None,
                     canonical_sha1="collided-hash", exclusion_status="collision")

    assert row["normalized_entropy"] is None
    assert row["top1_prior"] is None
    assert row["top4_mass"] is None
    assert row["top8_mass"] is None
    assert row["raw_policy_role"] is None
    assert row["anchor_run"] is False
    assert row["root_value_stm"] is None
    assert row["anchor_eligible"] is None
    assert row["exclusion_status"] == "collision"
    assert row["band"] == proposal["band"]
    assert row["ply_bucket"] == "late"


def test_screen_row_ineligible_role_keeps_populated_policy_feats():
    """ineligible_role: the raw-policy pass DID run (feats populated) even
    though the CLASSIFICATION landed in the grey zone (role=None) -- only
    raw_policy_role and the (never-run) anchor fields are null."""
    proposal = _v2_proposal(9, 16, ("early_mid", None))
    row = screen_row(proposal, feats=_SCREEN_FEATS, role=None, anchor_run=False,
                     root_value_stm=None, anchor_eligible=None,
                     canonical_sha1="grey-zone-hash",
                     exclusion_status="ineligible_role")

    assert row["normalized_entropy"] == 0.95
    assert row["top1_prior"] == 0.01
    assert row["top4_mass"] == 0.03
    assert row["top8_mass"] == 0.05
    assert row["raw_policy_role"] is None
    assert row["anchor_run"] is False
    assert row["root_value_stm"] is None
    assert row["anchor_eligible"] is None
    assert row["exclusion_status"] == "ineligible_role"


def test_screen_row_ineligible_anchor_has_non_null_anchor_fields():
    """ineligible_anchor: the anchor DID run (anchor_run=True) -- unlike
    collision/ineligible_role, root_value_stm/anchor_eligible are non-null
    even though the row is ultimately excluded."""
    proposal = _v2_proposal(11, 92, ("late", "b400_plus"))
    row = screen_row(proposal, feats=_SCREEN_FEATS, role="control", anchor_run=True,
                     root_value_stm=0.4, anchor_eligible=False,
                     canonical_sha1="far-from-even-hash",
                     exclusion_status="ineligible_anchor")

    assert row["raw_policy_role"] == "control"
    assert row["anchor_run"] is True
    assert row["root_value_stm"] == 0.4
    assert row["anchor_eligible"] is False
    assert row["exclusion_status"] == "ineligible_anchor"


def test_screen_row_enforces_anchor_run_nullness_contract():
    """screen_row's own defensive contract: anchor_run=True requires
    non-null root_value_stm/anchor_eligible, and anchor_run=False requires
    them null -- violating either raises rather than silently persisting an
    inconsistent row."""
    proposal = _v2_proposal(1, 0, ("opening", None))
    with pytest.raises(AssertionError):
        screen_row(proposal, feats=_SCREEN_FEATS, role="target", anchor_run=True,
                  root_value_stm=None, anchor_eligible=True,
                  canonical_sha1="x", exclusion_status="kept")
    with pytest.raises(AssertionError):
        screen_row(proposal, feats=None, role=None, anchor_run=False,
                  root_value_stm=0.1, anchor_eligible=None,
                  canonical_sha1="x", exclusion_status="collision")


# --- the required config: V2Config / load_v2_config -------------------------
#
# `_v2_config_fixture` (Task B8) is the ONE place the v2 config's full schema
# lives on this file's test side -- every fabricated-config test below routes
# through it (`**overrides` steers exactly the fields a given test cares
# about), so a future required-key addition needs exactly one edit here
# instead of N independently hand-typed dicts silently drifting apart.
# (tests/test_fpu_diagnostic_modes.py owns a second, deliberately-separate
# `_v2_config_dict` for Group 1's own fixtures -- see that file's docstring;
# both are kept in sync with `_V2_CONFIG_REQUIRED_KEYS`, the real source of
# truth, and both are exercised against the REAL `load_v2_config`, so a drift
# would fail loudly in whichever file forgot the key, not silently pass.)

def _v2_config_fixture(tmp_path, **overrides):
    """Every `_V2_CONFIG_REQUIRED_KEYS` key, fabricated -- including the five
    Task B8 top-level paths (spec Sec 2.2 "New top-level (paths)":
    `config_schema_version`, `protocol_path`, `match_summary_path`,
    `replay_dir`, `report_out`) and `expected_fingerprints`'s real nine
    measured-identity sub-keys (spec Sec 2.2 "expected_fingerprints
    (extended)") -- the exact shape `fpu_dev_reservoir_protocol.derive_config`
    emits, cross-checked directly by the producer/consumer round-trip tests
    below (`test_derive_config_round_trips_through_load_v2_config` et al.)."""
    cfg = {
        "source_index_path": str(tmp_path / "reservoir_index.jsonl"),
        "seed_range": [20270000, 20274800],
        "selection_seed": 20260712,
        "phase_allocation": {f"{role}|{phase}": alloc
                             for (role, phase), alloc in SPLIT_ALLOC_V2.items()},
        "late_floors": dict(LATE_TARGET_FLOORS),
        "enumerator_params": {"min_ply_gap": MIN_PLY_GAP,
                              "max_per_cell_per_game": MAX_PER_CELL_PER_GAME},
        "new_collapse_stratum": "ply_bucket",
        "checkpoint": str(tmp_path / "checkpoint.npz"),
        "forbidden_manifests": [str(tmp_path / "forbidden_a.csv")],
        "screen_out": str(tmp_path / "fpu_dev_source_screen.csv"),
        "select_out": str(tmp_path / "fpu_dev_corpus_v2_manifest.csv"),
        # Task B8 -- spec Sec 2.2 "New top-level (paths)".
        "config_schema_version": 1,
        "protocol_path": str(tmp_path / "reservoir_protocol.json"),
        "match_summary_path": str(tmp_path / "match_summary.json"),
        "replay_dir": str(tmp_path / "replays"),
        "report_out": str(tmp_path / "qualify_report.json"),
        # Task B8 -- spec Sec 2.2 "expected_fingerprints (extended)": the
        # real nine measured identities `derive_config` emits (the checkpoint
        # identity split into three roles; `protocol_sha1`/`match_summary_
        # sha1` added).
        "expected_fingerprints": {
            "protocol_sha1": "deadbeef-protocol",
            "source_index_sha1": "deadbeef",
            "replay_data_sha1": "deadbeef-replay",
            "match_summary_sha1": "deadbeef-summary",
            "source_file_sha1s": {},
            "forbidden_manifest_sha1s": {},
            "reservoir_checkpoint_a_identity": "ckpt_a.npz:deadbeef-a",
            "reservoir_checkpoint_b_identity": "ckpt_b.npz:deadbeef-b",
            "anchor_checkpoint_identity": "ckpt_a.npz:deadbeef-a",
        },
    }
    cfg.update(overrides)
    return cfg


def test_load_v2_config_loads_a_complete_config(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(_v2_config_fixture(tmp_path)))

    config = load_v2_config(str(p))

    assert isinstance(config, V2Config)
    assert config.config_path == str(p)
    assert config.source_index_path == str(tmp_path / "reservoir_index.jsonl")
    assert config.seed_range == (20270000, 20274800)
    assert config.selection_seed == 20260712
    assert config.new_collapse_stratum == "ply_bucket"
    assert config.checkpoint == str(tmp_path / "checkpoint.npz")
    assert config.forbidden_manifests == (str(tmp_path / "forbidden_a.csv"),)
    assert config.screen_out == str(tmp_path / "fpu_dev_source_screen.csv")
    assert config.select_out == str(tmp_path / "fpu_dev_corpus_v2_manifest.csv")
    # Task B8: the five new required top-level paths round-trip onto V2Config.
    assert config.config_schema_version == 1
    assert config.protocol_path == str(tmp_path / "reservoir_protocol.json")
    assert config.match_summary_path == str(tmp_path / "match_summary.json")
    assert config.replay_dir == str(tmp_path / "replays")
    assert config.report_out == str(tmp_path / "qualify_report.json")
    assert config.expected_fingerprints == {
        "protocol_sha1": "deadbeef-protocol",
        "source_index_sha1": "deadbeef",
        "replay_data_sha1": "deadbeef-replay",
        "match_summary_sha1": "deadbeef-summary",
        "source_file_sha1s": {},
        "forbidden_manifest_sha1s": {},
        "reservoir_checkpoint_a_identity": "ckpt_a.npz:deadbeef-a",
        "reservoir_checkpoint_b_identity": "ckpt_b.npz:deadbeef-b",
        "anchor_checkpoint_identity": "ckpt_a.npz:deadbeef-a",
    }
    # v1-matching defaults when the config itself omits these two knobs.
    assert config.eval_batch_size == 14
    assert config.stall_flush_sims == 48


def test_load_v2_config_honors_explicit_throughput_overrides(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(_v2_config_fixture(
        tmp_path, eval_batch_size=7, stall_flush_sims=99)))

    config = load_v2_config(str(p))

    assert config.eval_batch_size == 7
    assert config.stall_flush_sims == 99


def test_load_v2_config_raises_on_missing_required_key(tmp_path):
    raw = _v2_config_fixture(tmp_path)
    del raw["checkpoint"]
    p = tmp_path / "config.json"
    p.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="checkpoint"):
        load_v2_config(str(p))


def test_load_v2_config_raises_naming_every_missing_key(tmp_path):
    raw = _v2_config_fixture(tmp_path)
    del raw["checkpoint"]
    del raw["late_floors"]
    p = tmp_path / "config.json"
    p.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="checkpoint") as excinfo:
        load_v2_config(str(p))
    assert "late_floors" in str(excinfo.value)


def test_load_v2_config_raises_naming_every_missing_new_schema_key(tmp_path):
    """Task B8: the five NEW required top-level keys are enforced exactly
    like the pre-existing ones -- dropping all five at once names every one
    in the single raised error (no silent omission)."""
    raw = _v2_config_fixture(tmp_path)
    new_keys = ("config_schema_version", "protocol_path", "match_summary_path",
               "replay_dir", "report_out")
    for key in new_keys:
        del raw[key]
    p = tmp_path / "config.json"
    p.write_text(json.dumps(raw))

    with pytest.raises(ValueError) as excinfo:
        load_v2_config(str(p))
    for key in new_keys:
        assert key in str(excinfo.value)


# ---------------------------------------------------------------------------
# Task B8 -- `_V2_CORPUS_SOURCES` gains the qualification module (v1's own
# `_CORPUS_SOURCES` stays untouched), and the producer/consumer round-trip: a
# config `fpu_dev_reservoir_protocol.derive_config` actually emits must load
# cleanly through the extended `load_v2_config` (spec Sec 2.2/Sec 6/Sec 9).
# ---------------------------------------------------------------------------

def test_v2_corpus_sources_contains_the_qualification_module():
    """Spec Sec 2.2 amendment 4: 'the qualification module is
    result-determining for the corpus it produces' -- added to the v2 source
    set only."""
    names = [p.name for p in _V2_CORPUS_SOURCES]
    assert "fpu_dev_reservoir_protocol.py" in names


def test_v2_corpus_sources_paths_all_exist_on_disk():
    """Sanity: a typo'd filename would otherwise silently hash to
    `fpu_provenance`'s `"missing"` sentinel instead of failing loud."""
    missing = [str(p) for p in _V2_CORPUS_SOURCES if not p.exists()]
    assert missing == []


def test_v1_corpus_sources_is_byte_unchanged_by_task_b8():
    """v1's OWN result-determining source set must stay untouched -- Task B8
    only ever touches `fpu_dev_corpus_v2._V2_CORPUS_SOURCES` (the v2 set),
    never `build_fpu_dev_corpus._CORPUS_SOURCES`. Pinned as the exact prior
    tuple (membership AND order), reconstructed from `_MODULE_DIR` rather
    than a hardcoded absolute path so this stays portable across checkouts."""
    module_dir = Path(inspect.getfile(load_forbidden_hashes)).resolve().parent
    assert _CORPUS_SOURCES == (
        module_dir / "build_fpu_dev_corpus.py",
        module_dir / "mcts.py",
        module_dir / "fpu_state_hash.py",
        module_dir / "goal_line_trigger_probe_cases.py",
        module_dir / "game" / "twixt_state.py",
    )
    assert len(_CORPUS_SOURCES) == 5


def _minimal_protocol_and_measurements(**protocol_overrides):
    """The SMALLEST `(protocol, measurements)` pair
    `fpu_dev_reservoir_protocol.derive_config` will accept -- it never
    validates protocol SHAPE (that is `build_protocol`'s job, in the sibling
    module) or performs I/O, it only maps already-declared/measured fields
    (see that function's own docstring), so a hand-built dict covering
    exactly the keys it reads is sufficient for a producer/consumer
    round-trip -- no need to reconstruct a fully protocol-CONFORMANT
    reservoir (that machinery is tests/test_fpu_dev_reservoir_protocol.py's
    own `_conformant_reservoir`, which this file does not import -- these two
    test files stay independent)."""
    protocol = {
        "anchor": "checkpoint_a",
        "base_seed": 20270000,
        "games": 6,
        "source_index_path": "reservoir_index.jsonl",
        "selection_seed": 20260712,
        "phase_allocation": {},
        "late_floors": {},
        "enumerator_params": {},
        "new_collapse_stratum": "ply_bucket",
        "checkpoint_a": {"path": "ckpt_a.npz", "identity": "ckpt_a.npz:aaa"},
        "checkpoint_b": {"path": "ckpt_b.npz", "identity": "ckpt_b.npz:bbb"},
        "forbidden_manifests": ["forbidden_a.csv"],
        "screen_out": "fpu_dev_source_screen.csv",
        "select_out": "fpu_dev_corpus_v2_manifest.csv",
        "mcts_eval_batch_size": 14,
        "mcts_stall_flush_sims": 48,
        "config_schema_version": 1,
        "match_summary_path": "match_summary.json",
        "replay_dir": "replays",
        "report_out": "qualify_report.json",
    }
    protocol.update(protocol_overrides)
    measurements = ReservoirMeasurements(
        jsonl_rows=[], sidecars_by_idx={}, summary={},
        checkpoint_identities={"reservoir_a": "ckpt_a.npz:aaa",
                               "reservoir_b": "ckpt_b.npz:bbb",
                               "anchor": "ckpt_a.npz:aaa"},
        generation_source_sha1s={}, generation_git_commit="deadbeef-git",
        source_index_sha1="deadbeef-index", replay_data_sha1="deadbeef-replay",
        match_summary_sha1="deadbeef-summary",
        source_file_sha1s={"fpu_dev_corpus_v2.py": "deadbeef-src"},
        forbidden_manifest_sha1s={"forbidden_a.csv": "deadbeef-forbidden"},
    )
    return protocol, measurements


def test_derive_config_round_trips_through_load_v2_config(tmp_path):
    """THE producer/consumer cross-check (Task B8 brief): a config
    `fpu_dev_reservoir_protocol.derive_config` actually emits loads cleanly
    through the extended `load_v2_config` -- proving `V2Config`'s new
    required fields are not merely SHAPED like spec Sec 2.2, but hard-match
    what the real producer emits, field for field."""
    protocol, measurements = _minimal_protocol_and_measurements()
    derived = derive_config(
        protocol, measurements,
        protocol_path=str(tmp_path / "reservoir_protocol.json"))
    p = tmp_path / "config.json"
    p.write_text(json.dumps(derived))

    config = load_v2_config(str(p))

    assert isinstance(config, V2Config)
    assert config.config_schema_version == derived["config_schema_version"]
    assert config.protocol_path == derived["protocol_path"]
    assert config.match_summary_path == derived["match_summary_path"]
    assert config.replay_dir == derived["replay_dir"]
    assert config.report_out == derived["report_out"]
    assert config.checkpoint == protocol["checkpoint_a"]["path"]
    assert config.eval_batch_size == protocol["mcts_eval_batch_size"]
    assert config.stall_flush_sims == protocol["mcts_stall_flush_sims"]
    assert config.expected_fingerprints == derived["expected_fingerprints"]
    assert set(config.expected_fingerprints) == {
        "protocol_sha1", "source_index_sha1", "replay_data_sha1",
        "match_summary_sha1", "source_file_sha1s", "forbidden_manifest_sha1s",
        "reservoir_checkpoint_a_identity", "reservoir_checkpoint_b_identity",
        "anchor_checkpoint_identity",
    }


@pytest.mark.parametrize("key", sorted(_V2_CONFIG_REQUIRED_KEYS))
def test_derive_config_output_missing_any_required_key_raises(tmp_path, key):
    """The same cross-check, inverted: a REAL `derive_config` output with any
    ONE required key stripped is refused by `load_v2_config`, naming it --
    exercised against the genuine producer shape, not just a hand-fabricated
    test dict."""
    protocol, measurements = _minimal_protocol_and_measurements()
    derived = derive_config(
        protocol, measurements,
        protocol_path=str(tmp_path / "reservoir_protocol.json"))
    del derived[key]
    p = tmp_path / "config.json"
    p.write_text(json.dumps(derived))

    with pytest.raises(ValueError, match=key):
        load_v2_config(str(p))


# --- the fpu-off 400-sim anchor's own small pure pieces ----------------------

def test_v2_anchor_sims_is_400():
    assert ANCHOR_SIMS_V2 == 400


def test_v2_anchor_seed_is_deterministic_and_input_sensitive():
    assert _v2_anchor_seed(3, 41) == _v2_anchor_seed(3, 41)          # deterministic
    assert _v2_anchor_seed(3, 41) != _v2_anchor_seed(3, 42)          # ply-sensitive
    assert _v2_anchor_seed(3, 41) != _v2_anchor_seed(4, 41)          # game-sensitive
    assert _v2_anchor_seed(3, 41) == ANCHOR_SEED_BASE_V2 ^ 3 ^ 41


# --- static wiring: signatures, argparse, and lazy-import resolution --------
# NONE of these invoke run_screen/main/MCTS -- only signature/source-text/
# find_spec inspection, exactly like Task 4's own
# test_v2_feasible_true_only_ever_via_a_completed_witness (inspect.getsource).

def test_v2_operator_functions_exist_with_expected_call_signatures():
    """run_screen/main/load_v2_config exist, are callable, and take exactly
    the parameters the brief's interfaces specify."""
    assert callable(run_screen)
    assert list(inspect.signature(run_screen).parameters) == ["config"]

    assert callable(main)
    main_params = inspect.signature(main).parameters
    assert list(main_params) == ["argv"]
    assert main_params["argv"].default is None

    assert callable(load_v2_config)
    assert list(inspect.signature(load_v2_config).parameters) == ["path"]

    assert callable(classify_exclusion)
    assert set(inspect.signature(classify_exclusion).parameters) == {
        "collided", "role", "anchor_eligible_val"}

    assert callable(screen_row)
    assert set(inspect.signature(screen_row).parameters) == {
        "proposal", "feats", "role", "anchor_run", "root_value_stm",
        "anchor_eligible", "canonical_sha1", "exclusion_status"}


def test_v2_mode_argument_accepts_screen_and_select():
    """argparse wiring (widened by Task 6): --mode is `("screen", "select")` and
    --config is required for BOTH. `--screen` (the PERSISTED screen artifact
    `select` re-reads) is REQUIRED by `select` and REJECTED by `screen` -- which is
    how "screen and select are NEVER the same invocation" is enforced at the CLI,
    and how an operator cannot mistake `--screen` for naming the screen's OUTPUT
    (that is `config.screen_out`)."""
    args = _parse_v2_args(["--mode", "screen", "--config", "cfg.json"])
    assert args.mode == "screen"
    assert args.config == "cfg.json"
    assert args.screen is None

    args = _parse_v2_args(["--mode", "select", "--config", "cfg.json",
                           "--screen", "screen.csv"])
    assert args.mode == "select"
    assert args.screen == "screen.csv"

    with pytest.raises(SystemExit):
        _parse_v2_args(["--config", "cfg.json"])           # --mode is required
    with pytest.raises(SystemExit):
        _parse_v2_args(["--mode", "screen"])                # --config is required
    with pytest.raises(SystemExit):
        _parse_v2_args(["--mode", "select"])                # --config is required
    with pytest.raises(SystemExit):                         # select REQUIRES --screen
        _parse_v2_args(["--mode", "select", "--config", "cfg.json"])
    with pytest.raises(SystemExit):                         # ...and screen REJECTS it
        _parse_v2_args(["--mode", "screen", "--config", "cfg.json",
                        "--screen", "screen.csv"])
    with pytest.raises(SystemExit):
        _parse_v2_args(["--mode", "nonsense", "--config", "cfg.json"])


def test_v2_operator_lazy_imports_resolve_without_executing_them():
    """Every heavy import inside the operator functions (`.eval_runner`,
    `.mcts`, `.build_teacher_calibration_manifest`) must resolve to a REAL,
    importable module -- checked via `importlib.util.find_spec`, which
    locates a module's spec WITHOUT running its top-level code (unlike
    `import`/`importlib.import_module`), so this check can never itself pull
    mlx/torch into sys.modules, regardless of what those modules do at
    import time. Source-text inspection (the SAME `inspect.getsource`
    pattern Task 4 already uses, e.g.
    test_v2_feasible_true_only_ever_via_a_completed_witness) confirms each
    import statement lives INSIDE the relevant operator function's own body
    -- never at this module's top level, which is exactly why importing
    fpu_dev_corpus_v2 never triggers them (see
    test_v2_module_import_pulls_no_gpu_or_mlx below) -- and that the
    expected call sites are actually present."""
    import importlib.util

    for mod_name in ("scripts.GPU.alphazero.eval_runner",
                     "scripts.GPU.alphazero.mcts",
                     "scripts.GPU.alphazero.build_teacher_calibration_manifest"):
        assert importlib.util.find_spec(mod_name) is not None, mod_name

    anchor_src = inspect.getsource(_build_v2_anchor_search_fn)
    assert "from .eval_runner import" in anchor_src
    assert "from .mcts import MCTS" in anchor_src
    assert "search_with_root" in anchor_src
    assert "fpu_policy_mass_reduction=None" in anchor_src

    screen_src = inspect.getsource(run_screen)
    assert "from .build_teacher_calibration_manifest import _teacher_infer" in screen_src
    assert "_build_v2_anchor_search_fn(" in screen_src
    assert "enumerate_v2_proposals(" in screen_src
    assert "classify_exclusion(" in screen_src
    assert "screen_row(" in screen_src
    assert "v2_preflight_source(" in screen_src   # gates before the evaluator loads
    assert "write_screen_csv(" in screen_src
    assert "write_screen_meta(" in screen_src


# --- screen artifact persistence: pure stdlib I/O, safe to exercise directly
# (no MCTS/GPU/MLX/checkpoint anywhere in write_screen_csv / write_screen_meta
# / v2_screen_provenance -- unlike run_screen/main, these are ordinary file
# writers over already-computed data, so calling them directly does not
# violate "never invoke run_screen/main/MCTS").

def test_write_screen_csv_round_trips_null_and_tuple_fields(tmp_path):
    """A kept row's real values and a collision row's null policy/anchor
    columns both round-trip through a real CSV write + read: None becomes an
    empty field (never a fabricated value) and the tuple-valued
    proposal_cell round-trips as its str()."""
    kept_row = screen_row(
        _v2_proposal(5, 13, ("opening", None)), feats=_SCREEN_FEATS, role="target",
        anchor_run=True, root_value_stm=0.2, anchor_eligible=True,
        canonical_sha1="kept-hash", exclusion_status="kept")
    collision_row = screen_row(
        _v2_proposal(6, 0, ("opening", None)), feats=None, role=None,
        anchor_run=False, root_value_stm=None, anchor_eligible=None,
        canonical_sha1="collided-hash", exclusion_status="collision")
    out_csv = tmp_path / "fpu_dev_source_screen.csv"

    write_screen_csv([kept_row, collision_row], str(out_csv))

    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["exclusion_status"] for r in rows] == ["kept", "collision"]
    assert rows[0]["proposal_cell"] == str(("opening", None))
    assert rows[0]["root_value_stm"] == "0.2"
    assert rows[1]["root_value_stm"] == ""          # None -> empty CSV field
    assert rows[1]["anchor_eligible"] == ""
    assert rows[1]["normalized_entropy"] == ""


def test_v2_screen_provenance_has_the_sec_1_8_fingerprint_keys(tmp_path):
    """v2_screen_provenance -- pure/stdlib-only via fpu_provenance -- carries
    every fingerprint the brief names (config hash + source_file_sha1s,
    replay_data_sha1, source_index_sha1, protocol hash, match-summary hash,
    THREE checkpoint identities, forbidden-manifest hashes, runtime_
    provenance) PLUS `screen_csv_sha1`, the screen ARTIFACT's own hash (Task
    B10: the one link the ten INPUT identities leave unfingerprinted). None
    inputs resolve to fpu_provenance's own "none" sentinel rather than
    raising -- including the two reservoir-checkpoint identities, which
    degrade to "none" when there is no `protocol_path` to read them from."""
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")

    prov = v2_screen_provenance(
        config_path=str(config_path), source_index_path=None,
        protocol_path=None, match_summary_path=None, checkpoint=None,
        forbidden_manifests=[], base_mcts_config={"n_simulations": 400})

    assert set(prov) == {
        "config_sha1", "protocol_sha1", "match_summary_sha1",
        "source_file_sha1s", "source_index_sha1", "replay_data_sha1",
        "reservoir_checkpoint_a_identity", "reservoir_checkpoint_b_identity",
        "anchor_checkpoint_identity", "forbidden_manifest_sha1s",
        "screen_csv_sha1", "base_mcts_config", "add_noise", "runtime_provenance"}
    assert prov["config_sha1"] != "none"            # config_path IS a real file
    assert prov["protocol_sha1"] == "none"           # protocol_path was None
    assert prov["match_summary_sha1"] == "none"      # match_summary_path was None
    assert prov["source_index_sha1"] == "none"       # source_index_path was None
    assert prov["reservoir_checkpoint_a_identity"] == "none"
    assert prov["reservoir_checkpoint_b_identity"] == "none"
    assert prov["anchor_checkpoint_identity"] == "none"
    assert prov["screen_csv_sha1"] == "none"         # screen_csv defaulted to None
    assert prov["add_noise"] is False
    assert prov["base_mcts_config"] == {"n_simulations": 400}
    assert "fpu_dev_corpus_v2.py" in prov["source_file_sha1s"]
    assert "python_version" in prov["runtime_provenance"]


def test_v2_screen_provenance_reservoir_checkpoint_identities_come_from_the_protocol(
        tmp_path):
    """`reservoir_checkpoint_a_identity` / `reservoir_checkpoint_b_identity` are
    NOT read from any direct path parameter (there isn't one) -- they come from
    the PROTOCOL JSON at `protocol_path`'s own `checkpoint_a`/`checkpoint_b`
    `"path"` fields (design Sec 2.1: the two reservoir players' paths live only
    in the protocol; `config`/`v2_screen_provenance`'s `checkpoint` parameter is
    just the single anchor). Two DISTINCT checkpoint files -> two DISTINCT
    identities, and `anchor_checkpoint_identity` (from the SEPARATE `checkpoint`
    parameter) is independent of both."""
    ckpt_a = tmp_path / "ckpt_a.npz"
    ckpt_a.write_bytes(b"checkpoint-a-bytes")
    ckpt_b = tmp_path / "ckpt_b.npz"
    ckpt_b.write_bytes(b"checkpoint-b-bytes-different")
    protocol_path = tmp_path / "reservoir_protocol.json"
    protocol_path.write_text(json.dumps({
        "checkpoint_a": {"path": str(ckpt_a)},
        "checkpoint_b": {"path": str(ckpt_b)},
    }))

    prov = v2_screen_provenance(
        config_path=None, source_index_path=None, protocol_path=str(protocol_path),
        match_summary_path=None, checkpoint=str(ckpt_a), forbidden_manifests=[],
        base_mcts_config=None)

    assert prov["reservoir_checkpoint_a_identity"] == (
        f"ckpt_a.npz:{fpu_provenance.file_sha1(str(ckpt_a))}")
    assert prov["reservoir_checkpoint_b_identity"] == (
        f"ckpt_b.npz:{fpu_provenance.file_sha1(str(ckpt_b))}")
    assert prov["anchor_checkpoint_identity"] == (
        f"ckpt_a.npz:{fpu_provenance.file_sha1(str(ckpt_a))}")
    assert (prov["reservoir_checkpoint_a_identity"]
            != prov["reservoir_checkpoint_b_identity"])
    # the anchor happens to equal reservoir_a here (same file) but is computed
    # from a DIFFERENT parameter (`checkpoint`, not the protocol) -- pinned by
    # test_v2_three_checkpoint_identities_are_distinct_and_independently_checked
    # below, which proves that independence under tampering, not just equality.
    assert prov["anchor_checkpoint_identity"] == prov["reservoir_checkpoint_a_identity"]


def test_v2_screen_provenance_reservoir_checkpoint_identities_degrade_gracefully(
        tmp_path):
    """A missing/unreadable/malformed protocol -- or one missing a checkpoint
    role -- degrades the two reservoir-checkpoint identities to `"none"` rather
    than raising FROM PROVENANCE COMPUTATION itself (the same best-effort
    philosophy as every other identity here); `validate_screen_identities` is
    what turns the resulting mismatch into a raised refusal, not this
    function."""
    kw = dict(config_path=None, source_index_path=None, match_summary_path=None,
              checkpoint=None, forbidden_manifests=[], base_mcts_config=None)

    missing = v2_screen_provenance(
        protocol_path=str(tmp_path / "does-not-exist.json"), **kw)
    assert missing["reservoir_checkpoint_a_identity"] == "none"
    assert missing["reservoir_checkpoint_b_identity"] == "none"

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("not valid json {{{")
    malformed = v2_screen_provenance(protocol_path=str(malformed_path), **kw)
    assert malformed["reservoir_checkpoint_a_identity"] == "none"
    assert malformed["reservoir_checkpoint_b_identity"] == "none"

    incomplete_path = tmp_path / "incomplete.json"
    incomplete_path.write_text(json.dumps({"checkpoint_a": {"path": "x"}}))
    incomplete = v2_screen_provenance(protocol_path=str(incomplete_path), **kw)
    # role "a" resolves to a (nonexistent) path -> a "name:missing" identity string,
    # NOT the "none" sentinel (that is reserved for a genuinely ABSENT path).
    assert incomplete["reservoir_checkpoint_a_identity"] == "x:missing"
    assert incomplete["reservoir_checkpoint_b_identity"] == "none"   # role "b" absent entirely


def test_v2_checkpoint_identity_format_matches_fpu_dev_reservoir_protocol(tmp_path):
    """Task B10 brief: "Cross-check the exact key names + format against B7's
    `derive_config` output and B3's `measure_reservoir` so (A)/(B)/(C) agree."

    `measure_reservoir` (B3) pins `checkpoint_identities["reservoir_a"/"reservoir_b"
    /"anchor"]` via `fpu_dev_reservoir_protocol._checkpoint_identity`, and
    `derive_config` (B7) copies those verbatim into the config's pre-registered (A)
    `expected_fingerprints["reservoir_checkpoint_a_identity"]` etc. -- so for (A) to
    ever agree with THIS module's (B)/(C) recompute, `v2_screen_provenance`'s own
    checkpoint-hashing rule must produce the IDENTICAL string, over the SAME file, as
    `fpu_dev_reservoir_protocol._checkpoint_identity`. Proven directly here (not just
    by code-reading) -- the two functions are independent (Sec 6: no shared import
    for this idiom), so only a byte-for-byte comparison over a REAL file can show
    they compute the same thing."""
    from scripts.GPU.alphazero.fpu_dev_reservoir_protocol import (
        _checkpoint_identity as reservoir_protocol_checkpoint_identity)
    ckpt = tmp_path / "some_checkpoint.npz"
    ckpt.write_bytes(b"identity-format-cross-check-bytes")

    other_module_identity = reservoir_protocol_checkpoint_identity(str(ckpt))
    this_module_identity = v2_screen_provenance(
        config_path=None, source_index_path=None, protocol_path=None,
        match_summary_path=None, checkpoint=str(ckpt), forbidden_manifests=[],
        base_mcts_config=None)["anchor_checkpoint_identity"]

    assert other_module_identity == this_module_identity
    assert other_module_identity.startswith("some_checkpoint.npz:")


def test_v2_screen_provenance_fingerprints_the_screen_artifact_itself(tmp_path):
    """`screen_csv_sha1` really is the screen CSV's own bytes: it changes when the
    artifact changes. Without it the screen's OUTPUT -- the very rows `select`
    re-derives the manifest from -- is the one unfingerprinted link in the chain."""
    screen_csv = tmp_path / "fpu_dev_source_screen.csv"
    screen_csv.write_text("game_idx,ply\n1,2\n")
    kw = dict(config_path=None, source_index_path=None, protocol_path=None,
              match_summary_path=None, checkpoint=None,
              forbidden_manifests=[], base_mcts_config=None)

    before = v2_screen_provenance(screen_csv=str(screen_csv), **kw)["screen_csv_sha1"]
    screen_csv.write_text("game_idx,ply\n1,3\n")          # one byte of one row
    after = v2_screen_provenance(screen_csv=str(screen_csv), **kw)["screen_csv_sha1"]

    assert before != "none" and after != "none"
    assert before != after


def test_write_screen_meta_enriches_with_provenance_without_clobbering(tmp_path):
    """write_screen_meta adds a DERIVED "provenance" block without disturbing
    the caller's own meta keys (mirrors v1's write_meta contract), and that
    block carries all ELEVEN Task B10 identities (a meta built without
    `protocol_path`/`match_summary_path` keys degrades those two identities'
    inputs to "none" via `.get()` -- never a crash)."""
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    out_csv = tmp_path / "fpu_dev_source_screen.csv"

    write_screen_meta(str(out_csv), {
        "config_path": str(config_path), "source_index_path": None,
        "checkpoint": None, "forbidden_manifests": [], "n_proposals": 0,
        "base_mcts_config": None,
    })

    meta_path = out_csv.with_name(out_csv.name + ".meta.json")
    meta = json.loads(meta_path.read_text())
    assert meta["n_proposals"] == 0                  # caller key passes through untouched
    assert meta["source_index_path"] is None
    assert "provenance" in meta
    assert meta["provenance"]["config_sha1"] != "none"
    assert set(SCREEN_IDENTITY_KEYS) <= set(meta["provenance"])
    assert meta["provenance"]["protocol_sha1"] == "none"
    assert meta["provenance"]["match_summary_sha1"] == "none"


# --- import purity -----------------------------------------------------------

def test_v2_module_import_pulls_no_gpu_or_mlx():
    """The whole v2 module -- the PURE SECTION (constants, enumerator,
    sampler, preflight; the preflight's only impure part is a stdlib
    json/pathlib file read) AND the Task-5 OPERATOR SHELL (`run_screen`,
    `main`, `_build_v2_anchor_search_fn`, `load_v2_config`) -- must stay
    importable without ever touching GPU/MLX: the operator shell's heavy
    imports (`.eval_runner`, `.mcts`, `.build_teacher_calibration_manifest`)
    are all LAZY, inside the functions that need them, never at module top
    level."""
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; import scripts.GPU.alphazero.fpu_dev_corpus_v2 as m; "
         "print(sorted(k for k in sys.modules if 'mlx' in k or 'torch' in k))"],
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


# ---------------------------------------------------------------------------
# Task 6 -- the PURE `select` stage: identity hard-match + post-screen
# role/floor qualification + deterministic selection
#
# Frozen design ref: docs/superpowers/specs/2026-07-12-fpu-dev-corpus-v2-phase-design.md
#   Sec 1.6 (`screen`/`select` are SEPARATE operator invocations; `select` is
#   PURE and re-runnable from the persisted screen alone), Sec 1.7 (STAGE 2 of
#   the two-stage feasibility split -- the post-screen qualification the
#   role-AGNOSTIC geometric preflight structurally could not make), Sec 1.8
#   (the pre-registered fingerprints the select stage hard-matches).
# v2 plan Task 6.
#
# EVERYTHING HERE IS PURE. `select` never loads the evaluator, MCTS, GPU/MLX or
# a checkpoint -- it reads FILE BYTES (config / index / replays / checkpoint /
# forbidden manifests) for the identity hashes and nothing else. So unlike the
# Task-5 `screen` tests, these exercise the real functions end-to-end on real
# tmp_path files; only `main` itself stays uninvoked (plan Global Constraints:
# "No main()/MCTS/operator run in tests"), verified STATICALLY instead, exactly
# as Task 5 verifies `run_screen`'s wiring.
#
# The fixtures below fabricate a complete SCREEN (rows in the real
# SCREEN_FIELDNAMES schema, built by the real `screen_row`/`classify_exclusion`)
# from the Task-3 sampler pools, staying PHYSICALLY HONEST throughout: n_legal is
# the tight floor `528 - ply`, so `band_of(n_legal)` really is the row's band,
# `side_to_move_for_ply` really is its side and `ply_bucket_of` really is its
# phase -- all ASSERTED in `_screen_row_for`, never hand-typed.
# ---------------------------------------------------------------------------

def _screen_row_for(r, status="kept"):
    """One physically-honest SCREEN_FIELDNAMES row for a fabricated Task-3
    sampler row `r` (game_idx, role, phase, band, side, ply, canonical_sha1), at
    exclusion status `status`.

    Every derived field comes from the REAL v1/v2 primitives -- `band_of`,
    `side_to_move_for_ply`, `ply_bucket_of`, `anchor_eligible`,
    `classify_exclusion`, `screen_row` -- so a fixture can never drift from the
    screen the operator stage would actually write, and the row's own
    role/anchor nullness obeys `screen_row`'s contract by construction rather
    than by hand.
    """
    ply, n_legal = r["ply"], 528 - r["ply"]           # the tight physical floor
    assert band_of(n_legal) == r["band"], (r, n_legal)
    assert side_to_move_for_ply(ply) == r["side"], r
    assert ply_bucket_of(ply) == r["phase"], r

    proposal = {
        "game_idx": r["game_idx"], "ply": ply, "side": r["side"],
        "phase": r["phase"], "n_legal": n_legal, "band": r["band"],
        "proposal_cell": (r["phase"], r["band"] if r["phase"] == "late" else None),
    }
    collided = status == "collision"
    # A collided/grey-zone proposal has NO role: the raw-policy pass either never
    # ran (collision) or landed in the grey band (ineligible_role).
    role = None if status in ("collision", "ineligible_role") else r["role"]
    # The anchor only ever runs for a real role; a `kept` root is near-even, an
    # `ineligible_anchor` root is not -- and the REAL `anchor_eligible` predicate
    # is what decides, from the fixture's own root value.
    root_value_stm = None if role is None else (0.1 if status == "kept" else 0.4)
    anchor_elig = None if role is None else anchor_eligible(root_value_stm)
    assert anchor_elig is (None if role is None else status == "kept"), status

    got_status, anchor_run = classify_exclusion(
        collided=collided, role=role, anchor_eligible_val=anchor_elig)
    assert got_status == status, (got_status, status)
    return screen_row(
        proposal, feats=(None if collided else _SCREEN_FEATS), role=role,
        anchor_run=anchor_run, root_value_stm=root_value_stm,
        anchor_eligible=anchor_elig, canonical_sha1=r["canonical_sha1"],
        exclusion_status=status)


def _screen_from_pool(pool_rows, extra=()):
    """A complete fabricated SCREEN: every Task-3 pool row as `kept`, plus each
    `(row, status)` in `extra` as a NON-kept proposal -- the excluded rows a real
    screen persists alongside the kept ones (design Sec 1.6: every proposal is
    recorded, never only the survivors)."""
    return ([_screen_row_for(r, "kept") for r in pool_rows]
            + [_screen_row_for(r, st) for r, st in extra])


def _proposals_from_screen(screen_rows):
    """The ROLE-AGNOSTIC proposal geometry the Task-4 preflight would have seen:
    EVERY screened proposal -- kept and excluded alike -- projected back to the
    `enumerate_v2_proposals` schema. The screen schema is a strict superset of the
    proposal schema, so this is a pure PROJECTION, never a re-derivation -- which
    is what lets a test run stage 1 and stage 2 over the very same geometry."""
    by_game = defaultdict(list)
    for row in screen_rows:
        by_game[row["game_idx"]].append(
            {k: row[k] for k in ("game_idx", "ply", "side", "phase", "n_legal",
                                 "band", "proposal_cell")})
    return dict(by_game)


def _late_candidates(screen_rows, band):
    """Every screened late/`band` proposal, IGNORING role -- what the role-agnostic
    geometric preflight counts as a late CANDIDATE for the floor."""
    return [r for r in screen_rows if r["proposal_cell"] == ("late", band)]


def _late_target_kept(screen_rows, band):
    """The late/`band` rows that actually survived the screen AS TARGETS -- what
    LATE_TARGET_FLOORS really counts. The gap between this and `_late_candidates`
    is precisely why the two-stage feasibility split exists."""
    return [r for r in screen_rows
            if r["exclusion_status"] == "kept" and r["raw_policy_role"] == "target"
            and r["phase"] == "late" and r["band"] == band]


def _extra_late_b200_299_candidates(start_gi, per_status=10):
    """(row, status) pairs supplying ABUNDANT late/b200_299 CANDIDATES that no
    role-aware count can use: `per_status` games each of `ineligible_role`,
    `ineligible_anchor` and `collision`. Physically honest (b200_299 needs ply >=
    229, so red@230 / black@243 -- both 'late')."""
    extra, gi = [], start_gi
    for status in ("ineligible_role", "ineligible_anchor", "collision"):
        for _ in range(per_status):
            extra += [(r, status) for r in _v2_game(gi, "target", "late", "b200_299")]
            gi += 1
    return extra


def _screen_late_target_floor_unmeetable():
    """THE correction-1 fixture: ample late/b200_299 CANDIDATES, too few TARGETS.

    The (target, late) cell is amply supplied for its 45-row quota (85 games / 170
    rows) and its b300_399 floor (20 games / 40 rows), but holds only FIVE
    b200_299 target games -- 10 rows against the >= 12 floor. Meanwhile the SAME
    screen carries 40 further late/b200_299 games (10 kept `control` from the pool
    spec + 30 excluded as ineligible_role/ineligible_anchor/collision), so the
    ROLE-AGNOSTIC geometry is comfortably fine: the Task-4 preflight passes on it.
    Only a ROLE-AWARE, post-screen check can see the floor is unmeetable.
    """
    pool = _v2_pool({("target", "late"): [("b400_plus", 60), ("b300_399", 20),
                                          ("b200_299", 5)]})
    return _screen_from_pool(pool, extra=_extra_late_b200_299_candidates(90_000))


def _screen_target_role_starved():
    """The ROLE half of the same correction: ample midgame CANDIDATES, too few
    TARGETS. (target, midgame) is supplied by only 20 kept games (40 rows) against
    its 45-row demand, while 40 further midgame games are screened as
    ineligible_role / ineligible_anchor -- so the geometry is fine and only the
    role-aware capacity check can refuse."""
    pool = _v2_pool({("target", "midgame"): [("b400_plus", 20)]})
    extra, gi = [], 80_000
    for status in ("ineligible_role", "ineligible_anchor"):
        for _ in range(20):
            extra += [(r, status)
                      for r in _v2_game(gi, "target", "midgame", "b400_plus")]
            gi += 1
    return _screen_from_pool(pool, extra=extra)


def _v2_screen_artifact(tmp_path, screen_rows, *,
                        forbidden_hashes=("not-a-real-dev-corpus-hash",)):
    """A COMPLETE, self-consistent screen artifact -- exactly what `run_screen`
    writes -- over REAL files on disk, plus the loaded `V2Config`.

    Writes the reservoir replays + index, TWO DISTINCT reservoir checkpoints (`a`
    doubling as the anchor, exactly like the real frozen protocol: "anchor:
    checkpoint_a"), a protocol JSON naming them, a match-summary file, a forbidden
    manifest, the config (PRE-REGISTERING the nine fingerprints it can know), and
    the screen CSV ITSELF; then derives the `.meta.json` through the REAL
    `write_screen_meta` and reads it back. So the fixture's provenance -- including
    `screen_csv_sha1` -- and its `n_proposals` / `status_counts` are the genuine
    values for these very rows, never hand-typed, and a test that tampers with
    anything is tampering with a real artifact.

    Ordering is load-bearing and NOT circular: the nine pre-registered fingerprints
    are independent of BOTH the config (a file cannot contain its own hash) and the
    screen (which does not exist when the config is authored), so they are computed
    FIRST, written INTO the config, and only THEN are the config's and the screen's
    own hashes taken.

    The IDENTITY axis and the ROW axis of `select` remain independent -- no identity
    reads a row's contents, and no row reads a file hash -- so each test below can
    perturb exactly one. `forbidden_hashes` is a knob so a test can make the config's
    OWN forbidden manifest hold a hash the manifest will select (the only honest way
    to exercise `assert_disjoint` now that `forbidden` is bound to those files).
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    replays = []
    for gi in range(3):
        p = tmp_path / f"replay_{gi}.json"
        p.write_text(json.dumps(_honest_replay(gi, 330)))
        replays.append(p)
    index = tmp_path / "reservoir_index.jsonl"
    index.write_text("".join(
        json.dumps({"game_idx": gi, "n_moves": 330, "winner": "red",
                    "replay_path": str(p)}) + "\n"
        for gi, p in enumerate(replays)))
    # TWO DISTINCT checkpoint files (Task B10): `checkpoint` doubles as BOTH the
    # protocol's `checkpoint_a` AND the screen anchor (`config.checkpoint`) --
    # the real pinned protocol's own shape ("anchor: checkpoint_a") -- while
    # `checkpoint_b` is a genuinely DIFFERENT file, so `reservoir_checkpoint_a_
    # identity` / `reservoir_checkpoint_b_identity` / `anchor_checkpoint_identity`
    # are never accidentally aliased to the same value by construction.
    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"fake-checkpoint-a-bytes-never-loaded-by-select")
    checkpoint_b = tmp_path / "checkpoint_b.npz"
    checkpoint_b.write_bytes(b"fake-checkpoint-b-bytes-different-never-loaded")
    protocol_path = tmp_path / "reservoir_protocol.json"
    protocol_path.write_text(json.dumps({
        "checkpoint_a": {"path": str(checkpoint)},
        "checkpoint_b": {"path": str(checkpoint_b)},
    }))
    match_summary_path = tmp_path / "match_summary.json"
    match_summary_path.write_text(json.dumps({"a_win_rate": 0.5, "b_win_rate": 0.5}))
    forbidden = tmp_path / "forbidden_a.csv"
    forbidden.write_text("canonical_position_sha1\n"
                         + "".join(f"{h}\n" for h in forbidden_hashes))
    paths = dict(source_index_path=str(index), protocol_path=str(protocol_path),
                 match_summary_path=str(match_summary_path), checkpoint=str(checkpoint),
                 forbidden_manifests=[str(forbidden)])

    # (A) the nine PRE-REGISTERED fingerprints, from the SAME provenance helper the
    # screen stage itself uses -- so a config can never pre-register a value the
    # screen would compute differently.
    prov = v2_screen_provenance(config_path=None, screen_csv=None,
                                base_mcts_config=None, **paths)
    expected = {k: prov[k] for k in PREREGISTERED_IDENTITY_KEYS}

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_v2_config_fixture(
        tmp_path, expected_fingerprints=expected, **paths)))
    config = load_v2_config(str(config_path))

    # The screen artifact itself, through the REAL writers. CSV FIRST, so the meta
    # fingerprints the bytes it just produced -- exactly the order `run_screen` uses.
    write_screen_csv(screen_rows, config.screen_out)
    write_screen_meta(config.screen_out, {
        "config_path": str(config_path),
        "screen_csv": config.screen_out,
        "n_proposals": len(screen_rows),
        "status_counts": dict(Counter(r["exclusion_status"] for r in screen_rows)),
        "row_counts": screen_row_counts(screen_rows),
        "base_mcts_config": {"n_simulations": ANCHOR_SIMS_V2},
        **paths,
    })
    meta = json.loads(Path(config.screen_out + ".meta.json").read_text())
    files = dict(paths, config_path=str(config_path), replays=replays,
                 screen_csv=config.screen_out, checkpoint_b=str(checkpoint_b))
    return config, meta, files


def _v2_faithful_screen_artifact(tmp_path, screen_rows, *, anchor="checkpoint_a"):
    """A screen artifact whose config is the GENUINE `derive_config` output over a
    FULL protocol + a truly-measurable on-disk reservoir -- so `select`'s design Sec
    5 re-derive+byte-compare (`measure_reservoir` -> `derive_config` ->
    `canonical_json_bytes` byte-compare) runs FOR REAL, not stubbed.

    `anchor` (`"checkpoint_a"` default, mirroring the real frozen protocol; or
    `"checkpoint_b"`, a B7-supported path) selects WHICH reservoir player is the
    single screen anchor -- so a test can make the anchor file DISTINCT from
    `checkpoint_a` and prove `reservoir_checkpoint_a_identity` /
    `anchor_checkpoint_identity` are genuinely independent, not aliased.

    Distinct from `_v2_screen_artifact` (which writes a MINIMAL protocol -- checkpoint
    paths only -- and pre-registers the config's fingerprints directly from
    `v2_screen_provenance`, so `measure_reservoir` cannot read it and the identity
    tests stub the re-derive). Here the protocol carries EVERY field `measure_
    reservoir`/`derive_config` read, is written CANONICALLY (so `protocol_sha1` as a
    whole-file hash equals `derive_config`'s canonical-bytes hash), and the config IS
    `derive_config(protocol, measure_reservoir(protocol))` -- so both (a) the eleven
    identities still hard-match [A (`derive_config`) == B (`write_screen_meta`) == C
    (`validate_screen_identities`'s recompute); `_V2_CORPUS_SOURCES` IS
    `fpu_dev_reservoir_protocol.QUALIFICATION_SOURCE_FILES` so `source_file_sha1s`
    agrees, and `load_game_index`/`_load_jsonl_rows` yield the same replay set so
    `replay_data_sha1` agrees] AND (b) select's re-derive byte-compare PASSES on the
    faithful config and FAILS the instant any config field is tampered.

    Heavier than `_v2_screen_artifact` (measure_reservoir reads the 13 generation
    modules + the qualification sources + runs `git`), so used only by the few tests
    that must drive the REAL re-derive. Returns `(config, meta, files)`."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    replays = []
    for gi in range(3):
        p = tmp_path / f"replay_{gi}.json"
        p.write_text(json.dumps(_honest_replay(gi, 330)))
        replays.append(p)
    index = tmp_path / "reservoir_index.jsonl"
    index.write_text("".join(
        json.dumps({"game_idx": gi, "n_moves": 330, "winner": "red",
                    "replay_path": str(p)}) + "\n"
        for gi, p in enumerate(replays)))
    checkpoint_a = tmp_path / "checkpoint.npz"
    checkpoint_a.write_bytes(b"fake-checkpoint-a-bytes-never-loaded-by-select")
    checkpoint_b = tmp_path / "checkpoint_b.npz"
    checkpoint_b.write_bytes(b"fake-checkpoint-b-bytes-different-never-loaded")
    match_summary = tmp_path / "match_summary.json"
    match_summary.write_text(json.dumps({"a_win_rate": 0.5, "b_win_rate": 0.5}))
    forbidden = tmp_path / "forbidden_a.csv"
    forbidden.write_text("canonical_position_sha1\nnot-a-real-dev-corpus-hash\n")
    protocol_path = tmp_path / "reservoir_protocol.json"

    # A FULL protocol -- every field `measure_reservoir`/`derive_config` read (the
    # same shape as `_minimal_protocol_and_measurements`'s protocol, pointed at the
    # real files above). Written CANONICALLY so `file_sha1(protocol)` (what
    # `v2_screen_provenance` computes for `protocol_sha1`) equals
    # `sha1(canonical_json_bytes(protocol))` (what `derive_config` computes).
    protocol = {
        "anchor": anchor,
        "base_seed": 20270000,
        "games": 3,
        "source_index_path": str(index),
        "selection_seed": 20260712,
        "phase_allocation": {f"{role}|{phase}": alloc
                             for (role, phase), alloc in SPLIT_ALLOC_V2.items()},
        "late_floors": dict(LATE_TARGET_FLOORS),
        "enumerator_params": {"min_ply_gap": MIN_PLY_GAP,
                              "max_per_cell_per_game": MAX_PER_CELL_PER_GAME},
        "new_collapse_stratum": "ply_bucket",
        "checkpoint_a": {"path": str(checkpoint_a), "identity": "ckpt_a"},
        "checkpoint_b": {"path": str(checkpoint_b), "identity": "ckpt_b"},
        "forbidden_manifests": [str(forbidden)],
        "screen_out": str(tmp_path / "fpu_dev_source_screen.csv"),
        "select_out": str(tmp_path / "fpu_dev_corpus_v2_manifest.csv"),
        "mcts_eval_batch_size": 14,
        "mcts_stall_flush_sims": 48,
        "config_schema_version": 1,
        "match_summary_path": str(match_summary),
        "replay_dir": str(tmp_path / "replays"),
        "report_out": str(tmp_path / "qualify_report.json"),
    }
    protocol_path.write_bytes(canonical_json_bytes(protocol))

    measurements = measure_reservoir(protocol)
    derived = derive_config(
        protocol, measurements, protocol_path=str(protocol_path))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(derived))
    config = load_v2_config(str(config_path))

    write_screen_csv(screen_rows, config.screen_out)
    write_screen_meta(config.screen_out, {
        "config_path": str(config_path),
        "source_index_path": str(index),
        "protocol_path": str(protocol_path),
        "match_summary_path": str(match_summary),
        "checkpoint": config.checkpoint,
        "forbidden_manifests": list(config.forbidden_manifests),
        "screen_csv": config.screen_out,
        "n_proposals": len(screen_rows),
        "status_counts": dict(Counter(r["exclusion_status"] for r in screen_rows)),
        "row_counts": screen_row_counts(screen_rows),
        "base_mcts_config": {"n_simulations": ANCHOR_SIMS_V2},
    })
    meta = json.loads(Path(config.screen_out + ".meta.json").read_text())
    files = dict(config_path=str(config_path), protocol_path=str(protocol_path),
                 match_summary_path=str(match_summary), screen_csv=config.screen_out,
                 checkpoint=str(checkpoint_a), checkpoint_a=str(checkpoint_a),
                 checkpoint_b=str(checkpoint_b),
                 source_index_path=str(index), replays=replays)
    return config, meta, files


def _select(meta, config, *, forbidden=None, screen_csv_path=None,
            verify_config_rederivation=(lambda _config: None)):
    """`select_final_manifest`, wired the ONE way it accepts: `forbidden` loaded from
    the very manifests whose bytes the identity check hard-matches, and the screen
    artifact's own path (whose bytes are the eleventh identity, and from which `select`
    READS its own rows -- there is no row argument to pass). A test that means to BREAK
    one of those wirings passes it explicitly.

    The design Sec 5 re-derive+byte-compare (`verify_config_rederivation`) is STUBBED
    to a no-op here: `_v2_screen_artifact` (the fixture the identity/forgery/
    composition tests below use) writes a MINIMAL protocol -- checkpoint paths only --
    which the REAL re-derive's `measure_reservoir` cannot read (it needs a full
    protocol). Those tests are about the identity chain / forgeries / composition, NOT
    the re-derive, so they stub it. The REAL re-derive is exercised END TO END, over a
    genuinely measurable reservoir + a truly-derived config, by the
    `_v2_faithful_screen_artifact` tests further down (positive pass + tampered-field
    refusal), and its call ORDER by `test_v2_select_call_order...` +
    `test_v2_select_runs_the_config_rederivation_before_reading_rows`."""
    return select_final_manifest(
        meta, config,
        forbidden=(load_forbidden_hashes(config.forbidden_manifests)
                   if forbidden is None else forbidden),
        screen_csv_path=(config.screen_out if screen_csv_path is None
                         else screen_csv_path),
        verify_config_rederivation=verify_config_rederivation)


def _forge_screen_csv(csv_path, predicate, replacements, *, n):
    """Hand-edit exactly `n` matching rows of a PERSISTED screen CSV -- the forger's
    tool: a text editor. Returns nothing; asserts the forgery actually landed, so a
    fixture that silently matches zero rows can never make an attack test vacuous."""
    lines = Path(csv_path).read_text().splitlines(keepends=True)
    forged = 0
    for i, line in enumerate(lines):
        if forged == n:
            break
        if predicate(line):
            for old, new in replacements:
                line = line.replace(old, new)
            lines[i] = line
            forged += 1
    assert forged == n, f"the forgery fixture matched {forged} rows, expected {n}"
    Path(csv_path).write_text("".join(lines))


def _restamp_screen_csv_hash(meta, csv_path):
    """What a forger does after editing the artifact: re-record its hash in the meta.
    Isolates the SECOND lock (`row_counts`) so it can be proven independently of the
    first (`screen_csv_sha1`)."""
    meta["provenance"]["screen_csv_sha1"] = v2_screen_provenance(
        config_path=None, source_index_path=None, protocol_path=None,
        match_summary_path=None, checkpoint=None,
        forbidden_manifests=[], base_mcts_config=None,
        screen_csv=str(csv_path))["screen_csv_sha1"]


def _validate(meta, config, *, forbidden_paths=None, screen_csv_path=None):
    """`validate_screen_identities`, wired the same one way (see `_select`)."""
    return validate_screen_identities(
        meta, config,
        forbidden_paths=(config.forbidden_manifests if forbidden_paths is None
                         else forbidden_paths),
        screen_csv_path=(config.screen_out if screen_csv_path is None
                         else screen_csv_path))


def _restamp_config_hash(meta, config_path):
    """Re-record the screen meta's `config_sha1` for a config whose BYTES a test just
    changed. Necessary to ISOLATE another identity: `config_sha1` is checked first, so
    without this the self-referential check would fire and the test would pass for the
    wrong reason."""
    meta["provenance"]["config_sha1"] = v2_screen_provenance(
        config_path=config_path, source_index_path=None, protocol_path=None,
        match_summary_path=None, checkpoint=None,
        forbidden_manifests=[], base_mcts_config=None)["config_sha1"]


# --- the required config: every key the plan names is REQUIRED ---------------

def test_v2_config_required_keys_are_exactly_the_plan_s_list():
    """The config REQUIRES every key the plan names -- source reservoir + seed
    range, selection seed, allocation, floors, enumerator params,
    `new_collapse_stratum`, expected fingerprints -- plus the checkpoint /
    forbidden manifests / output paths the two stages consume, PLUS (Task B8,
    spec Sec 2.2) the five new top-level paths a qualified config must also
    carry (`config_schema_version`, `protocol_path`, `match_summary_path`,
    `replay_dir`, `report_out`). Pinned as a SET so a later key can be added
    but none silently dropped."""
    assert set(_V2_CONFIG_REQUIRED_KEYS) == {
        "source_index_path", "seed_range", "selection_seed", "phase_allocation",
        "late_floors", "enumerator_params", "new_collapse_stratum", "checkpoint",
        "forbidden_manifests", "screen_out", "select_out", "expected_fingerprints",
        "config_schema_version", "protocol_path", "match_summary_path",
        "replay_dir", "report_out",
    }


@pytest.mark.parametrize("key", sorted(_V2_CONFIG_REQUIRED_KEYS))
def test_v2_load_config_raises_on_each_missing_required_key(tmp_path, key):
    """EVERY required key, dropped one at a time, makes `load_v2_config` raise
    NAMING it -- never a silent default ("no default source, no default stride")."""
    raw = _v2_config_fixture(tmp_path)
    del raw[key]
    p = tmp_path / "config.json"
    p.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match=key):
        load_v2_config(str(p))


# --- validate_screen_identities: the ELEVEN identities, hard-matched ----------

def test_v2_the_eleven_identities_are_frozen():
    """The frozen identity list (design Sec 1.8; extended Task B10, spec Sec
    2.2/Sec 5): TEN that fingerprint the screen's INPUTS -- config, protocol,
    match summary, source index, replay data, THREE checkpoint identities
    (reservoir A, reservoir B, anchor), source files, forbidden manifests --
    plus `screen_csv_sha1`, which fingerprints the screen's OUTPUT (the
    artifact `select` re-derives the manifest FROM; without it the screen's
    own rows are the one unfingerprinted link). Pinned as the EXACT declared
    ORDER too (not just membership), since that order is what determines
    which identity a caller sees named first when more than one disagrees.

    TWO of them cannot be pre-registered by a config and are matched
    screen-recorded(B) vs fresh-recompute(C): `config_sha1` (a file cannot contain
    its own hash) and `screen_csv_sha1` (the screen does not exist when the config
    is authored)."""
    assert SCREEN_IDENTITY_KEYS == (
        "config_sha1", "protocol_sha1", "match_summary_sha1", "source_index_sha1",
        "replay_data_sha1", "reservoir_checkpoint_a_identity",
        "reservoir_checkpoint_b_identity", "anchor_checkpoint_identity",
        "source_file_sha1s", "forbidden_manifest_sha1s", "screen_csv_sha1")
    assert len(SCREEN_IDENTITY_KEYS) == 11
    assert SELF_REFERENTIAL_IDENTITY == "config_sha1"
    assert SCREEN_ARTIFACT_IDENTITY == "screen_csv_sha1"
    assert set(UNPREREGISTERABLE_IDENTITIES) == {"config_sha1", "screen_csv_sha1"}
    assert set(PREREGISTERED_IDENTITY_KEYS) == (
        set(SCREEN_IDENTITY_KEYS) - set(UNPREREGISTERABLE_IDENTITIES))
    assert len(PREREGISTERED_IDENTITY_KEYS) == 9
    # Every REMEDIATION-bearing identity has a non-empty message (Task B10
    # brief: "each of the 11 has a REMEDIATION message") -- the module-level
    # `assert set(_IDENTITY_REMEDIATION) == set(SCREEN_IDENTITY_KEYS)` already
    # guards this at IMPORT time; re-asserted here so a failure is reported as
    # an ordinary test rather than a collection-time error.
    from scripts.GPU.alphazero.fpu_dev_corpus_v2 import _IDENTITY_REMEDIATION
    assert set(_IDENTITY_REMEDIATION) == set(SCREEN_IDENTITY_KEYS)
    assert all(isinstance(msg, str) and msg for msg in _IDENTITY_REMEDIATION.values())


def test_v2_identities_pass_when_all_three_sources_agree(tmp_path):
    """The happy path: config-expected (A) == screen-recorded (B) == fresh
    recompute (C) for every identity -> no raise, and the VERIFIED recompute is
    RETURNED so the caller can record what was proven without re-hashing the
    reservoir a second time."""
    config, meta, _files = _v2_screen_artifact(
        tmp_path, _screen_from_pool(_abundant_pool_v2()))

    verified = _validate(meta, config)

    assert set(SCREEN_IDENTITY_KEYS) <= set(verified)
    for key in SCREEN_IDENTITY_KEYS:
        assert verified[key] == meta["provenance"][key]      # (C) == (B), verbatim


@pytest.mark.parametrize("key", sorted(SCREEN_IDENTITY_KEYS))
def test_v2_any_of_the_eleven_identities_differing_raises_naming_it(tmp_path, key):
    """ANY of the eleven recorded identities differing from the recompute is a HARD
    STOP naming that identity -- proven INDEPENDENTLY, one identity at a time, for
    EVERY one of the eleven (Task B10 brief: "a per-identity raise test for every
    one of the eleven at select time") -- and the message says what to DO about it."""
    config, meta, _files = _v2_screen_artifact(
        tmp_path, _screen_from_pool(_abundant_pool_v2()))
    tampered = copy.deepcopy(meta)
    tampered["provenance"][key] = "tampered-identity-value"

    with pytest.raises(ValueError, match=key) as excinfo:
        _validate(tampered, config)
    assert "->" in str(excinfo.value)                     # the remediation guidance


def test_v2_three_checkpoint_identities_are_distinct_and_independently_checked(
        tmp_path):
    """Task B10: "the three checkpoint identities are distinct and each
    independently checked (tamper reservoir_a but not b -> refuses naming a)".

    The fixture's `checkpoint_a`/anchor and `checkpoint_b` are genuinely
    DIFFERENT files, so first prove the HAPPY-PATH values really are distinct
    (not merely differently-named copies of the same string) -- then tamper
    EACH reservoir checkpoint file INDEPENDENTLY and show each refusal names
    only the checkpoint that actually changed."""
    config, meta, files = _v2_screen_artifact(
        tmp_path, _screen_from_pool(_abundant_pool_v2()))
    fp = config.expected_fingerprints
    assert fp["reservoir_checkpoint_a_identity"] != fp["reservoir_checkpoint_b_identity"]
    # the fixture's anchor IS checkpoint_a (mirrors the real frozen protocol:
    # "anchor: checkpoint_a") -- so these two are EQUAL, by construction, not
    # independently wrong.
    assert fp["anchor_checkpoint_identity"] == fp["reservoir_checkpoint_a_identity"]
    _validate(meta, config)                                  # clean: passes

    # Tamper ONLY reservoir B's file on disk -- A and the anchor (the SAME file
    # as A here) are untouched, so ONLY `reservoir_checkpoint_b_identity` may
    # legitimately be named (SCREEN_IDENTITY_KEYS checks `reservoir_checkpoint_a_
    # identity` BEFORE `..._b_identity`, so if `a` were somehow also affected,
    # the raise would name `a` instead -- this assertion would then fail loudly
    # rather than silently passing for the wrong reason).
    ckpt_b_path = Path(files["checkpoint_b"])
    ckpt_b_path.write_bytes(ckpt_b_path.read_bytes() + b"-tampered")
    with pytest.raises(ValueError, match="reservoir_checkpoint_b_identity") as excinfo:
        _validate(meta, config)
    assert "reservoir_checkpoint_a_identity" not in str(excinfo.value)
    assert "anchor_checkpoint_identity" not in str(excinfo.value)


def test_v2_reservoir_a_and_anchor_checkpoint_identities_are_independent_at_select(
        tmp_path):
    """B10 review gap: `reservoir_checkpoint_a_identity` vs `anchor_checkpoint_
    identity` -- the checkpoint-split pairing that matters -- proven GENUINELY
    INDEPENDENT, not aliased. EVERY other fixture sets the anchor to the SAME file
    as `checkpoint_a` (the real `anchor: "checkpoint_a"`), so `a` and `anchor` share
    a value and no tamper can tell them apart. Here the faithful fixture uses
    `anchor="checkpoint_b"` (a B7-supported path -- `derive_config` computes the
    anchor identity from `protocol[anchor]["path"]`, NOT aliased to reservoir_a), so
    the anchor file is checkpoint_b, DISTINCT from checkpoint_a.

    Tampering ONLY checkpoint_a's file -> `select` refuses (at the eleven-identity
    hard-match, before any selection) naming `reservoir_checkpoint_a_identity` and
    NOT `anchor_checkpoint_identity` (= checkpoint_b, untouched) NOR
    `reservoir_checkpoint_b_identity` (also checkpoint_b, untouched). Run through the
    REAL `select_final_manifest` (no injected verifier) -- the operator-facing path.
    """
    screen = _screen_from_pool(_abundant_pool_v2())
    config, meta, files = _v2_faithful_screen_artifact(
        tmp_path, screen, anchor="checkpoint_b")

    # HAPPY-PATH: the anchor identity EQUALS reservoir_b (same file, anchor=b) but
    # DIFFERS from reservoir_a -- the exact split the tamper below distinguishes.
    fp = config.expected_fingerprints
    assert fp["anchor_checkpoint_identity"] == fp["reservoir_checkpoint_b_identity"]
    assert fp["anchor_checkpoint_identity"] != fp["reservoir_checkpoint_a_identity"]
    assert fp["reservoir_checkpoint_a_identity"] != fp["reservoir_checkpoint_b_identity"]

    ckpt_a_path = Path(files["checkpoint_a"])
    ckpt_a_path.write_bytes(ckpt_a_path.read_bytes() + b"-tampered")

    with pytest.raises(ValueError, match="reservoir_checkpoint_a_identity") as excinfo:
        select_final_manifest(
            meta, config,
            forbidden=load_forbidden_hashes(config.forbidden_manifests),
            screen_csv_path=config.screen_out)
    msg = str(excinfo.value)
    assert "anchor_checkpoint_identity" not in msg      # the anchor (=b) is untouched
    assert "reservoir_checkpoint_b_identity" not in msg  # reservoir_b (=b) is untouched


def test_v2_identity_catches_a_protocol_mutated_after_screening(tmp_path):
    """`protocol_sha1` covers the frozen `reservoir_protocol.json` ITSELF: the
    config (A) and the screen meta (B) agree perfectly -- so an (A)-vs-(B)-only
    check passes -- yet the protocol file was MUTATED ON DISK (a harmless-looking
    added key that does not touch `checkpoint_a`/`checkpoint_b`, so the checkpoint
    identities stay correct and only `protocol_sha1` can catch it). Only the fresh
    recompute (C), re-reading the protocol file, can see it."""
    config, meta, files = _v2_screen_artifact(
        tmp_path, _screen_from_pool(_abundant_pool_v2()))
    assert (config.expected_fingerprints["protocol_sha1"]
            == meta["provenance"]["protocol_sha1"])            # (A) == (B): agreed

    protocol = json.loads(Path(files["protocol_path"]).read_text())
    protocol["an_added_field_never_read_by_anyone"] = True      # mutated!
    Path(files["protocol_path"]).write_text(json.dumps(protocol))

    with pytest.raises(ValueError, match="protocol_sha1"):
        _validate(meta, config)


def test_v2_identity_catches_a_match_summary_mutated_after_screening(tmp_path):
    """`match_summary_sha1` covers the generated reservoir's match-summary file:
    (A) and (B) agree, yet the summary was MUTATED ON DISK after screening. Only
    the fresh recompute (C) can see it, and it must hard-stop on
    `match_summary_sha1`."""
    config, meta, files = _v2_screen_artifact(
        tmp_path, _screen_from_pool(_abundant_pool_v2()))
    assert (config.expected_fingerprints["match_summary_sha1"]
            == meta["provenance"]["match_summary_sha1"])        # (A) == (B): agreed

    summary = json.loads(Path(files["match_summary_path"]).read_text())
    summary["a_win_rate"] = 0.999                                # mutated!
    Path(files["match_summary_path"]).write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="match_summary_sha1"):
        _validate(meta, config)


@pytest.mark.parametrize("key", sorted(PREREGISTERED_IDENTITY_KEYS))
def test_v2_config_pre_registration_is_not_decorative(tmp_path, key):
    """(A) really participates. Screen (B) and recompute (C) agree perfectly, yet a
    config whose PRE-REGISTERED expectation differs still hard-stops -- so the Sec
    1.8 pre-registration is a real gate, not decoration. (A (B)-vs-(C)-only check
    would let a config that expected a DIFFERENT reservoir/checkpoint through.)"""
    config, meta, files = _v2_screen_artifact(
        tmp_path, _screen_from_pool(_abundant_pool_v2()))
    raw = json.loads(Path(files["config_path"]).read_text())
    raw["expected_fingerprints"][key] = "a-pre-registration-that-does-not-match"
    Path(files["config_path"]).write_text(json.dumps(raw))
    config = load_v2_config(files["config_path"])
    _restamp_config_hash(meta, files["config_path"])

    with pytest.raises(ValueError, match=key):
        _validate(meta, config)


def test_v2_identity_catches_a_replay_mutated_after_screening(tmp_path):
    """THE rationale for the fresh recompute (C). The config (A) and the screen
    meta (B) agree perfectly with each other -- so an (A)-vs-(B)-only check passes
    -- yet a replay file was MUTATED ON DISK after the screen was written. Only a
    recompute from the inputs the config NAMES can see it, and it must hard-stop on
    `replay_data_sha1`."""
    config, meta, files = _v2_screen_artifact(
        tmp_path, _screen_from_pool(_abundant_pool_v2()))
    assert (config.expected_fingerprints["replay_data_sha1"]
            == meta["provenance"]["replay_data_sha1"])          # (A) == (B): agreed

    files["replays"][1].write_text(json.dumps(_honest_replay(1, 331)))   # mutated!

    with pytest.raises(ValueError, match="replay_data_sha1"):
        _validate(meta, config)


def test_v2_identity_catches_a_config_rewritten_after_screening(tmp_path):
    """The SELF-REFERENTIAL identity, (B) vs (C): recompute the hash of the config
    file we were HANDED and require the screen to have recorded it -- which is what
    PROVES this screen was produced by THIS config. Rewriting the config's bytes
    after the screen was written must hard-stop."""
    config, meta, files = _v2_screen_artifact(
        tmp_path, _screen_from_pool(_abundant_pool_v2()))
    raw = json.loads(Path(files["config_path"]).read_text())
    raw["eval_batch_size"] = 7                       # a real, if harmless, edit
    Path(files["config_path"]).write_text(json.dumps(raw))
    rewritten = load_v2_config(files["config_path"])

    with pytest.raises(ValueError, match="config_sha1"):
        _validate(meta, rewritten)


def test_v2_identity_catches_a_screen_csv_edited_after_screening(tmp_path):
    """The SCREEN-ARTIFACT identity, (B) vs (C) -- the hole the ten INPUT identities
    left open. Every input is untouched, so all ten still match; only the screen's
    OWN bytes changed. `select` must refuse, naming `screen_csv_sha1`."""
    config, meta, files = _v2_screen_artifact(
        tmp_path, _screen_from_pool(_abundant_pool_v2()))
    _validate(meta, config)                               # clean: passes

    csv_path = Path(files["screen_csv"])
    csv_path.write_text(csv_path.read_text().replace("kept", "collision", 1))

    with pytest.raises(ValueError, match="screen_csv_sha1"):
        _validate(meta, config)


def test_v2_identity_hard_matches_the_forbidden_manifests_actually_used(tmp_path):
    """The forbidden manifests are matched BY CONTENT, not by path: handing
    `select` a DIFFERENT forbidden file than the one `screen` excluded against
    hard-stops, so the hashes fed to `assert_disjoint` are provably the ones the
    screen's collision filter used."""
    config, meta, _files = _v2_screen_artifact(
        tmp_path, _screen_from_pool(_abundant_pool_v2()))
    other = tmp_path / "forbidden_a.csv.other"
    other.write_text("canonical_position_sha1\na-different-forbidden-hash\n")

    with pytest.raises(ValueError, match="forbidden_manifest_sha1s"):
        _validate(meta, config, forbidden_paths=[str(other)])


def test_v2_source_file_mismatch_names_the_module_and_says_what_to_do(tmp_path):
    """The identity likeliest to fire in practice (any edit to the seven frozen
    modules) must not just dump two seven-entry dicts: it names the DIFFERING
    basename(s) and tells the operator what to do."""
    config, meta, _files = _v2_screen_artifact(
        tmp_path, _screen_from_pool(_abundant_pool_v2()))
    tampered = copy.deepcopy(meta)
    tampered["provenance"]["source_file_sha1s"]["mcts.py"] = "as-if-mcts-were-edited"

    with pytest.raises(ValueError, match="source_file_sha1s") as excinfo:
        _validate(tampered, config)

    msg = str(excinfo.value)
    assert "mcts.py" in msg                       # WHICH module
    assert "fpu_state_hash.py" not in msg.split("differing entr")[1]   # only that one
    assert "FROZEN for the life of a screen artifact" in msg
    assert "re-screen" in msg


@pytest.mark.parametrize("key", sorted(PREREGISTERED_IDENTITY_KEYS))
def test_v2_a_missing_pre_registration_raises(tmp_path, key):
    """A config that pre-registers only SOME identities cannot silently skip the
    rest: every one of the nine is REQUIRED in `expected_fingerprints`."""
    config, meta, files = _v2_screen_artifact(
        tmp_path, _screen_from_pool(_abundant_pool_v2()))
    raw = json.loads(Path(files["config_path"]).read_text())
    del raw["expected_fingerprints"][key]
    Path(files["config_path"]).write_text(json.dumps(raw))
    config = load_v2_config(files["config_path"])
    _restamp_config_hash(meta, files["config_path"])

    with pytest.raises(ValueError, match=key):
        _validate(meta, config)


def test_v2_a_screen_meta_without_provenance_raises(tmp_path):
    """A screen meta carrying no `provenance` block -- or one missing an identity
    -- cannot be validated, so it is REFUSED rather than vacuously passed."""
    config, meta, _files = _v2_screen_artifact(
        tmp_path, _screen_from_pool(_abundant_pool_v2()))

    no_block = {k: v for k, v in meta.items() if k != "provenance"}
    with pytest.raises(ValueError, match="provenance"):
        _validate(no_block, config)

    missing_key = copy.deepcopy(meta)
    del missing_key["provenance"]["anchor_checkpoint_identity"]
    with pytest.raises(ValueError, match="anchor_checkpoint_identity"):
        _validate(missing_key, config)


# --- validate_screen_rows_against_meta: the rows must match their own meta ----

def test_v2_the_cross_checked_row_key_covers_what_selection_reads():
    """The composition key is `(exclusion_status, raw_policy_role, phase, band)` --
    every field `post_screen_qualification` and `sample_v2_rows` key on to decide which
    SPLIT_ALLOC_V2 cell a row fills and which LATE_TARGET_FLOORS band it counts toward.

    `exclusion_status` ALONE is not enough, and that is the whole point: see
    `test_v2_screen_role_flip_on_already_kept_rows_raises`."""
    assert SCREEN_ROW_KEY_FIELDS == (
        "exclusion_status", "raw_policy_role", "phase", "band")
    assert set(SCREEN_ROW_KEY_FIELDS) <= set(SCREEN_FIELDNAMES)


def test_v2_screen_rows_matching_their_meta_pass(tmp_path):
    """The happy path: rows read back from the artifact agree with the screen's own
    recorded n_proposals + row_counts -> returns None, silently."""
    screen = _screen_from_pool(_abundant_pool_v2())
    _config, meta, files = _v2_screen_artifact(tmp_path, screen)

    assert validate_screen_rows_against_meta(
        read_screen_csv(files["screen_csv"]), meta) is None


def test_v2_screen_row_count_disagreeing_with_the_meta_raises(tmp_path):
    """Truncation / appended rows: the row COUNT must match `n_proposals`."""
    screen = _screen_from_pool(_abundant_pool_v2())
    _config, meta, _files = _v2_screen_artifact(tmp_path, screen)

    with pytest.raises(ValueError, match="n_proposals"):
        validate_screen_rows_against_meta(screen[:-1], meta)     # one row dropped


def test_v2_screen_status_flip_disagreeing_with_the_meta_raises(tmp_path):
    """One `kept` row edited to `collision`: the row count is identical, but the
    composition histogram no longer matches the meta -- and the raise SAYS which
    composition key drifted, and by how much."""
    screen = _screen_from_pool(_abundant_pool_v2())
    _config, meta, _files = _v2_screen_artifact(tmp_path, screen)

    flipped = copy.deepcopy(screen)
    flipped[0]["exclusion_status"] = "collision"

    with pytest.raises(ValueError, match="ROW COMPOSITION") as excinfo:
        validate_screen_rows_against_meta(flipped, meta)
    assert "kept" in str(excinfo.value) and "collision" in str(excinfo.value)


def test_v2_screen_role_flip_on_already_kept_rows_raises(tmp_path):
    """*** THE HOLE AN `exclusion_status`-ONLY HISTOGRAM LEAVES OPEN. ***

    Flip `raw_policy_role` from `control` to `target` on rows that are ALREADY `kept`.
    The row COUNT is unchanged and the `exclusion_status` histogram is unchanged -- so
    the meta's `n_proposals` and `status_counts` both remain CORRECT AND UNTOUCHED --
    yet an unmeetable >=12 late-TARGET floor becomes satisfiable, which is the one
    thing STAGE 2 exists to prevent.

    Only a histogram over the fields SELECTION actually reads can see it, and it must.
    """
    screen = _screen_from_pool(_abundant_pool_v2())
    _config, meta, _files = _v2_screen_artifact(tmp_path, screen)

    flipped = copy.deepcopy(screen)
    n = 0
    for r in flipped:
        if n < 4 and r["exclusion_status"] == "kept" and r["raw_policy_role"] == "control":
            r["raw_policy_role"] = "target"
            n += 1
    assert n == 4

    # The two OLD attestations are still perfectly honest about these rows...
    assert len(flipped) == meta["n_proposals"]
    assert (dict(Counter(r["exclusion_status"] for r in flipped))
            == meta["status_counts"])
    # ...and the composition histogram still catches it.
    with pytest.raises(ValueError, match="ROW COMPOSITION") as excinfo:
        validate_screen_rows_against_meta(flipped, meta)
    assert "kept|control" in str(excinfo.value)
    assert "kept|target" in str(excinfo.value)


def test_v2_a_meta_that_cannot_cross_check_its_rows_raises(tmp_path):
    """A meta recording no `row_counts` is REFUSED, not vacuously passed -- an
    `exclusion_status`-only attestation cannot prove row COMPOSITION, and a screen
    whose composition cannot be cross-checked is not an evidence artifact."""
    screen = _screen_from_pool(_abundant_pool_v2())
    _config, meta, _files = _v2_screen_artifact(tmp_path, screen)
    blind = {k: v for k, v in meta.items() if k != "row_counts"}

    with pytest.raises(ValueError, match="cannot be cross-checked"):
        validate_screen_rows_against_meta(screen, blind)


# --- post_screen_qualification: STAGE 2 (role + floors) ----------------------

def test_v2_qualification_passes_on_a_qualifying_screen():
    """A screen whose kept rows can satisfy the exact SPLIT_ALLOC_V2 roles AND both
    late-target floors qualifies silently (returns None)."""
    kept = kept_rows_from_screen(_screen_from_pool(_abundant_pool_v2()))

    assert post_screen_qualification(kept) is None


def test_v2_qualification_raises_when_a_late_target_floor_is_unmeetable_though_the_geometry_is_fine():
    """THE load-bearing STAGE-2 test (design Sec 1.7's own worked example).

    The SAME screen, read two ways:
      * ROLE-AGNOSTIC (what the Task-4 geometric preflight sees): late/b200_299
        CANDIDATES are abundant -- 45 games, 90 proposals -- and the preflight,
        run over exactly this geometry, returns feasible=True with a witness.
      * ROLE-AWARE (what only the post-screen `select` can see): just 10 of those
        proposals survive the screen AS TARGETS, below the >= 12 floor.
    So stage 1 passes and stage 2 must REFUSE, naming the band. This is precisely
    the correction the two-stage split exists for, and it is why the preflight's
    pass can never be mistaken for a corpus guarantee.
    """
    screen = _screen_late_target_floor_unmeetable()
    floor = LATE_TARGET_FLOORS["b200_299"]

    # 1. the geometry really is fine -- ample CANDIDATES, and stage 1 says FEASIBLE.
    assert len(_late_candidates(screen, "b200_299")) >= 4 * floor
    report = v2_geometry_feasibility(_proposals_from_screen(screen))
    assert report.feasible, report.binding_constraint
    assert report.witness is not None

    # 2. ...yet too few of those candidates classify as TARGET.
    assert len(_late_target_kept(screen, "b200_299")) < floor

    # 3. stage 2 refuses, naming the floor's band -- BEFORE any selection.
    with pytest.raises(ValueError, match="b200_299"):
        post_screen_qualification(kept_rows_from_screen(screen))


def test_v2_qualification_raises_when_a_role_count_is_unmeetable_though_the_geometry_is_fine():
    """The ROLE half of the same correction: midgame CANDIDATES are abundant (and
    stage 1 passes), but only 40 of them survive as `target` against the (target,
    midgame) cell's 45-row demand -- so stage 2 refuses, NAMING THE CELL.

    Matched on `midgame` (not the stage label, which the FLOOR error carries too),
    so this test can actually tell a role failure from a floor failure."""
    screen = _screen_target_role_starved()
    report = v2_geometry_feasibility(_proposals_from_screen(screen))
    assert report.feasible, report.binding_constraint

    with pytest.raises(ValueError, match="midgame") as excinfo:
        post_screen_qualification(kept_rows_from_screen(screen))
    assert "target" in str(excinfo.value)
    assert "b200_299" not in str(excinfo.value)      # NOT the late-floor failure


def test_v2_qualification_is_necessary_not_sufficient():
    """The honest scope, pinned. Qualification is a NECESSARY-condition check, NOT a
    simulation of the sampler: `_pool_gap_starved_cell_v2` clears every role count
    (8 games x <=2 = 16 >= the 15-row demand) and both floors, so it QUALIFIES --
    and the sampler still (correctly) refuses it, because each of those games'
    rows sit < MIN_PLY_GAP apart and only ONE of them is ever pickable.

    `sample_v2_rows` remains the exact-or-raise authority; qualification passing is
    never a promise that selection will succeed -- only that it is not already
    provably doomed on roles or floors."""
    kept = kept_rows_from_screen(_screen_from_pool(_pool_gap_starved_cell_v2()))

    assert post_screen_qualification(kept) is None            # qualifies...
    with pytest.raises(ValueError, match="shortfall"):        # ...and still refused
        sample_v2_rows(kept, seed=20260712)


# --- select_final_manifest: composition, determinism, refusal ORDER -----------

def test_v2_select_final_manifest_yields_the_exact_v2_composition_and_floors(tmp_path):
    """The end-to-end pure select: exact SPLIT_ALLOC_V2 composition (240 rows, every
    (role, phase, split) cell exact), both late-target floors met, no duplicate
    hash, and every manifest row carrying BOTH `band` AND `ply_bucket` (design Sec
    1.4 -- the diagnostic stratifies by phase while still recording bands)."""
    screen = _screen_from_pool(_abundant_pool_v2())
    config, meta, _files = _v2_screen_artifact(tmp_path, screen)

    rows, stats = _select(meta, config)

    assert len(rows) == CORPUS_SIZE_V2 == 240
    counts = Counter((r["role"], r["ply_bucket"], r["split"]) for r in rows)
    for (role, phase), alloc in SPLIT_ALLOC_V2.items():
        for split, quota in alloc.items():
            assert counts[(role, phase, split)] == quota, (role, phase, split)

    late_target_bands = Counter(
        r["band"] for r in rows if r["role"] == "target" and r["ply_bucket"] == "late")
    for band, floor in LATE_TARGET_FLOORS.items():
        assert late_target_bands[band] >= floor, (band, late_target_bands)

    for r in rows:
        assert set(r) == set(MANIFEST_FIELDNAMES_V2)
        assert r["band"] == r["branching_band"]        # the deliberate v1-name alias
        assert r["ply_bucket"] in PHASES
    hashes = [r["canonical_position_sha1"] for r in rows]
    assert len(set(hashes)) == len(hashes)
    assert stats["n_rows"] == CORPUS_SIZE_V2
    assert stats["selection_seed"] == config.selection_seed
    assert sorted(stats["identities_verified"]) == sorted(SCREEN_IDENTITY_KEYS)
    assert stats["screen_rows_cross_checked"] is True
    # The VERIFIED recompute is handed back so the caller records what was PROVEN
    # rather than re-hashing the whole reservoir a second time.
    assert stats["verified_screen_provenance"]["screen_csv_sha1"] == (
        meta["provenance"]["screen_csv_sha1"])


def test_v2_select_final_manifest_is_deterministic(tmp_path):
    """The screen-cache reproducibility property: the SAME persisted screen + the
    SAME config seed always re-derive a BYTE-identical manifest (and identical
    stats) -- which is what makes `select` re-runnable and reviewable from the
    screen artifact alone (design Sec 1.6)."""
    screen = _screen_from_pool(_abundant_pool_v2())
    config, meta, _files = _v2_screen_artifact(tmp_path, screen)

    rows_a, stats_a = _select(meta, config)
    rows_b, stats_b = _select(meta, config)

    assert rows_a == rows_b
    assert stats_a == stats_b


def test_v2_select_reads_its_rows_from_the_hashed_artifact_not_from_an_argument():
    """*** THE DECOY-ROWS HOLE, closed STRUCTURALLY. ***

    `screen_csv_sha1` hashes the FILE. If `select_final_manifest` also took a
    `screen_rows` argument, it would hash one thing and select from another: an honest
    CSV and an honest meta on disk, both matching perfectly, while the ROWS handed in
    were a decoy -- and rows are never hashed. So the parameter does not exist. The
    rows are READ from `screen_csv_path`, the artifact whose bytes were just
    hard-matched, and can come from nowhere else: the bug class is UNREPRESENTABLE, not
    merely detected.

    (Same principle that binds `forbidden` to `config.forbidden_manifests`: verify --
    or better, structurally prevent -- rather than trust the caller to wire it right.)
    """
    params = list(inspect.signature(select_final_manifest).parameters)
    assert "screen_rows" not in params
    # The design Sec 5 re-derive is an INJECTED dependency (`verify_config_
    # rederivation`, keyword-only, default None -> the real lazy import) -- the
    # anti-decoy property (`"screen_rows" not in params`) is preserved.
    assert params == ["screen_meta", "config", "forbidden", "screen_csv_path",
                      "verify_config_rederivation"]

    body = _function_body(select_final_manifest)
    assert "read_screen_csv(screen_csv_path)" in body
    # ...and the read happens AFTER the hard-match AND after the config re-derive:
    # never parse an artifact you have not first proven is the right one.
    assert body.index("validate_screen_identities(") < body.index("read_screen_csv(")
    assert body.index("verify_config_rederivation(config)") < body.index("read_screen_csv(")
    assert body.index("read_screen_csv(") < body.index("validate_screen_rows_against_meta(")

    # `main` must not read the rows either -- it hands over the PATH, nothing else.
    main_body = _function_body(main)
    assert "screen_csv_path=args.screen" in main_body
    assert "read_screen_csv(" not in main_body


def test_v2_select_refuses_on_failed_identities_BEFORE_any_selection(tmp_path, monkeypatch):
    """ORDER, proven: a bad identity refuses BEFORE selection runs at all.

    The very same screen artifact samples FINE (asserted first, as a positive control),
    so the refusal is attributable to the identity and nothing else -- and with
    `sample_v2_rows` replaced by a landmine, the identity error still surfaces,
    which proves selection was never even attempted.
    """
    from scripts.GPU.alphazero import fpu_dev_corpus_v2 as mod
    screen = _screen_from_pool(_abundant_pool_v2())
    config, meta, _files = _v2_screen_artifact(tmp_path, screen)

    rows, _stats = _select(meta, config)
    assert len(rows) == CORPUS_SIZE_V2            # this artifact WOULD sample fine

    def _landmine(*args, **kwargs):
        raise AssertionError("select must refuse BEFORE any selection")
    monkeypatch.setattr(mod, "sample_v2_rows", _landmine)
    monkeypatch.setattr(mod, "post_screen_qualification", _landmine)

    tampered = copy.deepcopy(meta)
    tampered["provenance"]["source_index_sha1"] = "tampered"
    with pytest.raises(ValueError, match="source_index_sha1"):
        _select(tampered, config)


def test_v2_select_refuses_on_failed_qualification_BEFORE_any_selection(tmp_path, monkeypatch):
    """ORDER, proven: a failed qualification refuses BEFORE selection runs. With
    `sample_v2_rows` replaced by a landmine, the floor error still surfaces -- so
    the sampler was never reached."""
    from scripts.GPU.alphazero import fpu_dev_corpus_v2 as mod
    screen = _screen_late_target_floor_unmeetable()
    config, meta, _files = _v2_screen_artifact(tmp_path, screen)

    def _landmine(*args, **kwargs):
        raise AssertionError("select must refuse BEFORE any selection")
    monkeypatch.setattr(mod, "sample_v2_rows", _landmine)

    with pytest.raises(ValueError, match="b200_299"):
        _select(meta, config)


def test_v2_select_refuses_a_screen_csv_edited_after_screening(tmp_path, monkeypatch):
    """The artifact identity, end-to-end through `select`: an otherwise-perfect
    invocation whose screen CSV was edited on disk is refused BEFORE any selection."""
    from scripts.GPU.alphazero import fpu_dev_corpus_v2 as mod
    screen = _screen_from_pool(_abundant_pool_v2())
    config, meta, files = _v2_screen_artifact(tmp_path, screen)

    csv_path = Path(files["screen_csv"])
    csv_path.write_text(csv_path.read_text().replace("kept", "collision", 1))

    def _landmine(*args, **kwargs):
        raise AssertionError("select must refuse BEFORE any selection")
    monkeypatch.setattr(mod, "sample_v2_rows", _landmine)

    with pytest.raises(ValueError, match="screen_csv_sha1"):
        _select(meta, config)


def test_v2_select_refuses_the_status_flip_forgery(tmp_path, monkeypatch):
    """*** FORGERY 1 ***: flip EIGHT `ineligible_role` late/b200_299 rows to
    `kept`/`target` in the persisted CSV. No evaluator, no MCTS -- just typing. The row
    count is unchanged and every INPUT file is untouched, so all ten input identities
    still match; an unmeetable >=12 late-TARGET floor becomes satisfiable.

    REFUSED twice over, each layer proven independently:
      1. `screen_csv_sha1` -- the artifact's own bytes changed;
      2. and, after the forger ALSO re-stamps that hash, the ROW-COMPOSITION histogram.
    """
    from scripts.GPU.alphazero import fpu_dev_corpus_v2 as mod
    screen = _screen_late_target_floor_unmeetable()
    config, meta, files = _v2_screen_artifact(tmp_path, screen)
    csv_path = Path(files["screen_csv"])

    # 0. the HONEST screen is refused for the RIGHT reason (the unmeetable floor).
    with pytest.raises(ValueError, match="b200_299"):
        _select(meta, config)

    _forge_screen_csv(
        csv_path,
        lambda ln: (",late,298,b200_299," in ln
                    and ln.rstrip().endswith(",ineligible_role")),
        [(",,False,,,", ",target,True,0.1,True,"), (",ineligible_role", ",kept")],
        n=8)

    # the forgery WORKS on the rows: qualification now (dishonestly) passes...
    attacked = read_screen_csv(str(csv_path))
    assert len(attacked) == len(screen)                        # row count unchanged
    assert len(_late_target_kept(attacked, "b200_299")) >= LATE_TARGET_FLOORS["b200_299"]
    post_screen_qualification(kept_rows_from_screen(attacked))

    def _landmine(*args, **kwargs):
        raise AssertionError("select must refuse BEFORE any selection")
    monkeypatch.setattr(mod, "sample_v2_rows", _landmine)

    # LAYER 1: the artifact's bytes.
    with pytest.raises(ValueError, match="screen_csv_sha1"):
        _select(meta, config)

    # LAYER 2, independently: re-stamp the hash, and the composition histogram fires.
    restamped = copy.deepcopy(meta)
    _restamp_screen_csv_hash(restamped, csv_path)
    with pytest.raises(ValueError, match="ROW COMPOSITION"):
        _select(restamped, config)


def test_v2_select_refuses_the_role_flip_forgery_that_leaves_status_counts_honest(
        tmp_path, monkeypatch):
    """*** FORGERY 2 -- the one an `exclusion_status`-only histogram let through. ***

    Flip `raw_policy_role` from `control` to `target` on rows that are ALREADY `kept`.
    The row count is unchanged AND the `exclusion_status` histogram is unchanged, so
    the meta's `n_proposals` and `status_counts` stay CORRECT AND UNTOUCHED -- the
    forger need only re-stamp `screen_csv_sha1`. Yet the honest 10 late-TARGET
    b200_299 rows become 12+, forging the >=12 floor.

    REFUSED: the ROW-COMPOSITION histogram (which keys on `raw_policy_role`) sees it,
    BEFORE any selection.
    """
    from scripts.GPU.alphazero import fpu_dev_corpus_v2 as mod
    screen = _screen_late_target_floor_unmeetable()
    config, meta, files = _v2_screen_artifact(tmp_path, screen)
    csv_path = Path(files["screen_csv"])

    # 4 kept CONTROL late/b200_299 rows -> target. `_V2_POOL_SPEC` gives (control, late)
    # ten b200_299 games (20 kept rows), so there is ample honest supply to corrupt.
    _forge_screen_csv(
        csv_path,
        lambda ln: (",late,298,b200_299," in ln and ",control,True," in ln
                    and ln.rstrip().endswith(",kept")),
        [(",control,True,", ",target,True,")],
        n=4)

    forged = read_screen_csv(str(csv_path))
    # The OLD attestations remain perfectly honest about these rows...
    assert len(forged) == meta["n_proposals"]
    assert dict(Counter(r["exclusion_status"] for r in forged)) == meta["status_counts"]
    # ...and the floor really is forged: an honest 10 becomes >= the 12 required.
    assert len(_late_target_kept(forged, "b200_299")) >= LATE_TARGET_FLOORS["b200_299"]
    post_screen_qualification(kept_rows_from_screen(forged))     # would have SELECTED

    # The forger re-stamps the ONE meta field that changed -- and is still refused.
    restamped = copy.deepcopy(meta)
    _restamp_screen_csv_hash(restamped, csv_path)

    def _landmine(*args, **kwargs):
        raise AssertionError("select must refuse BEFORE any selection")
    monkeypatch.setattr(mod, "sample_v2_rows", _landmine)

    with pytest.raises(ValueError, match="ROW COMPOSITION") as excinfo:
        _select(restamped, config)
    assert "kept|control|late|b200_299" in str(excinfo.value)
    assert "kept|target|late|b200_299" in str(excinfo.value)


def test_v2_select_binds_forbidden_to_the_manifests_it_hard_matched(tmp_path):
    """`forbidden` is not an opaque caller argument. Passing anything other than
    `load_forbidden_hashes(config.forbidden_manifests)` -- the very files whose bytes
    the identity check just matched -- is REFUSED.

    Without this, `forbidden=set()` silently turns `assert_disjoint` into a no-op
    while `stats["n_forbidden_hashes"] = 0` is written into the manifest's own meta
    AS EVIDENCE: an artifact misrepresenting its own provenance."""
    screen = _screen_from_pool(_abundant_pool_v2())
    config, meta, _files = _v2_screen_artifact(tmp_path, screen)

    _select(meta, config)                            # correctly wired: fine

    with pytest.raises(ValueError, match="forbidden"):
        _select(meta, config, forbidden=set())                    # the no-op set
    with pytest.raises(ValueError, match="forbidden"):
        _select(meta, config, forbidden={"a-hash-from-somewhere-else"})


def test_v2_select_enforces_disjointness_against_forbidden(tmp_path):
    """`assert_disjoint` is a REAL backstop on the COMPLETED manifest (v1's own
    contract: "a raise here means that per-candidate discard had a gap"). A screen
    whose kept rows include a position the config's forbidden manifest names -- a
    collision the screen's own filter should have excluded -- is refused, not
    shipped.

    Honest by construction: the colliding hash is put in the config's REAL forbidden
    manifest BEFORE its fingerprints are taken, so all eleven identities still match
    and only `assert_disjoint` can fire."""
    screen = _screen_from_pool(_abundant_pool_v2())
    config, meta, _files = _v2_screen_artifact(tmp_path / "clean", screen)
    rows, _stats = _select(meta, config)
    collide = rows[0]["canonical_position_sha1"]

    config2, meta2, _f2 = _v2_screen_artifact(
        tmp_path / "collide", screen, forbidden_hashes=(collide,))

    with pytest.raises(ValueError, match="assert_disjoint"):
        _select(meta2, config2)


def test_v2_select_reruns_identically_from_the_persisted_screen_csv(tmp_path):
    """The whole point of the two-artifact workflow: `select` is re-runnable from the
    PERSISTED screen alone -- it now READS every row through `read_screen_csv`, so the
    CSV's own type coercions (int/float/bool/null) are on the ONLY path there is, and
    a lossy one would break selection outright.

    Pinned here: the collision row's nulls survive as nulls (never a fabricated 0.0),
    the tuple-valued `proposal_cell` round-trips as its persisted text without
    disturbing selection, and two runs over the same artifact are identical."""
    screen = _screen_from_pool(
        _abundant_pool_v2(),
        extra=[(r, "collision") for r in _v2_game(70_000, "target", "opening",
                                                  "b400_plus")])
    config, meta, files = _v2_screen_artifact(tmp_path, screen)

    reread = read_screen_csv(files["screen_csv"])
    assert len(reread) == len(screen)
    assert [r["exclusion_status"] for r in reread] == [
        r["exclusion_status"] for r in screen]
    # the collision row's nulls survive as nulls, never as fabricated zeros
    collided = [r for r in reread if r["exclusion_status"] == "collision"]
    assert collided and all(
        r["normalized_entropy"] is None and r["root_value_stm"] is None
        and r["anchor_eligible"] is None and r["anchor_run"] is False
        for r in collided)
    # `proposal_cell` comes back as its persisted TEXT, not a tuple -- `select` never
    # reads it, which is exactly why that is safe.
    assert reread[0]["proposal_cell"] == str(screen[0]["proposal_cell"])

    rows_a, stats_a = _select(meta, config)
    rows_b, stats_b = _select(meta, config)
    assert rows_a == rows_b and stats_a == stats_b
    assert len(rows_a) == CORPUS_SIZE_V2


def test_v2_write_select_csv_round_trips_the_manifest(tmp_path):
    """The select artifact persists in MANIFEST_FIELDNAMES_V2 order, carrying the v1
    column names the diagnostic reads (`position_ply`, `canonical_position_sha1`,
    `branching_band`, `ply_bucket`, `split`, `role`) so a v2 manifest is consumable
    by `diagnose_fpu_policy_mass` unchanged."""
    screen = _screen_from_pool(_abundant_pool_v2())
    config, meta, _files = _v2_screen_artifact(tmp_path, screen)
    rows, _stats = _select(meta, config)
    out_csv = tmp_path / "fpu_dev_corpus_v2_manifest.csv"

    write_select_csv(rows, str(out_csv))

    with open(out_csv, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == MANIFEST_FIELDNAMES_V2
        back = list(reader)
    assert len(back) == CORPUS_SIZE_V2
    assert {r["split"] for r in back} == set(SPLITS)
    assert {r["role"] for r in back} == {"target", "control"}
    assert all(r["branching_band"] == r["band"] for r in back)
    assert all(r["ply_bucket"] in PHASES for r in back)


# --- `select` is PURE by contract -------------------------------------------

def _function_body(fn):
    """A function's source with its DOCSTRING stripped, so an assertion about what
    the code DOES (and in what ORDER) can never be satisfied by prose that merely
    describes it. A function with NO docstring (e.g. `main`) has no prose to strip, so
    its source IS its body."""
    src = inspect.getsource(fn)
    parts = src.split('"""')
    if len(parts) < 3:                             # no docstring: nothing to strip
        return src
    return '"""'.join(parts[2:])                   # def ... """docstring""" body


def test_v2_select_stage_never_touches_the_evaluator():
    """`select` is PURE: its whole call chain -- `select_final_manifest`,
    `validate_screen_identities`, `validate_screen_rows_against_meta`,
    `post_screen_qualification`, `kept_rows_from_screen`, `read_screen_csv` --
    mentions no evaluator / MCTS / checkpoint-loading machinery ANYWHERE (docstrings
    included), so it holds for every code path, not just the ones a test happens to
    drive. The same `inspect.getsource` idiom Tasks 4-5 already use."""
    heavy = ("eval_runner", "from .mcts", "MCTS(", "_teacher_infer",
             "_build_v2_anchor_search_fn", "search_with_root",
             "_default_evaluator_factory")
    for fn in (select_final_manifest, validate_screen_identities,
               validate_screen_rows_against_meta, post_screen_qualification,
               kept_rows_from_screen, read_screen_csv):
        src = inspect.getsource(fn)
        for needle in heavy:
            assert needle not in src, (fn.__name__, needle)


def test_v2_select_call_order_puts_every_refusal_before_any_selection():
    """The refusal ORDER is structural, not incidental: in `select_final_manifest`'s
    executable BODY (docstring stripped -- prose must not be able to satisfy this),
    the identity hard-match, the design Sec 5 config re-derive, the row READ from the
    artifact it just matched, the row/meta cross-check, the forbidden-set binding and
    the qualification ALL precede the sampler, which precedes `assert_disjoint`. So
    every refusal ALWAYS fires before a single row is selected."""
    body = _function_body(select_final_manifest)
    calls = ("validate_screen_identities(", "verify_config_rederivation(config)",
             "read_screen_csv(", "validate_screen_rows_against_meta(",
             "load_forbidden_hashes(", "post_screen_qualification(",
             "sample_v2_rows(", "assert_disjoint(")
    for needle in calls:
        assert needle in body, needle
    order = [body.index(needle) for needle in calls]
    assert order == sorted(order), order


# ---------------------------------------------------------------------------
# Task B10 correction (design Sec 5): `select` ALSO re-derives + byte-compares
# the config -- the "checked twice" guarantee -- so a NON-hashed config field
# (selection_seed, select_out, a floor) tampered BETWEEN screen and select is
# caught at select, not only by B9's pre-GPU `precheck_before_screen`. These
# drive the REAL re-derive (default `verify_config_rederivation`, NOT stubbed)
# over `_v2_faithful_screen_artifact`'s genuinely-measurable reservoir.
# ---------------------------------------------------------------------------

def test_v2_faithful_screen_artifact_selects_cleanly_with_the_real_rederivation(tmp_path):
    """POSITIVE control + the shared-fixture proof: a screen whose config is the
    GENUINE `derive_config` output over a measurable reservoir passes BOTH the
    eleven-identity hard-match AND the design Sec 5 re-derive+byte-compare (run FOR
    REAL -- no `verify_config_rederivation` injected), and selects the exact 240-row
    v2 composition. (If (A) `derive_config` and (C) `v2_screen_provenance` disagreed
    on any identity, or the re-derive byte-compare were wrong, this would fail.)"""
    screen = _screen_from_pool(_abundant_pool_v2())
    config, meta, _files = _v2_faithful_screen_artifact(tmp_path, screen)

    # NO `verify_config_rederivation` -> select lazily imports and runs the REAL
    # `fpu_dev_reservoir_protocol.rederive_and_assert_config_unchanged`.
    rows, stats = select_final_manifest(
        meta, config,
        forbidden=load_forbidden_hashes(config.forbidden_manifests),
        screen_csv_path=config.screen_out)

    assert len(rows) == CORPUS_SIZE_V2 == 240
    assert sorted(stats["identities_verified"]) == sorted(SCREEN_IDENTITY_KEYS)


@pytest.mark.parametrize("field, tamper", [
    ("selection_seed", lambda v: v + 1),
    ("select_out", lambda v: v + ".tampered"),
])
def test_v2_select_re_derives_and_refuses_a_tampered_non_hashed_config_field(
        tmp_path, field, tamper):
    """THE load-bearing case (design Sec 5, the B10 correction). `selection_seed`
    and `select_out` carry NO hash of their own -- they are not in `expected_
    fingerprints` -- so the eleven-identity hard-match (which passes: `config_sha1`
    is the hash of the UNCHANGED config FILE, and the tampered field lives only in
    the in-memory `V2Config`) cannot see an edit to them. ONLY select's re-derive +
    byte-compare catches it, and it must RAISE naming a re-derivation mismatch,
    BEFORE any selection.

    The tamper is an in-memory `dataclasses.replace` (a field edited after the config
    was loaded -- exactly the between-screen-and-select tamper Sec 5's second check
    exists for); `config.config_path`/`protocol_path` still point at the faithful
    on-disk files, so the fresh re-derivation is honest and the byte-compare diff is
    solely the tampered field."""
    screen = _screen_from_pool(_abundant_pool_v2())
    config, meta, _files = _v2_faithful_screen_artifact(tmp_path, screen)
    tampered = dataclasses.replace(config, **{field: tamper(getattr(config, field))})

    with pytest.raises(ValueError, match="re-derivation"):
        select_final_manifest(
            meta, tampered,
            forbidden=load_forbidden_hashes(tampered.forbidden_manifests),
            screen_csv_path=tampered.screen_out)


def test_v2_select_runs_the_config_rederivation_before_reading_rows(tmp_path, monkeypatch):
    """The re-derive is wired BEFORE the row read / qualification / selection (a
    runtime complement to the source-order `test_v2_select_call_order...`): an
    injected verifier that RAISES surfaces its error while `read_screen_csv`,
    `post_screen_qualification` and `sample_v2_rows` are all landmines -- proving
    none of them ran first."""
    from scripts.GPU.alphazero import fpu_dev_corpus_v2 as mod
    screen = _screen_from_pool(_abundant_pool_v2())
    config, meta, _files = _v2_screen_artifact(tmp_path, screen)   # minimal is fine -- we INJECT

    def _landmine(*args, **kwargs):
        raise AssertionError("select reached row-read/selection before the re-derive")
    monkeypatch.setattr(mod, "read_screen_csv", _landmine)
    monkeypatch.setattr(mod, "post_screen_qualification", _landmine)
    monkeypatch.setattr(mod, "sample_v2_rows", _landmine)

    def _raising_verifier(_config):
        raise ValueError("config re-derivation mismatch (injected)")

    with pytest.raises(ValueError, match="re-derivation mismatch"):
        _select(meta, config, verify_config_rederivation=_raising_verifier)


# --- the dedicated preflight-refusal exception (Task-5 review Minor) ----------

def test_v2_preflight_infeasible_is_a_dedicated_exception():
    """A ZERO-COST preflight refusal and a crash after HOURS of real GPU work must
    not be indistinguishable. `run_screen` raises the dedicated
    `V2PreflightInfeasible` at its one preflight site and `main` catches ONLY that
    -- so any OTHER failure propagates as a raw traceback instead of masquerading as
    a cheap "stop-don't-retune" exit 2."""
    assert issubclass(V2PreflightInfeasible, ValueError)

    screen_src = inspect.getsource(run_screen)
    assert "raise V2PreflightInfeasible(" in screen_src
    assert "raise ValueError(" not in screen_src

    main_src = inspect.getsource(main)
    assert "except V2PreflightInfeasible" in main_src
    assert "except ValueError" not in main_src
    assert "except Exception" not in main_src


# --- `main --mode select` (argparse only -- main() itself is never invoked) ---

def test_v2_main_select_branch_is_wired(tmp_path):
    """STATIC wiring of `main`'s select branch (`main` itself is never invoked --
    plan Global Constraints): it reads only the DERIVED `.meta.json`, loads the
    forbidden hashes from the CONFIG (the one wiring `select_final_manifest` accepts),
    hands over the screen's PATH (`select` reads the rows itself, from the artifact it
    hard-matches), and writes `config.select_out` + its meta -- reusing the VERIFIED
    recompute rather than re-hashing the reservoir a second time."""
    src = inspect.getsource(main)
    for needle in ('args.mode == "select"', "meta.json",
                   "load_forbidden_hashes(config.forbidden_manifests)",
                   "select_final_manifest(", "screen_csv_path=args.screen",
                   "write_select_csv(", "write_screen_meta(", "config.select_out",
                   'stats.pop("verified_screen_provenance")', "provenance=verified"):
        assert needle in src, needle


def test_v2_run_screen_attests_to_the_artifact_it_just_wrote():
    """STATIC wiring of `run_screen` (never invoked -- it is the operator/GPU stage):
    it records `screen_csv` in the meta (so `write_screen_meta` hashes the artifact it
    just produced) plus the `n_proposals` / `row_counts` that `select` cross-checks the
    rows against -- and it writes the CSV BEFORE the meta, so the hash covers the bytes
    actually on disk.

    `row_counts` (the COMPOSITION histogram), not `status_counts`, is the load-bearing
    attestation: see `SCREEN_ROW_KEY_FIELDS`."""
    body = _function_body(run_screen)
    for needle in ('"screen_csv": config.screen_out', '"n_proposals": len(rows)',
                   '"row_counts": screen_row_counts(rows)'):
        assert needle in body, needle
    assert body.index("write_screen_csv(") < body.index("write_screen_meta(")


# ---------------------------------------------------------------------------
# Task B9 (pre-op hardening plan, design Sec 5/Sec 6) -- the `run_screen`
# pre-evaluator precheck. `precheck_before_screen` itself (fpu_dev_reservoir_
# protocol.py) is verified DIRECTLY and exhaustively in tests/test_fpu_dev_
# reservoir_protocol.py -- it is pure of GPU, unlike `run_screen`. Here: ONLY
# the STATIC wiring -- that `run_screen` calls it, via a lazy import, BEFORE
# any other work -- and that importing both modules, in EITHER order, never
# cycles. `run_screen` itself is never invoked (plan Global Constraints: it
# is the operator/GPU stage).
# ---------------------------------------------------------------------------

def test_run_screen_calls_precheck_before_screen_before_the_lazy_evaluator_import():
    """Spec Sec 5/Sec 6: an hours-long screen must never start on stale or
    tampered inputs, so `run_screen` calls `precheck_before_screen` BEFORE
    any checkpoint/evaluator work -- in fact before EVERYTHING else in the
    function, including its own pre-existing v1-style geometric preflight
    (`v2_preflight_source`). `run_screen` lazily imports it INSIDE its own
    body (never at this module's top level -- Sec 6's circular-import
    resolution: `fpu_dev_reservoir_protocol` already top-level-imports FROM
    this module, so a top-level import back here would cycle -- see
    `test_fpu_dev_corpus_v2_and_fpu_dev_reservoir_protocol_import_fresh_
    without_cycling` below). Source-order assertion only, docstring
    stripped (`_function_body`) so prose can never satisfy it."""
    body = _function_body(run_screen)
    assert "from .fpu_dev_reservoir_protocol import precheck_before_screen" in body
    assert "precheck_before_screen(config)" in body

    precheck_import_idx = body.index(
        "from .fpu_dev_reservoir_protocol import precheck_before_screen")
    precheck_call_idx = body.index("precheck_before_screen(config)")
    v1_style_preflight_idx = body.index("v2_preflight_source(")
    evaluator_import_idx = body.index(
        "from .build_teacher_calibration_manifest import _teacher_infer")

    assert precheck_import_idx < precheck_call_idx, body
    assert precheck_call_idx < v1_style_preflight_idx, body
    assert precheck_call_idx < evaluator_import_idx, body


def test_fpu_dev_corpus_v2_and_fpu_dev_reservoir_protocol_import_fresh_without_cycling():
    """Verifies the Sec 6 circular-import resolution actually holds, by
    IMPORTING both modules fresh, in a subprocess, in EACH order -- an
    in-process import would be unreliable (a prior test in this same pytest
    session may already have imported one or both, masking a genuine
    cycle). Also confirms `run_screen`'s lazy import genuinely never fires at
    IMPORT time: merely `import`ing `fpu_dev_corpus_v2` (never calling `run_
    screen`) must leave `fpu_dev_reservoir_protocol` out of `sys.modules` --
    the complement to tests/test_fpu_dev_reservoir_protocol.py::
    test_module_imports_only_pure_names_from_fpu_dev_corpus_v2, which proves
    the same one-directional shape from that OTHER module's own top-level
    imports."""
    import subprocess
    import sys

    for script in (
        # fpu_dev_corpus_v2 first, then fpu_dev_reservoir_protocol.
        "import scripts.GPU.alphazero.fpu_dev_corpus_v2 as a\n"
        "import scripts.GPU.alphazero.fpu_dev_reservoir_protocol as b\n"
        "print('ok')\n",
        # fpu_dev_reservoir_protocol first, then fpu_dev_corpus_v2 (reverse
        # order -- proves neither module's TOP LEVEL depends on import
        # order, which is what "no cycle" actually guarantees).
        "import scripts.GPU.alphazero.fpu_dev_reservoir_protocol as b\n"
        "import scripts.GPU.alphazero.fpu_dev_corpus_v2 as a\n"
        "print('ok')\n",
    ):
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True)
        assert out.returncode == 0 and out.stdout.strip() == "ok", (out.returncode, out.stderr)

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; import scripts.GPU.alphazero.fpu_dev_corpus_v2 as m; "
         "print(any('fpu_dev_reservoir_protocol' in k for k in sys.modules))"],
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
