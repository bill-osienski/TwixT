"""Regression test proving the v1 branching-band + <=50%-ply-bucket-cap
corpus design is mathematically IMPOSSIBLE on the project's fixed 24x24
board.

Frozen design ref: docs/superpowers/specs/2026-07-12-fpu-dev-corpus-v2-phase-design.md
  section 0 ("Why v2 (the impossibility that retired v1)").
v2 plan Task 0 ("Impossibility regression test (the WHY, and a guard)").

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
"""
from scripts.GPU.alphazero.build_fpu_dev_corpus import (
    CORPUS_SIZE,
    QUOTA_PER_BAND,
    SPLIT_ALLOC,
    ply_bucket_of,
)
from scripts.GPU.alphazero.game.twixt_state import TwixtState


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
