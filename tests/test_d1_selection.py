"""D1 position selection -- plan 12.1-12.3, the FROZEN rule. NO EXECUTION.

No model is loaded, no JVM started, no seed drawn or registered, no T1j query
issued, no game played. Everything here reads the published L0 record through
D0's digest-verified binding and recomputes deterministic board facts, exactly
as D0 itself does.
"""
import dataclasses
import hashlib
import json

import pytest

from scripts.GPU.alphazero import d0_postmortem as D0
from scripts.GPU.alphazero import d1_selection as SEL
from scripts.GPU.alphazero import fpu_state_hash as FSH
from scripts.GPU.alphazero.game.twixt_state import TwixtState

RECORD = "docs/superpowers/evidence/2026-08-27-t1j-l0-canonical-match/06_l0_match_results.jsonl"
PLAN_JSON = "docs/superpowers/evidence/2026-08-26-t1j-l0-larger-match/01_l0_match_plan.json"


def _state(moves):
    st = TwixtState()
    for m in moves:
        st = st.apply_move(m)
    return st


# ══════════════════ 12.2: the canonical digest, implemented LITERALLY ═════════

def test_the_digest_is_sha256_not_the_existing_sha1_helper():
    """12.2 freezes SHA-256. `fpu_state_hash` offers a SHA-1 over a SUPERSET key
    whose extra fields happen to be constant across this cohort -- so it would
    dedupe identically and still not be the digest the preregistration named."""
    st = _state([(11, 11), (12, 13)])
    d = SEL.canonical_digest(st)
    assert len(d) == 64 and int(d, 16) >= 0
    assert d != FSH.canonical_state_sha1(st)


def test_the_digest_payload_is_exactly_the_three_frozen_fields():
    """Recomputed here from the frozen wording, not from the implementation."""
    st = _state([(11, 11), (12, 13)])
    pegs = sorted((r, c, p) for (r, c), p in st.pegs.items())
    payload = json.dumps((st.to_move, pegs, sorted(st.bridges)), sort_keys=True)
    assert SEL.canonical_digest(st) == hashlib.sha256(payload.encode()).hexdigest()


def test_a_field_outside_the_frozen_three_does_not_enter_the_digest():
    """NEGATIVE CONTROL on the payload. `max_plies_limit` is in the SHA-1
    helper's key and is NOT one of 12.2's three fields, so it must not move the
    digest -- otherwise the payload silently grew."""
    st = _state([(11, 11), (12, 13)])
    other = dataclasses.replace(st, max_plies_limit=99)
    assert other.max_plies_limit != st.max_plies_limit
    assert SEL.canonical_digest(other) == SEL.canonical_digest(st)
    assert FSH.canonical_state_sha1(other) != FSH.canonical_state_sha1(st)


@pytest.mark.parametrize("mutate", ["to_move", "pegs", "bridges"])
def test_each_frozen_field_changes_the_digest(mutate):
    st = _state([(11, 11), (12, 13), (13, 12)])
    if mutate == "to_move":
        other = dataclasses.replace(st, to_move="red" if st.to_move == "black" else "black")
    elif mutate == "pegs":
        other = st.apply_move(next(m for m in st.legal_moves()))
    else:
        other = dataclasses.replace(st, bridges=set())
        assert st.bridges != other.bridges, "the fixture grew no bridge to remove"
    assert SEL.canonical_digest(other) != SEL.canonical_digest(st)


def test_transpositions_collapse_to_one_digest():
    """The digest is a DEDUPLICATION LABEL: two move orders, one position."""
    a = _state([(11, 11), (12, 13), (13, 12), (10, 13)])
    b = _state([(13, 12), (12, 13), (11, 11), (10, 13)])
    assert a.pegs == b.pegs and a.bridges == b.bridges and a.to_move == b.to_move
    assert SEL.canonical_digest(a) == SEL.canonical_digest(b)


# ═══════════════ 12.1: the frozen selection rule, applied to the record ══════

@pytest.fixture(scope="module")
def bound():
    return D0.bind_record(RECORD, PLAN_JSON)


@pytest.fixture(scope="module")
def plies(bound):
    return SEL.discovery_plies(bound)


@pytest.fixture(scope="module")
def selection(bound):
    return SEL.select_all(bound)


def test_the_frozen_column_names_still_exist_in_ply_features():
    """12.1 asks for exactly this check: a frozen name that drifts from the code
    is a preregistration that no longer binds anything."""
    produced = D0.recomputable_columns()
    declared = {s["column"] for s in D0.CANDIDATE_SIGNATURES}
    for sig in SEL.SIGNATURES:
        assert sig["column"] in produced, sig
        assert sig["column"] in declared, sig


def test_selection_never_reaches_the_confirmation_half(plies):
    assert plies, "no discovery plies were produced; every check below is vacuous"
    assert {r["rep"] for r in plies} == set(D0.DISCOVERY_REPS)


def test_every_ply_carries_a_prefix_whose_length_is_its_own_ply(plies):
    for r in plies:
        assert len(r["prefix"]) == r["ply"], r["task_id"]


def test_the_carried_prefix_replays_to_the_recorded_digest(plies):
    """12.7 aborts on a prefix that does not replay to its digest, so the
    selection must not hand one over that cannot."""
    for r in plies[::97]:                       # every 97th: the whole set is ~3k
        assert SEL.canonical_digest(_state(r["prefix"])) == r["digest"], r["task_id"]


def test_the_frozen_counts_are_reproduced(selection):
    """The numbers 12.1 and 12.4 froze, recomputed from the record."""
    got = {(c["signature"], c["role"]): len(c["rows"]) for c in selection["cohorts"]}
    assert got == {("mover_fragmentation", "position"): 101,
                   ("mover_fragmentation", "control"): 60,
                   ("created_threat", "position"): 30,
                   ("created_threat", "control"): 36}
    assert selection["n_positions"] == SEL.N_POSITIONS == 227


def test_the_frozen_cell_counts_are_reproduced(selection):
    cells = {c["signature"]: c["n_cells"] for c in selection["cohorts"]
             if c["role"] == "position"}
    assert cells == {"mover_fragmentation": 36, "created_threat": 12}


def test_every_selected_position_has_our_incumbent_to_move(selection):
    for c in selection["cohorts"]:
        assert c["rows"], c["signature"]
        for r in c["rows"]:
            assert D0.moved_by(r["colour_arm"], r["mover"]) == "ours", r["task_id"]


def test_positions_hold_the_signature_and_controls_negate_it(selection):
    for c in selection["cohorts"]:
        want = c["role"] == "position"
        for r in c["rows"]:
            assert bool(r[c["column"]]) is want, (c["signature"], c["role"], r["ply"])


def test_no_cohort_retains_two_positions_with_one_digest(selection):
    for c in selection["cohorts"]:
        digests = [r["digest"] for r in c["rows"]]
        assert len(set(digests)) == len(digests), c["signature"]


def test_no_cell_exceeds_the_cap_of_three(selection):
    for c in selection["cohorts"]:
        counts: dict = {}
        for r in c["rows"]:
            k = SEL.cell(r)
            counts[k] = counts.get(k, 0) + 1
        assert counts and max(counts.values()) <= SEL.PER_CELL_CAP == 3, c["signature"]


def test_each_cohort_is_in_the_frozen_total_order(selection):
    for c in selection["cohorts"]:
        keys = [(r["task_id"], r["ply"]) for r in c["rows"]]
        assert keys == sorted(keys), c["signature"]


def test_controls_come_only_from_cells_that_hold_a_selected_position(selection):
    """12.3: matched by construction, not by post-hoc pairing."""
    by_sig = {}
    for c in selection["cohorts"]:
        by_sig.setdefault(c["signature"], {})[c["role"]] = c
    for sig, roles in by_sig.items():
        cells = {SEL.cell(r) for r in roles["position"]["rows"]}
        assert cells, sig
        for r in roles["control"]["rows"]:
            assert SEL.cell(r) in cells, (sig, r["task_id"], r["ply"])


# ────────────────────────────── the seed assignment ──────────────────────────

def test_every_position_draws_from_inside_the_reserved_interval(selection):
    lo, hi = SEL.SEED_INTERVAL
    seeds = [r["seed"] for c in selection["cohorts"] for r in c["rows"]]
    assert len(seeds) == 227
    assert all(lo <= s < hi for s in seeds)


def test_the_seed_assignment_is_injective_and_exhausts_the_block(selection):
    seeds = [r["seed"] for c in selection["cohorts"] for r in c["rows"]]
    assert sorted(seeds) == list(range(*SEL.SEED_INTERVAL))


def test_the_seed_assignment_is_deterministic(bound):
    a = SEL.select_all(bound)
    b = SEL.select_all(bound)
    key = lambda s: [(r["task_id"], r["ply"], r["seed"]) for c in s["cohorts"] for r in c["rows"]]
    assert key(a) == key(b)


def test_the_reserved_block_is_absent_from_every_real_registry():
    """RESERVED here, REGISTERED nowhere: 12.5 makes registering part of the
    execution authorization, not of preparation.

    Each registry is asserted NON-EMPTY first. An absence check over an empty
    collection passes vacuously, which would make this test decorative.
    """
    from scripts.GPU.alphazero import e4_screen_reference as REF
    for name in ("ACCOUNTED_SEED_INTERVALS", "EXPOSED_SEED_INTERVALS",
                 "RETIRED_SEED_INTERVALS", "TEST_ONLY_SEED_INTERVALS"):
        assert getattr(REF, name), f"vacuous: {name} is empty"
    for seed in range(*SEL.SEED_INTERVAL):
        assert not any(REF.seed_status(seed).values()), seed


def test_the_frozen_rule_retains_the_same_position_in_two_cohorts(selection):
    """RECORDED, NOT HIDDEN, and PINNED so it cannot drift silently.

    12.1 and 12.3 deduplicate WITHIN a cohort; nothing in 12 deduplicates ACROSS
    them. So a board state can be a `mover_fragmentation` position and a
    `created_threat` control at once, and 227 retained positions cover fewer
    distinct states than that. The frozen counts already contain this -- cohorts
    that shared a digest would total 203, not 227 -- so it is what was frozen,
    not a departure from it.

    The consequence is real and belongs in the record: those states are queried
    TWICE per depth per cohort, i.e. four independent JVMs rather than two, and
    12.7's determinism check compares only within a pair. The extra pair is not
    compared against the first.
    """
    digests = [r["digest"] for r in selection["positions"]]
    assert len(digests) == 227
    assert len(set(digests)) == 203, "the cross-cohort overlap changed"
    seeds_per_state: dict = {}
    for r in selection["positions"]:
        seeds_per_state.setdefault(r["digest"], []).append(r["seed"])
    shared = {d: s for d, s in seeds_per_state.items() if len(s) > 1}
    assert len(shared) == 24
    assert all(len(s) == 2 for s in shared.values())
    # Every duplicate is the SAME ply of the SAME game seen from two cohorts.
    for digest in shared:
        rows = [r for r in selection["positions"] if r["digest"] == digest]
        assert len({(r["task_id"], r["ply"]) for r in rows}) == 1, rows[0]["task_id"]


def test_a_cohort_that_departs_from_the_frozen_table_is_refused(bound, monkeypatch):
    """The reconciliation inside `select_all`, REACHED ALONE.

    The frozen-count test above asserts the counts directly, so it passes
    whether or not `select_all` also checks them -- an injected-defect control
    proved that removing the internal check changed nothing it could see. This
    drives the check itself by declaring an expectation the record cannot meet,
    which is the case where it is the only thing standing between a drifted rule
    and a silently different cohort.
    """
    bad = tuple(dict(s, positions=s["positions"] + 1) for s in SEL.SIGNATURES)
    monkeypatch.setattr(SEL, "SIGNATURES", bad)
    with pytest.raises(SEL.D1SelectionError, match="12.1 froze"):
        SEL.select_all(bound)
