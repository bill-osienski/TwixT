"""The narrow schema-5 dispatch hook in the v2 selector -- plan Task 8 Step 1.

Two obligations, and the second is the one that is easy to lose:

  * raw schema 5 alone produces an EFFECTIVE schema-5 profile and the v18
    role/split vocabulary;
  * declared schemas 2, 3 and 4 stay byte-for-byte what they were, including
    the historical `3/4 -> effective 2` normalization, which rides inside
    `AllocationProfile.fingerprint()` and therefore inside every frozen v16/v17
    artifact. It reads like a bug. It must not be "fixed".

The legacy half is enforced against `tests/golden/v18_v2_selector_pre_edit_basis/`,
captured at the pre-edit module hash 9efd554b.
"""
import collections
import json
from pathlib import Path

import pytest

from scripts.GPU.alphazero import fpu_dev_corpus_v2 as V

BASIS = Path(__file__).parent / "golden" / "v18_v2_selector_pre_edit_basis"
SCHEMAS = (2, 3, 4)


def _basis(name):
    return json.loads((BASIS / name).read_text())


def _canon(obj):
    return json.dumps(obj, sort_keys=True, default=str)


def v18_raw(**over):
    """A fresh, minimal, VALID schema-5 config on every call.

    A factory, not a module constant: `over` and the parameterized mutations
    below edit nested dicts, and a shared constant would leak those edits into
    later cases.
    """
    raw = {
        "config_schema_version": 5,
        # The frozen v18 run identity. Both halves are enforced by a
        # SCHEMA-5-LOCAL rule -- `eval_runner.interpretation_forbidden` is never
        # consulted, because it raises on this deliberately unregistered kind.
        "run_kind": "v18_preflight_sizing",
        "scientific_interpretation_forbidden": True,
        "phase_allocation": {f"{r}|{p}": dict(a)
                             for (r, p), a in V.allocation_for_schema(5).items()},
        "late_floors": {},
        "late_target_band_minima": {},
        "max_per_game": 2,
        "min_ply_gap": 12,
        "side_tol": 2,
        "corpus_size": 40,
    }
    raw.update(over)
    return raw


def test_v18_raw_is_actually_accepted_by_the_parser():
    """Guard the guard: if the factory drifts out of validity, every positive
    case below would pass or fail for the wrong reason."""
    assert V.parse_allocation_profile(v18_raw(), source="x").schema_version == 5


def v18_profile(**over):
    return V.parse_allocation_profile(v18_raw(**over), source="v18-test")


def v18_pool():
    """A feasible schema-5 pool: side-balanced, <=2 rows/game, plies >=12 apart.

    Synthetic on purpose -- the real corpus lives under gitignored logs/, so a
    suite test may not depend on it.
    """
    rows, gi = [], 0
    demand = {("target", "late"): 16, ("identity_witness", "late"): 4,
              ("flip_control", "late"): 4, ("representative", "opening"): 4,
              ("representative", "early_mid"): 4, ("representative", "midgame"): 4,
              ("representative", "late"): 4}
    base = {"opening": 10, "early_mid": 40, "midgame": 70, "late": 100}
    for (role, phase), n in demand.items():
        for k in range(n * 2):                       # twice the demand
            gi += 1
            rows.append({
                "game_idx": gi, "ply": base[phase] + 14 * (k % 3),
                "side": "red" if k % 2 == 0 else "black",
                "role": role, "phase": phase, "band": "b400_plus",
                "canonical_sha1": f"{gi:040d}", "n_legal": 500,
                "exclusion_status": "kept", "raw_policy_role": role,
                "root_value_stm": 0.0, "normalized_entropy": 0.5,
                "top1_prior": 0.2, "top4_mass": 0.6, "top8_mass": 0.8,
                "anchor_run": "x", "anchor_eligible": "true",
                "proposal_cell": f"{role}|{phase}", "ply_bucket": "b1",
            })
    return rows


# --- the historical normalization, PINNED ------------------------------------

def test_declared_three_and_four_normalize_to_effective_schema_two():
    """The load-bearing pin. `parse_allocation_profile` accepts declared 2/3/4
    but builds effective schema 2 for all of them, and that value is carried in
    `fingerprint()` -- so every frozen v16/v17 artifact depends on it."""
    for schema in SCHEMAS:
        raw = _basis(f"schema{schema}.profile.json")
        raw.pop("selection_seed")
        assert raw["config_schema_version"] == schema
        profile = V.parse_allocation_profile(raw, source=f"schema{schema}")
        assert profile.schema_version == 2, (
            f"declared {schema} must normalize to EFFECTIVE 2; this is not a bug")
    assert [V.effective_schema_for(d) for d in (2, 3, 4)] == [2, 2, 2]
    assert V.effective_schema_for(5) == 5


def test_sources_json_records_declared_and_effective_and_they_still_hold():
    recorded = _basis("sources.json")
    assert recorded["schema_version_normalization"]["declared_to_effective"] == {
        "2": 2, "3": 2, "4": 2}
    for schema in SCHEMAS:
        entry = recorded["schemas"][f"schema{schema}"]
        raw = _basis(f"schema{schema}.profile.json")
        raw.pop("selection_seed")
        profile = V.parse_allocation_profile(raw, source="x")
        assert entry["declared_config_schema_version"] == schema
        assert entry["effective_allocation_profile_schema_version"] == profile.schema_version
        assert entry["effective_allocation_profile_fingerprint"] == profile.fingerprint()


# --- legacy byte identity ----------------------------------------------------

@pytest.mark.parametrize("schema", SCHEMAS)
def test_legacy_selector_output_is_byte_identical(schema):
    raw = _basis(f"schema{schema}.profile.json")
    seed = raw.pop("selection_seed")
    profile = V.parse_allocation_profile(raw, source=f"schema{schema}")
    rows = _basis(f"schema{schema}.rows.json")
    expected = _basis(f"schema{schema}.selector_output.json")

    report = V.post_screen_qualification_report([dict(r) for r in rows], profile)
    selected, stats = V.sample_v2_rows([dict(r) for r in rows], seed=seed,
                                       alloc=profile)
    assert _canon(report) == _canon(expected["post_screen_qualification_report"])
    assert _canon({"rows": selected, "stats": stats}) == _canon(
        expected["sample_v2_rows"])


@pytest.mark.parametrize("schema", SCHEMAS)
def test_legacy_stats_carry_no_split_assignment_key(schema):
    """`stats` is serialized into selector artifacts, so a key that does not
    exist today would necessarily change legacy bytes."""
    raw = _basis(f"schema{schema}.profile.json")
    seed = raw.pop("selection_seed")
    profile = V.parse_allocation_profile(raw, source="x")
    rows = _basis(f"schema{schema}.rows.json")
    _sel, stats = V.sample_v2_rows([dict(r) for r in rows], seed=seed, alloc=profile)
    assert "split_assignment" not in stats


@pytest.mark.parametrize("schema", SCHEMAS)
def test_legacy_fingerprint_gains_no_splits_key(schema):
    raw = _basis(f"schema{schema}.profile.json")
    raw.pop("selection_seed")
    profile = V.parse_allocation_profile(raw, source="x")
    assert "splits" not in profile.fingerprint()
    assert profile.splits == V.SPLITS == ("tuning", "frozen_check")


def test_legacy_profile_is_unchanged_and_constants_are_not_mutated():
    legacy = V.AllocationProfile.legacy()
    assert legacy.schema_version == 1
    assert "splits" not in legacy.fingerprint()
    # RESOLVERS, not mutated constants.
    assert V._ROLES == ("target", "control")
    assert V.SPLITS == ("tuning", "frozen_check")
    assert V.SPLIT_ALLOC_V2[("target", "opening")] == {"tuning": 30, "frozen_check": 15}
    assert V.CORPUS_SIZE == 240


# --- the schema-5 dispatch ---------------------------------------------------

@pytest.mark.parametrize("schema", (1, 2, 3, 4))
def test_resolvers_return_the_legacy_vocabulary_below_five(schema):
    assert V.roles_for_schema(schema) == V._ROLES
    assert V.splits_for_schema(schema) == V.SPLITS
    assert V.allocation_for_schema(schema) == {
        c: dict(a) for c, a in V.SPLIT_ALLOC_V2.items()}


def test_resolvers_return_the_v18_vocabulary_at_five():
    assert V.V18_SCHEMA == 5
    assert V.roles_for_schema(5) == (
        "target", "identity_witness", "flip_control", "representative")
    assert V.splits_for_schema(5) == ("all",)
    assert V.allocation_for_schema(5) == {
        c: dict(a) for c, a in V.SPLIT_ALLOC_V18.items()}


def test_schema_five_is_not_four():
    """Reusing 4 would let v18 semantics collide with authenticated v17
    artifacts -- the real v17 development config IS schema 4."""
    assert V.V18_SCHEMA != 4
    assert _basis("sources.json")["schemas"]["schema4"]["declared_config_schema_version"] == 4


def test_parser_accepts_five_and_still_rejects_beyond():
    assert v18_profile().schema_version == 5
    for bad in (1, 6, "5", None):
        with pytest.raises(ValueError, match="config_schema_version"):
            V.parse_allocation_profile(v18_raw(config_schema_version=bad),
                                       source="x")


def test_schema_five_profile_shape_is_the_frozen_forty_row_allocation():
    profile = v18_profile()
    assert profile.splits == ("all",)
    assert profile.corpus_size == 40
    assert profile.split_totals == {"all": 40}
    assert profile.quota_by_phase == {"late": 28, "opening": 4,
                                      "early_mid": 4, "midgame": 4}
    assert profile.allocation == {c: dict(a) for c, a in V.SPLIT_ALLOC_V18.items()}
    assert profile.fingerprint()["splits"] == ["all"]
    assert profile.fingerprint()["schema_version"] == 5
    # Targets, identity witnesses and flip controls are late-only.
    for role in ("target", "identity_witness", "flip_control"):
        phases = {p for (r, p) in profile.allocation if r == role}
        assert phases == {"late"}, role
    assert {p for (r, p) in profile.allocation if r == "representative"} == {
        "opening", "early_mid", "midgame", "late"}


def test_schema_five_rejects_the_legacy_role_and_split_vocabulary():
    with pytest.raises(ValueError, match="unknown role"):
        V.parse_allocation_profile(
            v18_raw(phase_allocation={"control|late": {"all": 40}}), source="x")
    with pytest.raises(ValueError, match="split name"):
        V.parse_allocation_profile(
            v18_raw(phase_allocation={"target|late": {"tuning": 40}}), source="x")


def test_legacy_schemas_reject_the_v18_vocabulary():
    """Built from the REAL schema-4 config with only the vocabulary swapped, so
    the refusal is provably about the roles/splits and not about the v18 run
    identity (which schema 4 also refuses, for its own reason)."""
    legacy_raw = _basis("schema4.profile.json")
    legacy_raw.pop("selection_seed")
    legacy_raw["phase_allocation"] = v18_raw()["phase_allocation"]
    legacy_raw["corpus_size"] = 40
    with pytest.raises(ValueError, match="unknown role|split name"):
        V.parse_allocation_profile(legacy_raw, source="x")


# --- the frozen v18 run identity ---------------------------------------------

def test_profile_run_kinds_for_five_is_exactly_the_v18_kind():
    """Not a widening of the v17 set: schema 5 accepts ONE kind and none of the
    historical modes. Schemas 1-4 keep their own sets untouched."""
    assert V.profile_run_kinds_for(5) == ("v18_preflight_sizing",)
    assert V.PROFILE_RUN_KINDS_V18 == ("v18_preflight_sizing",)
    assert V.profile_run_kinds_for(4) == V.PROFILE_RUN_KINDS_V3
    assert V.profile_run_kinds_for(3) == V.PROFILE_RUN_KINDS_V3
    assert V.profile_run_kinds_for(2) == V.PROFILE_RUN_KINDS
    assert V.PROFILE_RUN_KINDS == ("production", "tooling_smoke")
    assert "v18_preflight_sizing" not in V.PROFILE_RUN_KINDS_V3


@pytest.mark.parametrize("kind", ["production", "tooling_smoke", "development",
                                  "held_out"])
def test_schema_five_refuses_every_non_v18_run_kind(kind):
    with pytest.raises(ValueError, match="run_kind"):
        V.parse_allocation_profile(v18_raw(run_kind=kind), source="x")


def test_schema_five_refuses_an_interpretable_label():
    with pytest.raises(ValueError, match="contradicts run_kind"):
        V.parse_allocation_profile(
            v18_raw(scientific_interpretation_forbidden=False), source="x")


@pytest.mark.parametrize("label", [1, "true", None])
def test_schema_five_refuses_a_non_true_label(label):
    """`True` EXACTLY -- a truthy 1 is not the frozen label."""
    with pytest.raises(ValueError, match="contradicts run_kind"):
        V.parse_allocation_profile(
            v18_raw(scientific_interpretation_forbidden=label), source="x")


def test_schema_five_requires_the_label_key_at_all():
    raw = v18_raw()
    raw.pop("scientific_interpretation_forbidden")
    with pytest.raises(ValueError, match="must carry"):
        V.parse_allocation_profile(raw, source="x")


def test_schema_five_never_delegates_to_the_global_interpretation_policy():
    """The CALL STRUCTURE, not just the outcome.

    `eval_runner.interpretation_forbidden` RAISES on a kind it does not know
    (eval_runner.py:56-66) and "v18_preflight_sizing" is deliberately absent
    from KNOWN_RUN_KINDS, so a delegating schema-5 branch would fail outright.
    Asserting zero calls also proves the local rule is not merely shadowing a
    delegation that still happens -- and the schema-4 half proves the legacy
    delegation was not simply deleted.
    """
    from scripts.GPU.alphazero import eval_runner
    calls = []
    real = eval_runner.interpretation_forbidden

    def spy(run_kind):
        calls.append(run_kind)
        return real(run_kind)

    eval_runner.interpretation_forbidden = spy
    try:
        V.parse_allocation_profile(v18_raw(), source="x")
        assert calls == []
        legacy = _basis("schema4.profile.json")
        legacy.pop("selection_seed")
        V.parse_allocation_profile(legacy, source="x")
        assert calls == [legacy["run_kind"]]
    finally:
        eval_runner.interpretation_forbidden = real


def test_the_v18_run_kind_is_not_registered_with_the_match_runner():
    """Registering it globally would widen the match runner's accepted label
    surface for a kind that never runs a match."""
    from scripts.GPU.alphazero import eval_runner
    assert "v18_preflight_sizing" not in eval_runner.KNOWN_RUN_KINDS
    with pytest.raises(ValueError):
        eval_runner.interpretation_forbidden("v18_preflight_sizing")


# --- the allocation is an ACCEPTANCE RULE, not a default ---------------------

@pytest.mark.parametrize("mutate, err", [
    (lambda c: c["phase_allocation"].pop("target|late"), "missing cell"),
    (lambda c: c["phase_allocation"]["target|late"].update({"all": 15}),
     "altered count"),
    (lambda c: (c["phase_allocation"]["target|late"].clear(),
                c["phase_allocation"]["target|late"].update({"tuning": 16})),
     "split name"),
    (lambda c: c["phase_allocation"].update({"bogus|late": {"all": 1}}),
     "unknown role"),
    # A KNOWN role in a phase the frozen table does not allocate -- distinct
    # from "unknown role", and the same condition as an extra cell.
    (lambda c: c["phase_allocation"].update({"target|opening": {"all": 4}}),
     "extra cell"),
    (lambda c: c["phase_allocation"].update({"target|opening": {"all": 4}}),
     "out-of-table role/phase"),
    (lambda c: c.update({"late_target_band_minima": {"all": {"b400_plus": 1}}}),
     "band minima"),
    (lambda c: c.update({"late_floors": {"b400_plus": 1}}), "band minima"),
])
def test_schema_five_parser_refusals(mutate, err):
    cfg = v18_raw()
    mutate(cfg)
    with pytest.raises(ValueError, match=err):
        V.parse_allocation_profile(cfg, source="x")


def test_correct_grand_total_with_shifted_quotas_is_still_refused():
    """The blocker this closes: validating the vocabulary and the TOTAL lets a
    config move counts between two otherwise valid cells and still pass."""
    cfg = v18_raw()
    cfg["phase_allocation"]["target|late"]["all"] = 15
    cfg["phase_allocation"]["representative|late"]["all"] = 5    # total still 40
    assert sum(a["all"] for a in cfg["phase_allocation"].values()) == 40
    with pytest.raises(ValueError, match="altered count"):
        V.parse_allocation_profile(cfg, source="x")


def test_schema_five_accepts_a_reordered_but_identical_allocation():
    """Cell ORDER is normalized before comparison; the normalization is what the
    profile's own `cell_order` records, so reordering cannot silently change
    selection order either."""
    cfg = v18_raw()
    cfg["phase_allocation"] = dict(reversed(list(cfg["phase_allocation"].items())))
    profile = V.parse_allocation_profile(cfg, source="x")
    assert profile.allocation == V.allocation_for_schema(5)
    assert profile.cell_order == v18_profile().cell_order


def test_allocation_for_schema_is_a_copy_not_the_frozen_constant():
    a = V.allocation_for_schema(5)
    a[("target", "late")]["all"] = 999
    assert V.SPLIT_ALLOC_V18[("target", "late")] == {"all": 16}


# --- splits and band_floor_cell are DERIVED, never stored --------------------

def test_splits_is_a_schema_derived_property_not_a_stored_field():
    """A stored field would permit an effective schema-5 profile carrying the
    legacy pair (or the reverse), and nothing downstream could tell which was
    true."""
    import dataclasses
    assert "splits" not in {f.name for f in
                            dataclasses.fields(V.AllocationProfile)}
    with pytest.raises(TypeError):
        V.AllocationProfile(
            schema_version=5, run_kind="v18_preflight_sizing",
            allocation={}, band_minima_total={}, band_minima_per_split={},
            max_per_game=1, min_ply_gap=12, side_tol=0,
            splits=("tuning", "frozen_check"))


def test_splits_follow_the_schema_on_a_hand_built_profile():
    hand = V.AllocationProfile(
        schema_version=5, run_kind="v18_preflight_sizing",
        allocation=V.allocation_for_schema(5), band_minima_total={},
        band_minima_per_split={}, max_per_game=1, min_ply_gap=12, side_tol=0)
    assert hand.splits == ("all",)
    assert V.AllocationProfile.legacy().splits == V.SPLITS


def test_band_floor_cell_is_derived_and_none_under_schema_five():
    assert v18_profile().band_floor_cell is None
    assert V.band_floor_cell_for(5) is None
    assert V.AllocationProfile.legacy().band_floor_cell == V.LATE_TARGET_CELL
    for schema in (1, 2, 3, 4):
        assert V.band_floor_cell_for(schema) == V.LATE_TARGET_CELL
    assert V.LATE_TARGET_CELL == ("target", "late")     # constant unmutated


@pytest.mark.parametrize("schema", SCHEMAS)
def test_legacy_band_floor_cell_is_unchanged_on_the_real_profiles(schema):
    raw = _basis(f"schema{schema}.profile.json")
    raw.pop("selection_seed")
    profile = V.parse_allocation_profile(raw, source="x")
    assert profile.band_floor_cell == V.LATE_TARGET_CELL


def test_schema_five_cannot_authenticate_v17_band_geometry():
    """With no cell to constrain, band minima are a parse error rather than an
    ignored field -- otherwise a v18 profile could carry v17 floors."""
    for key in ("late_floors", "late_target_band_minima"):
        cfg = v18_raw()
        cfg[key] = ({"b300_399": 4} if key == "late_floors"
                    else {"all": {"b300_399": 4}})
        with pytest.raises(ValueError, match="no band floor cell"):
            V.parse_allocation_profile(cfg, source="x")


# --- the fingerprint records the strategy it actually ran --------------------

def test_schema_five_fingerprint_records_the_assignment_strategy():
    fp = v18_profile().fingerprint()
    assert fp["assignment_strategy"] == "one_split"
    assert fp["splits"] == ["all"]


@pytest.mark.parametrize("schema", SCHEMAS)
def test_legacy_fingerprint_gains_no_assignment_strategy_key(schema):
    raw = _basis(f"schema{schema}.profile.json")
    raw.pop("selection_seed")
    profile = V.parse_allocation_profile(raw, source="x")
    assert "assignment_strategy" not in profile.fingerprint()
    assert profile.assignment_strategy == "two_way"


def test_the_recorded_strategy_is_the_one_the_selector_branches_on():
    """Same property, both consumers -- an artifact claiming `one_split` cannot
    have been produced by the two-way greedy."""
    profile, rows = v18_profile(), v18_pool()
    assert profile.assignment_strategy == "one_split"
    _sel, stats = V.sample_v2_rows(rows, seed=20260731, alloc=profile)
    assert {a["split"] for a in stats["split_assignment"]} == {"all"}
    assert stats["split_assignment_version"] == V.V18_SCHEMA


# --- one split, and its observable ------------------------------------------

def test_schema_five_selects_the_whole_corpus_into_one_split():
    profile, rows = v18_profile(), v18_pool()
    assert V.post_screen_qualification_report(rows, profile)["status"] == "PASS"
    selected, stats = V.sample_v2_rows(rows, seed=20260731, alloc=profile)
    assert len(selected) == 40
    assert {r["split"] for r in selected} == {"all"}
    assert collections.Counter(r["role"] for r in selected) == {
        "target": 16, "representative": 16, "identity_witness": 4,
        "flip_control": 4}


def test_split_assignment_covers_every_retained_game_not_only_selected():
    """The frozen observable. Without it a test can only inspect selected rows,
    which a two-way assigner could satisfy while mis-assigning the rest."""
    profile, rows = v18_profile(), v18_pool()
    selected, stats = V.sample_v2_rows(rows, seed=20260731, alloc=profile)
    assignment = stats["split_assignment"]
    drawn_from = {r["game_idx"] for r in selected}
    assert len(assignment) == len({r["game_idx"] for r in rows})
    assert len(assignment) > len(drawn_from)
    assert {a["split"] for a in assignment} == {"all"}


def test_split_assignment_is_a_sorted_record_list_not_a_map():
    """A `{game_idx: split}` map would serialize the int identity as a JSON
    object key and silently change its type on reload."""
    profile, rows = v18_profile(), v18_pool()
    _sel, stats = V.sample_v2_rows(rows, seed=20260731, alloc=profile)
    assignment = stats["split_assignment"]
    assert isinstance(assignment, list)
    assert assignment == sorted(assignment, key=lambda a: a["game_idx"])
    assert all(set(a) == {"game_idx", "split"} for a in assignment)
    reloaded = json.loads(json.dumps(assignment))
    assert all(isinstance(a["game_idx"], int) for a in reloaded)
    assert reloaded == assignment


def test_schema_five_selection_is_deterministic():
    profile, rows = v18_profile(), v18_pool()
    first, stats_a = V.sample_v2_rows(rows, seed=20260731, alloc=profile)
    second, stats_b = V.sample_v2_rows(rows, seed=20260731, alloc=profile)
    assert _canon(first) == _canon(second)
    assert _canon(stats_a) == _canon(stats_b)


def test_schema_five_qualification_still_emits_late_target_bands_as_empty():
    """`sizing_analysis_core`'s band loop then iterates zero times and needs no
    edit."""
    report = V.post_screen_qualification_report(v18_pool(), v18_profile())
    assert report["late_target_bands"] == {}
    assert "late_target_bands" in report


def test_infeasible_schema_five_geometry_raises_rather_than_relaxing():
    profile = v18_profile()
    starved = [r for r in v18_pool()
               if (r["role"], r["phase"]) != ("representative", "late")]
    assert V.post_screen_qualification_report(starved, profile)["status"] == "GATE_FAIL"
    with pytest.raises(ValueError, match="capacity"):
        V.sample_v2_rows(starved, seed=20260731, alloc=profile)
