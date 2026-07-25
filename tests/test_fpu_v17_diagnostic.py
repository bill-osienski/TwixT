"""v17 Task 5 -- modes, pure gates, selection, and artifact identities.

Frozen design ref:
`docs/superpowers/specs/2026-07-24-v17-baseline-preserving-policy-mass-fpu-design.md`
(SHA-1 `944f358c0e3ef66503d2cbb56e31dabd145bafc2`) §§7-9.

Every fixture is fabricated, so each threshold is pinned on BOTH sides of its
boundary -- the value that must pass and the neighbouring value that must fail.
No GPU, no evaluator, no checkpoint weights.
"""
import csv
import json
import pathlib

import pytest

from scripts.GPU.alphazero import diagnose_fpu_baseline_policy_mass as v17
from scripts.GPU.alphazero import diagnose_fpu_policy_mass as v16
from scripts.GPU.alphazero import fpu_provenance
from scripts.GPU.alphazero import fpu_v17_protocol as protocol
from scripts.GPU.alphazero import fpu_v17_provenance as prov

CKPT = "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"
SRC = "scripts/GPU/alphazero/mcts.py"


@pytest.fixture
def clean_tree(monkeypatch):
    monkeypatch.setattr(fpu_provenance, "worktree_clean", lambda: True)


# ---------------------------------------------------------------------------
# Row fabrication
# ---------------------------------------------------------------------------

def row(sha, coefficient, role="target", side="red", ply_bucket="late", **over):
    base = dict(canonical_sha1=sha, role=role, side=side,
                ply_bucket=ply_bucket, coefficient=coefficient,
                seed=1234, add_noise=False,
                selected_move=1, selected_prior=0.02, selected_prior_rank=1,
                root_value_stm=0.0, parent_value=0.0, selected_child_q=0.0,
                top_share=0.5, eff_children=100.0, replies=100, collapse=False,
                lock_in=False, explored_mass=0.25, stabilization_sim=200,
                complete=True, tree_signature=[[1, 100, "0x0.0p+0"]])
    base.update(over)
    base["search_result_sha1"] = v17.search_result_sha1(
        {**base, "search_result_sha1": ""})
    return base


def corpus(n_targets=16, n_controls=16, configs=(None, 0.0), **cand):
    """One row per config per position, in the frozen development geometry.
    `cand` overrides are applied only to NON-shipped, non-zero rows."""
    rows = []
    for i in range(n_targets + n_controls):
        sha = f"pos{i:03d}"
        role = "target" if i < n_targets else "control"
        for c in configs:
            over = dict(cand) if (c is not None and c != 0.0) else {}
            bucket = "late" if i < n_targets else v17.PHASES[
                (i - n_targets) % 4]
            rows.append(row(sha, c, role=role, ply_bucket=bucket,
                            side="red" if i % 2 == 0 else "black", **over))
    return rows


def rehash(rows):
    """Re-seal rows after a test mutates them. `search_result_sha1` binds the
    payload, so an in-place edit must be re-sealed exactly as a real producer
    would -- that binding is itself under test elsewhere."""
    for r in rows:
        r["search_result_sha1"] = v17.search_result_sha1(
            {**r, "search_result_sha1": ""})
    return rows


def heldout_corpus(configs, **cand):
    """§8.1 geometry: 24 targets + 32 controls."""
    return corpus(n_targets=24, n_controls=32, configs=configs, **cand)


# ---------------------------------------------------------------------------
# Re-export identity: no second copy of any frozen definition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "prior_rank", "reply_reduction", "top_share", "lock_in_event", "_percentile",
    "_mean", "progress", "selected_a_verdict", "dev_safety_verdict",
    "COLLAPSE_TOP_SHARE", "DEV_NEW_COLLAPSE_TARGET", "DEV_P95_MOVER",
    "DEV_COMPOUND_EFF", "DEV_COMPOUND_TOPSHARE", "DEV_CONTROL_FLIP",
    "DEV_CONTROL_P95", "DEV_LOCKIN_MARGIN", "PERCENTILE_Q", "V_REF",
    "A_REPLY_REDUCTION", "A_PROGRESS", "A_NEW_COLLAPSE_MAX", "A_TOPSHARE_MAX",
])
def test_v17_reexports_are_the_v16_objects(name):
    """Proves 'import, rather than copy': v17 holds no independent definition
    that could drift from the frozen one."""
    assert getattr(v17, name) is getattr(v16, name)


# `MODES` names a different concept in each module -- v16's search stages
# ("tuning", "frozen_check") vs v17's diagnostic modes -- so it is the one
# shared uppercase name that legitimately differs.
DELIBERATELY_REDEFINED = {"MODES"}


def test_v17_defines_no_shadowing_threshold():
    """v17 may add gate constants the v16 module lacks (§7.3/§8.2/§9 are new),
    but must never redefine a threshold it already has under the same name."""
    shared = {n for n in set(dir(v17)) & set(dir(v16))
              if n.isupper() and not n.startswith("_")}
    for name in shared - DELIBERATELY_REDEFINED:
        assert getattr(v17, name) is getattr(v16, name), name
    assert v16.MODES == ("tuning", "frozen_check") and v17.MODES != v16.MODES


# ---------------------------------------------------------------------------
# Modes -- exact config sets, and A/B/C/D excluded from selection
# ---------------------------------------------------------------------------

def test_development_runs_shipped_zero_and_the_whole_grid():
    assert v17.configs_for_mode("development") == (None, 0.0) + tuple(prov.GRID)


def test_smoke_runs_only_plumbing_configs():
    assert v17.configs_for_mode("tooling_smoke") == (None, 0.0, 0.35)


@pytest.mark.parametrize("mode", ["held_out", "abcd"])
def test_later_stages_take_exactly_one_frozen_coefficient(mode):
    assert v17.configs_for_mode(mode, frozen_coefficient=0.25) == (None, 0.25)
    with pytest.raises(prov.ProtocolViolation, match="exactly one frozen"):
        v17.configs_for_mode(mode)
    for bad in (0.0, None):
        with pytest.raises(prov.ProtocolViolation):
            v17.configs_for_mode(mode, frozen_coefficient=bad)
    with pytest.raises(prov.ProtocolViolation, match="frozen grid"):
        v17.configs_for_mode(mode, frozen_coefficient=0.30)


def test_development_may_not_be_handed_a_coefficient():
    """Development SELECTS; being given one would invert the stage order."""
    with pytest.raises(prov.ProtocolViolation, match="SELECTS"):
        v17.configs_for_mode("development", frozen_coefficient=0.25)


def test_smoke_may_not_be_handed_a_coefficient():
    with pytest.raises(prov.ProtocolViolation, match="no frozen coefficient"):
        v17.configs_for_mode("tooling_smoke", frozen_coefficient=0.25)


def test_unknown_mode_refused():
    with pytest.raises(prov.ProtocolViolation, match="unknown mode"):
        v17.configs_for_mode("acceptance")


def test_abcd_cannot_widen_to_a_grid():
    """§9: 'No other positive coefficient runs. These probes do not select or
    tune the coefficient.'"""
    assert len(v17.configs_for_mode("abcd", frozen_coefficient=0.35)) == 2


# ---------------------------------------------------------------------------
# Pairing completeness
# ---------------------------------------------------------------------------

def test_complete_pairing_accepts_a_full_corpus():
    rows = corpus(configs=(None, 0.0, 0.35))
    v17.require_complete_pairing(rows, (None, 0.0, 0.35))


def test_missing_config_row_is_refused():
    rows = [r for r in corpus(configs=(None, 0.0, 0.35))
            if not (r["canonical_sha1"] == "pos003" and r["coefficient"] == 0.35)]
    with pytest.raises(prov.ProtocolViolation, match="incomplete pairing"):
        v17.require_complete_pairing(rows, (None, 0.0, 0.35))


def test_incomplete_search_row_is_refused():
    rows = corpus(configs=(None, 0.35))
    rows[1]["complete"] = False
    with pytest.raises(prov.ProtocolViolation, match="incomplete search"):
        v17.require_complete_pairing(rows, (None, 0.35))


@pytest.mark.parametrize("field", ["root_value_stm", "top_share", "eff_children",
                                   "explored_mass"])
def test_nonfinite_metric_is_refused(field):
    rows = corpus(configs=(None, 0.35))
    rows[1][field] = float("nan")
    with pytest.raises(prov.ProtocolViolation, match="nonfinite"):
        v17.require_complete_pairing(rows, (None, 0.35))


def test_missing_required_field_is_refused():
    rows = corpus(configs=(None, 0.35))
    del rows[1]["explored_mass"]
    with pytest.raises(prov.ProtocolViolation, match="missing fields"):
        v17.require_complete_pairing(rows, (None, 0.35))


def test_empty_rows_refused():
    with pytest.raises(prov.ProtocolViolation, match="no rows"):
        v17.require_complete_pairing([], (None,))


# ---------------------------------------------------------------------------
# §7.1 r=0 identity prerequisite
# ---------------------------------------------------------------------------

def test_zero_identity_holds_on_identical_rows():
    v17.require_zero_identity(corpus(configs=(None, 0.0)))


@pytest.mark.parametrize("field,value", [
    ("selected_move", 2), ("root_value_stm", 0.01), ("top_share", 0.51),
    ("eff_children", 99.0), ("replies", 99), ("collapse", True),
    ("explored_mass", 0.26), ("stabilization_sim", 199),
])
def test_zero_identity_failure_is_detected(field, value):
    rows = corpus(configs=(None, 0.0))
    for r in rows:
        if r["coefficient"] == 0.0 and r["canonical_sha1"] == "pos000":
            r[field] = value
    with pytest.raises(prov.ProtocolViolation, match="r=0 identity FAILED"):
        v17.require_zero_identity(rows)


def test_zero_identity_requires_both_arms():
    with pytest.raises(prov.ProtocolViolation, match="needs both"):
        v17.require_zero_identity(corpus(configs=(None,)))


# ---------------------------------------------------------------------------
# §7.2 development safety -- both sides of every boundary
# ---------------------------------------------------------------------------

def _safety(rows, r=0.35, lockin=0, margin=v17.DEV_LOCKIN_MARGIN_V17):
    return v17.dev_safety_v17(rows, r, shipped_lockin=lockin, lockin_margin=margin)


def test_zero_new_collapse_passes_and_one_rejects():
    """16 targets: 1/16 = 6.25% >= 5%, so the gate permits only zero."""
    rows = corpus(configs=(None, 0.35))
    assert not _safety(rows).rejected
    for r in rows:
        if r["coefficient"] == 0.35 and r["canonical_sha1"] == "pos000":
            r["collapse"] = True
    rehash(rows)
    assert "target_new_collapse_rate" in " ".join(_safety(rows).reasons)


def test_control_flip_permits_one_and_rejects_two():
    """16 controls: 1/16 = 6.25% < 10% passes; 2/16 = 12.5% rejects."""
    def flipped(k):
        rows = corpus(configs=(None, 0.35))
        n = 0
        for r in rows:
            if (r["coefficient"] == 0.35 and r["role"] == "control" and n < k):
                r["selected_move"], r["selected_prior"] = 99, 0.001
                n += 1
        return rehash(rows)
    assert not _safety(flipped(1)).rejected
    assert any("control_flip_rate" in x for x in _safety(flipped(2)).reasons)


def test_lockin_margin_is_one_for_development_and_two_for_heldout():
    rows = corpus(configs=(None, 0.35))
    n = 0
    for r in rows:
        if r["coefficient"] == 0.35 and r["role"] == "target" and n < 2:
            r["lock_in"] = True
            n += 1
    rehash(rows)
    assert any("lockin_count" in x for x in _safety(rows, lockin=0).reasons)
    assert not _safety(rows, lockin=0,
                       margin=v17.HELDOUT_LOCKIN_MARGIN).rejected


def test_p95_mover_delta_boundary():
    below = corpus(configs=(None, 0.35), root_value_stm=0.34)
    at = corpus(configs=(None, 0.35), root_value_stm=0.35)
    assert not _safety(below).rejected
    assert any("p95_mover_delta" in x for x in _safety(at).reasons)


def test_compound_gate_needs_both_arms():
    """>=50% eff reduction AND >=0.15 top-share increase; either alone passes."""
    eff_only = corpus(configs=(None, 0.35), eff_children=50.0, top_share=0.6)
    both = corpus(configs=(None, 0.35), eff_children=50.0, top_share=0.65)
    assert not _safety(eff_only).rejected
    assert any("compound" in x for x in _safety(both).reasons)


def test_per_stratum_subgate_is_disabled_for_v17():
    """v17 rows carry no `band` key at all; the v16 grouping helper raises on a
    missing key, so this also proves the sub-gate never runs."""
    rows = corpus(configs=(None, 0.35))
    assert "band" not in rows[0]
    verdict = _safety(rows)
    assert not any("band[" in x for x in verdict.reasons)
    assert "band_new_collapse_rates" not in verdict.metrics


# ---------------------------------------------------------------------------
# §7.3 development mechanism -- both sides
# ---------------------------------------------------------------------------

def test_reply_reduction_boundary():
    """1 - 50/100 is exactly 0.5 in IEEE754, so this boundary is exact.
    `eff_children` is reduced too, so only the reply gate is under test."""
    at = corpus(configs=(None, 0.35), replies=50, eff_children=70.0)
    below = corpus(configs=(None, 0.35), replies=51, eff_children=70.0)
    assert v17.dev_mechanism_verdict(at, 0.35).passed
    assert any("reply_reduction" in x
               for x in v17.dev_mechanism_verdict(below, 0.35).reasons)


def test_minimum_targets_with_fewer_replies():
    def with_fewer(k):
        rows = corpus(configs=(None, 0.35), replies=100)
        n = 0
        for r in rows:
            if r["coefficient"] == 0.35 and r["role"] == "target" and n < k:
                r["replies"] = 1
                n += 1
        return rehash(rows)
    assert any("targets_with_fewer_replies" in x
               for x in v17.dev_mechanism_verdict(with_fewer(7), 0.35).reasons)
    assert not any("targets_with_fewer_replies" in x
                   for x in v17.dev_mechanism_verdict(with_fewer(8), 0.35).reasons)


def test_eff_children_reduction_must_be_positive_but_under_fifty_percent():
    zero = corpus(configs=(None, 0.35), replies=10, eff_children=100.0)
    ok = corpus(configs=(None, 0.35), replies=10, eff_children=70.0)
    too_much = corpus(configs=(None, 0.35), replies=10, eff_children=50.0)
    assert any("mean_eff_children_reduction" in x
               for x in v17.dev_mechanism_verdict(zero, 0.35).reasons)
    assert v17.dev_mechanism_verdict(ok, 0.35).passed
    assert any("mean_eff_children_reduction" in x
               for x in v17.dev_mechanism_verdict(too_much, 0.35).reasons)


def test_rank_worsening_permits_one_and_rejects_two():
    def worsened(k):
        rows = corpus(configs=(None, 0.35), replies=10, eff_children=70.0)
        n = 0
        for r in rows:
            if r["coefficient"] == 0.35 and r["role"] == "target" and n < k:
                r["selected_move"], r["selected_prior_rank"] = 42, 11
                n += 1
        return rehash(rows)
    assert v17.dev_mechanism_verdict(worsened(1), 0.35).passed
    assert any("rank_worsened" in x
               for x in v17.dev_mechanism_verdict(worsened(2), 0.35).reasons)


# ---------------------------------------------------------------------------
# Selection -- smallest passing, never best
# ---------------------------------------------------------------------------

def _passing_rows_for(grid_points):
    """Rows where exactly `grid_points` clear §§7.2-7.3."""
    rows = corpus(configs=(None, 0.0) + tuple(prov.GRID))
    for r in rows:
        if r["coefficient"] in grid_points:
            r["replies"], r["eff_children"] = 10, 70.0
    return rehash(rows)


def test_selects_the_smallest_passing_not_the_best():
    rows = _passing_rows_for({0.25, 0.45})
    chosen, table = v17.select_smallest_passing(rows, shipped_lockin=0)
    assert chosen == 0.25
    assert table[0.25].passed and table[0.45].passed
    assert not table[0.15].passed


def test_returns_none_when_no_coefficient_passes():
    chosen, table = v17.select_smallest_passing(corpus(
        configs=(None, 0.0) + tuple(prov.GRID)), shipped_lockin=0)
    assert chosen is None
    assert set(table) == set(prov.GRID)
    assert not any(g.passed for g in table.values())


def test_every_grid_point_is_evaluated_and_persisted():
    _chosen, table = v17.select_smallest_passing(
        _passing_rows_for({0.45}), shipped_lockin=0)
    assert sorted(table) == sorted(prov.GRID)


# ---------------------------------------------------------------------------
# §8.2 held-out
# ---------------------------------------------------------------------------

def test_heldout_transfer_floor_boundary():
    above = corpus(configs=(None, 0.25), replies=75)   # 25%
    below = corpus(configs=(None, 0.25), replies=85)   # 15%
    assert v17.heldout_verdict(above, 0.25, shipped_lockin=0).passed
    assert any("heldout_reply_reduction" in x for x in
               v17.heldout_verdict(below, 0.25, shipped_lockin=0).reasons)


def test_exactly_twenty_percent_by_count_passes_the_floor():
    """Mathematical conformance, not a threshold change.

    §7.0's frozen aggregate is still what every artifact REPORTS, but the gate
    DECISION uses exact integer arithmetic: `1 - 80/100` is
    0.19999999999999996 in IEEE754 and would reject a reduction that is exactly
    20% by count. The reported float is unchanged.
    """
    assert 1 - 80 / 100 < 0.20                       # the float really is below
    at_boundary = corpus(configs=(None, 0.25), replies=80)
    res = v17.heldout_verdict(at_boundary, 0.25, shipped_lockin=0)
    assert res.passed
    assert res.metrics["heldout_reply_reduction_exact"] == "1/5"
    assert res.metrics["heldout_reply_reduction"] == 1 - 80 / 100   # still frozen


def test_exact_arithmetic_also_governs_the_attenuation_classification():
    """An exactly-50% reduction is NOT attenuated; a hair under it is."""
    exact_half = corpus(configs=(None, 0.25), replies=50)
    assert v17.heldout_verdict(exact_half, 0.25,
                               shipped_lockin=0).metrics[
                                   "attenuated_but_present"] is False
    just_under = corpus(configs=(None, 0.25), replies=51)
    assert v17.heldout_verdict(just_under, 0.25,
                               shipped_lockin=0).metrics[
                                   "attenuated_but_present"] is True


def test_exact_arithmetic_governs_the_development_fifty_percent_gate():
    at = corpus(configs=(None, 0.35), replies=50, eff_children=70.0)
    below = corpus(configs=(None, 0.35), replies=51, eff_children=70.0)
    assert v17.dev_mechanism_verdict(at, 0.35).passed
    assert v17.dev_mechanism_verdict(at, 0.35).metrics[
        "reply_reduction_exact"] == "1/2"
    assert not v17.dev_mechanism_verdict(below, 0.35).passed


def test_heldout_classifies_attenuated_but_present():
    rows = corpus(configs=(None, 0.25), replies=70)    # 30%
    res = v17.heldout_verdict(rows, 0.25, shipped_lockin=0)
    assert res.passed and res.metrics["attenuated_but_present"] is True
    strong = v17.heldout_verdict(corpus(configs=(None, 0.25), replies=40),
                                 0.25, shipped_lockin=0)
    assert strong.metrics["attenuated_but_present"] is False


def test_heldout_collateral_failure_rejects_even_with_mechanism():
    rows = corpus(configs=(None, 0.25), replies=10, root_value_stm=0.5)
    assert not v17.heldout_verdict(rows, 0.25, shipped_lockin=0).passed


# ---------------------------------------------------------------------------
# §9 A/B/C/D
# ---------------------------------------------------------------------------

def _a_rows(reply_red=0.6, prog=0.6, collapse=0, tsi=0.0):
    """A rows shaped for the imported selected_a_verdict aggregator."""
    off, r = 0.25702582687976244, 0.25702582687976244 - prog * (
        0.25702582687976244 - v16.V_REF)
    return [{"off_value": off, "r_value": r, "replies_ref": 100,
             "replies_x": 100 * (1 - reply_red),
             "new_collapse": i < collapse, "top_share_inc": tsi}
            for i in range(30)]


R = 0.25   # a frozen grid point; A/B/C/D always runs one


def test_a_gate_passes_only_when_every_criterion_holds():
    assert v17.abcd_verdict("A", coefficient=R, n=30, mean=-0.1, over=15,
                            severe=5, a_rows=_a_rows()).passed


@pytest.mark.parametrize("kwargs,expect", [
    (dict(mean=0.01, severe=5), "A_mean"),
    (dict(mean=-0.1, severe=6), "A_severe"),
])
def test_a_gate_count_criteria(kwargs, expect):
    res = v17.abcd_verdict("A", coefficient=R, n=30, over=15,
                           a_rows=_a_rows(), **kwargs)
    assert any(expect in x for x in res.reasons)


def test_a_severe_boundary_is_five_not_six():
    """§9 was corrected pre-freeze from <=6/30 to <=5/30 (v14b parity)."""
    assert v17.abcd_verdict("A", coefficient=R, n=30, mean=-0.1, over=15,
                            severe=5, a_rows=_a_rows()).passed
    assert not v17.abcd_verdict("A", coefficient=R, n=30, mean=-0.1, over=15,
                                severe=6, a_rows=_a_rows()).passed


@pytest.mark.parametrize("gate,ok,bad", [
    ("B", dict(n=18, mean=-0.24, over=2, severe=0),
     dict(n=18, mean=-0.24, over=3, severe=0)),
    ("B", dict(n=18, mean=-0.24, over=2, severe=0),
     dict(n=18, mean=-0.24, over=2, severe=1)),
    ("C", dict(n=30, mean=0.099, over=10, severe=4),
     dict(n=30, mean=0.1, over=10, severe=4)),
    ("C", dict(n=30, mean=0.099, over=10, severe=4),
     dict(n=30, mean=0.099, over=11, severe=4)),
    ("D", dict(n=30, mean=0.0, over=4, severe=0),
     dict(n=30, mean=0.001, over=4, severe=0)),
    ("D", dict(n=30, mean=0.0, over=4, severe=0),
     dict(n=30, mean=0.0, over=4, severe=1)),
])
def test_bcd_boundaries(gate, ok, bad):
    assert v17.abcd_verdict(gate, coefficient=R, **ok).passed
    assert not v17.abcd_verdict(gate, coefficient=R, **bad).passed


def test_a_gate_requires_case_rows():
    with pytest.raises(prov.ProtocolViolation, match="per-case rows"):
        v17.abcd_verdict("A", coefficient=R, n=30, mean=-0.1, over=15, severe=5)


def test_unknown_abcd_gate_refused():
    with pytest.raises(prov.ProtocolViolation, match="unknown A/B/C/D gate"):
        v17.abcd_verdict("E", coefficient=R, n=1, mean=0.0, over=0, severe=0)


# --- frozen cardinalities and scalar sanity (adversarial round 2) ----------

@pytest.mark.parametrize("gate,frozen_n", [("A", 30), ("B", 18), ("C", 30), ("D", 30)])
def test_abcd_cardinality_is_frozen(gate, frozen_n):
    """B previously passed with n=1: a truncated case set must be a refusal,
    never a pass on fewer cases."""
    assert v17.ABCD_CARDINALITY[gate] == frozen_n
    for bad_n in (1, frozen_n - 1, frozen_n + 1):
        with pytest.raises(prov.ProtocolViolation, match="frozen cardinality"):
            v17.abcd_verdict(gate, coefficient=R, n=bad_n, mean=0.0, over=0,
                             severe=0, a_rows=_a_rows())


def test_a_case_rows_must_match_the_frozen_cardinality():
    with pytest.raises(prov.ProtocolViolation, match="frozen cardinality"):
        v17.abcd_verdict("A", coefficient=R, n=30, mean=-0.1, over=15, severe=5,
                         a_rows=_a_rows()[:29])


@pytest.mark.parametrize("bad", [
    dict(over=-1, severe=0), dict(over=31, severe=0), dict(over=True, severe=0),
    dict(over=5, severe=6),          # severe must be a subset of over
])
def test_abcd_counts_must_be_valid_integers(bad):
    with pytest.raises(prov.ProtocolViolation):
        v17.abcd_verdict("C", coefficient=R, n=30, mean=0.0, **bad)


@pytest.mark.parametrize("mean", [float("nan"), float("inf"), "0.0", True])
def test_abcd_mean_must_be_finite(mean):
    with pytest.raises(prov.ProtocolViolation, match="must be finite"):
        v17.abcd_verdict("C", coefficient=R, n=30, mean=mean, over=0, severe=0)


@pytest.mark.parametrize("bad", [None, 0.0, 0.30])
def test_abcd_requires_one_frozen_positive_coefficient(bad):
    with pytest.raises(prov.ProtocolViolation):
        v17.abcd_verdict("C", coefficient=bad, n=30, mean=0.0, over=0, severe=0)


def test_abcd_gate_result_records_the_real_coefficient():
    """A NaN coefficient cannot appear in canonical no-NaN JSON."""
    res = v17.abcd_verdict("C", coefficient=R, n=30, mean=0.0, over=0, severe=0)
    assert res.coefficient == R
    json.dumps(res.coefficient, allow_nan=False)


# ---------------------------------------------------------------------------
# Artifacts -- scientific modes must POPULATE identities
# ---------------------------------------------------------------------------

def _artifact(mode, tmp_path, **over):
    manifest = tmp_path / "m.csv"
    manifest.write_text("case_id\n1\n")
    index = tmp_path / "i.json"
    index.write_text("{}")
    replay = tmp_path / "r.json"
    replay.write_text('{"moves": []}')
    coefficient = over.pop("coefficient", 0.25)
    configs = v17.configs_for_mode(
        mode, frozen_coefficient=None if mode in ("development", "tooling_smoke")
        else coefficient)
    rows = (corpus(configs=configs) if mode in ("development", "tooling_smoke")
            else heldout_corpus(configs))
    kwargs = dict(mode=mode, coefficient=coefficient, rows=rows, gates={},
                  checkpoints={"a": CKPT},
                  effective_mcts_config={"shipped": {"n_simulations": 400}},
                  protocol_sha1="0" * 40, manifest=str(manifest),
                  source_index=str(index), replay_paths=[str(replay)],
                  source_files=[SRC])
    kwargs.update(over)
    return v17.build_artifact(**kwargs)


def test_scientific_artifact_populates_every_identity(clean_tree, tmp_path):
    art = _artifact("held_out", tmp_path)
    ids = art["provenance"]["identities"]
    assert all(ids[k] for k in ("manifest_sha1", "source_index_sha1",
                                "replay_data_sha1"))
    assert art["provenance"]["checkpoints"]["a"]
    assert art["provenance"]["source_file_sha1s"]["mcts.py"]


def test_scientific_artifact_persists_the_complete_paired_rows(clean_tree, tmp_path):
    """A count alone would make every gate number unauditable."""
    art = _artifact("held_out", tmp_path)
    assert art["n_rows"] == len(art["rows"]) == 112     # 56 positions x 2 configs
    for row in art["rows"]:
        assert set(row) == set(v17.REQUIRED_ROW_FIELDS)
    json.dumps(art, allow_nan=False)                    # canonical-JSON safe


@pytest.mark.parametrize("null", ["manifest", "source_index", "replay_paths",
                                  "checkpoints", "source_files",
                                  "protocol_sha1", "effective_mcts_config"])
@pytest.mark.parametrize("mode", ["development", "held_out"])
def test_scientific_artifact_refuses_a_null_identity(clean_tree, tmp_path, mode, null):
    empty = {"checkpoints": {}, "source_files": [],
             "effective_mcts_config": {}}.get(null, None)
    with pytest.raises(prov.ProtocolViolation, match="must "):
        _artifact(mode, tmp_path, **{null: empty})


def test_scientific_artifact_records_the_complete_effective_config(clean_tree,
                                                                   tmp_path):
    """A bare coefficient label would not let anyone reproduce the search."""
    art = _artifact("held_out", tmp_path,
                    effective_mcts_config=v17.effective_configs((None, 0.25)))
    cfg = art["effective_mcts_config"]["0.25"]
    assert cfg["n_simulations"] == 400 and cfg["add_noise"] is False
    assert cfg["fpu_value"] == 0.0
    assert cfg[prov.CONFIG_FIELD] == 0.25
    assert cfg["eval_batch_size"] == 14 and cfg["stall_flush_sims"] == 48
    # the COMPLETE dataclass, not a hand-built subset
    import dataclasses
    from scripts.GPU.alphazero.mcts import MCTSConfig
    assert set(dataclasses.asdict(MCTSConfig())) <= set(cfg)
    for extra in ("c_puct", "temp_high", "temp_low", "root_edge_band_penalty",
                  "closeout_td1_visit_forcing_enabled", "dirichlet_alpha"):
        assert extra in cfg, extra
    assert art["protocol_sha1"]


def _manifest_rows(mode, **over):
    """Manifest rows in the mode's complete frozen geometry."""
    geometry = v17.CORPUS_GEOMETRY[mode]
    rows, i = [], 0
    for role, count in geometry.items():
        for _ in range(count):
            rows.append({"role": role, "side": "red" if i % 2 == 0 else "black",
                         "canonical_sha1": f"p{i:03d}", "game_idx": i,
                         "position_ply": 50,
                         "ply_bucket": "late" if role == "target"
                         else v17.PHASES[len(rows) % 4]})
            i += 1
    for k, v in over.items():
        rows[0][k] = v
    return rows


@pytest.mark.parametrize("mode,geometry", [
    ("development", {"target": 16, "control": 16}),
    ("held_out", {"target": 24, "control": 32}),
])
def test_corpus_geometry_is_frozen_per_mode(mode, geometry):
    assert v17.CORPUS_GEOMETRY[mode] == geometry
    v17.require_corpus_geometry(mode, _manifest_rows(mode))
    with pytest.raises(prov.ProtocolViolation, match="corpus roles"):
        v17.require_corpus_geometry(mode, _manifest_rows(mode)[:-1])


@pytest.mark.parametrize("mode,per_side", [("development", 16), ("held_out", 28)])
def test_corpus_side_balance_is_enforced(mode, per_side):
    """An all-red corpus satisfies the role counts but violates §6.2/§8.1."""
    assert v17.SIDE_BALANCE[mode] == per_side
    all_red = [{**r, "side": "red"} for r in _manifest_rows(mode)]
    with pytest.raises(prov.ProtocolViolation, match="side balance"):
        v17.require_corpus_geometry(mode, all_red)


def test_corpus_rejects_duplicate_canonical_positions():
    rows = _manifest_rows("development")
    rows[1]["canonical_sha1"] = rows[0]["canonical_sha1"]
    with pytest.raises(prov.ProtocolViolation, match="repeats canonical"):
        v17.require_corpus_geometry("development", rows)


def test_corpus_rejects_more_than_two_positions_per_game():
    rows = _manifest_rows("development")
    for i in range(3):
        rows[i]["game_idx"] = 999
        rows[i]["position_ply"] = 50 + i * 20
    with pytest.raises(prov.ProtocolViolation, match="frozen cap"):
        v17.require_corpus_geometry("development", rows)


def test_corpus_rejects_positions_closer_than_twelve_plies():
    rows = _manifest_rows("development")
    rows[0]["game_idx"] = rows[1]["game_idx"] = 777
    rows[0]["position_ply"], rows[1]["position_ply"] = 50, 61
    with pytest.raises(prov.ProtocolViolation, match="ply spacing"):
        v17.require_corpus_geometry("development", rows)
    rows[1]["position_ply"] = 62
    v17.require_corpus_geometry("development", rows)


def test_artifact_refuses_wrong_corpus_geometry(clean_tree, tmp_path):
    with pytest.raises(prov.ProtocolViolation, match="corpus roles"):
        _artifact("development", tmp_path, coefficient=None,
                  rows=corpus(n_targets=15, n_controls=16,
                              configs=(None, 0.0) + tuple(prov.GRID)))


def test_artifact_enforces_mode_coefficient_legality(clean_tree, tmp_path):
    for bad in (None, 0.0, 0.30):
        with pytest.raises(prov.ProtocolViolation):
            _artifact("held_out", tmp_path, coefficient=bad)


def test_artifact_runs_pairing_and_zero_identity_itself(clean_tree, tmp_path):
    configs = (None, 0.0) + tuple(prov.GRID)
    rows = [r for r in corpus(configs=configs)
            if not (r["canonical_sha1"] == "pos000" and r["coefficient"] == 0.45)]
    with pytest.raises(prov.ProtocolViolation, match="incomplete pairing"):
        _artifact("development", tmp_path, coefficient=None, rows=rows)
    broken = corpus(configs=configs)
    for r in broken:
        if r["coefficient"] == 0.0 and r["canonical_sha1"] == "pos000":
            r["selected_move"] = 99
    rehash(broken)
    with pytest.raises(prov.ProtocolViolation, match="r=0 identity FAILED"):
        _artifact("development", tmp_path, coefficient=None, rows=broken)


def test_tooling_smoke_artifact_may_omit_identities(clean_tree, tmp_path):
    art = v17.build_artifact(mode="tooling_smoke", coefficient=None, rows=[],
                             gates={}, checkpoints={})
    assert art["provenance"]["scientific_interpretation_forbidden"] is True


def test_artifact_is_timestamp_free(clean_tree, tmp_path):
    blob = json.dumps(_artifact("held_out", tmp_path))
    for banned in ("timestamp", "generated_at", "datetime"):
        assert banned not in blob


# ---------------------------------------------------------------------------
# CLI / config refusals
# ---------------------------------------------------------------------------

def _cli(monkeypatch, tmp_path, **over):
    monkeypatch.setattr(fpu_provenance, "worktree_clean", lambda: True)
    manifest = tmp_path / "m.csv"
    manifest.write_text("canonical_sha1,game_idx,position_ply,side,role,"
                        "replay_path\np0,1,50,red,target,r.json\n")
    args = ["--mode", "development", "--checkpoint", CKPT,
            "--manifest", str(manifest), "--seed-base", "20260725",
            "--out", prov.OUTPUT_ROOT + "/x.json"]
    for k, v in over.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    return v17.main(args)


@pytest.mark.parametrize("override", [
    {"eval_batch_size": 16}, {"stall_flush_sims": 16}, {"stall_flush_sims": 0},
])
def test_cli_refuses_free_batching_overrides(monkeypatch, tmp_path, override, capsys):
    """§2.4: free CLI overrides are rejected, not honoured."""
    assert _cli(monkeypatch, tmp_path, **override) == 2
    assert "batching" in capsys.readouterr().out


def test_cli_refuses_a_v16_output_root(monkeypatch, tmp_path, capsys):
    assert _cli(monkeypatch, tmp_path,
                out="logs/eval/fpu_v16_policy_mass_v2/x.json") == 2
    assert "v17 root" in capsys.readouterr().out


def test_cli_refuses_a_dirty_tree_for_scientific_modes(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(fpu_provenance, "worktree_clean", lambda: False)
    manifest = tmp_path / "m.csv"
    manifest.write_text("canonical_sha1,game_idx,position_ply,side,role,"
                        "replay_path\np0,1,50,red,target,r.json\n")
    assert v17.main(["--mode", "development", "--checkpoint", CKPT,
                     "--manifest", str(manifest), "--seed-base", "1",
                     "--out", prov.OUTPUT_ROOT + "/x.json"]) == 2
    assert "clean worktree" in capsys.readouterr().out


def test_cli_refuses_a_coefficient_in_development(monkeypatch, tmp_path, capsys):
    assert _cli(monkeypatch, tmp_path, frozen_coefficient=0.25) == 2
    assert "SELECTS" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# End-to-end through run_diagnostic with an injected searcher: the real
# pipeline -- manifest -> rows -> validation -> gates -> artifact -- with no GPU
# ---------------------------------------------------------------------------

SEEDS = {"development": (20310000, 1600), "held_out": (20312000, 2200),
         "tooling_smoke": (20309000, 32)}


def _selector_chain(tmp_path, mode="development"):
    """A COMPLETE, self-consistent selector artifact chain in the real shapes:
    replays -> source index JSONL -> screen CSV -> selector config -> manifest
    (+ .meta.json sidecar) -> post-screen qualification report.

    Every recorded SHA-1 is the real hash of the file it names, so the
    diagnostic's authentication chain is exercised end to end rather than
    against a fabricated stand-in.
    """
    geometry = v17.CORPUS_GEOMETRY[mode]
    n_targets = geometry.get("target", 0)
    n_controls = geometry.get("control", 0)
    per_phase = (v17.CONTROL_PHASE_QUOTA.get(mode) or {}).get("opening", 0)

    rows, index_lines = [], []
    i = 0
    # distinct game ranges per mode: §8.1 forbids held-out sharing ANY game
    # with Stage 1, and the fixture must not violate the rule it exercises.
    game_base = {"development": 900000, "held_out": 950000,
                 "tooling_smoke": 970000}[mode]

    def add(role, bucket):
        nonlocal i
        game = game_base + i
        replay = tmp_path / f"replay_{mode}_{i}.json"
        replay.write_text(json.dumps(
            {"board_size": 24, "n_moves": 60,
             "moves": [{"row": 1, "col": 1}] * 60}))
        rows.append({"canonical_position_sha1": f"v17{mode}{i:04d}",
                     "game_idx": game, "position_ply": 50,
                     "side": "red" if i % 2 == 0 else "black",
                     "role": role, "ply_bucket": bucket, "split": mode})
        index_lines.append(json.dumps({"game_idx": game, "n_moves": 60,
                                       "winner": "red",
                                       "replay_path": str(replay)}))
        i += 1

    for _ in range(n_targets):
        add("target", "late")                      # targets are late-only
    for phase in v17.PHASES:
        for _ in range(per_phase if per_phase else n_controls // 4):
            add("control", phase)

    index = tmp_path / f"source_index_{mode}.jsonl"
    index.write_text("\n".join(index_lines) + "\n")

    cols = ["canonical_position_sha1", "game_idx", "position_ply", "side",
            "role", "ply_bucket", "split"]

    def write_csv(path, records):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in records:
                w.writerow(r)

    # the screen is a SUPERSET of the manifest, as the real selector's is
    screen = tmp_path / f"screen_{mode}.csv"
    extra = [{**rows[0], "canonical_position_sha1": f"v17{mode}extra{k}",
              "game_idx": game_base + 5000 + k} for k in range(5)]
    write_csv(screen, rows + extra)

    manifest = tmp_path / f"manifest_{mode}.csv"
    write_csv(manifest, rows)

    config = tmp_path / f"selector_config_{mode}.json"
    config.write_text(json.dumps(
        {"config_schema_version": 2, "run_kind": mode,
         "select_out": str(manifest), "screen_out": str(screen),
         "source_index_path": str(index), "checkpoint": CKPT,
         "max_per_game": 2, "min_ply_gap": 12, "side_tol": 0,
         "corpus_size": len(rows),
         "phase_allocation": v17.CONTROL_PHASE_QUOTA.get(mode, {}),
         "forbidden_manifests": list(v17.FORBIDDEN_CORPORA)}, sort_keys=True))

    prov_block = {
        "config_sha1": fpu_provenance.file_sha1(str(config)),
        "screen_csv_sha1": fpu_provenance.file_sha1(str(screen)),
        "source_index_sha1": fpu_provenance.file_sha1(str(index)),
        "replay_data_sha1": "r" * 40,
        "protocol_sha1": "p" * 40,
        "anchor_checkpoint_identity":
            f"model_iter_0001.safetensors:{fpu_provenance.file_sha1(CKPT)}",
        "forbidden_manifest_sha1s": {
            pathlib.Path(f).name: fpu_provenance.file_sha1(f)
            for f in v17.FORBIDDEN_CORPORA},
    }
    sidecar = tmp_path / (manifest.name + ".meta.json")
    sidecar.write_text(json.dumps(
        {"config_path": str(config), "source_index_path": str(index),
         "checkpoint": CKPT, "forbidden_manifests": list(v17.FORBIDDEN_CORPORA),
         "screen_csv": str(screen), "n_rows": len(rows), "run_kind": mode,
         "selection_seed": 20260725, "fieldnames": cols,
         "screen_meta_provenance": prov_block}, sort_keys=True))

    report = tmp_path / f"post_screen_{mode}.json"
    report.write_text(json.dumps(
        {"status": "PASS", "selector_error": None, "run_kind": mode,
         "config_path": str(config),
         "config_sha1": prov_block["config_sha1"],
         "screen_csv_sha1": prov_block["screen_csv_sha1"],
         "protocol_sha1": prov_block["protocol_sha1"],
         "selection_seed": 20260725, "binding_constraint": None,
         "no_manifest_written": False}, sort_keys=True))
    return {"manifest": manifest, "index": index, "screen": screen,
            "config": config, "sidecar": sidecar, "report": report}


def _manifest(tmp_path, mode="development"):
    return _selector_chain(tmp_path, mode)["manifest"]


def _source_index(tmp_path, mode="development"):
    return tmp_path / f"source_index_{mode}.jsonl"


def _qualification(tmp_path, mode, manifest_path, **over):
    """The post-screen report, optionally corrupted for a refusal test."""
    path = tmp_path / f"post_screen_{mode}.json"
    if over:
        doc = json.loads(path.read_text())
        doc.update(over)
        path = tmp_path / f"post_screen_{mode}_bad.json"
        path.write_text(json.dumps(doc, sort_keys=True))
    return str(path)


def _pair(tmp_path, mode, coefficient=None):
    base, games = SEEDS[mode]
    doc = protocol.build_protocol(run_kind=mode, coefficient=coefficient,
                                  base_seed=base, games=games,
                                  checkpoints={v17.ANCHOR_ROLE: CKPT})
    ppath, cpath = tmp_path / "p.json", tmp_path / "c.json"
    protocol.emit(ppath, doc)
    protocol.emit(cpath, protocol.derive_config(doc))
    return str(ppath), str(cpath)


def _fake_searcher(**cand):
    def search(manifest_row, coefficient):
        over = dict(cand) if (coefficient is not None and coefficient != 0.0) else {}
        return row(manifest_row["canonical_sha1"], coefficient,
                   role=manifest_row["role"], side=manifest_row["side"],
                   ply_bucket=manifest_row["ply_bucket"], **over)
    return search


def _run(tmp_path, monkeypatch, mode, **over):
    monkeypatch.setattr(prov, "OUTPUT_ROOT", str(tmp_path))
    coefficient = over.pop("frozen_coefficient", None)
    ppath, cpath = _pair(tmp_path, mode, coefficient)
    mpath = _manifest(tmp_path, mode)
    kwargs = dict(mode=mode, manifest_path=str(mpath),
                  checkpoint=CKPT, out_path=str(tmp_path / f"{mode}.json"),
                  seed_base=SEEDS[mode][0], frozen_coefficient=coefficient,
                  source_index=str(_source_index(tmp_path, mode)),
                  protocol_path=ppath, config_path=cpath,
                  qualification_report=_qualification(tmp_path, mode, mpath),
                  searcher=_fake_searcher())
    kwargs.update(over)
    return v17.run_diagnostic(**kwargs)


def test_run_diagnostic_end_to_end_emits_a_complete_artifact(clean_tree, tmp_path,
                                                             monkeypatch):
    art = _run(tmp_path, monkeypatch, "development",
               searcher=_fake_searcher(replies=10, eff_children=70.0))
    assert (tmp_path / "development.json").exists()
    assert art["n_rows"] == len(art["rows"]) == 32 * 7
    assert sorted(art["gates"]) == [str(g) for g in sorted(prov.GRID)]
    assert art["coefficient"] == 0.15
    assert art["provenance"]["identities"]["replay_data_sha1"]
    assert art["protocol_sha1"] and art["effective_mcts_config"]
    json.dumps(art, allow_nan=False)


def test_run_diagnostic_refuses_a_broken_zero_identity(clean_tree, tmp_path,
                                                       monkeypatch):
    def bad(manifest_row, coefficient):
        r = row(manifest_row["canonical_sha1"], coefficient,
                role=manifest_row["role"], side=manifest_row["side"])
        if coefficient == 0.0:
            r["selected_move"] = 99
        return rehash([r])[0]
    with pytest.raises(prov.ProtocolViolation, match="r=0 identity FAILED"):
        _run(tmp_path, monkeypatch, "development", searcher=bad)


def test_run_diagnostic_held_out_runs_one_coefficient(clean_tree, tmp_path,
                                                      monkeypatch):
    stage1 = _manifest(tmp_path, "development")
    art = _run(tmp_path, monkeypatch, "held_out", frozen_coefficient=0.25,
               stage1_manifest=str(stage1),
               searcher=_fake_searcher(replies=70))
    assert art["configs"] == [None, 0.25]
    assert art["gates"]["held_out"]["passed"] is True
    assert art["gates"]["held_out"]["metrics"]["attenuated_but_present"] is True


def test_run_diagnostic_requires_a_verified_protocol(clean_tree, tmp_path,
                                                     monkeypatch):
    monkeypatch.setattr(prov, "OUTPUT_ROOT", str(tmp_path))
    with pytest.raises(prov.ProtocolViolation, match="verified protocol"):
        v17.run_diagnostic(
            mode="development", manifest_path=str(_manifest(tmp_path)),
            checkpoint=CKPT, out_path=str(tmp_path / "o.json"),
            seed_base=1, source_index="x", searcher=_fake_searcher())


def test_run_diagnostic_refuses_wrong_manifest_geometry(clean_tree, tmp_path,
                                                        monkeypatch):
    monkeypatch.setattr(prov, "OUTPUT_ROOT", str(tmp_path))
    ppath, cpath = _pair(tmp_path, "development")
    bad = tmp_path / "bad.csv"
    bad.write_text("canonical_position_sha1,game_idx,position_ply,side,role,"
                   "ply_bucket,split\n"
                   "p0,900000,50,red,target,late,development\n")
    good = _manifest(tmp_path)
    with pytest.raises(prov.ProtocolViolation, match="corpus roles"):
        v17.run_diagnostic(
            mode="development", manifest_path=str(bad), checkpoint=CKPT,
            out_path=str(tmp_path / "o.json"), seed_base=1,
            source_index=str(_source_index(tmp_path)), protocol_path=ppath,
            config_path=cpath,
            qualification_report=_qualification(tmp_path, "development", good),
            searcher=_fake_searcher())


def test_missing_identity_is_caught_before_any_search(clean_tree, tmp_path,
                                                      monkeypatch):
    """A missing source_index previously surfaced only after all seven
    development searches had run for a position."""
    monkeypatch.setattr(prov, "OUTPUT_ROOT", str(tmp_path))
    ppath, cpath = _pair(tmp_path, "development")
    calls = []

    def counting(manifest_row, coefficient):
        calls.append((manifest_row["canonical_sha1"], coefficient))
        return row(manifest_row["canonical_sha1"], coefficient,
                   role=manifest_row["role"], side=manifest_row["side"])
    with pytest.raises(prov.ProtocolViolation, match="source_index"):
        v17.run_diagnostic(
            mode="development", manifest_path=str(_manifest(tmp_path)),
            checkpoint=CKPT, out_path=str(tmp_path / "o.json"), seed_base=1,
            source_index=None, protocol_path=ppath, config_path=cpath,
            searcher=counting)
    assert calls == []


@pytest.mark.parametrize("field", ["checkpoint", "manifest_path", "source_index"])
def test_unreadable_inputs_are_caught_in_preflight(clean_tree, tmp_path,
                                                   monkeypatch, field):
    monkeypatch.setattr(prov, "OUTPUT_ROOT", str(tmp_path))
    ppath, cpath = _pair(tmp_path, "development")
    kwargs = dict(mode="development", manifest_path=str(_manifest(tmp_path)),
                  checkpoint=CKPT, out_path=str(tmp_path / "o.json"),
                  seed_base=1, source_index=cpath, protocol_path=ppath,
                  config_path=cpath, searcher=_fake_searcher())
    kwargs[field] = str(tmp_path / "nope")
    with pytest.raises(prov.ProtocolViolation):
        v17.run_diagnostic(**kwargs)


# ---------------------------------------------------------------------------
# Stage 4: A/B/C/D against the Task 1 frozen baseline
# ---------------------------------------------------------------------------

def _frozen(gate):
    with open(v17.ABCD_BASELINE_PATH) as f:
        return json.load(f)["abcd_frozen_baseline"][gate]


def _shipped_cases(gate, **mutate):
    """A shipped run that exactly reproduces the Task 1 frozen baseline."""
    with open(v17.ABCD_MOVES_PATH) as f:
        moves = {c["case_id"]: c for c in json.load(f)["gates"][gate]["cases"]}
    cases = [{"case_id": c["case_id"],
              "black_value": float(c["probe_black_root_value_repr"]),
              "selected_move": moves[c["case_id"]]["selected_move"]}
             for c in _frozen(gate)["cases"]]
    if mutate:
        cases[0].update(mutate)
    return cases


@pytest.mark.parametrize("gate", ["A", "B", "C", "D"])
def test_shipped_baseline_reproduces_the_task1_freeze(gate):
    out = v17.verify_abcd_baseline(gate, _shipped_cases(gate))
    assert out["n"] == v17.ABCD_CARDINALITY[gate]
    assert out["max_abs_delta"] == 0.0
    assert (out["over"], out["severe"]) == (_frozen(gate)["over"],
                                            _frozen(gate)["severe"])


@pytest.mark.parametrize("gate", ["A", "B", "C", "D"])
def test_shipped_value_drift_invalidates_stage_4(gate):
    with pytest.raises(prov.ProtocolViolation, match="frozen baseline"):
        v17.verify_abcd_baseline(gate, _shipped_cases(gate, black_value=9.0))


@pytest.mark.parametrize("gate", ["A", "B", "C", "D"])
def test_shipped_selected_move_drift_invalidates_stage_4(gate):
    with pytest.raises(prov.ProtocolViolation, match="selected move"):
        v17.verify_abcd_baseline(gate, _shipped_cases(gate, selected_move=[0, 0]))


def test_truncated_shipped_run_invalidates_stage_4():
    with pytest.raises(prov.ProtocolViolation, match="cardinality"):
        v17.verify_abcd_baseline("B", _shipped_cases("B")[:17])


def _candidate_cases(gate, value):
    return [{"case_id": c["case_id"], "black_value": value,
             "selected_move": [0, 0]} for c in _frozen(gate)["cases"]]


def _a_case_rows(n=30):
    off = 0.25702582687976244
    return [{"off_value": off, "r_value": off - 0.6 * (off - v16.V_REF),
             "replies_ref": 100, "replies_x": 40, "new_collapse": False,
             "top_share_inc": 0.0} for _ in range(n)]


def test_run_abcd_applies_all_four_verdicts():
    gates = v17.run_abcd(
        coefficient=0.25,
        shipped_by_gate={g: _shipped_cases(g) for g in v17.ABCD_GATES},
        candidate_by_gate={g: _candidate_cases(g, -0.3) for g in v17.ABCD_GATES},
        a_rows=_a_case_rows())
    assert gates["all_passed"] is True
    for g in v17.ABCD_GATES:
        assert gates[g]["passed"] is True
    assert gates["baseline_validation"]["A"]["max_abs_delta"] == 0.0


def test_run_abcd_fails_when_any_single_gate_fails():
    candidate = {g: _candidate_cases(g, -0.3) for g in v17.ABCD_GATES}
    candidate["C"] = _candidate_cases("C", 0.6)
    gates = v17.run_abcd(
        coefficient=0.25,
        shipped_by_gate={g: _shipped_cases(g) for g in v17.ABCD_GATES},
        candidate_by_gate=candidate, a_rows=_a_case_rows())
    assert gates["C"]["passed"] is False
    assert gates["all_passed"] is False


def test_run_abcd_requires_all_four_gates():
    with pytest.raises(prov.ProtocolViolation, match="all four gates"):
        v17.run_abcd(coefficient=0.25,
                     shipped_by_gate={"A": _shipped_cases("A")},
                     candidate_by_gate={"A": _candidate_cases("A", -0.3)},
                     a_rows=_a_case_rows())


@pytest.mark.parametrize("bad", [None, 0.0, 0.30])
def test_run_abcd_requires_a_frozen_positive_coefficient(bad):
    with pytest.raises(prov.ProtocolViolation):
        v17.run_abcd(coefficient=bad,
                     shipped_by_gate={g: _shipped_cases(g) for g in v17.ABCD_GATES},
                     candidate_by_gate={g: _candidate_cases(g, -0.3)
                                        for g in v17.ABCD_GATES},
                     a_rows=_a_case_rows())



def test_manifest_missing_columns_refused(tmp_path):
    _manifest(tmp_path)
    bad = tmp_path / "bad.csv"
    bad.write_text("canonical_position_sha1\np0\n")
    with pytest.raises(prov.ProtocolViolation, match="missing columns"):
        v17.load_manifest(str(bad),
                          source_index=str(_source_index(tmp_path)))


# ---------------------------------------------------------------------------
# The v16 parameterization is byte-identical
# ---------------------------------------------------------------------------

LEGACY = "tests/golden/fpu_v16_dev_safety_verdicts.json"


def test_v16_dev_safety_verdicts_unchanged_by_the_parameterization():
    """66 fixtures captured BEFORE the keyword parameterization must still
    reproduce exactly -- reasons, rejected flag, and every metric."""
    with open(LEGACY) as f:
        expected = json.load(f)
    r0 = v16.FpuRunConfig("r0", 0.0)
    refs = {"absolute_off": v16.ABSOLUTE_OFF, "r0": r0}
    from importlib import import_module
    cap = import_module("tests.fpu_v16_legacy_verdict_cases")
    for key, want in expected.items():
        name, ref_name, *variant = key.split("|")
        kwargs = {}
        if variant == ["census"]:
            kwargs["include_stratum_census"] = True
        elif variant == ["ply_bucket"]:
            kwargs["stratum_key"] = "ply_bucket"
        got = v16.dev_safety_verdict(cap.CASES[name], refs[ref_name], 1, 3, **kwargs)
        assert got.rejected == want["rejected"], key
        assert list(got.reasons) == want["reasons"], key
        assert json.loads(json.dumps(got.metrics, default=str)) == want["metrics"], key


# ---------------------------------------------------------------------------
# Adversarial round 1. Each of these was ACCEPTED by the first implementation
# while 106 focused tests passed; the probe found them by asking what the layer
# accepts rather than what it rejects.
# ---------------------------------------------------------------------------

def test_selection_evaluates_exactly_the_frozen_grid():
    """The grid is not a parameter at all. It can be neither widened (§13
    forbids extending) nor narrowed -- evaluating fewer points would let a
    caller hide a passing smaller coefficient and select a larger one."""
    import inspect
    assert "grid" not in inspect.signature(v17.select_smallest_passing).parameters
    _chosen, table = v17.select_smallest_passing(
        corpus(configs=(None, 0.0) + tuple(prov.GRID)), shipped_lockin=0)
    assert sorted(table) == sorted(prov.GRID)


def test_heldout_refuses_an_off_grid_coefficient():
    """An off-grid value here would mean development selected something the
    frozen grid never contained."""
    with pytest.raises(prov.ProtocolViolation, match="frozen grid"):
        v17.heldout_verdict(corpus(configs=(None, 0.30), replies=10), 0.30,
                            shipped_lockin=0)


def test_duplicate_position_coefficient_rows_are_refused():
    """A duplicate double-counts in every aggregate."""
    rows = corpus(configs=(None, 0.35))
    rows.append(dict(rows[1]))
    with pytest.raises(prov.ProtocolViolation, match="duplicate rows"):
        v17.require_complete_pairing(rows, (None, 0.35))


@pytest.mark.parametrize("field,value", [("replies", -5), ("eff_children", -1.0)])
def test_negative_counts_are_refused_by_the_gates_directly(field, value):
    """Defence in depth: require_complete_pairing is the front door, but a gate
    called directly must still refuse rather than aggregate corrupt rows."""
    rows = corpus(configs=(None, 0.35), **{field: value})
    with pytest.raises(prov.ProtocolViolation, match="must be"):
        v17.dev_mechanism_verdict(rows, 0.35)
    with pytest.raises(prov.ProtocolViolation):
        v17.require_complete_pairing(rows, (None, 0.35))


def test_zero_shipped_denominator_is_invalid_not_an_automatic_pass():
    """§7.0 states this explicitly. It previously surfaced as a
    ZeroDivisionError from inside the imported helper."""
    rows = [row(f"p{i}", c, replies=0) for i in range(16) for c in (None, 0.35)]
    with pytest.raises(prov.ProtocolViolation, match="zero shipped denominator"):
        v17.dev_mechanism_verdict(rows, 0.35)


def test_candidate_without_a_shipped_partner_is_refused_at_the_gate():
    rows = corpus(configs=(None, 0.35))
    rows.append(row("ghost", 0.35))
    # the centralized validator catches it first, as incomplete pairing
    with pytest.raises(prov.ProtocolViolation, match="incomplete pairing"):
        v17.dev_mechanism_verdict(rows, 0.35)


@pytest.mark.parametrize("field,value", [
    ("top_share", 1.5), ("top_share", -0.1),
    ("explored_mass", 1.5), ("explored_mass", -0.1),
])
def test_shares_outside_the_unit_interval_are_refused(field, value):
    rows = corpus(configs=(None, 0.35), **{field: value})
    with pytest.raises(prov.ProtocolViolation, match="outside \\[0, 1\\]"):
        v17.require_complete_pairing(rows, (None, 0.35))


def test_boolean_replies_are_refused():
    """`bool` subclasses `int`, the same trap the protocol module had."""
    rows = corpus(configs=(None, 0.35), replies=True)
    with pytest.raises(prov.ProtocolViolation, match="must be an int"):
        v17.require_complete_pairing(rows, (None, 0.35))


def _abcd_searcher(candidate_value=-0.3):
    """Shipped reproduces the Task 1 freeze exactly; the candidate does not."""
    frozen = {g: {c["case_id"]: c for c in _frozen(g)["cases"]}
              for g in v17.ABCD_GATES}
    with open(v17.ABCD_MOVES_PATH) as f:
        moves_doc = json.load(f)["gates"]
    moves = {g: {c["case_id"]: c for c in moves_doc[g]["cases"]}
             for g in v17.ABCD_GATES}

    def search(case, coefficient):
        g, cid = case["gate"], case["case_id"]
        if coefficient is None:
            return {"case_id": cid,
                    "black_value": float(frozen[g][cid]["probe_black_root_value_repr"]),
                    "selected_move": moves[g][cid]["selected_move"],
                    "replies": 100, "top_share": 0.5, "collapse": False}
        return {"case_id": cid, "black_value": candidate_value,
                "selected_move": [0, 0], "replies": 40, "top_share": 0.5,
                "collapse": False}
    return search


def test_abcd_stage_is_operational_end_to_end(clean_tree, tmp_path, monkeypatch):
    """Finding 1: abcd previously searched and then emitted gates={}."""
    monkeypatch.setattr(prov, "OUTPUT_ROOT", str(tmp_path))
    doc = protocol.build_protocol(run_kind="abcd", coefficient=0.25,
                                  checkpoints={"anchor": CKPT})
    ppath, cpath = tmp_path / "p.json", tmp_path / "c.json"
    protocol.emit(ppath, doc)
    protocol.emit(cpath, protocol.derive_config(doc))
    art = v17.run_diagnostic(
        mode="abcd", manifest_path="", checkpoint=CKPT,
        out_path=str(tmp_path / "abcd.json"), seed_base=0,
        frozen_coefficient=0.25, protocol_path=str(ppath),
        config_path=str(cpath), searcher=_abcd_searcher())
    assert (tmp_path / "abcd.json").exists()
    assert art["gates"]["all_passed"] is True
    for g in v17.ABCD_GATES:
        assert art["gates"][g]["passed"] is True
        assert art["gates"]["baseline_validation"][g]["max_abs_delta"] == 0.0
    assert art["protocol_sha1"] and art["effective_mcts_config"]
    json.dumps(art, allow_nan=False)


def test_abcd_stage_invalidates_on_shipped_drift(clean_tree, tmp_path, monkeypatch):
    monkeypatch.setattr(prov, "OUTPUT_ROOT", str(tmp_path))
    doc = protocol.build_protocol(run_kind="abcd", coefficient=0.25,
                                  checkpoints={"anchor": CKPT})
    ppath, cpath = tmp_path / "p.json", tmp_path / "c.json"
    protocol.emit(ppath, doc)
    protocol.emit(cpath, protocol.derive_config(doc))
    base = _abcd_searcher()

    def drifting(case, coefficient):
        out = base(case, coefficient)
        if coefficient is None and case["case_id"] == _frozen("A")["cases"][0]["case_id"]:
            out["black_value"] += 1.0
        return out
    with pytest.raises(prov.ProtocolViolation, match="frozen baseline"):
        v17.run_diagnostic(mode="abcd", manifest_path="", checkpoint=CKPT,
                           out_path=str(tmp_path / "o.json"), seed_base=0,
                           frozen_coefficient=0.25, protocol_path=str(ppath),
                           config_path=str(cpath), searcher=drifting)


def test_abcd_canonical_cases_match_the_frozen_cardinalities():
    cases = v17.load_abcd_cases()
    assert {g: len(v) for g, v in cases.items()} == v17.ABCD_CARDINALITY
    for gate, rows in cases.items():
        assert all(r["seed"] and r["case_id"] for r in rows), gate


# ---------------------------------------------------------------------------
# Adversarial round 3. Each was ACCEPTED while 174 focused tests passed.
# ---------------------------------------------------------------------------

def test_verified_protocol_constrains_the_runtime_coefficient(clean_tree, tmp_path,
                                                              monkeypatch):
    """A valid r=0.25 protocol previously accompanied a runtime r=0.45."""
    monkeypatch.setattr(prov, "OUTPUT_ROOT", str(tmp_path))
    ppath, cpath = _pair(tmp_path, "held_out", 0.25)
    with pytest.raises(prov.ProtocolViolation, match="does not match the verified"):
        v17.run_diagnostic(
            mode="held_out", manifest_path=str(_manifest(tmp_path, "held_out")),
            checkpoint=CKPT, out_path=str(tmp_path / "o.json"),
            seed_base=SEEDS["held_out"][0], frozen_coefficient=0.45,
            source_index=cpath, protocol_path=ppath, config_path=cpath,
            searcher=_fake_searcher())


def test_verified_protocol_constrains_the_runtime_checkpoint(clean_tree, tmp_path,
                                                             monkeypatch):
    """Any readable checkpoint previously passed."""
    monkeypatch.setattr(prov, "OUTPUT_ROOT", str(tmp_path))
    ppath, cpath = _pair(tmp_path, "development")
    other = tmp_path / "other.safetensors"
    other.write_bytes(b"not the anchor checkpoint")
    mpath = _manifest(tmp_path)
    with pytest.raises(prov.ProtocolViolation, match="is not the protocol"):
        v17.run_diagnostic(
            mode="development", manifest_path=str(mpath),
            checkpoint=str(other), out_path=str(tmp_path / "o.json"),
            seed_base=1, source_index=cpath, protocol_path=ppath,
            config_path=cpath,
            qualification_report=_qualification(tmp_path, "development", mpath),
            searcher=_fake_searcher())


@pytest.mark.parametrize("gate", ["A", "B", "C", "D"])
def test_nan_cannot_authenticate_as_a_frozen_shipped_value(gate):
    """abs(NaN - x) is NaN and every comparison against NaN is false, so a
    corrupted below-threshold case previously authenticated cleanly."""
    with pytest.raises(prov.ProtocolViolation, match="not a finite number"):
        v17.verify_abcd_baseline(gate, _shipped_cases(gate, black_value=float("nan")))
    for bad in (float("inf"), float("-inf")):
        with pytest.raises(prov.ProtocolViolation, match="not a finite number"):
            v17.verify_abcd_baseline(gate, _shipped_cases(gate, black_value=bad))


def test_task1_freeze_is_authenticated_by_sha1(tmp_path):
    """Stage 4 may not be pointed at a rewritten baseline and rebase itself."""
    forged = tmp_path / "forged.json"
    with open(v17.ABCD_BASELINE_PATH) as f:
        doc = json.load(f)
    doc["abcd_frozen_baseline"]["A"]["over"] = 0
    forged.write_text(json.dumps(doc))
    with pytest.raises(prov.ProtocolViolation, match="may not be rewritten"):
        v17._authenticate_task1_freeze(str(forged), v17.ABCD_MOVES_PATH)
    with pytest.raises(prov.ProtocolViolation, match="may not be rewritten"):
        v17._authenticate_task1_freeze(v17.ABCD_BASELINE_PATH, str(forged))
    v17._authenticate_task1_freeze(v17.ABCD_BASELINE_PATH, v17.ABCD_MOVES_PATH)


def test_abcd_persists_every_paired_case(clean_tree, tmp_path, monkeypatch):
    """216 searches previously vanished behind four aggregate verdicts."""
    monkeypatch.setattr(prov, "OUTPUT_ROOT", str(tmp_path))
    doc = protocol.build_protocol(run_kind="abcd", coefficient=0.25,
                                  checkpoints={"anchor": CKPT})
    ppath, cpath = tmp_path / "p.json", tmp_path / "c.json"
    protocol.emit(ppath, doc)
    protocol.emit(cpath, protocol.derive_config(doc))
    art = v17.run_diagnostic(
        mode="abcd", manifest_path="", checkpoint=CKPT,
        out_path=str(tmp_path / "abcd.json"), seed_base=0,
        frozen_coefficient=0.25, protocol_path=str(ppath),
        config_path=str(cpath), searcher=_abcd_searcher())
    assert art["n_cases"] == len(art["cases"]) == 108
    for case in art["cases"]:
        assert set(case) == set(v17.ABCD_CASE_FIELDS)
        assert case["abs_delta_vs_frozen"] == 0.0
        assert case["replay_path"].endswith(".json")
    assert {c["gate"] for c in art["cases"]} == set(v17.ABCD_GATES)
    json.dumps(art, allow_nan=False)


def test_abcd_replay_identity_hashes_replays_not_probe_csvs(clean_tree, tmp_path,
                                                            monkeypatch):
    """replay_data_sha1 previously fingerprinted the four canonical CSVs while
    the replay bytes actually read stayed unbound."""
    monkeypatch.setattr(prov, "OUTPUT_ROOT", str(tmp_path))
    doc = protocol.build_protocol(run_kind="abcd", coefficient=0.25,
                                  checkpoints={"anchor": CKPT})
    ppath, cpath = tmp_path / "p.json", tmp_path / "c.json"
    protocol.emit(ppath, doc)
    protocol.emit(cpath, protocol.derive_config(doc))
    art = v17.run_diagnostic(
        mode="abcd", manifest_path="", checkpoint=CKPT,
        out_path=str(tmp_path / "abcd.json"), seed_base=0,
        frozen_coefficient=0.25, protocol_path=str(ppath),
        config_path=str(cpath), searcher=_abcd_searcher())
    replays = sorted({c["replay_path"] for c in art["cases"]})
    assert all(p.endswith(".json") and "probe_cases" not in p for p in replays)
    assert art["provenance"]["identities"]["replay_data_sha1"] == \
        fpu_provenance.replay_data_sha1(replays)
    ids = art["identities"]
    assert ids["task1_baseline_sha1"] == v17.ABCD_BASELINE_SHA1
    assert ids["task1_moves_sha1"] == v17.ABCD_MOVES_SHA1
    assert ids["config_sha1"] and ids["canonical_probe_sources"]


def test_stage4_source_identities_cover_the_probe_chain():
    for name in ("capture_v17_abcd_selected_moves.py", "probe_eval.py",
                 "opening_diagnostics.py", "game/twixt_state.py"):
        assert name in v17.RESULT_DETERMINING_MODULES


def test_paired_rows_must_share_a_seed():
    """§7 runs every coefficient 'using identical per-position seeds'."""
    rows = corpus(configs=(None, 0.35))
    for r in rows:
        if r["coefficient"] == 0.35 and r["canonical_sha1"] == "pos000":
            r["seed"] = 999
    with pytest.raises(prov.ProtocolViolation, match="seed drift"):
        v17.require_complete_pairing(rehash(rows), (None, 0.35))


# ---------------------------------------------------------------------------
# Adversarial round 4: the real Stage-4 searcher must RUN, not merely look
# right. The previous source-text assertion missed a TypeError on every case.
# ---------------------------------------------------------------------------

def test_real_abcd_searcher_executes_and_decodes_the_move(tmp_path):
    """Drives the ACTUAL searcher with a CPU stub evaluator. `top_move` is an
    encoded int, so indexing it as a pair raised TypeError on the first case."""
    from tests.fpu_search_fixture import FakeEvaluator
    from scripts.GPU.alphazero.mcts import decode_move, encode_move

    # A tiny real replay the canonical reconstruction path can read.
    moves = [{"row": 2, "col": 2}, {"row": 3, "col": 3},
             {"row": 2, "col": 4}, {"row": 4, "col": 3}]
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps({"board_size": 6, "n_moves": len(moves),
                                  "moves": moves}))
    searcher = v17._real_abcd_searcher(CKPT, evaluator=FakeEvaluator())
    case = {"gate": "A", "case_id": "c0", "seed": 1234, "position_ply": 2,
            "side_to_move": "red", "replay_path": str(replay),
            "cases_source": "x", "game_idx": 0}
    out = searcher(case, None)
    assert isinstance(out["selected_move"], list) and len(out["selected_move"]) == 2
    r, c = out["selected_move"]
    assert isinstance(r, int) and isinstance(c, int)
    assert decode_move(encode_move(r, c)) == (r, c)
    assert isinstance(out["replies"], int) and out["replies"] >= 0
    assert 0.0 <= out["top_share"] <= 1.0
    assert isinstance(out["collapse"], bool)
    assert -1.0 <= out["black_value"] <= 1.0
    # and a positive coefficient runs the same path
    assert searcher(case, 0.25)["selected_move"]


def test_reply_metric_uses_the_leader_reply_node_definition():
    """The A mechanism gate must measure visited children of the final root
    LEADER, which is what the imported v16 `_position_features` computes."""
    import inspect
    assert "_n_visited_children(top)" in inspect.getsource(v16._position_features)


# --- round-4 protocol/identity surface -------------------------------------

def test_checkpoint_binds_to_its_role_not_set_membership(tmp_path):
    """A development protocol names calib020_0001 AND 0379; searching the
    generation opponent was previously accepted."""
    other = tmp_path / "opponent.safetensors"
    other.write_bytes(b"0379 stand-in")
    cfg = {"coefficient": None,
           "checkpoints": {v17.ANCHOR_ROLE: CKPT, "opponent": str(other)}}
    v17.bind_protocol_to_runtime(cfg, coefficient=None, checkpoint=CKPT)
    with pytest.raises(prov.ProtocolViolation, match="is not the protocol"):
        v17.bind_protocol_to_runtime(cfg, coefficient=None, checkpoint=str(other))
    with pytest.raises(prov.ProtocolViolation, match="no 'anchor' checkpoint"):
        v17.bind_protocol_to_runtime({"coefficient": None,
                                      "checkpoints": {"b": str(other)}},
                                     coefficient=None, checkpoint=str(other))


def test_qualification_authenticates_the_real_selector_sidecar(tmp_path):
    m = _manifest(tmp_path)
    idx = str(_source_index(tmp_path))
    out = v17.authenticate_qualification(
        str(m), mode="development", source_index=idx, config_path=None,
        checkpoint=CKPT, post_screen_report=_qualification(tmp_path,
                                                           "development", m))
    assert out["sidecar_sha1"] and out["source_index_sha1"]
    assert out["selector_forbidden_manifests"] == list(v17.FORBIDDEN_CORPORA)


def test_missing_selector_sidecar_is_refused(tmp_path):
    """A bare manifest is not evidence that its rows were qualified."""
    m = _manifest(tmp_path)
    (tmp_path / (m.name + ".meta.json")).unlink()
    with pytest.raises(prov.ProtocolViolation, match="sidecar"):
        v17.authenticate_qualification(
            str(m), mode="development", source_index=str(_source_index(tmp_path)),
            config_path=None, checkpoint=CKPT)


def test_sidecar_must_describe_the_inputs_actually_used(tmp_path):
    m = _manifest(tmp_path)
    idx = _source_index(tmp_path)
    side = tmp_path / (m.name + ".meta.json")
    doc = json.loads(side.read_text())

    # a different source index
    other = tmp_path / "other_index.jsonl"
    other.write_text(idx.read_text())
    with pytest.raises(prov.ProtocolViolation, match="source index"):
        v17.authenticate_qualification(
            str(m), mode="development", source_index=str(other),
            config_path=None, checkpoint=CKPT)

    # a drifted source index (same name, different bytes)
    doc["screen_meta_provenance"]["source_index_sha1"] = "0" * 40
    side.write_text(json.dumps(doc, sort_keys=True))
    with pytest.raises(prov.ProtocolViolation, match="selector qualified"):
        v17.authenticate_qualification(
            str(m), mode="development", source_index=str(idx),
            config_path=None, checkpoint=CKPT)


def test_sidecar_checkpoint_must_match_the_searched_checkpoint(tmp_path):
    m = _manifest(tmp_path)
    side = tmp_path / (m.name + ".meta.json")
    doc = json.loads(side.read_text())
    other = tmp_path / "other.safetensors"
    other.write_bytes(b"not the anchor")
    doc["checkpoint"] = str(other)
    side.write_text(json.dumps(doc, sort_keys=True))
    with pytest.raises(prov.ProtocolViolation, match="qualified checkpoint"):
        v17.authenticate_qualification(
            str(m), mode="development", source_index=str(_source_index(tmp_path)),
            config_path=None, checkpoint=CKPT)


@pytest.mark.parametrize("over,pattern", [
    ({"status": "GATE_FAIL"}, "did not PASS"),
    ({"selector_error": "capacity 0 < demand 45"}, "did not PASS"),
])
def test_failed_post_screen_report_is_refused(tmp_path, over, pattern):
    m = _manifest(tmp_path)
    with pytest.raises(prov.ProtocolViolation, match=pattern):
        v17.authenticate_qualification(
            str(m), mode="development", source_index=str(_source_index(tmp_path)),
            config_path=None, checkpoint=CKPT,
            post_screen_report=_qualification(tmp_path, "development", m, **over))


def test_disjointness_is_computed_not_asserted(tmp_path):
    """An earlier draft accepted a hand-written JSON claiming zero overlaps
    against an unrelated corpus with checked_positions=0."""
    m = _manifest(tmp_path)
    rows = v17.load_manifest(str(m), source_index=str(_source_index(tmp_path)))
    clean = v17.compute_disjointness(rows, forbidden=v17.FORBIDDEN_CORPORA)
    assert clean["overlaps"] == [] and clean["checked_positions"] == 32
    assert clean["forbidden_corpora"] == list(v17.FORBIDDEN_CORPORA)
    # a row genuinely taken from a forbidden corpus is detected
    with open(v17.FORBIDDEN_CORPORA[0], newline="") as f:
        stolen = next(r for r in csv.DictReader(f) if r.get("canonical_position_sha1"))
    tainted = [{**rows[0], "canonical_sha1": stolen["canonical_position_sha1"]}] + rows[1:]
    hit = v17.compute_disjointness(tainted, forbidden=v17.FORBIDDEN_CORPORA)
    assert hit["overlaps"] and hit["overlaps"][0]["corpus"] == v17.FORBIDDEN_CORPORA[0]


def test_disjointness_requires_a_forbidden_set(tmp_path):
    rows = v17.load_manifest(str(_manifest(tmp_path)),
                             source_index=str(_source_index(tmp_path)))
    with pytest.raises(prov.ProtocolViolation, match="non-empty forbidden"):
        v17.compute_disjointness(rows, forbidden=[])


def test_real_production_manifest_loads(tmp_path):
    """The concrete integration check: the actual v16 production selector
    output must be consumable, since it is the schema Task 10 emits."""
    d = ("logs/eval/fpu_v16_policy_mass_v2/"
         "production_v2_b400amend_4000g_seed20300000")
    rows = v17.load_manifest(
        f"{d}/fpu_dev_corpus_v2_manifest.csv",
        source_index=f"{d}/calib020_0001_vs_0379_4000g_w4_seed20300000_games.jsonl")
    assert len(rows) == 120
    for r in rows:
        assert r["canonical_sha1"] == r["canonical_position_sha1"]
        assert r["replay_path"].endswith(".json")
        assert r["role"] in ("target", "control")
        assert isinstance(r["game_idx"], int)


def test_scientific_artifact_records_config_and_disjointness(clean_tree, tmp_path,
                                                             monkeypatch):
    art = _run(tmp_path, monkeypatch, "development",
               searcher=_fake_searcher(replies=10, eff_children=70.0))
    ids = art["identities"]
    assert ids["config_sha1"] and ids["source_index_sha1"]
    assert ids["qualification_report_sha1"]
    assert ids["disjointness"]["overlaps"] == []
    assert ids["disjointness"]["forbidden_corpora"]


def test_probe_sources_are_authenticated_against_the_frozen_hashes(tmp_path,
                                                                   monkeypatch):
    """An altered probe CSV must not reach the contemporaneous search."""
    with open(v17.ABCD_BASELINE_PATH) as f:
        doc = json.load(f)
    doc["abcd_frozen_baseline"]["A"]["source_sha1"] = "0" * 40
    forged = tmp_path / "b.json"
    forged.write_text(json.dumps(doc))
    monkeypatch.setattr(v17, "ABCD_BASELINE_SHA1",
                        fpu_provenance.file_sha1(str(forged)))
    with pytest.raises(prov.ProtocolViolation, match="expected the frozen"):
        v17.load_abcd_cases(baseline_path=str(forged),
                            moves_path=v17.ABCD_MOVES_PATH)


def test_b_manifest_is_authenticated_and_recorded():
    assert fpu_provenance.file_sha1(v17.B_MANIFEST_PATH) == v17.B_MANIFEST_SHA1


# ---------------------------------------------------------------------------
# Adversarial round 6: the selector chain must be REQUIRED and REVALIDATED,
# and cross-corpus identity must be canonical, not reservoir-local.
# ---------------------------------------------------------------------------

def test_post_screen_report_is_required(tmp_path):
    """It was optional, so omitting it authenticated nothing."""
    m = _manifest(tmp_path)
    with pytest.raises(prov.ProtocolViolation, match="requires the selector"):
        v17.authenticate_qualification(
            str(m), mode="development", source_index=str(_source_index(tmp_path)),
            config_path=None, checkpoint=CKPT, post_screen_report=None)


def test_hand_written_pass_report_is_refused(tmp_path):
    """A minimal {"status":"PASS"} document is not the selector's report."""
    m = _manifest(tmp_path)
    forged = tmp_path / "forged_report.json"
    forged.write_text(json.dumps({"status": "PASS", "selector_error": None}))
    with pytest.raises(prov.ProtocolViolation, match="hand-written PASS"):
        v17.authenticate_qualification(
            str(m), mode="development", source_index=str(_source_index(tmp_path)),
            config_path=None, checkpoint=CKPT, post_screen_report=str(forged))


def test_tampered_selector_config_is_refused(tmp_path):
    """Readability is not authentication: the bytes must match config_sha1."""
    chain = _selector_chain(tmp_path)
    chain["config"].write_text(json.dumps({"tampered": True}))
    with pytest.raises(prov.ProtocolViolation, match="selector config"):
        v17.authenticate_qualification(
            str(chain["manifest"]), mode="development",
            source_index=str(chain["index"]), config_path=None, checkpoint=CKPT,
            post_screen_report=str(chain["report"]))


def test_tampered_screen_is_refused(tmp_path):
    chain = _selector_chain(tmp_path)
    with open(chain["screen"], "a") as f:
        f.write("extra,1,1,red,control,late,development\n")
    with pytest.raises(prov.ProtocolViolation, match="screen"):
        v17.authenticate_qualification(
            str(chain["manifest"]), mode="development",
            source_index=str(chain["index"]), config_path=None, checkpoint=CKPT,
            post_screen_report=str(chain["report"]))


def test_manifest_is_bound_by_screen_membership(tmp_path):
    """The selector records no manifest hash, so an edited manifest beside a
    genuine sidecar was indistinguishable. Fabricated rows are not in the
    hash-pinned screen."""
    chain = _selector_chain(tmp_path)
    rows = list(csv.DictReader(open(chain["manifest"], newline="")))
    rows[0]["canonical_position_sha1"] = "fabricated_position"
    with open(chain["manifest"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(prov.ProtocolViolation, match="absent from the"):
        v17.authenticate_qualification(
            str(chain["manifest"]), mode="development",
            source_index=str(chain["index"]), config_path=None, checkpoint=CKPT,
            post_screen_report=str(chain["report"]))


def test_disjointness_uses_canonical_state_identity_not_game_index():
    """Game indices are reservoir-local. An exact A-probe position reached from
    a different reservoir must still collide."""
    from scripts.GPU.alphazero.fpu_state_hash import canonical_state_sha1
    from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
    a_csv = v17.FORBIDDEN_CORPORA[2]
    case = next(r for r in csv.DictReader(open(a_csv, newline=""))
                if r["checkpoint"] == "0001")
    replay = json.loads(open(case["replay_path"]).read())
    state = position_state(replay, int(case["position_ply"]), case["side_to_move"])
    exact = canonical_state_sha1(state)
    # same position, DIFFERENT game index -- the old pair-identity missed this
    out = v17.compute_disjointness(
        [{"canonical_sha1": exact, "game_idx": 12345678, "position_ply": 999}],
        forbidden=[a_csv])
    assert out["overlaps"], "an exact canonical collision was missed"
    assert out["overlaps"][0]["canonical_sha1"] == [exact]


def test_gate_b_hashes_join_its_authenticated_replay_manifest():
    """B's cases carry no replay_path; its manifest supplies the join."""
    hashes = v17.forbidden_position_hashes(v17.FORBIDDEN_CORPORA[3])
    assert len(hashes) == v17.ABCD_CARDINALITY["B"]
    assert all(len(h) == 40 for h in hashes)


def test_held_out_requires_stage1_disjointness(clean_tree, tmp_path, monkeypatch):
    """§8.1: complete GAME and position disjointness from Stage 1."""
    monkeypatch.setattr(prov, "OUTPUT_ROOT", str(tmp_path))
    with pytest.raises(prov.ProtocolViolation, match="Stage-1 development"):
        _run(tmp_path, monkeypatch, "held_out", frozen_coefficient=0.25,
             searcher=_fake_searcher(replies=70))


def test_held_out_sharing_a_stage1_game_is_refused(clean_tree, tmp_path,
                                                   monkeypatch):
    stage1 = _manifest(tmp_path, "development")
    rows = list(csv.DictReader(open(stage1, newline="")))
    held = _selector_chain(tmp_path, "held_out")
    held_rows = list(csv.DictReader(open(held["manifest"], newline="")))
    held_rows[0]["game_idx"] = rows[0]["game_idx"]        # a shared game
    with open(held["manifest"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(held_rows[0]))
        w.writeheader()
        w.writerows(held_rows)
    out = v17.compute_disjointness(
        [{"canonical_sha1": r["canonical_position_sha1"],
          "game_idx": int(r["game_idx"]), "position_ply": 50} for r in held_rows],
        forbidden=[v17.FORBIDDEN_CORPORA[0]],
        forbidden_games={"stage1": [int(r["game_idx"]) for r in rows]})
    assert out["game_overlaps"]


@pytest.mark.parametrize("mode,per_phase", [("development", 4), ("held_out", 8)])
def test_control_phase_quotas_are_enforced(mode, per_phase):
    assert v17.CONTROL_PHASE_QUOTA[mode] == {p: per_phase for p in v17.PHASES}
    rows = _manifest_rows(mode)
    v17.require_corpus_geometry(mode, rows)
    skewed = [dict(r) for r in rows]
    for r in skewed:
        if r["role"] == "control" and r["ply_bucket"] == "opening":
            r["ply_bucket"] = "late"
            break
    with pytest.raises(prov.ProtocolViolation, match="phase quotas"):
        v17.require_corpus_geometry(mode, skewed)


def test_manifest_without_a_phase_column_is_refused(tmp_path):
    assert "ply_bucket" in v17.MANIFEST_REQUIRED_COLUMNS
    bad = tmp_path / "nophase.csv"
    bad.write_text("canonical_position_sha1,game_idx,position_ply,side,role\n"
                   "p0,1,50,red,target\n")
    _selector_chain(tmp_path)
    with pytest.raises(prov.ProtocolViolation, match="ply_bucket"):
        v17.load_manifest(str(bad), source_index=str(_source_index(tmp_path)))


def test_control_row_with_an_unknown_phase_is_refused():
    rows = [dict(r) for r in _manifest_rows("development")]
    for r in rows:
        if r["role"] == "control":
            r["ply_bucket"] = "endgame"
            break
    with pytest.raises(prov.ProtocolViolation, match="ply_bucket"):
        v17.require_corpus_geometry("development", rows)


@pytest.mark.parametrize("module", ["build_fpu_dev_corpus.py", "fpu_state_hash.py"])
def test_imported_loaders_are_in_source_provenance(module):
    """load_manifest imports load_game_index and disjointness imports
    load_forbidden_hashes / canonical_state_sha1, so those modules determine
    the result and must be hashed."""
    assert module in v17.RESULT_DETERMINING_MODULES


def test_no_shadowed_definitions_in_the_v17_modules():
    """A duplicated top-level def silently shadows the earlier one, so the
    module runs code that is not the code under review. This bit twice during
    Task 5 -- once in the module, once in this test file -- and both times the
    focused suite still passed against the stale implementation.
    """
    import ast
    import pathlib
    for path in ("scripts/GPU/alphazero/diagnose_fpu_baseline_policy_mass.py",
                 "scripts/GPU/alphazero/fpu_v17_protocol.py",
                 "scripts/GPU/alphazero/fpu_v17_provenance.py",
                 __file__):
        tree = ast.parse(pathlib.Path(path).read_text())
        names = [n.name for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        dupes = sorted({n for n in names if names.count(n) > 1})
        assert not dupes, f"{path} shadows {dupes}"


def test_consumed_artifact_inventory_is_documented():
    """Every artifact this stage consumes from another tool must be named in
    the module docstring with its producer and its authenticator, so the
    integration surface is reviewable without reading the whole module."""
    doc = v17.__doc__
    for artifact in ("selection manifest", "sidecar", "selector config",
                     "post-screen", "source index", "replay", "probe CSV",
                     "goal-line manifest", "Task 1 freeze"):
        assert artifact in doc, artifact


def test_no_frozen_value_is_defined_twice():
    """Every frozen value must have exactly ONE definition, so a change is a
    one-place change. Checks the module surface for a second literal copy."""
    import pathlib
    az = pathlib.Path("scripts/GPU/alphazero")
    sources = {p.name: p.read_text() for p in az.glob("*.py")
               if p.name.startswith(("fpu_v17_", "diagnose_fpu_baseline"))}
    joined = "\n".join(sources.values())
    # the frozen design hash: one definition, no literal repeats
    assert joined.count(prov.FROZEN_DESIGN_SHA1) == 1
    # over/severe thresholds are referenced, never re-typed
    body = sources["diagnose_fpu_baseline_policy_mass.py"]
    assert "v >= 0.25" not in body and "v >= 0.50" not in body
    assert "OVER_THRESHOLD" in body and "SEVERE_THRESHOLD" in body


def test_capture_tool_triple_matches_the_frozen_constants():
    """`capture_v17_abcd_selected_moves.py` declares its own batching literals.

    It is deliberately NOT edited to import them: its bytes are recorded as
    `capture_tool_sha1` in the immutable Task 1 baseline, so changing the file
    would invalidate authenticated evidence for no scientific gain. Guarding
    instead means a future divergence is caught without touching frozen code.
    """
    from scripts.GPU.alphazero import capture_v17_abcd_selected_moves as cap
    assert (cap.EVAL_BATCH_SIZE, cap.STALL_FLUSH_SIMS,
            cap.PENDING_VIRTUAL_VISITS) == prov.BATCHING
    assert cap.MCTS_SIMS == prov.MCTS_SIMS
    assert cap.CHECKPOINT_ROW_FILTER == "0001"
