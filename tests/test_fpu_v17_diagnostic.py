"""v17 Task 5 -- modes, pure gates, selection, and artifact identities.

Frozen design ref:
`docs/superpowers/specs/2026-07-24-v17-baseline-preserving-policy-mass-fpu-design.md`
(SHA-1 `944f358c0e3ef66503d2cbb56e31dabd145bafc2`) §§7-9.

Every fixture is fabricated, so each threshold is pinned on BOTH sides of its
boundary -- the value that must pass and the neighbouring value that must fail.
No GPU, no evaluator, no checkpoint weights.
"""
import json

import pytest

from scripts.GPU.alphazero import diagnose_fpu_baseline_policy_mass as v17
from scripts.GPU.alphazero import diagnose_fpu_policy_mass as v16
from scripts.GPU.alphazero import fpu_provenance
from scripts.GPU.alphazero import fpu_v17_provenance as prov

CKPT = "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"
SRC = "scripts/GPU/alphazero/mcts.py"


@pytest.fixture
def clean_tree(monkeypatch):
    monkeypatch.setattr(fpu_provenance, "worktree_clean", lambda: True)


# ---------------------------------------------------------------------------
# Row fabrication
# ---------------------------------------------------------------------------

def row(sha, coefficient, role="target", **over):
    base = dict(canonical_sha1=sha, role=role, coefficient=coefficient,
                selected_move=1, selected_prior=0.02, selected_prior_rank=1,
                root_value_stm=0.0, parent_value=0.0, top_share=0.5,
                eff_children=100.0, replies=100, collapse=False, lock_in=False,
                explored_mass=0.25, stabilization_sim=200, complete=True)
    base.update(over)
    return base


def corpus(n_targets=16, n_controls=16, configs=(None, 0.0), **cand):
    """One shipped + one row per config per position. `cand` overrides are
    applied only to NON-shipped, non-zero rows."""
    rows = []
    for i in range(n_targets + n_controls):
        sha = f"pos{i:03d}"
        role = "target" if i < n_targets else "control"
        for c in configs:
            over = dict(cand) if (c is not None and c != 0.0) else {}
            rows.append(row(sha, c, role=role, **over))
    return rows


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
        return rows
    assert not _safety(flipped(1)).rejected
    assert any("control_flip_rate" in x for x in _safety(flipped(2)).reasons)


def test_lockin_margin_is_one_for_development_and_two_for_heldout():
    rows = corpus(configs=(None, 0.35))
    n = 0
    for r in rows:
        if r["coefficient"] == 0.35 and r["role"] == "target" and n < 2:
            r["lock_in"] = True
            n += 1
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
        return rows
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
        return rows
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
    return rows


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


def test_heldout_floor_at_exactly_twenty_percent_by_count_is_ulp_fragile():
    """Documents real, faithful behaviour rather than papering over it.

    §7.0 freezes the aggregate as `1 - sum(candidate)/sum(shipped)`. At an
    exactly-20%-by-count reduction that evaluates to 0.19999999999999996 in
    IEEE754, just below the §8.2 floor, so the gate REJECTS. The 50% gate is
    unaffected because 0.5 is exactly representable. This is not changed here:
    the formula is frozen, and silently widening it would be a deviation.
    """
    assert 1 - 80 / 100 < v17.HELDOUT_REPLY_REDUCTION
    assert 1 - 50 / 100 >= v17.DEV_REPLY_REDUCTION
    at_count_boundary = corpus(configs=(None, 0.25), replies=80)
    assert not v17.heldout_verdict(at_count_boundary, 0.25,
                                   shipped_lockin=0).passed


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


def test_a_gate_passes_only_when_every_criterion_holds():
    ok = v17.abcd_verdict("A", n=30, mean=-0.1, over=3, severe=5,
                          a_rows=_a_rows())
    assert ok.passed


@pytest.mark.parametrize("kwargs,expect", [
    (dict(mean=0.01, severe=5), "A_mean"),
    (dict(mean=-0.1, severe=6), "A_severe"),
])
def test_a_gate_count_criteria(kwargs, expect):
    res = v17.abcd_verdict("A", n=30, over=3, a_rows=_a_rows(), **kwargs)
    assert any(expect in x for x in res.reasons)


def test_a_severe_boundary_is_five_not_six():
    """§9 was corrected pre-freeze from <=6/30 to <=5/30 (v14b parity)."""
    assert v17.abcd_verdict("A", n=30, mean=-0.1, over=3, severe=5,
                            a_rows=_a_rows()).passed
    assert not v17.abcd_verdict("A", n=30, mean=-0.1, over=3, severe=6,
                                a_rows=_a_rows()).passed


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
    assert v17.abcd_verdict(gate, **ok).passed
    assert not v17.abcd_verdict(gate, **bad).passed


def test_a_gate_requires_case_rows():
    with pytest.raises(prov.ProtocolViolation, match="per-case rows"):
        v17.abcd_verdict("A", n=30, mean=-0.1, over=3, severe=5)


def test_unknown_abcd_gate_refused():
    with pytest.raises(prov.ProtocolViolation, match="unknown A/B/C/D gate"):
        v17.abcd_verdict("E", n=1, mean=0.0, over=0, severe=0)


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
    kwargs = dict(mode=mode, coefficient=0.25, rows=[], gates={},
                  checkpoints={"a": CKPT}, manifest=str(manifest),
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


@pytest.mark.parametrize("null", ["manifest", "source_index", "replay_paths",
                                  "checkpoints", "source_files"])
@pytest.mark.parametrize("mode", ["development", "held_out", "abcd"])
def test_scientific_artifact_refuses_a_null_identity(clean_tree, tmp_path, mode, null):
    with pytest.raises(prov.ProtocolViolation, match="must populate"):
        _artifact(mode, tmp_path, **{null: None if null != "checkpoints" else {}})


def test_tooling_smoke_artifact_may_omit_identities(clean_tree, tmp_path):
    art = v17.build_artifact(mode="tooling_smoke", coefficient=0.35, rows=[],
                             gates={}, checkpoints={})
    assert art["provenance"]["scientific_interpretation_forbidden"] is True


def test_artifact_is_timestamp_free(clean_tree, tmp_path):
    blob = json.dumps(_artifact("held_out", tmp_path))
    for banned in ("timestamp", "generated_at", "datetime"):
        assert banned not in blob


# ---------------------------------------------------------------------------
# CLI / config refusals
# ---------------------------------------------------------------------------

def _cli(monkeypatch, **over):
    monkeypatch.setattr(fpu_provenance, "worktree_clean", lambda: True)
    args = ["--mode", "development", "--checkpoint", CKPT,
            "--out-dir", prov.OUTPUT_ROOT + "/x"]
    for k, v in over.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    return v17.main(args)


def test_cli_accepts_the_frozen_configuration(monkeypatch):
    assert _cli(monkeypatch) == 0


@pytest.mark.parametrize("override", [
    {"eval_batch_size": 16}, {"stall_flush_sims": 16}, {"stall_flush_sims": 0},
])
def test_cli_refuses_free_batching_overrides(monkeypatch, override, capsys):
    """§2.4: free CLI overrides are rejected, not honoured."""
    assert _cli(monkeypatch, **override) == 2
    assert "batching" in capsys.readouterr().out


def test_cli_refuses_a_v16_output_root(monkeypatch, capsys):
    assert _cli(monkeypatch, out_dir="logs/eval/fpu_v16_policy_mass_v2") == 2
    assert "v17 root" in capsys.readouterr().out


def test_cli_refuses_a_dirty_tree_for_scientific_modes(monkeypatch, capsys):
    monkeypatch.setattr(fpu_provenance, "worktree_clean", lambda: False)
    args = ["--mode", "development", "--checkpoint", CKPT,
            "--out-dir", prov.OUTPUT_ROOT + "/x"]
    assert v17.main(args) == 2
    assert "clean worktree" in capsys.readouterr().out


def test_cli_refuses_a_coefficient_in_development(monkeypatch, capsys):
    assert _cli(monkeypatch, frozen_coefficient=0.25) == 2
    assert "SELECTS" in capsys.readouterr().out


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

def test_selection_grid_cannot_be_widened():
    """§13 forbids extending the grid. The `grid` argument exists so a test can
    exercise a subset, never so a caller can add a coefficient."""
    rows = corpus(configs=(None, 0.0, 0.55))
    with pytest.raises(prov.ProtocolViolation, match="outside the frozen grid"):
        v17.select_smallest_passing(rows, shipped_lockin=0, grid=[0.55])
    with pytest.raises(prov.ProtocolViolation, match="outside the frozen grid"):
        v17.select_smallest_passing(rows, shipped_lockin=0, grid=[0.25, 0.30])
    # a genuine subset is still fine
    v17.select_smallest_passing(corpus(configs=(None, 0.25)), shipped_lockin=0,
                                grid=[0.25])


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
    with pytest.raises(prov.ProtocolViolation, match="negative"):
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
    with pytest.raises(prov.ProtocolViolation, match="without a shipped partner"):
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
    with pytest.raises(prov.ProtocolViolation, match="non-negative int"):
        v17.require_complete_pairing(rows, (None, 0.35))
