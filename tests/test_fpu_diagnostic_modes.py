"""Task 7 pure tests: typed FPU run-configs, exact stage+mode config sets,
the frozen §6 gate boundaries (executable pre-registration), and the
controls-artifact fingerprint / r0-qualification guards.

NO MCTS / checkpoint / real corpus is touched here -- every test drives a pure
function or a FABRICATED dict. `main(--mode,--stage)` (the 400-sim operator
phase) is never invoked. Frozen refs: design §5/§6
(docs/superpowers/specs/2026-07-10-context-relative-fpu-policy-mass-design.md),
plan Task 7 (brief `.superpowers/sdd/task-7-brief.md`).

The `_dev_rejects` / `_a_passes` builders below are THIN: each constructs a
MINIMAL `rows` set isolating exactly one gate metric at a given value and calls
the module verdict fn. To realise a rate boundary exactly the builders pick the
denominator the brief pins (0.05=5/100, 0.0499=499/10000, band 0.10=2/20 /
0.0999=999/10000, control-flip 0.10=1/10 / 0.099=99/1000). The band case
additionally dilutes with a second all-clean band so the OVERALL target rate
independently stays < 0.05 -- this isolates the per-band gate from the
overall target-rate gate (band A alone at 2/20==0.10 would otherwise ALSO
trip target_new_collapse_rate>=0.05, so the boundary assertion would not
actually pin the band-gate comparator). For the p95 boundary every |delta|
is made identical, so the pinned value is the p95 under ANY percentile
convention (the gate fn uses the linear-interpolation `_percentile` shared
with diagnose_fpu_sweep).
"""
import json
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.GPU.alphazero import build_fpu_dev_corpus as bfdc
from scripts.GPU.alphazero import diagnose_fpu_policy_mass as dfpm
from scripts.GPU.alphazero import fpu_provenance as prov
from scripts.GPU.alphazero.diagnose_fpu_policy_mass import (
    FpuRunConfig, ABSOLUTE_OFF, R0, GRID, validate_stage_mode,
    lock_in_event, progress, reply_reduction, prior_rank,
    dev_safety_verdict, selected_a_verdict)
from scripts.GPU.alphazero.diagnose_fpu_policy_mass import (
    V_REF, top_share, validate_controls_fingerprint, require_r0_qualified,
    require_matching_mode, _leader_child)
from scripts.GPU.alphazero.diagnose_fpu_policy_mass import (
    CANDIDATE_DEV_ROW_FIELDNAMES, _dev_target_row, _dev_control_row,
    _dev_rows_vs, _candidate_dev_records)
from scripts.GPU.alphazero.diagnose_fpu_policy_mass import (
    _resolve_v2_stratum, run_controls_stage, run_candidates_stage)
from scripts.GPU.alphazero.mcts import MCTSNode, encode_move, visit_leader_move


# ---------------------------------------------------------------------------
# Typed configs (fix 9)
# ---------------------------------------------------------------------------

def test_labels_and_grid_are_explicit_and_positive():
    assert [c.label for c in GRID] == ["r0.10", "r0.20", "r0.35", "r0.50", "r0.75"]
    assert [c.reduction for c in GRID] == [0.10, 0.20, 0.35, 0.50, 0.75]
    assert ABSOLUTE_OFF.reduction is None and R0.reduction == 0.0 and ABSOLUTE_OFF != R0


def test_configs_are_frozen_and_hashable():
    # frozen dataclass -> value equality + hashable (needed for the exact-set check)
    assert FpuRunConfig("r0.20", 0.20) == GRID[1]
    assert len({ABSOLUTE_OFF, R0, *GRID}) == 7
    with pytest.raises(Exception):
        ABSOLUTE_OFF.reduction = 0.5   # frozen


# ---------------------------------------------------------------------------
# validate_stage_mode -- EXACT sets (fix 3)
# ---------------------------------------------------------------------------

def test_stage_mode_exact_sets():
    tun = [{"split": "tuning"}]; frz = [{"split": "frozen_check"}]
    validate_stage_mode(tun, mode="tuning", stage="controls", run_configs=[ABSOLUTE_OFF, R0])
    validate_stage_mode(tun, mode="tuning", stage="candidates", run_configs=list(GRID))
    validate_stage_mode(frz, mode="frozen_check", stage="controls", run_configs=[ABSOLUTE_OFF, R0])
    validate_stage_mode(frz, mode="frozen_check", stage="candidates",
                        run_configs=[FpuRunConfig("r0.20", 0.20)])
    for bad in (
        dict(cases=frz, mode="tuning", stage="controls", run_configs=[ABSOLUTE_OFF, R0]),   # wrong split
        dict(cases=tun, mode="tuning", stage="controls", run_configs=[ABSOLUTE_OFF]),        # not exact set
        dict(cases=tun, mode="tuning", stage="candidates", run_configs=[ABSOLUTE_OFF]+list(GRID)),  # superset
        dict(cases=tun, mode="tuning", stage="candidates", run_configs=list(GRID)[:-1]),     # subset
        dict(cases=frz, mode="frozen_check", stage="candidates",
             run_configs=[FpuRunConfig("r0.20", 0.20), FpuRunConfig("r0.35", 0.35)]),        # >1 nonzero
        dict(cases=frz, mode="frozen_check", stage="candidates", run_configs=[R0]),          # zero-r not allowed
        dict(cases=tun, mode="tuning", stage="bogus", run_configs=[ABSOLUTE_OFF, R0]),        # bad stage
    ):
        with pytest.raises(ValueError):
            validate_stage_mode(**bad)


def test_stage_mode_rejects_missing_split_field():
    with pytest.raises(ValueError):
        validate_stage_mode([{"role": "target"}], mode="tuning", stage="controls",
                            run_configs=[ABSOLUTE_OFF, R0])


# ---------------------------------------------------------------------------
# §6.0 formula exactness
# ---------------------------------------------------------------------------

def test_formula_exactness():
    assert abs(progress(0.30, 0.13) - (0.30 - 0.13) / (0.30 - (-0.0451))) < 1e-9
    assert abs(reply_reduction(200, 100) - 0.5) < 1e-9
    assert prior_rank({1: 0.5, 2: 0.3, 3: 0.3}, 2) == 2


def test_prior_rank_strictly_greater():
    # top move -> rank 1; ties never inflate the rank (strictly-greater count)
    assert prior_rank({1: 0.5, 2: 0.3, 3: 0.3}, 1) == 1
    assert prior_rank({1: 0.4, 2: 0.4, 3: 0.4}, 3) == 1     # all tie -> nobody is strictly greater
    assert prior_rank({1: 0.9, 2: 0.05, 3: 0.05}, 2) == 2


def test_top_share_of_node():
    class _N:
        def __init__(self, v): self.visit_count = v
    class _Root:
        visit_count = 10
        children = {1: _N(7), 2: _N(3)}
    assert abs(top_share(_Root()) - 0.7) < 1e-12


# ---------------------------------------------------------------------------
# §6.1 lock-in event + §6.2/§6.3 gate boundaries -- the executable
# pre-registration (fix 8). `_dev_rejects` / `_a_passes` isolate ONE metric.
# ---------------------------------------------------------------------------

_LOCKIN_BASE = 5   # arbitrary fixed lock-in baseline for isolation


def _safe_target(**over):
    row = dict(role="target", band="b200_299", new_collapse=False, lock_in=False,
               mover_delta=0.0, eff_children_reduction=0.0, top_share_inc=0.0)
    row.update(over)
    return row


def _safe_control(**over):
    row = dict(role="control", mover_delta=0.0, control_flip_to_lower_prior=False)
    row.update(over)
    return row


def _rejects(rows):
    return dev_safety_verdict(rows, ref=R0,
                              r0_lockin=_LOCKIN_BASE, absoff_lockin=_LOCKIN_BASE).rejected


def _dev_rejects(**metric):
    """Build a minimal rows set isolating the single named metric at its value,
    then return whether dev_safety_verdict rejects. Exactly one metric per call
    (the compound gate takes the eff_reduction+top_share_inc PAIR)."""
    if "new_collapse_rate" in metric:
        num, den = {0.05: (5, 100), 0.0499: (499, 10000)}[metric["new_collapse_rate"]]
        rows = [_safe_target(band="solo", new_collapse=(i < num)) for i in range(den)]
        return _rejects(rows)          # single band, both values < 0.10 so per-band never fires
    if "band_new_collapse_rate" in metric:
        x = metric["band_new_collapse_rate"]
        # Isolate the BAND gate from the independent overall target-rate gate:
        # band A alone carries the pinned rate (n>=20 so the band gate is
        # active); band B is a second, all-clean band sized so the OVERALL
        # target new-collapse rate stays < 0.05 even at x==0.10. Without this
        # dilution, band A alone (2/20==0.10) would ALSO trip
        # target_new_collapse_rate>=0.05 on its own, so the boundary
        # assertion would still pass even if the band gate were deleted.
        num, den, dilution = {0.10: (2, 20, 21), 0.0999: (999, 10000, 10000)}[x]
        rows = [_safe_target(band="A", new_collapse=(i < num)) for i in range(den)]
        rows += [_safe_target(band="B", new_collapse=False) for _ in range(dilution)]
        return _rejects(rows)
    if "lockin_count" in metric:
        count = metric["lockin_count"](_LOCKIN_BASE)
        rows = [_safe_target(lock_in=True) for _ in range(count)]
        return _rejects(rows)
    if "p95_mover_delta" in metric:
        x = metric["p95_mover_delta"]
        rows = [_safe_target(mover_delta=x) for _ in range(8)]   # identical -> p95 == x
        return _rejects(rows)
    if "eff_reduction" in metric:      # compound gate: eff_reduction AND top_share_inc
        rows = [_safe_target(eff_children_reduction=metric["eff_reduction"],
                             top_share_inc=metric["top_share_inc"]) for _ in range(8)]
        return _rejects(rows)
    if "control_lowprior_flip_rate" in metric:
        num, den = {0.10: (1, 10), 0.099: (99, 1000)}[metric["control_lowprior_flip_rate"]]
        rows = [_safe_control(control_flip_to_lower_prior=(i < num)) for i in range(den)]
        return _rejects(rows)
    raise AssertionError(f"unknown metric {metric!r}")


def _a_passes(*, reply_reduction, progress, a_new_collapse, a_top_share_inc):
    """Build A rows realising the four aggregate quantities exactly, then return
    whether selected_a_verdict passes. replies cancel across n identical rows;
    r_value = progress*V_REF reproduces the target progress with V_off = 0."""
    n = max(int(a_new_collapse), 1)
    rows = [dict(off_value=0.0, r_value=progress * V_REF,
                 replies_ref=1.0, replies_x=1.0 - reply_reduction,
                 top_share_inc=a_top_share_inc, new_collapse=(i < a_new_collapse))
            for i in range(n)]
    return selected_a_verdict(rows).passed


def test_lock_in_event_boundaries():
    assert lock_in_event(dict(selected_move_prior_rank=11, selected_move_prior=0.005,
        explored_mass_at_stabilization=0.20, stabilization_sim=80, final_root_top_share=0.95))
    assert not lock_in_event(dict(selected_move_prior_rank=10, selected_move_prior=0.005,
        explored_mass_at_stabilization=0.20, stabilization_sim=80, final_root_top_share=0.95))
    # each remaining field just across its edge also disqualifies
    assert not lock_in_event(dict(selected_move_prior_rank=11, selected_move_prior=0.01,
        explored_mass_at_stabilization=0.20, stabilization_sim=80, final_root_top_share=0.95))
    assert not lock_in_event(dict(selected_move_prior_rank=11, selected_move_prior=0.005,
        explored_mass_at_stabilization=0.25, stabilization_sim=80, final_root_top_share=0.95))
    assert not lock_in_event(dict(selected_move_prior_rank=11, selected_move_prior=0.005,
        explored_mass_at_stabilization=0.20, stabilization_sim=101, final_root_top_share=0.95))
    assert not lock_in_event(dict(selected_move_prior_rank=11, selected_move_prior=0.005,
        explored_mass_at_stabilization=0.20, stabilization_sim=80, final_root_top_share=0.89))


def test_gate_boundaries_table_driven():          # fix 8 -- the executable preregistration
    assert lock_in_event(dict(selected_move_prior_rank=11, selected_move_prior=0.005,
        explored_mass_at_stabilization=0.20, stabilization_sim=80, final_root_top_share=0.95))
    assert not lock_in_event(dict(selected_move_prior_rank=10, selected_move_prior=0.005,
        explored_mass_at_stabilization=0.20, stabilization_sim=80, final_root_top_share=0.95))
    # dev-safety: exact reject boundaries
    assert _dev_rejects(new_collapse_rate=0.05)      and not _dev_rejects(new_collapse_rate=0.0499)
    assert _dev_rejects(band_new_collapse_rate=0.10) and not _dev_rejects(band_new_collapse_rate=0.0999)
    assert _dev_rejects(lockin_count=lambda base: base + 3) and not _dev_rejects(lockin_count=lambda base: base + 2)
    assert _dev_rejects(p95_mover_delta=0.35)        and not _dev_rejects(p95_mover_delta=0.349)
    assert _dev_rejects(eff_reduction=0.50, top_share_inc=0.15)          # compound: both
    assert not _dev_rejects(eff_reduction=0.50, top_share_inc=0.14)
    assert _dev_rejects(control_lowprior_flip_rate=0.10) and not _dev_rejects(control_lowprior_flip_rate=0.099)
    # selected-A: exact pass boundaries
    assert _a_passes(reply_reduction=0.50, progress=0.50, a_new_collapse=2, a_top_share_inc=0.15)
    assert not _a_passes(reply_reduction=0.49, progress=0.50, a_new_collapse=2, a_top_share_inc=0.15)
    assert not _a_passes(reply_reduction=0.50, progress=0.49, a_new_collapse=2, a_top_share_inc=0.15)
    assert not _a_passes(reply_reduction=0.50, progress=0.50, a_new_collapse=3, a_top_share_inc=0.15)
    assert not _a_passes(reply_reduction=0.50, progress=0.50, a_new_collapse=2, a_top_share_inc=0.16)


def test_dev_safety_verdict_reports_reasons():
    v = dev_safety_verdict(
        [_safe_target(band="solo", new_collapse=(i < 5)) for i in range(100)],
        ref=R0, r0_lockin=_LOCKIN_BASE, absoff_lockin=_LOCKIN_BASE)
    assert v.rejected and v.reasons               # reasons non-empty on reject
    clean = dev_safety_verdict([_safe_target() for _ in range(30)],
                               ref=R0, r0_lockin=_LOCKIN_BASE, absoff_lockin=_LOCKIN_BASE)
    assert not clean.rejected and not clean.reasons


def test_dev_safety_lockin_baseline_follows_ref():
    # r=0.0 is gated vs absolute_off (baseline = absoff_lockin); a candidate vs
    # r0 uses r0_lockin. Same rows, different ref -> different baseline picked.
    rows = [_safe_target(lock_in=True) for _ in range(6)]     # 6 lock-ins
    assert dev_safety_verdict(rows, ref=ABSOLUTE_OFF, r0_lockin=10, absoff_lockin=3).rejected  # 6 > 3+2
    assert not dev_safety_verdict(rows, ref=R0, r0_lockin=10, absoff_lockin=3).rejected        # 6 <= 10+2


# ---------------------------------------------------------------------------
# Controls-artifact guards (driven by a FABRICATED controls_gate.json)
# ---------------------------------------------------------------------------

def _fingerprint():
    # The split selection-context / run-context fingerprint (design §12.2/§12.5).
    return {
        "selection_context": {
            "source_file_sha1s": {"mcts.py": "h1", "diagnose_fpu_policy_mass.py": "h2",
                                  "build_fpu_dev_corpus.py": "h3"},
            "checkpoint_identity": "model_iter_0001:deadbeef",
            "dev_manifest_sha1": "aaaa1111", "source_index_sha1": "src1",
            "replay_data_sha1": "rd1", "mcts_sims": 400,
            "base_mcts_config": {"c_puct": 1.5, "eval_batch_size": 14,
                                 "stall_flush_sims": 48, "fpu_policy_mass_reduction": None},
            "seeds": {"seed_base": 20260711, "eval_batch_size": 14},
            "grid": [["r0.10", 0.10], ["r0.20", 0.20]],
        },
        "run_context": {
            "selected_a": {"present": False, "manifest_sha1": None}, "add_noise": False,
            "git_commit": "cafef00d", "worktree_clean": True, "mode": "tuning",
            "stage": "controls", "observer_schema_version": 1,
            "runtime_provenance": {"python_version": "x", "mlx_version": None,
                                   "platform": "p", "machine": "m"},
        },
    }


def test_candidate_stage_refuses_stale_controls():
    fp = _fingerprint()
    gate = {"r0_qualified": True, "fingerprint": fp}
    validate_controls_fingerprint(gate, fp)                       # exact match -> no raise
    # a differing run_context (e.g. selected-A present, different stage) must NOT
    # fail it -- only the shared selection_context is hard-matched.
    other_run = json.loads(json.dumps(fp))
    other_run["run_context"]["selected_a"]["present"] = True
    other_run["run_context"]["stage"] = "candidates"
    validate_controls_fingerprint(gate, other_run)
    # ANY selection_context field change DOES fail it
    for bad_key in ("checkpoint_identity", "dev_manifest_sha1", "source_index_sha1",
                    "replay_data_sha1", "mcts_sims"):
        stale = json.loads(json.dumps(fp)); stale["selection_context"][bad_key] = "CHANGED"
        with pytest.raises(ValueError):
            validate_controls_fingerprint(gate, stale)
    with pytest.raises(ValueError):
        validate_controls_fingerprint({"r0_qualified": True}, fp)  # fingerprint absent


def test_r0_fail_blocks_candidates():
    require_r0_qualified({"r0_qualified": True, "fingerprint": _fingerprint()})   # ok
    for bad in ({"r0_qualified": False}, {}, {"r0_qualified": None}):
        with pytest.raises(ValueError):
            require_r0_qualified(bad)


def test_require_matching_mode_refuses_cross_mode_controls():
    # A controls artifact carries the split it was produced for (`mode`); its
    # lock-in baselines and r0_qualified are split-specific and must not be
    # reused across modes (e.g. a tuning-derived controls run consumed by a
    # frozen_check candidates run would silently apply the wrong lock-in
    # baselines / r0_qualified with no error).
    require_matching_mode({"mode": "tuning"}, "tuning")              # matching -> no raise
    require_matching_mode({"mode": "frozen_check"}, "frozen_check")  # matching -> no raise
    with pytest.raises(ValueError):
        require_matching_mode({"mode": "tuning"}, "frozen_check")
    with pytest.raises(ValueError):
        require_matching_mode({"mode": "frozen_check"}, "tuning")
    with pytest.raises(ValueError):
        require_matching_mode({}, "tuning")                          # mode absent


# ---------------------------------------------------------------------------
# Comparator parity (review finding #1): the canonical visit-leader
# comparator `min(visited, key=lambda c: (-c.visit_count, c.move))` is
# spelled independently in THREE places -- `mcts.visit_leader_move`,
# `continuation_extraction._best_child`, and this module's `_leader_child`.
# They are byte-identical today, but nothing pins the two test-importable
# copies together, so a future tie-break tweak to either one alone would
# silently desync the observer's leader from the diagnostic's. Synthetic
# trees follow the same hand-built `MCTSNode` pattern as
# tests/test_fpu_trace_observer.py; no real search/GPU/MLX involved.
# ---------------------------------------------------------------------------

def test_visit_leader_move_matches_diagnostic_leader_child():
    A, B, C = encode_move(0, 0), encode_move(0, 1), encode_move(0, 2)

    # Genuine tie decides the outcome: A and B are BOTH visit_count=4, so the
    # winner is only decidable by the lowest-move-id tie-break -- this
    # exercises the tie-break itself, not just the max.
    tie_root = MCTSNode(state=None)
    tie_root.children[A] = MCTSNode(state=None, parent=tie_root, move=A, visit_count=4, value_sum=1.0)
    tie_root.children[B] = MCTSNode(state=None, parent=tie_root, move=B, visit_count=4, value_sum=-2.0)
    assert visit_leader_move(tie_root) == _leader_child(tie_root).move == A

    # Clear max: C strictly leads on visit_count, no tie-break needed.
    max_root = MCTSNode(state=None)
    max_root.children[A] = MCTSNode(state=None, parent=max_root, move=A, visit_count=4, value_sum=1.0)
    max_root.children[B] = MCTSNode(state=None, parent=max_root, move=B, visit_count=4, value_sum=-2.0)
    max_root.children[C] = MCTSNode(state=None, parent=max_root, move=C, visit_count=9, value_sum=0.5)
    assert visit_leader_move(max_root) == _leader_child(max_root).move == C

    # No completed visits anywhere (only a pending, zero-visit child) -> both
    # copies agree on None.
    root_no_visits = MCTSNode(state=None)
    root_no_visits.children[A] = MCTSNode(state=None, parent=root_no_visits, move=A, visit_count=0)
    assert visit_leader_move(root_no_visits) is None and _leader_child(root_no_visits) is None


# ---------------------------------------------------------------------------
# Task A1 -- propagate ply_bucket into the dev rows (v2-gated; v1 stays
# byte-identical). Spec §0/§9: docs/superpowers/specs/
# 2026-07-14-fpu-v2-reservoir-protocol-qualification-design.md. Brief:
# .superpowers/sdd/preop-task-A1-brief.md.
#
# `_dev_target_row`/`_dev_control_row`/`_dev_rows_vs`/`_candidate_dev_records`
# consume `_position_features`-shaped `cand`/`ref` dicts (NOT the already-built
# gate-row dicts `_safe_target`/`_safe_control` fabricate above), so this
# section has its own minimal feature + manifest-row fixtures.
# ---------------------------------------------------------------------------

def _feat(root_value_stm=0.0, top_share=0.5, effective_children=3.0, collapsed=False,
         top_move=1, top_move_prior=0.3, trace=None):
    return {
        "root_value_stm": root_value_stm, "top_share": top_share,
        "effective_children": effective_children, "collapsed": collapsed,
        "top_move": top_move, "top_move_prior": top_move_prior,
        "trace": trace or {"selected_move_prior_rank": 1, "selected_move_prior": 0.5,
                           "explored_mass_at_stabilization": 0.9, "stabilization_sim": 10,
                           "final_root_top_share": 0.5},
    }


def _manifest_row(sha, role, band="b200_299", ply_bucket=None):
    """A raw dev-corpus manifest row as `_load_dev_rows` (csv.DictReader)
    produces it. Both v1 and v2 manifests carry a `ply_bucket` column, so it is
    present here whenever the caller supplies one -- carrying it in the SOURCE
    row is not what gates propagation into the gate/persisted rows;
    `carry_ply_bucket` is."""
    row = {"canonical_position_sha1": sha, "role": role, "branching_band": band}
    if ply_bucket is not None:
        row["ply_bucket"] = ply_bucket
    return row


def test_dev_target_control_row_default_omits_ply_bucket():
    cand, ref = _feat(root_value_stm=0.3, collapsed=True), _feat(root_value_stm=0.1)
    trow = _dev_target_row("b200_299", cand, ref)
    crow = _dev_control_row(cand, ref)
    assert "ply_bucket" not in trow
    assert "ply_bucket" not in crow


def test_dev_target_control_row_carries_ply_bucket_when_given():
    cand, ref = _feat(root_value_stm=0.3, collapsed=True), _feat(root_value_stm=0.1)
    trow = _dev_target_row("b200_299", cand, ref, ply_bucket="late")
    crow = _dev_control_row(cand, ref, ply_bucket="late")
    assert trow["ply_bucket"] == "late"
    assert crow["ply_bucket"] == "late"


def test_dev_rows_vs_carries_ply_bucket_when_enabled():
    cand = {"s1": _feat(root_value_stm=0.3, collapsed=True), "s2": _feat(root_value_stm=0.2)}
    ref = {"s1": _feat(root_value_stm=0.1), "s2": _feat(root_value_stm=0.1)}
    rows = [_manifest_row("s1", "target", ply_bucket="late"),
           _manifest_row("s2", "control", ply_bucket="late")]
    out = _dev_rows_vs(rows, cand, ref, carry_ply_bucket=True)
    assert len(out) == 2
    assert all(r["ply_bucket"] == "late" for r in out)


def test_dev_rows_vs_default_is_byte_identical_to_today():
    cand = {"s1": _feat(root_value_stm=0.3, top_share=0.6, effective_children=2.0,
                       collapsed=True, top_move=5, top_move_prior=0.2),
           "s2": _feat(root_value_stm=0.2)}
    ref = {"s1": _feat(root_value_stm=0.1, top_share=0.5, effective_children=4.0,
                      top_move=5, top_move_prior=0.4),
          "s2": _feat(root_value_stm=0.1)}
    # Source rows carry ply_bucket -- exactly like a real v1 OR v2 manifest row
    # would (both have the column). Default carry_ply_bucket=False must still
    # drop it entirely: the FLAG, not the source data, decides.
    rows = [_manifest_row("s1", "target", ply_bucket="late"),
           _manifest_row("s2", "control", ply_bucket="late")]
    out = _dev_rows_vs(rows, cand, ref)
    assert len(out) == 2
    for r in out:
        assert "ply_bucket" not in r
    # Literal dict match, hand-computed from the inputs above (not re-derived
    # from _dev_target_row/_dev_control_row) so a shared bug could not hide a
    # regression here.
    assert out[0] == {
        "role": "target", "band": "b200_299", "new_collapse": True,
        "lock_in": False, "mover_delta": pytest.approx(0.2),
        "eff_children_reduction": pytest.approx(0.5),
        "top_share_inc": pytest.approx(0.1),
    }
    assert out[1] == {
        "role": "control", "mover_delta": pytest.approx(0.1),
        "control_flip_to_lower_prior": False,
    }


def test_candidate_dev_records_v1_mode_emits_exact_fieldnames():
    cand = {"s1": _feat(root_value_stm=0.3, collapsed=True), "s2": _feat(root_value_stm=0.2)}
    ref = {"s1": _feat(root_value_stm=0.1), "s2": _feat(root_value_stm=0.1)}
    rows = [_manifest_row("s1", "target", ply_bucket="late"),
           _manifest_row("s2", "control", ply_bucket="late")]
    recs = _candidate_dev_records(rows, cand, ref, "r0.20", "absolute_off")
    assert len(recs) == 2
    for r in recs:
        assert set(r.keys()) == set(CANDIDATE_DEV_ROW_FIELDNAMES)
        assert "ply_bucket" not in r


def test_candidate_dev_records_v2_mode_carries_ply_bucket(tmp_path):
    cand = {"s1": _feat(root_value_stm=0.3, collapsed=True), "s2": _feat(root_value_stm=0.2)}
    ref = {"s1": _feat(root_value_stm=0.1), "s2": _feat(root_value_stm=0.1)}
    rows = [_manifest_row("s1", "target", ply_bucket="late"),
           _manifest_row("s2", "control", ply_bucket="mid")]
    recs = _candidate_dev_records(rows, cand, ref, "r0.20", "absolute_off",
                                  carry_ply_bucket=True)
    assert len(recs) == 2
    by_sha = {r["canonical_sha1"]: r for r in recs}
    assert by_sha["s1"]["ply_bucket"] == "late"
    assert by_sha["s2"]["ply_bucket"] == "mid"

    # v2 sibling fieldnames list: additive, v1 constant left untouched.
    from scripts.GPU.alphazero.diagnose_fpu_policy_mass import (
        CANDIDATE_DEV_ROW_FIELDNAMES_V2, _write_csv)
    assert set(CANDIDATE_DEV_ROW_FIELDNAMES_V2) == set(CANDIDATE_DEV_ROW_FIELDNAMES) | {"ply_bucket"}
    assert "ply_bucket" not in CANDIDATE_DEV_ROW_FIELDNAMES
    for r in recs:
        assert set(r.keys()) == set(CANDIDATE_DEV_ROW_FIELDNAMES_V2)

    # round-trips through the CSV writer against the v2 schema
    import csv
    path = tmp_path / "candidate_dev_rows_v2.csv"
    _write_csv(str(path), CANDIDATE_DEV_ROW_FIELDNAMES_V2, recs)
    with open(path, newline="") as f:
        back = list(csv.DictReader(f))
    assert {r["canonical_sha1"]: r["ply_bucket"] for r in back} == {"s1": "late", "s2": "mid"}


# ---------------------------------------------------------------------------
# Task A2 -- `--dev-corpus-config` option + 5 identity checks + coupled
# stratum/ply_bucket threading into the 3 production `dev_safety_verdict`
# call sites (spec §0/§9; brief .superpowers/sdd/preop-task-A2-brief.md).
#
# `_resolve_v2_stratum(args)` only ever reads the config JSON + the manifest's
# sibling `.meta.json` -- fabricated with `tmp_path`, no MCTS/evaluator/GPU.
# The COUPLING tests below go one level up: they invoke the REAL
# `run_controls_stage` / `run_candidates_stage` (the actual production call
# sites) with every heavy/lazy-imported dependency (evaluator load, MCTS
# search, the replay-index reader, selected-A) monkeypatched to a
# deterministic fake -- so the production WIRING is exercised end-to-end while
# staying zero-GPU/zero-MCTS, and `dev_safety_verdict` itself is left REAL
# (just spied) so a wiring bug would surface exactly as it would in
# production (a missing-key ValueError when carry_ply_bucket and stratum_key
# disagree).
# ---------------------------------------------------------------------------

def _v2_config_dict(**overrides):
    """Every `_V2_CONFIG_REQUIRED_KEYS` key, fabricated. The three keys A2
    hard-matches (`select_out`, `source_index_path`, `new_collapse_stratum`)
    are steered by the caller; everything else is inert filler `load_v2_config`
    requires present but `_resolve_v2_stratum` never reads."""
    d = {
        "source_index_path": "src.jsonl",
        "seed_range": [0, 6],
        "selection_seed": 1,
        "phase_allocation": {},
        "late_floors": {},
        "enumerator_params": {},
        "new_collapse_stratum": "ply_bucket",
        "checkpoint": "ckpt.npz",
        "forbidden_manifests": [],
        "screen_out": "screen.csv",
        "select_out": "manifest.csv",
        "expected_fingerprints": {},
    }
    d.update(overrides)
    return d


def _write_v2_config(tmp_path, name="config.json", **overrides):
    path = tmp_path / name
    path.write_text(json.dumps(_v2_config_dict(**overrides)))
    return str(path)


def _write_manifest_meta(dev_manifest_path, *, config_sha1,
                         new_collapse_stratum="ply_bucket"):
    meta = {"new_collapse_stratum": new_collapse_stratum,
            "provenance": {"config_sha1": config_sha1}}
    Path(f"{dev_manifest_path}.meta.json").write_text(json.dumps(meta))


def _faithful_v2_setup(tmp_path):
    """A config + manifest `.meta.json` pair where all five A2 checks agree."""
    dev_manifest = str(tmp_path / "manifest.csv")
    source_jsonl = str(tmp_path / "src.jsonl")
    config_path = _write_v2_config(tmp_path, select_out=dev_manifest,
                                   source_index_path=source_jsonl)
    _write_manifest_meta(dev_manifest, config_sha1=prov.file_sha1(config_path))
    return dev_manifest, source_jsonl, config_path


def _stage_args(**over):
    base = dict(dev_corpus_config=None, dev_manifest="manifest.csv",
               source_jsonl="src.jsonl")
    base.update(over)
    return types.SimpleNamespace(**base)


def test_resolve_v2_stratum_v1_no_config_returns_band_and_reads_nothing():
    # Paths point nowhere -- if the resolver tried to read ANYTHING it would
    # raise (FileNotFoundError); returning cleanly proves it read no config.
    args = _stage_args(dev_corpus_config=None,
                       dev_manifest="/nonexistent/dir/manifest.csv",
                       source_jsonl="/nonexistent/dir/src.jsonl")
    assert _resolve_v2_stratum(args) == "band"


def test_resolve_v2_stratum_v2_all_checks_agree_returns_ply_bucket(tmp_path):
    dev_manifest, source_jsonl, config_path = _faithful_v2_setup(tmp_path)
    args = _stage_args(dev_corpus_config=config_path, dev_manifest=dev_manifest,
                       source_jsonl=source_jsonl)
    assert _resolve_v2_stratum(args) == "ply_bucket"


def test_resolve_v2_stratum_select_out_mismatch_raises(tmp_path):
    dev_manifest, source_jsonl, config_path = _faithful_v2_setup(tmp_path)
    args = _stage_args(dev_corpus_config=config_path,
                       dev_manifest=str(tmp_path / "other_manifest.csv"),
                       source_jsonl=source_jsonl)
    with pytest.raises(ValueError, match="select_out"):
        _resolve_v2_stratum(args)


def test_resolve_v2_stratum_source_index_path_mismatch_raises(tmp_path):
    dev_manifest, source_jsonl, config_path = _faithful_v2_setup(tmp_path)
    args = _stage_args(dev_corpus_config=config_path, dev_manifest=dev_manifest,
                       source_jsonl=str(tmp_path / "other_src.jsonl"))
    with pytest.raises(ValueError, match="source_index_path"):
        _resolve_v2_stratum(args)


def test_resolve_v2_stratum_manifest_config_sha1_mismatch_raises(tmp_path):
    dev_manifest, source_jsonl, config_path = _faithful_v2_setup(tmp_path)
    _write_manifest_meta(dev_manifest, config_sha1="deadbeef" * 5)   # wrong hash
    args = _stage_args(dev_corpus_config=config_path, dev_manifest=dev_manifest,
                       source_jsonl=source_jsonl)
    with pytest.raises(ValueError, match="config_sha1"):
        _resolve_v2_stratum(args)


def test_resolve_v2_stratum_meta_stratum_disagrees_with_config_raises(tmp_path):
    dev_manifest, source_jsonl, config_path = _faithful_v2_setup(tmp_path)
    _write_manifest_meta(dev_manifest, config_sha1=prov.file_sha1(config_path),
                         new_collapse_stratum="band")   # config says ply_bucket
    args = _stage_args(dev_corpus_config=config_path, dev_manifest=dev_manifest,
                       source_jsonl=source_jsonl)
    with pytest.raises(ValueError, match="new_collapse_stratum"):
        _resolve_v2_stratum(args)


def test_resolve_v2_stratum_non_ply_bucket_config_stratum_raises(tmp_path):
    dev_manifest = str(tmp_path / "manifest.csv")
    source_jsonl = str(tmp_path / "src.jsonl")
    config_path = _write_v2_config(tmp_path, select_out=dev_manifest,
                                   source_index_path=source_jsonl,
                                   new_collapse_stratum="band")
    _write_manifest_meta(dev_manifest, config_sha1=prov.file_sha1(config_path),
                         new_collapse_stratum="band")   # meta agrees w/ config
    args = _stage_args(dev_corpus_config=config_path, dev_manifest=dev_manifest,
                       source_jsonl=source_jsonl)
    with pytest.raises(ValueError, match="ply_bucket"):
        _resolve_v2_stratum(args)


# ---------------------------------------------------------------------------
# Coupling: the resolved stratum + carry_ply_bucket actually reach the 3
# production `dev_safety_verdict` call sites (1 controls + 2 candidates), and
# the persisted candidate_dev_rows.csv picks the matching field list.
# ---------------------------------------------------------------------------

@dataclass
class _FakeMctsCfg:
    c_puct: float = 1.5
    fpu_policy_mass_reduction: object = None
    eval_batch_size: int = 14
    stall_flush_sims: int = 48
    n_simulations: int = 400


def _fake_make_evaluator_and_base_cfg(checkpoint, eval_batch_size, stall_flush_sims):
    return object(), _FakeMctsCfg(eval_batch_size=eval_batch_size,
                                  stall_flush_sims=stall_flush_sims)


def _fake_make_base_cfg(eval_batch_size, stall_flush_sims):
    return _FakeMctsCfg(eval_batch_size=eval_batch_size,
                        stall_flush_sims=stall_flush_sims)


def _canned_feat():
    """`_feat()` (Task A1's fixture) plus `replies` -- `_controls_case_row`
    (exercised here via the real `run_controls_stage`) needs it; A1's own
    tests never construct a `_controls_case_row`, so its shared `_feat()`
    fixture has no reason to carry it. Kept local rather than widening A1's
    fixture for a need that is specific to this production-path coupling
    test."""
    feat = _feat()
    feat["replies"] = 0
    return feat


def _fake_run_configs_over_corpus(dev_rows, run_configs, *, evaluator, base_cfg,
                                  replay_by_game, seed_base):
    # A single canned, all-clean feature dict shared by every (config,
    # position) pair -- the gate MATH is already pinned elsewhere; this test
    # is purely about the ply_bucket/stratum_key WIRING.
    canned = _canned_feat()
    return {c.label: {r["canonical_position_sha1"]: canned for r in dev_rows}
            for c in run_configs}


def _fake_load_selected_a(args, evaluator, base_cfg, run_configs):
    return {}


def _fake_load_game_index(source_jsonl):
    return []


def _v2_dev_rows():
    """A minimal (target + control) dev-row set carrying `ply_bucket` in the
    SOURCE data -- exactly like a real v2 manifest. Carrying it in the source
    does not by itself make it appear downstream (proven by the v1 coupling
    test below): only the resolved stratum's `carry_ply_bucket` flag does."""
    return [
        {"canonical_position_sha1": "s1", "role": "target", "branching_band": "b200_299",
         "ply_bucket": "late", "game_idx": "0", "position_ply": "10", "side": "red",
         "split": "tuning"},
        {"canonical_position_sha1": "s2", "role": "target", "branching_band": "b200_299",
         "ply_bucket": "mid", "game_idx": "0", "position_ply": "20", "side": "black",
         "split": "tuning"},
        {"canonical_position_sha1": "s3", "role": "control", "branching_band": "b200_299",
         "ply_bucket": "late", "game_idx": "1", "position_ply": "15", "side": "red",
         "split": "tuning"},
    ]


def _patch_operator_internals(monkeypatch):
    """Replace every heavy/lazy-imported piece `run_controls_stage`/
    `run_candidates_stage` touch with a deterministic fake, so the REAL
    functions run end to end with zero MCTS/evaluator/GPU. `dev_safety_verdict`
    is deliberately left untouched by this helper -- callers wrap it with
    their own recording spy so a wiring bug surfaces as a real ValueError."""
    monkeypatch.setattr(dfpm, "_make_evaluator_and_base_cfg",
                        _fake_make_evaluator_and_base_cfg)
    monkeypatch.setattr(dfpm, "_make_base_cfg", _fake_make_base_cfg)
    monkeypatch.setattr(dfpm, "_run_configs_over_corpus", _fake_run_configs_over_corpus)
    monkeypatch.setattr(dfpm, "_load_selected_a", _fake_load_selected_a)
    monkeypatch.setattr(bfdc, "load_game_index", _fake_load_game_index)


def _spy_dev_safety_verdict(monkeypatch):
    """Wrap the REAL `dev_safety_verdict` to record (rows, stratum_key) per
    call while still executing the real gate -- a wiring bug (carry_ply_bucket
    disagreeing with the passed stratum_key) raises exactly as it would in
    production (dev_safety_verdict's own missing-key ValueError)."""
    calls = []
    real = dfpm.dev_safety_verdict

    def _spy(rows, *a, **kw):
        rows = list(rows)
        calls.append((rows, kw.get("stratum_key", "band")))
        return real(rows, *a, **kw)

    monkeypatch.setattr(dfpm, "dev_safety_verdict", _spy)
    return calls


def test_production_paths_carry_ply_bucket_and_resolved_stratum_with_v2_config(
        tmp_path, monkeypatch):
    _patch_operator_internals(monkeypatch)
    calls = _spy_dev_safety_verdict(monkeypatch)

    dev_manifest, source_jsonl, config_path = _faithful_v2_setup(tmp_path)
    dev_rows = _v2_dev_rows()
    out_dir = tmp_path / "out"
    args = types.SimpleNamespace(
        mode="tuning", dev_manifest=dev_manifest, source_jsonl=source_jsonl,
        selected_a_manifest="selected_a.csv", checkpoint="ckpt.npz",
        out_dir=str(out_dir), frozen_r=None, tuning_result=None,
        seed_base=20260711, eval_batch_size=14, stall_flush_sims=48,
        dev_corpus_config=config_path)

    assert run_controls_stage(args, dev_rows, [ABSOLUTE_OFF, R0]) == 0
    assert run_candidates_stage(args, dev_rows, list(GRID)) == 0

    # 1 controls-stage call site + 2 candidates-stage call sites x len(GRID)
    assert len(calls) == 1 + 2 * len(GRID)
    for rows, stratum_key in calls:
        assert stratum_key == "ply_bucket"
        assert rows and all("ply_bucket" in r for r in rows)

    # the persisted candidate_dev_rows.csv picks the v2 (ply_bucket) schema
    import csv
    with open(out_dir / "candidate_dev_rows.csv", newline="") as f:
        header = next(csv.reader(f))
    assert "ply_bucket" in header


def test_production_paths_v1_no_config_omit_ply_bucket_and_use_band(
        tmp_path, monkeypatch):
    _patch_operator_internals(monkeypatch)
    calls = _spy_dev_safety_verdict(monkeypatch)

    dev_rows = _v2_dev_rows()      # source carries ply_bucket -- must be dropped
    out_dir = tmp_path / "out"
    args = types.SimpleNamespace(
        mode="tuning", dev_manifest=str(tmp_path / "manifest.csv"),
        source_jsonl=str(tmp_path / "src.jsonl"),
        selected_a_manifest="selected_a.csv", checkpoint="ckpt.npz",
        out_dir=str(out_dir), frozen_r=None, tuning_result=None,
        seed_base=20260711, eval_batch_size=14, stall_flush_sims=48,
        dev_corpus_config=None)                           # v1 -- no config

    assert run_controls_stage(args, dev_rows, [ABSOLUTE_OFF, R0]) == 0
    assert run_candidates_stage(args, dev_rows, list(GRID)) == 0

    assert len(calls) == 1 + 2 * len(GRID)
    for rows, stratum_key in calls:
        assert stratum_key == "band"
        assert rows and all("ply_bucket" not in r for r in rows)

    import csv
    with open(out_dir / "candidate_dev_rows.csv", newline="") as f:
        header = next(csv.reader(f))
    assert "ply_bucket" not in header
