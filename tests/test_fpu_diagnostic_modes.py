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

import pytest

from scripts.GPU.alphazero.diagnose_fpu_policy_mass import (
    FpuRunConfig, ABSOLUTE_OFF, R0, GRID, validate_stage_mode,
    lock_in_event, progress, reply_reduction, prior_rank,
    dev_safety_verdict, selected_a_verdict)
from scripts.GPU.alphazero.diagnose_fpu_policy_mass import (
    V_REF, top_share, validate_controls_fingerprint, require_r0_qualified,
    require_matching_mode, _leader_child)
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
