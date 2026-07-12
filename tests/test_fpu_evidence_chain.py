"""Evidence-chain hardening (Part 1) pure tests -- design §12.2/§12.3/§12.5.

Covers the split selection-context/run-context fingerprint (#5), the immutable
frozen-check coefficient guard (#2), and the tuning-only selected-A mode gate
(#3). Every test drives a PURE function or a FABRICATED dict / temp file -- no
MCTS / evaluator / checkpoint / real corpus, and `main()` (the 400-sim operator
phase) is never invoked. Frozen refs:
docs/superpowers/specs/2026-07-10-context-relative-fpu-policy-mass-design.md
(§12), brief .superpowers/sdd/hardening-brief-1.md.

The selection-context / run-context split is the crux: `selection_context` is
the SHARED, result-determining identity every stage of one protocol run (and
tuning-vs-frozen) must match EXACTLY; `run_context` is RECORDED but never
cross-matched, so a legitimate difference there (selected-A present in tuning
vs absent in frozen, a different stage, git/runtime provenance) must NOT fail
the join.
"""
import csv
import json
import math
from dataclasses import dataclass

import pytest

from scripts.GPU.alphazero import fpu_provenance as prov
from scripts.GPU.alphazero.diagnose_fpu_policy_mass import (
    ABSOLUTE_OFF, CONTROLS_CASE_FIELDNAMES, GRID, R0, V_REF,
    build_run_fingerprint, dev_safety_verdict, selected_a_verdict,
    validate_controls_fingerprint, require_frozen_matches_tuning,
    validate_selected_a_mode, verify_recomputed_controls,
    _candidate_dev_records, _candidate_result_record, _controls_case_row,
    _write_csv)


# ---------------------------------------------------------------------------
# #5 -- fpu_provenance helpers (stdlib-only; no mlx import)
# ---------------------------------------------------------------------------

def test_file_sha1_content_sensitive_and_sentinels(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    h1 = prov.file_sha1(str(f))
    f.write_bytes(b"hell0")
    h2 = prov.file_sha1(str(f))
    assert len(h1) == 40 and h1 != h2                 # content-sensitive
    assert prov.file_sha1(None) == "none"             # absent -> sentinel
    assert prov.file_sha1(str(tmp_path / "nope.bin")) == "missing"


def test_source_file_sha1s_keyed_by_basename_and_content_sensitive(tmp_path):
    a = tmp_path / "a.py"; b = tmp_path / "b.py"
    a.write_bytes(b"AAAA"); b.write_bytes(b"BBBB")
    d1 = prov.source_file_sha1s([str(a), str(b)])
    assert set(d1) == {"a.py", "b.py"}                # keyed by basename
    a.write_bytes(b"AAAA2")
    d2 = prov.source_file_sha1s([str(a), str(b)])
    assert d2["a.py"] != d1["a.py"] and d2["b.py"] == d1["b.py"]   # only changed file moves


def test_replay_data_sha1_order_independent_and_content_sensitive(tmp_path):
    p = tmp_path / "p.json"; q = tmp_path / "q.json"
    p.write_text('{"g": 1}'); q.write_text('{"g": 2}')
    h_pq = prov.replay_data_sha1([str(p), str(q)])
    h_qp = prov.replay_data_sha1([str(q), str(p)])
    assert h_pq == h_qp and len(h_pq) == 40           # order-independent-by-path
    q.write_text('{"g": 3}')
    assert prov.replay_data_sha1([str(p), str(q)]) != h_pq        # content-sensitive


def test_runtime_provenance_has_keys_without_importing_mlx():
    rp = prov.runtime_provenance()
    assert set(rp) >= {"python_version", "mlx_version", "platform", "machine"}
    assert isinstance(rp["python_version"], str)
    assert rp["mlx_version"] is None or isinstance(rp["mlx_version"], str)
    assert isinstance(rp["platform"], str) and isinstance(rp["machine"], str)


def test_git_helpers_are_typed():
    assert isinstance(prov.worktree_clean(), bool)
    assert isinstance(prov.git_commit(), str)


# ---------------------------------------------------------------------------
# #5 -- build_run_fingerprint split structure
# ---------------------------------------------------------------------------

@dataclass
class _FakeCfg:
    c_puct: float = 1.5
    fpu_policy_mass_reduction: object = None
    eval_batch_size: int = 14
    stall_flush_sims: int = 48
    n_simulations: int = 400


def _seeds():
    return {"seed_base": 20260711, "eval_batch_size": 14, "stall_flush_sims": 48}


def _fp(tmp_path, *, mode, stage, selected_a):
    """Build a real split fingerprint from temp files (identical shared inputs
    across calls -> identical selection_context; selected_a only moves
    run_context)."""
    ckpt = tmp_path / "ck.npz"; ckpt.write_bytes(b"CKPT-BYTES")
    devm = tmp_path / "dev.csv"; devm.write_text("split\ntuning\n")
    src = tmp_path / "src.jsonl"; src.write_text('{"game_idx": 0}\n')
    rp = tmp_path / "r0.json"; rp.write_text('{"moves": []}')
    sa = None
    if selected_a:
        saf = tmp_path / "a.csv"; saf.write_text("case\n"); sa = str(saf)
    return build_run_fingerprint(
        dev_manifest=str(devm), checkpoint=str(ckpt), base_cfg=_FakeCfg(),
        source_jsonl=str(src), replay_paths=[str(rp)], seeds=_seeds(),
        selected_a_manifest=sa, mode=mode, stage=stage)


def test_build_run_fingerprint_split_structure(tmp_path):
    fp = _fp(tmp_path, mode="tuning", stage="candidates", selected_a=True)
    assert set(fp) == {"selection_context", "run_context"}

    sel = fp["selection_context"]
    assert set(sel) >= {"source_file_sha1s", "checkpoint_identity", "dev_manifest_sha1",
                        "source_index_sha1", "replay_data_sha1", "base_mcts_config",
                        "mcts_sims", "seeds", "grid"}
    assert sel["mcts_sims"] == 400
    assert sel["base_mcts_config"]["c_puct"] == 1.5          # FULL asdict, not a subset
    assert sel["grid"] == [[c.label, c.reduction] for c in GRID]
    # RF1: the hard-matched source set includes the state-RECONSTRUCTION deps
    # (goal_line_trigger_probe_cases.py + game/twixt_state.py), which are equally
    # result-determining -- they rebuild the position every search runs on.
    assert set(sel["source_file_sha1s"]) == {
        "diagnose_fpu_policy_mass.py", "mcts.py", "build_fpu_dev_corpus.py",
        "goal_line_trigger_probe_cases.py", "twixt_state.py"}

    run = fp["run_context"]
    assert set(run) >= {"selected_a", "add_noise", "git_commit", "worktree_clean",
                        "runtime_provenance", "mode", "stage", "observer_schema_version"}
    assert run["add_noise"] is False                          # explicit
    assert run["selected_a"]["present"] is True and run["selected_a"]["manifest_sha1"]
    assert run["mode"] == "tuning" and run["stage"] == "candidates"


def test_selection_context_shared_across_selected_a_presence(tmp_path):
    # The crux of #5: selected-A present (tuning) vs absent (frozen) changes ONLY
    # run_context; the shared selection_context is byte-identical, so a frozen
    # stage can join a tuning controls run.
    fp_present = _fp(tmp_path, mode="tuning", stage="candidates", selected_a=True)
    fp_absent = _fp(tmp_path, mode="frozen_check", stage="candidates", selected_a=False)
    assert fp_present["selection_context"] == fp_absent["selection_context"]
    assert fp_present["run_context"]["selected_a"] != fp_absent["run_context"]["selected_a"]

    # validate_controls_fingerprint accepts across the run_context difference
    gate = {"r0_qualified": True, "mode": "tuning", "fingerprint": fp_present}
    validate_controls_fingerprint(gate, fp_absent)            # selection_context matches


# ---------------------------------------------------------------------------
# #5 -- validate_controls_fingerprint compares ONLY selection_context
#       (fabricated dicts; no fingerprint build)
# ---------------------------------------------------------------------------

def _sel():
    return {
        "source_file_sha1s": {"mcts.py": "h1", "diagnose_fpu_policy_mass.py": "h2",
                              "build_fpu_dev_corpus.py": "h3"},
        "checkpoint_identity": "model_iter_0001:deadbeef",
        "dev_manifest_sha1": "d1", "source_index_sha1": "s1", "replay_data_sha1": "rd1",
        "base_mcts_config": {"c_puct": 1.5, "eval_batch_size": 14, "fpu_policy_mass_reduction": None},
        "mcts_sims": 400, "seeds": {"seed_base": 1},
        "grid": [["r0.20", 0.20]],
    }


def _run(**over):
    r = {"selected_a": {"present": False, "manifest_sha1": None}, "add_noise": False,
         "git_commit": "cafef00d", "worktree_clean": True, "mode": "tuning",
         "stage": "controls", "observer_schema_version": 1,
         "runtime_provenance": {"python_version": "x", "mlx_version": None,
                                "platform": "p", "machine": "m"}}
    r.update(over)
    return r


def test_validate_controls_fingerprint_matches_selection_ignores_run():
    sel = _sel()
    gate = {"fingerprint": {"selection_context": sel,
                            "run_context": _run(selected_a={"present": True})}}
    # differing run_context (selected-A present vs absent, different stage) -> OK
    validate_controls_fingerprint(
        gate, {"selection_context": dict(sel),
               "run_context": _run(selected_a={"present": False}, stage="candidates")})
    # any selection_context field change DOES fail it
    for bad_key in ("checkpoint_identity", "dev_manifest_sha1", "source_index_sha1",
                    "replay_data_sha1", "mcts_sims"):
        bad = {"selection_context": {**sel, bad_key: "CHANGED"}}
        with pytest.raises(ValueError):
            validate_controls_fingerprint(gate, bad)
    # a nested base_mcts_config change fails too
    with pytest.raises(ValueError):
        validate_controls_fingerprint(
            gate, {"selection_context": {**sel, "base_mcts_config": {"c_puct": 2.0}}})


def test_validate_controls_fingerprint_requires_blocks():
    with pytest.raises(ValueError):
        validate_controls_fingerprint({}, {"selection_context": _sel()})       # no fingerprint
    with pytest.raises(ValueError):
        validate_controls_fingerprint({"fingerprint": {}},
                                      {"selection_context": _sel()})            # no selection_context


# ---------------------------------------------------------------------------
# #2 -- require_frozen_matches_tuning (immutable frozen coefficient)
# ---------------------------------------------------------------------------

def _tuning_result(sel, *, smallest_safe_r="r0.20", mode="tuning"):
    # JSON round-trip: the real guard loads this from disk, so tuples->lists etc.
    return json.loads(json.dumps({
        "mode": mode, "smallest_safe_r": smallest_safe_r, "candidates": [],
        "fingerprint": {"selection_context": sel, "run_context": _run(stage="candidates")}}))


def test_require_frozen_matches_tuning_ok():
    sel = _sel()
    require_frozen_matches_tuning(_tuning_result(sel), frozen_reduction=0.20,
                                  expected_selection_context=dict(sel))


def test_require_frozen_matches_tuning_rejections():
    sel = _sel()
    tr = _tuning_result(sel)
    # (d) frozen_reduction != the tuning-selected coefficient (r0.20 -> 0.20)
    with pytest.raises(ValueError):
        require_frozen_matches_tuning(tr, frozen_reduction=0.35,
                                      expected_selection_context=dict(sel))
    # (b) null smallest_safe_r
    with pytest.raises(ValueError):
        require_frozen_matches_tuning(_tuning_result(sel, smallest_safe_r=None),
                                      frozen_reduction=0.20, expected_selection_context=dict(sel))
    # (a) wrong mode
    with pytest.raises(ValueError):
        require_frozen_matches_tuning(_tuning_result(sel, mode="frozen_check"),
                                      frozen_reduction=0.20, expected_selection_context=dict(sel))
    # (c) mismatched selection_context
    with pytest.raises(ValueError):
        require_frozen_matches_tuning(tr, frozen_reduction=0.20,
                                      expected_selection_context={**sel, "checkpoint_identity": "OTHER"})
    # smallest_safe_r not a GRID label
    with pytest.raises(ValueError):
        require_frozen_matches_tuning(_tuning_result(sel, smallest_safe_r="rBOGUS"),
                                      frozen_reduction=0.20, expected_selection_context=dict(sel))


def test_frozen_locks_to_tuning_selection_end_to_end(tmp_path):
    # Full path: build a real tuning fingerprint + a real frozen fingerprint from
    # identical shared inputs, then lock frozen to the tuning selection.
    tun_fp = _fp(tmp_path, mode="tuning", stage="candidates", selected_a=True)
    frz_fp = _fp(tmp_path, mode="frozen_check", stage="candidates", selected_a=False)
    tuning_result = json.loads(json.dumps(
        {"mode": "tuning", "smallest_safe_r": "r0.35", "candidates": [], "fingerprint": tun_fp}))
    require_frozen_matches_tuning(tuning_result, frozen_reduction=0.35,
                                  expected_selection_context=frz_fp["selection_context"])
    with pytest.raises(ValueError):                       # any other r is refused
        require_frozen_matches_tuning(tuning_result, frozen_reduction=0.20,
                                      expected_selection_context=frz_fp["selection_context"])


# ---------------------------------------------------------------------------
# #3 -- selected-A is tuning-only
# ---------------------------------------------------------------------------

def test_validate_selected_a_mode():
    validate_selected_a_mode("tuning", True)              # ok
    validate_selected_a_mode("frozen_check", False)       # ok
    with pytest.raises(SystemExit):
        validate_selected_a_mode("tuning", False)         # tuning REQUIRES selected-A
    with pytest.raises(SystemExit):
        validate_selected_a_mode("frozen_check", True)    # frozen FORBIDS selected-A


# ---------------------------------------------------------------------------
# RF2 -- replay_data_sha1 length-delimits each file's bytes (Part-1 fix)
# ---------------------------------------------------------------------------

def test_replay_data_sha1_delimits_file_boundaries(tmp_path):
    # Same total concatenated bytes ("XYZ" in sorted [a,b] order), different file
    # PARTITION. Without a per-file delimiter both fold to sha1(b"XYZ") and
    # COLLIDE; delimiting each file's bytes must make the two partitions differ.
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_bytes(b"XY"); b.write_bytes(b"Z")
    h1 = prov.replay_data_sha1([str(a), str(b)])
    a.write_bytes(b"X"); b.write_bytes(b"YZ")
    h2 = prov.replay_data_sha1([str(a), str(b)])
    assert h1 != h2 and len(h1) == 40 and len(h2) == 40
    # a genuinely empty file is still delimited distinctly from an absent boundary
    a.write_bytes(b""); b.write_bytes(b"XYZ")
    h3 = prov.replay_data_sha1([str(a), str(b)])
    assert h3 != h1 and h3 != h2


# ---------------------------------------------------------------------------
# #4a -- SafetyVerdict.metrics EXPOSES the computed numbers; the §6 gate LOGIC
# (rejected + reasons) stays byte-identical (thresholds/comparators/AND-OR).
# ---------------------------------------------------------------------------

def _tgt(**over):
    row = dict(role="target", band="b200_299", new_collapse=False, lock_in=False,
               mover_delta=0.0, eff_children_reduction=0.0, top_share_inc=0.0)
    row.update(over)
    return row


def _ctl(**over):
    row = dict(role="control", mover_delta=0.0, control_flip_to_lower_prior=False)
    row.update(over)
    return row


def test_safety_verdict_reasons_and_rejected_unchanged_plus_metrics():
    # A single-gate input (overall target new-collapse rate == 0.05, one n>=20
    # band whose 0.05 < 0.10 so the band gate does NOT fire, no lock-ins, zero
    # mover/eff/top-share). The §6 logic must yield EXACTLY this rejected/reasons
    # pair -- pinned byte-for-byte -- and metrics must carry every computed number.
    rows = [_tgt(new_collapse=(i < 5)) for i in range(100)]
    v = dev_safety_verdict(rows, ref=R0, r0_lockin=5, absoff_lockin=9)
    assert v.rejected is True
    assert v.reasons == ("target_new_collapse_rate=0.0500>=0.05",)   # UNCHANGED
    assert v.metrics == {
        "target_new_collapse_rate": 0.05,
        "band_new_collapse_rates": {"b200_299": 0.05},
        "target_lockin_count": 0,
        "lockin_baseline": 5,                       # ref=R0 -> r0_lockin
        "target_p95_mover_delta": 0.0,
        "mean_eff_children_reduction": 0.0,
        "mean_top_share_increase": 0.0,
    }


def test_safety_verdict_lockin_baseline_follows_ref_in_metrics():
    # Same rows, different ref -> the exposed lockin_baseline follows ref, exactly
    # as the (unchanged) gate picks its baseline.
    rows = [_tgt(lock_in=True) for _ in range(6)]
    v_off = dev_safety_verdict(rows, ref=ABSOLUTE_OFF, r0_lockin=10, absoff_lockin=3)
    v_r0 = dev_safety_verdict(rows, ref=R0, r0_lockin=10, absoff_lockin=3)
    assert v_off.rejected is True and v_off.metrics["lockin_baseline"] == 3   # absoff
    assert v_r0.rejected is False and v_r0.metrics["lockin_baseline"] == 10   # r0
    assert v_off.metrics["target_lockin_count"] == 6 == v_r0.metrics["target_lockin_count"]


def test_safety_verdict_control_metrics_and_clean_case():
    # Control subset only -> control metrics present, target metrics absent.
    ctl = [_ctl(control_flip_to_lower_prior=(i < 1), mover_delta=0.4) for i in range(10)]
    v = dev_safety_verdict(ctl, ref=R0, r0_lockin=5, absoff_lockin=5)
    assert v.rejected is True                       # flip 0.10 AND p95 0.40 both trip
    assert v.reasons == ("control_flip_rate=0.1000>=0.1",
                         "control_p95_mover_delta=0.4000>=0.35")
    assert v.metrics == {"control_flip_rate": 0.1, "control_p95_mover_delta": 0.4}
    # a wholly clean target set: not rejected, no reasons, metrics still populated
    clean = dev_safety_verdict([_tgt() for _ in range(30)],
                               ref=R0, r0_lockin=5, absoff_lockin=5)
    assert clean.rejected is False and clean.reasons == ()
    assert clean.metrics["target_new_collapse_rate"] == 0.0
    assert clean.metrics["target_p95_mover_delta"] == 0.0


# ---------------------------------------------------------------------------
# #4b -- verify_recomputed_controls: EXACT round-trip passes, a real diff aborts.
# ---------------------------------------------------------------------------

def _dev_row(sha="s1", role="target", band="b200_299", game_idx="7", ply="42",
             side="red"):
    return {"canonical_position_sha1": sha, "role": role, "branching_band": band,
            "game_idx": game_idx, "position_ply": ply, "side": side}


def _feats(root_value=0.1, top_share=0.5, eff=3.25, replies=4, collapsed=False,
           prior=0.02, rank=3, mass=0.4, stab=80, final_top=0.5):
    return {
        "root_value_stm": root_value, "top_share": top_share,
        "effective_children": eff, "replies": replies, "collapsed": collapsed,
        "top_move": 111, "top_move_prior": 0.3,
        "trace": {"selected_move_prior": prior, "selected_move_prior_rank": rank,
                  "explored_mass_at_stabilization": mass, "stabilization_sim": stab,
                  "final_root_top_share": final_top},
    }


def _persist_and_read(tmp_path, dev_rows, feats_by_label):
    """Write controls_cases.csv exactly as run_controls_stage does, read it back
    (csv-native strings)."""
    rows = []
    for r in dev_rows:
        for label, by_sha in feats_by_label.items():
            rows.append(_controls_case_row(r, label, by_sha[r["canonical_position_sha1"]]))
    path = tmp_path / "controls_cases.csv"
    _write_csv(str(path), CONTROLS_CASE_FIELDNAMES, rows)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_verify_recomputed_controls_faithful_round_trip(tmp_path):
    dev_rows = [_dev_row("s1"), _dev_row("s2", side="black")]
    # A None-valued field (stab=None -> stabilization_sim None) exercises the
    # None->'' round-trip; a not-exactly-representable float (0.1) exercises the
    # shortest-repr exactness.
    off = {"s1": _feats(root_value=0.1, stab=None), "s2": _feats(root_value=1/3)}
    r0 = {"s1": _feats(root_value=-0.2), "s2": _feats(root_value=0.0, collapsed=True)}
    feats_by_label = {ABSOLUTE_OFF.label: off, R0.label: r0}
    persisted = _persist_and_read(tmp_path, dev_rows, feats_by_label)

    recomputed = {ABSOLUTE_OFF.label: {}, R0.label: {}}
    for r in dev_rows:
        sha = r["canonical_position_sha1"]
        for label, by_sha in feats_by_label.items():
            recomputed[label][sha] = _controls_case_row(r, label, by_sha[sha])
    count, rows_sha1 = verify_recomputed_controls(persisted, recomputed)
    assert count == 4 and len(rows_sha1) == 40
    # deterministic hash: re-running on the same faithful recompute is identical
    count2, rows_sha1_2 = verify_recomputed_controls(persisted, recomputed)
    assert (count2, rows_sha1_2) == (count, rows_sha1)


def test_verify_recomputed_controls_aborts_on_real_diff(tmp_path):
    dev_rows = [_dev_row("s1")]
    off = {"s1": _feats(root_value=0.1)}
    r0 = {"s1": _feats(root_value=-0.2)}
    persisted = _persist_and_read(tmp_path, dev_rows,
                                  {ABSOLUTE_OFF.label: off, R0.label: r0})

    def _recompute(off_by_sha, r0_by_sha):
        return {ABSOLUTE_OFF.label: {s: _controls_case_row(dev_rows[0], ABSOLUTE_OFF.label, f)
                                     for s, f in off_by_sha.items()},
                R0.label: {s: _controls_case_row(dev_rows[0], R0.label, f)
                           for s, f in r0_by_sha.items()}}

    # a faithful recompute passes (sanity)
    count, _ = verify_recomputed_controls(persisted, _recompute(off, r0))
    assert count == 2
    # a 1-ULP float difference is caught (shortest-repr is injective over floats)
    off_ulp = {"s1": _feats(root_value=math.nextafter(0.1, 1.0))}
    with pytest.raises(ValueError):
        verify_recomputed_controls(persisted, _recompute(off_ulp, r0))
    # a bool flip is caught
    off_bool = {"s1": _feats(root_value=0.1, collapsed=True)}
    with pytest.raises(ValueError):
        verify_recomputed_controls(persisted, _recompute(off_bool, r0))
    # a missing (config, sha) recompute is caught
    with pytest.raises(ValueError):
        verify_recomputed_controls(persisted, {ABSOLUTE_OFF.label: {}, R0.label: {}})


# ---------------------------------------------------------------------------
# #4c -- complete candidate artifacts: joinable dev rows + numeric summaries.
# ---------------------------------------------------------------------------

def test_candidate_dev_records_joinable_by_sha(tmp_path):
    dev_rows = [_dev_row("s1", role="target"), _dev_row("s2", role="control")]
    cand = {"s1": _feats(root_value=0.3, top_share=0.6, collapsed=True),
            "s2": _feats(root_value=0.3)}
    ref = {"s1": _feats(root_value=0.1, top_share=0.5),
           "s2": _feats(root_value=0.1)}
    recs = _candidate_dev_records(dev_rows, cand, ref, "r0.20", ABSOLUTE_OFF.label)
    assert {r["canonical_sha1"] for r in recs} == {"s1", "s2"}       # joinable
    assert all(r["candidate_config"] == "r0.20"
               and r["reference"] == "absolute_off" for r in recs)
    trow = next(r for r in recs if r["role"] == "target")
    crow = next(r for r in recs if r["role"] == "control")
    assert trow["band"] == "b200_299" and trow["new_collapse"] is True     # collapsed & not ref
    assert crow["mover_delta"] == pytest.approx(0.2) and crow["band"] == ""
    # round-trips through the CSV writer with a stable union schema
    from scripts.GPU.alphazero.diagnose_fpu_policy_mass import CANDIDATE_DEV_ROW_FIELDNAMES
    path = tmp_path / "candidate_dev_rows.csv"
    _write_csv(str(path), CANDIDATE_DEV_ROW_FIELDNAMES, recs)
    with open(path, newline="") as f:
        back = list(csv.DictReader(f))
    assert len(back) == 2 and {r["canonical_sha1"] for r in back} == {"s1", "s2"}


def test_candidate_result_record_carries_numeric_summaries():
    clean = [_tgt() for _ in range(30)] + [_ctl() for _ in range(10)]
    v_off = dev_safety_verdict(clean, ref=ABSOLUTE_OFF, r0_lockin=5, absoff_lockin=5)
    v_r0 = dev_safety_verdict(clean, ref=R0, r0_lockin=5, absoff_lockin=5)
    a_rows = [dict(off_value=0.0, r_value=0.5 * V_REF, replies_ref=1.0,
                   replies_x=0.5, top_share_inc=0.0, new_collapse=False)]
    a_verdict = selected_a_verdict(a_rows)
    rec = _candidate_result_record(GRID[1], v_off, v_r0, a_verdict, safe=True)
    assert rec["config"] == "r0.20" and rec["reduction"] == 0.20 and rec["safe"] is True
    assert rec["metrics_vs_absolute_off"]["target_new_collapse_rate"] == 0.0
    assert rec["metrics_vs_r0"]["control_flip_rate"] == 0.0
    assert set(rec["selected_a_metrics"]) >= {
        "reply_reduction", "progress", "a_new_collapse", "a_top_share_inc", "passed"}
    json.dumps(rec)                                     # fully JSON-serializable
    # no selected-A (frozen_check) -> selected_a_metrics is None, still serializable
    rec2 = _candidate_result_record(GRID[0], v_off, v_r0, None, safe=False)
    assert rec2["selected_a_metrics"] is None and rec2["selected_a_passed"] is None
    json.dumps(rec2)
