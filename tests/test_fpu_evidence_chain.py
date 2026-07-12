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
import json
from dataclasses import dataclass

import pytest

from scripts.GPU.alphazero import fpu_provenance as prov
from scripts.GPU.alphazero.diagnose_fpu_policy_mass import (
    GRID, build_run_fingerprint, validate_controls_fingerprint,
    require_frozen_matches_tuning, validate_selected_a_mode)


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
    assert set(sel["source_file_sha1s"]) == {
        "diagnose_fpu_policy_mass.py", "mcts.py", "build_fpu_dev_corpus.py"}

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
