"""v17 Task 4 -- protocol/provenance schemas, validators, and tamper checks.

Frozen design ref:
`docs/superpowers/specs/2026-07-24-v17-baseline-preserving-policy-mass-fpu-design.md`
(SHA-1 `944f358c0e3ef66503d2cbb56e31dabd145bafc2`) §2.4, §4, §12.

Every check here is pure: no evaluator, no checkpoint weights, no search. The
point of the module under test is that a violation is refused BEFORE any of
those are touched.
"""
import json
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import fpu_provenance
from scripts.GPU.alphazero import fpu_v17_protocol as proto
from scripts.GPU.alphazero import fpu_v17_provenance as prov

CKPT = "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"


@pytest.fixture
def clean_tree(monkeypatch):
    """The working tree is legitimately dirty during development, so the
    scientific-run refusal is exercised explicitly instead of incidentally."""
    monkeypatch.setattr(fpu_provenance, "worktree_clean", lambda: True)


@pytest.fixture
def out_root(tmp_path, monkeypatch):
    monkeypatch.setattr(prov, "OUTPUT_ROOT", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

def test_schemas_are_versioned():
    assert prov.SCHEMA_VERSION == 1
    assert proto.PROTOCOL_SCHEMA_VERSION == 1
    assert proto.CONFIG_SCHEMA_VERSION == 1


def test_frozen_grid_batching_and_formula():
    assert prov.GRID == (0.15, 0.20, 0.25, 0.35, 0.45)
    assert prov.BATCHING == (14, 48, 8)
    assert prov.MCTS_SIMS == 400
    assert prov.FORMULA_ID == "fpu_v17_baseline_policy_mass"
    assert prov.CONFIG_FIELD == "fpu_shipped_policy_mass_reduction"
    assert "Q_parent" not in prov.FORMULA


def test_run_kinds_and_abcd_is_scientific():
    assert set(prov.RUN_KINDS) == {"tooling_smoke", "development", "held_out",
                                   "abcd", "strength", "external_validation"}
    assert prov.is_scientific("abcd")
    assert not prov.is_scientific("tooling_smoke")
    for kind in prov.SCIENTIFIC_RUN_KINDS:
        assert prov.is_scientific(kind)
    with pytest.raises(prov.ProtocolViolation):
        prov.validate_run_kind("acceptance")


def test_frozen_seed_ranges_match_the_design():
    assert prov.SEED_RANGES["development"] == (20310000, 1600)
    assert prov.SEED_RANGES["held_out"] == (20312000, 2200)
    assert prov.SEED_RANGES["strength"] == (20320000, 800)
    assert prov.SEED_RANGES["external_validation"] == (20330000, 800)
    assert prov.SEED_RANGES["tooling_smoke"] == (20309000, 32)
    assert prov.SEED_RANGES["abcd"] is None      # replays fixed manifests


def test_frozen_design_sha1_is_the_real_file():
    assert fpu_provenance.file_sha1(prov.FROZEN_DESIGN_PATH) == prov.FROZEN_DESIGN_SHA1
    assert prov.verify_frozen_design() == prov.FROZEN_DESIGN_SHA1


def test_edited_frozen_design_is_refused(monkeypatch):
    monkeypatch.setattr(prov, "FROZEN_DESIGN_SHA1", "0" * 40)
    with pytest.raises(prov.ProtocolViolation, match="frozen"):
        prov.verify_frozen_design()


# ---------------------------------------------------------------------------
# §2.4 batching -- every override rejected
# ---------------------------------------------------------------------------

def test_batching_accepts_only_the_frozen_triple():
    assert prov.validate_batching((14, 48, 8)) == (14, 48, 8)
    for bad in [(14, 16, 8), (16, 48, 8), (14, 48, 0), (14, 48), (8, 48, 14)]:
        with pytest.raises(prov.ProtocolViolation, match="batching"):
            prov.validate_batching(bad)


def test_mctsconfig_default_stall_flush_is_rejected():
    """The exact silent drift §2.4 exists to catch: MCTSConfig defaults
    stall_flush_sims to 16, but v17 must derive 48."""
    from scripts.GPU.alphazero.mcts import MCTSConfig
    with pytest.raises(prov.ProtocolViolation, match="batching"):
        prov.validate_batching(MCTSConfig())
    from scripts.GPU.alphazero.eval_runner import EvalConfig, cfg_from
    ok = cfg_from(EvalConfig(mcts_sims=400, mcts_eval_batch_size=14,
                             mcts_stall_flush_sims=48))
    assert prov.validate_batching(ok) == (14, 48, 8)


# ---------------------------------------------------------------------------
# §4 / §13 coefficient grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r", [None, 0.0, 0.15, 0.20, 0.25, 0.35, 0.45])
def test_shipped_zero_and_grid_points_accepted(r):
    assert prov.validate_coefficient(r) == r


@pytest.mark.parametrize("r", [0.10, 0.30, 0.40, 0.55, 0.5])
def test_off_grid_coefficients_refused(r):
    with pytest.raises(prov.ProtocolViolation, match="frozen grid"):
        prov.validate_coefficient(r)


# ---------------------------------------------------------------------------
# Seed ranges and consumed evidence (§1.2)
# ---------------------------------------------------------------------------

def test_seed_range_must_match_the_frozen_stage_range():
    assert prov.validate_seed_range("development", 20310000, 1600)
    for bad in [(20310000, 1601), (20310001, 1600), (20311000, 1600)]:
        with pytest.raises(prov.ProtocolViolation, match="seed range"):
            prov.validate_seed_range("development", *bad)


def test_consumed_production_seeds_are_refused(monkeypatch):
    """Even if a stage range were mis-set to the v16 production reservoir, the
    disjointness rule refuses it."""
    monkeypatch.setitem(prov.SEED_RANGES, "development", (20300000, 4000))
    with pytest.raises(prov.ProtocolViolation, match="consumed evidence"):
        prov.validate_seed_range("development", 20300000, 4000)


def test_abcd_has_no_generated_seed_range():
    with pytest.raises(prov.ProtocolViolation, match="no generated seed range"):
        prov.validate_seed_range("abcd", 1, 1)


# ---------------------------------------------------------------------------
# §12 output root -- v16 roots refused
# ---------------------------------------------------------------------------

def test_output_must_be_under_the_v17_root():
    assert prov.validate_output_path(prov.OUTPUT_ROOT + "/x.json")


@pytest.mark.parametrize("v16", [
    "logs/eval/fpu_v16_policy_mass_v2/analysis/x.json",
    "logs/eval/fpu_policy_mass/x.json",
    "logs/eval/v16a_fpu_unbiased/x.json",
    "logs/eval/fpu_dev_corpus/x.json",
])
def test_v16_roots_are_refused_as_output_targets(v16):
    with pytest.raises(prov.ProtocolViolation, match="not under the v17 root"):
        prov.validate_output_path(v16)


# ---------------------------------------------------------------------------
# §12 clean worktree; tooling-smoke isolation
# ---------------------------------------------------------------------------

def test_scientific_run_kinds_require_a_clean_worktree(monkeypatch):
    monkeypatch.setattr(fpu_provenance, "worktree_clean", lambda: False)
    for kind in prov.SCIENTIFIC_RUN_KINDS:
        with pytest.raises(prov.ProtocolViolation, match="clean worktree"):
            prov.require_clean_worktree(kind)
    prov.require_clean_worktree("tooling_smoke")     # allowed to run dirty


def test_tooling_smoke_artifacts_refused_in_scientific_modes():
    smoke = {"run_kind": "tooling_smoke"}
    for kind in prov.SCIENTIFIC_RUN_KINDS:
        with pytest.raises(prov.ProtocolViolation, match="tooling_smoke"):
            prov.require_not_tooling_smoke(smoke, consumer_run_kind=kind)
    prov.require_not_tooling_smoke(smoke, consumer_run_kind="tooling_smoke")


# ---------------------------------------------------------------------------
# Provenance record
# ---------------------------------------------------------------------------

def test_provenance_records_the_required_fields(clean_tree):
    rec = prov.build_provenance(run_kind="development", coefficient=0.35,
                                checkpoints={"a": CKPT},
                                source_files=["scripts/GPU/alphazero/mcts.py"])
    assert rec["formula_id"] == prov.FORMULA_ID
    assert rec["grid"] == list(prov.GRID)
    assert rec["frozen_design"]["sha1"] == prov.FROZEN_DESIGN_SHA1
    assert rec["mcts"] == {"n_simulations": 400, "add_noise": False,
                           "eval_batch_size": 14, "stall_flush_sims": 48,
                           "pending_virtual_visits": 8}
    assert rec["checkpoints"]["a"] == "209cf2d4fd24a48553d259dd71b4954867b9473e"
    # source hashes are keyed by BASENAME (checkout-location-independent),
    # which is `fpu_provenance.source_file_sha1s`'s documented convention
    assert rec["source_file_sha1s"]["mcts.py"] == \
        fpu_provenance.file_sha1("scripts/GPU/alphazero/mcts.py")
    assert rec["scientific"] is True
    assert "scientific_interpretation_forbidden" not in rec


@pytest.mark.parametrize("kwargs", [
    {"checkpoints": {"a": "checkpoints/does_not_exist.safetensors"}},
    {"checkpoints": {"a": ""}},
    {"source_files": ["scripts/GPU/alphazero/does_not_exist.py"]},
])
def test_unreadable_inputs_are_refused_not_recorded_as_sentinels(clean_tree, kwargs):
    """`file_sha1` returns "none"/"missing" rather than raising, which is right
    for a fingerprint but must never be frozen into a v17 protocol in place of
    a real hash."""
    with pytest.raises(prov.ProtocolViolation, match="placeholder hash"):
        prov.build_provenance(run_kind="development", **kwargs)


def test_tooling_smoke_provenance_is_self_labelling():
    rec = prov.build_provenance(run_kind="tooling_smoke")
    assert rec["scientific"] is False
    assert rec["scientific_interpretation_forbidden"] is True


def test_provenance_has_no_timestamp(clean_tree):
    """§12: canonical artifacts must be byte-identical across reruns."""
    blob = json.dumps(prov.build_provenance(run_kind="development"))
    for banned in ("timestamp", "generated_at", "datetime", "_at\":"):
        assert banned not in blob
    assert prov.build_provenance(run_kind="development") == \
        prov.build_provenance(run_kind="development")


# ---------------------------------------------------------------------------
# Protocol / config lifecycle
# ---------------------------------------------------------------------------

def _dev_protocol():
    return proto.build_protocol(run_kind="development", coefficient=None,
                                base_seed=20310000, games=1600,
                                checkpoints={"a": CKPT})


def test_build_protocol_enforces_the_frozen_rules(clean_tree):
    doc = _dev_protocol()
    assert doc["schema_version"] == 1 and doc["artifact_kind"] == "protocol"
    with pytest.raises(prov.ProtocolViolation, match="frozen grid"):
        proto.build_protocol(run_kind="development", coefficient=0.55,
                             base_seed=20310000, games=1600)
    with pytest.raises(prov.ProtocolViolation, match="seed range"):
        proto.build_protocol(run_kind="development", base_seed=20310000, games=99)
    with pytest.raises(prov.ProtocolViolation, match="generates no games"):
        proto.build_protocol(run_kind="abcd", base_seed=1, games=1)


def test_build_protocol_refuses_a_dirty_tree_for_scientific_runs(monkeypatch):
    monkeypatch.setattr(fpu_provenance, "worktree_clean", lambda: False)
    with pytest.raises(prov.ProtocolViolation, match="clean worktree"):
        proto.build_protocol(run_kind="development", base_seed=20310000, games=1600)
    proto.build_protocol(run_kind="tooling_smoke", base_seed=20309000, games=32)


def test_emit_is_canonical_and_byte_identical_twice(clean_tree, out_root):
    doc = _dev_protocol()
    path = out_root / "protocol.json"
    assert proto.emit(path, doc) is proto.WriteStatus.WRITTEN
    first = path.read_bytes()
    assert proto.emit(path, doc) is proto.WriteStatus.UNCHANGED
    assert path.read_bytes() == first
    assert first.endswith(b"\n")
    assert json.loads(first) == json.loads(json.dumps(doc))


def test_emit_refuses_to_overwrite_different_bytes(clean_tree, out_root):
    path = out_root / "protocol.json"
    proto.emit(path, _dev_protocol())
    before = path.read_bytes()
    with pytest.raises(ValueError, match="refusing to overwrite"):
        proto.emit(path, proto.build_protocol(
            run_kind="tooling_smoke", base_seed=20309000, games=32))
    assert path.read_bytes() == before          # untouched


def test_emit_refuses_a_v16_root(clean_tree):
    with pytest.raises(prov.ProtocolViolation, match="not under the v17 root"):
        proto.emit("logs/eval/fpu_v16_policy_mass_v2/protocol.json", _dev_protocol())


def test_config_is_a_pure_function_of_the_protocol(clean_tree):
    doc = _dev_protocol()
    assert proto.derive_config(doc) == proto.derive_config(doc)
    cfg = proto.derive_config(doc)
    assert cfg["shipped_branch"] is True
    assert cfg["seed_range"] == [20310000, 20311600]
    assert cfg["mcts"]["stall_flush_sims"] == 48
    assert cfg["frozen_design_sha1"] == prov.FROZEN_DESIGN_SHA1


def test_zero_coefficient_takes_the_shipped_branch_in_the_config(clean_tree):
    doc = proto.build_protocol(run_kind="development", coefficient=0.0,
                               base_seed=20310000, games=1600)
    cfg = proto.derive_config(doc)
    assert cfg["coefficient"] == 0.0 and cfg["shipped_branch"] is True
    pos = proto.derive_config(proto.build_protocol(
        run_kind="development", coefficient=0.35,
        base_seed=20310000, games=1600))
    assert pos["shipped_branch"] is False


def test_derive_config_rejects_a_non_protocol_document():
    with pytest.raises(prov.ProtocolViolation, match="expected a protocol"):
        proto.derive_config({"artifact_kind": "config"})


# ---------------------------------------------------------------------------
# Tamper detection -- re-derive and byte-compare
# ---------------------------------------------------------------------------

def _pair(tmp, clean=True):
    doc = _dev_protocol()
    ppath, cpath = tmp / "protocol.json", tmp / "config.json"
    proto.emit(ppath, doc)
    proto.emit(cpath, proto.derive_config(doc))
    return ppath, cpath


def test_untampered_pair_verifies(clean_tree, out_root):
    ppath, cpath = _pair(out_root)
    cfg = proto.load_verified(ppath, cpath, consumer_run_kind="development")
    assert cfg["run_kind"] == "development"


@pytest.mark.parametrize("field,value", [
    ("coefficient", 0.35), ("base_seed", 20310001), ("games", 1599),
    ("board_size", 30), ("run_kind", "held_out"),
])
def test_tampered_protocol_field_is_caught(clean_tree, out_root, field, value):
    ppath, cpath = _pair(out_root)
    doc = proto.load_json(ppath)
    doc[field] = value
    ppath.write_bytes(json.dumps(doc).encode())
    with pytest.raises(prov.ProtocolViolation):
        proto.load_verified(ppath, cpath, consumer_run_kind="development")


@pytest.mark.parametrize("mutate", [
    lambda c: c.__setitem__("coefficient", 0.35),
    lambda c: c["mcts"].__setitem__("stall_flush_sims", 16),
    lambda c: c["mcts"].__setitem__("n_simulations", 800),
    lambda c: c.__setitem__("shipped_branch", False),
    lambda c: c.__setitem__("seed_range", [20310000, 20310800]),
    lambda c: c.__setitem__("frozen_design_sha1", "0" * 40),
])
def test_tampered_config_field_is_caught(clean_tree, out_root, mutate):
    ppath, cpath = _pair(out_root)
    cfg = proto.load_json(cpath)
    mutate(cfg)
    cpath.write_bytes(json.dumps(cfg).encode())
    with pytest.raises(prov.ProtocolViolation, match="byte-match"):
        proto.load_verified(ppath, cpath, consumer_run_kind="development")


def test_consumer_run_kind_must_match_the_protocol(clean_tree, out_root):
    ppath, cpath = _pair(out_root)
    with pytest.raises(prov.ProtocolViolation, match="consumer"):
        proto.load_verified(ppath, cpath, consumer_run_kind="held_out")


def test_smoke_artifacts_cannot_feed_a_scientific_stage(clean_tree, out_root):
    doc = proto.build_protocol(run_kind="tooling_smoke", base_seed=20309000,
                               games=32)
    ppath, cpath = out_root / "p.json", out_root / "c.json"
    proto.emit(ppath, doc)
    proto.emit(cpath, proto.derive_config(doc))
    with pytest.raises(prov.ProtocolViolation, match="tooling_smoke"):
        proto.load_verified(ppath, cpath, consumer_run_kind="development")


def test_exit_codes_are_exported():
    assert (proto.EXIT_OK, proto.EXIT_USAGE, proto.EXIT_MISMATCH,
            proto.EXIT_GATE_FAIL) == (0, 2, 3, 4)


# ---------------------------------------------------------------------------
# Import purity -- a fresh subprocess, so another test module's earlier import
# in this same session cannot mask a violation.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", ["fpu_v17_provenance", "fpu_v17_protocol"])
def test_module_import_pulls_no_gpu_or_mlx(module):
    script = (f"import sys\nimport scripts.GPU.alphazero.{module}\n"
              "print(sorted(k for k in sys.modules if 'mlx' in k or 'torch' in k))\n")
    out = subprocess.run([sys.executable, "-c", script],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]"


# ===========================================================================
# Adversarial review round 1. Each block closes a hole found by direct probing
# after the first Task 4 implementation, where 61 focused tests passed while
# the protocol was still permissive.
# ===========================================================================

# --- (1) the match-smoke range must not widen any other stage --------------

@pytest.mark.parametrize("kind", ["development", "held_out", "strength",
                                  "external_validation"])
def test_match_smoke_seeds_are_refused_for_every_other_run_kind(kind):
    """§5.4's [20309100, 20309108) block lives INSIDE the tooling_smoke stage.
    It previously satisfied every generated run kind."""
    with pytest.raises(prov.ProtocolViolation, match="seed range"):
        prov.validate_seed_range(kind, *prov.MATCH_SMOKE_SEEDS)


def test_match_smoke_seeds_remain_valid_for_tooling_smoke():
    assert prov.validate_seed_range("tooling_smoke", *prov.MATCH_SMOKE_SEEDS)
    assert prov.validate_seed_range("tooling_smoke", 20309000, 32)


# --- (2) the config binds the COMPLETE protocol, provenance included -------

def test_config_embeds_a_canonical_protocol_sha1(clean_tree):
    doc = _dev_protocol()
    cfg = proto.derive_config(doc)
    assert cfg["protocol_sha1"] == proto.protocol_sha1(doc)
    assert len(cfg["protocol_sha1"]) == 40


@pytest.mark.parametrize("mutate", [
    lambda p: p["provenance"].__setitem__("formula_id", "overwritten"),
    lambda p: p["provenance"].__setitem__("git_commit", "0" * 40),
    lambda p: p["provenance"].__setitem__("worktree_clean", False),
    lambda p: p["provenance"]["mcts"].__setitem__("stall_flush_sims", 16),
    lambda p: p["provenance"].__setitem__("grid", [0.55]),
    lambda p: p["provenance"]["frozen_design"].__setitem__("sha1", "0" * 40),
])
def test_provenance_only_tampering_is_detected(clean_tree, out_root, mutate):
    """No derived field reads the provenance block, so before the
    protocol_sha1 binding these edits left the config verifying."""
    ppath, cpath = _pair(out_root)
    doc = proto.load_json(ppath)
    mutate(doc)
    ppath.write_bytes(json.dumps(doc).encode())
    with pytest.raises(prov.ProtocolViolation, match="byte-match"):
        proto.load_verified(ppath, cpath, consumer_run_kind="development")


def test_protocol_sha1_is_canonical_not_insertion_ordered(clean_tree):
    doc = _dev_protocol()
    reordered = dict(reversed(list(doc.items())))
    assert proto.protocol_sha1(reordered) == proto.protocol_sha1(doc)


# --- (3) schema versions and exact key sets --------------------------------

@pytest.mark.parametrize("version", [0, 2, 999, None, "1"])
def test_unsupported_protocol_schema_version_is_refused(clean_tree, version):
    with pytest.raises(prov.ProtocolViolation, match="schema_version"):
        proto.derive_config({**_dev_protocol(), "schema_version": version})


def test_unsupported_config_schema_version_is_refused(clean_tree):
    doc = _dev_protocol()
    cfg = {**proto.derive_config(doc), "schema_version": 999}
    with pytest.raises(prov.ProtocolViolation, match="schema_version"):
        proto.verify_config_matches(doc, cfg)


@pytest.mark.parametrize("key", sorted(proto.PROTOCOL_KEYS - {"schema_version",
                                                              "artifact_kind"}))
def test_missing_protocol_key_is_refused(clean_tree, key):
    doc = {k: v for k, v in _dev_protocol().items() if k != key}
    with pytest.raises(prov.ProtocolViolation, match="missing required keys"):
        proto.derive_config(doc)


def test_unknown_protocol_key_is_refused(clean_tree):
    with pytest.raises(prov.ProtocolViolation, match="unknown keys"):
        proto.derive_config({**_dev_protocol(), "smuggled": 1})


def test_config_key_set_is_exact(clean_tree):
    assert set(proto.derive_config(_dev_protocol())) == set(proto.CONFIG_KEYS)


# --- (4) frozen scalar types and settings ----------------------------------

@pytest.mark.parametrize("bad", [False, True, "0.35", complex(0.35)])
def test_non_numeric_coefficients_are_refused(bad):
    """`bool` subclasses `int`, so `False == 0.0` would have silently selected
    the shipped branch."""
    with pytest.raises(prov.ProtocolViolation, match="coefficient"):
        prov.validate_coefficient(bad)


@pytest.mark.parametrize("base,games", [
    (20310000.0, 1600), (20310000, 1600.0), (20310000.0, 1600.0),
    (True, 1600), ("20310000", 1600),
])
def test_non_integer_seeds_and_counts_are_refused(base, games):
    with pytest.raises(prov.ProtocolViolation, match="must be an int"):
        prov.validate_seed_range("development", base, games)


@pytest.mark.parametrize("bad", [30, 19, 24.0, "24", True])
def test_board_size_is_frozen_at_24(bad):
    with pytest.raises(prov.ProtocolViolation):
        prov.validate_board_size(bad)


def test_frozen_board_size_is_accepted():
    assert prov.validate_board_size(24) == prov.BOARD_SIZE == 24


def test_build_protocol_refuses_a_non_frozen_board_size(clean_tree):
    with pytest.raises(prov.ProtocolViolation, match="board_size"):
        proto.build_protocol(run_kind="development", base_seed=20310000,
                             games=1600, board_size=30)


# --- (5) extra may not overwrite protected provenance ----------------------

@pytest.mark.parametrize("extra", [
    {"formula_id": "overwritten"},
    {"scientific": True},
    {"scientific_interpretation_forbidden": False},
    {"grid": [0.55]},
    {"run_kind": "development"},
    {"frozen_design": {"sha1": "0" * 40}},
    {"identities": {}},
    {"worktree_clean": True},
])
def test_extra_cannot_overwrite_protected_provenance_keys(extra):
    with pytest.raises(prov.ProtocolViolation, match="protected provenance keys"):
        prov.build_provenance(run_kind="tooling_smoke", extra=extra)


def test_namespaced_extra_is_still_allowed():
    rec = prov.build_provenance(run_kind="tooling_smoke",
                                extra={"operator_note": "smoke rerun"})
    assert rec["operator_note"] == "smoke rerun"
    assert rec["formula_id"] == prov.FORMULA_ID
    assert rec["scientific"] is False


# --- (6) complete identity set ---------------------------------------------

def test_identity_fields_exist_and_default_to_null():
    ids = prov.build_provenance(run_kind="tooling_smoke")["identities"]
    assert set(ids) == {"manifest_sha1", "source_index_sha1", "replay_data_sha1"}
    assert all(v is None for v in ids.values())


def test_identity_fields_are_hashed_when_supplied(clean_tree, tmp_path):
    manifest = tmp_path / "m.csv"
    manifest.write_text("case_id\n1\n")
    index = tmp_path / "i.json"
    index.write_text("{}")
    replay = tmp_path / "r.json"
    replay.write_text('{"moves": []}')
    ids = prov.build_provenance(run_kind="development", manifest=str(manifest),
                                source_index=str(index),
                                replay_paths=[str(replay)])["identities"]
    assert ids["manifest_sha1"] == fpu_provenance.file_sha1(str(manifest))
    assert ids["source_index_sha1"] == fpu_provenance.file_sha1(str(index))
    assert ids["replay_data_sha1"] == fpu_provenance.replay_data_sha1([str(replay)])
    assert not set(ids.values()) & set(prov.SENTINEL_HASHES)


@pytest.mark.parametrize("kwargs", [
    {"manifest": "logs/eval/does_not_exist.csv"},
    {"source_index": "logs/eval/does_not_exist.json"},
])
def test_unreadable_identity_inputs_are_refused(clean_tree, kwargs):
    with pytest.raises(prov.ProtocolViolation, match="placeholder hash"):
        prov.build_provenance(run_kind="development", **kwargs)


def test_replay_data_hash_tracks_contents_not_paths(clean_tree, tmp_path):
    replay = tmp_path / "a.json"
    replay.write_text('{"x": 1}')
    first = prov.build_provenance(
        run_kind="development", replay_paths=[str(replay)])["identities"]
    replay.write_text('{"x": 2}')
    second = prov.build_provenance(
        run_kind="development", replay_paths=[str(replay)])["identities"]
    assert first["replay_data_sha1"] != second["replay_data_sha1"]


def test_duplicate_source_basenames_are_refused(clean_tree, tmp_path):
    """`source_file_sha1s` keys by basename, which would silently collapse two
    same-named files in different packages into a single identity."""
    for pkg in ("p1", "p2"):
        (tmp_path / pkg).mkdir()
        (tmp_path / pkg / "mcts.py").write_text(f"# {pkg}\n")
    with pytest.raises(prov.ProtocolViolation, match="basenames are not unique"):
        prov.build_provenance(run_kind="development", source_files=[
            str(tmp_path / "p1" / "mcts.py"), str(tmp_path / "p2" / "mcts.py")])
