"""Sec 2.2.2: parameterize for A/6400 WITHOUT changing v17 default behavior.
No GPU: argument plumbing and authentication wiring only."""
import inspect
import json
from pathlib import Path

import pytest

from scripts.GPU.alphazero import capture_v18_a6400 as M

SIX_K_REF = ("logs/eval/v15_budget_check/a_predrop_base_6400sims.csv/"
             "position_probe_cases.csv")
SIX_K_REF_SHA1 = "a17d4737c747e2799253bebbc3d0261e0e697114"


def test_v17_defaults_are_unchanged():
    assert M.MCTS_SIMS == 400
    assert (M.EVAL_BATCH_SIZE, M.STALL_FLUSH_SIMS, M.PENDING_VIRTUAL_VISITS) == (14, 48, 8)
    assert set(M.GATES) == {"A", "B", "C", "D"}
    assert M.GATES["A"]["base_seed"] == 20260616
    assert M.GATES["A"]["seed_rule"] == "base ^ game_idx ^ position_ply"


def test_mcts_config_defaults_to_400_and_accepts_an_override():
    assert M.mcts_config().n_simulations == 400
    assert M.mcts_config(6400).n_simulations == 6400


def test_mcts_config_preserves_the_batching_triple_at_any_sim_count():
    cfg = M.mcts_config(6400)
    assert (cfg.eval_batch_size, cfg.stall_flush_sims, cfg.pending_virtual_visits) == (14, 48, 8)


def test_capture_signature_defaults_reproduce_v17_behavior():
    p = inspect.signature(M.capture).parameters
    assert p["mode"].default == "v17_prechange_abcd"


def test_cli_exposes_only_mode_and_out():
    ns = M.build_parser().parse_args([])
    assert ns.mode == "v17_prechange_abcd" and ns.out == M.OUT
    # No caller-nominated reference: the mode fixes every scientific parameter.
    assert not hasattr(ns, "auth_source")
    assert not hasattr(ns, "mcts_sims")
    assert not hasattr(ns, "gates")


def test_v18_mode_is_fully_self_constrained():
    m = M.MODES["v18_preflight_a6400"]
    assert m["gates"] == ("A",)
    assert m["mcts_sims"] == 6400
    assert m["auth_source"] == SIX_K_REF
    assert m["auth_sha1"] == SIX_K_REF_SHA1
    assert m["base_seed"] == 20260616
    assert m["seed_rule"] == "base ^ game_idx ^ position_ply"
    assert m["batching"] == (14, 48, 8)


def test_default_mode_emits_the_existing_schema_with_no_new_keys():
    """Adding mcts_sims / auth_source / gate-list fields unconditionally would
    change the v17 default bytes and break its own byte-identity regression."""
    v17_keys = set(M.document_keys("v17_prechange_abcd"))
    v18_keys = set(M.document_keys("v18_preflight_a6400"))
    assert v17_keys == set(M.LEGACY_DOCUMENT_KEYS)
    assert v18_keys - v17_keys                      # new fields exist...
    assert not (v17_keys - set(M.LEGACY_DOCUMENT_KEYS))   # ...but only in v18


def test_unknown_mode_rejected():
    with pytest.raises((SystemExit, ValueError, KeyError)):
        M.capture(mode="whatever")


def test_case_identity_is_the_full_tuple_not_just_case_id():
    # Same case_id, different ply: equal case_id SETS would have passed.
    with pytest.raises(ValueError, match="case set"):
        M.authenticate_against(source_rows=[_src()],
                               captured=[_cap(position_ply=9)])


# --- A/6,400 reference bundle: BUILDER tests --------------------------------
# The builder's job is correct, canonical, atomic emission and correctly
# COMPUTED fields. Tamper attacks live with load_verified_a6400_bundle in
# Task 9, which is what must resist a hand-edited file.

# The frozen key set, written out INDEPENDENTLY of the implementation. Comparing
# against M.A6400_BUNDLE_KEYS would let an added key plus an edited tuple pass.
EXPECTED_BUNDLE_KEYS = {
    "artifact_kind", "schema_version",
    "capture_run_1_path", "capture_run_1_sha1",
    "capture_run_2_path", "capture_run_2_sha1",
    "byte_identical",
    "historical_source_path", "historical_source_sha1",
    "authentication",
    "run_kind", "scientific_interpretation_forbidden",
}


# Tracked 30-case fixture, committed under
#   tests/golden/a6400_bundle_fixture/{source_rows.json,capture.json}
# A clean checkout cannot derive these identities from the real artifact: the
# frozen A cases CSV lives under gitignored logs/. Revision 12 called
# `M.testing.synthetic_capture`, a test-only namespace that does not exist and
# must not be added to a production module -- fixture construction belongs here.
FIXTURE_DIR = Path(__file__).parent / "golden" / "a6400_bundle_fixture"


def _fixture(name):
    return json.loads((FIXTURE_DIR / name).read_text())


@pytest.fixture
def source_rows():
    rows = _fixture("source_rows.json")
    assert len(rows) == 30
    return rows


@pytest.fixture
def capture_doc():
    """One capture document carrying the full 30-case set, matching source_rows
    so `authenticate_against`'s exact-30 rule is satisfiable."""
    doc = _fixture("capture.json")
    assert len(doc["cases"]) == 30
    return doc


@pytest.fixture
def captures(tmp_path, capture_doc):
    """Two byte-identical captures plus a third that genuinely differs."""
    import json
    other = json.loads(json.dumps(capture_doc))
    other["cases"][0]["top_share_repr"] = "0.99"
    run1, run2, run2d = (tmp_path / "r1.json", tmp_path / "r2.json",
                         tmp_path / "r2d.json")
    for p, d in ((run1, capture_doc), (run2, capture_doc), (run2d, other)):
        p.write_text(json.dumps(d, sort_keys=True))
    return str(run1), str(run2), str(run2d)


@pytest.fixture
def frozen_source(monkeypatch, source_rows):
    """Point the wrapper's frozen historical source loader at the fixture, so
    the 30-case authentication succeeds without the real 6,400 artifact."""
    monkeypatch.setattr(M, "_load_frozen_a6400_source", lambda: source_rows)


def test_frozen_source_loader_opens_exactly_the_frozen_path(monkeypatch,
                                                            source_rows):
    """BEHAVIORAL, not documentary.

    Revision 13 asserted the path appeared in the loader's DOCSTRING, which a
    loader reading a byte-identical copy from anywhere else would also pass.
    Instrument the real read/hash boundary and assert the path actually used.
    """
    frozen = M.MODES["v18_preflight_a6400"]["auth_source"]
    raw = b'[{"case_id": "x"}]'
    seen = []
    monkeypatch.setattr(M, "_read_bytes",
                        lambda p: (seen.append(("read", p)), raw)[1])
    monkeypatch.setattr(M, "sha1_bytes",
                        lambda b: (seen.append(("hash", b)),
                                   M.MODES["v18_preflight_a6400"]["auth_sha1"])[1])
    monkeypatch.setattr(M, "_parse_source_rows",
                        lambda b: (seen.append(("parse", b)), source_rows)[1])

    assert M._load_frozen_a6400_source() == source_rows

    # Exactly one read, of the frozen path, then hash, then parse.
    assert [kind for kind, _ in seen] == ["read", "hash", "parse"]
    assert seen[0][1] == frozen
    # THE decisive assertion: the hashed object and the parsed object are the
    # SAME bytes, not two reads that happened to agree. Revision 15 hashed a
    # path and parsed a path, so a file changed in between would authenticate
    # one sequence and parse another -- ordering alone could not catch it.
    assert seen[1][1] is raw
    assert seen[2][1] is raw


def test_frozen_source_loader_refuses_a_hash_mismatch_without_parsing(
        monkeypatch, source_rows):
    parses = []
    monkeypatch.setattr(M, "_read_bytes", lambda p: b"whatever")
    monkeypatch.setattr(M, "sha1_bytes", lambda b: "0" * 40)
    monkeypatch.setattr(M, "_parse_source_rows",
                        lambda b: (parses.append(b), source_rows)[1])
    with pytest.raises(ValueError, match="a17d4737"):
        M._load_frozen_a6400_source()
    assert parses == []         # authentication precedes any parse


def test_frozen_source_loader_takes_no_path_argument():
    """Path substitution must be impossible by construction: the frozen path is
    not a parameter."""
    import inspect
    assert inspect.signature(M._load_frozen_a6400_source).parameters == {}


def test_bundle_emission_is_canonical_and_byte_reproducible(tmp_path, captures, frozen_source):
    run1, run2, _ = captures
    a = M.build_a6400_reference_bundle(run1, run2, str(tmp_path / "a.json"))
    b = M.build_a6400_reference_bundle(run1, run2, str(tmp_path / "b.json"))
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert a == b


def test_builder_returns_the_sha1_of_the_bytes_it_wrote(tmp_path, captures, frozen_source):
    import hashlib
    run1, run2, _ = captures
    p = tmp_path / "bundle.json"
    returned = M.build_a6400_reference_bundle(run1, run2, str(p))
    assert returned == hashlib.sha1(p.read_bytes()).hexdigest()


def test_bundle_never_contains_its_own_digest(tmp_path, captures, frozen_source):
    """A file cannot carry the hash of its own complete bytes."""
    run1, run2, _ = captures
    p = tmp_path / "bundle.json"
    sha = M.build_a6400_reference_bundle(run1, run2, str(p))
    assert sha not in p.read_text()


def test_builder_refuses_differing_captures_and_writes_nothing(
        tmp_path, captures, frozen_source):
    """A reference bundle over two DIFFERENT captures is not a valid artifact:
    its single `authentication` block would be undefined as to which run it
    describes. Refuse rather than record byte_identical False."""
    run1, _run2, run2_different = captures
    p = tmp_path / "bundle.json"
    with pytest.raises(ValueError, match="byte-identical"):
        M.build_a6400_reference_bundle(run1, run2_different, str(p))
    assert not p.exists()


def test_builder_refuses_when_authentication_fails_and_writes_nothing(
        tmp_path, captures, source_rows, monkeypatch):
    import copy
    run1, run2, _ = captures
    bad = copy.deepcopy(source_rows)
    bad[0]["probe_black_root_value"] = "0.99"
    monkeypatch.setattr(M, "_load_frozen_a6400_source", lambda: bad)
    p = tmp_path / "bundle.json"
    with pytest.raises(ValueError):
        M.build_a6400_reference_bundle(run1, run2, str(p))
    assert not p.exists()


def test_authentication_report_is_returned_and_stored_once(
        tmp_path, captures, source_rows, capture_doc, frozen_source):
    """The bundle's 30-entry block is exactly what authenticate_against returns,
    and both captures must produce canonically identical reports."""
    import json as _json
    run1, run2, _ = captures
    p = tmp_path / "bundle.json"
    M.build_a6400_reference_bundle(run1, run2, str(p))
    expected = M.authenticate_against(source_rows, capture_doc["cases"])
    assert _json.loads(p.read_text())["authentication"] == expected


def test_byte_identical_is_computed_not_copied(tmp_path, captures, frozen_source):
    """On the valid path the field is an INVARIANT, and the builder computes it
    rather than accepting any claim."""
    import json
    run1, run2, _ = captures
    p = tmp_path / "bundle.json"
    M.build_a6400_reference_bundle(run1, run2, str(p))
    assert json.loads(p.read_text())["byte_identical"] is True


def test_authentication_block_covers_all_thirty_cases(tmp_path, captures,
                                                      frozen_source):
    import json
    run1, run2, _ = captures
    p = tmp_path / "bundle.json"
    M.build_a6400_reference_bundle(run1, run2, str(p))
    assert len(json.loads(p.read_text())["authentication"]) == 30


def test_bundle_has_the_exact_frozen_key_set(tmp_path, captures, frozen_source):
    import json
    run1, run2, _ = captures
    p = tmp_path / "bundle.json"
    M.build_a6400_reference_bundle(run1, run2, str(p))
    # Compared against the INDEPENDENT literal above, not M.A6400_BUNDLE_KEYS.
    assert set(json.loads(p.read_text())) == EXPECTED_BUNDLE_KEYS


def test_implementation_key_tuple_matches_the_independent_literal():
    assert set(M.A6400_BUNDLE_KEYS) == EXPECTED_BUNDLE_KEYS


def test_bundle_emission_is_atomic(tmp_path, captures, frozen_source, monkeypatch):
    """Failure AFTER the temp file exists must leave the destination absent and
    no temp file behind.

    Revision 10's version passed a nonexistent input, which fails while opening
    it -- before any output write is attempted -- so it proved nothing about
    atomicity. Force the failure at the rename instead.
    """
    import os
    run1, run2, _ = captures
    p = tmp_path / "bundle.json"

    def boom(src, dst):
        raise OSError("simulated failure during replace")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        M.build_a6400_reference_bundle(run1, run2, str(p))
    assert not p.exists()
    assert list(tmp_path.glob("*.tmp*")) == []


def test_atomic_write_leaves_an_existing_destination_unchanged(
        tmp_path, captures, frozen_source, monkeypatch):
    import os
    run1, run2, _ = captures
    p = tmp_path / "bundle.json"
    p.write_text("PRIOR CONTENT")
    monkeypatch.setattr(os, "replace",
                        lambda s, d: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        M.build_a6400_reference_bundle(run1, run2, str(p))
    assert p.read_text() == "PRIOR CONTENT"


def test_unknown_gate_rejected():
    with pytest.raises((SystemExit, ValueError, KeyError)):
        M.resolve_gates("A,Z")


def test_auth_source_case_set_must_match_exactly():
    """Authentication compares against a DIFFERENT artifact, so a case-set
    mismatch must abort rather than authenticate a subset."""
    with pytest.raises(ValueError, match="case set"):
        M.authenticate_against(
            source_rows=[{"case_id": "x", "probe_black_root_value": "0.0",
                          "probe_top1_share": "0.5"}],
            captured=[{"case_id": "y", "recomputed_black_value_repr": "0.0",
                       "top_share_repr": "0.5"}])


def _src(**over):
    """A source row carrying the FULL six-field identity, so an authentication
    test fails on the condition it names rather than on a missing key."""
    row = {"case_id": "x", "game_idx": 1, "position_ply": 7,
           "side_to_move": "black", "replay_path": "p",
           "canonical_state_sha1": "a" * 40,
           "probe_black_root_value": "0.25", "probe_top1_share": "0.50"}
    row.update(over)
    return row


def _cap(**over):
    row = {"case_id": "x", "game_idx": 1, "position_ply": 7,
           "side_to_move": "black", "replay_path": "p",
           "canonical_state_sha1": "a" * 40,
           "recomputed_black_value_repr": "0.25", "top_share_repr": "0.50"}
    row.update(over)
    return row


def test_auth_checks_value_and_top_share_not_value_alone():
    with pytest.raises(ValueError, match="top_share"):
        M.authenticate_against(source_rows=[_src()],
                               captured=[_cap(top_share_repr="0.99")])


def test_auth_checks_value_too():
    with pytest.raises(ValueError, match="value"):
        M.authenticate_against(source_rows=[_src()],
                               captured=[_cap(recomputed_black_value_repr="0.99")])


def test_auth_passes_when_both_statistics_and_full_identity_match():
    M.authenticate_against(source_rows=[_src()], captured=[_cap()])


def test_record_kind_is_stamped_and_interpretation_forbidden():
    assert M.record_envelope("v18_preflight_a6400")["run_kind"] == "v18_preflight_a6400"
    assert M.record_envelope("v18_preflight_a6400")[
        "scientific_interpretation_forbidden"] is True


def test_identity_normalizes_csv_strings_against_captured_integers():
    """The historical CSV supplies "347"/"73"; a capture emits 347/73. Without
    normalization every one of the 30 cases reads as BOTH missing and
    unexpected -- a total failure that says nothing about the data."""
    src = dict(_src(), game_idx="347", position_ply="73")
    cap = dict(_cap(), game_idx=347, position_ply=73)
    assert M._identity(src) == M._identity(cap)
    M.authenticate_against([src], [cap])


def test_identity_requires_every_field_including_the_canonical_hash():
    """`.get()` leniency let a source and a capture that BOTH omit
    canonical_state_sha1 compare as None == None, so an incomplete capture
    authenticated against an incomplete source."""
    for field in M.CASE_IDENTITY:
        incomplete = {k: v for k, v in _src().items() if k != field}
        with pytest.raises(ValueError, match="case set"):
            M._identity(incomplete)
    with pytest.raises(ValueError, match="case set"):
        M.authenticate_against([{k: v for k, v in _src().items()
                                 if k != "canonical_state_sha1"}], [_cap()])


def test_duplicate_case_is_refused_even_when_the_identity_set_matches():
    """A 31-row capture repeating one valid case has the SAME identity set, and
    a by-case_id dict silently collapses the duplicate."""
    src, cap = _src(), _cap()
    with pytest.raises(ValueError, match="duplicate"):
        M.authenticate_against([src], [cap, dict(cap)])
    with pytest.raises(ValueError, match="duplicate"):
        M.authenticate_against([src, dict(src)], [cap])


def test_expected_case_count_is_enforced_when_requested():
    with pytest.raises(ValueError, match="expected exactly 30"):
        M.authenticate_against([_src()], [_cap()], expected_cases=30)
    assert M.A6400_EXPECTED_CASES == 30


def _synthetic_replay(tmp_path, moves=((5, 5), (7, 8), (9, 3), (11, 14), (13, 6))):
    """A small, legal replay written under tmp_path -- no gitignored data."""
    replay = {"board_size": 24, "n_moves": len(moves),
              "moves": [{"ply": i, "player": "red" if i % 2 == 0 else "black",
                         "row": r, "col": c} for i, (r, c) in enumerate(moves)]}
    path = tmp_path / "game_000000.json"
    path.write_text(json.dumps(replay))
    return path


def test_canonical_hash_derivation_is_portable_and_position_dependent(tmp_path):
    """PORTABLE guard on the real derivation.

    The live case-347 check below skips whenever the gitignored reservoir is
    absent, so on a clean checkout or in CI it cannot catch a
    `canonical_state_sha1_for` that hashes metadata. This one always runs.
    """
    from scripts.GPU.alphazero.fpu_state_hash import canonical_state_sha1
    from scripts.GPU.alphazero.position_probe_cases import position_state

    path = _synthetic_replay(tmp_path)
    row = {"case_id": "synthetic", "game_idx": 0, "position_ply": 4,
           "side_to_move": "red", "replay_path": str(path)}
    replay = json.loads(path.read_text())
    expected = canonical_state_sha1(position_state(replay, 4, "red"))
    assert M.canonical_state_sha1_for(row) == expected
    assert len(expected) == 40

    # Position-dependent: a different ply is a different board, so a digest of
    # metadata -- or of the replay file -- would collide here.
    earlier = dict(row, position_ply=3, side_to_move="black")
    assert M.canonical_state_sha1_for(earlier) != M.canonical_state_sha1_for(row)
    # ... and independent of where the file happens to live.
    moved = tmp_path / "renamed.json"
    moved.write_text(path.read_text())
    assert M.canonical_state_sha1_for(dict(row, replay_path=str(moved))) == expected


_A347_REPLAY = Path("logs/eval/calib020_0001_vs_0379_800g_w4_seed20115_replays/"
                    "game_000347.json")


@pytest.mark.skipif(not _A347_REPLAY.exists(), reason="replay reservoir absent")
def test_canonical_hash_is_derived_from_the_position_not_from_metadata():
    """Exercises the REAL derivation, not the fixture file.

    Without this a `canonical_state_sha1_for` that hashed metadata would pass
    every other test, because the fixture already carries correct values and the
    schema tests patch the function out.
    """
    row = {"case_id": "black_loss_game_000347_predrop_ply_73_drop_75",
           "game_idx": 347, "position_ply": 73, "side_to_move": "black",
           "replay_path": str(_A347_REPLAY)}
    assert M.canonical_state_sha1_for(row) == (
        "2555f653254f1b4d4c75bbd72d1f60e62adc7c38")
    # A different ply is a different position, so a metadata-only digest that
    # ignored the board would collide here.
    other = dict(row, position_ply=72, side_to_move="red")
    assert M.canonical_state_sha1_for(other) != M.canonical_state_sha1_for(row)


def test_fixture_canonical_hashes_are_real_position_hashes(source_rows):
    """The fixture must not label a metadata digest as a canonical state hash:
    that would mask the missing production field and test nothing."""
    prov = _fixture("provenance.json")
    assert prov["canonical_state_sha1_is_real"] is True
    by_game = {r["game_idx"]: r for r in source_rows}
    # Spot-checked against the real derivation for case 347.
    assert by_game[347]["canonical_state_sha1"] == (
        "2555f653254f1b4d4c75bbd72d1f60e62adc7c38")
    for row in source_rows:
        assert len(row["canonical_state_sha1"]) == 40
        int(row["canonical_state_sha1"], 16)
    assert len({r["canonical_state_sha1"] for r in source_rows}) == 30


def test_fixture_provenance_names_its_authenticated_source():
    """The fixture is DERIVED from the frozen historical artifact, which lives
    under gitignored logs/. Provenance is what lets a clean checkout know what
    it was derived from."""
    prov = _fixture("provenance.json")
    assert prov["generated_from_path"] == SIX_K_REF
    assert prov["generated_from_sha1"] == SIX_K_REF_SHA1
    # The canonical hashes come from replay bytes, so that reservoir is part of
    # the chain and its pinned aggregate is recorded too.
    assert prov["replay_reservoir_sha1"] == (
        "427d4ab669a81fe409de7da6d7c458056aff306e")
    assert prov["replay_reservoir_sha1"] == (
        M.A6400_REPLAY_RESERVOIR["replay_data_sha1"])
    assert prov["replay_reservoir"] == M.A6400_REPLAY_RESERVOIR["dir"]
    assert prov["replay_reservoir_n_games"] == 800
    assert prov["n_cases"] == 30
    assert prov["mcts_sims"] == 6400
    assert prov["base_seed"] == 20260616
    assert len(prov["case_identities"]) == 30
    assert len({c["case_id"] for c in prov["case_identities"]}) == 30


def test_capture_emits_the_frozen_v18_schema_with_a_no_search_fake(
        monkeypatch, source_rows, capture_doc):
    """Exercise the ACTUAL producer, not just `document_keys`. Revision 27
    advertised legacy-plus-seven while `capture()` built a different shape, so
    the advertised schema was never what a consumer received."""
    by_case = {r["case_id"]: r for r in source_rows}
    monkeypatch.setattr(M, "_load_frozen_a6400_source", lambda: source_rows)
    monkeypatch.setattr(M, "load_gate_cases", lambda gate: [
        {k: v for k, v in r.items() if k in M.CASE_IDENTITY} for r in source_rows])
    monkeypatch.setattr(M, "canonical_state_sha1_for",
                        lambda row: by_case[row["case_id"]]["canonical_state_sha1"])
    monkeypatch.setattr(M, "_default_evaluator_factory", lambda ckpt: object())
    monkeypatch.setattr(M, "sha1", lambda p: (
        M.A6400_SOURCE_SHA1 if p == M.A6400_SOURCE else M.A6400_CHECKPOINT_SHA1))
    monkeypatch.setattr(M, "authenticate_replay_reservoir",
                        lambda phase="opening": M.A6400_REPLAY_RESERVOIR["replay_data_sha1"])
    monkeypatch.setattr(M, "selected_move_for", lambda ev, row, cfg, seed, rule: {
        "selected_move": [1, 2], "selected_move_visits": 10, "tied_with_top": 0,
        "top_share_repr": by_case[row["case_id"]]["probe_top1_share"],
        "recomputed_black_value_repr":
            by_case[row["case_id"]]["probe_black_root_value"],
        "seed": seed ^ int(row["game_idx"]) ^ int(row["position_ply"])})

    doc = M.capture(mode="v18_preflight_a6400")
    assert set(doc) == set(M.document_keys("v18_preflight_a6400"))
    assert doc["run_kind"] == "v18_preflight_a6400"
    assert doc["scientific_interpretation_forbidden"] is True
    assert doc["mcts_sims"] == 6400
    assert doc["gate_list"] == ["A"]
    assert doc["auth_source_sha1"] == SIX_K_REF_SHA1
    assert doc["replay_reservoir_sha1"] == M.A6400_REPLAY_RESERVOIR["replay_data_sha1"]
    assert doc["checkpoint"] == M.CHECKPOINT
    assert doc["checkpoint_sha1"] == M.A6400_CHECKPOINT_SHA1
    assert doc["mcts"]["batching_triple"] == [14, 48, 8]
    assert doc["mcts"]["add_noise"] is False
    assert doc["mcts"]["base_seed"] == 20260616
    assert doc["mcts"]["seed_rule"] == "base ^ game_idx ^ position_ply"
    assert doc["source_case_count"] == 30
    assert len(doc["cases"]) == 30
    assert len(doc["authentication"]) == 30
    for case in doc["cases"]:
        assert len(case["canonical_state_sha1"]) == 40
    assert len({c["canonical_state_sha1"] for c in doc["cases"]}) == 30


def test_preflight_identity_check_precedes_evaluator_construction(
        monkeypatch, source_rows):
    """A 30 x 6,400-sim capture is expensive: every identity defect is
    detectable from metadata, so it must be caught before any evaluator."""
    built = []
    monkeypatch.setattr(M, "_load_frozen_a6400_source", lambda: source_rows)
    # Gate rows disagree with the source on one ply -> a case-set mismatch.
    monkeypatch.setattr(M, "load_gate_cases", lambda gate: [
        dict({k: v for k, v in r.items() if k in M.CASE_IDENTITY},
             position_ply=r["position_ply"] + (1 if i == 0 else 0))
        for i, r in enumerate(source_rows)])
    monkeypatch.setattr(M, "canonical_state_sha1_for",
                        lambda row: row["canonical_state_sha1"])
    monkeypatch.setattr(M, "_default_evaluator_factory",
                        lambda ckpt: built.append(ckpt))
    monkeypatch.setattr(M, "sha1", lambda p: (
        M.A6400_SOURCE_SHA1 if p == M.A6400_SOURCE else M.A6400_CHECKPOINT_SHA1))
    monkeypatch.setattr(M, "authenticate_replay_reservoir",
                        lambda phase="opening": M.A6400_REPLAY_RESERVOIR["replay_data_sha1"])

    with pytest.raises(ValueError, match="BEFORE search"):
        M.capture(mode="v18_preflight_a6400")
    assert built == [], "an evaluator was constructed despite a bad case set"


def _no_search_capture_env(monkeypatch, source_rows, built):
    """Wire capture() for a zero-search run and record evaluator construction."""
    by_case = {r["case_id"]: r for r in source_rows}
    monkeypatch.setattr(M, "_load_frozen_a6400_source", lambda: source_rows)
    monkeypatch.setattr(M, "load_gate_cases", lambda gate: [
        {k: v for k, v in r.items() if k in M.CASE_IDENTITY} for r in source_rows])
    monkeypatch.setattr(M, "canonical_state_sha1_for",
                        lambda row: by_case[row["case_id"]]["canonical_state_sha1"])
    monkeypatch.setattr(M, "_default_evaluator_factory",
                        lambda ckpt: built.append(ckpt))
    monkeypatch.setattr(M, "selected_move_for", lambda ev, row, cfg, seed, rule: {
        "selected_move": [1, 2], "selected_move_visits": 10, "tied_with_top": 0,
        "top_share_repr": by_case[row["case_id"]]["probe_top1_share"],
        "recomputed_black_value_repr":
            by_case[row["case_id"]]["probe_black_root_value"],
        "seed": seed ^ int(row["game_idx"]) ^ int(row["position_ply"])})
    return by_case


def test_checkpoint_pin_is_imported_from_the_authenticated_definition():
    from scripts.GPU.alphazero.v18_control_pool import SELECTED_UNIVERSE
    assert M.CHECKPOINT == SELECTED_UNIVERSE["anchor_checkpoint"]
    assert M.A6400_CHECKPOINT_SHA1 == SELECTED_UNIVERSE["checkpoint_sha1s"][M.CHECKPOINT]
    assert M.A6400_CHECKPOINT_SHA1 == "209cf2d4fd24a48553d259dd71b4954867b9473e"


def test_wrong_opening_checkpoint_identity_builds_no_evaluator(
        monkeypatch, source_rows):
    """The evaluator LOADS the checkpoint bytes, so they are bound first."""
    built = []
    _no_search_capture_env(monkeypatch, source_rows, built)
    monkeypatch.setattr(M, "sha1", lambda p: (
        M.A6400_SOURCE_SHA1 if p == M.A6400_SOURCE else "0" * 40))
    monkeypatch.setattr(M, "authenticate_replay_reservoir",
                        lambda phase="opening": M.A6400_REPLAY_RESERVOIR["replay_data_sha1"])
    with pytest.raises(ValueError, match="checkpoint"):
        M.capture(mode="v18_preflight_a6400")
    assert built == []


def test_wrong_opening_reservoir_identity_builds_no_evaluator(
        monkeypatch, source_rows):
    built = []
    _no_search_capture_env(monkeypatch, source_rows, built)
    monkeypatch.setattr(M, "sha1", lambda p: (
        M.A6400_SOURCE_SHA1 if p == M.A6400_SOURCE else M.A6400_CHECKPOINT_SHA1))

    def refuse(phase="opening"):
        raise ValueError("seed20115: replay_data_sha1 drifted")

    monkeypatch.setattr(M, "authenticate_replay_reservoir", refuse)
    with pytest.raises(ValueError, match="replay_data_sha1"):
        M.capture(mode="v18_preflight_a6400")
    assert built == []


def test_checkpoint_mutation_during_the_capture_yields_no_document(
        monkeypatch, source_rows):
    """`selected_move_for` runs for hours between the two checks."""
    built, calls = [], {"n": 0}
    _no_search_capture_env(monkeypatch, source_rows, built)
    monkeypatch.setattr(M, "authenticate_replay_reservoir",
                        lambda phase="opening": M.A6400_REPLAY_RESERVOIR["replay_data_sha1"])

    def drifting(path):
        if path == M.A6400_SOURCE:
            return M.A6400_SOURCE_SHA1
        calls["n"] += 1
        return M.A6400_CHECKPOINT_SHA1 if calls["n"] == 1 else "0" * 40

    monkeypatch.setattr(M, "sha1", drifting)
    with pytest.raises(ValueError, match="checkpoint"):
        M.capture(mode="v18_preflight_a6400")
    assert built, "the opening check should have passed"
    assert calls["n"] >= 2, "the closing checkpoint check never ran"


def test_reservoir_mutation_during_the_capture_yields_no_document(
        monkeypatch, source_rows):
    built, calls = [], {"n": 0}
    _no_search_capture_env(monkeypatch, source_rows, built)
    monkeypatch.setattr(M, "sha1", lambda p: (
        M.A6400_SOURCE_SHA1 if p == M.A6400_SOURCE else M.A6400_CHECKPOINT_SHA1))

    def drifting(phase="opening"):
        calls["n"] += 1
        if calls["n"] == 1:
            return M.A6400_REPLAY_RESERVOIR["replay_data_sha1"]
        return "0" * 40

    monkeypatch.setattr(M, "authenticate_replay_reservoir", drifting)
    with pytest.raises(ValueError, match="changed during the capture"):
        M.capture(mode="v18_preflight_a6400")
    assert built, "the opening check should have passed"
    assert calls["n"] >= 2, "the closing reservoir check never ran"


def test_authentication_phases_are_exactly_the_production_sequence(
        monkeypatch, source_rows):
    """The artifact records what was AUTHENTICATED, not a fresh unpinned read.

    This exercises the REAL `_load_frozen_a6400_source`, because replacing it
    wholesale hides the `pre_derivation` check that guards the bytes the
    canonical hashes are reconstructed from -- an earlier version of this test
    saw only two calls and wrongly claimed that was the whole structure.

    Counting is what makes reuse testable: on the passing path a third read
    returns the same bytes, so only the CALL STRUCTURE distinguishes reusing an
    authenticated value from taking an unauthenticated one at emission time.
    """
    built, phases, ckpt_reads = [], [], []
    by_case = {r["case_id"]: r for r in source_rows}
    # The frozen CSV shape: strings, and NO canonical hash column.
    raw_rows = [{k: v for k, v in r.items() if k != "canonical_state_sha1"}
                for r in source_rows]
    for row in raw_rows:
        row["game_idx"] = str(row["game_idx"])
        row["position_ply"] = str(row["position_ply"])

    monkeypatch.setattr(M, "_read_bytes", lambda p: b"csv-bytes")
    monkeypatch.setattr(M, "sha1_bytes", lambda b: M.A6400_SOURCE_SHA1)
    monkeypatch.setattr(M, "_parse_source_rows", lambda b: raw_rows)
    monkeypatch.setattr(M, "canonical_state_sha1_for",
                        lambda row: by_case[row["case_id"]]["canonical_state_sha1"])
    monkeypatch.setattr(M, "load_gate_cases", lambda gate: [
        {k: v for k, v in r.items() if k in M.CASE_IDENTITY} for r in source_rows])
    monkeypatch.setattr(M, "_default_evaluator_factory",
                        lambda ckpt: built.append(ckpt))
    monkeypatch.setattr(M, "selected_move_for", lambda ev, row, cfg, seed, rule: {
        "selected_move": [1, 2], "selected_move_visits": 10, "tied_with_top": 0,
        "top_share_repr": by_case[row["case_id"]]["probe_top1_share"],
        "recomputed_black_value_repr":
            by_case[row["case_id"]]["probe_black_root_value"],
        "seed": seed ^ int(row["game_idx"]) ^ int(row["position_ply"])})

    def counting_sha1(path):
        if path == M.A6400_SOURCE:
            return M.A6400_SOURCE_SHA1
        ckpt_reads.append(path)
        return M.A6400_CHECKPOINT_SHA1

    def counting_reservoir(phase="opening"):
        phases.append(phase)
        return M.A6400_REPLAY_RESERVOIR["replay_data_sha1"]

    monkeypatch.setattr(M, "sha1", counting_sha1)
    monkeypatch.setattr(M, "authenticate_replay_reservoir", counting_reservoir)

    doc = M.capture(mode="v18_preflight_a6400")
    assert doc["checkpoint_sha1"] == M.A6400_CHECKPOINT_SHA1
    assert doc["replay_reservoir_sha1"] == M.A6400_REPLAY_RESERVOIR["replay_data_sha1"]
    # The reservoir guards three distinct spans; the checkpoint only two.
    assert phases == ["pre_derivation", "opening", "closing"], phases
    assert len(ckpt_reads) == 2, (
        f"{len(ckpt_reads)} checkpoint reads; expected exactly the opening and "
        f"closing authentications, with the document reusing one of them")
    assert list(M.RESERVOIR_PHASES) == ["pre_derivation", "opening", "closing"]


def test_reservoir_phase_argument_is_validated():
    with pytest.raises(ValueError, match="unknown phase"):
        M.authenticate_replay_reservoir("whenever")


def test_fixture_identities_agree_across_both_files():
    ident = lambda r: (r["case_id"], r["game_idx"], r["position_ply"],
                       r["side_to_move"], r["replay_path"],
                       r["canonical_state_sha1"])
    src = [ident(r) for r in _fixture("source_rows.json")]
    cap = [ident(r) for r in _fixture("capture.json")["cases"]]
    assert src == cap
    assert len(set(src)) == 30
    recorded = {(c["case_id"], c["game_idx"], c["position_ply"],
                 c["side_to_move"]) for c in _fixture("provenance.json")["case_identities"]}
    assert recorded == {(i[0], i[1], i[2], i[3]) for i in src}
