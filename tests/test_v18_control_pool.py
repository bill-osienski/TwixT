"""Non-A discovery control pool -- spec Sec 2.2 and Sec 2.2.3.

The pool supplies EVERY numeric threshold, so its independence from A and from
established acceptance positions is the load-bearing property here.
"""
import json
from pathlib import Path

import pytest

from scripts.GPU.alphazero import v18_control_pool as P
from scripts.GPU.alphazero import v18_preflight_criteria as C


def test_forbidden_sources_cover_a_and_all_four_gates():
    names = {s["name"] for s in P.FORBIDDEN_SOURCES}
    for required in ("gate_A", "gate_B", "gate_C", "gate_D",
                     "v16a_neutral_consumed", "v16_production_selected",
                     "v17_development_selected", "a_replay_games"):
        assert required in names, required


def test_gate_c_source_is_recorded_as_degenerate_and_forbidden():
    """Revision 1 proposed this as the CONTROL source. It is 240 rows over 30
    distinct positions that are exactly gate C. Pin the finding so it cannot be
    silently reintroduced."""
    c = next(s for s in P.FORBIDDEN_SOURCES if s["name"] == "gate_C")
    assert c["distinct_positions"] == 30
    assert c["total_rows"] == 240
    assert c["rejected_as_control_source"] is True


def test_apply_exclusions_removes_canonical_hash_matches():
    rows = [{"canonical_sha1": "a" * 40, "game_content_sha1": "1" * 40},
            {"canonical_sha1": "b" * 40, "game_content_sha1": "2" * 40}]
    kept, report = P.apply_exclusions(rows, {"a" * 40}, set())
    assert [r["canonical_sha1"] for r in kept] == ["b" * 40]
    assert report["excluded_by_canonical_hash"] == 1


def test_apply_exclusions_removes_whole_games_by_replay_content_sha1():
    """Game identity is the replay's CONTENT hash, never (dir, game_idx).
    `game_idx` is reservoir-local, so index comparison both invents overlaps and
    misses a copied game that was renumbered -- the lesson already recorded in
    diagnose_fpu_baseline_policy_mass.game_identities:1694."""
    rows = [{"canonical_sha1": "c" * 40, "game_content_sha1": "7" * 40},
            {"canonical_sha1": "e" * 40, "game_content_sha1": "8" * 40}]
    kept, report = P.apply_exclusions(rows, set(), {"7" * 40})
    assert [r["game_content_sha1"] for r in kept] == ["8" * 40]
    assert report["excluded_by_game"] == 1


def test_game_identity_helper_is_imported_not_reimplemented():
    from scripts.GPU.alphazero import diagnose_fpu_baseline_policy_mass as D
    assert P.game_identities is D.game_identities


def test_renumbered_copy_of_a_forbidden_game_is_still_excluded():
    rows = [{"canonical_sha1": "a" * 40, "game_content_sha1": "7" * 40,
             "game_idx": 999, "replay_dir": "some/other/dir"}]
    kept, _report = P.apply_exclusions(rows, set(), {"7" * 40})
    assert kept == []


def test_exclusion_report_is_non_vacuous():
    """A pool whose exclusions remove nothing has not been verified. The freeze
    must record counts so a zero is visible rather than implied."""
    rows = [{"canonical_sha1": "f" * 40, "game_content_sha1": "1" * 40}]
    _kept, report = P.apply_exclusions(rows, set(), set())
    assert report["excluded_by_canonical_hash"] == 0
    assert report["excluded_by_game"] == 0
    assert report["input_rows"] == 1


def test_freeze_refuses_a_universe_that_collapses_to_empty(tmp_path):
    with pytest.raises(ValueError, match="collapsed"):
        P.freeze_source_universe(out_path=str(tmp_path / "x.json"),
                                 universe_name="__empty__", seed=20260729)


def test_freeze_refuses_when_distinct_games_are_fewer_than_the_minimum(tmp_path):
    """The failure mode that killed the revision-1 control source: plenty of
    ROWS, far too few distinct games. Because the cohort takes at most one
    position per game, distinct GAMES is the binding supply."""
    with pytest.raises(ValueError, match="distinct"):
        P.freeze_source_universe(out_path=str(tmp_path / "x.json"),
                                 universe_name="__degenerate__", seed=20260729)


def test_frozen_universe_record_is_byte_reproducible(tmp_path):
    a = P.freeze_source_universe(str(tmp_path / "a.json"), "__fixture__", 20260729)
    b = P.freeze_source_universe(str(tmp_path / "b.json"), "__fixture__", 20260729)
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert a["universe_sha1"] == b["universe_sha1"]


def test_frozen_universe_record_stamps_the_scope_labels(tmp_path):
    rec = P.freeze_source_universe(str(tmp_path / "a.json"), "__fixture__", 20260729)
    assert rec["run_kind"] == "shipped_only_preflight_source_universe"
    assert rec["scientific_interpretation_forbidden"] is True
    assert rec["selection_is_independent_of_residual_exposure"] is True


def test_universe_is_exactly_800_games():
    assert C.UNIVERSE["n_games"] == 800
    assert P.UNIVERSE["n_games"] == 800
    for spec in P.CANDIDATE_UNIVERSES:
        if spec["name"].startswith("__"):
            continue
        assert spec["min_distinct_games"] == 800, spec["name"]


def test_fewer_than_800_eligible_games_refuses(tmp_path):
    # 799 survivors is a STOP, never a 799-game universe: the sizing ladder and
    # the census ceiling are both derived from exactly 800.
    with pytest.raises(ValueError, match="distinct"):
        P.freeze_source_universe(str(tmp_path / "a.json"), "__short__", 20260729)


def test_zero_yield_games_are_retained_in_all_game_ids(tmp_path):
    """sizing_analysis_core's all_game_ids is the COMPLETE universe including
    games that yielded ZERO kept rows -- excluding them biases success upward."""
    rec = P.freeze_source_universe(str(tmp_path / "a.json"), "__fixture__", 20260729)
    yielding = {r["game_content_sha1"] for r in rec["census_positions"]}
    assert set(rec["all_game_ids"]) >= yielding
    assert len(rec["all_game_ids"]) == rec["n_games"]
    # The fixture deliberately contains a game that yields no census row at all.
    assert rec["zero_yield_games"] >= 1
    assert len(rec["all_game_ids"]) > len(yielding)


def test_position_exclusions_never_drop_a_game_from_all_game_ids(tmp_path):
    """Order of operations: game exclusions at step 2, position exclusions at
    step 5. Applying canonical hashes early would drop a whole game for holding
    one forbidden position, silently defeating zero-yield retention."""
    rec = P.freeze_source_universe(str(tmp_path / "a.json"), "__fixture__",
                                   20260729)
    excluded = P.freeze_source_universe(
        str(tmp_path / "b.json"), "__fixture__", 20260729,
        extra_forbidden_hashes={rec["census_positions"][0]["canonical_sha1"]})
    assert excluded["n_games"] == rec["n_games"]
    assert set(excluded["all_game_ids"]) == set(rec["all_game_ids"])
    assert len(excluded["census_positions"]) == len(rec["census_positions"]) - 1
    assert excluded["exclusion_report"]["excluded_by_canonical_hash"] == 1


def test_census_phase_allocation_is_1_1_1_3():
    assert C.CENSUS["phase_strata"] == {
        "opening": 1, "early_mid": 1, "midgame": 1, "late": 3}
    assert sum(C.CENSUS["phase_strata"].values()) == C.CENSUS["positions_per_game"] == 6
    assert P.CENSUS is C.CENSUS


def test_missing_phase_contributes_zero_and_is_reported():
    # A 40-ply game has no midgame (61-90) and no late (>=91) position at all.
    game = P.synthetic_game(n_moves=40, game_idx=0)
    rows, per_phase = P.census_for_game(game, "sha", "path")
    assert per_phase["midgame"] == 0
    assert per_phase["late"] == 0
    assert per_phase["opening"] == 1
    assert {r["phase"] for r in rows} == {"opening", "early_mid"}


def test_missing_phase_is_never_backfilled_from_another_phase():
    game = P.synthetic_game(n_moves=40, game_idx=0)
    rows, _ = P.census_for_game(game, "sha", "path")
    # 6 slots exist but only 2 phases are populated: the empty strata stay empty
    # rather than being topped up from the phases that do have positions.
    assert len(rows) == 2
    assert sum(1 for r in rows if r["phase"] == "opening") == 1
    assert sum(1 for r in rows if r["phase"] == "early_mid") == 1


def test_single_slot_phases_use_q_one_half_ceil_index():
    # Opening plies 0..30 -> 31 qualifying; ceil(0.5*31) = 16 -> the 16th
    # smallest, 1-indexed, which is ply 15.
    game = P.synthetic_game(n_moves=200, game_idx=0)
    rows, _ = P.census_for_game(game, "sha", "path")
    opening = [r for r in rows if r["phase"] == "opening"]
    assert len(opening) == 1
    assert opening[0]["position_ply"] == 15


def test_late_slots_use_q_quarter_half_three_quarter():
    game = P.synthetic_game(n_moves=200, game_idx=0)
    rows, per_phase = P.census_for_game(game, "sha", "path")
    assert per_phase["late"] == 3
    late = sorted(r["position_ply"] for r in rows if r["phase"] == "late")
    # late plies are 91..199 -> 109 qualifying; ceil(q*109) at 1/4, 2/4, 3/4
    # gives ranks 28, 55, 82 -> plies 118, 145, 172.
    assert late == [118, 145, 172]


def test_late_slots_collapse_without_replacement_when_supply_is_short():
    # One late ply cannot fill three slots, and must not be emitted three times.
    game = P.synthetic_game(n_moves=92, game_idx=0)
    rows, per_phase = P.census_for_game(game, "sha", "path")
    assert per_phase["late"] == 1
    assert len([r for r in rows if r["phase"] == "late"]) == 1


def test_census_ceiling_of_4800_aborts_before_evaluator_load(tmp_path):
    assert C.CENSUS["max_total_searches"] == 4800 == 800 * 6
    with pytest.raises(ValueError, match="4800|ceiling"):
        P.assert_census_within_ceiling(4801)
    assert P.assert_census_within_ceiling(4800) is None


def test_criteria_constants_are_imported_not_restated():
    # Object identity, not equality: a copy would silently drift.
    assert P.UNIVERSE is C.UNIVERSE
    assert P.CENSUS is C.CENSUS
    assert P.PHASE_WINDOWS is C.PHASE_WINDOWS


def test_universe_binds_summary_jsonl_sidecars_and_checkpoints(tmp_path):
    """Binding the directory is insufficient -- bind the artifacts.

    Revision 22: `black_checkpoint_sha1` / `red_checkpoint_sha1` are GONE. The
    source alternates colours 400/400, so a fixed black/red identity is a false
    claim; the record carries the pair, the anchor and the full colour schedule.
    """
    rec = P.freeze_source_universe(str(tmp_path / "a.json"), "__fixture__", 20260729)
    for key in ("summary_sha1", "jsonl_sha1", "replay_data_sha1",
                "checkpoint_sha1s", "anchor_checkpoint", "games_by_colour"):
        assert key in rec, key
    assert "black_checkpoint_sha1" not in rec
    assert "red_checkpoint_sha1" not in rec


def test_freeze_emits_a_census_but_selects_no_cohort(tmp_path):
    rec = P.freeze_source_universe(str(tmp_path / "a.json"), "__fixture__", 20260729)
    assert "census_positions" in rec
    assert "matched_cohort" not in rec        # Task 4b's job, after measurement


def test_selection_never_reads_residual_exposure(tmp_path):
    """The universe must be frozen WITHOUT looking at the statistic it will later
    calibrate, or the threshold is fitted to its own sample. Nothing here could
    read residuals anyway -- they do not exist pre-search -- so this test also
    documents that invariant."""
    rec = P.freeze_source_universe(str(tmp_path / "a.json"), "__fixture__", 20260729)
    assert "exposure" not in json.dumps(rec["selection_inputs"]).lower()
    assert "residual" not in json.dumps(rec["selection_inputs"]).lower()
    # root_value is checkpoint-contaminated and must not be a selection input.
    # Structural, not a substring scan: the field list is the behaviour, and
    # `selection_inputs` legitimately NAMES root_value when explaining why it is
    # excluded.
    assert "root_value" not in rec["selection_inputs"]["fields_used"]
    assert rec["selection_inputs"]["value_based_rule"].startswith("NONE")
    assert rec["selection_inputs"]["near_even_rule"].startswith("NONE")
    assert rec["selection_inputs"]["clip_statistic_rule"].startswith("NONE")


def test_replay_data_hash_uses_the_established_helper():
    """A local re-derivation is not an identity. The established helper is
    length-DELIMITED per file, so a byte repartition across two replays cannot
    collide; folding textual digests together loses that property. On the same
    800 files the two disagree: 4ae58edf... (local) vs 13e6b3d6... (established).
    """
    from scripts.GPU.alphazero import fpu_provenance
    assert not hasattr(P, "_replay_data_sha1"), (
        "the invented hash must be deleted, not merely unused")
    assert P.SELECTED_UNIVERSE["replay_data_sha1"] == (
        "13e6b3d6414be580bef2b9ff1b02d2f3a29ba445")
    # The pinned value IS what the established helper produces.
    assert fpu_provenance.replay_data_sha1.__module__.endswith("fpu_provenance")


def test_selected_universe_identity_is_bound_in_tracked_code():
    s = P.SELECTED_UNIVERSE
    assert s["name"] == "seed20116"
    assert s["n_games"] == 800
    assert s["summary_sha1"] == "18a015fa804fc0d3866feb42e2d637ce11e87930"
    assert s["jsonl_sha1"] == "789ab890f606aebe87dead98b2207d2dc4760c65"
    assert s["replay_data_sha1"] == "13e6b3d6414be580bef2b9ff1b02d2f3a29ba445"
    for key in ("summary_sha1", "jsonl_sha1", "replay_data_sha1"):
        assert len(s[key]) == 40, key
    # The pairing is colour-balanced, so the identity is the unordered PAIR plus
    # the split -- not a fixed black/red assignment.
    assert set(s["checkpoint_pair"]) == set(s["checkpoint_sha1s"])
    assert s["games_per_colour"] == 400
    assert s["games_per_colour"] * 2 == s["n_games"]
    # There is no fixed colour assignment in this source, so a key claiming one
    # would be a false claim.
    assert "black_checkpoint" not in s
    assert "red_checkpoint" not in s
    assert s["anchor_checkpoint"] in s["checkpoint_pair"]
    assert "calib020-from0409/model_iter_0001" in s["anchor_checkpoint"]
    assert s["checkpoint_sha1s"][s["anchor_checkpoint"]] == (
        "209cf2d4fd24a48553d259dd71b4954867b9473e")
    assert sorted(s["checkpoint_sha1s"].values()) == sorted(
        ("209cf2d4fd24a48553d259dd71b4954867b9473e",
         "8ad62ac432c35c6ea9b0630b8a2b8c572a0b03a1"))
    # The zero-margin condition is recorded with the identity it qualifies, so a
    # future reader cannot treat 800 as a comfortable number.
    assert "EXACTLY 800" in s["exact_800_has_no_margin"]


def test_freeze_refuses_any_real_universe_other_than_the_selected_one(tmp_path):
    for other in ("v17_development", "v16_production"):
        with pytest.raises(ValueError, match="selected source"):
            P.freeze_source_universe(str(tmp_path / "x.json"), other, 20260729)


def test_forbidden_sources_are_byte_pinned_not_only_count_pinned():
    """A replacement file with an unchanged row count passes a count check.
    Every forbidden evidence file therefore carries a byte identity."""
    for source in P.FORBIDDEN_SOURCES:
        if source["kind"] == "replay_dir":
            for key in ("summary_sha1", "jsonl_sha1", "replay_data_sha1"):
                assert len(source[key]) == 40, (source["name"], key)
        else:
            assert len(P.FORBIDDEN_SOURCE_SHA1S[source["name"]]) == 40, source["name"]
    assert P.FORBIDDEN_SOURCE_SHA1S["v17_development_selected"] == (
        "15b0228edc1ed605fea799694d4ca0eda3e3468b")
    a = next(s for s in P.FORBIDDEN_SOURCES if s["name"] == "a_replay_games")
    assert a["summary_sha1"] == "bf1e3701ca8591295bd1e70b2a88a84087fad316"
    assert a["jsonl_sha1"] == "fb0944ae0333b951a817d0393919b45f2a12fd78"
    assert a["replay_data_sha1"] == "427d4ab669a81fe409de7da6d7c458056aff306e"


def test_authentication_refuses_a_moved_artifact(monkeypatch):
    """Non-vacuity: the byte check must reject a CHANGED FILE, not a patched
    hash function. Mutating the bytes is the real attack -- a replacement CSV
    with the same row count."""
    real_read = Path.read_bytes

    def tampered(self, *a, **k):
        raw = real_read(self, *a, **k)
        if str(self).endswith("calib020_post_opening_sweep/position_probe_cases.csv"):
            return raw + b"\n"          # one byte, same row count
        return raw

    monkeypatch.setattr(Path, "read_bytes", tampered)
    with pytest.raises(ValueError, match="pinned"):
        P.authenticate_forbidden_sources()


def test_frozen_record_reconciles_with_its_selected_rows(tmp_path):
    rec = P.freeze_source_universe(str(tmp_path / "a.json"), "__fixture__", 20260729)
    g = rec["census_geometry"]
    # phase counts describe the rows actually carried
    assert sum(g["per_phase_after"].values()) == len(rec["census_positions"])
    assert sum(rec["per_phase"].values()) == len(rec["census_positions"])
    # position exclusions reconcile
    assert (g["positions_before_position_exclusions"]
            - g["positions_excluded_by_canonical_hash"]
            == g["positions_after_position_exclusions"] == len(rec["census_positions"]))
    # game exclusions reconcile
    assert (g["games_seen"] - g["games_excluded_by_content_sha1"]
            == g["games_surviving_game_exclusion"])
    assert g["games_selected"] == len(rec["all_game_ids"]) == rec["n_games"]
    # zero-yield games are still recorded
    yielding = {r["game_content_sha1"] for r in rec["census_positions"]}
    assert yielding <= set(rec["all_game_ids"])
    assert rec["zero_yield_games"] == len(rec["all_game_ids"]) - len(yielding)


def test_reconciliation_check_rejects_a_mismatched_record():
    """The reconciliation must be capable of failing."""
    bad = {
        "census_geometry": {
            "per_phase_after": {"opening": 5, "early_mid": 0, "midgame": 0, "late": 0},
            "positions_before_position_exclusions": 5,
            "positions_after_position_exclusions": 5,
            "positions_excluded_by_canonical_hash": 0,
            "games_seen": 2, "games_excluded_by_content_sha1": 0,
            "games_surviving_game_exclusion": 2, "games_selected": 2,
        },
        "census_positions": [], "all_game_ids": ["a", "b"], "n_games": 2,
        "zero_yield_games": 2,
    }
    with pytest.raises(ValueError, match="per_phase_after sums"):
        P._assert_record_reconciles(bad)


def test_geometry_before_and_after_are_both_recorded(tmp_path):
    rec = P.freeze_source_universe(
        str(tmp_path / "a.json"), "__fixture__", 20260729,
        extra_forbidden_hashes={
            P.freeze_source_universe(str(tmp_path / "seed.json"), "__fixture__",
                                     20260729)["census_positions"][0]["canonical_sha1"]})
    g = rec["census_geometry"]
    assert g["positions_excluded_by_canonical_hash"] == 1
    assert (g["positions_before_position_exclusions"]
            > g["positions_after_position_exclusions"])
    assert sum(g["per_phase_before"].values()) == g["positions_before_position_exclusions"]
    assert sum(g["per_phase_after"].values()) == g["positions_after_position_exclusions"]


# --- non-vacuity: the authentication chain must reject real tampering --------

REAL = "seed20116"


def _real_spec():
    return P._universe_spec(REAL)


def _patch_bytes(monkeypatch, target: str, mutate):
    """Make one path return mutated bytes, leaving every other read intact."""
    real_read = Path.read_bytes

    def fake(self, *a, **k):
        raw = real_read(self, *a, **k)
        return mutate(raw) if str(self).endswith(target) else raw

    monkeypatch.setattr(Path, "read_bytes", fake)


@pytest.mark.slow
def test_jsonl_and_sidecar_disagreement_is_refused(monkeypatch):
    """Reach the CROSS-FIELD validator, not just the hash gate.

    Mutating the JSONL alone stops at the pinned-hash mismatch, so the test
    would stay green even if the JSONL/sidecar comparison were deleted. Repin
    the expected hash to the mutated bytes: authentication then passes and the
    winner disagreement must be what refuses.
    """
    import hashlib
    spec = _real_spec()
    original = Path(spec["jsonl"]).read_bytes()
    mutated = original.replace(b'"winner": "black"', b'"winner": "red"', 1)
    assert mutated != original

    _patch_bytes(monkeypatch, "_replay_games.jsonl", lambda _r: mutated)
    monkeypatch.setitem(P.SELECTED_UNIVERSE, "jsonl_sha1",
                        hashlib.sha1(mutated).hexdigest())

    with pytest.raises(ValueError, match="contradicts the replay"):
        P.authenticate_selected_universe(spec)


@pytest.mark.slow
def test_jsonl_hash_gate_still_refuses_an_unpinned_mutation(monkeypatch):
    """The hash gate remains the first line of defence when NOT repinned."""
    spec = _real_spec()
    original = Path(spec["jsonl"]).read_bytes()
    _patch_bytes(monkeypatch, "_replay_games.jsonl",
                 lambda _r: original.replace(b'"winner": "black"', b'"winner": "red"', 1))
    with pytest.raises(ValueError, match="hashes|pinned"):
        P.authenticate_selected_universe(spec)


def test_forbidden_replay_reservoirs_are_bound():
    """Authenticating the probe CSVs is not enough: the canonical exclusions are
    reconstructed FROM these sidecars, so the reservoirs themselves are pinned."""
    by_name = {r["name"]: r for r in P.FORBIDDEN_REPLAY_RESERVOIRS}
    assert set(by_name) == {"seed20115", "seed35791", "seed40937"}
    assert by_name["seed20115"]["replay_data_sha1"] == (
        "427d4ab669a81fe409de7da6d7c458056aff306e")
    assert by_name["seed35791"]["replay_data_sha1"] == (
        "d36b01c0993095e07785666316028f0c875eed7b")
    assert by_name["seed40937"]["replay_data_sha1"] == (
        "80aa2068319cdbe0429100b736d293f5b8bc437e")
    for name, r in by_name.items():
        assert r["n_games"] == 800, name
        assert len(r["replay_data_sha1"]) == 40, name
        assert r["referenced_by"], name
    # The A source's own pinned aggregate must agree with its reservoir entry.
    a = next(s for s in P.FORBIDDEN_SOURCES if s["name"] == "a_replay_games")
    assert a["replay_data_sha1"] == by_name["seed20115"]["replay_data_sha1"]


def test_every_referenced_replay_lies_in_a_pinned_reservoir():
    _verified, payloads = P.authenticate_forbidden_sources()
    checked = 0
    for source in P.FORBIDDEN_SOURCES:
        kind = source["kind"]
        if kind in ("replay_dir", "corpus_manifest_csv"):
            continue
        if kind == "goal_line_manifest_json":
            rows = [{"replay_path": c["replay_path"]}
                    for c in json.loads(payloads[source["name"]].decode())["cases"]]
        else:
            rows = P._parse_csv(payloads[source["name"]])
        for row in rows:
            P._reservoir_for(row["replay_path"])       # raises if unpinned
            checked += 1
    assert checked > 0
    with pytest.raises(ValueError, match="outside every pinned"):
        P._reservoir_for("logs/eval/some_other_dir/game_000000.json")


@pytest.mark.slow
def test_gate_b_replay_drift_leaves_no_artifact(tmp_path, monkeypatch):
    """Gate B's positions come from seed35791. Drift there changes the exclusion
    SET while every evidence-file hash stays unchanged -- so the reservoir must
    be re-authenticated before anything is written."""
    _drift_reservoir_and_assert_refusal(tmp_path, monkeypatch, "seed35791")


@pytest.mark.slow
def test_gate_c_replay_drift_leaves_no_artifact(tmp_path, monkeypatch):
    """Gate C's positions come from seed40937."""
    _drift_reservoir_and_assert_refusal(tmp_path, monkeypatch, "seed40937")


@pytest.mark.slow
def test_a_game_identity_source_drift_leaves_no_artifact(tmp_path, monkeypatch):
    """seed20115 supplies BOTH the A game identities and the gate A/D/v16a
    positions, so drift there corrupts the game exclusion set as well."""
    _drift_reservoir_and_assert_refusal(tmp_path, monkeypatch, "seed20115")


def _drift_reservoir_and_assert_refusal(tmp_path, monkeypatch, reservoir_name):
    from scripts.GPU.alphazero import fpu_provenance
    target = next(r for r in P.FORBIDDEN_REPLAY_RESERVOIRS
                  if r["name"] == reservoir_name)
    real = fpu_provenance.replay_data_sha1
    state = {"opened": False}

    def drifting(paths):
        paths = list(paths)
        in_target = bool(paths) and str(Path(paths[0]).parent) == target["dir"]
        if in_target and state["opened"]:
            return "0" * 40                    # drift, seen by the closing check
        if in_target:
            state["opened"] = True             # opening check passes truthfully
        return real(paths)

    monkeypatch.setattr(fpu_provenance, "replay_data_sha1", drifting)
    out = tmp_path / "must_not_exist.json"
    with pytest.raises(ValueError, match="changed during the freeze"):
        P.freeze_source_universe(str(out), REAL, 20260729)
    assert state["opened"], "the opening reservoir check never ran"
    assert not out.exists(), "a refused freeze must leave no partial artifact"


@pytest.mark.slow
def test_report_refuses_any_real_universe_other_than_the_selected_one():
    """Step 5 chose seed20116. Leaving a route to enumerate candidates 2 and 3
    afterwards is a post-selection inspection path."""
    for other in ("v17_development", "v16_production"):
        with pytest.raises(ValueError, match="selected source"):
            P.report_universe(other)


@pytest.mark.slow
def test_report_authenticates_and_uses_the_snapshot(monkeypatch):
    """The report must run the same authenticated snapshot path as the freeze:
    read-only is not the same as unauthenticated."""
    seen = {"forbidden": 0, "reservoirs": [], "snapshot": None}
    real_sources = P.authenticate_forbidden_sources
    real_reservoirs = P.authenticate_forbidden_reservoirs
    real_enumerate = P.enumerate_census

    def spy_sources(*a, **k):
        seen["forbidden"] += 1
        return real_sources(*a, **k)

    def spy_reservoirs(when="opening"):
        seen["reservoirs"].append(when)
        return real_reservoirs(when)

    def spy_enumerate(spec, **kwargs):
        seen["snapshot"] = kwargs.get("snapshot")
        return real_enumerate(spec, **kwargs)

    monkeypatch.setattr(P, "authenticate_forbidden_sources", spy_sources)
    monkeypatch.setattr(P, "authenticate_forbidden_reservoirs", spy_reservoirs)
    monkeypatch.setattr(P, "enumerate_census", spy_enumerate)

    out = P.report_universe(REAL)
    assert out["authenticated"] is True
    assert seen["forbidden"] == 1
    # opening AND closing, the closing one via reverify_all_replay_sources
    assert seen["reservoirs"] == ["opening", "closing"]
    # the authenticated snapshot actually reached enumeration
    assert seen["snapshot"] is not None
    assert len(seen["snapshot"]["replays"]) == 800
    assert seen["snapshot"]["jsonl_rows_matched_1to1"] == 800


@pytest.mark.slow
def test_report_refuses_on_selected_universe_drift(monkeypatch):
    from scripts.GPU.alphazero import fpu_provenance
    real = fpu_provenance.replay_data_sha1
    state = {"opened": False}

    def drifting(paths):
        paths = list(paths)
        target = str(Path(paths[0]).parent) == P._universe_spec(REAL)["replay_dir"]
        if target and state["opened"]:
            return "0" * 40
        if target:
            state["opened"] = True
        return real(paths)

    monkeypatch.setattr(fpu_provenance, "replay_data_sha1", drifting)
    with pytest.raises(ValueError, match="changed during the freeze"):
        P.report_universe(REAL)


@pytest.mark.slow
def test_report_refuses_on_forbidden_reservoir_drift(monkeypatch):
    from scripts.GPU.alphazero import fpu_provenance
    target = next(r for r in P.FORBIDDEN_REPLAY_RESERVOIRS
                  if r["name"] == "seed40937")
    real = fpu_provenance.replay_data_sha1
    state = {"opened": False}

    def drifting(paths):
        paths = list(paths)
        in_target = bool(paths) and str(Path(paths[0]).parent) == target["dir"]
        if in_target and state["opened"]:
            return "0" * 40
        if in_target:
            state["opened"] = True
        return real(paths)

    monkeypatch.setattr(fpu_provenance, "replay_data_sha1", drifting)
    with pytest.raises(ValueError, match="changed during the freeze"):
        P.report_universe(REAL)
    assert state["opened"]


def test_closing_check_covers_every_replay_source():
    import inspect
    src = inspect.getsource(P.reverify_all_replay_sources)
    assert "_reverify_replay_data" in src
    assert "authenticate_forbidden_reservoirs" in src


@pytest.mark.slow
def test_checkpoint_pair_mutation_is_refused(monkeypatch):
    spec = _real_spec()
    monkeypatch.setitem(P.SELECTED_UNIVERSE, "checkpoint_pair",
                        ("checkpoints/does-not-exist/a.safetensors",
                         "checkpoints/does-not-exist/b.safetensors"))
    with pytest.raises(ValueError, match="checkpoint pair"):
        P.authenticate_selected_universe(spec)


@pytest.mark.slow
def test_colour_schedule_imbalance_is_refused(monkeypatch):
    spec = _real_spec()
    # A 400/400 source checked against a 401/399 expectation must refuse: the
    # schedule is part of the identity, not a cosmetic statistic.
    monkeypatch.setitem(P.SELECTED_UNIVERSE, "games_per_colour", 399)
    with pytest.raises(ValueError, match="colour schedule"):
        P.authenticate_selected_universe(spec)


@pytest.mark.slow
def test_mutation_between_authentication_and_census_is_detected(monkeypatch, tmp_path):
    """The load-bearing one, driven END TO END on the real universe.

    The opening authentication passes truthfully, the census consumes the
    snapshot, and the sidecars then drift before the record is written. The
    closing re-authentication must catch it -- otherwise the artifact would
    carry a hash for bytes it never measured -- and nothing may be written.
    """
    from scripts.GPU.alphazero import fpu_provenance
    calls = {"n": 0}
    real = fpu_provenance.replay_data_sha1

    def drifting(paths):
        calls["n"] += 1
        # Calls 1..n authenticate truthfully (forbidden A dir, then the selected
        # dir); the FINAL closing check sees drift.
        return "0" * 40 if calls["n"] >= 3 else real(paths)

    monkeypatch.setattr(fpu_provenance, "replay_data_sha1", drifting)
    out = tmp_path / "must_not_exist.json"
    with pytest.raises(ValueError, match="changed during the freeze"):
        P.freeze_source_universe(str(out), REAL, 20260729)
    assert calls["n"] >= 3, "the closing re-authentication never ran"
    assert not out.exists(), "a refused freeze must leave no partial artifact"


def test_closing_reverification_compares_against_the_pin(monkeypatch):
    from scripts.GPU.alphazero import fpu_provenance
    monkeypatch.setattr(fpu_provenance, "replay_data_sha1", lambda paths: "0" * 40)
    with pytest.raises(ValueError, match="changed during the freeze"):
        P._reverify_replay_data(_real_spec())


def test_no_partial_artifact_when_the_freeze_refuses(tmp_path, monkeypatch):
    """A refusal must leave nothing behind -- a half-written record would be
    indistinguishable from a frozen one."""
    out = tmp_path / "refused.json"
    monkeypatch.setattr(P, "_assert_record_reconciles",
                        lambda rec: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(ValueError, match="boom"):
        P.freeze_source_universe(str(out), "__fixture__", 20260729)
    assert not out.exists()


def test_exclusions_refuse_unauthenticated_payloads():
    """Exclusions may only be derived from authenticated bytes, so an empty
    payload map is a refusal rather than an empty exclusion set."""
    with pytest.raises(ValueError, match="authenticated payload"):
        P.forbidden_canonical_hashes({})


def test_authenticated_bytes_are_the_bytes_parsed(monkeypatch):
    """`_snapshot` must hash the buffer it returns, not re-read the path."""
    reads = {"n": 0}
    real_read = Path.read_bytes

    def counting(self, *a, **k):
        reads["n"] += 1
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_bytes", counting)
    raw, sha = P._snapshot("gate_C", P.FORBIDDEN_SOURCES[2]["path"],
                           P.FORBIDDEN_SOURCE_SHA1S["gate_C"])
    assert reads["n"] == 1, "authentication must not reopen the path"
    import hashlib as _h
    assert _h.sha1(raw).hexdigest() == sha
    assert P._parse_csv(raw), "the authenticated buffer is what gets parsed"


def test_report_writes_nothing(tmp_path, monkeypatch):
    """Step 5's report path is read-only: it must not emit a universe artifact
    or any other file."""
    import builtins
    real_open = builtins.open
    writes = []

    def watched(path, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            writes.append((str(path), mode))
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", watched)
    out = P.report_universe("__fixture__")
    assert writes == [], writes
    assert out["n_games"] >= 1
    assert "universe_sha1" not in out       # a report binds nothing
