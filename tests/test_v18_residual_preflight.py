"""v18 shipped-only residual preflight CLI -- plan Task 7.

Measurement only: no verdict, no derived threshold. Uses the CPU fake evaluator
from `tests/fpu_search_fixture.py`, so no GPU is needed and the search that runs
is genuinely shipped search.
"""
import json
import random
from pathlib import Path

import pytest

from scripts.GPU.alphazero import diagnose_v18_residual_preflight as M
from scripts.GPU.alphazero import v18_control_pool as P
from scripts.GPU.alphazero import v18_preflight_criteria as C
from scripts.GPU.alphazero.fpu_dev_reservoir_protocol import canonical_json_bytes
from tests.fpu_search_fixture import FakeEvaluator


# --- fixtures ---------------------------------------------------------------

def _replay(tmp_path, name, seed=7, size=8, n_moves=30):
    """A real replay, built by PLAYING legal moves.

    Two properties matter and both come from the move list. `position_state`
    sets `max_plies_limit = replay["n_moves"]`, so a short replay makes every
    depth-1 node terminal and the tree never reaches depth 2 -- there would be
    nothing for the walker or the crossover to measure. And a small board keeps
    400 shipped sims under a tenth of a second while still producing several
    hundred eligible depth-2 leaves.
    """
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    rng = random.Random(seed)
    state = TwixtState(active_size=size, to_move="red", max_plies_limit=None)
    moves = []
    while len(moves) < n_moves and not state.is_terminal():
        legal = list(state.legal_moves())
        move = legal[rng.randrange(len(legal))]
        moves.append(move)
        state = state.apply_move(move)
    doc = {"board_size": size, "n_moves": len(moves),
           "moves": [{"ply": i, "player": "red" if i % 2 == 0 else "black",
                      "row": int(r), "col": int(c)}
                     for i, (r, c) in enumerate(moves)]}
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return path


def _case(tmp_path, idx, population="census", ply=2):
    path = _replay(tmp_path, f"game_{idx:06d}.json", seed=idx)
    return {
        "population": population,
        "case_id": f"{population}_{idx}",
        "source_universe_ordinal": idx,
        "game_content_sha1": f"{idx:040x}",
        "game_idx": idx,
        "position_ply": ply,
        "side_to_move": "red" if ply % 2 == 0 else "black",
        "canonical_state_sha1": f"{idx:040d}",
        "phase": "opening",
        "replay_path": str(path),
    }


@pytest.fixture
def cases(tmp_path):
    return [_case(tmp_path, 1, "selected_a"), _case(tmp_path, 2, "census"),
            _case(tmp_path, 3, "census", ply=4)]


@pytest.fixture
def criteria_path(tmp_path):
    path = tmp_path / "criteria.json"
    C.emit_frozen_criteria(str(path))
    return path


@pytest.fixture
def universe_path(tmp_path):
    path = tmp_path / "universe.json"
    P.freeze_source_universe(str(path), "__fixture__", 20260729)
    return path


@pytest.fixture
def run(tmp_path, cases, criteria_path, universe_path, monkeypatch):
    """Drive the PRODUCTION entry point, patching only INTERNAL seams.

    The production signature is exactly (out_dir, criteria_path,
    universe_path): a caller cannot supply rows, an evaluator or an
    authentication bypass. Tests reach the same places through the seams the
    module itself uses.
    """
    counter = {"n": 0}
    monkeypatch.setattr(M, "_derive_cases",
                        lambda payload: (list(cases), "a" * 40))
    monkeypatch.setattr(M, "_make_evaluator", lambda: FakeEvaluator())
    monkeypatch.setattr(M, "_authenticate_search_inputs",
                        lambda phase: {"phase": "unit-test"})
    monkeypatch.setattr(M, "assert_runtime_matches_records",
                        lambda *r, **k: None)
    monkeypatch.setattr(M.fpu_provenance, "worktree_clean", lambda: True)
    monkeypatch.setattr(M, "reproduce_universe", lambda raw: None)
    monkeypatch.setattr(M, "_reject_fixture_universe", lambda payload: None)

    def _run(**over):
        counter["n"] += 1
        kwargs = dict(out_dir=str(tmp_path / f"out{counter['n']}"),
                      criteria_path=str(criteria_path),
                      universe_path=str(universe_path))
        cases_override = over.pop("_cases", None)
        if cases_override is not None:
            monkeypatch.setattr(M, "_derive_cases",
                                lambda payload: (list(cases_override), "a" * 40))
        evaluator = over.pop("evaluator_factory", None)
        if evaluator is not None:
            monkeypatch.setattr(M, "_make_evaluator", evaluator)
        over.pop("require_clean_runtime", None)
        kwargs.update(over)
        return M.run_preflight(**kwargs)
    return _run


def _root(cases_):
    """One shipped search tree, for build_row-level tests."""
    return M.search_one(FakeEvaluator(), M.shipped_config(), cases_[0])


# --- build_row --------------------------------------------------------------

def test_build_row_tags_population(cases):
    root = _root(cases)
    row = M.build_row(root, C.CAP_GRID, 1.5, dict(cases[0]))
    assert row["population"] == "selected_a"
    assert row["population"] in M.POPULATIONS


def test_build_row_asserts_synchronous_provenance(cases, monkeypatch):
    seen = []
    real = M.assert_synchronous_tree
    monkeypatch.setattr(M, "assert_synchronous_tree",
                        lambda root, sims, **kw: (seen.append((sims, kw)),
                                                  real(root, sims, **kw))[1])
    root = _root(cases)
    M.build_row(root, C.CAP_GRID, 1.5, dict(cases[0]))
    assert seen and seen[0][0] == M.SIMULATIONS == 400
    assert seen[0][1] == {"search_execution_mode": M.SEARCH_EXECUTION_MODE}


def test_row_carries_per_cap_exposure_for_every_grid_cap(cases):
    root = _root(cases)
    row = M.build_row(root, C.CAP_GRID, 1.5, dict(cases[0]))
    for cap in C.CAP_GRID:
        for prefix in ("would_clip", "clipped_amount", "revisit_to_depth3_rate"):
            assert f"{prefix}_{cap}" in row, (prefix, cap)
    assert set(row["crossover"]) == {str(c) for c in C.CAP_GRID}


def test_row_carries_the_primary_and_both_descriptive_formulas(cases):
    root = _root(cases)
    row = M.build_row(root, C.CAP_GRID, 1.5, dict(cases[0]))
    assert C.PRIMARY_EXPOSURE_COLUMN in row
    assert "exposure_descriptive_count" in row
    assert "exposure_descriptive_clipped_mass" in row
    # The descriptives are reported, never able to overturn the primary.
    for spec in C.DESCRIPTIVE_EXPOSURE_FORMULAS.values():
        assert spec["can_rescue_primary_failure"] is False


# --- the routing / provenance boundary --------------------------------------

def test_measurement_routes_through_search_with_root_and_never_search_from_root(
        run, cases, monkeypatch):
    from scripts.GPU.alphazero import mcts as mcts_module
    calls = {"with_root": 0, "from_root": 0}
    real_with = mcts_module.MCTS.search_with_root

    def spy_with(self, *a, **k):
        calls["with_root"] += 1
        return real_with(self, *a, **k)

    def spy_from(self, *a, **k):
        calls["from_root"] += 1
        raise AssertionError("search_from_root must never be called")

    monkeypatch.setattr(mcts_module.MCTS, "search_with_root", spy_with)
    monkeypatch.setattr(mcts_module.MCTS, "search_from_root", spy_from)
    run()
    assert calls["from_root"] == 0
    assert calls["with_root"] == len(cases)
    # Supplemental tripwire only -- source text does not establish which path
    # ran, and the module's own comment explains WHY the batched route is
    # unused, so scope this to real references rather than prose.
    code = "\n".join(line.split("#")[0]
                     for line in Path(M.__file__).read_text().splitlines())
    assert ".search_from_root" not in code
    assert "search_from_root(" not in code


def test_search_execution_mode_is_not_accepted_from_caller_metadata(cases):
    root = _root(cases)
    meta = dict(cases[0], search_execution_mode="synchronous")
    with pytest.raises(ValueError, match="search_execution_mode"):
        M.build_row(root, C.CAP_GRID, 1.5, meta)


def test_valid_run_passes_the_constant_and_labels_the_artifact_with_it(
        run, monkeypatch, tmp_path):
    seen = []
    real = M.assert_synchronous_tree
    monkeypatch.setattr(M, "assert_synchronous_tree",
                        lambda root, sims, **kw: (seen.append(kw), real(root, sims, **kw))[1])
    result = run()
    assert seen, "assert_synchronous_tree was never called"
    for kwargs in seen:
        assert kwargs["search_execution_mode"] is M.SEARCH_EXECUTION_MODE
    artifact = json.loads(Path(result["preflight_artifact"]).read_text())
    assert artifact["search_execution_mode"] == M.SEARCH_EXECUTION_MODE


def test_a_mutated_mode_constant_refuses_and_writes_no_artifact(
        run, monkeypatch, tmp_path):
    """It must refuse BEFORE any search, not after.

    `assert_synchronous_tree` would reject the mode anyway, but only once every
    position had already been searched -- so a no-artifact assertion alone
    cannot tell an early refusal from a wasted multi-hour run. Counting searches
    is what distinguishes them.
    """
    from scripts.GPU.alphazero import mcts as mcts_module
    searches = []
    real = mcts_module.MCTS.search_with_root
    monkeypatch.setattr(mcts_module.MCTS, "search_with_root",
                        lambda self, *a, **k: (searches.append(1),
                                               real(self, *a, **k))[1])
    monkeypatch.setattr(M, "SEARCH_EXECUTION_MODE", "batched_waiter")
    out = tmp_path / "mutated"
    with pytest.raises(ValueError):
        run(out_dir=str(out))
    assert searches == [], "searches ran under a mutated mode constant"
    assert not (out / "preflight_artifact.json").exists()


def test_no_cli_flag_can_set_the_execution_mode():
    namespace = M.build_parser().parse_args(
        ["--criteria", "c.json", "--universe", "u.json", "--out", "o"])
    assert not hasattr(namespace, "search_execution_mode")
    assert not hasattr(namespace, "mode")


# --- artifacts --------------------------------------------------------------

def test_artifact_is_byte_reproducible_across_two_runs(run, tmp_path):
    first = run(out_dir=str(tmp_path / "a"))
    second = run(out_dir=str(tmp_path / "b"))
    assert (Path(first["preflight_artifact"]).read_bytes()
            == Path(second["preflight_artifact"]).read_bytes())
    assert first["preflight_artifact_sha1"] == second["preflight_artifact_sha1"]
    assert b"timestamp" not in Path(first["preflight_artifact"]).read_bytes()


def test_artifact_stamps_run_kind_scope_boundary_and_forbidden_flag(run):
    artifact = json.loads(Path(run()["preflight_artifact"]).read_text())
    assert artifact["run_kind"] == "shipped_only_preflight"
    assert artifact["scientific_interpretation_forbidden"] is True
    assert artifact["scope_boundary"] == C.SCOPE_BOUNDARY
    assert all(v is False for v in artifact["scope_boundary"].values())


def test_artifact_binds_the_criteria_sha1_and_the_universe_sha1(
        run, criteria_path, universe_path):
    import hashlib
    result = run()
    artifact = json.loads(Path(result["preflight_artifact"]).read_text())
    assert artifact["criteria_sha1"] == hashlib.sha1(
        criteria_path.read_bytes()).hexdigest()
    assert artifact["universe_sha1"] == hashlib.sha1(
        universe_path.read_bytes()).hexdigest()
    assert len(artifact["criteria_sha1"]) == len(artifact["universe_sha1"]) == 40


def test_census_positions_csv_carries_the_full_schema(run):
    import csv
    result = run()
    rows = list(csv.DictReader(open(result["census_positions"])))
    assert rows
    assert set(rows[0]) == set(C.CENSUS_SCHEMA), (
        set(rows[0]) ^ set(C.CENSUS_SCHEMA))
    # The matcher and the sizing both join on these.
    for row in rows:
        assert row["population"] in M.POPULATIONS
        assert len(row["game_content_sha1"]) == 40
        int(row["position_ply"])


def test_residual_rows_and_crossover_tables_are_emitted(run):
    import csv
    result = run()
    residual = list(csv.DictReader(open(result["residual_rows"])))
    crossover = list(csv.DictReader(open(result["crossover_tables"])))
    for cap in C.CAP_GRID:
        assert f"would_clip_{cap}" in (residual[0] if residual else {f"would_clip_{cap}": 1})
    # one crossover row per (case, cap)
    assert len(crossover) == len(result["rows"]) * len(C.CAP_GRID)
    assert {r["cap"] for r in crossover} == {str(c) for c in C.CAP_GRID}
    for row in crossover:
        float(row["predicted_reply_reduction"])       # signed, never clamped


# --- seeds ------------------------------------------------------------------

def test_selected_a_and_census_use_different_base_seeds():
    assert M.BASE_SEEDS["selected_a"] == 20260616
    assert M.BASE_SEEDS["census"] == 20260730
    assert M.BASE_SEEDS["selected_a"] != M.BASE_SEEDS["census"]
    a = {"population": "selected_a", "game_idx": 3, "position_ply": 7}
    assert M.derived_seed(a) == 20260616 ^ 3 ^ 7
    # The census does NOT use XOR: with game_idx and position_ply both < 1024 it
    # admits at most 1024 distinct values, and the real 1,974-row census
    # collapses to 841 -- 1,133 forced duplicates.
    c = {"population": "census", "game_content_sha1": "a" * 40, "position_ply": 7}
    assert M.derived_seed(c) != 20260730 ^ 3 ^ 7
    assert M.derived_seed(c) == M.derived_seed(dict(c))
    assert M.derived_seed(dict(c, position_ply=8)) != M.derived_seed(c)
    assert M.derived_seed(dict(c, game_content_sha1="b" * 40)) != M.derived_seed(c)
    assert M.SEED_POLICY is C.SEED_POLICY


def test_seed_audit_compares_complete_derived_sets_not_base_seeds(cases):
    sets = M.derived_seed_sets(cases)
    assert set(sets) == {"selected_a", "census"}
    assert sets["selected_a"] & sets["census"] == set()
    report = M.assert_seed_sets_disjoint(cases)
    assert report["intersection_size"] == 0
    assert report["selected_a_size"] == 1 and report["census_size"] == 2


def test_census_seed_uniqueness_is_required_and_selected_a_duplicates_are_not(cases):
    """Asymmetric by design: the census must be unique, selected-A's three
    historical duplicate groups are accepted provenance."""
    duplicate = dict(cases[2], game_content_sha1=cases[1]["game_content_sha1"],
                     position_ply=cases[1]["position_ply"], case_id="dupe")
    assert M.derived_seed(duplicate) == M.derived_seed(cases[1])
    with pytest.raises(ValueError, match="distinct seeds"):
        M.assert_seed_sets_disjoint([cases[0], cases[1], duplicate])
    # The historical A rule keeps its duplicates without complaint.
    a_dupe = dict(cases[0], case_id="a_dupe")
    report = M.assert_seed_sets_disjoint([cases[0], a_dupe, cases[1]])
    assert report["selected_a_cases"] == 2 and report["selected_a_size"] == 1


def test_a_seed_collision_aborts_before_any_evaluator_call(run, tmp_path, cases):
    """A collision is a pre-search STOP: by the execution phase the bases are
    committed and embedded in the emitted criteria."""
    built = []
    duplicate = dict(cases[2], game_content_sha1=cases[1]["game_content_sha1"],
                     position_ply=cases[1]["position_ply"], case_id="dupe")
    out = tmp_path / "collision"
    with pytest.raises(ValueError, match="seed"):
        run(_cases=[cases[0], cases[1], duplicate], out_dir=str(out),
            evaluator_factory=lambda: built.append(1))
    assert built == [], "an evaluator was constructed despite a seed collision"
    assert not out.exists()


# --- refusals (the non-vacuity evidence) ------------------------------------

def test_cli_refuses_when_the_frozen_criteria_artifact_is_missing(run, tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        run(criteria_path=str(tmp_path / "nope.json"))


def test_cli_refuses_when_the_criteria_sha1_does_not_match(run, criteria_path):
    payload = json.loads(criteria_path.read_text())
    payload["separation"]["min_auc"] = 0.55         # tampered threshold
    criteria_path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="criteria"):
        run()


def test_cli_refuses_a_nonzero_cap_search_configuration(run, monkeypatch):
    import dataclasses
    real = M.shipped_config

    def capped():
        cfg = real()
        object.__setattr__(cfg, "fpu_value", -0.20) if not dataclasses.is_dataclass(cfg) \
            else setattr(cfg, "fpu_value", -0.20)
        return cfg

    monkeypatch.setattr(M, "shipped_config", capped)
    with pytest.raises(ValueError, match="shipped"):
        run()


def test_cli_refuses_on_a_dirty_worktree(run, monkeypatch):
    monkeypatch.setattr(M.fpu_provenance, "worktree_clean", lambda: False)
    with pytest.raises(ValueError, match="worktree"):
        run()


def test_cli_emits_no_verdict_key(run):
    import re
    artifact = json.loads(Path(run()["preflight_artifact"]).read_text())
    blob = json.dumps(artifact).lower()
    forbidden = re.compile(r'"[^"]*(verdict|pass|fail|selected_formula|r_min|r_max)[^"]*"\s*:')
    assert not forbidden.search(blob), forbidden.search(blob).group(0)


def test_measurement_computes_no_threshold(run):
    artifact = json.loads(Path(run()["preflight_artifact"]).read_text())
    for banned in ("exposure_cutoff", "verdict", "r_min", "r_max",
                   "selected_formula", "conversion_efficiency"):
        assert banned not in artifact


# --- revision 31: the evidence chain ----------------------------------------

def test_production_path_accepts_no_caller_supplied_cases_or_c_puct():
    """A caller that supplies rows can measure arbitrary replays while the
    artifact still records a legitimate universe SHA-1; a caller that supplies
    c_puct changes the selection arithmetic the crossover mirrors."""
    import inspect
    params = inspect.signature(M.run_preflight).parameters
    # EXACTLY three, all keyword-only. No cases, no evaluator, no
    # authentication bypass: each would let a caller emit a normally shaped
    # artifact over evidence the module never authenticated.
    assert set(params) == {"out_dir", "criteria_path", "universe_path"}
    for name, param in params.items():
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, name
        assert param.default is inspect.Parameter.empty, name
    parser_dest = {a.dest for a in M.build_parser()._actions}
    assert "c_puct" not in parser_dest
    assert parser_dest == {"help", "criteria", "universe", "out"}


def test_census_cases_are_derived_from_the_verified_universe_record(universe_path):
    payload = json.loads(universe_path.read_text())
    cases = M.census_cases_from_universe(payload)
    assert len(cases) == len(payload["census_positions"])
    ordinals = {sha: i for i, sha in enumerate(payload["all_game_ids"])}
    for case, row in zip(cases, payload["census_positions"]):
        assert case["population"] == "census"
        # Task 4 names it canonical_sha1; the census schema names it
        # canonical_state_sha1. Converted, never carried under two names.
        assert case["canonical_state_sha1"] == row["canonical_sha1"]
        assert case["source_universe_ordinal"] == ordinals[row["game_content_sha1"]]
    # A row naming a game outside all_game_ids cannot be measured.
    tampered = dict(payload)
    tampered["census_positions"] = [dict(payload["census_positions"][0],
                                         game_content_sha1="f" * 40)]
    with pytest.raises(ValueError, match="absent from all_game_ids"):
        M.census_cases_from_universe(tampered)


def test_universe_record_is_re_derived_not_merely_labelled(run, universe_path):
    """Label checks alone would let a record with a substituted census pass
    while the artifact still recorded a legitimate universe SHA-1."""
    payload = json.loads(universe_path.read_text())
    payload["census_positions"] = payload["census_positions"][:-1]   # counts now lie
    universe_path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError):
        run()


def test_universe_selection_inputs_must_match_the_committed_module(run, universe_path):
    payload = json.loads(universe_path.read_text())
    payload["selection_inputs"] = dict(payload["selection_inputs"],
                                       value_based_rule="root_value_stm <= 0.3")
    universe_path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="selection_inputs"):
        run()


def test_records_must_describe_this_commit_and_a_clean_tree(criteria_path):
    """Every case is CONSTRUCTED, never borrowed from the ambient tree.

    The original first assertion required the freshly emitted record to be
    REJECTED, which only happens when the working tree is dirty -- so the test
    passed only while `git status --porcelain` was non-empty and failed at a
    genuinely clean HEAD. That is precisely the state Execution Step 1 demands
    ("`git status --porcelain` empty ... Full suite green"), so as written the
    two halves of that gate could never both hold. A provenance test must not
    depend on the environment it is asserting about.
    """
    payload = json.loads(criteria_path.read_text())
    head = M.fpu_provenance.git_commit()

    # Passes in EITHER environment: the record is built to be correct.
    clean = dict(payload, worktree_clean=True, git_commit=head)
    M.assert_runtime_matches_records(("criteria", clean), expected_commit=head)

    # The dirty case is built, not observed.
    dirty = dict(clean, worktree_clean=False)
    with pytest.raises(ValueError, match="dirty worktree"):
        M.assert_runtime_matches_records(("criteria", dirty),
                                         expected_commit=head)

    with pytest.raises(ValueError, match="git_commit"):
        M.assert_runtime_matches_records(("criteria", dict(clean, git_commit="0" * 40)),
                                         expected_commit=head)
    # The expected commit is SUPPLIED, never reread: a record matching the
    # current HEAD must still fail when the bracket captured a different one.
    with pytest.raises(ValueError, match="git_commit"):
        M.assert_runtime_matches_records(("criteria", clean),
                                         expected_commit="9" * 40)


def test_the_four_artifacts_are_published_as_one_transaction(run, tmp_path,
                                                             monkeypatch):
    """Writing the CSVs directly and the JSON afterwards leaves partial
    scientific outputs behind when a later step fails."""
    out = tmp_path / "transactional"
    real = M.canonical_json_bytes
    monkeypatch.setattr(M, "canonical_json_bytes",
                        lambda obj: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        run(out_dir=str(out))
    assert not out.exists(), "a failed run left partial artifacts behind"
    assert list(tmp_path.glob(".*staging*")) == []
    monkeypatch.setattr(M, "canonical_json_bytes", real)
    result = run(out_dir=str(tmp_path / "ok"))
    assert sorted(p.name for p in Path(result["census_positions"]).parent.iterdir()) == [
        "census_positions.csv", "crossover_tables.csv", "preflight_artifact.json",
        "residual_rows.csv"]


def test_artifact_binds_the_csv_hashes_so_the_set_is_mutually_bound(run):
    import hashlib
    result = run()
    artifact = json.loads(Path(result["preflight_artifact"]).read_text())
    for key, path in (("census_positions_sha1", result["census_positions"]),
                      ("residual_rows_sha1", result["residual_rows"]),
                      ("crossover_tables_sha1", result["crossover_tables"])):
        assert artifact[key] == hashlib.sha1(Path(path).read_bytes()).hexdigest()


def test_output_directory_must_not_already_exist(run, tmp_path):
    out = tmp_path / "already"
    out.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        run(out_dir=str(out))


def test_measurement_source_module_list_includes_the_producer_itself(run):
    assert "scripts/GPU/alphazero/diagnose_v18_residual_preflight.py" in \
        M.MEASUREMENT_SOURCE_MODULES
    for module in ("mcts.py", "v18_tree_walk.py", "v18_crossover.py",
                   "v18_preflight_criteria.py", "v18_control_pool.py",
                   "fpu_state_hash.py"):
        assert any(m.endswith(module) for m in M.MEASUREMENT_SOURCE_MODULES), module
    artifact = json.loads(Path(run()["preflight_artifact"]).read_text())
    assert set(artifact["source_sha1s"]) == set(M.MEASUREMENT_SOURCE_MODULES)


def test_full_frozen_configuration_is_asserted(run, monkeypatch):
    cfg = M.shipped_config()
    assert cfg.c_puct == M.FROZEN_C_PUCT == 1.5
    assert (cfg.eval_batch_size, cfg.stall_flush_sims,
            cfg.pending_virtual_visits) == M.BATCHING_TRIPLE == (14, 48, 8)
    assert cfg.n_simulations == M.SIMULATIONS == 400

    real = M.shipped_config

    def wrong_c_puct():
        bad = real()
        bad.c_puct = 2.5
        return bad

    monkeypatch.setattr(M, "shipped_config", wrong_c_puct)
    with pytest.raises(ValueError, match="c_puct"):
        run()


def test_artifact_records_the_authenticated_search_inputs(run):
    artifact = json.loads(Path(run()["preflight_artifact"]).read_text())
    assert "authenticated_search_inputs" in artifact
    assert artifact["c_puct"] == M.FROZEN_C_PUCT
    assert artifact["batching_triple"] == list(M.BATCHING_TRIPLE)
    assert artifact["population_order"] == ["selected_a", "census"]


def test_search_input_drift_between_opening_and_closing_yields_no_artifact(
        run, tmp_path, monkeypatch):
    """The checkpoint and both reservoirs are re-authenticated after the last
    search. Drift between the two checks must publish nothing."""
    calls = {"n": 0}

    def drifting(phase):
        calls["n"] += 1
        return {"checkpoint_sha1": "a" * 40 if calls["n"] == 1 else "b" * 40}

    monkeypatch.setattr(M, "_authenticate_search_inputs", drifting)
    out = tmp_path / "drifted"
    with pytest.raises(ValueError, match="changed during the measurement"):
        run(out_dir=str(out))
    assert calls["n"] == 2, "the closing authentication never ran"
    assert not out.exists()


def test_search_inputs_are_authenticated_before_the_evaluator(run, monkeypatch):
    order = []
    real = M._authenticate_search_inputs
    monkeypatch.setattr(M, "_authenticate_search_inputs",
                        lambda phase: (order.append(phase), real(phase))[1])
    run(evaluator_factory=lambda: (order.append("evaluator"), FakeEvaluator())[1])
    assert order == ["opening", "evaluator", "closing"], order


# --- revision 32 ------------------------------------------------------------

def test_production_rejects_a_fixture_universe(universe_path):
    """A fixture record has no reservoir behind it, so nothing it claims can be
    reproduced. The whole chain would pass on data that was never generated."""
    payload = json.loads(universe_path.read_text())
    assert payload["selected_universe"] == "fixture"
    with pytest.raises(ValueError, match="FIXTURE"):
        M._reject_fixture_universe(payload)
    with pytest.raises(ValueError, match="SELECTED_UNIVERSE"):
        M._reject_fixture_universe(dict(payload, selected_universe={"name": "x"}))
    M._reject_fixture_universe(
        dict(payload, selected_universe=dict(P.SELECTED_UNIVERSE)))


def test_coherently_tampered_universe_fails_reproduction(universe_path, monkeypatch):
    """A substituted census whose counts and geometry were RECOMPUTED to match
    passes `_assert_record_reconciles` untouched: internal arithmetic only
    proves self-consistency. Only re-emission catches it."""
    payload = json.loads(universe_path.read_text())
    kept = payload["census_positions"][:-1]
    yielding = {r["game_content_sha1"] for r in kept}
    coherent = dict(payload)
    coherent["census_positions"] = kept
    per_phase = {ph: sum(1 for r in kept if r["phase"] == ph)
                 for ph in ("opening", "early_mid", "midgame", "late")}
    coherent["per_phase"] = per_phase
    geometry = dict(payload["census_geometry"])
    geometry["positions_after_position_exclusions"] = len(kept)
    geometry["positions_before_position_exclusions"] = (
        len(kept) + geometry["positions_excluded_by_canonical_hash"])
    geometry["per_phase_after"] = per_phase
    coherent["census_geometry"] = geometry
    coherent["zero_yield_games"] = len(payload["all_game_ids"]) - len(yielding)
    # Self-consistent: the committed reconciliation accepts it.
    P._assert_record_reconciles(coherent)
    # Reproduction does not.
    supplied = canonical_json_bytes(coherent)
    monkeypatch.setattr(M.control_pool, "freeze_source_universe",
                        lambda path, name, seed: Path(path).write_bytes(
                            canonical_json_bytes(payload)))
    with pytest.raises(ValueError, match="does not reproduce"):
        M.reproduce_universe(supplied)
    # The untampered bytes do reproduce.
    M.reproduce_universe(canonical_json_bytes(payload))


def test_selected_a_source_bytes_are_authenticated(monkeypatch):
    """The reach/separation source carries an existing pin; parsing it without
    checking that pin authenticates nothing."""
    from scripts.GPU.alphazero import v18_control_pool as pool
    assert pool.FORBIDDEN_SOURCE_SHA1S["gate_A"] == \
        "175c73ef2c761df83ccf5f5cd935152093f8dfb1"
    cases, sha1 = M.selected_a_cases()
    assert sha1 == pool.FORBIDDEN_SOURCE_SHA1S["gate_A"]
    assert len(cases) == 30
    assert {c["population"] for c in cases} == {"selected_a"}
    for case in cases:
        assert len(case["canonical_state_sha1"]) == 40

    real = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self, *a, **k: (
        real(self, *a, **k) + b"\n" if str(self).endswith(M.A_SOURCE.split("/")[-1])
        and "black_loss" in str(self) else real(self, *a, **k)))
    with pytest.raises(ValueError, match="pinned"):
        M.selected_a_cases()


def test_selected_a_binds_the_reservoir_before_deriving_identities(monkeypatch):
    phases = []
    real = M.capture_v18.authenticate_replay_reservoir
    monkeypatch.setattr(M.capture_v18, "authenticate_replay_reservoir",
                        lambda phase="opening": (phases.append(phase),
                                                 real(phase))[1])
    M.selected_a_cases()
    assert phases and phases[0] == "pre_derivation"


def test_add_noise_is_one_validated_constant(run, tmp_path, monkeypatch):
    """Two independent `False` literals would let a change at the call site run
    noisy search while the artifact still published add_noise: false."""
    assert M.ADD_NOISE is False
    source = Path(M.__file__).read_text()
    assert "add_noise=ADD_NOISE" in source
    assert "add_noise=False" not in source

    from scripts.GPU.alphazero import mcts as mcts_module
    searches = []
    real = mcts_module.MCTS.search_with_root
    monkeypatch.setattr(mcts_module.MCTS, "search_with_root",
                        lambda self, *a, **k: (searches.append(1),
                                               real(self, *a, **k))[1])
    monkeypatch.setattr(M, "ADD_NOISE", True)
    out = tmp_path / "noisy"
    with pytest.raises(ValueError, match="noiseless|ADD_NOISE"):
        run(out_dir=str(out))
    assert searches == [], "searches ran with noise enabled"
    assert not out.exists()


def test_existing_output_is_refused_before_any_search(run, tmp_path, monkeypatch):
    from scripts.GPU.alphazero import mcts as mcts_module
    searches, evaluators = [], []
    real = mcts_module.MCTS.search_with_root
    monkeypatch.setattr(mcts_module.MCTS, "search_with_root",
                        lambda self, *a, **k: (searches.append(1),
                                               real(self, *a, **k))[1])
    out = tmp_path / "occupied"
    out.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        run(out_dir=str(out),
            evaluator_factory=lambda: (evaluators.append(1), FakeEvaluator())[1])
    assert searches == [] and evaluators == []


def test_seed_audit_reports_the_complete_asymmetric_policy(run):
    artifact = json.loads(Path(run()["preflight_artifact"]).read_text())
    audit = artifact["seed_audit"]
    # "rule" named only the selected-A XOR, which the census does not use.
    assert "rule" not in audit
    assert audit["policy"] == C.SEED_POLICY
    assert audit["policy"]["census"]["rule"] == "sha1_digest"


def test_selected_a_seed_shape_is_checked_against_the_frozen_criteria():
    import csv as _csv
    rows = [r for r in _csv.DictReader(open(M.A_SOURCE))
            if r["checkpoint"] == "0001"]
    cases = [{"population": "selected_a", "game_idx": int(r["game_idx"]),
              "position_ply": int(r["position_ply"])} for r in rows]
    audit = M.assert_seed_sets_disjoint(cases)
    assert audit["selected_a_matches_frozen_criteria"] is True
    assert (audit["selected_a_cases"], audit["selected_a_size"]) == (30, 27)
    assert audit["duplicate_seeds_within_population"]["selected_a"] == 3


def test_result_determining_source_set_is_complete(run):
    for required in ("evaluator.py", "local_evaluator.py", "network.py",
                     "game/twixt_state.py", "fpu_provenance.py",
                     "capture_v17_abcd_selected_moves.py",
                     "diagnose_v18_residual_preflight.py"):
        assert any(m.endswith(required) for m in M.MEASUREMENT_SOURCE_MODULES), required
    for module in M.MEASUREMENT_SOURCE_MODULES:
        assert Path(module).exists(), module
    artifact = json.loads(Path(run()["preflight_artifact"]).read_text())
    assert set(artifact["source_sha1s"]) == set(M.MEASUREMENT_SOURCE_MODULES)


def test_universe_verification_actually_calls_reproduction(universe_path, monkeypatch):
    """Non-vacuity for the call SITE: patching `reproduce_universe` in other
    tests would hide its removal, so assert it is invoked with the exact bytes
    that were hashed."""
    seen = []
    monkeypatch.setattr(M, "_reject_fixture_universe", lambda payload: None)
    monkeypatch.setattr(M, "reproduce_universe", lambda raw: seen.append(raw))
    payload, sha1 = M.load_verified_universe(str(universe_path))
    assert seen == [universe_path.read_bytes()]
    import hashlib
    assert sha1 == hashlib.sha1(seen[0]).hexdigest()


# --- revision 33 ------------------------------------------------------------

# Written out INDEPENDENTLY of MEASUREMENT_SOURCE_MODULES: comparing against the
# module's own tuple would let an omission pass by simply not being listed.
REQUIRED_SOURCE_MODULES = {
    "scripts/GPU/alphazero/mcts.py",
    "scripts/GPU/alphazero/opening_diagnostics.py",
    "scripts/GPU/alphazero/eval_runner.py",
    "scripts/GPU/alphazero/evaluator.py",
    "scripts/GPU/alphazero/local_evaluator.py",
    "scripts/GPU/alphazero/probe_eval.py",
    "scripts/GPU/alphazero/network.py",
    "scripts/GPU/alphazero/game/__init__.py",
    "scripts/GPU/alphazero/game/twixt_state.py",
    "scripts/GPU/alphazero/position_probe_cases.py",
    "scripts/GPU/alphazero/goal_line_trigger_probe_cases.py",
    "scripts/GPU/alphazero/fpu_state_hash.py",
    "scripts/GPU/alphazero/fpu_provenance.py",
    "scripts/GPU/alphazero/fpu_dev_reservoir_protocol.py",
    "scripts/GPU/alphazero/diagnose_fpu_baseline_policy_mass.py",
    "scripts/GPU/alphazero/fpu_v17_provenance.py",
    "scripts/GPU/alphazero/v18_provisional_backup.py",
    "scripts/GPU/alphazero/v18_tree_walk.py",
    "scripts/GPU/alphazero/v18_crossover.py",
    "scripts/GPU/alphazero/v18_preflight_criteria.py",
    "scripts/GPU/alphazero/v18_control_pool.py",
    "scripts/GPU/alphazero/capture_v18_a6400.py",
    "scripts/GPU/alphazero/capture_v17_abcd_selected_moves.py",
    "scripts/GPU/alphazero/diagnose_v18_residual_preflight.py",
}


def test_result_determining_set_covers_the_independent_required_list(run):
    missing = REQUIRED_SOURCE_MODULES - set(M.MEASUREMENT_SOURCE_MODULES)
    assert not missing, f"omitted from the frozen set: {sorted(missing)}"
    for module in M.MEASUREMENT_SOURCE_MODULES:
        assert Path(module).exists(), module
    artifact = json.loads(Path(run()["preflight_artifact"]).read_text())
    assert set(artifact["source_sha1s"]) == set(M.MEASUREMENT_SOURCE_MODULES)
    assert REQUIRED_SOURCE_MODULES <= set(artifact["source_sha1s"])


def test_runtime_identity_covers_head_worktree_and_every_source():
    identity = M.runtime_identity()
    assert set(identity) == {"git_commit", "worktree_clean", "source_sha1s"}
    assert set(identity["source_sha1s"]) == set(M.MEASUREMENT_SOURCE_MODULES)
    assert all(len(v) == 40 for v in identity["source_sha1s"].values())


def test_a_mid_run_source_change_writes_nothing(run, tmp_path, monkeypatch):
    """HEAD and the source tree have hours to move during a real run. The trees
    were produced by the OPENING state, so an artifact describing the ending
    bytes would misattribute them."""
    calls = {"n": 0}
    real = M.runtime_identity

    def drifting():
        calls["n"] += 1
        identity = real()
        if calls["n"] > 1:                      # a source moved mid-run
            identity = dict(identity, source_sha1s=dict(
                identity["source_sha1s"],
                **{"scripts/GPU/alphazero/mcts.py": "0" * 40}))
        return identity

    monkeypatch.setattr(M, "runtime_identity", drifting)
    out = tmp_path / "drifted_source"
    with pytest.raises(ValueError, match="running code changed"):
        run(out_dir=str(out))
    assert calls["n"] == 2, "the closing runtime check never ran"
    assert not out.exists()


def test_a_mid_run_head_change_writes_nothing(run, tmp_path, monkeypatch):
    calls = {"n": 0}
    real = M.runtime_identity

    def drifting():
        calls["n"] += 1
        identity = real()
        return identity if calls["n"] == 1 else dict(identity, git_commit="0" * 40)

    monkeypatch.setattr(M, "runtime_identity", drifting)
    out = tmp_path / "drifted_head"
    with pytest.raises(ValueError, match="running code changed"):
        run(out_dir=str(out))
    assert not out.exists()


def test_a_dirty_worktree_at_the_opening_runtime_check_writes_nothing(
        run, tmp_path, monkeypatch):
    real = M.runtime_identity
    monkeypatch.setattr(M, "runtime_identity",
                        lambda: dict(real(), worktree_clean=False))
    out = tmp_path / "dirty_runtime"
    with pytest.raises(ValueError, match="dirty worktree"):
        run(out_dir=str(out))
    assert not out.exists()


def test_artifact_publishes_the_opening_runtime_identity(run, monkeypatch):
    """On the passing path a third read returns identical bytes, so only the
    CALL STRUCTURE distinguishes republishing the authenticated opening identity
    from taking an unauthenticated one at emission time."""
    captured, calls, hashed = {}, {"n": 0}, []
    real = M.runtime_identity
    real_sha1 = M.fpu_provenance.file_sha1

    def recording():
        calls["n"] += 1
        identity = real()
        captured.setdefault("opening", identity)
        return identity

    def counting(path):
        if path in M.MEASUREMENT_SOURCE_MODULES:
            hashed.append(path)
        return real_sha1(path)

    monkeypatch.setattr(M, "runtime_identity", recording)
    monkeypatch.setattr(M.fpu_provenance, "file_sha1", counting)
    artifact = json.loads(Path(run()["preflight_artifact"]).read_text())
    assert artifact["runtime_identity_bracketed"] is True
    assert artifact["git_commit"] == captured["opening"]["git_commit"]
    assert artifact["source_sha1s"] == captured["opening"]["source_sha1s"]
    assert calls["n"] == 2, "expected exactly an opening and a closing identity"
    # Exactly two passes over the module list: opening and closing. A third
    # would be an unauthenticated read at emission time.
    assert len(hashed) == 2 * len(M.MEASUREMENT_SOURCE_MODULES), len(hashed)


def test_census_reservoir_is_resolved_by_the_selected_universe_name():
    import inspect
    source = inspect.getsource(M._authenticate_census_reservoir)
    assert 'SELECTED_UNIVERSE["name"]' in source
    assert "CANDIDATE_UNIVERSES[0]" not in source


def test_module_docstring_and_dead_code_are_clean():
    source = Path(M.__file__).read_text()
    assert "emits four artifacts" in source
    assert "_cases` exists for the unit tests" not in source
    assert not hasattr(M, "SEED_RULE")
    assert not hasattr(M, "_atomic_write")
    assert "MEASUREMENT_SOURCE_MODULES = (" in source.split('"""', 2)[2]


def test_head_moving_during_evidence_derivation_writes_nothing(
        run, tmp_path, monkeypatch):
    """The bracket must open BEFORE the evidence is verified.

    Criteria verification, universe re-emission and case derivation all read the
    source tree. If the bracket opened after them, a clean HEAD move during
    those steps would become the opening identity, the closing check would
    agree with it, and the artifact would bind the new HEAD to records
    authenticated under the old one.
    """
    state = {"head": "1" * 40}
    real_cases = M._derive_cases
    monkeypatch.setattr(M, "runtime_identity",
                        lambda: {"git_commit": state["head"],
                                 "worktree_clean": True, "source_sha1s": {}})

    def moving_head(payload):
        state["head"] = "2" * 40          # HEAD moves mid-derivation
        return real_cases(payload)

    monkeypatch.setattr(M, "_derive_cases", moving_head)
    out = tmp_path / "head_moved"
    with pytest.raises(ValueError, match="running code changed"):
        run(out_dir=str(out))
    assert not out.exists()


def test_record_check_takes_the_commit_rather_than_rereading_head():
    """Rereading HEAD here would compare records against whatever the tree is
    at that instant, not the state the run is bracketed by."""
    import inspect
    params = inspect.signature(M.assert_runtime_matches_records).parameters
    assert "expected_commit" in params
    assert params["expected_commit"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "fpu_provenance.git_commit()" not in inspect.getsource(
        M.assert_runtime_matches_records)


def test_records_are_checked_against_the_captured_commit_not_a_fresh_read(
        run, tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(M, "runtime_identity",
                        lambda: {"git_commit": "7" * 40, "worktree_clean": True,
                                 "source_sha1s": {}})
    monkeypatch.setattr(M, "assert_runtime_matches_records",
                        lambda *r, **k: seen.update(k))
    run(out_dir=str(tmp_path / "captured"))
    assert seen == {"expected_commit": "7" * 40}
