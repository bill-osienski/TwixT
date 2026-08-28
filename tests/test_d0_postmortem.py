"""D0: zero-inference postmortem of the canonical 64-game L0 match.

NO EXECUTION. No model, no JVM, no T1j query, no inference, no game, no seed,
no training. Every test here reads the published record and computes
deterministic board facts from our own rules engine.

The gate under test that matters most is the DISCOVERY/CONFIRMATION SPLIT.
Diagnostic features may be computed on repetitions 0 and 1 only; repetitions 2
and 3 stay closed until D2 freezes a hypothesis. That is enforced by two
separate functions rather than one function with a flag, because a flag
defaults -- and a default is a switch-off.
"""
import json

import pytest

from scripts.GPU.alphazero import d0_postmortem as D0
from scripts.GPU.alphazero import l0_match_plan as PLAN
from scripts.GPU.alphazero import l0_match_rules as RULES

RECORD = "docs/superpowers/evidence/2026-08-27-t1j-l0-canonical-match/06_l0_match_results.jsonl"
PLAN_JSON = "docs/superpowers/evidence/2026-08-26-t1j-l0-larger-match/01_l0_match_plan.json"


def _rows(path=RECORD):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _write(tmp_path, rows, name="tampered.jsonl"):
    p = tmp_path / name
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(p)


# --------------------------------------------------------------- digest binding

def test_bind_record_binds_the_canonical_record():
    b = D0.bind_record(RECORD, PLAN_JSON)
    assert b.plan_sha256 == PLAN.L0_PLAN_SHA256
    assert b.task_digest == RULES.L0_TASK_DIGEST
    assert len(b.tasks) == 64


def test_bind_record_refuses_a_record_whose_header_plan_digest_was_edited(tmp_path):
    rows = _rows()
    for r in rows:
        if r["record_type"] == "run_header":
            r["plan_sha256"] = "0" * 64
    with pytest.raises(D0.D0BindingError, match="plan sha256"):
        D0.bind_record(_write(tmp_path, rows), PLAN_JSON)


def test_bind_record_refuses_a_record_whose_embedded_tasks_were_edited(tmp_path):
    """Editing a task must break the recomputed digest, not just the pinned one."""
    rows = _rows()
    for r in rows:
        if r["record_type"] == "run_header":
            r["identity"]["plan"]["plan"]["tasks"][0]["opening"] = "o9_forged"
    with pytest.raises(D0.D0BindingError, match="task digest"):
        D0.bind_record(_write(tmp_path, rows), PLAN_JSON)


def test_bind_record_refuses_a_plan_file_that_is_not_the_pinned_plan(tmp_path):
    forged = tmp_path / "plan.json"
    forged.write_text(json.dumps({"tasks": []}), encoding="utf-8")
    with pytest.raises(D0.D0BindingError, match="plan sha256"):
        D0.bind_record(RECORD, str(forged))


def test_moves_cannot_be_read_without_a_bound_record():
    """[A1]-3: digest binding is a PRECONDITION of reconstruction, not a peer.

    The only producer of a Bound is bind_record, which refuses on any digest
    mismatch. So there is no path from an unverified record to a move.
    """
    with pytest.raises(D0.D0BindingError, match="not a bound record"):
        D0.game_moves({"tasks": []}, "l0match-000-strong6-o1_center-t1j_red-r0")


# ------------------------------------------------- reconstruction from the record

def test_game_moves_prepends_the_opening_prefix_from_the_embedded_plan():
    b = D0.bind_record(RECORD, PLAN_JSON)
    tid = "l0match-000-strong6-o1_center-t1j_red-r0"
    moves = D0.game_moves(b, tid)
    assert moves[:6] == b.openings["o1_center"]
    assert len(moves) == b.results[tid]["plies"]


def test_every_task_reconstructs_to_its_recorded_winner_and_length():
    """Integrity over BOTH halves -- permitted by 4.2, and not a diagnostic."""
    b = D0.bind_record(RECORD, PLAN_JSON)
    checked = 0
    for tid, res in b.results.items():
        st = D0.replay(b, tid)
        assert st.ply == res["plies"], tid
        assert st.winner() == res["winner"], tid
        checked += 1
    assert checked == 64


# ------------------------------------------------------ THE SPLIT, as a real gate

def test_game_features_computes_on_a_discovery_repetition():
    b = D0.bind_record(RECORD, PLAN_JSON)
    tid = next(t for t, s in b.starts.items() if s["rep"] in D0.DISCOVERY_REPS)
    rows = D0.game_features(b, tid)
    assert rows and all(r["task_id"] == tid for r in rows)


@pytest.mark.parametrize("rep", D0.CONFIRMATION_REPS)
def test_game_features_refuses_a_confirmation_repetition(rep):
    """NEGATIVE CONTROL. Confirmation diagnostics stay closed until D2."""
    b = D0.bind_record(RECORD, PLAN_JSON)
    tid = next(t for t, s in b.starts.items() if s["rep"] == rep)
    with pytest.raises(D0.D0ScopeError, match="confirmation"):
        D0.game_features(b, tid)


def test_the_split_is_two_functions_and_game_features_takes_no_override():
    """A flag would default, and a default is a switch-off. There is no flag."""
    import inspect
    params = inspect.signature(D0.game_features).parameters
    assert list(params) == ["bound", "task_id"], params


def test_discovery_and_confirmation_partition_the_64_games():
    b = D0.bind_record(RECORD, PLAN_JSON)
    disc = D0.discovery_task_ids(b)
    conf = [t for t, s in b.starts.items() if s["rep"] in D0.CONFIRMATION_REPS]
    assert len(disc) == 32 and len(conf) == 32
    assert set(disc).isdisjoint(conf)
    assert len(set(disc) | set(conf)) == 64


# ------------------------------------------------------------------- inventory

def test_inventory_covers_both_halves():
    b = D0.bind_record(RECORD, PLAN_JSON)
    inv = D0.inventory(b)
    assert inv["n_games"] == 64
    assert inv["discovery"]["n_games"] == 32
    assert inv["confirmation"]["n_games"] == 32
    assert inv["reconstructed_ok"] == 64


def test_inventory_carries_outcome_counts_but_no_board_diagnostics():
    """Outcome counts are permitted on both halves; features are not."""
    b = D0.bind_record(RECORD, PLAN_JSON)
    inv = D0.inventory(b)
    assert inv["confirmation"]["t1j_points"] > 0          # outcomes ARE present
    flat = json.dumps(inv)
    for banned in ("largest_component", "boundary_distance", "threat",
                   "legal_moves", "new_bridges"):
        assert banned not in flat, f"inventory leaked diagnostic {banned!r}"


# ------------------------------------------------------------ board derivations

from scripts.GPU.alphazero.game.twixt_state import TwixtState


def _mini(moves, active=5):
    """Build a small state by legal alternating play, so bridges are real."""
    st = TwixtState(active_size=active)
    for m in moves:
        st = st.apply_move(m)
    return st


def test_component_boundary_distance_is_geometric_and_per_component():
    """[A1]-4: the NEW derivation. Red's goals are rows 0 and active-1."""
    st = TwixtState(pegs={(5, 10): "red"}, ply=1, to_move="black")
    comps = D0.component_boundary_distances(st, "red")
    assert comps == [{"size": 1, "to_goal1": 5, "to_goal2": 18}]


def test_component_boundary_distance_reports_each_disconnected_component():
    st = TwixtState(pegs={(5, 10): "red", (20, 2): "red"}, ply=2, to_move="black")
    comps = D0.component_boundary_distances(st, "red")
    assert sorted(c["to_goal1"] for c in comps) == [5, 20]
    assert sorted(c["to_goal2"] for c in comps) == [3, 18]


def test_component_boundary_distance_uses_columns_for_black():
    """Black connects col 0 <-> col active-1, so its distances are column-based."""
    st = TwixtState(pegs={(10, 3): "black"}, ply=1, to_move="red")
    assert D0.component_boundary_distances(st, "black") == [
        {"size": 1, "to_goal1": 3, "to_goal2": 20}]


def test_bridged_pegs_form_one_component_with_the_nearer_distance():
    st = _mini([(0, 1), (1, 3), (2, 2)])          # red (0,1)-(2,2) are knight-linked
    comps = D0.component_boundary_distances(st, "red")
    assert comps == [{"size": 2, "to_goal1": 0, "to_goal2": 2}]


def test_one_ply_threats_finds_every_immediately_winning_move():
    """Both completions count. An earlier version of this test asserted only
    (4,1) and missed (4,3); the scan is exhaustive by design, not shortest-path."""
    st = _mini([(0, 1), (1, 3), (2, 2), (3, 1)])
    assert st.to_move == "red"
    assert D0.one_ply_threats(st) == [(4, 1), (4, 3)]


def test_one_ply_threats_is_empty_on_a_quiet_position():
    assert D0.one_ply_threats(_mini([(0, 1), (1, 3)])) == []


def test_a_ply_that_leaves_the_opponent_a_win_is_recorded_as_ignored():
    """Red is one move from winning; black plays elsewhere and does not stop it."""
    st = _mini([(0, 1), (1, 3), (2, 2)])           # black to move, red threatens (4,1)
    assert st.to_move == "black"
    row = D0.ply_features(st, (3, 3))              # black plays away
    assert row["under_threat"] is True
    assert row["answered_threat"] is False
    assert row["ignored_threat"] is True


def test_a_ply_that_occupies_the_winning_hole_is_recorded_as_answered():
    """Red holds (0,1) and (4,1); its only completion is (2,2), which black may take."""
    st = _mini([(0, 1), (1, 3), (4, 1)])
    assert st.to_move == "black"
    row = D0.ply_features(st, (2, 2))
    assert row["under_threat"] is True
    assert row["answered_threat"] is True
    assert row["ignored_threat"] is False


def test_a_threat_on_the_movers_own_goal_edge_cannot_be_occupied_by_the_opponent():
    """A rules asymmetry the derivation must not paper over: black may never sit
    on rows 0 or active-1, so a red completion there is unanswerable by
    occupation. `answered_threat` must not be reachable by an illegal defence."""
    st = _mini([(0, 1), (1, 3), (2, 2)])
    assert st.to_move == "black"
    assert D0.one_ply_threats(D0._flip(st)) == [(4, 1), (4, 3)]
    assert st.is_valid_placement(4, 1) is False
    with pytest.raises(ValueError, match="Illegal move"):
        D0.ply_features(st, (4, 1))


def test_a_winning_move_is_recorded_as_an_immediate_win():
    st = _mini([(0, 1), (1, 3), (2, 2), (3, 1)])
    row = D0.ply_features(st, (4, 1))
    assert row["immediate_win"] is True


def test_ply_features_counts_pegs_bridges_components_and_holes():
    st = _mini([(0, 1), (1, 3), (2, 2)])
    row = D0.ply_features(st, (3, 3))
    assert row["pegs_red"] == 2 and row["pegs_black"] == 1
    assert row["bridges_red"] == 1 and row["bridges_black"] == 0
    assert row["components_red"] == 1 and row["largest_component_red"] == 2
    assert row["empty_holes"] == 5 * 5 - 3
    assert row["legal_move_count"] == len(st.legal_moves())


def test_blocked_bridge_opportunities_counts_knight_neighbours_that_did_not_link():
    """A same-colour knight neighbour that forms no bridge was blocked by a crossing."""
    st = _mini([(0, 1), (1, 3), (2, 2)])
    row = D0.ply_features(st, (3, 3))
    assert row["blocked_bridge_opportunities"] >= 0
    assert row["new_bridges"] == 0                 # black (3,3) links to nothing


# ------------------------------------------------------- aggregation and the gate

def test_every_aggregate_retains_its_denominator():
    b = D0.bind_record(RECORD, PLAN_JSON)
    rows = D0.game_features(b, D0.discovery_task_ids(b)[0])
    agg = D0.aggregate(rows)
    for dimension, groups in agg.items():
        for key, cell in groups.items():
            assert cell["n"] > 0, f"{dimension}/{key} has no denominator"


def test_aggregate_groups_by_opening_colour_arm_winner_and_phase():
    b = D0.bind_record(RECORD, PLAN_JSON)
    rows = D0.game_features(b, D0.discovery_task_ids(b)[0])
    assert set(D0.aggregate(rows)) == {"opening", "colour_arm", "winner", "phase"}


def test_phase_is_derived_here_and_not_imported_from_the_d1_analyzer():
    assert D0.phase_of(3, 40) == "opening"
    assert D0.phase_of(39, 40) == "late"
    assert len({D0.phase_of(p, 40) for p in range(40)}) == 4


def test_gate_refuses_a_signature_confined_to_one_opening():
    sig = {"name": "s", "column": "ignored_threat", "colour_scope": "both",
           "d1_observable": "root visit share of the answering move"}
    rows = [{"opening": "o1_center", "colour_arm": "t1j_red", "ignored_threat": True}]
    v = D0.evaluate_gate([sig], rows)
    assert v["verdict"] == "NO_GO"
    assert "more than one opening" in v["signatures"][0]["failed"][0]


def test_gate_refuses_a_signature_present_in_one_arm_but_claiming_both():
    sig = {"name": "s", "column": "ignored_threat", "colour_scope": "both",
           "d1_observable": "root visit share"}
    rows = [{"opening": o, "colour_arm": "t1j_red", "ignored_threat": True}
            for o in ("o1_center", "o2_offcenter")]
    v = D0.evaluate_gate([sig], rows)
    assert v["verdict"] == "NO_GO"
    assert any("colour arm" in f for f in v["signatures"][0]["failed"])


def test_gate_accepts_an_explicitly_colour_scoped_signature():
    sig = {"name": "s", "column": "ignored_threat", "colour_scope": "t1j_red",
           "d1_observable": "root visit share"}
    rows = [{"opening": o, "colour_arm": "t1j_red", "ignored_threat": True,
             "plies_from_terminal": d}
            for o in ("o1_center", "o2_offcenter") for d in (1, 9)]
    assert D0.evaluate_gate([sig], rows)["verdict"] == "GO"


def test_gate_refuses_a_signature_with_no_named_d1_observable():
    sig = {"name": "s", "column": "ignored_threat", "colour_scope": "both",
           "d1_observable": ""}
    rows = [{"opening": o, "colour_arm": a, "ignored_threat": True}
            for o in ("o1_center", "o2_offcenter") for a in ("t1j_red", "t1j_black")]
    v = D0.evaluate_gate([sig], rows)
    assert v["verdict"] == "NO_GO"
    assert any("observable" in f for f in v["signatures"][0]["failed"])


def test_gate_refuses_a_column_the_holdout_could_not_recompute():
    """Condition 3: the signature must be a column ply_features itself produces."""
    sig = {"name": "s", "column": "hand_labelled_blunder", "colour_scope": "both",
           "d1_observable": "root visit share"}
    rows = [{"opening": o, "colour_arm": a, "hand_labelled_blunder": True}
            for o in ("o1_center", "o2_offcenter") for a in ("t1j_red", "t1j_black")]
    v = D0.evaluate_gate([sig], rows)
    assert v["verdict"] == "NO_GO"
    assert any("held-out" in f for f in v["signatures"][0]["failed"])


def test_no_candidate_signatures_at_all_is_a_clean_no_go():
    assert D0.evaluate_gate([], [])["verdict"] == "NO_GO"


# ------------------------------------------ [A1]-6 structural control, by AST

def test_the_module_imports_no_d1_telemetry_vocabulary():
    """AST, never grep: this module's own docstring NAMES the banned module, so
    a grep would match the prose forbidding the import and pass vacuously."""
    import ast
    import pathlib
    tree = ast.parse(pathlib.Path(D0.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported |= {f"{node.module or ''}.{a.name}" for a in node.names}
    assert not any("eval_loss" in m for m in imported), imported
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for banned in ("root_value", "selected_visit_rank", "root_top1_share"):
        assert banned not in used, f"D1 telemetry identifier {banned!r} used"


def test_the_grep_this_control_replaces_would_have_passed_vacuously():
    """Proves the AST check is not redundant: the banned names ARE in the file."""
    import pathlib
    text = pathlib.Path(D0.__file__).read_text(encoding="utf-8")
    assert "eval_loss_replay_analysis" in text and "root_value" in text


# --------------- closing two holes where the tests above passed vacuously

def test_game_features_rows_carry_the_full_section_4_3_feature_set():
    """game_features must actually CALL ply_features. It did not, and the
    aggregate tests above still passed -- they only checked dimension names."""
    b = D0.bind_record(RECORD, PLAN_JSON)
    rows = D0.game_features(b, D0.discovery_task_ids(b)[0])
    required = {
        "task_id", "opening", "colour_arm", "rep", "ply", "mover", "winner",
        "plies_from_terminal", "phase", "legal_move_count", "empty_holes",
        "pegs_red", "pegs_black", "bridges_red", "bridges_black",
        "components_red", "components_black",
        "largest_component_red", "largest_component_black",
        "min_goal1_distance_red", "min_goal2_distance_red",
        "min_goal1_distance_black", "min_goal2_distance_black",
        "new_bridges", "blocked_bridge_opportunities", "immediate_win",
        "under_threat", "answered_threat", "ignored_threat", "created_threat",
    }
    missing = required - set(rows[0])
    assert not missing, f"ply_features never reached the row: {sorted(missing)}"


def test_aggregate_phase_cells_are_real_phases_not_none():
    b = D0.bind_record(RECORD, PLAN_JSON)
    rows = D0.game_features(b, D0.discovery_task_ids(b)[0])
    keys = set(D0.aggregate(rows)["phase"])
    assert "None" not in keys and keys <= set(D0.PHASES), keys


def test_the_ast_control_rejects_an_injected_d1_import():
    """NEGATIVE CONTROL for the control. A check that has never rejected
    anything has not been shown to bind."""
    import ast
    import pathlib
    text = pathlib.Path(D0.__file__).read_text(encoding="utf-8")
    injected = text.replace(
        "from . import l0_match_plan as PLAN",
        "from . import l0_match_plan as PLAN\nfrom . import eval_loss_replay_analysis")
    assert injected != text, "injection anchor missing; the control proves nothing"
    imported = set()
    for node in ast.walk(ast.parse(injected)):
        if isinstance(node, ast.ImportFrom):
            imported |= {f"{node.module or ''}.{a.name}" for a in node.names}
    assert any("eval_loss" in m for m in imported), "control failed to catch the defect"


# ------------------------------------------------------------------- the run

def test_run_d0_covers_the_discovery_half_only_and_binds_identity():
    r = D0.run_d0(RECORD, PLAN_JSON)
    assert r["identity"]["task_digest"] == RULES.L0_TASK_DIGEST
    assert r["inventory"]["n_games"] == 64            # inventory sees both halves
    assert r["n_discovery_games"] == 32
    assert r["n_discovery_plies"] == 1329             # 1137 recorded + 6*32 opening
    assert {row["rep"] for row in r["rows"]} <= set(D0.DISCOVERY_REPS)


def test_run_d0_never_emits_a_confirmation_game_row():
    r = D0.run_d0(RECORD, PLAN_JSON)
    b = D0.bind_record(RECORD, PLAN_JSON)
    conf = {t for t, s in b.starts.items() if s["rep"] in D0.CONFIRMATION_REPS}
    assert conf and not ({row["task_id"] for row in r["rows"]} & conf)


def test_gate_refuses_a_signature_that_only_restates_the_ending():
    """4.1 forbids calling a move bad because the mover later lost. A signature
    whose every hit sits at ONE fixed distance from the terminal ply is a
    restatement of the ending, not a structural pattern -- and the real match
    contains exactly such a candidate: all 32 `ignored_threat` events are the
    penultimate ply of the 32 games."""
    sig = {"name": "s", "column": "ignored_threat", "colour_scope": "both",
           "d1_observable": "root visit share of the answering move"}
    rows = [{"opening": o, "colour_arm": a, "ignored_threat": True,
             "plies_from_terminal": 1}
            for o in ("o1_center", "o2_offcenter") for a in ("t1j_red", "t1j_black")]
    v = D0.evaluate_gate([sig], rows)
    assert v["verdict"] == "NO_GO"
    assert any("terminal ply" in f for f in v["signatures"][0]["failed"])


def test_gate_accepts_a_signature_spread_across_distances_from_the_terminal():
    sig = {"name": "s", "column": "ignored_threat", "colour_scope": "both",
           "d1_observable": "root visit share"}
    rows = [{"opening": o, "colour_arm": a, "ignored_threat": True,
             "plies_from_terminal": d}
            for o in ("o1_center", "o2_offcenter") for a in ("t1j_red", "t1j_black")
            for d in (1, 7, 15)]
    assert D0.evaluate_gate([sig], rows)["verdict"] == "GO"


def test_mover_more_fragmented_is_threshold_free():
    """A comparative boolean, so D0 chooses no cutoff. Picking a threshold after
    seeing the data is what D2 exists to prevent."""
    st = TwixtState(pegs={(5, 10): "red", (20, 2): "red", (9, 3): "black"},
                    ply=3, to_move="red")
    row = D0.ply_features(st, (12, 12))
    assert row["components_red"] == 2 and row["components_black"] == 1
    assert row["mover_more_fragmented"] is True


def test_main_writes_the_evidence_package_and_exits_zero(tmp_path):
    out = tmp_path / "d0"
    rc = D0.main(["--record", RECORD, "--plan", PLAN_JSON, "--out", str(out)])
    assert rc == 0
    written = sorted(p.name for p in out.iterdir())
    assert written == ["01_identity.json", "02_inventory.json",
                       "03_aggregates.json", "04_gate.json", "05_by_system.json"]
    gate = json.loads((out / "04_gate.json").read_text())
    assert gate["verdict"] in ("GO", "NO_GO")


def test_main_refuses_to_overwrite_an_existing_evidence_file(tmp_path):
    """Evidence is create-only. A rerun must not silently replace a record."""
    out = tmp_path / "d0"
    out.mkdir()
    (out / "01_identity.json").write_text("{}")
    with pytest.raises(D0.D0Error, match="exists"):
        D0.main(["--record", RECORD, "--plan", PLAN_JSON, "--out", str(out)])


def test_aggregate_means_divide_by_the_values_that_actually_existed():
    """4.3: every aggregate retains ITS denominator. Summing only the non-None
    values while dividing by the row count understates every column that is ever
    None -- which min_goal*_distance_* is, before a colour has any peg."""
    rows = [{"opening": "o", "colour_arm": "a", "winner": "red", "phase": "early",
             "plies_from_terminal": 3, "x": 10},
            {"opening": "o", "colour_arm": "a", "winner": "red", "phase": "early",
             "plies_from_terminal": 2, "x": None}]
    cell = D0.aggregate(rows)["opening"]["o"]
    assert cell["n"] == 2
    assert cell["counts"]["x"] == 1, "the column's own denominator is not reported"
    assert cell["means"]["x"] == 10.0, f"mean divided by the wrong denominator: {cell['means']['x']}"


def test_d0_spawns_git_and_nothing_else(monkeypatch):
    """D0 starts no JVM, no model runtime, no engine -- only `git rev-parse`.

    The non-emptiness assertion is load-bearing: `programs <= {"git"}` is
    VACUOUSLY TRUE of an empty list, so a control that only checked the subset
    would pass even if identity() had stopped spawning anything at all.
    """
    import subprocess
    spawned = []
    real = subprocess.run

    def spy(args, *a, **k):
        spawned.append(args[0] if isinstance(args, (list, tuple)) else args)
        return real(args, *a, **k)

    monkeypatch.setattr(subprocess, "run", spy)
    b = D0.bind_record(RECORD, PLAN_JSON)
    ident = D0.identity(b)
    assert spawned, "no program was spawned at all -- the subset check below would be vacuous"
    assert set(spawned) == {"git"}, spawned
    assert ident["repo_commit"] and len(ident["repo_commit"]) == 40


def test_d0_imports_no_model_runtime():
    """A module that cannot import a runtime cannot load a checkpoint."""
    import ast
    import pathlib
    tree = ast.parse(pathlib.Path(D0.__file__).read_text(encoding="utf-8"))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0:
            mods.add((n.module or "").split(".")[0])
    banned = {"torch", "onnx", "onnxruntime", "mlx", "safetensors", "numpy"}
    assert not (mods & banned), sorted(mods & banned)


# ═════════════ the incumbent-vs-T1j mover cut: ONE shared definition ═════════
#
# `by_system` computed this inline, so D1's selection rule -- which must pick
# plies where OUR INCUMBENT is to move -- would have had to state it a second
# time. Two copies of "which engine moved" is one copy too many: the cut decides
# what D1 interrogates at all. It is extracted here and reused, not restated.

BY_SYSTEM_RECORDED = ("docs/superpowers/evidence/2026-08-27-t1j-d0-postmortem/"
                      "05_by_system.json")


@pytest.fixture(scope="module")
def discovery_rows():
    return D0.run_d0(RECORD, PLAN_JSON)["rows"]


def test_moved_by_reads_the_arm_suffix_as_t1js_colour():
    assert D0.moved_by("t1j_red", "red") == "t1j"
    assert D0.moved_by("t1j_red", "black") == "ours"
    assert D0.moved_by("t1j_black", "black") == "t1j"
    assert D0.moved_by("t1j_black", "red") == "ours"


def test_by_system_classification_is_unchanged_by_the_extraction(discovery_rows):
    """REGRESSION CONTROL against evidence written by the PRE-EXTRACTION code.

    05_by_system.json was produced by the inline expression on 2026-08-27. If the
    extracted helper classified even one ply differently, a hit count or a
    denominator here would move.
    """
    recorded = json.load(open(BY_SYSTEM_RECORDED, encoding="utf-8"))
    assert recorded, "the recorded evidence is empty; the comparison would be vacuous"
    assert D0.by_system(discovery_rows) == recorded


def test_every_discovery_ply_is_attributed_to_exactly_one_engine(discovery_rows):
    """Non-vacuity: the cut must actually split, not label everything 'ours'."""
    who = {D0.moved_by(r["colour_arm"], r["mover"]) for r in discovery_rows}
    assert who == {"ours", "t1j"}
