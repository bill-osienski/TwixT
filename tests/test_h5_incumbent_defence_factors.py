"""H5 incumbent defence -- controlled one-factor study (EXPLORATORY) -- card §1-§9 A1.

SYNTHETIC ONLY: no model is loaded and the real incumbent makes no decision. The search
is the REAL frozen MCTS path driven by the steered stub evaluator of the defence study's
tests; refutation and readout logic also run on hand-built trees and engine positions with
known answers. Messages fixed.
"""
import contextlib
import copy
import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import pytest

from scripts.GPU.alphazero import h4_runner as R4
from scripts.GPU.alphazero import h5_cap_completion as CC
from scripts.GPU.alphazero import h5_incumbent_defence as D
from scripts.GPU.alphazero import h5_incumbent_defence_factors as X
from scripts.GPU.alphazero import h5_incumbent_defence_factors_authorization as AUTH
from scripts.GPU.alphazero import twixtbot_g3_reference as G3
from scripts.GPU.alphazero.mcts import MCTSNode, encode_move as enc
from tests.test_h5_cap_completion import ONE, TWO
from tests.test_h5_incumbent_defence import BY_ID, CONFIG, FIX, IDENT, Steered, tree

ROOT = pathlib.Path(__file__).resolve().parents[1]
B_REPRO, B_SEED3 = X.Arm("B-repro", -1), X.Arm("B-seed", -3)
F02, S1600 = X.Arm("F", -1, fpu=-0.20), X.Arm("S", -1, budget=1600)


def build(arm, ev=None):
    return X.arm_builder(arm, CONFIG, ev or Steered(), IDENT)


@pytest.fixture(scope="module")
def stub_pins():
    """What B-repro must reproduce UNDER THE STUB: the capture-off fingerprints at seed -1."""
    pins = {}
    for p in FIX:
        r = D.decide(build(B_REPRO), CC.state_of(p["moves"]), capture=False)
        pins[p["id"]] = D.fingerprint(r["root"], r["counts"], r["root_value"])
    return pins


# ─────────────────────────────── the gate and the frozen names (§9 A1.6) ──────────────────────

def test_the_gate_is_false_as_published():
    assert AUTH.H5_INCUMBENT_DEFENCE_FACTORS_AUTHORIZED is False


@pytest.mark.parametrize("entry", ["run", "run_claimed"])
def test_every_public_entry_refuses_with_the_gate_shut_BEFORE_anything(entry, monkeypatch):
    for owner, name in ((D, "_frozen"), (D, "_identity"), (X, "_claim"), (CC, "_watch"),
                        (R4, "load_production_evaluator"), (os, "mkdir")):
        monkeypatch.setattr(owner, name, lambda *a, **k: pytest.fail("touched before the gate"))
    with pytest.raises(X.FactorUnauthorized):
        getattr(X, entry)()


def test_the_CLI_exits_5_and_takes_no_output_directory(tmp_path):
    r = subprocess.run([sys.executable, "-m", X.MODULE, "run"], cwd=ROOT, capture_output=True,
                       text=True)
    assert r.returncode == X.EXIT_UNAUTHORIZED == 5, r.stderr
    r = subprocess.run([sys.executable, "-m", X.MODULE, "run", "--out-dir", str(tmp_path)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "--out-dir" in r.stderr, "an output directory was accepted"


def test_the_NAMES_and_the_PLAN_are_the_frozen_ones():
    assert X.MODULE == "scripts.GPU.alphazero.h5_incumbent_defence_factors"
    assert X.out_dir() == pathlib.Path(R4.EVIDENCE_ROOT) / \
        "t1j-h5-incumbent-defence-factors-EXPLORATORY"
    assert [a.label for a in X.plan()] == [
        "B-repro", "B-seed@seed-3", "B-seed@seed-5", "F@fpu-0.2", "F@fpu-0.5", "F@fpu-1.0",
        "S@sims1600", "S@sims6400"]
    assert len(X.plan()) * len(D.POSITION_IDS) * 2 == 80, "not the card's 80 decisions"
    assert X.PREDICTED_SWEEP == {-0.20: {"E1": True, "E2": True},
                                 -0.50: {"E1": False, "E2": True},
                                 -1.0: {"E1": False, "E2": False}}


def test_the_PINNED_FINGERPRINTS_are_the_committed_912b601f_ones():
    rec = json.loads((ROOT / "docs/superpowers/evidence/2026-10-02-t1j-h5-incumbent-defence-"
                             "EXPLORATORY/record.json").read_text())
    assert {r["position"]: r["fingerprint"] for r in rec["rows"]} == X.PINNED_FINGERPRINTS


def test_seed_minus_1_TASKS_are_the_defence_study_s_exactly():
    assert X.tasks_for_seed(IDENT, -1) == D.tasks_for(IDENT)
    assert {c: t["seed"] for c, t in X.tasks_for_seed(IDENT, -5).items()} == \
        {"red": -5, "black": -5}


# ─────────────────────────────── arm construction and the config diff (§9 A1.3) ───────────────

def test_each_arm_differs_from_the_frozen_config_in_EXACTLY_its_named_fields():
    state = CC.state_of(BY_ID["E1"]["moves"])
    for arm in X.plan():
        a = build(arm)(state.to_move, False)
        assert X.config_diff(a, CONFIG) == arm.allowed_diff(), arm.label
    s = build(X.Arm("S", -1, budget=6400))(state.to_move, False)
    assert s.config.mcts_sims == s.mcts.config.n_simulations == 6400


def test_a_SECOND_FACTOR_change_is_refused(monkeypatch):
    real = X.arm_builder

    def leaky(arm, config, evaluator, identity):
        b = real(arm, config, evaluator, identity)

        def build_(colour, capture):
            agent = b(colour, capture)
            agent.mcts.config = dataclasses.replace(agent.mcts.config, c_puct=2.0)
            return agent
        return build_
    with pytest.raises(X.FactorStop, match="S@sims1600 E1: check 'configuration \\(off\\)'"):
        X.observe(S1600, leaky(S1600, CONFIG, Steered(), IDENT), CONFIG, BY_ID["E1"])


def test_a_factor_that_was_NOT_APPLIED_is_refused():
    with pytest.raises(X.FactorStop, match="check 'configuration \\(off\\)'"):
        X.observe(S1600, build(B_REPRO), CONFIG, BY_ID["E1"])   # a baseline agent, called S


def test_an_S_arm_whose_two_budget_fields_disagree_is_refused_by_the_agent_s_own_guard():
    """Caught one layer EARLIER than this module: the unchanged agent's `validate_ply`
    compares the EvalConfig budget with the root visits before any check of ours runs."""
    from scripts.GPU.alphazero.eval_integrity import IntegrityError

    def half(colour, capture):
        agent = build(S1600)(colour, capture)
        agent.mcts.config = dataclasses.replace(agent.mcts.config, n_simulations=400)
        return agent
    with pytest.raises(IntegrityError, match="simulation budget mismatch"):
        X.observe(S1600, half, CONFIG, BY_ID["E1"])


def test_an_F_arm_running_ANOTHER_LEVEL_than_its_declared_one_is_refused():
    other = X.arm_builder(X.Arm("F", -1, fpu=-0.50), CONFIG, Steered(), IDENT)
    with pytest.raises(X.FactorStop, match="F@fpu-0.2 CD1: check 'factor value \\(off\\)'"):
        X.observe(F02, other, CONFIG, BY_ID["CD1"])


def test_the_WRONG_SEED_is_refused():
    wrong = X.arm_builder(X.Arm("B-seed", -5), CONFIG, Steered(), IDENT)
    with pytest.raises(X.FactorStop, match="check 'seed and colour \\(off\\)'"):
        X.observe(B_SEED3, wrong, CONFIG, BY_ID["E1"])


def test_a_root_budget_other_than_the_declared_one_is_refused(monkeypatch):
    real = D.decide

    def short(*a, **k):
        r = real(*a, **k)
        r["root"] = copy.copy(r["root"])
        r["root"].visit_count -= 1
        return r
    monkeypatch.setattr(D, "decide", short)
    with pytest.raises(X.FactorStop, match="check 'budget \\(off\\)'"):
        X.observe(B_SEED3, build(B_SEED3), CONFIG, BY_ID["E1"])


def test_the_S_ARM_runs_its_declared_budget():
    row = X.observe(S1600, build(S1600), CONFIG, BY_ID["CD1"])
    assert row["budget"] == 1600 and "budget (on)" in row["checks"]


def test_the_F_ARM_runs_with_its_FPU_and_no_move_requirement():
    row = X.observe(F02, build(F02), CONFIG, {**BY_ID["CD1"], "recorded": (0, 1)})
    assert row["fpu"] == -0.20 and "factor value (on)" in row["checks"]
    assert "B-repro move" not in row["checks"]


# ─────────────────────────────── B-repro vs B-seed (§9 A1.3) ───────────────────────────────

def test_B_REPRO_STOPS_on_any_fingerprint_other_than_the_pinned_one():
    with pytest.raises(X.FactorStop, match="B-repro E1: check 'B-repro fingerprint'"):
        X.observe(B_REPRO, build(B_REPRO), CONFIG, BY_ID["E1"])   # the stub is not the model


def test_B_REPRO_passes_on_its_pin_and_STOPS_on_another_move(monkeypatch, stub_pins):
    monkeypatch.setattr(X, "PINNED_FINGERPRINTS", stub_pins)
    row = X.observe(B_REPRO, build(B_REPRO), CONFIG, BY_ID["E1"])
    assert {"B-repro move", "B-repro fingerprint"} <= set(row["checks"])
    with pytest.raises(X.FactorStop, match="check 'B-repro move'"):
        X.observe(B_REPRO, build(B_REPRO), CONFIG, {**BY_ID["E1"], "recorded": (16, 9)})


def test_B_SEED_has_NO_historical_move_requirement():
    row = X.observe(B_SEED3, build(B_SEED3), CONFIG, {**BY_ID["E1"], "recorded": (16, 9)})
    assert row["seed"] == -3 and "B-repro move" not in row["checks"]


def test_OFF_ON_equivalence_still_binds_every_arm(monkeypatch):
    real = G3.SeededReferenceAgent.__call__

    def drawing(self, state):
        move = real(self, state)
        if self.capture:
            self.readout_rng.random()
        return move
    monkeypatch.setattr(G3.SeededReferenceAgent, "__call__", drawing)
    with pytest.raises(X.FactorStop, match="B-seed@seed-3 E1: check 'readout stream'"):
        X.observe(B_SEED3, build(B_SEED3), CONFIG, BY_ID["E1"])


def test_a_READER_that_changes_the_tree_is_refused_in_every_arm(monkeypatch):
    real = D.read

    def touching(root, *a):
        root.nn_value += 1e-9
        return real(root, *a)
    monkeypatch.setattr(D, "read", touching)
    with pytest.raises(X.FactorStop, match="check 'reader left the tree unchanged \\(off\\)'"):
        X.observe(B_SEED3, build(B_SEED3), CONFIG, BY_ID["E1"])


# ─────────────────────────────── refutation after the returned move (§9 A1.4) ─────────────────

def test_REFUTATION_TERMINAL_when_the_returned_move_wins():
    r = X.refutation_after(tree({(12, 4): 1.0}, {}), CC.state_of(ONE), (12, 4))
    assert (r["status"], r["winner"], r["refutations"]) == ("TERMINAL", "red", [])


def test_REFUTATION_NEUTRALIZED_when_the_returned_move_defends():
    state = CC.state_of(ONE[:23])                # black to move, red threatens (12,4)
    assert CC.threats_against(state) == [(12, 4)]
    r = X.refutation_after(tree({(12, 4): 1.0}, {}), state, (12, 4))
    assert (r["status"], r["refutation_set"]) == ("NEUTRALIZED", [])


def test_REFUTATION_set_is_RECOMPUTED_and_may_differ_from_the_frozen_W():
    state = CC.state_of(TWO[:23])                # black to move, red has TWO winning holes
    assert CC.threats_against(state) == [(12, 2), (12, 4)]
    root = tree({(12, 2): 0.5, (3, 3): 0.5}, {(12, 2): (5, 1.0)}, expanded=[(12, 2)])
    r = X.refutation_after(root, state, (12, 2))
    assert (r["status"], r["refutation_set"]) == ("OPEN", [[12, 4]]), \
        "the refutation set was not recomputed after the returned move"
    assert [x["category"] for x in r["refutations"]] == ["R1"]
    root.children[enc(12, 2)].children[enc(12, 4)] = MCTSNode(state=None, visit_count=3)
    assert X.refutation_after(root, state, (12, 2))["refutations"][0]["category"] == "R2"


def test_REFUTATION_R0_when_the_returned_move_was_never_expanded():
    state = CC.state_of(ONE[:23])
    r = X.refutation_after(tree({(2, 21): 1.0}, {(2, 21): (1, 0.0)}), state, (2, 21))
    assert (r["status"], [x["category"] for x in r["refutations"]]) == ("OPEN", ["R0"])


# ─────────────────────────────── the sweep and the readout variants ───────────────────────────

def test_the_FULL_SWEEP_definition():
    one = tree({(1, 1): 0.5, (2, 2): 0.5}, {(1, 1): (1, -0.5), (2, 2): (1, -0.2)})
    assert X.sweep(one, {(1, 1): 1, (2, 2): 1, (3, 3): 0})["full_sweep"]
    two = tree({(1, 1): 0.5, (2, 2): 0.5}, {(1, 1): (2, -0.5), (2, 2): (1, -0.2)})
    s = X.sweep(two, {(1, 1): 2, (2, 2): 1})
    assert not s["full_sweep"] and (s["max_visits"], s["histogram"]) == (2, {"1": 1, "2": 1})
    tied2 = tree({(1, 1): 0.5, (2, 2): 0.5}, {(1, 1): (2, -0.5), (2, 2): (2, -0.2)})
    assert not X.sweep(tied2, {(1, 1): 2, (2, 2): 2})["full_sweep"], \
        "an all-tied tree at 2 visits is not a one-visit sweep"


def test_the_READOUT_VARIANTS_break_the_tie_by_prior_and_by_root_mover_value():
    t = tree({(1, 1): 0.1, (2, 2): 0.6, (3, 3): 0.3},
             {(1, 1): (1, -0.9), (2, 2): (1, 0.4), (3, 3): (1, 0.1)})
    rv = X.readout_variants(t, {(1, 1): 1, (2, 2): 1, (3, 3): 1}, [(1, 1)])
    assert rv["R-prior"] == {"move": [2, 2], "defends": False}
    assert rv["R-value"] == {"move": [1, 1], "defends": True}, \
        "value must be the ROOT mover's (q_node negated)"


# ─────────────────────────────── the six-cell rule, both directions (§9 A1.1) ─────────────────

def rows_for(sweeps=None, defends=None):
    """Minimal COMPLETE-run rows: every arm at every fixture. `sweeps[(label, p)]` and
    `defends[(label, p)]` override the clean positive baseline."""
    sweeps, defends = sweeps or {}, defends or {}
    rows = []
    for arm in X.plan():
        for p in D.POSITION_IDS:
            full = True
            if arm.name == "F":
                full = X.PREDICTED_SWEEP[arm.fpu].get(p, False)
            full = sweeps.get((arm.label, p), full)
            rows.append({"arm": arm.label, "position": p,
                         "defends": defends.get((arm.label, p), p.startswith("CD")),
                         "reading": {"sweep": {"full_sweep": full}}})
    return rows


def test_the_RULE_a_clean_positive_run_is_consistent():
    v = X.interpret(rows_for())
    assert v["verdict"] == "consistent with the H-FPU-SWEEP conjecture at E1 and E2", v
    assert v["problems"] == [] and len(v["cells"]) == 6
    assert all(c["match"] for c in v["cells"].values())


@pytest.mark.parametrize("override,problem", [
    ({("F@fpu-0.2", "E1"): False}, "cell fpu-0.2 E1: predicted persists, observed breaks"),
    ({("F@fpu-0.5", "E1"): True}, "cell fpu-0.5 E1: predicted breaks, observed persists"),
    ({("F@fpu-1.0", "E2"): True}, "cell fpu-1.0 E2: predicted breaks, observed persists"),
    ({("B-seed@seed-5", "E2"): False}, "B-seed -5 E2: no full sweep"),
])
def test_the_RULE_any_single_failure_is_not_supported(override, problem):
    v = X.interpret(rows_for(sweeps=override))
    assert v["verdict"] == "not supported as stated" and problem in v["problems"], v["problems"]


def test_the_RULE_does_not_ask_CD1_to_sweep():
    v = X.interpret(rows_for(sweeps={("B-seed@seed-3", "CD1"): False, ("F@fpu-1.0", "CD1"): True}))
    assert v["verdict"].startswith("consistent"), "CD1 is a contrast, not a sweep fixture"


def test_the_RULE_needs_B_REPRO():
    rows = [r for r in rows_for() if not (r["arm"] == "B-repro" and r["position"] == "E3")]
    assert "B-repro incomplete" in X.interpret(rows)["problems"]


def test_REPAIR_is_per_fixture_and_names_control_harm():
    v = X.interpret(rows_for(defends={("F@fpu-1.0", "E1"): True, ("F@fpu-1.0", "CD1"): False}))
    r = v["repair"]["F@fpu-1.0"]
    assert (r["repairs"], r["control_harm"], r["note"]) == \
        (["E1"], ["CD1"], "repair with control harm")
    assert v["repair"]["B-repro"]["repairs"] == [] and v["repair"]["B-repro"]["note"] is None


# ─────────────────────────────── the child's run (record on every exit) ───────────────────────

def claimed(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    d = tmp_path / X.OUT_NAME
    X._claim(d)
    return d


def log_steps(d):
    return [json.loads(line)["step"] for line in (d / "log.jsonl").read_text().splitlines()]


FAST = [B_REPRO, B_SEED3]


def test_RUN_a_COMPLETE_plan_is_recorded_and_interpreted(tmp_path, monkeypatch, stub_pins):
    monkeypatch.setattr(X, "PINNED_FINGERPRINTS", stub_pins)
    monkeypatch.setattr(X, "plan", lambda: FAST)
    d = claimed(tmp_path)
    rec = X._run(FIX, CONFIG, IDENT, Steered, d)
    assert (rec["status"], rec["failure"], rec["interpretable"]) == ("COMPLETE", None, True)
    assert len(rec["rows"]) == 10 and log_steps(d) == ["model_load"] + ["observation"] * 10
    assert rec["interpretation"]["verdict"] == "not supported as stated", \
        "a two-arm plan has no F cells; it must not read as consistent"
    assert all(r["reading"]["readout_variants"] for r in rec["rows"] if r["arm"] == "B-repro")


def test_RUN_a_FAILURE_keeps_every_completed_observation(tmp_path, monkeypatch, stub_pins):
    monkeypatch.setattr(X, "PINNED_FINGERPRINTS", stub_pins)
    bad, real = X.Arm("S", -1, budget=1600), X.arm_builder

    def leaky_S(arm, *a):
        b = real(arm, *a)
        if arm.name != "S":
            return b
        return lambda colour, capture: _with_extra_field(b(colour, capture))
    monkeypatch.setattr(X, "arm_builder", leaky_S)
    d = claimed(tmp_path)
    rec = X._run(FIX, CONFIG, IDENT, Steered, d, arms=[B_REPRO, bad])
    f = rec["failure"]
    assert (rec["status"], f["failing_step"], f["exception_type"]) == \
        ("STOP", "observe:S@sims1600:E1", "FactorStop")
    assert len(rec["rows"]) == 5 and [r["arm"] for r in rec["rows"]] == ["B-repro"] * 5
    assert log_steps(d) == ["model_load"] + ["observation"] * 5, "completed observations lost"
    assert rec["interpretation"] is None and rec["interpretation_note"]


def _with_extra_field(agent):
    agent.mcts.config = dataclasses.replace(agent.mcts.config, c_puct=1.6)
    return agent


def test_RUN_B_REPRO_failure_STOPS_before_every_other_arm(tmp_path):
    d = claimed(tmp_path)
    rec = X._run(FIX, CONFIG, IDENT, Steered, d)          # real pins: the stub cannot match
    assert (rec["status"], rec["failure"]["failing_step"]) == ("STOP", "observe:B-repro:E1")
    assert rec["rows"] == [] and "B-repro fingerprint" in rec["failure"]["reason"]


def test_RUN_an_INTERRUPTION_is_recorded_and_re_raised(tmp_path, monkeypatch, stub_pins):
    monkeypatch.setattr(X, "PINNED_FINGERPRINTS", stub_pins)
    d = claimed(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        X._run(FIX, CONFIG, IDENT, lambda: Steered(fail_on_call=3000), d, arms=FAST)
    rec = json.loads((d / "record.json").read_text())
    assert rec["status"] == "INTERRUPTED" and rec["failure"]["failing_step"].startswith("observe:")
    assert log_steps(d) == ["model_load"] + ["observation"] * len(rec["rows"])


def test_RUN_an_INEXACT_plan_is_INCOMPLETE_naming_what_is_missing(tmp_path, monkeypatch,
                                                                  stub_pins):
    monkeypatch.setattr(X, "PINNED_FINGERPRINTS", stub_pins)
    monkeypatch.setattr(X, "plan", lambda: FAST)
    rec = X._run(FIX, CONFIG, IDENT, Steered, claimed(tmp_path), arms=[B_REPRO])
    f = rec["failure"]
    assert (rec["status"], f["failing_step"], f["exception_type"]) == \
        ("INCOMPLETE", "completion_check", None)
    assert f["missing"] == [["B-seed@seed-3", p] for p in sorted(D.POSITION_IDS)]
    assert rec["interpretation"] is None


def test_RUN_a_MODEL_LOAD_failure_and_the_DEADLINE_are_STOPs(tmp_path):
    def loader():
        raise FileNotFoundError("checkpoint not found")
    rec = X._run(FIX, CONFIG, IDENT, loader, claimed(tmp_path / "a"))
    assert (rec["status"], rec["failure"]["failing_step"]) == ("STOP", "model_load")
    t = [0.0]

    def slow():
        t[0] += 10_000.0
        return Steered()
    rec = X._run(FIX, CONFIG, IDENT, slow, claimed(tmp_path / "b"), clock=lambda: t[0])
    assert (rec["status"], rec["failure"]["failing_step"]) == ("STOP", "observe:B-repro:E1")
    assert "deadline passed" in rec["failure"]["reason"]


def test_the_CHILD_takes_only_an_UNCONSUMED_claim_with_ITS_nonce(tmp_path, monkeypatch):
    d = tmp_path / X.OUT_NAME
    claim, nonce = X._claim(d)
    monkeypatch.setenv(X.CLAIM_ENV, "0" * 32)
    with pytest.raises(X.FactorRefused, match="nonce does not match"):
        X._accept_claim(d)
    monkeypatch.setenv(X.CLAIM_ENV, nonce)
    for name in X.CHILD_STREAMS.values():
        (d / name).write_text("")
    assert X._accept_claim(d) == claim
    (d / "log.jsonl").write_text("")
    with pytest.raises(X.FactorRefused, match="unconsumed claim"):
        X._accept_claim(d)


# ─────────────────────────────── the public entry, real state ───────────────────────────────

@pytest.fixture
def open_gate(monkeypatch, tmp_path, stub_pins):
    monkeypatch.setattr(AUTH, "H5_INCUMBENT_DEFENCE_FACTORS_AUTHORIZED", True)
    monkeypatch.setattr(R4, "EVIDENCE_ROOT", str(tmp_path))
    monkeypatch.setattr(R4, "load_production_evaluator", lambda identity: Steered())
    monkeypatch.setattr(X, "PINNED_FINGERPRINTS", stub_pins)
    return tmp_path


def inproc(monkeypatch):
    def watch(argv, *, limit_s, env, cwd, stdout_path, stderr_path):
        assert list(argv[1:]) == ["-m", X.MODULE, "run-claimed"], argv
        monkeypatch.setenv(X.CLAIM_ENV, env[X.CLAIM_ENV])
        t0 = time.monotonic()
        with open(stdout_path, "x") as so, open(stderr_path, "x") as se, \
                contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
            rc = X.main(list(argv[3:]))
        return {"child_pid": os.getpid(), "limit_s": limit_s, "terminated": False,
                "cause": None, "signal": None, "rc": rc, "elapsed_s": time.monotonic() - t0,
                "survivors_killed": False}
    monkeypatch.setattr(CC, "_watch", watch)


def test_PUBLIC_ENTRY_runs_in_the_real_state(open_gate, monkeypatch):
    monkeypatch.setattr(X, "plan", lambda: FAST)
    inproc(monkeypatch)
    res = X.run()
    out = X.out_dir()
    rec = json.loads((out / "record.json").read_text())
    assert (res["status"], res["record_status"], res["child_rc"]) == \
        ("COMPLETE", "COMPLETE", 0), rec["failure"]
    assert res == json.loads((out / X.WATCHDOG_FILE).read_text())
    assert res["journal"]["lines"] == 11 and rec["incumbent_identity"] == IDENT


def test_PUBLIC_ENTRY_refuses_an_occupied_destination_before_anything(open_gate, monkeypatch):
    X.out_dir().mkdir()
    monkeypatch.setattr(CC, "_watch", lambda *a, **k: pytest.fail("a child was started"))
    with pytest.raises(X.FactorRefused, match="occupied"):
        X.run()
    assert list(X.out_dir().iterdir()) == []


def test_PUBLIC_ENTRY_the_REAL_child_reads_the_CLOSED_on_disk_gate(open_gate):
    res = X.run()
    out = X.out_dir()
    assert (res["status"], res["child_exit"]) == ("EXITED_WITHOUT_RECORD", "exit status 5")
    assert "UNAUTHORIZED" in (out / X.CHILD_STREAMS["stderr"]).read_text()


HANG = r'''
import json, os, signal, sys, time
d = sys.argv[1]
open(os.path.join(d, "log.jsonl"), "x").write('{"step": "model_load"}\n')
json.dump({"status": "COMPLETE"}, open(os.path.join(d, "record.json"), "x"))
sys.stdout.write("OUT before\n")
signal.pthread_sigmask(signal.SIG_BLOCK, signal.valid_signals())
time.sleep(60)
'''


def test_PUBLIC_ENTRY_a_hung_child_is_TERMINATED_never_COMPLETE(open_gate, monkeypatch):
    monkeypatch.setattr(X, "_child_argv", lambda: [sys.executable, "-c", HANG, str(X.out_dir())])
    monkeypatch.setattr(X, "HARD_LIMIT_S", 3.0)
    t0 = time.monotonic()
    res = X.run()
    out = X.out_dir()
    assert time.monotonic() - t0 < 20, "the parent did not enforce its limit"
    assert (res["status"], res["record_status"], res["child_exit"]) == \
        ("TERMINATED", "COMPLETE", "killed by signal SIGKILL")
    assert (out / X.CHILD_STREAMS["stdout"]).read_text() == "OUT before\n"
    raw = (out / "log.jsonl").read_bytes()
    assert res["journal"] == {"file": "log.jsonl", "lines": 1,
                              "sha256": hashlib.sha256(raw).hexdigest()}
