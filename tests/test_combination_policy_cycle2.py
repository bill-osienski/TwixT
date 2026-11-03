"""THE SECOND POLICY-IMPROVEMENT CYCLE, BUILD A (amendment
`docs/superpowers/2026-11-02-combination-policy-second-cycle-amendment.md`, approved 2026-11-02): `run cycle2_generate`
and `run cycle2_train`, their closed gates, wrappers and generation seed and closing copies. CPU only: fake evaluators,
tiny random networks and temporary outputs. Nothing here loads the real model, runs a seed procedure on the real
repository, opens a real gate or plays a real game.

It reuses the first cycle's checks and fixtures. What is new is the teacher: the binding and probe before any worker
starts, the fit's start, the corpus's separation, and that the first cycle's evidence and kinds stay as they are."""
import difflib
import hashlib
import io
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time

import numpy as np
import pytest

from scripts.GPU.alphazero import combination_policy_cycle2_generate_authorization as GATE_GEN
from scripts.GPU.alphazero import combination_policy_cycle2_generate2_authorization as GATE_GEN2
from scripts.GPU.alphazero import combination_policy_cycle2_screen_authorization as GATE_CS
from scripts.GPU.alphazero import combination_policy_cycle2_strength_authorization as GATE_CST
from scripts.GPU.alphazero import combination_policy_cycle2_train_authorization as GATE_TRAIN
from scripts.GPU.alphazero import combination_policy_pilot as P
from scripts.GPU.alphazero import mirror_halving_combination as K
from scripts.GPU.alphazero import seed_band_2027 as SB
from tests import _combination_fakes as F
from tests.test_combination_policy_generate_launch import PINNED_FILES, _load, _split, _synthetic
from tests.test_combination_policy_launch import STUB_BG, STUB_IDLE, STUB_PILOT, _alive, _git
from tests.test_combination_policy_pilot import (GOOD_INNER, GOOD_OUTER, TINY_ARCH, GenHarness, _conclude, _recipe,
                                                 _tiny_network, _train_seams, corpus_game)  # noqa: F401

ROOT = pathlib.Path(__file__).resolve().parents[1]
EV = ROOT / "docs/superpowers/evidence"
AMEND = "docs/superpowers/2026-11-02-combination-policy-second-cycle-amendment.md"
AMEND2 = "docs/superpowers/2026-11-02-combination-policy-second-cycle-generation-attempt2-amendment.md"
GEN, GEN2, TRAIN = "cycle2_generate", "cycle2_generate2", "cycle2_train"       # GEN is spent: VOID, ledger entry 47
CS, CST = "cycle2_screen", "cycle2_strength"                                   # Build B
SURVEY_1, SURVEY = ("combination-policy-generate-seed-survey/01_seed_survey.py",
                    "combination-policy-cycle2-generate-seed-survey/01_seed_survey.py")
CLOSING_1, CLOSING = ("combination-policy-generate-seed-closing/01_derive_usage.py",
                      "combination-policy-cycle2-generate-seed-closing/01_derive_usage.py")
SURVEY2 = "combination-policy-cycle2-generate2-seed-survey/01_seed_survey.py"
CLOSING2 = "combination-policy-cycle2-generate2-seed-closing/01_derive_usage.py"
_REAL_GEN_SEAMS, _REAL_TRAIN_SEAMS = P._real_gen_seams, P._real_train_seams   # bound before the isolation replaces them


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """EVERY test (#74): the pilot's real seams, its record boundary and the real cache are unreachable, on the CPU."""
    def unreachable(*_a, **_k):
        raise AssertionError("a test reached a real seam or the real record boundary")
    for name in ("_real_gen_seams", "_real_train_seams", "_real_screen_seams", "conclude"):
        monkeypatch.setattr(P, name, unreachable)
    monkeypatch.setattr(P, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv(P.SETTING, P.SETTING_VALUE)
    import mlx.core as mx
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


# ─────────────────────────────── the frozen amendment, the build state, the first cycle ───────────────────────────────

def test_THE_AMENDMENT_IS_FROZEN_PINNED_AND_NAMES_THE_REAL_BINDINGS():
    rel, sha = P.CYCLE2_AMENDMENT
    assert rel == AMEND and P.M._sha((ROOT / rel).read_bytes()) == sha
    assert P.KINDS[GEN]["amendment"] == P.KINDS[GEN2]["base_amendment"] == P.KINDS[TRAIN]["base_amendment"] == \
        P.CYCLE2_AMENDMENT
    text = (ROOT / rel).read_text(encoding="utf-8")
    flat = " ".join(text.split())
    assert "FROZEN 2026-11-02" in text.splitlines()[0]
    assert P.M.CHECKPOINT_SHA1 in text and P.TRAINED_HEAD["sha256"] in text and P.TRAINED_HEAD["file"] in text
    assert "may change the candidate set at a recorded root" in flat
    assert "Neither a changed candidate set nor better targets is guaranteed" in flat
    assert "through the actual uncompiled generation evaluator" in flat


def test_THE_ONE_RUN_WENT_VOID_AT_THE_VERSION_CHECK_BEFORE_ANY_GAME_AND_THE_BLOCK_IS_CLOSED():
    """BUILT 2026-11-02 as the build-only state; RE-ANCHORED at the seed preparation (survey b657bee5, registration 8a24c602,
    post-proof 14869ca1). RE-ANCHORED 2026-11-02 (the run, 9ff326a7 -> 9449a1e1, and its closing): the survey is CLEAN and
    both proofs HOLD, each 55/55 on a clean tree and bound to the survey record; the plan is the copy's build_plan; the
    post-proof proved the registry the run launched with. The ONE run, launched from the one-line gate opening, went VOID
    at the version check (Python 3.14.10 against the card's 3.14.9): exit 3, gate restored, no survivors, no checkpoint
    read, no teacher installed, no game_start or claims event, no rows directory. It admits nothing. K1 RECONCILES, so the
    WHOLE block is EXPOSED (no claims event: the rule cannot establish k) and RETIRED, each exactly once and last, every
    seed refused; `run cycle2_generate` stays one-shot. No block literal here: the proofs scan tests/."""
    from scripts.GPU.alphazero import e4_screen_reference as REG
    here = (EV / SURVEY).parent
    assert sorted(x.name for x in here.iterdir() if x.name != "__pycache__") == \
        ["01_seed_survey.py", "02_seed_survey_run.txt", "03_collision_proof_pre_run.txt",
         "04_collision_proof_post_run.txt"]
    raw = (here / "02_seed_survey_run.txt").read_bytes()
    survey, sha = raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()
    assert survey.splitlines()[2].endswith("; working tree clean") and "controls passed: 55/55" in survey
    assert survey.endswith("SURVEY VERDICT   : THE COMBINATION-POLICY CYCLE-2 GENERATION BLOCK IS CLEAN\n")
    for rel, mode in (("03_collision_proof_pre_run.txt", "pre"), ("04_collision_proof_post_run.txt", "post")):
        text = (here / rel).read_text(encoding="utf-8")
        assert text.splitlines()[2].endswith("; working tree clean")
        assert f"02_seed_survey_run.txt sha256 {sha} (read at " in text
        assert f"EXPECTED STATE ({mode}): HOLDS" in text and "controls passed: 55/55" in text
    post = (here / "04_collision_proof_post_run.txt").read_text(encoding="utf-8")
    plan = json.loads((ROOT / P.KINDS[GEN]["plan"]).read_text(encoding="utf-8"))
    n = P.KINDS[GEN]["slots"]
    assert type(plan["base"]) is int and plan["n_slots"] == n == 1000
    mod = _load(SURVEY, "_c2_survey_registered")
    assert plan == mod.build_plan(plan["base"], sha)
    block = (plan["base"], plan["base"] + n)
    for mode in ("survey", "pre", "post"):
        with pytest.raises(mod.Refused, match="each mode runs once"):
            mod.main(["x", "--mode", mode])
    # the run
    run_dir, evidence = _path(GEN).parent, ROOT / P.KINDS[GEN]["evidence"]
    assert sorted(x.name for x in run_dir.iterdir()) == ["01_wrapper.sh", "bg_check.txt", "idle_check.txt", "outer.json",
                                                         "pgid.txt", "plan.json", "run.stderr.txt", "run.stdout.txt"]
    assert sorted(x.name for x in evidence.iterdir()) == ["progress.jsonl", "record.json"]
    outer = json.loads((run_dir / "outer.json").read_text(encoding="utf-8"))
    rec = json.loads((evidence / "record.json").read_text(encoding="utf-8"))
    assert (outer["outcome"], outer["inner_exit"], outer["outer_exit"]) == ("VOID", 3, 3)
    assert outer["gate_restored"] is True and outer["survivors"] == "" and outer["record_by"] == "script"
    assert outer["term_s"] == 21590 and outer["grace_s"] == 10 and outer["head"] == rec["head"]
    assert (rec["status"], rec["run"], rec["verdict"], rec["void"]["step"]) == ("VOID", GEN, None, "versions")
    assert rec["versions"] == {**P.VERSIONS, "python": "3.14.10"} and P.VERSIONS["python"] == "3.14.9"
    assert rec["amendment_sha256"] == P.CYCLE2_AMENDMENT[1]
    for key in ("checkpoint_sha1", "teacher", "teacher_probe", "teacher_digests", "seeds", "telemetry", "background_checks"):
        assert key not in rec, key                           # the check stopped it before any of these
    events = [json.loads(x)["event"] for x in (evidence / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events == ["start", "end"]
    assert not (pathlib.Path.home() / "twixt-cache/combination-policy-pilot" / GEN).exists()
    gate_rel = "scripts/GPU/alphazero/combination_policy_cycle2_generate_authorization.py"
    assert subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"{outer['head']}^"], capture_output=True, text=True,
                          check=True).stdout.strip() == outer["pre_open"]
    opened = subprocess.run(["git", "-C", str(ROOT), "diff", "--unified=0", outer["pre_open"], outer["head"]],
                            capture_output=True, text=True, check=True).stdout
    assert [ln for ln in opened.splitlines() if ln[:1] in "+-" and ln[:3] not in ("+++", "---")] == \
        [f"-{WRAPPERS[GEN]['gate']} = False", f"+{WRAPPERS[GEN]['gate']} = True"] and f"b/{gate_rel}" in opened
    reg_rel = "scripts/GPU/alphazero/e4_screen_reference.py"
    launched = subprocess.run(["git", "-C", str(ROOT), "show", f"{outer['head']}:{reg_rel}"], capture_output=True,
                              check=True).stdout
    assert f"\nregistry {reg_rel} sha256 {hashlib.sha256(launched).hexdigest()}\n" in post     # proved = launched
    assert subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "9449a1e1", "HEAD", "--", str(run_dir),
                           str(evidence)]).returncode == 0
    # the closing
    closing = (EV / CLOSING).parent
    assert sorted(x.name for x in closing.iterdir() if x.name != "__pycache__") == \
        ["01_derive_usage.py", "02_derive_usage_run.txt"]
    k1 = (closing / "02_derive_usage_run.txt").read_text(encoding="utf-8")
    assert k1.splitlines()[2].endswith("; working tree clean") and "controls passed: 21/21" in k1
    assert "game starts (slots, as logged): []\nclaims events: []\n" in k1
    assert k1.endswith(f"VERDICT: RECONCILES -- EXPOSE {list(block)}; RETIRE {list(block)}\n")
    for intervals in (REG.ACCOUNTED_SEED_INTERVALS, REG.EXPOSED_SEED_INTERVALS, REG.RETIRED_SEED_INTERVALS):
        # RE-ANCHORED 2026-11-02 (the new attempt's registration 4c599111): no longer last in ACCOUNTED; later entries lie above
        i = intervals.index(block)
        assert intervals.count(block) == 1 and all(lo >= block[1] for lo, _hi in intervals[i + 1:])
        assert all(hi <= block[0] for lo, hi in intervals[:i] if hi > SB.BAND[0])
    assert all(REG.seed_is_accounted(s) and REG.seed_is_exposed(s) and REG.seed_is_retired(s) and
               REG.seed_is_unavailable(s) and not REG.seed_is_test_only(s) for s in range(*block))
    # the gates, the rest of the state
    assert GATE_GEN.COMBINATION_POLICY_CYCLE2_GENERATE_AUTHORIZED is False
    assert GATE_TRAIN.COMBINATION_POLICY_CYCLE2_TRAIN_AUTHORIZED is False
    assert (P.KINDS[GEN]["gate"][0], P.KINDS[TRAIN]["gate"][0]) == (GATE_GEN, GATE_TRAIN)
    assert not any(k.startswith("cycle2") and k not in (GEN, GEN2, TRAIN, CS, CST) for k in P.KINDS)
    # RE-ANCHORED 2026-11-02 (the fit, cafc78b4 -> 200e7eee): the fit's own state test holds its evidence


def test_THE_VOID_RUN_ADMITS_NOTHING_AND_ITS_KIND_STAYS_ONE_SHOT(tmp_path, monkeypatch, capsys):
    """The real committed readers: the fit is refused on generation's VOID record. With the gate open in this process only
    and admission stubbed, `run cycle2_generate` refuses as one-shot before any seam."""
    spent = P.KINDS[GEN]["evidence"]
    assert P.KINDS[TRAIN]["corpus"] == GEN2 and all(rel != spent for rel, *_ in P.ADMISSION[TRAIN])
    assert all(rel != spent for _run, rel in P.OUTER_ADMISSION[TRAIN])
    gate, name = P.KINDS[GEN]["gate"]
    monkeypatch.setattr(gate, name, True)
    monkeypatch.setattr(P, "_mlx_loaded", lambda: False)
    monkeypatch.setattr(P, "admitted", lambda k, *_a: {"stub": k})
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"not a model")
    assert P.main(["run", GEN, "--checkpoint", str(ckpt)]) == P.EXIT_REFUSED
    assert "the run is one-shot" in capsys.readouterr().err


def test_THE_FIRST_CYCLE_S_EVIDENCE_AND_HEAD_ARE_UNCHANGED():
    dirs = [f"docs/superpowers/evidence/combination-policy-{k}{s}"
            for k in ("generate", "train", "screen", "strength")
            for s in ("", "-run", "-seed-survey", "-seed-closing")]
    assert subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "0557dff1", "HEAD", "--", *dirs]).returncode == 0
    raw = (ROOT / P.TRAINED_HEAD["file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == P.TRAINED_HEAD["sha256"]
    assert subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", P.CARD]).returncode == 0
    assert P.M._sha((ROOT / P.CARD).read_bytes()) == P.CARD_SHA256


@pytest.mark.parametrize("kind", ["generate", "train"])
def test_THE_FIRST_CYCLE_S_KINDS_STAY_ONE_SHOT_even_with_their_gates_open(tmp_path, monkeypatch, capsys, kind):
    gate, name = P.KINDS[kind]["gate"]
    monkeypatch.setattr(gate, name, True)
    monkeypatch.setattr(P, "_mlx_loaded", lambda: False)
    monkeypatch.setattr(P, "admitted", lambda k, *_a: {"stub": k})
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"not a model")
    assert P.main(["run", kind, "--checkpoint", str(ckpt)]) == P.EXIT_REFUSED
    assert "the run is one-shot" in capsys.readouterr().err


def test_ONLY_THE_TEACHER_AND_THE_CORPUS_DIFFER_the_settings_recipe_and_block_size_are_the_first_cycle_s():
    one, two = P.KINDS["generate"], P.KINDS[GEN]
    assert (two["slots"], two["keep_rows"]) == (one["slots"], one["keep_rows"]) == (1000, True)
    assert (P.N_WORKERS, P.MAX_BATCH_ROWS) == (18, 14)
    assert P.RECIPE == {"updates": 3000, "batch": 64, "lr": 1e-3, "l2": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
                        "bias_correction": False, "init_seed": 20261028, "order_seed": 20261029, "min_kl": 0.01,
                        "heldout_mod": 10, "heldout_rem": 9, "from_ply": 20,
                        "arch": {"in_channels": 30, "hidden": 128, "n_blocks": 6}}
    paths = {P.KINDS[k][f] for k in P.KINDS for f in ("evidence", "plan") if f in P.KINDS[k]}
    assert len(paths) == sum(f in P.KINDS[k] for k in P.KINDS for f in ("evidence", "plan"))   # nothing shared


# ─────────────────────────────── generation: the binding and the teacher-reach probe ───────────────────────────────

DIGESTS = {"encoder.": "e", "value_head.": "v", "policy_head.": "teacher"}
BINDING = {"file": "first/head.npz", "sha256": "a" * 64, "digests": dict(DIGESTS), "bytes": b"x"}


def _fake(seed):
    ev = F.FakeEvaluator(seed)
    ev.network = object()
    return ev


def _gen_harness(tmp_path, monkeypatch, calls, kind=GEN, **change):
    monkeypatch.setitem(P.KINDS[kind], "slots", 3)
    h = GenHarness(tmp_path, kind=kind, n=3)
    generate = h.generate

    def counted(**kw):
        calls.append("generate")
        return generate(**kw)
    s = h.seams
    s.generate, s.load_evaluator, s.teacher = counted, lambda: _fake(7), BINDING
    s.digests = lambda net: {**DIGESTS, "policy_head.": "parent"}
    s.load_parent = lambda: _fake(8)

    def install(net):
        calls.append("install")
        return dict(DIGESTS)
    s.install_teacher = install
    for k, v in change.items():
        setattr(s, k, v)
    return h


def _run_gen(tmp_path, h):
    return _conclude(GEN, lambda o, r, lg: P._generate(GEN, h.seams, o, r, lg), tmp_path / "ev")


def test_GENERATION_BINDS_AND_PROBES_THE_TEACHER_BEFORE_ANY_WORKER_and_writes_its_own_corpus(tmp_path, monkeypatch):
    calls = []
    h = _gen_harness(tmp_path, monkeypatch, calls)
    code, rec = _run_gen(tmp_path, h)
    assert code == P.EXIT_COMPLETED, rec.get("void")
    assert calls == ["install", "generate"]
    assert rec["teacher"] == {k: BINDING[k] for k in ("file", "sha256", "digests")} and "bytes" not in rec["teacher"]
    assert rec["teacher_digests"] == DIGESTS and rec["teacher_probe"]["max_prior_abs_diff"] > 0.0
    assert (rec["amendment"], rec["amendment_sha256"]) == P.CYCLE2_AMENDMENT
    games = [json.loads(x) for x in (tmp_path / "ev" / "games.jsonl").read_text().splitlines()]
    assert all(P._sha_file(tmp_path / "cache" / GEN / g["rows_file"]) == g["rows_sha256"] for g in games)
    assert not (tmp_path / "cache" / "generate").exists()


@pytest.mark.parametrize("what, change", [
    ("the served network's digests", {"install_teacher": lambda net: {**DIGESTS, "policy_head.": "other"}}),
    ("the parent's encoder", {"digests": lambda net: {**DIGESTS, "encoder.": "other", "policy_head.": "parent"}}),
    ("the parent's value head", {"digests": lambda net: {**DIGESTS, "value_head.": "other", "policy_head.": "parent"}}),
    ("a parent that is the teacher", {"digests": lambda net: dict(DIGESTS)}),
    ("equal opening priors", {"load_parent": lambda: _fake(7)})],
    ids=["digests", "encoder", "value-head", "parent-is-teacher", "equal-priors"])
def test_A_FAILED_BINDING_OR_PROBE_IS_A_VOID_BEFORE_ANY_WORKER_OR_ROW(tmp_path, monkeypatch, what, change):
    calls = []
    h = _gen_harness(tmp_path, monkeypatch, calls, **change)
    code, rec = _run_gen(tmp_path, h)
    assert code == P.EXIT_VOID and rec["status"] == "VOID" and rec["void"]["step"] == "teacher", (what, rec.get("void"))
    assert "generate" not in calls and not (tmp_path / "cache").exists()


def test_A_PROBE_PARENT_SHARING_THE_SERVED_EVALUATOR_OR_NETWORK_IS_A_VOID(tmp_path, monkeypatch):
    calls = []
    served = _fake(7)
    other = _fake(8)
    other.network = served.network
    for parent in (served, other):
        h = _gen_harness(tmp_path, monkeypatch, calls, load_evaluator=lambda: served, load_parent=lambda p=parent: p)
        code, rec = _conclude(GEN, lambda o, r, lg: P._generate(GEN, h.seams, o, r, lg),
                              tmp_path / f"ev{id(parent)}")
        assert rec["void"]["step"] == "teacher" and "shares" in rec["void"]["detail"]
    assert "generate" not in calls


@pytest.mark.parametrize("kind, teacher", [(GEN, None), ("generate", BINDING)], ids=["no-teacher", "first-cycle"])
def test_A_KIND_AND_ITS_SEAMS_MUST_AGREE_ABOUT_A_TEACHER(tmp_path, monkeypatch, kind, teacher):
    monkeypatch.setitem(P.KINDS[kind], "slots", 3)
    h = GenHarness(tmp_path, kind=kind, n=3)
    h.seams.teacher = teacher
    code, rec = _conclude(kind, lambda o, r, lg: P._generate(kind, h.seams, o, r, lg), tmp_path / "ev")
    assert code == P.EXIT_VOID and rec["void"]["step"] == "teacher"


# ─────────────────────────────── the teacher on a tiny real network, through the real seams ───────────────────────────

def _teacher_arrays(perturb=True):
    """A first-cycle head stand-in: arm R on the tiny network, its residual output moved. Returns (arrays, digests)."""
    import mlx.core as mx

    from scripts.GPU.alphazero import policy_head_capacity as PHC
    net = _tiny_network()
    PHC.install(net, "R", init_seed=P.RECIPE["init_seed"])
    if perturb:
        w = net.policy_head.extra_out.weight
        net.policy_head.extra_out.weight = mx.array(np.random.default_rng(3).normal(0, 0.2, w.shape).astype(np.float32))
    arrays = {k: v for k, v in P.snapshot(net).items() if k.startswith("policy_head.")}
    return arrays, P.network_digests(net)


def _binding(arrays, digests):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return {"file": "first/head.npz", "sha256": "a" * 64, "digests": digests, "bytes": buf.getvalue()}


def _real_gen(monkeypatch, binding):
    from scripts.GPU.alphazero import probe_eval
    monkeypatch.setattr(probe_eval, "load_network_for_scoring", lambda path, verbose=False: (_tiny_network(), 30, 128, 1))
    return _REAL_GEN_SEAMS(pathlib.Path("/unused"), binding)


def test_THE_REAL_SEAMS_REACH_THE_HEAD_THROUGH_THE_UNCOMPILED_GENERATION_EVALUATOR(monkeypatch):
    """The real `_real_gen_seams` over a tiny network: both sides are uncompiled LocalGPUEvaluators, the digests are the
    teacher's, the priors differ and the value path does not. No checkpoint is read."""
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    arrays, digests = _teacher_arrays()
    seams = _real_gen(monkeypatch, _binding(arrays, digests))
    served = seams.load_evaluator()
    parent = seams.load_parent()
    assert type(served) is type(parent) is LocalGPUEvaluator and served._use_compile is parent._use_compile is False
    rec = {}
    P._bind_teacher(seams, served, rec)
    assert rec["teacher_digests"] == digests and served._compiled_forward is None
    assert rec["teacher_probe"]["max_prior_abs_diff"] > 0.0 and rec["teacher_probe"]["value_abs_diff"] == 0.0
    assert not any(m.training for _n, m in served.network.named_modules())
    assert np.array_equal(np.asarray(P.snapshot(served.network)["policy_head.extra_out.weight"]),
                          arrays["policy_head.extra_out.weight"])


def test_THE_REAL_PROBE_VOIDS_A_HEAD_THAT_DOES_NOT_CHANGE_THE_OPENING_PRIORS(monkeypatch):
    """Arm R with its residual at zero passes the digest checks (it has arm R's arrays) yet equals the parent: only the
    probe can catch it, through the real evaluator."""
    arrays, digests = _teacher_arrays(perturb=False)
    seams = _real_gen(monkeypatch, _binding(arrays, digests))
    with pytest.raises(P.Void) as e:
        P._bind_teacher(seams, seams.load_evaluator(), {})
    assert e.value.step == "teacher" and "does not reach the head" in e.value.detail


def test_THE_REAL_SEAMS_REFUSE_SAVED_ARRAYS_THAT_ARE_NOT_ARM_R_S_HEAD(monkeypatch):
    arrays, digests = _teacher_arrays()
    arrays.pop("policy_head.extra_out.weight")
    seams = _real_gen(monkeypatch, _binding(arrays, digests))
    with pytest.raises(P.Void) as e:
        P._bind_teacher(seams, seams.load_evaluator(), {})
    assert e.value.step == "head"


# ─────────────────────────────── the fit: the teacher's start ───────────────────────────────

def _teacher_seams(game, arrays, digests, played=None, order=None, **kw):
    seams = _train_seams(game, **kw)
    binding = {"file": "first/head.npz", "sha256": "a" * 64, "digests": digests}
    good = {"arch": dict(TINY_ARCH), "teacher": binding, "checkpoint_sha1": P.M.CHECKPOINT_SHA1}
    seams.read_record = lambda rel: {**good, **(played or {})}
    seams.teacher, seams.corpus_evidence = binding, "the-corpus"

    def install(net):
        if order is not None:
            order.append("install")
        return P._install_r_arrays(net, arrays)
    seams.install_teacher = install
    return seams


def _fit(tmp_path, seams, recipe, name="train"):
    return _conclude(TRAIN, lambda o, r, lg: P._train(recipe, seams, o, r, lg), tmp_path / name)


def test_THE_FIT_STARTS_FROM_ALL_THREE_TEACHER_DIGESTS_AND_LEAVES_THE_VALUE_BITWISE_THE_PARENT_S(tmp_path, corpus_game):
    arrays, digests = _teacher_arrays()
    assert np.any(arrays["policy_head.extra_out.weight"])          # the learned residual branch is not reset
    code, rec = _fit(tmp_path, _teacher_seams(corpus_game, arrays, digests), _recipe())
    assert code == P.EXIT_COMPLETED and rec["verdict"] == "ELIGIBLE", (rec.get("void"), rec.get("stop"))
    assert rec["digests_before"] == digests and rec["teacher"]["digests"] == digests
    assert rec["digests_after"]["encoder."] == digests["encoder."]
    assert rec["digests_after"]["value_head."] == digests["value_head."]
    assert rec["digests_after"]["policy_head."] != digests["policy_head."]
    assert rec["value_check"]["bitwise_equal"] and rec["value_check"]["max_abs_diff"] == 0.0
    assert rec["surface"]["parameters"] == 8840 and "r_init_sha256" not in rec
    assert rec["device"].startswith("Device(cpu") and len(rec["losses"]) == 12
    assert rec["heldout_after"]["ce"] < rec["prefit_heldout"]["ce"]
    with np.load(tmp_path / "train" / "head.npz") as z:
        assert sorted(z.files) == sorted(arrays)                   # the head alone: no optimizer state
        assert not np.array_equal(z["policy_head.extra_out.weight"], arrays["policy_head.extra_out.weight"])
    assert (rec["amendment"], rec["amendment_sha256"]) == P.CYCLE2_ATTEMPT2_AMENDMENT
    assert (rec["base_amendment"], rec["base_amendment_sha256"]) == P.CYCLE2_AMENDMENT


def test_THE_FIT_MEASURES_FROM_THE_TEACHER_NOT_THE_PARENT(tmp_path, corpus_game):
    arrays, digests = _teacher_arrays()
    _c, rec = _fit(tmp_path, _teacher_seams(corpus_game, arrays, digests), _recipe(), "t")
    _c, first = _fit(tmp_path, _train_seams(corpus_game), _recipe(), "p")
    assert rec["prefit_heldout"]["ce"] != first["prefit_heldout"]["ce"]
    assert first["digests_before"]["policy_head."] != digests["policy_head."] and "r_init_sha256" in first


def test_THE_VALUE_BASELINE_IS_TAKEN_FROM_THE_PARENT_BEFORE_THE_HEAD_IS_INSTALLED(tmp_path, corpus_game, monkeypatch):
    order = []
    real = P.values

    def spy(*a, **k):
        order.append("values")
        return real(*a, **k)
    monkeypatch.setattr(P, "values", spy)
    arrays, digests = _teacher_arrays()
    _c, rec = _fit(tmp_path, _teacher_seams(corpus_game, arrays, digests, order=order), _recipe())
    assert order == ["values", "install", "values"] and rec["verdict"] == "ELIGIBLE"


def test_THE_OPTIMIZER_IS_FRESH_ONE_ADAM_UNINITIALIZED_AT_ITS_FIRST_UPDATE(tmp_path, corpus_game, monkeypatch):
    import mlx.optimizers as optim
    made, seen = [], []
    base = optim.Adam

    class Spy(base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

        def update(self, model, grads):
            seen.append(self._initialized)
            return super().update(model, grads)
    monkeypatch.setattr(optim, "Adam", Spy)
    arrays, digests = _teacher_arrays()
    _c, rec = _fit(tmp_path, _teacher_seams(corpus_game, arrays, digests), _recipe())
    assert rec["verdict"] == "ELIGIBLE" and len(made) == 1 and len(seen) == 12
    assert seen[0] is False and all(seen[1:])


@pytest.mark.parametrize("recipe, rule", [
    ({"min_kl": 1e9}, "prefit_kl"), ({"l2": float("inf")}, "nonfinite"), ({"lr": 0.0, "l2": 0.0}, "heldout_not_improved")],
    ids=["prefit-kl", "non-finite", "heldout"])
def test_THE_FIRST_CYCLE_S_STOP_RULES_FIRE_FROM_THE_TEACHER(tmp_path, corpus_game, recipe, rule):
    arrays, digests = _teacher_arrays()
    code, rec = _fit(tmp_path, _teacher_seams(corpus_game, arrays, digests), _recipe(**recipe))
    assert code == P.EXIT_COMPLETED and rec["verdict"] == "STOP" and rec["stop"]["rule"] == rule


def test_A_CHANGED_VALUE_OUTPUT_STOPS_THE_FIT(tmp_path, corpus_game, monkeypatch):
    real, calls = P.values, []

    def shifted(*a, **k):
        calls.append(1)
        out = real(*a, **k)
        return out + 1e-6 if len(calls) == 2 else out
    monkeypatch.setattr(P, "values", shifted)
    arrays, digests = _teacher_arrays()
    _c, rec = _fit(tmp_path, _teacher_seams(corpus_game, arrays, digests), _recipe())
    assert rec["verdict"] == "STOP" and rec["stop"]["rule"] == "value_changed"
    assert not (tmp_path / "train" / "head.npz").exists()


@pytest.mark.parametrize("change, where", [
    ({"policy_head.": "0" * 64}, "policy head"), ({"encoder.": "0" * 64}, "encoder"), ({"value_head.": "0" * 64}, "value")],
    ids=["policy-head", "encoder", "value-head"])
def test_ANY_OF_THE_THREE_STARTING_DIGESTS_OFF_IS_A_VOID_BEFORE_FITTING(tmp_path, corpus_game, change, where):
    arrays, digests = _teacher_arrays()
    wrong = {**digests, **change}
    seams = _teacher_seams(corpus_game, arrays, wrong)
    code, rec = _fit(tmp_path, seams, _recipe())
    assert code == P.EXIT_VOID and rec["void"]["step"] == "teacher" and "starting digests" in rec["void"]["detail"], where
    assert "losses" not in rec and not (tmp_path / "train" / "head.npz").exists()


@pytest.mark.parametrize("played", [{"teacher": None}, {"teacher": {"file": "x", "sha256": "b" * 64, "digests": {}}},
                                    {"checkpoint_sha1": "0" * 40}], ids=["no-teacher", "another-teacher", "checkpoint"])
def test_A_CORPUS_THAT_DID_NOT_PLAY_THIS_TEACHER_IS_A_VOID_BEFORE_IT_IS_READ(tmp_path, corpus_game, played):
    arrays, digests = _teacher_arrays()
    seams = _teacher_seams(corpus_game, arrays, digests, played=played)

    def no_corpus():
        raise AssertionError("the corpus was read")
    seams.read_games = no_corpus
    code, rec = _fit(tmp_path, seams, _recipe())
    assert code == P.EXIT_VOID and rec["void"]["step"] == "teacher" and "did not play" in rec["void"]["detail"]


# ─────────────────────────────── the corpus's separation, through the real seam ───────────────────────────────

def test_THE_REAL_FIT_SEAM_READS_ONLY_THE_NEW_CORPUS(tmp_path, monkeypatch):
    """A file under the first corpus's name is not reachable from the second cycle's reader, whose games and rows are its
    own kind's; the first cycle's reader keeps its own paths."""
    arrays, digests = _teacher_arrays()
    root = tmp_path / "cache"
    for kind in ("generate", GEN2):
        (root / kind).mkdir(parents=True)
    np.savez(root / "generate" / "only_first.npz", a=np.zeros(1))
    np.savez(root / GEN2 / "only_second.npz", a=np.ones(1))
    seams = _REAL_TRAIN_SEAMS(pathlib.Path("/unused"), GEN2, _binding(arrays, digests), P.CYCLE2_VERSIONS)
    first = _REAL_TRAIN_SEAMS(pathlib.Path("/unused"))
    assert seams.corpus_evidence == P.KINDS[GEN2]["evidence"] and seams.teacher is not None
    assert seams.pinned_versions == P.CYCLE2_VERSIONS and first.pinned_versions is None
    assert first.teacher is None and first.corpus_evidence is None
    second_game = {"rows_file": "only_second.npz", "rows_sha256": P._sha_file(root / GEN2 / "only_second.npz")}
    first_game = {"rows_file": "only_first.npz", "rows_sha256": P._sha_file(root / "generate" / "only_first.npz")}
    assert seams.read_rows(second_game)["a"].tolist() == [1.0]
    assert first.read_rows(first_game)["a"].tolist() == [0.0]
    with pytest.raises(FileNotFoundError):
        seams.read_rows(first_game)
    with pytest.raises(P.Void, match="not the file generation wrote"):
        seams.read_rows({**second_game, "rows_sha256": "0" * 64})
    # RE-ANCHORED 2026-11-02 (the new corpus, 09ccb288): the reader returns its games, on its own block; the first one's are never read
    plan = json.loads((ROOT / P.KINDS[GEN2]["plan"]).read_text(encoding="utf-8"))
    read = seams.read_games()
    assert len(read) == plan["n_slots"] and all(g["seed"] == plan["base"] + g["slot"] for g in read)


def test_THE_REAL_FIT_SEAM_INSTALLS_THE_TEACHER_S_SAVED_BYTES(monkeypatch):
    arrays, digests = _teacher_arrays()
    seams = _REAL_TRAIN_SEAMS(pathlib.Path("/unused"), GEN2, _binding(arrays, digests))
    from scripts.GPU.alphazero import policy_head_capacity as PHC
    net = _tiny_network()
    assert tuple(sorted(seams.install_teacher(net))) == PHC.SURFACES["R"][0]
    assert P.network_digests(net) == digests


# ─────────────────────────────── the entry and admission ───────────────────────────────

@pytest.mark.parametrize("kind", [GEN, GEN2, TRAIN])
def test_THE_ENTRY_REFUSES_ON_A_CLOSED_GATE(monkeypatch, capsys, kind):
    monkeypatch.setattr(P, "_mlx_loaded", lambda: False)
    assert P.main(["run", kind, "--checkpoint", "/x"]) == P.EXIT_REFUSED
    assert "is closed" in capsys.readouterr().err


@pytest.mark.parametrize("kind, key", [(GEN, "amendment"), (GEN2, "amendment"), (GEN2, "base_amendment"),
                                       (TRAIN, "amendment"), (TRAIN, "base_amendment")])
def test_THE_ENTRY_REFUSES_AN_AMENDMENT_THAT_IS_NOT_THE_PINNED_ONE(monkeypatch, capsys, kind, key):
    gate, name = P.KINDS[kind]["gate"]
    monkeypatch.setattr(gate, name, True)
    monkeypatch.setattr(P, "_mlx_loaded", lambda: False)
    monkeypatch.setattr(P, "admitted", lambda k, *_a: {"stub": k})
    monkeypatch.setitem(P.KINDS[kind], key, (AMEND, "0" * 64))
    assert P.main(["run", kind, "--checkpoint", "/x"]) == P.EXIT_REFUSED
    assert "is not the pinned amendment" in capsys.readouterr().err


def _open_entry(monkeypatch, tmp_path, kind, seen):
    """The gate open in this process only; admission and the record boundary stubbed; the head read for real."""
    gate, name = P.KINDS[kind]["gate"]
    monkeypatch.setattr(gate, name, True)
    monkeypatch.setattr(P, "_mlx_loaded", lambda: False)
    monkeypatch.setattr(P, "admitted", lambda k, *_a: {"stub": k})
    monkeypatch.setitem(P.KINDS[kind], "evidence", str(tmp_path / "ev"))
    monkeypatch.setattr(P, "conclude", lambda k, execute, out, admission, head=None: (0, {"status": "COMPLETED"}))
    monkeypatch.setattr(P, "_real_gen_seams", lambda *a, **k: seen.update(gen=a))
    monkeypatch.setattr(P, "_real_train_seams", lambda *a, **k: seen.update(train=a))
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"not a model")
    return ckpt


def test_RUN_CYCLE2_GENERATE_PASSES_THE_COMMITTED_TEACHER_TO_THE_GENERATION_SEAMS(tmp_path, monkeypatch, capsys):
    seen = {}
    ckpt = _open_entry(monkeypatch, tmp_path, GEN2, seen)
    assert P.main(["run", GEN2, "--checkpoint", str(ckpt)]) == 0
    assert "train" not in seen and seen["gen"][0] == ckpt
    teacher = seen["gen"][1]
    assert teacher["sha256"] == P.TRAINED_HEAD["sha256"] and teacher["digests"] == P.TRAINED_HEAD["digests"]
    assert hashlib.sha256(teacher["bytes"]).hexdigest() == P.TRAINED_HEAD["sha256"]


def test_RUN_CYCLE2_TRAIN_PASSES_ITS_CORPUS_AND_THE_TEACHER_TO_THE_TRAINING_SEAMS(tmp_path, monkeypatch, capsys):
    seen = {}
    ckpt = _open_entry(monkeypatch, tmp_path, TRAIN, seen)
    assert P.main(["run", TRAIN, "--checkpoint", str(ckpt)]) == 0
    assert "gen" not in seen and seen["train"][:2] == (ckpt, GEN2) and seen["train"][3] == P.CYCLE2_VERSIONS
    assert seen["train"][2]["sha256"] == P.TRAINED_HEAD["sha256"]


def test_RUN_CYCLE2_GENERATE_REFUSES_AN_EXISTING_ROWS_DIRECTORY(tmp_path, monkeypatch, capsys):
    seen = {}
    ckpt = _open_entry(monkeypatch, tmp_path, GEN2, seen)
    (tmp_path / "cache" / GEN2).mkdir(parents=True)
    assert P.main(["run", GEN2, "--checkpoint", str(ckpt)]) == P.EXIT_REFUSED
    assert "the corpus is written once" in capsys.readouterr().err and not seen


def test_THE_TEACHER_COMES_FROM_THE_COMMITTED_FIRST_TRAINING_RUN_for_both_kinds():
    """The real committed readers, read-only."""
    for kind in (GEN, GEN2, TRAIN):
        h = P.trained_head(P._read_record, P._read_committed_bytes, kind)
        assert h["file"] == P.KINDS["train"]["evidence"] + "/head.npz" and h["sha256"] == P.TRAINED_HEAD["sha256"]
    train_rel = P.KINDS["train"]["evidence"]
    for kind in (GEN, GEN2):
        assert P.admitted(kind, P._read_record, P._read_outer) == {
            train_rel: "ELIGIBLE", P.TIMING_EVIDENCE: "COMPLETED", f"{train_rel}-run/outer.json": "COMPLETED"}


def _readers(have, outers):
    def rec(rel):
        if rel not in have:
            raise FileNotFoundError(rel)
        return have[rel]

    def out(rel):
        if rel not in outers:
            raise FileNotFoundError(rel)
        return outers[rel]
    return rec, out


FIRST, FIRST_RUN = P.KINDS["train"]["evidence"], P.KINDS["train"]["evidence"] + "-run"
CORPUS, CORPUS_RUN = P.KINDS[GEN2]["evidence"], str(pathlib.Path(P.KINDS[GEN2]["plan"]).parent)
ELIGIBLE = {**GOOD_INNER, "verdict": "ELIGIBLE"}
FULL = {FIRST: ELIGIBLE, CORPUS: GOOD_INNER, P.TIMING_EVIDENCE: {"status": "COMPLETED", "card_sha256": P.CARD_SHA256}}
FULL_OUTER = {FIRST_RUN: GOOD_OUTER, CORPUS_RUN: GOOD_OUTER}


def test_THE_FIT_IS_ADMITTED_BY_BOTH_RUNS_RECORDS_AND_OUTERS_AND_THE_TIMING():
    assert P.admitted(TRAIN, *_readers(FULL, FULL_OUTER)) == {
        FIRST: "ELIGIBLE", CORPUS: "COMPLETED", P.TIMING_EVIDENCE: "COMPLETED", f"{FIRST_RUN}/outer.json": "COMPLETED",
        f"{CORPUS_RUN}/outer.json": "COMPLETED"}


@pytest.mark.parametrize("have, outers, why", [
    ({FIRST: ELIGIBLE, P.TIMING_EVIDENCE: FULL[P.TIMING_EVIDENCE]}, FULL_OUTER, "does not exist"),
    ({**FULL, CORPUS: {**GOOD_INNER, "status": "VOID"}}, FULL_OUTER, "not COMPLETED"),
    ({**FULL, FIRST: {**GOOD_INNER, "verdict": "STOP"}}, FULL_OUTER, "not COMPLETED"),
    ({**FULL, CORPUS: {**GOOD_INNER, "card_sha256": "0" * 64}}, FULL_OUTER, "not COMPLETED"),
    (FULL, {FIRST_RUN: GOOD_OUTER}, "outer.json does not exist"),
    (FULL, {FIRST_RUN: GOOD_OUTER, CORPUS_RUN: {**GOOD_OUTER, "outcome": "TIMEOUT"}}, "outcome is 'TIMEOUT'"),
    (FULL, {FIRST_RUN: {**GOOD_OUTER, "survivors": "1 "}, CORPUS_RUN: GOOD_OUTER}, "survivors")],
    ids=["no-corpus", "corpus-void", "first-stop", "other-card", "no-corpus-outer", "corpus-timeout", "first-survivors"])
def test_THE_FIT_IS_REFUSED_WITHOUT_ANY_OF_THEM(have, outers, why):
    with pytest.raises(P.Refused, match=why):
        P.admitted(TRAIN, *_readers(have, outers))


# ─────────────────────────────── the wrappers ───────────────────────────────

WRAPPERS = {
    GEN: {"base": EV / "combination-policy-generate-run/01_wrapper.sh", "term": 21590, "stem": "cycle2-generate",
          "gate": "COMBINATION_POLICY_CYCLE2_GENERATE_AUTHORIZED",
          "gate_rel": "combination_policy_cycle2_generate_authorization.py"},
    GEN2: {"base": EV / "combination-policy-cycle2-generate-run/01_wrapper.sh", "term": 21590, "stem": "cycle2-generate2",
           "gate": "COMBINATION_POLICY_CYCLE2_GENERATE2_AUTHORIZED",
           "gate_rel": "combination_policy_cycle2_generate2_authorization.py"},
    TRAIN: {"base": EV / "combination-policy-train-run/01_wrapper.sh", "term": 2810, "stem": "cycle2-train",
            "gate": "COMBINATION_POLICY_CYCLE2_TRAIN_AUTHORIZED",
            "gate_rel": "combination_policy_cycle2_train_authorization.py"},
    CS: {"base": EV / "combination-policy-screen-run/01_wrapper.sh", "term": 3650, "stem": "cycle2-screen",
         "gate": "COMBINATION_POLICY_CYCLE2_SCREEN_AUTHORIZED", "gate_rel": "combination_policy_cycle2_screen_authorization.py"},
    CST: {"base": EV / "combination-policy-strength-run/01_wrapper.sh", "term": 50390, "stem": "cycle2-strength",
          "gate": "COMBINATION_POLICY_CYCLE2_STRENGTH_AUTHORIZED",
          "gate_rel": "combination_policy_cycle2_strength_authorization.py"}}


def _path(kind):
    return EV / f"combination-policy-{WRAPPERS[kind]['stem']}-run/01_wrapper.sh"


def _code(path):
    return path.read_text().split("set -u\n", 1)[1]


@pytest.mark.parametrize("kind", [GEN, GEN2, TRAIN])
def test_THE_WRAPPER_IS_THE_FIRST_CYCLE_S_WITH_ONLY_THE_STATED_DIFFERENCES(kind):
    """The paths, the gate and the `run` argument, line for line; TERM_S is the first cycle's. The header is this cycle's
    paragraph followed by the first cycle's header unchanged."""
    w = WRAPPERS[kind]
    old, new = _code(w["base"]).splitlines(), _code(_path(kind)).splitlines()
    assert len(old) == len(new)
    assert [b for a, b in zip(old, new) if a != b] == [
        f"OUT_REL=docs/superpowers/evidence/combination-policy-{w['stem']}-run",
        f"RUN_REL=docs/superpowers/evidence/combination-policy-{w['stem']}",
        f"GATE_REL=scripts/GPU/alphazero/{w['gate_rel']}", f"GATE={w['gate']}",
        '/usr/bin/perl -e "$SUPERVISOR" "$OUT_REL/pgid.txt" "$TERM_S" "$GRACE_S" env MLX_ENABLE_TF32=0 PYTHONPATH=. '
        f'"$PY" -B -m "$MODULE" run {kind} --checkpoint "$CKPT" \\']
    head_old = w["base"].read_text().split("set -u\n", 1)[0]
    head = _path(kind).read_text().split("set -u\n", 1)[0]
    assert head.startswith("#!/bin/bash\n# THE SECOND POLICY-IMPROVEMENT CYCLE: ") and AMEND in head
    assert head.endswith(head_old[len("#!/bin/bash\n"):])
    assert os.access(_path(kind), os.R_OK) and subprocess.run(["bash", "-n", str(_path(kind))]).returncode == 0
    assert _path(kind).stat().st_mode == w["base"].stat().st_mode


@pytest.mark.parametrize("kind", [GEN, GEN2, TRAIN, CS, CST])
def test_THE_WRAPPER_SERVES_ITS_KIND_S_PATHS(kind):
    w, spec = _code(_path(kind)), P.KINDS[kind]
    out, run = (re.search(rf"^{k}=(\S+)$", w, re.M).group(1) for k in ("OUT_REL", "RUN_REL"))
    assert run == spec["evidence"] and out == str(_path(kind).parent.relative_to(ROOT))
    assert re.search(r"^GATE_REL=(\S+)$", w, re.M).group(1) == str(pathlib.Path(spec["gate"][0].__file__).relative_to(ROOT))
    assert spec["gate"][1] == WRAPPERS[kind]["gate"] and f"RECORD_KIND={P.RECORD_KIND}\n" in w
    assert 'CKPT="$MAIN/checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"' in w
    first = {P.KINDS[k]["evidence"] for k in P.KINDS if not k.startswith("cycle2")}
    assert run not in first and out not in {f"{p}-run" for p in first}


@pytest.mark.parametrize("kind", [GEN, GEN2])
def test_THE_GENERATION_CEILING_IS_THE_AMENDMENT_S_SIX_HOURS(kind):
    w = _code(_path(kind))
    term, grace = (int(re.search(rf"^{k}=(\d+)$", w, re.M).group(1)) for k in ("TERM_S", "GRACE_S"))
    assert (term, grace) == (21590, 10) and term + grace == 6 * 3600 and term + 2 * grace == 21610


def _fit_ceiling(games_path):
    """The timing record's formula on a corpus's counts x 1.5, rounded up to the minute (slots and row counts only --
    never a winner or reason)."""
    timing = json.loads((ROOT / P.TIMING_EVIDENCE / "record.json").read_text(encoding="utf-8"))
    counts = [(g["slot"], g["n_rows"]) for g in
              (json.loads(line) for line in games_path.read_text(encoding="utf-8").splitlines())]
    rows = sum(n for _s, n in counts)
    held = sum(n for s, n in counts if s % P.RECIPE["heldout_mod"] == P.RECIPE["heldout_rem"])
    t = timing["timing"]
    projected = math.fsum([t["mean_update_seconds"] * P.RECIPE["updates"], t["per_row_pair_forward_seconds"] * (rows + held),
                           t["per_row_pair_forward_seconds"] * held, t["per_to_tensor_seconds"] * rows])
    return int(math.ceil(timing["spec"]["ceiling_factor"] * projected / 60.0) * 60)


def test_THE_FIT_CEILING_IS_THE_FORMULA_ON_THE_NEW_CORPUS_S_ACTUAL_COUNTS():
    """BUILT provisional, on the first corpus's derivation. RE-ANCHORED 2026-11-02 (the user's review): the header states the
    derivation from the new corpus's counts, recomputed here from its committed games and the timing record. The executable
    limits did not change: TERM 2,810 s, KILL 2,820 s, observation through 2,830 s."""
    new = ROOT / P.KINDS[GEN2]["evidence"] / "games.jsonl"
    counts = [(g["slot"], g["n_rows"]) for g in (json.loads(x) for x in new.read_text(encoding="utf-8").splitlines())]
    rows = sum(n for _s, n in counts)
    held = sum(n for s, n in counts if s % P.RECIPE["heldout_mod"] == P.RECIPE["heldout_rem"])
    w = _code(_path(TRAIN))
    term, grace = (int(re.search(rf"^{k}=(\d+)$", w, re.M).group(1)) for k in ("TERM_S", "GRACE_S"))
    assert (term, grace) == (2810, 10) and term + grace == _fit_ceiling(new) == 2820
    header = " ".join(ln.lstrip("# ") for ln in _path(TRAIN).read_text(encoding="utf-8").split("set -u\n", 1)[0].splitlines())
    assert "PROVISIONAL" not in header
    assert f"{rows:,} rows, {held:,} held out" in header and "2,820 s" in header and "through 2830 s" in header


# ─────────────────────────────── fake launches of the REAL wrappers ───────────────────────────────

def _fake_repo(tmp, kind, term_s=None):
    """A throwaway repository: the real wrapper (TERM_S changed only when a test says so), its gate opened by a one-line
    commit, and three stubs at this kind's paths."""
    w = WRAPPERS[kind]
    repo = tmp / "repo"
    pkg = repo / "scripts/GPU/alphazero"
    pkg.mkdir(parents=True)
    for d in (repo / "scripts", repo / "scripts/GPU", pkg):
        (d / "__init__.py").write_text("")
    gate = pkg / w["gate_rel"]
    gate.write_text(f'"""stub gate"""\n{w["gate"]} = False\n')
    (pkg / "idle_gate.py").write_text(STUB_IDLE)
    (pkg / "background_compute.py").write_text(STUB_BG)
    stub = STUB_PILOT.replace("combination-policy-throughput", f"combination-policy-{w['stem']}").replace(
        '"head": head, "written_by": "script"}', '"head": head, "written_by": "script",\n'
        '                                                 "amendment_sha256": os.environ["FAKE_AMENDMENT_SHA"]}')
    assert stub.count(f"combination-policy-{w['stem']}") == 2 and "FAKE_AMENDMENT_SHA" in stub
    (pkg / "combination_policy_pilot.py").write_text(stub)
    wdir = repo / f"docs/superpowers/evidence/combination-policy-{w['stem']}-run"
    wdir.mkdir(parents=True)
    raw = _path(kind).read_bytes()
    if term_s is not None:
        assert raw.count(f"\nTERM_S={w['term']}\n".encode()) == 1
        raw = raw.replace(f"\nTERM_S={w['term']}\n".encode(), f"\nTERM_S={term_s}\n".encode())
    (wdir / "01_wrapper.sh").write_bytes(raw)
    (repo / ".venv/bin").mkdir(parents=True)
    (repo / ".venv/bin/python").symlink_to(sys.executable)
    ckpt = repo / "checkpoints/alphazero-v2-calib020-from0409"
    ckpt.mkdir(parents=True)
    (ckpt / "model_iter_0001.safetensors").write_bytes(b"not a model")
    (repo / ".gitignore").write_text(".venv/\ncheckpoints/\n__pycache__/\nscratch/\n")
    _git(tmp, "init", "-q", str(repo))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    gate.write_text(gate.read_text().replace(f"{w['gate']} = False", f"{w['gate']} = True"))
    _git(repo, "commit", "-qam", "open the gate")
    (repo / "scratch").mkdir()
    return repo, wdir, gate


def _launch(repo, wdir, idle="idle", bg="clear", run="ok", timeout=60):
    env = {k: v for k, v in os.environ.items() if k not in ("MLX_ENABLE_TF32", "PYTHONPATH")}
    env.update(FAKE_IDLE=idle, FAKE_BG=bg, FAKE_RUN=run, FAKE_OUT=str(repo / "scratch"), PYTHONDONTWRITEBYTECODE="1",
               FAKE_CARD_SHA=P.CARD_SHA256, FAKE_AMENDMENT_SHA=P.CYCLE2_AMENDMENT[1])
    return subprocess.run(["bash", str(wdir / "01_wrapper.sh")], cwd=str(repo), env=env, capture_output=True,
                          text=True, timeout=timeout)


@pytest.mark.parametrize("kind", [GEN2, TRAIN, CS, CST])
@pytest.mark.parametrize("idle, bg", [("busy", "clear"), ("idle", "boinc"), ("idle", "failed")],
                         ids=["busy", "boinc", "check-fails"])
def test_FAKE_LAUNCH_A_BUSY_MACHINE_BOINC_OR_A_FAILED_CHECK_PREVENTS_THE_LAUNCH(tmp_path, kind, idle, bg):
    repo, wdir, gate = _fake_repo(tmp_path, kind)
    p = _launch(repo, wdir, idle, bg)
    assert p.returncode == 1 and ": nothing launched" in p.stderr, p.stderr
    assert "FAKE_LAUNCHED" not in p.stdout + p.stderr
    assert sorted(x.name for x in wdir.iterdir()) == ["01_wrapper.sh"]
    assert f"{WRAPPERS[kind]['gate']} = False\n" in gate.read_text()


@pytest.mark.parametrize("kind", [GEN2, TRAIN, CS, CST])
def test_FAKE_LAUNCH_AN_IDLE_MACHINE_REACHES_ITS_RUN_with_its_exact_arguments_and_ceiling(tmp_path, kind):
    repo, wdir, gate = _fake_repo(tmp_path, kind)
    p = _launch(repo, wdir, run="record")
    assert p.returncode == 0, p.stderr
    launched = json.loads((wdir / "run.stdout.txt").read_text().split("FAKE_LAUNCHED", 1)[1])
    ckpt = str(repo.resolve() / "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors")
    assert launched["argv"] == ["run", kind, "--checkpoint", ckpt] and launched["tf32"] == "0"
    outer = json.loads((wdir / "outer.json").read_text())
    assert outer["outcome"] == "COMPLETED" and outer["survivors"] == "" and outer["gate_restored"] is True
    assert outer["term_s"] == WRAPPERS[kind]["term"] and outer["grace_s"] == 10 and outer["record_by"] == "script"
    rec = json.loads((wdir.parent / f"combination-policy-{WRAPPERS[kind]['stem']}" / "record.json").read_text())
    assert rec["amendment_sha256"] == P.CYCLE2_AMENDMENT[1] and not P.outer_faults(outer, rec)
    assert f"{WRAPPERS[kind]['gate']} = False\n" in gate.read_text()


@pytest.mark.parametrize("kind", [GEN2, TRAIN, CS, CST])
def test_FAKE_LAUNCH_THE_CEILING_TERMINATES_THE_WHOLE_GROUP_at_its_paths(tmp_path, kind):
    """TERM_S changed to 2 in this copy only: the run and its child outlive it; the group TERM ends both, and the
    wrapper's VOID record lands in this kind's evidence directory."""
    repo, wdir, gate = _fake_repo(tmp_path, kind, term_s=2)
    t = time.monotonic()
    p = _launch(repo, wdir, run="outlive")
    child = int((repo / "scratch/child.pid").read_text())
    assert time.monotonic() - t < 30 and p.returncode == 124, p.stderr
    for _ in range(50):
        if not _alive(child):
            break
        time.sleep(0.1)
    assert not _alive(child)
    outer = json.loads((wdir / "outer.json").read_text())
    assert outer["outcome"] == "TIMEOUT" and outer["inner_exit"] == 124 and outer["survivors"] == ""
    rec = json.loads((wdir.parent / f"combination-policy-{WRAPPERS[kind]['stem']}" / "record.json").read_text())
    assert outer["record_by"] == "wrapper" and rec["status"] == "VOID" and rec["void"]["step"] == "timeout"
    assert f"{WRAPPERS[kind]['gate']} = False\n" in gate.read_text()


# ─────────────────────────────── the generation seed-survey copy (built, never run here) ──────────────────────────────

def test_THE_SURVEY_COPY_IS_THE_FIRST_CORPUS_S_WITH_ONLY_THE_STATED_DIFFERENCES():
    """Below the docstring: the paths, the plan's kind, the module name, the titles and verdict lines, line for line. The
    block size, method, rule, scope and controls are the first corpus's. The docstring is this cycle's paragraph followed
    by the first corpus's description unchanged."""
    doc1, code1 = _split(SURVEY_1)
    doc, code = _split(SURVEY)
    assert doc.endswith(doc1) and AMEND in doc[:-len(doc1)]
    a, b = code1.splitlines(), code.splitlines()
    assert len(a) == len(b)
    assert [y.strip() for x, y in zip(a, b) if x != y] == [
        'HERE = "docs/superpowers/evidence/combination-policy-cycle2-generate-seed-survey"',
        'PLAN = "docs/superpowers/evidence/combination-policy-cycle2-generate-run/plan.json"',
        'VERDICT_OK = "THE COMBINATION-POLICY CYCLE-2 GENERATION BLOCK IS CLEAN"',
        'spec = importlib.util.spec_from_file_location(f"_combination_policy_cycle2_generate_{key}", p)',
        'return {"kind": "combination-policy-cycle2-generate-plan/1", "base": base, "n_slots": N_SLOTS,',
        'faults.append("the cycle-2 generation plan is absent or not committed")',
        'faults.append("the cycle-2 generation plan is not build_plan of the surveyed block")',
        "say(f\"COMBINATION POLICY PILOT -- CYCLE-2 GENERATION GAME-SEED {'SURVEY' if mode == 'survey' else "
        "'COLLISION PROOF, mode ' + mode.upper()}\")",
        "say(f\"PROOF VERDICT ({mode}): {'THE COMBINATION-POLICY CYCLE-2 GENERATION BLOCK IS CLEAN AND THE STATE IS AS "
        "EXPECTED' if ok else 'NOT PROVEN -- STOP'}\""]


def test_THE_SURVEY_COPY_SERVES_THE_KIND_IN_THE_2027_BAND_and_holds_no_seed_literal():
    survey = _load(SURVEY2, "_c2_survey")
    kind = P.KINDS[GEN2]
    assert survey.N_SLOTS == kind["slots"] == 1000 and survey.PLAN == kind["plan"]
    S, _Pf, SEL = survey.load_method()
    assert S.N == 1000 and S.scan is SEL.scan and SEL.__name__.endswith("seed_band_2027")
    assert survey.PINNED == _load(SURVEY_1, "_c1_survey_pins").PINNED
    base = SEL.candidate([], [], 1000)[1]
    assert survey.build_plan(base, "0" * 64) == {"kind": "combination-policy-cycle2-generate2-plan/1", "base": base,
                                                 "n_slots": 1000, "survey_record_sha256": "0" * 64}
    assert set(survey.OUT.values()) & set(_load(SURVEY, "_c2_first_out").OUT.values()) == set()
    code = _split(SURVEY2)[1]
    assert not SB.scan(code) and not re.search(r"\b20[0-9]{7}\b", code)


def test_THE_SURVEY_COPY_S_CONTROLS_ALL_PASS_on_a_synthetic_2027_block():
    survey = _load(SURVEY2, "_c2_survey_controls")
    S, Pf, SEL, block, prior, masks = _synthetic(survey)
    rec = {"head": "0" * 40, "registry_sha256": "0" * 64, "base": block[0], "block": block}
    groups = {"state": survey._state_controls(S, Pf, rec, "a" * 64),
              "plants": S._plant_controls(block, prior, masks, {}, True),
              "exemptions": S._exemption_controls(block), "enumeration": Pf._enumeration_controls(S),
              "two-band": survey._two_band_controls(S, SEL, block, prior, masks)}
    assert {k: len(v) for k, v in groups.items()} == {"state": 12, "plants": 8, "exemptions": 6, "enumeration": 6,
                                                      "two-band": 10}
    for k, v in groups.items():
        assert all(ok for _l, ok, _d in v), (k, [r for r in v if not r[1]])


def _mode(repo, mode, monkeypatch, capsys):
    """Run the copy's `main` for one mode, loaded from the synthetic repository, so every output is there."""
    import importlib.util
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spec = importlib.util.spec_from_file_location(f"_c2_survey_{mode}", repo / "docs/superpowers/evidence" / SURVEY2)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    code = mod.main(["x", "--mode", mode])
    capsys.readouterr()
    return mod, code, (repo / mod.OUT[mode]).read_text(encoding="utf-8")


def test_THE_THREE_MODES_REHEARSED_ON_A_SYNTHETIC_REPOSITORY(tmp_path, monkeypatch, capsys):
    """survey -> pre -> registration -> post, on a throwaway repository holding only the copy, the pinned method, a
    synthetic registry and two planted literals. Nothing of the real repository is scanned or written: this is no
    survey of it."""
    from scripts.GPU.alphazero import e4_screen_reference as REG
    repo = tmp_path / "synthetic"

    def git(*a):
        return subprocess.check_output(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo,
                                       text=True).strip()
    for rel in (*PINNED_FILES, f"docs/superpowers/evidence/{SURVEY2}"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes((ROOT / rel).read_bytes())
    top = max([SB.BAND[0]] + [hi for lo, hi in REG.ACCOUNTED_SEED_INTERVALS if hi > SB.BAND[0]])
    old, new = SB.BAND_2026[0] + 61_000, top + 23_456
    reg = repo / "scripts/GPU/alphazero/e4_screen_reference.py"
    reg.write_text(f"ACCOUNTED_SEED_INTERVALS = (\n    ({old}, {old + 10}),\n)\nEXPOSED_SEED_INTERVALS = ()\n"
                   "RETIRED_SEED_INTERVALS = ()\nTEST_ONLY_SEED_INTERVALS = ()\n")
    (repo / "tests").mkdir()
    (repo / "tests/test_planted.py").write_text(f"A = {old + 500}\nB = {new:_}\n")
    (repo / "docs/superpowers/evidence/combination-policy-cycle2-generate2-run").mkdir(parents=True)
    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "synthetic")

    mod, code, survey = _mode(repo, "survey", monkeypatch, capsys)
    assert code == 0, survey[-2500:]
    anchor, base, block = SB.candidate(REG.ACCOUNTED_SEED_INTERVALS, [new], 1000)
    assert anchor == new and f"BASE = {base}; block = [{block[0]}, {block[1]}) -- 1000 seeds" in survey
    assert survey.endswith("SURVEY VERDICT   : THE COMBINATION-POLICY CYCLE-2 GENERATION ATTEMPT-2 BLOCK IS CLEAN\n")
    passed, total = re.search(r"^controls passed: (\d+)/(\d+)$", survey, re.M).groups()
    assert passed == total and int(total) == 55, total
    git("add", "-A")
    git("commit", "-qm", "survey record")
    _m, code, pre = _mode(repo, "pre", monkeypatch, capsys)
    assert code == 0 and "EXPECTED STATE (pre): HOLDS" in pre, pre[-2500:]
    git("add", "-A")
    git("commit", "-qm", "pre record")
    sha = hashlib.sha256((repo / mod.OUT["survey"]).read_bytes()).hexdigest()
    reg.write_text(reg.read_text().replace(f"    ({old}, {old + 10}),\n)",
                                           f"    ({old}, {old + 10}),\n    ({block[0]}, {block[1]}),\n)"))
    (repo / mod.PLAN).write_text(json.dumps(mod.build_plan(block[0], sha), indent=1, sort_keys=True) + "\n")
    git("add", "-A")
    git("commit", "-qm", "registration")
    _m, code, post = _mode(repo, "post", monkeypatch, capsys)
    assert code == 0 and "EXPECTED STATE (post): HOLDS" in post, post[-2500:]
    assert post.endswith("PROOF VERDICT (post): THE COMBINATION-POLICY CYCLE-2 GENERATION ATTEMPT-2 BLOCK IS CLEAN AND THE "
                         "STATE IS AS EXPECTED\n")


# ─────────────────────────────── the generation closing copy (built, never run here) ──────────────────────────────────

def test_THE_CLOSING_COPY_IS_THE_FIRST_CORPUS_S_WITH_ONLY_THE_STATED_DIFFERENCES():
    doc1, code1 = _split(CLOSING_1)
    doc, code = _split(CLOSING)
    assert doc.endswith(doc1) and AMEND in doc[:-len(doc1)]
    a, b = code1.splitlines(), code.splitlines()
    assert len(a) == len(b) and not [t for t in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
                                     if t[0] in ("insert", "delete")]
    assert {y.strip() for x, y in zip(a, b) if x != y} == {
        'HERE = "docs/superpowers/evidence/combination-policy-cycle2-generate-seed-closing"',
        'RUN = "docs/superpowers/evidence/combination-policy-cycle2-generate"',
        'RUN_DIR = "docs/superpowers/evidence/combination-policy-cycle2-generate-run"',
        'lines = ["COMBINATION POLICY PILOT -- CYCLE-2 GENERATION SEED-USAGE DERIVATION (closing)", "=" * 96,'}


def test_THE_CLOSING_COPY_SERVES_THE_KIND_and_holds_no_seed_literal_and_its_controls_pass_at_the_registered_base():
    closing = _load(CLOSING, "_c2_closing")
    kind = P.KINDS[GEN]
    assert closing.N_SLOTS == kind["slots"] == 1000 and closing.INPUTS[0] == kind["plan"]
    assert closing.RUN == kind["evidence"] and closing.RUN_DIR == str(pathlib.Path(kind["plan"]).parent)
    assert closing.OUT == "docs/superpowers/evidence/combination-policy-cycle2-generate-seed-closing/02_derive_usage_run.txt"
    code = _split(CLOSING)[1]
    assert not SB.scan(code) and not re.search(r"\b20[0-9]{7}\b", code)
    results = closing._controls(json.loads((ROOT / kind["plan"]).read_text(encoding="utf-8"))["base"])
    assert len(results) == 21 and all(ok for _l, ok, _d in results), [r for r in results if not r[1]]


# ─────────────────────────────── the new generation attempt: the pin, its kind and its copies ──────────────────────────

def test_THE_ATTEMPT_2_AMENDMENT_IS_PINNED_AND_THE_PIN_IS_PER_KIND():
    rel, sha = P.CYCLE2_ATTEMPT2_AMENDMENT
    assert rel == AMEND2 and P.M._sha((ROOT / rel).read_bytes()) == sha
    text = (ROOT / rel).read_text(encoding="utf-8")
    assert AMEND in text and "3.14.10" in text and "COMBINATION_POLICY_CYCLE2_GENERATE2_AUTHORIZED" in text
    assert P.CYCLE2_VERSIONS == {**P.VERSIONS, "python": "3.14.10"} and P.VERSIONS["python"] == "3.14.9"
    assert {k for k, v in P.KINDS.items() if "versions" in v} == {GEN2, TRAIN, CS, CST}
    assert P.KINDS[GEN2]["versions"] == P.KINDS[TRAIN]["versions"] == P.CYCLE2_VERSIONS
    assert P.KINDS[GEN2]["amendment"] == P.KINDS[TRAIN]["amendment"] == P.CYCLE2_ATTEMPT2_AMENDMENT


def test_THE_LIVE_ENVIRONMENT_IS_THE_PINNED_ONE():
    """Not a seam: the suites' doubles supply the expected versions, which is how entry 47's mismatch got past them. This
    reads the real interpreter and packages, so an upgrade fails here before it can reach a seed sequence or a launch."""
    assert P._versions() == P.CYCLE2_VERSIONS


def test_THE_NEW_ATTEMPT_RAN_ONCE_COMPLETED_AND_ITS_BLOCK_IS_CLOSED():
    """BUILT 2026-11-02 as the build-only state. RE-ANCHORED 2026-11-02 at the seed preparation (survey 4fbb855a, pre-proof
    0043d826, registration 4c599111, post-proof 96245e21). RE-ANCHORED 2026-11-02 (the run, fc86a031 -> 09ccb288, and its
    closing, K1 41fad25f + K2 ba2ef9af): the survey is CLEAN and both proofs HOLD, each 55/55 on a clean tree and bound to
    the survey record; the plan is the copy's build_plan; the post-proof proved the registry the run launched with. The ONE
    run, launched from the one-line gate opening, COMPLETED inside its ceiling: outer exit 0, gate restored, no survivors,
    the pinned versions, the teacher bound by digests and its probe differing from the parent. Every slot was claimed,
    started and ended once, slot g on seed BASE + g, and every row file is the one generation wrote (games.jsonl is read
    for slot, seed, row-count and file fields only -- never a winner or reason). K1 RECONCILES, so the WHOLE block is EXPOSED
    and RETIRED, each exactly once and last, ACCOUNTED unchanged, every seed refused. Every mode refuses to run again. The
    gates stay closed. No block literal here: the proofs scan tests/. RE-ANCHORED 2026-11-02 (the fit, cafc78b4 -> 200e7eee): its
    evidence moved to its own state test below."""
    from scripts.GPU.alphazero import e4_screen_reference as REG
    here = (EV / SURVEY2).parent
    assert sorted(x.name for x in here.iterdir() if x.name != "__pycache__") == \
        ["01_seed_survey.py", "02_seed_survey_run.txt", "03_collision_proof_pre_run.txt",
         "04_collision_proof_post_run.txt"]
    raw = (here / "02_seed_survey_run.txt").read_bytes()
    survey, sha = raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()
    assert survey.splitlines()[2].endswith("; working tree clean") and "controls passed: 55/55" in survey
    assert survey.endswith("SURVEY VERDICT   : THE COMBINATION-POLICY CYCLE-2 GENERATION ATTEMPT-2 BLOCK IS CLEAN\n")
    for rel, mode in (("03_collision_proof_pre_run.txt", "pre"), ("04_collision_proof_post_run.txt", "post")):
        text = (here / rel).read_text(encoding="utf-8")
        assert text.splitlines()[2].endswith("; working tree clean")
        assert f"02_seed_survey_run.txt sha256 {sha} (read at " in text
        assert f"EXPECTED STATE ({mode}): HOLDS" in text and "controls passed: 55/55" in text
        assert text.endswith(f"PROOF VERDICT ({mode}): THE COMBINATION-POLICY CYCLE-2 GENERATION ATTEMPT-2 BLOCK IS CLEAN AND THE "
                             "STATE IS AS EXPECTED\n")
    post = (here / "04_collision_proof_post_run.txt").read_text(encoding="utf-8")
    plan = json.loads((ROOT / P.KINDS[GEN2]["plan"]).read_text(encoding="utf-8"))
    n = P.KINDS[GEN2]["slots"]
    assert type(plan["base"]) is int and plan["n_slots"] == n == 1000
    mod = _load(SURVEY2, "_c2_survey2_registered")
    assert plan == mod.build_plan(plan["base"], sha)
    block = (plan["base"], plan["base"] + n)
    first = json.loads((ROOT / P.KINDS[GEN]["plan"]).read_text(encoding="utf-8"))
    assert first["base"] + first["n_slots"] <= block[0]
    assert SB.BAND[0] <= block[0] < block[1] <= SB.BAND[1] and block[0] % SB.ALIGN == 0
    assert f"block = [{block[0]}, {block[1]}) -- {n} seeds" in survey
    assert f"THE BLOCK, read from the record: [{block[0]}, {block[1]}) -- BASE {block[0]}" in post
    for mode in ("survey", "pre", "post"):
        with pytest.raises(mod.Refused, match="each mode runs once"):
            mod.main(["x", "--mode", mode])
    # the run
    run_dir, evidence = _path(GEN2).parent, ROOT / P.KINDS[GEN2]["evidence"]
    assert sorted(x.name for x in run_dir.iterdir()) == ["01_wrapper.sh", "bg_check.txt", "idle_check.txt", "outer.json",
                                                         "pgid.txt", "plan.json", "run.stderr.txt", "run.stdout.txt"]
    assert sorted(x.name for x in evidence.iterdir()) == ["games.jsonl", "progress.jsonl", "record.json"]
    outer = json.loads((run_dir / "outer.json").read_text(encoding="utf-8"))
    rec = json.loads((evidence / "record.json").read_text(encoding="utf-8"))
    assert P.outer_faults(outer, rec) == [] and (outer["term_s"], outer["grace_s"]) == (21590, 10)
    assert (rec["status"], rec["run"], rec["verdict"]) == ("COMPLETED", GEN2, None)
    assert rec["versions"] == P.CYCLE2_VERSIONS and rec["amendment_sha256"] == P.CYCLE2_ATTEMPT2_AMENDMENT[1]
    assert rec["base_amendment_sha256"] == P.CYCLE2_AMENDMENT[1] and rec["seeds"] == {"base": block[0], "n_slots": n}
    head_file = ROOT / rec["teacher"]["file"]
    assert rec["teacher"]["sha256"] == hashlib.sha256(head_file.read_bytes()).hexdigest()
    assert rec["teacher_digests"] == rec["teacher"]["digests"] and rec["teacher_probe"]["max_prior_abs_diff"] > 0.0
    assert rec["checkpoint_sha1"] == json.loads((ROOT / P.KINDS["generate"]["evidence"] / "record.json").read_text(
        encoding="utf-8"))["checkpoint_sha1"]
    events = [json.loads(x) for x in (evidence / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[0]["event"] == "start" and events[-1]["event"] == "end"
    assert sorted(e["slot"] for e in events if e["event"] == "game_start") == list(range(n))
    assert sorted(e["slot"] for e in events if e["event"] == "game_end") == list(range(n))
    assert [e["claims"] for e in events if e["event"] == "claims"] == [n]
    games = [json.loads(x) for x in (evidence / "games.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [g["slot"] for g in sorted(games, key=lambda g: g["slot"])] == list(range(n))
    assert all(g["seed"] == block[0] + g["slot"] for g in games)
    assert sum(g["n_rows"] for g in games) == rec["summary"]["rows"] and rec["summary"]["games"] == n
    rows_dir = pathlib.Path.home() / "twixt-cache/combination-policy-pilot" / GEN2
    assert sorted(x.name for x in rows_dir.iterdir()) == sorted(g["rows_file"] for g in games)
    assert all(P._sha_file(rows_dir / g["rows_file"]) == g["rows_sha256"] for g in games)
    gate_rel = "scripts/GPU/alphazero/combination_policy_cycle2_generate2_authorization.py"
    assert subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"{outer['head']}^"], capture_output=True, text=True,
                          check=True).stdout.strip() == outer["pre_open"]
    opened = subprocess.run(["git", "-C", str(ROOT), "diff", "--unified=0", outer["pre_open"], outer["head"]],
                            capture_output=True, text=True, check=True).stdout
    assert [ln for ln in opened.splitlines() if ln[:1] in "+-" and ln[:3] not in ("+++", "---")] == \
        [f"-{WRAPPERS[GEN2]['gate']} = False", f"+{WRAPPERS[GEN2]['gate']} = True"] and f"b/{gate_rel}" in opened
    reg_rel = "scripts/GPU/alphazero/e4_screen_reference.py"
    launched = subprocess.run(["git", "-C", str(ROOT), "show", f"{outer['head']}:{reg_rel}"], capture_output=True,
                              check=True).stdout
    assert f"\nregistry {reg_rel} sha256 {hashlib.sha256(launched).hexdigest()}\n" in post     # proved = launched
    assert subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "09ccb288", "HEAD", "--", str(run_dir),
                           str(evidence)]).returncode == 0
    # the closing
    closing = (EV / CLOSING2).parent
    assert sorted(x.name for x in closing.iterdir() if x.name != "__pycache__") == \
        ["01_derive_usage.py", "02_derive_usage_run.txt"]
    k1 = (closing / "02_derive_usage_run.txt").read_text(encoding="utf-8")
    assert k1.splitlines()[2].endswith("; working tree clean") and "controls passed: 21/21" in k1
    assert f"\nclaims events: [{n}]\n" in k1
    assert k1.endswith(f"VERDICT: RECONCILES -- EXPOSE {list(block)}; RETIRE {list(block)}\n")
    for intervals in (REG.ACCOUNTED_SEED_INTERVALS, REG.EXPOSED_SEED_INTERVALS, REG.RETIRED_SEED_INTERVALS):
        # RE-ANCHORED 2026-11-02 (the screen's registration 322aae07): no longer last in ACCOUNTED; later entries lie above
        i = intervals.index(block)
        assert intervals.count(block) == 1 and all(lo >= block[1] for lo, _hi in intervals[i + 1:])
        assert all(hi <= block[0] for lo, hi in intervals[:i] if hi > SB.BAND[0])
    assert all(REG.seed_is_accounted(s) and REG.seed_is_exposed(s) and REG.seed_is_retired(s) and
               REG.seed_is_unavailable(s) and not REG.seed_is_test_only(s) for s in range(*block))
    # the gates, the rest of the state
    assert GATE_GEN2.COMBINATION_POLICY_CYCLE2_GENERATE2_AUTHORIZED is False and P.KINDS[GEN2]["gate"][0] is GATE_GEN2
    assert GATE_TRAIN.COMBINATION_POLICY_CYCLE2_TRAIN_AUTHORIZED is False


def test_THE_FIT_RAN_ONCE_COMPLETED_ELIGIBLE_FROM_THE_FIRST_HEAD_ON_THE_NEW_CORPUS():
    """BUILT 2026-11-02 at the run (cafc78b4 -> 200e7eee): the ONE run COMPLETED inside the ceiling, launched from the one-line
    gate opening, gate restored, no survivors. Admitted through the real committed readers. On the CPU in the pinned
    versions, with the recipe unchanged, from the first head (the teacher's three digests), the new corpus's real counts.
    The pre-fit KL is at or above the floor; 3,000 finite losses; the parent's value baseline bitwise equal on every held-out
    row; encoder and value-head digests unchanged while the policy head's changed; held-out CE below its pre-fit value, so
    ELIGIBLE. head.npz holds exactly the digest and sha256 recorded, on the permitted surface, and the first head's bytes
    are unchanged. Every other gate stays closed. Read-only."""
    run_dir, evidence = _path(TRAIN).parent, ROOT / P.KINDS[TRAIN]["evidence"]
    assert GATE_TRAIN.COMBINATION_POLICY_CYCLE2_TRAIN_AUTHORIZED is False and P.KINDS[TRAIN]["gate"][0] is GATE_TRAIN
    assert sorted(x.name for x in run_dir.iterdir()) == ["01_wrapper.sh", "bg_check.txt", "idle_check.txt", "outer.json",
                                                         "pgid.txt", "run.stderr.txt", "run.stdout.txt"]
    assert sorted(x.name for x in evidence.iterdir()) == ["head.npz", "progress.jsonl", "record.json"]
    outer = json.loads((run_dir / "outer.json").read_text(encoding="utf-8"))
    rec = json.loads((evidence / "record.json").read_text(encoding="utf-8"))
    assert P.outer_faults(outer, rec) == [] and outer["wall_seconds"] < outer["term_s"] == 2810 and outer["grace_s"] == 10
    assert subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"{outer['head']}^"], capture_output=True, text=True,
                          check=True).stdout.strip() == outer["pre_open"]
    opened = subprocess.run(["git", "-C", str(ROOT), "diff", "--unified=0", outer["pre_open"], outer["head"]],
                            capture_output=True, text=True, check=True).stdout
    assert [ln for ln in opened.splitlines() if ln[:1] in "+-" and ln[:3] not in ("+++", "---")] == \
        [f"-{WRAPPERS[TRAIN]['gate']} = False", f"+{WRAPPERS[TRAIN]['gate']} = True"]
    assert (rec["status"], rec["run"], rec["verdict"], rec.get("stop"), rec["written_by"]) == \
        ("COMPLETED", TRAIN, "ELIGIBLE", None, "script")
    assert rec["admission"] == P.admitted(TRAIN, P._read_record, P._read_outer)
    assert rec["versions"] == P.CYCLE2_VERSIONS and rec["device"] == "Device(cpu, 0)"
    assert rec["amendment_sha256"] == P.CYCLE2_ATTEMPT2_AMENDMENT[1] and rec["base_amendment_sha256"] == P.CYCLE2_AMENDMENT[1]
    assert rec["recipe"] == json.loads(json.dumps(P.RECIPE)) and rec["setting"] == rec["setting_at_end"] == \
        {P.SETTING: P.SETTING_VALUE}
    corpus = json.loads((ROOT / P.KINDS[GEN2]["evidence"] / "record.json").read_text(encoding="utf-8"))
    teacher = {k: P.TRAINED_HEAD[k] for k in ("file", "sha256", "digests")}
    assert rec["teacher"] == corpus["teacher"] == teacher and rec["checkpoint_sha1"] == corpus["checkpoint_sha1"]
    assert rec["digests_before"] == teacher["digests"]
    assert hashlib.sha256((ROOT / teacher["file"]).read_bytes()).hexdigest() == teacher["sha256"]    # the first head is untouched
    rows = corpus["summary"]["rows"]
    held = rec["corpus"]["heldout_rows"]
    assert (rec["corpus"]["games"], rec["corpus"]["rows"], rec["corpus"]["train_rows"]) == (1000, rows, rows - held)
    assert rec["prefit_train"]["kl"] >= P.RECIPE["min_kl"] and rec["prefit_train"]["rows"] == rows - held
    assert len(rec["losses"]) == P.RECIPE["updates"] and all(math.isfinite(x) for x in rec["losses"])
    assert rec["value_check"] == {"rows": held, "bitwise_equal": True, "max_abs_diff": 0.0}
    before, after = rec["digests_before"], rec["digests_after"]
    assert before["encoder."] == after["encoder."] and before["value_head."] == after["value_head."]
    assert before["policy_head."] != after["policy_head."]
    assert rec["heldout_after"]["rows"] == rec["prefit_heldout"]["rows"] == held
    assert rec["heldout_after"]["ce"] < rec["prefit_heldout"]["ce"]
    raw = (evidence / "head.npz").read_bytes()
    assert rec["head_file"] == "head.npz" and hashlib.sha256(raw).hexdigest() == rec["head_sha256"] != teacher["sha256"]
    with np.load(evidence / "head.npz") as z:
        head = {k: z[k] for k in z.files}
    with np.load(ROOT / teacher["file"]) as z:
        first = {k: z[k] for k in z.files}
    surface = set(rec["surface"]["names"])
    assert sorted(head) == sorted(first) and surface < set(head) and sum(head[k].size for k in surface) == 8840
    assert all(np.array_equal(head[k], first[k]) for k in set(head) - surface)     # BatchNorm's running statistics: frozen
    assert any(not np.array_equal(head[k], first[k]) for k in surface)
    assert P.digest(head, "policy_head.") == after["policy_head."]
    assert subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "200e7eee", "HEAD", "--", str(run_dir),
                           str(evidence)]).returncode == 0


def test_THE_NEW_ATTEMPT_IS_THE_SAME_RUN_ON_ITS_OWN_PATHS():
    one, two = P.KINDS[GEN], P.KINDS[GEN2]
    for key in ("role", "teacher", "slots", "keep_rows"):
        assert one[key] == two[key], key
    assert two["evidence"].endswith("combination-policy-cycle2-generate2") and two["plan"] != one["plan"]
    assert two["gate"][0] is not one["gate"][0] and P.ADMISSION[GEN2] == P.ADMISSION[GEN]
    assert P.OUTER_ADMISSION[GEN2] == P.OUTER_ADMISSION[GEN]


def test_THE_NEW_ATTEMPT_GENERATES_ON_ITS_PIN(tmp_path, monkeypatch):
    calls = []
    h = _gen_harness(tmp_path, monkeypatch, calls, kind=GEN2, versions=lambda: dict(P.CYCLE2_VERSIONS))
    code, rec = _conclude(GEN2, lambda o, r, lg: P._generate(GEN2, h.seams, o, r, lg), tmp_path / "ev")
    assert code == P.EXIT_COMPLETED, rec.get("void")
    assert calls == ["install", "generate"] and rec["versions"] == P.CYCLE2_VERSIONS
    assert (rec["amendment"], rec["amendment_sha256"]) == P.CYCLE2_ATTEMPT2_AMENDMENT
    assert (rec["base_amendment"], rec["base_amendment_sha256"]) == P.CYCLE2_AMENDMENT
    assert len(list((tmp_path / "cache" / GEN2).iterdir())) == 3 and not (tmp_path / "cache" / GEN).exists()


@pytest.mark.parametrize("kind, live", [(GEN2, P.VERSIONS), (GEN2, {**P.CYCLE2_VERSIONS, "mlx": "0.0"}),
                                        (GEN, P.CYCLE2_VERSIONS)], ids=["attempt-2-on-3.14.9", "attempt-2-other-mlx",
                                                                        "first-attempt-on-3.14.10"])
def test_EACH_GENERATION_KIND_REFUSES_ANY_VERSIONS_BUT_ITS_OWN_PIN(tmp_path, monkeypatch, kind, live):
    calls = []
    h = _gen_harness(tmp_path, monkeypatch, calls, kind=kind, versions=lambda: dict(live))
    code, rec = _conclude(kind, lambda o, r, lg: P._generate(kind, h.seams, o, r, lg), tmp_path / "ev")
    assert code == P.EXIT_VOID and rec["void"]["step"] == "versions" and calls == []


def test_THE_FIT_ON_ITS_PIN_RUNS_AND_ON_ANY_OTHER_VERSIONS_IS_A_VOID(tmp_path, corpus_game):
    arrays, digests = _teacher_arrays()
    seams = _teacher_seams(corpus_game, arrays, digests)
    seams.pinned_versions, seams.versions = P.CYCLE2_VERSIONS, lambda: dict(P.CYCLE2_VERSIONS)
    _c, rec = _fit(tmp_path, seams, _recipe(), "good")
    assert rec["verdict"] == "ELIGIBLE" and rec["versions"] == P.CYCLE2_VERSIONS
    seams = _teacher_seams(corpus_game, arrays, digests)
    seams.pinned_versions = P.CYCLE2_VERSIONS                 # the live versions are still the first cycle's
    code, rec = _fit(tmp_path, seams, _recipe(), "bad")
    assert code == P.EXIT_VOID and rec["void"]["step"] == "versions"


def test_THE_NEW_SURVEY_COPY_IS_THE_FIRST_ATTEMPT_S_WITH_ONLY_THE_STATED_DIFFERENCES():
    doc1, code1 = _split(SURVEY)
    doc, code = _split(SURVEY2)
    assert doc.endswith(doc1) and AMEND2 in doc[:-len(doc1)]
    a, b = code1.splitlines(), code.splitlines()
    assert len(a) == len(b)
    assert [y.strip() for x, y in zip(a, b) if x != y] == [
        'HERE = "docs/superpowers/evidence/combination-policy-cycle2-generate2-seed-survey"',
        'PLAN = "docs/superpowers/evidence/combination-policy-cycle2-generate2-run/plan.json"',
        'VERDICT_OK = "THE COMBINATION-POLICY CYCLE-2 GENERATION ATTEMPT-2 BLOCK IS CLEAN"',
        'spec = importlib.util.spec_from_file_location(f"_combination_policy_cycle2_generate2_{key}", p)',
        'return {"kind": "combination-policy-cycle2-generate2-plan/1", "base": base, "n_slots": N_SLOTS,',
        'faults.append("the cycle-2 generation attempt-2 plan is absent or not committed")',
        'faults.append("the cycle-2 generation attempt-2 plan is not build_plan of the surveyed block")',
        "say(f\"COMBINATION POLICY PILOT -- CYCLE-2 GENERATION ATTEMPT-2 GAME-SEED {'SURVEY' if mode == 'survey' else "
        "'COLLISION PROOF, mode ' + mode.upper()}\")",
        "say(f\"PROOF VERDICT ({mode}): {'THE COMBINATION-POLICY CYCLE-2 GENERATION ATTEMPT-2 BLOCK IS CLEAN AND THE STATE IS "
        "AS EXPECTED' if ok else 'NOT PROVEN -- STOP'}\""]


def test_THE_NEW_CLOSING_COPY_IS_THE_FIRST_ATTEMPT_S_AND_SERVES_THE_KIND_and_its_controls_pass():
    doc1, code1 = _split(CLOSING)
    doc, code = _split(CLOSING2)
    assert doc.endswith(doc1) and AMEND2 in doc[:-len(doc1)]
    a, b = code1.splitlines(), code.splitlines()
    assert len(a) == len(b)
    assert {y.strip() for x, y in zip(a, b) if x != y} == {
        'HERE = "docs/superpowers/evidence/combination-policy-cycle2-generate2-seed-closing"',
        'RUN = "docs/superpowers/evidence/combination-policy-cycle2-generate2"',
        'RUN_DIR = "docs/superpowers/evidence/combination-policy-cycle2-generate2-run"',
        'lines = ["COMBINATION POLICY PILOT -- CYCLE-2 GENERATION ATTEMPT-2 SEED-USAGE DERIVATION (closing)", "=" * 96,'}
    closing = _load(CLOSING2, "_c2_closing2")
    kind = P.KINDS[GEN2]
    assert closing.N_SLOTS == kind["slots"] == 1000 and closing.INPUTS[0] == kind["plan"]
    assert closing.RUN == kind["evidence"] and closing.RUN_DIR == str(pathlib.Path(kind["plan"]).parent)
    assert not SB.scan(code) and not re.search(r"\b20[0-9]{7}\b", code)
    results = closing._controls(json.loads((ROOT / kind["plan"]).read_text(encoding="utf-8"))["base"])
    assert len(results) == 21 and all(ok for _l, ok, _d in results), [r for r in results if not r[1]]

# ─────────────────────────────── Build B: the candidate, the two evaluation kinds, their wrappers and seed copies ─────────

from tests.test_combination_policy_screen import PARENT_D, ScreenFake, screen_seams, stub_play  # noqa: E402
from tests.test_combination_policy_strength import _skewed  # noqa: E402

C2_D = dict(P.CYCLE2_HEAD["digests"])
C2_HEAD = {"file": P.CYCLE2_HEAD["file"], "sha256": P.CYCLE2_HEAD["sha256"], "digests": C2_D, "bytes": b""}
FIT, FIT_RUN = P.KINDS[TRAIN]["evidence"], "docs/superpowers/evidence/combination-policy-cycle2-train-run"
CS_EV, CS_RUN = P.KINDS[CS]["evidence"], "docs/superpowers/evidence/combination-policy-cycle2-screen-run"
FAKE_BASE, SCREEN_BASE = 777000, 700000
SURVEYS = {CS: ("combination-policy-screen-seed-survey/01_seed_survey.py", "combination-policy-cycle2-screen-seed-survey/01_seed_survey.py"),
           CST: ("combination-policy-strength-seed-survey/01_seed_survey.py", "combination-policy-cycle2-strength-seed-survey/01_seed_survey.py")}
CLOSINGS = {CS: ("combination-policy-screen-seed-closing/01_derive_usage.py", "combination-policy-cycle2-screen-seed-closing/01_derive_usage.py"),
            CST: ("combination-policy-strength-seed-closing/01_derive_usage.py", "combination-policy-cycle2-strength-seed-closing/01_derive_usage.py")}


def test_THE_CANDIDATE_IS_THE_ONE_FIT_S_HEAD_AND_ITS_REAL_RECORDS_ADMIT_THE_SCREEN_ONLY():
    """Read-only, through the real committed readers: the fit's record and outer.json admit the new screen, the binding is
    exactly the committed bytes, and strength stays refused until the new screen's CONTINUE exists."""
    assert P.CYCLE2_HEAD["sha256"] == "c7e968027708d0b500517566ca8ffa26b03c7e436fea49f864f006ca2392b6c8"
    assert P.admitted(CS, P._read_record, P._read_outer) == {FIT: "ELIGIBLE", f"{FIT_RUN}/outer.json": "COMPLETED"}
    h = P.trained_head(P._read_record, P._read_committed_bytes, CS)
    raw = (ROOT / P.CYCLE2_HEAD["file"]).read_bytes()
    assert h["bytes"] == raw and hashlib.sha256(raw).hexdigest() == h["sha256"] == P.CYCLE2_HEAD["sha256"]
    rec = json.loads((ROOT / FIT / "record.json").read_text(encoding="utf-8"))
    assert h["digests"] == rec["digests_after"] and rec["head_sha256"] == h["sha256"] and rec["run"] == TRAIN
    assert P.digest(P._npz_arrays(raw), "policy_head.") == h["digests"]["policy_head."]
    assert h["sha256"] != P.TRAINED_HEAD["sha256"] and h["digests"]["policy_head."] != P.TRAINED_HEAD["digests"]["policy_head."]
    assert {p: h["digests"][p] for p in ("encoder.", "value_head.")} == \
        {p: P.TRAINED_HEAD["digests"][p] for p in ("encoder.", "value_head.")}
    # RE-ANCHORED 2026-11-02 (the new screen, 0860246e -> a87e750e, CONTINUE): the real records now admit strength too, and bind
    # the same bytes; only its closed gate and the user's approval stand between it and a launch
    assert P.admitted(CST, P._read_record, P._read_outer) == {
        FIT: "ELIGIBLE", f"{FIT_RUN}/outer.json": "COMPLETED", CS_EV: "CONTINUE", f"{CS_RUN}/outer.json": "COMPLETED"}
    assert P.trained_head(P._read_record, P._read_committed_bytes, CST)["bytes"] == raw
    # the first cycle's kinds keep their own head
    assert P.trained_head(P._read_record, P._read_committed_bytes, "screen")["sha256"] == P.TRAINED_HEAD["sha256"]


def test_THE_TWO_KINDS_ARE_THE_RULES_ON_THEIR_OWN_PATHS_AND_THE_SECOND_CYCLE_S_PIN():
    for kind, role, pairs in ((CS, "screen", 32), (CST, "strength", 509)):
        spec = P.KINDS[kind]
        assert (spec["role"], spec["pairs"], spec["fit"]) == (role, pairs, TRAIN) and spec["candidate"] is P.CYCLE2_HEAD
        assert spec["versions"] == P.CYCLE2_VERSIONS and spec["versions"]["python"] == "3.14.10"
        assert spec["amendment"] == P.CYCLE2_ATTEMPT2_AMENDMENT and spec["base_amendment"] == P.CYCLE2_AMENDMENT
        assert spec["evidence"] != P.KINDS[role]["evidence"] and spec["plan"] != P.KINDS[role]["plan"]
    assert P.KINDS[CST]["screen_kind"] == CS and "versions" not in P.KINDS["screen"] and "versions" not in P.KINDS["strength"]
    assert P.ADMISSION[CS] == [(FIT, "verdict", "ELIGIBLE")]
    assert P.ADMISSION[CST] == [(FIT, "verdict", "ELIGIBLE"), (CS_EV, "verdict", "CONTINUE")]
    assert P.OUTER_ADMISSION[CS] == [(FIT_RUN, FIT)] and P.OUTER_ADMISSION[CST] == [(FIT_RUN, FIT), (CS_RUN, CS_EV)]
    assert (K.SCREEN["pairs"], K.STRENGTH["pairs"]) == (32, 509)
    flat = " ".join((ROOT / AMEND).read_text(encoding="utf-8").split())
    assert "plays the **original proven combination**, not the first trained head" in flat and "`0.40`" in flat


GOOD_FIT = {"run": TRAIN, "status": "COMPLETED", "verdict": "ELIGIBLE", "card_sha256": P.CARD_SHA256, "head": "a" * 40,
            "head_file": "head.npz", "head_sha256": P.CYCLE2_HEAD["sha256"], "digests_after": C2_D,
            "digests_before": dict(P.TRAINED_HEAD["digests"]), "versions": dict(P.CYCLE2_VERSIONS),
            "teacher": {k: P.TRAINED_HEAD[k] for k in ("file", "sha256", "digests")}}
GOOD_C2_SCREEN = {"run": CS, "status": "COMPLETED", "verdict": "CONTINUE", "card_sha256": P.CARD_SHA256, "head": "c" * 40,
                  "trained_head": {k: P.CYCLE2_HEAD[k] for k in ("file", "sha256", "digests")},
                  "digests": {K.CANDIDATE: C2_D, P.REFERENCE: dict(PARENT_D)}, "checkpoint_sha1": P.M.CHECKPOINT_SHA1}


def _c2_readers(fit=None, screen=None, data=None):
    recs = {FIT: GOOD_FIT if fit is None else fit, CS_EV: GOOD_C2_SCREEN if screen is None else screen}

    def read_record(rel):
        if recs[rel] is False:
            raise FileNotFoundError(rel)
        return recs[rel]

    def read_bytes(rel):
        assert rel == P.CYCLE2_HEAD["file"], rel
        return (ROOT / rel).read_bytes() if data is None else data
    return read_record, read_bytes


def test_A_GOOD_FIT_RECORD_BINDS_BOTH_KINDS():
    for kind in (CS, CST):
        assert P.trained_head(*_c2_readers(), kind)["sha256"] == P.CYCLE2_HEAD["sha256"]


@pytest.mark.parametrize("change, why", [
    ({"run": "train"}, "not an ELIGIBLE training run"), ({"verdict": "STOP"}, "not an ELIGIBLE training run"),
    ({"head_sha256": P.TRAINED_HEAD["sha256"]}, "not the pinned head"),
    ({"digests_after": dict(P.TRAINED_HEAD["digests"])}, "digests after training"),
    ({"teacher": None}, "did not start from the first head"),
    ({"digests_before": {**P.TRAINED_HEAD["digests"], "policy_head.": "another"}}, "did not start from the first head"),
    ({"versions": dict(P.VERSIONS)}, "ran on")],
    ids=["first-fit", "stop", "first-head-sha", "first-head-digests", "no-teacher", "other-start", "other-versions"])
@pytest.mark.parametrize("kind", [CS, CST])
def test_THE_BINDING_REFUSES_ANY_OTHER_FIT(kind, change, why):
    with pytest.raises(P.Refused, match=why):
        P.trained_head(*_c2_readers(fit={**GOOD_FIT, **change}), kind)


def test_THE_BINDING_REFUSES_OTHER_BYTES_AND_THE_FIRST_HEAD_S_BYTES():
    for data in (b"altered", (ROOT / P.TRAINED_HEAD["file"]).read_bytes()):
        with pytest.raises(P.Refused, match="is not the pinned bytes"):
            P.trained_head(*_c2_readers(data=data), CS)


@pytest.mark.parametrize("change", [{"trained_head": None}, {"digests": {K.CANDIDATE: dict(P.TRAINED_HEAD["digests"])}},
                                    {"checkpoint_sha1": "0" * 40},
                                    {"trained_head": {k: P.TRAINED_HEAD[k] for k in ("file", "sha256", "digests")}}],
                         ids=["absent", "first-head-digests", "checkpoint", "first-head-binding"])
def test_STRENGTH_NEEDS_THE_NEW_SCREEN_TO_HAVE_PLAYED_THIS_HEAD(change):
    with pytest.raises(P.Refused, match="did not play the pinned head"):
        P.trained_head(*_c2_readers(screen={**GOOD_C2_SCREEN, **change}), CST)
    with pytest.raises(P.Refused, match="does not exist"):
        P.trained_head(*_c2_readers(screen=False), CST)


def _c2_plans(kind):
    games = 2 * P.KINDS[kind]["pairs"]

    def read(rel):
        return {P.KINDS[kind]["plan"]: {"base": FAKE_BASE, "n_slots": games},
                P.KINDS[CS]["plan"]: {"base": FAKE_BASE if kind == CS else SCREEN_BASE, "n_slots": 64}}[rel]
    return read


def _c2_seams(kind, play, **over):
    def install(net):
        net.trained = True
        return dict(C2_D)
    base = dict(versions=lambda: dict(P.CYCLE2_VERSIONS), read_plan=_c2_plans(kind), install_head=install)
    return screen_seams(play, **{**base, **over})


def _c2_run(tmp_path, kind, seams):
    from tests.test_combination_policy_screen import _REAL_CONCLUDE
    return _REAL_CONCLUDE(kind, lambda o, r, lg: P._screen(seams, C2_HEAD, o, r, lg, kind), tmp_path / "ev", {})


@pytest.mark.parametrize("kind, outcome, verdict", [
    (CS, lambda g, cr: K.CANDIDATE, "CONTINUE"), (CS, lambda g, cr: P.REFERENCE, "STOP"),
    (CST, lambda g, cr: K.CANDIDATE, "PASS"), (CST, lambda g, cr: K.CANDIDATE if g % 4 < 2 else P.REFERENCE, "NON_PASS")],
    ids=["screen-continue", "screen-stop", "strength-pass", "strength-even"])
def test_THE_FROZEN_RULE_DECIDES_ON_THE_ROLE_S_FIELDS(tmp_path, kind, outcome, verdict):
    code, rec = _c2_run(tmp_path, kind, _c2_seams(kind, stub_play(outcome)))
    role = P.KINDS[kind]["role"]
    assert code == P.EXIT_COMPLETED and rec["verdict"] == rec[role]["verdict"] == verdict, rec.get("void")
    assert rec["versions"] == P.CYCLE2_VERSIONS and rec["trained_head"]["sha256"] == P.CYCLE2_HEAD["sha256"]
    assert (rec["amendment_sha256"], rec["base_amendment_sha256"]) == (P.CYCLE2_ATTEMPT2_AMENDMENT[1], P.CYCLE2_AMENDMENT[1])
    assert kind not in rec
    games = [json.loads(x) for x in (tmp_path / "ev" / "games.jsonl").read_text().splitlines()]
    assert len(games) == 2 * P.KINDS[kind]["pairs"]
    assert {g["pairing_id"] for g in games} == {f"combination-policy-{WRAPPERS[kind]['stem']}"}
    agents = rec["protocol"]["agents"]
    assert agents[0]["mcts_sims"] == agents[1]["mcts_sims"] == 200
    assert P.M.CHECKPOINT in agents[1].values() and f"{P.M.CHECKPOINT}+{P.CYCLE2_HEAD['file']}" in agents[0].values()
    assert set(rec["protocol"]["evaluators"].values()) == {"MirrorAveragingEvaluator"}
    assert len(set(rec["protocol"]["classes"].values())) == 1 and "RootSequentialHalving" in rec["protocol"]["classes"][K.CANDIDATE]


@pytest.mark.parametrize("kw, guard", [({"seconds": 1.2}, "search"), ({"extra_candidate_calls": 2}, "evaluations")],
                         ids=["search-seconds", "evaluations"])
def test_BOTH_COMPUTE_GUARDS_STILL_STOP_A_PASS(tmp_path, kw, guard):
    code, rec = _c2_run(tmp_path, CST, _c2_seams(CST, _skewed(lambda g, cr: K.CANDIDATE, **kw)))
    assert code == P.EXIT_COMPLETED and rec["verdict"] == "NOT_COMPUTE_MATCHED", guard
    assert rec["strength"]["compute_matched"] is False and rec["strength"]["bound_above_null"] is True
    assert rec["guards"]["evaluations_ratio"] > 0 and rec["guards"]["search_seconds_ratio"] > 0


@pytest.mark.parametrize("kind", [CS, CST])
@pytest.mark.parametrize("over, step, why", [
    ({"versions": lambda: dict(P.VERSIONS)}, "versions", "pinned"),
    ({"versions": lambda: {**P.CYCLE2_VERSIONS, "mlx": "0.0"}}, "versions", "pinned"),
    ({"digests": lambda net: dict(P.TRAINED_HEAD["digests"])}, "reference", "first trained head"),
    ({"install_head": lambda net: dict(P.TRAINED_HEAD["digests"])}, "head", "not the trained head's"),
    ({"digests": lambda net: {**PARENT_D, "value_head.": "other"}}, "reference", "not calib020_0001's parent"),
    ({"checkpoint_sha1": lambda: "0" * 40}, "checkpoint", "calib020_0001"),
    ({"seed_ok": lambda s: False}, "seeds", "not ACCOUNTED")],
    ids=["first-cycle-python", "other-mlx", "first-head-as-reference", "first-head-as-candidate", "other-parent", "checkpoint", "seeds"])
def test_EACH_PRECONDITION_IS_A_VOID_BEFORE_ANY_GAME(tmp_path, kind, over, step, why):
    calls = []
    code, rec = _c2_run(tmp_path, kind, _c2_seams(kind, stub_play(lambda g, cr: K.CANDIDATE, calls=calls), **over))
    assert code == P.EXIT_VOID and rec["void"]["step"] == step and why in rec["void"]["detail"], rec.get("void")
    assert calls == [] and rec["verdict"] is None


def test_STRENGTH_S_BLOCK_MUST_NOT_OVERLAP_THE_NEW_SCREEN_S(tmp_path):
    seams = _c2_seams(CST, stub_play(lambda g, cr: K.CANDIDATE), read_plan=lambda rel: {
        P.KINDS[CST]["plan"]: {"base": FAKE_BASE, "n_slots": 1018}, P.KINDS[CS]["plan"]: {"base": FAKE_BASE + 10, "n_slots": 64},
        P.KINDS["screen"]["plan"]: {"base": SCREEN_BASE, "n_slots": 64}}[rel])
    code, rec = _c2_run(tmp_path, CST, seams)
    assert code == P.EXIT_VOID and rec["void"]["step"] == "seeds" and "overlaps" in rec["void"]["detail"]


def test_THE_FIRST_CYCLE_S_SCREEN_STILL_PINS_ITS_OWN_PYTHON_AND_PAIRING_ID(tmp_path):
    from tests.test_combination_policy_screen import _REAL_CONCLUDE, HEAD
    code, rec = _REAL_CONCLUDE("screen", lambda o, r, lg: P._screen(
        screen_seams(stub_play(lambda g, cr: K.CANDIDATE), versions=lambda: dict(P.CYCLE2_VERSIONS)), HEAD, o, r, lg),
        tmp_path / "ev", {})
    assert code == P.EXIT_VOID and rec["void"]["step"] == "versions" and "are not the card's" in rec["void"]["detail"]


@pytest.mark.parametrize("kind", [CS, CST])
def test_THE_NEW_ENTRIES_REFUSE_ON_A_CLOSED_GATE_AN_ALTERED_AMENDMENT_OR_AN_EXISTING_OUTPUT(monkeypatch, capsys, tmp_path, kind):
    monkeypatch.setattr(P, "_mlx_loaded", lambda: False)
    assert P.main(["run", kind, "--checkpoint", "/x"]) == P.EXIT_REFUSED and "is closed" in capsys.readouterr().err
    gate, name = P.KINDS[kind]["gate"]
    monkeypatch.setattr(gate, name, True)
    monkeypatch.setitem(P.KINDS[kind], "base_amendment", (AMEND, "0" * 64))
    assert P.main(["run", kind, "--checkpoint", "/x"]) == P.EXIT_REFUSED and "pinned amendment" in capsys.readouterr().err
    monkeypatch.setitem(P.KINDS[kind], "base_amendment", P.CYCLE2_AMENDMENT)
    monkeypatch.setattr(P, "admitted", lambda k, *_a: {"stub": k})
    monkeypatch.setattr(P, "trained_head", lambda *_a: dict(C2_HEAD))
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"not a model")
    monkeypatch.setitem(P.KINDS[kind], "evidence", str(tmp_path))
    assert P.main(["run", kind, "--checkpoint", str(ckpt)]) == P.EXIT_REFUSED and "one-shot" in capsys.readouterr().err


def test_RUN_CYCLE2_SCREEN_PASSES_THE_COMMITTED_CANDIDATE_TO_THE_MATCH_SEAMS(tmp_path, monkeypatch, capsys):
    seen = {}
    gate, name = P.KINDS[CS]["gate"]
    monkeypatch.setattr(gate, name, True)
    monkeypatch.setattr(P, "_mlx_loaded", lambda: False)
    monkeypatch.setattr(P, "admitted", lambda k, *_a: {"stub": k})
    monkeypatch.setitem(P.KINDS[CS], "evidence", str(tmp_path / "ev"))
    monkeypatch.setattr(P, "conclude", lambda k, execute, out, admission, head=None: (0, {"status": "COMPLETED"}))
    monkeypatch.setattr(P, "_real_screen_seams", lambda c, h: seen.update(ckpt=c, head=h))
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"not a model")
    assert P.main(["run", CS, "--checkpoint", str(ckpt)]) == 0
    assert seen["ckpt"] == ckpt and seen["head"]["sha256"] == P.CYCLE2_HEAD["sha256"]
    assert hashlib.sha256(seen["head"]["bytes"]).hexdigest() == P.CYCLE2_HEAD["sha256"]
    capsys.readouterr()


@pytest.mark.parametrize("kind", [CS, CST])
def test_THE_EVALUATION_WRAPPER_IS_THE_FIRST_CYCLE_S_WITH_ONLY_THE_STATED_DIFFERENCES(kind):
    w = WRAPPERS[kind]
    old, new = _code(w["base"]).splitlines(), _code(_path(kind)).splitlines()
    assert len(old) == len(new)
    changed = [b for a, b in zip(old, new) if a != b]
    term = ["TERM_S=3650"] if kind == CS else ["TERM_S=50390"]             # RE-ANCHORED 2026-11-02: strength's set from the screen
    assert changed[:4] == [f"OUT_REL=docs/superpowers/evidence/combination-policy-{w['stem']}-run",
                           f"RUN_REL=docs/superpowers/evidence/combination-policy-{w['stem']}",
                           f"GATE_REL=scripts/GPU/alphazero/{w['gate_rel']}", f"GATE={w['gate']}"]
    assert changed[4:] == term + ['/usr/bin/perl -e "$SUPERVISOR" "$OUT_REL/pgid.txt" "$TERM_S" "$GRACE_S" env MLX_ENABLE_TF32=0 '
                                  f'PYTHONPATH=. "$PY" -B -m "$MODULE" run {kind} --checkpoint "$CKPT" \\']
    head_old = w["base"].read_text().split("set -u\n", 1)[0]
    head = _path(kind).read_text().split("set -u\n", 1)[0]
    assert head.startswith("#!/bin/bash\n# THE SECOND POLICY-IMPROVEMENT CYCLE: ") and AMEND in head and AMEND2 in head
    assert head.endswith(head_old[len("#!/bin/bash\n"):]) and "BUILT ONLY" in head
    assert subprocess.run(["bash", "-n", str(_path(kind))]).returncode == 0 and _path(kind).stat().st_mode == w["base"].stat().st_mode


def test_THE_SCREEN_CEILING_IS_THE_COMPLETED_SEARCH_TIMINGS_ON_THIS_CORPUS_S_MOVE_COUNTS():
    """Amendment §7, recomputed from the committed records. The candidate's whole-run seconds per search in the first strength
    comparison x the new corpus's plies per game x 64 x 1.5, rounded up to the minute."""
    cost = json.loads((ROOT / P.KINDS["strength"]["evidence"] / "record.json").read_text(encoding="utf-8"))["cost_totals"][K.CANDIDATE]
    rate = cost["search_seconds"] / cost["searches"]
    games = [json.loads(x) for x in (ROOT / P.KINDS[GEN2]["evidence"] / "games.jsonl").read_text(encoding="utf-8").splitlines()]
    plies = sum(g["n_moves"] for g in games) / len(games)
    projected = rate * plies * 64
    ceiling = int(math.ceil(1.5 * projected / 60.0) * 60)
    w = _code(_path(CS))
    term, grace = (int(re.search(rf"^{k}=(\d+)$", w, re.M).group(1)) for k in ("TERM_S", "GRACE_S"))
    assert (round(rate, 4), plies, round(projected, 1)) == (0.6002, 62.515, 2401.4)
    assert ceiling == 3660 == term + grace and grace == 10
    header = " ".join(ln.lstrip("# ") for ln in _path(CS).read_text(encoding="utf-8").split("set -u\n", 1)[0].splitlines())
    assert "0.6002 s per combination search" in header and "62.515 plies per game" in header and "3,660 s" in header
    assert "through 3670 s" in header


def test_THE_STRENGTH_CEILING_IS_THE_NEW_SCREEN_S_WALL_TIME_SCALED():
    """BUILT provisional. RE-ANCHORED 2026-11-02 at the screen (a87e750e) while the placeholder stood. RE-ANCHORED 2026-11-02 (the
    user's review): the wrapper now holds §7's derivation from the new screen's completed wall time, recomputed here, and its header
    states it: TERM 50,390 s, KILL 50,400 s, observation through 50,410 s."""
    outer = json.loads((ROOT / CS_RUN / "outer.json").read_text(encoding="utf-8"))
    rec = json.loads((ROOT / CS_EV / "record.json").read_text(encoding="utf-8"))
    assert P.outer_faults(outer, rec) == [] and rec["verdict"] == "CONTINUE"
    derived = int(math.ceil(outer["wall_seconds"] * 1018 / 64 * 1.5 / 60.0) * 60)
    w = _code(_path(CST))
    term, grace = (int(re.search(rf"^{k}=(\d+)$", w, re.M).group(1)) for k in ("TERM_S", "GRACE_S"))
    assert (term, grace) == (50390, 10) and term + grace == derived == 50400 and outer["wall_seconds"] == 2110
    header = " ".join(ln.lstrip("# ") for ln in _path(CST).read_text(encoding="utf-8").split("set -u\n", 1)[0].splitlines())
    assert "PROVISIONAL" not in header.split("THE COMBINATION POLICY PILOT")[0]
    assert "2,110 s x 1,018 / 64 x 1.5 = 50,343.3 s" in header and "50,400 s" in header and "through 50410 s" in header


@pytest.mark.parametrize("kind", [CS, CST])
def test_THE_EVALUATION_SURVEY_COPY_IS_THE_FIRST_CYCLE_S_AND_SERVES_ITS_KIND(kind):
    base, rel = SURVEYS[kind]
    name = P.KINDS[kind]["role"]
    doc1, code1 = _split(base)
    doc, code = _split(rel)
    assert doc.endswith(doc1) and AMEND in doc[:-len(doc1)]
    a, b = code1.splitlines(), code.splitlines()
    assert len(a) == len(b)
    U = name.upper()
    assert [y.strip() for x, y in zip(a, b) if x != y] == [
        f'HERE = "docs/superpowers/evidence/combination-policy-cycle2-{name}-seed-survey"',
        f'PLAN = "docs/superpowers/evidence/combination-policy-cycle2-{name}-run/plan.json"',
        f'VERDICT_OK = "THE COMBINATION-POLICY CYCLE-2 {U} BLOCK IS CLEAN"',
        f'spec = importlib.util.spec_from_file_location(f"_combination_policy_cycle2_{name}_{{key}}", p)',
        f'return {{"kind": "combination-policy-cycle2-{name}-plan/1", "base": base, "n_slots": N_SLOTS,',
        f'faults.append("the cycle-2 {name} plan is absent or not committed")',
        f'faults.append("the cycle-2 {name} plan is not build_plan of the surveyed block")',
        f"say(f\"COMBINATION POLICY PILOT -- CYCLE-2 {U} GAME-SEED {{'SURVEY' if mode == 'survey' else 'COLLISION PROOF, mode ' + mode.upper()}}\")",
        f'say("SLOTS (combination_policy_pilot run cycle2_{name}: game g plays seed BASE + g, in order, in one process):")',
        f"say(f\"PROOF VERDICT ({{mode}}): {{'THE COMBINATION-POLICY CYCLE-2 {U} BLOCK IS CLEAN AND THE STATE IS AS EXPECTED' if ok else 'NOT PROVEN -- STOP'}}\""]
    survey = _load(rel, f"_c2_{name}_survey")
    assert survey.N_SLOTS == 2 * P.KINDS[kind]["pairs"] and survey.PLAN == P.KINDS[kind]["plan"]
    assert survey.PINNED == _load(base, f"_c1_{name}_survey").PINNED
    assert not SB.scan(code) and not re.search(r"\b20[0-9]{7}\b", code)
    S, Pf, SEL, block, prior, masks = _synthetic(survey)
    rec = {"head": "0" * 40, "registry_sha256": "0" * 64, "base": block[0], "block": block}
    groups = {"state": survey._state_controls(S, Pf, rec, "a" * 64), "plants": S._plant_controls(block, prior, masks, {}, True),
              "exemptions": S._exemption_controls(block), "enumeration": Pf._enumeration_controls(S),
              "two-band": survey._two_band_controls(S, SEL, block, prior, masks)}
    assert {k: len(v) for k, v in groups.items()} == {"state": 12, "plants": 8, "exemptions": 6, "enumeration": 6, "two-band": 10}
    for k, v in groups.items():
        assert all(ok for _l, ok, _d in v), (k, [r for r in v if not r[1]])


@pytest.mark.parametrize("kind", [CS, CST])
def test_THE_EVALUATION_SURVEY_S_THREE_MODES_REHEARSED_ON_A_SYNTHETIC_REPOSITORY(tmp_path, monkeypatch, capsys, kind):
    """Nothing of the real repository is scanned or written: this is no survey of it."""
    import importlib.util
    from scripts.GPU.alphazero import e4_screen_reference as REG
    rel, name = SURVEYS[kind][1], P.KINDS[kind]["role"]
    n, repo = 2 * P.KINDS[kind]["pairs"], tmp_path / "synthetic"

    def git(*a):
        return subprocess.check_output(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo, text=True).strip()

    def mode(m):
        monkeypatch.setattr(sys, "dont_write_bytecode", True)
        spec = importlib.util.spec_from_file_location(f"_c2_{name}_{m}", repo / "docs/superpowers/evidence" / rel)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        code = mod.main(["x", "--mode", m])
        capsys.readouterr()
        return mod, code, (repo / mod.OUT[m]).read_text(encoding="utf-8")
    for r in (*PINNED_FILES, f"docs/superpowers/evidence/{rel}"):
        (repo / r).parent.mkdir(parents=True, exist_ok=True)
        (repo / r).write_bytes((ROOT / r).read_bytes())
    top = max([SB.BAND[0]] + [hi for lo, hi in REG.ACCOUNTED_SEED_INTERVALS if hi > SB.BAND[0]])
    old, new = SB.BAND_2026[0] + 61_000, top + 23_456
    reg = repo / "scripts/GPU/alphazero/e4_screen_reference.py"
    reg.write_text(f"ACCOUNTED_SEED_INTERVALS = (\n    ({old}, {old + 10}),\n)\nEXPOSED_SEED_INTERVALS = ()\n"
                   "RETIRED_SEED_INTERVALS = ()\nTEST_ONLY_SEED_INTERVALS = ()\n")
    (repo / "tests").mkdir()
    (repo / "tests/test_planted.py").write_text(f"A = {old + 500}\nB = {new:_}\n")
    (repo / f"docs/superpowers/evidence/combination-policy-cycle2-{name}-run").mkdir(parents=True)
    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "synthetic")
    mod, code, survey = mode("survey")
    assert code == 0, survey[-2500:]
    _anchor, _base, block = SB.candidate(REG.ACCOUNTED_SEED_INTERVALS, [new], n)
    assert f"block = [{block[0]}, {block[1]}) -- {n} seeds" in survey
    assert survey.endswith(f"SURVEY VERDICT   : THE COMBINATION-POLICY CYCLE-2 {name.upper()} BLOCK IS CLEAN\n")
    git("add", "-A")
    git("commit", "-qm", "survey record")
    _m, code, pre = mode("pre")
    assert code == 0 and "EXPECTED STATE (pre): HOLDS" in pre, pre[-2500:]
    git("add", "-A")
    git("commit", "-qm", "pre record")
    sha = hashlib.sha256((repo / mod.OUT["survey"]).read_bytes()).hexdigest()
    reg.write_text(reg.read_text().replace(f"    ({old}, {old + 10}),\n)", f"    ({old}, {old + 10}),\n    ({block[0]}, {block[1]}),\n)"))
    (repo / mod.PLAN).write_text(json.dumps(mod.build_plan(block[0], sha), indent=1, sort_keys=True) + "\n")
    git("add", "-A")
    git("commit", "-qm", "registration")
    _m, code, post = mode("post")
    assert code == 0 and "EXPECTED STATE (post): HOLDS" in post, post[-2500:]


@pytest.mark.parametrize("kind", [CS, CST])
def test_THE_EVALUATION_CLOSING_COPY_IS_THE_FIRST_CYCLE_S_AND_ITS_CONTROLS_PASS(kind):
    base, rel = CLOSINGS[kind]
    name = P.KINDS[kind]["role"]
    doc1, code1 = _split(base)
    doc, code = _split(rel)
    assert doc.endswith(doc1) and AMEND in doc[:-len(doc1)]
    a, b = code1.splitlines(), code.splitlines()
    assert len(a) == len(b)
    assert {y.strip() for x, y in zip(a, b) if x != y} == {
        f'HERE = "docs/superpowers/evidence/combination-policy-cycle2-{name}-seed-closing"',
        f'RUN = "docs/superpowers/evidence/combination-policy-cycle2-{name}"',
        f'RUN_DIR = "docs/superpowers/evidence/combination-policy-cycle2-{name}-run"',
        f'lines = ["COMBINATION POLICY PILOT -- CYCLE-2 {name.upper()} SEED-USAGE DERIVATION (closing)", "=" * 96,'}
    closing = _load(rel, f"_c2_{name}_closing")
    spec = P.KINDS[kind]
    assert closing.N_SLOTS == 2 * spec["pairs"] and closing.INPUTS[0] == spec["plan"] and closing.RUN == spec["evidence"]
    assert not SB.scan(code) and not re.search(r"\b20[0-9]{7}\b", code)
    results = closing._controls(FAKE_BASE)
    assert len(results) == 15 and all(ok for _l, ok, _d in results), [r for r in results if not r[1]]


def test_THE_NEW_SCREEN_RAN_ONCE_COMPLETED_CONTINUE_AND_ITS_BLOCK_IS_CLOSED():
    """BUILT 2026-11-02 as the build-only state. RE-ANCHORED 2026-11-02 at the seed preparation (survey 1b2c5a8e, pre-proof 3df7caca,
    registration 322aae07, post-proof 884389bb). RE-ANCHORED 2026-11-02 (the run, 0860246e -> a87e750e, and its closing, K1 6a360aa0
    + K2 291b6cc9): the survey is CLEAN and both proofs HOLD, each 55/55 on a clean tree and bound to the survey record; the plan is
    the copy's build_plan; the post-proof proved the registry the run launched with. The ONE run, launched from the one-line gate
    opening, COMPLETED inside its ceiling: outer exit 0, gate restored, no survivors, the pinned versions, the candidate's bytes and
    digests, the original parent as the reference (the first screen's own reference digests), its probe nonzero; all 64 games
    started in order, slot g on seed BASE + g. The verdict is the frozen rule's on the recorded pair scores. K1 RECONCILES, so the
    WHOLE block is EXPOSED and RETIRED, each exactly once and last, ACCOUNTED unchanged, every seed refused. Every mode refuses to
    run again. Both gates are closed; strength has no seed work. No block literal here: the proofs scan tests/."""
    from scripts.GPU.alphazero import e4_screen_reference as REG
    rel = SURVEYS[CS][1]
    here = (EV / rel).parent
    assert sorted(x.name for x in here.iterdir() if x.name != "__pycache__") == \
        ["01_seed_survey.py", "02_seed_survey_run.txt", "03_collision_proof_pre_run.txt", "04_collision_proof_post_run.txt"]
    raw = (here / "02_seed_survey_run.txt").read_bytes()
    survey, sha = raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()
    assert survey.splitlines()[2].endswith("; working tree clean") and "controls passed: 55/55" in survey
    assert survey.endswith("SURVEY VERDICT   : THE COMBINATION-POLICY CYCLE-2 SCREEN BLOCK IS CLEAN\n")
    for name, mode in (("03_collision_proof_pre_run.txt", "pre"), ("04_collision_proof_post_run.txt", "post")):
        text = (here / name).read_text(encoding="utf-8")
        assert text.splitlines()[2].endswith("; working tree clean")
        assert f"02_seed_survey_run.txt sha256 {sha} (read at " in text
        assert f"EXPECTED STATE ({mode}): HOLDS" in text and "controls passed: 55/55" in text
    post = (here / "04_collision_proof_post_run.txt").read_text(encoding="utf-8")
    plan = json.loads((ROOT / P.KINDS[CS]["plan"]).read_text(encoding="utf-8"))
    n = 2 * P.KINDS[CS]["pairs"]
    assert type(plan["base"]) is int and plan["n_slots"] == n == 64
    mod = _load(rel, "_c2_screen_registered")
    assert plan == mod.build_plan(plan["base"], sha)
    block = (plan["base"], plan["base"] + n)
    for mode in ("survey", "pre", "post"):
        with pytest.raises(mod.Refused, match="each mode runs once"):
            mod.main(["x", "--mode", mode])
    # the run
    run_dir, evidence = _path(CS).parent, ROOT / CS_EV
    assert sorted(x.name for x in run_dir.iterdir()) == ["01_wrapper.sh", "bg_check.txt", "idle_check.txt", "outer.json",
                                                         "pgid.txt", "plan.json", "run.stderr.txt", "run.stdout.txt"]
    assert sorted(x.name for x in evidence.iterdir()) == ["games", "games.jsonl", "progress.jsonl", "record.json"]
    assert sorted(x.name for x in (evidence / "games").iterdir()) == [f"game_{i:06d}.json" for i in range(n)]
    outer = json.loads((run_dir / "outer.json").read_text(encoding="utf-8"))
    rec = json.loads((evidence / "record.json").read_text(encoding="utf-8"))
    assert P.outer_faults(outer, rec) == [] and outer["wall_seconds"] < outer["term_s"] == 3650 and outer["grace_s"] == 10
    assert (rec["status"], rec["run"], rec["written_by"]) == ("COMPLETED", CS, "script")
    assert rec["versions"] == P.CYCLE2_VERSIONS and rec["seeds"] == {"base": block[0], "n_slots": n}
    assert (rec["amendment_sha256"], rec["base_amendment_sha256"]) == (P.CYCLE2_ATTEMPT2_AMENDMENT[1], P.CYCLE2_AMENDMENT[1])
    assert rec["admission"] == {FIT: "ELIGIBLE", f"{FIT_RUN}/outer.json": "COMPLETED"}
    assert rec["trained_head"] == {k: P.CYCLE2_HEAD[k] for k in ("file", "sha256", "digests")}
    assert rec["digests"][K.CANDIDATE] == P.CYCLE2_HEAD["digests"] and rec["checkpoint_sha1"] == P.M.CHECKPOINT_SHA1
    first = json.loads((EV / "combination-policy-screen/record.json").read_text(encoding="utf-8"))
    assert rec["digests"][P.REFERENCE] == first["digests"][P.REFERENCE]                       # the original proven combination
    assert rec["digests"][P.REFERENCE]["policy_head."] != P.TRAINED_HEAD["digests"]["policy_head."]
    assert rec["head_probe"]["max_prior_abs_diff"] > 0.0 and rec["protocol"] == {**first["protocol"], "agents": rec["protocol"]["agents"]}
    assert [a["mcts_sims"] for a in rec["protocol"]["agents"]] == [a["mcts_sims"] for a in first["protocol"]["agents"]]
    events = [json.loads(x) for x in (evidence / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["slot"] for e in events if e["event"] == "game_start"] == list(range(n))
    assert all(e["ok"] for e in events if e["event"] == "bg_check") and rec["background_checks"] == sum(
        e["event"] == "bg_check" for e in events)
    games = [json.loads(x) for x in (evidence / "games.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [g["slot"] for g in games] == list(range(n)) and {g["pairing_id"] for g in games} == {"combination-policy-cycle2-screen"}
    assert rec["screen"] == K.screen_verdict(rec["pair_scores"]) and rec["verdict"] == rec["screen"]["verdict"]
    gate_rel = "scripts/GPU/alphazero/combination_policy_cycle2_screen_authorization.py"
    assert subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"{outer['head']}^"], capture_output=True, text=True,
                          check=True).stdout.strip() == outer["pre_open"]
    opened = subprocess.run(["git", "-C", str(ROOT), "diff", "--unified=0", outer["pre_open"], outer["head"]],
                            capture_output=True, text=True, check=True).stdout
    assert [ln for ln in opened.splitlines() if ln[:1] in "+-" and ln[:3] not in ("+++", "---")] == \
        [f"-{WRAPPERS[CS]['gate']} = False", f"+{WRAPPERS[CS]['gate']} = True"] and f"b/{gate_rel}" in opened
    reg_rel = "scripts/GPU/alphazero/e4_screen_reference.py"
    launched = subprocess.run(["git", "-C", str(ROOT), "show", f"{outer['head']}:{reg_rel}"], capture_output=True, check=True).stdout
    assert f"\nregistry {reg_rel} sha256 {hashlib.sha256(launched).hexdigest()}\n" in post     # proved = launched
    assert subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "a87e750e", "HEAD", "--", str(run_dir),
                           str(evidence)]).returncode == 0
    # the closing
    closing = (EV / CLOSINGS[CS][1]).parent
    assert sorted(x.name for x in closing.iterdir() if x.name != "__pycache__") == ["01_derive_usage.py", "02_derive_usage_run.txt"]
    k1 = (closing / "02_derive_usage_run.txt").read_text(encoding="utf-8")
    assert k1.splitlines()[2].endswith("; working tree clean") and "controls passed: 15/15" in k1
    assert k1.endswith(f"VERDICT: RECONCILES -- EXPOSE {list(block)}; RETIRE {list(block)}\n")
    for intervals in (REG.ACCOUNTED_SEED_INTERVALS, REG.EXPOSED_SEED_INTERVALS, REG.RETIRED_SEED_INTERVALS):
        assert intervals.count(block) == 1 and intervals[-1] == block
        assert all(hi <= block[0] for lo, hi in intervals[:-1] if hi > SB.BAND[0])
    assert all(REG.seed_is_accounted(s) and REG.seed_is_exposed(s) and REG.seed_is_retired(s) and REG.seed_is_unavailable(s)
               and not REG.seed_is_test_only(s) for s in range(*block))
    # the gates, strength
    assert GATE_CS.COMBINATION_POLICY_CYCLE2_SCREEN_AUTHORIZED is False and P.KINDS[CS]["gate"][0] is GATE_CS
    assert GATE_CST.COMBINATION_POLICY_CYCLE2_STRENGTH_AUTHORIZED is False and P.KINDS[CST]["gate"][0] is GATE_CST
    assert sorted(x.name for x in _path(CST).parent.iterdir()) == ["01_wrapper.sh"] and not (ROOT / P.KINDS[CST]["evidence"]).exists()
    assert [x.name for x in (EV / SURVEYS[CST][1]).parent.iterdir() if x.name != "__pycache__"] == ["01_seed_survey.py"]
    assert [x.name for x in (EV / CLOSINGS[CST][1]).parent.iterdir() if x.name != "__pycache__"] == ["01_derive_usage.py"]
